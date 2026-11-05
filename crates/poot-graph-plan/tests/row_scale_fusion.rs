//! POOT-1456: row regions with a per-row scalar result and with side outputs (multi-output row regions).
//!
//! A norm feeding a per-row scale and a narrowed contraction (the row-scaled f16 guard of POOT-1454 design.md
//! section 4.1) plans one row kernel that writes the scaled activation and its scale, and the contraction applies
//! the scale in its epilogue: the dispatch count is the unscaled block's. Every row drives the production `compile`
//! (or the `fuse` pass on its own) and reads the program or graph it returns.

use std::collections::HashMap;

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::op::{BinOp, RedOp, UnOp};
use poot_graph_ir::ops::rmsnorm;
use poot_graph_ir::{
    Builder, FragmentDtype, Graph, MatmulPrecision, OpKind, Scalar, TensorType, Traced, ValueId,
};
use poot_graph_plan::{CompileLimits, CompileOptions, Program, Target, compile, fuse};
use poot_target::{AmdArch, Backend, DeviceCaps, MatrixUnits, Queried, SubgroupSupport};
use poot_tensor::{DType, HostTensor};

const M: usize = 96;
const K: usize = 64;

/// ROCm caps with the RDNA3 WMMA lowering, as the ROCm runtime reports them on gfx11: native two-byte elements.
fn rocm_wmma() -> Target {
    let mut caps = DeviceCaps::rocm_default();
    caps.matrix_units = Queried::Known(MatrixUnits::amd("gfx1151", 32));
    caps.subgroup = Queried::Known(SubgroupSupport::Present {
        min_size: 32,
        max_size: 32,
    });
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps,
    }
}

fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

/// `h = rmsnorm(x)` read by `projections` BF16 projections, summed. Under the narrow policy on a target with f16
/// fragments, `compile` puts the row scale of POOT-1454 design.md section 4.1 between them (POOT-1457): each
/// projection reads `h / s` with `s = max(rowmax(|h|) / 65504, 1)` and multiplies its result by `s`. `norm_scale`
/// scales the norm weight (a norm cancels the scale of its input), so a normed value past f16's range can be made.
fn block(projections: usize, norm_scale: f32) -> (Graph, HashMap<ValueId, Value>) {
    let b = Builder::for_tests();
    let mut inputs = HashMap::new();
    let x = b.constant("x", TensorType::f32(vec![M, K])).unwrap();
    inputs.insert(
        x.id(),
        Value::Host(HostTensor::f32(vec![M, K], values(M * K, 1, 1.0))),
    );
    let norm = b.constant("norm", TensorType::f32(vec![K])).unwrap();
    inputs.insert(
        norm.id(),
        Value::Host(HostTensor::f32(vec![K], values(K, 2, norm_scale))),
    );
    let h = rmsnorm(&b, x, norm, 1e-6).unwrap();
    let mut out: Option<Traced> = None;
    for p in 0..projections {
        let name = format!("w{p}");
        let w = b
            .constant(&name, TensorType::new(vec![K, K], DType::BF16))
            .unwrap();
        let bits = values(K * K, 3 + p as u64, 1.0)
            .into_iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        inputs.insert(w.id(), Value::Host(HostTensor::bf16(vec![K, K], bits)));
        let y = b.matmul(h, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        out = Some(match out {
            Some(sum) => b.binary(BinOp::Add, sum, y).unwrap(),
            None => y,
        });
    }
    (
        b.finish(out.expect("at least one projection")).unwrap(),
        inputs,
    )
}

fn compiled(g: &Graph, policy: MatmulPrecision) -> Program {
    compile(
        g,
        &rocm_wmma(),
        &CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(policy),
    )
    .expect("the block compiles")
}

/// The block under the narrow policy with the f16 fragments alone: each projection rounds to f16 and reads its
/// activation row-scaled (POOT-1457).
fn compiled_f16(g: &Graph) -> Program {
    let options = CompileOptions {
        fragment_only: Some(FragmentDtype::F16),
        ..CompileOptions::new(CompileLimits::STANDARD)
            .with_matmul_precision(MatmulPrecision::Narrow)
    };
    compile(g, &rocm_wmma(), &options).expect("the block compiles")
}

/// RADV through wgpu: f16 fragments only, so the narrow policy rounds every projection to f16.
fn radv() -> Target {
    Target::Gpu {
        backend: poot_target::Backend::SpirvVulkan,
        caps: DeviceCaps::wgpu_rdna3_igpu(),
    }
}

/// The number of kernel launches the program's plans hold.
fn dispatches(program: &Program) -> usize {
    program
        .planned()
        .map(|(_, plan)| plan.dispatches().len())
        .sum()
}

fn output_bits(g: &Graph, inputs: &HashMap<ValueId, Value>) -> Vec<u32> {
    eval(g, inputs, EvalOptions::default())
        .expect("the graph evaluates")
        .output
        .into_host()
        .expect("a dense graph")
        .as_f32()
        .expect("an f32 output")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// The program's row regions as `(result dtype, side output count)`.
fn row_regions(program: &Program) -> Vec<(DType, usize)> {
    program
        .planned()
        .filter_map(|(eqn, _)| match &eqn.op {
            OpKind::FusedRow(region) => Some((
                program.graph().aval(eqn.out).dtype,
                region.side_outputs.len(),
            )),
            _ => None,
        })
        .collect()
}

/// SC-1 (the card's acceptance; since POOT-1457 the scale is the planner's f16 guard): a norm -> per-row scale ->
/// narrowed contraction block plans the dispatch count of the unscaled block (the same block narrowed to bf16, the
/// ROCm default, which needs no scale), the scale adding none, with one and with three projections sharing the scaled activation. The
/// norm, the scale and the divide are one row kernel that writes the narrowed activation and the scale (its side
/// output), and each projection's epilogue multiplies by the scale. Mutation: in `fuse`, admit no shared value into
/// a row region (`shared` = `single_use`); the scale costs its own kernels again and the counts differ.
#[test]
fn a_row_scale_between_a_norm_and_its_projections_costs_no_dispatch() {
    for projections in [1, 3] {
        let (g, _) = block(projections, 1.0);
        let plain = compiled(&g, MatmulPrecision::Narrow);
        let scaled = compiled_f16(&g);
        assert_eq!(
            dispatches(&scaled),
            dispatches(&plain),
            "{projections} projections: the scale adds no dispatch"
        );
        assert_eq!(
            row_regions(&scaled),
            vec![(DType::F16, 1)],
            "{projections} projections: one row kernel stores the narrowed activation and writes the scale"
        );
        let projections_read_scale = scaled
            .planned()
            .filter(|(eqn, _)| matches!(eqn.op, OpKind::ContractionEpilogue { .. }))
            .count();
        assert_eq!(
            projections_read_scale, projections,
            "every projection is one epilogue"
        );
    }
}

/// SC-2: the fused program (on RADV, where the narrow policy rounds to f16 and so row-scales, POOT-1457) computes
/// what the declared semantics compute, bit for bit: the oracle of the compiled graph (one row region with a side output, its projection, the epilogues) equals the oracle of the traced graph
/// under the same operand precision, both for activations in f16's range (the scale is 1) and past it (the scale
/// divides). Mutation: in the oracle's `side_output_definition` define the side output as the region's result
/// (`output: region.output`); the oracle refuses the value it computes for the scale (`published [96, 64] for
/// declared f32[96,1]`).
#[test]
fn the_fused_row_scale_evaluates_bit_for_bit_as_the_declared_block() {
    for norm_scale in [1.0, 1.0e6] {
        let (g, inputs) = block(3, norm_scale);
        let program = compile(
            &g,
            &radv(),
            &CompileOptions::new(CompileLimits::STANDARD)
                .with_matmul_precision(MatmulPrecision::Narrow),
        )
        .expect("the block compiles");
        assert!(
            program
                .graph()
                .eqns()
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::SideOutput { .. })),
            "the program has a side output"
        );
        let declared =
            poot_graph_plan::with_operand_precision(&g, &radv(), MatmulPrecision::Narrow).unwrap();
        assert_eq!(
            output_bits(program.graph(), &inputs),
            output_bits(&declared, &inputs),
            "norm scale {norm_scale}"
        );
    }
}

/// SC-3: the KV-quant scale shape `scale = rowmax(|x|) * (1/127)` read by the quantizing divide `x / scale` is one
/// row region: the divided row is its result and the scale its side output (the gap POOT-1454 design.md 8.2 named:
/// a row region could not output a per-row scalar). Mutation: in `fuse`, give the region no side outputs (`.take(0)`)
/// and keep the projections; the graph check refuses the projection (`NoSuchSideOutput { index: 0, sides: 0 }`).
#[test]
fn the_kv_quant_scale_and_its_divide_are_one_row_region() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4, 32])).unwrap();
    let magnitude = b
        .binary(BinOp::Max, x, b.unary(UnOp::Neg, x).unwrap())
        .unwrap();
    let scale = b
        .binary_scalar(
            BinOp::Mul,
            b.reduce(RedOp::Max, magnitude, 1, true).unwrap(),
            Scalar::F32(1.0 / 127.0),
        )
        .unwrap();
    let quantized = b.binary(BinOp::Div, x, scale).unwrap();
    // The scale is read past the region too, as the cache stores it next to the quantized row.
    let out = b
        .binary(
            BinOp::Add,
            b.reduce(RedOp::Sum, quantized, 1, true).unwrap(),
            scale,
        )
        .unwrap();
    let g = b.finish(out).unwrap();
    let fused = fuse(&g, poot_graph_plan::RowOutputs::Multiple).unwrap();
    let ops: Vec<&str> = fused.eqns().iter().map(|eqn| eqn.op.kind_name()).collect();
    assert_eq!(ops, vec!["fused_row"], "one row region: {fused:?}");
    let OpKind::FusedRow(region) = &fused.eqns()[0].op else {
        unreachable!()
    };
    assert!(
        region.side_outputs.is_empty(),
        "everything is read inside it"
    );

    // With the divided row read by a contraction (the cache write), the scale and the row leave the region.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4, 32])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![32, 8])).unwrap();
    let magnitude = b
        .binary(BinOp::Max, x, b.unary(UnOp::Neg, x).unwrap())
        .unwrap();
    let scale = b
        .binary_scalar(
            BinOp::Mul,
            b.reduce(RedOp::Max, magnitude, 1, true).unwrap(),
            Scalar::F32(1.0 / 127.0),
        )
        .unwrap();
    let quantized = b.binary(BinOp::Div, x, scale).unwrap();
    let out = b
        .binary(BinOp::Mul, b.matmul(quantized, w).unwrap(), scale)
        .unwrap();
    let g = b.finish(out).unwrap();
    let fused = fuse(&g, poot_graph_plan::RowOutputs::Multiple).unwrap();
    let ops: Vec<&str> = fused.eqns().iter().map(|eqn| eqn.op.kind_name()).collect();
    assert_eq!(
        ops,
        vec!["fused_row", "side_output", "matmul", "binary"],
        "the scale is the row region's side output: {fused:?}"
    );
}

/// SC-4: a value read outside the region before the region's last member cannot be a side output: the region's
/// equation sits at its last member, after that reader. The grouping falls back to single-use edges (the norm
/// alone, the residual add apart). The same block with the reader after the norm fuses the add in, writing the
/// residual as a side output. Mutation: make `RowGroup::is_legal` accept every group; `fuse` commits a reader
/// before its value's definition and the graph check refuses it.
#[test]
fn a_side_output_read_before_the_region_ends_falls_back_to_single_use_fusion() {
    let build = |reader_first: bool| {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![4, 16])).unwrap();
        let y = b.constant("y", TensorType::f32(vec![4, 16])).unwrap();
        let norm = b.constant("norm", TensorType::f32(vec![16])).unwrap();
        let residual = b.binary(BinOp::Add, x, y).unwrap();
        let early = reader_first.then(|| b.transpose(residual, vec![1, 0]).unwrap());
        let h = rmsnorm(&b, residual, norm, 1e-6).unwrap();
        let moved = early.unwrap_or_else(|| b.transpose(residual, vec![1, 0]).unwrap());
        // A concat moves data, so no region takes it in: the norm's last member stays the region's last.
        let out = b
            .concat(0, &[b.transpose(moved, vec![1, 0]).unwrap(), h])
            .unwrap();
        b.finish(out).unwrap()
    };
    let side_outputs = |g: &Graph| {
        fuse(g, poot_graph_plan::RowOutputs::Multiple)
            .unwrap()
            .eqns()
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::SideOutput { .. }))
            .count()
    };
    assert_eq!(
        side_outputs(&build(false)),
        1,
        "the residual is a side output"
    );
    assert_eq!(
        side_outputs(&build(true)),
        0,
        "read too early: no side output"
    );
}

/// SC-5: only a region's result may be a pinned value (a graph output or carried state): a group whose other
/// member is pinned falls back to single-use fusion. Mutation: drop the pinned clause of `RowGroup::is_legal`; the
/// pinned sibling becomes a side output with no reader and the graph check finds the result undefined.
#[test]
fn a_pinned_member_is_never_a_side_output() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4, 16])).unwrap();
    let y = b.constant("y", TensorType::f32(vec![4, 16])).unwrap();
    let norm = b.constant("norm", TensorType::f32(vec![16])).unwrap();
    // The residual feeds the norm and a doubled copy the graph publishes as a witness: the copy would join the
    // norm's region through the shared residual, ahead of the norm's last member.
    let residual = b.binary(BinOp::Add, x, y).unwrap();
    let doubled = b
        .binary_scalar(BinOp::Mul, residual, Scalar::F32(2.0))
        .unwrap();
    let h = rmsnorm(&b, residual, norm, 1e-6).unwrap();
    let g = b
        .finish(poot_graph_ir::Outputs::new(h).with_witness(
            poot_graph_ir::ValidationId(1),
            "doubled",
            doubled,
        ))
        .unwrap();
    let fused = fuse(&g, poot_graph_plan::RowOutputs::Multiple).unwrap();
    assert!(
        !fused
            .eqns()
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::SideOutput { .. })),
        "{fused:?}"
    );
}

/// SC-8: a rewrite the planner refuses is reverted with its projections, as one rewrite. A BF16 residual and the
/// norm that reads it fuse into one row region writing the residual as a side output; the planner refuses a row
/// region whose result is a two-byte float it does not narrow, so `compile` restores the unfused chain, region and
/// projection together, and the program compiles. Mutation: in `revert_unplannable_rewrites` restore a region
/// without its projections (drop the `grouped` extension of `pending`); the kept projection reads a restored
/// equation and the compile fails the graph check.
#[test]
fn a_refused_row_region_is_reverted_with_its_side_outputs() {
    let b = Builder::for_tests();
    let bf16 = |shape: Vec<usize>| TensorType::new(shape, DType::BF16);
    let x = b.constant("x", bf16(vec![4, 16])).unwrap();
    let y = b.constant("y", bf16(vec![4, 16])).unwrap();
    let norm = b.constant("norm", bf16(vec![16])).unwrap();
    let residual = b.binary(BinOp::Add, x, y).unwrap();
    let h = rmsnorm(&b, residual, norm, 1e-6).unwrap();
    let moved = b.transpose(residual, vec![1, 0]).unwrap();
    let out = b
        .concat(0, &[b.transpose(moved, vec![1, 0]).unwrap(), h])
        .unwrap();
    let g = b.finish(out).unwrap();
    let fused = fuse(&g, poot_graph_plan::RowOutputs::Multiple).unwrap();
    assert!(
        fused
            .eqns()
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::SideOutput { .. })),
        "the fused graph writes the residual as a side output"
    );
    let program = compiled(&g, MatmulPrecision::Exact);
    assert!(
        program
            .graph()
            .eqns()
            .iter()
            .all(|eqn| !matches!(eqn.op, OpKind::SideOutput { .. } | OpKind::FusedRow(_))),
        "the refused region and its projection are reverted together"
    );
}
