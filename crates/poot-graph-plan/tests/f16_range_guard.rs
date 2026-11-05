//! POOT-1457: the f16 range guard. A contraction the narrow policy rounds to f16 reads its activation row-scaled,
//! `x / s` with `s = max(rowmax(|x|) / 65504, 1)`, and multiplies its result by `s`
//! (`receipts/1457-f16-range-guard/design.md`).
//!
//! Every row drives the production `compile` on RADV caps (f16 fragments only, so the narrow policy rounds to f16)
//! and on gfx1151 caps cut down to their f16 fragments (`fragment_only`: ROCm narrows to bf16 by default), and
//! evaluates the compiled graph with the CPU oracle, which computes the program's declared rounding: the oracle of a
//! narrowed op rounds its operands as the device does.

use std::collections::HashMap;

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::op::BinOp;
use poot_graph_ir::ops::{rmsnorm, silu};
use poot_graph_ir::{
    Builder, FragmentDtype, Graph, MatmulPrecision, OpKind, OperandPrecision, TensorType, Traced,
    ValueId,
};
use poot_graph_plan::{CompileLimits, CompileOptions, Program, Target, compile};
use poot_target::{AmdArch, Backend, DeviceCaps, MatrixUnits, Queried, SubgroupSupport};
use poot_tensor::HostTensor;

const M: usize = 48;
const K: usize = 64;
const F: usize = 96;

const F16: OperandPrecision = OperandPrecision::Narrow(FragmentDtype::F16);

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

fn radv() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: DeviceCaps::wgpu_rdna3_igpu(),
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

/// How the block's projections read their activations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Projections {
    /// Plain matmuls: the narrow policy chooses their precision, and the f16 guard scales them.
    Policy,
    /// Matmuls that declare `Narrow(f16)` themselves: the pass leaves a declared precision as it stands, so they
    /// round to f16 unscaled. This is the f16 program without the guard.
    DeclaredF16,
}

/// A transformer block over `[K, N]` f32 weights: `h = rmsnorm(x)`, `out = x + h Wq + (silu(h Wg) * h Wu) Wd`. The
/// norm-fed projections read a row region's value; the down projection reads an epilogue's. `norm_scale` scales the
/// norm weight, which sets the size of `h` (a norm cancels the scale of its input), and so of every activation.
fn block(projections: Projections, norm_scale: f32) -> (Graph, HashMap<ValueId, Value>) {
    let b = Builder::for_tests();
    let mut inputs = HashMap::new();
    let mut constant = |name: &str, shape: Vec<usize>, seed: u64, scale: f32| {
        let n = shape.iter().product();
        let t = b.constant(name, TensorType::f32(shape.clone())).unwrap();
        inputs.insert(
            t.id(),
            Value::Host(HostTensor::f32(shape, values(n, seed, scale))),
        );
        t
    };
    let x = constant("x", vec![M, K], 1, 1.0);
    let norm = constant("norm", vec![K], 2, norm_scale);
    let wq = constant("wq", vec![K, K], 3, 0.25);
    let wg = constant("wg", vec![K, F], 4, 0.25);
    let wu = constant("wu", vec![K, F], 5, 0.25);
    let wd = constant("wd", vec![F, K], 6, 0.25);
    let matmul = |a: Traced, w: Traced| match projections {
        Projections::Policy => b.matmul(a, w).unwrap(),
        Projections::DeclaredF16 => b.matmul_as(a, w, F16).unwrap(),
    };
    let h = rmsnorm(&b, x, norm, 1e-6).unwrap();
    let q = matmul(h, wq);
    let gate = silu(&b, matmul(h, wg)).unwrap();
    let act = b.binary(BinOp::Mul, gate, matmul(h, wu)).unwrap();
    let down = matmul(act, wd);
    let out = b
        .binary(BinOp::Add, b.binary(BinOp::Add, x, q).unwrap(), down)
        .unwrap();
    (b.finish(out).unwrap(), inputs)
}

/// A target with f16 fragments: its name, the target, and the one fragment dtype the pass may narrow to on it.
type F16Target = (&'static str, Target, Option<FragmentDtype>);

/// RADV, and gfx1151 cut down to its f16 fragments.
fn f16_targets() -> [F16Target; 2] {
    [
        ("radv", radv(), None),
        ("gfx1151 f16", rocm_wmma(), Some(FragmentDtype::F16)),
    ]
}

fn compiled(g: &Graph, (_, target, only): &F16Target, policy: MatmulPrecision) -> Program {
    let options = CompileOptions {
        fragment_only: *only,
        ..CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(policy)
    };
    compile(g, target, &options).expect("the block compiles")
}

fn output(g: &Graph, inputs: &HashMap<ValueId, Value>) -> Vec<f32> {
    eval(g, inputs, EvalOptions::default())
        .expect("the graph evaluates")
        .output
        .into_host()
        .expect("a dense graph")
        .as_f32()
        .expect("an f32 output")
        .to_vec()
}

/// The precision of every contraction the program plans.
fn contraction_precisions(program: &Program) -> Vec<OperandPrecision> {
    program
        .planned()
        .filter(|(eqn, _)| {
            matches!(
                eqn.op,
                OpKind::MatMul { .. }
                    | OpKind::DenseContraction { .. }
                    | OpKind::ContractionEpilogue { .. }
            )
        })
        .filter_map(|(eqn, _)| eqn.op.operand_precision())
        .collect()
}

/// SC-2, the guard: a block whose activations reach about 1e6 (past f16's 65504 at the norm output, and about 1e11
/// at the down projection's input) compiles under the narrow policy with every contraction in f16, and the compiled
/// program's output is finite and within f16's rounding of the exact program's. The norm-fed activations are scaled
/// in the norm's row region; the down projection's input, an epilogue's value, in a row region of its own. Mutation:
/// in `select_operand_precision` push the narrowed contraction without `emit_row_scaled`; the activations round to
/// infinity and the output is not finite.
#[test]
fn activations_past_the_f16_range_give_a_finite_correct_output() {
    let (g, inputs) = block(Projections::Policy, 1.0e6);
    let exact = output(
        compiled(&g, &("gfx1151", rocm_wmma(), None), MatmulPrecision::Exact).graph(),
        &inputs,
    );
    let scale = exact.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        scale > 1.0e9,
        "the block's values exceed f16's range by far: {scale}"
    );
    for target in f16_targets() {
        let name = target.0;
        let program = compiled(&g, &target, MatmulPrecision::Narrow);
        assert_eq!(
            contraction_precisions(&program),
            vec![F16; 4],
            "every contraction rounds to f16"
        );
        let narrow = output(program.graph(), &inputs);
        let worst = narrow
            .iter()
            .zip(&exact)
            .map(|(n, e)| (n - e).abs())
            .fold(0.0f32, |m, d| if d.is_nan() { f32::NAN } else { m.max(d) });
        assert!(
            narrow.iter().all(|v| v.is_finite()),
            "{name}: the narrow output is not finite: worst error {worst}"
        );
        // Each f16 rounding is within 2^-11 relative; two contractions in sequence stay well inside 1e-2.
        assert!(
            worst <= 1.0e-2 * scale,
            "{name}: worst error {worst} against an output of size {scale}"
        );
    }
}

/// SC-4, in-range identity: where every row is inside f16's range, each scale is exactly 1, so the guarded program
/// computes bit for bit what the f16 program without the guard computes (the same block with each matmul declaring
/// `Narrow(f16)` itself). Mutation: in `emit_row_scaled` drop the `max(.., 1)` (`s = rowmax / 65504`); each row is
/// scaled up to the range's edge, rounds elsewhere, and the bits differ.
#[test]
fn rows_in_range_compute_the_unguarded_f16_program_bit_for_bit() {
    for target in f16_targets() {
        let name = target.0;
        let (guarded, inputs) = block(Projections::Policy, 1.0);
        let guarded = compiled(&guarded, &target, MatmulPrecision::Narrow);
        let (unguarded, unguarded_inputs) = block(Projections::DeclaredF16, 1.0);
        let unguarded = compiled(&unguarded, &target, MatmulPrecision::Narrow);
        assert_eq!(contraction_precisions(&guarded), vec![F16; 4]);
        assert_eq!(contraction_precisions(&unguarded), vec![F16; 4]);
        assert!(
            guarded.graph().eqns().len() > unguarded.graph().eqns().len(),
            "{name}: the guarded program holds the scale steps"
        );
        let bits = |values: Vec<f32>| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(output(guarded.graph(), &inputs)),
            bits(output(unguarded.graph(), &unguarded_inputs)),
            "{name}"
        );
    }
}

/// No contraction narrows to f16 unguarded: a matmul of two runtime values (outside attention) has no weight side
/// to leave unscaled, so the guard does not take it, and with f16 fragments alone it stays exact, while the
/// projection beside it, over a weight, narrows to f16. Mutation: in `f16_guard` treat every operand as a weight
/// (`weights.holds` returns true); the runtime product narrows to f16 and the row goes red.
#[test]
fn a_product_of_two_runtime_values_never_narrows_to_f16() {
    let b = Builder::for_tests();
    // A runtime input: a value computed from constants alone is fixed before the run, so it would count as a weight.
    let x = b
        .slot_named(
            poot_graph_ir::Slot::Activation,
            "x",
            TensorType::f32(vec![M, K]),
        )
        .unwrap();
    let norm = b.constant("norm", TensorType::f32(vec![K])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![K, K])).unwrap();
    let h = rmsnorm(&b, x, norm, 1e-6).unwrap();
    let projected = b.matmul(h, w).unwrap();
    // `h [M, K] x projected^T [K, M]`: both operands are computed at run time.
    let runtime = b
        .matmul(projected, b.transpose(h, vec![1, 0]).unwrap())
        .unwrap();
    let g = b.finish(runtime).unwrap();
    for target in f16_targets() {
        let name = target.0;
        let program = compiled(&g, &target, MatmulPrecision::Narrow);
        assert_eq!(
            contraction_precisions(&program),
            vec![F16, OperandPrecision::Exact],
            "{name}"
        );
    }
}

/// The oracle's graph (`with_operand_precision` over the traced graph) and the compiled program narrow the same
/// contractions to the same dtypes, for a dense projection and for a packed (quantized) one. The traced graph reads
/// the weight through ops `compile` folds into the contraction before it selects the precision (a transpose, a
/// decode of the packed bytes), so the guard must decide by what the operand is computed from, not by the ops in
/// between. Mutation: in `Weights::of` mark only bound bytes, not values computed from them; the oracle's packed
/// matmul reads a decoded value, takes no guard and stays exact on RADV while the program narrows it to f16.
#[test]
fn the_oracle_graph_and_the_program_narrow_the_same_contractions() {
    use poot_quant::PackedWeight;
    use poot_quant::format::WeightFormat;
    let projection = |packed: bool| {
        let b = Builder::for_tests();
        let x = b
            .slot_named(
                poot_graph_ir::Slot::Activation,
                "x",
                TensorType::f32(vec![4, 256]),
            )
            .unwrap();
        let y = if packed {
            let descriptor = PackedWeight::try_new(WeightFormat::Q4_1, [64, 256]).unwrap();
            poot_graph_ir::test_support::packed_graphs::packed_linear(
                &b, x, "layer", descriptor, None, None,
            )
            .unwrap()
        } else {
            let w = b
                .constant(
                    "w",
                    TensorType::new(vec![64, 256], poot_tensor::DType::BF16),
                )
                .unwrap();
            b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap()
        };
        b.finish(y).unwrap()
    };
    let narrowed = |g: &Graph| -> Vec<OperandPrecision> {
        g.eqns()
            .iter()
            .filter_map(|eqn| eqn.op.operand_precision())
            .filter(|precision| !precision.is_exact())
            .collect()
    };
    for packed in [false, true] {
        let g = projection(packed);
        for target in f16_targets() {
            let (name, device, only) = &target;
            let program = compiled(&g, &target, MatmulPrecision::Narrow);
            // `with_operand_precision` sees every unit; on gfx1151 cut down to f16 the program's choice is RADV's.
            let oracle_target = if only.is_some() { radv() } else { *device };
            let declared = poot_graph_plan::with_operand_precision(
                &g,
                &oracle_target,
                MatmulPrecision::Narrow,
            )
            .unwrap();
            assert_eq!(
                narrowed(&declared),
                vec![F16],
                "{name}, packed {packed}: the oracle's graph"
            );
            assert_eq!(
                narrowed(program.graph()),
                vec![F16],
                "{name}, packed {packed}: the program"
            );
        }
    }
}
