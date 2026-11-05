//! The f16 range under the default `Narrow` (POOT-1474, `receipts/1471-narrow-without-range-guards/design.md`
//! section 7): a contraction the narrow policy rounds to f16 reads its activation unscaled, so an activation past
//! f16's range becomes infinity and the output is not finite.
//!
//! Every row drives the production `compile` on RADV caps (f16 fragments only, so the narrow policy rounds to f16)
//! and on gfx1151 caps cut down to their f16 fragments (`fragment_only`; f16 is also ROCm's default since
//! POOT-1458), and
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
        caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
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

/// A transformer block over `[K, N]` f32 weights: `h = rmsnorm(x)`, `out = x + h Wq + (silu(h Wg) * h Wu) Wd`. The
/// norm-fed projections read a row region's value; the down projection reads an epilogue's. `norm_scale` scales the
/// norm weight, which sets the size of `h` (a norm cancels the scale of its input), and so of every activation.
fn block(norm_scale: f32) -> (Graph, HashMap<ValueId, Value>) {
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
    let matmul = |a: Traced, w: Traced| b.matmul(a, w).unwrap();
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

/// The default's failure mode: a block whose activations reach about 1e6 (past f16's 65504 at the norm output, and
/// about 1e11 at the down projection's input) compiles under the narrow policy with every contraction in f16 and
/// no scale step, and the compiled program's output is not finite (infinity reaches it, not a clamped value); the
/// exact program over the same data is finite. Serving refuses the non-finite logits with a typed error.
/// Mutation: make the narrowing saturate in the oracle (`poot-eval` f16 rounding clamps to 65504); the output turns
/// finite and this row fails.
#[test]
fn activations_past_the_f16_range_give_a_non_finite_output() {
    let (g, inputs) = block(1.0e6);
    let exact = output(
        compiled(&g, &("gfx1151", rocm_wmma(), None), MatmulPrecision::Exact).graph(),
        &inputs,
    );
    let scale = exact.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        scale > 1.0e9 && exact.iter().all(|v| v.is_finite()),
        "the block's values exceed f16's range by far and are finite in f32: {scale}"
    );
    let eqns_exact = compiled(&g, &("gfx1151", rocm_wmma(), None), MatmulPrecision::Exact)
        .graph()
        .eqns()
        .len();
    for target in f16_targets() {
        let name = target.0;
        let program = compiled(&g, &target, MatmulPrecision::Narrow);
        assert_eq!(
            contraction_precisions(&program),
            vec![F16; 4],
            "every contraction rounds to f16"
        );
        assert_eq!(
            program.graph().eqns().len(),
            eqns_exact,
            "{name}: no scale step is emitted"
        );
        let narrow = output(program.graph(), &inputs);
        assert!(
            narrow.iter().any(|v| !v.is_finite()),
            "{name}: the overflowed activations reached the output as finite values"
        );
    }
}

/// The rule has no operand guard: a matmul of two runtime values (outside attention) narrows to f16 like the
/// projection beside it, which reads a weight. Mutation: in `fragment_dtype` offer a matmul whose weight operand is
/// computed at run time no candidate; the runtime product stays exact and this row fails.
#[test]
fn a_product_of_two_runtime_values_narrows_to_f16() {
    let b = Builder::for_tests();
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
        assert_eq!(contraction_precisions(&program), vec![F16, F16], "{name}");
    }
}

/// The oracle's graph (`with_operand_precision` over the traced graph) and the compiled program narrow the same
/// contractions to the same dtypes, for a dense projection and for a packed (quantized) one. The traced graph reads
/// the weight through ops `compile` folds into the contraction before it selects the precision (a transpose, a
/// decode of the packed bytes), so the decision must hold across them. Mutation: in `fragment_dtype` give a
/// `ContractionHead::Packed` no candidate; the compiled packed contraction stays exact while the oracle's graph,
/// which still holds a plain matmul over the decoded weight, narrows it, and the two disagree.
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
