//! POOT-1456: a norm, its per-row scale and the narrowed projections that read the scaled activation, as one row
//! kernel with a side output, against the same block planned without it.
//!
//! The block is the row-scaled guard of POOT-1454 design.md section 4.1: `h = rmsnorm(x)`,
//! `s = max(rowmax(|h|) / limit, 1)`, `y_i = ((h / s) W_i^T) * s`, summed. `compile` fuses the norm, the absmax,
//! the scale and the divide into one row region that writes `h / s` and `s` (its side output), and each
//! projection's epilogue multiplies by `s`. The reference reads `h` for the scale through two transposes, which no
//! region takes in: the same values, planned as a norm kernel, a scale kernel and a divide. The scale limit sits far
//! below the activation, so every row's scale divides (an f16 limit of 65504 would leave `s = 1` on this data and
//! the divide and the multiply identities).

use poot_graph_ir::op::{BinOp, RedOp, UnOp};
use poot_graph_ir::{Builder, MatmulPrecision, OpKind, Scalar, TensorType, Traced};
use poot_tensor::{DType, HostTensor};

use crate::Fixture;

const K: usize = 64;
const PROJECTIONS: usize = 3;
/// The scale's limit: far below the normed activation, so `s > 1` on every row.
const LIMIT: f32 = 0.125;

/// How the block's scale reads the normed activation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScaleRead {
    /// As the region computes it: the norm, the scale and the divide fuse into one kernel.
    Direct,
    /// Through two transposes, which no region takes in: the scale and the divide are kernels of their own.
    ThroughTransposes,
}

fn block(name: &'static str, m: usize, read: ScaleRead) -> Fixture {
    let b = Builder::for_tests();
    let x = b
        .slot_named(
            poot_graph_ir::Slot::Activation,
            "x",
            TensorType::f32(vec![m, K]),
        )
        .unwrap();
    let norm = b.constant("norm", TensorType::f32(vec![K])).unwrap();
    let h = poot_graph_ir::ops::rmsnorm(&b, x, norm, 1e-6).unwrap();
    let scaled_from = match read {
        ScaleRead::Direct => h,
        ScaleRead::ThroughTransposes => b
            .transpose(b.transpose(h, vec![1, 0]).unwrap(), vec![1, 0])
            .unwrap(),
    };
    let magnitude = b
        .binary(
            BinOp::Max,
            scaled_from,
            b.unary(UnOp::Neg, scaled_from).unwrap(),
        )
        .unwrap();
    let rowmax = b.reduce(RedOp::Max, magnitude, 1, true).unwrap();
    let scale = b
        .binary_scalar(
            BinOp::Max,
            b.binary_scalar(BinOp::Div, rowmax, Scalar::F32(LIMIT))
                .unwrap(),
            Scalar::F32(1.0),
        )
        .unwrap();
    let activation = b.binary(BinOp::Div, h, scale).unwrap();
    let mut out: Option<Traced> = None;
    for p in 0..PROJECTIONS {
        let w = b
            .constant(&format!("w{p}"), TensorType::new(vec![K, K], DType::BF16))
            .unwrap();
        let y = b
            .binary(
                BinOp::Mul,
                b.matmul(activation, b.transpose(w, vec![1, 0]).unwrap())
                    .unwrap(),
                scale,
            )
            .unwrap();
        out = Some(match out {
            Some(sum) => b.binary(BinOp::Add, sum, y).unwrap(),
            None => y,
        });
    }
    let graph = b.finish(out.expect("projections")).unwrap();
    let key = graph.meta(x.id()).slot_key().unwrap().clone();
    let steps = (0..2u64)
        .map(|step| {
            vec![crate::StepFixture {
                key: key.clone(),
                tensor: HostTensor::f32(vec![m, K], crate::fill(m * K, 0x1456 + step)),
            }]
        })
        .collect();
    Fixture {
        name,
        store: crate::store_with_bf16(&graph, &["w0", "w1", "w2"]),
        graph,
        steps,
    }
}

/// The block fused and its reference, at a few rows (`M = 5`, the register tiles) and at a prefill whose tiles fill
/// a matrix unit (`M = 160`).
pub fn row_scale_pairs() -> Vec<(Fixture, Fixture)> {
    vec![
        (
            block("row_scale_m5", 5, ScaleRead::Direct),
            block("row_scale_m5_reference", 5, ScaleRead::ThroughTransposes),
        ),
        (
            block("row_scale_m160", 160, ScaleRead::Direct),
            block(
                "row_scale_m160_reference",
                160,
                ScaleRead::ThroughTransposes,
            ),
        ),
    ]
}

/// The kernel launches and the side outputs of `fixture`'s program, compiled for the device `exec` drives.
fn launches_and_side_outputs(
    exec: &dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
    fixture: &Fixture,
    policy: MatmulPrecision,
) -> (usize, usize) {
    let staged = crate::staged_under(exec, fixture, target, policy);
    staged
        .stages()
        .fold((0, 0), |(launches, sides), (_, _, program)| {
            program
                .planned()
                .fold((launches, sides), |(l, s), (eqn, plan)| {
                    (
                        l + plan.dispatches().len(),
                        s + usize::from(matches!(eqn.op, OpKind::SideOutput { .. })),
                    )
                })
        })
}

/// Run each pair on `exec` under `policy`: the fused block writes its scale as a side output and launches fewer
/// kernels than its reference, which writes none of its own, and every step's output is the reference's, bit for bit. The
/// row kernel's reductions are the norm's own and an exact maximum, so fusing the scale and the divide into the
/// norm changes no value.
pub fn run_fused_against_reference(
    exec: &mut dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
    pairs: &[(Fixture, Fixture)],
    policy: MatmulPrecision,
) -> Result<(), String> {
    for (fused, reference) in pairs {
        let (fused_launches, fused_sides) = launches_and_side_outputs(exec, target, fused, policy);
        let (reference_launches, reference_sides) =
            launches_and_side_outputs(exec, target, reference, policy);
        // Under an f16 narrow policy the planner's range guard (POOT-1457) scales both blocks' activations again, in
        // a row region of its own with its scale as a side output; the fused block still writes one more.
        if fused_sides <= reference_sides {
            return Err(format!(
                "{}: {fused_sides} side outputs fused, {reference_sides} in the reference",
                fused.name
            ));
        }
        if fused_launches >= reference_launches {
            return Err(format!(
                "{}: {fused_launches} launches fused, {reference_launches} in the reference",
                fused.name
            ));
        }
        let bits = |outputs: Vec<HostTensor>| -> Result<Vec<Vec<u32>>, String> {
            outputs
                .iter()
                .map(|tensor| {
                    tensor
                        .to_f32()
                        .map(|values| values.iter().map(|v| v.to_bits()).collect())
                        .map_err(|e| format!("an output does not read as f32: {e}"))
                })
                .collect()
        };
        let got = bits(crate::run_outputs_under(exec, target, fused, policy)?)?;
        let want = bits(crate::run_outputs_under(exec, target, reference, policy)?)?;
        for (step, (got, want)) in got.iter().zip(&want).enumerate() {
            if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
                return Err(format!(
                    "{}: step {step}, element {i}: {} fused, {} in the reference",
                    fused.name,
                    f32::from_bits(got[i]),
                    f32::from_bits(want[i])
                ));
            }
        }
    }
    Ok(())
}
