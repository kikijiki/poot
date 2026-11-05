//! POOT-1457: the f16 range guard on a device. Under the narrow policy a contraction that rounds its activation to
//! f16 reads it row-scaled, `x / s` with `s = max(rowmax(|x|) / 65504, 1)`, and multiplies its result by `s`
//! (`receipts/1457-f16-range-guard/design.md`).
//!
//! The block is a transformer block over `[K, N]` f32 weights: `h = rmsnorm(x)`, `out = x + h Wq + (silu(h Wg) * h
//! Wu) Wd`. The norm-fed projections take their scale from the norm's row region; the down projection's input is
//! an epilogue's value and takes it from a row region of its own.
//!
//! The signs are fixed so that no sum cancels: `x < 0` (so the absolute value's negation is what the scale reads),
//! `Wq, Wd >= 0` and `Wg, Wu <= 0`, so the gate and up products are positive and the output is the large positive
//! down projection plus a smaller negative rest. Past f16's range the terms reach 1e11; where a sum of such terms
//! cancels to a small element, one f16 rounding that a different f32 accumulation order sends the other way moves
//! that element by more than the per-element tolerance, a property of the data and not of the device.

use poot_graph_ir::op::BinOp;
use poot_graph_ir::{
    Builder, FragmentDtype, MatmulPrecision, OpKind, OperandPrecision, TensorType, Traced,
};
use poot_tensor::HostTensor;

use crate::Fixture;

const K: usize = 64;
const F: usize = 96;

/// The norm weight's scale that puts the normed activation past f16's range: `|h|` reaches about `1e6 x 0.25`.
const PAST_RANGE: f32 = 1.0e6;

/// How the block's projections read their activations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Projections {
    /// Plain matmuls: the policy chooses their precision, and the guard scales the f16 ones.
    Policy,
    /// Matmuls that declare `Narrow(f16)` themselves: a declared precision stands, so they round unscaled. This is
    /// the f16 program without the guard.
    DeclaredF16,
}

fn block(name: &'static str, m: usize, projections: Projections, norm_scale: f32) -> Fixture {
    let b = Builder::for_tests();
    let x = b
        .slot_named(
            poot_graph_ir::Slot::Activation,
            "x",
            TensorType::f32(vec![m, K]),
        )
        .unwrap();
    let norm = b.constant("norm", TensorType::f32(vec![K])).unwrap();
    let weight = |name: &str, rows: usize, cols: usize| {
        b.constant(name, TensorType::f32(vec![rows, cols])).unwrap()
    };
    let (wq, wg, wu, wd) = (
        weight("wq", K, K),
        weight("wg", K, F),
        weight("wu", K, F),
        weight("wd", F, K),
    );
    let matmul = |a: Traced, w: Traced| match projections {
        Projections::Policy => b.matmul(a, w).unwrap(),
        Projections::DeclaredF16 => b
            .matmul_as(a, w, OperandPrecision::Narrow(FragmentDtype::F16))
            .unwrap(),
    };
    let h = poot_graph_ir::ops::rmsnorm(&b, x, norm, 1e-6).unwrap();
    let q = matmul(h, wq);
    let gate = poot_graph_ir::ops::silu(&b, matmul(h, wg)).unwrap();
    let act = b.binary(BinOp::Mul, gate, matmul(h, wu)).unwrap();
    let out = b
        .binary(
            BinOp::Add,
            b.binary(BinOp::Add, x, q).unwrap(),
            matmul(act, wd),
        )
        .unwrap();
    let graph = b.finish(out).unwrap();
    let key = graph.meta(x.id()).slot_key().unwrap().clone();
    let steps = (0..2u64)
        .map(|step| {
            vec![crate::StepFixture {
                key: key.clone(),
                tensor: HostTensor::f32(
                    vec![m, K],
                    crate::fill(m * K, 0x1457 + step)
                        .into_iter()
                        .map(|v| -v.abs() - 0.01)
                        .collect(),
                ),
            }]
        })
        .collect();
    let store = crate::store_with(&graph, |name, data| {
        let sign = match name {
            "norm" => norm_scale,
            "wg" | "wu" => -1.0,
            _ => 1.0,
        };
        data.iter_mut().for_each(|v| *v = sign * (v.abs() + 0.01));
    });
    Fixture {
        name,
        graph,
        store,
        steps,
    }
}

/// The block with its normed activation past f16's range, at a few rows (`M = 5`, the register tiles) and at a
/// prefill whose tiles fill a matrix unit (`M = 160`).
pub fn past_range_fixtures() -> Vec<Fixture> {
    vec![
        block(
            "f16_guard_past_range_m5",
            5,
            Projections::Policy,
            PAST_RANGE,
        ),
        block(
            "f16_guard_past_range_m160",
            160,
            Projections::Policy,
            PAST_RANGE,
        ),
    ]
}

/// The block in f16's range, guarded, beside the same block whose matmuls declare `Narrow(f16)` (unguarded).
pub fn in_range_pairs() -> Vec<(Fixture, Fixture)> {
    [
        ("f16_guard_in_range_m5", "f16_unguarded_in_range_m5", 5),
        (
            "f16_guard_in_range_m160",
            "f16_unguarded_in_range_m160",
            160,
        ),
    ]
    .into_iter()
    .map(|(guarded, unguarded, m)| {
        (
            block(guarded, m, Projections::Policy, 1.0),
            block(unguarded, m, Projections::DeclaredF16, 1.0),
        )
    })
    .collect()
}

/// Every contraction of `fixture`'s program, compiled under the narrow policy for the device `exec` drives, rounds
/// to f16 (so the device's own unit is what the row exercises).
fn assert_f16_contractions(
    exec: &dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
    fixture: &Fixture,
) -> Result<(), String> {
    let staged = crate::staged_under(exec, fixture, target, MatmulPrecision::Narrow);
    let precisions: Vec<OperandPrecision> = staged
        .stages()
        .flat_map(|(_, _, program)| {
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
                .collect::<Vec<_>>()
        })
        .collect();
    let f16 = OperandPrecision::Narrow(FragmentDtype::F16);
    if precisions.len() != 4 || precisions.iter().any(|&p| p != f16) {
        return Err(format!(
            "{}: the four contractions do not all round to f16: {precisions:?}",
            fixture.name
        ));
    }
    Ok(())
}

/// SC-3: the block with activations past f16's range runs under the narrow policy on `exec` with f16
/// contractions, every step's output is finite, and it matches the CPU oracle of the declared semantics (the guarded
/// graph) within the narrow class's tolerance.
pub fn run_past_range(
    exec: &mut dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
) -> Result<(), String> {
    let fixtures = past_range_fixtures();
    for fixture in &fixtures {
        assert_f16_contractions(exec, target, fixture)?;
        let outputs = crate::run_outputs_under(exec, target, fixture, MatmulPrecision::Narrow)?;
        for (step, output) in outputs.iter().enumerate() {
            let values = output
                .to_f32()
                .map_err(|e| format!("{}: an output does not read as f32: {e}", fixture.name))?;
            if let Some(i) = values.iter().position(|v| !v.is_finite()) {
                return Err(format!(
                    "{}: step {step}, element {i} is {}",
                    fixture.name, values[i]
                ));
            }
        }
    }
    crate::precision::run_parity_declared(exec, target, &fixtures, MatmulPrecision::Narrow)
}

/// SC-4: where every row is in f16's range the scale is exactly 1, so the guarded block's output on `exec` is bit
/// for bit the unguarded f16 block's, at every step.
pub fn run_in_range_identity(
    exec: &mut dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
) -> Result<(), String> {
    for (guarded, unguarded) in in_range_pairs() {
        assert_f16_contractions(exec, target, &guarded)?;
        assert_f16_contractions(exec, target, &unguarded)?;
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
        let got = bits(crate::run_outputs_under(
            exec,
            target,
            &guarded,
            MatmulPrecision::Narrow,
        )?)?;
        let want = bits(crate::run_outputs_under(
            exec,
            target,
            &unguarded,
            MatmulPrecision::Narrow,
        )?)?;
        for (step, (got, want)) in got.iter().zip(&want).enumerate() {
            if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
                return Err(format!(
                    "{}: step {step}, element {i}: {} guarded, {} unguarded",
                    guarded.name,
                    f32::from_bits(got[i]),
                    f32::from_bits(want[i])
                ));
            }
        }
    }
    Ok(())
}
