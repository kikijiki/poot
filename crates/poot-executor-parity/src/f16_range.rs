//! The f16 range on a device under the default `Narrow` (POOT-1474,
//! `receipts/1471-narrow-without-range-guards/design.md` section 7). A contraction or attention the policy rounds
//! to f16 reads its operand unscaled, so a value past f16's range becomes infinity and reaches the output as a
//! non-finite value.
//!
//! The block is a transformer block over `[K, N]` f32 weights: `h = rmsnorm(x)`, `out = x + h Wq + (silu(h Wg) * h
//! Wu) Wd`. The norm-fed projections take their scale from the norm's row region; the down projection's input is
//! an epilogue's value and takes it from a row region of its own.
//!
//! The weights are positive after the gate and up (negative) signs, so no sum cancels to a finite value by
//! accident: the infinity of an overflowed activation survives to the output.

use poot_graph_ir::op::BinOp;
use poot_graph_ir::{
    Builder, FragmentDtype, MatmulPrecision, OpKind, OperandPrecision, TensorType, Traced,
};
use poot_tensor::HostTensor;

use crate::Fixture;
use crate::precision::{FRAGMENT_HEADS, Heads, SMALL_HEADS, attention_fixture};

const K: usize = 64;
const F: usize = 96;

/// The norm weight's scale that puts the normed activation past f16's range: `|h|` reaches about `1e6 x 0.25`.
const PAST_RANGE: f32 = 1.0e6;

fn block(name: &'static str, m: usize) -> Fixture {
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
    let matmul = |a: Traced, w: Traced| b.matmul(a, w).unwrap();
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
                    crate::fill(m * K, 0x1474 + step)
                        .into_iter()
                        .map(|v| -v.abs() - 0.01)
                        .collect(),
                ),
            }]
        })
        .collect();
    let store = crate::store_with(&graph, |name, data| {
        let sign = match name {
            "norm" => PAST_RANGE,
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
pub fn block_fixtures() -> Vec<Fixture> {
    vec![
        block("f16_past_range_m5", 5),
        block("f16_past_range_m160", 160),
    ]
}

/// The attentions of `precision::attention_fixtures`' shapes with every query row past f16's range (`|q|` up to
/// 1e6): a decode row against a 64-key cache, a 33-row causal prefill, and a 160-row prefill of 16 heads whose rows
/// fill a fragment attention where the device has a matrix unit.
pub fn attention_fixtures() -> Vec<Fixture> {
    let shapes: [(&'static str, Heads, usize, usize); 3] = [
        ("attention_f16_past_range_decode", SMALL_HEADS, 1, 64),
        ("attention_f16_past_range_prefill", SMALL_HEADS, 33, 33),
        (
            "attention_f16_past_range_fragment",
            FRAGMENT_HEADS,
            160,
            160,
        ),
    ];
    shapes
        .into_iter()
        .map(|(name, heads, lq, lk)| {
            attention_fixture(name, OperandPrecision::Exact, heads, (lq, lk), (2.0e6, 2.0))
        })
        .collect()
}

/// Every contraction and attention of `fixture`'s program, compiled under the narrow policy for the device `exec`
/// drives, rounds to f16 (so the device's own unit is what the row exercises), and `count` of them exist.
fn assert_f16_operands(
    exec: &dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
    fixture: &Fixture,
    count: usize,
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
                            | OpKind::Attention(_)
                    )
                })
                .filter_map(|(eqn, _)| eqn.op.operand_precision())
                .collect::<Vec<_>>()
        })
        .collect();
    let f16 = OperandPrecision::Narrow(FragmentDtype::F16);
    if precisions.len() != count || precisions.iter().any(|&p| p != f16) {
        return Err(format!(
            "{}: expected {count} f16 operations, found {precisions:?}",
            fixture.name
        ));
    }
    Ok(())
}

/// The default's failure mode on a device: with activations (or queries) past f16's range, every contraction and
/// attention runs under the narrow policy in f16 with no scale, and every step's output has a non-finite element:
/// the device neither clamps nor flushes the overflow to a finite value.
pub fn run_past_range(
    exec: &mut dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
) -> Result<(), String> {
    let fixtures = block_fixtures()
        .into_iter()
        .map(|fixture| (fixture, 4))
        .chain(attention_fixtures().into_iter().map(|fixture| (fixture, 1)));
    for (fixture, count) in fixtures {
        assert_f16_operands(exec, target, &fixture, count)?;
        let outputs = crate::run_outputs_under(exec, target, &fixture, MatmulPrecision::Narrow)?;
        for (step, output) in outputs.iter().enumerate() {
            let values = output
                .to_f32()
                .map_err(|e| format!("{}: an output does not read as f32: {e}", fixture.name))?;
            if values.iter().all(|v| v.is_finite()) {
                return Err(format!(
                    "{}: step {step}: the overflow reached the output as finite values",
                    fixture.name
                ));
            }
        }
    }
    Ok(())
}
