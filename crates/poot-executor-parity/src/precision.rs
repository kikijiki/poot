//! POOT-1430: attention fixtures that declare their operand precision on the op.
//!
//! A contraction picks its precision from the driver's policy (the parity rows compile the projection fixtures
//! under each policy and hold the device to the oracle of `poot_graph_plan::with_operand_precision`). An attention
//! fixture instead states the precision on its `Attention` equation, so the oracle evaluates exactly the graph the
//! device compiles: the flash attention recognizer, the attention schedules and the scalar rounding of Q, K, V and
//! the probabilities are what the row runs.

use poot_graph_ir::{AttentionAttrs, Builder, OpKind, Operand, OperandPrecision, TensorType};
use poot_tensor::{DType, HostTensor};

use crate::{Fixture, const_fixture};

const HQ: usize = 4;
const HKV: usize = 2;
const D: usize = 16;

fn centered(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * scale
        })
        .collect()
}

/// An additive causal mask `[1, 1, lq, lk]`: query row `i` sees keys `0..=i + lk - lq`.
fn causal_mask(lq: usize, lk: usize) -> Vec<f32> {
    (0..lq)
        .flat_map(|i| (0..lk).map(move |j| if j <= i + lk - lq { 0.0 } else { -1.0e9 }))
        .collect()
}

/// One attention equation over an F32 query and an F16 key/value cache (the engine's default KV dtype), reading
/// its operands as `precision` says: a decode row against a 64-key cache, and a 33-row causal prefill. Both hold
/// the keys and values as F16 constants, so the K and V a `Narrow(f16)` product reads are exact in the fragment
/// dtype and only Q and the probabilities round.
pub fn attention_fixtures(precision: OperandPrecision) -> Vec<Fixture> {
    [
        ("attention_decode", 1usize, 64usize),
        ("attention_prefill", 33, 33),
    ]
    .into_iter()
    .map(|(name, lq, lk)| attention_fixture(name, precision, lq, lk))
    .collect()
}

fn attention_fixture(
    name: &'static str,
    precision: OperandPrecision,
    lq: usize,
    lk: usize,
) -> Fixture {
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, HQ, lq, D]))
        .unwrap();
    let kv_type = TensorType::new(vec![1, HKV, lk, D], DType::F16);
    let k = b.constant("k", kv_type.clone()).unwrap();
    let v = b.constant("v", kv_type).unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, lq, lk]))
        .unwrap();
    let mut plan = b.append_plan(0).unwrap();
    let out = plan
        .equation(
            OpKind::Attention(AttentionAttrs {
                precision,
                ..AttentionAttrs::new(HQ / HKV, 0.25, None)
            }),
            [q, k, v, mask].map(|t| Operand::Value(t.id())).to_vec(),
        )
        .expect("the attention types");
    plan.declare_result(out).unwrap();
    let mut prepared = b.preflight_append(plan).unwrap();
    let out = b.commit_append(&mut prepared).unwrap();
    let graph = b.finish(out).unwrap();
    let f16 = |seed, n| {
        HostTensor::f16(
            vec![1, HKV, lk, D],
            centered(seed, n, 4.0)
                .into_iter()
                .map(poot_quant::scalar::f32_to_f16)
                .collect::<Vec<u16>>(),
        )
    };
    const_fixture(
        name,
        graph,
        vec![
            (
                "q",
                HostTensor::f32(vec![1, HQ, lq, D], centered(11, HQ * lq * D, 40.0)),
            ),
            ("k", f16(12, HKV * lk * D)),
            ("v", f16(13, HKV * lk * D)),
            (
                "mask",
                HostTensor::f32(vec![1, 1, lq, lk], causal_mask(lq, lk)),
            ),
        ],
        2,
    )
}

/// Run every fixture on `exec` compiled under the operand-precision policy `policy`, each against the CPU oracle of
/// the semantics the policy declares for `target` ([`poot_graph_plan::with_operand_precision`]): the device must
/// compute the declared rounding, whatever schedule the planner picked, to within the program's published class.
///
/// The row must show it narrowed: every fixture's compiled program holds at least one equation that reads its
/// operands under a non-exact precision, or the call fails. A device whose caps report no matrix unit compiles
/// `Exact` under any policy, and a run of such a program would pass by comparing exact with exact.
pub fn run_parity_declared(
    exec: &mut dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
    fixtures: &[Fixture],
    policy: poot_graph_ir::MatmulPrecision,
) -> Result<(), String> {
    use poot_eval::test_support::weight_map_oracle::oracle_for;
    for fixture in fixtures {
        let narrowed = crate::staged_under(exec, fixture, target, policy)
            .stages()
            .flat_map(|(_, _, program)| {
                program
                    .planned()
                    .filter_map(|(eqn, _)| eqn.op.operand_precision())
                    .collect::<Vec<_>>()
            })
            .any(|precision| !precision.is_exact());
        if !narrowed {
            return Err(format!(
                "{}: no equation of the program compiled under {policy:?} reads narrowed operands",
                fixture.name
            ));
        }
        let declared = poot_graph_plan::with_operand_precision(&fixture.graph, &target, policy)
            .map_err(|e| format!("{}: declare the precision: {e}", fixture.name))?;
        let mut oracle = oracle_for(&declared, &fixture.store);
        crate::run_parity_under(exec, target, fixture, policy, &mut oracle)?;
    }
    Ok(())
}
