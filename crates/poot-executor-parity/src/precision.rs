//! POOT-1430: attention fixtures that declare their operand precision on the op.
//!
//! A contraction picks its precision from the driver's policy (the parity rows compile the projection fixtures
//! under each policy and hold the device to the oracle of `poot_graph_plan::with_operand_precision`). An attention
//! fixture instead states the precision on its `Attention` equation, so the oracle evaluates exactly the graph the
//! device compiles: the flash attention recognizer, the attention schedules and the scalar rounding of Q, K, V and
//! the probabilities are what the row runs.

use poot_graph_ir::{
    AttentionAttrs, Builder, MatmulPrecision, OpKind, Operand, OperandPrecision, TensorType,
};
use poot_tensor::{DType, HostTensor};

use crate::{Fixture, const_fixture};

/// Query heads, key/value heads and head width of the small fixtures; [`FRAGMENT_HEADS`] is the shape whose rows
/// fill a tiled fragment attention.
type Heads = (usize, usize, usize);
const SMALL_HEADS: Heads = (4, 2, 16);
/// 16 query heads over 4 key/value heads of width 64: four query heads per key/value head pack a block of four
/// positions into the 16 rows of a fragment, and 160 positions give a device's compute units more blocks than it
/// has.
const FRAGMENT_HEADS: Heads = (16, 4, 64);

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
/// its operands as `precision` says: a decode row against a 64-key cache, a 33-row causal prefill, and a 160-row
/// prefill of 16 heads whose rows share staged key blocks on the matrix unit of a device that has one (POOT-1437).
/// All hold the keys and values as F16 constants, so the K and V a `Narrow(f16)` product reads are exact in the
/// fragment dtype and only Q and the probabilities round.
pub fn attention_fixtures(precision: OperandPrecision) -> Vec<Fixture> {
    [
        (
            "attention_decode",
            SMALL_HEADS,
            1usize,
            64usize,
            (40.0, 4.0),
        ),
        ("attention_prefill", SMALL_HEADS, 33, 33, (40.0, 4.0)),
        // Soft scores and small values: the probabilities of a key block are rounded against the running maximum, the
        // oracle's against the final one, so the two differ by a bf16 ulp of the probabilities, times the values, in
        // the output (6e-3 at values of 2 over 160 keys, outside the 5e-3 class).
        (
            "attention_prefill_fragment",
            FRAGMENT_HEADS,
            160,
            160,
            (4.0, 1.0),
        ),
    ]
    .into_iter()
    .map(|(name, heads, lq, lk, scales)| {
        attention_fixture(name, precision, heads, (lq, lk), scales)
    })
    .collect()
}

fn attention_fixture(
    name: &'static str,
    precision: OperandPrecision,
    (hq, hkv, d): Heads,
    (lq, lk): (usize, usize),
    (q_scale, kv_scale): (f32, f32),
) -> Fixture {
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, hq, lq, d]))
        .unwrap();
    let kv_type = TensorType::new(vec![1, hkv, lk, d], DType::F16);
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
                ..AttentionAttrs::new(hq / hkv, 0.25, None)
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
            vec![1, hkv, lk, d],
            centered(seed, n, kv_scale)
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
                HostTensor::f32(vec![1, hq, lq, d], centered(11, hq * lq * d, q_scale)),
            ),
            ("k", f16(12, hkv * lk * d)),
            ("v", f16(13, hkv * lk * d)),
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

/// The fragment micro-tile the planner picks for the `attention_prefill_fragment` fixture of `precision` on the device
/// `exec` drives, or an error naming the schedule it picked instead: the row that proves the parity run exercised the
/// matrix unit and not the register tiles.
pub fn planned_fragment_attention(
    exec: &dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
    precision: OperandPrecision,
) -> Result<poot_kernelgen::AttentionMicro, String> {
    use poot_kernelgen::{AttentionMicro, AttentionSchedule, KernelRequest};
    let fixture = attention_fixtures(precision)
        .into_iter()
        .find(|fixture| fixture.name == "attention_prefill_fragment")
        .expect("the fragment fixture");
    let staged = crate::staged_under(exec, &fixture, target, MatmulPrecision::Exact);
    let (_, _, program) = staged.stages().next().ok_or("no stage")?;
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::Attention(_)))
        .ok_or("no attention equation")?;
    let spec = match program.kernel_choice(eqn) {
        poot_graph_plan::KernelChoice::Generated(KernelRequest::Attention(spec)) => spec.clone(),
        poot_graph_plan::KernelChoice::Generated(KernelRequest::AttentionRows(rows)) => {
            rows.spec.clone()
        }
        other => return Err(format!("not a generated attention: {other:?}")),
    };
    match spec.schedule {
        AttentionSchedule::Tiled {
            micro: micro @ AttentionMicro::Fragment { .. },
            ..
        } => Ok(micro),
        other => Err(format!("the planner picked {other:?}, not fragments")),
    }
}
