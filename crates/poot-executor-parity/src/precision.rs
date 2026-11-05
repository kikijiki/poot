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
pub(crate) type Heads = (usize, usize, usize);
pub(crate) const SMALL_HEADS: Heads = (4, 2, 16);
/// 16 query heads over 4 key/value heads of width 64: four query heads per key/value head pack a block of four
/// positions into the 16 rows of a fragment, and 160 positions give a device's compute units more blocks than it
/// has.
pub(crate) const FRAGMENT_HEADS: Heads = (16, 4, 64);

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
        attention_fixture(name, precision, heads, (lq, lk), scales, 0.25)
    })
    .collect()
}

/// One attention over a query of magnitude up to `q_scale / 2` and an F16 key/value cache of magnitude up to
/// `kv_scale / 2`, its scores multiplied by `score_scale`.
pub(crate) fn attention_fixture(
    name: &'static str,
    precision: OperandPrecision,
    (hq, hkv, d): Heads,
    (lq, lk): (usize, usize),
    (q_scale, kv_scale): (f32, f32),
    score_scale: f32,
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
                ..AttentionAttrs::new(hq / hkv, score_scale, None)
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
    policy: impl Into<crate::Policy>,
) -> Result<(), String> {
    use poot_eval::test_support::weight_map_oracle::oracle_for;
    let policy = policy.into();
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
        let declared = poot_graph_plan::with_operand_precision(
            &fixture.graph,
            &policy.declaring_target(target),
            policy.precision,
        )
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

/// POOT-1448: a transformer block's projections over a norm, each reading the normed activation narrowed, with `m`
/// rows: `h = rmsnorm(x)`, `q = h Wq^T`, `gate = h Wg^T`, `up = h Wu^T`, `out = x + q + (silu(gate) * up) Wd^T`,
/// every weight a checkpoint-order BF16 `[N, K]` matrix. Under a narrowed policy every reader of `h` and of
/// `silu(gate) * up` narrows it, so `compile` stores both narrow on a target that has the store, and the residual,
/// which f32 arithmetic reads, stays f32. Two steps with different activations.
fn narrowed_block_fixture(name: &'static str, m: usize) -> Fixture {
    const K: usize = 64;
    const F: usize = 96;
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
        let w = b
            .constant(name, TensorType::new(vec![rows, cols], DType::BF16))
            .unwrap();
        b.transpose(w, vec![1, 0]).unwrap()
    };
    let h = poot_graph_ir::ops::rmsnorm(&b, x, norm, 1e-6).unwrap();
    let q = b.matmul(h, weight("wq", K, K)).unwrap();
    let gate = b.matmul(h, weight("wg", F, K)).unwrap();
    let up = b.matmul(h, weight("wu", F, K)).unwrap();
    let act = b
        .binary(
            poot_graph_ir::op::BinOp::Mul,
            poot_graph_ir::ops::silu(&b, gate).unwrap(),
            up,
        )
        .unwrap();
    let down = b.matmul(act, weight("wd", K, F)).unwrap();
    let residual = b.binary(poot_graph_ir::op::BinOp::Add, x, q).unwrap();
    let out = b
        .binary(poot_graph_ir::op::BinOp::Add, residual, down)
        .unwrap();
    let graph = b.finish(out).unwrap();
    let key = graph.meta(x.id()).slot_key().unwrap().clone();
    let steps = (0..2u64)
        .map(|step| {
            vec![crate::StepFixture {
                key: key.clone(),
                tensor: HostTensor::f32(vec![m, K], crate::fill(m * K, 0x1448 + step)),
            }]
        })
        .collect();
    Fixture {
        name,
        store: crate::store_with_bf16(&graph, &["wq", "wg", "wu", "wd"]),
        graph,
        steps,
    }
}

/// POOT-1448: [`narrowed_block_fixture`] at a few rows (`M = 5`, the register tiles) and a prefill whose tiles fill a
/// matrix unit (`M = 160`).
pub fn narrowed_block_fixtures() -> Vec<Fixture> {
    vec![
        narrowed_block_fixture("narrowed_block_m5", 5),
        narrowed_block_fixture("narrowed_block_m160", 160),
    ]
}

/// POOT-1448: the dtype of every value `fixture`'s program, compiled for the device `exec` drives under `policy`,
/// stores in a two-byte float that the traced graph computes in f32, with the op that produces it: the values the
/// planner stored narrow.
pub fn narrow_stored_values(
    exec: &dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
    fixture: &Fixture,
    policy: impl Into<crate::Policy>,
) -> Vec<(String, DType)> {
    crate::staged_under(exec, fixture, target, policy)
        .stages()
        .flat_map(|(_, _, program)| {
            program
                .planned()
                .filter_map(|(eqn, _)| {
                    let dtype = program.graph().aval(eqn.out).dtype;
                    matches!(dtype, DType::BF16 | DType::F16)
                        .then(|| (eqn.op.kind_name().to_string(), dtype))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// POOT-1448: an activation whose every value lies just above a bf16 rounding midpoint, produced on the device by a
/// pointwise kernel (`x * ones * ones`, which computes `x`) and read by one projection over positive BF16 weights:
/// `out = (x * ones * ones) W^T`, `M = 160`, `K = 64`, `N = 64`. Rounded to nearest even, every value rounds up a bf16
/// ulp; a store that truncated would round every value down, and over positive weights the outputs would move by
/// about 7.8e-3 of their size, past the narrow-storage tolerance (5e-3): the row that the producer's narrow store
/// rounds, not only that it narrows.
pub fn narrowed_rounding_fixture() -> Fixture {
    const M: usize = 160;
    const K: usize = 64;
    const N: usize = 64;
    let b = Builder::for_tests();
    let x = b
        .slot_named(
            poot_graph_ir::Slot::Activation,
            "x",
            TensorType::f32(vec![M, K]),
        )
        .unwrap();
    let ones = b.constant("ones", TensorType::f32(vec![K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![N, K], DType::BF16))
        .unwrap();
    let mul = poot_graph_ir::op::BinOp::Mul;
    let scaled = b
        .binary(mul, b.binary(mul, x, ones).unwrap(), ones)
        .unwrap();
    let out = b
        .matmul(scaled, b.transpose(w, vec![1, 0]).unwrap())
        .unwrap();
    let graph = b.finish(out).unwrap();
    let key = graph.meta(x.id()).slot_key().unwrap().clone();
    // `1 + (1/2 + u/4) ulp` for a bf16 ulp of 2^-7 at 1, `u` in [0, 1): strictly above the midpoint.
    let above_midpoints = |seed: u64| -> Vec<f32> {
        crate::fill(M * K, seed)
            .iter()
            .map(|v| 1.0 + (0.5 + 0.125 * (v + 1.0)) * 2f32.powi(-7))
            .collect()
    };
    let steps = (0..2u64)
        .map(|step| {
            vec![crate::StepFixture {
                key: key.clone(),
                tensor: HostTensor::f32(vec![M, K], above_midpoints(0x1448 + step)),
            }]
        })
        .collect();
    let weights: Vec<f32> = crate::fill(N * K, 0x5EED)
        .iter()
        .map(|v| f32::from_bits((0.5 + 0.25 * v).to_bits() & 0xffff_0000))
        .collect();
    Fixture {
        name: "narrowed_rounding",
        store: crate::store_of(vec![
            ("ones", vec![K], DType::F32, vec![1.0; K]),
            ("w", vec![N, K], DType::BF16, weights),
        ]),
        graph,
        steps,
    }
}
