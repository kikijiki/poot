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

/// Query heads, key/value heads and head width of a fixture; [`FRAGMENT_HEADS`] is the shape whose rows fill a tiled
/// fragment attention.
pub(crate) type Heads = (usize, usize, usize);
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
/// its operands as `precision` says: a 160-row
/// prefill of 16 heads whose rows share staged key blocks on the matrix unit of a device that has one (POOT-1437).
/// All hold the keys and values as F16 constants, so the K and V a `Narrow(f16)` product reads are exact in the
/// fragment dtype and only Q and the probabilities round. No decode row or small prefill is here: a decode attention is a
/// streaming or split-KV schedule and a prefill whose rows do not fill the device has no fragment tiling, so the
/// planner leaves them exact and refuses a declared narrowing (POOT-1476).
pub fn attention_fixtures(precision: OperandPrecision) -> Vec<Fixture> {
    [
        // Soft scores and small values: the probabilities of a key block are rounded against the running maximum, the
        // oracle's against the final one, so the two differ by a bf16 ulp of the probabilities, times the values, in
        // the output (6e-3 at values of 2 over 160 keys, outside the 5e-3 class).
        (
            "attention_prefill_fragment",
            FRAGMENT_HEADS,
            160usize,
            160usize,
            (4.0, 1.0),
        ),
    ]
    .into_iter()
    .map(|(name, heads, lq, lk, scales)| {
        attention_fixture(name, precision, heads, (lq, lk), scales)
    })
    .collect()
}

pub(crate) fn attention_fixture(
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
    policy: impl Into<crate::Policy>,
) -> Result<(), String> {
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
        run_against_declared(exec, target, fixture, policy)?;
    }
    Ok(())
}

/// [`run_parity_declared`] for the driver's weight format (POOT-1477): every fixture's compiled program must hold
/// at least one weight the graph converts (declared F16 over the checkpoint's BF16), or the call fails: a program
/// that kept the checkpoint's format would pass by comparing bf16 with bf16. The oracle evaluates the declared
/// graph, whose weights the oracle binds converted as the bind does.
pub fn run_parity_weight_format(
    exec: &mut dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
    fixtures: &[Fixture],
    policy: impl Into<crate::Policy>,
) -> Result<(), String> {
    let policy = policy.into();
    for fixture in fixtures {
        let converted = crate::staged_under(exec, fixture, target, policy)
            .stages()
            .map(|(_, _, program)| {
                let g = program.graph();
                g.inputs()
                    .iter()
                    .filter(|&&id| g.meta(id).stored_dtype() != g.aval(id).dtype)
                    .count()
            })
            .sum::<usize>();
        if converted == 0 {
            return Err(format!(
                "{}: no weight of the program compiled under {policy:?} is converted",
                fixture.name
            ));
        }
        run_against_declared(exec, target, fixture, policy)?;
    }
    Ok(())
}

/// `fixture` run on `exec` under `policy` against the CPU oracle of the graph the policy declares for `target`.
fn run_against_declared(
    exec: &mut dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
    fixture: &Fixture,
    policy: crate::Policy,
) -> Result<(), String> {
    use poot_eval::test_support::weight_map_oracle::oracle_for;
    let declared = if policy.declared {
        poot_graph_plan::declare_operand_precision(
            &fixture.graph,
            &policy.declaring_target(target),
            policy.precision,
        )
    } else {
        poot_graph_plan::with_operand_precision(
            &fixture.graph,
            &policy.declaring_target(target),
            policy.precision,
        )
    }
    .map_err(|e| format!("{}: declare the precision: {e}", fixture.name))?;
    let mut oracle = oracle_for(&declared, &fixture.store);
    crate::run_parity_under(exec, target, fixture, policy, &mut oracle)
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
    let staged = crate::try_staged_under(exec, &fixture, target, MatmulPrecision::Exact)
        .map_err(|error| format!("the planner refused the declared attention: {error}"))?;
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

/// POOT-1477: a BF16 embedding the driver stores as f16, gathered by two token inputs and widened:
/// `out = (embed[a] + embed[b]) * 2^20` over a table of odd width whose magnitudes are `2^e (1 + k/128)` for `e` from
/// -20 to 3: the low exponents are bf16 values f16 holds only as subnormals (16 levels per binade, a relative
/// rounding error up to 3%, far past the narrow-storage tolerance once the scale lifts them to `[1, 4)`), the rest
/// are f16 normals, so the row covers both branches of the kernel's decode. An oracle that read the bf16 table, or a
/// device that read the bf16 lane, would differ. The table has two gathers, so the planner keeps it a device
/// constant instead of hosting its rows (a hosted table is never a device buffer). Two steps with different ids.
pub fn weight_format_embedding_fixture() -> Fixture {
    const VOCAB: usize = 24;
    const WIDTH: usize = 5;
    const IDS: usize = 6;
    let b = Builder::for_tests();
    let embed = b
        .constant("embed", TensorType::new(vec![VOCAB, WIDTH], DType::BF16))
        .unwrap();
    let scale = b.constant("scale", TensorType::f32(vec![WIDTH])).unwrap();
    let ids = TensorType::new(vec![IDS], DType::I32);
    let first = b
        .slot_named(poot_graph_ir::Slot::Token, "first", ids.clone())
        .unwrap();
    let second = b
        .slot_named(poot_graph_ir::Slot::Token, "second", ids)
        .unwrap();
    let rows = |token| {
        b.cast(b.gather(embed, 0, token).unwrap(), DType::F32)
            .unwrap()
    };
    let sum = b
        .binary(poot_graph_ir::op::BinOp::Add, rows(first), rows(second))
        .unwrap();
    let out = b.binary(poot_graph_ir::op::BinOp::Mul, sum, scale).unwrap();
    let graph = b.finish(out).unwrap();
    let keys = [first, second].map(|t| graph.meta(t.id()).slot_key().unwrap().clone());
    let table: Vec<f32> = (0..VOCAB * WIDTH)
        .map(|i| {
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            sign * (1.0 + ((i * 37) % 128) as f32 / 128.0) * 2f32.powi(-20 + (i as i32 * 5) % 24)
        })
        .collect();
    let steps = [
        ([0, 23, 7, 7, 12, 1], [5, 5, 22, 0, 9, 17]),
        ([3, 4, 5, 6, 7, 8], [23, 22, 21, 20, 19, 18]),
    ]
    .into_iter()
    .map(|(a, b)| {
        keys.iter()
            .zip([a, b])
            .map(|(key, ids)| crate::StepFixture {
                key: key.clone(),
                tensor: HostTensor::i32(vec![IDS], ids.to_vec()),
            })
            .collect()
    })
    .collect();
    Fixture {
        name: "weight_format_embedding",
        store: crate::store_of(vec![
            ("embed", vec![VOCAB, WIDTH], DType::BF16, table),
            ("scale", vec![WIDTH], DType::F32, vec![2f32.powi(20); WIDTH]),
        ]),
        graph,
        steps,
    }
}

/// POOT-1477: a language-model step at `m` rows: the embedding gathered by token (hosted: one gather, untied), a
/// projection and a head, every device weight a BF16 table the driver stores as f16. `m = 1` plans the decode GEMV, `m = 64` the tiled schedules; the program holds the same weight
/// format for both.
pub fn weight_format_lm_fixtures() -> Vec<Fixture> {
    const VOCAB: usize = 48;
    const K: usize = 64;
    const N: usize = 32;
    [
        ("weight_format_lm_m1", 1usize),
        ("weight_format_lm_m64", 64),
    ]
    .into_iter()
    .map(|(name, m)| {
        let b = Builder::for_tests();
        let embed = b
            .constant("embed", TensorType::new(vec![VOCAB, K], DType::BF16))
            .unwrap();
        let head = b
            .constant("head", TensorType::new(vec![VOCAB, N], DType::BF16))
            .unwrap();
        let w = b
            .constant("w", TensorType::new(vec![N, K], DType::BF16))
            .unwrap();
        let token = b
            .slot(
                poot_graph_ir::Slot::Token,
                TensorType::new(vec![m], DType::I32),
            )
            .unwrap();
        let x = b
            .cast(b.gather(embed, 0, token).unwrap(), DType::F32)
            .unwrap();
        let h = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let logits = b.matmul(h, b.transpose(head, vec![1, 0]).unwrap()).unwrap();
        let graph = b.finish(logits).unwrap();
        let key = graph.meta(token.id()).slot_key().unwrap().clone();
        let steps = (0..2i32)
            .map(|step| {
                vec![crate::StepFixture {
                    key: key.clone(),
                    tensor: HostTensor::i32(
                        vec![m],
                        (0..m as i32)
                            .map(|i| (7 * i + 3 * step) % VOCAB as i32)
                            .collect(),
                    ),
                }]
            })
            .collect();
        let bf16 = |seed: u64, n: usize| -> Vec<f32> {
            crate::fill(n, seed)
                .iter()
                .map(|v| f32::from_bits((v * 2.0).to_bits() & 0xffff_0000))
                .collect()
        };
        Fixture {
            name,
            store: crate::store_of(vec![
                (
                    "embed",
                    vec![VOCAB, K],
                    DType::BF16,
                    bf16(0x1477, VOCAB * K),
                ),
                ("head", vec![VOCAB, N], DType::BF16, bf16(0x1478, VOCAB * N)),
                ("w", vec![N, K], DType::BF16, bf16(0x1479, N * K)),
            ]),
            graph,
            steps,
        }
    })
    .collect()
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
