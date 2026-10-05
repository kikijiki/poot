//! Differential CPU-oracle test for spec 249 (MoE shared-pool batched decode):
//! `poot_models::moe_decode::trace_moe_decode_kv_masked_batched_shared_pool` and its per-arch wrappers
//! (`granite::trace_granite_decode_kv_masked_batched_shared_pool`,
//! `qwen3moe::trace_qwen3_moe_decode_kv_masked_batched_shared_pool`) must give each batch row the same logits as an
//! independent single-sequence contiguous decode (`trace_granite_decode_kv_masked`/`trace_qwen3_moe_decode_kv_masked`)
//! over that row's token stream (the "B independent batch-1 decodes" contract). The harness follows
//! `gemma4_batched_decode_matches_sequential_single_seq`: a tight `pool_slots < batch*cap` shared pool, an interleaved
//! global slot map, and two different per-row token streams so a batch mixup shows as a wrong logits row.
//!
//! Weights are filled per name via `bind_const` as in `moe_paged_prefill`; both tracers declare identical weight names.
//!
//! Also covers Mixtral's batched shared-pool decode tracer (`mixtral::trace_mixtral_decode_kv_masked_batched_shared_pool`),
//! the dense `[E,...]`-stacked-weights path (not the expert-pool one), built on the same generic core.
//! `mixtral_batched_shared_pool_decode_matches_sequential_staggered` drives a staggered admit/evict schedule (not every
//! row active every step, one physical slot reused).

use crate::{EvalBudget, EvalError, EvalOptions, Value};
use poot_graph_ir::{Graph, Slot, Storage, ValueId};
use poot_models::granite::{
    GraniteParams, MoeShape, trace_granite_decode_kv_masked,
    trace_granite_decode_kv_masked_batched_shared_pool,
};
use poot_models::mixtral::{
    MixtralParams, trace_mixtral_decode_kv_masked,
    trace_mixtral_decode_kv_masked_batched_shared_pool,
};
use poot_models::qwen2::Qwen2Config;
use poot_models::qwen3moe::{
    Qwen3MoeParams, trace_qwen3_moe_decode_kv_masked,
    trace_qwen3_moe_decode_kv_masked_batched_shared_pool,
};
use poot_tensor::HostTensor;
use std::collections::HashMap as Map;

use super::helpers::*;
use poot_test_util::max_abs_error;

/// Deterministic per-name weight fill, identical to `moe_paged_prefill::bind_const` (decode and prefill tracers for an
/// arch share weight names).
fn bind_const(name: &str, shape: &[usize]) -> HostTensor {
    let seed: u64 = name.bytes().fold(1469598103934665603u64, |acc, c| {
        (acc ^ c as u64).wrapping_mul(1099511628211)
    });
    HostTensor::f32(
        shape.to_vec(),
        fill(shape.iter().product::<usize>().max(1), seed)
            .iter()
            .map(|v| v * 0.1)
            .collect(),
    )
}

/// Bind one single-sequence contiguous decode step (`trace_granite_decode_kv_masked`/`trace_qwen3_moe_decode_kv_masked`):
/// scalar `Token`, `Pos` filled with `pos` (shape `[1,1]` for the card-550 dense path, scalar for MoE's unconverted
/// path), scalar `SeqLen` and a `[cap]` causal `Mask` for MoE's unconverted path (0 for `t <= pos`, else -1e9), and
/// every `Const` by name. `State` is left unbound; the caller threads the carried caches.
fn bind_seq_step(g: &Graph, tok: u32, pos: usize, cap: usize) -> Map<ValueId, HostTensor> {
    let mut inputs = Map::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                inputs.insert(id, HostTensor::i32(vec![], vec![tok as i32]));
            }
            Storage::Slot(Slot::Pos) => {
                // Dense (card 550) tracers declare Pos `[1,1]`; MoE tracers (out of card 550's scope) still
                // declare it scalar. Fill by numel so both bind from the same `pos` value.
                let n = m.aval.numel().max(1);
                inputs.insert(
                    id,
                    HostTensor::i32(m.aval.shape.clone(), vec![pos as i32; n]),
                );
            }
            Storage::Slot(Slot::SeqLen) => {
                inputs.insert(id, HostTensor::i32(vec![], vec![(pos + 1) as i32]));
            }
            Storage::Slot(Slot::Mask) => {
                let mask: Vec<f32> = (0..cap)
                    .map(|t| if t <= pos { 0.0 } else { -1.0e9 })
                    .collect();
                inputs.insert(id, HostTensor::f32(vec![cap], mask));
            }
            Storage::Slot(other) => {
                unreachable!("unexpected slot {other:?} in single-sequence MoE decode")
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(id, bind_const(name, &m.aval.shape));
            }
            Storage::Computed(computed) => {
                inputs.insert(id, HostTensor::f32(computed.shape(), computed.values_f32()));
            }
            Storage::State => {}
            Storage::Device => unreachable!(),
        }
    }
    inputs
}

/// Bind one batched shared-pool decode step: `[batch]` `Token`/`Pos` (all rows at the same `step`), scalar `SeqLen`, a
/// `[batch,cap]` causal `Mask`, the `[batch,cap]` global `SlotMap`, and every `Const` by name.
fn bind_batched_step(
    g: &Graph,
    tokens: &[u32],
    step: usize,
    cap: usize,
    batch: usize,
    global_slotmap: &[i32],
) -> Map<ValueId, HostTensor> {
    let mut inputs = Map::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![batch], tokens.iter().map(|&t| t as i32).collect()),
                );
            }
            Storage::Slot(Slot::Pos) => {
                inputs.insert(id, HostTensor::i32(vec![batch], vec![step as i32; batch]));
            }
            Storage::Slot(Slot::SeqLen) => {
                inputs.insert(id, HostTensor::i32(vec![], vec![(step + 1) as i32]));
            }
            Storage::Slot(Slot::Mask) => {
                let row: Vec<f32> = (0..cap)
                    .map(|t| if t <= step { 0.0 } else { -1.0e9 })
                    .collect();
                let mask: Vec<f32> = (0..batch).flat_map(|_| row.clone()).collect();
                inputs.insert(id, HostTensor::f32(vec![batch, cap], mask));
            }
            Storage::Slot(Slot::SlotMap) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![batch, cap], global_slotmap.to_vec()),
                );
            }
            Storage::Slot(other) => {
                unreachable!("unexpected slot {other:?} in batched shared-pool MoE decode")
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(id, bind_const(name, &m.aval.shape));
            }
            Storage::Computed(computed) => {
                inputs.insert(id, HostTensor::f32(computed.shape(), computed.values_f32()));
            }
            Storage::State => {}
            Storage::Device => unreachable!(),
        }
    }
    inputs
}

fn zero_state(g: &Graph) -> Vec<HostTensor> {
    g.state
        .iter()
        .map(|&(sid, _)| HostTensor::zeros(g.aval(sid).shape.clone()))
        .collect()
}

// `row`/`step` index both `row_tokens` and other per-row/per-step structures built inline in the loop body, so an
// iterator rewrite would need a second index anyway.
#[allow(clippy::needless_range_loop)]
#[test]
fn granitemoe_batched_shared_pool_decode_matches_sequential_single_seq() {
    let cfg = Qwen2Config {
        vocab: 32,
        hidden: 16,
        inter: 24,
        layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        rotary_dim: 4,
        eps: 1e-6,
        max_pos: 32,
        qkv_bias: false,
        qk_norm: false,
        ..Default::default()
    };
    let gp = GraniteParams {
        moe: Some(MoeShape {
            n_experts: 4,
            top_k: 2,
            inter: 12,
        }),
        embed_mult: 1.5,
        attn_mult: 0.0625,
        residual_mult: 0.7,
        logits_scale: 3.0,
    };
    let batch = 2usize;
    let cap = 6usize;
    let n_steps = 4usize; // < cap, headroom past the last generated position
    let pool_slots = batch * n_steps; // tight: forces the shared-pool slot-map indirection to matter
    // two different token streams per row, so a batch mixup shows as a wrong logits row.
    let row_tokens: [[u32; 4]; 2] = [[1, 3, 2, 0], [5, 6, 4, 7]];

    // --- Reference: for each row, n_steps sequential single-sequence contiguous decode steps from zero state over that
    // row's own token stream. ---
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(batch);
    for row in 0..batch {
        let g = trace_granite_decode_kv_masked(cfg, gp, cap);
        let mut caches = zero_state(&g);
        let mut last_logits = HostTensor::zeros(vec![1, 1, cfg.vocab]);
        for (pos, &tok) in row_tokens[row].iter().enumerate() {
            let mut inputs = bind_seq_step(&g, tok, pos, cap);
            for (ci, &(sid, _)) in g.state.iter().enumerate() {
                inputs.insert(sid, caches[ci].clone());
            }
            let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
                let values: std::collections::HashMap<ValueId, Value> = inputs
                    .iter()
                    .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                    .collect();
                let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })()
            .expect("granite decode eval");
            last_logits = logits;
            caches = new_states;
        }
        assert_eq!(last_logits.shape(), vec![1, 1, cfg.vocab]);
        ref_logits.push(last_logits.as_f32().unwrap().to_vec());
    }

    // --- Batched shared-pool trace: n_steps steps over both rows at once, zero-start state, one shared pool. Interleaved
    // global slot map: row r's logical position t lands at pool slot t*batch + r. ---
    let g = trace_granite_decode_kv_masked_batched_shared_pool(cfg, gp, cap, batch, pool_slots);
    g.validate()
        .expect("granitemoe batched shared-pool decode graph should validate");
    let mut caches = zero_state(&g);
    let mut batched_logits = HostTensor::zeros(vec![batch, 1, cfg.vocab]);
    for step in 0..n_steps {
        let tokens: Vec<u32> = (0..batch).map(|row| row_tokens[row][step]).collect();
        let global_slotmap: Vec<i32> = (0..batch)
            .flat_map(|row| {
                (0..cap).map(move |t| {
                    if t < n_steps {
                        (t * batch + row) as i32
                    } else {
                        0
                    }
                })
            })
            .collect();
        let mut inputs = bind_batched_step(&g, &tokens, step, cap, batch, &global_slotmap);
        for (ci, &(sid, _)) in g.state.iter().enumerate() {
            inputs.insert(sid, caches[ci].clone());
        }
        let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: std::collections::HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .expect("granitemoe batched decode eval");
        batched_logits = logits;
        caches = new_states;
    }
    assert_eq!(batched_logits.shape(), vec![batch, 1, cfg.vocab]);

    for row in 0..batch {
        let got = &batched_logits.as_f32().unwrap()[row * cfg.vocab..(row + 1) * cfg.vocab];
        let want = &ref_logits[row];
        let err = max_abs_error(got, want);
        assert!(
            err < 1e-4,
            "granitemoe row {row}: batched vs sequential logits max_abs={err:.3e} >= 1e-4"
        );
    }
}

// `row`/`step` index both `row_tokens` and other per-row/per-step structures built inline in the loop body, so an
// iterator rewrite would need a second index anyway.
#[allow(clippy::needless_range_loop)]
#[test]
fn qwen3_moe_batched_shared_pool_decode_matches_sequential_single_seq() {
    let cfg = Qwen2Config {
        vocab: 32,
        hidden: 16,
        inter: 24,
        layers: 3,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        rotary_dim: 4,
        eps: 1e-6,
        max_pos: 32,
        qkv_bias: false,
        qk_norm: true,
        ..Default::default()
    };
    // Mixed dense/MoE layers (layer 0 dense, layer 1 routed, layer 2 dense): the per-layer switch must survive the batched
    // shared-pool decode tracer, not just prefill.
    let mp = Qwen3MoeParams {
        n_experts: 4,
        top_k: 2,
        inter: 12,
        sparse_layer: vec![false, true, false],
    };
    let batch = 2usize;
    let cap = 6usize;
    let n_steps = 4usize;
    let pool_slots = batch * n_steps;
    let row_tokens: [[u32; 4]; 2] = [[1, 3, 2, 0], [5, 6, 4, 7]];

    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(batch);
    for row in 0..batch {
        let g = trace_qwen3_moe_decode_kv_masked(cfg, mp.clone(), cap);
        let mut caches = zero_state(&g);
        let mut last_logits = HostTensor::zeros(vec![1, 1, cfg.vocab]);
        for (pos, &tok) in row_tokens[row].iter().enumerate() {
            let mut inputs = bind_seq_step(&g, tok, pos, cap);
            for (ci, &(sid, _)) in g.state.iter().enumerate() {
                inputs.insert(sid, caches[ci].clone());
            }
            let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
                let values: std::collections::HashMap<ValueId, Value> = inputs
                    .iter()
                    .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                    .collect();
                let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })()
            .expect("qwen3-moe decode eval");
            last_logits = logits;
            caches = new_states;
        }
        assert_eq!(last_logits.shape(), vec![1, 1, cfg.vocab]);
        ref_logits.push(last_logits.as_f32().unwrap().to_vec());
    }

    let g = trace_qwen3_moe_decode_kv_masked_batched_shared_pool(cfg, mp, cap, batch, pool_slots);
    g.validate()
        .expect("qwen3-moe batched shared-pool decode graph should validate");
    let mut caches = zero_state(&g);
    let mut batched_logits = HostTensor::zeros(vec![batch, 1, cfg.vocab]);
    for step in 0..n_steps {
        let tokens: Vec<u32> = (0..batch).map(|row| row_tokens[row][step]).collect();
        let global_slotmap: Vec<i32> = (0..batch)
            .flat_map(|row| {
                (0..cap).map(move |t| {
                    if t < n_steps {
                        (t * batch + row) as i32
                    } else {
                        0
                    }
                })
            })
            .collect();
        let mut inputs = bind_batched_step(&g, &tokens, step, cap, batch, &global_slotmap);
        for (ci, &(sid, _)) in g.state.iter().enumerate() {
            inputs.insert(sid, caches[ci].clone());
        }
        let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: std::collections::HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .expect("qwen3-moe batched decode eval");
        batched_logits = logits;
        caches = new_states;
    }
    assert_eq!(batched_logits.shape(), vec![batch, 1, cfg.vocab]);

    for row in 0..batch {
        let got = &batched_logits.as_f32().unwrap()[row * cfg.vocab..(row + 1) * cfg.vocab];
        let want = &ref_logits[row];
        let err = max_abs_error(got, want);
        assert!(
            err < 1e-4,
            "qwen3-moe row {row}: batched vs sequential logits max_abs={err:.3e} >= 1e-4"
        );
    }
}

/// One staggered-admission request for [`mixtral_batched_shared_pool_decode_matches_sequential_staggered`]: occupies
/// `slot` for `tokens.len()` consecutive engine steps from `admit_step`, replaying `tokens` one per step (position
/// 0-based). Requests may reuse a `slot` in non-overlapping ranges.
struct MixtralReq {
    slot: usize,
    admit_step: usize,
    tokens: Vec<u32>,
}

/// Staggered schedule (5 requests over 4 physical slots, one slot reused): step 0 has one active slot, step 5 has all 4, step 7 admits a
/// request into a just-freed slot while two others are in flight.
fn mixtral_staggered_schedule() -> Vec<MixtralReq> {
    vec![
        MixtralReq {
            slot: 0,
            admit_step: 0,
            tokens: vec![0, 3, 1, 5, 2, 4],
        },
        MixtralReq {
            slot: 1,
            admit_step: 2,
            tokens: vec![4, 1, 0, 2],
        },
        MixtralReq {
            slot: 2,
            admit_step: 3,
            tokens: vec![5, 2, 4, 1, 3],
        },
        MixtralReq {
            slot: 3,
            admit_step: 5,
            tokens: vec![1, 0, 5],
        },
        // Reuses slot 0 after the first request there finishes at engine step 6.
        MixtralReq {
            slot: 0,
            admit_step: 7,
            tokens: vec![2, 4, 0, 3],
        },
    ]
}

/// Bind one batched shared-pool decode step where physical rows sit at independent positions. Unlike
/// [`bind_batched_step`]'s shared `step`, a staggered schedule needs each row's own `pos`; a free row stays at `pos = 0`
/// (as `Runner::continuous_schedule`; its output is discarded). `[batch]` `Token`/`Pos`, scalar `SeqLen`, a `[batch,cap]`
/// `Mask` with each row's causal row (`t <= positions[row]`), the `[batch,cap]` global `SlotMap`, and every `Const`.
fn bind_batched_step_staggered(
    g: &Graph,
    tokens: &[u32],
    positions: &[usize],
    cap: usize,
    batch: usize,
    global_slotmap: &[i32],
) -> Map<ValueId, HostTensor> {
    let mut inputs = Map::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![batch], tokens.iter().map(|&t| t as i32).collect()),
                );
            }
            Storage::Slot(Slot::Pos) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![batch], positions.iter().map(|&p| p as i32).collect()),
                );
            }
            Storage::Slot(Slot::SeqLen) => {
                inputs.insert(
                    id,
                    HostTensor::i32(
                        vec![],
                        vec![(positions.iter().max().copied().unwrap_or(0) + 1) as i32],
                    ),
                );
            }
            Storage::Slot(Slot::Mask) => {
                let mask: Vec<f32> = positions
                    .iter()
                    .flat_map(|&pos| (0..cap).map(move |t| if t <= pos { 0.0 } else { -1.0e9 }))
                    .collect();
                inputs.insert(id, HostTensor::f32(vec![batch, cap], mask));
            }
            Storage::Slot(Slot::SlotMap) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![batch, cap], global_slotmap.to_vec()),
                );
            }
            Storage::Slot(other) => {
                unreachable!("unexpected slot {other:?} in batched shared-pool MoE decode")
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(id, bind_const(name, &m.aval.shape));
            }
            Storage::Computed(computed) => {
                inputs.insert(id, HostTensor::f32(computed.shape(), computed.values_f32()));
            }
            Storage::State => {}
            Storage::Device => unreachable!(),
        }
    }
    inputs
}

/// Mixtral's batched shared-pool decode tracer (dense `[E,...]`-stacked path, not the expert-pool one) must give each
/// active physical row the same logits as an independent sequential `trace_mixtral_decode_kv_masked` run over that
/// request's token stream, across a staggered admit/evict schedule (see `mixtral_staggered_schedule`). Pool
/// addressing is private per physical row (row r's position `t` -> pool slot `r*cap+t`, `pool_slots = n_slots*cap`):
/// the test targets per-row MoE-routing/attention math across admission, eviction and slot reuse, not packing density.
#[test]
fn mixtral_batched_shared_pool_decode_matches_sequential_staggered() {
    let cfg = Qwen2Config {
        vocab: 32,
        hidden: 16,
        inter: 24,
        layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        rotary_dim: 4,
        eps: 1e-6,
        max_pos: 32,
        qkv_bias: false,
        qk_norm: false,
        ..Default::default()
    };
    let mp = MixtralParams {
        n_experts: 4,
        top_k: 2,
        inter: 12,
    };
    let n_slots = 4usize;
    let cap = 8usize;
    // Private per-physical-row pool region (no cross-row sharing pressure; see the doc comment).
    let pool_slots = n_slots * cap;
    let reqs = mixtral_staggered_schedule();
    let max_step = reqs
        .iter()
        .map(|r| r.admit_step + r.tokens.len())
        .max()
        .unwrap();

    // --- Reference: for each request, its own sequential decode from a blank KV state over its own token stream. ---
    let mut ref_logits: Vec<Vec<Vec<f32>>> = reqs
        .iter()
        .map(|r| vec![Vec::new(); r.tokens.len()])
        .collect();
    for (ri, r) in reqs.iter().enumerate() {
        let g = trace_mixtral_decode_kv_masked(cfg, mp, cap);
        let mut caches = zero_state(&g);
        for (pos, &tok) in r.tokens.iter().enumerate() {
            let mut inputs = bind_seq_step(&g, tok, pos, cap);
            for (ci, &(sid, _)) in g.state.iter().enumerate() {
                inputs.insert(sid, caches[ci].clone());
            }
            let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
                let values: std::collections::HashMap<ValueId, Value> = inputs
                    .iter()
                    .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                    .collect();
                let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })()
            .expect("mixtral sequential decode eval");
            ref_logits[ri][pos] = logits.as_f32().unwrap().to_vec();
            caches = new_states;
        }
    }

    // --- Batched shared-pool trace: one graph across every engine step, admit/evict per `mixtral_staggered_schedule`; a free
    // physical row feeds `(token=0, pos=0)` and its output row is discarded (as `Runner::continuous_schedule`). ---
    let g = trace_mixtral_decode_kv_masked_batched_shared_pool(cfg, mp, cap, n_slots, pool_slots);
    g.validate()
        .expect("mixtral batched shared-pool decode graph should validate");
    let mut caches = zero_state(&g);
    let global_slotmap: Vec<i32> = (0..n_slots)
        .flat_map(|row| (0..cap).map(move |t| (row * cap + t) as i32))
        .collect();

    for t in 0..max_step {
        let mut tokens = vec![0u32; n_slots];
        let mut positions = vec![0usize; n_slots];
        for r in &reqs {
            if t < r.admit_step || t >= r.admit_step + r.tokens.len() {
                continue;
            }
            let local_pos = t - r.admit_step;
            tokens[r.slot] = r.tokens[local_pos];
            positions[r.slot] = local_pos;
        }
        let mut inputs =
            bind_batched_step_staggered(&g, &tokens, &positions, cap, n_slots, &global_slotmap);
        for (ci, &(sid, _)) in g.state.iter().enumerate() {
            inputs.insert(sid, caches[ci].clone());
        }
        let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: std::collections::HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .expect("mixtral batched decode eval");
        caches = new_states;

        for (ri, r) in reqs.iter().enumerate() {
            if t < r.admit_step || t >= r.admit_step + r.tokens.len() {
                continue;
            }
            let local_pos = t - r.admit_step;
            let got = &logits.as_f32().unwrap()[r.slot * cfg.vocab..(r.slot + 1) * cfg.vocab];
            let want = &ref_logits[ri][local_pos];
            let err = max_abs_error(got, want);
            assert!(
                err < 1e-4,
                "req {ri} slot {} engine step {t} local_pos {local_pos}: batched vs sequential mixtral \
                 logits max_abs={err:.3e} >= 1e-4",
                r.slot
            );
        }
    }
}
