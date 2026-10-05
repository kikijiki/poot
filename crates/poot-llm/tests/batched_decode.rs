//! Spec 027 sub-step 2 - the batched decode graph executor-equivalence oracle (SC-001 / FR-002).
//!
//! Trace the qwen2 family's decode step at `rows = B > 1` and B independent `rows = 1` steps, run them
//! over the SAME weights and, for each row, the SAME (token, pos, KV-cache) inputs, then assert each batched
//! row's logits equal the corresponding single-row decode (max|diff| <= 1e-5). This is a pure executor-
//! equivalence check on CPU eval: the batch axis must not leak across rows. No checkpoint, runs in
//! milliseconds.

mod common;

use common::{slot_maps, step, token_pos, zero_state};
use poot_executor_parity::dense::{Dense, Family, paged_step, step as step_shape};
use poot_executor_parity::weight_map::MappedModel;
use poot_models::model::{LogitRows, Phase};
use poot_tensor::HostTensor;

const VOCAB: usize = 32;
const BATCH: usize = 3;
const CAP: usize = 8;

/// Deterministic pseudo-random fill seeded by a name (so a given cache gets identical data in the
/// batched graph and every single-row graph) - an LCG over a cheap string hash, mapped to [-1, 1].
fn fill(name: &str, n: usize) -> Vec<f32> {
    let mut s = poot_test_util::seed_of(name);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// A tiny qwen2 (with qkv bias) so the eval is instant but exercises every block op.
fn tiny() -> MappedModel {
    Dense::new(Family::Qwen2)
        .vocab(VOCAB)
        .dims(16, 24, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(32)
        .f32_model()
}

/// A fixed per-row token stream: `tokens[step][row]`.
fn token_stream(n_steps: usize) -> Vec<Vec<i32>> {
    (0..n_steps)
        .map(|s| {
            (0..BATCH)
                .map(|r| (((s * BATCH + r) * 7 + 1) % VOCAB) as i32)
                .collect()
        })
        .collect()
}

/// Drive `rows = BATCH` decode from zero caches over `n_steps` (all rows advance one token each) on the
/// contiguous layout and on the paged layout `pooled` describes, and assert the logits agree.
///
/// `read(r, t)` is the pool slot of row `r`'s logical position `t`; a position past the step is never read
/// unmasked. The write map sends each pool slot to the flat token index writing it (or -1).
fn assert_paged_matches_contiguous(
    label: &str,
    pool_slots: usize,
    n_steps: usize,
    read: impl Fn(usize, usize) -> usize,
) {
    let m = tiny();
    let toks = token_stream(n_steps);
    let cont = m
        .model
        .trace(Phase::Decode, step_shape(BATCH, 1, CAP, LogitRows::Last))
        .unwrap();
    let paged = m
        .model
        .trace(
            Phase::Decode,
            paged_step(BATCH, 1, CAP, pool_slots, LogitRows::Last),
        )
        .unwrap();
    // The paged state is smaller than the per-row caches only when the pool is sized for live usage.
    let numel = |g: &poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs>| -> usize {
        g.state.iter().map(|&(si, _)| g.aval(si).numel()).sum()
    };
    if pool_slots < BATCH * CAP {
        assert!(
            numel(&paged) < numel(&cont),
            "shared pool ({}) must be smaller than per-row caches ({})",
            numel(&paged),
            numel(&cont)
        );
    }

    let mut c_state = zero_state(&cont);
    let mut p_state = zero_state(&paged);
    for (s, ids) in toks.iter().enumerate() {
        let pos = vec![s as i32; BATCH];
        let slots = token_pos(BATCH, 1, ids, &pos);
        let (cl, next) = step(&m, &cont, &slots, &c_state);
        c_state = next;

        let read_map: Vec<i32> = (0..BATCH)
            .flat_map(|r| (0..CAP).map(move |t| (r, t)))
            .map(|(r, t)| read(r, t) as i32)
            .collect();
        let mut write = vec![-1i32; pool_slots];
        for r in 0..BATCH {
            write[read(r, s)] = r as i32;
        }
        let mut paged_slots = token_pos(BATCH, 1, ids, &pos);
        paged_slots.extend(slot_maps(BATCH, CAP, pool_slots, read_map, write));
        let (pl, next) = step(&m, &paged, &paged_slots, &p_state);
        p_state = next;

        assert_eq!(cl.shape(), pl.shape());
        for (j, (a, b)) in cl
            .as_f32()
            .unwrap()
            .iter()
            .zip(pl.as_f32().unwrap())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-4,
                "batched {label} != contiguous at step {s}, logit {j}: {a} vs {b}"
            );
        }
    }
}

/// Spec 045 / card 046b: the batched paged decode must produce logits identical to the batched contiguous
/// decode, with each row using its own (non-identity) region of the pool and its positions laid out in
/// reverse, so the per-row paged write + gather is math-preserving. Drives both from zero caches over
/// several steps. Tiny synthetic config, CPU eval.
#[test]
fn batched_paged_matches_batched_contiguous() {
    assert_paged_matches_contiguous("paged", BATCH * CAP, 5, |r, t| r * CAP + (CAP - 1 - t));
}

/// 046b shared-pool oracle: all rows share one pool of `pool_slots` slots sized for the aggregate live usage
/// (`pool_slots = 15 < B*cap = 24`), with rows interleaved in the pool (slot for (row r, pos t) = t*B + r),
/// a shared, non-block-diagonal layout, so one pool sized for aggregate usage is math-preserving. Positions
/// past the live prefix point at slot 0 (in range, masked out).
#[test]
fn batched_shared_pool_matches_batched_contiguous() {
    let n_steps = 5;
    assert_paged_matches_contiguous("shared-pool", BATCH * n_steps, n_steps, |r, t| {
        if t < n_steps { t * BATCH + r } else { 0 }
    });
}

#[test]
fn batched_decode_rows_equal_batch1_decodes() {
    let m = tiny();
    let (hkv, d) = (2, 4);
    let tokens = [3, 7, 1];
    let pos = [2, 5, 0];
    let gb = m
        .model
        .trace(Phase::Decode, step_shape(BATCH, 1, CAP, LogitRows::Last))
        .unwrap();
    let g1 = m
        .model
        .trace(Phase::Decode, step_shape(1, 1, CAP, LogitRows::Last))
        .unwrap();
    assert_eq!(gb.state.len(), g1.state.len());

    // random per-row KV caches [B, hkv, cap, d], one per state input, in state order.
    let caches: Vec<HostTensor> = gb
        .state
        .iter()
        .enumerate()
        .map(|(i, &(si, _))| {
            let shape = gb.aval(si).shape.clone();
            assert_eq!(shape, vec![BATCH, hkv, CAP, d]);
            HostTensor::f32(shape, fill(&format!("cache{i}"), BATCH * hkv * CAP * d))
        })
        .collect();
    let (logits_b, _) = step(&m, &gb, &token_pos(BATCH, 1, &tokens, &pos), &caches);
    assert_eq!(logits_b.shape(), vec![BATCH, 1, VOCAB]);

    let stride = hkv * CAP * d;
    for b in 0..BATCH {
        let row_caches: Vec<HostTensor> = caches
            .iter()
            .map(|c| {
                HostTensor::f32(
                    vec![1, hkv, CAP, d],
                    c.as_f32().unwrap()[b * stride..(b + 1) * stride].to_vec(),
                )
            })
            .collect();
        let (logits_1, _) = step(
            &m,
            &g1,
            &token_pos(1, 1, &[tokens[b]], &[pos[b]]),
            &row_caches,
        );
        assert_eq!(logits_1.shape(), vec![1, 1, VOCAB]);

        let row = &logits_b.as_f32().unwrap()[b * VOCAB..(b + 1) * VOCAB];
        // Row `b` of the batched graph differs from the batch-1 graph if this panics.
        poot_test_util::assert_close(row, logits_1.as_f32().unwrap(), 1e-5);
    }
}
