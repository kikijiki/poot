//! Spec 029 sub-step 2: checks the prompt-lookup verify forward against sequential decode.
//! A verify is an L-token prefill step at runtime offset `pos` against a carried KV cache; its
//! per-position logits must equal the decode step run one token at a time at positions
//! `pos..pos+L` (carrying the cache). Tiny synthetic config with deterministic random weights, CPU
//! eval: no model, no GPU.

mod common;

use common::{slot_maps, step, token_pos, zero_state};
use poot_executor_parity::dense::{Dense, Family, paged_step, plain, step as step_shape};
use poot_executor_parity::weight_map::MappedModel;
use poot_graph_ir::{Graph, ValidationOutputs};
use poot_graph_plan::passes_without_target as optimize;
use poot_models::model::{LogitRows, Phase};
use poot_tensor::HostTensor;

const VOCAB: usize = 32;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

fn tiny() -> MappedModel {
    Dense::new(Family::Qwen2)
        .vocab(VOCAB)
        .dims(16, 24, 2)
        .heads(4, 4)
        .head_dim(4)
        .max_positions(32)
        .f32_model()
}

/// The one-token decode graph over `cap` positions.
fn decode_graph(m: &MappedModel, cap: usize) -> Graph<ValidationOutputs> {
    m.model
        .trace(Phase::Decode, step_shape(1, 1, cap, LogitRows::Last))
        .unwrap()
}

/// The verify graph: an `l`-token query with logits at every row, over `cap` positions.
fn verify_graph(m: &MappedModel, l: usize, cap: usize) -> Graph<ValidationOutputs> {
    m.model
        .trace(Phase::Prefill, step_shape(1, l, cap, LogitRows::All))
        .unwrap()
}

/// One decode step of `tok` at `pos` over carried `caches`.
fn decode(
    m: &MappedModel,
    g: &Graph<ValidationOutputs>,
    caches: &[HostTensor],
    tok: i32,
    pos: usize,
) -> (HostTensor, Vec<HostTensor>) {
    step(m, g, &token_pos(1, 1, &[tok], &[pos as i32]), caches)
}

/// One verify step of the `toks` block at absolute `pos`.
fn verify(
    m: &MappedModel,
    g: &Graph<ValidationOutputs>,
    caches: &[HostTensor],
    toks: &[i32],
    pos: usize,
) -> HostTensor {
    let positions: Vec<i32> = (0..toks.len()).map(|i| (pos + i) as i32).collect();
    step(m, g, &token_pos(1, toks.len(), toks, &positions), caches).0
}

/// Spec 045 (paged KV, card 046a): the paged decode graph must produce token-identical logits to
/// the contiguous fixed-KV decode for any consistent slot map (FR-001). The slot map is a
/// non-identity (reverse) layout, so the reorder is exercised.
#[test]
fn paged_decode_matches_contiguous_decode_cpu() {
    let m = tiny();
    let cap = 20;
    let n_steps = 12; // < cap
    let cont = decode_graph(&m, cap);
    let paged = m
        .model
        .trace(Phase::Decode, paged_step(1, 1, cap, cap, LogitRows::Last))
        .unwrap();
    let read: Vec<i32> = (0..cap as i32).map(|t| cap as i32 - 1 - t).collect();

    let mut c_caches = zero_state(&cont);
    let mut p_caches = zero_state(&paged);

    // A fixed pseudo-random token stream fed to both graphs (independent of greedy choice).
    let toks: Vec<i32> = (0..n_steps).map(|i| ((i * 7 + 3) % VOCAB) as i32).collect();

    for (pos, &tok) in toks.iter().enumerate() {
        let (clog, cnew) = decode(&m, &cont, &c_caches, tok, pos);
        c_caches = cnew;

        let mut write = vec![-1i32; cap];
        write[read[pos] as usize] = 0;
        let mut slots = token_pos(1, 1, &[tok], &[pos as i32]);
        slots.extend(slot_maps(1, cap, cap, read.clone(), write));
        let (plog, pnew) = step(&m, &paged, &slots, &p_caches);
        p_caches = pnew;

        assert_eq!(clog.as_f32().unwrap().len(), plog.as_f32().unwrap().len());
        for (j, (a, b)) in clog
            .as_f32()
            .unwrap()
            .iter()
            .zip(plog.as_f32().unwrap())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-4,
                "paged != contiguous at pos {pos}, logit {j}: {a} vs {b}"
            );
        }
        // Greedy choice agrees too.
        assert_eq!(
            argmax(clog.as_f32().unwrap()),
            argmax(plog.as_f32().unwrap()),
            "paged greedy token != contiguous at pos {pos}"
        );
    }
}

/// Decode `toks` one token at a time from empty caches: every position's logits, and the cache after the
/// first `snapshot` positions were written.
fn sequential(
    m: &MappedModel,
    dg: &Graph<ValidationOutputs>,
    toks: &[i32],
    snapshot: usize,
) -> (Vec<Vec<f32>>, Vec<HostTensor>) {
    let mut caches = zero_state(dg);
    let mut logits = Vec::new();
    let mut taken = caches.clone();
    for (pos, &tok) in toks.iter().enumerate() {
        let (l, new) = decode(m, dg, &caches, tok, pos);
        caches = new;
        logits.push(l.as_f32().unwrap().to_vec());
        if pos + 1 == snapshot {
            taken = caches.clone();
        }
    }
    (logits, taken)
}

fn row(logits: &HostTensor, i: usize) -> &[f32] {
    &logits.as_f32().unwrap()[i * VOCAB..(i + 1) * VOCAB]
}

/// A verify whose first query token re-feeds an already-cached position.
/// `generate_prompt_lookup` verifies at `pos = m-1` with query `[tokens[m-1], drafts..]`, so
/// position `m-1` is rewritten. The re-feed must be idempotent: verify logits at the overlapping
/// and new positions must equal sequential decode. A failure is a re-feed/DUS bug, not the loop.
#[test]
fn verify_forward_with_refed_position_matches_sequential_decode() {
    let m = tiny();
    // Prefix [0,p) written by decode, then position p too (cache holds [0,p]); verify at pos=p
    // re-feeds position p (query[0]) and writes [p, p+l): the loop's overlap by one.
    let (p, l, cap) = (3usize, 4usize, 8usize);
    let toks: Vec<i32> = vec![5, 11, 2, 9, 14, 3, 7]; // need tokens through pos p+l-1 = 6

    let dg = decode_graph(&m, cap);
    let vg = verify_graph(&m, l, cap);
    // Cache after writing positions [0,p] (one more position than the sub-step-2 test).
    let (ref_logits, overlap_caches) = sequential(&m, &dg, &toks[..p + l], p + 1);

    let vlogits = verify(&m, &vg, &overlap_caches, &toks[p..p + l], p);
    assert_eq!(vlogits.shape(), vec![1, l, VOCAB]);

    for i in 0..l {
        // Verify position `i` (re-fed) differs from sequential decode at pos `p + i` if this panics.
        poot_test_util::assert_close(row(&vlogits, i), &ref_logits[p + i], 1e-5);
    }
}

/// Spec 052 cache-carried draft validation: a draft with its own carried KV and per-round
/// rollback must propose the same k tokens as a fresh full re-prefill of the confirmed prefix,
/// every round, across rollbacks of varying depth. A speculative target's output is correct for
/// any draft (bad proposals are rejected), so `tokens == greedy` does not test the draft; only
/// cache-carried proposals == re-prefill proposals does. Rollbacks are forced (accept
/// `a = round % (k+1)`, then a fixed bonus) so depths 0..k are all exercised. CPU eval.
#[test]
fn cache_carried_draft_proposals_match_reprefill_cpu() {
    let m = tiny();
    let k = 3usize;
    let rounds = 6usize; // a = 0,1,2,3,0,1 -> every rollback depth in 0..=k is exercised
    let prompt: Vec<i32> = vec![5, 11, 2, 9, 5, 11];
    // KV-buffer size. Rope is indexed by absolute position, so positions (not cap) must stay
    // < max_pos; the loop asserts that each round.
    let cap = 32;

    let dg = decode_graph(&m, cap);
    let step_of = |caches: &[HostTensor], tok: i32, pos: usize| decode(&m, &dg, caches, tok, pos);
    // Re-prefill reference: fresh KV, prefill `confirmed`, then greedily propose k tokens.
    let reprefill_propose = |confirmed: &[i32]| -> Vec<i32> {
        let mut caches = zero_state(&dg);
        let mut last = HostTensor::scalar(0.0);
        for (pos, &tok) in confirmed.iter().enumerate() {
            let (logits, new) = step_of(&caches, tok, pos);
            caches = new;
            if pos + 1 == confirmed.len() {
                last = logits;
            }
        }
        let len = confirmed.len();
        let mut drafts = Vec::with_capacity(k);
        for i in 0..k {
            let di = argmax(last.as_f32().unwrap()) as i32;
            drafts.push(di);
            let (logits, new) = step_of(&caches, di, len + i);
            caches = new;
            last = logits;
        }
        drafts
    };

    // Cache-carried draft: prefill the prompt once, then carry KV and running logits across rounds.
    let mut confirmed = prompt.clone();
    let mut dcaches = zero_state(&dg);
    let mut draft_last = HostTensor::scalar(0.0);
    for (pos, &tok) in prompt.iter().enumerate() {
        let (logits, new) = step_of(&dcaches, tok, pos);
        dcaches = new;
        if pos + 1 == prompt.len() {
            draft_last = logits;
        }
    }

    let mut rollbacks = 0usize;
    for round in 0..rounds {
        let len = confirmed.len();
        assert!(
            len + k <= cap,
            "round {round}: confirmed {len} + k {k} exceeds the cache ({cap})"
        );
        // Cache-carried propose: k tokens from the carried KV (advances dcaches to len+k).
        let mut cc = Vec::with_capacity(k);
        for i in 0..k {
            let di = argmax(draft_last.as_f32().unwrap()) as i32;
            cc.push(di);
            let (logits, new) = step_of(&dcaches, di, len + i);
            dcaches = new;
            draft_last = logits;
        }
        // Must equal a fresh re-prefill of the same confirmed prefix.
        let reff = reprefill_propose(&confirmed);
        assert_eq!(
            cc, reff,
            "round {round}: cache-carried proposals differ from re-prefill"
        );

        // Force a rollback of depth a (accept a of k), then a fixed bonus independent of the draft.
        let a = round % (k + 1);
        if a < k {
            rollbacks += 1;
        }
        for &t in cc.iter().take(a) {
            confirmed.push(t);
        }
        let bonus = ((len * 7 + 3) % VOCAB) as i32;
        confirmed.push(bonus);
        // Rollback and carry: feed the bonus at pos len+a (overwrites the rejected cc[a..] in the
        // fixed-cap KV); draft_last then predicts the next round's first token.
        let (logits, new) = step_of(&dcaches, bonus, len + a);
        dcaches = new;
        draft_last = logits;
    }
    assert!(rollbacks > 0, "no rollback was exercised");
}

/// Card 094 bug 2: causal invariance of the verify step under random nonzero weights. Row `i` of an
/// L-token verify query must depend only on query rows `0..=i` (plus the prefix KV cache); a
/// later draft token must never change an earlier row's logits. Two verify calls sharing a prefix
/// `t0` (A = `[t0,a1,a2,a3]`, B = `[t0,b1,b2,b3]`) must give identical row-0 logits, and generally
/// row i must match whenever A and B agree on rows `0..=i`. Per-row logits are also compared to a
/// sequential decode reference, so a failure names the first row that leaks.
#[test]
fn verify_forward_is_causally_invariant_to_later_draft_tokens() {
    let m = tiny();
    let (p, l, cap) = (3usize, 4usize, 12usize);
    let t0 = 5;
    let prefix: Vec<i32> = vec![5, 11, 2]; // p tokens, positions 0..p

    let dg = decode_graph(&m, cap);
    let vg = verify_graph(&m, l, cap);

    // The prefix KV cache [0,p), built sequentially with the decode graph.
    let (_, prefix_caches) = sequential(&m, &dg, &prefix, p);

    // Two verify queries sharing t0 but diverging at every later position.
    let query_a: Vec<i32> = vec![t0, 13, 4, 21];
    let query_b: Vec<i32> = vec![t0, 27, 17, 8];
    assert_eq!(query_a[0], query_b[0]);
    assert!((1..l).all(|i| query_a[i] != query_b[i]));

    let logits_a = verify(&m, &vg, &prefix_caches, &query_a, p);
    let logits_b = verify(&m, &vg, &prefix_caches, &query_b, p);
    assert_eq!(logits_a.shape(), vec![1, l, VOCAB]);
    assert_eq!(logits_b.shape(), vec![1, l, VOCAB]);

    // Row 0 (sees only t0 + prefix) must be identical between A and B.
    // Row 0 leaks later draft tokens (A differs from B) if this panics.
    poot_test_util::assert_close(row(&logits_a, 0), row(&logits_b, 0), 1e-5);

    // C = A with only the last position changed; every row except the last must match A.
    let mut query_c = query_a.clone();
    query_c[l - 1] = query_b[l - 1];
    let logits_c = verify(&m, &vg, &prefix_caches, &query_c, p);
    for i in 0..l - 1 {
        // Row `i` leaks the later (last-position) draft token if this panics.
        poot_test_util::assert_close(row(&logits_a, i), row(&logits_c, i), 1e-5);
    }

    // Row i of query A must equal a sequential decode of [t0,a1,..,ai] at pos p+i from the same
    // prefix cache; pins which row first diverges.
    let mut seq_caches = prefix_caches.clone();
    for (i, &tok) in query_a.iter().enumerate() {
        let (seq_logits, new) = decode(&m, &dg, &seq_caches, tok, p + i);
        seq_caches = new;
        // Verify row `i` differs from sequential decode at pos `p + i` if this panics.
        poot_test_util::assert_close(row(&logits_a, i), seq_logits.as_f32().unwrap(), 1e-5);
    }
}

/// Card 094 bug 2 root cause: the verify step's raw decomposition (`ops::attention_masked`) is
/// causally correct (`verify_forward_is_causally_invariant_to_later_draft_tokens`), but serving
/// lowers through `compile`'s pass pipeline, which includes `flash_attention_capped`.
/// That pass classified any multi-row (`L>1`) query as `FlashAttentionPrefill`, whose op
/// (`ops::attention_prefill` and its eval/GPU kernels) assumes K/V length == Q's `L` with a square
/// `[L,L]` mask. The verify step's K/V is the full `[.,.,cap,.]` carried cache (`cap > L` once a
/// prefix exists) with a `[L,cap]` mask: same op-chain shape, different semantics, so it attended
/// only the first `L` cache slots and corrupted every row (including row 0) once any prefix
/// existed. Fixed in `match_attention` (transform.rs) by requiring K's sequence axis to equal Q's
/// `L`; a longer K falls back to the decomposition. This test pins that `optimize(verify graph)`
/// computes the raw graph's logits.
#[test]
fn optimize_preserves_verify_forward_semantics() {
    let m = tiny();
    let (p, l, cap) = (3usize, 4usize, 12usize);
    let dg = decode_graph(&m, cap);
    let vg = plain(verify_graph(&m, l, cap));
    let vg_opt = optimize(&vg);

    // A non-empty prefix cache (cap=12 > l=4) exposes the bug; an empty-cache verify at pos=0 can
    // have cap==l and would not.
    let prefix: Vec<i32> = vec![5, 11, 2];
    let (_, caches) = sequential(&m, &dg, &prefix, p);

    let query: Vec<i32> = vec![9, 13, 4, 21];
    let positions: Vec<i32> = (0..l).map(|i| (p + i) as i32).collect();
    let slots = token_pos(1, l, &query, &positions);
    let (logits_raw, _) =
        poot_test_util::weight_map_oracle::eval_mapped(&vg, &m.store, &m.map, &slots, &caches);
    let (logits_opt, _) =
        poot_test_util::weight_map_oracle::eval_mapped(&vg_opt, &m.store, &m.map, &slots, &caches);

    assert_eq!(logits_raw.shape(), vec![1, l, VOCAB]);
    assert_eq!(logits_opt.shape(), vec![1, l, VOCAB]);
    // A panic here means optimize() changes verify semantics.
    poot_test_util::assert_close(
        logits_raw.as_f32().unwrap(),
        logits_opt.as_f32().unwrap(),
        1e-4,
    );
}

#[test]
fn verify_forward_matches_sequential_decode() {
    let m = tiny();
    let (p, l, cap) = (3usize, 3usize, 8usize); // 3-token prefix, verify 3 tokens at pos=3
    let toks: Vec<i32> = vec![5, 11, 2, 9, 14, 3]; // p + l tokens (< vocab)

    let dg = decode_graph(&m, cap);
    let vg = verify_graph(&m, l, cap);

    // Sequential reference: empty caches, decode tokens 0..p+l one at a time carrying the cache;
    // record each position's logits and snapshot the cache after the prefix [0,p).
    let (ref_logits, prefix_caches) = sequential(&m, &dg, &toks, p);

    // Verify forward: the L-token query at pos=p against the prefix cache in one shot.
    let vlogits = verify(&m, &vg, &prefix_caches, &toks[p..p + l], p);
    assert_eq!(vlogits.shape(), vec![1, l, VOCAB]);

    for i in 0..l {
        // Verify position `i` differs from sequential decode at pos `p + i` if this panics.
        poot_test_util::assert_close(row(&vlogits, i), &ref_logits[p + i], 1e-5);
    }
}
