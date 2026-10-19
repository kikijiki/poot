//! The incremental block-key cache against a full re-pool of the raw keys (POOT-1143 stage 1).
//!
//! Both graphs share the query, the scoring, the eligibility and the budgeted rank
//! ([`query`], [`select_blocks`]); they differ only in how the block keys are made. The reference
//! carries every raw key and pools the whole capacity each step (the indexer before the cache); the
//! cache carries block keys and the unfinished block's raw keys. On the CPU oracle the keys are the same
//! ops over the same inputs, so the masks (hence every query's selected set, ties included) and every
//! completed block key must be bitwise equal at every step of every plan.

use poot_graph_ir::rope_table::RopeFlavor;
use poot_graph_ir::{BinOp, Builder, Graph, RedOp, Scalar, Slot, TraceLimits, Traced};
use poot_quant::weights::{QsaRole, WeightId, WeightRole};

use super::*;
use crate::components::attention::carried_cache;
use crate::components::norm::NormKind;
use crate::components::standard::oracle::{Fx, random};
use crate::model::{LogitRows, StepShape};
use poot_tensor::DType;

/// The indexer's shape under test.
#[derive(Clone, Copy, Debug)]
struct Dims {
    heads: usize,
    dim: usize,
    width: usize,
    rotary: usize,
    ratio: usize,
    budget: usize,
}

/// A small indexer: blocks of four, two kept per query, so a query with three complete blocks drops one.
const SMALL: Dims = Dims {
    heads: 2,
    dim: 8,
    width: 6,
    rotary: 4,
    ratio: 4,
    budget: 8,
};

/// Flash-Next's indexer: four heads of 128, 512 blocks of four kept (`indexer_budget` 2048), a rotary
/// quarter-width pair set. Synthetic weights; the model's hidden width does not enter the block keys.
const FLASH_NEXT: Dims = Dims {
    heads: 4,
    dim: 128,
    width: 32,
    rotary: 64,
    ratio: 4,
    budget: 2048,
};

impl Dims {
    fn params(self) -> QsaParams {
        QsaParams::new(
            self.heads,
            self.dim,
            self.budget,
            self.ratio,
            RopeParams::new(self.rotary, 10_000.0, &RopeFlavor::Plain).unwrap(),
            NormParams::of(NormKind::RmsPlusOne, 1e-6).unwrap(),
        )
        .unwrap()
    }

    fn fixture(self) -> Fx {
        let mut fx = Fx::new(0x51a);
        let id = |role| WeightId::layer(0, WeightRole::Qsa(role));
        fx.f32(id(QsaRole::Q), &[self.heads * self.dim, self.width]);
        fx.f32(id(QsaRole::K), &[self.dim, self.width]);
        fx.f32(id(QsaRole::QNorm), &[self.dim]);
        fx.f32(id(QsaRole::KNorm), &[self.dim]);
        fx
    }

    fn weights(self, fx: &Fx) -> QsaWeights {
        QsaWeights::new(&fx.map(), 0, self.width, &self.params()).unwrap()
    }
}

/// The carried state as the oracle takes it.
fn views(state: &[Vec<f32>]) -> Vec<&[f32]> {
    state.iter().map(Vec::as_slice).collect()
}

fn shape(rows: usize, tokens: usize, cap: usize) -> StepShape {
    StepShape::new(
        rows.try_into().unwrap(),
        tokens.try_into().unwrap(),
        cap.try_into().unwrap(),
    )
    .with_logits(LogitRows::All)
}

/// The full re-pool of `keys` `[rows, 1, cap, dim]`: the whole capacity pooled, normed and rotated.
fn repool(
    b: &Builder,
    keys: Traced,
    w: &QsaWeights,
    d: Dims,
    cap: usize,
) -> Result<Traced, TraceError> {
    let rows = b.aval(keys)?.shape[0];
    let nb = cap / d.ratio;
    let summed = b.reduce(
        RedOp::Sum,
        b.reshape(keys, vec![rows, 1, nb, d.ratio, d.dim])?,
        3,
        true,
    )?;
    let pooled = b.reshape(
        b.binary_scalar(BinOp::Mul, summed, Scalar::F32(1.0 / d.ratio as f32))?,
        vec![rows, 1, nb, d.dim],
    )?;
    let starts = b.binary_scalar(BinOp::Mul, b.iota_i32(nb)?, Scalar::I32(d.ratio as i32))?;
    let block_pos = b.broadcast(b.reshape(starts, vec![1, nb])?, vec![rows, nb])?;
    let p = d.params();
    apply_rope(
        b,
        norm(b, pooled, &w.k_norm, p.norm)?,
        &rope_rows(b, p.rope, block_pos, cap)?,
    )
}

/// The reference's block keys: every raw key carried, the whole capacity re-pooled each step.
fn repooled_keys(
    b: &Builder,
    step: &Step,
    x: Traced,
    w: &QsaWeights,
    d: Dims,
) -> Result<Traced, TraceError> {
    let s = step.shape();
    let (rows, t, cap) = (s.rows.get(), s.tokens.get(), s.capacity.get());
    let raw = b.reshape(linear(b, x, &w.k, None)?, vec![rows, 1, t, d.dim])?;
    let write_at = first_writes(b, step)?;
    let keys = carried_cache(b, step, &write_at, 0, "index_reference", raw, 1, d.dim)?;
    repool(b, keys, w, d, cap)
}

/// The additive penalty of a position the selection drops.
const DROPPED: f32 = -1e30;

/// `(indicator - 1) * 1e30`: zero where the `{0, 1}` indicator is one, [`DROPPED`] where it is zero.
fn penalty(b: &Builder, indicator: Traced) -> Result<Traced, TraceError> {
    Ok(b.binary_scalar(
        BinOp::Mul,
        b.binary_scalar(BinOp::Sub, indicator, Scalar::F32(1.0))?,
        Scalar::F32(-DROPPED),
    )?)
}

/// Spread a per-block value `[rows, 1, tokens, blocks]` over the `ratio` positions of each block.
fn per_position(b: &Builder, blocks: Traced, ratio: usize) -> Result<Traced, TraceError> {
    let shape = b.aval(blocks)?.shape;
    let [rows, one, t, nb] = shape[..] else {
        panic!("per_position takes [rows, 1, tokens, blocks]");
    };
    let wide = b.broadcast(
        b.reshape(blocks, vec![rows, one, t, nb, 1])?,
        vec![rows, one, t, nb, ratio],
    )?;
    Ok(b.reshape(wide, vec![rows, one, t, nb * ratio])?)
}

/// `causal` plus [`DROPPED`] on every position outside `selected` (the penalty of each block the query keeps,
/// `[rows, 1, tokens, blocks]`) and outside the query's own unfinished block, which is always visible: the
/// dense mask the stage-1 attention read, kept as the reference for the ids the attention now reads.
fn with_tail(
    b: &Builder,
    step: &Step,
    selected: Traced,
    causal: Traced,
    p: &QsaParams,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let (ratio, nb) = (p.ratio, cap / p.ratio);
    let wide = vec![rows, 1, t, nb];
    let pos = b.broadcast(
        b.reshape(b.cast(step.pos(), DType::F32)?, vec![rows, 1, t, 1])?,
        wide.clone(),
    )?;
    let first = b.broadcast(
        b.reshape(
            b.binary_scalar(BinOp::Mul, b.iota(nb)?, Scalar::F32(ratio as f32))?,
            vec![1, 1, 1, nb],
        )?,
        wide,
    )?;
    let last = b.binary_scalar(BinOp::Add, first, Scalar::F32((ratio - 1) as f32))?;
    let complete = b.binary(BinOp::Ge, pos, last)?;
    let started = b.binary(BinOp::Ge, pos, first)?;
    let tail = penalty(b, b.binary(BinOp::Sub, started, complete)?)?;
    let selection = b.binary(
        BinOp::Max,
        per_position(b, selected, ratio)?,
        per_position(b, tail, ratio)?,
    )?;
    Ok(b.binary(BinOp::Add, selection, causal)?)
}

/// The op's selection as the dense mask: the keep indicator of the best blocks, then [`with_tail`].
fn op_selection_mask(
    b: &Builder,
    step: &Step,
    q: Traced,
    block_k: Traced,
    causal: Traced,
    p: &QsaParams,
) -> Result<Traced, TraceError> {
    let kept = select_blocks(b, step, q, block_k, p)?.keep;
    with_tail(b, step, penalty(b, b.cast(kept, DType::F32)?)?, causal, p)
}

/// The selection as it was before `OpKind::BlockTopK` (stage 1's chain), kept as the oracle for exact set
/// equality: eligibility as an additive penalty, a pairwise stable rank of the penalised scores, the keep mask
/// of the budget, and eligibility again, because a budget above the eligible count fills with ineligible
/// blocks.
fn legacy_selection_mask(
    b: &Builder,
    step: &Step,
    q: Traced,
    block_k: Traced,
    causal: Traced,
    p: &QsaParams,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let (ratio, nb) = (p.ratio, cap / p.ratio);
    let scores = block_scores(b, q, block_k, p.dim)?;
    let wide = vec![rows, 1, t, nb];
    let pos = b.broadcast(
        b.reshape(b.cast(step.pos(), DType::F32)?, vec![rows, 1, t, 1])?,
        wide.clone(),
    )?;
    let first = b.broadcast(
        b.reshape(
            b.binary_scalar(BinOp::Mul, b.iota(nb)?, Scalar::F32(ratio as f32))?,
            vec![1, 1, 1, nb],
        )?,
        wide,
    )?;
    let last = b.binary_scalar(BinOp::Add, first, Scalar::F32((ratio - 1) as f32))?;
    let eligible = penalty(b, b.binary(BinOp::Ge, pos, last)?)?;
    let rank = ops::stable_descending_rank(b, b.binary(BinOp::Add, scores, eligible)?)?;
    let kept = penalty(b, ops::top_k_keep_mask(b, rank, p.block_topk())?)?;
    with_tail(b, step, b.binary(BinOp::Add, eligible, kept)?, causal, p)
}

/// What a traced step outputs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Output {
    /// The cache's mask through the legacy selection chain ([`legacy_selection_mask`]), nothing else.
    LegacyMask,
    /// The cache's mask `[rows * tokens * capacity]` (its block keys are its carried state).
    CachedMask,
    /// The reference's mask followed by its re-pooled block keys.
    ReferenceMask,
    /// The cache's block keys alone, no selection.
    CachedKeys,
    /// The reference's re-pooled block keys alone, no selection.
    ReferenceKeys,
}

/// The traced step: `x` and `causal` are consts.
fn trace(fx: &Fx, d: Dims, s: StepShape, output: Output) -> Graph {
    let (rows, t, cap) = (s.rows.get(), s.tokens.get(), s.capacity.get());
    let nb = cap / d.ratio;
    let b = Builder::new(TraceLimits::for_tests());
    let step = Step::new(&b, s).unwrap();
    let p = d.params();
    let w = d.weights(fx);
    let x = b
        .constant("x", TensorType::f32(vec![rows, t, d.width]))
        .unwrap();
    let causal = b
        .constant("causal", TensorType::f32(vec![rows, 1, t, cap]))
        .unwrap();
    let flat = |v: Traced, len: usize| b.reshape(v, vec![len]).unwrap();
    let out = match output {
        Output::CachedMask => flat(
            {
                let q = query(&b, &step, x, &w, &p).unwrap();
                let keys = block_keys(&b, &step, 0, x, &w, &p).unwrap();
                op_selection_mask(&b, &step, q, keys, causal, &p).unwrap()
            },
            rows * t * cap,
        ),
        Output::LegacyMask => {
            let q = query(&b, &step, x, &w, &p).unwrap();
            let keys = block_keys(&b, &step, 0, x, &w, &p).unwrap();
            flat(
                legacy_selection_mask(&b, &step, q, keys, causal, &p).unwrap(),
                rows * t * cap,
            )
        }
        Output::CachedKeys => flat(
            block_keys(&b, &step, 0, x, &w, &p).unwrap(),
            rows * nb * d.dim,
        ),
        Output::ReferenceKeys => flat(
            repooled_keys(&b, &step, x, &w, d).unwrap(),
            rows * nb * d.dim,
        ),
        Output::ReferenceMask => {
            let q = query(&b, &step, x, &w, &p).unwrap();
            let keys = repooled_keys(&b, &step, x, &w, d).unwrap();
            let mask = op_selection_mask(&b, &step, q, keys, causal, &p).unwrap();
            b.concat(
                0,
                &[flat(mask, rows * t * cap), flat(keys, rows * nb * d.dim)],
            )
            .unwrap()
        }
    };
    step.finish(b, out).unwrap()
}

/// One step of a plan: `t` tokens, each row starting at its own position.
struct Chunk {
    t: usize,
    starts: Vec<usize>,
}

fn chunk(t: usize, starts: &[usize]) -> Chunk {
    Chunk {
        t,
        starts: starts.to_vec(),
    }
}

fn zero_state(g: &Graph) -> Vec<Vec<f32>> {
    g.state()
        .map(|(input, _)| vec![0.0; g.aval(input).shape.iter().product()])
        .collect()
}

/// `0` where the key position is at or before the query's, [`DROPPED`] after it.
fn causal_rows(pos: &[i32], cap: usize) -> Vec<f32> {
    pos.iter()
        .flat_map(|&q| (0..cap).map(move |j| if j as i32 <= q { 0.0 } else { DROPPED }))
        .collect()
}

/// Every element of `got` bitwise equal to `want`'s, `what` naming the first that is not.
fn assert_bitwise(label: &str, what: impl Fn(usize) -> String, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: element count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}: {}: got {g}, reference {w}",
            what(i)
        );
    }
}

/// Run `plan` through the cache and the reference with the same inputs; compare every step.
fn assert_plan_matches(label: &str, d: Dims, rows: usize, cap: usize, plan: &[Chunk]) {
    let fx = d.fixture();
    let nb = cap / d.ratio;
    // One input per (row, absolute position): a rewritten position gets the same key again.
    let inputs: Vec<Vec<f32>> = (0..rows)
        .map(|r| random(cap * d.width, 100 + r as u64))
        .collect();
    let (mut cache_state, mut ref_state) = (None, None);
    for (n, Chunk { t, starts }) in plan.iter().enumerate() {
        let s = shape(rows, *t, cap);
        let cached = trace(&fx, d, s, Output::CachedMask);
        let reference = trace(&fx, d, s, Output::ReferenceMask);
        let cache_in = cache_state.take().unwrap_or_else(|| zero_state(&cached));
        let ref_in = ref_state.take().unwrap_or_else(|| zero_state(&reference));
        let x: Vec<f32> = (0..rows)
            .flat_map(|r| inputs[r][starts[r] * d.width..(starts[r] + t) * d.width].to_vec())
            .collect();
        let pos: Vec<i32> = (0..rows)
            .flat_map(|r| (starts[r]..starts[r] + t).map(|p| p as i32))
            .collect();
        let tokens = vec![0i32; rows * t];
        let causal = causal_rows(&pos, cap);
        let consts: [(&str, &[f32]); 2] = [("x", &x), ("causal", &causal)];
        let slots: [(Slot, &[i32]); 2] = [(Slot::Token, &tokens), (Slot::Pos, &pos)];
        let (mask, next_cache) = fx.eval(&cached, &consts, &slots, &views(&cache_in));
        let (want, next_ref) = fx.eval(&reference, &consts, &slots, &views(&ref_in));
        let (want_mask, want_keys) = want.split_at(rows * t * cap);

        // Tier 3: every query's mask (so its selected set) is identical.
        assert_bitwise(
            &format!("{label}, step {n}"),
            |i| {
                format!(
                    "mask element {i} (row {}, query {}, position {})",
                    i / (t * cap),
                    i / cap % t,
                    i % cap
                )
            },
            &mask,
            want_mask,
        );
        // Tier 2: every completed block key is the re-pooled one, bit for bit. (A block the step
        // leaves unfinished holds a partial key the mask hides; it is not compared.)
        for (r, start) in starts.iter().enumerate() {
            let complete = (start + t) / d.ratio;
            let at = r * nb * d.dim..(r * nb + complete) * d.dim;
            assert_bitwise(
                &format!("{label}, step {n}, row {r}"),
                |i| format!("block {} key element {}", i / d.dim, i % d.dim),
                &next_cache[0][at.clone()],
                &want_keys[at],
            );
        }
        cache_state = Some(next_cache);
        ref_state = Some(next_ref);
    }
}

/// Run `plan` through the op's selection and the legacy chain's over the same carried block keys, and compare
/// every step's mask word for word: the exact top-k set equality of stage 2 on the oracle.
fn assert_legacy_selection_matches(label: &str, d: Dims, rows: usize, cap: usize, plan: &[Chunk]) {
    let fx = d.fixture();
    let inputs: Vec<Vec<f32>> = (0..rows)
        .map(|r| random(cap * d.width, 300 + r as u64))
        .collect();
    let (mut new_state, mut old_state) = (None, None);
    for (n, Chunk { t, starts }) in plan.iter().enumerate() {
        let s = shape(rows, *t, cap);
        let (new, old) = (
            trace(&fx, d, s, Output::CachedMask),
            trace(&fx, d, s, Output::LegacyMask),
        );
        let new_in = new_state.take().unwrap_or_else(|| zero_state(&new));
        let old_in = old_state.take().unwrap_or_else(|| zero_state(&old));
        let x: Vec<f32> = (0..rows)
            .flat_map(|r| inputs[r][starts[r] * d.width..(starts[r] + t) * d.width].to_vec())
            .collect();
        let pos: Vec<i32> = (0..rows)
            .flat_map(|r| (starts[r]..starts[r] + t).map(|p| p as i32))
            .collect();
        let causal = causal_rows(&pos, cap);
        let tokens = vec![0i32; rows * t];
        let consts: [(&str, &[f32]); 2] = [("x", &x), ("causal", &causal)];
        let slots: [(Slot, &[i32]); 2] = [(Slot::Token, &tokens), (Slot::Pos, &pos)];
        let (mask, next_new) = fx.eval(&new, &consts, &slots, &views(&new_in));
        let (want, next_old) = fx.eval(&old, &consts, &slots, &views(&old_in));
        assert_bitwise(
            &format!("{label}, step {n}"),
            |i| {
                format!(
                    "mask element {i} (row {}, query {}, position {})",
                    i / (t * cap),
                    i / cap % t,
                    i % cap
                )
            },
            &mask,
            &want,
        );
        new_state = Some(next_new);
        old_state = Some(next_old);
    }
}

/// SC-1143-9, the selection stage end to end on the oracle (design section 5, tier 3): the op's masks equal the
/// legacy rank chain's at every query of whole, chunked, token-at-a-time and two-row plans, over scores that
/// cross the budget (a query with three complete blocks drops one). Mutations: see
/// `block_top_k::tests` (tie placement, eligibility, NaN canonicalization); here, the budget one short
/// (`k: p.block_topk() - 1`) goes red at the first query past the budget.
#[test]
fn the_op_selects_what_the_legacy_rank_chain_selected() {
    let cap = 32;
    assert_legacy_selection_matches("whole prompt", SMALL, 1, cap, &[chunk(32, &[0])]);
    assert_legacy_selection_matches(
        "chunks",
        SMALL,
        1,
        cap,
        &[
            chunk(5, &[0]),
            chunk(3, &[5]),
            chunk(8, &[8]),
            chunk(16, &[16]),
        ],
    );
    assert_legacy_selection_matches(
        "token at a time",
        SMALL,
        1,
        cap,
        &(0..cap).map(|p| chunk(1, &[p])).collect::<Vec<_>>(),
    );
    assert_legacy_selection_matches(
        "two rows",
        SMALL,
        2,
        cap,
        &[chunk(9, &[0, 0]), chunk(7, &[9, 4]), chunk(16, &[16, 11])],
    );
}

/// SC-1143-1 (block keys) and SC-1143-3 (selection, per layer): one whole prompt, aligned chunks, a
/// token at a time, and chunks whose first position is off the block (a block started by one chunk and
/// finished by the next, two chunks and a decode inside one block). Mutations, each red:
/// the cache written one block early (`start - 1` as the write start); RoPE at a position past the
/// block's first; the tail slots not read (every span key taken from this step); the tail slots not
/// updated.
#[test]
fn the_cache_selects_exactly_what_a_full_repool_selects() {
    let cap = 32;
    assert_plan_matches("whole prompt", SMALL, 1, cap, &[chunk(32, &[0])]);
    assert_plan_matches(
        "aligned chunks",
        SMALL,
        1,
        cap,
        &[chunk(8, &[0]), chunk(8, &[8]), chunk(16, &[16])],
    );
    assert_plan_matches(
        "token at a time",
        SMALL,
        1,
        cap,
        &(0..cap).map(|p| chunk(1, &[p])).collect::<Vec<_>>(),
    );
    assert_plan_matches(
        "off-block chunks",
        SMALL,
        1,
        cap,
        &[
            chunk(5, &[0]),
            chunk(3, &[5]),
            chunk(7, &[8]),
            chunk(1, &[15]),
            chunk(16, &[16]),
        ],
    );
}

/// SC-1143-2: the step reaches the end of the capacity from inside a block, where the touched-block
/// write would run past the cache and slides left. Mutation: drop the slide (`start = first_block`);
/// the oracle refuses the out-of-range write.
#[test]
fn a_step_ending_at_the_capacity_keeps_every_block_in_place() {
    let cap = 16;
    assert_plan_matches(
        "ends at capacity",
        SMALL,
        1,
        cap,
        &[chunk(3, &[0]), chunk(13, &[3])],
    );
    assert_plan_matches(
        "one block",
        SMALL,
        1,
        SMALL.ratio,
        &[chunk(2, &[0]), chunk(2, &[2])],
    );
    assert_plan_matches(
        "last token",
        SMALL,
        1,
        cap,
        &[chunk(15, &[0]), chunk(1, &[15])],
    );
}

/// SC-1143-4: rows at different positions in one step, one of them rewriting positions it already
/// holds (a rollback to position 1): each row's tail and block keys follow its own first position.
/// Mutation: take row 0's first position for every row (`first_writes` read at index 0); row 1's
/// keys go red.
#[test]
fn rows_at_different_positions_keep_their_own_blocks() {
    assert_plan_matches(
        "two rows, one rolled back",
        SMALL,
        2,
        32,
        &[
            chunk(3, &[0, 0]),
            chunk(5, &[3, 1]),
            chunk(1, &[8, 6]),
            chunk(8, &[9, 7]),
        ],
    );
}

/// The state a request would carry after `depth` positions, built without running them: random raw
/// keys for positions `0..depth` stand for the history. The cache gets those keys pooled once by the
/// full re-pool (every block complete below `depth`, the block keys cache's invariant) and the raw
/// keys of the unfinished block in its tail slots; the reference gets the raw keys themselves. The step
/// under test then runs on both.
fn seeded(fx: &Fx, d: Dims, cap: usize, depth: usize) -> (Vec<f32>, Vec<f32>) {
    let nb = cap / d.ratio;
    let mut raw = random(depth * d.dim, 7);
    raw.resize(cap * d.dim, 0.0);
    let b = Builder::new(TraceLimits::for_tests());
    let w = d.weights(fx);
    let keys = b
        .constant("raw", TensorType::f32(vec![1, 1, cap, d.dim]))
        .unwrap();
    let out = repool(&b, keys, &w, d, cap).unwrap();
    let flat = b.reshape(out, vec![nb * d.dim]).unwrap();
    let g = b.finish(flat).unwrap();
    let (pooled, _) = fx.eval(&g, &[("raw", &raw)], &[], &[]);
    let mut blocks = pooled;
    // Only the complete blocks are cache state; the unfinished one is rebuilt from the tail.
    blocks[(depth / d.ratio) * d.dim..].fill(0.0);
    // Slot `j` holds the latest position below `depth` congruent to `j`.
    let tail: Vec<f32> = (0..d.ratio)
        .flat_map(|j| {
            (0..depth).rev().find(|p| p % d.ratio == j).map_or_else(
                || vec![0.0; d.dim],
                |p| raw[p * d.dim..(p + 1) * d.dim].to_vec(),
            )
        })
        .collect();
    // The reference's carried state is the raw cache; the cache's is the blocks then the tail.
    let cache_state = [blocks, tail].concat();
    (cache_state, raw)
}

/// One decode (or chunk) step of `t` tokens at `depth`, on Flash-Next's indexer over `cap` positions,
/// against the re-pooled reference. With `selection` the masks are compared as well as the completed
/// block keys; without it only the keys, which decide the selection, are (the rank chain of the
/// masks is a `blocks x blocks` pairwise tensor per query, which the oracle takes minutes to evaluate
/// past 8K and cannot hold at 128K).
fn assert_depth_matches(label: &str, cap: usize, depth: usize, t: usize, selection: bool) {
    let (d, fx) = (FLASH_NEXT, FLASH_NEXT.fixture());
    let nb = cap / d.ratio;
    let (seed_cache, seed_raw) = seeded(&fx, d, cap, depth);
    let s = shape(1, t, cap);
    let (cached, reference) = if selection {
        (Output::CachedMask, Output::ReferenceMask)
    } else {
        (Output::CachedKeys, Output::ReferenceKeys)
    };
    let (cached, reference) = (trace(&fx, d, s, cached), trace(&fx, d, s, reference));
    let split = nb * d.dim;
    let cache_in = vec![seed_cache[..split].to_vec(), seed_cache[split..].to_vec()];
    let x = random(t * d.width, 11);
    let pos: Vec<i32> = (depth..depth + t).map(|p| p as i32).collect();
    let tokens = vec![0i32; t];
    let causal = causal_rows(&pos, cap);
    let consts: [(&str, &[f32]); 2] = [("x", &x), ("causal", &causal)];
    let slots: [(Slot, &[i32]); 2] = [(Slot::Token, &tokens), (Slot::Pos, &pos)];
    let (got, next_cache) = fx.eval(&cached, &consts, &slots, &views(&cache_in));
    let (want, _) = fx.eval(&reference, &consts, &slots, &[&seed_raw]);
    let (got_keys, want_keys) = if selection {
        let (got_mask, want_mask) = (&got[..], &want[..t * cap]);
        assert_bitwise(
            label,
            |i| format!("mask element {i} (query {}, position {})", i / cap, i % cap),
            got_mask,
            want_mask,
        );
        // The selection is not trivially everything: past the budget a query drops positions.
        let kept = got_mask[..cap].iter().filter(|&&m| m == 0.0).count();
        if (depth + 1) / d.ratio > d.budget / d.ratio {
            assert!(kept < depth + 1, "{label}: the budget drops nothing");
        } else {
            assert_eq!(kept, depth + 1, "{label}: under the budget all is kept");
        }
        let legacy = trace(&fx, d, s, Output::LegacyMask);
        let (legacy_mask, _) = fx.eval(&legacy, &consts, &slots, &views(&cache_in));
        assert_bitwise(
            &format!("{label} against the legacy rank chain"),
            |i| format!("mask element {i} (query {}, position {})", i / cap, i % cap),
            got_mask,
            &legacy_mask,
        );
        (&next_cache[0][..], &want[t * cap..])
    } else {
        (&got[..], &want[..])
    };
    let complete = (depth + t) / d.ratio;
    assert_bitwise(
        label,
        |i| format!("block {} key element {}", i / d.dim, i % d.dim),
        &got_keys[..complete * d.dim],
        &want_keys[..complete * d.dim],
    );
}

/// SC-1143-7, the depth sweep on Flash-Next's indexer (4 heads of 128, budget 2048, ratio 4), selection
/// and keys against the full re-pool through the same selection: d0 (`cap` 2048: every complete block
/// fits the budget, so the selection is everything) and 8K (2048 blocks, 512 kept). Mutations: the ones
/// of the small plans, at depth.
#[test]
fn the_depth_sweep_selects_what_a_full_repool_selects_real_dims() {
    assert_depth_matches("d0", 2048, 2048 - 5, 1, true);
    assert_depth_matches("8K", 8192, 8192 - 5, 1, true);
}

/// The 32K row of the sweep with its selection (8192 blocks, 512 kept): the oracle evaluates the
/// 8192 x 8192 rank chain in about four minutes (release), so it is run by name, not in the gate. The keys at 32K are
/// in `the_block_keys_at_depth_are_the_repooled_ones_real_dims`.
#[test]
#[ignore = "about four minutes in a release build on the CPU oracle; run by name with --release: -E 'test(=the_32k_selection_matches_a_full_repool_real_dims)'"]
fn the_32k_selection_matches_a_full_repool_real_dims() {
    assert_depth_matches("32K", 32768, 32768 - 5, 1, true);
}

/// SC-1143-7 at the depths the oracle cannot rank: the block keys after a decode step at 32K and at
/// 128K (`cap` 131072, 32768 blocks), and an unaligned 8-token chunk at 8K. At 128K the rank chain is a
/// 32768 x 32768 pairwise tensor per query, so no device baseline exists (POOT-1143 ruling): the
/// baseline is the oracle's full re-pool, and the selection is a pure function of those keys and the
/// query. Mutation: the cache written one block early (`start - 1`); the keys of the block before the
/// step's go red.
#[test]
fn the_block_keys_at_depth_are_the_repooled_ones_real_dims() {
    assert_depth_matches("8K chunk", 8192, 8192 - 13, 8, false);
    assert_depth_matches("32K", 32768, 32768 - 5, 1, false);
    assert_depth_matches("128K", 131_072, 131_072 - 5, 1, false);
}
