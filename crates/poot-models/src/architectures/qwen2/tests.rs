use super::*;
use poot_graph_ir::{Graph, OpKind, Storage, ValidationOutputs};
use std::num::NonZeroUsize;

use crate::model::{KvLayout, LogitRows, Model, Phase, StepShape};
use crate::registry::Registry;

/// The registry's `family` model over its own fixture checkpoint, its config adjusted by `tweak`
/// (the checkpoint's tensor shapes do not depend on the fields these tests change).
fn model(family: &str, tweak: impl FnOnce(&mut serde_json::Value)) -> Box<dyn Model> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|e| e.family.as_str() == family)
        .unwrap_or_else(|| panic!("no family {family}"));
    let mut fixture = (entry.fixture)();
    tweak(&mut fixture.config);
    (entry.build)(&fixture.raw(), &fixture.store).unwrap()
}

fn fixture_model(family: &str) -> Box<dyn Model> {
    model(family, |_| {})
}

fn shape(
    rows: usize,
    tokens: usize,
    capacity: usize,
    kv: KvLayout,
    logits: LogitRows,
) -> StepShape {
    let nz = |n| NonZeroUsize::new(n).unwrap();
    StepShape {
        rows: nz(rows),
        tokens: nz(tokens),
        capacity: nz(capacity),
        kv,
        logits,
    }
}

fn contiguous(rows: usize, tokens: usize, capacity: usize) -> StepShape {
    shape(
        rows,
        tokens,
        capacity,
        KvLayout::Contiguous,
        LogitRows::Last,
    )
}

/// Config facts the structural rows read back: `(layers, kv_heads, head_dim, vocab)`.
fn dims(model: &dyn Model, family: &str) -> (usize, usize, usize, usize) {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|e| e.family.as_str() == family)
        .unwrap();
    let config = (entry.fixture)().config;
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let head_dim = config["head_dim"]
        .as_u64()
        .map_or(n("hidden_size") / n("num_attention_heads"), |d| d as usize);
    (
        n("num_hidden_layers"),
        n("num_key_value_heads"),
        head_dim,
        model.config().vocab,
    )
}

/// A traced step carries one K and one V cache per layer, each `[rows, kv_heads, cap, head_dim]`,
/// and yields `[rows, 1, vocab]` logits.
fn assert_contiguous_state(g: &Graph<ValidationOutputs>, family: &str, rows: usize, cap: usize) {
    let m = fixture_model(family);
    let (layers, kv_heads, head_dim, vocab) = dims(&*m, family);
    g.validate().expect("the traced graph validates");
    assert_eq!(g.aval(g.output).shape, vec![rows, 1, vocab]);
    assert_eq!(g.state.len(), 2 * layers);
    for (si, _so) in &g.state {
        assert_eq!(g.aval(*si).shape, vec![rows, kv_heads, cap, head_dim]);
        assert_eq!(g.values[*si].storage, Storage::State);
    }
}

#[test]
fn decode_traces_and_validates() {
    let m = fixture_model("qwen2");
    let g = m.trace(Phase::Decode, contiguous(1, 1, 16)).unwrap();
    // structural well-formedness: every operand defined before use, stored avals match infer.
    g.validate().expect("graph should validate");
    assert!(g.eqns.len() > 100);
    assert_eq!(g.aval(g.output).shape, vec![1, 1, 48]);
}

#[test]
fn prefill_traces_and_validates_both_archs() {
    // qwen2 (bias, no QK-norm) and qwen3 (no bias, QK-norm) are two registry families.
    for family in ["qwen2", "qwen3"] {
        let m = fixture_model(family);
        let g = m.trace(Phase::Prefill, contiguous(1, 5, 8)).unwrap();
        g.validate().expect("prefill graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, m.config().vocab]);
    }
}

#[test]
fn gemma3_decode_traces_with_state_pairs() {
    let g = fixture_model("gemma3")
        .trace(Phase::Decode, contiguous(1, 1, 16))
        .unwrap();
    assert_contiguous_state(&g, "gemma3", 1, 16);
}

#[test]
fn gemma3_prefill_traces_with_state_pairs() {
    // the batched prefill must carry the SAME state pairs as the decode graph.
    let g = fixture_model("gemma3")
        .trace(Phase::Prefill, contiguous(1, 5, 16))
        .unwrap();
    assert_contiguous_state(&g, "gemma3", 1, 16);
}

/// Gemma 3 must flash-fuse its prefill under `optimize()` like plain qwen2 (card 189);
/// `optimize_fuses_flash_prefill_no_materialized_scores` covers qwen2. Gemma 3 reuses
/// `attention_prefill` with a scalar `attn_scale` and a per-layer local/global rope split; the
/// sliding-window mask only changes mask values, so the fixture's window is representative.
/// Graph-transform-only gate.
#[test]
fn optimize_fuses_flash_prefill_gemma_no_materialized_scores() {
    assert_flash_fused_prefill("gemma3", KvLayout::Contiguous);
}

#[test]
fn decode_traces_with_state_pairs() {
    let g = fixture_model("qwen2")
        .trace(Phase::Decode, contiguous(1, 1, 16))
        .unwrap();
    assert_contiguous_state(&g, "qwen2", 1, 16);
}

#[test]
fn rope_fusion_collapses_every_rope_in_the_decode_graph() {
    use poot_graph_ir::op::UnOp;
    use poot_graph_plan::{cse, dce, rope_fusion};
    // On the registry's qwen2 decode graph, rope_fusion must collapse the rotate-half chain for both
    // the q and k rope in every layer to one `Rope` op each and remove every rotate-half `Neg` (the
    // dispatch-count lever). The dense components once emitted `x * -1` for the rotate-half sign,
    // which the matcher does not read, so no rope fused on any registry body.
    let m = fixture_model("qwen2");
    let (layers, _, head_dim, _) = dims(&*m, "qwen2");
    let g = m.trace(Phase::Decode, contiguous(1, 1, 16)).unwrap();
    let base = dce(&cse(&g));
    let fused = dce(&rope_fusion(&cse(&g)));
    fused.validate().expect("fused decode graph valid");
    let ropes: Vec<usize> = fused
        .eqns
        .iter()
        .filter_map(|e| match e.op {
            OpKind::Rope { rot } => Some(rot),
            _ => None,
        })
        .collect();
    assert_eq!(ropes.len(), 2 * layers, "one fused Rope per q/k per layer");
    // qwen2 is full-rotary, so every Rope covers the whole head dim.
    assert!(
        ropes.iter().all(|&rot| rot == head_dim),
        "full-rotary: rot == head_dim"
    );
    // Other ops negate too (`silu` is `x * exp(-x)`); only the rotate-half Negs, one per rope, go.
    let count_negs = |g: &Graph<ValidationOutputs>| {
        g.eqns
            .iter()
            .filter(|e| matches!(e.op, OpKind::Unary(UnOp::Neg)))
            .count()
    };
    assert_eq!(
        count_negs(&base) - count_negs(&fused),
        ropes.len(),
        "every rotate-half Neg (one per rope) is fused away"
    );
    // Every op but a `Reshape` alias is one dispatch (`poot_graph_ir::analysis::dispatch_count`, which
    // takes a graph without a validation channel).
    let dispatch_count = |g: &Graph<ValidationOutputs>| {
        g.eqns
            .iter()
            .filter(|e| !matches!(e.op, OpKind::Reshape { .. }))
            .count()
    };
    let (before, after) = (dispatch_count(&base), dispatch_count(&fused));
    assert!(
        after < before && before - after >= 5 * ropes.len(),
        "rope fusion cuts >=5 dispatches per rope: {before} -> {after} ({} ropes)",
        ropes.len()
    );
}

#[test]
fn collapse_reshape_chains_shrinks_the_decode_graph() {
    use poot_graph_plan::{collapse_reshape_chains, elide_noop_transposes};
    // Eliding the seq-1 attention transposes exposes reshape->reshape runs (q/k/v reshape then
    // no-op transpose-as-reshape) in every layer; collapse_reshape_chains must remove some and
    // keep the graph valid (value-equivalence is covered by poot-eval's
    // reshape_collapse_is_value_identical and the fusion-equivalence fuzzers).
    let g = fixture_model("qwen2")
        .trace(Phase::Decode, contiguous(1, 1, 16))
        .unwrap();
    let elided = elide_noop_transposes(&g);
    let collapsed = collapse_reshape_chains(&elided);
    collapsed.validate().expect("collapsed decode graph valid");
    assert!(
        collapsed.eqns.len() < elided.eqns.len(),
        "collapse should shrink the decode graph: {} -> {} eqns",
        elided.eqns.len(),
        collapsed.eqns.len()
    );
    assert_eq!(
        collapsed.aval(collapsed.output).shape,
        vec![1, 1, 48],
        "output unchanged"
    );
}

#[test]
fn prefill_traces_with_state_pairs_both_archs() {
    // the batched prefill-fill carries the same [1,n_kv_heads,cap,head_dim] cache state pairs as
    // the masked decode it must match, for qwen2 (bias) and qwen3 (QK-norm).
    for family in ["qwen2", "qwen3"] {
        let g = fixture_model(family)
            .trace(Phase::Prefill, contiguous(1, 5, 8))
            .unwrap();
        assert_contiguous_state(&g, family, 1, 8);
    }
}

#[test]
fn batched_decode_traces_with_state_pairs() {
    // the batched decode at B>1 carries [B,n_kv_heads,cap,head_dim] per-row caches and yields
    // [B,1,vocab] logits. Structural only.
    let m = fixture_model("qwen2");
    let g = m.trace(Phase::Decode, contiguous(3, 1, 16)).unwrap();
    assert_contiguous_state(&g, "qwen2", 3, 16);
    // B=1 must reduce to the same output shape as the scalar decode.
    let g1 = m.trace(Phase::Decode, contiguous(1, 1, 16)).unwrap();
    g1.validate().expect("B=1 batched decode should validate");
    assert_eq!(g1.aval(g1.output).shape, vec![1, 1, 48]);
}

#[test]
fn decode_handles_qwen3() {
    // arch-general: qwen3 (no bias, QK-norm) traces too.
    let g = fixture_model("qwen3")
        .trace(Phase::Decode, contiguous(1, 1, 8))
        .unwrap();
    g.validate().expect("qwen3 kv decode should validate");
    assert_eq!(g.state.len(), 2 * 2);
}

#[test]
fn exactly_two_slots() {
    // A contiguous step declares `Slot::Token` and `Slot::Pos` and nothing else.
    let g = fixture_model("qwen2")
        .trace(Phase::Decode, contiguous(1, 1, 16))
        .unwrap();
    assert_eq!(g.slots.len(), 2);
    let mut kinds: Vec<Slot> = g.slots.iter().map(|(_, s)| *s).collect();
    kinds.sort_by_key(|s| format!("{s:?}"));
    assert_eq!(kinds, vec![Slot::Pos, Slot::Token]);
}

#[test]
fn inputs_are_const_slot_or_state() {
    let g = fixture_model("qwen2")
        .trace(Phase::Decode, contiguous(1, 1, 16))
        .unwrap();
    for &id in &g.inputs {
        assert!(matches!(
            g.values[id].storage,
            Storage::Const | Storage::Computed(_) | Storage::Slot(_) | Storage::State
        ));
    }
}

/// The flash gate for `family`'s prefill under `kv`: `optimize()` must fuse attention into
/// `FlashAttentionPrefill` (one per layer) and leave no materialized `[1,Hq,L,L]` softmax(QK^T)
/// score `MatMul` (card 183; that O(L^2) matrix blew past the wgpu grid cap at long context,
/// docs/updates/0483). `L=2048` is far from every fixture dimension, avoiding a false positive
/// where a per-row weight matmul or attention reshape collides with `L` on some axis. This is the
/// graph-transform half; the callers must also run the pipeline rather than a `cse`-only path.
fn assert_flash_fused_prefill(family: &str, kv: KvLayout) {
    let n = 2048usize;
    let m = model(family, |c| c["max_position_embeddings"] = 4096.into());
    let (layers, ..) = dims(&*m, family);
    let g = m
        .trace(Phase::Prefill, shape(1, n, n, kv, LogitRows::Last))
        .unwrap();
    let opt = crate::test_support::optimize(&g);

    let flash_prefill = opt
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::FlashAttentionPrefill { .. }))
        .count();
    assert_eq!(
        flash_prefill, layers,
        "expected one fused flash-prefill op per layer"
    );

    // no MatMul should produce a full [.,Hq,L,L] (or larger) score-matrix-shaped output; flash
    // attention never materializes it.
    for e in &opt.eqns {
        if matches!(e.op, OpKind::MatMul) {
            // Shape-aware, not a size threshold (Card 557: the projections are plain `MatMul`s too
            // now, and an `[1,L,inter]` MLP output can exceed L*L elements): a rank-4 `[.,Hq,L,L']`
            // output with both trailing dims >= n is the materialized softmax(QK^T) score matrix.
            let shape = &opt.aval(e.out).shape;
            let r = shape.len();
            assert!(
                !(r == 4 && shape[r - 2] >= n && shape[r - 1] >= n),
                "found a materialized N^2-scale MatMul the flash fusion should have replaced: \
                 shape={shape:?}"
            );
        }
    }
}

#[test]
fn optimize_fuses_flash_prefill_no_materialized_scores() {
    assert_flash_fused_prefill("qwen2", KvLayout::Contiguous);
}

/// `VlmRunner::caption_streaming` (crates/poot-llm/src/vlm.rs) prefills from `trace_prefill_kv_embeds`,
/// not plain `trace_prefill_kv` (card 189). That tracer delegates to the same `trace_prefill_kv_impl`
/// body (only the token-slot-vs-embeds-constant input differs), so this proves the flash-fusion
/// guarantee of `optimize_fuses_flash_prefill_no_materialized_scores` for the tracer the VLM path uses.
///
/// `L=2048` is far from every config dimension below (`vocab=49280`, `hidden=576`, `inter=1536`,
/// `head_dim=64`, `q_dim=576`, `kv_dim=192`), avoiding a false positive where a per-row weight matmul
/// or attention reshape collides with `L` on some axis and is miscounted as a score matrix.
#[test]
fn optimize_fuses_flash_prefill_vlm_embeds_no_materialized_scores() {
    // SmolVLM-256m text config (`crate::vlm::TextConfig::smolvlm` via `poot_models::qwen2::Qwen2Config`,
    // mirrored here since poot-models does not depend on poot-llm), with `layers` shrunk from 30 to 2
    // for a fast host-only trace.
    let cfg = Qwen2Config {
        vocab: 49280,
        hidden: 576,
        inter: 1536,
        layers: 2,
        n_heads: 9,
        n_kv_heads: 3,
        head_dim: 64,
        rotary_dim: 64,
        eps: 1e-5,
        max_pos: 8192,
        qkv_bias: false,
        qk_norm: false,
        ..Default::default()
    };
    let n = 2048usize;
    let g = trace_prefill_kv_embeds(cfg, n, n);
    let opt = crate::test_support::optimize(&g);

    let flash_prefill = opt
        .eqns
        .iter()
        .filter(|e| matches!(e.op, poot_graph_ir::OpKind::FlashAttentionPrefill { .. }))
        .count();
    assert_eq!(
        flash_prefill, cfg.layers,
        "expected one fused flash-prefill op per layer"
    );

    // no MatMul should produce a full [.,Hq,L,L] (or larger) score-matrix-shaped output; flash
    // attention never materializes it.
    for e in &opt.eqns {
        if matches!(e.op, poot_graph_ir::OpKind::MatMul) {
            // Shape-aware, not a size threshold (Card 557: the projections are plain `MatMul`s too
            // now, and an `[1,L,inter]` MLP output can exceed L*L elements): a rank-4 `[.,Hq,L,L']`
            // output with both trailing dims >= n is the materialized softmax(QK^T) score matrix.
            let shape = &opt.aval(e.out).shape;
            let r = shape.len();
            assert!(
                !(r == 4 && shape[r - 2] >= n && shape[r - 1] >= n),
                "found a materialized N^2-scale MatMul the flash fusion should have replaced: \
                 shape={shape:?}"
            );
        }
    }
}

/// Without `optimize()` (the raw `cse`-only path), the same graph does materialize the naive
/// `[1,Hq,L,L]` score matrix (card 183). Kept as a contrast so a caller reverting to the raw path
/// is obvious.
#[test]
fn cse_only_leaves_naive_materialized_scores() {
    let n = 2048usize;
    let m = model("qwen2", |c| c["max_position_embeddings"] = 4096.into());
    let g = m.trace(Phase::Prefill, contiguous(1, n, n)).unwrap();
    let g = poot_graph_plan::cse(&g);
    let has_naive_scores = g.eqns.iter().any(|e| {
        matches!(e.op, OpKind::MatMul) && g.aval(e.out).shape.iter().product::<usize>() >= n * n
    });
    assert!(
        has_naive_scores,
        "expected the raw cse-only graph to still materialize the N^2 score matrix"
    );
}

/// The paged layout (`ScatterUpdate` by an inverse slot map instead of a contiguous
/// `DynamicUpdateSlice`) must flash-fuse like the contiguous one (card 189). `pool=n+8` (not
/// `pool==n`) shows the gate does not pass only when `pool==n`. Attention spans only the `l=n`
/// prompt positions, so the extra `ScatterUpdate`/transpose traffic around the cache write must not
/// stop the flash matcher.
#[test]
fn optimize_fuses_flash_prefill_paged_no_materialized_scores() {
    let n = 2048usize;
    assert_flash_fused_prefill(
        "qwen2",
        KvLayout::Paged {
            pool_slots: NonZeroUsize::new(n + 8).unwrap(),
        },
    );
}

#[test]
fn olmo2_prefill_traces_with_state_pairs() {
    // the olmo2 batched prefill must carry the same state pairs (names, order, shapes) as its
    // decode, so the filled cache drops into that decode loop (card 190).
    let m = fixture_model("olmo2");
    let g = m.trace(Phase::Prefill, contiguous(1, 5, 16)).unwrap();
    assert_contiguous_state(&g, "olmo2", 1, 16);
    let decode = m.trace(Phase::Decode, contiguous(1, 1, 16)).unwrap();
    let shapes = |g: &Graph<ValidationOutputs>| -> Vec<Vec<usize>> {
        g.state
            .iter()
            .map(|(si, _)| g.aval(*si).shape.clone())
            .collect()
    };
    assert_eq!(shapes(&g), shapes(&decode));
}

/// olmo2 flash-prefill host gate (card 190): the prefill must flash-fuse like plain qwen2.
#[test]
fn optimize_fuses_flash_prefill_olmo2_no_materialized_scores() {
    assert_flash_fused_prefill("olmo2", KvLayout::Contiguous);
}
