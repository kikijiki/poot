use poot_graph_ir::Slot;
use poot_graph_ir::{Graph, OpKind, Storage};
use std::num::NonZeroUsize;

use crate::model::{KvLayout, LogitRows, Model, StepShape, StepSpec};
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
    StepShape::new(nz(rows), nz(tokens), nz(capacity))
        .with_kv(kv)
        .with_logits(logits)
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
    let n = |f: &str| crate::registry::Fixture::count(&config, f);
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
fn assert_contiguous_state(g: &Graph, family: &str, rows: usize, cap: usize) {
    let m = fixture_model(family);
    let (layers, kv_heads, head_dim, vocab) = dims(&*m, family);
    assert_eq!(g.aval(g.result()).shape, vec![rows, 1, vocab]);
    assert_eq!(g.state().count(), 2 * layers);
    for (si, _so) in g.state() {
        assert_eq!(g.aval(si).shape, vec![rows, kv_heads, cap, head_dim]);
        assert_eq!(g.values()[si].storage, Storage::State);
    }
}

#[test]
fn decode_traces_and_validates() {
    let m = fixture_model("qwen2");
    let g = m
        .trace(
            &StepSpec::step(contiguous(1, 1, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    // structural well-formedness: every operand defined before use, stored avals match infer.
    assert!(g.eqns().len() > 100);
    assert_eq!(g.aval(g.result()).shape, vec![1, 1, 48]);
}

#[test]
fn prefill_traces_and_validates_both_archs() {
    // qwen2 (bias, no QK-norm) and qwen3 (no bias, QK-norm) are two registry families.
    for family in ["qwen2", "qwen3"] {
        let m = fixture_model(family);
        let g = m
            .trace(
                &StepSpec::step(contiguous(1, 5, 8)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        assert_eq!(g.aval(g.result()).shape, vec![1, 1, m.config().vocab]);
    }
}

#[test]
fn gemma3_decode_traces_with_state_pairs() {
    let g = fixture_model("gemma3")
        .trace(
            &StepSpec::step(contiguous(1, 1, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    assert_contiguous_state(&g, "gemma3", 1, 16);
}

#[test]
fn gemma3_prefill_traces_with_state_pairs() {
    // the batched prefill must carry the SAME state pairs as the decode graph.
    let g = fixture_model("gemma3")
        .trace(
            &StepSpec::step(contiguous(1, 5, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
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
        .trace(
            &StepSpec::step(contiguous(1, 1, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
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
    let g = m
        .trace(
            &StepSpec::step(contiguous(1, 1, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let base = dce(&cse(&g).unwrap()).unwrap();
    let fused = dce(&rope_fusion(&cse(&g).unwrap()).unwrap().0).unwrap();
    let ropes: Vec<usize> = fused
        .eqns()
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
    let count_negs = |g: &Graph| {
        g.eqns()
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
    let dispatch_count = |g: &Graph| {
        g.eqns()
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
        .trace(
            &StepSpec::step(contiguous(1, 1, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let elided = elide_noop_transposes(&g).unwrap();
    let collapsed = collapse_reshape_chains(&elided).unwrap();
    assert!(
        collapsed.eqns().len() < elided.eqns().len(),
        "collapse should shrink the decode graph: {} -> {} eqns",
        elided.eqns().len(),
        collapsed.eqns().len()
    );
    assert_eq!(
        collapsed.aval(collapsed.result()).shape,
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
            .trace(
                &StepSpec::step(contiguous(1, 5, 8)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        assert_contiguous_state(&g, family, 1, 8);
    }
}

#[test]
fn batched_decode_traces_with_state_pairs() {
    // the batched decode at B>1 carries [B,n_kv_heads,cap,head_dim] per-row caches and yields
    // [B,1,vocab] logits. Structural only.
    let m = fixture_model("qwen2");
    let g = m
        .trace(
            &StepSpec::step(contiguous(3, 1, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    assert_contiguous_state(&g, "qwen2", 3, 16);
    // B=1 must reduce to the same output shape as the scalar decode.
    let g1 = m
        .trace(
            &StepSpec::step(contiguous(1, 1, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    assert_eq!(g1.aval(g1.result()).shape, vec![1, 1, 48]);
}

#[test]
fn decode_handles_qwen3() {
    // arch-general: qwen3 (no bias, QK-norm) traces too.
    let g = fixture_model("qwen3")
        .trace(
            &StepSpec::step(contiguous(1, 1, 8)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    assert_eq!(g.state().count(), 2 * 2);
}

#[test]
fn exactly_two_slots() {
    // A contiguous step declares `Slot::Token` and `Slot::Pos` and nothing else.
    let g = fixture_model("qwen2")
        .trace(
            &StepSpec::step(contiguous(1, 1, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    assert_eq!(g.slots().len(), 2);
    let mut kinds: Vec<Slot> = g.slots().iter().map(|(_, s)| *s).collect();
    kinds.sort_by_key(|s| format!("{s:?}"));
    assert_eq!(kinds, vec![Slot::Pos, Slot::Token]);
}

#[test]
fn inputs_are_const_slot_or_state() {
    let g = fixture_model("qwen2")
        .trace(
            &StepSpec::step(contiguous(1, 1, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    for &id in g.inputs() {
        assert!(matches!(
            g.values()[id].storage,
            Storage::Const | Storage::Computed(_) | Storage::Slot(_) | Storage::State
        ));
    }
}

/// The flash gate for `family`'s prefill under `kv`: `optimize()` must fuse attention into
/// `Attention` (one per layer) and leave no materialized `[1,Hq,L,L]` softmax(QK^T)
/// score `MatMul` (card 183; that O(L^2) matrix blew past the wgpu grid cap at long context).
/// `L=2048` is far from every fixture dimension, avoiding a false positive where a per-row weight matmul or
/// attention reshape collides with `L` on some axis. This is the graph-transform half;
/// the callers must also run the pipeline rather than a `cse`-only path.
fn assert_flash_fused_prefill(family: &str, kv: KvLayout) {
    let n = 2048usize;
    let m = model(family, |c| c["max_position_embeddings"] = 4096.into());
    let (layers, ..) = dims(&*m, family);
    let g = m
        .trace(
            &StepSpec::step(shape(1, n, n, kv, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let opt = poot_graph_plan::passes_without_target(&g).unwrap();

    let flash_prefill = opt
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::Attention(_)))
        .count();
    assert_eq!(
        flash_prefill, layers,
        "expected one fused flash-prefill op per layer"
    );

    // no MatMul should produce a full [.,Hq,L,L] (or larger) score-matrix-shaped output; flash
    // attention never materializes it.
    for e in opt.eqns() {
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

/// A VLM prompt prefills through a step carrying `EMBEDS` (the caller's rows scattered over the
/// embedded tokens), not a plain step (card 189). The embeds are a select ahead of the layers, so
/// this proves the flash-fusion guarantee of `optimize_fuses_flash_prefill_no_materialized_scores`
/// for the step the VLM text tower runs. `L=2048` is far from every fixture dimension, avoiding a
/// false positive where a per-row weight matmul or attention reshape collides with `L` on some axis
/// and is miscounted as a score matrix.
#[test]
fn optimize_fuses_flash_prefill_vlm_embeds_no_materialized_scores() {
    let n = 2048usize;
    let m = model("qwen2", |c| c["max_position_embeddings"] = 4096.into());
    let (layers, ..) = dims(&*m, "qwen2");
    let g = m
        .trace(
            &StepSpec::step(shape(1, n, n, KvLayout::Contiguous, LogitRows::Last))
                .carrying(crate::components::embed::EMBEDS),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let opt = poot_graph_plan::passes_without_target(&g).unwrap();

    let flash_prefill = opt
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::Attention(_)))
        .count();
    assert_eq!(
        flash_prefill, layers,
        "expected one fused flash-prefill op per layer"
    );
    for e in opt.eqns() {
        if matches!(e.op, OpKind::MatMul) {
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
    let g = m
        .trace(
            &StepSpec::step(contiguous(1, n, n)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let g = poot_graph_plan::cse(&g).unwrap();
    let has_naive_scores = g.eqns().iter().any(|e| {
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
            seqs: NonZeroUsize::MIN,
        },
    );
}

#[test]
fn olmo2_prefill_traces_with_state_pairs() {
    // the olmo2 batched prefill must carry the same state pairs (names, order, shapes) as its
    // decode, so the filled cache drops into that decode loop (card 190).
    let m = fixture_model("olmo2");
    let g = m
        .trace(
            &StepSpec::step(contiguous(1, 5, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    assert_contiguous_state(&g, "olmo2", 1, 16);
    let decode = m
        .trace(
            &StepSpec::step(contiguous(1, 1, 16)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let shapes = |g: &Graph| -> Vec<Vec<usize>> {
        g.state().map(|(si, _)| g.aval(si).shape.clone()).collect()
    };
    assert_eq!(shapes(&g), shapes(&decode));
}

/// olmo2 flash-prefill host gate (card 190): the prefill must flash-fuse like plain qwen2.
#[test]
fn optimize_fuses_flash_prefill_olmo2_no_materialized_scores() {
    assert_flash_fused_prefill("olmo2", KvLayout::Contiguous);
}

// The Qwen2.5-VL text tower: qwen2 with sectioned RoPE over `Slot::MropePosition`.

mod vl {
    use poot_graph_ir::Slot;
    use poot_quant::weights::WeightStore;

    use super::super::model::{VL_ENTRY, VL_FAMILY};
    use super::*;
    use crate::components::standard::oracle;
    use crate::components::testing::{CAP, close, step_shape};
    use crate::model::{ConfigReason, ModelError};
    use crate::registry::Fixture;

    /// The text tower over its own fixture with `tweak` applied to the config. The tensors do not
    /// depend on the fields these rows change, and the plain qwen2 fixture holds the same bytes.
    fn build(
        tweak: impl FnOnce(&mut serde_json::Value),
    ) -> (Fixture, Result<Box<dyn Model>, ModelError>) {
        let mut fixture = (VL_ENTRY.fixture)();
        tweak(&mut fixture.config);
        let built = (VL_ENTRY.build)(&fixture.raw(), &fixture.store);
        (fixture, built)
    }

    fn vl_model() -> (Fixture, Box<dyn Model>) {
        let (fixture, built) = build(|_| {});
        (fixture, built.unwrap())
    }

    /// The error a config the tower refuses reports: its field and reason, for the VL family.
    fn refusal(tweak: impl FnOnce(&mut serde_json::Value)) -> (&'static str, ConfigReason) {
        match build(tweak).1 {
            Err(ModelError::Config {
                family,
                field,
                reason,
            }) => {
                assert_eq!(family, VL_FAMILY);
                (field, reason)
            }
            Err(other) => panic!("expected a config error, got {other}"),
            Ok(_) => panic!("the config was accepted"),
        }
    }

    const SECTION: &str = "rope_scaling.mrope_section";

    /// The released config shape builds: `rope_scaling` names `mrope` and its three section widths.
    /// The same config under `rope_parameters` with `rope_type` builds the same tower.
    #[test]
    fn the_released_shape_builds_under_either_rope_block() {
        let (fixture, model) = vl_model();
        assert_eq!(model.config().family, VL_FAMILY);
        let relocated = build(|c| {
            let scaling = c["rope_scaling"].take();
            c["rope_parameters"] = serde_json::json!({
                "rope_type": "mrope",
                "mrope_section": scaling["mrope_section"],
            });
        })
        .1
        .unwrap();
        let tokens = [3, 17, 40, 8, 25];
        let (a, _) = vl_step(&*model, &fixture.store, &tokens, 0, None, &[]);
        let (b, _) = vl_step(&*relocated, &fixture.store, &tokens, 0, None, &[]);
        close(&a, &b);
    }

    /// Only `qwen2_5_vl` builds: Qwen2-VL (a different position scheme) and plain qwen2 are refused
    /// by the `model_type` field. Mutation: drop the model type check; this goes red with
    /// "the config was accepted".
    #[test]
    fn another_model_type_is_refused() {
        for kind in ["qwen2_vl", "qwen2"] {
            assert_eq!(
                refusal(|c| c["model_type"] = kind.into()),
                ("model_type", ConfigReason::Unsupported),
                "{kind}"
            );
        }
    }

    /// A checkpoint without QKV bias is not the Qwen2.5-VL convention. Mutation: drop the
    /// `attention_bias` check.
    #[test]
    fn a_checkpoint_without_qkv_bias_is_refused() {
        assert_eq!(
            refusal(|c| c["attention_bias"] = false.into()),
            ("attention_bias", ConfigReason::Unsupported)
        );
    }

    /// An explicit `head_dim` other than `hidden_size / num_attention_heads` (64 / 4 = 16) is
    /// refused. Mutation: drop the head width check.
    #[test]
    fn a_head_width_other_than_hidden_over_heads_is_refused() {
        assert_eq!(
            refusal(|c| c["head_dim"] = 32.into()),
            (
                "head_dim",
                ConfigReason::Mismatch {
                    expected: 16,
                    found: 32
                }
            )
        );
    }

    /// A partial rotary is refused: the sections cover the whole head. Mutation: drop the
    /// `partial_rotary_factor` check.
    #[test]
    fn a_partial_rotary_is_refused() {
        assert_eq!(
            refusal(|c| c["partial_rotary_factor"] = 0.5.into()),
            (
                "partial_rotary_factor",
                ConfigReason::Mismatch {
                    expected: 16,
                    found: 8
                }
            )
        );
    }

    /// Section widths that do not cover the rotary pairs exactly are refused: too few (3 of 8
    /// pairs) and too many (9 of 8). Mutation: drop the sum check in `RopeParams::sectioned`.
    #[test]
    fn section_widths_must_cover_the_rotary_pairs() {
        for (widths, found) in [([1, 1, 1], 3), ([3, 3, 3], 9)] {
            assert_eq!(
                refusal(|c| c["rope_scaling"]["mrope_section"] = widths.into()),
                (SECTION, ConfigReason::Mismatch { expected: 8, found }),
                "{widths:?}"
            );
        }
    }

    /// A sum past `usize` is an overflow, not a wrapped (and so accepted) width. Mutation: add the
    /// widths with a wrapping sum in `RopeParams::sectioned`.
    #[test]
    fn section_widths_that_overflow_are_refused() {
        // 2^63 + 2^63 + 8 wraps to 8 as a u64: the exact pairs of the fixture.
        let half = 1u64 << 63;
        assert_eq!(
            refusal(|c| c["rope_scaling"]["mrope_section"] = serde_json::json!([half, half, 8])),
            (SECTION, ConfigReason::Overflow)
        );
    }

    /// An odd rotary width is refused whatever the sections sum to: a head of 15 has 7 full pairs
    /// and a trailing dim, and sections `[2, 2, 3]` would cover those 7 pairs. Mutation: drop the
    /// even-width check in `RopeParams::new`.
    #[test]
    fn an_odd_rotary_width_is_refused() {
        assert_eq!(
            refusal(|c| {
                c["hidden_size"] = 60.into();
                c["rope_scaling"]["mrope_section"] = serde_json::json!([2, 2, 3]);
            }),
            ("rotary_dim", ConfigReason::NotDivisible { by: 2 })
        );
    }

    /// Malformed section metadata is a typed error naming the key: absent, not a list, the wrong
    /// length, or a width that is not an unsigned integer. Mutation: accept a two-width list
    /// (pad it with a zero); the length row goes red.
    #[test]
    fn malformed_section_metadata_is_refused() {
        let key = |c: &mut serde_json::Value, value: serde_json::Value| {
            c["rope_scaling"]["mrope_section"] = value;
        };
        assert_eq!(
            refusal(|c| {
                c["rope_scaling"]
                    .as_object_mut()
                    .unwrap()
                    .remove("mrope_section");
            }),
            (SECTION, ConfigReason::Missing)
        );
        assert_eq!(
            refusal(|c| key(c, "2,3,3".into())),
            (SECTION, ConfigReason::WrongType)
        );
        assert_eq!(
            refusal(|c| key(c, serde_json::json!([4, 4]))),
            (SECTION, ConfigReason::Length { expected: 3 })
        );
        assert_eq!(
            refusal(|c| key(c, serde_json::json!([2, -3, 3]))),
            (SECTION, ConfigReason::WrongType)
        );
    }

    /// One step over `tokens` from `start`, logits at every token, with the sectioned
    /// positions `mrope` (three axes of `tokens.len()` ids, `None` for the text positions on all
    /// three) over `state` (zeros when empty). Returns the logits and the carried state.
    fn vl_step(
        model: &dyn Model,
        store: &WeightStore,
        tokens: &[i32],
        start: i32,
        mrope: Option<[&[i32]; 3]>,
        state: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let pos: Vec<i32> = (start..).take(tokens.len()).collect();
        vl_step_at(model, store, tokens, &pos, mrope, state)
    }

    /// [`vl_step`] with the cache positions `pos` given outright.
    fn vl_step_at(
        model: &dyn Model,
        store: &WeightStore,
        tokens: &[i32],
        pos: &[i32],
        mrope: Option<[&[i32]; 3]>,
        state: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let g = model
            .trace(
                &StepSpec::step(step_shape(tokens.len(), CAP, LogitRows::All)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let zeros = state
            .is_empty()
            .then(|| g.state().map(|(si, _)| vec![0.0; g.aval(si).numel()]));
        let zeros: Vec<Vec<f32>> = zeros.into_iter().flatten().collect();
        let state: Vec<&[f32]> = if state.is_empty() { &zeros } else { state }
            .iter()
            .map(Vec::as_slice)
            .collect();
        let packed: Vec<i32> = match mrope {
            Some(axes) => axes.concat(),
            None => pos.repeat(3),
        };
        oracle::eval_slots(
            &g,
            store,
            model.weights(),
            &[],
            &[
                (Slot::Token, tokens),
                (Slot::Pos, pos),
                (Slot::MropePosition, &packed),
            ],
            (&[], &[]),
            &state,
        )
    }

    const PROMPT: [i32; 5] = [3, 17, 40, 8, 25];

    /// A text-only prompt (every token at its text position on all three axes) is plain qwen2: the
    /// prefill logits at every position and the decode step after it equal the plain model's over
    /// the same weights. Mutation: take the sin table from the cos table in `apply_rope_sectioned`;
    /// the equality goes red.
    #[test]
    fn text_positions_on_all_three_axes_are_the_plain_qwen2_model() {
        let (fixture, vl) = vl_model();
        let plain = fixture_model("qwen2");
        let plain_store = (plain_entry().fixture)().store;
        assert_eq!(
            plain_store.iter().count(),
            fixture.store.iter().count(),
            "both fixtures hold the same tensors"
        );
        let (want, plain_state) = oracle::run_step(&*plain, &plain_store, CAP, &PROMPT, 0, &[]);
        let (got, state) = vl_step(&*vl, &fixture.store, &PROMPT, 0, None, &[]);
        close(&got, &want);
        let (want, _) = oracle::run_step(
            &*plain,
            &plain_store,
            CAP,
            &[11],
            PROMPT.len() as i32,
            &plain_state,
        );
        let (got, _) = vl_step(
            &*vl,
            &fixture.store,
            &[11],
            PROMPT.len() as i32,
            None,
            &state,
        );
        close(&got, &want);
    }

    fn plain_entry() -> crate::registry::FamilyEntry {
        super::super::model::ENTRY
    }

    /// Each axis of `Slot::MropePosition` reaches the logits through its own section: moving only
    /// the temporal, only the height or only the width ids of a prompt (unevenly) changes them, each
    /// differently, moving all three equals plain qwen2 at those positions, and moving none does not
    /// change them. Mutation: build the tower's `Step` without
    /// `with_mrope`; the trace panics at the sectioned attention.
    #[test]
    fn each_axis_changes_the_logits_through_its_own_section() {
        let (fixture, vl) = vl_model();
        let text: Vec<i32> = (0..PROMPT.len() as i32).collect();
        // Not a uniform shift: RoPE is relative, so moving every id alike changes no score.
        let moved: Vec<i32> = vec![0, 2, 3, 5, 7];
        let (text, moved) = (text.as_slice(), moved.as_slice());
        let run = |axes: [&[i32]; 3]| vl_step(&*vl, &fixture.store, &PROMPT, 0, Some(axes), &[]).0;
        let base = run([text, text, text]);
        let (plain, _) = vl_step(&*vl, &fixture.store, &PROMPT, 0, None, &[]);
        close(&base, &plain);
        // Independent reference: with all three axes on the moved ids the cache sits at the moved
        // positions too, and the tower is plain qwen2 at those positions (holes in the cache and all).
        let plain = fixture_model("qwen2");
        let plain_store = (plain_entry().fixture)().store;
        let g = plain
            .trace(
                &StepSpec::step(step_shape(PROMPT.len(), CAP, LogitRows::All)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state()
            .map(|(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let zeros: Vec<&[f32]> = zeros.iter().map(Vec::as_slice).collect();
        let (want, _) = oracle::eval_slots(
            &g,
            &plain_store,
            plain.weights(),
            &[],
            &[(Slot::Token, &PROMPT), (Slot::Pos, moved)],
            (&[], &[]),
            &zeros,
        );
        let (got, _) = vl_step_at(
            &*vl,
            &fixture.store,
            &PROMPT,
            moved,
            Some([moved, moved, moved]),
            &[],
        );
        close(&got, &want);
        // Each single axis moves the logits, and no two axes move them alike: a section read from
        // the wrong axis would repeat another axis's logits.
        let singles: Vec<(&str, Vec<f32>)> = [
            ("temporal", [moved, text, text]),
            ("height", [text, moved, text]),
            ("width", [text, text, moved]),
        ]
        .into_iter()
        .map(|(axis, axes)| (axis, run(axes)))
        .collect();
        let differ = |a: &[f32], b: &[f32]| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-5);
        for (axis, out) in &singles {
            assert!(differ(out, &base), "the {axis} ids do not reach the logits");
        }
        for (i, (a, out_a)) in singles.iter().enumerate() {
            for (b, out_b) in &singles[i + 1..] {
                assert!(
                    differ(out_a, out_b),
                    "the {a} and {b} ids move the logits alike"
                );
            }
        }
    }
}
