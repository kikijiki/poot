//! Helpers for this crate's own `#[cfg(test)]` modules: stage-split assignments and a hand-built layered
//! decode graph (the cross-crate fixtures live in [`crate::test_support`]; chained evaluation and input
//! binding live in `poot_eval::test_support::stage_chain`). No model crate is involved: the traced real
//! qwen2 rows are poot-models' tests.

// ---- stage splits ---------------------------------------------------------------------------

use crate::passes::{CutSite, StageAssignment, split_stages};
use poot_graph_ir::{
    BinOp, Builder, Graph, IndexRange, Slot, StateRole, TensorType,
    ops::{attention_masked, causal_mask_from_pos, rmsnorm, silu},
};
use poot_tensor::DType;

use crate::StageSplitError;
use crate::passes::stage_split::StageSplit;

/// Cut `g` before each listed equation.
pub(crate) fn assignment_before(
    g: &Graph,
    cuts: &[usize],
) -> Result<StageAssignment, StageSplitError> {
    let sites: Vec<CutSite> = cuts.iter().map(|&cut| CutSite::Before(cut)).collect();
    StageAssignment::from_sites(g, &sites)
}

/// Cut `g` after each listed layer.
pub(crate) fn assignment_after_layers(
    g: &Graph,
    layers: &[usize],
) -> Result<StageAssignment, StageSplitError> {
    let sites: Vec<CutSite> = layers
        .iter()
        .map(|&layer| CutSite::AfterLayer(layer))
        .collect();
    StageAssignment::from_sites(g, &sites)
}

/// Split `g` before each listed equation.
pub(crate) fn split_before(g: &Graph, cuts: &[usize]) -> Result<StageSplit, StageSplitError> {
    split_stages(g, &assignment_before(g, cuts)?)
}

/// Split `g` after each listed layer.
pub(crate) fn split_after_layers(
    g: &Graph,
    layers: &[usize],
) -> Result<StageSplit, StageSplitError> {
    split_stages(g, &assignment_after_layers(g, layers)?)
}

// ---- the layered decode fixture --------------------------------------------------------------

/// Layers of [`layered_decode`]'s default fixture.
pub(crate) const LAYERS: usize = 3;

const VOCAB: usize = 16;
const HIDDEN: usize = 8;
const HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / HEADS;
const INTER: usize = 16;
/// Positions of every layer's KV cache.
const CAP: usize = 6;

/// A decode step of a `layers`-layer dense transformer built by hand: an embedding gather, per layer a
/// pre-norm attention over a carried KV cache (written at the `Pos` slot) and a gated MLP, each with a
/// residual add, then the final norm and the head. The causal mask is computed once from `Pos` and read by
/// every layer, so a cut after any layer crosses it as well as the residual stream. Every layer sits in its
/// own `layer_scope`, which is what `AfterLayer` cuts read.
pub(crate) fn layered_decode(layers: usize) -> Graph {
    let b = Builder::for_tests();
    let token = b
        .slot(Slot::Token, TensorType::new(vec![1, 1], DType::I32))
        .unwrap();
    // A position is below the cache capacity: declared as the tracer does, so the `I32 -> F32` cast of the
    // position in the mask is proven exact.
    let pos = b
        .slot_ranged(
            Slot::Pos,
            TensorType::new(vec![1, 1], DType::I32),
            IndexRange::below(CAP).unwrap(),
        )
        .unwrap();
    let embed = b
        .constant("embed", TensorType::f32(vec![VOCAB, HIDDEN]))
        .unwrap();
    let mut x = b.gather(embed, 0, token).unwrap();
    let mask = causal_mask_from_pos(&b, pos, CAP, None).unwrap();
    let write_at = b.reshape(pos, vec![]).unwrap();
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut state = Vec::new();
    for layer in 0..layers {
        let _scope = b.layer_scope(layer);
        let weight = |name: &str, shape: Vec<usize>| {
            b.constant(&format!("l{layer}.{name}"), TensorType::f32(shape))
                .unwrap()
        };
        let attn_norm = weight("attn_norm", vec![HIDDEN]);
        let h = rmsnorm(&b, x, attn_norm, 1e-6).unwrap();
        let heads = |t| {
            let t = b.reshape(t, vec![1, 1, HEADS, HEAD_DIM]).unwrap();
            b.transpose(t, vec![0, 2, 1, 3]).unwrap()
        };
        let q = heads(b.matmul(h, weight("wq", vec![HIDDEN, HIDDEN])).unwrap());
        let k = heads(b.matmul(h, weight("wk", vec![HIDDEN, HIDDEN])).unwrap());
        let v = heads(b.matmul(h, weight("wv", vec![HIDDEN, HIDDEN])).unwrap());
        let cache_type = TensorType::f32(vec![1, HEADS, CAP, HEAD_DIM]);
        let k_in = b
            .state_input(
                &format!("l{layer}.k_cache"),
                cache_type.clone(),
                StateRole::Recurrent,
            )
            .unwrap();
        let v_in = b
            .state_input(
                &format!("l{layer}.v_cache"),
                cache_type,
                StateRole::Recurrent,
            )
            .unwrap();
        let k_out = b.dynamic_update_slice_dyn(k_in, k, write_at, 2).unwrap();
        let v_out = b.dynamic_update_slice_dyn(v_in, v, write_at, 2).unwrap();
        state.push((k_in, k_out));
        state.push((v_in, v_out));
        let attended = attention_masked(&b, q, k_out, v_out, 1, scale, mask).unwrap();
        let attended = b.transpose(attended, vec![0, 2, 1, 3]).unwrap();
        let attended = b.reshape(attended, vec![1, 1, HIDDEN]).unwrap();
        let o = b
            .matmul(attended, weight("wo", vec![HIDDEN, HIDDEN]))
            .unwrap();
        x = b.binary(BinOp::Add, x, o).unwrap();

        let mlp_norm = weight("mlp_norm", vec![HIDDEN]);
        let h = rmsnorm(&b, x, mlp_norm, 1e-6).unwrap();
        let gate = b.matmul(h, weight("gate", vec![HIDDEN, INTER])).unwrap();
        let up = b.matmul(h, weight("up", vec![HIDDEN, INTER])).unwrap();
        let gated = b.binary(BinOp::Mul, silu(&b, gate).unwrap(), up).unwrap();
        let down = b
            .matmul(gated, weight("down", vec![INTER, HIDDEN]))
            .unwrap();
        x = b.binary(BinOp::Add, x, down).unwrap();
    }
    let final_norm = b
        .constant("final_norm", TensorType::f32(vec![HIDDEN]))
        .unwrap();
    let h = rmsnorm(&b, x, final_norm, 1e-6).unwrap();
    let head = b
        .constant("lm_head", TensorType::f32(vec![HIDDEN, VOCAB]))
        .unwrap();
    let logits = b.matmul(h, head).unwrap();
    b.finish(poot_graph_ir::Outputs::new(logits).with_state(&state))
        .expect("the layered decode fixture is a valid graph")
}
