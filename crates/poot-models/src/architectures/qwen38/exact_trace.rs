//! Card 449 H0b: whole-model exact packed Qwen4Exp prefill/decode tracers.
//!
//! These are thin wrappers over the shared whole-model scaffolding in
//! [`super::trace`] (`trace_qwen38_prefill_impl` / `trace_qwen38_decode_impl`) with
//! [`Qwen4ExpWholeModelMode::Exact`]: packed FFN (Card 362), option-D in-graph PLE ids
//! (history State), and the sharded E4M3 PLE embedding (Card 363). Attention/GDN/HC/embed
//! constants and the layer loop are one definition with the dense synthetic path.
//!
//! Dense attention/GDN/HC/embed constants keep the dense tracers' F32 names and layouts; the
//! Runner-side binder maps them to sealed BF16 checkpoint owners. Packed I8 components, routed
//! BF16 role constants, and E4M3 PLE shards bind by owner identity.

#[cfg(test)]
use super::trace::{Qwen4ExpWholeModelMode, trace_qwen38_decode_impl, trace_qwen38_prefill_impl};
use super::*;

/// One layer's packed FFN sources for the exact whole-model tracer. `dense` is ordered
/// Router, SharedGate, SharedUp, SharedDown, SharedRouterGate - each a checkpoint-named BF16
/// constant in PyTorch `[out, in]` layout, transposed in-graph the way
/// `Qwen4ExpPreparedRoutedAdapter::trace_layer` does.
#[derive(Clone, Debug)]
pub struct Qwen4ExpExactLayerSources {
    pub tables: Qwen4ExpPackedExpertTables,
    pub dense: [Qwen4ExpExactDenseRole; 5],
}

/// One BF16 routed-dense role constant declared by the exact whole-model tracer.
#[derive(Clone, Debug)]
pub struct Qwen4ExpExactDenseRole {
    pub name: String,
    /// Checkpoint PyTorch layout `[out, in]`.
    pub shape: [usize; 2],
}

/// Whole-model exact packed sources: one entry per decoder layer, plus the PLE shard ranges for
/// the (single) PLE layer when `mcfg.ple` is `Some`.
#[derive(Clone, Debug)]
pub struct Qwen4ExpExactPackedSources {
    pub layers: Vec<Qwen4ExpExactLayerSources>,
    /// Sharded E4M3 n-gram embedding rows in logical shard order; empty only when PLE is absent.
    pub ple_shards: Vec<Qwen4ExpPleShardRange>,
    /// Checkpoint name of the PLE embedding's BF16 `weight_scale` scalar (F32 in-graph).
    pub ple_weight_scale_name: String,
}

/// Whole-model exact packed prefill error.
#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpExactTraceError {
    #[error("Qwen4Exp exact prefill needs {expected} packed layers, got {actual}")]
    LayerCount { expected: usize, actual: usize },
    #[error("Qwen4Exp exact PLE is configured but no PLE shard ranges were provided")]
    MissingPleShards,
    #[error("Qwen4Exp exact prefill capacity {capacity} is below seq_len {seq_len}")]
    CapacityBelowSeqLen { seq_len: usize, capacity: usize },
    #[error(transparent)]
    Packed(#[from] Qwen4ExpPackedRoutedError),
}

pub(crate) fn declare_routed_dense(
    b: &Builder,
    sources: &Qwen4ExpExactLayerSources,
) -> [Traced; 5] {
    std::array::from_fn(|index| {
        let row = &sources.dense[index];
        let source = b.constant(&row.name, TensorType::new(row.shape.to_vec(), DType::BF16));
        b.transpose(b.cast(source, DType::F32), vec![1, 0])
    })
}

pub(crate) fn exact_ple_embeddings(
    b: &Builder,
    ngram_ids: Traced,
    sources: &Qwen4ExpExactPackedSources,
    ple: &Qwen4ExpPleConfig,
) -> Traced {
    assert!(
        !sources.ple_shards.is_empty(),
        "exact PLE requires at least one shard range"
    );
    let row_width = ple.head_dim_per_ngram();
    let heads = ple.ngram_heads();
    let id_shape = b.aval(ngram_ids).shape;
    assert_eq!(id_shape.len(), 2, "n-gram ids must be [L, heads]");
    let l = id_shape[0];
    let scale = b.constant(&sources.ple_weight_scale_name, TensorType::f32(vec![]));
    // Flatten `[L, heads]` to one id per row so the sharded lookup's scalar offset/mask arithmetic
    // stays rank-1 (the Card 363 composition indexes one id at a time). Head offsets already place
    // every head in the combined sharded table (production `QWEN4EXP_PLE_ROW_WIDTH`).
    let flat = b.reshape(ngram_ids, vec![l * heads]);
    let rows = qwen4exp_ple_scaled_row(b, flat, &sources.ple_shards, row_width, scale);
    // rows: [L*heads, head_dim] -> [1, L, ple_embed_dim] (row-major matches [L, heads, head_dim]).
    b.reshape(rows, vec![1, l, heads * row_width])
}

/// Whole-model exact packed prefill (Card 449 H0b). Same layer structure as
/// [`trace_qwen38_prefill`], with packed FFN, option-D in-graph PLE ids (history State), and the
/// sharded E4M3 PLE embedding. `seq_len` is the real prompt length (no token padding); `capacity`
/// is the fixed KV/idx cache size carried into decode (`>= seq_len`, multiple of the compress
/// ratio). Prefill's `graph.state` layout matches [`trace_qwen4exp_exact_decode`] so a prefill
/// output state feeds decode directly.
#[cfg(test)]
pub(crate) fn trace_qwen4exp_exact_prefill(
    mcfg: &Qwen4ExpModelConfig,
    seq_len: usize,
    capacity: usize,
    sources: &Qwen4ExpExactPackedSources,
) -> Result<Graph, Qwen4ExpExactTraceError> {
    if mcfg.ple.is_some() && sources.ple_shards.is_empty() {
        return Err(Qwen4ExpExactTraceError::MissingPleShards);
    }
    if capacity < seq_len {
        return Err(Qwen4ExpExactTraceError::CapacityBelowSeqLen { seq_len, capacity });
    }
    trace_qwen38_prefill_impl(
        mcfg,
        seq_len,
        Qwen4ExpWholeModelMode::Exact(sources),
        Some(capacity),
    )
}

/// Whole-model exact packed decode (Card 449 H0b). Same layer structure as
/// [`trace_qwen38_decode`], with packed FFN, option-D in-graph PLE ids (history + conv-cache
/// State), and the sharded E4M3 PLE embedding.
#[cfg(test)]
pub(crate) fn trace_qwen4exp_exact_decode(
    mcfg: &Qwen4ExpModelConfig,
    cap: usize,
    sources: &Qwen4ExpExactPackedSources,
) -> Result<Graph, Qwen4ExpExactTraceError> {
    if mcfg.ple.is_some() && sources.ple_shards.is_empty() {
        return Err(Qwen4ExpExactTraceError::MissingPleShards);
    }
    trace_qwen38_decode_impl(mcfg, cap, Qwen4ExpWholeModelMode::Exact(sources))
}
