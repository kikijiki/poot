use super::*;

/// Trace a full-sequence prefill forward over `seq_len` tokens (positions `0..seq_len`), returning the
/// logits for the last position only (`[1,1,vocab]`). Const binders use the exact HF safetensors names so
/// a loader can bind by name + shape (transposing `[out,in]` linear weights to the `[in,out]` matmul
/// layout); the computed rope tables are named `rope.cos`, `rope.sin`, and the causal mask is the
/// `mask.prefill` step input (card 550a). This is the host coherence
/// path; the captured decode path is `trace_decode`.
pub fn trace_prefill(cfg: Qwen2Config, seq_len: usize) -> Graph {
    trace_prefill_impl(cfg, seq_len)
}

/// Trace a batched prefill that fills the fixed-capacity K/V cache (spec 023). It consumes `n` prompt
/// tokens (positions `0..n`) in one multi-token forward and writes each layer's RoPE'd K and V into the
/// same per-layer cache state tensors `trace_decode_kv_masked` uses (`[1, n_kv_heads, cap, head_dim]`,
/// same `state_input` names and `state_in/state_out` pairing), so a single-token masked decode continues
/// from the filled cache. `cap` is the cache capacity (`>= n`).
///
/// Unlike [`trace_decode_kv_masked`], this is N-specialized: the CPU correctness oracle for the batched
/// fill. Per-position K/V are computed as token-by-token decode computes them (`rope_prefill` slices the
/// same cos/sin rows `rope` gathers), then the `[1, Hkv, n, D]` block is scattered into the zero-seeded
/// cache at slot 0 (`DynamicUpdateSlice` along the seq axis). Arch-general (qwen2 bias / qwen3 QK-norm).
/// Trace a batched prefill that fills the fixed-capacity K/V cache (spec 023). It consumes `n` prompt
/// tokens (positions `0..n`) in one multi-token forward and writes each layer's RoPE'd K and V into the
/// same per-layer cache state tensors `trace_decode_kv_masked` uses (`[1, n_kv_heads, cap, head_dim]`,
/// same `state_input` names and `state_in/state_out` pairing), so a single-token masked decode continues
/// from the filled cache. `cap` is the cache capacity (`>= n`).
///
/// Unlike [`trace_decode_kv_masked`], this is N-specialized: the CPU correctness oracle for the batched
/// fill. Per-position K/V are computed as token-by-token decode computes them (`rope_prefill` slices the
/// same cos/sin rows `rope` gathers), then the `[1, Hkv, n, D]` block is scattered into the zero-seeded
/// cache at slot 0 (`DynamicUpdateSlice` along the seq axis). Arch-general (qwen2 bias / qwen3 QK-norm).
/// Output is the last position's logits `[1,1,vocab]`; state is the `2*layers` cache pairs.
pub fn trace_prefill_kv(cfg: Qwen2Config, n: usize, cap: usize) -> Graph {
    trace_prefill_kv_impl(cfg, n, cap, false, None)
}

/// Prefill from input embeddings (spec 049, the VLM path): identical to [`trace_prefill_kv`] but takes a
/// `vlm.input_embeds [n, hidden]` step input instead of token ids + the embedding gather. The VLM splices
/// vision tokens into the text embeddings (poot_models::vision) and feeds the result here; text decode
/// Prefill from input embeddings (spec 049, the VLM path): identical to [`trace_prefill_kv`] but takes a
/// `vlm.input_embeds [n, hidden]` step input instead of token ids + the embedding gather. The VLM splices
/// vision tokens into the text embeddings (poot_models::vision) and feeds the result here; text decode
/// then continues from the filled KV. Equivalent to `trace_prefill_kv` fed `gather(embed_tokens, tokens)`.
pub fn trace_prefill_kv_embeds(cfg: Qwen2Config, n: usize, cap: usize) -> Graph {
    trace_prefill_kv_impl(cfg, n, cap, true, None)
}

/// Qwen2.5-VL text-tower prefill from already-spliced embeddings into the fixed-capacity KV cache: the
/// mRoPE sibling of [`trace_prefill_kv_embeds`]. It consumes the same `vlm.input_embeds` `[n, hidden]`
/// step input and fills the same carried K/V state, with Q/K rotation from one packed `[3,n]`
/// `Slot::MropePosition` split through the shared sectioned-RoPE helper. The layer body is
/// Qwen2.5-VL text-tower prefill from already-spliced embeddings into the fixed-capacity KV cache: the
/// mRoPE sibling of [`trace_prefill_kv_embeds`]. It consumes the same `vlm.input_embeds` `[n, hidden]`
/// step input and fills the same carried K/V state, with Q/K rotation from one packed `[3,n]`
/// `Slot::MropePosition` split through the shared sectioned-RoPE helper. The layer body is
/// [`trace_prefill_kv_impl`].
pub fn trace_qwen2_5_vl_prefill_kv_embeds(cfg: Qwen2Config, n: usize, cap: usize) -> Graph {
    let sections = cfg
        .mrope_section
        .expect("trace_qwen2_5_vl_prefill_kv_embeds needs cfg.mrope_section");
    assert!(
        cfg.qkv_bias && !cfg.qk_norm,
        "trace_qwen2_5_vl_prefill_kv_embeds supports only the Qwen2.5-VL Qwen2 text-tower convention \
         (qkv_bias=true, qk_norm=false)"
    );
    trace_prefill_kv_impl(cfg, n, cap, true, Some(sections))
}
