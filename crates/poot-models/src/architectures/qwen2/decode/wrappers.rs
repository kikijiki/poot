use super::*;

/// Trace a constant-shape fixed-capacity decode: an identical graph for every position, so one captured
/// graph replays the whole decode. Unlike the growing-KV [`trace_decode`], this takes only `cap`: the KV slot write uses the runtime `Pos` slot
/// (`dynamic_update_slice_dyn`), and attention reads the full `cap` cache with an additive `Mask` slot
/// (`[cap]`, 0 for `t <= pos`, -inf for `t > pos`). Arch-general (qwen2 bias / qwen3 QK-norm). Output is
/// the logits; state is the `2*layers` cache pairs (input-output aliased). Uses decomposed attention.
pub fn trace_decode_kv_masked(cfg: Qwen2Config, cap: usize) -> Graph {
    trace_decode_kv_masked_impl(cfg, cap)
}

/// Batched masked single-token decode (spec 027): [`trace_decode_kv_masked`] generalized to `batch` rows
/// that each advance one token per forward at independent positions over independent KV caches. Same
/// decomposition over a leading batch axis B: a `[B]` token slot, a `[B]` absolute position slot, a
/// `[B, cap]` additive mask, and `[B, n_kv_heads, cap, head_dim]` KV caches. Ordinary per-row RoPE uses
/// [`rope_batched`]. When `cfg.mrope_section` is set, an axis-major I32 `[3,B]` `Slot::MropePosition`
/// supplies Q/K rotation through [`rope_sectioned_batched`], while the absolute `[B]` position still
/// drives KV addressing and masks. The per-row KV scatter is B unrolled `dynamic_update_slice` writes
/// concatenated back to `[B,...]`; everything else broadcasts over the leading axis. Each row's logits
/// depend only on that row's inputs, so the result equals B independent batch-1 decodes.
pub fn trace_decode_kv_masked_batched(cfg: Qwen2Config, cap: usize, batch: usize) -> Graph {
    trace_decode_kv_masked_batched_impl(cfg, cap, batch)
}
