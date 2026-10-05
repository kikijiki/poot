/// Reduce-only KV fit: the largest `(n_slots, cap)` whose KV pool fits `kv_budget_bytes`, given the
/// per-(slot, token) KV cost. Returns the requested config unchanged if it fits or the budget is unknown
/// (`0`); never increases either value. Reduces `cap` first (down to `min_cap`, preserving batch width);
/// only if even `min_cap` per slot overflows does it drop `n_slots`. The pool scales ~linearly with
/// `n_slots * cap`, so this is exact up to the caller's block rounding.
pub(crate) fn fit_kv_to_budget(
    n_slots: usize,
    req_cap: usize,
    bytes_per_slot_token: u64,
    kv_budget_bytes: u64,
    min_cap: usize,
) -> (usize, usize) {
    if n_slots == 0 || bytes_per_slot_token == 0 || kv_budget_bytes == 0 {
        return (n_slots, req_cap); // unknown / degenerate -> leave the requested config alone
    }
    let per_cap = bytes_per_slot_token * n_slots as u64; // bytes added per +1 cap, across all slots
    let max_cap = (kv_budget_bytes / per_cap) as usize;
    if max_cap >= req_cap {
        return (n_slots, req_cap); // fits as requested
    }
    if max_cap >= min_cap {
        return (n_slots, max_cap); // reduce cap, keep the batch width
    }
    // even min_cap per slot across all slots overflows: drop slots too, keeping cap at the floor.
    let per_slot_at_min = bytes_per_slot_token * min_cap as u64;
    let max_slots = ((kv_budget_bytes / per_slot_at_min) as usize).max(1);
    (max_slots.min(n_slots), min_cap)
}

/// Per-slot-token multiplier for [`fit_kv_to_budget`] on PTX/ROCm: 2x, versus
/// `KV_POOL_VRAM_MULTIPLIER`'s 3x on wgpu. The components:
///
/// - 1x the base pool (`caches` / `g.state`, `2*layers*kv_heads*head_dim*dtype_bytes`), on every backend.
/// - +1x `gather_shared_pool`'s kread/vread readback (`poot-models/src/qwen2.rs`): a per-k/v-per-layer
///   `[n_slots,hkv,cap,d]` tensor materialized by the shared graph and resident for the captured graph's
///   lifetime (a capture cannot free a buffer mid-capture), on every backend.
/// - No ping-pong twin. wgpu's `poot-gpu::fresh_kv_pairs` allocates a fresh same-size buffer for the
///   state pair's other parity; PTX and ROCm bind `state_out` in place onto `state_in` instead, through
///   the shared `poot_executor::Engine`'s own `inplace` commit map (`Device::in_place_donation`,
///   cards 548/549 moved this off each backend's own deleted bespoke executor), so the KV write lands
///   in the base pool.
pub(crate) const KV_POOL_VRAM_MULTIPLIER_INPLACE: u64 = 2;

/// wgpu's KV-pool VRAM price multiplier, per (layer, slot, token).
///
/// 3x, not [`KV_POOL_VRAM_MULTIPLIER_INPLACE`]'s 2x: wgpu's decode step binds `state_out` to its own
/// buffer rather than in place onto `state_in`, so the ping-pong twin is a real second copy, and the
/// final gather readback is a third.
///
/// Module-scope and `pub(crate)` so the pricing tests import it instead of keeping their own copy of
/// `3`, which would stay green if the real multiplier changed (the card 159 OOM class).
pub(crate) const KV_POOL_VRAM_MULTIPLIER: u64 = 3;

/// Closed-form price, per (layer, slot, token), for a ROCm VRAM cost that neither
/// [`KV_POOL_VRAM_MULTIPLIER_INPLACE`] nor the attn-broadcast term covers (card 269).
/// `scatter_shared_pool`/`gather_shared_pool` (`poot-models/src/qwen2.rs`, shared with qwen3next and
/// gemma4) write/read the shared KV pool as a `(0..batch)` chain of per-row `slice`/`reshape`/`gather`/
/// `transpose` ops, since the graph IR has no batched dynamic-index scatter/gather primitive. Each of the
/// `n_slots` per-row intermediates in `gather_shared_pool` is `cap`-sized and stays resident for the
/// captured graph's lifetime (`capture_decode_batched` never frees a dead buffer), so the resident cost is
/// `n_slots` separate `[cap,hkv,d]`-ish buffers, across scatter and gather, k and v, and all layers.
///
/// Measured on an MI300X (card 267): at the fitted shape `n_slots=64 cap=8755`, capture failed
/// with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. Varying only `cap` at `n_slots=64`, `cap=256` captured and
/// `cap=2000` failed 63% through with VRAM pinned at the card's limit, so it scales with `cap`.
/// Extrapolating that run and subtracting the already-priced terms gives roughly 94 KiB per (layer, slot,
/// token); this constant rounds up to ~150 KiB. It is the only calibration point; no margin pass like
/// `PTX_CAPTURE_MARGIN_NUM`/`_DEN` was run.
pub(crate) const ROCM_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN: u64 = 150_000;

/// wgpu's `batch_engine_loop` has the same unpriced shared-pool-unroll cost that
/// [`ROCM_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN`] covers on ROCm (card 269):
///
/// - `scatter_shared_pool`/`gather_shared_pool` are backend-neutral tracer functions, so wgpu's batched
///   decode graph has the same `(0..n_slots)`-unrolled `slice`/`reshape`/`gather`/`transpose`/
///   `dynamic_update_slice` chain.
/// - `GpuExecutor::build_decode_cache` runs `compute_views(&graph, Backend::SpirvVulkan)`, and
///   `is_strided_capable` (`poot-graph-plan/src/predicates.rs`) depends on op type only, so the per-row
///   ops that fail to view-promote on ROCm fail on wgpu too and plan as `PlannedKind::Compute`.
/// - `GpuExecutor::materialize` allocates a fresh persistent `ctx.alloc_f32` buffer for every non-state
///   `ValueId` a compute eqn produces, held in `decode_cache` until the graph identity or shape changes.
///
/// `KV_POOL_VRAM_MULTIPLIER = 3` prices only the base pool, the ping-pong twin and the final gather
/// result per k/v/layer.
///
/// No wgpu-specific calibration exists: driving a real allocation to its fitted boundary needs a quiet
/// shared iGPU or an isolated Vulkan GPU, and an overrun hits wgpu's uncaptured-error path
/// (`Context::alloc_f32` calls `create_buffer` with no error scope), so none was attempted. The constant
/// reuses the ROCm value, since the buffer shapes are identical; its headroom (94 KiB measured, 150 KiB
/// priced) absorbs allocator differences. It is an estimate by analogy.
pub(crate) const WGPU_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN: u64 =
    ROCM_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN;
