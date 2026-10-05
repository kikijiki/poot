use std::sync::mpsc::Sender;
use std::time::Instant;

use anyhow::{Result, anyhow};
use poot_llm::driver::block_table::PagedKvCache;
use poot_llm::driver::prefix_cache::PrefixIdentity;
use poot_llm::{LoraAdapterLease, MropeDecodeState, Sampler};

use crate::batch::slot::Slot;
use crate::types::GenEvent;

// Recompute preemption (spec 237): when KV-pool exhaustion blocks an admission even after evicting unused
// prefixes, preempt a running slot (free its KV blocks) and requeue it instead of failing the admission.
// A preempted request's KV is never copied to host memory ("swap" is out of scope). An ordinary request
// registers into the shared content-hash prefix cache under its effective-weights identity and resumes
// through `admit_prefix_for`. An mRoPE request cannot use that token-only identity safely: it resets
// without registration and resumes from position zero through a fresh reservation.

/// Which active slot to preempt when KV-pool exhaustion blocks an admission: the one with the fewest
/// tokens generated (ties: lowest slot index), which minimizes discarded work. `generated` carries
/// across a preemption cycle, so a request already preempted once does not look cheap again, which
/// avoids bouncing the same requests under sustained overload (not a full fairness guarantee; see spec
/// 237's known limitations).
///
/// Every active slot qualifies: `tokens[..pos]` is always the resident, valid KV span, which is what
/// registration relies on (registering beyond it would poison a future prefix hit). mRoPE victims are
/// reset without registration because their three-axis rotary identity is not in the token-only cache key.
pub(crate) fn select_preemption_victim(slots: &[Option<Slot>]) -> Option<usize> {
    slots
        .iter()
        .enumerate()
        .filter_map(|(j, s)| s.as_ref().map(|sl| (j, sl.generated)))
        .min_by_key(|&(j, generated)| (generated, j))
        .map(|(j, _)| j)
}

/// A slot preempted by `select_preemption_victim`, carrying what is needed to resume it: the full token
/// history (`Slot::tokens`), the original `prompt_len`/`max_new` (stop-sequence and budget accounting
/// stay unchanged), `generated` (victim selection sees true progress across repeated preemptions), the
/// live `Sampler` (moved, so RNG state, penalty history, guided-decoding state and logprobs are kept),
/// and the same reply channel (the client just sees a longer gap).
pub(crate) struct PreemptedSeq {
    pub(crate) tokens: Vec<u32>,
    pub(crate) prompt_len: usize,
    pub(crate) generated: usize,
    pub(crate) max_new: usize,
    pub(crate) sampler: Sampler,
    pub(crate) stop: Vec<String>,
    pub(crate) stream: bool,
    pub(crate) reply: Sender<GenEvent>,
    pub(crate) decode_start: Option<Instant>,
    pub(crate) last_token: Option<Instant>,
    /// See [`Slot::lora_adapter`]'s doc comment - carried across preemption so a resumed request keeps
    /// its adapter selection.
    pub(crate) lora_adapter: LoraAdapterLease,
    /// Round-trips the request-owned rotary source through recompute preemption.
    pub(crate) mrope: Option<MropeDecodeState>,
}

/// Preempt slot `victim` (recompute mode) and return its state for requeue. An ordinary slot registers
/// its resident `tokens[..pos]` span in the identity-scoped prefix cache (`register_and_reset_for` pins
/// only whole `BLOCK_SIZE` blocks, so a partial tail is freed). An mRoPE slot resets without registration,
/// since identical tokens can have different three-axis positions and so incompatible rotated K; its
/// resume recomputes the full history from position zero.
pub(crate) fn preempt_slot(
    slots: &mut [Option<Slot>],
    paged: &mut PagedKvCache,
    victim: usize,
) -> PreemptedSeq {
    let slot = slots[victim]
        .take()
        .expect("preemption victim must be an active slot");
    if slot.mrope.is_some() {
        // Prefix identity is token-only. An mRoPE slot's rotated K also depends on its three prompt
        // axes, so registering it would let identical tokens with different axes alias incompatible KV.
        paged.reset(victim);
    } else {
        // Register under the slot's own effective-weights identity (base, or the request's adapter), so
        // a later request can only reuse this K/V under the same weights (spec 248, ADR-0030).
        let identity = slot.lora_adapter.prefix_identity();
        paged.register_and_reset_for(victim, &slot.tokens[..slot.pos], identity);
    }
    PreemptedSeq {
        tokens: slot.tokens,
        prompt_len: slot.prompt_len,
        generated: slot.generated,
        max_new: slot.max_new,
        sampler: slot.sampler,
        stop: slot.stop,
        stream: slot.stream,
        reply: slot.reply,
        decode_start: slot.decode_start,
        last_token: slot.last_token,
        lora_adapter: slot.lora_adapter,
        mrope: slot.mrope,
    }
}

/// Reserve a request without consulting the token-only prefix cache, escalating through the same
/// eviction and recompute-preemption tiers as ordinary admission. Used by mRoPE because its KV identity
/// includes position axes that `PagedKvCache` does not key today.
pub(crate) fn reserve_without_prefix_with_preemption(
    slots: &mut [Option<Slot>],
    paged: &mut PagedKvCache,
    j: usize,
    total_tokens: usize,
) -> (Result<()>, Vec<PreemptedSeq>) {
    if paged.append(j, total_tokens).is_ok() {
        return (Ok(()), Vec::new());
    }
    paged.evict_prefix();
    if paged.append(j, total_tokens).is_ok() {
        return (Ok(()), Vec::new());
    }
    let mut requeued = Vec::new();
    while let Some(victim) = select_preemption_victim(slots) {
        requeued.push(preempt_slot(slots, paged, victim));
        paged.evict_prefix();
        if paged.append(j, total_tokens).is_ok() {
            return (Ok(()), requeued);
        }
    }
    (
        Err(anyhow!(
            "no free KV blocks even after preempting every other active slot"
        )),
        requeued,
    )
}

/// Reserve KV blocks for `tokens` (a prompt, or a resumed sequence's full history) into free slot `j`,
/// escalating through retry tiers: (1) plain `admit_prefix_for`; (2) evict unused prefixes (card 046c);
/// (3) preempt active slots one at a time (`select_preemption_victim`), calling `evict_prefix` after each
/// (a fresh preemption's blocks are pinned in the prefix cache by `register_and_reset_for`, not
/// immediately free) and retrying `admit_prefix_for` until it succeeds or no victim remains. `identity`
/// is the admitting request's effective-weights identity (base, or its adapter), so only K/V computed
/// under the same weights matches (spec 248, ADR-0030). Returns the `admit_prefix_for` result (the
/// `skip`) and every `PreemptedSeq` produced (empty unless tier 3 ran) for the caller's requeue.
///
/// Tier 3 removes one active slot per iteration, so it runs at most `slots.len()` times.
pub(crate) fn admit_prefix_with_preemption(
    slots: &mut [Option<Slot>],
    paged: &mut PagedKvCache,
    j: usize,
    tokens: &[u32],
    max_new: usize,
    identity: PrefixIdentity,
) -> (Result<usize>, Vec<PreemptedSeq>) {
    if let Ok(skip) = paged.admit_prefix_for(j, tokens, max_new, identity) {
        return (Ok(skip), Vec::new());
    }
    paged.evict_prefix();
    if let Ok(skip) = paged.admit_prefix_for(j, tokens, max_new, identity) {
        return (Ok(skip), Vec::new());
    }
    let mut requeued = Vec::new();
    while let Some(victim) = select_preemption_victim(slots) {
        requeued.push(preempt_slot(slots, paged, victim));
        // the victim's now-registered blocks are pinned by the cache, not yet free - reclaim any
        // cache-exclusive blocks (this victim's and any other stale registration) before retrying.
        paged.evict_prefix();
        if let Ok(skip) = paged.admit_prefix_for(j, tokens, max_new, identity) {
            return (Ok(skip), requeued);
        }
    }
    (
        Err(anyhow!(
            "no free KV blocks even after preempting every other active slot"
        )),
        requeued,
    )
}

/// Resume a `PreemptedSeq` into free slot `j`. An ordinary request matches its full token history
/// against the token-only prefix cache and reconstructs the slot at that `skip`. An mRoPE request never
/// queries that cache: it reserves fresh storage, seeks its cursor to zero and recomputes the whole
/// history. Both reserve only the remaining budget (`max_new - generated`, floored at 1 defensively),
/// since generated tokens are already in `tokens`.
///
/// `prefilled` is always `true`: resumption takes the token-by-token replay path, never the one-shot
/// fast-prefill. That forward writes KV over `tokens[..Slot::prompt_len]`, which for a resumed slot would
/// conflate the true prompt boundary (needed for the stream/stop-sequence cut, restored from
/// `PreemptedSeq::prompt_len`) with the history to recompute (`tokens.len()`, usually longer). Keeping
/// fast-prefill to fresh admissions costs the fast-prefill speedup only on a resumed sequence's
/// recomputed span (spec 237's future work).
///
/// On reservation failure returns `Err(preempted)`, the value handed in (an mRoPE cursor is restored if
/// a defensive seek fails), not an `anyhow::Error`: `admit_prefix` and `append` are all-or-nothing, and
/// the caller needs the value back to requeue it (its `Sampler`, reply channel and history are not
/// `Clone`), otherwise the request would be silently dropped.
// `PreemptedSeq` is large, but `resume_preempted` runs only under KV contention at admission, never
// per token, so boxing it is not worth it.
#[allow(clippy::result_large_err)]
pub(crate) fn resume_preempted(
    paged: &mut PagedKvCache,
    j: usize,
    preempted: PreemptedSeq,
) -> std::result::Result<Slot, PreemptedSeq> {
    let remaining = preempted.max_new.saturating_sub(preempted.generated).max(1);
    let mut preempted = preempted;
    let skip = if preempted.mrope.is_some() {
        if paged.append(j, preempted.tokens.len() + remaining).is_err() {
            return Err(preempted);
        }
        0
    } else {
        let identity = preempted.lora_adapter.prefix_identity();
        match paged.admit_prefix_for(j, &preempted.tokens, remaining, identity) {
            Ok(skip) => skip,
            Err(_) => return Err(preempted),
        }
    };
    if let Some(mrope) = &mut preempted.mrope
        && mrope.seek(skip).is_err()
    {
        paged.reset(j);
        return Err(preempted);
    }
    Ok(Slot {
        tokens: preempted.tokens,
        prompt_len: preempted.prompt_len,
        pos: skip,
        generated: preempted.generated,
        max_new: preempted.max_new,
        sampler: preempted.sampler,
        stop: preempted.stop,
        stream: preempted.stream,
        reply: preempted.reply,
        prefilled: true,
        decode_start: preempted.decode_start,
        last_token: preempted.last_token,
        lora_adapter: preempted.lora_adapter,
        mrope: preempted.mrope,
    })
}
