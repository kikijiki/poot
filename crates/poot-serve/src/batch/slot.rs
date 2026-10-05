use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use poot_llm::{LoraAdapterLease, MropeDecodeState, MropePosition, Sampler};

use crate::batch::preempt::PreemptedSeq;
use crate::metrics::Metrics;
use crate::types::{GenEvent, Job};

/// An active decode slot in the batched engine: a request's running token sequence + its independent
/// position, plus its sampler/stop/streaming and the reply channel to its connection thread.
pub(crate) struct Slot {
    pub(crate) tokens: Vec<u32>, // prompt + generated so far
    pub(crate) prompt_len: usize,
    pub(crate) pos: usize,
    pub(crate) generated: usize,
    pub(crate) max_new: usize,
    pub(crate) sampler: Sampler,
    pub(crate) stop: Vec<String>,
    pub(crate) stream: bool,
    pub(crate) reply: Sender<GenEvent>,
    /// A freshly admitted slot (no reused prefix) is fast-prefilled: its whole prompt KV is filled in one
    /// paged forward before its first decode step. Set once the slot has been handled (prefilled or skipped).
    pub(crate) prefilled: bool,
    /// Set on the first emitted token; `cumulative_ms` in `PerTokenInfo` is measured from it (excludes
    /// prefill and queue wait).
    pub(crate) decode_start: Option<Instant>,
    /// Wall time of the last emitted token; used for `since_prev_ms`.
    pub(crate) last_token: Option<Instant>,
    /// This request's immutable LoRA adapter selection and lifetime ownership. The index is read into the
    /// batched step's `[n_slots]` vector every iteration. The same handle moves through
    /// [`PreemptedSeq::lora_adapter`] during recompute preemption, so the selected pool slot cannot be
    /// unloaded or reused until this request leaves the engine.
    pub(crate) lora_adapter: LoraAdapterLease,
    /// Typed rotary cursor owned by this request. `None` is the ordinary plain-RoPE path.
    pub(crate) mrope: Option<MropeDecodeState>,
}

/// Move one validated job into its active-slot representation. All device admission loops use this so
/// the request-owned adapter lease cannot be replaced by a backend-local index or inert handle.
pub(crate) fn admit_job_to_slot(
    job: Job,
    tokens: Vec<u32>,
    pos: usize,
    max_new: usize,
    prefilled: bool,
    mrope: Option<MropeDecodeState>,
) -> Slot {
    Slot {
        prompt_len: tokens.len(),
        tokens,
        pos,
        generated: 0,
        max_new,
        sampler: job.sampler,
        stop: job.stop,
        stream: job.stream,
        reply: job.reply,
        prefilled,
        decode_start: None,
        last_token: None,
        lora_adapter: job.lora_adapter,
        mrope,
    }
}

/// Reject work before admission. The job is consumed so its adapter lease is released whether or not
/// the failure event is delivered.
pub(crate) fn reject_job(job: Job, metrics: &Metrics, reason: impl Into<String>) {
    let _ = job.reply.send(GenEvent::failed(metrics, reason));
}

pub(crate) fn fail_active_slots(slots: &mut [Option<Slot>], metrics: &Metrics, reason: &str) {
    for slot in slots.iter_mut().filter_map(Option::take) {
        let _ = slot.reply.send(GenEvent::failed(metrics, reason));
    }
}

/// Finalize every request still owned by a batch loop when its input disconnects. The receiver is
/// drained too, so leases on work that never reached admission are released like active and preempted ones.
pub(crate) fn shutdown_batch_requests(
    rx: &Receiver<Job>,
    slots: &mut [Option<Slot>],
    preempted: Option<&mut VecDeque<PreemptedSeq>>,
    metrics: &Metrics,
    reason: &str,
) {
    while let Ok(job) = rx.try_recv() {
        reject_job(job, metrics, reason);
    }
    fail_active_slots(slots, metrics, reason);
    if let Some(preempted) = preempted {
        for seq in preempted.drain(..) {
            let _ = seq.reply.send(GenEvent::failed(metrics, reason));
        }
    }
}

/// Build the typed mRoPE row vector for one fixed-width decode step. Free rows get an inert tuple;
/// active rows must own a cursor synchronized to their scheduler token index. Read-only.
pub(crate) fn build_mrope_rows(
    slots: &[Option<Slot>],
    required: bool,
) -> Result<Option<Vec<MropePosition>>> {
    if !required {
        if slots.iter().flatten().any(|slot| slot.mrope.is_some()) {
            bail!("typed mRoPE scheduler state is attached to a plain-RoPE decode graph");
        }
        return Ok(None);
    }

    let mut rows = Vec::with_capacity(slots.len());
    for (j, slot) in slots.iter().enumerate() {
        match slot {
            None => rows.push(MropePosition::collapsed(0)),
            Some(slot) => {
                let state = slot
                    .mrope
                    .as_ref()
                    .ok_or_else(|| anyhow!("active mRoPE slot {j} has no typed decode state"))?;
                if state.token_index() != slot.pos {
                    bail!(
                        "active mRoPE slot {j} cursor {} does not match scheduler token index {}",
                        state.token_index(),
                        slot.pos
                    );
                }
                rows.push(state.current());
            }
        }
    }
    Ok(Some(rows))
}

/// Advance a slot's absolute KV cursor and typed rotary cursor as one scheduler transition.
pub(crate) fn advance_slot_position(slot: &mut Slot) -> Result<()> {
    let next_pos = slot
        .pos
        .checked_add(1)
        .ok_or_else(|| anyhow!("slot position overflow"))?;
    if let Some(mrope) = &mut slot.mrope {
        mrope
            .advance()
            .map_err(|e| anyhow!("mRoPE scheduler advance: {e}"))?;
        if mrope.token_index() != next_pos {
            bail!(
                "mRoPE scheduler cursor {} did not advance to absolute token index {next_pos}",
                mrope.token_index()
            );
        }
    }
    slot.pos = next_pos;
    Ok(())
}
