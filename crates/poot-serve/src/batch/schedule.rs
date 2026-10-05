use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, TryRecvError};

/// Fill free slots from the engine input without blocking: hand queued jobs to `admit` while any slot
/// is free, and stop at the first `Empty` (input stays open) or `Disconnected`. On disconnect the
/// flag is set after the queue drained, so jobs that arrived before the close are admitted first and
/// the loop reports closed only when it next goes fully idle. Per-backend `admit` work (encoding, KV
/// reservation, capture-epoch bumps) stays at the call site.
pub(crate) fn drain_ready_jobs<J, S>(
    rx: &Receiver<J>,
    slots: &mut [Option<S>],
    input_closed: &mut bool,
    mut admit: impl FnMut(&mut [Option<S>], J),
) {
    while slots.iter().any(|s| s.is_none()) {
        match rx.try_recv() {
            Ok(job) => admit(slots, job),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                *input_closed = true;
                break;
            }
        }
    }
}

/// Resume preempted sequences from the front of `preempted` into free slots, before the loop pulls
/// new jobs, so a victim keeps its place ahead of newer work. On the first shortfall the sequence
/// goes back to the FRONT of the queue (its place is kept, nothing is dropped) and the step stops:
/// `break`, not `continue`, so a short pool cannot spin on the same (slot, sequence) pair. An empty
/// queue with a free slot also ends the step. Backend-specific effects of one attempt (logging,
/// capture-epoch bumps) belong inside the `resume` closure, which returns the slot on success and
/// the sequence back on failure.
pub(crate) fn resume_into_free_slots<Q, S>(
    slots: &mut [Option<S>],
    preempted: &mut VecDeque<Q>,
    mut resume: impl FnMut(usize, Q) -> std::result::Result<S, Q>,
) {
    while let Some(free_j) = slots.iter().position(|s| s.is_none()) {
        let Some(seq) = preempted.pop_front() else {
            break;
        };
        match resume(free_j, seq) {
            Ok(slot) => {
                slots[free_j] = Some(slot);
            }
            Err(seq) => {
                preempted.push_front(seq);
                break;
            }
        }
    }
}

/// A `max_num_batched_tokens` token-budget scheduler for the fast paged prefill admission step (spec 233
/// Phase 1). It bounds how much prefill work one iteration absorbs so a burst of long-prompt admissions
/// cannot stall concurrent decode slots. Each pending admission costs `prompt_len` tokens (or the next
/// chunk's size when chunked); one that does not fit is deferred to a later iteration (the caller leaves
/// `Slot::prefilled == false`).
///
/// Active decode slots reserve 1 token each off the budget first, so decode never starves. `pending` is
/// `(slot_index, prompt_len)` in FIFO order (slots admit into the lowest free index). Returns the slot
/// indices to admit this iteration, a best-effort bin-pack in `pending` order: a big prompt that does not
/// fit is skipped without blocking a smaller one behind it. Pure, so it is testable without a device.
pub(crate) fn schedule_prefill_admissions(
    pending: &[(usize, usize)],
    active_decode_slots: usize,
    max_num_batched_tokens: usize,
) -> Vec<usize> {
    let mut budget = max_num_batched_tokens.saturating_sub(active_decode_slots);
    let mut admitted = Vec::with_capacity(pending.len());
    for &(j, prompt_len) in pending {
        if prompt_len <= budget {
            admitted.push(j);
            budget -= prompt_len;
        }
        // else: defer (do not break - a later, smaller pending prompt may still fit this iteration).
    }
    admitted
}
