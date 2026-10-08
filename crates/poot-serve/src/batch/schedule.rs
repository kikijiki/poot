/// The `max_num_batched_tokens` budget of one step's prefill work. It bounds how much prefill one
/// iteration absorbs so a burst of long prompts cannot stall the decoding rows. Each pending prefill row
/// costs the tokens its next piece feeds; one that does not fit waits for a later step.
///
/// Decoding rows reserve one token each off the budget first, so decode never starves. `pending` is
/// `(row_index, piece_tokens)` in FIFO order. Returns the row indices to step, a best-effort bin-pack in
/// `pending` order: a big piece that does not fit is skipped without blocking a smaller one behind it.
/// Pure, so it is testable without a device.
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
