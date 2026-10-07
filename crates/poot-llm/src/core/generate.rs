//! Generation limits and the greedy pick shared by the sampling code.

use crate::core::sampler::{SamplerFault, check_pickable};

/// Upper bound on a single-sequence generation's KV span. The
/// single-sequence drivers size a KV allocation from it, so an unbounded `prompt_len` or `max_new`
/// would allocate multiple TB and abort, and a value near `usize::MAX` would overflow (release builds
/// skip the check) into an undersized buffer and an out-of-bounds GPU write. A sanity ceiling
/// (256K), not a per-model context bound: a near-ceiling span may still not fit the device.
pub(crate) const MAX_PREFILL_CAP: usize = 262_144;

/// The index of the largest logit (the first on a tie). A [`SamplerFault`] when `v` holds a NaN or `+inf`, or no
/// finite element: an unchecked maximum turns an all-NaN row into token 0.
pub(crate) fn argmax(v: &[f32]) -> Result<usize, SamplerFault> {
    check_pickable(v)?;
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    Ok(best)
}
