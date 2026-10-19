//! Card 551b: the sampling suffix the driver appends after a decode
//! graph's raw logits, so every backend samples through the same `OpKind::SampleToken` contract path
//! (R472-007) instead of a per-backend fused entry point. Card 734 moved this file here from `core/`.
//!
//! [`with_head`] appends the suffix via card 551a's `Builder::resume` + `sample_head`
//! (`poot_graph_ir::ops::sampling`); [`Head`] names which suffix (if any) a compiled decode
//! entry carries; [`rule_of`] gates which requests the suffix can express (R-551b-2: penalties, bias,
//! logprobs and a guided-decoding constraint still need the host path); [`SuffixRows`] derives one
//! device row's `seed`/`params`/`top_k` from a [`Sampler`]'s knobs; [`read_tokens`] turns the suffix's
//! `[rows, 2]` I32 readback into a token or Card 601's typed fault. The host draw is not here: [`Sampler::pick`]
//! is a plain exact function (ADR-0114 amendment), and a device row is held to it by the nucleus-mass tolerance.

use poot_graph_ir::builder::Resumed;
use poot_graph_ir::op::SampleRule;
use poot_graph_ir::ops::sampling::sample_head;
use poot_graph_ir::{Builder, BuilderError, Graph, Outputs, TraceLimits};

use crate::core::sampler::{Sampler, SamplerFault};

/// Which suffix (if any) a compiled entry carries after its raw logits. `Logits` is the raw graph:
/// the host sampling path, a sampled speculative window's verification and the VLM caption path,
/// which need the full logits, never a suffix. `Sample(rule)` is [`with_head`]'s
/// appended suffix, read back as `[rows, 2]` I32 `(token, non_finite_index)` through [`read_tokens`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Head {
    /// The raw logits graph, with no sampling suffix.
    Logits,
    /// The graph with the sampling suffix of this rule appended.
    Sample(SampleRule),
}

impl Head {
    /// The suffix of a greedy request: the on-device argmax.
    pub const GREEDY: Head = Head::Sample(SampleRule::Greedy);
}

/// Append `head`'s suffix after `g`'s raw logits output (card 551a): reopens the finished graph
/// (`Builder::resume`), appends [`sample_head`]'s suffix, and finishes again over the same state
/// pairs. The resumed builder keeps every witness `g` declares, and the suffix only appends values,
/// so a validation-bearing graph keeps its checks. `Logits` is a no-op (the raw graph, unchanged).
pub(crate) fn with_head(g: Graph, head: Head, limits: TraceLimits) -> Result<Graph, BuilderError> {
    match head {
        Head::Logits => Ok(g),
        Head::Sample(rule) => {
            let Resumed { builder, outputs } = Builder::resume(g, limits)?;
            let sh = sample_head(&builder, outputs.result, rule)?;
            builder.finish(Outputs {
                result: sh.out,
                ..outputs
            })
        }
    }
}

/// Card 551b: `None` when `sampler`'s request needs the host path (a penalty,
/// bias, logprob recording or a guided-decoding constraint - anything [`Sampler::is_greedy`]/
/// [`Sampler::is_simple_temperature`] reject), `Some(rule)` otherwise. Greedy always maps to
/// [`SampleRule::Greedy`] (SC-003 needs the device suffix's non-finite readback on the greedy path
/// too, not just sampled); a simple temperature draw maps through [`SampleRule::for_truncation`].
pub(crate) fn rule_of(sampler: &Sampler) -> Option<SampleRule> {
    if sampler.is_greedy() {
        return Some(SampleRule::Greedy);
    }
    if !sampler.is_simple_temperature() {
        return None;
    }
    Some(SampleRule::for_truncation(sampler.top_k, sampler.top_p))
}

/// This row's `params` column vector (card 551a: `[inv_temp, floor_offset, noise_scale]`, `+top_p`
/// for [`SampleRule::GumbelTopKTopP`]), read by [`SuffixRows::for_seed`] (a device row). `floor_offset` folds min-p into a
/// logit floor in the *raw* (unscaled) logit space `sample_one_row` compares against: keeping index
/// `i` needs `logits[i] >= max + floor_offset`, i.e. `exp((logits[i]-max)/T) >= min_p`, i.e.
/// `floor_offset = T * ln(min_p)` (`-inf` disables the floor when `min_p <= 0`). `noise_scale` is
/// always 1 (the ordinary Gumbel-max trick: unbiased only when the noise rides the temperature-scaled
/// logits unscaled itself).
fn row_params(rule: SampleRule, temperature: f32, min_p: f32, top_p: f32) -> Vec<f32> {
    let inv_temp = 1.0 / temperature;
    let floor_offset = if min_p > 0.0 {
        temperature * min_p.ln()
    } else {
        f32::NEG_INFINITY
    };
    let mut params = vec![inv_temp, floor_offset, 1.0f32];
    if rule.has_top_p() {
        params.push(top_p);
    }
    params
}

/// A device suffix row's `seed`/`params`/`top_k` for one decode step (card 551a R-551a-4; R-551b-2):
/// built from a [`Sampler`]'s current knobs and its next [`Sampler::next_device_seed`] draw.
pub(crate) struct SuffixRows {
    pub(crate) seed: i32,
    pub(crate) params: Vec<f32>,
    pub(crate) top_k: Option<i32>,
}

impl SuffixRows {
    /// `rule` must be a non-`Greedy` rule [`rule_of`] returned for `sampler` (`Greedy` declares no
    /// seed/params/top_k inputs at all, so there is nothing for this to fill).
    pub(crate) fn push(sampler: &mut Sampler, rule: SampleRule) -> SuffixRows {
        let seed = sampler.next_device_seed() as i32;
        Self::for_seed(sampler, rule, seed)
    }

    /// Like [`Self::push`] but for a decode step whose output will be discarded (card 551b:
    /// `generate_kv_gpu_cached_sampled`'s cached re-encode loop steps every prompt position through
    /// the same entry before it reaches the position it must actually pick at). `seed = 0` is
    /// arbitrary (the output is never read), and - unlike `push` - this does not advance
    /// `sampler`'s rng stream, so the eventual real pick still draws the same sequence a loop with no
    /// throwaway positions would.
    pub(crate) fn placeholder(sampler: &Sampler, rule: SampleRule) -> SuffixRows {
        Self::for_seed(sampler, rule, 0)
    }

    fn for_seed(sampler: &Sampler, rule: SampleRule, seed: i32) -> SuffixRows {
        debug_assert_ne!(
            rule,
            SampleRule::Greedy,
            "SuffixRows: Greedy declares no sampler-slot inputs to fill"
        );
        let params = row_params(rule, sampler.temperature, sampler.min_p, sampler.top_p);
        let top_k = rule.has_top_k().then_some(sampler.top_k as i32);
        SuffixRows {
            seed,
            params,
            top_k,
        }
    }
}

/// Read a decode step's `Sample(rule)` output: the `[2]` I32 `(token, non_finite_index)` row card
/// 551a's suffix always produces for a single-row decode step. `non_finite_index >= 0` is Card 601's
/// typed fault (R-551a-2: the device readback carries the index, never the value - `value: None`);
/// no token is committed for that step.
pub(crate) fn read_tokens(bytes: &[u8]) -> std::result::Result<u32, SamplerFault> {
    let ints: &[i32] = bytemuck::cast_slice(bytes);
    debug_assert_eq!(
        ints.len(),
        2,
        "a decode step's sampling suffix is always exactly one row"
    );
    let (token, non_finite) = (ints[0], ints[1]);
    if non_finite >= 0 {
        return Err(SamplerFault::NonFiniteLogit {
            index: non_finite as usize,
            value: None,
        });
    }
    Ok(token as u32)
}
