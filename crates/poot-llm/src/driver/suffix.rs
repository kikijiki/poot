//! The sampling suffix the driver appends after a decode graph's raw logits, so every backend samples
//! through the same `OpKind::SampleToken` contract path (R472-007) instead of a per-backend fused entry
//! point. Card 734 moved this file here from `core/`.
//!
//! Everything a request asks of its sampler runs in that suffix (POOT-1268): the draw, the penalties and
//! logit bias read from a per-sequence history kept on the device, a guided-decoding mask, and top-n
//! logprobs. [`with_head`] appends the suffix via `Builder::resume` + `sample_head`
//! (`poot_graph_ir::ops::sampling`); [`Head`] names which suffix (if any) a compiled decode entry carries;
//! [`need_of`] and [`narrowest`] pick, from a step's rows, the prepared head that covers all of them;
//! [`DrawRow`] derives one device row's seed, knobs and top-k from a [`Sampler`]'s knobs; [`read_pick`]
//! turns the suffix's readback into a token (and its logprobs) or Card 601's typed fault. The host draw is
//! not here: [`Sampler::pick`] is a plain exact function (ADR-0114 amendment) that the device rows are
//! held to by the nucleus-mass tolerance, and the only head that still needs it is the logits head of a
//! caller that asked for the full logits.

use poot_graph_ir::builder::Resumed;
use poot_graph_ir::op::SampleRule;
use poot_graph_ir::ops::sampling::{Adjust, LogprobRecord, SampleSpec, sample_head};
use poot_graph_ir::{BoundaryId, Builder, BuilderError, Graph, Outputs, TraceLimits};
use poot_models::model::{KvLayout, StepShape};
use std::num::NonZeroUsize;

use crate::core::sampler::{Sampler, SamplerFault, TokenLogprob};

/// Which suffix (if any) a compiled entry carries after its raw logits. `Logits` is the raw graph: the
/// full logits for a caller that asked for them - a sampled speculative window's verification, a beam
/// search, the VLM caption path - never a sampling suffix. `Sample(spec)` is [`with_head`]'s appended
/// suffix, read back through [`read_pick`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Head {
    /// The raw logits graph, with no sampling suffix.
    Logits,
    /// The graph with the sampling suffix of this spec appended.
    Sample(SampleSpec),
}

impl Head {
    /// The suffix of a greedy request: the on-device argmax.
    pub const GREEDY: Head = Head::Sample(SampleSpec::draw(SampleRule::Greedy));

    /// The narrowest head that serves `sampler`'s request exactly.
    #[must_use]
    pub fn serving(sampler: &Sampler) -> Head {
        Head::Sample(need_of(sampler))
    }
}

/// The rows of the history pool an adjusted head declares: one per sequence under a paged layout, the
/// step's rows under the contiguous one (the sequence slots that layout serves).
fn pool_rows(shape: StepShape) -> NonZeroUsize {
    match shape.kv {
        KvLayout::Paged { seqs, .. } => seqs,
        KvLayout::Contiguous => shape.rows,
    }
}

/// Append `head`'s suffix after `g`'s raw logits output: reopens the finished graph
/// (`Builder::resume`), appends [`sample_head`]'s suffix, and finishes again over the same state pairs
/// (and the sampler's history pair when the head adjusts). The resumed builder keeps every witness `g`
/// declares, and the suffix only appends values, so a validation-bearing graph keeps its checks. `Logits`
/// is a no-op (the raw graph, unchanged).
///
/// A guided head exports the raw logits as the boundary the compile cuts at ([`CompileOptions::with_late`]):
/// the stages before the cut run on the device while the host builds the mask the suffix adds. A head
/// whose record is the I32 `(token, flag)` pair exports the picked token `[rows, 1]` as the value a chained
/// step carries into the next step's token slot.
pub(crate) fn with_head(
    g: Graph,
    head: Head,
    shape: StepShape,
    limits: TraceLimits,
) -> Result<HeadGraph, BuilderError> {
    match head {
        Head::Logits => Ok(HeadGraph {
            graph: g,
            late: None,
            picked: None,
        }),
        Head::Sample(spec) => {
            let Resumed { builder, outputs } = Builder::resume(g, limits)?;
            let pool = (spec.adjust != Adjust::None).then(|| pool_rows(shape));
            let raw = outputs.result;
            let sh = sample_head(&builder, raw, spec, pool)?;
            let mut outputs = Outputs {
                result: sh.out,
                ..outputs
            };
            if let Some(pair) = sh.history {
                outputs = outputs.with_state(&[pair]);
            }
            let late = (spec.adjust == Adjust::Guided).then(|| BoundaryId(raw.id()));
            if let Some(boundary) = late {
                outputs = outputs.with_export(boundary, raw);
            }
            // The pair's token column, shaped like the token slot `[rows, 1]` it feeds: a step with one
            // pick per row (a verify window has several and does not chain).
            let pair = sh.out;
            let pair_shape = builder.aval(pair)?.shape;
            let one_pick = matches!(pair_shape.as_slice(), [_, 2] | [_, 1, 2]);
            let picked = if spec.logprobs.is_none() && late.is_none() && one_pick {
                let rows = pair_shape[0];
                let last = pair_shape.len() - 1;
                let token = builder.reshape(builder.slice(pair, last, 0, 1)?, vec![rows, 1])?;
                let boundary = BoundaryId(token.id());
                outputs = outputs.with_export(boundary, token);
                Some(boundary)
            } else {
                None
            };
            Ok(HeadGraph {
                graph: builder.finish(outputs)?,
                late,
                picked,
            })
        }
    }
}

/// A graph with its sampling suffix appended, and the values the suffix exports.
pub(crate) struct HeadGraph {
    pub(crate) graph: Graph,
    /// The raw logits a guided head exports: where the compile cuts so the host can build the mask late.
    pub(crate) late: Option<BoundaryId>,
    /// The picked token `[rows, 1]` a chainable head exports.
    pub(crate) picked: Option<BoundaryId>,
}

/// What one row asks of a head: the narrowest spec that serves its sampler.
pub(crate) fn need_of(sampler: &Sampler) -> SampleSpec {
    SampleSpec {
        rule: sampler.draw_rule(),
        adjust: sampler.adjustment(),
        logprobs: sampler.logprob_count(),
    }
}

/// The spec that serves both `a` and `b`: the wider draw, an adjustment if either needs one, the more
/// alternatives.
pub(crate) fn join(a: SampleSpec, b: SampleSpec) -> SampleSpec {
    SampleSpec {
        rule: a.rule.max(b.rule),
        adjust: a.adjust.max(b.adjust),
        logprobs: a.logprobs.max(b.logprobs),
    }
}

/// Whether a head of `head`'s spec serves a request of `need`. A narrower rule's knobs are the wider
/// rule's neutral values (top-k 0, top-p 1, a greedy row's noise scale 0), an unadjusted row is the
/// adjusted head's neutral row, and a row that wants fewer alternatives reads a prefix of the record.
pub(crate) fn covers(head: SampleSpec, need: SampleSpec) -> bool {
    head.rule >= need.rule && head.adjust >= need.adjust && head.logprobs >= need.logprobs
}

/// The narrowest of `prepared` that serves `need`: the cheapest draw, then no adjustment before an
/// adjusted one, then fewest alternatives. `None` when no prepared head serves it.
pub(crate) fn narrowest(prepared: &[SampleSpec], need: SampleSpec) -> Option<SampleSpec> {
    prepared
        .iter()
        .copied()
        .filter(|&head| covers(head, need))
        .min_by_key(|head| (head.rule, head.adjust, head.logprobs))
}

/// This row's `params` column vector (card 551a: `[inv_temp, floor_offset, noise_scale]`, `+top_p` for a
/// head whose rule has the top-p cut), read by [`DrawRow`]. `floor_offset` folds min-p into a logit floor
/// in the *raw* (unscaled) logit space `sample_one_row` compares against: keeping index `i` needs
/// `logits[i] >= max + floor_offset`, i.e. `exp((logits[i]-max)/T) >= min_p`, i.e. `floor_offset = T *
/// ln(min_p)` (`-inf` disables the floor when `min_p <= 0`). `noise_scale` is 1 for a sampled row (the
/// ordinary Gumbel-max trick: unbiased only when the noise rides the temperature-scaled logits unscaled
/// itself) and 0 for a greedy row sharing the dispatch, whose `inv_temp = 1` then reduces its perturbed
/// value to its logit.
fn row_params(rule: SampleRule, sampler: &Sampler) -> Vec<f32> {
    let mut params = if sampler.temperature <= 0.0 {
        vec![1.0, f32::NEG_INFINITY, 0.0]
    } else {
        let floor_offset = if sampler.min_p > 0.0 {
            sampler.temperature * sampler.min_p.ln()
        } else {
            f32::NEG_INFINITY
        };
        vec![1.0 / sampler.temperature, floor_offset, 1.0]
    };
    if rule.has_top_p() {
        params.push(if sampler.temperature <= 0.0 {
            1.0
        } else {
            sampler.top_p
        });
    }
    params
}

/// A device suffix row's `seed`/`params`/`top_k` for one decode step (card 551a R-551a-4; R-551b-2):
/// built from a [`Sampler`]'s current knobs and its next draw seed, under the rule of the step's head.
pub(crate) struct DrawRow {
    pub(crate) seed: i32,
    pub(crate) params: Vec<f32>,
    pub(crate) top_k: Option<i32>,
}

impl DrawRow {
    /// The row a picking lane binds. `rule` is the head's: a non-greedy rule. A sampled sampler consumes
    /// one seed of its stream (both the host pick and a device row draw the identical per-step noise from
    /// it); a greedy one draws nothing and keeps its stream where it was.
    pub(crate) fn pick(sampler: &mut Sampler, rule: SampleRule) -> DrawRow {
        let seed = if sampler.temperature > 0.0 {
            sampler.next_device_seed() as i32
        } else {
            0
        };
        Self::for_seed(sampler, rule, seed)
    }

    /// Like [`Self::pick`] for a lane whose output is discarded (a row that does not pick, a masked
    /// row): `seed = 0`, and the sampler's stream does not advance, so the eventual real pick still draws
    /// the same sequence a loop with no throwaway steps would.
    pub(crate) fn placeholder(sampler: &Sampler, rule: SampleRule) -> DrawRow {
        Self::for_seed(sampler, rule, 0)
    }

    fn for_seed(sampler: &Sampler, rule: SampleRule, seed: i32) -> DrawRow {
        debug_assert_ne!(
            rule,
            SampleRule::Greedy,
            "DrawRow: Greedy declares no sampler-slot inputs to fill"
        );
        let top_k = rule.has_top_k().then(|| {
            if sampler.temperature <= 0.0 {
                0
            } else {
                i32::try_from(sampler.top_k).unwrap_or(i32::MAX)
            }
        });
        DrawRow {
            seed,
            params: row_params(rule, sampler),
            top_k,
        }
    }
}

/// The penalty triple of a lane that applies none: `(repetition, presence, frequency)` of `(1, 0, 0)`.
pub(crate) const NEUTRAL_PENALTY: [f32; 3] = [1.0, 0.0, 0.0];

/// One decoded pick: the token, and the logprobs a head with logprobs published for it.
pub(crate) struct Pick {
    pub(crate) token: u32,
    pub(crate) logprob: Option<TokenLogprob>,
}

/// Read pick `index` (a row, or a row's position) of a decode step's readback under `spec`: the `[2]` I32
/// `(token, non_finite_index)` pair of a plain suffix, or the F32 [`LogprobRecord`] row of a head with
/// logprobs. `non_finite_index >= 0` is Card 601's typed fault (R-551a-2: the device readback carries
/// the index, never the value - `value: None`); no token is committed for that step.
pub(crate) fn read_pick(
    spec: SampleSpec,
    bytes: &[u8],
    index: usize,
) -> std::result::Result<Pick, SamplerFault> {
    let fault = |non_finite: i64| SamplerFault::NonFiniteLogit {
        index: non_finite as usize,
        value: None,
    };
    let Some(n) = spec.logprobs else {
        let ints: &[i32] = bytemuck::cast_slice(bytes);
        let (token, non_finite) = (ints[2 * index], ints[2 * index + 1]);
        if non_finite >= 0 {
            return Err(fault(i64::from(non_finite)));
        }
        return Ok(Pick {
            token: token as u32,
            logprob: None,
        });
    };
    let record = LogprobRecord { n };
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    let row = &floats[index * record.width()..(index + 1) * record.width()];
    let non_finite = row[1] as i64;
    if non_finite >= 0 {
        return Err(fault(non_finite));
    }
    let token = row[0] as u32;
    let top = row[LogprobRecord::HEADER..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[token, logprob]| (token as u32, logprob))
        .collect();
    Ok(Pick {
        token,
        logprob: Some(TokenLogprob {
            token,
            logprob: row[2],
            top,
        }),
    })
}
