//! Card 551b: the sampling suffix the Runner appends after a dense decode
//! graph's raw logits, so every backend samples through the same `OpKind::SampleToken` contract path
//! (R472-007) instead of a per-backend fused entry point. Card 734 moved this file here from `core/`.
//!
//! [`with_head`] appends the suffix via card 551a's `Builder::resume` + `sample_head`
//! (`poot_graph_ir::ops::sampling`); [`Head`] names which suffix (if any) a compiled decode
//! entry carries; [`rule_of`] gates which requests the suffix can express (R-551b-2: penalties, bias,
//! logprobs and a guided-decoding constraint still need the host path); [`SuffixRows`] derives one
//! device row's `seed`/`params`/`top_k` from a [`Sampler`]'s knobs; [`read_tokens`] turns the suffix's
//! `[rows, 2]` I32 readback into a token or Card 601's typed fault; [`eval_sample_token`] runs the
//! identical `SampleToken`/`RandomUniform` semantics on the host, through `poot-eval`'s own walk, so
//! [`Sampler::pick`]'s host draw and a device suffix row cannot drift apart (R-551b-2).

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value};
use poot_graph_ir::op::SampleRule;
use poot_graph_ir::ops::sampling::sample_head;
use poot_graph_ir::{Builder, Graph, NoValidations, Slot, TensorType, ValidationOutputs, ValueId};
use poot_tensor::DType;
use poot_tensor::HostTensor;

use crate::core::sampler::{Sampler, SamplerFault};
use crate::error::Result;

/// Which suffix (if any) a compiled entry carries after its raw logits. `Logits` is the raw graph:
/// the host sampling path, and the Runner's `core::speculative` verification and VLM caption path,
/// which need the full logits, never a suffix. `Sample(rule)` is [`with_head`]'s
/// appended suffix, read back as `[rows, 2]` I32 `(token, non_finite_index)` through [`read_tokens`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Head {
    Logits,
    Sample(SampleRule),
}

impl Head {
    /// The suffix of a greedy request: the on-device argmax.
    pub const GREEDY: Head = Head::Sample(SampleRule::Greedy);
}

/// `g` (and its appended suffix inputs' ids), after [`with_head`]. `Logits` returns `g` itself
/// unchanged and no ids; `Sample(rule)` returns the three ids [`SuffixRows`]'s
/// tensors bind by - `None` for a tag `rule` does not declare (R-551a-4: `Greedy` declares none).
pub(crate) struct Appended {
    pub(crate) graph: Graph,
    pub(crate) seed: Option<ValueId>,
    pub(crate) params: Option<ValueId>,
    pub(crate) top_k: Option<ValueId>,
}

/// Append `head`'s suffix after `g`'s raw logits output (card 551a): reopens the
/// finished graph (`Builder::resume`), appends [`sample_head`]'s suffix, and finishes again over the
/// same (possibly unchanged) state pairs. `Logits` is a no-op (the raw graph, unchanged) - infallible,
/// since `resume`/`sample_head`/`finish_with_state` are.
pub(crate) fn with_head(g: Graph, head: Head) -> Appended {
    match head {
        Head::Logits => Appended {
            graph: g,
            seed: None,
            params: None,
            top_k: None,
        },
        Head::Sample(rule) => {
            let resumed = Builder::resume(g);
            let sh = sample_head(&resumed.builder, resumed.out, rule);
            let graph = resumed.builder.finish_with_state(sh.out, &resumed.state);
            Appended {
                graph,
                seed: sh.seed.map(|t| t.id),
                params: sh.params.map(|t| t.id),
                top_k: sh.top_k.map(|t| t.id),
            }
        }
    }
}

/// [`with_head`] over a validation-bearing graph, which is what [`poot_models::model::Model::trace`]
/// returns. The suffix only appends values, so every declared validation keeps its value id: the
/// declarations are set aside, the plain graph is extended and they are attached again.
pub(crate) fn append_head(
    g: Graph<ValidationOutputs>,
    head: Head,
) -> (Graph<ValidationOutputs>, SuffixIds) {
    let validations = g.validation_outputs().to_vec();
    let plain = Graph {
        values: g.values,
        inputs: g.inputs,
        consts: g.consts,
        slots: g.slots,
        eqns: g.eqns,
        output: g.output,
        validations: NoValidations,
        state: g.state,
    };
    let appended = with_head(plain, head);
    let ids = SuffixIds {
        seed: appended.seed,
        params: appended.params,
        top_k: appended.top_k,
    };
    (appended.graph.with_validations(validations), ids)
}

/// The suffix's own input ids: `None` for a tag the head's rule does not declare.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SuffixIds {
    pub(crate) seed: Option<ValueId>,
    pub(crate) params: Option<ValueId>,
    pub(crate) top_k: Option<ValueId>,
}

/// Which [`SampleRule`] a temperature draw's `top_k`/`top_p` knobs select (R-551a-4's three rule
/// shapes beyond `Greedy`): shared by [`rule_of`] (the device gate) and [`Sampler::pick_from`] (every
/// host draw, regardless of whether the request is eligible for the device suffix), so the two
/// classifications cannot drift. `top_k <= 0`/`top_p >= 1.0` mean "disabled", matching the kernel/
/// oracle convention (deval.md section 8.1).
pub(crate) fn temperature_rule(top_k: usize, top_p: f32) -> SampleRule {
    if top_p < 1.0 {
        SampleRule::GumbelTopKTopP
    } else if top_k > 0 {
        SampleRule::GumbelTopK
    } else {
        SampleRule::Gumbel
    }
}

/// Card 551b: `None` when `sampler`'s request needs the host path (a penalty,
/// bias, logprob recording or a guided-decoding constraint - anything [`Sampler::is_greedy`]/
/// [`Sampler::is_simple_temperature`] reject), `Some(rule)` otherwise. Greedy always maps to
/// [`SampleRule::Greedy`] (SC-003 needs the device suffix's non-finite readback on the greedy path
/// too, not just sampled); a simple temperature draw maps through [`temperature_rule`].
pub(crate) fn rule_of(sampler: &Sampler) -> Option<SampleRule> {
    if sampler.is_greedy() {
        return Some(SampleRule::Greedy);
    }
    if !sampler.is_simple_temperature() {
        return None;
    }
    Some(temperature_rule(sampler.top_k, sampler.top_p))
}

/// This row's `params` column vector (card 551a: `[inv_temp, floor_offset, noise_scale]`, `+top_p`
/// for [`SampleRule::GumbelTopKTopP`]), shared by [`SuffixRows::push`] (a device row) and
/// [`eval_sample_token`] (the host draw), so the two cannot drift. `floor_offset` folds min-p into a
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
    if matches!(rule, SampleRule::GumbelTopKTopP) {
        params.push(top_p);
    }
    params
}

/// A device suffix row's `seed`/`params`/`top_k` for one decode step (card 551a R-551a-4; R-551b-2):
/// built from a [`Sampler`]'s current knobs and its next [`Sampler::next_device_seed`] draw - the
/// identical derivation [`eval_sample_token`]'s host pick uses, so a seeded `Sampler` gives the same
/// token whether this step's pick runs on the host or through a device suffix entry.
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
        let top_k = matches!(rule, SampleRule::GumbelTopK | SampleRule::GumbelTopKTopP)
            .then_some(sampler.top_k as i32);
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

/// Evaluate `SampleToken(rule)` over one row of already-adjusted `logits` (bias/penalties/the
/// guided-decoding mask: the caller's job, as [`Sampler::pick`] does before ever reaching this) on
/// the host, through a standalone graph run by `poot-eval`'s own walk (the identical
/// `OpKind::SampleToken`/`OpKind::RandomUniform` arms [`with_head`]'s device suffix compiles): the
/// host pick's math and a device suffix row cannot drift apart, because both resolve through the
/// same production op semantics (R-551b-2) - a seeded `Sampler` on the same logits gives the same
/// token either way. `seed` is this row's [`Sampler::next_device_seed`] draw.
///
/// `rule` must not be [`SampleRule::Greedy`] (greedy has a closed-form host answer,
/// [`crate::core::generate::argmax`], and needs no noise/params graph).
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_sample_token(
    rule: SampleRule,
    logits: &[f32],
    seed: u32,
    temperature: f32,
    min_p: f32,
    top_k: usize,
    top_p: f32,
) -> Result<(i32, i32)> {
    debug_assert_ne!(
        rule,
        SampleRule::Greedy,
        "eval_sample_token: Greedy has a closed-form host answer (argmax), call that instead"
    );
    let b = Builder::new();
    let vocab = logits.len();
    let logits_in = b.slot(Slot::Activation, TensorType::new(vec![vocab], DType::F32));
    let head = sample_head(&b, logits_in, rule);
    let g = b.finish(head.out);

    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    inputs.insert(
        logits_in.id,
        HostTensor::f32(vec![vocab], logits.to_vec()).into(),
    );
    if let Some(seed_id) = head.seed {
        inputs.insert(
            seed_id.id,
            HostTensor::i32(vec![], vec![seed as i32]).into(),
        );
    }
    if let Some(params_id) = head.params {
        let params = row_params(rule, temperature, min_p, top_p);
        inputs.insert(
            params_id.id,
            HostTensor::f32(vec![params.len()], params).into(),
        );
    }
    if let Some(top_k_id) = head.top_k {
        inputs.insert(
            top_k_id.id,
            HostTensor::i32(vec![], vec![top_k as i32]).into(),
        );
    }

    let result = poot_eval::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))?;
    let out = result.output.into_host()?;
    let ints = out
        .as_i32()
        .expect("SampleToken's output is always the I32 side payload");
    Ok((ints[0], ints[1]))
}
