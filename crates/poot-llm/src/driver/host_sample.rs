//! The host pick over a logits row (Card 734): the full-logits head - a sampled verify window under a
//! draft policy, which needs the target's whole distribution - reads the step's logits back and picks with
//! [`Sampler::pick`]. No ordinary request takes it: the penalties, the bias, the guided mask and the
//! logprobs all run in the sampling suffix on the device (POOT-1268).

use poot_executor::{ExecError, StepOutputs};

use crate::core::sampler::{Sampler, SamplerFault};

/// The step's logits row as f32, whatever lane the head's output was planned in.
pub(crate) fn read_logits(outputs: &mut StepOutputs<'_>) -> Result<Vec<f32>, ExecError> {
    let tensor = outputs.to_host()?;
    Ok(tensor.to_f32()?.into_owned())
}

/// `sampler`'s pick over one logits row: bias, penalties, the constraint mask and the draw, with
/// Card 601's typed fault for a non-finite or empty row.
pub(crate) fn pick(sampler: &mut Sampler, logits: &[f32]) -> Result<u32, SamplerFault> {
    let token = sampler.pick(logits)?;
    Ok(u32::try_from(token).expect("a vocabulary index is a token id"))
}
