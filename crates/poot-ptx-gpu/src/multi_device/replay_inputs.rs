//! ADR-0099 item 3: per-replay input admission and the typed write that runs it.
//!
//! [`admit_replay_inputs`] is model-free, so the host test drives it directly;
//! `write_replay_inputs` runs it before touching a device and writes nothing when it fails.

#[cfg(test)]
use poot_graph_plan::multi_device::replay::{ReplayContract, StaticBufferId};

#[cfg(test)]
use super::error::CapturedMultiDeviceError;
use super::executor::PtxCapturedMultiDeviceExecutor;

/// Model-free admission for one batch of per-replay input writes (ADR-0099 item 3): every id must
/// exist in `contract`, must be declared a per-replay input, and must carry exactly the declared
/// byte count. [`PtxCapturedMultiDeviceExecutor::write_replay_inputs`] runs this before touching a
/// device, so it is the row a host test drives.
#[cfg(test)]
pub(crate) fn admit_replay_inputs(
    contract: &ReplayContract,
    inputs: &[(StaticBufferId, &[u8])],
) -> Result<(), CapturedMultiDeviceError> {
    for (id, bytes) in inputs {
        let buffer = contract
            .buffers
            .iter()
            .find(|buffer| &buffer.id == id)
            .ok_or_else(|| CapturedMultiDeviceError::UnknownBufferUpload { id: id.clone() })?;
        if !buffer.per_replay_input {
            return Err(CapturedMultiDeviceError::ReplayInputNotPerReplay { id: id.clone() });
        }
        if buffer.bytes != bytes.len() as u64 {
            return Err(CapturedMultiDeviceError::ReplayInputSizeMismatch {
                id: id.clone(),
                declared: buffer.bytes,
                actual: bytes.len(),
            });
        }
    }
    Ok(())
}

impl PtxCapturedMultiDeviceExecutor {
    /// Rewrite the contract's per-replay input buffers before the next replay (ADR-0099 item 3).
    ///
    /// Admission runs first, so a batch that touches a non-input or the wrong size is a typed error
    /// and writes nothing. Every accepted write is enqueued on its buffer's rank stream in the
    /// order given, which places it after everything already queued on that stream and before the
    /// replay's captured graphs, and each rank is synchronized before returning so the payloads
    /// need not outlive the call. Outputs stay read through [`Self::download`].
    #[cfg(test)]
    pub(crate) fn write_replay_inputs(
        &self,
        inputs: &[(StaticBufferId, &[u8])],
    ) -> Result<(), CapturedMultiDeviceError> {
        admit_replay_inputs(&self.contract, inputs)?;
        for (id, bytes) in inputs {
            let resident = self
                .buffers
                .get(id)
                .ok_or_else(|| CapturedMultiDeviceError::UnknownBufferUpload { id: id.clone() })?;
            self.transport.write_rank_bytes(&resident.buffer, bytes)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use poot_graph_plan::multi_device::CommunicationPlan;
    use poot_graph_plan::multi_device::replay::{ReplayContract, StaticBufferId, StaticBufferKind};

    use super::super::executor::tests::{replay_input_buffer, static_buffer};
    use super::*;
    fn input_contract() -> ReplayContract {
        ReplayContract {
            buffers: vec![
                replay_input_buffer("input:x", 16),
                static_buffer("fixed:w", StaticBufferKind::PackedWeight, 16),
            ],
            dispatch_count: 1,
            communication: CommunicationPlan { rows: Vec::new() },
        }
    }

    // -- ADR-0099 item 3: model-free per-replay input admission ----------------------

    /// Red under: the `per_replay_input` branch of `admit_replay_inputs` disabled. Observed
    /// 2026-09-26: the batch admits, so `expect_err` panics "writing a capture-fixed buffer must
    /// reject: ()". Green without the mutation.
    #[test]
    fn admit_replay_inputs_rejects_a_buffer_the_contract_does_not_mark_as_input() {
        let contract = input_contract();
        let fixed = StaticBufferId("fixed:w".into());
        let payload: &[u8] = &[0u8; 16];
        let error = admit_replay_inputs(&contract, &[(fixed, payload)])
            .expect_err("writing a capture-fixed buffer must reject");
        assert!(
            matches!(
                error,
                CapturedMultiDeviceError::ReplayInputNotPerReplay { ref id }
                    if *id == StaticBufferId("fixed:w".into())
            ),
            "expected ReplayInputNotPerReplay, got {error:?}"
        );
    }

    /// Red under: the size branch of `admit_replay_inputs` disabled. Observed 2026-09-26: the
    /// batch admits, so `expect_err` panics "a short per-replay input must reject: ()". Green
    /// without the mutation.
    #[test]
    fn admit_replay_inputs_rejects_a_size_mismatch() {
        let contract = input_contract();
        let input = StaticBufferId("input:x".into());
        let short: &[u8] = &[0u8; 8];
        let error = admit_replay_inputs(&contract, &[(input, short)])
            .expect_err("a short per-replay input must reject");
        assert!(
            matches!(
                error,
                CapturedMultiDeviceError::ReplayInputSizeMismatch {
                    ref id,
                    declared: 16,
                    actual: 8,
                } if *id == StaticBufferId("input:x".into())
            ),
            "expected ReplayInputSizeMismatch, got {error:?}"
        );
    }

    /// Red under: the id lookup falls back to the contract's first buffer when the id is absent.
    /// Observed 2026-09-26: the batch admits, so `expect_err` panics "an id outside the contract
    /// must reject: ()". Green without the mutation.
    #[test]
    fn admit_replay_inputs_rejects_a_buffer_the_contract_never_declared() {
        let contract = input_contract();
        let missing = StaticBufferId("input:absent".into());
        let payload: &[u8] = &[0u8; 16];
        let error = admit_replay_inputs(&contract, &[(missing, payload)])
            .expect_err("an id outside the contract must reject");
        assert!(
            matches!(
                error,
                CapturedMultiDeviceError::UnknownBufferUpload { ref id }
                    if *id == StaticBufferId("input:absent".into())
            ),
            "expected UnknownBufferUpload, got {error:?}"
        );
    }

    #[test]
    fn admit_replay_inputs_accepts_a_well_formed_batch() {
        let contract = input_contract();
        let input = StaticBufferId("input:x".into());
        let payload: &[u8] = &[7u8; 16];
        admit_replay_inputs(&contract, &[(input, payload)]).expect("a sized input batch admits");
    }
}
