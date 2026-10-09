//! Step 3 of [`super::compile_staged`]: the expert placement. Only the resident placement exists;
//! pooling is the layer-segmentation transform and expert-residency manager that land with the
//! pool card, so asking for it is a typed refusal.

use super::{ExpertPlacement, StagedCompileError, Unsupported};

pub(super) fn apply(experts: ExpertPlacement) -> Result<(), StagedCompileError> {
    match experts {
        ExpertPlacement::AllResident => Ok(()),
        ExpertPlacement::Pooled => Err(StagedCompileError::Unsupported(Unsupported::ExpertPool)),
    }
}
