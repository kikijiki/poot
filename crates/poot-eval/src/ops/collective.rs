//! Collectives on the single-rank oracle: `AllReduce` and `AllGather`.

use std::num::NonZeroU32;

use poot_graph_ir::ValueId;
use poot_tensor::HostTensor;

use crate::EvalError;

/// `AllReduce` over one rank: the identity. The oracle evaluates one rank's graph; a collective across
/// ranks is a cut between stage programs, never an equation the oracle evaluates.
pub(crate) fn single_rank_identity(x: HostTensor) -> HostTensor {
    x
}

/// `AllGather` over `world` ranks. `world == 1` is the identity. Beyond one rank the output is `world`
/// times larger along the gather axis and this evaluator holds one rank's slice only, so there is no
/// value to return: a typed refusal, not the input.
pub(crate) fn all_gather(
    x: HostTensor,
    world: NonZeroU32,
    eqn: ValueId,
) -> Result<HostTensor, EvalError> {
    if world.get() > 1 {
        return Err(EvalError::Unsupported {
            eqn,
            op: "all_gather",
            detail: format!(
                "the oracle holds one rank, so an all_gather over {world} ranks has no value"
            ),
        });
    }
    Ok(x)
}
