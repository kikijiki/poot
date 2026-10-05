//! Collectives on the single-rank oracle: `AllReduce` and `AllGather`.

use poot_tensor::HostTensor;

/// `AllReduce` and `AllGather` at world size 1: the collective is the identity. The multi-rank test
/// harness sums (`AllReduce`) or concatenates along `axis` (`AllGather`) per-rank evaluations outside
/// the graph (card 049b).
pub(crate) fn single_rank_identity(x: HostTensor) -> HostTensor {
    x
}
