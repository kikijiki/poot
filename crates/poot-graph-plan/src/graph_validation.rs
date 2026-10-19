//! Graph-structure, execution-lane, and validation-packet admission checks.

use crate::PlanError;
use poot_graph_ir::{ValidationPacketLayout, ValueId};

/// One graph value copied into the canonical validation packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationPacketSource {
    /// The graph value whose lanes are copied into the packet.
    pub value: ValueId,
    /// Index of the packet lane the value's first element lands in.
    pub first_lane: usize,
    /// How many packet lanes the value occupies.
    pub lane_count: usize,
}

/// Backend-neutral, declaration-ordered packet layout and source plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationPacketPlan {
    /// The canonical packet layout: which declared validation output sits in which lane range.
    pub layout: ValidationPacketLayout,
    /// One source per packet-resident value, in declaration order.
    pub sources: Vec<ValidationPacketSource>,
}

/// Card 259: is this flash-attention op's mask per head (one bias row/plane per query head, which ALiBi
/// needs) or the broadcast single row/plane every non-ALiBi caller passes? Read off the mask tensor's
/// shape, `[B, Hm, ..]`: `Hm == 1` is broadcast, `Hm == Hq` is per-head, anything else is a shape error
/// (reject clearly rather than misread a hot shared path).
///
/// The CPU oracle needs no copy of this rule: it evaluates the flash op from its decomposition, whose
/// additive mask broadcasts over the head axis exactly when `Hm == 1` (Card 556). The four flash
/// lowering arms turn the answer into element strides: a metadata word for the imported kernels, a
/// baked constant for the synthesized ones.
pub(crate) fn flash_mask_per_head(
    mask_shape: &[usize],
    hq: usize,
    what: &str,
) -> Result<bool, PlanError> {
    let hm = mask_shape.get(1).copied().unwrap_or(1);
    if hm == 1 {
        Ok(false)
    } else if hm == hq {
        Ok(true)
    } else {
        Err(PlanError::BadShape(format!(
            "{what} mask head axis must be 1 (broadcast) or Hq={hq} (per-head, ALiBi), got {hm} in \
             mask shape {mask_shape:?}"
        )))
    }
}
