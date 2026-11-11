//! Stable rank equation planning: one workgroup per row running the bitonic sort of
//! `poot_kernelgen::StableRankSpec`.
//!
//! The kernel has no schedule to choose; what it needs of the device is a workgroup of
//! [`StableRankSpec::lanes`] lanes and `8 * next_pow2(experts)` bytes of workgroup memory. A device below that
//! refuses with a typed [`Capability::RankExtent`], never the quadratic chain the op replaces.

use super::{BodyLimits, Capability, Eqn, KernelRequest, OpKind, PlanError, Planned, value_ids};
use crate::planner::{GraphTables, StorageDtypes};
use crate::refusal::Site;
use poot_kernelgen::StableRankSpec;
use poot_target::{Backend, DeviceCaps};

/// Whether this family lowers these storage dtypes: its bodies are f32 only, so a 2-byte float or a packed
/// E4M3FN payload is refused by name.
pub(super) fn admits_dtypes(dtypes: StorageDtypes) -> Result<(), Capability> {
    dtypes.refuse_unwired()
}

pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    backend: Backend,
    caps: &DeviceCaps,
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    let OpKind::StableRank(attrs) = &eqn.op else {
        unreachable!("plan_eqn routes only stable rank ops here")
    };
    let site = Site::new(g, eqn, backend, limits);
    let operand_types: Vec<_> = value_ids(eqn)
        .iter()
        .map(|&id| g.aval(id).clone())
        .collect();
    let out = eqn
        .op
        .infer(&operand_types)
        .map_err(|e| PlanError::BadShape(format!("stable rank: {e}")))?;
    let experts = out.shape.last().copied().unwrap_or(1);
    let spec = StableRankSpec {
        rows: out.numel() / experts,
        experts,
        masked: attrs.masked,
    };
    let lanes = spec.lanes();
    let fits = caps.max_workgroup_size[0] as usize >= lanes
        && caps.max_workgroup_invocations as usize >= lanes
        && u64::from(caps.lds_bytes) >= spec.lds_bytes();
    if !fits {
        return Err(site.refuse(Capability::RankExtent {
            experts,
            lanes,
            max_invocations: caps.max_workgroup_invocations,
            lds_needed: spec.lds_bytes(),
            lds_bytes: caps.lds_bytes,
        }));
    }
    Planned::generated(site, KernelRequest::StableRank(spec), backend, caps)
}
