//! Block top-k equation planning: one fixed-shape workgroup per query running the radix select of
//! `poot_kernelgen::BlockTopKSpec`.
//!
//! The kernel is linear in the blocks a query scores, so it has no schedule to choose; what it needs of the
//! device is a workgroup of [`BLOCK_TOP_K_LANES`] lanes and the workgroup memory of its histogram and scans.
//! A device below that refuses with a typed [`Capability::BlockTopKWorkgroup`], never a quadratic fallback.

use super::{BodyLimits, Capability, Eqn, KernelRequest, OpKind, PlanError, Planned, value_ids};
use crate::planner::{GraphTables, StorageDtypes};
use crate::refusal::Site;
use poot_kernelgen::{BLOCK_TOP_K_LANES, BlockTopKSpec};
use poot_target::{Backend, DeviceCaps};

/// Whether this family lowers these storage dtypes: its bodies are f32/i32 only, so a 2-byte float or a
/// packed E4M3FN payload is refused by name.
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
    let OpKind::BlockTopK(attrs) = &eqn.op else {
        unreachable!("plan_eqn routes only block top-k ops here")
    };
    let site = Site::new(g, eqn, backend, limits);
    let operand_types: Vec<_> = value_ids(eqn)
        .iter()
        .map(|&id| g.aval(id).clone())
        .collect();
    let layout = attrs
        .layout(&operand_types)
        .map_err(|e| PlanError::BadShape(format!("block top-k: {e}")))?;
    let spec = BlockTopKSpec {
        queries: layout.rows * layout.tokens,
        blocks: layout.blocks,
        ratio: attrs.ratio,
        k: attrs.k,
    };
    let lanes = BLOCK_TOP_K_LANES;
    let fits = caps.max_workgroup_size[0] as usize >= lanes
        && caps.max_workgroup_invocations as usize >= lanes
        && u64::from(caps.lds_bytes) >= spec.lds_bytes();
    if !fits {
        return Err(site.refuse(Capability::BlockTopKWorkgroup {
            lanes,
            max_invocations: caps.max_workgroup_invocations,
            lds_needed: spec.lds_bytes(),
            lds_bytes: caps.lds_bytes,
        }));
    }
    Planned::generated(site, KernelRequest::BlockTopK(spec), backend, caps)
}
