//! Scan equation planning: a bounded streaming schedule for the GDN delta rule and the Mamba2 SSD.
//!
//! A scan's packed result `[1, H, L + S, W]` is written by one or more launches of the streaming scan kernel.
//! Each launch covers a token range (its phase) and hands the carried state to the next through the result's
//! own state rows, so the serial work of one dispatch is the phase's, not the sequence's: a device with a
//! known `max_dispatch_work` gets as many launches as keep every one within it, one without a calibrated
//! bound gets one launch. A single token that cannot fit the bound is refused by the resource check, never
//! launched. The lane count is the widest power of two whose state columns fit the device's workgroup memory.

use super::*;
use poot_graph_ir::ScanFamily;
use poot_kernelgen::{ScanKernel, ScanPhase, ScanSpec};
use poot_target::{Backend, DeviceCaps, Queried};

/// The widest workgroup a scan launches: its lanes each own one state column.
const MAX_COLS: usize = 64;

pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    backend: Backend,
    caps: &DeviceCaps,
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    let OpKind::Scan(attrs) = &eqn.op else {
        unreachable!("plan_eqn routes only scan ops here")
    };
    let site = Site::new(g, eqn, backend, limits);
    let operand_types: Vec<_> = value_ids(eqn)
        .iter()
        .map(|&id| g.aval(id).clone())
        .collect();
    let layout = attrs
        .layout(&operand_types)
        .map_err(|e| PlanError::BadShape(format!("scan: {e}")))?;
    let lanes_limit = (caps.max_workgroup_size[0] as usize)
        .min(caps.max_workgroup_invocations as usize)
        .min(MAX_COLS);
    let rows = layout.state_rows;
    let cols = std::iter::successors(Some(MAX_COLS), |cols| (*cols > 1).then_some(cols / 2))
        .find(|&cols| cols <= lanes_limit && 4 * rows * cols <= caps.lds_bytes as usize)
        .ok_or_else(|| {
            site.refuse(Capability::ScanLds {
                rows,
                lds_bytes: caps.lds_bytes,
            })
        })?;
    let whole = ScanSpec {
        kernel: match attrs.family {
            ScanFamily::DeltaRule => ScanKernel::DeltaRule,
            ScanFamily::Ssd => ScanKernel::Ssd,
        },
        heads: layout.heads,
        len: layout.len,
        rows,
        width: layout.width,
        cols: cols.min(layout.width),
        phase: ScanPhase {
            start: 0,
            count: layout.len,
        },
    };
    let per_launch = match caps.max_dispatch_work {
        Queried::Known(budget) => whole.tokens_within(budget).min(layout.len),
        Queried::Unknown => layout.len,
    };
    let phase = |start: usize| ScanSpec {
        phase: ScanPhase {
            start,
            count: per_launch.min(layout.len - start),
        },
        ..whole.clone()
    };
    if per_launch == layout.len {
        return Planned::generated(site, KernelRequest::Scan(whole), backend, caps);
    }
    let launches = (0..layout.len)
        .step_by(per_launch)
        .map(|start| {
            Planned::generated_dispatch(site, KernelRequest::Scan(phase(start)), backend, caps)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Planned::split(launches)
}
