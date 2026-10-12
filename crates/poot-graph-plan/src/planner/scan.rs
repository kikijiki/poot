//! Scan equation planning: a bounded streaming schedule for the GDN delta rule and the Mamba2 SSD.
//!
//! A scan's packed result `[1, H, L + S, W]` is written by one or more launches of the streaming scan kernel.
//! Each launch covers a token range (its phase) and hands the carried state to the next through the result's
//! own state rows, so the serial work of one dispatch is the phase's, not the sequence's: a device with a
//! known `max_dispatch_work` gets as many launches as keep every one within it, one without a calibrated
//! bound gets one launch. A single token that cannot fit the bound is refused by the resource check, never
//! launched.
//!
//! The workgroup shape is [`scan_geometry`]: one lane per state column, or, for a long sequence whose unsplit
//! launch leaves the device under-filled, each column's rows shared between `split` lanes so the per-token
//! serial chain is `rows / split` rows long. The rule reads the layout and the device's limits, nothing else.

use super::{BodyLimits, Capability, Eqn, KernelRequest, OpKind, PlanError, Planned, value_ids};
use crate::planner::GraphTables;
use crate::refusal::Site;
use poot_graph_ir::{ScanFamily, ScanLayout};
use poot_kernelgen::{ScanKernel, ScanPhase, ScanSpec};
use poot_target::{Backend, DeviceCaps, Queried};

/// The most state columns one workgroup owns.
const MAX_COLS: usize = 64;

/// The most lanes a scan workgroup launches.
const MAX_LANES: usize = 256;

/// The fewest state rows a lane walks once a column is shared: below it the two barriers a token costs a
/// split schedule are not paid back by the shorter serial chain.
const MIN_ROWS_PER_LANE: usize = 8;

/// The shortest sequence a launch shares state columns for. The split was measured to win from 512 steps (and
/// not to be distinguishable from the unsplit schedule at 64 on the same device), so shorter sequences keep
/// one lane per column.
const MIN_SPLIT_STEPS: usize = 512;

/// Lanes per compute unit below which an unsplit launch leaves the device idle: one wave of 64 lanes. An
/// unsplit launch that already fills the device (`heads * width` lanes at least this many per compute unit)
/// gains nothing from more lanes, measured on a 64-head state.
const LANES_PER_COMPUTE_UNIT: usize = 64;

/// The shape of one scan workgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScanGeometry {
    /// State columns per workgroup (before the layout's width bounds them).
    pub state_cols: usize,
    /// Lanes sharing one state column's rows.
    pub split: usize,
}

/// The workgroup shape for `layout` on `caps`, or `None` when not even one column fits in workgroup memory.
///
/// The unsplit shape is one lane per column, the widest power of two of columns, at most [`MAX_COLS`], whose
/// state fits workgroup memory. A split shape is chosen only where it was measured to win: a sequence of at
/// least [`MIN_SPLIT_STEPS`] steps whose unsplit launch leaves the device under-filled
/// ([`LANES_PER_COMPUTE_UNIT`]). It is the largest power of two that divides the state's rows, leaves each
/// lane at least [`MIN_ROWS_PER_LANE`] rows and fits the workgroup limits, with the columns again the widest
/// whose state and partial-sum planes (`rows + 2 * split` floats per column) fit and whose lanes
/// (`state_cols * split`) fit the limits. A split too wide for any column to fit is halved until one does.
/// Both families share the rule and its thresholds, which are POOT-667's measured values, unchanged: the
/// SSD's one-barrier alternating-plane reduction was measured against one lane per column on wgpu, ROCm and
/// PTX at 4096 steps (3.9x to 4.8x) and at 512 steps (3.6x to 4.5x), and was indistinguishable from it at
/// 64 steps and on a device-filling head count, where the rule plans the same launch (POOT-1112 bank cases,
/// `candidates/POOT-1112`). The rule reads the shapes and device limits only.
pub(crate) fn scan_geometry(layout: &ScanLayout, caps: &DeviceCaps) -> Option<ScanGeometry> {
    let rows = layout.state_rows;
    debug_assert!(rows > 0, "a scan layout has at least one state row");
    let limit = (caps.max_workgroup_size[0] as usize).min(caps.max_workgroup_invocations as usize);
    let lds_bytes = caps.lds_bytes as usize;
    let halves = |n: usize| std::iter::successors(Some(n), |n| (*n > 1).then_some(n / 2));
    let fit = |split: usize, lanes_limit: usize| {
        let planes = if split == 1 { 0 } else { 2 * split };
        halves(MAX_COLS)
            .find(|&cols| cols * split <= lanes_limit && 4 * cols * (rows + planes) <= lds_bytes)
            .map(|state_cols| ScanGeometry { state_cols, split })
    };
    let unsplit = fit(1, limit.min(MAX_COLS))?;
    let launched_lanes = layout.heads * layout.width.next_multiple_of(unsplit.state_cols);
    let fills_device = launched_lanes >= caps.compute_units as usize * LANES_PER_COMPUTE_UNIT;
    if layout.len < MIN_SPLIT_STEPS || fills_device {
        return Some(unsplit);
    }
    let lanes_limit = limit.min(MAX_LANES);
    let share = (rows / MIN_ROWS_PER_LANE).clamp(1, lanes_limit.max(1));
    let widest = 1usize << share.ilog2();
    // `rows > 0` (a layout has at least one state row), so the lowest set bit exists.
    let divisor = rows.isolate_lowest_one();
    Some(
        halves(widest.min(divisor))
            .find_map(|split| fit(split, lanes_limit))
            .unwrap_or(unsplit),
    )
}

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
    let rows = layout.state_rows;
    let ScanGeometry { state_cols, split } = scan_geometry(&layout, caps).ok_or_else(|| {
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
        cols: state_cols.min(layout.width) * split,
        row_split: split,
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
