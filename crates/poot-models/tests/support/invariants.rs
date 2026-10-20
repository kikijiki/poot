//! The launch invariants every compiled plan must satisfy, stated from the device's caps and the
//! equation's output, never from the planner's own launch arithmetic. A digest says a plan did not
//! move; these say a plan is not wrong: a launch that leaves output elements unwritten, a workgroup
//! the device cannot run, or workgroup memory past the device's budget.
#![allow(dead_code, reason = "shared by test binaries; each uses part of it")]
use poot_graph_plan::{Dispatch, KernelChoice, Plan, Program, test_support::gpu_caps};
use poot_kernelgen::{BodyUse, KernelRequest};
use poot_target::DeviceCaps;

/// Whether one thread of `dispatch` writes at most one output element, so that the launched threads
/// must number at least the output's elements. A row or attention kernel strides its lanes over a row or a head, a
/// contraction, a fused pointwise region and a scan give a thread several outputs (a register tile, a
/// vector pair, a state row) or a workgroup one (a lane tree per column), and an imported template is
/// a shipped body of its own contract; their coverage is the kernel's, and the rest of these
/// invariants still bind them.
pub(crate) fn writes_one_element_per_thread(dispatch: &Dispatch) -> bool {
    matches!(
        dispatch.choice(),
        KernelChoice::Generated(
            KernelRequest::Elementwise(_)
                | KernelRequest::Movement(_)
                | KernelRequest::Rope(_)
                | KernelRequest::TopK(_)
                | KernelRequest::BlockTopK(_)
        )
    )
}

/// Every way the plans of `program` break a launch invariant on the caps of its own target, one
/// sentence each, naming the equation and the launch. Empty when the program is sound.
///
/// Per launch: the grid is positive and the workgroup counts it rounds up to fit the device's
/// grid ceiling; the workgroup shape fits the device's per-dimension and total invocation limits; the
/// workgroup memory the body declares fits the device's budget. Per equation whose kernel writes one
/// element per thread: the threads launched, summed over its launches, number at least the elements of
/// its output.
pub(crate) fn violations(program: &Program) -> Vec<String> {
    let caps = gpu_caps(program.target());
    let mut found = Vec::new();
    for (index, (eqn, plan)) in program.planned().enumerate() {
        let Plan::Dispatches(dispatches) = plan else {
            continue;
        };
        let at = |launch: usize| format!("eqn {index} ({}) launch {launch}", eqn.op.name());
        let mut threads = 0u64;
        for (launch, dispatch) in dispatches.iter().enumerate() {
            launch_violations(&at(launch), dispatch, &caps, &mut found);
            threads += dispatch
                .launch()
                .grid
                .iter()
                .map(|&x| u64::from(x))
                .product::<u64>();
        }
        if dispatches.iter().all(writes_one_element_per_thread) {
            let elements = program
                .graph()
                .aval(eqn.out)
                .shape
                .iter()
                .map(|&d| d as u64)
                .product::<u64>();
            if threads < elements {
                found.push(format!(
                    "{}: {threads} threads launched for an output of {elements} elements",
                    at(0)
                ));
            }
        }
    }
    found
}

/// The per-launch invariants of one dispatch.
fn launch_violations(at: &str, dispatch: &Dispatch, caps: &DeviceCaps, found: &mut Vec<String>) {
    let grid = dispatch.launch().grid;
    let workgroup = dispatch.body().workgroup_size;
    for axis in 0..3 {
        if grid[axis] == 0 || workgroup[axis] == 0 {
            found.push(format!(
                "{at}: grid {grid:?} or workgroup {workgroup:?} is empty on axis {axis}"
            ));
            return;
        }
        // The grid counts threads; the executor dispatches the workgroups that cover them.
        let groups = grid[axis].div_ceil(workgroup[axis]);
        if groups > caps.max_grid[axis] {
            found.push(format!(
                "{at}: {groups} workgroups on axis {axis} pass the device's {}",
                caps.max_grid[axis]
            ));
        }
        if workgroup[axis] > caps.max_workgroup_size[axis] {
            found.push(format!(
                "{at}: workgroup {workgroup:?} passes the device's per-axis {:?}",
                caps.max_workgroup_size
            ));
        }
    }
    let invocations = workgroup.iter().map(|&x| u64::from(x)).product::<u64>();
    if invocations > u64::from(caps.max_workgroup_invocations) {
        found.push(format!(
            "{at}: {invocations} invocations per workgroup pass the device's {}",
            caps.max_workgroup_invocations
        ));
    }
    match BodyUse::measure(dispatch.body()) {
        Ok(used) if used.lds_bytes > u64::from(caps.lds_bytes) => found.push(format!(
            "{at}: {} bytes of workgroup memory pass the device's {}",
            used.lds_bytes, caps.lds_bytes
        )),
        Ok(_) => {}
        Err(error) => found.push(format!("{at}: workgroup memory is unmeasurable: {error}")),
    }
}
