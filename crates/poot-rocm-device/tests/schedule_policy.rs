//! The scan's shared-column schedule against the unsplit one on ROCm (POOT-667), through the one
//! contract path: three regimes at a realistic head width (`D = 128`), each compared element for element
//! (every output row and the carried state) with the CPU oracle's evaluation of the same graph.
//!
//! - a short sequence keeps one lane per state column;
//! - a long sequence over few heads, whose unsplit launch leaves the device under-filled, shares each
//!   column's rows between lanes;
//! - a long sequence over enough heads to fill the device keeps one lane per column.
//!
//! The Mamba2 SSD runs the same regimes (POOT-1112) plus the state widths that give every row split the
//! planner can pick: `N = 8, 16, 32, 64, 128` plan splits 1, 2, 4, 8, 16, each over 1024 steps, so the
//! alternating-plane one-barrier reduction (a lane one token ahead is a real-concurrency hazard the CPU
//! interpreter cannot show) is compared with the oracle at every split.
//!
//! Each regime also asserts the planned `row_split`, so a policy that drifts to the other schedule is red
//! here even though both schedules agree with the oracle. A split sequence is finally stepped in short
//! unsplit pieces and the two assembled results compared with each other.
//!
//! The crowded SSD row (heads = compute units - 1, the most workgroups an under-filled launch can have
//! resident) is the one that exposes a lane running a token ahead: with few heads no lane gets ahead.
//!
//! Mutations (guard `poot-kernelgen`'s `scan.rs`, each applied once with a fresh `POOT_KERNEL_CACHE_DIR`,
//! observed red, restored): sum one partial too few in the cross-lane reduction, the long delta-rule regime
//! goes red; skip the SSD cross-lane reduction (`y_acc = y_part`), the first split SSD row goes red on wgpu
//! and ROCm; make the SSD plane parity constant 0, the crowded SSD row goes red on ROCm (`element 4336: got
//! 0.0358, want -0.0174`) and on PTX (RTX A4000: `element 1924496: got -0.48178893, want -0.41417295`), and
//! stays green on wgpu, where the hazard did not fire at 512, 1024 or 4096 steps or at the crowded fixture:
//! the plane-parity hazard has ROCm and PTX mutation coverage only.

use poot_eval::{EvalBudget, EvalOptions, eval};
use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::assert_close;
use poot_executor_parity::scan::{check_steps, numerics_class, streamed_in_pieces};
use poot_graph_ir::test_support::scan::ScanFixture;
use poot_graph_ir::{Graph, OpKind};
use poot_graph_plan::test_support::gpu_caps;
use poot_graph_plan::{
    CompileLimits, CompileOptions, KernelChoice, Target, compile, production_plan::Submission,
};
use poot_kernelgen::KernelRequest;
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;

fn oracle(
    graph: &Graph,
    values: &std::collections::HashMap<poot_graph_ir::ValueId, poot_tensor::HostTensor>,
) -> poot_tensor::HostTensor {
    let values = values
        .iter()
        .map(|(&id, t)| (id, poot_eval::Value::from(t.clone())))
        .collect();
    eval(graph, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap_or_else(|e| panic!("oracle: {e}"))
        .output
        .into_host()
        .unwrap()
}

fn open() -> Option<(Box<dyn Executor>, Target)> {
    let device = open_or_skip(DeviceBackend::Rocm, RocmDevice::new())?;
    let target = device.target();
    Some((Box::new(Engine::new(device)), target))
}

/// The `row_split` the planner chose for `fixture` on `target`.
fn planned_split(fixture: &ScanFixture, target: &Target) -> usize {
    let program = compile(
        &fixture.graph().graph,
        target,
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
    )
    .expect("the scan compiles");
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::Scan(_)))
        .expect("the program keeps the scan");
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Scan(spec)) => spec.row_split,
        other => panic!("the scan is not generated: {other:?}"),
    }
}

#[test]
fn scan_schedules_match_the_oracle_in_every_regime_on_rocm() {
    let Some((mut exec, target)) = open() else {
        return;
    };
    // Heads that fill the device: more lanes than one wave per compute unit at one lane per column.
    let filling = gpu_caps(&target).compute_units as usize * 64 / 128 + 1;
    let ssd_filling = gpu_caps(&target).compute_units as usize + 1;
    // The most heads that still leave the device under-filled: the split launch with the most workgroups
    // resident at once, where waves of one column are skewed the most.
    let ssd_crowded = (gpu_caps(&target).compute_units as usize)
        .saturating_sub(1)
        .max(1);
    // The crowded launch shares the rows exactly as the same state width does with two heads: only the
    // fill differs, and it is below the fill bound by construction. The expectation is the planner's own
    // choice for the uncrowded fixture, so a device's limits move both sides together.
    let crowded_split = planned_split(&ScanFixture::ssd(2, 1024, 64, 128, 64), &target);
    assert!(
        crowded_split > 1,
        "the uncrowded N=128 launch shares the rows"
    );
    // `Some(split)` pins the planned split exactly; `None` only that the rows are shared.
    for (label, whole, split) in [
        (
            "delta short L=64",
            ScanFixture::delta_rule(2, 64, 128, 64),
            Some(1),
        ),
        (
            "delta long L=1024",
            ScanFixture::delta_rule(2, 1024, 128, 64),
            None,
        ),
        (
            "delta filled L=1024",
            ScanFixture::delta_rule(filling, 1024, 128, 64),
            Some(1),
        ),
        (
            "ssd short L=64",
            ScanFixture::ssd(2, 64, 64, 128, 64),
            Some(1),
        ),
        (
            "ssd filled L=1024",
            ScanFixture::ssd(ssd_filling, 1024, 64, 128, 64),
            Some(1),
        ),
        (
            "ssd crowded N=128 L=1024",
            ScanFixture::ssd(ssd_crowded, 1024, 64, 128, 64),
            Some(crowded_split),
        ),
        (
            "ssd N=8 L=1024",
            ScanFixture::ssd(2, 1024, 64, 8, 64),
            Some(1),
        ),
        (
            "ssd N=16 L=1024",
            ScanFixture::ssd(2, 1024, 64, 16, 64),
            Some(2),
        ),
        (
            "ssd N=32 L=1024",
            ScanFixture::ssd(2, 1024, 64, 32, 64),
            Some(4),
        ),
        (
            "ssd N=64 L=1024",
            ScanFixture::ssd(2, 1024, 64, 64, 64),
            Some(8),
        ),
        (
            "ssd N=128 L=1024",
            ScanFixture::ssd(2, 1024, 64, 128, 64),
            Some(16),
        ),
    ] {
        let planned = planned_split(&whole, &target);
        match split {
            Some(want) => assert_eq!(planned, want, "{label}: the planned row split"),
            None => assert!(planned > 1, "{label}: the rows are not shared ({planned})"),
        }
        check_steps(&mut *exec, target, whole, whole.len, 1, label, &oracle);
    }
    for long in [
        ScanFixture::delta_rule(2, 1024, 128, 64),
        ScanFixture::ssd(2, 1024, 64, 128, 64),
    ] {
        let one_launch = check_steps(&mut *exec, target, long, long.len, 1, "long whole", &oracle);
        let pieces = streamed_in_pieces(&mut *exec, target, long, 16, "long pieces");
        assert_close(
            &one_launch[0],
            &pieces,
            numerics_class(target, &long).tolerance(),
            "split schedule against unsplit pieces",
        );
    }
}
