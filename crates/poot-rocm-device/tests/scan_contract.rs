//! The recurrence scan composite (POOT-634, POOT-1052) on ROCm, both families, through the one contract path
//! (`compile_staged`, `add_entry`, `step`): every output and carried-state element of the packed result within
//! tier 2 of the CPU oracle's evaluation of the same graph (its decomposition), across consecutive steps
//! whose carried state is the previous step's committed `s_out`. Sequences of 64, 1024 and 4096 positions run
//! in one launch and, stepped in pieces, through the carried state; each is compared with the oracle and the
//! two with each other. The multi-launch split a
//! `max_dispatch_work` bound makes is proven on the interpreter (`poot-kernelgen/tests/scan.rs`) and the
//! planner (`poot-graph-plan/tests/scan_contract.rs`): a real device's caps carry no calibrated bound, so it
//! launches once.
//!
//! Mutation (guards `poot-kernelgen`'s `scan.rs`, applied once with a fresh `POOT_KERNEL_CACHE_DIR`, observed
//! red, restored): drop the final-state publish; the continuation steps go red.

use poot_eval::{EvalBudget, EvalOptions, eval};
use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::scan::{check_steps, measure, streamed_in_pieces};
use poot_graph_ir::Graph;
use poot_graph_ir::test_support::scan::ScanFixture;
use poot_graph_plan::Target;
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;

use poot_executor_parity::{TIER2, assert_close};

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

/// SC-003 (GDN): a padded tail (`L = 6, C = 4`), two chunks at a realistic head width (`L = 72, C = 64,
/// D = 128`) and a three-step continuation.
#[test]
fn delta_rule_scan_matches_the_oracle_on_rocm() {
    let Some((mut exec, target)) = open() else {
        return;
    };
    for (whole, piece, steps, label) in [
        (ScanFixture::delta_rule(3, 0, 8, 4), 6, 1, "tail L=6 C=4"),
        (
            ScanFixture::delta_rule(2, 0, 128, 64),
            72,
            1,
            "L=72 C=64 D=128",
        ),
        (
            ScanFixture::delta_rule(2, 0, 16, 4),
            4,
            3,
            "continuation 3x4",
        ),
    ] {
        check_steps(&mut *exec, target, whole, piece, steps, label, &oracle);
    }
}

/// SC-003 (SSD): `L = 7, C = 4`, `L = 64, C = 16` at a Mamba2-shaped state, and a three-step continuation.
#[test]
fn ssd_scan_matches_the_oracle_on_rocm() {
    let Some((mut exec, target)) = open() else {
        return;
    };
    for (whole, piece, steps, label) in [
        (ScanFixture::ssd(3, 0, 8, 16, 4), 7, 1, "L=7 C=4"),
        (
            ScanFixture::ssd(2, 0, 64, 128, 16),
            64,
            1,
            "L=64 C=16 P=64 N=128",
        ),
        (ScanFixture::ssd(2, 0, 8, 16, 4), 4, 3, "continuation 3x4"),
    ] {
        check_steps(&mut *exec, target, whole, piece, steps, label, &oracle);
    }
}

/// The lengths the card names: for each family and `L` in 64, 1024, 4096 the one-launch packed result agrees
/// with the oracle and with the same scan stepped in 16-position pieces whose carried state the entry commits
/// between steps.
#[test]
fn long_scan_agrees_with_the_oracle_and_with_stepped_pieces_on_rocm() {
    let Some((mut exec, target)) = open() else {
        return;
    };
    for base in [
        ScanFixture::delta_rule(2, 0, 128, 64),
        ScanFixture::ssd(2, 0, 64, 128, 64),
    ] {
        for len in [64, 1024, 4096] {
            let label = format!("{:?} L={len}", base.family);
            let whole = check_steps(&mut *exec, target, base, len, 1, &label, &oracle);
            let stepped = streamed_in_pieces(
                &mut *exec,
                target,
                ScanFixture { len, ..base },
                16,
                &format!("{label} pieces"),
            );
            assert_close(
                &whole[0],
                &stepped,
                TIER2,
                &format!("{label}: one launch against stepped pieces"),
            );
        }
    }
}

/// The timing receipt: step time of the scan at a Qwen3-Next head shape (32 heads,
/// `D = 128`) and a Nemotron-H one. Measurement, not acceptance: run by exact name
/// under the GPU lock with `--no-capture` (the receipt in `docs/dscan.md`).
#[test]
#[ignore = "measurement; run by exact name under the GPU lock"]
fn measure_scan_time_on_rocm() {
    let Some((mut exec, target)) = open() else {
        return;
    };
    for base in [
        ScanFixture::delta_rule(32, 0, 128, 64),
        ScanFixture::ssd(64, 0, 64, 128, 64),
    ] {
        for len in [64, 256, 1024, 4096] {
            let fixture = ScanFixture { len, ..base };
            let time = measure(&mut *exec, target, &fixture, 5);
            eprintln!(
                "MEASURE {:?} L={len}: wall={:.2}ms device={:?}",
                fixture.family, time.wall_ms, time.device_ms
            );
        }
    }
}
