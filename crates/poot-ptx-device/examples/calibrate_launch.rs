//! `calibrate-launch` on the NVIDIA PTX device (see `benchmarks/performance-bank/calibrate.rs`); run it through
//! `scripts/calibrate-launch ptx`, which holds the GPU lock and runs one process per measurement. On a machine with no
//! NVIDIA GPU, `scripts/ptx-pod-lane --calibrate` runs it on a RunPod pod.

#[path = "../../../benchmarks/performance-bank/calibrate.rs"]
mod calibrate;
// The bank runner is shared whole; the receipt path (`run`, `run_case`) is the bank examples' and unused here.
#[expect(
    dead_code,
    reason = "the bank's receipt path is unused by this example"
)]
#[path = "../../../benchmarks/performance-bank/runner.rs"]
mod runner;

fn main() -> std::process::ExitCode {
    let device = || poot_ptx_device::PtxDevice::new_with_device_timing().map_err(|e| e.to_string());
    match calibrate::run("ptx", device) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("calibrate_launch: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
