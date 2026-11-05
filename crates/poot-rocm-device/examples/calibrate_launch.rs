//! `calibrate-launch` on the ROCm device (see `benchmarks/performance-bank/calibrate.rs`); run it through
//! `scripts/calibrate-launch rocm`, which holds the GPU lock and runs one process per shape.

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
    let device =
        || poot_rocm_device::RocmDevice::new_with_device_timing().map_err(|e| e.to_string());
    match calibrate::run("rocm", device) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("calibrate_launch: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
