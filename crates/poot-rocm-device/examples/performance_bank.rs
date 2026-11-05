//! The performance bank's ROCm runner: measures one case of `benchmarks/performance-bank/cases.json`
//! through the executor contract on this backend and writes the raw receipt. The measurement, the
//! fixtures and the reference are shared in `benchmarks/performance-bank/runner.rs`; this file only opens the
//! two devices (one without device timing for wall time, one with it for device time). Run it through `python3 benchmarks/harness/performance_bank.py record`, which names the device and
//! driver, completes the receipt's identity from the case definition and hashes the binary.
//!
//! Needs real ROCm hardware: without it the run fails and no receipt is written (a missing receipt is
//! missing coverage, never a skipped pass).

#[path = "../../../benchmarks/performance-bank/runner.rs"]
mod runner;

fn main() -> std::process::ExitCode {
    // Wall time is measured on a device without timestamps and device time on one with them: the
    // per-dispatch timestamp work must not land in the wall numbers.
    let devices = (
        poot_rocm_device::RocmDevice::new(),
        poot_rocm_device::RocmDevice::new_with_device_timing(),
    );
    let (wall_device, timed_device) = match devices {
        (Ok(wall), Ok(timed)) => (wall, timed),
        (Err(error), _) | (_, Err(error)) => {
            eprintln!("performance_bank: no ROCm device ({error}); nothing was measured");
            return std::process::ExitCode::FAILURE;
        }
    };
    match runner::run("rocm", wall_device, timed_device) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("performance_bank: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
