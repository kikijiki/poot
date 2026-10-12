//! The performance bank's NVIDIA PTX runner: measures one case of `benchmarks/performance-bank/cases.json`
//! through the executor contract on this backend and writes the raw receipt. The measurement, the
//! fixtures and the reference are shared in `benchmarks/performance-bank/runner.rs`; this file only opens the
//! device. Run it through `python3 benchmarks/harness/performance_bank.py record`, which names the device and
//! driver, completes the receipt's identity from the case definition and hashes the binary.
//!
//! Needs real NVIDIA PTX hardware: without it the run fails and no receipt is written (a missing receipt is
//! missing coverage, never a skipped pass).

#[path = "../../../benchmarks/performance-bank/runner.rs"]
mod runner;

fn main() -> std::process::ExitCode {
    let device = match poot_ptx_device::PtxDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("performance_bank: no NVIDIA PTX device ({error}); nothing was measured");
            return std::process::ExitCode::FAILURE;
        }
    };
    match runner::run("ptx", device) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("performance_bank: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
