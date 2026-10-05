//! Card 516a F1: `--pcie-publication-receipt` sets `HSA_ALLOCATE_QUEUE_DEV_MEM` itself, so a launch
//! without it reaches the device checks instead of failing the queue-ring provenance check. On an APU
//! that means the documented SKIP (exit 0); with the device hidden it means the no-GPU error; on a
//! discrete GPU the receipt runs. Removing the binary's own `set_var` fails every case with the
//! provenance conflict.

use std::process::Command;

#[test]
fn pcie_receipt_without_the_env_reaches_the_device_checks() {
    // `std::env::var`, not `env!`: the latter bakes the path into this test binary's compiled object code
    // at build time, and a shared compile cache (kache) that reuses that object across worktrees by
    // source-content hash would then serve whichever worktree's path happened to compile it first
    // (card 530's build.rs fix; card 543 review). `std::env::var` reads the environment cargo sets fresh
    // for every test-binary invocation, so it is correct regardless of which worktree compiled the binary.
    let output = Command::new(
        std::env::var("CARGO_BIN_EXE_poot-rocm-check")
            .expect("CARGO_BIN_EXE_poot-rocm-check must be set by cargo for test binaries"),
    )
    .arg("--pcie-publication-receipt")
    .env_remove("HSA_ALLOCATE_QUEUE_DEV_MEM")
    // Required-backend switches turn the APU skip into a failure by design; this row checks the
    // unrequired launch.
    .env_remove("POOT_REQUIRE_ROCM")
    .output()
    .expect("run poot-rocm-check");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let skipped_on_apu = output.status.success()
        && stdout.contains("SKIP: --pcie-publication-receipt requires a discrete PCIe GPU");
    let receipt_passed = output.status.success() && stdout.contains("OK: pcie-publication-receipt");
    let no_device = !output.status.success()
        && (stderr.contains("no GPU agent found")
            || stderr.contains("libhsa-runtime64.so.1 not found"));
    assert!(
        skipped_on_apu || receipt_passed || no_device,
        "expected the APU skip, a passing receipt, or a missing-device error; got status {:?}\n\
         stdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
}
