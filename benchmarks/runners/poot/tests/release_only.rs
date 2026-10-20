//! The runner binary refuses to measure when it is not a release build, through its real entry point.
//! A `cargo test` binary is a dev-profile build, so this one is built exactly the way the refusal exists for.

use std::process::Command;

/// A debug runner started on a measurement exits non-zero, says why, and prints no result line for the
/// harness to turn into a row. Mutation: drop the `non_release_build` check in `main`; the runner then tries
/// to read the checkpoint, exits for a different reason, and the reason assertion goes red.
#[cfg(debug_assertions)]
#[test]
fn a_debug_runner_refuses_to_measure() {
    let output = Command::new(env!("CARGO_BIN_EXE_poot-bench-runner"))
        .args(["--model-dir", "/nonexistent", "--mode", "single"])
        .output()
        .expect("the runner binary starts");

    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "no result line: {:?}",
        output.stdout
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("benchmark results come from `cargo build --release` only"),
        "{stderr}"
    );
}

/// `--version` is not a measurement: a debug runner still names its build.
#[test]
fn any_runner_names_its_build_on_version() {
    let output = Command::new(env!("CARGO_BIN_EXE_poot-bench-runner"))
        .arg("--version")
        .output()
        .expect("the runner binary starts");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("poot-bench-runner "), "{stdout}");
}
