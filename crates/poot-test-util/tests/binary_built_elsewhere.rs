//! A test binary built in one checkout and run from another reads the second checkout's files (POOT-1354).
//!
//! The shared target directories reuse a test binary across worktrees by source content, so the binary a gate runs
//! may have been built in a checkout that is gone or foreign. This builds a tiny crate in directory A, then runs
//! the binary it produced with the environment cargo and nextest set for a run from directory B (`CARGO_MANIFEST_DIR`
//! and the working directory), where a fixture file differs. The probe holds one test that reads through
//! `poot_test_util::manifest_dir()` and one control that reads through the compile-time macro; the control reads A's
//! fixture, which is the failure this guards against.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The probe crate's test file: identical in A and B. `@DIR@` is filled in when it is written, so this file never
/// holds the compile-time form the lint rejects.
const PROBE: &str = r#"
use std::path::Path;

fn fixture(dir: &Path) -> String {
    std::fs::read_to_string(dir.join("fixture.txt")).unwrap().trim().to_string()
}

#[test]
fn reads_through_the_shared_helper() {
    println!("HELPER={}", fixture(&poot_test_util::manifest_dir()));
}

#[test]
fn reads_through_the_compile_time_macro() {
    println!("BAKED={}", fixture(Path::new(env!("CARGO_MANIFEST_@DIR@"))));
}
"#;

fn probe_crate(dir: &Path, fixture: &str, util: &Path) {
    std::fs::create_dir_all(dir.join("tests")).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"reuse-probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n\
             [dev-dependencies]\npoot-test-util = {{ path = {util:?} }}\n"
        ),
    )
    .unwrap();
    std::fs::write(dir.join("tests/probe.rs"), PROBE.replace("@DIR@", "DIR")).unwrap();
    std::fs::write(dir.join("fixture.txt"), fixture).unwrap();
}

/// The test executable `cargo test --no-run` built in `dir`.
fn build_probe(dir: &Path, target: &Path, lock: &Path) -> PathBuf {
    std::fs::copy(lock, dir.join("Cargo.lock")).unwrap();
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .args(["test", "--no-run", "--offline", "--message-format=json"])
        .current_dir(dir)
        .env("CARGO_TARGET_DIR", target)
        .env_remove("CARGO_MANIFEST_DIR")
        .output()
        .expect("run cargo");
    assert!(
        output.status.success(),
        "building the probe failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter(|line| line.contains("\"executable\":\"") && line.contains("\"name\":\"probe\""))
        .filter_map(|line| {
            let rest = line.split("\"executable\":\"").nth(1)?;
            rest.split('"').next().map(PathBuf::from)
        })
        .next()
        .expect("cargo reported the probe executable")
}

/// What the probe printed for `name` (`HELPER` or `BAKED`) when run as cargo would run it from `checkout`.
fn run_from(executable: &Path, checkout: &Path, test: &str, name: &str) -> String {
    let output = Command::new(executable)
        .args([test, "--exact", "--nocapture"])
        .current_dir(checkout)
        .env("CARGO_MANIFEST_DIR", checkout)
        .output()
        .expect("run the probe");
    assert!(
        output.status.success(),
        "the probe `{test}` run from {} failed:\n{}",
        checkout.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let prefix = format!("{name}=");
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no {name} line in:\n{stdout}"))
        .to_string()
}

/// Mutation: make `poot_test_util::manifest_dir` return a path captured at build time (for example
/// the compile-time form of the manifest directory); the helper line then reads `fixture-a` and this goes red.
#[test]
fn a_binary_built_in_one_checkout_reads_the_files_of_the_checkout_it_runs_from() {
    let root = poot_test_util::unique_temp_path("binary-built-elsewhere");
    let (a, b) = (root.join("a"), root.join("b"));
    let util = poot_test_util::manifest_dir();
    probe_crate(&a, "fixture-a", &util);
    probe_crate(&b, "fixture-b", &util);
    let lock = poot_test_util::repo_root().join("Cargo.lock");
    let executable = build_probe(&a, &root.join("target"), &lock);

    let helper = run_from(&executable, &b, "reads_through_the_shared_helper", "HELPER");
    let baked = run_from(
        &executable,
        &b,
        "reads_through_the_compile_time_macro",
        "BAKED",
    );
    assert_eq!(
        baked, "fixture-a",
        "the control must show the binary carries A's path"
    );
    assert_eq!(
        helper, "fixture-b",
        "the helper must follow the run, not the build"
    );
}
