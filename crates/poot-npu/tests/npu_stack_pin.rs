//! POOT-1620: the recorded Intel NPU stack pin (`npu-stack.pin`) is the one the flake builds.
//!
//! The NarrowingContract leaves interior precision to the vendor compiler at a pinned version, and the real-device
//! rows (`poot-npu-rows`) fail when the device reports another compiler. This model-free row ties the pin to the
//! flake's `npuUserMode.version`, so changing either one alone is red before any device runs.

const PIN: &str = include_str!("../npu-stack.pin");
const FLAKE: &str = include_str!("../../../flake.nix");

/// The value of the pin file's `key` line.
fn pinned(key: &str) -> &'static str {
    PIN.lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| line.strip_prefix(key)?.strip_prefix(' '))
        .unwrap_or_else(|| panic!("npu-stack.pin has no `{key}` line"))
        .trim()
}

/// The `version` of the flake's `intel-npu-user-mode` derivation.
fn flake_release() -> &'static str {
    let from = FLAKE
        .find("pname = \"intel-npu-user-mode\";")
        .expect("flake.nix names the intel-npu-user-mode derivation");
    FLAKE[from..]
        .lines()
        .find_map(|line| line.trim().strip_prefix("version = \""))
        .and_then(|rest| rest.split('"').next())
        .expect("the derivation has a version")
}

/// The pin's release is the flake's, and its compiler is a `major.minor` pair.
/// Mutation: change the pin's `release` line (or the flake's `version`); the release row goes red naming both.
#[test]
fn the_recorded_pin_matches_the_flake() {
    assert_eq!(
        pinned("release"),
        flake_release(),
        "npu-stack.pin `release` differs from flake.nix `npuUserMode.version`: a vendor bump needs the whole-step \
         rows re-validated, then the pin updated"
    );
    let compiler = pinned("compiler");
    let (major, minor) = compiler.split_once('.').expect("compiler is `major.minor`");
    assert!(
        major.parse::<u16>().is_ok() && minor.parse::<u16>().is_ok(),
        "compiler `{compiler}` is not `major.minor`"
    );
}
