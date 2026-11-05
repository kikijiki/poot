//! The host toolchain a codegen test shells out to (`llc`, `spirv-val`, `spirv-dis`, the AMD clang).
//!
//! The devshell provides every one of them (`flake.nix`), so a missing tool is a broken environment, never a
//! reason to pass: a test that returned early without `llc` turned a devshell regression green with its
//! NVPTX, SPIR-V or AMDGCN half unexecuted (POOT-1467). A test calls [`require_tools`] or [`require_amd_clang`]
//! and runs on; there is no skip and no environment switch that makes one.

use std::process::Command;

/// Whether `tool` starts and answers `--version`.
fn runs(tool: impl AsRef<std::ffi::OsStr>) -> bool {
    Command::new(tool).arg("--version").output().is_ok()
}

/// The tools of `tools` that cannot be started.
fn missing<'a>(tools: &[&'a str], runs: impl Fn(&str) -> bool) -> Vec<&'a str> {
    tools.iter().copied().filter(|tool| !runs(tool)).collect()
}

/// Panic unless every tool in `tools` runs.
///
/// # Panics
///
/// Names the missing tools and the devshell that provides them.
pub fn require_tools(tools: &[&str]) {
    let missing = missing(tools, |tool| runs(tool));
    assert!(
        missing.is_empty(),
        "toolchain not on PATH: {}; the test needs it and the devshell provides it (run inside `nix develop`)",
        missing.join(", ")
    );
}

/// The AMD-flavored clang named by `POOT_AMD_CLANG`, which `compile()` for `Target::AmdGcn` needs.
///
/// # Panics
///
/// When `POOT_AMD_CLANG` is unset or names a clang that does not run.
pub fn require_amd_clang() -> std::path::PathBuf {
    let clang = std::env::var_os("POOT_AMD_CLANG")
        .expect("POOT_AMD_CLANG is not set; the devshell exports it (run inside `nix develop`)");
    assert!(
        runs(&clang),
        "POOT_AMD_CLANG={} does not run; the devshell exports a working one",
        clang.to_string_lossy()
    );
    clang.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_tools_that_do_not_run_are_missing() {
        let runs = |tool: &str| tool == "llc";
        assert_eq!(
            missing(&["llc", "spirv-val", "spirv-dis"], runs),
            ["spirv-val", "spirv-dis"]
        );
        assert!(missing(&["llc"], runs).is_empty());
    }

    #[test]
    #[should_panic(expected = "toolchain not on PATH: poot-no-such-tool")]
    fn a_tool_that_does_not_exist_fails_the_test() {
        require_tools(&["poot-no-such-tool"]);
    }
}
