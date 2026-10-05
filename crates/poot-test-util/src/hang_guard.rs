//! Opt-in guard for `#[ignore]`d device tests that deliberately hang this box's shared GPU.
//!
//! Several `#[ignore]`d device tests (for example the BLOOM-scale tied-lm_head diagnostics) trip a real
//! wgpu/RADV display-watchdog TDR: a ring timeout, an automatic ring reset, and new GL clients wedged until
//! the compositor restarts (the 2026-10-03 incident). Running a whole package with `--run-ignored all`
//! force-runs them. Each such test calls [`require_gpu_hang_opt_in`] as its first statement, so the `#[ignore]`
//! is a first lock and the environment opt-in is a second: force-running is a deliberate act.

/// The environment variable that lets a deliberately GPU-hanging test run.
pub const ALLOW_GPU_HANG_VARIABLE: &str = "POOT_ALLOW_GPU_HANG";

/// Proof that the caller set `POOT_ALLOW_GPU_HANG=1` and may run a deliberately GPU-hanging test.
pub struct GpuHangOptIn;

/// `Some(GpuHangOptIn)` only when `POOT_ALLOW_GPU_HANG` is exactly `"1"`; otherwise prints a `SKIP:` line
/// naming the variable and returns `None`.
///
/// A `#[ignore]`d test that deliberately hangs or TDRs the shared GPU calls this as its first statement:
///
/// ```text
/// let Some(_allow_hang) = poot_test_util::require_gpu_hang_opt_in("my_hanging_test") else {
///     return;
/// };
/// ```
///
/// The `#[ignore]` stays: a whole-package `--run-ignored all` still skips unless the opt-in is set.
pub fn require_gpu_hang_opt_in(test_name: &str) -> Option<GpuHangOptIn> {
    require_gpu_hang_opt_in_with(|name| std::env::var(name).ok(), test_name)
}

/// [`require_gpu_hang_opt_in`] with an injected environment reader, for a test that sets only one variable.
pub fn require_gpu_hang_opt_in_with(
    get: impl Fn(&str) -> Option<String>,
    test_name: &str,
) -> Option<GpuHangOptIn> {
    if get(ALLOW_GPU_HANG_VARIABLE).as_deref() == Some("1") {
        return Some(GpuHangOptIn);
    }
    eprintln!(
        "SKIP: {test_name} can hang or TDR the shared GPU; set {ALLOW_GPU_HANG_VARIABLE}=1 to run it anyway"
    );
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An environment holding exactly the given variables.
    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    #[test]
    fn a_hang_test_skips_without_the_opt_in_variable() {
        assert!(require_gpu_hang_opt_in_with(env(&[]), "hangs").is_none());
        assert!(
            require_gpu_hang_opt_in_with(env(&[("POOT_REQUIRE_WGPU", "1")]), "hangs").is_none()
        );
    }

    #[test]
    fn a_hang_test_runs_only_when_the_opt_in_variable_is_exactly_one() {
        assert!(
            require_gpu_hang_opt_in_with(env(&[("POOT_ALLOW_GPU_HANG", "1")]), "hangs").is_some()
        );
        assert!(
            require_gpu_hang_opt_in_with(env(&[("POOT_ALLOW_GPU_HANG", "0")]), "hangs").is_none()
        );
        assert!(
            require_gpu_hang_opt_in_with(env(&[("POOT_ALLOW_GPU_HANG", "true")]), "hangs")
                .is_none()
        );
        assert!(
            require_gpu_hang_opt_in_with(env(&[("POOT_ALLOW_GPU_HANG", "")]), "hangs").is_none()
        );
    }
}
