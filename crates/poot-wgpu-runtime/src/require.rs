/// `POOT_REQUIRE_WGPU=1` turns the "no wgpu device -> tests skip cleanly" convention into a loud
/// failure. GPU tests guard with
/// `match Context/Executor::new() { Err(e) => return /* skip */ }`, which counts as PASS, and
/// nextest hides passing-test stderr, so a suite run in a broken environment (outside `nix develop`,
/// missing driver, no GPU) reports green while executing nothing. With the var set, a failed context
/// open panics instead (`just test-device-wgpu` sets it). Failure path only: a successful open never
/// reads the environment, and unset/other values change nothing.
///
/// Card 537 (ADR-0104 decision 5): a test-harness convention, not a production execution choice - it
/// changes a test binary's pass/skip/fail verdict on the already-failed "no device" path, never a
/// [`crate::Context`]'s or executor's computed result. Kept, per the Scope exception, alongside the
/// debugging-only variables this card documents in the sibling crates.
pub(crate) fn require_gpu_check<E: std::fmt::Display>(e: E) -> E {
    require_gpu_check_with(|name| std::env::var(name).ok(), e)
}

pub(crate) fn require_gpu_check_with<E: std::fmt::Display>(
    get: impl Fn(&str) -> Option<String>,
    e: E,
) -> E {
    poot_runtime_common::DeviceBackend::Wgpu.fail_if_required(get, e)
}
