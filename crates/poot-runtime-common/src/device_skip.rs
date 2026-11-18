//! Device-test skip helpers (card 671: `poot-runtime-common::DeviceBackend`'s `open_or_skip[_with]` had no
//! production caller, only device-row tests across the GPU backend crates - dead under the strict dead-pub
//! rule: a test reference is never a use).
//!
//! [`fail_if_required`] panics with the exact "required {label} device unavailable instead of skipping" message
//! when the backend is required, otherwise hands the error straight back; the skip helpers print their own
//! `POOT-SKIP:` line after it. Device constructors never read the switch: a test opens through these helpers.
//!
//! `skip_unavailable`/`skip_unavailable_with` (the no-value-to-pass-through twins of these) were deleted by
//! card 548: ROCm's own device-row tests (`poot-rocm-device/src/tests/helpers.rs`) were their last caller.

use crate::DeviceBackend;

/// Pass `error` back, or panic when `backend` is required: the failure message of every required device that is
/// not there. A skip guard that handed the error on would turn a broken environment into a silent pass.
/// `get` reads the environment (injected so a test can set only one variable).
pub fn fail_if_required<E: std::fmt::Display>(
    backend: DeviceBackend,
    get: impl Fn(&str) -> Option<String>,
    error: E,
) -> E {
    if backend.required(get) {
        panic!(
            "required {} device unavailable instead of skipping ({}=1): {error}",
            backend.label(),
            backend.variable(),
        );
    }
    error
}

/// Pass a successfully opened device through, or skip (panicking first when `backend` is required).
pub fn open_or_skip<T, E: std::fmt::Display>(
    backend: DeviceBackend,
    opened: Result<T, E>,
) -> Option<T> {
    open_or_skip_with(backend, |name| std::env::var(name).ok(), opened)
}

/// [`open_or_skip`] with an injected environment reader, for a test that sets only one variable.
pub fn open_or_skip_with<T, E: std::fmt::Display>(
    backend: DeviceBackend,
    get: impl Fn(&str) -> Option<String>,
    opened: Result<T, E>,
) -> Option<T> {
    match opened {
        Ok(device) => Some(device),
        Err(error) => {
            let error = fail_if_required(backend, get, error);
            eprintln!("POOT-SKIP: {} device unavailable: {error}", backend.label());
            None
        }
    }
}

/// Skip a row whose device opened but lacks a capability the row needs (`what`), panicking first when `backend` is
/// required: a lane that claims the device claims the capability too.
pub fn skip_unsupported(backend: DeviceBackend, what: &str) {
    skip_unsupported_with(backend, |name| std::env::var(name).ok(), what);
}

/// [`skip_unsupported`] with an injected environment reader, for a test that sets only one variable.
pub fn skip_unsupported_with(
    backend: DeviceBackend,
    get: impl Fn(&str) -> Option<String>,
    what: &str,
) {
    let what = fail_if_required(backend, get, what);
    eprintln!("POOT-SKIP: {} device lacks {what}", backend.label());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An environment that sets exactly the named variables to `1`.
    fn env_with(names: &'static [&'static str]) -> impl Fn(&str) -> Option<String> {
        move |name| names.contains(&name).then(|| "1".to_string())
    }

    /// Card 671: these two tests moved here from poot-runtime-common (which deleted
    /// skip_unavailable[_with]/open_or_skip[_with] as dead production code) - the
    /// Option-wrapping/skip-printing behavior this module reconstructs.
    #[test]
    fn a_device_that_opened_is_returned_even_when_required() {
        let opened: Result<u32, &str> = Ok(7);
        assert_eq!(
            open_or_skip_with(DeviceBackend::Ptx, env_with(&["POOT_REQUIRE_PTX"]), opened),
            Some(7)
        );
    }

    #[test]
    fn a_missing_device_is_a_skip_when_the_backend_is_not_required() {
        for backend in DeviceBackend::ALL {
            let opened: Result<u32, &str> = Err("no device");
            assert_eq!(open_or_skip_with(backend, |_| None, opened), None);
        }
        // Another backend's variable does not turn this backend's skip into a failure.
        let opened: Result<u32, &str> = Err("no device");
        assert_eq!(
            open_or_skip_with(DeviceBackend::Ptx, env_with(&["POOT_REQUIRE_ROCM"]), opened),
            None
        );
    }

    #[test]
    #[should_panic(expected = "required wgpu device unavailable")]
    fn a_missing_capability_fails_when_the_backend_is_required() {
        skip_unsupported_with(
            DeviceBackend::Wgpu,
            env_with(&["POOT_REQUIRE_WGPU"]),
            "TIMESTAMP_QUERY",
        );
    }

    /// The skip guard hands the error back untouched unless the backend is required.
    #[test]
    fn fail_if_required_passes_the_error_through_unless_required() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        assert_eq!(
            fail_if_required(DeviceBackend::Rocm, |_| None, "no agent"),
            "no agent"
        );
        let payload = catch_unwind(AssertUnwindSafe(|| {
            fail_if_required(
                DeviceBackend::Rocm,
                env_with(&["POOT_REQUIRE_ROCM"]),
                "no agent",
            );
        }))
        .expect_err("a required backend must fail the open");
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .expect("device backend panics carry a message");
        assert!(
            message.contains("required ROCm device unavailable"),
            "{message}"
        );
        assert!(message.contains("POOT_REQUIRE_ROCM=1"), "{message}");
        assert!(message.contains("no agent"), "{message}");
    }

    #[test]
    fn a_missing_capability_is_a_skip_when_the_backend_is_not_required() {
        skip_unsupported_with(DeviceBackend::Wgpu, |_| None, "TIMESTAMP_QUERY");
        skip_unsupported_with(
            DeviceBackend::Wgpu,
            env_with(&["POOT_REQUIRE_ROCM"]),
            "TIMESTAMP_QUERY",
        );
    }
}
