//! Device-test skip helpers (card 671: `poot-runtime-common::DeviceBackend`'s `open_or_skip[_with]` had no
//! production caller, only device-row tests across the GPU backend crates - dead under the strict dead-pub
//! rule: a test reference is never a use).
//!
//! Reconstructed here from [`DeviceBackend::fail_if_required`] and [`DeviceBackend::label`], the two pieces of
//! the original methods that already had (and keep) a production caller: `fail_if_required` panics with the
//! exact "required {label} device unavailable instead of skipping" message when the backend is required,
//! otherwise hands the error straight back - which is exactly what the deleted methods did before printing
//! their own `SKIP:` line.
//!
//! `skip_unavailable`/`skip_unavailable_with` (the no-value-to-pass-through twins of these) were deleted by
//! card 548: ROCm's own device-row tests (`poot-rocm-gpu/src/tests/helpers.rs`) were their last caller.

use poot_runtime_common::DeviceBackend;

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
            let error = backend.fail_if_required(get, error);
            eprintln!("SKIP: {} device unavailable: {error}", backend.label());
            None
        }
    }
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
}
