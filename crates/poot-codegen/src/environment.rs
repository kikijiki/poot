//! The process boundary of the kernel compiler: the one place in this crate that reads the process environment.
//!
//! The compiler's tools (`POOT_LLC`, `POOT_AIE_LLC`, `POOT_AIE_IRON`, `POOT_AMD_CLANG`, `POOT_ROCM_DEVICE_LIBS`)
//! and the cache location (`POOT_KERNEL_CACHE`, `POOT_KERNEL_CACHE_DIR`, `XDG_CACHE_HOME`, `HOME`) are properties
//! of the machine the process runs on, not choices of a caller: a nix devshell exports the tools, and the cache
//! is shared by every process of the user. Neither changes what a kernel computes (`epoch` folds the resolved
//! tool and its version into every cache key, so a different tool never serves another tool's artifact).
//!
//! Everything else in the crate takes these as values ([`crate::Toolchain`], [`crate::CacheEnv`]), built from a
//! lookup function, so a test presents an environment without mutating the process's. The `library_environment` test of
//! `poot-test-util` lists this file as the only one in this crate allowed to read it.

/// The value of the process environment variable `name`, when it is set and valid unicode.
pub(crate) fn lookup(name: &str) -> Option<String> {
    std::env::var(name).ok()
}
