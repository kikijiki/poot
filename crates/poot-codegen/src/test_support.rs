//! Fixtures for the compiler's own rows: a cache entry a test plants.

use crate::{KernelCache, KernelKey};

/// Publish `bytes` under `key` whatever they are: the stale or foreign entry a cache directory can be
/// left holding. The cache's own `store` is crate-private, so a test that needs a wrong entry on disk
/// (and to see it refused) plants it here.
pub fn plant_entry(cache: &KernelCache, key: &KernelKey, bytes: &[u8]) -> std::io::Result<()> {
    cache.store(key, bytes)
}
