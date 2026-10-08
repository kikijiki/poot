//! Kernel artifacts shared by the engines of one planning pass (POOT-1068).

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use poot_codegen::{ArtifactTooLarge, CompileError};
use poot_target::Backend;

/// Compiled kernel artifacts, keyed by the kernel body's content fingerprint
/// (`poot_codegen::KernelKey::fingerprint`, the same key `Engine::kernel` indexes by), shared by every
/// engine built with [`Engine::with_shared_kernels`](crate::Engine::with_shared_kernels).
///
/// The startup fit plans many candidates, each through its own planning `Engine`; the candidates draw on
/// largely the same kernels, so one table resolves each distinct kernel through the kernel cache once
/// and every later candidate reuses the bytes. Clones share the table. An artifact depends on the backend
/// it was compiled for, so the table is bound to one [`Backend`] and an engine of another is refused; the
/// per-program artifact limit is checked again on every hit. An engine without a table keeps no artifacts
/// at all: the executing engine's cache behavior is unchanged.
#[derive(Clone, Debug)]
pub struct SharedKernels {
    backend: Backend,
    artifacts: Arc<Mutex<HashMap<u64, Arc<[u8]>>>>,
}

/// An engine for one backend was given a [`SharedKernels`] bound to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("shared kernels are bound to {table:?}; this engine drives {engine:?}")]
pub struct SharedKernelsMismatch {
    pub table: Backend,
    pub engine: Backend,
}

impl SharedKernels {
    /// An empty table for engines driving `backend`.
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            artifacts: Arc::default(),
        }
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// The artifact for `fingerprint` and whether `resolve` ran for it. `resolve` runs under the table's
    /// lock, so concurrent engines never resolve one kernel twice; a failed resolution stores nothing. A
    /// stored artifact longer than `max_artifact_bytes` is [`ArtifactTooLarge`], as a cache hit would be.
    pub(crate) fn get_or_resolve(
        &self,
        fingerprint: u64,
        max_artifact_bytes: NonZeroU64,
        resolve: impl FnOnce() -> Result<Vec<u8>, CompileError>,
    ) -> Result<(Arc<[u8]>, bool), CompileError> {
        let mut artifacts = self.artifacts.lock().expect("shared kernel table");
        let (bytes, resolved) = match artifacts.get(&fingerprint) {
            Some(bytes) => (Arc::clone(bytes), false),
            None => {
                let bytes: Arc<[u8]> = resolve()?.into();
                artifacts.insert(fingerprint, Arc::clone(&bytes));
                (bytes, true)
            }
        };
        let length = bytes.len() as u64;
        if length > max_artifact_bytes.get() {
            return Err(ArtifactTooLarge {
                bytes: length,
                limit: max_artifact_bytes.get(),
            }
            .into());
        }
        Ok((bytes, resolved))
    }
}
