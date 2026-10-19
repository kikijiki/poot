//! Kernel artifacts shared by the engines of one planning pass (POOT-1068).

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use poot_codegen::{ArtifactTooLarge, CompileError, CompiledArtifact, KernelId};
use poot_target::TargetId;

/// Compiled kernel artifacts, keyed by the kernel's content id
/// (`poot_codegen::KernelKey::id`, the same key `Engine::kernel` indexes by), shared by every
/// engine built with [`Engine::with_shared_kernels`](crate::Engine::with_shared_kernels).
///
/// The startup fit plans many candidates, each through its own planning `Engine`; the candidates draw on
/// largely the same kernels, so one table resolves each distinct kernel through the kernel cache once
/// and every later candidate reuses the bytes. Clones share the table. An artifact depends on the backend
/// it was compiled for, so the table is bound to one [`TargetId`] and an engine of another is refused; the
/// per-program artifact limit is checked again on every hit. An engine without a table keeps no artifacts
/// at all: the executing engine's cache behavior is unchanged.
#[derive(Clone, Debug)]
pub struct SharedKernels {
    target: TargetId,
    artifacts: Arc<Mutex<HashMap<KernelId, CompiledArtifact>>>,
}

/// An engine was given a [`SharedKernels`] it cannot use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SharedKernelsMismatch {
    /// The table is bound to another target than the engine's.
    #[error("shared kernels are bound to {table:?}; this engine drives {engine:?}")]
    Target {
        /// The target the table was created for.
        table: TargetId,
        /// The target of the engine that was refused.
        engine: TargetId,
    },
    /// The table holds byte artifacts, and an AIE engine's kernels are linked artifacts.
    #[error("shared kernels hold byte artifacts; the AIE engine for {engine:?} links its kernels")]
    LinkedArtifacts {
        /// The AIE target of the engine that was refused.
        engine: TargetId,
    },
}

impl SharedKernels {
    /// An empty table for engines driving `target`.
    pub fn new(target: TargetId) -> Self {
        Self {
            target,
            artifacts: Arc::default(),
        }
    }

    /// The target this table's artifacts were compiled for.
    pub fn target(&self) -> TargetId {
        self.target
    }

    /// The artifact for `id` and whether `resolve` ran for it. `resolve` runs under the table's
    /// lock, so concurrent engines never resolve one kernel twice; a failed resolution stores nothing. A
    /// stored artifact longer than `max_artifact_bytes` is [`ArtifactTooLarge`], as a cache hit would be.
    pub(crate) fn get_or_resolve(
        &self,
        id: KernelId,
        max_artifact_bytes: NonZeroU64,
        resolve: impl FnOnce() -> Result<CompiledArtifact, CompileError>,
    ) -> Result<(CompiledArtifact, bool), CompileError> {
        let mut artifacts = self.artifacts.lock().expect("shared kernel table");
        let (artifact, resolved) = match artifacts.get(&id) {
            Some(artifact) => (artifact.clone(), false),
            None => {
                let artifact = resolve()?;
                artifacts.insert(id, artifact.clone());
                (artifact, true)
            }
        };
        let length = artifact.len() as u64;
        if length > max_artifact_bytes.get() {
            return Err(ArtifactTooLarge {
                bytes: length,
                limit: max_artifact_bytes.get(),
            }
            .into());
        }
        Ok((artifact, resolved))
    }
}
