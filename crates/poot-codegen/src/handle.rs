//! The one place a [`poot_runtime_common::CompiledKernel`] is built (ADR-0102 amendment): every
//! runtime dispatch, launch, load and replay entry point takes a `&CompiledKernel` instead of raw kernel
//! bytes, and this crate is the sole caller of its `unsafe` constructor. The bytes reach [`kernel_handle`]
//! only inside a [`CompiledArtifact`], the sealed token [`compile_kernel`] and
//! [`crate::KernelCache::load_or_compile`] return: it carries the [`KernelId`] of the body and target it was
//! compiled for, and nothing outside this crate can build one from bytes. [`kernel_handle`] refuses a token
//! of another body, then derives the entry point, the trap flag and the argument schema straight from
//! `body`, so the schema cannot drift from what the emitter actually compiled.

use std::path::Path;
use std::sync::Arc;

use poot_kernel_ir::{Body, Ty};
use poot_runtime_common::{
    AieLinkedArtifact, ArgAccess, ArgSchema, CompiledKernel, KernelCode, VpuGraphArtifact,
};
use poot_target::{AieTargetId, Backend, ElementKind, TargetId};

use crate::{CompileError, KernelId, Target, compile};

/// The native [`ElementKind`] a kernel buffer parameter's slice element type lowers to, or `None` for a
/// type no runtime buffer carries. Every buffer parameter emitted for wgpu/Vulkan/ROCm/PTX dispatch is one
/// of the four named here.
fn element_kind_of(ty: &Ty) -> Option<ElementKind> {
    match ty {
        Ty::F32 => Some(ElementKind::F32),
        // A packed-word buffer's native representation carries i32 and u32 payloads identically (both
        // 4-byte words the runtime copies and binds the same way; only the kernel body's own typed GEP
        // distinguishes signed from unsigned reads). See `poot_wgpu_runtime::KernelBuffer::read_only_u32`.
        Ty::I32 | Ty::U32 => Some(ElementKind::I32),
        Ty::BF16 => Some(ElementKind::Bf16),
        Ty::F16 => Some(ElementKind::F16),
        _ => None,
    }
}

/// Derive the argument schema straight from `body`'s parameters, in binding order: every kernel
/// dispatched through the runtimes' `CompiledKernel` entry points takes only buffer-slice parameters
/// (`&[T]`/`&mut [T]`), inputs first and (for these entry points) a single output last. `Err(index)` names
/// the first parameter that is not a buffer slice of a runtime-representable element type.
pub(crate) fn try_arg_schema(body: &Body) -> Result<Box<[ArgSchema]>, usize> {
    body.params()
        .enumerate()
        .map(|(index, local)| {
            let Ty::Ref { mutable, pointee } = body.local_ty(local) else {
                return Err(index);
            };
            let Ty::Slice(elem) = pointee.as_ref() else {
                return Err(index);
            };
            let element = element_kind_of(elem).ok_or(index)?;
            Ok(ArgSchema::new(
                element,
                if *mutable {
                    ArgAccess::Write
                } else {
                    ArgAccess::Read
                },
            ))
        })
        .collect()
}

/// [`try_arg_schema`] for the GPU targets, whose bodies this crate's own emitter produced: a parameter
/// that is not a runtime buffer slice is a codegen bug, not a caller error.
fn arg_schema(body: &Body) -> Box<[ArgSchema]> {
    try_arg_schema(body).unwrap_or_else(|index| {
        panic!(
            "kernel parameter {index} is not a buffer slice of a runtime-representable element type: {:?} \
             (poot-codegen only ever emits F32/I32/U32/BF16/F16 buffer parameters)",
            body.params().nth(index).map(|l| body.local_ty(l))
        )
    })
}

/// A target with no byte-artifact kernel handle: an AIE core's kernel is a linked artifact, and
/// [`Target::SpirvOpenCl`] has no runtime [`Backend`] to dispatch it (no Level Zero GPU consumer exists; an Intel
/// GPU backend is a future card).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} has no byte-artifact kernel handle")]
pub struct NotAByteTarget(pub Target);

/// The backends whose kernel is an artifact the compiler's tools wrote as bytes (SPIR-V, PTX, HSACO). An AIE
/// kernel is not one: it is a validated linked artifact, built by [`aie_kernel_handle`], so [`kernel_handle`]
/// takes a [`CompiledArtifact`] and a byte artifact can never be presented as an AIE kernel.
impl TryFrom<Target> for Backend {
    type Error = NotAByteTarget;

    fn try_from(target: Target) -> Result<Self, NotAByteTarget> {
        match target {
            Target::SpirvVulkan => Ok(Backend::SpirvVulkan),
            Target::Nvptx => Ok(Backend::Nvptx),
            Target::AmdGcn(arch) => Ok(Backend::AmdGcn(arch)),
            Target::AieCore | Target::SpirvOpenCl => Err(NotAByteTarget(target)),
        }
    }
}

/// `backend`'s entry-point symbol for `body`: always `"main"` for SpirvVulkan (the emitter hardcodes it, see
/// `emit/core.rs`), else `body.name` (the AMDGCN `.kd` symbol and the PTX `.visible .entry` name both come
/// from it).
fn entry_point(backend: Backend, body: &Body) -> Box<str> {
    match backend {
        Backend::SpirvVulkan => "main".into(),
        Backend::Nvptx | Backend::AmdGcn(_) => body.name.clone().into_boxed_str(),
    }
}

fn kernel_code(backend: Backend, bytes: &[u8]) -> KernelCode {
    match backend {
        Backend::SpirvVulkan => KernelCode::spirv_from_le_bytes(bytes),
        Backend::Nvptx => {
            let text = std::str::from_utf8(bytes)
                .expect("PTX codegen output is not valid UTF-8 (a toolchain regression)");
            KernelCode::Ptx(text.into())
        }
        Backend::AmdGcn(_) => KernelCode::Hsaco(bytes.into()),
    }
}

/// The reverse mapping, total and lossless: every [`Backend`] a device capability descriptor or a planned
/// `Target` names has exactly one codegen [`Target`]. The executor engine uses this to pick which `Target`
/// to compile a program's bodies for from the device's own backend, instead of writing its own copy.
impl From<Backend> for Target {
    fn from(backend: Backend) -> Target {
        match backend {
            Backend::SpirvVulkan => Target::SpirvVulkan,
            Backend::Nvptx => Target::Nvptx,
            Backend::AmdGcn(arch) => Target::AmdGcn(arch),
        }
    }
}

/// The codegen [`Target`] of an AIE target (a closed enum, so no id is dropped).
impl From<AieTargetId> for Target {
    fn from(target: AieTargetId) -> Target {
        match target {
            AieTargetId::Xdna2 => Target::AieCore,
        }
    }
}

/// A target identity with no codegen [`Target`]: a VPU compiles a graph the vendor lowers, not a `Body`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} compiles no kernel Body: its regions go through `emit_vpu`")]
pub struct NoBodyTarget(pub TargetId);

/// The codegen [`Target`] of a target identity that compiles kernel bodies (a GPU backend or an AIE
/// target). A VPU target is the typed [`NoBodyTarget`]: its regions are emitted by [`crate::emit_vpu`].
impl TryFrom<TargetId> for Target {
    type Error = NoBodyTarget;

    fn try_from(target: TargetId) -> Result<Target, NoBodyTarget> {
        match target {
            TargetId::Gpu(backend) => Ok(backend.into()),
            TargetId::Aie(target) => Ok(target.into()),
            TargetId::Vpu(_) => Err(NoBodyTarget(target)),
        }
    }
}

/// A compiled byte artifact (SPIR-V, PTX or an HSACO image) sealed to the kernel it was compiled from. Only
/// this crate builds one, from the compiler's own output ([`compile_kernel`]) or from a cache entry that
/// proved it was stored for this kernel ([`crate::KernelCache::load_or_compile`]); its fields are private
/// and [`kernel_handle`] is the only consumer. Cloning shares the bytes.
#[derive(Clone, Debug)]
pub struct CompiledArtifact {
    id: KernelId,
    target: Backend,
    bytes: Arc<[u8]>,
    built: bool,
}

impl CompiledArtifact {
    pub(crate) fn new(id: KernelId, target: Backend, bytes: Vec<u8>, built: bool) -> Self {
        Self {
            id,
            target,
            bytes: bytes.into(),
            built,
        }
    }

    /// The content id of the body and target this artifact was compiled for.
    pub fn id(&self) -> KernelId {
        self.id
    }

    /// The target this artifact was compiled for.
    pub fn target(&self) -> Backend {
        self.target
    }

    /// The artifact's length in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the artifact is empty (a compiler that produced nothing).
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// `true` when producing this artifact ran the compiler, `false` when a verified cache entry served it.
    pub fn built(&self) -> bool {
        self.built
    }
}

/// Compile `body` for `target` into `out` and return the sealed artifact read back from it. The one entry
/// for a caller that compiles without a [`crate::KernelCache`]; `out`'s directory must exist.
pub fn compile_kernel(
    body: &Body,
    target: Backend,
    out: &Path,
) -> Result<CompiledArtifact, CompileError> {
    let codegen = Target::from(target);
    compile(body, codegen, out)?;
    let bytes = std::fs::read(out)?;
    Ok(CompiledArtifact::new(
        KernelId::of(body, codegen),
        target,
        bytes,
        true,
    ))
}

/// A [`CompiledArtifact`] was presented with a body other than the one it was compiled from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "artifact {artifact} was compiled from another body or target than the one presented ({body})"
)]
pub struct ArtifactMismatch {
    /// The id the artifact carries.
    pub artifact: KernelId,
    /// The id of the body and target the handle was requested for.
    pub body: KernelId,
}

/// Build the dispatchable [`CompiledKernel`] for `body` from the sealed `artifact` compiled for it. This
/// is the one place in the workspace that calls [`CompiledKernel::new`] for a byte artifact; every field it
/// fills is read straight from `body`, so the handle's schema can never drift from what the emitter
/// actually compiled. An artifact compiled from a different body (or for a different target) is an
/// [`ArtifactMismatch`], never a handle.
///
/// There is no way to pass raw bytes:
///
/// ```compile_fail
/// use poot_codegen::kernel_handle;
/// use poot_target::Backend;
///
/// fn raw_bytes(body: &poot_kernel_ir::Body) {
///     let _ = kernel_handle(body, Backend::SpirvVulkan, vec![0u8; 4]);
/// }
/// ```
///
/// # Panics
///
/// Panics if `body`'s parameters are not all buffer-slice references with a runtime-representable element
/// type - every body this crate's own emitter compiles for wgpu/Vulkan/ROCm/PTX dispatch has that shape,
/// so a body that does not is a codegen bug, not a caller error.
pub fn kernel_handle(
    body: &Body,
    artifact: &CompiledArtifact,
) -> Result<CompiledKernel, ArtifactMismatch> {
    let gpu = artifact.target;
    let expected = KernelId::of(body, gpu.into());
    if artifact.id != expected {
        return Err(ArtifactMismatch {
            artifact: artifact.id,
            body: expected,
        });
    }
    let entry_point = entry_point(gpu, body);
    let args = arg_schema(body);
    let has_trap = body.has_trap();
    let code = kernel_code(gpu, &artifact.bytes);
    // SAFETY: `artifact` is only constructible inside this crate, from the compiler's output for a body and
    // target or from a cache entry whose recorded id matched, and it carries the id of that body and
    // target; the id was just checked equal to `body`'s on that target, so `code` is the artifact compiled
    // from this exact `body`. `entry_point`, `args` and `has_trap` are read straight from that same
    // `body`, so they describe the compiled code exactly, not a caller-guessed shape.
    Ok(unsafe { CompiledKernel::new(TargetId::Gpu(gpu), entry_point, code, args, has_trap) })
}

/// The [`CompiledKernel`] for an AIE region's linked `artifact`. Its arguments are the artifact's shape
/// ABI, binding for binding: the host side of the region. Which core runs which body was checked tile by
/// tile when the artifact was compiled (`compile_aie`), and the artifact itself is only constructible
/// through the validator, so there is nothing left to refuse here.
pub fn aie_kernel_handle(artifact: AieLinkedArtifact) -> CompiledKernel {
    let args: Box<[ArgSchema]> = artifact
        .abi()
        .bindings()
        .iter()
        .map(|binding| ArgSchema::new(binding.storage().element(), binding.access()))
        .collect();
    let entry_point: Box<str> = artifact.entry().into();
    let target = TargetId::Aie(artifact.target());
    // SAFETY: `artifact` came out of the artifact validator (an xclbin, a whole instruction stream and a
    // self-consistent shape ABI), and `args` is that ABI's bindings in order with their element kinds and
    // write flags, so the schema describes the artifact exactly. Every tile body was checked against the
    // recipe the ABI belongs to at compile time, and a trapping body was refused there, so `has_trap` is
    // false. The generator's proved access ranges are inside every binding's extent (checked at
    // construction).
    unsafe { CompiledKernel::new(target, entry_point, KernelCode::Aie(artifact), args, false) }
}

/// The [`CompiledKernel`] for an Intel NPU region's `artifact`. Its arguments are the artifact's inputs and then
/// its outputs: the region reads the first and writes the rest.
pub fn vpu_kernel_handle(artifact: VpuGraphArtifact) -> CompiledKernel {
    let args: Box<[ArgSchema]> = artifact
        .inputs()
        .iter()
        .map(|b| ArgSchema::new(b.storage().element(), ArgAccess::Read))
        .chain(
            artifact
                .outputs()
                .iter()
                .map(|b| ArgSchema::new(b.storage().element(), ArgAccess::Write)),
        )
        .collect();
    let target = TargetId::Vpu(artifact.target());
    // SAFETY: `artifact` is only constructible through `VpuGraphArtifact::new`, which admits only dense F16,
    // F32 and I32 bindings of non-empty shape, and `args` is exactly its inputs (read) then outputs
    // (written) in order, so the schema describes the artifact. The graph has no trap word, and the device
    // checks the compiled graph's own argument list against these bindings before it executes.
    unsafe {
        CompiledKernel::new(
            target,
            "poot_region",
            KernelCode::Vpu(artifact),
            args,
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;
    use crate::KernelCache;
    use poot_kernel_ir::fixtures::add_kernel;
    use poot_runtime_common::KernelArgError;

    /// The sealed artifact a cache hands out for `body` when `bytes` were stored under its key: the same
    /// path every production handle takes (`load_or_compile`), served from the cache so no toolchain is
    /// needed.
    fn cached_artifact(name: &str, body: &Body, target: Backend, bytes: &[u8]) -> CompiledArtifact {
        let dir =
            std::env::temp_dir().join(format!("poot-handle-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = KernelCache::at(&dir, target.into());
        cache.store(&cache.key(body), bytes).unwrap();
        let artifact = cache
            .load_or_compile(body, NonZeroU64::new(1 << 20).unwrap())
            .expect("a stored entry is a hit");
        assert!(!artifact.built(), "the artifact must come from the cache");
        let _ = std::fs::remove_dir_all(&dir);
        artifact
    }

    #[test]
    fn add_kernel_schema_is_two_reads_and_one_write_of_f32() {
        let body = add_kernel();
        let artifact = cached_artifact("schema", &body, Backend::SpirvVulkan, &[0u8; 4]);
        let handle = kernel_handle(&body, &artifact).unwrap();
        assert_eq!(handle.entry_point(), "main");
        assert_eq!(handle.target(), TargetId::Gpu(Backend::SpirvVulkan));
        let args = handle.args();
        assert_eq!(args.len(), 3, "two inputs + one output");
        assert_eq!(args[0].element(), ElementKind::F32);
        assert_eq!(args[0].access(), ArgAccess::Read);
        assert_eq!(args[1].element(), ElementKind::F32);
        assert_eq!(args[1].access(), ArgAccess::Read);
        assert_eq!(args[2].element(), ElementKind::F32);
        assert_eq!(args[2].access(), ArgAccess::Write);
    }

    #[test]
    fn ptx_and_rocm_entry_points_are_the_body_name_not_main() {
        let body = add_kernel();
        let ptx_artifact = cached_artifact("ptx", &body, Backend::Nvptx, b"// ptx text");
        let ptx = kernel_handle(&body, &ptx_artifact).unwrap();
        assert_eq!(ptx.entry_point(), body.name);
        let rocm_target = Backend::AmdGcn(crate::AmdArch::gfx1151());
        let rocm_artifact = cached_artifact("rocm", &body, rocm_target, &[0u8; 4]);
        let rocm = kernel_handle(&body, &rocm_artifact).unwrap();
        assert_eq!(rocm.entry_point(), body.name);
    }

    #[test]
    fn schema_mismatch_from_a_wrong_element_kind_is_reported_by_index() {
        let body = add_kernel();
        let artifact = cached_artifact("elements", &body, Backend::SpirvVulkan, &[0u8; 4]);
        let handle = kernel_handle(&body, &artifact).unwrap();
        // F32 and I32 share a 4-byte native word and are schema-compatible by design (see
        // `poot_runtime_common::kernel::element_kinds_compatible`); Bf16's 2-byte word is a genuine
        // width mismatch against this kernel's F32 schema.
        let wrong = [ElementKind::F32, ElementKind::Bf16, ElementKind::F32];
        assert_eq!(
            poot_runtime_common::check_kernel_args(&handle, &wrong),
            Err(KernelArgError::ElementKind {
                index: 1,
                expected: ElementKind::F32,
                actual: ElementKind::Bf16,
            })
        );
    }

    /// SC-002: an artifact compiled from body A cannot build a handle for body B, nor for the same body on
    /// another target.
    #[test]
    fn an_artifact_of_another_body_or_target_builds_no_handle() {
        let body_a = add_kernel();
        let mut body_b = add_kernel();
        body_b.name = "another_kernel".to_string();
        let artifact = cached_artifact("mismatch", &body_a, Backend::SpirvVulkan, &[0u8; 4]);

        let wrong_body = kernel_handle(&body_b, &artifact).unwrap_err();
        assert_eq!(wrong_body.artifact, artifact.id());
        assert_ne!(wrong_body.body, artifact.id());

        let nvptx = cached_artifact("mismatch-ptx", &body_a, Backend::Nvptx, b"// ptx text");
        assert_ne!(nvptx.id(), artifact.id(), "the target is part of the id");
        assert!(kernel_handle(&body_a, &nvptx).is_ok());
    }

    /// A byte artifact can never be presented as an AIE kernel: the byte-handle constructor takes
    /// [`Backend`], and the AIE core target does not convert to it.
    #[test]
    fn the_aie_core_target_is_not_a_byte_artifact_target() {
        assert_eq!(
            Backend::try_from(Target::AieCore),
            Err(NotAByteTarget(Target::AieCore))
        );
        assert_eq!(Backend::try_from(Target::Nvptx), Ok(Backend::Nvptx));
    }

    #[test]
    #[should_panic(expected = "must be a whole number of 4-byte words")]
    fn spirv_artifact_bytes_must_be_word_aligned() {
        let body = add_kernel();
        let artifact = cached_artifact("misaligned", &body, Backend::SpirvVulkan, &[0u8; 5]);
        let _ = kernel_handle(&body, &artifact);
    }
}
