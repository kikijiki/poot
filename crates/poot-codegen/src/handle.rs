//! The one place a [`poot_runtime_common::CompiledKernel`] is built (ADR-0102 amendment): every
//! runtime dispatch, launch, load and replay entry point takes a `&CompiledKernel` instead of raw kernel
//! bytes, and this crate is the sole caller of its `unsafe` constructor. Callers pass the artifact bytes
//! [`compile`] or a [`crate::KernelCache`] produced for the same `body`/`target` pair; [`kernel_handle`]
//! derives the entry point, the trap flag and the argument schema straight from `body`, so the schema
//! cannot drift from what the emitter actually compiled.

use poot_kernel_ir::{Body, Ty};
use poot_runtime_common::{AieLinkedArtifact, ArgAccess, ArgSchema, CompiledKernel, KernelCode};
use poot_target::{AieTargetId, Backend, ElementKind};

use crate::Target;
use crate::aie::{AieAbiMismatch, check_body_abi};

/// The native [`ElementKind`] a kernel buffer parameter's slice element type lowers to, or `None` for a
/// type no runtime buffer carries. Every buffer parameter emitted for wgpu/Vulkan/ROCm/PTX dispatch is one
/// of the four named here.
fn element_kind_of(ty: &Ty) -> Option<ElementKind> {
    match ty {
        Ty::F32 => Some(ElementKind::F32),
        // A packed-word buffer's native representation carries i32 and u32 payloads identically (both
        // 4-byte words the runtime copies and binds the same way; only the kernel body's own typed GEP
        // distinguishes signed from unsigned reads). See `poot_runtime::KernelBuffer::read_only_u32`.
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

/// The targets whose kernel is an artifact the compiler's tools wrote as bytes (SPIR-V, PTX, HSACO).
/// An AIE kernel is not one: it is a validated linked artifact, built by [`aie_kernel_handle`], so
/// [`kernel_handle`] takes this type and a byte artifact can never be presented as an AIE kernel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GpuTarget {
    SpirvVulkan,
    Nvptx,
    AmdGcn(poot_target::AmdArch),
}

impl From<GpuTarget> for Target {
    fn from(target: GpuTarget) -> Target {
        match target {
            GpuTarget::SpirvVulkan => Target::SpirvVulkan,
            GpuTarget::Nvptx => Target::Nvptx,
            GpuTarget::AmdGcn(arch) => Target::AmdGcn(arch),
        }
    }
}

/// An AIE core target has no byte-artifact kernel handle: its kernel is a linked artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} has no byte-artifact kernel handle: its kernel is a linked artifact")]
pub struct NotAByteTarget(pub Target);

impl TryFrom<Target> for GpuTarget {
    type Error = NotAByteTarget;

    fn try_from(target: Target) -> Result<Self, NotAByteTarget> {
        match target {
            Target::SpirvVulkan => Ok(GpuTarget::SpirvVulkan),
            Target::Nvptx => Ok(GpuTarget::Nvptx),
            Target::AmdGcn(arch) => Ok(GpuTarget::AmdGcn(arch)),
            Target::AieCore => Err(NotAByteTarget(target)),
        }
    }
}

impl GpuTarget {
    /// This target's entry-point symbol for `body`: always `"main"` for SpirvVulkan (the emitter hardcodes
    /// it, see `emit/core.rs`), else `body.name` (the AMDGCN `.kd` symbol and the PTX `.visible .entry` name
    /// both come from it).
    fn entry_point(self, body: &Body) -> Box<str> {
        match self {
            GpuTarget::SpirvVulkan => "main".into(),
            GpuTarget::Nvptx | GpuTarget::AmdGcn(_) => body.name.clone().into_boxed_str(),
        }
    }

    fn backend(self) -> Backend {
        match self {
            GpuTarget::SpirvVulkan => Backend::SpirvVulkan,
            GpuTarget::Nvptx => Backend::Nvptx,
            GpuTarget::AmdGcn(arch) => Backend::AmdGcn(arch),
        }
    }

    fn code(self, bytes: Vec<u8>) -> KernelCode {
        match self {
            GpuTarget::SpirvVulkan => KernelCode::spirv_from_le_bytes(&bytes),
            GpuTarget::Nvptx => {
                let text = String::from_utf8(bytes)
                    .expect("PTX codegen output is not valid UTF-8 (a toolchain regression)");
                KernelCode::Ptx(text.into_boxed_str())
            }
            GpuTarget::AmdGcn(_) => KernelCode::Hsaco(bytes.into_boxed_slice()),
        }
    }
}

/// The reverse mapping, total and lossless: every [`Backend`] a device capability descriptor or a planned
/// [`poot_target::Backend`]-tagged `Target` names has exactly one codegen [`Target`] (the one AIE target
/// id is a closed enum, so no id is dropped). The executor engine uses this to pick which `Target` to
/// compile a program's bodies for from the device's own backend, instead of writing its own copy.
impl From<Backend> for Target {
    fn from(backend: Backend) -> Target {
        match backend {
            Backend::SpirvVulkan => Target::SpirvVulkan,
            Backend::Nvptx => Target::Nvptx,
            Backend::AmdGcn(arch) => Target::AmdGcn(arch),
            Backend::Aie(AieTargetId::Xdna2) => Target::AieCore,
        }
    }
}

/// Build the dispatchable [`CompiledKernel`] for `body`, from the artifact `bytes` [`crate::compile`] or a
/// [`crate::KernelCache`] produced for the exact same `body`/`target` pair (untouched: not decoded,
/// re-encoded, or taken from a different body's entry). This is the one place in the workspace that calls
/// [`CompiledKernel::new`] for a byte artifact; every field it fills is read straight from `body`, so the
/// handle's schema can never drift from what the emitter actually compiled.
///
/// # Panics
///
/// Panics if `body`'s parameters are not all buffer-slice references with a runtime-representable element
/// type - every body this crate's own emitter compiles for wgpu/Vulkan/ROCm/PTX dispatch has that shape,
/// so a body that does not is a codegen bug, not a caller error.
pub fn kernel_handle(body: &Body, target: GpuTarget, bytes: Vec<u8>) -> CompiledKernel {
    let gpu = target;
    let entry_point = gpu.entry_point(body);
    let args = arg_schema(body);
    let has_trap = body.has_trap();
    let code = gpu.code(bytes);
    // SAFETY: `code` is the artifact `compile`/`KernelCache` produced for this exact `body` and `target`
    // (the caller's obligation, stated in this function's doc); `entry_point`, `args` and `has_trap` are
    // read straight from that same `body`, so they describe the compiled code exactly, not a
    // caller-guessed shape.
    unsafe { CompiledKernel::new(gpu.backend(), entry_point, code, args, has_trap) }
}

/// The [`CompiledKernel`] for an AIE `body` and the `artifact` linked for it. Fallible: the artifact's shape
/// ABI must be exactly `body`'s buffer parameters (count, element kind and which ones are written), and a
/// body that can trap has no home in an ABI without an error word. Nothing is wrapped on a mismatch.
pub fn aie_kernel_handle(
    body: &Body,
    artifact: AieLinkedArtifact,
) -> Result<CompiledKernel, AieAbiMismatch> {
    check_body_abi(body, artifact.abi())?;
    let args = try_arg_schema(body).expect("check_body_abi accepted every parameter");
    let entry_point: Box<str> = artifact.entry().into();
    let target = Backend::Aie(artifact.target());
    // SAFETY: `artifact` came out of the artifact validator (an xclbin, a whole instruction stream and a
    // self-consistent shape ABI), and `check_body_abi` proved its bindings are exactly `body`'s buffer
    // parameters in order with the same element kinds and write flags; `args` is read from that same
    // body, and a trapping body was refused, so `has_trap` is false as `Body::has_trap()` says. The
    // generator's proved access ranges are inside every binding's extent (checked at construction).
    Ok(unsafe { CompiledKernel::new(target, entry_point, KernelCode::Aie(artifact), args, false) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_kernel_ir::fixtures::add_kernel;
    use poot_runtime_common::KernelArgError;

    #[test]
    fn add_kernel_schema_is_two_reads_and_one_write_of_f32() {
        let body = add_kernel();
        let handle = kernel_handle(&body, GpuTarget::SpirvVulkan, vec![0u8; 4]);
        assert_eq!(handle.entry_point(), "main");
        assert_eq!(handle.target(), Backend::SpirvVulkan);
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
        let ptx = kernel_handle(&body, GpuTarget::Nvptx, b"// ptx text".to_vec());
        assert_eq!(ptx.entry_point(), body.name);
        let rocm = kernel_handle(
            &body,
            GpuTarget::AmdGcn(crate::AmdArch::gfx1151()),
            vec![0u8; 4],
        );
        assert_eq!(rocm.entry_point(), body.name);
    }

    #[test]
    fn schema_mismatch_from_a_wrong_element_kind_is_reported_by_index() {
        let body = add_kernel();
        let handle = kernel_handle(&body, GpuTarget::SpirvVulkan, vec![0u8; 4]);
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

    /// A byte artifact can never be presented as an AIE kernel: the byte-handle constructor takes
    /// [`GpuTarget`], and the AIE core target does not convert to it.
    #[test]
    fn the_aie_core_target_is_not_a_byte_artifact_target() {
        assert_eq!(
            GpuTarget::try_from(Target::AieCore),
            Err(NotAByteTarget(Target::AieCore))
        );
        assert_eq!(GpuTarget::try_from(Target::Nvptx), Ok(GpuTarget::Nvptx));
    }

    #[test]
    #[should_panic(expected = "must be a whole number of 4-byte words")]
    fn spirv_artifact_bytes_must_be_word_aligned() {
        let body = add_kernel();
        let _ = kernel_handle(&body, GpuTarget::SpirvVulkan, vec![0u8; 5]);
    }
}
