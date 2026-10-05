//! The one place a [`poot_runtime_common::CompiledKernel`] is built (card 608, ADR-0102 amendment): every
//! runtime dispatch, launch, load and replay entry point takes a `&CompiledKernel` instead of raw kernel
//! bytes, and this crate is the sole caller of its `unsafe` constructor. Callers pass the artifact bytes
//! [`compile`] or a [`crate::KernelCache`] produced for the same `body`/`target` pair; [`kernel_handle`]
//! derives the entry point, the trap flag and the argument schema straight from `body`, so the schema
//! cannot drift from what the emitter actually compiled.

use poot_kernel_ir::{Body, Ty};
use poot_runtime_common::{ArgAccess, ArgSchema, CompiledKernel, KernelCode};
use poot_target::{Backend, ElementKind};

use crate::Target;

/// The native [`ElementKind`] a kernel buffer parameter's slice element type lowers to. Every buffer
/// parameter emitted for wgpu/Vulkan/ROCm/PTX dispatch is one of these four; anything else is a codegen
/// bug (a `Body` this crate's own emitter could not have produced for these targets).
fn element_kind_of(ty: &Ty) -> ElementKind {
    match ty {
        Ty::F32 => ElementKind::F32,
        // A packed-word buffer's native representation carries i32 and u32 payloads identically (both
        // 4-byte words the runtime copies and binds the same way; only the kernel body's own typed GEP
        // distinguishes signed from unsigned reads). See `poot_runtime::KernelBuffer::read_only_u32`.
        Ty::I32 | Ty::U32 => ElementKind::I32,
        Ty::BF16 => ElementKind::Bf16,
        Ty::F16 => ElementKind::F16,
        other => panic!(
            "kernel buffer parameter element type has no runtime ElementKind: {other:?} \
             (poot-codegen only ever emits F32/I32/U32/BF16/F16 buffer parameters)"
        ),
    }
}

/// Derive the argument schema straight from `body`'s parameters, in binding order: every kernel
/// dispatched through the runtimes' `CompiledKernel` entry points takes only buffer-slice parameters
/// (`&[T]`/`&mut [T]`), inputs first and (for these entry points) a single output last.
fn arg_schema(body: &Body) -> Box<[ArgSchema]> {
    body.params()
        .map(|local| {
            let ty = body.local_ty(local);
            let Ty::Ref { mutable, pointee } = ty else {
                panic!("kernel parameter {local:?} is not a buffer reference: {ty:?}");
            };
            let Ty::Slice(elem) = pointee.as_ref() else {
                panic!("kernel parameter {local:?} is not a slice reference: {ty:?}");
            };
            ArgSchema::new(
                element_kind_of(elem),
                if *mutable {
                    ArgAccess::Write
                } else {
                    ArgAccess::Read
                },
            )
        })
        .collect()
}

/// This target's entry-point symbol for `body`: always `"main"` for SpirvVulkan (the emitter hardcodes
/// it, see `emit/core.rs`), else `body.name` (the AMDGCN `.kd` symbol and the PTX `.visible .entry` name
/// both come from it).
fn entry_point_of(body: &Body, target: Target) -> Box<str> {
    match target {
        Target::SpirvVulkan => "main".into(),
        Target::Nvptx | Target::AmdGcn(_) | Target::AieCore => body.name.clone().into_boxed_str(),
    }
}

fn backend_of(target: Target) -> Backend {
    match target {
        Target::SpirvVulkan => Backend::SpirvVulkan,
        Target::Nvptx => Backend::Nvptx,
        Target::AmdGcn(arch) => Backend::AmdGcn(arch),
        Target::AieCore => {
            panic!(
                "no runtime CompiledKernel for AieCore: the NPU run_bytes path is out of card 608's scope"
            )
        }
    }
}

/// The reverse of [`backend_of`], and total where it is partial: every [`Backend`] a device
/// capability descriptor or a planned [`poot_target::Backend`]-tagged `Target` names has exactly one
/// codegen [`Target`] (there is no `Backend::AieCore`, so this direction never panics). The executor
/// engine (card 546a) uses this to pick which `Target` to compile a program's bodies for from the
/// device's own backend, instead of writing its own copy of this match.
impl From<Backend> for Target {
    fn from(backend: Backend) -> Target {
        match backend {
            Backend::SpirvVulkan => Target::SpirvVulkan,
            Backend::Nvptx => Target::Nvptx,
            Backend::AmdGcn(arch) => Target::AmdGcn(arch),
        }
    }
}

fn kernel_code_of(target: Target, bytes: Vec<u8>) -> KernelCode {
    match target {
        Target::SpirvVulkan => KernelCode::spirv_from_le_bytes(&bytes),
        Target::Nvptx => {
            let text = String::from_utf8(bytes)
                .expect("PTX codegen output is not valid UTF-8 (a toolchain regression)");
            KernelCode::Ptx(text.into_boxed_str())
        }
        Target::AmdGcn(_) => KernelCode::Hsaco(bytes.into_boxed_slice()),
        Target::AieCore => {
            panic!(
                "no runtime CompiledKernel for AieCore: the NPU run_bytes path is out of card 608's scope"
            )
        }
    }
}

/// Build the dispatchable [`CompiledKernel`] for `body`, from the artifact `bytes` [`crate::compile`] or a
/// [`crate::KernelCache`] produced for the exact same `body`/`target` pair (untouched: not decoded,
/// re-encoded, or taken from a different body's entry). This is the one place in the workspace that calls
/// [`CompiledKernel::new`]; every field it fills is read straight from `body`, so the handle's schema can
/// never drift from what the emitter actually compiled.
///
/// # Panics
///
/// Panics if `target` is [`Target::AieCore`] (out of scope: the NPU `run_bytes` path is deferred, decision
/// D4) or if `body`'s parameters are not all buffer-slice references with a runtime-representable element
/// type - every body this crate's own emitter compiles for wgpu/Vulkan/ROCm/PTX dispatch has that shape,
/// so a body that does not is a codegen bug, not a caller error.
pub fn kernel_handle(body: &Body, target: Target, bytes: Vec<u8>) -> CompiledKernel {
    let entry_point = entry_point_of(body, target);
    let args = arg_schema(body);
    let has_trap = body.has_trap();
    let code = kernel_code_of(target, bytes);
    // SAFETY: `code` is the artifact `compile`/`KernelCache` produced for this exact `body` and `target`
    // (the caller's obligation, stated in this function's doc); `entry_point`, `args` and `has_trap` are
    // read straight from that same `body`, so they describe the compiled code exactly, not a
    // caller-guessed shape.
    unsafe { CompiledKernel::new(backend_of(target), entry_point, code, args, has_trap) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_kernel_ir::fixtures::add_kernel;
    use poot_runtime_common::KernelArgError;

    #[test]
    fn add_kernel_schema_is_two_reads_and_one_write_of_f32() {
        let body = add_kernel();
        let handle = kernel_handle(&body, Target::SpirvVulkan, vec![0u8; 4]);
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
        let ptx = kernel_handle(&body, Target::Nvptx, b"// ptx text".to_vec());
        assert_eq!(ptx.entry_point(), body.name);
        let rocm = kernel_handle(
            &body,
            Target::AmdGcn(crate::AmdArch::gfx1151()),
            vec![0u8; 4],
        );
        assert_eq!(rocm.entry_point(), body.name);
    }

    #[test]
    fn schema_mismatch_from_a_wrong_element_kind_is_reported_by_index() {
        let body = add_kernel();
        let handle = kernel_handle(&body, Target::SpirvVulkan, vec![0u8; 4]);
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

    #[test]
    #[should_panic(expected = "AieCore")]
    fn aie_core_target_has_no_compiled_kernel() {
        let body = add_kernel();
        let _ = kernel_handle(&body, Target::AieCore, vec![0u8; 4]);
    }

    #[test]
    #[should_panic(expected = "must be a whole number of 4-byte words")]
    fn spirv_artifact_bytes_must_be_word_aligned() {
        let body = add_kernel();
        let _ = kernel_handle(&body, Target::SpirvVulkan, vec![0u8; 5]);
    }
}
