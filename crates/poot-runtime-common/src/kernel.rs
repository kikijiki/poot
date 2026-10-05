//! [`CompiledKernel`]: the one kernel handle every runtime dispatch takes, produced only by
//! `poot-codegen` (card 608, ADR-0102 amendment). Before this type, wgpu's `dispatch`/`dispatch_dev`/
//! `launch`/`submit_dispatches`/`build_cached_dispatch`, ROCm's `load_hsaco`, and PTX's `dispatch_dev`/
//! `dispatch_cooperative` took raw kernel bytes (SPIR-V words, an HSACO ELF, or PTX text) with no
//! argument-schema check; only raw Vulkan's `SpirvKernel` (card 516b) recorded even a buffer count. Any
//! safe caller could hand a dispatch call arbitrary bytes, or the wrong number or native format of
//! buffers, and make the device read or write out of bounds.
//!
//! `CompiledKernel`'s fields are private and its only constructor is `unsafe`
//! ([`CompiledKernel::new`]). `poot-codegen` depends on this leaf crate and is the one caller of that
//! constructor (an `unsafe` block stating the invariant), so safe code outside the compiler cannot mint a
//! handle:
//!
//! ```compile_fail,E0133
//! fn from_any_bytes(code: poot_runtime_common::KernelCode) -> poot_runtime_common::CompiledKernel {
//!     poot_runtime_common::CompiledKernel::new(
//!         poot_target::Backend::SpirvVulkan,
//!         "main",
//!         code,
//!         Vec::new(),
//!         false,
//!     )
//! }
//! ```
//!
//! Every dispatch, launch, load and replay entry point card 608 touches takes a `&CompiledKernel` and
//! checks the bound buffers' count and native element kind against [`CompiledKernel::args`] in release
//! builds, returning a typed error on a mismatch.

use poot_target::{Backend, ElementKind};

/// Whether a kernel argument is read, or written back to the caller's buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArgAccess {
    Read,
    Write,
}

/// One kernel argument's contract: the native word format the compiled code reads or writes there, and
/// whether the kernel writes it back. poot's kernel ABI takes no bare scalars: every argument binds a
/// whole device buffer, one schema entry per data buffer in binding order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArgSchema {
    element: ElementKind,
    access: ArgAccess,
}

impl ArgSchema {
    pub const fn new(element: ElementKind, access: ArgAccess) -> Self {
        Self { element, access }
    }

    pub const fn element(&self) -> ElementKind {
        self.element
    }

    pub const fn access(&self) -> ArgAccess {
        self.access
    }
}

/// The compiled machine code for one target, in the form its runtime loads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelCode {
    /// SPIR-V words (wgpu's `create_shader_module_passthrough`, `poot-vulkan-runtime`'s
    /// `create_shader_module`). Decoded from the little-endian artifact bytes once, here, instead of by
    /// every dispatch call site.
    SpirvWords(Box<[u32]>),
    /// PTX module text (`cuModuleLoadData` accepts the textual image directly).
    Ptx(Box<str>),
    /// An HSACO ELF (`hsa_code_object_reader_create_from_memory`).
    Hsaco(Box<[u8]>),
}

impl KernelCode {
    /// Decode little-endian SPIR-V bytes into [`KernelCode::SpirvWords`]: the one decode every SPIR-V
    /// caller uses (`poot-codegen`'s `kernel_handle`, the `#[kernel]` host wrapper), instead of each
    /// repeating the `chunks_exact(4)` conversion.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` is not a whole number of 4-byte words: every real SPIR-V module is, so this
    /// signals a caller passing something other than compiled SPIR-V.
    pub fn spirv_from_le_bytes(bytes: &[u8]) -> Self {
        assert!(
            bytes.len().is_multiple_of(4),
            "SPIR-V bytes must be a whole number of 4-byte words, got {}",
            bytes.len()
        );
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        KernelCode::SpirvWords(words.into_boxed_slice())
    }
}

/// A kernel a runtime may dispatch: the target it was compiled for, its entry point, its argument
/// schema, and whether its body can trap (an `Assert`/`Unreachable` reserves one more error-word binding
/// the runtime zeroes before dispatch and checks after). See the module doc for the soundness contract.
#[derive(Clone, Debug)]
pub struct CompiledKernel {
    target: Backend,
    entry_point: Box<str>,
    code: KernelCode,
    args: Box<[ArgSchema]>,
    has_trap: bool,
}

impl CompiledKernel {
    /// Wrap compiled kernel code as a dispatchable handle.
    ///
    /// # Safety
    ///
    /// `code` must be machine code `poot-codegen` compiled for `target` from a
    /// [`poot_kernel_ir::Body`](https://docs.rs/poot-kernel-ir) that passed `Body::verify`, whose entry
    /// point is `entry_point`, whose parameters are exactly `args` in binding order (each argument's
    /// native element kind and whether the kernel writes it), and whose `has_trap` equals
    /// `Body::has_trap()` for that body. Every access the code makes to argument `i` must stay within
    /// the bound buffer's element count at dispatch time (the codegen-side half of that contract: the
    /// runtime bounds the length it publishes to the code from the buffer's own recorded capacity).
    pub unsafe fn new(
        target: Backend,
        entry_point: impl Into<Box<str>>,
        code: KernelCode,
        args: impl Into<Box<[ArgSchema]>>,
        has_trap: bool,
    ) -> Self {
        CompiledKernel {
            target,
            entry_point: entry_point.into(),
            code,
            args: args.into(),
            has_trap,
        }
    }

    pub fn target(&self) -> Backend {
        self.target
    }

    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    pub fn code(&self) -> &KernelCode {
        &self.code
    }

    /// This kernel's argument schema, one entry per data buffer it binds, in binding order.
    pub fn args(&self) -> &[ArgSchema] {
        &self.args
    }

    /// Whether this kernel's body can trap (see the struct doc): when true, dispatch must bind one more
    /// reserved error-word binding after the data buffers.
    pub fn has_trap(&self) -> bool {
        self.has_trap
    }
}

/// A dispatch's bound buffers did not meet a [`CompiledKernel`]'s argument schema (card 608, SC-002): the
/// one error every runtime's schema check returns, before any submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelArgError {
    /// The caller bound a different number of buffers than the schema declares.
    Count { expected: usize, actual: usize },
    /// Argument `index`'s bound buffer has a different native element kind than the schema declares.
    ElementKind {
        index: usize,
        expected: ElementKind,
        actual: ElementKind,
    },
}

impl std::fmt::Display for KernelArgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelArgError::Count { expected, actual } => write!(
                f,
                "kernel argument count mismatch: schema declares {expected}, dispatch bound {actual}"
            ),
            KernelArgError::ElementKind {
                index,
                expected,
                actual,
            } => write!(
                f,
                "kernel argument {index} element-kind mismatch: schema declares {expected}, bound buffer is {actual}"
            ),
        }
    }
}

impl std::error::Error for KernelArgError {}

/// Whether a bound buffer's native element kind meets an argument's schema entry: exact agreement, or
/// the same native word width. `poot`'s movement kernels (slice, reshape-materialize, gather, scatter,
/// `dynamic_update_slice`, concat) are dtype-generic at a fixed word width by design - a slice op moves
/// 4-byte words whether the logical payload is F32 or exact-I32 bits, and the runtime's own buffer
/// storage (a separate, narrower contract; see `poot_target::BufferStorage`) already carries the
/// caller's real logical dtype where that distinction matters. What a kernel dispatch itself can be
/// unsound about is a narrower buffer than the kernel's per-thread indexing assumes (BF16/F16's 2-byte
/// word bound to an F32/I32 4-byte argument, or vice versa) - a real out-of-bounds risk this check must
/// catch; same-width words never are.
fn element_kinds_compatible(expected: ElementKind, actual: ElementKind) -> bool {
    expected == actual || expected.byte_width() == actual.byte_width()
}

/// Check `elements` (one native [`ElementKind`] per bound buffer, in binding order) against an argument
/// `schema`: the one release-build check every dispatch, launch, load and replay entry point named in card
/// 608 runs before any submission. Shared so no runtime re-derives its own count/element comparison.
///
/// This is the schema-only half of [`check_kernel_args`], for a caller that only has a `&[ArgSchema]` (card
/// 608): ROCm's `KernelHandle`/`HsacoModule` carry the schema copied out of the
/// `CompiledKernel` `load_hsaco` received, since a ROCm dispatch's kernarg segment is one opaque byte blob
/// by the time it reaches `RocmContext::dispatch`, with no `CompiledKernel` to ask.
pub fn check_element_schema(
    schema: &[ArgSchema],
    elements: &[ElementKind],
) -> Result<(), KernelArgError> {
    if elements.len() != schema.len() {
        return Err(KernelArgError::Count {
            expected: schema.len(),
            actual: elements.len(),
        });
    }
    for (index, (arg, &actual)) in schema.iter().zip(elements).enumerate() {
        if !element_kinds_compatible(arg.element(), actual) {
            return Err(KernelArgError::ElementKind {
                index,
                expected: arg.element(),
                actual,
            });
        }
    }
    Ok(())
}

/// Check `elements` (one native [`ElementKind`] per bound buffer, in binding order) against `kernel`'s
/// argument schema. See [`check_element_schema`] for the comparison itself.
pub fn check_kernel_args(
    kernel: &CompiledKernel,
    elements: &[ElementKind],
) -> Result<(), KernelArgError> {
    check_element_schema(kernel.args(), elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(kinds: &[ElementKind]) -> Box<[ArgSchema]> {
        kinds
            .iter()
            .map(|&element| ArgSchema::new(element, ArgAccess::Read))
            .collect()
    }

    fn kernel(args: Box<[ArgSchema]>) -> CompiledKernel {
        // SAFETY: test-only kernel; never dispatched, only used to check the schema comparison.
        unsafe {
            CompiledKernel::new(
                Backend::SpirvVulkan,
                "main",
                KernelCode::SpirvWords(Box::new([])),
                args,
                false,
            )
        }
    }

    /// `check_kernel_args` is a thin wrapper over `check_element_schema(kernel.args(), ..)` (card 608): a caller with only a bare schema, not a whole `CompiledKernel` (ROCm's
    /// `KernelHandle`), gets the identical count/width comparison.
    #[test]
    fn check_element_schema_matches_check_kernel_args_on_the_same_schema() {
        let schema = schema(&[ElementKind::F32, ElementKind::Bf16]);
        let k = kernel(schema.clone());
        let elements = [ElementKind::F32, ElementKind::F16];
        assert_eq!(
            check_element_schema(&schema, &elements),
            check_kernel_args(&k, &elements)
        );
    }

    #[test]
    fn matching_elements_pass() {
        let k = kernel(schema(&[ElementKind::F32, ElementKind::I32]));
        assert_eq!(
            check_kernel_args(&k, &[ElementKind::F32, ElementKind::I32]),
            Ok(())
        );
    }

    #[test]
    fn fewer_buffers_than_schema_is_a_count_error() {
        let k = kernel(schema(&[ElementKind::F32, ElementKind::F32]));
        assert_eq!(
            check_kernel_args(&k, &[ElementKind::F32]),
            Err(KernelArgError::Count {
                expected: 2,
                actual: 1
            })
        );
    }

    /// A movement kernel typed `Slice<f32>` generically (poot's slice/reshape/concat/scatter bodies)
    /// still accepts an I32-tagged buffer, and vice versa: same 4-byte native word, no OOB risk. Only a
    /// different WIDTH is a schema violation (see `wrong_element_kind_is_reported_at_its_index`).
    #[test]
    fn same_width_f32_and_i32_are_interchangeable() {
        let k = kernel(schema(&[ElementKind::F32, ElementKind::I32]));
        assert_eq!(
            check_kernel_args(&k, &[ElementKind::I32, ElementKind::F32]),
            Ok(())
        );
    }

    #[test]
    fn wrong_element_kind_is_reported_at_its_index() {
        let k = kernel(schema(&[ElementKind::F32, ElementKind::I32]));
        assert_eq!(
            check_kernel_args(&k, &[ElementKind::F32, ElementKind::Bf16]),
            Err(KernelArgError::ElementKind {
                index: 1,
                expected: ElementKind::I32,
                actual: ElementKind::Bf16,
            })
        );
    }
}
