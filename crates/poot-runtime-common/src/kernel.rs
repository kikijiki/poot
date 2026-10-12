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
//!         poot_target::TargetId::Gpu(poot_target::Backend::SpirvVulkan),
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

use poot_target::{AieTargetId, BufferStorage, ElementKind, TargetId};

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
    /// A linked AIE xclbin plus its instruction stream, validated against its shape ABI at construction.
    Aie(AieLinkedArtifact),
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        KernelCode::SpirvWords(words.into_boxed_slice())
    }
}

/// A kernel a runtime may dispatch: the target it was compiled for, its entry point, its argument
/// schema, and whether its body can trap (an `Assert`/`Unreachable` reserves one more error-word binding
/// the runtime zeroes before dispatch and checks after). See the module doc for the soundness contract.
#[derive(Clone, Debug)]
pub struct CompiledKernel {
    target: TargetId,
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
        target: TargetId,
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

    pub fn target(&self) -> TargetId {
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

/// Why an AIE artifact or binding record is not a loadable, self-consistent kernel (POOT-664). Raised by
/// the artifact validator, before a loader or XRT sees any byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AieArtifactError {
    XclbinMagic,
    InstructionStream {
        bytes: usize,
    },
    NoEntry,
    NoToolchain,
    NoBindings,
    Storage {
        index: usize,
        storage: BufferStorage,
    },
    ZeroDimension {
        index: usize,
    },
    ExtentOverflow {
        index: usize,
    },
    AccessOutOfRange {
        index: usize,
        accessed: u64,
        elements: u64,
    },
    Malformed(&'static str),
}

impl std::fmt::Display for AieArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AieArtifactError::XclbinMagic => {
                write!(f, "the xclbin does not start with the `xclbin2` file magic")
            }
            AieArtifactError::InstructionStream { bytes } => write!(
                f,
                "the instruction stream is empty or not a whole number of 4-byte words ({bytes} bytes)"
            ),
            AieArtifactError::NoEntry => write!(f, "the artifact names no entry kernel"),
            AieArtifactError::NoToolchain => {
                write!(f, "the artifact carries no toolchain identity")
            }
            AieArtifactError::NoBindings => write!(
                f,
                "a shape ABI needs at least one binding and at least one written binding"
            ),
            AieArtifactError::Storage { index, storage } => write!(
                f,
                "binding {index}: storage {storage:?} is not a dense F32/I32 word buffer"
            ),
            AieArtifactError::ZeroDimension { index } => {
                write!(f, "binding {index}: a dimension is zero")
            }
            AieArtifactError::ExtentOverflow { index } => write!(
                f,
                "binding {index}: the shape's element count or byte extent overflows"
            ),
            AieArtifactError::AccessOutOfRange {
                index,
                accessed,
                elements,
            } => write!(
                f,
                "binding {index}: the compiler-proved access of {accessed} elements exceeds the {elements} declared"
            ),
            AieArtifactError::Malformed(what) => {
                write!(
                    f,
                    "the serialized artifact is truncated or malformed: {what}"
                )
            }
        }
    }
}

impl std::error::Error for AieArtifactError {}

/// One tensor binding of a shape-specialized AIE kernel: which host buffer it names (by position), the
/// exact physical and logical storage it must have, its logical dimensions, whether the kernel writes
/// it, and how many leading elements the compiler proved the kernel touches. The shape is part of the
/// kernel: a different shape is a different artifact, never a runtime argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieBinding {
    storage: BufferStorage,
    access: ArgAccess,
    dims: Box<[u64]>,
    elements: u64,
    accessed_elements: u64,
}

impl AieBinding {
    /// A dense F32 or I32 binding of `dims` (rank 0 is one scalar element). `accessed_elements` is the
    /// generator's proved access bound; it may not exceed the shape.
    pub fn new(
        index: usize,
        storage: BufferStorage,
        access: ArgAccess,
        dims: &[u64],
        accessed_elements: u64,
    ) -> Result<Self, AieArtifactError> {
        // Exactly the canonical F32 or I32 word buffer: the serialized form records the element kind
        // alone, so a binding with any other logical dtype could not round-trip as the ABI it was built as.
        if storage != BufferStorage::f32() && storage != BufferStorage::i32() {
            return Err(AieArtifactError::Storage { index, storage });
        }
        if dims.contains(&0) {
            return Err(AieArtifactError::ZeroDimension { index });
        }
        let elements = dims
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d))
            .filter(|n| {
                n.checked_mul(storage.element().byte_width() as u64)
                    .is_some()
            })
            .ok_or(AieArtifactError::ExtentOverflow { index })?;
        if accessed_elements > elements {
            return Err(AieArtifactError::AccessOutOfRange {
                index,
                accessed: accessed_elements,
                elements,
            });
        }
        Ok(Self {
            storage,
            access,
            dims: dims.into(),
            elements,
            accessed_elements,
        })
    }

    pub const fn storage(&self) -> BufferStorage {
        self.storage
    }

    pub const fn access(&self) -> ArgAccess {
        self.access
    }

    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Logical element count: the length the core is given.
    pub const fn elements(&self) -> u64 {
        self.elements
    }

    pub const fn accessed_elements(&self) -> u64 {
        self.accessed_elements
    }
}

/// The ordered, compiler-derived binding list of an AIE kernel (the shape and argument ABI).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieShapeAbi {
    bindings: Box<[AieBinding]>,
}

impl AieShapeAbi {
    pub fn new(bindings: Vec<AieBinding>) -> Result<Self, AieArtifactError> {
        let writes = bindings
            .iter()
            .any(|binding| binding.access == ArgAccess::Write);
        if bindings.is_empty() || !writes || u32::try_from(bindings.len()).is_err() {
            return Err(AieArtifactError::NoBindings);
        }
        Ok(Self {
            bindings: bindings.into(),
        })
    }

    pub fn bindings(&self) -> &[AieBinding] {
        &self.bindings
    }

    /// The core's ABI is interleaved `(ptr, u32 len)` pairs; these are the lengths, in binding order.
    #[cfg(any(test, feature = "test-support"))]
    pub fn core_lengths(&self) -> Vec<u64> {
        self.bindings.iter().map(AieBinding::elements).collect()
    }
}

/// The loadable form of an AIE kernel: a linked xclbin and its instruction stream with the shape ABI,
/// target, toolchain identity and resources they were built for. It can only be built through
/// [`AieLinkedArtifact::new`] (the validator) or [`AieLinkedArtifact::from_bytes`], so an instance is
/// always self-consistent; whether it is the artifact of a given body is `poot-codegen`'s handle check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieLinkedArtifact {
    xclbin: Box<[u8]>,
    instructions: Box<[u8]>,
    entry: Box<str>,
    abi: AieShapeAbi,
    target: AieTargetId,
    toolchain: Box<str>,
    tile_local_bytes: u64,
}

const XCLBIN_MAGIC: &[u8; 8] = b"xclbin2\0";
const AIE_BUNDLE_MAGIC: &[u8; 8] = b"POOTAIE1";

impl AieLinkedArtifact {
    /// Validate and wrap a linked artifact.
    pub fn new(
        xclbin: Vec<u8>,
        instructions: Vec<u8>,
        entry: &str,
        abi: AieShapeAbi,
        target: AieTargetId,
        toolchain: &str,
        tile_local_bytes: u64,
    ) -> Result<Self, AieArtifactError> {
        if !xclbin.starts_with(XCLBIN_MAGIC) {
            return Err(AieArtifactError::XclbinMagic);
        }
        if instructions.is_empty() || !instructions.len().is_multiple_of(4) {
            return Err(AieArtifactError::InstructionStream {
                bytes: instructions.len(),
            });
        }
        if entry.is_empty() {
            return Err(AieArtifactError::NoEntry);
        }
        if toolchain.is_empty() {
            return Err(AieArtifactError::NoToolchain);
        }
        Ok(Self {
            xclbin: xclbin.into(),
            instructions: instructions.into(),
            entry: entry.into(),
            abi,
            target,
            toolchain: toolchain.into(),
            tile_local_bytes,
        })
    }

    pub fn xclbin(&self) -> &[u8] {
        &self.xclbin
    }

    pub fn instructions(&self) -> &[u8] {
        &self.instructions
    }

    pub fn entry(&self) -> &str {
        &self.entry
    }

    pub fn abi(&self) -> &AieShapeAbi {
        &self.abi
    }

    pub const fn target(&self) -> AieTargetId {
        self.target
    }

    /// The canonical identity of the Peano and IRON builds that produced this artifact.
    pub fn toolchain(&self) -> &str {
        &self.toolchain
    }

    /// Tile-local bytes the link recipe reserves (FIFO buffers at their depth plus scratch).
    pub const fn tile_local_bytes(&self) -> u64 {
        self.tile_local_bytes
    }

    /// Serialize as one self-describing blob (what the kernel cache stores). Every field is
    /// length-prefixed, so [`AieLinkedArtifact::from_bytes`] is the exact inverse.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.xclbin.len() + self.instructions.len() + 128);
        out.extend_from_slice(AIE_BUNDLE_MAGIC);
        put_bytes(&mut out, &self.xclbin);
        put_bytes(&mut out, &self.instructions);
        put_bytes(&mut out, self.entry.as_bytes());
        put_bytes(&mut out, self.toolchain.as_bytes());
        put_u64(&mut out, u64::from(self.target.abi_revision()));
        put_bytes(&mut out, self.target.canonical().as_bytes());
        put_u64(&mut out, self.tile_local_bytes);
        put_u64(&mut out, self.abi.bindings.len() as u64);
        for binding in &self.abi.bindings {
            out.push(match binding.storage.element() {
                ElementKind::F32 => 0,
                _ => 1,
            });
            out.push(u8::from(binding.access == ArgAccess::Write));
            put_u64(&mut out, binding.accessed_elements);
            put_u64(&mut out, binding.dims.len() as u64);
            for &dim in &binding.dims {
                put_u64(&mut out, dim);
            }
        }
        out
    }

    /// Parse and re-validate a blob [`AieLinkedArtifact::to_bytes`] wrote for `target`. A blob for a
    /// different target (ISA, platform or ABI revision) is malformed here, not loaded.
    pub fn from_bytes(bytes: &[u8], target: AieTargetId) -> Result<Self, AieArtifactError> {
        let mut r = Reader(bytes);
        if r.take(AIE_BUNDLE_MAGIC.len())? != AIE_BUNDLE_MAGIC {
            return Err(AieArtifactError::Malformed("bundle magic"));
        }
        let xclbin = r.bytes()?.to_vec();
        let instructions = r.bytes()?.to_vec();
        let entry = std::str::from_utf8(r.bytes()?)
            .map_err(|_| AieArtifactError::Malformed("entry is not UTF-8"))?
            .to_owned();
        let toolchain = std::str::from_utf8(r.bytes()?)
            .map_err(|_| AieArtifactError::Malformed("toolchain is not UTF-8"))?
            .to_owned();
        let revision = r.u64()?;
        let recorded = r.bytes()?;
        if revision != u64::from(target.abi_revision()) || recorded != target.canonical().as_bytes()
        {
            return Err(AieArtifactError::Malformed("built for a different target"));
        }
        let tile_local_bytes = r.u64()?;
        let count = r.u64()?;
        let mut bindings = Vec::new();
        for index in
            0..usize::try_from(count).map_err(|_| AieArtifactError::Malformed("binding count"))?
        {
            let &[element, write] = r.take(2)? else {
                return Err(AieArtifactError::Malformed("binding header"));
            };
            let storage = match element {
                0 => BufferStorage::f32(),
                1 => BufferStorage::i32(),
                _ => return Err(AieArtifactError::Malformed("binding element kind")),
            };
            let accessed = r.u64()?;
            let rank = r.u64()?;
            let dims = (0..rank).map(|_| r.u64()).collect::<Result<Vec<_>, _>>()?;
            let access = match write {
                0 => ArgAccess::Read,
                1 => ArgAccess::Write,
                _ => return Err(AieArtifactError::Malformed("binding access")),
            };
            bindings.push(AieBinding::new(index, storage, access, &dims, accessed)?);
        }
        if !r.0.is_empty() {
            return Err(AieArtifactError::Malformed("trailing bytes"));
        }
        Self::new(
            xclbin,
            instructions,
            &entry,
            AieShapeAbi::new(bindings)?,
            target,
            &toolchain,
            tile_local_bytes,
        )
    }
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_u64(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], AieArtifactError> {
        if self.0.len() < n {
            return Err(AieArtifactError::Malformed("truncated"));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn u64(&mut self) -> Result<u64, AieArtifactError> {
        let raw = self.take(8)?;
        Ok(u64::from_le_bytes(raw.try_into().expect("8 bytes")))
    }

    fn bytes(&mut self) -> Result<&'a [u8], AieArtifactError> {
        let len = usize::try_from(self.u64()?)
            .map_err(|_| AieArtifactError::Malformed("length does not fit"))?;
        self.take(len)
    }
}

/// One host buffer a dispatch binds to an AIE kernel: its full storage record, logical dimensions and
/// whether the runtime may let the kernel write it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(any(test, feature = "test-support"))]
pub struct AieBound<'a> {
    pub storage: BufferStorage,
    pub dims: &'a [u64],
    pub writable: bool,
}

/// A dispatch's bound buffers did not meet an AIE kernel's shape ABI: the one error the AIE loader
/// returns before XRT sees a buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(any(test, feature = "test-support"))]
pub enum AieBindError {
    Count {
        expected: usize,
        actual: usize,
    },
    Storage {
        index: usize,
        expected: BufferStorage,
        actual: BufferStorage,
    },
    ReadOnlyWritten {
        index: usize,
    },
    /// The bound buffer's dimensions are not the shape the kernel was compiled for (equal element counts
    /// under different dimensions are different shapes).
    Dims {
        index: usize,
        expected: Box<[u64]>,
        actual: Box<[u64]>,
    },
}

#[cfg(any(test, feature = "test-support"))]
impl std::fmt::Display for AieBindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AieBindError::Count { expected, actual } => write!(
                f,
                "shape ABI declares {expected} bindings, dispatch bound {actual}"
            ),
            AieBindError::Storage {
                index,
                expected,
                actual,
            } => write!(
                f,
                "binding {index}: ABI storage {expected:?}, bound buffer {actual:?}"
            ),
            AieBindError::ReadOnlyWritten { index } => write!(
                f,
                "binding {index}: the kernel writes it but the buffer is not writable"
            ),
            AieBindError::Dims {
                index,
                expected,
                actual,
            } => write!(
                f,
                "binding {index}: ABI shape is {expected:?}, bound buffer is {actual:?}"
            ),
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
impl std::error::Error for AieBindError {}

/// Check `bound` (binding order) against `artifact`'s shape ABI: count, exact storage (an F32 buffer is
/// not an I32 one although both are 4-byte words), writability of every written binding, and the exact
/// dimensions the kernel was specialized for.
#[cfg(any(test, feature = "test-support"))]
pub fn check_aie_bindings(
    artifact: &AieLinkedArtifact,
    bound: &[AieBound],
) -> Result<(), AieBindError> {
    let bindings = artifact.abi().bindings();
    if bound.len() != bindings.len() {
        return Err(AieBindError::Count {
            expected: bindings.len(),
            actual: bound.len(),
        });
    }
    for (index, (binding, buffer)) in bindings.iter().zip(bound).enumerate() {
        if binding.storage() != buffer.storage {
            return Err(AieBindError::Storage {
                index,
                expected: binding.storage(),
                actual: buffer.storage,
            });
        }
        if binding.access() == ArgAccess::Write && !buffer.writable {
            return Err(AieBindError::ReadOnlyWritten { index });
        }
        if binding.dims() != buffer.dims {
            return Err(AieBindError::Dims {
                index,
                expected: binding.dims().into(),
                actual: buffer.dims.into(),
            });
        }
    }
    Ok(())
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
                TargetId::Gpu(poot_target::Backend::SpirvVulkan),
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
