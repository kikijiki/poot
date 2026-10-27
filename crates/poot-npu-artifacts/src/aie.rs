//! The host side of an AIE (XDNA2) kernel: the linked xclbin and instruction stream, the shape ABI that says
//! where every region value sits in the host arenas, and the validators that refuse a malformed artifact or
//! a mismatched binding before any loader or XRT sees a byte (POOT-664, POOT-1160).

use std::num::NonZeroU32;

use poot_target::{AieTargetId, ArgAccess, BufferStorage, ElementKind};

/// Why an AIE artifact or binding record is not a loadable, self-consistent kernel (POOT-664). Raised by
/// the artifact validator, before a loader or XRT sees any byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AieArtifactError {
    /// The xclbin does not start with the `xclbin2` file magic.
    XclbinMagic,
    /// The instruction stream is empty or not a whole number of 4-byte words.
    InstructionStream {
        /// The stream's length in bytes.
        bytes: usize,
    },
    /// The artifact names no entry kernel.
    NoEntry,
    /// The artifact carries no toolchain identity.
    NoToolchain,
    /// The shape ABI has no binding, or no written binding.
    NoBindings,
    /// A binding's storage is not a dense F32/I32 word buffer.
    Storage {
        /// The binding position.
        index: usize,
        /// The storage the binding declared.
        storage: BufferStorage,
    },
    /// A binding has a zero dimension.
    ZeroDimension {
        /// The binding position.
        index: usize,
    },
    /// A binding's element count or byte extent overflows.
    ExtentOverflow {
        /// The binding position.
        index: usize,
    },
    /// The compiler-proved access count exceeds the binding's declared elements.
    AccessOutOfRange {
        /// The binding position.
        index: usize,
        /// Elements the compiler proved the kernel touches.
        accessed: u64,
        /// Elements the binding declares.
        elements: u64,
    },
    /// The serialized artifact is truncated or malformed.
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

/// One tensor binding of a shape-specialized AIE region: which host buffer it names (by position), the
/// exact physical and logical storage it must have, its logical dimensions, whether the region writes
/// it, and how many leading elements the compiler proved the region touches. The shape is part of the
/// artifact: a different shape is a different artifact, never a runtime argument. Where the binding lives
/// in the artifact's host arenas is derived from the binding list by [`AieShapeAbi::new`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieBinding {
    storage: BufferStorage,
    access: ArgAccess,
    dims: Box<[u64]>,
    elements: u64,
    accessed_elements: u64,
    /// Index into [`AieShapeAbi::arenas`] of the arena this binding lives in.
    arena: usize,
    /// Offset of the binding's first element, in 32-bit words from the start of its arena.
    offset: u64,
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
            arena: 0,
            offset: 0,
        })
    }

    /// The exact physical and logical storage the bound buffer must have.
    pub const fn storage(&self) -> BufferStorage {
        self.storage
    }

    /// Whether the kernel writes the binding.
    pub const fn access(&self) -> ArgAccess {
        self.access
    }

    /// The logical dimensions the kernel was compiled for.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Logical element count: the length the core is given.
    pub const fn elements(&self) -> u64 {
        self.elements
    }

    /// Leading elements the compiler proved the kernel touches.
    pub const fn accessed_elements(&self) -> u64 {
        self.accessed_elements
    }

    /// Index into [`AieShapeAbi::arenas`] of the arena the binding lives in.
    pub const fn arena(&self) -> usize {
        self.arena
    }

    /// Offset of the binding's first element, in 32-bit words from the start of its arena.
    pub const fn offset(&self) -> u64 {
        self.offset
    }
}

/// One host buffer the firmware binds to a region: every binding of one access and element kind lives in
/// it, one after the other. The firmware's dispatch ABI binds a fixed handful of host buffers whatever the
/// region's tensor count, so a region addresses its tensors by offset inside these, never by buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AieArena {
    access: ArgAccess,
    element: ElementKind,
    words: u64,
}

impl AieArena {
    /// Whether the bindings in the arena are read or written by the region.
    pub const fn access(&self) -> ArgAccess {
        self.access
    }

    /// The element kind of every binding in the arena.
    pub const fn element(&self) -> ElementKind {
        self.element
    }

    /// The arena's length in 32-bit words: the sum of its bindings' elements.
    pub const fn words(&self) -> u64 {
        self.words
    }
}

/// The ordered, compiler-derived binding list of an AIE region (the shape and argument ABI), and the host
/// arenas they are laid out in. The layout is a function of the binding list, so two artifacts with the
/// same bindings agree on every offset and a loader cannot disagree with the linker about one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieShapeAbi {
    bindings: Box<[AieBinding]>,
    arenas: Box<[AieArena]>,
}

impl AieShapeAbi {
    /// The ABI of `bindings`; refuses an empty list or one with no written binding. The arenas are the
    /// distinct (access, element kind) pairs in the order reads then writes, F32 then I32; a binding is
    /// placed in its arena after the bindings before it.
    pub fn new(mut bindings: Vec<AieBinding>) -> Result<Self, AieArtifactError> {
        let writes = bindings
            .iter()
            .any(|binding| binding.access == ArgAccess::Write);
        if bindings.is_empty() || !writes || u32::try_from(bindings.len()).is_err() {
            return Err(AieArtifactError::NoBindings);
        }
        let mut arenas: Vec<AieArena> = Vec::new();
        for binding in &bindings {
            let (access, element) = (binding.access, binding.storage.element());
            if !arenas
                .iter()
                .any(|arena| arena.access == access && arena.element == element)
            {
                arenas.push(AieArena {
                    access,
                    element,
                    words: 0,
                });
            }
        }
        arenas.sort_by_key(|arena| {
            (
                arena.access == ArgAccess::Write,
                arena.element != ElementKind::F32,
            )
        });
        for (index, binding) in bindings.iter_mut().enumerate() {
            let arena_index = arenas
                .iter()
                .position(|arena| {
                    arena.access == binding.access && arena.element == binding.storage.element()
                })
                .expect("every binding's arena was created above");
            let arena = &mut arenas[arena_index];
            binding.arena = arena_index;
            binding.offset = arena.words;
            arena.words = arena
                .words
                .checked_add(binding.elements)
                .ok_or(AieArtifactError::ExtentOverflow { index })?;
        }
        Ok(Self {
            bindings: bindings.into(),
            arenas: arenas.into(),
        })
    }

    /// The bindings, in host-buffer order.
    pub fn bindings(&self) -> &[AieBinding] {
        &self.bindings
    }

    /// The host arenas the bindings are laid out in: argument `i` of the firmware's dispatch is arena `i`.
    pub fn arenas(&self) -> &[AieArena] {
        &self.arenas
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

    /// The xclbin image the loader hands to XRT.
    pub fn xclbin(&self) -> &[u8] {
        &self.xclbin
    }

    /// The transfer-descriptor instruction stream.
    pub fn instructions(&self) -> &[u8] {
        &self.instructions
    }

    /// The entry kernel's name.
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// The shape and argument ABI.
    pub fn abi(&self) -> &AieShapeAbi {
        &self.abi
    }

    /// The AIE target the artifact was built for.
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
}

/// One host buffer a dispatch binds to an AIE kernel: its full storage record, logical dimensions and
/// whether the runtime may let the kernel write it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(any(test, feature = "test-support"))]
pub struct AieBound<'a> {
    /// The buffer's storage record.
    pub storage: BufferStorage,
    /// The buffer's logical dimensions.
    pub dims: &'a [u64],
    /// Whether the runtime may let the kernel write it.
    pub writable: bool,
}

/// A dispatch's bound buffers did not meet an AIE kernel's shape ABI: the one error the AIE loader
/// returns before XRT sees a buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(any(test, feature = "test-support"))]
pub enum AieBindError {
    /// The dispatch bound a different number of buffers than the ABI declares.
    Count {
        /// Bindings the ABI declares.
        expected: usize,
        /// Buffers the dispatch bound.
        actual: usize,
    },
    /// A bound buffer's storage record differs from the ABI's.
    Storage {
        /// The binding position.
        index: usize,
        /// The storage the ABI declares.
        expected: BufferStorage,
        /// The storage of the bound buffer.
        actual: BufferStorage,
    },
    /// The kernel writes the binding but the bound buffer is not writable.
    ReadOnlyWritten {
        /// The binding position.
        index: usize,
    },
    /// The bound buffer's dimensions are not the shape the kernel was compiled for (equal element counts
    /// under different dimensions are different shapes).
    Dims {
        /// The binding position.
        index: usize,
        /// The dimensions the kernel was compiled for.
        expected: Box<[u64]>,
        /// The dimensions of the bound buffer.
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

/// One run of an AIE kernel: its instruction stream executes once over the dispatch's host buffer objects
/// (the `inputs` then the `output`, in the kernel's shape-ABI binding order) and ends; there is no grid.
/// It names the target the kernel was linked for and the stream's length in 32-bit instruction words (a
/// loaded artifact's stream is validated non-empty and word-aligned). A device implementation (POOT-641)
/// must read `instruction_words` and refuse an invocation that disagrees with the stream of the kernel it
/// loaded; the GPU devices read only `target`, to refuse the launch kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AieInvocation {
    target: AieTargetId,
    instruction_words: NonZeroU32,
}

impl AieInvocation {
    /// An invocation of a stream of `instruction_words` words linked for `target`.
    pub const fn new(target: AieTargetId, instruction_words: NonZeroU32) -> Self {
        Self {
            target,
            instruction_words,
        }
    }

    /// The target the kernel was linked for.
    pub const fn target(&self) -> AieTargetId {
        self.target
    }

    /// The instruction stream's length, in 32-bit words.
    pub const fn instruction_words(&self) -> NonZeroU32 {
        self.instruction_words
    }
}
