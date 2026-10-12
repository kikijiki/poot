//! Dependency-neutral ownership, layout and decoding of weight storage formats.
//!
//! Every weight storage format has one descriptor ([`format::WeightFormat::descriptor`]) and one
//! scalar decoder ([`decode`]) that interprets it; [`format`] holds the design. Packed-weight
//! payloads preserve checkpoint-order source bytes. The crate does not parse checkpoints, define
//! graph values or derive backend transport layouts; it decodes values only when a caller asks.
//!
//! `unsafe` is denied crate-wide except in `stored/mapping.rs`, the one file mapping behind
//! [`StoredBytes`] (ADR-0103 amendment of 2026-10-09).

#![deny(unsafe_code)]

mod blocks;
mod content;
pub mod decode;
mod encode;
pub mod format;
pub mod plan;
mod planar;
pub mod scalar;
mod stored;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
#[cfg(test)]
mod tests;
pub mod weights;

use std::fmt;
use std::sync::Arc;

use content::ContentCheck;
use decode::DecodeError;
use format::{Axis, GroupMap, Storage, WeightFormat};

pub use encode::{EncodeError, encode};
pub use format::OperandRole;
pub use stored::{Access, FileIdentity, MapError, SliceError, SourceChanged, StoredBytes};

/// A logical shape axis rejected by descriptor construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalAxis {
    Out,
    K,
}

/// One source tensor of a [`PackedWeight`]: one GGUF buffer holding every block field, or one
/// safetensors tensor per operand role. [`PackedWeight::sources`] lists a weight's roles in the
/// descriptor's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceRole {
    /// The one buffer of a block format ([`format::Storage::Blocks`]).
    Blocks,
    /// One safetensors tensor of a planar format ([`format::Storage::Planar`]).
    Planar(OperandRole),
}

/// Typed failures from [`PackedWeight`] or [`PackedPayload`] construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedWeightError {
    ZeroExtent {
        axis: LogicalAxis,
    },
    /// A block format's `K` is not a whole number of `block_values`-value blocks.
    PartialBlock {
        format: WeightFormat,
        k: usize,
        block_values: usize,
    },
    /// `format` has no `Scale` operand (F32, F16, BF16): a packed weight always carries a scale.
    DenseFormat {
        format: WeightFormat,
    },
    /// A source's shape or byte accounting overflowed `usize`.
    Layout(DecodeError),
    /// [`PackedPayload::try_new`] was not given every source [`PackedWeight::sources`] names.
    MissingSource {
        format: WeightFormat,
        role: SourceRole,
    },
    /// [`PackedPayload::try_new`] was given a source `format` does not name.
    UnexpectedSource {
        format: WeightFormat,
        role: SourceRole,
    },
    SourceLengthMismatch {
        format: WeightFormat,
        role: SourceRole,
        expected: usize,
        actual: usize,
    },
    /// A stored `Float`-encoded field decodes to NaN or an infinity (ADR-0101 decision 4):
    /// checked for every format, block or planar, at every stored element. GGUF's `d`/`dmin` and
    /// safetensors scales may legitimately be zero or negative; only finiteness is a fault.
    NonFiniteField {
        format: WeightFormat,
        role: SourceRole,
        operand: OperandRole,
        element: usize,
    },
    NonZeroUnusedHighNibble {
        row: usize,
        source_index: usize,
        byte: u8,
    },
    /// A stored `GroupIndex` value (GPTQ act-order `g_idx[k]`) names no group: negative, or `>=`
    /// the format's group count. Checked once at construction (dquant.md 4, "kernels decode the
    /// admitted domain only"), so no decoder - the CPU oracle or a device kernel - ever reads an
    /// out-of-range group index; `poot_quant::decode::FormatDescriptor::group_of` already refused
    /// this per element for the CPU oracle, but a device kernel has no such check
    /// (`emit_group_of` trusts the admitted domain), so the fault must be caught here instead.
    GroupIndexOutOfRange {
        format: WeightFormat,
        k: usize,
        value: i64,
        groups: usize,
    },
    /// [`PackedPayload::gather_rows`] over a planar format: its rows are not whole byte runs of one
    /// source (a GPTQ/AWQ code tensor is K-major), so a row gather would have to repack.
    RowGatherStorage {
        format: WeightFormat,
    },
    /// [`PackedPayload::concat_rows`] was given no parts: there is no weight to stack.
    RowConcatEmpty,
    /// [`PackedPayload::concat_rows`] over a planar operand whose stored rows do not follow `out`
    /// one block of rows at a time (a K-major GPTQ/AWQ tensor, or a whole-axis grid): stacking the
    /// parts would have to repack it.
    RowConcatLayout {
        format: WeightFormat,
        role: OperandRole,
    },
    /// [`PackedPayload::concat_rows`] part of `rows` rows that is not a whole number of `role`'s
    /// `block_rows`-row blocks: its last block would straddle the boundary to the next part
    /// (an E4M3 128x128 scale block shared by the tail of `gate` and the head of `up`).
    RowConcatUnaligned {
        format: WeightFormat,
        role: OperandRole,
        rows: usize,
        block_rows: usize,
    },
    /// [`PackedPayload::gather_rows`] or [`PackedPayload::concat_rows`] parts that do not share one
    /// format and `K`.
    RowGatherParts {
        first: PackedWeight,
        other: PackedWeight,
    },
    /// [`PackedPayload::gather_rows`] named a row past the parts' stacked rows.
    RowOutOfRange {
        row: usize,
        rows: usize,
    },
    /// [`PackedStack::try_new`] was given no experts: there is no table to build.
    StackEmpty,
    /// [`PackedStack::try_new`] experts that do not share one format and shape: a stacked table has
    /// one descriptor, so the experts' slabs are all alike.
    StackParts {
        first: PackedWeight,
        other: PackedWeight,
    },
}

impl From<DecodeError> for PackedWeightError {
    fn from(error: DecodeError) -> Self {
        Self::Layout(error)
    }
}

impl fmt::Display for PackedWeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroExtent { axis } => write!(f, "logical {axis:?} extent must be positive"),
            Self::PartialBlock {
                format,
                k,
                block_values,
            } => write!(
                f,
                "{format:?} K={k} is not a whole number of {block_values}-value blocks"
            ),
            Self::DenseFormat { format } => {
                write!(f, "{format:?} has no Scale operand and cannot be packed")
            }
            Self::Layout(error) => write!(f, "{error}"),
            Self::MissingSource { format, role } => {
                write!(f, "{format:?} needs a {role:?} source")
            }
            Self::UnexpectedSource { format, role } => {
                write!(f, "{format:?} does not have a {role:?} source")
            }
            Self::SourceLengthMismatch {
                format,
                role,
                expected,
                actual,
            } => write!(
                f,
                "{format:?} {role:?} source length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteField {
                format,
                role,
                operand,
                element,
            } => write!(
                f,
                "{format:?} {role:?} {operand:?} element {element} is not finite"
            ),
            Self::NonZeroUnusedHighNibble {
                row,
                source_index,
                byte,
            } => write!(
                f,
                "row {row} source byte {source_index} has nonzero unused high nibble: {byte:#04x}"
            ),
            Self::GroupIndexOutOfRange {
                format,
                k,
                value,
                groups,
            } => write!(
                f,
                "{format:?} g_idx[{k}] = {value} names no group (must be 0..{groups})"
            ),
            Self::RowGatherStorage { format } => write!(
                f,
                "{format:?} is planar: its rows are not byte runs of one source, so they cannot be gathered"
            ),
            Self::RowConcatEmpty => {
                write!(f, "a row concatenation needs at least one part, got none")
            }
            Self::RowConcatLayout { format, role } => write!(
                f,
                "{format:?} {role:?} rows are not stored out-major in blocks, so they cannot be concatenated"
            ),
            Self::RowConcatUnaligned {
                format,
                role,
                rows,
                block_rows,
            } => write!(
                f,
                "{format:?} {role:?}: a {rows}-row part is not a whole number of {block_rows}-row blocks, so a block would straddle the concatenation boundary"
            ),
            Self::RowGatherParts { first, other } => write!(
                f,
                "row-gather or row-concat parts must share one format and K: {first:?} vs {other:?}"
            ),
            Self::RowOutOfRange { row, rows } => {
                write!(f, "row {row} is past the parts' {rows} stacked rows")
            }
            Self::StackEmpty => write!(f, "an expert stack needs at least one expert, got none"),
            Self::StackParts { first, other } => write!(
                f,
                "expert stack parts must share one format and shape: {first:?} vs {other:?}"
            ),
        }
    }
}

impl std::error::Error for PackedWeightError {}

/// One logical `[out, K]` packed weight: which format, which shape. Every source fact - which
/// tensors it needs ([`Self::sources`]), their shapes and byte lengths ([`Self::source_shape`],
/// [`Self::source_bytes`]) - is derived from `format`'s [`format::FormatDescriptor`], never
/// restated here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedWeight {
    format: WeightFormat,
    shape: [usize; 2],
}

impl PackedWeight {
    /// Reject a zero extent, a `K` that is not a whole number of blocks for a block format, and a
    /// dense format (no `Scale` operand). Also proves every source's shape and byte accounting fits
    /// `usize`, so [`Self::source_shape`] and [`Self::source_bytes`] are infallible afterwards.
    pub fn try_new(format: WeightFormat, shape: [usize; 2]) -> Result<Self, PackedWeightError> {
        let [out, k] = shape;
        if out == 0 {
            return Err(PackedWeightError::ZeroExtent {
                axis: LogicalAxis::Out,
            });
        }
        if k == 0 {
            return Err(PackedWeightError::ZeroExtent {
                axis: LogicalAxis::K,
            });
        }
        let descriptor = format.descriptor();
        if let Storage::Blocks(layout) = descriptor.storage
            && !k.is_multiple_of(layout.values)
        {
            return Err(PackedWeightError::PartialBlock {
                format,
                k,
                block_values: layout.values,
            });
        }
        if !descriptor.has_scale() {
            return Err(PackedWeightError::DenseFormat { format });
        }

        let weight = Self { format, shape };
        for role in weight.sources() {
            weight.try_source_bytes(role)?;
        }
        Ok(weight)
    }

    pub const fn format(self) -> WeightFormat {
        self.format
    }

    pub const fn shape(self) -> [usize; 2] {
        self.shape
    }

    /// `out * K`.
    pub fn logical_values(self) -> usize {
        self.shape[0] * self.shape[1]
    }

    /// This weight's source tensors, in the descriptor's order.
    pub fn sources(self) -> Vec<SourceRole> {
        match self.format.descriptor().storage {
            Storage::Blocks(_) => vec![SourceRole::Blocks],
            Storage::Planar(layout) => layout
                .operands
                .iter()
                .map(|operand| SourceRole::Planar(operand.role))
                .collect(),
        }
    }

    fn try_source_shape(self, role: SourceRole) -> Result<[usize; 2], DecodeError> {
        let descriptor = self.format.descriptor();
        match role {
            SourceRole::Blocks => {
                let Storage::Blocks(layout) = descriptor.storage else {
                    return Err(DecodeError::WrongStorage {
                        format: self.format,
                    });
                };
                let row_blocks = self.shape[1] / layout.values;
                let row_bytes =
                    row_blocks
                        .checked_mul(layout.bytes)
                        .ok_or(DecodeError::Overflow {
                            format: self.format,
                        })?;
                Ok([self.shape[0], row_bytes])
            }
            SourceRole::Planar(operand_role) => descriptor.source_shape(operand_role, self.shape),
        }
    }

    fn try_source_bytes(self, role: SourceRole) -> Result<usize, DecodeError> {
        let [rows, row_bytes] = self.try_source_shape(role)?;
        rows.checked_mul(row_bytes).ok_or(DecodeError::Overflow {
            format: self.format,
        })
    }

    /// The `[rows, row_bytes]` grid of `role`'s source tensor: `rows` stored rows, each
    /// `row_bytes` bytes.
    ///
    /// Panics if `role` is not one of [`Self::sources`]: [`Self::try_new`] already proved every
    /// source of this weight fits `usize`, so the only way this can fail is a role this weight does
    /// not have.
    pub fn source_shape(self, role: SourceRole) -> [usize; 2] {
        self.try_source_shape(role).unwrap_or_else(|error| {
            panic!("{role:?} is not a source of {:?}: {error}", self.format)
        })
    }

    /// `source_shape(role)`'s two extents multiplied.
    pub fn source_bytes(self, role: SourceRole) -> usize {
        let [rows, row_bytes] = self.source_shape(role);
        rows * row_bytes
    }

    /// The bytes one expert of a stacked table holds of `role`: [`Self::source_bytes`] padded up to
    /// whole `u32` words, so every expert's slab starts on a word boundary and a kernel finds expert
    /// `e`'s source at word `e * slab_words(role)`.
    pub fn slab_bytes(self, role: SourceRole) -> usize {
        self.source_bytes(role).next_multiple_of(4)
    }

    /// [`Self::slab_bytes`] in `u32` words: the per-expert stride of a stacked table's `role` source.
    pub fn slab_words(self, role: SourceRole) -> usize {
        self.slab_bytes(role) / 4
    }

    /// The most experts one stacked table of this weight holds with every source addressable: the indexed kernels
    /// reach expert `e`'s slab at bit offset `e * slab_words(role) * 32` held in 32 bits, so a table of `E` experts
    /// needs `E * slab_words(role) * 32 <= u32::MAX` for every [`Self::sources`] role (512 MiB of one source). A
    /// larger stack is split into expert ranges, each its own table.
    pub fn max_stack_experts(self) -> usize {
        self.sources()
            .into_iter()
            .map(|role| u32::MAX as usize / (self.slab_words(role) * 32).max(1))
            .min()
            .unwrap_or(0)
    }

    /// `source_bytes` summed over every [`Self::sources`]: the total authoritative byte footprint
    /// of this weight, for byte accounting and limits.
    pub fn total_source_bytes(self) -> usize {
        self.sources()
            .into_iter()
            .map(|role| self.source_bytes(role))
            .sum()
    }
}

/// One payload source: its role and its bytes. Identity is role + bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PayloadSource {
    role: SourceRole,
    bytes: StoredBytes,
}

/// A no-copy runtime binding for one component of an immutable packed payload.
///
/// Cloning this value clones only the outer [`Arc<PackedPayload>`]. The source byte owners stay
/// private to the payload and are exposed only as borrowed slices.
#[derive(Clone, Debug)]
pub struct PackedComponentRef {
    owner: Arc<PackedPayload>,
    role: SourceRole,
}

impl PackedComponentRef {
    pub fn new(owner: Arc<PackedPayload>, role: SourceRole) -> Self {
        Self { owner, role }
    }

    pub const fn role(&self) -> SourceRole {
        self.role
    }

    pub fn owner(&self) -> &Arc<PackedPayload> {
        &self.owner
    }

    pub fn weight(&self) -> PackedWeight {
        self.owner.weight()
    }

    pub fn bytes(&self) -> &[u8] {
        self.owner.bytes(self.role)
    }

    pub fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }
}

/// The experts of one routed projection as one table: `experts` packed weights of one format and shape,
/// held as their own payloads and read together as a stacked `[experts, out, K]` weight. A stacked
/// table is never one payload: each expert keeps its own source bytes (ADR-0109), and the device
/// upload lays them back to back at [`PackedWeight::slab_bytes`] strides.
///
/// Key convention: the GGUF loader stores a rank-3 expert tensor `key` as one entry per expert named
/// `{key}.{expert}`, and a stack's [`weights::WeightView::ExpertStack`] lists those keys in expert order. Test
/// fixtures that name a stack `S` hold its experts as `S.0`, `S.1`, .. (`poot-graph-ir`'s
/// `test_support::const_weights::ConstWeights` and `weight_map_oracle::expert_owners` read that spelling).
#[derive(Clone, Debug)]
pub struct PackedStack {
    weight: PackedWeight,
    owners: Vec<Arc<PackedPayload>>,
}

impl PackedStack {
    /// Stack `owners` in order: refused when empty or when the experts differ in format or shape.
    pub fn try_new(owners: Vec<Arc<PackedPayload>>) -> Result<Self, PackedWeightError> {
        let first = owners.first().ok_or(PackedWeightError::StackEmpty)?.weight;
        if let Some(other) = owners
            .iter()
            .map(|owner| owner.weight)
            .find(|&w| w != first)
        {
            return Err(PackedWeightError::StackParts { first, other });
        }
        Ok(Self {
            weight: first,
            owners,
        })
    }

    /// The one descriptor every expert shares.
    pub const fn weight(&self) -> PackedWeight {
        self.weight
    }

    pub fn experts(&self) -> usize {
        self.owners.len()
    }

    /// Expert `expert`'s own payload. Panics when `expert >= experts()`.
    pub fn owner(&self, expert: usize) -> &Arc<PackedPayload> {
        &self.owners[expert]
    }

    /// Decode logical row `row` of expert `expert` into `out` ([`PackedPayload::decode_row`]).
    pub fn decode_row(
        &self,
        expert: usize,
        row: usize,
        out: &mut [f32],
    ) -> Result<(), DecodeError> {
        self.owners[expert].decode_row(row, out)
    }
}

impl PartialEq for PackedStack {
    fn eq(&self, other: &Self) -> bool {
        self.owners.len() == other.owners.len()
            && self
                .owners
                .iter()
                .zip(&other.owners)
                .all(|(a, b)| Arc::ptr_eq(a, b) || a == b)
    }
}

impl Eq for PackedStack {}

/// A no-copy runtime binding for one source role of a [`PackedStack`] (the stacked counterpart of
/// [`PackedComponentRef`]): cloning clones only the outer [`Arc`].
#[derive(Clone, Debug)]
pub struct PackedStackComponentRef {
    stack: Arc<PackedStack>,
    role: SourceRole,
}

impl PackedStackComponentRef {
    pub fn new(stack: Arc<PackedStack>, role: SourceRole) -> Self {
        Self { stack, role }
    }

    pub const fn role(&self) -> SourceRole {
        self.role
    }

    pub fn stack(&self) -> &Arc<PackedStack> {
        &self.stack
    }

    pub fn weight(&self) -> PackedWeight {
        self.stack.weight()
    }

    pub fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.stack, &other.stack)
    }

    /// The stacked table's `role` bytes as the device holds them: every expert's source padded to
    /// [`PackedWeight::slab_bytes`], back to back (the one definition of that layout; the executor
    /// binder pads a mapped stack the same way from its stored runs).
    pub fn slab_bytes(&self) -> Vec<u8> {
        let weight = self.stack.weight();
        let slab = weight.slab_bytes(self.role);
        let mut bytes = Vec::with_capacity(slab * self.stack.experts());
        for expert in 0..self.stack.experts() {
            let source = self.stack.owner(expert).bytes(self.role);
            bytes.extend_from_slice(source);
            bytes.resize((expert + 1) * slab, 0);
        }
        bytes
    }
}

impl PartialEq for PackedStackComponentRef {
    fn eq(&self, other: &Self) -> bool {
        self.role == other.role && self.same_owner(other)
    }
}

impl Eq for PackedStackComponentRef {}

/// The device transport of one packed source's bytes (dquant.md D9): its native bytes, unchanged,
/// as zero-filled little-endian `u32` words plus one zero guard word, so a generated kernel's
/// two-word bit read (`poot_kernelgen`'s `emit_read_bits`) never reads past the buffer. The one
/// definition every executor's const bind uploads (card 642); no backend re-lays out or re-encodes
/// the bytes (ADR-0101 decision 5). It takes the bytes, not a [`PackedComponentRef`]: a mapped
/// weight's source is the stored runs its view reads (whole block rows of a fused entry, or several
/// entries back to back, Card 564), which need not be one payload's whole source. A second copy of
/// this function anywhere else is the bug Card 546a review found and fixed.
#[cfg(any(test, feature = "test-support"))]
pub fn packed_source_words(bytes: &[u8]) -> Vec<u32> {
    let mut words = Vec::with_capacity(bytes.len().div_ceil(4) + 1);
    words.extend(bytes.chunks(4).map(|chunk| {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        u32::from_le_bytes(word)
    }));
    words.push(0);
    words
}

impl PartialEq for PackedComponentRef {
    fn eq(&self, other: &Self) -> bool {
        self.role == other.role && self.same_owner(other)
    }
}

impl Eq for PackedComponentRef {}

/// Authoritative, immutable source storage for one packed weight.
#[derive(Debug, Clone)]
pub struct PackedPayload {
    weight: PackedWeight,
    /// One entry per [`PackedWeight::sources`] role, in that order.
    sources: Vec<PayloadSource>,
}

impl PartialEq for PackedPayload {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight && self.sources == other.sources
    }
}

impl Eq for PackedPayload {}

impl PackedPayload {
    /// Bind `weight` to its source bytes, one per [`PackedWeight::sources`] role (any order,
    /// exactly those roles, each of the byte length [`PackedWeight::source_bytes`] states).
    pub fn try_new(
        weight: PackedWeight,
        sources: impl IntoIterator<Item = (SourceRole, StoredBytes)>,
    ) -> Result<Self, PackedWeightError> {
        let mut provided: Vec<(SourceRole, StoredBytes)> = sources.into_iter().collect();
        let mut ordered = Vec::with_capacity(provided.len());
        for role in weight.sources() {
            let position = provided
                .iter()
                .position(|(source_role, _)| *source_role == role)
                .ok_or(PackedWeightError::MissingSource {
                    format: weight.format,
                    role,
                })?;
            let (_, bytes) = provided.remove(position);
            let expected = weight.source_bytes(role);
            if bytes.len() != expected {
                return Err(PackedWeightError::SourceLengthMismatch {
                    format: weight.format,
                    role,
                    expected,
                    actual: bytes.len(),
                });
            }
            ordered.push(PayloadSource { role, bytes });
        }
        if let Some((role, _)) = provided.into_iter().next() {
            return Err(PackedWeightError::UnexpectedSource {
                format: weight.format,
                role,
            });
        }

        validate_content(weight, &ordered)?;

        Ok(Self {
            weight,
            sources: ordered,
        })
    }

    pub const fn weight(&self) -> PackedWeight {
        self.weight
    }

    /// `role`'s source bytes. Panics if `role` is not one of this payload's sources (see
    /// [`PackedWeight::sources`]).
    pub fn bytes(&self, role: SourceRole) -> &[u8] {
        self.source(role).bytes.as_slice()
    }

    /// Every source's owner, in [`PackedWeight::sources`] order.
    pub(crate) fn stored(&self) -> impl Iterator<Item = &StoredBytes> {
        self.sources.iter().map(|source| &source.bytes)
    }

    fn source(&self, role: SourceRole) -> &PayloadSource {
        self.sources
            .iter()
            .find(|source| source.role == role)
            .unwrap_or_else(|| panic!("{role:?} is not a source of {:?}", self.weight.format()))
    }

    /// A new block-format payload whose logical rows are `rows` of `parts` stacked in order (row
    /// `r` of the stack is row `r - offset` of the part holding it): the loader's row slicing, row
    /// permutation and row concatenation (fused q|k|v slices, llama's q/k un-permute, gate||up), done
    /// on the stored blocks byte for byte, never through a decode. A row of a block format is a whole
    /// run of `K / block_values` blocks, so the gathered bytes are exactly the stored ones.
    pub fn gather_rows(parts: &[&Self], rows: &[usize]) -> Result<Self, PackedWeightError> {
        let Some(first) = parts.first().map(|part| part.weight) else {
            return Err(PackedWeightError::RowOutOfRange {
                row: rows.first().copied().unwrap_or(0),
                rows: 0,
            });
        };
        let format = first.format();
        if !matches!(format.descriptor().storage, Storage::Blocks(_)) {
            return Err(PackedWeightError::RowGatherStorage { format });
        }
        if let Some(other) = parts
            .iter()
            .map(|part| part.weight)
            .find(|weight| weight.format() != format || weight.shape()[1] != first.shape()[1])
        {
            return Err(PackedWeightError::RowGatherParts { first, other });
        }
        let weight = PackedWeight::try_new(format, [rows.len(), first.shape()[1]])?;
        let [_, row_bytes] = first.source_shape(SourceRole::Blocks);
        let stacked: usize = parts.iter().map(|part| part.weight.shape()[0]).sum();
        let locate = |row: usize| {
            let mut local = row;
            parts
                .iter()
                .find(|part| {
                    let held = part.weight.shape()[0];
                    if local < held {
                        return true;
                    }
                    local -= held;
                    false
                })
                .map(|part| (*part, local))
                .ok_or(PackedWeightError::RowOutOfRange { row, rows: stacked })
        };
        // One part and an ascending contiguous run of rows is one contiguous byte run of the
        // part's source: a view of it, not a copy.
        let contiguous = rows.windows(2).all(|pair| pair[1] == pair[0] + 1);
        if let ([only], true, Some(&first_row)) = (parts, contiguous, rows.first())
            && first_row + rows.len() <= only.weight.shape()[0]
        {
            let start = first_row * row_bytes;
            let view = only
                .source(SourceRole::Blocks)
                .bytes
                .slice(start..start + rows.len() * row_bytes)
                .expect("the run lies within the part's rows");
            return Self::try_new(weight, [(SourceRole::Blocks, view)]);
        }
        let mut bytes = Vec::with_capacity(rows.len() * row_bytes);
        for &row in rows {
            let (part, local) = locate(row)?;
            let start = local * row_bytes;
            bytes.extend_from_slice(&part.bytes(SourceRole::Blocks)[start..start + row_bytes]);
        }
        Self::try_new(weight, [(SourceRole::Blocks, StoredBytes::from(bytes))])
    }

    /// Decode logical row `row` (all `K` values) into `out` through the one scalar decoder,
    /// without materializing a dense weight: one [`decode::FormatDescriptor::decode_blocks`] over
    /// the row's blocks for a block format, [`decode::FormatDescriptor::decode_planar_row`] (the
    /// per-value planar decode with its source checks done once per row) for a planar one. `O(K)`
    /// per row, so a whole weight decodes in `O(out * K)`. This is the CPU oracle's only packed
    /// read (`poot-eval`'s `decode_packed_row`).
    pub fn decode_row(&self, row: usize, out: &mut [f32]) -> Result<(), DecodeError> {
        let shape = self.weight.shape;
        if row >= shape[0] {
            return Err(DecodeError::Coordinate {
                format: self.weight.format,
                coordinate: [row, 0],
                shape,
            });
        }
        if out.len() != shape[1] {
            return Err(DecodeError::RowOutput {
                format: self.weight.format,
                k: shape[1],
                len: out.len(),
            });
        }
        let descriptor = self.weight.format.descriptor();
        match descriptor.storage {
            Storage::Blocks(_) => {
                let [_, row_bytes] = self.weight.source_shape(SourceRole::Blocks);
                let start = row * row_bytes;
                let bytes =
                    &self.source(SourceRole::Blocks).bytes.as_slice()[start..start + row_bytes];
                descriptor.decode_blocks(bytes, out)
            }
            Storage::Planar(_) => {
                let sources: Vec<(OperandRole, &[u8])> = self
                    .sources
                    .iter()
                    .map(|source| {
                        let SourceRole::Planar(role) = source.role else {
                            unreachable!("a planar payload holds only Planar sources")
                        };
                        (role, source.bytes.as_slice())
                    })
                    .collect();
                descriptor.decode_planar_row(shape, &sources, row, out)?;
                Ok(())
            }
        }
    }
}

/// Content checks for every packed format (ADR-0103 decision 4): every stored value of a
/// `Float`-encoded field, block or planar, must decode finite, and (E2M1 only) an odd `K`'s packed
/// padding nibble must be zero. Positivity is not checked: GGUF `d`/`dmin` and safetensors scales may
/// legitimately be zero or negative. A scheme's own value-formula and range checks are
/// Card 541's oracle, not this content check.
fn validate_content(
    weight: PackedWeight,
    sources: &[PayloadSource],
) -> Result<(), PackedWeightError> {
    let format = weight.format();
    let descriptor = format.descriptor();
    let source_bytes = |role: SourceRole| -> &[u8] {
        sources
            .iter()
            .find(|source| source.role == role)
            .expect("PackedPayload::try_new already proved every source role is present")
            .bytes
            .as_slice()
    };
    let check = ContentCheck::of(format)?;
    match (&*check, descriptor.storage) {
        (ContentCheck::Blocks(check), _) => {
            let bytes = source_bytes(SourceRole::Blocks);
            if let Some((operand, element)) = check.first_non_finite(bytes, &mut ()) {
                return Err(PackedWeightError::NonFiniteField {
                    format,
                    role: SourceRole::Blocks,
                    operand,
                    element,
                });
            }
        }
        (ContentCheck::Planar(checks), Storage::Planar(layout)) => {
            for (operand, check) in layout.operands.iter().zip(checks) {
                let role = SourceRole::Planar(operand.role);
                let bytes = source_bytes(role);
                if let Some(element) =
                    check.and_then(|check| check.first_non_finite(bytes, &mut ()))
                {
                    return Err(PackedWeightError::NonFiniteField {
                        format,
                        role,
                        operand: operand.role,
                        element,
                    });
                }
                if operand.role == OperandRole::GroupIndex
                    && let WeightFormat::Gptq {
                        groups: GroupMap::Indexed { groups },
                    } = format
                    && let Some((k, value)) = descriptor.first_invalid_group_index(
                        operand,
                        weight.shape(),
                        bytes,
                        groups.get(),
                    )
                {
                    return Err(PackedWeightError::GroupIndexOutOfRange {
                        format,
                        k,
                        value,
                        groups: groups.get(),
                    });
                }
            }
        }
        _ => unreachable!("a format's compiled content check follows its storage"),
    }
    validate_adjacent_k_padding(weight, sources)
}

/// When a planar format's `Codes` operand packs two K values per byte (E2M1) and `K` is odd, the last
/// byte of every row has one unused (padding) nibble that must be zero. Keyed on the descriptor's
/// own sub-byte packing operand: a fact about the storage, not a registered-cell special case.
fn validate_adjacent_k_padding(
    weight: PackedWeight,
    sources: &[PayloadSource],
) -> Result<(), PackedWeightError> {
    let descriptor = weight.format().descriptor();
    let Storage::Planar(layout) = descriptor.storage else {
        return Ok(());
    };
    let Some(codes) = layout
        .operands
        .iter()
        .find(|operand| operand.role == OperandRole::Codes)
    else {
        return Ok(());
    };
    let Some(packing) = codes.packing else {
        return Ok(());
    };
    let k = weight.shape()[1];
    if packing.values_per_word != 2 || packing.axis != Axis::K || k.is_multiple_of(2) {
        return Ok(());
    }

    let role = SourceRole::Planar(OperandRole::Codes);
    let weight_bytes = sources
        .iter()
        .find(|source| source.role == role)
        .expect("PackedPayload::try_new already proved every source role is present")
        .bytes
        .as_slice();
    let [_, row_bytes] = weight.source_shape(role);
    for row in 0..weight.shape()[0] {
        let source_index = (row + 1) * row_bytes - 1;
        let byte = weight_bytes[source_index];
        if byte & 0xf0 != 0 {
            return Err(PackedWeightError::NonZeroUnusedHighNibble {
                row,
                source_index,
                byte,
            });
        }
    }
    Ok(())
}
