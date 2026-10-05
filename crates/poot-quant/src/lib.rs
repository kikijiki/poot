//! Dependency-neutral ownership, layout and decoding of weight storage formats.
//!
//! Every weight storage format has one descriptor ([`format::WeightFormat::descriptor`]) and one
//! scalar decoder ([`decode`]) that interprets it; [`format`] holds the design. Packed-weight
//! payloads preserve checkpoint-order source bytes. The crate does not parse checkpoints, define
//! graph values or derive backend transport layouts; it decodes values only when a caller asks.

mod blocks;
pub mod decode;
pub mod format;
pub mod plan;
mod planar;
pub mod scalar;
#[cfg(test)]
mod tests;
pub mod weights;

use std::fmt;
use std::sync::Arc;
use std::sync::OnceLock;

use decode::DecodeError;
use format::{Axis, GroupMap, Storage, WeightFormat};

pub use format::OperandRole;

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

    /// `source_bytes` summed over every [`Self::sources`]: the total authoritative byte footprint
    /// of this weight, for byte accounting and limits.
    pub fn total_source_bytes(self) -> usize {
        self.sources()
            .into_iter()
            .map(|role| self.source_bytes(role))
            .sum()
    }
}

/// Content fingerprint of one packed source buffer: its byte length plus a 64-bit FNV-1a of every
/// byte. It is a pure function of the bytes, and packed payload bytes are immutable, so a payload
/// computes it once and hands the same value out forever after.
///
/// The fold has a serial dependency chain (one xor + one multiply per byte, never vectorizable),
/// which costs roughly a billion bytes per second - at a 27 GB checkpoint that is tens of seconds.
/// [`PackedPayload::source_fingerprint`] is therefore the only supported way to obtain one: a
/// caller that re-folds payload bytes per graph step pays that price every step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedSourceFingerprint {
    pub byte_len: usize,
    pub fnv1a64: u64,
}

/// The workspace's one FNV-1a 64 ([`poot_runtime_common::fnv1a_bytes`]) over `bytes`. Private on
/// purpose: [`PackedPayload::source_fingerprint`] is the only supported way to obtain one, so no
/// caller can re-fold payload bytes per graph step.
fn fingerprint_source(bytes: &[u8]) -> PackedSourceFingerprint {
    PackedSourceFingerprint {
        byte_len: bytes.len(),
        fnv1a64: poot_runtime_common::fnv1a_bytes(bytes),
    }
}

/// One immutable buffer this crate owns: a dense tensor's whole stored bytes
/// ([`weights::DenseWeight`]), or one [`SourceRole`]'s bytes inside a [`PackedPayload`]. Owns the
/// bytes (today an `Arc<[u8]>`; a later mmap region changes this one type, not its callers) and
/// memoizes the content fingerprint ([`Self::fingerprint`]), so the byte-ownership policy and the
/// content identity for "bytes read off a checkpoint" live in one place (card 540a).
#[derive(Debug)]
pub struct StoredBytes {
    bytes: Arc<[u8]>,
    fingerprint: OnceLock<PackedSourceFingerprint>,
}

impl StoredBytes {
    pub fn new(bytes: Arc<[u8]>) -> Self {
        Self {
            bytes,
            fingerprint: OnceLock::new(),
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    /// A cheap `Arc` clone (a refcount bump, not a byte copy) of the owned buffer: for a caller that
    /// wants to hand these exact bytes to another `Arc`-backed owner (e.g. `poot_eval::Tensor`'s
    /// native bf16/f16 `Payload`) without decoding through an intermediate `Vec<u8>` copy.
    pub fn as_arc(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// This buffer's content fingerprint, folded at most once and memoized from then on (see
    /// [`PackedSourceFingerprint`]'s docs for the cost this amortizes).
    pub fn fingerprint(&self) -> PackedSourceFingerprint {
        *self
            .fingerprint
            .get_or_init(|| fingerprint_source(&self.bytes))
    }
}

impl Clone for StoredBytes {
    fn clone(&self) -> Self {
        Self {
            bytes: Arc::clone(&self.bytes),
            fingerprint: self.fingerprint.clone(),
        }
    }
}

impl PartialEq for StoredBytes {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl Eq for StoredBytes {}

impl From<Arc<[u8]>> for StoredBytes {
    fn from(bytes: Arc<[u8]>) -> Self {
        Self::new(bytes)
    }
}

/// One payload source: its role and its bytes. Identity is role + bytes only ([`StoredBytes`]
/// excludes the fingerprint memo from equality): the memo is observability of when the bytes were
/// folded, not part of what two sources are.
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

    /// The content fingerprint of this component's source bytes, folded at most once per payload
    /// ([`PackedPayload::source_fingerprint`]): the identity a device const cache keys a resident
    /// packed source on (card 642), so a step that re-binds the same weight costs a lookup, never a
    /// re-fold of the bytes.
    pub fn fingerprint(&self) -> PackedSourceFingerprint {
        self.owner.source_fingerprint(self.role)
    }
}

/// The device transport of one packed source's bytes (dquant.md D9): its native bytes, unchanged,
/// as zero-filled little-endian `u32` words plus one zero guard word, so a generated kernel's
/// two-word bit read (`poot_kernelgen`'s `emit_read_bits`) never reads past the buffer. The one
/// definition every executor's const bind uploads (card 642); no backend re-lays out or re-encodes
/// the bytes (ADR-0101 decision 5). It takes the bytes, not a [`PackedComponentRef`]: a mapped
/// weight's source is the stored runs its view reads (whole block rows of a fused entry, or several
/// entries back to back, Card 564), which need not be one payload's whole source. A second copy of
/// this function anywhere else is the bug Card 546a review found and fixed.
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
///
/// The source fingerprints are memoized here (see [`Self::source_fingerprint`]): the bytes never
/// change, so folding them again on every graph step would be pure waste - at Card 453's 27 GB
/// checkpoint a per-step fold measured 21 s of a 33 s decode step.
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
        sources: impl IntoIterator<Item = (SourceRole, Arc<[u8]>)>,
    ) -> Result<Self, PackedWeightError> {
        let mut provided: Vec<(SourceRole, Arc<[u8]>)> = sources.into_iter().collect();
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
            ordered.push(PayloadSource {
                role,
                bytes: StoredBytes::new(bytes),
            });
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

    fn source(&self, role: SourceRole) -> &PayloadSource {
        self.sources
            .iter()
            .find(|source| source.role == role)
            .unwrap_or_else(|| panic!("{role:?} is not a source of {:?}", self.weight.format()))
    }

    /// The content fingerprint of `role`'s source bytes, folded at most once per payload and
    /// memoized from then on (card 453's checkpoint fold cost). Reached through
    /// [`PackedComponentRef::fingerprint`], the device const caches' identity for a packed source.
    pub(crate) fn source_fingerprint(&self, role: SourceRole) -> PackedSourceFingerprint {
        self.source(role).bytes.fingerprint()
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
        let mut bytes = Vec::with_capacity(rows.len() * row_bytes);
        for &row in rows {
            let mut local = row;
            let part = parts
                .iter()
                .find(|part| {
                    let held = part.weight.shape()[0];
                    if local < held {
                        return true;
                    }
                    local -= held;
                    false
                })
                .ok_or(PackedWeightError::RowOutOfRange { row, rows: stacked })?;
            let start = local * row_bytes;
            bytes.extend_from_slice(&part.bytes(SourceRole::Blocks)[start..start + row_bytes]);
        }
        Self::try_new(weight, [(SourceRole::Blocks, Arc::from(bytes))])
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
    match descriptor.storage {
        Storage::Blocks(_) => {
            let bytes = source_bytes(SourceRole::Blocks);
            if let Some((operand, element)) = descriptor.first_non_finite_block_field(bytes) {
                return Err(PackedWeightError::NonFiniteField {
                    format,
                    role: SourceRole::Blocks,
                    operand,
                    element,
                });
            }
        }
        Storage::Planar(layout) => {
            for operand in layout.operands {
                let role = SourceRole::Planar(operand.role);
                let bytes = source_bytes(role);
                if let Some(element) =
                    descriptor.first_non_finite_planar_operand(operand, weight.shape(), bytes)?
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
