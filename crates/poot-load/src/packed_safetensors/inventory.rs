//! Authenticated artifact identity, tensor inventory, retained handles, and owner caches.

use poot_quant::weights::WeightStore;

use super::{
    Arc, BTreeMap, BTreeSet, File, FilePin, Hash, HashMap, HashSet, Hasher, MixedAccountingTable,
    MixedByteQuantity, MixedReportInputs, MixedWeightStoreError, OperandRole, PackedDecisionPair,
    PackedPayload, PackedWeight, PackedWeightDecision, PackedWeightError, Path, RangeCheck, Read,
    SHA256, Seek, SourceRole, StagedExactRow, StagedMixedSelection, StagedOwnerRef, StagedRow,
    ValidatedExactSelection, ValidatedMixedSelection, ValidatedSelection, WeightFormat, admit_file,
    admit_shard, artifact_cache_identity, build_mixed_report, category_for_disposition,
    checked_add, commit_mixed_publication, commit_packed_publication, disposition_selects_source,
    enforce_limit, exact_kind_for_disposition, file_length, fmt, open_regular,
    parse_header_entries, parse_unique_json, parse_weight_map, prepare_mixed_publication,
    prepare_packed_publication, read_exact_file, read_header, read_header_length, read_source_arc,
    report_selected_accounting, shape_2, validate_index_header_bijection,
    validate_manifest_and_limits, validate_relative_path, validate_selection_row,
    validate_shard_overlaps,
};

pub(crate) const CONFIG_FILE: &str = "config.json";

pub(crate) const INDEX_FILE: &str = "model.safetensors.index.json";

pub(crate) const HASH_BUFFER_BYTES: usize = 4096;

/// The cache-identity placeholder for a shard's file digest when it has no manifest pin (card
/// 544): the reader never hashes a whole unpinned shard just to fill this field, so the identity
/// records "no pin was given" instead of a real content digest. Cache-key uniqueness for an
/// unpinned shard still comes from its filename, length and header digest, all cheap to know.
const NO_PIN_DIGEST: Sha256Digest = Sha256Digest::new([0u8; 32]);

pub(crate) trait RetainedReader: Read + Seek {
    fn file_length(&self) -> Result<u64, std::io::Error>;
}

impl RetainedReader for File {
    fn file_length(&self) -> Result<u64, std::io::Error> {
        self.metadata().map(|metadata| metadata.len())
    }
}

pub(crate) type RetainedHandle = Box<dyn RetainedReader>;

pub(crate) type OpenedShards = BTreeMap<String, (Option<PackedShardManifest>, RetainedHandle)>;

/// Opaque ownership of the one bounded `config.json` snapshot used for dispatch and authentication.
///
/// This type is mechanically public only for the workspace-internal `poot-load` to `poot-llm`
/// handoff. Its byte owner and retained handle can only be consumed by
/// [`AuthenticatedSafetensorsHandleSet::authenticate_with_retained_config`].
#[cfg(test)]
pub(crate) struct RetainedConfigCapability {
    file: RetainedHandle,
    bytes: Vec<u8>,
    pre_read_length: usize,
    pre_read_regular: bool,
}

#[cfg(test)]
impl fmt::Debug for RetainedConfigCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetainedConfigCapability")
            .field("bytes", &self.bytes.len())
            .field("pre_read_length", &self.pre_read_length)
            .field("pre_read_regular", &self.pre_read_regular)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
impl RetainedConfigCapability {
    /// Borrow the bounded bytes retained for dispatch.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the file length observed before the bounded read.
    pub(crate) const fn pre_read_length(&self) -> usize {
        self.pre_read_length
    }

    /// Return whether the pre-read metadata identified a regular file.
    pub(crate) const fn pre_read_regular(&self) -> bool {
        self.pre_read_regular
    }
}

/// Bounded-open and retain the sole `config.json` allocation used by exact-model dispatch.
#[cfg(test)]
pub(crate) fn retain_config_for_authentication(
    root: impl AsRef<Path>,
    limit: usize,
) -> Result<RetainedConfigCapability, PackedSafetensorsError> {
    if limit == 0 {
        return Err(PackedSafetensorsError::ZeroLimit {
            kind: PackedLimitKind::ConfigBytes,
        });
    }
    let mut file = open_regular(root.as_ref(), CONFIG_FILE)?;
    let pre_read_length = file_length(file.as_ref(), CONFIG_FILE)?;
    let mut bytes = Vec::with_capacity(pre_read_length.min(limit.saturating_add(1)));
    file.by_ref()
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| PackedSafetensorsError::Io {
            file: CONFIG_FILE.to_string(),
            error,
        })?;
    enforce_limit(PackedLimitKind::ConfigBytes, limit, bytes.len())?;
    Ok(RetainedConfigCapability {
        file,
        bytes,
        pre_read_length,
        pre_read_regular: true,
    })
}

/// An exact SHA-256 digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sha256Digest(pub(crate) [u8; 32]);

impl Sha256Digest {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl From<[u8; 32]> for Sha256Digest {
    fn from(value: [u8; 32]) -> Self {
        Self::new(value)
    }
}

/// Hash a byte slice using the artifact identity algorithm.
pub fn sha256_digest(bytes: &[u8]) -> Sha256Digest {
    let digest = ring::digest::digest(&SHA256, bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_ref());
    Sha256Digest(out)
}

/// Exact identity for one shard in a packed artifact.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PackedShardManifest {
    pub filename: String,
    pub file_length: usize,
    pub file_sha256: Sha256Digest,
    pub header_length: usize,
    pub header_sha256: Sha256Digest,
}

/// Caller-supplied immutable identity for the local artifact.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PackedArtifactManifest {
    pub repository: String,
    pub revision: String,
    pub config_length: usize,
    pub config_sha256: Sha256Digest,
    pub index_sha256: Sha256Digest,
    pub shards: Vec<PackedShardManifest>,
}

/// Explicit nonzero bounds for all attacker-controlled metadata and selected data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedSafetensorsLimits {
    pub config_bytes: usize,
    pub index_bytes: usize,
    pub header_bytes_per_shard: usize,
    pub shard_count: usize,
    pub tensor_entries: usize,
    pub selected_source_bytes: usize,
    pub packed_source_bytes: usize,
}

/// A data-section-relative half-open byte span.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceSpan {
    pub start: usize,
    pub end: usize,
}

impl SourceSpan {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn len(self) -> Result<usize, PackedSafetensorsError> {
        self.end
            .checked_sub(self.start)
            .ok_or(PackedSafetensorsError::InvalidSpan {
                file: "<selection>".to_string(),
                tensor: "selection".to_string(),
                span: self,
                data_length: 0,
            })
    }

    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// One exact weight/scale request. Model-local code may construct these rows later.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PackedSelectionRow {
    pub linear_id: String,
    pub descriptor: PackedWeight,
    pub weight_name: String,
    pub scale_name: String,
    pub shard: String,
    pub weight_span: SourceSpan,
    pub scale_span: SourceSpan,
    pub weight_dtype: String,
    pub scale_dtype: String,
    pub weight_shape: [usize; 2],
    pub scale_shape: [usize; 2],
}

/// An artifact's repository and revision label, independent of any digest pin (card 544): a
/// capability-admitted checkpoint still needs a real cache identity, so this is always required,
/// never drawn from an optional manifest. One parameter instead of two keeps
/// [`AuthenticatedSafetensorsHandleSet::authenticate_retained_config_bytes`] under clippy's
/// argument-count lint.
pub(crate) struct ArtifactLabel {
    pub(crate) repository: String,
    pub(crate) revision: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ArtifactIdentityFields {
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) config_length: usize,
    pub(crate) config_sha256: Sha256Digest,
    pub(crate) index_length: usize,
    pub(crate) index_sha256: Sha256Digest,
    pub(crate) shards: Vec<PackedShardManifest>,
}

/// Immutable identity for every file authenticated by one artifact manifest.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AuthenticatedArtifactIdentity(pub(crate) Arc<ArtifactIdentityFields>);

impl AuthenticatedArtifactIdentity {
    pub fn repository(&self) -> &str {
        &self.0.repository
    }

    pub fn revision(&self) -> &str {
        &self.0.revision
    }

    pub fn config_length(&self) -> usize {
        self.0.config_length
    }

    pub fn config_sha256(&self) -> Sha256Digest {
        self.0.config_sha256
    }

    pub fn index_length(&self) -> usize {
        self.0.index_length
    }

    pub fn index_sha256(&self) -> Sha256Digest {
        self.0.index_sha256
    }

    pub fn shards(&self) -> &[PackedShardManifest] {
        &self.0.shards
    }
}

#[derive(Debug)]
pub(crate) struct InventorySession;

/// An opaque key valid only for the authenticated session that produced it.
#[derive(Clone)]
pub struct AuthenticatedInventoryRowKey {
    pub(crate) session: Arc<InventorySession>,
    pub(crate) ordinal: usize,
}

impl fmt::Debug for AuthenticatedInventoryRowKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthenticatedInventoryRowKey")
            .field("ordinal", &self.ordinal)
            .finish_non_exhaustive()
    }
}

impl PartialEq for AuthenticatedInventoryRowKey {
    fn eq(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal && Arc::ptr_eq(&self.session, &other.session)
    }
}

impl Eq for AuthenticatedInventoryRowKey {}

impl Hash for AuthenticatedInventoryRowKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.session).hash(state);
        self.ordinal.hash(state);
    }
}

/// Immutable authenticated metadata for one tensor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AuthenticatedTensorDescriptor {
    pub(crate) name: String,
    pub(crate) shard: String,
    pub(crate) span: SourceSpan,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
}

impl AuthenticatedTensorDescriptor {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shard(&self) -> &str {
        &self.shard
    }

    pub const fn span(&self) -> SourceSpan {
        self.span
    }

    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// One row borrowed from an authenticated inventory.
#[derive(Clone, Debug)]
pub struct AuthenticatedInventoryRow<'a> {
    pub(crate) key: AuthenticatedInventoryRowKey,
    pub(crate) descriptor: &'a AuthenticatedTensorDescriptor,
}

impl AuthenticatedInventoryRow<'_> {
    pub fn key(&self) -> AuthenticatedInventoryRowKey {
        self.key.clone()
    }

    pub const fn descriptor(&self) -> &AuthenticatedTensorDescriptor {
        self.descriptor
    }

    pub fn name(&self) -> &str {
        self.descriptor.name()
    }

    pub fn shard(&self) -> &str {
        self.descriptor.shard()
    }

    pub const fn span(&self) -> SourceSpan {
        self.descriptor.span()
    }

    pub fn dtype(&self) -> &str {
        self.descriptor.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        self.descriptor.shape()
    }
}

/// Deterministically ordered metadata borrowed from one held snapshot.
#[derive(Clone, Copy)]
pub struct AuthenticatedInventory<'a> {
    pub(crate) session: &'a Arc<InventorySession>,
    pub(crate) descriptors: &'a [AuthenticatedTensorDescriptor],
}

impl<'a> AuthenticatedInventory<'a> {
    pub const fn len(self) -> usize {
        self.descriptors.len()
    }

    pub const fn is_empty(self) -> bool {
        self.descriptors.is_empty()
    }

    pub fn rows(self) -> impl ExactSizeIterator<Item = AuthenticatedInventoryRow<'a>> + 'a {
        self.descriptors
            .iter()
            .enumerate()
            .map(|(ordinal, descriptor)| AuthenticatedInventoryRow {
                key: AuthenticatedInventoryRowKey {
                    session: Arc::clone(self.session),
                    ordinal,
                },
                descriptor,
            })
    }
}

/// The model-neutral disposition assigned to one authenticated inventory row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorDisposition {
    PackedWeight {
        linear_id: String,
        format: WeightFormat,
        logical_shape: [usize; 2],
    },
    PackedScale {
        linear_id: String,
    },
    StandaloneE4m3,
    DenseBf16,
    DenseF32,
    DenseI64,
    Deferred,
    Excluded,
}

/// One classification decision containing no caller-reconstructed source metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InventoryDecision {
    pub key: AuthenticatedInventoryRowKey,
    pub disposition: TensorDisposition,
}

impl InventoryDecision {
    pub fn new(key: AuthenticatedInventoryRowKey, disposition: TensorDisposition) -> Self {
        Self { key, disposition }
    }
}

/// Exact standalone source types admitted by mixed assembly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum ExactSourceKind {
    E4m3 = 0,
    Bf16 = 1,
    F32 = 2,
    I64 = 3,
}

impl ExactSourceKind {
    pub const ALL: [Self; 4] = [Self::E4m3, Self::Bf16, Self::F32, Self::I64];

    pub(crate) const fn dtype(self) -> &'static str {
        match self {
            Self::E4m3 => "F8_E4M3",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::I64 => "I64",
        }
    }
}

/// One immutable, typed owner of exact checkpoint source bytes.
#[derive(Clone, Debug)]
pub struct ExactSourceOwner {
    pub(crate) artifact: AuthenticatedArtifactIdentity,
    pub(crate) descriptor: AuthenticatedTensorDescriptor,
    pub(crate) kind: ExactSourceKind,
    pub(crate) bytes: Arc<[u8]>,
}

impl ExactSourceOwner {
    pub const fn artifact(&self) -> &AuthenticatedArtifactIdentity {
        &self.artifact
    }

    pub const fn descriptor(&self) -> &AuthenticatedTensorDescriptor {
        &self.descriptor
    }

    pub const fn kind(&self) -> ExactSourceKind {
        self.kind
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Which selected source buffer failed to read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedBufferKind {
    Weight,
    Scale,
}

/// A caller-controlled bound rejected by authentication or selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedLimitKind {
    ConfigBytes,
    IndexBytes,
    HeaderBytesPerShard,
    ShardCount,
    TensorEntries,
    SelectedSourceBytes,
    PackedSourceBytes,
}

/// An exact selection field that disagreed with the authenticated header or descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedSelectionField {
    Shard,
    WeightSpan,
    ScaleSpan,
    WeightDtype,
    ScaleDtype,
    WeightShape,
    ScaleShape,
    WeightSourceBytes,
    ScaleSourceBytes,
}

/// A cache whose capacity could not be reserved before publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnerCacheKind {
    Packed,
    ExactSource,
}

/// Typed failures from packed artifact authentication and selection.
#[derive(Debug, thiserror::Error)]
pub enum PackedSafetensorsError {
    #[error("{file}: I/O failed: {error}")]
    Io {
        file: String,
        #[source]
        error: std::io::Error,
    },
    #[error("{file}: invalid JSON: {reason}")]
    Json { file: String, reason: String },
    #[error("{file}: duplicate JSON key {key:?}")]
    DuplicateJsonKey { file: String, key: String },
    #[error("manifest repository and revision labels must be nonempty")]
    EmptyArtifactLabel,
    #[error("artifact manifest must contain at least one shard")]
    EmptyShardManifest,
    #[error("limit {kind:?} must be nonzero")]
    ZeroLimit { kind: PackedLimitKind },
    #[error("limit {kind:?} is {limit} bytes/items, observed {actual}")]
    LimitExceeded {
        kind: PackedLimitKind,
        limit: usize,
        actual: usize,
    },
    #[error("invalid relative shard path {path:?}")]
    InvalidShardPath { path: String },
    #[error("duplicate shard manifest row {filename:?}")]
    DuplicateShardManifest { filename: String },
    #[error("{file}: expected a regular file")]
    NotRegularFile { file: String },
    #[error("{file}: file length mismatch: expected {expected}, got {actual}")]
    FileLengthMismatch {
        file: String,
        expected: usize,
        actual: usize,
    },
    #[error("config length mismatch: expected {expected}, got {actual}")]
    ConfigLengthMismatch { expected: usize, actual: usize },
    #[error("{file}: digest mismatch")]
    DigestMismatch { file: String },
    #[error("{file}: header length mismatch: expected {expected}, got {actual}")]
    HeaderLengthMismatch {
        file: String,
        expected: usize,
        actual: usize,
    },
    #[error("{file}: header digest mismatch")]
    HeaderDigestMismatch { file: String },
    #[error("index has no object-valued weight_map")]
    MissingWeightMap,
    #[error("index weight_map value for {tensor:?} is not a string")]
    InvalidWeightMapValue { tensor: String },
    #[error("index shard set differs from manifest: index={index:?}, manifest={manifest:?}")]
    ShardSetMismatch {
        index: Vec<String>,
        manifest: Vec<String>,
    },
    #[error("{file}: tensor {tensor:?} metadata is invalid: {reason}")]
    InvalidTensorMetadata {
        file: String,
        tensor: String,
        reason: String,
    },
    #[error("{file}: tensor {tensor:?} has invalid span {span:?} for {data_length} data bytes")]
    InvalidSpan {
        file: String,
        tensor: String,
        span: SourceSpan,
        data_length: usize,
    },
    #[error("{file}: tensor {tensor:?} byte count overflows usize")]
    TensorByteCountOverflow { file: String, tensor: String },
    #[error(
        "{file}: tensor {tensor:?} has {actual} source bytes, expected {expected} for its dtype and shape"
    )]
    TensorByteCountMismatch {
        file: String,
        tensor: String,
        expected: usize,
        actual: usize,
    },
    #[error("tensor name {tensor:?} appears in both {first_shard:?} and {second_shard:?}")]
    DuplicateTensorAcrossShards {
        tensor: String,
        first_shard: String,
        second_shard: String,
    },
    #[error("index tensor {tensor:?} is missing from shard {shard:?}")]
    IndexTensorMissing { tensor: String, shard: String },
    #[error(
        "header tensor {tensor:?} in {header_shard:?} is mapped by the index to {index_shard:?}"
    )]
    IndexHeaderShardMismatch {
        tensor: String,
        index_shard: String,
        header_shard: String,
    },
    #[error("header tensor {tensor:?} in {shard:?} is absent from the index")]
    HeaderTensorMissingFromIndex { tensor: String, shard: String },
    #[error("{shard}: tensor spans overlap: {first:?} and {second:?}")]
    OverlappingTensorSpans {
        shard: String,
        first: String,
        second: String,
    },
    #[error("duplicate selection linear id {linear_id:?}")]
    DuplicateLinearId { linear_id: String },
    #[error("selection reuses source tensor {tensor:?}")]
    ReusedSourceName { tensor: String },
    #[error("selection reuses source span {span:?} in shard {shard:?}")]
    ReusedSourceSpan { shard: String, span: SourceSpan },
    #[error("selection {linear_id:?} aliases weight and scale name or span")]
    AliasedPair { linear_id: String },
    #[error("selection {linear_id:?} tensor {tensor:?} is absent")]
    MissingSelectionTensor { linear_id: String, tensor: String },
    #[error("selection {linear_id:?} is split between shards {weight_shard:?} and {scale_shard:?}")]
    SplitPair {
        linear_id: String,
        weight_shard: String,
        scale_shard: String,
    },
    #[error("selection {linear_id:?} {field:?} mismatch: expected {expected}, got {actual}")]
    SelectionMismatch {
        linear_id: String,
        field: PackedSelectionField,
        expected: String,
        actual: String,
    },
    #[error("checked arithmetic overflow while deriving {field}")]
    ArithmeticOverflow { field: &'static str },
    #[error(
        "selection {linear_id:?} {buffer:?} read failed after {completed_payloads} payloads: {error}"
    )]
    SelectedRead {
        linear_id: String,
        buffer: PackedBufferKind,
        completed_payloads: usize,
        #[source]
        error: std::io::Error,
    },
    #[error(
        "selection {linear_id:?} payload construction failed after {completed_payloads} payloads: {error}"
    )]
    PayloadBuild {
        linear_id: String,
        completed_payloads: usize,
        #[source]
        error: PackedWeightError,
    },
    #[error("authenticated artifact changed before publication: {file}")]
    ArtifactChanged { file: String },
    #[error("inventory key {ordinal} belongs to another authenticated session")]
    ForeignInventoryKey { ordinal: usize },
    #[error("inventory key {ordinal} is outside the authenticated inventory")]
    InvalidInventoryKey { ordinal: usize },
    #[error("inventory key {ordinal} was classified more than once")]
    DuplicateInventoryKey { ordinal: usize },
    #[error("authenticated tensor {tensor:?} has no disposition")]
    MissingInventoryDecision { tensor: String },
    #[error("tensor {tensor:?} with dtype {dtype:?} is incompatible with {disposition:?}")]
    DispositionDtypeMismatch {
        tensor: String,
        dtype: String,
        disposition: TensorDisposition,
    },
    #[error("packed linear {linear_id:?} has more than one {component:?} decision")]
    DuplicatePackedComponent {
        linear_id: String,
        component: PackedBufferKind,
    },
    #[error("packed linear {linear_id:?} is missing its {component:?} decision")]
    MissingPackedComponent {
        linear_id: String,
        component: PackedBufferKind,
    },
    #[error("packed linear {linear_id:?} descriptor is invalid: {error}")]
    PackedDescriptor {
        linear_id: String,
        #[source]
        error: PackedWeightError,
    },
    #[error("exact source {tensor:?} read failed after {completed_owners} owners: {error}")]
    ExactSourceRead {
        tensor: String,
        completed_owners: usize,
        #[source]
        error: std::io::Error,
    },
    #[error("could not reserve {additional} entries in the {cache:?} owner cache")]
    CacheReservation {
        cache: OwnerCacheKind,
        additional: usize,
    },
    #[error("building the weight store from this snapshot's published rows failed: {error}")]
    WeightStoreBuild {
        #[source]
        error: MixedWeightStoreError,
    },
}

/// A classifier failure or a typed shared-loader failure.
#[derive(Debug, thiserror::Error)]
pub enum MixedLoadError<E> {
    #[error("mixed inventory classifier rejected the snapshot: {0}")]
    Classifier(E),
    #[error(transparent)]
    Assembly(#[from] PackedSafetensorsError),
}

/// Exact aggregate ownership and I/O accounting for one successful selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackedLoadReport {
    pub config_bytes: usize,
    pub index_bytes: usize,
    pub header_bytes_total: usize,
    pub header_bytes_max: usize,
    pub initial_artifact_hash_bytes: usize,
    pub final_verification_bytes: usize,
    pub selected_weight_source_bytes: usize,
    pub selected_scale_source_bytes: usize,
    pub source_padding_bits: usize,
    pub packed_source_bytes: usize,
    pub peak_io_buffer_bytes: usize,
    pub peak_unpublished_payload_bytes: usize,
    pub cold_new_owner_bytes: usize,
    pub warm_reused_owner_bytes: usize,
    pub selected_range_read_bytes: usize,
    pub payload_builds: usize,
    pub forbidden_f32_weight_bytes: usize,
}

/// One selected packed owner and its exact source provenance.
#[derive(Clone, Debug)]
pub struct LoadedPackedLinear {
    pub linear_id: String,
    pub descriptor: PackedWeight,
    pub weight_name: String,
    pub scale_name: String,
    pub shard: String,
    pub weight_span: SourceSpan,
    pub scale_span: SourceSpan,
    pub owner: Arc<PackedPayload>,
}

/// Atomically published rows and their accounting.
#[derive(Clone, Debug)]
pub struct PackedLoadResult {
    pub rows: Vec<LoadedPackedLinear>,
    pub report: PackedLoadReport,
}

/// Mutually exclusive source-accounting categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum MixedSourceCategory {
    Packed = 0,
    StandaloneE4m3 = 1,
    DenseBf16 = 2,
    DenseF32 = 3,
    DenseI64 = 4,
    DeferredOrExcluded = 5,
}

impl MixedSourceCategory {
    pub const ALL: [Self; 6] = [
        Self::Packed,
        Self::StandaloneE4m3,
        Self::DenseBf16,
        Self::DenseF32,
        Self::DenseI64,
        Self::DeferredOrExcluded,
    ];
}

/// One table-indexed byte quantity and its aggregate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MixedCategoryBytes {
    pub(crate) values: [usize; MixedSourceCategory::ALL.len()],
    pub(crate) aggregate: usize,
}

impl MixedCategoryBytes {
    pub const fn get(self, category: MixedSourceCategory) -> usize {
        self.values[category as usize]
    }

    pub const fn aggregate(self) -> usize {
        self.aggregate
    }
}

/// Exact-source build counts indexed by [`ExactSourceKind`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExactSourceBuildCounts {
    pub(crate) values: [usize; ExactSourceKind::ALL.len()],
    pub(crate) total: usize,
}

impl ExactSourceBuildCounts {
    pub const fn get(self, kind: ExactSourceKind) -> usize {
        self.values[kind as usize]
    }

    pub const fn total(self) -> usize {
        self.total
    }
}

/// Checked ownership, metadata, and I/O accounting for one mixed snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MixedLoadReport {
    pub source_bytes: MixedCategoryBytes,
    pub selected_source_bytes: MixedCategoryBytes,
    pub selected_range_read_bytes: MixedCategoryBytes,
    pub new_owner_bytes: MixedCategoryBytes,
    pub reused_owner_bytes: MixedCategoryBytes,
    pub aggregate_owner_bytes: MixedCategoryBytes,
    pub peak_in_flight_read_bytes: MixedCategoryBytes,
    pub peak_unpublished_owner_bytes: MixedCategoryBytes,
    pub config_bytes: usize,
    pub index_bytes: usize,
    pub header_bytes_total: usize,
    pub header_bytes_max: usize,
    pub metadata_bytes: usize,
    pub initial_hash_io_bytes: usize,
    pub final_hash_io_bytes: usize,
    pub hash_buffer_bytes: usize,
    pub packed_weight_source_bytes: usize,
    pub packed_scale_source_bytes: usize,
    pub packed_source_padding_bits: usize,
    pub forbidden_f32_weight_bytes: usize,
    pub packed_payload_builds: usize,
    pub exact_source_builds: ExactSourceBuildCounts,
    /// [`WeightStore::total_stored_bytes`] of this snapshot's own [`MixedLoadResult::store`]
    /// (card 540a): the stored footprint of every published row, read from the store itself
    /// instead of a second, hand-kept count (ADR-0103 decision 1: never a widened decode).
    pub stored_bytes: usize,
}

/// One immutable inventory row and its published disposition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassifiedInventoryRow {
    pub descriptor: AuthenticatedTensorDescriptor,
    pub disposition: TensorDisposition,
}

/// One exact-source owner and its authenticated provenance, packaged for a *consumer's own*
/// retained subset (e.g. [`poot_models`]'s per-model ownership plans, keyed by layer/role instead
/// of tensor name). Not [`MixedLoadResult`]'s own field: the mixed loader's one load output is its
/// [`WeightStore`] plus [`ExactSourceMetadata`], keyed by tensor name (card 540a); a
/// consumer that wants a smaller, differently-keyed slice builds its own `Vec` of these from that
/// store and metadata, rather than the loader keeping a second, parallel copy of every row.
#[derive(Clone, Debug)]
pub struct LoadedExactSource {
    pub descriptor: AuthenticatedTensorDescriptor,
    pub kind: ExactSourceKind,
    pub owner: Arc<ExactSourceOwner>,
}

/// Checkpoint provenance for one packed linear's [`WeightStore`] entry (`WeightEntry::Packed`,
/// keyed by linear id): the weight/scale tensor names and shard byte spans the payload's bytes
/// were read from. Not on [`PackedPayload`] itself (card 540a): a payload only knows
/// its decoded [`PackedWeight`] descriptor and bytes, never where in the checkpoint they came
/// from, so this is the one place that fact lives once `MixedLoadResult` stops keeping a second,
/// parallel row per linear.
#[derive(Clone, Debug)]
pub struct PackedSourceMetadata {
    pub weight_name: String,
    pub scale_name: String,
    pub shard: String,
    pub weight_span: SourceSpan,
    pub scale_span: SourceSpan,
}

/// Checkpoint provenance for one exact/dense [`WeightStore`] entry (`WeightEntry::Dense`, keyed
/// by tensor name): the shard byte span, the [`ExactSourceKind`] the classifier assigned this
/// row, and the [`ExactSourceOwner`] handle. The handle is not a second copy of the bytes: the
/// store's [`DenseWeight`](poot_quant::weights::DenseWeight) is built by `Arc::clone`-ing this same
/// owner's bytes (card 540a), so both point at one allocation. It is kept because a
/// `DenseWeight` carries only a stored dtype and shape, while the exact executors need the
/// authenticated source itself: `poot-eval`'s exact values (`ExactValue`) are built
/// from an `Arc<ExactSourceOwner>` (its authenticated descriptor and kind, checked again at bind),
/// and a warm owner-cache hit is proven by that `Arc`'s identity. Provenance alone could not
/// rebuild either without re-reading the checkpoint.
#[derive(Clone, Debug)]
pub struct ExactSourceMetadata {
    pub kind: ExactSourceKind,
    pub shard: String,
    pub span: SourceSpan,
    pub owner: Arc<ExactSourceOwner>,
}

/// Atomically published packed and exact-source owners from one authenticated snapshot: one
/// [`WeightStore`] entry per published row (`WeightEntry::Packed` keyed by linear id,
/// `WeightEntry::Dense` keyed by tensor name), plus the checkpoint provenance the store cannot
/// express, in one map per kind keyed the same way (card 540a: the store is this
/// snapshot's one load output, not a view derived from - and dropped after - a parallel
/// `packed`/`exact_sources` pair of vectors).
#[derive(Clone, Debug)]
pub struct MixedLoadResult {
    pub artifact: AuthenticatedArtifactIdentity,
    pub inventory: Vec<ClassifiedInventoryRow>,
    pub store: WeightStore,
    pub packed_metadata: BTreeMap<String, PackedSourceMetadata>,
    pub exact_metadata: BTreeMap<String, ExactSourceMetadata>,
    pub report: MixedLoadReport,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PackedOwnerCacheKey {
    pub(crate) artifact: Sha256Digest,
    pub(crate) descriptor: PackedWeight,
    pub(crate) weight_name: String,
    pub(crate) scale_name: String,
    pub(crate) shard: String,
    pub(crate) weight_span: SourceSpan,
    pub(crate) scale_span: SourceSpan,
    pub(crate) weight_dtype: String,
    pub(crate) scale_dtype: String,
    pub(crate) weight_shape: [usize; 2],
    pub(crate) scale_shape: [usize; 2],
}

/// Explicit in-memory reuse for canonical payload owners.
#[derive(Default)]
pub struct PackedOwnerCache {
    pub(crate) owners: HashMap<PackedOwnerCacheKey, Arc<PackedPayload>>,
}

impl PackedOwnerCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.owners.len()
    }

    pub fn is_empty(&self) -> bool {
        self.owners.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ExactSourceCacheKey {
    pub(crate) artifact: AuthenticatedArtifactIdentity,
    pub(crate) descriptor: AuthenticatedTensorDescriptor,
    pub(crate) kind: ExactSourceKind,
}

/// Explicit in-memory reuse for exact standalone source owners.
#[derive(Default)]
pub struct ExactSourceOwnerCache {
    pub(crate) owners: HashMap<ExactSourceCacheKey, Arc<ExactSourceOwner>>,
}

impl ExactSourceOwnerCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.owners.len()
    }

    pub fn is_empty(&self) -> bool {
        self.owners.is_empty()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TensorEntry {
    pub(crate) shard: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) span: SourceSpan,
}

pub(crate) struct ShardState {
    pub(crate) manifest: Option<PackedShardManifest>,
    pub(crate) file: RetainedHandle,
    pub(crate) data_start: usize,
}

/// Authenticated files retained across metadata validation, selection reads, and final verification.
pub struct AuthenticatedSafetensorsHandleSet {
    pub(crate) artifact_cache_identity: Sha256Digest,
    pub(crate) artifact_identity: AuthenticatedArtifactIdentity,
    pub(crate) config_bytes: Vec<u8>,
    pub(crate) inventory_session: Arc<InventorySession>,
    pub(crate) inventory: Vec<AuthenticatedTensorDescriptor>,
    #[cfg(test)]
    pub(crate) index_bytes: Vec<u8>,
    pub(crate) shards: BTreeMap<String, ShardState>,
    pub(crate) tensors: BTreeMap<String, TensorEntry>,
    pub(crate) authentication_report: PackedLoadReport,
    pub(crate) limits: PackedSafetensorsLimits,
}

fn open_index_and_shards(
    root: &Path,
    manifest: &mut PackedArtifactManifest,
) -> Result<(RetainedHandle, OpenedShards), PackedSafetensorsError> {
    let index_file = open_regular(root, INDEX_FILE)?;
    let mut shard_manifests = BTreeMap::new();
    for shard in manifest.shards.drain(..) {
        validate_relative_path(&shard.filename)?;
        let filename = shard.filename.clone();
        if shard_manifests.insert(filename.clone(), shard).is_some() {
            return Err(PackedSafetensorsError::DuplicateShardManifest { filename });
        }
    }
    manifest.shards = shard_manifests.values().cloned().collect();

    let mut opened_shards = BTreeMap::new();
    for (filename, shard_manifest) in shard_manifests {
        let file = open_regular(root, &filename)?;
        opened_shards.insert(filename, (Some(shard_manifest), file));
    }
    Ok((index_file, opened_shards))
}

fn validate_config_length(
    config_file: &dyn RetainedReader,
    manifest: &PackedArtifactManifest,
    limits: PackedSafetensorsLimits,
) -> Result<usize, PackedSafetensorsError> {
    let config_length = file_length(config_file, CONFIG_FILE)?;
    if config_length != manifest.config_length {
        return Err(PackedSafetensorsError::ConfigLengthMismatch {
            expected: manifest.config_length,
            actual: config_length,
        });
    }
    enforce_limit(
        PackedLimitKind::ConfigBytes,
        limits.config_bytes,
        config_length,
    )?;
    Ok(config_length)
}

impl AuthenticatedSafetensorsHandleSet {
    /// Open and authenticate the complete local file set before any selection is inspected.
    pub fn authenticate(
        root: impl AsRef<Path>,
        mut manifest: PackedArtifactManifest,
        limits: PackedSafetensorsLimits,
    ) -> Result<Self, PackedSafetensorsError> {
        validate_manifest_and_limits(&manifest, limits)?;
        let root = root.as_ref();

        let config_file = open_regular(root, CONFIG_FILE)?;
        let (index_file, opened_shards) = open_index_and_shards(root, &mut manifest)?;

        Self::authenticate_retained(manifest, limits, config_file, index_file, opened_shards)
    }

    /// Authenticate using config bytes already read from the retained `config.json` handle.
    ///
    /// This is an internal cross-crate seam for exact-model dispatch: it lets a caller parse the same byte owner
    /// that this transaction authenticates, without reopening the config path or allocating a second snapshot.
    #[cfg(test)]
    pub(crate) fn authenticate_with_retained_config(
        root: impl AsRef<Path>,
        mut manifest: PackedArtifactManifest,
        limits: PackedSafetensorsLimits,
        retained_config: RetainedConfigCapability,
    ) -> Result<Self, PackedSafetensorsError> {
        validate_manifest_and_limits(&manifest, limits)?;
        let root = root.as_ref();
        if !retained_config.pre_read_regular {
            return Err(PackedSafetensorsError::NotRegularFile {
                file: CONFIG_FILE.to_string(),
            });
        }
        if retained_config.pre_read_length != manifest.config_length {
            return Err(PackedSafetensorsError::ConfigLengthMismatch {
                expected: manifest.config_length,
                actual: retained_config.pre_read_length,
            });
        }

        let (index_file, opened_shards) = open_index_and_shards(root, &mut manifest)?;

        let label = ArtifactLabel {
            repository: manifest.repository.clone(),
            revision: manifest.revision.clone(),
        };
        Self::authenticate_retained_config_bytes(
            label,
            Some(manifest),
            limits,
            retained_config.file,
            retained_config.bytes,
            index_file,
            opened_shards,
        )
    }

    pub(crate) fn authenticate_retained(
        manifest: PackedArtifactManifest,
        limits: PackedSafetensorsLimits,
        mut config_file: RetainedHandle,
        index_file: RetainedHandle,
        opened_shards: OpenedShards,
    ) -> Result<Self, PackedSafetensorsError> {
        validate_manifest_and_limits(&manifest, limits)?;

        let config_length = validate_config_length(config_file.as_ref(), &manifest, limits)?;
        let config_bytes = read_exact_file(config_file.as_mut(), CONFIG_FILE, config_length)?;
        let label = ArtifactLabel {
            repository: manifest.repository.clone(),
            revision: manifest.revision.clone(),
        };
        Self::authenticate_retained_config_bytes(
            label,
            Some(manifest),
            limits,
            config_file,
            config_bytes,
            index_file,
            opened_shards,
        )
    }

    /// The one generic integrity step for this artifact's small files and every shard (card 544,
    /// ADR-0103 decision 4). `manifest` is optional integrity data: `Some` pins config, index and
    /// every shard's length and digest, verified once through [`admit_file`]/[`admit_shard`];
    /// `None` admits the same files by capability alone - structural validation (header parses,
    /// declared shapes and dtypes fit, spans do not overlap, the index and every header agree)
    /// still runs unconditionally, so a checkpoint with no pin is not exempt from that, only from
    /// identity comparison. `repository`/`revision` are always required labels, independent of a
    /// pin, so a capability-admitted artifact still has a real cache identity.
    pub(super) fn authenticate_retained_config_bytes(
        label: ArtifactLabel,
        manifest: Option<PackedArtifactManifest>,
        limits: PackedSafetensorsLimits,
        config_file: RetainedHandle,
        config_bytes: Vec<u8>,
        mut index_file: RetainedHandle,
        opened_shards: OpenedShards,
    ) -> Result<Self, PackedSafetensorsError> {
        let ArtifactLabel {
            repository,
            revision,
        } = label;
        if repository.is_empty() || revision.is_empty() {
            return Err(PackedSafetensorsError::EmptyArtifactLabel);
        }
        if let Some(manifest) = &manifest {
            validate_manifest_and_limits(manifest, limits)?;
        }
        if opened_shards.is_empty() {
            return Err(PackedSafetensorsError::EmptyShardManifest);
        }
        enforce_limit(
            PackedLimitKind::ShardCount,
            limits.shard_count,
            opened_shards.len(),
        )?;

        let probed_config_length = file_length(config_file.as_ref(), CONFIG_FILE)?;
        enforce_limit(
            PackedLimitKind::ConfigBytes,
            limits.config_bytes,
            probed_config_length,
        )?;
        if config_bytes.len() != probed_config_length {
            return Err(PackedSafetensorsError::ConfigLengthMismatch {
                expected: probed_config_length,
                actual: config_bytes.len(),
            });
        }
        let config_length = config_bytes.len();
        let config_sha256 = sha256_digest(&config_bytes);
        if let Some(manifest) = &manifest {
            if config_length != manifest.config_length {
                return Err(PackedSafetensorsError::ConfigLengthMismatch {
                    expected: manifest.config_length,
                    actual: config_length,
                });
            }
            if config_sha256 != manifest.config_sha256 {
                return Err(PackedSafetensorsError::DigestMismatch {
                    file: CONFIG_FILE.to_string(),
                });
            }
        }

        let index_length = file_length(index_file.as_ref(), INDEX_FILE)?;
        enforce_limit(
            PackedLimitKind::IndexBytes,
            limits.index_bytes,
            index_length,
        )?;
        let index_pin = manifest.as_ref().map(|manifest| FilePin {
            length: index_length,
            sha256: manifest.index_sha256,
        });
        let index_bytes = admit_file(index_file.as_mut(), INDEX_FILE, index_pin)?;
        let index_sha256 = sha256_digest(&index_bytes);
        let index_json = parse_unique_json(&index_bytes, INDEX_FILE)?;
        let index_map = parse_weight_map(&index_json)?;
        let index_shards = index_map.values().cloned().collect::<BTreeSet<_>>();
        for shard in &index_shards {
            validate_relative_path(shard)?;
        }
        let manifest_shards = opened_shards.keys().cloned().collect::<BTreeSet<_>>();
        if index_shards != manifest_shards {
            return Err(PackedSafetensorsError::ShardSetMismatch {
                index: index_shards.into_iter().collect(),
                manifest: manifest_shards.into_iter().collect(),
            });
        }

        let mut shards = BTreeMap::new();
        let mut identity_shards = Vec::new();
        let mut tensors = BTreeMap::new();
        let mut header_bytes_total = 0usize;
        let mut header_bytes_max = 0usize;
        let mut tensor_count = 0usize;
        let initial_artifact_hash_bytes =
            checked_add(config_length, index_length, "initial_artifact_hash_bytes")?;

        for (filename, (shard_manifest, mut file)) in opened_shards {
            let actual_length = file_length(file.as_ref(), &filename)?;
            if let Some(shard_manifest) = &shard_manifest
                && actual_length != shard_manifest.file_length
            {
                return Err(PackedSafetensorsError::FileLengthMismatch {
                    file: filename,
                    expected: shard_manifest.file_length,
                    actual: actual_length,
                });
            }
            let (header_length, data_start) =
                read_header_length(file.as_mut(), &filename, actual_length)?;
            if let Some(shard_manifest) = &shard_manifest
                && header_length != shard_manifest.header_length
            {
                return Err(PackedSafetensorsError::HeaderLengthMismatch {
                    file: filename,
                    expected: shard_manifest.header_length,
                    actual: header_length,
                });
            }
            enforce_limit(
                PackedLimitKind::HeaderBytesPerShard,
                limits.header_bytes_per_shard,
                header_length,
            )?;
            let header_bytes = read_header(file.as_mut(), &filename, header_length)?;
            if let Some(shard_manifest) = &shard_manifest
                && sha256_digest(&header_bytes) != shard_manifest.header_sha256
            {
                return Err(PackedSafetensorsError::HeaderDigestMismatch { file: filename });
            }
            let data_length = actual_length.checked_sub(data_start).ok_or(
                PackedSafetensorsError::ArithmeticOverflow {
                    field: "shard data length",
                },
            )?;
            let header_json = parse_unique_json(&header_bytes, &filename)?;
            let header_tensor_count = header_json
                .as_object()
                .ok_or_else(|| PackedSafetensorsError::Json {
                    file: filename.clone(),
                    reason: "top level is not an object".to_string(),
                })?
                .keys()
                .filter(|name| name.as_str() != "__metadata__")
                .count();
            tensor_count = checked_add(tensor_count, header_tensor_count, "tensor entry count")?;
            enforce_limit(
                PackedLimitKind::TensorEntries,
                limits.tensor_entries,
                tensor_count,
            )?;
            let parsed = parse_header_entries(&header_json, &filename, data_length)?;
            for (name, entry) in parsed {
                if let Some(previous) = tensors.insert(name.clone(), entry.clone()) {
                    return Err(PackedSafetensorsError::DuplicateTensorAcrossShards {
                        tensor: name,
                        first_shard: previous.shard,
                        second_shard: entry.shard,
                    });
                }
            }
            validate_shard_overlaps(&filename, &tensors)?;
            header_bytes_total =
                checked_add(header_bytes_total, header_length, "header_bytes_total")?;
            header_bytes_max = header_bytes_max.max(header_length);
            identity_shards.push(PackedShardManifest {
                filename: filename.clone(),
                file_length: actual_length,
                file_sha256: shard_manifest
                    .as_ref()
                    .map(|shard_manifest| shard_manifest.file_sha256)
                    .unwrap_or(NO_PIN_DIGEST),
                header_length,
                header_sha256: sha256_digest(&header_bytes),
            });
            shards.insert(
                filename,
                ShardState {
                    manifest: shard_manifest,
                    file,
                    data_start,
                },
            );
        }

        validate_index_header_bijection(&index_map, &tensors)?;

        let identity_fields = ArtifactIdentityFields {
            repository,
            revision,
            config_length,
            config_sha256,
            index_length,
            index_sha256,
            shards: identity_shards,
        };
        let artifact_cache_identity = artifact_cache_identity(&identity_fields);
        let artifact_identity = AuthenticatedArtifactIdentity(Arc::new(identity_fields));
        let inventory = tensors
            .iter()
            .map(|(name, entry)| AuthenticatedTensorDescriptor {
                name: name.clone(),
                shard: entry.shard.clone(),
                span: entry.span,
                dtype: entry.dtype.clone(),
                shape: entry.shape.clone(),
            })
            .collect();
        Ok(Self {
            artifact_cache_identity,
            artifact_identity,
            config_bytes,
            inventory_session: Arc::new(InventorySession),
            inventory,
            #[cfg(test)]
            index_bytes,
            shards,
            tensors,
            authentication_report: PackedLoadReport {
                config_bytes: config_length,
                index_bytes: index_length,
                header_bytes_total,
                header_bytes_max,
                initial_artifact_hash_bytes,
                peak_io_buffer_bytes: HASH_BUFFER_BYTES,
                ..PackedLoadReport::default()
            },
            limits,
        })
    }

    /// Return the immutable identity authenticated for this held snapshot.
    pub const fn artifact_identity(&self) -> &AuthenticatedArtifactIdentity {
        &self.artifact_identity
    }

    /// Return the exact `config.json` bytes whose length and digest were verified for this held snapshot.
    /// Consumers must parse this buffer instead of reopening the snapshot path after authentication.
    pub fn config_bytes(&self) -> &[u8] {
        &self.config_bytes
    }

    /// Borrow the complete authenticated tensor inventory in deterministic name order.
    pub fn inventory(&self) -> AuthenticatedInventory<'_> {
        AuthenticatedInventory {
            session: &self.inventory_session,
            descriptors: &self.inventory,
        }
    }

    /// Classify and atomically load packed and standalone exact-source owners.
    pub fn load_mixed<E>(
        &mut self,
        packed_cache: &mut PackedOwnerCache,
        exact_cache: &mut ExactSourceOwnerCache,
        classifier: impl FnOnce(AuthenticatedInventory<'_>) -> Result<Vec<InventoryDecision>, E>,
    ) -> Result<MixedLoadResult, MixedLoadError<E>> {
        self.load_mixed_with_staging_observer(packed_cache, exact_cache, classifier, |_| {})
    }

    pub(crate) fn load_mixed_with_staging_observer<E>(
        &mut self,
        packed_cache: &mut PackedOwnerCache,
        exact_cache: &mut ExactSourceOwnerCache,
        classifier: impl FnOnce(AuthenticatedInventory<'_>) -> Result<Vec<InventoryDecision>, E>,
        observe_cold_owner: impl FnMut(StagedOwnerRef<'_>),
    ) -> Result<MixedLoadResult, MixedLoadError<E>> {
        self.load_mixed_with_observers(
            packed_cache,
            exact_cache,
            classifier,
            |_| {},
            |_| {},
            observe_cold_owner,
        )
    }

    pub(crate) fn load_mixed_with_observers<E>(
        &mut self,
        packed_cache: &mut PackedOwnerCache,
        exact_cache: &mut ExactSourceOwnerCache,
        classifier: impl FnOnce(AuthenticatedInventory<'_>) -> Result<Vec<InventoryDecision>, E>,
        mut observe_cache_probe: impl FnMut(OwnerCacheKind),
        mut observe_owner_construction: impl FnMut(OwnerCacheKind),
        mut observe_cold_owner: impl FnMut(StagedOwnerRef<'_>),
    ) -> Result<MixedLoadResult, MixedLoadError<E>> {
        let decisions = classifier(self.inventory()).map_err(MixedLoadError::Classifier)?;
        let validated = self.validate_mixed_decisions(decisions)?;

        let mut packed_report = self.authentication_report;
        report_selected_accounting(&mut packed_report, &validated.packed)?;
        let report = build_mixed_report(
            MixedReportInputs {
                authentication: self.authentication_report,
                accounting: validated.accounting.clone(),
                packed: &validated.packed,
                exact: &validated.exact,
                packed_cache,
                exact_cache,
                artifact_cache_identity: self.artifact_cache_identity,
                packed_report,
            },
            &mut observe_cache_probe,
        )?;
        let ValidatedMixedSelection {
            packed,
            exact,
            inventory,
            ..
        } = validated;
        let packed = {
            let mut packed_observer = |owner: &Arc<PackedPayload>| {
                let staged_owner = StagedOwnerRef::Packed(owner);
                let _ = staged_owner.strong_count();
                observe_cold_owner(staged_owner);
            };
            self.stage_packed(
                packed,
                packed_cache,
                &mut packed_report,
                &mut observe_cache_probe,
                &mut observe_owner_construction,
                &mut packed_observer,
            )?
        };
        debug_assert_eq!(packed_report.payload_builds, report.packed_payload_builds);
        let exact = self.stage_exact_sources(
            exact,
            exact_cache,
            &mut observe_cache_probe,
            &mut observe_owner_construction,
            &mut observe_cold_owner,
        )?;
        debug_assert_eq!(
            exact.iter().filter(|row| row.cold).count(),
            report.exact_source_builds.total()
        );
        let staged = StagedMixedSelection {
            packed,
            exact,
            inventory,
            report,
        };
        let prepared = prepare_mixed_publication(
            staged,
            self.artifact_identity.clone(),
            packed_cache,
            exact_cache,
        )?;
        let final_hash_io_bytes = self.verify_before_publication(&prepared.checks)?;
        Ok(commit_mixed_publication(
            prepared,
            final_hash_io_bytes,
            packed_cache,
            exact_cache,
        ))
    }

    /// Load a complete selection transaction into canonical owners.
    pub fn load(
        &mut self,
        selections: &[PackedSelectionRow],
        cache: &mut PackedOwnerCache,
    ) -> Result<PackedLoadResult, PackedSafetensorsError> {
        self.load_with_staging_observer(selections, cache, |_| {})
    }

    pub(crate) fn load_with_staging_observer(
        &mut self,
        selections: &[PackedSelectionRow],
        cache: &mut PackedOwnerCache,
        mut observe_cold_owner: impl FnMut(&Arc<PackedPayload>),
    ) -> Result<PackedLoadResult, PackedSafetensorsError> {
        let validated = self.validate_selections(selections)?;
        let mut report = self.authentication_report;
        report_selected_accounting(&mut report, &validated)?;
        let mut ignore_cache_probe = |_| {};
        let mut ignore_owner_construction = |_| {};

        let staged = self.stage_packed(
            validated,
            cache,
            &mut report,
            &mut ignore_cache_probe,
            &mut ignore_owner_construction,
            &mut observe_cold_owner,
        )?;
        let prepared = prepare_packed_publication(staged, report, cache)?;
        let final_verification_bytes = self.verify_before_publication(&prepared.checks)?;
        Ok(commit_packed_publication(
            prepared,
            final_verification_bytes,
            cache,
        ))
    }

    pub(crate) fn stage_packed(
        &mut self,
        validated: Vec<ValidatedSelection>,
        cache: &PackedOwnerCache,
        report: &mut PackedLoadReport,
        observe_cache_probe: &mut impl FnMut(OwnerCacheKind),
        observe_owner_construction: &mut impl FnMut(OwnerCacheKind),
        observe_cold_owner: &mut impl FnMut(&Arc<PackedPayload>),
    ) -> Result<Vec<StagedRow>, PackedSafetensorsError> {
        let mut staged = Vec::with_capacity(validated.len());
        let mut completed_payloads = 0usize;
        for row in validated {
            let key = row.cache_key(self.artifact_cache_identity);
            observe_cache_probe(OwnerCacheKind::Packed);
            if let Some(owner) = cache.owners.get(&key) {
                report.warm_reused_owner_bytes = checked_add(
                    report.warm_reused_owner_bytes,
                    row.selection.descriptor.total_source_bytes(),
                    "warm_reused_owner_bytes",
                )?;
                staged.push(StagedRow {
                    row,
                    key,
                    owner: Arc::clone(owner),
                    cold: false,
                });
                continue;
            }
            observe_owner_construction(OwnerCacheKind::Packed);

            let shard = self.shards.get_mut(&row.selection.shard).ok_or_else(|| {
                PackedSafetensorsError::MissingSelectionTensor {
                    linear_id: row.selection.linear_id.clone(),
                    tensor: row.selection.weight_name.clone(),
                }
            })?;
            let weight = read_source_arc(
                shard.file.as_mut(),
                shard.data_start,
                row.selection.weight_span,
            )
            .map_err(|error| PackedSafetensorsError::SelectedRead {
                linear_id: row.selection.linear_id.clone(),
                buffer: PackedBufferKind::Weight,
                completed_payloads,
                error,
            })?;
            let scale = read_source_arc(
                shard.file.as_mut(),
                shard.data_start,
                row.selection.scale_span,
            )
            .map_err(|error| PackedSafetensorsError::SelectedRead {
                linear_id: row.selection.linear_id.clone(),
                buffer: PackedBufferKind::Scale,
                completed_payloads,
                error,
            })?;
            let payload = PackedPayload::try_new(
                row.selection.descriptor,
                [
                    (SourceRole::Planar(OperandRole::Codes), weight),
                    (SourceRole::Planar(OperandRole::Scale), scale),
                ],
            )
            .map_err(|error| PackedSafetensorsError::PayloadBuild {
                linear_id: row.selection.linear_id.clone(),
                completed_payloads,
                error,
            })?;
            let owner = Arc::new(payload);
            observe_cold_owner(&owner);
            completed_payloads = checked_add(completed_payloads, 1, "completed_payloads")?;
            report.cold_new_owner_bytes = checked_add(
                report.cold_new_owner_bytes,
                row.selection.descriptor.total_source_bytes(),
                "cold_new_owner_bytes",
            )?;
            report.selected_range_read_bytes = checked_add(
                report.selected_range_read_bytes,
                row.selection.descriptor.total_source_bytes(),
                "selected_range_read_bytes",
            )?;
            report.payload_builds = checked_add(report.payload_builds, 1, "payload_builds")?;
            staged.push(StagedRow {
                row,
                key,
                owner,
                cold: true,
            });
        }
        Ok(staged)
    }

    pub(crate) fn validate_mixed_decisions(
        &self,
        decisions: Vec<InventoryDecision>,
    ) -> Result<ValidatedMixedSelection, PackedSafetensorsError> {
        let mut dispositions = vec![None; self.inventory.len()];
        for decision in decisions {
            let ordinal = decision.key.ordinal;
            if !Arc::ptr_eq(&decision.key.session, &self.inventory_session) {
                return Err(PackedSafetensorsError::ForeignInventoryKey { ordinal });
            }
            let slot = dispositions
                .get_mut(ordinal)
                .ok_or(PackedSafetensorsError::InvalidInventoryKey { ordinal })?;
            if slot.replace(decision.disposition).is_some() {
                return Err(PackedSafetensorsError::DuplicateInventoryKey { ordinal });
            }
        }
        for (ordinal, disposition) in dispositions.iter().enumerate() {
            if disposition.is_none() {
                return Err(PackedSafetensorsError::MissingInventoryDecision {
                    tensor: self.inventory[ordinal].name.clone(),
                });
            }
        }

        let dispositions = dispositions
            .into_iter()
            .map(|disposition| disposition.expect("complete inventory checked above"))
            .collect::<Vec<_>>();
        let mut pairs: BTreeMap<String, PackedDecisionPair> = BTreeMap::new();
        let mut exact = Vec::new();
        let mut inventory = Vec::with_capacity(self.inventory.len());
        let mut selected_names = HashSet::new();
        let mut selected_spans = HashSet::new();
        let mut accounting = MixedAccountingTable::default();

        for (ordinal, (descriptor, disposition)) in
            self.inventory.iter().zip(dispositions.iter()).enumerate()
        {
            let source_bytes = descriptor.span.end - descriptor.span.start;
            let category = category_for_disposition(disposition);
            accounting.checked_add(MixedByteQuantity::Source, category, source_bytes)?;
            inventory.push(ClassifiedInventoryRow {
                descriptor: descriptor.clone(),
                disposition: disposition.clone(),
            });
            if disposition_selects_source(disposition) {
                if !selected_names.insert(descriptor.name.clone()) {
                    return Err(PackedSafetensorsError::ReusedSourceName {
                        tensor: descriptor.name.clone(),
                    });
                }
                if !selected_spans.insert((descriptor.shard.clone(), descriptor.span)) {
                    return Err(PackedSafetensorsError::ReusedSourceSpan {
                        shard: descriptor.shard.clone(),
                        span: descriptor.span,
                    });
                }
            }
            match disposition {
                TensorDisposition::PackedWeight {
                    linear_id,
                    format,
                    logical_shape,
                } => {
                    let pair = pairs.entry(linear_id.clone()).or_default();
                    if pair
                        .weight
                        .replace(PackedWeightDecision {
                            ordinal,
                            format: *format,
                            logical_shape: *logical_shape,
                        })
                        .is_some()
                    {
                        return Err(PackedSafetensorsError::DuplicatePackedComponent {
                            linear_id: linear_id.clone(),
                            component: PackedBufferKind::Weight,
                        });
                    }
                    accounting.checked_add(
                        MixedByteQuantity::SelectedSource,
                        MixedSourceCategory::Packed,
                        source_bytes,
                    )?;
                }
                TensorDisposition::PackedScale { linear_id } => {
                    let pair = pairs.entry(linear_id.clone()).or_default();
                    if pair.scale.replace(ordinal).is_some() {
                        return Err(PackedSafetensorsError::DuplicatePackedComponent {
                            linear_id: linear_id.clone(),
                            component: PackedBufferKind::Scale,
                        });
                    }
                    accounting.checked_add(
                        MixedByteQuantity::SelectedSource,
                        MixedSourceCategory::Packed,
                        source_bytes,
                    )?;
                }
                TensorDisposition::StandaloneE4m3
                | TensorDisposition::DenseBf16
                | TensorDisposition::DenseF32
                | TensorDisposition::DenseI64 => {
                    let kind = exact_kind_for_disposition(disposition)
                        .expect("standalone dispositions have an exact source kind");
                    if descriptor.dtype != kind.dtype() {
                        return Err(PackedSafetensorsError::DispositionDtypeMismatch {
                            tensor: descriptor.name.clone(),
                            dtype: descriptor.dtype.clone(),
                            disposition: disposition.clone(),
                        });
                    }
                    accounting.checked_add(
                        MixedByteQuantity::SelectedSource,
                        category,
                        source_bytes,
                    )?;
                    let data_start = self
                        .shards
                        .get(&descriptor.shard)
                        .expect("authenticated inventory names an open shard")
                        .data_start;
                    exact.push(ValidatedExactSelection {
                        descriptor: descriptor.clone(),
                        kind,
                        data_start,
                        key: ExactSourceCacheKey {
                            artifact: self.artifact_identity.clone(),
                            descriptor: descriptor.clone(),
                            kind,
                        },
                    });
                }
                TensorDisposition::Deferred | TensorDisposition::Excluded => {}
            }
        }

        let mut packed_rows = Vec::with_capacity(pairs.len());
        for (linear_id, pair) in pairs {
            let weight =
                pair.weight
                    .ok_or_else(|| PackedSafetensorsError::MissingPackedComponent {
                        linear_id: linear_id.clone(),
                        component: PackedBufferKind::Weight,
                    })?;
            let scale_ordinal =
                pair.scale
                    .ok_or_else(|| PackedSafetensorsError::MissingPackedComponent {
                        linear_id: linear_id.clone(),
                        component: PackedBufferKind::Scale,
                    })?;
            let weight_source = &self.inventory[weight.ordinal];
            let scale_source = &self.inventory[scale_ordinal];
            if weight_source.shard != scale_source.shard {
                return Err(PackedSafetensorsError::SplitPair {
                    linear_id,
                    weight_shard: weight_source.shard.clone(),
                    scale_shard: scale_source.shard.clone(),
                });
            }
            let descriptor =
                PackedWeight::try_new(weight.format, weight.logical_shape).map_err(|error| {
                    PackedSafetensorsError::PackedDescriptor {
                        linear_id: linear_id.clone(),
                        error,
                    }
                })?;
            let weight_shape =
                shape_2(&linear_id, PackedSelectionField::WeightShape, weight_source)?;
            let scale_shape = shape_2(&linear_id, PackedSelectionField::ScaleShape, scale_source)?;
            packed_rows.push(PackedSelectionRow {
                linear_id,
                descriptor,
                weight_name: weight_source.name.clone(),
                scale_name: scale_source.name.clone(),
                shard: weight_source.shard.clone(),
                weight_span: weight_source.span,
                scale_span: scale_source.span,
                weight_dtype: weight_source.dtype.clone(),
                scale_dtype: scale_source.dtype.clone(),
                weight_shape,
                scale_shape,
            });
        }

        let packed = self.validate_selections(&packed_rows)?;
        let selected_total = accounting
            .summarize(MixedByteQuantity::SelectedSource)?
            .aggregate();
        enforce_limit(
            PackedLimitKind::SelectedSourceBytes,
            self.limits.selected_source_bytes,
            selected_total,
        )?;
        Ok(ValidatedMixedSelection {
            packed,
            exact,
            inventory,
            accounting,
        })
    }

    pub(crate) fn stage_exact_sources(
        &mut self,
        validated: Vec<ValidatedExactSelection>,
        cache: &ExactSourceOwnerCache,
        observe_cache_probe: &mut impl FnMut(OwnerCacheKind),
        observe_owner_construction: &mut impl FnMut(OwnerCacheKind),
        observe_cold_owner: &mut impl FnMut(StagedOwnerRef<'_>),
    ) -> Result<Vec<StagedExactRow>, PackedSafetensorsError> {
        let mut staged = Vec::with_capacity(validated.len());
        let mut completed_owners = 0usize;
        for row in validated {
            observe_cache_probe(OwnerCacheKind::ExactSource);
            if let Some(owner) = cache.owners.get(&row.key) {
                staged.push(StagedExactRow {
                    row,
                    owner: Arc::clone(owner),
                    cold: false,
                });
                continue;
            }
            observe_owner_construction(OwnerCacheKind::ExactSource);
            let shard = self
                .shards
                .get_mut(&row.descriptor.shard)
                .expect("authenticated exact source names an open shard");
            let bytes = read_source_arc(shard.file.as_mut(), row.data_start, row.descriptor.span)
                .map_err(|error| PackedSafetensorsError::ExactSourceRead {
                tensor: row.descriptor.name.clone(),
                completed_owners,
                error,
            })?;
            #[cfg(test)]
            let read_allocation = Arc::as_ptr(&bytes);
            let owner = Arc::new(ExactSourceOwner {
                artifact: self.artifact_identity.clone(),
                descriptor: row.descriptor.clone(),
                kind: row.kind,
                bytes,
            });
            let staged_owner = StagedOwnerRef::Exact {
                #[cfg(test)]
                read_allocation,
                owner: &owner,
            };
            #[cfg(test)]
            debug_assert!(staged_owner.preserves_read_allocation());
            let _ = staged_owner.strong_count();
            observe_cold_owner(staged_owner);
            completed_owners = checked_add(completed_owners, 1, "completed exact-source owners")?;
            staged.push(StagedExactRow {
                row,
                owner,
                cold: true,
            });
        }
        Ok(staged)
    }

    pub(crate) fn validate_selections(
        &self,
        selections: &[PackedSelectionRow],
    ) -> Result<Vec<ValidatedSelection>, PackedSafetensorsError> {
        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        let mut spans = HashSet::new();
        let mut selected_source_bytes = 0usize;
        let mut packed_source_bytes = 0usize;
        let mut out = Vec::with_capacity(selections.len());

        for selection in selections {
            if !ids.insert(selection.linear_id.clone()) {
                return Err(PackedSafetensorsError::DuplicateLinearId {
                    linear_id: selection.linear_id.clone(),
                });
            }
            if selection.weight_name == selection.scale_name
                || selection.weight_span == selection.scale_span
            {
                return Err(PackedSafetensorsError::AliasedPair {
                    linear_id: selection.linear_id.clone(),
                });
            }
            for name in [&selection.weight_name, &selection.scale_name] {
                if !names.insert(name.clone()) {
                    return Err(PackedSafetensorsError::ReusedSourceName {
                        tensor: name.clone(),
                    });
                }
            }
            for span in [selection.weight_span, selection.scale_span] {
                if !spans.insert((selection.shard.clone(), span)) {
                    return Err(PackedSafetensorsError::ReusedSourceSpan {
                        shard: selection.shard.clone(),
                        span,
                    });
                }
            }

            let weight = self.tensors.get(&selection.weight_name).ok_or_else(|| {
                PackedSafetensorsError::MissingSelectionTensor {
                    linear_id: selection.linear_id.clone(),
                    tensor: selection.weight_name.clone(),
                }
            })?;
            let scale = self.tensors.get(&selection.scale_name).ok_or_else(|| {
                PackedSafetensorsError::MissingSelectionTensor {
                    linear_id: selection.linear_id.clone(),
                    tensor: selection.scale_name.clone(),
                }
            })?;
            if weight.shard != scale.shard {
                return Err(PackedSafetensorsError::SplitPair {
                    linear_id: selection.linear_id.clone(),
                    weight_shard: weight.shard.clone(),
                    scale_shard: scale.shard.clone(),
                });
            }

            validate_selection_row(selection, weight, scale)?;
            selected_source_bytes = checked_add(
                selected_source_bytes,
                checked_add(
                    selection.weight_span.len()?,
                    selection.scale_span.len()?,
                    "selected row source bytes",
                )?,
                "selected source bytes",
            )?;
            enforce_limit(
                PackedLimitKind::SelectedSourceBytes,
                self.limits.selected_source_bytes,
                selected_source_bytes,
            )?;
            packed_source_bytes = checked_add(
                packed_source_bytes,
                selection.descriptor.total_source_bytes(),
                "packed source bytes",
            )?;
            enforce_limit(
                PackedLimitKind::PackedSourceBytes,
                self.limits.packed_source_bytes,
                packed_source_bytes,
            )?;
            let data_start = self
                .shards
                .get(&selection.shard)
                .ok_or_else(|| PackedSafetensorsError::SelectionMismatch {
                    linear_id: selection.linear_id.clone(),
                    field: PackedSelectionField::Shard,
                    expected: selection.shard.clone(),
                    actual: "<missing authenticated shard>".to_string(),
                })?
                .data_start;
            out.push(ValidatedSelection {
                selection: selection.clone(),
                data_start,
            });
        }
        Ok(out)
    }

    /// The one remaining full-content pass over each shard (card 544): config and index were
    /// already admitted once, fully, in [`Self::authenticate_retained_config_bytes`] and never
    /// reread from disk, so re-verifying them here would be the exact second hash R475-014 named.
    /// A shard streams here at most once - hashed against its pin when the manifest names one,
    /// and always cross-checked against `checks` (the weight/scale spans this shard staged
    /// between authentication and now), so a race is still caught even with no pin.
    pub(crate) fn verify_before_publication(
        &mut self,
        checks: &BTreeMap<String, Vec<RangeCheck>>,
    ) -> Result<usize, PackedSafetensorsError> {
        let mut verified = 0usize;
        for (filename, shard) in &mut self.shards {
            let shard_checks = checks.get(filename).map(Vec::as_slice).unwrap_or(&[]);
            let pin = shard.manifest.as_ref().map(|manifest| FilePin {
                length: manifest.file_length,
                sha256: manifest.file_sha256,
            });
            verified = checked_add(
                verified,
                admit_shard(shard.file.as_mut(), filename, pin, shard_checks)?,
                "final_verification_bytes",
            )?;
        }
        Ok(verified)
    }
}
