//! The loader's one error type, [`LoadError`]: every refusal names the file, tensor or config field that
//! produced it.

/// Why loading a checkpoint failed: reading or mapping a file, parsing its container or config, or a tensor that cannot become a weight.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// Reading a checkpoint file or directory failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A checkpoint file could not be mapped (not a regular file, or the mapping failed).
    #[error("map: {0}")]
    Map(#[from] poot_quant::MapError),
    /// A mapped checkpoint file was modified in place, or replaced under the descriptor, while it was
    /// being loaded (ADR-0103 amendment): the validated bytes are no longer the file's bytes.
    #[error("{0}")]
    SourceChanged(#[from] poot_quant::SourceChanged),
    /// A JSON document (`config.json`, a safetensors header, an index) did not parse.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// The safetensors container or a tensor's metadata is malformed; the message names the defect.
    #[error("safetensors: {0}")]
    SafeTensors(String),
    /// A safetensors header names a dtype this loader has no `DType` for.
    #[error("unrecognized checkpoint dtype {name:?}")]
    UnknownCheckpointDtype {
        /// The dtype string as the file spells it.
        name: String,
    },
    /// A LoRA adapter factor is stored in a dtype the adapter path does not carry: only F32, F16 and BF16
    /// factors are read, as stored.
    #[error("lora adapter tensor {tensor}: dtype {dtype:?} is not F32, F16 or BF16")]
    LoraFactorDtype {
        /// The adapter tensor's name.
        tensor: String,
        /// Its stored dtype.
        dtype: poot_tensor::DType,
    },
    /// An FP8 block-wise linear has two scale tensors that could both apply, so which one is meant is ambiguous.
    #[error("{prefix}: FP8 block-wise scale is ambiguous because both {first} and {second} exist")]
    Fp8AmbiguousScale {
        /// The linear's tensor-name prefix.
        prefix: String,
        /// One scale tensor name that exists.
        first: String,
        /// The other scale tensor name that exists.
        second: String,
    },
    /// Two tensors in one checkpoint share a name.
    #[error("duplicate tensor name {name:?} {scope}")]
    DuplicateTensorName {
        /// The repeated tensor name.
        name: String,
        /// Where the repeat was found (`"within one header"`, `"across model shards"` or `"within one GGUF directory"`).
        scope: String,
    },
    /// A GGUF header lists one metadata key twice, so which value is meant is ambiguous.
    #[error("duplicate gguf metadata key {key:?}")]
    DuplicateGgufMetadataKey {
        /// The repeated key.
        key: String,
    },
    /// `general.alignment` is not a non-zero power of two (or not an integer), so the tensor data
    /// start cannot be computed.
    #[error("gguf general.alignment = {value}: must be a non-zero power of two")]
    InvalidGgufAlignment {
        /// The rejected value, as the file spells it.
        value: String,
    },
    /// A shard index names a file that is not a bare file name inside the model directory.
    #[error("shard index names {name:?}: a shard must be a plain file name in the model directory")]
    UnsafeShardName {
        /// The shard name as the index spells it.
        name: String,
    },
    /// A `quantization_config` field names a variant poot has no decoder for (a silent decode with
    /// the wrong convention is worse than a refusal): `field` is the config key, `value` what it carried.
    #[error("quant config {field} = {value}: unsupported")]
    UnsupportedQuantConfig {
        /// The `quantization_config` key.
        field: &'static str,
        /// The rejected value.
        value: String,
    },
    /// An `adapter_config.json` field asks for behaviour the adapter loader does not apply, so loading
    /// the adapter would silently ignore it.
    #[error("adapter_config.json {field}: unsupported, per-module overrides are not applied")]
    UnsupportedLoraConfig {
        /// The `adapter_config.json` key.
        field: &'static str,
    },
    /// [`crate::gguf::weight_format_of`] has no [`poot_quant::format::WeightFormat`] for this ggml
    /// type: a removed legacy id, or a real ggml scheme with no packed path (ADR-0103 decision 2).
    #[error("gguf tensor {tensor:?}: unsupported ggml type {ggml_type}")]
    UnsupportedGgufType {
        /// The GGUF tensor name.
        tensor: String,
        /// The ggml type id the file declares.
        ggml_type: u32,
    },
    /// [`crate::gguf::read_gguf`] found a quantized tensor of a rank it has no owner shape for
    /// (2 for a plain weight, 3 for stacked experts).
    #[error("gguf tensor {tensor:?}: unsupported rank {rank} for a quantized tensor")]
    UnsupportedGgufTensorRank {
        /// The GGUF tensor name.
        tensor: String,
        /// The tensor's rank.
        rank: usize,
    },
    /// A GGUF tensor's bytes end before its declared shape does.
    #[error("gguf tensor data truncated: {need} bytes needed, {have} present")]
    GgufTensorTruncated {
        /// Bytes the declared shape needs.
        need: usize,
        /// Bytes the file holds from the tensor's offset.
        have: usize,
    },
    /// The shards given for a split GGUF checkpoint are not the set its metadata declares, or the
    /// directory lacks tensors the metadata counts.
    #[error("gguf split checkpoint: {0}")]
    GgufSplit(#[from] GgufSplitError),
    /// A planned job's byte range lies outside its checkpoint file (a plan bug: each reader checks its ranges
    /// against the container before planning, so this is the pipeline's own guard).
    #[error("tensor {tensor:?}: bytes {need} needed, file holds {have}")]
    TensorOutsideFile {
        /// Name of the tensor whose range is outside the file.
        tensor: String,
        /// Bytes of file the planned range requires.
        need: usize,
        /// Bytes the file holds.
        have: usize,
    },
    /// A GGUF quantized tensor's bytes were refused as a packed weight.
    #[error("gguf packed weight: {0}")]
    GgufPackedWeight(#[from] poot_quant::PackedWeightError),
    /// A GGUF dense tensor's bytes were refused as a dense weight.
    #[error("gguf dense weight: {0}")]
    GgufDenseWeight(#[from] poot_quant::weights::DenseWeightError),
    /// Two GGUF tensors resolved to one weight-store key.
    #[error("gguf weight store: {0}")]
    GgufDuplicateWeightKey(#[from] poot_quant::weights::DuplicateWeightKey),
    /// A quantized safetensors linear whose tensors do not form the packed weight its scheme
    /// describes (card 545a): a missing or mistyped part, or a shape the format cannot hold.
    #[error("quantized linear tensor {tensor}: {reason}")]
    QuantizedLinearLayout {
        /// The linear's tensor name or prefix.
        tensor: String,
        /// What is missing, mistyped or mis-shaped.
        reason: String,
    },
    /// A `QuantScheme` config field (from the checkpoint's `quantization_config`, not any one
    /// tensor) fails its own precondition (card 545b): `field` names the config
    /// key, `value` the rejected value it carried.
    #[error("quant config {field} = {value}: invalid")]
    InvalidQuantConfig {
        /// The `quantization_config` key.
        field: &'static str,
        /// The rejected value.
        value: usize,
    },
    /// A quantized safetensors linear's bytes refused by the packed payload's content checks
    /// (lengths, finiteness, group-index range).
    #[error("quantized linear {linear}: {source}")]
    QuantizedLinearPayload {
        /// The packed weight's key (`{prefix}.weight`); `source` names the offending role.
        linear: String,
        /// The packed payload's content-check refusal.
        #[source]
        source: poot_quant::PackedWeightError,
    },
}

/// Why a set of GGUF shards is not the split checkpoint it claims to be.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GgufSplitError {
    /// No shard was given.
    #[error("no shard given")]
    NoShards,
    /// The number of shards given is not the `split.count` the first shard declares.
    #[error("{given} shards given, the first declares split.count = {declared:?}")]
    Count {
        /// `split.count` of the first shard, if it has one.
        declared: Option<usize>,
        /// How many shards were given.
        given: usize,
    },
    /// A shard is not in the position its `split.no` declares.
    #[error("shard at position {position} declares split.no = {declared:?}")]
    Order {
        /// The shard's position in the set given.
        position: usize,
        /// Its `split.no`, if it has one.
        declared: Option<u64>,
    },
    /// One tensor name is listed by two shards.
    #[error("tensor {tensor:?} is listed by shards {} and {}", shards.0, shards.1)]
    DuplicateTensor {
        /// The tensor name.
        tensor: String,
        /// The two shard positions that list it.
        shards: (usize, usize),
    },
    /// The directory lists fewer tensors than `split.tensors.count` declares: shards are missing.
    #[error(
        "the shards list {found} tensors, the checkpoint declares split.tensors.count = {declared}"
    )]
    Incomplete {
        /// `split.tensors.count`.
        declared: usize,
        /// Tensors the given shards list.
        found: usize,
    },
}
