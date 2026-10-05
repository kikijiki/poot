#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors: {0}")]
    SafeTensors(String),
    #[error("unrecognized checkpoint dtype {name:?}")]
    UnknownCheckpointDtype { name: String },
    #[error("checkpoint dtype {name:?} has no f32 decode")]
    UndecodableCheckpointDtype { name: String },
    #[error("{prefix}: FP8 block-wise scale is ambiguous because both {first} and {second} exist")]
    Fp8AmbiguousScale {
        prefix: String,
        first: String,
        second: String,
    },
    #[error("duplicate safetensors tensor name {name:?} {scope}")]
    DuplicateTensorName { name: String, scope: String },
    #[error("config has no rope_theta and model_type {model_type:?} has no reference default")]
    MissingRopeTheta { model_type: String },
    #[error(
        "config rope_theta {value} for model_type {model_type:?} is not a positive finite number"
    )]
    InvalidRopeTheta { model_type: String, value: f32 },
    #[error("config has no eos_token_id and model_type {model_type:?} has no reference default")]
    MissingEosTokenId { model_type: String },
    /// [`crate::gguf::weight_format_of`] has no [`poot_quant::format::WeightFormat`] for this ggml
    /// type: a removed legacy id, or a real ggml scheme with no packed path (ADR-0103 decision 2).
    #[error("gguf tensor {tensor:?}: unsupported ggml type {ggml_type}")]
    UnsupportedGgufType { tensor: String, ggml_type: u32 },
    /// [`crate::gguf::read_gguf`] found a quantized tensor of a rank it has no owner shape for
    /// (2 for a plain weight, 3 for stacked experts).
    #[error("gguf tensor {tensor:?}: unsupported rank {rank} for a quantized tensor")]
    UnsupportedGgufTensorRank { tensor: String, rank: usize },
    /// A GGUF tensor's bytes end before its declared shape does.
    #[error("gguf tensor data truncated: {need} bytes needed, {have} present")]
    GgufTensorTruncated { need: usize, have: usize },
    #[error("gguf packed weight: {0}")]
    GgufPackedWeight(#[from] poot_quant::PackedWeightError),
    #[error("gguf dense weight: {0}")]
    GgufDenseWeight(#[from] poot_quant::weights::DenseWeightError),
    #[error("gguf weight store: {0}")]
    GgufDuplicateWeightKey(#[from] poot_quant::weights::DuplicateWeightKey),
    /// A quantized safetensors linear whose tensors do not form the packed weight its scheme
    /// describes (card 545a): a missing or mistyped part, or a shape the format cannot hold.
    #[error("quantized linear tensor {tensor}: {reason}")]
    QuantizedLinearLayout { tensor: String, reason: String },
    /// A `QuantScheme` config field (from the checkpoint's `quantization_config`, not any one
    /// tensor) fails its own precondition (card 545b): `field` names the config
    /// key, `value` the rejected value it carried.
    #[error("quant config {field} = {value}: invalid")]
    InvalidQuantConfig { field: &'static str, value: usize },
    /// A quantized safetensors linear's bytes refused by the packed payload's content checks
    /// (lengths, finiteness, group-index range).
    #[error("quantized linear {linear}: {source}")]
    QuantizedLinearPayload {
        /// The packed weight's key (`{prefix}.weight`); `source` names the offending role.
        linear: String,
        #[source]
        source: poot_quant::PackedWeightError,
    },
}
