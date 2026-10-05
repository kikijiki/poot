//! Model-local parsing for the official MiniMax-M2.5 config and checkpoint index.
//!
//! Metadata only: no safetensors shard is opened, no E4M3FN value decoded, and no dense
//! compatibility tensor allocated, so the no-full-f32 invariant in spec 349 holds.

#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use poot_quant::format::{ScaleEncoding, WeightFormat};
use poot_quant::{OperandRole, PackedWeight, PackedWeightError, SourceRole};

mod manifest;

pub use manifest::{
    MiniMaxM2Manifest, MiniMaxM2ManifestEntry, MiniMaxM2ManifestError, MiniMaxM2ManifestReport,
    MiniMaxM2Namespace,
};

pub const MINIMAX_M25_MODEL_TYPE: &str = "minimax_m2";
pub const MINIMAX_M25_ARCHITECTURE: &str = "MiniMaxM2ForCausalLM";

/// The packed format every MiniMax-M2 quantized linear uses: E4M3 weights with little-endian F32
/// inverse scales over 128-by-128 blocks.
///
/// A constant because [`MiniMaxM2Config`] rejects any `quantization_config` other than
/// `quant_method=fp8`, `fmt=float8_e4m3fn`, `weight_block_size=[128, 128]`.
pub const MINIMAX_M25_PACKED_FORMAT: WeightFormat = WeightFormat::E4m3Block128 {
    scale: ScaleEncoding::F32,
};

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM2QuantConfig {
    pub activation_scheme: String,
    pub format: String,
    pub quant_method: String,
    pub weight_block_size: [usize; 2],
    pub modules_to_not_convert: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub max_position_embeddings: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub use_cache: bool,
    pub use_mtp: bool,
    pub num_mtp_modules: usize,
    pub mtp_transformer_layers: usize,
    pub quantization: MiniMaxM2QuantConfig,
}

impl MiniMaxM2Config {
    pub fn from_slice(json: &[u8]) -> Result<Self, MiniMaxM2ConfigError> {
        let raw: RawMiniMaxM2Config = serde_json::from_slice(json)?;
        raw.try_into()
    }

    /// Parse the exact MiniMax-M2.5 profile pinned by card 349; other `minimax_m2` members are rejected.
    #[cfg(test)]
    pub(crate) fn from_m25_slice(json: &[u8]) -> Result<Self, MiniMaxM2ConfigError> {
        let config = Self::from_slice(json)?;
        for (field, actual, expected) in [
            ("vocab_size", config.vocab_size, 200_064),
            ("hidden_size", config.hidden_size, 3_072),
            ("intermediate_size", config.intermediate_size, 1_536),
            ("num_hidden_layers", config.num_hidden_layers, 62),
            ("num_attention_heads", config.num_attention_heads, 48),
            ("num_key_value_heads", config.num_key_value_heads, 8),
            ("head_dim", config.head_dim, 128),
            ("rotary_dim", config.rotary_dim, 64),
            (
                "max_position_embeddings",
                config.max_position_embeddings,
                196_608,
            ),
            ("num_local_experts", config.num_local_experts, 256),
            ("num_experts_per_tok", config.num_experts_per_tok, 8),
            ("num_mtp_modules", config.num_mtp_modules, 3),
            ("mtp_transformer_layers", config.mtp_transformer_layers, 1),
        ] {
            if actual != expected {
                return Err(unsupported(field, expected.to_string(), actual.to_string()));
            }
        }
        if config.rms_norm_eps != 1e-6 {
            return Err(unsupported(
                "rms_norm_eps",
                "0.000001",
                config.rms_norm_eps.to_string(),
            ));
        }
        if config.rope_theta != 5_000_000.0 {
            return Err(unsupported(
                "rope_theta",
                "5000000",
                config.rope_theta.to_string(),
            ));
        }
        if !config.use_cache || !config.use_mtp {
            return Err(unsupported(
                "cache/MTP flags",
                "use_cache=true and use_mtp=true",
                format!("use_cache={}, use_mtp={}", config.use_cache, config.use_mtp),
            ));
        }
        Ok(config)
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn attention_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    #[cfg(test)]
    pub(crate) fn fp8_scale_shape(&self, out_dim: usize, in_dim: usize) -> [usize; 2] {
        let [block_out, block_in] = self.quantization.weight_block_size;
        [out_dim.div_ceil(block_out), in_dim.div_ceil(block_in)]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MiniMaxM2ConfigError {
    #[error("MiniMax-M2 config json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported MiniMax-M2 {field}: expected {expected}, got {actual}")]
    Unsupported {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("invalid MiniMax-M2 {field}: {reason}")]
    Invalid { field: &'static str, reason: String },
}

fn unsupported(
    field: &'static str,
    expected: impl Into<String>,
    actual: impl Into<String>,
) -> MiniMaxM2ConfigError {
    MiniMaxM2ConfigError::Unsupported {
        field,
        expected: expected.into(),
        actual: actual.into(),
    }
}

fn invalid(field: &'static str, reason: impl Into<String>) -> MiniMaxM2ConfigError {
    MiniMaxM2ConfigError::Invalid {
        field,
        reason: reason.into(),
    }
}

#[derive(serde::Deserialize)]
struct RawMiniMaxM2Config {
    architectures: Vec<String>,
    attn_type_list: Vec<u8>,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    model_type: String,
    mtp_transformer_layers: usize,
    num_attention_heads: usize,
    num_experts_per_tok: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    num_local_experts: usize,
    num_mtp_modules: usize,
    qk_norm_type: String,
    quantization_config: RawMiniMaxM2QuantConfig,
    rms_norm_eps: f32,
    rope_theta: f32,
    rotary_dim: usize,
    scoring_func: String,
    shared_intermediate_size: usize,
    tie_word_embeddings: bool,
    use_cache: bool,
    use_mtp: bool,
    use_qk_norm: bool,
    use_routing_bias: bool,
    vocab_size: usize,
    #[serde(default)]
    sliding_window: Option<usize>,
}

#[derive(serde::Deserialize)]
struct RawMiniMaxM2QuantConfig {
    activation_scheme: String,
    fmt: String,
    quant_method: String,
    weight_block_size: [usize; 2],
    modules_to_not_convert: Vec<String>,
}

impl TryFrom<RawMiniMaxM2Config> for MiniMaxM2Config {
    type Error = MiniMaxM2ConfigError;

    fn try_from(raw: RawMiniMaxM2Config) -> Result<Self, Self::Error> {
        if raw.model_type != MINIMAX_M25_MODEL_TYPE {
            return Err(unsupported(
                "model_type",
                MINIMAX_M25_MODEL_TYPE,
                raw.model_type,
            ));
        }
        if raw.architectures.len() != 1 || raw.architectures[0] != MINIMAX_M25_ARCHITECTURE {
            return Err(unsupported(
                "architectures",
                format!("[{MINIMAX_M25_ARCHITECTURE:?}]"),
                format!("{:?}", raw.architectures),
            ));
        }
        if raw.hidden_act != "silu" {
            return Err(unsupported("hidden_act", "silu", raw.hidden_act));
        }
        if raw.sliding_window.is_some() {
            return Err(unsupported(
                "sliding_window",
                "null/full causal attention",
                format!("{:?}", raw.sliding_window),
            ));
        }
        if raw.qk_norm_type != "per_layer" || !raw.use_qk_norm {
            return Err(unsupported(
                "QK normalization",
                "qk_norm_type=per_layer and use_qk_norm=true",
                format!(
                    "qk_norm_type={:?}, use_qk_norm={}",
                    raw.qk_norm_type, raw.use_qk_norm
                ),
            ));
        }
        if raw.scoring_func != "sigmoid" || !raw.use_routing_bias {
            return Err(unsupported(
                "router",
                "scoring_func=sigmoid and use_routing_bias=true",
                format!(
                    "scoring_func={:?}, use_routing_bias={}",
                    raw.scoring_func, raw.use_routing_bias
                ),
            ));
        }
        if raw.shared_intermediate_size != 0 {
            return Err(unsupported(
                "shared_intermediate_size",
                "0",
                raw.shared_intermediate_size.to_string(),
            ));
        }
        if raw.tie_word_embeddings {
            return Err(unsupported("tie_word_embeddings", "false", "true"));
        }

        for (field, value) in [
            ("vocab_size", raw.vocab_size),
            ("hidden_size", raw.hidden_size),
            ("intermediate_size", raw.intermediate_size),
            ("num_hidden_layers", raw.num_hidden_layers),
            ("num_attention_heads", raw.num_attention_heads),
            ("num_key_value_heads", raw.num_key_value_heads),
            ("head_dim", raw.head_dim),
            ("rotary_dim", raw.rotary_dim),
            ("max_position_embeddings", raw.max_position_embeddings),
            ("num_local_experts", raw.num_local_experts),
            ("num_experts_per_tok", raw.num_experts_per_tok),
        ] {
            if value == 0 {
                return Err(invalid(field, "must be positive"));
            }
        }
        if !raw
            .num_attention_heads
            .is_multiple_of(raw.num_key_value_heads)
        {
            return Err(invalid(
                "num_key_value_heads",
                "must divide num_attention_heads",
            ));
        }
        if raw.num_experts_per_tok > raw.num_local_experts {
            return Err(invalid(
                "num_experts_per_tok",
                "must not exceed num_local_experts",
            ));
        }
        if raw.rotary_dim > raw.head_dim || !raw.rotary_dim.is_multiple_of(2) {
            return Err(invalid(
                "rotary_dim",
                "must be even and no larger than head_dim",
            ));
        }
        if raw.attn_type_list.len() != raw.num_hidden_layers
            || raw.attn_type_list.iter().any(|&kind| kind != 1)
        {
            return Err(unsupported(
                "attn_type_list",
                format!("{} entries, all 1", raw.num_hidden_layers),
                format!("{:?}", raw.attn_type_list),
            ));
        }
        if !raw.rms_norm_eps.is_finite() || raw.rms_norm_eps <= 0.0 {
            return Err(invalid("rms_norm_eps", "must be finite and positive"));
        }
        if !raw.rope_theta.is_finite() || raw.rope_theta <= 0.0 {
            return Err(invalid("rope_theta", "must be finite and positive"));
        }

        let quant = raw.quantization_config;
        if quant.quant_method != "fp8"
            || quant.fmt != "float8_e4m3fn"
            || quant.activation_scheme != "dynamic"
        {
            return Err(unsupported(
                "quantization_config",
                "quant_method=fp8, fmt=float8_e4m3fn, activation_scheme=dynamic",
                format!(
                    "quant_method={:?}, fmt={:?}, activation_scheme={:?}",
                    quant.quant_method, quant.fmt, quant.activation_scheme
                ),
            ));
        }
        if quant.weight_block_size != [128, 128] {
            return Err(unsupported(
                "quantization_config.weight_block_size",
                "[128, 128]",
                format!("{:?}", quant.weight_block_size),
            ));
        }
        let expected_exclusions = BTreeSet::from([
            "e_score_correction_bias".to_string(),
            "gate".to_string(),
            "lm_head".to_string(),
        ]);
        let actual_exclusions = quant
            .modules_to_not_convert
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if quant.modules_to_not_convert.len() != expected_exclusions.len()
            || actual_exclusions != expected_exclusions
        {
            return Err(unsupported(
                "quantization_config.modules_to_not_convert",
                format!("{:?}", expected_exclusions),
                format!("{:?}", quant.modules_to_not_convert),
            ));
        }

        Ok(Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rotary_dim: raw.rotary_dim,
            max_position_embeddings: raw.max_position_embeddings,
            num_local_experts: raw.num_local_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw.rope_theta,
            use_cache: raw.use_cache,
            use_mtp: raw.use_mtp,
            num_mtp_modules: raw.num_mtp_modules,
            mtp_transformer_layers: raw.mtp_transformer_layers,
            quantization: MiniMaxM2QuantConfig {
                activation_scheme: quant.activation_scheme,
                format: quant.fmt,
                quant_method: quant.quant_method,
                weight_block_size: quant.weight_block_size,
                modules_to_not_convert: quant.modules_to_not_convert,
            },
        })
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MiniMaxM2CheckpointIndex {
    pub declared_total_size: u64,
    pub tensor_count: usize,
    pub referenced_shards: Vec<String>,
    pub declared_shard_count: usize,
    pub missing_declared_shard_ordinals: Vec<usize>,
}

#[cfg(test)]
impl MiniMaxM2CheckpointIndex {
    pub(crate) fn from_slice(
        json: &[u8],
        config: &MiniMaxM2Config,
    ) -> Result<Self, MiniMaxM2IndexError> {
        let raw: RawCheckpointIndex = serde_json::from_slice(json)?;
        let actual_names = raw.weight_map.keys().cloned().collect::<BTreeSet<_>>();
        let expected_names = MiniMaxM2SourceTable::new(config)?.names();
        if let Some(name) = expected_names.difference(&actual_names).next() {
            return Err(MiniMaxM2IndexError::MissingTensor(name.clone()));
        }
        if let Some(name) = actual_names.difference(&expected_names).next() {
            return Err(MiniMaxM2IndexError::UnexpectedTensor(name.clone()));
        }

        let referenced_shards = raw
            .weight_map
            .values()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if referenced_shards.is_empty() {
            return Err(MiniMaxM2IndexError::NoShards);
        }

        let mut declared_shard_count = None;
        let mut referenced_ordinals = BTreeSet::new();
        for shard in &referenced_shards {
            let (ordinal, total) = parse_shard_name(shard)?;
            if let Some(expected) = declared_shard_count {
                if total != expected {
                    return Err(MiniMaxM2IndexError::InconsistentShardCount {
                        shard: shard.clone(),
                        expected,
                        actual: total,
                    });
                }
            } else {
                declared_shard_count = Some(total);
            }
            referenced_ordinals.insert(ordinal);
        }
        let declared_shard_count = declared_shard_count.expect("nonempty shard set checked above");
        let missing_declared_shard_ordinals = (0..declared_shard_count)
            .filter(|ordinal| !referenced_ordinals.contains(ordinal))
            .collect();

        Ok(Self {
            declared_total_size: raw.metadata.total_size,
            tensor_count: actual_names.len(),
            referenced_shards,
            declared_shard_count,
            missing_declared_shard_ordinals,
        })
    }
}

#[cfg(test)]
#[derive(Debug, thiserror::Error)]
pub(crate) enum MiniMaxM2IndexError {
    #[error("MiniMax-M2 checkpoint index json: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    SourceTable(#[from] MiniMaxM2SourceTableError),
    #[error("MiniMax-M2 checkpoint index is missing tensor {0}")]
    MissingTensor(String),
    #[error("MiniMax-M2 checkpoint index has unexpected tensor {0}")]
    UnexpectedTensor(String),
    #[error("MiniMax-M2 checkpoint index references no shards")]
    NoShards,
    #[error("MiniMax-M2 checkpoint index has malformed shard name {0}")]
    MalformedShardName(String),
    #[error("MiniMax-M2 checkpoint shard {shard} declares {actual} shards, expected {expected}")]
    InconsistentShardCount {
        shard: String,
        expected: usize,
        actual: usize,
    },
}

#[cfg(test)]
#[derive(serde::Deserialize)]
struct RawCheckpointIndex {
    metadata: RawCheckpointMetadata,
    weight_map: BTreeMap<String, String>,
}

#[cfg(test)]
#[derive(serde::Deserialize)]
struct RawCheckpointMetadata {
    total_size: u64,
}

#[cfg(test)]
fn parse_shard_name(name: &str) -> Result<(usize, usize), MiniMaxM2IndexError> {
    let malformed = || MiniMaxM2IndexError::MalformedShardName(name.to_string());
    let body = name
        .strip_prefix("model-")
        .and_then(|name| name.strip_suffix(".safetensors"))
        .ok_or_else(malformed)?;
    let (ordinal, total) = body.split_once("-of-").ok_or_else(malformed)?;
    if ordinal.len() != 5
        || total.len() != 5
        || !ordinal.bytes().all(|byte| byte.is_ascii_digit())
        || !total.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(malformed());
    }
    let ordinal = ordinal.parse::<usize>().map_err(|_| malformed())?;
    let total = total.parse::<usize>().map_err(|_| malformed())?;
    if total == 0 || ordinal >= total {
        return Err(malformed());
    }
    Ok((ordinal, total))
}

/// The dtype one intentionally dense MiniMax-M2 checkpoint tensor carries.
///
/// The pinned `quantization_config.modules_to_not_convert` is
/// `["gate", "e_score_correction_bias", "lm_head"]`; the audited headers add the embedding and every
/// norm. Router rows are F32, the rest BF16; none is quantized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MiniMaxM2DenseRole {
    /// Token embedding, the untied LM head, and every RMSNorm weight.
    Bf16,
    /// Router gate weights and their selection-only correction biases.
    F32,
}

impl MiniMaxM2DenseRole {
    /// The checkpoint dtype name this role stores, for error messages and manifest rows.
    pub const fn source_dtype(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }
}

/// One intentionally dense source: its checkpoint name, what it is, and its logical shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MiniMaxM2DenseSource {
    name: String,
    role: MiniMaxM2DenseRole,
    shape: Vec<usize>,
}

impl MiniMaxM2DenseSource {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn role(&self) -> MiniMaxM2DenseRole {
        self.role
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Logical elements this row stores. Every MiniMax dense shape is rank one or two, so the
    /// product cannot overflow a `usize` on any supported target.
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// One packed source: the linear it belongs to, and the checked layout of its E4M3 weight and
/// little-endian F32 inverse-scale pair.
///
/// The checkpoint spells the components `{linear_id}.weight` and `{linear_id}.weight_scale_inv`,
/// a different namespace from the graph names owned by `poot_graph_ir::PackedSourceName` (card 379).
/// This row is where the two meet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MiniMaxM2PackedSource {
    linear_id: String,
    descriptor: PackedWeight,
}

impl MiniMaxM2PackedSource {
    pub fn linear_id(&self) -> &str {
        &self.linear_id
    }

    pub const fn descriptor(&self) -> PackedWeight {
        self.descriptor
    }

    /// This component's checkpoint name. The graph name is
    /// `poot_graph_ir::PackedSourceName::new(self.linear_id(), component.source_role())`.
    pub(crate) fn checkpoint_name(&self, component: MiniMaxM2PackedComponent) -> String {
        format!("{}{}", self.linear_id, component.checkpoint_suffix())
    }

    /// Both checkpoint names of this linear, weight first, in the order card 379 pairs them.
    pub(crate) fn checkpoint_names(&self) -> [String; 2] {
        [
            MiniMaxM2PackedComponent::Weight,
            MiniMaxM2PackedComponent::Scale,
        ]
        .map(|component| self.checkpoint_name(component))
    }
}

/// One component of a MiniMax-M2 packed linear: the checkpoint (`{linear_id}.weight` /
/// `{linear_id}.weight_scale_inv`) and the graph namespace (card 379's `PackedSourceName`) agree on
/// exactly these two per packed source, never any other [`SourceRole`]. `poot-load` cannot name
/// `PackedSourceName` itself (`poot-graph-ir` is not a dependency, by spec 365's delivery split), so
/// this is the type the bridge is stated in; [`Self::source_role`] recovers the shared role for a
/// caller that names `PackedSourceName`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MiniMaxM2PackedComponent {
    Weight,
    Scale,
}

impl MiniMaxM2PackedComponent {
    /// The dotted suffix the MiniMax-M2 checkpoint appends to a linear id for this component.
    pub(crate) const fn checkpoint_suffix(self) -> &'static str {
        match self {
            Self::Weight => ".weight",
            Self::Scale => ".weight_scale_inv",
        }
    }

    /// The other component of the same packed linear.
    pub(crate) const fn other(self) -> Self {
        match self {
            Self::Weight => Self::Scale,
            Self::Scale => Self::Weight,
        }
    }

    /// The shared graph-namespace role (card 379's `PackedSourceName`).
    pub const fn source_role(self) -> SourceRole {
        match self {
            Self::Weight => SourceRole::Planar(OperandRole::Codes),
            Self::Scale => SourceRole::Planar(OperandRole::Scale),
        }
    }
}

/// The four attention projections, in the order [`MiniMaxM2LayerSources`] stores them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MiniMaxM2AttentionProjection {
    Q,
    K,
    V,
    O,
}

impl MiniMaxM2AttentionProjection {
    pub const ALL: [Self; 4] = [Self::Q, Self::K, Self::V, Self::O];

    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Q => "q_proj",
            Self::K => "k_proj",
            Self::V => "v_proj",
            Self::O => "o_proj",
        }
    }
}

/// The three routed-expert projections of `w2(silu(w1(x)) * w3(x))`, in storage order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MiniMaxM2ExpertProjection {
    W1,
    W2,
    W3,
}

impl MiniMaxM2ExpertProjection {
    pub const ALL: [Self; 3] = [Self::W1, Self::W2, Self::W3];

    pub const fn suffix(self) -> &'static str {
        match self {
            Self::W1 => "w1",
            Self::W2 => "w2",
            Self::W3 => "w3",
        }
    }
}

/// Every source one decoder block reads.
///
/// Fixed-arity records are named fields; the two sequences (layers, experts) are tables indexed by
/// numeric id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MiniMaxM2LayerSources {
    input_norm: MiniMaxM2DenseSource,
    post_attention_norm: MiniMaxM2DenseSource,
    q_norm: MiniMaxM2DenseSource,
    k_norm: MiniMaxM2DenseSource,
    router_gate: MiniMaxM2DenseSource,
    router_bias: MiniMaxM2DenseSource,
    attention: [MiniMaxM2PackedSource; 4],
    experts: [Vec<MiniMaxM2PackedSource>; 3],
}

impl MiniMaxM2LayerSources {
    pub const fn input_norm(&self) -> &MiniMaxM2DenseSource {
        &self.input_norm
    }

    pub const fn post_attention_norm(&self) -> &MiniMaxM2DenseSource {
        &self.post_attention_norm
    }

    pub const fn q_norm(&self) -> &MiniMaxM2DenseSource {
        &self.q_norm
    }

    pub const fn k_norm(&self) -> &MiniMaxM2DenseSource {
        &self.k_norm
    }

    pub const fn router_gate(&self) -> &MiniMaxM2DenseSource {
        &self.router_gate
    }

    pub const fn router_bias(&self) -> &MiniMaxM2DenseSource {
        &self.router_bias
    }

    pub fn attention(&self, projection: MiniMaxM2AttentionProjection) -> &MiniMaxM2PackedSource {
        &self.attention[projection as usize]
    }

    /// One projection's packed sources for every routed expert, indexed by numeric expert id.
    pub fn experts(&self, projection: MiniMaxM2ExpertProjection) -> &[MiniMaxM2PackedSource] {
        &self.experts[projection as usize]
    }

    /// Every dense source of this block, in storage order.
    pub fn dense(&self) -> impl Iterator<Item = &MiniMaxM2DenseSource> {
        [
            &self.input_norm,
            &self.post_attention_norm,
            &self.q_norm,
            &self.k_norm,
            &self.router_gate,
            &self.router_bias,
        ]
        .into_iter()
    }

    /// Every packed source of this block: the four attention projections, then each expert's
    /// `w1`/`w2`/`w3` in numeric expert order.
    pub fn packed(&self) -> impl Iterator<Item = &MiniMaxM2PackedSource> {
        let experts = (0..self.experts[0].len())
            .flat_map(move |expert| MiniMaxM2ExpertProjection::ALL.map(move |p| (p, expert)))
            .map(move |(projection, expert)| &self.experts(projection)[expert]);
        self.attention.iter().chain(experts)
    }
}

/// Every text-graph source of one MiniMax-M2 configuration, in one deterministic order.
///
/// Both the model graph and the disposition manifest read this table, so each name is defined in one
/// place and a caller cannot index a row the constructor did not write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MiniMaxM2SourceTable {
    embedding: MiniMaxM2DenseSource,
    final_norm: MiniMaxM2DenseSource,
    lm_head: MiniMaxM2DenseSource,
    layers: Vec<MiniMaxM2LayerSources>,
}

impl MiniMaxM2SourceTable {
    pub fn new(config: &MiniMaxM2Config) -> Result<Self, MiniMaxM2SourceTableError> {
        let hidden = config.hidden_size;
        let expert_inner = config.intermediate_size;
        let expert_count = config.num_local_experts;
        let vocab = config.vocab_size;
        let bf16 = |name: String, shape: Vec<usize>| MiniMaxM2DenseSource {
            name,
            role: MiniMaxM2DenseRole::Bf16,
            shape,
        };
        let f32_row = |name: String, shape: Vec<usize>| MiniMaxM2DenseSource {
            name,
            role: MiniMaxM2DenseRole::F32,
            shape,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let attention = MiniMaxM2AttentionProjection::ALL
                .map(|projection| {
                    let [out_dim, in_dim] = match projection {
                        MiniMaxM2AttentionProjection::Q => [config.q_dim(), hidden],
                        MiniMaxM2AttentionProjection::K | MiniMaxM2AttentionProjection::V => {
                            [config.kv_dim(), hidden]
                        }
                        MiniMaxM2AttentionProjection::O => [hidden, config.q_dim()],
                    };
                    packed_source(
                        format!("{prefix}.self_attn.{}", projection.suffix()),
                        [out_dim, in_dim],
                    )
                })
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let experts = MiniMaxM2ExpertProjection::ALL
                .map(|projection| {
                    let [out_dim, in_dim] = match projection {
                        MiniMaxM2ExpertProjection::W1 | MiniMaxM2ExpertProjection::W3 => {
                            [expert_inner, hidden]
                        }
                        MiniMaxM2ExpertProjection::W2 => [hidden, expert_inner],
                    };
                    (0..expert_count)
                        .map(|expert| {
                            packed_source(
                                format!(
                                    "{prefix}.block_sparse_moe.experts.{expert}.{}",
                                    projection.suffix()
                                ),
                                [out_dim, in_dim],
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;

            layers.push(MiniMaxM2LayerSources {
                input_norm: bf16(format!("{prefix}.input_layernorm.weight"), vec![hidden]),
                post_attention_norm: bf16(
                    format!("{prefix}.post_attention_layernorm.weight"),
                    vec![hidden],
                ),
                q_norm: bf16(
                    format!("{prefix}.self_attn.q_norm.weight"),
                    vec![config.q_dim()],
                ),
                k_norm: bf16(
                    format!("{prefix}.self_attn.k_norm.weight"),
                    vec![config.kv_dim()],
                ),
                // `nn.Linear(hidden, experts)` stores `[out, in]`, so the router weight is
                // `[experts, hidden]` in the checkpoint and the graph transposes it.
                router_gate: f32_row(
                    format!("{prefix}.block_sparse_moe.gate.weight"),
                    vec![expert_count, hidden],
                ),
                router_bias: f32_row(
                    format!("{prefix}.block_sparse_moe.e_score_correction_bias"),
                    vec![expert_count],
                ),
                attention: attention
                    .try_into()
                    .expect("one row per attention projection"),
                experts: experts.try_into().expect("one table per expert projection"),
            });
        }

        Ok(Self {
            embedding: bf16("model.embed_tokens.weight".to_string(), vec![vocab, hidden]),
            final_norm: bf16("model.norm.weight".to_string(), vec![hidden]),
            lm_head: bf16("lm_head.weight".to_string(), vec![vocab, hidden]),
            layers,
        })
    }

    pub const fn embedding(&self) -> &MiniMaxM2DenseSource {
        &self.embedding
    }

    pub const fn final_norm(&self) -> &MiniMaxM2DenseSource {
        &self.final_norm
    }

    pub const fn lm_head(&self) -> &MiniMaxM2DenseSource {
        &self.lm_head
    }

    pub fn layers(&self) -> &[MiniMaxM2LayerSources] {
        &self.layers
    }

    /// Every dense source, globals first and then each block in layer order.
    pub fn dense(&self) -> impl Iterator<Item = &MiniMaxM2DenseSource> {
        [&self.embedding, &self.final_norm, &self.lm_head]
            .into_iter()
            .chain(self.layers.iter().flat_map(MiniMaxM2LayerSources::dense))
    }

    /// Every packed source, in layer order.
    pub fn packed(&self) -> impl Iterator<Item = &MiniMaxM2PackedSource> {
        self.layers.iter().flat_map(MiniMaxM2LayerSources::packed)
    }

    /// Every checkpoint tensor name this table accounts for: one per dense row, two per packed pair.
    pub fn names(&self) -> BTreeSet<String> {
        self.dense()
            .map(|source| source.name.clone())
            .chain(
                self.packed()
                    .flat_map(MiniMaxM2PackedSource::checkpoint_names),
            )
            .collect()
    }
}

fn packed_source(
    linear_id: String,
    logical_shape: [usize; 2],
) -> Result<MiniMaxM2PackedSource, MiniMaxM2SourceTableError> {
    let descriptor =
        PackedWeight::try_new(MINIMAX_M25_PACKED_FORMAT, logical_shape).map_err(|source| {
            MiniMaxM2SourceTableError::PackedLayout {
                linear_id: linear_id.clone(),
                logical_shape,
                source,
            }
        })?;
    Ok(MiniMaxM2PackedSource {
        linear_id,
        descriptor,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum MiniMaxM2SourceTableError {
    #[error(
        "MiniMax-M2 packed linear {linear_id} with logical shape {logical_shape:?} has no valid packed layout"
    )]
    PackedLayout {
        linear_id: String,
        logical_shape: [usize; 2],
        #[source]
        source: PackedWeightError,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small, structurally exact configuration: every namespace of the pinned profile at test-sized
    /// dimensions.
    pub(super) fn small_config_json(layers: usize, experts: usize) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "architectures": ["MiniMaxM2ForCausalLM"],
            "attn_type_list": vec![1; layers],
            "head_dim": 8,
            "hidden_act": "silu",
            "hidden_size": 16,
            "intermediate_size": 12,
            "max_position_embeddings": 64,
            "model_type": "minimax_m2",
            "mtp_transformer_layers": 1,
            "num_attention_heads": 4,
            "num_experts_per_tok": 2.min(experts),
            "num_hidden_layers": layers,
            "num_key_value_heads": 2,
            "num_local_experts": experts,
            "num_mtp_modules": 3,
            "qk_norm_type": "per_layer",
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "float8_e4m3fn",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["gate", "e_score_correction_bias", "lm_head"]
            },
            "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000,
            "rotary_dim": 4,
            "scoring_func": "sigmoid",
            "shared_intermediate_size": 0,
            "tie_word_embeddings": false,
            "use_cache": true,
            "use_mtp": true,
            "use_qk_norm": true,
            "use_routing_bias": true,
            "vocab_size": 32
        }))
        .expect("serialize config fixture")
    }

    pub(super) fn small_config() -> MiniMaxM2Config {
        MiniMaxM2Config::from_slice(&small_config_json(1, 2)).expect("valid config fixture")
    }

    /// The exact pinned MiniMax-M2.5 profile from spec 365's identity table.
    pub(super) fn exact_m25_config_value() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["MiniMaxM2ForCausalLM"],
            "attn_type_list": vec![1; 62],
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "intermediate_size": 1536,
            "max_position_embeddings": 196608,
            "model_type": "minimax_m2",
            "mtp_transformer_layers": 1,
            "num_attention_heads": 48,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 62,
            "num_key_value_heads": 8,
            "num_local_experts": 256,
            "num_mtp_modules": 3,
            "qk_norm_type": "per_layer",
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "float8_e4m3fn",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["gate", "e_score_correction_bias", "lm_head"]
            },
            "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000,
            "rotary_dim": 64,
            "scoring_func": "sigmoid",
            "shared_intermediate_size": 0,
            "tie_word_embeddings": false,
            "use_cache": true,
            "use_mtp": true,
            "use_qk_norm": true,
            "use_routing_bias": true,
            "vocab_size": 200064
        })
    }

    pub(super) fn exact_m25_config() -> MiniMaxM2Config {
        let bytes = serde_json::to_vec(&exact_m25_config_value()).expect("serialize exact config");
        MiniMaxM2Config::from_m25_slice(&bytes).expect("accept exact M2.5")
    }

    fn small_index_value() -> serde_json::Value {
        let names = [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.block_sparse_moe.e_score_correction_bias",
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.k_proj.weight_scale_inv",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.o_proj.weight_scale_inv",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight_scale_inv",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.v_proj.weight_scale_inv",
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_inv",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight_scale_inv",
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
            "model.layers.0.block_sparse_moe.experts.0.w3.weight_scale_inv",
            "model.layers.0.block_sparse_moe.experts.1.w1.weight",
            "model.layers.0.block_sparse_moe.experts.1.w1.weight_scale_inv",
            "model.layers.0.block_sparse_moe.experts.1.w2.weight",
            "model.layers.0.block_sparse_moe.experts.1.w2.weight_scale_inv",
            "model.layers.0.block_sparse_moe.experts.1.w3.weight",
            "model.layers.0.block_sparse_moe.experts.1.w3.weight_scale_inv",
        ];
        assert_eq!(names.len(), 29, "fixture tensor count must stay explicit");
        let weight_map = names
            .into_iter()
            .enumerate()
            .map(|(index, name)| {
                let shard = if index % 2 == 0 {
                    "model-00000-of-00003.safetensors"
                } else {
                    "model-00001-of-00003.safetensors"
                };
                (
                    name.to_string(),
                    serde_json::Value::String(shard.to_string()),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        serde_json::json!({"metadata":{"total_size":1234}, "weight_map":weight_map})
    }

    /// The source table accounts for every index name once, splits packed and dense by the pinned
    /// counts, and orders experts by numeric id.
    ///
    /// Red under: dropping a scale name, classifying a router row as packed, or ordering experts by the
    /// lexical `{expert}` string, which would put expert 10 before expert 2.
    #[test]
    fn source_table_accounts_for_every_index_name_once() {
        let layers = 2;
        let experts = 12;
        let config =
            MiniMaxM2Config::from_slice(&small_config_json(layers, experts)).expect("config");
        let table = MiniMaxM2SourceTable::new(&config).expect("source table");

        let dense = table.dense().count();
        let packed = table.packed().count();
        assert_eq!(
            dense,
            3 + layers * 6,
            "three globals plus six rows per block"
        );
        assert_eq!(
            packed,
            layers * (4 + experts * 3),
            "four attention plus three per expert, per block"
        );
        assert_eq!(
            table.names().len(),
            dense + 2 * packed,
            "one name per dense row and two per packed pair, none colliding"
        );

        // Exactly the router rows are F32; everything else dense is BF16.
        let f32_rows = table
            .dense()
            .filter(|source| source.role() == MiniMaxM2DenseRole::F32)
            .count();
        assert_eq!(f32_rows, 2 * layers, "one gate and one bias per block");

        // Expert order is numeric, so expert 10 follows expert 9, not expert 1.
        let ids = table.layers()[0]
            .experts(MiniMaxM2ExpertProjection::W1)
            .iter()
            .map(|source| {
                source
                    .linear_id()
                    .rsplit_once('.')
                    .and_then(|(prefix, _)| prefix.rsplit_once('.'))
                    .map(|(_, id)| id.parse::<usize>().expect("numeric expert id"))
                    .expect("expert linear id")
            })
            .collect::<Vec<_>>();
        assert_eq!(ids, (0..experts).collect::<Vec<_>>());

        // Both components of one linear keep the checkpoint spelling.
        let attention = table.layers()[1].attention(MiniMaxM2AttentionProjection::O);
        assert_eq!(
            attention.checkpoint_names(),
            [
                "model.layers.1.self_attn.o_proj.weight".to_string(),
                "model.layers.1.self_attn.o_proj.weight_scale_inv".to_string(),
            ]
        );
        assert_eq!(
            attention.descriptor().shape(),
            [config.hidden_size, config.q_dim()],
            "o_proj contracts the query width back to hidden"
        );
    }

    #[test]
    fn config_parses_flat_qk_norm_partial_rope_and_block_fp8() {
        let config = small_config();
        assert_eq!(config.q_dim(), 32);
        assert_eq!(config.kv_dim(), 16);
        assert_eq!(config.kv_groups(), 2);
        assert_eq!(config.rotary_dim, 4);
        assert_eq!(config.fp8_scale_shape(32, 16), [1, 1]);
        assert_eq!(config.quantization.format, "float8_e4m3fn");
    }

    #[test]
    fn exact_m25_profile_accepts_only_the_pinned_dimensions() {
        let exact = exact_m25_config_value();
        let config = exact_m25_config();
        assert_eq!(config.q_dim(), 6144);
        assert_eq!(config.kv_dim(), 1024);
        assert_eq!(config.fp8_scale_shape(6144, 3072), [48, 24]);

        let mut drift = exact;
        drift["num_hidden_layers"] = serde_json::json!(61);
        drift["attn_type_list"] = serde_json::json!(vec![1; 61]);
        let error = MiniMaxM2Config::from_m25_slice(
            &serde_json::to_vec(&drift).expect("serialize drifted config"),
        )
        .expect_err("M2.5 dimension drift must fail");
        assert!(error.to_string().contains("num_hidden_layers"));
    }

    #[test]
    fn config_rejects_architecture_router_and_quantization_drift() {
        let cases = [("model_type", "llama"), ("scoring_func", "softmax")];
        for (field, replacement) in cases {
            let mut value: serde_json::Value =
                serde_json::from_slice(&small_config_json(1, 2)).expect("parse fixture value");
            value[field] = serde_json::Value::String(replacement.to_string());
            let error = MiniMaxM2Config::from_slice(
                &serde_json::to_vec(&value).expect("serialize corrupted config"),
            )
            .expect_err("corrupted config must fail");
            assert!(error.to_string().contains(field), "error was {error}");
        }

        let mut value: serde_json::Value =
            serde_json::from_slice(&small_config_json(1, 2)).expect("parse fixture value");
        value["quantization_config"]["fmt"] = serde_json::json!("float8_e5m2");
        let error = MiniMaxM2Config::from_slice(
            &serde_json::to_vec(&value).expect("serialize corrupted config"),
        )
        .expect_err("wrong FP8 format must fail");
        assert!(error.to_string().contains("quantization_config"));
    }

    #[test]
    fn index_requires_the_exact_tensor_set_and_surfaces_missing_declared_shard() {
        let bytes = serde_json::to_vec(&small_index_value()).expect("serialize index fixture");
        let index = MiniMaxM2CheckpointIndex::from_slice(&bytes, &small_config())
            .expect("exact synthetic index must validate");
        assert_eq!(index.tensor_count, 29);
        assert_eq!(index.declared_total_size, 1234);
        assert_eq!(index.referenced_shards.len(), 2);
        assert_eq!(index.declared_shard_count, 3);
        assert_eq!(index.missing_declared_shard_ordinals, [2]);
    }

    #[test]
    fn index_rejects_missing_extra_and_inconsistent_shard_metadata() {
        let cases = ["missing", "extra", "inconsistent", "malformed"];
        for case in cases {
            let mut value = small_index_value();
            let map = value["weight_map"]
                .as_object_mut()
                .expect("fixture weight map");
            match case {
                "missing" => {
                    map.remove("model.layers.0.self_attn.q_proj.weight_scale_inv");
                }
                "extra" => {
                    map.insert(
                        "model.layers.0.self_attn.q_proj.dense_fallback".to_string(),
                        serde_json::json!("model-00000-of-00003.safetensors"),
                    );
                }
                "inconsistent" => {
                    *map.values_mut().next().expect("one shard") =
                        serde_json::json!("model-00000-of-00004.safetensors");
                }
                "malformed" => {
                    for shard in map.values_mut() {
                        *shard = serde_json::json!("weights.safetensors");
                    }
                }
                _ => unreachable!(),
            }
            let bytes = serde_json::to_vec(&value).expect("serialize corrupted index");
            let error = MiniMaxM2CheckpointIndex::from_slice(&bytes, &small_config())
                .expect_err("corrupted index must fail");
            let message = error.to_string();
            match case {
                "missing" => assert!(message.contains("q_proj.weight_scale_inv")),
                "extra" => assert!(message.contains("dense_fallback")),
                "inconsistent" => assert!(message.contains("declares 4 shards")),
                "malformed" => assert!(message.contains("weights.safetensors")),
                _ => unreachable!(),
            }
        }
    }
}
