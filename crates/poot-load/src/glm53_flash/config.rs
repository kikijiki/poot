use super::*;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Glm53FlashAttentionKind {
    LinearAttention,
    DeepseekSparseAttention,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Glm53FlashMlpKind {
    Dense,
    Sparse,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Glm53FlashIndexerKind {
    Full,
    Shared,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Glm53FlashLinearAttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub short_conv_kernel_size: usize,
    pub gate_lower_bound: f32,
    pub kda_layers: Vec<usize>,
    pub full_attn_layers: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Glm53FlashTextConfig {
    pub model_type: String,
    pub dtype: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_nextn_predict_layers: usize,
    pub max_position_embeddings: usize,
    pub first_k_dense_replace: usize,
    pub layer_types: Vec<Glm53FlashAttentionKind>,
    pub mlp_layer_types: Vec<Glm53FlashMlpKind>,
    pub indexer_types: Vec<Glm53FlashIndexerKind>,
    pub linear_attn_config: Glm53FlashLinearAttentionConfig,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub mla_use_nope: bool,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub index_kpool: usize,
    pub index_kpool_always_select_tail: bool,
    pub index_kpool_compress: bool,
    pub index_share_for_mtp_iteration: bool,
    pub indexer_rope_interleave: bool,
    pub mhc: bool,
    pub hc_mult: usize,
    pub hc_eps: f32,
    pub hc_sinkhorn_iters: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub scoring_func: String,
    pub topk_method: String,
    pub moe_router_dtype: String,
    pub swiglu_limit: f32,
    pub use_cache: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Glm53FlashVisionConfig {
    pub model_type: String,
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub out_hidden_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm53FlashQuantizationConfig {
    pub activation_scheme: String,
    pub fmt: String,
    pub quant_method: String,
    pub modules_to_not_convert: Vec<String>,
    pub weight_block_size: [usize; 2],
}

/// Indexer schedule proven by [`Glm53FlashHfConfig::validate`]: every text layer uses one indexer kind.
///
/// Only validation constructs it, so a consumer maps the kind without re-checking the per-layer list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glm53FlashIndexerSchedule {
    pub(crate) kind: Glm53FlashIndexerKind,
}

impl Glm53FlashIndexerSchedule {
    /// The schedule of a per-layer list whose rows all share one kind, or `None` for an empty or mixed list.
    pub(crate) fn uniform(layers: &[Glm53FlashIndexerKind]) -> Option<Self> {
        let (&kind, rest) = layers.split_first()?;
        rest.iter()
            .all(|&other| other == kind)
            .then_some(Self { kind })
    }

    /// The indexer kind shared by every text layer.
    pub const fn kind(self) -> Glm53FlashIndexerKind {
        self.kind
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Glm53FlashHfConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub transformers_version: String,
    pub text_config: Glm53FlashTextConfig,
    pub vision_config: Glm53FlashVisionConfig,
    pub quantization_config: Glm53FlashQuantizationConfig,
}

impl Glm53FlashHfConfig {
    pub fn from_slice(bytes: &[u8]) -> Result<Self, Glm53FlashMetadataError> {
        let config: Self = serde_json::from_slice(bytes)?;
        config.validate()?;
        Ok(config)
    }

    /// Revalidate the pinned architecture after a caller has retained or modified the public config.
    ///
    /// Returns the indexer schedule the validation proves.
    pub fn validate(&self) -> Result<Glm53FlashIndexerSchedule, Glm53FlashMetadataError> {
        let invalid = |message: &str| Glm53FlashMetadataError::InvalidConfig(message.to_string());

        if self.model_type != GLM53_FLASH_MODEL_TYPE
            || self.architectures.len() != 1
            || self.architectures[0] != GLM53_FLASH_ARCHITECTURE
        {
            return Err(invalid("outer architecture/model_type mismatch"));
        }
        if self.transformers_version != "5.16.0" {
            return Err(invalid("transformers_version must be 5.16.0"));
        }
        let text = &self.text_config;
        if text.model_type != GLM53_FLASH_TEXT_MODEL_TYPE
            || self.vision_config.model_type != GLM53_FLASH_VISION_MODEL_TYPE
        {
            return Err(invalid("nested model_type mismatch"));
        }
        if text.dtype != "bfloat16"
            || text.vocab_size != 154_880
            || text.hidden_size != 4_096
            || text.intermediate_size != 12_288
            || text.moe_intermediate_size != 2_048
            || text.num_hidden_layers != 45
            || text.num_attention_heads != 64
            || text.num_key_value_heads != 64
            || text.num_nextn_predict_layers != 1
            || text.max_position_embeddings != 1_048_576
        {
            return Err(invalid("published text dimensions mismatch"));
        }
        if self.vision_config.depth != 24
            || self.vision_config.hidden_size != 1_024
            || self.vision_config.num_heads != 16
            || self.vision_config.out_hidden_size != text.hidden_size
        {
            return Err(invalid("published vision dimensions mismatch"));
        }
        if text.layer_types.len() != text.num_hidden_layers
            || text.mlp_layer_types.len() != text.num_hidden_layers
            || text.indexer_types.len() != text.num_hidden_layers
        {
            return Err(invalid("per-layer schedule length mismatch"));
        }
        for layer in 0..text.num_hidden_layers {
            let expected_attention = if layer % 4 == 3 {
                Glm53FlashAttentionKind::DeepseekSparseAttention
            } else {
                Glm53FlashAttentionKind::LinearAttention
            };
            let expected_mlp = if layer < 3 {
                Glm53FlashMlpKind::Dense
            } else {
                Glm53FlashMlpKind::Sparse
            };
            if text.layer_types[layer] != expected_attention
                || text.mlp_layer_types[layer] != expected_mlp
            {
                return Err(invalid("published per-layer schedule mismatch"));
            }
        }
        let indexer_schedule = Glm53FlashIndexerSchedule::uniform(&text.indexer_types)
            .filter(|schedule| schedule.kind() == Glm53FlashIndexerKind::Full)
            .ok_or_else(|| invalid("published indexer schedule mismatch"))?;
        if text.first_k_dense_replace != 3 {
            return Err(invalid("first_k_dense_replace must be 3"));
        }

        let linear = &text.linear_attn_config;
        let expected_kda = (0..text.num_hidden_layers)
            .filter(|layer| layer % 4 != 3)
            .collect::<Vec<_>>();
        let expected_full = (0..text.num_hidden_layers)
            .filter(|layer| layer % 4 == 3)
            .collect::<Vec<_>>();
        if linear.num_heads != 64
            || linear.head_dim != 128
            || linear.short_conv_kernel_size != 4
            || linear.gate_lower_bound != -5.0
            || linear.kda_layers != expected_kda
            || linear.full_attn_layers != expected_full
        {
            return Err(invalid("published linear-attention contract mismatch"));
        }
        if text.q_lora_rank != 1_536
            || text.kv_lora_rank != 512
            || text.qk_head_dim != 256
            || text.qk_nope_head_dim != 256
            || text.qk_rope_head_dim != 0
            || text.v_head_dim != 256
            || text.head_dim != 0
            || !text.mla_use_nope
        {
            return Err(invalid("published NoPE MLA contract mismatch"));
        }
        if text.rms_norm_eps != 1.0e-5 {
            return Err(invalid("rms_norm_eps must be 1e-5"));
        }
        if text.index_n_heads != 32
            || text.index_head_dim != 128
            || text.index_topk != 2_048
            || text.index_kpool != 4
            || !text.index_kpool_always_select_tail
            || !text.index_kpool_compress
            || !text.index_share_for_mtp_iteration
            || !text.indexer_rope_interleave
        {
            return Err(invalid("published DSA indexer contract mismatch"));
        }
        if !text.mhc || text.hc_mult != 4 || text.hc_eps != 1.0e-6 || text.hc_sinkhorn_iters != 20 {
            return Err(invalid("published mHC contract mismatch"));
        }
        if text.n_group != 1
            || text.topk_group != 1
            || text.n_routed_experts != 288
            || text.n_shared_experts != 1
            || text.num_experts_per_tok != 8
            || !text.norm_topk_prob
            || text.routed_scaling_factor != 2.5
            || text.scoring_func != "sigmoid"
            || text.topk_method != "noaux_tc"
            || text.moe_router_dtype != "float32"
            || text.swiglu_limit != 10.0
            || !text.use_cache
        {
            return Err(invalid("published MoE/FFN contract mismatch"));
        }

        let quant = &self.quantization_config;
        if quant.quant_method != "fp8"
            || quant.fmt != "e4m3"
            || quant.activation_scheme != "dynamic"
            || quant.weight_block_size != [128, 128]
        {
            return Err(invalid("published FP8 quantization contract mismatch"));
        }
        let exclusion_count = quant.modules_to_not_convert.len();
        let unique_exclusion_count = quant
            .modules_to_not_convert
            .iter()
            .collect::<BTreeSet<_>>()
            .len();
        let mut canonical_exclusions = quant.modules_to_not_convert.clone();
        canonical_exclusions.sort_unstable();
        let canonical_exclusions = canonical_exclusions.join("\n") + "\n";
        if exclusion_count != GLM53_FLASH_QUANT_EXCLUSION_COUNT
            || unique_exclusion_count != exclusion_count
            || sha256_hex(canonical_exclusions.as_bytes()) != GLM53_FLASH_EXCLUSION_SHA256
        {
            return Err(invalid("published FP8 exclusion inventory mismatch"));
        }
        Ok(indexer_schedule)
    }
}
