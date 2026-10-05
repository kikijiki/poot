use poot_tensor::DType;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;

#[cfg(test)]
use poot_graph_ir::op::PackedWeight;
#[cfg(test)]
use poot_graph_ir::ops::{PackedLinearGraphRow, packed_indexed_linear, packed_linear, rmsnorm};
#[cfg(test)]
use poot_graph_ir::{BinOp, Builder, Graph, RedOp, Scalar, Slot, StateRole, TensorType, Traced};
#[cfg(test)]
use poot_load::glm53_flash::{
    Glm53FlashAttentionKind, Glm53FlashHfConfig, Glm53FlashIndexerKind, Glm53FlashMlpKind,
};
use poot_load::glm53_flash::{Glm53FlashTextInventoryReport, Glm53FlashTextLoadResult};
use poot_load::packed_safetensors::{
    ExactSourceKind, LoadedExactSource, LoadedPackedLinear, MixedLoadResult,
};
#[cfg(test)]
use poot_quant::format::{ScaleEncoding, WeightFormat};
use poot_quant::weights::WeightEntry;

#[cfg(test)]
use crate::deepseek3::deepseek3_router_gate_and_ids;
#[cfg(test)]
use crate::deepseek4::{deepseek4_clamped_swiglu, hyper_connection, hyper_connection_combine};

#[cfg(test)]
use super::Glm53FlashTraceError;
use super::{
    GLM5NEXT_DSA_STATE_PAIR_COUNT, GLM5NEXT_KDA_STATE_PAIR_COUNT, GLM53_FLASH_REVISION,
    Glm5NextDsaConfig, Glm5NextDsaRole, Glm5NextKdaConfig,
};
#[cfg(test)]
use super::{
    Glm5NextDsaInputs, Glm5NextDsaScope, Glm5NextDsaSourceClass, Glm5NextIndexerKind,
    Glm5NextKdaInputs, Glm5NextKdaScope, dense_source_linear, glm5next_dsa_block_with_projections,
    glm5next_dsa_role_specs, glm5next_dsa_state_specs, glm5next_kda_block, glm5next_kda_role_specs,
    glm5next_kda_state_specs,
};

#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_LAYER_COUNT: usize = 45;
#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_KDA_LAYER_COUNT: usize = 34;
#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_DENSE_FFN_LAYER_COUNT: usize = 3;
#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_STATE_PAIR_COUNT: usize = 101;
#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_DENSE_SOURCE_COUNT: usize = 1_067;
#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_BF16_SOURCE_COUNT: usize = 777;
#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_F32_SOURCE_COUNT: usize = 290;
#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_PACKED_SOURCE_COUNT: usize = 36_467;
#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_DSA_PACKED_COUNT: usize = 44;
#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_DENSE_FFN_PACKED_COUNT: usize = 9;
#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_ROUTED_EXPERT_PACKED_COUNT: usize = 36_288;
#[cfg(test)]
pub(crate) const GLM5NEXT_TEXT_SHARED_EXPERT_PACKED_COUNT: usize = 126;

#[cfg(test)]
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
#[cfg(test)]
const FINAL_NORM: &str = "model.language_model.norm.weight";
#[cfg(test)]
const LM_HEAD: &str = "lm_head.weight";

/// Tag of the full-text graph's caller-owned `Slot::Activation` boundary, as passed to
/// `Builder::slot_named`. The traced input is named `"{lowercased kind}.{tag}"`, and its value is
/// 1.0 for an ordinary step (that step's query is valid). A consumer resolves it by kind plus this
/// tag, never by slot kind alone, because `Slot::Activation` has no model-engine convention.
#[cfg(test)]
pub(crate) const GLM5NEXT_QUERY_VALIDITY_TAG: &str = "glm5next.query_validity";

/// Tag of the full-text graph's `Slot::Mask` boundary (same naming rule as
/// [`GLM5NEXT_QUERY_VALIDITY_TAG`]). Unlike the additive attention masks elsewhere, this row is
/// 0/1 causal visibility: the DSA reference reads `visibility[i] > 0.5`.
#[cfg(test)]
pub(crate) const GLM5NEXT_VISIBILITY_TAG: &str = "glm5next.visibility";

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Glm5NextTextScope {
    Text,
    ImageVideo,
    Mtp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm5NextTextAttentionKind {
    Kda,
    Dsa,
}

impl Glm5NextTextAttentionKind {
    /// Fixed decode-state pairs one block of this kind carries.
    pub const fn state_pair_count(self) -> usize {
        match self {
            Self::Kda => GLM5NEXT_KDA_STATE_PAIR_COUNT,
            Self::Dsa => GLM5NEXT_DSA_STATE_PAIR_COUNT,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm5NextTextFfnKind {
    Dense,
    Sparse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glm5NextTextLayerRow {
    pub layer: usize,
    pub attention: Glm5NextTextAttentionKind,
    pub ffn: Glm5NextTextFfnKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Glm5NextTextProjection {
    Gate,
    Up,
    Down,
}

#[cfg(test)]
impl Glm5NextTextProjection {
    const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    const fn suffix(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }

    const fn logical_shape(self, hidden: usize, intermediate: usize) -> [usize; 2] {
        match self {
            Self::Gate | Self::Up => [intermediate, hidden],
            Self::Down => [hidden, intermediate],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Glm5NextPackedTextRole {
    Dsa {
        role: Glm5NextDsaRole,
    },
    DenseFfn {
        projection: Glm5NextTextProjection,
    },
    RoutedExpert {
        expert: usize,
        projection: Glm5NextTextProjection,
    },
    SharedExpert {
        projection: Glm5NextTextProjection,
    },
}

/// Source class named by a text source-plan or ownership error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm5NextTextSourceKind {
    Dense,
    Packed,
}

impl fmt::Display for Glm5NextTextSourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Dense => "dense",
            Self::Packed => "packed",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm5NextDenseSourceSpec {
    pub name: String,
    /// Card 359 source kind. The loader rejects a row whose safetensors dtype differs from its kind.
    pub kind: ExactSourceKind,
    /// Graph dtype of `kind`.
    pub dtype: DType,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm5NextPackedSourceSpec {
    pub layer: usize,
    pub role: Glm5NextPackedTextRole,
    pub linear_id: String,
    pub descriptor: PackedWeight,
}

/// Selection-only-bias sigmoid router parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm5NextRouterConfig {
    pub experts: usize,
    pub top_k: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scale: f32,
}

/// Manifold-constrained hyper-connection parameters shared by every mHC site.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MhcConfig {
    streams: usize,
    eps: f32,
    sinkhorn_iters: usize,
}

/// One of the two mHC sites in every layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(test)]
enum MhcSite {
    Attention,
    Ffn,
}

#[cfg(test)]
impl MhcSite {
    const ALL: [Self; 2] = [Self::Attention, Self::Ffn];

    /// Checkpoint names of this site's projection, base, and scale rows, in that order.
    fn source_names(self, prefix: &str) -> [String; 3] {
        let site = match self {
            Self::Attention => "attn",
            Self::Ffn => "ffn",
        };
        ["fn", "base", "scale"].map(|part| format!("{prefix}.hc_{site}_{part}"))
    }
}

#[cfg(test)]
fn layer_prefix(layer: usize) -> String {
    format!("model.language_model.layers.{layer}")
}

#[cfg(test)]
fn input_norm_name(prefix: &str) -> String {
    format!("{prefix}.input_layernorm.weight")
}

#[cfg(test)]
fn post_attention_norm_name(prefix: &str) -> String {
    format!("{prefix}.post_attention_layernorm.weight")
}

#[cfg(test)]
fn router_gate_name(prefix: &str) -> String {
    format!("{prefix}.mlp.gate.weight")
}

#[cfg(test)]
fn router_bias_name(prefix: &str) -> String {
    format!("{prefix}.mlp.gate.e_score_correction_bias")
}

/// The GLM-5.3-Flash main-text architecture, validated once.
///
/// Outside this module only `Glm5NextText::exact` constructs it, so a source plan or trace built from it never
/// revalidates the config or its SHA-256 exclusion inventory.
#[derive(Clone, Debug, PartialEq)]
pub struct Glm5NextText {
    vocab_size: usize,
    hidden_size: usize,
    rms_norm_eps: f32,
    mhc: MhcConfig,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    swiglu_limit: f32,
    router: Glm5NextRouterConfig,
    kda: Glm5NextKdaConfig,
    dsa: Glm5NextDsaConfig,
    layers: Vec<Glm5NextTextLayerRow>,
}

impl Glm5NextText {
    /// Validate the pinned config and derive the text architecture from it.
    ///
    /// The KDA and DSA block configs are derived from the loaded config and must equal the pinned block configs
    /// that Cards 366 and 367 checked against committed metadata.
    #[cfg(test)]
    pub(crate) fn exact(config: &Glm53FlashHfConfig) -> Result<Self, Glm53FlashTraceError> {
        let indexer_schedule = config.validate()?;
        let text = &config.text_config;
        let linear = &text.linear_attn_config;
        let kda = Glm5NextKdaConfig {
            hidden_size: text.hidden_size,
            num_heads: linear.num_heads,
            head_dim: linear.head_dim,
            conv_width: linear.short_conv_kernel_size,
            // Assumption: Kimi Linear KDA sizes `f_a_proj`/`g_a_proj` to the linear head width (llama.cpp
            // `src/models/kimi-linear.cpp` at `ee445f9`). No GLM source states it; the committed GLM shapes
            // `[128, 4096]` agree.
            gate_rank: linear.head_dim,
            gate_lower_bound: linear.gate_lower_bound,
        };
        if kda != Glm5NextKdaConfig::exact() {
            return Err(Glm53FlashTraceError::TextKdaConfig { derived: kda });
        }
        let indexer_kind = match indexer_schedule.kind() {
            Glm53FlashIndexerKind::Full => Glm5NextIndexerKind::Full,
            Glm53FlashIndexerKind::Shared => Glm5NextIndexerKind::Shared,
        };
        let dsa = Glm5NextDsaConfig {
            hidden_size: text.hidden_size,
            query_rank: text.q_lora_rank,
            kv_rank: text.kv_lora_rank,
            num_heads: text.num_attention_heads,
            qk_head_dim: text.qk_head_dim,
            v_head_dim: text.v_head_dim,
            index_n_heads: text.index_n_heads,
            index_head_dim: text.index_head_dim,
            index_topk: text.index_topk,
            index_kpool: text.index_kpool,
            qk_rope_head_dim: text.qk_rope_head_dim,
            mla_use_nope: text.mla_use_nope,
            index_kpool_always_select_tail: text.index_kpool_always_select_tail,
            index_kpool_compress: text.index_kpool_compress,
            indexer_kind,
        };
        if dsa != Glm5NextDsaConfig::exact() {
            return Err(Glm53FlashTraceError::TextDsaConfig { derived: dsa });
        }
        let layers = text
            .layer_types
            .iter()
            .zip(&text.mlp_layer_types)
            .enumerate()
            .map(|(layer, (attention, ffn))| Glm5NextTextLayerRow {
                layer,
                attention: match attention {
                    Glm53FlashAttentionKind::LinearAttention => Glm5NextTextAttentionKind::Kda,
                    Glm53FlashAttentionKind::DeepseekSparseAttention => {
                        Glm5NextTextAttentionKind::Dsa
                    }
                },
                ffn: match ffn {
                    Glm53FlashMlpKind::Dense => Glm5NextTextFfnKind::Dense,
                    Glm53FlashMlpKind::Sparse => Glm5NextTextFfnKind::Sparse,
                },
            })
            .collect();
        Ok(Self {
            vocab_size: text.vocab_size,
            hidden_size: text.hidden_size,
            rms_norm_eps: text.rms_norm_eps,
            mhc: MhcConfig {
                streams: text.hc_mult,
                eps: text.hc_eps,
                sinkhorn_iters: text.hc_sinkhorn_iters,
            },
            intermediate_size: text.intermediate_size,
            moe_intermediate_size: text.moe_intermediate_size,
            swiglu_limit: text.swiglu_limit,
            router: Glm5NextRouterConfig {
                experts: text.n_routed_experts,
                top_k: text.num_experts_per_tok,
                n_group: text.n_group,
                topk_group: text.topk_group,
                routed_scale: text.routed_scaling_factor,
            },
            kda,
            dsa,
            layers,
        })
    }

    pub fn layers(&self) -> &[Glm5NextTextLayerRow] {
        &self.layers
    }

    pub const fn router(&self) -> Glm5NextRouterConfig {
        self.router
    }

    pub const fn kda(&self) -> Glm5NextKdaConfig {
        self.kda
    }

    pub const fn dsa(&self) -> Glm5NextDsaConfig {
        self.dsa
    }

    pub fn state_pair_count(&self) -> usize {
        self.layers
            .iter()
            .map(|row| row.attention.state_pair_count())
            .sum()
    }
}

/// Graph-source inventory derived from one validated [`Glm5NextText`].
///
/// This is graph metadata only. It owns no checkpoint bytes and makes no executor or backend claim.
#[derive(Clone, Debug)]
pub struct Glm5NextTextSourcePlan {
    text: Glm5NextText,
    dense: BTreeMap<String, Glm5NextDenseSourceSpec>,
    packed: BTreeMap<(usize, Glm5NextPackedTextRole), Glm5NextPackedSourceSpec>,
}

impl Glm5NextTextSourcePlan {
    /// Build the exact 45-layer source plan and check it against the committed inventory counts.
    #[cfg(test)]
    pub(crate) fn exact(config: &Glm53FlashHfConfig) -> Result<Self, Glm53FlashTraceError> {
        let plan = Self::for_text(Glm5NextText::exact(config)?)?;
        plan.check_exact_counts()?;
        Ok(plan)
    }

    #[cfg(test)]
    fn for_text(text: Glm5NextText) -> Result<Self, Glm53FlashTraceError> {
        let hidden = text.hidden_size;
        let streams = text.mhc.streams;
        let mhc_mix = streams
            .checked_add(2)
            .and_then(|width| width.checked_mul(streams))
            .ok_or(Glm53FlashTraceError::TextShapeOverflow {
                field: "mHC mix width",
            })?;
        let mhc_input =
            streams
                .checked_mul(hidden)
                .ok_or(Glm53FlashTraceError::TextShapeOverflow {
                    field: "mHC input width",
                })?;
        let mut rows = SourceRows::default();
        rows.dense(EMBEDDING, DType::BF16, vec![text.vocab_size, hidden])?;

        for row in &text.layers {
            let layer = row.layer;
            let prefix = layer_prefix(layer);
            for site in MhcSite::ALL {
                let [projection, base, scale] = site.source_names(&prefix);
                rows.dense(&projection, DType::BF16, vec![mhc_mix, mhc_input])?;
                rows.dense(&base, DType::F32, vec![mhc_mix])?;
                rows.dense(&scale, DType::F32, vec![3])?;
            }
            rows.dense(&input_norm_name(&prefix), DType::BF16, vec![hidden])?;
            rows.dense(
                &post_attention_norm_name(&prefix),
                DType::BF16,
                vec![hidden],
            )?;

            let attention_prefix = format!("{prefix}.self_attn");
            match row.attention {
                Glm5NextTextAttentionKind::Kda => {
                    for spec in
                        glm5next_kda_role_specs(GLM53_FLASH_REVISION, &attention_prefix, text.kda)?
                    {
                        rows.dense(&spec.name, spec.dtype, spec.shape)?;
                    }
                }
                Glm5NextTextAttentionKind::Dsa => {
                    for spec in
                        glm5next_dsa_role_specs(GLM53_FLASH_REVISION, &attention_prefix, text.dsa)?
                    {
                        match spec.source_class {
                            Glm5NextDsaSourceClass::ExactDense => {
                                rows.dense(&spec.name, spec.dtype, spec.shape)?;
                            }
                            Glm5NextDsaSourceClass::PackedProjection => {
                                let linear_id =
                                    spec.name.strip_suffix(".weight").ok_or_else(|| {
                                        Glm53FlashTraceError::TextPackedSourceName {
                                            name: spec.name.clone(),
                                        }
                                    })?;
                                let logical_shape = <[usize; 2]>::try_from(spec.shape.as_slice())
                                    .map_err(|_| {
                                    Glm53FlashTraceError::TextPackedSourceRank {
                                        name: spec.name.clone(),
                                        shape: spec.shape.clone(),
                                    }
                                })?;
                                rows.packed(
                                    layer,
                                    Glm5NextPackedTextRole::Dsa { role: spec.role },
                                    linear_id,
                                    logical_shape,
                                )?;
                            }
                        }
                    }
                }
            }

            match row.ffn {
                Glm5NextTextFfnKind::Dense => {
                    for projection in Glm5NextTextProjection::ALL {
                        rows.packed(
                            layer,
                            Glm5NextPackedTextRole::DenseFfn { projection },
                            &format!("{prefix}.mlp.{}", projection.suffix()),
                            projection.logical_shape(hidden, text.intermediate_size),
                        )?;
                    }
                }
                Glm5NextTextFfnKind::Sparse => {
                    let experts = text.router.experts;
                    rows.dense(
                        &router_gate_name(&prefix),
                        DType::BF16,
                        vec![experts, hidden],
                    )?;
                    rows.dense(&router_bias_name(&prefix), DType::F32, vec![experts])?;
                    for expert in 0..experts {
                        for projection in Glm5NextTextProjection::ALL {
                            rows.packed(
                                layer,
                                Glm5NextPackedTextRole::RoutedExpert { expert, projection },
                                &format!("{prefix}.mlp.experts.{expert}.{}", projection.suffix()),
                                projection.logical_shape(hidden, text.moe_intermediate_size),
                            )?;
                        }
                    }
                    for projection in Glm5NextTextProjection::ALL {
                        rows.packed(
                            layer,
                            Glm5NextPackedTextRole::SharedExpert { projection },
                            &format!("{prefix}.mlp.shared_experts.{}", projection.suffix()),
                            projection.logical_shape(hidden, text.moe_intermediate_size),
                        )?;
                    }
                }
            }
        }

        rows.dense(FINAL_NORM, DType::BF16, vec![hidden])?;
        rows.dense(LM_HEAD, DType::BF16, vec![text.vocab_size, hidden])?;
        Ok(Self {
            text,
            dense: rows.dense,
            packed: rows.packed,
        })
    }

    pub const fn text(&self) -> &Glm5NextText {
        &self.text
    }

    pub fn layers(&self) -> &[Glm5NextTextLayerRow] {
        self.text.layers()
    }

    #[cfg(test)]
    pub(crate) fn dense_sources(&self) -> impl ExactSizeIterator<Item = &Glm5NextDenseSourceSpec> {
        self.dense.values()
    }

    /// Packed sources in layer, then role order.
    #[cfg(test)]
    pub(crate) fn packed_sources(
        &self,
    ) -> impl ExactSizeIterator<Item = &Glm5NextPackedSourceSpec> {
        self.packed.values()
    }

    pub fn state_pair_count(&self) -> usize {
        self.text.state_pair_count()
    }

    #[cfg(test)]
    fn dense(&self, name: &str) -> Result<&Glm5NextDenseSourceSpec, Glm53FlashTraceError> {
        self.dense
            .get(name)
            .ok_or_else(|| Glm53FlashTraceError::TextMissingDenseSource {
                name: name.to_string(),
            })
    }

    #[cfg(test)]
    fn packed(
        &self,
        layer: usize,
        role: Glm5NextPackedTextRole,
    ) -> Result<&Glm5NextPackedSourceSpec, Glm53FlashTraceError> {
        self.packed
            .get(&(layer, role))
            .ok_or(Glm53FlashTraceError::TextMissingPackedSource { layer, role })
    }

    /// Ordered Card 369 graph rows for one routed-expert projection of `layer`.
    #[cfg(test)]
    fn routed_rows(
        &self,
        layer: usize,
        projection: Glm5NextTextProjection,
    ) -> Result<Vec<PackedLinearGraphRow>, Glm53FlashTraceError> {
        (0..self.text.router.experts)
            .map(|expert| {
                let source = self.packed(
                    layer,
                    Glm5NextPackedTextRole::RoutedExpert { expert, projection },
                )?;
                Ok(PackedLinearGraphRow {
                    ordinal: expert,
                    linear_id: source.linear_id.clone(),
                    descriptor: source.descriptor,
                })
            })
            .collect()
    }

    #[cfg(test)]
    fn check_exact_counts(&self) -> Result<(), Glm53FlashTraceError> {
        let layers = self.layers();
        let kda_layers = layers
            .iter()
            .filter(|row| row.attention == Glm5NextTextAttentionKind::Kda)
            .count();
        let dense_ffn_layers = layers
            .iter()
            .filter(|row| row.ffn == Glm5NextTextFfnKind::Dense)
            .count();
        let dense_rows = |dtype| self.dense.values().filter(|row| row.dtype == dtype).count();
        let mut packed_roles = [0usize; 4];
        for row in self.packed.values() {
            packed_roles[match row.role {
                Glm5NextPackedTextRole::Dsa { .. } => 0,
                Glm5NextPackedTextRole::DenseFfn { .. } => 1,
                Glm5NextPackedTextRole::RoutedExpert { .. } => 2,
                Glm5NextPackedTextRole::SharedExpert { .. } => 3,
            }] += 1;
        }
        for (field, expected, actual) in [
            ("layers", GLM5NEXT_TEXT_LAYER_COUNT, layers.len()),
            ("KDA layers", GLM5NEXT_TEXT_KDA_LAYER_COUNT, kda_layers),
            (
                "dense FFN layers",
                GLM5NEXT_TEXT_DENSE_FFN_LAYER_COUNT,
                dense_ffn_layers,
            ),
            (
                "state pairs",
                GLM5NEXT_TEXT_STATE_PAIR_COUNT,
                self.state_pair_count(),
            ),
            (
                "dense sources",
                GLM5NEXT_TEXT_DENSE_SOURCE_COUNT,
                self.dense.len(),
            ),
            (
                "BF16 dense sources",
                GLM5NEXT_TEXT_BF16_SOURCE_COUNT,
                dense_rows(DType::BF16),
            ),
            (
                "F32 dense sources",
                GLM5NEXT_TEXT_F32_SOURCE_COUNT,
                dense_rows(DType::F32),
            ),
            (
                "packed sources",
                GLM5NEXT_TEXT_PACKED_SOURCE_COUNT,
                self.packed.len(),
            ),
            (
                "DSA packed sources",
                GLM5NEXT_TEXT_DSA_PACKED_COUNT,
                packed_roles[0],
            ),
            (
                "dense FFN packed sources",
                GLM5NEXT_TEXT_DENSE_FFN_PACKED_COUNT,
                packed_roles[1],
            ),
            (
                "routed expert packed sources",
                GLM5NEXT_TEXT_ROUTED_EXPERT_PACKED_COUNT,
                packed_roles[2],
            ),
            (
                "shared expert packed sources",
                GLM5NEXT_TEXT_SHARED_EXPERT_PACKED_COUNT,
                packed_roles[3],
            ),
        ] {
            if actual != expected {
                return Err(Glm53FlashTraceError::TextInventoryCount {
                    field,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
}

/// Source rows collected while a plan is derived. Every insert rejects a repeated key.
#[derive(Default)]
#[cfg(test)]
struct SourceRows {
    dense: BTreeMap<String, Glm5NextDenseSourceSpec>,
    packed: BTreeMap<(usize, Glm5NextPackedTextRole), Glm5NextPackedSourceSpec>,
}

#[cfg(test)]
impl SourceRows {
    fn dense(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Result<(), Glm53FlashTraceError> {
        let kind = match dtype {
            DType::BF16 => ExactSourceKind::Bf16,
            DType::F32 => ExactSourceKind::F32,
            dtype => {
                return Err(Glm53FlashTraceError::TextDenseSourceDtype {
                    name: name.to_string(),
                    dtype,
                });
            }
        };
        let row = Glm5NextDenseSourceSpec {
            name: name.to_string(),
            kind,
            dtype,
            shape,
        };
        if self.dense.insert(name.to_string(), row).is_some() {
            return Err(Glm53FlashTraceError::TextDuplicateSource {
                kind: Glm5NextTextSourceKind::Dense,
                name: name.to_string(),
            });
        }
        Ok(())
    }

    fn packed(
        &mut self,
        layer: usize,
        role: Glm5NextPackedTextRole,
        linear_id: &str,
        logical_shape: [usize; 2],
    ) -> Result<(), Glm53FlashTraceError> {
        let descriptor = PackedWeight::try_new(
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::F32,
            },
            logical_shape,
        )
        .map_err(|_| Glm53FlashTraceError::TextPackedDescriptor {
            linear_id: linear_id.to_string(),
            logical_shape,
        })?;
        let row = Glm5NextPackedSourceSpec {
            layer,
            role,
            linear_id: linear_id.to_string(),
            descriptor,
        };
        if self.packed.insert((layer, role), row).is_some() {
            return Err(Glm53FlashTraceError::TextDuplicateSource {
                kind: Glm5NextTextSourceKind::Packed,
                name: linear_id.to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Glm5NextTextOwnershipError {
    #[error("GLM-5.3-Flash owner artifact needs revision {expected}, got {actual}")]
    Revision {
        expected: &'static str,
        actual: String,
    },
    #[error("GLM-5.3-Flash owner plan received a non-exact classifier report: {actual:?}")]
    InventoryReport {
        actual: Glm53FlashTextInventoryReport,
    },
    #[error("GLM-5.3-Flash owner plan repeats {kind} source {name}")]
    DuplicateSource {
        kind: Glm5NextTextSourceKind,
        name: String,
    },
    #[error("GLM-5.3-Flash owner plan has unexpected {kind} source {name}")]
    UnexpectedSource {
        kind: Glm5NextTextSourceKind,
        name: String,
    },
    #[error("GLM-5.3-Flash owner plan is missing {kind} source {name}")]
    MissingSource {
        kind: Glm5NextTextSourceKind,
        name: String,
    },
    #[error("GLM-5.3-Flash packed owner {name} does not match the graph descriptor")]
    PackedDescriptor { name: String },
    #[error("GLM-5.3-Flash exact owner {name} has kind {actual:?}, expected {expected:?}")]
    DenseKind {
        name: String,
        expected: ExactSourceKind,
        actual: ExactSourceKind,
    },
    #[error("GLM-5.3-Flash exact owner {name} has shape {actual:?}, expected {expected:?}")]
    DenseShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// Model-local retention of the exact Card 359 owners used by the graph source plan.
///
/// Retained rows are clones of the loaded rows: payload owners are shared through their `Arc`s, while each row's
/// names, shard, spans, and descriptors are copied. This plan does not allocate an executor buffer, decode a
/// packed weight, or attach a source to a backend.
#[derive(Clone, Debug)]
pub struct Glm5NextTextOwnershipPlan {
    dense: Vec<LoadedExactSource>,
    packed: Vec<LoadedPackedLinear>,
}

impl Glm5NextTextOwnershipPlan {
    /// Retain exactly the sources `source_plan` names from one classifier load.
    ///
    /// Every retained name must be a plan row, and the plan has no layer-45 MTP or `model.visual.*` row, so a
    /// deferred or excluded tensor cannot be retained.
    pub fn from_loaded(
        source_plan: Glm5NextTextSourcePlan,
        loaded: &Glm53FlashTextLoadResult,
    ) -> Result<Self, Glm5NextTextOwnershipError> {
        let plan = Self::from_mixed(source_plan, loaded.mixed(), loaded.report())?;
        // The accessors are `cfg(test)`; this non-test adapter is the field read that keeps
        // the private payload live for `dead_code` (the tests destructure it directly).
        let _retained = (plan.dense.len(), plan.packed.len());
        Ok(plan)
    }

    /// Retain exactly the sources `source_plan` names from one mixed Card 359 load.
    ///
    /// `from_loaded` is the classifier-result wrapper; this lower-level entry is for callers that
    /// already hold a [`MixedLoadResult`] and its inventory report (Runner admission test stages,
    /// ownership-plan unit rows).
    pub(crate) fn from_mixed(
        source_plan: Glm5NextTextSourcePlan,
        mixed: &MixedLoadResult,
        report: Glm53FlashTextInventoryReport,
    ) -> Result<Self, Glm5NextTextOwnershipError> {
        if mixed.artifact.revision() != GLM53_FLASH_REVISION {
            return Err(Glm5NextTextOwnershipError::Revision {
                expected: GLM53_FLASH_REVISION,
                actual: mixed.artifact.revision().to_string(),
            });
        }
        if report != Glm53FlashTextInventoryReport::exact() {
            return Err(Glm5NextTextOwnershipError::InventoryReport { actual: report });
        }
        // Card 540a: `MixedLoadResult`'s own load output is its `store` plus
        // `exact_metadata`/`packed_metadata` (keyed by tensor name / linear id), not a parallel
        // `Vec` per row. This plan's own retained subset (fewer rows, kept by layer/role - see the
        // struct doc) still wants that per-row shape, so it is rebuilt here, locally, from the
        // store and metadata rather than the loader keeping a second copy of every row.
        let exact_rows: Vec<LoadedExactSource> = mixed
            .exact_metadata
            .values()
            .map(|meta| LoadedExactSource {
                descriptor: meta.owner.descriptor().clone(),
                kind: meta.kind,
                owner: Arc::clone(&meta.owner),
            })
            .collect();
        let packed_rows: Vec<LoadedPackedLinear> = mixed
            .packed_metadata
            .iter()
            .map(|(linear_id, meta)| {
                let owner = match mixed.store.get(linear_id) {
                    Some(WeightEntry::Packed(owner)) => Arc::clone(owner),
                    _ => unreachable!(
                        "packed_metadata and store are populated together, keyed by linear id"
                    ),
                };
                LoadedPackedLinear {
                    linear_id: linear_id.clone(),
                    descriptor: owner.weight(),
                    weight_name: meta.weight_name.clone(),
                    scale_name: meta.scale_name.clone(),
                    shard: meta.shard.clone(),
                    weight_span: meta.weight_span,
                    scale_span: meta.scale_span,
                    owner,
                }
            })
            .collect();
        let dense = retain_dense_sources(&source_plan, &exact_rows)?;
        let packed = retain_packed_sources(&source_plan, &packed_rows)?;
        Ok(Self { dense, packed })
    }

    #[cfg(test)]
    pub(crate) fn dense_sources(&self) -> &[LoadedExactSource] {
        &self.dense
    }

    #[cfg(test)]
    pub(crate) fn packed_sources(&self) -> &[LoadedPackedLinear] {
        &self.packed
    }

    #[cfg(test)]
    pub(crate) const fn portable_execution_admitted(&self) -> bool {
        false
    }

    #[cfg(test)]
    pub(crate) const fn public_capability(&self) -> bool {
        false
    }
}

/// Adapter from the classifier's load result into the ownership plan. The name is shared with
/// `poot_load`'s unrelated `from_loaded` helpers; this path keeps `Glm53FlashTextLoadResult` reachable
/// from non-test code.
pub fn from_loaded(
    source_plan: Glm5NextTextSourcePlan,
    loaded: &Glm53FlashTextLoadResult,
) -> Result<Glm5NextTextOwnershipPlan, Glm5NextTextOwnershipError> {
    Glm5NextTextOwnershipPlan::from_mixed(source_plan, loaded.mixed(), loaded.report())
}

fn retain_dense_sources(
    plan: &Glm5NextTextSourcePlan,
    loaded: &[LoadedExactSource],
) -> Result<Vec<LoadedExactSource>, Glm5NextTextOwnershipError> {
    let mut by_name = HashMap::with_capacity(loaded.len());
    for source in loaded {
        let name = source.descriptor.name();
        if !plan.dense.contains_key(name) {
            return Err(Glm5NextTextOwnershipError::UnexpectedSource {
                kind: Glm5NextTextSourceKind::Dense,
                name: name.to_string(),
            });
        }
        if by_name.insert(name, source).is_some() {
            return Err(Glm5NextTextOwnershipError::DuplicateSource {
                kind: Glm5NextTextSourceKind::Dense,
                name: name.to_string(),
            });
        }
    }
    let mut retained = Vec::with_capacity(plan.dense.len());
    for spec in plan.dense.values() {
        let source = by_name.get(spec.name.as_str()).ok_or_else(|| {
            Glm5NextTextOwnershipError::MissingSource {
                kind: Glm5NextTextSourceKind::Dense,
                name: spec.name.clone(),
            }
        })?;
        // Card 359 already rejected a row whose safetensors dtype differs from its kind.
        if source.kind != spec.kind {
            return Err(Glm5NextTextOwnershipError::DenseKind {
                name: spec.name.clone(),
                expected: spec.kind,
                actual: source.kind,
            });
        }
        if source.descriptor.shape() != spec.shape.as_slice() {
            return Err(Glm5NextTextOwnershipError::DenseShape {
                name: spec.name.clone(),
                expected: spec.shape.clone(),
                actual: source.descriptor.shape().to_vec(),
            });
        }
        retained.push((*source).clone());
    }
    Ok(retained)
}

fn retain_packed_sources(
    plan: &Glm5NextTextSourcePlan,
    loaded: &[LoadedPackedLinear],
) -> Result<Vec<LoadedPackedLinear>, Glm5NextTextOwnershipError> {
    let specs = plan
        .packed
        .values()
        .map(|spec| (spec.linear_id.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let mut by_name = HashMap::with_capacity(loaded.len());
    for source in loaded {
        let name = source.linear_id.as_str();
        if !specs.contains_key(name) {
            return Err(Glm5NextTextOwnershipError::UnexpectedSource {
                kind: Glm5NextTextSourceKind::Packed,
                name: name.to_string(),
            });
        }
        if by_name.insert(name, source).is_some() {
            return Err(Glm5NextTextOwnershipError::DuplicateSource {
                kind: Glm5NextTextSourceKind::Packed,
                name: name.to_string(),
            });
        }
    }
    let mut retained = Vec::with_capacity(plan.packed.len());
    for spec in plan.packed.values() {
        let source = by_name.get(spec.linear_id.as_str()).ok_or_else(|| {
            Glm5NextTextOwnershipError::MissingSource {
                kind: Glm5NextTextSourceKind::Packed,
                name: spec.linear_id.clone(),
            }
        })?;
        if source.descriptor != spec.descriptor
            || source.owner.weight() != spec.descriptor
            || !is_component_name(&source.weight_name, &spec.linear_id, ".weight")
            || !is_component_name(&source.scale_name, &spec.linear_id, ".weight_scale_inv")
        {
            return Err(Glm5NextTextOwnershipError::PackedDescriptor {
                name: spec.linear_id.clone(),
            });
        }
        retained.push((*source).clone());
    }
    Ok(retained)
}

fn is_component_name(name: &str, linear_id: &str, suffix: &str) -> bool {
    name.strip_prefix(linear_id) == Some(suffix)
}

#[cfg(test)]
fn packed_projection(
    b: &Builder,
    x: Traced,
    source: &Glm5NextPackedSourceSpec,
) -> Result<Traced, Glm53FlashTraceError> {
    packed_linear(b, x, &source.linear_id, source.descriptor, None, None)
        .map_err(Glm53FlashTraceError::TextGraph)
}

#[cfg(test)]
fn collapse_stream_mean(b: &Builder, streams: Traced, hc: usize) -> Traced {
    let streams = b.transpose(streams, vec![0, 1, 3, 2]);
    b.binary_scalar(
        BinOp::Mul,
        b.reduce(RedOp::Sum, streams, 3, false),
        Scalar::F32(1.0 / hc as f32),
    )
}

/// GLM5Next's selection-only-bias sigmoid router, shared by the exact stack and bounded CPU tests.
///
/// Returns repeated inputs, rank-ordered expert ids, and normalized unbiased weights for the selected rows of
/// one `[1, hidden]` token.
#[cfg(test)]
pub(crate) fn glm5next_router_routes(
    b: &Builder,
    x: Traced,
    logits: Traced,
    correction_bias: Traced,
    router: Glm5NextRouterConfig,
) -> Result<(Traced, Traced, Traced), Glm53FlashTraceError> {
    let Glm5NextRouterConfig {
        experts,
        top_k,
        n_group,
        topk_group,
        routed_scale,
    } = router;
    for (field, requirement, actual, valid) in [
        ("experts", "nonzero", experts, experts > 0),
        (
            "top_k",
            "between 1 and experts",
            top_k,
            (1..=experts).contains(&top_k),
        ),
        (
            "n_group",
            "a nonzero divisor of experts",
            n_group,
            n_group > 0 && experts.is_multiple_of(n_group),
        ),
        (
            "topk_group",
            "between 1 and n_group",
            topk_group,
            (1..=n_group).contains(&topk_group),
        ),
    ] {
        if !valid {
            return Err(Glm53FlashTraceError::TextRouterConfig {
                field,
                requirement,
                actual,
            });
        }
    }
    let x_type = b.aval(x);
    let hidden = match x_type.shape.as_slice() {
        &[1, hidden] if x_type.dtype == DType::F32 => Some(hidden),
        _ => None,
    };
    let Some(hidden) = hidden else {
        return Err(Glm53FlashTraceError::TextRouterInput {
            role: "input",
            actual: x_type,
        });
    };
    for (role, value, expected) in [
        ("logits", logits, TensorType::f32(vec![1, experts])),
        (
            "correction bias",
            correction_bias,
            TensorType::f32(vec![experts]),
        ),
    ] {
        let actual = b.aval(value);
        if actual != expected {
            return Err(Glm53FlashTraceError::TextRouterInput { role, actual });
        }
    }
    let (dense_weights, ids) = deepseek3_router_gate_and_ids(
        b,
        logits,
        correction_bias,
        top_k,
        n_group,
        topk_group,
        routed_scale,
    );
    let ids = b.reshape(ids, vec![top_k]);
    let weights = b.reshape(b.gather(dense_weights, 1, ids), vec![top_k]);
    let repeated = b.broadcast(x, vec![top_k, hidden]);
    Ok((repeated, ids, weights))
}

/// Trace the one-token GLM-5.3-Flash main-text graph for `sources`.
///
/// Card 367 bounds `capacity` to its device-free pairwise selection limit. This function constructs graph
/// semantics and source names only; it does not bind a portable executor or admit production capacity.
#[cfg(test)]
pub(crate) fn trace_glm5next_full_text(
    sources: &Glm5NextTextSourcePlan,
    scope: Glm5NextTextScope,
    capacity: usize,
) -> Result<Graph, Glm53FlashTraceError> {
    // Reject a deferred scope or an unsupported state capacity before any graph append.
    if scope != Glm5NextTextScope::Text {
        return Err(Glm53FlashTraceError::TextUnsupportedScope { scope });
    }
    glm5next_kda_state_specs(sources.text.kda, 1)?;
    glm5next_dsa_state_specs(sources.text.dsa, capacity, 1)?;
    let mut trace = Glm5NextTextTrace::new(sources, capacity);
    let mut streams = trace.embed()?;
    for (layer, &row) in sources.layers().iter().enumerate() {
        let _layer_scope = trace.b.layer_scope(layer);
        streams = trace.text_layer(row, streams)?;
    }
    let logits = trace.text_head(streams)?;
    Ok(trace.b.finish_with_state(logits, &trace.state))
}

/// One text graph under construction.
#[cfg(test)]
struct Glm5NextTextTrace<'a> {
    b: Builder,
    plan: &'a Glm5NextTextSourcePlan,
    capacity: usize,
    token: Traced,
    position: Traced,
    query_validity: Traced,
    visibility: Traced,
    state: Vec<(Traced, Traced)>,
}

#[cfg(test)]
impl<'a> Glm5NextTextTrace<'a> {
    fn new(plan: &'a Glm5NextTextSourcePlan, capacity: usize) -> Self {
        let b = Builder::new();
        let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
        let position = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
        let query_validity = b.slot_named(
            Slot::Activation,
            GLM5NEXT_QUERY_VALIDITY_TAG,
            TensorType::f32(vec![1, 1]),
        );
        let visibility = b.slot_named(
            Slot::Mask,
            GLM5NEXT_VISIBILITY_TAG,
            TensorType::f32(vec![1, capacity]),
        );
        Self {
            b,
            plan,
            capacity,
            token,
            position,
            query_validity,
            visibility,
            state: Vec::with_capacity(plan.state_pair_count()),
        }
    }

    fn dense(&self, name: &str) -> Result<Traced, Glm53FlashTraceError> {
        let spec = self.plan.dense(name)?;
        Ok(self
            .b
            .constant(name, TensorType::new(spec.shape.clone(), spec.dtype)))
    }

    fn dense_f32(&self, name: &str) -> Result<Traced, Glm53FlashTraceError> {
        Ok(self.b.cast(self.dense(name)?, DType::F32))
    }

    fn exact_linear(&self, x: Traced, name: &str) -> Result<Traced, Glm53FlashTraceError> {
        Ok(dense_source_linear(&self.b, x, self.dense(name)?))
    }

    fn norm(&self, x: Traced, name: &str) -> Result<Traced, Glm53FlashTraceError> {
        Ok(rmsnorm(
            &self.b,
            x,
            self.dense_f32(name)?,
            self.plan.text.rms_norm_eps,
        ))
    }

    /// Expand the BF16 token embedding into identical mHC streams `[1, 1, streams, hidden]`.
    fn embed(&self) -> Result<Traced, Glm53FlashTraceError> {
        let text = &self.plan.text;
        let embedding = self.dense(EMBEDDING)?;
        let embedded = self
            .b
            .cast(self.b.gather(embedding, 0, self.token), DType::F32);
        let embedded = self.b.reshape(embedded, vec![1, 1, 1, text.hidden_size]);
        Ok(self
            .b
            .broadcast(embedded, vec![1, 1, text.mhc.streams, text.hidden_size]))
    }

    fn text_layer(
        &mut self,
        row: Glm5NextTextLayerRow,
        streams: Traced,
    ) -> Result<Traced, Glm53FlashTraceError> {
        let prefix = layer_prefix(row.layer);
        let (post, comb, collapsed) = self.mhc_map(streams, &prefix, MhcSite::Attention)?;
        let input = self.norm(collapsed, &input_norm_name(&prefix))?;
        let attention_prefix = format!("{prefix}.self_attn");
        let output = match row.attention {
            Glm5NextTextAttentionKind::Kda => self.kda_attention(&attention_prefix, input)?,
            Glm5NextTextAttentionKind::Dsa => {
                self.dsa_attention(row.layer, &attention_prefix, input)?
            }
        };
        let streams = self.mhc_combine(post, comb, output, streams);

        let (post, comb, collapsed) = self.mhc_map(streams, &prefix, MhcSite::Ffn)?;
        let input = self.norm(collapsed, &post_attention_norm_name(&prefix))?;
        let output = match row.ffn {
            Glm5NextTextFfnKind::Dense => self.packed_mlp(row.layer, input, |projection| {
                Glm5NextPackedTextRole::DenseFfn { projection }
            })?,
            Glm5NextTextFfnKind::Sparse => self.sparse_ffn(row.layer, &prefix, input)?,
        };
        Ok(self.mhc_combine(post, comb, output, streams))
    }

    /// Map widened streams through one mHC site's own projection, base, and scale rows.
    fn mhc_map(
        &self,
        streams: Traced,
        prefix: &str,
        site: MhcSite,
    ) -> Result<(Traced, Traced, Traced), Glm53FlashTraceError> {
        let text = &self.plan.text;
        let [projection, base, scale] = site.source_names(prefix);
        let projection = self.b.transpose(self.dense_f32(&projection)?, vec![1, 0]);
        Ok(hyper_connection(
            &self.b,
            streams,
            projection,
            self.dense(&base)?,
            self.dense(&scale)?,
            text.mhc.streams,
            text.hidden_size,
            1,
            text.rms_norm_eps,
            text.mhc.eps,
            text.mhc.sinkhorn_iters,
        ))
    }

    fn mhc_combine(&self, post: Traced, comb: Traced, output: Traced, residual: Traced) -> Traced {
        let text = &self.plan.text;
        hyper_connection_combine(
            &self.b,
            post,
            comb,
            output,
            residual,
            text.mhc.streams,
            1,
            text.hidden_size,
        )
    }

    fn kda_attention(&mut self, prefix: &str, x: Traced) -> Result<Traced, Glm53FlashTraceError> {
        let cfg = self.plan.text.kda;
        let [conv_state, recurrent_state] = glm5next_kda_state_specs(cfg, 1)?.map(|spec| {
            self.b.state_input(
                &format!("{prefix}.{}", spec.suffix),
                spec.ty,
                StateRole::Recurrent,
            )
        });
        let output = glm5next_kda_block(
            &self.b,
            GLM53_FLASH_REVISION,
            prefix,
            Glm5NextKdaScope::Kda,
            cfg,
            Glm5NextKdaInputs {
                x,
                validity: Some(self.query_validity),
                conv_state,
                recurrent_state,
            },
        )?;
        self.state.extend([
            (conv_state, output.conv_state),
            (recurrent_state, output.recurrent_state),
        ]);
        Ok(output.y)
    }

    fn dsa_attention(
        &mut self,
        layer: usize,
        prefix: &str,
        x: Traced,
    ) -> Result<Traced, Glm53FlashTraceError> {
        let cfg = self.plan.text.dsa;
        let [k_state, v_state, indexer_state] = glm5next_dsa_state_specs(cfg, self.capacity, 1)?
            .map(|spec| {
                self.b.state_input(
                    &format!("{prefix}.{}", spec.suffix),
                    spec.ty,
                    StateRole::Recurrent,
                )
            });
        let plan = self.plan;
        let (output, _) = glm5next_dsa_block_with_projections(
            &self.b,
            GLM53_FLASH_REVISION,
            prefix,
            Glm5NextDsaScope::Dsa,
            cfg,
            self.capacity,
            Glm5NextDsaInputs {
                x,
                query_validity: self.query_validity,
                visibility: self.visibility,
                position: self.position,
                k_state,
                v_state,
                indexer_state,
            },
            |b, role, x| {
                let source = plan.packed(layer, Glm5NextPackedTextRole::Dsa { role })?;
                packed_projection(b, x, source)
            },
        )?;
        self.state.extend([
            (k_state, output.k_state),
            (v_state, output.v_state),
            (indexer_state, output.indexer_state),
        ]);
        Ok(output.y)
    }

    /// Clamped SwiGLU through three packed projections, used by dense layers and the shared expert.
    fn packed_mlp(
        &self,
        layer: usize,
        x: Traced,
        role: fn(Glm5NextTextProjection) -> Glm5NextPackedTextRole,
    ) -> Result<Traced, Glm53FlashTraceError> {
        let source = |projection| self.plan.packed(layer, role(projection));
        let gate = packed_projection(&self.b, x, source(Glm5NextTextProjection::Gate)?)?;
        let up = packed_projection(&self.b, x, source(Glm5NextTextProjection::Up)?)?;
        let activated = deepseek4_clamped_swiglu(&self.b, gate, up, self.plan.text.swiglu_limit);
        packed_projection(&self.b, activated, source(Glm5NextTextProjection::Down)?)
    }

    fn sparse_ffn(
        &self,
        layer: usize,
        prefix: &str,
        x: Traced,
    ) -> Result<Traced, Glm53FlashTraceError> {
        let text = &self.plan.text;
        let b = &self.b;
        let flat = b.reshape(x, vec![1, text.hidden_size]);
        let logits = self.exact_linear(flat, &router_gate_name(prefix))?;
        let bias = self.dense(&router_bias_name(prefix))?;
        let (repeated, ids, weights) = glm5next_router_routes(b, flat, logits, bias, text.router)?;
        let gate_rows = self.plan.routed_rows(layer, Glm5NextTextProjection::Gate)?;
        let up_rows = self.plan.routed_rows(layer, Glm5NextTextProjection::Up)?;
        let down_rows = self.plan.routed_rows(layer, Glm5NextTextProjection::Down)?;
        let gate = packed_indexed_linear(b, repeated, ids, &gate_rows)?;
        let up = packed_indexed_linear(b, repeated, ids, &up_rows)?;
        let activated = deepseek4_clamped_swiglu(b, gate, up, text.swiglu_limit);
        let routed = packed_indexed_linear(b, activated, ids, &down_rows)?;
        let routed = b.binary(
            BinOp::Mul,
            routed,
            b.reshape(weights, vec![text.router.top_k, 1]),
        );
        let routed = b.reshape(
            b.reduce(RedOp::Sum, routed, 0, false),
            vec![1, 1, text.hidden_size],
        );
        let shared = self.packed_mlp(layer, x, |projection| {
            Glm5NextPackedTextRole::SharedExpert { projection }
        })?;
        Ok(b.binary(BinOp::Add, routed, shared))
    }

    /// Collapse the streams by unweighted mean, then apply the final RMSNorm and LM head.
    fn text_head(&self, streams: Traced) -> Result<Traced, Glm53FlashTraceError> {
        let hidden = collapse_stream_mean(&self.b, streams, self.plan.text.mhc.streams);
        let hidden = self.norm(hidden, FINAL_NORM)?;
        self.exact_linear(hidden, LM_HEAD)
    }
}

#[cfg(test)]
mod tests;
