//! DeepSeek-V2 behind [`Model`], and the MLA-plus-routed-MoE body its V3 and V3.2 relatives share.
//!
//! A DeepSeek layer is the standard pre-norm pair ([`standard_layer`]) whose mixer is multi-head
//! latent attention ([`mla`]) and whose feed-forward is a dense SwiGLU for the leading
//! `first_k_dense_replace` layers and a routed mixture with an always-active shared expert after
//! them ([`moe`]). The families differ in config keys and router:
//!
//! - `deepseek_v2` (HF) and `deepseek2` (GGUF): softmax scoring with group-limited greedy routing.
//!   llama.cpp's `deepseek2` also names V3 checkpoints; the gating function the GGUF declares
//!   (`expert_gating_func`) chooses the router, as `scoring_func` does for an HF config.
//! - `deepseek_v3` ([`super::deepseek3`]): sigmoid scoring with a selection-only bias.
//! - `deepseek_v32` ([`super::deepseek32`]): V3 plus the sparse-attention indexer ([`dsa`]).
//!
//! RoPE is DeepSeek's interleaved pairing over the rotated slice, with YaRN scaling whose
//! `mscale` terms are resolved here: the table's magnitude (`attention_factor`) and a separate
//! softmax-scale correction (`mscale_all_dim`) are different scalars.
//!
//! The up projection binds as stored: HF's one `kv_b_proj`, or llama.cpp's split per-head `k_b` and
//! `v_b` (no rebuilt buffer).

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::rope_table::RopeFlavor;
use poot_graph_ir::{Builder, Graph, ValidationOutputs};
use poot_quant::weights::{
    AttnRole, DsaRole, ExpertProj, FfnRole, MlaRole, NormRole, WeightMap, WeightRole, WeightStore,
};

use crate::chat::ChatFormat;
use crate::components::dims::{Field, Fields};
use crate::components::dsa::{DsaParams, DsaWeights, dsa_select};
use crate::components::ffn::{GatedFfnWeights, gated_ffn};
use crate::components::linear::WeightError;
use crate::components::mla::{KvForm, MlaParams, MlaWeights, Selection, mla};
use crate::components::moe::{self, MoeParams, MoeWeights};
use crate::components::norm::NormParams;
use crate::components::rope::RopeParams;
use crate::components::router::{RouterBias, RouterKind, SigmoidTopK};
use crate::components::standard::{
    BlockInput, LayerNorms, LayerParams, StackParams, StackWeights, Step, check_step,
    standard_layer, standard_stack,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use crate::names::{
    ExpertNames, MoeLayers, MoeNames, NameRow, Presence, SharedNames, Source, family_weights_routed,
};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

pub const FAMILY: FamilyKey = FamilyKey::new("deepseek2");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "deepseek_v2"),
        (ConfigSource::GgufArchitecture, "deepseek2"),
    ],
    build,
    fixture,
    variants: &[],
};

/// How a router scores the experts before it selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scoring {
    Softmax,
    Sigmoid,
}

/// What a DeepSeek family fixes about its configs: its name, the model type's default router and
/// whether it carries the sparse-attention indexer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Kind {
    pub family: FamilyKey,
    pub scoring: Scoring,
    pub dsa: bool,
}

const V2: Kind = Kind {
    family: FAMILY,
    scoring: Scoring::Softmax,
    dsa: false,
};

/// How the routed experts are chosen and weighted: the router's scoring, the group limit and the
/// scale on the selected weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Routing {
    pub scoring: Scoring,
    pub groups: usize,
    pub group_top: usize,
    pub scale: f32,
    pub renormalize: bool,
}

/// `moe` with the group limit and the routed scale `routing` names. A single group (V2-Lite's greedy
/// selection) and a scale of `1.0` are the identity, so they add no option and the traced router
/// carries no group-eligibility ops and no multiply by one.
fn with_routing(
    moe: MoeParams,
    routing: &Routing,
) -> Result<MoeParams, crate::components::standard::ParamError> {
    let moe = match routing.scoring {
        Scoring::Softmax => moe,
        Scoring::Sigmoid => moe.with_router_bias(RouterBias::SelectionOnly)?,
    };
    let moe = if routing.groups > 1 {
        moe.with_group_limit(routing.groups, routing.group_top)?
    } else {
        moe
    };
    if routing.scale == 1.0 {
        Ok(moe)
    } else {
        moe.with_routed_scale(routing.scale)
    }
}

/// The typed config of a DeepSeek MLA family: every value checked once, here, so tracing cannot
/// fail on it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DeepseekConfig {
    pub vocab: usize,
    pub width: usize,
    pub layers: usize,
    pub max_positions: usize,
    pub mla: MlaParams,
    pub layer: LayerParams,
    pub stack: StackParams,
    pub dense_inter: usize,
    /// Layers below this index use the dense feed-forward, the rest route.
    pub first_routed: usize,
    pub moe: MoeParams,
    pub routing: Routing,
    pub dsa: Option<DsaParams>,
    pub head_required: bool,
    pub eos: BTreeSet<u32>,
    pub bos: Option<u32>,
    pub kv: KvForm,
    pub shared: bool,
}

const fn field(hf: &'static str, gguf: &'static str) -> Field {
    Field::new(hf, gguf)
}

const HEADS: Field = field("num_attention_heads", "deepseek2.attention.head_count");
const Q_RANK: Field = field("q_lora_rank", "deepseek2.attention.q_lora_rank");
const KV_RANK: Field = field("kv_lora_rank", "deepseek2.attention.kv_lora_rank");
const NOPE: Field = field("qk_nope_head_dim", "");
const ROPE_DIM: Field = field("qk_rope_head_dim", "deepseek2.rope.dimension_count");
const KEY_LENGTH: Field = field("", "deepseek2.attention.key_length_mla");
const V_DIM: Field = field("v_head_dim", "deepseek2.attention.value_length_mla");
const WIDTH: Field = field("hidden_size", "deepseek2.embedding_length");
const LAYERS: Field = field("num_hidden_layers", "deepseek2.block_count");
const EPS: Field = field("rms_norm_eps", "deepseek2.attention.layer_norm_rms_epsilon");
const MAX_POSITIONS: Field = field("max_position_embeddings", "deepseek2.context_length");
const THETA: Field = field("rope_theta", "deepseek2.rope.freq_base");
const DENSE_INTER: Field = field("intermediate_size", "deepseek2.feed_forward_length");
const MOE_INTER: Field = field(
    "moe_intermediate_size",
    "deepseek2.expert_feed_forward_length",
);
const EXPERTS: Field = field("n_routed_experts", "deepseek2.expert_count");
const SHARED: Field = field("n_shared_experts", "deepseek2.expert_shared_count");
const TOP_K: Field = field("num_experts_per_tok", "deepseek2.expert_used_count");
const FIRST_DENSE: Field = field(
    "first_k_dense_replace",
    "deepseek2.leading_dense_block_count",
);
const GROUPS: Field = field("n_group", "deepseek2.expert_group_count");
const GROUP_TOP: Field = field("topk_group", "deepseek2.expert_group_used_count");
const SCALE: Field = field("routed_scaling_factor", "deepseek2.expert_weights_scale");
const NORM_TOPK: Field = field("norm_topk_prob", "deepseek2.expert_weights_norm");
const INDEX_HEADS: Field = Field::hf("index_n_heads");
const INDEX_DIM: Field = Field::hf("index_head_dim");
const INDEX_TOPK: Field = Field::hf("index_topk");

/// `llama.cpp`'s `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` and `_SIGMOID`.
const GATING_SOFTMAX: u64 = 1;
const GATING_SIGMOID: u64 = 2;

impl DeepseekConfig {
    pub(crate) fn from_raw(raw: &RawConfig<'_>, kind: Kind) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, kind.family);
        let gguf = matches!(raw, RawConfig::Gguf(_));
        let heads = fields.count(HEADS, None)?;
        let kv_rank = fields.count(KV_RANK, None)?;
        let q_rank = match fields.opt_u64(Q_RANK)? {
            None => None,
            Some(r) => Some(
                usize::try_from(r)
                    .map_err(|_| fields.error("q_lora_rank", ConfigReason::WrongType))?,
            ),
        };
        let rope_dim = fields.count(ROPE_DIM, None)?;
        // A GGUF states the whole per-head query width, not the unrotated part.
        let nope = if gguf {
            fields
                .count(KEY_LENGTH, None)?
                .checked_sub(rope_dim)
                .filter(|&n| n > 0)
                .ok_or(fields.error(fields.key(KEY_LENGTH), ConfigReason::Zero))?
        } else {
            fields.count(NOPE, None)?
        };
        let v_dim = fields.count(V_DIM, None)?;
        let width = fields.count(WIDTH, None)?;
        let layers = fields.count(LAYERS, None)?;
        let max_positions = fields.count(MAX_POSITIONS, None)?;
        if u32::try_from(max_positions).is_err() {
            return Err(fields.error(
                fields.key(MAX_POSITIONS),
                ConfigReason::Exceeds {
                    max: u32::MAX as usize,
                },
            ));
        }
        let eps = fields.float(EPS, 1e-6)?;
        let theta = fields.float(THETA, 10_000.0)?;
        if fields.flag(Field::hf("attention_bias"), false)? {
            return Err(fields.error("attention_bias", ConfigReason::Unsupported));
        }
        if fields
            .opt_u64(Field::hf("moe_layer_freq"))?
            .is_some_and(|f| f != 1)
        {
            return Err(fields.error("moe_layer_freq", ConfigReason::Unsupported));
        }

        let (flavor, softmax_scale_sq) = yarn(&fields, max_positions)?;
        let norm = NormParams::new(eps).map_err(|e| e.for_family(kind.family))?;
        let rope =
            RopeParams::new(rope_dim, theta, &flavor).map_err(|e| e.for_family(kind.family))?;
        let qk = nope + rope_dim;
        let scale = softmax_scale_sq / (qk as f32).sqrt();
        let kv = if gguf { KvForm::Split } else { KvForm::Fused };
        let mla = MlaParams::new(
            heads,
            kv_rank,
            q_rank,
            nope,
            rope_dim,
            v_dim,
            norm,
            rope.interleaved(),
            scale,
            kv,
        )
        .map_err(|e| e.for_family(kind.family))?;

        let dsa = if kind.dsa {
            if q_rank.is_none() {
                return Err(fields.error("q_lora_rank", ConfigReason::Missing));
            }
            Some(
                DsaParams::new(
                    fields.count(INDEX_HEADS, None)?,
                    fields.count(INDEX_DIM, None)?,
                    fields.count(INDEX_TOPK, None)?,
                    rope,
                )
                .map_err(|e| e.for_family(kind.family))?,
            )
        } else {
            None
        };

        let routing = routing(&fields, kind)?;
        let experts = fields.count(EXPERTS, None)?;
        let moe_inter = fields.count(MOE_INTER, None)?;
        let n_shared = fields.opt_u64(SHARED)?.map_or(Ok(0), |n| {
            usize::try_from(n)
                .map_err(|_| fields.error(fields.key(SHARED), ConfigReason::WrongType))
        })?;
        let mut moe = with_routing(
            MoeParams::new(
                kind.family,
                width,
                moe_inter,
                experts,
                fields.count(TOP_K, None)?,
                router_kind(&routing),
            )?,
            &routing,
        )
        .map_err(|e| e.for_family(kind.family))?;
        if n_shared > 0 {
            moe = moe
                .with_shared(moe_inter * n_shared)
                .map_err(|e| e.for_family(kind.family))?;
        }
        let first_routed = fields
            .opt_u64(FIRST_DENSE)?
            .map_or(Ok(0), |n| {
                usize::try_from(n)
                    .map_err(|_| fields.error(fields.key(FIRST_DENSE), ConfigReason::WrongType))
            })?
            .min(layers);
        let dense_inter = if first_routed > 0 {
            fields.count(DENSE_INTER, None)?
        } else {
            0
        };
        let head_required = match raw {
            RawConfig::HfJson { config, .. } => !config
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            RawConfig::Gguf(_) => false,
        };
        Ok(Self {
            vocab: fields.vocab()?,
            width,
            layers,
            max_positions,
            mla,
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            dense_inter,
            first_routed,
            moe,
            routing,
            dsa,
            head_required,
            eos: fields.eos(None)?,
            bos: fields.bos()?,
            kv,
            shared: n_shared > 0,
        })
    }

    fn routed(&self, layer: usize) -> bool {
        layer >= self.first_routed
    }
}

/// The router a config declares. HF names the scoring function (`softmax` or `sigmoid`) and the
/// top-k method (`greedy`, `group_limited_greedy` or `noaux_tc`); a GGUF names the gating function
/// and the group counts (a `deepseek2` GGUF also holds V3 checkpoints). A router that normalizes
/// the selected weights does so only when asked: `norm_topk_prob`, `expert_weights_norm`.
fn routing(fields: &Fields<'_>, kind: Kind) -> Result<Routing, ModelError> {
    let scale = fields.float(SCALE, 1.0)?;
    let renormalize = fields.flag(NORM_TOPK, kind.scoring == Scoring::Sigmoid)?;
    let count = |field: Field, default: usize| -> Result<usize, ModelError> {
        fields.opt_u64(field)?.map_or(Ok(default), |n| {
            usize::try_from(n).map_err(|_| fields.error(fields.key(field), ConfigReason::WrongType))
        })
    };
    let (scoring, grouped) = match fields.raw() {
        RawConfig::HfJson { config, .. } => {
            let text = |key: &'static str| {
                config
                    .get(key)
                    .filter(|v| !v.is_null())
                    .map(|v| v.as_str().ok_or(fields.error(key, ConfigReason::WrongType)))
                    .transpose()
            };
            let scoring = match text("scoring_func")? {
                None => kind.scoring,
                Some("softmax") => Scoring::Softmax,
                Some("sigmoid") => Scoring::Sigmoid,
                Some(_) => return Err(fields.error("scoring_func", ConfigReason::Unsupported)),
            };
            let grouped = match text("topk_method")? {
                None | Some("group_limited_greedy" | "noaux_tc") => true,
                Some("greedy") => false,
                Some(_) => return Err(fields.error("topk_method", ConfigReason::Unsupported)),
            };
            (scoring, grouped)
        }
        RawConfig::Gguf(_) => {
            let scoring = match fields.opt_u64(Field::new("", "deepseek2.expert_gating_func"))? {
                None => kind.scoring,
                Some(GATING_SOFTMAX) => Scoring::Softmax,
                Some(GATING_SIGMOID) => Scoring::Sigmoid,
                Some(_) => {
                    return Err(
                        fields.error("deepseek2.expert_gating_func", ConfigReason::Unsupported)
                    );
                }
            };
            (scoring, true)
        }
    };
    let groups = if grouped { count(GROUPS, 1)? } else { 1 };
    let group_top = if grouped {
        count(GROUP_TOP, groups)?
    } else {
        1
    };
    if groups == 0 {
        return Err(fields.error(fields.key(GROUPS), ConfigReason::Zero));
    }
    if group_top == 0 || group_top > groups {
        return Err(fields.error(
            fields.key(GROUP_TOP),
            if group_top == 0 {
                ConfigReason::Zero
            } else {
                ConfigReason::Exceeds { max: groups }
            },
        ));
    }
    Ok(Routing {
        scoring,
        groups,
        group_top,
        scale,
        renormalize,
    })
}

/// The router `routing` names: the softmax of V2, or V3's sigmoid with a selection-only bias.
fn router_kind(routing: &Routing) -> RouterKind {
    match routing.scoring {
        Scoring::Softmax => RouterKind::SoftmaxTopK {
            renormalize: routing.renormalize,
        },
        Scoring::Sigmoid => {
            RouterKind::SigmoidBiasedTopK(SigmoidTopK::new().with_renormalize(routing.renormalize))
        }
    }
}

/// DeepSeek's YaRN: the RoPE flavor and the extra softmax-scale factor `get_mscale(factor,
/// mscale_all_dim)^2` (`1.0` when the config has no `mscale_all_dim`), read from HF's
/// `rope_scaling` or a GGUF's `rope.scaling.*` keys. Absent or `default` scaling is plain RoPE and
/// a factor of one; any other type is refused.
fn yarn(
    fields: &Fields<'_>,
    max_positions: usize,
) -> Result<(RopeFlavor<'static>, f32), ModelError> {
    struct Scaling {
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original: usize,
        attention_factor: Option<f32>,
        mscale: Option<f32>,
        mscale_all_dim: Option<f32>,
    }
    let scaling = match fields.raw() {
        RawConfig::HfJson { config, .. } => {
            let Some(block) = config.get("rope_scaling").filter(|v| !v.is_null()) else {
                return Ok((RopeFlavor::Plain, 1.0));
            };
            let wrong = |field| fields.error(field, ConfigReason::WrongType);
            let block = block.as_object().ok_or(wrong("rope_scaling"))?;
            let kind = block
                .get("rope_type")
                .or_else(|| block.get("type"))
                .ok_or(fields.error("rope_scaling.rope_type", ConfigReason::Missing))?
                .as_str()
                .ok_or(wrong("rope_scaling.rope_type"))?;
            match kind {
                "default" => return Ok((RopeFlavor::Plain, 1.0)),
                "yarn" => {}
                _ => return Err(fields.error("rope_scaling.rope_type", ConfigReason::Unsupported)),
            }
            let float = |key: &'static str| {
                block
                    .get(key)
                    .filter(|v| !v.is_null())
                    .map(|v| v.as_f64().map(|v| v as f32).ok_or(wrong(key)))
                    .transpose()
            };
            let original = block
                .get("original_max_position_embeddings")
                .filter(|v| !v.is_null())
                .map(|v| {
                    v.as_u64()
                        .and_then(|v| usize::try_from(v).ok())
                        .ok_or(wrong("rope_scaling.original_max_position_embeddings"))
                })
                .transpose()?
                .unwrap_or(max_positions);
            Scaling {
                factor: float("factor")?
                    .ok_or(fields.error("rope_scaling.factor", ConfigReason::Missing))?,
                beta_fast: float("beta_fast")?.unwrap_or(32.0),
                beta_slow: float("beta_slow")?.unwrap_or(1.0),
                original,
                attention_factor: float("attention_factor")?,
                mscale: float("mscale")?,
                mscale_all_dim: float("mscale_all_dim")?,
            }
        }
        RawConfig::Gguf(gguf) => {
            let kind = gguf
                .get("deepseek2.rope.scaling.type")
                .and_then(|v| v.as_str());
            if kind != Some("yarn") {
                return Ok((RopeFlavor::Plain, 1.0));
            }
            let float = |key: &'static str| -> Result<Option<f32>, ModelError> {
                gguf.get(key)
                    .map(|v| v.as_f32().ok_or(fields.error(key, ConfigReason::WrongType)))
                    .transpose()
            };
            let original_key = "deepseek2.rope.scaling.original_context_length";
            let original = gguf
                .get(original_key)
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or(fields.error(original_key, ConfigReason::WrongType))
                })
                .transpose()?
                .unwrap_or(max_positions);
            // The GGUF carries one multiplier, `0.1 * mscale_all_dim`, and cannot say `mscale`
            // separately: the converter wrote both from the same value.
            let mscale_all_dim =
                float("deepseek2.rope.scaling.yarn_log_multiplier")?.map(|v| v / 0.1);
            Scaling {
                factor: float("deepseek2.rope.scaling.factor")?.unwrap_or(1.0),
                beta_fast: float("deepseek2.rope.scaling.yarn_beta_fast")?.unwrap_or(32.0),
                beta_slow: float("deepseek2.rope.scaling.yarn_beta_slow")?.unwrap_or(1.0),
                original,
                attention_factor: float("deepseek2.rope.scaling.attn_factor")?,
                mscale: mscale_all_dim,
                mscale_all_dim,
            }
        }
    };
    let factor = scaling.factor.max(1e-6);
    // `yarn_get_mscale`: one at a scale of at most one, else a log ramp.
    let get_mscale = |scale: f32, mscale: f32| {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * mscale * scale.ln() + 1.0
        }
    };
    // The table's magnitude: an explicit override, else the `mscale` / `mscale_all_dim` ratio
    // (released DeepSeek-V2 configs set both to 0.707, so one), else the generic YaRN default.
    let attention_factor = scaling.attention_factor.unwrap_or_else(|| {
        match (scaling.mscale, scaling.mscale_all_dim) {
            (Some(m), Some(all)) if m != 0.0 && all != 0.0 => {
                get_mscale(factor, m) / get_mscale(factor, all)
            }
            _ => get_mscale(factor, 1.0),
        }
    });
    // The softmax scale's own correction: `mscale_all_dim`'s square, or nothing.
    let softmax_scale_sq = match scaling.mscale_all_dim {
        Some(all) if all != 0.0 => {
            let m = get_mscale(factor, all);
            m * m
        }
        _ => 1.0,
    };
    Ok((
        RopeFlavor::Yarn {
            factor,
            original: scaling.original,
            beta_fast: scaling.beta_fast,
            beta_slow: scaling.beta_slow,
            attention_factor: Some(attention_factor),
        },
        softmax_scale_sq,
    ))
}

// ---- weight names -----------------------------------------------------------------------------

const HF_PREFIX: &str = "model.layers.";
const GGUF_PREFIX: &str = "blk.";

const fn row_linear(prefix: &'static str, role: WeightRole, stem: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerLinear { prefix, stem },
        presence: Presence::Required,
    }
}

const fn row_tensor(prefix: &'static str, role: WeightRole, suffix: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerTensor { prefix, suffix },
        presence: Presence::Required,
    }
}

/// HF rows every DeepSeek MLA checkpoint carries: the model-level weights, the two norms, the
/// latent projections, the output projection and the dense feed-forward of the leading layers.
const HF_BASE: &[NameRow] = &[
    NameRow {
        role: WeightRole::Embed,
        source: Source::Tensor("model.embed_tokens.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::FinalNorm,
        source: Source::Tensor("model.norm.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::Head,
        source: Source::Linear("lm_head"),
        presence: Presence::TiedTo(WeightRole::Embed),
    },
    row_tensor(
        HF_PREFIX,
        WeightRole::Norm(NormRole::Attn),
        ".input_layernorm.weight",
    ),
    row_tensor(
        HF_PREFIX,
        WeightRole::Norm(NormRole::Ffn),
        ".post_attention_layernorm.weight",
    ),
    row_linear(
        HF_PREFIX,
        WeightRole::Attn(AttnRole::O),
        ".self_attn.o_proj",
    ),
    row_linear(
        HF_PREFIX,
        WeightRole::Mla(MlaRole::KvA),
        ".self_attn.kv_a_proj_with_mqa",
    ),
    row_tensor(
        HF_PREFIX,
        WeightRole::Mla(MlaRole::KvANorm),
        ".self_attn.kv_a_layernorm.weight",
    ),
    row_linear(
        HF_PREFIX,
        WeightRole::Mla(MlaRole::KvB),
        ".self_attn.kv_b_proj",
    ),
    row_linear(HF_PREFIX, WeightRole::Ffn(FfnRole::Gate), ".mlp.gate_proj"),
    row_linear(HF_PREFIX, WeightRole::Ffn(FfnRole::Up), ".mlp.up_proj"),
    row_linear(HF_PREFIX, WeightRole::Ffn(FfnRole::Down), ".mlp.down_proj"),
];

const HF_Q_PLAIN: &[NameRow] = &[row_linear(
    HF_PREFIX,
    WeightRole::Attn(AttnRole::Q),
    ".self_attn.q_proj",
)];

const HF_Q_LOW_RANK: &[NameRow] = &[
    row_linear(
        HF_PREFIX,
        WeightRole::Mla(MlaRole::QA),
        ".self_attn.q_a_proj",
    ),
    row_tensor(
        HF_PREFIX,
        WeightRole::Mla(MlaRole::QANorm),
        ".self_attn.q_a_layernorm.weight",
    ),
    row_linear(
        HF_PREFIX,
        WeightRole::Mla(MlaRole::QB),
        ".self_attn.q_b_proj",
    ),
];

/// The sparse-attention indexer's rows (V3.2).
pub(crate) const HF_DSA: &[NameRow] = &[
    row_linear(
        HF_PREFIX,
        WeightRole::Dsa(DsaRole::Q),
        ".self_attn.indexer.wq_b",
    ),
    row_linear(
        HF_PREFIX,
        WeightRole::Dsa(DsaRole::K),
        ".self_attn.indexer.wk",
    ),
    row_tensor(
        HF_PREFIX,
        WeightRole::Dsa(DsaRole::KNorm),
        ".self_attn.indexer.k_norm.weight",
    ),
    row_tensor(
        HF_PREFIX,
        WeightRole::Dsa(DsaRole::KNormBias),
        ".self_attn.indexer.k_norm.bias",
    ),
    row_linear(
        HF_PREFIX,
        WeightRole::Dsa(DsaRole::Weights),
        ".self_attn.indexer.weights_proj",
    ),
];

/// GGUF rows beside the same set: llama.cpp's names, with the up projection stored split per head.
const GGUF_BASE: &[NameRow] = &[
    NameRow {
        role: WeightRole::Embed,
        source: Source::Tensor("token_embd.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::FinalNorm,
        source: Source::Tensor("output_norm.weight"),
        presence: Presence::Required,
    },
    NameRow {
        role: WeightRole::Head,
        source: Source::Linear("output"),
        presence: Presence::TiedTo(WeightRole::Embed),
    },
    row_tensor(
        GGUF_PREFIX,
        WeightRole::Norm(NormRole::Attn),
        ".attn_norm.weight",
    ),
    row_tensor(
        GGUF_PREFIX,
        WeightRole::Norm(NormRole::Ffn),
        ".ffn_norm.weight",
    ),
    row_linear(GGUF_PREFIX, WeightRole::Attn(AttnRole::O), ".attn_output"),
    row_linear(GGUF_PREFIX, WeightRole::Mla(MlaRole::KvA), ".attn_kv_a_mqa"),
    row_tensor(
        GGUF_PREFIX,
        WeightRole::Mla(MlaRole::KvANorm),
        ".attn_kv_a_norm.weight",
    ),
    row_linear(GGUF_PREFIX, WeightRole::Mla(MlaRole::KB), ".attn_k_b"),
    row_linear(GGUF_PREFIX, WeightRole::Mla(MlaRole::VB), ".attn_v_b"),
    row_linear(GGUF_PREFIX, WeightRole::Ffn(FfnRole::Gate), ".ffn_gate"),
    row_linear(GGUF_PREFIX, WeightRole::Ffn(FfnRole::Up), ".ffn_up"),
    row_linear(GGUF_PREFIX, WeightRole::Ffn(FfnRole::Down), ".ffn_down"),
];

const GGUF_Q_PLAIN: &[NameRow] = &[row_linear(
    GGUF_PREFIX,
    WeightRole::Attn(AttnRole::Q),
    ".attn_q",
)];

const GGUF_Q_LOW_RANK: &[NameRow] = &[
    row_linear(GGUF_PREFIX, WeightRole::Mla(MlaRole::QA), ".attn_q_a"),
    row_tensor(
        GGUF_PREFIX,
        WeightRole::Mla(MlaRole::QANorm),
        ".attn_q_a_norm.weight",
    ),
    row_linear(GGUF_PREFIX, WeightRole::Mla(MlaRole::QB), ".attn_q_b"),
];

const HF_EXPERTS: &[ExpertNames] = &[
    ExpertNames {
        proj: ExpertProj::Gate,
        stack: None,
        each: Some((".mlp.experts.", ".gate_proj.weight")),
        bias: None,
    },
    ExpertNames {
        proj: ExpertProj::Up,
        stack: None,
        each: Some((".mlp.experts.", ".up_proj.weight")),
        bias: None,
    },
    ExpertNames {
        proj: ExpertProj::Down,
        stack: None,
        each: Some((".mlp.experts.", ".down_proj.weight")),
        bias: None,
    },
];

const GGUF_EXPERTS: &[ExpertNames] = &[
    ExpertNames {
        proj: ExpertProj::Gate,
        stack: Some(".ffn_gate_exps.weight"),
        each: None,
        bias: None,
    },
    ExpertNames {
        proj: ExpertProj::Up,
        stack: Some(".ffn_up_exps.weight"),
        each: None,
        bias: None,
    },
    ExpertNames {
        proj: ExpertProj::Down,
        stack: Some(".ffn_down_exps.weight"),
        each: None,
        bias: None,
    },
];

/// The routed layers' names: the router's bias exists for a sigmoid router, the shared expert when
/// the config has one.
fn moe_names(source: ConfigSource, routing: &Routing, shared: bool) -> MoeNames {
    let biased = routing.scoring == Scoring::Sigmoid;
    match source {
        ConfigSource::HfModelType => MoeNames {
            prefix: HF_PREFIX,
            router: ".mlp.gate",
            router_bias: biased.then_some(".mlp.gate.e_score_correction_bias"),
            shared: shared.then_some(SharedNames {
                gate: ".mlp.shared_experts.gate_proj",
                up: ".mlp.shared_experts.up_proj",
                down: ".mlp.shared_experts.down_proj",
                scalar_gate: None,
            }),
            experts: HF_EXPERTS,
        },
        ConfigSource::GgufArchitecture => MoeNames {
            prefix: GGUF_PREFIX,
            router: ".ffn_gate_inp",
            router_bias: biased.then_some(".exp_probs_b.bias"),
            shared: shared.then_some(SharedNames {
                gate: ".ffn_gate_shexp",
                up: ".ffn_up_shexp",
                down: ".ffn_down_shexp",
                scalar_gate: None,
            }),
            experts: GGUF_EXPERTS,
        },
    }
}

// ---- the model --------------------------------------------------------------------------------

#[derive(Debug)]
enum Ffn {
    Dense(Box<GatedFfnWeights>),
    Routed(Box<MoeWeights>),
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    mla: MlaWeights,
    dsa: Option<DsaWeights>,
    ffn: Ffn,
}

#[derive(Debug)]
pub struct Deepseek {
    params: DeepseekConfig,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
    family: FamilyKey,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    build_family(V2, raw, store)
}

/// The model of a DeepSeek MLA family `kind` over `raw` and `store`.
pub(crate) fn build_family(
    kind: Kind,
    raw: &RawConfig<'_>,
    store: &WeightStore,
) -> Result<Box<dyn Model>, ModelError> {
    let p = DeepseekConfig::from_raw(raw, kind)?;
    let low_rank = p.mla.q_rank().is_some();
    let hf = matches!(raw.source(), ConfigSource::HfModelType);
    let q: &[NameRow] = match (hf, low_rank) {
        (true, true) => HF_Q_LOW_RANK,
        (true, false) => HF_Q_PLAIN,
        (false, true) => GGUF_Q_LOW_RANK,
        (false, false) => GGUF_Q_PLAIN,
    };
    let base: &[NameRow] = if hf { HF_BASE } else { GGUF_BASE };
    // The GGUF's up projection is split per head; HF's is one tensor. Each table holds only its
    // own form.
    let dsa: &[NameRow] = if p.dsa.is_some() { HF_DSA } else { &[] };
    let tables: [&[NameRow]; 3] = [base, q, dsa];
    let names = moe_names(raw.source(), &p.routing, p.shared);
    let sparse: Vec<bool> = (0..p.layers).map(|l| p.routed(l)).collect();
    let weights = family_weights_routed(
        kind.family,
        store,
        &tables,
        p.layers,
        p.head_required,
        MoeLayers {
            names: &names,
            sparse: &sparse,
            experts: p.moe.experts(),
        },
    )?;
    let weight_error = |e: WeightError| e.for_family(kind.family);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers = (0..p.layers)
        .map(|l| {
            let ffn = if p.routed(l) {
                Ffn::Routed(Box::new(MoeWeights::new(&weights, l, &p.moe)?))
            } else {
                Ffn::Dense(Box::new(GatedFfnWeights::new(
                    &weights,
                    l,
                    p.width,
                    p.dense_inter,
                )?))
            };
            Ok(Layer {
                norms: LayerNorms::new(&weights, l, p.width)?,
                mla: MlaWeights::new(&weights, l, p.width, &p.mla)?,
                dsa: p
                    .dsa
                    .as_ref()
                    .map(|dsa| {
                        let q_rank = p.mla.q_rank().expect("DSA needs the low-rank query");
                        DsaWeights::new(&weights, l, p.width, q_rank, dsa)
                    })
                    .transpose()?,
                ffn,
            })
        })
        .collect::<Result<Vec<_>, WeightError>>()
        .map_err(weight_error)?;
    let config = ModelConfig {
        family: kind.family,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        prompt_bos: None,
        output: ModelOutput::Logits { vocab: p.vocab },
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
    };
    Ok(Box::new(Deepseek {
        params: p,
        weights,
        stack,
        layers,
        config,
        family: kind.family,
    }))
}

impl Model for Deepseek {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace(
        &self,
        phase: Phase,
        shape: StepShape,
    ) -> Result<Graph<ValidationOutputs>, TraceError> {
        let p = &self.params;
        check_step(self.family, phase, shape, p.max_positions)?;
        moe::check_step(self.family, shape, &p.moe)?;
        let b = Builder::new();
        let step = Step::new(&b, shape);
        let logits = standard_stack(&b, &step, &self.stack, &p.stack, p.layers, |b, l, x| {
            let layer = &self.layers[l];
            standard_layer(
                b,
                x,
                &layer.norms,
                &p.layer,
                |b, h| match (&layer.dsa, &p.dsa) {
                    (Some(w), Some(dsa)) => {
                        let select =
                            |b: &Builder, sel: Selection| dsa_select(b, &step, l, h, sel, w, dsa);
                        mla(b, &step, l, h, &layer.mla, &p.mla, Some(&select))
                    }
                    _ => mla(b, &step, l, h, &layer.mla, &p.mla, None),
                },
                |b, BlockInput { block: h, .. }| match &layer.ffn {
                    Ffn::Dense(w) => gated_ffn(b, h, w),
                    Ffn::Routed(w) => moe::moe(b, &step, l, h, None, w, &p.moe),
                },
            )
        });
        Ok(step.finish(b, logits))
    }
}

// ---- the fixture ------------------------------------------------------------------------------

/// The fixture's config: a tiny V2 with a low-rank query, YaRN with `mscale`, group-limited
/// routing and two shared experts, whose widths are multiples of 32 where a Q8_0 copy needs them.
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "deepseek_v2",
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "n_routed_experts": 8,
        "n_shared_experts": 2,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "n_group": 4,
        "topk_group": 2,
        "topk_method": "group_limited_greedy",
        "scoring_func": "softmax",
        "routed_scaling_factor": 2.0,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "yarn",
            "factor": 4.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 16,
            "mscale": 0.707,
            "mscale_all_dim": 0.707
        },
        "tie_word_embeddings": false,
        "eos_token_id": 47,
    })
}

/// The tensors of an HF DeepSeek MLA checkpoint of `config` (`bias` adds the V3 selection bias,
/// `indexer` the V3.2 indexer's weights), by HF name.
pub(crate) fn hf_shapes(
    config: &serde_json::Value,
    bias: bool,
    indexer: bool,
) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let (vocab, w) = (n("vocab_size"), n("hidden_size"));
    let (heads, nope, rope, v) = (
        n("num_attention_heads"),
        n("qk_nope_head_dim"),
        n("qk_rope_head_dim"),
        n("v_head_dim"),
    );
    let (kv_rank, q_rank) = (n("kv_lora_rank"), n("q_lora_rank"));
    let qk = nope + rope;
    let (experts, moe_inter, dense_inter) = (
        n("n_routed_experts"),
        n("moe_intermediate_size"),
        n("intermediate_size"),
    );
    let shared = moe_inter * n("n_shared_experts");
    let mut shapes = vec![
        ("model.embed_tokens.weight".to_string(), vec![vocab, w]),
        ("model.norm.weight".to_string(), vec![w]),
        ("lm_head.weight".to_string(), vec![vocab, w]),
    ];
    for l in 0..n("num_hidden_layers") {
        let k = |s: &str| format!("model.layers.{l}.{s}");
        shapes.extend([
            (k("input_layernorm.weight"), vec![w]),
            (k("post_attention_layernorm.weight"), vec![w]),
            (k("self_attn.q_a_proj.weight"), vec![q_rank, w]),
            (k("self_attn.q_a_layernorm.weight"), vec![q_rank]),
            (k("self_attn.q_b_proj.weight"), vec![heads * qk, q_rank]),
            (
                k("self_attn.kv_a_proj_with_mqa.weight"),
                vec![kv_rank + rope, w],
            ),
            (k("self_attn.kv_a_layernorm.weight"), vec![kv_rank]),
            (
                k("self_attn.kv_b_proj.weight"),
                vec![heads * (nope + v), kv_rank],
            ),
            (k("self_attn.o_proj.weight"), vec![w, heads * v]),
        ]);
        if indexer {
            let (hi, di) = (n("index_n_heads"), n("index_head_dim"));
            shapes.extend([
                (k("self_attn.indexer.wq_b.weight"), vec![hi * di, q_rank]),
                (k("self_attn.indexer.wk.weight"), vec![di, w]),
                (k("self_attn.indexer.k_norm.weight"), vec![di]),
                (k("self_attn.indexer.k_norm.bias"), vec![di]),
                (k("self_attn.indexer.weights_proj.weight"), vec![hi, w]),
            ]);
        }
        if l < n("first_k_dense_replace") {
            shapes.extend([
                (k("mlp.gate_proj.weight"), vec![dense_inter, w]),
                (k("mlp.up_proj.weight"), vec![dense_inter, w]),
                (k("mlp.down_proj.weight"), vec![w, dense_inter]),
            ]);
            continue;
        }
        shapes.push((k("mlp.gate.weight"), vec![experts, w]));
        if bias {
            shapes.push((k("mlp.gate.e_score_correction_bias"), vec![experts]));
        }
        shapes.extend([
            (k("mlp.shared_experts.gate_proj.weight"), vec![shared, w]),
            (k("mlp.shared_experts.up_proj.weight"), vec![shared, w]),
            (k("mlp.shared_experts.down_proj.weight"), vec![w, shared]),
        ]);
        for e in 0..experts {
            for (proj, shape) in [
                ("gate_proj", [moe_inter, w]),
                ("up_proj", [moe_inter, w]),
                ("down_proj", [w, moe_inter]),
            ] {
                shapes.push((k(&format!("mlp.experts.{e}.{proj}.weight")), shape.to_vec()));
            }
        }
    }
    shapes
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_shapes(&config, false, false);
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
pub(crate) mod family_tests {
    //! The rows every DeepSeek MLA family owes, parameterized by its entry and fixture so
    //! `deepseek3` and `deepseek32` run the same ones.

    use poot_graph_ir::OpKind;
    use poot_graph_plan::{WeightFormats, bind_packed_weights};
    use poot_load::gguf::GgufValue;
    use poot_quant::weights::{HandleFormat, WeightEntry};
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, assert_chunked_prefill_equals_decode, assert_missing_weight_is_named, build, close,
        close_all, decoded, f32_entry, gguf_tensor, greedy_tokens, q8_0_store, read_back_owned,
        step_shape, values,
    };
    use crate::model::LogitRows;
    use crate::registry::Registry;

    /// The projections a Q8_0 copy packs: every HF `*_proj.weight` whose input width is a block
    /// multiple.
    fn is_packable(store: &WeightStore) -> impl Fn(&str) -> bool + '_ {
        |key| {
            key.ends_with("_proj.weight")
                && store
                    .get(key)
                    .is_some_and(|e| e.shape().len() == 2 && e.shape()[1] % 32 == 0)
        }
    }

    /// SC-004 (F6): a store missing one required tensor, the MLA latent, an expert's projection, the
    /// router and the head included, fails `Registry::build` with the typed missing-weight error
    /// naming it, before any trace. Mutation: skip the row check in `map_moe_layer`; red.
    pub(crate) fn assert_missing_tensors_are_named(fixture: &Fixture, removed: &[&str]) {
        let registry = Registry::builtin().unwrap();
        for removed in removed {
            assert_missing_weight_is_named(&registry, fixture, removed);
        }
    }

    /// Chunked prefill continuing from `Pos` equals token-by-token decode at every logit and every
    /// carried cache element (the one body serves both phases).
    pub(crate) fn assert_chunks_equal_decode(entry: &FamilyEntry, fixture: &Fixture) {
        let model = build(entry, &fixture.raw(), &fixture.store);
        assert_chunked_prefill_equals_decode(&*model, &fixture.store);
    }

    /// The BF16 store and its Q8_0 copy trace through the one body: the Q8_0 graph holds one
    /// `PackedDequant` per packed weight and no packed const, and each equals the same body over
    /// its store decoded to F32, in prefill and a decode step.
    pub(crate) fn assert_bf16_and_q8_0_trace_and_match(entry: &FamilyEntry, fixture: &Fixture) {
        let raw = fixture.raw();
        let prompt = [3, 17, 40, 8];
        let packed_store = q8_0_store(&fixture.store, is_packable(&fixture.store));
        for (label, store) in [("bf16", fixture.store.clone()), ("q8_0", packed_store)] {
            let model = build(entry, &raw, &store);
            let g = model
                .trace(
                    Phase::Prefill,
                    step_shape(prompt.len(), CAP, LogitRows::All),
                )
                .unwrap();
            let packed = model
                .weights()
                .iter()
                .filter(|(_, _, h)| matches!(h.format, HandleFormat::Packed(_)))
                .count();
            let bound =
                bind_packed_weights(&g, &WeightFormats::from_weight_map(model.weights())).unwrap();
            let dequants = bound
                .eqns
                .iter()
                .filter(|e| matches!(e.op, OpKind::PackedDequant { .. }))
                .count();
            let indexed = bound
                .eqns
                .iter()
                .filter(|e| {
                    matches!(
                        e.op,
                        OpKind::PackedContraction {
                            rows: poot_graph_ir::ContractionRows::Indexed { .. },
                            ..
                        }
                    )
                })
                .count();
            let stacks = model
                .weights()
                .iter()
                .filter(|(_, _, h)| {
                    matches!(h.format, HandleFormat::Packed(_)) && h.shape.len() == 3
                })
                .count();
            assert_eq!(
                dequants + stacks,
                packed,
                "{label}: one decode per packed weight"
            );
            assert_eq!(
                indexed, stacks,
                "{label}: one indexed contraction per packed stack"
            );
            assert_eq!(packed > 0, label == "q8_0", "{label}: {packed} packed");

            let reference_store = decoded(&store);
            let reference = build(entry, &raw, &reference_store);
            let (got, state) = run_step(&*model, &store, Phase::Prefill, CAP, &prompt, 0, &[]);
            let (want, want_state) = run_step(
                &*reference,
                &reference_store,
                Phase::Prefill,
                CAP,
                &prompt,
                0,
                &[],
            );
            close(&got, &want);
            close_all(&state, &want_state);
            let (got, _) = run_step(&*model, &store, Phase::Decode, CAP, &[5], 4, &state);
            let (want, _) = run_step(
                &*reference,
                &reference_store,
                Phase::Decode,
                CAP,
                &[5],
                4,
                &want_state,
            );
            close(&got, &want);
        }
    }

    /// llama.cpp's GGUF of an HF fixture: the same values under llama.cpp's names, `kv_b_proj`
    /// split into the per-head `attn_k_b` `[heads, rank, nope]` and `attn_v_b` `[heads, v, rank]`
    /// tensors, experts stacked.
    fn gguf_of(fixture: &Fixture) -> (poot_load::gguf::GgufIndex, WeightStore) {
        let n = |f: &str| fixture.config[f].as_u64().unwrap();
        let (heads, nope, rope, v, rank) = (
            n("num_attention_heads") as usize,
            n("qk_nope_head_dim") as usize,
            n("qk_rope_head_dim") as usize,
            n("v_head_dim") as usize,
            n("kv_lora_rank") as usize,
        );
        let mut tensors = Vec::new();
        let mut stacks: std::collections::BTreeMap<String, Vec<(usize, &WeightEntry)>> =
            Default::default();
        for (key, entry) in fixture.store.iter() {
            let hf = key.as_str();
            let plain = |name: &str| (name.to_string(), entry);
            let named = match hf {
                "model.embed_tokens.weight" => plain("token_embd.weight"),
                "model.norm.weight" => plain("output_norm.weight"),
                "lm_head.weight" => plain("output.weight"),
                _ => {
                    let rest = hf.strip_prefix("model.layers.").unwrap();
                    let (layer, tensor) = rest.split_once('.').unwrap();
                    if let Some(expert) = tensor.strip_prefix("mlp.experts.") {
                        let (index, proj) = expert.split_once('.').unwrap();
                        let stack = match proj {
                            "gate_proj.weight" => "ffn_gate_exps.weight",
                            "up_proj.weight" => "ffn_up_exps.weight",
                            "down_proj.weight" => "ffn_down_exps.weight",
                            other => panic!("{other}"),
                        };
                        stacks
                            .entry(format!("blk.{layer}.{stack}"))
                            .or_default()
                            .push((index.parse().unwrap(), entry));
                        continue;
                    }
                    if tensor == "self_attn.kv_b_proj.weight" {
                        let fused = values(entry);
                        let per_head = nope + v;
                        let (mut k_b, mut v_b) = (Vec::new(), Vec::new());
                        for h in 0..heads {
                            let at = |c: usize, r: usize| fused[(h * per_head + c) * rank + r];
                            for r in 0..rank {
                                k_b.extend((0..nope).map(|c| at(c, r)));
                            }
                            for c in 0..v {
                                v_b.extend((0..rank).map(|r| at(nope + c, r)));
                            }
                        }
                        let layer_name = |s: &str| format!("blk.{layer}.{s}");
                        tensors.push(gguf_tensor(
                            &layer_name("attn_k_b.weight"),
                            &f32_entry(vec![heads, rank, nope], &k_b),
                        ));
                        tensors.push(gguf_tensor(
                            &layer_name("attn_v_b.weight"),
                            &f32_entry(vec![heads, v, rank], &v_b),
                        ));
                        continue;
                    }
                    let name = match tensor {
                        "input_layernorm.weight" => "attn_norm.weight",
                        "post_attention_layernorm.weight" => "ffn_norm.weight",
                        "self_attn.q_a_proj.weight" => "attn_q_a.weight",
                        "self_attn.q_a_layernorm.weight" => "attn_q_a_norm.weight",
                        "self_attn.q_b_proj.weight" => "attn_q_b.weight",
                        "self_attn.kv_a_proj_with_mqa.weight" => "attn_kv_a_mqa.weight",
                        "self_attn.kv_a_layernorm.weight" => "attn_kv_a_norm.weight",
                        "self_attn.o_proj.weight" => "attn_output.weight",
                        "mlp.gate_proj.weight" => "ffn_gate.weight",
                        "mlp.up_proj.weight" => "ffn_up.weight",
                        "mlp.down_proj.weight" => "ffn_down.weight",
                        "mlp.gate.weight" => "ffn_gate_inp.weight",
                        "mlp.gate.e_score_correction_bias" => "exp_probs_b.bias",
                        "mlp.shared_experts.gate_proj.weight" => "ffn_gate_shexp.weight",
                        "mlp.shared_experts.up_proj.weight" => "ffn_up_shexp.weight",
                        "mlp.shared_experts.down_proj.weight" => "ffn_down_shexp.weight",
                        other => panic!("unmapped fixture tensor {other}"),
                    };
                    (format!("blk.{layer}.{name}"), entry)
                }
            };
            tensors.push(gguf_tensor(&named.0, named.1));
        }
        for (name, mut experts) in stacks {
            experts.sort_by_key(|&(index, _)| index);
            let count = experts.len() as u64;
            let (_, mut dims, ty, mut bytes) = gguf_tensor(&name, experts[0].1);
            for &(_, entry) in &experts[1..] {
                bytes.extend(gguf_tensor(&name, entry).3);
            }
            dims.push(count);
            tensors.push((name, dims, ty, bytes));
        }

        let u = |v: u64| GgufValue::U32(v as u32);
        let f = |key: &str| GgufValue::F32(fixture.config[key].as_f64().unwrap() as f32);
        let scaling = &fixture.config["rope_scaling"];
        let sf = |key: &str| GgufValue::F32(scaling[key].as_f64().unwrap() as f32);
        let sigmoid = fixture.config["scoring_func"] == "sigmoid";
        let mut kvs = vec![
            (
                "general.architecture".to_string(),
                GgufValue::Str("deepseek2".into()),
            ),
            ("deepseek2.block_count".into(), u(n("num_hidden_layers"))),
            ("deepseek2.embedding_length".into(), u(n("hidden_size"))),
            ("deepseek2.attention.head_count".into(), u(heads as u64)),
            (
                "deepseek2.attention.q_lora_rank".into(),
                u(n("q_lora_rank")),
            ),
            ("deepseek2.attention.kv_lora_rank".into(), u(rank as u64)),
            (
                "deepseek2.attention.key_length_mla".into(),
                u((nope + rope) as u64),
            ),
            ("deepseek2.attention.value_length_mla".into(), u(v as u64)),
            ("deepseek2.rope.dimension_count".into(), u(rope as u64)),
            (
                "deepseek2.attention.layer_norm_rms_epsilon".into(),
                f("rms_norm_eps"),
            ),
            (
                "deepseek2.context_length".into(),
                u(n("max_position_embeddings")),
            ),
            ("deepseek2.rope.freq_base".into(), f("rope_theta")),
            (
                "deepseek2.rope.scaling.type".into(),
                GgufValue::Str("yarn".into()),
            ),
            ("deepseek2.rope.scaling.factor".into(), sf("factor")),
            (
                "deepseek2.rope.scaling.original_context_length".into(),
                u(n_of(scaling, "original_max_position_embeddings")),
            ),
            (
                "deepseek2.rope.scaling.yarn_beta_fast".into(),
                sf("beta_fast"),
            ),
            (
                "deepseek2.rope.scaling.yarn_beta_slow".into(),
                sf("beta_slow"),
            ),
            (
                "deepseek2.rope.scaling.yarn_log_multiplier".into(),
                GgufValue::F32(0.1 * scaling["mscale_all_dim"].as_f64().unwrap() as f32),
            ),
            (
                "deepseek2.feed_forward_length".into(),
                u(n("intermediate_size")),
            ),
            ("deepseek2.expert_count".into(), u(n("n_routed_experts"))),
            (
                "deepseek2.expert_feed_forward_length".into(),
                u(n("moe_intermediate_size")),
            ),
            (
                "deepseek2.expert_used_count".into(),
                u(n("num_experts_per_tok")),
            ),
            (
                "deepseek2.expert_shared_count".into(),
                u(n("n_shared_experts")),
            ),
            (
                "deepseek2.leading_dense_block_count".into(),
                u(n("first_k_dense_replace")),
            ),
            ("deepseek2.expert_group_count".into(), u(n("n_group"))),
            (
                "deepseek2.expert_group_used_count".into(),
                u(n("topk_group")),
            ),
            (
                "deepseek2.expert_weights_scale".into(),
                f("routed_scaling_factor"),
            ),
            (
                "deepseek2.expert_gating_func".into(),
                u(if sigmoid { 2 } else { 1 }),
            ),
            ("tokenizer.ggml.eos_token_id".into(), u(n("eos_token_id"))),
            (
                "tokenizer.ggml.tokens".into(),
                GgufValue::Array(
                    (0..n("vocab_size"))
                        .map(|t| GgufValue::Str(format!("t{t}")))
                        .collect(),
                ),
            ),
        ];
        if sigmoid {
            kvs.push((
                "deepseek2.expert_weights_norm".into(),
                GgufValue::Bool(fixture.config["norm_topk_prob"].as_bool().unwrap()),
            ));
        }
        read_back_owned(&kvs, &tensors)
    }

    fn n_of(block: &serde_json::Value, key: &str) -> u64 {
        block[key].as_u64().unwrap()
    }

    /// SC-001/SC-004 (GGUF half, R-566a-8): a GGUF of the fixture resolves by its architecture
    /// key and binds `attn_k_b` and `attn_v_b` as stored (no rebuilt `kv_b_proj`), and computes what
    /// the HF body computes over the same values, in prefill and a decode step. Mutation: read the
    /// split `k_b` as `[heads, nope, rank]` (the `Fused` orientation); every output goes red.
    pub(crate) fn assert_gguf_equals_the_hf_body(entry: &FamilyEntry, fixture: &Fixture) {
        let (index, gguf_store) = gguf_of(fixture);
        let registry = Registry::builtin().unwrap();
        let raw = RawConfig::Gguf(&index);
        assert_eq!(registry.resolve(&raw).unwrap().family, FAMILY);
        let model = registry.build(&raw, &gguf_store).unwrap();
        assert!(
            model
                .weights()
                .iter()
                .all(|(id, _, _)| !matches!(id.role, WeightRole::Mla(MlaRole::KvB))),
            "no rebuilt kv_b_proj"
        );
        assert!(
            model
                .weights()
                .iter()
                .any(|(id, _, _)| id.role == WeightRole::Mla(MlaRole::KB)),
            "k_b is bound as stored"
        );
        let reference_store = decoded(&fixture.store);
        let reference = build(entry, &fixture.raw(), &reference_store);
        let prompt = [3, 17, 40, 8];
        let (got, state) = run_step(&*model, &gguf_store, Phase::Prefill, CAP, &prompt, 0, &[]);
        let (want, want_state) = run_step(
            &*reference,
            &reference_store,
            Phase::Prefill,
            CAP,
            &prompt,
            0,
            &[],
        );
        close(&got, &want);
        close_all(&state, &want_state);
        let (got, _) = run_step(&*model, &gguf_store, Phase::Decode, CAP, &[5], 4, &state);
        let (want, _) = run_step(
            &*reference,
            &reference_store,
            Phase::Decode,
            CAP,
            &[5],
            4,
            &want_state,
        );
        close(&got, &want);
    }

    fn config_error_of(
        entry: &FamilyEntry,
        fixture: &Fixture,
        config: &serde_json::Value,
    ) -> (&'static str, ConfigReason) {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        match (entry.build)(&raw, &fixture.store) {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("{config}: expected a config error, got {other:?}"),
        }
    }

    /// Tier 3 on the oracle (ADR-0101): the fixture's 24 greedy tokens after the 7-token prompt equal
    /// the literals recorded from the pre-driver engine's CPU generate at the base commit `64602bced`.
    pub(crate) fn assert_generates_the_recorded_tokens(
        entry: &FamilyEntry,
        fixture: &Fixture,
        recorded: [u32; 24],
    ) {
        let model = build(entry, &fixture.raw(), &fixture.store);
        let got = greedy_tokens(&*model, &fixture.store, &[3, 17, 40, 8, 25, 11, 2], 24, 32);
        assert_eq!(got, recorded);
    }

    /// A config the family cannot trace is a typed error naming the field.
    pub(crate) fn assert_bad_configs_are_typed(entry: &FamilyEntry, fixture: &Fixture) {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "moe_layer_freq",
                json!(2),
                "moe_layer_freq",
                ConfigReason::Unsupported,
            ),
            (
                "scoring_func",
                json!("tanh"),
                "scoring_func",
                ConfigReason::Unsupported,
            ),
            (
                "topk_method",
                json!("magic"),
                "topk_method",
                ConfigReason::Unsupported,
            ),
            (
                "num_experts_per_tok",
                json!(9),
                "top_k",
                ConfigReason::Exceeds { max: 8 },
            ),
            (
                "qk_rope_head_dim",
                json!(5),
                "rotary_dim",
                ConfigReason::NotDivisible { by: 2 },
            ),
            (
                "topk_group",
                json!(5),
                "topk_group",
                ConfigReason::Exceeds { max: 4 },
            ),
            (
                "n_routed_experts",
                json!(0),
                "n_routed_experts",
                ConfigReason::Zero,
            ),
            (
                "rope_scaling",
                json!({"type": "linear", "factor": 2.0}),
                "rope_scaling.rope_type",
                ConfigReason::Unsupported,
            ),
        ];
        for (key, value, field, reason) in cases {
            let mut config = fixture.config.clone();
            config[*key] = value.clone();
            assert_eq!(
                config_error_of(entry, fixture, &config),
                (*field, *reason),
                "{key} = {value}"
            );
        }
    }

    /// SC-002 (family half): the sigmoid and softmax routers, the shared expert and the dense
    /// prefix each reach the logits: a different scaling factor changes them.
    pub(crate) fn assert_the_routing_scale_reaches_the_logits(
        entry: &FamilyEntry,
        fixture: &Fixture,
    ) {
        let run = |scale: f64| {
            let mut config = fixture.config.clone();
            config["routed_scaling_factor"] = json!(scale);
            let model = build(
                entry,
                &RawConfig::HfJson {
                    config: &config,
                    generation: None,
                },
                &fixture.store,
            );
            run_step(
                &*model,
                &fixture.store,
                Phase::Prefill,
                CAP,
                &[3, 17, 40],
                0,
                &[],
            )
            .0
        };
        assert_ne!(run(1.0), run(3.0));
    }
}

#[cfg(test)]
mod tests {
    use super::family_tests::*;
    use super::*;

    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        assert_missing_tensors_are_named(
            &fixture(),
            &[
                "model.layers.1.mlp.gate.weight",
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.1.mlp.experts.7.down_proj.weight",
                "model.layers.1.mlp.shared_experts.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.layers.0.self_attn.q_a_proj.weight",
                "model.layers.1.self_attn.kv_b_proj.weight",
                "model.layers.0.self_attn.kv_a_layernorm.weight",
                "lm_head.weight",
            ],
        );
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        assert_chunks_equal_decode(&ENTRY, &fixture());
    }

    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body() {
        assert_bf16_and_q8_0_trace_and_match(&ENTRY, &fixture());
    }

    #[test]
    fn a_gguf_equals_the_hf_body() {
        assert_gguf_equals_the_hf_body(&ENTRY, &fixture());
    }

    #[test]
    fn a_config_deepseek_cannot_trace_is_a_typed_error_naming_the_field() {
        assert_bad_configs_are_typed(&ENTRY, &fixture());
    }

    #[test]
    fn the_routing_scale_reaches_the_logits() {
        assert_the_routing_scale_reaches_the_logits(&ENTRY, &fixture());
    }

    /// A single group and a scale of one add no router option, so a greedy V2-Lite layer traces no
    /// group or scale ops; a real group limit and scale are options.
    #[test]
    fn the_identity_group_limit_and_scale_add_no_router_option() {
        let identity = Routing {
            scoring: Scoring::Softmax,
            groups: 1,
            group_top: 1,
            scale: 1.0,
            renormalize: false,
        };
        let plain = MoeParams::new(ENTRY.family, 8, 4, 6, 2, router_kind(&identity)).unwrap();
        assert_eq!(with_routing(plain, &identity).unwrap(), plain);
        let limited = Routing {
            groups: 3,
            group_top: 2,
            ..identity
        };
        assert_ne!(with_routing(plain, &limited).unwrap(), plain);
        let scaled = Routing {
            scale: 2.5,
            ..identity
        };
        assert_ne!(with_routing(plain, &scaled).unwrap(), plain);
    }

    /// SC-001 (tier 3 on the oracle): the pre-driver engine's recorded tokens. Mutation: swap two experts'
    /// ids in the router arm; the first tokens change.
    #[test]
    fn the_fixture_generates_the_tokens_the_runner_recorded() {
        assert_generates_the_recorded_tokens(
            &ENTRY,
            &fixture(),
            [
                11, 32, 18, 32, 18, 32, 29, 38, 32, 20, 6, 19, 20, 6, 19, 20, 6, 19, 20, 6, 19, 20,
                6, 19,
            ],
        );
    }
}
