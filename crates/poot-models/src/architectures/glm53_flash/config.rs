//! GLM-5.3-Flash's typed config: every value checked once, here, so tracing cannot fail on it. The
//! checkpoint is HF's multimodal wrapper; the text tower's config lives under `text_config`. Every
//! dimension, count and the per-layer mixer and feed-forward schedule is the checkpoint's: a
//! fine-tune with fewer layers or experts, or another mixer order, is the same family.

use std::collections::BTreeSet;

use super::FAMILY;
use crate::components::dims::{Field, Fields};
use crate::components::dsa::PooledDsaParams;
use crate::components::ffn::Glu;
use crate::components::kda::KdaParams;
use crate::components::mhc::MhcParams;
use crate::components::moe::MoeParams;
use crate::components::norm::NormParams;
use crate::components::router::{RouterBias, RouterKind, SigmoidTopK};
use crate::components::standard::ParamError;
use crate::model::{ConfigReason, ModelError};
use crate::registry::RawConfig;

/// A layer's token mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mixer {
    /// Kimi Delta Attention: a gated delta rule over a carried matrix state (`linear_attention`).
    Kda,
    /// DeepSeek sparse attention over a latent cache with a pooled indexer
    /// (`deepseek_sparse_attention`).
    Dsa,
}

/// A layer's feed-forward block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ffn {
    /// One gated feed-forward block of `intermediate_size`.
    Dense,
    /// The routed experts and the shared expert.
    Sparse,
}

/// One layer's kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerKind {
    /// The layer's token mixer.
    pub mixer: Mixer,
    /// The layer's feed-forward block.
    pub ffn: Ffn,
}

/// GLM-5.3-Flash's config.
#[derive(Clone, Debug, PartialEq)]
pub struct Glm53Config {
    pub(super) vocab: usize,
    pub(super) width: usize,
    pub(super) max_positions: usize,
    pub(super) layers: Vec<LayerKind>,
    /// The multi-token-prediction layers the checkpoint stores after the text layers.
    pub(super) mtp: Option<MtpLayers>,
    pub(super) tied_head: bool,
    pub(super) norm: NormParams,
    pub(super) mhc: MhcParams,
    /// Present when some layer is of the kind that reads it.
    pub(super) kda: Option<KdaParams>,
    pub(super) dsa: Option<PooledDsaParams>,
    pub(super) dense: Option<DenseDims>,
    pub(super) moe: Option<MoeParams>,
    pub(super) eos: BTreeSet<u32>,
    pub(super) bos: Option<u32>,
}

/// The multi-token-prediction layers of a checkpoint: how many it stores and the feed-forward kind
/// of each. An MTP layer is a DSA block (latent attention under the pooled indexer, no
/// hyper-connections), whose feed-forward is the last text layer's kind, as in vLLM
/// `Glm5NextDecoderLayer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MtpLayers {
    pub count: usize,
    pub ffn: Ffn,
}

/// The dense feed-forward block: its width and gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DenseDims {
    pub inter: usize,
    pub glu: Glu,
}

/// The text tower's config of an HF `glm5_next` config: `text_config`, with the token ids and the
/// head tie the wrapper config carries (the wrapper's value wins).
fn text_config(config: &serde_json::Value) -> Result<serde_json::Value, ModelError> {
    let wrong = |reason| ModelError::Config {
        family: FAMILY,
        field: "text_config",
        reason,
    };
    let mut text = config
        .get("text_config")
        .filter(|v| !v.is_null())
        .ok_or(wrong(ConfigReason::Missing))?
        .clone();
    let object = text.as_object_mut().ok_or(wrong(ConfigReason::WrongType))?;
    for key in ["eos_token_id", "bos_token_id", "tie_word_embeddings"] {
        if let Some(value) = config.get(key).filter(|v| !v.is_null()) {
            object.insert(key.to_string(), value.clone());
        }
    }
    Ok(text)
}

impl Glm53Config {
    /// Reads the typed config from an HF `config.json` (the family is HF-only: a GGUF is refused as
    /// `Unsupported`); the per-layer schedule comes from `layer_types` and the dense-layer prefix.
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        // The family is HF-only: a GGUF never names it.
        let RawConfig::HfJson { config, generation } = raw else {
            return Err(ModelError::Config {
                family: FAMILY,
                field: "general.architecture",
                reason: ConfigReason::Unsupported,
            });
        };
        let text = text_config(config)?;
        let raw = RawConfig::HfJson {
            config: &text,
            generation: *generation,
        };
        let fields = Fields::new(&raw, FAMILY);
        let param = |e: ParamError| e.for_family(FAMILY);
        let count = |key: &'static str| fields.count(Field::hf(key), None);
        let float = |key: &'static str, default: f32| fields.float(Field::hf(key), default);

        // What the tracer does not implement is refused by name, not assumed.
        let require = |key: &'static str, accepted: &[&str]| -> Result<(), ModelError> {
            match text.get(key).filter(|v| !v.is_null()) {
                None => Ok(()),
                Some(value) if value.as_str().is_some_and(|s| accepted.contains(&s)) => Ok(()),
                Some(_) => Err(fields.error(key, ConfigReason::Unsupported)),
            }
        };
        require("hidden_act", &["silu"])?;
        require("scoring_func", &["sigmoid"])?;
        require("topk_method", &["noaux_tc"])?;
        for (key, supported) in [
            ("mla_use_nope", true),
            ("index_kpool_compress", true),
            ("index_kpool_always_select_tail", true),
            ("attention_bias", false),
        ] {
            if fields
                .opt_bool(Field::hf(key))?
                .is_some_and(|v| v != supported)
            {
                return Err(fields.error(key, ConfigReason::Unsupported));
            }
        }

        let width = count("hidden_size")?;
        let layers = count("num_hidden_layers")?;
        let max_positions = count("max_position_embeddings")?;
        let eps = float("rms_norm_eps", 1e-5)?;
        let norm = NormParams::new(eps).map_err(param)?;
        let mhc = MhcParams::new(
            count("hc_mult")?,
            eps,
            float("hc_eps", 1e-6)?,
            count("hc_sinkhorn_iters")?,
        )
        .map_err(param)?;

        let kinds = layer_kinds(&fields, &text, layers)?;
        let mtp_count = fields
            .opt_u64(Field::hf("num_nextn_predict_layers"))?
            .map_or(Ok(0), |n| {
                usize::try_from(n)
                    .map_err(|_| fields.error("num_nextn_predict_layers", ConfigReason::WrongType))
            })?;
        let mtp = (mtp_count > 0).then(|| MtpLayers {
            count: mtp_count,
            ffn: kinds.last().map_or(Ffn::Sparse, |kind| kind.ffn),
        });
        // The parameters of a kind are read exactly when a text layer or an MTP layer has it.
        let has = |mixer: Mixer| {
            kinds.iter().any(|k| k.mixer == mixer) || (mixer == Mixer::Dsa && mtp.is_some())
        };
        let has_ffn = |ffn: Ffn| {
            kinds.iter().any(|k| k.ffn == ffn) || mtp.is_some_and(|layers| layers.ffn == ffn)
        };

        let kda = has(Mixer::Kda)
            .then(|| -> Result<KdaParams, ModelError> {
                let linear = text
                    .get("linear_attn_config")
                    .filter(|v| !v.is_null())
                    .ok_or(fields.error("linear_attn_config", ConfigReason::Missing))?;
                let nested = |key: &'static str| -> Result<&serde_json::Value, ModelError> {
                    linear
                        .get(key)
                        .filter(|v| !v.is_null())
                        .ok_or(fields.error(key, ConfigReason::Missing))
                };
                let nested_count = |key: &'static str| -> Result<usize, ModelError> {
                    nested(key)?
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or(fields.error(key, ConfigReason::WrongType))
                };
                KdaParams::new(
                    nested_count("num_heads")?,
                    nested_count("head_dim")?,
                    nested_count("short_conv_kernel_size")?,
                    nested("gate_lower_bound")?
                        .as_f64()
                        .map(|v| v as f32)
                        .ok_or(fields.error("gate_lower_bound", ConfigReason::WrongType))?,
                    norm,
                )
                .map_err(param)
            })
            .transpose()?;

        let dsa = has(Mixer::Dsa)
            .then(|| -> Result<PooledDsaParams, ModelError> {
                if fields
                    .opt_u64(Field::hf("qk_rope_head_dim"))?
                    .is_some_and(|d| d != 0)
                {
                    return Err(fields.error("qk_rope_head_dim", ConfigReason::Unsupported));
                }
                if let Some(types) = text.get("indexer_types").filter(|v| !v.is_null()) {
                    let types = types
                        .as_array()
                        .ok_or(fields.error("indexer_types", ConfigReason::WrongType))?;
                    if types.len() != layers {
                        return Err(fields
                            .error("indexer_types", ConfigReason::Length { expected: layers }));
                    }
                    if types.iter().any(|t| t.as_str() != Some("full")) {
                        return Err(fields.error("indexer_types", ConfigReason::Unsupported));
                    }
                }
                PooledDsaParams::new(
                    count("num_attention_heads")?,
                    count("q_lora_rank")?,
                    count("kv_lora_rank")?,
                    count("qk_nope_head_dim")?,
                    count("v_head_dim")?,
                    norm,
                    count("index_n_heads")?,
                    count("index_head_dim")?,
                    count("index_topk")?,
                    count("index_kpool")?,
                )
                .map_err(param)
            })
            .transpose()?;

        let glu = Glu::clamped_silu(float("swiglu_limit", 10.0)?).map_err(param)?;
        let dense = has_ffn(Ffn::Dense)
            .then(|| -> Result<DenseDims, ModelError> {
                Ok(DenseDims {
                    inter: count("intermediate_size")?,
                    glu,
                })
            })
            .transpose()?;
        let moe = has_ffn(Ffn::Sparse)
            .then(|| -> Result<MoeParams, ModelError> {
                let renormalize = fields.flag(Field::hf("norm_topk_prob"), true)?;
                let inter = count("moe_intermediate_size")?;
                let shared = fields.count(Field::hf("n_shared_experts"), Some(1))?;
                let groups = fields.count(Field::hf("n_group"), Some(1))?;
                let keep = fields.count(Field::hf("topk_group"), Some(1))?;
                let mut params = MoeParams::new(
                    FAMILY,
                    width,
                    inter,
                    count("n_routed_experts")?,
                    count("num_experts_per_tok")?,
                    RouterKind::SigmoidBiasedTopK(SigmoidTopK::new().with_renormalize(renormalize)),
                )?
                .with_shared(inter * shared)
                .map_err(param)?
                .with_routed_scale(float("routed_scaling_factor", 1.0)?)
                .map_err(param)?
                .with_glu(glu)
                .with_router_bias(RouterBias::SelectionOnly)
                .map_err(param)?;
                if groups > 1 {
                    params = params.with_group_limit(groups, keep).map_err(param)?;
                }
                Ok(params)
            })
            .transpose()?;

        Ok(Self {
            vocab: fields.vocab()?,
            width,
            max_positions,
            layers: kinds,
            mtp,
            tied_head: fields.flag(Field::hf("tie_word_embeddings"), false)?,
            norm,
            mhc,
            kda,
            dsa,
            dense,
            moe,
            eos: fields.eos(None)?,
            bos: fields.bos()?,
        })
    }
}

/// A per-layer list of strings: `key`, which must name `layers` entries.
fn layer_strings<'a>(
    fields: &Fields<'_>,
    text: &'a serde_json::Value,
    key: &'static str,
    layers: usize,
) -> Result<Option<Vec<&'a str>>, ModelError> {
    let Some(list) = text.get(key).filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let list = list
        .as_array()
        .ok_or(fields.error(key, ConfigReason::WrongType))?;
    if list.len() != layers {
        return Err(fields.error(key, ConfigReason::Length { expected: layers }));
    }
    list.iter()
        .map(|v| v.as_str().ok_or(fields.error(key, ConfigReason::WrongType)))
        .collect::<Result<_, _>>()
        .map(Some)
}

/// The mixer and the feed-forward of every layer: `layer_types` names the mixers and
/// `mlp_layer_types` the feed-forwards (or, without it, `first_k_dense_replace` leading dense layers).
fn layer_kinds(
    fields: &Fields<'_>,
    text: &serde_json::Value,
    layers: usize,
) -> Result<Vec<LayerKind>, ModelError> {
    let mixers = layer_strings(fields, text, "layer_types", layers)?
        .ok_or(fields.error("layer_types", ConfigReason::Missing))?;
    let ffns = layer_strings(fields, text, "mlp_layer_types", layers)?;
    let dense_leading = fields
        .opt_u64(Field::hf("first_k_dense_replace"))?
        .map_or(Ok(0), |n| {
            usize::try_from(n)
                .ok()
                .filter(|&n| n <= layers)
                .ok_or(fields.error(
                    "first_k_dense_replace",
                    ConfigReason::Exceeds { max: layers },
                ))
        })?;
    (0..layers)
        .map(|l| {
            let mixer = match mixers[l] {
                "linear_attention" => Mixer::Kda,
                "deepseek_sparse_attention" => Mixer::Dsa,
                _ => return Err(fields.error("layer_types", ConfigReason::Unsupported)),
            };
            let ffn = match ffns.as_ref().map(|f| f[l]) {
                Some("dense") => Ffn::Dense,
                Some("sparse") => Ffn::Sparse,
                Some(_) => return Err(fields.error("mlp_layer_types", ConfigReason::Unsupported)),
                None if l < dense_leading => Ffn::Dense,
                None => Ffn::Sparse,
            };
            Ok(LayerKind { mixer, ffn })
        })
        .collect()
}
