//! Qwen4Exp's typed config and weight-name table, both data: every config value is checked once,
//! here, so tracing cannot fail on it, and the checkpoint's tensor names map onto typed weight ids
//! in one table. Nothing here pins a shipped checkpoint: any layer, expert and head count the
//! architecture admits loads (a finetune of a different size is the same family).

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_quant::weights::{
    AttnRole, ExpertProj, GdnRole, HyperRole, HyperSite, PleRole, QsaRole, WeightId, WeightKey,
    WeightMap, WeightMapBuilder, WeightMapError, WeightRole, WeightStore, WeightView,
};

use super::gated_residual::HyperParams;
use super::qsa::QsaParams;
use super::{FAMILY, LayerKind};
use crate::components::attention::{AttentionParams, QkNorm};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::gdn::{GdnHeadOrder, GdnParams};
use crate::components::moe::{GateUpForm, MoeParams};
use crate::components::norm::{NormKind, NormParams};
use crate::components::ple::{PleParams, PleSpec};
use crate::components::rope::RopeParams;
use crate::components::router::RouterKind;
use crate::components::standard::ParamError;
use crate::model::{ConfigReason, ModelError};
use crate::names::{ExpertNames, MoeNames, SharedNames, map_moe_layer};
use crate::registry::RawConfig;

/// The config facts the HF `qwen4_exp` class defaults. The family is HF-only: there is no GGUF
/// architecture for it, so the GGUF keys name nothing.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("qwen4_exp"),
    hf_theta: 10_000.0,
    gguf_theta: 10_000.0,
    eps: 1e-6,
    eos: None,
};

const EXPERT_INTER: Field = Field::hf("moe_intermediate_size");
const SHARED_INTER: Field = Field::hf("shared_expert_intermediate_size");
const EXPERTS: Field = Field::hf("num_experts");
const TOP_K: Field = Field::hf("num_experts_per_tok");
const NORM_TOPK_PROB: Field = Field::hf("norm_topk_prob");
const KEY_HEADS: Field = Field::hf("linear_num_key_heads");
const VALUE_HEADS: Field = Field::hf("linear_num_value_heads");
const KEY_HEAD_DIM: Field = Field::hf("linear_key_head_dim");
const VALUE_HEAD_DIM: Field = Field::hf("linear_value_head_dim");
const CONV_KERNEL: Field = Field::hf("linear_conv_kernel_dim");
const PARTIAL_ROTARY: Field = Field::hf("partial_rotary_factor");
const INDEX_HEADS: Field = Field::hf("indexer_n_heads");
const INDEX_KV_HEADS: Field = Field::hf("indexer_kv_heads");
const INDEX_DIM: Field = Field::hf("indexer_head_dim");
const INDEX_BUDGET: Field = Field::hf("indexer_budget");
const INDEX_RATIO: Field = Field::hf("indexer_compress_ratio");
const HC_COUNT: Field = Field::hf("hc_count");
const HC_RANK: Field = Field::hf("hc_lowrank");
const PLE_EMBED: Field = Field::hf("ple_embed_dim");
const PLE_KERNEL: Field = Field::hf("ple_conv_kernel_size");
const NGRAM_SIZE: Field = Field::hf("ngram_size");
const NGRAM_BASE: Field = Field::hf("ngram_vocab_size_base");
const NGRAM_HEADS: Field = Field::hf("heads_per_ngram");
const NGRAM_DIVISIBLE: Field = Field::hf("make_ngram_vocab_size_divisible_by");
const NGRAM_SEED: Field = Field::hf("seed");

/// The reference model's default hash seed (`configuration_qwen4_exp.py`).
const DEFAULT_SEED: u64 = 1234;

/// Qwen4Exp's typed config.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4ExpParams {
    pub(super) vocab: usize,
    pub(super) width: usize,
    pub(super) max_positions: usize,
    pub(super) kinds: Vec<LayerKind>,
    pub(super) attention: AttentionParams,
    pub(super) qsa: QsaParams,
    pub(super) gdn: GdnParams,
    pub(super) moe: MoeParams,
    pub(super) hyper: HyperParams,
    /// One per layer: the PLE block in front of it, for the layers `ple_layer_ids` names.
    pub(super) ple: Vec<Option<PleParams>>,
    /// `indexer_compress_ratio`: the prefill granule, and the width of an indexer block.
    pub(super) ratio: NonZeroUsize,
    pub(super) head_required: bool,
    pub(super) eos: BTreeSet<u32>,
    pub(super) bos: Option<u32>,
}

/// The text tower's config: `text_config`, with the token ids and the tie flag the wrapper config
/// carries.
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

impl Qwen4ExpParams {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
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
        let param_error = |e: ParamError| e.for_family(FAMILY);
        let dims = DenseDims::read_with_inter(&fields, &SPEC, EXPERT_INTER)?;
        for unsupported in ["attention_bias", "use_sliding_window"] {
            if fields.flag(Field::hf(unsupported), false)? {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        let kinds = layer_kinds(&text, dims.layers, &fields)?;

        // Every `Qwen4ExpTextRMSNorm` stores its scale as an offset from one.
        let norm = NormParams::of(NormKind::RmsPlusOne, dims.eps).map_err(param_error)?;
        let factor = fields
            .opt_f32(PARTIAL_ROTARY)?
            .ok_or(fields.error("partial_rotary_factor", ConfigReason::Missing))?;
        if !(factor.is_finite() && factor > 0.0 && factor <= 1.0) {
            return Err(fields.error("partial_rotary_factor", ConfigReason::NotFinitePositive));
        }
        let rotary = ((dims.head_dim as f32 * factor).round() as usize).max(1);
        let rope = dims.rope(&fields, &SPEC, rotary)?;
        let attention = AttentionParams::new(dims.heads, dims.kv_heads, dims.head_dim, false, rope)
            .and_then(|a| a.with_output_gate())
            .map_err(param_error)?
            .with_qk_norm(QkNorm::PerHead(norm));

        let ratio = fields.count(INDEX_RATIO, None)?;
        let qsa = qsa_params(&fields, norm, rope, ratio)?;

        let head_dim = fields.count(KEY_HEAD_DIM, None)?;
        if fields.count(VALUE_HEAD_DIM, Some(head_dim))? != head_dim {
            return Err(fields.error("linear_value_head_dim", ConfigReason::Unsupported));
        }
        // The gated norm scale is plain (not an offset from one); the decay is stored as `A_log`.
        let gdn = GdnParams::new(
            fields.count(KEY_HEADS, None)?,
            fields.count(VALUE_HEADS, None)?,
            head_dim,
            fields.count(CONV_KERNEL, None)?,
            NormParams::new(dims.eps).map_err(param_error)?,
            GdnHeadOrder::Grouped,
        )
        .map_err(param_error)?
        .with_log_decay();

        let renormalize = fields.flag(NORM_TOPK_PROB, true)?;
        let moe = MoeParams::new(
            FAMILY,
            dims.width,
            dims.inter,
            fields.count(EXPERTS, None)?,
            fields.count(TOP_K, None)?,
            RouterKind::SoftmaxTopK { renormalize },
        )?
        .with_shared(fields.count(SHARED_INTER, None)?)
        .and_then(|m| m.with_shared_gate())
        .map_err(param_error)?;

        let hyper = HyperParams::new(
            dims.width,
            fields.count(HC_COUNT, None)?,
            fields.count(HC_RANK, None)?,
            dims.eps,
        )
        .map_err(param_error)?;
        // The hash's fill is the text model's own `eos_token_id`; the stop set (`dims.eos`) takes the
        // wrapper's ids over it, as a generation config does.
        let own_eos = config
            .get("text_config")
            .and_then(|t| t.get("eos_token_id"))
            .filter(|v| !v.is_null());
        let ple = ple_params(&text, own_eos, &fields, &dims, &kinds, hyper.streams())?;

        let ratio = NonZeroUsize::new(ratio).expect("counted non-zero above");
        Ok(Self {
            vocab: dims.vocab,
            width: dims.width,
            max_positions: dims.max_positions,
            kinds,
            attention,
            qsa,
            gdn,
            moe,
            hyper,
            ple,
            ratio,
            head_required: dims.head_required,
            eos: dims.eos,
            bos: dims.bos,
        })
    }
}

fn qsa_params(
    fields: &Fields<'_>,
    norm: NormParams,
    rope: RopeParams,
    ratio: usize,
) -> Result<QsaParams, ModelError> {
    // The indexer is MQA-shaped: one shared key per token.
    if fields.count(INDEX_KV_HEADS, None)? != 1 {
        return Err(fields.error("indexer_kv_heads", ConfigReason::Unsupported));
    }
    QsaParams::new(
        fields.count(INDEX_HEADS, None)?,
        fields.count(INDEX_DIM, None)?,
        fields.count(INDEX_BUDGET, None)?,
        ratio,
        rope,
        norm,
    )
    .map_err(|e| e.for_family(FAMILY))
}

/// Which layers are full attention: `layer_types`, one `"full_attention"` or `"linear_attention"`
/// per layer.
fn layer_kinds(
    text: &serde_json::Value,
    layers: usize,
    fields: &Fields<'_>,
) -> Result<Vec<LayerKind>, ModelError> {
    let types = text
        .get("layer_types")
        .filter(|v| !v.is_null())
        .ok_or(fields.error("layer_types", ConfigReason::Missing))?;
    let wrong = || fields.error("layer_types", ConfigReason::WrongType);
    let types = types.as_array().ok_or_else(wrong)?;
    if types.len() != layers {
        return Err(fields.error("layer_types", ConfigReason::Length { expected: layers }));
    }
    types
        .iter()
        .map(|t| match t.as_str().ok_or_else(wrong)? {
            "full_attention" => Ok(LayerKind::FullAttention),
            "linear_attention" => Ok(LayerKind::LinearAttention),
            _ => Err(fields.error("layer_types", ConfigReason::Unsupported)),
        })
        .collect()
}

/// The PLE block of each layer `ple_layer_ids` names. The ids are one-indexed (`[2]` is the
/// second layer), as the reference config validates them, and every one names a linear-attention
/// layer.
fn ple_params(
    text: &serde_json::Value,
    own_eos: Option<&serde_json::Value>,
    fields: &Fields<'_>,
    dims: &DenseDims,
    kinds: &[LayerKind],
    streams: usize,
) -> Result<Vec<Option<PleParams>>, ModelError> {
    let mut out: Vec<Option<PleParams>> = vec![None; kinds.len()];
    let Some(ids) = text.get("ple_layer_ids").filter(|v| !v.is_null()) else {
        return Ok(out);
    };
    let wrong = || fields.error("ple_layer_ids", ConfigReason::WrongType);
    let ids = ids.as_array().ok_or_else(wrong)?;
    if ids.is_empty() {
        return Ok(out);
    }
    let eos = match own_eos {
        Some(id) => id
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .ok_or(fields.error("eos_token_id", ConfigReason::WrongType))?,
        None => fields
            .token(Field::hf("eos_token_id"))?
            .ok_or(fields.error("eos_token_id", ConfigReason::Missing))?,
    };
    let seed = fields.opt_u64(NGRAM_SEED)?.unwrap_or(DEFAULT_SEED);
    let base = fields
        .opt_u64(NGRAM_BASE)?
        .ok_or(fields.error("ngram_vocab_size_base", ConfigReason::Missing))?;
    let spec = |ordinal| -> Result<PleSpec, ModelError> {
        Ok(PleSpec {
            width: dims.width,
            streams,
            embed_dim: fields.count(PLE_EMBED, None)?,
            conv_kernel: fields.count(PLE_KERNEL, None)?,
            ngram_size: fields.count(NGRAM_SIZE, None)?,
            heads_per_ngram: fields.count(NGRAM_HEADS, None)?,
            ngram_vocab_base: base,
            vocab_divisible_by: fields.count(NGRAM_DIVISIBLE, None)?,
            seed,
            vocab: dims.vocab,
            eos,
            eps: dims.eps,
            ordinal,
        })
    };
    for (ordinal, id) in ids.iter().enumerate() {
        let id = id.as_u64().ok_or_else(wrong)? as usize;
        if !(1..=kinds.len()).contains(&id) {
            return Err(fields.error("ple_layer_ids", ConfigReason::Exceeds { max: kinds.len() }));
        }
        let layer = id - 1;
        if kinds[layer] != LayerKind::LinearAttention || out[layer].is_some() {
            return Err(fields.error("ple_layer_ids", ConfigReason::Unsupported));
        }
        out[layer] = Some(PleParams::new(spec(ordinal)?).map_err(|e| e.for_family(FAMILY))?);
    }
    Ok(out)
}

/// Where the checkpoint keeps the text tower.
const TEXT: &str = "model.language_model";
const LAYERS: &str = "model.language_model.layers.";

const MOE_SHARED: SharedNames = SharedNames {
    gate: ".mlp.shared_expert.gate_proj",
    up: ".mlp.shared_expert.up_proj",
    down: ".mlp.shared_expert.down_proj",
    scalar_gate: Some(".mlp.shared_expert_gate"),
};

/// A checkpoint that fuses gate and up into `mlp.experts.gate_up_proj` (`[experts, 2 * inter,
/// width]`, gate rows first).
const MOE_FUSED: MoeNames = MoeNames {
    prefix: LAYERS,
    router: ".mlp.gate",
    router_bias: None,
    shared: Some(MOE_SHARED),
    experts: &[
        ExpertNames {
            proj: ExpertProj::GateUp,
            stack: Some(".mlp.experts.gate_up_proj"),
            each: None,
            bias: None,
        },
        ExpertNames {
            proj: ExpertProj::Down,
            stack: Some(".mlp.experts.down_proj"),
            each: None,
            bias: None,
        },
    ],
};

/// A checkpoint that stores each expert's three projections as their own tensors (the packed
/// release; a dense one of the same layout stacks them).
const MOE_SPLIT: MoeNames = MoeNames {
    prefix: LAYERS,
    router: ".mlp.gate",
    router_bias: None,
    shared: Some(MOE_SHARED),
    experts: &[
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
    ],
};

/// How a store holds the routed experts: the stored layout decides the gate/up form, as a config
/// cannot say. A tensor named per expert (`mlp.experts.{e}.`) is the split layout; a store with none
/// is read as fused, so a store missing its experts is told to lack the fused tensor.
pub(super) fn expert_layout(store: &WeightStore) -> (&'static MoeNames, GateUpForm) {
    const PER_EXPERT: &str = ".mlp.experts.";
    let split = store.keys().any(|key| {
        key.as_str()
            .split_once(PER_EXPERT)
            .is_some_and(|(_, rest)| rest.starts_with(|c: char| c.is_ascii_digit()))
    });
    if split {
        (&MOE_SPLIT, GateUpForm::Split)
    } else {
        (&MOE_FUSED, GateUpForm::FusedHalves)
    }
}

/// Map `id` onto the stored tensor `key`.
fn put(map: &mut WeightMapBuilder<'_>, id: WeightId, key: String) -> Result<(), ModelError> {
    map.map(id, WeightView::Stored(WeightKey::from(key)))
        .map_err(|source| ModelError::Weight {
            family: FAMILY,
            source,
        })?;
    Ok(())
}

/// The weight map of a checkpoint: the model-level weights and, per layer, the two Hyper-Connections
/// gates, the mixer its kind names, the PLE block where the layer has one and the routed block. A
/// tensor the store lacks is [`WeightMapError::Missing`], naming its key, before any trace.
pub(super) fn weights_of(
    p: &Qwen4ExpParams,
    store: &WeightStore,
    moe_names: &MoeNames,
) -> Result<WeightMap, ModelError> {
    let mut map = WeightMap::builder(store);
    let missing = |source: WeightMapError| ModelError::Weight {
        family: FAMILY,
        source,
    };
    // The stored key of a linear: `{stem}.weight`, or the bare stem a packed loader names it.
    let linear = |stem: &str| {
        let dense = format!("{stem}.weight");
        if store.contains(&dense) || !store.contains(stem) {
            dense
        } else {
            stem.to_string()
        }
    };

    let embed = WeightId::model(WeightRole::Embed);
    let embed_key = format!("{TEXT}.embed_tokens.weight");
    put(&mut map, embed, embed_key.clone())?;
    let head = WeightId::model(WeightRole::Head);
    let head_key = linear("lm_head");
    if store.contains(&head_key) {
        put(&mut map, head, head_key)?;
    } else if p.head_required {
        return Err(missing(WeightMapError::Missing {
            id: head,
            key: WeightKey::from(head_key),
        }));
    } else {
        put(&mut map, head, embed_key)?;
    }
    // One Hyper-Connections gate: the grouped norm scale, the two low-rank projections and, unless
    // it is the model-level mixer, the per-stream injection weights.
    let hyper = |map: &mut WeightMapBuilder<'_>,
                 id: &dyn Fn(HyperRole) -> WeightId,
                 stem: String,
                 inject: bool|
     -> Result<(), ModelError> {
        put(map, id(HyperRole::Norm), format!("{stem}.hc_norm.weight"))?;
        for (role, suffix) in [
            (HyperRole::MixDown, ".input_mix_weight_down"),
            (HyperRole::MixUp, ".input_mix_weight_up"),
        ] {
            put(map, id(role), linear(&format!("{stem}{suffix}")))?;
        }
        if inject {
            put(
                map,
                id(HyperRole::Inject),
                linear(&format!("{stem}.block_inject_weight")),
            )?;
        }
        Ok(())
    };
    hyper(
        &mut map,
        &|role| WeightId::model(WeightRole::Hyper(HyperSite::Final, role)),
        format!("{TEXT}.hyper_connection_mixer"),
        false,
    )?;

    for (l, kind) in p.kinds.iter().enumerate() {
        let layer = |suffix: &str| format!("{LAYERS}{l}{suffix}");
        let id = |role| WeightId::layer(l, role);
        for (site, stem) in [
            (HyperSite::Attn, ".attn_hyper_connection"),
            (HyperSite::Mlp, ".mlp_hyper_connection"),
        ] {
            hyper(
                &mut map,
                &|role| id(WeightRole::Hyper(site, role)),
                layer(stem),
                true,
            )?;
        }
        if p.ple[l].is_some() {
            // The official release stores the table as numbered E4M3 shards under one scale; a
            // smaller checkpoint stores it as one dense weight.
            let table = layer(".ple.ple_embedding.ngram_embedding");
            let shard = |ordinal: u32| format!("{table}.shard_{ordinal}.weight");
            if store.contains(&shard(0)) {
                for ordinal in (0..).take_while(|&ordinal| store.contains(&shard(ordinal))) {
                    put(
                        &mut map,
                        id(WeightRole::Ple(PleRole::TableShard(ordinal))),
                        shard(ordinal),
                    )?;
                }
                put(
                    &mut map,
                    id(WeightRole::Ple(PleRole::TableScale)),
                    format!("{table}.weight_scale"),
                )?;
            } else {
                put(
                    &mut map,
                    id(WeightRole::Ple(PleRole::Table)),
                    format!("{table}.weight"),
                )?;
            }
            for (role, key) in [
                (PleRole::Key, linear(&layer(".ple.key_proj"))),
                (PleRole::Value, linear(&layer(".ple.value_proj"))),
                (PleRole::KeyNorm, layer(".ple.norm_key.weight")),
                (PleRole::QueryNorm, layer(".ple.norm_query.weight")),
                (PleRole::ConvNorm, layer(".ple.norm_conv.weight")),
                (PleRole::Conv, layer(".ple.conv1d.weight")),
            ] {
                put(&mut map, id(WeightRole::Ple(role)), key)?;
            }
        }
        match kind {
            LayerKind::FullAttention => {
                for (role, stem) in [
                    (AttnRole::Q, ".self_attn.q_proj"),
                    (AttnRole::K, ".self_attn.k_proj"),
                    (AttnRole::V, ".self_attn.v_proj"),
                    (AttnRole::O, ".self_attn.o_proj"),
                ] {
                    put(&mut map, id(WeightRole::Attn(role)), linear(&layer(stem)))?;
                }
                for (role, key) in [
                    (AttnRole::QNorm, ".self_attn.q_norm.weight"),
                    (AttnRole::KNorm, ".self_attn.k_norm.weight"),
                ] {
                    put(&mut map, id(WeightRole::Attn(role)), layer(key))?;
                }
                // The indexer's query and key rows are one fused projection, query rows first.
                let fused = linear(&layer(".self_attn.indexer.index_qk_proj"));
                let query_rows = p.qsa.query_rows();
                for (role, rows) in [
                    (QsaRole::Q, 0..query_rows),
                    (QsaRole::K, query_rows..query_rows + p.qsa.dim()),
                ] {
                    let view = WeightView::RowRange {
                        key: WeightKey::from(fused.as_str()),
                        rows,
                    };
                    map.map(id(WeightRole::Qsa(role)), view).map_err(missing)?;
                }
                for (role, key) in [
                    (QsaRole::QNorm, ".self_attn.indexer.q_layernorm.weight"),
                    (QsaRole::KNorm, ".self_attn.indexer.k_layernorm.weight"),
                ] {
                    put(&mut map, id(WeightRole::Qsa(role)), layer(key))?;
                }
            }
            LayerKind::LinearAttention => {
                let gdn = |role| id(WeightRole::Gdn(role));
                for (role, stem) in [
                    (GdnRole::Qkv, ".linear_attn.in_proj_qkv"),
                    (GdnRole::Z, ".linear_attn.in_proj_z"),
                    (GdnRole::Beta, ".linear_attn.in_proj_b"),
                    (GdnRole::Alpha, ".linear_attn.in_proj_a"),
                    (GdnRole::Out, ".linear_attn.out_proj"),
                ] {
                    put(&mut map, gdn(role), linear(&layer(stem)))?;
                }
                for (role, key) in [
                    (GdnRole::DtBias, ".linear_attn.dt_bias"),
                    (GdnRole::Conv, ".linear_attn.conv1d.weight"),
                    (GdnRole::A, ".linear_attn.A_log"),
                    (GdnRole::GateNorm, ".linear_attn.norm.weight"),
                ] {
                    put(&mut map, gdn(role), layer(key))?;
                }
            }
        }
    }
    for l in 0..p.kinds.len() {
        map_moe_layer(&mut map, store, moe_names, l, p.moe.experts()).map_err(missing)?;
    }
    Ok(map.build())
}
