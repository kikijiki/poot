//! Qwen4Exp's typed config and weight-name table, both data: every config value is checked once,
//! here, so tracing cannot fail on it, and the checkpoint's tensor names map onto typed weight ids
//! in one table per format. Nothing here pins a shipped checkpoint: any layer, expert and head count
//! the architecture admits loads (a finetune of a different size is the same family).
//!
//! Two formats carry the architecture. An HF checkpoint (`qwen4_exp`) states its config in
//! `config.json` and stores the reference model's tensors. A GGUF (`qwen4exp`) states it in
//! metadata and stores what llama.cpp's converter wrote: the one-offset norm scales folded to
//! plain scales, the GDN decay as `-exp(A_log)` with its value heads in tiled order, the indexer's
//! query and key rows as two tensors, the n-gram table as one tensor, and the table's hash
//! constants as metadata arrays.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_load::gguf::GgufIndex;
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
use crate::components::ple::{NgramHash, PleParams, PleSpec};
use crate::components::rope::RopeParams;
use crate::components::router::RouterKind;
use crate::components::standard::ParamError;
use crate::model::{ConfigReason, ModelError};
use crate::names::{ExpertNames, MoeNames, SharedNames, map_moe_layer};
use crate::registry::RawConfig;

/// The config facts the HF `qwen4_exp` class and llama.cpp's `qwen4exp` default.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("qwen4exp"),
    hf_theta: 10_000.0,
    gguf_theta: 10_000_000.0,
    eps: 1e-6,
    eos: None,
};

const EXPERT_INTER: Field = Field::new(
    "moe_intermediate_size",
    "qwen4exp.expert_feed_forward_length",
);
const SHARED_INTER: Field = Field::new(
    "shared_expert_intermediate_size",
    "qwen4exp.expert_shared_feed_forward_length",
);
const EXPERTS: Field = Field::new("num_experts", "qwen4exp.expert_count");
const TOP_K: Field = Field::new("num_experts_per_tok", "qwen4exp.expert_used_count");
const NORM_TOPK_PROB: Field = Field::hf("norm_topk_prob");
const KEY_HEADS: Field = Field::new("linear_num_key_heads", "qwen4exp.ssm.group_count");
const VALUE_HEADS: Field = Field::new("linear_num_value_heads", "qwen4exp.ssm.time_step_rank");
const KEY_HEAD_DIM: Field = Field::new("linear_key_head_dim", "qwen4exp.ssm.state_size");
const VALUE_HEAD_DIM: Field = Field::hf("linear_value_head_dim");
const CONV_KERNEL: Field = Field::new("linear_conv_kernel_dim", "qwen4exp.ssm.conv_kernel");
const INNER_SIZE: Field = Field::new("", "qwen4exp.ssm.inner_size");
const PARTIAL_ROTARY: Field = Field::hf("partial_rotary_factor");
const ROTARY_DIM: Field = Field::new("", "qwen4exp.rope.dimension_count");
const INDEX_HEADS: Field = Field::new("indexer_n_heads", "qwen4exp.attention.indexer.head_count");
const INDEX_KV_HEADS: Field = Field::hf("indexer_kv_heads");
const INDEX_DIM: Field = Field::new("indexer_head_dim", "qwen4exp.attention.indexer.key_length");
const INDEX_BUDGET: Field = Field::new("indexer_budget", "qwen4exp.attention.indexer.top_k");
const INDEX_RATIO: Field = Field::hf("indexer_compress_ratio");
const HC_COUNT: Field = Field::new("hc_count", "qwen4exp.hyper_connection.count");
const HC_RANK: Field = Field::new("hc_lowrank", "qwen4exp.hyper_connection.low_rank");
const PLE_EMBED: Field = Field::hf("ple_embed_dim");
const PLE_KERNEL: Field = Field::new("ple_conv_kernel_size", "qwen4exp.ple.conv_kernel");
const NGRAM_SIZE: Field = Field::new("ngram_size", "qwen4exp.ple.ngram_size");
const NGRAM_BASE: Field = Field::hf("ngram_vocab_size_base");
const NGRAM_HEADS: Field = Field::new("heads_per_ngram", "qwen4exp.ple.heads_per_ngram");
const NGRAM_DIVISIBLE: Field = Field::hf("make_ngram_vocab_size_divisible_by");
const NGRAM_SEED: Field = Field::hf("seed");

/// The GGUF metadata only a GGUF carries: the per-layer compress ratios (zero on a linear-attention
/// layer, the indexer's block width on a full-attention one), the layer interval they must agree
/// with, and the n-gram block's layer list, row width, end-of-sequence id and stored hash constants.
const GGUF_RATIOS: &str = "qwen4exp.attention.compress_ratios";
const GGUF_INTERVAL: Field = Field::new("", "qwen4exp.full_attention_interval");
const GGUF_PLE_LAYERS: &str = "qwen4exp.ple.layers";
const GGUF_PLE_ROW: Field = Field::new("", "qwen4exp.embedding_length_per_layer_input");
const GGUF_PLE_EOS: Field = Field::new("", "qwen4exp.ple.eos_token_id");
const GGUF_PLE_MULTIPLIERS: &str = "qwen4exp.ple.layer_multipliers";
const GGUF_PLE_OFFSETS: &str = "qwen4exp.ple.head_offsets";
const GGUF_PLE_VOCABS: &str = "qwen4exp.ple.head_vocab_sizes";

/// The reference model's default hash seed (`configuration_qwen4_exp.py`).
const DEFAULT_SEED: u64 = 1234;

/// Which format a checkpoint is in: the two name the same architecture with different tensors and
/// stored conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    /// An HF `config.json` and its safetensors.
    Hf,
    /// A GGUF as llama.cpp's converter writes it.
    Gguf,
}

/// Qwen4Exp's typed config.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Qwen4ExpParams {
    pub(super) layout: Layout,
    pub(super) vocab: usize,
    pub(super) width: usize,
    pub(super) max_positions: usize,
    pub(super) kinds: Vec<LayerKind>,
    pub(super) attention: AttentionParams,
    pub(super) qsa: QsaParams,
    pub(super) gdn: GdnParams,
    pub(super) moe: MoeParams,
    pub(super) hyper: HyperParams,
    /// One per layer: the PLE block in front of it, for the layers the checkpoint names.
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
    /// Reads the typed config from an HF `config.json` or a GGUF's metadata; a field the family
    /// cannot serve is a typed [`ModelError::Config`] naming the key the checkpoint used.
    pub(crate) fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        match raw {
            RawConfig::HfJson { config, generation } => {
                let text = text_config(config)?;
                // The hash's fill is the text model's own `eos_token_id`; the stop set (`dims.eos`)
                // takes the wrapper's ids over it, as a generation config does.
                let own_eos = config
                    .get("text_config")
                    .and_then(|t| t.get("eos_token_id"))
                    .filter(|v| !v.is_null());
                let raw = RawConfig::HfJson {
                    config: &text,
                    generation: *generation,
                };
                Self::read(&raw, Layout::Hf, &text, own_eos)
            }
            RawConfig::Gguf(_) => Self::read(raw, Layout::Gguf, &serde_json::Value::Null, None),
        }
    }

    /// The config of `raw` in `layout`; `text` is the HF text tower (null for a GGUF).
    fn read(
        raw: &RawConfig<'_>,
        layout: Layout,
        text: &serde_json::Value,
        own_eos: Option<&serde_json::Value>,
    ) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let param_error = |e: ParamError| e.for_family(FAMILY);
        let dims = DenseDims::read_with_inter(&fields, &SPEC, EXPERT_INTER)?;
        if layout == Layout::Hf {
            for unsupported in ["attention_bias", "use_sliding_window"] {
                if fields.flag(Field::hf(unsupported), false)? {
                    return Err(fields.error(unsupported, ConfigReason::Unsupported));
                }
            }
        }
        let (kinds, ratio) = match raw {
            RawConfig::HfJson { .. } => (
                layer_kinds(text, dims.layers, &fields)?,
                fields.count(INDEX_RATIO, None)?,
            ),
            RawConfig::Gguf(gguf) => gguf_layer_kinds(gguf, dims.layers, &fields)?,
        };

        // HF stores every `Qwen4ExpTextRMSNorm` scale as an offset from one; a GGUF folds the one in.
        let norm_kind = match layout {
            Layout::Hf => NormKind::RmsPlusOne,
            Layout::Gguf => NormKind::Rms,
        };
        let norm = NormParams::of(norm_kind, dims.eps).map_err(param_error)?;
        let rotary = match raw {
            RawConfig::HfJson { .. } => {
                let factor = fields
                    .opt_f32(PARTIAL_ROTARY)?
                    .ok_or(fields.error("partial_rotary_factor", ConfigReason::Missing))?;
                if !(factor.is_finite() && factor > 0.0 && factor <= 1.0) {
                    return Err(
                        fields.error("partial_rotary_factor", ConfigReason::NotFinitePositive)
                    );
                }
                ((dims.head_dim as f32 * factor).round() as usize).max(1)
            }
            RawConfig::Gguf(_) => fields.count(ROTARY_DIM, None)?,
        };
        let rope = dims.rope(&fields, &SPEC, rotary)?;
        let attention = AttentionParams::new(dims.heads, dims.kv_heads, dims.head_dim, false, rope)
            .and_then(|a| a.with_output_gate())
            .map_err(param_error)?
            .with_qk_norm(QkNorm::PerHead(norm));

        let qsa = qsa_params(&fields, norm, rope, ratio)?;

        let head_dim = fields.count(KEY_HEAD_DIM, None)?;
        if fields.count(VALUE_HEAD_DIM, Some(head_dim))? != head_dim {
            return Err(fields.error("linear_value_head_dim", ConfigReason::Unsupported));
        }
        let (key_heads, value_heads) = (
            fields.count(KEY_HEADS, None)?,
            fields.count(VALUE_HEADS, None)?,
        );
        if let Some(inner) = fields.opt_u64(INNER_SIZE)?
            && inner as usize != value_heads * head_dim
        {
            return Err(fields.error(fields.key(INNER_SIZE), ConfigReason::Unsupported));
        }
        // The gated norm scale is plain (not an offset from one) in both formats. HF stores the
        // decay as `A_log` with the value heads grouped by key head; a GGUF stores `-exp(A_log)`
        // with the value heads in tiled order.
        let gdn = GdnParams::new(
            key_heads,
            value_heads,
            head_dim,
            fields.count(CONV_KERNEL, None)?,
            NormParams::new(dims.eps).map_err(param_error)?,
            match layout {
                Layout::Hf => GdnHeadOrder::Grouped,
                Layout::Gguf => GdnHeadOrder::Tiled,
            },
        )
        .map_err(param_error)?;
        let gdn = match layout {
            Layout::Hf => gdn.with_log_decay(),
            Layout::Gguf => gdn,
        };

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
            norm,
        )
        .map_err(param_error)?;
        let ple = match raw {
            RawConfig::HfJson { .. } => {
                ple_params(text, own_eos, &fields, &dims, &kinds, hyper.streams(), norm)?
            }
            RawConfig::Gguf(gguf) => {
                gguf_ple_params(gguf, &fields, &dims, &kinds, hyper.streams(), norm)?
            }
        };

        let ratio = NonZeroUsize::new(ratio)
            .ok_or_else(|| fields.error(INDEX_RATIO.hf, ConfigReason::Zero))?;
        Ok(Self {
            layout,
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
    // The indexer is MQA-shaped: one shared key per token (a GGUF states no key-head count).
    if fields.count(INDEX_KV_HEADS, Some(1))? != 1 {
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

/// A GGUF array of unsigned integers under `key`; `None` when the key is absent.
fn gguf_u64s(
    gguf: &GgufIndex,
    fields: &Fields<'_>,
    key: &'static str,
) -> Result<Option<Vec<u64>>, ModelError> {
    let wrong = || fields.error(key, ConfigReason::WrongType);
    gguf.get(key)
        .map(|value| {
            value
                .as_array()
                .ok_or_else(wrong)?
                .iter()
                .map(|v| v.as_u64().ok_or_else(wrong))
                .collect()
        })
        .transpose()
}

/// Which layers are full attention, and the indexer's block width: the per-layer compress ratios,
/// zero on a linear-attention layer and the one block width on a full-attention layer. The layer
/// interval a GGUF may also state must agree with them.
fn gguf_layer_kinds(
    gguf: &GgufIndex,
    layers: usize,
    fields: &Fields<'_>,
) -> Result<(Vec<LayerKind>, usize), ModelError> {
    let ratios = gguf_u64s(gguf, fields, GGUF_RATIOS)?
        .ok_or(fields.error(GGUF_RATIOS, ConfigReason::Missing))?;
    if ratios.len() != layers {
        return Err(fields.error(GGUF_RATIOS, ConfigReason::Length { expected: layers }));
    }
    let mut ratio = None;
    let mut kinds = Vec::with_capacity(layers);
    for &r in &ratios {
        if r == 0 {
            kinds.push(LayerKind::LinearAttention);
            continue;
        }
        if *ratio.get_or_insert(r) != r {
            // One block width: the prefill granule and the indexer's pooling share it.
            return Err(fields.error(GGUF_RATIOS, ConfigReason::Unsupported));
        }
        kinds.push(LayerKind::FullAttention);
    }
    let ratio = ratio.ok_or(fields.error(GGUF_RATIOS, ConfigReason::Unsupported))?;
    let ratio =
        usize::try_from(ratio).map_err(|_| fields.error(GGUF_RATIOS, ConfigReason::WrongType))?;
    if let Some(interval) = fields.opt_u64(GGUF_INTERVAL)? {
        let interval = usize::try_from(interval)
            .ok()
            .filter(|&n| n > 0)
            .ok_or(fields.error(fields.key(GGUF_INTERVAL), ConfigReason::Zero))?;
        let agrees = kinds.iter().enumerate().all(|(l, kind)| {
            (*kind == LayerKind::FullAttention) == (l + 1).is_multiple_of(interval)
        });
        if !agrees {
            return Err(fields.error(fields.key(GGUF_INTERVAL), ConfigReason::Unsupported));
        }
    }
    Ok((kinds, ratio))
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
    norm: NormParams,
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
            hash: NgramHash::Derived {
                vocab_base: base,
                vocab_divisible_by: fields.count(NGRAM_DIVISIBLE, None)?,
                seed,
            },
            vocab: dims.vocab,
            eos,
            norm,
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

/// The GGUF tensor that holds the n-gram table.
const GGUF_PLE_TABLE: &str = "per_layer_token_embd.weight";

/// The PLE block a GGUF names (`ple.layers`, zero-indexed). A GGUF holds one table and one set of
/// hash constants, so it can describe one PLE layer. The hash constants are read as stored, and the
/// table's row count is the table tensor's own (the converter pads it).
fn gguf_ple_params(
    gguf: &GgufIndex,
    fields: &Fields<'_>,
    dims: &DenseDims,
    kinds: &[LayerKind],
    streams: usize,
    norm: NormParams,
) -> Result<Vec<Option<PleParams>>, ModelError> {
    let mut out: Vec<Option<PleParams>> = vec![None; kinds.len()];
    let layers = gguf_u64s(gguf, fields, GGUF_PLE_LAYERS)?.unwrap_or_default();
    let [layer] = layers[..] else {
        if layers.is_empty() {
            return Ok(out);
        }
        return Err(fields.error(GGUF_PLE_LAYERS, ConfigReason::Unsupported));
    };
    let layer = usize::try_from(layer)
        .ok()
        .filter(|&l| l < kinds.len())
        .ok_or(fields.error(
            GGUF_PLE_LAYERS,
            ConfigReason::Exceeds {
                max: kinds.len() - 1,
            },
        ))?;
    if kinds[layer] != LayerKind::LinearAttention {
        return Err(fields.error(GGUF_PLE_LAYERS, ConfigReason::Unsupported));
    }
    let required = |key: &'static str| {
        gguf_u64s(gguf, fields, key)?.ok_or(fields.error(key, ConfigReason::Missing))
    };
    let (ngram_size, heads_per_ngram) = (
        fields.count(NGRAM_SIZE, None)?,
        fields.count(NGRAM_HEADS, None)?,
    );
    // The table's rows are `[rows, row]`, one hash head's share of the embedding per row.
    let row = fields.count(GGUF_PLE_ROW, None)?;
    let table = gguf
        .tensors
        .get(GGUF_PLE_TABLE)
        .ok_or(fields.error(GGUF_PLE_TABLE, ConfigReason::Missing))?;
    let rows = match table.dims[..] {
        [_, rows] => usize::try_from(rows)
            .map_err(|_| fields.error(GGUF_PLE_TABLE, ConfigReason::WrongType))?,
        _ => return Err(fields.error(GGUF_PLE_TABLE, ConfigReason::WrongType)),
    };
    let eos = fields
        .token(GGUF_PLE_EOS)?
        .ok_or(fields.error(fields.key(GGUF_PLE_EOS), ConfigReason::Missing))?;
    let spec = PleSpec {
        width: dims.width,
        streams,
        embed_dim: row * (ngram_size.saturating_sub(1)) * heads_per_ngram,
        conv_kernel: fields.count(PLE_KERNEL, None)?,
        ngram_size,
        heads_per_ngram,
        hash: NgramHash::Stored {
            multipliers: required(GGUF_PLE_MULTIPLIERS)?,
            head_offsets: required(GGUF_PLE_OFFSETS)?,
            head_vocab_sizes: required(GGUF_PLE_VOCABS)?,
            rows,
        },
        vocab: dims.vocab,
        eos,
        norm,
        ordinal: 0,
    };
    out[layer] = Some(PleParams::new(spec).map_err(|e| e.for_family(FAMILY))?);
    Ok(out)
}

/// One Hyper-Connections gate's tensor names: the grouped norm scale, the two low-rank
/// projections and, for a layer's gate, the per-stream injection.
struct HyperNames {
    norm: &'static str,
    down: &'static str,
    up: &'static str,
    inject: &'static str,
}

/// A checkpoint format's tensor names. A layer's tensors are `{layers}{l}{suffix}`; the model-level
/// ones are whole names. A `linear` stem also reads the packed loader's bare linear id (see
/// [`weights_of`]).
struct Names {
    embed: &'static str,
    head: &'static str,
    /// The gate that collapses the streams into the head: `{final_stem}` plus a [`HyperNames`] suffix.
    final_gate: HyperNames,
    layers: &'static str,
    attn_gate: HyperNames,
    mlp_gate: HyperNames,
    attention: AttentionNames,
    gdn: GdnNames,
    ple: PleNames,
}

struct AttentionNames {
    q: &'static str,
    k: &'static str,
    v: &'static str,
    o: &'static str,
    q_norm: &'static str,
    k_norm: &'static str,
    /// The indexer's tensors.
    indexer: IndexerNames,
}

/// Where the indexer's query and key rows are: one fused projection, query rows first (HF), or one
/// tensor each (a GGUF).
#[derive(Clone, Copy)]
enum IndexerProj {
    Fused(&'static str),
    Split { q: &'static str, k: &'static str },
}

struct IndexerNames {
    proj: IndexerProj,
    q_norm: &'static str,
    k_norm: &'static str,
}

struct GdnNames {
    qkv: &'static str,
    z: &'static str,
    beta: &'static str,
    alpha: &'static str,
    out: &'static str,
    dt_bias: &'static str,
    conv: &'static str,
    a: &'static str,
    norm: &'static str,
}

struct PleNames {
    /// The n-gram table: a layer suffix (HF), or a whole name shared by the model (a GGUF).
    table: PleTable,
    key: &'static str,
    value: &'static str,
    key_norm: &'static str,
    query_norm: &'static str,
    conv_norm: &'static str,
    conv: &'static str,
}

#[derive(Clone, Copy)]
enum PleTable {
    /// `{layers}{l}{stem}.weight`, or numbered E4M3 shards under one scale.
    PerLayer(&'static str),
    /// One tensor for the model.
    Model(&'static str),
}

/// A checkpoint that fuses gate and up into `mlp.experts.gate_up_proj` (`[experts, 2 * inter,
/// width]`, gate rows first).
const MOE_FUSED: MoeNames = MoeNames {
    prefix: HF_LAYERS,
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
    prefix: HF_LAYERS,
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

const MOE_SHARED: SharedNames = SharedNames {
    gate: ".mlp.shared_expert.gate_proj",
    up: ".mlp.shared_expert.up_proj",
    down: ".mlp.shared_expert.down_proj",
    scalar_gate: Some(".mlp.shared_expert_gate"),
};

/// A GGUF: each projection is one stacked tensor `[experts, out, in]`, which the loader keeps as
/// one packed owner per expert.
const MOE_GGUF: MoeNames = MoeNames {
    prefix: GGUF_LAYERS,
    router: ".ffn_gate_inp",
    router_bias: None,
    shared: Some(SharedNames {
        gate: ".ffn_gate_shexp",
        up: ".ffn_up_shexp",
        down: ".ffn_down_shexp",
        scalar_gate: Some(".ffn_gate_inp_shexp"),
    }),
    experts: &[
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
    ],
};

/// Where each format keeps the layers.
const HF_LAYERS: &str = "model.language_model.layers.";
const GGUF_LAYERS: &str = "blk.";

/// An HF layer's gate: `{stem}` plus the suffix.
const HF_GATE: HyperNames = HyperNames {
    norm: ".hc_norm.weight",
    down: ".input_mix_weight_down",
    up: ".input_mix_weight_up",
    inject: ".block_inject_weight",
};

/// The HF checkpoint's names. The routed experts' layout is the store's to decide
/// ([`expert_moe_names`]).
const HF_NAMES: Names = Names {
    embed: "model.language_model.embed_tokens.weight",
    head: "lm_head",
    final_gate: HF_GATE,
    layers: HF_LAYERS,
    attn_gate: HyperNames {
        norm: ".attn_hyper_connection.hc_norm.weight",
        down: ".attn_hyper_connection.input_mix_weight_down",
        up: ".attn_hyper_connection.input_mix_weight_up",
        inject: ".attn_hyper_connection.block_inject_weight",
    },
    mlp_gate: HyperNames {
        norm: ".mlp_hyper_connection.hc_norm.weight",
        down: ".mlp_hyper_connection.input_mix_weight_down",
        up: ".mlp_hyper_connection.input_mix_weight_up",
        inject: ".mlp_hyper_connection.block_inject_weight",
    },
    attention: AttentionNames {
        q: ".self_attn.q_proj",
        k: ".self_attn.k_proj",
        v: ".self_attn.v_proj",
        o: ".self_attn.o_proj",
        q_norm: ".self_attn.q_norm.weight",
        k_norm: ".self_attn.k_norm.weight",
        indexer: IndexerNames {
            proj: IndexerProj::Fused(".self_attn.indexer.index_qk_proj"),
            q_norm: ".self_attn.indexer.q_layernorm.weight",
            k_norm: ".self_attn.indexer.k_layernorm.weight",
        },
    },
    gdn: GdnNames {
        qkv: ".linear_attn.in_proj_qkv",
        z: ".linear_attn.in_proj_z",
        beta: ".linear_attn.in_proj_b",
        alpha: ".linear_attn.in_proj_a",
        out: ".linear_attn.out_proj",
        dt_bias: ".linear_attn.dt_bias",
        conv: ".linear_attn.conv1d.weight",
        a: ".linear_attn.A_log",
        norm: ".linear_attn.norm.weight",
    },
    ple: PleNames {
        table: PleTable::PerLayer(".ple.ple_embedding.ngram_embedding"),
        key: ".ple.key_proj",
        value: ".ple.value_proj",
        key_norm: ".ple.norm_key.weight",
        query_norm: ".ple.norm_query.weight",
        conv_norm: ".ple.norm_conv.weight",
        conv: ".ple.conv1d.weight",
    },
};

/// The GGUF's names (`llama.cpp`'s `qwen4exp`), as the real UD-Q2_K_XL set lists them: no
/// `output_norm` (the final gate carries it), the PLE table without a block prefix.
const GGUF_NAMES: Names = Names {
    embed: "token_embd.weight",
    head: "output",
    final_gate: HyperNames {
        norm: "output_hc_norm.weight",
        down: "output_hc_down",
        up: "output_hc_up",
        inject: "",
    },
    layers: GGUF_LAYERS,
    attn_gate: HyperNames {
        norm: ".hc_attn_norm.weight",
        down: ".hc_attn_down",
        up: ".hc_attn_up",
        inject: ".hc_attn_inject",
    },
    mlp_gate: HyperNames {
        norm: ".hc_ffn_norm.weight",
        down: ".hc_ffn_down",
        up: ".hc_ffn_up",
        inject: ".hc_ffn_inject",
    },
    attention: AttentionNames {
        q: ".attn_q",
        k: ".attn_k",
        v: ".attn_v",
        o: ".attn_output",
        q_norm: ".attn_q_norm.weight",
        k_norm: ".attn_k_norm.weight",
        indexer: IndexerNames {
            proj: IndexerProj::Split {
                q: ".indexer.q_proj",
                k: ".indexer.k_proj",
            },
            q_norm: ".indexer.q_norm.weight",
            k_norm: ".indexer.k_norm.weight",
        },
    },
    gdn: GdnNames {
        qkv: ".attn_qkv",
        z: ".attn_gate",
        beta: ".ssm_beta",
        alpha: ".ssm_alpha",
        out: ".ssm_out",
        dt_bias: ".ssm_dt.bias",
        conv: ".ssm_conv1d.weight",
        a: ".ssm_a",
        norm: ".ssm_norm.weight",
    },
    ple: PleNames {
        table: PleTable::Model(GGUF_PLE_TABLE),
        key: ".ple_key",
        value: ".ple_value",
        key_norm: ".ple_norm_key.weight",
        query_norm: ".ple_norm_query.weight",
        conv_norm: ".ple_norm_conv.weight",
        conv: ".ple_conv1d.weight",
    },
};

/// How a store holds the routed experts, and so the gate/up form: an HF store's layout decides, as a
/// config cannot say. A tensor named per expert (`mlp.experts.{e}.`) is the split layout; a store with
/// none is read as fused, so a store missing its experts is told to lack the fused tensor. A GGUF's
/// experts are always its separate stacked gate and up tensors.
pub(super) fn expert_layout(
    layout: Layout,
    store: &WeightStore,
) -> (&'static MoeNames, GateUpForm) {
    const PER_EXPERT: &str = ".mlp.experts.";
    if layout == Layout::Gguf {
        return (&MOE_GGUF, GateUpForm::Split);
    }
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
    let names = match p.layout {
        Layout::Hf => &HF_NAMES,
        Layout::Gguf => &GGUF_NAMES,
    };
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
    put(&mut map, embed, names.embed.to_string())?;
    let head = WeightId::model(WeightRole::Head);
    let head_key = linear(names.head);
    if store.contains(&head_key) {
        put(&mut map, head, head_key)?;
    } else if p.head_required {
        return Err(missing(WeightMapError::Missing {
            id: head,
            key: WeightKey::from(head_key),
        }));
    } else {
        put(&mut map, head, names.embed.to_string())?;
    }
    // One Hyper-Connections gate: the grouped norm scale, the two low-rank projections and, unless
    // it is the model-level mixer, the per-stream injection weights. `at` joins a gate's name to
    // the tensor's stem.
    let hyper = |map: &mut WeightMapBuilder<'_>,
                 id: &dyn Fn(HyperRole) -> WeightId,
                 stem: &str,
                 gate: &HyperNames,
                 inject: bool|
     -> Result<(), ModelError> {
        put(map, id(HyperRole::Norm), format!("{stem}{}", gate.norm))?;
        for (role, suffix) in [(HyperRole::MixDown, gate.down), (HyperRole::MixUp, gate.up)] {
            put(map, id(role), linear(&format!("{stem}{suffix}")))?;
        }
        if inject {
            put(
                map,
                id(HyperRole::Inject),
                linear(&format!("{stem}{}", gate.inject)),
            )?;
        }
        Ok(())
    };
    hyper(
        &mut map,
        &|role| WeightId::model(WeightRole::Hyper(HyperSite::Final, role)),
        match p.layout {
            Layout::Hf => "model.language_model.hyper_connection_mixer",
            Layout::Gguf => "",
        },
        &names.final_gate,
        false,
    )?;

    for (l, kind) in p.kinds.iter().enumerate() {
        let layer = |suffix: &str| format!("{}{l}{suffix}", names.layers);
        let id = |role| WeightId::layer(l, role);
        for (site, gate) in [
            (HyperSite::Attn, &names.attn_gate),
            (HyperSite::Mlp, &names.mlp_gate),
        ] {
            hyper(
                &mut map,
                &|role| id(WeightRole::Hyper(site, role)),
                &layer(""),
                gate,
                true,
            )?;
        }
        if p.ple[l].is_some() {
            let ple = &names.ple;
            match ple.table {
                PleTable::PerLayer(stem) => {
                    // The official release stores the table as numbered E4M3 shards under one
                    // scale; a smaller checkpoint stores it as one dense weight.
                    let table = layer(stem);
                    let shard = |ordinal: u32| format!("{table}.shard_{ordinal}.weight");
                    if store.contains(&shard(0)) {
                        for ordinal in (0..).take_while(|&ordinal| store.contains(&shard(ordinal)))
                        {
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
                }
                PleTable::Model(name) => put(
                    &mut map,
                    id(WeightRole::Ple(PleRole::Table)),
                    name.to_string(),
                )?,
            }
            for (role, key) in [
                (PleRole::Key, linear(&layer(ple.key))),
                (PleRole::Value, linear(&layer(ple.value))),
                (PleRole::KeyNorm, layer(ple.key_norm)),
                (PleRole::QueryNorm, layer(ple.query_norm)),
                (PleRole::ConvNorm, layer(ple.conv_norm)),
                (PleRole::Conv, layer(ple.conv)),
            ] {
                put(&mut map, id(WeightRole::Ple(role)), key)?;
            }
        }
        match kind {
            LayerKind::FullAttention => {
                let a = &names.attention;
                for (role, stem) in [
                    (AttnRole::Q, a.q),
                    (AttnRole::K, a.k),
                    (AttnRole::V, a.v),
                    (AttnRole::O, a.o),
                ] {
                    put(&mut map, id(WeightRole::Attn(role)), linear(&layer(stem)))?;
                }
                for (role, key) in [(AttnRole::QNorm, a.q_norm), (AttnRole::KNorm, a.k_norm)] {
                    put(&mut map, id(WeightRole::Attn(role)), layer(key))?;
                }
                match a.indexer.proj {
                    // The query and key rows are one fused projection, query rows first.
                    IndexerProj::Fused(stem) => {
                        let fused = linear(&layer(stem));
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
                    }
                    IndexerProj::Split { q, k } => {
                        for (role, stem) in [(QsaRole::Q, q), (QsaRole::K, k)] {
                            put(&mut map, id(WeightRole::Qsa(role)), linear(&layer(stem)))?;
                        }
                    }
                }
                for (role, key) in [
                    (QsaRole::QNorm, a.indexer.q_norm),
                    (QsaRole::KNorm, a.indexer.k_norm),
                ] {
                    put(&mut map, id(WeightRole::Qsa(role)), layer(key))?;
                }
            }
            LayerKind::LinearAttention => {
                let g = &names.gdn;
                let gdn = |role| id(WeightRole::Gdn(role));
                for (role, stem) in [
                    (GdnRole::Qkv, g.qkv),
                    (GdnRole::Z, g.z),
                    (GdnRole::Beta, g.beta),
                    (GdnRole::Alpha, g.alpha),
                    (GdnRole::Out, g.out),
                ] {
                    put(&mut map, gdn(role), linear(&layer(stem)))?;
                }
                for (role, key) in [
                    (GdnRole::DtBias, g.dt_bias),
                    (GdnRole::Conv, g.conv),
                    (GdnRole::A, g.a),
                    (GdnRole::GateNorm, g.norm),
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
