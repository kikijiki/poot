//! The wrapper every standard-stack family shares.
//!
//! A decoder of the standard shape is `standard_stack` over `standard_layer` with an attention and a
//! feed-forward block per layer. What differs between qwen3, olmo2, phi3, mpt, bloom, granite, gemma2,
//! gpt-oss and the rest is data: the config keys that name the family, how its config reads into
//! [`StandardParams`], and how its checkpoint's tensor names map onto weight roles. A family is a
//! [`StandardFamily`] saying exactly that; the params, the layer weights, the [`Model`] impl and the
//! build are written once, here ([`StandardModel`], [`build_standard`]). A family whose layers are not
//! this shape (an MLA, a state-space or a hybrid layer, an encoder) keeps its own model.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_graph_ir::{Builder, Graph};
use poot_quant::weights::{WeightMap, WeightStore};
use poot_text::{ChatFormat, ToolCallCodec};

use crate::components::attention::{AttentionParams, AttentionWeights, QkNorm, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields};
use crate::components::embed::EMBEDS;
use crate::components::ffn::{
    GatedFfnWeights, Gelu, PlainMlpWeights, gated_ffn, geglu_ffn, plain_mlp,
};
use crate::components::linear::WeightError;
use crate::components::moe::{self, MoeParams, MoeWeights};
use crate::components::standard::{
    BlockInput, LayerNorms, LayerParams, NormPlacement, StackParams, StackWeights, Step,
    decoder_program, embeddable_decoder_schema, standard_layer, standard_stack,
};
use crate::io::{ModelSchema, ProgramId};
use crate::model::{ConfigReason, FamilyKey, Model, ModelConfig, ModelError, StepSpec, TraceError};
use crate::names::{
    FusedDims, MoeLayers, MoeNames, NameRow, family_weights, family_weights_fused,
    family_weights_routed,
};
use crate::registry::{ConfigSource, FamilyEntry, ModelParts, RawConfig};

/// What a standard-stack family declares: the keys that name it, its config and its tensor names.
pub trait StandardFamily {
    /// The family's key, which every error it reports carries.
    const FAMILY: FamilyKey;
    /// The config keys that name the family, each with the config source it is read from.
    const KEYS: &'static [(ConfigSource, &'static str)];

    /// Reads and checks the config once, so tracing cannot fail on it.
    fn configure(raw: &RawConfig<'_>) -> Result<StandardParams, ModelError>;

    /// Maps the checkpoint's tensor names onto the model's weight roles. `params` is the family's
    /// own, which a checkpoint's tensors may still refine (llama's per-frequency RoPE factors).
    fn weights(
        raw: &RawConfig<'_>,
        store: &WeightStore,
        params: &mut StandardParams,
    ) -> Result<WeightMap, ModelError>;
}

/// A dense decoder family: the keys that name it, the defaults its config omits and its one attention
/// option. Everything else is the standard decoder, so such a family needs no other code.
pub(crate) trait DenseFamily {
    /// The family's key, which every error it reports carries.
    const FAMILY: FamilyKey;
    /// The config keys that name the family.
    const KEYS: &'static [(ConfigSource, &'static str)];
    /// The config facts the family fixes: its GGUF keys and the defaults a checkpoint may omit.
    const SPEC: DenseSpec;
    /// The chat template family of its tokenizer.
    const CHAT: ChatFormat;
    /// How the family spells a function call in its output; `None` for a family with no tool-call
    /// markup.
    fn tool_codec() -> Option<Arc<dyn ToolCallCodec>> {
        None
    }
    /// Config flags the body cannot serve: a checkpoint that sets one is refused, naming it.
    const UNSUPPORTED: &'static [&'static str] = &["attention_bias"];
    /// The family's Q/K norm, when it has one.
    const QK_NORM: Option<QkNormScope> = None;
    /// Where the layer's two norms sit around its blocks.
    const PLACEMENT: NormPlacement = NormPlacement::Pre;
    /// The name rows an HF checkpoint is read through.
    const HF_TABLES: &'static [&'static [NameRow]] = &[crate::names::HF_BASE];
    /// The name rows a GGUF checkpoint is read through.
    const GGUF_TABLES: &'static [&'static [NameRow]] = &[crate::names::gguf::GGUF_BASE];
}

/// Where a Q/K RMSNorm normalizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QkNormScope {
    /// Over each head's dims.
    PerHead,
    /// Over the whole projection.
    Projection,
}

impl<D: DenseFamily> StandardFamily for D {
    const FAMILY: FamilyKey = D::FAMILY;
    const KEYS: &'static [(ConfigSource, &'static str)] = D::KEYS;

    fn configure(raw: &RawConfig<'_>) -> Result<StandardParams, ModelError> {
        let fields = Fields::new(raw, D::FAMILY);
        let dims = DenseDims::read(&fields, &D::SPEC)?;
        for unsupported in D::UNSUPPORTED {
            if fields.flag(Field::hf(unsupported), false)? {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        let norm = dims.norm(&fields, &D::SPEC)?;
        let attention = dims.attention(&fields, &D::SPEC, false)?;
        let attention = match D::QK_NORM {
            None => attention,
            Some(QkNormScope::PerHead) => attention.with_qk_norm(QkNorm::PerHead(norm)),
            Some(QkNormScope::Projection) => attention.with_qk_norm(QkNorm::Projection(norm)),
        };
        let mut params = StandardParams::dense(&dims, norm, attention, D::CHAT, D::tool_codec());
        params.layer = params.layer.with_placement(D::PLACEMENT);
        Ok(params)
    }

    fn weights(
        raw: &RawConfig<'_>,
        store: &WeightStore,
        params: &mut StandardParams,
    ) -> Result<WeightMap, ModelError> {
        let tables = match raw.source() {
            ConfigSource::HfModelType => D::HF_TABLES,
            ConfigSource::GgufArchitecture => D::GGUF_TABLES,
        };
        family_weights(
            D::FAMILY,
            store,
            tables,
            params.layers,
            params.head_required,
        )
    }
}

/// The weight map of a family whose layers route as `params.mlp` says: `tables` and `names` are the
/// checkpoint's dense rows and its experts' names. A family with no routed layer reads the dense
/// rows alone.
pub(crate) fn weights_with_experts(
    family: FamilyKey,
    store: &WeightStore,
    tables: &[&[NameRow]],
    names: &MoeNames,
    params: &StandardParams,
) -> Result<WeightMap, ModelError> {
    // The rows per head, for the tables whose Q/K rows are stored paired per head.
    let fused = FusedDims {
        head_dim: params
            .attention
            .first()
            .map_or(0, AttentionParams::head_dim),
        ..FusedDims::NONE
    };
    match params.routed() {
        None => family_weights_fused(
            family,
            store,
            tables,
            params.layers,
            params.head_required,
            fused,
        ),
        Some(moe) => family_weights_routed(
            family,
            store,
            tables,
            params.layers,
            params.head_required,
            fused,
            MoeLayers {
                names,
                sparse: &params
                    .mlp
                    .iter()
                    .map(|mlp| matches!(mlp, Mlp::Routed(_)))
                    .collect::<Vec<_>>(),
                experts: moe.experts(),
            },
        ),
    }
}

/// The Q/K rows of a llama-architecture GGUF are paired over a whole head ([`crate::names::Source::PairedHeads`]),
/// which is the half-split layout of the rotation only when the rotation spans the whole head: refuse a layer
/// that rotates fewer dims.
pub(crate) fn require_whole_head_rotation(
    family: FamilyKey,
    params: &StandardParams,
) -> Result<(), ModelError> {
    for attention in &params.attention {
        match attention.rotary() {
            Some(rotary) if rotary != attention.head_dim() => {
                return Err(ModelError::Config {
                    family,
                    field: "rope.dimension_count",
                    reason: ConfigReason::Mismatch {
                        expected: attention.head_dim(),
                        found: rotary,
                    },
                });
            }
            Some(_) | None => {}
        }
    }
    Ok(())
}

/// The registry entry of family `F`.
pub const fn entry<F: StandardFamily>() -> FamilyEntry {
    FamilyEntry {
        family: F::FAMILY,
        keys: F::KEYS,
        build: build_standard::<F>,
    }
}

/// How a layer's two norms are weighted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerNormKind {
    /// A scale each.
    Scale,
    /// A scale and a bias each (BLOOM, GPT-NeoX).
    Biased,
    /// Four scales: before and after each block (Gemma 2 and 3).
    Sandwich,
}

/// One layer's feed-forward block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mlp {
    /// `down(silu(gate) * up)`.
    Gated {
        /// The hidden width.
        inter: usize,
    },
    /// `down(gelu_tanh(gate) * up)`.
    Geglu {
        /// The hidden width.
        inter: usize,
    },
    /// `down(gelu(up))`, the projections biased or not.
    Plain {
        /// The hidden width.
        inter: usize,
        /// The activation.
        act: Gelu,
        /// Whether the up and down projections carry biases.
        biased: bool,
    },
    /// A top-k mixture of experts.
    Routed(MoeParams),
}

/// A family's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug)]
pub struct StandardParams {
    /// Vocabulary size (embedding rows).
    pub vocab: usize,
    /// Hidden width.
    pub width: usize,
    /// Layer count.
    pub layers: usize,
    /// The context length in positions.
    pub max_positions: usize,
    /// One per layer: a family whose layers attend alike repeats one value.
    pub attention: Vec<AttentionParams>,
    /// One per layer.
    pub mlp: Vec<Mlp>,
    /// How each layer's norms and residual joins are laid out.
    pub layer: LayerParams,
    /// The embedding scale, final norm and logit options around the layers.
    pub stack: StackParams,
    /// How each layer's two norms are weighted.
    pub norms: LayerNormKind,
    /// The final norm carries a bias.
    pub final_norm_bias: bool,
    /// A LayerNorm over the embedded tokens (BLOOM).
    pub embed_norm: bool,
    /// The checkpoint carries a head tensor (it does not tie the embedding).
    pub head_required: bool,
    /// The end-of-sequence token ids.
    pub eos: BTreeSet<u32>,
    /// The beginning-of-sequence token id, when the checkpoint names one.
    pub bos: Option<u32>,
    /// A token the prompt must start with, when the family is trained with one.
    pub prompt_bos: Option<u32>,
    /// The chat template family of the tokenizer.
    pub chat: ChatFormat,
    /// How the family spells a function call in its output; `None` for a family with no tool-call
    /// markup.
    pub tool_codec: Option<Arc<dyn ToolCallCodec>>,
    /// The model takes embedding rows in place of tokens and offers an `encode` program.
    pub embeddable: bool,
}

impl StandardParams {
    /// The params of a plain dense decoder read by [`DenseDims`]: the same `attention` and gated
    /// feed-forward at every layer, scale-only pre-norms.
    pub fn dense(
        dims: &DenseDims,
        norm: crate::components::norm::NormParams,
        attention: AttentionParams,
        chat: ChatFormat,
        tool_codec: Option<Arc<dyn ToolCallCodec>>,
    ) -> Self {
        Self {
            vocab: dims.vocab,
            width: dims.width,
            layers: dims.layers,
            max_positions: dims.max_positions,
            attention: vec![attention; dims.layers],
            mlp: vec![Mlp::Gated { inter: dims.inter }; dims.layers],
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            norms: LayerNormKind::Scale,
            final_norm_bias: false,
            embed_norm: false,
            head_required: dims.head_required,
            eos: dims.eos.clone(),
            bos: dims.bos,
            prompt_bos: None,
            chat,
            tool_codec,
            embeddable: false,
        }
    }

    /// The width of the first layer's dense feed-forward; `0` when that layer routes.
    pub fn dense_inter(&self) -> usize {
        match self.mlp[0] {
            Mlp::Gated { inter } | Mlp::Geglu { inter } | Mlp::Plain { inter, .. } => inter,
            Mlp::Routed(_) => 0,
        }
    }

    /// The mixture parameters, when any layer routes.
    pub fn routed(&self) -> Option<&MoeParams> {
        self.mlp.iter().find_map(|mlp| match mlp {
            Mlp::Routed(params) => Some(params),
            _ => None,
        })
    }
}

/// A layer's feed-forward weights.
#[derive(Debug)]
enum MlpWeights {
    Gated(GatedFfnWeights),
    Geglu(GatedFfnWeights),
    Plain(PlainMlpWeights, Gelu),
    Routed(Box<MoeWeights>, MoeParams),
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    attn: AttentionWeights,
    mlp: MlpWeights,
}

/// A built standard-stack model: its params, the name-resolved weights, the stack-level tensors and
/// each layer's weights. Owns its weights; holds no per-request state.
#[derive(Debug)]
pub(crate) struct StandardModel {
    params: StandardParams,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
    schema: ModelSchema,
}

/// Builds family `F`'s model over `store`.
pub fn build_standard<F: StandardFamily>(
    raw: &RawConfig<'_>,
    store: &WeightStore,
) -> Result<ModelParts, ModelError> {
    let family = F::FAMILY;
    let mut p = F::configure(raw)?;
    let weights = F::weights(raw, store, &mut p)?;
    let weight_error = |e: WeightError| e.for_family(family);
    let mut stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    if p.final_norm_bias {
        stack = stack
            .with_final_norm_bias(&weights, p.width)
            .map_err(weight_error)?;
    }
    if p.embed_norm {
        stack = stack
            .with_embed_norm(&weights, p.width)
            .map_err(weight_error)?;
    }
    let layers = (0..p.layers)
        .map(|l| {
            let norms = match p.norms {
                LayerNormKind::Scale => LayerNorms::new(&weights, l, p.width)?,
                LayerNormKind::Biased => LayerNorms::biased(&weights, l, p.width)?,
                LayerNormKind::Sandwich => LayerNorms::sandwich(&weights, l, p.width)?,
            };
            let mlp = match p.mlp[l] {
                Mlp::Gated { inter } => {
                    MlpWeights::Gated(GatedFfnWeights::new(&weights, l, p.width, inter)?)
                }
                Mlp::Geglu { inter } => {
                    MlpWeights::Geglu(GatedFfnWeights::new(&weights, l, p.width, inter)?)
                }
                Mlp::Plain { inter, act, biased } => MlpWeights::Plain(
                    PlainMlpWeights::new(&weights, l, p.width, inter, biased)?,
                    act,
                ),
                Mlp::Routed(moe) => {
                    MlpWeights::Routed(Box::new(MoeWeights::new(&weights, l, &moe)?), moe)
                }
            };
            Ok(Layer {
                norms,
                attn: AttentionWeights::new(&weights, l, p.width, &p.attention[l])?,
                mlp,
            })
        })
        .collect::<Result<Vec<_>, WeightError>>()
        .map_err(weight_error)?;
    let schema = if p.embeddable {
        embeddable_decoder_schema(p.vocab, p.width, p.attention[0].is_sectioned())
    } else {
        ModelSchema::builder().step(decoder_program(p.vocab))
    }
    .build_for(family)?;
    let config = ModelConfig {
        family,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        prompt_bos: p.prompt_bos,
        prefill_granule: NonZeroUsize::MIN,
        chat: p.chat,
        tool_codec: p.tool_codec.clone(),
    };
    Ok(ModelParts::new(Box::new(StandardModel {
        params: p,
        weights,
        stack,
        layers,
        config,
        schema,
    })))
}

impl Model for StandardModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, spec: &StepSpec, b: Builder) -> Result<Graph, TraceError> {
        let p = &self.params;
        let shape = spec.shape;
        let step = Step::new(&b, shape)?;
        let step = if p.embeddable && spec.program == ProgramId::ENCODE {
            step.returning_hidden()
        } else {
            step
        };
        let step = if p.embeddable && spec.carries(EMBEDS) {
            step.with_embeds(&b, p.width)?
        } else {
            step
        };
        let step = if p.attention[0].is_sectioned() {
            step.with_mrope(&b)?
        } else {
            step
        };
        let logits = standard_stack(&b, &step, &self.stack, &p.stack, p.layers, |b, l, x| {
            let layer = &self.layers[l];
            standard_layer(
                b,
                x,
                &layer.norms,
                &p.layer,
                |b, h| attention(b, &step, l, h, &layer.attn, &p.attention[l]),
                |b, BlockInput { block: h, .. }| match &layer.mlp {
                    MlpWeights::Gated(w) => gated_ffn(b, h, w),
                    MlpWeights::Geglu(w) => geglu_ffn(b, h, w),
                    MlpWeights::Plain(w, act) => plain_mlp(b, h, w, *act),
                    MlpWeights::Routed(w, params) => moe::moe(b, &step, l, h, None, w, params),
                },
            )
        })?;
        Ok(step.finish(b, logits)?)
    }
}
