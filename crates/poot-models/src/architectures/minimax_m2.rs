//! MiniMax-M2 behind [`Model`]: a standard pre-norm block with GQA attention, RMSNorm over the
//! whole Q and K projections before a partial RoPE (`QkNorm::Projection`), whose feed-forward
//! routes every layer through a mixture of experts. The router is
//! [`RouterKind::SigmoidBiasedTopK`]: sigmoid scores, the `top_k` of score plus the
//! `e_score_correction_bias` (selection only), and the weights of the selected experts taken from
//! the unbiased scores and divided by their sum. There is no group limit, shared expert or routed
//! scaling factor.
//!
//! Every dimension comes from the checkpoint's config: a fine-tune with fewer layers or experts is
//! the same family. The checkpoint's quantization (FP8 blocks) is the loader's business; the
//! family names its tensors and reads them as stored.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, Graph, ValidationOutputs};
use poot_quant::weights::{DenseWeight, ExpertProj, WeightEntry, WeightMap, WeightStore};

use crate::chat::ChatFormat;
use crate::components::attention::{AttentionParams, AttentionWeights, QkNorm, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::linear::WeightError;
use crate::components::moe::{self, MoeParams, MoeWeights};
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
    ExpertNames, HF_BASE, HF_QK_NORM, MoeLayers, MoeNames, family_weights_routed, hf_base_shapes,
};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

pub const FAMILY: FamilyKey = FamilyKey::new("minimax_m2");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, "minimax_m2")],
    build,
    fixture,
    variants: &[("finetune", finetune_fixture)],
};

/// MiniMax-M2's config facts: the released config states `rms_norm_eps` and `rope_theta`, so these
/// defaults only serve a config that omits them (HF's `MiniMaxM2Config`). The family is HF-only;
/// the GGUF keys are never read.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("minimax-m2"),
    hf_theta: 5_000_000.0,
    gguf_theta: 5_000_000.0,
    eps: 1e-6,
    eos: None,
};

const EXPERTS: Field = Field::hf("num_local_experts");
const TOP_K: Field = Field::hf("num_experts_per_tok");
const ROTARY: Field = Field::hf("rotary_dim");

const HF_MOE: MoeNames = MoeNames {
    prefix: "model.layers.",
    router: ".block_sparse_moe.gate",
    router_bias: Some(".block_sparse_moe.e_score_correction_bias"),
    experts: &[
        ExpertNames {
            proj: ExpertProj::Gate,
            stack: None,
            each: Some((".block_sparse_moe.experts.", ".w1.weight")),
            bias: None,
        },
        ExpertNames {
            proj: ExpertProj::Up,
            stack: None,
            each: Some((".block_sparse_moe.experts.", ".w3.weight")),
            bias: None,
        },
        ExpertNames {
            proj: ExpertProj::Down,
            stack: None,
            each: Some((".block_sparse_moe.experts.", ".w2.weight")),
            bias: None,
        },
    ],

    shared: None,
};

/// MiniMax-M2's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM2Config {
    vocab: usize,
    width: usize,
    layers: usize,
    max_positions: usize,
    attention: AttentionParams,
    layer: LayerParams,
    stack: StackParams,
    moe: MoeParams,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
}

impl MiniMaxM2Config {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        // The family is HF-only: a GGUF never names it, so any other source is refused.
        let RawConfig::HfJson { config, .. } = raw else {
            return Err(ModelError::Config {
                family: FAMILY,
                field: "general.architecture",
                reason: ConfigReason::Unsupported,
            });
        };
        let fields = Fields::new(raw, FAMILY);
        let param = |e: crate::components::standard::ParamError| e.for_family(FAMILY);
        // `intermediate_size` is the per-expert width: every layer routes.
        let dims = DenseDims::read(&fields, &SPEC)?;
        // The router and the block this family traces: refuse a config that asks for another.
        let scoring = config.get("scoring_func").filter(|v| !v.is_null());
        if scoring.is_some_and(|v| v.as_str() != Some("sigmoid")) {
            return Err(fields.error("scoring_func", ConfigReason::Unsupported));
        }
        for field in ["use_routing_bias", "use_qk_norm"] {
            if !fields.flag(Field::hf(field), true)? {
                return Err(fields.error(field, ConfigReason::Unsupported));
            }
        }
        if fields
            .opt_u64(Field::hf("shared_intermediate_size"))?
            .is_some_and(|n| n != 0)
        {
            return Err(fields.error("shared_intermediate_size", ConfigReason::Unsupported));
        }
        let norm = dims.norm(&fields, &SPEC)?;
        let rotary = fields.count(ROTARY, None)?;
        let rope = dims.rope(&fields, &SPEC, rotary)?;
        let attention = AttentionParams::new(dims.heads, dims.kv_heads, dims.head_dim, false, rope)
            .map_err(param)?
            .with_qk_norm(QkNorm::Projection(norm));
        let moe = MoeParams::new(
            FAMILY,
            dims.width,
            dims.inter,
            fields.count(EXPERTS, None)?,
            fields.count(TOP_K, None)?,
            RouterKind::SigmoidBiasedTopK(SigmoidTopK::new()),
        )?
        .with_router_bias(RouterBias::SelectionOnly)
        .map_err(param)?;
        Ok(Self {
            attention,
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            vocab: dims.vocab,
            width: dims.width,
            layers: dims.layers,
            max_positions: dims.max_positions,
            moe,
            head_required: dims.head_required,
            eos: dims.eos,
            bos: dims.bos,
        })
    }
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    attn: AttentionWeights,
    moe: MoeWeights,
}

#[derive(Debug)]
pub struct MiniMaxM2 {
    params: MiniMaxM2Config,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = MiniMaxM2Config::from_raw(raw)?;
    let sparse = vec![true; p.layers];
    let weights = family_weights_routed(
        FAMILY,
        store,
        &[HF_BASE, HF_QK_NORM],
        p.layers,
        p.head_required,
        MoeLayers {
            names: &HF_MOE,
            sparse: &sparse,
            experts: p.moe.experts(),
        },
    )?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers = (0..p.layers)
        .map(|l| {
            Ok(Layer {
                norms: LayerNorms::new(&weights, l, p.width)?,
                attn: AttentionWeights::new(&weights, l, p.width, &p.attention)?,
                moe: MoeWeights::new(&weights, l, &p.moe)?,
            })
        })
        .collect::<Result<Vec<_>, WeightError>>()
        .map_err(weight_error)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        prompt_bos: None,
        output: ModelOutput::Logits { vocab: p.vocab },
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
    };
    Ok(Box::new(MiniMaxM2 {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for MiniMaxM2 {
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
        check_step(FAMILY, phase, shape, p.max_positions)?;
        moe::check_step(FAMILY, shape, &p.moe)?;
        let b = Builder::new();
        let step = Step::new(&b, shape);
        let logits = standard_stack(&b, &step, &self.stack, &p.stack, p.layers, |b, l, x| {
            let layer = &self.layers[l];
            standard_layer(
                b,
                x,
                &layer.norms,
                &p.layer,
                |b, h| attention(b, &step, l, h, &layer.attn, &p.attention),
                |b, BlockInput { block: h, .. }| moe::moe(b, &step, l, h, None, &layer.moe, &p.moe),
            )
        });
        Ok(step.finish(b, logits))
    }
}

/// The fixture's config: a tiny untied MiniMax-M2 with grouped-query heads (4 over 2), a partial
/// RoPE (8 of 16 dims) and 8 experts of which 2 route, whose projection widths are multiples of 32
/// (so a Q8_0 copy of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "minimax_m2",
        "vocab_size": 48,
        "hidden_size": 64,
        "intermediate_size": 32,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rotary_dim": 8,
        "scoring_func": "sigmoid",
        "use_routing_bias": true,
        "use_qk_norm": true,
        "shared_intermediate_size": 0,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_theta": 5000000.0,
        "tie_word_embeddings": false,
        "eos_token_id": 47,
    })
}

fn fixture() -> Fixture {
    fixture_of(fixture_config())
}

/// A fine-tune of the fixture's architecture with other counts than any pinned release: one layer,
/// four experts of which three route, four key-value heads and a RoPE as wide as the head. It
/// ships no manifest and is the same family.
fn finetune_fixture() -> Fixture {
    let mut config = fixture_config();
    config["num_hidden_layers"] = serde_json::json!(1);
    config["num_local_experts"] = serde_json::json!(4);
    config["num_experts_per_tok"] = serde_json::json!(3);
    config["num_key_value_heads"] = serde_json::json!(4);
    config["rotary_dim"] = serde_json::json!(16);
    fixture_of(config)
}

/// A BF16 checkpoint of `config`'s dimensions, one tensor per name the family reads.
fn fixture_of(config: serde_json::Value) -> Fixture {
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let (h, experts, inter) = (
        n("hidden_size"),
        n("num_local_experts"),
        n("intermediate_size"),
    );
    let d = n("head_dim");
    let (q, kv) = (n("num_attention_heads") * d, n("num_key_value_heads") * d);
    let mut shapes: Vec<_> = hf_base_shapes(&config, false)
        .into_iter()
        .filter(|(name, _)| !name.contains(".mlp."))
        .collect();
    for l in 0..n("num_hidden_layers") {
        let k = |s: &str| format!("model.layers.{l}.{s}");
        shapes.push((k("self_attn.q_norm.weight"), vec![q]));
        shapes.push((k("self_attn.k_norm.weight"), vec![kv]));
        shapes.push((k("block_sparse_moe.gate.weight"), vec![experts, h]));
        shapes.push((k("block_sparse_moe.e_score_correction_bias"), vec![experts]));
        for e in 0..experts {
            for (proj, shape) in [("w1", [inter, h]), ("w3", [inter, h]), ("w2", [h, inter])] {
                shapes.push((
                    k(&format!("block_sparse_moe.experts.{e}.{proj}.weight")),
                    shape.to_vec(),
                ));
            }
        }
    }
    let mut fixture = Fixture::bf16(config, shapes);
    fixture.store = with_scaled_head(&fixture.store);
    fixture
}

/// The head's power-of-two scale: exact in BF16, and it spreads the logits so a greedy decode's
/// top-two margins stay far above the f32 rounding difference between a device and a reference.
const HEAD_SCALE: u16 = 3;

/// `store` with the `lm_head.weight` entries' BF16 exponent raised by [`HEAD_SCALE`] (a multiply by
/// `2^HEAD_SCALE`, exact for every normal value).
fn with_scaled_head(store: &WeightStore) -> WeightStore {
    let mut out = WeightStore::builder();
    for (key, entry) in store.iter() {
        let entry = match entry {
            WeightEntry::Dense(dense) if key.as_str() == "lm_head.weight" => {
                let bytes: Vec<u8> = dense
                    .bytes()
                    .as_slice()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|&b| {
                        let bits = u16::from_le_bytes(b);
                        let scaled = (f32::from_bits(u32::from(bits) << 16)
                            * f32::from(1u16 << HEAD_SCALE))
                        .to_bits()
                            >> 16;
                        (scaled as u16).to_le_bytes()
                    })
                    .collect();
                WeightEntry::Dense(
                    DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), bytes.into())
                        .expect("the scaled head keeps its shape"),
                )
            }
            other => other.clone(),
        };
        out.insert(key.clone(), entry).expect("keys are unique");
    }
    out.build()
}

#[cfg(test)]
mod tests;
