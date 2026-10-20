//! The second fixture family of the add-a-model standing test (ADR-0104 rule 3): "neox_parallel", a
//! GPT-NeoX / Pythia style decoder defined outside every production crate, like [`crate::family`] but
//! with the two properties that made GPT-NeoX the probe of the add-a-model contract:
//!
//! - **Parallel residual.** Both blocks of a layer read the layer's input and their outputs join the
//!   residual together (`x + attn(norm1(x)) + mlp(norm2(x))`), the config's `use_parallel_residual`.
//!   It is one option of the shared layer ([`NormPlacement::Parallel`]).
//! - **Per-head interleaved fused QKV.** One `query_key_value` tensor whose rows interleave per head
//!   (`q_h, k_h, v_h` for each head), read as one weight and split in the graph
//!   ([`AttentionParams::with_fused_qkv`]), unlike [`crate::family`]'s contiguous `q | k | v`.
//!
//! The family is its keys, its config and its tensor names; the stack, the layer and the model are the
//! shared `StandardFamily` wrapper. If adding it needed a production edit, this file could not compile.

use std::sync::Arc;

use poot_graph_ir::rope_table::RopeFlavor;
use poot_models::components::attention::AttentionParams;
use poot_models::components::dims::{Field, Fields};
use poot_models::components::ffn::Gelu;
use poot_models::components::norm::{NormKind, NormParams};
use poot_models::components::rope::RopeParams;
use poot_models::components::standard::{LayerParams, NormPlacement, StackParams};
use poot_models::components::standard_family::{
    LayerNormKind, Mlp, StandardFamily, StandardParams, entry,
};
use poot_models::model::{ConfigReason, FamilyKey, ModelError};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightKey, WeightStore};
use poot_tensor::DType;
use poot_text::ChatFormat;
use poot_weights::{AttnRole, FfnRole, NormRole, WeightId, WeightMap, WeightRole, WeightView};

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("poot_fixture_neox_parallel");

/// The HF `model_type` that names the family.
const MODEL_TYPE: &str = "poot_fixture_neox_parallel";

/// The registry entry: what `Registry::builtin_plus` takes.
pub(crate) const ENTRY: FamilyEntry = entry::<NeoxParallel>();

/// The vocabulary of the fixture checkpoint, and its end-of-sequence id.
pub(crate) const VOCAB: usize = 48;
pub(crate) const EOS: u32 = 47;

struct NeoxParallel;

impl StandardFamily for NeoxParallel {
    const FAMILY: FamilyKey = FAMILY;
    const KEYS: &'static [(ConfigSource, &'static str)] =
        &[(ConfigSource::HfModelType, MODEL_TYPE)];

    fn configure(raw: &RawConfig<'_>) -> Result<StandardParams, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let count = |name: &'static str| fields.count(Field::hf(name), None);
        let width = count("hidden_size")?;
        let heads = count("num_attention_heads")?;
        if !width.is_multiple_of(heads) {
            return Err(fields.error("hidden_size", ConfigReason::NotDivisible { by: heads }));
        }
        let head_dim = width / heads;
        let pct = fields.float(Field::hf("rotary_pct"), 0.25)?;
        if !(pct.is_finite() && pct > 0.0 && pct <= 1.0) {
            return Err(fields.error("rotary_pct", ConfigReason::NotFinitePositive));
        }
        let rotary = ((head_dim as f32 * pct) as usize) & !1;
        let theta = fields.float(Field::hf("rotary_emb_base"), 10_000.0)?;
        let rope =
            RopeParams::new(rotary, theta, &RopeFlavor::Plain).map_err(|e| e.for_family(FAMILY))?;
        let attention = AttentionParams::new(heads, heads, head_dim, true, rope)
            .map_err(|e| e.for_family(FAMILY))?
            .with_out_bias()
            .with_fused_qkv()
            .map_err(|e| e.for_family(FAMILY))?;
        let eps = fields.float(Field::hf("layer_norm_eps"), 1e-5)?;
        let norm = NormParams::of(NormKind::Layer, eps).map_err(|e| e.for_family(FAMILY))?;
        // GPT-NeoX's default is the parallel residual; a config may turn it off.
        let layer = if fields.flag(Field::hf("use_parallel_residual"), true)? {
            LayerParams::new(norm).with_placement(NormPlacement::Parallel)
        } else {
            LayerParams::new(norm)
        };
        let layers = count("num_hidden_layers")?;
        Ok(StandardParams {
            vocab: fields.vocab()?,
            width,
            layers,
            max_positions: count("max_position_embeddings")?,
            attention: vec![attention; layers],
            mlp: vec![
                Mlp::Plain {
                    inter: count("intermediate_size")?,
                    act: Gelu::Erf,
                    biased: true,
                };
                layers
            ],
            layer,
            stack: StackParams::new(norm),
            norms: LayerNormKind::Biased,
            final_norm_bias: true,
            embed_norm: false,
            head_required: false,
            eos: fields.eos(None)?,
            bos: fields.bos()?,
            prompt_bos: None,
            chat: ChatFormat::ChatML,
            tool_codec: Some(Arc::new(crate::family::NeoxishToolCodec)),
            embeddable: false,
        })
    }

    fn weights(
        _raw: &RawConfig<'_>,
        store: &WeightStore,
        params: &mut StandardParams,
    ) -> Result<WeightMap, ModelError> {
        let weight_error = |source| ModelError::Weight {
            family: FAMILY,
            source,
        };
        let stored = |name: &str| WeightView::Stored(WeightKey::new(name));
        let mut map = WeightMap::builder(store);
        let mut add =
            |id, name: String| map.map(id, stored(&name)).map(|_| ()).map_err(weight_error);
        add(WeightId::model(WeightRole::Embed), names::EMBED.into())?;
        add(
            WeightId::model(WeightRole::FinalNorm),
            names::FINAL_NORM.into(),
        )?;
        add(
            WeightId::model(WeightRole::FinalNormBias),
            names::FINAL_NORM_BIAS.into(),
        )?;
        add(WeightId::model(WeightRole::Head), names::HEAD.into())?;
        for l in 0..params.layers {
            for (role, tail) in [
                (WeightRole::Norm(NormRole::Attn), "input_layernorm.weight"),
                (WeightRole::Norm(NormRole::AttnBias), "input_layernorm.bias"),
                (
                    WeightRole::Norm(NormRole::Ffn),
                    "post_attention_layernorm.weight",
                ),
                (
                    WeightRole::Norm(NormRole::FfnBias),
                    "post_attention_layernorm.bias",
                ),
                // One fused tensor, read whole: its rows interleave per head.
                (
                    WeightRole::Attn(AttnRole::Qkv),
                    "attention.query_key_value.weight",
                ),
                (
                    WeightRole::Attn(AttnRole::QkvBias),
                    "attention.query_key_value.bias",
                ),
                (WeightRole::Attn(AttnRole::O), "attention.dense.weight"),
                (WeightRole::Attn(AttnRole::OBias), "attention.dense.bias"),
                (WeightRole::Ffn(FfnRole::Up), "mlp.dense_h_to_4h.weight"),
                (WeightRole::Ffn(FfnRole::UpBias), "mlp.dense_h_to_4h.bias"),
                (WeightRole::Ffn(FfnRole::Down), "mlp.dense_4h_to_h.weight"),
                (WeightRole::Ffn(FfnRole::DownBias), "mlp.dense_4h_to_h.bias"),
            ] {
                add(WeightId::layer(l, role), names::layer(l, tail))?;
            }
        }
        Ok(map.build())
    }
}

/// The checkpoint's tensor name table: where each role of the model sits in a GPT-NeoX checkpoint.
mod names {
    pub(super) const EMBED: &str = "gpt_neox.embed_in.weight";
    pub(super) const FINAL_NORM: &str = "gpt_neox.final_layer_norm.weight";
    pub(super) const FINAL_NORM_BIAS: &str = "gpt_neox.final_layer_norm.bias";
    pub(super) const HEAD: &str = "embed_out.weight";

    pub(super) fn layer(l: usize, tail: &str) -> String {
        format!("gpt_neox.layers.{l}.{tail}")
    }
}

/// The fixture's config: two layers of four 16-wide heads (RoPE over 4 dims of each) and the parallel
/// residual, the 48-word vocabulary of [`VOCAB`]. `use_parallel_residual` is the option the rows turn
/// off to see that it decides the tokens.
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": MODEL_TYPE,
        "vocab_size": VOCAB,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "rotary_pct": 0.25,
        "rotary_emb_base": 10000.0,
        "layer_norm_eps": 1e-5,
        "max_position_embeddings": 64,
        "use_parallel_residual": true,
        "tie_word_embeddings": false,
        "eos_token_id": EOS,
    })
}

/// Every tensor of a checkpoint of `config`'s dims, by its GPT-NeoX name.
fn fixture_shapes(config: &serde_json::Value) -> Vec<(String, Vec<usize>)> {
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let (v, w, inter) = (n("vocab_size"), n("hidden_size"), n("intermediate_size"));
    let mut shapes = vec![
        (names::EMBED.to_string(), vec![v, w]),
        (names::FINAL_NORM.to_string(), vec![w]),
        (names::FINAL_NORM_BIAS.to_string(), vec![w]),
        (names::HEAD.to_string(), vec![v, w]),
    ];
    for l in 0..n("num_hidden_layers") {
        for (tail, shape) in [
            ("input_layernorm.weight", vec![w]),
            ("input_layernorm.bias", vec![w]),
            ("post_attention_layernorm.weight", vec![w]),
            ("post_attention_layernorm.bias", vec![w]),
            ("attention.query_key_value.weight", vec![3 * w, w]),
            ("attention.query_key_value.bias", vec![3 * w]),
            ("attention.dense.weight", vec![w, w]),
            ("attention.dense.bias", vec![w]),
            ("mlp.dense_h_to_4h.weight", vec![inter, w]),
            ("mlp.dense_h_to_4h.bias", vec![inter]),
            ("mlp.dense_4h_to_h.weight", vec![w, inter]),
            ("mlp.dense_4h_to_h.bias", vec![w]),
        ] {
            shapes.push((names::layer(l, tail), shape));
        }
    }
    shapes
}

/// How big a tensor's entries are, by what the tensor is (see [`crate::family`]): a norm scale sits near
/// one, a bias is small, and the fused Q/K/V projection is large enough that attention scores separate
/// the positions.
fn magnitude(name: &str) -> (f32, f32) {
    // (centre, spread): entries are `centre + spread * u`, `u` uniform in [-0.5, 0.5).
    if name.ends_with("layernorm.weight") || name.ends_with("layer_norm.weight") {
        (1.0, 0.2)
    } else if name.ends_with(".bias") {
        (0.0, 0.1)
    } else if name.contains("query_key_value") {
        (0.0, 2.0)
    } else {
        (0.0, 0.5)
    }
}

/// The synthetic checkpoint [`ENTRY`] is tested on: BF16 tensors of deterministic pseudo-random values
/// (one xorshift stream over every tensor in order), sized by [`magnitude`].
pub(crate) fn fixture() -> Fixture {
    fixture_of(fixture_config())
}

/// The same checkpoint under `config`'s options (the weights depend on its dims only).
pub(crate) fn fixture_of(config: serde_json::Value) -> Fixture {
    let mut store = WeightStore::builder();
    let mut seed = 0x9e37_79b9u32;
    for (name, shape) in fixture_shapes(&config) {
        let (centre, spread) = magnitude(&name);
        let n: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(2 * n);
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let u = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let value = centre + spread * u;
            bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        let dense = DenseWeight::try_new(DType::BF16, shape, bytes.into())
            .expect("the fixture bytes match their shape");
        store
            .insert(name, WeightEntry::Dense(dense))
            .expect("the fixture names are unique");
    }
    Fixture {
        config,
        generation: None,
        preprocessor: None,
        store: store.build(),
    }
}
