//! Dense decoder models of chosen dimensions behind [`Model`], for every test, bench and tool that
//! needs a realistic family graph: the config and the HF-named checkpoint of one family at the
//! dimensions the caller picks, built through the shipped registry (so the graph is what
//! [`Model::trace`] returns for that checkpoint, not a second tracer).
//!
//! A [`Dense`] starts at its family's fixture dimensions; each setter changes one fact of the
//! config. Weights are deterministic: norm scales near 1, everything else small and centred, or
//! whatever `values` returns for a tensor name.

use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_graph_ir::{Graph, NoValidations, Storage, ValidationChannel, ValidationOutputs};
use poot_models::model::{KvLayout, LogitRows, StepShape};
use poot_models::registry::{RawConfig, Registry};
use poot_quant::format::WeightFormat;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::DType;

use crate::weight_map::{MappedModel, Projections};

/// The dense families a fixture can be built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Q/K/V biases, no QK norm.
    Qwen2,
    /// No bias, per-head Q/K norm, an explicit head dimension.
    Qwen3,
    /// No bias, no norm on Q/K.
    Llama,
    /// Post-norm sublayers and Q/K norm over the whole projection width.
    Olmo2,
    /// Sandwich norms, per-head Q/K norm, local and global layers, GeGLU.
    Gemma3,
    /// Sandwich norms, local and global layers, attention and final logit softcaps, GeGLU.
    Gemma2,
}

/// The values of one tensor: its HF name and shape in, its elements out.
pub type Values = dyn Fn(&str, &[usize]) -> Vec<f32>;

/// One dense checkpoint's dimensions and config.
#[derive(Clone, Debug)]
pub struct Dense {
    pub family: Family,
    pub vocab: usize,
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    /// The explicit head dimension; `None` is `hidden / heads`.
    pub head_dim: Option<usize>,
    pub max_positions: usize,
    config: serde_json::Map<String, serde_json::Value>,
}

fn random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

impl Dense {
    /// `family` at its fixture's dimensions (the registry's tiny deterministic checkpoint).
    pub fn new(family: Family) -> Self {
        let (layers, head_dim) = match family {
            Family::Qwen2 | Family::Llama | Family::Olmo2 => (2, None),
            Family::Qwen3 => (2, Some(32)),
            Family::Gemma3 => (3, Some(16)),
            Family::Gemma2 => (4, Some(16)),
        };
        let mut config = serde_json::Map::new();
        let mut set = |key: &str, value: serde_json::Value| {
            config.insert(key.to_string(), value);
        };
        set(
            "model_type",
            match family {
                Family::Qwen2 => "qwen2",
                Family::Qwen3 => "qwen3",
                Family::Llama => "llama",
                Family::Olmo2 => "olmo2",
                Family::Gemma3 => "gemma3_text",
                Family::Gemma2 => "gemma2",
            }
            .into(),
        );
        set("rms_norm_eps", 1e-6.into());
        set("rope_theta", 10000.0.into());
        set("tie_word_embeddings", false.into());
        set("bos_token_id", 1.into());
        match family {
            Family::Gemma3 => {
                set("rope_local_base_freq", 100.0.into());
                set("sliding_window", 3.into());
                set("sliding_window_pattern", 2.into());
                set("query_pre_attn_scalar", 20.0.into());
            }
            Family::Gemma2 => {
                set("sliding_window", 3.into());
                set("query_pre_attn_scalar", 20.0.into());
                set("attn_logit_softcapping", 5.0.into());
                set("final_logit_softcapping", 3.0.into());
            }
            _ => {}
        }
        Self {
            family,
            vocab: 48,
            hidden: 64,
            inter: 96,
            layers,
            heads: 4,
            kv_heads: 2,
            head_dim,
            max_positions: 64,
            config,
        }
    }

    /// One config key set to `value` (`None` is JSON null, which a family reads as absent).
    pub fn with(mut self, key: &str, value: impl Into<serde_json::Value>) -> Self {
        self.config.insert(key.to_string(), value.into());
        self
    }

    pub fn vocab(mut self, vocab: usize) -> Self {
        self.vocab = vocab;
        self
    }

    pub fn dims(mut self, hidden: usize, inter: usize, layers: usize) -> Self {
        (self.hidden, self.inter, self.layers) = (hidden, inter, layers);
        self
    }

    pub fn heads(mut self, heads: usize, kv_heads: usize) -> Self {
        (self.heads, self.kv_heads) = (heads, kv_heads);
        self
    }

    pub fn head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = Some(head_dim);
        self
    }

    pub fn max_positions(mut self, max_positions: usize) -> Self {
        self.max_positions = max_positions;
        self
    }

    /// The HF config of this checkpoint.
    pub fn config(&self) -> serde_json::Value {
        let mut config = self.config.clone();
        let mut set = |key: &str, value: serde_json::Value| {
            config.insert(key.to_string(), value);
        };
        set("vocab_size", self.vocab.into());
        set("hidden_size", self.hidden.into());
        set("intermediate_size", self.inter.into());
        set("num_hidden_layers", self.layers.into());
        set("num_attention_heads", self.heads.into());
        set("num_key_value_heads", self.kv_heads.into());
        set("max_position_embeddings", self.max_positions.into());
        set("eos_token_id", (self.vocab - 1).into());
        if let Some(d) = self.head_dim {
            set("head_dim", d.into());
        }
        config.into()
    }

    fn head(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden / self.heads)
    }

    /// Every tensor of the checkpoint: its HF name and shape.
    pub fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        let (v, h, inter) = (self.vocab, self.hidden, self.inter);
        let d = self.head();
        let (q, kv) = (self.heads * d, self.kv_heads * d);
        let mut tensors = vec![
            ("model.embed_tokens.weight".to_string(), vec![v, h]),
            ("model.norm.weight".to_string(), vec![h]),
            ("lm_head.weight".to_string(), vec![v, h]),
        ];
        for l in 0..self.layers {
            let k = |s: &str| format!("model.layers.{l}.{s}");
            let mut layer = vec![
                (k("post_attention_layernorm.weight"), vec![h]),
                (k("self_attn.q_proj.weight"), vec![q, h]),
                (k("self_attn.k_proj.weight"), vec![kv, h]),
                (k("self_attn.v_proj.weight"), vec![kv, h]),
                (k("self_attn.o_proj.weight"), vec![h, q]),
                (k("mlp.gate_proj.weight"), vec![inter, h]),
                (k("mlp.up_proj.weight"), vec![inter, h]),
                (k("mlp.down_proj.weight"), vec![h, inter]),
            ];
            if self.family != Family::Olmo2 {
                layer.push((k("input_layernorm.weight"), vec![h]));
            }
            match self.family {
                Family::Qwen2 => layer.extend([
                    (k("self_attn.q_proj.bias"), vec![q]),
                    (k("self_attn.k_proj.bias"), vec![kv]),
                    (k("self_attn.v_proj.bias"), vec![kv]),
                ]),
                Family::Qwen3 => layer.extend([
                    (k("self_attn.q_norm.weight"), vec![d]),
                    (k("self_attn.k_norm.weight"), vec![d]),
                ]),
                Family::Olmo2 => layer.extend([
                    (k("post_feedforward_layernorm.weight"), vec![h]),
                    (k("self_attn.q_norm.weight"), vec![q]),
                    (k("self_attn.k_norm.weight"), vec![kv]),
                ]),
                Family::Gemma3 => layer.extend([
                    (k("pre_feedforward_layernorm.weight"), vec![h]),
                    (k("post_feedforward_layernorm.weight"), vec![h]),
                    (k("self_attn.q_norm.weight"), vec![d]),
                    (k("self_attn.k_norm.weight"), vec![d]),
                ]),
                Family::Gemma2 => layer.extend([
                    (k("pre_feedforward_layernorm.weight"), vec![h]),
                    (k("post_feedforward_layernorm.weight"), vec![h]),
                ]),
                Family::Llama => {}
            }
            tensors.extend(layer);
        }
        tensors
    }

    /// The checkpoint: projections as `projections` (Q8_0 needs every projection width a multiple
    /// of 32), the embedding as `embed`, every other tensor BF16 (or F32 values from `values`).
    pub fn store(
        &self,
        projections: Projections,
        embed: DType,
        values: Option<&Values>,
    ) -> WeightStore {
        let mut store = WeightStore::builder();
        for (seed, (key, shape)) in self.tensors().into_iter().enumerate() {
            let seed = seed as u64 + 1;
            let n: usize = shape.iter().product();
            let entry = if let Some(values) = values {
                let data = values(&key, &shape);
                assert_eq!(data.len(), n, "{key}: values for its shape");
                dense(DType::F32, shape, &data)
            } else if key.ends_with("_proj.weight") && projections == Projections::Q8_0 {
                WeightEntry::Packed(Arc::new(poot_test_util::packed::random_payload(
                    WeightFormat::Q8_0,
                    [shape[0], shape[1]],
                    seed,
                )))
            } else if key.contains("norm") {
                let data: Vec<f32> = random(n, seed).iter().map(|u| 1.0 + 0.2 * u).collect();
                dense(DType::BF16, shape, &data)
            } else {
                let data: Vec<f32> = random(n, seed).iter().map(|u| 0.5 * u).collect();
                let dtype = if key == "model.embed_tokens.weight" {
                    embed
                } else {
                    DType::BF16
                };
                dense(dtype, shape, &data)
            };
            store.insert(key, entry).unwrap();
        }
        store.build()
    }

    /// The model over a BF16 checkpoint.
    pub fn model(&self) -> MappedModel {
        self.build(Projections::Bf16, DType::BF16, None)
    }

    /// The model over a checkpoint of zeros stored as `dtype` (`F32` or `BF16`), for a test that
    /// reads only the traced graph (or binds its own weights) at a real checkpoint's dimensions:
    /// zero pages cost no time to fill.
    pub fn zeroed_model(&self, dtype: DType) -> MappedModel {
        let width = match dtype {
            DType::F32 => 4,
            DType::BF16 => 2,
            other => panic!("fixture weights are F32 or BF16, not {other:?}"),
        };
        let mut store = WeightStore::builder();
        for (key, shape) in self.tensors() {
            let n: usize = shape.iter().product();
            let dense =
                DenseWeight::try_new(dtype, shape, Arc::from(vec![0u8; width * n])).unwrap();
            store.insert(key, WeightEntry::Dense(dense)).unwrap();
        }
        self.register(store.build())
    }

    /// The model over an F32 checkpoint: its graph's consts are all F32, so a store that holds
    /// every const as F32 (the hash-seeded fill of a fixture) binds it.
    pub fn f32_model(&self) -> MappedModel {
        self.build(
            Projections::Bf16,
            DType::F32,
            Some(&|name, shape| {
                let n = shape.iter().product();
                let scale = if name.contains("norm") { 0.2 } else { 0.5 };
                random(n, name.len() as u64 + 1)
                    .iter()
                    .map(|u| {
                        if name.contains("norm") {
                            1.0 + scale * u
                        } else {
                            scale * u
                        }
                    })
                    .collect()
            }),
        )
    }

    /// The model over the checkpoint [`Dense::store`] builds.
    pub fn build(
        &self,
        projections: Projections,
        embed: DType,
        values: Option<&Values>,
    ) -> MappedModel {
        self.register(self.store(projections, embed, values))
    }

    /// The model the registry builds over `store`.
    fn register(&self, store: WeightStore) -> MappedModel {
        let config = self.config();
        let raw = RawConfig::HfJson {
            config: &config,
            generation: None,
        };
        let model = Registry::builtin()
            .unwrap()
            .build(&raw, &store)
            .unwrap_or_else(|e| panic!("{:?}: {e}", self.family));
        let map = Arc::new(model.weights().clone());
        MappedModel {
            model,
            store: Arc::new(store),
            map,
        }
    }
}

fn dense(dtype: DType, shape: Vec<usize>, values: &[f32]) -> WeightEntry {
    let bytes: Vec<u8> = match dtype {
        DType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::BF16 => values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        other => panic!("fixture weights are F32 or BF16, not {other:?}"),
    };
    WeightEntry::Dense(DenseWeight::try_new(dtype, shape, Arc::from(bytes)).unwrap())
}

/// The weights `g` reads, as a store keyed by the const names `g` declares (the form
/// [`poot_executor::WeightSource::ConstNames`] binds and the CPU oracle reads): each dense const
/// the model's map names, materialized from `m`'s checkpoint. A const the map does not name is a
/// panic, not a silent zero.
pub fn const_named_store<V: ValidationChannel>(m: &MappedModel, g: &Graph<V>) -> WeightStore {
    let ids: std::collections::HashMap<String, _> = m
        .map
        .iter()
        .map(|(id, _, _)| (id.const_name(), id))
        .collect();
    let mut store = WeightStore::builder();
    for &input in &g.inputs {
        let meta = g.meta(input);
        if meta.storage != Storage::Const {
            continue;
        }
        let name = meta.name.as_deref().expect("a const has a name");
        let id = *ids
            .get(name)
            .unwrap_or_else(|| panic!("the weight map names no const {name}"));
        store
            .insert(name, m.map.materialize(id, &m.store).unwrap())
            .unwrap();
    }
    store.build()
}

/// `g` as an ordinary [`Graph`], for the fixtures and passes that take one. A dense family declares
/// no validation output, so nothing is dropped; a graph that declared one is a panic.
pub fn plain(g: Graph<ValidationOutputs>) -> Graph {
    assert!(
        g.validation_outputs().is_empty(),
        "a dense family declares no validation output"
    );
    let Graph {
        values,
        inputs,
        consts,
        slots,
        eqns,
        output,
        state,
        validations: _,
    } = g;
    Graph {
        values,
        inputs,
        consts,
        slots,
        eqns,
        output,
        validations: NoValidations,
        state,
    }
}

/// One contiguous-cache step of `rows` sequences of `tokens` new tokens over `capacity` positions.
pub fn step(rows: usize, tokens: usize, capacity: usize, logits: LogitRows) -> StepShape {
    StepShape {
        rows: NonZeroUsize::new(rows).unwrap(),
        tokens: NonZeroUsize::new(tokens).unwrap(),
        capacity: NonZeroUsize::new(capacity).unwrap(),
        kv: KvLayout::Contiguous,
        logits,
    }
}

/// [`step`] over one paged KV pool of `pool_slots` positions.
pub fn paged_step(
    rows: usize,
    tokens: usize,
    capacity: usize,
    pool_slots: usize,
    logits: LogitRows,
) -> StepShape {
    StepShape {
        kv: KvLayout::Paged {
            pool_slots: NonZeroUsize::new(pool_slots).unwrap(),
        },
        ..step(rows, tokens, capacity, logits)
    }
}
