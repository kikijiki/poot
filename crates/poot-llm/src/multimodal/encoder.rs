//! Encoder embedding models (BERT / MiniLM / E5 / BGE), card 057, spec 050. Loads a sentence-transformers
//! checkpoint and produces a sentence embedding via pooling over the encoder's token hidden states.

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::builder::Builder;
use poot_load::safetensors;
use poot_models::encoder::{
    BertConfig, declare_bert_constants, declare_cross_encoder_constants, trace_bert_encoder,
    trace_cross_encoder, trace_cross_encoder_batched,
};
use poot_quant::weights::WeightStore;
use poot_tensor::HostTensor;
use tokenizers::Tokenizer;

use super::{DenseWeights, materialize_f32};
use crate::checkpoint::gguf::transpose2d;
use crate::error::{Result, ResultExt};

/// Load a BERT/MiniLM encoder's weights into the `name -> Tensor` map the encoder graph binds (spec 050
/// slice 3). Re-keys the HF BERT names to the short names `declare_bert_constants` uses; the `*.weight`
/// linears are transposed to poot's `[in, out]`, the embedding tables + LayerNorms copy straight. `prefix`
/// is prepended to every HF name: "" for a `BertModel` embedding checkpoint, "bert." for a
/// `BertForSequenceClassification` cross-encoder (whose encoder weights live under `bert.`).
pub fn load_bert_weights(
    dir: impl AsRef<std::path::Path>,
    cfg: &BertConfig,
    prefix: &str,
) -> Result<DenseWeights> {
    let st = safetensors::load_weight_store(dir).map_err(|e| err!("load safetensors: {e}"))?;
    let mut m = HashMap::new();
    let straight = |st: &WeightStore, name: &str| -> Result<HostTensor> {
        let full = format!("{prefix}{name}");
        materialize_f32(st, &full).map_err(|e| err!("missing {full}: {e}"))
    };
    let lin = |st: &WeightStore, name: &str| -> Result<HostTensor> {
        let full = format!("{prefix}{name}");
        let t = materialize_f32(st, &full).map_err(|e| err!("missing {full}: {e}"))?;
        Ok(transpose2d(&t))
    };
    m.insert(
        "word_emb".into(),
        straight(&st, "embeddings.word_embeddings.weight")?,
    );
    m.insert(
        "pos_emb".into(),
        straight(&st, "embeddings.position_embeddings.weight")?,
    );
    m.insert(
        "type_emb".into(),
        straight(&st, "embeddings.token_type_embeddings.weight")?,
    );
    m.insert(
        "emb_lnw".into(),
        straight(&st, "embeddings.LayerNorm.weight")?,
    );
    m.insert(
        "emb_lnb".into(),
        straight(&st, "embeddings.LayerNorm.bias")?,
    );
    for l in 0..cfg.layers {
        let p = format!("encoder.layer.{l}");
        m.insert(
            format!("wq{l}"),
            lin(&st, &format!("{p}.attention.self.query.weight"))?,
        );
        m.insert(
            format!("bq{l}"),
            straight(&st, &format!("{p}.attention.self.query.bias"))?,
        );
        m.insert(
            format!("wk{l}"),
            lin(&st, &format!("{p}.attention.self.key.weight"))?,
        );
        m.insert(
            format!("bk{l}"),
            straight(&st, &format!("{p}.attention.self.key.bias"))?,
        );
        m.insert(
            format!("wv{l}"),
            lin(&st, &format!("{p}.attention.self.value.weight"))?,
        );
        m.insert(
            format!("bv{l}"),
            straight(&st, &format!("{p}.attention.self.value.bias"))?,
        );
        m.insert(
            format!("wo{l}"),
            lin(&st, &format!("{p}.attention.output.dense.weight"))?,
        );
        m.insert(
            format!("bo{l}"),
            straight(&st, &format!("{p}.attention.output.dense.bias"))?,
        );
        m.insert(
            format!("l1w{l}"),
            straight(&st, &format!("{p}.attention.output.LayerNorm.weight"))?,
        );
        m.insert(
            format!("l1b{l}"),
            straight(&st, &format!("{p}.attention.output.LayerNorm.bias"))?,
        );
        m.insert(
            format!("f1w{l}"),
            lin(&st, &format!("{p}.intermediate.dense.weight"))?,
        );
        m.insert(
            format!("f1b{l}"),
            straight(&st, &format!("{p}.intermediate.dense.bias"))?,
        );
        m.insert(
            format!("f2w{l}"),
            lin(&st, &format!("{p}.output.dense.weight"))?,
        );
        m.insert(
            format!("f2b{l}"),
            straight(&st, &format!("{p}.output.dense.bias"))?,
        );
        m.insert(
            format!("l2w{l}"),
            straight(&st, &format!("{p}.output.LayerNorm.weight"))?,
        );
        m.insert(
            format!("l2b{l}"),
            straight(&st, &format!("{p}.output.LayerNorm.bias"))?,
        );
    }
    Ok(m)
}

/// A loaded BERT/MiniLM encoder embedding model (spec 050 slice 3): the weights + tokenizer + config, ready to
/// produce L2-normalized sentence embeddings via mean pooling.
pub struct EncoderRunner {
    cfg: BertConfig,
    weights: DenseWeights,
    tokenizer: Tokenizer,
    pooling: EncoderPooling,
}

/// The subset of an HF BERT `config.json` the encoder graph needs. Drives [`bert_config_from_dir`] so the
/// encoder runs any BERT-shaped checkpoint (MiniLM / E5 / BGE), not just the hardcoded MiniLM-L6 shape.
#[derive(serde::Deserialize)]
struct HfBertConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_bert_eps")]
    layer_norm_eps: f32,
    #[serde(default = "default_hidden_act")]
    hidden_act: String,
}

fn default_hidden_act() -> String {
    "gelu".into()
}

fn default_max_pos() -> usize {
    512
}
fn default_bert_eps() -> f32 {
    1e-12
}

/// What a BERT-class checkpoint is for, detected from `config.json` (card 057 auto-detect).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BertKind {
    /// `BertModel` (sentence-transformers): produces sentence embeddings -> [`EncoderRunner`].
    Embedding,
    /// `BertForSequenceClassification` (a cross-encoder reranker): scores a query+doc pair ->
    /// [`CrossEncoderRunner`].
    CrossEncoder,
}

/// Detect a BERT-class checkpoint's kind from `dir/config.json`: a `BertForSequenceClassification`
/// architecture is a cross-encoder reranker; any other `Bert*` architecture (or `model_type == "bert"`) is an
/// embedding encoder. Returns `None` for a decoder / non-BERT checkpoint, or a missing/unparseable config.
pub fn bert_kind(dir: impl AsRef<std::path::Path>) -> Option<BertKind> {
    let path = dir.as_ref().join("config.json");
    let text = std::fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let archs: Vec<&str> = v
        .get("architectures")
        .and_then(|a| a.as_array())
        .map(|arr| arr.iter().filter_map(|s| s.as_str()).collect())
        .unwrap_or_default();
    if archs.contains(&"BertForSequenceClassification") {
        return Some(BertKind::CrossEncoder);
    }
    let is_bert = v.get("model_type").and_then(|m| m.as_str()) == Some("bert")
        || archs.iter().any(|s| s.starts_with("Bert"));
    is_bert.then_some(BertKind::Embedding)
}

/// Parse `dir/config.json` into the encoder [`BertConfig`] (shape only). `head_dim = hidden_size / heads`.
fn bert_config_from_dir(dir: impl AsRef<std::path::Path>) -> Result<BertConfig> {
    let path = dir.as_ref().join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| err!("read config.json: {e}"))?;
    let hf: HfBertConfig =
        serde_json::from_str(&text).map_err(|e| err!("parse config.json: {e}"))?;
    if !hf.hidden_size.is_multiple_of(hf.num_attention_heads) {
        bail!(
            "bert hidden_size {} not divisible by num_attention_heads {}",
            hf.hidden_size,
            hf.num_attention_heads
        );
    }
    Ok(BertConfig {
        vocab: hf.vocab_size,
        n_heads: hf.num_attention_heads,
        head_dim: hf.hidden_size / hf.num_attention_heads,
        intermediate: hf.intermediate_size,
        layers: hf.num_hidden_layers,
        max_pos: hf.max_position_embeddings,
        eps: hf.layer_norm_eps,
        // HF "gelu" is the exact erf form (BERT default); the tanh approximation is "gelu_pytorch_tanh" or
        // the older "gelu_new". Anything else falls back to exact erf.
        gelu_tanh: matches!(hf.hidden_act.as_str(), "gelu_pytorch_tanh" | "gelu_new"),
    })
}

/// How a sentence-transformers model pools the encoder's token hidden states into one sentence vector.
/// MiniLM/E5 mean-pool; BGE/GTE take the `[CLS]` token. Read from the model's `1_Pooling/config.json`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EncoderPooling {
    /// Average over all token positions (MiniLM, E5).
    Mean,
    /// The `[CLS]` token (position 0) hidden state (BGE, GTE).
    Cls,
}

/// Read the pooling mode from `dir/1_Pooling/config.json` (the sentence-transformers pooling module). CLS
/// when `pooling_mode_cls_token` is true, else mean. Defaults to mean when the file is absent (a bare BERT
/// checkpoint with no pooling module - the MiniLM convention).
fn read_pooling_mode(dir: impl AsRef<std::path::Path>) -> EncoderPooling {
    let path = dir.as_ref().join("1_Pooling").join("config.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return EncoderPooling::Mean;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return EncoderPooling::Mean;
    };
    if v.get("pooling_mode_cls_token").and_then(|c| c.as_bool()) == Some(true) {
        EncoderPooling::Cls
    } else {
        EncoderPooling::Mean
    }
}

impl EncoderRunner {
    /// Load a BERT/MiniLM-class checkpoint directory (BERT weights + the WordPiece `tokenizer.json`). The
    /// shape is read from `config.json`, so any BERT-shaped sentence encoder loads (E5/BGE/MiniLM), not just
    /// MiniLM-L6. The pooling mode (mean vs `[CLS]`) is read from `1_Pooling/config.json`.
    pub fn load(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = bert_config_from_dir(dir)?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| err!("load tokenizer: {e}"))?;
        // Sentence-transformers ship Fixed(128) padding in tokenizer.json. poot embeds one sentence at a time and the
        // encoder has no attention mask, so padding would feed [PAD] into unmasked attention and the mean pool.
        // Drop it: each encode returns just the real tokens.
        tokenizer.with_padding(None);
        Ok(EncoderRunner {
            weights: load_bert_weights(dir, &cfg, "")?,
            tokenizer,
            cfg,
            pooling: read_pooling_mode(dir),
        })
    }

    /// The pooling mode this model uses (mean or `[CLS]`), from its `1_Pooling/config.json`.
    pub fn pooling(&self) -> EncoderPooling {
        self.pooling
    }

    /// The hidden / embedding dimension (384 for MiniLM-L6).
    pub fn dim(&self) -> usize {
        self.cfg.hidden()
    }

    /// WordPiece token count for `text` (incl. the `[CLS]`/`[SEP]` specials), for `usage` reporting.
    pub fn token_count(&self, text: &str) -> Result<usize> {
        Ok(self
            .tokenizer
            .encode(text, true)
            .map_err(|e| err!("encode: {e}"))?
            .get_ids()
            .len())
    }

    /// Embed a sentence: WordPiece-tokenize (with `[CLS]`/`[SEP]`), run the encoder, pool the token hidden
    /// states (mean or `[CLS]`, per the model's `1_Pooling` config), and L2-normalize. Returns a `[hidden]`
    /// vector (cosine == dot product).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_with(text, self.pooling)
    }

    /// Like [`Self::embed`] but with an explicit pooling mode - the per-request `pooling` override on
    /// `/v1/embeddings` (card 091) - instead of the model's `1_Pooling` default.
    pub fn embed_with(&self, text: &str, pooling: EncoderPooling) -> Result<Vec<f32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| err!("encode: {e}"))?;
        let ids = enc.get_ids();
        let seq = ids.len();
        if seq == 0 {
            bail!("cannot embed empty text");
        }
        let h = self.cfg.hidden();
        let b = Builder::new();
        let tokens = b.constant("tokens", poot_graph_ir::types::TensorType::f32(vec![seq]));
        let (w, binds) = declare_bert_constants(&b, &self.cfg);
        let out = trace_bert_encoder(&b, tokens, &w, &self.cfg, seq);
        let g = b.finish(out);

        let mut inputs = HashMap::new();
        inputs.insert(
            tokens.id,
            Value::from(HostTensor::f32(
                vec![seq],
                ids.iter().map(|&t| t as f32).collect(),
            )),
        );
        for (id, name) in &binds {
            inputs.insert(*id, Value::from(self.weights[name].clone()));
        }
        let hidden = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .context("encoder eval")?
            .output
            .into_host()
            .context("encoder eval")?; // [1, L, h]

        // pool the token hidden states, then L2-normalize.
        let mut emb = vec![0.0f32; h];
        match pooling {
            // CLS pooling: just the [CLS] token (position 0).
            EncoderPooling::Cls => emb.copy_from_slice(&hidden.as_f32().unwrap()[..h]),
            // mean pooling: average over the L tokens.
            EncoderPooling::Mean => {
                for pos in 0..seq {
                    for (j, e) in emb.iter_mut().enumerate() {
                        *e += hidden.as_f32().unwrap()[pos * h + j];
                    }
                }
            }
        }
        let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in emb.iter_mut() {
            *v /= norm;
        }
        Ok(emb)
    }

    /// Bi-encoder rerank: score each `doc` by cosine similarity to `query` (the L2-normalized embeddings'
    /// dot product), returning `(index, score)` sorted most-relevant first. Mirrors `Runner::rerank` on the
    /// higher-quality encoder embeddings.
    pub fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<(usize, f32)>> {
        let q = self.embed(query)?;
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(docs.len());
        for (i, doc) in docs.iter().enumerate() {
            let d = self.embed(doc)?;
            let score = q.iter().zip(&d).map(|(a, b)| a * b).sum::<f32>();
            scored.push((i, score));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

/// Load a cross-encoder's weights: the BERT encoder (under the `bert.` prefix) plus the pooler + classifier
/// head, keyed to the names `declare_cross_encoder_constants` binds. `*.weight` linears transpose to
/// `[in, out]`; biases copy straight.
fn load_cross_encoder_weights(
    dir: impl AsRef<std::path::Path>,
    cfg: &BertConfig,
) -> Result<DenseWeights> {
    let dir = dir.as_ref();
    let mut m = load_bert_weights(dir, cfg, "bert.")?;
    let st = safetensors::load_weight_store(dir).map_err(|e| err!("load safetensors: {e}"))?;
    let straight = |name: &str| -> Result<HostTensor> {
        materialize_f32(&st, name).map_err(|e| err!("missing {name}: {e}"))
    };
    let lin = |name: &str| -> Result<HostTensor> {
        let t = materialize_f32(&st, name).map_err(|e| err!("missing {name}: {e}"))?;
        Ok(transpose2d(&t))
    };
    m.insert("pooler_w".into(), lin("bert.pooler.dense.weight")?);
    m.insert("pooler_b".into(), straight("bert.pooler.dense.bias")?);
    m.insert("cls_w".into(), lin("classifier.weight")?);
    m.insert("cls_b".into(), straight("classifier.bias")?);
    Ok(m)
}

/// A loaded cross-encoder reranker (`BertForSequenceClassification`, card 057): scores a (query, document)
/// pair jointly through one BERT forward and the classifier head. The bi-encoder [`EncoderRunner::rerank`]
/// embeds query and doc separately; this costs one forward pass per pair.
pub struct CrossEncoderRunner {
    cfg: BertConfig,
    weights: DenseWeights,
    tokenizer: Tokenizer,
}

impl CrossEncoderRunner {
    /// Load a `BertForSequenceClassification` cross-encoder (e.g. `ms-marco-MiniLM-L-6-v2`): the BERT shape
    /// from `config.json`, the `bert.`-prefixed encoder weights + the pooler/classifier head, and the
    /// WordPiece tokenizer (padding disabled - pairs are scored one at a time, no attention mask).
    pub fn load(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = bert_config_from_dir(dir)?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| err!("load tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        Ok(CrossEncoderRunner {
            weights: load_cross_encoder_weights(dir, &cfg)?,
            tokenizer,
            cfg,
        })
    }

    /// Relevance logit for a (query, document) pair: tokenize `[CLS] query [SEP] doc [SEP]` (with segment ids
    /// 0/1), run the cross-encoder, return the classifier logit. Higher = more relevant (the model's default
    /// output activation is identity, so the raw logit is the score; it is not bounded to [0, 1]).
    pub fn score(&self, query: &str, doc: &str) -> Result<f32> {
        let enc = self
            .tokenizer
            .encode((query, doc), true)
            .map_err(|e| err!("encode pair: {e}"))?;
        let ids = enc.get_ids();
        let type_ids = enc.get_type_ids();
        let seq = ids.len();
        if seq == 0 {
            bail!("cannot score an empty pair");
        }
        let b = Builder::new();
        let tokens = b.constant("tokens", poot_graph_ir::types::TensorType::f32(vec![seq]));
        let token_types = b.constant(
            "token_types",
            poot_graph_ir::types::TensorType::f32(vec![seq]),
        );
        let (w, binds) = declare_cross_encoder_constants(&b, &self.cfg);
        let out = trace_cross_encoder(&b, tokens, token_types, &w, &self.cfg, seq);
        let g = b.finish(out);

        let mut inputs = HashMap::new();
        inputs.insert(
            tokens.id,
            Value::from(HostTensor::f32(
                vec![seq],
                ids.iter().map(|&t| t as f32).collect(),
            )),
        );
        inputs.insert(
            token_types.id,
            Value::from(HostTensor::f32(
                vec![seq],
                type_ids.iter().map(|&t| t as f32).collect(),
            )),
        );
        for (id, name) in &binds {
            inputs.insert(*id, Value::from(self.weights[name].clone()));
        }
        let logit = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .context("cross-encoder eval")?
            .output
            .into_host()
            .context("cross-encoder eval")?; // [1]
        Ok(logit.as_f32().unwrap()[0])
    }

    /// Relevance logits for many documents against one query, in a SINGLE batched forward (card 091). Each
    /// `[CLS] query [SEP] doc [SEP]` pair is right-padded to the batch's longest length; an additive key-padding
    /// mask keeps every real position from attending to padding, so the `[CLS]` logits equal the per-pair
    /// [`Self::score`] results. Returns one logit per document, in input order. Empty `docs` -> empty vec.
    pub fn score_batch(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let batch = docs.len();
        // tokenize every pair, then right-pad to the longest (pad token id 0 = [PAD], segment 0).
        let mut ids: Vec<Vec<u32>> = Vec::with_capacity(batch);
        let mut types: Vec<Vec<u32>> = Vec::with_capacity(batch);
        for doc in docs {
            let enc = self
                .tokenizer
                .encode((query, *doc), true)
                .map_err(|e| err!("encode pair: {e}"))?;
            if enc.get_ids().is_empty() {
                bail!("cannot score an empty pair");
            }
            ids.push(enc.get_ids().to_vec());
            types.push(enc.get_type_ids().to_vec());
        }
        let seq = ids.iter().map(|v| v.len()).max().unwrap();
        // flat [batch*seq] token / segment indices, and the additive key-padding mask [batch,1,1,seq].
        let mut tok_flat = vec![0.0f32; batch * seq];
        let mut typ_flat = vec![0.0f32; batch * seq];
        let mut mask = vec![0.0f32; batch * seq]; // [batch,1,1,seq] flattened
        for i in 0..batch {
            let real = ids[i].len();
            for j in 0..seq {
                let off = i * seq + j;
                if j < real {
                    tok_flat[off] = ids[i][j] as f32;
                    typ_flat[off] = types[i][j] as f32;
                } else {
                    mask[off] = -1.0e9; // padded key: zero weight after softmax
                }
            }
        }
        let b = Builder::new();
        let tt = poot_graph_ir::types::TensorType::f32(vec![batch * seq]);
        let tokens = b.constant("tokens", tt.clone());
        let token_types = b.constant("token_types", tt);
        let mask_c = b.constant(
            "kmask",
            poot_graph_ir::types::TensorType::f32(vec![batch, 1, 1, seq]),
        );
        let (w, binds) = declare_cross_encoder_constants(&b, &self.cfg);
        let out =
            trace_cross_encoder_batched(&b, tokens, token_types, mask_c, &w, &self.cfg, batch, seq);
        let g = b.finish(out);

        let mut inputs = HashMap::new();
        inputs.insert(
            tokens.id,
            Value::from(HostTensor::f32(vec![batch * seq], tok_flat)),
        );
        inputs.insert(
            token_types.id,
            Value::from(HostTensor::f32(vec![batch * seq], typ_flat)),
        );
        inputs.insert(
            mask_c.id,
            Value::from(HostTensor::f32(vec![batch, 1, 1, seq], mask)),
        );
        for (id, name) in &binds {
            inputs.insert(*id, Value::from(self.weights[name].clone()));
        }
        let logits = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .context("batched cross-encoder eval")?
            .output
            .into_host()
            .context("batched cross-encoder eval")?; // [batch]
        Ok(logits.as_f32().unwrap().to_vec())
    }

    /// Cross-encoder rerank: score each `doc` against `query` jointly and return `(index, score)` sorted
    /// most-relevant first. One batched forward over all documents ([`Self::score_batch`]).
    pub fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<(usize, f32)>> {
        let mut scored: Vec<(usize, f32)> = self
            .score_batch(query, docs)?
            .into_iter()
            .enumerate()
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(shape: Vec<usize>, data: Vec<f32>) -> HostTensor {
        HostTensor::f32(shape, data)
    }

    #[test]
    fn transpose2d_transposes_a_known_matrix() {
        // [2,3] row-major [[1,2,3],[4,5,6]] -> [3,2] [[1,4],[2,5],[3,6]]. This is the HF [out,in] ->
        // poot [in,out] linear-weight transpose every encoder projection relies on; a wrong index here
        // silently corrupts every embedding.
        let t = raw(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = transpose2d(&t);
        assert_eq!(out.shape(), vec![3, 2]);
        assert_eq!(
            out.as_f32().unwrap().to_vec(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    fn write_cfg(dir: &std::path::Path, hidden: usize, heads: usize, act: &str) {
        std::fs::create_dir_all(dir).unwrap();
        let cfg = serde_json::json!({
            "vocab_size": 30522, "hidden_size": hidden, "num_attention_heads": heads,
            "num_hidden_layers": 6, "intermediate_size": hidden * 4, "hidden_act": act,
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    }

    #[test]
    fn bert_config_derives_head_dim_and_gelu_variant() {
        let base = std::env::temp_dir().join(format!("poot-enc-cfg-{}", std::process::id()));
        // exact-erf gelu (BERT default) -> gelu_tanh = false; the tanh approximations -> true.
        for (act, want_tanh) in [
            ("gelu", false),
            ("gelu_pytorch_tanh", true),
            ("gelu_new", true),
            ("relu", false), // unknown act falls back to exact erf
        ] {
            let d = base.join(act);
            write_cfg(&d, 384, 12, act);
            let cfg = bert_config_from_dir(&d).expect("parse config");
            assert_eq!(cfg.head_dim, 32, "384/12");
            assert_eq!(cfg.n_heads, 12);
            assert_eq!(cfg.hidden(), 384);
            assert_eq!(cfg.gelu_tanh, want_tanh, "act={act}");
            let _ = std::fs::remove_dir_all(&d);
        }
        // hidden_size not divisible by heads is a clean error, not a truncating head_dim.
        let bad = base.join("bad");
        write_cfg(&bad, 100, 12, "gelu");
        assert!(bert_config_from_dir(&bad).is_err());
        let _ = std::fs::remove_dir_all(&bad);
    }

    #[test]
    fn read_pooling_mode_reads_cls_flag_else_mean() {
        let base = std::env::temp_dir().join(format!("poot-enc-pool-{}", std::process::id()));
        // absent 1_Pooling -> Mean (bare BERT convention).
        let none = base.join("none");
        std::fs::create_dir_all(&none).unwrap();
        assert_eq!(read_pooling_mode(&none), EncoderPooling::Mean);

        // cls_token = true -> Cls.
        let cls = base.join("cls");
        std::fs::create_dir_all(cls.join("1_Pooling")).unwrap();
        std::fs::write(
            cls.join("1_Pooling").join("config.json"),
            br#"{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}"#,
        )
        .unwrap();
        assert_eq!(read_pooling_mode(&cls), EncoderPooling::Cls);

        // cls_token = false -> Mean.
        let mean = base.join("mean");
        std::fs::create_dir_all(mean.join("1_Pooling")).unwrap();
        std::fs::write(
            mean.join("1_Pooling").join("config.json"),
            br#"{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true}"#,
        )
        .unwrap();
        assert_eq!(read_pooling_mode(&mean), EncoderPooling::Mean);
        let _ = std::fs::remove_dir_all(&base);
    }
}
