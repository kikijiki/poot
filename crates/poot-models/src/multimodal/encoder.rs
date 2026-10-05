//! Encoder embedding models (BERT / MiniLM / E5 / BGE class), spec 050 and card 057. A non-causal,
//! full-attention transformer with a pooling head for sentence embeddings. The block shape matches the SigLIP
//! vision encoder, so this reuses its MHA + LayerNorm + GELU blocks; the differences are post-norm (SigLIP is
//! pre-norm), the token/position/type embeddings, and mean pooling.

use poot_graph_ir::builder::{Builder, Traced};
use poot_graph_ir::op::{BinOp, UnOp};
use poot_graph_ir::ops::{attention_masked, gelu, gelu_erf, layernorm, linear};

use crate::vision::VisionBlockWeights;

/// Shape parameters for a BERT/MiniLM encoder (from its config).
#[derive(Clone, Copy, Debug)]
pub struct BertConfig {
    pub vocab: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub max_pos: usize,
    pub eps: f32,
    /// MLP activation: `false` (default) = exact erf GELU (HF `hidden_act: "gelu"`, the BERT default);
    /// `true` = the tanh approximation (`gelu_pytorch_tanh`/`gelu_new`).
    pub gelu_tanh: bool,
}

impl BertConfig {
    pub fn hidden(&self) -> usize {
        self.n_heads * self.head_dim
    }
}

/// BERT input embeddings (spec 050 slice 2): `word_embeddings[tokens] + position_embeddings[0..L] +
/// token_type_embeddings[0]`, summed, then LayerNorm. `tokens` is an `[L]` index; token type 0 (a single
/// sentence). Returns `[1, L, hidden]` for the encoder blocks.
#[allow(clippy::too_many_arguments)]
pub fn trace_bert_embeddings(
    b: &Builder,
    tokens: Traced,
    word_emb: Traced,
    pos_emb: Traced,
    type_emb: Traced,
    ln_w: Traced,
    ln_b: Traced,
    seq: usize,
    hidden: usize,
    eps: f32,
) -> Traced {
    let word = b.gather(word_emb, 0, tokens); // [L, h]
    let pos = b.slice(pos_emb, 0, 0, seq); // [L, h] (the first L position rows)
    let type0 = b.slice(type_emb, 0, 0, 1); // [1, h] (token type 0), broadcasts over L
    let sum = b.binary(BinOp::Add, b.binary(BinOp::Add, word, pos), type0);
    let normed = layernorm(b, sum, ln_w, ln_b, eps);
    b.reshape(normed, vec![1, seq, hidden])
}

/// All weights for a BERT/MiniLM encoder: the embedding tables (word/position/token-type) + the embedding
/// LayerNorm, and one [`VisionBlockWeights`] per transformer layer.
pub struct BertEncoderWeights {
    pub word_emb: Traced,
    pub pos_emb: Traced,
    pub type_emb: Traced,
    pub emb_ln_w: Traced,
    pub emb_ln_b: Traced,
    pub layers: Vec<VisionBlockWeights>,
}

/// Declare a BERT encoder's weights as named graph constants (spec 050 slice 3), returning the weights struct
/// + the `(id, name)` pairs to bind from a loaded weight map. The names match `poot_llm::encoder::load_bert_weights`.
pub fn declare_bert_constants(
    b: &Builder,
    cfg: &BertConfig,
) -> (BertEncoderWeights, Vec<(poot_graph_ir::ValueId, String)>) {
    let mut bd = Vec::new();
    let h = cfg.hidden();
    let i = cfg.intermediate;
    let mut c = |name: String, shape: Vec<usize>| -> Traced {
        let t = b.constant(&name, poot_graph_ir::types::TensorType::f32(shape));
        bd.push((t.id, name));
        t
    };
    let word_emb = c("word_emb".into(), vec![cfg.vocab, h]);
    let pos_emb = c("pos_emb".into(), vec![cfg.max_pos, h]);
    let type_emb = c("type_emb".into(), vec![2, h]);
    let emb_ln_w = c("emb_lnw".into(), vec![h]);
    let emb_ln_b = c("emb_lnb".into(), vec![h]);
    let mut layers = Vec::with_capacity(cfg.layers);
    for l in 0..cfg.layers {
        layers.push(VisionBlockWeights {
            ln1_w: c(format!("l1w{l}"), vec![h]),
            ln1_b: c(format!("l1b{l}"), vec![h]),
            mha: crate::vision::MhaWeights {
                wq: c(format!("wq{l}"), vec![h, h]),
                bq: c(format!("bq{l}"), vec![h]),
                wk: c(format!("wk{l}"), vec![h, h]),
                bk: c(format!("bk{l}"), vec![h]),
                wv: c(format!("wv{l}"), vec![h, h]),
                bv: c(format!("bv{l}"), vec![h]),
                wo: c(format!("wo{l}"), vec![h, h]),
                bo: c(format!("bo{l}"), vec![h]),
            },
            ln2_w: c(format!("l2w{l}"), vec![h]),
            ln2_b: c(format!("l2b{l}"), vec![h]),
            fc1_w: c(format!("f1w{l}"), vec![h, i]),
            fc1_b: c(format!("f1b{l}"), vec![i]),
            fc2_w: c(format!("f2w{l}"), vec![i, h]),
            fc2_b: c(format!("f2b{l}"), vec![h]),
        });
    }
    (
        BertEncoderWeights {
            word_emb,
            pos_emb,
            type_emb,
            emb_ln_w,
            emb_ln_b,
            layers,
        },
        bd,
    )
}

/// The full BERT/MiniLM encoder (spec 050 slice 2): embeddings -> `layers` post-norm blocks. Returns the
/// `[1, L, hidden]` final hidden states; the caller mean-pools + L2-normalizes them into a sentence vector.
pub fn trace_bert_encoder(
    b: &Builder,
    tokens: Traced,
    w: &BertEncoderWeights,
    cfg: &BertConfig,
    seq: usize,
) -> Traced {
    let h = cfg.hidden();
    let mut x = trace_bert_embeddings(
        b, tokens, w.word_emb, w.pos_emb, w.type_emb, w.emb_ln_w, w.emb_ln_b, seq, h, cfg.eps,
    );
    for blk in &w.layers {
        x = trace_bert_block(b, x, blk, cfg.n_heads, cfg.head_dim, cfg.eps, cfg.gelu_tanh);
    }
    x
}

/// One post-norm BERT/MiniLM transformer block (spec 050 slice 1): `LayerNorm(x + attn(x))`, then
/// `LayerNorm(y + mlp(y))` (LayerNorms after each residual, unlike SigLIP's pre-norm). Reuses the vision
/// multi-head self-attention (full bidirectional) and shares its weight container ([`VisionBlockWeights`]):
/// `ln1` = attention-output LayerNorm, `ln2` = output LayerNorm, `fc1`/`fc2` = intermediate / output dense.
pub fn trace_bert_block(
    b: &Builder,
    x: Traced,
    w: &VisionBlockWeights,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    gelu_tanh: bool,
) -> Traced {
    let attn = crate::vision::trace_vision_mha(b, x, &w.mha, n_heads, head_dim);
    let x = layernorm(b, b.binary(BinOp::Add, x, attn), w.ln1_w, w.ln1_b, eps);
    let fc1 = linear(b, x, w.fc1_w, Some(w.fc1_b));
    let act = if gelu_tanh {
        gelu(b, fc1)
    } else {
        gelu_erf(b, fc1)
    };
    let fc2 = linear(b, act, w.fc2_w, Some(w.fc2_b));
    layernorm(b, b.binary(BinOp::Add, x, fc2), w.ln2_w, w.ln2_b, eps)
}

/// `tanh(x)` composed as `(e^{2x} - 1) / (e^{2x} + 1)` (a hand composition, not `UnOp::Tanh`). The BERT
/// pooler's activation. Pooler pre-activations are small, so the un-shifted `exp(2x)` form does not overflow.
fn tanh(b: &Builder, x: Traced) -> Traced {
    let two_x = b.binary(BinOp::Add, x, x);
    let e = b.unary(UnOp::Exp, two_x);
    let num = b.binary_scalar(BinOp::Add, e, poot_graph_ir::Scalar::F32(-1.0));
    let den = b.binary_scalar(BinOp::Add, e, poot_graph_ir::Scalar::F32(1.0));
    b.binary(BinOp::Mul, num, b.unary(UnOp::Recip, den))
}

/// BERT input embeddings with per-token segment ids (cross-encoder, spec 050): like
/// [`trace_bert_embeddings`] but gathers `token_type_embeddings[token_types]` per position (query segment 0,
/// document segment 1) instead of the single type-0 row. `token_types` is an `[L]` index parallel to
/// `tokens`. Returns `[1, L, hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn trace_bert_embeddings_typed(
    b: &Builder,
    tokens: Traced,
    token_types: Traced,
    word_emb: Traced,
    pos_emb: Traced,
    type_emb: Traced,
    ln_w: Traced,
    ln_b: Traced,
    seq: usize,
    hidden: usize,
    eps: f32,
) -> Traced {
    let word = b.gather(word_emb, 0, tokens); // [L, h]
    let pos = b.slice(pos_emb, 0, 0, seq); // [L, h] (the first L position rows)
    let types = b.gather(type_emb, 0, token_types); // [L, h] (per-token segment embedding)
    let sum = b.binary(BinOp::Add, b.binary(BinOp::Add, word, pos), types);
    let normed = layernorm(b, sum, ln_w, ln_b, eps);
    b.reshape(normed, vec![1, seq, hidden])
}

/// A cross-encoder reranker (`BertForSequenceClassification`, card 057): the BERT encoder plus the `[CLS]`
/// pooler (`bert.pooler.dense` + tanh) and the single-logit `classifier`. `pooler_w`/`cls_w` are poot's
/// `[in, out]` layout (`[h, h]` and `[h, 1]`).
pub struct CrossEncoderWeights {
    pub enc: BertEncoderWeights,
    pub pooler_w: Traced,
    pub pooler_b: Traced,
    pub cls_w: Traced,
    pub cls_b: Traced,
}

/// Declare a cross-encoder's weights as named graph constants: the BERT encoder ([`declare_bert_constants`])
/// plus `pooler_w`/`pooler_b`/`cls_w`/`cls_b`. The names match `poot_llm::encoder`'s cross-encoder loader.
pub fn declare_cross_encoder_constants(
    b: &Builder,
    cfg: &BertConfig,
) -> (CrossEncoderWeights, Vec<(poot_graph_ir::ValueId, String)>) {
    let (enc, mut bd) = declare_bert_constants(b, cfg);
    let h = cfg.hidden();
    let mut c = |name: String, shape: Vec<usize>| -> Traced {
        let t = b.constant(&name, poot_graph_ir::types::TensorType::f32(shape));
        bd.push((t.id, name));
        t
    };
    let pooler_w = c("pooler_w".into(), vec![h, h]);
    let pooler_b = c("pooler_b".into(), vec![h]);
    let cls_w = c("cls_w".into(), vec![h, 1]);
    let cls_b = c("cls_b".into(), vec![1]);
    (
        CrossEncoderWeights {
            enc,
            pooler_w,
            pooler_b,
            cls_w,
            cls_b,
        },
        bd,
    )
}

/// The full cross-encoder reranker forward (card 057): the BERT encoder over the joint
/// `[CLS] query [SEP] doc [SEP]` sequence, then the `[CLS]` pooler (dense + tanh) and the classifier
/// (linear -> 1). Returns the `[1]` relevance logit (higher = more relevant; the model's default output
/// activation is identity, so this raw logit is the score).
pub fn trace_cross_encoder(
    b: &Builder,
    tokens: Traced,
    token_types: Traced,
    w: &CrossEncoderWeights,
    cfg: &BertConfig,
    seq: usize,
) -> Traced {
    let h = cfg.hidden();
    let mut x = trace_bert_embeddings_typed(
        b,
        tokens,
        token_types,
        w.enc.word_emb,
        w.enc.pos_emb,
        w.enc.type_emb,
        w.enc.emb_ln_w,
        w.enc.emb_ln_b,
        seq,
        h,
        cfg.eps,
    );
    for blk in &w.enc.layers {
        x = trace_bert_block(b, x, blk, cfg.n_heads, cfg.head_dim, cfg.eps, cfg.gelu_tanh);
    }
    // pool the [CLS] token (position 0), then the pooler dense+tanh and the single-logit classifier.
    let cls = b.slice(x, 1, 0, 1); // [1, 1, h]
    let cls = b.reshape(cls, vec![1, h]);
    let pooled = tanh(b, linear(b, cls, w.pooler_w, Some(w.pooler_b))); // [1, h]
    let logit = linear(b, pooled, w.cls_w, Some(w.cls_b)); // [1, 1]
    b.reshape(logit, vec![1])
}

/// Batched BERT typed embeddings (card 091): `tokens`/`token_types` are FLAT `[batch*seq]` indices, row-major
/// over (batch, position). Same math as [`trace_bert_embeddings_typed`] but gathers flat then reshapes to
/// `[batch, seq, hidden]`; the position rows `[seq, hidden]` broadcast over the batch.
#[allow(clippy::too_many_arguments)]
pub fn trace_bert_embeddings_typed_batched(
    b: &Builder,
    tokens: Traced,
    token_types: Traced,
    word_emb: Traced,
    pos_emb: Traced,
    type_emb: Traced,
    ln_w: Traced,
    ln_b: Traced,
    batch: usize,
    seq: usize,
    hidden: usize,
    eps: f32,
) -> Traced {
    let word = b.reshape(b.gather(word_emb, 0, tokens), vec![batch, seq, hidden]);
    let types = b.reshape(b.gather(type_emb, 0, token_types), vec![batch, seq, hidden]);
    let pos = b.reshape(b.slice(pos_emb, 0, 0, seq), vec![1, seq, hidden]); // broadcasts over batch
    let sum = b.binary(BinOp::Add, b.binary(BinOp::Add, word, pos), types);
    layernorm(b, sum, ln_w, ln_b, eps) // [batch, seq, hidden]
}

/// Batched, key-padding-masked BERT self-attention (card 091): `x` is `[batch, seq, hidden]`, `mask` is the
/// additive key mask `[batch, 1, 1, seq]` (0 for a valid key, large-negative for a padded key, broadcast over
/// heads and query positions). Same composition as [`trace_bert_block`]'s MHA, generalized to a leading batch
/// dim. Returns `[batch, seq, hidden]`.
fn trace_bert_mha_batched(
    b: &Builder,
    x: Traced,
    w: &crate::vision::MhaWeights,
    n_heads: usize,
    head_dim: usize,
    mask: Traced,
) -> Traced {
    let sh = &b.aval(x).shape;
    let (batch, seq) = (sh[0], sh[1]);
    let hidden = n_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head = |proj: Traced| {
        let r = b.reshape(proj, vec![batch, seq, n_heads, head_dim]);
        b.transpose(r, vec![0, 2, 1, 3]) // [batch, n_heads, seq, head_dim]
    };
    let q = head(linear(b, x, w.wq, Some(w.bq)));
    let k = head(linear(b, x, w.wk, Some(w.bk)));
    let v = head(linear(b, x, w.wv, Some(w.bv)));
    let attn = attention_masked(b, q, k, v, 1, scale, mask); // [batch, n_heads, seq, head_dim]
    let merged = b.transpose(attn, vec![0, 2, 1, 3]);
    let merged = b.reshape(merged, vec![batch, seq, hidden]);
    linear(b, merged, w.wo, Some(w.bo))
}

/// One batched, masked POST-norm BERT block (card 091): like [`trace_bert_block`] but with a leading batch dim
/// and a key-padding mask on the attention.
#[allow(clippy::too_many_arguments)]
fn trace_bert_block_batched(
    b: &Builder,
    x: Traced,
    w: &VisionBlockWeights,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    gelu_tanh: bool,
    mask: Traced,
) -> Traced {
    let attn = trace_bert_mha_batched(b, x, &w.mha, n_heads, head_dim, mask);
    let x = layernorm(b, b.binary(BinOp::Add, x, attn), w.ln1_w, w.ln1_b, eps);
    let fc1 = linear(b, x, w.fc1_w, Some(w.fc1_b));
    let act = if gelu_tanh {
        gelu(b, fc1)
    } else {
        gelu_erf(b, fc1)
    };
    let fc2 = linear(b, act, w.fc2_w, Some(w.fc2_b));
    layernorm(b, b.binary(BinOp::Add, x, fc2), w.ln2_w, w.ln2_b, eps)
}

/// Batched cross-encoder reranker forward (card 091): score `batch` padded (query, doc) pairs in ONE pass.
/// `tokens`/`token_types` are flat `[batch*seq]` indices (each pair right-padded to `seq`); `mask` is the
/// additive key-padding mask `[batch, 1, 1, seq]` (0 for real tokens, large-negative for padding). Returns the
/// `[batch]` relevance logits - equal to running [`trace_cross_encoder`] per pair, since only the `[CLS]`
/// position (which never attends to a padded key) feeds the classifier.
#[allow(clippy::too_many_arguments)]
pub fn trace_cross_encoder_batched(
    b: &Builder,
    tokens: Traced,
    token_types: Traced,
    mask: Traced,
    w: &CrossEncoderWeights,
    cfg: &BertConfig,
    batch: usize,
    seq: usize,
) -> Traced {
    let h = cfg.hidden();
    let mut x = trace_bert_embeddings_typed_batched(
        b,
        tokens,
        token_types,
        w.enc.word_emb,
        w.enc.pos_emb,
        w.enc.type_emb,
        w.enc.emb_ln_w,
        w.enc.emb_ln_b,
        batch,
        seq,
        h,
        cfg.eps,
    );
    for blk in &w.enc.layers {
        x = trace_bert_block_batched(
            b,
            x,
            blk,
            cfg.n_heads,
            cfg.head_dim,
            cfg.eps,
            cfg.gelu_tanh,
            mask,
        );
    }
    let cls = b.reshape(b.slice(x, 1, 0, 1), vec![batch, h]); // [batch, h] (the [CLS] row per pair)
    let pooled = tanh(b, linear(b, cls, w.pooler_w, Some(w.pooler_b))); // [batch, h]
    let logit = linear(b, pooled, w.cls_w, Some(w.cls_b)); // [batch, 1]
    b.reshape(logit, vec![batch])
}
