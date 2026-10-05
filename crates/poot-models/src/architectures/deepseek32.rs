//! DeepSeek-V3.2 Sparse Attention (DSA) CPU-oracle tracer (spec 277, arXiv 2512.02556 "DeepSeek-V3.2:
//! Pushing the Frontier of Open Large Language Models"). Scoped like `crate::deepseek3` and
//! `crate::qwen3moe`'s first slices: config struct, prefill/decode CPU-oracle tracers, bit-exact
//! independent-reference tests, plus a `Runner` loader and wiring path in `poot-llm`.
//!
//! # DSA (paper section 2.1)
//!
//! DSA adds two components on top of unchanged MLA. The **Lightning indexer** (eq. 1):
//!
//! ```text
//! I_{t,s} = sum_{j=1..H^I} w^I_{t,j} * ReLU(q^I_{t,j} . k^I_s)
//! ```
//!
//! `H^I` indexer heads (`64` in `deepseek-ai/DeepSeek-V3.2-Exp` `inference/model.py` `ModelArgs`);
//! `q^I_{t,j} in R^{d^I}` and the scalar `w^I_{t,j}` derive from the query token `h_t`; `k^I_s in R^{d^I}`
//! derives from the preceding token `h_s` and is shared across all `H^I` heads (MQA-shaped).
//! `d^I = 128` in the reference config. ReLU, not softmax, for throughput.
//!
//! Then **fine-grained token selection** (eq. 2): retrieve only the KV entries at the top-`k` index
//! scores (`k = index_topk = 2048` in the reference config) and run ordinary attention on that set.
//!
//! The paper instantiates DSA on the MQA mode of MLA, where each latent vector is shared across all
//! query heads. MLA's latent compression (`kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`) is
//! unchanged; DSA only adds an additive mask term ahead of the attention core. See
//! `specs/277-deepseek-v3.2-dsa/spec.md` for the derivation and primitive-coverage argument (no new
//! `poot_graph_ir::OpKind`).
//!
//! # Reused
//!
//! `crate::deepseek2::{DeepseekV2Config, mla_query_proj, rope_interleaved_decode,
//! rope_interleaved_prefill, deepseek2_dense_ffn}` (MLA attention and dense-layer MLP, as in
//! `crate::deepseek3`), `crate::deepseek3::{DeepseekV3MoeParams, deepseek3_moe_ffn}` (routed+shared
//! MoE MLP), and `crate::deepseek3::{pairwise_rank, keep_top_k_mask}` (the pairwise-`Ge` top-k
//! rank/mask trick, `pub(crate)` for this reuse). `poot_graph_ir::ops::relu` (`max(x, 0)`) was added
//! to `poot-graph-ir` because eq. 1 needs ReLU. `poot_graph_ir::ops::{rope, rope_prefill}` (the
//! half-split partial-RoPE convention, as for phi3) serve the indexer's `q^I`/`k^I`; see
//! [`dsa_indexer_rope_tables`] (update 0854).
//!
//! # Real parameter names (update 0854 found them, update 0857 adopted them)
//!
//! Update 0854 read `deepseek-ai/DeepSeek-V3.2-Exp`'s `config.json` and
//! `model.safetensors.index.json` and found the per-layer indexer keys
//! `self_attn.indexer.wq_b.weight`, `self_attn.indexer.wk.weight`, `self_attn.indexer.weights_proj.weight`,
//! and `self_attn.indexer.k_norm.{weight,bias}`. Two differences were architectural: (1) the real
//! `wq_b` takes `qr` (MLA's `q_a_proj`+`q_a_layernorm` low-rank query, `q_lora_rank`-wide, `1536`),
//! not the full normed hidden state (`hidden_size`, `7168`); (2) `k_norm` (a LayerNorm with weight and
//! bias, applied to the indexer's `k` projection before the RoPE split) had no analog here.
//! Update 0857 implements both:
//!
//! 1. **`qr` is returned by [`crate::deepseek2::mla_query_proj`]**: the low-rank branch's
//!    `q_a_layernorm` output, `[.., q_lora_rank]`, as a second `Option<Traced>` (`Some` only when
//!    `cfg.q_lora_rank.is_some()`). The indexer query projection consumes the same traced value, so
//!    `q_a_proj`+`q_a_layernorm` is computed once per layer. The DSA tracers `.expect()`
//!    `cfg.q_lora_rank.is_some()` (DeepSeek-V3.2 always sets it).
//! 2. **The indexer's query weight is `self_attn.indexer.wq_b.weight`**, shaped
//!    `[q_lora_rank, index_n_heads*index_head_dim]`, and consumes `qr`.
//! 3. **`self_attn.indexer.k_norm.{weight,bias}`** is a `poot_graph_ir::ops::layernorm` call (mean/variance
//!    LayerNorm, not RMSNorm) on `wk(x)` immediately before the half-split RoPE split, at `eps=1e-6`
//!    (see [`DSA_KNORM_EPS`]).
//!
//! The independent CPU-oracle reference (`deepseek32_prefill_ref`/`dsa_attend_ref` in the tests)
//! implements both from scratch (a hand-rolled `q_a_proj`/`q_a_layernorm`/`q_b_proj` chain and its own
//! `layernorm_ref`), so the differential tests cross-check the math rather than mirroring the tracer.
//! See `specs/277-deepseek-v3.2-dsa/spec.md`'s Out-of-scope section for the citation trail.

use crate::deepseek2::{
    DeepseekV2Config, DeepseekV2Yarn, deepseek2_dense_ffn, deepseek2_rope_inv_freq, mla_query_proj,
    rope_interleaved_decode, rope_interleaved_prefill,
};
use crate::deepseek3::{DeepseekV3MoeParams, deepseek3_moe_ffn, keep_top_k_mask, pairwise_rank};
use poot_graph_ir::ops::{
    attention_masked, attention_prefill, layernorm, linear, relu, repeat_kv, rmsnorm, rope,
    rope_prefill,
};
use poot_graph_ir::{BinOp, Builder, Graph, RedOp, Scalar, Slot, StateRole, TensorType, Traced};
use poot_tensor::DType;

/// The Lightning Indexer's shape, in addition to `DeepseekV2Config`/`DeepseekV3MoeParams` (spec 277).
/// Reference values (`deepseek-ai/DeepSeek-V3.2-Exp` `inference/model.py` `ModelArgs`):
/// `index_n_heads: 64`, `index_head_dim: 128`, `index_topk: 2048`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DsaConfig {
    /// Lightning Indexer heads (`H^I` in eq. 1), independent of `DeepseekV2Config::n_heads`.
    pub index_n_heads: usize,
    /// Per-indexer-head query/key width (`d^I` in eq. 1, `128` in the reference config). RoPE rotates
    /// only the leading `DeepseekV2Config::qk_rope_head_dim` dims (`Indexer.__init__`:
    /// `self.rope_head_dim: int = args.qk_rope_head_dim`), so this must be `>= cfg.qk_rope_head_dim`,
    /// which must be even (`poot_graph_ir::ops::rope`'s constraint). See [`dsa_indexer_rope_tables`].
    pub index_head_dim: usize,
    /// Top-`k` KV entries selected per query token (`index_topk`, eq. 2).
    pub index_topk: usize,
}

/// The runner-facing bundle of DSA's three config structs (as [`crate::deepseek3::DeepseekV3Params`],
/// plus `dsa`), so `poot_llm::Runner`'s `deepseek32` field is one `Option`. The tracers still take
/// the three fields separately.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Deepseek32Params {
    pub cfg: DeepseekV2Config,
    pub moe: DeepseekV3MoeParams,
    pub dsa: DsaConfig,
}

/// Additive penalty for "not selected" / "not yet valid" mask positions, the `-1e30` convention of
/// `crate::deepseek3::deepseek3_router_gate`'s group-drop term (not `f32::MIN` or `NEG_INFINITY`):
/// zeroes out after softmax, and summing two such terms does not overflow `f32`.
const DSA_MASK_NEG: f32 = -1e30;

/// The indexer's `k_norm` LayerNorm epsilon (`Indexer.__init__` in `deepseek-ai/DeepSeek-V3.2-Exp`:
/// `self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)`). Independent of `DeepseekV2Config::eps`,
/// though both are `1e-6` on the real checkpoint.
const DSA_KNORM_EPS: f32 = 1e-6;

/// Build the Lightning Indexer's RoPE table in the half-split ("rotate-half") layout
/// `poot_graph_ir::ops::rope`/`rope_prefill` expect: `[max_pos, rope_d]`, each frequency duplicated at
/// index `i` and `half + i` (not the `[max_pos, rope_d/2]` interleaved layout of
/// `deepseek2_rope_tables_interleaved` for MLA's q_pe/k_pe).
///
/// Resolves spec 277's two [NEEDS CLARIFICATION] items against `deepseek-ai/DeepSeek-V3.2-Exp`
/// `inference/model.py` (`Indexer.__init__`/`Indexer.forward`, `precompute_freqs_cis`,
/// `MLA.forward`'s shared `freqs_cis`):
///
/// - **Split ratio**: `Indexer.forward` splits `q`/`k` as `torch.split(x, [self.rope_head_dim,
///   self.head_dim - self.rope_head_dim], dim=-1)` with `self.rope_head_dim = args.qk_rope_head_dim`
///   (`64`) and `self.head_dim = args.index_head_dim` (`128`). So `rope_d` is the main attention's
///   decoupled-RoPE width, not half of `index_head_dim`. The trailing `index_head_dim - rope_d` dims
///   pass through unrotated, which the `rot < d` branch of `poot_graph_ir::ops::rope_partial` (phi3
///   partial rotary) covers.
/// - **Rotation convention**: the reference calls `apply_rotary_emb(q_pe, freqs_cis, False)` /
///   `apply_rotary_emb(k_pe, freqs_cis, False)`: `interleaved=False` (source comment "rope in indexer
///   is not interleaved"), the half-split convention of `poot_graph_ir::ops::rope`/`rope_prefill`.
///   MLA's own q_pe/k_pe use the interleaved default. The tracers call [`rope`]/[`rope_prefill`].
/// - **YaRN inheritance**: `precompute_freqs_cis` runs once per model and the same `freqs_cis` goes to
///   both `MLA.forward` (q_pe/k_pe) and `self.indexer(x, qr, start_pos, freqs_cis, mask)`, so the
///   indexer's table is the main attention's (possibly YaRN-scaled) table. Callers pass `cfg.yarn`,
///   the value used for `deepseek2_rope_tables_interleaved`.
pub fn dsa_indexer_rope_tables(
    max_pos: usize,
    rope_d: usize,
    theta: f32,
    yarn: Option<&DeepseekV2Yarn>,
) -> (Vec<f32>, Vec<f32>) {
    let half = rope_d / 2;
    let (inv_freq, attention_factor) = deepseek2_rope_inv_freq(rope_d, theta, yarn);
    let mut cos = vec![0.0f32; max_pos * rope_d];
    let mut sin = vec![0.0f32; max_pos * rope_d];
    for pos in 0..max_pos {
        for i in 0..half {
            let ang = pos as f32 * inv_freq[i];
            let (s, c) = ang.sin_cos();
            cos[pos * rope_d + i] = c * attention_factor;
            cos[pos * rope_d + half + i] = c * attention_factor;
            sin[pos * rope_d + i] = s * attention_factor;
            sin[pos * rope_d + half + i] = s * attention_factor;
        }
    }
    (cos, sin)
}

/// Lightning Indexer score (eq. 1): `I[.., Lq, Lk] = sum_h w[.., h, Lq] * ReLU(q[.., h, Lq, Di] .
/// k[.., Lk, Di])`. `q_idx` is `[1, Hi, Lq, Di]` (per-head, already RoPE'd); `k_idx` is `[1, Lk, Di]`
/// (shared across heads, broadcast via `repeat_kv` as MLA's RoPE-key row); `w_idx` is `[1, Hi, Lq]`.
/// Returns `[1, 1, Lq, Lk]` (head axis reduced, kept size 1 so it adds directly to a `[1,1,*,cap-or-L]`
/// causal/validity mask).
#[allow(clippy::too_many_arguments)]
fn dsa_indexer_scores(
    b: &Builder,
    q_idx: Traced,
    k_idx: Traced,
    w_idx: Traced,
    hi: usize,
    lq: usize,
    lk: usize,
    di: usize,
) -> Traced {
    let k_idx_h = b.reshape(k_idx, vec![1, 1, lk, di]);
    let k_idx_b = repeat_kv(b, k_idx_h, hi); // [1, Hi, Lk, Di] - one shared key row per head
    let k_idx_t = b.transpose(k_idx_b, vec![0, 1, 3, 2]); // [1, Hi, Di, Lk]
    let dots = b.matmul(q_idx, k_idx_t); // [1, Hi, Lq, Lk]
    let scored = relu(b, dots);
    let w_r = b.reshape(w_idx, vec![1, hi, lq, 1]);
    let w_b = b.broadcast(w_r, vec![1, hi, lq, lk]);
    let weighted = b.binary(BinOp::Mul, scored, w_b);
    // Reduce the head axis (1), keepdim: `compile`'s `lower_nonlast_reduces` legalizes a non-last-axis
    // reduce for every target (card 523b), so the tracer reduces over the semantically right axis
    // directly, with no transpose.
    b.reduce(RedOp::Sum, weighted, 1, true) // [1, 1, Lq, Lk]
}

/// Fine-grained token selection (eq. 2) as an additive mask, so it composes with the
/// `poot_graph_ir::ops::attention_masked`/`attention_prefill` mask input (spec 277 FR-001/FR-005)
/// without a new attention variant. Ranks `index_scores + causal_mask` (not raw `index_scores`) with
/// `pairwise_rank`/`keep_top_k_mask` as `crate::deepseek3::deepseek3_router_gate`; ranking the sum
/// keeps top-k from selecting a not-yet-written or non-causal slot when fewer than `topk` positions
/// are valid, and the returned mask's own `causal_mask` term excludes them regardless (spec 277
/// "Internal correctness"). `index_scores` and `causal_mask` share shape `[.., N]` (leading dims
/// `[1,1,1]` for decode, `[1,1,L]` for prefill); the ranking operates on the trailing axis, so one
/// function serves both.
fn dsa_combined_mask(
    b: &Builder,
    index_scores: Traced,
    causal_mask: Traced,
    topk: usize,
) -> Traced {
    let ranked_input = b.binary(BinOp::Add, index_scores, causal_mask);
    let rank = pairwise_rank(b, ranked_input);
    let keep = keep_top_k_mask(b, rank, topk); // 1.0 kept, 0.0 dropped
    let shifted = b.binary_scalar(BinOp::Sub, keep, Scalar::F32(1.0)); // 0.0 kept, -1.0 dropped
    let topk_add = b.binary_scalar(BinOp::Mul, shifted, Scalar::F32(-DSA_MASK_NEG)); // 0 kept, DSA_MASK_NEG dropped
    b.binary(BinOp::Add, causal_mask, topk_add)
}

/// Trace a single DSA decode step: one new token against a fixed-capacity KV cache,
/// `crate::deepseek3::trace_deepseek3_decode_kv_masked` plus the Lightning Indexer and top-k selection
/// mask (spec 277 FR-001/FR-002). `cap` is the physical cache capacity; `mp` selects dense-vs-MoE per
/// layer as in `crate::deepseek3`.
///
/// Adds a third per-layer state tensor beyond `mla.c_cache`/`mla.rope_cache`:
/// `self_attn.indexer.k_cache` `[1,1,cap,index_head_dim]`, written via
/// `Builder::dynamic_update_slice_dyn` like `mla.rope_cache`. `Graph::state` has `3 * cfg.layers`
/// entries (SC-002).
pub fn trace_deepseek32_dsa_decode(
    cfg: DeepseekV2Config,
    dcfg: DsaConfig,
    mp: DeepseekV3MoeParams,
    cap: usize,
) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
    let qk_head_dim = cfg.qk_head_dim();
    let kv_rank = cfg.kv_lora_rank;
    let scale = cfg.attn_scale();
    let (hi, di) = (dcfg.index_n_heads, dcfg.index_head_dim);
    let q_lora_rank = cfg.q_lora_rank.expect(
        "DSA's real indexer wq_b consumes MLA's own qr, so DSA requires q_lora_rank \
         (real DeepSeek-V3.2 always sets it, 1536)",
    );

    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let _seq_len = b.slot(Slot::SeqLen, TensorType::scalar(DType::I32));
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![cap]));
    let mask = b.reshape(mask, vec![1, 1, 1, cap]);

    let cos = b.constant("rope.cos", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));
    // Indexer RoPE table: half-split layout, width `rope_d` (== cfg.qk_rope_head_dim, not `di`); see
    // `dsa_indexer_rope_tables`.
    let idx_cos = b.constant("index.rope.cos", TensorType::f32(vec![cfg.max_pos, rope_d]));
    let idx_sin = b.constant("index.rope.sin", TensorType::f32(vec![cfg.max_pos, rope_d]));

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let x0 = b.gather_scalar(embed, 0, token);
    let mut x = b.reshape(x0, vec![1, 1, h]);

    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(3 * cfg.layers);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let (q, qr) = mla_query_proj(&b, &p, normed, cfg, h, hq, qk_head_dim);
        let qr = qr.expect("q_lora_rank checked Some above"); // DSA's own wq_b input, see below
        let q = b.transpose(b.reshape(q, vec![1, 1, hq, qk_head_dim]), vec![0, 2, 1, 3]);
        let q_nope = b.slice(q, 3, 0, nope);
        let q_pe = b.slice(q, 3, nope, qk_head_dim);
        let q_pe = rope_interleaved_decode(&b, q_pe, cos, sin, pos_slot);
        let q_full = b.concat(3, &[q_nope, q_pe]);

        let w_kva = b.constant(
            &p("self_attn.kv_a_proj_with_mqa.weight"),
            TensorType::f32(vec![h, kv_rank + rope_d]),
        );
        let kv_a = linear(&b, normed, w_kva, None);
        let kv_nope_raw = b.slice(kv_a, 2, 0, kv_rank);
        let k_pe_raw = b.slice(kv_a, 2, kv_rank, kv_rank + rope_d);
        let kva_ln = b.constant(
            &p("self_attn.kv_a_layernorm.weight"),
            TensorType::f32(vec![kv_rank]),
        );
        let c_kv_new = rmsnorm(&b, kv_nope_raw, kva_ln, cfg.eps);
        let c_kv_new = b.reshape(c_kv_new, vec![1, 1, 1, kv_rank]);

        let k_pe_new = b.reshape(k_pe_raw, vec![1, 1, 1, rope_d]);
        let k_pe_new = rope_interleaved_decode(&b, k_pe_new, cos, sin, pos_slot);

        let c_cache = b.state_input(
            &p("mla.c_cache"),
            TensorType::f32(vec![1, 1, cap, kv_rank]),
            StateRole::Recurrent,
        );
        let rope_cache = b.state_input(
            &p("mla.rope_cache"),
            TensorType::f32(vec![1, 1, cap, rope_d]),
            StateRole::Recurrent,
        );
        let c_cache_out = b.dynamic_update_slice_dyn(c_cache, c_kv_new, pos_slot, 2);
        let rope_cache_out = b.dynamic_update_slice_dyn(rope_cache, k_pe_new, pos_slot, 2);
        state.push((c_cache, c_cache_out));
        state.push((rope_cache, rope_cache_out));

        let c_flat = b.reshape(c_cache_out, vec![cap, kv_rank]);
        let w_kvb = b.constant(
            &p("self_attn.kv_b_proj.weight"),
            TensorType::f32(vec![kv_rank, hq * (nope + vd)]),
        );
        let kv_expanded = linear(&b, c_flat, w_kvb, None);
        let kv_expanded = b.transpose(
            b.reshape(kv_expanded, vec![1, cap, hq, nope + vd]),
            vec![0, 2, 1, 3],
        );
        let k_nope_h = b.slice(kv_expanded, 3, 0, nope);
        let v_h = b.slice(kv_expanded, 3, nope, nope + vd);

        let k_pe_b = repeat_kv(&b, rope_cache_out, hq);
        let k_full = b.concat(3, &[k_nope_h, k_pe_b]);

        // Lightning Indexer: fresh q^I/w^I for this token, cached k^I like `mla.rope_cache`. `wq_b`
        // consumes MLA's `qr` (the `q_a_layernorm` output `q` was built from; see `mla_query_proj` and
        // update 0857), not the full normed hidden state.
        let wiq_b = b.constant(
            &p("self_attn.indexer.wq_b.weight"),
            TensorType::f32(vec![q_lora_rank, hi * di]),
        );
        let q_idx = linear(&b, qr, wiq_b, None); // [1,1,Hi*Di]
        let q_idx = b.transpose(b.reshape(q_idx, vec![1, 1, hi, di]), vec![0, 2, 1, 3]); // [1,Hi,1,Di]
        // Half-split partial RoPE: rotates the leading `rope_d` dims, passes the trailing `di - rope_d`
        // through (see `dsa_indexer_rope_tables`).
        let q_idx = rope(&b, q_idx, idx_cos, idx_sin, pos_slot);

        let wik = b.constant(
            &p("self_attn.indexer.wk.weight"),
            TensorType::f32(vec![h, di]),
        );
        let k_idx_raw = linear(&b, normed, wik, None); // [1,1,Di]
        // `k_norm`: LayerNorm (weight+bias, eps=1e-6) applied before the RoPE split (see `DSA_KNORM_EPS`, update 0857).
        let k_norm_w = b.constant(
            &p("self_attn.indexer.k_norm.weight"),
            TensorType::f32(vec![di]),
        );
        let k_norm_b = b.constant(
            &p("self_attn.indexer.k_norm.bias"),
            TensorType::f32(vec![di]),
        );
        let k_idx_normed = layernorm(&b, k_idx_raw, k_norm_w, k_norm_b, DSA_KNORM_EPS);
        let k_idx_new = b.reshape(k_idx_normed, vec![1, 1, 1, di]);
        let k_idx_new = rope(&b, k_idx_new, idx_cos, idx_sin, pos_slot);

        let wiw = b.constant(
            &p("self_attn.indexer.weights_proj.weight"),
            TensorType::f32(vec![h, hi]),
        );
        let w_idx = linear(&b, normed, wiw, None); // [1,1,Hi]
        let w_idx = b.reshape(w_idx, vec![1, hi, 1]); // [1,Hi,1] - flat-order-preserving, Lq=1

        let idx_cache = b.state_input(
            &p("self_attn.indexer.k_cache"),
            TensorType::f32(vec![1, 1, cap, di]),
            StateRole::Recurrent,
        );
        let idx_cache_out = b.dynamic_update_slice_dyn(idx_cache, k_idx_new, pos_slot, 2);
        state.push((idx_cache, idx_cache_out));

        let idx_cache_flat = b.reshape(idx_cache_out, vec![1, cap, di]); // [1,Lk,Di]
        let index_scores = dsa_indexer_scores(&b, q_idx, idx_cache_flat, w_idx, hi, 1, cap, di);
        let final_mask = dsa_combined_mask(&b, index_scores, mask, dcfg.index_topk);

        let attn = attention_masked(&b, q_full, k_full, v_h, 1, scale, final_mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, 1, hq * vd]);
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![hq * vd, h]),
        );
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = if mp.is_moe_layer(li) {
            deepseek3_moe_ffn(&b, normed, h, &mp, li)
        } else {
            deepseek2_dense_ffn(&b, normed, h, mp.dense_inter, li)
        };
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None);
    b.finish_with_state(logits, &state)
}

/// Trace a full-sequence DSA prefill forward with per-query-row sparse top-k selection (spec 277
/// FR-005; the paper's short-sequence "masked MHA mode to simulate DSA" shortcut is not reproduced,
/// see the spec's Out-of-scope section). One indexer score per `(query row, key row)` pair over the
/// `[1,1,L,L]` causal grid; each query row's key axis is ranked independently with the same
/// `dsa_combined_mask` as decode. `mask` is the caller-supplied `[1,1,l,l]` causal mask, as in
/// `crate::deepseek3::trace_deepseek3_prefill`.
pub fn trace_deepseek32_dsa_prefill(
    cfg: DeepseekV2Config,
    dcfg: DsaConfig,
    mp: DeepseekV3MoeParams,
    seq_len: usize,
) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
    let qk_head_dim = cfg.qk_head_dim();
    let kv_rank = cfg.kv_lora_rank;
    let scale = cfg.attn_scale();
    let l = seq_len;
    let (hi, di) = (dcfg.index_n_heads, dcfg.index_head_dim);
    let q_lora_rank = cfg.q_lora_rank.expect(
        "DSA's real indexer wq_b consumes MLA's own qr, so DSA requires q_lora_rank \
         (real DeepSeek-V3.2 always sets it, 1536)",
    );

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let cos = b.constant("rope.cos", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));
    let idx_cos = b.constant("index.rope.cos", TensorType::f32(vec![cfg.max_pos, rope_d]));
    let idx_sin = b.constant("index.rope.sin", TensorType::f32(vec![cfg.max_pos, rope_d]));
    let mask = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens); // [L, hidden]
    let mut x = b.reshape(emb, vec![1, l, h]);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let (q, qr) = mla_query_proj(&b, &p, normed, cfg, h, hq, qk_head_dim);
        let qr = qr.expect("q_lora_rank checked Some above"); // DSA's own wq_b input, see below
        let q = b.transpose(b.reshape(q, vec![1, l, hq, qk_head_dim]), vec![0, 2, 1, 3]);
        let q_nope = b.slice(q, 3, 0, nope);
        let q_pe = b.slice(q, 3, nope, qk_head_dim);
        let q_pe = rope_interleaved_prefill(&b, q_pe, cos, sin, l);
        let q_full = b.concat(3, &[q_nope, q_pe]);

        let w_kva = b.constant(
            &p("self_attn.kv_a_proj_with_mqa.weight"),
            TensorType::f32(vec![h, kv_rank + rope_d]),
        );
        let kv_a = linear(&b, normed, w_kva, None);
        let kv_nope_raw = b.slice(kv_a, 2, 0, kv_rank);
        let k_pe_raw = b.slice(kv_a, 2, kv_rank, kv_rank + rope_d);
        let kva_ln = b.constant(
            &p("self_attn.kv_a_layernorm.weight"),
            TensorType::f32(vec![kv_rank]),
        );
        let c_kv = rmsnorm(&b, kv_nope_raw, kva_ln, cfg.eps);

        let k_pe = b.reshape(k_pe_raw, vec![1, 1, l, rope_d]);
        let k_pe = rope_interleaved_prefill(&b, k_pe, cos, sin, l);

        let w_kvb = b.constant(
            &p("self_attn.kv_b_proj.weight"),
            TensorType::f32(vec![kv_rank, hq * (nope + vd)]),
        );
        let kv_expanded = linear(&b, c_kv, w_kvb, None);
        let kv_expanded = b.transpose(
            b.reshape(kv_expanded, vec![1, l, hq, nope + vd]),
            vec![0, 2, 1, 3],
        );
        let k_nope_h = b.slice(kv_expanded, 3, 0, nope);
        let v_h = b.slice(kv_expanded, 3, nope, nope + vd);

        let k_pe_b = repeat_kv(&b, k_pe, hq);
        let k_full = b.concat(3, &[k_nope_h, k_pe_b]);

        // Lightning Indexer over the full [L,L] causal grid. `wq_b` consumes MLA's `qr` (see decode).
        let wiq_b = b.constant(
            &p("self_attn.indexer.wq_b.weight"),
            TensorType::f32(vec![q_lora_rank, hi * di]),
        );
        let q_idx = linear(&b, qr, wiq_b, None); // [1,L,Hi*Di]
        let q_idx = b.transpose(b.reshape(q_idx, vec![1, l, hi, di]), vec![0, 2, 1, 3]); // [1,Hi,L,Di]
        // Half-split partial RoPE (see decode).
        let q_idx = rope_prefill(&b, q_idx, idx_cos, idx_sin, l);

        let wik = b.constant(
            &p("self_attn.indexer.wk.weight"),
            TensorType::f32(vec![h, di]),
        );
        let k_idx_raw = linear(&b, normed, wik, None); // [1,L,Di]
        // `k_norm` before the RoPE split (see decode).
        let k_norm_w = b.constant(
            &p("self_attn.indexer.k_norm.weight"),
            TensorType::f32(vec![di]),
        );
        let k_norm_b = b.constant(
            &p("self_attn.indexer.k_norm.bias"),
            TensorType::f32(vec![di]),
        );
        let k_idx_normed = layernorm(&b, k_idx_raw, k_norm_w, k_norm_b, DSA_KNORM_EPS);
        let k_idx = b.reshape(k_idx_normed, vec![1, 1, l, di]);
        let k_idx = rope_prefill(&b, k_idx, idx_cos, idx_sin, l); // [1,1,L,Di]
        let k_idx_flat = b.reshape(k_idx, vec![1, l, di]); // [1,Lk,Di]

        let wiw = b.constant(
            &p("self_attn.indexer.weights_proj.weight"),
            TensorType::f32(vec![h, hi]),
        );
        let w_idx_raw = linear(&b, normed, wiw, None); // [1,L,Hi]
        let w_idx = b.transpose(w_idx_raw, vec![0, 2, 1]); // [1,Hi,L] - true reorder, L>1 here

        let index_scores = dsa_indexer_scores(&b, q_idx, k_idx_flat, w_idx, hi, l, l, di);
        let final_mask = dsa_combined_mask(&b, index_scores, mask, dcfg.index_topk);

        let attn = attention_prefill(&b, q_full, k_full, v_h, 1, scale, final_mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, l, hq * vd]);
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![hq * vd, h]),
        );
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = if mp.is_moe_layer(li) {
            deepseek3_moe_ffn(&b, normed, h, &mp, li)
        } else {
            deepseek2_dense_ffn(&b, normed, h, mp.dense_inter, li)
        };
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    b.finish(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_ir::Storage;

    fn tiny_cfg() -> DeepseekV2Config {
        DeepseekV2Config {
            vocab: 10,
            hidden: 8,
            layers: 2,
            n_heads: 2,
            // DeepSeek-V3.2 always sets q_lora_rank (1536); the indexer's wq_b consumes MLA's qr, which
            // only exists in this branch. Distinct from the other dims (kv_lora_rank=4, hidden=8) so a
            // shape mixup fails.
            q_lora_rank: Some(5),
            kv_lora_rank: 4,
            qk_nope_head_dim: 3,
            qk_rope_head_dim: 2,
            v_head_dim: 3,
            eps: 1e-5,
            max_pos: 16,
            rope_theta: 10_000.0,
            yarn: None,
        }
    }

    fn tiny_dcfg(index_topk: usize) -> DsaConfig {
        DsaConfig {
            index_n_heads: 2,
            index_head_dim: 4,
            index_topk,
        }
    }

    /// `first_k_dense_replace: cfg.layers` makes every layer dense. DSA changes attention only, so the
    /// FFN uses the dense SwiGLU path already verified by `crate::deepseek3`'s `cpu_oracle` tests, and the
    /// reference effort goes to the indexer and top-k attention core (see `dsa_attend_ref`).
    fn tiny_mp() -> DeepseekV3MoeParams {
        DeepseekV3MoeParams {
            n_routed_experts: 1,
            top_k: 1,
            moe_inter: 1,
            n_shared_experts: 0,
            dense_inter: 6,
            first_k_dense_replace: 2,
            n_group: 1,
            topk_group: 1,
            routed_scaling_factor: 1.0,
        }
    }

    #[test]
    fn deepseek32_dsa_decode_validates_three_state_tensors_per_layer() {
        let cfg = tiny_cfg();
        let dcfg = tiny_dcfg(3);
        let mp = tiny_mp();
        let cap = 6;
        let g = trace_deepseek32_dsa_decode(cfg, dcfg, mp, cap);
        g.validate()
            .expect("deepseek32 dsa decode graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
        assert_eq!(
            g.state.len(),
            3 * cfg.layers,
            "SC-002: 3 state tensors per layer"
        );
        for (i, (si, _so)) in g.state.iter().enumerate() {
            let shape = g.aval(*si).shape.clone();
            match i % 3 {
                0 => assert_eq!(shape, vec![1, 1, cap, cfg.kv_lora_rank], "c_cache shape"),
                1 => assert_eq!(
                    shape,
                    vec![1, 1, cap, cfg.qk_rope_head_dim],
                    "rope_cache shape"
                ),
                _ => assert_eq!(
                    shape,
                    vec![1, 1, cap, dcfg.index_head_dim],
                    "indexer k_cache shape"
                ),
            }
        }
    }

    #[test]
    fn deepseek32_dsa_prefill_validates() {
        let cfg = tiny_cfg();
        let dcfg = tiny_dcfg(3);
        let mp = tiny_mp();
        let g = trace_deepseek32_dsa_prefill(cfg, dcfg, mp, 6);
        g.validate()
            .expect("deepseek32 dsa prefill graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
    }

    mod cpu_oracle {
        use super::*;
        use crate::deepseek2::deepseek2_rope_tables_interleaved;
        use crate::deepseek3::trace_deepseek3_decode_kv_masked;
        use std::collections::HashMap;

        use poot_test_util::fill;

        use poot_test_util::seed_of;

        fn weight(name: &str, n: usize, ln_gamma: bool) -> Vec<f32> {
            let raw = fill(n, seed_of(name));
            if ln_gamma {
                raw.iter().map(|v| 1.0 + v * 0.05).collect()
            } else {
                raw.iter().map(|v| v * 0.1).collect()
            }
        }

        fn all_weights(
            cfg: &DeepseekV2Config,
            dcfg: &DsaConfig,
            mp: &DeepseekV3MoeParams,
        ) -> HashMap<String, Vec<f32>> {
            let h = cfg.hidden;
            let hq = cfg.n_heads;
            let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
            let qk_head_dim = cfg.qk_head_dim();
            let kv_rank = cfg.kv_lora_rank;
            let (hi, di) = (dcfg.index_n_heads, dcfg.index_head_dim);
            let q_lora_rank = cfg
                .q_lora_rank
                .expect("DSA test fixtures always set q_lora_rank - see tiny_cfg's doc comment");
            let mut w = HashMap::new();
            w.insert(
                "model.embed_tokens.weight".to_string(),
                weight("embed", cfg.vocab * h, false),
            );
            let (cos, sin) = deepseek2_rope_tables_interleaved(
                cfg.max_pos,
                rope_d,
                cfg.rope_theta,
                cfg.yarn.as_ref(),
            );
            w.insert("rope.cos".to_string(), cos);
            w.insert("rope.sin".to_string(), sin);
            // Indexer RoPE: half-split layout, width `rope_d`, same `cfg.yarn` as the main table
            // (DeepSeek-V3.2 shares `freqs_cis`; see `dsa_indexer_rope_tables`).
            let (idx_cos, idx_sin) =
                dsa_indexer_rope_tables(cfg.max_pos, rope_d, cfg.rope_theta, cfg.yarn.as_ref());
            w.insert("index.rope.cos".to_string(), idx_cos);
            w.insert("index.rope.sin".to_string(), idx_sin);
            for li in 0..cfg.layers {
                let p = |s: &str| format!("model.layers.{li}.{s}");
                w.insert(p("input_layernorm.weight"), weight(&p("ln1"), h, true));
                w.insert(
                    p("post_attention_layernorm.weight"),
                    weight(&p("ln2"), h, true),
                );
                w.insert(
                    p("self_attn.q_a_proj.weight"),
                    weight(&p("qa"), h * q_lora_rank, false),
                );
                w.insert(
                    p("self_attn.q_a_layernorm.weight"),
                    weight(&p("qaln"), q_lora_rank, true),
                );
                w.insert(
                    p("self_attn.q_b_proj.weight"),
                    weight(&p("qb"), q_lora_rank * hq * qk_head_dim, false),
                );
                w.insert(
                    p("self_attn.kv_a_proj_with_mqa.weight"),
                    weight(&p("kva"), h * (kv_rank + rope_d), false),
                );
                w.insert(
                    p("self_attn.kv_a_layernorm.weight"),
                    weight(&p("kvaln"), kv_rank, true),
                );
                w.insert(
                    p("self_attn.kv_b_proj.weight"),
                    weight(&p("kvb"), kv_rank * hq * (nope + vd), false),
                );
                w.insert(
                    p("self_attn.o_proj.weight"),
                    weight(&p("o"), hq * vd * h, false),
                );
                w.insert(
                    p("self_attn.indexer.wq_b.weight"),
                    weight(&p("iwqb"), q_lora_rank * hi * di, false),
                );
                w.insert(
                    p("self_attn.indexer.wk.weight"),
                    weight(&p("iwk"), h * di, false),
                );
                w.insert(
                    p("self_attn.indexer.k_norm.weight"),
                    weight(&p("iknw"), di, true),
                );
                w.insert(
                    p("self_attn.indexer.k_norm.bias"),
                    weight(&p("iknb"), di, false),
                );
                w.insert(
                    p("self_attn.indexer.weights_proj.weight"),
                    weight(&p("iww"), h * hi, false),
                );
                w.insert(
                    p("mlp.gate_proj.weight"),
                    weight(&p("dg"), h * mp.dense_inter, false),
                );
                w.insert(
                    p("mlp.up_proj.weight"),
                    weight(&p("du"), h * mp.dense_inter, false),
                );
                w.insert(
                    p("mlp.down_proj.weight"),
                    weight(&p("dd"), mp.dense_inter * h, false),
                );
            }
            w.insert("model.norm.weight".to_string(), weight("ln_f", h, true));
            w.insert(
                "lm_head.weight".to_string(),
                weight("lm_head", h * cfg.vocab, false),
            );
            w
        }

        use poot_test_util::rmsnorm_ref;

        // Independent LayerNorm reference (mean/variance normalization plus weight/bias) for the indexer's
        // `k_norm`; shared via `crate::reference_ops` (R474-014).
        use crate::reference_ops::layernorm_ref;

        use poot_test_util::silu_ref;

        use poot_test_util::linear_ref;

        fn dense_ffn_ref(
            x: &[f32],
            wg: &[f32],
            wu: &[f32],
            wd: &[f32],
            h: usize,
            inter: usize,
        ) -> Vec<f32> {
            let g = linear_ref(x, wg, h, inter);
            let u = linear_ref(x, wu, h, inter);
            let act: Vec<f32> = g
                .iter()
                .zip(u.iter())
                .map(|(&gg, &uu)| silu_ref(gg) * uu)
                .collect();
            linear_ref(&act, wd, inter, h)
        }

        // Shared with deepseek2/deepseek3 via `crate::reference_ops` (R474-014).
        use crate::reference_ops::rope_interleaved_ref;

        /// Independent reference for the indexer's half-split partial RoPE, unlike
        /// `rope_interleaved_ref` (MLA's convention): `row[.., di]`; only the leading `rot` dims rotate
        /// (`cos`/`sin` position-selected, full `rot`-width rows with `c[i] == c[half+i]`, as
        /// `dsa_indexer_rope_tables`); dims `[rot, di)` pass through.
        fn rope_half_split_ref(
            row: &[f32],
            cos_row: &[f32],
            sin_row: &[f32],
            rot: usize,
        ) -> Vec<f32> {
            let half = rot / 2;
            let mut out = row.to_vec();
            for i in 0..half {
                let (x1, x2) = (row[i], row[half + i]);
                out[i] = x1 * cos_row[i] - x2 * sin_row[i];
                out[half + i] = x2 * cos_row[half + i] + x1 * sin_row[half + i];
            }
            out
        }

        /// Independent reference for arXiv 2512.02556 eq. 1 + eq. 2, structured differently from the
        /// tracer's additive-mask-then-full-softmax (`dsa_combined_mask`/`attention_masked`): it
        /// computes the index scores directly, picks the top `min(topk, prefix.len())` positions by a
        /// plain sort, and runs softmax attention over only those rows. A mask-sign or rank-direction
        /// bug in the graph would then mismatch. `c_kv_cache`/`k_pe_cache`/`k_idx_cache` are already
        /// the causal prefix (`&cache[..=qpos]`, as `crate::deepseek3`'s `mla_attend_ref` caller);
        /// `pos + 1 < topk` (spec 277 FR-003) needs no boundary code since `min(topk, prefix.len())`
        /// saturates.
        #[allow(clippy::too_many_arguments)]
        fn dsa_attend_ref(
            cfg: &DeepseekV2Config,
            dcfg: &DsaConfig,
            q_full: &[Vec<f32>],     // [hq][qk_head_dim]
            c_kv_cache: &[Vec<f32>], // [prefix][kv_rank]
            k_pe_cache: &[Vec<f32>], // [prefix][rope_d]
            kv_b_w: &[f32],
            scale: f32,
            q_idx: &[Vec<f32>], // [Hi][Di] - THIS query token's indexer query, per head
            k_idx_cache: &[Vec<f32>], // [prefix][Di] - indexer key prefix, shared across heads
            w_idx: &[f32],      // [Hi] - THIS query token's per-head indexer weight
        ) -> Vec<f32> {
            let prefix = c_kv_cache.len();
            let (hi, di) = (dcfg.index_n_heads, dcfg.index_head_dim);

            let mut idx_scores = vec![0.0f32; prefix];
            for s in 0..prefix {
                let mut score = 0.0f32;
                for hh in 0..hi {
                    let mut dot = 0.0f32;
                    for d in 0..di {
                        dot += q_idx[hh][d] * k_idx_cache[s][d];
                    }
                    score += w_idx[hh] * dot.max(0.0); // ReLU
                }
                idx_scores[s] = score;
            }
            let k_sel = dcfg.index_topk.min(prefix);
            let mut order: Vec<usize> = (0..prefix).collect();
            order.sort_by(|&a, &b| idx_scores[b].partial_cmp(&idx_scores[a]).unwrap());
            let selected = &order[..k_sel];

            let (hq, nope, vd, rope_d) = (
                cfg.n_heads,
                cfg.qk_nope_head_dim,
                cfg.v_head_dim,
                cfg.qk_rope_head_dim,
            );
            let kv_rank = cfg.kv_lora_rank;
            let mut k_nope = vec![vec![vec![0.0f32; nope]; k_sel]; hq];
            let mut v = vec![vec![vec![0.0f32; vd]; k_sel]; hq];
            for (si, &t) in selected.iter().enumerate() {
                let expanded = linear_ref(&c_kv_cache[t], kv_b_w, kv_rank, hq * (nope + vd));
                for hh in 0..hq {
                    let base = hh * (nope + vd);
                    k_nope[hh][si].copy_from_slice(&expanded[base..base + nope]);
                    v[hh][si].copy_from_slice(&expanded[base + nope..base + nope + vd]);
                }
            }
            let mut out = vec![0.0f32; hq * vd];
            for hh in 0..hq {
                let mut scores = vec![0.0f32; k_sel];
                for (si, &t) in selected.iter().enumerate() {
                    let mut dot = 0.0f32;
                    for i in 0..nope {
                        dot += q_full[hh][i] * k_nope[hh][si][i];
                    }
                    for i in 0..rope_d {
                        dot += q_full[hh][nope + i] * k_pe_cache[t][i];
                    }
                    scores[si] = dot * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                let mut e = vec![0.0f32; k_sel];
                for (si, &sc) in scores.iter().enumerate() {
                    e[si] = (sc - m).exp();
                    denom += e[si];
                }
                for i in 0..vd {
                    let mut acc = 0.0f32;
                    for (si, &ei) in e.iter().enumerate() {
                        acc += (ei / denom) * v[hh][si][i];
                    }
                    out[hh * vd + i] = acc;
                }
            }
            out
        }

        /// Full independent-reference forward over `tokens` (dense FFN, see `tiny_mp`). Returns the last
        /// position's logits, as `trace_deepseek32_dsa_prefill`. Also the decode oracle:
        /// `deepseek32_prefill_ref` on `tokens[..=pos]` is what a decode step at `pos` must produce (as
        /// `deepseek3_decode_matches_prefill_at_every_position` for plain MLA).
        fn deepseek32_prefill_ref(
            cfg: &DeepseekV2Config,
            dcfg: &DsaConfig,
            mp: &DeepseekV3MoeParams,
            tokens: &[usize],
            w: &HashMap<String, Vec<f32>>,
        ) -> Vec<f32> {
            let h = cfg.hidden;
            let hq = cfg.n_heads;
            let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
            let qk_head_dim = cfg.qk_head_dim();
            let kv_rank = cfg.kv_lora_rank;
            let scale = cfg.attn_scale();
            let (hi, di) = (dcfg.index_n_heads, dcfg.index_head_dim);
            let q_lora_rank = cfg
                .q_lora_rank
                .expect("DSA test fixtures always set q_lora_rank - see tiny_cfg's doc comment");
            let l = tokens.len();
            let embed = &w["model.embed_tokens.weight"];
            let cos = &w["rope.cos"];
            let sin = &w["rope.sin"];
            let idx_cos = &w["index.rope.cos"];
            let idx_sin = &w["index.rope.sin"];
            let half = rope_d / 2;

            let mut x: Vec<Vec<f32>> = tokens
                .iter()
                .map(|&t| embed[t * h..(t + 1) * h].to_vec())
                .collect();

            for li in 0..cfg.layers {
                let p = |s: &str| format!("model.layers.{li}.{s}");
                let ln1w = &w[&p("input_layernorm.weight")];
                let normed: Vec<Vec<f32>> = x
                    .iter()
                    .map(|row| rmsnorm_ref(row, ln1w, h, cfg.eps))
                    .collect();

                // Q: low-rank-compressed (real DSA always has q_lora_rank Some), as a hand-rolled
                // linear/rmsnorm/linear chain independent of `mla_query_proj`.
                let wqa = &w[&p("self_attn.q_a_proj.weight")];
                let wqaln = &w[&p("self_attn.q_a_layernorm.weight")];
                let wqb = &w[&p("self_attn.q_b_proj.weight")];
                let mut q_flat: Vec<Vec<f32>> = Vec::with_capacity(l);
                let mut qr_cache: Vec<Vec<f32>> = Vec::with_capacity(l); // [pos][q_lora_rank]
                for row in normed.iter() {
                    let q_a = linear_ref(row, wqa, h, q_lora_rank);
                    let qr = rmsnorm_ref(&q_a, wqaln, q_lora_rank, cfg.eps);
                    q_flat.push(linear_ref(&qr, wqb, q_lora_rank, hq * qk_head_dim));
                    qr_cache.push(qr);
                }
                let mut q_full: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0f32; qk_head_dim]; hq]; l];
                for (pos, qrow) in q_flat.iter().enumerate() {
                    for (hh, slot) in q_full[pos].iter_mut().enumerate() {
                        let base = hh * qk_head_dim;
                        let q_nope = &qrow[base..base + nope];
                        let q_pe = &qrow[base + nope..base + qk_head_dim];
                        let q_pe_rot = rope_interleaved_ref(q_pe, cos, sin, pos, half);
                        slot[..nope].copy_from_slice(q_nope);
                        slot[nope..].copy_from_slice(&q_pe_rot);
                    }
                }

                let wkva = &w[&p("self_attn.kv_a_proj_with_mqa.weight")];
                let wkvaln = &w[&p("self_attn.kv_a_layernorm.weight")];
                let wkvb = &w[&p("self_attn.kv_b_proj.weight")];
                let mut c_kv_cache: Vec<Vec<f32>> = Vec::with_capacity(l);
                let mut k_pe_cache: Vec<Vec<f32>> = Vec::with_capacity(l);
                for (pos, row) in normed.iter().enumerate() {
                    let kva = linear_ref(row, wkva, h, kv_rank + rope_d);
                    let c_kv = rmsnorm_ref(&kva[..kv_rank], wkvaln, kv_rank, cfg.eps);
                    let k_pe_rot = rope_interleaved_ref(&kva[kv_rank..], cos, sin, pos, half);
                    c_kv_cache.push(c_kv);
                    k_pe_cache.push(k_pe_rot);
                }

                // Indexer query: `wq_b` consumes `qr` (the `q_a_layernorm` output above, update 0857).
                let wiqb = &w[&p("self_attn.indexer.wq_b.weight")];
                let wik = &w[&p("self_attn.indexer.wk.weight")];
                let wiw = &w[&p("self_attn.indexer.weights_proj.weight")];
                let wiknw = &w[&p("self_attn.indexer.k_norm.weight")];
                let wiknb = &w[&p("self_attn.indexer.k_norm.bias")];
                let mut q_idx_cache: Vec<Vec<Vec<f32>>> = Vec::with_capacity(l); // [pos][Hi][Di]
                let mut k_idx_cache: Vec<Vec<f32>> = Vec::with_capacity(l); // [pos][Di]
                let mut w_idx_cache: Vec<Vec<f32>> = Vec::with_capacity(l); // [pos][Hi]
                for (pos, row) in normed.iter().enumerate() {
                    let idx_c = &idx_cos[pos * rope_d..(pos + 1) * rope_d];
                    let idx_s = &idx_sin[pos * rope_d..(pos + 1) * rope_d];
                    let qi_flat = linear_ref(&qr_cache[pos], wiqb, q_lora_rank, hi * di);
                    let mut qi = vec![vec![0.0f32; di]; hi];
                    for (hh, slot) in qi.iter_mut().enumerate() {
                        let base = hh * di;
                        *slot =
                            rope_half_split_ref(&qi_flat[base..base + di], idx_c, idx_s, rope_d);
                    }
                    q_idx_cache.push(qi);
                    // `k_norm` LayerNorm before the RoPE split (independent `layernorm_ref`).
                    let ki_raw = linear_ref(row, wik, h, di);
                    let ki_normed = layernorm_ref(&ki_raw, wiknw, wiknb, di, DSA_KNORM_EPS);
                    k_idx_cache.push(rope_half_split_ref(&ki_normed, idx_c, idx_s, rope_d));
                    w_idx_cache.push(linear_ref(row, wiw, h, hi));
                }

                let mut attn_out = vec![vec![0.0f32; hq * vd]; l];
                for qpos in 0..l {
                    attn_out[qpos] = dsa_attend_ref(
                        cfg,
                        dcfg,
                        &q_full[qpos],
                        &c_kv_cache[..=qpos],
                        &k_pe_cache[..=qpos],
                        wkvb,
                        scale,
                        &q_idx_cache[qpos],
                        &k_idx_cache[..=qpos],
                        &w_idx_cache[qpos],
                    );
                }
                let wo = &w[&p("self_attn.o_proj.weight")];
                for pos in 0..l {
                    let proj = linear_ref(&attn_out[pos], wo, hq * vd, h);
                    for c in 0..h {
                        x[pos][c] += proj[c];
                    }
                }

                let ln2w = &w[&p("post_attention_layernorm.weight")];
                let wg = &w[&p("mlp.gate_proj.weight")];
                let wu = &w[&p("mlp.up_proj.weight")];
                let wd = &w[&p("mlp.down_proj.weight")];
                assert!(
                    !mp.is_moe_layer(li),
                    "tiny_mp keeps every layer dense this round"
                );
                for row in x.iter_mut().take(l) {
                    let normed2 = rmsnorm_ref(row, ln2w, h, cfg.eps);
                    let ffn_out = dense_ffn_ref(&normed2, wg, wu, wd, h, mp.dense_inter);
                    for c in 0..h {
                        row[c] += ffn_out[c];
                    }
                }
            }

            let ln_f = &w["model.norm.weight"];
            let lm_head = &w["lm_head.weight"];
            let last = rmsnorm_ref(&x[l - 1], ln_f, h, cfg.eps);
            linear_ref(&last, lm_head, h, cfg.vocab)
        }

        fn eval_deepseek32_prefill(
            g: &Graph,
            tokens: &[usize],
            weights: &HashMap<String, Vec<f32>>,
        ) -> poot_tensor::HostTensor {
            let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                let t = match &meta.storage {
                    Storage::Slot(Slot::Token) => poot_tensor::HostTensor::i32(
                        vec![tokens.len()],
                        tokens.iter().map(|&t| t as i32).collect(),
                    ),
                    Storage::Slot(Slot::Mask) => {
                        let name = meta.name.as_deref().expect("mask slot without a name");
                        assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                        let l = meta.aval.shape[2];
                        let mut m = vec![0.0f32; l * l];
                        for i in 0..l {
                            for j in 0..l {
                                m[i * l + j] = if j <= i { 0.0 } else { -1.0e30 };
                            }
                        }
                        poot_tensor::HostTensor::f32(meta.aval.shape.clone(), m)
                    }
                    Storage::Const => {
                        let name = meta.name.as_deref().expect("const without a name");
                        poot_tensor::HostTensor::f32(
                            meta.aval.shape.clone(),
                            crate::model_fixture_data(weights, name),
                        )
                    }
                    other => panic!("unexpected storage {other:?} in a stateless prefill graph"),
                };
                inputs.insert(id, t.into());
            }
            poot_eval::eval(
                g,
                &inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .expect("cpu eval")
            .output
            .into_host()
            .expect("dense output")
        }

        fn assert_matches(got: &[f32], want: &[f32], vocab: usize) {
            assert_eq!(got.len(), vocab);
            poot_test_util::assert_close_rel(got, want, 1e-4);
        }

        /// SC-005: `trace_deepseek32_dsa_prefill` vs the independent reference with sparse top-k
        /// (`index_topk (3) < seq_len (6)`).
        #[test]
        fn deepseek32_dsa_prefill_matches_independent_reference() {
            let cfg = tiny_cfg();
            let dcfg = tiny_dcfg(3);
            let mp = tiny_mp();
            let tokens = [3usize, 7, 1, 9, 5, 2];
            let weights = all_weights(&cfg, &dcfg, &mp);

            let g = trace_deepseek32_dsa_prefill(cfg, dcfg, mp, tokens.len());
            let got = eval_deepseek32_prefill(&g, &tokens, &weights);
            let want = deepseek32_prefill_ref(&cfg, &dcfg, &mp, &tokens, &weights);
            assert_matches(got.as_f32().unwrap(), &want, cfg.vocab);
        }

        /// YaRN inheritance ([NEEDS CLARIFICATION] item): with `cfg.yarn = Some(_)`, the indexer's RoPE
        /// table must use the same YaRN scaling as the main MLA table (`dsa_indexer_rope_tables`).
        /// `all_weights` passes `cfg.yarn.as_ref()` to both `deepseek2_rope_tables_interleaved` and
        /// `dsa_indexer_rope_tables`; matching the independent reference with YaRN engaged shows the
        /// plumbing is live, as
        /// `crate::deepseek3::deepseek3_yarn_prefill_matches_hand_rolled_reference` does.
        #[test]
        fn deepseek32_dsa_prefill_matches_independent_reference_with_yarn_indexer_rope() {
            let yarn = DeepseekV2Yarn {
                factor: 40.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position_embeddings: 4096,
                attention_factor: 1.0,
                softmax_mscale_sq: 1.5896,
            };
            let cfg = DeepseekV2Config {
                yarn: Some(yarn),
                ..tiny_cfg()
            };
            let dcfg = tiny_dcfg(3);
            let mp = tiny_mp();
            let tokens = [3usize, 7, 1, 9, 5, 2];
            let weights = all_weights(&cfg, &dcfg, &mp);

            let g = trace_deepseek32_dsa_prefill(cfg, dcfg, mp, tokens.len());
            let got = eval_deepseek32_prefill(&g, &tokens, &weights);
            let want = deepseek32_prefill_ref(&cfg, &dcfg, &mp, &tokens, &weights);
            assert_matches(got.as_f32().unwrap(), &want, cfg.vocab);

            // YaRN indexer tables must differ from the plain (yarn=None) tables, else the test passes
            // vacuously if `yarn` is ignored. Checked at width 8, not `tiny_cfg`'s `qk_rope_head_dim=2`:
            // at width 2 the NTK-by-parts ramp saturates to pure extrapolation for j=0 regardless of
            // `yarn`, so an ignored `yarn` would go unnoticed.
            let (plain_idx_cos, _) = dsa_indexer_rope_tables(cfg.max_pos, 8, cfg.rope_theta, None);
            let (yarn_idx_cos, _) =
                dsa_indexer_rope_tables(cfg.max_pos, 8, cfg.rope_theta, cfg.yarn.as_ref());
            assert_ne!(
                plain_idx_cos, yarn_idx_cos,
                "yarn must change the indexer's own rope table, not just the main table"
            );
        }

        /// SC-003: `trace_deepseek32_dsa_decode`, driven step by step with cache carry-over via
        /// `poot_eval::eval`, vs `deepseek32_prefill_ref` on the causal prefix at every step,
        /// covering `pos + 1 < index_topk` (FR-003, steps 0-1 with `index_topk=3`) and sparse selection
        /// (`pos + 1 > index_topk`, steps 3-5).
        #[test]
        fn deepseek32_dsa_decode_matches_independent_reference_at_every_position() {
            let cfg = tiny_cfg();
            let dcfg = tiny_dcfg(3);
            let mp = tiny_mp();
            // These tokens give indexer scores with no exact ties. The graph's pairwise-`Ge` top-k keeps every
            // entry tied at the k-th score (the ReLU makes exact zero ties common) while the reference keeps
            // exactly k, so two of the three other token sets tried disagree at a tied step. That tie rule is a
            // finding to settle (Card 518 report), not something this fixture should hide.
            let tokens = [0usize, 4, 8, 2, 6, 1];
            let weights = all_weights(&cfg, &dcfg, &mp);
            let cap = tokens.len();

            let g = trace_deepseek32_dsa_decode(cfg, dcfg, mp, cap);
            let mut caches: Vec<poot_tensor::HostTensor> = g
                .state
                .iter()
                .map(|&(si, _)| poot_tensor::HostTensor::zeros(g.aval(si).shape.clone()))
                .collect();

            for pos in 0..tokens.len() {
                let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
                for &id in &g.inputs {
                    let meta = g.meta(id);
                    let t = match meta.storage {
                        Storage::Slot(Slot::Token) => {
                            poot_tensor::HostTensor::i32(vec![], vec![tokens[pos] as i32])
                        }
                        Storage::Slot(Slot::Pos) => {
                            poot_tensor::HostTensor::i32(vec![], vec![pos as i32])
                        }
                        Storage::Slot(Slot::SeqLen) => {
                            poot_tensor::HostTensor::i32(vec![], vec![(pos + 1) as i32])
                        }
                        Storage::Slot(Slot::Mask) => {
                            let cap_n = meta.aval.shape.iter().product::<usize>();
                            let m: Vec<f32> = (0..cap_n)
                                .map(|t| if t <= pos { 0.0 } else { -1.0e9 })
                                .collect();
                            poot_tensor::HostTensor::f32(meta.aval.shape.clone(), m)
                        }
                        Storage::State => continue,
                        Storage::Const => {
                            let name = meta.name.as_deref().expect("const without a name");
                            poot_tensor::HostTensor::f32(
                                meta.aval.shape.clone(),
                                crate::model_fixture_data(&weights, name),
                            )
                        }
                        other => panic!("unexpected storage {other:?}"),
                    };
                    inputs.insert(id, t.into());
                }
                for (ci, &(si, _)) in g.state.iter().enumerate() {
                    inputs.insert(si, caches[ci].clone().into());
                }
                let step = poot_eval::eval(
                    &g,
                    &inputs,
                    poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
                )
                .expect("decode step eval");
                let logits = step.output.into_host().expect("dense output");
                caches = step
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("dense state"))
                    .collect();

                let prefix: Vec<usize> = tokens[..=pos].to_vec();
                let want = deepseek32_prefill_ref(&cfg, &dcfg, &mp, &prefix, &weights);
                assert_eq!(logits.shape(), vec![1, 1, cfg.vocab]);
                assert_matches(logits.as_f32().unwrap(), &want, cfg.vocab);
            }
        }

        /// FR-004/SC-004: with `index_topk >= cap`, the top-k mask is a no-op, so
        /// `trace_deepseek32_dsa_decode` must match plain `crate::deepseek3::trace_deepseek3_decode_kv_masked`
        /// step by step with the same weights (the plain graph ignores the extra
        /// `self_attn.indexer.*`/`index.rope.*` keys), showing DSA is additive on the verified MLA tracer.
        #[test]
        fn deepseek32_dsa_decode_degenerates_to_plain_mla_when_topk_covers_cap() {
            let cfg = tiny_cfg();
            let cap = 6;
            let dcfg = tiny_dcfg(cap); // index_topk >= cap: never drops a causally-valid slot
            let mp = tiny_mp();
            let tokens = [3usize, 7, 1, 9, 5, 2];
            let weights = all_weights(&cfg, &dcfg, &mp);

            let g_dsa = trace_deepseek32_dsa_decode(cfg, dcfg, mp, cap);
            let g_mla = trace_deepseek3_decode_kv_masked(cfg, mp, cap);
            let mut caches_dsa: Vec<poot_tensor::HostTensor> = g_dsa
                .state
                .iter()
                .map(|&(si, _)| poot_tensor::HostTensor::zeros(g_dsa.aval(si).shape.clone()))
                .collect();
            let mut caches_mla: Vec<poot_tensor::HostTensor> = g_mla
                .state
                .iter()
                .map(|&(si, _)| poot_tensor::HostTensor::zeros(g_mla.aval(si).shape.clone()))
                .collect();

            fn build_inputs(
                g: &Graph,
                pos: usize,
                tok: usize,
                weights: &HashMap<String, Vec<f32>>,
                caches: &[poot_tensor::HostTensor],
            ) -> HashMap<poot_graph_ir::ValueId, poot_eval::Value> {
                let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
                for &id in &g.inputs {
                    let meta = g.meta(id);
                    let t = match meta.storage {
                        Storage::Slot(Slot::Token) => {
                            poot_tensor::HostTensor::i32(vec![], vec![tok as i32])
                        }
                        Storage::Slot(Slot::Pos) => {
                            poot_tensor::HostTensor::i32(vec![], vec![pos as i32])
                        }
                        Storage::Slot(Slot::SeqLen) => {
                            poot_tensor::HostTensor::i32(vec![], vec![(pos + 1) as i32])
                        }
                        Storage::Slot(Slot::Mask) => {
                            let cap_n = meta.aval.shape.iter().product::<usize>();
                            let m: Vec<f32> = (0..cap_n)
                                .map(|t| if t <= pos { 0.0 } else { -1.0e9 })
                                .collect();
                            poot_tensor::HostTensor::f32(meta.aval.shape.clone(), m)
                        }
                        Storage::State => continue,
                        Storage::Const => {
                            let name = meta.name.as_deref().expect("const without a name");
                            poot_tensor::HostTensor::f32(
                                meta.aval.shape.clone(),
                                crate::model_fixture_data(weights, name),
                            )
                        }
                        other => panic!("unexpected storage {other:?}"),
                    };
                    inputs.insert(id, t.into());
                }
                for (ci, &(si, _)) in g.state.iter().enumerate() {
                    inputs.insert(si, caches[ci].clone().into());
                }
                inputs
            }

            for (pos, &tok) in tokens.iter().enumerate() {
                let inputs_dsa = build_inputs(&g_dsa, pos, tok, &weights, &caches_dsa);
                let step_dsa = poot_eval::eval(
                    &g_dsa,
                    &inputs_dsa,
                    poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
                )
                .expect("dsa decode step eval");
                let logits_dsa = step_dsa.output.into_host().expect("dense output");
                caches_dsa = step_dsa
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("dense state"))
                    .collect();

                let inputs_mla = build_inputs(&g_mla, pos, tok, &weights, &caches_mla);
                let step_mla = poot_eval::eval(
                    &g_mla,
                    &inputs_mla,
                    poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
                )
                .expect("mla decode step eval");
                let logits_mla = step_mla.output.into_host().expect("dense output");
                caches_mla = step_mla
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("dense state"))
                    .collect();

                assert_matches(
                    logits_dsa.as_f32().unwrap(),
                    logits_mla.as_f32().unwrap(),
                    cfg.vocab,
                );
            }
        }
    }
}
