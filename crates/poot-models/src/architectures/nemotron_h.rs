//! Nemotron-H (NVIDIA hybrid Mamba2/attention family) tracer, spec 279 (arXiv 2504.03624). Composition
//! functions plus CPU-oracle tests against an independent Rust reference; F32 only. See
//! `specs/279-nemotron-h/spec.md` for the derivation from the released config and modeling code.
//!
//! # Layer structure
//!
//! Each layer is exactly one of three kinds, not a mixer + FFN pair:
//! `residual = x; x = rmsnorm(x); x = mixer(x); return residual + x`, where `mixer` is a Mamba2 SSM, a
//! GQA attention block, or a plain MLP. `hybrid_override_pattern` gives one character per layer:
//! `M` = Mamba2, `*` = attention, `-` = MLP. The 8B pattern (52 chars = `num_hidden_layers`, 4 attention
//! layers) is `M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-`; the 56B pattern (118 chars, 10
//! attention layers) has the same alternating shape.
//!
//! Mamba2 mixer (8B: `mamba_num_heads=128`, `mamba_head_dim=64`, `ssm_state_size=128`, `n_groups=8`,
//! `conv_kernel=4`): `in_proj` splits into a gate `z`, a combined `xBC` block, and `dt`. A depthwise
//! causal conv1d + SiLU runs over the whole `xBC` block (x, B and C together; see
//! `nemotron_h_mamba_layer`), then `xBC` is split into `x`/`B`/`C`, `Delta = softplus(dt + dt_bias)`, and
//! the SSD recurrence (`poot_graph_ir::ops::mamba2_ssd_decode`) runs per head. The output gate is a gated
//! RMSNorm, `rmsnorm(scan_out * silu(gate), weight)` (matching `mamba_ssm`'s `RMSNormGated`: the RMS
//! reduction runs on the already-gated value). `n_groups=8` is smaller than `mamba_num_heads=128`, so 16
//! heads share one `B`/`C` group; `nemotron_h_mamba_layer` broadcasts groups to heads via `repeat_kv`
//! (blocked convention `h -> h/n_rep`, matching `mamba_ssm`'s `repeat_interleave`). This is not
//! `repeat_kv_tiled`, the interleaved-tile convention Qwen3-Next's Gated-DeltaNet uses.
//!
//! Attention mixer: GQA softmax attention with no positional embedding (NoPE); `modeling_nemotron_h.py`
//! runs `scaled_dot_product_attention` on the raw projected q/k/v.
//!
//! MLP-only layer (`mlp_hidden_act: "relu2"`): `down_proj(relu(up_proj(x))^2)`, a single up/down pair
//! with squared ReLU and no gate projection.
//!
//! # Reused ops
//!
//! `poot_graph_ir::ops::{mamba2_ssd_decode, causal_conv1d_decode, softplus, relu, rmsnorm, repeat_kv,
//! linear, attention}`; no new `OpKind`. The grouped B/C repeat is `repeat_kv`, the gated RMSNorm is
//! `rmsnorm` fed a pre-gated input, and `relu2` is `relu` self-multiplied.

use poot_graph_ir::builder::{Builder, Traced};
use poot_graph_ir::graph::{Graph, Slot, StateRole};
use poot_graph_ir::op::BinOp;
use poot_graph_ir::ops::{
    attention, attention_masked, attention_prefill, causal_conv1d_decode, causal_conv1d_prefill,
    linear, mamba2_ssd_decode, mamba2_ssd_prefill, relu, repeat_kv, rmsnorm, silu, softplus,
};
use poot_graph_ir::types::TensorType;
use poot_tensor::DType;

/// One layer's mixer kind, decoded from a `hybrid_override_pattern` character.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NemotronHLayerKind {
    /// `'M'` - a Mamba2 SSM-only layer.
    Mamba,
    /// `'*'` - a GQA softmax-attention-only layer (NoPE).
    Attention,
    /// `'-'` - a plain (non-gated) squared-ReLU MLP-only layer.
    Mlp,
}

/// Decode a `hybrid_override_pattern` string into one [`NemotronHLayerKind`] per layer, in order.
/// Panics on any character other than `M`/`*`/`-`.
pub fn parse_hybrid_pattern(pattern: &str) -> Vec<NemotronHLayerKind> {
    pattern
        .chars()
        .map(|c| match c {
            'M' => NemotronHLayerKind::Mamba,
            '*' => NemotronHLayerKind::Attention,
            '-' => NemotronHLayerKind::Mlp,
            other => panic!("nemotron_h hybrid_override_pattern: unexpected layer char {other:?}"),
        })
        .collect()
}

/// Nemotron-H Mamba2 mixer shape (real 8B config field names).
#[derive(Clone, Copy, Debug)]
pub struct NemotronHMambaConfig {
    pub hidden: usize,
    pub mamba_num_heads: usize,
    pub mamba_head_dim: usize,
    /// Number of `B`/`C` groups; `mamba_num_heads / n_groups` heads share each group (real 8B: 128/8=16).
    pub n_groups: usize,
    pub ssm_state: usize,
    pub conv_kernel: usize,
}

impl NemotronHMambaConfig {
    /// `expand * hidden` in the real config's own terms; the SSM's per-token channel width.
    pub fn inner(&self) -> usize {
        self.mamba_num_heads * self.mamba_head_dim
    }
    /// Combined conv1d channel width: `x` (inner) plus `B` and `C` (each `n_groups * ssm_state`), the joint `xBC` conv.
    pub fn conv_channels(&self) -> usize {
        self.inner() + 2 * self.n_groups * self.ssm_state
    }
    /// Heads sharing one `B`/`C` group.
    pub fn heads_per_group(&self) -> usize {
        self.mamba_num_heads / self.n_groups
    }
}

/// Nemotron-H attention mixer shape (real 8B config field names). NoPE: no rope fields.
#[derive(Clone, Copy, Debug)]
pub struct NemotronHAttnConfig {
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Whole-model Nemotron-H shape: the per-layer-kind sequence plus the two mixer configs, what a loader
/// needs to build one [`NemotronHLayerWeights`] per layer and drive
/// [`trace_nemotron_h_hybrid_stack_prefill`]. Adds `vocab_size`, `mlp_inter` (`intermediate_size`, shared
/// by every MLP layer) and `eps` (`rms_norm_eps`/`layer_norm_epsilon`, both `1e-05`, treated as one value).
#[derive(Clone, Debug)]
pub struct NemotronHConfig {
    pub vocab_size: usize,
    pub hidden: usize,
    pub mlp_inter: usize,
    pub eps: f32,
    pub pattern: Vec<NemotronHLayerKind>,
    pub mamba: NemotronHMambaConfig,
    pub attn: NemotronHAttnConfig,
}

/// One Nemotron-H Mamba2 decode step. `x` is `[1,1,H]`. Weight shapes:
/// `norm_w[H]`, `w_z[H,Inner]`, `w_xbc[H,ConvC]`, `w_dt[H,Hq]`, `dt_bias[Hq]`, `w_conv[K,ConvC]`,
/// `conv_bias[ConvC]`, `a_param`/`d_param[1,Hq,1,1]`, `gate_norm_w[Inner]`, `w_out[Inner,H]`. State:
/// `conv_cache[1,K-1,ConvC]`, `h_in[1,Hq,N,P]`. Returns `(out[1,1,H], conv_cache_out, h_out)`.
///
/// `conv_bias`: the real checkpoint's depthwise conv1d has a per-channel bias (`use_conv_bias: true`,
/// tensor `backbone.layers.{i}.mixer.conv1d.bias`, `[ConvC]`). `causal_conv1d_decode`/`_prefill` take no
/// bias, so it is added by a broadcast `Add` between the conv and the SiLU. Omitting it silently gives
/// wrong numbers on a real checkpoint.
#[allow(clippy::too_many_arguments)]
pub fn nemotron_h_mamba_layer(
    b: &Builder,
    x: Traced,
    cfg: &NemotronHMambaConfig,
    norm_w: Traced,
    w_z: Traced,
    w_xbc: Traced,
    w_dt: Traced,
    dt_bias: Traced,
    w_conv: Traced,
    conv_bias: Traced,
    a_param: Traced,
    d_param: Traced,
    gate_norm_w: Traced,
    w_out: Traced,
    conv_cache: Traced,
    h_in: Traced,
    eps: f32,
) -> (Traced, Traced, Traced) {
    let (hq, p, n) = (cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.ssm_state);
    let g = cfg.n_groups;
    let n_rep = cfg.heads_per_group();
    let inner = cfg.inner();

    let xn = rmsnorm(b, x, norm_w, eps);
    let z = linear(b, xn, w_z, None); // [1,1,Inner] gate
    let xbc_raw = linear(b, xn, w_xbc, None); // [1,1,ConvC]
    let dt_raw = linear(b, xn, w_dt, Some(dt_bias)); // [1,1,Hq]

    // Joint conv over x/B/C together (matches the real `causal_conv1d_fn` scope).
    let (conv_raw, conv_cache_out) =
        causal_conv1d_decode(b, xbc_raw, w_conv, conv_cache, cfg.conv_kernel);
    let conv_out = b.binary(BinOp::Add, conv_raw, conv_bias); // FR-007: real checkpoint's conv1d bias
    let xbc_act = silu(b, conv_out); // [1,1,ConvC]

    let x_part = b.slice(xbc_act, 2, 0, inner); // [1,1,Inner]
    let b_part = b.slice(xbc_act, 2, inner, inner + g * n); // [1,1,G*N]
    let c_part = b.slice(xbc_act, 2, inner + g * n, inner + 2 * g * n); // [1,1,G*N]

    let x_heads = b.reshape(x_part, vec![1, hq, 1, p]);
    let b_groups = b.reshape(b_part, vec![1, g, 1, n]);
    let c_groups = b.reshape(c_part, vec![1, g, 1, n]);
    // blocked repeat (h -> h/n_rep) like `repeat_interleave(B, repeats=heads_per_group)`, not GDN's `repeat_kv_tiled`.
    let b_heads = repeat_kv(b, b_groups, n_rep); // [1,Hq,1,N]
    let c_heads = repeat_kv(b, c_groups, n_rep); // [1,Hq,1,N]

    let delta = softplus(b, b.reshape(dt_raw, vec![1, hq, 1, 1]));
    let (y, h_out) = mamba2_ssd_decode(b, x_heads, b_heads, c_heads, delta, a_param, d_param, h_in);

    let y_flat = b.reshape(y, vec![1, 1, inner]);
    let z_silu = silu(b, z);
    let gated = b.binary(BinOp::Mul, y_flat, z_silu); // gate BEFORE the norm reduction (FR-003)
    let normed = rmsnorm(b, gated, gate_norm_w, eps);
    let out = linear(b, normed, w_out, None);
    let result = b.binary(BinOp::Add, x, out);
    (result, conv_cache_out, h_out)
}

/// One Nemotron-H attention decode step (no RoPE). `x` is `[1,1,H]`. Weight shapes:
/// `norm_w[H]`, `w_q[H,Aq*D]`, `w_k`/`w_v[H,Akv*D]`, `w_o[Aq*D,H]`. `k_cache`/`v_cache` are
/// `[1,Akv,cap,D]`; `pos` is the trace-time write slot (CPU-oracle only). Returns
/// `(out[1,1,H], k_cache_out, v_cache_out)`.
#[allow(clippy::too_many_arguments)]
pub fn nemotron_h_attention_layer(
    b: &Builder,
    x: Traced,
    cfg: &NemotronHAttnConfig,
    norm_w: Traced,
    w_q: Traced,
    w_k: Traced,
    w_v: Traced,
    w_o: Traced,
    k_cache: Traced,
    v_cache: Traced,
    pos: usize,
    eps: f32,
) -> (Traced, Traced, Traced) {
    let (aq, akv, d) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let n_rep = aq / akv;
    let scale = 1.0 / (d as f32).sqrt();

    let xn = rmsnorm(b, x, norm_w, eps);
    let q = b.transpose(
        b.reshape(linear(b, xn, w_q, None), vec![1, 1, aq, d]),
        vec![0, 2, 1, 3],
    );
    let k = b.transpose(
        b.reshape(linear(b, xn, w_k, None), vec![1, 1, akv, d]),
        vec![0, 2, 1, 3],
    );
    let v = b.transpose(
        b.reshape(linear(b, xn, w_v, None), vec![1, 1, akv, d]),
        vec![0, 2, 1, 3],
    );

    let k_cache_out = b.dynamic_update_slice(k_cache, k, pos, 2);
    let v_cache_out = b.dynamic_update_slice(v_cache, v, pos, 2);
    let k_valid = b.slice(k_cache_out, 2, 0, pos + 1);
    let v_valid = b.slice(v_cache_out, 2, 0, pos + 1);

    let o = attention(b, q, k_valid, v_valid, n_rep, scale); // [1,Aq,1,D]
    let o_flat = b.reshape(b.transpose(o, vec![0, 2, 1, 3]), vec![1, 1, aq * d]);
    let out = linear(b, o_flat, w_o, None);
    let result = b.binary(BinOp::Add, x, out);
    (result, k_cache_out, v_cache_out)
}

/// One Nemotron-H attention decode step over a fixed-capacity cache, for a whole-model decode graph built
/// once and replayed at every position ([`trace_nemotron_h_decode`]). [`nemotron_h_attention_layer`] bakes
/// `pos` in as a trace-time `usize` and reads a growing `[0..=pos]` slice, so it needs a fresh graph per
/// step. This variant takes a runtime `pos_slot` (`[]`-shaped scalar, written via
/// [`poot_graph_ir::builder::Builder::dynamic_update_slice_dyn`]) and reads the full `cap`-wide caches
/// through [`poot_graph_ir::ops::attention_masked`]'s additive `mask` (`[1,1,1,cap]`: 0 for `t <= pos`,
/// large-negative otherwise, the `Slot::Mask` convention of the other fixed-KV decode tracers). `x` is
/// `[1,1,H]`; weight shapes as in [`nemotron_h_attention_layer`]. Returns `(out[1,1,H], k_cache_out,
/// v_cache_out)`.
#[allow(clippy::too_many_arguments)]
pub fn nemotron_h_attention_layer_decode_masked(
    b: &Builder,
    x: Traced,
    cfg: &NemotronHAttnConfig,
    norm_w: Traced,
    w_q: Traced,
    w_k: Traced,
    w_v: Traced,
    w_o: Traced,
    k_cache: Traced,
    v_cache: Traced,
    pos_slot: Traced,
    mask: Traced,
    eps: f32,
) -> (Traced, Traced, Traced) {
    let (aq, akv, d) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let n_rep = aq / akv;
    let scale = 1.0 / (d as f32).sqrt();

    let xn = rmsnorm(b, x, norm_w, eps);
    let q = b.transpose(
        b.reshape(linear(b, xn, w_q, None), vec![1, 1, aq, d]),
        vec![0, 2, 1, 3],
    );
    let k = b.transpose(
        b.reshape(linear(b, xn, w_k, None), vec![1, 1, akv, d]),
        vec![0, 2, 1, 3],
    );
    let v = b.transpose(
        b.reshape(linear(b, xn, w_v, None), vec![1, 1, akv, d]),
        vec![0, 2, 1, 3],
    );

    let k_cache_out = b.dynamic_update_slice_dyn(k_cache, k, pos_slot, 2);
    let v_cache_out = b.dynamic_update_slice_dyn(v_cache, v, pos_slot, 2);

    let o = attention_masked(b, q, k_cache_out, v_cache_out, n_rep, scale, mask); // [1,Aq,1,D]
    let o_flat = b.reshape(b.transpose(o, vec![0, 2, 1, 3]), vec![1, 1, aq * d]);
    let out = linear(b, o_flat, w_o, None);
    let result = b.binary(BinOp::Add, x, out);
    (result, k_cache_out, v_cache_out)
}

/// One Nemotron-H MLP-only decode step: `down_proj(relu(up_proj(x))^2)`, no gate. `x` is `[1,1,H]`.
pub fn nemotron_h_mlp_layer(
    b: &Builder,
    x: Traced,
    norm_w: Traced,
    w_up: Traced,
    w_down: Traced,
    eps: f32,
) -> Traced {
    let xn = rmsnorm(b, x, norm_w, eps);
    let up = linear(b, xn, w_up, None);
    let r = relu(b, up);
    let sq = b.binary(BinOp::Mul, r, r);
    let out = linear(b, sq, w_down, None);
    b.binary(BinOp::Add, x, out)
}

/// Nemotron-H Mamba2 mixer prefill (whole-sequence twin of [`nemotron_h_mamba_layer`]): the same
/// composition (pre-norm, in_proj split, joint xBC conv+SiLU, grouped B/C repeat, softplus(dt), SSD, gated
/// RMSNorm, out_proj, residual) over all `L` positions in one graph via [`causal_conv1d_prefill`] +
/// [`mamba2_ssd_prefill`]. Matches `L` sequential decode calls from a zero conv cache / zero SSM state
/// (checked by the `cpu_oracle` tests; fresh start is the scope of every prefill entry point here).
///
/// `x` is `[1,L,H]`; weight shapes as in [`nemotron_h_mamba_layer`]. `tril` is `[1,mamba_num_heads,L,L]`,
/// lower-triangular ones including the diagonal (as [`poot_graph_ir::ops::decay_mask_from_gates`]
/// expects). Returns `(out[1,L,H], conv_cache_out[1,K-1,ConvC], h_out[1,Hq,N,P])`; the state has the same
/// shapes as decode's `conv_cache`/`h_in`, so decode can be seeded from it.
#[allow(clippy::too_many_arguments)]
pub fn nemotron_h_mamba_layer_prefill(
    b: &Builder,
    x: Traced,
    cfg: &NemotronHMambaConfig,
    norm_w: Traced,
    w_z: Traced,
    w_xbc: Traced,
    w_dt: Traced,
    dt_bias: Traced,
    w_conv: Traced,
    conv_bias: Traced,
    a_param: Traced,
    d_param: Traced,
    gate_norm_w: Traced,
    w_out: Traced,
    tril: Traced,
    eps: f32,
) -> (Traced, Traced, Traced) {
    let (hq, p, n) = (cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.ssm_state);
    let g = cfg.n_groups;
    let n_rep = cfg.heads_per_group();
    let inner = cfg.inner();
    let l = b.aval(x).shape[1];

    let xn = rmsnorm(b, x, norm_w, eps); // [1,L,H]
    let z = linear(b, xn, w_z, None); // [1,L,Inner] gate
    let xbc_raw = linear(b, xn, w_xbc, None); // [1,L,ConvC]
    let dt_raw = linear(b, xn, w_dt, Some(dt_bias)); // [1,L,Hq]

    // Joint conv over x/B/C for all L positions; per-channel conv1d bias as an Add, as in decode.
    let conv_raw = causal_conv1d_prefill(b, xbc_raw, w_conv, cfg.conv_kernel); // [1,L,ConvC]
    let conv_out = b.binary(BinOp::Add, conv_raw, conv_bias);
    let xbc_act = silu(b, conv_out);

    let x_part = b.slice(xbc_act, 2, 0, inner); // [1,L,Inner]
    let b_part = b.slice(xbc_act, 2, inner, inner + g * n); // [1,L,G*N]
    let c_part = b.slice(xbc_act, 2, inner + g * n, inner + 2 * g * n); // [1,L,G*N]

    let x_heads = b.transpose(b.reshape(x_part, vec![1, l, hq, p]), vec![0, 2, 1, 3]); // [1,Hq,L,P]
    let b_groups = b.transpose(b.reshape(b_part, vec![1, l, g, n]), vec![0, 2, 1, 3]); // [1,G,L,N]
    let c_groups = b.transpose(b.reshape(c_part, vec![1, l, g, n]), vec![0, 2, 1, 3]); // [1,G,L,N]
    let b_heads = repeat_kv(b, b_groups, n_rep); // [1,Hq,L,N] - BLOCKED repeat, same convention as decode.
    let c_heads = repeat_kv(b, c_groups, n_rep); // [1,Hq,L,N]

    // dt_raw is [1,L,Hq] but mamba2_ssd_prefill wants head-major [1,Hq,L,1]: needs a transpose, not a reshape.
    let dt_heads = b.transpose(b.reshape(dt_raw, vec![1, l, hq, 1]), vec![0, 2, 1, 3]); // [1,Hq,L,1]
    let delta = softplus(b, dt_heads);
    let (y, h_out) =
        mamba2_ssd_prefill(b, x_heads, b_heads, c_heads, delta, a_param, d_param, tril);

    let y_flat = b.reshape(b.transpose(y, vec![0, 2, 1, 3]), vec![1, l, inner]); // [1,L,Inner]
    let z_silu = silu(b, z);
    let gated = b.binary(BinOp::Mul, y_flat, z_silu); // gate BEFORE the norm reduction (FR-003, same as decode)
    let normed = rmsnorm(b, gated, gate_norm_w, eps);
    let out = linear(b, normed, w_out, None);
    let result = b.binary(BinOp::Add, x, out);

    let k1 = cfg.conv_kernel - 1;
    let conv_cache_out = b.slice(xbc_raw, 1, l - k1, l); // last K-1 RAW (pre-conv) inputs, [1,K-1,ConvC]

    (result, conv_cache_out, h_out)
}

/// Nemotron-H attention mixer prefill (whole-sequence twin of [`nemotron_h_attention_layer`], no RoPE).
/// Needs no new IR composition: decode and prefill of plain GQA differ only in the KV write and mask, and
/// both exist as [`poot_graph_ir::ops::attention`] / [`poot_graph_ir::ops::attention_prefill`].
///
/// `x` is `[1,L,H]`; `causal_mask` is `[1,1,L,L]` additive (0 on/below the diagonal, large-negative above),
/// as [`poot_graph_ir::ops::attention_prefill`] expects. Returns `(out[1,L,H], k[1,Akv,L,D],
/// v[1,Akv,L,D])`; `k`/`v` have the layout of [`nemotron_h_attention_layer`]'s caches, so decode can be
/// seeded from them.
#[allow(clippy::too_many_arguments)]
pub fn nemotron_h_attention_layer_prefill(
    b: &Builder,
    x: Traced,
    cfg: &NemotronHAttnConfig,
    norm_w: Traced,
    w_q: Traced,
    w_k: Traced,
    w_v: Traced,
    w_o: Traced,
    causal_mask: Traced,
    eps: f32,
) -> (Traced, Traced, Traced) {
    let (aq, akv, d) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let n_rep = aq / akv;
    let scale = 1.0 / (d as f32).sqrt();
    let l = b.aval(x).shape[1];

    let xn = rmsnorm(b, x, norm_w, eps);
    let q = b.transpose(
        b.reshape(linear(b, xn, w_q, None), vec![1, l, aq, d]),
        vec![0, 2, 1, 3],
    ); // [1,Aq,L,D]
    let k = b.transpose(
        b.reshape(linear(b, xn, w_k, None), vec![1, l, akv, d]),
        vec![0, 2, 1, 3],
    ); // [1,Akv,L,D]
    let v = b.transpose(
        b.reshape(linear(b, xn, w_v, None), vec![1, l, akv, d]),
        vec![0, 2, 1, 3],
    ); // [1,Akv,L,D]

    let o = attention_prefill(b, q, k, v, n_rep, scale, causal_mask); // [1,Aq,L,D]
    let o_flat = b.reshape(b.transpose(o, vec![0, 2, 1, 3]), vec![1, l, aq * d]);
    let out = linear(b, o_flat, w_o, None);
    let result = b.binary(BinOp::Add, x, out);
    (result, k, v)
}

/// Per-layer weight bundle for one entry of a hybrid stack, tagged by [`NemotronHLayerKind`], used by
/// [`trace_nemotron_h_hybrid_stack_prefill`] to dispatch each layer to its prefill composition.
pub enum NemotronHLayerWeights {
    Mamba {
        norm_w: Traced,
        w_z: Traced,
        w_xbc: Traced,
        w_dt: Traced,
        dt_bias: Traced,
        w_conv: Traced,
        conv_bias: Traced,
        a_param: Traced,
        d_param: Traced,
        gate_norm_w: Traced,
        w_out: Traced,
    },
    Attention {
        norm_w: Traced,
        w_q: Traced,
        w_k: Traced,
        w_v: Traced,
        w_o: Traced,
    },
    Mlp {
        norm_w: Traced,
        w_up: Traced,
        w_down: Traced,
    },
}

/// Final per-layer recurrent state after a [`trace_nemotron_h_hybrid_stack_prefill`] call, one entry per
/// layer, with the same shapes as [`nemotron_h_mamba_layer`]/[`nemotron_h_attention_layer`]'s state args.
pub enum NemotronHLayerState {
    Mamba { conv_cache: Traced, h: Traced },
    Attention { k: Traced, v: Traced },
    Mlp,
}

/// Whole-prompt prefill dispatch over a hybrid stack, driven by [`parse_hybrid_pattern`] output: a full
/// `[1,L,H]` sequence through every layer in one graph.
///
/// `pattern`/`weights` are parallel (one [`NemotronHLayerWeights`] per `pattern` entry; a kind mismatch
/// panics). `mamba_tril` is `[1,mamba_num_heads,L,L]` ([`nemotron_h_mamba_layer_prefill`]'s `tril`);
/// `attn_causal_mask` is `[1,1,L,L]` additive ([`nemotron_h_attention_layer_prefill`]'s `causal_mask`);
/// each is shared across layers of its kind. MLP layers reuse [`nemotron_h_mlp_layer`] unchanged (it is
/// stateless and works for any leading `L`, checked by `nemotron_h_mlp_layer_prefill_matches_per_token_decode`).
/// Returns `(out[1,L,H], states[pattern.len()])`.
#[allow(clippy::too_many_arguments)]
pub fn trace_nemotron_h_hybrid_stack_prefill(
    b: &Builder,
    pattern: &[NemotronHLayerKind],
    weights: &[NemotronHLayerWeights],
    x0: Traced,
    mcfg: &NemotronHMambaConfig,
    acfg: &NemotronHAttnConfig,
    mamba_tril: Traced,
    attn_causal_mask: Traced,
    eps: f32,
) -> (Traced, Vec<NemotronHLayerState>) {
    assert_eq!(
        pattern.len(),
        weights.len(),
        "trace_nemotron_h_hybrid_stack_prefill: pattern/weights length mismatch"
    );
    let mut cur = x0;
    let mut states = Vec::with_capacity(pattern.len());
    for (kind, w) in pattern.iter().zip(weights.iter()) {
        match (kind, w) {
            (
                NemotronHLayerKind::Mamba,
                NemotronHLayerWeights::Mamba {
                    norm_w,
                    w_z,
                    w_xbc,
                    w_dt,
                    dt_bias,
                    w_conv,
                    conv_bias,
                    a_param,
                    d_param,
                    gate_norm_w,
                    w_out,
                },
            ) => {
                let (out, conv_cache_out, h_out) = nemotron_h_mamba_layer_prefill(
                    b,
                    cur,
                    mcfg,
                    *norm_w,
                    *w_z,
                    *w_xbc,
                    *w_dt,
                    *dt_bias,
                    *w_conv,
                    *conv_bias,
                    *a_param,
                    *d_param,
                    *gate_norm_w,
                    *w_out,
                    mamba_tril,
                    eps,
                );
                cur = out;
                states.push(NemotronHLayerState::Mamba {
                    conv_cache: conv_cache_out,
                    h: h_out,
                });
            }
            (
                NemotronHLayerKind::Attention,
                NemotronHLayerWeights::Attention {
                    norm_w,
                    w_q,
                    w_k,
                    w_v,
                    w_o,
                },
            ) => {
                let (out, k_out, v_out) = nemotron_h_attention_layer_prefill(
                    b,
                    cur,
                    acfg,
                    *norm_w,
                    *w_q,
                    *w_k,
                    *w_v,
                    *w_o,
                    attn_causal_mask,
                    eps,
                );
                cur = out;
                states.push(NemotronHLayerState::Attention { k: k_out, v: v_out });
            }
            (
                NemotronHLayerKind::Mlp,
                NemotronHLayerWeights::Mlp {
                    norm_w,
                    w_up,
                    w_down,
                },
            ) => {
                cur = nemotron_h_mlp_layer(b, cur, *norm_w, *w_up, *w_down, eps);
                states.push(NemotronHLayerState::Mlp);
            }
            _ => panic!(
                "trace_nemotron_h_hybrid_stack_prefill: layer kind/weight-bundle mismatch at some layer"
            ),
        }
    }
    (cur, states)
}

/// Top-level Nemotron-H prefill entry point: embeddings, [`trace_nemotron_h_hybrid_stack_prefill`],
/// final norm, lm_head; returns only the last position's logits. Same shape as
/// `crate::bloom::trace_bloom_prefill` and `crate::qwen3next`'s prefill trace.
///
/// Fresh-start prefill only (`h_in = 0` / zero KV cache). See `specs/279-nemotron-h/spec.md` (Out of
/// scope) for what `Runner`/`poot-serve` integration still needs.
///
/// Expects these named constants bound at eval time: `backbone.embeddings.weight` `[vocab,hidden]` (gather
/// source, untransposed; never assumed tied to `lm_head.weight`, since `tie_word_embeddings: false`),
/// the per-layer
/// constants from `crates/poot-llm/src/nemotron_h_load.rs::build_nemotron_h_weights`
/// (`layers.{i}.norm.weight` plus the per-kind `layers.{i}.mixer.*` set), and `backbone.norm_f.weight`
/// `[hidden]` / `lm_head.weight` `[hidden,vocab]` (pre-transposed, so there is no in-graph `Transpose` of
/// a large weight, as in `crate::bloom`), plus the `mask.prefill` step input (card 550a)
/// `[1,1,L,L]` additive (as [`nemotron_h_attention_layer_prefill`]'s `causal_mask`); the
/// `[1,mamba_num_heads,L,L]` SSD tril (as [`nemotron_h_mamba_layer_prefill`]'s `tril`) is computed
/// in-graph from an `iota` comparison.
pub fn trace_nemotron_h_prefill(cfg: &NemotronHConfig, seq_len: usize) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let l = seq_len;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let embed = b.constant(
        "backbone.embeddings.weight",
        TensorType::f32(vec![cfg.vocab_size, h]),
    );
    let emb = b.gather(embed, 0, tokens); // [L, H]
    let x0 = b.reshape(emb, vec![1, l, h]);

    // The SSD chunked-recurrence tril, computed in-graph (card 550a): ones on/below the diagonal of an
    // `iota >= iota^T` comparison, tiled over the mamba heads.
    let iota = b.iota(l);
    let rows = b.broadcast(b.reshape(iota, vec![l, 1]), vec![l, l]);
    let cols = b.broadcast(b.reshape(iota, vec![1, l]), vec![l, l]);
    let tril = b.binary(BinOp::Ge, rows, cols);
    let mamba_tril = b.broadcast(
        b.reshape(tril, vec![1, 1, l, l]),
        vec![1, cfg.mamba.mamba_num_heads, l, l],
    );
    let attn_causal_mask = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));

    let mut weights: Vec<NemotronHLayerWeights> = Vec::with_capacity(cfg.pattern.len());
    for (li, kind) in cfg.pattern.iter().enumerate() {
        let norm_w = b.constant(
            &format!("layers.{li}.norm.weight"),
            TensorType::f32(vec![h]),
        );
        let p = |s: &str| format!("layers.{li}.mixer.{s}");
        match kind {
            NemotronHLayerKind::Mamba => {
                let inner = cfg.mamba.inner();
                let conv_c = cfg.mamba.conv_channels();
                let hq = cfg.mamba.mamba_num_heads;
                let k = cfg.mamba.conv_kernel;
                weights.push(NemotronHLayerWeights::Mamba {
                    norm_w,
                    w_z: b.constant(&p("z.weight"), TensorType::f32(vec![h, inner])),
                    w_xbc: b.constant(&p("xbc.weight"), TensorType::f32(vec![h, conv_c])),
                    w_dt: b.constant(&p("dt.weight"), TensorType::f32(vec![h, hq])),
                    dt_bias: b.constant(&p("dt_bias"), TensorType::f32(vec![hq])),
                    w_conv: b.constant(&p("conv1d.weight"), TensorType::f32(vec![k, conv_c])),
                    conv_bias: b.constant(&p("conv1d.bias"), TensorType::f32(vec![conv_c])),
                    a_param: b.constant(&p("a"), TensorType::f32(vec![1, hq, 1, 1])),
                    d_param: b.constant(&p("d"), TensorType::f32(vec![1, hq, 1, 1])),
                    gate_norm_w: b.constant(&p("gate_norm.weight"), TensorType::f32(vec![inner])),
                    w_out: b.constant(&p("out_proj.weight"), TensorType::f32(vec![inner, h])),
                });
            }
            NemotronHLayerKind::Attention => {
                let qd = cfg.attn.num_heads * cfg.attn.head_dim;
                let kvd = cfg.attn.num_kv_heads * cfg.attn.head_dim;
                weights.push(NemotronHLayerWeights::Attention {
                    norm_w,
                    w_q: b.constant(&p("q_proj.weight"), TensorType::f32(vec![h, qd])),
                    w_k: b.constant(&p("k_proj.weight"), TensorType::f32(vec![h, kvd])),
                    w_v: b.constant(&p("v_proj.weight"), TensorType::f32(vec![h, kvd])),
                    w_o: b.constant(&p("o_proj.weight"), TensorType::f32(vec![qd, h])),
                });
            }
            NemotronHLayerKind::Mlp => {
                weights.push(NemotronHLayerWeights::Mlp {
                    norm_w,
                    w_up: b.constant(
                        &p("up_proj.weight"),
                        TensorType::f32(vec![h, cfg.mlp_inter]),
                    ),
                    w_down: b.constant(
                        &p("down_proj.weight"),
                        TensorType::f32(vec![cfg.mlp_inter, h]),
                    ),
                });
            }
        }
    }

    let (out, _states) = trace_nemotron_h_hybrid_stack_prefill(
        &b,
        &cfg.pattern,
        &weights,
        x0,
        &cfg.mamba,
        &cfg.attn,
        mamba_tril,
        attn_causal_mask,
        cfg.eps,
    );

    let norm_f_w = b.constant("backbone.norm_f.weight", TensorType::f32(vec![h]));
    let normed = rmsnorm(&b, out, norm_f_w, cfg.eps);
    let last = b.slice(normed, 1, l - 1, l); // only the last position feeds the LM head
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab_size]));
    let logits = linear(&b, last, lm_head, None);
    b.finish(logits)
}

/// Top-level Nemotron-H decode entry point: a single fixed-`cap` graph, built once and replayed for every
/// generated token (like `crate::deepseek4::trace_deepseek4_hybrid_stack_decode` and
/// `crate::qwen38::trace_qwen38_decode`). Embeds one token, dispatches each layer by
/// [`parse_hybrid_pattern`] kind as in [`trace_nemotron_h_hybrid_stack_prefill`], then final norm and
/// `lm_head`.
///
/// [`nemotron_h_mamba_layer`] (fixed-size `conv_cache[1,K-1,ConvC]` and `h_in[1,Hq,N,P]`, independent of
/// `cap` and position) and [`nemotron_h_mlp_layer`] (stateless) are reused unchanged. The attention mixer
/// uses [`nemotron_h_attention_layer_decode_masked`] instead, because [`nemotron_h_attention_layer`] bakes
/// `pos` in at trace time.
///
/// Unlike QSA's decode tracer, `cap` has no chunk-multiple or compression-rate constraint. This avoids
/// `Runner::generate`'s O(n) CPU re-prefill per step; it does not fix a correctness bug.
///
/// Same named constants as [`trace_nemotron_h_prefill`] (`backbone.embeddings.weight`, `layers.{i}.*`,
/// `backbone.norm_f.weight`, `lm_head.weight`); the prefill-only mask step input and SSD tril
/// computation have no decode counterpart. The additive decode mask
/// is the standard `Slot::Mask` `[cap]` row `Runner::bind_decode` synthesizes (`decode_mask_row`), so
/// `Runner::bind_decode` needs no new arm.
pub fn trace_nemotron_h_decode(cfg: &NemotronHConfig, cap: usize) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;

    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let _seq_len = b.slot(Slot::SeqLen, TensorType::scalar(DType::I32));
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![cap]));
    let mask = b.reshape(mask, vec![1, 1, 1, cap]);

    let embed = b.constant(
        "backbone.embeddings.weight",
        TensorType::f32(vec![cfg.vocab_size, h]),
    );
    let x0 = b.gather_scalar(embed, 0, token);
    let mut cur = b.reshape(x0, vec![1, 1, h]);

    let mut state: Vec<(Traced, Traced)> = Vec::new();
    for (li, kind) in cfg.pattern.iter().enumerate() {
        let norm_w = b.constant(
            &format!("layers.{li}.norm.weight"),
            TensorType::f32(vec![h]),
        );
        let p = |s: &str| format!("layers.{li}.mixer.{s}");
        match kind {
            NemotronHLayerKind::Mamba => {
                let inner = cfg.mamba.inner();
                let conv_c = cfg.mamba.conv_channels();
                let hq = cfg.mamba.mamba_num_heads;
                let k = cfg.mamba.conv_kernel;
                let w_z = b.constant(&p("z.weight"), TensorType::f32(vec![h, inner]));
                let w_xbc = b.constant(&p("xbc.weight"), TensorType::f32(vec![h, conv_c]));
                let w_dt = b.constant(&p("dt.weight"), TensorType::f32(vec![h, hq]));
                let dt_bias = b.constant(&p("dt_bias"), TensorType::f32(vec![hq]));
                let w_conv = b.constant(&p("conv1d.weight"), TensorType::f32(vec![k, conv_c]));
                let conv_bias = b.constant(&p("conv1d.bias"), TensorType::f32(vec![conv_c]));
                let a_param = b.constant(&p("a"), TensorType::f32(vec![1, hq, 1, 1]));
                let d_param = b.constant(&p("d"), TensorType::f32(vec![1, hq, 1, 1]));
                let gate_norm_w = b.constant(&p("gate_norm.weight"), TensorType::f32(vec![inner]));
                let w_out = b.constant(&p("out_proj.weight"), TensorType::f32(vec![inner, h]));
                let conv_in = b.state_input(
                    &format!("layers.{li}.mixer.conv_cache"),
                    TensorType::f32(vec![1, k - 1, conv_c]),
                    StateRole::Recurrent,
                );
                let h_in = b.state_input(
                    &format!("layers.{li}.mixer.ssm_state"),
                    TensorType::f32(vec![1, hq, cfg.mamba.ssm_state, cfg.mamba.mamba_head_dim]),
                    StateRole::Recurrent,
                );
                let (out, conv_out, h_out) = nemotron_h_mamba_layer(
                    &b,
                    cur,
                    &cfg.mamba,
                    norm_w,
                    w_z,
                    w_xbc,
                    w_dt,
                    dt_bias,
                    w_conv,
                    conv_bias,
                    a_param,
                    d_param,
                    gate_norm_w,
                    w_out,
                    conv_in,
                    h_in,
                    cfg.eps,
                );
                cur = out;
                state.push((conv_in, conv_out));
                state.push((h_in, h_out));
            }
            NemotronHLayerKind::Attention => {
                let qd = cfg.attn.num_heads * cfg.attn.head_dim;
                let kvd = cfg.attn.num_kv_heads * cfg.attn.head_dim;
                let w_q = b.constant(&p("q_proj.weight"), TensorType::f32(vec![h, qd]));
                let w_k = b.constant(&p("k_proj.weight"), TensorType::f32(vec![h, kvd]));
                let w_v = b.constant(&p("v_proj.weight"), TensorType::f32(vec![h, kvd]));
                let w_o = b.constant(&p("o_proj.weight"), TensorType::f32(vec![qd, h]));
                let k_cache = b.state_input(
                    &format!("layers.{li}.mixer.k_cache"),
                    TensorType::f32(vec![1, cfg.attn.num_kv_heads, cap, cfg.attn.head_dim]),
                    StateRole::Recurrent,
                );
                let v_cache = b.state_input(
                    &format!("layers.{li}.mixer.v_cache"),
                    TensorType::f32(vec![1, cfg.attn.num_kv_heads, cap, cfg.attn.head_dim]),
                    StateRole::Recurrent,
                );
                let (out, k_out, v_out) = nemotron_h_attention_layer_decode_masked(
                    &b, cur, &cfg.attn, norm_w, w_q, w_k, w_v, w_o, k_cache, v_cache, pos_slot,
                    mask, cfg.eps,
                );
                cur = out;
                state.push((k_cache, k_out));
                state.push((v_cache, v_out));
            }
            NemotronHLayerKind::Mlp => {
                let w_up = b.constant(
                    &p("up_proj.weight"),
                    TensorType::f32(vec![h, cfg.mlp_inter]),
                );
                let w_down = b.constant(
                    &p("down_proj.weight"),
                    TensorType::f32(vec![cfg.mlp_inter, h]),
                );
                cur = nemotron_h_mlp_layer(&b, cur, norm_w, w_up, w_down, cfg.eps);
            }
        }
    }

    let norm_f_w = b.constant("backbone.norm_f.weight", TensorType::f32(vec![h]));
    let normed = rmsnorm(&b, cur, norm_f_w, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab_size]));
    let logits = linear(&b, normed, lm_head, None);
    b.finish_with_state(logits, &state)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_hybrid_pattern` against the verbatim `hybrid_override_pattern` strings from
    /// `nvidia/Nemotron-H-8B-Base-8K/config.json` and `nvidia/Nemotron-H-56B-Base-8K/config.json`, whose
    /// lengths are checked against each config's `num_hidden_layers` (a hand-transcribed 56B string was
    /// once 107 chars instead of 118).
    const PATTERN_8B: &str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-";
    const PATTERN_56B: &str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-";

    #[test]
    fn parse_hybrid_pattern_matches_real_8b_config() {
        let layers = parse_hybrid_pattern(PATTERN_8B);
        assert_eq!(
            layers.len(),
            52,
            "matches real config's num_hidden_layers: 52"
        );
        let attn = layers
            .iter()
            .filter(|k| **k == NemotronHLayerKind::Attention)
            .count();
        let mamba = layers
            .iter()
            .filter(|k| **k == NemotronHLayerKind::Mamba)
            .count();
        let mlp = layers
            .iter()
            .filter(|k| **k == NemotronHLayerKind::Mlp)
            .count();
        assert_eq!(attn, 4, "4 '*' chars in the real 8B pattern string");
        assert_eq!(mamba, 24);
        assert_eq!(mlp, 24);
        assert_eq!(attn + mamba + mlp, 52);
    }

    #[test]
    fn parse_hybrid_pattern_matches_real_56b_config() {
        let layers = parse_hybrid_pattern(PATTERN_56B);
        assert_eq!(
            layers.len(),
            118,
            "matches real config's num_hidden_layers: 118"
        );
        let attn = layers
            .iter()
            .filter(|k| **k == NemotronHLayerKind::Attention)
            .count();
        assert_eq!(attn, 10, "10 '*' chars in the real 56B pattern string");
    }

    #[test]
    #[should_panic(expected = "unexpected layer char")]
    fn parse_hybrid_pattern_rejects_unknown_char() {
        parse_hybrid_pattern("M-X");
    }

    mod cpu_oracle {
        use super::*;
        use poot_eval::{EvalBudget, EvalOptions, Value, eval};
        use poot_graph_ir::types::TensorType;
        use poot_tensor::HostTensor;
        use std::collections::HashMap;

        use poot_test_util::fill;

        use poot_test_util::seed_of;

        fn eps_silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }
        fn eps_relu(x: f32) -> f32 {
            x.max(0.0)
        }
        fn eps_softplus(x: f32) -> f32 {
            (1.0 + x.exp()).ln()
        }

        fn mm(x: &[f32], w: &[f32], inn: usize, out: usize) -> Vec<f32> {
            (0..out)
                .map(|j| (0..inn).map(|i| x[i] * w[i * out + j]).sum())
                .collect()
        }

        /// The toy 9-layer stack: the first 9 chars of the real 8B `hybrid_override_pattern` (`M-M-M-M*-`).
        const TOY_PATTERN: &str = "M-M-M-M*-";

        struct ToyWeights {
            // per-layer-index weights, only the fields relevant to that layer's kind are populated.
            norm: Vec<Vec<f32>>,
            // mamba
            m_wz: HashMap<usize, Vec<f32>>,
            m_wxbc: HashMap<usize, Vec<f32>>,
            m_wdt: HashMap<usize, Vec<f32>>,
            m_dtbias: HashMap<usize, Vec<f32>>,
            m_wconv: HashMap<usize, Vec<f32>>,
            m_convbias: HashMap<usize, Vec<f32>>,
            m_a: HashMap<usize, Vec<f32>>,
            m_d: HashMap<usize, Vec<f32>>,
            m_gatenorm: HashMap<usize, Vec<f32>>,
            m_wout: HashMap<usize, Vec<f32>>,
            // attention
            a_wq: HashMap<usize, Vec<f32>>,
            a_wk: HashMap<usize, Vec<f32>>,
            a_wv: HashMap<usize, Vec<f32>>,
            a_wo: HashMap<usize, Vec<f32>>,
            // mlp
            f_wup: HashMap<usize, Vec<f32>>,
            f_wdown: HashMap<usize, Vec<f32>>,
        }

        fn gen_weights(
            pattern: &[NemotronHLayerKind],
            mcfg: &NemotronHMambaConfig,
            acfg: &NemotronHAttnConfig,
            hidden: usize,
            mlp_inter: usize,
        ) -> ToyWeights {
            let inner = mcfg.inner();
            let conv_c = mcfg.conv_channels();
            let mut w = ToyWeights {
                norm: Vec::new(),
                m_wz: HashMap::new(),
                m_wxbc: HashMap::new(),
                m_wdt: HashMap::new(),
                m_dtbias: HashMap::new(),
                m_wconv: HashMap::new(),
                m_convbias: HashMap::new(),
                m_a: HashMap::new(),
                m_d: HashMap::new(),
                m_gatenorm: HashMap::new(),
                m_wout: HashMap::new(),
                a_wq: HashMap::new(),
                a_wk: HashMap::new(),
                a_wv: HashMap::new(),
                a_wo: HashMap::new(),
                f_wup: HashMap::new(),
                f_wdown: HashMap::new(),
            };
            for (li, kind) in pattern.iter().enumerate() {
                w.norm.push(fill(hidden, seed_of(&format!("l{li}.norm"))));
                match kind {
                    NemotronHLayerKind::Mamba => {
                        w.m_wz
                            .insert(li, fill(hidden * inner, seed_of(&format!("l{li}.mz"))));
                        w.m_wxbc
                            .insert(li, fill(hidden * conv_c, seed_of(&format!("l{li}.mxbc"))));
                        w.m_wdt.insert(
                            li,
                            fill(
                                hidden * mcfg.mamba_num_heads,
                                seed_of(&format!("l{li}.mdt")),
                            ),
                        );
                        w.m_dtbias.insert(
                            li,
                            fill(mcfg.mamba_num_heads, seed_of(&format!("l{li}.mdtb"))),
                        );
                        w.m_wconv.insert(
                            li,
                            fill(mcfg.conv_kernel * conv_c, seed_of(&format!("l{li}.mconv"))),
                        );
                        w.m_convbias
                            .insert(li, fill(conv_c, seed_of(&format!("l{li}.mconvb"))));
                        let a: Vec<f32> = (0..mcfg.mamba_num_heads)
                            .map(|h| -0.4 - 0.1 * h as f32)
                            .collect();
                        let d: Vec<f32> = (0..mcfg.mamba_num_heads)
                            .map(|h| 0.15 + 0.05 * h as f32)
                            .collect();
                        w.m_a.insert(li, a);
                        w.m_d.insert(li, d);
                        // near-1 but not identically 1, so the gated-rmsnorm gate order is numerically visible.
                        let gn: Vec<f32> = fill(inner, seed_of(&format!("l{li}.mgn")))
                            .iter()
                            .map(|v| 1.0 + v * 0.2)
                            .collect();
                        w.m_gatenorm.insert(li, gn);
                        w.m_wout
                            .insert(li, fill(inner * hidden, seed_of(&format!("l{li}.mout"))));
                    }
                    NemotronHLayerKind::Attention => {
                        let qd = acfg.num_heads * acfg.head_dim;
                        let kvd = acfg.num_kv_heads * acfg.head_dim;
                        w.a_wq
                            .insert(li, fill(hidden * qd, seed_of(&format!("l{li}.aq"))));
                        w.a_wk
                            .insert(li, fill(hidden * kvd, seed_of(&format!("l{li}.ak"))));
                        w.a_wv
                            .insert(li, fill(hidden * kvd, seed_of(&format!("l{li}.av"))));
                        w.a_wo
                            .insert(li, fill(qd * hidden, seed_of(&format!("l{li}.ao"))));
                    }
                    NemotronHLayerKind::Mlp => {
                        w.f_wup
                            .insert(li, fill(hidden * mlp_inter, seed_of(&format!("l{li}.fup"))));
                        w.f_wdown.insert(
                            li,
                            fill(mlp_inter * hidden, seed_of(&format!("l{li}.fdown"))),
                        );
                    }
                }
            }
            w
        }

        /// Per-layer recurrent state of the hand reference, mirroring the traced graph's state tensors (conv
        /// history + SSM state per Mamba layer, k/v history per attention layer).
        enum RefState {
            Mamba {
                xbc_hist: Vec<Vec<f32>>,
                ssm: Vec<f32>,
            },
            Attention {
                k_hist: Vec<Vec<f32>>,
                v_hist: Vec<Vec<f32>>,
            },
            Mlp,
        }

        #[allow(clippy::too_many_arguments)]
        fn run_reference(
            pattern: &[NemotronHLayerKind],
            mcfg: &NemotronHMambaConfig,
            acfg: &NemotronHAttnConfig,
            hidden: usize,
            mlp_inter: usize,
            eps: f32,
            w: &ToyWeights,
            xs: &[Vec<f32>], // per-timestep hidden-state input, len L, each [hidden]
        ) -> Vec<Vec<f32>> {
            let (hq, p, n, g) = (
                mcfg.mamba_num_heads,
                mcfg.mamba_head_dim,
                mcfg.ssm_state,
                mcfg.n_groups,
            );
            let n_rep_m = mcfg.heads_per_group();
            let inner = mcfg.inner();
            let (aq, akv, d) = (acfg.num_heads, acfg.num_kv_heads, acfg.head_dim);
            let n_rep_a = aq / akv;
            let scale = 1.0 / (d as f32).sqrt();

            let mut state: Vec<RefState> = pattern
                .iter()
                .map(|k| match k {
                    NemotronHLayerKind::Mamba => RefState::Mamba {
                        xbc_hist: Vec::new(),
                        ssm: vec![0.0; hq * n * p],
                    },
                    NemotronHLayerKind::Attention => RefState::Attention {
                        k_hist: Vec::new(),
                        v_hist: Vec::new(),
                    },
                    NemotronHLayerKind::Mlp => RefState::Mlp,
                })
                .collect();

            let mut outputs = Vec::new();
            for xt in xs {
                let mut x = xt.clone();
                for (li, kind) in pattern.iter().enumerate() {
                    // shared pre-norm (RMSNorm, same eps for all layer kinds).
                    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
                    let denom = (ms + eps).sqrt();
                    let xn: Vec<f32> = x
                        .iter()
                        .zip(&w.norm[li])
                        .map(|(v, g)| (v / denom) * g)
                        .collect();

                    match kind {
                        NemotronHLayerKind::Mamba => {
                            let RefState::Mamba { xbc_hist, ssm } = &mut state[li] else {
                                unreachable!()
                            };
                            let conv_c = mcfg.conv_channels();
                            let z = mm(&xn, &w.m_wz[&li], hidden, inner);
                            let xbc_raw = mm(&xn, &w.m_wxbc[&li], hidden, conv_c);
                            let mut dt_raw = mm(&xn, &w.m_wdt[&li], hidden, hq);
                            for (h, dt) in dt_raw.iter_mut().enumerate() {
                                *dt += w.m_dtbias[&li][h];
                            }
                            xbc_hist.push(xbc_raw.clone());
                            let kk = mcfg.conv_kernel;
                            let wconv = &w.m_wconv[&li];
                            let convbias = &w.m_convbias[&li];
                            let mut xbc_act = vec![0.0f32; conv_c];
                            let t = xbc_hist.len() - 1;
                            for (c, slot) in xbc_act.iter_mut().enumerate() {
                                let mut acc = convbias[c];
                                for k in 0..kk {
                                    let idx = t as isize - (kk as isize - 1) + k as isize;
                                    if idx >= 0 {
                                        acc += wconv[k * conv_c + c] * xbc_hist[idx as usize][c];
                                    }
                                }
                                *slot = eps_silu(acc);
                            }
                            let x_part = &xbc_act[0..inner];
                            let b_part = &xbc_act[inner..inner + g * n];
                            let c_part = &xbc_act[inner + g * n..inner + 2 * g * n];

                            let a_par = &w.m_a[&li];
                            let d_par = &w.m_d[&li];
                            let mut y = vec![0.0f32; inner];
                            for h in 0..hq {
                                let grp = h / n_rep_m;
                                let delta_h = eps_softplus(dt_raw[h]);
                                let a_bar = (delta_h * a_par[h]).exp();
                                for nn in 0..n {
                                    let b_val = b_part[grp * n + nn];
                                    let b_bar = delta_h * b_val;
                                    for pp in 0..p {
                                        let xv = x_part[h * p + pp];
                                        let idx = (h * n + nn) * p + pp;
                                        ssm[idx] = a_bar * ssm[idx] + b_bar * xv;
                                    }
                                }
                                for pp in 0..p {
                                    let mut acc = 0.0f32;
                                    for nn in 0..n {
                                        acc += c_part[grp * n + nn] * ssm[(h * n + nn) * p + pp];
                                    }
                                    y[h * p + pp] = acc + d_par[h] * x_part[h * p + pp];
                                }
                            }
                            // gate before the rmsnorm reduction.
                            let gated: Vec<f32> = y
                                .iter()
                                .zip(&z)
                                .map(|(yv, zv)| yv * eps_silu(*zv))
                                .collect();
                            let gms: f32 = gated.iter().map(|v| v * v).sum::<f32>() / inner as f32;
                            let gdenom = (gms + eps).sqrt();
                            let gn = &w.m_gatenorm[&li];
                            let normed: Vec<f32> = gated
                                .iter()
                                .zip(gn)
                                .map(|(v, gg)| (v / gdenom) * gg)
                                .collect();
                            let out = mm(&normed, &w.m_wout[&li], inner, hidden);
                            for j in 0..hidden {
                                x[j] += out[j];
                            }
                        }
                        NemotronHLayerKind::Attention => {
                            let RefState::Attention { k_hist, v_hist } = &mut state[li] else {
                                unreachable!()
                            };
                            let qd = aq * d;
                            let kvd = akv * d;
                            let q = mm(&xn, &w.a_wq[&li], hidden, qd);
                            let k = mm(&xn, &w.a_wk[&li], hidden, kvd);
                            let v = mm(&xn, &w.a_wv[&li], hidden, kvd);
                            k_hist.push(k);
                            v_hist.push(v);
                            let l = k_hist.len();
                            let mut o = vec![0.0f32; qd];
                            for h in 0..aq {
                                let kv = h / n_rep_a;
                                let mut scores = vec![0.0f32; l];
                                for (t, kt) in k_hist.iter().enumerate() {
                                    let mut dot = 0.0f32;
                                    for e in 0..d {
                                        dot += q[h * d + e] * kt[kv * d + e];
                                    }
                                    scores[t] = dot * scale;
                                }
                                let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                                let exps: Vec<f32> =
                                    scores.iter().map(|s| (s - mx).exp()).collect();
                                let sum: f32 = exps.iter().sum();
                                for (t, vt) in v_hist.iter().enumerate() {
                                    let wgt = exps[t] / sum;
                                    for e in 0..d {
                                        o[h * d + e] += wgt * vt[kv * d + e];
                                    }
                                }
                            }
                            let out = mm(&o, &w.a_wo[&li], qd, hidden);
                            for j in 0..hidden {
                                x[j] += out[j];
                            }
                        }
                        NemotronHLayerKind::Mlp => {
                            let up = mm(&xn, &w.f_wup[&li], hidden, mlp_inter);
                            let sq: Vec<f32> = up.iter().map(|v| eps_relu(*v).powi(2)).collect();
                            let out = mm(&sq, &w.f_wdown[&li], mlp_inter, hidden);
                            for j in 0..hidden {
                                x[j] += out[j];
                            }
                        }
                    }
                }
                outputs.push(x);
            }
            outputs
        }

        /// Build and run the traced decode loop for the toy stack, comparing every step against
        /// `run_reference`. `n_groups` is a parameter so one code path covers the grouped case
        /// (`n_groups < mamba_num_heads`) and the degenerate one (`n_groups == mamba_num_heads`, `repeat_kv`
        /// with `n_rep=1`).
        fn run_traced_and_compare(n_groups: usize) {
            let hidden = 6usize;
            let mcfg = NemotronHMambaConfig {
                hidden,
                mamba_num_heads: 4,
                mamba_head_dim: 2,
                n_groups,
                ssm_state: 3,
                conv_kernel: 3,
            };
            let acfg = NemotronHAttnConfig {
                hidden,
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 2,
            };
            let mlp_inter = 10usize;
            let eps = 1e-6f32;
            let cap = 4usize;
            let pattern = parse_hybrid_pattern(TOY_PATTERN);
            assert_eq!(
                pattern,
                vec![
                    NemotronHLayerKind::Mamba,
                    NemotronHLayerKind::Mlp,
                    NemotronHLayerKind::Mamba,
                    NemotronHLayerKind::Mlp,
                    NemotronHLayerKind::Mamba,
                    NemotronHLayerKind::Mlp,
                    NemotronHLayerKind::Mamba,
                    NemotronHLayerKind::Attention,
                    NemotronHLayerKind::Mlp,
                ],
                "TOY_PATTERN is the literal first 9 chars of the real 8B hybrid_override_pattern"
            );
            let w = gen_weights(&pattern, &mcfg, &acfg, hidden, mlp_inter);

            let xs: Vec<Vec<f32>> = (0..cap)
                .map(|t| fill(hidden, seed_of(&format!("x{t}"))))
                .collect();

            let want = run_reference(&pattern, &mcfg, &acfg, hidden, mlp_inter, eps, &w, &xs);

            // per-layer running state for the traced-graph decode loop.
            let inner = mcfg.inner();
            let conv_c = mcfg.conv_channels();
            let mut mamba_conv: HashMap<usize, HostTensor> = HashMap::new();
            let mut mamba_ssm: HashMap<usize, HostTensor> = HashMap::new();
            let mut attn_k: HashMap<usize, HostTensor> = HashMap::new();
            let mut attn_v: HashMap<usize, HostTensor> = HashMap::new();
            for (li, kind) in pattern.iter().enumerate() {
                match kind {
                    NemotronHLayerKind::Mamba => {
                        mamba_conv.insert(
                            li,
                            HostTensor::f32(
                                vec![1, mcfg.conv_kernel - 1, conv_c],
                                vec![0.0; (mcfg.conv_kernel - 1) * conv_c],
                            ),
                        );
                        mamba_ssm.insert(
                            li,
                            HostTensor::f32(
                                vec![1, mcfg.mamba_num_heads, mcfg.ssm_state, mcfg.mamba_head_dim],
                                vec![
                                    0.0;
                                    mcfg.mamba_num_heads * mcfg.ssm_state * mcfg.mamba_head_dim
                                ],
                            ),
                        );
                    }
                    NemotronHLayerKind::Attention => {
                        attn_k.insert(
                            li,
                            HostTensor::f32(
                                vec![1, acfg.num_kv_heads, cap, acfg.head_dim],
                                vec![0.0; acfg.num_kv_heads * cap * acfg.head_dim],
                            ),
                        );
                        attn_v.insert(
                            li,
                            HostTensor::f32(
                                vec![1, acfg.num_kv_heads, cap, acfg.head_dim],
                                vec![0.0; acfg.num_kv_heads * cap * acfg.head_dim],
                            ),
                        );
                    }
                    NemotronHLayerKind::Mlp => {}
                }
            }

            let mut got = Vec::new();
            for (pos, xt) in xs.iter().enumerate() {
                let b = Builder::new();
                let x0 = b.constant("x", TensorType::f32(vec![1, 1, hidden]));

                // per-layer constant weight handles + state_inputs, rebuilt each step (retrace-per-step, as in
                // poot-eval's `hybrid_model_decode_loop_matches_prefill`).
                let mut cur = x0;
                let mut state_pairs: Vec<(Traced, Traced)> = Vec::new();
                let mut mconv_out: HashMap<usize, Traced> = HashMap::new();
                let mut mssm_out: HashMap<usize, Traced> = HashMap::new();
                let mut ak_out: HashMap<usize, Traced> = HashMap::new();
                let mut av_out: HashMap<usize, Traced> = HashMap::new();
                let mut inp: HashMap<poot_graph_ir::ValueId, HostTensor> = HashMap::new();
                inp.insert(x0.id, HostTensor::f32(vec![1, 1, hidden], xt.clone()));

                for (li, kind) in pattern.iter().enumerate() {
                    let norm_w = b.constant(&format!("l{li}.norm"), TensorType::f32(vec![hidden]));
                    inp.insert(norm_w.id, HostTensor::f32(vec![hidden], w.norm[li].clone()));
                    match kind {
                        NemotronHLayerKind::Mamba => {
                            let wz = b.constant(
                                &format!("l{li}.wz"),
                                TensorType::f32(vec![hidden, inner]),
                            );
                            let wxbc = b.constant(
                                &format!("l{li}.wxbc"),
                                TensorType::f32(vec![hidden, conv_c]),
                            );
                            let wdt = b.constant(
                                &format!("l{li}.wdt"),
                                TensorType::f32(vec![hidden, mcfg.mamba_num_heads]),
                            );
                            let dtb = b.constant(
                                &format!("l{li}.dtb"),
                                TensorType::f32(vec![mcfg.mamba_num_heads]),
                            );
                            let wconv = b.constant(
                                &format!("l{li}.wconv"),
                                TensorType::f32(vec![mcfg.conv_kernel, conv_c]),
                            );
                            let convb =
                                b.constant(&format!("l{li}.convb"), TensorType::f32(vec![conv_c]));
                            let ap = b.constant(
                                &format!("l{li}.a"),
                                TensorType::f32(vec![1, mcfg.mamba_num_heads, 1, 1]),
                            );
                            let dp = b.constant(
                                &format!("l{li}.d"),
                                TensorType::f32(vec![1, mcfg.mamba_num_heads, 1, 1]),
                            );
                            let gn = b.constant(&format!("l{li}.gn"), TensorType::f32(vec![inner]));
                            let wout = b.constant(
                                &format!("l{li}.wout"),
                                TensorType::f32(vec![inner, hidden]),
                            );
                            let conv_in = b.state_input(
                                &format!("l{li}.conv"),
                                TensorType::f32(vec![1, mcfg.conv_kernel - 1, conv_c]),
                                StateRole::Recurrent,
                            );
                            let ssm_in = b.state_input(
                                &format!("l{li}.ssm"),
                                TensorType::f32(vec![
                                    1,
                                    mcfg.mamba_num_heads,
                                    mcfg.ssm_state,
                                    mcfg.mamba_head_dim,
                                ]),
                                StateRole::Recurrent,
                            );
                            inp.insert(
                                wz.id,
                                HostTensor::f32(vec![hidden, inner], w.m_wz[&li].clone()),
                            );
                            inp.insert(
                                wxbc.id,
                                HostTensor::f32(vec![hidden, conv_c], w.m_wxbc[&li].clone()),
                            );
                            inp.insert(
                                wdt.id,
                                HostTensor::f32(
                                    vec![hidden, mcfg.mamba_num_heads],
                                    w.m_wdt[&li].clone(),
                                ),
                            );
                            inp.insert(
                                dtb.id,
                                HostTensor::f32(
                                    vec![mcfg.mamba_num_heads],
                                    w.m_dtbias[&li].clone(),
                                ),
                            );
                            inp.insert(
                                wconv.id,
                                HostTensor::f32(
                                    vec![mcfg.conv_kernel, conv_c],
                                    w.m_wconv[&li].clone(),
                                ),
                            );
                            inp.insert(
                                convb.id,
                                HostTensor::f32(vec![conv_c], w.m_convbias[&li].clone()),
                            );
                            inp.insert(
                                ap.id,
                                HostTensor::f32(
                                    vec![1, mcfg.mamba_num_heads, 1, 1],
                                    w.m_a[&li].clone(),
                                ),
                            );
                            inp.insert(
                                dp.id,
                                HostTensor::f32(
                                    vec![1, mcfg.mamba_num_heads, 1, 1],
                                    w.m_d[&li].clone(),
                                ),
                            );
                            inp.insert(
                                gn.id,
                                HostTensor::f32(vec![inner], w.m_gatenorm[&li].clone()),
                            );
                            inp.insert(
                                wout.id,
                                HostTensor::f32(vec![inner, hidden], w.m_wout[&li].clone()),
                            );
                            inp.insert(conv_in.id, mamba_conv[&li].clone());
                            inp.insert(ssm_in.id, mamba_ssm[&li].clone());

                            let (out, conv_out, ssm_out) = nemotron_h_mamba_layer(
                                &b, cur, &mcfg, norm_w, wz, wxbc, wdt, dtb, wconv, convb, ap, dp,
                                gn, wout, conv_in, ssm_in, eps,
                            );
                            cur = out;
                            mconv_out.insert(li, conv_out);
                            mssm_out.insert(li, ssm_out);
                            state_pairs.push((conv_in, conv_out));
                            state_pairs.push((ssm_in, ssm_out));
                        }
                        NemotronHLayerKind::Attention => {
                            let qd = acfg.num_heads * acfg.head_dim;
                            let kvd = acfg.num_kv_heads * acfg.head_dim;
                            let wq =
                                b.constant(&format!("l{li}.wq"), TensorType::f32(vec![hidden, qd]));
                            let wk = b
                                .constant(&format!("l{li}.wk"), TensorType::f32(vec![hidden, kvd]));
                            let wv = b
                                .constant(&format!("l{li}.wv"), TensorType::f32(vec![hidden, kvd]));
                            let wo =
                                b.constant(&format!("l{li}.wo"), TensorType::f32(vec![qd, hidden]));
                            let kc_in = b.state_input(
                                &format!("l{li}.kc"),
                                TensorType::f32(vec![1, acfg.num_kv_heads, cap, acfg.head_dim]),
                                StateRole::Recurrent,
                            );
                            let vc_in = b.state_input(
                                &format!("l{li}.vc"),
                                TensorType::f32(vec![1, acfg.num_kv_heads, cap, acfg.head_dim]),
                                StateRole::Recurrent,
                            );
                            inp.insert(
                                wq.id,
                                HostTensor::f32(vec![hidden, qd], w.a_wq[&li].clone()),
                            );
                            inp.insert(
                                wk.id,
                                HostTensor::f32(vec![hidden, kvd], w.a_wk[&li].clone()),
                            );
                            inp.insert(
                                wv.id,
                                HostTensor::f32(vec![hidden, kvd], w.a_wv[&li].clone()),
                            );
                            inp.insert(
                                wo.id,
                                HostTensor::f32(vec![qd, hidden], w.a_wo[&li].clone()),
                            );
                            inp.insert(kc_in.id, attn_k[&li].clone());
                            inp.insert(vc_in.id, attn_v[&li].clone());

                            let (out, kc_out, vc_out) = nemotron_h_attention_layer(
                                &b, cur, &acfg, norm_w, wq, wk, wv, wo, kc_in, vc_in, pos, eps,
                            );
                            cur = out;
                            ak_out.insert(li, kc_out);
                            av_out.insert(li, vc_out);
                            state_pairs.push((kc_in, kc_out));
                            state_pairs.push((vc_in, vc_out));
                        }
                        NemotronHLayerKind::Mlp => {
                            let wup = b.constant(
                                &format!("l{li}.wup"),
                                TensorType::f32(vec![hidden, mlp_inter]),
                            );
                            let wdown = b.constant(
                                &format!("l{li}.wdown"),
                                TensorType::f32(vec![mlp_inter, hidden]),
                            );
                            inp.insert(
                                wup.id,
                                HostTensor::f32(vec![hidden, mlp_inter], w.f_wup[&li].clone()),
                            );
                            inp.insert(
                                wdown.id,
                                HostTensor::f32(vec![mlp_inter, hidden], w.f_wdown[&li].clone()),
                            );
                            cur = nemotron_h_mlp_layer(&b, cur, norm_w, wup, wdown, eps);
                        }
                    }
                }

                let g = b.finish_with_state(cur, &state_pairs);
                let inp: HashMap<poot_graph_ir::ValueId, Value> =
                    inp.into_iter().map(|(k, v)| (k, v.into())).collect();
                let step = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                    .expect("nemotron_h decode step eval");
                let out_t = step.output.into_host().expect("dense output");
                let new_states: Vec<HostTensor> = step
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("dense state"))
                    .collect();
                got.push(out_t.as_f32().unwrap().to_vec());

                // state_pairs was pushed in layer order; walk it the same way to route results back.
                let mut si = 0usize;
                for (li, kind) in pattern.iter().enumerate() {
                    match kind {
                        NemotronHLayerKind::Mamba => {
                            mamba_conv.insert(li, new_states[si].clone());
                            mamba_ssm.insert(li, new_states[si + 1].clone());
                            si += 2;
                        }
                        NemotronHLayerKind::Attention => {
                            attn_k.insert(li, new_states[si].clone());
                            attn_v.insert(li, new_states[si + 1].clone());
                            si += 2;
                        }
                        NemotronHLayerKind::Mlp => {}
                    }
                }
            }

            for (t, (g, w)) in got.iter().zip(&want).enumerate() {
                for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
                    let diff = (a - b).abs();
                    let tol = 1e-4 * b.abs().max(1.0);
                    assert!(
                        diff <= tol,
                        "step {t} element {i}: got {a} want {b} (n_groups={n_groups})"
                    );
                }
            }
        }

        #[test]
        fn nemotron_h_hybrid_stack_decode_matches_reference_grouped_bc() {
            // n_groups=2 < mamba_num_heads=4: the grouped B/C repeat.
            run_traced_and_compare(2);
        }

        #[test]
        fn nemotron_h_hybrid_stack_decode_matches_reference_degenerate_groups() {
            // n_groups == mamba_num_heads (repeat_kv n_rep=1, a no-op broadcast): the grouped composition
            // reduces to the ungrouped case.
            run_traced_and_compare(4);
        }

        #[test]
        fn nemotron_h_mlp_layer_prefill_matches_per_token_decode() {
            // nemotron_h_mlp_layer is stateless and position-free, so one call on [1,L,H] should equal L
            // calls on [1,1,H] slices concatenated.
            let (hidden, mlp_inter, l) = (6usize, 10usize, 5usize);
            let eps = 1e-6f32;
            let norm = fill(hidden, seed_of("mlp.norm"));
            let wup = fill(hidden * mlp_inter, seed_of("mlp.up"));
            let wdown = fill(mlp_inter * hidden, seed_of("mlp.down"));
            let xs: Vec<Vec<f32>> = (0..l)
                .map(|t| fill(hidden, seed_of(&format!("mlp.x{t}"))))
                .collect();

            // one call, L positions at once.
            let b = Builder::new();
            let x0 = b.constant("x", TensorType::f32(vec![1, l, hidden]));
            let norm_w = b.constant("norm", TensorType::f32(vec![hidden]));
            let w_up = b.constant("wup", TensorType::f32(vec![hidden, mlp_inter]));
            let w_down = b.constant("wdown", TensorType::f32(vec![mlp_inter, hidden]));
            let out = nemotron_h_mlp_layer(&b, x0, norm_w, w_up, w_down, eps);
            let g = b.finish(out);
            let mut xflat = Vec::with_capacity(l * hidden);
            for xt in &xs {
                xflat.extend_from_slice(xt);
            }
            let mut inp: HashMap<poot_graph_ir::ValueId, Value> = HashMap::new();
            inp.insert(x0.id, HostTensor::f32(vec![1, l, hidden], xflat).into());
            inp.insert(
                norm_w.id,
                HostTensor::f32(vec![hidden], norm.clone()).into(),
            );
            inp.insert(
                w_up.id,
                HostTensor::f32(vec![hidden, mlp_inter], wup.clone()).into(),
            );
            inp.insert(
                w_down.id,
                HostTensor::f32(vec![mlp_inter, hidden], wdown.clone()).into(),
            );
            let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .expect("mlp prefill eval")
                .output
                .into_host()
                .expect("dense output");

            // L separate [1,1,H] calls, same weights.
            let mut want = Vec::with_capacity(l * hidden);
            for xt in &xs {
                let b1 = Builder::new();
                let x1 = b1.constant("x", TensorType::f32(vec![1, 1, hidden]));
                let n1 = b1.constant("norm", TensorType::f32(vec![hidden]));
                let u1 = b1.constant("wup", TensorType::f32(vec![hidden, mlp_inter]));
                let d1 = b1.constant("wdown", TensorType::f32(vec![mlp_inter, hidden]));
                let o1 = nemotron_h_mlp_layer(&b1, x1, n1, u1, d1, eps);
                let g1 = b1.finish(o1);
                let mut inp1: HashMap<poot_graph_ir::ValueId, Value> = HashMap::new();
                inp1.insert(
                    x1.id,
                    HostTensor::f32(vec![1, 1, hidden], xt.clone()).into(),
                );
                inp1.insert(n1.id, HostTensor::f32(vec![hidden], norm.clone()).into());
                inp1.insert(
                    u1.id,
                    HostTensor::f32(vec![hidden, mlp_inter], wup.clone()).into(),
                );
                inp1.insert(
                    d1.id,
                    HostTensor::f32(vec![mlp_inter, hidden], wdown.clone()).into(),
                );
                let o = eval(&g1, &inp1, EvalOptions::new(EvalBudget::UNBOUNDED))
                    .expect("mlp decode-step eval")
                    .output
                    .into_host()
                    .expect("dense output");
                want.extend_from_slice(o.as_f32().unwrap());
            }

            poot_test_util::assert_close(got.as_f32().unwrap(), &want, 1e-5);
        }

        /// Prefill/decode consistency: one `trace_nemotron_h_hybrid_stack_prefill` graph over the toy 9-layer
        /// stack (same `TOY_PATTERN`, `gen_weights` and inputs as above) for all `cap` positions matches the
        /// already-verified traced decode graph stepped over the same tokens and weights. This is stronger
        /// than comparing against the hand reference alone, since that could share a mistake. Also checks
        /// the final per-layer state (SSM state + conv cache per Mamba layer, K/V per attention layer), which
        /// is what lets an engine hand off from prefill to decode.
        fn run_prefill_matches_decode_loop(n_groups: usize) {
            let hidden = 6usize;
            let mcfg = NemotronHMambaConfig {
                hidden,
                mamba_num_heads: 4,
                mamba_head_dim: 2,
                n_groups,
                ssm_state: 3,
                conv_kernel: 3,
            };
            let acfg = NemotronHAttnConfig {
                hidden,
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 2,
            };
            let mlp_inter = 10usize;
            let eps = 1e-6f32;
            let cap = 4usize;
            let pattern = parse_hybrid_pattern(TOY_PATTERN);
            let w = gen_weights(&pattern, &mcfg, &acfg, hidden, mlp_inter);
            let xs: Vec<Vec<f32>> = (0..cap)
                .map(|t| fill(hidden, seed_of(&format!("x{t}"))))
                .collect();

            let inner = mcfg.inner();
            let conv_c = mcfg.conv_channels();

            // --- (1) traced decode loop over `cap` steps, capturing the final per-layer state. ---
            let mut mamba_conv: HashMap<usize, HostTensor> = HashMap::new();
            let mut mamba_ssm: HashMap<usize, HostTensor> = HashMap::new();
            let mut attn_k: HashMap<usize, HostTensor> = HashMap::new();
            let mut attn_v: HashMap<usize, HostTensor> = HashMap::new();
            for (li, kind) in pattern.iter().enumerate() {
                match kind {
                    NemotronHLayerKind::Mamba => {
                        mamba_conv.insert(
                            li,
                            HostTensor::f32(
                                vec![1, mcfg.conv_kernel - 1, conv_c],
                                vec![0.0; (mcfg.conv_kernel - 1) * conv_c],
                            ),
                        );
                        mamba_ssm.insert(
                            li,
                            HostTensor::f32(
                                vec![1, mcfg.mamba_num_heads, mcfg.ssm_state, mcfg.mamba_head_dim],
                                vec![
                                    0.0;
                                    mcfg.mamba_num_heads * mcfg.ssm_state * mcfg.mamba_head_dim
                                ],
                            ),
                        );
                    }
                    NemotronHLayerKind::Attention => {
                        attn_k.insert(
                            li,
                            HostTensor::f32(
                                vec![1, acfg.num_kv_heads, cap, acfg.head_dim],
                                vec![0.0; acfg.num_kv_heads * cap * acfg.head_dim],
                            ),
                        );
                        attn_v.insert(
                            li,
                            HostTensor::f32(
                                vec![1, acfg.num_kv_heads, cap, acfg.head_dim],
                                vec![0.0; acfg.num_kv_heads * cap * acfg.head_dim],
                            ),
                        );
                    }
                    NemotronHLayerKind::Mlp => {}
                }
            }

            let mut decode_got: Vec<Vec<f32>> = Vec::new();
            for (pos, xt) in xs.iter().enumerate() {
                let b = Builder::new();
                let x0 = b.constant("x", TensorType::f32(vec![1, 1, hidden]));
                let mut cur = x0;
                let mut state_pairs: Vec<(Traced, Traced)> = Vec::new();
                let mut inp: HashMap<poot_graph_ir::ValueId, HostTensor> = HashMap::new();
                inp.insert(x0.id, HostTensor::f32(vec![1, 1, hidden], xt.clone()));

                for (li, kind) in pattern.iter().enumerate() {
                    let norm_w = b.constant(&format!("l{li}.norm"), TensorType::f32(vec![hidden]));
                    inp.insert(norm_w.id, HostTensor::f32(vec![hidden], w.norm[li].clone()));
                    match kind {
                        NemotronHLayerKind::Mamba => {
                            let wz = b.constant(
                                &format!("l{li}.wz"),
                                TensorType::f32(vec![hidden, inner]),
                            );
                            let wxbc = b.constant(
                                &format!("l{li}.wxbc"),
                                TensorType::f32(vec![hidden, conv_c]),
                            );
                            let wdt = b.constant(
                                &format!("l{li}.wdt"),
                                TensorType::f32(vec![hidden, mcfg.mamba_num_heads]),
                            );
                            let dtb = b.constant(
                                &format!("l{li}.dtb"),
                                TensorType::f32(vec![mcfg.mamba_num_heads]),
                            );
                            let wconv = b.constant(
                                &format!("l{li}.wconv"),
                                TensorType::f32(vec![mcfg.conv_kernel, conv_c]),
                            );
                            let convb =
                                b.constant(&format!("l{li}.convb"), TensorType::f32(vec![conv_c]));
                            let ap = b.constant(
                                &format!("l{li}.a"),
                                TensorType::f32(vec![1, mcfg.mamba_num_heads, 1, 1]),
                            );
                            let dp = b.constant(
                                &format!("l{li}.d"),
                                TensorType::f32(vec![1, mcfg.mamba_num_heads, 1, 1]),
                            );
                            let gn = b.constant(&format!("l{li}.gn"), TensorType::f32(vec![inner]));
                            let wout = b.constant(
                                &format!("l{li}.wout"),
                                TensorType::f32(vec![inner, hidden]),
                            );
                            let conv_in = b.state_input(
                                &format!("l{li}.conv"),
                                TensorType::f32(vec![1, mcfg.conv_kernel - 1, conv_c]),
                                StateRole::Recurrent,
                            );
                            let ssm_in = b.state_input(
                                &format!("l{li}.ssm"),
                                TensorType::f32(vec![
                                    1,
                                    mcfg.mamba_num_heads,
                                    mcfg.ssm_state,
                                    mcfg.mamba_head_dim,
                                ]),
                                StateRole::Recurrent,
                            );
                            inp.insert(
                                wz.id,
                                HostTensor::f32(vec![hidden, inner], w.m_wz[&li].clone()),
                            );
                            inp.insert(
                                wxbc.id,
                                HostTensor::f32(vec![hidden, conv_c], w.m_wxbc[&li].clone()),
                            );
                            inp.insert(
                                wdt.id,
                                HostTensor::f32(
                                    vec![hidden, mcfg.mamba_num_heads],
                                    w.m_wdt[&li].clone(),
                                ),
                            );
                            inp.insert(
                                dtb.id,
                                HostTensor::f32(
                                    vec![mcfg.mamba_num_heads],
                                    w.m_dtbias[&li].clone(),
                                ),
                            );
                            inp.insert(
                                wconv.id,
                                HostTensor::f32(
                                    vec![mcfg.conv_kernel, conv_c],
                                    w.m_wconv[&li].clone(),
                                ),
                            );
                            inp.insert(
                                convb.id,
                                HostTensor::f32(vec![conv_c], w.m_convbias[&li].clone()),
                            );
                            inp.insert(
                                ap.id,
                                HostTensor::f32(
                                    vec![1, mcfg.mamba_num_heads, 1, 1],
                                    w.m_a[&li].clone(),
                                ),
                            );
                            inp.insert(
                                dp.id,
                                HostTensor::f32(
                                    vec![1, mcfg.mamba_num_heads, 1, 1],
                                    w.m_d[&li].clone(),
                                ),
                            );
                            inp.insert(
                                gn.id,
                                HostTensor::f32(vec![inner], w.m_gatenorm[&li].clone()),
                            );
                            inp.insert(
                                wout.id,
                                HostTensor::f32(vec![inner, hidden], w.m_wout[&li].clone()),
                            );
                            inp.insert(conv_in.id, mamba_conv[&li].clone());
                            inp.insert(ssm_in.id, mamba_ssm[&li].clone());

                            let (out, conv_out, ssm_out) = nemotron_h_mamba_layer(
                                &b, cur, &mcfg, norm_w, wz, wxbc, wdt, dtb, wconv, convb, ap, dp,
                                gn, wout, conv_in, ssm_in, eps,
                            );
                            cur = out;
                            state_pairs.push((conv_in, conv_out));
                            state_pairs.push((ssm_in, ssm_out));
                        }
                        NemotronHLayerKind::Attention => {
                            let qd = acfg.num_heads * acfg.head_dim;
                            let kvd = acfg.num_kv_heads * acfg.head_dim;
                            let wq =
                                b.constant(&format!("l{li}.wq"), TensorType::f32(vec![hidden, qd]));
                            let wk = b
                                .constant(&format!("l{li}.wk"), TensorType::f32(vec![hidden, kvd]));
                            let wv = b
                                .constant(&format!("l{li}.wv"), TensorType::f32(vec![hidden, kvd]));
                            let wo =
                                b.constant(&format!("l{li}.wo"), TensorType::f32(vec![qd, hidden]));
                            let kc_in = b.state_input(
                                &format!("l{li}.kc"),
                                TensorType::f32(vec![1, acfg.num_kv_heads, cap, acfg.head_dim]),
                                StateRole::Recurrent,
                            );
                            let vc_in = b.state_input(
                                &format!("l{li}.vc"),
                                TensorType::f32(vec![1, acfg.num_kv_heads, cap, acfg.head_dim]),
                                StateRole::Recurrent,
                            );
                            inp.insert(
                                wq.id,
                                HostTensor::f32(vec![hidden, qd], w.a_wq[&li].clone()),
                            );
                            inp.insert(
                                wk.id,
                                HostTensor::f32(vec![hidden, kvd], w.a_wk[&li].clone()),
                            );
                            inp.insert(
                                wv.id,
                                HostTensor::f32(vec![hidden, kvd], w.a_wv[&li].clone()),
                            );
                            inp.insert(
                                wo.id,
                                HostTensor::f32(vec![qd, hidden], w.a_wo[&li].clone()),
                            );
                            inp.insert(kc_in.id, attn_k[&li].clone());
                            inp.insert(vc_in.id, attn_v[&li].clone());

                            let (out, kc_out, vc_out) = nemotron_h_attention_layer(
                                &b, cur, &acfg, norm_w, wq, wk, wv, wo, kc_in, vc_in, pos, eps,
                            );
                            cur = out;
                            state_pairs.push((kc_in, kc_out));
                            state_pairs.push((vc_in, vc_out));
                        }
                        NemotronHLayerKind::Mlp => {
                            let wup = b.constant(
                                &format!("l{li}.wup"),
                                TensorType::f32(vec![hidden, mlp_inter]),
                            );
                            let wdown = b.constant(
                                &format!("l{li}.wdown"),
                                TensorType::f32(vec![mlp_inter, hidden]),
                            );
                            inp.insert(
                                wup.id,
                                HostTensor::f32(vec![hidden, mlp_inter], w.f_wup[&li].clone()),
                            );
                            inp.insert(
                                wdown.id,
                                HostTensor::f32(vec![mlp_inter, hidden], w.f_wdown[&li].clone()),
                            );
                            cur = nemotron_h_mlp_layer(&b, cur, norm_w, wup, wdown, eps);
                        }
                    }
                }

                let g = b.finish_with_state(cur, &state_pairs);
                let inp: HashMap<poot_graph_ir::ValueId, Value> =
                    inp.into_iter().map(|(k, v)| (k, v.into())).collect();
                let step = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                    .expect("nemotron_h decode step eval");
                let out_t = step.output.into_host().expect("dense output");
                let new_states: Vec<HostTensor> = step
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("dense state"))
                    .collect();
                decode_got.push(out_t.as_f32().unwrap().to_vec());

                let mut si = 0usize;
                for (li, kind) in pattern.iter().enumerate() {
                    match kind {
                        NemotronHLayerKind::Mamba => {
                            mamba_conv.insert(li, new_states[si].clone());
                            mamba_ssm.insert(li, new_states[si + 1].clone());
                            si += 2;
                        }
                        NemotronHLayerKind::Attention => {
                            attn_k.insert(li, new_states[si].clone());
                            attn_v.insert(li, new_states[si + 1].clone());
                            si += 2;
                        }
                        NemotronHLayerKind::Mlp => {}
                    }
                }
            }

            // --- (2) one-shot traced prefill graph over all `cap` positions. ---
            let b = Builder::new();
            let x0 = b.constant("x", TensorType::f32(vec![1, cap, hidden]));
            let mut inp: HashMap<poot_graph_ir::ValueId, HostTensor> = HashMap::new();
            let mut xflat = Vec::with_capacity(cap * hidden);
            for xt in &xs {
                xflat.extend_from_slice(xt);
            }
            inp.insert(x0.id, HostTensor::f32(vec![1, cap, hidden], xflat));

            let mut mamba_tril = vec![0.0f32; mcfg.mamba_num_heads * cap * cap];
            for h in 0..mcfg.mamba_num_heads {
                for t in 0..cap {
                    for j in 0..=t {
                        mamba_tril[(h * cap + t) * cap + j] = 1.0;
                    }
                }
            }
            let tril_c = b.constant(
                "mtril",
                TensorType::f32(vec![1, mcfg.mamba_num_heads, cap, cap]),
            );
            inp.insert(
                tril_c.id,
                HostTensor::f32(vec![1, mcfg.mamba_num_heads, cap, cap], mamba_tril),
            );

            let mut cmask = vec![0.0f32; cap * cap];
            for t in 0..cap {
                for j in 0..cap {
                    if j > t {
                        cmask[t * cap + j] = -1.0e9;
                    }
                }
            }
            let cmask_c = b.constant("cmask", TensorType::f32(vec![1, 1, cap, cap]));
            inp.insert(cmask_c.id, HostTensor::f32(vec![1, 1, cap, cap], cmask));

            let mut weights: Vec<NemotronHLayerWeights> = Vec::new();
            for (li, kind) in pattern.iter().enumerate() {
                let norm_w = b.constant(&format!("l{li}.norm"), TensorType::f32(vec![hidden]));
                inp.insert(norm_w.id, HostTensor::f32(vec![hidden], w.norm[li].clone()));
                match kind {
                    NemotronHLayerKind::Mamba => {
                        let wz =
                            b.constant(&format!("l{li}.wz"), TensorType::f32(vec![hidden, inner]));
                        let wxbc = b.constant(
                            &format!("l{li}.wxbc"),
                            TensorType::f32(vec![hidden, conv_c]),
                        );
                        let wdt = b.constant(
                            &format!("l{li}.wdt"),
                            TensorType::f32(vec![hidden, mcfg.mamba_num_heads]),
                        );
                        let dtb = b.constant(
                            &format!("l{li}.dtb"),
                            TensorType::f32(vec![mcfg.mamba_num_heads]),
                        );
                        let wconv = b.constant(
                            &format!("l{li}.wconv"),
                            TensorType::f32(vec![mcfg.conv_kernel, conv_c]),
                        );
                        let convb =
                            b.constant(&format!("l{li}.convb"), TensorType::f32(vec![conv_c]));
                        let ap = b.constant(
                            &format!("l{li}.a"),
                            TensorType::f32(vec![1, mcfg.mamba_num_heads, 1, 1]),
                        );
                        let dp = b.constant(
                            &format!("l{li}.d"),
                            TensorType::f32(vec![1, mcfg.mamba_num_heads, 1, 1]),
                        );
                        let gn = b.constant(&format!("l{li}.gn"), TensorType::f32(vec![inner]));
                        let wout = b
                            .constant(&format!("l{li}.wout"), TensorType::f32(vec![inner, hidden]));
                        inp.insert(
                            wz.id,
                            HostTensor::f32(vec![hidden, inner], w.m_wz[&li].clone()),
                        );
                        inp.insert(
                            wxbc.id,
                            HostTensor::f32(vec![hidden, conv_c], w.m_wxbc[&li].clone()),
                        );
                        inp.insert(
                            wdt.id,
                            HostTensor::f32(
                                vec![hidden, mcfg.mamba_num_heads],
                                w.m_wdt[&li].clone(),
                            ),
                        );
                        inp.insert(
                            dtb.id,
                            HostTensor::f32(vec![mcfg.mamba_num_heads], w.m_dtbias[&li].clone()),
                        );
                        inp.insert(
                            wconv.id,
                            HostTensor::f32(vec![mcfg.conv_kernel, conv_c], w.m_wconv[&li].clone()),
                        );
                        inp.insert(
                            convb.id,
                            HostTensor::f32(vec![conv_c], w.m_convbias[&li].clone()),
                        );
                        inp.insert(
                            ap.id,
                            HostTensor::f32(
                                vec![1, mcfg.mamba_num_heads, 1, 1],
                                w.m_a[&li].clone(),
                            ),
                        );
                        inp.insert(
                            dp.id,
                            HostTensor::f32(
                                vec![1, mcfg.mamba_num_heads, 1, 1],
                                w.m_d[&li].clone(),
                            ),
                        );
                        inp.insert(
                            gn.id,
                            HostTensor::f32(vec![inner], w.m_gatenorm[&li].clone()),
                        );
                        inp.insert(
                            wout.id,
                            HostTensor::f32(vec![inner, hidden], w.m_wout[&li].clone()),
                        );
                        weights.push(NemotronHLayerWeights::Mamba {
                            norm_w,
                            w_z: wz,
                            w_xbc: wxbc,
                            w_dt: wdt,
                            dt_bias: dtb,
                            w_conv: wconv,
                            conv_bias: convb,
                            a_param: ap,
                            d_param: dp,
                            gate_norm_w: gn,
                            w_out: wout,
                        });
                    }
                    NemotronHLayerKind::Attention => {
                        let qd = acfg.num_heads * acfg.head_dim;
                        let kvd = acfg.num_kv_heads * acfg.head_dim;
                        let wq =
                            b.constant(&format!("l{li}.wq"), TensorType::f32(vec![hidden, qd]));
                        let wk =
                            b.constant(&format!("l{li}.wk"), TensorType::f32(vec![hidden, kvd]));
                        let wv =
                            b.constant(&format!("l{li}.wv"), TensorType::f32(vec![hidden, kvd]));
                        let wo =
                            b.constant(&format!("l{li}.wo"), TensorType::f32(vec![qd, hidden]));
                        inp.insert(
                            wq.id,
                            HostTensor::f32(vec![hidden, qd], w.a_wq[&li].clone()),
                        );
                        inp.insert(
                            wk.id,
                            HostTensor::f32(vec![hidden, kvd], w.a_wk[&li].clone()),
                        );
                        inp.insert(
                            wv.id,
                            HostTensor::f32(vec![hidden, kvd], w.a_wv[&li].clone()),
                        );
                        inp.insert(
                            wo.id,
                            HostTensor::f32(vec![qd, hidden], w.a_wo[&li].clone()),
                        );
                        weights.push(NemotronHLayerWeights::Attention {
                            norm_w,
                            w_q: wq,
                            w_k: wk,
                            w_v: wv,
                            w_o: wo,
                        });
                    }
                    NemotronHLayerKind::Mlp => {
                        let wup = b.constant(
                            &format!("l{li}.wup"),
                            TensorType::f32(vec![hidden, mlp_inter]),
                        );
                        let wdown = b.constant(
                            &format!("l{li}.wdown"),
                            TensorType::f32(vec![mlp_inter, hidden]),
                        );
                        inp.insert(
                            wup.id,
                            HostTensor::f32(vec![hidden, mlp_inter], w.f_wup[&li].clone()),
                        );
                        inp.insert(
                            wdown.id,
                            HostTensor::f32(vec![mlp_inter, hidden], w.f_wdown[&li].clone()),
                        );
                        weights.push(NemotronHLayerWeights::Mlp {
                            norm_w,
                            w_up: wup,
                            w_down: wdown,
                        });
                    }
                }
            }

            let (out, states) = trace_nemotron_h_hybrid_stack_prefill(
                &b, &pattern, &weights, x0, &mcfg, &acfg, tril_c, cmask_c, eps,
            );
            let g_out = b.finish(out);
            let inp: HashMap<poot_graph_ir::ValueId, Value> =
                inp.into_iter().map(|(k, v)| (k, v.into())).collect();
            let prefill_out = eval(&g_out, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .expect("nemotron_h prefill eval")
                .output
                .into_host()
                .expect("dense output");

            // prefill_out is [1,cap,hidden] row-major; decode_got[t] is one [hidden] row per step.
            for (t, want_row) in decode_got.iter().enumerate() {
                let row = &prefill_out.as_f32().unwrap()[t * hidden..(t + 1) * hidden];
                for (i, (a, b)) in row.iter().zip(want_row.iter()).enumerate() {
                    let diff = (a - b).abs();
                    let tol = 1e-4 * b.abs().max(1.0);
                    assert!(
                        diff <= tol,
                        "step {t} element {i}: prefill {a} decode {b} (n_groups={n_groups})"
                    );
                }
            }

            // Final per-layer state: clone the finished graph per state value (only `.output` differs) and
            // eval each against the same inputs.
            for (li, (kind, state)) in pattern.iter().zip(states.iter()).enumerate() {
                match (kind, state) {
                    (NemotronHLayerKind::Mamba, NemotronHLayerState::Mamba { conv_cache, h }) => {
                        let mut gc = g_out.clone();
                        gc.output = conv_cache.id;
                        let got_conv = eval(&gc, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                            .expect("prefill conv_cache eval")
                            .output
                            .into_host()
                            .expect("dense output");
                        let want_conv = &mamba_conv[&li];
                        for (i, (a, w)) in got_conv
                            .as_f32()
                            .unwrap()
                            .iter()
                            .zip(want_conv.as_f32().unwrap().iter())
                            .enumerate()
                        {
                            let diff = (a - w).abs();
                            assert!(
                                diff <= 1e-4 * w.abs().max(1.0),
                                "layer {li} conv_cache element {i}: prefill {a} decode {w}"
                            );
                        }
                        let mut gh = g_out.clone();
                        gh.output = h.id;
                        let got_h = eval(&gh, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                            .expect("prefill h_out eval")
                            .output
                            .into_host()
                            .expect("dense output");
                        let want_h = &mamba_ssm[&li];
                        for (i, (a, w)) in got_h
                            .as_f32()
                            .unwrap()
                            .iter()
                            .zip(want_h.as_f32().unwrap().iter())
                            .enumerate()
                        {
                            let diff = (a - w).abs();
                            assert!(
                                diff <= 1e-4 * w.abs().max(1.0),
                                "layer {li} ssm state element {i}: prefill {a} decode {w}"
                            );
                        }
                    }
                    (NemotronHLayerKind::Attention, NemotronHLayerState::Attention { k, v }) => {
                        let mut gk = g_out.clone();
                        gk.output = k.id;
                        let got_k = eval(&gk, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                            .expect("prefill k eval")
                            .output
                            .into_host()
                            .expect("dense output");
                        let want_k = &attn_k[&li];
                        for (i, (a, w)) in got_k
                            .as_f32()
                            .unwrap()
                            .iter()
                            .zip(want_k.as_f32().unwrap().iter())
                            .enumerate()
                        {
                            let diff = (a - w).abs();
                            assert!(
                                diff <= 1e-4 * w.abs().max(1.0),
                                "layer {li} k element {i}: prefill {a} decode {w}"
                            );
                        }
                        let mut gv = g_out.clone();
                        gv.output = v.id;
                        let got_v = eval(&gv, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                            .expect("prefill v eval")
                            .output
                            .into_host()
                            .expect("dense output");
                        let want_v = &attn_v[&li];
                        for (i, (a, w)) in got_v
                            .as_f32()
                            .unwrap()
                            .iter()
                            .zip(want_v.as_f32().unwrap().iter())
                            .enumerate()
                        {
                            let diff = (a - w).abs();
                            assert!(
                                diff <= 1e-4 * w.abs().max(1.0),
                                "layer {li} v element {i}: prefill {a} decode {w}"
                            );
                        }
                    }
                    (NemotronHLayerKind::Mlp, NemotronHLayerState::Mlp) => {}
                    _ => panic!("layer {li}: kind/state mismatch"),
                }
            }
        }

        #[test]
        fn nemotron_h_hybrid_stack_prefill_matches_decode_loop_grouped_bc() {
            // grouped B/C: n_groups=2 < mamba_num_heads=4.
            run_prefill_matches_decode_loop(2);
        }

        #[test]
        fn nemotron_h_hybrid_stack_prefill_matches_decode_loop_degenerate_groups() {
            // degenerate groups: n_groups == mamba_num_heads.
            run_prefill_matches_decode_loop(4);
        }

        /// Host-side decode-graph binder for [`trace_nemotron_h_decode`] (there is no `Runner` here, so
        /// `poot_llm::Runner::bind_decode` is reproduced locally): `Slot::Token`/`Slot::Pos`/`Slot::SeqLen`
        /// scalars, a `[cap]` additive causal mask row (0 for `t <= pos`, large-negative otherwise), named
        /// consts from `weights`, and state from `caches` in `g.state` order.
        fn bind_nemotron_h_decode_step(
            g: &Graph,
            weights: &HashMap<String, HostTensor>,
            token: u32,
            pos: usize,
            cap: usize,
            caches: &[HostTensor],
        ) -> HashMap<poot_graph_ir::ValueId, Value> {
            use poot_graph_ir::graph::Storage;
            let mut inputs = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                let t = match meta.storage {
                    Storage::Slot(Slot::Token) => HostTensor::i32(vec![], vec![token as i32]),
                    Storage::Slot(Slot::Pos) => HostTensor::i32(vec![], vec![pos as i32]),
                    Storage::Slot(Slot::SeqLen) => HostTensor::i32(vec![], vec![(pos + 1) as i32]),
                    Storage::Slot(Slot::Mask) => {
                        let mut row = vec![0.0f32; cap];
                        for (t, slot) in row.iter_mut().enumerate() {
                            if t > pos {
                                *slot = -1.0e9;
                            }
                        }
                        HostTensor::f32(vec![cap], row)
                    }
                    Storage::Slot(other) => {
                        panic!("unexpected slot {other:?} in nemotron_h decode graph")
                    }
                    Storage::State => continue,
                    Storage::Const => {
                        let name = meta.name.as_deref().expect("const without a name");
                        weights
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| panic!("no weight bound for {name}"))
                    }
                    Storage::Computed(computed) => {
                        HostTensor::f32(computed.shape(), computed.values_f32())
                    }
                    Storage::Device => panic!("device value in input set"),
                };
                inputs.insert(id, t.into());
            }
            if !caches.is_empty() {
                for (ci, &(si, _)) in g.state.iter().enumerate() {
                    inputs.insert(si, caches[ci].clone().into());
                }
            }
            inputs
        }

        /// Build the named-weight map both [`trace_nemotron_h_decode`] and [`trace_nemotron_h_prefill`]
        /// expect (they use the same constant names), for the toy stack's [`gen_weights`] output plus a
        /// synthetic embedding table / final norm / lm_head.
        #[allow(clippy::too_many_arguments)]
        fn toy_top_level_weights(
            pattern: &[NemotronHLayerKind],
            mcfg: &NemotronHMambaConfig,
            acfg: &NemotronHAttnConfig,
            hidden: usize,
            mlp_inter: usize,
            vocab: usize,
            w: &ToyWeights,
            embed: &[f32],
            norm_f: &[f32],
            lm_head: &[f32],
        ) -> HashMap<String, HostTensor> {
            let inner = mcfg.inner();
            let conv_c = mcfg.conv_channels();
            let mut weights: HashMap<String, HostTensor> = HashMap::new();
            weights.insert(
                "backbone.embeddings.weight".to_string(),
                HostTensor::f32(vec![vocab, hidden], embed.to_vec()),
            );
            weights.insert(
                "backbone.norm_f.weight".to_string(),
                HostTensor::f32(vec![hidden], norm_f.to_vec()),
            );
            weights.insert(
                "lm_head.weight".to_string(),
                HostTensor::f32(vec![hidden, vocab], lm_head.to_vec()),
            );
            for (li, kind) in pattern.iter().enumerate() {
                weights.insert(
                    format!("layers.{li}.norm.weight"),
                    HostTensor::f32(vec![hidden], w.norm[li].clone()),
                );
                match kind {
                    NemotronHLayerKind::Mamba => {
                        weights.insert(
                            format!("layers.{li}.mixer.z.weight"),
                            HostTensor::f32(vec![hidden, inner], w.m_wz[&li].clone()),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.xbc.weight"),
                            HostTensor::f32(vec![hidden, conv_c], w.m_wxbc[&li].clone()),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.dt.weight"),
                            HostTensor::f32(
                                vec![hidden, mcfg.mamba_num_heads],
                                w.m_wdt[&li].clone(),
                            ),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.dt_bias"),
                            HostTensor::f32(vec![mcfg.mamba_num_heads], w.m_dtbias[&li].clone()),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.conv1d.weight"),
                            HostTensor::f32(vec![mcfg.conv_kernel, conv_c], w.m_wconv[&li].clone()),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.conv1d.bias"),
                            HostTensor::f32(vec![conv_c], w.m_convbias[&li].clone()),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.a"),
                            HostTensor::f32(
                                vec![1, mcfg.mamba_num_heads, 1, 1],
                                w.m_a[&li].clone(),
                            ),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.d"),
                            HostTensor::f32(
                                vec![1, mcfg.mamba_num_heads, 1, 1],
                                w.m_d[&li].clone(),
                            ),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.gate_norm.weight"),
                            HostTensor::f32(vec![inner], w.m_gatenorm[&li].clone()),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.out_proj.weight"),
                            HostTensor::f32(vec![inner, hidden], w.m_wout[&li].clone()),
                        );
                    }
                    NemotronHLayerKind::Attention => {
                        let qd = acfg.num_heads * acfg.head_dim;
                        let kvd = acfg.num_kv_heads * acfg.head_dim;
                        weights.insert(
                            format!("layers.{li}.mixer.q_proj.weight"),
                            HostTensor::f32(vec![hidden, qd], w.a_wq[&li].clone()),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.k_proj.weight"),
                            HostTensor::f32(vec![hidden, kvd], w.a_wk[&li].clone()),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.v_proj.weight"),
                            HostTensor::f32(vec![hidden, kvd], w.a_wv[&li].clone()),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.o_proj.weight"),
                            HostTensor::f32(vec![qd, hidden], w.a_wo[&li].clone()),
                        );
                    }
                    NemotronHLayerKind::Mlp => {
                        weights.insert(
                            format!("layers.{li}.mixer.up_proj.weight"),
                            HostTensor::f32(vec![hidden, mlp_inter], w.f_wup[&li].clone()),
                        );
                        weights.insert(
                            format!("layers.{li}.mixer.down_proj.weight"),
                            HostTensor::f32(vec![mlp_inter, hidden], w.f_wdown[&li].clone()),
                        );
                    }
                }
            }
            weights
        }

        /// Independent-reference test for [`trace_nemotron_h_decode`]: a whole-model `cap`-step decode
        /// trajectory with real-magnitude synthetic weights, compared per position against a hand-rolled
        /// reference that reuses [`run_reference`] plus a hand-written embedding gather and final
        /// rmsnorm/lm_head. Tolerance `1e-4` absolute-relative.
        #[test]
        fn trace_nemotron_h_decode_matches_independent_reference() {
            let hidden = 6usize;
            let vocab = 12usize;
            let mcfg = NemotronHMambaConfig {
                hidden,
                mamba_num_heads: 4,
                mamba_head_dim: 2,
                n_groups: 2,
                ssm_state: 3,
                conv_kernel: 3,
            };
            let acfg = NemotronHAttnConfig {
                hidden,
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 2,
            };
            let mlp_inter = 10usize;
            let eps = 1e-6f32;
            let cap = 5usize;
            let pattern = parse_hybrid_pattern(TOY_PATTERN);
            let w = gen_weights(&pattern, &mcfg, &acfg, hidden, mlp_inter);

            let tokens: Vec<u32> = (0..cap).map(|t| ((t * 3 + 1) % vocab) as u32).collect();
            let embed = fill(vocab * hidden, seed_of("embed"));
            let xs: Vec<Vec<f32>> = tokens
                .iter()
                .map(|&t| embed[t as usize * hidden..(t as usize + 1) * hidden].to_vec())
                .collect();

            let hidden_out = run_reference(&pattern, &mcfg, &acfg, hidden, mlp_inter, eps, &w, &xs);
            let norm_f = fill(hidden, seed_of("norm_f"));
            let lm_head = fill(hidden * vocab, seed_of("lm_head"));
            let want_logits: Vec<Vec<f32>> = hidden_out
                .iter()
                .map(|hstate| {
                    let ms: f32 = hstate.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
                    let denom = (ms + eps).sqrt();
                    let xn: Vec<f32> = hstate
                        .iter()
                        .zip(&norm_f)
                        .map(|(v, g)| (v / denom) * g)
                        .collect();
                    mm(&xn, &lm_head, hidden, vocab)
                })
                .collect();

            let weights = toy_top_level_weights(
                &pattern, &mcfg, &acfg, hidden, mlp_inter, vocab, &w, &embed, &norm_f, &lm_head,
            );
            let cfg = NemotronHConfig {
                vocab_size: vocab,
                hidden,
                mlp_inter,
                eps,
                pattern: pattern.clone(),
                mamba: mcfg,
                attn: acfg,
            };
            let g = trace_nemotron_h_decode(&cfg, cap); // built ONCE, replayed every step
            let mut caches: Vec<HostTensor> = g
                .state
                .iter()
                .map(|&(si, _)| HostTensor::zeros(g.aval(si).shape.clone()))
                .collect();
            for (pos, &tok) in tokens.iter().enumerate() {
                let inputs = bind_nemotron_h_decode_step(&g, &weights, tok, pos, cap, &caches);
                let step = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                    .expect("nemotron_h decode step eval");
                let logits = step.output.into_host().expect("dense output");
                caches = step
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("dense state"))
                    .collect();
                for (i, (a, want)) in logits
                    .as_f32()
                    .unwrap()
                    .iter()
                    .zip(&want_logits[pos])
                    .enumerate()
                {
                    let diff = (a - want).abs();
                    let tol = 1e-4 * want.abs().max(1.0);
                    assert!(diff <= tol, "pos {pos} logit {i}: got {a} want {want}");
                }
            }
        }

        /// Decode-loop-matches-prefill for [`trace_nemotron_h_decode`] (as `qwen38_decode_loop_matches_prefill`):
        /// the same weights bound into both top-level tracers, `trace_nemotron_h_prefill`'s one-shot `cap`-token
        /// call vs `cap` sequential decode steps from an empty cache, compared at the final position. Stronger
        /// than the independent-reference test: it shows the fixed-cap `Slot::Pos`/`Slot::Mask` attention
        /// ([`nemotron_h_attention_layer_decode_masked`]) reduces to the verified prefill path.
        #[test]
        fn trace_nemotron_h_decode_loop_matches_prefill() {
            let hidden = 6usize;
            let vocab = 12usize;
            let mcfg = NemotronHMambaConfig {
                hidden,
                mamba_num_heads: 4,
                mamba_head_dim: 2,
                n_groups: 2,
                ssm_state: 3,
                conv_kernel: 3,
            };
            let acfg = NemotronHAttnConfig {
                hidden,
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 2,
            };
            let mlp_inter = 10usize;
            let eps = 1e-6f32;
            let cap = 5usize;
            let pattern = parse_hybrid_pattern(TOY_PATTERN);
            let w = gen_weights(&pattern, &mcfg, &acfg, hidden, mlp_inter);
            let tokens: Vec<u32> = (0..cap).map(|t| ((t * 5 + 2) % vocab) as u32).collect();
            let embed = fill(vocab * hidden, seed_of("embed2"));
            let norm_f = fill(hidden, seed_of("norm_f2"));
            let lm_head = fill(hidden * vocab, seed_of("lm_head2"));

            let weights = toy_top_level_weights(
                &pattern, &mcfg, &acfg, hidden, mlp_inter, vocab, &w, &embed, &norm_f, &lm_head,
            );
            let cfg = NemotronHConfig {
                vocab_size: vocab,
                hidden,
                mlp_inter,
                eps,
                pattern: pattern.clone(),
                mamba: mcfg,
                attn: acfg,
            };

            // --- (1) one-shot prefill over all `cap` tokens. ---
            let prefill_weights = weights.clone();
            let mut cmask = vec![0.0f32; cap * cap];
            for t in 0..cap {
                for j in 0..cap {
                    if j > t {
                        cmask[t * cap + j] = -1.0e9;
                    }
                }
            }

            let gp = trace_nemotron_h_prefill(&cfg, cap);
            let mut pin: HashMap<poot_graph_ir::ValueId, Value> = HashMap::new();
            for &id in &gp.inputs {
                let meta = gp.meta(id);
                match meta.storage {
                    poot_graph_ir::graph::Storage::Slot(Slot::Token) => {
                        pin.insert(
                            id,
                            HostTensor::i32(vec![cap], tokens.iter().map(|&t| t as i32).collect())
                                .into(),
                        );
                    }
                    poot_graph_ir::graph::Storage::Slot(Slot::Mask) => {
                        let name = meta.name.as_deref().expect("mask slot without a name");
                        assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                        pin.insert(
                            id,
                            HostTensor::f32(meta.aval.shape.clone(), cmask.clone()).into(),
                        );
                    }
                    poot_graph_ir::graph::Storage::Const => {
                        let name = meta.name.as_deref().expect("const without a name");
                        let t = prefill_weights
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| panic!("no weight for {name}"));
                        pin.insert(id, t.into());
                    }
                    other => panic!("unexpected storage {other:?} in nemotron_h prefill graph"),
                }
            }
            let prefill_logits = eval(&gp, &pin, EvalOptions::new(EvalBudget::UNBOUNDED))
                .expect("nemotron_h prefill eval")
                .output
                .into_host()
                .expect("dense output");

            // --- (2) `cap` sequential decode steps from an empty cache. ---
            let gd = trace_nemotron_h_decode(&cfg, cap); // built ONCE
            let mut caches: Vec<HostTensor> = gd
                .state
                .iter()
                .map(|&(si, _)| HostTensor::zeros(gd.aval(si).shape.clone()))
                .collect();
            let mut decode_logits = None;
            for (pos, &tok) in tokens.iter().enumerate() {
                let inputs = bind_nemotron_h_decode_step(&gd, &weights, tok, pos, cap, &caches);
                let step = eval(&gd, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                    .expect("nemotron_h decode step eval");
                let logits = step.output.into_host().expect("dense output");
                caches = step
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("dense state"))
                    .collect();
                decode_logits = Some(logits);
            }
            let decode_logits = decode_logits.expect("at least one decode step ran");

            for (i, (a, want)) in decode_logits
                .as_f32()
                .unwrap()
                .iter()
                .zip(prefill_logits.as_f32().unwrap().iter())
                .enumerate()
            {
                let diff = (a - want).abs();
                let tol = 1e-4 * want.abs().max(1.0);
                assert!(diff <= tol, "logit {i}: decode {a} prefill {want}");
            }
        }
    }
}
