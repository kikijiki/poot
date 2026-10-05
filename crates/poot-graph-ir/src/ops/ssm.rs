use super::*;

/// Decode step of linear attention (card 038: Qwen3.5 / MiniMax-style recurrence). Instead of a
/// growing KV cache + softmax, each head carries a fixed-size recurrent state matrix
/// `S [1, Hq, D_k, D_v]` (the running sum of outer products), so decode is O(1) in sequence length:
///
/// ```text
/// S_t = decay * S_{t-1} + k_t^T v_t        (outer product update, [D_k, D_v])
/// o_t = q_t @ S_t                          ([1, D_k] @ [D_k, D_v] -> [1, D_v])
/// ```
///
/// `q[1,Hq,1,D_k]`, `k`/`v[1,Hkv,1,D_k|D_v]` (GQA via `repeat_kv`), `s_in[1,Hq,D_k,D_v]`. Returns
/// `(o[1,Hq,1,D_v], s_out[1,Hq,D_k,D_v])`; the caller wires `s_out` back as the carried state
/// (`finish_with_state`), like the KV cache. `decay` is a scalar gate; per-head / data-dependent
/// gating is [`linear_attention_decode_gated`]. Pure primitive composition, shared by
/// tracer and eval.
pub fn linear_attention_decode(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    decay: f32,
    s_in: Traced,
) -> (Traced, Traced) {
    let k = repeat_kv(b, k, n_rep); // [1,Hq,1,D_k]
    let v = repeat_kv(b, v, n_rep); // [1,Hq,1,D_v]
    let kt = b.transpose(k, vec![0, 1, 3, 2]); // [1,Hq,D_k,1]
    let kv = b.matmul(kt, v); // outer product [1,Hq,D_k,D_v]
    let s_decayed = b.binary_scalar(BinOp::Mul, s_in, Scalar::F32(decay));
    let s_out = b.binary(BinOp::Add, s_decayed, kv); // S_t = decay*S + k^T v
    let o = b.matmul(q, s_out); // q @ S_t -> [1,Hq,1,D_v]
    (o, s_out)
}

/// Gated linear-attention decode (card 038): [`linear_attention_decode`] with a data-dependent
/// per-key-channel decay instead of a scalar (the GLA / gated-delta-net form of hybrid models); the
/// caller computes the gate `g_t` from the token. `gate[1,Hq,D_k,1]` decays each row `a` of the state
/// by `g_t[a]` (broadcast over `D_v`):
///
/// ```text
/// S_t = g_t * S_{t-1} + k_t^T v_t          (g_t broadcasts [D_k,1] over [D_k,D_v])
/// o_t = q_t @ S_t
/// ```
///
/// Same shapes and return contract as [`linear_attention_decode`]; `decay: f32` becomes a `gate`
/// tensor multiplied with broadcast. Pure primitive composition.
pub fn linear_attention_decode_gated(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    gate: Traced,
    s_in: Traced,
) -> (Traced, Traced) {
    let k = repeat_kv(b, k, n_rep); // [1,Hq,1,D_k]
    let v = repeat_kv(b, v, n_rep); // [1,Hq,1,D_v]
    let kt = b.transpose(k, vec![0, 1, 3, 2]); // [1,Hq,D_k,1]
    let kv = b.matmul(kt, v); // outer product [1,Hq,D_k,D_v]
    let s_gated = b.binary(BinOp::Mul, s_in, gate); // gate [1,Hq,D_k,1] broadcasts over D_v
    let s_out = b.binary(BinOp::Add, s_gated, kv); // S_t = g_t*S + k^T v
    let o = b.matmul(q, s_out); // q @ S_t -> [1,Hq,1,D_v]
    (o, s_out)
}

/// Mamba2 / SSD decode step (card 038): a selective SSM step is a gated linear-attention step, so
/// this is a thin discretization wrapper over [`linear_attention_decode_gated`]. Mamba2's per-head
/// recurrence `h_t = Abar_t * h_{t-1} + Bbar_t^T x_t`, `y_t = C_t @ h_t + D * x_t` with
/// `Abar = exp(Delta * A)`, `Bbar = Delta * B` maps onto gated linear attention with query `C`, key
/// `Bbar`, value `x`, and per-head scalar gate `Abar` (broadcast over the state):
///
/// ```text
/// Abar = exp(Delta (.) A)        // [1,Hq,1,1] per-head gate
/// Bbar = Delta (.) B             // [1,Hq,1,N]
/// (o, h_out) = gated_linear_attention(q=C, k=Bbar, v=x, gate=Abar, h_in)
/// y = o + D (.) x                // residual skip
/// ```
///
/// `x[1,Hq,1,P]` (input / value, P = head dim), `b_in`/`c_in[1,Hq,1,N]` (input-dependent SSM B/C, N =
/// state dim), `delta[1,Hq,1,1]` (input-dependent timestep), `a_param`/`d_skip[1,Hq,1,1]` (learned
/// per-head A / D), `h_in[1,Hq,N,P]` (carried SSM state). Returns `(y[1,Hq,1,P], h_out[1,Hq,N,P])`.
/// Pure primitive composition; the SSM block for nemotron_h / Mamba2 hybrid layers.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssd_decode(
    b: &Builder,
    x: Traced,
    b_in: Traced,
    c_in: Traced,
    delta: Traced,
    a_param: Traced,
    d_skip: Traced,
    h_in: Traced,
) -> (Traced, Traced) {
    let a_bar = b.unary(UnOp::Exp, b.binary(BinOp::Mul, delta, a_param)); // exp(Delta * A) [1,Hq,1,1]
    let b_bar = b.binary(BinOp::Mul, delta, b_in); // Delta * B [1,Hq,1,N]
    let (o, h_out) = linear_attention_decode_gated(b, c_in, b_bar, x, 1, a_bar, h_in);
    let skip = b.binary(BinOp::Mul, d_skip, x); // D * x [1,Hq,1,P]
    let y = b.binary(BinOp::Add, o, skip);
    (y, h_out)
}

/// Mamba2 / SSD prefill (nemotron_h spec 279): the whole-sequence twin of [`mamba2_ssd_decode`], a
/// thin wrapper over [`decay_mask_from_gates`] + [`linear_attention_prefill`]. The plain
/// (non-per-channel) prefill primitives suffice because Mamba2's gate `Abar_t = exp(Delta_t * A)` is
/// a per-head scalar (one per head per position), unlike Qwen3-Next's Gated-DeltaNet, whose gate is
/// per state cell and needs the chunked UT-transform (`gdn_prefill_chunked`).
///
/// Assumes a fresh start (`h_in = 0`), like every other prefill entry point here
/// ([`causal_conv1d_prefill`]'s zero left-pad, `qwen3next_gdn_prefill_block`, [`attention_prefill`]).
///
/// ```text
/// Abar_t = exp(Delta_t (.) A)             // [1,Hq,L,1] per-head, per-position scalar gate
/// Bbar_t = Delta_t (.) B_t                // [1,Hq,L,N]
/// mask   = decay_mask_from_gates(Abar, tril)      // [1,Hq,L,L], mask[t,j] = prod_{j<k<=t} Abar_k
/// o      = linear_attention_prefill(q=C, k=Bbar, v=x, mask)   // [1,Hq,L,P]
/// y      = o + D (.) x                    // residual skip, same as the decode form
/// ```
///
/// The final state `h_out` (to seed a [`mamba2_ssd_decode`] continuation) needs no sequential scan:
/// unrolling from `h_{-1}=0` gives `h_{L-1} = sum_j (prod_{k=j+1}^{L-1} Abar_k) Bbar_j^T x_j`, and the
/// weight `prod_{j<k<=L-1} Abar_k` is exactly the mask's last row (`t = L-1`). So `h_out` is one more
/// weighted-outer-product matmul reusing that row.
///
/// `x[1,Hq,L,P]` (input / value), `b_in`/`c_in[1,Hq,L,N]` (input-dependent SSM B/C, already broadcast
/// to `Hq` heads by the caller via [`repeat_kv`], as in [`mamba2_ssd_decode`]), `delta`
/// (`[1,Hq,L,1]`), `a_param`/`d_skip[1,Hq,1,1]` (learned per-head A / D), `tril[1,Hq,L,L]`
/// (lower-triangular ones including the diagonal, per-head-broadcast, as [`decay_mask_from_gates`]
/// expects). Returns `(y[1,Hq,L,P], h_out[1,Hq,N,P])`.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssd_prefill(
    b: &Builder,
    x: Traced,
    b_in: Traced,
    c_in: Traced,
    delta: Traced,
    a_param: Traced,
    d_skip: Traced,
    tril: Traced,
) -> (Traced, Traced) {
    let shape = b.aval(x).shape;
    let (hq, l) = (shape[1], shape[2]);

    let a_bar = b.unary(UnOp::Exp, b.binary(BinOp::Mul, delta, a_param)); // [1,Hq,L,1]
    let b_bar = b.binary(BinOp::Mul, delta, b_in); // Delta * B [1,Hq,L,N]

    let gates = b.reshape(a_bar, vec![1, hq, 1, l]); // [1,Hq,L,1] -> [1,Hq,1,L], same L-major memory order
    let mask = decay_mask_from_gates(b, gates, tril, hq, l); // [1,Hq,L,L]

    let y_lin = linear_attention_prefill(b, c_in, b_bar, x, 1, mask); // [1,Hq,L,P]
    let skip = b.binary(BinOp::Mul, d_skip, x); // D * x [1,Hq,L,P], d_skip broadcasts over L
    let y = b.binary(BinOp::Add, y_lin, skip);

    // h_out = sum_j weight_j * Bbar_j^T x_j, weight_j = mask's own last row (t=L-1): prod_{j<k<=L-1} Abar_k.
    let weight_row = b.slice(mask, 2, l - 1, l); // [1,Hq,1,L]
    let weight_col = b.reshape(weight_row, vec![1, hq, l, 1]); // same L-major order, now column-shaped
    let weighted_b = b.binary(BinOp::Mul, b_bar, weight_col); // [1,Hq,L,N], broadcast over N
    let weighted_b_t = b.transpose(weighted_b, vec![0, 1, 3, 2]); // [1,Hq,N,L]
    let h_out = b.matmul(weighted_b_t, x); // [1,Hq,N,L] @ [1,Hq,L,P] -> [1,Hq,N,P]

    (y, h_out)
}

/// Decode step of a depthwise causal conv1d (card 038: the short conv before Mamba's SSM). Per
/// channel, `out[c] = sum_k w[k][c] * window[k][c]`, where `window` is the last `K-1` inputs (the
/// carried conv cache) plus the current token; the cache shifts in the new token each step. `x[B,1,C]`,
/// `w[K,C]` (per-channel kernel, shared across the batch), `cache[B,K-1,C]` (zero-initialized).
/// Returns `(out[B,1,C], new_cache[B,K-1,C])`; the caller carries `new_cache` as State. `B >= 1` is
/// read off `x`; rows are independent (as in [`gated_delta_net_decode`]). Pure primitive composition
/// (concat, broadcast multiply, reduce, slice); Mamba's bias + SiLU are the caller's. `k` is the
/// kernel size.
pub fn causal_conv1d_decode(
    b: &Builder,
    x: Traced,
    w: Traced,
    cache: Traced,
    k: usize,
) -> (Traced, Traced) {
    let x_shape = b.aval(x).shape;
    let batch = x_shape[0];
    let c = *x_shape.last().expect("conv input is at least 1-D");
    let window = b.concat(1, &[cache, x]); // [B, K, C] = last K-1 inputs ++ current
    // sum the length-K window per channel. Reduce over the LAST axis (move K there): a non-last-axis reduce
    // is unreliable, so transpose [B,K,C] -> [B,C,K], multiply by the (transposed) kernel, reduce axis 2.
    let window_t = b.transpose(window, vec![0, 2, 1]); // [B, C, K]
    let wk = b.transpose(b.reshape(w, vec![1, k, c]), vec![0, 2, 1]); // [1, C, K], broadcasts over B
    let weighted = b.binary(BinOp::Mul, wk, window_t); // [B, C, K]
    let summed = b.reduce(RedOp::Sum, weighted, 2, true); // [B, C, 1]
    let out = b.reshape(summed, vec![batch, 1, c]); // [B, 1, C]
    let new_cache = b.slice(window, 1, 1, k); // drop the oldest -> [B,K-1,C]
    (out, new_cache)
}

/// Batched causal depthwise conv1d for prefill (card 153 / spec 139 B5): the same conv as
/// [`causal_conv1d_decode`], computed for all `L` positions at once (the GDN block's conv feeds Q/K/V,
/// so chunked GDN prefill needs it over the whole chunk). `x[1,L,C]`, `w[K,C]`. Returns `out[1,L,C]`.
///
/// `out[t,c] = sum_{j=0..K-1} w[j,c] * x[t-(K-1)+j,c]`, with `x[<0,c] = 0`, matching
/// [`causal_conv1d_decode`] per position (`causal_conv1d_prefill_matches_decode`). No bias, no
/// activation; the caller applies SiLU (e.g. `qwen3next_gdn_block`).
///
/// Left-pad `x` by `K-1` zeros along the sequence axis (concat of zeros; there is no `Pad`
/// primitive), then for each of the `K` taps (small static constant, unrolled at trace time) slice
/// the shifted window `x_pad[:, j:j+L, :]`, multiply by `w[j,:]`, and sum. The zero pad is `x`'s first
/// time-slice times 0, broadcast to `[1,K-1,C]`, rather than a new host constant.
pub fn causal_conv1d_prefill(b: &Builder, x: Traced, w: Traced, k: usize) -> Traced {
    causal_conv1d_prefill_dilated(b, x, w, k, 1)
}

/// Dilated generalization of [`causal_conv1d_prefill`] (spec 282's N-gram/PLE): the `K` taps are
/// `dilation` positions apart: `out[t,c] = sum_{j=0..K-1} w[j,c] * x[t - (K-1-j)*dilation, c]`, with
/// `x[<0,c] = 0`. `dilation == 1` is exactly [`causal_conv1d_prefill`] (which delegates here).
///
/// Needed by `Qwen4ExpTextPLELayer._short_conv` (`modeling_qwen4_exp.py`), a depthwise and dilated
/// `nn.Conv1d` (`ple_conv_kernel_size = 4`, `ngram_size = 3`, so the causal left pad is
/// `(K-1)*dilation = 9`). The pad width and tap stride are trace-time constants.
pub fn causal_conv1d_prefill_dilated(
    b: &Builder,
    x: Traced,
    w: Traced,
    k: usize,
    dilation: usize,
) -> Traced {
    assert!(dilation >= 1, "dilation must be >= 1");
    let shape = b.aval(x).shape;
    let (l, c) = (shape[1], shape[2]);
    let pad = (k - 1) * dilation;

    // zero-pad the left by (K-1)*dilation: x's own first time-slice, zeroed, broadcast to [1, pad, C].
    let x_pad = if pad == 0 {
        x
    } else {
        let first = b.slice(x, 1, 0, 1); // [1, 1, C]
        let zero_slice = b.binary_scalar(BinOp::Mul, first, Scalar::F32(0.0)); // [1, 1, C] zeros
        let zeros = b.broadcast(zero_slice, vec![1, pad, c]); // [1, pad, C]
        b.concat(1, &[zeros, x]) // [1, pad+L, C]
    };

    // K-tap shift-multiply-add (K static, unrolled at trace time - the sum in causal_conv1d_decode's math).
    let mut acc: Option<Traced> = None;
    for j in 0..k {
        let off = j * dilation;
        let window = b.slice(x_pad, 1, off, off + l); // [1, L, C] = x_pad[j*dilation ..][..L]
        let wj = b.reshape(b.slice(w, 0, j, j + 1), vec![1, 1, c]); // [1, 1, C] = w[j, :]
        let term = b.binary(BinOp::Mul, window, wj); // broadcast wj over L
        acc = Some(match acc {
            None => term,
            Some(prev) => b.binary(BinOp::Add, prev, term),
        });
    }
    acc.expect("k >= 1")
}

/// Dilated decode-step counterpart of [`causal_conv1d_prefill_dilated`] (spec 282): one new token
/// against a carried `[B, (K-1)*dilation, C]` conv cache. `out[c] = sum_{j=0..K-1} w[j,c] *
/// window[j*dilation, c]` where `window = cache ++ x` is `(K-1)*dilation + 1` long, so the last tap
/// lands on the current token. Returns `(out[B,1,C], new_cache[B,(K-1)*dilation,C])`; the cache drops
/// its oldest entry each step.
///
/// Not shared with [`causal_conv1d_decode`]: that one computes the `dilation == 1` math via
/// transpose-then-reduce on the hot Gated-DeltaNet decode path and is left as is.
pub fn causal_conv1d_decode_dilated(
    b: &Builder,
    x: Traced,
    w: Traced,
    cache: Traced,
    k: usize,
    dilation: usize,
) -> (Traced, Traced) {
    assert!(dilation >= 1, "dilation must be >= 1");
    let x_shape = b.aval(x).shape;
    let batch = x_shape[0];
    let c = *x_shape.last().expect("conv input is at least 1-D");
    let state_len = (k - 1) * dilation;
    let window = b.concat(1, &[cache, x]); // [B, (K-1)*dilation + 1, C]

    let mut acc: Option<Traced> = None;
    for j in 0..k {
        let off = j * dilation;
        let tap = b.slice(window, 1, off, off + 1); // [B, 1, C]
        let wj = b.reshape(b.slice(w, 0, j, j + 1), vec![1, 1, c]); // [1, 1, C] = w[j, :]
        let term = b.binary(BinOp::Mul, tap, wj);
        acc = Some(match acc {
            None => term,
            Some(prev) => b.binary(BinOp::Add, prev, term),
        });
    }
    let out = b.reshape(acc.expect("k >= 1"), vec![batch, 1, c]);
    let new_cache = b.slice(window, 1, 1, state_len + 1); // drop the oldest
    (out, new_cache)
}

/// Gated delta-net decode step (card 135c): the per-token recurrence of Qwen3-Next (qwen35moe) hybrid
/// linear-attention layers. The state is a per-head \[D,D\] matrix updated by the delta rule with a
/// scalar per-head forget gate. Source: `build_delta_net_autoregressive` in llama.cpp
/// `delta-net-base.cpp`.
///
/// Per head h, given forget-gate arg `g_h` (scalar, negative in practice so `exp(g_h) in (0,1)`),
/// delta-rule weight `beta_h` (scalar in (0,1)), and L2-normed q_h/k_h/v_h (caller normalizes):
///
/// ```text
/// q      = q * (1/sqrt(D))              # scale
/// decay  = exp(g)                       # per-head scalar gate in (0,1)
/// S      = S * decay                    # forget: scale down the whole D x D state
/// kv     = k @ S                        # [D]  = S^T k  (k treated as a 1-row matrix)
/// delta  = beta * (v - kv)              # [D]  delta-rule correction
/// S      = S + outer(k, delta)          # rank-1 update: S[i,j] += k[i]*delta[j]
/// o      = q @ S                        # [D]  = S^T q
/// ```
///
/// `q`/`k`/`v[1,H,1,D]`, `g`/`beta[1,H,1,1]` (scalar per head), `s_in[1,H,D,D]` (recurrent
/// state). Returns `(o[1,H,1,D], s_out[1,H,D,D])`. Wire `s_out` back as the carried state via
/// `finish_with_state`. Pure primitive composition; no new IR primitive.
///
/// Qwen3-Next repeats q/k from H_k=16 to H_v=32 heads before this call (forward-pass doc sec 3.3);
/// all inputs are at the V-head count.
pub fn gated_delta_net_decode(
    b: &Builder,
    q: Traced,    // [1, H, 1, D]  L2-normed query (caller normalizes)
    k: Traced,    // [1, H, 1, D]  L2-normed key (caller normalizes)
    v: Traced,    // [1, H, 1, D]  value
    g: Traced,    // [1, H, 1, 1]  forget-gate arg (exp(g) gives the per-head decay in (0,1))
    beta: Traced, // [1, H, 1, 1]  delta-rule weight in (0,1)
    s_in: Traced, // [1, H, D, D]  recurrent state (carried across decode steps)
) -> (Traced, Traced) {
    let d = b.aval(q).shape[3];
    // 1. Scale query by 1/sqrt(D).
    let q_scaled = b.binary_scalar(BinOp::Mul, q, Scalar::F32(1.0 / (d as f32).sqrt())); // [1,H,1,D]
    // 2. Per-head scalar decay = exp(g); g is negative in the model so decay is in (0,1).
    let decay = b.unary(UnOp::Exp, g); // [1,H,1,1]
    // 3. Forget: decay the whole D x D state. decay [1,H,1,1] broadcasts over [1,H,D,D].
    let s_decayed = b.binary(BinOp::Mul, s_in, decay); // [1,H,D,D]
    // 4. kv = k @ S  (equivalent to S^T k, treating k as a 1-row matrix).
    //    [1,H,1,D] @ [1,H,D,D] = [1,H,1,D].
    let kv = b.matmul(k, s_decayed); // [1,H,1,D]
    // 5. Delta-rule correction: beta * (v - S^T k).
    let residual = b.binary(BinOp::Sub, v, kv); // [1,H,1,D]
    let delta = b.binary(BinOp::Mul, beta, residual); // [1,H,1,D]
    // 6. Rank-1 outer-product update outer(k, delta) = k^col @ delta^row.
    //    k_col = transpose(k, [0,1,3,2]) = [1,H,D,1]; outer = [1,H,D,1] @ [1,H,1,D] = [1,H,D,D].
    let k_col = b.transpose(k, vec![0, 1, 3, 2]); // [1,H,D,1]
    let outer = b.matmul(k_col, delta); // [1,H,D,D]
    // 7. New state.
    let s_out = b.binary(BinOp::Add, s_decayed, outer); // [1,H,D,D]
    // 8. Output: q_scaled @ S_new  (equivalent to S_new^T q, since q @ S = S^T q as a vector).
    let o = b.matmul(q_scaled, s_out); // [1,H,1,D]
    (o, s_out)
}
