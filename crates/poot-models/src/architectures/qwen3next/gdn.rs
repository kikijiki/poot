use super::*;
use poot_graph_ir::ops::silu;

/// Qwen3-Next Gated DeltaNet (GDN) linear-attention decode step (card 135c).
///
/// The 30 recurrent layers in Qwen3-Next (qwen35moe / Qwen3.6-35B-A3B) use the delta-rule linear
/// attention with a per-head scalar forget gate. Sources: `build_layer_attn_linear` in
/// `qwen35moe.cpp` and `build_delta_net_autoregressive` in `delta-net-base.cpp`.
///
/// ## Forward pass (one decode step)
///
/// ```text
/// # Projections
/// qkv_mixed = x @ w_qkv                            # [1,1,conv_dim]  Q|K|V concatenated
/// z         = x @ w_gate                            # [1,1,value_dim] output gate
/// beta      = sigmoid(x @ w_beta)                   # [1,1,H_v]       per-head correction weight
/// alpha     = softplus((x @ w_alpha) + dt_bias)     # [1,1,H_v]       dt (positive)
/// g         = alpha * ssm_a                          # [1,1,H_v]       forget-gate arg (negative)
///
/// # Short causal conv over concatenated QKV, then silu activation
/// (conv_out, conv_cache_out) = causal_conv1d_decode(qkv_mixed, w_conv, conv_cache_in, K)
/// conv_out = silu(conv_out)                          # [1,1,conv_dim]
///
/// # Split into Q/K/V and reshape to [batch, heads, seq=1, head_dim]
/// q [1,H_k,1,D] = conv_out[:,:,0        : key_dim ]
/// k [1,H_k,1,D] = conv_out[:,:,key_dim  : 2*key_dim]
/// v [1,H_v,1,D] = conv_out[:,:,2*key_dim:          ]
///
/// # L2 normalize Q and K per head (no weight; eps on sum-of-squares, not mean)
/// q = l2_norm(q, eps); k = l2_norm(k, eps)
///
/// # GQA broadcast: repeat q,k from H_k to H_v (n_rep = H_v / H_k, 2 in the real model)
/// q = repeat_kv(q, n_rep)  # [1,H_v,1,D]
/// k = repeat_kv(k, n_rep)
///
/// # Gated delta-net recurrence (per head)
/// g, beta -> [1,H_v,1,1]
/// (o, s_out) = gated_delta_net_decode(q, k, v, g, beta, s_in)  # o: [1,H_v,1,D]
///
/// # Gated output RMSNorm: rmsnorm(o, norm_w) * silu(z_shaped)
/// o = rmsnorm(o, norm_w, eps) * silu(reshape(z, [1,H_v,1,D]))
///
/// # Flatten and out-project
/// cur = reshape(o, [1,1,value_dim]) @ w_out         # [1,1,H]
/// ```
///
/// ## Arguments
///
/// - `x`: input after pre-attention RMSNorm, shape `[B, 1, H]`. `B` is read off `x`'s shape (card 188), so
///   `B>1` computes `B` independent decode steps with no cross-row term.
/// - `w_qkv`: QKV in-proj `[H, conv_dim]` where `conv_dim = 2*key_dim + value_dim`
///   (`attn_qkv.weight` in the checkpoint).
/// - `w_gate`: output gate proj `[H, value_dim]` (`attn_gate.weight`).
/// - `w_conv`: depthwise conv kernel `[K, conv_dim]` (`ssm_conv1d.weight`).
/// - `w_beta`: beta proj `[H, num_v_heads]` (`ssm_beta.weight`).
/// - `w_alpha`: alpha / dt proj `[H, num_v_heads]` (`ssm_alpha.weight`).
/// - `dt_bias`: dt bias added before softplus `[num_v_heads]` (`ssm_dt.bias`).
/// - `ssm_a`: per-head log-decay `[num_v_heads]` (stored as negative so `exp(g) in (0,1)`; `ssm_a`).
/// - `norm_w`: gated RMSNorm weight `[head_dim]` (`ssm_norm.weight`).
/// - `w_out`: out proj `[value_dim, H]` (`ssm_out.weight`).
/// - `conv_cache_in`: conv state `[B, K-1, conv_dim]` (caller carries via `finish_with_state`).
/// - `s_in`: GDN recurrent state `[B, num_v_heads, head_dim, head_dim]`
///   (`ssm_states_all`; caller carries via `finish_with_state`).
/// - `num_k_heads`: K/Q head count H_k (`ssm.group_count` = 16 in the real model).
/// - `num_v_heads`: V head count H_v (`ssm.time_step_rank` = 32 in the real model).
/// - `head_dim`: per-head dimension D (`ssm.state_size` = 128; head_k_dim == head_v_dim).
/// - `conv_k`: conv kernel size K (`ssm.conv_kernel` = 4).
/// - `eps`: epsilon for RMSNorm and L2 norm (1e-6 in the real model).
/// - `head_order`: V-head order of the weights' checkpoint source ([`GdnHeadOrder`]).
///
/// Returns `(out [B, 1, H], conv_cache_out [B, K-1, conv_dim], s_out [B, H_v, D, D])`.
/// Wire the two state pairs via `finish_with_state`.
#[allow(clippy::too_many_arguments)]
pub fn qwen3next_gdn_block(
    b: &Builder,
    x: Traced,
    w_qkv: Traced,
    w_gate: Traced,
    w_conv: Traced,
    w_beta: Traced,
    w_alpha: Traced,
    dt_bias: Traced,
    ssm_a: Traced,
    norm_w: Traced,
    w_out: Traced,
    conv_cache_in: Traced,
    s_in: Traced,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
    conv_k: usize,
    eps: f32,
    head_order: GdnHeadOrder,
) -> (Traced, Traced, Traced) {
    assert!(
        num_v_heads.is_multiple_of(num_k_heads),
        "num_v_heads={num_v_heads} must be divisible by num_k_heads={num_k_heads}"
    );
    let qkv_mixed = linear(b, x, w_qkv, None);
    let z = linear(b, x, w_gate, None);
    let beta_raw = linear(b, x, w_beta, None);
    let beta = sigmoid(b, beta_raw);
    let alpha_raw = linear(b, x, w_alpha, None);
    let alpha = softplus(b, b.binary(BinOp::Add, alpha_raw, dt_bias));
    let g = b.binary(BinOp::Mul, alpha, ssm_a);
    let (o_flat, conv_cache_out, s_out) = qwen3next_gdn_projected(
        b,
        qkv_mixed,
        z,
        beta,
        g,
        w_conv,
        norm_w,
        conv_cache_in,
        s_in,
        num_k_heads,
        num_v_heads,
        head_dim,
        conv_k,
        eps,
        head_order,
    );
    let cur = linear(b, o_flat, w_out, None);
    (cur, conv_cache_out, s_out)
}

/// Projection-independent GDN decode semantics shared with Qwen3.5 packed linears.
///
/// Projected QKV and Z plus the graph-computed beta and decay values enter directly. The first return
/// value is the flattened gated GDN result before the output projection. Dense beta/alpha projections
/// can therefore stay dense while QKV, Z, and output projection ownership remains with a packed caller.
/// `head_order` is the V-head order of the caller's checkpoint source.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen3next_gdn_projected(
    b: &Builder,
    qkv_mixed: Traced,
    z: Traced,
    beta: Traced,
    g: Traced,
    w_conv: Traced,
    norm_w: Traced,
    conv_cache_in: Traced,
    s_in: Traced,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
    conv_k: usize,
    eps: f32,
    head_order: GdnHeadOrder,
) -> (Traced, Traced, Traced) {
    assert!(
        num_v_heads.is_multiple_of(num_k_heads),
        "num_v_heads={num_v_heads} must be divisible by num_k_heads={num_k_heads}"
    );
    // Card 188: derive the leading/batch axis from the projected QKV value instead of hardcoding 1, so this
    // one function serves both the single-sequence and batched decode tracers.
    let batch = b.aval(qkv_mixed).shape[0];
    let n_rep = num_v_heads / num_k_heads;
    let key_dim = num_k_heads * head_dim;
    let value_dim = num_v_heads * head_dim;
    let conv_dim = 2 * key_dim + value_dim;

    // 1. Depthwise causal conv1d over concatenated QKV channels, then silu.
    //    causal_conv1d_decode: x[B,1,C], w[K,C], cache[B,K-1,C] -> (out[B,1,C], new_cache[B,K-1,C]).
    let (conv_out, conv_cache_out) =
        causal_conv1d_decode(b, qkv_mixed, w_conv, conv_cache_in, conv_k); // [B,1,conv_dim]
    let conv_out = silu(b, conv_out); // [B, 1, conv_dim]

    // 3. Split conv output into Q (key_dim), K (key_dim), V (value_dim).
    let q_flat = b.slice(conv_out, 2, 0, key_dim); // [B, 1, key_dim]
    let k_flat = b.slice(conv_out, 2, key_dim, 2 * key_dim); // [B, 1, key_dim]
    let v_flat = b.slice(conv_out, 2, 2 * key_dim, conv_dim); // [B, 1, value_dim]

    // 4. Reshape to [B, heads, seq=1, head_dim] for per-head ops.
    let q = b.reshape(q_flat, vec![batch, num_k_heads, 1, head_dim]); // [B, H_k, 1, D]
    let k = b.reshape(k_flat, vec![batch, num_k_heads, 1, head_dim]); // [B, H_k, 1, D]
    let v = b.reshape(v_flat, vec![batch, num_v_heads, 1, head_dim]); // [B, H_v, 1, D]

    // 5. L2 normalize Q and K per head. ggml_l2_norm: x / sqrt(sum(x^2) + eps), no weight.
    let q = l2_norm_last(b, q, eps); // [B, H_k, 1, D]
    let k = l2_norm_last(b, k, eps); // [B, H_k, 1, D]

    // 6. GQA broadcast: repeat q,k from H_k to H_v heads in the source's head order.
    let q = head_order.expand(b, q, n_rep); // [B, H_v, 1, D]
    let k = head_order.expand(b, k, n_rep); // [B, H_v, 1, D]

    // 7. Reshape g and beta from [B, 1, H_v] to [B, H_v, 1, 1] for gated_delta_net_decode.
    let g = b.reshape(g, vec![batch, num_v_heads, 1, 1]); // [B, H_v, 1, 1]
    let beta = b.reshape(beta, vec![batch, num_v_heads, 1, 1]); // [B, H_v, 1, 1]

    // 8. Gated delta-net recurrence. s_in: [B, H_v, D, D]. Batch-generic: each row reads only its own row of
    // q/k/v/g/beta/s_in, so this computes B independent recurrences.
    let (o, s_out) = gated_delta_net_decode(b, q, k, v, g, beta, s_in);
    // o: [B, H_v, 1, D], s_out: [B, H_v, D, D]

    // 9. Gated output RMSNorm: rmsnorm(o, norm_w, eps) * silu(z_shaped).
    //    norm_w [D] broadcasts over [B, H_v, 1, D].
    let o_norm = rmsnorm(b, o, norm_w, eps); // [B, H_v, 1, D]
    //    z [B, 1, value_dim] -> [B, H_v, 1, D] for elementwise gate.
    let z_shaped = b.reshape(z, vec![batch, num_v_heads, 1, head_dim]); // [B, H_v, 1, D]
    let silu_z = silu(b, z_shaped); // [B, H_v, 1, D]
    let o_gated = b.binary(BinOp::Mul, o_norm, silu_z); // [B, H_v, 1, D]

    // 10. Flatten for the caller-owned output projection.
    let o_flat = b.reshape(o_gated, vec![batch, 1, value_dim]); // [B, 1, value_dim]
    (o_flat, conv_cache_out, s_out)
}

/// Qwen3-Next Gated DeltaNet (GDN) linear-attention prefill block (spec 139 P3 / card 153): the L>1
/// counterpart of [`qwen3next_gdn_block`], for the first `L` positions of a prompt. It always starts a new
/// conv cache (matching [`causal_conv1d_prefill`]'s implicit zero left pad, spec 139 B5); `s_in` is threaded
/// through, typically zero.
///
/// Numerics: the chunked UT-transform recurrence ([`gdn_prefill_chunked`]) is mathematically equivalent to but
/// not bit-identical to `L` sequential [`qwen3next_gdn_block`] calls. At real dims (H_v=32, D=128, C=64) the
/// post-prefill GDN state differs by ~2e-7 to ~6e-4 absolute depending on decay magnitude, within the ~1e-4
/// relative real-GPU-vs-CPU-oracle tolerance convention. The divergence shrinks over further decode steps,
/// since GDN's per-step gating (`exp(g) in (0,1)`) contracts the state. Same accepted class as card 059's
/// dense chunked-vs-token-by-token prefill (a near-tied argmax can flip). See
/// `gdn_decode_continuation_error_shrinks_after_chunked_prefill_real_dims` and
/// `gdn_prefill_chunked_decay_magnitude_sweep_real_dims` in `poot-eval`'s `linear_attention` tests.
///
/// Same weights, projection shapes and op order as [`qwen3next_gdn_block`]; only the two ops that must see the
/// whole chunk change: [`causal_conv1d_decode`] -> [`causal_conv1d_prefill`] and [`gated_delta_net_decode`] ->
/// [`gdn_prefill_chunked`]. The extra reshape/transpose moves the `L` axis from middle (channel-last,
/// `causal_conv1d_prefill`'s `[1,L,C]`) to third (head-major, `gdn_prefill_chunked`'s `[1,H,L,D]`); decode's
/// `L=1` makes it a no-op reshape.
///
/// ## Forward pass (all `L` positions at once)
///
/// ```text
/// qkv_mixed = x @ w_qkv                              # [1,L,conv_dim]
/// z         = x @ w_gate                              # [1,L,value_dim]
/// beta      = sigmoid(x @ w_beta)                     # [1,L,H_v]
/// alpha     = softplus((x @ w_alpha) + dt_bias)       # [1,L,H_v]
/// g         = alpha * ssm_a                            # [1,L,H_v]
///
/// # Batched causal conv over concatenated QKV, then silu activation (fresh cache: causal_conv1d_prefill
/// # zero-pads its own left edge, so there is no conv_cache_in argument here)
/// conv_out       = causal_conv1d_prefill(qkv_mixed, w_conv, K)
/// conv_out       = silu(conv_out)                      # [1,L,conv_dim]
/// conv_cache_out = last (K-1) RAW qkv_mixed positions   # [1,K-1,conv_dim] - seeds decode's cache
///
/// # Split into Q/K/V, reshape+transpose to [batch, heads, seq=L, head_dim]
/// q [1,H_k,L,D] = conv_out[:,:,0        : key_dim ]
/// k [1,H_k,L,D] = conv_out[:,:,key_dim  : 2*key_dim]
/// v [1,H_v,L,D] = conv_out[:,:,2*key_dim:          ]
///
/// # L2 normalize Q and K per head (no weight; eps on sum-of-squares, not mean)
/// q = l2_norm(q, eps); k = l2_norm(k, eps)
///
/// # Chunked gated delta-net recurrence (per head; q,k repeat from H_k to H_v in the source head order
/// # just before the call)
/// g, beta -> [1,H_v,L,1]
/// (o, s_out) = gdn_prefill_chunked(q, k, v, g, beta, s_in, tril_incl, tril_strict, C)  # o: [1,H_v,L,D]
///
/// # Gated output RMSNorm: rmsnorm(o, norm_w) * silu(z_shaped)
/// o = rmsnorm(o, norm_w, eps) * silu(reshape+transpose(z, [1,H_v,L,D]))
///
/// # Flatten and out-project
/// cur = reshape(transpose(o), [1,L,value_dim]) @ w_out         # [1,L,H]
/// ```
///
/// ## Arguments
///
/// Same weights as [`qwen3next_gdn_block`] (`w_qkv`, `w_gate`, `w_conv`, `w_beta`, `w_alpha`,
/// `dt_bias`, `ssm_a`, `norm_w`, `w_out`), plus:
///
/// - `x`: input after pre-attention RMSNorm, shape `[1, L, H]` (L>1).
/// - `s_in`: GDN recurrent state `[1, num_v_heads, head_dim, head_dim]` (zero for the very first
///   prefill of a sequence; threaded for generality, like `gdn_prefill_chunked`'s own `s_in`).
/// - `tril_incl`: `[1,1,C,C]` lower-triangular ones INCLUDING the diagonal (S2, `gdn_prefill_chunked`).
/// - `tril_strict`: `[1,1,C,C]` STRICTLY lower-triangular ones, zero diagonal (S2).
/// - `num_k_heads`, `num_v_heads`, `head_dim`, `conv_k`, `eps`: same meaning as
///   [`qwen3next_gdn_block`].
/// - `chunk`: the static UT-transform chunk size `C` (`gdn_prefill_chunked`'s `chunk`).
/// - `head_order`: same meaning as [`qwen3next_gdn_block`].
///
/// Returns `(out [1, L, H], conv_cache_out [1, K-1, conv_dim], s_out [1, H_v, D, D])` - wire both state
/// pairs (like [`qwen3next_gdn_block`]'s) to seed the decode block's `conv_cache_in`/`s_in` for the
/// following decode steps.
#[allow(clippy::too_many_arguments)]
pub fn qwen3next_gdn_prefill_block(
    b: &Builder,
    x: Traced,
    w_qkv: Traced,
    w_gate: Traced,
    w_conv: Traced,
    w_beta: Traced,
    w_alpha: Traced,
    dt_bias: Traced,
    ssm_a: Traced,
    norm_w: Traced,
    w_out: Traced,
    s_in: Traced,
    tril_incl: Traced,
    tril_strict: Traced,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
    conv_k: usize,
    chunk: usize,
    eps: f32,
    head_order: GdnHeadOrder,
) -> (Traced, Traced, Traced) {
    assert!(
        num_v_heads.is_multiple_of(num_k_heads),
        "num_v_heads={num_v_heads} must be divisible by num_k_heads={num_k_heads}"
    );
    let qkv_mixed = linear(b, x, w_qkv, None);
    let z = linear(b, x, w_gate, None);
    let beta_raw = linear(b, x, w_beta, None);
    let beta = sigmoid(b, beta_raw);
    let alpha_raw = linear(b, x, w_alpha, None);
    let alpha = softplus(b, b.binary(BinOp::Add, alpha_raw, dt_bias));
    let g = b.binary(BinOp::Mul, alpha, ssm_a);
    let (o_flat, conv_cache_out, s_out) = qwen3next_gdn_prefill_projected(
        b,
        qkv_mixed,
        z,
        beta,
        g,
        w_conv,
        norm_w,
        s_in,
        tril_incl,
        tril_strict,
        num_k_heads,
        num_v_heads,
        head_dim,
        conv_k,
        chunk,
        eps,
        head_order,
    );
    let cur = linear(b, o_flat, w_out, None);
    (cur, conv_cache_out, s_out)
}

/// Projection-independent GDN prefill semantics shared with Qwen3.5 packed linears.
///
/// The projected QKV and Z plus graph-computed beta and decay values are consumed without another weight
/// representation. The first return value remains in `[1, L, value_dim]` form for the caller's
/// output projection. `head_order` is the V-head order of the caller's checkpoint source.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen3next_gdn_prefill_projected(
    b: &Builder,
    qkv_mixed: Traced,
    z: Traced,
    beta: Traced,
    g: Traced,
    w_conv: Traced,
    norm_w: Traced,
    s_in: Traced,
    tril_incl: Traced,
    tril_strict: Traced,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
    conv_k: usize,
    chunk: usize,
    eps: f32,
    head_order: GdnHeadOrder,
) -> (Traced, Traced, Traced) {
    assert!(
        num_v_heads.is_multiple_of(num_k_heads),
        "num_v_heads={num_v_heads} must be divisible by num_k_heads={num_k_heads}"
    );
    let n_rep = num_v_heads / num_k_heads;
    let key_dim = num_k_heads * head_dim;
    let value_dim = num_v_heads * head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let l = b.aval(qkv_mixed).shape[1];

    // 1. Batched depthwise causal conv1d over concatenated QKV channels (PREFILL form), then silu.
    //    causal_conv1d_prefill: x[1,L,C], w[K,C] -> out[1,L,C] (implicit zero left-pad, spec 139 B5).
    let conv_out = causal_conv1d_prefill(b, qkv_mixed, w_conv, conv_k); // [1, L, conv_dim]
    let conv_out = silu(b, conv_out); // [1, L, conv_dim]

    // Final conv cache to seed the decode block: the last (K-1) RAW (pre-conv) qkv_mixed positions,
    // matching causal_conv1d_decode's cache semantics (the last K-1 raw inputs). Reuses
    // causal_conv1d_prefill's own left-pad trick (a zeroed first slice, broadcast) so this is correct
    // even when L < K-1.
    let first = b.slice(qkv_mixed, 1, 0, 1); // [1, 1, conv_dim]
    let zero_slice = b.binary_scalar(BinOp::Mul, first, Scalar::F32(0.0)); // [1, 1, conv_dim] zeros
    let zeros = b.broadcast(zero_slice, vec![1, conv_k - 1, conv_dim]); // [1, K-1, conv_dim]
    let x_pad = b.concat(1, &[zeros, qkv_mixed]); // [1, K-1+L, conv_dim]
    let conv_cache_out = b.slice(x_pad, 1, l, l + conv_k - 1); // [1, K-1, conv_dim]

    // 3. Split conv output into Q (key_dim), K (key_dim), V (value_dim). Still channel-last [1,L,*].
    let q_flat = b.slice(conv_out, 2, 0, key_dim); // [1, L, key_dim]
    let k_flat = b.slice(conv_out, 2, key_dim, 2 * key_dim); // [1, L, key_dim]
    let v_flat = b.slice(conv_out, 2, 2 * key_dim, conv_dim); // [1, L, value_dim]

    // 4. Reshape + transpose to [1, heads, L, head_dim] (gdn_prefill_chunked's head-major layout).
    let q = b.reshape(q_flat, vec![1, l, num_k_heads, head_dim]);
    let q = b.transpose(q, vec![0, 2, 1, 3]); // [1, H_k, L, D]
    let k = b.reshape(k_flat, vec![1, l, num_k_heads, head_dim]);
    let k = b.transpose(k, vec![0, 2, 1, 3]); // [1, H_k, L, D]
    let v = b.reshape(v_flat, vec![1, l, num_v_heads, head_dim]);
    let v = b.transpose(v, vec![0, 2, 1, 3]); // [1, H_v, L, D]

    // 5. L2 normalize Q and K per head. ggml_l2_norm: x / sqrt(sum(x^2) + eps), no weight.
    let q = l2_norm_last(b, q, eps); // [1, H_k, L, D]
    let k = l2_norm_last(b, k, eps); // [1, H_k, L, D]

    // 6. Reshape g and beta from [1, L, H_v] to [1, H_v, L, 1] (gdn_prefill_chunked's layout). Unlike
    //    the decode block (L=1, a trivial reshape), this needs a real transpose.
    let g = b.reshape(g, vec![1, l, num_v_heads, 1]);
    let g = b.transpose(g, vec![0, 2, 1, 3]); // [1, H_v, L, 1]
    let beta = b.reshape(beta, vec![1, l, num_v_heads, 1]);
    let beta = b.transpose(beta, vec![0, 2, 1, 3]); // [1, H_v, L, 1]

    // 7. Chunked gated delta-net recurrence over all L positions. q/k are expanded to H_v here in the
    //    source's head order, so gdn_prefill_chunked's own tiled repeat sees n_rep = 1 and emits nothing.
    //    For tiled sources this emits the same equations, in the same order, as the repeat inside the call.
    let q = head_order.expand(b, q, n_rep); // [1, H_v, L, D]
    let k = head_order.expand(b, k, n_rep); // [1, H_v, L, D]
    let (o, s_out) = gdn_prefill_chunked(b, q, k, v, g, beta, s_in, tril_incl, tril_strict, chunk);
    // o: [1, H_v, L, D], s_out: [1, H_v, D, D]

    // 8. Gated output RMSNorm: rmsnorm(o, norm_w, eps) * silu(z_shaped).
    let o_norm = rmsnorm(b, o, norm_w, eps); // [1, H_v, L, D]
    let z_shaped = b.reshape(z, vec![1, l, num_v_heads, head_dim]);
    let z_shaped = b.transpose(z_shaped, vec![0, 2, 1, 3]); // [1, H_v, L, D]
    let silu_z = silu(b, z_shaped); // [1, H_v, L, D]
    let o_gated = b.binary(BinOp::Mul, o_norm, silu_z); // [1, H_v, L, D]

    // 9. Flatten (back to channel-last) for the caller-owned output projection.
    let o_back = b.transpose(o_gated, vec![0, 2, 1, 3]); // [1, L, H_v, D]
    let o_flat = b.reshape(o_back, vec![1, l, value_dim]); // [1, L, value_dim]
    (o_flat, conv_cache_out, s_out)
}

/// Card 188 Increment 3 (spec `specs/188-qwen3next-batching/spec.md`, FR-004): slot-addressed variant of
/// [`qwen3next_gdn_prefill_block`]. Writes a newly admitted sequence's `(conv_cache_out, s_out)` into row `slot`
/// of a shared `[n_slots, ...]` GDN state pool without disturbing any other slot or layer.
///
/// Calls [`qwen3next_gdn_prefill_block`] unmodified, then composes two `DynamicUpdateSlice` writes on axis 0
/// (the slot axis); no new IR primitive. `DynamicUpdateSlice` (`poot-graph-ir/src/op.rs`) passes every row
/// outside `[slot, slot+1)` through byte-unchanged, so slot isolation holds structurally.
///
/// `slot` is a graph value (like [`poot_graph_ir::builder::Builder::dynamic_update_slice_dyn`]'s `index`), as in
/// the shared-pool KV write (`poot-models/src/qwen2.rs::scatter_shared_pool`), so one captured graph can serve
/// different target slots.
///
/// ## Arguments
///
/// Same as [`qwen3next_gdn_prefill_block`] (`x`, weights, `s_in`, `tril_incl`, `tril_strict`, head/dim
/// config, `head_order`), plus:
///
/// - `conv_pool`: the shared conv-cache pool for this GDN layer, `[n_slots, K-1, conv_dim]`.
/// - `ssm_pool`: the shared SSM-state pool for this GDN layer, `[n_slots, H_v, D, D]`.
/// - `slot`: scalar index (graph value) of the pool row this admission writes.
///
/// Returns `(cur [1, L, H], conv_pool_out [n_slots, K-1, conv_dim], ssm_pool_out [n_slots, H_v, D, D])` -
/// the full updated pools, ready to become the next step's pool inputs.
#[allow(clippy::too_many_arguments)]
pub fn qwen3next_gdn_prefill_into_slot(
    b: &Builder,
    x: Traced,
    w_qkv: Traced,
    w_gate: Traced,
    w_conv: Traced,
    w_beta: Traced,
    w_alpha: Traced,
    dt_bias: Traced,
    ssm_a: Traced,
    norm_w: Traced,
    w_out: Traced,
    s_in: Traced,
    tril_incl: Traced,
    tril_strict: Traced,
    conv_pool: Traced,
    ssm_pool: Traced,
    slot: Traced,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
    conv_k: usize,
    chunk: usize,
    eps: f32,
    head_order: GdnHeadOrder,
) -> (Traced, Traced, Traced) {
    let (cur, conv_cache_out, s_out) = qwen3next_gdn_prefill_block(
        b,
        x,
        w_qkv,
        w_gate,
        w_conv,
        w_beta,
        w_alpha,
        dt_bias,
        ssm_a,
        norm_w,
        w_out,
        s_in,
        tril_incl,
        tril_strict,
        num_k_heads,
        num_v_heads,
        head_dim,
        conv_k,
        chunk,
        eps,
        head_order,
    );
    // Row-local writes: axis 0 is the slot axis for both pools. DynamicUpdateSlice only overwrites
    // [slot, slot+1) of its operand; every other row is passed through unchanged (op.rs semantics).
    let conv_pool_out = b.dynamic_update_slice_dyn(conv_pool, conv_cache_out, slot, 0);
    let ssm_pool_out = b.dynamic_update_slice_dyn(ssm_pool, s_out, slot, 0);
    (cur, conv_pool_out, ssm_pool_out)
}

/// Card 188 Increment 5: zero one row of the GDN state pool (every GDN layer's `conv_cache`/`ssm_state`),
/// leaving the attention KV pool and every other GDN pool row byte-unchanged. `slot` is a Rust-level literal
/// (the graph is retraced per target slot; admission time only, off the hot path).
///
/// A newly admitted sequence's GDN pool row must be zeroed before its first batched-decode step (unlike the
/// attention pool, whose stale rows are masked out of the softmax; see
/// `qwen3next_decode_trace_batched_shared_pool`). An engine's admission path calls this graph once via
/// `run_resident_kv`-style execution.
///
/// The attention KV state pairs use the same names/shapes as the decode graphs so the caller's full
/// `caches: Vec<DeviceBuffer>` (in `g.state` order) binds directly. They are a passthrough through a fresh
/// `Reshape`-to-same-shape node, not a same-`ValueId` alias: an alias would give the ping-pong buffer swap
/// nothing to copy into the other parity's buffer, which is unsafe under replay (see
/// `poot-gpu/src/lib.rs`'s `DecodeCache::kv_pairs`).
///
/// The GDN pool's zero value is synthesized in-graph (`existing_row * 0.0`), so no extra weight needs binding.
pub fn qwen3next_zero_gdn_slot_trace(
    cfg: &Qwen3NextConfig,
    pool_slots: usize,
    gdn_n_slots: usize,
    slot: usize,
) -> Graph {
    assert!(
        slot < gdn_n_slots,
        "slot {slot} out of range (gdn_n_slots={gdn_n_slots})"
    );
    let b = Builder::new();
    let mut state: Vec<(Traced, Traced)> = Vec::new();
    let mut any: Option<Traced> = None;

    for li in 0..cfg.n_layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        if cfg.is_attn_layer(li) {
            let (nkv, hd) = (cfg.n_kv_heads, cfg.head_dim);
            let shape = vec![pool_slots, nkv, hd];
            let kc = b.state_input(
                &p("kv.k_cache"),
                TensorType::f32(shape.clone()),
                StateRole::Recurrent,
            );
            let vc = b.state_input(
                &p("kv.v_cache"),
                TensorType::f32(shape.clone()),
                StateRole::Recurrent,
            );
            // Passthrough: a real per-step Reshape-to-same-shape eqn (not a same-id alias - see the
            // doc comment above), so the ping-pong buffer swap always has something to copy.
            let kc_out = b.reshape(kc, shape.clone());
            let vc_out = b.reshape(vc, shape);
            state.push((kc, kc_out));
            state.push((vc, vc_out));
            any = Some(kc_out);
        } else {
            let (hk, hv, hd, ck) = (
                cfg.gdn_num_k_heads,
                cfg.gdn_num_v_heads,
                cfg.gdn_head_dim,
                cfg.conv_k,
            );
            let key_dim = hk * hd;
            let value_dim = hv * hd;
            let conv_dim = 2 * key_dim + value_dim;
            let conv_pool = b.state_input(
                &p("gdn.conv_cache"),
                TensorType::f32(vec![gdn_n_slots, ck - 1, conv_dim]),
                StateRole::Recurrent,
            );
            let ssm_pool = b.state_input(
                &p("gdn.ssm_state"),
                TensorType::f32(vec![gdn_n_slots, hv, hd, hd]),
                StateRole::Recurrent,
            );
            // Zero-shaped update rows, synthesized from the pool itself (no new named Const).
            let zero_conv = {
                let row = b.slice(conv_pool, 0, 0, 1); // [1, K-1, conv_dim]
                b.binary_scalar(BinOp::Mul, row, Scalar::F32(0.0))
            };
            let zero_ssm = {
                let row = b.slice(ssm_pool, 0, 0, 1); // [1, H_v, D, D]
                b.binary_scalar(BinOp::Mul, row, Scalar::F32(0.0))
            };
            let conv_pool_out = b.dynamic_update_slice(conv_pool, zero_conv, slot, 0);
            let ssm_pool_out = b.dynamic_update_slice(ssm_pool, zero_ssm, slot, 0);
            state.push((conv_pool, conv_pool_out));
            state.push((ssm_pool, ssm_pool_out));
            any = Some(conv_pool_out);
        }
    }
    let out = any.expect("qwen3next_zero_gdn_slot_trace needs at least one layer");
    b.finish_with_state(out, &state)
}

/// Card 188 Increment 4 (spec `specs/188-qwen3next-batching/spec.md`, FR-003/FR-004/FR-005): the batched
/// decode counterpart of [`qwen3next_gdn_prefill_into_slot`]: one GDN decode step for `batch` concurrent
/// sequences, each addressed into a shared `[n_slots, ...]` GDN state pool. Pure primitive composition.
///
/// 1. Gather: each row `r`'s `(conv_cache, ssm_state)` is read from the pool at slot `gdn_slot_map[r]` via an
///    axis-0 [`Builder::gather`] (`gdn_slot_map` is a `[batch]` i32 graph value, the GDN analogue of the
///    attention pool's `[batch,cap]` `Slot::SlotMap`; see `poot-models/src/qwen2.rs::gather_shared_pool`).
/// 2. Compute: [`qwen3next_gdn_block`] unmodified (`B = batch` from `x`'s shape); `B` independent recurrences.
/// 3. Scatter-back: each row's updated state is written into its own pool slot by a `batch`-length fold of
///    [`Builder::dynamic_update_slice_dyn`] writes, each reading the previous write's output (the pattern of
///    `scatter_shared_pool`). Pool rows not named by any `gdn_slot_map` entry pass through byte-unchanged
///    (FR-005).
///
/// ## In-place state hazard
///
/// The scatter-back fold is a sequential producer/consumer chain, so there is no WAR hazard within a step.
/// Row `r`'s write depends only on `conv_cache_out[r]`/`s_out[r]`, never another row's write, so fold order does
/// not affect the result. Cross-step state (`finish_with_state`'s ping-pong twin buffer, spec 137) is wired as
/// in every single-sequence GDN caller.
///
/// ## Arguments
///
/// Same as [`qwen3next_gdn_block`] (`x [batch,1,H]`, weights, head/dim config, `head_order`), plus:
///
/// - `conv_pool`: the shared conv-cache pool for this GDN layer, `[n_slots, K-1, conv_dim]`.
/// - `ssm_pool`: the shared SSM-state pool for this GDN layer, `[n_slots, H_v, D, D]`.
/// - `gdn_slot_map`: `[batch]` i32 - row `r`'s target GDN pool slot this step. Distinct active rows must
///   carry distinct slots (an engine invariant, not checked here).
/// - `batch`: number of decode rows in this step (equals `x`'s leading dim; passed to size intermediate
///   slices, as in `gather_shared_pool`).
///
/// Returns `(cur [batch,1,H], conv_pool_out [n_slots,K-1,conv_dim], ssm_pool_out [n_slots,H_v,D,D])`.
#[allow(clippy::too_many_arguments)]
pub fn qwen3next_gdn_decode_batched_pool(
    b: &Builder,
    x: Traced,
    w_qkv: Traced,
    w_gate: Traced,
    w_conv: Traced,
    w_beta: Traced,
    w_alpha: Traced,
    dt_bias: Traced,
    ssm_a: Traced,
    norm_w: Traced,
    w_out: Traced,
    conv_pool: Traced,
    ssm_pool: Traced,
    gdn_slot_map: Traced,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
    conv_k: usize,
    eps: f32,
    batch: usize,
    head_order: GdnHeadOrder,
) -> (Traced, Traced, Traced) {
    // 1. GATHER: row r's own state = pool[gdn_slot_map[r]].
    let conv_cache_in = b.gather(conv_pool, 0, gdn_slot_map); // [batch, K-1, conv_dim]
    let s_in = b.gather(ssm_pool, 0, gdn_slot_map); // [batch, H_v, D, D]

    // 2. COMPUTE: qwen3next_gdn_block, unmodified, B=batch derived from x's own shape.
    let (cur, conv_cache_out, s_out) = qwen3next_gdn_block(
        b,
        x,
        w_qkv,
        w_gate,
        w_conv,
        w_beta,
        w_alpha,
        dt_bias,
        ssm_a,
        norm_w,
        w_out,
        conv_cache_in,
        s_in,
        num_k_heads,
        num_v_heads,
        head_dim,
        conv_k,
        eps,
        head_order,
    );

    // 3. SCATTER-BACK: a batch-length fold of row-local DynamicUpdateSlice writes (as in scatter_shared_pool),
    // each depending on the previous row's write output; rows not named by gdn_slot_map are unchanged.
    let conv_pool_out = (0..batch).fold(conv_pool, |pool, row| {
        let update_row = b.slice(conv_cache_out, 0, row, row + 1); // [1, K-1, conv_dim]
        let slot_row = b.reshape(b.slice(gdn_slot_map, 0, row, row + 1), vec![]); // scalar
        b.dynamic_update_slice_dyn(pool, update_row, slot_row, 0)
    });
    let ssm_pool_out = (0..batch).fold(ssm_pool, |pool, row| {
        let update_row = b.slice(s_out, 0, row, row + 1); // [1, H_v, D, D]
        let slot_row = b.reshape(b.slice(gdn_slot_map, 0, row, row + 1), vec![]); // scalar
        b.dynamic_update_slice_dyn(pool, update_row, slot_row, 0)
    });

    (cur, conv_pool_out, ssm_pool_out)
}
