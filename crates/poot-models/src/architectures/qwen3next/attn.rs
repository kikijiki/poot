use super::*;

/// Qwen3-Next gated full-attention decode step (section 4).
///
/// This is the 10-layer full-attention block of Qwen3-Next (qwen35moe / Qwen3.6-35B-A3B), NOT
/// plain Qwen3 attention. It adds a per-head output gate: the attention output is multiplied by
/// `sigmoid(gate)` BEFORE the o-projection.
///
/// ## Query/gate interleave (section 4.2)
///
/// `wq` produces `n_heads * 2 * head_dim` values. Within each head's `2*head_dim` block:
/// - indices `[0 .. head_dim)` = query
/// - indices `[head_dim .. 2*head_dim)` = gate
///
/// For head `h`, weight rows `[h*2*head_dim .. h*2*head_dim+head_dim]` = query and
/// `[h*2*head_dim+head_dim .. (h+1)*2*head_dim]` = gate. The split is implemented by
/// reshaping the flat output to `[1, 1, n_heads, 2*head_dim]` and slicing axis 3.
///
/// ## QK-norm placement
///
/// Per-head RMSNorm over `head_dim` is applied AFTER the projection and BEFORE RoPE, matching
/// the Qwen3-style QK-norm.
///
/// ## RoPE
///
/// Partial NeoX RoPE: only the first `rotary_dim` of `head_dim` dims are rotated. For text-only
/// decode, IMRoPE with sections `[11,11,10,0]` reduces to plain NeoX partial rope (section 4.4).
/// `cos`/`sin` tables have shape `[max_pos, rotary_dim]`.
///
/// ## KV cache threading
///
/// The caller creates the KV cache
/// state inputs via `b.state_input()` and passes them here; this function scatters the new
/// token's k/v, slices the valid prefix, and returns the updated cache tensors. The caller
/// registers the `(k_cache_in, k_cache_out)` / `(v_cache_in, v_cache_out)` pairs with
/// `finish_with_state`.
///
/// ## Arguments
///
/// - `x`: input after the pre-attention RMSNorm, shape `[1, 1, H]`.
/// - `wq`: query+gate weight `[H, n_heads * 2 * head_dim]` (per-head interleaved layout).
/// - `wk`: key weight `[H, n_kv_heads * head_dim]`.
/// - `wv`: value weight `[H, n_kv_heads * head_dim]`.
/// - `wo`: output projection `[n_heads * head_dim, H]`.
/// - `q_norm_w`: per-head Q RMSNorm weight `[head_dim]`.
/// - `k_norm_w`: per-head K RMSNorm weight `[head_dim]`.
/// - `cos`, `sin`: partial RoPE tables `[max_pos, rotary_dim]`.
/// - `pos`: scalar i32 Traced (runtime position, used for RoPE gather and KV cache scatter).
/// - `k_cache_in`, `v_cache_in`: carried KV caches `[1, n_kv_heads, cap, head_dim]`.
/// - `n_heads`, `n_kv_heads`, `head_dim`: attention dims (`n_rep = n_heads / n_kv_heads`).
/// - `pos_idx`: static position index (= the current decode step, for slicing the valid prefix).
/// - `eps`: RMSNorm epsilon.
///
/// Returns `(out [1, 1, H], k_cache_out, v_cache_out)`.
#[allow(clippy::too_many_arguments)]
pub fn qwen3next_gated_attention(
    b: &Builder,
    x: Traced,
    wq: Traced,
    wk: Traced,
    wv: Traced,
    wo: Traced,
    q_norm_w: Traced,
    k_norm_w: Traced,
    cos: Traced,
    sin: Traced,
    pos: Traced,
    k_cache_in: Traced,
    v_cache_in: Traced,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos_idx: usize,
    eps: f32,
) -> (Traced, Traced, Traced) {
    assert!(
        n_heads.is_multiple_of(n_kv_heads),
        "n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
    );
    let qg_flat = linear(b, x, wq, None);
    let (q_raw, gate_raw) = qwen3next_split_query_gate(b, qg_flat, n_heads, head_dim);
    let k_flat = linear(b, x, wk, None);
    let v_flat = linear(b, x, wv, None);
    let (attn_flat, k_cache_out, v_cache_out) = qwen3next_gated_attention_projected(
        b, q_raw, gate_raw, k_flat, v_flat, q_norm_w, k_norm_w, cos, sin, pos, k_cache_in,
        v_cache_in, n_heads, n_kv_heads, head_dim, pos_idx, eps,
    );
    let out = linear(b, attn_flat, wo, None);
    (out, k_cache_out, v_cache_out)
}

/// Split an already projected interleaved Q/gate value without changing its source ownership.
pub(crate) fn qwen3next_split_query_gate(
    b: &Builder,
    qg_flat: Traced,
    n_heads: usize,
    head_dim: usize,
) -> (Traced, Traced) {
    let shape = b.aval(qg_flat).shape;
    let qg = b.reshape(qg_flat, vec![shape[0], shape[1], n_heads, 2 * head_dim]);
    let query = b.slice(qg, 3, 0, head_dim);
    let gate = b.slice(qg, 3, head_dim, 2 * head_dim);
    (query, gate)
}

/// Projection-independent gated-attention decode semantics shared with Qwen3.5 packed linears.
///
/// `q_raw`, `gate_raw`, `k_flat`, and `v_flat` are the already projected values. The returned first
/// value is the flattened gated attention result before the output projection. Keeping both
/// projection boundaries outside this core lets dense and packed callers share one attention
/// definition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen3next_gated_attention_projected(
    b: &Builder,
    q_raw: Traced,
    gate_raw: Traced,
    k_flat: Traced,
    v_flat: Traced,
    q_norm_w: Traced,
    k_norm_w: Traced,
    cos: Traced,
    sin: Traced,
    pos: Traced,
    k_cache_in: Traced,
    v_cache_in: Traced,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos_idx: usize,
    eps: f32,
) -> (Traced, Traced, Traced) {
    assert!(
        n_heads.is_multiple_of(n_kv_heads),
        "n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
    );
    let n_rep = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_dim = n_heads * head_dim;

    // 1. QK-norm (Qwen3-style: per-head RMSNorm over head_dim, after projection, before RoPE).
    let q_normed = rmsnorm(b, q_raw, q_norm_w, eps); // [1, 1, n_heads, head_dim]
    let k_shaped = b.reshape(k_flat, vec![1, 1, n_kv_heads, head_dim]); // [1, 1, n_kv_heads, hd]
    let k_normed = rmsnorm(b, k_shaped, k_norm_w, eps); // [1, 1, n_kv_heads, head_dim]
    let v_shaped = b.reshape(v_flat, vec![1, 1, n_kv_heads, head_dim]); // [1, 1, n_kv_heads, hd]

    // 2. Transpose to [batch=1, heads, seq=1, head_dim] for attention and RoPE.
    let q4 = b.transpose(q_normed, vec![0, 2, 1, 3]); // [1, n_heads, 1, head_dim]
    let k4 = b.transpose(k_normed, vec![0, 2, 1, 3]); // [1, n_kv_heads, 1, head_dim]
    let v4 = b.transpose(v_shaped, vec![0, 2, 1, 3]); // [1, n_kv_heads, 1, head_dim]

    // 3. Partial NeoX RoPE on Q and K (first rotary_dim dims; passthrough for the rest).
    let q4 = rope(b, q4, cos, sin, pos);
    let k4 = rope(b, k4, cos, sin, pos);

    // 4. KV cache update: scatter new k/v into position `pos_idx` (baked literal, axis 2 = seq).
    //    The graph is pos-specialized via the literal index;
    //    the runtime `pos` Traced value is used only for the rope table gather above.
    let k_cache_out = b.dynamic_update_slice(k_cache_in, k4, pos_idx, 2);
    let v_cache_out = b.dynamic_update_slice(v_cache_in, v4, pos_idx, 2);

    // 5. GQA: attend over the valid prefix [0 .. pos_idx+1]. n_rep query heads per KV head.
    let k_valid = b.slice(k_cache_out, 2, 0, pos_idx + 1); // [1, n_kv_heads, pos_idx+1, hd]
    let v_valid = b.slice(v_cache_out, 2, 0, pos_idx + 1);
    let attn = attention(b, q4, k_valid, v_valid, n_rep, scale); // [1, n_heads, 1, head_dim]

    // 6. Output gate: multiply attention output by sigmoid(gate) BEFORE o-projection.
    // gate_raw is [1, 1, n_heads, head_dim]; transpose to [1, n_heads, 1, head_dim] to match attn.
    let gate4 = b.transpose(gate_raw, vec![0, 2, 1, 3]); // [1, n_heads, 1, head_dim]
    let gate_sig = sigmoid(b, gate4); // [1, n_heads, 1, head_dim]
    let attn_gated = b.binary(BinOp::Mul, attn, gate_sig); // [1, n_heads, 1, head_dim]

    // 7. Flatten for the caller-owned output projection.
    let attn_back = b.transpose(attn_gated, vec![0, 2, 1, 3]); // [1, 1, n_heads, head_dim]
    let attn_flat = b.reshape(attn_back, vec![1, 1, q_dim]); // [1, 1, n_heads*head_dim]
    (attn_flat, k_cache_out, v_cache_out)
}

/// Qwen3-Next gated full-attention prefill block (card 158): the L>1 counterpart of
/// [`qwen3next_gated_attention`], for the first `L` positions of a prompt (fresh KV cache: the caller passes
/// zero-seeded `k_cache_in`/`v_cache_in`, like [`crate::qwen2::trace_prefill_kv`]).
///
/// Same weights, query|gate split, QK-norm placement and sigmoid output gate as the decode block. Only the ops
/// that must see the whole chunk change: [`rope`] -> [`rope_prefill`], the single-position write -> a batched
/// write of all `L` positions at cache slot 0 (`crate::qwen2::trace_prefill_kv_impl` pattern), and
/// [`attention`] -> the causal-masked [`attention_prefill`]. Since the cache holds exactly the just-computed
/// `k4`/`v4` at `0..L`, attention reads them directly.
///
/// ## Forward pass (all `L` positions at once)
///
/// ```text
/// Qg = x @ wq -> [1,L,n_heads*2*head_dim]; split per-head into q [1,L,n_heads,hd] and gate.
/// q_normed = rmsnorm(q, q_norm_w); k_normed = rmsnorm(k, k_norm_w)   # QK-norm, before RoPE
/// q4, k4, v4 -> [1, heads, L, head_dim] (transpose)
/// q4 = rope_prefill(q4, cos, sin, L); k4 = rope_prefill(k4, cos, sin, L)   # batched partial RoPE
/// k_cache_out = dynamic_update_slice(k_cache_in, k4, 0, axis=2)            # write all L at slot 0
/// v_cache_out = dynamic_update_slice(v_cache_in, v4, 0, axis=2)
/// attn = attention_prefill(q4, k4, v4, n_rep, scale, causal_mask)          # [1,n_heads,L,head_dim]
/// attn_gated = attn * sigmoid(gate4)                                       # output gate, before o-proj
/// out = flatten(attn_gated) @ wo                                          # [1,L,H]
/// ```
///
/// ## Arguments
///
/// Same weights as [`qwen3next_gated_attention`] (`wq`, `wk`, `wv`, `wo`, `q_norm_w`, `k_norm_w`,
/// `cos`, `sin`), plus:
///
/// - `x`: input after the pre-attention RMSNorm, shape `[1, L, H]` (L>1).
/// - `mask`: additive causal mask `[1, 1, L, L]` (0 on/below the diagonal, large-negative above),
///   same convention as [`poot_graph_ir::ops::attention_prefill`].
/// - `k_cache_in`, `v_cache_in`: carried KV caches `[1, n_kv_heads, cap, head_dim]` (zero-seeded for
///   a fresh prefill; `cap >= L`).
/// - `n_heads`, `n_kv_heads`, `head_dim`, `eps`: same meaning as [`qwen3next_gated_attention`]
///   (no `pos`/`pos_idx`: prefill always starts at position 0 and fills `cap`-capacity slots `0..L`).
///
/// Returns `(out [1, L, H], k_cache_out, v_cache_out)` - wire both cache pairs (like
/// [`qwen3next_gated_attention`]'s) so decode resumes reading the filled prefix right after prefill.
#[allow(clippy::too_many_arguments)]
pub fn qwen3next_gated_attention_prefill(
    b: &Builder,
    x: Traced,
    wq: Traced,
    wk: Traced,
    wv: Traced,
    wo: Traced,
    q_norm_w: Traced,
    k_norm_w: Traced,
    cos: Traced,
    sin: Traced,
    mask: Traced,
    k_cache_in: Traced,
    v_cache_in: Traced,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> (Traced, Traced, Traced) {
    assert!(
        n_heads.is_multiple_of(n_kv_heads),
        "n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
    );
    let qg_flat = linear(b, x, wq, None);
    let (q_raw, gate_raw) = qwen3next_split_query_gate(b, qg_flat, n_heads, head_dim);
    let k_flat = linear(b, x, wk, None);
    let v_flat = linear(b, x, wv, None);
    let (attn_flat, k_cache_out, v_cache_out) = qwen3next_gated_attention_prefill_projected(
        b, q_raw, gate_raw, k_flat, v_flat, q_norm_w, k_norm_w, cos, sin, mask, k_cache_in,
        v_cache_in, n_heads, n_kv_heads, head_dim, eps,
    );
    let out = linear(b, attn_flat, wo, None);
    (out, k_cache_out, v_cache_out)
}

/// Projection-independent gated-attention prefill semantics shared with Qwen3.5 packed linears.
///
/// The projected Q/gate, K, and V values enter in checkpoint projection order. The first return
/// value is the flattened gated attention result that the caller sends through its output linear.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen3next_gated_attention_prefill_projected(
    b: &Builder,
    q_raw: Traced,
    gate_raw: Traced,
    k_flat: Traced,
    v_flat: Traced,
    q_norm_w: Traced,
    k_norm_w: Traced,
    cos: Traced,
    sin: Traced,
    mask: Traced,
    k_cache_in: Traced,
    v_cache_in: Traced,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> (Traced, Traced, Traced) {
    assert!(
        n_heads.is_multiple_of(n_kv_heads),
        "n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
    );
    let n_rep = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_dim = n_heads * head_dim;
    let l = b.aval(q_raw).shape[1];

    // 1. QK-norm (per-head RMSNorm over head_dim, after projection, before RoPE).
    let q_normed = rmsnorm(b, q_raw, q_norm_w, eps); // [1, L, n_heads, head_dim]
    let k_shaped = b.reshape(k_flat, vec![1, l, n_kv_heads, head_dim]); // [1, L, n_kv_heads, hd]
    let k_normed = rmsnorm(b, k_shaped, k_norm_w, eps); // [1, L, n_kv_heads, head_dim]
    let v_shaped = b.reshape(v_flat, vec![1, l, n_kv_heads, head_dim]); // [1, L, n_kv_heads, hd]

    // 2. Transpose to [batch=1, heads, seq=L, head_dim] for attention and RoPE.
    let q4 = b.transpose(q_normed, vec![0, 2, 1, 3]); // [1, n_heads, L, head_dim]
    let k4 = b.transpose(k_normed, vec![0, 2, 1, 3]); // [1, n_kv_heads, L, head_dim]
    let v4 = b.transpose(v_shaped, vec![0, 2, 1, 3]); // [1, n_kv_heads, L, head_dim]

    // 3. Batched partial NeoX RoPE on Q and K, all L positions (0..L) at once.
    let q4 = rope_prefill(b, q4, cos, sin, l);
    let k4 = rope_prefill(b, k4, cos, sin, l);

    // 4. KV cache fill: write all L positions at slot 0 (axis 2 = seq), mirroring the qwen2
    //    trace_prefill_kv_impl contiguous-prefill pattern (a fresh cache, unlike decode's single-pos
    //    dynamic_update_slice at pos_idx).
    let k_cache_out = b.dynamic_update_slice(k_cache_in, k4, 0, 2);
    let v_cache_out = b.dynamic_update_slice(v_cache_in, v4, 0, 2);

    // 5. GQA causal attention over all L positions. k4/v4 are exactly what was just written into the
    //    cache at 0..L, so attending on them directly is identical to reading the filled cache prefix.
    let attn = attention_prefill(b, q4, k4, v4, n_rep, scale, mask); // [1, n_heads, L, head_dim]

    // 6. Output gate: multiply attention output by sigmoid(gate) BEFORE o-projection.
    let gate4 = b.transpose(gate_raw, vec![0, 2, 1, 3]); // [1, n_heads, L, head_dim]
    let gate_sig = sigmoid(b, gate4); // [1, n_heads, L, head_dim]
    let attn_gated = b.binary(BinOp::Mul, attn, gate_sig); // [1, n_heads, L, head_dim]

    // 7. Flatten for the caller-owned output projection.
    let attn_back = b.transpose(attn_gated, vec![0, 2, 1, 3]); // [1, L, n_heads, head_dim]
    let attn_flat = b.reshape(attn_back, vec![1, l, q_dim]); // [1, L, n_heads*head_dim]
    (attn_flat, k_cache_out, v_cache_out)
}

/// L2 normalize over the last axis: `x / sqrt(sum(x^2) + eps)`.
///
/// Differs from RMSNorm: no learned weight, and eps is added to the SUM of squares (not the mean).
/// Per `ggml_l2_norm` in the llama.cpp reference (section 3.3).
pub(crate) fn l2_norm_last(b: &Builder, x: Traced, eps: f32) -> Traced {
    let ndim = b.aval(x).shape.len();
    let sq = b.binary(BinOp::Mul, x, x); // x^2
    let ss = b.reduce(RedOp::Sum, sq, ndim - 1, true); // sum(x^2) over last axis, keepdim
    let ss_eps = b.binary_scalar(BinOp::Add, ss, Scalar::F32(eps)); // sum(x^2) + eps
    let norm = b.unary(UnOp::Sqrt, ss_eps); // sqrt(sum(x^2) + eps)
    b.binary(BinOp::Div, x, norm) // x / norm
}
