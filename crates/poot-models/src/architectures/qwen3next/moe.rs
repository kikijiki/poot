use super::*;

/// Qwen3-Next MoE FFN + shared-expert block.
///
/// `x` has shape `[1, L, H]`. All weight tensors are in poot convention (pre-transposed `[in, out]`).
/// Returns `[1, L, H]`.
///
/// Arguments:
/// - `x`: input after the pre-FFN RMSNorm, shape `[1, L, H]`.
/// - `router_w`: router weight `[H, E]`.
/// - `w_in`: fused per-expert gate||up weight `[E, H, 2*I]` (gate rows first, up rows second along the I dim).
/// - `w_out`: per-expert down weight `[E, I, H]`.
/// - `shexp_gate_w`: shared expert gate proj `[H, I]`.
/// - `shexp_up_w`: shared expert up proj `[H, I]`.
/// - `shexp_down_w`: shared expert down proj `[I, H]`.
/// - `shexp_gin_w`: shared expert scalar gate `[H, 1]` (produces one scalar per token).
/// - `n_experts`: total number of routed experts (256 in the real model).
/// - `top_k`: experts selected per token (8 in the real model).
/// - `inter`: per-expert intermediate dimension (512 in the real model).
#[allow(clippy::too_many_arguments)]
pub fn qwen3next_moe_ffn(
    b: &Builder,
    x: Traced,
    router_w: Traced,
    w_in: Traced,
    w_out: Traced,
    shexp_gate_w: Traced,
    shexp_up_w: Traced,
    shexp_down_w: Traced,
    shexp_gin_w: Traced,
    n_experts: usize,
    top_k: usize,
    inter: usize,
) -> Traced {
    let shape = b.aval(x).shape;
    let l = shape[shape.len() - 2];
    let h = shape[shape.len() - 1];

    // 1. Routed MoE: softmax over all n_experts, top-k, renormalize, per-expert SwiGLU, weighted sum.
    //    moe() dispatches to moe_sparse (L=1) or moe_dense (L>1) automatically.
    let moe_out = moe(b, x, router_w, w_in, w_out, n_experts, top_k, inter); // [1, L, H]

    // 2. Shared expert SwiGLU. Work on [L, H] for the matmuls.
    let xm = b.reshape(x, vec![l, h]); // [L, H]
    let sg = linear(b, xm, shexp_gate_w, None); // [L, I]
    let su = linear(b, xm, shexp_up_w, None); // [L, I]
    let act = swiglu(b, sg, su); // [L, I]  -- silu(gate) * up
    let shexp = linear(b, act, shexp_down_w, None); // [L, H]
    let shexp = b.reshape(shexp, vec![1, l, h]); // [1, L, H]

    // 3. Scalar sigmoid gate on the shared expert.
    //    shexp_gin_w is [H, 1]; linear produces [L, 1], sigmoid -> still [L, 1].
    let g_raw = linear(b, xm, shexp_gin_w, None); // [L, 1]
    let g = sigmoid(b, g_raw); // [L, 1]  -- one scalar per token
    let g = b.reshape(g, vec![1, l, 1]); // [1, L, 1] -- broadcast over H

    // 4. Combine: moe_out + sigmoid(g) * shexp.
    let gated = b.binary(BinOp::Mul, g, shexp); // [1, L, H]  -- broadcast [1,L,1] * [1,L,H]
    b.binary(BinOp::Add, moe_out, gated) // [1, L, H]
}
