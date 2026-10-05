use super::*;

/// Softplus: `ln(1 + exp(x))`, the positive activation Mamba applies to its timestep
/// (`Delta = softplus(dt_proj + dt_bias)`). Overflows in `exp` for very large `x`; Mamba's
/// pre-activation is small.
pub fn softplus(b: &Builder, x: Traced) -> Traced {
    let e = b.unary(UnOp::Exp, x);
    let one_plus = b.binary_scalar(BinOp::Add, e, Scalar::F32(1.0));
    b.unary(UnOp::Log, one_plus)
}

/// ReLU: `max(x, 0)`, as `BinOp::Max` against the scalar literal `0.0` (one op instead of
/// `x * ge(x,0)`). Used by DeepSeek-V3.2's Lightning Indexer score (`crate::deepseek32`, arXiv
/// 2512.02556 eq. 1: `I_t,s = sum_j w_j * ReLU(q_j . k_s)`). Pure primitive composition.
pub fn relu(b: &Builder, x: Traced) -> Traced {
    b.binary_scalar(BinOp::Max, x, Scalar::F32(0.0))
}

/// Sigmoid: `1 / (1 + exp(-x))` - elementwise logistic function. Used as a gate (Qwen3-Next shared-expert
/// gate, gated attention output) and as a smooth non-linearity. Pure primitive composition (neg, exp, add,
/// recip); no new IR op needed.
pub fn sigmoid(b: &Builder, x: Traced) -> Traced {
    let neg_x = b.unary(UnOp::Neg, x);
    let e = b.unary(UnOp::Exp, neg_x);
    let one_plus = b.binary_scalar(BinOp::Add, e, Scalar::F32(1.0));
    b.unary(UnOp::Recip, one_plus)
}

/// Hyperbolic tangent: the [`UnOp::Tanh`] primitive.
pub fn tanh(b: &Builder, x: Traced) -> Traced {
    b.unary(UnOp::Tanh, x)
}

/// Logit softcapping (Gemma2 / Grok): `softcap(x, c) = c * tanh(x / c)`, a smooth clamp of each logit
/// into `(-c, c)`. Applied to the attention scores (before softmax) and the final output logits
/// (before sampling). Composition over [`tanh`]; no new IR primitive. As `c -> inf` it approaches the
/// identity, so a large cap is a no-op on modest logits.
pub fn softcap(b: &Builder, x: Traced, cap: f32) -> Traced {
    let scaled = b.binary_scalar(BinOp::Mul, x, Scalar::F32(1.0 / cap)); // x / c
    let t = tanh(b, scaled);
    b.binary_scalar(BinOp::Mul, t, Scalar::F32(cap)) // c * tanh(x / c)
}

/// Build a data-dependent decay-causal mask for gated linear attention from per-step gates.
///
/// `gates` is `[1, H, 1, L]`, the positive per-step decay factors `g[h,k]` (typically `exp(-something)`,
/// in `(0, 1]`). `tril` is `[1, H, L, L]`, lower-triangular ones (including the diagonal). Returns the
/// multiplicative decay mask `[1, H, L, L]` with `mask[h,t,j] = prod_{j < k <= t} g[h,k]` for `j <= t`,
/// and `0` above the diagonal: exactly the `mask` [`linear_attention_prefill`] multiplies into `Q K^T`.
///
/// Computed in log space: `mask[h,t,j] = exp(cumlog[h,t] - cumlog[h,j])` where
/// `cumlog[h,t] = sum_{k <= t} log g[h,k]` is the prefix sum of the log gates, formed by the `tril`
/// matmul. This needs the `Log` primitive because real gated models (Mamba2 prefill, Gated DeltaNet)
/// have input-dependent gates that cannot be precomputed on the host. With a constant gate `g` it
/// reduces to `g^(t-j)` on and below the diagonal.
pub fn decay_mask_from_gates(
    b: &Builder,
    gates: Traced,
    tril: Traced,
    h: usize,
    l: usize,
) -> Traced {
    let lg = b.unary(UnOp::Log, gates); // [1,H,1,L] log gates
    let lg_col = b.reshape(lg, vec![1, h, l, 1]); // [1,H,L,1]
    let cumlog = b.matmul(tril, lg_col); // [1,H,L,L] @ [1,H,L,1] -> [1,H,L,1] = prefix sum of log gates
    let cum_row = b.reshape(cumlog, vec![1, h, 1, l]); // [1,H,1,L] = cumlog[j] (same L values, moved axis)
    let diff = b.binary(BinOp::Sub, cumlog, cum_row); // [1,H,L,L] diff[t,j] = cumlog[t] - cumlog[j]
    let prod = b.unary(UnOp::Exp, diff); // exp(diff) = prod_{j<k<=t} g[k] (and 1 on the diagonal)
    b.binary(BinOp::Mul, prod, tril) // zero the strict upper triangle (j > t)
}

/// SiLU / swish: `x / (1 + exp(-x))`. Written as the single division rather than `x * sigmoid(x)`
/// (which rounds twice); this is the form every device fuses and the oracle's historical bits.
pub fn silu(b: &Builder, x: Traced) -> Traced {
    let e = b.unary(UnOp::Exp, b.unary(UnOp::Neg, x));
    let den = b.binary_scalar(BinOp::Add, e, Scalar::F32(1.0));
    b.binary(BinOp::Div, x, den)
}

/// GELU, tanh approximation (`gelu_pytorch_tanh`): `0.5 * x * (1 + tanh(z))` with
/// `z = sqrt(2/pi) * (x + 0.044715 * x^3)`, written in its sigmoid form `x / (1 + exp(-2z))` (the same
/// value, one rounding fewer than going through [`tanh`]). The cube keeps the oracle's association,
/// `((0.044715 * x) * x) * x`.
pub fn gelu(b: &Builder, x: Traced) -> Traced {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    let cubic = b.binary(
        BinOp::Mul,
        b.binary(
            BinOp::Mul,
            b.binary_scalar(BinOp::Mul, x, Scalar::F32(0.044715)),
            x,
        ),
        x,
    );
    let z = b.binary_scalar(BinOp::Mul, b.binary(BinOp::Add, x, cubic), Scalar::F32(c));
    let e = b.unary(UnOp::Exp, b.binary_scalar(BinOp::Mul, z, Scalar::F32(-2.0)));
    let den = b.binary_scalar(BinOp::Add, e, Scalar::F32(1.0));
    b.binary(BinOp::Div, x, den)
}

/// GELU, exact erf form (HF `hidden_act: "gelu"`): `0.5 * x * (1 + erf(x / sqrt(2)))` over the
/// [`UnOp::Erf`] primitive.
pub fn gelu_erf(b: &Builder, x: Traced) -> Traced {
    let erf = b.unary(
        UnOp::Erf,
        b.binary_scalar(BinOp::Mul, x, Scalar::F32(std::f32::consts::FRAC_1_SQRT_2)),
    );
    let one_plus = b.binary_scalar(BinOp::Add, erf, Scalar::F32(1.0));
    b.binary(
        BinOp::Mul,
        b.binary_scalar(BinOp::Mul, x, Scalar::F32(0.5)),
        one_plus,
    )
}

/// SwiGLU activation: `silu(gate) * up` (the matmuls are the caller's; this is the pointwise core).
pub fn swiglu(b: &Builder, gate: Traced, up: Traced) -> Traced {
    let g = silu(b, gate);
    b.binary(BinOp::Mul, g, up)
}

/// GeGLU activation: `gelu(gate) * up` (Gemma's MLP core; the gelu-tanh approximation). Same shape as
/// [`swiglu`] with gelu in place of silu.
pub fn geglu(b: &Builder, gate: Traced, up: Traced) -> Traced {
    let g = gelu(b, gate);
    b.binary(BinOp::Mul, g, up)
}
