use super::*;

/// RMSNorm over the last axis: x * rsqrt(mean(x^2) + eps) * w. (Plain RMSNorm, no `(1+w)` Gemma offset.)
pub fn rmsnorm(b: &Builder, x: Traced, w: Traced, eps: f32) -> Traced {
    let xn = rmsnorm_core(b, x, eps);
    b.binary(BinOp::Mul, xn, w)
}

/// RMSNorm over the last axis with no learned scale: `x * rsqrt(mean(x^2) + eps)`. Gemma4's V-norm
/// (section 4) normalizes V on every layer
/// with no weight tensor, matching llama.cpp's bare `ggml_rms_norm`. Shares [`rmsnorm_core`] with
/// [`rmsnorm`].
pub fn rmsnorm_no_weight(b: &Builder, x: Traced, eps: f32) -> Traced {
    rmsnorm_core(b, x, eps)
}

/// The shared RMSNorm core (normalize, no scale): `x * rsqrt(mean(x^2) + eps)`.
pub(crate) fn rmsnorm_core(b: &Builder, x: Traced, eps: f32) -> Traced {
    let shape = b.aval(x).shape;
    let axis = shape.len() - 1;
    let n = shape[axis] as f32;
    let sq = b.binary(BinOp::Mul, x, x);
    let ss = b.reduce(RedOp::Sum, sq, axis, true);
    let ms = b.binary_scalar(BinOp::Mul, ss, Scalar::F32(1.0 / n));
    let mse = b.binary_scalar(BinOp::Add, ms, Scalar::F32(eps));
    let den = b.unary(UnOp::Sqrt, mse);
    b.binary(BinOp::Div, x, den)
}

/// LayerNorm over the last axis (SigLIP vision encoder): `(x - mean) / sqrt(var + eps) * w + b`,
/// `var = mean((x - mean)^2)`. `w`/`b` are `[H]` vectors broadcast over the leading dims.
pub fn layernorm(b: &Builder, x: Traced, w: Traced, bias: Traced, eps: f32) -> Traced {
    let scaled = layernorm_core(b, x, w, eps);
    b.binary(BinOp::Add, scaled, bias)
}

/// LayerNorm over the last axis with no learned bias: `(x - mean) / sqrt(var + eps) * w`. MPT uses
/// this profile (`MPTConfig.no_bias` defaults to `true`; `modeling_mpt.py` sets `norm_1.bias = None`),
/// unlike BLOOM's biased [`layernorm`]. Shares [`layernorm_core`] with it.
pub fn layernorm_no_bias(b: &Builder, x: Traced, w: Traced, eps: f32) -> Traced {
    layernorm_core(b, x, w, eps)
}

/// The shared LayerNorm core (normalize, no bias): `(x - mean) / sqrt(var + eps) * w`.
pub(crate) fn layernorm_core(b: &Builder, x: Traced, w: Traced, eps: f32) -> Traced {
    let shape = b.aval(x).shape;
    let axis = shape.len() - 1;
    let n = shape[axis] as f32;
    let sum = b.reduce(RedOp::Sum, x, axis, true);
    let mean = b.binary_scalar(BinOp::Mul, sum, Scalar::F32(1.0 / n)); // [.., 1]
    let centered = b.binary(BinOp::Sub, x, mean);
    let sq = b.binary(BinOp::Mul, centered, centered);
    let var_sum = b.reduce(RedOp::Sum, sq, axis, true);
    let var = b.binary_scalar(BinOp::Mul, var_sum, Scalar::F32(1.0 / n));
    let var_eps = b.binary_scalar(BinOp::Add, var, Scalar::F32(eps));
    let den = b.unary(UnOp::Sqrt, var_eps);
    let normed = b.binary(BinOp::Div, centered, den);
    b.binary(BinOp::Mul, normed, w)
}
