use super::*;
use poot_graph_ir::ops::silu;

/// Widen a single-stream `[.., H]` tensor to the `[.., hc_count*H]` Hyper-Connections stream by tiling the
/// same `H`-sized vector `hc_count` times along the feature axis, as the real
/// `hidden_states.repeat(1, 1, hc_count)` (`Qwen4ExpTextModel.forward`, `modeling_qwen4_exp.py` line 1491).
/// Composed from existing primitives: insert a size-1 axis before the last axis (`Reshape`), expand it to
/// `hc_count` (`Broadcast`), then flatten the last two axes (`Reshape`).
pub(crate) fn qwen38_widen_embedding(b: &Builder, x: Traced, hc_count: usize) -> Traced {
    let shape = b.aval(x).shape; // [.., H]
    let h = *shape.last().expect("embedding must have at least one axis");
    let insert_at = shape.len() - 1;
    let mut unsq = shape.clone();
    unsq.insert(insert_at, 1); // [.., 1, H]
    let mut wide = unsq.clone();
    wide[insert_at] = hc_count; // [.., hc_count, H]
    let mut flat = shape.clone();
    let last = flat.len() - 1;
    flat[last] = hc_count * h; // [.., hc_count*H]

    let x1 = b.reshape(x, unsq);
    let x2 = b.broadcast(x1, wide);
    b.reshape(x2, flat)
}

/// Grouped RMSNorm over a widened `[.., hc_count*H]` Hyper-Connections stream: the real
/// `Qwen4ExpTextRMSNorm(dim=hc_count*H, group_size=H)` (also used ungrouped for `q_norm`/`k_norm`/the
/// indexer norms; see [`trace_qwen38_prefill`] on the shared `+1.0` loader convention). Reshape to
/// `[.., hc_count, H]`, normalize each `H`-sized group by its own mean-square, reshape back, then
/// multiply by the full-length `[hc_count*H]` weight (the weight has no group structure; see
/// `modeling_qwen4_exp.py` lines 173-184). Plain multiply, as [`poot_graph_ir::ops::rmsnorm`]: the `+1.0`
/// offset is the loader's job, as for `q_norm`/`k_norm`.
pub(crate) fn qwen38_grouped_rmsnorm(
    b: &Builder,
    x: Traced,
    w: Traced,
    hc_count: usize,
    h: usize,
    eps: f32,
) -> Traced {
    let shape = b.aval(x).shape; // [.., hc_count*H]
    let mut grouped = shape.clone();
    let last = grouped.len() - 1;
    grouped[last] = hc_count;
    grouped.push(h); // [.., hc_count, H]
    let gaxis = grouped.len() - 1;

    let xg = b.reshape(x, grouped);
    let sq = b.binary(BinOp::Mul, xg, xg);
    let ss = b.reduce(RedOp::Sum, sq, gaxis, true);
    let ms = b.binary_scalar(BinOp::Mul, ss, Scalar::F32(1.0 / h as f32));
    let mse = b.binary_scalar(BinOp::Add, ms, Scalar::F32(eps));
    let den = b.unary(UnOp::Sqrt, mse);
    let normed_g = b.binary(BinOp::Div, xg, den);
    let normed = b.reshape(normed_g, shape);
    b.binary(BinOp::Mul, normed, w)
}

/// Shared core of [`qwen38_gated_residual`] (`use_combine=True`) and [`qwen38_gated_residual_mixer`]
/// (`use_combine=False`): the `Qwen4ExpTextGatedResidual.forward` prefix "grouped-norm, low-rank gate,
/// weighted mean-pool over the `hc_count` axis" (`modeling_qwen4_exp.py` lines 1033-1039):
///
/// ```text
/// hyper_input_normed = hc_norm(hyper_input)                                     // grouped RMSNorm
/// gate = sigmoid(up(silu(down(hyper_input_normed) / hc_count)))                 // low-rank bottleneck
/// mixed_input = mean_over_hc_count(gate.reshape(..,hc_count,H) * hyper_input_normed.reshape(..,hc_count,H))
/// ```
///
/// Returns `(hyper_input_normed [.., hc_count*H], mixed_input [.., H])`; [`qwen38_gated_residual`] needs
/// the normed value again for its `block_inject_weight` projection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen38_gated_residual_core(
    b: &Builder,
    hyper_input: Traced,
    p: &str,
    hc_count: usize,
    h: usize,
    lowrank: usize,
    eps: f32,
) -> (Traced, Traced) {
    let hc_h = hc_count * h;
    let norm_w = b.constant(&format!("{p}.hc_norm.weight"), TensorType::f32(vec![hc_h]));
    let normed = qwen38_grouped_rmsnorm(b, hyper_input, norm_w, hc_count, h, eps);

    let w_down = b.constant(
        &format!("{p}.input_mix_weight_down.weight"),
        TensorType::f32(vec![hc_h, lowrank]),
    );
    let w_up = b.constant(
        &format!("{p}.input_mix_weight_up.weight"),
        TensorType::f32(vec![lowrank, hc_h]),
    );
    let down = linear(b, normed, w_down, None); // [.., lowrank]
    let down_scaled = b.binary_scalar(BinOp::Mul, down, Scalar::F32(1.0 / hc_count as f32));
    let down_act = silu(b, down_scaled);
    let up = linear(b, down_act, w_up, None); // [.., hc_h]
    let gate = sigmoid(b, up);

    let shape = b.aval(hyper_input).shape;
    let mut grouped = shape.clone();
    let last = grouped.len() - 1;
    grouped[last] = hc_count;
    grouped.push(h); // [.., hc_count, H]
    let gaxis = grouped.len() - 2; // the hc_count axis (H stays last)

    let gate_g = b.reshape(gate, grouped.clone());
    let normed_g = b.reshape(normed, grouped);
    let weighted = b.binary(BinOp::Mul, gate_g, normed_g);
    let summed = b.reduce(RedOp::Sum, weighted, gaxis, false); // [.., H]
    let mixed_input = b.binary_scalar(BinOp::Mul, summed, Scalar::F32(1.0 / hc_count as f32));

    (normed, mixed_input)
}

/// `Qwen4ExpTextGatedResidual.forward` with `use_combine=True` (the per-layer `attn_hyper_connection`/
/// `mlp_hyper_connection`, tensor prefix `{p}.{attn,mlp}_hyper_connection`, `modeling_qwen4_exp.py` lines
/// 1015-1043). `hyper_input` is the caller's widened `[.., hc_count*H]` stream. Returns
/// `(mixed_input [.., H], injection_weights [.., hc_count])`: the caller runs its sub-layer
/// (attention/GDN/MLP) on `mixed_input`, then combines the `[.., H]` output with `injection_weights` via
/// [`qwen38_gated_residual_inject`].
///
/// The residual add uses the raw pre-norm stream, not the normed one. `Qwen4ExpTextGatedResidual.forward`
/// returns `mixed_input, hyper_input, injection_weights` (line 1043), where `hyper_input` is its own raw
/// parameter, not the local `hyper_input_normed` from line 1033, and `hidden_states = hyper_input +
/// injection.flatten(-2)` (lines 1311/1317) adds it back. So this function returns no normed value, and
/// the caller combines [`qwen38_gated_residual_inject`]'s result with the same `hyper_input` it already
/// holds.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen38_gated_residual(
    b: &Builder,
    hyper_input: Traced,
    p: &str,
    hc_count: usize,
    h: usize,
    lowrank: usize,
    eps: f32,
) -> (Traced, Traced) {
    let hc_h = hc_count * h;
    let (normed, mixed_input) =
        qwen38_gated_residual_core(b, hyper_input, p, hc_count, h, lowrank, eps);

    let w_inject = b.constant(
        &format!("{p}.block_inject_weight.weight"),
        TensorType::f32(vec![hc_h, hc_count]),
    );
    let inject_logit = linear(b, normed, w_inject, None); // [.., hc_count]
    let inject_scaled =
        b.binary_scalar(BinOp::Mul, inject_logit, Scalar::F32(1.0 / hc_count as f32));
    let inject_sig = sigmoid(b, inject_scaled);
    let injection_weights = b.binary_scalar(BinOp::Mul, inject_sig, Scalar::F32(2.0));

    (mixed_input, injection_weights)
}

/// Combine a sub-layer's `[.., H]` output with [`qwen38_gated_residual`]'s `injection_weights
/// [.., hc_count]` into an additive update to the original widened stream `hyper_input` (the value passed
/// to that call; see its doc comment): real source lines 1310-1311/1316-1317,
/// `injection = sublayer_out.unsqueeze(-2) * injection_weights.unsqueeze(-1)` then
/// `hidden_states = hyper_input + injection.flatten(-2)`. The `unsqueeze`/`flatten` pair is a `Reshape`
/// on each side of a broadcasting `Binary::Mul`.
pub(crate) fn qwen38_gated_residual_inject(
    b: &Builder,
    hyper_input: Traced,
    sublayer_out: Traced,
    injection_weights: Traced,
) -> Traced {
    let out_shape = b.aval(sublayer_out).shape; // [.., H]
    let mut sub_row = out_shape.clone();
    let n = sub_row.len();
    sub_row.insert(n - 1, 1); // [.., 1, H]
    let sub_r = b.reshape(sublayer_out, sub_row);

    let w_shape = b.aval(injection_weights).shape; // [.., hc_count]
    let mut w_col = w_shape.clone();
    w_col.push(1); // [.., hc_count, 1]
    let w_c = b.reshape(injection_weights, w_col);

    let injection = b.binary(BinOp::Mul, sub_r, w_c); // [.., hc_count, H], broadcast
    let hyper_shape = b.aval(hyper_input).shape;
    let injection_flat = b.reshape(injection, hyper_shape);
    b.binary(BinOp::Add, hyper_input, injection_flat)
}

/// `Qwen4ExpTextGatedResidual.forward` with `use_combine=False` (the model-level
/// `hyper_connection_mixer`, tensor prefix `hyper_connection_mixer`, no `block_inject_weight`;
/// `modeling_qwen4_exp.py` lines 1024/1040-1041). Collapses the final widened `[.., hc_count*H]` stream to
/// one `[.., H]` stream, applied once after the last decoder layer, before `lm_head`
/// (`Qwen4ExpTextModel.forward`, lines 1404/1504). Shares [`qwen38_gated_residual_core`] with
/// [`qwen38_gated_residual`], with no injection step.
pub(crate) fn qwen38_gated_residual_mixer(
    b: &Builder,
    hyper_input: Traced,
    p: &str,
    hc_count: usize,
    h: usize,
    lowrank: usize,
    eps: f32,
) -> Traced {
    let (_normed, mixed_input) =
        qwen38_gated_residual_core(b, hyper_input, p, hc_count, h, lowrank, eps);
    mixed_input
}
