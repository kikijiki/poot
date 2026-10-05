//! Cast, i8 pack/quant round trips, layernorm/rmsnorm, fused pointwise/reduction fuzz, swiglu, rope.

use crate::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::Slot;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::{rmsnorm, rope, swiglu};
use poot_graph_ir::types::TensorType;
use poot_tensor::DType;
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::{assert_close_rel, max_abs_error_f64};

/// Card 045: `cast(x, BF16)` infers dtype BF16 (same shape) and rounds each value to bf16 precision
/// (round-to-nearest-even); `cast(x, F32)` is the identity. Executors' cast kernels must match.
#[test]
fn cast_f32_to_bf16_rounds_and_infers_dtype() {
    // reference round-to-nearest-even to bf16 (top 16 bits), widened back to f32.
    let ref_bf16 = |f: f32| -> f32 {
        if f.is_nan() {
            return f;
        }
        let bits = f.to_bits();
        let bias = 0x7FFF + ((bits >> 16) & 1);
        f32::from_bits((bits.wrapping_add(bias)) & 0xFFFF_0000)
    };
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![6]));
    let y = b.cast(x, DType::BF16);
    assert_eq!(b.aval(y).dtype, DType::BF16, "cast retypes to BF16");
    assert_eq!(b.aval(y).shape, vec![6], "cast keeps the shape");
    let g = b.finish(y);
    let vals = vec![1.0f32, 0.1, 3.13159, -2.73828, 12345.678, 1e-7];
    let mut inputs = HashMap::new();
    inputs.insert(x.id, Value::from(HostTensor::f32(vec![6], vals.clone())));
    // `Cast(F32, BF16)` publishes a BF16 tensor of words.
    let out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    assert_eq!(out.dtype(), DType::BF16, "cast publishes BF16 words");
    let words = out.as_half().expect("BF16 words");
    let widened = out.to_f32().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(
            widened[i].to_bits(),
            ref_bf16(v).to_bits(),
            "cast->bf16 [{i}] v={v}: got {} want {}",
            widened[i],
            ref_bf16(v)
        );
        assert_eq!(
            u32::from(words[i]),
            ref_bf16(v).to_bits() >> 16,
            "bf16 word [{i}] must be the top 16 bits of the rounded value"
        );
    }

    // cast to F32 (widening) is the identity.
    let b2 = Builder::new();
    let x2 = b2.constant("x", TensorType::new(vec![3], DType::BF16));
    let y2 = b2.cast(x2, DType::F32);
    assert_eq!(b2.aval(y2).dtype, DType::F32);
    let g2 = b2.finish(y2);
    let mut in2 = HashMap::new();
    in2.insert(
        x2.id,
        Value::from(
            crate::ops::cast::narrow_f32(vec![3], &[0.5, 0.25, -0.75], DType::BF16).unwrap(),
        ),
    );
    let o2 = eval(&g2, &in2, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    assert_eq!(o2.as_f32().unwrap(), &[0.5f32, 0.25, -0.75]);
}

/// Spec 135: `cast(x, F16)` infers dtype F16 (same shape) and rounds to f16 precision via `poot_load`'s
/// f32<->f16 encode/decode; `cast(x, F32)` is the identity. The f16 analog of
/// `cast_f32_to_bf16_rounds_and_infers_dtype`.
#[test]
fn cast_f32_to_f16_rounds_and_infers_dtype() {
    let ref_f16 = |f: f32| poot_quant::scalar::f16_to_f32(poot_load::gguf::f32_to_f16(f));
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![6]));
    let y = b.cast(x, DType::F16);
    assert_eq!(b.aval(y).dtype, DType::F16, "cast retypes to F16");
    assert_eq!(b.aval(y).shape, vec![6], "cast keeps the shape");
    let g = b.finish(y);
    let vals = vec![1.0f32, 0.1, 3.13159, -2.73828, 1234.5, 1e-5];
    let mut inputs = HashMap::new();
    inputs.insert(x.id, Value::from(HostTensor::f32(vec![6], vals.clone())));
    // `Cast(F32, F16)` publishes an F16 tensor of words.
    let out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    assert_eq!(out.dtype(), DType::F16, "cast publishes F16 words");
    let widened = out.to_f32().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(
            widened[i].to_bits(),
            ref_f16(v).to_bits(),
            "cast->f16 [{i}] v={v}: got {} want {}",
            widened[i],
            ref_f16(v)
        );
    }

    // cast to F32 (widening) is the identity.
    let b2 = Builder::new();
    let x2 = b2.constant("x", TensorType::new(vec![3], DType::F16));
    let y2 = b2.cast(x2, DType::F32);
    assert_eq!(b2.aval(y2).dtype, DType::F32);
    let g2 = b2.finish(y2);
    let mut in2 = HashMap::new();
    in2.insert(
        x2.id,
        Value::from(HostTensor::f16(
            vec![3],
            [0.5f32, 0.25, -0.75]
                .map(poot_load::gguf::f32_to_f16)
                .to_vec(),
        )),
    );
    let o2 = eval(&g2, &in2, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    assert_eq!(o2.as_f32().unwrap(), &[0.5f32, 0.25, -0.75]);
}

/// Spec 135: an F16 `MatMul` narrows its f32 accumulator to f16 in the oracle, mirroring the BF16 arm
/// and GPU kernels that store f16 on write; otherwise results diverge by up to 1 f16 ULP.
#[test]
fn matmul_f16_output_narrows_the_accumulator() {
    let b = Builder::new();
    let a = b.constant("a", TensorType::new(vec![1, 2, 3], DType::F16));
    let w = b.constant("w", TensorType::new(vec![3, 2], DType::F16));
    let mm = b.matmul(a, w);
    assert_eq!(b.aval(mm).dtype, DType::F16, "matmul keeps the F16 dtype");
    let g = b.finish(mm);
    let mut inputs = HashMap::new();
    inputs.insert(
        a.id,
        Value::from(HostTensor::f16(
            vec![1, 2, 3],
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .map(poot_load::gguf::f32_to_f16)
                .to_vec(),
        )),
    );
    inputs.insert(
        w.id,
        Value::from(HostTensor::f16(
            vec![3, 2],
            [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6]
                .map(poot_load::gguf::f32_to_f16)
                .to_vec(),
        )),
    );
    // `matmul` narrows its F16-dtype output to f16 words; widen them to compare.
    let out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    // row 0: [1,2,3] . cols -> [1*0.1+2*0.3+3*0.5, 1*0.2+2*0.4+3*0.6] = [2.2, 2.8]
    // row 1: [4,5,6] . cols -> [4*0.1+5*0.3+6*0.5, 4*0.2+5*0.4+6*0.6] = [4.9, 6.4]
    let want_f32 = [2.2f32, 2.8, 4.9, 6.4];
    for (i, (&got, &w)) in out
        .to_f32()
        .unwrap()
        .iter()
        .zip(want_f32.iter())
        .enumerate()
    {
        let want_f16 = poot_quant::scalar::f16_to_f32(poot_load::gguf::f32_to_f16(w));
        assert_eq!(
            got.to_bits(),
            want_f16.to_bits(),
            "matmul[{i}]: got {got} want f16-rounded {want_f16} (f32 {w})"
        );
    }
}

#[test]
fn pack_unpack_i8_graph_round_trips() {
    // spec 048: PackI8 -> UnpackI8 round-trips int8 codes losslessly; the packed tensor is I32 with last
    // dim ceil(cols/4) and carries the exact the I32 words payload.

    let rows = 3usize;
    let cols = 10usize; // not a multiple of 4 -> the tail word zero-pads.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, cols]));
    let packed = b.pack_i8(x);
    let unpacked = b.unpack_i8(packed, cols);
    // the packed value is I32 with last dim ceil(10/4) = 3.
    assert_eq!(b.aval(packed).dtype, DType::I32);
    assert_eq!(b.aval(packed).shape, vec![rows, cols.div_ceil(4)]);
    assert_eq!(b.aval(unpacked).dtype, DType::F32);
    let xi = x.id;

    // codes in [-127, 127], including the extremes and negatives.
    let codes: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 37) % 255) as f32 - 127.0)
        .collect();

    // check the packed tensor itself is an int tensor (eval the pack node alone).
    let gp = b.finish(packed);
    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        Value::from(HostTensor::f32(vec![rows, cols], codes.clone())),
    );
    let packed_t = eval(&gp, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    assert!(packed_t.as_i32().is_some(), "PackI8 output must carry ints");
    assert_eq!(packed_t.shape(), vec![rows, cols.div_ceil(4)]);

    // round-trip via a fresh graph (the builder consumed `packed` into gp).
    let b2 = Builder::new();
    let x2 = b2.constant("x", TensorType::f32(vec![rows, cols]));
    let rt = b2.unpack_i8(b2.pack_i8(x2), cols);
    let x2i = x2.id;
    let g = b2.finish(rt);
    let mut in2 = HashMap::new();
    in2.insert(
        x2i,
        Value::from(HostTensor::f32(vec![rows, cols], codes.clone())),
    );
    let got = eval(&g, &in2, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    assert_eq!(
        got.as_f32().unwrap(),
        &codes[..],
        "pack/unpack must round-trip exactly"
    );
}

#[test]
fn packed_i8_survives_copy_ops_in_eval() {
    // spec 048: the eval oracle must carry the exact i32 payload through copy ops (gather, transpose),
    // since the f32 mirror is lossy for arbitrary 32-bit words and UnpackI8 needs an int tensor.
    let (rows, cols) = (4usize, 8usize);
    let codes: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 53) % 255) as f32 - 127.0)
        .collect();

    // gather: pack -> gather(axis 0, reverse permutation) -> unpack must give the reverse-row-permuted codes.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, cols]));
    let packed = b.pack_i8(x);
    let idx = b.constant("idx", TensorType::f32(vec![rows]));
    let rt = b.unpack_i8(b.gather(packed, 0, idx), cols);
    let (xi, ii) = (x.id, idx.id);
    let g = b.finish(rt);
    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        Value::from(HostTensor::f32(vec![rows, cols], codes.clone())),
    );
    inputs.insert(
        ii,
        Value::from(HostTensor::f32(
            vec![rows],
            (0..rows).rev().map(|i| i as f32).collect(),
        )),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    for r in 0..rows {
        let src = (rows - 1 - r) * cols;
        assert_eq!(
            &got.as_f32().unwrap()[r * cols..(r + 1) * cols],
            &codes[src..src + cols],
            "gather dropped the int payload at row {r}"
        );
    }

    // transpose: pack -> transpose -> transpose-back -> unpack must round-trip exactly.
    let b2 = Builder::new();
    let x2 = b2.constant("x", TensorType::f32(vec![rows, cols]));
    let p = b2.pack_i8(x2); // [rows, w] i32
    let t = b2.transpose(p, vec![1, 0]); // [w, rows]
    let back = b2.transpose(t, vec![1, 0]); // [rows, w]
    let rt2 = b2.unpack_i8(back, cols);
    let x2i = x2.id;
    let g2 = b2.finish(rt2);
    let mut in2 = HashMap::new();
    in2.insert(
        x2i,
        Value::from(HostTensor::f32(vec![rows, cols], codes.clone())),
    );
    let got2 = eval(&g2, &in2, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    assert_eq!(
        got2.as_f32().unwrap(),
        &codes[..],
        "transpose round-trip dropped the int payload"
    );
}

#[test]
fn layernorm_matches_direct() {
    // spec 049: LayerNorm = (x - mean)/sqrt(var + eps) * w + b over the last axis, vs a direct computation.
    use poot_graph_ir::ops::layernorm;
    let b = Builder::new();
    let n = 16usize;
    let x = b.constant("x", TensorType::f32(vec![1, 1, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let bias = b.constant("b", TensorType::f32(vec![n]));
    let eps = 1e-6f32;
    let out = layernorm(&b, x, w, bias, eps);
    let (xi, wi, bi) = (x.id, w.id, bias.id);
    let g = b.finish(out);

    let xd = fill(n, 3);
    let wd = fill(n, 4);
    let bd = fill(n, 5);
    let mut inputs = HashMap::new();
    inputs.insert(xi, Value::from(HostTensor::f32(vec![1, 1, n], xd.clone())));
    inputs.insert(wi, Value::from(HostTensor::f32(vec![n], wd.clone())));
    inputs.insert(bi, Value::from(HostTensor::f32(vec![n], bd.clone())));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");

    let mean = xd.iter().sum::<f32>() / n as f32;
    let var = xd.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let den = (var + eps).sqrt();
    let want: Vec<f32> = xd
        .iter()
        .zip(&wd)
        .zip(&bd)
        .map(|((x, w), b)| ((x - mean) / den) * w + b)
        .collect();
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}

#[test]
fn rmsnorm_matches_direct() {
    let b = Builder::new();
    let n = 16usize;
    let x = b.constant("x", TensorType::f32(vec![1, 1, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let eps = 1e-6f32;
    let out = rmsnorm(&b, x, w, eps);
    let (xi, wi) = (x.id, w.id);
    let g = b.finish(out);

    let xd = fill(n, 1);
    let wd = fill(n, 2);
    let mut inputs = HashMap::new();
    inputs.insert(xi, Value::from(HostTensor::f32(vec![1, 1, n], xd.clone())));
    inputs.insert(wi, Value::from(HostTensor::f32(vec![n], wd.clone())));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");

    let ms = xd.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let den = (ms + eps).sqrt();
    let want: Vec<f32> = xd.iter().zip(&wd).map(|(x, w)| (x / den) * w).collect();
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}

/// Real-magnitude check: RMSNorm and LayerNorm at width 896 (Qwen2.5-0.5B hidden size) with two outlier
/// channels (+300 and -250) on a [-8, 8] baseline, against an independent f64 ground truth. Real hidden
/// states have such "massive activation" channels; `n=16` constant-fill inputs above do not. The
/// mean-of-squares sums only non-negative terms, so it cannot cancel and should stay tight.
#[test]
fn rmsnorm_and_layernorm_stay_accurate_at_real_hidden_dim_with_outlier_channels() {
    use poot_graph_ir::ops::layernorm;
    let n = 896usize; // real Qwen2.5-0.5B hidden size
    let eps = 1e-6f32; // real eps used by every Qwen2/Qwen3 config in poot-models
    let mut xd: Vec<f32> = fill(n, 41).iter().map(|v| v * 8.0).collect();
    xd[17] = 300.0; // "massive activation" outlier channel (real, documented LLM phenomenon)
    xd[500] = -250.0;
    let wd: Vec<f32> = fill(n, 42).iter().map(|v| v * 0.5 + 1.0).collect(); // weights near 1.0

    // RMSNorm.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let out = rmsnorm(&b, x, w, eps);
    let (xi, wi) = (x.id, w.id);
    let g = b.finish(out);
    let mut inputs = HashMap::new();
    inputs.insert(xi, Value::from(HostTensor::f32(vec![1, 1, n], xd.clone())));
    inputs.insert(wi, Value::from(HostTensor::f32(vec![n], wd.clone())));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");

    let ms64: f64 = xd.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
    let den64 = (ms64 + eps as f64).sqrt();
    let want_rms: Vec<f64> = xd
        .iter()
        .zip(&wd)
        .map(|(&x, &w)| ((x as f64) / den64) * (w as f64))
        .collect();
    let worst_rms = max_abs_error_f64(got.as_f32().unwrap(), &want_rms);
    assert!(
        worst_rms < 1e-4,
        "rmsnorm at real hidden_dim=896 + outlier channels vs f64 ground truth: err={worst_rms:e}"
    );

    // LayerNorm (same activations/eps; also has a bias vector).
    let bd: Vec<f32> = fill(n, 43).iter().map(|v| v * 0.1).collect();
    let bl = Builder::new();
    let xl = bl.constant("x", TensorType::f32(vec![1, 1, n]));
    let wl = bl.constant("w", TensorType::f32(vec![n]));
    let bias = bl.constant("b", TensorType::f32(vec![n]));
    let outl = layernorm(&bl, xl, wl, bias, eps);
    let (xli, wli, bli) = (xl.id, wl.id, bias.id);
    let gl = bl.finish(outl);
    let mut inl = HashMap::new();
    inl.insert(xli, Value::from(HostTensor::f32(vec![1, 1, n], xd.clone())));
    inl.insert(wli, Value::from(HostTensor::f32(vec![n], wd.clone())));
    inl.insert(bli, Value::from(HostTensor::f32(vec![n], bd.clone())));
    let gotl = eval(&gl, &inl, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");

    let mean64: f64 = xd.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var64: f64 = xd
        .iter()
        .map(|&v| ((v as f64) - mean64).powi(2))
        .sum::<f64>()
        / n as f64;
    let lden64 = (var64 + eps as f64).sqrt();
    let want_ln: Vec<f64> = xd
        .iter()
        .zip(&wd)
        .zip(&bd)
        .map(|((&x, &w), &bb)| (((x as f64) - mean64) / lden64) * (w as f64) + (bb as f64))
        .collect();
    let worst_ln = max_abs_error_f64(gotl.as_f32().unwrap(), &want_ln);
    assert!(
        worst_ln < 1e-3,
        "layernorm at real hidden_dim=896 + outlier channels vs f64 ground truth: err={worst_ln:e}"
    );
    eprintln!(
        "rmsnorm/layernorm at real hidden_dim=896 with +300/-250 outlier channels: rms_err={worst_rms:e} \
         layernorm_err={worst_ln:e} (both well within tolerance, as sum-of-nonneg-terms predicts)"
    );
}

#[test]
fn swiglu_matches_direct() {
    let b = Builder::new();
    let n = 16usize;
    let gate = b.constant("gate", TensorType::f32(vec![1, 1, n]));
    let up = b.constant("up", TensorType::f32(vec![1, 1, n]));
    let out = swiglu(&b, gate, up);
    let (gi, ui) = (gate.id, up.id);
    let g = b.finish(out);

    let gd = fill(n, 3);
    let ud = fill(n, 4);
    let mut inputs = HashMap::new();
    inputs.insert(gi, Value::from(HostTensor::f32(vec![1, 1, n], gd.clone())));
    inputs.insert(ui, Value::from(HostTensor::f32(vec![1, 1, n], ud.clone())));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");

    let want: Vec<f32> = gd
        .iter()
        .zip(&ud)
        .map(|(z, u)| (z / (1.0 + (-z).exp())) * u)
        .collect();
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-6);
}

#[test]
fn rope_matches_direct() {
    // single head, D=8, pos=3.
    let b = Builder::new();
    let d = 8usize;
    let max_pos = 16usize;
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
    let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let out = rope(&b, x, cos, sin, pos);
    let (xi, ci, si, pi) = (x.id, cos.id, sin.id, pos.id);
    let g = b.finish(out);

    let xd = fill(d, 5);
    let cosd = fill(max_pos * d, 6);
    let sind = fill(max_pos * d, 7);
    let posv = 3usize;
    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        Value::from(HostTensor::f32(vec![1, 1, 1, d], xd.clone())),
    );
    inputs.insert(
        ci,
        Value::from(HostTensor::f32(vec![max_pos, d], cosd.clone())),
    );
    inputs.insert(
        si,
        Value::from(HostTensor::f32(vec![max_pos, d], sind.clone())),
    );
    inputs.insert(pi, Value::from(HostTensor::i32(vec![], vec![posv as i32])));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");

    // direct half-split rope: out = x*cos_row + rotate_half(x)*sin_row, where rotate_half = [-x2, x1].
    let cos_row = &cosd[posv * d..posv * d + d];
    let sin_row = &sind[posv * d..posv * d + d];
    let half = d / 2;
    let mut rh = vec![0.0f32; d];
    for i in 0..half {
        rh[i] = -xd[half + i];
        rh[half + i] = xd[i];
    }
    let want: Vec<f32> = (0..d)
        .map(|i| xd[i] * cos_row[i] + rh[i] * sin_row[i])
        .collect();
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}

#[test]
fn rope_partial_matches_direct() {
    // phi3 partial rotary: head_dim D=8 but only the first ROT=4 dims rotate; [4,8) pass through.
    // The cos/sin tables are ROT-wide ([max_pos, 4]); the op infers the rotary width from them.
    let b = Builder::new();
    let (d, rot) = (8usize, 4usize);
    let max_pos = 16usize;
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, rot]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, rot]));
    let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let out = rope(&b, x, cos, sin, pos);
    let (xi, ci, si, pi) = (x.id, cos.id, sin.id, pos.id);
    let g = b.finish(out);

    let xd = fill(d, 5);
    let cosd = fill(max_pos * rot, 6);
    let sind = fill(max_pos * rot, 7);
    let posv = 3usize;
    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        Value::from(HostTensor::f32(vec![1, 1, 1, d], xd.clone())),
    );
    inputs.insert(
        ci,
        Value::from(HostTensor::f32(vec![max_pos, rot], cosd.clone())),
    );
    inputs.insert(
        si,
        Value::from(HostTensor::f32(vec![max_pos, rot], sind.clone())),
    );
    inputs.insert(pi, Value::from(HostTensor::i32(vec![], vec![posv as i32])));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");

    // direct: rotate the first ROT dims (half-split at ROT/2), pass [ROT, D) through unchanged.
    let cos_row = &cosd[posv * rot..posv * rot + rot];
    let sin_row = &sind[posv * rot..posv * rot + rot];
    let half = rot / 2;
    let mut rh = vec![0.0f32; rot];
    for i in 0..half {
        rh[i] = -xd[half + i];
        rh[half + i] = xd[i];
    }
    let mut want = vec![0.0f32; d];
    for i in 0..rot {
        want[i] = xd[i] * cos_row[i] + rh[i] * sin_row[i];
    }
    want[rot..d].copy_from_slice(&xd[rot..d]); // pass-through
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}

/// SC-006: every exact-range Cast violation is `EvalError::Cast(CastFault)` with the
/// literal `from`/`to`/`index`/`value` - `Cast(I32, F32)` of `16_777_217` (one past the largest
/// exactly-f32-representable integer) with and without a `cast_authority` attached (card 554d's
/// `exact_i32.rs::exact_i32_errors_preserve_typed_sources` already covers the with-authority case;
/// this proves the without-authority path hits the exact same fault, not a different one), plus
/// `Cast(F32, I32)` of a non-integer value and of NaN, and `Cast(I32, I8)` of a value outside
/// `-128..=127`.
///
/// Mutation: remove the `F32_EXACT_I32_MAX` range check in `walk.rs`'s `(I32, F32)` cast arm; the
/// `16_777_217` row publishes `16777216.0` instead of faulting, and this test goes red.
#[test]
fn cast_fault_carries_the_literal_from_to_index_and_value() {
    use crate::{CastFault, CastOperand, EvalError};

    // Cast(I32, F32) of 16_777_217 (one past F32_EXACT_I32_MAX), no cast_authority attached.
    let b = Builder::new();
    let x = i32_constant(&b, "x", vec![2]).unwrap();
    let y = b.cast(x, poot_tensor::DType::F32);
    let g = b.finish(y);
    let inputs = HashMap::from([(
        x.id,
        Value::from(poot_tensor::HostTensor::i32(vec![2], vec![0, 16_777_217])),
    )]);
    let error = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap_err();
    assert!(
        matches!(
            &error,
            EvalError::Cast(fault)
                if fault.index == 1 && fault.value == CastOperand::I32(16_777_217)
        ),
        "{error:?}"
    );
    let CastFault { from, to, .. } = *match error {
        EvalError::Cast(fault) => fault,
        _ => unreachable!(),
    };
    assert_eq!(from, poot_tensor::DType::I32);
    assert_eq!(to, poot_tensor::DType::F32);

    // Cast(F32, I32) of a non-integer value.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![2]));
    let y = b.cast(x, poot_tensor::DType::I32);
    let g = b.finish(y);
    let inputs = HashMap::from([(x.id, Value::from(HostTensor::f32(vec![2], vec![0.0, 1.5])))]);
    let error = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap_err();
    assert!(
        matches!(
            &error,
            EvalError::Cast(fault) if fault.index == 1 && fault.value == CastOperand::F32(1.5)
        ),
        "{error:?}"
    );

    // Cast(F32, I32) of NaN.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1]));
    let y = b.cast(x, poot_tensor::DType::I32);
    let g = b.finish(y);
    let inputs = HashMap::from([(x.id, Value::from(HostTensor::f32(vec![1], vec![f32::NAN])))]);
    let error = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap_err();
    let EvalError::Cast(fault) = &error else {
        panic!("expected a Cast fault, got {error:?}")
    };
    assert_eq!(fault.index, 0);
    let CastOperand::F32(got) = fault.value else {
        panic!("expected an F32 operand, got {:?}", fault.value)
    };
    assert!(got.is_nan(), "{got}");

    // Cast(I32, I8) of a value outside -128..=127.
    let b = Builder::new();
    let x = i32_constant(&b, "x", vec![2]).unwrap();
    let y = b.cast(x, poot_tensor::DType::I8);
    let g = b.finish(y);
    let inputs = HashMap::from([(
        x.id,
        Value::from(poot_tensor::HostTensor::i32(vec![2], vec![0, 300])),
    )]);
    let error = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap_err();
    assert!(
        matches!(
            &error,
            EvalError::Cast(fault) if fault.index == 1 && fault.value == CastOperand::I32(300)
        ),
        "{error:?}"
    );
}

/// SC-002 (card 555, deval B2): the cast table's literal bits, not cross-lane equality (one shared
/// function would make a cross-lane row blind to a lane-specific bug). F32->BF16 ties round to even
/// (`0x3F808000 -> 0x3F80`, `0x3F818000 -> 0x3F82`); a NaN stays a NaN with its high payload bits
/// (the bf16 word keeps 7 mantissa bits); F32->F16 overflow rounds to `+inf`; the E4M3 codec's own boundary byte
/// round-trips both ways.
///
/// Mutation: in `ops::cast::narrow_f32`'s BF16/F16 arms, truncate instead of round-to-nearest-even
/// (`f32::from_bits(bits & 0xFFFF_0000)`, no bias add). Confirmed red: both tie rows below publish
/// the rounded-down neighbor (`0x3F80 -> 0x3F80` still, but `0x3F818000 -> 0x3F81`, not `0x3F82`);
/// reverted, confirmed green.
#[test]
fn cast_table_literal_bits_sc002() {
    fn cast_bits(from: DType, to: DType, bits: u32) -> u32 {
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![1], from));
        let y = b.cast(x, to);
        let g = b.finish(y);
        let inputs = HashMap::from([(
            x.id,
            Value::from(HostTensor::f32(vec![1], vec![f32::from_bits(bits)])),
        )]);
        let out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("cast_quant_norm_rope tests evaluate dense graphs");
        out.to_f32().unwrap()[0].to_bits()
    }

    // F32 -> BF16: round to nearest, ties to even. The result widens exactly from its bf16 word, so
    // its low 16 bits are zero.
    assert_eq!(
        cast_bits(DType::F32, DType::BF16, 0x3F80_8000) & 0xFFFF_0000,
        0x3F80_0000,
        "exact tie rounds down to the even mantissa"
    );
    assert_eq!(
        cast_bits(DType::F32, DType::BF16, 0x3F81_8000) & 0xFFFF_0000,
        0x3F82_0000,
        "exact tie rounds up to the even mantissa"
    );

    // A NaN stays a NaN through the BF16 word: the quiet bit is forced and the high payload bits
    // (sign, exponent, the top mantissa bits) survive; the low 16 payload bits are not representable.
    let nan_payload = f32::from_bits(0x7FC1_2345);
    assert!(nan_payload.is_nan(), "fixture must be a NaN");
    assert_eq!(
        cast_bits(DType::F32, DType::BF16, nan_payload.to_bits()),
        0x7FC1_0000
    );

    // F32 -> F16: overflow saturates to infinity, not the largest finite f16 (65504.0).
    assert!(f32::from_bits(cast_bits(DType::F32, DType::F16, 65520.0f32.to_bits())).is_infinite());

    // The E4M3 codec's boundary byte round-trips: 0x7E decodes to 448.0 (E4M3's largest finite
    // magnitude) and 448.0 re-encodes to 0x7E; 0x7F (the canonical NaN byte) decodes to NaN. Routed
    // through `Cast(E4M3FN, F32)`/`Cast(F32, E4M3FN)` (card 555), not asserted
    // against `poot_quant::scalar` directly: that would pin only the codec, not the cast table's own
    // routing to it.
    let decode_via_cast = |byte: u8| -> f32 {
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![1], DType::E4M3FN));
        let y = b.cast(x, DType::F32);
        let g = b.finish(y);
        let inputs = HashMap::from([(
            x.id,
            Value::Host(crate::fp8::e4m3fn_tensor(vec![1], vec![byte]).unwrap()),
        )]);
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("cast_quant_norm_rope tests evaluate dense graphs")
            .as_f32()
            .unwrap()[0]
    };
    assert_eq!(decode_via_cast(0x7E), 448.0);
    assert!(decode_via_cast(0x7F).is_nan());

    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1]));
    let y = b.cast(x, DType::E4M3FN);
    let g = b.finish(y);
    let inputs = HashMap::from([(x.id, Value::from(HostTensor::f32(vec![1], vec![448.0f32])))]);
    let Value::Host(encoded) = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
    else {
        panic!("Cast(F32, E4M3FN) must publish e4m3fn storage")
    };
    assert_eq!(encoded.view().bytes(), [0x7E]);

    // I32 127 -> I8 127: the carded literal row for the existing X3 range-checked widen.
    let b = Builder::new();
    let x = i32_constant(&b, "x", vec![1]).unwrap();
    let y = b.cast(x, DType::I8);
    let g = b.finish(y);
    let inputs = HashMap::from([(x.id, Value::from(HostTensor::i32(vec![1], vec![127])))]);
    let out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    assert_eq!(
        out.dtype(),
        DType::I8,
        "Cast(I32, I8) publishes an I8 tensor"
    );
    assert_eq!(out.view().bytes(), [127u8], "two's complement byte");
}
