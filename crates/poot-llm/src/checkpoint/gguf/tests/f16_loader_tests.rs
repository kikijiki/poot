//! F16 weight residency: mirrors `bf16_loader_tests` for F16, plus the end-to-end gate the f16-matmul GPU
//! consumer needs: the loader keeps f16 words through `transpose2d`/`build_weights` (no silent widening
//! to f32), and an F16-tagged graph MatMul bound to the resulting HostTensor computes the same result as
//! an f32 reference.

use poot_eval::{EvalBudget, EvalOptions, materialize_dense};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::types::TensorType;
use poot_load::safetensors::load_weight_store_bytes;
use poot_tensor::DType;
use poot_tensor::HostTensor;

use crate::checkpoint::gguf::transpose2d;

/// Build a minimal in-memory SafeTensors payload containing a single F16 tensor.
fn make_f16_safetensors(name: &str, shape: &[usize], values_f32: &[f32]) -> Vec<u8> {
    let f16_bytes: Vec<u8> = values_f32
        .iter()
        .flat_map(|&v| poot_load::gguf::f32_to_f16(v).to_le_bytes())
        .collect();
    let data_len = f16_bytes.len();
    let shape_str = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let header_json = format!(
        "{{\"{}\":{{\"dtype\":\"F16\",\"shape\":[{}],\"data_offsets\":[0,{data_len}]}}}}",
        name, shape_str
    );
    let header_bytes = header_json.as_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&f16_bytes);
    out
}

/// The f16 bit patterns of `values`, the words an F16 `HostTensor` holds.
fn f16_words(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&v| poot_load::gguf::f32_to_f16(v))
        .collect()
}

#[test]
fn f16_checkpoint_tensor_stays_f16_words() {
    // A 2x3 F16 tensor: six values (all exactly representable in f16), held as exactly the f16 words the
    // checkpoint stored (no f32 mirror).
    let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes = make_f16_safetensors("weight", &[2, 3], &vals);
    let st = load_weight_store_bytes(&bytes).expect("parse f16 safetensors");
    let rt = materialize_dense(&st, "weight").expect("weight tensor");

    assert_eq!(rt.dtype(), DType::F16);
    assert_eq!(rt.shape(), [2, 3]);
    assert!(rt.as_f32().is_none(), "an f16 tensor has no f32 payload");
    assert_eq!(rt.as_half().expect("f16 words"), f16_words(&vals));
    assert_eq!(&*rt.to_f32().expect("widen f16"), vals.as_slice());
}

#[test]
fn f32_source_stays_f32() {
    let vals: Vec<f32> = vec![1.5, 2.5, 3.5];
    let data: Vec<u8> = vals.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let data_len = data.len();
    let header_json = format!(
        "{{\"weight\":{{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,{data_len}]}}}}"
    );
    let header_bytes = header_json.as_bytes();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header_bytes);
    bytes.extend_from_slice(&data);
    let st = load_weight_store_bytes(&bytes).expect("parse f32 safetensors");
    let rt = materialize_dense(&st, "weight").expect("weight tensor");
    assert_eq!(rt.dtype(), DType::F32);
    assert!(rt.as_half().is_none(), "an f32 source has no 16-bit words");
}

#[test]
fn f16_dtype_flow_detected_via_q_proj() {
    // Mirrors bf16_dtype_flow_detected_via_q_proj: an F16 q_proj loads as F16 (not BF16), an F32 q_proj as
    // F32. This is the dtype Runner::load reads to set proj_dtype to DType::F16.
    let vals: Vec<f32> = vec![0.5f32; 4];
    let f16_bytes = make_f16_safetensors("model.layers.0.self_attn.q_proj.weight", &[2, 2], &vals);
    let st = load_weight_store_bytes(&f16_bytes).expect("parse");
    let rt = materialize_dense(&st, "model.layers.0.self_attn.q_proj.weight").expect("tensor");
    assert_eq!(rt.dtype(), DType::F16, "F16 q_proj.weight stays F16");
    assert_eq!(rt.as_half().unwrap().len(), 4, "4 words");
}

#[test]
fn f16_transpose2d_keeps_f16_words_not_widened() {
    // transpose2d is what build_weights calls for every "*.proj.weight" tensor. An F16 tensor must come out
    // F16 (its words moved into the [c,r] layout), never widened to f32.
    let (r, c) = (2usize, 3usize);
    let vals: Vec<f32> = (0..r * c).map(|i| i as f32 * 0.5).collect();
    let words = f16_words(&vals);
    let rt = HostTensor::f16(vec![r, c], words.clone());
    let t = transpose2d(&rt);
    assert_eq!(t.shape(), [c, r], "transpose2d must swap dims");
    assert_eq!(t.dtype(), DType::F16, "transpose2d must not widen f16");
    let got = t.as_half().expect("f16 words");
    for i in 0..r {
        for j in 0..c {
            assert_eq!(
                got[j * r + i],
                words[i * c + j],
                "transposed word [{i},{j}]"
            );
        }
    }
}

/// End-to-end gate: an F16-tagged graph `MatMul`, bound to the Tensor from the load->transpose2d
/// residency path (native f16 bytes kept), computes the same result as an independent f32 reference matmul
/// within f16 rounding tolerance. This checks the loader-to-eval plumbing, not just byte preservation.
#[test]
fn f16_matmul_through_loaded_weight_matches_f32_reference() {
    // w: checkpoint-shape [out=2, in=3] (HF linear layout), transposed by transpose2d to the matmul layout
    // [in=3, out=2], carrying native f16 bytes throughout.
    let w_vals: Vec<f32> = vec![0.5, -1.25, 2.0, 0.25, -0.75, 1.5]; // exact in f16
    let rt = HostTensor::f16(vec![2, 3], f16_words(&w_vals));
    let w_tensor = transpose2d(&rt); // shape [3, 2], still f16 words
    assert_eq!(w_tensor.dtype(), DType::F16, "loaded weight must stay f16");
    let w_f32 = w_tensor.to_f32().expect("widen f16 weight").into_owned();

    // a[1,2,3] @ w[3,2] -> [1,2,2], all F16-tagged (DType::F16 flows from proj_dtype).
    let b = Builder::new();
    let a = b.constant("a", TensorType::new(vec![1, 2, 3], DType::F16));
    let w = b.constant("w", TensorType::new(vec![3, 2], DType::F16));
    let mm = b.matmul(a, w);
    assert_eq!(
        b.aval(mm).dtype,
        DType::F16,
        "matmul must keep the F16 dtype"
    );
    let g = b.finish(mm);

    let a_vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut inputs = std::collections::HashMap::new();
    inputs.insert(
        a.id,
        poot_eval::Value::from(HostTensor::f16(vec![1, 2, 3], f16_words(&a_vals))),
    );
    inputs.insert(w.id, poot_eval::Value::from(w_tensor.clone()));
    let out = poot_eval::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("eval f16 matmul")
        .output
        .into_host()
        .expect("dense output");

    // Independent f32 reference: plain matmul against the same logical weight values (the widened f16
    // weight), narrowed to f16 as the oracle's MatMul does for an F16-dtype output.
    let (m, k, n) = (2usize, 3usize, 2usize);
    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_vals[i * k + kk] * w_f32[kk * n + j];
            }
            want[i * n + j] = poot_quant::scalar::f16_to_f32(poot_load::gguf::f32_to_f16(acc));
        }
    }
    assert_eq!(out.dtype(), DType::F16);
    assert_eq!(out.shape(), [1, 2, 2]);
    for (i, (&got, &w)) in out.to_f32().unwrap().iter().zip(&want).enumerate() {
        assert_eq!(
            got.to_bits(),
            w.to_bits(),
            "f16 matmul[{i}]: got {got} want {w}"
        );
    }
}

/// F16 residency fuzz: an F16-tagged matmul (f16-word weight through `transpose2d`) must be bit-exact to
/// an independent f32-accumulate-then-narrow reference across random shapes. The contract: keep native f16
/// bytes through bind, accumulate in f32, then narrow the output to f16. Both sides consume the same
/// widened-f16 inputs and narrow only the result, so any divergence is a bug in the F16 matmul K-loop or
/// output narrowing.
///
/// Random space (fixed seed, 100 cases): m in 1..6, k in 1..20, n in 1..6; a and w are random f32
/// pre-rounded to f16.
#[test]
fn f16_matmul_fuzz_matches_f32_narrowed_reference() {
    // Deterministic xorshift64 step (shared mutable seed across the nested uses below).
    fn xs(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }
    // Draw an f32 in [-2, 2) rounded to the nearest f16 (exact in f16).
    fn next_f16(s: &mut u64) -> f32 {
        let u = ((xs(s) >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0;
        poot_quant::scalar::f16_to_f32(poot_load::gguf::f32_to_f16(u))
    }
    let mut seed = 0xf16f_1220_c0ff_ee11u64;

    let mut cases = 0usize;
    for _ in 0..100u64 {
        let m = 1 + (xs(&mut seed) % 5) as usize;
        let k = 1 + (xs(&mut seed) % 19) as usize;
        let n = 1 + (xs(&mut seed) % 5) as usize;

        // Weight comes through the loader as native f16: checkpoint [n, k] -> transpose2d [k, n].
        let w_ck: Vec<f32> = (0..n * k).map(|_| next_f16(&mut seed)).collect();
        let rt = HostTensor::f16(vec![n, k], f16_words(&w_ck));
        let w_tensor = transpose2d(&rt); // [k, n], still f16 words
        assert_eq!(w_tensor.dtype(), DType::F16, "loaded weight must stay f16");
        let w_f32 = w_tensor.to_f32().expect("widen f16 weight").into_owned();

        let a_vals: Vec<f32> = (0..m * k).map(|_| next_f16(&mut seed)).collect();

        // a[1,m,k] @ w[k,n] -> [1,m,n], all F16-tagged.
        let b = Builder::new();
        let a = b.constant("a", TensorType::new(vec![1, m, k], DType::F16));
        let w = b.constant("w", TensorType::new(vec![k, n], DType::F16));
        let mm = b.matmul(a, w);
        assert_eq!(b.aval(mm).dtype, DType::F16, "matmul keeps F16 dtype");
        let g = b.finish(mm);
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(
            a.id,
            poot_eval::Value::from(HostTensor::f16(vec![1, m, k], f16_words(&a_vals))),
        );
        inputs.insert(w.id, poot_eval::Value::from(w_tensor.clone()));
        let out = poot_eval::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("eval f16 matmul")
            .output
            .into_host()
            .expect("dense output");

        // Independent reference: f32-accumulate over the same widened inputs, narrow output to f16.
        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_vals[i * k + kk] * w_f32[kk * n + j];
                }
                want[i * n + j] = poot_quant::scalar::f16_to_f32(poot_load::gguf::f32_to_f16(acc));
            }
        }
        assert_eq!(out.shape(), [1, m, n], "m={m} k={k} n={n}: shape");
        for (i, (&got, &w)) in out.to_f32().unwrap().iter().zip(&want).enumerate() {
            assert_eq!(
                got.to_bits(),
                w.to_bits(),
                "m={m} k={k} n={n} out[{i}]: got {got} want {w}"
            );
        }
        cases += 1;
    }
    eprintln!("f16 residency matmul fuzz: {cases} random shapes, all bit-exact vs f32-narrowed");
}
