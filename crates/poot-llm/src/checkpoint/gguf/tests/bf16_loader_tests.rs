use poot_eval::materialize_dense;
use poot_load::safetensors::load_weight_store_bytes;
use poot_tensor::DType;

/// Build a minimal in-memory SafeTensors payload containing a single BF16 tensor.
fn make_bf16_safetensors(name: &str, shape: &[usize], values_f32: &[f32]) -> Vec<u8> {
    // Encode f32 values as bf16 (top 2 bytes of the f32 bit pattern, little-endian).
    let bf16_bytes: Vec<u8> = values_f32
        .iter()
        .flat_map(|&v| {
            let bits = v.to_bits();
            [(bits >> 16) as u8, (bits >> 24) as u8]
        })
        .collect();
    let data_len = bf16_bytes.len();
    let shape_str = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let header_json = format!(
        "{{\"{}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[0,{data_len}]}}}}",
        name, shape_str
    );
    let header_bytes = header_json.as_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&bf16_bytes);
    out
}

/// Same but F32.
fn make_f32_safetensors(name: &str, shape: &[usize], values_f32: &[f32]) -> Vec<u8> {
    let data: Vec<u8> = values_f32.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let data_len = data.len();
    let shape_str = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let header_json = format!(
        "{{\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[0,{data_len}]}}}}",
        name, shape_str
    );
    let header_bytes = header_json.as_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&data);
    out
}

#[test]
fn bf16_checkpoint_tensor_stays_bf16_words() {
    // A 2x3 BF16 tensor: six values, held as exactly the bf16 words the checkpoint stored (no f32 mirror).
    let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes = make_bf16_safetensors("weight", &[2, 3], &vals);
    let st = load_weight_store_bytes(&bytes).expect("parse bf16 safetensors");
    let rt = materialize_dense(&st, "weight").expect("weight tensor");

    assert_eq!(rt.dtype(), DType::BF16);
    assert_eq!(rt.shape(), [2, 3]);
    assert!(rt.as_f32().is_none(), "a bf16 tensor has no f32 payload");

    // The words are the top 16 bits of each f32's bit pattern.
    let expected_words: Vec<u16> = vals.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    assert_eq!(rt.as_half().expect("bf16 words"), expected_words);

    // The explicit widening is exact for these values.
    assert_eq!(&*rt.to_f32().expect("widen bf16"), vals.as_slice());
}

#[test]
fn f32_source_stays_f32() {
    let vals: Vec<f32> = vec![1.5, 2.5, 3.5];
    let bytes = make_f32_safetensors("weight", &[3], &vals);
    let st = load_weight_store_bytes(&bytes).expect("parse f32 safetensors");
    let rt = materialize_dense(&st, "weight").expect("weight tensor");
    assert_eq!(rt.dtype(), DType::F32);
    assert!(rt.as_half().is_none(), "an f32 source has no 16-bit words");
    assert_eq!(rt.as_f32().expect("f32 values"), vals);
}

#[test]
fn bf16_dtype_flow_detected_via_q_proj() {
    // A safetensors with a bf16 q_proj loads as a BF16 tensor, an f32 q_proj as F32. This is the dtype the
    // Runner reads to set proj_dtype.
    let vals: Vec<f32> = vec![0.5f32; 4];
    let bf16_bytes =
        make_bf16_safetensors("model.layers.0.self_attn.q_proj.weight", &[2, 2], &vals);
    let st = load_weight_store_bytes(&bf16_bytes).expect("parse");
    let rt = materialize_dense(&st, "model.layers.0.self_attn.q_proj.weight").expect("tensor");
    assert_eq!(rt.dtype(), DType::BF16, "BF16 q_proj.weight stays BF16");
    assert_eq!(rt.as_half().unwrap().len(), 4, "4 words");

    let f32_bytes = make_f32_safetensors("model.layers.0.self_attn.q_proj.weight", &[2, 2], &vals);
    let st2 = load_weight_store_bytes(&f32_bytes).expect("parse");
    let rt2 = materialize_dense(&st2, "model.layers.0.self_attn.q_proj.weight").expect("tensor");
    assert_eq!(rt2.dtype(), DType::F32, "F32 q_proj.weight stays F32");
}
