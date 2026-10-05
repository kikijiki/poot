//! `materialize_dense`'s native bf16/f16 word passthrough (card 543; moved from `poot-load`'s own
//! tests when `materialize_dense` moved here, since it returns this crate's `HostTensor`).

use crate::materialize_dense;
use poot_load::safetensors;
use poot_tensor::DType;

/// Loading a bf16 tensor yields a BF16 tensor holding the raw checkpoint words (no f32 payload);
/// widening is lossless.
#[test]
fn bf16_native_bytes_preserved() {
    // Minimal in-memory safetensors file with one BF16 tensor: 1.0 (0x3F80) and -0.5 (0xBF00).
    let bf16_vals: Vec<u8> = vec![
        0x80, 0x3F, // 1.0 in bf16 LE
        0x00, 0xBF, // -0.5 in bf16 LE
    ];
    let header = serde_json::json!({
        "w": {
            "dtype": "BF16",
            "shape": [2],
            "data_offsets": [0_u64, 4_u64]
        }
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let header_len = header_bytes.len() as u64;
    let mut file: Vec<u8> = Vec::new();
    file.extend_from_slice(&header_len.to_le_bytes());
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&bf16_vals);
    let tmp = poot_test_util::unique_temp_path("poot_test_bf16_native.safetensors");
    std::fs::write(&tmp, &file).unwrap();
    let store = safetensors::load_weight_store_file(&tmp).unwrap();

    let t = materialize_dense(&store, "w").expect("w must be present");
    assert_eq!(t.dtype(), DType::BF16);
    assert!(t.as_f32().is_none(), "a BF16 tensor has no f32 payload");
    // the exact raw words from the safetensors file
    assert_eq!(
        t.as_half().expect("BF16 words"),
        &[0x3F80u16, 0xBF00],
        "native words must match the raw checkpoint words"
    );
    // widening is lossless
    assert_eq!(&*t.to_f32().unwrap(), &[1.0f32, -0.5]);
    // f32 source: stays F32, no half words
    let raw_f32 = 1.0f32.to_bits().to_le_bytes();
    let header2 = serde_json::json!({
        "x": {"dtype": "F32", "shape": [1], "data_offsets": [0_u64, 4_u64]}
    });
    let hb2 = serde_json::to_vec(&header2).unwrap();
    let hl2 = hb2.len() as u64;
    let mut f2: Vec<u8> = Vec::new();
    f2.extend_from_slice(&hl2.to_le_bytes());
    f2.extend_from_slice(&hb2);
    f2.extend_from_slice(&raw_f32);
    let tmp2 = poot_test_util::unique_temp_path("poot_test_f32_native.safetensors");
    std::fs::write(&tmp2, &f2).unwrap();
    let store2 = safetensors::load_weight_store_file(&tmp2).unwrap();
    let t2 = materialize_dense(&store2, "x").expect("x must be present");
    assert_eq!(t2.dtype(), DType::F32);
    assert!(
        t2.as_half().is_none(),
        "F32 source must carry no half words"
    );
}

/// Spec 135 Phase 2: as `bf16_native_bytes_preserved` for F16. Loading an F16 tensor yields an F16
/// tensor holding the raw words; a WMMA/f16-matmul GPU consumer uploads the native words.
#[test]
fn f16_native_bytes_preserved() {
    // Minimal in-memory safetensors file with one F16 tensor: 1.0 (0x3C00) and -0.5 (0xB800).
    let f16_vals: Vec<u8> = vec![
        0x00, 0x3C, // 1.0 in f16 LE
        0x00, 0xB8, // -0.5 in f16 LE
    ];
    let header = serde_json::json!({
        "w": {
            "dtype": "F16",
            "shape": [2],
            "data_offsets": [0_u64, 4_u64]
        }
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let header_len = header_bytes.len() as u64;
    let mut file: Vec<u8> = Vec::new();
    file.extend_from_slice(&header_len.to_le_bytes());
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&f16_vals);
    let tmp = poot_test_util::unique_temp_path("poot_test_f16_native.safetensors");
    std::fs::write(&tmp, &file).unwrap();
    let store = safetensors::load_weight_store_file(&tmp).unwrap();

    let t = materialize_dense(&store, "w").expect("w must be present");
    assert_eq!(t.dtype(), DType::F16);
    assert!(t.as_f32().is_none(), "an F16 tensor has no f32 payload");
    // the exact raw words from the safetensors file
    assert_eq!(
        t.as_half().expect("F16 words"),
        &[0x3C00u16, 0xB800],
        "native words must match the raw checkpoint words"
    );
    assert_eq!(&*t.to_f32().unwrap(), &[1.0f32, -0.5]);
    // f32 source: stays F32.
    let raw_f32 = 1.0f32.to_bits().to_le_bytes();
    let header2 = serde_json::json!({
        "x": {"dtype": "F32", "shape": [1], "data_offsets": [0_u64, 4_u64]}
    });
    let hb2 = serde_json::to_vec(&header2).unwrap();
    let hl2 = hb2.len() as u64;
    let mut f2: Vec<u8> = Vec::new();
    f2.extend_from_slice(&hl2.to_le_bytes());
    f2.extend_from_slice(&hb2);
    f2.extend_from_slice(&raw_f32);
    let tmp2 = poot_test_util::unique_temp_path("poot_test_f32_native_for_f16.safetensors");
    std::fs::write(&tmp2, &f2).unwrap();
    let store2 = safetensors::load_weight_store_file(&tmp2).unwrap();
    let t2 = materialize_dense(&store2, "x").expect("x must be present");
    assert_eq!(t2.dtype(), DType::F32);
    assert!(
        t2.as_half().is_none(),
        "F32 source must carry no half words"
    );
}
