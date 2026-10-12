//! `materialize_dense`'s native bf16/f16 word passthrough (card 543; moved from `poot-load`'s own
//! tests when `materialize_dense` moved here, since it returns this crate's `HostTensor`).

use crate::materialize_dense;
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::DType;

/// A one-entry store holding `name` as a dense tensor of `dtype` over the raw little-endian `bytes`.
fn dense_store(name: &str, dtype: DType, shape: Vec<usize>, bytes: Vec<u8>) -> WeightStore {
    let dense = DenseWeight::try_new(dtype, shape, StoredBytes::from(bytes)).unwrap();
    let mut builder = WeightStore::builder();
    builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    builder.build()
}

/// A bf16 dense entry yields a BF16 tensor holding the raw stored words (no f32 payload); widening is
/// lossless.
#[test]
fn bf16_native_bytes_preserved() {
    // 1.0 (0x3F80) and -0.5 (0xBF00), little-endian.
    let store = dense_store("w", DType::BF16, vec![2], vec![0x80, 0x3F, 0x00, 0xBF]);

    let t = materialize_dense(&store, "w").expect("w must be present");
    assert_eq!(t.dtype(), DType::BF16);
    assert!(t.as_f32().is_none(), "a BF16 tensor has no f32 payload");
    assert_eq!(
        t.as_half().expect("BF16 words"),
        &[0x3F80u16, 0xBF00],
        "native words must match the raw stored words"
    );
    // widening is lossless
    assert_eq!(&*t.to_f32().unwrap(), &[1.0f32, -0.5]);

    // f32 source: stays F32, no half words
    let store2 = dense_store(
        "x",
        DType::F32,
        vec![1],
        1.0f32.to_bits().to_le_bytes().to_vec(),
    );
    let t2 = materialize_dense(&store2, "x").expect("x must be present");
    assert_eq!(t2.dtype(), DType::F32);
    assert!(
        t2.as_half().is_none(),
        "F32 source must carry no half words"
    );
}

/// Spec 135 Phase 2: as `bf16_native_bytes_preserved` for F16. A dense F16 entry yields an F16 tensor
/// holding the raw words; a WMMA/f16-matmul GPU consumer uploads the native words.
#[test]
fn f16_native_bytes_preserved() {
    // 1.0 (0x3C00) and -0.5 (0xB800), little-endian.
    let store = dense_store("w", DType::F16, vec![2], vec![0x00, 0x3C, 0x00, 0xB8]);

    let t = materialize_dense(&store, "w").expect("w must be present");
    assert_eq!(t.dtype(), DType::F16);
    assert!(t.as_f32().is_none(), "an F16 tensor has no f32 payload");
    assert_eq!(
        t.as_half().expect("F16 words"),
        &[0x3C00u16, 0xB800],
        "native words must match the raw stored words"
    );
    assert_eq!(&*t.to_f32().unwrap(), &[1.0f32, -0.5]);

    // f32 source: stays F32.
    let store2 = dense_store(
        "x",
        DType::F32,
        vec![1],
        1.0f32.to_bits().to_le_bytes().to_vec(),
    );
    let t2 = materialize_dense(&store2, "x").expect("x must be present");
    assert_eq!(t2.dtype(), DType::F32);
    assert!(
        t2.as_half().is_none(),
        "F32 source must carry no half words"
    );
}
