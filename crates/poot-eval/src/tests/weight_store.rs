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

/// The meta of a BF16 `[n]` weight `w` after the graph converts it to F16 (`GraphEdit::convert_weight`), and of the
/// same weight unconverted.
fn weight_metas(n: usize) -> (poot_graph_ir::ValueMeta, poot_graph_ir::ValueMeta) {
    use poot_graph_ir::{Builder, TensorType};
    let b = Builder::for_tests();
    let w = b
        .constant("w", TensorType::new(vec![n], DType::BF16))
        .unwrap();
    let out = b.cast(w, DType::F32).unwrap();
    let graph = b.finish(out).unwrap();
    let mut edit = graph.edit();
    edit.convert_weight(w.id()).unwrap();
    let converted = edit.commit().unwrap();
    (converted.meta(w.id()).clone(), graph.meta(w.id()).clone())
}

/// POOT-1477: a weight the graph converts binds as the bind converts it: each bf16 rounded to nearest even f16
/// (normal, subnormal-in-f16 and underflowing values, both signs, signed zero), the same words
/// `lanes::Encoding::Bf16AsF16Words` uploads, and an unconverted weight binds as stored.
///
/// Mutation: in `declared_weight` return `stored` unchanged; the converted weight is BF16 and this goes red.
#[test]
fn a_converted_weight_binds_as_the_f16_the_bind_uploads() {
    use crate::declared_weight;
    let bits: [u16; 6] = [0x3f80, 0xbf00, 0x3580, 0x3000, 0x8000, 0x0001];
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let store = dense_store("w", DType::BF16, vec![bits.len()], bytes);
    let stored = materialize_dense(&store, "w").unwrap();
    let (converted, plain) = weight_metas(bits.len());

    let declared = declared_weight("w", &converted, stored.clone()).unwrap();
    assert_eq!(declared.dtype(), DType::F16);
    let want: Vec<u16> = bits
        .iter()
        .map(|&b| poot_quant::scalar::f32_to_f16(poot_quant::scalar::bf16_to_f32(b)))
        .collect();
    assert_eq!(declared.as_half().unwrap(), want.as_slice());
    assert_ne!(
        declared.as_half().unwrap()[2],
        poot_quant::scalar::f32_to_f16(0.0),
        "0x3580 (2^-20-ish) survives as an f16 subnormal, not zero"
    );
    assert_eq!(
        declared_weight("w", &plain, stored.clone()).unwrap(),
        stored,
        "an unconverted weight binds as stored"
    );
}

/// POOT-1477: a bf16 that rounds past f16's range (65536) is the same typed refusal the bind gives, naming the
/// element, not an infinity in the oracle's weight. Mutation: delete the `0x7c00` test in `declared_weight`.
#[test]
fn a_converted_weight_past_the_f16_range_is_refused_like_the_bind() {
    use crate::{MaterializeError, declared_weight};
    let bits: [u16; 3] = [0x3f80, 0x4780, 0x3f80];
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let store = dense_store("w", DType::BF16, vec![3], bytes);
    let stored = materialize_dense(&store, "w").unwrap();
    let (converted, _) = weight_metas(3);
    assert_eq!(
        declared_weight("w", &converted, stored),
        Err(MaterializeError::PastF16Range {
            name: "w".to_string(),
            index: 1,
            bits: 0x4780
        })
    );
}
