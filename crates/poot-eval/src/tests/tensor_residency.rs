//! The half-width carriers hold exactly their own 16-bit words: BF16 and F16 share the `Half` storage
//! class, differ only in dtype, and never carry an f32 mirror.

use poot_tensor::{DType, HostData, HostTensor};

#[test]
fn f16_carries_exactly_its_words_and_correct_numel() {
    let words: Vec<u16> = vec![0x3C00, 0xB800, 0x0000, 0x7C00]; // 1.0, -0.5, 0, +inf
    let t = HostTensor::f16(vec![2, 2], words.clone());
    assert_eq!(t.dtype(), DType::F16);
    assert_eq!(t.as_half(), Some(words.as_slice()));
    assert!(t.as_f32().is_none(), "an F16 tensor has no f32 payload");
    assert_eq!(t.numel(), 4);
    let widened = t.to_f32().unwrap();
    assert_eq!(&widened[..2], &[1.0, -0.5]);
    assert!(widened[3].is_infinite());
}

#[test]
fn bf16_and_f16_numel_and_storage_agree_on_the_same_shape() {
    let shape = vec![3, 5];
    let n: usize = shape.iter().product();
    let words = vec![0u16; n];
    let bf16_t = HostTensor::bf16(shape.clone(), words.clone());
    let f16_t = HostTensor::f16(shape, words);
    assert_eq!(bf16_t.numel(), n);
    assert_eq!(bf16_t.numel(), f16_t.numel());
    assert!(matches!(bf16_t.data(), HostData::Half(_)));
    assert!(matches!(f16_t.data(), HostData::Half(_)));
    assert_ne!(bf16_t.dtype(), f16_t.dtype());
}
