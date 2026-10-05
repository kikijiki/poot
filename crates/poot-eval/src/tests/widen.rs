//! HostTensor -> Value widening (Card 440 slice 2/3 residual, Card 447 phase 3; folded into
//! `From<HostTensor> for Value` at card 545b).
//!
//! `HostTensor -> Value` (via `From`/`.into()`) is total and lossless; the reverse is partial. These
//! tests pin payload preservation so a mutation of `From<HostTensor> for Value` (e.g. zero-filling
//! data, or returning a non-Host variant) fails.

use crate::Value;
use poot_tensor::{DType, HostTensor};

#[test]
fn widen_preserves_host_payload_shape_dtype_and_arc_identity() {
    let tensor = HostTensor::f32(vec![2, 3], vec![0.5, -1.0, 2.25, 3.0, -4.5, 6.0]);
    let data_ptr = tensor.as_f32().unwrap().as_ptr();
    let value = Value::from(tensor.clone());
    let Value::Host(got) = &value else {
        panic!("widening must produce Value::Host, got {value:?}");
    };
    assert_eq!(got.shape(), tensor.shape());
    assert_eq!(got.dtype(), DType::F32);
    assert_eq!(got.as_f32().unwrap(), tensor.as_f32().unwrap());
    assert_eq!(got.as_f32().unwrap().as_ptr(), data_ptr, "no payload copy");
    assert_eq!(got, &tensor);
}

#[test]
fn widen_is_total_over_tensor_kinds_and_keeps_each_payload() {
    let ints = HostTensor::i32(vec![3], vec![7, -3, 11]);
    let scalar = HostTensor::i32(vec![], vec![5]);
    let float = HostTensor::f32(vec![1], vec![0.25]);
    let half = HostTensor::bf16(vec![2], vec![0x3F80, 0xBF00]);
    for (label, tensor) in [
        ("ints", ints),
        ("scalar", scalar),
        ("float", float),
        ("bf16", half),
    ] {
        let value = Value::from(tensor.clone());
        let Value::Host(got) = &value else {
            panic!("{label}: widening must produce Value::Host, got {value:?}");
        };
        assert_eq!(got, &tensor, "{label}: tensor must survive widening");
        assert_eq!(got.dtype(), tensor.dtype(), "{label}");
        assert_eq!(got.as_i32(), tensor.as_i32(), "{label}");
        assert_eq!(got.as_half(), tensor.as_half(), "{label}");
        assert_eq!(
            got.as_f32().map(<[f32]>::as_ptr),
            tensor.as_f32().map(<[f32]>::as_ptr),
            "{label}: no copy"
        );
    }
    assert!(
        matches!(Value::from(HostTensor::i32(vec![1], vec![0])), Value::Host(t) if t.as_i32().is_some()),
        "I32 words must remain authoritative after widening"
    );
}
