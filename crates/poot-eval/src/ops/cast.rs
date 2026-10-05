//! `Cast`: the float-precision narrowing casts, the storage-aware E4M3FN casts, BF16 widening, and the exact-I32
//! to F32 range check.

use crate::{EvalError, Value, fp8};
use poot_tensor::{DType, HostTensor};

/// The largest magnitude up to which every I32 word is exactly representable as an f32 (2^24).
pub(crate) const F32_EXACT_I32_MAX: i32 = 16_777_216;

/// Narrow `values` to `to`'s half-width words (round-to-nearest, ties-to-even; a NaN stays a NaN).
/// `to` must be `BF16` or `F16`.
pub(crate) fn narrow_f32(
    shape: Vec<usize>,
    values: &[f32],
    to: DType,
) -> Result<HostTensor, EvalError> {
    match to {
        DType::BF16 => Ok(HostTensor::bf16(
            shape,
            values
                .iter()
                .map(|&v| poot_runtime_common::f32_to_bf16(v))
                .collect::<Vec<u16>>(),
        )),
        DType::F16 => Ok(HostTensor::f16(
            shape,
            values
                .iter()
                .map(|&v| poot_load::gguf::f32_to_f16(v))
                .collect::<Vec<u16>>(),
        )),
        other => Err(EvalError::unsupported(
            "cast",
            format!("{other} is not a half-width float"),
        )),
    }
}

/// Store an equation's F32 result as its declared dtype: the arithmetic ran in f32, so a BF16/F16
/// result is narrowed once, here, as the GPU kernels do on store. Every other result is already in
/// its declared dtype's class and is returned as is.
pub(crate) fn store_as(result: HostTensor, declared: DType) -> Result<HostTensor, EvalError> {
    match (result.dtype(), declared) {
        (DType::F32, DType::BF16 | DType::F16) => {
            let values = result.to_f32()?;
            narrow_f32(result.shape().to_vec(), &values, declared)
        }
        _ => Ok(result),
    }
}

/// The float-precision `Cast` among `F32`, `BF16` and `F16`: the source widens exactly to f32 and the
/// result is narrowed to `to` (round-to-nearest-even), or stays f32 for `to == F32`.
pub(crate) fn cast_float(x: &HostTensor, to: DType) -> Result<HostTensor, EvalError> {
    let values = x.to_f32()?;
    match to {
        DType::F32 => Ok(HostTensor::f32(x.shape().to_vec(), values.into_owned())),
        DType::BF16 | DType::F16 => narrow_f32(x.shape().to_vec(), &values, to),
        other => Err(EvalError::unsupported(
            "cast",
            format!("{other} is not a float-precision cast target"),
        )),
    }
}

/// The storage-aware walk's `Cast`: the identity for a same-dtype cast, and the E4M3FN encode/decode
/// pair `F32 -> E4M3FN` and `E4M3FN -> F32` over the raw-byte carrier. `walk.rs` handles every other
/// admitted pair itself (BF16/F16 precision, I32 range checks); this is the E4M3FN-only remainder.
pub(crate) fn cast_value(from: DType, to: DType, value: Value) -> Result<Value, EvalError> {
    match (from, to, value) {
        (from, to, value) if from == to => Ok(value),
        (DType::F32, DType::E4M3FN, Value::Host(t)) => Ok(Value::Host(fp8::encode_e4m3fn_tensor(
            t.shape().to_vec(),
            &t.to_f32()?,
        )?)),
        (DType::E4M3FN, DType::F32, Value::Host(t)) => Ok(Value::Host(HostTensor::f32(
            t.shape().to_vec(),
            t.to_f32()?.into_owned(),
        ))),
        (from, to, _) => Err(EvalError::unsupported("cast", format!("{from} -> {to}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn narrow_one(value: f32) -> f32 {
        narrow_f32(vec![1], &[value], DType::F16)
            .unwrap()
            .to_f32()
            .unwrap()[0]
    }

    /// The Cast oracle's f16 narrowing keeps NaN (R482-003).
    #[test]
    fn f16_narrowing_keeps_nan_and_rounds_to_nearest() {
        assert!(narrow_one(f32::NAN).is_nan());
        assert!(narrow_one(f32::from_bits(0x7f80_0001)).is_nan());
        let x = 1.0 + 0.75 / 1024.0;
        assert_eq!(
            narrow_one(x),
            1.0 + 1.0 / 1024.0,
            "0.75 ulp rounds up, not toward zero"
        );
    }
}
