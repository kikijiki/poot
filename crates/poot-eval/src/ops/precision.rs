//! Operand precision (POOT-1430): how a contraction's operands are read before the products.
//!
//! [`round_to_fragment`] is the one oracle definition of the narrowing conversion: round to nearest even, ties to
//! even, non-saturating. Every device schedule must reproduce it bit for bit (the tier-1 storage row), so it is
//! public to the tests that hold a kernel's staged values against it.

use poot_graph_ir::{FragmentDtype, OperandPrecision};
use poot_quant::scalar::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
use poot_tensor::DType;

/// `x` rounded to `dtype` (round to nearest even) and widened back to f32. An f32 beyond the dtype's range becomes
/// infinity, never the largest finite value: a non-finite operand then reaches the result and is a fault there
/// (ADR-0101 decision 4), not a silent clamp.
pub fn round_to_fragment(dtype: FragmentDtype, x: f32) -> f32 {
    match dtype {
        FragmentDtype::Bf16 => bf16_to_f32(f32_to_bf16(x)),
        FragmentDtype::F16 => f16_to_f32(f32_to_f16(x)),
    }
}

/// The terms one operand enters the products as under `precision`, each a full-extent f32 vector: one term for an
/// exact or narrowed operand, two (`hi`, `lo`) for a split f32 operand. An operand already stored in the
/// fragment dtype is exact in it and enters once, whatever the precision. `precision` must not be
/// [`OperandPrecision::Exact`]; an exact operand is read as stored, with no copy.
///
/// `lo` is `rne(x - hi)` for a finite `hi` and `0` for an infinite or NaN one, so an overflowing operand stays one
/// infinity and is not turned into the NaN of `inf - inf`.
pub(crate) fn operand_terms(
    values: &[f32],
    stored: DType,
    precision: OperandPrecision,
) -> Vec<Vec<f32>> {
    let Some(dtype) = precision.fragment_dtype() else {
        unreachable!("an exact operand is read as stored");
    };
    let fragment = match dtype {
        FragmentDtype::Bf16 => DType::BF16,
        FragmentDtype::F16 => DType::F16,
    };
    if stored == fragment {
        return vec![values.to_vec()];
    }
    let hi: Vec<f32> = values
        .iter()
        .map(|&x| round_to_fragment(dtype, x))
        .collect();
    match precision {
        OperandPrecision::Split(_) => {
            let lo = values
                .iter()
                .zip(&hi)
                .map(|(&x, &h)| {
                    if h.is_finite() {
                        round_to_fragment(dtype, x - h)
                    } else {
                        0.0
                    }
                })
                .collect();
            vec![hi, lo]
        }
        _ => vec![hi],
    }
}
