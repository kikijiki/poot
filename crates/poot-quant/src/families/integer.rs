//! The integer family: the code as an integer, minus the zero point when the format has one.

/// `code - zero` as the f32 the scale multiplies (`Q4_0`'s `q - 8` is the code's own offset, GPTQ's
/// `q - (z + 1)` is the `zero` operand).
pub(crate) const fn value(code: i64, zero: i64) -> f32 {
    (code - zero) as f32
}
