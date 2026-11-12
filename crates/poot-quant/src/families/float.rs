//! The float family: the code is a float, and a NaN is not a value.

/// Whether a decoded float is a value: a stored NaN is a decode error, never a weight.
pub(crate) fn is_value(number: f32) -> bool {
    !number.is_nan()
}
