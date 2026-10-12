use super::{BinOp, Builder, BuilderError, Scalar, Traced, word_pair_prefix};

#[derive(Clone, Copy)]
pub(crate) struct U64Words {
    pub(crate) low: Traced,
    pub(crate) high: Traced,
}

/// Limb slices kept as `[..., 1]` so a later last-axis Concat can pack them without a reshape barrier.
pub(crate) fn counter_lanes(b: &Builder, value: Traced) -> Result<U64Words, BuilderError> {
    let axis = counter_prefix(b, value)?.len();
    Ok(U64Words {
        low: b.slice(value, axis, 0, 1)?,
        high: b.slice(value, axis, 1, 2)?,
    })
}

pub(crate) fn pack_i32_lanes(b: &Builder, parts: &[Traced]) -> Result<Traced, BuilderError> {
    let axis = b.aval(parts[0])?.shape.len() - 1;
    b.concat(axis, parts)
}

pub(crate) fn zero_like(b: &Builder, value: Traced) -> Result<Traced, BuilderError> {
    b.binary_scalar(BinOp::And, value, Scalar::I32(0))
}

pub(crate) fn i32_lit(b: &Builder, like: Traced, value: i32) -> Result<Traced, BuilderError> {
    b.binary_scalar(BinOp::Add, zero_like(b, like)?, Scalar::I32(value))
}

pub(crate) fn counter_prefix(b: &Builder, value: Traced) -> Result<Vec<usize>, BuilderError> {
    let ty = b.aval(value)?;
    word_pair_prefix("u64 counter", &ty)
}
