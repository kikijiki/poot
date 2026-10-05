use super::*;

#[derive(Clone, Copy)]
pub(crate) struct U64Words {
    pub(crate) low: Traced,
    pub(crate) high: Traced,
}

/// Limb slices kept as `[..., 1]` so a later last-axis Concat can pack them without a reshape barrier.
pub(crate) fn counter_lanes(b: &Builder, value: Traced) -> U64Words {
    let axis = counter_prefix(b, value).len();
    U64Words {
        low: b.slice(value, axis, 0, 1),
        high: b.slice(value, axis, 1, 2),
    }
}

pub(crate) fn pack_i32_lanes(b: &Builder, parts: &[Traced]) -> Traced {
    let axis = b.aval(parts[0]).shape.len() - 1;
    b.concat(axis, parts)
}

pub(crate) fn zero_like(b: &Builder, value: Traced) -> Traced {
    b.binary_scalar(BinOp::And, value, Scalar::I32(0))
}

pub(crate) fn i32_lit(b: &Builder, like: Traced, value: i32) -> Traced {
    b.binary_scalar(BinOp::Add, zero_like(b, like), Scalar::I32(value))
}

pub(crate) fn counter_prefix(b: &Builder, value: Traced) -> Vec<usize> {
    let ty = b.aval(value);
    assert_eq!(ty.dtype, DType::I32, "u64 counter limbs must be I32");
    assert_eq!(
        ty.shape.last(),
        Some(&2),
        "u64 counter shape must end in [low,high]"
    );
    ty.shape[..ty.shape.len() - 1].to_vec()
}
