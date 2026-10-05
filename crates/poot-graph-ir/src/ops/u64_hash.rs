//! Little-endian two-limb u64 helpers for hash-style compositions (sibling of the counter ops).
//!
//! Values are I32 `[..., 2]` bit patterns, the same representation the counter ops use. Limb
//! intermediates are kept as `[..., 1]` lanes so a later last-axis `Concat` packs them without a
//! reshape barrier (same convention as `counter_lanes`). Multiplies split through 16-bit halves so
//! every multiplication operand is under 2^16. The remainder path is Horner in base 16: each step is
//! `(r * 16 + nibble) % modulus` with `r < modulus`, so intermediates stay below `16 * modulus`. A
//! modulus under 2^28 therefore never overflows a u32 lane (checked by [`u64_rem_u32`]).

use super::counter::{U64Words, counter_lanes, i32_lit, pack_i32_lanes, zero_like};
use super::*;

/// Why a u64-hash graph cannot be built for the requested modulus.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum U64HashError {
    #[error(
        "u64_rem_u32 modulus must be in 1..2^28 so Horner intermediates stay below 2^32; got {modulus}"
    )]
    ModulusOutOfRange { modulus: u32 },
}

/// Widen an I32 lane of shape `S` to `[low, high]` lanes of shape `S ++ [1]` (zero-extended u32 ->
/// u64). A scalar becomes `[1]`.
fn u64_lanes_from_i32(b: &Builder, token: Traced) -> U64Words {
    let ty = b.aval(token);
    assert_eq!(ty.dtype, DType::I32, "u64_lanes_from_i32 needs an I32 lane");
    let low = if ty.shape.is_empty() {
        b.reshape(token, vec![1])
    } else {
        let mut s = ty.shape.clone();
        s.push(1);
        b.reshape(token, s)
    };
    U64Words {
        low,
        high: zero_like(b, low),
    }
}

/// Split a packed `[..., 2]` value into `[low, high]` lanes of shape `[..., 1]`.
fn unpack(b: &Builder, value: Traced) -> U64Words {
    counter_lanes(b, value)
}

/// Pack `[low, high]` lanes of shape `[..., 1]` back to `[..., 2]`.
fn pack(b: &Builder, value: U64Words) -> Traced {
    pack_i32_lanes(b, &[value.low, value.high])
}

/// One 32x32 -> 64 unsigned product of I32 storage words shaped `[..., 1]`, split through 16-bit
/// halves so every multiplication operand is under 2^16 and every product fits in a u32 lane.
fn u32_mul_wide(b: &Builder, left: Traced, right: Traced) -> U64Words {
    let l0 = b.binary_scalar(BinOp::And, left, Scalar::I32(0xffff));
    let l1 = b.binary_scalar(BinOp::Shr, left, Scalar::I32(16));
    let r0 = b.binary_scalar(BinOp::And, right, Scalar::I32(0xffff));
    let r1 = b.binary_scalar(BinOp::Shr, right, Scalar::I32(16));
    let p00 = b.binary(BinOp::Mul, l0, r0);
    let p01 = b.binary(BinOp::Mul, l0, r1);
    let p10 = b.binary(BinOp::Mul, l1, r0);
    let p11 = b.binary(BinOp::Mul, l1, r1);
    // mid = (p00 >> 16) + (p01 & 0xffff) + (p10 & 0xffff), at most 3*(2^16-1) < 2^18.
    let mid = b.binary_scalar(BinOp::Shr, p00, Scalar::I32(16));
    let mid = b.binary(
        BinOp::Add,
        mid,
        b.binary_scalar(BinOp::And, p01, Scalar::I32(0xffff)),
    );
    let mid = b.binary(
        BinOp::Add,
        mid,
        b.binary_scalar(BinOp::And, p10, Scalar::I32(0xffff)),
    );
    let low = b.binary(
        BinOp::Or,
        b.binary_scalar(BinOp::And, p00, Scalar::I32(0xffff)),
        b.binary_scalar(BinOp::Shl, mid, Scalar::I32(16)),
    );
    let carry = b.binary_scalar(BinOp::Shr, mid, Scalar::I32(16));
    let high = b.binary(
        BinOp::Add,
        p11,
        b.binary_scalar(BinOp::Shr, p01, Scalar::I32(16)),
    );
    let high = b.binary(
        BinOp::Add,
        high,
        b.binary_scalar(BinOp::Shr, p10, Scalar::I32(16)),
    );
    let high = b.binary(BinOp::Add, high, carry);
    U64Words { low, high }
}

fn u64_mul_const_lanes(b: &Builder, value: U64Words, multiplier: u64) -> U64Words {
    let mult_low = (multiplier as u32) as i32;
    let mult_high = ((multiplier >> 32) as u32) as i32;
    // Schoolbook on 32-bit limbs; only the low 64 product bits are kept (callers keep the
    // mathematical product inside u64).
    let vl = value.low;
    let vh = value.high;
    let ml = i32_lit(b, vl, mult_low);
    let mh = i32_lit(b, vl, mult_high);
    let p00 = u32_mul_wide(b, vl, ml);
    let p01 = u32_mul_wide(b, vl, mh);
    let p10 = u32_mul_wide(b, vh, ml);
    // low64 = p00 + ((p01 + p10) << 32), so low = p00.low, high = p00.high + p01.low + p10.low
    // (wrapping: the overflow past bit 64 is discarded).
    let high = b.binary(BinOp::Add, p00.high, p01.low);
    let high = b.binary(BinOp::Add, high, p10.low);
    U64Words { low: p00.low, high }
}

/// Widen an I32 lane of shape `S` and multiply by a `u64` constant to packed u64 `S ++ [2]`.
pub fn u64_mul_i32_const(b: &Builder, token: Traced, multiplier: u64) -> Traced {
    pack(
        b,
        u64_mul_const_lanes(b, u64_lanes_from_i32(b, token), multiplier),
    )
}

/// Bitwise XOR of two packed little-endian u64 values of matching shape.
pub fn u64_xor(b: &Builder, left: Traced, right: Traced) -> Traced {
    let (l, r) = (unpack(b, left), unpack(b, right));
    pack(
        b,
        U64Words {
            low: b.binary(BinOp::Xor, l.low, r.low),
            high: b.binary(BinOp::Xor, l.high, r.high),
        },
    )
}

/// Horner base-16 `rem_euclid` of a packed little-endian u64 by a nonzero `modulus`, as a
/// non-negative I32 of the prefix shape. Every intermediate is `(r * 16 + nibble)` with
/// `r < modulus`, so `modulus` must be `< 2^28` to keep that sum inside a u32 lane.
pub fn u64_rem_u32(b: &Builder, value: Traced, modulus: u32) -> Result<Traced, U64HashError> {
    if modulus == 0 || modulus >= (1 << 28) {
        return Err(U64HashError::ModulusOutOfRange { modulus });
    }
    let ty = b.aval(value);
    assert_eq!(ty.dtype, DType::I32, "u64_rem_u32 needs an I32 value");
    assert_eq!(
        ty.shape.last(),
        Some(&2),
        "u64_rem_u32 shape must end in [low,high]"
    );
    let prefix = ty.shape[..ty.shape.len() - 1].to_vec();
    let lanes = unpack(b, value);
    let mod_lit = Scalar::I32(modulus as i32);
    // Seed `r` as a prefix-shaped zero (the low lane is zero, reshaped off the trailing unit axis).
    let mut r = b.reshape(zero_like(b, lanes.low), prefix.clone());
    // Most-significant nibble first: high word shifts 28..0, then low word shifts 28..0.
    for word in [lanes.high, lanes.low] {
        for shift in (0..=28u32).rev().step_by(4) {
            let nibble = b.binary_scalar(
                BinOp::And,
                b.binary_scalar(BinOp::Shr, word, Scalar::I32(shift as i32)),
                Scalar::I32(0xf),
            );
            let nibble = b.reshape(nibble, prefix.clone());
            let scaled = b.binary_scalar(BinOp::Mul, r, Scalar::I32(16));
            let step = b.binary(BinOp::Add, scaled, nibble);
            r = b.binary_scalar(BinOp::RemU, step, mod_lit);
        }
    }
    Ok(r)
}
