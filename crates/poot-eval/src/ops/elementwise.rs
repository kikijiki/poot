//! Pointwise `Unary`, `Binary` and `Select`, one function per (op, dtype).
//!
//! The I32 word functions are the one definition of I32 pointwise semantics: the dense walk, the
//! exact-I32 lane and both fused replays call them, so no lane restates I32 arithmetic. `Select` is
//! one bitwise definition ([`select_word`]) for every lane type.

use std::sync::Arc;

use poot_graph_ir::{BinOp, UnOp};

use super::{broadcast_source, element_count, row_major_strides};
use crate::operand::{initialized_arc_slice, try_initialized_arc_slice};
use crate::{EvalError, broadcast_flat, unravel};
use poot_tensor::HostTensor;

/// F32 `Unary`. I32-only ops never reach it: the walks route an I32 output to [`unary_i32`]. Reads
/// through [`HostTensor::to_f32`], so a BF16/F16 operand widens exactly.
pub(crate) fn unary_f32(op: UnOp, x: &HostTensor) -> Result<HostTensor, EvalError> {
    let data: Vec<f32> = x
        .to_f32()?
        .iter()
        .map(|&v| match op {
            UnOp::Neg => -v,
            UnOp::Exp => v.exp(),
            UnOp::Log => v.ln(),
            UnOp::Sqrt => v.sqrt(),
            UnOp::Recip => 1.0 / v,
            UnOp::Round => v.round(),
            // True libm tanh/erf, independent of the devices' expansions, so GPU==CPU rows check device
            // accuracy.
            UnOp::Tanh => libm::tanhf(v),
            UnOp::Erf => libm::erff(v),
            UnOp::Not | UnOp::Clz => unreachable!("I32-only unary ops are evaluated through ints"),
        })
        .collect();
    Ok(HostTensor::f32(x.shape().to_vec(), data))
}

/// The one definition of the F32 `Max` combiner, shared by `BinOp::Max` and `RedOp::Max`: max-number
/// (`BinOp::Max`'s documented rule, `llvm.maxnum` in the LLVM-based lowerings): a NaN operand yields the other
/// operand, and only two NaNs yield NaN.
pub(crate) fn max_number(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// F32 `Binary` with numpy broadcasting to `out_shape`. Reads through [`HostTensor::to_f32`], so a
/// BF16/F16 operand widens exactly.
pub(crate) fn binary_f32(
    op: BinOp,
    a: &HostTensor,
    b: &HostTensor,
    out_shape: &[usize],
) -> Result<HostTensor, EvalError> {
    let (a_view, b_view) = (a.to_f32()?, b.to_f32()?);
    let n: usize = out_shape.iter().product();
    let mut data = vec![0.0f32; n];
    for (flat, slot) in data.iter_mut().enumerate() {
        let idx = unravel(flat, out_shape);
        let av = a_view[broadcast_flat(&idx, a.shape())];
        let bv = b_view[broadcast_flat(&idx, b.shape())];
        *slot = match op {
            BinOp::Add => av + bv,
            BinOp::Sub => av - bv,
            BinOp::Mul => av * bv,
            BinOp::Div => av / bv,
            BinOp::Max => max_number(av, bv),
            BinOp::Ge => (av >= bv) as i32 as f32,
            BinOp::GeU
            | BinOp::RemU
            | BinOp::And
            | BinOp::Or
            | BinOp::Xor
            | BinOp::Shl
            | BinOp::Shr => {
                unreachable!("I32-only binary ops are evaluated through ints")
            }
        };
    }
    Ok(HostTensor::f32(out_shape.to_vec(), data))
}

/// A row-major I32 operand: its shape and its authoritative words (never an f32 mirror, which loses
/// word patterns above 2^24).
#[derive(Clone, Copy)]
pub(crate) struct I32Operand<'a> {
    pub shape: &'a [usize],
    pub words: &'a [i32],
}

/// The I32 `Binary` word. Arithmetic is the modulo-2^32 contract of plain LLVM integer add/sub/mul;
/// `Ge`/`GeU` are signed and unsigned ordering as a 0/1 word; `RemU` is unsigned remainder and fails
/// closed on a zero divisor; shifts use the low five bits of the shift amount. `Div` has no total
/// exact semantics and is refused.
pub(crate) fn binary_i32_word(op: BinOp, left: i32, right: i32) -> Result<i32, EvalError> {
    Ok(match op {
        BinOp::Add => left.wrapping_add(right),
        BinOp::Sub => left.wrapping_sub(right),
        BinOp::Mul => left.wrapping_mul(right),
        BinOp::Max => left.max(right),
        BinOp::Ge => i32::from(left >= right),
        BinOp::GeU => i32::from((left as u32) >= (right as u32)),
        BinOp::RemU => {
            // Fail closed on a zero divisor: unsigned remainder has no defined value there, and the
            // graph contract for hash moduli is that they are fixed nonzero primes.
            let divisor = right as u32;
            if divisor == 0 {
                return Err(EvalError::unsupported(
                    "binary",
                    "I32 Binary(RemU) divide by zero: the divisor lane is the u32 value 0",
                ));
            }
            ((left as u32) % divisor) as i32
        }
        BinOp::And => left & right,
        BinOp::Or => left | right,
        BinOp::Xor => left ^ right,
        BinOp::Shl => ((left as u32) << ((right as u32) & 31)) as i32,
        BinOp::Shr => ((left as u32) >> ((right as u32) & 31)) as i32,
        BinOp::Div => return Err(i32_div_refused()),
    })
}

/// Refuse an I32 `Binary` op that has no total word semantics (`Div`) before any operand is read, so
/// the refusal does not depend on the operands (an empty or malformed operand is refused the same way).
pub(crate) fn check_binary_i32(op: BinOp) -> Result<(), EvalError> {
    match op {
        BinOp::Div => Err(i32_div_refused()),
        _ => Ok(()),
    }
}

fn i32_div_refused() -> EvalError {
    EvalError::unsupported("binary", "I32 Binary(Div) has no total exact semantics")
}

/// The I32 `Unary` word: bitwise `Not` and `Clz` (leading zeros of the u32 pattern). Float unaries
/// have no I32 meaning and are refused.
pub(crate) fn unary_i32_word(op: UnOp, value: i32) -> Result<i32, EvalError> {
    match op {
        UnOp::Not => Ok(!value),
        UnOp::Clz => Ok((value as u32).leading_zeros() as i32),
        _ => Err(EvalError::unsupported(
            "unary",
            format!("I32 Unary({op:?}) has no integer semantics"),
        )),
    }
}

/// A `Select` lane: a 32-bit-or-narrower word the select moves by bit pattern. The select never
/// reads the value as a number, so a NaN payload, an infinity or `-0.0` passes through unchanged.
pub(crate) trait SelectLane: Copy {
    fn to_bits(self) -> u32;
    fn from_bits(bits: u32) -> Self;
    /// The tensor's words in this lane type, `None` when it stores another.
    fn lanes_of(tensor: &HostTensor) -> Option<&[Self]>;
}

impl SelectLane for i32 {
    fn to_bits(self) -> u32 {
        self as u32
    }
    fn from_bits(bits: u32) -> Self {
        bits as i32
    }
    fn lanes_of(tensor: &HostTensor) -> Option<&[Self]> {
        tensor.as_i32()
    }
}

impl SelectLane for f32 {
    fn to_bits(self) -> u32 {
        f32::to_bits(self)
    }
    fn from_bits(bits: u32) -> Self {
        f32::from_bits(bits)
    }
    fn lanes_of(tensor: &HostTensor) -> Option<&[Self]> {
        tensor.as_f32()
    }
}

impl SelectLane for u16 {
    fn to_bits(self) -> u32 {
        u32::from(self)
    }
    fn from_bits(bits: u32) -> Self {
        bits as u16
    }
    fn lanes_of(tensor: &HostTensor) -> Option<&[Self]> {
        tensor.as_half()
    }
}

/// The `Select` word, the one definition of `OpKind::Select`: the bit pattern of `if_true` where the
/// condition word is nonzero, else of `if_false`, as `(t & mask) | (f & !mask)` with an all-ones mask.
pub(crate) fn select_word<W: SelectLane>(condition: i32, if_true: W, if_false: W) -> W {
    let mask = 0u32.wrapping_sub(u32::from(condition != 0));
    W::from_bits((if_true.to_bits() & mask) | (if_false.to_bits() & !mask))
}

/// I32 `Binary` with numpy broadcasting to `out_shape`, word by word through [`binary_i32_word`].
/// Fills the output `Arc` in place (554c/554d): no intermediate `Vec` copied into
/// it afterward.
pub(crate) fn binary_i32(
    op: BinOp,
    left: I32Operand<'_>,
    right: I32Operand<'_>,
    out_shape: &[usize],
) -> Result<Arc<[i32]>, EvalError> {
    check_binary_i32(op)?;
    let count = element_count(out_shape)?;
    let out_strides = row_major_strides(out_shape)?;
    let left_strides = row_major_strides(left.shape)?;
    let right_strides = row_major_strides(right.shape)?;
    try_initialized_arc_slice(count, |flat| {
        binary_i32_word(
            op,
            left.words[broadcast_source(flat, out_shape, &out_strides, left.shape, &left_strides)],
            right.words
                [broadcast_source(flat, out_shape, &out_strides, right.shape, &right_strides)],
        )
    })
}

/// I32 `Unary`, word by word through [`unary_i32_word`]; the output has the operand's shape.
pub(crate) fn unary_i32(op: UnOp, x: I32Operand<'_>) -> Result<Arc<[i32]>, EvalError> {
    try_initialized_arc_slice(x.words.len(), |index| unary_i32_word(op, x.words[index]))
}

/// A row-major `Select` branch: its shape and its words in the lane type `W`.
#[derive(Clone, Copy)]
pub(crate) struct LaneOperand<'a, W> {
    pub shape: &'a [usize],
    pub words: &'a [W],
}

/// `Select` with numpy broadcasting to `out_shape`, word by word through [`select_word`].
pub(crate) fn select<W: SelectLane>(
    condition: I32Operand<'_>,
    if_true: LaneOperand<'_, W>,
    if_false: LaneOperand<'_, W>,
    out_shape: &[usize],
) -> Result<Arc<[W]>, EvalError> {
    let count = element_count(out_shape)?;
    let out_strides = row_major_strides(out_shape)?;
    // Strides are per operand, not per element: hoist them out of the element loop.
    let strides = [
        row_major_strides(condition.shape)?,
        row_major_strides(if_true.shape)?,
        row_major_strides(if_false.shape)?,
    ];
    Ok(initialized_arc_slice(count, |flat| {
        let at = |shape: &[usize], strides: &[usize]| {
            broadcast_source(flat, out_shape, &out_strides, shape, strides)
        };
        select_word(
            condition.words[at(condition.shape, &strides[0])],
            if_true.words[at(if_true.shape, &strides[1])],
            if_false.words[at(if_false.shape, &strides[2])],
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The float `Max` rule of `BinOp::Max` is max-number: `Max(NaN, 1)` is 1 in either operand order, and
    /// only two NaNs give NaN.
    ///
    /// Mutation: make `max_number` return NaN when either operand is NaN; the first two cells come back NaN
    /// and this goes red on `Max(NaN, 1) must be 1`.
    #[test]
    fn max_ignores_a_nan_operand_and_is_nan_only_for_two_nans() {
        let a = HostTensor::f32(vec![4], vec![f32::NAN, 1.0, f32::NAN, 2.0]);
        let b = HostTensor::f32(vec![4], vec![1.0, f32::NAN, f32::NAN, 1.0]);
        let out = binary_f32(BinOp::Max, &a, &b, &[4]).unwrap();
        let out = out.as_f32().unwrap();
        assert_eq!(out[0], 1.0, "Max(NaN, 1) must be 1, got {}", out[0]);
        assert_eq!(out[1], 1.0, "Max(1, NaN) must be 1, got {}", out[1]);
        assert!(out[2].is_nan(), "Max(NaN, NaN) must be NaN, got {}", out[2]);
        assert_eq!(out[3], 2.0);
    }
}
