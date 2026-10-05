//! Pointwise `Unary`, `Binary` and `Select`, one function per (op, dtype).
//!
//! The I32 word functions are the one definition of I32 pointwise semantics: the dense walk, the
//! exact-I32 lane and both fused replays call them, so no lane restates I32 arithmetic.

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
            BinOp::Max => av.max(bv),
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

/// The I32 `Select` word: `if_false + condition * (if_true - if_false)` in wrapping arithmetic (the
/// IR's definition of `OpKind::Select`).
pub(crate) fn select_i32_word(condition: i32, if_true: i32, if_false: i32) -> i32 {
    if_false.wrapping_add(condition.wrapping_mul(if_true.wrapping_sub(if_false)))
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

/// I32 `Select` with numpy broadcasting to `out_shape`, word by word through [`select_i32_word`].
pub(crate) fn select_i32(
    condition: I32Operand<'_>,
    if_true: I32Operand<'_>,
    if_false: I32Operand<'_>,
    out_shape: &[usize],
) -> Result<Arc<[i32]>, EvalError> {
    let count = element_count(out_shape)?;
    let out_strides = row_major_strides(out_shape)?;
    let operands = [condition, if_true, if_false];
    let strides = operands
        .iter()
        .map(|operand| row_major_strides(operand.shape))
        .collect::<Result<Vec<_>, _>>()?;
    let word = |flat: usize, which: usize| {
        let operand = operands[which];
        operand.words[broadcast_source(
            flat,
            out_shape,
            &out_strides,
            operand.shape,
            &strides[which],
        )]
    };
    Ok(initialized_arc_slice(count, |flat| {
        select_i32_word(word(flat, 0), word(flat, 1), word(flat, 2))
    }))
}
