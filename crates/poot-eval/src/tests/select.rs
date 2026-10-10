//! POOT-1033: `Select` is a bitwise select at every dtype, never `if_false + cond * (if_true - if_false)`.
//!
//! The expectations are hand-written bit patterns. A NaN, an infinity or `-0.0` in the selected branch
//! must come out with its exact bits, and the same value in the unselected branch must not reach the
//! output at all (`0 * NaN` and `0 * Inf` are NaN, which is what the arithmetic form produced).

use std::collections::HashMap;

use poot_graph_ir::{Builder, TensorType};
use poot_tensor::{DType, HostTensor};

use crate::{EvalBudget, EvalOptions, Value, eval};

const QUIET_NAN_PAYLOAD: u32 = 0x7FC0_1234;
const SIGNALING_NAN: u32 = 0x7F80_0001;
const NEGATIVE_NAN_PAYLOAD: u32 = 0xFFC0_0042;
const POSITIVE_INFINITY: u32 = 0x7F80_0000;
const NEGATIVE_INFINITY: u32 = 0xFF80_0000;
const NEGATIVE_ZERO: u32 = 0x8000_0000;

fn select_f32_bits(shape: &[usize], cond: &[i32], if_true: &[u32], if_false: &[u32]) -> Vec<u32> {
    let b = Builder::new();
    let ty = |dtype| TensorType::new(shape.to_vec(), dtype);
    let c = b.constant("cond", ty(DType::I32));
    let t = b.constant("if_true", ty(DType::F32));
    let f = b.constant("if_false", ty(DType::F32));
    let out = b.select(c, t, f);
    let g = b.finish(out);
    let floats = |bits: &[u32]| {
        HostTensor::f32(
            shape.to_vec(),
            bits.iter().map(|&w| f32::from_bits(w)).collect(),
        )
    };
    let inputs = HashMap::from([
        (
            c.id(),
            Value::from(HostTensor::i32(shape.to_vec(), cond.to_vec())),
        ),
        (t.id(), Value::from(floats(if_true))),
        (f.id(), Value::from(floats(if_false))),
    ]);
    let out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    out.as_f32().unwrap().iter().map(|v| v.to_bits()).collect()
}

/// The selected branch passes through bit for bit, payloads and signs included, for a condition of 1, 2
/// and -1 (any nonzero word selects `if_true`) and 0 (selects `if_false`).
#[test]
fn float_select_passes_the_selected_branch_bits_through() {
    let got = select_f32_bits(
        &[6],
        &[1, 0, 2, -1, 0, 1],
        &[
            QUIET_NAN_PAYLOAD,
            1.0f32.to_bits(),
            POSITIVE_INFINITY,
            NEGATIVE_ZERO,
            5.0f32.to_bits(),
            NEGATIVE_INFINITY,
        ],
        &[
            3.0f32.to_bits(),
            SIGNALING_NAN,
            7.0f32.to_bits(),
            NEGATIVE_NAN_PAYLOAD,
            NEGATIVE_ZERO,
            QUIET_NAN_PAYLOAD,
        ],
    );
    assert_eq!(
        got,
        [
            QUIET_NAN_PAYLOAD,
            SIGNALING_NAN,
            POSITIVE_INFINITY,
            NEGATIVE_ZERO,
            NEGATIVE_ZERO,
            NEGATIVE_INFINITY
        ],
        "got {got:#010x?}"
    );
}

/// A NaN or an infinity in the unselected branch never reaches the output: the lanes below select a finite
/// branch while the other one is poisoned.
#[test]
fn float_select_keeps_a_poisoned_unselected_branch_out_of_the_result() {
    let got = select_f32_bits(
        &[4],
        &[1, 0, 1, 0],
        &[
            2.0f32.to_bits(),
            QUIET_NAN_PAYLOAD,
            4.0f32.to_bits(),
            POSITIVE_INFINITY,
        ],
        &[
            QUIET_NAN_PAYLOAD,
            3.0f32.to_bits(),
            NEGATIVE_INFINITY,
            6.0f32.to_bits(),
        ],
    );
    assert_eq!(
        got,
        [
            2.0f32.to_bits(),
            3.0f32.to_bits(),
            4.0f32.to_bits(),
            6.0f32.to_bits()
        ],
        "got {got:#010x?}"
    );
}

/// A condition of `[2, 1]` broadcasts across `[2, 3]` branches.
#[test]
fn float_select_broadcasts_the_condition() {
    let b = Builder::new();
    let c = b.constant("cond", TensorType::new(vec![2, 1], DType::I32));
    let t = b.constant("if_true", TensorType::new(vec![2, 3], DType::F32));
    let f = b.constant("if_false", TensorType::new(vec![3], DType::F32));
    let out = b.select(c, t, f);
    let g = b.finish(out);
    let inputs = HashMap::from([
        (c.id(), Value::from(HostTensor::i32(vec![2, 1], vec![1, 0]))),
        (
            t.id(),
            Value::from(HostTensor::f32(
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )),
        ),
        (
            f.id(),
            Value::from(HostTensor::f32(
                vec![3],
                vec![f32::NAN, f32::INFINITY, -0.0],
            )),
        ),
    ]);
    let out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let bits: Vec<u32> = out.as_f32().unwrap().iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        bits,
        [
            1.0f32.to_bits(),
            2.0f32.to_bits(),
            3.0f32.to_bits(),
            f32::NAN.to_bits(),
            f32::INFINITY.to_bits(),
            NEGATIVE_ZERO
        ],
        "got {bits:#010x?}"
    );
}

/// The half-width dtypes select by their own bit patterns too: a BF16 NaN payload and `-0.0` come out
/// exactly, and the unselected NaN stays out.
#[test]
fn bf16_select_passes_half_width_bits_through() {
    let b = Builder::new();
    let ty = |dtype| TensorType::new(vec![4], dtype);
    let c = b.constant("cond", ty(DType::I32));
    let t = b.constant("if_true", ty(DType::BF16));
    let f = b.constant("if_false", ty(DType::BF16));
    let out = b.select(c, t, f);
    let g = b.finish(out);
    let inputs = HashMap::from([
        (
            c.id(),
            Value::from(HostTensor::i32(vec![4], vec![1, 0, 1, 0])),
        ),
        (
            t.id(),
            Value::from(HostTensor::bf16(
                vec![4],
                vec![0x7FC1, 0x3F80, 0x8000, 0x7F80],
            )),
        ),
        (
            f.id(),
            Value::from(HostTensor::bf16(
                vec![4],
                vec![0x4000, 0x7FC2, 0x7F80, 0xFFC3],
            )),
        ),
    ]);
    let out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.as_half().unwrap(), [0x7FC1, 0x7FC2, 0x8000, 0xFFC3]);
}

/// The half-width bit patterns a select must carry unchanged: payload NaNs of both signs, a signaling NaN
/// (low mantissa bit only), the infinities and `-0.0`, in the BF16 or the F16 layout.
fn half_patterns(dtype: DType) -> [u16; 6] {
    match dtype {
        DType::BF16 => [0x7FC1, 0xFFC3, 0x7F81, 0x7F80, 0xFF80, 0x8000],
        _ => [0x7E01, 0xFE03, 0x7C01, 0x7C00, 0xFC00, 0x8000],
    }
}

fn half_tensor(dtype: DType, shape: Vec<usize>, words: Vec<u16>) -> HostTensor {
    match dtype {
        DType::BF16 => HostTensor::bf16(shape, words),
        _ => HostTensor::f16(shape, words),
    }
}

/// POOT-1041: `Select` over two-byte floats is the same bit-moving select as the F32 and I32 forms, for a
/// branch that comes straight from an input AND for one that went through the movement ops that carry an
/// owner-backed view (reshape, transpose) or a materialized copy (broadcast, slice, concat). The expectations
/// are hand-written words: the signaling NaN keeps its payload (a trip through F32 would quiet it) and the
/// unselected branch's NaN stays out. Mutation: read the movement operands through `get` instead of `stored`
/// in `evaluate_dense`; the BF16 concat refuses mixed lanes.
#[test]
fn narrow_select_moves_exact_words_through_movement_ops() {
    for dtype in [DType::BF16, DType::F16] {
        let words = half_patterns(dtype);
        let b = Builder::new();
        let ty = |shape: &[usize], dtype| TensorType::new(shape.to_vec(), dtype);
        let cond = b.constant("cond", ty(&[6], DType::I32));
        let on = b.constant("if_true", ty(&[6], dtype));
        let off = b.constant("if_false", ty(&[6], dtype));
        // The true branch: the words reshaped to [2,3] and transposed twice (owner-backed views), then split
        // along the last axis and re-joined.
        let moved = b.transpose(
            b.transpose(b.reshape(on, vec![2, 3]), vec![1, 0]),
            vec![1, 0],
        );
        let joined = b.reshape(
            b.concat(1, &[b.slice(moved, 1, 0, 1), b.slice(moved, 1, 1, 3)]),
            vec![6],
        );
        // The false branch: a broadcast of the same words from a leading axis, reshaped back.
        let widened = b.reshape(
            b.broadcast(b.reshape(off, vec![1, 6]), vec![2, 6]),
            vec![12],
        );
        let other = b.slice(widened, 0, 6, 12);
        let out = b.select(cond, joined, other);
        let g = b.finish(out);
        let inputs = HashMap::from([
            (
                cond.id(),
                Value::from(HostTensor::i32(vec![6], vec![1, 0, 1, 0, 1, 0])),
            ),
            (
                on.id(),
                Value::from(half_tensor(dtype, vec![6], words.to_vec())),
            ),
            (
                off.id(),
                Value::from(half_tensor(
                    dtype,
                    vec![6],
                    words.iter().rev().copied().collect(),
                )),
            ),
        ]);
        let out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();
        assert_eq!(out.dtype(), dtype);
        assert_eq!(
            out.as_half().unwrap(),
            [words[0], words[4], words[2], words[2], words[4], words[0]],
            "{dtype:?}"
        );
    }
}
