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
            c.id,
            Value::from(HostTensor::i32(shape.to_vec(), cond.to_vec())),
        ),
        (t.id, Value::from(floats(if_true))),
        (f.id, Value::from(floats(if_false))),
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
        (c.id, Value::from(HostTensor::i32(vec![2, 1], vec![1, 0]))),
        (
            t.id,
            Value::from(HostTensor::f32(
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )),
        ),
        (
            f.id,
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
            c.id,
            Value::from(HostTensor::i32(vec![4], vec![1, 0, 1, 0])),
        ),
        (
            t.id,
            Value::from(HostTensor::bf16(
                vec![4],
                vec![0x7FC1, 0x3F80, 0x8000, 0x7F80],
            )),
        ),
        (
            f.id,
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
