//! The CPU oracle's operation semantics: one function per (op, dtype), grouped by op family.
//!
//! Every evaluation walk (the dense walk, the storage-aware walk, the exact-I32 lane and the BF16
//! table) calls these functions instead of restating an op's arithmetic in its own arm, so two walks
//! cannot disagree about what an op computes. A walk still decides which operands
//! it admits and which carrier it publishes; the arithmetic lives here.

pub(crate) mod cast;
pub(crate) mod collective;
pub(crate) mod contraction;
pub(crate) mod elementwise;
pub(crate) mod gather;
pub(crate) mod index;
pub(crate) mod index_rule;
pub(crate) mod movement;
pub(crate) mod packed;
pub(crate) mod reduce;
pub(crate) mod sampling;

pub(crate) use packed::{
    evaluate_packed_contraction, evaluate_packed_dequant, evaluate_packed_row_gather,
    packed_components, packed_row_ids,
};

use poot_tensor::HostTensor;

use crate::EvalError;

/// The exact I32 words of an operand that must be an I32 tensor: the one typed read every op that
/// consumes words shares (never an f32 value reinterpreted).
pub(crate) fn i32_lane<'a>(t: &'a HostTensor, op: &'static str) -> Result<&'a [i32], EvalError> {
    t.as_i32().ok_or_else(|| {
        EvalError::unsupported(op, format!("expected an I32 tensor, got {}", t.dtype()))
    })
}

/// Row-major element count of `shape`, refusing an overflowing product. One lane-neutral helper
/// (554c, 554d): `tensor::checked_numel`, `exact_bf16::checked_numel`
/// and `exact_dense::checked_numel` used to restate this with their own typed overflow, so the dense
/// walk's I32 `Binary`/`Select` shape overflow used to come back mislabeled as one lane's error, with
/// a bogus equation id (`EvalError::Unsupported { eqn: 0, .. }`); the typed, eqn-free
/// `EvalError::ElementCountOverflow` names the real defect instead.
pub(crate) fn element_count(shape: &[usize]) -> Result<usize, EvalError> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| EvalError::ElementCountOverflow {
            shape: shape.to_vec(),
        })
}

/// Row-major strides of `shape`, refusing an overflowing product.
pub(crate) fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, EvalError> {
    let mut reversed = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for extent in shape.iter().rev() {
        reversed.push(stride);
        stride = stride
            .checked_mul(*extent)
            .ok_or_else(|| EvalError::ElementCountOverflow {
                shape: shape.to_vec(),
            })?;
    }
    reversed.reverse();
    Ok(reversed)
}

/// The flat index of a (right-aligned, numpy-broadcast) operand of `shape` that output element `flat`
/// reads; a size-1 axis contributes 0.
pub(crate) fn broadcast_source(
    flat: usize,
    output_shape: &[usize],
    output_strides: &[usize],
    shape: &[usize],
    input_strides: &[usize],
) -> usize {
    let offset = output_shape.len() - shape.len();
    shape
        .iter()
        .enumerate()
        .map(|(axis, extent)| {
            let coordinate = if *extent == 1 {
                0
            } else {
                (flat / output_strides[offset + axis]) % output_shape[offset + axis]
            };
            coordinate * input_strides[axis]
        })
        .sum()
}
