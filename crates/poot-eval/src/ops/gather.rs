//! `Gather`: one generic function over `T: Copy`, shared by every data carrier (the dense f32/i32
//! lanes, the exact-I32 word lane, the owner-backed lazy checkpoint read, the Spec-376 BF16 table and
//! E4M3FN), and one index rule ([`index_at`]) shared with every other index-consuming op.
//!
//! `out = data[..axis] ++ index.shape ++ data[axis+1..]`. A scalar index drops the axis
//! (embedding/RoPE-row lookup); a vector index `[L]` replaces the axis with `L` (a whole prompt's
//! embeddings).

use poot_tensor::{DType, HostData, HostTensor};
use std::sync::Arc;

use poot_graph_ir::ValueId;

use super::index_rule::{IndexValue, index_at};
use super::{element_count, row_major_strides};
use crate::exact_bf16::ExactBf16TensorView;
use crate::exact_value::ExactValue;
use crate::operand::initialized_arc_slice;
use crate::{EvalError, gather_source_flat};

/// `out[flat] = data_at(source)`, `source` the row-major element of a conceptual tensor of
/// `data_shape` that `index` selects. Both `data_at` and `index_at_pos` are read lazily, once per
/// *output*/*index* element respectively, never once per source element: an owner-backed source far
/// larger than the gathered rows (a checkpoint's embedding table) is never materialized just to
/// select a handful of them (card 396), and `index_at_pos` reads straight from the caller's own I32 or
/// F32 slice - no intermediate `Vec<IndexValue>` the size of `index` (card 554c
/// allocation bound). Every index value is checked against `data_shape[axis]` through [`index_at`]
/// before any read.
pub(crate) fn gather<T: Copy>(
    data_at: impl Fn(usize) -> T,
    data_shape: &[usize],
    index_at_pos: impl Fn(usize) -> IndexValue,
    index_len: usize,
    index_shape: &[usize],
    axis: usize,
    eqn: ValueId,
) -> Result<Arc<[T]>, EvalError> {
    let len = data_shape[axis];
    for position in 0..index_len {
        index_at(index_at_pos(position), position, len, eqn)?;
    }
    let data_strides = row_major_strides(data_shape)?;
    let index_strides = row_major_strides(index_shape)?;
    let mut out_shape = data_shape[..axis].to_vec();
    out_shape.extend_from_slice(index_shape);
    out_shape.extend_from_slice(&data_shape[axis + 1..]);
    let output_count = element_count(&out_shape)?;
    Ok(initialized_arc_slice(output_count, |flat| {
        let source = gather_source_flat(
            flat,
            &out_shape,
            &data_strides,
            &index_strides,
            axis,
            |position| {
                index_at(index_at_pos(position), position, len, eqn)
                    .expect("already validated above")
            },
        );
        data_at(source)
    }))
}

/// An index operand's element count and its by-position reader (see [`index_reader`]).
pub(crate) type IndexReader<'a> = (usize, Box<dyn Fn(usize) -> IndexValue + 'a>);

/// An index operand read lazily from its own typed words: its element count and a reader of one
/// [`IndexValue`] by position (I32 words stay exact, F32 values keep their own fault kinds). Any
/// other dtype is refused.
pub(crate) fn index_reader(index: &HostTensor) -> Result<IndexReader<'_>, EvalError> {
    match index.data() {
        HostData::I32(words) => Ok((
            words.len(),
            Box::new(move |p: usize| IndexValue::I32(words[p])),
        )),
        HostData::F32(values) => Ok((
            values.len(),
            Box::new(move |p: usize| IndexValue::F32(values[p])),
        )),
        _ => Err(EvalError::unsupported(
            "gather",
            format!("an index operand is F32 or I32, got {}", index.dtype()),
        )),
    }
}

/// [`gather`] over a host tensor of any dtype, in its own storage class: the output keeps the
/// table's dtype (a BF16 table yields BF16 words, an E4M3FN table E4M3FN bytes).
pub(crate) fn gather_host(
    table: &HostTensor,
    index_at_pos: impl Fn(usize) -> IndexValue,
    index_len: usize,
    index_shape: &[usize],
    axis: usize,
    eqn: ValueId,
) -> Result<HostTensor, EvalError> {
    let mut out_shape = table.shape()[..axis].to_vec();
    out_shape.extend_from_slice(index_shape);
    out_shape.extend_from_slice(&table.shape()[axis + 1..]);
    let shape = table.shape();
    let data = match table.data() {
        HostData::F32(d) => HostData::F32(gather(
            |i| d[i],
            shape,
            index_at_pos,
            index_len,
            index_shape,
            axis,
            eqn,
        )?),
        HostData::I32(d) => HostData::I32(gather(
            |i| d[i],
            shape,
            index_at_pos,
            index_len,
            index_shape,
            axis,
            eqn,
        )?),
        HostData::Half(d) => HostData::Half(gather(
            |i| d[i],
            shape,
            index_at_pos,
            index_len,
            index_shape,
            axis,
            eqn,
        )?),
        HostData::Bytes(d) => HostData::Bytes(gather(
            |i| d[i],
            shape,
            index_at_pos,
            index_len,
            index_shape,
            axis,
            eqn,
        )?),
    };
    Ok(HostTensor::new(table.dtype(), out_shape, data)?)
}

impl crate::Value {
    /// The `Gather` entry point outside the walk's own equation loop (tests, and any future caller
    /// with no equation id in scope): every carrier `OpKind::Gather` admits except [`ExactValue`],
    /// whose row-selecting lanes only exist inside the walk's own I32-index dispatch
    /// (`walk::evaluate_gather`). `eqn` reads as `0` (the `EvalError::unsupported` convention).
    ///
    /// `index` is read through its own typed words: I32 words route through the exact `IndexValue`
    /// path (an id above 2^24 is never rounded through f32, card 363's E4M3 PLE n-gram ids), F32
    /// values through the f32 fault kinds.
    pub fn gather(&self, axis: usize, index: &HostTensor) -> Result<Self, EvalError> {
        const NO_EQN: ValueId = 0;
        match self {
            crate::Value::Host(table) => {
                let (index_len, index_at_pos) = index_reader(index)?;
                Ok(
                    gather_host(table, index_at_pos, index_len, index.shape(), axis, NO_EQN)?
                        .into(),
                )
            }
            crate::Value::Owner(value) => Err(value.unsupported("Gather")),
            crate::Value::Packed(_) => {
                Err(EvalError::Packed(crate::PackedEvalError::WrongConsumer {
                    operation: "Gather",
                }))
            }
        }
    }
}

/// [`gather`] over an owner-backed exact source (an F32 or BF16 checkpoint tensor read lazily through
/// its zero-copy view): the counterpart that keeps card 396's "never materialize the whole source"
/// property for the exact-I32 lane's large embedding-table reads.
pub(crate) fn gather_owner(
    source: &ExactValue,
    index_at_pos: impl Fn(usize) -> IndexValue,
    index_len: usize,
    index_shape: &[usize],
    axis: usize,
    eqn: ValueId,
) -> Result<Arc<[f32]>, EvalError> {
    match source {
        ExactValue::Bf16(view) => gather(
            |flat| view.value(flat),
            view.shape(),
            index_at_pos,
            index_len,
            index_shape,
            axis,
            eqn,
        ),
        ExactValue::Dense(view) if view.dtype() == DType::BF16 => {
            // Cheap alias, not a decode: `from_owner_view` only wraps the same `Arc<ExactSourceOwner>`.
            let bf16 = ExactBf16TensorView::from_owner_view(view).expect(
                "DenseOwnerTensorView::new only ever pairs dtype BF16 with an owner of \
                 ExactSourceKind::Bf16",
            );
            gather(
                |flat| bf16.value(flat),
                view.shape(),
                index_at_pos,
                index_len,
                index_shape,
                axis,
                eqn,
            )
        }
        ExactValue::Dense(view) => {
            debug_assert_eq!(
                view.dtype(),
                DType::F32,
                "DenseOwnerTensorView::new only admits BF16 or F32"
            );
            let bytes = view.bytes();
            gather(
                |flat| {
                    let start = flat * 4;
                    f32::from_le_bytes(bytes[start..start + 4].try_into().expect(
                        "DenseOwnerTensorView::new already validated byte length against numel",
                    ))
                },
                view.shape(),
                index_at_pos,
                index_len,
                index_shape,
                axis,
                eqn,
            )
        }
        ExactValue::I32(_) => {
            unreachable!("gather_owner is never called for ExactValue::I32")
        }
    }
}
