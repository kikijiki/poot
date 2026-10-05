//! Index-producing and index-consuming ops: `Iota`, `ArgTopK`, `PackI8`, `UnpackI8`, and the E4M3
//! `DenseRowGather`.

use poot_graph_ir::ValueId;

use super::index_rule::{IndexValue, index_at};
use poot_tensor::HostTensor;

use crate::EvalError;

/// `Iota { len }`: the range `[0, len)` in F32. The folded `Storage::Computed` constant the walks
/// materialize must agree with this element for element.
pub(crate) fn iota(len: usize) -> HostTensor {
    HostTensor::f32(vec![len], (0..len).map(|i| i as f32).collect::<Vec<_>>())
}

/// `PackI8`: pack 4 int8 codes per i32 word along the last axis (spec 048); the output is an I32
/// tensor. `x` is normally an f32 quantization result (`round`/`clamp` do the real work); a BF16/F16
/// operand widens exactly, and an I32 operand's already-integer words (exact below 2^24) are read as
/// values.
pub(crate) fn pack_i8(x: &HostTensor, out_shape: &[usize]) -> Result<HostTensor, EvalError> {
    let l = *x.shape().last().unwrap();
    let w = l.div_ceil(4);
    let source = x.to_f32()?;
    let rows = source.len() / l.max(1);
    let mut ints = vec![0i32; rows * w];
    for r in 0..rows {
        for c in 0..l {
            let byte = (source[r * l + c].round().clamp(-127.0, 127.0) as i32 as i8) as u8;
            ints[r * w + c / 4] |= (byte as i32) << ((c % 4) * 8);
        }
    }
    Ok(HostTensor::i32(out_shape.to_vec(), ints))
}

/// `UnpackI8 { len }`, the inverse of [`pack_i8`]: read each byte from the i32 words as signed i8,
/// widened to f32. The input must be an I32 tensor.
pub(crate) fn unpack_i8(
    x: &HostTensor,
    len: usize,
    out_shape: &[usize],
) -> Result<HostTensor, EvalError> {
    let w = *x.shape().last().unwrap();
    let rows = out_shape.iter().product::<usize>() / len.max(1);
    let words = x.as_i32().ok_or_else(|| {
        EvalError::unsupported(
            "unpack_i8",
            "UnpackI8 input must be an I32 tensor of packed words",
        )
    })?;
    let mut data = vec![0.0f32; rows * len];
    for r in 0..rows {
        for c in 0..len {
            let word = words[r * w + c / 4] as u32;
            let byte = (word >> ((c % 4) * 8)) as u8;
            data[r * len + c] = byte as i8 as f32;
        }
    }
    Ok(HostTensor::f32(out_shape.to_vec(), data))
}

/// Card 405: the E4M3 row of `DENSE_ROW_GATHER_SOURCE_DTYPES`: the table's rows selected by
/// authoritative I32 ids (through the one [`super::gather::gather`] engine) and decoded to f32 in one
/// step.
pub(crate) fn dense_row_gather_e4m3(
    table: &HostTensor,
    index: &HostTensor,
    out_shape: &[usize],
    eqn: ValueId,
) -> Result<HostTensor, EvalError> {
    let rows = index.as_i32().ok_or_else(|| {
        EvalError::unsupported("dense_row_gather", "the row ids must be an I32 tensor")
    })?;
    let bytes = table.view().bytes();
    let selected = super::gather::gather(
        |i| bytes[i],
        table.shape(),
        |p| IndexValue::I32(rows[p]),
        rows.len(),
        index.shape(),
        0,
        eqn,
    )?;
    let data: Vec<f32> = selected
        .iter()
        .map(|&b| poot_quant::scalar::e4m3fn_to_f32(b))
        .collect();
    Ok(HostTensor::f32(out_shape.to_vec(), data))
}

/// Batched argtop-k index extraction (spec 136): `rank[..,E] -> [..,k]` expert ids. `out[..,r]` is the
/// index `i` along the last axis with `rank[..,i] == r`, for `r` in `0..k`; per leading index this equals
/// `scatter(iota, rank)` sliced to `[0..k]`, without materializing the full `[..,E]` scatter. Each row's
/// `rank` must be a permutation of `0..E`; [`poot_graph_ir::ops::stable_descending_rank`] guarantees that
/// for tied and canonicalized non-finite scores. Ids are emitted as F32. `ArgTopK` only inverts rank and
/// owns no score-order semantic.
pub(crate) fn arg_top_k(
    rank: &HostTensor,
    k: usize,
    eqn: ValueId,
) -> Result<HostTensor, EvalError> {
    let e = *rank
        .shape()
        .last()
        .expect("ArgTopK input needs a last (expert) axis");
    let view = rank.to_f32()?;
    let rows = view.len() / e.max(1);
    let mut out_shape = rank.shape().to_vec();
    *out_shape.last_mut().unwrap() = k;
    let mut data = vec![0.0f32; rows * k];
    for l in 0..rows {
        let row = &view[l * e..(l + 1) * e];
        for (i, &ri) in row.iter().enumerate() {
            let r = index_at(IndexValue::F32(ri), l * e + i, e, eqn)?;
            if r < k {
                data[l * k + r] = i as f32;
            }
        }
    }
    Ok(HostTensor::f32(out_shape, data))
}
