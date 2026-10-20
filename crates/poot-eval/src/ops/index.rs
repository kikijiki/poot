//! Index-producing and index-consuming ops: `Iota`, `ArgTopK` and the E4M3
//! `DenseRowGather`.

use poot_graph_ir::ValueId;

use super::index_rule::{IndexValue, index_at};
use poot_tensor::HostTensor;

use crate::EvalError;

/// Batched argtop-k index extraction (spec 136): `rank[..,E] -> [..,k]` expert ids. `out[..,r]` is the
/// index `i` along the last axis with `rank[..,i] == r`, for `r` in `0..k`; per leading index this equals
/// `scatter(iota, rank)` sliced to `[0..k]`, without materializing the full `[..,E]` scatter. Each row's
/// `rank` must be a permutation of `0..E`; [`poot_graph_ir::ops::stable_descending_rank`] guarantees that
/// for tied and canonicalized non-finite scores. Ids are emitted as I32. `ArgTopK` only inverts rank and
/// owns no score-order semantic. The rank is an F32 value (an integral count computed by arithmetic), the
/// one place an [`IndexValue::F32`] reaches the index rule.
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
    let mut data = vec![0i32; rows * k];
    for l in 0..rows {
        let row = &view[l * e..(l + 1) * e];
        for (i, &ri) in row.iter().enumerate() {
            let r = index_at(IndexValue::F32(ri), l * e + i, e, eqn)?;
            if r < k {
                data[l * k + r] = i as i32;
            }
        }
    }
    Ok(HostTensor::i32(out_shape, data))
}
