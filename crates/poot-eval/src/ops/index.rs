//! Index-producing and index-consuming ops: `Iota`, `ArgTopK` and the E4M3
//! `DenseRowGather`.

use poot_graph_ir::{ExpertGrouping, ExpertRows, ValueId};
use poot_tensor::HostData;

use super::index_rule::{IndexValue, index_at};
use super::movement::index_values;
use poot_tensor::HostTensor;

use crate::{EvalError, GroupingFault};

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

/// `ExpertGroup`: `ids [M]` to the [`ExpertGrouping`] layout, bucket by bucket (bucket `experts` for an id outside
/// `0..experts`, so a bad id is carried to the contraction that fails it closed instead of being dropped here),
/// each bucket's rows in row order and its run padded with the sentinel `M` to whole tiles of `block` positions,
/// then each tile's bucket (`experts` for the empty tiles past the last run).
pub(crate) fn expert_group(
    ids: &HostTensor,
    experts: usize,
    block: usize,
) -> Result<HostTensor, EvalError> {
    let HostData::I32(words) = ids.data() else {
        return Err(EvalError::unsupported(
            "expert_group",
            format!("the ids are I32, got {}", ids.dtype()),
        ));
    };
    let rows = words.len();
    let grouping = ExpertGrouping {
        rows,
        experts,
        block,
    };
    let bucket = |id: i32| {
        usize::try_from(id)
            .ok()
            .filter(|&e| e < experts)
            .unwrap_or(experts)
    };
    let sentinel = i32::try_from(rows).map_err(|_| {
        EvalError::unsupported("expert_group", format!("{rows} rows overflow an I32 word"))
    })?;
    let mut order = Vec::with_capacity(grouping.positions());
    let mut tiles = Vec::with_capacity(grouping.tiles());
    for b in 0..=experts {
        let start = order.len();
        order.extend(
            (0..rows)
                .filter(|&row| bucket(words[row]) == b)
                .map(|row| row as i32),
        );
        let placed = order.len() - start;
        let padded = placed.div_ceil(block) * block;
        order.resize(start + padded, sentinel);
        tiles.resize(tiles.len() + padded / block, b as i32);
    }
    order.resize(grouping.positions(), sentinel);
    tiles.resize(grouping.tiles(), experts as i32);
    order.extend(tiles);
    Ok(HostTensor::i32(vec![grouping.len()], order))
}

/// The expert each of the `rows` rows of an indexed contraction reads, from its expert operand as `select` says:
/// the per-row ids, each checked against `experts` by [`index_at`], or a grouping, whose padded order must place
/// every row exactly once ([`GroupingFault`]) and whose tile buckets are checked as ids are (a tile of the invalid
/// bucket is the typed refusal an invalid id is).
pub(crate) fn row_experts(
    operand: &HostTensor,
    select: ExpertRows,
    rows: usize,
    experts: usize,
    eqn: ValueId,
) -> Result<Vec<usize>, EvalError> {
    let block = match select {
        ExpertRows::PerRow => {
            return index_values(operand)?
                .into_iter()
                .enumerate()
                .map(|(position, value)| index_at(value, position, experts, eqn))
                .collect();
        }
        ExpertRows::Grouped { block } => block,
    };
    let fault = |fault| EvalError::ExpertGrouping { eqn, rows, fault };
    let grouping = ExpertGrouping {
        rows,
        experts,
        block,
    };
    let HostData::I32(words) = operand.data() else {
        return Err(EvalError::unsupported(
            "indexed_matmul",
            format!("an expert grouping is I32, got {}", operand.dtype()),
        ));
    };
    if words.len() != grouping.len() {
        return Err(fault(GroupingFault::Length {
            expected: grouping.len(),
            actual: words.len(),
        }));
    }
    let positions = grouping.positions();
    let mut expert_of: Vec<Option<usize>> = vec![None; rows];
    for (position, &word) in words[..positions].iter().enumerate() {
        let row = usize::try_from(word)
            .ok()
            .filter(|&row| row <= rows)
            .ok_or(fault(GroupingFault::Word { position, word }))?;
        if row == rows {
            continue;
        }
        let tile = positions + position / block;
        let expert = index_at(IndexValue::I32(words[tile]), tile, experts, eqn)?;
        if expert_of[row].replace(expert).is_some() {
            return Err(fault(GroupingFault::Repeated { row }));
        }
    }
    expert_of
        .into_iter()
        .enumerate()
        .map(|(row, expert)| expert.ok_or(fault(GroupingFault::Missing { row })))
        .collect()
}
