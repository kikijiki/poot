//! Stable top-k routing helpers used by MoE graph evaluation.

use poot_tensor::HostTensor;

pub(crate) fn canonical_router_score(v: f32) -> f32 {
    if v.is_nan() || v == f32::NEG_INFINITY {
        f32::MIN
    } else if v == f32::INFINITY {
        f32::MAX
    } else if v == 0.0 {
        // Graph comparisons treat both IEEE zero encodings as equal. Normalize the representation before
        // `total_cmp` so this independent stable-sort oracle has the same lower-id tie rule.
        0.0
    } else {
        v
    }
}

pub(crate) fn checked_top_k_shape(x: &HostTensor, k: usize, op: &str) -> usize {
    let e = *x
        .shape()
        .last()
        .expect("topk_gate input has an expert axis");
    assert!(e > 0, "{op} needs at least one expert");
    assert!(
        (1..=e).contains(&k),
        "{op} needs 1 <= k <= E, got k={k}, E={e}"
    );
    e
}

/// The F32 values the routing oracles read: an F32 tensor, or a panic naming the oracle.
fn f32_rows<'a>(x: &'a HostTensor, op: &str) -> &'a [f32] {
    x.as_f32()
        .unwrap_or_else(|| panic!("{op} reads an F32 tensor, got {}", x.dtype()))
}

pub(crate) fn stable_top_k_row(row: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    // Stable sort preserves the initial lower-index order when canonical scores are equal. This remains
    // independent from the graph's pairwise comparison/rank construction.
    idx.sort_by(|&a, &b| canonical_router_score(row[b]).total_cmp(&canonical_router_score(row[a])));
    idx.truncate(k);
    idx
}

/// Independent stable-sort oracle for router selection. Returns rank-ordered expert ids as F32 with
/// shape `[.., k]`; unlike gate weights, ids stay evidence of selection when a softmax term underflows.
pub fn top_k_ids(x: &HostTensor, k: usize) -> HostTensor {
    let e = checked_top_k_shape(x, k, "top_k_ids");
    let data = f32_rows(x, "top_k_ids");
    let rows = data.len() / e;
    let mut out = Vec::with_capacity(rows * k);
    for row in data.chunks(e) {
        out.extend(stable_top_k_row(row, k).into_iter().map(|id| id as f32));
    }
    let mut shape = x.shape().to_vec();
    *shape.last_mut().expect("top_k_ids expert axis") = k;
    HostTensor::f32(shape, out)
}

/// Independent exact selection-mask oracle. Returns `1.0` for exactly `k` selected experts per row and
/// `0.0` elsewhere, separately from the normalized weights.
pub fn top_k_mask(x: &HostTensor, k: usize) -> HostTensor {
    let e = checked_top_k_shape(x, k, "top_k_mask");
    let data = f32_rows(x, "top_k_mask");
    let mut out = vec![0.0f32; data.len()];
    for (row_in, row_out) in data.chunks(e).zip(out.chunks_mut(e)) {
        for id in stable_top_k_row(row_in, k) {
            row_out[id] = 1.0;
        }
    }
    HostTensor::f32(x.shape().to_vec(), out)
}

/// Router logits `[.., E]` -> gate weights `[.., E]`: per row, keep the k largest (stable sort, lower index
/// wins ties, including signed-zero ties), softmax over exactly those, zero the rest. NaN and negative
/// infinity compare as `f32::MIN`; positive infinity as `f32::MAX`. Selected terms may underflow to zero,
/// so use [`top_k_ids`] or [`top_k_mask`] for selection evidence. A sort-based differential oracle for
/// the primitive graph composition, not an execution path.
pub fn top_k_gate(x: &HostTensor, k: usize) -> HostTensor {
    let e = checked_top_k_shape(x, k, "top_k_gate");
    let data = f32_rows(x, "top_k_gate");
    let mut out = vec![0.0f32; data.len()];
    for (row_in, row_out) in data.chunks(e).zip(out.chunks_mut(e)) {
        let top = stable_top_k_row(row_in, k);
        let m = top
            .iter()
            .map(|&i| canonical_router_score(row_in[i]))
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for &i in &top {
            denom += (canonical_router_score(row_in[i]) - m).exp();
        }
        for &i in &top {
            row_out[i] = (canonical_router_score(row_in[i]) - m).exp() / denom;
        }
    }
    HostTensor::f32(x.shape().to_vec(), out)
}
