//! Row-major index geometry shared by the CPU oracle's ops: strides, flat/multi-index conversion,
//! numpy broadcast and the Gather source index.

pub(crate) fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

pub(crate) fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let st = strides(shape);
    let mut idx = vec![0usize; shape.len()];
    for i in 0..shape.len() {
        idx[i] = flat / st[i];
        flat %= st[i];
    }
    idx
}

/// Map a full (out) multi-index to a flat index into an operand of `in_shape`, applying numpy broadcast
/// (right-aligned; a size-1 dim contributes 0). `out_idx` has rank >= in_shape rank.
pub(crate) fn broadcast_flat(out_idx: &[usize], in_shape: &[usize]) -> usize {
    let st = strides(in_shape);
    let off = out_idx.len() - in_shape.len();
    let mut flat = 0;
    for (i, &dim) in in_shape.iter().enumerate() {
        let coord = if dim == 1 { 0 } else { out_idx[off + i] };
        flat += coord * st[i];
    }
    flat
}

/// The row-major source element that Gather output element `flat` reads. `row(position)` is the selected row
/// of row-major index element `position`; the caller has range-checked it.
pub(crate) fn gather_source_flat(
    flat: usize,
    out_shape: &[usize],
    data_strides: &[usize],
    index_strides: &[usize],
    axis: usize,
    row: impl Fn(usize) -> usize,
) -> usize {
    let irank = index_strides.len();
    let out_idx = unravel(flat, out_shape);
    let (pre, rest) = out_idx.split_at(axis);
    let (idx_coords, post) = rest.split_at(irank);
    let idx_flat: usize = idx_coords
        .iter()
        .zip(index_strides)
        .map(|(c, st)| c * st)
        .sum();
    // reassemble the source index: pre ++ [row] ++ post.
    let mut src = row(idx_flat) * data_strides[axis];
    for (d, &c) in pre.iter().enumerate() {
        src += c * data_strides[d];
    }
    for (d, &c) in post.iter().enumerate() {
        src += c * data_strides[axis + 1 + d];
    }
    src
}
