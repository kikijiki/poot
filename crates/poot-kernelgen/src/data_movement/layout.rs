use super::*;

/// Standalone broadcast `out = in` broadcast to `out_shape` (right-aligned; broadcast dims map to 0).
pub fn broadcast(name: &str, out_shape: &[usize], in_shape: &[usize]) -> Body {
    broadcast_dt(name, Ty::F32, out_shape, in_shape)
}

/// Broadcast `in` to `out_shape` in storage dtype `dt` (spec 024; a pure copy, no casts).
pub fn broadcast_dt(name: &str, dt: Ty, out_shape: &[usize], in_shape: &[usize]) -> Body {
    let eff = broadcast_eff_strides(out_shape, in_shape);
    let terms: Vec<(usize, usize)> = (0..out_shape.len()).map(|d| (d, eff[d])).collect();
    index_remap_copy(
        name,
        dt,
        &row_major_strides(out_shape),
        out_shape,
        &terms,
        0,
    )
}

/// Transpose in storage dtype `dt` (spec 024; a pure copy, no casts).
pub fn transpose_dt(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    in_shape: &[usize],
    perm: &[usize],
) -> Body {
    let in_strides = row_major_strides(in_shape);
    let terms: Vec<(usize, usize)> = (0..out_shape.len())
        .map(|d| (d, in_strides[perm[d]]))
        .collect();
    index_remap_copy(
        name,
        dt,
        &row_major_strides(out_shape),
        out_shape,
        &terms,
        0,
    )
}
