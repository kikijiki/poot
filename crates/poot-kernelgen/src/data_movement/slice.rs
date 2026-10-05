use super::*;

/// Slice in storage dtype `dt` (spec 024; a pure copy, no casts).
pub fn slice_dt(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    in_shape: &[usize],
    axis: usize,
    start: usize,
) -> Body {
    let in_strides = row_major_strides(in_shape);
    let terms: Vec<(usize, usize)> = (0..out_shape.len()).map(|d| (d, in_strides[d])).collect();
    index_remap_copy(
        name,
        dt,
        &row_major_strides(out_shape),
        out_shape,
        &terms,
        start * in_strides[axis],
    )
}
