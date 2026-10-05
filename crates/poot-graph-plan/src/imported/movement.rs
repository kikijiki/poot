//! Data-movement imported kernels (card 044): gather/scatter/index-remap/concat/DUS bodies, authored in
//! Rust (`pootc/kernels/movement/*.rs`) and the row-major/broadcast/index-remap geometry helpers that
//! build the [`crate::ComputeMeta`] buffers these shape-generic kernels read their dims from. 560b
//! retires this family into kernelgen.

kernel_assets! {
    family = body_movement;

    GatherAxis0 {
        name: "gather_axis0",
        source: "gather_axis0",
        entry: "__poot_kernel_gather_axis0",
        params: 3,
        dest: GraphPlan,
    },
    GatherAxis {
        name: "gather_axis",
        source: "gather_axis",
        entry: "__poot_kernel_gather_axis",
        params: 4,
        dest: GraphPlan,
    },
    ScatterAxis0 {
        name: "scatter_axis0",
        source: "scatter_axis0",
        entry: "__poot_kernel_scatter_axis0",
        params: 3,
        dest: GraphPlan,
    },
    ScatterUpdate {
        name: "scatter_update",
        source: "scatter_update",
        entry: "__poot_kernel_scatter_update",
        params: 4,
        dest: GraphPlan,
    },
    IndexRemap {
        name: "index_remap",
        source: "index_remap",
        entry: "__poot_kernel_index_remap",
        params: 3,
        dest: GraphPlan,
    },
    DynUpdateSlice {
        name: "dyn_update_slice",
        source: "dyn_update_slice",
        entry: "__poot_kernel_dyn_update_slice",
        params: 5,
        dest: GraphPlan,
    },
    Concat2 {
        name: "concat2",
        source: "concat2",
        entry: "__poot_kernel_concat2",
        params: 4,
        dest: GraphPlan,
    },
}

/// Row-major (C-contiguous) strides of `shape`: `strides[d] = product(shape[d+1..])`, innermost = 1.
pub(crate) fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// Broadcast-effective input strides of `operand` against `out` (right-aligned): 0 where the operand is
/// broadcast (the dim is 1 or missing), else the operand's own row-major stride for that dim. The
/// `src_term` an [`index_remap_meta`] broadcast uses, so a broadcast dim's coordinate contributes
/// nothing to `src`.
pub(crate) fn broadcast_eff_strides(out: &[usize], operand: &[usize]) -> Vec<usize> {
    let own = row_major_strides(operand);
    let offset = out.len() - operand.len();
    (0..out.len())
        .map(|d| {
            if d < offset || operand[d - offset] == 1 {
                0
            } else {
                own[d - offset]
            }
        })
        .collect()
}

/// Build the [`ImportedKernel::IndexRemap`] metadata buffer for an index-remap copy: `out[i] =
/// input[src]` with `src = src_base + sum_d coord_d * src_term[d]`, `coord_d = (i / out_stride[d]) %
/// out_dim[d]`. The caller supplies the per-out-dim `src_term` (the input stride that out dim maps to)
/// and the flat `src_base`; this packs `[rank, src_base, (out_stride, out_dim, src_term) x rank]`
/// (out_stride is the row-major stride of `out_shape`).
pub(crate) fn index_remap_meta(
    out_shape: &[usize],
    src_terms: &[usize],
    src_base: usize,
) -> Vec<u32> {
    let out_strides = row_major_strides(out_shape);
    let mut m = Vec::with_capacity(2 + out_shape.len() * 3);
    m.push(out_shape.len() as u32);
    m.push(src_base as u32);
    for d in 0..out_shape.len() {
        m.push(out_strides[d] as u32);
        m.push(out_shape[d] as u32);
        m.push(src_terms[d] as u32);
    }
    m
}

/// The dynamic update-slice (contiguous KV-cache write) metadata buffer for
/// [`ImportedKernel::DynUpdateSlice`]: `[rank, axis, extent, (out_stride, out_dim, upd_stride) x rank]`.
/// `out_stride` is the row-major stride of `out_shape`; `upd_stride` is the row-major stride of the
/// update shape (`out_shape` with `out_shape[axis]` replaced by `extent`).
pub(crate) fn dus_meta(out_shape: &[usize], axis: usize, extent: usize) -> Vec<u32> {
    let out_strides = row_major_strides(out_shape);
    let mut upd_shape = out_shape.to_vec();
    upd_shape[axis] = extent;
    let upd_strides = row_major_strides(&upd_shape);
    let mut m = Vec::with_capacity(3 + out_shape.len() * 3);
    m.push(out_shape.len() as u32);
    m.push(axis as u32);
    m.push(extent as u32);
    for d in 0..out_shape.len() {
        m.push(out_strides[d] as u32);
        m.push(out_shape[d] as u32);
        m.push(upd_strides[d] as u32);
    }
    m
}

/// Build the [`ImportedKernel::Concat2`] metadata buffer for the two-input concat along `axis`: `[rank,
/// axis, a_axis_len, (out_stride, out_dim, a_stride, b_stride) x rank]`. The strides are the row-major
/// strides of `out_shape`, `a_shape`, `b_shape`.
pub(crate) fn concat2_meta(
    out_shape: &[usize],
    axis: usize,
    a_shape: &[usize],
    b_shape: &[usize],
) -> Vec<u32> {
    let out_strides = row_major_strides(out_shape);
    let a_strides = row_major_strides(a_shape);
    let b_strides = row_major_strides(b_shape);
    let mut m = Vec::with_capacity(3 + out_shape.len() * 4);
    m.push(out_shape.len() as u32);
    m.push(axis as u32);
    m.push(a_shape[axis] as u32);
    for d in 0..out_shape.len() {
        m.push(out_strides[d] as u32);
        m.push(out_shape[d] as u32);
        m.push(a_strides[d] as u32);
        m.push(b_strides[d] as u32);
    }
    m
}
