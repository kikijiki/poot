//! Per-eqn routing: the strided-view promotion pass (`compute_views`, spec 132) and the dispatch
//! predicates and launch-grid math (`is_decode_gemv`, `is_tiled_gemm`, `is_batched_tiled_gemm`,
//! `is_decode_attn_gemv`, `is_elementwise_2d`, [`default_grid_in`], ...). The planner computes each
//! eqn's launch grid once, alongside the kernel choice it shapes, and stores it in the [`Plan`]
//! (card 525); no separate dispatch-time function re-derives it. Also the `*_plan` helpers that build
//! a chunked [`Plan`] for ops whose dispatch can exceed a single-launch cap (RADV 2^15-tile miscompile,
//! iGPU display-watchdog TDR; see each function's doc).

use poot_tensor::DType;
use std::collections::HashMap;

use poot_graph_ir::op::{OpKind, UnOp as GUnOp};
use poot_graph_ir::{Eqn, Graph, Operand, TensorType, ValidationChannel, ValueId, ValueMeta};
use poot_kernelgen as kg;
use poot_target::{Backend, DeviceCaps, TensorCoreSupport};

use crate::imported::*;
use crate::{
    ContractionSpec, KernelRequest, Layout, MovementSpec, PlanError, Planned, Site, fty_ty,
};

/// The codegen capability query for the equation's backend (card 522, R468-014). `Backend`'s `AmdGcn`
/// payload already carries the same `AmdArch` `poot_codegen::Target::AmdGcn` takes, so the conversion is
/// a bare re-tag, not a new dependency.
fn codegen_capability(backend: Backend) -> poot_codegen::Capability {
    poot_codegen::capability(codegen_target(backend))
}

/// The codegen target a planner backend compiles for.
pub(crate) fn codegen_target(backend: Backend) -> poot_codegen::Target {
    match backend {
        Backend::SpirvVulkan => poot_codegen::Target::SpirvVulkan,
        Backend::Nvptx => poot_codegen::Target::Nvptx,
        Backend::AmdGcn(arch) => poot_codegen::Target::AmdGcn(arch),
    }
}

/// The value-operand ids of an eqn, in order (literals dropped) - these are the kernel's data params.
pub fn value_ids(eqn: &Eqn) -> Vec<ValueId> {
    eqn.inputs
        .iter()
        .filter_map(|o| match o {
            Operand::Value(v) => Some(*v),
            Operand::Lit(_) => None,
        })
        .collect()
}

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

/// The value and equation tables of one graph, borrowed without its result metadata.
///
/// Per-equation planning and the dispatch predicates read only these tables, so their bodies are compiled
/// once for every validation channel and the public entry points are thin generic wrappers. It has no
/// output, validation, or state roots, so it cannot stand in for a graph in liveness, view, or storage
/// analysis.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphTables<'g> {
    pub(crate) values: &'g [ValueMeta],
    pub(crate) eqns: &'g [Eqn],
}

impl<'g> GraphTables<'g> {
    pub(crate) fn aval(self, id: ValueId) -> &'g TensorType {
        &self.values[id].aval
    }
}

impl<'g, V: ValidationChannel> From<&'g Graph<V>> for GraphTables<'g> {
    fn from(graph: &'g Graph<V>) -> Self {
        Self {
            values: &graph.values,
            eqns: &graph.eqns,
        }
    }
}

/// The eqns that read each value as a value-operand, by index into `g.eqns` (the reverse of a producer
/// map). [`compute_views`] uses it to check that every consumer of a movement op's output can take a
/// strided input.
fn consumers(g: GraphTables<'_>) -> HashMap<ValueId, Vec<usize>> {
    let mut out: HashMap<ValueId, Vec<usize>> = HashMap::new();
    for (i, eqn) in g.eqns.iter().enumerate() {
        for v in value_ids(eqn) {
            out.entry(v).or_default().push(i);
        }
    }
    out
}

/// Can this eqn read a strided (non-row-major) value operand directly, with no materialize copy first?
///
/// Conservative allowlist: elementwise `Binary` against another value (`binary_broadcast_dt_views`),
/// `Fused` / `FusedRow` (`fused_views` / `fused_row_parallel_views`), and the `Unary` sub-ops that share
/// `build_unary_dt`'s skeleton (`Neg`/`Recip`/`Sqrt`/`Round`/`Exp`/`Log`, via [`map_unary_views_grid`]).
/// Everything else (contractions, gather/scatter, concat, standalone `Reduce`, `Reshape`,
/// `DynamicUpdateSlice`) keeps materializing.
///
/// `Binary` against a literal is excluded: it lowers to `binary_scalar_dt`, which reads its value operand
/// as a flat `x[i]`, so a view would be misread. This is checked on `eqn.inputs`, not just the `OpKind`.
///
/// `Unary(Tanh | Erf)` is excluded: they have bespoke bodies with no view-capable variant. A
/// Transpose/Slice/Broadcast feeding only one of them still materializes. `Neg` is included because the
/// qwen2 decode graph's RoPE `rotate_half` (`concat(neg(slice), slice)`) needs it.
fn is_strided_capable(eqn: &Eqn) -> bool {
    match &eqn.op {
        OpKind::Binary(_) => matches!(eqn.inputs.get(1), Some(Operand::Value(_))),
        OpKind::Fused(_) | OpKind::FusedRow(_) => true,
        OpKind::Unary(uop) => matches!(
            uop,
            GUnOp::Neg | GUnOp::Recip | GUnOp::Sqrt | GUnOp::Round | GUnOp::Exp | GUnOp::Log
        ),
        _ => false,
    }
}

/// Is `op` one of the 3 movement ops promoted to a view (Transpose/Slice/Broadcast)? Reshape stays on
/// the separate zero-cost `Plan::Alias` path. Used by [`compute_views`] for the candidate check and to let
/// a promoted movement op count as a strided-capable consumer of another one it composes with.
fn is_movement_view_candidate(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::Transpose { .. } | OpKind::Slice { .. } | OpKind::Broadcast { .. }
    )
}

/// Which values become a [`Plan::View`] (no dispatch) instead of a materialize copy, composing a whole
/// chain of movement ops into one Layout each (spec 132). WGPU and AMD GCN only; any other backend gets an
/// empty map and takes the pre-existing movement lowering.
///
/// Two passes, since deciding promotion needs consumers (later in topological order) and composing a
/// Layout needs the input's Layout (earlier):
///
/// - Pass A walks `g.eqns` in reverse and decides the promoted set. A `Transpose`/`Slice`/`Broadcast`
///   output (not pinned, F32) is promoted when every consumer is `is_strided_capable` or is itself a
///   movement-view candidate already promoted (consumers are visited first in reverse order).
/// - Pass B walks forward and composes each promoted value's Layout onto its input's Layout
///   (`Layout::contiguous` if the input was not promoted). A chain tail therefore has a Layout expressed
///   in terms of the original contiguous source, matching `poot_eval::apply_movement` over the whole
///   chain (see the `view_..._chain_matches_oracle_fuzz` test).
///
/// A value with no consumer is not promoted. A pinned value, a non-F32 dtype, or a non-strided-capable leaf
/// consumer down the chain keeps the rest of the chain materialized. Unpromoted ops keep the
/// `Plan::ComputeMeta` index-remap copy.
///
/// Every pinned value ([`Graph::pinned_values`]: the output, each validation value, and both sides of each
/// state pair) keeps its own materialized buffer, so a validation packet can copy it row-major from offset 0.
pub fn compute_views<V: ValidationChannel>(
    g: &Graph<V>,
    backend: Backend,
) -> HashMap<ValueId, Layout> {
    let mut views: HashMap<ValueId, Layout> = HashMap::new();
    if !codegen_capability(backend).strided_views {
        return views;
    }
    let cons = consumers(g.into());
    let pinned: std::collections::HashSet<ValueId> = g.pinned_values().collect();

    // Pass A (reverse topological order): decide the promoted set (no strides yet).
    let mut promoted: std::collections::HashSet<ValueId> = std::collections::HashSet::new();
    for eqn in g.eqns.iter().rev() {
        let out = eqn.out;
        if pinned.contains(&out)
            || g.aval(out).dtype != DType::F32
            || !is_movement_view_candidate(&eqn.op)
        {
            continue;
        }
        let all_strided_capable = cons.get(&out).is_some_and(|users| {
            users.iter().all(|&i| {
                let c = &g.eqns[i];
                is_strided_capable(c)
                    || (is_movement_view_candidate(&c.op) && promoted.contains(&c.out))
            })
        });
        if all_strided_capable {
            promoted.insert(out);
        }
    }

    // Pass B (forward topological order): compose each promoted value's Layout from its input's Layout.
    for eqn in &g.eqns {
        let out = eqn.out;
        if !promoted.contains(&out) {
            continue;
        }
        let ids = value_ids(eqn);
        let in_shape = g.aval(ids[0]).shape.clone();
        let in_layout = views
            .get(&ids[0])
            .cloned()
            .unwrap_or_else(|| Layout::contiguous(&in_shape));
        let (strides, offset) = match &eqn.op {
            // Transpose permutes the view's strides; the base offset does not move.
            OpKind::Transpose { perm } => (
                (0..perm.len())
                    .map(|d| in_layout.strides[perm[d]])
                    .collect(),
                in_layout.offset,
            ),
            // Slice keeps the strides; the offset advances by start*stride[axis].
            OpKind::Slice { axis, start, .. } => {
                let out_shape = &g.aval(out).shape;
                (
                    (0..out_shape.len()).map(|d| in_layout.strides[d]).collect(),
                    in_layout.offset + start * in_layout.strides[*axis],
                )
            }
            // `view_eff_strides`: 0 on a size-1 logical dim, else the view's own stride, right-aligned.
            OpKind::Broadcast { shape: s } => kg::view_eff_strides(s, &in_shape, &in_layout),
            _ => unreachable!(
                "promoted only ever contains Transpose/Slice/Broadcast outputs (is_movement_view_candidate)"
            ),
        };
        views.insert(out, Layout { strides, offset });
    }
    views
}

/// Workgroup width (lanes per output column) for the wgpu LDS decode-GEMV (`kernelgen::gemv_lds`), used by
/// `plan_eqn` for both the `[w,1,1]` workgroup and the planned `N*GEMV_WIDTH`-thread grid. 128 measured
/// best on RADV STRIX_HALO (Qwen2.5-0.5B decode, fused ms/tok: w=32 50.9, w=64 53.7, w=128 39.8, w=256 45.6).
pub const GEMV_WIDTH: usize = 128;

/// Output columns owned by one workgroup of the coalesced imported decode-GEMV
/// (`pootc/tests/kernels/gemv_coalesced.rs`), used by `plan_eqn`: the body bakes `GEMV_TILE = 32` and its
/// tiled grid arm plans one workgroup per `GEMV_TILE` columns. 32 = one wave32's contiguous
/// `W[k, col0:col0+32]` run; `STRIPS = GEMV_WIDTH/GEMV_TILE = 4` K-stripes fold through LDS. The
/// kernelgen chunk body (`kg::gemv_lds`) is still one workgroup per output element and does not use this.
pub const GEMV_TILE: usize = 32;

/// Does this MatMul take the decode-GEMV path (`kernelgen::gemv_lds`)? Requires M==1 (`out_shape[-2]==1`)
/// and a rank-2 `[K,N]` weight (`inputs[1]`), which separates a projection GEMV (`a[B,1,K] @ W[K,N]`) from
/// a per-head attention matmul (`q[B,Hq,1,D] @ k[B,Hkv,cap,D]`, batched weight). `out_numel = B*N` must fit
/// the 2-D workgroup grid ([`gemv_grid`]).
///
/// Card 671/ADR-0113: this used to have a `pub` generic-over-`V` wrapper for cross-crate tests, with no
/// production caller of its own - deleted. Cross-crate tests that used to call it now assert on the `Plan`
/// [`crate::plan_eqn_analyzed`] returns instead (its key always starts with `"matmul:gemv"` for this path,
/// `"matmul:tiled"` for [`is_tiled_gemm_in`]'s).
pub(crate) fn is_decode_gemv_in(
    g: GraphTables<'_>,
    backend: Backend,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> bool {
    if !matches!(eqn.op, OpKind::MatMul | OpKind::MatMulBias) {
        return false;
    }
    // The imported decode GEMV is f32; bf16 decode on NVPTX/AmdGcn takes the tensor-core path.
    if g.aval(eqn.out).dtype != DType::F32 {
        return false;
    }
    let r = out_shape.len();
    if r < 2 || out_shape[r - 2] != 1 || out_numel == 0 {
        return false;
    }
    // B=1 is a plain `Plan::Compute`; batched B>1 is a `Plan::ComputeMeta` (`dims = [B]` buffer). On NVPTX
    // that buffer lives in the executor's const cache, keeping a stable device address across CUDA-graph
    // replay. The coalesced imported body lays one workgroup per [`GEMV_TILE`] columns over a 2-D grid
    // (`col0 = GroupY*x_groups+GroupX` times the tile), so the Y cap is on the tiled workgroup count;
    // reject only if that would overflow.
    if gemv_grid(out_numel.div_ceil(GEMV_TILE).max(1), caps).1 > caps.max_grid[1] as usize {
        return false;
    }
    // NVPTX only: for large decode projections (gate/up/down/lm_head, N in the thousands) the GEMV's serial
    // lane-0 LDS reduction and stride-N weight reads make it ~2x slower than `matmul_batched` on Ampere
    // (found by the 2026-07-03 benchmark). Keep the GEMV only for small N (the weight's output dim).
    if matches!(backend, Backend::Nvptx) && out_shape[out_shape.len() - 1] > 256 {
        return false;
    }
    // The weight (inputs[1]) must be rank-2 [K,N]; inputs[2], if present, is the bias [N].
    let ids = value_ids(eqn);
    ids.len() >= 2 && g.aval(ids[1]).shape.len() == 2
}

/// The 2-D workgroup grid `(x_groups, y_groups)` for a decode GEMV over `out_numel` elements (one workgroup
/// per element). X fills up to `caps.max_grid[0]` (the planner reads the target's
/// own measured cap, never a baked wgpu-specific literal) and the remainder spills onto Y, so a
/// ~150k-vocab `lm_head` fits. For `out_numel <= caps.max_grid[0]` this is `(out_numel, 1)`. Used by
/// `plan_eqn`, which bakes `x_groups` into the kernel and the planned grid.
pub fn gemv_grid(out_numel: usize, caps: &DeviceCaps) -> (usize, usize) {
    kg::fold_groups(out_numel, caps.max_grid[0])
}

/// Plan a decode-GEMV (`is_decode_gemv`'s `M==1` shared-weight matmul). Below `caps.watchdog_budget`'s
/// `decode_gemv_out_elems` output elements, or on a device with no watchdog budget at all (no display
/// watchdog off wgpu/RADV, card 163), this is the coalesced imported kernel: `Plan::Compute` for `B=1`,
/// `Plan::ComputeMeta` with a `dims=[B]` buffer for `B>1`. Above the trigger it splits into
/// `kernelgen::gemv_lds` dispatches over disjoint column ranges of the same output buffer - that chunk
/// body is still one workgroup per output element with the stride-N read (`elem_offset` baked per chunk;
/// the coalesced import has no offset param), so it is the shape gate that keeps the old access pattern.
/// `chunk_elems` is capped at `caps.max_grid[0]` because `Plan::ComputeChunks`' `groups` is a single 1-D
/// workgroup count. `k`/`n` are the weight's `[K,N]` dims; `bias` selects the `MatMulBias` epilogue.
///
/// A single dispatch at BLOOM's scale (`N=250880, K=1024`) completes in ~0.6s alone, but the same dispatch
/// right after another large one (a materializing `transpose` of the same weight, ~3.7s) trips the Strix
/// Halo iGPU's display-watchdog TDR within ~1.5s, even though `GpuExecutor::run` polls after every submit.
/// The trigger is the total busy stretch across a rapid sequence of large dispatches, not one dispatch's
/// duration. Splitting gives display work scheduling gaps.
///
/// `N` is the dominant hazard axis: BLOOM `N=250880` hangs; gpt-oss `N=201088` (tied, `K=32`) and SmolLM3
/// `N=128256` (tied, `K=2048`, higher `out_numel*K` than BLOOM's `2.57e8`) do not. So the trigger is
/// element-count-only (unlike the K-scaled `serial_dequant_work`), and the box's own wgpu/RADV device is
/// measured at 220_000 (`DeviceCaps::wgpu_rdna3_igpu`), between the proven-safe 201088 and the
/// proven-hanging 250880.
///
/// `weight_bf16` selects the packed-BF16-weight body variant (same coalesced shape; the weight is
/// `&[u32]` lanes holding checkpoint BF16, widened in-register) under the eligibility rule
/// [`crate::matmul_bf16_decode_gemv_eligible`] - dtype-driven planning, not a separate menu. The
/// chunk arm is never taken with `weight_bf16` (eligibility excludes the over-trigger chunk shape on a
/// device with a watchdog budget), so that path stays f32-only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_gemv_plan(
    site: Site<'_>,
    backend: Backend,
    k: usize,
    n: usize,
    out_numel: usize,
    bias: bool,
    weight_bf16: bool,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    // One workgroup of GEMV_WIDTH lanes per output element on the same 2-D spill grid every
    // is_decode_gemv-routed body uses (see `is_decode_gemv_in`'s doc); shared by every branch below
    // (single-seq, batched, and chunked all read the same weight layout at the same grid shape).
    let nwg = out_numel.div_ceil(GEMV_TILE).max(1);
    let gx = nwg.min(caps.max_grid[0] as usize);
    let gy = nwg.div_ceil(gx);
    let grid = [(gx * GEMV_WIDTH) as u32, gy as u32, 1];
    // B=1 (`out_numel == n`) takes no metadata; a batched decode binds `dims = [B]`.
    let batch_meta = (out_numel != n).then(|| vec![(out_numel / n.max(1)) as u32]);
    if weight_bf16 {
        debug_assert!(
            caps.watchdog_budget
                .is_none_or(|w| out_numel <= w.decode_gemv_out_elems as usize),
            "bf16 decode GEMV eligibility excludes the over-trigger chunk shape"
        );
        let kernel = match (batch_meta.is_some(), bias) {
            (false, false) => ImportedKernel::GemvCoalescedBf16,
            (false, true) => ImportedKernel::GemvCoalescedBiasBf16,
            (true, false) => ImportedKernel::GemvBatchedCoalescedBf16,
            (true, true) => ImportedKernel::GemvBatchedCoalescedBiasBf16,
        };
        return Ok(Planned::imported(kernel, backend, batch_meta, grid));
    }
    let over_trigger = caps
        .watchdog_budget
        .is_some_and(|w| out_numel > w.decode_gemv_out_elems as usize);
    if !over_trigger {
        let kernel = match (batch_meta.is_some(), bias) {
            (false, false) => ImportedKernel::GemvCoalesced,
            (false, true) => ImportedKernel::GemvCoalescedBias,
            (true, false) => ImportedKernel::GemvBatchedCoalesced,
            (true, true) => ImportedKernel::GemvBatchedCoalescedBias,
        };
        return Ok(Planned::imported(kernel, backend, batch_meta, grid));
    }
    let request = |elems, offset| {
        KernelRequest::Contraction(ContractionSpec::GemvChunk {
            k,
            n,
            width: GEMV_WIDTH,
            bias,
            elems,
            offset,
        })
    };
    gemv_chunks(site, backend, out_numel, request, caps)
}

/// A decode GEMV as `kg::gemv_lds` dispatches over disjoint output ranges of one buffer: one workgroup of
/// [`GEMV_WIDTH`] lanes per output element, at most `caps.max_grid[0]` of them per chunk (`Plan::ComputeChunks`'
/// `groups` is a single 1-D workgroup count). `request(elems, offset)` builds one chunk's request, so the weight
/// layout and bias are the caller's.
pub(crate) fn gemv_chunks(
    site: Site<'_>,
    backend: Backend,
    out_numel: usize,
    request: impl Fn(usize, usize) -> KernelRequest,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    let chunk_elems = caps.max_grid[0] as usize;
    let n_chunks = out_numel.div_ceil(chunk_elems);
    let chunks = (0..n_chunks)
        .map(|ci| {
            let offset = ci * chunk_elems;
            let elems = (out_numel - offset).min(chunk_elems);
            Planned::generated_chunk(site, request(elems, offset), backend, caps)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Planned::chunked(chunks))
}

/// The [`kg::Schedule::Gemv`] a `DenseContraction` decode GEMV runs (M == 1 once every leading dim folds into
/// M), or `None` when the equation is not one or its launch would not fit the device's X grid. Every weight
/// dtype the fold admits (F32, packed F16, packed BF16) and every backend take the same generated body; the
/// launch comes from [`dense_gemv_launch`].
pub(crate) fn dense_decode_gemv_in(
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> Option<kg::Schedule> {
    if !matches!(eqn.op, OpKind::DenseContraction { .. }) {
        return None;
    }
    let n = *out_shape.last()?;
    if out_shape.len() < 2 || n == 0 || out_numel != n {
        return None;
    }
    let launch = dense_gemv_launch(n, caps.compute_units);
    (n.div_ceil(launch.cols as usize) <= caps.max_grid[0] as usize).then_some(kg::Schedule::Gemv {
        width: launch.width,
        cols: launch.cols,
        unroll: launch.unroll,
    })
}

/// The launch shape [`dense_gemv_launch`] picks for a dense Gemv (a [`kg::Schedule::Gemv`]'s fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenseGemvLaunch {
    pub(crate) width: u32,
    pub(crate) cols: u32,
    pub(crate) unroll: u32,
}

/// The Gemv launch policy for a dense `[N, K]` contraction with `n` output columns (Card 727): `width` 256 and `unroll` 4, and `cols` 4 while that still launches at least two workgroups per
/// compute unit, else 2. A weight row is contiguous along K, so fewer `cols` give each row more lanes
/// (`256 / cols`) and a longer contiguous run per trip; the narrower fallback keeps a small `N` from starving
/// the device. Set from the card's sweep on gfx1151 over the qwen2.5-0.5b BF16 decode shapes (wgpu Gemv device
/// time per token: `cols` 64 7.1 ms, 32 6.0, 16 5.3, 8 4.9, 4 4.7; ROCm decode: 78 to 81 tok/s from `cols` 16 to
/// 2). The `{64, 32}, else 16` table the spike measured is the `[K, N]` column-tile body's, not this one's.
pub(crate) fn dense_gemv_launch(n: usize, compute_units: u32) -> DenseGemvLaunch {
    let min_workgroups = 2 * compute_units as usize;
    let cols = if n.div_ceil(4) >= min_workgroups {
        4
    } else {
        2
    };
    DenseGemvLaunch {
        width: 256,
        cols,
        unroll: 4,
    }
}

/// Workgroup width for the 2-D-folded one-thread-per-element imported kernels ([`OpKind::ScatterUpdate`],
/// [`OpKind::DynamicUpdateSlice`]'s runtime-index `ComputeMeta` form): always 256 (the portable Vulkan
/// max). Both kernels bake `WORKGROUP_SIZE = 256` (`pootc/tests/kernels/scatter_update.rs` /
/// `dyn_update_slice.rs`) to compute their own `x_groups`, so the dispatch must not use the generic
/// 64-or-256 wg-bump.
pub const ELEMENTWISE_2D_WIDTH: usize = 256;

/// The 2-D workgroup grid `(x_groups, y_groups)` for a one-thread-per-element kernel at
/// [`ELEMENTWISE_2D_WIDTH`] lanes/workgroup over `out_numel` elements. Like [`gemv_grid`] but each
/// workgroup covers `ELEMENTWISE_2D_WIDTH` elements. A batched shared-pool KV scatter/DUS write can exceed
/// the `65535*256 = 16,776,960`-element 1-D ceiling (card 159: a 4-slot/1024-cap gemma4-dense local-layer
/// pool write is 16,781,312). Used by `plan_eqn`, which always bakes `workgroup_size=[256,1,1]` for these
/// ops and plans the matching `group_id = group_y*x_groups + group_x` grid.
pub fn elementwise_2d_grid(out_numel: usize, caps: &DeviceCaps) -> (usize, usize) {
    kg::fold_groups(
        out_numel.div_ceil(ELEMENTWISE_2D_WIDTH).max(1),
        caps.max_grid[0],
    )
}

/// LDS tile edge for the wgpu prefill tiled GEMM (`tiled_gemm`). A workgroup of
/// `GEMM_TILE*GEMM_TILE` lanes computes a `(2*GEMM_TILE) x GEMM_TILE` output tile (each lane computes 2
/// output rows). 8 gives 64 lanes, one RADV wave64: a 256-lane workgroup miscomputes on this wgpu/RADV
/// stack (upper waves' LDS writes are not visible after the barrier; output rows come back ~0) even
/// though the SPIR-V is valid. A bigger tile needs that bug fixed. Used by `plan_eqn` for the body and the
/// planned grid.
pub const GEMM_TILE: usize = 8;

/// Workgroup width for [`kg::scatter_update_chunked_dt`] chunks; matches the imported
/// `scatter_update.kir.json`'s baked `workgroup_size` ([64,1,1]) so a chunk's `groups` and dispatch agree.
pub(crate) const SCATTER_UPDATE_WG: usize = 64;

/// Plan a `ScatterUpdate` (card 059: `base[POOL,rest]`, `src[N,rest]`, `inv[POOL]` -> `out[POOL,rest]`).
/// It is `O(POOL*rest)`: one thread per output element, since every pool row is passed through from `base`
/// or overwritten from `src`. At or below `caps.watchdog_budget`'s `scatter_update_work` elements, or on
/// a device with no watchdog budget, it is the single imported `Plan::Compute`. Otherwise it splits into
/// `scatter_update_chunked_dt` dispatches over disjoint element ranges of the same output (card 159: a
/// `pool_slots~=1025, Hkv=16, D=256` scatter is ~4.2M elements, over `poot-gpu`'s 4M `flush_work` cap
/// alone; the batched gemma4-dense pool prefill's per-layer KV write reaches millions of elements twice
/// per layer x 60 layers, so the box's own wgpu/RADV device keeps each chunk to 1M, a small fraction of
/// that cap, so chunks batch with their neighbors).
pub(crate) fn scatter_update_plan(
    site: Site<'_>,
    backend: Backend,
    base_shape: &[usize],
    out_numel: usize,
    odt: DType,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    let rest = numel(&base_shape[1..]);
    let use_imported = odt == DType::F32;
    let Some(chunk_work) = caps
        .watchdog_budget
        .map(|w| w.scatter_update_work as usize)
        .filter(|&budget| out_numel > budget)
    else {
        // The imported F32 body always bakes the ELEMENTWISE_2D_WIDTH-wide fold (card 159 Inc 3); the
        // kernelgen dt-generic fallback (non-F32) is the plain one-thread-per-output default.
        if use_imported {
            let (x, y) = elementwise_2d_grid(out_numel, caps);
            return Ok(Planned::imported(
                ImportedKernel::ScatterUpdate,
                backend,
                None,
                [(x * ELEMENTWISE_2D_WIDTH) as u32, y as u32, 1],
            ));
        }
        return Planned::generated(
            site,
            KernelRequest::Movement(MovementSpec::ScatterUpdate {
                dt: fty_ty(odt),
                rest,
                numel: out_numel,
            }),
            backend,
            caps,
        );
    };
    // Large pool on wgpu: the imported kernel has no elem_offset param, so chunks use kernelgen.
    let chunk_elems = (chunk_work / SCATTER_UPDATE_WG).max(1) * SCATTER_UPDATE_WG;
    let n_chunks = out_numel.div_ceil(chunk_elems);
    let chunks = (0..n_chunks)
        .map(|ci| {
            let offset = ci * chunk_elems;
            let elems = (out_numel - offset).min(chunk_elems);
            Planned::generated_chunk(
                site,
                KernelRequest::Movement(MovementSpec::ScatterUpdateChunk {
                    dt: fty_ty(odt),
                    rest,
                    offset,
                    groups: elems.div_ceil(SCATTER_UPDATE_WG),
                    width: SCATTER_UPDATE_WG,
                }),
                backend,
                caps,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Planned::chunked(chunks))
}

// FLASH_LDS_CAP / FLASH_PREFILL_LDS_CAP: both size the imported flash-decode/flash-prefill kernels' LDS
// output scratch `o[D]` (spec 055). One owner, `poot_target` (card 522); the attention kernel choice
// (`planner/attention.rs`, Card 557) imports both directly.

/// Lane width (workgroup size) for the LDS-cooperative synthesized flash decode (card 143,
/// `kernelgen::flash_region_decode`). Used by `plan_eqn` for both the `[w,1,1]` workgroup and the planned
/// `B*Hq*w`-thread grid (`B*Hq` workgroups).
///
/// Each lane stages a partial online-softmax triple `(m, l, o[D])` in LDS: `w*(D+2)` f32. wgpu's default
/// workgroup-storage limit is 16352 bytes (4088 f32, `downlevel_defaults()`, see `Context::new`); the
/// budget is 2048 f32, half of that, and width is capped at 64 lanes (more lanes only lengthen the serial
/// LDS fold in the epilogue). Head dims 64 and 128 give w=30 and w=15; D<=6 clamps to 64.
pub fn flash_decode_width(d: usize) -> usize {
    const LDS_BUDGET_F32: usize = 2048;
    (LDS_BUDGET_F32 / (d + 2)).clamp(1, 64)
}

/// Is this `MatMul` a candidate for the generated tiled GEMM (`kg::tiled_region`, card 099b), the
/// contraction choice `planner::matmul` makes first (Card 557; it was the `tile_matmuls` retag)? Backend
/// agnostic: output `[..,M,N]` with M>1 (M==1 is the decode GEMV), F32 operands, and a rank-2 `[K,N]`
/// shared weight (input 1). BF16 operands stay off it so the WMMA / tensor-core arms see them, and F16
/// operands because the generated body is f32-only (its LDS tiles and buffer reads are all `Ty::F32`; an
/// F16 matmul takes the dtype-generic serial kernel). The dispatch-cap chunking is the choice's own.
pub(crate) fn is_tileable_matmul_in(g: GraphTables<'_>, eqn: &Eqn) -> bool {
    if !matches!(eqn.op, OpKind::MatMul) {
        return false;
    }
    let out = &g.aval(eqn.out).shape;
    let r = out.len();
    if r < 2 || out[r - 2] <= 1 {
        return false;
    }
    let ids = value_ids(eqn);
    ids.len() == 2
        && ids.iter().all(|&id| g.aval(id).dtype == DType::F32)
        && g.aval(ids[1]).shape.len() == 2
}

/// Does this MatMul take the wgpu prefill tiled-GEMM path (`tiled_gemm`)? Like `is_decode_gemv_in` for M>1:
/// a `MatMul`/`MatMulBias` whose weight (`inputs[1]`) is rank-2 `[K,N]`. The launch is
/// `ceil(M/ts)*ceil(N/ts)` workgroups (chunked on a device with a confirmed ceiling,
/// `caps.known_miscompiles.tiled_gemm_max_workgroups`, card 095).
///
/// Card 671/ADR-0113: same as `is_decode_gemv_in`'s doc - the `pub` generic-over-`V` wrapper had no
/// production caller and was deleted; cross-crate tests assert on the planned `Plan`'s key instead.
pub(crate) fn is_tiled_gemm_in(
    g: GraphTables<'_>,
    backend: Backend,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
) -> bool {
    // The imported tiled GEMM bodies (MatMul/MatMulBias) are f32-only and fail where the codegen
    // capability query says imported LDS-array bodies do not compile (R468-015: their Rust-compiled LDS
    // arrays have a zeroinitializer in addrspace(3), which AMDGPU LLVM rejects); there the generated
    // tiled GEMM (`is_tileable_matmul_in`, WorkgroupLocalDecl with undef init) is the tiled path. BF16
    // matmul takes the tc-arm on NVPTX and matmul_batched_dt on AmdGcn.
    if g.aval(eqn.out).dtype != DType::F32
        || !matches!(eqn.op, OpKind::MatMul | OpKind::MatMulBias)
        || !codegen_capability(backend).imported_lds_array_bodies
    {
        return false;
    }
    let r = out_shape.len();
    if r < 2 {
        return false;
    }
    // Fold every leading dim into M (card 167, as in `is_tileable_dequant`): the decode tracer's spurious
    // `[batch,1,N]` middle axis (always 1) made a `shape[r-2] == 1` guard reject tiling for batched decode.
    let (m, n) = (out_numel / out_shape[r - 1].max(1), out_shape[r - 1]);
    if m == 1 {
        return false; // M==1 is the decode GEMV; M>1 only here.
    }
    // tile is (2*GEMM_TILE) rows x GEMM_TILE cols (thread-coarsened 2 rows/lane).
    let groups = m.div_ceil(2 * GEMM_TILE) * n.div_ceil(GEMM_TILE);
    // Any size is handled (the lowering chunks at the device's known-miscompile ceiling, cards 095/096);
    // only an empty launch is gated.
    if groups == 0 {
        return false;
    }
    // The weight (inputs[1]) must be rank-2 [K,N] (shared, no batch).
    let ids = value_ids(eqn);
    ids.len() >= 2 && g.aval(ids[1]).shape.len() == 2
}

/// Card 096: plan for a shared (`b_count=1`) wgpu tiled GEMM, M=`mrows` K=`kk` N=`nn`, with optional fused
/// `bias`. One dispatch below `caps.known_miscompiles`' `tiled_gemm_max_workgroups` (card 095's RADV
/// miscompile ceiling; `None` on a device with no confirmed one); otherwise `ceil(groups / (cap-1))`
/// dispatches over disjoint tile ranges (order is free), each with a per-chunk tile offset in the
/// metadata buffer.
pub(crate) fn tiled_gemm_plan(
    backend: Backend,
    mrows: usize,
    kk: usize,
    nn: usize,
    bias: bool,
    caps: &DeviceCaps,
) -> Planned {
    let ts = GEMM_TILE;
    let groups = mrows.div_ceil(2 * ts) * nn.div_ceil(ts);
    // Card 044: the imported coarsened tiled GEMM, f32. Same grid and `[a, b, (bias,) out]` buffers as
    // kernelgen's `tiled_gemm_dt`; dims ride in a `[M,K,N,offset]` metadata buffer, and one
    // shape-generic body covers every chunk.
    let kernel = if bias {
        ImportedKernel::TiledGemmBias
    } else {
        ImportedKernel::TiledGemm
    };
    let over_cap = caps
        .known_miscompiles
        .tiled_gemm_max_workgroups
        .is_some_and(|max| groups >= max as usize);
    if !over_cap {
        return Planned::imported(
            kernel,
            backend,
            Some(vec![mrows as u32, kk as u32, nn as u32, 0]),
            // GEMM_TILE^2 lanes per (2*GEMM_TILE) x GEMM_TILE output tile (see `is_tiled_gemm_in`'s doc).
            [(groups * ts * ts) as u32, 1, 1],
        );
    }
    // Card 096: the tile offset rides in the metadata buffer of each sub-cap dispatch.
    // `over_cap` is only true when a cap exists, so this is always `Some`.
    let chunk = caps.known_miscompiles.tiled_gemm_max_workgroups.unwrap() as usize - 1;
    let n_chunks = groups.div_ceil(chunk);
    let chunks = (0..n_chunks)
        .map(|i| {
            let offset = i * chunk;
            Planned::imported_chunk(
                kernel,
                backend,
                vec![mrows as u32, kk as u32, nn as u32, offset as u32],
                (groups - offset).min(chunk),
            )
        })
        .collect();
    Planned::chunked(chunks)
}

pub(crate) fn is_batched_tiled_gemm_in(
    g: GraphTables<'_>,
    backend: Backend,
    eqn: &Eqn,
    out_shape: &[usize],
    _out_numel: usize,
    caps: &DeviceCaps,
) -> bool {
    // The imported batched tiled GEMM body is f32-only and fails where the codegen capability query says
    // imported LDS-array bodies do not compile (R468-015); that target falls back to serial
    // matmul_batched_dt.
    if g.aval(eqn.out).dtype != DType::F32
        || !codegen_capability(backend).imported_lds_array_bodies
        || !matches!(eqn.op, OpKind::MatMul)
    {
        return false;
    }
    let r = out_shape.len();
    if r < 3 || out_shape[r - 2] <= 1 {
        return false;
    }
    let ids = value_ids(eqn);
    if ids.len() < 2 {
        return false;
    }
    let (a, b) = (&g.aval(ids[0]).shape, &g.aval(ids[1]).shape);
    // A, B, and out share the same leading batch dims.
    if a.len() != r || b.len() != r {
        return false;
    }
    if a[..r - 2] != out_shape[..r - 2] || b[..r - 2] != out_shape[..r - 2] {
        return false;
    }
    let (m, n, k) = (out_shape[r - 2], out_shape[r - 1], a[r - 1]);
    if a[r - 2] != m || b[r - 2] != k || b[r - 1] != n {
        return false;
    }
    if g.aval(eqn.out).dtype != DType::F32 {
        return false;
    }
    let batch: usize = out_shape[..r - 2].iter().product();
    let groups = batch * m.div_ceil(2 * GEMM_TILE) * n.div_ceil(GEMM_TILE);
    // Card 522: one owner for the tile ceiling, `caps.known_miscompiles` (card 095's RADV
    // miscompile at or above `tiled_gemm_max_workgroups`); a device with no confirmed ceiling has none.
    groups > 0
        && caps
            .known_miscompiles
            .tiled_gemm_max_workgroups
            .is_none_or(|max| groups < max as usize)
}

pub(crate) fn is_decode_attn_gemv_in(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> bool {
    if g.aval(eqn.out).dtype != DType::F32 || !matches!(eqn.op, OpKind::MatMul) {
        return false;
    }
    let r = out_shape.len();
    if r < 3 || out_shape[r - 2] != 1 || out_numel == 0 {
        return false;
    }
    let ids = value_ids(eqn);
    if ids.len() < 2 {
        return false;
    }
    let (a, b) = (&g.aval(ids[0]).shape, &g.aval(ids[1]).shape);
    if a.len() != r || b.len() != r {
        return false;
    }
    // A, B, and out share the same leading batch dims (repeat_kv already expanded K/V to Hq).
    if a[..r - 2] != out_shape[..r - 2] || b[..r - 2] != out_shape[..r - 2] {
        return false;
    }
    let (n, k) = (out_shape[r - 1], a[r - 1]);
    if a[r - 2] != 1 || b[r - 2] != k || b[r - 1] != n {
        return false;
    }
    // Exclude Q @ K^T (see doc above).
    if matches!(
        g.eqns.iter().find(|e| e.out == ids[1]).map(|e| &e.op),
        Some(OpKind::Transpose { .. })
    ) {
        return false;
    }
    // One workgroup per output element on a 2-D grid (gemv_lds); Y must clear the target's grid cap.
    gemv_grid(out_numel, caps).1 <= caps.max_grid[1] as usize
}

/// Does this `IndexedMatMul` take the wgpu indexed LDS-GEMV path ([`kernelgen::indexed_gemv_lds`])? The
/// gather-free sparse-MoE decode GEMV: SpirvVulkan, f32, with `out_numel = M*N <= caps.max_grid[0]` (one
/// workgroup per output element, within the target's own `gridDim.x` cap - never the
/// wgpu-specific `65535` literal). Otherwise the naive one-thread-per-output `indexed_matmul_dt`.
pub(crate) fn is_indexed_gemv_in(
    g: GraphTables<'_>,
    backend: Backend,
    eqn: &Eqn,
    out_numel: usize,
    caps: &DeviceCaps,
) -> bool {
    backend == Backend::SpirvVulkan
        && matches!(eqn.op, OpKind::IndexedMatMul)
        && out_numel <= caps.max_grid[0] as usize
        && g.aval(eqn.out).dtype == DType::F32
}

pub(crate) fn is_elementwise_2d_in(
    g: GraphTables<'_>,
    backend: Backend,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> bool {
    // ScatterUpdate/DynamicUpdateSlice/Transpose/Slice/Broadcast/Concat use F32-only imported kernels
    // (`ImportedKernel::IndexRemap`, `ImportedKernel::Concat2`, scatter/DUS bodies). Binary/Unary/MatMul build
    // dtype-parameterized kernelgen bodies (`plan_eqn` picks the dtype from the output), so a BF16/F16 op
    // past the `caps.max_grid[0] * ELEMENTWISE_2D_WIDTH` ceiling needs the fold too or it hits the same
    // `GridCap` crash. The ceiling is the target's own cap, never the wgpu-specific
    // `65535` literal: a fixture with a smaller `max_grid` must fold sooner.
    let f32_only = g.aval(eqn.out).dtype != DType::F32;
    let fold_ceiling = caps.max_grid[0] as usize * ELEMENTWISE_2D_WIDTH;
    match &eqn.op {
        OpKind::ScatterUpdate => !f32_only,
        OpKind::DynamicUpdateSlice { .. } => {
            !f32_only && matches!(eqn.inputs[2], Operand::Value(_))
        }
        // Materializing path only (a promoted `Plan::View` never launches), above the wg-bump ceiling.
        OpKind::Transpose { .. } | OpKind::Slice { .. } | OpKind::Broadcast { .. } => {
            !f32_only && out_numel > fold_ceiling
        }
        // Binary (broadcast-value and scalar/literal forms), any dtype.
        OpKind::Binary(_) | OpKind::Select => out_numel > fold_ceiling,
        // Unary (e.g. `Exp` over the full attention-score tensor), any dtype.
        OpKind::Unary(_) => out_numel > fold_ceiling,
        // An F32 or (Card 1007) packed-F16 contraction over a checkpoint-order weight: the one-thread-per-output
        // fallback when neither the GEMV nor the tiled GEMM is taken. The tiled GEMM is a `custom_grid` that
        // `finalize` already skips.
        OpKind::DenseContraction {
            weight: DType::F32 | DType::F16,
        } => {
            out_numel > fold_ceiling
                && dense_decode_gemv_in(eqn, out_shape, out_numel, caps).is_none()
        }
        // Naive fallback only, when every specialized MatMul plan declines (mirrors `plan_eqn`'s routing).
        OpKind::MatMul | OpKind::MatMulBias => {
            out_numel > fold_ceiling
                && !is_decode_gemv_in(g, backend, eqn, out_shape, out_numel, caps)
                && !is_tiled_gemm_in(g, backend, eqn, out_shape, out_numel)
                && !is_batched_tiled_gemm_in(g, backend, eqn, out_shape, out_numel, caps)
                && !is_decode_attn_gemv_in(g, eqn, out_shape, out_numel, caps)
                && !matmul_takes_tensorcore(g, backend, eqn, out_shape)
        }
        // Concat's 2-input imported path only (`ImportedKernel::Concat2`; the N-input fallback is kernelgen).
        // Measured: gemma4-dense batched decode's shared-pool KV gather (`gather_shared_pool`) folds rows via
        // `concat(acc, row)`; at n_slots=4/cap=1024 a local-layer fold is `4*16*1024*256 = 16,777,216`
        // elements (`GridCap([65536,1,1])`, card 159).
        OpKind::Concat { .. } => !f32_only && eqn.inputs.len() == 2 && out_numel > fold_ceiling,
        _ => false,
    }
}

/// The `M` (activation rows per block) a `PackedContraction` equation's own shapes imply: `numel/k`
/// for a dense (`blocks == 1`) contraction, else `activation_shape[1]` (`OpKind::PackedContraction`'s
/// own rank-3 `[blocks, M, K]` shape, `op.rs`'s `infer`), which `packed_dequant::try_plan_contraction`'s
/// schedule choice reads. The launch it plans is exempt from the workgroup bump by its request type
/// (Card 727), not by re-deriving this shape.
pub(crate) fn packed_contraction_m_per_block(
    g: GraphTables<'_>,
    eqn: &Eqn,
    descriptor: poot_quant::PackedWeight,
    blocks: usize,
) -> usize {
    let ids = value_ids(eqn);
    let activation_shape = &g.aval(ids[0]).shape;
    let k = descriptor.shape()[1];
    if blocks == 1 {
        activation_shape.iter().product::<usize>() / k
    } else {
        activation_shape[1]
    }
}

/// Does this `MatMul` eqn take a WMMA/tensor-core plan (AMD RDNA3 or NVPTX)? Factored out of
/// [`is_elementwise_2d`] (card 181) so it can exclude the tensor-core path without duplicating
/// `plan_eqn`'s gate (`amd_tc`/`tc`). `MatMulBias` never takes tensor cores (no WMMA bias epilogue), so
/// only `MatMul`'s bf16-operand and 16-aligned-dims gates are checked.
fn matmul_takes_tensorcore(
    g: GraphTables<'_>,
    backend: Backend,
    eqn: &Eqn,
    out_shape: &[usize],
) -> bool {
    if !matches!(eqn.op, OpKind::MatMul) {
        return false;
    }
    let ids = value_ids(eqn);
    if ids.len() < 2 {
        return false;
    }
    let a = g.aval(ids[0]).shape.clone();
    let operands_bf16 = g.aval(ids[0]).dtype == DType::BF16 && g.aval(ids[1]).dtype == DType::BF16;
    if !operands_bf16 {
        return false;
    }
    let r = out_shape.len();
    if r < 2 || a.is_empty() {
        return false;
    }
    let (mm, nn, kk) = (out_shape[r - 2], out_shape[r - 1], a[a.len() - 1]);
    let aligned = mm.is_multiple_of(16) && nn.is_multiple_of(16) && kk.is_multiple_of(16);
    if !aligned {
        return false;
    }
    match backend {
        Backend::Nvptx => true,
        Backend::AmdGcn(arch) => {
            arch.tensor_core == TensorCoreSupport::Wmma16x16x16Rdna3
                && out_shape[..r - 2].iter().product::<usize>() == 1
        }
        _ => false,
    }
}

/// The launch grid (threads) of an imported one-thread-per-output-element kernel, folded onto the
/// [`ELEMENTWISE_2D_WIDTH`] 2-D grid when [`is_elementwise_2d_in`] says this eqn's kernel bakes that
/// fold (card 159; past the wgpu 65535-workgroup cap, or an imported kernel that always bakes it). A
/// generated kernel gets the same grid from `generate`, through [`elementwise_fold`].
pub(crate) fn default_grid_in(
    g: GraphTables<'_>,
    backend: Backend,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> [u32; 3] {
    if is_elementwise_2d_in(g, backend, eqn, out_shape, out_numel, caps) {
        let (x, y) = elementwise_2d_grid(out_numel, caps);
        [(x * ELEMENTWISE_2D_WIDTH) as u32, y as u32, 1]
    } else {
        [out_numel as u32, 1, 1]
    }
}

/// The 2-D fold decision of a generated one-thread-per-output-element kernel: the planner decides
/// whether this equation folds ([`is_elementwise_2d_in`]), and `generate` derives the body's `x_groups`
/// and the launch grid from it.
pub(crate) fn elementwise_fold(
    g: GraphTables<'_>,
    backend: Backend,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> kg::Fold {
    kg::Fold {
        numel: out_numel,
        two_d: is_elementwise_2d_in(g, backend, eqn, out_shape, out_numel, caps),
        width: ELEMENTWISE_2D_WIDTH,
    }
}

/// Workgroup width (lanes per row) for the parallel [`OpKind::FusedRow`] kernel: at most 128, and no more
/// than the row width. Shared by `plan_eqn` and the `FusedRow` launch grid it bakes alongside it.
pub fn row_parallel_width(n_cols: usize) -> usize {
    n_cols.clamp(1, 128)
}
