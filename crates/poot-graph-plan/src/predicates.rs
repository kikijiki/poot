//! Per-eqn routing: the strided-view promotion pass (`compute_views`, spec 132) and the dispatch
//! predicates and launch-grid math (`is_decode_gemv`, `is_tileable_matmul`, `is_batched_tileable_matmul`,
//! `is_decode_attn_gemv`, `is_elementwise_2d`, [`default_grid_in`], ...). The planner computes each
//! eqn's launch grid once, alongside the kernel choice it shapes, and stores it in the [`Plan`]
//! (card 525); no separate dispatch-time function re-derives it. Also the `*_plan` helpers that build
//! a chunked [`Plan`] for ops whose dispatch can exceed a single-launch cap (RADV 2^15-tile miscompile,
//! iGPU display-watchdog TDR; see each function's doc).

use crate::planner::Planned;
use crate::refusal::Site;
use poot_kernelgen::Layout;
use poot_tensor::DType;
use std::collections::HashMap;

use poot_graph_ir::op::{ContractionHead, OpKind};
use poot_graph_ir::{Eqn, Graph, Operand, TensorType, ValueId, ValueMeta, ValueTable};
use poot_kernelgen as kg;
use poot_target::{Backend, DeviceCaps, TargetId};

use crate::PlanError;
use crate::planner::PlannedDispatch;
use crate::{Binding, Sink, Source};
use poot_kernelgen::{ContractionSpec, KernelRequest};

/// The codegen capability query for the target (card 522, R468-014). `Backend`'s `AmdGcn` payload already
/// carries the same `AmdArch` `poot_codegen::Target::AmdGcn` takes, so the conversion is a bare re-tag, not
/// a new dependency.
fn codegen_capability(target: poot_codegen::Target) -> poot_codegen::Capability {
    poot_codegen::capability(target)
}

/// The activation read a [`kg::Schedule::Gemv`] of `unroll`-element runs over activation rows of `k` values takes on
/// `backend` (POOT-1405). [`kg::GemvActivation::Vec4`] when the target's codegen reads one param both in aligned groups
/// and element by element ([`poot_codegen::Capability::mixed_vector_reads`]) and every full run starts on a group: row
/// `b` starts at element `b * k` and a full run at a multiple of `unroll` inside it, so `k` and `unroll` both multiples
/// of four put every full run on a 16-byte boundary of a buffer the executor binds 16-byte aligned (the binding
/// contract in `poot-executor`). Else [`kg::GemvActivation::Scalar`]. Both read the same values into the same products
/// in the same order, so the choice never changes an output bit.
pub(crate) fn gemv_activation(backend: Backend, k: usize, unroll: u32) -> kg::GemvActivation {
    let group = kg::GemvActivation::VEC4_LANES;
    let aligned = k.is_multiple_of(group) && (unroll as usize).is_multiple_of(group);
    if aligned && codegen_capability(backend.into()).mixed_vector_reads {
        kg::GemvActivation::Vec4
    } else {
        kg::GemvActivation::Scalar
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

/// Element count of a tensor of `shape`: the product of its extents, floored at one so a scalar (`[]`) and a zero-extent shape both count as a single element.
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
    pub(crate) values: &'g ValueTable<ValueMeta>,
    pub(crate) eqns: &'g [Eqn],
}

impl<'g> GraphTables<'g> {
    pub(crate) fn aval(self, id: ValueId) -> &'g TensorType {
        &self.values[id].aval
    }
}

impl<'g> From<&'g Graph> for GraphTables<'g> {
    fn from(graph: &'g Graph) -> Self {
        Self {
            values: graph.values(),
            eqns: graph.eqns(),
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
/// Every pointwise op can: an elementwise `Binary` (against another value or a literal) and a `Unary` are
/// pointwise regions of one step, `Fused` / `FusedRow` are regions of many, and the pointwise synthesizer
/// (`fused_views`) reads each leaf through its own layout. The indexed contraction reads its weight operand
/// (operand 1 only) strided, so a `Transpose` of the stacked expert weight is a view there. Everything else
/// (other contractions, gather/scatter, concat, standalone `Reduce`, `Reshape`, `DynamicUpdateSlice`) keeps
/// materializing. The qwen2 decode graph's RoPE `rotate_half` (`concat(neg(slice), slice)`) is the case this
/// exists for.
fn is_strided_capable(eqn: &Eqn, value: ValueId) -> bool {
    match &eqn.op {
        OpKind::Binary(_) | OpKind::Unary(_) | OpKind::Fused(_) | OpKind::FusedRow(_) => true,
        // The indexed contraction reads its stacked weight (operand 1 only) through a transposed layout, so a
        // `Transpose` feeding it is never materialized. Its activation, ids and epilogue leaves are read flat.
        OpKind::IndexedMatMul
        | OpKind::ContractionEpilogue {
            head: ContractionHead::Indexed,
            ..
        } => reads_as_indexed_weight(eqn, value),
        _ => false,
    }
}

/// Whether `eqn` is an indexed contraction that reads `value` as its stacked weight and nowhere else.
fn reads_as_indexed_weight(eqn: &Eqn, value: ValueId) -> bool {
    matches!(
        eqn.op,
        OpKind::IndexedMatMul
            | OpKind::ContractionEpilogue {
                head: ContractionHead::Indexed,
                ..
            }
    ) && matches!(eqn.inputs.get(1), Some(Operand::Value(weight)) if *weight == value)
        && value_ids(eqn).iter().filter(|&&v| v == value).count() == 1
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
///   output (not pinned, F32, or BF16 read only as an indexed contraction's weight) is promoted when every
///   consumer is `is_strided_capable` or is itself a
///   movement-view candidate already promoted (consumers are visited first in reverse order).
/// - Pass B walks forward and composes each promoted value's Layout onto its input's Layout
///   (`Layout::contiguous` if the input was not promoted). A chain tail therefore has a Layout expressed
///   in terms of the original contiguous source, matching `poot_eval::apply_movement` over the whole
///   chain (see the `view_..._chain_matches_oracle_fuzz` test).
///
/// A value with no consumer is not promoted. A pinned value, a non-F32 dtype, or a non-strided-capable leaf
/// consumer down the chain keeps the rest of the chain materialized. Unpromoted ops keep the
/// metadata-launch index-remap copy.
///
/// Every pinned value ([`Graph::pinned_values`]: the output, each validation value, and both sides of each
/// state pair) keeps its own materialized buffer, so a validation packet can copy it row-major from offset 0.
pub fn compute_views(g: &Graph, target: impl Into<TargetId>) -> HashMap<ValueId, Layout> {
    let mut views: HashMap<ValueId, Layout> = HashMap::new();
    // A target that compiles no kernel bodies (a VPU) has no strided-view bodies to read a view through.
    let strided = poot_codegen::Target::try_from(target.into())
        .is_ok_and(|target| codegen_capability(target).strided_views);
    if !strided {
        return views;
    }
    let cons = consumers(g.into());
    let pinned: std::collections::HashSet<ValueId> = g.pinned_values().collect();

    // Pass A (reverse topological order): decide the promoted set (no strides yet).
    let mut promoted: std::collections::HashSet<ValueId> = std::collections::HashSet::new();
    for eqn in g.eqns().iter().rev() {
        let out = eqn.out;
        if pinned.contains(&out) || !is_movement_view_candidate(&eqn.op) {
            continue;
        }
        // A strided view is read element by element, so any f32 reader that takes strides reads it. A BF16 view
        // is only ever the packed expert stack an indexed contraction reads in place; its other readers have no
        // BF16 lane.
        //
        // An I32 view is read only by float pointwise regions (a region whose output is F32 reads an I32 leaf
        // through its layout like any other): a cast step or an exact step of the region consumes the word in
        // place, where a materialized I32 copy would cost a dispatch. An exact-I32 reader keeps its own plan
        // for an I32 movement.
        let admitted = match g.aval(out).dtype {
            DType::F32 => true,
            DType::I32 => cons.get(&out).is_some_and(|users| {
                users.iter().all(|&i| {
                    matches!(g.eqns()[i].op, OpKind::Fused(_))
                        && g.aval(g.eqns()[i].out).dtype == DType::F32
                })
            }),
            DType::BF16 => cons.get(&out).is_some_and(|users| {
                users
                    .iter()
                    .all(|&i| reads_as_indexed_weight(&g.eqns()[i], out))
            }),
            _ => false,
        };
        if !admitted {
            continue;
        }
        let all_strided_capable = cons.get(&out).is_some_and(|users| {
            users.iter().all(|&i| {
                let c = &g.eqns()[i];
                is_strided_capable(c, out)
                    || (is_movement_view_candidate(&c.op) && promoted.contains(&c.out))
            })
        });
        if all_strided_capable {
            promoted.insert(out);
        }
    }

    // Pass B (forward topological order): compose each promoted value's Layout from its input's Layout.
    for eqn in g.eqns() {
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
pub(crate) const GEMV_WIDTH: usize = 128;

/// Does this MatMul take the decode-GEMV path (`kernelgen::gemv_lds`)? Requires M==1 (`out_shape[-2]==1`)
/// and a rank-2 `[K,N]` weight (`inputs[1]`), which separates a projection GEMV (`a[B,1,K] @ W[K,N]`) from
/// a per-head attention matmul (`q[B,Hq,1,D] @ k[B,Hkv,cap,D]`, batched weight). `out_numel = B*N` must fit
/// the 2-D workgroup grid ([`gemv_grid`]).
pub(crate) fn is_decode_gemv_in(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> bool {
    if !matches!(eqn.op, OpKind::MatMul { .. }) {
        return false;
    }
    // The decode GEMV is f32; bf16 decode on NVPTX/AmdGcn takes the tensor-core path.
    if g.aval(eqn.out).dtype != DType::F32 {
        return false;
    }
    let r = out_shape.len();
    if r < 2 || out_shape[r - 2] != 1 || out_numel == 0 {
        return false;
    }
    // One workgroup per output element on a 2-D grid (`gemv_lds`'s `col = GroupY*x_groups+GroupX`), so the Y
    // cap is on the output count; reject only if that would overflow.
    if gemv_grid(out_numel, caps).1 > caps.max_grid[1] as usize {
        return false;
    }
    // A device can have a measured width above which the batched matmul beats the GEMV (NVIDIA Ampere: the GEMV's
    // serial lane-0 LDS reduction and stride-N weight reads make it ~2x slower for large decode projections,
    // gate/up/down/lm_head with N in the thousands). A device with no measurement keeps the GEMV at any width.
    if caps
        .decode_gemv_max_out_cols
        .is_some_and(|max| out_shape[out_shape.len() - 1] > max as usize)
    {
        return false;
    }
    // The weight (inputs[1]) must be rank-2 [K,N].
    let ids = value_ids(eqn);
    ids.len() >= 2 && g.aval(ids[1]).shape.len() == 2
}

/// The 2-D workgroup grid `(x_groups, y_groups)` for a decode GEMV over `out_numel` elements (one workgroup
/// per element). X fills up to `caps.max_grid[0]` (the planner reads the target's
/// own measured cap, never a baked wgpu-specific literal) and the remainder spills onto Y, so a
/// ~150k-vocab `lm_head` fits. For `out_numel <= caps.max_grid[0]` this is `(out_numel, 1)`. Used by
/// `plan_eqn`, which bakes `x_groups` into the kernel and the planned grid.
pub(crate) fn gemv_grid(out_numel: usize, caps: &DeviceCaps) -> (usize, usize) {
    kg::fold_groups(out_numel, caps.max_grid[0])
}

/// Plan a decode-GEMV (`is_decode_gemv`'s `M==1` shared-weight matmul): the one generated `kernelgen::gemv_lds`
/// body, one workgroup of [`GEMV_WIDTH`] lanes per output element. Below `caps.watchdog_budget`'s
/// `decode_gemv_out_elems` output elements, or on a device with no watchdog budget at all (no display
/// watchdog off wgpu/RADV, card 163), it is one dispatch over the whole output on a 2-D grid. Above the
/// trigger it splits into dispatches of that same body over disjoint ranges of the same output buffer
/// (`offset` baked per range). The trigger changes the launch, never the body or its summation order.
/// `chunk_elems` is capped at `caps.max_grid[0]` because a split plan's dispatch is a single 1-D workgroup
/// count. `k`/`n` are the weight's `[K,N]` dims.
///
/// A single dispatch at BLOOM's scale (`N=250880, K=1024`) completes in ~0.6s alone, but the same dispatch
/// right after another large one (a materializing `transpose` of the same weight, ~3.7s) trips the Strix
/// Halo iGPU's display-watchdog TDR within ~1.5s, even though the deleted `GpuExecutor::run` polled after every submit.
/// The trigger is the total busy stretch across a rapid sequence of large dispatches, not one dispatch's
/// duration. Splitting gives display work scheduling gaps.
///
/// `N` is the dominant hazard axis: BLOOM `N=250880` hangs; gpt-oss `N=201088` (tied, `K=32`) and SmolLM3
/// `N=128256` (tied, `K=2048`, higher `out_numel*K` than BLOOM's `2.57e8`) do not. So the trigger is
/// element-count-only (unlike the K-scaled `serial_dequant_work`), and the box's own wgpu/RADV device is
/// measured at 220_000 (`DeviceCaps::wgpu_rdna3_igpu`), between the proven-safe 201088 and the
/// proven-hanging 250880.
pub(crate) fn decode_gemv_plan(
    site: Site<'_>,
    backend: Backend,
    k: usize,
    n: usize,
    out_numel: usize,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    let request = |elems, offset| {
        KernelRequest::Contraction(ContractionSpec::GemvChunk {
            k,
            n,
            width: GEMV_WIDTH,
            elems,
            offset,
        })
    };
    let over_trigger = caps
        .watchdog_budget
        .is_some_and(|w| out_numel > w.decode_gemv_out_elems as usize);
    if !over_trigger {
        return Planned::generated(site, request(out_numel, 0), backend, caps);
    }
    gemv_chunks(site, backend, out_numel, request, caps)
}

/// A decode GEMV as `kg::gemv_lds` dispatches over disjoint output ranges of one buffer: one workgroup of
/// [`GEMV_WIDTH`] lanes per output element, at most `caps.max_grid[0]` of them per dispatch (a split plan's dispatch is a
/// single 1-D workgroup count). `request(elems, offset)` builds one dispatch's request, so the weight layout is
/// the caller's.
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
            Planned::generated_dispatch(site, request(elems, offset), backend, caps)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Planned::split(chunks)
}

/// The launch a `DenseContraction` decode GEMV runs as a [`kg::Schedule::Gemv`] (M == 1 once every leading dim folds
/// into M), or `None` when the equation is not one or its launch would not fit the device's X grid. Every weight
/// dtype the fold admits (F32, packed F16, packed BF16) and every backend take the same generated body; the
/// launch comes from [`dense_gemv_launch`], the activation read from the planner's [`gemv_activation`]
/// ([`DenseGemvLaunch::schedule`]).
pub(crate) fn dense_decode_gemv_in(
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> Option<DenseGemvLaunch> {
    eqn.op.dense_weight()?;
    let n = *out_shape.last()?;
    if out_shape.len() < 2 || n == 0 || out_numel != n {
        return None;
    }
    let launch = dense_gemv_launch(n, caps.compute_units);
    (n.div_ceil(launch.cols as usize) <= caps.max_grid[0] as usize).then_some(launch)
}

/// The launch shape [`dense_gemv_launch`] picks for a dense Gemv (a [`kg::Schedule::Gemv`]'s fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenseGemvLaunch {
    pub(crate) width: u32,
    pub(crate) cols: u32,
    pub(crate) unroll: u32,
}

impl DenseGemvLaunch {
    /// The Gemv of this launch over activation rows of `k` values on `backend`, its activation read from
    /// [`gemv_activation`].
    pub(crate) fn schedule(self, backend: Backend, k: usize) -> kg::Schedule {
        kg::Schedule::Gemv {
            width: self.width,
            cols: self.cols,
            unroll: self.unroll,
            activation: gemv_activation(backend, k, self.unroll),
        }
    }
}

/// The Gemv launch policy for a dense `[N, K]` contraction with `n` output columns (Card 727): `width` 256 and `unroll` 4, and `cols` 4 while that still launches at least two workgroups per
/// compute unit, else 2. A weight row is contiguous along K, so fewer `cols` give each row more lanes
/// (`256 / cols`) and a longer contiguous run per trip; the narrower fallback keeps a small `N` from starving
/// the device. Set from the card's sweep on gfx1151 over the qwen2.5-0.5b BF16 decode shapes (wgpu Gemv device
/// time per token: `cols` 64 7.1 ms, 32 6.0, 16 5.3, 8 4.9, 4 4.7; ROCm decode: 78 to 81 tok/s from `cols` 16 to
/// 2). The `{64, 32}, else 16` table the spike measured is the `[K, N]` column-tile body's, not this one's.
pub(crate) fn dense_gemv_launch(n: usize, compute_units: u32) -> DenseGemvLaunch {
    let min_workgroups = DENSE_GEMV_WORKGROUPS_PER_CU * compute_units as usize;
    let cols = if n.div_ceil(DENSE_GEMV_WIDE_COLS as usize) >= min_workgroups {
        DENSE_GEMV_WIDE_COLS
    } else {
        DENSE_GEMV_NARROW_COLS
    };
    DenseGemvLaunch {
        width: DENSE_GEMV_WIDTH,
        cols,
        unroll: DENSE_GEMV_UNROLL,
    }
}

/// Lanes in a dense Gemv workgroup ([`dense_gemv_launch`]).
const DENSE_GEMV_WIDTH: u32 = 256;
/// Contiguous `K` elements a dense Gemv lane reads per trip.
const DENSE_GEMV_UNROLL: u32 = 4;
/// Output columns a dense Gemv workgroup owns while `N` still fills the device.
const DENSE_GEMV_WIDE_COLS: u32 = 4;
/// Output columns a dense Gemv workgroup owns for a small `N`, so it does not starve the device.
const DENSE_GEMV_NARROW_COLS: u32 = 2;
/// Workgroups per compute unit the wide column count must still launch.
const DENSE_GEMV_WORKGROUPS_PER_CU: usize = 2;

/// Workgroup width of a one-thread-per-element kernel folded onto a 2-D grid: always 256 (the portable
/// Vulkan max). A folded body derives its global index from this width, so the dispatch must not use the
/// generic 64-or-256 wg-bump.
pub(crate) const ELEMENTWISE_2D_WIDTH: usize = 256;

/// The workgroup width a scatter-update chunk is sized in: a generated body's default width, so a chunk's
/// thread count and its dispatch agree.
pub(crate) const SCATTER_UPDATE_WG: usize = 64;

/// Plan a `ScatterUpdate` (`base[POOL,rest]`, `src[N,rest]`, `slots[N]` -> `out[POOL,rest]`). The output
/// is the pool itself, donated in place ([`crate::state_commit`] refuses a graph that cannot), so the
/// kernel is `O(N*rest)`: one thread per source unit, reading `src` and `slots` and storing into the named
/// slot. The base is never read, so the dispatch binds `[src, slots] -> out`. At or below
/// `caps.watchdog_budget`'s `scatter_update_work` source units, or on a device with no watchdog budget,
/// it is one dispatch. Otherwise it splits into dispatches over disjoint source-unit windows (card 159: the
/// per-layer KV write of a wide batched prefill is millions of elements, over `poot-wgpu-device`'s 4M
/// `flush_work` cap alone, so the box's own wgpu/RADV device keeps each chunk to 1M, a small fraction of
/// that cap, so chunks batch with their neighbors).
pub(crate) fn scatter_update_plan(
    site: Site<'_>,
    backend: Backend,
    storage: &kg::Storage,
    written: Option<&poot_kernel_ir::Ty>,
    out_shape: &[usize],
    src_shape: Vec<usize>,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    let units = storage.units(&src_shape);
    // The kernel's reads are the equation's `src` and `slots` operands; `base` is the output's own buffer.
    let bound = |dispatch: PlannedDispatch| {
        dispatch.bound(Binding::new(
            vec![Source::Operand(1), Source::Operand(2)],
            Sink::Out,
        ))
    };
    let spec = |window| {
        let spec = kg::MovementSpec::new(
            storage.clone(),
            out_shape.to_vec(),
            kg::MovementOp::ScatterRows {
                src_shape: src_shape.clone(),
                window,
            },
        );
        match written {
            Some(ty) => spec.writing(ty.clone()),
            None => spec,
        }
    };
    let Some(chunk_work) = caps
        .watchdog_budget
        .map(|w| w.scatter_update_work as usize)
        .filter(|&budget| units > budget)
    else {
        let fold_ceiling = caps.max_grid[0] as usize * ELEMENTWISE_2D_WIDTH;
        let whole = Planned::generated_dispatch(
            site,
            KernelRequest::Movement(spec(None).folded(units > fold_ceiling)),
            backend,
            caps,
        )?;
        return Planned::split(vec![bound(whole)]);
    };
    let chunk_units = (chunk_work / SCATTER_UPDATE_WG).max(1) * SCATTER_UPDATE_WG;
    let chunks = (0..units.div_ceil(chunk_units))
        .map(|chunk| {
            let offset = chunk * chunk_units;
            let remaining = (units - offset).min(chunk_units);
            Planned::generated_dispatch(
                site,
                KernelRequest::Movement(spec(Some(kg::UnitWindow {
                    offset,
                    threads: remaining.div_ceil(SCATTER_UPDATE_WG) * SCATTER_UPDATE_WG,
                }))),
                backend,
                caps,
            )
            .map(bound)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Planned::split(chunks)
}

/// The most lanes the streaming attention schedule uses ([`attention_lanes`]).
const MAX_LANES: usize = 64;

/// The lane count of the streaming attention schedule: the widest power of two, at most 64, whose
/// partials fit the device. Each lane stages one online-softmax triple `(m, l, o[dv])` in workgroup memory,
/// so `lanes` lanes need `4 * (lanes * dv + 2 * lanes)` bytes of `caps.lds_bytes`, and `lanes` may not
/// exceed the workgroup limits. More than 64 lanes only lengthen the serial per-lane combine. `None` when
/// not even one lane fits.
pub(crate) fn attention_lanes(dv: usize, caps: &DeviceCaps) -> Option<usize> {
    let lanes_limit = (caps.max_workgroup_size[0] as usize)
        .min(caps.max_workgroup_invocations as usize)
        .min(MAX_LANES);
    let lds_bytes = caps.lds_bytes as usize;
    std::iter::successors(Some(MAX_LANES), |lanes| (*lanes > 1).then_some(lanes / 2))
        .find(|&lanes| lanes <= lanes_limit && 4 * (lanes * dv + 2 * lanes) <= lds_bytes)
}

/// Whether a block-selected attention serves the `n_rep` query heads of a key/value head in one workgroup
/// (`rows` is `batch * hq * lq`). Grouping loads each selected key and value row once for the group, not once
/// per head, which is what a prefill chunk's rows spend their time on (Native Sparse Attention's
/// group-centric loading, arXiv 2502.11089 section 3.4); it divides the workgroups by `n_rep`, so it is kept
/// for the shapes that still fill the device afterwards, [`WORKGROUPS_PER_COMPUTE_UNIT`] groups per compute
/// unit. A few-row decode keeps a workgroup per head, whose parallelism it needs more than the shared loads.
fn groups_selected_heads(rows: usize, n_rep: usize, caps: &DeviceCaps) -> bool {
    n_rep > 1 && rows / n_rep >= caps.compute_units as usize * WORKGROUPS_PER_COMPUTE_UNIT
}

/// How a block-selected attention schedules its workgroups: whether one serves the `n_rep` heads of a key/value
/// head, and the lanes it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SelectedSchedule {
    pub(crate) grouped: bool,
    pub(crate) lanes: usize,
}

/// The schedule of a block-selected attention over `rows` (`batch * hq * lq`), or `None` when the device's
/// workgroup memory holds not even one lane of a head.
///
/// Grouping is chosen when the rows keep the device full ([`groups_selected_heads`]) and the device's
/// workgroup memory holds the `n_rep` head lanes' partials ([`selected_attention_lanes`]). A device that
/// cannot hold them plans the ungrouped schedule, a workgroup per head with the lanes a dense row gets: that
/// is the schedule the device ran before grouping existed, so the capability limit costs the shared key loads and nothing
/// else. The request records the choice (`MaskKind::Blocks { grouped }`), so it is visible in the plan.
pub(crate) fn selected_attention_schedule(
    rows: usize,
    n_rep: usize,
    dv: usize,
    caps: &DeviceCaps,
) -> Option<SelectedSchedule> {
    let grouped = groups_selected_heads(rows, n_rep, caps)
        .then(|| selected_attention_lanes(n_rep, dv, caps))
        .flatten();
    match grouped {
        Some(lanes) => Some(SelectedSchedule {
            grouped: true,
            lanes,
        }),
        None => attention_lanes(dv, caps).map(|lanes| SelectedSchedule {
            grouped: false,
            lanes,
        }),
    }
}

/// The key-walking slices of the block-selected attention schedule ([`selected_attention_lanes`]).
const SELECTED_SLICES: usize = 2;

/// The lane count of the block-selected attention schedule: `n_rep` head lanes (the query heads of one
/// key/value head, which read the same selected keys) for each of the widest [`SELECTED_SLICES`]-or-fewer
/// power-of-two number of slices whose partials fit the device (`4 * (lanes * dv + 2 * lanes)` bytes of
/// `caps.lds_bytes`, `lanes` within the workgroup limits). `None` when the head lanes of one slice do not fit.
pub(crate) fn selected_attention_lanes(
    n_rep: usize,
    dv: usize,
    caps: &DeviceCaps,
) -> Option<usize> {
    let lanes_limit =
        (caps.max_workgroup_size[0] as usize).min(caps.max_workgroup_invocations as usize);
    let lds_bytes = caps.lds_bytes as usize;
    std::iter::successors(Some(SELECTED_SLICES), |slices| {
        (*slices > 1).then_some(slices / 2)
    })
    .map(|slices| n_rep * slices)
    .find(|&lanes| lanes <= lanes_limit && 4 * (lanes * dv + 2 * lanes) <= lds_bytes)
}

/// The most key chunks one decode row is split into: past this the combine stage's per-row reduction costs
/// more than the extra workgroups hide.
const MAX_ATTENTION_SPLITS: usize = 16;
/// The fewest keys a split walks. A shorter chunk leaves a workgroup's lanes mostly idle and the launch
/// is the combine's overhead alone.
const MIN_KEYS_PER_SPLIT: usize = 128;
/// Workgroups per compute unit the split count aims at: two resident workgroups keep a CU busy while one
/// waits on its K/V loads.
const WORKGROUPS_PER_COMPUTE_UNIT: usize = 2;

/// The split-KV factor of a decode attention (`Lq == 1`), or `None` for the streaming schedule.
///
/// The streaming schedule runs `batch * heads` workgroups; at a few heads that leaves most of a many-CU
/// device idle while each workgroup walks the whole cache (the Strix Halo shape: 2 workgroups on 40 CUs).
/// Splitting the keys across `splits` workgroups and reducing the partial `(m, l, o)` states in a second
/// launch is Flash-Decoding (Dao et al., <https://pytorch.org/blog/flash-decoding/>, retrieved 2026-10-11);
/// vLLM's Triton decode attention chooses its split count the same way, from the occupancy the
/// batch-and-head grid leaves
/// (<https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py>,
/// retrieved 2026-10-11). The count follows `caps.compute_units` and the cache length, never the live
/// length (which is a run-time value: a split a short length leaves empty publishes a neutral partial).
///
/// Streaming is kept when any of these holds, so a request never needs a capability the device lacks:
/// the rows already fill the device, the cache is too short for two chunks of [`MIN_KEYS_PER_SPLIT`], the
/// partial-state scratch exceeds `caps.max_buffer_bytes`, or the launch's 2-D fold would pass `max_grid`.
/// `spec` is the streaming-schedule request being considered.
pub(crate) fn attention_splits(
    spec: &poot_kernelgen::AttentionSpec,
    caps: &DeviceCaps,
) -> Option<usize> {
    if spec.lq != 1 {
        return None;
    }
    let rows = spec.row_groups();
    let compute_units = caps.compute_units as usize;
    let wanted = (compute_units * WORKGROUPS_PER_COMPUTE_UNIT).div_ceil(rows);
    let splits = wanted
        .min(MAX_ATTENTION_SPLITS)
        .min(spec.candidates() / MIN_KEYS_PER_SPLIT);
    if splits < 2 || rows >= compute_units {
        return None;
    }
    let split = poot_kernelgen::AttentionSpec {
        schedule: poot_kernelgen::AttentionSchedule::SplitKv {
            lanes: spec.schedule.lanes(),
            splits,
        },
        ..spec.clone()
    };
    let scratch_bytes = split.partial_elems()? as u64 * 4;
    if scratch_bytes > caps.max_buffer_bytes {
        return None;
    }
    let (_, y) = poot_kernelgen::fold_groups(split.workgroups(), caps.max_grid[0]);
    (y <= caps.max_grid[1] as usize).then_some(splits)
}

/// Workgroup widths the tiled attention schedule considers, widest first: one wave64 and its multiples up to the
/// widest workgroup every GPU target allows (`attention_tiling`).
const TILED_LANES: [usize; 3] = [256, 128, 64];
/// The most tile rows (query positions times packed heads) one tiled attention workgroup serves.
const TILED_MAX_ROWS: usize = 128;
/// Keys per staged block of the tiled attention schedule, widest first. A 64-key block was measured 20-30% slower
/// than a 32-key one on ROCm and wgpu (Strix Halo) and no faster on PTX (POOT-1402 bank sweep): it doubles the key
/// tile and the score accumulators for half the barriers, and the barriers were not what bounded the kernel.
const TILED_KEY_BLOCKS: [usize; 2] = [32, 16];
/// The `f32` accumulators (score tile plus output tile) one lane of a tiled attention holds: half of the 128
/// registers below which RDNA3 keeps 12 waves per SIMD resident (1536 VGPRs per lane slot) and an NVIDIA SM 16
/// warps (65536 registers), the other half going to the step's staged operands, the addresses and the row state.
/// An 80-accumulator tile was measured slower than a 40-accumulator one on wgpu (POOT-1402 bank sweep).
const TILED_ACCUMULATORS_PER_LANE: usize = 64;
/// The widest lane tile column count a tiled attention considers: a lane's columns are one register each.
const TILED_MAX_LANE_COLS: usize = 16;

/// The register tile one lane of a `lanes`-lane workgroup holds over a `rows x cols` tile, or `None` when no
/// grid of whole tiles covers it: every `r x c` with `(rows / r) * (cols / c) == lanes`, the most square (the
/// fewest LDS reads per FMA, `(r + c) / (r * c)` at a fixed `r * c`), ties to fewer columns.
fn attention_lane_tile(rows: usize, cols: usize, lanes: usize) -> Option<poot_kernelgen::LaneTile> {
    let cells = rows * cols;
    if !cells.is_multiple_of(lanes) {
        return None;
    }
    let per_lane = cells / lanes;
    (1..=per_lane.min(rows))
        .filter(|r| rows.is_multiple_of(*r) && per_lane.is_multiple_of(*r))
        .map(|r| (r, per_lane / r))
        .filter(|&(r, c)| {
            c <= TILED_MAX_LANE_COLS && cols.is_multiple_of(c) && (rows / r) * (cols / c) == lanes
        })
        .min_by_key(|&(r, c)| (r + c, c))
        .map(|(rows, cols)| poot_kernelgen::LaneTile { rows, cols })
}

/// The tiled (FlashAttention-2) schedule of a multi-row attention, or `None` to keep the streaming one.
///
/// A tiled workgroup stages each block of keys and values once for all of its `rows = br * n_rep` tile rows, so
/// its global traffic per FLOP falls as `1 / rows` (FlashAttention-2, Dao, arXiv 2307.08691, section 3). The
/// choice maximizes that reuse within what the device holds, from the shapes and caps alone:
///
/// - the workgroup is at most the device's widest;
/// - each lane's two register tiles hold at most [`TILED_ACCUMULATORS_PER_LANE`] accumulators;
/// - the staged tiles fit `caps.lds_bytes`;
/// - the row groups (`batch * hkv * ceil(lq / br)` workgroups) give every compute unit at least one.
///
/// Among the tilings that pass, the widest workgroup wins, then the most tile rows (reuse), then the widest key
/// block (fewer barriers per FLOP). The staged tiles take most of a compute unit's workgroup memory, so one
/// workgroup is resident per compute unit and its lanes are all the latency hiding the unit has: 256 lanes over
/// 32 rows ran 30-40% faster than 128 lanes over 64 rows on ROCm (POOT-1402 bank sweep). A block-selected attention
/// and a single-row (decode) one are not tiled: their rows share no key blocks.
pub(crate) fn attention_tiling(
    spec: &poot_kernelgen::AttentionSpec,
    caps: &DeviceCaps,
) -> Option<poot_kernelgen::AttentionSchedule> {
    if spec.lq < 2 || matches!(spec.mask, poot_kernelgen::MaskKind::Blocks { .. }) {
        return None;
    }
    let lanes_limit =
        (caps.max_workgroup_size[0] as usize).min(caps.max_workgroup_invocations as usize);
    let positions = std::iter::successors(Some(1usize), |br| Some(br * 2))
        .take_while(|br| br * spec.n_rep <= TILED_MAX_ROWS && *br < 2 * spec.lq);
    let candidates = positions.flat_map(|br| {
        TILED_LANES
            .into_iter()
            .filter(move |&lanes| lanes <= lanes_limit)
            .flat_map(move |lanes| TILED_KEY_BLOCKS.map(|bc| (br, lanes, bc)))
    });
    candidates
        .filter_map(|(br, lanes, bc)| {
            let rows = br * spec.n_rep;
            let micro = poot_kernelgen::AttentionMicro {
                score: attention_lane_tile(rows, bc, lanes)?,
                out: attention_lane_tile(rows, spec.dv, lanes)?,
            };
            if micro.score.cells() + micro.out.cells() > TILED_ACCUMULATORS_PER_LANE {
                return None;
            }
            let schedule = poot_kernelgen::AttentionSchedule::Tiled {
                br,
                bc,
                lanes,
                micro,
            };
            let tiled = poot_kernelgen::AttentionSpec {
                schedule,
                ..spec.clone()
            };
            let lds = tiled.tiled_lds_bytes()?.ok()?;
            let fills = tiled.row_groups() >= caps.compute_units as usize;
            (lds <= caps.lds_bytes as usize && fills).then_some(((lanes, rows, bc), schedule))
        })
        .max_by_key(|&(rank, _)| rank)
        .map(|(_, schedule)| schedule)
}

/// The most row groups one dispatch of the streaming attention `spec` may run, or `None` when the device has no
/// watchdog budget or the whole of `spec` fits it: a row costs the keys it can see (all `lk`, the sliding
/// window's, or a block selection's) times `dk + dv`, a row group the cost of its rows, and a run of row
/// groups carries at most [`poot_target::WatchdogBudget::attention_work`] of that. A single row group heavier
/// than the budget still goes out, alone. POOT-1313: an 8192-token gemma-3-1b-it global layer was one 2 s dispatch that lost the device.
pub(crate) fn attention_rows_per_dispatch(
    spec: &poot_kernelgen::AttentionSpec,
    caps: &DeviceCaps,
) -> Option<usize> {
    let budget = caps.watchdog_budget?.attention_work;
    let group_work = attention_row_work(spec).saturating_mul(spec.rows_per_workgroup() as u64);
    let groups = usize::try_from(budget / group_work)
        .unwrap_or(usize::MAX)
        .max(1);
    (groups < spec.row_groups()).then_some(groups)
}

/// The key-value multiply-accumulates one row of `spec` can cost: the keys it can see times `dk + dv`. The keys
/// are bounded from above: a sliding layer's window, else every candidate. A live key bound still counts the
/// capacity `lk`, since the length is only known at run time, so the split is never coarser than the cache
/// needs.
fn attention_row_work(spec: &poot_kernelgen::AttentionSpec) -> u64 {
    let keys = match spec.mask {
        poot_kernelgen::MaskKind::Causal {
            window: Some(window),
        } => window.min(spec.lk),
        _ => spec.candidates(),
    };
    (keys as u64)
        .saturating_mul((spec.dk + spec.dv) as u64)
        .max(1)
}

/// The [`crate::Dispatch::work`] weight, in the executor's flush unit, of the whole streaming attention
/// `spec`: a dispatch carrying [`poot_target::WatchdogBudget::attention_work`] weighs one submit
/// ([`poot_target::SUBMIT_WORK`]). The default weight, the output element count, ignores the keys a row walks,
/// so a submit of several split runs would hold several times the watchdog budget (POOT-1313: the gemma
/// 8192-token prefill still lost the device with the runs split but flushed together).
pub(crate) fn attention_watchdog_work(
    spec: &poot_kernelgen::AttentionSpec,
    caps: &DeviceCaps,
) -> Option<u64> {
    let budget = caps.watchdog_budget?.attention_work;
    let total = u128::from(attention_row_work(spec)) * spec.rows() as u128;
    let work = total * u128::from(poot_target::SUBMIT_WORK) / u128::from(budget.max(1));
    Some(u64::try_from(work).unwrap_or(u64::MAX).max(1))
}

/// Is this `MatMul` a candidate for the shared-weight register-tiled GEMM (`kg::DenseTiledWeight::KnF32`, card 099b),
/// the contraction choice `planner::matmul` makes (Card 557; it was the `tile_matmuls` retag)? Backend
/// agnostic: F32 operands, a rank-2 `[K,N]` shared weight (input 1), and `M > 1` once every leading dim folds
/// into M (card 167: the decode tracer's `[batch, 1, N]` middle axis is always 1; `M == 1` is the decode
/// GEMV). BF16 operands stay off it so the WMMA / tensor-core arms see them, and F16 operands because the
/// generated body is f32-only (its LDS tiles and buffer reads are all `Ty::F32`; an F16 matmul takes the
/// dtype-generic serial kernel). The dispatch-cap chunking is the choice's own.
pub(crate) fn is_tileable_matmul_in(g: GraphTables<'_>, eqn: &Eqn) -> bool {
    // A `ContractionEpilogue` over a `MatMul` head is judged by its head operands (its extra inputs are the
    // epilogue's leaves); its output is the head's.
    let head_operands = match &eqn.op {
        OpKind::MatMul { .. } => 2,
        OpKind::ContractionEpilogue {
            head: ContractionHead::MatMul,
            ..
        } => ContractionHead::MatMul.operands(),
        _ => return false,
    };
    let out = &g.aval(eqn.out).shape;
    let Some(&n) = out.last() else {
        return false;
    };
    if out.len() < 2 || out.iter().product::<usize>() / n.max(1) <= 1 {
        return false;
    }
    let ids = &value_ids(eqn)[..head_operands];
    ids.len() == 2
        && ids.iter().all(|&id| g.aval(id).dtype == DType::F32)
        && g.aval(ids[1]).shape.len() == 2
}

/// Does this `MatMul` take the register-tiled GEMM with a weight slab per block (`blocks > 1`)? A
/// rank-3+ `A[..,M,K] @ B[..,K,N]` with `M > 1` whose operands and output share the same leading batch dims and
/// F32 dtype: the per-expert MoE GEMM (rank 3) and the prefill attention matmuls (rank 4, `B*H` batch). Every
/// batch entry reads its own weight slab. The dispatch-cap chunking is the tiled choice's own, so no size
/// declines it.
pub(crate) fn is_batched_tileable_matmul_in(g: GraphTables<'_>, eqn: &Eqn) -> bool {
    if !matches!(eqn.op, OpKind::MatMul { .. }) {
        return false;
    }
    let out = &g.aval(eqn.out).shape;
    let r = out.len();
    if r < 3 || out[r - 2] <= 1 || out.iter().product::<usize>() == 0 {
        return false;
    }
    let ids = value_ids(eqn);
    if ids.len() < 2 || g.aval(eqn.out).dtype != DType::F32 {
        return false;
    }
    if ids[..2].iter().any(|&id| g.aval(id).dtype != DType::F32) {
        return false;
    }
    let (a, b) = (&g.aval(ids[0]).shape, &g.aval(ids[1]).shape);
    // A, B, and out share the same leading batch dims.
    if a.len() != r || b.len() != r || a[..r - 2] != out[..r - 2] || b[..r - 2] != out[..r - 2] {
        return false;
    }
    let (m, n, k) = (out[r - 2], out[r - 1], a[r - 1]);
    a[r - 2] == m && b[r - 2] == k && b[r - 1] == n
}

pub(crate) fn is_decode_attn_gemv_in(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> bool {
    if g.aval(eqn.out).dtype != DType::F32 || !matches!(eqn.op, OpKind::MatMul { .. }) {
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

/// The most activation rows an `IndexedMatMul` runs as the Gemv (`M` rows are the `k` selected experts of a
/// decode step, or a few such steps batched): above it the register-tiled expert walk shares each expert's weight
/// tile across the rows that selected it.
pub(crate) const INDEXED_GEMV_MAX_ROWS: usize = 32;

/// The launch an `IndexedMatMul` of `m` rows and `n` columns runs as a [`kg::Schedule::Gemv`] on every backend, or
/// `None` when it is not a decode shape or its launch would not fit the device's X grid. The launch policy is the
/// dense Gemv's ([`dense_gemv_launch`]) over the `m * n` outputs.
pub(crate) fn indexed_gemv_in(m: usize, n: usize, caps: &DeviceCaps) -> Option<DenseGemvLaunch> {
    if m == 0 || n == 0 || m > INDEXED_GEMV_MAX_ROWS {
        return None;
    }
    let launch = dense_gemv_launch(m * n, caps.compute_units);
    ((m * n).div_ceil(launch.cols as usize) <= caps.max_grid[0] as usize).then_some(launch)
}

/// Whether an `IndexedMatMul` (or a `ContractionEpilogue` over one) has a generated lowering by dtype: f32
/// activation and output, an I32 id operand, and a stacked weight that is f32 or BF16. A BF16 weight
/// additionally has to be a const, or a promoted view of one, which only the planner can tell
/// ([`view_source`] over the views table); `plan_indexed` refuses any other BF16 weight, so no BF16 buffer is
/// read as packed lanes without having been uploaded as them.
pub(crate) fn indexed_matmul_lowers(g: GraphTables<'_>, eqn: &Eqn) -> bool {
    let ids = value_ids(eqn);
    g.aval(eqn.out).dtype == DType::F32
        && ids.len() >= 3
        && g.aval(ids[0]).dtype == DType::F32
        && matches!(g.aval(ids[1]).dtype, DType::F32 | DType::BF16)
        && g.aval(ids[2]).dtype == DType::I32
}

/// The value a promoted view `value` reads in place ([`compute_views`]'s table): the input of the movement op
/// that produced it, followed through every view in the chain; `value` itself when it is no view. The one answer to
/// "what buffer does this view read", shared by the planner and the storage plan.
pub(crate) fn view_source(
    g: GraphTables<'_>,
    views: &HashMap<ValueId, Layout>,
    mut value: ValueId,
) -> ValueId {
    while views.contains_key(&value) {
        let producer = g.eqns.iter().find(|eqn| eqn.out == value);
        match producer.and_then(|eqn| value_ids(eqn).first().copied()) {
            Some(input) => value = input,
            None => break,
        }
    }
    value
}

/// The threads a data-movement equation launches: one per buffer unit of its output, which is a packed
/// E4M3FN word rather than a logical element.
fn movement_units(g: GraphTables<'_>, eqn: &Eqn, out_shape: &[usize], out_numel: usize) -> usize {
    if g.aval(eqn.out).dtype == DType::E4M3FN {
        kg::Storage::BYTE_LANES.units(out_shape)
    } else {
        out_numel
    }
}

pub(crate) fn is_elementwise_2d_in(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> bool {
    // Every generated one-thread-per-unit body past `caps.max_grid[0] * ELEMENTWISE_2D_WIDTH` needs the fold,
    // or it hits the `GridCap` crash a flat launch cannot clear however large its workgroup is. The ceiling
    // is the target's own cap, never the wgpu-specific `65535` literal: a fixture with a smaller `max_grid`
    // must fold sooner.
    let fold_ceiling = caps.max_grid[0] as usize * ELEMENTWISE_2D_WIDTH;
    match &eqn.op {
        // Data movement: one thread per output buffer unit (a packed E4M3FN word, else an element). The
        // materializing path only (a promoted `Plan::View` never launches).
        OpKind::DynamicUpdateSlice { .. }
        | OpKind::Transpose { .. }
        | OpKind::Slice { .. }
        | OpKind::Broadcast { .. }
        | OpKind::Concat { .. }
        | OpKind::Gather { .. } => movement_units(g, eqn, out_shape, out_numel) > fold_ceiling,
        // The in-place row write: one thread per source unit, so the fold follows the source rows.
        OpKind::ScatterUpdate => {
            let ids = value_ids(eqn);
            let src_shape = &g.aval(ids[1]).shape;
            let units = if g.aval(eqn.out).dtype == DType::E4M3FN {
                kg::Storage::BYTE_LANES.units(src_shape)
            } else {
                src_shape.iter().product()
            };
            units > fold_ceiling
        }
        // Binary (broadcast-value and scalar/literal forms), any dtype.
        OpKind::Binary(_) | OpKind::Select => out_numel > fold_ceiling,
        // Unary (e.g. `Exp` over the full attention-score tensor), any dtype.
        OpKind::Unary(_) => out_numel > fold_ceiling,
        // A float pointwise region (the exact-I32 regions launch their own packed grid).
        OpKind::Fused(_) => g.aval(eqn.out).dtype != DType::I32 && out_numel > fold_ceiling,
        // An F32 or (Card 1007) packed-F16 contraction over a checkpoint-order weight: the one-thread-per-output
        // fallback when neither the GEMV nor the tiled GEMM is taken. The tiled GEMM is a `custom_grid` that
        // `finalize` already skips.
        OpKind::DenseContraction {
            weight: DType::F32 | DType::F16,
        } => {
            out_numel > fold_ceiling
                && dense_decode_gemv_in(eqn, out_shape, out_numel, caps).is_none()
        }
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

/// The 2-D fold of the one-thread-per-output `MatMul` fallback (the serial route): it folds when its outputs
/// pass `caps.max_grid[0] * ELEMENTWISE_2D_WIDTH`, like every generated one-thread-per-unit body. The route is
/// the caller's (`planner::matmul`), so no predicate re-derives it here.
pub(crate) fn serial_matmul_fold(out_numel: usize, caps: &DeviceCaps) -> kg::Fold {
    kg::Fold {
        numel: out_numel,
        two_d: out_numel > caps.max_grid[0] as usize * ELEMENTWISE_2D_WIDTH,
        width: ELEMENTWISE_2D_WIDTH,
    }
}

/// The 2-D fold decision of a generated one-thread-per-output-element kernel: the planner decides
/// whether this equation folds ([`is_elementwise_2d_in`]), and `generate` derives the body's `x_groups`
/// and the launch grid from it.
pub(crate) fn elementwise_fold(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> kg::Fold {
    kg::Fold {
        numel: out_numel,
        two_d: is_elementwise_2d_in(g, eqn, out_shape, out_numel, caps),
        width: ELEMENTWISE_2D_WIDTH,
    }
}

/// The most lanes one row of the parallel [`OpKind::FusedRow`] kernel uses.
const FUSED_ROW_MAX_WIDTH: usize = 128;

/// Workgroup width (lanes per row) for the parallel [`OpKind::FusedRow`] kernel: at most
/// [`FUSED_ROW_MAX_WIDTH`], and no more than the row width. Shared by `plan_eqn` and the `FusedRow` launch grid
/// it bakes alongside it.
pub(crate) fn row_parallel_width(n_cols: usize) -> usize {
    n_cols.clamp(1, FUSED_ROW_MAX_WIDTH)
}
