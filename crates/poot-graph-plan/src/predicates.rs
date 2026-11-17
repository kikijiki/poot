//! Per-eqn routing: the strided-view promotion pass (`compute_views`, spec 132) and the dispatch
//! predicates and launch-grid math (`is_tileable_matmul`, `is_batched_tileable_matmul`,
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
use poot_target::{Backend, DeviceCaps, Queried, SubgroupSupport, TargetId};

use crate::planner::PlannedDispatch;
use crate::{Binding, Sink, Source};
use crate::{Capability, PlanError};
use poot_kernelgen::KernelRequest;

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

/// What a consumer needs of the layout it reads one value operand through.
///
/// One rule decides it, [`operand_demand`], and every reader of it agrees: the promotion pass
/// ([`compute_views`]) asks it before it makes a value a view, the bar loop asks [`LayoutDemand::admits`] of the
/// composed layout, the planner families plan the layout they were promised, and [`check_operand_layouts`] proves
/// the plan against it on every compile. A value is a view only where its demand is not `Dense`; where the demand
/// is not met, one movement copy reads the chain's root through the composed layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayoutDemand {
    /// Any layout inside the source: the reader indexes every element through its layout (a pointwise leaf, a
    /// movement's data, a concat part).
    Any,
    /// Row-major from the root's base: the consumer reads the buffer flat (a contraction, a gather, an attention,
    /// a packed payload). A value read this way is never a view.
    Dense,
    /// The indexed contraction's stacked weight `[E, K, N]`: row-major, or the checkpoint's `[E, N, K]` transpose
    /// of it, which the contraction reads in place.
    IndexedWeight,
}

impl LayoutDemand {
    /// Whether a value of `shape` read through `layout` meets this demand.
    pub(crate) fn admits(self, layout: &Layout, shape: &[usize]) -> bool {
        match self {
            LayoutDemand::Any => true,
            LayoutDemand::Dense => *layout == Layout::contiguous(shape),
            LayoutDemand::IndexedWeight => {
                *layout == Layout::contiguous(shape)
                    || matches!(shape, &[_, k, n] if *layout == Layout {
                        strides: vec![k * n, 1, k],
                        offset: 0,
                    })
            }
        }
    }
}

/// The layout demand of value operand `position` of `eqn` (an index into `eqn.inputs`): the one function behind
/// view promotion, the bar loop, the planner families and the compile invariant. A literal reads nothing.
///
/// Every pointwise op reads its leaves through their layouts (the pointwise synthesizer, `fused_views`): an
/// elementwise `Binary`, a `Unary`, a `Fused` region of f32 or exact-I32 leaves, a `FusedRow` of f32 leaves. A
/// movement reads its data through the composed layout of the chain it ends, and a `Concat` reads each part
/// through its own: both over element storage (f32 or I32 words; a packed payload and a two-byte element are read
/// dense). The indexed contraction reads its stacked weight (operand 1 only) strided, so a `Transpose` of the
/// stack is never materialized. Everything else reads its operands flat.
pub(crate) fn operand_demand(g: GraphTables<'_>, eqn: &Eqn, position: usize) -> LayoutDemand {
    let Some(Operand::Value(value)) = eqn.inputs.get(position) else {
        return LayoutDemand::Dense;
    };
    let words = matches!(g.aval(*value).dtype, DType::F32 | DType::I32);
    match &eqn.op {
        OpKind::Binary(_) | OpKind::Unary(_) | OpKind::Fused(_) if words => LayoutDemand::Any,
        OpKind::FusedRow(_) if g.aval(*value).dtype == DType::F32 => LayoutDemand::Any,
        OpKind::Transpose { .. }
        | OpKind::Slice { .. }
        | OpKind::Broadcast { .. }
        | OpKind::Reshape { .. }
        | OpKind::Concat { .. }
            if words =>
        {
            LayoutDemand::Any
        }
        // The stacked weight is read in place only where it is read once: as the activation, the ids or an
        // epilogue leaf it is read flat.
        OpKind::IndexedMatMul { .. }
        | OpKind::ContractionEpilogue {
            head: ContractionHead::Indexed { .. },
            ..
        } if position == 1 && value_ids(eqn).iter().filter(|&&v| v == *value).count() == 1 => {
            LayoutDemand::IndexedWeight
        }
        _ => LayoutDemand::Dense,
    }
}

/// Is `op` one of the 4 movement ops promoted to a view (Transpose/Slice/Broadcast/Reshape)? Used by
/// [`compute_views`] for the candidate check. A `Reshape` is a view only over a value that is itself a view (and
/// only when the view's layout splits, inserts, removes or chain-merges axes, [`reshape_strides`]); over a dense
/// value it stays the zero-cost `Plan::Alias`.
fn is_movement_view_candidate(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::Transpose { .. }
            | OpKind::Slice { .. }
            | OpKind::Broadcast { .. }
            | OpKind::Reshape { .. }
    )
}

/// The element strides that read `in_shape` (laid out with `in_strides`) as `out_shape` without copying, or
/// `None` when no single stride per output axis does: NumPy's no-copy reshape rule. Axes of extent 1 carry
/// no stride, a run of input axes merges into one output axis only when each axis's stride is the next
/// one's stride times its extent, and an input axis splits into output axes at the row-major strides within it.
/// An input with a zero extent has no elements to address and is refused.
fn reshape_strides(
    in_shape: &[usize],
    in_strides: &[usize],
    out_shape: &[usize],
) -> Option<Vec<usize>> {
    if in_shape.contains(&0) || out_shape.contains(&0) {
        return None;
    }
    // The input axes that matter: extent 1 axes are free to be dropped, whatever their stride.
    let old: Vec<(usize, usize)> = in_shape
        .iter()
        .zip(in_strides)
        .filter(|&(&extent, _)| extent != 1)
        .map(|(&extent, &stride)| (extent, stride))
        .collect();
    let mut out = vec![0usize; out_shape.len()];
    let (mut oi, mut oj) = (0usize, 0usize);
    while oi < old.len() && oj < out_shape.len() {
        // Grow the input group [oi, ni) and the output group [oj, nj) until they cover the same element count.
        let (mut ni, mut nj) = (oi + 1, oj + 1);
        let (mut old_prod, mut new_prod) = (old[oi].0, out_shape[oj]);
        while old_prod != new_prod {
            if old_prod < new_prod {
                old_prod *= old.get(ni)?.0;
                ni += 1;
            } else {
                new_prod *= out_shape.get(nj)?;
                nj += 1;
            }
        }
        // The input group must chain into one run: each stride is the next one's times the next one's extent.
        if !(oi..ni - 1).all(|k| old[k].1 == old[k + 1].1 * old[k + 1].0) {
            return None;
        }
        // The output axes of the group are the row-major split of the run, ending in the run's last stride.
        let mut stride = old[ni - 1].1;
        for k in (oj..nj).rev() {
            out[k] = stride;
            stride *= out_shape[k];
        }
        (oi, oj) = (ni, nj);
    }
    // Trailing output axes of extent 1 (the input is exhausted) carry no stride; any other leftover is a
    // mismatched element count, which a validated Reshape never has.
    if old[oi..].is_empty() && out_shape[oj..].iter().all(|&extent| extent == 1) {
        Some(out)
    } else {
        None
    }
}

/// The layout of the output of movement `op` read from its input, which is of logical shape `in_shape` and sits in
/// its source buffer as `in_layout` (`Layout::contiguous` when the input is no view): the composition both
/// [`compute_views`] and the movement planner apply, so a chain ends in one layout over its root whether its links
/// are views or one materialized copy. `None` for a reshape whose axes `in_layout` cannot merge or split
/// ([`reshape_strides`]).
pub(crate) fn compose_movement(
    op: &OpKind,
    in_shape: &[usize],
    in_layout: &Layout,
    out_shape: &[usize],
) -> Option<Layout> {
    match op {
        OpKind::Transpose { perm } => Some(in_layout.permuted(perm)),
        OpKind::Slice { axis, start, .. } => Some(in_layout.sliced(*axis, *start)),
        OpKind::Broadcast { shape } => Some(in_layout.broadcast(in_shape, shape)),
        OpKind::Reshape { shape } => {
            reshape_strides(in_shape, &in_layout.strides, shape).map(|strides| Layout {
                strides,
                offset: in_layout.offset,
            })
        }
        _ => unreachable!("only a movement view candidate composes a layout (out {out_shape:?})"),
    }
}

/// The logical shape of the buffer `value` is read from: the shape of the first value down its chain of promoted
/// views that is no view. A view's layout is expressed over that shape ([`compose_layouts`]), so it is the bound
/// every reader of the layout checks.
pub(crate) fn view_source_shape(
    g: GraphTables<'_>,
    views: &HashMap<ValueId, Layout>,
    value: ValueId,
) -> Vec<usize> {
    let mut current = value;
    while views.contains_key(&current) {
        let producer = g
            .eqns
            .iter()
            .rev()
            .find(|eqn| eqn.out == current)
            .expect("a promoted view has a producing equation");
        current = value_ids(producer)[0];
    }
    g.aval(current).shape.clone()
}

/// Which values become a [`Plan::View`] (no dispatch) instead of a materialize copy, composing a whole
/// chain of movement ops into one Layout each (spec 132). WGPU and AMD GCN only; any other backend gets an
/// empty map and takes the pre-existing movement lowering.
///
/// Two passes, since deciding promotion needs consumers and composing a Layout needs the input's Layout
/// (earlier):
///
/// - Pass A decides the promoted set. A `Transpose`/`Slice`/`Broadcast`/`Reshape` output (not pinned, with at
///   least one reader) is promoted when no reader's [`operand_demand`] is `Dense`. A movement reader counts as a
///   reader whether or not it is promoted itself: the chain's tail then materializes as one copy that reads the
///   root through the whole chain's layout, instead of one copy per link. A reshape that is not promoted is the
///   exception: it is free over a dense value, so it keeps the value under it dense.
/// - Pass B walks forward and composes each promoted value's Layout onto its input's Layout
///   (`Layout::contiguous` if the input was not promoted). A chain tail therefore has a Layout expressed
///   in terms of the original contiguous source, matching `poot_eval::apply_movement` over the whole
///   chain (see the `view_..._chain_matches_oracle_fuzz` test).
///
/// A `Reshape` composes only over a view whose strides it can split or merge ([`reshape_strides`]), and that
/// depends on the input's composed Layout, which pass B alone knows. The reshape's input is barred when it cannot,
/// so it materializes and the reshape reads it dense. A promoted value whose composed layout a reader's demand
/// does not [`admit`](LayoutDemand::admits) is barred too. Barring only shrinks the promoted set, so the loop ends.
///
/// A value with no consumer is not promoted. A pinned value keeps the chain's materialization at its own copy.
///
/// Every pinned value ([`Graph::pinned_values`]: the output, each validation value, and both sides of each
/// state pair) keeps its own materialized buffer, so a validation packet can copy it row-major from offset 0.
pub fn compute_views(g: &Graph, target: impl Into<TargetId>) -> HashMap<ValueId, Layout> {
    // A target that compiles no kernel bodies (a VPU) has no strided-view bodies to read a view through.
    let strided = poot_codegen::Target::try_from(target.into())
        .is_ok_and(|target| codegen_capability(target).strided_views);
    if !strided {
        return HashMap::new();
    }
    let tables = GraphTables::from(g);
    let readers = readers(tables);
    let pinned: std::collections::HashSet<ValueId> = g.pinned_values().collect();
    let mut barred: std::collections::HashSet<ValueId> = std::collections::HashSet::new();
    loop {
        let promoted = promoted_movement(tables, &readers, &pinned, &barred);
        let (views, unreadable) = compose_layouts(tables, &promoted, &readers);
        if unreadable.is_empty() {
            return views;
        }
        barred.extend(unreadable);
    }
}

/// Every operand read of every value: the eqn (an index into `g.eqns`) and the position in its inputs.
fn readers(g: GraphTables<'_>) -> HashMap<ValueId, Vec<(usize, usize)>> {
    let mut out: HashMap<ValueId, Vec<(usize, usize)>> = HashMap::new();
    for (i, eqn) in g.eqns.iter().enumerate() {
        for (position, operand) in eqn.inputs.iter().enumerate() {
            if let Operand::Value(v) = operand {
                out.entry(*v).or_default().push((i, position));
            }
        }
    }
    out
}

/// Pass A of [`compute_views`] (reverse topological order, so a reader's own promotion is known): the promoted
/// set, no strides yet. A value in `barred` is never promoted.
///
/// A reader that is a movement counts whether or not it is promoted, with one exception: a `Reshape` that is not
/// promoted reads its input dense, where over a dense value it is the zero-cost `Plan::Alias`. Promoting the view
/// under it would make the reshape a copy, and every reshape of a value that several of them read a copy of its
/// own, where the dense value costs one.
fn promoted_movement(
    g: GraphTables<'_>,
    readers: &HashMap<ValueId, Vec<(usize, usize)>>,
    pinned: &std::collections::HashSet<ValueId>,
    barred: &std::collections::HashSet<ValueId>,
) -> std::collections::HashSet<ValueId> {
    let mut promoted = std::collections::HashSet::new();
    for eqn in g.eqns.iter().rev() {
        let out = eqn.out;
        if !is_movement_view_candidate(&eqn.op) || pinned.contains(&out) || barred.contains(&out) {
            continue;
        }
        let readable = readers.get(&out).is_some_and(|reads| {
            reads.iter().all(|&(i, position)| {
                let reader = &g.eqns[i];
                let dense_reshape =
                    matches!(reader.op, OpKind::Reshape { .. }) && !promoted.contains(&reader.out);
                !dense_reshape && operand_demand(g, reader, position) != LayoutDemand::Dense
            })
        });
        if readable {
            promoted.insert(out);
        }
    }
    promoted
}

/// Pass B of [`compute_views`] (forward topological order): each promoted value's Layout composed from its
/// input's, and the values to bar: the input of a reshape (promoted or not) whose input Layout has no stride for
/// the new shape, and a promoted value whose composed Layout a reader's demand does not admit. A reshape over a
/// value that is no view has no Layout of its own: it stays a `Plan::Alias`.
fn compose_layouts(
    g: GraphTables<'_>,
    promoted: &std::collections::HashSet<ValueId>,
    readers: &HashMap<ValueId, Vec<(usize, usize)>>,
) -> (HashMap<ValueId, Layout>, Vec<ValueId>) {
    let mut views: HashMap<ValueId, Layout> = HashMap::new();
    let mut unreadable = Vec::new();
    for eqn in g.eqns {
        if !is_movement_view_candidate(&eqn.op) {
            continue;
        }
        let out = eqn.out;
        let input = value_ids(eqn)[0];
        let in_shape = &g.aval(input).shape;
        let in_layout = match (views.get(&input), &eqn.op) {
            (Some(layout), _) => layout.clone(),
            (None, OpKind::Reshape { .. }) => continue,
            (None, _) => Layout::contiguous(in_shape),
        };
        let Some(layout) = compose_movement(&eqn.op, in_shape, &in_layout, &g.aval(out).shape)
        else {
            unreadable.push(input);
            continue;
        };
        if !promoted.contains(&out) {
            continue;
        }
        let shape = &g.aval(out).shape;
        let admitted = readers.get(&out).is_some_and(|reads| {
            reads.iter().all(|&(i, position)| {
                operand_demand(g, &g.eqns[i], position).admits(&layout, shape)
            })
        });
        if admitted {
            views.insert(out, layout);
        } else {
            unreadable.push(out);
        }
    }
    (views, unreadable)
}

/// The compile invariant of the layout rule: every value operand that is a promoted view meets its reader's
/// [`operand_demand`]. [`compute_views`] already promotes by that demand, so this holds by construction; it is checked
/// again here, on every compile, because a view a reader cannot read would be read as if it were dense (wrong
/// elements, no error). A violation is the typed [`Capability::OperandLayout`] naming the equation.
pub(crate) fn check_operand_layouts(
    g: GraphTables<'_>,
    views: &HashMap<ValueId, Layout>,
    target: impl Into<TargetId> + Copy,
    limits: poot_kernelgen::BodyLimits,
) -> Result<(), PlanError> {
    for eqn in g.eqns {
        for (position, operand) in eqn.inputs.iter().enumerate() {
            let Operand::Value(value) = operand else {
                continue;
            };
            let Some(layout) = views.get(value) else {
                continue;
            };
            if !operand_demand(g, eqn, position).admits(layout, &g.aval(*value).shape) {
                return Err(Site::new(g, eqn, target, limits)
                    .refuse(Capability::OperandLayout { value: *value }));
            }
        }
    }
    Ok(())
}

/// Workgroup width (lanes per output column) for the LDS decode attention GEMV (`kernelgen::attn_scores_v_gemv_lds`),
/// used by `plan_eqn` for both the `[w,1,1]` workgroup and the planned `N*width`-thread grid: the device's calibrated
/// [`poot_target::calibration::PerfCalibration::lds_gemv_width`] (128 measured best on RADV STRIX_HALO,
/// Qwen2.5-0.5B decode).
pub(crate) fn lds_gemv_width(caps: &DeviceCaps) -> usize {
    caps.gemm_launch().calibrated.figures().lds_gemv_width as usize
}

/// The 2-D workgroup grid `(x_groups, y_groups)` for a decode attention GEMV over `out_numel` elements (one
/// workgroup per element). X fills up to `caps.max_grid[0]` (the planner reads the target's own measured cap,
/// never a baked wgpu-specific literal) and the remainder spills onto Y. For `out_numel <= caps.max_grid[0]`
/// this is `(out_numel, 1)`.
pub(crate) fn gemv_grid(out_numel: usize, caps: &DeviceCaps) -> (usize, usize) {
    kg::fold_groups(out_numel, caps.max_grid[0])
}

/// The launch a `DenseContraction` decode GEMV runs as a [`kg::Schedule::Gemv`] (M == 1 once every leading dim folds
/// into M), or `None` when the equation is not one or its launch would not fit the device's X grid. Every weight
/// dtype the fold admits (F32, packed F16, packed BF16) and every backend take the same generated body; the
/// launch comes from [`gemv_launch`], the activation read from the planner's [`gemv_activation`]
/// ([`GemvLaunch::schedule`]). `cols_pin` replaces the launch's columns per workgroup for the calibration sweep
/// ([`crate::CompileOptions::gemv_cols_pin`]); a pin the workgroup width does not divide is no launch.
pub(crate) fn dense_decode_gemv_in(
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    cols_pin: Option<u32>,
    caps: &DeviceCaps,
) -> Option<GemvLaunch> {
    eqn.op.dense_weight()?;
    let n = *out_shape.last()?;
    if out_shape.len() < 2 || n == 0 || out_numel != n {
        return None;
    }
    let planned = gemv_launch(n, caps);
    let launch = match cols_pin {
        Some(cols) if cols == 0 || !planned.width.is_multiple_of(cols) => return None,
        Some(cols) => GemvLaunch { cols, ..planned },
        None => planned,
    };
    (n.div_ceil(launch.cols as usize) <= caps.max_grid[0] as usize).then_some(launch)
}

/// The launch shape [`gemv_launch`] picks for a Gemv (a [`kg::Schedule::Gemv`]'s fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GemvLaunch {
    pub(crate) width: u32,
    pub(crate) cols: u32,
    pub(crate) unroll: u32,
    /// The adjacent output rows a lane accumulates over one activation read.
    pub(crate) rows: kg::GemvRows,
    /// The trips of weight loads a lane keeps in flight.
    pub(crate) depth: kg::LoadDepth,
}

impl GemvLaunch {
    /// The launch as the cost model reads it, over `outputs` dot products of depth `k`.
    pub(crate) fn cost_shape(self, outputs: usize, k: usize) -> kg::GemvShape {
        kg::GemvShape {
            outputs,
            k,
            width: self.width,
            cols: self.cols * self.rows.get(),
        }
    }

    /// The Gemv of this launch over activation rows of `k` values on `backend`, its activation read from
    /// [`gemv_activation`].
    pub(crate) fn schedule(self, backend: Backend, k: usize) -> kg::Schedule {
        kg::Schedule::Gemv {
            width: self.width,
            cols: self.cols,
            unroll: self.unroll,
            activation: gemv_activation(backend, k, self.unroll),
            rows: self.rows,
            depth: self.depth,
        }
    }
}

/// The Gemv launch of a dense `[N, K]` weight with `outputs` output columns (card 727). (A packed weight's is the
/// argmin of a cost model over the body its format emits: `planner::gemv_geometry`.)
///
/// `width` and `unroll` are the device's [`poot_target::calibration::GemmLaunch`] geometry. `cols` is the calibrated
/// `wide` while that still launches [`poot_target::calibration::GemmLaunch::fill_workgroups_per_unit`] workgroups per
/// compute unit, else its `narrow`: a weight row is contiguous along K, so fewer `cols` give each row more lanes
/// (`width / cols`) and a longer contiguous run per trip, and the narrower fallback keeps a small `N` from starving the
/// device. A device whose compute units are unknown cannot show that the wide columns fill it, so it takes the narrow
/// ones, which launch the most workgroups. One row per lane and one trip in flight: the dense body's own.
pub(crate) fn gemv_launch(outputs: usize, caps: &DeviceCaps) -> GemvLaunch {
    let launch = caps.gemm_launch();
    let table = launch.calibrated.figures().dense_gemv_cols;
    let wide_fills = caps.compute_units.known().is_some_and(|units| {
        outputs.div_ceil(table.wide as usize)
            >= launch.fill_workgroups_per_unit as usize * units as usize
    });
    GemvLaunch {
        width: launch.gemv_width,
        cols: if wide_fills { table.wide } else { table.narrow },
        unroll: launch.gemv_unroll,
        rows: kg::GemvRows::ONE,
        depth: kg::LoadDepth::ONE,
    }
}

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

/// The bytes one lane reads of a key or value row in one load.
const VECTOR_LOAD_BYTES: usize = 16;

/// The clustered key walk of an attention ([`poot_kernelgen::KeyWalk`]) and the lanes of the workgroup that runs
/// it, or the capability the device lacks.
///
/// A lane's run is the widest power-of-two count of `kv_bytes`-wide elements that one 16-byte load covers
/// and that divides both head widths and a split key's parts (`shared_dim`, `0` for a dense key). The cluster is the
/// power of two of lanes that covers the wider head with that run; a head too wide for 32 lanes at the load's run
/// takes a longer one. The `n_rep` heads' query, output and softmax state and the key and value runs live in
/// registers across the key loop: while they exceed the lane's held share of the register file, the cluster doubles
/// and the run halves, until the cluster is a full 32 lanes; a state that exceeds the lane's whole file is refused
/// ([`Capability::AttentionRegisters`]). The workgroup is the lanes that saturate a compute unit
/// at [`DeviceCaps::gemm_launch`]'s residency (two workgroups resident), in whole subgroups, halved while the end
/// combine of the clusters (`clusters * n_rep * (dv + 2)` words) exceeds the device's workgroup memory.
pub(crate) fn attention_key_walk(
    (dk, dv, shared_dim): (usize, usize, usize),
    n_rep: usize,
    kv_bytes: usize,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<(usize, kg::KeyWalk), Capability> {
    let collective_lanes = poot_kernel_ir::COLLECTIVE_LANES;
    let Queried::Known(SubgroupSupport::Present { min_size, max_size }) = caps.subgroup else {
        return Err(Capability::SubgroupShuffle);
    };
    if min_size < collective_lanes || !codegen_capability(backend.into()).subgroup_shuffle {
        return Err(Capability::SubgroupShuffle);
    }
    let launch = caps.gemm_launch();
    let lanes_limit = caps.max_workgroup_size[0].min(caps.max_workgroup_invocations);
    let saturating = launch.saturating_lanes / launch.fill_workgroups_per_unit;
    let lanes = saturating.min(lanes_limit) / max_size * max_size;
    if lanes == 0 {
        return Err(Capability::SubgroupShuffle);
    }

    let widths = [dk, dv, dk - shared_dim];
    let mut run = (VECTOR_LOAD_BYTES / kv_bytes).next_power_of_two();
    while run > 1 && !widths.iter().all(|width| width.is_multiple_of(run)) {
        run /= 2;
    }
    let cluster_cap = collective_lanes as usize;
    while dk.max(dv).div_ceil(run) > cluster_cap {
        run *= 2;
        if !widths.iter().all(|width| width.is_multiple_of(run)) {
            return Err(Capability::AttentionHeadWidth { dk, dv });
        }
    }
    let mut cluster = dk.max(dv).div_ceil(run).next_power_of_two();
    let registers = |run: usize| {
        let run_words = (run * kv_bytes).div_ceil(4);
        n_rep * (2 * run) + 2 * n_rep + 2 * run_words
    };
    // The held share leaves the other half of the lane's file to the step's temporaries; a wider cluster spreads
    // the state over more lanes. A cluster at 32 lanes that still exceeds the share keeps it (the compiler's spill
    // is slow, not wrong, as for the accumulators) until the state cannot fit the lane's whole file.
    let held = launch.register_budget.loop_registers(lanes) as usize;
    while registers(run) > held && cluster < cluster_cap && run > 1 {
        cluster *= 2;
        run /= 2;
    }
    let file = launch.register_budget.lane_registers(lanes) as usize;
    if registers(run) > file {
        return Err(Capability::AttentionRegisters {
            needed: registers(run) as u32,
            budget: file as u32,
        });
    }

    let lds_bytes = caps.lds_bytes as usize;
    let combine_bytes = |lanes: usize| 4 * (lanes / cluster) * n_rep * (dv + 2);
    let subgroup = max_size as usize;
    let mut lanes = lanes as usize;
    while combine_bytes(lanes) > lds_bytes {
        if !(lanes / 2).is_multiple_of(subgroup) || lanes / 2 < subgroup {
            return Err(Capability::AttentionLds {
                dv,
                lds_bytes: caps.lds_bytes,
            });
        }
        lanes /= 2;
    }
    Ok((lanes, kg::KeyWalk { cluster, run }))
}

/// The split-KV schedule of a decode attention (`Lq == 1`), or `None` for the streaming schedule.
///
/// The streaming schedule runs `batch * heads` workgroups; at a few heads that leaves most of a many-CU
/// device idle while each workgroup walks the whole cache (the Strix Halo shape: 2 workgroups on 40 CUs).
/// Splitting the keys across `splits` workgroups and reducing the partial `(m, l, o)` states in a second
/// launch is Flash-Decoding (Dao et al., <https://pytorch.org/blog/flash-decoding/>, retrieved 2026-10-11);
/// vLLM's Triton decode attention chooses its split count the same way, from the occupancy the
/// batch-and-head grid leaves
/// (<https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py>,
/// retrieved 2026-10-11). The capacity bounds the grid, the live range partitions it: `splits` follows
/// `caps.compute_units` and the cache length, and is only the ceiling. The live length is a run-time value, so
/// the kernel cuts each row's visited keys into at most that many shares of at least `min_units` steps
/// ([`poot_kernelgen::AttentionSchedule::SplitKv`]), and the workgroups past the row's shares return.
///
/// Streaming is kept when any of these holds, so a request never needs a capability the device lacks:
/// the row groups (one per key/value head) already fill the device, the cache is too short for two shares of
/// `clusters * attention_split_min_steps` keys
/// ([`poot_target::calibration::PerfCalibration::attention_split_min_steps`]), the partial-state scratch exceeds
/// `caps.max_buffer_bytes`, or the launch's 2-D fold would pass `max_grid`.
/// `spec` is the streaming-schedule request being considered.
pub(crate) fn attention_splits(
    spec: &poot_kernelgen::AttentionSpec,
    caps: &DeviceCaps,
) -> Option<poot_kernelgen::AttentionSchedule> {
    if spec.lq != 1 {
        return None;
    }
    let rows = spec.row_groups();
    // Splitting fills an idle device; with no compute-unit count there is no idle to prove, so keep streaming.
    let compute_units = caps.compute_units.known()? as usize;
    // The split count aims at the launch's resident workgroups per compute unit: two keep a CU busy while one
    // waits on its K/V loads.
    let fill = caps.gemm_launch().fill_workgroups_per_unit as usize;
    let wanted = (compute_units * fill).div_ceil(rows);
    let walk = spec.schedule.walk()?;
    // A share is whole steps of the walk, `clusters` keys each, and at least `attention_split_min_steps` of them.
    let clusters = spec.schedule.lanes() / walk.cluster;
    let min_units = clusters * caps.perf_calibration.figures().attention_split_min_steps as usize;
    let splits = wanted.min(spec.candidates() / min_units);
    if splits < 2 || rows >= compute_units {
        return None;
    }
    let schedule = poot_kernelgen::AttentionSchedule::SplitKv {
        lanes: spec.schedule.lanes(),
        splits,
        min_units,
        walk,
    };
    let split = poot_kernelgen::AttentionSpec {
        schedule,
        ..spec.clone()
    };
    let scratch_bytes = split.partial_elems()? as u64 * 4;
    if scratch_bytes > caps.max_buffer_bytes {
        return None;
    }
    let (_, y) = poot_kernelgen::fold_groups(split.workgroups(), caps.max_grid[0]);
    (y <= caps.max_grid[1] as usize).then_some(schedule)
}

/// The most tile rows (query positions times packed heads) one tiled attention workgroup serves.
const TILED_MAX_ROWS: usize = 128;
/// Keys per staged block of the fragment schedule, widest first. A fragment step multiplies 16 keys at a time and
/// a block costs five barriers and the softmax's row combine whatever its width, so a wider block spends them on
/// more multiplies.
const FRAGMENT_KEY_BLOCKS: [usize; 3] = [64, 32, 16];
/// Workgroup widths the fragment schedule considers, widest first: whole waves of 32 lanes, down to the one wave a
/// runtime that cannot widen its subgroups allows.
const FRAGMENT_LANES: [usize; 5] = [512, 256, 128, 64, 32];
/// The widest lane tile column count a tiled attention considers: a lane's columns are one register each.
const TILED_MAX_LANE_COLS: usize = 16;

/// The register tiles one lane of a `lanes`-lane workgroup can hold over a `rows x cols` tile: every `r x c` with
/// `(rows / r) * (cols / c) == lanes`, none when no grid of whole tiles covers it. Which one is best depends on
/// what the whole schedule costs (the workgroup memory its layout takes, the LDS reads per FMA), so the planner
/// weighs them all ([`attention_tiling_of`]).
fn attention_lane_tiles(
    rows: usize,
    cols: usize,
    lanes: usize,
) -> impl Iterator<Item = poot_kernelgen::LaneTile> {
    let cells = rows * cols;
    let per_lane = if cells.is_multiple_of(lanes) {
        cells / lanes
    } else {
        0
    };
    (1..=per_lane.min(rows))
        .filter(move |r| rows.is_multiple_of(*r) && per_lane.is_multiple_of(*r))
        .map(move |r| (r, per_lane / r))
        .filter(move |&(r, c)| {
            c <= TILED_MAX_LANE_COLS && cols.is_multiple_of(c) && (rows / r) * (cols / c) == lanes
        })
        .map(|(rows, cols)| poot_kernelgen::LaneTile { rows, cols })
}

/// The matrix-unit micro-tile a tiled attention can use for operands of `spec`'s precision on `caps`: the fragment
/// dtype its precision narrows to and the device's fragment config for it (a 16x16x16 multiply held across the
/// 32 lanes of a wave), `None` for an exact precision or a device without that matrix unit.
fn attention_fragment_config(
    spec: &poot_kernelgen::AttentionSpec,
    caps: &DeviceCaps,
) -> Option<poot_target::FragmentConfig> {
    let poot_graph_ir::OperandPrecision::Narrow(dtype) = spec.precision else {
        return None;
    };
    let config = *caps.matrix_units.known()?.config(dtype)?;
    (config.shape == poot_target::FragmentShape::M16N16K16
        && config.subgroup == poot_kernel_ir::COLLECTIVE_LANES)
        .then_some(config)
}

/// The wave grids `waves` waves form over a `rows x cols` tile whose share is whole 16x16 fragments.
fn attention_wave_grids(
    rows: usize,
    cols: usize,
    waves: usize,
) -> impl Iterator<Item = poot_kernelgen::WaveTile> {
    let edge = poot_target::FragmentShape::M16N16K16.m as usize;
    (1..=waves)
        .filter(move |waves_m| waves.is_multiple_of(*waves_m))
        .filter_map(move |waves_m| {
            let waves_n = waves / waves_m;
            let (m, n) = (edge * waves_m, edge * waves_n);
            (rows.is_multiple_of(m) && cols.is_multiple_of(n)).then_some(poot_kernelgen::WaveTile {
                waves_m,
                waves_n,
                rows: (rows / m) as u32,
                cols: (cols / n) as u32,
            })
        })
}

/// The fragments one wave holds over a `rows x cols` tile that `waves` waves cover, where both operands are
/// loaded every step: the most square grid (the fewest operand fragment loads per multiply, as
/// [`attention_lane_tile`] weighs lane tiles), ties to fewer wave columns (fewer lanes to combine a row over).
fn attention_wave_tile(rows: usize, cols: usize, waves: usize) -> Option<poot_kernelgen::WaveTile> {
    attention_wave_grids(rows, cols, waves).min_by_key(|tile| (tile.rows + tile.cols, tile.waves_n))
}

/// The fragments one wave holds over a `rows x cols` score tile whose A operand (Q) the wave holds in registers:
/// the grid with the fewest fragment rows per wave (each holds `dk / 16` A fragments for the whole key loop, and
/// a step loads only the B fragments, `1 / rows` a multiply), ties to fewer wave columns.
fn attention_held_wave_tile(
    rows: usize,
    cols: usize,
    waves: usize,
) -> Option<poot_kernelgen::WaveTile> {
    attention_wave_grids(rows, cols, waves).min_by_key(|tile| (tile.rows, tile.waves_n))
}

/// The micro-tiles of a tile of `rows` and the schedule's `bc`, `dv` and `dk` widths over `lanes` lanes with Q
/// where `q` says, and the registers each lane holds for them across the key loop: matrix fragments of `config`
/// (one wave per 32 lanes) where `fragment` names it (the f32 accumulator fragments of both products and, with Q
/// held, its operand fragments), else register tiles (their f32 accumulators; Q is always staged). A register
/// tiling has a micro-tile for every pair of lane tiles ([`attention_lane_tiles`]); a fragment tiling the one
/// wave grid per product its weighing picks.
fn attention_micros(
    fragment: Option<poot_target::FragmentConfig>,
    (rows, bc, dv, dk): (usize, usize, usize, usize),
    lanes: usize,
    q: poot_kernelgen::QOperand,
) -> Vec<(poot_kernelgen::AttentionMicro, usize)> {
    match fragment {
        None => {
            if q == poot_kernelgen::QOperand::Held {
                return Vec::new();
            }
            attention_lane_tiles(rows, bc, lanes)
                .flat_map(|score| {
                    attention_lane_tiles(rows, dv, lanes).map(move |out| {
                        (
                            poot_kernelgen::AttentionMicro::Scalar { score, out },
                            score.cells() + out.cells(),
                        )
                    })
                })
                .collect()
        }
        Some(config) => {
            let lane_count = config.subgroup as usize;
            let waves = lanes / lane_count;
            if !lanes.is_multiple_of(lane_count) || waves > config.waves as usize {
                return Vec::new();
            }
            let score = match q {
                poot_kernelgen::QOperand::Staged => attention_wave_tile(rows, bc, waves),
                poot_kernelgen::QOperand::Held => attention_held_wave_tile(rows, bc, waves),
            };
            let (Some(score), Some(out)) = (score, attention_wave_tile(rows, dv, waves)) else {
                return Vec::new();
            };
            let fragments = (score.rows * score.cols + out.rows * out.cols) as usize;
            let accumulators =
                fragments * poot_kernel_ir::accumulator_map::ELEMENTS_PER_LANE as usize;
            let held_q = match q {
                poot_kernelgen::QOperand::Staged => 0,
                poot_kernelgen::QOperand::Held => {
                    score.rows as usize
                        * (dk / config.shape.k as usize)
                        * config.operand_registers() as usize
                }
            };
            vec![(
                poot_kernelgen::AttentionMicro::Fragment {
                    dtype: config.dtype,
                    layout: config.layout,
                    score,
                    out,
                    q,
                },
                accumulators + held_q,
            )]
        }
    }
}

/// The tiled (FlashAttention-2) schedule of a multi-row attention, or `None` to keep the streaming one.
///
/// A tiled workgroup stages each block of keys and values once for all of its `rows = br * n_rep` tile rows, so
/// its global traffic per FLOP falls as `1 / rows` (FlashAttention-2, Dao, arXiv 2307.08691, section 3). The
/// choice maximizes that reuse within what the device holds, from the shapes and caps alone:
///
/// - the workgroup is at most the device's widest;
/// - each lane holds at most its register budget across the key loop: a register tile's two accumulator tiles at
///   most `register_budget.accumulators` f32 registers (a quarter of what the lane keeps resident: 64 on RDNA3 and
///   Ampere, where 80 accumulators measured slower than 40 on wgpu, POOT-1402 bank sweep); a fragment tiling's
///   accumulator fragments, and its held
///   Q fragments, at most half of what one lane of the workgroup keeps resident of the device's register file
///   ([`poot_target::RegisterFile::per_lane_of`]: the addressing limit or the workgroup's share of the file), the
///   other half carrying a step's loaded operands, the staging and the softmax's row state;
/// - the staged tiles fit `caps.lds_bytes`;
/// - the row groups (`batch * hkv * ceil(lq / br)` workgroups) give every compute unit at least one.
///
/// Where the operands narrow to a dtype the device has a matrix unit for ([`attention_fragment_config`]) and the
/// device reports its register file, the candidates are matrix fragments: waves of 32 lanes over the 16-row,
/// 16-key and 16-channel fragments of the two products, a workgroup of `lanes / 32` waves, with Q staged in
/// workgroup memory or held in registers ([`poot_kernelgen::QOperand`]). Where no fragment tiling fits, or the
/// operands are exact, they are register tiles. A fragment tiling runs no fewer lanes than the register tiles would.
///
/// Among the register tilings that pass, the widest workgroup wins, then the most tile rows (reuse), then the widest
/// key block (fewer barriers per FLOP). The staged tiles take most of a compute unit's workgroup memory, so one
/// workgroup is resident per compute unit and its lanes are all the latency hiding the unit has: 256 lanes over
/// 32 rows ran 30-40% faster than 128 lanes over 64 rows on ROCm (POOT-1402 bank sweep). Among the fragment tilings
/// the most tile rows win, then the narrowest workgroup, then the widest key block, then a held Q. The rows set how
/// often every key block is restaged. At equal rows more waves split the same fragments into smaller wave tiles
/// (more operand loads per multiply, more owners per row to combine) and measured slower every time, more than a
/// wider key block repays (ROCm, the bank's 8192-position cell, f16 cache: 64 rows over 256 lanes 23.3 ms against
/// 25.2 over 512, POOT-1440; 128 rows with Q held over 256 lanes 17.5 ms, over 512 lanes 25.0 at the same 32-key
/// block and 20.5 at a 64-key one, POOT-1447). The block sets how many barriers and softmax combines a key costs.
/// A held Q drops the Q fragment loads of every key block (at equal tiling 17.5 against 18.2 ms) and frees its
/// workgroup memory. A block-selected attention and a single-row (decode) one are not tiled: their rows share no
/// key blocks.
pub(crate) fn attention_tiling(
    spec: &poot_kernelgen::AttentionSpec,
    caps: &DeviceCaps,
) -> Option<poot_kernelgen::AttentionSchedule> {
    if spec.lq < 2 || matches!(spec.mask, poot_kernelgen::MaskKind::Blocks { .. }) {
        return None;
    }
    let fragment = attention_fragment_config(spec, caps)
        .filter(|config| spec.dk.is_multiple_of(config.shape.k as usize));
    let registers = attention_tiling_of(spec, caps, None, 0);
    // A workgroup's lanes are all the latency hiding a compute unit has (one is resident per unit), so a fragment
    // tiling must run at least as many as the register tiles it replaces. wgpu cannot widen its subgroups, so its
    // fragment workgroup is one wave of 32 lanes: measured on the bank's `prefill-attention-*` cells (Strix Halo,
    // POOT-1437) that is 11.6x (2048) and 13.8x (8192) slower than the f32 register tiles, where the multi-wave
    // ROCm tiling is 2.0x and 2.4x faster.
    let at_least = registers.map_or(0, |schedule| schedule.lanes());
    // A narrowed or split attention runs only on a matrix unit (POOT-1476): the register tiles are the exact schedule,
    // and a narrowed read would round for no speed there.
    fragment
        .and_then(|config| attention_tiling_of(spec, caps, Some(config), at_least))
        .or(registers.filter(|_| spec.precision.is_exact()))
}

/// [`attention_tiling`]'s best tiling over the micro-tiles `fragment` selects (`None`: register tiles), of workgroups
/// of at least `min_lanes` lanes. A fragment tiling needs the device's register file: without it no budget is
/// known and none is chosen.
fn attention_tiling_of(
    spec: &poot_kernelgen::AttentionSpec,
    caps: &DeviceCaps,
    fragment: Option<poot_target::FragmentConfig>,
    min_lanes: usize,
) -> Option<poot_kernelgen::AttentionSchedule> {
    use poot_kernelgen::QOperand;
    let launch = caps.gemm_launch();
    let lanes_limit =
        (caps.max_workgroup_size[0] as usize).min(caps.max_workgroup_invocations as usize);
    let positions = std::iter::successors(Some(1usize), |br| Some(br * 2))
        .take_while(|br| br * spec.n_rep <= TILED_MAX_ROWS && *br < 2 * spec.lq);
    // A fragment workgroup is whole waves, down to the one wave a runtime that cannot widen its subgroups allows.
    let register_lanes: Vec<usize> = launch.tiled_lanes().collect();
    let (widths, key_blocks, operands): (&[usize], &[usize], &[QOperand]) = match fragment {
        Some(_) => (
            &FRAGMENT_LANES,
            &FRAGMENT_KEY_BLOCKS,
            &[QOperand::Staged, QOperand::Held],
        ),
        // The scalar schedule's key blocks are the device's calibrated figure
        // ([`poot_target::calibration::PerfCalibration::attention_key_blocks`]).
        None => (
            &register_lanes,
            launch.calibrated.figures().attention_key_blocks,
            &[QOperand::Staged],
        ),
    };
    // The registers one lane of a `lanes`-lane workgroup may hold across the key loop: a register tiling's
    // accumulators, or a fragment tiling's accumulators and held Q (none where the device reports no register file).
    let budget = |lanes: usize| -> Option<usize> {
        let lanes = u32::try_from(lanes).ok()?;
        match fragment {
            None => Some(launch.register_budget.accumulators(lanes) as usize),
            Some(_) => launch.register_budget.held(lanes).map(|held| held as usize),
        }
    };
    let candidates = positions.flat_map(|br| {
        widths
            .iter()
            .copied()
            .filter(move |&lanes| (min_lanes..=lanes_limit).contains(&lanes))
            .flat_map(move |lanes| {
                key_blocks
                    .iter()
                    .flat_map(move |&bc| operands.iter().map(move |&q| (br, lanes, bc, q)))
            })
    });
    candidates
        .flat_map(|(br, lanes, bc, q)| {
            let rows = br * spec.n_rep;
            let micros = attention_micros(fragment, (rows, bc, spec.dv, spec.dk), lanes, q);
            micros
                .into_iter()
                .map(move |(micro, held)| (br, lanes, bc, q, micro, held))
        })
        .filter_map(|(br, lanes, bc, q, micro, held)| {
            let rows = br * spec.n_rep;
            if held > budget(lanes)? {
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
            // The tiling must fill the device; one whose size is unknown cannot show it does.
            let fills = caps
                .compute_units
                .known()
                .is_some_and(|units| tiled.row_groups() >= units as usize);
            // Larger ranks first; a fragment tiling's width counts down (the narrowest wins). Among the
            // micro-tiles of one register tiling the fewest LDS reads per lane and key block win, then the
            // narrower tiles: the pairs the workgroup memory rejects are not candidates.
            let rank = match fragment {
                Some(_) => (
                    rows,
                    usize::MAX - lanes,
                    bc,
                    usize::from(q == QOperand::Held),
                    0,
                ),
                None => (
                    lanes,
                    rows,
                    bc,
                    0,
                    usize::MAX - attention_lds_reads(micro, spec, bc),
                ),
            };
            (lds <= caps.lds_bytes as usize && fills).then_some((rank, schedule))
        })
        .max_by_key(|&(rank, _)| rank)
        .map(|(_, schedule)| schedule)
}

/// The operand elements one lane reads from workgroup memory per key block of a register tiling: the score
/// product steps `dk` times over `rows + cols` of its tile, the output product `bc` times over those of its own.
/// The FMAs are the tiling's, so fewer reads is a better micro-tile (the most square). Fragment tilings: 0.
fn attention_lds_reads(
    micro: poot_kernelgen::AttentionMicro,
    spec: &poot_kernelgen::AttentionSpec,
    bc: usize,
) -> usize {
    match micro {
        poot_kernelgen::AttentionMicro::Scalar { score, out } => {
            (score.rows + score.cols) * spec.dk + (out.rows + out.cols) * bc
        }
        poot_kernelgen::AttentionMicro::Fragment { .. } => 0,
    }
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
            head: head @ ContractionHead::MatMul { .. },
            ..
        } => head.operands(),
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
    // One workgroup per output element on a 2-D grid (attn_scores_v_gemv_lds); Y must clear the target's grid cap.
    gemv_grid(out_numel, caps).1 <= caps.max_grid[1] as usize
}

/// How the rows of a per-row indexed contraction (`IndexedMatMul`, an indexed `PackedContraction`) run. `M` rows are
/// the experts a decode step selects, or the routed rows of a prefill chunk: few rows share no expert and each is
/// cheapest as its own Gemv, many share experts and the register-tiled schedule over the rows grouped by expert reads
/// each expert's weight tile once for all of them. Where one gives way to the other is the cost model's
/// ([`gemv_wins`]), not a row count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexedRows {
    /// Each row runs the per-row Gemv over its own expert's weight rows.
    Gemv,
    /// The rows are grouped by expert (`group_expert_rows`) and run the register-tiled schedule.
    Grouped,
}

/// The launch geometry a packed decode Gemv must run (POOT-1608), for the calibration sweep that times one
/// `(cols, rows, run, depth)` of the planner's own search space. The planner scores only the geometry equal to the pin:
/// it plans that launch, refuses the contraction with the generator's own refusal where the lane's registers do not
/// hold it, and plans the contraction as it would without a Gemv where the launch does not fit the workgroup limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedGemvPin {
    /// The columns per workgroup.
    pub cols: u32,
    /// The adjacent output rows a lane accumulates over one activation read.
    pub rows: u32,
    /// The lane's run of `K` elements per trip.
    pub run: u32,
    /// The trips of weight loads a lane keeps in flight.
    pub depth: u32,
}

/// The launch an `IndexedMatMul` of `m` rows and `n` columns would run as a [`kg::Schedule::Gemv`] on every backend,
/// or `None` when its launch would not fit the device's X grid even at one lane per column. The launch policy is the
/// dense Gemv's ([`gemv_launch`]) over the `m * n` outputs, widened to fit the grid ([`widened_to_fit_grid`]).
pub(crate) fn indexed_gemv_in(m: usize, n: usize, caps: &DeviceCaps) -> Option<GemvLaunch> {
    if m == 0 || n == 0 {
        return None;
    }
    let outputs = m.checked_mul(n)?;
    widened_to_fit_grid(gemv_launch(outputs, caps), outputs, caps)
}

/// `launch` with its columns per workgroup doubled (up to the workgroup width) until the `outputs` dot products launch
/// no more workgroups than the device's X grid holds, rather than folding or refusing; `None` when even a workgroup
/// per lane does not fit. The grid-fit rule of every indexed Gemv, dense and packed.
pub(crate) fn widened_to_fit_grid(
    launch: GemvLaunch,
    outputs: usize,
    caps: &DeviceCaps,
) -> Option<GemvLaunch> {
    let limit = caps.max_grid[0].max(1) as usize;
    std::iter::successors(Some(launch.cols), |&cols| cols.checked_mul(2))
        .take_while(|&cols| cols <= launch.width)
        .find(|&cols| outputs.div_ceil(cols as usize) <= limit)
        .map(|cols| GemvLaunch { cols, ..launch })
}

/// Whether a per-row indexed contraction runs as the Gemv `gemv` rather than as `grouped` (its rows grouped by
/// expert, `kg::TiledRows::Grouped`): the cost model's answer ([`kg::RegisterTiling::gemv_beats_grouped`]) unless the
/// compile pinned one. `gemv` is `None` where no Gemv launch fits the device, and then the rows are grouped.
pub(crate) fn gemv_wins(
    pin: Option<IndexedRows>,
    gemv: Option<kg::GemvShape>,
    grouped: kg::TiledShape,
    caps: &DeviceCaps,
) -> bool {
    match (pin, gemv) {
        (Some(IndexedRows::Grouped), _) | (_, None) => false,
        (Some(IndexedRows::Gemv), Some(_)) => true,
        (None, Some(gemv)) => kg::RegisterTiling::gemv_beats_grouped(gemv, grouped, caps),
    }
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
            ..
        } => {
            out_numel > fold_ceiling
                && dense_decode_gemv_in(eqn, out_shape, out_numel, None, caps).is_none()
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

#[cfg(test)]
mod reshape_stride_tests {
    use super::reshape_strides;

    /// The element offset `reshape_strides` addresses for row-major flat index `flat` of `shape`.
    fn offset(shape: &[usize], strides: &[usize], mut flat: usize) -> usize {
        let mut at = 0;
        for (&extent, &stride) in shape.iter().zip(strides).rev() {
            at += flat % extent * stride;
            flat /= extent;
        }
        at
    }

    /// POOT-1564 SC-l: every stride vector `reshape_strides` returns reads, for each flat index of the new
    /// shape, the element the old strided view holds at that flat index (NumPy's no-copy contract). Mutation:
    /// split an axis with its input stride instead of the row-major split inside it
    /// (`stride *= out_shape[k]` becomes `stride`): the split rows are off.
    #[test]
    fn a_composed_reshape_reads_the_element_the_view_holds_at_each_flat_index() {
        // (input shape, input strides, output shape): a slice of a wider row, a transpose, a broadcast
        // (stride 0), unit axes in both, a merge over a chain, and a split.
        let cases: &[(&[usize], &[usize], &[usize])] = &[
            (&[1, 1, 6], &[6144, 6144, 1], &[1, 1, 3, 2]),
            (&[2, 6], &[12, 1], &[2, 3, 2]),
            (&[2, 3, 4], &[12, 4, 1], &[6, 4]),
            (&[2, 3, 4], &[12, 4, 1], &[24]),
            (&[4, 3], &[1, 4], &[4, 3, 1]),
            (&[4, 3], &[1, 4], &[1, 4, 3]),
            (&[3, 1, 4], &[4, 99, 1], &[3, 4]),
            (&[5, 4], &[0, 1], &[5, 2, 2]),
            (&[2, 3], &[0, 0], &[6]),
            (&[1], &[7], &[1, 1]),
            (&[6, 4], &[8, 1], &[2, 3, 4]),
        ];
        for &(in_shape, in_strides, out_shape) in cases {
            let strides = reshape_strides(in_shape, in_strides, out_shape).unwrap_or_else(|| {
                panic!("{in_shape:?} {in_strides:?} -> {out_shape:?} has a stride per output axis")
            });
            let count: usize = in_shape.iter().product();
            for flat in 0..count {
                assert_eq!(
                    offset(out_shape, &strides, flat),
                    offset(in_shape, in_strides, flat),
                    "{in_shape:?} {in_strides:?} -> {out_shape:?} at flat index {flat}"
                );
            }
        }
    }

    /// POOT-1564 SC-m: a merge of axes whose strides do not chain has no single stride, so the reshape is
    /// refused and the slice under it materializes: the query/gate split of a `[hq, 2d]` row, merged to
    /// `hq * d`, and a transposed pair merged. Mutation: drop the chain check in `reshape_strides` (the `all`
    /// over `old[k].1 == old[k + 1].1 * old[k + 1].0`): these return a stride and the row goes red.
    #[test]
    fn a_merge_across_a_stride_gap_has_no_stride() {
        // `[hq = 3, d = 2]` read at stride `(2d, 1) = (4, 1)`: the gap of `d` between rows is not `d`.
        assert_eq!(reshape_strides(&[3, 2], &[4, 1], &[6]), None);
        // A transposed `[3, 4]` (strides 1, 3) merged into 12 elements.
        assert_eq!(reshape_strides(&[4, 3], &[1, 4], &[12]), None);
        // A broadcast row (stride 0) merged with a dense axis.
        assert_eq!(reshape_strides(&[2, 3], &[0, 1], &[6]), None);
    }
}
