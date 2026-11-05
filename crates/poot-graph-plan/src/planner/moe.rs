//! MoE equation planning: the indexed matmul (with its fused gate-weighting epilogue), the ArgTopK rank
//! inversion and the expert grouping of the routed rows.

use super::matmul::{TiledDispatch, planned_dispatches, register_tiled_launch};
use super::{
    BodyLimits, Capability, ContractionHead, ContractionSpec, DType, Eqn, HashMap, KernelRequest,
    OpKind, PlanError, Planned, TopKSpec, ValueId, kg, matmul, value_ids,
};
use crate::dtype_widen::ConstLanes;
use crate::planner::{GraphTables, Layout, StorageDtypes, finalize};
use crate::predicates::{indexed_gemv_in, indexed_matmul_lowers, view_source};
use crate::refusal::Site;
use poot_graph_ir::ExpertRows;
use poot_target::{Backend, BufferStorage, DeviceCaps};

/// Whether this family lowers these storage dtypes: the indexed matmul and the rank inversion are f32/i32
/// bodies (`indexed_matmul_lowers`), so a 2-byte float or a packed E4M3FN payload is refused by name.
pub(super) fn admits_dtypes(dtypes: StorageDtypes) -> Result<(), Capability> {
    dtypes.refuse_unwired()
}

/// Plan one MoE-family equation (IndexedMatMul, its `ContractionEpilogue`, or ArgTopK).
#[expect(clippy::too_many_arguments)]
pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    views: &HashMap<ValueId, Layout>,
    odt: DType,
    caps: &DeviceCaps,
    limits: BodyLimits,
    const_lanes: &ConstLanes,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let ids = value_ids(eqn);
    let mut planned = match &eqn.op {
        // card 088: x[M,K], W[E,K,N], idx[M] -> [M,N]; each row reads its own expert's slab in-kernel. With a
        // routed gate weighting fused behind it (`ContractionEpilogue`), the epilogue runs over each sum.
        OpKind::IndexedMatMul { .. } => plan_indexed(site, g, eqn, backend, views, caps, None)?,
        OpKind::ContractionEpilogue {
            head: head @ ContractionHead::Indexed { .. },
            region,
        } => {
            let epilogue = matmul::epilogue_of(
                site,
                g,
                eqn,
                head.operands(),
                region,
                out_shape,
                odt,
                const_lanes,
                // Only the dense tiled body stores a narrowed output (POOT-1448).
                false,
            )?;
            plan_indexed(site, g, eqn, backend, views, caps, Some(epilogue))?
        }
        OpKind::ArgTopK { k } => {
            // spec 136 P2: rank[..,E] -> [..,k] expert ids, one thread per output element, a naive
            // serial scan over E per thread (FR-003; routing is not the hot path, no perf work here).
            // `e` and `k` are both baked (trace-time constants); F32-only (fork c), so no `dt`/`fty`.
            let rank_shape = shape(ids[0]);
            let Some(&e) = rank_shape.last() else {
                return Err(PlanError::BadShape(
                    "ArgTopK rank operand needs a last (expert) axis, got a 0-D shape".into(),
                ));
            };
            Planned::generated(
                site,
                KernelRequest::TopK(TopKSpec {
                    e,
                    k: *k,
                    numel: out_numel,
                }),
                backend,
                caps,
            )?
        }
        OpKind::ExpertGroup { experts, block } => Planned::generated(
            site,
            KernelRequest::ExpertGroup(kg::ExpertGroupSpec {
                experts: *experts,
                block: *block,
            }),
            backend,
            caps,
        )?,
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | side_output_ops!()
        | packed_ops!()
        | elementwise_ops!()
        | matmul_ops!()
        | attention_ops!()
        | scan_ops!()
        | block_top_k_ops!()
        | cast_ops!()
        | movement_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only MoE ops here"),
    };
    finalize(
        &mut planned,
        g,
        eqn,
        out_shape,
        out_numel,
        backend,
        odt,
        caps,
    )?;
    Ok(planned)
}

/// How the stacked weight `w` (logical `[E, K, N]`) sits in its buffer: row-major `[E, K, N]` when it is no view,
/// and the checkpoint's `[E, N, K]` when it is the view a `Transpose { perm: [0, 2, 1] }` of such a stack became
/// ([`crate::compute_views`]), which the contraction reads in place. Any other view is refused, typed.
fn weight_layout(
    site: Site<'_>,
    value: ValueId,
    shape: &[usize],
    views: &HashMap<ValueId, Layout>,
) -> Result<kg::WeightLayout, PlanError> {
    stored_layout(value, shape, views)
        .ok_or_else(|| site.refuse(Capability::OperandLayout { value }))
}

/// [`weight_layout`]'s layout, `None` for a view it refuses.
fn stored_layout(
    value: ValueId,
    shape: &[usize],
    views: &HashMap<ValueId, Layout>,
) -> Option<kg::WeightLayout> {
    let Some(view) = views.get(&value) else {
        return Some(kg::WeightLayout::Kn);
    };
    let (k, n) = (shape[1], shape[2]);
    let stored_nk = Layout {
        strides: vec![k * n, 1, k],
        offset: 0,
    };
    (*view == stored_nk).then_some(kg::WeightLayout::Nk)
}

/// The storage the stacked weight `w` is read in: f32, or BF16 packed lanes, which only a const upload (or a view
/// of one, read in place) has; `None` for any other.
fn stack_storage(
    g: GraphTables<'_>,
    views: &HashMap<ValueId, Layout>,
    w: ValueId,
) -> Option<BufferStorage> {
    match g.aval(w).dtype {
        DType::F32 => Some(BufferStorage::f32()),
        DType::BF16
            if g.values[view_source(g, views, w)]
                .storage
                .holds_weight_bytes() =>
        {
            Some(BufferStorage::bf16_packed())
        }
        _ => None,
    }
}

/// The tiled shape of a dense indexed contraction of `m` rows of `k` against `experts` stacked `[K, n]` weights
/// stored in `layout` and `weight`, its rows meeting the tiles as `rows` says. The one description both the planner
/// and the `group_expert_rows` pass's block choice read.
fn indexed_tiled_shape(
    (m, k, n, _experts): (usize, usize, usize, usize),
    layout: kg::WeightLayout,
    weight: BufferStorage,
    precision: poot_graph_ir::OperandPrecision,
    rows: kg::TiledRows,
) -> kg::TiledShape {
    kg::TiledShape {
        blocks: 1,
        m_per_block: m,
        out_per_block: n,
        k,
        layout,
        rows,
        terms: kg::StagedTerms::of(precision, None, kg::stored_fragment_dtype(weight)),
        weight_bits: kg::dense_value_bits(weight),
        activation_bits: kg::activation_bits(None),
    }
}

/// The tiled shape of a per-row dense indexed contraction (`IndexedMatMul`) the planner would tile on `caps` (its
/// rows past the Gemv's), as grouped rows over its `experts + 1` buckets (the block is the caller's to choose);
/// `None` for one it runs as a Gemv or cannot plan. What the `group_expert_rows` pass groups.
pub(crate) fn grouping_candidate(
    g: &poot_graph_ir::Graph,
    eqn: &Eqn,
    views: &HashMap<ValueId, Layout>,
    caps: &DeviceCaps,
) -> Option<kg::TiledShape> {
    let OpKind::IndexedMatMul {
        rows: ExpertRows::PerRow,
        precision,
    } = eqn.op
    else {
        return None;
    };
    let tables = GraphTables::from(g);
    let ids = value_ids(eqn);
    if !indexed_matmul_lowers(tables, eqn) {
        return None;
    }
    let (x, w) = (&tables.aval(ids[0]).shape, &tables.aval(ids[1]).shape);
    let (m, k, experts, n) = (x[0], x[1], w[0], w[2]);
    if experts == 0 || indexed_gemv_in(m, n, caps).is_some() {
        return None;
    }
    Some(indexed_tiled_shape(
        (m, k, n, experts),
        stored_layout(ids[1], w, views)?,
        stack_storage(tables, views, ids[1])?,
        precision,
        kg::TiledRows::Grouped {
            groups: experts + 1,
            block: 0,
        },
    ))
}

/// One generated indexed contraction ([`ContractionSpec::Indexed`]): `x[M, K]` against the stacked `W[E, K, N]`
/// (or its checkpoint `[E, N, K]` view), optionally with an epilogue over each sum. Per-row ids run the Gemv at
/// decode row counts ([`indexed_gemv_in`], from shapes and the device alone) and are refused past them; rows grouped
/// by expert ([`ExpertRows::Grouped`]) run the tiled schedule over the grouping ([`register_tiled_launch`] over one
/// block of the grouping's positions, the tiling's output tile the grouping's block), split and folded as every
/// tiled contraction is.
fn plan_indexed(
    site: Site<'_>,
    g: GraphTables<'_>,
    eqn: &Eqn,
    backend: Backend,
    views: &HashMap<ValueId, Layout>,
    caps: &DeviceCaps,
    epilogue: Option<kg::Epilogue>,
) -> Result<Planned, PlanError> {
    let ids = value_ids(eqn);
    let precision = eqn.op.operand_precision().unwrap_or_default();
    if !indexed_matmul_lowers(g, eqn) {
        return Err(site.refuse(Capability::DtypeLowering));
    }
    let (x, w) = (g.aval(ids[0]).shape.clone(), g.aval(ids[1]).shape.clone());
    let (m, k, experts, n) = (x[0], x[1], w[0], w[2]);
    let layout = weight_layout(site, ids[1], &w, views)?;
    let weight =
        stack_storage(g, views, ids[1]).ok_or_else(|| site.refuse(Capability::DtypeLowering))?;
    if experts == 0 {
        return Err(site.refuse(Capability::OperandShape {
            value: ids[1],
            shape: w,
            expected: vec![1, k, n],
        }));
    }
    let overflow = || site.refuse(Capability::DispatchSizeOverflow);
    let word = |extent: usize| u32::try_from(extent).map_err(|_| overflow());
    let rows = match &eqn.op {
        OpKind::IndexedMatMul { rows, .. }
        | OpKind::ContractionEpilogue {
            head: ContractionHead::Indexed { rows, .. },
            ..
        } => *rows,
        _ => unreachable!("plan_indexed plans an indexed head"),
    };
    let (schedule, dispatches) = match (rows, indexed_gemv_in(m, n, caps)) {
        (ExpertRows::PerRow, Some(launch)) => {
            let schedule = launch.schedule(backend, k);
            let threads = word(schedule.grid_threads(1, m, n))?;
            let meta = vec![word(k)?, word(n)?, word(m)?, 0, 0, word(experts)?];
            (
                schedule,
                vec![TiledDispatch {
                    grid: [threads, 1, 1],
                    meta,
                }],
            )
        }
        (ExpertRows::PerRow, None) => return Err(site.refuse(Capability::UngroupedExpertRows)),
        (ExpertRows::Grouped { block }, _) => {
            let shape = indexed_tiled_shape(
                (m, k, n, experts),
                layout,
                weight,
                precision,
                kg::TiledRows::Grouped {
                    groups: experts + 1,
                    block,
                },
            );
            let (tiling, mut dispatches) = register_tiled_launch(site, shape, caps)?;
            let experts = word(experts)?;
            for dispatch in &mut dispatches {
                dispatch.meta.push(experts);
            }
            (tiling.schedule(), dispatches)
        }
    };
    planned_dispatches(
        site,
        backend,
        caps,
        dispatches,
        |TiledDispatch { grid, meta }| {
            KernelRequest::Contraction(ContractionSpec::Indexed(kg::IndexedContraction {
                schedule,
                layout,
                weight,
                epilogue: epilogue.clone(),
                grid,
                meta,
                precision,
            }))
        },
    )
}
