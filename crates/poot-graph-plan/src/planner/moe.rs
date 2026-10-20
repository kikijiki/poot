//! MoE equation planning: the indexed matmul (with its fused gate-weighting epilogue) and the ArgTopK rank
//! inversion.

use super::matmul::{FoldedTileGrid, folded_tile_grid};
use super::{
    BodyLimits, Capability, ContractionHead, ContractionSpec, DType, Eqn, HashMap, KernelRequest,
    OpKind, PlanError, Planned, TopKSpec, ValueId, kg, matmul, value_ids,
};
use crate::planner::{GraphTables, Layout, finalize};
use crate::predicates::{GEMM_TILE, indexed_gemv_in, indexed_matmul_lowers, view_source};
use crate::refusal::Site;
use poot_target::{Backend, BufferStorage, DeviceCaps};

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
    packed_operands: &[bool],
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let ids = value_ids(eqn);
    let mut planned = match &eqn.op {
        // card 088: x[M,K], W[E,K,N], idx[M] -> [M,N]; each row reads its own expert's slab in-kernel. With a
        // routed gate weighting fused behind it (`ContractionEpilogue`), the epilogue runs over each sum.
        OpKind::IndexedMatMul => plan_indexed(site, g, eqn, backend, views, caps, None)?,
        OpKind::ContractionEpilogue {
            head: ContractionHead::Indexed,
            region,
        } => {
            let epilogue = matmul::epilogue_of(
                site,
                g,
                eqn,
                ContractionHead::Indexed.operands(),
                region,
                out_shape,
                odt,
                packed_operands,
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
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
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
    let Some(view) = views.get(&value) else {
        return Ok(kg::WeightLayout::Kn);
    };
    let (k, n) = (shape[1], shape[2]);
    let stored_nk = Layout {
        strides: vec![k * n, 1, k],
        offset: 0,
    };
    if *view == stored_nk {
        Ok(kg::WeightLayout::Nk)
    } else {
        Err(site.refuse(Capability::OperandLayout { value }))
    }
}

/// One generated indexed contraction ([`ContractionSpec::Indexed`]): `x[M, K]` against the stacked `W[E, K, N]`
/// (or its checkpoint `[E, N, K]` view) with row `m` reading slab `idx[m]`, optionally with an epilogue over each
/// sum. The schedule is chosen from shapes and the device alone ([`indexed_gemv_in`]): the Gemv at decode row
/// counts, else the segmented Tiled, whose grid is folded onto `GroupY` above the device's X extent and refused
/// with a typed error beyond that too.
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
    if !indexed_matmul_lowers(g, eqn) {
        return Err(site.refuse(Capability::DtypeLowering));
    }
    let (x, w) = (g.aval(ids[0]).shape.clone(), g.aval(ids[1]).shape.clone());
    let (m, k, experts, n) = (x[0], x[1], w[0], w[2]);
    let layout = weight_layout(site, ids[1], &w, views)?;
    // The BF16 stack is read as packed lanes, which only a const upload (or a view of one, read in place) has.
    let weight = match g.aval(ids[1]).dtype {
        DType::F32 => BufferStorage::f32(),
        DType::BF16
            if g.values[view_source(g, views, ids[1])]
                .storage
                .holds_weight_bytes() =>
        {
            BufferStorage::bf16_packed()
        }
        _ => return Err(site.refuse(Capability::DtypeLowering)),
    };
    if experts == 0 {
        return Err(site.refuse(Capability::OperandShape {
            value: ids[1],
            shape: w,
            expected: vec![1, k, n],
        }));
    }
    let overflow = || site.refuse(Capability::DispatchSizeOverflow);
    let (schedule, grid, x_groups) = match indexed_gemv_in(m, n, caps) {
        Some(schedule) => {
            let threads = u32::try_from(schedule.grid_threads(1, m, n)).map_err(|_| overflow())?;
            (schedule, [threads, 1, 1], 0)
        }
        None => {
            let tile = kg::TileSize::new(GEMM_TILE as u32)
                .map_err(|error| site.refuse(Capability::KernelGen(error)))?;
            let ts = tile.get() as usize;
            let tiles = m
                .div_ceil(ts)
                .checked_mul(n.div_ceil(ts))
                .ok_or_else(overflow)?;
            let FoldedTileGrid { x_groups, grid } = folded_tile_grid(site, tiles, ts, caps)?;
            (kg::Schedule::Tiled { tile }, grid, x_groups)
        }
    };
    let meta = [k, n, experts, m]
        .into_iter()
        .map(|extent| u32::try_from(extent).map_err(|_| overflow()))
        .chain([Ok(x_groups)])
        .collect::<Result<Vec<_>, _>>()?;
    Planned::generated(
        site,
        KernelRequest::Contraction(ContractionSpec::Indexed(kg::IndexedContraction {
            schedule,
            layout,
            weight,
            epilogue,
            grid,
            meta,
        })),
        backend,
        caps,
    )
}
