//! Matmul and contraction-epilogue matmul planning, including the choice of the synthesized tiled GEMM.

use super::{
    BodyLimits, Capability, ContractionHead, ContractionSpec, DType, Eqn, FusedOperand,
    FusedRegion, Graph, KernelRequest, MatmulShapes, OpKind, Operand, PlanError, Planned, Scalar,
    ValueId, WeightLayout, fty_ty, kg, out_numel_saturating, value_ids,
};
use crate::dtype_widen::ConstLanes;
use crate::kernel_mapping::{
    check_widening_step, float_region_leaf_storage, map_fused_op, narrowing_store_op,
};
use crate::planner::{GraphTables, StorageDtypes};
use crate::predicates::{
    decode_gemv_plan, indexed_matmul_lowers, is_batched_tileable_matmul_in, is_decode_attn_gemv_in,
    is_decode_gemv_in, is_tileable_matmul_in, lds_gemv_width, packed_contraction_m_per_block,
    serial_matmul_fold,
};
use crate::refusal::{FusedStepGap, Site};
use poot_graph_ir::{ContractionRows, FragmentDtype, LocalId, OperandPrecision};
use poot_target::{Backend, DeviceCaps};

/// The launch of a `Tiled` contraction of `tiles` workgroups of `tile * tile` threads: the tiles laid out on the X
/// extent, or, above `caps.max_grid[0]`, folded onto a 2-D `(x_groups, y_groups)` grid the body reconstructs as
/// `gy * x_groups + gx` (`x_groups` rides in the kernel's metadata). Beyond `caps.max_grid[1]` too, or on any
/// `usize -> u32` overflow, a typed refusal naming the equation, before any dispatch: `finalize`'s generic
/// workgroup bump never touches a tiled body (it would corrupt the fixed-size LDS tile).
pub(super) struct FoldedTileGrid {
    /// The X extent in tiles; the metadata word the body reads.
    pub(super) x_groups: u32,
    /// The dispatch grid in threads.
    pub(super) grid: [u32; 3],
}

pub(super) fn folded_tile_grid(
    site: Site<'_>,
    tiles: usize,
    width: usize,
    caps: &DeviceCaps,
) -> Result<FoldedTileGrid, PlanError> {
    let overflow = || site.refuse(Capability::DispatchSizeOverflow);
    let tiles_u32 = u32::try_from(tiles).map_err(|_| overflow())?;
    let limit_x = caps.max_grid[0].max(1);
    let (x_groups, y_groups) = if tiles_u32 <= limit_x {
        (tiles_u32, 1)
    } else {
        let y_groups = tiles_u32.div_ceil(limit_x);
        if y_groups > caps.max_grid[1] {
            return Err(site.refuse(Capability::DispatchGridLimit {
                threads: tiles.saturating_mul(width),
                workgroups_x: tiles,
                workgroup_width: width,
                limit: limit_x as usize,
            }));
        }
        (limit_x, y_groups)
    };
    let threads_x = u32::try_from(width)
        .ok()
        .and_then(|width| x_groups.checked_mul(width))
        .ok_or_else(overflow)?;
    Ok(FoldedTileGrid {
        x_groups,
        grid: [threads_x, y_groups, 1],
    })
}

/// How a plain `MatMul` is lowered. The one decision [`plan`] and the contraction-epilogue fuse rule
/// ([`epilogue_head_lowers`]) share, so a head fuses exactly when the serial kernel is its plan anyway. Every
/// route is a generated contraction: the request in the plan's `KernelChoice` is the route, and the launch
/// shaping reads that, not a second derivation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MatMulRoute {
    /// The register-tiled GEMM (`kg::DenseTiledWeight::KnF32`) over an f32 `[K, N]` weight shared by every row.
    Tileable,
    /// The register-tiled GEMM with a `[K, N]` weight slab per batch entry (block): the per-expert MoE GEMM and
    /// the prefill attention matmuls.
    BatchedTileable,
    /// The decode GEMV (`M == 1`, rank-2 `[K, N]` f32 weight). A BF16 `[K, N]` weight lands here too and is
    /// refused by the mixed-dtype guard in `plan_eqn`: its only lowering is the dense Gemv over `[N, K]`.
    DecodeGemv,
    /// The decode attention `scores @ V` GEMV.
    AttnGemv,
    /// The one-thread-per-output kernel every other `MatMul` lands on.
    Serial,
}

/// Choose `eqn`'s `MatMul` route.
pub(super) fn matmul_route(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    caps: &DeviceCaps,
) -> MatMulRoute {
    if is_decode_gemv_in(g, eqn, out_shape, out_numel, caps) {
        MatMulRoute::DecodeGemv
    } else if is_tileable_matmul_in(g, eqn) {
        MatMulRoute::Tileable
    } else if is_batched_tileable_matmul_in(g, eqn) {
        MatMulRoute::BatchedTileable
    } else if is_decode_attn_gemv_in(g, eqn, out_shape, out_numel, caps) {
        MatMulRoute::AttnGemv
    } else {
        MatMulRoute::Serial
    }
}

/// Whether `eqn`, a contraction head, has an epilogue lowering that is also the plan it would get without one:
/// the epilogue fuse rule forms a [`OpKind::ContractionEpilogue`] only over such a head, so no head loses a faster schedule (a tensor-core,
/// tiled or decode-GEMV `MatMul`) to gain a fused store. The generated Gemv, serial and tiled bodies carry an
/// epilogue; a `MatMul` head's fragment, GEMV and batched-weight tiled schedules do not.
pub(crate) fn epilogue_head_lowers(g: &Graph, eqn: &Eqn, caps: &DeviceCaps) -> bool {
    let tables = GraphTables::from(g);
    let out_shape = &g.aval(eqn.out).shape;
    let out_numel = out_numel_saturating(out_shape);
    match &eqn.op {
        OpKind::MatMul { .. } => {
            out_shape.len() >= 2
                && matches!(
                    matmul_route(tables, eqn, out_shape, out_numel, caps),
                    MatMulRoute::Tileable | MatMulRoute::Serial
                )
        }
        // A dense head takes the generated Gemv at `M == 1` and the generated tiled GEMM above it, and both carry
        // an epilogue, so every dense head lowers (the weight dtypes the dense planner admits).
        OpKind::DenseContraction { .. } => true,
        // Every indexed schedule (Gemv and the register-tiled schedule over the grouped rows) carries an epilogue.
        OpKind::IndexedMatMul { .. } => indexed_matmul_lowers(tables, eqn),
        // The packed Gemv (one activation row per block) carries an epilogue; the block-diagonal tiled schedule does not.
        OpKind::PackedContraction {
            descriptor, rows, ..
        } => {
            crate::packed_block_float::admit_packed(
                crate::packed_block_float::PACKED_LOWERING,
                *descriptor,
                crate::packed_block_float::PackedKernelOpKind::Contraction,
                caps,
            )
            .is_ok()
                && match rows {
                    ContractionRows::Blocks(blocks) => {
                        packed_contraction_m_per_block(tables, eqn, *descriptor, *blocks) == 1
                    }
                    // An indexed table carries an epilogue on both its schedules (Gemv and the register-tiled schedule over the grouped rows).
                    ContractionRows::Indexed { .. } => true,
                }
        }
        _ => false,
    }
}

/// The kernelgen [`kg::Epilogue`] of a `ContractionEpilogue`'s `region` over output shape `out_shape` and
/// dtype `odt`: local 0 is the head result, locals `1..n_inputs` the eqn's extra inputs (leaves), later locals
/// the steps. A leaf has the head's dtype (a float the generated bodies round through), or, under an f32
/// epilogue, is a BF16 weight (a bias) that a `Cast` step widens by reading it through its packed lane
/// (`const_lanes.packed(position)`, the storage plan's choice); anything else is a typed refusal. A region whose last
/// step narrows its f32 result (POOT-1448) is an f32 epilogue with a narrow store, which `stores_narrow` says the
/// body that runs it has; without it the step is refused.
#[expect(clippy::too_many_arguments)]
pub(super) fn epilogue_of(
    site: Site<'_>,
    g: GraphTables<'_>,
    eqn: &Eqn,
    head_operands: usize,
    region: &FusedRegion,
    out_shape: &[usize],
    odt: DType,
    const_lanes: &ConstLanes,
    stores_narrow: bool,
) -> Result<kg::Epilogue, PlanError> {
    // A region whose last step narrows its f32 result (POOT-1448) computes in f32 and stores the narrow dtype.
    let narrowed = region.narrowed_output();
    let odt = if narrowed.is_some() { DType::F32 } else { odt };
    let dt = match odt {
        DType::F32 | DType::BF16 | DType::F16 => fty_ty(odt),
        _ => return Err(site.refuse(Capability::DtypeLowering)),
    };
    let last_step = region.steps.len().checked_sub(1);
    let mut leaves = Vec::new();
    for (position, operand) in eqn.inputs.iter().enumerate().skip(head_operands) {
        let Operand::Value(value) = operand else {
            return Err(site.refuse(Capability::DtypeLowering));
        };
        let leaf_dtype = g.aval(*value).dtype;
        let storage = if leaf_dtype == odt {
            None
        } else if odt == DType::F32 && leaf_dtype == DType::BF16 {
            let packed = const_lanes.packed(position);
            Some(float_region_leaf_storage(site, leaf_dtype, packed)?)
        } else {
            return Err(site.refuse(Capability::DtypeLowering));
        };
        leaves.push(kg::EpilogueLeaf {
            shape: g.aval(*value).shape.clone(),
            storage,
        });
    }
    let n_inputs = region.n_inputs;
    let local = |id: LocalId| match id.index() {
        0 => kg::EpilogueInput::Head,
        id if id < n_inputs => kg::EpilogueInput::Leaf(id - 1),
        id => kg::EpilogueInput::Step(id - n_inputs),
    };
    let steps = region
        .steps
        .iter()
        .enumerate()
        .map(|(j, step)| {
            let op = match narrowed {
                Some(to) if Some(j) == last_step => narrowing_store_op(site, to, stores_narrow)?,
                _ => {
                    check_widening_step(site, step.op, &step.inputs, |id| {
                        (1..n_inputs).contains(&id.index())
                    })?;
                    map_fused_op(site, step.op)?
                }
            };
            Ok(kg::EpilogueStep {
                op,
                inputs: step
                    .inputs
                    .iter()
                    .map(|operand| match operand {
                        FusedOperand::Local(id) => Ok(local(*id)),
                        FusedOperand::Lit(Scalar::F32(value)) => Ok(kg::EpilogueInput::Lit(*value)),
                        FusedOperand::Lit(Scalar::I32(value)) => {
                            Ok(kg::EpilogueInput::Lit(*value as f32))
                        }
                        FusedOperand::PackLane { .. } => Err(site
                            .refuse(Capability::FusedStep(FusedStepGap::PackedLaneInFloatRegion))),
                    })
                    .collect::<Result<Vec<_>, PlanError>>()?,
            })
        })
        .collect::<Result<Vec<_>, PlanError>>()?;
    Ok(kg::Epilogue {
        dt,
        out_shape: out_shape.to_vec(),
        leaves,
        steps,
        output: local(region.output),
    })
}

/// Whether this family lowers these storage dtypes: the matmul bodies take BF16 and F16 operands and
/// outputs, and never a packed E4M3FN payload.
pub(super) fn admits_dtypes(dtypes: StorageDtypes) -> Result<(), Capability> {
    if dtypes.touches_e4m3fn() {
        Err(Capability::E4m3Storage)
    } else {
        Ok(())
    }
}

/// The tiled shape of `blocks` products of `m_per_block` rows against an f32 `[K, N]` weight (the plain matmul's tiled
/// route).
fn kn_f32_shape(
    precision: OperandPrecision,
    blocks: usize,
    m_per_block: usize,
    n: usize,
    k: usize,
) -> kg::TiledShape {
    kg::TiledShape {
        blocks,
        m_per_block,
        out_per_block: n,
        k,
        layout: WeightLayout::Kn,
        rows: poot_kernelgen::TiledRows::Dense,
        terms: kg::StagedTerms::of(precision, None, None),
        weight_bits: kg::DenseTiledWeight::KnF32.bits(),
        activation_bits: kg::activation_bits(None),
    }
}

/// The tiled shape the cost model ranks for a `MatMul` that takes a tiled route, read as it declares (POOT-1476); `None`
/// for any other op or route, none of which has a matrix-unit schedule.
pub(super) fn matmul_cost_shape(g: &Graph, eqn: &Eqn, caps: &DeviceCaps) -> Option<kg::TiledShape> {
    let OpKind::MatMul { precision, .. } = eqn.op else {
        return None;
    };
    let tables = GraphTables::from(g);
    let out_shape = &g.aval(eqn.out).shape;
    let out_numel = out_numel_saturating(out_shape);
    let ids = value_ids(eqn);
    let a = &g.aval(ids[0]).shape;
    let r = out_shape.len();
    let (nn, kk) = (out_shape[r - 1], a[a.len() - 1]);
    match matmul_route(tables, eqn, out_shape, out_numel, caps) {
        MatMulRoute::Tileable => Some(kn_f32_shape(precision, 1, out_numel / nn.max(1), nn, kk)),
        MatMulRoute::BatchedTileable => Some(kn_f32_shape(
            precision,
            out_shape[..r - 2].iter().product(),
            out_shape[r - 2],
            nn,
            kk,
        )),
        MatMulRoute::DecodeGemv | MatMulRoute::AttnGemv | MatMulRoute::Serial => None,
    }
}

/// Plan one matmul-family equation (`MatMul`, or a `ContractionEpilogue` over a `MatMul` head).
#[expect(clippy::too_many_arguments)]
pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
    limits: BodyLimits,
    const_lanes: &ConstLanes,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let ids = value_ids(eqn);
    let fty = fty_ty;
    // How the equation reads its operands: every schedule below implements it.
    let precision = eqn.op.operand_precision().unwrap_or_default();
    let generate = |request: KernelRequest| Planned::generated(site, request, backend, caps);
    let planned = match &eqn.op {
        OpKind::MatMul { .. } => {
            let (a, b) = (shape(ids[0]), shape(ids[1]));
            let r = out_shape.len();
            let (nn, kk) = (out_shape[r - 1], a[a.len() - 1]);
            let route = matmul_route(g, eqn, out_shape, out_numel, caps);
            let shapes = || MatmulShapes {
                out_shape: out_shape.to_vec(),
                a_shape: a.clone(),
                b_shape: b.clone(),
            };
            // card 181 Bug 2: the one-thread-per-output reference kernel folds past the wg-bump's own
            // ceiling onto the same 2-D grid Unary/Binary/scatter/DUS use (see `is_elementwise_2d`).
            let serial = || {
                refuse_unless_exact(site, precision)?;
                generate(KernelRequest::Contraction(ContractionSpec::Serial {
                    dt: fty(odt),
                    epilogue: None,
                    shapes: shapes(),
                    fold: serial_matmul_fold(out_numel, caps),
                    precision,
                }))
            };
            // Every route plans a kernel whose single element type covers operands and output, so a mixed
            // operand/output matmul has no lowering: `widen_mismatched_matmul_dtypes` casts every such operand
            // before planning, and one that reaches here (a BF16 `[K, N]` decode projection) is refused by name
            // rather than read as f32 words.
            if ids[..2].iter().any(|&id| g.aval(id).dtype != odt) {
                return Err(site.refuse(Capability::DtypeLowering));
            }
            match route {
                MatMulRoute::Tileable => {
                    // The register-tiled GEMM over the f32 `[K, N]` weight, staged along `N` (POOT-1403).
                    // `is_tileable_matmul_in` gates the shapes (M > 1, rank-2 `[K, N]` weight) and the F32
                    // operands this f32 body reads.
                    dense_tiled_plan(
                        site,
                        backend,
                        kn_f32_shape(precision, 1, out_numel / nn.max(1), nn, kk),
                        kg::DenseTiledWeight::KnF32,
                        None,
                        precision,
                        None,
                        caps,
                    )?
                }
                MatMulRoute::BatchedTileable => {
                    // A[..,M,K] @ B[..,K,N] -> [..,M,N], one block of the register-tiled GEMM per batch entry
                    // over its own weight slab: E for the MoE per-expert GEMM (rank-3), or B*H for the prefill
                    // attention matmuls (rank-4). The same body as the shared-weight GEMM, block-major in its
                    // tile order, so the same dispatch splitting covers it.
                    dense_tiled_plan(
                        site,
                        backend,
                        kn_f32_shape(
                            precision,
                            out_shape[..r - 2].iter().product(),
                            out_shape[r - 2],
                            nn,
                            kk,
                        ),
                        kg::DenseTiledWeight::KnF32,
                        None,
                        precision,
                        None,
                        caps,
                    )?
                }
                MatMulRoute::DecodeGemv => {
                    // Decode GEMV (M=1 shared-weight matmul, single-seq or batched): the one generated
                    // `gemv_lds` body, one workgroup of `lds_gemv_width` lanes per output element, B baked as a
                    // const and the batch read from the flat output index, so the dispatch binds
                    // [x, weight, out] with no metadata buffer. `decode_gemv_plan` also owns the large-N
                    // watchdog-safety chunking (card 258): above the trigger it launches the same body over
                    // disjoint ranges of the output, never a different body.
                    refuse_unless_exact(site, precision)?;
                    decode_gemv_plan(site, backend, kk, nn, out_numel, precision, caps)?
                }
                MatMulRoute::AttnGemv => {
                    // decode attention scores@V: LDS-parallel reduction over cap (card 143 Lever 1b), the
                    // batched-per-row sibling of is_decode_gemv's shared-weight decode GEMV.
                    refuse_unless_exact(site, precision)?;
                    generate(KernelRequest::Contraction(
                        ContractionSpec::AttnScoresVGemv {
                            cap: kk,
                            d: nn,
                            width: lds_gemv_width(caps),
                            numel: out_numel,
                            precision,
                        },
                    ))?
                }
                MatMulRoute::Serial => {
                    // The naive one-thread-per-output fallback. A batched `Q @ K^T` at long ISL crosses
                    // `is_batched_tiled_gemm`'s device-known-miscompile tile-count ceiling well before its
                    // output numel crosses the 2-D fold's ceiling, so it lands here.
                    serial()?
                }
            }
        }
        OpKind::ContractionEpilogue {
            head: head @ ContractionHead::MatMul { .. },
            region,
        } => {
            // The head's operands and result share one dtype (the fuse rule's contract), and the generated
            // serial kernel is the lowering: it rounds the f32 accumulator through that dtype before the
            // epilogue reads it, as the unfused program's store does.
            if ids[..head.operands()]
                .iter()
                .any(|&id| g.aval(id).dtype != odt)
            {
                return Err(site.refuse(Capability::DtypeLowering));
            }
            let epilogue = epilogue_of(
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
            if matmul_route(g, eqn, out_shape, out_numel, caps) == MatMulRoute::Tileable {
                // The route `matmul_route` gives the unfused head: the generated tiled GEMM, storing the
                // epilogue's result per output element.
                let (a, nn) = (shape(ids[0]), out_shape[out_shape.len() - 1]);
                return dense_tiled_plan(
                    site,
                    backend,
                    kn_f32_shape(precision, 1, out_numel / nn.max(1), nn, a[a.len() - 1]),
                    kg::DenseTiledWeight::KnF32,
                    Some(epilogue),
                    precision,
                    None,
                    caps,
                );
            }
            refuse_unless_exact(site, precision)?;
            generate(KernelRequest::Contraction(ContractionSpec::Serial {
                dt: fty(odt),
                epilogue: Some(epilogue),
                shapes: MatmulShapes {
                    out_shape: out_shape.to_vec(),
                    a_shape: shape(ids[0]),
                    b_shape: shape(ids[1]),
                },
                fold: serial_matmul_fold(out_numel, caps),
                precision,
            }))?
        }
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | side_output_ops!()
        | packed_ops!()
        | moe_ops!()
        | elementwise_ops!()
        | attention_ops!()
        | scan_ops!()
        | block_top_k_ops!()
        | cast_ops!()
        | movement_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only matmul ops here"),
    };
    Ok(planned)
}

/// One dispatch of a register-tiled contraction: its thread grid and metadata buffer (`[K, out_per_block,
/// m_per_block, x_groups, tile_offset]`).
pub(super) struct TiledDispatch {
    pub(super) grid: [u32; 3],
    pub(super) meta: Vec<u32>,
}

/// The register tiling of `shape` on `caps` ([`kg::RegisterTiling::choose`]) and its launch: the tiles in
/// dispatches over disjoint, contiguous tile ranges, each folded onto the device's grid ([`folded_tile_grid`]),
/// with metadata `[K, out_per_block, m_per_block, x_groups, tile_offset]` (an indexed contraction appends its own).
/// One range covers every tile unless the device bounds a dispatch: its watchdog budget
/// (`packed_tiled_work` multiply-accumulates, [`tiles_per_dispatch`]) and its confirmed tiled-GEMM workgroup
/// ceiling (`known_miscompiles.tiled_gemm_max_workgroups`: a dispatch stays below it). Every `usize -> u32` step
/// is checked: an overflow is a typed refusal naming the equation, before any dispatch.
pub(super) fn register_tiled_launch(
    site: Site<'_>,
    shape: kg::TiledShape,
    caps: &DeviceCaps,
) -> Result<(kg::RegisterTiling, Vec<TiledDispatch>), PlanError> {
    let overflow = || site.refuse(Capability::DispatchSizeOverflow);
    // A narrowed or split read runs on a matrix unit or not at all (POOT-1476): a scalar tiling would round for no speed,
    // and the pass that narrowed the op asked this same function. A refusal, never a silent exact plan.
    let tiling = if shape.terms.fragment.is_some() {
        kg::RegisterTiling::fragment_tiling(shape, caps)
            .ok_or_else(|| site.refuse(Capability::NoMatrixUnitSchedule))?
    } else {
        kg::RegisterTiling::choose(shape, caps)
    };
    let (tile_rows, tile_cols) = tiling.output_tile();
    // A grouping's tile is one output tile: a tiling of other rows would read two buckets' rows as one expert's.
    if !shape.admits_tile_rows(tile_rows) {
        return Err(site.refuse(Capability::GroupedTileRows {
            block: match shape.rows {
                kg::TiledRows::Grouped { block, .. } => block,
                kg::TiledRows::Dense => tile_rows,
            },
            tile_rows,
        }));
    }
    let grid_rows = shape.grid_rows();
    let tiles = shape
        .blocks
        .checked_mul(grid_rows.div_ceil(tile_rows))
        .and_then(|v| v.checked_mul(shape.out_per_block.div_ceil(tile_cols)))
        .ok_or_else(overflow)?;
    let mut per_dispatch = tiles_per_dispatch(tiles, tile_rows * tile_cols, shape.k, caps);
    if let Some(ceiling) = caps.known_miscompiles.tiled_gemm_max_workgroups {
        per_dispatch = per_dispatch.min((ceiling as usize).saturating_sub(1).max(1));
    }
    let per_dispatch = wave_aligned_chunk(
        tiles,
        per_dispatch,
        caps.compute_units.known().map_or(1, |units| units as usize),
        caps.max_grid[0].max(1) as usize,
    );
    let extents = [shape.k, shape.out_per_block, grid_rows]
        .into_iter()
        .map(|extent| u32::try_from(extent).map_err(|_| overflow()))
        .collect::<Result<Vec<_>, _>>()?;
    let dispatches = (0..tiles)
        .step_by(per_dispatch.max(1))
        .map(|offset| {
            let count = (tiles - offset).min(per_dispatch);
            let FoldedTileGrid { x_groups, grid } =
                folded_tile_grid(site, count, tiling.workgroup_lanes(), caps)?;
            let offset = u32::try_from(offset).map_err(|_| overflow())?;
            let mut meta = extents.clone();
            meta.extend([x_groups, offset]);
            Ok(TiledDispatch { grid, meta })
        })
        .collect::<Result<Vec<_>, PlanError>>()?;
    Ok((tiling, dispatches))
}

/// The refusal of a narrowed or split read at a schedule that has no matrix unit (a serial or scalar body, a GEMV
/// with no matrix-unit alternative): it would round for no speed (POOT-1476).
pub(super) fn refuse_unless_exact(
    site: Site<'_>,
    precision: OperandPrecision,
) -> Result<(), PlanError> {
    if precision.is_exact() {
        Ok(())
    } else {
        Err(site.refuse(Capability::NoMatrixUnitSchedule))
    }
}

/// How many tiles of a register-tiled contraction one dispatch covers: all of them on a device with no watchdog
/// budget, else as many as keep a dispatch within `packed_tiled_work` multiply-accumulates (one tile is
/// `tile_outputs * k` of them, `tile_outputs` the outputs one workgroup computes), and at least one; a chunk that
/// folds is whole rows of the X grid.
fn tiles_per_dispatch(tiles: usize, tile_outputs: usize, k: usize, caps: &DeviceCaps) -> usize {
    let Some(budget) = caps.watchdog_budget else {
        return tiles;
    };
    let per_tile = (tile_outputs * k).max(1) as u64;
    let within_budget = usize::try_from(budget.packed_tiled_work / per_tile)
        .unwrap_or(usize::MAX)
        .clamp(1, tiles.max(1));
    // A chunk that folds onto several grid rows is a whole number of them: the folded grid launches full rows,
    // and a partial one would launch (and redo) the next chunk's first tiles.
    let row = caps.max_grid[0].max(1) as usize;
    if within_budget < tiles && within_budget > row {
        within_budget - within_budget % row
    } else {
        within_budget
    }
}

/// The tiles per dispatch of a launch of `tiles` that may carry at most `max_per_dispatch`: the bound rounded down to
/// whole waves of the device's `units` compute units, which run one tile each. A dispatch of 50 tiles on 40 units
/// takes two waves, the second one 10 tiles wide, and the dispatch after it starts only when that wave ends; 40 + 40
/// takes the same two waves with both full, so the split that fills every wave but the last costs the least. A
/// bound within one wave, or a chunk that folds onto several grid rows of `row` tiles (a whole number of them, see
/// [`tiles_per_dispatch`]), is left as it is.
fn wave_aligned_chunk(tiles: usize, max_per_dispatch: usize, units: usize, row: usize) -> usize {
    let units = units.max(1);
    if max_per_dispatch >= tiles || max_per_dispatch <= units {
        return max_per_dispatch;
    }
    let aligned = max_per_dispatch - max_per_dispatch % units;
    if aligned > row {
        max_per_dispatch
    } else {
        aligned
    }
}

/// Plan an `M > 1` dense contraction over `weight` as the register-tiled GEMM ([`register_tiled_launch`]), with
/// `epilogue` (f32) applied to each output before its store: one generated request per dispatch, a split plan when
/// the launch is several.
#[expect(clippy::too_many_arguments)] // the shape, both operands' storage, the epilogue and the device
pub(super) fn dense_tiled_plan(
    site: Site<'_>,
    backend: Backend,
    shape: kg::TiledShape,
    weight: kg::DenseTiledWeight,
    epilogue: Option<kg::Epilogue>,
    precision: OperandPrecision,
    activation_stored: Option<FragmentDtype>,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    let (tiling, dispatches) = register_tiled_launch(site, shape, caps)?;
    planned_dispatches(
        site,
        backend,
        caps,
        dispatches,
        |TiledDispatch { grid, meta }| {
            KernelRequest::Contraction(ContractionSpec::DenseTiled(kg::DenseTiledContraction {
                weight,
                schedule: tiling.schedule(),
                epilogue: epilogue.clone(),
                grid,
                meta,
                precision,
                activation_stored,
            }))
        },
    )
}

/// One generated request per dispatch: the plain plan of one, or a split plan of several.
pub(super) fn planned_dispatches(
    site: Site<'_>,
    backend: Backend,
    caps: &DeviceCaps,
    dispatches: Vec<TiledDispatch>,
    request: impl Fn(TiledDispatch) -> KernelRequest,
) -> Result<Planned, PlanError> {
    match <[TiledDispatch; 1]>::try_from(dispatches) {
        Ok([dispatch]) => Planned::generated(site, request(dispatch), backend, caps),
        Err(dispatches) => Planned::split(
            dispatches
                .into_iter()
                .map(|dispatch| Planned::generated_dispatch(site, request(dispatch), backend, caps))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::wave_aligned_chunk;

    /// A split launch fills every wave of every dispatch but the last: the bound rounds down to whole waves of the
    /// device's units, and a launch within its bound or a bound within one wave is not touched. Mutation: return
    /// `max_per_dispatch` for a split; the first row goes red.
    #[test]
    fn a_split_launch_fills_whole_waves_per_dispatch() {
        assert_eq!(
            wave_aligned_chunk(80, 50, 40, 65_535),
            40,
            "50 + 30 becomes 40 + 40"
        );
        assert_eq!(
            wave_aligned_chunk(320, 50, 40, 65_535),
            40,
            "eight full waves"
        );
        assert_eq!(wave_aligned_chunk(304, 190, 40, 65_535), 160, "160 + 144");
        assert_eq!(
            wave_aligned_chunk(7, 7, 40, 65_535),
            7,
            "under the bound: one dispatch"
        );
        assert_eq!(
            wave_aligned_chunk(100, 30, 40, 65_535),
            30,
            "a bound inside one wave"
        );
    }

    /// A chunk that folds onto several grid rows is a whole number of them, so the wave rounding gives way to the
    /// row. Mutation: drop the row guard; 280 tiles fold onto two and four fifths rows and this row goes red.
    #[test]
    fn a_folded_chunk_stays_whole_grid_rows() {
        assert_eq!(
            wave_aligned_chunk(700, 300, 40, 100),
            300,
            "280 would end mid-row"
        );
        assert_eq!(
            wave_aligned_chunk(700, 200, 40, 100),
            200,
            "200 is whole rows and whole waves"
        );
    }
}
