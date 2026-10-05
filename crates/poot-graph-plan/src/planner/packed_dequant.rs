//! Card 542a: planning for `PackedDequant`/`PackedContraction` when the descriptor-driven table
//! (`packed_block_float::PACKED_LOWERING`) admits the descriptor's format - the block-32 GGUF schemes
//! today (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, MXFP4). A format the table does not admit for this op
//! is not planned here, so `try_plan` below returns `Ok(None)` for it and the caller falls back to
//! today's unconditional refusal.
//!
//! `OpKind::PackedDequant`/`PackedContraction` carry their carrier operands in
//! `descriptor.sources()`'s role order (`op.rs`'s shape inference proves this at graph-build time), so
//! the ordinary `Plan::ComputeMeta` consumer (`value_ids(eqn)` bound to kernel params 1..=n, in every
//! executor's typed-value walk and its own `run()`) already binds every source in role order: D9's
//! "role-ordered source binding" is this convention, not new per-executor code. The one place role
//! order has to be threaded by hand is the generated kernel's own declared parameter order
//! (`poot_kernelgen::packed_kernel`), which must match `eqn.inputs`'s order exactly - `PackedDequant`
//! declares `(words, metadata, out)` (its one `Blocks` source), `PackedContraction` declares
//! `(activation, words, metadata, out)` (activation is `eqn.inputs[0]`, the carrier is
//! `eqn.inputs[1]`).

use super::*;
use poot_kernelgen::{PackedKernelOp, RowSelect, Schedule};
use poot_target::{Backend, DeviceCaps};

/// `Ok(None)` when card 542a's table does not admit this equation's descriptor for this op: the
/// caller falls back to `Capability::ImportedKernelPlanning`, the same refusal every unadmitted
/// imported op gets today.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
    limits: BodyLimits,
) -> Result<Option<Planned>, PlanError> {
    match &eqn.op {
        OpKind::PackedDequant { descriptor } => try_plan_materialize(
            g,
            eqn,
            out_shape,
            out_numel,
            backend,
            odt,
            caps,
            limits,
            *descriptor,
        ),
        OpKind::PackedContraction { descriptor, blocks } => try_plan_contraction(
            g,
            eqn,
            out_shape,
            out_numel,
            backend,
            odt,
            caps,
            limits,
            *descriptor,
            *blocks,
        ),
        OpKind::PackedRowGather { descriptor } => try_plan_row_gather(
            g,
            eqn,
            out_shape,
            out_numel,
            backend,
            odt,
            caps,
            limits,
            *descriptor,
        ),
        _ => unreachable!("try_plan is called only for imported_ops!()"),
    }
}

/// `out[o, k] = decode(o, k)`: the standalone `PackedDequant` equation. Params `(words, metadata,
/// out)` - `descriptor.sources()` is `PackedDequant`'s whole operand list (`eqn.inputs`), so there is
/// nothing else to bind in front of the metadata word.
#[allow(clippy::too_many_arguments)]
fn try_plan_materialize(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
    limits: BodyLimits,
    descriptor: poot_quant::PackedWeight,
) -> Result<Option<Planned>, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let op = PackedKernelOp::Materialize;
    let spec = match crate::packed_block_float::select_packed_kernel(
        crate::packed_block_float::PACKED_LOWERING,
        descriptor,
        op,
        backend,
    ) {
        Ok(spec) => spec,
        Err(_) => return Ok(None),
    };
    let k = descriptor.shape()[1];
    let mut planned = Planned::generated(
        site,
        KernelRequest::Packed(PackedRequest {
            name: "packed_materialize",
            spec,
            grid: [out_numel as u32, 1, 1],
            meta: vec![k as u32],
        }),
        backend,
        caps,
    )?;
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
    Ok(Some(planned))
}

/// `out[.., :] = decode(ids[..], :)`: the `PackedRowGather` equation (quantized token
/// embeddings). Params `(words, ids, metadata, out)` - `eqn.inputs` is the one `Blocks` carrier then
/// the row ids, bound to kernel params 1..=n in that order; one thread per output element.
#[allow(clippy::too_many_arguments)]
fn try_plan_row_gather(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
    limits: BodyLimits,
    descriptor: poot_quant::PackedWeight,
) -> Result<Option<Planned>, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let spec = match crate::packed_block_float::select_packed_kernel(
        crate::packed_block_float::PACKED_LOWERING,
        descriptor,
        PackedKernelOp::RowGather,
        backend,
    ) {
        Ok(spec) => spec,
        Err(_) => return Ok(None),
    };
    let k = descriptor.shape()[1];
    let mut planned = Planned::generated(
        site,
        KernelRequest::Packed(PackedRequest {
            name: "packed_row_gather",
            spec,
            grid: [out_numel as u32, 1, 1],
            meta: vec![k as u32],
        }),
        backend,
        caps,
    )?;
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
    Ok(Some(planned))
}

/// `out[row, col] = sum over k of activation[row, k] * decode(weight_row(row, col), k)`: the
/// `PackedContraction` equation. Params `(activation, words, metadata, out)` - `eqn.inputs[0]` is the
/// activation, `eqn.inputs[1]` the one `Blocks` carrier. The schedule is chosen from shapes alone
/// (dquant.md D3, section 5): `M == 1` (one activation row per block) is `Schedule::Gemv`, its launch
/// shape from the output count and the device's compute units ([`gemv_schedule`], card 653); `M > 1`
/// is `Schedule::Tiled` (card 542b, dquant.md D6/Q9), so GGUF prefill does not fall back to
/// `Schedule::Serial` now that Card 545b has deleted the hand-written `TiledRegionDequant` family.
/// Card 658: `Schedule::Tiled` now lays its row-tiles out per block (`poot_kernelgen`'s
/// `contraction_tiled` doc), so a block-diagonal split (`blocks > 1`) no longer needs `m_per_block` a
/// multiple of `tile` and no longer falls back to the per-thread-unbounded `Schedule::Serial` (since
/// deleted) - the same no-cooperative-loading shape that pathologically hung the watchdog at real
/// prefill dimensions before Card 163. Never from the format, matching
/// every other packed-kernel schedule choice.
///
/// **Card 658 review F1/F2/F7: the Tiled grid is folded and checked, never truncated.** `tiles =
/// blocks * ceil(m_per_block/tile) * ceil(out_per_block/tile)` is this schedule's real workgroup
/// count ([`Schedule::grid_threads`] divided by `tile*tile`); above `caps.max_grid[0]` it is folded
/// onto a 2-D `(x_groups, y_groups)` grid the body reconstructs as `gy*x_groups+gx`
/// (`poot_kernelgen`'s `contraction_tiled` doc - the same `GroupY`-spill convention `gemv_lds`/
/// `flash_region_prefill` use, `x_groups` riding in `metadata[3]` rather than baked, matching this
/// kernel family's own shape-in-metadata design). Above `caps.max_grid[1]` too, or on any `usize ->
/// u32` overflow along the way, `try_plan_contraction` refuses with `Capability::DispatchGridLimit`/
/// `DispatchSizeOverflow` naming the equation, before any dispatch - `finalize`'s generic wg-bump
/// never touches this arm (`kernel_mapping::shape_launch` exempts every packed contraction request by
/// type), since bumping the baked 64-lane workgroup would corrupt the fixed-size LDS tile regardless of
/// the grid.
#[allow(clippy::too_many_arguments)]
fn try_plan_contraction(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
    limits: BodyLimits,
    descriptor: poot_quant::PackedWeight,
    blocks: usize,
) -> Result<Option<Planned>, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let k = descriptor.shape()[1];
    let out = descriptor.shape()[0];
    let out_per_block = out / blocks;
    let m_per_block = packed_contraction_m_per_block(g, eqn, descriptor, blocks);
    let rows = blocks * m_per_block;
    let (schedule, meta, grid) = if m_per_block == 1 {
        let schedule = gemv_schedule(rows * out_per_block, caps.compute_units);
        let grid = [
            schedule.grid_threads(blocks, m_per_block, out_per_block) as u32,
            1,
            1,
        ];
        (schedule, vec![k as u32, out_per_block as u32], grid)
    } else {
        let tile = poot_kernelgen::TileSize::new(GEMM_TILE as u32)
            .map_err(|error| site.refuse(Capability::KernelGen(error)))?;
        let schedule = Schedule::Tiled { tile };
        let ts = tile.get() as usize;
        let tiles = blocks
            .checked_mul(m_per_block.div_ceil(ts))
            .and_then(|v| v.checked_mul(out_per_block.div_ceil(ts)))
            .ok_or_else(|| site.refuse(Capability::DispatchSizeOverflow))?;
        let tiles_u32 =
            u32::try_from(tiles).map_err(|_| site.refuse(Capability::DispatchSizeOverflow))?;
        let limit_x = caps.max_grid[0].max(1);
        let (x_groups, y_groups) = if tiles_u32 <= limit_x {
            (tiles_u32, 1u32)
        } else {
            let y_groups = tiles_u32.div_ceil(limit_x);
            if y_groups > caps.max_grid[1] {
                return Err(site.refuse(Capability::DispatchGridLimit {
                    threads: tiles,
                    workgroups_x: tiles,
                    workgroup_width: ts * ts,
                    limit: limit_x as usize,
                }));
            }
            (limit_x, y_groups)
        };
        let threads_x = x_groups
            .checked_mul((ts * ts) as u32)
            .ok_or_else(|| site.refuse(Capability::DispatchSizeOverflow))?;
        let meta = vec![k as u32, out_per_block as u32, m_per_block as u32, x_groups];
        (schedule, meta, [threads_x, y_groups, 1])
    };
    let op = PackedKernelOp::Contraction {
        rows: RowSelect::Dense,
        schedule,
    };
    let spec = match crate::packed_block_float::select_packed_kernel(
        crate::packed_block_float::PACKED_LOWERING,
        descriptor,
        op,
        backend,
    ) {
        Ok(spec) => spec,
        Err(_) => return Ok(None),
    };
    let mut planned = Planned::generated(
        site,
        KernelRequest::Packed(PackedRequest {
            name: "packed_contraction",
            spec,
            grid,
            meta,
        }),
        backend,
        caps,
    )?;
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
    Ok(Some(planned))
}

/// The `Schedule::Gemv` launch policy for a packed contraction with `outputs` output columns (card
/// 653): the widest candidate `cols` that still launches at least two
/// workgroups per compute unit (S60-3), else the narrowest - over this arm's own measured table.
/// A packed weight is `[N, K]` with `K` contiguous, so `cols` sets how many lanes share one row
/// (`256 / cols`) and so how long a contiguous run each trip reads; at well under two bytes per
/// element the packed body wants more lanes per row than the dense BF16 column-tile table ({64, 32},
/// else 16, 4-element runs) gives. Chosen from a per-kernel sweep on gfx1151 over the qwen2.5-0.5b
/// GGUF projection shapes (card 653's receipt): {16, 8} with 8-element runs was 3.7 ms of Gemv
/// device time per Q4_K_M token against 4.2 ms for the dense table and 9.7 ms for the pre-card
/// one-column-per-workgroup body.
fn gemv_schedule(outputs: usize, compute_units: u32) -> Schedule {
    let min_workgroups = 2 * compute_units as usize;
    let cols = [16, 8]
        .into_iter()
        .find(|&cols| outputs.div_ceil(cols) >= min_workgroups)
        .unwrap_or(8);
    Schedule::Gemv {
        width: 256,
        cols: cols as u32,
        unroll: 8,
    }
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Builder, OpKind, Slot, TensorType};
    use poot_kernelgen::{KernelRequest, PackedKernelOp, RowSelect, Schedule};
    use poot_quant::PackedWeight;
    use poot_quant::format::WeightFormat;
    use poot_target::{Backend, DeviceCaps};

    use crate::{Capability, PlanError};

    use crate::{
        CompileError, CompileOptions, FusionPolicy, KernelChoice, Plan, Submission, Target, compile,
    };

    /// SC-003 (card 653): a decode (`M == 1`) packed contraction compiled for a device with a
    /// fixture `DeviceCaps::compute_units` plans the Gemv launch of `gemv_schedule`'s table - 16
    /// columns per workgroup while that still launches `2 x compute_units` workgroups, else 8,
    /// 256 lanes, 8-element runs - in the plan key, the grid and the body's workgroup. `N = 1280`
    /// and `1264` straddle the 80-workgroup floor at 40 compute units; the same `N = 896` flips
    /// with the compute units alone. Driven through `compile`. Mutation: pin
    /// `cols` to 16 in `gemv_schedule`; every row whose entry is 8 goes red.
    #[test]
    fn packed_decode_gemv_launch_follows_the_compute_unit_table() {
        let k = 256;
        for (compute_units, n, cols) in [
            (40, 151_936, 16),
            (40, 4864, 16),
            (40, 1280, 16),
            (40, 1264, 8),
            (40, 896, 8),
            (40, 128, 8),
            (4, 896, 16),
        ] {
            let what = format!("compute_units={compute_units} N={n}");
            let builder = Builder::new();
            let x = builder.slot_named(Slot::Activation, "x", TensorType::f32(vec![1, k]));
            let weight = PackedWeight::try_new(WeightFormat::Q8_0, [n, k]).unwrap();
            let out = poot_graph_ir::ops::packed_linear(&builder, x, "layer", weight, None, None)
                .unwrap();
            let target = Target {
                backend: Backend::SpirvVulkan,
                caps: DeviceCaps {
                    compute_units,
                    ..DeviceCaps::wgpu_rdna3_igpu()
                },
            };
            let options = CompileOptions {
                execution: Submission::Replay,
                fusion: FusionPolicy::Full,
                limits: crate::CompileLimits::STANDARD,
            };
            let program = compile(&builder.finish(out), &target, &options)
                .unwrap_or_else(|error| panic!("{what}: {error}"));
            let (eqn, plan) = program
                .planned()
                .find(|(eqn, _)| matches!(eqn.op, OpKind::PackedContraction { .. }))
                .unwrap_or_else(|| panic!("{what}: no PackedContraction planned"));
            let Plan::ComputeMeta { body, grid, .. } = plan else {
                panic!("{what}: expected a ComputeMeta plan, got {plan:?}");
            };
            let schedule = Schedule::Gemv {
                width: 256,
                cols,
                unroll: 8,
            };
            let KernelChoice::Generated(KernelRequest::Packed(request)) =
                program.kernel_choice(eqn)
            else {
                panic!("{what}: a packed contraction is a generated request");
            };
            assert_eq!(
                request.spec,
                poot_kernelgen::PackedKernelSpec {
                    format: WeightFormat::Q8_0,
                    op: PackedKernelOp::Contraction {
                        rows: RowSelect::Dense,
                        schedule,
                    },
                },
                "{what}"
            );
            assert_eq!(
                *grid,
                [n.div_ceil(cols as usize) as u32 * 256, 1, 1],
                "{what}"
            );
            assert_eq!(body.workgroup_size, [256, 1, 1], "{what}");
        }
    }

    /// SC-001 (card 658, GPU-free): a block-diagonal `PackedContraction` (`blocks = 4`) whose
    /// `m_per_block = 5` is not a multiple of `GEMM_TILE` (8) - the exact shape
    /// `try_plan_contraction` used to refuse to `Schedule::Tiled` and fall back to the per-thread-
    /// unbounded `Schedule::Serial` for - compiles to `Schedule::Tiled`,
    /// never `Schedule::Serial`. Driven through `compile`, over
    /// `ops::packed_block_diagonal_linear` (Card 385's block-diagonal builder, the same chain
    /// `packed_block_diagonal_chain_is_recognized_as_one_contraction` proves folds to one
    /// `PackedContraction`). Mutation: restore the old `else if blocks == 1 { Tiled } else { Serial
    /// }` arm in `try_plan_contraction`; the plan key goes from `...:Tiled { tile: 8 }:4` to
    /// `...:Serial:4` and this test goes red.
    #[test]
    fn block_diagonal_ragged_m_per_block_compiles_to_tiled_not_serial() {
        let (blocks, m_per_block, out_per_block, k) = (4usize, 5usize, 2usize, 35usize);
        let out = blocks * out_per_block;
        let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [out, k]).unwrap();
        let builder = Builder::new();
        let x = builder.slot_named(
            Slot::Activation,
            "x",
            TensorType::f32(vec![blocks, m_per_block, k]),
        );
        let out_value = poot_graph_ir::ops::packed_block_diagonal_linear(
            &builder, x, "layer", descriptor, blocks, None,
        )
        .unwrap();
        let target = Target {
            backend: Backend::SpirvVulkan,
            caps: DeviceCaps::wgpu_rdna3_igpu(),
        };
        let options = CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::Full,
            limits: crate::CompileLimits::STANDARD,
        };
        let program = compile(&builder.finish(out_value), &target, &options).unwrap();
        let (eqn, plan) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::PackedContraction { .. }))
            .unwrap_or_else(|| panic!("no PackedContraction planned"));
        let Plan::ComputeMeta { grid, .. } = plan else {
            panic!("expected a ComputeMeta plan, got {plan:?}");
        };
        let schedule = Schedule::Tiled {
            tile: poot_kernelgen::TileSize::new(super::GEMM_TILE as u32).unwrap(),
        };
        let KernelChoice::Generated(KernelRequest::Packed(request)) = program.kernel_choice(eqn)
        else {
            panic!("a packed contraction is a generated request");
        };
        assert_eq!(
            request.spec,
            poot_kernelgen::PackedKernelSpec {
                format: WeightFormat::E2m1Row32,
                op: PackedKernelOp::Contraction {
                    rows: RowSelect::Dense,
                    schedule,
                },
            }
        );
        assert_eq!(
            *grid,
            [
                schedule.grid_threads(blocks, m_per_block, out_per_block) as u32,
                1,
                1
            ]
        );
    }

    /// Builds a `[blocks, m_per_block, k]` block-diagonal activation against a `[blocks*out_per_block,
    /// k]` descriptor, folded to one `PackedContraction` eqn via the real recognizer
    /// (`transform::recognize_packed_contractions`, the same pass `compile` runs) - not through
    /// `compile` itself, since `compile`'s pipeline reverts a rewrite the target cannot plan (treating
    /// an over-cap `PackedContraction` like any other unplannable rewrite)
    /// and re-reports it as an unrelated "escaped dequant" error, burying the refusal this test wants
    /// to observe. [`crate::plan_eqn_analyzed`], the still-production entry point every wgpu executor
    /// call site and test now uses, drives `try_plan_contraction` directly off the already-recognized
    /// eqn instead, the real entry point one level below the rewrite pipeline.
    /// The index (not a clone) of the one `PackedContraction` eqn: `plan_eqn_analyzed`'s `ensure_eqn`
    /// checks the equation's address against the analyzed graph's own `eqns` storage (card 671 - a
    /// cloned `Eqn` detached from that storage is correctly rejected, which a clone here used to
    /// paper over when these tests called the deleted, identity-unchecked `plan_eqn`).
    fn over_cap_tiled_eqn(
        blocks: usize,
        m_per_block: usize,
        out_per_block: usize,
    ) -> (poot_graph_ir::Graph, usize) {
        let k = 32;
        let out = blocks * out_per_block;
        let descriptor = PackedWeight::try_new(WeightFormat::Q4_0, [out, k]).unwrap();
        let builder = Builder::new();
        let x = builder.slot_named(
            Slot::Activation,
            "x",
            TensorType::f32(vec![blocks, m_per_block, k]),
        );
        let out_value = poot_graph_ir::ops::packed_block_diagonal_linear(
            &builder, x, "layer", descriptor, blocks, None,
        )
        .unwrap();
        let graph = crate::recognize_packed_contractions(&builder.finish(out_value));
        let index = graph
            .eqns
            .iter()
            .position(|eqn| matches!(eqn.op, OpKind::PackedContraction { .. }))
            .unwrap_or_else(|| panic!("not recognized as one PackedContraction"));
        (graph, index)
    }

    /// Fixture `DeviceCaps` whose `max_grid` is tiny (`[4, 4, 4]`, never the real wgpu `65_535` - the
    /// mechanism this exercises is cap-size-agnostic, so a small cap keeps the fixture tiny rather
    /// than needing a multi-hundred-thousand-row activation to cross the real cap).
    fn tiny_grid_caps() -> DeviceCaps {
        DeviceCaps {
            max_grid: [4, 4, 4],
            ..DeviceCaps::wgpu_rdna3_igpu()
        }
    }

    /// SC-001 (card 658 review F1/F2, GPU-free): `tiles = ceil(40/8) * ceil(8/8) = 5` exceeds a fixture
    /// `max_grid[0] = 4`, so `try_plan_contraction` folds onto `(x_groups, y_groups) = (4, 2)` rather
    /// than refusing or truncating - `y_groups <= max_grid[1] = 4`, so the fold fits. The plan's `grid`
    /// carries the folded dims and `body.workgroup_size` stays the Tiled body's baked `[64, 1, 1]`:
    /// `finalize`'s generic wg-bump never corrupts the 64-lane LDS tile for this arm
    /// (`kernel_mapping::shape_launch` exempts every packed contraction request by type).
    ///
    /// Mutation: in `try_plan_contraction`, drop the `tiles_u32 <= limit_x` fold and always use
    /// `(tiles_u32, 1)` - `grid` goes from `[256, 2, 1]` to `[320, 1, 1]` and this test goes red.
    #[test]
    fn over_cap_tiled_contraction_folds_onto_a_2d_grid_not_the_wg_bump() {
        let (graph, index) = over_cap_tiled_eqn(1, 40, 8);
        let eqn = &graph.eqns[index];
        let plan = crate::plan_eqn_analyzed(
            &crate::ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &tiny_grid_caps(),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .unwrap();
        let Plan::ComputeMeta {
            body, grid, meta, ..
        } = plan
        else {
            panic!("expected a ComputeMeta plan, got {plan:?}");
        };
        assert_eq!(grid, [4 * 64, 2, 1], "folded grid: x_groups=4, y_groups=2");
        assert_eq!(
            body.workgroup_size,
            [64, 1, 1],
            "the wg-bump must not touch this arm's baked 64-lane workgroup"
        );
        assert_eq!(
            meta.get(3),
            Some(&4),
            "metadata[3] carries the folded x_groups"
        );
    }

    /// SC-001 (card 658 review F1/F2, GPU-free): `tiles = ceil(200/8) * ceil(8/8) = 25` exceeds the
    /// fixture `max_grid[0] = 4`, and even folded (`y_groups = ceil(25/4) = 7`) exceeds `max_grid[1] =
    /// 4` too - no 2-D grid can address every tile, so `try_plan_contraction` refuses with
    /// `Capability::DispatchGridLimit` naming the equation, before any dispatch (SC-001's named
    /// alternative to a bounded schedule).
    ///
    /// Mutation: in `try_plan_contraction`, drop the `y_groups > caps.max_grid[1]` check and always
    /// accept the fold - `plan_eqn` returns `Ok` instead of `Err(Refused(DispatchGridLimit))` and this
    /// test goes red.
    #[test]
    fn over_cap_tiled_contraction_refuses_when_even_the_fold_cannot_address_every_tile() {
        let (graph, index) = over_cap_tiled_eqn(1, 200, 8);
        let eqn = &graph.eqns[index];
        let error = crate::plan_eqn_analyzed(
            &crate::ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &tiny_grid_caps(),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .err()
        .unwrap_or_else(|| panic!("expected a refusal, plan_eqn succeeded"));
        let PlanError::Refused(refusal) = error else {
            panic!("expected PlanError::Refused, got {error:?}");
        };
        assert!(
            matches!(refusal.op, OpKind::PackedContraction { .. }),
            "refusal must name the PackedContraction equation, got {:?}",
            refusal.op
        );
        assert!(
            matches!(refusal.missing, Capability::DispatchGridLimit { .. }),
            "expected Capability::DispatchGridLimit, got {:?}",
            refusal.missing
        );
    }

    /// SC-001 (card 669, GPU-free): the refusal `try_plan_contraction` raises while probing a claimed
    /// over-cap block-diagonal contraction (`blocks = 2`, `m_per_block = 200`, `out_per_block = 8`,
    /// `K = 32`, Q4_0: `tiles = 50`, `y_groups = 13` above the fixture `max_grid = [4, 4, 4]`) reaches
    /// the `compile()` caller as the typed `Capability::DispatchGridLimit` refusal naming the
    /// contracted equation. `revert_unplannable_rewrites` puts the decomposition back when the claim
    /// refuses, so the escape gate used to report the restored graph's downstream symptom
    /// (`packed dequant v2 Q4_0 [16, 32] reaches unsupported movement reshape [2, 8, 32]`) instead -
    /// card 658, ADR-0104's pipeline-entry diagnostic.
    ///
    /// Mutation: drop the recorded refusal in the revert path (card 669); this row sees the movement
    /// escape error above and fails.
    #[test]
    fn over_cap_block_diagonal_compile_reports_the_grid_refusal_not_the_escape_symptom() {
        let (blocks, m_per_block, out_per_block, k) = (2usize, 200usize, 8usize, 32usize);
        let descriptor =
            PackedWeight::try_new(WeightFormat::Q4_0, [blocks * out_per_block, k]).unwrap();
        let builder = Builder::new();
        let x = builder.slot_named(
            Slot::Activation,
            "x",
            TensorType::f32(vec![blocks, m_per_block, k]),
        );
        let out_value = poot_graph_ir::ops::packed_block_diagonal_linear(
            &builder, x, "layer", descriptor, blocks, None,
        )
        .unwrap();
        let out_id = out_value.id;
        let graph = builder.finish(out_value);
        let target = Target {
            backend: Backend::SpirvVulkan,
            caps: tiny_grid_caps(),
        };
        let options = CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::Full,
            limits: crate::CompileLimits::STANDARD,
        };
        let Err(error) = compile(&graph, &target, &options) else {
            panic!("the over-cap contraction must not compile");
        };
        match error {
            CompileError::Plan(plan_error) => match *plan_error {
                PlanError::Refused(refusal) => {
                    assert_eq!(
                        refusal.eqn, out_id,
                        "the refusal must name the contracted equation"
                    );
                    assert!(
                        matches!(refusal.op, OpKind::PackedContraction { .. }),
                        "refusal must name the PackedContraction, got {:?}",
                        refusal.op
                    );
                    assert!(
                        matches!(refusal.missing, Capability::DispatchGridLimit { .. }),
                        "expected Capability::DispatchGridLimit, got {:?}",
                        refusal.missing
                    );
                }
                other => panic!("expected PlanError::Refused, got {other:?}"),
            },
            other => panic!(
                "expected the typed planner refusal behind the reverted claim, got {other:?}"
            ),
        }
    }

    /// SC-001 (card 658, GPU-free): the real `wgpu_rdna3_igpu` grid cap, not a
    /// fixture - `blocks = 4, m_per_block = 1030, out_per_block = 1024` is `out_numel = 4,218,880 >
    /// 4,194,240`, the exact `out_numel.div_ceil(64) > 65_535` threshold where `finalize`'s generic
    /// wg-bump would rewrite a one-thread-per-output body's workgroup from 64 to 256 - round 1's own
    /// reproduction showed a bumped Tiled body computes wrong values at this scale (`element 0: got
    /// 120.47945 want 570.88153`). `tiles = 4 * ceil(1030/8) * ceil(1024/8) = 66,048` exceeds
    /// `max_grid[0] = 65_535`, folding to `(x_groups, y_groups) = (65_535, 2)`. This is the one test
    /// that actually exercises `kernel_mapping::shape_launch`'s contraction exemption at the scale the
    /// hazard fires at - the fold-success/refusal rows above use a `max_grid: [4, 4, 4]` fixture four
    /// orders of magnitude below it, so disabling the exemption left them green.
    ///
    /// Mutation: in `kernel_mapping::shape_launch` set `contraction` to `false` for every request
    /// (dropping the exemption from `custom_grid`). Red: `body.workgroup_size` comes back `[256, 1, 1]`,
    /// not `[64, 1, 1]` - the generic bump fired on the 64-lane-baked Tiled body. Restored; green.
    #[test]
    fn over_cap_tiled_contraction_at_real_wgpu_caps_keeps_the_baked_workgroup() {
        let (blocks, m_per_block, out_per_block) = (4usize, 1030usize, 1024usize);
        let out_numel = blocks * m_per_block * out_per_block;
        assert!(
            out_numel > 4_194_240,
            "fixture must be above the wg-bump threshold: {out_numel}"
        );
        let (graph, index) = over_cap_tiled_eqn(blocks, m_per_block, out_per_block);
        let eqn = &graph.eqns[index];
        let plan = crate::plan_eqn_analyzed(
            &crate::ExactI32StorageAnalysis::new(&graph),
            &graph,
            eqn,
            Backend::SpirvVulkan,
            &DeviceCaps::wgpu_rdna3_igpu(),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .unwrap_or_else(|error| panic!("expected a bounded plan, got a refusal: {error}"));
        let Plan::ComputeMeta {
            body, grid, meta, ..
        } = plan
        else {
            panic!("expected a ComputeMeta plan, got {plan:?}");
        };
        assert_eq!(
            body.workgroup_size,
            [64, 1, 1],
            "finalize's wg-bump must not touch this arm's baked 64-lane workgroup at real scale"
        );
        assert_eq!(
            grid,
            [65_535 * 64, 2, 1],
            "folded grid at real wgpu caps: x_groups=65_535, y_groups=2"
        );
        assert_eq!(
            meta.get(3),
            Some(&65_535),
            "metadata[3] carries the folded x_groups"
        );
    }
}
