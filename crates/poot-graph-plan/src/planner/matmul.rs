//! Matmul and contraction-epilogue matmul planning, including the choice of the synthesized tiled GEMM.

use super::*;
use poot_graph_ir::ContractionRows;
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
    tile: usize,
    caps: &DeviceCaps,
) -> Result<FoldedTileGrid, PlanError> {
    let overflow = || site.refuse(Capability::DispatchSizeOverflow);
    let width = tile * tile;
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
/// ([`epilogue_head_lowers`]) share, so a head fuses exactly when the serial kernel is its plan anyway.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MatMulRoute {
    /// The generated tiled GEMM (`kg::tiled_region`) over an f32 `[K, N]` weight.
    Tileable,
    /// The decode GEMV (`M == 1`, rank-2 `[K, N]` f32 weight). A BF16 `[K, N]` weight lands here too and is
    /// refused by the mixed-dtype guard in `plan_eqn`: its only lowering is the dense Gemv over `[N, K]`.
    DecodeGemv,
    /// RDNA3 WMMA tensor cores (bf16 operands, 16-aligned dims).
    AmdTensorCore,
    /// NVPTX WMMA tensor cores.
    NvptxTensorCore,
    /// RADV cooperative matrix.
    Coopmat,
    /// The imported tiled GEMM (wgpu prefill).
    TiledGemm,
    /// The imported batched tiled GEMM.
    BatchedTiledGemm,
    /// The decode attention `scores @ V` GEMV.
    AttnGemv,
    /// The one-thread-per-output kernel every other `MatMul` lands on.
    Serial,
}

/// A `MatMul`'s route and whether its mixed operand/output dtypes have a lowering on it.
#[derive(Clone, Copy, Debug)]
pub(super) struct RoutedMatMul {
    pub(super) route: MatMulRoute,
    /// A mixed operand and output dtype pairing lowers only through the tensor-core kernels and the coopmat
    /// kernel.
    pub(super) lowers_mixed_dtypes: bool,
}

/// Choose `eqn`'s `MatMul` route.
#[allow(clippy::too_many_arguments)]
pub(super) fn matmul_route(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
) -> RoutedMatMul {
    let ids = value_ids(eqn);
    let tileable = RoutedMatMul {
        route: MatMulRoute::Tileable,
        lowers_mixed_dtypes: false,
    };
    if is_tileable_matmul_in(g, eqn) {
        return tileable;
    }
    let a = g.aval(ids[0]).shape.clone();
    let r = out_shape.len();
    let (mm, nn, kk) = (out_shape[r - 2], out_shape[r - 1], a[a.len() - 1]);
    // spec 025: on NVPTX, a bf16 matmul whose M, N, K are all 16-aligned runs on tensor cores
    // (WMMA m16n16k16). Otherwise (SPIR-V, f32, or a non-aligned dim) the serial kernel is the
    // reference + fallback. The `tc:` cache tag keeps the tensor-core kernel from colliding with it.
    // spec 130 FR-004: on AmdGcn, a bf16 matmul with 16-aligned M/N/K (any batch dims) runs on
    // AMD WMMA tensor cores (RDNA3: llvm.amdgcn.wmma.*). Gate on Wmma16x16x16Rdna3 only; CDNA
    // (gfx9xx / CdnaMfma) and unknown arch fall through to the serial imported kernel. Gate on
    // operand dtype, not output dtype: for mixed-precision matmul (bf16 operands, f32 output)
    // the output dtype is F32, so odt-gating would skip the tensor-core path. matmul_tensorcore
    // always reads bf16 inputs; `dt` only controls the output/epilogue.
    // spec 135: F16 deliberately does not join this operand-dtype gate. `matmul_tensorcore` (both
    // the AMD WMMA and NVPTX `tc` arms) hard-codes BF16 tensor-core fragment loads; F16 and
    // BF16 are different 2-byte bit layouts, so routing F16 operands through it would reinterpret
    // f16 bits as bf16 (wrong numerics, not a crash). F16 tensor-core lowering is card 154
    // (blocked on the SPIR-V/Vulkan coopmat path). A whole-graph F16 matmul (odt = F16) falls
    // through the `operands_bf16 == false` gate to the serial `matmul_batched_dt` fallback, which is
    // dtype-generic (`fty(odt) = Ty::F16`).
    // Card 235: factored into `dtype_widen::matmul_{amd_tc,nvptx_tc}_eligible` so the
    // widen-mismatched-dtype-operand transform reuses the same eligibility decision.
    let (a_dt, b_dt) = (g.aval(ids[0]).dtype, g.aval(ids[1]).dtype);
    let amd_tc = matmul_amd_tc_eligible(backend, a_dt, b_dt, mm, nn, kk);
    let tc = matmul_nvptx_tc_eligible(backend, a_dt, b_dt, mm, nn, kk);
    // Card 154: RADV cooperative-matrix (SPIR-V/Vulkan tensor cores). F16 operands (RADV has no
    // VK_KHR_shader_bfloat16), 16-aligned M/N/K, any batch dims (see `matmul_spirv_coopmat_eligible`). F32
    // output only (config 13 has no f16-accumulate variant).
    let coopmat =
        odt == DType::F32 && matmul_spirv_coopmat_eligible(backend, a_dt, b_dt, mm, nn, kk);
    let lowers_mixed_dtypes = amd_tc || tc || coopmat;
    let route = if is_decode_gemv_in(g, backend, eqn, out_shape, out_numel, caps) {
        MatMulRoute::DecodeGemv
    } else if amd_tc {
        MatMulRoute::AmdTensorCore
    } else if tc {
        MatMulRoute::NvptxTensorCore
    } else if coopmat {
        MatMulRoute::Coopmat
    } else if is_tiled_gemm_in(g, backend, eqn, out_shape, out_numel) {
        MatMulRoute::TiledGemm
    } else if is_batched_tiled_gemm_in(g, backend, eqn, out_shape, out_numel, caps) {
        MatMulRoute::BatchedTiledGemm
    } else if is_decode_attn_gemv_in(g, eqn, out_shape, out_numel, caps) {
        MatMulRoute::AttnGemv
    } else {
        MatMulRoute::Serial
    };
    RoutedMatMul {
        route,
        lowers_mixed_dtypes,
    }
}

/// Whether `eqn`, a contraction head, has an epilogue lowering that is also the plan it would get without one:
/// the epilogue fuse rule forms a [`OpKind::ContractionEpilogue`] only over such a head, so no head loses a faster schedule (a tensor-core,
/// tiled or decode-GEMV `MatMul`) to gain a fused store. The generated Gemv, serial and tiled bodies carry an
/// epilogue; a `MatMul` head's fragment, GEMV and imported schedules do not.
pub(crate) fn epilogue_head_lowers<V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &Eqn,
    backend: Backend,
    caps: &DeviceCaps,
) -> bool {
    let tables = GraphTables::from(g);
    let out_shape = &g.aval(eqn.out).shape;
    let out_numel = out_numel_saturating(out_shape);
    match &eqn.op {
        OpKind::MatMul => {
            out_shape.len() >= 2
                && matches!(
                    matmul_route(
                        tables,
                        eqn,
                        out_shape,
                        out_numel,
                        backend,
                        g.aval(eqn.out).dtype,
                        caps,
                    )
                    .route,
                    MatMulRoute::Tileable | MatMulRoute::Serial
                )
        }
        // A dense head takes the generated Gemv at `M == 1` and the generated tiled GEMM above it, and both carry
        // an epilogue, so every dense head lowers (the weight dtypes the dense planner admits).
        OpKind::DenseContraction { .. } => true,
        // Every indexed schedule (Gemv and segmented Tiled) carries an epilogue.
        OpKind::IndexedMatMul => indexed_matmul_lowers(tables, eqn),
        // The packed Gemv (one activation row per block) carries an epilogue; the block-diagonal tiled schedule does not.
        OpKind::PackedContraction { descriptor, rows } => {
            crate::packed_block_float::admit_packed(
                crate::packed_block_float::PACKED_LOWERING,
                *descriptor,
                crate::packed_block_float::PackedKernelOpKind::Contraction,
                backend,
            )
            .is_ok()
                && match rows {
                    ContractionRows::Blocks(blocks) => {
                        packed_contraction_m_per_block(tables, eqn, *descriptor, *blocks) == 1
                    }
                    // An indexed table carries an epilogue on both its schedules (Gemv and segmented Tiled).
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
/// (`packed_operands[position]`, the storage plan's choice); anything else is a typed refusal.
#[allow(clippy::too_many_arguments)]
pub(super) fn epilogue_of(
    site: Site<'_>,
    g: GraphTables<'_>,
    eqn: &Eqn,
    head_operands: usize,
    region: &FusedRegion,
    out_shape: &[usize],
    odt: DType,
    packed_operands: &[bool],
) -> Result<kg::Epilogue, PlanError> {
    let dt = match odt {
        DType::F32 | DType::BF16 | DType::F16 => fty_ty(odt),
        _ => return Err(site.refuse(Capability::DtypeLowering)),
    };
    let mut leaves = Vec::new();
    for (position, operand) in eqn.inputs.iter().enumerate().skip(head_operands) {
        let Operand::Value(value) = operand else {
            return Err(site.refuse(Capability::DtypeLowering));
        };
        let leaf_dtype = g.aval(*value).dtype;
        let storage = if leaf_dtype == odt {
            None
        } else if odt == DType::F32 && leaf_dtype == DType::BF16 {
            let packed = packed_operands.get(position).copied().unwrap_or(false);
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
    let local = |id: usize| match id {
        0 => kg::EpilogueInput::Head,
        id if id < n_inputs => kg::EpilogueInput::Leaf(id - 1),
        id => kg::EpilogueInput::Step(id - n_inputs),
    };
    let steps = region
        .steps
        .iter()
        .map(|step| {
            check_widening_step(site, step.op, &step.inputs, |id| {
                (1..n_inputs).contains(&id)
            })?;
            Ok(kg::EpilogueStep {
                op: map_fused_op(site, step.op)?,
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

/// Plan one matmul-family equation (`MatMul`, or a `ContractionEpilogue` over a `MatMul` head).
#[allow(clippy::too_many_arguments)]
pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
    limits: BodyLimits,
    packed_operands: &[bool],
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let ids = value_ids(eqn);
    let fty = fty_ty;
    let generate = |request: KernelRequest| Planned::generated(site, request, backend, caps);
    let planned = match &eqn.op {
        OpKind::MatMul => {
            let (a, b) = (shape(ids[0]), shape(ids[1]));
            let r = out_shape.len();
            let (nn, kk) = (out_shape[r - 1], a[a.len() - 1]);
            let routed = matmul_route(g, eqn, out_shape, out_numel, backend, odt, caps);
            let shapes = || MatmulShapes {
                out_shape: out_shape.to_vec(),
                a_shape: a.clone(),
                b_shape: b.clone(),
            };
            // card 181 Bug 2: the one-thread-per-output reference kernel folds past the wg-bump's own
            // ceiling onto the same 2-D grid Unary/Binary/scatter/DUS use (see `is_elementwise_2d`).
            let serial = || {
                generate(KernelRequest::Contraction(ContractionSpec::Serial {
                    dt: fty(odt),
                    epilogue: None,
                    shapes: shapes(),
                    fold: elementwise_fold(g, backend, eqn, out_shape, out_numel, caps),
                }))
            };
            // Every arm below except the tensor-core ones (`amd_tc`/`tc`, which read bf16 operands by
            // construction) plans a kernel whose single element type `fty(odt)` covers operands and
            // output. A mixed operand/output matmul (`to_mixed_bf16`: bf16 operands, f32 output) is only
            // correct when a tensor-core arm fires; on any other route (e.g. AmdGcn with a non-RDNA3 arch
            // such as CDNA or RDNA4, where `amd_tc` declines and the eqn falls to the serial
            // `matmul_batched_dt`) the planned f32 kernel would read the 2-byte operand buffers as f32
            // words: silent garbage plus OOB reads, with a cache key colliding with the actual f32
            // matmul of the same shapes. Reject by name instead; `to_mixed_bf16` has no
            // "RDNA3+ hardware" gate of its own to rely on.
            let operand_dts = (g.aval(ids[0]).dtype, g.aval(ids[1]).dtype);
            if routed.route != MatMulRoute::Tileable
                && (operand_dts.0 != odt || operand_dts.1 != odt)
                && !routed.lowers_mixed_dtypes
            {
                // Mixed operand and output dtypes lower only through the tensor-core kernels (bf16
                // operands, 16-aligned M/N/K, RDNA3 or NVPTX), the SPIR-V coopmat kernel (f16 operands,
                // 16-aligned M/N/K). A BF16 `[K, N]` decode projection has no lowering and lands here.
                return Err(site.refuse(Capability::DtypeLowering));
            }
            match routed.route {
                MatMulRoute::Tileable => {
                    // card 099b / spec 060: the synthesizer generates the tiled-GEMM Body
                    // element-by-element from the tile descriptor (`kg::tiled_region`) instead of reusing
                    // a hand-authored kernel. Dims are baked as consts, so this is a plain
                    // a single plain dispatch (no metadata buffer). f32 / 2-D: `is_tileable_matmul_in` gates the
                    // shapes (M>1, rank-2 [K,N] weight) and the F32 operands this f32-only body reads.
                    // `ceil(M/2ts) * ceil(N/ts)` workgroups of `ts*ts` lanes.
                    tiled_region_plan(
                        site,
                        backend,
                        TiledGemm {
                            m: out_numel / nn.max(1),
                            k: kk,
                            n: nn,
                            layout: WeightLayout::Kn,
                            weight: poot_target::BufferStorage::f32(),
                        },
                        None,
                        caps,
                    )?
                }
                MatMulRoute::DecodeGemv => {
                    // wgpu decode GEMV (M=1 shared-weight matmul, single-seq or batched): the coalesced
                    // imported body - one workgroup owns GEMV_TILE output columns and reads
                    // W[k, col0:col0+TILE] contiguously per k (card 035 raised occupancy with a strided
                    // per-column workgroup; the stride-N read then dominated decode bandwidth at ~20 GB/s
                    // vs the ~215 GB/s floor, so the tile-of-columns reshape). Same Body for any batch B
                    // (B is via the grid / the batched twin). card 044: for single-sequence decode (B=1,
                    // out_numel == N), dispatch the GEMV authored in Rust and imported by pootc
                    // (`crates/pootc/tests/kernels/gemv_coalesced.rs`). It is shape-generic
                    // (k = x.len(), N = out.len()) with the same [x, weight, out] buffers and 2-D grid
                    // contract as `gemv_lds`, so no dims buffer or executor change is needed. Batched
                    // decode (B>1) runs the imported batched kernel, with its `dims = [B]` bound as a
                    // launch metadata buffer (kernelgen bakes B as a const).
                    // card 258: `decode_gemv_plan` also owns the large-N watchdog-safety chunking
                    // (SpirvVulkan only); see its doc comment - that arm still generates a `GemvChunk`
                    // (one workgroup per output element), the shape gate that keeps the old stride-N
                    // access pattern.
                    decode_gemv_plan(site, backend, kk, nn, out_numel, caps)?
                }
                MatMulRoute::AmdTensorCore => {
                    // The tensor-core body bakes one block (8 warps, 256 threads) per 16x16 output tile.
                    // spec 130 FR-004: AMD WMMA tensor cores (RDNA3 / gfx11xx). For mixed-precision (bf16
                    // operands -> f32 output), matmul_tensorcore with Ty::F32 uses only WmmaStore
                    // (AMD-supported), not WmmaStoreLds (NVPTX-only). For pure-bf16 (bf16 operands -> bf16
                    // output), matmul_tensorcore with Ty::BF16 needs WmmaStoreLds for the f32->bf16 LDS
                    // narrowing, which is not wired for AMDGPU in emit.rs; fall back to the serial bf16
                    // kernel.
                    if odt == DType::F32 {
                        generate(KernelRequest::Contraction(ContractionSpec::TensorCore {
                            dt: Ty::F32,
                            shapes: shapes(),
                        }))?
                    } else {
                        serial()?
                    }
                }
                MatMulRoute::NvptxTensorCore => {
                    // The NVPTX WMMA body bakes one 256-thread block per 16x16 output tile, per batch
                    // element (`matmul_nvptx_tc_eligible` allows batch dims, unlike the AMD arm above):
                    // `batch_count * tiles_m * tiles_n` blocks total (wmma.rs).
                    generate(KernelRequest::Contraction(ContractionSpec::TensorCore {
                        dt: fty(odt),
                        shapes: shapes(),
                    }))?
                }
                MatMulRoute::Coopmat => {
                    // card 154 / R469-003: RADV coopmat bakes one subgroup (32 invocations) per 16x16
                    // output tile, not the 256-thread WMMA block above.
                    generate(KernelRequest::Contraction(ContractionSpec::Coopmat {
                        shapes: shapes(),
                    }))?
                }
                MatMulRoute::TiledGemm => {
                    // wgpu prefill tiled GEMM (M>1 shared-weight projection): LDS ts x ts tile reuse so
                    // each weight element is read O(M/ts) times instead of O(M) (card 035). Split over
                    // sub-2^15 dispatches above the RADV miscompile threshold (card 096). f32 on
                    // SpirvVulkan.
                    tiled_gemm_plan(backend, out_numel / nn.max(1), kk, nn, caps)?
                }
                MatMulRoute::BatchedTiledGemm => {
                    // wgpu batched tiled GEMM: A[..,M,K] @ B[..,K,N] -> [..,M,N], one tiled GEMM per
                    // batch with per-batch buffer offsets. batch = product of the leading dims: E for the
                    // MoE per-expert GEMM (rank-3), or B*H for the prefill attention matmuls (rank-4).
                    // card 044: runs the imported batched-weight coarsened tiled GEMM (per-batch bases).
                    // Drop-in for kernelgen's `tiled_gemm_dt(b_count>1)`: same grid + buffers; the
                    // `[E,M,K,N]` it bakes as consts ride in a metadata buffer (launch metadata).
                    let ec: usize = out_shape[..r - 2].iter().product();
                    let mm = out_shape[r - 2];
                    let groups = mm.div_ceil(2 * GEMM_TILE) * nn.div_ceil(GEMM_TILE);
                    Planned::imported(
                        ImportedKernel::TiledGemmBatched,
                        backend,
                        Some(vec![ec as u32, mm as u32, kk as u32, nn as u32]),
                        [(ec * groups * GEMM_TILE * GEMM_TILE) as u32, 1, 1],
                    )?
                }
                MatMulRoute::AttnGemv => {
                    // decode attention scores@V: LDS-parallel reduction over cap (card 143 Lever 1b), the
                    // batched-per-row sibling of is_decode_gemv's shared-weight decode GEMV.
                    generate(KernelRequest::Contraction(
                        ContractionSpec::AttnScoresVGemv {
                            cap: kk,
                            d: nn,
                            width: GEMV_WIDTH,
                            numel: out_numel,
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
            head: ContractionHead::MatMul,
            region,
        } => {
            // The head's operands and result share one dtype (the fuse rule's contract), and the generated
            // serial kernel is the lowering: it rounds the f32 accumulator through that dtype before the
            // epilogue reads it, as the unfused program's store does.
            if ids[..ContractionHead::MatMul.operands()]
                .iter()
                .any(|&id| g.aval(id).dtype != odt)
            {
                return Err(site.refuse(Capability::DtypeLowering));
            }
            let epilogue = epilogue_of(
                site,
                g,
                eqn,
                ContractionHead::MatMul.operands(),
                region,
                out_shape,
                odt,
                packed_operands,
            )?;
            if is_tileable_matmul_in(g, eqn) {
                // The route `matmul_route` gives the unfused head: the generated tiled GEMM, storing the
                // epilogue's result per output element.
                let (a, nn) = (shape(ids[0]), out_shape[out_shape.len() - 1]);
                return tiled_region_plan(
                    site,
                    backend,
                    TiledGemm {
                        m: out_numel / nn.max(1),
                        k: a[a.len() - 1],
                        n: nn,
                        layout: WeightLayout::Kn,
                        weight: poot_target::BufferStorage::f32(),
                    },
                    Some(epilogue),
                    caps,
                );
            }
            generate(KernelRequest::Contraction(ContractionSpec::Serial {
                dt: fty(odt),
                epilogue: Some(epilogue),
                shapes: MatmulShapes {
                    out_shape: out_shape.to_vec(),
                    a_shape: shape(ids[0]),
                    b_shape: shape(ids[1]),
                },
                fold: elementwise_fold(g, backend, eqn, out_shape, out_numel, caps),
            }))?
        }
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | packed_ops!()
        | moe_ops!()
        | elementwise_ops!()
        | attention_ops!()
        | scan_ops!()
        | cast_ops!()
        | movement_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only matmul ops here"),
    };
    Ok(planned)
}

/// One generated tiled GEMM: `m` rows of an `[m, k]` activation against an `n`-column weight read in `layout`
/// from `weight` storage.
#[derive(Clone, Copy)]
pub(super) struct TiledGemm {
    pub(super) m: usize,
    pub(super) k: usize,
    pub(super) n: usize,
    pub(super) layout: WeightLayout,
    pub(super) weight: poot_target::BufferStorage,
}

/// Plan the generated tiled GEMM over `m` rows of an `[m, k]` activation and a weight in `layout`:
/// `ceil(m/2ts) * ceil(n/ts)` workgroups of `ts*ts` lanes, dims baked as consts (a single plain dispatch, no
/// metadata buffer), f32 / 2-D. Like `tiled_gemm_plan`, a GEMM at or above
/// `caps.known_miscompiles.tiled_gemm_max_workgroups` tiles is split into sub-cap dispatches (card 096), each a
/// baked tile-offset variant covering a disjoint tile range; a device with no confirmed ceiling (`None`) has no
/// such cap (single dispatch).
///
/// `gemm.weight` is the weight buffer's planned storage. The `[K, N]` matmul body reads an f32 weight only; the
/// checkpoint-order `[N, K]` body also reads a packed-F16 one (Card 1007), decoding it in-register.
///
/// `epilogue` (f32) is applied to each output element's sum before its store, in every chunk.
pub(super) fn tiled_region_plan(
    site: Site<'_>,
    backend: Backend,
    gemm: TiledGemm,
    epilogue: Option<kg::Epilogue>,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    let TiledGemm {
        m,
        k,
        n,
        layout,
        weight,
    } = gemm;
    if layout == WeightLayout::Kn && weight != poot_target::BufferStorage::f32() {
        return Err(site.refuse(Capability::DtypeLowering));
    }
    let groups = m.div_ceil(2 * GEMM_TILE) * n.div_ceil(GEMM_TILE);
    let over_cap = caps
        .known_miscompiles
        .tiled_gemm_max_workgroups
        .is_some_and(|max| groups >= max as usize);
    let region = |offset: usize, groups: usize| {
        let tile = GEMM_TILE;
        KernelRequest::Contraction(match layout {
            WeightLayout::Kn => ContractionSpec::TiledRegion {
                m,
                k,
                n,
                tile,
                offset,
                groups,
                epilogue: epilogue.clone(),
            },
            WeightLayout::Nk => ContractionSpec::DenseTiled {
                m,
                k,
                n,
                tile,
                offset,
                groups,
                weight,
                epilogue: epilogue.clone(),
            },
        })
    };
    if !over_cap {
        return Planned::generated(site, region(0, groups), backend, caps);
    }
    // `over_cap` is only true when a cap exists, so this is always `Some`.
    let chunk = caps.known_miscompiles.tiled_gemm_max_workgroups.unwrap() as usize - 1;
    let n_chunks = groups.div_ceil(chunk);
    let chunks = (0..n_chunks)
        .map(|ci| {
            let offset = ci * chunk;
            Planned::generated_dispatch(
                site,
                region(offset, (groups - offset).min(chunk)),
                backend,
                caps,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Planned::split(chunks)
}
