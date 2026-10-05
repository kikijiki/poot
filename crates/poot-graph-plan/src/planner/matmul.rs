//! Matmul and bias-epilogue matmul planning, including the choice of the synthesized tiled GEMM.

use super::*;
use poot_target::{Backend, DeviceCaps};

/// Plan one matmul-family equation (MatMul/MatMulBias). The contraction choice reads the compile's
/// `fusion` policy: [`FusionPolicy::MoeHangGuard`] keeps a matmul off the generated tiled GEMM (cards
/// 186/192: the tiled GEMM hung ROCm/AMDGCN on a routed-expert router `MatMul`).
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
    fusion: FusionPolicy,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let ids = value_ids(eqn);
    let fty = fty_ty;
    let generate = |request: KernelRequest| Planned::generated(site, request, backend, caps);
    let planned = match &eqn.op {
        OpKind::MatMul if fusion == FusionPolicy::Full && is_tileable_matmul_in(g, eqn) => {
            // card 099b / spec 060: the synthesizer generates the tiled-GEMM Body element-by-element
            // from the tile descriptor (`kg::tiled_region`) instead of reusing a hand-authored kernel.
            // Dims are baked as consts, so this is a plain `Plan::Compute` (no metadata buffer). f32 /
            // 2-D: `is_tileable_matmul_in` gates the shapes (M>1, rank-2 [K,N] weight) and the F32
            // operands this f32-only body reads. `ceil(M/2ts) * ceil(N/ts)` workgroups of `ts*ts` lanes.
            let a = shape(ids[0]);
            let r = out_shape.len();
            let (nn, kk) = (out_shape[r - 1], a[a.len() - 1]);
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
                caps,
            )?
        }
        OpKind::MatMul => {
            let (a, b) = (shape(ids[0]), shape(ids[1]));
            // spec 025: on NVPTX, a bf16 matmul whose M, N, K are all 16-aligned runs on tensor cores
            // (WMMA m16n16k16). Otherwise (SPIR-V, f32, or a non-aligned dim) the serial kernel is the
            // reference + fallback. The `tc:` cache tag keeps the tensor-core kernel from colliding with it.
            let r = out_shape.len();
            let (mm, nn, kk) = (out_shape[r - 2], out_shape[r - 1], a[a.len() - 1]);
            // spec 130 FR-004: on AmdGcn, a bf16 matmul with 16-aligned M/N/K and no batch dims runs on
            // AMD WMMA tensor cores (RDNA3: llvm.amdgcn.wmma.*). Gate on Wmma16x16x16Rdna3 only; CDNA
            // (gfx9xx / CdnaMfma) and unknown arch fall through to the serial imported kernel. Gate on
            // operand dtype, not output dtype: for mixed-precision matmul (bf16 operands, f32 output)
            // the output dtype is F32, so odt-gating would skip the tensor-core path. matmul_tensorcore
            // always reads bf16 inputs; `dt` only controls the output/epilogue.
            // spec 135: F16 deliberately does not join this operand-dtype gate. `matmul_tensorcore` (both
            // the AMD WMMA and NVPTX `tc` arms below) hard-codes BF16 tensor-core fragment loads; F16 and
            // BF16 are different 2-byte bit layouts, so routing F16 operands through it would reinterpret
            // f16 bits as bf16 (wrong numerics, not a crash). F16 tensor-core lowering is card 154
            // (blocked on the SPIR-V/Vulkan coopmat path). A whole-graph F16 matmul (odt = F16) falls
            // through the `operands_bf16 == false` gate to the serial `matmul_batched_dt` fallback at the
            // bottom of this arm, which is dtype-generic (`fty(odt) = Ty::F16`).
            // Card 235: factored into `dtype_widen::matmul_{amd_tc,nvptx_tc}_eligible` so the
            // widen-mismatched-dtype-operand transform reuses the same eligibility decision.
            let (a_dt, b_dt) = (g.aval(ids[0]).dtype, g.aval(ids[1]).dtype);
            let batch_dims_product = out_shape[..r - 2].iter().product::<usize>();
            let amd_tc = ids.len() >= 2
                && matmul_amd_tc_eligible(backend, a_dt, b_dt, mm, nn, kk, batch_dims_product);
            let tc = ids.len() >= 2 && matmul_nvptx_tc_eligible(backend, a_dt, b_dt, mm, nn, kk);
            // Card 154: RADV cooperative-matrix (SPIR-V/Vulkan tensor cores). F16 operands (RADV has no
            // VK_KHR_shader_bfloat16), 16-aligned M/N, K == 16 exactly (no K-loop, see
            // `matmul_spirv_coopmat_eligible`), no batch dims. F32 output only (config 13 has no
            // f16-accumulate variant).
            let coopmat = ids.len() >= 2
                && odt == DType::F32
                && matmul_spirv_coopmat_eligible(
                    backend,
                    a_dt,
                    b_dt,
                    mm,
                    nn,
                    kk,
                    batch_dims_product,
                );
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
            // Decode bf16 GEMV (dtype-driven, sibling of the tensor-core arms): F32 act x BF16 weight
            // at M==1 plans the packed-u32 in-kernel-widen body instead of being rejected or cast.
            let bf16_gemv = ids.len() >= 2
                && matmul_bf16_decode_gemv_eligible(
                    backend,
                    operand_dts.0,
                    operand_dts.1,
                    odt,
                    mm,
                    nn,
                    kk,
                    g.aval(ids[1]).shape.len(),
                    out_numel,
                    caps,
                );
            if (operand_dts.0 != odt || operand_dts.1 != odt)
                && !(amd_tc || tc || coopmat || bf16_gemv)
            {
                // Mixed operand and output dtypes lower only through the tensor-core kernels (bf16
                // operands, 16-aligned M/N/K, RDNA3 or NVPTX), the SPIR-V coopmat kernel (f16 operands,
                // 16-aligned M/N, K == 16) or the BF16 decode GEMV (F32xB16 -> F32 at M == 1 under the
                // chunk trigger).
                return Err(site.refuse(Capability::DtypeLowering));
            }
            let shapes = || MatmulShapes {
                out_shape: out_shape.to_vec(),
                a_shape: a.clone(),
                b_shape: b.clone(),
            };
            // card 181 Bug 2: the one-thread-per-output reference kernel folds past the wg-bump's own
            // ceiling onto the same 2-D grid Unary/Binary/scatter/DUS use (see `is_elementwise_2d`).
            let serial = |dt: Ty, bias: bool| {
                generate(KernelRequest::Contraction(ContractionSpec::Serial {
                    dt,
                    bias,
                    shapes: shapes(),
                    fold: elementwise_fold(g, backend, eqn, out_shape, out_numel, caps),
                }))
            };
            if is_decode_gemv_in(g, backend, eqn, out_shape, out_numel, caps) {
                // wgpu decode GEMV (M=1 shared-weight matmul, single-seq or batched): the coalesced
                // imported body - one workgroup owns GEMV_TILE output columns and reads
                // W[k, col0:col0+TILE] contiguously per k (card 035 raised occupancy with a strided
                // per-column workgroup; the stride-N read then dominated decode bandwidth at ~20 GB/s
                // vs the ~215 GB/s floor, so the tile-of-columns reshape). Same Body for any batch B
                // (B is via the grid / the batched twin). card 044: for single-sequence decode (B=1,
                // out_numel == N), dispatch the GEMV authored in Rust and imported by pootc
                // (`crates/pootc/tests/kernels/gemv_coalesced.rs`). It is shape-generic
                // (k = x.len(), N = out.len()) with the same [x, weight, out] buffers and 2-D grid
                // contract as `gemv_lds`, so no dims buffer or executor change is needed. Batched decode
                // (B>1) runs the imported batched kernel, with its `dims = [B]` bound as a ComputeMeta
                // metadata buffer (kernelgen bakes B as a const).
                // card 258: `decode_gemv_plan` also owns the large-N watchdog-safety chunking (SpirvVulkan
                // only); see its doc comment - that arm still generates a `GemvChunk` (one workgroup per
                // output element), the shape gate that keeps the old stride-N access pattern.
                decode_gemv_plan(site, backend, kk, nn, out_numel, false, bf16_gemv, caps)?
            } else if amd_tc {
                // The tensor-core body bakes one block (8 warps, 256 threads) per 16x16 output tile.
                // spec 130 FR-004: AMD WMMA tensor cores (RDNA3 / gfx11xx). For mixed-precision (bf16
                // operands -> f32 output), matmul_tensorcore with Ty::F32 uses only WmmaStore
                // (AMD-supported), not WmmaStoreLds (NVPTX-only). For pure-bf16 (bf16 operands -> bf16
                // output), matmul_tensorcore with Ty::BF16 needs WmmaStoreLds for the f32->bf16 LDS
                // narrowing, which is not wired for AMDGPU in emit.rs; fall back to the serial bf16 kernel.
                if odt == DType::F32 {
                    generate(KernelRequest::Contraction(ContractionSpec::TensorCore {
                        dt: Ty::F32,
                        shapes: shapes(),
                    }))?
                } else {
                    serial(fty(odt), false)?
                }
            } else if tc {
                // The NVPTX WMMA body bakes one 256-thread block per 16x16 output tile, per batch element
                // (`matmul_nvptx_tc_eligible` allows batch dims, unlike the AMD arm above):
                // `batch_count * tiles_m * tiles_n` blocks total (wmma.rs).
                generate(KernelRequest::Contraction(ContractionSpec::TensorCore {
                    dt: fty(odt),
                    shapes: shapes(),
                }))?
            } else if coopmat {
                // card 154 / R469-003: RADV coopmat bakes one subgroup (32 invocations) per 16x16
                // output tile, not the 256-thread WMMA block above.
                generate(KernelRequest::Contraction(ContractionSpec::Coopmat {
                    shapes: shapes(),
                }))?
            } else if is_tiled_gemm_in(g, backend, eqn, out_shape, out_numel) {
                // wgpu prefill tiled GEMM (M>1 shared-weight projection): LDS ts x ts tile reuse so each
                // weight element is read O(M/ts) times instead of O(M) (card 035). Chunked over sub-2^15
                // dispatches above the RADV miscompile threshold (card 096). f32 on SpirvVulkan.
                let mrows = out_numel / nn.max(1);
                tiled_gemm_plan(backend, mrows, kk, nn, false, caps)
            } else if is_batched_tiled_gemm_in(g, backend, eqn, out_shape, out_numel, caps) {
                // wgpu batched tiled GEMM: A[..,M,K] @ B[..,K,N] -> [..,M,N], one tiled GEMM per batch with
                // per-batch buffer offsets. batch = product of the leading dims: E for the MoE per-expert
                // GEMM (rank-3), or B*H for the prefill attention matmuls (rank-4).
                // card 044: runs the imported batched-weight coarsened tiled GEMM (per-batch bases). Drop-in
                // for kernelgen's `tiled_gemm_dt(b_count>1)`: same grid + buffers; the `[E,M,K,N]` it bakes
                // as consts ride in a metadata buffer (`Plan::ComputeMeta`).
                let r = out_shape.len();
                let ec: usize = out_shape[..r - 2].iter().product();
                let (mm, nn, kk) = (out_shape[r - 2], out_shape[r - 1], a[a.len() - 1]);
                let groups = mm.div_ceil(2 * GEMM_TILE) * nn.div_ceil(GEMM_TILE);
                Planned::imported(
                    ImportedKernel::TiledGemmBatched,
                    backend,
                    Some(vec![ec as u32, mm as u32, kk as u32, nn as u32]),
                    [(ec * groups * GEMM_TILE * GEMM_TILE) as u32, 1, 1],
                )
            } else if is_decode_attn_gemv_in(g, eqn, out_shape, out_numel, caps) {
                // decode attention scores@V: LDS-parallel reduction over cap (card 143 Lever 1b), the
                // batched-per-row sibling of is_decode_gemv's shared-weight decode GEMV.
                let r = out_shape.len();
                let (nn, kk) = (out_shape[r - 1], a[r - 1]);
                generate(KernelRequest::Contraction(
                    ContractionSpec::AttnScoresVGemv {
                        cap: kk,
                        d: nn,
                        width: GEMV_WIDTH,
                        numel: out_numel,
                    },
                ))?
            } else {
                // The naive one-thread-per-output fallback. A batched `Q @ K^T` at long ISL crosses
                // `is_batched_tiled_gemm`'s device-known-miscompile tile-count ceiling well before its
                // output numel crosses the 2-D fold's ceiling, so it lands here.
                serial(fty(odt), false)?
            }
        }
        OpKind::MatMulBias => {
            // fused bias epilogue: the kernel takes a third buffer (bias[N]) and adds it after the dot.
            let (a, b) = (shape(ids[0]), shape(ids[1]));
            if is_decode_gemv_in(g, backend, eqn, out_shape, out_numel, caps) {
                // wgpu decode GEMV with the bias epilogue (the q/k/v projections, M=1); on the naive kernel
                // these were the biggest decode cost (k/v proj N=128 -> 128 threads). card 035.
                let nn = out_shape[out_shape.len() - 1];
                let kk = a[a.len() - 1];
                // card 044: single-sequence (B=1) bias GEMV runs the imported-from-Rust kernel (bias is the
                // eqn's 3rd input buffer; see `ImportedKernel::GemvCoalesced` for the no-bias case).
                // card 258: `decode_gemv_plan` also owns the large-N watchdog-safety chunking (SpirvVulkan
                // only); see its doc comment.
                let weight_bf16 = ids.len() >= 2
                    && matmul_bf16_decode_gemv_eligible(
                        backend,
                        g.aval(ids[0]).dtype,
                        g.aval(ids[1]).dtype,
                        g.aval(eqn.out).dtype,
                        out_shape[out_shape.len() - 2],
                        nn,
                        kk,
                        g.aval(ids[1]).shape.len(),
                        out_numel,
                        caps,
                    );
                decode_gemv_plan(site, backend, kk, nn, out_numel, true, weight_bf16, caps)?
            } else if is_tiled_gemm_in(g, backend, eqn, out_shape, out_numel) {
                // wgpu prefill tiled GEMM with the bias epilogue (the q/k/v projections at M>1; chunked above
                // the 2^15 threshold, card 096). card 035.
                let (kk, nn) = (a[a.len() - 1], out_shape[out_shape.len() - 1]);
                let mrows = out_numel / nn.max(1);
                tiled_gemm_plan(backend, mrows, kk, nn, true, caps)
            } else {
                // the tensor-core path does not carry bias yet, so this uses the serial batched kernel.
                // card 181 Bug 2: same 2-D fold as the no-bias `MatMul` fallback above.
                generate(KernelRequest::Contraction(ContractionSpec::Serial {
                    dt: fty(odt),
                    bias: true,
                    shapes: MatmulShapes {
                        out_shape: out_shape.to_vec(),
                        a_shape: a,
                        b_shape: b,
                    },
                    fold: elementwise_fold(g, backend, eqn, out_shape, out_numel, caps),
                }))?
            }
        }
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | packed_ops!()
        | moe_ops!()
        | elementwise_ops!()
        | attention_ops!()
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
/// `ceil(m/2ts) * ceil(n/ts)` workgroups of `ts*ts` lanes, dims baked as consts (a plain `Plan::Compute`, no
/// metadata buffer), f32 / 2-D. Like `tiled_gemm_plan`, a GEMM at or above
/// `caps.known_miscompiles.tiled_gemm_max_workgroups` tiles is split into sub-cap dispatches (card 096), each a
/// baked tile-offset variant covering a disjoint tile range; a device with no confirmed ceiling (`None`) has no
/// such cap (single dispatch).
///
/// `gemm.weight` is the weight buffer's planned storage. The `[K, N]` matmul body reads an f32 weight only; the
/// checkpoint-order `[N, K]` body also reads a packed-F16 one (Card 1007), decoding it in-register.
pub(super) fn tiled_region_plan(
    site: Site<'_>,
    backend: Backend,
    gemm: TiledGemm,
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
            },
            WeightLayout::Nk => ContractionSpec::DenseTiled {
                m,
                k,
                n,
                tile,
                offset,
                groups,
                weight,
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
            Planned::generated_chunk(
                site,
                region(offset, (groups - offset).min(chunk)),
                backend,
                caps,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Planned::chunked(chunks))
}
