//! Packed block-float / dense reader equation planning: the dense contraction (BF16 and F32 weights), the
//! packed BF16 and E4M3FN row gathers, and the i8 pack/unpack pair.

use super::*;
use poot_target::{Backend, DeviceCaps};

/// Plan one packed-reader equation (DenseContraction/DenseRowGather/PackI8/UnpackI8).
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
    let mut planned = match &eqn.op {
        // Card 380: the mixed F32-activation x BF16-weight contraction, weight in checkpoint `[N, K]`
        // order. `fold_dense_contractions` produced this equation, so the `Transpose` the LM head
        // would otherwise materialize is already gone. The imported kernel reads the weight as packed u32
        // lanes and widens in-register: SPIR-V rejects `Ty::BF16` for scalars and buffers alike, so this
        // is the only portable BF16 read on wgpu, and it keeps the device buffer at the checkpoint's two
        // bytes per element rather than the four a widening `Cast` would need.
        //
        // One thread per output element (the generic wg-bump below still applies). `rows` folds every
        // leading output dim into M, matching the kernel's
        // `output_flat / output_width` row split. At M == 1 the generated dense Gemv takes it instead
        // (Card 727, the branch below), reading the same packed lanes.
        //
        // AmdGcn is the ROCm typed packed walk's lane (card 450 ROCm): it uploads the same packed u32
        // lanes, so the very body runs there, with its receipt in the DeepSeek-V4 decode CPU-oracle
        // parity row.
        //
        // Nvptx is the PTX typed packed walk's lane (card 450 PTX): `upload_typed_input` there uploads
        // the same packed u32 lanes, so the identical body and key run on NVPTX, with its receipt in
        // the DeepSeek-V4 decode CPU-oracle parity row on the PTX device.
        OpKind::DenseContraction {
            weight: DType::BF16,
        } => {
            let r = out_shape.len();
            let n = out_shape[r - 1];
            let k = shape(ids[1])[1];
            let rows = out_numel / n.max(1);
            if let Some(schedule) = dense_decode_gemv_in(eqn, out_shape, out_numel, caps) {
                // M == 1: the generated dense Gemv over the packed lanes, whose lanes read contiguous
                // runs of each `[N, K]` weight row (the one-thread-per-element imported body below reads
                // it at a stride of `K * 2` bytes per lane, ~2 GB/s on the lm_head).
                dense_gemv(
                    site,
                    backend,
                    ValueStorage::new(StorageKind::Bf16Packed).buffer_storage(),
                    schedule,
                    k,
                    n,
                    caps,
                )?
            } else {
                Planned::imported(
                    ImportedKernel::DenseBf16Contraction,
                    backend,
                    Some(vec![rows as u32, k as u32, n as u32]),
                    [out_numel as u32, 1, 1],
                )
            }
        }
        // Card 645: the same contraction over an F32 weight in checkpoint `[N, K]` order, so a family written
        // against the checkpoint's orientation never runs a device `transpose`. The weight is an ordinary f32
        // buffer, so generated bodies read it directly, by shape:
        //
        // - M == 1: the generated dense Gemv (Card 727), the BF16 arm's body over this storage. A column's
        //   lanes read contiguous runs of its weight row (the `[N, K]` layout is already the coalesced one).
        // - M > 1: the generated tiled GEMM reading B through `[N, K]` strides, unless the compile's fusion
        //   policy keeps a matmul off it (the MoE router hang, cards 186/192); then the one-thread-per-output
        //   kernel.
        //
        // Every body is generated from a request that records the weight layout, so no key is shared with the
        // `[K, N]` matmul bodies. No tensor-core `[N, K]` body exists here (Card 727).
        //
        // Card 1007: an F16 weight takes the same bodies. Its buffer is packed two binary16 elements per `u32`
        // word (`StorageKind::F16Packed` on every backend), and each request names that storage, so the body
        // decodes each half in-register: no widened f32 copy of the weight and no native-F16 body. The request's
        // storage and the value's planned storage are one decision, `dense_contraction_weight_storage`.
        OpKind::DenseContraction {
            weight: dtype @ (DType::F32 | DType::F16),
        } => {
            let Some(weight) = dense_contraction_weight_storage(*dtype) else {
                return Err(site.refuse(Capability::DtypeLowering));
            };
            let weight = ValueStorage::new(weight).buffer_storage();
            let r = out_shape.len();
            let n = out_shape[r - 1];
            let k = shape(ids[1])[1];
            if let Some(schedule) = dense_decode_gemv_in(eqn, out_shape, out_numel, caps) {
                dense_gemv(site, backend, weight, schedule, k, n, caps)?
            } else if fusion == FusionPolicy::Full {
                matmul::tiled_region_plan(
                    site,
                    backend,
                    matmul::TiledGemm {
                        m: out_numel / n.max(1),
                        k,
                        n,
                        layout: WeightLayout::Nk,
                        weight,
                    },
                    caps,
                )?
            } else {
                Planned::generated(
                    site,
                    KernelRequest::Contraction(ContractionSpec::DenseSerial {
                        weight,
                        shapes: MatmulShapes {
                            out_shape: out_shape.to_vec(),
                            a_shape: shape(ids[0]),
                            b_shape: shape(ids[1]),
                        },
                        fold: elementwise_fold(g, backend, eqn, out_shape, out_numel, caps),
                    }),
                    backend,
                    caps,
                )?
            }
        }
        // PTX has actual two-byte BF16 storage, but its typed packed walk reads the checkpoint bytes as
        // packed u32 lanes like wgpu's and ROCm's, so it lowers through the same imported body above
        // rather than a second native-BF16 kernel. This arm stays as the named fallback for a weight
        // dtype no backend lowers; a `DenseContraction` reaching it is a bug, not a silent fallback to
        // a kernel that would read the weight at the wrong element width.
        OpKind::DenseContraction { .. } => {
            return Err(site.refuse(Capability::DtypeLowering));
        }
        // Card 381: the embedding gather over a packed-BF16 table, widened to F32 in the same kernel.
        // `fold_dense_bf16_row_gathers` produced this equation, so the BF16 intermediate between the
        // gather and the widening `Cast` is gone; a kernel that produces a BF16 value on wgpu would have
        // to write two elements per u32 lane.
        //
        // One thread per output element and no metadata buffer: the row width is derived in-kernel as
        // `out.len() / index.len()`, like the f32 `gather0:imported` arm above: the generic
        // one-thread-per-output grid (the wg-bump below still applies).
        //
        // AmdGcn is the ROCm typed packed walk's lane (card 450 ROCm): it uploads the same packed u32
        // lanes through `poot_rocm_gpu::upload_typed_input`, so the very body runs there. Its receipt
        // is the DeepSeek-V4 decode CPU-oracle parity row, which walks this kernel on the iGPU.
        //
        // Nvptx is the PTX typed packed walk's lane (card 450 PTX): `poot_ptx_gpu::upload_typed_input`
        // uploads the same packed u32 lanes, so the identical body and key run on NVPTX, with its
        // receipt in the DeepSeek-V4 decode CPU-oracle parity row on the PTX device.
        OpKind::DenseRowGather {
            source: DType::BF16,
        } => Planned::imported(
            ImportedKernel::PackedBf16RowGather,
            backend,
            None,
            [out_numel as u32, 1, 1],
        ),
        // Card 449 D1: the same row gather over a packed-E4M3FN table (Qwen4Exp's sharded PLE
        // embedding). Same u32 transport, same row selection, same in-register widening, and the
        // same 3-param `Plan::Compute` with no metadata buffer; only the byte-to-f32 decode differs
        // (E4M3FN instead of BF16). The index is the exact-I32 one `fold_dense_bf16_row_gathers`
        // pinned, which is what keeps n-gram ids above `2^24` from rounding through f32.
        //
        // Scoped to SpirvVulkan by name: wgpu is the only typed packed walk with a
        // Qwen4Exp route today, so no ROCm or PTX receipt exists for this body yet. Nvptx and
        // AmdGcn keep the named refusal below rather than claiming an untested kernel.
        OpKind::DenseRowGather {
            source: DType::E4M3FN,
        } if backend == Backend::SpirvVulkan => Planned::imported(
            ImportedKernel::PackedE4m3RowGather,
            backend,
            None,
            [out_numel as u32, 1, 1],
        ),
        // Scoped to the backends that have a kernel, by name, like the contraction above.
        // The BF16 row lowers on all three backends and the E4M3FN row on SpirvVulkan; an admitted
        // source dtype with no packed-lane kernel on this backend still refuses by name rather than
        // falling back to a gather kernel that would read the table at the wrong element width.
        OpKind::DenseRowGather { .. } => {
            return Err(site.refuse(Capability::DtypeLowering));
        }
        OpKind::PackI8 => {
            // f32 codes [.., L] -> i32 words [.., ceil(L/4)] (spec 048). Fixed dtypes (input f32,
            // output i32). card 044: the imported kernel runs on both backends; `L` is not recoverable
            // from `W = ceil(L/4)`, so it rides in a `[L]` ComputeMeta buffer (`W` derived in-kernel).
            let a = shape(ids[0]);
            let Some(&l) = a.last() else {
                return Err(PlanError::BadShape(
                    "PackI8 input needs at least 1 dimension, got a 0-D shape".into(),
                ));
            };
            Planned::imported(
                ImportedKernel::PackI8,
                backend,
                Some(vec![l as u32]),
                [out_numel as u32, 1, 1],
            )
        }
        OpKind::UnpackI8 { len } => {
            // i32 words [.., ceil(len/4)] -> f32 codes [.., len] (spec 048), fixed dtypes. card 044:
            // the imported inverse of pack_i8 on both backends; `len` rides in a `[len]` ComputeMeta
            // buffer (`W = ceil(len/4)` derived in-kernel).
            Planned::imported(
                ImportedKernel::UnpackI8,
                backend,
                Some(vec![*len as u32]),
                [out_numel as u32, 1, 1],
            )
        }
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | moe_ops!()
        | elementwise_ops!()
        | matmul_ops!()
        | attention_ops!()
        | cast_ops!()
        | movement_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only packed-reader ops here"),
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

/// The `DenseContraction` decode GEMV over an `[N, K]` weight held in `weight` storage: one generated body per
/// storage and launch, shape-generic (`K` and `N` ride in its metadata), so every weight dtype and backend takes
/// the same loop nest (Card 727).
fn dense_gemv(
    site: Site<'_>,
    backend: Backend,
    weight: poot_target::BufferStorage,
    schedule: kg::Schedule,
    k: usize,
    n: usize,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    Planned::generated(
        site,
        KernelRequest::Contraction(ContractionSpec::DenseGemv {
            weight,
            layout: WeightLayout::Nk,
            schedule,
            k,
            n,
        }),
        backend,
        caps,
    )
}
