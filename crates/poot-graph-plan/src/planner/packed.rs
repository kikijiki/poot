//! Packed block-float / dense reader equation planning: the dense contraction (BF16 and F32 weights), the
//! packed BF16 and E4M3FN row gathers, and the i8 pack/unpack pair.

use super::{
    BodyLimits, Capability, ContractionHead, ContractionSpec, DType, Eqn, ImportedKernel,
    KernelRequest, OpKind, PlanError, Planned, ValueId, WeightLayout, kg, matmul, value_ids,
};
use crate::dtype_widen::ConstLanes;
use crate::planner::{GraphTables, StorageDtypes, finalize};
use crate::predicates::dense_decode_gemv_in;
use crate::refusal::Site;
use crate::storage_analysis::dense_contraction_weight_storage;
use crate::types::{StorageKind, ValueStorage};
use poot_graph_ir::{FragmentDtype, OperandPrecision};
use poot_target::{Backend, DeviceCaps};

/// Whether this family lowers `op` over these storage dtypes. The E4M3FN table row gather reads the
/// payload as row-padded u32 words on every device (so it needs no native E4M3FN storage); no other dense
/// reader takes an E4M3FN operand, and none produces a 2-byte float (a dense contraction is F32 out) but a dense
/// contraction epilogue whose region narrows its f32 result (POOT-1448), whose lowering decides whether its body
/// stores one.
pub(super) fn admits_dtypes(op: &OpKind, dtypes: StorageDtypes) -> Result<(), Capability> {
    match op {
        OpKind::DenseRowGather {
            source: DType::E4M3FN,
        } => Ok(()),
        OpKind::ContractionEpilogue { region, .. }
            if region.narrowed_output() == Some(dtypes.out) && !dtypes.touches_e4m3fn() =>
        {
            Ok(())
        }
        _ => dtypes.refuse_unwired(),
    }
}

/// The dense contractions, the ops whose activation may be stored in its fragment dtype (POOT-1448).
fn is_dense_contraction(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::DenseContraction { .. }
            | OpKind::GroupedDenseContraction { .. }
            | OpKind::ContractionEpilogue {
                head: ContractionHead::Dense { .. },
                ..
            }
    )
}

/// Plan one packed-reader equation (DenseContraction/DenseRowGather).
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
    // How the equation reads its operands, and the tiles a split one stages for a weight in `weight`'s storage.
    let precision = eqn.op.operand_precision().unwrap_or_default();
    // The fragment dtype the activation is stored in, when the planner stored it narrow (POOT-1448): read as stored
    // by the tiled body, which needs native two-byte elements for it.
    let activation_stored = match ids.first() {
        Some(&activation) if is_dense_contraction(&eqn.op) => match g.aval(activation).dtype {
            DType::BF16 | DType::F16 if !caps.native_two_byte_floats => {
                return Err(site.refuse(Capability::DtypeLowering));
            }
            DType::BF16 => Some(FragmentDtype::Bf16),
            DType::F16 => Some(FragmentDtype::F16),
            _ => None,
        },
        _ => None,
    };
    let terms = |weight: poot_target::BufferStorage| {
        kg::StagedTerms::of(
            precision,
            activation_stored,
            kg::stored_fragment_dtype(weight),
        )
    };
    let mut planned = match &eqn.op {
        // Card 645: the contraction over a weight in checkpoint `[N, K]` order, so a family written against the
        // checkpoint's orientation never runs a device `transpose`. The generated bodies read the weight in its
        // planned storage, by shape:
        //
        // - M == 1: the generated dense Gemv (Card 727). A column's
        //   lanes read contiguous runs of its weight row (the `[N, K]` layout is already the coalesced one).
        // - M > 1: the register-tiled GEMM (POOT-1403), staging runs of the weight's rows.
        //
        // Every body is generated from a request that records the weight layout, so no key is shared with the
        // `[K, N]` matmul bodies.
        //
        // Card 1007: an F16 weight takes the same bodies. Its buffer is packed two binary16 elements per `u32`
        // word (`StorageKind::F16Packed` on every backend), and each request names that storage, so the body
        // decodes each half in-register: no widened f32 copy of the weight and no native-F16 body. The request's
        // storage and the value's planned storage are one decision, `dense_contraction_weight_storage`.
        //
        // Card 1024: a BF16 weight takes the same bodies. It is the checkpoint bytes packed two per `u32` word
        // (the only portable BF16 read on wgpu, where SPIR-V rejects `Ty::BF16`; the ROCm and PTX typed walks
        // upload the same lanes), widened in-register, so the device buffer stays at two bytes per element.
        // Its storage is the BF16 const rule ([`value_storage_kind`]), so a BF16 weight may have other readers.
        // POOT-1477: under the narrow policy the driver declares a weight only dense readers use F16
        // (`store_weights_f16`), so it takes the F16 arm above: the bind rounds it to f16 once.
        OpKind::DenseContraction {
            weight: dtype @ (DType::F32 | DType::F16 | DType::BF16),
            ..
        } => {
            let kind = match dtype {
                DType::BF16 => StorageKind::Bf16Packed,
                _ => match dense_contraction_weight_storage(*dtype) {
                    Some(kind) => kind,
                    None => return Err(site.refuse(Capability::DtypeLowering)),
                },
            };
            let weight = ValueStorage::new(kind).buffer_storage();
            let r = out_shape.len();
            let n = out_shape[r - 1];
            let k = shape(ids[1])[1];
            if let Some(launch) = dense_decode_gemv_in(eqn, out_shape, out_numel, caps) {
                if activation_stored.is_some() {
                    return Err(site.refuse(Capability::DtypeLowering));
                }
                let schedule = launch.schedule(backend, k);
                dense_gemv(site, backend, weight, schedule, k, n, None, precision, caps)?
            } else {
                matmul::dense_tiled_plan(
                    site,
                    backend,
                    kg::TiledShape {
                        blocks: 1,
                        m_per_block: out_numel / n.max(1),
                        out_per_block: n,
                        k,
                        layout: WeightLayout::Nk,
                        rows: poot_kernelgen::TiledRows::Dense,
                        terms: terms(weight),
                        weight_bits: kg::dense_value_bits(weight),
                        activation_bits: kg::activation_bits(activation_stored),
                    },
                    kg::DenseTiledWeight::Nk(weight),
                    None,
                    precision,
                    activation_stored,
                    caps,
                )?
            }
        }
        // A dense contraction followed by its pointwise epilogue (`fuse_contraction_epilogues`): the same
        // weight storage (BF16 included) and Gemv as above at `M == 1`, else the register-tiled GEMM, each applying
        // the epilogue to the head's f32 result before the store.
        OpKind::ContractionEpilogue {
            head: head @ ContractionHead::Dense { weight: dtype, .. },
            region,
        } => {
            let storage = match dtype {
                DType::BF16 => StorageKind::Bf16Packed,
                dtype => dense_contraction_weight_storage(*dtype)
                    .ok_or_else(|| site.refuse(Capability::DtypeLowering))?,
            };
            let weight = ValueStorage::new(storage).buffer_storage();
            let gemv = dense_decode_gemv_in(eqn, out_shape, out_numel, caps);
            let epilogue = matmul::epilogue_of(
                site,
                g,
                eqn,
                head.operands(),
                region,
                out_shape,
                odt,
                const_lanes,
                // The tiled body stores a narrowed output into a buffer of native two-byte elements.
                gemv.is_none() && caps.native_two_byte_floats,
            )?;
            let k = shape(ids[1])[1];
            let n = out_shape[out_shape.len() - 1];
            if let Some(launch) = gemv {
                if activation_stored.is_some() {
                    return Err(site.refuse(Capability::DtypeLowering));
                }
                let schedule = launch.schedule(backend, k);
                dense_gemv(
                    site,
                    backend,
                    weight,
                    schedule,
                    k,
                    n,
                    Some(epilogue),
                    precision,
                    caps,
                )?
            } else {
                matmul::dense_tiled_plan(
                    site,
                    backend,
                    kg::TiledShape {
                        blocks: 1,
                        m_per_block: out_numel / n.max(1),
                        out_per_block: n,
                        k,
                        layout: WeightLayout::Nk,
                        rows: poot_kernelgen::TiledRows::Dense,
                        terms: terms(weight),
                        weight_bits: kg::dense_value_bits(weight),
                        activation_bits: kg::activation_bits(activation_stored),
                    },
                    kg::DenseTiledWeight::Nk(weight),
                    Some(epilogue),
                    precision,
                    activation_stored,
                    caps,
                )?
            }
        }
        // The grouped contraction (`fold_dense_contractions`' `[0, 2, 1]` spelling): `groups` independent
        // `[M, K] x [N, K]^T` products over row blocks of one `[groups * N, K]` weight, in packed storage (BF16, or
        // the F16 `store_weights_f16` declares). That is the register-tiled GEMM's block-diagonal form: group `g`'s
        // rows read weight rows `g * N ..`, so it takes the same body as the plain dense contraction.
        OpKind::GroupedDenseContraction {
            weight: dtype @ (DType::BF16 | DType::F16),
            groups,
            ..
        } => {
            let k = shape(ids[1])[1];
            let kind = match dtype {
                DType::BF16 => StorageKind::Bf16Packed,
                _ => StorageKind::F16Packed,
            };
            let weight = ValueStorage::new(kind).buffer_storage();
            matmul::dense_tiled_plan(
                site,
                backend,
                kg::TiledShape {
                    blocks: *groups,
                    m_per_block: out_shape[1],
                    out_per_block: out_shape[2],
                    k,
                    layout: WeightLayout::Nk,
                    rows: poot_kernelgen::TiledRows::Dense,
                    terms: terms(weight),
                    weight_bits: kg::dense_value_bits(weight),
                    activation_bits: kg::activation_bits(activation_stored),
                },
                kg::DenseTiledWeight::Nk(weight),
                None,
                precision,
                activation_stored,
                caps,
            )?
        }
        // PTX has actual two-byte BF16 storage, but its typed packed walk reads the checkpoint bytes as
        // packed u32 lanes like wgpu's and ROCm's, so it lowers through the same generated bodies above
        // rather than a second native-BF16 kernel. This arm stays as the named fallback for a weight
        // dtype no backend lowers; a `DenseContraction` reaching it is a bug, not a silent fallback to
        // a kernel that would read the weight at the wrong element width.
        OpKind::DenseContraction { .. } | OpKind::GroupedDenseContraction { .. } => {
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
        // lanes through `poot_rocm_device::upload_typed_input`, so the very body runs there. Its receipt
        // is the DeepSeek-V4 decode CPU-oracle parity row, which walks this kernel on the iGPU.
        //
        // Nvptx is the PTX typed packed walk's lane (card 450 PTX): `poot_ptx_device::upload_typed_input`
        // uploads the same packed u32 lanes, so the identical body and key run on NVPTX, with its
        // receipt in the DeepSeek-V4 decode CPU-oracle parity row on the PTX device.
        OpKind::DenseRowGather {
            source: DType::BF16,
        } => Planned::imported(
            ImportedKernel::PackedBf16RowGather,
            backend,
            None,
            [out_numel as u32, 1, 1],
        )?,
        // POOT-1477: the same gather over the packed-f16 table `store_weights_f16` declares for an embedding the
        // driver stores as f16 (two binary16 elements per lane, the layout of a dense F16 weight). Its row
        // values are the converted weight's, so the embedding and a tied head read one buffer.
        OpKind::DenseRowGather { source: DType::F16 } => Planned::imported(
            ImportedKernel::PackedF16RowGather,
            backend,
            None,
            [out_numel as u32, 1, 1],
        )?,
        // Card 449 D1: the same row gather over a packed-E4M3FN table (Qwen4Exp's sharded PLE
        // embedding). Same u32 transport, same row selection, same in-register widening, and the
        // same 3-param a single plain dispatch with no metadata buffer; only the byte-to-f32 decode differs
        // (E4M3FN instead of BF16). The index is the exact-I32 one `fold_dense_bf16_row_gathers`
        // pinned, which is what keeps n-gram ids above `2^24` from rounding through f32.
        //
        // All three typed packed walks (SpirvVulkan, AmdGcn, Nvptx) upload the table as the same
        // row-padded u32 lanes, so the identical body and key run on each.
        OpKind::DenseRowGather {
            source: DType::E4M3FN,
        } => Planned::imported(
            ImportedKernel::PackedE4m3RowGather,
            backend,
            None,
            [out_numel as u32, 1, 1],
        )?,
        // The BF16 and E4M3FN rows lower on every backend; an admitted source dtype with no
        // packed-lane kernel refuses by name rather than falling back to a gather kernel that would
        // read the table at the wrong element width.
        OpKind::DenseRowGather { .. } => {
            return Err(site.refuse(Capability::DtypeLowering));
        }
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | side_output_ops!()
        | moe_ops!()
        | elementwise_ops!()
        | matmul_ops!()
        | attention_ops!()
        | scan_ops!()
        | block_top_k_ops!()
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

/// The dense decode GEMV over an `[N, K]` weight held in `weight` storage, with `epilogue` applied to each column's
/// sum: one generated body per storage, launch and epilogue, shape-generic (`K` and `N` ride in its metadata), so
/// every weight dtype and backend takes the same loop nest (Card 727).
#[expect(clippy::too_many_arguments)]
fn dense_gemv(
    site: Site<'_>,
    backend: Backend,
    weight: poot_target::BufferStorage,
    schedule: kg::Schedule,
    k: usize,
    n: usize,
    epilogue: Option<kg::Epilogue>,
    precision: OperandPrecision,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    // The longest binding: the `[N, K]` weight counted in elements (at least its words), above the `K` activation,
    // the `N` output, an epilogue leaf of at most `N` and the two metadata words.
    let longest = (n as u64)
        .saturating_mul(k as u64)
        .max(k as u64)
        .max(n as u64)
        .max(2);
    Planned::generated_bounded(
        site,
        KernelRequest::Contraction(ContractionSpec::DenseGemv {
            weight,
            schedule,
            k,
            n,
            epilogue,
            precision,
        }),
        backend,
        caps,
        longest,
    )
}
