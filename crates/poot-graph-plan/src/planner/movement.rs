//! Data-movement equation planning: gather/scatter/slice/concat/transpose/broadcast/reshape and dynamic
//! KV-cache updates, all through the one generated movement kernel (`poot_kernelgen::MovementSpec`). The
//! tensor-parallel collectives (AllReduce/AllGather) are refused: a collective is a cut between stages.
//!
//! A movement names its storage, not its dtype: 32-bit values (and 16-bit natively stored ones) move
//! element by element, an E4M3FN payload moves as packed byte lanes with its rows padded to a word. The
//! generator copies bits either way, so no storage is decoded or re-encoded on the way.

use super::{
    BodyLimits, Capability, ContractionHead, DType, Eqn, HashMap, KernelRequest, OpKind, Operand,
    PlanError, Planned, Scalar, ValueId, fty_ty, numel, value_ids,
};
use crate::planner::{GraphTables, Layout, Lowering, OutputBuffer, StorageDtypes};
use crate::predicates::{ELEMENTWISE_2D_WIDTH, is_elementwise_2d_in, scatter_update_plan};
use crate::refusal::Site;
use crate::{Binding, Sink, Source};
use poot_kernelgen::{MovementOp, MovementSpec, Storage, UpdateIndex};
use poot_target::{AieTargetId, Backend, DeviceCaps};

/// The storage a movement over values of `dtype` reads and writes: the packed byte rows of an E4M3FN
/// value, else one element per value in its own element type (an I32 value keeps its i32 bits).
fn movement_storage(dtype: DType) -> Storage {
    match dtype {
        DType::E4M3FN => Storage::BYTE_LANES,
        _ => Storage::Elements {
            carrier: fty_ty(dtype),
        },
    }
}

/// Plan a gather for an AIE core (POOT-1174): the movement request every target plans, under the target's
/// serialized one-core schedule. A gather copies bits through an `I32` index, so it runs on the core for any
/// `F32` or `I32` data; every other data movement reads GPU capabilities and is refused as such.
pub(super) fn plan_serialized(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    target: AieTargetId,
    odt: DType,
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, target, limits);
    let axis = match &eqn.op {
        OpKind::Gather { axis } => axis,
        // A collective is a cut between stage programs on every target; any other movement's lowering
        // reads GPU capabilities.
        OpKind::AllReduce { .. } | OpKind::AllGather { .. } => {
            return Err(site.refuse(Capability::CollectiveInStage));
        }
        _ => return Err(site.refuse(Capability::GpuOnly)),
    };
    let ids = value_ids(eqn);
    if g.aval(ids[1]).dtype != DType::I32 {
        return Err(site.refuse(Capability::IndexOperand));
    }
    let data = g.aval(ids[0]).shape.clone();
    let spec = MovementSpec::new(
        movement_storage(odt),
        out_shape.to_vec(),
        MovementOp::Gather {
            index_numel: numel(&g.aval(ids[1]).shape),
            in_shape: data,
            axis: *axis,
        },
    );
    Planned::lowered(site, KernelRequest::Movement(spec), Lowering::Aie(target))
}

/// Whether this family lowers `op` over these storage dtypes. Movement is dtype-agnostic bytes, so a packed
/// E4M3FN payload moves wherever the device stores it natively, and a 2-byte float moves everywhere; the
/// collectives are wired for neither. A write into a carried buffer (an update or a scatter) stores in the
/// buffer's own element type, so it writes BF16 only where the device has a bf16 lane (SPIR-V has none); F16
/// is native everywhere.
pub(super) fn admits_dtypes(
    op: &OpKind,
    dtypes: StorageDtypes,
    caps: &DeviceCaps,
) -> Result<(), Capability> {
    let collective = matches!(op, OpKind::AllReduce { .. } | OpKind::AllGather { .. });
    if dtypes.touches_e4m3fn()
        && (collective || caps.e4m3_storage == poot_target::E4m3Storage::None)
    {
        return Err(Capability::E4m3Storage);
    }
    let write = matches!(
        op,
        OpKind::ScatterUpdate | OpKind::DynamicUpdateSlice { .. }
    );
    let bf16_write = write && dtypes.out == DType::BF16 && !caps.native_two_byte_floats;
    if dtypes.out_is_half() && (collective || bf16_write) {
        return Err(Capability::DtypeLowering);
    }
    Ok(())
}

/// Plan one data-movement equation (including the e4m3fn packed forms) and refuse the collectives.
/// `output` says whether the output buffer already holds an update's base (a donated state), in which
/// case the update writes only its window.
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
    output: OutputBuffer,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let ids = value_ids(eqn);
    let storage = movement_storage(odt);
    let packed = odt == DType::E4M3FN;
    // A write of f32 values into a two-byte buffer stores them narrowed (the IR's `stores_as`): the body
    // reads the update in its own element type and converts on the store.
    let written = match &eqn.op {
        OpKind::DynamicUpdateSlice { .. } | OpKind::ScatterUpdate
            if g.aval(ids[1]).dtype != odt =>
        {
            Some(fty_ty(g.aval(ids[1]).dtype))
        }
        _ => None,
    };
    let with_written = |spec: MovementSpec| match &written {
        Some(ty) => spec.writing(ty.clone()),
        None => spec,
    };
    // A packed payload is never read through a strided view: the layout has no row padding to stride over.
    let view = views.get(&eqn.out).filter(|_| !packed);
    // One thread per output buffer unit, folded onto a 2-D grid past the device's x-grid cap.
    let generate = |op: MovementOp| {
        let spec = with_written(MovementSpec::new(storage.clone(), out_shape.to_vec(), op));
        let two_d = is_elementwise_2d_in(g, eqn, out_shape, out_numel, caps);
        Planned::generated(
            site,
            KernelRequest::Movement(spec.folded(two_d)),
            backend,
            caps,
        )
    };
    // A packed E4M3FN payload has no zero-element form: refuse before the generator sees it.
    let refuse_zero = |empty: bool| {
        if packed && empty {
            return Err(site.refuse(Capability::ZeroElementE4m3));
        }
        Ok(())
    };
    let planned = match &eqn.op {
        OpKind::Reshape { .. } if packed => {
            let input_shape = shape(ids[0]);
            let input_row_len = input_shape.last().copied().unwrap_or(1);
            let output_row_len = out_shape.last().copied().unwrap_or(1);
            refuse_zero(out_shape.contains(&0) || input_row_len == 0 || output_row_len == 0)?;
            // Rows that keep their length, or whose length is a whole number of words either side, share
            // one layout: nothing to move. Otherwise row padding differs and the bytes repack.
            if input_row_len == output_row_len
                || (input_row_len.is_multiple_of(4) && output_row_len.is_multiple_of(4))
            {
                Planned::alias(ids[0])
            } else {
                generate(MovementOp::Reshape {
                    in_shape: input_shape,
                })?
            }
        }
        OpKind::Reshape { .. } => Planned::alias(ids[0]),
        // A collective is a cut point between stage programs (`compile_staged` places the transfer between
        // them), never an equation inside one: no stage program holds it, so planning it is a refusal, not
        // an identity that would silently drop the cross-device exchange.
        OpKind::AllReduce { .. } | OpKind::AllGather { .. } => {
            return Err(site.refuse(Capability::CollectiveInStage));
        }
        OpKind::Gather { axis } => {
            let data = shape(ids[0]);
            let index_shape = shape(ids[1]);
            refuse_zero(out_shape.contains(&0) || data[*axis] == 0)?;
            generate(MovementOp::Gather {
                in_shape: data,
                axis: *axis,
                index_numel: numel(&index_shape),
            })?
        }
        OpKind::ScatterUpdate => {
            let src_shape = shape(ids[1]);
            refuse_zero(
                out_shape.contains(&0) || src_shape.contains(&0) || shape(ids[2]).contains(&0),
            )?;
            // card 059: base[POOL,..rest], src[N,..rest], inv[POOL] -> base. card 159 Inc3: a large pool
            // chunks by output-unit range (see `scatter_update_plan`) so a wide shared-pool KV write
            // (gemma4-dense batched prefill) does not trip `poot-wgpu-device`'s display-watchdog `flush_work` cap
            // on its own.
            scatter_update_plan(
                site,
                backend,
                &storage,
                written.as_ref(),
                out_shape,
                src_shape,
                caps,
            )?
        }
        OpKind::Transpose { perm } => {
            refuse_zero(out_shape.contains(&0))?;
            if let Some(layout) = view {
                // spec 132 phase 1: `compute_views` already proved every consumer of this value is
                // strided-capable (and this backend is SpirvVulkan): no dispatch, no materialize copy.
                Planned::view(ids[0], layout.clone())
            } else if packed && perm.iter().copied().eq(0..perm.len()) {
                Planned::alias(ids[0])
            } else {
                generate(MovementOp::Transpose {
                    in_shape: shape(ids[0]),
                    perm: perm.clone(),
                })?
            }
        }
        OpKind::Slice { axis, start, end } => {
            refuse_zero(out_shape.contains(&0))?;
            let input_shape = shape(ids[0]);
            if let Some(layout) = view {
                // spec 132 phase 1: see the Transpose view arm above.
                Planned::view(ids[0], layout.clone())
            } else if packed && *start == 0 && *end == input_shape[*axis] {
                Planned::alias(ids[0])
            } else {
                generate(MovementOp::Slice {
                    in_shape: input_shape,
                    axis: *axis,
                    start: *start,
                })?
            }
        }
        OpKind::Broadcast { .. } => {
            refuse_zero(out_shape.contains(&0))?;
            let input_shape = shape(ids[0]);
            if let Some(layout) = view {
                // spec 132 phase 1: see the Transpose view arm above.
                Planned::view(ids[0], layout.clone())
            } else if packed && input_shape == out_shape {
                Planned::alias(ids[0])
            } else {
                generate(MovementOp::Broadcast {
                    in_shape: input_shape,
                })?
            }
        }
        OpKind::Concat { axis } => {
            let in_shapes: Vec<Vec<usize>> = ids.iter().map(|&id| shape(id)).collect();
            refuse_zero(
                out_shape.contains(&0) || in_shapes.iter().any(|shape| shape.contains(&0)),
            )?;
            if packed && ids.len() == 1 {
                Planned::alias(ids[0])
            } else {
                generate(MovementOp::Concat {
                    axis: *axis,
                    in_shapes,
                })?
            }
        }
        OpKind::DynamicUpdateSlice { axis } => {
            let operand_shape = shape(ids[0]);
            let update_shape = shape(ids[1]);
            refuse_zero(operand_shape.contains(&0) || update_shape.contains(&0))?;
            let extent = update_shape[*axis];
            let index = match &eqn.inputs[2] {
                // A baked literal index (G2 pos-specialized): the static form, two input buffers.
                Operand::Lit(Scalar::I32(index)) => {
                    let Ok(index) = usize::try_from(*index) else {
                        return Err(site.refuse(Capability::IndexOperand));
                    };
                    UpdateIndex::Static(index)
                }
                // A runtime index read from a buffer (G3d constant-shape decode): three input buffers
                // (operand, update, index), so a captured dispatch is valid at every position. The index
                // is an I32 word, read as i32 on every backend.
                Operand::Value(_) => UpdateIndex::Runtime,
                _ => return Err(site.refuse(Capability::IndexOperand)),
            };
            if let UpdateIndex::Static(start) = index {
                let axis_len = operand_shape[*axis];
                if start.checked_add(extent).is_none_or(|end| end > axis_len) {
                    let mut expected = operand_shape.clone();
                    expected[*axis] = start.saturating_add(extent).max(axis_len);
                    return Err(site.refuse(Capability::OperandShape {
                        value: ids[0],
                        shape: operand_shape,
                        expected,
                    }));
                }
            }
            match output {
                OutputBuffer::Fresh => generate(MovementOp::UpdateSlice {
                    update_shape,
                    axis: *axis,
                    index,
                })?,
                // POOT-1389: the donated base is already in the output buffer, so the kernel reads only
                // the update (and its runtime index) and launches over the window it writes, never the
                // whole state.
                OutputBuffer::HoldsBase => {
                    let mut reads = vec![Source::Operand(1)];
                    if index == UpdateIndex::Runtime {
                        reads.push(Source::Operand(2));
                    }
                    let spec = with_written(MovementSpec::new(
                        storage.clone(),
                        out_shape.to_vec(),
                        MovementOp::WriteSlice {
                            update_shape,
                            axis: *axis,
                            index,
                        },
                    ));
                    let two_d = spec.threads() > caps.max_grid[0] as usize * ELEMENTWISE_2D_WIDTH;
                    let write = Planned::generated_dispatch(
                        site,
                        KernelRequest::Movement(spec.folded(two_d)),
                        backend,
                        caps,
                    )?;
                    Planned::split(vec![write.bound(Binding::new(reads, Sink::Out))])?
                }
            }
        }
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | packed_ops!()
        | moe_ops!()
        | elementwise_ops!()
        | matmul_ops!()
        | attention_ops!()
        | scan_ops!()
        | block_top_k_ops!()
        | cast_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only data-movement ops here"),
    };
    Ok(planned)
}
