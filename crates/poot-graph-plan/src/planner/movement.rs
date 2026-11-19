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
use crate::predicates::{
    ELEMENTWISE_2D_WIDTH, LayoutDemand, compose_movement, is_elementwise_2d_in,
    scatter_update_plan, view_source_shape,
};
use crate::refusal::Site;
use crate::{Binding, Sink, Source};
use poot_kernelgen::{MovementOp, MovementRead, MovementSpec, Storage, UpdateIndex};
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

/// What this family needs of the layout it reads a value operand through. A `Transpose`, `Slice`, `Broadcast` or
/// `Reshape` reads its data through the composed layout of the chain it ends, and a `Concat` reads each part
/// through its own, both over element storage (f32 or I32 words; a two-byte element is read dense). A gather,
/// a scatter, a dynamic update and the collectives read their operands flat.
pub(super) fn layout_demand(g: GraphTables<'_>, eqn: &Eqn, value: ValueId) -> LayoutDemand {
    let words = matches!(g.aval(value).dtype, DType::F32 | DType::I32);
    match &eqn.op {
        OpKind::Transpose { .. }
        | OpKind::Slice { .. }
        | OpKind::Broadcast { .. }
        | OpKind::Reshape { .. }
        | OpKind::Concat { .. }
            if words =>
        {
            LayoutDemand::Any
        }
        _ => LayoutDemand::Dense,
    }
}

/// Plan one data-movement equation (including the e4m3fn packed forms) and refuse the collectives.
/// `output` says whether the output buffer already holds an update's base (a donated state), in which
/// case the update writes only its window.
#[expect(
    clippy::too_many_arguments,
    reason = "a per-op planner arm: the graph, equation and output facts plus the lowering context (backend, device caps, limits, pins, constant lanes) every arm receives alike from the dispatch in `planner.rs`"
)]
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
    // A promoted movement is a view of its root (`compute_views`); a packed payload is never one: its rows are
    // padded to a word, so no layout strides over them.
    let view = views.get(&eqn.out).filter(|_| !packed);
    // One thread per output buffer unit, folded onto a 2-D grid past the device's x-grid cap.
    let generate = |op: MovementOp| {
        let spec = with_written(MovementSpec::new(storage.clone(), out_shape.to_vec(), op));
        let two_d = is_elementwise_2d_in(g, eqn, out_shape, out_numel, caps);
        Planned::generated(
            site,
            KernelRequest::Movement(spec.launched(two_d, caps)),
            backend,
            caps,
        )
    };
    // The one copy that materializes a movement chain's tail: it reads the chain's root through the layout the
    // chain composes to, from the input's own layout when the input is a view.
    let copy = || {
        let input = ids[0];
        let in_shape = shape(input);
        let in_layout = views
            .get(&input)
            .cloned()
            .unwrap_or_else(|| Layout::contiguous(&in_shape));
        let from = compose_movement(&eqn.op, &in_shape, &in_layout, out_shape)
            .ok_or_else(|| site.refuse(Capability::OperandLayout { value: input }))?;
        generate(MovementOp::Copy {
            source: view_source_shape(g, views, input),
            from,
        })
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
                generate(MovementOp::Copy {
                    source: input_shape,
                    from: Layout::contiguous(out_shape),
                })?
            }
        }
        // A reshape of a view is a view of the same buffer with the split or merged strides
        // (`compute_views`); a reshape of a dense value is the buffer itself. A reshape that is not promoted
        // but reads a view (its value must be dense) is the one copy of the chain.
        OpKind::Reshape { .. } => match view {
            Some(layout) => Planned::view(ids[0], layout.clone()),
            None if views.contains_key(&ids[0]) => copy()?,
            None => Planned::alias(ids[0]),
        },
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
                // spec 132 phase 1: `compute_views` already proved every consumer of this value reads a layout
                // (and this backend is SpirvVulkan): no dispatch, no materialize copy.
                Planned::view(ids[0], layout.clone())
            } else if packed && perm.iter().copied().eq(0..perm.len()) {
                Planned::alias(ids[0])
            } else {
                copy()?
            }
        }
        OpKind::Slice { axis, start, end } => {
            refuse_zero(out_shape.contains(&0))?;
            if let Some(layout) = view {
                // spec 132 phase 1: see the Transpose view arm above.
                Planned::view(ids[0], layout.clone())
            } else if packed && *start == 0 && *end == shape(ids[0])[*axis] {
                Planned::alias(ids[0])
            } else {
                copy()?
            }
        }
        OpKind::Broadcast { .. } => {
            refuse_zero(out_shape.contains(&0))?;
            if let Some(layout) = view {
                // spec 132 phase 1: see the Transpose view arm above.
                Planned::view(ids[0], layout.clone())
            } else if packed && shape(ids[0]) == out_shape {
                Planned::alias(ids[0])
            } else {
                copy()?
            }
        }
        OpKind::Concat { axis } => {
            let inputs: Vec<MovementRead> = ids
                .iter()
                .map(|&id| MovementRead {
                    shape: shape(id),
                    source: view_source_shape(g, views, id),
                    from: views
                        .get(&id)
                        .cloned()
                        .unwrap_or_else(|| Layout::contiguous(&shape(id))),
                })
                .collect();
            refuse_zero(
                out_shape.contains(&0) || inputs.iter().any(|input| input.shape.contains(&0)),
            )?;
            if packed && ids.len() == 1 {
                Planned::alias(ids[0])
            } else {
                generate(MovementOp::Concat {
                    axis: *axis,
                    inputs,
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
                        KernelRequest::Movement(spec.launched(two_d, caps)),
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
        | side_output_ops!()
        | packed_ops!()
        | moe_ops!()
        | elementwise_ops!()
        | matmul_ops!()
        | attention_ops!()
        | scan_ops!()
        | block_top_k_ops!()
        | stable_rank_ops!()
        | cast_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only data-movement ops here"),
    };
    Ok(planned)
}
