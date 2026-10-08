//! Data-movement equation planning: gather/scatter/slice/concat/transpose/broadcast/reshape and dynamic
//! KV-cache updates, all through the one generated movement kernel (`poot_kernelgen::MovementSpec`), and
//! the tensor-parallel collectives (AllReduce/AllGather).
//!
//! A movement names its storage, not its dtype: 32-bit values (and 16-bit natively stored ones) move
//! element by element, an E4M3FN payload moves as packed byte lanes with its rows padded to a word. The
//! generator copies bits either way, so no storage is decoded or re-encoded on the way.

use super::*;
use poot_kernelgen::{MovementOp, MovementSpec, Storage, UpdateIndex};
use poot_target::{Backend, DeviceCaps};

/// The storage a movement over values of `dtype` reads and writes: the packed byte rows of an E4M3FN
/// value, else one element per value in its movement type (an exact-I32 value keeps its i32 bits).
pub(crate) fn movement_storage(dtype: DType, exact_i32: bool) -> Storage {
    match dtype {
        DType::E4M3FN => Storage::BYTE_LANES,
        _ => Storage::Elements {
            carrier: movement_ty(dtype, exact_i32),
        },
    }
}

/// The element type of a movement's F32 or I32 index buffer.
fn index_ty(exact_i32: bool) -> Ty {
    if exact_i32 { Ty::I32 } else { Ty::F32 }
}

/// Plan one data-movement equation (including the e4m3fn packed forms and the collectives).
#[allow(clippy::too_many_arguments)]
pub(super) fn plan(
    analysis: ExactI32Requirements<'_>,
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    views: &HashMap<ValueId, Layout>,
    odt: DType,
    world_size: usize,
    caps: &DeviceCaps,
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let ids = value_ids(eqn);
    let storage = movement_storage(odt, analysis.required(eqn.out));
    let packed = odt == DType::E4M3FN;
    // A packed payload is never read through a strided view: the layout has no row padding to stride over.
    let view = views.get(&eqn.out).filter(|_| !packed);
    // One thread per output buffer unit, folded onto a 2-D grid past the device's x-grid cap.
    let generate = |op: MovementOp| {
        let spec = MovementSpec::new(storage.clone(), out_shape.to_vec(), op);
        let two_d = is_elementwise_2d_in(g, backend, eqn, out_shape, out_numel, caps);
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
        // card 049a (tensor-parallel). At world_size=1 both collectives are the identity: with one
        // replica AllReduce(x)=x and AllGather(x)=x (the shard is the full tensor), and the op is
        // shape-preserving in the single-rank graph, so the output aliases the input buffer, like
        // Reshape. This makes a sharded TP graph GPU-executable at ws=1 (FR-007).
        //
        // At world_size>1 the collective lowers to `Plan::Collective`, a marker the multi-rank executor
        // consumes: AllReduce runs poot's own ring reduce-scatter + all-gather over real cross-device
        // buffers (combine per `op` along the shard `axis`), AllGather runs the ring all-gather (concat
        // along `axis`). Cross-device comms do not belong in this pure device-local planner (`plan_eqn`
        // cannot know peer buffers), so it only tags which collective goes here; the single-device
        // executors reject `Plan::Collective` (they plan at ws=1 and never see it) rather than run a
        // wrong local kernel.
        OpKind::AllReduce { op, axis } => {
            if world_size <= 1 {
                Planned::alias(ids[0])
            } else {
                Planned::collective(CollectiveKind::AllReduce, *op, *axis)
            }
        }
        OpKind::AllGather { axis } => {
            if world_size <= 1 {
                Planned::alias(ids[0])
            } else {
                // `op` is unused for AllGather (it concatenates, no reduction); carry Sum as a filler.
                Planned::collective(CollectiveKind::AllGather, RedOp::Sum, *axis)
            }
        }
        OpKind::Gather { axis } => {
            let data = shape(ids[0]);
            let index_shape = shape(ids[1]);
            let exact_index = analysis.required(ids[1]);
            refuse_zero(out_shape.contains(&0) || data[*axis] == 0)?;
            // An E4M3FN gather indexes by the graph's f32 index convention.
            if packed && g.aval(ids[1]).dtype != DType::F32 {
                return Err(site.refuse(Capability::DtypeLowering));
            }
            generate(MovementOp::Gather {
                in_shape: data,
                axis: *axis,
                index_dtype: index_ty(exact_index),
                index_numel: numel(&index_shape),
            })?
        }
        OpKind::Scatter { axis: 0 } => generate(MovementOp::PermuteRows {
            src_shape: shape(ids[0]),
            index_dtype: index_ty(analysis.required(ids[1])),
        })?,
        OpKind::Scatter { axis: 1.. } => {
            unreachable!(
                "a non-axis-0 scatter is refused in planner.rs (Capability::ScatterNonZeroAxis) \
                 before this per-op dispatch runs"
            )
        }
        OpKind::ScatterUpdate => {
            let src_shape = shape(ids[1]);
            refuse_zero(
                out_shape.contains(&0) || src_shape.contains(&0) || shape(ids[2]).contains(&0),
            )?;
            // card 059: base[POOL,..rest], src[N,..rest], inv[POOL] -> base. card 159 Inc3: a large pool
            // chunks by output-unit range (see `scatter_update_plan`) so a wide shared-pool KV write
            // (gemma4-dense batched prefill) does not trip `poot-gpu`'s display-watchdog `flush_work` cap
            // on its own.
            let inverse_dtype = index_ty(analysis.required(ids[2]));
            scatter_update_plan(
                site,
                backend,
                &storage,
                out_shape,
                src_shape,
                inverse_dtype,
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
                // An F32 literal is a dense index only; an E4M3FN update takes an I32 literal.
                Operand::Lit(Scalar::F32(index)) if !packed => UpdateIndex::Static(*index as usize),
                // A runtime index read from a buffer (G3d constant-shape decode): three input buffers
                // (operand, update, index), so a captured dispatch is valid at every position. The index
                // is read on the lane its value's component is stored on: an exact-I32 component (a
                // paged/shared-pool physical slot, or an exact-I32 `Slot::Pos`) as i32, every other value
                // as the f32 mirror. The element type is part of the request, so the two never share a
                // kernel identity (R469-005, card 529) and the storage analysis sees which lane reads it.
                Operand::Value(index_id) => {
                    let index_aval = g.aval(*index_id);
                    if packed && (index_aval.dtype != DType::F32 || !index_aval.shape.is_empty()) {
                        return Err(site.refuse(Capability::DtypeLowering));
                    }
                    UpdateIndex::Runtime {
                        dtype: index_ty(analysis.required(*index_id)),
                    }
                }
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
            generate(MovementOp::UpdateSlice {
                update_shape,
                axis: *axis,
                index,
            })?
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
        | cast_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only data-movement ops here"),
    };
    Ok(planned)
}
