//! Data-movement equation planning: gather/scatter/slice/concat/transpose/broadcast/reshape,
//! dynamic KV-cache updates, the packed E4M3FN variants of those ops, and the tensor-parallel
//! collectives (AllReduce/AllGather).

use super::*;
use poot_target::{Backend, DeviceCaps};

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
    let fty = fty_ty;
    let generate = |request: KernelRequest| Planned::generated(site, request, backend, caps);
    // Spec 149 SC-005: packed-row writers own one physical u32 output word per thread, so the launch
    // uses the row-padded word count of the output allocation, not out_numel (card 525).
    let e4m3 = |op: E4m3Movement| {
        generate(KernelRequest::Movement(MovementSpec::E4m3 {
            out_shape: out_shape.to_vec(),
            op,
        }))
    };
    let update_static = |axis: usize, index: usize, extent: usize| {
        generate(KernelRequest::Movement(MovementSpec::UpdateSliceStatic {
            dt: fty(odt),
            out_shape: out_shape.to_vec(),
            axis,
            index,
            extent,
            numel: out_numel,
        }))
    };
    let planned = match &eqn.op {
        OpKind::ScatterUpdate if odt == DType::E4M3FN => {
            let base_shape = shape(ids[0]);
            let src_shape = shape(ids[1]);
            let inverse_shape = shape(ids[2]);
            if out_numel == 0
                || src_shape.iter().product::<usize>() == 0
                || inverse_shape.iter().product::<usize>() == 0
            {
                return Err(site.refuse(Capability::ZeroElementE4m3));
            }
            e4m3(E4m3Movement::ScatterUpdate {
                base_shape,
                src_shape,
            })?
        }
        OpKind::Transpose { perm } if odt == DType::E4M3FN => {
            if out_shape.iter().product::<usize>() == 0 {
                return Err(site.refuse(Capability::ZeroElementE4m3));
            }
            if perm.iter().copied().eq(0..perm.len()) {
                Planned::alias(ids[0])
            } else {
                e4m3(E4m3Movement::Transpose {
                    in_shape: shape(ids[0]),
                    perm: perm.clone(),
                })?
            }
        }
        OpKind::Slice { axis, start, end } if odt == DType::E4M3FN => {
            if out_shape.contains(&0) {
                return Err(site.refuse(Capability::ZeroElementE4m3));
            }
            let input_shape = shape(ids[0]);
            if *start == 0 && *end == input_shape[*axis] {
                Planned::alias(ids[0])
            } else {
                e4m3(E4m3Movement::Slice {
                    in_shape: input_shape,
                    axis: *axis,
                    start: *start,
                })?
            }
        }
        OpKind::Concat { axis } if odt == DType::E4M3FN => {
            let input_shapes: Vec<Vec<usize>> = ids.iter().map(|&id| shape(id)).collect();
            if out_shape.iter().product::<usize>() == 0
                || input_shapes
                    .iter()
                    .any(|input_shape| input_shape.iter().product::<usize>() == 0)
            {
                return Err(site.refuse(Capability::ZeroElementE4m3));
            }
            if ids.len() == 1 {
                Planned::alias(ids[0])
            } else {
                e4m3(E4m3Movement::Concat {
                    axis: *axis,
                    in_shapes: input_shapes,
                })?
            }
        }
        OpKind::Gather { axis } if odt == DType::E4M3FN => {
            if out_shape.contains(&0) {
                return Err(site.refuse(Capability::ZeroElementE4m3));
            }
            if g.aval(ids[1]).dtype != DType::F32 {
                return Err(site.refuse(Capability::DtypeLowering));
            }
            let input_shape = shape(ids[0]);
            if input_shape[*axis] == 0 {
                return Err(site.refuse(Capability::ZeroElementE4m3));
            }
            e4m3(E4m3Movement::Gather {
                in_shape: input_shape,
                axis: *axis,
                index_shape: shape(ids[1]),
            })?
        }
        OpKind::DynamicUpdateSlice { axis } if odt == DType::E4M3FN => {
            let operand_shape = shape(ids[0]);
            let update_shape = shape(ids[1]);
            if operand_shape.contains(&0) || update_shape.contains(&0) {
                return Err(site.refuse(Capability::ZeroElementE4m3));
            }
            let extent = update_shape[*axis];
            match eqn.inputs.get(2) {
                Some(Operand::Lit(Scalar::I32(index))) => {
                    let Ok(index) = usize::try_from(*index) else {
                        return Err(site.refuse(Capability::IndexOperand));
                    };
                    let axis_len = operand_shape[*axis];
                    if index.checked_add(extent).is_none_or(|end| end > axis_len) {
                        let mut expected = operand_shape.clone();
                        expected[*axis] = index.saturating_add(extent).max(axis_len);
                        return Err(site.refuse(Capability::OperandShape {
                            value: ids[0],
                            shape: operand_shape.clone(),
                            expected,
                        }));
                    }
                    e4m3(E4m3Movement::DynamicUpdateSlice {
                        operand_shape,
                        update_shape,
                        axis: *axis,
                        index,
                    })?
                }
                Some(Operand::Value(index_id)) => {
                    if g.aval(*index_id).dtype != DType::F32 || !g.aval(*index_id).shape.is_empty()
                    {
                        return Err(site.refuse(Capability::IndexOperand));
                    }
                    e4m3(E4m3Movement::DynamicUpdateSliceRuntime {
                        operand_shape,
                        update_shape,
                        axis: *axis,
                    })?
                }
                _ => {
                    return Err(site.refuse(Capability::IndexOperand));
                }
            }
        }
        OpKind::Broadcast { .. } if odt == DType::E4M3FN => {
            if out_shape.contains(&0) {
                return Err(site.refuse(Capability::ZeroElementE4m3));
            }
            let input_shape = shape(ids[0]);
            if input_shape == out_shape {
                Planned::alias(ids[0])
            } else {
                e4m3(E4m3Movement::Broadcast {
                    in_shape: input_shape,
                })?
            }
        }
        OpKind::Reshape { .. } if odt == DType::E4M3FN => {
            let input_shape = shape(ids[0]);
            let input_row_len = input_shape.last().copied().unwrap_or(1);
            let output_row_len = out_shape.last().copied().unwrap_or(1);
            if out_shape.contains(&0) || input_row_len == 0 || output_row_len == 0 {
                return Err(site.refuse(Capability::ZeroElementE4m3));
            }
            if input_row_len == output_row_len
                || (input_row_len.is_multiple_of(4) && output_row_len.is_multiple_of(4))
            {
                Planned::alias(ids[0])
            } else {
                e4m3(E4m3Movement::RepackReshape {
                    input_row_len,
                    output_row_len,
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
        OpKind::Gather { axis: 0 } => {
            let data = shape(ids[0]);
            let rest = numel(&data[1..]);
            let exact_index = analysis.required(ids[1]);
            let exact_data = exact_i32_gather_data(analysis, eqn, ids[0])?;
            // card 044: the f32 embedding gather runs the imported-from-Rust kernel, a 3-param drop-in
            // for `gather_axis0_dt` (rest derived in-kernel as out.len()/index.len(), so no metadata
            // buffer). On all three backends: it is a plain `Plan::Compute` (no ComputeMeta), so the
            // NVPTX engine dispatches it through the same imported-Body path as the B=1 decode GEMV.
            // Any non-f32 dtype (the kernelgen kernel is dt-generic) falls back to kernelgen.
            if odt == DType::F32 && !exact_index {
                Planned::imported(
                    ImportedKernel::GatherAxis0,
                    backend,
                    None,
                    [out_numel as u32, 1, 1],
                )
            } else {
                generate(KernelRequest::Movement(MovementSpec::GatherAxis0 {
                    dt: movement_ty(odt, exact_data),
                    index_dt: if exact_index { Ty::I32 } else { Ty::F32 },
                    rest,
                    numel: out_numel,
                }))?
            }
        }
        OpKind::Gather { axis } => {
            // non-axis-0 gather (card 043): the general kernel. `inner` is the contiguous run after the gather
            // axis, `axis_len` the gathered dim, `idx_numel` the index size (1 for a scalar index).
            let data = shape(ids[0]);
            let index = shape(ids[1]);
            let inner = numel(&data[axis + 1..]);
            let axis_len = data[*axis];
            let idx_numel = numel(&index);
            let exact_index = analysis.required(ids[1]);
            let exact_data = exact_i32_gather_data(analysis, eqn, ids[0])?;
            // card 044: on f32 the general gather runs the imported-from-Rust kernel on all three
            // backends. Unlike the axis-0 gather, `inner` and `axis_len` are not derivable from the
            // buffer lengths, so they ride in a `[inner, axis_len]` ComputeMeta buffer (`idx_numel` is
            // derived in-kernel). Any non-f32 dtype keeps the dt-generic kernelgen kernel.
            if odt == DType::F32 && !exact_index {
                Planned::imported(
                    ImportedKernel::GatherAxis,
                    backend,
                    Some(vec![inner as u32, axis_len as u32]),
                    [out_numel as u32, 1, 1],
                )
            } else {
                generate(KernelRequest::Movement(MovementSpec::GatherAxis {
                    dt: movement_ty(odt, exact_data),
                    index_dt: if exact_index { Ty::I32 } else { Ty::F32 },
                    inner,
                    axis_len,
                    idx_numel,
                    numel: out_numel,
                }))?
            }
        }
        OpKind::Scatter { axis: 0 } => {
            let src = shape(ids[0]);
            let rest = numel(&src[1..]);
            // card 044: the f32 axis-0 scatter runs the imported-from-Rust kernel on all three
            // backends, a 3-param drop-in for `scatter_axis0_dt` (rest derived in-kernel as
            // src.len()/index.len(), no metadata buffer), mirroring the gather swap. It is a plain
            // `Plan::Compute`, so the NVPTX executor runs it through the same imported-Body path as the
            // GPTQ dequant / embedding gather. Any non-f32 dtype keeps kernelgen.
            if odt == DType::F32 {
                Planned::imported(
                    ImportedKernel::ScatterAxis0,
                    backend,
                    None,
                    [out_numel as u32, 1, 1],
                )
            } else {
                generate(KernelRequest::Movement(MovementSpec::ScatterAxis0 {
                    dt: fty(odt),
                    rest,
                    numel: out_numel,
                }))?
            }
        }
        OpKind::Scatter { axis: 1.. } => {
            unreachable!(
                "a non-axis-0 scatter is refused in planner.rs (Capability::ScatterNonZeroAxis) \
                 before this per-op dispatch runs"
            )
        }
        OpKind::ScatterUpdate => {
            // card 059: base[POOL,..rest], src[N,..rest], inv[POOL] -> base; rest = product of
            // base[1..]. card 159 Inc3: large f32 pools on wgpu/RADV chunk by output-element range (see
            // `scatter_update_plan`) so a wide shared-pool KV write (gemma4-dense batched prefill) does
            // not trip `poot-gpu`'s display-watchdog `flush_work` cap on its own.
            scatter_update_plan(site, backend, &shape(ids[0]), out_numel, odt, caps)?
        }
        OpKind::Transpose { .. } if views.contains_key(&eqn.out) => {
            // spec 132 phase 1: `compute_views` already proved every consumer of this value is
            // strided-capable (and this backend is SpirvVulkan): no dispatch, no materialize copy.
            Planned::view(ids[0], views[&eqn.out].clone())
        }
        OpKind::Transpose { perm } => {
            let a = shape(ids[0]);
            let exact_i32 = analysis.required(eqn.out);
            // card 044: transpose / slice / broadcast are all kernelgen's one `index_remap_copy`
            // primitive. On f32 they run the single imported kernel, with the per-op strides in a
            // ComputeMeta buffer, on all three backends (the PTX executor binds the dims buffer from
            // its const cache); these appear all over the decode graph. Transpose: out dim d reads
            // input stride `in_strides[perm[d]]`, no base offset.
            if odt == DType::F32 {
                let in_strides = row_major_strides(&a);
                let src_terms: Vec<usize> =
                    (0..out_shape.len()).map(|d| in_strides[perm[d]]).collect();
                Planned::imported(
                    ImportedKernel::IndexRemap,
                    backend,
                    Some(index_remap_meta(out_shape, &src_terms, 0)),
                    default_grid_in(g, backend, eqn, out_shape, out_numel, caps),
                )
            } else {
                generate(KernelRequest::Movement(MovementSpec::Transpose {
                    dt: movement_ty(odt, exact_i32),
                    out_shape: out_shape.to_vec(),
                    in_shape: a,
                    perm: perm.clone(),
                    numel: out_numel,
                }))?
            }
        }
        OpKind::Slice { .. } if views.contains_key(&eqn.out) => {
            // spec 132 phase 1: see the Transpose view arm above.
            Planned::view(ids[0], views[&eqn.out].clone())
        }
        OpKind::Slice { axis, start, .. } => {
            let a = shape(ids[0]);
            let exact_i32 = analysis.required(eqn.out);
            // card 044: slice via the imported index_remap kernel (all three backends). out dim d reads input
            // stride `in_strides[d]`; the slice offset rides in `src_base = start * in_strides[axis]`.
            if odt == DType::F32 {
                let in_strides = row_major_strides(&a);
                let src_terms: Vec<usize> = (0..out_shape.len()).map(|d| in_strides[d]).collect();
                Planned::imported(
                    ImportedKernel::IndexRemap,
                    backend,
                    Some(index_remap_meta(
                        out_shape,
                        &src_terms,
                        start * in_strides[*axis],
                    )),
                    default_grid_in(g, backend, eqn, out_shape, out_numel, caps),
                )
            } else {
                generate(KernelRequest::Movement(MovementSpec::Slice {
                    dt: movement_ty(odt, exact_i32),
                    out_shape: out_shape.to_vec(),
                    in_shape: a,
                    axis: *axis,
                    start: *start,
                    numel: out_numel,
                }))?
            }
        }
        OpKind::Broadcast { .. } if views.contains_key(&eqn.out) => {
            // spec 132 phase 1: see the Transpose view arm above.
            Planned::view(ids[0], views[&eqn.out].clone())
        }
        OpKind::Broadcast { shape: s } => {
            let a = shape(ids[0]);
            let exact_i32 = analysis.required(eqn.out);
            // card 044: broadcast via the imported index_remap kernel (all three backends). The effective input
            // stride is 0 on every broadcast dim (right-aligned), else the operand's own row-major stride.
            if odt == DType::F32 {
                let src_terms = broadcast_eff_strides(s, &a);
                Planned::imported(
                    ImportedKernel::IndexRemap,
                    backend,
                    Some(index_remap_meta(s, &src_terms, 0)),
                    default_grid_in(g, backend, eqn, out_shape, out_numel, caps),
                )
            } else {
                generate(KernelRequest::Movement(MovementSpec::Broadcast {
                    dt: movement_ty(odt, exact_i32),
                    out_shape: s.clone(),
                    in_shape: a,
                    numel: out_numel,
                }))?
            }
        }
        OpKind::Concat { axis } if ids.len() == 2 => {
            let (a, b) = (shape(ids[0]), shape(ids[1]));
            let exact_i32 = analysis.required(eqn.out);
            // card 044: the f32 two-input concat (RoPE rotate_half, KV append, MoE row assembly) runs the
            // imported branchless kernel on all three backends - the per-dim a/b strides ride in a ComputeMeta
            // buffer. Any non-f32 dtype keeps kernelgen.
            if odt == DType::F32 {
                Planned::imported(
                    ImportedKernel::Concat2,
                    backend,
                    Some(concat2_meta(out_shape, *axis, &a, &b)),
                    default_grid_in(g, backend, eqn, out_shape, out_numel, caps),
                )
            } else {
                generate(KernelRequest::Movement(MovementSpec::Concat2 {
                    dt: movement_ty(odt, exact_i32),
                    out_shape: out_shape.to_vec(),
                    axis: *axis,
                    a_shape: a,
                    b_shape: b,
                    numel: out_numel,
                }))?
            }
        }
        OpKind::Concat { axis } => {
            // the N != 2 case (>2 inputs, or the degenerate 1-input copy): the general N-input kernel
            // (card 043), so concat stays on-device for any input count. `is_elementwise_2d_in`'s
            // Concat arm only matches the 2-input case above, so this is always the plain default.
            let exact_i32 = analysis.required(eqn.out);
            generate(KernelRequest::Movement(MovementSpec::ConcatN {
                dt: movement_ty(odt, exact_i32),
                out_shape: out_shape.to_vec(),
                axis: *axis,
                in_shapes: ids.iter().map(|&id| shape(id)).collect(),
                numel: out_numel,
            }))?
        }
        OpKind::DynamicUpdateSlice { axis } => {
            // extent is the update's axis length (static); ids[1] is the update in both index forms.
            let extent = shape(ids[1])[*axis];
            match &eqn.inputs[2] {
                // baked literal index (G2 pos-specialized): static kernel, 2 input buffers.
                Operand::Lit(Scalar::I32(v)) => update_static(*axis, *v as usize, extent)?,
                Operand::Lit(Scalar::F32(v)) => update_static(*axis, *v as usize, extent)?,
                // runtime index read from a buffer (G3d constant-shape decode): dynamic kernel, 3 input
                // buffers (operand, update, index). The index value is not baked.
                // card 044: on f32 (the contiguous KV-cache write hot path) this runs the imported
                // kernel on all three backends: the per-dim strides ride in a ComputeMeta buffer, `idx`
                // is read from the index buffer at runtime (so captures stay valid across positions).
                // Non-f32 data keeps kernelgen. An I32 index (a paged/shared-pool physical slot, or an
                // exact-I32 `Slot::Pos`) reads the same buffer bytes as an F32 index here: both are
                // real, load-bearing shapes (`components::kv_pool::scatter_shared_pool`'s `phys_row`,
                // `Slot::SlotMap` being I32 everywhere), so the arm keeps one body for either dtype. But
                // R469-005 (card 529): the index's OWN dtype is part of the generated request, not just
                // the data dtype `odt` - an F32-index and an I32-index DUS must never collide on one
                // kernel identity, since a future change to how either dtype binds must not silently
                // affect the other's cache entry. The typed packed walk widens an I32 index to F32 there
                // (statically exact for axis extent < 2^24); this arm is the resident path.
                Operand::Value(index_id) if odt == DType::F32 => Planned::imported_indexed(
                    ImportedKernel::DynUpdateSlice,
                    backend,
                    Some(dus_meta(out_shape, *axis, extent)),
                    default_grid_in(g, backend, eqn, out_shape, out_numel, caps),
                    Some(g.aval(*index_id).dtype),
                ),
                Operand::Value(index_id) => {
                    generate(KernelRequest::Movement(MovementSpec::UpdateSliceRuntime {
                        dt: fty(odt),
                        index_dt: if g.aval(*index_id).dtype == DType::I32 {
                            Ty::I32
                        } else {
                            Ty::F32
                        },
                        out_shape: out_shape.to_vec(),
                        axis: *axis,
                        extent,
                        numel: out_numel,
                    }))?
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
        | cast_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only data-movement ops here"),
    };
    Ok(planned)
}
