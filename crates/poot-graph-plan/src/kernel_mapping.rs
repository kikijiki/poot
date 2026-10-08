//! Mapping graph-level fused operations to kernel-IR operations, and the one finalizer every planned
//! kernel passes through before it can run.
//!
//! [`finalize`] shapes the launch (workgroup bump, 2-D fold, the plan key that names it) and then proves the kernel fits the device:
//! generated and imported kernels alike are checked against the measured [`DeviceCaps`] by
//! [`validate_kernel_resources`], so no plan that cannot run reaches a loader.

use poot_kernelgen::{KernelRequirements, SerialWork};
use poot_target::{Backend, BufferStorage, DeviceCaps};

use crate::device_validation::validate_kernel_resources;
use crate::imported::{SAMPLER_LANES, SerialLoop};
use crate::*;

/// A region-local id -> a fused-kernel operand: `< n_inputs` is a leaf, otherwise a prior step.
pub(crate) fn map_fused_local(id: usize, n_inputs: usize) -> kg::FusedInput {
    if id < n_inputs {
        kg::FusedInput::Leaf(id)
    } else {
        kg::FusedInput::Step(id - n_inputs)
    }
}

pub(crate) fn map_fused_input(
    site: Site<'_>,
    o: &FusedOperand,
    n_inputs: usize,
) -> Result<kg::FusedInput, PlanError> {
    match o {
        FusedOperand::Local(id) => Ok(map_fused_local(*id, n_inputs)),
        FusedOperand::Lit(Scalar::F32(v)) => Ok(kg::FusedInput::Lit(*v)),
        FusedOperand::Lit(Scalar::I32(v)) => Ok(kg::FusedInput::Lit(*v as f32)),
        FusedOperand::PackLane { .. } => {
            Err(site.refuse(Capability::FusedStep(FusedStepGap::PackedLaneInFloatRegion)))
        }
    }
}

pub(crate) fn map_fused_input_i32(
    site: Site<'_>,
    o: &FusedOperand,
    n_inputs: usize,
) -> Result<kg::FusedInput, PlanError> {
    match o {
        FusedOperand::Local(id) => Ok(map_fused_local(*id, n_inputs)),
        FusedOperand::Lit(Scalar::I32(v)) => Ok(kg::FusedInput::LitI32(*v)),
        FusedOperand::Lit(Scalar::F32(_)) => {
            Err(site.refuse(Capability::FusedStep(FusedStepGap::F32LiteralInExactRegion)))
        }
        FusedOperand::PackLane { input, lane } => Ok(kg::FusedInput::LeafLane {
            leaf: *input,
            lane: *lane,
        }),
    }
}

pub(crate) fn map_fused_op(site: Site<'_>, op: FusedOp) -> Result<kg::FusedScalarOp, PlanError> {
    Ok(match op {
        FusedOp::Binary(b) => kg::FusedScalarOp::Binary(map_binop(b)),
        FusedOp::Unary(GUnOp::Neg) => kg::FusedScalarOp::Unary(UnOp::Neg),
        FusedOp::Unary(GUnOp::Recip) => kg::FusedScalarOp::Recip,
        FusedOp::Unary(GUnOp::Exp) => kg::FusedScalarOp::Math(MathOp::Exp),
        FusedOp::Unary(GUnOp::Log) => kg::FusedScalarOp::Math(MathOp::Log),
        FusedOp::Unary(GUnOp::Sqrt) => kg::FusedScalarOp::Math(MathOp::Sqrt),
        FusedOp::Unary(GUnOp::Tanh) => kg::FusedScalarOp::Tanh,
        FusedOp::Unary(GUnOp::Erf) => kg::FusedScalarOp::Erf,
        // The widening itself is the leaf's lane decode: see `FusedScalarOp::Widen`.
        FusedOp::Cast { to: DType::F32 } => kg::FusedScalarOp::Widen,
        FusedOp::Unary(_) | FusedOp::Select | FusedOp::Cast { .. } => {
            return Err(site.refuse(Capability::FusedStep(FusedStepGap::OpInFloatRegion(op))));
        }
    })
}

/// The kernel op of one step of a pointwise region (`OpKind::Fused`): the one place `Select` is admitted
/// into a float region, whose I32 condition leaf only a pointwise kernel reads. A row or epilogue region
/// reads float lanes only and goes through [`map_fused_op`], which refuses it.
pub(crate) fn map_pointwise_op(
    site: Site<'_>,
    op: FusedOp,
    exact_i32: bool,
) -> Result<kg::FusedScalarOp, PlanError> {
    if exact_i32 {
        map_fused_op_i32(site, op)
    } else if op == FusedOp::Select {
        Ok(kg::FusedScalarOp::Select)
    } else {
        map_fused_op(site, op)
    }
}

pub(crate) fn map_fused_op_i32(
    site: Site<'_>,
    op: FusedOp,
) -> Result<kg::FusedScalarOp, PlanError> {
    Ok(match op {
        FusedOp::Binary(GBinOp::GeU) => kg::FusedScalarOp::GeU,
        FusedOp::Binary(GBinOp::RemU) => kg::FusedScalarOp::RemU,
        FusedOp::Binary(b) => kg::FusedScalarOp::Binary(map_binop(b)),
        FusedOp::Unary(GUnOp::Not) => kg::FusedScalarOp::Unary(UnOp::Not),
        FusedOp::Unary(GUnOp::Clz) => kg::FusedScalarOp::Clz,
        FusedOp::Select => kg::FusedScalarOp::Select,
        FusedOp::Unary(_) | FusedOp::Cast { .. } => {
            return Err(site.refuse(Capability::FusedStep(FusedStepGap::OpInExactRegion(op))));
        }
    })
}

/// Refuses a `Cast` step of a float region unless it widens one of the region's own inputs (`is_input`): the
/// lowering widens by reading the input through its narrow lane, so a `Cast` of a computed value has no
/// lowering.
pub(crate) fn check_widening_step(
    site: Site<'_>,
    op: FusedOp,
    inputs: &[FusedOperand],
    is_input: impl Fn(usize) -> bool,
) -> Result<(), PlanError> {
    match (op, inputs) {
        (FusedOp::Cast { .. }, [FusedOperand::Local(id)]) if is_input(*id) => Ok(()),
        (FusedOp::Cast { .. }, _) => {
            Err(site.refuse(Capability::FusedStep(FusedStepGap::OpInFloatRegion(op))))
        }
        _ => Ok(()),
    }
}

/// How region input `value`'s elements sit in its buffer, for a region that computes in f32: f32 as stored, or
/// the packed BF16 lane (`packed`, the storage plan's choice for this operand) a `Cast` step widens in
/// registers. A narrow input with no packed lane, or any other narrow dtype, has no body.
pub(crate) fn float_region_leaf_storage(
    site: Site<'_>,
    dtype: DType,
    packed: bool,
) -> Result<BufferStorage, PlanError> {
    match dtype {
        DType::BF16 if packed => Ok(BufferStorage::bf16_packed()),
        DType::BF16 | DType::F16 => Err(site.refuse(Capability::DtypeLowering)),
        _ => Ok(BufferStorage::f32()),
    }
}

pub(crate) fn map_binop(op: GBinOp) -> BinOp {
    match op {
        GBinOp::Add => BinOp::Add,
        GBinOp::Sub => BinOp::Sub,
        GBinOp::Mul => BinOp::Mul,
        GBinOp::Div => BinOp::Div,
        GBinOp::Max => BinOp::Max,
        GBinOp::Ge => BinOp::Ge,
        GBinOp::And => BinOp::BitAnd,
        GBinOp::Or => BinOp::BitOr,
        GBinOp::Xor => BinOp::BitXor,
        GBinOp::Shl => BinOp::Shl,
        GBinOp::Shr => BinOp::Shr,
        GBinOp::GeU => unreachable!("GeU lowers through the typed unsigned-I32 kernel path"),
        GBinOp::RemU => unreachable!("RemU lowers through the typed unsigned-I32 kernel path"),
    }
}

/// Map a graph [`GUnOp`] to the kernelgen unary operation of a float (or bf16/f16) kernel. One
/// definition per op, for the plain and the strided-view read alike: `Tanh` and `Erf` have bespoke
/// bodies with no view support, so `is_strided_capable` never lets a view reach them and `generate` refuses the combination if one ever does.
pub(crate) fn map_unary(site: Site<'_>, op: GUnOp) -> Result<UnaryOp, PlanError> {
    Ok(match op {
        GUnOp::Neg => UnaryOp::Basic(UnOp::Neg),
        GUnOp::Recip => UnaryOp::Recip,
        GUnOp::Sqrt => UnaryOp::Math(MathOp::Sqrt),
        GUnOp::Round => UnaryOp::Math(MathOp::Round),
        GUnOp::Exp => UnaryOp::Math(MathOp::Exp),
        GUnOp::Log => UnaryOp::Math(MathOp::Log),
        GUnOp::Tanh => UnaryOp::Tanh,
        GUnOp::Erf => UnaryOp::Erf,
        GUnOp::Not | GUnOp::Clz => {
            // I32-only primitives: they lower through the typed exact-integer kernels, never here.
            return Err(site.refuse(Capability::DtypeLowering));
        }
    })
}

/// The one finalizer: shape the launch of `planned`, then check every kernel it dispatches against
/// `caps`. Takes the plan by `&mut` so the launch-owning families run it at their own tail while the
/// dispatcher runs it after its match.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize(
    planned: &mut Planned,
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
) -> Result<(), PlanError> {
    shape_launch(planned, g, eqn, out_shape, out_numel, backend, odt, caps)?;
    validate_planned_resources(planned, g, eqn, out_numel, backend, caps)
}

/// What a choice's kernel declares beyond its body: a generated request declares its own; a shipped
/// template declares through [`ImportedKernel::needs`], and its serial work follows from the equation's
/// shapes (see [`SerialLoop`]). A split plan's
/// dispatches each carry their share (`chunks` of them, as the engine weighs submit batching).
fn choice_requirements(
    choice: &KernelChoice,
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_numel: usize,
    chunks: usize,
) -> KernelRequirements {
    match choice {
        KernelChoice::Generated(request) => request.requirements(),
        KernelChoice::Imported { kernel, .. } => {
            let needs = kernel.needs();
            let dim = |operand: usize, from_end: usize| match eqn.inputs.get(operand) {
                Some(Operand::Value(id)) => {
                    let shape = &g.aval(*id).shape;
                    shape.len().checked_sub(from_end).map(|i| shape[i])
                }
                _ => None,
            };
            let out_numel = out_numel as u64;
            let serial_work = match needs.serial {
                SerialLoop::None => None,
                SerialLoop::ContractionDepth => dim(0, 1).map(|k| (out_numel, k as u64)),
                SerialLoop::RowVocab { passes } => dim(0, 1).map(|vocab| {
                    // `[rows, 2]` output: one workgroup of lanes per row.
                    let lanes = (out_numel / 2) * SAMPLER_LANES;
                    (lanes, passes * (vocab as u64).div_ceil(SAMPLER_LANES))
                }),
            };
            KernelRequirements {
                subgroup_lanes: needs.subgroup_lanes,
                serial_work: serial_work.map(|(threads, steps_per_thread)| SerialWork {
                    threads: threads.div_ceil(chunks.max(1) as u64),
                    steps_per_thread,
                }),
            }
        }
        KernelChoice::NoDispatch(_) => KernelRequirements::default(),
    }
}

/// The workgroup shape of a single-dispatch plan.
fn single_workgroup(plan: &Plan) -> Option<[u32; 3]> {
    plan.single().map(|dispatch| dispatch.body.workgroup_size)
}

/// Check every kernel `planned` dispatches: its measured body against `caps`, and the requirements its
/// choice declares. Imported and generated kernels take the same path.
fn validate_planned_resources(
    planned: &Planned,
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_numel: usize,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<(), PlanError> {
    let refuse = |refusal| refusal_at(g, eqn, backend, Capability::KernelResources(refusal));
    let dispatches = planned.plan.dispatches();
    for dispatch in dispatches {
        validate_kernel_resources(
            &dispatch.body,
            choice_requirements(&planned.choice, g, eqn, out_numel, dispatches.len()),
            caps,
        )
        .map_err(refuse)?;
    }
    Ok(())
}

/// Shape the launch of a planned equation: the wgpu grid-cap workgroup bump, the elementwise 2-D fold
/// override, and the typed E4M3FN packed-dispatch ceiling check. A bump changes the body the plan's
/// key names, so it also changes the key.
#[allow(clippy::too_many_arguments)]
fn shape_launch(
    planned: &mut Planned,
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
) -> Result<(), PlanError> {
    let workgroup_before = single_workgroup(&planned.plan);
    let ids = value_ids(eqn);
    let has_e4m3fn_input = ids.iter().any(|&id| g.aval(id).dtype == DType::E4M3FN);
    // wgpu/Vulkan caps gridDim.x at 65535. A 1-thread-per-element kernel (grid = out_numel) at a large
    // output exceeds it, e.g. the Q=N prefill's attention scores [1,Hq,N,N] and the wide
    // matmuls/elementwise at long prompts, which made the wgpu prefill panic (`GridCap`) at N >= ~512.
    // Bump the baked workgroup size so gridDim = ceil(out_numel/wg) stays under the cap (the global
    // thread id is workgroup-size-aware, so a larger wg is correct; 256 is the portable Vulkan max and
    // covers outputs up to ~16.7M elems). FusedRow/Flash own their own wg+grid (row/head parallel), so
    // leave those untouched. NVPTX has no such cap, but the larger wg is harmless there.
    // The bump applies to the one-thread-per-output kernels (the default `out_numel` grid), whether
    // they dispatch as a single plain dispatch or as a metadata dispatch (the card-044
    // transpose/slice/broadcast layout swaps via the imported index_remap). It must not touch the ops
    // that size their own launch.
    //
    // Every contraction request owns its launch, by type: its generator builds the body
    // and the grid that shapes it together, and a Gemv, tiled or attention-GEMV body sizes its LDS
    // reduction or tile by its baked width, so a bump would corrupt it; the one-thread-per-output
    // contraction bodies run on `Fold`'s fixed-width workgroups. The rest are named below: the row- and
    // head-parallel ops and the shipped matmul bodies, whose launch is the template's.
    let contraction = matches!(
        &planned.choice,
        KernelChoice::Generated(KernelRequest::Contraction(_))
    );
    let plan = &mut planned.plan;
    let custom_grid = contraction
        || matches!(
            eqn.op,
            OpKind::FusedRow(_)
            | OpKind::Attention(_)
            | OpKind::Scan(_)
            // Card 551a: every SampleToken body is one fixed-64-lane workgroup per row (the LDS
            // reduce arrays are sized for exactly 64 lanes); the generic bump below would silently
            // corrupt that reduction by redispatching at workgroup_size[0] = 256.
            | OpKind::SampleToken { .. }
        )
        || is_decode_gemv_in(g, backend, eqn, out_shape, out_numel, caps)
        || is_tiled_gemm_in(g, backend, eqn, out_shape, out_numel)
        || is_batched_tiled_gemm_in(g, backend, eqn, out_shape, out_numel, caps)
        || is_elementwise_2d_in(g, backend, eqn, out_shape, out_numel, caps);
    if !contraction && is_elementwise_2d_in(g, backend, eqn, out_shape, out_numel, caps) {
        // card 159 Inc 3: the imported scatter_update/dyn_update_slice kernels bake their own
        // `x_groups` from `out.len()` assuming a fixed `WORKGROUP_SIZE = 256` (see
        // `is_elementwise_2d`), so the dispatch wg must always be 256, unconditionally (not the
        // conditional 64-or-256 bump below, which would leave small dispatches at the kernel's stock 64
        // and desync the two `x_groups` derivations once `out_numel` crosses the 65535*64 threshold at
        // wg=64 but not yet 65535*256 at wg=256).
        if let Some(dispatch) = plan.single_mut() {
            dispatch.body.workgroup_size[0] = ELEMENTWISE_2D_WIDTH as u32;
        }
    } else if !custom_grid && let Some(dispatch) = plan.single_mut() {
        let wg0 = dispatch.body.workgroup_size[0].max(1) as usize;
        if out_numel.div_ceil(wg0) > caps.max_grid[0] as usize {
            dispatch.body.workgroup_size[0] = 256;
        }
    }
    if single_workgroup(plan) != workgroup_before
        && let Some(Dispatch { body, key, .. }) = plan.single_mut()
    {
        // The choice digest names the body the generator built; the bump above made it a different
        // kernel on this device, so the key must say so (two devices with different grid caps must not
        // share a key for two different bodies).
        let [x, y, z] = body.workgroup_size;
        key.push_str(&format!("+wg{x}x{y}x{z}"));
    }
    // The E4M3FN pack and unpack casts use a one-dimensional global index. Unlike the movement bodies and
    // the dense elementwise kernels, they have no 2-D folded form for dispatches beyond the target's x-grid
    // (`caps.max_grid[0]`). Reject such a launch while planning, before compilation or buffer allocation. Use
    // the exact packed output word count for E4M3FN writers; decode owns one thread per logical f32 output.
    if backend == Backend::SpirvVulkan
        && (odt == DType::E4M3FN || has_e4m3fn_input)
        && let Some(Dispatch { body, launch, .. }) = plan.single()
        && launch.meta.is_empty()
        && launch.grid[1] == 1
    {
        let dispatch_threads = if odt == DType::E4M3FN {
            let row_len = out_shape.last().copied().unwrap_or(1);
            let rows = if out_shape.is_empty() {
                Some(1)
            } else {
                out_shape[..out_shape.len() - 1]
                    .iter()
                    .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            };
            rows.and_then(|rows| rows.checked_mul(row_len.div_ceil(4)))
        } else {
            Some(out_numel)
        };
        let Some(dispatch_threads) = dispatch_threads else {
            return Err(refusal_at(
                g,
                eqn,
                backend,
                Capability::DispatchSizeOverflow,
            ));
        };
        let workgroup_width = body.workgroup_size[0].max(1) as usize;
        let workgroups_x = dispatch_threads.div_ceil(workgroup_width);
        let limit = caps.max_grid[0] as usize;
        if workgroups_x > limit {
            return Err(refusal_at(
                g,
                eqn,
                backend,
                Capability::DispatchGridLimit {
                    threads: dispatch_threads,
                    workgroups_x,
                    workgroup_width,
                    limit,
                },
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{FUSABLE_FLOAT_UNARY_OPS, fuse};
    use poot_graph_ir::{Builder, OpKind, UnOp};
    use poot_target::{AmdArch, Backend};
    use poot_test_util::device_caps::default_caps_for;

    use crate::ExactI32StorageAnalysis;
    use crate::plan_eqn_analyzed;

    /// Card 536b: every unary `fuse::is_fusable` admits into an f32 pointwise region has a
    /// real planner lowering here, on every compute backend, or fusion legality and the kernel table
    /// have drifted apart again. Drives the real entry point (`plan_eqn` against a `fuse()`-produced
    /// region), not a direct `map_fused_op` call with a hand-built `Site`: the
    /// contract is "the planner can build this region," not "`map_fused_op` happens to return `Ok`".
    /// Mutation: drop one op from `FUSABLE_FLOAT_UNARY_OPS` (fuse.rs) without removing its
    /// `map_fused_op` arm (or the reverse); either way this loop's `unwrap` panics on that op's backend.
    #[test]
    fn fused_region_kernel_table_covers_every_fusable_unary() {
        let backends = [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ];
        for &op in FUSABLE_FLOAT_UNARY_OPS {
            // A two-step chain (`Neg` then `op`) so `fuse()` forms an actual `OpKind::Fused` region
            // (a lone unary stays unfused - fusion needs >= 2 members).
            let b = Builder::new();
            let x = b.constant("x", poot_graph_ir::TensorType::f32(vec![4]));
            let n = b.unary(UnOp::Neg, x);
            let y = b.unary(op, n);
            let graph = b.finish(y);
            let fused = fuse(&graph);
            let eqn = fused
                .eqns
                .iter()
                .find(|e| matches!(e.op, OpKind::Fused(_)))
                .unwrap_or_else(|| panic!("{op:?} did not fuse into a region: {fused:?}"));
            let analysis = ExactI32StorageAnalysis::new(&fused);
            for backend in backends {
                plan_eqn_analyzed(
                    &analysis,
                    &fused,
                    eqn,
                    backend,
                    &default_caps_for(backend),
                    &poot_test_util::graph_fixtures::roomy_body_limits(),
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "fuse.rs admits {op:?} into a pointwise region but map_fused_op refuses it \
                             on {backend:?}: {e}"
                    )
                });
            }
        }
    }
}
