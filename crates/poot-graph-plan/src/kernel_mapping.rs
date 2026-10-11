//! Mapping graph-level fused operations to kernel-IR operations, and the one finalizer every planned
//! kernel passes through before it can run.
//!
//! [`finalize`] shapes the launch (workgroup bump, 2-D fold, the plan key that names it) and then proves the kernel fits the device:
//! generated and imported kernels alike are checked against the measured [`DeviceCaps`] by
//! [`validate_kernel_resources`], so no plan that cannot run reaches a loader.

use poot_kernelgen::{KernelRequirements, SerialWork};
use poot_runtime_common::{
    AieArtifactError, AieBinding, AieLinkedArtifact, AieShapeAbi, ArgAccess,
};
use poot_target::{AieCaps, AieTargetId, Backend, BufferStorage, DeviceCaps, Queried};

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

/// A region-local operand of an exact-I32 step inside a float region: a local, or an I32 literal. A packed
/// lane belongs to an exact-I32 region's own store, and an f32 literal has no I32 meaning.
pub(crate) fn map_mixed_input_i32(
    site: Site<'_>,
    o: &FusedOperand,
    n_inputs: usize,
) -> Result<kg::FusedInput, PlanError> {
    match o {
        FusedOperand::PackLane { .. } => {
            Err(site.refuse(Capability::FusedStep(FusedStepGap::PackedLaneInFloatRegion)))
        }
        other => map_fused_input_i32(site, other, n_inputs),
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
/// into a float region, whose I32 condition leaf only a pointwise kernel reads, and where a `Cast` of an I32
/// value to f32 is a step (`local_dtype` is the region's type of a local). A row or epilogue region reads
/// float lanes only and goes through [`map_fused_op`], which refuses both. Any other `Cast` has no pointwise
/// lowering: the kernels read every leaf as f32, so a narrow leaf has no lane to widen.
pub(crate) fn map_pointwise_step(
    site: Site<'_>,
    op: FusedOp,
    inputs: &[FusedOperand],
    exact_i32: bool,
    local_dtype: impl Fn(usize) -> DType,
) -> Result<kg::FusedScalarOp, PlanError> {
    match (op, inputs) {
        (FusedOp::Cast { to: DType::F32 }, [FusedOperand::Local(id)])
            if local_dtype(*id) == DType::I32 =>
        {
            Ok(kg::FusedScalarOp::I32ToF32)
        }
        (FusedOp::Cast { .. }, _) => {
            Err(site.refuse(Capability::FusedStep(FusedStepGap::OpInFloatRegion(op))))
        }
        _ if exact_i32 => map_fused_op_i32(site, op),
        (FusedOp::Select, _) => Ok(kg::FusedScalarOp::Select),
        _ => map_fused_op(site, op),
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
    check_operand_bindings(planned, eqn)?;
    // An AIE core runs one sequential loop. Its schedule was chosen at generation (the request is the
    // ordinary one); the GPU launch shaping and the device resource finalizer do not apply, and what does
    // is checked here, by type: the plan must be a generated, grid-free, workgroup-memory-free body.
    if let Backend::Aie(_) = backend {
        return validate_aie_schedule(planned, g, eqn, backend);
    }
    shape_launch(planned, g, eqn, out_shape, out_numel, backend, odt, caps)?;
    validate_planned_resources(planned, g, eqn, out_numel, backend, caps)
}

/// Every operand a dispatch binds is one the equation has. [`DispatchPlan::new`] cannot know the
/// equation, so the one place both are in hand checks the binding against it.
fn check_operand_bindings(planned: &Planned, eqn: &Eqn) -> Result<(), PlanError> {
    let operands = value_ids(eqn).len();
    for (dispatch, plan_dispatch) in planned.plan.dispatches().iter().enumerate() {
        for source in plan_dispatch.binding().reads() {
            if let Source::Operand(operand) = *source
                && operand >= operands
            {
                return Err(DispatchPlanError::OperandOutOfRange {
                    dispatch,
                    operand,
                    operands,
                }
                .into());
            }
        }
    }
    Ok(())
}

/// Check that `planned` is an AIE sequential schedule: every dispatch comes from a [`KernelChoice::Sequential`]
/// (the one body-and-bindings generation the target's schedule produces). A shipped GPU template, or a
/// request generated as a thread grid, has no AIE form. Plans that launch nothing (an alias, a view) pass.
fn validate_aie_schedule(
    planned: &Planned,
    g: GraphTables<'_>,
    eqn: &Eqn,
    backend: Backend,
) -> Result<(), PlanError> {
    match planned.choice {
        KernelChoice::Sequential { .. } | KernelChoice::NoDispatch(_) => Ok(()),
        KernelChoice::Generated(_) | KernelChoice::Imported { .. } => {
            Err(refusal_at(g, eqn, backend, Capability::AieSchedule))
        }
    }
}

/// Why an AIE limit check refused: the limit that is unknown, or the need and the offer.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-641"))]
pub(crate) enum AieLimitRefusal {
    /// The artifact was linked for another target than the limits describe.
    #[error("the artifact was linked for {artifact:?}, the limits describe {caps:?}")]
    Target {
        artifact: AieTargetId,
        caps: AieTargetId,
    },
    /// The target's limit is not measured, so the artifact's need of it cannot be admitted.
    #[error("the target's {limit} is not measured")]
    Unknown { limit: &'static str },
    #[error("the artifact needs {needed} {resource}; the target offers {available}")]
    Exceeded {
        resource: &'static str,
        needed: u64,
        available: u64,
    },
}

fn within<T: Into<u64> + Copy>(
    limit: Queried<T>,
    resource: &'static str,
    needed: u64,
) -> Result<(), AieLimitRefusal> {
    match limit {
        Queried::Unknown => Err(AieLimitRefusal::Unknown { limit: resource }),
        Queried::Known(available) if needed > available.into() => Err(AieLimitRefusal::Exceeded {
            resource,
            needed,
            available: available.into(),
        }),
        Queried::Known(_) => Ok(()),
    }
}

/// The finalization only a linked artifact can answer, run before it is loaded: its tile-local
/// reservation, host bindings and instruction stream against `caps`. An unmeasured limit refuses.
///
/// The ordinary planner entry (`plan_eqn`) cannot run this at plan time: its target caps are the GPU-shaped
/// `DeviceCaps`, which has no AIE limits (the tagged carrier of spec 1002 is the follow-up).
#[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-641"))]
pub(crate) fn validate_linked_artifact(
    artifact: &AieLinkedArtifact,
    caps: &AieCaps,
) -> Result<(), AieLimitRefusal> {
    if artifact.target() != caps.target {
        return Err(AieLimitRefusal::Target {
            artifact: artifact.target(),
            caps: caps.target,
        });
    }
    within(
        caps.tile_local_bytes,
        "tile-local bytes",
        artifact.tile_local_bytes(),
    )?;
    within(
        caps.host_bindings,
        "host bindings",
        artifact.abi().bindings().len() as u64,
    )?;
    within(
        caps.instruction_bytes,
        "instruction bytes",
        artifact.instructions().len() as u64,
    )
}

/// The shape ABI of a planned sequential schedule: one binding per body parameter, in binding order,
/// each with the dimensions and access range the generator proved when it produced the body. The ABI is
/// read from the plan's own choice, so the body and the ABI come from the one generation the planner ran.
#[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-641"))]
pub(crate) fn aie_shape_abi(choice: &KernelChoice) -> Result<AieShapeAbi, AieAbiRefusal> {
    let KernelChoice::Sequential { bindings, .. } = choice else {
        return Err(AieAbiRefusal::NotSequential);
    };
    let bindings = bindings
        .iter()
        .enumerate()
        .map(|(index, binding)| {
            AieBinding::new(
                index,
                BufferStorage::f32(),
                if binding.writes {
                    ArgAccess::Write
                } else {
                    ArgAccess::Read
                },
                &binding.dims,
                binding.accessed,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AieShapeAbi::new(bindings)?)
}

/// Why a plan has no AIE shape ABI.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-641"))]
pub(crate) enum AieAbiRefusal {
    #[error("the plan's choice is not an AIE sequential schedule")]
    NotSequential,
    #[error(transparent)]
    Abi(#[from] AieArtifactError),
}

/// What a choice's kernel declares beyond its body: a generated request declares its own; a shipped
/// template declares through [`ImportedKernel::needs`], and its serial work follows from the equation's
/// shapes (see [`SerialLoop`]). `share` is the fraction of the equation's output this dispatch covers
/// (`numerator / denominator`): a split plan's dispatches each carry their own launch's share.
fn choice_requirements(
    choice: &KernelChoice,
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_numel: usize,
    share: (u128, u128),
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
                    // `share.0 <= share.1`, so the apportioned count never exceeds `threads`.
                    threads: (u128::from(threads) * share.0).div_ceil(share.1.max(1)) as u64,
                    steps_per_thread,
                }),
            }
        }
        // An AIE core has no workgroup, subgroup or dispatch-work budget: its limits are `AieCaps`, checked
        // on the linked artifact (`validate_linked_artifact`).
        KernelChoice::NoDispatch(_) | KernelChoice::Sequential { .. } => {
            KernelRequirements::default()
        }
    }
}

/// The total thread count of one launch. `u128`: three `u32` extents cannot overflow it.
fn launch_threads(launch: &Launch) -> u128 {
    launch
        .grid
        .iter()
        .map(|&extent| u128::from(extent))
        .product()
}

/// The workgroup shape of a single-dispatch plan.
fn single_workgroup(plan: &Plan) -> Option<[u32; 3]> {
    plan.single().map(|dispatch| dispatch.body().workgroup_size)
}

/// Check every kernel `planned` dispatches: its measured body against `caps`, and the requirements its own
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
    // A dispatch covers the part of the output its launch's thread count is of the plan's total.
    let total_threads: u128 = dispatches.iter().map(|d| launch_threads(d.launch())).sum();
    for dispatch in dispatches {
        // A zero-thread plan covers its output whole (nothing to apportion).
        let share = match total_threads {
            0 => (1, 1),
            total => (launch_threads(dispatch.launch()), total),
        };
        validate_kernel_resources(
            dispatch.body(),
            choice_requirements(dispatch.choice(), g, eqn, out_numel, share),
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
            dispatch.set_workgroup_x(ELEMENTWISE_2D_WIDTH as u32);
        }
    } else if !custom_grid && let Some(dispatch) = plan.single_mut() {
        let wg0 = dispatch.body().workgroup_size[0].max(1) as usize;
        if out_numel.div_ceil(wg0) > caps.max_grid[0] as usize {
            dispatch.set_workgroup_x(256);
        }
    }
    if single_workgroup(plan) != workgroup_before
        && let Some(dispatch) = plan.single_mut()
    {
        // The choice digest names the body the generator built; the bump above made it a different
        // kernel on this device, so the key must say so (two devices with different grid caps must not
        // share a key for two different bodies).
        let [x, y, z] = dispatch.body().workgroup_size;
        dispatch.append_key(&format!("+wg{x}x{y}x{z}"));
    }
    // The E4M3FN pack and unpack casts use a one-dimensional global index. Unlike the movement bodies and
    // the dense elementwise kernels, they have no 2-D folded form for dispatches beyond the target's x-grid
    // (`caps.max_grid[0]`). Reject such a launch while planning, before compilation or buffer allocation. Use
    // the exact packed output word count for E4M3FN writers; decode owns one thread per logical f32 output.
    if backend == Backend::SpirvVulkan
        && (odt == DType::E4M3FN || has_e4m3fn_input)
        && let Some(dispatch) = plan.single()
        && dispatch.launch().meta.is_empty()
        && dispatch.launch().grid[1] == 1
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
        let workgroup_width = dispatch.body().workgroup_size[0].max(1) as usize;
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

    use crate::plan_eqn;

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
            for backend in backends {
                plan_eqn(
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

    // ---- POOT-664: the AIE sequential schedule is a choice inside the ordinary planner -----------------

    use std::collections::HashMap;
    use std::num::NonZeroU64;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};

    use poot_codegen::{
        AieCompileError, KernelCache, Target as CodegenTarget, Toolchain, aie_kernel_handle,
        compile_aie,
    };
    use poot_graph_ir::op::RedOp;
    use poot_graph_ir::{Graph, TensorType};
    use poot_runtime_common::{AieBindError, AieBound, check_aie_bindings};
    use poot_target::{AieCaps, AieTargetId, DeviceCaps, Queried};
    use poot_tensor::DType;

    use crate::{Capability, PlanError};

    use super::{AieLimitRefusal, aie_shape_abi, validate_linked_artifact};

    const AIE: Backend = Backend::Aie(AieTargetId::Xdna2);

    fn aie_caps() -> AieCaps {
        // SYNTHETIC limits: roomy enough for a [2,3] kernel, not a device measurement.
        AieCaps::synthetic(AieTargetId::Xdna2, 1 << 16, 8, 1 << 20)
    }

    fn limits() -> poot_kernelgen::BodyLimits {
        poot_test_util::graph_fixtures::roomy_body_limits()
    }

    fn graph_of(build: impl FnOnce(&Builder) -> poot_graph_ir::Traced) -> Graph {
        let b = Builder::new();
        let out = build(&b);
        b.finish(out)
    }

    fn add_graph(a_shape: Vec<usize>, b_shape: Vec<usize>, dtype: DType) -> Graph {
        graph_of(|b| {
            let x = b.constant("a", TensorType::new(a_shape, dtype));
            let y = b.constant("b", TensorType::new(b_shape, dtype));
            b.binary(poot_graph_ir::BinOp::Add, x, y)
        })
    }

    fn last_eqn(g: &Graph) -> &poot_graph_ir::Eqn {
        g.eqns.last().expect("the graph has an equation")
    }

    /// The ordinary planner entry for an AIE target. The `DeviceCaps` it takes are GPU limits the AIE
    /// schedule never reads (the tagged carrier of spec 1002 is the follow-up), so a fixture value is passed.
    fn plan_aie(g: &Graph) -> Result<crate::Planned, PlanError> {
        crate::plan_eqn_choice(
            g,
            last_eqn(g),
            AIE,
            &HashMap::new(),
            &DeviceCaps::wgpu_rdna3_igpu(),
            &limits(),
        )
    }

    fn aie_refusal(g: &Graph) -> Capability {
        match plan_aie(g) {
            Err(PlanError::Refused(refusal)) => {
                assert_eq!(refusal.target, AIE);
                refusal.missing
            }
            Err(other) => panic!("expected a typed refusal, got {other:?}"),
            Ok(_) => panic!("the equation must be refused for an AIE core"),
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("poot-aie-plan-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn script(dir: &Path, name: &str, body: &str) -> String {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// Scripted Peano and IRON drivers (the compiler's protocol handling, not machine code).
    fn scripted_toolchain(dir: &Path, with_iron: bool) -> Toolchain {
        let peano = script(
            dir,
            "peano",
            r#"case "$1" in --version) echo "fake-peano (aie2p)"; exit 0;; esac
for last; do :; done
printf '\177ELF' > "$last"; printf 'poot_seq_binary' >> "$last""#,
        );
        let iron = script(
            dir,
            "iron",
            r#"case "$1" in --version) echo "fake-iron"; exit 0;; --platforms) echo npu2; exit 0;; esac
while [ $# -gt 0 ]; do case "$1" in --xclbin) x="$2";; --insts) i="$2";; esac; shift; done
printf 'xclbin2\0payload' > "$x"; printf 'IIII' > "$i""#,
        );
        Toolchain::from_env_with(move |name| match name {
            "POOT_AIE_LLC" => Some(peano.clone()),
            "POOT_AIE_IRON" => with_iron.then(|| iron.clone()),
            _ => None,
        })
    }

    /// The planner positive: `plan_eqn` itself plans an F32 `[2,3]` add for an AIE target, as the ordinary
    /// elementwise request under the sequential schedule on the existing plan types (one dispatch, one
    /// invocation, a one-thread workgroup). The compiler links its body against the ABI derived from that
    /// request, the ordinary handle wraps it, and a fake loader sees three ordered arguments and core
    /// lengths `[6, 6, 6]`. A fake loader does not claim numerical execution or hardware.
    /// Mutation: restore the `Backend::Aie` refusal in `finalize` or give the AIE arm of `build` a GPU
    /// grid; the plan or the dispatch-shape assertion fails.
    #[test]
    fn plan_eqn_plans_an_f32_add_under_the_aie_sequential_schedule_and_it_links() {
        let g = add_graph(vec![2, 3], vec![2, 3], DType::F32);
        let planned = plan_aie(&g).unwrap();
        let dispatch = planned.plan.single().expect("one dispatch");
        assert_eq!(dispatch.body.workgroup_size, [1, 1, 1]);
        assert_eq!(dispatch.launch.grid, [1, 1, 1]);
        assert!(dispatch.launch.meta.is_empty());

        let abi = aie_shape_abi(&planned.choice).unwrap();
        assert_eq!(abi.core_lengths(), vec![6, 6, 6]);
        // The ABI is read from the plan's own generation: the dimensions are the graph's, in order.
        for binding in abi.bindings() {
            assert_eq!(binding.dims(), &[2, 3]);
        }

        let dir = scratch("positive");
        let tc = scripted_toolchain(&dir, true);
        let cache = KernelCache::at(dir.join("cache"), CodegenTarget::AieCore);
        let compiled = compile_aie(
            &dispatch.body,
            &abi,
            AieTargetId::Xdna2,
            &tc,
            &cache,
            NonZeroU64::new(1 << 20).unwrap(),
        )
        .unwrap();
        validate_linked_artifact(&compiled.artifact, &aie_caps()).unwrap();
        let artifact = compiled.artifact.clone();
        let handle = aie_kernel_handle(&dispatch.body, compiled.artifact).unwrap();
        assert_eq!(handle.target(), AIE);
        assert_eq!(handle.args().len(), 3);

        let dims = [2u64, 3];
        let bound = |writable| AieBound {
            storage: poot_target::BufferStorage::f32(),
            dims: &dims,
            writable,
        };
        assert_eq!(
            check_aie_bindings(&artifact, &[bound(false), bound(false), bound(true)]),
            Ok(())
        );
        assert_eq!(
            check_aie_bindings(&artifact, &[bound(true), bound(false), bound(false)]),
            Err(AieBindError::ReadOnlyWritten { index: 2 })
        );
    }

    /// The same ordinary request path refuses what has no sequential schedule, each with the planner's own
    /// typed refusal: a family the generator has no loop for, a broadcast operand, a non-F32 element, an
    /// op not proved on the core. Mutation: make `sequential_schedule` fall back to the GPU generator, or
    /// drop its operand-shape check; the matching row returns `Ok`.
    #[test]
    fn plan_eqn_refuses_for_aie_what_has_no_sequential_schedule() {
        let kernel_gen = |capability: Capability| match capability {
            Capability::KernelGen(poot_kernelgen::KernelGenError::Unsupported {
                request,
                reason,
            }) => (request, reason),
            other => panic!("expected a generator refusal, got {other:?}"),
        };
        let neg = graph_of(|b| {
            let x = b.constant("x", TensorType::f32(vec![4]));
            b.unary(poot_graph_ir::UnOp::Neg, x)
        });
        assert_eq!(kernel_gen(aie_refusal(&neg)).0, "elementwise");
        let broadcast = add_graph(vec![2, 3], vec![1, 3], DType::F32);
        assert!(kernel_gen(aie_refusal(&broadcast)).1.contains("broadcast"));
        let ints = add_graph(vec![2, 3], vec![2, 3], DType::I32);
        assert_eq!(kernel_gen(aie_refusal(&ints)).0, "elementwise");
        let max = graph_of(|b| {
            let x = b.constant("x", TensorType::f32(vec![4]));
            let y = b.constant("y", TensorType::f32(vec![4]));
            b.binary(poot_graph_ir::BinOp::Max, x, y)
        });
        assert_eq!(kernel_gen(aie_refusal(&max)).0, "sequential_binary");
        // A collective is a cut between stage programs on every target.
        let reduce = graph_of(|b| {
            let x = b.constant("x", TensorType::f32(vec![2, 3]));
            b.all_reduce(x, RedOp::Sum, 0)
        });
        assert!(matches!(
            aie_refusal(&reduce),
            Capability::CollectiveInStage
        ));
    }

    /// A shipped GPU template (an imported kernel) reaches finalization and is refused by type there.
    /// Mutation: make `validate_aie_schedule` return `Ok(())`; the template is planned for an AIE core.
    #[test]
    fn an_imported_gpu_template_is_not_an_aie_schedule() {
        let g = graph_of(|b| {
            let seed = b.constant("seed", TensorType::new(vec![2], DType::I32));
            b.random_uniform(seed, 4)
        });
        assert_eq!(aie_refusal(&g), Capability::AieSchedule);
    }

    /// Finalization of the linked artifact reads the target's limits: an unmeasured limit refuses (never
    /// a guess), and a reservation past a measured one refuses with both numbers. Mutation: treat
    /// `Unknown` as unbounded, or drop the `needed > available` comparison.
    #[test]
    fn a_linked_artifact_is_refused_against_unknown_and_exceeded_target_limits() {
        let add = add_graph(vec![2, 3], vec![2, 3], DType::F32);
        let planned = plan_aie(&add).unwrap();
        let body = planned.plan.single().unwrap().body.clone();
        let abi = aie_shape_abi(&planned.choice).unwrap();
        let dir = scratch("limits");
        let compiled = compile_aie(
            &body,
            &abi,
            AieTargetId::Xdna2,
            &scripted_toolchain(&dir, true),
            &KernelCache::at(dir.join("cache"), CodegenTarget::AieCore),
            NonZeroU64::new(1 << 20).unwrap(),
        )
        .unwrap();
        let artifact = &compiled.artifact;
        assert_eq!(validate_linked_artifact(artifact, &aie_caps()), Ok(()));
        let unmeasured = AieCaps::unmeasured(AieTargetId::Xdna2);
        assert_eq!(
            validate_linked_artifact(artifact, &unmeasured),
            Err(AieLimitRefusal::Unknown {
                limit: "tile-local bytes"
            })
        );
        // Three FIFOs of 6 words at depth 2.
        let tiny = AieCaps {
            tile_local_bytes: Queried::Known(143),
            ..aie_caps()
        };
        assert_eq!(
            validate_linked_artifact(artifact, &tiny),
            Err(AieLimitRefusal::Exceeded {
                resource: "tile-local bytes",
                needed: 144,
                available: 143
            })
        );
        let two_bindings = AieCaps {
            host_bindings: Queried::Known(2),
            ..aie_caps()
        };
        assert_eq!(
            validate_linked_artifact(artifact, &two_bindings),
            Err(AieLimitRefusal::Exceeded {
                resource: "host bindings",
                needed: 3,
                available: 2
            })
        );
        let no_room = AieCaps {
            instruction_bytes: Queried::Known(3),
            ..aie_caps()
        };
        assert_eq!(
            validate_linked_artifact(artifact, &no_room),
            Err(AieLimitRefusal::Exceeded {
                resource: "instruction bytes",
                needed: 4,
                available: 3
            })
        );
    }

    /// A planned AIE kernel without the IRON driver fails the ordinary compile with the named stage and no
    /// entry: the planner admitted the plan, the toolchain boundary is the compiler's.
    #[test]
    fn a_planned_aie_kernel_without_the_iron_driver_is_a_typed_compile_error() {
        let add = add_graph(vec![2, 3], vec![2, 3], DType::F32);
        let planned = plan_aie(&add).unwrap();
        let body = planned.plan.single().unwrap().body.clone();
        let abi = aie_shape_abi(&planned.choice).unwrap();
        let dir = scratch("no-iron");
        let cache = KernelCache::at(dir.join("cache"), CodegenTarget::AieCore);
        let error = compile_aie(
            &body,
            &abi,
            AieTargetId::Xdna2,
            &scripted_toolchain(&dir, false),
            &cache,
            NonZeroU64::new(1 << 20).unwrap(),
        )
        .map(|_| ())
        .unwrap_err();
        assert!(
            matches!(
                error,
                AieCompileError::ToolchainMissing {
                    stage: poot_codegen::AieStage::Iron,
                    ..
                }
            ),
            "got {error:?}"
        );
        assert!(!cache.dir().exists() || std::fs::read_dir(cache.dir()).unwrap().next().is_none());
    }
}
