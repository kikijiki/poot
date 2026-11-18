//! Mapping graph-level fused operations to kernel-IR operations, and the one finalizer every planned
//! kernel passes through before it can run.
//!
//! [`finalize`] checks that every dispatch is the kernel its choice names at the launch its request declared, and then proves the kernel fits the device:
//! generated and imported kernels alike are checked against the measured [`DeviceCaps`] by
//! [`validate_kernel_resources`], so no plan that cannot run reaches a loader.

use crate::planner::Planned;
use crate::predicates::GraphTables;
use crate::refusal::{FusedStepGap, Site, refusal_at};
use crate::types::DispatchPlanError;
use poot_kernelgen::KernelRequirements;
use poot_target::{AieCaps, AieTargetId, Backend, BufferStorage, DeviceCaps};

use crate::device_validation::validate_kernel_resources;
use crate::npu::NpuLowering;
use crate::{Capability, KernelChoice, Plan, PlanError, Source, value_ids};
use poot_graph_ir::Eqn;
use poot_graph_ir::Scalar;
use poot_graph_ir::op::BinOp as GBinOp;
use poot_graph_ir::op::FusedOp;
use poot_graph_ir::op::FusedOperand;
use poot_graph_ir::op::LocalId;
use poot_graph_ir::op::UnOp as GUnOp;
use poot_kernel_ir::BinOp;
use poot_kernel_ir::MathOp;
use poot_kernel_ir::UnOp;
use poot_kernelgen as kg;
use poot_tensor::DType;

/// A region-local id -> a kernel value: `< n_inputs` is a leaf, otherwise a prior step.
pub(crate) fn map_row_value(id: LocalId, n_inputs: usize) -> kg::RowValue {
    let id = id.index();
    if id < n_inputs {
        kg::RowValue::Leaf(kg::LeafId::new(id))
    } else {
        kg::RowValue::Step(kg::StepId::new(id - n_inputs))
    }
}

/// A region-local id -> a fused-kernel operand: `< n_inputs` is a leaf, otherwise a prior step.
pub(crate) fn map_fused_local(id: LocalId, n_inputs: usize) -> kg::FusedInput {
    map_row_value(id, n_inputs).into()
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
            leaf: kg::LeafId::new(input.index()),
            lane: kg::LaneId::new(lane.index()),
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

/// The kernel op of a float region's last step when it narrows the region's f32 result to the two-byte float `to`
/// (POOT-1448, [`poot_graph_ir::FusedRegion::narrowed_output`]): [`kg::FusedScalarOp::Narrow`], whose body rounds the
/// value once and stores it narrow. `stores` says whether the body the region runs in has that store: a target
/// with native two-byte elements, and a body that writes one. Without it the step is refused by name
/// ([`Capability::DtypeLowering`]), so a value the planner cannot store narrow stays f32.
pub(crate) fn narrowing_store_op(
    site: Site<'_>,
    to: DType,
    stores: bool,
) -> Result<kg::FusedScalarOp, PlanError> {
    let dtype = match to {
        DType::BF16 => poot_target::FragmentDtype::Bf16,
        DType::F16 => poot_target::FragmentDtype::F16,
        _ => return Err(site.refuse(Capability::DtypeLowering)),
    };
    if stores {
        Ok(kg::FusedScalarOp::Narrow(dtype))
    } else {
        Err(site.refuse(Capability::DtypeLowering))
    }
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
    local_dtype: impl Fn(LocalId) -> DType,
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
    is_input: impl Fn(LocalId) -> bool,
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

/// The kernelgen step of a float (or two-byte float) unary op. `Not` and `Clz` are I32-only primitives: they
/// lower through the exact-integer step ([`map_fused_op_i32`]), never here.
pub(crate) fn map_unary(site: Site<'_>, op: GUnOp) -> Result<kg::FusedScalarOp, PlanError> {
    match op {
        GUnOp::Not | GUnOp::Clz => Err(site.refuse(Capability::DtypeLowering)),
        op => map_fused_op(site, FusedOp::Unary(op)),
    }
}

/// The one GPU finalizer: check the launch of `planned` is the one its choice declared, then check every kernel it dispatches against
/// `caps`. Takes the plan by `&mut` so the launch-owning families run it at their own tail while the
/// dispatcher runs it after its match. An AIE target finalizes through [`finalize_aie`].
#[expect(clippy::too_many_arguments)]
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
    check_launch_ownership(planned, backend)?;
    check_e4m3_dispatch_ceiling(planned, g, eqn, out_shape, out_numel, backend, odt, caps)?;
    validate_planned_resources(planned, g, eqn, backend, caps)
}

/// The AIE finalizer. An AIE core runs one sequential loop. Its schedule was chosen at generation (the
/// request is the ordinary one); the GPU launch shaping and the device resource finalizer do not apply, and
/// what does is checked here, by type: the plan must be a serialized, grid-free, workgroup-memory-free
/// schedule, and the shape ABI it links against, which the NPU's `npu` lowering derives, must fit the
/// target's measured limits (an unmeasured limit refuses). The limits only the linked artifact can answer
/// are the loader's, before it wraps the artifact.
pub(crate) fn finalize_aie(
    planned: &Planned,
    g: GraphTables<'_>,
    eqn: &Eqn,
    target: AieTargetId,
    caps: &AieCaps,
    npu: &dyn NpuLowering,
) -> Result<(), PlanError> {
    check_operand_bindings(planned, eqn)?;
    let refuse = |missing| refusal_at(g, eqn, target, missing);
    match planned.choice {
        KernelChoice::NoDispatch(_) => Ok(()),
        KernelChoice::Serialized { .. } => npu
            .finalize_serialized(&planned.choice, caps)
            .map_err(refuse),
        KernelChoice::Generated(_) | KernelChoice::Imported { .. } => {
            Err(refuse(Capability::AieSchedule))
        }
    }
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

/// What a choice's kernel declares beyond its body: a generated request declares its own; a shipped
/// template declares nothing, because every resource it uses is measured from its body.
fn choice_requirements(choice: &KernelChoice) -> KernelRequirements {
    match choice {
        KernelChoice::Generated(request) => request.requirements(),
        // An AIE core has no workgroup or subgroup budget: its limits are `AieCaps`, checked on the linked
        // artifact (the NPU loader's check).
        KernelChoice::Imported { .. }
        | KernelChoice::NoDispatch(_)
        | KernelChoice::Serialized { .. } => KernelRequirements::default(),
    }
}

/// Check every kernel `planned` dispatches: its measured body against `caps`, and the requirements its own
/// choice declares. Imported and generated kernels take the same path.
fn validate_planned_resources(
    planned: &Planned,
    g: GraphTables<'_>,
    eqn: &Eqn,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<(), PlanError> {
    let refuse = |refusal| refusal_at(g, eqn, backend, Capability::KernelResources(refusal));
    // The backstop of the planner's own fit check (split-KV attention falls back to streaming when its
    // scratch does not fit): a plan that declares a scratch the device cannot allocate is refused here.
    if let Plan::Dispatches(plan) = &planned.plan {
        for decl in plan.scratch() {
            if decl.bytes() > caps.max_buffer_bytes {
                return Err(refusal_at(
                    g,
                    eqn,
                    backend,
                    Capability::ScratchBytes {
                        needed: decl.bytes(),
                        max_buffer_bytes: caps.max_buffer_bytes,
                    },
                ));
            }
        }
    }
    for dispatch in planned.plan.dispatches() {
        validate_kernel_resources(
            dispatch.body(),
            choice_requirements(dispatch.choice()),
            caps,
        )
        .map_err(refuse)?;
    }
    Ok(())
}

/// A planned dispatch is the kernel its choice names, at the launch its request declared: its key is the
/// choice's digest (a body narrowed to 32-bit indices says so with `:u32`), and an elementwise kernel's
/// workgroup is the width the choice declared before the body was generated. Nothing reshapes a launch after
/// planning: a body widened, or a key suffixed, behind the request's back is two devices sharing a key for two
/// different kernels, and is refused here.
fn check_launch_ownership(planned: &Planned, backend: Backend) -> Result<(), PlanError> {
    for (dispatch, planned_dispatch) in planned.plan.dispatches().iter().enumerate() {
        let choice = planned_dispatch.choice();
        let named = choice.key(backend.into());
        let narrowed = matches!(
            planned_dispatch.body().index_width,
            poot_kernel_ir::IndexWidth::U32(_)
        );
        let expected = if narrowed {
            format!("{named}:u32")
        } else {
            named
        };
        if planned_dispatch.key() != expected {
            return Err(DispatchPlanError::KeyNotChoice {
                dispatch,
                key: planned_dispatch.key().to_owned(),
                expected,
            }
            .into());
        }
        if let Some(declared) = choice.elementwise_workgroup_x() {
            let actual = planned_dispatch.body().workgroup_size[0];
            if actual != declared {
                return Err(DispatchPlanError::WorkgroupNotDeclared {
                    dispatch,
                    declared,
                    actual,
                }
                .into());
            }
        }
    }
    Ok(())
}

/// The typed E4M3FN packed-dispatch ceiling check.
#[expect(clippy::too_many_arguments)]
fn check_e4m3_dispatch_ceiling(
    planned: &Planned,
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
) -> Result<(), PlanError> {
    let plan = &planned.plan;
    let ids = value_ids(eqn);
    let has_e4m3fn_input = ids.iter().any(|&id| g.aval(id).dtype == DType::E4M3FN);
    // The E4M3FN pack and unpack casts use a one-dimensional global index. Unlike the movement bodies and
    // the dense elementwise kernels, they have no 2-D folded form for dispatches beyond the target's x-grid
    // (`caps.max_grid[0]`). Reject such a launch while planning, before compilation or buffer allocation. Use
    // the exact packed output word count for E4M3FN writers; decode owns one thread per logical f32 output.
    if caps.e4m3_storage == poot_target::E4m3Storage::PackedLanes
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
    use crate::fuse;
    use crate::fuse::FUSABLE_FLOAT_UNARY_OPS;
    use poot_graph_ir::{Builder, Eqn, Graph, OpKind, TensorType, UnOp};
    use poot_target::{AieCaps, AieTargetId, AmdArch, Backend};
    use poot_test_util::device_caps::default_caps_for;

    use super::*;
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
            let b = Builder::for_tests();
            let x = b
                .constant("x", poot_graph_ir::TensorType::f32(vec![4]))
                .unwrap();
            let n = b.unary(UnOp::Neg, x).unwrap();
            let y = b.unary(op, n).unwrap();
            let graph = b.finish(y).unwrap();
            let fused = fuse(&graph, crate::RowOutputs::Multiple).unwrap();
            let eqn = fused
                .eqns()
                .iter()
                .find(|e| matches!(e.op, OpKind::Fused(_)))
                .unwrap_or_else(|| panic!("{op:?} did not fuse into a region: {fused:?}"));
            for backend in backends {
                plan_eqn(
                    &fused,
                    eqn,
                    backend,
                    &default_caps_for(backend),
                    &crate::test_support::roomy_body_limits(),
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

    /// A lowering the finalizer must not reach: the choice it is handed never gets as far as asking it.
    #[derive(Debug)]
    struct NeverAsked;

    impl NpuLowering for NeverAsked {
        fn finalize_serialized(
            &self,
            _choice: &KernelChoice,
            _caps: &AieCaps,
        ) -> Result<(), Capability> {
            panic!("a plan that is not serialized is refused before the lowering is asked")
        }

        fn region_fits(
            &self,
            _graph: &Graph,
            _plans: &[Plan],
            _region: &crate::Region,
            _target: AieTargetId,
            _caps: &AieCaps,
        ) -> Result<(), Capability> {
            unreachable!("finalization asks no region question")
        }

        fn narrow(
            &self,
            _graph: &Graph,
            _caps: &poot_target::VpuCaps,
        ) -> Result<Graph, poot_graph_ir::EditError> {
            unreachable!("finalization asks no Intel question")
        }

        fn supports_eqn(
            &self,
            _graph: &Graph,
            _eqn: &Eqn,
            _caps: &poot_target::VpuCaps,
        ) -> Result<(), Capability> {
            unreachable!("finalization asks no Intel question")
        }

        fn check_region(
            &self,
            _graph: &Graph,
            _region: &crate::Region,
            _caps: &poot_target::VpuCaps,
        ) -> Result<(), crate::NpuRegionRefusal> {
            unreachable!("finalization asks no Intel question")
        }
    }

    /// POOT-1289: the AIE finalizer admits only a serialized schedule. Every AIE lowering the planner
    /// reaches is serialized, so the finalizer is handed a GPU plan (a thread grid, the choice a GPU
    /// target makes for the same add) and refuses it by name instead of linking a body that assumes a grid.
    ///
    /// Mutation: let the `Generated` arm of `finalize_aie` return `Ok(())`; the grid plan is admitted and
    /// this row goes red (`expected AieSchedule`).
    #[test]
    fn the_aie_finalizer_refuses_a_plan_that_is_not_a_serialized_schedule() {
        let b = Builder::for_tests();
        let x = b.constant("a", TensorType::f32(vec![2, 3])).unwrap();
        let y = b.constant("b", TensorType::f32(vec![2, 3])).unwrap();
        let sum = b.binary(poot_graph_ir::BinOp::Add, x, y).unwrap();
        let add = b.finish(sum).unwrap();
        let eqn = add.eqns().last().unwrap();
        let gpu = crate::Target::Gpu {
            backend: Backend::SpirvVulkan,
            caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
        };
        let grid = crate::plan_eqn_choice(
            &add,
            eqn,
            &gpu,
            &std::collections::HashMap::new(),
            &crate::test_support::roomy_body_limits(),
            None,
        )
        .expect("the GPU plans the add");
        assert!(matches!(grid.choice, KernelChoice::Generated(_)));
        let caps = poot_target::AieCaps {
            target: AieTargetId::Xdna2,
            tile_local_bytes: poot_target::Queried::Unknown,
            host_bindings: poot_target::Queried::Unknown,
            instruction_bytes: poot_target::Queried::Unknown,
            memory_budget_bytes: poot_target::Queried::Unknown,
            region: poot_target::aie::RegionLimits {
                compute_tiles: poot_target::Queried::Unknown,
                tile_input_channels: poot_target::Queried::Unknown,
                tile_output_channels: poot_target::Queried::Unknown,
                host_input_streams: poot_target::Queried::Unknown,
                host_output_streams: poot_target::Queried::Unknown,
            },
            step_widths: poot_target::StepWidths::NONE,
        };
        match finalize_aie(
            &grid,
            GraphTables::from(&add),
            eqn,
            AieTargetId::Xdna2,
            &caps,
            &NeverAsked,
        ) {
            Err(PlanError::Refused(refusal)) => {
                assert_eq!(refusal.missing, Capability::AieSchedule);
            }
            other => panic!("expected AieSchedule, got {:?}", other.map(|_| ())),
        }
    }
}
