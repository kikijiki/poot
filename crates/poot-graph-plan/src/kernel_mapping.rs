//! Mapping graph-level fused operations to kernel-IR operations, and the one finalizer every planned
//! kernel passes through before it can run.
//!
//! [`finalize`] shapes the launch (workgroup bump, 2-D fold, the plan key that names it) and then proves the kernel fits the device:
//! generated and imported kernels alike are checked against the measured [`DeviceCaps`] by
//! [`validate_kernel_resources`], so no plan that cannot run reaches a loader.

use crate::planner::Planned;
use crate::predicates::{
    ELEMENTWISE_2D_WIDTH, GraphTables, is_batched_tiled_gemm_in, is_decode_gemv_in,
    is_elementwise_2d_in, is_tiled_gemm_in,
};
use crate::refusal::{FusedStepGap, Site, refusal_at};
use crate::types::{DispatchPlanError, Launch};
use poot_codegen::{AieAbiMismatch, AieFifo, AieLinkRecipe, AieTile, FifoEnd};
use poot_kernelgen::{KernelRequirements, SerialWork};
use poot_runtime_common::{
    AieArtifactError, AieBinding, AieLinkedArtifact, AieShapeAbi, ArgAccess,
};
use poot_target::{AieCaps, AieTargetId, Backend, BufferStorage, DeviceCaps, ElementKind, Queried};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::device_validation::validate_kernel_resources;
use crate::imported::{SAMPLER_LANES, SerialLoop};
use crate::{Capability, KernelChoice, Plan, PlanError, Region, Sink, Source, value_ids};
use poot_graph_ir::Operand;
use poot_graph_ir::Scalar;
use poot_graph_ir::op::BinOp as GBinOp;
use poot_graph_ir::op::FusedOp;
use poot_graph_ir::op::FusedOperand;
use poot_graph_ir::op::OpKind;
use poot_graph_ir::op::UnOp as GUnOp;
use poot_graph_ir::{Eqn, Graph, ValueId};
use poot_kernel_ir::BinOp;
use poot_kernel_ir::MathOp;
use poot_kernel_ir::UnOp;
use poot_kernelgen as kg;
use poot_kernelgen::KernelRequest;
use poot_kernelgen::UnaryOp;
use poot_tensor::DType;

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

/// The one GPU finalizer: shape the launch of `planned`, then check every kernel it dispatches against
/// `caps`. Takes the plan by `&mut` so the launch-owning families run it at their own tail while the
/// dispatcher runs it after its match. An AIE target finalizes through [`finalize_aie`].
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
    shape_launch(planned, g, eqn, out_shape, out_numel, backend, odt, caps)?;
    validate_planned_resources(planned, g, eqn, out_numel, backend, caps)
}

/// The AIE finalizer. An AIE core runs one sequential loop. Its schedule was chosen at generation (the
/// request is the ordinary one); the GPU launch shaping and the device resource finalizer do not apply, and
/// what does is checked here, by type: the plan must be a generated, grid-free, workgroup-memory-free body,
/// and the shape ABI it links against must fit the target's measured limits (an unmeasured limit refuses).
/// The limits only the linked artifact can answer are checked by [`validate_linked_artifact`] before load.
pub(crate) fn finalize_aie(
    planned: &Planned,
    g: GraphTables<'_>,
    eqn: &Eqn,
    target: AieTargetId,
    caps: &AieCaps,
) -> Result<(), PlanError> {
    check_operand_bindings(planned, eqn)?;
    let refuse = |missing| refusal_at(g, eqn, target, missing);
    match planned.choice {
        KernelChoice::NoDispatch(_) => Ok(()),
        KernelChoice::Serialized { .. } => {
            let abi = aie_shape_abi(&planned.choice)
                .map_err(|refusal| refuse(Capability::AieAbi(refusal)))?;
            check_shape_abi_limits(&abi, caps)
                .map_err(|refusal| refuse(Capability::AieLimit(refusal)))
        }
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

/// Why an AIE limit check refused: the limit that is unknown, or the need and the offer.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AieLimitRefusal {
    /// The artifact was linked for another target than the limits describe.
    #[error("the artifact was linked for {artifact:?}, the limits describe {caps:?}")]
    Target {
        /// The target the artifact was linked for.
        artifact: AieTargetId,
        /// The target the limits describe.
        caps: AieTargetId,
    },
    /// The target's limit is not measured, so the artifact's need of it cannot be admitted.
    #[error("the target's {limit} is not measured")]
    Unknown {
        /// Name of the unmeasured limit.
        limit: &'static str,
    },
    /// The artifact needs more of a measured resource than the target offers.
    #[error("{resource}: needed {needed}, the target offers {available}")]
    Exceeded {
        /// Name of the resource that is over its limit.
        resource: &'static str,
        /// What the artifact needs, in the resource's own unit.
        needed: u64,
        /// What the target offers, in the same unit.
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

/// The part of finalization a plan answers on its own: the host buffers its shape ABI needs against the
/// target's. The firmware binds a region's host side as arenas, so what is counted is the arenas the
/// bindings are laid out in, not the bindings. An unmeasured limit refuses.
fn check_shape_abi_limits(abi: &AieShapeAbi, caps: &AieCaps) -> Result<(), AieLimitRefusal> {
    within(
        caps.host_bindings,
        "host buffers",
        abi.arenas().len() as u64,
    )
}

/// A region's link recipe against everything the target bounds about one region, before anything is
/// linked: the compute tiles it places, the DMA channels each tile and the shim open, the tile-local bytes
/// of its busiest tile and the host buffers of its arenas. This is the one definition of "a region fits":
/// `package` closes a region where it stops being true and the loader refuses an artifact where it never
/// was. An unmeasured limit refuses.
pub(crate) fn validate_region_recipe(
    recipe: &AieLinkRecipe,
    caps: &AieCaps,
) -> Result<(), AieLimitRefusal> {
    let limits = &caps.region;
    within(
        limits.compute_tiles,
        "compute tiles",
        recipe.tiles().len() as u64,
    )?;
    for tile in 0..recipe.tiles().len() {
        let (reads, writes) = recipe.tile_streams(tile);
        within(
            limits.tile_input_channels,
            "tile input channels",
            u64::from(reads),
        )?;
        within(
            limits.tile_output_channels,
            "tile output channels",
            u64::from(writes),
        )?;
    }
    let (reads, writes) = recipe.host_streams();
    within(
        limits.host_input_streams,
        "host input streams",
        u64::from(reads),
    )?;
    within(
        limits.host_output_streams,
        "host output streams",
        u64::from(writes),
    )?;
    validate_recipe_tile_local(recipe.tile_local_bytes(), caps)?;
    check_shape_abi_limits(recipe.abi(), caps)
}

/// The tile-local bytes a link recipe reserves, against the target's limit, before anything is linked: the
/// recipe's FIFO buffers are fixed by the shape ABI, so an over-limit plan is refused here as a typed value
/// instead of as a toolchain failure. An unmeasured limit refuses.
fn validate_recipe_tile_local(reserved_bytes: u64, caps: &AieCaps) -> Result<(), AieLimitRefusal> {
    within(
        caps.tile_local_bytes,
        "link recipe tile-local bytes",
        reserved_bytes,
    )
}

/// The finalization only a linked artifact can answer, run by the loader before it wraps the artifact: the
/// artifact was linked for `caps`' target, and its tile-local reservation, host bindings and instruction
/// stream fit the target's limits. An unmeasured limit refuses.
pub fn validate_linked_artifact(
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
    check_shape_abi_limits(artifact.abi(), caps)?;
    within(
        caps.instruction_bytes,
        "instruction bytes",
        artifact.instructions().len() as u64,
    )
}

/// The shape ABI of a planned sequential schedule: one binding per body parameter, in binding order,
/// each with the dimensions and access range the generator proved when it produced the body. The ABI is
/// read from the plan's own choice, so the body and the ABI come from the one generation the planner ran.
pub fn aie_shape_abi(choice: &KernelChoice) -> Result<AieShapeAbi, AieAbiRefusal> {
    let KernelChoice::Serialized { bindings, .. } = choice else {
        return Err(AieAbiRefusal::NotSerialized);
    };
    let bindings = bindings
        .iter()
        .enumerate()
        .map(|(index, binding)| {
            AieBinding::new(
                index,
                word_storage(binding.element),
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

/// The canonical word buffer of an element kind an AIE binding may have (F32, else I32). The shape ABI
/// refuses any other storage when it builds the binding.
fn word_storage(element: ElementKind) -> BufferStorage {
    match element {
        ElementKind::F32 => BufferStorage::f32(),
        _ => BufferStorage::i32(),
    }
}

/// Why a plan has no AIE shape ABI, or a region has no link recipe.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum AieAbiRefusal {
    /// The plan's kernel choice is not an AIE serialized schedule, so it has no shape ABI.
    #[error("the plan's choice is not an AIE serialized schedule")]
    NotSerialized,
    /// The linked artifact's ABI could not be derived.
    #[error(transparent)]
    Abi(#[from] AieArtifactError),
    /// An equation of a region does not plan as one scratch-free dispatch that reads its operands and
    /// writes its output: a core runs one body once, so there is nothing else to place on a tile.
    #[error("equation {eqn:?} does not plan as one scratch-free dispatch over its operands")]
    NotOneDispatch {
        /// The value the equation produces.
        eqn: ValueId,
    },
    /// An equation's sequential bindings are not its operands in order followed by its one output.
    #[error("equation {eqn:?}: the sequential bindings are not its operands then its output")]
    Bindings {
        /// The value the equation produces.
        eqn: ValueId,
    },
    /// Two equations bind one value with different dimensions or element kinds.
    #[error(
        "value {value:?} is bound with different dimensions or elements by two equations of the region"
    )]
    ReaderDims {
        /// The value that is read twice.
        value: ValueId,
    },
    /// The region's wiring is not a link recipe.
    #[error(transparent)]
    Recipe(#[from] AieAbiMismatch),
    /// A region was asked of a program that was not compiled for an AIE target.
    #[error("the program was not compiled for an AIE target, so it has no region recipes")]
    NotAnAieProgram,
}

/// What the region's equations declare of one value: its dimensions, element kind and proved access.
type Declared<'a> = (&'a [u64], ElementKind, u64);

/// Record the dimensions, element kind and proved access `binding` declares for `value`. A second
/// declaration must have the same dimensions and element kind; the larger proved access wins.
fn note_dims<'a>(
    seen: &mut HashMap<ValueId, Declared<'a>>,
    value: ValueId,
    binding: &'a kg::SerializedBinding,
) -> Result<(), AieAbiRefusal> {
    match seen.entry(value) {
        Entry::Vacant(slot) => {
            slot.insert((&binding.dims, binding.element, binding.accessed));
        }
        Entry::Occupied(mut slot) => {
            if *slot.get().0 != *binding.dims || slot.get().1 != binding.element {
                return Err(AieAbiRefusal::ReaderDims { value });
            }
            slot.get_mut().2 = slot.get().2.max(binding.accessed);
        }
    }
    Ok(())
}

/// The link recipe of `region`: one tile per equation, in program order, and one FIFO per value. A value the
/// region reads from outside streams in from the host, a value it returns streams out to the host, and a
/// value only its own equations read moves from the tile that produces it to the tiles that read it. The
/// host side is the shape ABI of the values read and then the values returned, so the last binding is always
/// a written one and a dispatch binds them in that order.
///
/// Each tile's ports are its body's parameters: the equation's operands, then its output. The bodies and
/// the proved access ranges come from the same planner run as `plans`, so the recipe cannot disagree with
/// the bodies it is linked against; `poot-codegen` still checks every body against it.
pub(crate) fn aie_region_recipe(
    g: &Graph,
    plans: &[Plan],
    region: &Region,
    target: AieTargetId,
) -> Result<AieLinkRecipe, AieAbiRefusal> {
    /// One member's dispatch seen as a tile program.
    struct Member<'a> {
        out: ValueId,
        reads: Vec<ValueId>,
        bindings: &'a [kg::SerializedBinding],
    }
    let members = region
        .members()
        .map(|index| {
            let eqn = &g.eqns()[index];
            let not_one = AieAbiRefusal::NotOneDispatch { eqn: eqn.out };
            let plan = &plans[index];
            let dispatch = plan.single().ok_or(not_one.clone())?;
            if !matches!(plan, Plan::Dispatches(d) if d.scratch().is_empty())
                || dispatch.binding().write() != Sink::Out
            {
                return Err(not_one);
            }
            let operands = value_ids(eqn);
            let reads = dispatch
                .binding()
                .reads()
                .iter()
                .map(|source| match *source {
                    Source::Operand(operand) => operands.get(operand).copied(),
                    Source::Scratch(_) => None,
                })
                .collect::<Option<Vec<_>>>()
                .ok_or(AieAbiRefusal::NotOneDispatch { eqn: eqn.out })?;
            let KernelChoice::Serialized { bindings, .. } = dispatch.choice() else {
                return Err(AieAbiRefusal::NotSerialized);
            };
            let ordered = bindings.len() == reads.len() + 1
                && bindings.iter().rev().skip(1).all(|b| !b.writes)
                && bindings.last().is_some_and(|b| b.writes);
            if !ordered {
                return Err(AieAbiRefusal::Bindings { eqn: eqn.out });
            }
            Ok(Member {
                out: eqn.out,
                reads,
                bindings,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // The dimensions and proved access of every value the host sees: what the first equation that touches
    // it declares. Two readers must agree on the dimensions; the larger proved access wins.
    let mut seen: HashMap<ValueId, Declared<'_>> = HashMap::new();
    for member in &members {
        for (&value, binding) in member.reads.iter().zip(member.bindings) {
            note_dims(&mut seen, value, binding)?;
        }
        let write = member.bindings.last().expect("checked non-empty above");
        note_dims(&mut seen, member.out, write)?;
    }

    let host_binding = |index: usize, value: ValueId, access: ArgAccess| {
        let (dims, element, accessed) = seen[&value];
        AieBinding::new(index, word_storage(element), access, dims, accessed)
    };
    let mut host = Vec::new();
    for (index, &value) in region.reads().iter().enumerate() {
        host.push(host_binding(index, value, ArgAccess::Read)?);
    }
    for (index, &value) in region.writes().iter().enumerate() {
        host.push(host_binding(
            region.reads().len() + index,
            value,
            ArgAccess::Write,
        )?);
    }
    let abi = AieShapeAbi::new(host)?;

    // One FIFO per value: the region's reads first, then each equation's output.
    let mut fifo_of: HashMap<ValueId, usize> = HashMap::new();
    let mut fifos: Vec<AieFifo> = Vec::new();
    let mut open = |value: ValueId, producer: FifoEnd| {
        let (dims, element, _) = seen[&value];
        fifo_of.insert(value, fifos.len());
        fifos.push(AieFifo {
            producer,
            consumers: Vec::new(),
            depth: AieLinkRecipe::FIFO_DEPTH,
            elements: dims.iter().product(),
            element,
        });
    };
    for (binding, &value) in region.reads().iter().enumerate() {
        open(value, FifoEnd::Host { binding });
    }
    for (tile, member) in members.iter().enumerate() {
        open(
            member.out,
            FifoEnd::Tile {
                tile,
                port: member.reads.len(),
            },
        );
    }
    let mut tiles = Vec::with_capacity(members.len());
    for (tile, member) in members.iter().enumerate() {
        let mut ports = Vec::with_capacity(member.reads.len() + 1);
        for (port, value) in member.reads.iter().enumerate() {
            let fifo = *fifo_of
                .get(value)
                .ok_or(AieAbiRefusal::NotOneDispatch { eqn: member.out })?;
            fifos[fifo].consumers.push(FifoEnd::Tile { tile, port });
            ports.push(fifo);
        }
        ports.push(fifo_of[&member.out]);
        tiles.push(AieTile { ports });
    }
    for (index, &value) in region.writes().iter().enumerate() {
        let fifo = *fifo_of
            .get(&value)
            .ok_or(AieAbiRefusal::NotOneDispatch { eqn: value })?;
        fifos[fifo].consumers.push(FifoEnd::Host {
            binding: region.reads().len() + index,
        });
    }
    Ok(AieLinkRecipe::new(target, abi, tiles, fifos)?)
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
        KernelChoice::NoDispatch(_) | KernelChoice::Serialized { .. } => {
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
            | OpKind::BlockTopK(_)
            | OpKind::SelectedAttention(_)
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
    use crate::fuse;
    use crate::fuse::FUSABLE_FLOAT_UNARY_OPS;
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
            let b = Builder::for_tests();
            let x = b
                .constant("x", poot_graph_ir::TensorType::f32(vec![4]))
                .unwrap();
            let n = b.unary(UnOp::Neg, x).unwrap();
            let y = b.unary(op, n).unwrap();
            let graph = b.finish(y).unwrap();
            let fused = fuse(&graph).unwrap();
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

    // ---- POOT-664: the AIE sequential schedule is a choice inside the ordinary planner -----------------

    use std::collections::HashMap;
    use std::num::NonZeroU64;
    use std::path::PathBuf;

    use poot_codegen::{
        AieCompileError, KernelCache, Target as CodegenTarget, aie_kernel_handle, compile_aie,
        test_support::single_tile_recipe,
    };
    use poot_graph_ir::op::RedOp;
    use poot_graph_ir::{Graph, TensorType};
    use poot_runtime_common::{AieBindError, AieBound, check_aie_bindings};
    use poot_target::aie::RegionLimits;
    use poot_target::{AieCaps, AieTargetId, Queried, TargetId};
    use poot_tensor::DType;

    use crate::{Capability, KernelChoice, PlanError, Target};

    use super::{AieLimitRefusal, aie_shape_abi, validate_linked_artifact};

    const AIE: TargetId = TargetId::Aie(AieTargetId::Xdna2);

    fn aie_caps() -> AieCaps {
        // SYNTHETIC limits: roomy enough for a [2,3] kernel, not a device measurement.
        AieCaps {
            target: AieTargetId::Xdna2,
            tile_local_bytes: Queried::Known(1 << 16),
            host_bindings: Queried::Known(8),
            instruction_bytes: Queried::Known(1 << 20),
            memory_budget_bytes: Queried::Known(1 << 30),
            region: RegionLimits {
                compute_tiles: Queried::Known(32),
                tile_input_channels: Queried::Known(2),
                tile_output_channels: Queried::Known(2),
                host_input_streams: Queried::Known(16),
                host_output_streams: Queried::Known(16),
            },
        }
    }

    /// A target nobody has measured: every bound unknown, so no plan that needs one is admitted.
    fn unmeasured_caps() -> AieCaps {
        AieCaps {
            tile_local_bytes: Queried::Unknown,
            host_bindings: Queried::Unknown,
            instruction_bytes: Queried::Unknown,
            memory_budget_bytes: Queried::Unknown,
            region: RegionLimits {
                compute_tiles: Queried::Unknown,
                tile_input_channels: Queried::Unknown,
                tile_output_channels: Queried::Unknown,
                host_input_streams: Queried::Unknown,
                host_output_streams: Queried::Unknown,
            },
            ..aie_caps()
        }
    }

    fn limits() -> poot_kernelgen::BodyLimits {
        crate::test_support::roomy_body_limits()
    }

    fn graph_of(build: impl FnOnce(&Builder) -> poot_graph_ir::Traced) -> Graph {
        let b = Builder::for_tests();
        let out = build(&b);
        b.finish(out).unwrap()
    }

    fn add_graph(a_shape: Vec<usize>, b_shape: Vec<usize>, dtype: DType) -> Graph {
        graph_of(|b| {
            let x = b.constant("a", TensorType::new(a_shape, dtype)).unwrap();
            let y = b.constant("b", TensorType::new(b_shape, dtype)).unwrap();
            b.binary(poot_graph_ir::BinOp::Add, x, y).unwrap()
        })
    }

    fn last_eqn(g: &Graph) -> &poot_graph_ir::Eqn {
        g.eqns().last().expect("the graph has an equation")
    }

    /// The ordinary planner entry for an AIE target carrying `caps`.
    fn plan_aie_with(g: &Graph, caps: AieCaps) -> Result<crate::planner::Planned, PlanError> {
        let target = Target::Aie {
            target: AieTargetId::Xdna2,
            caps,
        };
        crate::plan_eqn_choice(g, last_eqn(g), &target, &HashMap::new(), &limits())
    }

    fn plan_aie(g: &Graph) -> Result<crate::planner::Planned, PlanError> {
        plan_aie_with(g, aie_caps())
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

    /// The planner positive: `plan_eqn` itself plans an F32 `[2,3]` add for an AIE target, as the ordinary
    /// elementwise request under the serialized schedule on the existing plan types (one dispatch, one
    /// invocation, a one-thread workgroup). The compiler links its body against the ABI derived from that
    /// request, the ordinary handle wraps it, and a fake loader sees three ordered arguments and core
    /// lengths `[6, 6, 6]`. A fake loader does not claim numerical execution or hardware.
    /// Mutation: refuse the elementwise family in `plan_aie_eqn` or give `build_serialized` a GPU grid; the
    /// plan or the dispatch-shape assertion fails.
    #[test]
    fn plan_eqn_plans_an_f32_add_under_the_aie_serialized_schedule_and_it_links() {
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
        let tc = poot_codegen::test_support::scripted_toolchain(&dir, true);
        let cache = KernelCache::at(dir.join("cache"), CodegenTarget::AieCore);
        let compiled = compile_aie(
            &[&dispatch.body],
            &single_tile_recipe(AieTargetId::Xdna2, &abi),
            &tc,
            &cache,
            NonZeroU64::new(1 << 20).unwrap(),
        )
        .unwrap();
        validate_linked_artifact(&compiled.artifact, &aie_caps()).unwrap();
        let artifact = compiled.artifact.clone();
        let handle = aie_kernel_handle(compiled.artifact);
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

    /// A unary F32 equation plans for an AIE target as the ordinary request under the serialized schedule:
    /// two bindings (the operand it reads, the output it writes), one loop over every element, and the op's
    /// numerical contract carried in the choice, so the plan key tells `exp` from `tanh`.
    /// Mutation: give `exp` the `tanh` contract in `AieOp::contract`; the contract assertion fails, and the
    /// two plan keys no longer differ by more than the request.
    #[test]
    fn plan_eqn_plans_a_unary_f32_under_its_numerical_contract() {
        let unary = |op| {
            graph_of(|b| {
                let x = b.constant("x", TensorType::f32(vec![3, 4])).unwrap();
                b.unary(op, x).unwrap()
            })
        };
        let mut keys = Vec::new();
        for (op, id) in [
            (poot_graph_ir::UnOp::Exp, "f32-exp-software-v1"),
            (poot_graph_ir::UnOp::Tanh, "f32-tanh-software-v1"),
            (poot_graph_ir::UnOp::Neg, "f32-neg-v1"),
        ] {
            let planned = plan_aie(&unary(op)).unwrap();
            let KernelChoice::Serialized {
                contract, bindings, ..
            } = &planned.choice
            else {
                panic!("{op:?} is planned as a serialized schedule");
            };
            assert_eq!(contract.id, id);
            assert_eq!(
                bindings.iter().map(|b| b.writes).collect::<Vec<_>>(),
                [false, true]
            );
            // The buffers are bound at the graph's own dimensions.
            assert!(
                bindings
                    .iter()
                    .all(|b| b.dims == [3, 4] && b.accessed == 12)
            );
            let dispatch = planned.plan.single().expect("one dispatch");
            assert_eq!(dispatch.body.workgroup_size, [1, 1, 1]);
            assert_eq!(
                aie_shape_abi(&planned.choice).unwrap().core_lengths(),
                [12, 12]
            );
            keys.push(dispatch.key.clone());
        }
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 3, "each op is its own kernel");
    }

    /// A last-axis reduction and a gather plan for an AIE target as the ordinary requests under the
    /// serialized schedule, bound at the graph's own dimensions: the reduction's output keeps its unit
    /// axis, the gather's index is an I32 buffer in an arena of its own, and the plan's contract is the
    /// op's. Mutation: bind the gather's index as `ElementKind::F32` in `serialized_movement`, or take the
    /// dimensions from the request instead of the graph in `build_serialized`; the assertions fail.
    #[test]
    fn plan_eqn_plans_a_row_reduction_and_a_gather_with_the_graphs_dimensions() {
        use poot_target::ElementKind;
        let sum = graph_of(|b| {
            let x = b.constant("x", TensorType::f32(vec![5, 8])).unwrap();
            b.reduce(RedOp::Sum, x, 1, true).unwrap()
        });
        let planned = plan_aie(&sum).unwrap();
        let KernelChoice::Serialized {
            contract, bindings, ..
        } = &planned.choice
        else {
            panic!("a row sum is planned as a serialized schedule");
        };
        assert_eq!(contract.id, "f32-row-sum-sequential-v1");
        let seen: Vec<_> = bindings.iter().map(|b| (&b.dims[..], b.accessed)).collect();
        assert_eq!(seen, [(&[5u64, 8][..], 40), (&[5u64, 1][..], 5)]);

        let gather = graph_of(|b| {
            let table = b.constant("table", TensorType::f32(vec![10, 3])).unwrap();
            let ids = b
                .constant("ids", TensorType::new(vec![4], DType::I32))
                .unwrap();
            b.gather(table, 0, ids).unwrap()
        });
        let planned = plan_aie(&gather).unwrap();
        let KernelChoice::Serialized {
            contract, bindings, ..
        } = &planned.choice
        else {
            panic!("a gather is planned as a serialized schedule");
        };
        assert_eq!(contract.id, "gather-copy-v1");
        let seen: Vec<_> = bindings
            .iter()
            .map(|b| (&b.dims[..], b.element, b.writes))
            .collect();
        assert_eq!(
            seen,
            [
                (&[10u64, 3][..], ElementKind::F32, false),
                (&[4u64][..], ElementKind::I32, false),
                (&[4u64, 3][..], ElementKind::F32, true),
            ]
        );
        let abi = aie_shape_abi(&planned.choice).unwrap();
        let arenas: Vec<_> = abi
            .arenas()
            .iter()
            .map(|a| (a.access(), a.element(), a.words()))
            .collect();
        assert_eq!(
            arenas,
            [
                (poot_runtime_common::ArgAccess::Read, ElementKind::F32, 30),
                (poot_runtime_common::ArgAccess::Read, ElementKind::I32, 4),
                (poot_runtime_common::ArgAccess::Write, ElementKind::F32, 12),
            ]
        );
    }

    /// The same ordinary request path refuses what has no serialized schedule, each with the planner's own
    /// typed refusal: a family the generator has no loop for, a broadcast operand, a non-F32 element, an
    /// op not proved on the core. Mutation: make `serialized_schedule` fall back to the GPU generator, or
    /// drop its operand-shape check; the matching row returns `Ok`.
    #[test]
    fn plan_eqn_refuses_for_aie_what_has_no_serialized_schedule() {
        let kernel_gen = |capability: Capability| match capability {
            Capability::KernelGen(poot_kernelgen::KernelGenError::Unsupported {
                request,
                reason,
            }) => (request, reason),
            other => panic!("expected a generator refusal, got {other:?}"),
        };
        let round = graph_of(|b| {
            let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
            b.unary(poot_graph_ir::UnOp::Round, x).unwrap()
        });
        let (request, reason) = kernel_gen(aie_refusal(&round));
        assert_eq!(request, "serialized_schedule");
        assert!(reason.contains("no proved lowering"), "{reason}");
        let broadcast = add_graph(vec![2, 3], vec![1, 3], DType::F32);
        assert!(kernel_gen(aie_refusal(&broadcast)).1.contains("broadcast"));
        let ints = add_graph(vec![2, 3], vec![2, 3], DType::I32);
        assert_eq!(kernel_gen(aie_refusal(&ints)).0, "serialized_schedule");
        // A reduction over a leading axis has no row to loop over.
        let leading = graph_of(|b| {
            let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
            b.reduce(RedOp::Sum, x, 0, false).unwrap()
        });
        assert_eq!(
            aie_refusal(&leading),
            Capability::NonLastAxisReduce { axis: 0 }
        );
        // A half-precision reduction is not a row of F32 words.
        let bf16_rows = graph_of(|b| {
            let x = b
                .constant("x", TensorType::new(vec![2, 3], DType::BF16))
                .unwrap();
            b.reduce(RedOp::Sum, x, 1, false).unwrap()
        });
        assert_eq!(kernel_gen(aie_refusal(&bf16_rows)).0, "serialized_schedule");
        // A data movement other than a gather reads GPU capabilities.
        let transpose = graph_of(|b| {
            let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
            b.transpose(x, vec![1, 0]).unwrap()
        });
        assert_eq!(aie_refusal(&transpose), Capability::GpuOnly);
        // A collective is a cut between stage programs on every target.
        let reduce = graph_of(|b| {
            let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
            b.all_reduce(x, RedOp::Sum, 0).unwrap()
        });
        assert!(matches!(
            aie_refusal(&reduce),
            Capability::CollectiveInStage
        ));
    }

    /// A lowering that reads GPU capabilities (a shipped GPU template, a contraction) is a typed
    /// [`Capability::GpuOnly`] refusal for an AIE target, through the ordinary planner, on a deliberately
    /// unsupported fixture: the AIE target has no `DeviceCaps` to read, and the refusal names the target.
    /// Mutation: dispatch the family to its GPU planner for an AIE target (give `plan_aie_eqn` a `matmul`
    /// arm that plans with a fixture `DeviceCaps`); the contraction is planned as a GPU kernel and the row
    /// returns `Ok`.
    #[test]
    fn a_gpu_only_lowering_is_a_typed_refusal_for_an_aie_target() {
        let template = graph_of(|b| {
            let seed = b
                .constant("seed", TensorType::new(vec![2], DType::I32))
                .unwrap();
            b.random_uniform(seed, 4).unwrap()
        });
        assert_eq!(aie_refusal(&template), Capability::GpuOnly);
        let contraction = graph_of(|b| {
            let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
            let w = b.constant("w", TensorType::f32(vec![3, 4])).unwrap();
            b.matmul(x, w).unwrap()
        });
        assert_eq!(aie_refusal(&contraction), Capability::GpuOnly);
    }

    /// The planner finalizes an AIE plan against the target's own limits: the shape ABI it derives from the
    /// plan's choice lays its three bindings out in two host arenas (the reads, the write), an unmeasured
    /// limit refuses (never a guess), and a measured one the ABI exceeds refuses with both numbers.
    /// Mutation: drop `check_shape_abi_limits` from `finalize_aie`; both rows plan.
    #[test]
    fn the_planner_refuses_a_shape_abi_the_target_cannot_bind() {
        let add = add_graph(vec![2, 3], vec![2, 3], DType::F32);
        let refusal = |caps| match plan_aie_with(&add, caps) {
            Err(PlanError::Refused(refusal)) => refusal.missing,
            other => panic!("expected a typed refusal, got {:?}", other.map(|_| ())),
        };
        assert_eq!(
            refusal(unmeasured_caps()),
            Capability::AieLimit(AieLimitRefusal::Unknown {
                limit: "host buffers"
            })
        );
        let one = AieCaps {
            host_bindings: Queried::Known(1),
            ..aie_caps()
        };
        assert_eq!(
            refusal(one),
            Capability::AieLimit(AieLimitRefusal::Exceeded {
                resource: "host buffers",
                needed: 2,
                available: 1
            })
        );
        assert!(plan_aie(&add).is_ok(), "control: the roomy target plans it");
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
            &[&body],
            &single_tile_recipe(AieTargetId::Xdna2, &abi),
            &poot_codegen::test_support::scripted_toolchain(&dir, true),
            &KernelCache::at(dir.join("cache"), CodegenTarget::AieCore),
            NonZeroU64::new(1 << 20).unwrap(),
        )
        .unwrap();
        let artifact = &compiled.artifact;
        assert_eq!(validate_linked_artifact(artifact, &aie_caps()), Ok(()));
        let unmeasured = unmeasured_caps();
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
        let one_buffer = AieCaps {
            host_bindings: Queried::Known(1),
            ..aie_caps()
        };
        assert_eq!(
            validate_linked_artifact(artifact, &one_buffer),
            Err(AieLimitRefusal::Exceeded {
                resource: "host buffers",
                needed: 2,
                available: 1
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
            &[&body],
            &single_tile_recipe(AieTargetId::Xdna2, &abi),
            &poot_codegen::test_support::scripted_toolchain(&dir, false),
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
        let dir = cache
            .dir()
            .expect("the cache sits at an explicit directory");
        assert!(!dir.exists() || std::fs::read_dir(dir).unwrap().next().is_none());
    }
}
