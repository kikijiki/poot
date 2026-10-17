//! Typed planner refusals (ADR-0104 decision 2): a refusal names the equation, the op, the dtypes, the
//! target and the missing capability. It is never free text, so a caller matches on it instead of
//! restating the planner's gaps.

use crate::predicates::GraphTables;
use std::fmt;

use poot_kernelgen::BodyLimits;
use poot_target::TargetId;

use crate::{CollectiveKind, PlanError};
use poot_graph_ir::op::{FusedOp, OpKind};
use poot_graph_ir::{Eqn, Operand, ValueId};
use poot_tensor::DType;

/// One equation the planner cannot lower for one target.
///
/// Built only by the planner (`Site::refuse`), which fills every field from the equation it was
/// planning, so no site can leave one out. Only [`Refusal::missing`] varies by site.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    /// The refused equation, named by its output value (an equation has exactly one output).
    pub eqn: ValueId,
    /// The refused op, with its parameters.
    pub op: OpKind,
    /// The dtypes the equation reads and writes.
    pub dtypes: RefusalDtypes,
    /// The target the equation was planned for.
    pub target: TargetId,
    /// The capability the target lacks for this equation.
    pub missing: Capability,
}

/// The operand and output dtypes of a refused equation. A literal operand carries its scalar's dtype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefusalDtypes {
    /// The dtype of each operand, in operand order.
    pub operands: Vec<DType>,
    /// The dtype of the equation's output.
    pub output: DType,
}

/// A capability a target lacks, with the parameters that make the gap checkable.
///
/// Each variant is a real gap in the planner's lowering table or a precondition of a lowering that the
/// equation violates. None is roadmap wording: the variant says what is missing, and the enclosing
/// [`Refusal`] says for which equation and target.
#[derive(Clone, Debug, PartialEq)]
pub enum Capability {
    /// The equation is a cross-device collective (`AllReduce`, `AllGather`). A collective is a cut point
    /// between stage programs: `compile_staged` places the transfer between them, so no stage program
    /// holds one.
    CollectiveInStage,
    /// The target set lacks the cross-device operation a stage cut needs: the refused equation produces a
    /// value that crosses devices, and no link row admits `kind` between its producer's and consumer's
    /// devices. Raised before any stage compile or device call.
    Collective(CollectiveKind),
    /// The op lowers through imported-kernel production planning (the packed-block-float and production
    /// plan routes), never the generic per-equation planner.
    ImportedKernelPlanning,
    /// No kernel lowers this op for this combination of operand and output dtypes.
    DtypeLowering,
    /// The typed packed E4M3FN storage has no lowering for this op on this target.
    E4m3Storage,
    /// A packed E4M3FN kernel over a zero-element operand or output; its packed layout has no words to
    /// address.
    ZeroElementE4m3,
    /// The packed E4M3FN dispatch size overflows the host's accounting.
    DispatchSizeOverflow,
    /// The packed E4M3FN dispatch needs more x-workgroups than the target's grid allows.
    DispatchGridLimit {
        /// Threads the dispatch needs.
        threads: usize,
        /// X-workgroups those threads make at `workgroup_width`.
        workgroups_x: usize,
        /// Threads per workgroup.
        workgroup_width: usize,
        /// The target's x-grid limit, in workgroups.
        limit: usize,
    },
    /// A planned dispatch needs more workgroups along one grid axis than the target's grid allows. The
    /// two-stage greedy pick puts a row per Y workgroup and a row per X workgroup in its combine stage;
    /// the planner refuses rather than launching a grid the device cannot run.
    DispatchGridAxisLimit {
        /// The grid axis (0 = x) that overflows.
        axis: usize,
        /// Workgroups the dispatch needs along it.
        workgroups: usize,
        /// The target's limit along it.
        limit: usize,
    },
    /// The op has no I32 kernel (reduction, row fusion, a division without total exact semantics, a unary
    /// that is not an exact integer primitive, a remainder by a literal zero). Every such refusal names a
    /// missing kernel: an I32 value is always stored as I32 words.
    ExactI32Op,
    /// The equation is an unfolded `Iota`. It is a pure compile-time constant: `transform::fold_iota`
    /// folds it to a `Storage::Computed` graph constant before planning (R466-019), so no kernel exists.
    UnfoldedIota,
    /// A step of a fused region cannot be lowered in the region's arithmetic.
    FusedStep(FusedStepGap),
    /// The equation has the wrong number of operands for its lowering.
    OperandCount {
        /// Operands the lowering takes.
        expected: usize,
        /// Operands the equation has.
        actual: usize,
    },
    /// An operand's shape violates the lowering's precondition: `shape` is the offending value's actual
    /// shape, `expected` the shape the lowering required of it.
    OperandShape {
        /// The operand whose shape is wrong.
        value: ValueId,
        /// The operand's actual shape.
        shape: Vec<usize>,
        /// The shape the lowering required.
        expected: Vec<usize>,
    },
    /// An operand is a strided view the lowering cannot read: only the layouts the lowering names are admitted.
    OperandLayout {
        /// The strided operand.
        value: ValueId,
    },
    /// A literal in the first operand position, which the planner's operand grammar does not read.
    LiteralFirstOperand,
    /// The index operand is not a form the lowering reads.
    IndexOperand,
    /// A reduction over an axis other than the last has no kernel on this target.
    NonLastAxisReduce {
        /// The reduced axis.
        axis: usize,
    },
    /// `Scatter` has no kernel over an axis other than 0 on any target (Card 546a, R-546-4): the
    /// contract admits no host fallback, so this equation is refused outright rather than routed to
    /// a host step.
    ScatterNonZeroAxis {
        /// The scattered axis.
        axis: usize,
    },
    /// The attention kernel keeps one lane's running state (`dv` output channels and two scalars) in
    /// workgroup memory, and not even one lane's fits the device's `lds_bytes`.
    AttentionLds {
        /// Output channels of one lane's running state.
        dv: usize,
        /// The device's workgroup memory, in bytes.
        lds_bytes: u32,
    },
    /// The scan kernel keeps one state column (`rows` floats) per lane in workgroup memory, and not even one
    /// lane's fits the device's `lds_bytes`.
    ScanLds {
        /// State rows per lane.
        rows: usize,
        /// The device's workgroup memory, in bytes.
        lds_bytes: u32,
    },
    /// The block top-k kernel is one fixed workgroup of `lanes` lanes (one radix bin each) with `lds_needed` bytes of
    /// workgroup memory, and the device's workgroup, invocation or memory limits are below it.
    BlockTopKWorkgroup {
        /// Lanes the kernel needs in one workgroup.
        lanes: usize,
        /// The device's most invocations per workgroup.
        max_invocations: u32,
        /// Workgroup memory the kernel needs, in bytes.
        lds_needed: u64,
        /// The device's workgroup memory, in bytes.
        lds_bytes: u32,
    },
    /// A plan-local scratch buffer (`DispatchPlan::scratch`) is larger than the device's single-buffer
    /// ceiling. The split-KV attention rule chooses the streaming schedule when its scratch does not fit,
    /// so this is the backstop for a plan that reached validation anyway: a typed refusal, never a failed
    /// device allocation.
    ScratchBytes {
        /// Size of the scratch buffer the plan needs, in bytes.
        needed: u64,
        /// The device's single-buffer limit, in bytes.
        max_buffer_bytes: u64,
    },
    /// The selected generator's own precondition on its shape or count arguments rejected this
    /// equation (card 531b): the planner chose this lowering for the equation's dtypes and target, but
    /// forwarded it a static value the generator itself refuses to build a body for.
    KernelGen(poot_kernelgen::KernelGenError),
    /// The planned kernel's body, launch or declared requirements do not fit the device's measured
    /// [`poot_target::DeviceCaps`], or need a capability the device does not expose.
    KernelResources(crate::device_validation::ResourceRefusal),
    /// No row of `packed_block_float::PACKED_LOWERING` admits this format for this
    /// `PackedKernelOpKind` on this target (card 542a, dquant.md D3): the descriptor-driven packed
    /// kernel generator has not yet staged this format/op/backend combination (K-quants and planar
    /// formats are 542b/542c; a schedule a test table omits is the same gap).
    PackedLowering {
        /// The packed format with no admitting row.
        format: poot_quant::format::WeightFormat,
        /// The packed op with no admitting row.
        op: crate::packed_block_float::PackedKernelOpKind,
    },
    /// The plan is not an AIE sequential schedule: an AIE core runs one loop, so a plan built from a thread
    /// grid or a shipped GPU template cannot run there.
    AieSchedule,
    /// The equation's lowering reads GPU capabilities ([`poot_target::DeviceCaps`]) and the target is an
    /// AIE target, which has none.
    GpuOnly,
    /// The plan's shape ABI does not fit the AIE target's measured limits (POOT-664): the limit that is
    /// unknown, or the need and the offer.
    AieLimit(crate::AieLimitRefusal),
    /// The plan has no AIE shape ABI (the choice is not an AIE sequential schedule, or a binding is
    /// malformed).
    AieAbi(crate::AieAbiRefusal),
    /// The target computes in a narrower float than the equation's `dtype` (POOT-1159): an Intel NPU
    /// computes in FP16, so F32 or BF16 arithmetic on it would silently narrow the equation's semantics.
    /// Computing it narrower is an explicit graph transform the operator chooses, never the planner's
    /// default.
    Arithmetic {
        /// The float dtype the equation computes in.
        dtype: DType,
    },
    /// The Intel lowering table has no OpenVINO opset op with this op's semantics (POOT-1159).
    VpuOpset,
    /// An operand is an inline literal; the Intel lowering reads value operands only.
    LiteralOperand,
}

/// What a fused region step cannot be lowered to, by the region's arithmetic.
#[derive(Clone, Debug, PartialEq)]
pub enum FusedStepGap {
    /// A packed I32 lane read inside an f32 region.
    PackedLaneInFloatRegion,
    /// An f32 literal inside an exact I32 region.
    F32LiteralInExactRegion,
    /// An op an f32 region cannot hold.
    OpInFloatRegion(FusedOp),
    /// An op an exact I32 region cannot hold.
    OpInExactRegion(FusedOp),
}

/// A graph value whose storage a tensor-bound executor entry point cannot represent: those entry points
/// bind every value as four-byte f32 words, so a dtype with its own storage layout would be read at the
/// wrong element width. Unlike a [`Refusal`], this concerns a value, not an equation, and no target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageGap {
    /// E4M3FN values need authoritative raw bytes or packed-row storage.
    E4m3RawBytes,
    /// BF16 values need the packed u32 lanes only the typed value walk and the decode BF16 GEMV lane store.
    Bf16PackedLanes,
    /// An F16 dense-contraction weight is stored packed two elements per `u32` word (Card 1007), so it must be a
    /// const no other equation reads and no output observes.
    F16PackedLanes,
    /// An E4M3FN `DenseRowGather` table is stored packed four elements per `u32` word with padded rows
    /// (POOT-1060), so it must be a const no other equation reads and no output observes.
    E4m3PackedLanes,
    /// The dtype is none of the widths an Intel NPU region carries (F16, F32, I32).
    VpuDtype,
}

impl RefusalDtypes {
    pub(crate) fn of(g: GraphTables<'_>, eqn: &Eqn) -> Self {
        Self {
            operands: eqn
                .inputs
                .iter()
                .map(|operand| match operand {
                    Operand::Value(id) => g.aval(*id).dtype,
                    Operand::Lit(scalar) => scalar.dtype(),
                })
                .collect(),
            output: g.aval(eqn.out).dtype,
        }
    }
}

/// The equation being planned, its target and the limits any kernel it generates must fit. Every planner
/// refusal is built through [`Site::refuse`], so the equation, op, dtypes and target are always populated
/// from the equation itself and a site names only the missing capability.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Site<'a> {
    g: GraphTables<'a>,
    eqn: &'a Eqn,
    target: TargetId,
    limits: BodyLimits,
}

impl<'a> Site<'a> {
    pub(crate) fn new(
        g: GraphTables<'a>,
        eqn: &'a Eqn,
        target: impl Into<TargetId>,
        limits: BodyLimits,
    ) -> Self {
        Self {
            g,
            eqn,
            target: target.into(),
            limits,
        }
    }

    /// The body limits a kernel generated at this site is held to.
    pub(crate) fn limits(&self) -> &BodyLimits {
        &self.limits
    }

    pub(crate) fn refuse(self, missing: Capability) -> PlanError {
        refusal_at(self.g, self.eqn, self.target, missing)
    }
}

/// The refusal of `eqn` on `target` for `missing`, with its op and dtypes read from the equation. For a
/// caller that refuses without planning a kernel and so has no [`Site`].
pub(crate) fn refusal_at(
    g: GraphTables<'_>,
    eqn: &Eqn,
    target: impl Into<TargetId>,
    missing: Capability,
) -> PlanError {
    let target = target.into();
    PlanError::Refused(Box::new(Refusal {
        eqn: eqn.out,
        op: eqn.op.clone(),
        dtypes: RefusalDtypes::of(g, eqn),
        target,
        missing,
    }))
}

impl fmt::Display for RefusalDtypes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, dtype) in self.operands.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{dtype}")?;
        }
        write!(f, " -> {}", self.output)
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} = {} ({}) is refused on {:?}: {}",
            self.eqn,
            self.op.name(),
            self.dtypes,
            self.target,
            self.missing
        )
    }
}

impl std::error::Error for Refusal {}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Capability::CollectiveInStage => f.write_str(
                "a collective is a cut between stage programs, never an equation inside one",
            ),
            Capability::Collective(kind) => write!(
                f,
                "the target set has no link that performs {kind:?} between the devices this value crosses"
            ),
            Capability::ImportedKernelPlanning => f.write_str(
                "the op lowers through imported-kernel production planning, not the generic planner",
            ),
            Capability::DtypeLowering => {
                f.write_str("no kernel lowers this op for these operand and output dtypes")
            }
            Capability::E4m3Storage => {
                f.write_str("no packed E4M3FN storage lowering for this op on this target")
            }
            Capability::ZeroElementE4m3 => {
                f.write_str("no packed E4M3FN kernel over a zero-element operand or output")
            }
            Capability::DispatchSizeOverflow => {
                f.write_str("the packed E4M3FN dispatch size overflows host accounting")
            }
            Capability::DispatchGridLimit {
                threads,
                workgroups_x,
                workgroup_width,
                limit,
            } => write!(
                f,
                "the packed E4M3FN dispatch needs {threads} threads ({workgroups_x} x-workgroups at width {workgroup_width}), above the grid limit {limit}"
            ),
            Capability::DispatchGridAxisLimit {
                axis,
                workgroups,
                limit,
            } => write!(
                f,
                "the dispatch needs {workgroups} workgroups along grid axis {axis}, above the target's limit {limit}"
            ),
            Capability::ExactI32Op => f.write_str("no I32 kernel for this op"),
            Capability::UnfoldedIota => f.write_str(
                "iota must be folded to a computed constant before planning (transform::fold_iota)",
            ),
            Capability::FusedStep(gap) => write!(f, "{gap}"),
            Capability::OperandCount { expected, actual } => {
                write!(f, "the lowering reads {expected} operands, got {actual}")
            }
            Capability::OperandShape {
                value,
                shape,
                expected,
            } => {
                write!(f, "{value} has shape {shape:?}, the lowering needs {expected:?}")
            }
            Capability::OperandLayout { value } => {
                write!(f, "{value} is a strided view the lowering cannot read")
            }
            Capability::LiteralFirstOperand => {
                f.write_str("a literal first operand is outside the planner's operand grammar")
            }
            Capability::IndexOperand => {
                f.write_str("the index operand is not a form the lowering reads")
            }
            Capability::NonLastAxisReduce { axis } => {
                write!(f, "no kernel reduces over axis {axis}, only the last axis")
            }
            Capability::ScatterNonZeroAxis { axis } => {
                write!(f, "no kernel scatters over axis {axis}, only axis 0")
            }
            Capability::AttentionLds { dv, lds_bytes } => write!(
                f,
                "the attention kernel needs {} bytes of workgroup memory for one lane at value dim {dv}, the device has {lds_bytes}",
                4 * (dv + 2)
            ),
            Capability::ScanLds { rows, lds_bytes } => write!(
                f,
                "the scan kernel needs {} bytes of workgroup memory for one state column of {rows} rows, the device has {lds_bytes}",
                4 * rows
            ),
            Capability::BlockTopKWorkgroup {
                lanes,
                max_invocations,
                lds_needed,
                lds_bytes,
            } => write!(
                f,
                "the block top-k kernel needs a {lanes}-lane workgroup and {lds_needed} bytes of workgroup memory, the device has {max_invocations} invocations and {lds_bytes} bytes"
            ),
            Capability::ScratchBytes {
                needed,
                max_buffer_bytes,
            } => write!(
                f,
                "a scratch buffer of {needed} bytes exceeds the device's {max_buffer_bytes}-byte buffer limit"
            ),
            Capability::KernelGen(err) => write!(f, "kernel generator: {err}"),
            Capability::KernelResources(err) => write!(f, "kernel resources: {err}"),
            Capability::PackedLowering { format, op } => {
                write!(f, "no packed lowering admits {format:?} for {op:?}")
            }
            Capability::AieSchedule => f.write_str(
                "the plan is not an AIE sequential schedule (a shipped GPU template or a thread-grid kernel)",
            ),
            Capability::GpuOnly => f.write_str(
                "the lowering reads GPU capabilities and an AIE target has none",
            ),
            Capability::AieLimit(refusal) => write!(f, "AIE limit: {refusal}"),
            Capability::AieAbi(refusal) => write!(f, "AIE shape ABI: {refusal}"),
            Capability::Arithmetic { dtype } => write!(
                f,
                "the target does not compute in {dtype}; computing it narrower changes the equation's semantics"
            ),
            Capability::VpuOpset => {
                f.write_str("no OpenVINO opset op has this op's semantics")
            }
            Capability::LiteralOperand => {
                f.write_str("the lowering reads value operands only, not an inline literal")
            }
        }
    }
}

impl fmt::Display for FusedStepGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusedStepGap::PackedLaneInFloatRegion => {
                f.write_str("a packed I32 lane cannot appear in an f32 fused region")
            }
            FusedStepGap::F32LiteralInExactRegion => {
                f.write_str("an f32 literal cannot appear in an exact I32 fused region")
            }
            FusedStepGap::OpInFloatRegion(op) => {
                write!(f, "{op:?} cannot appear in an f32 fused region")
            }
            FusedStepGap::OpInExactRegion(op) => {
                write!(f, "{op:?} cannot appear in an exact I32 fused region")
            }
        }
    }
}

impl fmt::Display for StorageGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageGap::E4m3RawBytes => f.write_str(
                "generic executor tensors carry neither authoritative raw bytes nor packed-row storage",
            ),
            StorageGap::Bf16PackedLanes => f.write_str(
                "an executor tensor is four-byte f32 words, not the packed u32 lanes a BF16 consumer reads",
            ),
            StorageGap::F16PackedLanes => f.write_str(
                "a packed-F16 dense-contraction weight must be a const read only by dense contractions, \
                 not a computed value, an output or a buffer another kernel reads natively",
            ),
            StorageGap::E4m3PackedLanes => f.write_str(
                "a packed-E4M3 row-gather table must be a const read only by dense row gathers, \
                 not a computed value, an output or a buffer another kernel reads as raw bytes",
            ),
            StorageGap::VpuDtype => f.write_str(
                "an Intel NPU region carries only F16, F32 and I32 values",
            ),
        }
    }
}
