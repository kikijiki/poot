//! Typed planner refusals (ADR-0104 decision 2): a refusal names the equation, the op, the dtypes, the
//! target and the missing capability. It is never free text, so a caller matches on it instead of
//! restating the planner's gaps.

use std::fmt;

use poot_kernelgen::BodyLimits;
use poot_target::Backend;

use crate::*;

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
    pub target: Backend,
    /// The capability the target lacks for this equation.
    pub missing: Capability,
}

/// The operand and output dtypes of a refused equation. A literal operand carries its scalar's dtype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefusalDtypes {
    pub operands: Vec<DType>,
    pub output: DType,
}

/// A capability a target lacks, with the parameters that make the gap checkable.
///
/// Each variant is a real gap in the planner's lowering table or a precondition of a lowering that the
/// equation violates. None is roadmap wording: the variant says what is missing, and the enclosing
/// [`Refusal`] says for which equation and target.
#[derive(Clone, Debug, PartialEq)]
pub enum Capability {
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
        threads: usize,
        workgroups_x: usize,
        workgroup_width: usize,
        limit: usize,
    },
    /// The op has no exact I32 lowering (reduction, row fusion, a division without total exact
    /// semantics, a unary that is not an exact integer primitive, a remainder by a literal zero).
    ExactI32Op,
    /// The equation is an unfolded `Iota`. It is a pure compile-time constant: `transform::fold_iota`
    /// folds it to a `Storage::Computed` graph constant before planning (R466-019), so no kernel exists.
    UnfoldedIota,
    /// A step of a fused region cannot be lowered in the region's arithmetic.
    FusedStep(FusedStepGap),
    /// The equation has the wrong number of operands for its lowering.
    OperandCount { expected: usize, actual: usize },
    /// An operand's shape violates the lowering's precondition: `shape` is the offending value's actual
    /// shape, `expected` the shape the lowering required of it.
    OperandShape {
        value: ValueId,
        shape: Vec<usize>,
        expected: Vec<usize>,
    },
    /// A literal in the first operand position, which the planner's operand grammar does not read.
    LiteralFirstOperand,
    /// The index operand is not a form the lowering reads.
    IndexOperand,
    /// A reduction over an axis other than the last has no kernel on this target.
    NonLastAxisReduce { axis: usize },
    /// `Scatter` has no kernel over an axis other than 0 on any target (Card 546a, R-546-4): the
    /// contract admits no host fallback, so this equation is refused outright rather than routed to
    /// a host step.
    ScatterNonZeroAxis { axis: usize },
    /// The attention kernel holds one head row in workgroup memory, so the head dim is capped.
    HeadDimExceedsLdsCap { head_dim: usize, lds_cap: usize },
    /// The synthesized flash-prefill kernel has no attention-logit softcap.
    AttentionSoftcap,
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
        format: poot_quant::format::WeightFormat,
        op: crate::packed_block_float::PackedKernelOpKind,
    },
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
    /// Exact I32 values compile, but production bind, replay and typed readback are not wired.
    ExactI32BindReplay,
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
    target: Backend,
    limits: BodyLimits,
}

impl<'a> Site<'a> {
    pub(crate) fn new(
        g: GraphTables<'a>,
        eqn: &'a Eqn,
        target: Backend,
        limits: BodyLimits,
    ) -> Self {
        Self {
            g,
            eqn,
            target,
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
    target: Backend,
    missing: Capability,
) -> PlanError {
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
            "v{} = {} ({}) is refused on {:?}: {}",
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
            Capability::ExactI32Op => f.write_str("no exact I32 lowering for this op"),
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
                write!(f, "v{value} has shape {shape:?}, the lowering needs {expected:?}")
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
            Capability::HeadDimExceedsLdsCap { head_dim, lds_cap } => write!(
                f,
                "head dim {head_dim} exceeds the workgroup-memory cap {lds_cap} of the attention kernel"
            ),
            Capability::AttentionSoftcap => {
                f.write_str("the synthesized flash-prefill kernel has no attention-logit softcap")
            }
            Capability::KernelGen(err) => write!(f, "kernel generator: {err}"),
            Capability::KernelResources(err) => write!(f, "kernel resources: {err}"),
            Capability::PackedLowering { format, op } => {
                write!(f, "no packed lowering admits {format:?} for {op:?}")
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
            StorageGap::ExactI32BindReplay => {
                f.write_str("production bind, replay and typed readback are not wired")
            }
        }
    }
}
