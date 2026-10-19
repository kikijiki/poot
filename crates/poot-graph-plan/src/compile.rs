//! The one graph-to-program step (ADR-0100 decision 1, ADR-0104; card 532).
//!
//! [`compile`] takes a traced graph and a [`Target`] and returns a [`Program`], or the typed error that says
//! why the target cannot run it. It runs the graph passes in one fixed order, then `crate::plan_graph`,
//! which owns the planning decisions (kernel choice, launch grids, strided views, value storage and the slot
//! schema). No caller sequences passes or planner entries itself; an executor loads the program.
//!
//! Two properties are checked here, not by callers:
//!
//! - **Rewrite legality.** After every pass, each equation the pass rewrote is planned for the
//!   target. A rewrite the target refuses is reverted to the decomposition it replaced, so an optimization
//!   never turns a plannable graph into an unplannable one (a BF16 pointwise chain stays decomposed, since
//!   no target has a BF16 fused-region body). An equation the target cannot run in the graph as traced is
//!   still refused.
//! - **Published numerics (ADR-0101 decision 2).** Every pass that changed the graph records its numerics
//!   declaration. The program publishes the strongest of those declarations and of the traced graph's own
//!   baseline as its [`Tier2Class`], and compile fails closed when the output graph implies a looser class
//!   than the record allows (a pass that reassociates without recording it).

use crate::buffer_plan::plan_buffers;
use crate::dtype_widen::{
    fold_dense_bf16_row_gathers, fold_dense_contractions, lower_nonlast_reduces,
    widen_mismatched_matmul_dtypes,
};
use crate::graph_validation::ValidationPacketPlan;
use crate::planner::{plan_eqn_compiled, plan_graph, plan_vpu_region};
use crate::production_plan::Submission;
use crate::state_commit::{StateCommitPlan, state_commit_from_plans};
#[cfg(feature = "test-support")]
use crate::types::{DispatchPlanError, ScratchDecl};
use crate::types::{GraphStorage, ValueStorage};
#[cfg(feature = "test-support")]
use crate::{Binding, DispatchPlan};
use poot_codegen::VpuRegion;
use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};

use poot_graph_ir::analysis::{
    NumericalRewriteDecline, NumericsError, NumericsProperty, Tier2Class, implied_numerics,
};
use poot_graph_ir::{
    BoundaryId, ContractionRows, EditError, Eqn, Export, Graph, IndexRange, OpKind, Operand, Slot,
    ValueId, ValueTable, value_ranges,
};
use poot_target::{
    AieCaps, AieTargetId, Backend, DeviceCaps, Queried, TargetId, VpuCaps, VpuTargetId,
};
use poot_tensor::DType;

use crate::kernel_mapping::{AieAbiRefusal, aie_region_recipe};
use crate::passes::{
    self as transform, GraphBudget, LegalizeError, PackedDequantProductionError, PassDeclaration,
    declared,
};
pub use crate::passes::{LoraError, LoraPlan};
use crate::{
    BufferPlan, Dispatch, KernelChoice, Plan, PlanError, Refusal, compute_views, device_validation,
    planner,
};
use poot_codegen::AieLinkRecipe;
use poot_kernel_ir::Body;

mod package;
use package::package;
pub(crate) use package::whole_graph_region;
pub use package::{ArtifactGranularity, Region};

/// What a graph is compiled for: the target and the device's measured capabilities (card 522). Passes and
/// the planner read it; tracers never see it.
///
/// One enum, so a target's identity and its capability descriptor can never disagree: a GPU backend
/// carries [`DeviceCaps`], an AIE target carries [`AieCaps`], and no value pairs one with the other's
/// (POOT-1117). A consumer that needs GPU capabilities matches [`Target::Gpu`]; the planner answers a GPU-only
/// lowering on an AIE target with [`Capability::GpuOnly`].
///
/// An AIE target with GPU capabilities, and a GPU target with an AIE tag, are not constructible:
///
/// ```compile_fail,E0308
/// use poot_graph_plan::Target;
/// use poot_target::{AieTargetId, DeviceCaps};
/// let _ = Target::Aie {
///     target: AieTargetId::Xdna2,
///     caps: DeviceCaps::wgpu_rdna3_igpu(),
/// };
/// ```
///
/// ```compile_fail,E0308
/// use poot_graph_plan::Target;
/// use poot_target::{AieCaps, AieTargetId, Backend, Queried};
/// let _ = Target::Gpu {
///     backend: AieTargetId::Xdna2,
///     caps: AieCaps {
///         target: AieTargetId::Xdna2,
///         tile_local_bytes: Queried::Unknown,
///         host_bindings: Queried::Unknown,
///         instruction_bytes: Queried::Unknown,
///         memory_budget_bytes: Queried::Unknown,
///     },
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    /// A GPU lowering backend and what its device measured.
    Gpu {
        /// The lowering backend the program is generated for.
        backend: Backend,
        /// What the device measured: the limits and features the planner consults.
        caps: DeviceCaps,
    },
    /// An AMD XDNA target and the limits its runtime reports (unmeasured until POOT-641).
    Aie {
        /// The XDNA part the program is generated for.
        target: AieTargetId,
        /// The limits the part's runtime reports.
        caps: AieCaps,
    },
    /// An Intel NPU target and the capabilities its runtime reports: the vendor graph compiler compiles the
    /// OpenVINO graph poot emits for a region of equations.
    Vpu {
        /// The Intel NPU part the program is generated for.
        target: VpuTargetId,
        /// The capabilities the part's runtime reports.
        caps: VpuCaps,
    },
}

impl Target {
    /// The device memory budget its runtime measured, if it did: GPU memory for a GPU target, the host
    /// buffer-object budget for an AIE target. `None` is unknown, never unlimited.
    pub fn memory_budget_bytes(&self) -> Option<u64> {
        match self {
            Target::Gpu { caps, .. } => caps.memory_budget_bytes,
            Target::Aie { caps, .. } => match caps.memory_budget_bytes {
                Queried::Known(bytes) => Some(bytes),
                Queried::Unknown => None,
            },
            Target::Vpu { caps, .. } => match caps.memory_budget_bytes {
                Queried::Known(bytes) => Some(bytes),
                Queried::Unknown => None,
            },
        }
    }

    /// This target's identity, without its limits.
    pub const fn id(&self) -> TargetId {
        match self {
            Target::Gpu { backend, .. } => TargetId::Gpu(*backend),
            Target::Aie { target, .. } => TargetId::Aie(*target),
            Target::Vpu { target, .. } => TargetId::Vpu(*target),
        }
    }

    /// How big this target's compiled artifacts are. A GPU target loads one artifact per planned kernel. An
    /// XDNA target holds a bounded number of hardware contexts, one per loaded artifact, so its artifacts
    /// are regions of planned equations and [`compile`] groups them (see [`Region`]). An Intel NPU's vendor
    /// graph compiler takes the whole graph as one artifact.
    pub const fn granularity(&self) -> ArtifactGranularity {
        match self {
            Target::Gpu { .. } => ArtifactGranularity::Kernel,
            Target::Aie { caps, .. } => ArtifactGranularity::Region {
                limits: caps.region,
            },
            Target::Vpu { .. } => ArtifactGranularity::Graph,
        }
    }
}

/// Typed execution choices a caller makes for one compile. Never an environment variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileOptions {
    /// Executor submission mode. Compilation produces the same plans for either mode.
    pub execution: Submission,
    /// What compilation may construct and load. There is no default: every caller states its policy.
    pub limits: CompileLimits,
    /// The LoRA mode transform, applied to the traced graph before any other pass (POOT-579). `None`
    /// compiles the graph as traced.
    pub lora: Option<LoraPlan>,
}

impl CompileOptions {
    /// Options for `execution` under `limits`, with no mode transform.
    pub const fn new(execution: Submission, limits: CompileLimits) -> Self {
        Self {
            execution,
            limits,
            lora: None,
        }
    }

    /// These options with the LoRA transform of `plan`.
    pub fn with_lora(mut self, plan: LoraPlan) -> Self {
        self.lora = Some(plan);
        self
    }
}

/// What one compile may construct, and the largest artifact it may load: every field is a finite, nonzero
/// cap, so no compile is unbounded and none is configured by environment.
///
/// - `max_intermediate_values` and `max_intermediate_eqns` cap the value table and the equation list of
///   every graph a pass produces, the traced input and the final program included. A pass that would
///   append past either returns [`CompileError::Expansion`] before the append.
/// - `max_graph_literal_bytes` and `max_graph_owned_bytes` cap the inline scalar literals and the
///   requested owned storage (records, names, operand lists) of the graphs a pass builds through a
///   tracer [`Builder`](poot_graph_ir::Builder): the same two [`TraceLimits`](poot_graph_ir::TraceLimits)
///   fields a model trace states, applied to a graph a pass reopens.
/// - `max_body_instructions` and `max_body_locals` cap one generated kernel body (statements plus block
///   terminators, and locals with parameters). An over-limit request is a typed planner refusal carrying
///   `poot_kernelgen::KernelGenError::BodyLimit`.
/// - `max_artifact_bytes` caps one compiled kernel artifact an executor loads or caches; see
///   [`Program::limits`]. It bounds the artifact's bytes, not the external compiler's memory or time.
///
/// Wall time is telemetry, not a limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileLimits {
    /// Most values one graph's value table may hold, for every graph a pass produces.
    pub max_intermediate_values: NonZeroUsize,
    /// Most equations one graph may hold, for every graph a pass produces.
    pub max_intermediate_eqns: NonZeroUsize,
    /// Most bytes of inline scalar literals a pass-built graph may carry.
    pub max_graph_literal_bytes: NonZeroU64,
    /// Most bytes of owned storage (records, names, operand lists) a pass-built graph may request.
    pub max_graph_owned_bytes: NonZeroU64,
    /// Most instructions (statements plus block terminators) in one generated kernel body.
    pub max_body_instructions: NonZeroUsize,
    /// Most locals (parameters included) in one generated kernel body.
    pub max_body_locals: NonZeroUsize,
    /// Most bytes of one compiled kernel artifact an executor loads or caches.
    pub max_artifact_bytes: NonZeroU64,
}

impl CompileLimits {
    /// The limits a production compile runs under: a few orders above the largest graph and kernel a real
    /// model produces, and far below what exhausts a host.
    pub const STANDARD: Self = Self {
        max_intermediate_values: NonZeroUsize::new(1 << 24).unwrap(),
        max_intermediate_eqns: NonZeroUsize::new(1 << 24).unwrap(),
        max_graph_literal_bytes: NonZeroU64::new(256 << 20).unwrap(),
        max_graph_owned_bytes: NonZeroU64::new(4 << 30).unwrap(),
        max_body_instructions: NonZeroUsize::new(1 << 22).unwrap(),
        max_body_locals: NonZeroUsize::new(1 << 18).unwrap(),
        max_artifact_bytes: NonZeroU64::new(512 << 20).unwrap(),
    };

    /// The kernel-body half, as the generator takes it.
    pub(crate) fn body(&self) -> poot_kernelgen::BodyLimits {
        poot_kernelgen::BodyLimits {
            max_instructions: self.max_body_instructions,
            max_locals: self.max_body_locals,
        }
    }
}

/// What a graph is limited in, for [`ExpansionError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GraphResource {
    /// The graph's value table.
    #[error("values")]
    Values,
    /// The graph's equation list.
    #[error("equations")]
    Eqns,
}

/// A compile stage would have grown a graph past [`CompileLimits`]. `held` is what the graph held when the
/// stage was refused, `attempted` what it would have held; nothing past `held` was constructed unless the
/// stage was sized conservatively ahead of time (a pass that cannot count its own appends), in which case
/// `attempted` is the checked upper bound it was refused on.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "{stage}: the graph's {resource} would reach {attempted} (holding {held}), over the limit of {limit}"
)]
pub struct ExpansionError {
    /// The compile stage (pass name) that attempted the growth.
    pub stage: &'static str,
    /// Which of the graph's tables would have overflowed.
    pub resource: GraphResource,
    /// How many entries of `resource` the graph held when the stage began.
    pub held: usize,
    /// How many entries the stage would have left the graph holding.
    pub attempted: usize,
    /// The configured cap on `resource`, from [`CompileLimits`].
    pub limit: usize,
}

/// Why [`compile`] produced no program. Each cause is boxed (spec 384), so a `Result` carrying one stays
/// small; the `From` impls below keep `?` working.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    /// The planner cannot run the graph: a typed [`Refusal`] naming the equation and missing capability
    /// ([`PlanError::Refused`]), or a malformed graph.
    #[error(transparent)]
    Plan(Box<PlanError>),
    /// A pass produced ops that imply a looser numerics class than the program would publish.
    #[error(transparent)]
    Numerics(Box<NumericsError>),
    /// `legalize` (card 523a) found a constant over the target's `DeviceCaps::max_buffer_bytes` that
    /// neither the host nor the split rewrite can bring under it (SC-002).
    #[error(transparent)]
    Legalize(Box<LegalizeError>),
    /// The LoRA transform could not adapt the graph (POOT-579): a projection the plan names is absent,
    /// or a contraction the correction cannot be inserted after.
    #[error(transparent)]
    Lora(Box<LoraError>),
    /// A `PackedDequant` the contraction claim could not take escapes into ordinary execution (Card
    /// 355's gate, run by `compile` since card 642): no target materializes a packed weight outside a
    /// claimed contraction.
    #[error(transparent)]
    PackedDequant(Box<PackedDequantProductionError>),
    /// A pass would have grown the graph past [`CompileLimits`]: nothing past the limit was appended and no
    /// program exists.
    #[error(transparent)]
    Expansion(Box<ExpansionError>),
    /// An `I32 -> F32` cast whose operand the graph does not prove exactly representable (POOT-726).
    #[error(transparent)]
    CastRange(Box<CastRangeError>),
    /// An `IndexedMatMul` whose expert ids the graph does not prove inside `[0, E)` (POOT-1119).
    #[error(transparent)]
    ExpertIds(Box<ExpertIdsError>),
    /// A pass produced a graph the edit check refused: a defect of the pass, never of the input graph
    /// (every `Graph` is well formed on its own).
    #[error(transparent)]
    Edit(Box<EditError>),
}

/// A `Cast(I32 -> F32)` whose operand is not proven inside `[-2^24, 2^24]`, the integers an `f32`
/// holds exactly (POOT-726). The ranges are [`poot_graph_ir::value_ranges`]: what the equations and the
/// declared slot ranges prove, never a caller's claim. A graph that wants the cast declares the range of
/// its slot, clamps with [`poot_graph_ir::Builder::guard_index_bounds`], or does not widen the word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "equation {cast}: Cast I32 -> F32 of {operand}, whose proven range is {proven}, not inside the \
     exact f32 integers {exact}"
)]
pub struct CastRangeError {
    /// The cast equation's output value.
    pub cast: ValueId,
    /// The I32 value it widens.
    pub operand: ValueId,
    /// What the graph proves about `operand`.
    pub proven: IndexRange,
    /// What the cast needs: [`IndexRange::F32_EXACT`].
    pub exact: IndexRange,
}

/// Refuse every `Cast(I32 -> F32)` of `g` whose operand's proven range leaves the exact f32 integers.
/// The device casts convert without checking, so a word past `2^24` would round silently; the oracle
/// faults at run time on the same words, and this refuses the graph before any of them exist.
fn refuse_unproven_i32_casts(g: &Graph) -> Result<(), CastRangeError> {
    let facts = value_ranges(g);
    for eqn in g.eqns() {
        let (OpKind::Cast { to: DType::F32 }, [Operand::Value(operand)]) =
            (&eqn.op, eqn.inputs.as_slice())
        else {
            continue;
        };
        if g.aval(*operand).dtype != DType::I32 {
            continue;
        }
        let proven = facts[*operand].unwrap_or(IndexRange::I32);
        if !proven.within(&IndexRange::F32_EXACT) {
            return Err(CastRangeError {
                cast: eqn.out,
                operand: *operand,
                proven,
                exact: IndexRange::F32_EXACT,
            });
        }
    }
    Ok(())
}

/// Why an indexed contraction's ids are not admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertIdsGap {
    /// The proven range of the ids leaves `[0, E)`.
    OutOfRange,
    /// The range holds only because the index guard whose condition is `guard` replaced the lanes
    /// outside it, and no declared validation output observes that guard: a bad id would turn silently
    /// into a valid-looking expert.
    UnobservedGuard {
        /// The guard's condition value; nothing in the declared validation outputs reads it.
        guard: ValueId,
    },
}

/// An indexed contraction (`IndexedMatMul`, or the packed `PackedContraction` over an indexed stack) whose
/// per-row expert ids are not proven in `[0, E)` (POOT-1119). The ids are proven by
/// [`poot_graph_ir::value_ranges`] (an `ArgTopK`, a declared slot or weight range, a `Gather` of a ranged
/// table, an index guard) and a range that rests on a guard's clamp additionally needs that guard's witness
/// declared as a validation output. The device kernels do not check the id they read, and `E` is the
/// weight's own leading extent (or the stack's declared expert count): no caller supplies it.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "equation {eqn}: {op} ids {ids} are proven in {proven}, not inside [0, {experts}), the experts of its \
     weight: {gap:?}"
)]
pub struct ExpertIdsError {
    /// The contraction equation's output value.
    pub eqn: ValueId,
    /// The contraction's op name.
    pub op: String,
    /// Its ids operand.
    pub ids: ValueId,
    /// `E`: the leading extent of the stacked weight, or the packed stack's declared experts.
    pub experts: usize,
    /// What the graph proves about `ids`.
    pub proven: IndexRange,
    /// What is missing.
    pub gap: ExpertIdsGap,
}

/// Refuse every indexed contraction of `g` whose ids are not proven in `[0, E)` (POOT-1119), by the ranges
/// and guard witnesses the graph itself carries.
///
/// The ids consumers of an input graph are `IndexedMatMul` (`[x, w, ids]`, `E` the extent of `w`) and
/// `PackedContraction { rows: Indexed { experts } }` (`[x, ids, carriers..]`). A `ContractionEpilogue` over an
/// indexed head is built by the fuse pass after this check from one of those two, never traced.
fn refuse_unproven_expert_ids(g: &Graph) -> Result<(), ExpertIdsError> {
    let facts = value_ranges(g);
    let producers = transform::producer_map(g);
    for eqn in g.eqns() {
        // `g.validate()` (run by `compile` before the adapters and this check, and by the adapter itself)
        // infers every equation, and `OpKind::infer` gives each of these operands a rank (w rank 3, ids
        // rank 1), so none of them is a literal.
        let (ids, experts) = match (&eqn.op, eqn.inputs.as_slice()) {
            (OpKind::IndexedMatMul, [_, Operand::Value(weight), Operand::Value(ids)]) => {
                (*ids, g.aval(*weight).shape.first().copied().unwrap_or(0))
            }
            (
                OpKind::PackedContraction {
                    rows: ContractionRows::Indexed { experts },
                    ..
                },
                [_, Operand::Value(ids), ..],
            ) => (*ids, *experts),
            (OpKind::IndexedMatMul | OpKind::PackedContraction { .. }, operands) => unreachable!(
                "equation {}: {} has operands {operands:?}; infer gives its weight and ids a rank, \
                 so a literal there fails validation",
                eqn.out,
                eqn.op.name()
            ),
            _ => continue,
        };
        let proven = facts[ids].unwrap_or(IndexRange::I32);
        let refusal = |gap| ExpertIdsError {
            eqn: eqn.out,
            op: eqn.op.name(),
            ids,
            experts,
            proven,
            gap,
        };
        if !IndexRange::below(experts).is_some_and(|valid| proven.within(&valid)) {
            return Err(refusal(ExpertIdsGap::OutOfRange));
        }
        if let Some(guard) = transform::first_unobserved_clamp(g, &producers, &facts, ids) {
            return Err(refusal(ExpertIdsGap::UnobservedGuard { guard }));
        }
    }
    Ok(())
}

impl From<ExpertIdsError> for CompileError {
    fn from(error: ExpertIdsError) -> Self {
        CompileError::ExpertIds(Box::new(error))
    }
}

impl From<PlanError> for CompileError {
    fn from(error: PlanError) -> Self {
        CompileError::Plan(Box::new(error))
    }
}

impl From<NumericsError> for CompileError {
    fn from(error: NumericsError) -> Self {
        CompileError::Numerics(Box::new(error))
    }
}

impl From<LegalizeError> for CompileError {
    fn from(error: LegalizeError) -> Self {
        CompileError::Legalize(Box::new(error))
    }
}

impl From<LoraError> for CompileError {
    fn from(error: LoraError) -> Self {
        CompileError::Lora(Box::new(error))
    }
}

impl From<ExpansionError> for CompileError {
    fn from(error: ExpansionError) -> Self {
        CompileError::Expansion(Box::new(error))
    }
}

impl From<EditError> for CompileError {
    fn from(error: EditError) -> Self {
        CompileError::Edit(Box::new(error))
    }
}

impl From<CastRangeError> for CompileError {
    fn from(error: CastRangeError) -> Self {
        CompileError::CastRange(Box::new(error))
    }
}

impl From<PackedDequantProductionError> for CompileError {
    fn from(error: PackedDequantProductionError) -> Self {
        CompileError::PackedDequant(Box::new(error))
    }
}

/// One slot the executor writes every step: the value, its slot kind and the storage the plan chose for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotSchema {
    /// The value the executor writes into the slot.
    pub value: ValueId,
    /// Which slot kind the value is written through.
    pub slot: Slot,
    /// The storage the plan chose for the value.
    pub storage: ValueStorage,
}

/// A graph compiled for one [`Target`]: the optimized graph and every decision an executor needs to load
/// and replay it. Built only by [`compile`].
#[derive(Debug)]
pub struct Program {
    graph: Graph,
    /// One plan per equation of `graph`, index-aligned with `graph.eqns`.
    plans: Vec<Plan>,
    /// The kernel the planner chose for each equation, index-aligned with `plans`. The executor never
    /// reads it; [`Program::kernel_choice`] (test-support) is its one reader.
    #[cfg_attr(
        not(any(test, feature = "test-support")),
        expect(
            dead_code,
            reason = "read only by the test-support `kernel_choice` accessor"
        )
    )]
    choices: Vec<KernelChoice>,
    storage: GraphStorage,
    slots: Vec<SlotSchema>,
    /// The tier-2 comparison class this program publishes: the strongest of the traced graph's baseline
    /// and the declarations of the passes that changed it.
    numerics: Tier2Class,
    /// The passes that changed the graph, in the order they ran.
    passes: Vec<&'static str>,
    numerical_declines: Vec<NumericalRewriteDecline>,
    /// The target this program was compiled for (Card 546a): the contract's `add_entry` refuses a
    /// program compiled for a different target with a typed `LoadError::TargetMismatch`.
    target: Target,
    /// The executor submission mode this program records (Card 546a): capture-and-replay (the
    /// contract's only admitted mode) or eager (`Submission::Eager` survives for the pre-contract
    /// executors named in the).
    submission: Submission,
    /// The native state-commit plan, derived once from this program's own plans
    /// (`state_commit_from_plans`) rather than re-planning the graph a second time.
    state_commit: StateCommitPlan,
    /// The validation packet layout and source plan (ADR-0101 decision 2's validated session, Card
    /// 546a): a zero-lane packet for a graph with no validation outputs, checked
    /// by the contract before any readback the same way either way.
    validation: ValidationPacketPlan,
    /// The regions `package` grouped the planned equations into: empty for a target whose artifacts are
    /// kernels, otherwise a partition of the program's dispatching equations into contiguous runs.
    regions: Vec<Region>,
    /// The Intel NPU graph the program's one region runs: every equation lowered to an opset node, with the
    /// region's reads as its arguments and its writes as its results. `None` for every other target.
    vpu: Option<VpuRegion>,
    /// Lifetimes and arena-slot assignment for this program's locally-computed values (Card 547b),
    /// derived from `plans`, `storage`, `state_commit`, `validation` and `regions` once here.
    buffer_plan: BufferPlan,
    /// The limits this program was compiled under; an executor loads its kernels under the same
    /// `max_artifact_bytes`.
    limits: CompileLimits,
}

impl Program {
    /// The compiled graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The target this program was compiled for.
    pub fn target(&self) -> &Target {
        &self.target
    }

    /// The executor submission mode this program records.
    pub fn submission(&self) -> Submission {
        self.submission
    }

    /// The [`CompileLimits`] this program was compiled under. An executor reads
    /// `max_artifact_bytes` from here when it loads the program's kernels.
    pub fn limits(&self) -> &CompileLimits {
        &self.limits
    }

    /// The native state-commit plan (donation and explicit two-phase copies), derived from this
    /// program's own plans.
    pub fn state_commit(&self) -> &StateCommitPlan {
        &self.state_commit
    }

    /// The validation packet layout and source plan (a zero-lane packet when the graph has no
    /// validation outputs).
    pub fn validation(&self) -> &ValidationPacketPlan {
        &self.validation
    }

    /// Lifetimes and arena-slot assignment for this program's locally-computed values (Card 547b):
    /// the engine allocates one device buffer per arena slot from this, instead of one per value.
    pub fn buffer_plan(&self) -> &BufferPlan {
        &self.buffer_plan
    }

    /// The regions of the program: each is linked into one artifact and run in one invocation. Empty for a
    /// target whose artifacts are kernels ([`Target::granularity`]), where every dispatch is its own.
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// The Intel NPU graph the program's one region runs, ready for [`poot_codegen::emit_vpu`]; `None` for a
    /// program compiled for any other target.
    pub fn vpu_region(&self) -> Option<&VpuRegion> {
        self.vpu.as_ref()
    }

    /// What `region` is linked from: its link recipe and, in tile order, the body each tile runs. The
    /// recipe is derived from the same plans the bodies are, so the two cannot disagree.
    pub fn region_link(
        &self,
        region: &Region,
    ) -> Result<(AieLinkRecipe, Vec<&Body>), AieAbiRefusal> {
        let Target::Aie { target, .. } = self.target else {
            return Err(AieAbiRefusal::NotAnAieProgram);
        };
        let recipe = aie_region_recipe(&self.graph, &self.plans, region, target)?;
        let bodies = region
            .members()
            .map(|index| {
                self.plans[index]
                    .single()
                    .map(Dispatch::body)
                    .ok_or_else(|| AieAbiRefusal::NotOneDispatch {
                        eqn: self.graph.eqns()[index].out,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((recipe, bodies))
    }

    /// Where each exported boundary lives after optimization, CSE and renumbering: the compiled
    /// graph's [`Graph::exports`]. Several boundaries may resolve to one value.
    pub fn exports(&self) -> impl Iterator<Item = Export> + '_ {
        self.graph.exports()
    }

    /// The first export that is, after planning, a view (alias, reshape, strided view) of a carried-state
    /// input, with that input: a later stage reading it would see the value the stage's own commit
    /// wrote, not the one the uncut graph reads. `None` when every export is a computed or bound value.
    pub fn export_aliasing_state(&self) -> Option<(BoundaryId, ValueId)> {
        let mut root: ValueTable<ValueId> = self.graph.values().ids().collect();
        for (eqn, plan) in self.planned() {
            match plan {
                Plan::Alias(src) | Plan::View { src, .. } => root[eqn.out] = root[*src],
                Plan::Dispatches(_) | Plan::Graph => {}
            }
        }
        self.graph.exports().find_map(|export| {
            let source = root[export.value];
            (self.graph.meta(source).storage == poot_graph_ir::Storage::State)
                .then_some((export.id, source))
        })
    }

    /// The input binder of the compiled graph that consumes boundary `id` (the keyed slot
    /// [`BoundaryId::slot_key`] names), if this program has one.
    pub fn boundary_input(&self, id: BoundaryId) -> Option<ValueId> {
        let key = id.slot_key();
        self.graph
            .inputs()
            .iter()
            .copied()
            .find(|&input| self.graph.meta(input).slot_key() == Some(&key))
    }

    /// Each equation of [`Program::graph`] with its plan, in execution order.
    pub fn planned(&self) -> impl Iterator<Item = (&Eqn, &Plan)> {
        self.graph.eqns().iter().zip(&self.plans)
    }

    /// The kernel the planner chose for `eqn`, an equation of [`Program::graph`]: a generated request, an
    /// imported template, or no dispatch at all. Panics if `eqn` is not an equation of this program's
    /// graph.
    #[cfg(any(test, feature = "test-support"))]
    pub fn kernel_choice(&self, eqn: &Eqn) -> &KernelChoice {
        let index = self
            .graph
            .eqns()
            .iter()
            .position(|candidate| candidate.out == eqn.out)
            .expect("kernel_choice: the equation belongs to this program's graph");
        &self.choices[index]
    }

    /// Rebind every dispatch of equation `index`'s plan through `rebind`, as the executor's load-time
    /// refusal rows need a plan the planner would never emit (ADR-0113: gated, no production caller). The
    /// new plan declares the `scratch` buffers and passes [`DispatchPlan::new`] like any other.
    #[cfg(feature = "test-support")]
    pub fn rebind_dispatches(
        &mut self,
        index: usize,
        scratch: Vec<ScratchDecl>,
        rebind: impl Fn(&Binding) -> Binding,
    ) -> Result<(), DispatchPlanError> {
        let dispatches = self.plans[index]
            .dispatches()
            .iter()
            .map(|dispatch| dispatch.clone().with_binding(rebind(dispatch.binding())))
            .collect();
        self.plans[index] = Plan::Dispatches(DispatchPlan::new(dispatches, scratch)?);
        Ok(())
    }

    /// Every value's storage, as the plan chose it.
    pub fn storage(&self) -> &GraphStorage {
        &self.storage
    }

    /// The per-step slots and their storage.
    pub fn slots(&self) -> &[SlotSchema] {
        &self.slots
    }

    /// Numerical rewrites withheld during canonicalization, referring to the traced value ids.
    /// These are optimization diagnostics: compilation keeps the original arithmetic instead.
    pub fn numerical_declines(&self) -> &[NumericalRewriteDecline] {
        &self.numerical_declines
    }

    /// The tier-2 comparison class this program publishes (ADR-0101 decision 2): the strongest of the
    /// traced graph's baseline and the declarations of the passes that changed it. [`compile`] has
    /// already checked that the compiled graph implies nothing looser. A device result is compared
    /// with the oracle within [`Tier2Class::tolerance`] of it.
    pub fn numerics_class(&self) -> Tier2Class {
        self.numerics
    }

    /// The passes that changed the graph, in the order they ran (card 535a: a caller or test reads this
    /// to see whether a pass actually took effect - never by re-deriving it locally).
    pub fn passes(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.passes.iter().copied()
    }
}

/// Compile `g` for `target`: canonicalize, CSE and iota folding, rope and flash-attention recognition, DCE,
/// the packed-contraction and row-gather claims and their escape gate (cards 642/545a: every
/// `PackedDequant` is claimed into a `PackedContraction`/`PackedRowGather`, or the graph is refused), buffer-limit legalization, then the BF16-packed-weight recognition, non-last-axis reduce legalization
/// and backend dtype retyping (Card 534a), then fusion, then `crate::plan_graph`, which makes every
/// kernel choice (Card 557). The contraction epilogue fuse rule runs right before `fuse`.
///
/// Legalization (card 523a: `legalize`, reading the GPU target's caps to host or split a constant the target
/// cannot allocate as one device buffer) and Card 534a's step (`poot_graph_plan::prepare_target_graph`'s
/// guts, inlined here so every sub-pass is individually tracked and numerics-checked) join the pipeline
/// between DCE and fusion, each its own `PassDeclaration` row like every other pass here
/// (`passes::declared`), checked by `revert_unplannable_rewrites`/`PassDeclaration::verify` the same way.
/// `fold_dense_contractions` runs before `legalize`: an oversized weight is split as the `DenseContraction`
/// that reads it, so each chunk is the contraction the planner lowers (a BF16 `[K, N]` matmul over a
/// transposed chunk has no lowering to probe). `legalize` runs before `fold_dense_bf16_row_gathers`: it can
/// retag a `Gather` into a `Slot::TokenEmbed` input, and that fold pattern-matches a `Gather` specifically,
/// so a table `legalize` already hosted must not still look like one to it. Card 534a's step is the one
/// preparation every target needs before planning: no executor sequences its own copy of it any more.
///
/// After planning, a target whose artifacts are regions (`Target::granularity`) has its planned equations grouped
/// into regions by `package` before the buffer plan runs; a kernel target has none.
///
/// Every target runs every pass, in this one order, on every graph, except the GPU-limit stages (flash
/// prefill cap, legalization, matmul dtype widening, the contraction epilogue probe and the oversized-constant
/// re-check): they read [`DeviceCaps`], so an AIE target does not run them.
pub fn compile(
    g: &Graph,
    target: &Target,
    options: &CompileOptions,
) -> Result<Program, CompileError> {
    let budget = GraphBudget::of(&options.limits);
    budget.admit("compile input", g)?;
    // The LoRA mode transform comes first: it rewrites the traced graph, and everything after it
    // (the baseline numerics included) sees the adapted graph as the one it was handed.
    let adapted;
    let g = match &options.lora {
        Some(plan) => {
            adapted = adapt(g, plan, &budget)?;
            &adapted
        }
        None => g,
    };
    // Every `I32 -> F32` cast is proven exact from the graph as the caller wrote it, before a pass
    // rewrites it: a pass preserves the words, so its casts are the traced ones.
    refuse_unproven_i32_casts(g)?;
    refuse_unproven_expert_ids(g)?;
    // The passes that fit a graph to a GPU's limits read its capabilities; an AIE target has none of them
    // (no grid, no single-buffer limit, no tensor cores, no contraction lowering), so those passes are
    // GPU stages and an AIE compile does not run them.
    let gpu = match target {
        Target::Gpu { backend, caps } => Some((*backend, caps)),
        Target::Aie { .. } | Target::Vpu { .. } => None,
    };
    // A flash prefill launches `Hq * L` workgroups on one grid axis; cap the rewrite at the target's grid so
    // the planner is never handed a launch the device cannot issue.
    let max_prefill_groups = gpu.map(|(_, caps)| caps.max_grid[0] as usize);
    let mut pipeline = PassPipeline::new(g, target, options.limits);
    let (canonical, numerical_declines) = transform::canonicalize_with_declines(g)?;
    pipeline.after_apply(declared::CANONICALIZE, canonical)?;
    pipeline.run(declared::CSE, transform::cse)?;
    pipeline.run(declared::FOLD_IOTA, transform::fold_iota)?;
    pipeline.run(declared::ROPE_FUSION, transform::rope_fusion)?;
    pipeline.run(declared::FLASH_ATTENTION_CAPPED, |g| {
        transform::flash_attention_capped(g, max_prefill_groups)
    })?;
    pipeline.run(declared::DCE, transform::dce)?;
    // The two claiming passes are the only rewrites whose refusals the escape gate can mask: their
    // reverts put the packed decomposition back, and the gate is what then refuses that
    // decomposition. Their restored maps travel as one local from these runs straight to their
    // single consumer, so the refusal data exists exactly from the producing reverts to the gate
    // check and cannot be read stale (card 669, option b).
    let mut restored = pipeline.run(
        declared::RECOGNIZE_PACKED_CONTRACTIONS,
        transform::recognize_packed_contractions,
    )?;
    restored.extend(pipeline.run(
        declared::RECOGNIZE_PACKED_ROW_GATHERS,
        transform::recognize_packed_row_gathers,
    )?);
    pipeline.reject_escapes(restored)?;
    pipeline.run(declared::FOLD_DENSE_CONTRACTIONS, fold_dense_contractions)?;
    if let Some((_, caps)) = gpu {
        pipeline.try_run(declared::LEGALIZE, |g| {
            transform::legalize(g, caps, &options.limits)
        })?;
    }
    pipeline.run(
        declared::FOLD_DENSE_BF16_ROW_GATHERS,
        fold_dense_bf16_row_gathers,
    )?;
    pipeline.reserve(
        declared::LOWER_NONLAST_REDUCES.pass,
        lower_nonlast_growth(pipeline.graph()),
    )?;
    pipeline.run(declared::LOWER_NONLAST_REDUCES, lower_nonlast_reduces)?;
    if let Some((backend, _)) = gpu {
        pipeline.reserve(
            declared::WIDEN_MISMATCHED_MATMUL_DTYPES.pass,
            widen_growth(pipeline.graph()),
        )?;
        pipeline.run(declared::WIDEN_MISMATCHED_MATMUL_DTYPES, |g| {
            widen_mismatched_matmul_dtypes(g, backend)
        })?;
    }
    // Card 673: admit every validation witness here, on the unfused graph. `fuse` folds a witness
    // producer chain into one `OpKind::Fused` region the structural classifier does not see through,
    // so the walk must run before it. Every backend reaches this through `compile`; a noncanonical
    // witness is refused with the planner's typed `PlanError::ValidationWitnessNotCanonical`, naming
    // the witness, before any executor allocates.
    device_validation::admit_device_witnesses(pipeline.graph())?;
    // The contraction epilogue rule runs on the typed, dtype-prepared graph, right before `fuse` would claim
    // the same pointwise chains for a standalone region: a head fuses only where the planner's generated
    // lowering carries an epilogue (`epilogue_head_lowers`), so no head trades a faster schedule for it. Like
    // `fuse`.
    pipeline.run(declared::FUSE_CONTRACTION_EPILOGUES, |g| {
        transform::fuse_contraction_epilogues(g, |graph, head| match gpu {
            Some((backend, caps)) => planner::epilogue_head_lowers(graph, head, backend, caps),
            // The planner refuses every contraction for an AIE target, so no head lowers there.
            None => false,
        })
    })?;
    pipeline.run(declared::FUSE, transform::fuse)?;
    pipeline.run(
        declared::FUSE_WIDENING_CASTS,
        transform::fuse_widening_casts,
    )?;
    // After `fuse`: the casts it left between an I32 word and the float pointwise kernels that read it or
    // the exact-I32 kernels that write it become steps of those kernels (POOT-1108). GPU targets only: the
    // folded cast is a kernel step, and on a target that plans no kernel the rewrite commits an invalid graph
    // (a `ge` over an I32 and an F32 operand) instead of leaving the planner its typed refusal (POOT-1176).
    if gpu.is_some() {
        pipeline.run(declared::FUSE_INDEX_CASTS, transform::fuse_index_casts)?;
    }
    let (graph, numerics, passes) = pipeline.finish()?;
    // `legalize` fitted every constant, but a later pass's revert of an unplannable split (`after_apply`) can
    // put an oversized one back: re-check the graph the planner receives, never trust the pass's own result.
    if let Some((_, caps)) = gpu {
        transform::refuse_oversized_constants(&graph, caps.max_buffer_bytes)?;
    }
    let planned = plan_graph(&graph, target, &options.limits.body())?;
    let state_commit = state_commit_from_plans(&graph, &planned.plans)?;
    let validation = device_validation::validation_packet_plan(&graph);
    let regions = package(&graph, &planned.plans, target)?;
    let vpu = match target {
        Target::Vpu { caps, .. } => Some(plan_vpu_region(&graph, caps)?),
        Target::Gpu { .. } | Target::Aie { .. } => None,
    };
    let buffer_plan = plan_buffers(
        &graph,
        &planned.plans,
        &planned.storage,
        &state_commit,
        &validation,
        &regions,
    );
    Ok(Program {
        graph,
        plans: planned.plans,
        choices: planned.choices,
        storage: planned.storage,
        slots: planned.slots,
        numerics,
        passes,
        numerical_declines,
        target: *target,
        submission: options.execution,
        state_commit,
        validation,
        regions,
        vpu,
        buffer_plan,
        limits: options.limits,
    })
}

/// The LoRA mode transform over a whole graph, validated and within `budget`. [`compile`] applies it
/// to the graph it is handed; [`crate::compile_staged`] applies it once to the uncut graph and
/// compiles every stage without it.
pub(crate) fn adapt(
    g: &Graph,
    plan: &LoraPlan,
    budget: &GraphBudget,
) -> Result<Graph, CompileError> {
    let adapted = transform::apply_lora(g, plan, budget)?;
    budget.admit("apply_lora", &adapted)?;
    Ok(adapted)
}

/// The graph as it moves through the passes, with the record of the passes that changed it.
struct PassPipeline<'t> {
    graph: Graph,
    target: &'t Target,
    /// The strongest of the traced graph's own numerics (the unfused baseline every device run carries) and
    /// the declarations of the passes that changed it so far.
    recorded: NumericsProperty,
    /// The passes that changed the graph, in the order they ran.
    passes: Vec<&'static str>,
    limits: CompileLimits,
    budget: GraphBudget,
}

impl<'t> PassPipeline<'t> {
    fn new(g: &Graph, target: &'t Target, limits: CompileLimits) -> Self {
        Self {
            graph: g.clone(),
            target,
            recorded: implied_numerics(g),
            passes: Vec::new(),
            limits,
            budget: GraphBudget::of(&limits),
        }
    }

    /// Refuse, before the next pass runs, a pass whose growth over the current graph (a conservative
    /// bound the caller derived from it) would pass [`CompileLimits`]. For a pass that cannot reserve
    /// its own appends; a pass that can does so as it appends and needs no call here.
    fn reserve(&self, pass: &'static str, growth: Growth) -> Result<(), CompileError> {
        Ok(self
            .budget
            .reserve(pass, &self.graph, growth.values, growth.eqns)?)
    }

    /// The graph as the passes have rewritten it so far. Card 673's witness admission reads it
    /// immediately before `fuse`, which folds a witness producer chain into one `OpKind::Fused`
    /// region the structural classifier does not see through.
    fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Run one pass: revert every rewrite the target cannot plan, check the pass's declaration against what
    /// it produced, and record the declaration when the pass changed the graph. Returns the revert's
    /// restored map (empty when nothing was reverted) so a caller can hand it to the one later check
    /// that consumes it; a caller with no such check drops it with the statement.
    fn run(
        &mut self,
        declaration: PassDeclaration,
        apply: impl FnOnce(&Graph) -> Result<Graph, EditError>,
    ) -> Result<HashMap<ValueId, Refusal>, CompileError> {
        let out = apply(&self.graph)?;
        self.after_apply(declaration, out)
    }

    /// Like [`Self::run`], for a pass that can refuse the graph outright (`legalize`'s
    /// [`transform::LegalizeError`]) instead of always producing one: the apply step's error propagates
    /// through `?` before the shared revert/verify/record sequence runs.
    fn try_run<E>(
        &mut self,
        declaration: PassDeclaration,
        apply: impl FnOnce(&Graph) -> Result<Graph, E>,
    ) -> Result<HashMap<ValueId, Refusal>, CompileError>
    where
        CompileError: From<E>,
    {
        let out = apply(&self.graph)?;
        self.after_apply(declaration, out)
    }

    /// Shared tail of [`Self::run`]/[`Self::try_run`]: revert every rewrite the target cannot plan,
    /// check the pass's declaration against what it produced, record the declaration when the pass
    /// changed the graph, and return the revert's restored map - the typed refusal behind each
    /// equation the revert put back (card 669).
    fn after_apply(
        &mut self,
        declaration: PassDeclaration,
        out: Graph,
    ) -> Result<HashMap<ValueId, Refusal>, CompileError> {
        self.budget.admit(declaration.pass, &out)?;
        let (out, restored) =
            revert_unplannable_rewrites(&self.graph, out, self.target, &self.limits.body())?;
        declaration.verify(&out)?;
        if changed(&self.graph, &out) {
            self.recorded = self.recorded.strongest(declaration.property);
            self.passes.push(declaration.pass);
        }
        self.graph = out;
        Ok(restored)
    }

    /// Run the escape gate (`reject_packed_dequant_escapes`). A `PackedDequant` that escapes because a
    /// revert restored the decomposition of a claim the target refused reports that revert's typed
    /// refusal - the equation and the missing capability, at the pipeline entry ADR-0104 names -
    /// instead of the gate's downstream symptom of the same graph (card 669). An escape no entry in
    /// `restored` covers is the graph's own and keeps the gate's typed error.
    ///
    /// `restored` arrives as the local `compile` threaded from the two claiming passes' runs: it lives
    /// exactly from those producing reverts to this one consumer and is consumed here, so an entry can
    /// never be re-read after a later pass supersedes the equation it maps (card 669).
    fn reject_escapes(&mut self, restored: HashMap<ValueId, Refusal>) -> Result<(), CompileError> {
        match transform::reject_packed_dequant_escapes(&self.graph) {
            Ok(()) => self
                .after_apply(declared::REJECT_PACKED_DEQUANT_ESCAPES, self.graph.clone())
                .map(|_gate_restored| ()),
            Err(error) => Err(match restored.get(&escaped_value(&error)) {
                Some(refusal) => PlanError::Refused(Box::new(refusal.clone())).into(),
                None => error.into(),
            }),
        }
    }

    /// The compiled graph, the tier-2 class it publishes and the passes that changed it. Fails closed when
    /// the graph implies a looser class than the recorded passes and the baseline allow.
    fn finish(self) -> Result<(Graph, Tier2Class, Vec<&'static str>), NumericsError> {
        let implied = implied_numerics(&self.graph);
        if implied > self.recorded {
            return Err(NumericsError::UnderDeclared {
                pass: "compile",
                declared: self.recorded,
                implied,
            });
        }
        Ok((self.graph, self.recorded.tier2_class(), self.passes))
    }
}

/// The most a pass can add to a graph, from a bound its caller derived from the graph alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Growth {
    values: usize,
    eqns: usize,
}

/// What `lower_nonlast_reduces` can add: a non-last-axis `Reduce` becomes a transpose and a reduce, and a
/// `keepdim` one also a reshape, so one equation and one value more, or two and two.
fn lower_nonlast_growth(g: &Graph) -> Growth {
    let added = g
        .eqns()
        .iter()
        .fold(0usize, |added, eqn| match (&eqn.op, eqn.inputs.first()) {
            (OpKind::Reduce { axis, keepdim, .. }, Some(Operand::Value(input)))
                if axis + 1 < g.aval(*input).shape.len() =>
            {
                added.saturating_add(if *keepdim { 2 } else { 1 })
            }
            _ => added,
        });
    Growth {
        values: added,
        eqns: added,
    }
}

/// What `widen_mismatched_matmul_dtypes` can add: one cast per widened table of a `Gather`/`Slice`, and
/// at most two (one per operand) for a `MatMul` or `Binary`.
fn widen_growth(g: &Graph) -> Growth {
    let casts = g.eqns().iter().fold(0usize, |casts, eqn| {
        casts.saturating_add(match eqn.op {
            OpKind::Gather { axis: 0 } | OpKind::Slice { .. } => 1,
            OpKind::MatMul | OpKind::Binary(_) => 2,
            _ => 0,
        })
    });
    Growth {
        values: casts,
        eqns: casts,
    }
}

/// Whether a pass changed the graph: an equation added, removed, retyped or rewired, or a value added.
fn changed(before: &Graph, after: &Graph) -> bool {
    before.values().len() != after.values().len()
        || before.eqns().len() != after.eqns().len()
        || before.eqns().iter().zip(after.eqns()).any(|(a, b)| {
            a.out != b.out
                || a.op != b.op
                || a.inputs.len() != b.inputs.len()
                || a.inputs
                    .iter()
                    .zip(&b.inputs)
                    .any(|(x, y)| !same_operand(x, y))
        })
}

fn same_operand(a: &Operand, b: &Operand) -> bool {
    match (a, b) {
        (Operand::Value(x), Operand::Value(y)) => x == y,
        (Operand::Lit(x), Operand::Lit(y)) => x == y,
        _ => false,
    }
}

/// The `PackedDequant` output a [`PackedDequantProductionError`] names: every variant the escape gate
/// reports carries the error context of the escaping dequant Matched
/// exhaustively on purpose: a new variant forces this site to decide whether it carries an escape.
fn escaped_value(error: &PackedDequantProductionError) -> ValueId {
    use PackedDequantProductionError as Error;
    match error {
        Error::CarrierStorage { context, .. }
        | Error::CarrierIdentity { context, .. }
        | Error::CarrierEscape { context, .. }
        | Error::CarrierConsumerCount { context, .. }
        | Error::CarrierOperands { context }
        | Error::GraphOutput { context }
        | Error::ValidationOutput { context }
        | Error::StateOutput { context }
        | Error::ConsumerCount { context, .. }
        | Error::TransposePermutation { context, .. }
        | Error::Movement { context, .. }
        | Error::Unmatched { context } => context.value,
    }
}

/// Plan every equation `after` rewrote (a new op at an existing output, or a new output) for `target`, the
/// way `crate::plan_graph` plans (the graph's own strided views), and put back the decomposition of each
/// one the target refuses.
///
/// Every rewriting pass replaces a region's root equation in place (same output value) and leaves or drops
/// the members it absorbed. A refused root is restored from `before`, together with every member that
/// `after` dropped and the restored equations read. Equations are emitted in `before`'s order, which is
/// topological for both graphs because a rewritten root reads only values defined before its region. A
/// pass that defines an output `before` never had, or gives an output it had another type, cannot be taken
/// apart this way, so a refusal there reverts the whole pass.
///
/// Alongside the restored graph, returns the typed refusal behind it, keyed by every equation the revert
/// restored (card 669): each refused root's closure maps to its own [`Refusal`], the first root in probe
/// order owning a value two refused roots restored together. A whole-pass revert restores `before`
/// wholesale, attributing no individual equation to any one refusal, so it records nothing - an escape
/// there is the restored graph's own, not this revert's. [`PassPipeline::reject_escapes`] reports the
/// recorded refusal when the escape gate finds one of these restored values escaping; `compile` threads
/// the map there as a local and no other site reads it.
fn revert_unplannable_rewrites(
    before: &Graph,
    after: Graph,
    target: &Target,
    limits: &poot_kernelgen::BodyLimits,
) -> Result<(Graph, HashMap<ValueId, Refusal>), CompileError> {
    let before_at: HashMap<ValueId, usize> = before
        .eqns()
        .iter()
        .enumerate()
        .map(|(i, eqn)| (eqn.out, i))
        .collect();
    let rewritten = after.eqns().iter().filter(|eqn| {
        before_at
            .get(&eqn.out)
            .is_none_or(|&i| before.eqns()[i].op != eqn.op)
    });
    // Probe exactly as `plan_graph` plans: with the graph's own strided views.
    // A probe stricter than the final plan would drop a rewrite the target can run.
    let views = compute_views(&after, target.id());
    let mut refused: Vec<(ValueId, Refusal)> = Vec::new();
    for eqn in rewritten {
        match plan_eqn_compiled(&after, eqn, target, &views, limits) {
            Ok(_) => {}
            Err(PlanError::Refused(refusal)) => refused.push((eqn.out, *refusal)),
            Err(error) => return Err(error.into()),
        }
    }
    if refused.is_empty() {
        return Ok((after, HashMap::new()));
    }
    // A value `after` retyped is read by equations this revert keeps from `after` (a view whose op is
    // unchanged) and by ones it restores from `before`: neither side is well typed with the other, so the
    // pass cannot be taken apart root by root.
    let retypes = after.eqns().iter().any(|eqn| {
        before_at
            .get(&eqn.out)
            .is_some_and(|_| before.aval(eqn.out) != after.aval(eqn.out))
    });
    if retypes
        || after
            .eqns()
            .iter()
            .any(|eqn| !before_at.contains_key(&eqn.out))
    {
        return Ok((before.clone(), HashMap::new()));
    }

    let after_at: HashMap<ValueId, usize> = after
        .eqns()
        .iter()
        .enumerate()
        .map(|(i, eqn)| (eqn.out, i))
        .collect();
    let mut restored: HashMap<ValueId, Refusal> = HashMap::new();
    for (root, refusal) in refused {
        let mut pending = vec![root];
        while let Some(out) = pending.pop() {
            if restored.contains_key(&out) {
                continue;
            }
            restored.insert(out, refusal.clone());
            for operand in &before.eqns()[before_at[&out]].inputs {
                if let Operand::Value(input) = operand
                    && before_at.contains_key(input)
                    && !after_at.contains_key(input)
                {
                    pending.push(*input);
                }
            }
        }
    }
    let eqns = before
        .eqns()
        .iter()
        .filter_map(|eqn| {
            if restored.contains_key(&eqn.out) {
                Some(eqn.clone())
            } else {
                after_at.get(&eqn.out).map(|&i| after.eqns()[i].clone())
            }
        })
        .collect();
    Ok((crate::passes::with_eqns(&after, eqns)?, restored))
}

#[cfg(test)]
mod tests {
    use crate::planner::test_only::{per_eqn_plans_inside_compile, per_eqn_plans_outside_compile};
    use crate::test_support::{gpu_caps, gpu_caps_mut};
    use crate::{Capability, Dispatch, plan_eqn};
    use poot_graph_ir::analysis::Tier2Class;
    use poot_graph_ir::{
        BinOp, Builder, OpKind, RedOp, Scalar, Slot, StateRole, Storage, TensorType, UnOp,
    };
    use poot_kernelgen::{ContractionSpec, KernelRequest};
    use poot_tensor::DType;
    use poot_test_util::device_caps::default_caps_for;

    use super::*;

    fn wgpu() -> Target {
        Target::Gpu {
            backend: Backend::SpirvVulkan,
            caps: default_caps_for(Backend::SpirvVulkan),
        }
    }

    fn target_for(backend: Backend) -> Target {
        Target::Gpu {
            backend,
            caps: default_caps_for(backend),
        }
    }

    const REPLAY: CompileOptions =
        CompileOptions::new(Submission::Replay, crate::CompileLimits::STANDARD);

    fn refusal(result: Result<Program, CompileError>, what: &str) -> Refusal {
        match result {
            Err(CompileError::Plan(error)) => match *error {
                PlanError::Refused(refusal) => *refusal,
                other => panic!("{what}: expected a typed refusal, got {other:?}"),
            },
            other => panic!("{what}: expected a typed refusal, got {other:?}"),
        }
    }

    /// `exp(x * s) + b` over one row: a float pointwise chain `fuse` turns into one region.
    fn pointwise_chain() -> Graph {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![4, 8])).unwrap();
        let s = b.constant("s", TensorType::f32(vec![4, 8])).unwrap();
        let bias = b.constant("bias", TensorType::f32(vec![4, 8])).unwrap();
        let scaled = b.binary(BinOp::Mul, x, s).unwrap();
        let e = b.unary(UnOp::Exp, scaled).unwrap();
        let out = b.binary(BinOp::Add, e, bias).unwrap();
        b.finish(out).unwrap()
    }

    /// SC-002: the published class is the one the recorded passes imply. Fusion reassociates, so a graph
    /// fusion changed publishes `Reassociating`; the record names `fuse`, and no pass that left the graph
    /// alone is recorded.
    #[test]
    fn compile_publishes_the_class_of_the_passes_that_changed_the_graph() {
        let program = compile(&pointwise_chain(), &wgpu(), &REPLAY).expect("the chain compiles");
        let passes: Vec<&str> = program.passes().collect();
        assert_eq!(passes, ["fuse"], "only fusion changed the chain");
        assert!(
            program
                .graph()
                .eqns()
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Fused(_))),
            "the chain is one fused region"
        );
        assert_eq!(program.numerics_class(), Tier2Class::Reassociating);
    }

    /// SC-002: a graph no pass changes publishes its own baseline. A pure reshape moves no float value, so
    /// it is compared bit for bit.
    #[test]
    fn compile_of_an_untouched_movement_graph_publishes_bit_exact() {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![4, 8])).unwrap();
        let out = b.reshape(x, vec![8, 4]).unwrap();
        let program =
            compile(&b.finish(out).unwrap(), &wgpu(), &REPLAY).expect("a reshape compiles");
        assert_eq!(program.passes().count(), 0);
        assert_eq!(program.numerics_class(), Tier2Class::BitExact);
    }

    /// SC-002: the target's grid is data. The E4M3FN decode dispatch plans on the measured wgpu grid and is
    /// refused, naming the equation, the target and the synthetic limit, on a device whose x-grid holds
    /// fewer workgroups than the dispatch needs.
    #[test]
    fn compile_refuses_a_dispatch_beyond_the_target_grid() {
        let b = Builder::for_tests();
        let raw = b
            .constant("raw", TensorType::new(vec![1024, 1], DType::E4M3FN))
            .unwrap();
        let out = b.cast(raw, DType::F32).unwrap();
        let g = b.finish(out).unwrap();
        compile(&g, &wgpu(), &REPLAY).expect("the measured wgpu grid holds the dispatch");

        let mut narrow = wgpu();
        gpu_caps_mut(&mut narrow).max_grid = [2, 65_535, 65_535];
        let refusal = refusal(compile(&g, &narrow, &REPLAY), "a two-workgroup x-grid");
        assert_eq!(refusal.eqn, out.id());
        assert_eq!(refusal.target, TargetId::Gpu(Backend::SpirvVulkan));
        let Capability::DispatchGridLimit {
            workgroups_x,
            limit,
            ..
        } = refusal.missing
        else {
            panic!("expected a dispatch-grid refusal, got {refusal:?}");
        };
        assert_eq!(limit, 2);
        assert!(workgroups_x > limit, "{workgroups_x} <= {limit}");
    }

    /// Card 1006: the row-parallel softmax launches one workgroup per row, so a row count above the target's
    /// X grid cap must fold onto the Y grid instead of exceeding the cap. 128 rows on a 64-wide X grid plan
    /// as 64 x 2 workgroups, on every backend. Mutation: lay the rows out 1-D again (`fold_groups(rows,
    /// u32::MAX)` in `poot-kernelgen`'s `row`); the X dispatch is 128 groups and this goes red.
    #[test]
    fn compile_folds_a_row_parallel_softmax_past_the_target_x_grid() {
        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::f32(vec![1, 8, 16, 16]))
            .unwrap();
        let out = poot_graph_ir::ops::softmax(&b, x).unwrap();
        let g = b.finish(out).unwrap();
        for backend in [
            Backend::SpirvVulkan,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
            Backend::Nvptx,
        ] {
            let mut target = target_for(backend);
            gpu_caps_mut(&mut target).max_grid = [64, 65_535, 65_535];
            let program = compile(&g, &target, &REPLAY).expect("the softmax compiles");
            let (_, plan) = program
                .planned()
                .find(|(eqn, _)| matches!(eqn.op, OpKind::FusedRow(_)))
                .expect("the softmax fuses into one row region");
            let Some(Dispatch { body, launch, .. }) = plan.single() else {
                panic!("{backend:?}: the row region plans as one dispatch");
            };
            let grid = launch.grid;
            let width = body.workgroup_size[0] as usize;
            assert_eq!(
                (grid[0] as usize / width, grid[1] as usize, grid[2] as usize),
                (64, 2, 1),
                "{backend:?}: 128 rows fold onto a 64 x 2 workgroup grid"
            );
        }
    }

    /// SC-013 (R-546-4, GPU-free): `Scatter { axis: 1 }` has no live tracer (the public `Builder::scatter`
    /// always emits axis 0), so this builds a valid axis-0 scatter and mutates the stored equation's axis
    /// after tracing - the output type is axis-independent (`Scatter`'s `infer` arm returns the source's
    /// own type), so the graph stays otherwise well-formed. `compile` refuses it with a typed
    /// `Capability::ScatterNonZeroAxis` on every target, GPU-free. Mutation: remove the refusal guard
    /// at `planner.rs` (`OpKind::Scatter { axis }` with `axis >= 1`); `compile` then returns `Ok` and
    /// this row goes red.
    #[test]
    fn compile_refuses_a_non_axis_zero_scatter_on_every_target() {
        let b = Builder::for_tests();
        let src = b.constant("src", TensorType::f32(vec![4, 5])).unwrap();
        let index = b
            .constant("index", TensorType::new(vec![4], DType::I32))
            .unwrap();
        let out = b.scatter(src, index).unwrap();
        let traced = b.finish(out).unwrap();
        let scatter_idx = traced
            .eqns()
            .iter()
            .position(|eqn| matches!(eqn.op, OpKind::Scatter { .. }))
            .expect("the traced graph has one Scatter equation");
        let g = poot_graph_ir::test_support::graph_fixtures::derive(&traced, |draft| {
            draft.eqns_mut()[scatter_idx].op = OpKind::Scatter { axis: 1 };
        })
        .unwrap();

        for backend in [
            Backend::SpirvVulkan,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
            Backend::Nvptx,
        ] {
            let refusal = refusal(
                compile(&g, &target_for(backend), &REPLAY),
                &format!("{backend:?} non-axis-0 scatter"),
            );
            assert_eq!(refusal.eqn, out.id());
            assert_eq!(refusal.target, TargetId::Gpu(backend));
            assert_eq!(refusal.missing, Capability::ScatterNonZeroAxis { axis: 1 });
        }
    }

    /// An equation the traced graph itself asks for, which the target has no kernel for, is refused: no
    /// rewrite produced it, so there is no decomposition to fall back to. (Card 534a: a non-last-axis
    /// `Reduce` no longer demonstrates this - `lower_nonlast_reduces` is now compile's own rewrite for
    /// it - so this uses a mixed-dtype `ScatterUpdate` no pass rewrites instead.)
    #[test]
    fn compile_refuses_an_equation_the_traced_graph_cannot_run() {
        let b = Builder::for_tests();
        let base = b
            .constant("base", TensorType::new(vec![2, 5], DType::E4M3FN))
            .unwrap();
        let src = b.constant("src", TensorType::f32(vec![1, 5])).unwrap();
        let slots = b
            .constant("slots", TensorType::new(vec![1], DType::I32))
            .unwrap();
        let out = b.scatter_update(base, src, slots).unwrap();
        let refusal = refusal(
            compile(&b.finish(out).unwrap(), &wgpu(), &REPLAY),
            "E4M3FN ScatterUpdate over an f32 source",
        );
        assert_eq!(refusal.eqn, out.id());
        assert_eq!(refusal.missing, Capability::DtypeLowering);
    }

    /// R467-003: `optimize` alone fuses a BF16 pointwise chain into a region no target has a BF16 body
    /// for, so the optimized graph is unplannable while the traced one plans. `compile` keeps the
    /// decomposition (restoring the members fusion absorbed) and records no fusion.
    ///
    /// Card 534a: the operands are slots, not consts, since this fixture is testing fusion legality,
    /// not the stored-const lanes Card 1011 owns.
    #[test]
    fn compile_keeps_the_decomposition_when_the_target_refuses_the_fused_region() {
        let b = Builder::for_tests();
        let x = b
            .slot(Slot::Activation, TensorType::new(vec![4, 8], DType::BF16))
            .unwrap();
        let s = b
            .slot(Slot::Activation, TensorType::new(vec![4, 8], DType::BF16))
            .unwrap();
        let bias = b
            .slot(Slot::Activation, TensorType::new(vec![4, 8], DType::BF16))
            .unwrap();
        let scaled = b.binary(BinOp::Mul, x, s).unwrap();
        let e = b.unary(UnOp::Exp, scaled).unwrap();
        let out = b.binary(BinOp::Add, e, bias).unwrap();
        let g = b.finish(out).unwrap();

        let optimized = transform::fuse(&g).unwrap();
        let fused = optimized
            .eqns()
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Fused(_)))
            .expect("fuse fuses the chain");
        assert!(
            matches!(
                plan_eqn(
                    &optimized,
                    fused,
                    Backend::SpirvVulkan,
                    &gpu_caps(&wgpu()),
                    &crate::test_support::roomy_body_limits()
                ),
                Err(PlanError::Refused(_))
            ),
            "the premise: no target plans a BF16 fused region"
        );

        let program = compile(&g, &wgpu(), &REPLAY).expect("the decomposition plans");
        let ops: Vec<String> = program
            .graph()
            .eqns()
            .iter()
            .map(|eqn| eqn.op.name())
            .collect();
        assert_eq!(ops, ["mul", "exp", "add"], "the chain stays decomposed");
        assert_eq!(program.passes().count(), 0, "no pass changed the graph");
    }

    /// `fuse_index_casts` retypes the views between an I32 word's cast and the float region that reads it.
    /// A target that refuses the folded region has the equations it replaced put back, but the views stay
    /// I32, so the restored `ge` would read an I32 and an F32 operand. A pass that retypes a value is
    /// reverted whole, so `compile` reaches the planner's typed refusal instead of an invalid graph (POOT-1179).
    ///
    /// Mutation: drop `retypes ||` in `revert_unplannable_rewrites`; `compile` returns the untyped
    /// `Edit(Invalid(BinaryValueDtypeMismatch { operation: "ge", left: I32, right: F32 }))`.
    #[test]
    fn compile_reverts_a_retyping_pass_whole_when_the_target_refuses_its_region() {
        let n = 4096usize;
        let b = Builder::for_tests();
        let a = b
            .slot_named(Slot::Token, "a", TensorType::new(vec![n], DType::I32))
            .unwrap();
        let c = b
            .slot_named(Slot::Token, "c", TensorType::new(vec![n], DType::I32))
            .unwrap();
        let flag = b.binary(BinOp::GeU, a, c).unwrap();
        let x = b.constant("x", TensorType::f32(vec![1, n])).unwrap();
        let flag_row = b
            .reshape(b.cast(flag, DType::F32).unwrap(), vec![1, n])
            .unwrap();
        let hits = b
            .reduce(
                RedOp::Sum,
                b.binary(BinOp::Ge, flag_row, x).unwrap(),
                1,
                true,
            )
            .unwrap();
        // A second reader of the I32 flag keeps the cast from folding into its producer.
        let others = b
            .reduce(
                RedOp::Sum,
                b.cast(b.unary(UnOp::Not, flag).unwrap(), DType::F32)
                    .unwrap(),
                0,
                true,
            )
            .unwrap();
        let out = b
            .binary(BinOp::Add, b.reshape(hits, vec![1]).unwrap(), others)
            .unwrap();
        let g = b.finish(out).unwrap();

        let planned = compile(&g, &wgpu(), &REPLAY).expect("the roomy target folds the cast");
        assert!(planned.passes().any(|pass| pass == "fuse_index_casts"));

        let mut narrow = wgpu();
        gpu_caps_mut(&mut narrow).max_workgroup_invocations = 1;
        // The target plans nothing at one invocation per workgroup, so the reverted graph meets the
        // planner's typed refusal of the equation it was left with, not an invalid-graph edit error.
        let refused = refusal(compile(&g, &narrow, &REPLAY), "a one-invocation workgroup");
        assert_eq!(refused.op, OpKind::Binary(BinOp::GeU));
        assert!(matches!(refused.missing, Capability::KernelResources(_)));
    }

    /// `compile` plans each equation of the program exactly once: value storage reads the plans `plan_graph`
    /// made instead of planning the graph again. A graph no pass rewrites makes no legality probe, so its
    /// count is the equation count; a rewrite adds one probe per rewritten equation, on the intermediate
    /// graph, before `plan_graph`'s own pass.
    #[test]
    fn compile_plans_each_equation_exactly_once() {
        let optimized = compile(&pointwise_chain(), &wgpu(), &REPLAY)
            .expect("the chain compiles")
            .graph()
            .clone();
        let before = per_eqn_plans_inside_compile();
        let program = compile(&optimized, &wgpu(), &REPLAY).expect("the fused chain compiles");
        assert_eq!(
            program.passes().count(),
            0,
            "an optimized graph is not rewritten"
        );
        assert_eq!(
            per_eqn_plans_inside_compile() - before,
            program.graph().eqns().len() as u64,
            "one plan per equation"
        );

        let before = per_eqn_plans_inside_compile();
        let program = compile(&pointwise_chain(), &wgpu(), &REPLAY).expect("the chain compiles");
        assert_eq!(program.graph().eqns().len(), 1, "fusion leaves one region");
        assert_eq!(
            per_eqn_plans_inside_compile() - before,
            2,
            "one legality probe of the fused region, then plan_graph's one plan"
        );
    }

    /// SC-003: per-equation plans made inside `compile` are not counted; a direct planner call is.
    #[test]
    fn only_plans_outside_compile_are_counted() {
        let g = pointwise_chain();
        let before = per_eqn_plans_outside_compile();
        compile(&g, &wgpu(), &REPLAY).expect("the chain compiles");
        assert_eq!(per_eqn_plans_outside_compile(), before);
        plan_eqn(
            &g,
            &g.eqns()[0],
            Backend::SpirvVulkan,
            &gpu_caps(&wgpu()),
            &crate::test_support::roomy_body_limits(),
        )
        .expect("a mul plans");
        assert_eq!(per_eqn_plans_outside_compile(), before + 1);
    }

    /// SC-002: a MiniMax-M2-shaped MoE gating reduce over axis 0 - the same shape
    /// that planned on wgpu and refused on ROCm and PTX before Card 534a, because only wgpu's own
    /// pre-compile preparation ran `lower_nonlast_reduces` - now compiles for every backend, since
    /// `compile`'s own pipeline runs the legalization for all three. Mutation: comment out the
    /// `lower_nonlast_reduces` `pipeline.run` line above; `AmdGcn`/`Nvptx` refuse with
    /// `Capability::NonLastAxisReduce` (confirmed by hand per ADR-0090, not committed here since a
    /// permanent red test would fail every other row in this module too).
    #[test]
    fn compile_plans_a_non_last_axis_moe_gate_reduce_on_every_backend() {
        let b = Builder::for_tests();
        let logits = b
            .constant("expert_logits", TensorType::f32(vec![4, 2]))
            .unwrap();
        let gate = b.reduce(RedOp::Sum, logits, 0, false).unwrap();
        let g = b.finish(gate).unwrap();

        let inputs = HashMap::from([(
            logits.id(),
            poot_eval::Value::from(poot_tensor::HostTensor::f32(
                vec![4, 2],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )),
        )]);
        let oracle = poot_eval::eval(
            &g,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("the raw axis-0 reduce evaluates")
        .output
        .into_host()
        .expect("dense oracle output");

        for (name, backend) in [
            ("AmdGcn", Backend::AmdGcn(poot_target::AmdArch::gfx1151())),
            ("Nvptx", Backend::Nvptx),
        ] {
            let program = compile(&g, &target_for(backend), &REPLAY).unwrap_or_else(|error| {
                panic!(
                    "{name}: expected the axis-0 MoE gate reduce to compile now that \
                     `lower_nonlast_reduces` runs for every backend, got {error:?}"
                )
            });
            assert!(
                !program
                    .graph()
                    .eqns()
                    .iter()
                    .any(|eqn| matches!(eqn.op, OpKind::Reduce { axis: 0, .. })),
                "{name}: the axis-0 reduce must be legalized away, not left for the planner"
            );
            let compiled = poot_eval::eval(
                program.graph(),
                &inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .unwrap_or_else(|error| panic!("{name}: the compiled graph must evaluate: {error:?}"))
            .output
            .into_host()
            .unwrap_or_else(|error| panic!("{name}: compiled output not dense: {error:?}"));
            assert_eq!(
                compiled
                    .as_f32()
                    .unwrap()
                    .iter()
                    .map(|f| f.to_bits())
                    .collect::<Vec<_>>(),
                oracle
                    .as_f32()
                    .unwrap()
                    .iter()
                    .map(|f| f.to_bits())
                    .collect::<Vec<_>>(),
                "{name}: lower_nonlast_reduces must be bit-identical to the untransformed axis-0 reduce"
            );
        }
    }
    // card 523a review: `legalize` as a pass INSIDE `compile`'s own pipeline, driven
    // through `compile` itself rather than called bare.

    fn wgpu_with_limit(max_buffer_bytes: u64) -> Target {
        let mut target = wgpu();
        gpu_caps_mut(&mut target).max_buffer_bytes = max_buffer_bytes;
        target
    }

    /// A `[vocab, hidden]` f32 embed table gathered by a scalar `Slot::Token` (the decode shape), with
    /// an unsplit `lm_head`-shaped weight alongside it so `dce`/`fuse` see a realistic multi-value graph
    /// and the embed table is the only thing over `limit`.
    fn embed_gather_graph(vocab: usize, hidden: usize) -> Graph {
        let b = Builder::for_tests();
        let token = b.slot(Slot::Token, TensorType::scalar(DType::I32)).unwrap();
        let embed = b
            .constant(
                "model.embed_tokens.weight",
                TensorType::f32(vec![vocab, hidden]),
            )
            .unwrap();
        let out = b.gather_scalar(embed, 0, token).unwrap();
        b.finish(out).unwrap()
    }

    /// `x @ w^T`, `w` an `[n, k]` named constant read through a transpose (how the tracers declare every
    /// dense weight). `x` is a broadcast of one real scalar, not a declared `[1, k]` constant:
    /// `legalize`'s split rewrite never looks at the activation's storage, only the weight's, so a huge
    /// `k` (forcing even one row over the limit) stays a cheap fixture.
    fn matmul_graph(x_name: &str, k: usize, w_name: &str, n: usize) -> Graph {
        let b = Builder::for_tests();
        let x0 = b.constant(x_name, TensorType::f32(vec![1, 1])).unwrap();
        let x = b.broadcast(x0, vec![1, k]).unwrap();
        let w = b.constant(w_name, TensorType::f32(vec![n, k])).unwrap();
        let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        b.finish(out).unwrap()
    }

    fn legalize_error(result: Result<Program, CompileError>, what: &str) -> LegalizeError {
        match result {
            Err(CompileError::Legalize(error)) => *error,
            other => panic!("{what}: expected a typed legalize refusal, got {other:?}"),
        }
    }

    /// SC-002 (host), driven through `compile`: an embed table over the target's
    /// buffer limit compiles to a program whose graph binds `Slot::TokenEmbed`, not a device `Gather`,
    /// and `legalize` is recorded among the passes that changed the graph - `compile`'s own
    /// `Program::passes()`, not a bare call to `legalize`. Mutation (recorded, not committed): replacing
    /// `compile`'s `pipeline.try_run("legalize", ...)` call with a no-op pass-through (same effect as
    /// dropping the pass from the pipeline) failed this test AND
    /// [`compile_splits_an_oversized_matmul_weight`] with `legalize must be recorded as a pass that
    /// changed the graph` / `assertion failed: program.passes().any(|p| p == "legalize")`; restoring the
    /// real call made both green again.
    #[test]
    fn compile_hosts_an_oversized_embed_gather() {
        let g = embed_gather_graph(152_064, 3584); // qwen2.5-7b's shape, ~2.18 GB
        let target = wgpu_with_limit(2_147_483_647); // wgpu's raw maxBufferSize
        let program = compile(&g, &target, &REPLAY).expect("an embed table always hosts");

        assert!(
            program.passes().any(|p| p == "legalize"),
            "legalize must be recorded as a pass that changed the graph"
        );
        assert!(
            !program
                .graph()
                .eqns()
                .iter()
                .any(|e| matches!(e.op, OpKind::Gather { .. })),
            "the gather is replaced, not merely left decomposed"
        );
        let hosted = program
            .graph()
            .inputs()
            .iter()
            .find(|&&id| program.graph().meta(id).storage == Storage::Slot(Slot::TokenEmbed))
            .expect("compile's output graph declares the hosted embed input");
        assert_eq!(
            program.graph().meta(*hosted).name.as_deref(),
            Some("model.embed_tokens.weight")
        );
    }

    /// POOT-1182: a BF16 embed table over the buffer limit is hosted, and the rows the host gathers are
    /// weight bytes in the lane the plan chose, exactly as the table's own const would be. On wgpu that
    /// lane is the packed `u32` one (SPIR-V has no scalar BF16), so the widening `Cast` plans the
    /// packed body; planning it as the native two-byte cast failed at codegen on the real Qwen3.8-27B-FP8
    /// checkpoint (`bf16 compute is NVPTX-only`). Mutation: drop `Slot::TokenEmbed` from
    /// `Storage::holds_weight_bytes`; the slot's storage is `Dense(BF16)` and the cast the native one.
    #[test]
    fn compile_reads_hosted_bf16_embed_rows_through_the_packed_lane() {
        let b = Builder::for_tests();
        let token = b
            .slot(Slot::Token, TensorType::new(vec![1, 4], DType::I32))
            .unwrap();
        let embed = b
            .constant(
                "model.embed_tokens.weight",
                TensorType::new(vec![40_960, 64], DType::BF16),
            )
            .unwrap();
        let rows = b.gather(embed, 0, token).unwrap();
        let out = b.cast(rows, DType::F32).unwrap();
        let g = b.finish(out).unwrap();

        // The gather is folded to a `DenseRowGather` when the table fits, so only a target whose limit is
        // under the 5 MiB table hosts it.
        let target = wgpu_with_limit(1 << 20);
        let program = compile(&g, &target, &REPLAY).expect("a hosted BF16 embed compiles");
        let &hosted = program
            .graph()
            .inputs()
            .iter()
            .find(|&&id| program.graph().meta(id).storage == Storage::Slot(Slot::TokenEmbed))
            .expect("the oversized table is hosted");
        assert_eq!(
            program.storage().storage(hosted).buffer_storage(),
            poot_target::BufferStorage::bf16_packed()
        );
        let (cast, _) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::Cast { .. }))
            .expect("the widening cast survives compile");
        assert!(
            matches!(
                program.kernel_choice(cast),
                KernelChoice::Imported {
                    kernel: crate::ImportedKernel::PackedBf16ToF32,
                    ..
                }
            ),
            "{:?}",
            program.kernel_choice(cast)
        );
    }

    /// A target whose buffer limit holds the whole table compiles to a program that still binds the
    /// dense `Gather` - `legalize` is a no-op and is not recorded (matches SC-002's other outcome: a
    /// target under the limit never needs the rewrite).
    #[test]
    fn compile_leaves_a_small_embed_dense() {
        let g = embed_gather_graph(151_936, 896); // qwen2.5-0.5b, ~544 MB
        let target = wgpu_with_limit(2_147_483_647);
        let program = compile(&g, &target, &REPLAY).expect("under the limit, no rewrite needed");
        assert!(!program.passes().any(|p| p == "legalize"));
        assert!(
            program
                .graph()
                .eqns()
                .iter()
                .any(|e| matches!(e.op, OpKind::Gather { .. })),
            "the dense gather stays: this target can hold the table"
        );
    }

    /// SC-002 (split), driven through `compile`: a plain matmul weight over the target's buffer limit,
    /// with no gather-by-token pattern, compiles to a program whose graph has replaced the single
    /// oversized weight with several smaller chunk constants joined by a `Concat` - restoring the
    /// "split" capability card 523a's first landing dropped, now observed through the
    /// real entry point instead of a bare `legalize` call.
    #[test]
    fn compile_splits_an_oversized_matmul_weight() {
        let g = matmul_graph("x", 4096, "big.weight", 200_000); // ~3.05 GB
        let target = wgpu_with_limit(2_147_483_647);
        let program =
            compile(&g, &target, &REPLAY).expect("an oversized matmul weight always splits");

        assert!(program.passes().any(|p| p == "legalize"));
        let chunk_names: Vec<String> = program
            .graph()
            .consts()
            .iter()
            .filter_map(|&id| program.graph().meta(id).name.clone())
            .filter(|n| n.starts_with("big.weight.chunk"))
            .collect();
        assert_eq!(
            chunk_names.len(),
            2,
            "3.05 GB over a 2 GiB-ish limit needs 2 chunks"
        );
        assert!(
            program
                .graph()
                .eqns()
                .iter()
                .any(|e| matches!(e.op, OpKind::Concat { .. })),
            "the chunks join by concat"
        );
    }

    /// SC-002 (split), numeric: the split's column-chunk matmuls, concatenated, reproduce the unsplit
    /// matmul's every output exactly - each output column's dot product sums the same K terms in the
    /// same order whether the weight is one dense buffer or several column chunks, so this is a real
    /// numeric check, not merely a structural one.
    #[test]
    fn compile_split_matmul_matches_dense_matmul() {
        use poot_eval::{EvalBudget, EvalOptions, Value, eval};
        use poot_test_util::assert_close_rel;
        use std::collections::HashMap;

        let (k, n) = (4usize, 10usize);
        let g = matmul_graph("x", k, "w", n);
        // row_bytes = k*4 = 16 bytes; limit=48 (< 4096, so `usable == limit`) => 3 rows per chunk:
        // row ranges (0,3),(3,6),(6,9),(9,10) - 4 chunks, including an uneven final one.
        let target = wgpu_with_limit(48);
        let program = compile(&g, &target, &REPLAY).expect("splits");

        let w_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01 - 0.05).collect();
        let bind_seed = |ids: &[ValueId], graph: &Graph| -> HashMap<ValueId, Value> {
            ids.iter()
                .map(|&id| {
                    let m = graph.meta(id);
                    let t = match m.name.as_deref() {
                        Some("w") => poot_tensor::HostTensor::f32(vec![n, k], w_data.clone()),
                        Some("x") => poot_tensor::HostTensor::f32(vec![1, 1], vec![1.0]),
                        other => panic!("unexpected const {other:?}"),
                    };
                    (id, Value::from(t))
                })
                .collect()
        };
        let dense = eval(
            &g,
            &bind_seed(g.inputs(), &g),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap()
        .output
        .into_host()
        .unwrap();

        let gh = program.graph();
        let mut split_inputs: HashMap<ValueId, Value> = HashMap::new();
        let mut row = 0usize;
        for &id in gh.consts() {
            let m = gh.meta(id);
            let name = m.name.as_deref().unwrap();
            if let Some(rest) = name.strip_prefix("w.chunk") {
                let _: usize = rest.parse().expect("chunk index");
                let height = m.aval.shape[0];
                let data = w_data[row * k..(row + height) * k].to_vec();
                split_inputs.insert(
                    id,
                    Value::from(poot_tensor::HostTensor::f32(vec![height, k], data)),
                );
                row += height;
            } else {
                split_inputs.insert(
                    id,
                    Value::from(poot_tensor::HostTensor::f32(vec![1, 1], vec![1.0])),
                );
            }
        }
        assert_eq!(
            row, n,
            "the chunks must partition every weight row exactly once"
        );
        let split = eval(gh, &split_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();

        assert_close_rel(split.as_f32().unwrap(), dense.as_f32().unwrap(), 1e-6);
    }

    /// SC-002 (refuse): a matmul weight whose single column already exceeds the target's buffer limit
    /// has no rewrite `legalize` knows, so `compile` fails closed with a typed `CompileError::Legalize`
    /// before `plan_graph` ever runs - never a bare device allocation failure.
    #[test]
    fn compile_refuses_an_unlegalizable_oversized_constant() {
        let g = matmul_graph("x", 600_000_000, "big.weight", 2);
        let target = wgpu_with_limit(2_147_483_647);
        let error = legalize_error(
            compile(&g, &target, &REPLAY),
            "a weight whose single column exceeds the limit",
        );
        assert_eq!(
            error,
            LegalizeError::OversizedConstant {
                name: "big.weight".to_string(),
                bytes: 600_000_000u64 * 2 * 4,
                limit: 2_147_483_647,
            }
        );
    }

    /// A TIED F16 head: one `[vocab, hidden]` F16 weight read both by the token gather (so the
    /// sole-reader gate leaves it unhosted) and, through a transpose, by the head matmul. `legalize`
    /// splits the head; the planner cannot plan the F16 chunks, so `revert_unplannable_rewrites` undoes
    /// the split and the weight is whole again after the pass pipeline.
    fn tied_f16_head_graph(vocab: usize, hidden: usize) -> Graph {
        let b = Builder::for_tests();
        let token = b.slot(Slot::Token, TensorType::scalar(DType::I32)).unwrap();
        let w = b
            .constant(
                "tied.weight",
                TensorType::new(vec![vocab, hidden], DType::F16),
            )
            .unwrap();
        let embedded = b
            .cast(b.gather_scalar(w, 0, token).unwrap(), DType::F32)
            .unwrap();
        let x = b.reshape(embedded, vec![1, hidden]).unwrap();
        let head = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        b.finish(head).unwrap()
    }

    /// POOT-1050: `compile` never returns a program holding a constant over `max_buffer_bytes`, whether
    /// or not a pass split survived the unplannable-rewrite revert: it is a typed `OversizedConstant`
    /// or every constant fits. Driven on all three backends through `compile`.
    #[test]
    fn compile_never_succeeds_with_an_oversized_tied_f16_head() {
        let limit = 174_762;
        for backend in [
            Backend::SpirvVulkan,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
            Backend::Nvptx,
        ] {
            let mut target = target_for(backend);
            gpu_caps_mut(&mut target).max_buffer_bytes = limit;
            let g = tied_f16_head_graph(4096, 64);
            match compile(&g, &target, &REPLAY) {
                Err(CompileError::Legalize(error)) => assert!(
                    matches!(*error, LegalizeError::OversizedConstant { limit: l, .. } if l == limit),
                    "{backend:?}: {error:?}"
                ),
                Ok(program) => {
                    let graph = program.graph();
                    for &id in graph.consts() {
                        let aval = graph.aval(id);
                        let bytes = aval.numel() as u64 * aval.dtype.byte_size() as u64;
                        assert!(
                            bytes <= limit,
                            "{backend:?}: compiled with a {bytes}-byte const {:?} over {limit}",
                            graph.meta(id).name
                        );
                    }
                }
                Err(other) => panic!("{backend:?}: unexpected error {other:?}"),
            }
        }
    }

    // Card 642: the packed-contraction claim and Card 355's escape gate run inside
    // `compile`'s own pipeline, driven through `compile` itself.

    fn packed_linear_graph(format: poot_quant::format::WeightFormat, m: usize) -> Graph {
        let descriptor = poot_quant::PackedWeight::try_new(format, [8, 256]).unwrap();
        let b = Builder::for_tests();
        let x = b
            .slot_named(
                Slot::Activation,
                "packed-compile",
                TensorType::f32(vec![m, 256]),
            )
            .unwrap();
        let out =
            poot_graph_ir::ops::packed_linear(&b, x, "layer", descriptor, None, None).unwrap();
        b.finish(out).unwrap()
    }

    /// `compile` claims `packed_linear`'s `PackedDequant -> Transpose -> MatMul` as one planned
    /// `PackedContraction` reading every source in role order, for a GGUF block format (one carrier),
    /// AWQ (three) and act-order GPTQ (four), for decode (`M = 1`) and prefill (`M = 4`), on every
    /// backend; the record names the claim. Mutation: drop the `recognize_packed_contractions` pass
    /// from `compile`; every case is refused by the escape gate (`Unmatched`).
    #[test]
    fn compile_claims_packed_linear_as_a_planned_packed_contraction() {
        use poot_quant::format::{GroupMap, WeightFormat};
        use std::num::NonZeroUsize;
        let nz = |n| NonZeroUsize::new(n).unwrap();
        let formats = [
            WeightFormat::Q4_K,
            WeightFormat::Q8_0,
            WeightFormat::Awq { group_size: nz(64) },
            WeightFormat::Gptq {
                groups: GroupMap::Indexed { groups: nz(4) },
            },
        ];
        for backend in [
            Backend::SpirvVulkan,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
            Backend::Nvptx,
        ] {
            for format in formats {
                for m in [1, 4] {
                    let what = format!("{backend:?} {format:?} M={m}");
                    let program = compile(
                        &packed_linear_graph(format, m),
                        &target_for(backend),
                        &REPLAY,
                    )
                    .unwrap_or_else(|error| panic!("{what}: {error}"));
                    let planned: Vec<(&Eqn, &Plan)> = program.planned().collect();
                    assert!(
                        planned
                            .iter()
                            .all(|(eqn, _)| !matches!(eqn.op, OpKind::PackedDequant { .. })),
                        "{what}: a PackedDequant survived compile"
                    );
                    let (contraction, plan) = planned
                        .iter()
                        .find(|(eqn, _)| matches!(eqn.op, OpKind::PackedContraction { .. }))
                        .unwrap_or_else(|| panic!("{what}: no PackedContraction planned"));
                    let OpKind::PackedContraction { descriptor, .. } = contraction.op else {
                        unreachable!()
                    };
                    assert_eq!(
                        contraction.inputs.len(),
                        1 + descriptor.sources().len(),
                        "{what}"
                    );
                    for (operand, role) in contraction.inputs[1..].iter().zip(descriptor.sources())
                    {
                        let Operand::Value(carrier) = operand else {
                            panic!("{what}: literal carrier");
                        };
                        let name = program.graph().meta(*carrier).name.as_deref().unwrap();
                        assert_eq!(
                            poot_graph_ir::PackedSourceName::parse(name).unwrap().role(),
                            role,
                            "{what}: carrier out of role order"
                        );
                    }
                    assert!(
                        plan.single()
                            .is_some_and(|dispatch| !dispatch.launch().meta.is_empty()),
                        "{what}: {plan:?}"
                    );
                    assert!(
                        program
                            .passes()
                            .any(|pass| pass == "recognize_packed_contractions"),
                        "{what}: the claim is not recorded"
                    );
                }
            }
        }
    }

    /// A `PackedDequant` the claim cannot take (here its decoded weight is the graph output) is
    /// refused by `compile` with the gate's typed error, never materialized. Mutation: drop the
    /// `reject_packed_dequant_escapes` pass; `compile` plans a standalone Materialize and succeeds.
    #[test]
    fn compile_refuses_an_unclaimed_packed_dequant() {
        let descriptor =
            poot_quant::PackedWeight::try_new(poot_quant::format::WeightFormat::Q8_0, [8, 64])
                .unwrap();
        let b = Builder::for_tests();
        let sources: Vec<poot_graph_ir::Traced> =
            poot_graph_ir::packed_source_constants("layer", descriptor)
                .into_iter()
                .map(|(name, tensor_type)| b.constant(name.as_str(), tensor_type).unwrap())
                .collect();
        let decoded = b.packed_dequant(&sources, descriptor).unwrap();
        let graph = b.finish(decoded).unwrap();
        match compile(&graph, &wgpu(), &REPLAY) {
            Err(CompileError::PackedDequant(error)) => assert!(
                matches!(*error, PackedDequantProductionError::GraphOutput { .. }),
                "{error:?}"
            ),
            other => panic!("expected the escape gate's refusal, got {other:?}"),
        }
    }

    const MOE_EXPERTS: usize = 3;
    const MOE_OUT: usize = 8;
    const MOE_K: usize = 64;
    /// The contraction length of the stacked-table rows: a whole K-quant super-block.
    const STACK_K: usize = 256;

    fn moe_descriptor() -> poot_quant::PackedWeight {
        poot_quant::PackedWeight::try_new(poot_quant::format::WeightFormat::Q8_0, [MOE_OUT, MOE_K])
            .unwrap()
    }

    /// One expert's staged `PackedDequant`, as the builders stage it.
    fn staged_dequant(b: &Builder, linear_id: &str) -> poot_graph_ir::Traced {
        let sources: Vec<poot_graph_ir::Traced> =
            poot_graph_ir::packed_source_constants(linear_id, moe_descriptor())
                .into_iter()
                .map(|(name, tensor_type)| b.constant(name.as_str(), tensor_type).unwrap())
                .collect();
        b.packed_dequant(&sources, moe_descriptor()).unwrap()
    }

    /// Every packed layout the engine serves a routed projection in: a plain block format, a K-quant
    /// super-block, the two planar integer formats, and a codebook-scaled 4-bit block format.
    fn moe_formats() -> Vec<poot_quant::format::WeightFormat> {
        use poot_quant::format::{GroupMap, WeightFormat};
        use std::num::NonZeroUsize;
        vec![
            WeightFormat::Q8_0,
            WeightFormat::Q4_K,
            WeightFormat::Awq {
                group_size: NonZeroUsize::new(64).unwrap(),
            },
            WeightFormat::Gptq {
                groups: GroupMap::Indexed {
                    groups: NonZeroUsize::new(4).unwrap(),
                },
            },
            WeightFormat::Mxfp4,
        ]
    }

    fn stack_descriptor(format: poot_quant::format::WeightFormat) -> poot_quant::PackedWeight {
        poot_quant::PackedWeight::try_new(format, [MOE_OUT, STACK_K]).unwrap()
    }

    /// `packed_indexed_linear` over `MOE_EXPERTS` stacked experts of `format`.
    fn packed_moe_graph(format: poot_quant::format::WeightFormat) -> Graph {
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![3, STACK_K]))
            .unwrap();
        let selector = b
            .slot_named_ranged(
                Slot::Activation,
                "moe-ids",
                TensorType::new(vec![3], DType::I32),
                IndexRange::below(MOE_EXPERTS).unwrap(),
            )
            .unwrap();
        let table = poot_graph_ir::ops::PackedExpertStack {
            stack_id: "moe.experts".to_string(),
            descriptor: stack_descriptor(format),
            experts: MOE_EXPERTS,
        };
        let out = poot_graph_ir::ops::packed_indexed_linear(&b, x, selector, &table).unwrap();
        b.finish(out).unwrap()
    }

    /// SC-001 (compile half): a routed packed projection over a stacked table of every layout the engine
    /// serves compiles on every backend to exactly one planned `PackedContraction { rows: Indexed }` - one
    /// generated packed Gemv - and to no `PackedDequant`, `Materialize` or dense `IndexedMatMul`.
    /// Mutation: plan the indexed contraction as the block-diagonal form (`PackedRows::Blocks`); the request
    /// check fails.
    #[test]
    fn compile_plans_the_indexed_packed_contraction_on_every_backend() {
        for backend in [
            Backend::SpirvVulkan,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
            Backend::Nvptx,
        ] {
            for format in moe_formats() {
                let what = format!("{backend:?} {format:?}");
                let program = compile(&packed_moe_graph(format), &target_for(backend), &REPLAY)
                    .unwrap_or_else(|error| panic!("{what}: {error}"));
                let planned: Vec<_> = program.planned().collect();
                let [(contraction, _)] = planned.as_slice() else {
                    panic!(
                        "{what}: expected one planned equation, got {:?}",
                        planned
                            .iter()
                            .map(|(eqn, _)| eqn.op.name())
                            .collect::<Vec<_>>()
                    );
                };
                assert!(
                    matches!(
                        contraction.op,
                        OpKind::PackedContraction {
                            rows: poot_graph_ir::ContractionRows::Indexed {
                                experts: MOE_EXPERTS
                            },
                            ..
                        }
                    ),
                    "{what}: {:?}",
                    contraction.op
                );
                assert!(
                    matches!(
                        program.kernel_choice(contraction),
                        KernelChoice::Generated(KernelRequest::Contraction(
                            ContractionSpec::Packed(request)
                        )) if request.rows == poot_kernelgen::PackedRows::Indexed
                    ),
                    "{what}: {:?}",
                    program.kernel_choice(contraction)
                );
            }
        }
    }

    /// The W12 spelling of a packed expert table - one `PackedDequant -> Transpose -> Reshape` branch per
    /// expert, concatenated into `[E, K, out]` for a dense `IndexedMatMul` - is refused by the escape gate:
    /// there is no admitted chain that materializes an expert, whatever consumes the table. Mutation:
    /// re-admit a `Concat`-fed `IndexedMatMul` in `reject_packed_dequant_escapes`; this graph compiles.
    #[test]
    fn compile_refuses_the_materializing_packed_moe_chain() {
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![3, MOE_K]))
            .unwrap();
        let selector = b
            .slot_named_ranged(
                Slot::Activation,
                "moe-ids",
                TensorType::new(vec![3], DType::I32),
                IndexRange::below(2).unwrap(),
            )
            .unwrap();
        let branch = |id: &str| {
            let dequant = staged_dequant(&b, id);
            let t = b.transpose(dequant, vec![1, 0]).unwrap();
            b.reshape(t, vec![1, MOE_K, MOE_OUT]).unwrap()
        };
        let table = b
            .concat(0, &[branch("expert.0"), branch("expert.1")])
            .unwrap();
        let out = b.indexed_matmul(x, table, selector).unwrap();
        match compile(&b.finish(out).unwrap(), &wgpu(), &REPLAY) {
            Err(CompileError::PackedDequant(_)) => {}
            other => panic!(
                "expected the escape gate's refusal, got {:?}",
                other.map(|_| ())
            ),
        }
    }

    /// The element type of the `Slice<T>` param of `plan`'s first dispatch that reads `value`, the equation's
    /// value operand `position` (a data param comes first, in operand order).
    fn read_element(plan: &Plan, position: usize) -> poot_kernel_ir::Ty {
        let dispatch = plan
            .dispatches()
            .first()
            .unwrap_or_else(|| panic!("no dispatch: {plan:?}"));
        let params: Vec<_> = dispatch.body().params().collect();
        let mut ty = dispatch.body().local_ty(params[position]);
        if let poot_kernel_ir::Ty::Ref { pointee, .. } = ty {
            ty = pointee;
        }
        match ty {
            poot_kernel_ir::Ty::Slice(element) => (**element).clone(),
            other => panic!("param {position} is not a slice: {other:?}"),
        }
    }

    /// POOT-725, POOT-643's fixture (spike 562 F-9): one I32 `Slot::Pos` that is both a Gather index and an
    /// I32->F32 Cast source. Before, the Gather read it as an f32 mirror and the Cast as `Slice<i32>`, so the
    /// slot's one buffer was read on two lanes. Now the slot is dense I32 words and both readers read
    /// `Slice<i32>`, through the production planner. POOT-1108 folds the cast into the `Add` region that
    /// reads it, so that reader is the region's I32 leaf (a view of the slot).
    ///
    /// Mutation (never left in the tree): type the Gather's index param `Slice<f32>` in the movement
    /// generator -> `Plan(I32ReadAsF32 { value: 0, reader: .. })`.
    #[test]
    fn compile_reads_an_i32_slot_on_one_lane_for_gather_and_cast() {
        let b = Builder::for_tests();
        let idx = b
            .slot_ranged(
                Slot::Pos,
                TensorType::new(vec![4], DType::I32),
                IndexRange::below(16).unwrap(),
            )
            .unwrap();
        let table = b.constant("table", TensorType::f32(vec![16, 8])).unwrap();
        let gathered = b.gather(table, 0, idx).unwrap();
        let q = b.cast(idx, DType::F32).unwrap();
        let out = b
            .binary(
                BinOp::Add,
                gathered,
                b.broadcast(b.reshape(q, vec![4, 1]).unwrap(), vec![4, 8])
                    .unwrap(),
            )
            .unwrap();
        let g = b.finish(out).unwrap();

        let program = compile(&g, &wgpu(), &REPLAY).expect("every reader reads Slice<i32>");
        let graph = program.graph();
        let &(slot, _) = graph
            .slots()
            .iter()
            .find(|&&(_, kind)| kind == Slot::Pos)
            .expect("the Pos slot survives compile");
        assert_eq!(
            program.storage().storage(slot).buffer_storage(),
            poot_target::BufferStorage::i32()
        );
        let (_, gather_plan) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::Gather { .. }))
            .expect("the gather survives compile");
        let (region_eqn, region_plan) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::Fused(_)))
            .expect("the cast folds into the region that adds it");
        assert!(
            !graph
                .eqns()
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Cast { .. })),
            "no cast equation is left to launch"
        );
        // The region reads the slot through the reshape and broadcast views the cast result had.
        let leaf = region_eqn
            .inputs
            .iter()
            .position(|operand| {
                matches!(operand, Operand::Value(v) if graph.aval(*v).dtype == DType::I32)
            })
            .expect("the region reads an I32 word");
        // Gather operands are (table, index); the region's leaves are its inputs.
        assert_eq!(read_element(gather_plan, 1), poot_kernel_ir::Ty::I32);
        assert_eq!(read_element(region_plan, leaf), poot_kernel_ir::Ty::I32);
    }

    /// An I32 slot that is a Gather index and a `ScatterUpdate` slot map reads both on `Slice<i32>`, and
    /// `compile` plans one dense I32 buffer for the slot.
    ///
    /// Mutation (never left in the tree): type the scatter's slot map as `Ty::F32`
    /// (`kernelgen data_movement`) -> `Plan(I32ReadAsF32)`.
    #[test]
    fn compile_reads_an_i32_slot_on_one_lane_for_gather_and_scatter_update() {
        let b = Builder::for_tests();
        let idx = b
            .slot(Slot::Pos, TensorType::new(vec![4], DType::I32))
            .unwrap();
        let table = b.constant("table", TensorType::f32(vec![16, 8])).unwrap();
        let rows = b.gather(table, 0, idx).unwrap();
        let late = b
            .cast(
                b.binary_scalar(BinOp::GeU, idx, Scalar::I32(3)).unwrap(),
                DType::F32,
            )
            .unwrap();
        let update = b
            .binary(
                BinOp::Add,
                rows,
                b.broadcast(b.reshape(late, vec![4, 1]).unwrap(), vec![4, 8])
                    .unwrap(),
            )
            .unwrap();
        let cache = b
            .state_input("cache", TensorType::f32(vec![4, 8]), StateRole::Recurrent)
            .unwrap();
        let written = b.scatter_update(cache, update, idx).unwrap();
        let g = b
            .finish(poot_graph_ir::Outputs::new(written).with_state(&[(cache, written)]))
            .unwrap();

        let program = compile(&g, &wgpu(), &REPLAY).expect("every reader reads Slice<i32>");
        let &(slot, _) = program
            .graph()
            .slots()
            .iter()
            .find(|&&(_, kind)| kind == Slot::Pos)
            .expect("the Pos slot survives compile");
        assert_eq!(
            program.storage().storage(slot).buffer_storage(),
            poot_target::BufferStorage::i32()
        );
        let (_, scatter_plan) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::ScatterUpdate))
            .expect("the scatter survives compile");
        // The scatter kernel reads (src, slots): the base is the output's own buffer.
        assert_eq!(read_element(scatter_plan, 1), poot_kernel_ir::Ty::I32);
    }

    /// POOT-1148: the scatter kernel writes only the rows its slot map names, so its output must be donated to
    /// its base's state buffer. `compile` (the production entry) refuses a graph where it cannot be, naming
    /// the scatter: a step that also reads the pool as it was before the write, or a scatter over a value
    /// that is not carried state. Both targets differ from the donated shape only in what reads the base.
    ///
    /// Mutation: drop the `ScatterNotInPlace` check in `state_commit_from_plans`; both graphs compile and the
    /// rows go red with "expected ScatterNotInPlace, got Ok".
    #[test]
    fn compile_refuses_a_scatter_whose_output_cannot_be_donated_to_its_base() {
        let reads_old_pool = || {
            let b = Builder::for_tests();
            let pool = b
                .state_input(
                    "pool",
                    TensorType::f32(vec![8, 4]),
                    StateRole::Positional { axis: 0 },
                )
                .unwrap();
            let new = b
                .slot(Slot::Activation, TensorType::f32(vec![2, 4]))
                .unwrap();
            let write = b.write_map("write", 2, 8).unwrap();
            let read = b
                .slot_named(Slot::SlotMap, "read", TensorType::new(vec![3], DType::I32))
                .unwrap();
            let written = b.scatter_update(pool, new, write).unwrap();
            let stale = b.gather(pool, 0, read).unwrap();
            let out = b
                .binary(BinOp::Add, stale, b.gather(written, 0, read).unwrap())
                .unwrap();
            let g = b
                .finish(poot_graph_ir::Outputs::new(out).with_state(&[(pool, written)]))
                .unwrap();
            (g, written.id())
        };
        let not_state = || {
            let b = Builder::for_tests();
            let base = b
                .slot(Slot::Activation, TensorType::f32(vec![8, 4]))
                .unwrap();
            let new = b
                .slot_named(Slot::Activation, "new", TensorType::f32(vec![2, 4]))
                .unwrap();
            let write = b.write_map("write", 2, 8).unwrap();
            let written = b.scatter_update(base, new, write).unwrap();
            (b.finish(written).unwrap(), written.id())
        };
        for (what, (graph, scatter)) in [
            ("a pool read before the write", reads_old_pool()),
            ("a base that is not state", not_state()),
        ] {
            match compile(&graph, &wgpu(), &REPLAY) {
                Err(CompileError::Plan(error)) => assert!(
                    matches!(*error, PlanError::ScatterNotInPlace { value } if value == scatter),
                    "{what}: {error:?}"
                ),
                other => panic!(
                    "{what}: expected ScatterNotInPlace, got {:?}",
                    other.map(|_| ())
                ),
            }
        }
    }

    /// POOT-1149: attention that reads its K and V pools in place must read the pools the step wrote. A paged
    /// attention over the pools as they were before this step's write is not buildable: the pools' scatter
    /// updates donate their buffers, and a reader of the pre-write pool makes that impossible. `compile`
    /// refuses it with `ScatterNotInPlace`, naming the K scatter, where the same attention over the written
    /// pools compiles and fuses to one `Paged` attention.
    ///
    /// Mutation: in `write_paged`, view the pool state input instead of the scatter's output (the models' own
    /// tracer): `a_paged_decode_of_the_real_qwen2_config_dispatches_no_gather_or_transpose_of_the_pools` goes
    /// red with `the graph compiles: Plan(ScatterNotInPlace { value: v66 })`; and drop the `ScatterNotInPlace`
    /// check in `state_commit_from_plans` and this row goes red with "expected ScatterNotInPlace, got Ok".
    #[test]
    fn a_paged_attention_over_the_pools_before_the_write_is_refused_and_after_it_fuses() {
        let build = |stale: bool| {
            let b = Builder::for_tests();
            let pool = |name: &str| {
                b.state_input(
                    name,
                    TensorType::f32(vec![8, 2, 4]),
                    StateRole::Positional { axis: 0 },
                )
                .unwrap()
            };
            let (k_pool, v_pool) = (pool("k_pool"), pool("v_pool"));
            let new = |name: &str| {
                b.slot_named(Slot::Activation, name, TensorType::f32(vec![2, 2, 4]))
                    .unwrap()
            };
            let write = b.write_map("write", 2, 8).unwrap();
            let read = b.read_map("read", 1, 3, 8).unwrap();
            let k_written = b.scatter_update(k_pool, new("new_k"), write).unwrap();
            let v_written = b.scatter_update(v_pool, new("new_v"), write).unwrap();
            let (k_read, v_read) = if stale {
                (k_pool, v_pool)
            } else {
                (k_written, v_written)
            };
            let q = b
                .slot_named(Slot::Activation, "q", TensorType::f32(vec![1, 2, 1, 4]))
                .unwrap();
            let mask = b
                .slot_named(Slot::Activation, "mask", TensorType::f32(vec![1, 1, 1, 3]))
                .unwrap();
            let k = poot_graph_ir::ops::paged_view(&b, k_read, read).unwrap();
            let v = poot_graph_ir::ops::paged_view(&b, v_read, read).unwrap();
            let out = poot_graph_ir::ops::attention_masked(&b, q, k, v, 1, 0.5, mask).unwrap();
            let graph = b
                .finish(
                    poot_graph_ir::Outputs::new(out)
                        .with_state(&[(k_pool, k_written), (v_pool, v_written)]),
                )
                .unwrap();
            (graph, k_written.id())
        };
        let (stale, k_scatter) = build(true);
        match compile(&stale, &wgpu(), &REPLAY) {
            Err(CompileError::Plan(error)) => assert!(
                matches!(*error, PlanError::ScatterNotInPlace { value } if value == k_scatter),
                "{error:?}"
            ),
            other => panic!("expected ScatterNotInPlace, got {:?}", other.map(|_| ())),
        }
        let (fresh, _) = build(false);
        let program = compile(&fresh, &wgpu(), &REPLAY).expect("the post-write read compiles");
        let keys: Vec<_> = program
            .graph()
            .eqns()
            .iter()
            .filter_map(|eqn| match &eqn.op {
                OpKind::Attention(attrs) => Some(attrs.key),
                _ => None,
            })
            .collect();
        assert_eq!(keys, [poot_graph_ir::KeyLayout::Paged]);
    }

    /// POOT-1074: an I32 state pool read by a `Gather` and written back by a `ScatterUpdate` (the paged PLE
    /// token history) plans on every backend, each reading the pool as I32 words.
    #[test]
    fn compile_plans_an_i32_state_pool_gathered_and_scattered_on_one_lane() {
        let b = Builder::for_tests();
        let pool = b
            .state_input(
                "pool",
                TensorType::new(vec![4, 3], DType::I32),
                StateRole::Recurrent,
            )
            .unwrap();
        let read = b
            .slot(Slot::Pos, TensorType::new(vec![2], DType::I32))
            .unwrap();
        let write = b.write_map("write", 2, 4).unwrap();
        let x = b
            .slot(Slot::Activation, TensorType::new(vec![2, 3], DType::I32))
            .unwrap();
        let next = b
            .binary(BinOp::Add, b.gather(pool, 0, read).unwrap(), x)
            .unwrap();
        let updated = b.scatter_update(pool, next, write).unwrap();
        let g = b
            .finish(poot_graph_ir::Outputs::new(next).with_state(&[(pool, updated)]))
            .unwrap();
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
        ] {
            compile(&g, &target_for(backend), &REPLAY)
                .unwrap_or_else(|error| panic!("{backend:?}: {error:?}"));
        }
    }

    /// POOT-1074: an I32 pool written by a `ScatterUpdate` moves its source's words as I32. Here the
    /// updated pool is gathered as a table while the source is also cast to F32 by another state's update;
    /// every reader of the source reads `Slice<i32>`.
    #[test]
    fn compile_plans_an_i32_scatter_update_source_beside_a_cast_reader() {
        let b = Builder::for_tests();
        let pool = b
            .state_input(
                "pool",
                TensorType::new(vec![4, 3], DType::I32),
                StateRole::Recurrent,
            )
            .unwrap();
        let seen = b
            .state_input("seen", TensorType::f32(vec![2, 3]), StateRole::Recurrent)
            .unwrap();
        // The cast reader's words are proven exact: the source declares its range.
        let src = b
            .slot_ranged(
                Slot::Activation,
                TensorType::new(vec![2, 3], DType::I32),
                IndexRange::below(12).unwrap(),
            )
            .unwrap();
        let slots = b.write_map("write", 2, 4).unwrap();
        let at = b
            .slot(Slot::Pos, TensorType::new(vec![2], DType::I32))
            .unwrap();
        let updated = b.scatter_update(pool, src, slots).unwrap();
        let rows = b.gather(updated, 0, at).unwrap();
        let seen_next = b
            .binary(BinOp::Add, seen, b.cast(src, DType::F32).unwrap())
            .unwrap();
        let g = b
            .finish(
                poot_graph_ir::Outputs::new(rows).with_state(&[(pool, updated), (seen, seen_next)]),
            )
            .unwrap();
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
        ] {
            compile(&g, &target_for(backend), &REPLAY)
                .unwrap_or_else(|error| panic!("{backend:?}: {error:?}"));
        }
    }

    // POOT-726: which `I32 -> F32` casts `compile` proves exact, from value ranges alone.

    fn compile_verdict(g: &Graph) -> Result<(), CompileError> {
        compile(g, &wgpu(), &REPLAY).map(|_| ())
    }

    fn refused_cast(error: CompileError) -> CastRangeError {
        match error {
            CompileError::CastRange(error) => *error,
            other => panic!("expected a cast-range refusal, got {other:?}"),
        }
    }

    /// SC-001: an unbounded slot and `Add(pos, big)` past `2^24` are refused with a typed error naming
    /// the cast equation and its operand; the proven range is in the error.
    ///
    /// Mutation (never left in the tree): delete the `refuse_unproven_i32_casts(g)?` call in
    /// `compile`; both graphs compile and this test goes red at `unwrap_err`.
    #[test]
    fn compile_refuses_an_i32_cast_the_graph_does_not_prove_exact() {
        let b = Builder::for_tests();
        let ids = b
            .slot(Slot::Token, TensorType::new(vec![4], DType::I32))
            .unwrap();
        let cast = b.cast(ids, DType::F32).unwrap();
        let g = b.finish(cast).unwrap();
        let error = refused_cast(compile_verdict(&g).unwrap_err());
        assert_eq!(
            error,
            CastRangeError {
                cast: cast.id(),
                operand: ids.id(),
                proven: IndexRange::I32,
                exact: IndexRange::F32_EXACT,
            }
        );

        // `pos` is `[0, 16)`, so `pos + 2^24` is `[2^24, 2^24 + 16)`: one lane past the exact integers.
        let b = Builder::for_tests();
        let pos = b
            .slot_ranged(
                Slot::Pos,
                TensorType::new(vec![4], DType::I32),
                IndexRange::below(16).unwrap(),
            )
            .unwrap();
        let late = b
            .binary_scalar(BinOp::Add, pos, Scalar::I32(1 << 24))
            .unwrap();
        let cast = b.cast(late, DType::F32).unwrap();
        let g = b.finish(cast).unwrap();
        let error = refused_cast(compile_verdict(&g).unwrap_err());
        assert_eq!(error.cast, cast.id());
        assert_eq!(error.operand, late.id());
        assert_eq!(
            error.proven,
            IndexRange::new(1 << 24, (1 << 24) + 16).unwrap()
        );
        assert!(
            error
                .to_string()
                .contains(&format!("equation {}", cast.id()))
        );
    }

    /// SC-001: a declared position, `ArgTopK` ids, a `guard_index_bounds` output and the successor of one
    /// compile, with no authorization of any kind.
    ///
    /// Mutation (never left in the tree): make `Select`'s rule in `poot-graph-ir`'s `range.rs` drop its
    /// branch narrowing (union of the two raw branches); the guarded rows are refused and this goes red.
    /// Mutation: make the `Add` rule return the whole `i32` range; the `Add(guarded, 1)` row is refused.
    #[test]
    fn compile_proves_declared_positions_top_k_ids_and_guarded_selectors() {
        // A declared position.
        let b = Builder::for_tests();
        let pos = b
            .slot_ranged(
                Slot::Pos,
                TensorType::new(vec![4], DType::I32),
                IndexRange::below(1 << 20).unwrap(),
            )
            .unwrap();
        let out = b.cast(pos, DType::F32).unwrap();
        let g = b.finish(out).unwrap();
        compile_verdict(&g).expect("a declared position is exact");

        // `ArgTopK` ids over an extent of 16.
        let b = Builder::for_tests();
        let scores = b.constant("scores", TensorType::f32(vec![2, 16])).unwrap();
        let ids = b.arg_top_k(scores, 4).unwrap();
        let out = b.cast(ids, DType::F32).unwrap();
        let g = b.finish(out).unwrap();
        compile_verdict(&g).expect("top-k ids are below the extent");

        // A guard output over an unbounded selector, and its successor (the live length).
        for successor in [false, true] {
            let b = Builder::for_tests();
            let raw = b
                .slot(Slot::SlotMap, TensorType::new(vec![4], DType::I32))
                .unwrap();
            let guard = b.guard_index_bounds(raw, 4).unwrap();
            let used = if successor {
                b.binary_scalar(BinOp::Add, guard.guarded, Scalar::I32(1))
                    .unwrap()
            } else {
                guard.guarded
            };
            let out = b.cast(used, DType::F32).unwrap();
            let g = b.finish(out).unwrap();
            compile_verdict(&g)
                .unwrap_or_else(|error| panic!("guarded (successor {successor}): {error}"));
        }
    }

    /// SC-002: every cast the deleted `cast_authority` used to authorize, as a table. `range` is what
    /// the analysis says; `authority` is what the authority said for the same graph. The two differ on
    /// exactly the rows where the authority leant on a caller claim (the host-checked selector's
    /// `expert_count`) or on a shape (a guard over arithmetic, a guard bound the claim never checked).
    ///
    /// Mutation (never left in the tree): delete `refuse_unproven_i32_casts(g)?` from `compile`; every
    /// refused row compiles and this test goes red naming the row.
    #[test]
    fn compile_gives_the_cast_authority_cases_a_range_verdict() {
        use poot_graph_ir::{Outputs, ValidationId};

        #[derive(Debug, PartialEq, Eq)]
        enum Verdict {
            Proved,
            Refused,
        }
        fn verdict(g: &Graph) -> Verdict {
            match compile_verdict(g) {
                Ok(()) => Verdict::Proved,
                Err(CompileError::CastRange(_)) => Verdict::Refused,
                Err(other) => panic!("a refusal other than the cast range: {other:?}"),
            }
        }
        fn guarded_selector(bound: usize) -> Graph {
            // The `exact_i32.rs` selector chain: a Gather over an I32 weight table, the guard, the
            // selector Cast as the output, the guard's flag as the validation output.
            let b = Builder::for_tests();
            let table = b
                .constant("route.table", TensorType::new(vec![4], DType::I32))
                .unwrap();
            let indices = b
                .constant("route.indices", TensorType::new(vec![4], DType::I32))
                .unwrap();
            let raw = b.gather(table, 0, indices).unwrap();
            let guard = b.guard_index_bounds(raw, bound).unwrap();
            let selected = b.cast(guard.guarded, DType::F32).unwrap();
            b.finish(Outputs::new(selected).with_witness(
                ValidationId(3),
                "route.bounds",
                guard.witness,
            ))
            .unwrap()
        }

        // (row, authority's verdict, range analysis' verdict, graph)
        let mut rows: Vec<(&str, Verdict, Verdict, Verdict)> = Vec::new();

        // HostCheckedSelector: `Cast(Gather(table, index))` of an I32 weight table. The authority took
        // the caller's `expert_count`; nothing in the graph bounds the table.
        let b = Builder::for_tests();
        let table = b
            .constant("table", TensorType::new(vec![3], DType::I32))
            .unwrap();
        let index = b
            .constant("index", TensorType::new(vec![2], DType::I32))
            .unwrap();
        let out = b
            .cast(b.gather(table, 0, index).unwrap(), DType::F32)
            .unwrap();
        let g = b.finish(out).unwrap();
        rows.push((
            "host-checked selector over a weight table",
            Verdict::Proved,
            Verdict::Refused,
            verdict(&g),
        ));

        // The same selector over a table the graph defines (an iota): bounded by its definition.
        let b = Builder::for_tests();
        let table = b.iota_i32(3).unwrap();
        let index = b
            .constant("index", TensorType::new(vec![2], DType::I32))
            .unwrap();
        let out = b
            .cast(b.gather(table, 0, index).unwrap(), DType::F32)
            .unwrap();
        let g = b.finish(out).unwrap();
        rows.push((
            "host-checked selector over a defined table",
            Verdict::Proved,
            Verdict::Proved,
            verdict(&g),
        ));

        // GuardedSelector over a Gather: the guard's range, however the table was produced.
        rows.push((
            "guarded selector over a weight table",
            Verdict::Proved,
            Verdict::Proved,
            verdict(&guarded_selector(4)),
        ));

        // The guard bound 2^25 the authority never read: a selector the f32 cannot hold.
        rows.push((
            "guarded selector with a forged bound",
            Verdict::Proved,
            Verdict::Refused,
            verdict(&guarded_selector(1 << 25)),
        ));

        // Witness: the guard's flag cast, `[0, 2)` from the comparison rule alone.
        let b = Builder::for_tests();
        let raw = b
            .constant("raw", TensorType::new(vec![4], DType::I32))
            .unwrap();
        let flag = b.binary_scalar(BinOp::GeU, raw, Scalar::I32(4)).unwrap();
        let out = b.cast(flag, DType::F32).unwrap();
        let g = b.finish(out).unwrap();
        rows.push((
            "witness lane flag",
            Verdict::Proved,
            Verdict::Proved,
            verdict(&g),
        ));

        // An authorized cast whose source is neither a gather nor a guard nor a comparison.
        let b = Builder::for_tests();
        let left = b
            .constant("left", TensorType::new(vec![2], DType::I32))
            .unwrap();
        let right = b
            .constant("right", TensorType::new(vec![2], DType::I32))
            .unwrap();
        let out = b
            .cast(b.binary(BinOp::Sub, left, right).unwrap(), DType::F32)
            .unwrap();
        let g = b.finish(out).unwrap();
        rows.push((
            "difference of two weight words",
            Verdict::Refused,
            Verdict::Refused,
            verdict(&g),
        ));

        for (row, authority, expected, actual) in &rows {
            assert_eq!(actual, expected, "{row}: authority said {authority:?}");
        }
        // The rows where the verdicts differ are the authority's two holes, not a regression.
        let differing: Vec<&str> = rows
            .iter()
            .filter(|(_, authority, range, _)| authority != range)
            .map(|(row, ..)| *row)
            .collect();
        assert_eq!(
            differing,
            [
                "host-checked selector over a weight table",
                "guarded selector with a forged bound"
            ]
        );
    }

    /// POOT-1119 rows: `IndexedMatMul(x [3, 4], w [EXPERTS, 4, 5], ids [3])` over `ids` as built by
    /// `selector`, through the real `compile`.
    mod expert_ids {
        use super::*;
        use poot_graph_ir::{BuilderError, Outputs, Traced, ValidationId};

        const EXPERTS: usize = 4;

        type Ids = fn(&Builder) -> Result<Traced, BuilderError>;

        fn matmul(ids: impl FnOnce(&Builder) -> Result<Traced, BuilderError>) -> (Graph, Traced) {
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![3, 4])).unwrap();
            let w = b
                .constant("w", TensorType::f32(vec![EXPERTS, 4, 5]))
                .unwrap();
            let ids = ids(&b).unwrap();
            let out = b.indexed_matmul(x, w, ids).unwrap();
            (b.finish(out).unwrap(), out)
        }

        fn refused(error: CompileError) -> ExpertIdsError {
            match error {
                CompileError::ExpertIds(error) => *error,
                other => panic!("expected an expert-ids refusal, got {other:?}"),
            }
        }

        /// Ids that are an unbounded slot are refused, naming the equation, the ids and the proven range.
        ///
        /// Mutation (never left in the tree): call `refuse_unproven_expert_ids(g)` in `compile` and drop
        /// its result; the graph compiles and this goes red at `unwrap_err`.
        #[test]
        fn compile_refuses_ids_the_graph_does_not_prove_inside_the_experts() {
            let mut ids = None;
            let (g, out) = matmul(|b| {
                let t = b.slot_named(
                    Slot::Activation,
                    "ids",
                    TensorType::new(vec![3], DType::I32),
                )?;
                ids = Some(t.id());
                Ok(t)
            });
            let error = refused(compile_verdict(&g).unwrap_err());
            assert_eq!(
                error,
                ExpertIdsError {
                    eqn: out.id(),
                    op: "indexed_matmul".to_owned(),
                    ids: ids.unwrap(),
                    experts: EXPERTS,
                    proven: IndexRange::I32,
                    gap: ExpertIdsGap::OutOfRange,
                }
            );
            assert!(
                error
                    .to_string()
                    .contains(&format!("equation {}", out.id()))
            );

            // A declared range that is one expert too wide is no proof either.
            let (g, _) = matmul(|b| {
                b.slot_named_ranged(
                    Slot::Activation,
                    "ids",
                    TensorType::new(vec![3], DType::I32),
                    IndexRange::below(EXPERTS + 1).unwrap(),
                )
            });
            let error = refused(compile_verdict(&g).unwrap_err());
            assert_eq!(error.proven, IndexRange::below(EXPERTS + 1).unwrap());
            assert_eq!(error.gap, ExpertIdsGap::OutOfRange);
        }

        /// A declared slot, a `Gather` of a ranged weight table, and `ArgTopK` ids are admitted, with no
        /// caller-supplied expert count anywhere.
        ///
        /// Mutation (never left in the tree): make `Gather`'s rule in `range.rs` return the whole `i32`
        /// range; the gathered-table row is refused and this goes red naming it.
        #[test]
        fn compile_admits_declared_gathered_and_top_k_ids() {
            let admitted = |row: &str, ids: Ids| {
                let (g, _) = matmul(ids);
                compile_verdict(&g).unwrap_or_else(|error| panic!("{row}: {error}"));
            };
            admitted("declared slot", |b| {
                b.slot_named_ranged(
                    Slot::Activation,
                    "ids",
                    TensorType::new(vec![3], DType::I32),
                    IndexRange::below(EXPERTS).unwrap(),
                )
            });
            admitted("gather of a ranged table", |b| {
                let table = b.constant_ranged(
                    "route.table",
                    TensorType::new(vec![6], DType::I32),
                    IndexRange::below(EXPERTS).unwrap(),
                )?;
                let tokens = b.constant("tokens", TensorType::new(vec![3], DType::I32))?;
                b.gather(table, 0, tokens)
            });
            admitted("arg top-k", |b| {
                let scores = b.constant("scores", TensorType::f32(vec![1, EXPERTS]))?;
                let ids = b.arg_top_k(scores, 3)?;
                b.reshape(ids, vec![3])
            });
        }

        /// The traced packed MoE projection (`packed_indexed_linear`, a `PackedContraction` over an indexed
        /// stack, not an `IndexedMatMul`) is held to the same proof: unproven ids are refused with the
        /// stack's declared expert count, and the same graph over a ranged selector compiles.
        ///
        /// Mutation (never left in the tree): drop the `PackedContraction` arm of
        /// `refuse_unproven_expert_ids`; the unproven graph compiles and this goes red.
        #[test]
        fn a_packed_indexed_projection_needs_proven_ids_too() {
            let build = |declared: bool| {
                let b = Builder::for_tests();
                let x = b
                    .slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![3, STACK_K]))
                    .unwrap();
                let shape = TensorType::new(vec![3], DType::I32);
                let selector = if declared {
                    b.slot_named_ranged(
                        Slot::Activation,
                        "moe-ids",
                        shape,
                        IndexRange::below(MOE_EXPERTS).unwrap(),
                    )
                } else {
                    b.slot_named(Slot::Activation, "moe-ids", shape)
                }
                .unwrap();
                let table = poot_graph_ir::ops::PackedExpertStack {
                    stack_id: "moe.experts".to_string(),
                    descriptor: stack_descriptor(poot_quant::format::WeightFormat::Q8_0),
                    experts: MOE_EXPERTS,
                };
                let out =
                    poot_graph_ir::ops::packed_indexed_linear(&b, x, selector, &table).unwrap();
                (b.finish(out).unwrap(), out.id(), selector.id())
            };
            let (declared, ..) = build(true);
            compile_verdict(&declared).unwrap_or_else(|error| panic!("declared ids: {error}"));
            let (unproven, eqn, ids) = build(false);
            let error = refused(compile_verdict(&unproven).unwrap_err());
            assert_eq!(
                (error.eqn, error.ids, error.experts),
                (eqn, ids, MOE_EXPERTS)
            );
            assert!(error.op.starts_with("packed_contraction"), "{}", error.op);
            assert_eq!(error.gap, ExpertIdsGap::OutOfRange);
        }

        /// A guard puts ids in `[0, E)` by replacing the lanes outside it, so it admits them only when its
        /// witness is a declared validation output: the same graph is refused without the declaration.
        ///
        /// Mutation (never left in the tree): drop the `first_unobserved_clamp` check from
        /// `refuse_unproven_expert_ids`; the undeclared row compiles and this goes red.
        #[test]
        fn a_guarded_id_needs_its_witness_declared() {
            let build = |declared: bool| {
                let b = Builder::for_tests();
                let x = b.constant("x", TensorType::f32(vec![3, 4])).unwrap();
                let w = b
                    .constant("w", TensorType::f32(vec![EXPERTS, 4, 5]))
                    .unwrap();
                let raw = b
                    .slot_named(
                        Slot::Activation,
                        "ids",
                        TensorType::new(vec![3], DType::I32),
                    )
                    .unwrap();
                let guard = b.guard_index_bounds(raw, EXPERTS).unwrap();
                let out = b.indexed_matmul(x, w, guard.guarded).unwrap();
                let outputs = Outputs::new(out);
                let outputs = if declared {
                    outputs.with_witness(ValidationId(0), "ids.bounds", guard.witness)
                } else {
                    outputs
                };
                (b.finish(outputs).unwrap(), out.id(), guard.guarded.id())
            };
            let (declared, ..) = build(true);
            compile_verdict(&declared).unwrap_or_else(|error| panic!("declared witness: {error}"));

            let (undeclared, matmul, ids) = build(false);
            let error = refused(compile_verdict(&undeclared).unwrap_err());
            assert_eq!((error.eqn, error.ids), (matmul, ids));
            assert_eq!(error.proven, IndexRange::below(EXPERTS).unwrap());
            assert!(
                matches!(error.gap, ExpertIdsGap::UnobservedGuard { .. }),
                "{error:?}"
            );
        }
    }

    /// POOT-1074: an I32 carried value is one buffer across steps. The state `s` is read as a `Gather` index
    /// and its next value is `GeU(slot, 3)`. Both are dense I32 words on every backend.
    #[test]
    fn compile_stores_a_carried_i32_state_and_its_next_value_as_i32() {
        let b = Builder::for_tests();
        let s = b
            .state_input(
                "s",
                TensorType::new(vec![4], DType::I32),
                StateRole::Recurrent,
            )
            .unwrap();
        let table = b.constant("table", TensorType::f32(vec![16, 8])).unwrap();
        let gathered = b.gather(table, 0, s).unwrap();
        let pos = b
            .slot(Slot::Pos, TensorType::new(vec![4], DType::I32))
            .unwrap();
        let next = b.binary_scalar(BinOp::GeU, pos, Scalar::I32(3)).unwrap();
        let g = b
            .finish(poot_graph_ir::Outputs::new(gathered).with_state(&[(s, next)]))
            .unwrap();
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
        ] {
            let program = compile(&g, &target_for(backend), &REPLAY)
                .unwrap_or_else(|error| panic!("{backend:?}: {error:?}"));
            for value in [s.id(), next.id()] {
                assert_eq!(
                    program.storage().storage(value).buffer_storage(),
                    poot_target::BufferStorage::i32(),
                    "{backend:?} {value}"
                );
            }
        }
    }

    /// The same agreement reached through an alias: the I32 `[4]` slot is the Gather's index, and its
    /// `Reshape` round trip (a `Plan::Alias` of the slot's buffer) is the scatter's slot map. `compile`
    /// plans the slot as dense I32 for both readers.
    #[test]
    fn compile_reads_an_i32_buffer_on_one_lane_through_an_alias() {
        let b = Builder::for_tests();
        let idx = b
            .slot(Slot::Pos, TensorType::new(vec![4], DType::I32))
            .unwrap();
        let table = b.constant("table", TensorType::f32(vec![16, 8])).unwrap();
        let rows = b.gather(table, 0, idx).unwrap();
        let late = b
            .cast(
                b.binary_scalar(BinOp::GeU, idx, Scalar::I32(3)).unwrap(),
                DType::F32,
            )
            .unwrap();
        let update = b
            .binary(
                BinOp::Add,
                rows,
                b.broadcast(b.reshape(late, vec![4, 1]).unwrap(), vec![4, 8])
                    .unwrap(),
            )
            .unwrap();
        let cache = b
            .state_input("cache", TensorType::f32(vec![4, 8]), StateRole::Recurrent)
            .unwrap();
        let inverse = b
            .reshape(b.reshape(idx, vec![2, 2]).unwrap(), vec![4])
            .unwrap();
        let written = b.scatter_update(cache, update, inverse).unwrap();
        let g = b
            .finish(poot_graph_ir::Outputs::new(written).with_state(&[(cache, written)]))
            .unwrap();

        let program = compile(&g, &wgpu(), &REPLAY).expect("the slot's buffer is read as i32");
        let &(slot, _) = program
            .graph()
            .slots()
            .iter()
            .find(|&&(_, kind)| kind == Slot::Pos)
            .expect("the Pos slot survives compile");
        assert_eq!(
            program.storage().storage(slot).buffer_storage(),
            poot_target::BufferStorage::i32()
        );
    }

    // mutants-m4 H5: the packed-chain recognizers' `consumers[x].len() != 1 || is_graph_escape(g, x)`
    // guards, one per chain intermediate, and the escape gate's own transpose consumer count.

    /// The three chains `recognize_packed_contractions` claims, built from primitives exactly as
    /// `ops::packed_linear`, `ops::packed_block_diagonal_linear` and the reversed `(W x^T)^T`
    /// spelling stage them.
    #[derive(Clone, Copy, Debug)]
    enum PackedChain {
        /// `PackedDequant -> Transpose([1,0]) -> [Reshape] -> MatMul`.
        Canonical { reshape: bool },
        /// `PackedDequant -> Reshape([2, out/2, K]) -> Transpose([0,2,1]) -> MatMul`.
        Blocked,
        /// `x^T = Transpose(x)`, `PackedDequant -> [Reshape] -> MatMul(W, x^T) -> Transpose([1,0])`.
        Transposed { reshape: bool },
    }

    /// How a chain intermediate stays live outside the chain: `Escape` makes it a validation
    /// output (a liveness root, single consumer); `Consumer` gives it a second consumer (a reduce the
    /// validation output observes) without making it a root.
    #[derive(Clone, Copy, Debug)]
    enum Live {
        Escape,
        Consumer,
    }

    /// `chain` over a Q8_0 `[8, 64]` weight, with the intermediate `member` (if any) kept live by
    /// `live`. Returns the graph and each member's value.
    fn packed_chain_graph(
        chain: PackedChain,
        live: Option<(&str, Live)>,
    ) -> (
        poot_graph_ir::Graph,
        Vec<(&'static str, poot_graph_ir::ValueId)>,
    ) {
        let b = Builder::for_tests();
        let mut members = Vec::new();
        let out = match chain {
            PackedChain::Canonical { reshape } => {
                let shape = if reshape {
                    vec![1, 2, MOE_K]
                } else {
                    vec![2, MOE_K]
                };
                let x = b
                    .slot_named(Slot::Activation, "chain-x", TensorType::f32(shape))
                    .unwrap();
                let dequant = staged_dequant(&b, "layer");
                let transpose = b.transpose(dequant, vec![1, 0]).unwrap();
                members.extend([("dequant", dequant), ("transpose", transpose)]);
                let weight = if reshape {
                    let reshaped = b.reshape(transpose, vec![1, MOE_K, MOE_OUT]).unwrap();
                    members.push(("reshape", reshaped));
                    reshaped
                } else {
                    transpose
                };
                b.matmul(x, weight).unwrap()
            }
            PackedChain::Blocked => {
                let x = b
                    .slot_named(
                        Slot::Activation,
                        "chain-x",
                        TensorType::f32(vec![2, 2, MOE_K]),
                    )
                    .unwrap();
                let dequant = staged_dequant(&b, "layer");
                let reshape = b.reshape(dequant, vec![2, MOE_OUT / 2, MOE_K]).unwrap();
                let transpose = b.transpose(reshape, vec![0, 2, 1]).unwrap();
                members.extend([
                    ("dequant", dequant),
                    ("reshape", reshape),
                    ("transpose", transpose),
                ]);
                b.matmul(x, transpose).unwrap()
            }
            PackedChain::Transposed { reshape } => {
                let x = b
                    .slot_named(Slot::Activation, "chain-x", TensorType::f32(vec![2, MOE_K]))
                    .unwrap();
                let activation_t = b.transpose(x, vec![1, 0]).unwrap();
                let dequant = staged_dequant(&b, "layer");
                members.extend([("activation_t", activation_t), ("dequant", dequant)]);
                let weight = if reshape {
                    let reshaped = b.reshape(dequant, vec![MOE_OUT, MOE_K]).unwrap();
                    members.push(("reshape", reshaped));
                    reshaped
                } else {
                    dequant
                };
                let product = b.matmul(weight, activation_t).unwrap();
                members.push(("product", product));
                b.transpose(product, vec![1, 0]).unwrap()
            }
        };
        let witness = live.map(|(name, live)| {
            let &(_, member) = members
                .iter()
                .find(|(member, _)| *member == name)
                .unwrap_or_else(|| panic!("{chain:?} has no member {name}"));
            match live {
                Live::Escape => member,
                Live::Consumer => b.reduce(RedOp::Sum, member, 0, false).unwrap(),
            }
        });
        let validations: Vec<(poot_graph_ir::ValidationId, &str, poot_graph_ir::Traced)> = witness
            .into_iter()
            .map(|witness| (poot_graph_ir::ValidationId(1), "live-member", witness))
            .collect();
        let graph = poot_graph_ir::test_support::graph_fixtures::finish_with_validations(
            b,
            out,
            &validations,
        )
        .unwrap();
        let members = members
            .into_iter()
            .map(|(name, member)| (name, member.id()))
            .collect();
        (graph, members)
    }

    /// Unmodified, every chain compiles to one planned `PackedContraction`: the refusals below are
    /// the live intermediate's, not a chain the recognizer never took.
    #[test]
    fn compile_claims_every_packed_chain_spelling() {
        for chain in [
            PackedChain::Canonical { reshape: false },
            PackedChain::Canonical { reshape: true },
            PackedChain::Blocked,
            PackedChain::Transposed { reshape: false },
            PackedChain::Transposed { reshape: true },
        ] {
            let (graph, _) = packed_chain_graph(chain, None);
            let program = compile(&graph, &wgpu(), &REPLAY)
                .unwrap_or_else(|error| panic!("{chain:?}: {error}"));
            assert!(
                program
                    .planned()
                    .any(|(eqn, _)| matches!(eqn.op, OpKind::PackedContraction { .. })),
                "{chain:?}: no PackedContraction planned"
            );
            assert!(
                !program
                    .planned()
                    .any(|(eqn, _)| matches!(eqn.op, OpKind::PackedDequant { .. })),
                "{chain:?}: a PackedDequant survived"
            );
        }
    }

    /// The escape gate's refusal a live intermediate must produce.
    #[derive(Debug)]
    enum Gate {
        ValidationOutput,
        /// `ConsumerCount` naming this member as the shared internal value.
        ConsumerCount(&'static str),
        Movement,
        Unmatched,
    }

    /// Mutant H5 (mutants-m4.md): every recognizer guard `consumers[x].len() != 1 ||
    /// is_graph_escape(g, x)` could be made `&&`, because no fixture kept a chain intermediate live
    /// outside the chain. If a recognizer fuses such a chain, the fused op deletes a member another
    /// consumer still reads. Each row keeps one intermediate live one way; `compile` must leave the
    /// chain unclaimed and refuse it at the escape gate with the named variant. The canonical
    /// `transpose`/`Consumer` row also pins the gate's own transpose consumer count
    /// (`consumers[next.out].len() != 1`), which falls through to `Unmatched` when weakened.
    #[test]
    fn compile_refuses_a_packed_chain_with_a_live_intermediate() {
        use PackedChain::*;
        let rows: [(PackedChain, &str, Live, Gate); 22] = [
            (
                Canonical { reshape: false },
                "dequant",
                Live::Escape,
                Gate::ValidationOutput,
            ),
            (
                Canonical { reshape: false },
                "dequant",
                Live::Consumer,
                Gate::ConsumerCount("dequant"),
            ),
            (
                Canonical { reshape: false },
                "transpose",
                Live::Escape,
                Gate::Unmatched,
            ),
            (
                Canonical { reshape: false },
                "transpose",
                Live::Consumer,
                Gate::ConsumerCount("transpose"),
            ),
            (
                Canonical { reshape: true },
                "transpose",
                Live::Escape,
                Gate::Unmatched,
            ),
            (
                Canonical { reshape: true },
                "transpose",
                Live::Consumer,
                Gate::ConsumerCount("transpose"),
            ),
            (
                Canonical { reshape: true },
                "reshape",
                Live::Escape,
                Gate::Unmatched,
            ),
            (
                Canonical { reshape: true },
                "reshape",
                Live::Consumer,
                Gate::Unmatched,
            ),
            (Blocked, "dequant", Live::Escape, Gate::ValidationOutput),
            (
                Blocked,
                "dequant",
                Live::Consumer,
                Gate::ConsumerCount("dequant"),
            ),
            (Blocked, "reshape", Live::Escape, Gate::Movement),
            (Blocked, "reshape", Live::Consumer, Gate::Movement),
            (Blocked, "transpose", Live::Escape, Gate::Movement),
            (Blocked, "transpose", Live::Consumer, Gate::Movement),
            (
                Transposed { reshape: false },
                "activation_t",
                Live::Escape,
                Gate::Unmatched,
            ),
            (
                Transposed { reshape: false },
                "activation_t",
                Live::Consumer,
                Gate::Unmatched,
            ),
            (
                Transposed { reshape: false },
                "dequant",
                Live::Escape,
                Gate::ValidationOutput,
            ),
            (
                Transposed { reshape: false },
                "dequant",
                Live::Consumer,
                Gate::ConsumerCount("dequant"),
            ),
            (
                Transposed { reshape: false },
                "product",
                Live::Escape,
                Gate::Unmatched,
            ),
            (
                Transposed { reshape: false },
                "product",
                Live::Consumer,
                Gate::Unmatched,
            ),
            (
                Transposed { reshape: true },
                "reshape",
                Live::Escape,
                Gate::Movement,
            ),
            (
                Transposed { reshape: true },
                "reshape",
                Live::Consumer,
                Gate::Movement,
            ),
        ];
        for (chain, member, live, gate) in rows {
            let what = format!("{chain:?} {member} {live:?}");
            let (graph, members) = packed_chain_graph(chain, Some((member, live)));
            let value_of = |name: &str| {
                members
                    .iter()
                    .find(|(member, _)| *member == name)
                    .map(|&(_, value)| value)
                    .unwrap()
            };
            let error = match compile(&graph, &wgpu(), &REPLAY) {
                Err(CompileError::PackedDequant(error)) => error,
                other => panic!(
                    "{what}: expected the escape gate's {gate:?}, got {:?}",
                    other.map(|_| ())
                ),
            };
            let matched = match (&gate, &*error) {
                (Gate::ValidationOutput, PackedDequantProductionError::ValidationOutput { .. })
                | (Gate::Movement, PackedDequantProductionError::Movement { .. })
                | (Gate::Unmatched, PackedDequantProductionError::Unmatched { .. }) => true,
                (
                    Gate::ConsumerCount(name),
                    PackedDequantProductionError::ConsumerCount {
                        internal,
                        consumers: 2,
                        ..
                    },
                ) => *internal == value_of(name),
                _ => false,
            };
            assert!(matched, "{what}: expected {gate:?}, got {error:?}");
        }
    }

    /// Mutant H5 (mutants-m4.md), the recognizers' op checks: `!matches!(op, MatMul) ||
    /// inputs.len() != 2` made `&&` lets any two-input op stand in for the (inner) `MatMul`. With a
    /// square `[64, 64]` weight an elementwise `Add` has the contraction's output shape, so the
    /// shared infer-and-compare gate would not catch it: `Add(x, W^T)` and `(W + x^T)^T` must stay
    /// unclaimed and be refused as `Unmatched`, never compiled as a contraction.
    #[test]
    fn compile_refuses_a_square_packed_chain_under_a_non_matmul() {
        let square = poot_quant::PackedWeight::try_new(
            poot_quant::format::WeightFormat::Q8_0,
            [MOE_K, MOE_K],
        )
        .unwrap();
        for transposed in [false, true] {
            let b = Builder::for_tests();
            let x = b
                .slot_named(
                    Slot::Activation,
                    "square-x",
                    TensorType::f32(vec![MOE_K, MOE_K]),
                )
                .unwrap();
            let sources: Vec<poot_graph_ir::Traced> =
                poot_graph_ir::packed_source_constants("layer", square)
                    .into_iter()
                    .map(|(name, tensor_type)| b.constant(name.as_str(), tensor_type).unwrap())
                    .collect();
            let dequant = b.packed_dequant(&sources, square).unwrap();
            let out = if transposed {
                let x_t = b.transpose(x, vec![1, 0]).unwrap();
                let sum = b.binary(BinOp::Add, dequant, x_t).unwrap();
                b.transpose(sum, vec![1, 0]).unwrap()
            } else {
                let w_t = b.transpose(dequant, vec![1, 0]).unwrap();
                b.binary(BinOp::Add, x, w_t).unwrap()
            };
            let graph = b.finish(out).unwrap();
            match compile(&graph, &wgpu(), &REPLAY) {
                Err(CompileError::PackedDequant(error))
                    if matches!(*error, PackedDequantProductionError::Unmatched { .. }) => {}
                other => panic!(
                    "transposed={transposed}: expected Unmatched, got {:?}",
                    other.map(|_| ())
                ),
            }
        }
    }

    /// Mutant H5 (mutants-m4.md), the transposed recognizer's overlap check: `T(MatMul(A, B^T))`
    /// with `A` and `B` both packed dequants reads as a canonical chain (activation `A`, weight
    /// `B^T`) and as the reversed spelling (weight `A`, activation `B`). The first claim wins; with
    /// `replacements.contains_key(inner) || removed.contains(inner)` made `&&` the second claims the
    /// same `MatMul` too and each contraction deletes the other's operand. `A`'s dequant stays an
    /// unclaimed activation, so `compile` refuses it as `Unmatched`.
    #[test]
    fn compile_claims_a_matmul_of_two_packed_chains_once() {
        let b = Builder::for_tests();
        let a = staged_dequant(&b, "layer.a");
        let dequant_b = staged_dequant(&b, "layer.b");
        let b_t = b.transpose(dequant_b, vec![1, 0]).unwrap();
        let product = b.matmul(a, b_t).unwrap();
        let out = b.transpose(product, vec![1, 0]).unwrap();
        let graph = b.finish(out).unwrap();
        match compile(&graph, &wgpu(), &REPLAY) {
            Err(CompileError::PackedDequant(error))
                if matches!(*error, PackedDequantProductionError::Unmatched { .. }) => {}
            other => panic!("expected Unmatched, got {:?}", other.map(|_| ())),
        }
    }

    /// Mutant H5 (mutants-m4.md), `validate_packed_carriers`' state check: `state_in == carrier ||
    /// state_out == carrier` made `&&` admits a carrier on only one side of a carried-state pair
    /// (the existing fixture set both). A packed-linear weight carrier that is only a `state_out` is
    /// refused as a carried-state escape. The `state_in`-only side cannot reach this check through
    /// `compile`: graph validation refuses a `Const` state input first (`StateInputStorageMismatch`),
    /// and a `State`-storage carrier is `CarrierStorage`.
    #[test]
    fn compile_refuses_a_packed_carrier_that_is_only_a_state_output() {
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "state-x", TensorType::f32(vec![2, MOE_K]))
            .unwrap();
        let constants = poot_graph_ir::packed_source_constants("layer", moe_descriptor());
        let carrier_type = constants[0].1.clone();
        let sources: Vec<poot_graph_ir::Traced> = constants
            .into_iter()
            .map(|(name, tensor_type)| b.constant(name.as_str(), tensor_type).unwrap())
            .collect();
        let dequant = b.packed_dequant(&sources, moe_descriptor()).unwrap();
        let transpose = b.transpose(dequant, vec![1, 0]).unwrap();
        let out = b.matmul(x, transpose).unwrap();
        let state_in = b
            .state_input("carrier.state", carrier_type, StateRole::Recurrent)
            .unwrap();
        let graph = b
            .finish(poot_graph_ir::Outputs::new(out).with_state(&[(state_in, sources[0])]))
            .unwrap();
        match compile(&graph, &wgpu(), &REPLAY) {
            Err(CompileError::PackedDequant(error))
                if matches!(
                    *error,
                    PackedDequantProductionError::CarrierEscape {
                        destination: "carried state",
                        ..
                    }
                ) => {}
            other => panic!(
                "expected a carried-state CarrierEscape, got {:?}",
                other.map(|_| ())
            ),
        }
    }

    /// SC-003 (R-644-1): roles survive `compile`, the production path every graph takes. A fixture with
    /// one `Positional { axis: 2 }` KV-cache write and one `Recurrent` passthrough state compiles, and the
    /// program's graph reports each pair's traced role back through `state_pairs()`.
    ///
    /// Mutation (never left in the tree): `Graph::state_pairs()` returning `StateRole::Recurrent`
    /// unconditionally (ignoring `ValueMeta::state_role`) flips the positional pair's reported role to
    /// `Recurrent` and fails this test's first assertion.
    #[test]
    fn compile_state_pairs_report_the_traced_roles() {
        let b = Builder::for_tests();
        let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32)).unwrap();
        let kv_cache = b
            .state_input(
                "kv.cache",
                TensorType::f32(vec![1, 1, 4, 2]),
                StateRole::Positional { axis: 2 },
            )
            .unwrap();
        let update = b
            .constant("update", TensorType::f32(vec![1, 1, 1, 2]))
            .unwrap();
        let kv_out = b
            .dynamic_update_slice_dyn(kv_cache, update, pos_slot, 2)
            .unwrap();
        let gdn_state = b
            .state_input(
                "gdn.state",
                TensorType::f32(vec![2, 2]),
                StateRole::Recurrent,
            )
            .unwrap();
        let graph = b
            .finish(
                poot_graph_ir::Outputs::new(kv_out)
                    .with_state(&[(kv_cache, kv_out), (gdn_state, gdn_state)]),
            )
            .unwrap();

        let program = compile(&graph, &wgpu(), &REPLAY).expect("the state fixture compiles");
        let roles: std::collections::HashMap<ValueId, StateRole> = program
            .graph()
            .state_pairs()
            .map(|pair| (pair.input, pair.role))
            .collect();
        assert_eq!(
            roles[&kv_cache.id()],
            StateRole::Positional { axis: 2 },
            "the KV-cache pair's traced role must survive compile"
        );
        assert_eq!(
            roles[&gdn_state.id()],
            StateRole::Recurrent,
            "the recurrent pair's traced role must survive compile"
        );
    }

    // Card 545a: the packed row-gather claim (quantized token embeddings).

    fn embedding_descriptor() -> poot_quant::PackedWeight {
        poot_quant::PackedWeight::try_new(poot_quant::format::WeightFormat::Q6_K, [16, 256])
            .unwrap()
    }

    /// A packed embedding lookup over `ids_shape` row ids (`[]` is the decode token, `[L]` a
    /// prefill).
    fn packed_embedding_graph(ids_shape: Vec<usize>) -> Graph {
        let b = Builder::for_tests();
        let ids = b
            .slot(Slot::Token, TensorType::new(ids_shape, DType::I32))
            .unwrap();
        let rows = poot_graph_ir::ops::packed_embedding(
            &b,
            ids,
            "model.embed_tokens",
            embedding_descriptor(),
        )
        .unwrap();
        b.finish(rows).unwrap()
    }

    /// W16: `compile` claims `Gather(axis 0)` of a packed table as one planned `PackedRowGather` on
    /// every backend, for the decode token and a prefill id vector; no `PackedDequant` survives.
    /// Mutation: drop the `recognize_packed_row_gathers` pass; every row is refused by the gate.
    #[test]
    fn compile_claims_a_packed_embedding_as_a_planned_row_gather() {
        for backend in [
            Backend::SpirvVulkan,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
            Backend::Nvptx,
        ] {
            for ids_shape in [vec![], vec![5]] {
                let what = format!("{backend:?} ids {ids_shape:?}");
                let program = compile(
                    &packed_embedding_graph(ids_shape),
                    &target_for(backend),
                    &REPLAY,
                )
                .unwrap_or_else(|error| panic!("{what}: {error}"));
                let planned: Vec<(&Eqn, &Plan)> = program.planned().collect();
                assert!(
                    planned
                        .iter()
                        .all(|(eqn, _)| !matches!(eqn.op, OpKind::PackedDequant { .. })),
                    "{what}: a PackedDequant survived"
                );
                assert!(
                    planned.iter().any(|(eqn, plan)| matches!(
                        eqn.op,
                        OpKind::PackedRowGather { .. }
                    ) && plan
                        .single()
                        .is_some_and(|dispatch| !dispatch.launch().meta.is_empty())
                        && matches!(
                            program.kernel_choice(eqn),
                            KernelChoice::Generated(KernelRequest::Packed(request))
                                if request.name == "packed_row_gather"
                        )),
                    "{what}: no planned PackedRowGather"
                );
            }
        }
    }

    /// A former region source built from primitives. Mirrored by the wgpu acceptance fixture.
    fn former_region_chain_fixture() -> Graph {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![2, 5])).unwrap();
        let raw = b
            .constant("w.weight", TensorType::new(vec![3, 5], DType::E4M3FN))
            .unwrap();
        let decoded = b.cast(raw, DType::F32).unwrap();
        let scale = b
            .constant("w.weight_scale", TensorType::f32(vec![2, 3]))
            .unwrap();
        let out_index = b
            .constant(
                "w.fp8_out_block_index",
                TensorType::new(vec![3], DType::I32),
            )
            .unwrap();
        let in_index = b
            .constant("w.fp8_in_block_index", TensorType::new(vec![5], DType::I32))
            .unwrap();
        let rows = b.gather(scale, 0, out_index).unwrap();
        let expanded_scale = b.gather(rows, 1, in_index).unwrap();
        let scaled = b.binary(BinOp::Mul, decoded, expanded_scale).unwrap();
        let weight = b.transpose(scaled, vec![1, 0]).unwrap();
        let y = b.matmul(x, weight).unwrap();
        b.finish(y).unwrap()
    }

    fn dense_decode_fixture() -> Graph {
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(vec![1, 8]))
            .unwrap();
        let w = b.constant("w", TensorType::f32(vec![8, 8])).unwrap();
        let bias = b.constant("bias", TensorType::f32(vec![8])).unwrap();
        let y = poot_graph_ir::ops::linear(&b, x, w, Some(bias)).unwrap();
        b.finish(y).unwrap()
    }

    fn prefill_fixture() -> Graph {
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(vec![8, 8]))
            .unwrap();
        let w = b.constant("w", TensorType::f32(vec![8, 8])).unwrap();
        let bias = b.constant("bias", TensorType::f32(vec![8])).unwrap();
        let y = poot_graph_ir::ops::linear(&b, x, w, Some(bias)).unwrap();
        b.finish(y).unwrap()
    }

    fn packed_q8_0_linear_fixture() -> Graph {
        use poot_quant::PackedWeight;
        use poot_quant::format::WeightFormat;

        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(vec![1, 256]))
            .unwrap();
        let weight = PackedWeight::try_new(WeightFormat::Q8_0, [64, 256]).unwrap();
        let y = poot_graph_ir::ops::packed_linear(&b, x, "layer", weight, None, None).unwrap();
        b.finish(y).unwrap()
    }

    /// Submission mode does not affect the compiled plans.
    #[test]
    fn compile_output_does_not_depend_on_submission_mode() {
        let fixtures: [(&str, Graph); 4] = [
            ("dense_decode", dense_decode_fixture()),
            ("prefill", prefill_fixture()),
            ("packed_q8_0", packed_q8_0_linear_fixture()),
            ("former_region_chain", former_region_chain_fixture()),
        ];
        for (name, graph) in &fixtures {
            let replay = CompileOptions::new(Submission::Replay, crate::CompileLimits::STANDARD);
            let uncached = CompileOptions::new(Submission::Eager, crate::CompileLimits::STANDARD);
            let under_replay = compile(graph, &wgpu(), &replay)
                .unwrap_or_else(|error| panic!("{name}: CaptureReplay: {error}"));
            let under_uncached = compile(graph, &wgpu(), &uncached)
                .unwrap_or_else(|error| panic!("{name}: UncachedResident: {error}"));
            let replay_plans: Vec<String> = under_replay
                .planned()
                .map(|(_, plan)| format!("{plan:?}"))
                .collect();
            let uncached_plans: Vec<String> = under_uncached
                .planned()
                .map(|(_, plan)| format!("{plan:?}"))
                .collect();
            assert_eq!(
                replay_plans, uncached_plans,
                "{name}: CaptureReplay and UncachedResident produced different plans"
            );
        }
    }

    /// W16: a gather the claim does not take - along axis 1 - is still refused by the gate.
    #[test]
    fn compile_refuses_a_packed_gather_along_another_axis() {
        let b = Builder::for_tests();
        let ids = b
            .slot(Slot::Token, TensorType::new(vec![2], DType::I32))
            .unwrap();
        let sources: Vec<poot_graph_ir::Traced> =
            poot_graph_ir::packed_source_constants("table", embedding_descriptor())
                .into_iter()
                .map(|(name, tensor_type)| b.constant(name.as_str(), tensor_type).unwrap())
                .collect();
        let table = b.packed_dequant(&sources, embedding_descriptor()).unwrap();
        let columns = b.gather(table, 1, ids).unwrap();
        match compile(&b.finish(columns).unwrap(), &wgpu(), &REPLAY) {
            Err(CompileError::PackedDequant(_)) => {}
            other => panic!(
                "expected the escape gate's refusal, got {:?}",
                other.map(|_| ())
            ),
        }
    }

    /// SC-001 (card 1067), model-free, through `compile`: a stacked expert table over the target's buffer limit
    /// splits into expert ranges whose carriers each fit, and the recombined output equals the unsplit
    /// contraction on the CPU oracle (zero signs aside) for rows routed to every expert, the last (alone in its range)
    /// included, in an order that crosses the range boundaries. The limit is 544 bytes: two experts per piece,
    /// ranges 0..2, 2..4, 4..5.
    /// Mutation: in `split_oversized_expert_stacks` shift the ids of every piece after the first by one less than
    /// `range.start`. Red: `Index(IndexFault { eqn: 27, position: 3, value: I32(2), len: 2, kind: OutOfRange })`
    /// (the row routed to expert 3 reads local expert 2 of a two-expert piece).
    #[test]
    fn compile_splits_an_oversized_expert_stack_and_matches_the_unsplit_contraction() {
        let fixture = SplitStack::new();
        let routed = [0, 1, 2, 3, 4, 4, 0, 3, 2, 1, 4];
        let unsplit = fixture
            .run(&fixture.graph, &routed, &fixture.unsplit_stacks())
            .unwrap();
        let program = fixture.compile();
        assert_eq!(
            program.graph().consts().len(),
            3,
            "5 experts at 2 per buffer: ranges 0..2, 2..4, 4..5, one Q8_0 carrier each"
        );
        // The oracle runs the legalized graph: `fuse` turns the pieces' sum into one contraction epilogue, which
        // the oracle decomposes into unnamed values.
        let (split_graph, split_stacks) = fixture.oracle_split(&fixture.legalize());
        let split = fixture.run(&split_graph, &routed, &split_stacks).unwrap();
        let (split, unsplit) = (split.as_f32().unwrap(), unsplit.as_f32().unwrap());
        assert_eq!(split.len(), unsplit.len());
        for (index, (got, want)) in split.iter().zip(unsplit).enumerate() {
            assert_eq!(
                got,
                want,
                "element {index} (row {}, expert {}): split {got} vs unsplit {want}",
                index / 4,
                routed[index / 4]
            );
        }
        assert!(
            unsplit.iter().any(|v| *v != 0.0),
            "the fixture's outputs are not all zero"
        );
    }

    /// POOT-1119, through `compile`: the split of a routed stack keeps the expert-id proof. The ids of the
    /// unsplit graph are declared in `[0, E)`; each piece's ids (`max(id - lo, 0) % (hi - lo)`) are proven in
    /// the piece's own `[0, hi - lo)` by the range rules, so `compile` admits the split graph, and a split
    /// that lost the proof (ids left as shifted arithmetic) would be refused as `ExpertIds`.
    /// Mutation: in `split_oversized_expert_stacks` replace the `RemU` by the piece length with an `Add` of
    /// `0` (the ids stay unbounded arithmetic); the split graph is refused with `ExpertIds ... OutOfRange`
    /// and this goes red.
    #[test]
    fn a_split_expert_stack_keeps_its_ids_proven() {
        let fixture = SplitStack::new();
        let split = fixture.legalize();
        let pieces = split
            .eqns()
            .iter()
            .filter(|eqn| {
                matches!(
                    eqn.op,
                    OpKind::PackedContraction {
                        rows: ContractionRows::Indexed { .. },
                        ..
                    }
                )
            })
            .count();
        assert!(pieces > 1, "the stack split into {pieces} contractions");
        compile_verdict(&split).unwrap_or_else(|error| panic!("the split graph: {error}"));
    }

    /// The routed projection of a Q8_0 `[out, k]` stack of `experts` experts for one decode row.
    fn decode_stack_graph(experts: usize, out: usize, k: usize) -> Graph {
        use poot_graph_ir::ops::PackedExpertStack;
        let descriptor =
            poot_quant::PackedWeight::try_new(poot_quant::format::WeightFormat::Q8_0, [out, k])
                .unwrap();
        let table = PackedExpertStack {
            stack_id: "experts".to_string(),
            descriptor,
            experts,
        };
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(vec![1, k]))
            .unwrap();
        let ids = b
            .slot_named_ranged(
                Slot::Activation,
                "ids",
                TensorType::new(vec![1], DType::I32),
                IndexRange::below(experts).unwrap(),
            )
            .unwrap();
        let y = poot_graph_ir::ops::packed_indexed_linear(&b, x, ids, &table).unwrap();
        b.finish(y).unwrap()
    }

    /// SC-003 (card 1067), through `compile` at the real wgpu caps: 16 Q8_0 `[4096, 8192]` experts are 570 MB
    /// of one source, past the 2^32 bit offsets the kernels address (15 experts at most), yet under the 2 GiB
    /// buffer limit. The stack compiles to two pieces (15 + 1 experts) instead of the planner's
    /// `DispatchSizeOverflow`; one expert past the address range (`[32768, 16384]`, 570 MB alone) stays that
    /// typed refusal, since no split of a single expert helps.
    /// Mutation: drop `max_stack_experts` from `expert_ranges`' bound (start `per_chunk` at `usize::MAX`); the
    /// 570 MB stack is no longer split and compiles to the typed `DispatchSizeOverflow` refusal.
    #[test]
    fn compile_splits_a_stack_past_the_bit_offset_range_and_refuses_one_expert_past_it() {
        let program = compile(&decode_stack_graph(16, 4096, 8192), &wgpu(), &REPLAY)
            .expect("570 MB of experts splits");
        assert_eq!(program.graph().consts().len(), 2, "15 + 1 experts");
        let heights: Vec<usize> = program
            .graph()
            .consts()
            .iter()
            .map(|&id| program.graph().aval(id).shape[0])
            .collect();
        assert_eq!(heights, [15, 1]);

        let refused = refusal(
            compile(&decode_stack_graph(2, 32768, 16384), &wgpu(), &REPLAY),
            "one expert past the address range",
        );
        assert_eq!(refused.missing, Capability::DispatchSizeOverflow);
    }

    /// A Q8_0 `[4, 64]` stack of `SPLIT_EXPERTS` random experts routed from 11 rows: the unsplit routed
    /// projection, its split compile on a target whose buffer limit holds two experts, and the oracle inputs of both.
    struct SplitStack {
        graph: Graph,
        x: ValueId,
        ids: ValueId,
        descriptor: poot_quant::PackedWeight,
        owners: Vec<std::sync::Arc<poot_quant::PackedPayload>>,
    }

    const SPLIT_EXPERTS: usize = 5;

    impl SplitStack {
        /// The routed projection with its ids declared in `[0, SPLIT_EXPERTS)`, the graph `compile` admits.
        fn new() -> Self {
            Self::with_ids(true)
        }

        /// [`SplitStack::new`], with the ids declared or, when `declared` is false, unbounded: the oracle
        /// then meets an invalid id as the index fault the split must preserve, not as a slot-range
        /// fault. Such a graph is for the legalize and oracle rows, which never compile it.
        fn with_ids(declared: bool) -> Self {
            use poot_graph_ir::ops::PackedExpertStack;
            let format = poot_quant::format::WeightFormat::Q8_0;
            let descriptor = poot_quant::PackedWeight::try_new(format, [4, 64]).unwrap();
            let table = PackedExpertStack {
                stack_id: "experts".to_string(),
                descriptor,
                experts: SPLIT_EXPERTS,
            };
            let b = Builder::for_tests();
            let x = b
                .slot_named(Slot::Activation, "x", TensorType::f32(vec![11, 64]))
                .unwrap();
            let ids_type = TensorType::new(vec![11], DType::I32);
            let ids = if declared {
                b.slot_named_ranged(
                    Slot::Activation,
                    "ids",
                    ids_type,
                    IndexRange::below(SPLIT_EXPERTS).unwrap(),
                )
                .unwrap()
            } else {
                b.slot_named(Slot::Activation, "ids", ids_type).unwrap()
            };
            let y = poot_graph_ir::ops::packed_indexed_linear(&b, x, ids, &table).unwrap();
            Self {
                graph: b.finish(y).unwrap(),
                x: x.id(),
                ids: ids.id(),
                descriptor,
                owners: (0..SPLIT_EXPERTS)
                    .map(|e| {
                        std::sync::Arc::new(poot_quant::test_support::packed::random_payload(
                            format,
                            [4, 64],
                            e as u64 * 7 + 3,
                        ))
                    })
                    .collect(),
            }
        }

        /// The buffer limit that holds two experts of the one Q8_0 source.
        fn two_experts(&self) -> Target {
            let slab = self.descriptor.slab_bytes(poot_quant::SourceRole::Blocks) as u64;
            wgpu_with_limit(2 * slab)
        }

        fn compile(&self) -> Program {
            compile(&self.graph, &self.two_experts(), &REPLAY).expect("splits the stack")
        }

        fn legalize(&self) -> Graph {
            crate::passes::legalize(
                &self.graph,
                &gpu_caps(&self.two_experts()),
                &CompileLimits::STANDARD,
            )
            .expect("splits the stack")
        }

        fn unsplit_stacks(&self) -> impl Fn(&str) -> poot_quant::PackedStackComponentRef {
            let whole =
                std::sync::Arc::new(poot_quant::PackedStack::try_new(self.owners.clone()).unwrap());
            move |name| {
                let role = poot_graph_ir::PackedSourceName::parse(name).unwrap().role();
                poot_quant::PackedStackComponentRef::new(std::sync::Arc::clone(&whole), role)
            }
        }

        /// `split` as the oracle binds it, and the stack of each carrier. The oracle reads a carrier's role and
        /// stack id off a [`PackedSourceName`], which has no spelling for `{carrier}.chunkN`, so each chunk is
        /// renamed to the packed name of stack `{id}.chunkN`; its stack is the experts the chunks before it
        /// leave off, as the executor binder places them.
        fn oracle_split(
            &self,
            split: &Graph,
        ) -> (
            Graph,
            impl Fn(&str) -> poot_quant::PackedStackComponentRef + use<>,
        ) {
            use poot_graph_ir::PackedSourceName;
            let mut renames: Vec<(ValueId, String)> = Vec::new();
            let mut start = 0;
            let mut runs: HashMap<String, std::ops::Range<usize>> = HashMap::new();
            for &id in split.consts() {
                let meta = split.meta(id);
                let name = meta.name.clone().expect("a named carrier");
                let (carrier, index) = name
                    .rsplit_once(".chunk")
                    .expect("only chunk carriers remain");
                if index == "0" {
                    start = 0;
                }
                let source = PackedSourceName::parse(carrier).expect("a packed carrier");
                let renamed = PackedSourceName::new(
                    &format!("{}.chunk{index}", source.linear_id()),
                    source.role(),
                )
                .to_string();
                let end = start + meta.aval.shape[0];
                runs.insert(renamed.clone(), start..end);
                renames.push((id, renamed));
                start = end;
            }
            let graph = poot_graph_ir::test_support::graph_fixtures::derive(split, |draft| {
                for (id, name) in &renames {
                    draft.values_mut()[*id].name = Some(name.clone());
                }
            })
            .unwrap();
            let owners = self.owners.clone();
            (graph, move |name: &str| {
                let role = PackedSourceName::parse(name).unwrap().role();
                let part = std::sync::Arc::new(
                    poot_quant::PackedStack::try_new(owners[runs[name].clone()].to_vec()).unwrap(),
                );
                poot_quant::PackedStackComponentRef::new(part, role)
            })
        }

        fn run(
            &self,
            graph: &Graph,
            routed: &[i32],
            stacks: &dyn Fn(&str) -> poot_quant::PackedStackComponentRef,
        ) -> Result<poot_tensor::HostTensor, poot_eval::EvalError> {
            use poot_eval::{EvalBudget, EvalOptions, Value, eval};
            let activation: Vec<f32> = (0..11 * 64)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
                .collect();
            let mut inputs: HashMap<ValueId, Value> = HashMap::new();
            for &id in graph.inputs() {
                let value = if id == self.x {
                    Value::from(poot_tensor::HostTensor::f32(
                        vec![11, 64],
                        activation.clone(),
                    ))
                } else if id == self.ids {
                    Value::from(poot_tensor::HostTensor::i32(vec![11], routed.to_vec()))
                } else {
                    Value::from(stacks(
                        graph.meta(id).name.as_deref().expect("a named carrier"),
                    ))
                };
                inputs.insert(id, value);
            }
            Ok(
                eval(graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))?
                    .output
                    .into_host()
                    .unwrap(),
            )
        }
    }
}
