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

use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};

#[cfg(test)]
use poot_graph_ir::StateRole;
#[cfg(test)]
use poot_graph_ir::analysis::Tier2Class;
use poot_graph_ir::analysis::{
    NumericalRewriteDecline, NumericsError, NumericsProperty, PASS_DECLARATIONS, PassDeclaration,
    implied_numerics, verify_pass_numerics,
};
use poot_graph_ir::{Eqn, Graph, NoValidations, OpKind, Operand, Slot, ValidationChannel, ValueId};
use poot_target::{Backend, DeviceCaps};

use crate::passes::{self as transform, GraphBudget, LegalizeError, PackedDequantProductionError};
use crate::*;

/// What a graph is compiled for: the backend and the device's measured capabilities (card 522). Passes and
/// the planner read it; tracers never see it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    pub backend: Backend,
    pub caps: DeviceCaps,
}

/// Typed execution choices a caller makes for one compile. Never an environment variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileOptions {
    /// Executor submission mode. Compilation produces the same plans for either mode.
    pub execution: Submission,
    /// Whether `compile`'s fusion-adjacent passes run (card 535a). See [`FusionPolicy`].
    pub fusion: FusionPolicy,
    /// What compilation may construct and load. There is no default: every caller states its policy.
    pub limits: CompileLimits,
}

/// What one compile may construct, and the largest artifact it may load: every field is a finite, nonzero
/// cap, so no compile is unbounded and none is configured by environment.
///
/// - `max_intermediate_values` and `max_intermediate_eqns` cap the value table and the equation list of
///   every graph a pass produces, the traced input and the final program included. A pass that would
///   append past either returns [`CompileError::Expansion`] before the append.
/// - `max_body_instructions` and `max_body_locals` cap one generated kernel body (statements plus block
///   terminators, and locals with parameters). An over-limit request is a typed planner refusal carrying
///   `poot_kernelgen::KernelGenError::BodyLimit`.
/// - `max_artifact_bytes` caps one compiled kernel artifact an executor loads or caches; see
///   [`Program::limits`]. It bounds the artifact's bytes, not the external compiler's memory or time.
///
/// Wall time is telemetry, not a limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileLimits {
    pub max_intermediate_values: NonZeroUsize,
    pub max_intermediate_eqns: NonZeroUsize,
    pub max_body_instructions: NonZeroUsize,
    pub max_body_locals: NonZeroUsize,
    pub max_artifact_bytes: NonZeroU64,
}

impl CompileLimits {
    /// The limits a production compile runs under: a few orders above the largest graph and kernel a real
    /// model produces, and far below what exhausts a host.
    pub const STANDARD: Self = Self {
        max_intermediate_values: NonZeroUsize::new(1 << 24).unwrap(),
        max_intermediate_eqns: NonZeroUsize::new(1 << 24).unwrap(),
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
    #[error("values")]
    Values,
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
    pub stage: &'static str,
    pub resource: GraphResource,
    pub held: usize,
    pub attempted: usize,
    pub limit: usize,
}

/// Card 535a: a declared, typed policy `compile` consults while running its own fixed pass order (never a
/// second pipeline or a bypass around `compile`). [`FusionPolicy::MoeHangGuard`] skips exactly
/// `flash_attention_capped` and `fuse`, and is a planner input: the contraction choice keeps every
/// `MatMul` off the generated tiled GEMM (Card 557). Every other pass (`legalize`, the bias epilogue, the
/// BF16 folds, `lower_nonlast_reduces`, `widen_mismatched_matmul_dtypes`) still runs in `compile`'s one order.
/// `Program::passes()` records which passes ran and `Program::kernel_choice` which kernel each equation
/// took, so a caller or test can see the guard fire.
///
/// A MoE/Mixtral/DeepSeek-V3 routed-expert prefill graph has hung ROCm/AMDGCN under the guarded choices
/// (cards 186/192): the generated tiled GEMM took the MoE router's bare `MatMul`, and flash-attention
/// synthesis (`flash_attention_capped`, which feeds `fuse`'s pointwise fusion) was never pod-verified safe
/// on PTX for this shape either. A guarded graph's quantized weights are packed carriers
/// (`PackedDequant`/`PackedContraction`), which plan through their own imported-kernel path outside
/// `plan_eqn` entirely, which is what makes these graphs plannable at all under the guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionPolicy {
    /// Every pass in `compile`'s pipeline runs, and every kernel choice is open.
    Full,
    /// Cards 186/192's MoE-router-hang guard: `flash_attention_capped` and `fuse` are skipped, and no
    /// `MatMul` takes the generated tiled GEMM.
    MoeHangGuard,
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
    /// A `PackedDequant` the contraction claim could not take escapes into ordinary execution (Card
    /// 355's gate, run by `compile` since card 642): no target materializes a packed weight outside a
    /// claimed contraction.
    #[error(transparent)]
    PackedDequant(Box<PackedDequantProductionError>),
    /// A pass would have grown the graph past [`CompileLimits`]: nothing past the limit was appended and no
    /// program exists.
    #[error(transparent)]
    Expansion(Box<ExpansionError>),
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

impl From<ExpansionError> for CompileError {
    fn from(error: ExpansionError) -> Self {
        CompileError::Expansion(Box::new(error))
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
    pub value: ValueId,
    pub slot: Slot,
    pub storage: ValueStorage,
}

/// A graph compiled for one [`Target`]: the optimized graph and every decision an executor needs to load
/// and replay it. Built only by [`compile`].
#[derive(Debug)]
pub struct Program<V: ValidationChannel = NoValidations> {
    graph: Graph<V>,
    /// One plan per equation of `graph`, index-aligned with `graph.eqns`.
    plans: Vec<Plan>,
    /// The kernel the planner chose for each equation, index-aligned with `plans`. The executor never
    /// reads it.
    choices: Vec<KernelChoice>,
    storage: GraphStorage,
    slots: Vec<SlotSchema>,
    /// The numerics of the graph as traced: the unfused baseline every device run carries. Read back
    /// only by the test-only `reassociation_class`.
    #[cfg_attr(not(test), allow(dead_code))]
    baseline: NumericsProperty,
    /// The passes that changed the graph, in the order they ran.
    passes: Vec<PassDeclaration>,
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
    /// 546a): a zero-lane packet for a graph with no validation outputs (`NoValidations`), checked
    /// by the contract before any readback the same way either way.
    validation: ValidationPacketPlan,
    /// Lifetimes and arena-slot assignment for this program's locally-computed values (Card 547b),
    /// derived from `plans`, `storage`, `state_commit` and `validation` once here.
    buffer_plan: BufferPlan,
    /// The limits this program was compiled under; an executor loads its kernels under the same
    /// `max_artifact_bytes`.
    limits: CompileLimits,
}

impl<V: ValidationChannel> Program<V> {
    /// The compiled graph.
    pub fn graph(&self) -> &Graph<V> {
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

    /// Each equation of [`Program::graph`] with its plan, in execution order.
    pub fn planned(&self) -> impl Iterator<Item = (&Eqn, &Plan)> {
        self.graph.eqns.iter().zip(&self.plans)
    }

    /// The kernel the planner chose for `eqn`, an equation of [`Program::graph`]: a generated request, an
    /// imported template, or no dispatch at all. Panics if `eqn` is not an equation of this program's
    /// graph.
    pub fn kernel_choice(&self, eqn: &Eqn) -> &KernelChoice {
        let index = self
            .graph
            .eqns
            .iter()
            .position(|candidate| candidate.out == eqn.out)
            .expect("kernel_choice: the equation belongs to this program's graph");
        &self.choices[index]
    }

    /// Every value's storage, as the plan chose it.
    pub fn storage(&self) -> &GraphStorage {
        &self.storage
    }

    /// The per-step slots and their storage.
    pub fn slots(&self) -> &[SlotSchema] {
        &self.slots
    }

    /// The tier-2 comparison class this program publishes (ADR-0101 decision 2): the strongest of the
    /// traced graph's baseline and the declarations of the passes that changed it. [`compile`] has already
    /// checked that the compiled graph implies nothing looser.
    ///
    /// Card 546b: narrowed from `pub` (gated `any(test, feature = "test-support")`) to `pub(crate)` -
    /// its one cross-crate reader, `poot-gpu`'s own `src/tests/decode_cache_warmth.rs`, moved off
    /// `GpuExecutor::run_resident_kv_cached` and was deleted with it, leaving the `pub` accessor with
    /// no caller outside this crate's own tests (dead-pub has no exemption for an `any(test, ...)`
    /// gate); every remaining caller is this crate's own tests.
    #[cfg(test)]
    pub(crate) fn reassociation_class(&self) -> Tier2Class {
        recorded_numerics(self.baseline, &self.passes).tier2_class()
    }

    /// Numerical rewrites withheld during canonicalization, referring to the traced value ids.
    /// These are optimization diagnostics: compilation keeps the original arithmetic instead.
    pub fn numerical_declines(&self) -> &[NumericalRewriteDecline] {
        &self.numerical_declines
    }

    /// The passes that changed the graph, in the order they ran (card 535a: a caller or test reads this
    /// to see whether a declared [`CompileOptions`] choice, such as [`FusionPolicy::MoeHangGuard`],
    /// actually took effect - never by re-deriving it locally).
    pub fn passes(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.passes.iter().map(|declaration| declaration.pass)
    }
}

/// Compile `g` for `target`: canonicalize, CSE and iota folding, rope and flash-attention recognition, DCE,
/// the packed-contraction and row-gather claims and their escape gate (cards 642/545a: every
/// `PackedDequant` is claimed into a `PackedContraction`/`PackedRowGather`, is a W12 MoE branch, or
/// the graph is refused), buffer-limit legalization, then the BF16-packed-weight recognition, non-last-axis reduce legalization
/// and backend dtype retyping (Card 534a), then fusion, then `crate::plan_graph`, which makes every
/// kernel choice (Card 557). The bias epilogue fuse rule runs right after the escape gate.
///
/// Legalization (card 523a: `legalize`, reading `target.caps` to host or split a constant the target
/// cannot allocate as one device buffer) and Card 534a's step (`poot_graph_plan::prepare_target_graph`'s
/// guts, inlined here so every sub-pass is individually tracked and numerics-checked) join the pipeline
/// between DCE and fusion, each its own `PassDeclaration` row like every other pass here
/// (`PASS_DECLARATIONS`), checked by `revert_unplannable_rewrites`/`verify_pass_numerics` the same way.
/// `legalize` runs first: it can retag a `Gather` into a `Slot::TokenEmbed` input, and Card 534a's
/// `fold_dense_bf16_row_gathers` pattern-matches a `Gather` specifically, so a table `legalize` already
/// hosted must not still look like one to it. Card 534a's step is the one preparation every target needs
/// before planning: no executor sequences its own copy of it any more.
///
/// `options.fusion` (card 535a) can skip `flash_attention_capped` and `fuse` and keep the planner off
/// the generated tiled GEMM; see [`FusionPolicy`]. Every other pass, including `legalize`, still runs in
/// this same fixed order regardless.
pub fn compile<V: ValidationChannel>(
    g: &Graph<V>,
    target: &Target,
    options: &CompileOptions,
) -> Result<Program<V>, CompileError> {
    g.validate().map_err(PlanError::from)?;
    let budget = GraphBudget::of(&options.limits);
    budget.admit("compile input", g)?;
    let _scope = CompileScope::enter();
    // A flash prefill launches `Hq * L` workgroups on one grid axis; cap the rewrite at the target's grid so
    // the planner is never handed a launch the device cannot issue.
    let max_prefill_groups = target.caps.max_grid[0] as usize;
    let guarded = options.fusion == FusionPolicy::MoeHangGuard;
    let mut pipeline = PassPipeline::new(g, target, options.fusion, options.limits);
    let (canonical, numerical_declines) = transform::canonicalize_with_declines(g);
    pipeline.after_apply("canonicalize", canonical)?;
    pipeline.run("cse", transform::cse)?;
    pipeline.run("fold_iota", transform::fold_iota)?;
    pipeline.run("rope_fusion", transform::rope_fusion)?;
    if !guarded {
        pipeline.run("flash_attention_capped", |g| {
            transform::flash_attention_capped(g, Some(max_prefill_groups))
        })?;
    }
    pipeline.run("dce", transform::dce)?;
    // The two claiming passes are the only rewrites whose refusals the escape gate can mask: their
    // reverts put the packed decomposition back, and the gate is what then refuses that
    // decomposition. Their restored maps travel as one local from these runs straight to their
    // single consumer, so the refusal data exists exactly from the producing reverts to the gate
    // check and cannot be read stale (card 669, option b).
    let mut restored = pipeline.run(
        "recognize_packed_contractions",
        transform::recognize_packed_contractions,
    )?;
    restored.extend(pipeline.run(
        "recognize_packed_row_gathers",
        transform::recognize_packed_row_gathers,
    )?);
    pipeline.reject_escapes(restored)?;
    pipeline.run("fuse_bias_epilogues", transform::fuse_bias_epilogues)?;
    pipeline.try_run("legalize", |g| {
        transform::legalize(g, &target.caps, &options.limits)
    })?;
    pipeline.run("fold_dense_contractions", fold_dense_contractions)?;
    pipeline.run("fold_dense_bf16_row_gathers", fold_dense_bf16_row_gathers)?;
    pipeline.reserve(
        "lower_nonlast_reduces",
        lower_nonlast_growth(pipeline.graph()),
    )?;
    pipeline.run("lower_nonlast_reduces", lower_nonlast_reduces)?;
    pipeline.reserve(
        "widen_mismatched_matmul_dtypes",
        widen_growth(pipeline.graph()),
    )?;
    pipeline.run("widen_mismatched_matmul_dtypes", |g| {
        widen_mismatched_matmul_dtypes(g, target.backend, &target.caps)
    })?;
    // Card 673: admit every validation witness here, on the unfused graph. `fuse` folds a witness
    // producer chain into one `OpKind::Fused` region the structural classifier does not see through,
    // so the walk must run before it. Every backend reaches this through `compile`; a noncanonical
    // witness is refused with the planner's typed `PlanError::ValidationWitnessNotCanonical`, naming
    // the witness, before any executor allocates.
    device_validation::admit_device_witnesses(pipeline.graph())?;
    // Card 551a (SC-009): a graph decomposition, not a new Plan variant or executor mechanism
    // (ADR 0100/0112) - rewrites `SampleToken{Greedy}` over a large vocab into a chunked two-stage
    // reduction (parallel per-chunk workgroups instead of one workgroup's long serial scan). Runs
    // before `fuse` so the elementwise chains it introduces (the padding, the masking, the chunk
    // combine) fold into as few dispatches as `fuse`'s own patterns allow, the same reason
    // `rope_fusion` precedes it.
    pipeline.try_run("decompose_large_vocab_greedy", |g| {
        transform::decompose_large_vocab_greedy(g, &options.limits)
    })?;
    if !guarded {
        pipeline.run("fuse", transform::fuse)?;
    }
    let (graph, baseline, passes) = pipeline.finish()?;
    let planned = plan_graph(&graph, target, options.fusion, &options.limits.body())?;
    let state_commit = state_commit_from_plans(&graph, &planned.plans)?;
    let validation = device_validation::validation_packet_plan(&graph)?;
    let buffer_plan = plan_buffers(
        &graph,
        &planned.plans,
        &planned.storage,
        &state_commit,
        &validation,
    );
    Ok(Program {
        graph,
        plans: planned.plans,
        choices: planned.choices,
        storage: planned.storage,
        slots: planned.slots,
        baseline,
        passes,
        numerical_declines,
        target: *target,
        submission: options.execution,
        state_commit,
        validation,
        buffer_plan,
        limits: options.limits,
    })
}

/// The graph as it moves through the passes, with the record of the passes that changed it.
struct PassPipeline<'t, V: ValidationChannel> {
    graph: Graph<V>,
    target: &'t Target,
    /// The compile's fusion policy, which the rewrite-legality probes plan with exactly as
    /// `plan_graph` will.
    fusion: FusionPolicy,
    /// The numerics of the graph as traced: the unfused baseline every device run carries.
    baseline: NumericsProperty,
    passes: Vec<PassDeclaration>,
    limits: CompileLimits,
    budget: GraphBudget,
}

impl<'t, V: ValidationChannel> PassPipeline<'t, V> {
    fn new(g: &Graph<V>, target: &'t Target, fusion: FusionPolicy, limits: CompileLimits) -> Self {
        Self {
            graph: g.clone(),
            target,
            fusion,
            baseline: implied_numerics(g),
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
    fn graph(&self) -> &Graph<V> {
        &self.graph
    }

    /// Run one pass: revert every rewrite the target cannot plan, check the pass's declaration against what
    /// it produced, and record the declaration when the pass changed the graph. Returns the revert's
    /// restored map (empty when nothing was reverted) so a caller can hand it to the one later check
    /// that consumes it; a caller with no such check drops it with the statement.
    fn run(
        &mut self,
        pass: &'static str,
        apply: impl FnOnce(&Graph<V>) -> Graph<V>,
    ) -> Result<HashMap<ValueId, Refusal>, CompileError> {
        self.after_apply(pass, apply(&self.graph))
    }

    /// Like [`Self::run`], for a pass that can refuse the graph outright (`legalize`'s
    /// [`transform::LegalizeError`]) instead of always producing one: the apply step's error propagates
    /// through `?` before the shared revert/verify/record sequence runs.
    fn try_run<E>(
        &mut self,
        pass: &'static str,
        apply: impl FnOnce(&Graph<V>) -> Result<Graph<V>, E>,
    ) -> Result<HashMap<ValueId, Refusal>, CompileError>
    where
        CompileError: From<E>,
    {
        let out = apply(&self.graph)?;
        self.after_apply(pass, out)
    }

    /// Shared tail of [`Self::run`]/[`Self::try_run`]: revert every rewrite the target cannot plan,
    /// check the pass's declaration against what it produced, record the declaration when the pass
    /// changed the graph, and return the revert's restored map - the typed refusal behind each
    /// equation the revert put back (card 669).
    fn after_apply(
        &mut self,
        pass: &'static str,
        out: Graph<V>,
    ) -> Result<HashMap<ValueId, Refusal>, CompileError> {
        let declaration = declaration(pass);
        self.budget.admit(pass, &out)?;
        let (out, restored) = revert_unplannable_rewrites(
            &self.graph,
            out,
            self.target,
            self.fusion,
            &self.limits.body(),
        )?;
        verify_pass_numerics(&declaration, &out)?;
        if changed(&self.graph, &out) {
            self.passes.push(declaration);
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
                .after_apply("reject_packed_dequant_escapes", self.graph.clone())
                .map(|_gate_restored| ()),
            Err(error) => Err(
                match escaped_value(&error).and_then(|value| restored.get(&value)) {
                    Some(refusal) => PlanError::Refused(Box::new(refusal.clone())).into(),
                    None => error.into(),
                },
            ),
        }
    }

    /// The compiled graph, its baseline and the passes that changed it. Fails closed when the graph implies
    /// a looser class than the recorded passes and the baseline allow.
    fn finish(self) -> Result<(Graph<V>, NumericsProperty, Vec<PassDeclaration>), NumericsError> {
        let recorded = recorded_numerics(self.baseline, &self.passes);
        let implied = implied_numerics(&self.graph);
        if implied > recorded {
            return Err(NumericsError::UnderDeclared {
                pass: "compile",
                declared: recorded,
                implied,
            });
        }
        Ok((self.graph, self.baseline, self.passes))
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
fn lower_nonlast_growth<V: ValidationChannel>(g: &Graph<V>) -> Growth {
    let added = g
        .eqns
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
/// at most two (one per operand) for a `MatMul`, `MatMulBias` or `Binary`.
fn widen_growth<V: ValidationChannel>(g: &Graph<V>) -> Growth {
    let casts = g.eqns.iter().fold(0usize, |casts, eqn| {
        casts.saturating_add(match eqn.op {
            OpKind::Gather { axis: 0 } | OpKind::Slice { .. } => 1,
            OpKind::MatMul | OpKind::MatMulBias | OpKind::Binary(_) => 2,
            _ => 0,
        })
    });
    Growth {
        values: casts,
        eqns: casts,
    }
}

/// The strongest of a baseline and the declarations of the passes that ran.
fn recorded_numerics(baseline: NumericsProperty, passes: &[PassDeclaration]) -> NumericsProperty {
    passes
        .iter()
        .map(|declaration| declaration.property)
        .fold(baseline, NumericsProperty::strongest)
}

/// The numerics declaration of a pass in this pipeline. Every pass `compile` runs has one; a pass without one
/// is a bug in the pipeline, not a property of the input.
fn declaration(pass: &'static str) -> PassDeclaration {
    *PASS_DECLARATIONS
        .iter()
        .find(|declaration| declaration.pass == pass)
        .unwrap_or_else(|| panic!("compile runs pass {pass:?}, which has no numerics declaration"))
}

/// Whether a pass changed the graph: an equation added, removed, retyped or rewired, or a value added.
fn changed<V: ValidationChannel>(before: &Graph<V>, after: &Graph<V>) -> bool {
    before.values.len() != after.values.len()
        || before.eqns.len() != after.eqns.len()
        || before.eqns.iter().zip(&after.eqns).any(|(a, b)| {
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
/// reports carries the error context of the escaping dequant (`InvalidGraph` cannot name one). Matched
/// exhaustively on purpose: a new variant forces this site to decide whether it carries an escape.
fn escaped_value(error: &PackedDequantProductionError) -> Option<ValueId> {
    use PackedDequantProductionError as Error;
    match error {
        Error::InvalidGraph(_) => None,
        Error::PreexistingContraction { context }
        | Error::CarrierStorage { context, .. }
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
        | Error::Unmatched { context } => Some(context.value),
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
/// pass that defines an output `before` never had cannot be taken apart this way, so a refusal there reverts
/// the whole pass.
///
/// Alongside the restored graph, returns the typed refusal behind it, keyed by every equation the revert
/// restored (card 669): each refused root's closure maps to its own [`Refusal`], the first root in probe
/// order owning a value two refused roots restored together. A whole-pass revert restores `before`
/// wholesale, attributing no individual equation to any one refusal, so it records nothing - an escape
/// there is the restored graph's own, not this revert's. [`PassPipeline::reject_escapes`] reports the
/// recorded refusal when the escape gate finds one of these restored values escaping; `compile` threads
/// the map there as a local and no other site reads it.
fn revert_unplannable_rewrites<V: ValidationChannel>(
    before: &Graph<V>,
    after: Graph<V>,
    target: &Target,
    fusion: FusionPolicy,
    limits: &poot_kernelgen::BodyLimits,
) -> Result<(Graph<V>, HashMap<ValueId, Refusal>), CompileError> {
    let before_at: HashMap<ValueId, usize> = before
        .eqns
        .iter()
        .enumerate()
        .map(|(i, eqn)| (eqn.out, i))
        .collect();
    let rewritten = after.eqns.iter().filter(|eqn| {
        before_at
            .get(&eqn.out)
            .is_none_or(|&i| before.eqns[i].op != eqn.op)
    });
    // Probe exactly as `plan_graph` plans: with the graph's own strided views and one exact-I32 analysis.
    // A probe stricter than the final plan would drop a rewrite the target can run.
    let analysis = ExactI32StorageAnalysis::new(&after);
    let views = compute_views(&after, target.backend);
    let mut refused: Vec<(ValueId, Refusal)> = Vec::new();
    for eqn in rewritten {
        match plan_eqn_compiled(&analysis, &after, eqn, target, &views, fusion, limits) {
            Ok(_) => {}
            Err(PlanError::Refused(refusal)) => refused.push((eqn.out, *refusal)),
            Err(error) => return Err(error.into()),
        }
    }
    if refused.is_empty() {
        return Ok((after, HashMap::new()));
    }
    if after
        .eqns
        .iter()
        .any(|eqn| !before_at.contains_key(&eqn.out))
    {
        return Ok((before.clone(), HashMap::new()));
    }

    let after_at: HashMap<ValueId, usize> = after
        .eqns
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
            for operand in &before.eqns[before_at[&out]].inputs {
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
        .eqns
        .iter()
        .filter_map(|eqn| {
            if restored.contains_key(&eqn.out) {
                Some(eqn.clone())
            } else {
                after_at.get(&eqn.out).map(|&i| after.eqns[i].clone())
            }
        })
        .collect();
    Ok((Graph { eqns, ..after }, restored))
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{BinOp, Builder, OpKind, RedOp, Scalar, Slot, Storage, TensorType, UnOp};
    use poot_tensor::DType;
    use poot_test_util::device_caps::default_caps_for;

    use super::*;

    fn wgpu() -> Target {
        Target {
            backend: Backend::SpirvVulkan,
            caps: default_caps_for(Backend::SpirvVulkan),
        }
    }

    fn target_for(backend: Backend) -> Target {
        Target {
            backend,
            caps: default_caps_for(backend),
        }
    }

    const REPLAY: CompileOptions = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: crate::CompileLimits::STANDARD,
    };

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
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4, 8]));
        let s = b.constant("s", TensorType::f32(vec![4, 8]));
        let bias = b.constant("bias", TensorType::f32(vec![4, 8]));
        let scaled = b.binary(BinOp::Mul, x, s);
        let e = b.unary(UnOp::Exp, scaled);
        let out = b.binary(BinOp::Add, e, bias);
        b.finish(out)
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
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Fused(_))),
            "the chain is one fused region"
        );
        assert_eq!(program.reassociation_class(), Tier2Class::Reassociating);
    }

    /// Card 535a (cards 186/192's MoE-router-hang guard): `FusionPolicy::MoeHangGuard` skips `fuse`, so
    /// the same pointwise chain the test above shows fusing under `FusionPolicy::Full` records no passes
    /// (nothing else in the pipeline changes this graph) under the guard, and stays scalar-decomposed
    /// (never one `Fused` region).
    ///
    /// Mutation (recorded here, never left in the tree): deleting `compile`'s `if !guarded` checks around
    /// `flash_attention_capped`/`fuse` (i.e. always running them, ignoring `options.fusion`) failed this
    /// test with `passes = ["fuse"]` (expected `[]`); restoring the guard made it green again. The guard's
    /// planner half is [`moe_hang_guard_keeps_the_router_matmul_off_the_tiled_gemm`].
    #[test]
    fn compile_moe_hang_guard_skips_fuse() {
        let guarded = CompileOptions {
            fusion: FusionPolicy::MoeHangGuard,
            ..REPLAY
        };
        let program = compile(&pointwise_chain(), &wgpu(), &guarded)
            .expect("the chain still compiles under the guard");
        let passes: Vec<&str> = program.passes().collect();
        assert_eq!(
            passes,
            Vec::<&str>::new(),
            "fuse must not run under the guard"
        );
        assert!(
            !program
                .graph()
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Fused(_))),
            "the chain must stay scalar-decomposed under the guard"
        );
    }

    /// The kernel `program` chose for the routed-expert router `x @ w_gate` (weight `[hidden,
    /// n_experts]`) of every layer, and whether each is the generated tiled GEMM.
    fn router_choices_are_tiled(program: &Program, hidden: usize, n_experts: usize) -> Vec<bool> {
        let g = program.graph();
        g.eqns
            .iter()
            .filter(|eqn| {
                matches!(eqn.op, OpKind::MatMul)
                    && matches!(eqn.inputs.get(1), Some(Operand::Value(w))
                        if g.aval(*w).shape == [hidden, n_experts])
            })
            .map(|eqn| {
                matches!(
                    program.kernel_choice(eqn),
                    KernelChoice::Generated(KernelRequest::Contraction(
                        poot_kernelgen::ContractionSpec::TiledRegion { .. }
                    ))
                )
            })
            .collect()
    }

    /// Card 557 SC-005 (cards 186/192's MoE-router-hang guard): the guard is a planner input to the
    /// contraction choice. A routed-expert prefill (mixtral, two layers, four prompt tokens) compiled for
    /// AMDGCN plans every router `MatMul` on the generated tiled GEMM under `FusionPolicy::Full` and
    /// untiled under `FusionPolicy::MoeHangGuard`, GPU-free. Mutation: drop the guard input from the
    /// contraction choice (`planner/matmul.rs`'s `fusion == FusionPolicy::Full` guard); the guarded
    /// routers plan tiled and this row goes red.
    #[test]
    fn moe_hang_guard_keeps_the_router_matmul_off_the_tiled_gemm() {
        use poot_models::mixtral::{MixtralParams, trace_mixtral_prefill};
        use poot_models::qwen2::Qwen2Config;
        use poot_target::AmdArch;

        let cfg = Qwen2Config {
            vocab: 24,
            hidden: 16,
            inter: 16,
            layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 6,
            rotary_dim: 6,
            max_pos: 32,
            ..Qwen2Config::default()
        };
        let n_experts = 6;
        let params = MixtralParams {
            n_experts,
            top_k: 2,
            inter: 12,
        };
        let g = trace_mixtral_prefill(cfg, params, 4);
        let amd = target_for(Backend::AmdGcn(AmdArch::gfx1151()));

        let full = compile(&g, &amd, &REPLAY).expect("the routed prefill compiles under Full");
        let tiled = router_choices_are_tiled(&full, cfg.hidden, n_experts);
        assert_eq!(tiled.len(), cfg.layers, "one router matmul per layer");
        assert!(
            tiled.iter().all(|&tiled| tiled),
            "Full plans every router on the generated tiled GEMM: {tiled:?}"
        );

        let guarded = CompileOptions {
            fusion: FusionPolicy::MoeHangGuard,
            ..REPLAY
        };
        let program = compile(&g, &amd, &guarded).expect("the guard still compiles the prefill");
        let tiled = router_choices_are_tiled(&program, cfg.hidden, n_experts);
        assert_eq!(tiled.len(), cfg.layers, "one router matmul per layer");
        assert!(
            tiled.iter().all(|&tiled| !tiled),
            "the guard must keep every router off the generated tiled GEMM: {tiled:?}"
        );
    }

    /// SC-002: a graph no pass changes publishes its own baseline. A pure reshape moves no float value, so
    /// it is compared bit for bit.
    #[test]
    fn compile_of_an_untouched_movement_graph_publishes_bit_exact() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4, 8]));
        let out = b.reshape(x, vec![8, 4]);
        let program = compile(&b.finish(out), &wgpu(), &REPLAY).expect("a reshape compiles");
        assert_eq!(program.passes().count(), 0);
        assert_eq!(program.reassociation_class(), Tier2Class::BitExact);
    }

    /// SC-002: the target's grid is data. The E4M3FN decode dispatch plans on the measured wgpu grid and is
    /// refused, naming the equation, the target and the synthetic limit, on a device whose x-grid holds
    /// fewer workgroups than the dispatch needs.
    #[test]
    fn compile_refuses_a_dispatch_beyond_the_target_grid() {
        let b = Builder::new();
        let raw = b.constant("raw", TensorType::new(vec![1024, 1], DType::E4M3FN));
        let out = b.cast(raw, DType::F32);
        let g = b.finish(out);
        compile(&g, &wgpu(), &REPLAY).expect("the measured wgpu grid holds the dispatch");

        let mut narrow = wgpu();
        narrow.caps.max_grid = [2, 65_535, 65_535];
        let refusal = refusal(compile(&g, &narrow, &REPLAY), "a two-workgroup x-grid");
        assert_eq!(refusal.eqn, out.id);
        assert_eq!(refusal.target, Backend::SpirvVulkan);
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
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 8, 16, 16]));
        let out = poot_graph_ir::ops::softmax(&b, x);
        let g = b.finish(out);
        for backend in [
            Backend::SpirvVulkan,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
            Backend::Nvptx,
        ] {
            let mut target = target_for(backend);
            target.caps.max_grid = [64, 65_535, 65_535];
            let program = compile(&g, &target, &REPLAY).expect("the softmax compiles");
            let (_, plan) = program
                .planned()
                .find(|(eqn, _)| matches!(eqn.op, OpKind::FusedRow(_)))
                .expect("the softmax fuses into one row region");
            let Plan::Compute { body, grid, .. } = plan else {
                panic!("{backend:?}: the row region plans as Plan::Compute");
            };
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
        let b = Builder::new();
        let src = b.constant("src", TensorType::f32(vec![4, 5]));
        let index = b.constant("index", TensorType::new(vec![4], DType::I32));
        let out = b.scatter(src, index);
        let mut g = b.finish(out);
        let scatter_idx = g
            .eqns
            .iter()
            .position(|eqn| matches!(eqn.op, OpKind::Scatter { .. }))
            .expect("the traced graph has one Scatter equation");
        g.eqns[scatter_idx].op = OpKind::Scatter { axis: 1 };

        for backend in [
            Backend::SpirvVulkan,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
            Backend::Nvptx,
        ] {
            let refusal = refusal(
                compile(&g, &target_for(backend), &REPLAY),
                &format!("{backend:?} non-axis-0 scatter"),
            );
            assert_eq!(refusal.eqn, out.id);
            assert_eq!(refusal.target, backend);
            assert_eq!(refusal.missing, Capability::ScatterNonZeroAxis { axis: 1 });
        }
    }

    /// An equation the traced graph itself asks for, which the target has no kernel for, is refused: no
    /// rewrite produced it, so there is no decomposition to fall back to. (Card 534a: a non-last-axis
    /// `Reduce` no longer demonstrates this - `lower_nonlast_reduces` is now compile's own rewrite for
    /// it - so this uses a mixed-dtype `ScatterUpdate` no pass rewrites instead.)
    #[test]
    fn compile_refuses_an_equation_the_traced_graph_cannot_run() {
        let b = Builder::new();
        let base = b.constant("base", TensorType::new(vec![2, 5], DType::E4M3FN));
        let src = b.constant("src", TensorType::f32(vec![1, 5]));
        let inverse = b.constant("inverse", TensorType::f32(vec![2]));
        let out = b.scatter_update(base, src, inverse);
        let refusal = refusal(
            compile(&b.finish(out), &wgpu(), &REPLAY),
            "E4M3FN ScatterUpdate over an f32 source",
        );
        assert_eq!(refusal.eqn, out.id);
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
        let b = Builder::new();
        let x = b.slot(Slot::Activation, TensorType::new(vec![4, 8], DType::BF16));
        let s = b.slot(Slot::Activation, TensorType::new(vec![4, 8], DType::BF16));
        let bias = b.slot(Slot::Activation, TensorType::new(vec![4, 8], DType::BF16));
        let scaled = b.binary(BinOp::Mul, x, s);
        let e = b.unary(UnOp::Exp, scaled);
        let out = b.binary(BinOp::Add, e, bias);
        let g = b.finish(out);

        let optimized = transform::fuse(&g);
        let fused = optimized
            .eqns
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Fused(_)))
            .expect("fuse fuses the chain");
        assert!(
            matches!(
                plan_eqn_analyzed(
                    &ExactI32StorageAnalysis::new(&optimized),
                    &optimized,
                    fused,
                    Backend::SpirvVulkan,
                    &wgpu().caps,
                    &poot_test_util::graph_fixtures::roomy_body_limits()
                ),
                Err(PlanError::Refused(_))
            ),
            "the premise: no target plans a BF16 fused region"
        );

        let program = compile(&g, &wgpu(), &REPLAY).expect("the decomposition plans");
        let ops: Vec<String> = program
            .graph()
            .eqns
            .iter()
            .map(|eqn| eqn.op.name())
            .collect();
        assert_eq!(ops, ["mul", "exp", "add"], "the chain stays decomposed");
        assert_eq!(program.passes().count(), 0, "no pass changed the graph");
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
            program.graph().eqns.len() as u64,
            "one plan per equation"
        );

        let before = per_eqn_plans_inside_compile();
        let program = compile(&pointwise_chain(), &wgpu(), &REPLAY).expect("the chain compiles");
        assert_eq!(program.graph().eqns.len(), 1, "fusion leaves one region");
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
        plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&g),
            &g,
            &g.eqns[0],
            Backend::SpirvVulkan,
            &wgpu().caps,
            &poot_test_util::graph_fixtures::roomy_body_limits(),
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
        let b = Builder::new();
        let logits = b.constant("expert_logits", TensorType::f32(vec![4, 2]));
        let gate = b.reduce(RedOp::Sum, logits, 0, false);
        let g = b.finish(gate);

        let inputs = HashMap::from([(
            logits.id,
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
                    .eqns
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
        target.caps.max_buffer_bytes = max_buffer_bytes;
        target
    }

    /// A `[vocab, hidden]` f32 embed table gathered by a scalar `Slot::Token` (the decode shape), with
    /// an unsplit `lm_head`-shaped weight alongside it so `dce`/`fuse` see a realistic multi-value graph
    /// and the embed table is the only thing over `limit`.
    fn embed_gather_graph(vocab: usize, hidden: usize) -> Graph {
        let b = Builder::new();
        let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
        let embed = b.constant(
            "model.embed_tokens.weight",
            TensorType::f32(vec![vocab, hidden]),
        );
        let out = b.gather_scalar(embed, 0, token);
        b.finish(out)
    }

    /// `x @ w^T`, `w` an `[n, k]` named constant read through a transpose (how the tracers declare every
    /// dense weight). `x` is a broadcast of one real scalar, not a declared `[1, k]` constant:
    /// `legalize`'s split rewrite never looks at the activation's storage, only the weight's, so a huge
    /// `k` (forcing even one row over the limit) stays a cheap fixture.
    fn matmul_graph(x_name: &str, k: usize, w_name: &str, n: usize) -> Graph {
        let b = Builder::new();
        let x0 = b.constant(x_name, TensorType::f32(vec![1, 1]));
        let x = b.broadcast(x0, vec![1, k]);
        let w = b.constant(w_name, TensorType::f32(vec![n, k]));
        let out = b.matmul(x, b.transpose(w, vec![1, 0]));
        b.finish(out)
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
                .eqns
                .iter()
                .any(|e| matches!(e.op, OpKind::Gather { .. })),
            "the gather is replaced, not merely left decomposed"
        );
        let hosted = program
            .graph()
            .inputs
            .iter()
            .find(|&&id| program.graph().meta(id).storage == Storage::Slot(Slot::TokenEmbed))
            .expect("compile's output graph declares the hosted embed input");
        assert_eq!(
            program.graph().meta(*hosted).name.as_deref(),
            Some("model.embed_tokens.weight")
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
                .eqns
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
            .consts
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
                .eqns
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
            &bind_seed(&g.inputs, &g),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap()
        .output
        .into_host()
        .unwrap();

        let gh = program.graph();
        let mut split_inputs: HashMap<ValueId, Value> = HashMap::new();
        let mut row = 0usize;
        for &id in &gh.consts {
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

    // Card 642: the packed-contraction claim and Card 355's escape gate run inside
    // `compile`'s own pipeline, driven through `compile` itself.

    fn packed_linear_graph(format: poot_quant::format::WeightFormat, m: usize) -> Graph {
        let descriptor = poot_quant::PackedWeight::try_new(format, [8, 256]).unwrap();
        let b = Builder::new();
        let x = b.slot_named(
            Slot::Activation,
            "packed-compile",
            TensorType::f32(vec![m, 256]),
        );
        let out =
            poot_graph_ir::ops::packed_linear(&b, x, "layer", descriptor, None, None).unwrap();
        b.finish(out)
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
                    assert!(matches!(plan, Plan::ComputeMeta { .. }), "{what}: {plan:?}");
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
        let b = Builder::new();
        let sources: Vec<poot_graph_ir::Traced> =
            poot_graph_ir::packed_source_constants("layer", descriptor)
                .into_iter()
                .map(|(name, tensor_type)| b.constant(name.as_str(), tensor_type))
                .collect();
        let decoded = b.packed_dequant(&sources, descriptor);
        let graph = b.finish(decoded);
        match compile(&graph, &wgpu(), &REPLAY) {
            Err(CompileError::PackedDequant(error)) => assert!(
                matches!(*error, PackedDequantProductionError::GraphOutput { .. }),
                "{error:?}"
            ),
            other => panic!("expected the escape gate's refusal, got {other:?}"),
        }
    }

    // Card 642: the one packed MoE shape the escape gate admits until Card 632.

    const MOE_EXPERTS: usize = 2;
    const MOE_OUT: usize = 8;
    const MOE_K: usize = 64;

    fn moe_descriptor() -> poot_quant::PackedWeight {
        poot_quant::PackedWeight::try_new(poot_quant::format::WeightFormat::Q8_0, [MOE_OUT, MOE_K])
            .unwrap()
    }

    fn moe_rows() -> Vec<poot_graph_ir::ops::PackedLinearGraphRow> {
        (0..MOE_EXPERTS)
            .map(|ordinal| poot_graph_ir::ops::PackedLinearGraphRow {
                ordinal,
                linear_id: format!("expert.{ordinal}"),
                descriptor: moe_descriptor(),
            })
            .collect()
    }

    /// `packed_indexed_linear` (`grouped == false`) or `packed_grouped_linear` over two packed experts.
    fn packed_moe_graph(grouped: bool) -> Graph {
        let b = Builder::new();
        let x = b.slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![3, MOE_K]));
        let selector = b.slot_named(Slot::Activation, "moe-ids", TensorType::f32(vec![3]));
        let out = if grouped {
            poot_graph_ir::ops::packed_grouped_linear(&b, x, selector, &moe_rows())
        } else {
            poot_graph_ir::ops::packed_indexed_linear(&b, x, selector, &moe_rows())
        }
        .unwrap();
        b.finish(out)
    }

    /// One expert's staged `PackedDequant`, as the builders stage it.
    fn staged_dequant(b: &Builder, linear_id: &str) -> poot_graph_ir::Traced {
        let sources: Vec<poot_graph_ir::Traced> =
            poot_graph_ir::packed_source_constants(linear_id, moe_descriptor())
                .into_iter()
                .map(|(name, tensor_type)| b.constant(name.as_str(), tensor_type))
                .collect();
        b.packed_dequant(&sources, moe_descriptor())
    }

    /// A near miss of the canonical chain, built by hand: `Consumer` swaps the table's consumer for a
    /// reduce, `NoTranspose` drops the transpose (reshaping the `[out, K]` rows directly), `DenseBranch`
    /// concatenates a dense table next to a packed branch.
    enum NearMiss {
        Consumer,
        NoTranspose,
        DenseBranch,
    }

    fn near_miss_graph(kind: NearMiss) -> Graph {
        let b = Builder::new();
        let x_k = match kind {
            NearMiss::NoTranspose => MOE_OUT,
            _ => MOE_K,
        };
        let x = b.slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![3, x_k]));
        let selector = b.slot_named(Slot::Activation, "moe-ids", TensorType::f32(vec![3]));
        let branch = |id: &str| {
            let dequant = staged_dequant(&b, id);
            match kind {
                NearMiss::NoTranspose => b.reshape(dequant, vec![1, MOE_OUT, MOE_K]),
                _ => {
                    let t = b.transpose(dequant, vec![1, 0]);
                    b.reshape(t, vec![1, MOE_K, MOE_OUT])
                }
            }
        };
        let first = branch("expert.0");
        let second = match kind {
            NearMiss::DenseBranch => {
                b.constant("dense.expert", TensorType::f32(vec![1, MOE_K, MOE_OUT]))
            }
            _ => branch("expert.1"),
        };
        let table = b.concat(0, &[first, second]);
        let out = match kind {
            NearMiss::Consumer => b.reduce(RedOp::Sum, table, 0, false),
            _ => b.indexed_matmul(x, table, selector),
        };
        b.finish(out)
    }

    /// W12: the canonical indexed and grouped packed MoE chains compile on every backend, each expert
    /// dequant planned as `Materialize` and the table as the dense `IndexedMatMul`. Mutation: drop the
    /// `is_canonical_indexed_branch` admission from `reject_packed_dequant_escapes`; every row is
    /// refused (`Unmatched`).
    #[test]
    fn compile_admits_the_canonical_packed_moe_chain() {
        for backend in [
            Backend::SpirvVulkan,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
            Backend::Nvptx,
        ] {
            for grouped in [false, true] {
                let what = format!("{backend:?} grouped={grouped}");
                let program = compile(&packed_moe_graph(grouped), &target_for(backend), &REPLAY)
                    .unwrap_or_else(|error| panic!("{what}: {error}"));
                let dequants = program
                    .planned()
                    .filter(|(eqn, plan)| {
                        matches!(eqn.op, OpKind::PackedDequant { .. })
                            && matches!(plan, Plan::ComputeMeta { .. })
                            && matches!(
                                program.kernel_choice(eqn),
                                KernelChoice::Generated(KernelRequest::Packed(request))
                                    if request.name == "packed_materialize"
                            )
                    })
                    .count();
                assert_eq!(dequants, MOE_EXPERTS, "{what}: one Materialize per expert");
                assert!(
                    program
                        .planned()
                        .any(|(eqn, _)| matches!(eqn.op, OpKind::IndexedMatMul)),
                    "{what}: the table runs the dense IndexedMatMul"
                );
            }
        }
    }

    /// W12: a near miss of the canonical chain is still refused by the escape gate. Mutation: admit
    /// any `Transpose`/`Reshape`/`Concat` chain (drop the `IndexedMatMul` and dense-branch checks in
    /// `is_canonical_indexed_branch`); the `Consumer` and `DenseBranch` rows compile.
    #[test]
    fn compile_refuses_a_near_miss_packed_moe_chain() {
        for (what, kind) in [
            ("consumer", NearMiss::Consumer),
            ("no transpose", NearMiss::NoTranspose),
            ("dense branch", NearMiss::DenseBranch),
        ] {
            match compile(&near_miss_graph(kind), &wgpu(), &REPLAY) {
                Err(CompileError::PackedDequant(_)) => {}
                other => panic!(
                    "{what}: expected the escape gate's refusal, got {:?}",
                    other.map(|_| ())
                ),
            }
        }
    }

    /// Spike 562 F-9: one I32 `Slot::Pos` that is both a Gather index and an I32->F32 Cast source. Its
    /// component holds no exact-I32 arithmetic, so every reader reads the f32 mirror: the Gather through a
    /// `Slice<f32>` index, the Cast as an alias of the mirror bytes. Before the fix the Cast read
    /// `Slice<i32>` regardless and `compile` refused with `I32ReaderLaneConflict` (the wgpu executor, which
    /// does not run the check, bound dense I32 words the Gather read as floats). Mutation (never left in the
    /// tree): disabling that `Plan::Alias` arm -> `Plan(I32ReaderLaneConflict { value: 0, i32_reader: 3,
    /// f32_reader: 2 })`.
    #[test]
    fn compile_reads_a_mirror_i32_slot_on_one_lane_for_gather_and_cast() {
        let b = Builder::new();
        let idx = b.slot(Slot::Pos, TensorType::new(vec![4], DType::I32));
        let table = b.constant("table", TensorType::f32(vec![16, 8]));
        let gathered = b.gather(table, 0, idx);
        let q = b.cast(idx, DType::F32);
        let out = b.binary(
            BinOp::Add,
            gathered,
            b.broadcast(b.reshape(q, vec![4, 1]), vec![4, 8]),
        );
        let g = b.finish(out);

        let program =
            compile(&g, &wgpu(), &REPLAY).expect("every reader agrees on the mirror lane");
        let graph = program.graph();
        let &(slot, _) = graph
            .slots
            .iter()
            .find(|&&(_, kind)| kind == Slot::Pos)
            .expect("the Pos slot survives compile");
        assert!(program.storage().storage(slot).is_f32_mirror());
        let (_, cast_plan) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::Cast { .. }))
            .expect("the cast survives compile");
        assert!(
            matches!(cast_plan, Plan::Alias(source) if *source == slot),
            "a mirror-lane cast aliases its source, got {cast_plan:?}"
        );
    }

    /// The reader-lane contract on the production entry: an exact-I32 component (a `GeU` observes it) whose
    /// member is also a runtime DynamicUpdateSlice index, which the f32 DUS kernel reads through
    /// `Slice<f32>`, while the component's Gather reads it through `Slice<i32>`. No single buffer serves
    /// both, so `compile` refuses by value instead of producing a program that reads reinterpreted bits.
    /// Mutation (never left in the tree): discarding `check_agreement`'s result in
    /// `value_storage_of_plans` -> `compile` returns a `Program` and this test fails.
    #[test]
    fn compile_refuses_an_i32_value_read_on_both_lanes() {
        let b = Builder::new();
        let idx = b.slot(Slot::Pos, TensorType::new(vec![], DType::I32));
        let table = b.constant("table", TensorType::f32(vec![16, 8]));
        let row = b.gather(table, 0, idx);
        let late = b.cast(b.binary_scalar(BinOp::GeU, idx, Scalar::I32(3)), DType::F32);
        let update = b.binary(BinOp::Add, row, b.broadcast(late, vec![8]));
        let cache = b.state_input("cache", TensorType::f32(vec![32, 8]), StateRole::Recurrent);
        let written = b.dynamic_update_slice_dyn(cache, b.reshape(update, vec![1, 8]), idx, 0);
        let g = b.finish_with_state(written, &[(cache, written)]);

        let error = compile(&g, &wgpu(), &REPLAY).expect_err("the slot is read on both lanes");
        assert!(
            matches!(
                &error,
                CompileError::Plan(plan) if matches!(**plan, PlanError::I32ReaderLaneConflict { value, .. } if value == idx.id)
            ),
            "expected I32ReaderLaneConflict on v{}, got {error:?}",
            idx.id
        );
    }

    /// The same conflict reached through an alias: the exact-I32 `[1]` slot is the Gather's `Slice<i32>`
    /// index, and its `Reshape` to `[]` (a `Plan::Alias` of the slot's buffer) is the f32 DUS kernel's
    /// `Slice<f32>` index. Per value, each is read on one lane; per buffer, the slot is read on both, and
    /// the check names the slot.
    ///
    /// Mutation (never left in the tree): recording reads under the read value instead of its alias root
    /// in `I32ReadLanes::of_plans` -> `compile` returns a `Program` and this test fails.
    #[test]
    fn compile_refuses_an_i32_buffer_read_on_both_lanes_through_an_alias() {
        let b = Builder::new();
        let idx = b.slot(Slot::Pos, TensorType::new(vec![1], DType::I32));
        let table = b.constant("table", TensorType::f32(vec![16, 8]));
        let rows = b.gather(table, 0, idx);
        let late = b.cast(b.binary_scalar(BinOp::GeU, idx, Scalar::I32(3)), DType::F32);
        let update = b.binary(BinOp::Add, rows, b.broadcast(late, vec![1, 8]));
        let cache = b.state_input("cache", TensorType::f32(vec![32, 8]), StateRole::Recurrent);
        let start = b.reshape(idx, vec![]);
        let written = b.dynamic_update_slice_dyn(cache, update, start, 0);
        let g = b.finish_with_state(written, &[(cache, written)]);

        let error =
            compile(&g, &wgpu(), &REPLAY).expect_err("the slot's buffer is read on both lanes");
        assert!(
            matches!(
                &error,
                CompileError::Plan(plan) if matches!(**plan, PlanError::I32ReaderLaneConflict { value, .. } if value == idx.id)
            ),
            "expected I32ReaderLaneConflict on v{}, got {error:?}",
            idx.id
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
        poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs>,
        Vec<(&'static str, poot_graph_ir::ValueId)>,
    ) {
        let b = Builder::new();
        let mut members = Vec::new();
        let out = match chain {
            PackedChain::Canonical { reshape } => {
                let shape = if reshape {
                    vec![1, 2, MOE_K]
                } else {
                    vec![2, MOE_K]
                };
                let x = b.slot_named(Slot::Activation, "chain-x", TensorType::f32(shape));
                let dequant = staged_dequant(&b, "layer");
                let transpose = b.transpose(dequant, vec![1, 0]);
                members.extend([("dequant", dequant), ("transpose", transpose)]);
                let weight = if reshape {
                    let reshaped = b.reshape(transpose, vec![1, MOE_K, MOE_OUT]);
                    members.push(("reshape", reshaped));
                    reshaped
                } else {
                    transpose
                };
                b.matmul(x, weight)
            }
            PackedChain::Blocked => {
                let x = b.slot_named(
                    Slot::Activation,
                    "chain-x",
                    TensorType::f32(vec![2, 2, MOE_K]),
                );
                let dequant = staged_dequant(&b, "layer");
                let reshape = b.reshape(dequant, vec![2, MOE_OUT / 2, MOE_K]);
                let transpose = b.transpose(reshape, vec![0, 2, 1]);
                members.extend([
                    ("dequant", dequant),
                    ("reshape", reshape),
                    ("transpose", transpose),
                ]);
                b.matmul(x, transpose)
            }
            PackedChain::Transposed { reshape } => {
                let x = b.slot_named(Slot::Activation, "chain-x", TensorType::f32(vec![2, MOE_K]));
                let activation_t = b.transpose(x, vec![1, 0]);
                let dequant = staged_dequant(&b, "layer");
                members.extend([("activation_t", activation_t), ("dequant", dequant)]);
                let weight = if reshape {
                    let reshaped = b.reshape(dequant, vec![MOE_OUT, MOE_K]);
                    members.push(("reshape", reshaped));
                    reshaped
                } else {
                    dequant
                };
                let product = b.matmul(weight, activation_t);
                members.push(("product", product));
                b.transpose(product, vec![1, 0])
            }
        };
        let witness = live.map(|(name, live)| {
            let &(_, member) = members
                .iter()
                .find(|(member, _)| *member == name)
                .unwrap_or_else(|| panic!("{chain:?} has no member {name}"));
            match live {
                Live::Escape => member,
                Live::Consumer => b.reduce(RedOp::Sum, member, 0, false),
            }
        });
        let validations: Vec<(poot_graph_ir::ValidationId, &str, poot_graph_ir::Traced)> = witness
            .into_iter()
            .map(|witness| (poot_graph_ir::ValidationId(1), "live-member", witness))
            .collect();
        let graph = crate::test_support::finish_with_validations(b, out, &validations).unwrap();
        let members = members
            .into_iter()
            .map(|(name, member)| (name, member.id))
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
            let b = Builder::new();
            let x = b.slot_named(
                Slot::Activation,
                "square-x",
                TensorType::f32(vec![MOE_K, MOE_K]),
            );
            let sources: Vec<poot_graph_ir::Traced> =
                poot_graph_ir::packed_source_constants("layer", square)
                    .into_iter()
                    .map(|(name, tensor_type)| b.constant(name.as_str(), tensor_type))
                    .collect();
            let dequant = b.packed_dequant(&sources, square);
            let out = if transposed {
                let x_t = b.transpose(x, vec![1, 0]);
                let sum = b.binary(BinOp::Add, dequant, x_t);
                b.transpose(sum, vec![1, 0])
            } else {
                let w_t = b.transpose(dequant, vec![1, 0]);
                b.binary(BinOp::Add, x, w_t)
            };
            let graph = b.finish(out);
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
        let b = Builder::new();
        let a = staged_dequant(&b, "layer.a");
        let dequant_b = staged_dequant(&b, "layer.b");
        let b_t = b.transpose(dequant_b, vec![1, 0]);
        let product = b.matmul(a, b_t);
        let out = b.transpose(product, vec![1, 0]);
        let graph = b.finish(out);
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
        let b = Builder::new();
        let x = b.slot_named(Slot::Activation, "state-x", TensorType::f32(vec![2, MOE_K]));
        let constants = poot_graph_ir::packed_source_constants("layer", moe_descriptor());
        let carrier_type = constants[0].1.clone();
        let sources: Vec<poot_graph_ir::Traced> = constants
            .into_iter()
            .map(|(name, tensor_type)| b.constant(name.as_str(), tensor_type))
            .collect();
        let dequant = b.packed_dequant(&sources, moe_descriptor());
        let transpose = b.transpose(dequant, vec![1, 0]);
        let out = b.matmul(x, transpose);
        let state_in = b.state_input("carrier.state", carrier_type, StateRole::Recurrent);
        let graph = b.finish_with_state(out, &[(state_in, sources[0])]);
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
        let b = Builder::new();
        let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
        let kv_cache = b.state_input(
            "kv.cache",
            TensorType::f32(vec![1, 1, 4, 2]),
            StateRole::Positional { axis: 2 },
        );
        let update = b.constant("update", TensorType::f32(vec![1, 1, 1, 2]));
        let kv_out = b.dynamic_update_slice_dyn(kv_cache, update, pos_slot, 2);
        let gdn_state = b.state_input(
            "gdn.state",
            TensorType::f32(vec![2, 2]),
            StateRole::Recurrent,
        );
        let graph = b.finish_with_state(kv_out, &[(kv_cache, kv_out), (gdn_state, gdn_state)]);

        let program = compile(&graph, &wgpu(), &REPLAY).expect("the state fixture compiles");
        let roles: std::collections::HashMap<ValueId, StateRole> = program
            .graph()
            .state_pairs()
            .map(|pair| (pair.input, pair.role))
            .collect();
        assert_eq!(
            roles[&kv_cache.id],
            StateRole::Positional { axis: 2 },
            "the KV-cache pair's traced role must survive compile"
        );
        assert_eq!(
            roles[&gdn_state.id],
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
        let b = Builder::new();
        let ids = b.slot(Slot::Token, TensorType::f32(ids_shape));
        let rows = poot_graph_ir::ops::packed_embedding(
            &b,
            ids,
            "model.embed_tokens",
            embedding_descriptor(),
        )
        .unwrap();
        b.finish(rows)
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
                    ) && matches!(plan, Plan::ComputeMeta { .. })
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
    /// MoeHangGuard leaves this chain visible to the planner; Full fuses it before planning.
    fn former_region_chain_fixture() -> Graph {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![2, 5]));
        let raw = b.constant("w.weight", TensorType::new(vec![3, 5], DType::E4M3FN));
        let decoded = b.cast(raw, DType::F32);
        let scale = b.constant("w.weight_scale", TensorType::f32(vec![2, 3]));
        let out_index = b.constant("w.fp8_out_block_index", TensorType::f32(vec![3]));
        let in_index = b.constant("w.fp8_in_block_index", TensorType::f32(vec![5]));
        let rows = b.gather(scale, 0, out_index);
        let expanded_scale = b.gather(rows, 1, in_index);
        let scaled = b.binary(BinOp::Mul, decoded, expanded_scale);
        let weight = b.transpose(scaled, vec![1, 0]);
        let y = b.matmul(x, weight);
        b.finish(y)
    }

    fn dense_decode_fixture() -> Graph {
        let b = Builder::new();
        let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![1, 8]));
        let w = b.constant("w", TensorType::f32(vec![8, 8]));
        let bias = b.constant("bias", TensorType::f32(vec![8]));
        let y = poot_graph_ir::ops::linear(&b, x, w, Some(bias));
        b.finish(y)
    }

    fn prefill_fixture() -> Graph {
        let b = Builder::new();
        let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![8, 8]));
        let w = b.constant("w", TensorType::f32(vec![8, 8]));
        let bias = b.constant("bias", TensorType::f32(vec![8]));
        let y = poot_graph_ir::ops::linear(&b, x, w, Some(bias));
        b.finish(y)
    }

    fn packed_q8_0_linear_fixture() -> Graph {
        use poot_quant::PackedWeight;
        use poot_quant::format::WeightFormat;

        let b = Builder::new();
        let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![1, 256]));
        let weight = PackedWeight::try_new(WeightFormat::Q8_0, [64, 256]).unwrap();
        let y = poot_graph_ir::ops::packed_linear(&b, x, "layer", weight, None, None).unwrap();
        b.finish(y)
    }

    /// Submission mode does not affect the compiled plans. Both fusion policies matter: Full
    /// hides the old region shape, whereas MoeHangGuard exposed the baseline capture refusal.
    #[test]
    fn compile_output_does_not_depend_on_submission_mode() {
        let fixtures: [(&str, Graph); 4] = [
            ("dense_decode", dense_decode_fixture()),
            ("prefill", prefill_fixture()),
            ("packed_q8_0", packed_q8_0_linear_fixture()),
            ("former_region_chain", former_region_chain_fixture()),
        ];
        for (name, graph) in &fixtures {
            for fusion in [FusionPolicy::Full, FusionPolicy::MoeHangGuard] {
                let replay = CompileOptions {
                    execution: Submission::Replay,
                    fusion,
                    limits: crate::CompileLimits::STANDARD,
                };
                let uncached = CompileOptions {
                    execution: Submission::Eager,
                    fusion,
                    limits: crate::CompileLimits::STANDARD,
                };
                let under_replay = compile(graph, &wgpu(), &replay)
                    .unwrap_or_else(|error| panic!("{name}/{fusion:?}: CaptureReplay: {error}"));
                let under_uncached = compile(graph, &wgpu(), &uncached)
                    .unwrap_or_else(|error| panic!("{name}/{fusion:?}: UncachedResident: {error}"));
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
                    "{name}/{fusion:?}: CaptureReplay and UncachedResident produced different plans"
                );
            }
        }
    }

    /// W16: a gather the claim does not take - along axis 1 - is still refused by the gate.
    #[test]
    fn compile_refuses_a_packed_gather_along_another_axis() {
        let b = Builder::new();
        let ids = b.slot(Slot::Token, TensorType::f32(vec![2]));
        let sources: Vec<poot_graph_ir::Traced> =
            poot_graph_ir::packed_source_constants("table", embedding_descriptor())
                .into_iter()
                .map(|(name, tensor_type)| b.constant(name.as_str(), tensor_type))
                .collect();
        let table = b.packed_dequant(&sources, embedding_descriptor());
        let columns = b.gather(table, 1, ids);
        match compile(&b.finish(columns), &wgpu(), &REPLAY) {
            Err(CompileError::PackedDequant(_)) => {}
            other => panic!(
                "expected the escape gate's refusal, got {:?}",
                other.map(|_| ())
            ),
        }
    }
}
