//! Per-equation planning and exhaustive graph-op lowering.
//!
//! `plan_eqn_views_impl` is one thin dispatch table: it runs the shared dtype/storage guards, routes
//! each op family to its owned function in the sibling `planner/` child modules (each arm body moved
//! verbatim), then applies the shared launch-shaping postamble (`finalize`). Backend coverage and
//! fallback behavior stay auditable because every family keeps one exhaustive match over its ops.
//!
//! The partition of [`OpKind`] into families is written once, as the `*_ops!` pattern macros below. The
//! dispatcher and every family function match exhaustively over them, with no wildcard arm, so a new
//! `OpKind` variant fails to compile until it is assigned a family and that family handles it. A
//! refusal is a typed [`Refusal`], built by `Site::refuse`.

use poot_kernelgen::BodyLimits;
use poot_target::{Backend, DeviceCaps};

use crate::dtype_widen::bf16_const_is_packed;
use crate::*;
use poot_graph_ir::Storage;

/// Ops that lower through imported-kernel production planning, never through `plan_eqn`.
macro_rules! imported_ops {
    () => {
        OpKind::PackedDequant { .. }
            | OpKind::PackedContraction { .. }
            | OpKind::PackedRowGather { .. }
    };
}

/// Ops planned by `planner::packed`.
macro_rules! packed_ops {
    () => {
        OpKind::DenseContraction { .. }
            | OpKind::DenseRowGather { .. }
            | OpKind::PackI8
            | OpKind::UnpackI8 { .. }
    };
}

/// Ops planned by `planner::moe`.
macro_rules! moe_ops {
    () => {
        OpKind::IndexedMatMul | OpKind::ArgTopK { .. }
    };
}

/// Ops planned by `planner::elementwise`.
macro_rules! elementwise_ops {
    () => {
        OpKind::Binary(_)
            | OpKind::Select
            | OpKind::Unary(_)
            | OpKind::Rope { .. }
            | OpKind::Reduce { .. }
            | OpKind::Fused(_)
            | OpKind::FusedRow(_)
    };
}

/// Ops planned by `planner::matmul`.
macro_rules! matmul_ops {
    () => {
        OpKind::MatMul | OpKind::MatMulBias
    };
}

/// Ops planned by `planner::attention`.
macro_rules! attention_ops {
    () => {
        OpKind::FlashAttentionDecode { .. } | OpKind::FlashAttentionPrefill { .. }
    };
}

/// Ops planned by `planner::cast`.
macro_rules! cast_ops {
    () => {
        OpKind::Cast { .. }
    };
}

/// Ops planned by `planner::sampling` (card 551a).
macro_rules! sampling_ops {
    () => {
        OpKind::RandomUniform { .. } | OpKind::SampleToken { .. }
    };
}

/// Ops planned by `planner::movement`.
macro_rules! movement_ops {
    () => {
        OpKind::ScatterUpdate
            | OpKind::Transpose { .. }
            | OpKind::Slice { .. }
            | OpKind::Concat { .. }
            | OpKind::Gather { .. }
            | OpKind::Scatter { .. }
            | OpKind::Broadcast { .. }
            | OpKind::Reshape { .. }
            | OpKind::DynamicUpdateSlice { .. }
            | OpKind::AllReduce { .. }
            | OpKind::AllGather { .. }
    };
}

mod attention;
mod cast;
mod choice;
mod elementwise;
mod matmul;
mod moe;
mod movement;
mod packed;
mod packed_dequant;
mod sampling;

pub use choice::Planned;
pub use choice::{KernelChoice, NoDispatch};

use std::cell::Cell;

/// Everything [`plan_graph`] decides for one graph, which [`crate::compile`] assembles into a
/// [`crate::Program`].
pub(crate) struct PlannedGraph {
    /// One plan per equation, index-aligned with the graph's equations.
    pub(crate) plans: Vec<Plan>,
    /// The kernel choice behind each plan, index-aligned with `plans`.
    pub(crate) choices: Vec<KernelChoice>,
    pub(crate) storage: GraphStorage,
    pub(crate) slots: Vec<SlotSchema>,
}

/// Plan a whole graph for `target`: the one planner entry per graph. The graph-level decisions run in one
/// fixed order: strided views, one exact-I32 analysis, every equation's plan (kernel, launch grid, alias or
/// view), every value's storage and the slot schema. Only [`crate::compile`] calls it, after the graph
/// passes.
///
/// `compile` derives the native state-commit plan from this call's own `plans`
/// (`crate::state_commit::state_commit_from_plans`) rather than planning the graph a
/// second time.
pub(crate) fn plan_graph<V: ValidationChannel>(
    g: &Graph<V>,
    target: &Target,
    fusion: FusionPolicy,
    limits: &BodyLimits,
) -> Result<PlannedGraph, CompileError> {
    let _scope = CompileScope::enter();
    let backend = target.backend;
    let caps = &target.caps;
    let views = compute_views(g, backend);
    let analysis = ExactI32StorageAnalysis::new(g);
    let (plans, choices): (Vec<_>, Vec<_>) = g
        .eqns
        .iter()
        .map(|eqn| {
            plan_eqn_compiled(&analysis, g, eqn, target, &views, fusion, limits)
                .map(|Planned { plan, choice }| (plan, choice))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    // Storage reads the plans just made: no equation is planned a second time.
    let storage = value_storage_of_plans(g, backend, caps, plans.iter().map(Some))?;
    let slots = g
        .slots
        .iter()
        .map(|&(value, slot)| SlotSchema {
            value,
            slot,
            storage: storage.storage(value),
        })
        .collect();
    Ok(PlannedGraph {
        plans,
        choices,
        storage,
        slots,
    })
}

thread_local! {
    /// How many [`CompileScope`]s are open on this thread.
    static COMPILE_DEPTH: Cell<usize> = const { Cell::new(0) };
    /// Per-equation plans made on this thread while no [`CompileScope`] was open.
    #[cfg(test)]
    static PLANS_OUTSIDE_COMPILE: Cell<u64> = const { Cell::new(0) };
    /// Per-equation plans made on this thread inside a [`CompileScope`].
    #[cfg(test)]
    static PLANS_INSIDE_COMPILE: Cell<u64> = const { Cell::new(0) };
}

/// Marks this thread as inside [`crate::compile`] while it lives, so the per-equation plans compile makes
/// (its rewrite-legality checks and [`plan_graph`]) are told apart from a caller planning equations itself.
pub(crate) struct CompileScope(());

impl CompileScope {
    pub(crate) fn enter() -> Self {
        COMPILE_DEPTH.with(|depth| depth.set(depth.get() + 1));
        CompileScope(())
    }
}

impl Drop for CompileScope {
    fn drop(&mut self) {
        COMPILE_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

/// The number of per-equation plans made on this thread outside [`crate::compile`] so far (card 532
/// SC-003, [`only_plans_outside_compile_are_counted`](compile::tests::only_plans_outside_compile_are_counted)).
/// A consumer that has moved onto `compile` reaches the planner only through it, so running its path
/// leaves this count unchanged; a direct `plan_eqn*` call, or a graph-wide analysis that plans every
/// equation again (`storage_analysis`'s test-only `fixture_value_storage`, `state_commit`'s test-only
/// `commit_for_test`), raises it.
///
/// Card 546b: narrowed from `pub` (gated `any(test, feature = "test-support")`) to `pub(crate)` -
/// its one cross-crate reader, `poot-gpu`'s own `src/tests/decode_cache_warmth.rs`, moved off
/// `GpuExecutor::run_resident_kv_cached` and was deleted with it, leaving the `pub` accessor with no
/// caller outside this crate's own tests (dead-pub has no exemption for an `any(test, ...)` gate); the
/// remaining caller is this crate's own `compile.rs` test, which needs no more than `pub(crate)`.
#[cfg(test)]
pub(crate) fn per_eqn_plans_outside_compile() -> u64 {
    PLANS_OUTSIDE_COMPILE.with(Cell::get)
}

fn count_per_eqn_plan() {
    #[cfg(test)]
    if COMPILE_DEPTH.with(Cell::get) == 0 {
        PLANS_OUTSIDE_COMPILE.with(|count| count.set(count.get() + 1));
    } else {
        PLANS_INSIDE_COMPILE.with(|count| count.set(count.get() + 1));
    }
}

/// The number of per-equation plans made on this thread inside [`crate::compile`] so far: the rewrite
/// legality probes and [`plan_graph`]'s one pass.
#[cfg(test)]
pub(crate) fn per_eqn_plans_inside_compile() -> u64 {
    PLANS_INSIDE_COMPILE.with(Cell::get)
}

/// Plan one equation while reusing graph-wide representation analysis.
///
/// Bulk graph planners should construct one [`ExactI32StorageAnalysis`] and call this entry point for every
/// equation. `caps` is the device's measured [`DeviceCaps`] (a runtime's `Context::device_caps()`, card 522): every device rule (display-watchdog chunk budgets, the RADV tiled-GEMM ceiling, tensor-core
/// support) reads it, never a `Backend`-keyed default. A caller with no measured device yet (a test, the
/// plan-summary corpus) passes a documented fixture.
///
/// This is the single-device entry point: it plans at `world_size = 1`, so the tensor-parallel
/// collectives (`AllReduce`/`AllGather`) lower to [`Plan::Alias`] (identity). The multi-rank executor
/// (card 049a) calls [`plan_eqn_views_analyzed`] with the real world size, where the collectives lower to
/// [`Plan::Collective`].
pub fn plan_eqn_analyzed<V: ValidationChannel>(
    analysis: &ExactI32StorageAnalysis<'_, V>,
    g: &Graph<V>,
    eqn: &Eqn,
    backend: Backend,
    caps: &DeviceCaps,
    limits: &BodyLimits,
) -> Result<Plan, PlanError> {
    plan_eqn_views_analyzed(analysis, g, eqn, backend, 1, &HashMap::new(), caps, limits)
}

/// spec 024: map an output storage dtype to the kernel element `Ty`. bf16/f16 bridge to their own
/// 2-byte kernel type (arithmetic still accumulates in f32); everything else runs the plain f32 kernel
/// body. A free function so planners outside the main `match` (e.g. [`scatter_update_plan`]'s
/// chunking) can build a dtype-correct `Body` too.
pub(crate) fn fty_ty(d: DType) -> Ty {
    match d {
        DType::BF16 => Ty::BF16,
        DType::F16 => Ty::F16,
        DType::F32 | DType::I32 | DType::I8 => Ty::F32,
        DType::E4M3FN => {
            unreachable!("e4m3fn is rejected before scalar kernel type selection")
        }
        DType::Bool
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::U32
        | DType::I64
        | DType::U64
        | DType::F64
        | DType::E8M0 => unreachable!(
            "{d:?} is a checkpoint-storage-only dtype; no traced graph value declares it"
        ),
    }
}

/// Card 546a (S46-11, moved from `poot-gpu`'s `exec.rs::eqn_work_units`, deleted as a vestigial
/// wrapper before this card landed): one equation's watchdog work weight, the
/// planner-owned cost `Device::dispatch`'s caller (the contract engine) carries in `Dispatch::work`
/// to split a long run of dispatches across several submits before the device driver's TDR fires.
///
/// Every equation's weight is its output element count: the dense elementwise/reduction/contraction
/// kernels this crate plans are all one thread (or one serial per-thread reduction) per output
/// element. `PackedContraction` does not need its own arm: Card 658 retagged every packed contraction
/// onto `Schedule::Tiled`'s fold-aware grid (never `Schedule::Serial`'s per-thread-unbounded K loop),
/// so the O(K) serial-reduction gap an earlier design named is already closed - the output
/// element count bounds the per-thread work for every op class this planner emits.
pub fn dispatch_work<V: ValidationChannel>(g: &Graph<V>, eqn: &Eqn) -> u64 {
    numel(&g.aval(eqn.out).shape) as u64
}

/// Storage type for data-movement kernels. Unlike arithmetic kernels, these must preserve arbitrary I32
/// bit patterns instead of routing them through the legacy f32 compute convention.
pub(crate) fn movement_ty(d: DType, exact_i32: bool) -> Ty {
    if exact_i32 { Ty::I32 } else { fty_ty(d) }
}

/// [`plan_eqn_analyzed`] with caller-owned whole-graph representation analysis, generalized to a strided
/// -views map (spec 132 phase 1, `compute_views`) and a multi-GPU `world_size` (card 049a). A `views`
/// entry for a Transpose/Slice/Broadcast output routes that eqn to [`Plan::View`] (no dispatch) instead
/// of a materialize copy; a `views` entry for a Binary/Fused/FusedRow operand routes that leaf's
/// kernelgen read through the operand's [`Layout`] instead of assuming a plain row-major buffer. An
/// empty map is a no-op: every arm falls through to the all-copy lowering. At `world_size == 1` with an
/// empty map this is byte-identical to [`plan_eqn_analyzed`] (the collectives are the identity
/// [`Plan::Alias`], FR-007); at `world_size > 1` the collective eqns lower to [`Plan::Collective`] for
/// the multi-rank executor. Reads every device rule
/// (display-watchdog chunk budgets, the RADV tiled-GEMM ceiling) from the caller's `caps`, never a
/// `Backend`-keyed default (card 522: Card 532's `compile` entry point threads a real
/// per-device `DeviceCaps` to every call site; until a caller has one, it passes a documented fixture,
/// e.g. `poot_test_util::device_caps::default_caps_for`). `out_shape`/`out_numel` are graph-derived, not
/// caller-supplied (card 525 R469-019).
#[allow(clippy::too_many_arguments)]
pub fn plan_eqn_views_analyzed<V: ValidationChannel>(
    analysis: &ExactI32StorageAnalysis<'_, V>,
    g: &Graph<V>,
    eqn: &Eqn,
    backend: Backend,
    world_size: usize,
    views: &HashMap<ValueId, Layout>,
    caps: &DeviceCaps,
    limits: &BodyLimits,
) -> Result<Plan, PlanError> {
    plan_eqn_choice_analyzed(analysis, g, eqn, backend, world_size, views, caps, limits)
        .map(|planned| planned.plan)
}

/// [`plan_eqn_views_analyzed`], keeping the [`KernelChoice`] the plan was built from. A planner test that
/// asserts on the choice calls it directly. It plans as [`crate::compile`] does under
/// [`FusionPolicy::Full`]; `compile` itself plans through `plan_eqn_compiled` with the policy it was given.
#[allow(clippy::too_many_arguments)]
pub fn plan_eqn_choice_analyzed<V: ValidationChannel>(
    analysis: &ExactI32StorageAnalysis<'_, V>,
    g: &Graph<V>,
    eqn: &Eqn,
    backend: Backend,
    world_size: usize,
    views: &HashMap<ValueId, Layout>,
    caps: &DeviceCaps,
    limits: &BodyLimits,
) -> Result<Planned, PlanError> {
    plan_eqn_policy(
        analysis,
        g,
        eqn,
        backend,
        world_size,
        views,
        caps,
        FusionPolicy::Full,
        limits,
    )
}

/// The single-device per-equation plan [`crate::compile`] makes, in [`plan_graph`] and in its rewrite
/// legality probes: `target`'s backend and caps, the graph's strided `views`, and the compile's `fusion`
/// policy, which the contraction choice reads (Card 557: [`FusionPolicy::MoeHangGuard`] keeps a matmul
/// off the generated tiled GEMM).
pub(crate) fn plan_eqn_compiled<V: ValidationChannel>(
    analysis: &ExactI32StorageAnalysis<'_, V>,
    g: &Graph<V>,
    eqn: &Eqn,
    target: &Target,
    views: &HashMap<ValueId, Layout>,
    fusion: FusionPolicy,
    limits: &BodyLimits,
) -> Result<Planned, PlanError> {
    plan_eqn_policy(
        analysis,
        g,
        eqn,
        target.backend,
        1,
        views,
        &target.caps,
        fusion,
        limits,
    )
}

#[allow(clippy::too_many_arguments)]
fn plan_eqn_policy<V: ValidationChannel>(
    analysis: &ExactI32StorageAnalysis<'_, V>,
    g: &Graph<V>,
    eqn: &Eqn,
    backend: Backend,
    world_size: usize,
    views: &HashMap<ValueId, Layout>,
    caps: &DeviceCaps,
    fusion: FusionPolicy,
    limits: &BodyLimits,
) -> Result<Planned, PlanError> {
    analysis.ensure_graph(g)?;
    analysis.ensure_eqn(eqn)?;
    let out_shape = g.aval(eqn.out).shape.clone();
    let out_numel = out_numel_saturating(&out_shape);
    // Card 1011: whether a `Cast` reads its BF16 const source from packed `u32` lanes. The lane is the
    // storage plan's (`bf16_const_is_packed`), which needs the whole graph, so it is decided here.
    let packed_cast_source = matches!(eqn.op, OpKind::Cast { .. })
        && matches!(
            eqn.inputs.as_slice(),
            [Operand::Value(src)]
                if g.meta(*src).storage == Storage::Const
                    && g.aval(*src).dtype == DType::BF16
                    && bf16_const_is_packed(g, *src, backend, caps)
        );
    plan_eqn_views_impl(
        analysis.requirement_table(),
        GraphTables::from(g),
        eqn,
        &out_shape,
        out_numel,
        backend,
        world_size,
        views,
        caps,
        fusion,
        packed_cast_source,
        *limits,
    )
}

/// `numel(shape)` without the panic: a caller-supplied shape (R469-019: now always graph-derived, but
/// `exact_i32_planner_rejects_static_numel_overflow` proves a value's `aval` can itself hold an
/// out-of-range shape) can overflow `usize` on plain multiplication. Saturates instead, so the real
/// overflow diagnostic is `plan_eqn_views_impl`'s own `checked_mul` scan over `out_shape` (a typed
/// `PlanError::BadShape`), not a panic in the entry point that derives `out_numel` before that scan runs.
fn out_numel_saturating(shape: &[usize]) -> usize {
    shape
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim))
        .unwrap_or(usize::MAX)
        .max(1)
}

/// The per-equation planner body. `out_shape`/`out_numel` are the caller's already-derived
/// [`Graph::aval`] of `eqn.out` (card 525 R469-019: no entry point takes a caller-supplied shape that
/// could disagree with the graph's own inferred one). It reads only the value table and the exact-I32
/// requirement table, so it is not generic over the validation channel and is compiled once in this crate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_eqn_views_impl(
    analysis: ExactI32Requirements<'_>,
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    world_size: usize,
    views: &HashMap<ValueId, Layout>,
    caps: &DeviceCaps,
    fusion: FusionPolicy,
    packed_cast_source: bool,
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    count_per_eqn_plan();
    let site = Site::new(g, eqn, backend, limits);
    if matches!(eqn.op, imported_ops!()) {
        // Card 542a: a block-32 descriptor (the seven GGUF schemes `packed_block_float::
        // PACKED_LOWERING` admits) reaches here through the descriptor-driven lowering table.
        let odt = g.aval(eqn.out).dtype;
        if let Some(planned) =
            packed_dequant::try_plan(g, eqn, out_shape, out_numel, backend, odt, caps, limits)?
        {
            return Ok(planned);
        }
        return Err(site.refuse(Capability::ImportedKernelPlanning));
    }
    let ids = value_ids(eqn);

    // Spec 149 SC-005d/e/f/g/h/i/j/k/n: only storage-aware executor entry points bind the packed
    // representation. The shared planner exposes Cast/Reshape/Transpose/Slice/Gather/Concat/Broadcast/
    // DynamicUpdateSlice/ScatterUpdate bodies here for SPIR-V/wgpu and bounded PTX typed capture. Legacy
    // tensor-only entry points call `validate_graph_execution` first. Every other op/backend keeps a
    // named rejection before scalar type selection can reinterpret E4M3FN as f32/i8/i32.
    let has_e4m3fn_input = ids.iter().any(|&id| g.aval(id).dtype == DType::E4M3FN);
    if g.aval(eqn.out).dtype == DType::E4M3FN || has_e4m3fn_input {
        if matches!(eqn.op, OpKind::Gather { .. })
            && (ids.len() != 2
                || g.aval(ids[0]).dtype != DType::E4M3FN
                || g.aval(ids[1]).dtype != DType::F32)
        {
            return Err(site.refuse(Capability::DtypeLowering));
        }
        if matches!(eqn.op, OpKind::ScatterUpdate)
            && (ids.len() != 3
                || g.aval(ids[0]).dtype != DType::E4M3FN
                || g.aval(ids[1]).dtype != DType::E4M3FN
                || g.aval(ids[2]).dtype != DType::F32
                || g.aval(eqn.out).dtype != DType::E4M3FN)
        {
            return Err(site.refuse(Capability::DtypeLowering));
        }
        if matches!(eqn.op, OpKind::DynamicUpdateSlice { .. }) {
            let valid_storage = g.aval(eqn.out).dtype == DType::E4M3FN
                && matches!(eqn.inputs.first(), Some(Operand::Value(id)) if g.aval(*id).dtype == DType::E4M3FN)
                && matches!(eqn.inputs.get(1), Some(Operand::Value(id)) if g.aval(*id).dtype == DType::E4M3FN)
                && match eqn.inputs.get(2) {
                    Some(Operand::Lit(Scalar::I32(_))) => true,
                    Some(Operand::Value(id)) => g.aval(*id).dtype == DType::F32,
                    _ => false,
                };
            if !valid_storage {
                return Err(site.refuse(Capability::DtypeLowering));
            }
        }
        if !matches!(backend, Backend::SpirvVulkan | Backend::Nvptx) {
            return Err(site.refuse(Capability::E4m3Storage));
        }
        if matches!(eqn.op, OpKind::Concat { .. })
            && (g.aval(eqn.out).dtype != DType::E4M3FN
                || ids.iter().any(|&id| g.aval(id).dtype != DType::E4M3FN))
        {
            return Err(site.refuse(Capability::DtypeLowering));
        }
        if !matches!(
            eqn.op,
            OpKind::Cast { .. }
                | OpKind::Reshape { .. }
                | OpKind::Transpose { .. }
                | OpKind::Slice { .. }
                | OpKind::Gather { .. }
                | OpKind::Broadcast { .. }
                | OpKind::Concat { .. }
                | OpKind::ScatterUpdate
                | OpKind::DynamicUpdateSlice { .. }
                // Card 449 D1: the PLE sharded E4M3FN table lookup `fold_dense_bf16_row_gathers`
                // rewrites into one equation (the Card 381 BF16 row's second source dtype). The
                // table is the only E4M3FN operand - the index is exact I32 and the output F32 -
                // and `validate_typed_plan_storage` proves each operand's lane before allocation,
                // so no dtype rule beyond "this op may read E4M3FN" is needed here.
                | OpKind::DenseRowGather { source: DType::E4M3FN }
        ) {
            return Err(site.refuse(Capability::E4m3Storage));
        }
    }

    // Card 546a (R-546-4): a non-axis-0 scatter (the `Scatter` op is axis-0-only by definition, so this
    // is a defensive guard that no live tracer hits) is refused outright on every target: the contract
    // has no host fallback (Card 626 deleted `Plan::Host`). Non-axis-0 GATHER (`gather_axis_dt`)
    // and `>2` (and 1-input) CONCAT (`concat_n_dt`) run on-device (card 043); the MoE gate is a primitive
    // composition (card 061).
    if let OpKind::Scatter { axis } = eqn.op
        && axis >= 1
    {
        return Err(site.refuse(Capability::ScatterNonZeroAxis { axis }));
    }

    // spec 024: the output storage dtype. bf16 is wired only for MatMul, broadcast Binary (value
    // operand), and Reshape (a buffer alias, dtype-agnostic); reject bf16 elsewhere with a clear
    // diagnostic rather than silently emitting an f32 kernel on bf16 bytes. `fty` maps the
    // dtype to the kernel element Ty; `dtag` is the cache-key dtype tag (empty for f32, so f32 keys are
    // unchanged).
    let odt = g.aval(eqn.out).dtype;
    if odt == DType::I32
        && analysis.required(eqn.out)
        && out_shape
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .is_none()
    {
        return Err(PlanError::BadShape(format!(
            "exact I32 output shape {out_shape:?} overflows usize"
        )));
    }
    // spec 135: F16 bridges to Ty::F16 exactly like BF16 bridges to Ty::BF16 (storage is 2 bytes;
    // arithmetic accumulates in f32 in both cases). It must not fall into the `else` and be bridged to
    // Ty::F32 over 2-byte storage, which miscompiles.
    if odt == DType::BF16 || odt == DType::F16 {
        let wired = matches!(
            eqn.op,
            OpKind::MatMul
                | OpKind::MatMulBias
                | OpKind::Reshape { .. }
                | OpKind::Transpose { .. }
                | OpKind::Slice { .. }
                | OpKind::Broadcast { .. }
                | OpKind::Binary(_)
                | OpKind::Unary(_)
                | OpKind::Gather { .. }
                | OpKind::Concat { .. }
                | OpKind::DynamicUpdateSlice { .. }
                | OpKind::Reduce { .. }
                | OpKind::Cast { .. }
        );
        if !wired {
            return Err(site.refuse(Capability::DtypeLowering));
        }
    }
    if odt == DType::I32 && matches!(eqn.op, OpKind::Reduce { .. } | OpKind::FusedRow(_)) {
        return Err(site.refuse(Capability::ExactI32Op));
    }

    let mut planned = match &eqn.op {
        // The nullary `Iota` is a pure compile-time constant: the `fold_iota` pass rewrites it into a
        // `Storage::Computed` graph constant before planning, and a graph that reaches the planner with
        // one unfolded is refused here. There is deliberately no iota kernel.
        OpKind::Iota { .. } => return Err(site.refuse(Capability::UnfoldedIota)),
        // The packed and MoE families run `finalize` at their own tail and return here directly; every
        // other family falls through to the `finalize` below. No arm returns a plan without it.
        packed_ops!() => {
            return packed::plan(
                g, eqn, out_shape, out_numel, backend, odt, caps, limits, fusion,
            );
        }
        moe_ops!() => return moe::plan(g, eqn, out_shape, out_numel, backend, odt, caps, limits),
        elementwise_ops!() => elementwise::plan(
            analysis, g, eqn, out_shape, out_numel, backend, views, odt, caps, limits,
        )?,
        matmul_ops!() => matmul::plan(
            g, eqn, out_shape, out_numel, backend, odt, caps, limits, fusion,
        )?,
        attention_ops!() => attention::plan(g, eqn, out_shape, backend, caps, limits)?,
        cast_ops!() => cast::plan(
            analysis,
            g,
            eqn,
            out_shape,
            out_numel,
            backend,
            caps,
            limits,
            packed_cast_source,
        )?,
        movement_ops!() => movement::plan(
            analysis, g, eqn, out_shape, out_numel, backend, views, odt, world_size, caps, limits,
        )?,
        sampling_ops!() => sampling::plan(g, eqn, out_shape, out_numel, backend, odt, caps)?,
        imported_ops!() => {
            unreachable!("refused above: imported ops never enter the generic planner")
        }
    };
    finalize(
        &mut planned,
        g,
        eqn,
        out_shape,
        out_numel,
        backend,
        odt,
        caps,
    )?;
    Ok(planned)
}
