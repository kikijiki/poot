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
use poot_target::{AieCaps, AieTargetId, Backend, DeviceCaps};

use crate::dtype_widen::bf16_const_is_packed;
use crate::*;

/// Ops over a packed weight, planned by the descriptor-driven lowering table (`planner::packed_dequant`),
/// never through the per-family `plan_eqn` bodies.
macro_rules! imported_ops {
    () => {
        OpKind::PackedDequant { .. }
            | OpKind::PackedContraction { .. }
            | OpKind::PackedRowGather { .. }
            | OpKind::ContractionEpilogue {
                head: ContractionHead::Packed { .. },
                ..
            }
    };
}

/// Ops planned by `planner::packed`.
macro_rules! packed_ops {
    () => {
        OpKind::DenseContraction { .. }
            | OpKind::GroupedDenseContraction { .. }
            | OpKind::ContractionEpilogue {
                head: ContractionHead::Dense { .. },
                ..
            }
            | OpKind::DenseRowGather { .. }
            | OpKind::PackI8
            | OpKind::UnpackI8 { .. }
    };
}

/// Ops planned by `planner::moe`.
macro_rules! moe_ops {
    () => {
        OpKind::IndexedMatMul
            | OpKind::ContractionEpilogue {
                head: ContractionHead::Indexed,
                ..
            }
            | OpKind::ArgTopK { .. }
    };
}

/// What a family planner generates for: a GPU backend and the device's measured capabilities, or an AIE
/// core (whose limits are checked by its own finalizer, never read by a request builder). A family that has
/// a sequential schedule takes this instead of a `(Backend, &DeviceCaps)` pair, so it cannot read GPU
/// capabilities for a target that has none.
#[derive(Clone, Copy)]
pub(crate) enum Lowering<'a> {
    Gpu {
        backend: Backend,
        caps: &'a DeviceCaps,
    },
    Aie(AieTargetId),
}

impl Lowering<'_> {
    /// The target identity, for a refusal site.
    pub(crate) fn id(self) -> poot_target::TargetId {
        match self {
            Lowering::Gpu { backend, .. } => backend.into(),
            Lowering::Aie(target) => target.into(),
        }
    }
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
        OpKind::MatMul
            | OpKind::ContractionEpilogue {
                head: ContractionHead::MatMul,
                ..
            }
    };
}

/// Ops planned by `planner::attention`.
macro_rules! attention_ops {
    () => {
        OpKind::Attention(_)
    };
}

/// Ops planned by `planner::scan`.
macro_rules! scan_ops {
    () => {
        OpKind::Scan(_)
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
mod scan;

pub use choice::Planned;
pub use choice::{KernelChoice, NoDispatch};
pub(crate) use matmul::epilogue_head_lowers;

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
/// fixed order: strided views, every equation's plan (kernel, launch grid, alias or view), every value's
/// storage and the slot schema. Only [`crate::compile`] calls it, after the graph
/// passes.
///
/// `compile` derives the native state-commit plan from this call's own `plans`
/// (`crate::state_commit::state_commit_from_plans`) rather than planning the graph a
/// second time.
pub(crate) fn plan_graph(
    g: &Graph,
    target: &Target,
    limits: &BodyLimits,
) -> Result<PlannedGraph, CompileError> {
    let _scope = CompileScope::enter();
    let id = target.id();
    let views = compute_views(g, id);
    let (plans, choices): (Vec<_>, Vec<_>) = g
        .eqns()
        .iter()
        .map(|eqn| {
            plan_eqn_compiled(g, eqn, target, &views, limits)
                .map(|Planned { plan, choice }| (plan, choice))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    // Storage reads the plans just made: no equation is planned a second time.
    let storage = value_storage_of_plans(g, id, plans.iter().map(Some))?;
    let slots = g
        .slots()
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
/// leaves this count unchanged; a direct `plan_eqn*` call, or a graph-wide check that plans every
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

/// Plan one equation. `caps` is the device's measured [`DeviceCaps`] (a runtime's `Context::device_caps()`,
/// card 522): every device rule (display-watchdog chunk budgets, the RADV tiled-GEMM ceiling, tensor-core
/// support) reads it, never a `Backend`-keyed default. A caller with no measured device yet (a test, the
/// plan-summary corpus) passes a documented fixture.
///
/// A collective (`AllReduce`/`AllGather`) is refused with [`Capability::CollectiveInStage`]: it is a cut
/// between stage programs, never an equation one plans.
pub fn plan_eqn(
    g: &Graph,
    eqn: &Eqn,
    backend: Backend,
    caps: &DeviceCaps,
    limits: &BodyLimits,
) -> Result<Plan, PlanError> {
    plan_eqn_views(g, eqn, backend, &HashMap::new(), caps, limits)
}

/// spec 024: map an output storage dtype to the kernel element `Ty`. bf16/f16 bridge to their own
/// 2-byte kernel type (arithmetic still accumulates in f32); I32 is its own element type (POOT-725: no
/// I32 value is computed or stored through f32); everything else runs the plain f32 kernel body. A free
/// function so planners outside the main `match` can build a dtype-correct `Body` too.
pub(crate) fn fty_ty(d: DType) -> Ty {
    match d {
        DType::BF16 => Ty::BF16,
        DType::F16 => Ty::F16,
        DType::I32 => Ty::I32,
        DType::F32 | DType::I8 => Ty::F32,
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
pub fn dispatch_work(g: &Graph, eqn: &Eqn) -> u64 {
    numel(&g.aval(eqn.out).shape) as u64
}

/// [`plan_eqn`] generalized to a strided-views map (spec 132 phase 1, `compute_views`). A `views`
/// entry for a Transpose/Slice/Broadcast output routes that eqn to [`Plan::View`] (no dispatch) instead
/// of a materialize copy; a `views` entry for a Binary/Fused/FusedRow operand routes that leaf's
/// kernelgen read through the operand's [`Layout`] instead of assuming a plain row-major buffer. An
/// empty map is a no-op: every arm falls through to the all-copy lowering, byte-identical to
/// [`plan_eqn`]. Reads every device rule
/// (display-watchdog chunk budgets, the RADV tiled-GEMM ceiling) from the caller's `caps`, never a
/// `Backend`-keyed default (card 522: Card 532's `compile` entry point threads a real
/// per-device `DeviceCaps` to every call site; until a caller has one, it passes a documented fixture,
/// e.g. `poot_test_util::device_caps::default_caps_for`). `out_shape`/`out_numel` are graph-derived, not
/// caller-supplied (card 525 R469-019).
#[allow(clippy::too_many_arguments)]
pub fn plan_eqn_views(
    g: &Graph,
    eqn: &Eqn,
    backend: Backend,
    views: &HashMap<ValueId, Layout>,
    caps: &DeviceCaps,
    limits: &BodyLimits,
) -> Result<Plan, PlanError> {
    let target = Target::Gpu {
        backend,
        caps: *caps,
    };
    plan_eqn_choice(g, eqn, &target, views, limits).map(|planned| planned.plan)
}

/// [`plan_eqn_views`] for any [`Target`], keeping the [`KernelChoice`] the plan was built from: the one
/// per-equation planner entry. A GPU target plans a thread grid, an AIE target the same requests under its
/// sequential one-core schedule; an equation whose lowering reads GPU capabilities is refused for an AIE
/// target with [`Capability::GpuOnly`]. A planner test that asserts on the choice calls it directly.
pub fn plan_eqn_choice(
    g: &Graph,
    eqn: &Eqn,
    target: &Target,
    views: &HashMap<ValueId, Layout>,
    limits: &BodyLimits,
) -> Result<Planned, PlanError> {
    plan_eqn_policy(g, eqn, target, views, limits)
}

/// The single-device per-equation plan [`crate::compile`] makes, in [`plan_graph`] and in its rewrite
/// legality probes: `target` and the graph's strided `views`.
pub(crate) fn plan_eqn_compiled(
    g: &Graph,
    eqn: &Eqn,
    target: &Target,
    views: &HashMap<ValueId, Layout>,
    limits: &BodyLimits,
) -> Result<Planned, PlanError> {
    plan_eqn_policy(g, eqn, target, views, limits)
}

/// Whether `op` can read a BF16 const operand through a packed lane: the `Cast` kernel, and the fused regions
/// whose `Cast` steps widen their narrow inputs in registers.
fn reads_narrow_operands(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::Cast { .. } | OpKind::FusedRow(_) | OpKind::ContractionEpilogue { .. }
    )
}

fn plan_eqn_policy(
    g: &Graph,
    eqn: &Eqn,
    target: &Target,
    views: &HashMap<ValueId, Layout>,
    limits: &BodyLimits,
) -> Result<Planned, PlanError> {
    let out_shape = g.aval(eqn.out).shape.clone();
    let out_numel = out_numel_saturating(&out_shape);
    // Card 1011: which operands a narrow reader (a `Cast`, or a fused region that widens its input) reads from
    // packed `u32` lanes. The lane is the storage plan's (`bf16_const_is_packed`), which needs the whole graph,
    // so it is decided here.
    let packed_operands: Vec<bool> = if reads_narrow_operands(&eqn.op) {
        eqn.inputs
            .iter()
            .map(|operand| {
                matches!(
                    operand,
                    Operand::Value(src)
                        if g.meta(*src).storage.holds_weight_bytes()
                            && g.aval(*src).dtype == DType::BF16
                            && bf16_const_is_packed(g, views, *src, target.id())
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    plan_eqn_views_impl(
        GraphTables::from(g),
        eqn,
        &out_shape,
        out_numel,
        target,
        views,
        &packed_operands,
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

/// The per-equation planner entry over a [`Target`]: a GPU target runs the family dispatcher
/// ([`plan_gpu_eqn`]), an AIE target the sequential-schedule one ([`plan_aie_eqn`]). `out_shape`/`out_numel`
/// are the caller's already-derived
/// [`Graph::aval`] of `eqn.out` (card 525 R469-019: no entry point takes a caller-supplied shape that
/// could disagree with the graph's own inferred one). It reads only the value table, so it is not generic
/// over the validation channel and is compiled once in this crate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_eqn_views_impl(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    target: &Target,
    views: &HashMap<ValueId, Layout>,
    packed_operands: &[bool],
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    count_per_eqn_plan();
    match target {
        Target::Gpu { backend, caps } => plan_gpu_eqn(
            g,
            eqn,
            out_shape,
            out_numel,
            *backend,
            views,
            caps,
            packed_operands,
            limits,
        ),
        Target::Aie { target, caps } => plan_aie_eqn(
            g,
            eqn,
            out_shape,
            out_numel,
            *target,
            caps,
            views,
            packed_operands,
            limits,
        ),
    }
}

/// An AIE target's per-equation plan. The AIE core runs one sequential loop, and `poot-kernelgen` has a
/// sequential schedule for the elementwise family only, so that family is planned (through the same
/// request builder every target uses, under [`Lowering::Aie`]) and finalized against the target's own
/// limits. Every other family's lowering reads GPU capabilities ([`DeviceCaps`]) and is refused with
/// [`Capability::GpuOnly`]; a collective is a cut between stage programs on every target.
#[allow(clippy::too_many_arguments)]
fn plan_aie_eqn(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    target: AieTargetId,
    caps: &AieCaps,
    views: &HashMap<ValueId, Layout>,
    packed_operands: &[bool],
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, target, limits);
    // An AIE core has no E4M3FN storage lane: refused before scalar type selection can reinterpret it.
    let touches_e4m3fn = g.aval(eqn.out).dtype == DType::E4M3FN
        || value_ids(eqn)
            .iter()
            .any(|&id| g.aval(id).dtype == DType::E4M3FN);
    if touches_e4m3fn {
        return Err(site.refuse(Capability::E4m3Storage));
    }
    let odt = g.aval(eqn.out).dtype;
    let planned = match &eqn.op {
        elementwise_ops!() => elementwise::plan(
            g,
            eqn,
            out_shape,
            out_numel,
            Lowering::Aie(target),
            views,
            odt,
            limits,
            packed_operands,
        )?,
        OpKind::Iota { .. } => return Err(site.refuse(Capability::UnfoldedIota)),
        imported_ops!()
        | packed_ops!()
        | moe_ops!()
        | matmul_ops!()
        | attention_ops!()
        | scan_ops!()
        | cast_ops!()
        | sampling_ops!()
        | movement_ops!() => {
            // A collective is a cut between stage programs on every target; any other op's lowering
            // reads GPU capabilities.
            let missing = match eqn.op {
                OpKind::AllReduce { .. } | OpKind::AllGather { .. } => {
                    Capability::CollectiveInStage
                }
                _ => Capability::GpuOnly,
            };
            return Err(site.refuse(missing));
        }
    };
    finalize_aie(&planned, g, eqn, target, caps)?;
    Ok(planned)
}

/// A GPU target's per-equation planner body: one thin dispatch table. It runs the shared dtype/storage
/// guards, routes each op family to its owned function and applies the shared launch-shaping postamble
/// (`finalize`).
#[allow(clippy::too_many_arguments)]
fn plan_gpu_eqn(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    views: &HashMap<ValueId, Layout>,
    caps: &DeviceCaps,
    packed_operands: &[bool],
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    if matches!(eqn.op, imported_ops!()) {
        // Card 542a: a block-32 descriptor (the seven GGUF schemes `packed_block_float::
        // PACKED_LOWERING` admits) reaches here through the descriptor-driven lowering table.
        let odt = g.aval(eqn.out).dtype;
        if let Some(planned) = packed_dequant::try_plan(
            g,
            eqn,
            out_shape,
            out_numel,
            backend,
            odt,
            caps,
            limits,
            packed_operands,
        )? {
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
                || g.aval(ids[1]).dtype != DType::I32)
        {
            return Err(site.refuse(Capability::DtypeLowering));
        }
        if matches!(eqn.op, OpKind::ScatterUpdate)
            && (ids.len() != 3
                || g.aval(ids[0]).dtype != DType::E4M3FN
                || g.aval(ids[1]).dtype != DType::E4M3FN
                || g.aval(ids[2]).dtype != DType::I32
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
                    Some(Operand::Value(id)) => g.aval(*id).dtype == DType::I32,
                    _ => false,
                };
            if !valid_storage {
                return Err(site.refuse(Capability::DtypeLowering));
            }
        }
        // AmdGcn has no E4M3FN tensor storage, but the packed-lane row gather reads the table as
        // row-padded u32 words like the BF16 one, so it alone is admitted there.
        let admitted_row_gather = matches!(
            eqn.op,
            OpKind::DenseRowGather {
                source: DType::E4M3FN
            }
        );
        if !matches!(backend, Backend::SpirvVulkan | Backend::Nvptx) && !admitted_row_gather {
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
                // table is the only E4M3FN operand - the index is I32 and the output F32 -
                // and the storage plan gives the table its packed lane (`StorageKind::E4m3Packed`, or
                // refuses it by name), so no dtype rule beyond "this op may read E4M3FN" is needed here.
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
        && out_shape
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .is_none()
    {
        return Err(PlanError::BadShape(format!(
            "I32 output shape {out_shape:?} overflows usize"
        )));
    }
    // spec 135: F16 bridges to Ty::F16 exactly like BF16 bridges to Ty::BF16 (storage is 2 bytes;
    // arithmetic accumulates in f32 in both cases). It must not fall into the `else` and be bridged to
    // Ty::F32 over 2-byte storage, which miscompiles.
    if odt == DType::BF16 || odt == DType::F16 {
        let wired = matches!(
            eqn.op,
            OpKind::MatMul
                | OpKind::ContractionEpilogue { .. }
                | OpKind::Reshape { .. }
                | OpKind::Transpose { .. }
                | OpKind::Slice { .. }
                | OpKind::Broadcast { .. }
                | OpKind::Binary(_)
                | OpKind::Unary(_)
                | OpKind::Select
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
                g,
                eqn,
                out_shape,
                out_numel,
                backend,
                odt,
                caps,
                limits,
                packed_operands,
            );
        }
        moe_ops!() => {
            return moe::plan(
                g,
                eqn,
                out_shape,
                out_numel,
                backend,
                views,
                odt,
                caps,
                limits,
                packed_operands,
            );
        }
        elementwise_ops!() => elementwise::plan(
            g,
            eqn,
            out_shape,
            out_numel,
            Lowering::Gpu { backend, caps },
            views,
            odt,
            limits,
            packed_operands,
        )?,
        matmul_ops!() => matmul::plan(
            g,
            eqn,
            out_shape,
            out_numel,
            backend,
            odt,
            caps,
            limits,
            packed_operands,
        )?,
        attention_ops!() => attention::plan(g, eqn, backend, caps, limits)?,
        scan_ops!() => scan::plan(g, eqn, backend, caps, limits)?,
        cast_ops!() => cast::plan(
            g,
            eqn,
            out_shape,
            out_numel,
            backend,
            caps,
            limits,
            packed_operands.first().copied().unwrap_or(false),
        )?,
        movement_ops!() => movement::plan(
            g, eqn, out_shape, out_numel, backend, views, odt, caps, limits,
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
