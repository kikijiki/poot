//! Backend-neutral execution planning. [`compile`] is the one graph-to-program step (card 532): it runs the
//! graph passes for a [`Target`] and then `plan_graph`, returning a [`Program`] or a typed error; the wgpu
//! cached decode reaches the planner only through it, and the remaining per-equation callers move onto it
//! in Cards 535a and 535b.
//!
//! Given a graph-IR equation and its static output shape,
//! `plan_eqn` decides how to run it: synthesize a kernelgen [`Body`] (+ a cache key), run it on the host
//! (the rare layout-op fallback), or alias an input buffer (reshape). The wgpu, PTX, ROCm, and restricted
//! raw-Vulkan GPU executors consume this planner with `Backend::SpirvVulkan`, `Backend::Nvptx`, or
//! `Backend::AmdGcn`; target compilation and recorder dispatch remain runtime-specific. An AIE target
//! (`Target::Aie`, carrying `AieCaps`, never GPU caps) plans the same requests under a sequential one-core schedule instead of a thread grid
//! (POOT-664); a plan that is not that schedule is refused with `Capability::AieSchedule`. The generated
//! `Body` is named `"k"`, the
//! shared entry point used by the GPU emitters.
//!
//! # The graph passes are sealed behind `compile` (card 626, SC-001)
//!
//! [`compile`] is the one caller of every graph-to-graph transform (CSE, DCE, fusion, the tiling
//! retags): `passes` is a private module, so nothing outside this crate can name a pass or sequence
//! its own pipeline, and the pre-card-626 home (`poot_graph_ir::transform`) no longer exists at all.
//! The `compile_fail` doctests that prove it live in `poot-llm`'s crate docs (SC-001), a consumer
//! crate, which is where a bypass would have to be written.

/// Card 375b device witness admission: the structural classifier behind `plan_device_validation`.
pub mod device_validation;

/// Card 235: the widen-mismatched-matmul-operand-dtype graph transform + the tensor-core eligibility
/// predicate it shares with `plan_eqn`'s `OpKind::MatMul` arm (single source of truth).
pub mod dtype_widen;

/// Card 360: packed multi-device topology and communication planner (planner half only - see
/// `specs/360-packed-multi-device-topology-capture/spec.md`). Model-neutral device topology, placement,
/// communication and accounting, proved against a model-free virtual topology. The captured coordinator in
/// `poot-executor` runs the rows `compile_staged` plans from it.
pub mod multi_device;

/// Pure Card 356 contracts for table-driven block-float planning. Kernel import, graph recognition, and
/// backend upload consume these contracts but remain separate integration layers.
pub mod packed_block_float;

/// Native capture state-dependency analysis and simultaneous commit planning (card 325).
pub mod state_commit;

/// The staged program an executor loads: one or more [`Program`]s, each on one device (card 546a).
pub mod staged;

/// The one kernel-authoring route and asset manifest (card 559): `ImportedKernel::X.body()` for every
/// kernel authored in Rust and shipped as a committed asset, plus the row-major/broadcast/index-remap
/// geometry helpers that build the launch metadata buffers these kernels read their dims from.
mod imported;

/// The strided-view promotion pass (`compute_views`, spec 132) and the dispatch predicates + launch-grid
/// math (`is_decode_gemv`, `decode_gemv_plan`, and friends) that `plan_eqn_views` and its callers share.
mod predicates;

/// The small pub types every caller matches on: [`Backend`], [`Plan`], [`PlanError`], [`CollectiveKind`],
/// [`Dispatch`].
mod types;

/// Buffer-plan lifetimes and arena slots: allocation follows the buffer plan, not one buffer per
/// value (Card 547b).
mod buffer_plan;
mod compile;
mod graph_validation;
mod kernel_mapping;
/// Graph-to-graph transforms (card 626): private, so [`compile`] is their only caller anywhere outside
/// this crate. `packed_bind`'s weight-binding API is the one carve-out, re-exported `pub` below (it is
/// checkpoint-loading machinery, not a `compile` pass).
mod passes;
mod planner;
mod refusal;
mod storage_analysis;

pub use buffer_plan::{ArenaSlotId, BufferPlan};
pub use compile::{
    ArtifactGranularity, CastRangeError, CompileError, CompileLimits, CompileOptions,
    ExpansionError, ExpertIdsError, ExpertIdsGap, GraphResource, LoraError, LoraPlan, Program,
    Region, SlotSchema, Target, compile,
};
pub use imported::ImportedKernel;
#[cfg(any(test, feature = "test-support"))]
pub use imported::{AssetDest, AssetEntry, MANIFEST};
pub use kernel_mapping::{AieAbiRefusal, AieLimitRefusal, aie_shape_abi, validate_linked_artifact};
pub use passes::{
    AttentionDecline, BoundaryDescriptor, ExpertStackRefusal, FusionDecline, FusionDeclineReason,
    LegalizeError, PackedBindError, PackedConst, PackedDequantErrorContext,
    PackedDequantProductionError, PackedLayout, PoolError, PoolSite, PoolSpec, PooledWeight,
    RopeDecline, ShardError, ShardRule, ShardedGraph, StageSplitError, WeightFormats,
    WeightFormatsError, bind_packed_weights, recognize_packed_contractions,
    recognize_packed_row_gathers, routed_step_demand, shard,
};
// Card 626 (SC-001): `compile` alone runs the graph passes in production - `passes` is a private module, so a
// default build has no way to name or sequence one. Gated here (ADR-0113 point 3: no forwarder, consumers
// call the real thing through this feature edge) because pass-soundness and plan-shape tests across the
// workspace drive an individual pass directly.
pub use graph_validation::{ValidationPacketPlan, ValidationPacketSource};
#[cfg(any(test, feature = "test-support"))]
pub use passes::{
    RoutedExpertSite, canonicalize, collapse_reshape_chains, cse, dce, elide_noop_transposes,
    flash_attention_capped, fold_iota, fuse, legalize, passes_without_target, rope_fusion,
    routed_expert_sites,
};
pub use planner::{KernelChoice, NoDispatch, Planned, region_member_work};
#[cfg(any(test, feature = "test-support"))]
pub use planner::{plan_eqn, plan_eqn_choice, plan_eqn_views, vpu_refusals};
pub use predicates::{compute_views, numel, value_ids};
pub use refusal::{Capability, FusedStepGap, Refusal, RefusalDtypes, StorageGap};
pub use types::{
    Binding, CollectiveKind, Dispatch, DispatchPlan, DispatchPlanError, GraphStorage, Launch, Plan,
    PlanError, PlanKind, ScratchDecl, ScratchId, Sink, Source, ValueStorage,
};

#[cfg(test)]
mod tests;

/// CPU-oracle workgroup-interp tests for the coalesced decode-GEMV bodies (tile-ragged N/K, batched).
#[cfg(test)]
mod tests_gemv_coalesced;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
#[cfg(test)]
mod unit_helpers;
