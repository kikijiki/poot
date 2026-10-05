//! Backend-neutral execution planning. [`compile`] is the one graph-to-program step (card 532): it runs the
//! graph passes for a [`Target`] and then `plan_graph`, returning a [`Program`] or a typed error; the wgpu
//! cached decode reaches the planner only through it, and the remaining per-equation callers move onto it
//! in Cards 535a and 535b.
//!
//! Given a graph-IR equation and its static output shape,
//! `plan_eqn` decides how to run it: synthesize a kernelgen [`Body`] (+ a cache key), run it on the host
//! (the rare layout-op fallback), or alias an input buffer (reshape). The wgpu, PTX, ROCm, and restricted
//! raw-Vulkan GPU executors consume this planner with `Backend::SpirvVulkan`, `Backend::Nvptx`, or
//! `Backend::AmdGcn`; target compilation and recorder dispatch remain runtime-specific. NPU graph lowering
//! is a separate direct path and does not claim a planner backend. The generated `Body` is named `"k"`, the
//! shared entry point used by the GPU emitters.
//!
//! # The graph passes are sealed behind `compile` (card 626, SC-001)
//!
//! [`compile`] is the one caller of every graph-to-graph transform (CSE, DCE, fusion, the tiling
//! retags): `passes` is a private module, so nothing outside this crate can name a pass or sequence
//! its own pipeline, and the pre-card-626 home (`poot_graph_ir::transform`) no longer exists at all.
//! The `compile_fail` doctests that prove it live in `poot-llm`'s crate docs (SC-001), a consumer
//! crate, which is where a bypass would have to be written.

use poot_tensor::DType;
use std::collections::HashMap;

use poot_graph_ir::op::{
    BinOp as GBinOp, FusedOp, FusedOperand, OpKind, RedOp, RowStep, UnOp as GUnOp,
};

#[cfg(test)]
use poot_graph_ir::ValidationOutputs;
use poot_graph_ir::{
    Eqn, Graph, NoValidations, Operand, Scalar, ValidationChannel, ValidationPacketLayout, ValueId,
};

use poot_kernel_ir::{BinOp, Body, MathOp, Ty, UnOp};

use poot_kernelgen as kg;
use poot_kernelgen::{
    AttentionSpec, CastSpec, ContractionSpec, E4m3Movement, ElementwiseSpec, I32Binary,
    KernelRequest, MatmulShapes, MovementSpec, PackedRequest, PointwiseForm, PointwiseSpec,
    RopeSpec, RowSpec, TopKSpec, UnaryOp, ValueBinary, ViewOperand, WeightLayout,
};

// `Layout` (strided views, spec 132) is defined in poot-kernelgen, the lower crate the generated kernel
// bodies read through, and re-exported here for callers that depend only on poot-graph-plan (poot-gpu,
// tests). See `compute_views` for how a value earns a non-contiguous one.
pub use poot_kernelgen::Layout;

/// Card 375b device witness admission: the structural classifier behind `plan_device_validation`.
pub mod device_validation;

/// Card 235: the widen-mismatched-matmul-operand-dtype graph transform + the tensor-core eligibility
/// predicate it shares with `plan_eqn`'s `OpKind::MatMul`/`MatMulBias` arms (single source of truth).
pub mod dtype_widen;

/// Card 360: packed multi-device topology and communication planner (planner half only - see
/// `specs/360-packed-multi-device-topology-capture/spec.md`). Model-neutral device topology, placement,
/// communication, accounting, and replay-contract identity, proved against a model-free virtual topology
/// and a fake runtime. Card 408 owns the device-resident captured executor.
pub mod multi_device;

/// Pure Card 356 contracts for table-driven block-float planning. Kernel import, graph recognition, and
/// backend upload consume these contracts but remain separate integration layers.
pub mod packed_block_float;

/// Graph-level production planning for atomic multi-equation replacements.
pub mod production_plan;

/// Native capture state-dependency analysis and simultaneous commit planning (card 325).
pub mod state_commit;

/// The staged program an executor loads: one or more [`Program`]s, each on one device (card 546a).
pub mod staged;

/// The one kernel-authoring route and asset manifest (card 559): `ImportedKernel::X.body()` for every
/// kernel authored in Rust and shipped as a committed asset, plus the row-major/broadcast/index-remap
/// geometry helpers that build the `Plan::ComputeMeta` buffers these kernels read their dims from.
mod imported;

/// The strided-view promotion pass (`compute_views`, spec 132) and the dispatch predicates + launch-grid
/// math (`is_decode_gemv`, `decode_gemv_plan`, and friends) that `plan_eqn_views` and its callers share.
mod predicates;

/// The small pub types every caller matches on: [`Backend`], [`Plan`], [`PlanError`], [`CollectiveKind`],
/// [`ComputeChunk`].
mod types;

/// Buffer-plan lifetimes and arena slots: allocation follows the buffer plan, not one buffer per
/// value (Card 547b).
mod buffer_plan;

pub use buffer_plan::{ArenaSlotId, BufferPlan};
// `compile.rs`'s pipeline is this function's only caller (Card 547b, mirroring `dtype_widen`'s
// `pub(crate)` passes above): no crate outside `poot-graph-plan` plans buffer lifetimes itself.
pub(crate) use buffer_plan::plan_buffers;
pub use device_validation::{DEVICE_WITNESS_MAGNITUDE_BOUND, DeviceWitnessRejection};

pub use dtype_widen::{
    bf16_const_feeds_decode_bf16_gemv, bf16_const_feeds_only_packed_readers,
    matmul_amd_tc_eligible, matmul_bf16_decode_gemv_eligible, matmul_nvptx_tc_eligible,
    matmul_spirv_coopmat_eligible, matmul_tensor_core_eligible, prepare_target_graph,
    widen_mismatched_matmul_dtypes,
};
// `compile.rs`'s pipeline runs these directly (Card 534a): `pub(crate)`, not `pub`,
// so no crate outside `poot-graph-plan` - `poot-gpu`/`poot-rocm-gpu`/`poot-ptx-gpu` included - can name
// or sequence them itself (SC-001). `prepare_target_graph`/`widen_mismatched_matmul_dtypes` above are the only
// sanctioned entry points.
pub(crate) use dtype_widen::{
    fold_dense_bf16_row_gathers, fold_dense_contractions, lower_nonlast_reduces,
};

pub use packed_block_float::*;

pub use predicates::*;

pub use production_plan::*;

pub use staged::*;
pub use state_commit::*;

pub use types::*;

// `pootc` (a dev-dependency, card 559) reads `MANIFEST` to prove every committed asset matches its
// manifest entry and regenerates byte-identical to its kernel source. `ImportedKernel` is `pub` too:
// the `probe` family's kernels are each consumed by a different crate (`poot-rocm-gpu`'s `Device::copy`
// and executor-contract fixtures, `poot-gpu`'s importer-capability probes), through this one accessor.
pub use imported::{AssetDest, AssetEntry, ImportedKernel, MANIFEST};
// The `Plan::ComputeMeta` geometry helpers stay `pub(crate)`; this brings them into the crate-root
// namespace so every planner/predicates call site's `use crate::*` sees them.
pub(crate) use imported::{
    broadcast_eff_strides, concat2_meta, dus_meta, index_remap_meta, row_major_strides,
};

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

pub use compile::*;
pub use graph_validation::*;
pub(crate) use kernel_mapping::*;
pub use passes::{
    BoundaryDescriptor, PackedBindError, PackedConst, PackedLayout, WeightFormats,
    WeightFormatsError, bind_packed_weights,
};
// `poot-llm`'s CPU oracle (`core/cpu_oracle.rs`) claims a traced graph's packed contractions/row
// gathers the same way `compile` does, so its eval matches what the device plans - a real production
// cross-crate caller, not a test. `reject_packed_dequant_escapes` has no such caller (only `compile`'s
// own escape gate), so it stays sealed below.
pub use passes::{recognize_packed_contractions, recognize_packed_row_gathers};
// Card 626 (SC-001): `compile` alone runs these in production - `passes` is a private module, so a
// default build has no way to name or sequence one (`poot_graph_plan::passes::cse` does not resolve at
// all). Gated here (not sealed with the rest, ADR-0113 point 3: no forwarder, consumers call the real
// thing through this feature edge) because pass-soundness and plan-shape tests across the workspace
// drive an individual pass directly - this crate's own `tests/*.rs` and every crate that already
// enables `poot-test-util/graph-fixtures` (which carries `poot-graph-plan/test-support` back here) see
// these; a default build does not.
#[cfg(any(test, feature = "test-support"))]
pub use passes::{
    FUSABLE_FLOAT_UNARY_OPS, LegalizeError, PackedDequantErrorContext,
    PackedDequantProductionError, canonicalize, canonicalize_with_declines,
    collapse_reshape_chains, cse, dce, dce_with_roots, decompose_large_vocab_greedy,
    elide_noop_transposes, flash_attention_capped, fold_iota, fuse, fuse_bias_epilogues, legalize,
    passes_without_target, reject_packed_dequant_escapes, rope_fusion,
};
#[cfg(test)]
pub(crate) use passes::{StageAssignment, split_stages, split_stages_after_layers};
pub use planner::*;
pub use refusal::*;
pub use storage_analysis::*;

#[cfg(test)]
mod tests;

/// CPU-oracle workgroup-interp tests for the coalesced decode-GEMV bodies (tile-ragged N/K, batched).
#[cfg(test)]
mod tests_gemv_coalesced;

#[cfg(test)]
mod test_support;
