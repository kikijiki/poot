//! Planner and GPU-target fixtures with no production caller, for the planner, executor and device tests
//! across the workspace (card 671: dead under the strict dead-pub rule, a test reference is never a use).
//!
//! [`plan_eqn`] is [`crate::plan_eqn`] under [`roomy_body_limits`].
//!
//! ADR-0113: `is_decode_gemv`/`is_tiled_gemm` had no production caller at all (production routes through
//! this crate's private `_in` twins) and were deleted; their cross-crate callers assert on the `Plan`
//! [`plan_eqn`] returns instead of calling a test-only predicate.
//! `per_eqn_plans_outside_compile` reads this crate's private state (a crate-private thread-local
//! counter) and stays beside it under
//! `#[cfg(any(test, feature = "test-support"))]` (ADR-0113 point 2); their consumers call them directly
//! through that feature, not through a forwarder here (a forwarder whose only purpose is to be a non-test
//! reference is laundering, ADR-0113's Context).

use poot_graph_ir::{Eqn, Graph};
use poot_target::{Backend, DeviceCaps};

use crate::{Plan, PlanError, Target};

/// Kernel-body limits no fixture comes near, for a test that plans a kernel and is not about the limit.
pub fn roomy_body_limits() -> poot_kernelgen::BodyLimits {
    poot_kernelgen::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    }
}

/// Choose how to execute an eqn at `world_size = 1`: `poot-graph-plan`'s `plan_eqn` under
/// [`roomy_body_limits`].
pub fn plan_eqn(
    g: &Graph,
    eqn: &Eqn,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<Plan, PlanError> {
    crate::plan_eqn(g, eqn, backend, caps, &roomy_body_limits())
}

// --- GPU-target accessors for tests --------------------------------------------------------------------------
//
// A test that builds a `Target::Gpu` fixture reads its backend and capabilities directly: it knows which arm
// it constructed. Production consumers match the enum, and the planner refuses a GPU-only lowering on an AIE
// target with `Capability::GpuOnly`; a test that reads these off an AIE target has a bug, so these panic.

/// The GPU capabilities of the GPU target fixture `target`.
pub fn gpu_caps(target: &Target) -> DeviceCaps {
    match target {
        Target::Gpu { caps, .. } => *caps,
        Target::Aie { .. } | Target::Vpu { .. } => {
            panic!("{target:?} is an NPU target: it has no GPU DeviceCaps")
        }
    }
}

/// The GPU capabilities of the GPU target fixture `target`, for a test that shrinks one limit.
pub fn gpu_caps_mut(target: &mut Target) -> &mut DeviceCaps {
    match target {
        Target::Gpu { caps, .. } => caps,
        Target::Aie { .. } | Target::Vpu { .. } => {
            panic!("an NPU target has no GPU DeviceCaps")
        }
    }
}

/// The GPU backend of the GPU target fixture `target`.
pub fn gpu_backend(target: &Target) -> Backend {
    match target {
        Target::Gpu { backend, .. } => *backend,
        Target::Aie { .. } | Target::Vpu { .. } => {
            panic!("{target:?} is an NPU target: it has no GPU backend")
        }
    }
}

/// The graph `compile` hands the planner for `g` on the GPU target `backend` with `caps`: the one preparation
/// pipeline every target runs, for tests that assert on the prepared graph.
pub fn compiled_graph(
    g: &Graph,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<Graph, crate::CompileError> {
    let target = Target::Gpu {
        backend,
        caps: *caps,
    };
    let options = crate::CompileOptions::new(crate::CompileLimits::STANDARD);
    Ok(crate::compile(g, &target, &options)?.graph().clone())
}
