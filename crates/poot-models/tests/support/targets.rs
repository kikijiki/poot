//! The reference targets every poot-models planning test compiles for: the plan-summary gate and the
//! feature matrix share them, so a column of the matrix and a checked-in summary file name one target.
#![allow(
    dead_code,
    reason = "shared by two test binaries (plan_summary, feature_matrix); each uses part of it"
)]
use poot_graph_plan::{CompileLimits, CompileOptions, Target};
use poot_target::{AmdArch, Backend};

/// One planner target with the name its matrix column and its plan-summary file carry.
pub(crate) struct RefTarget {
    /// The column heading in the generated feature matrix.
    pub column: &'static str,
    /// The stem of the checked-in `plan_summary/<file>.tsv`.
    pub file: &'static str,
    pub target: Target,
}

/// A reference target over the device caps the test-util preset gives `backend`.
pub(crate) fn preset(column: &'static str, file: &'static str, backend: Backend) -> RefTarget {
    RefTarget {
        column,
        file,
        target: Target::Gpu {
            backend,
            caps: poot_test_util::device_caps::default_caps_for(backend),
        },
    }
}

/// wgpu (SPIR-V for Vulkan), ROCm (AMD GCN, gfx1151) and PTX (NVIDIA).
pub(crate) fn reference_targets() -> Vec<RefTarget> {
    vec![
        preset("wgpu", "spirv_vulkan", Backend::SpirvVulkan),
        preset("ROCm", "amd_gcn", Backend::AmdGcn(AmdArch::gfx1151())),
        preset("PTX", "nvptx", Backend::Nvptx),
    ]
}

/// The compile options every reference compile uses: replayed submission under the standard limits.
pub(crate) fn options() -> CompileOptions {
    CompileOptions::new(CompileLimits::STANDARD)
}
