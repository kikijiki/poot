//! The reference targets every poot-models planning test compiles for: the plan-summary gate and the
//! feature matrix share them, so a column of the matrix and the checked-in digests name one target.
#![allow(
    dead_code,
    reason = "shared by two test binaries (plan_summary, feature_matrix); each uses part of it"
)]
use poot_graph_plan::{CompileLimits, CompileOptions, Target};
use poot_target::{AmdArch, Backend};

/// One planner target with the name its matrix column and its plan digests carry.
pub(crate) struct RefTarget {
    /// The column heading in the generated feature matrix.
    pub column: &'static str,
    /// The target's id in the checked-in plan digests.
    pub id: &'static str,
    pub target: Target,
}

/// A reference target over the device caps the test-util preset gives `backend`.
pub(crate) fn preset(column: &'static str, id: &'static str, backend: Backend) -> RefTarget {
    RefTarget {
        column,
        id,
        target: Target::Gpu {
            backend,
            caps: poot_test_util::device_caps::default_caps_for(backend),
        },
    }
}

/// raw Vulkan: SPIR-V for the `poot-vulkan-runtime` device, over the caps that runtime measured on this
/// box's RADV iGPU, not wgpu's. The two lower to the same SPIR-V and plan differently because their devices
/// differ (workgroup memory, workgroup and grid limits).
pub(crate) fn vulkan_target() -> RefTarget {
    RefTarget {
        column: "Vulkan",
        id: "vulkan",
        target: Target::Gpu {
            backend: Backend::SpirvVulkan,
            caps: poot_test_util::device_caps::vulkan_rdna3_igpu(),
        },
    }
}

/// wgpu (SPIR-V for Vulkan), raw Vulkan (SPIR-V, its own device caps), ROCm (AMD GCN, gfx1151) and PTX
/// (NVIDIA).
pub(crate) fn reference_targets() -> Vec<RefTarget> {
    vec![
        preset("wgpu", "spirv_vulkan", Backend::SpirvVulkan),
        vulkan_target(),
        preset("ROCm", "amd_gcn", Backend::AmdGcn(AmdArch::gfx1151())),
        preset("PTX", "nvptx", Backend::Nvptx),
    ]
}

/// The compile options every reference compile uses: replayed submission under the standard limits.
pub(crate) fn options() -> CompileOptions {
    CompileOptions::new(CompileLimits::STANDARD)
}
