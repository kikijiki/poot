//! The reference targets every poot-models planning test compiles for: the plan-summary gate and the
//! supported-models tables share them, so a column of the tables and the checked-in digests name one target.
#![allow(
    dead_code,
    reason = "shared by two test binaries (plan_summary, supported_models); each uses part of it"
)]
use poot_graph_plan::multi_device::topology::DeviceId;
use poot_graph_plan::staged::{
    DevicePlacement, ExpertPlacement, Partition, StagedCompileError, StagedProgram, TargetSet,
    compile_staged,
};
use poot_graph_plan::{CompileError, CompileLimits, CompileOptions, Target};
use poot_target::{AmdArch, Backend};

/// One planner target with the name its matrix column and its plan digests carry.
pub(crate) struct RefTarget {
    /// The column heading in the generated supported-models tables.
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

/// `graph` compiled for `target` on one device through the production entry, which hosts every gather-only
/// table before the stages are compiled: what a model's plan on the target is, hosted tables included. A
/// refusal is the stage compile's typed [`CompileError`]; any other staged failure is a defect of the compiler,
/// never of the graph, and panics.
pub(crate) fn compile_production(
    graph: &poot_graph_ir::Graph,
    target: &Target,
    options: &CompileOptions,
) -> Result<StagedProgram, CompileError> {
    compile_staged(
        graph,
        &TargetSet::single(DeviceId(0), *target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        options,
    )
    .map_err(|error| match error {
        StagedCompileError::Compile(error) => error,
        other => panic!("the staged compile failed outside a stage compile: {other}"),
    })
}

/// The compile options every reference compile uses: replayed submission under the standard limits and the exact
/// lowering, whose plan shapes the digests and the literal launch counts record (the narrow default has its own
/// tier-3 gates).
pub(crate) fn options() -> CompileOptions {
    CompileOptions::new(CompileLimits::STANDARD)
        .with_matmul_precision(poot_graph_ir::MatmulPrecision::Exact)
}
