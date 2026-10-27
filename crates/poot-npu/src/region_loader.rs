//! The NPU's [`RegionLoader`]s: what compiles a program's regions for an NPU device (POOT-1244).
//!
//! The engine is one engine over every backend and knows no vendor. For a region program it asks the device's
//! loader for the region's kernel and launch: the AMD loader links the region's tile bodies against the
//! recipe the plans derive (Peano and IRON, through [`compile_aie`], behind the kernel cache) and checks the
//! linked artifact against the part's limits; the Intel loader serializes the planner's region to OpenVINO IR
//! and builds the artifact the vendor graph compiler takes. Both refuse with a typed cause and load nothing.

use std::num::{NonZeroU32, NonZeroU64};

use crate::{VpuEmitError, emit_vpu};
use crate::{aie_kernel_handle, vpu_kernel_handle};
use poot_codegen::{KernelCache, Target as CodegenTarget, Toolchain};
use poot_executor::{CompiledRegion, LaunchKind, RegionLoadError, RegionLoader};
use poot_graph_plan::{
    AieAbiRefusal, NpuHandle, NpuRegionRefusal, NpuTarget, Program, Region, Target,
};
use poot_npu_artifacts::{
    AieInvocation, AieLimitRefusal, VpuArtifactError, VpuBinding, VpuGraphArtifact, VpuInvocation,
    VpuLayer,
};
use poot_target::{AieCaps, AieTargetId, VpuTargetId};

use crate::aie_compile::{AieCompileError, compile_aie};
use crate::lowering::npu_lowering;
use crate::vpu_lowering::plan_vpu_region;
use crate::xdna_lowering::{region_link, validate_linked_artifact};

/// Why an AMD region could not be compiled for the device.
#[derive(Debug, thiserror::Error)]
pub enum AieLoadError {
    /// The region's plans have no link recipe to link against (POOT-664, POOT-1160).
    #[error(transparent)]
    Abi(#[from] AieAbiRefusal),
    /// A toolchain stage, the body/ABI check or the artifact validator refused the link.
    #[error(transparent)]
    Compile(#[from] AieCompileError),
    /// The linked artifact does not fit the part's measured limits (an unmeasured limit refuses).
    #[error(transparent)]
    Limit(#[from] AieLimitRefusal),
    /// A linked artifact's instruction stream is empty or longer than `u32::MAX` words, which the artifact
    /// validator and the artifact limit rule out: an invariant breach, named.
    #[error("the AIE instruction stream is {bytes} bytes")]
    InstructionStream {
        /// The stream's length in bytes.
        bytes: usize,
    },
}

/// Why an Intel NPU region could not be compiled for the device.
#[derive(Debug, thiserror::Error)]
pub enum VpuLoadError {
    /// The program was not compiled for an Intel NPU, so it has no graph region.
    #[error("the program was not compiled for an Intel NPU")]
    NotAnIntelProgram,
    /// The program's graph does not assemble into a region (compile admitted it, so this is a program built
    /// for another lowering).
    #[error(transparent)]
    Region(#[from] NpuRegionRefusal),
    /// The region could not be emitted as OpenVINO IR: the region the planner assembled is not well formed.
    /// An invariant breach in the lowering, named.
    #[error(transparent)]
    Emit(#[from] VpuEmitError),
    /// The region's argument ABI is not one the NPU device can bind.
    #[error(transparent)]
    Artifact(#[from] VpuArtifactError),
}

/// Links an AMD XDNA device's regions: the toolchain, the kernel cache and the limits of the part.
#[derive(Debug)]
pub struct AieRegionLoader {
    target: AieTargetId,
    caps: AieCaps,
    toolchain: Toolchain,
    cache: KernelCache,
}

impl AieRegionLoader {
    /// The loader of the part `caps` describes, with the toolchain the process environment names and the
    /// shared kernel cache.
    pub fn new(caps: AieCaps) -> Self {
        Self::with_toolchain(
            caps,
            Toolchain::from_env(),
            KernelCache::open(CodegenTarget::AieCore),
        )
    }

    /// The loader of the part `caps` describes over an explicit `toolchain` and `cache`.
    pub fn with_toolchain(caps: AieCaps, toolchain: Toolchain, cache: KernelCache) -> Self {
        Self {
            target: caps.target,
            caps,
            toolchain,
            cache,
        }
    }

    /// The region's link recipe and tile bodies are read from the program's own plans (`compile` already
    /// admitted the recipe against the part's limits when it packaged the region), the bodies are linked
    /// against it into one artifact, the linked artifact is checked against the limits before it is wrapped
    /// (what only a linked artifact can answer), and only then is it handed to the device.
    fn compile(
        &self,
        program: &Program,
        region: &Region,
        max_artifact_bytes: NonZeroU64,
    ) -> Result<CompiledRegion, AieLoadError> {
        let (recipe, bodies) = region_link(program, region)?;
        // The recipe holds the wiring and shapes, the bodies hold the arithmetic: both name the region.
        let identity = {
            let mut text = recipe.canonical();
            for body in &bodies {
                text.push_str(&format!("{}\n", self.cache.key(body).id()));
            }
            text
        };
        let compiled = compile_aie(
            &bodies,
            &recipe,
            &self.toolchain,
            &self.cache,
            max_artifact_bytes,
        )?;
        validate_linked_artifact(&compiled.artifact, &self.caps)?;
        let stream_bytes = compiled.artifact.instructions().len();
        let words = u32::try_from(stream_bytes / 4)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(AieLoadError::InstructionStream {
                bytes: stream_bytes,
            })?;
        Ok(CompiledRegion {
            identity,
            kernel: aie_kernel_handle(compiled.artifact),
            launch: LaunchKind::Aie(AieInvocation::new(self.target, words)),
        })
    }
}

impl RegionLoader for AieRegionLoader {
    fn lowering(&self) -> NpuHandle {
        npu_lowering()
    }

    fn compile_region(
        &self,
        program: &Program,
        region: &Region,
        max_artifact_bytes: NonZeroU64,
    ) -> Result<CompiledRegion, RegionLoadError> {
        Ok(self.compile(program, region, max_artifact_bytes)?)
    }
}

/// Builds an Intel NPU device's graph artifacts.
#[derive(Debug)]
pub struct VpuRegionLoader {
    target: VpuTargetId,
}

impl VpuRegionLoader {
    /// The loader of the Intel NPU part `target`.
    pub const fn new(target: VpuTargetId) -> Self {
        Self { target }
    }

    /// The planner's region is serialized to OpenVINO IR ([`emit_vpu`]) and handed to the device with the
    /// region's argument ABI (the program's own region reads, then writes, with the storage the buffer plan
    /// gave each value). The device compiles it with the vendor compiler and checks the compiled graph's
    /// arguments against that ABI.
    fn compile(&self, program: &Program, region: &Region) -> Result<CompiledRegion, VpuLoadError> {
        let Target::Npu(NpuTarget::Vpu { caps, .. }) = program.target() else {
            return Err(VpuLoadError::NotAnIntelProgram);
        };
        let graph = plan_vpu_region(program.graph(), region, caps)?;
        let ir = emit_vpu(&graph)?;
        let binding = |name: String, value, shape: Vec<usize>| {
            VpuBinding::new(
                name,
                program.storage().storage(value).buffer_storage(),
                shape,
            )
        };
        let inputs = region
            .reads()
            .iter()
            .zip(graph.input_names())
            .zip(graph.input_shapes())
            .map(|((&value, name), shape)| binding(name, value, shape))
            .collect();
        let outputs = region
            .writes()
            .iter()
            .zip(graph.output_names())
            .zip(graph.output_shapes())
            .map(|((&value, name), shape)| binding(name, value, shape))
            .collect();
        let layers = graph
            .layers()
            .map(|(name, origin)| VpuLayer::new(name, origin))
            .collect();
        let artifact = VpuGraphArtifact::new(
            self.target,
            ir.xml.into_bytes(),
            ir.weights,
            inputs,
            outputs,
            layers,
        )?;
        let digest = artifact.digest();
        Ok(CompiledRegion {
            identity: format!("{digest:016x}"),
            kernel: vpu_kernel_handle(artifact),
            launch: LaunchKind::VpuGraph(VpuInvocation::new(self.target, digest)),
        })
    }
}

impl RegionLoader for VpuRegionLoader {
    fn lowering(&self) -> NpuHandle {
        npu_lowering()
    }

    fn compile_region(
        &self,
        program: &Program,
        region: &Region,
        _max_artifact_bytes: NonZeroU64,
    ) -> Result<CompiledRegion, RegionLoadError> {
        Ok(self.compile(program, region)?)
    }
}
