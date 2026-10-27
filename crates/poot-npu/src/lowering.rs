//! The NPU's [`NpuLowering`]: what `poot-graph-plan` asks of an NPU while it compiles a program, answered
//! from this crate's tables (POOT-1244).

use poot_graph_ir::{EditError, Eqn, Graph};
use poot_graph_plan::{
    Capability, KernelChoice, NpuHandle, NpuLowering, NpuRegionRefusal, Plan, Region,
};
use poot_target::{AieCaps, AieTargetId, VpuCaps};

use crate::vpu_lowering::{lower_eqn, narrow_to_device_precision, plan_vpu_region};
use crate::xdna_lowering::{
    aie_region_recipe, aie_shape_abi, check_shape_abi_limits, validate_region_recipe,
};

/// The lowering of every NPU poot targets: the AMD XDNA region ABI and limit checks, and the Intel NPU's
/// opset table and narrowing pass.
#[derive(Clone, Copy, Debug)]
struct Lowering;

/// The lowering to hand [`poot_graph_plan::CompileOptions::npu`] when compiling for an NPU.
pub fn npu_lowering() -> NpuHandle {
    NpuHandle::new(Lowering)
}

impl NpuLowering for Lowering {
    fn finalize_serialized(&self, choice: &KernelChoice, caps: &AieCaps) -> Result<(), Capability> {
        let abi = aie_shape_abi(choice).map_err(Capability::AieAbi)?;
        check_shape_abi_limits(&abi, caps).map_err(Capability::AieLimit)
    }

    fn region_fits(
        &self,
        graph: &Graph,
        plans: &[Plan],
        region: &Region,
        target: AieTargetId,
        caps: &AieCaps,
    ) -> Result<(), Capability> {
        let recipe = aie_region_recipe(graph, plans, region, target).map_err(Capability::AieAbi)?;
        validate_region_recipe(&recipe, caps).map_err(Capability::AieLimit)
    }

    fn narrow(&self, graph: &Graph, caps: &VpuCaps) -> Result<Graph, EditError> {
        narrow_to_device_precision(graph, caps)
    }

    fn supports_eqn(&self, graph: &Graph, eqn: &Eqn, caps: &VpuCaps) -> Result<(), Capability> {
        lower_eqn(graph, eqn, caps).map(|_| ())
    }

    fn check_region(
        &self,
        graph: &Graph,
        region: &Region,
        caps: &VpuCaps,
    ) -> Result<(), NpuRegionRefusal> {
        plan_vpu_region(graph, region, caps).map(|_| ())
    }
}
