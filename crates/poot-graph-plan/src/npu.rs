//! The NPU's side of compilation, behind one capability object (POOT-1244, POOT-ADR-0100).
//!
//! The planner is the foundation every device shares. What only an NPU needs (the Intel vendor graph's opset
//! table and its narrowing pass, an AMD region's link recipe, shape ABI and limit checks) is the NPU crate's
//! (`poot-npu`): it implements [`NpuLowering`] and the caller hands it to [`compile`](crate::compile) in
//! [`CompileOptions::npu`](crate::CompileOptions::npu). A [`Target`](crate::Target) stays plain,
//! comparable data ([`NpuTarget`]); the object that decides is an explicit argument of the compile, never
//! hidden in the target.
//!
//! The AIE sequential schedule itself (the serialized request builders) stays in the planner: it is the
//! ordinary families planned under a one-core schedule. Only its ABI and limit finalization is asked of the
//! lowering.

use std::sync::Arc;

use poot_graph_ir::{EditError, Eqn, Graph, ValueId};
use poot_npu_artifacts::{AieAbiMismatch, AieArtifactError};
use poot_target::{AieCaps, AieTargetId, TargetId, VpuCaps, VpuTargetId};
use poot_tensor::DType;

use crate::{Capability, KernelChoice, Plan, Region, StorageGap};

/// An NPU a graph is compiled for, and the limits its runtime reports: plain data, like the GPU target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NpuTarget {
    /// An AMD XDNA part: its artifacts are linked regions of planned equations.
    Aie {
        /// The XDNA part the program is generated for.
        target: AieTargetId,
        /// The limits the part's runtime reports.
        caps: AieCaps,
    },
    /// An Intel NPU part: the vendor graph compiler compiles the whole graph as one artifact.
    Vpu {
        /// The Intel NPU part the program is generated for.
        target: VpuTargetId,
        /// The capabilities the part's runtime reports.
        caps: VpuCaps,
    },
}

impl NpuTarget {
    /// This target's identity, without its limits.
    pub const fn id(&self) -> TargetId {
        match self {
            NpuTarget::Aie { target, .. } => TargetId::Aie(*target),
            NpuTarget::Vpu { target, .. } => TargetId::Vpu(*target),
        }
    }
}

/// What the planner asks of an NPU's lowering. Every method answers for one vendor's targets; the planner
/// calls the ones its [`NpuTarget`] names.
pub trait NpuLowering: std::fmt::Debug + Send + Sync {
    /// AMD: finalize one equation's serialized sequential schedule: it must have a shape ABI, and the ABI
    /// must fit `caps` (an unmeasured limit refuses).
    ///
    /// # Errors
    ///
    /// The [`Capability`] the target lacks.
    fn finalize_serialized(&self, choice: &KernelChoice, caps: &AieCaps) -> Result<(), Capability>;

    /// AMD: whether `region` of `plans` is one artifact `caps` can link: the one definition of "a region
    /// fits", which packaging grows a region against and the loader enforces again on the linked artifact.
    ///
    /// # Errors
    ///
    /// The [`Capability`] the target lacks.
    fn region_fits(
        &self,
        graph: &Graph,
        plans: &[Plan],
        region: &Region,
        target: AieTargetId,
        caps: &AieCaps,
    ) -> Result<(), Capability>;

    /// Intel: the graph transform that makes the narrowing of F32 values to the device's arithmetic width
    /// explicit, run first so every later pass and the planner see the graph the device runs.
    ///
    /// # Errors
    ///
    /// [`EditError`] when the narrowed graph is not a valid graph.
    fn narrow(&self, graph: &Graph, caps: &VpuCaps) -> Result<Graph, EditError>;

    /// Intel: whether the lowering table has an opset op for `eqn`.
    ///
    /// # Errors
    ///
    /// The [`Capability`] the target lacks for the equation.
    fn supports_eqn(&self, graph: &Graph, eqn: &Eqn, caps: &VpuCaps) -> Result<(), Capability>;

    /// Intel: whether the graph assembles into one region of lowered equations. `region` is the program's one
    /// region, whose argument and result lists the artifact's ABI shares; the device's loader assembles the
    /// same region again to build the artifact.
    ///
    /// # Errors
    ///
    /// [`NpuRegionRefusal`] naming the first equation the table refuses, or the region argument the device
    /// does not carry.
    fn check_region(
        &self,
        graph: &Graph,
        region: &Region,
        caps: &VpuCaps,
    ) -> Result<(), NpuRegionRefusal>;
}

/// Why a graph does not assemble into one NPU region.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum NpuRegionRefusal {
    /// The table has no lowering for the equation that defines `eqn`.
    #[error("the equation defining {eqn} has no NPU lowering: {missing}")]
    Eqn {
        /// The value the equation defines.
        eqn: ValueId,
        /// What the target lacks for it.
        missing: Capability,
    },
    /// A region argument has a dtype the device does not carry.
    #[error("{value} ({dtype}) cannot be a region argument: {gap}")]
    Argument {
        /// The value the region reads.
        value: ValueId,
        /// Its dtype.
        dtype: DType,
        /// What about the dtype the device cannot carry.
        gap: StorageGap,
    },
}

/// The lowering a compile hands to the planner for an NPU target. Two handles are equal when they are the same
/// object.
#[derive(Clone, Debug)]
pub struct NpuHandle(Arc<dyn NpuLowering>);

impl NpuHandle {
    /// A handle on `lowering`.
    pub fn new(lowering: impl NpuLowering + 'static) -> Self {
        Self(Arc::new(lowering))
    }

    /// The lowering this handle holds.
    pub fn lowering(&self) -> &dyn NpuLowering {
        &*self.0
    }
}

impl PartialEq for NpuHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for NpuHandle {}

/// Why a plan has no AIE shape ABI, or a region has no link recipe.
///
/// This stays in the planner on purpose (POOT-1244 review): it names planner-level values ([`ValueId`] of the
/// region's graph), and `poot-npu-artifacts` (rank 1) cannot depend on `poot-graph-ir` (rank 4). Its vendor
/// causes are the artifact crate's own typed errors ([`AieArtifactError`], [`AieAbiMismatch`]), not copies.
/// The limit vocabulary, which names no graph value, lives with the artifacts ([`poot_npu_artifacts::AieLimitRefusal`]).
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum AieAbiRefusal {
    /// The plan's kernel choice is not an AIE serialized schedule, so it has no shape ABI.
    #[error("the plan's choice is not an AIE serialized schedule")]
    NotSerialized,
    /// The linked artifact's ABI could not be derived.
    #[error(transparent)]
    Abi(#[from] AieArtifactError),
    /// An equation of a region does not plan as one scratch-free dispatch that reads its operands and
    /// writes its output: a core runs one body once, so there is nothing else to place on a tile.
    #[error("equation {eqn:?} does not plan as one scratch-free dispatch over its operands")]
    NotOneDispatch {
        /// The value the equation produces.
        eqn: ValueId,
    },
    /// An equation's sequential bindings are not its operands in order followed by its one output.
    #[error("equation {eqn:?}: the sequential bindings are not its operands then its output")]
    Bindings {
        /// The value the equation produces.
        eqn: ValueId,
    },
    /// Two equations bind one value with different dimensions or element kinds.
    #[error(
        "value {value:?} is bound with different dimensions or elements by two equations of the region"
    )]
    ReaderDims {
        /// The value that is read twice.
        value: ValueId,
    },
    /// The region's wiring is not a link recipe.
    #[error(transparent)]
    Recipe(#[from] AieAbiMismatch),
    /// A region was asked of a program that was not compiled for an AIE target.
    #[error("the program was not compiled for an AIE target, so it has no region recipes")]
    NotAnAieProgram,
}
