//! NVIDIA/PTX execution for [`poot_graph_ir::Graph`] (Card 549): [`device::PtxDevice`] implements the
//! shared [`poot_executor::Device`] contract over CUDA graph record/replay
//! ([`poot_ptx_runtime::PtxContext::begin_capture`]/`end_capture`), exactly as `poot_gpu`'s
//! `WgpuDevice` does for wgpu. The one generic [`poot_executor::Engine`] owns planning, binding, state
//! sharing and the two-phase state commit; this crate contributes only the backend mechanism (kernel
//! loading/dispatch, a generated copy kernel, device-timed replay) plus multi-device PTX-to-PTX
//! peer-to-peer transport ([`p2p`], [`ring`], [`multi_device`], which stay on their own `PtxContext`
//! primitives per Card 584b and are untouched by the executor-contract migration).

pub mod device;
pub mod multi_device;
pub mod p2p;
mod precompile;
pub mod ring;

pub use precompile::precompile_graph;

use poot_graph_ir::{ExecutionValidationFailure, ValidationId, ValueId};
pub use poot_ptx_runtime::{PtxBuffer, PtxContext};
pub use poot_runtime_common::CompiledKernel;

pub use device::{PtxDevice, PtxKernel, PtxRecording};

/// An error enum never carries another poot error enum by value (spec 384). Every cause below is boxed, so
/// this type's size does not depend on the cause-chain depth. `std::io::Error` is exempt (already one
/// pointer wide). `thiserror` cannot box a `#[from]` field, so each boxed cause has a hand-written `From`
/// below and existing `?` keeps working.
#[derive(Debug, thiserror::Error)]
pub enum PtxGpuError {
    #[error("ptx runtime: {0}")]
    Ptx(#[source] Box<poot_ptx_runtime::PtxError>),
    #[error("codegen: {0}")]
    Codegen(#[source] Box<poot_codegen::CompileError>),
    #[error("plan: {0}")]
    Plan(#[source] Box<poot_graph_plan::PlanError>),
    #[error("stage: {0}")]
    Stage(#[source] Box<poot_graph_plan::CompileError>),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("value v{0} used before it was computed")]
    UseBeforeDef(ValueId),
    /// Card 642: a packed source carrier (an I8 const) was supplied a value that is not a packed
    /// component, or a packed component was supplied for a non-packed value.
    #[error("v{value}: a packed source binds a packed component, and only a packed source does")]
    PackedSourceValue { value: ValueId },
    #[error("{0}")]
    Graph(String),
    #[error(
        "op {0} planned as a cross-rank collective (world_size>1); this single-device executor \
         cannot run it - a multi-rank (multi-GPU) executor is required (card 049a)"
    )]
    CollectiveNeedsMultiRank(String),
    /// A validation lane of a packed walk was not exact zero. No output or state value was downloaded.
    #[error("validation: {0}")]
    Validation(#[source] Box<ExecutionValidationFailure>),
    #[error("validation packet has {actual} lanes; the graph declares {expected}")]
    ValidationPacketLength { expected: usize, actual: usize },
    #[error(
        "validation {id:?} reads v{value} through a strided view; a packet source must be materialized"
    )]
    ValidationPacketSource { id: ValidationId, value: ValueId },
}

impl From<poot_ptx_runtime::PtxError> for PtxGpuError {
    fn from(error: poot_ptx_runtime::PtxError) -> Self {
        PtxGpuError::Ptx(Box::new(error))
    }
}

impl From<poot_codegen::CompileError> for PtxGpuError {
    fn from(error: poot_codegen::CompileError) -> Self {
        PtxGpuError::Codegen(Box::new(error))
    }
}

impl From<poot_graph_plan::PlanError> for PtxGpuError {
    fn from(error: poot_graph_plan::PlanError) -> Self {
        PtxGpuError::Plan(Box::new(error))
    }
}

impl From<poot_graph_plan::CompileError> for PtxGpuError {
    fn from(error: poot_graph_plan::CompileError) -> Self {
        PtxGpuError::Stage(Box::new(error))
    }
}

impl From<ExecutionValidationFailure> for PtxGpuError {
    fn from(error: ExecutionValidationFailure) -> Self {
        PtxGpuError::Validation(Box::new(error))
    }
}
