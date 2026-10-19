//! NVIDIA/PTX execution for [`poot_graph_ir::Graph`] (Card 549): [`device::PtxDevice`] implements the
//! shared [`poot_executor::Device`] contract over CUDA graph record/replay
//! ([`poot_ptx_runtime::PtxContext::begin_capture`]/`end_capture`), exactly as `poot_wgpu_device`'s
//! `WgpuDevice` does for wgpu. The one generic [`poot_executor::Engine`] owns planning, binding, state
//! sharing and the two-phase state commit; this crate contributes only the backend mechanism (kernel
//! loading/dispatch, a generated copy kernel, device-timed replay) plus what a device adds to join a
//! multi-device program ([`multi_device`]: `Collectives` over capture-scoped sync events, peer copies
//! and a recorded add, driven by [`poot_executor::Coordinator`]).

pub mod device;
pub mod multi_device;

use poot_graph_ir::{ExecutionValidationFailure, ValidationId, ValueId};

pub use device::{PtxDevice, PtxKernel, PtxRecording};

/// An error enum never carries another poot error enum by value (spec 384). Every cause below is boxed, so
/// this type's size does not depend on the cause-chain depth. `std::io::Error` is exempt (already one
/// pointer wide). `thiserror` cannot box a `#[from]` field, so each boxed cause has a hand-written `From`
/// below and existing `?` keeps working.
#[derive(Debug, thiserror::Error)]
pub enum PtxGpuError {
    /// The PTX runtime (CUDA driver, buffer or capture layer) failed.
    #[error("ptx runtime: {0}")]
    Ptx(#[source] Box<poot_ptx_runtime::PtxError>),
    /// Generating or compiling a kernel for the `Nvptx` target failed.
    #[error("codegen: {0}")]
    Codegen(#[source] Box<poot_codegen::CompileError>),
    /// A compiled kernel artifact is not the one compiled from the body it was presented with.
    #[error("kernel artifact: {0}")]
    KernelArtifact(#[source] Box<poot_codegen::ArtifactMismatch>),
    /// Planning the graph for this device failed.
    #[error("plan: {0}")]
    Plan(#[source] Box<poot_graph_plan::PlanError>),
    /// Compiling a stage of a staged or multi-device program failed.
    #[error("stage: {0}")]
    Stage(#[source] Box<poot_graph_plan::CompileError>),
    /// A host I/O operation failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The walk read a value before any equation or input produced it; a binding bug.
    #[error("value {0} used before it was computed")]
    UseBeforeDef(ValueId),
    /// Card 642: a packed source carrier (an I8 const) was supplied a value that is not a packed
    /// component, or a packed component was supplied for a non-packed value.
    #[error("{value}: a packed source binds a packed component, and only a packed source does")]
    PackedSourceValue {
        /// The value the mismatched binding was supplied for.
        value: ValueId,
    },
    /// A graph-level refusal this backend raises as text, with no dedicated variant.
    #[error("{0}")]
    Graph(String),
    /// A validation lane of a packed walk was not exact zero. No output or state value was downloaded.
    #[error("validation: {0}")]
    Validation(#[source] Box<ExecutionValidationFailure>),
    /// The validation packet's lane count differs from the graph's declared validations.
    #[error("validation packet has {actual} lanes; the graph declares {expected}")]
    ValidationPacketLength {
        /// Lanes the graph declares.
        expected: usize,
        /// Lanes the packet carries.
        actual: usize,
    },
    #[error(
        "validation {id:?} reads {value} through a strided view; a packet source must be materialized"
    )]
    /// A validation packet source is read through a strided view; the source must be materialized.
    ValidationPacketSource {
        /// The validation whose packet source is strided.
        id: ValidationId,
        /// The strided value the packet reads.
        value: ValueId,
    },
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
