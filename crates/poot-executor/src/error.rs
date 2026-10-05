//! The contract's errors (dexec 3.5: `Fault` is kernel asserts only). Core review CR30
//! adds the failure-lifecycle distinction: an executable that may hold dirty state after a failure
//! needs a full reset before its next step (`NeedsReset`), and a device that failed to clean up after
//! an aborted transaction can never be reused (`Poisoned`).

use poot_graph_ir::{SlotKey, ValidationPacketError, ValueId};
use poot_graph_plan::{PlanKind, Target};
use poot_target::BufferStorage;
use poot_tensor::DType;

#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error(transparent)]
    Load(Box<LoadError>),
    #[error(transparent)]
    Bind(Box<BindError>),
    /// A kernel assert or unreachable fired (Card 531c). Never a device-level failure; raised only
    /// from `Device::synchronize` reading the staged fault words.
    #[error("kernel `{kernel}` trapped (assert/unreachable code {code})")]
    Fault { kernel: String, code: u32 },
    /// The validated session's planted packet failed before any output readback (R472-006, ADR-0101
    /// decision 2's validated session).
    #[error(transparent)]
    Validation(Box<ValidationPacketError>),
    #[error(transparent)]
    Device(Box<DeviceError>),
    /// A step output's bytes did not form a host tensor of the declared dtype and shape.
    #[error(transparent)]
    Carrier(#[from] poot_tensor::CarrierError),
    /// The executable's state may be dirty after a prior failure; `step` refuses until a complete
    /// `reset_state(All)` restores it (core review CR30).
    #[error("executable {0:?} needs a complete reset before its next step")]
    NeedsReset(crate::ExecutableId),
    /// The device could not establish a reusable state after an aborted transaction; no further work
    /// can run on it, and resetting an executable cannot clear this (core review CR30).
    #[error("the device is poisoned and cannot accept further work")]
    Poisoned,
}

impl From<LoadError> for ExecError {
    fn from(error: LoadError) -> Self {
        ExecError::Load(Box::new(error))
    }
}

impl From<BindError> for ExecError {
    fn from(error: BindError) -> Self {
        ExecError::Bind(Box::new(error))
    }
}

impl From<ValidationPacketError> for ExecError {
    fn from(error: ValidationPacketError) -> Self {
        ExecError::Validation(Box::new(error))
    }
}

impl From<DeviceError> for ExecError {
    fn from(error: DeviceError) -> Self {
        ExecError::Device(Box::new(error))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("program was compiled for {program:?}, the device is {device:?}")]
    TargetMismatch {
        program: Box<Target>,
        device: Box<Target>,
    },
    /// A graph const its executable's `WeightSource` binds no weight to: under `Map`, a name that is
    /// no mapped `WeightId::const_name` (even when a store key equals it); under `ConstNames`, a
    /// name no store entry has.
    #[error("const v{value} ({name}) is bound to no weight")]
    Unbound { value: ValueId, name: String },
    #[error("const v{value} has no name to bind by")]
    UnnamedConst { value: ValueId },
    #[error("slot v{value} has no structured slot key")]
    UnkeyedSlot { value: ValueId },
    #[error("weight {name}: stored {stored} cannot load as planned {planned}")]
    WeightFormat {
        name: String,
        stored: String,
        planned: BufferStorage,
    },
    /// Card 1007: a packed-F16 weight holds an infinity or NaN, which the generated bodies' in-register
    /// binary16 decode does not represent (its domain is the finite values).
    #[error(
        "weight {name}: element {index} is the non-finite F16 {bits:#06x}; a packed-F16 weight must be finite"
    )]
    NonFiniteF16Weight {
        name: String,
        index: usize,
        bits: u16,
    },
    /// Card 1008: a tracer declares the dtype its loader stores; a stored and declared dtype that differ are
    /// refused at load, never converted on upload.
    #[error("weight {name}: stored as {stored} but the graph declares {declared}")]
    WeightDtype {
        name: String,
        stored: DType,
        declared: DType,
    },
    #[error("weight {name}: stored shape {stored:?}, graph declares {declared:?}")]
    WeightShape {
        name: String,
        stored: Vec<usize>,
        declared: Vec<usize>,
    },
    /// Card 546a's contract binds a packed store entry by (store key, `SourceRole`); naming for an
    /// arbitrary role mapping is Card 642's. A packed entry with no such binding yet is refused by name.
    #[error("const v{value} ({name}): packed weight has no bound role")]
    UnboundPackedRole { value: ValueId, name: String },
    #[error("eqn {eqn} planned as {kind:?}, which no device loads")]
    UnloadablePlan { eqn: usize, kind: PlanKind },
    #[error("kernel {key}: {source}")]
    Codegen {
        key: String,
        source: poot_codegen::CompileError,
    },
    /// A graph input, a value's storage, or a program shape this card's contract does not implement.
    /// Never left as a silent fallback: each site names what it refused.
    #[error("not implemented: {0}")]
    Unimplemented(&'static str),
    #[error("unknown executable, entry, or stage handle")]
    UnknownHandle,
    /// A `StagedProgram` names more than one stage, or a stage on a device this executor does not
    /// drive. Card 581a's multi-stage composition is out of this card's scope.
    #[error("staged program names {0} stages; this card's engine drives exactly one")]
    NotSingleStage(usize),
}

#[derive(Debug, thiserror::Error)]
pub enum BindError {
    #[error("no input for slot {key}")]
    Missing { key: SlotKey },
    #[error("input {key} names no slot of this entry")]
    Unknown { key: SlotKey },
    #[error("slot {key} bound twice")]
    Duplicate { key: SlotKey },
    #[error("slot {key}: expected shape {expected:?}, got {got:?}")]
    Shape {
        key: SlotKey,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("slot {key}: expected element count {expected}, got {got}")]
    ElementCount {
        key: SlotKey,
        expected: usize,
        got: usize,
    },
    #[error("slot {key}: expected {expected:?}, got {got:?}")]
    Lane {
        key: SlotKey,
        expected: DType,
        got: DType,
    },
    /// A hosted embed gather's token id is not a row of its table.
    #[error("embed {name}: token {token} is not one of its {rows} rows")]
    TokenOutOfRange {
        name: String,
        token: i32,
        rows: usize,
    },
    /// A packed embed row did not decode.
    #[error("embed {name}: {source}")]
    RowDecode {
        name: String,
        source: poot_quant::decode::DecodeError,
    },
    /// A hosted embed's rows do not encode into the slot's planned lane.
    #[error("embed {name}: rows do not encode into {planned:?}")]
    TokenEmbedEncode {
        name: String,
        planned: poot_target::BufferStorage,
    },
    #[error("state {name}: existing {expected:?}, entry declares {got:?}")]
    StateSchema {
        name: String,
        expected: poot_graph_ir::TensorType,
        got: poot_graph_ir::TensorType,
    },
}

#[derive(Debug, thiserror::Error)]
#[error("{backend} device: {source}")]
pub struct DeviceError {
    pub backend: &'static str,
    pub source: Box<dyn std::error::Error + Send + Sync>,
}
