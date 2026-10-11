//! The contract's errors (dexec 3.5: `Fault` is kernel asserts only). Core review CR30
//! adds the failure-lifecycle distinction: an executable that may hold dirty state after a failure
//! needs a full reset before its next step (`NeedsReset`), and a device that failed to clean up after
//! an aborted transaction can never be reused (`Poisoned`).

use poot_graph_ir::{BoundaryId, SlotKey, ValidationPacketError, ValueId};
use poot_graph_plan::{DeviceId, StageId, SyncId, Target};
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
    /// The host that served a sync point refused it (an expert id outside the table, a pool too
    /// small for the step). The step did not complete: the executable needs a reset.
    #[error("host sync refused: {0}")]
    Host(Box<dyn std::error::Error + Send + Sync>),
    /// A step reached a sync point and the host handed in [`crate::NoSync`].
    #[error("sync point {point:?} needs a host: the step was run with NoSync")]
    NoHostSync { point: poot_graph_plan::SyncId },
    /// A coordinator refused a multi-device program or device before any device work.
    #[error(transparent)]
    Coordinator(Box<crate::CoordinatorError>),
    /// A stage of the program was compiled for a device this engine does not drive.
    #[error("stage {stage:?} is placed on {program:?}; this engine drives {engine:?}")]
    Placement {
        stage: StageId,
        program: DeviceId,
        engine: DeviceId,
    },
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
    /// A checkpoint file mapped under the executable's weight store changed after it was loaded
    /// (ADR-0103 amendment: the identity check at each bind batch). Nothing was uploaded from it.
    #[error(transparent)]
    SourceChanged(#[from] poot_quant::SourceChanged),
    #[error("program was compiled for {program:?}, the device is {device:?}")]
    TargetMismatch {
        program: Box<Target>,
        device: Box<Target>,
    },
    /// A graph const its executable's weight map binds no weight to: a name that is no mapped
    /// `WeightId::const_name` (even when a store key equals it).
    #[error("const {value} ({name}) is bound to no weight")]
    Unbound { value: ValueId, name: String },
    /// An AIE kernel's plan has no shape ABI to link against (POOT-664).
    #[error("kernel {key}: {source}")]
    AieAbi {
        key: String,
        source: poot_graph_plan::AieAbiRefusal,
    },
    /// A linked AIE artifact's instruction stream is empty or longer than `u32::MAX` words, which the
    /// artifact validator and the artifact limit rule out: an invariant breach, named.
    #[error("kernel {key}: the AIE instruction stream is {bytes} bytes")]
    AieInstructionStream { key: String, bytes: usize },
    /// An AIE kernel could not be linked: a toolchain stage, the body/ABI check or the artifact validator.
    #[error("kernel {key}: {source}")]
    AieCompile {
        key: String,
        source: poot_codegen::AieCompileError,
    },
    /// The linked AIE artifact does not fit the target's measured limits (an unmeasured limit refuses).
    #[error("kernel {key}: {source}")]
    AieLimit {
        key: String,
        source: poot_graph_plan::AieLimitRefusal,
    },
    /// The linked AIE artifact does not describe the body's buffer parameters.
    #[error("kernel {key}: {source}")]
    AieHandle {
        key: String,
        source: poot_codegen::AieAbiMismatch,
    },
    #[error("const {value} has no name to bind by")]
    UnnamedConst { value: ValueId },
    #[error("slot {value} has no structured slot key")]
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
    /// A weight table holds a word outside the range its graph declares
    /// ([`poot_graph_ir::Builder::constant_ranged`]): the graph's consumers rely on the declaration
    /// (a gathered word's cast to F32 was proven exact from it), so the weight is not uploaded.
    #[error("weight {name}: word {value} at index {index} is outside its declared range {range}")]
    WeightRange {
        name: String,
        index: usize,
        value: i32,
        range: poot_graph_ir::IndexRange,
    },
    #[error("weight {name}: stored shape {stored:?}, graph declares {declared:?}")]
    WeightShape {
        name: String,
        stored: Vec<usize>,
        declared: Vec<usize>,
    },
    /// Card 546a's contract binds a packed store entry by (store key, `SourceRole`); naming for an
    /// arbitrary role mapping is Card 642's. A packed entry with no such binding yet is refused by name.
    #[error("const {value} ({name}): packed weight has no bound role")]
    UnboundPackedRole { value: ValueId, name: String },
    /// A tensor-parallel rank's share of a weight could not be cut from the executable's map.
    #[error("rank weights: {0}")]
    RankWeights(#[from] poot_quant::weights::WeightMapError),
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
    /// Two stages declare the same slot key with a different aval or storage; they would share one
    /// host-written buffer.
    #[error("slot {key} is declared with different avals or storages by two stages")]
    SlotConflict { key: SlotKey },
    /// A link names a boundary its producing stage never exported.
    #[error("boundary {id:?} is read by stage {stage:?} but exported by no earlier stage")]
    BoundaryUnresolved { id: BoundaryId, stage: StageId },
    /// An exported value is a view of carried state read before the stage's commit: a later stage
    /// would see the committed value, not the one the uncut graph reads.
    #[error("boundary {id:?} is a view of carried-state input {state}, which the stage commits")]
    BoundaryAliasesState { id: BoundaryId, state: ValueId },
    /// An export names a value no equation of its stage produced.
    #[error("boundary {id:?} names a value its stage never materializes")]
    ExportUnmaterialized { id: BoundaryId },
    /// A pooled slot or map no sync point of the program declares, or one no stage reads: the
    /// engine could not say which stored table it pages or in which lane.
    #[error("pool slot {key} is declared by no sync point of the program, or read by no stage")]
    PoolUndeclared { key: SlotKey },
    /// A sync point reads a value the engine cannot hand a host as I32 expert ids.
    #[error("sync point {point:?} reads boundary {id:?}, which is not a plain I32 value")]
    SyncReadLane { point: SyncId, id: BoundaryId },
    /// A pooled table whose experts do not fall on whole device words of its lane cannot be refilled
    /// one row at a time.
    #[error("pooled table {name}: {experts} experts do not fill whole device words of {planned:?}")]
    PoolRowAlignment {
        name: String,
        experts: usize,
        planned: BufferStorage,
    },
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
    /// A step input holds a word outside the range its slot declares (POOT-726: `Token` in
    /// `[0, vocab)`, `Pos` in `[0, capacity)`). The graph's consumers rely on the declaration (an
    /// `I32 -> F32` cast of the slot was proven exact from it), so nothing dispatches.
    #[error("slot {key}: word {value} at index {index} is outside its declared range {range}")]
    Range {
        key: SlotKey,
        index: usize,
        value: i32,
        range: poot_graph_ir::IndexRange,
    },
    /// `reset_state(Row(row))` named a row outside the carried state's leading axis, or the state
    /// is not a whole number of equal rows.
    #[error("state {name}: cannot reset row {row} of its {rows} leading rows")]
    StateRow {
        name: String,
        row: usize,
        rows: usize,
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
    /// A sync wrote pool rows outside the table's pool (`first_row + len > pool_rows`).
    #[error("pooled table {table}: rows {first_row}..{end} are outside its {pool_rows} pool rows")]
    RowRange {
        table: String,
        first_row: usize,
        end: usize,
        pool_rows: usize,
    },
    /// A sync named an expert the table does not hold, or a table the site does not have.
    #[error("pooled table {table}: expert {expert} is not one of its {experts}")]
    ExpertRange {
        table: String,
        expert: usize,
        experts: usize,
    },
    /// A sync's map is not `[E]`.
    #[error("pool map {key}: {got} entries for {experts} experts")]
    MapLength {
        key: SlotKey,
        got: usize,
        experts: usize,
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
