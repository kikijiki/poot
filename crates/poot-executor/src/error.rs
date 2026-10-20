//! The contract's errors (dexec 3.5: `Fault` is kernel asserts only). Core review CR30
//! adds the failure-lifecycle distinction: an executable that may hold dirty state after a failure
//! needs a full reset before its next step (`NeedsReset`), and a device that failed to clean up after
//! an aborted transaction can never be reused (`Poisoned`).

use poot_graph_ir::{BoundaryId, SlotKey, ValidationPacketError, ValueId};
use poot_graph_plan::staged::SyncId;
use poot_graph_plan::{Target, multi_device::topology::DeviceId, staged::StageId};
use poot_target::BufferStorage;
use poot_tensor::DType;

/// Why an [`crate::Executor`] call failed. A failure that may have left an executable's state dirty is reported as [`ExecError::NeedsReset`] on the next step; a device that cannot clean up after an abort is [`ExecError::Poisoned`] for good.
#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    /// Loading weights, an entry or a kernel failed; see [`LoadError`].
    #[error(transparent)]
    Load(Box<LoadError>),
    /// A step input or state reset did not match what the entry declares; see [`BindError`].
    #[error(transparent)]
    Bind(Box<BindError>),
    /// A kernel assert or unreachable fired (Card 531c). Never a device-level failure; raised only
    /// from `Device::synchronize` reading the staged fault words.
    #[error("kernel `{kernel}` trapped (assert/unreachable code {code})")]
    Fault {
        /// The plan key of the kernel that trapped.
        kernel: String,
        /// The assert or unreachable code the kernel staged.
        code: u32,
    },
    /// The validated session's planted packet failed before any output readback (R472-006, ADR-0101
    /// decision 2's validated session).
    #[error(transparent)]
    Validation(Box<ValidationPacketError>),
    /// A dispatch asserted a length its buffer cannot hold; see [`DispatchError`]. Raised by the engine before
    /// any device sees the dispatch.
    #[error(transparent)]
    Dispatch(Box<DispatchError>),
    /// The backend device reported a failure; see [`DeviceError`].
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
    NoHostSync {
        /// The sync point the step reached.
        point: poot_graph_plan::staged::SyncId,
    },
    /// A coordinator refused a multi-device program or device before any device work.
    #[error(transparent)]
    Coordinator(Box<crate::CoordinatorError>),
    /// A stage of the program was compiled for a device this engine does not drive.
    #[error("stage {stage:?} is placed on {program:?}; this engine drives {engine:?}")]
    Placement {
        /// The stage whose placement is refused.
        stage: StageId,
        /// The device the stage was compiled for.
        program: DeviceId,
        /// The device this engine drives.
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

impl From<DispatchError> for ExecError {
    fn from(error: DispatchError) -> Self {
        ExecError::Dispatch(Box::new(error))
    }
}

impl From<DeviceError> for ExecError {
    fn from(error: DeviceError) -> Self {
        ExecError::Device(Box::new(error))
    }
}

/// Why the engine refused a dispatch before handing it to the device. The one check every backend shares:
/// a kernel trusts the element length it is published, so no length may exceed the buffer bound to it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DispatchError {
    /// An argument's asserted element count exceeds the capacity of the buffer bound to it.
    #[error(
        "dispatch argument {arg} asserts {elems} elements but its buffer holds {capacity} (the output is the last argument)"
    )]
    ArgExceedsBuffer {
        /// The argument's position: the inputs in kernel parameter order, then the output.
        arg: usize,
        /// The element count the dispatch asserted for the argument.
        elems: u32,
        /// The element capacity of the bound buffer ([`crate::Device::buffer_elems`]).
        capacity: usize,
    },
}

/// Why weights, an entry or a kernel could not be loaded onto a device. Every variant is raised before the failed load publishes anything an entry could run.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// A checkpoint file mapped under the executable's weight store changed after it was loaded
    /// (ADR-0103 amendment: the identity check at each bind batch). Nothing was uploaded from it.
    #[error(transparent)]
    SourceChanged(#[from] poot_quant::SourceChanged),
    /// The program was compiled for a different target than the device the engine drives.
    #[error("program was compiled for {program:?}, the device is {device:?}")]
    TargetMismatch {
        /// The target the program was compiled for.
        program: Box<Target>,
        /// The target of the device.
        device: Box<Target>,
    },
    /// A graph const its executable's weight map binds no weight to: a name that is no mapped
    /// `WeightId::const_name` (even when a store key equals it).
    #[error("const {value} ({name}) is bound to no weight")]
    Unbound {
        /// The const value with no weight.
        value: ValueId,
        /// The name the weight map was asked for.
        name: String,
    },
    /// An AIE region's plans have no link recipe to link against (POOT-664, POOT-1160).
    #[error("kernel {key}: {source}")]
    AieAbi {
        /// The region.
        key: String,
        /// Why the plans carry no link recipe.
        source: poot_graph_plan::AieAbiRefusal,
    },
    /// A linked AIE artifact's instruction stream is empty or longer than `u32::MAX` words, which the
    /// artifact validator and the artifact limit rule out: an invariant breach, named.
    #[error("kernel {key}: the AIE instruction stream is {bytes} bytes")]
    AieInstructionStream {
        /// The plan key of the kernel.
        key: String,
        /// The stream's length in bytes.
        bytes: usize,
    },
    /// An Intel NPU region could not be emitted as OpenVINO IR: the region the planner assembled is not well
    /// formed. An invariant breach in the lowering, named.
    #[error("kernel {key}: {source}")]
    VpuEmit {
        /// The key of the region.
        key: String,
        /// What the emitter refused.
        source: poot_codegen::VpuEmitError,
    },
    /// The region's argument ABI is not one the NPU device can bind.
    #[error("kernel {key}: {source}")]
    VpuArtifact {
        /// The key of the region.
        key: String,
        /// Why the artifact was refused.
        source: poot_runtime_common::VpuArtifactError,
    },
    /// The program has no kernel-body compiler for its target, but a per-equation kernel was asked for.
    #[error("target {target:?} compiles no per-equation kernel bodies")]
    NoKernelBodies {
        /// The target.
        target: poot_target::TargetId,
    },
    /// An AIE kernel could not be linked: a toolchain stage, the body/ABI check or the artifact validator.
    #[error("kernel {key}: {source}")]
    AieCompile {
        /// The plan key of the kernel.
        key: String,
        /// The toolchain, body/ABI or validator failure.
        source: poot_codegen::AieCompileError,
    },
    /// The linked AIE artifact does not fit the target's measured limits (an unmeasured limit refuses).
    #[error("kernel {key}: {source}")]
    AieLimit {
        /// The plan key of the kernel.
        key: String,
        /// Which measured limit the artifact exceeds or lacks.
        source: poot_graph_plan::AieLimitRefusal,
    },
    /// A graph const carries no name, so no weight can be bound to it.
    #[error("const {value} has no name to bind by")]
    UnnamedConst {
        /// The unnamed const value.
        value: ValueId,
    },
    /// A slot input carries no structured slot key, so no step input can be bound to it.
    #[error("slot {value} has no structured slot key")]
    UnkeyedSlot {
        /// The slot value without a key.
        value: ValueId,
    },
    /// A stored weight's format cannot be loaded into the lane the plan chose for it.
    #[error("weight {name}: stored {stored} cannot load as planned {planned}")]
    WeightFormat {
        /// The weight's name.
        name: String,
        /// Description of the stored format.
        stored: String,
        /// The device lane the plan chose.
        planned: BufferStorage,
    },
    /// Card 1007: a packed-F16 weight holds an infinity or NaN, which the generated bodies' in-register
    /// binary16 decode does not represent (its domain is the finite values).
    #[error(
        "weight {name}: element {index} is the non-finite F16 {bits:#06x}; a packed-F16 weight must be finite"
    )]
    NonFiniteF16Weight {
        /// The weight's name.
        name: String,
        /// Element index of the offending value.
        index: usize,
        /// The raw binary16 bits of the offending value.
        bits: u16,
    },
    /// Card 1008: a tracer declares the dtype its loader stores; a stored and declared dtype that differ are
    /// refused at load, never converted on upload.
    #[error("weight {name}: stored as {stored} but the graph declares {declared}")]
    WeightDtype {
        /// The weight's name.
        name: String,
        /// The dtype the store holds.
        stored: DType,
        /// The dtype the graph declares.
        declared: DType,
    },
    /// A weight table holds a word outside the range its graph declares
    /// ([`poot_graph_ir::Builder::constant_ranged`]): the graph's consumers rely on the declaration
    /// (a gathered word's cast to F32 was proven exact from it), so the weight is not uploaded.
    #[error("weight {name}: word {value} at index {index} is outside its declared range {range}")]
    WeightRange {
        /// The weight's name.
        name: String,
        /// Element index of the offending word.
        index: usize,
        /// The offending word.
        value: i32,
        /// The range the graph declares for the weight.
        range: poot_graph_ir::IndexRange,
    },
    /// A stored weight's shape differs from the shape the graph declares for it.
    #[error("weight {name}: stored shape {stored:?}, graph declares {declared:?}")]
    WeightShape {
        /// The weight's name.
        name: String,
        /// The shape the store holds.
        stored: Vec<usize>,
        /// The shape the graph declares.
        declared: Vec<usize>,
    },
    /// Card 546a's contract binds a packed store entry by (store key, `SourceRole`); naming for an
    /// arbitrary role mapping is Card 642's. A packed entry with no such binding yet is refused by name.
    #[error("const {value} ({name}): packed weight has no bound role")]
    UnboundPackedRole {
        /// The const value.
        value: ValueId,
        /// The const's name.
        name: String,
    },
    /// The executable's weight map does not stand against its store, or a rank's share of a weight could
    /// not be cut from it.
    #[error("weight map: {0}")]
    WeightMap(#[from] poot_quant::weights::WeightMapError),
    /// A kernel could not be compiled for the target.
    #[error("kernel {key}: {source}")]
    Codegen {
        /// The plan key of the kernel.
        key: String,
        /// The compile failure.
        source: poot_codegen::CompileError,
    },
    /// A compiled artifact was presented for a body it was not compiled from.
    #[error("kernel {key}: {source}")]
    KernelArtifact {
        /// The plan key of the kernel.
        key: String,
        /// The mismatch.
        source: poot_codegen::ArtifactMismatch,
    },
    /// A graph input, a value's storage, or a program shape this card's contract does not implement.
    /// Never left as a silent fallback: each site names what it refused.
    #[error("not implemented: {0}")]
    Unimplemented(&'static str),
    /// An executable, entry or stage handle names nothing this engine holds (unloaded, removed, or from another engine).
    #[error("unknown executable, entry, or stage handle")]
    UnknownHandle,
    /// Two stages declare the same slot key with a different aval or storage; they would share one
    /// host-written buffer.
    #[error("slot {key} is declared with different avals or storages by two stages")]
    SlotConflict {
        /// The slot key declared twice.
        key: SlotKey,
    },
    /// A link names a boundary its producing stage never exported.
    #[error("boundary {id:?} is read by stage {stage:?} but exported by no earlier stage")]
    BoundaryUnresolved {
        /// The boundary that has no producer.
        id: BoundaryId,
        /// The stage that reads it.
        stage: StageId,
    },
    /// An exported value is a view of carried state read before the stage's commit: a later stage
    /// would see the committed value, not the one the uncut graph reads.
    #[error("boundary {id:?} is a view of carried-state input {state}, which the stage commits")]
    BoundaryAliasesState {
        /// The exported boundary.
        id: BoundaryId,
        /// The carried-state input it views.
        state: ValueId,
    },
    /// An export names a value no equation of its stage produced.
    #[error("boundary {id:?} names a value its stage never materializes")]
    ExportUnmaterialized {
        /// The boundary whose value is never produced.
        id: BoundaryId,
    },
    /// A pooled slot or map no sync point of the program declares, or one no stage reads: the
    /// engine could not say which stored table it pages or in which lane.
    #[error("pool slot {key} is declared by no sync point of the program, or read by no stage")]
    PoolUndeclared {
        /// The pooled slot or map key.
        key: SlotKey,
    },
    /// A sync point reads a value the engine cannot hand a host as I32 expert ids.
    #[error("sync point {point:?} reads boundary {id:?}, which is not a plain I32 value")]
    SyncReadLane {
        /// The sync point that reads the value.
        point: SyncId,
        /// The boundary it reads.
        id: BoundaryId,
    },
    /// A pooled table whose experts do not fall on whole device words of its lane cannot be refilled
    /// one row at a time.
    #[error("pooled table {name}: {experts} experts do not fill whole device words of {planned:?}")]
    PoolRowAlignment {
        /// The pooled table's name.
        name: String,
        /// Number of experts the table's elements must divide into evenly.
        experts: usize,
        /// The device lane the plan chose.
        planned: BufferStorage,
    },
}

/// Why a step's inputs or a state reset did not match what the entry declares. Raised before anything dispatches or any state is zeroed.
#[derive(Debug, thiserror::Error)]
pub enum BindError {
    /// The entry declares a slot the step did not bind.
    #[error("no input for slot {key}")]
    Missing {
        /// The unbound slot.
        key: SlotKey,
    },
    /// The step bound an input that is no slot of the entry.
    #[error("input {key} names no slot of this entry")]
    Unknown {
        /// The unknown input key.
        key: SlotKey,
    },
    /// The step bound the same slot more than once.
    #[error("slot {key} bound twice")]
    Duplicate {
        /// The slot bound twice.
        key: SlotKey,
    },
    /// A bound input's declared shape differs from the slot's aval.
    #[error("slot {key}: expected shape {expected:?}, got {got:?}")]
    Shape {
        /// The slot.
        key: SlotKey,
        /// The shape the slot declares.
        expected: Vec<usize>,
        /// The shape the caller declared.
        got: Vec<usize>,
    },
    /// A bound input's element count differs from the slot's.
    #[error("slot {key}: expected element count {expected}, got {got}")]
    ElementCount {
        /// The slot.
        key: SlotKey,
        /// The element count the slot declares.
        expected: usize,
        /// The element count of the bound view.
        got: usize,
    },
    /// A bound input's dtype lane differs from the slot's.
    #[error("slot {key}: expected {expected:?}, got {got:?}")]
    Lane {
        /// The slot.
        key: SlotKey,
        /// The dtype the slot declares.
        expected: DType,
        /// The dtype of the bound view.
        got: DType,
    },
    /// A step input holds a word outside the range its slot declares (POOT-726: `Token` in
    /// `[0, vocab)`, `Pos` in `[0, capacity)`). The graph's consumers rely on the declaration (an
    /// `I32 -> F32` cast of the slot was proven exact from it), so nothing dispatches.
    #[error("slot {key}: word {value} at index {index} is outside its declared range {range}")]
    Range {
        /// The slot.
        key: SlotKey,
        /// Element index of the offending word.
        index: usize,
        /// The offending word.
        value: i32,
        /// The range the slot declares.
        range: poot_graph_ir::IndexRange,
    },
    /// A write-map slot was bound from a raw view. Its words are slot indices the kernel stores through,
    /// so only a checked [`poot_graph_ir::ForwardSlotMap`] binds it.
    #[error("slot {key}: a write map binds only as a checked ForwardSlotMap")]
    UncheckedWriteMap {
        /// The write-map slot.
        key: SlotKey,
    },
    /// A checked write map was bound to a slot that is not a write-map slot.
    #[error("slot {key}: a ForwardSlotMap binds only a write-map slot")]
    NotAWriteMap {
        /// The slot the map was bound to.
        key: SlotKey,
    },
    /// `reset_state(Row(row))` named a row outside the carried state's leading axis, or the state
    /// is not a whole number of equal rows.
    #[error("state {name}: cannot reset row {row} of its {rows} leading rows")]
    StateRow {
        /// The carried state's name.
        name: String,
        /// The row the reset named.
        row: usize,
        /// Rows on the state's leading axis.
        rows: usize,
    },
    /// A hosted embed gather's token id is not a row of its table.
    #[error("embed {name}: token {token} is not one of its {rows} rows")]
    TokenOutOfRange {
        /// The embed's name.
        name: String,
        /// The offending token id.
        token: i32,
        /// Rows in the embed table.
        rows: usize,
    },
    /// A sync wrote pool rows outside the table's pool (`first_row + len > pool_rows`).
    #[error("pooled table {table}: rows {first_row}..{end} are outside its {pool_rows} pool rows")]
    RowRange {
        /// The pooled table's name.
        table: String,
        /// First row the sync wrote.
        first_row: usize,
        /// One past the last row the sync wrote.
        end: usize,
        /// Rows in the table's pool.
        pool_rows: usize,
    },
    /// A sync named an expert the table does not hold, or a table the site does not have.
    #[error("pooled table {table}: expert {expert} is not one of its {experts}")]
    ExpertRange {
        /// The pooled table's name.
        table: String,
        /// The expert index the sync named.
        expert: usize,
        /// Experts the table holds.
        experts: usize,
    },
    /// A sync's map is not `[E]`.
    #[error("pool map {key}: {got} entries for {experts} experts")]
    MapLength {
        /// The pool map slot.
        key: SlotKey,
        /// Entries the sync supplied.
        got: usize,
        /// Entries the map must have, one per expert.
        experts: usize,
    },
    /// A carried state this executable already holds has a different aval than the entry declares for it.
    #[error("state {name}: existing {expected:?}, entry declares {got:?}")]
    StateSchema {
        /// The state's name.
        name: String,
        /// The aval of the existing state.
        expected: poot_graph_ir::TensorType,
        /// The aval the entry declares.
        got: poot_graph_ir::TensorType,
    },
}

/// A backend device failure, tagged with the backend that raised it.
#[derive(Debug, thiserror::Error)]
#[error("{backend} device: {source}")]
pub struct DeviceError {
    /// The backend's name.
    pub backend: &'static str,
    /// The backend's own error.
    pub source: Box<dyn std::error::Error + Send + Sync>,
}
