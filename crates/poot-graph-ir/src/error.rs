//! Typed errors for the graph IR.

use crate::graph::{Slot, Storage, ValueId};
use crate::types::{DType, TensorType};
use poot_quant::SourceRole;

/// A named graph storage class used in transactional append diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuilderValueNamespace {
    Input,
    Constant,
    Slot,
    EquationResult,
    NamedValue,
}

/// A graph collection whose checked growth or reservation failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuilderCollection {
    Values,
    Inputs,
    Constants,
    Slots,
    Equations,
    NameBytes,
}

/// A typed failure while staging, validating, or committing a graph fragment.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BuilderAppendError {
    #[error("append plan belongs to a different Builder")]
    ForeignBuilder,
    #[error("{namespace:?} name must not be empty")]
    EmptyName { namespace: BuilderValueNamespace },
    #[error("name {name:?} requested for {requested:?} collides with existing {existing:?}")]
    NameCollision {
        name: String,
        requested: BuilderValueNamespace,
        existing: BuilderValueNamespace,
    },
    #[error("staged input {name:?} has Device storage")]
    DeviceInput { name: String },
    #[error("staged equation result v{value} has non-Device storage {storage:?}")]
    EquationResultStorage { value: ValueId, storage: Storage },
    #[error("staged equation {equation} ({operation}) references invalid operand v{value}")]
    InvalidOperand {
        equation: usize,
        operation: String,
        value: ValueId,
    },
    #[error("staged equation {equation} ({operation}) uses v{value} before its definition")]
    OperandNotDefined {
        equation: usize,
        operation: String,
        value: ValueId,
    },
    #[error("staged equation {equation} ({operation}) redefines v{value}")]
    ValueRedefined {
        equation: usize,
        operation: String,
        value: ValueId,
    },
    #[error("staged equation {equation} ({operation}): {source}")]
    Inference {
        equation: usize,
        operation: String,
        #[source]
        source: ShapeError,
    },
    #[error("staged equation {equation} ({operation}) stores {stored}, inferred {inferred}")]
    OutputTypeMismatch {
        equation: usize,
        operation: String,
        stored: TensorType,
        inferred: TensorType,
    },
    #[error("staged Device value v{value} has no equation definition")]
    MissingDefinition { value: ValueId },
    #[error("append result v{value} is not one produced staged value")]
    InvalidResult { value: ValueId },
    #[error("append plan does not declare a result")]
    MissingResult,
    #[error("append plan already declares result v{value}")]
    ResultAlreadyDeclared { value: ValueId },
    #[error("checked {collection:?} growth overflow: current={current}, additional={additional}")]
    SizeOverflow {
        collection: BuilderCollection,
        current: usize,
        additional: usize,
    },
    #[error("could not reserve {additional} entries in {collection:?}")]
    Reserve {
        collection: BuilderCollection,
        additional: usize,
    },
    #[error("append plan is stale: expected generation {expected}, actual {actual}")]
    StalePlan { expected: u64, actual: u64 },
    #[error("append plan has already been committed")]
    ConsumedPlan,
    #[error("builder generation overflow")]
    GenerationOverflow,
    #[error("exact I32 input {name:?} shape {shape:?} overflows its element count")]
    ElementCountOverflow { name: String, shape: Vec<usize> },
    #[error("exact I32 slot {slot:?} tag must not be empty")]
    EmptySlotTag { slot: Slot },
    #[error("packed expert row table must not be empty")]
    EmptyPackedRows,
    #[error("packed row at index {index} has ordinal {actual}, expected {expected}")]
    PackedRowOrdinal {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("packed linear id at row {index} must not be empty")]
    EmptyPackedLinearId { index: usize },
    #[error("duplicate packed linear id {linear_id:?} at rows {first} and {second}")]
    DuplicatePackedLinearId {
        linear_id: String,
        first: usize,
        second: usize,
    },
    #[error("packed row {index} descriptor differs from row zero")]
    PackedDescriptorMismatch { index: usize },
    #[error("packed block-diagonal out={out} is not divisible by blocks={blocks}")]
    PackedBlockDiagonalBlocks { out: usize, blocks: usize },
    #[error("{field}={value} exceeds the inclusive exact-f32 integer bound {max}")]
    F32ExactIntegerRange {
        field: &'static str,
        value: usize,
        max: usize,
    },
    #[error("operand v{value} has type {actual}, expected {expected}")]
    OperandType {
        value: ValueId,
        expected: TensorType,
        actual: TensorType,
    },
}

/// A structural or binding-table failure returned by [`crate::Graph::validate`].
///
/// Value ids are kept as fields so callers can classify malformed graphs without parsing diagnostics.
/// Equation errors also retain the operation name and inference source needed to locate the failure.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GraphValidationError {
    #[error("validation output v{value} out of range for {value_count} values")]
    ValidationValueOutOfRange { value: ValueId, value_count: usize },
    #[error("duplicate validation id {id:?}")]
    DuplicateValidationId { id: crate::graph::ValidationId },
    #[error("validation {id:?} has an empty name")]
    EmptyValidationName { id: crate::graph::ValidationId },
    #[error("duplicate validation name {name:?}")]
    DuplicateValidationName { name: String },
    #[error("validation {id:?} value v{value} has dtype {actual}; expected f32")]
    ValidationDtype {
        id: crate::graph::ValidationId,
        value: ValueId,
        actual: DType,
    },
    #[error("validation {id:?} value v{value} has zero lanes")]
    EmptyValidationValue {
        id: crate::graph::ValidationId,
        value: ValueId,
    },
    #[error("validation {id:?} value v{value} shape {shape:?} overflows usize")]
    ValidationShapeOverflow {
        id: crate::graph::ValidationId,
        value: ValueId,
        shape: Vec<usize>,
    },
    #[error("validation packet lane accounting overflowed")]
    ValidationPacketLaneOverflow,
    #[error("validation packet byte accounting overflowed")]
    ValidationPacketByteOverflow,
    #[error("validation packet is {byte_len} bytes; maximum is {max_bytes}")]
    ValidationPacketTooLarge { byte_len: usize, max_bytes: usize },
    #[error("input v{value} out of range for {value_count} values")]
    InputOutOfRange { value: ValueId, value_count: usize },
    #[error("duplicate input binder v{value}")]
    DuplicateInput { value: ValueId },
    #[error("input binder v{value} has Device storage")]
    DeviceInput { value: ValueId },
    #[error("const table v{value} out of range for {value_count} values")]
    ConstOutOfRange { value: ValueId, value_count: usize },
    #[error("duplicate const table entry v{value}")]
    DuplicateConst { value: ValueId },
    #[error("const table v{value} is not an input binder")]
    ConstNotInput { value: ValueId },
    #[error("const table v{value} has {storage:?} storage; expected Const or State")]
    ConstStorageMismatch { value: ValueId, storage: Storage },
    #[error("slot table v{value} out of range for {value_count} values")]
    SlotOutOfRange { value: ValueId, value_count: usize },
    #[error("duplicate slot table entry v{value}")]
    DuplicateSlot { value: ValueId },
    #[error("slot table v{value} is not an input binder")]
    SlotNotInput { value: ValueId },
    #[error("slot table v{value} kind {table_slot:?} does not match storage {storage:?}")]
    SlotStorageMismatch {
        value: ValueId,
        table_slot: Slot,
        storage: Storage,
    },
    #[error("input binder v{value} with {storage:?} storage is missing from const table")]
    MissingConstBinding { value: ValueId, storage: Storage },
    #[error("input binder v{value} with {slot:?} slot storage is missing from slot table")]
    MissingSlotBinding { value: ValueId, slot: Slot },
    #[error("eqn {equation}: operand v{value} used before def")]
    EquationOperandNotDefined { equation: usize, value: ValueId },
    #[error(
        "eqn {equation} ({operation}): two value operands have different dtypes ({left} vs {right})"
    )]
    BinaryValueDtypeMismatch {
        equation: usize,
        operation: String,
        left: DType,
        right: DType,
    },
    #[error("eqn {equation} ({operation}): output v{value} out of range for {value_count} values")]
    EquationOutputOutOfRange {
        equation: usize,
        operation: String,
        value: ValueId,
        value_count: usize,
    },
    #[error("eqn {equation} ({operation}): output v{value} redefines an existing SSA value")]
    EquationOutputRedefined {
        equation: usize,
        operation: String,
        value: ValueId,
    },
    #[error("eqn {equation} ({operation}): {source}")]
    EquationInference {
        equation: usize,
        operation: String,
        #[source]
        source: ShapeError,
    },
    #[error("eqn {equation} ({operation}): stored aval {stored} != inferred {inferred}")]
    EquationOutputTypeMismatch {
        equation: usize,
        operation: String,
        stored: TensorType,
        inferred: TensorType,
    },
    #[error("output v{value} not defined")]
    OutputNotDefined { value: ValueId },
    #[error("validation {id:?} output v{value} not defined")]
    ValidationOutputNotDefined {
        id: crate::graph::ValidationId,
        value: ValueId,
    },
    #[error("state pair (v{state_input}, v{state_output}) out of range for {value_count} values")]
    StatePairOutOfRange {
        state_input: ValueId,
        state_output: ValueId,
        value_count: usize,
    },
    #[error("duplicate state destination v{value}")]
    DuplicateStateDestination { value: ValueId },
    #[error("state_in v{value} has {storage:?} storage; expected State")]
    StateInputStorageMismatch { value: ValueId, storage: Storage },
    #[error("state_in v{value} is not an input binder")]
    StateInputNotInput { value: ValueId },
    #[error("state_out v{value} not defined")]
    StateOutputNotDefined { value: ValueId },
    #[error(
        "state pair (v{state_input}, v{state_output}) type mismatch: {input_type} vs {output_type}"
    )]
    StateTypeMismatch {
        state_input: ValueId,
        state_output: ValueId,
        input_type: TensorType,
        output_type: TensorType,
    },
    #[error("state input v{value} has no StateRole")]
    MissingStateRole { value: ValueId },
    #[error("v{value} has a StateRole but {storage:?} storage; expected State")]
    UnexpectedStateRole { value: ValueId, storage: Storage },
    #[error(
        "state input v{value} declares Positional axis {axis} out of range for rank-{rank} value"
    )]
    StateRoleAxis {
        value: ValueId,
        axis: u8,
        rank: usize,
    },
}

/// A shape/dtype inference failure. Produced by [`crate::op::OpKind::infer`], the single source of truth
/// for result types. Inference is a pure host function, so these are fully unit-testable. A failure here
/// during tracing is a trace-time programmer error (the model wired incompatible shapes), so the
/// `Builder` surfaces it as a panic with context, the way `abstract_eval` raises in JAX; the typed error
/// is what the inference API and its tests return.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ShapeError {
    #[error("cannot broadcast shapes {a:?} and {b:?}")]
    Broadcast { a: Vec<usize>, b: Vec<usize> },

    #[error("reshape changes element count: {from:?} ({from_numel}) -> {to:?} ({to_numel})")]
    Reshape {
        from: Vec<usize>,
        to: Vec<usize>,
        from_numel: usize,
        to_numel: usize,
    },

    #[error("reshape element count overflows usize for shape {shape:?}")]
    ReshapeElementCountOverflow { shape: Vec<usize> },

    #[error("axis {axis} out of range for rank-{rank} tensor")]
    Axis { axis: usize, rank: usize },

    #[error("transpose perm {perm:?} does not match rank {rank}")]
    Perm { perm: Vec<usize>, rank: usize },

    #[error("slice {start}..{end} out of range for axis of length {len}")]
    Slice {
        start: usize,
        end: usize,
        len: usize,
    },

    #[error("concat: non-axis dim {dim} mismatch ({a} vs {b})")]
    Concat { dim: usize, a: usize, b: usize },

    #[error("matmul needs rank >= 2 operands, got {a:?} and {b:?}")]
    MatMulRank { a: Vec<usize>, b: Vec<usize> },

    #[error("matmul contracting dim mismatch: {a:?} x {b:?}")]
    MatMulContract { a: Vec<usize>, b: Vec<usize> },

    #[error("gather expects a scalar index, got shape {0:?}")]
    GatherIndex(Vec<usize>),

    #[error(
        "dynamic_update_slice: update {update:?} does not fit operand {operand:?} on axis {axis}"
    )]
    UpdateSlice {
        operand: Vec<usize>,
        update: Vec<usize>,
        axis: usize,
    },

    #[error("op expected {expected} inputs, got {got}")]
    Arity { expected: usize, got: usize },

    #[error("dtype mismatch: {a:?} vs {b:?}")]
    Dtype { a: TensorType, b: TensorType },

    #[error("{op} does not support {dtype}")]
    DtypeOp { op: &'static str, dtype: DType },

    /// `Cast` admits every other `(from, to)` pair; these nine have no evaluator definition (card 555): `BF16`/`F16` -> `I32`/`I8`, `I32`/`I8` -> `E4M3FN`, `E4M3FN` -> `I32`/`I8`, and
    /// `F32` -> `I8`.
    #[error("cast {from} -> {to} is not admitted")]
    CastUnsupported { from: DType, to: DType },

    #[error(
        "matmul operand dtypes {a} (a) and {b} (b) are not an allowed pairing (same arithmetic dtype, or F32 a with BF16/F16 b)"
    )]
    MatMulOperandDtype { a: DType, b: DType },

    #[error(
        "packed dequant {role:?} operand {field} mismatch: expected {expected:?}, got {actual:?}"
    )]
    PackedDequantOperand {
        role: SourceRole,
        field: &'static str,
        expected: Box<TensorType>,
        actual: Box<TensorType>,
    },

    #[error("packed contraction activation must end in K={expected_k}, got {actual:?}")]
    PackedContractionActivation {
        expected_k: usize,
        actual: TensorType,
    },

    #[error("packed contraction {operand} must use F32, got {actual}")]
    PackedContractionDtype {
        operand: &'static str,
        actual: DType,
    },

    #[error("packed contraction blocks={blocks} does not evenly divide out={out}")]
    PackedContractionBlocks { out: usize, blocks: usize },

    #[error(
        "dense contraction weight dtype {actual} is not admitted (see DENSE_CONTRACTION_WEIGHT_DTYPES)"
    )]
    DenseContractionWeightDtype { actual: DType },

    #[error("dense contraction {operand} dtype is wrong: got {actual}")]
    DenseContractionDtype {
        operand: &'static str,
        actual: DType,
    },

    #[error("dense contraction weight must be rank-2 [N, K] in checkpoint order, got {actual:?}")]
    DenseContractionWeightRank { actual: TensorType },

    #[error("dense contraction activation must end in K={expected_k}, got {actual:?}")]
    DenseContractionActivation {
        expected_k: usize,
        actual: TensorType,
    },

    #[error(
        "dense row gather source dtype {actual} is not admitted (see DENSE_ROW_GATHER_SOURCE_DTYPES)"
    )]
    DenseRowGatherSourceDtype { actual: DType },

    #[error("dense row gather {operand} dtype is wrong: got {actual}")]
    DenseRowGatherDtype {
        operand: &'static str,
        actual: DType,
    },

    #[error("dense row gather table must be rank-2 [V, R], got {actual:?}")]
    DenseRowGatherTableRank { actual: TensorType },

    #[error("arg_top_k: k={k} exceeds the expert axis extent {extent}")]
    TopK { k: usize, extent: usize },

    #[error(
        "flash attention q must be rank-4 [B, Hq, M, D] with n_rep={n_rep} dividing Hq ({form}), got {actual:?}"
    )]
    FlashAttentionQuery {
        form: &'static str,
        n_rep: usize,
        actual: Vec<usize>,
    },

    #[error("flash attention {operand} must be {expected:?}, got {actual:?}")]
    FlashAttentionOperand {
        operand: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("flash attention {operand} dtype {actual} must match q's {expected}")]
    FlashAttentionDtype {
        operand: &'static str,
        expected: DType,
        actual: DType,
    },

    #[error("rope rot={rot} must be even and within 1..=D for x {x:?}")]
    RopeWidth { rot: usize, x: Vec<usize> },

    #[error(
        "rope {operand} {actual:?} must broadcast into the rotated x {rotated:?} without growing it"
    )]
    RopeTable {
        operand: &'static str,
        rotated: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("rope {operand} dtype {actual} must match x's {expected}")]
    RopeDtype {
        operand: &'static str,
        expected: DType,
        actual: DType,
    },

    #[error("random_uniform seed must be I32, got {actual}")]
    RandomUniformSeedDtype { actual: DType },

    #[error("sample_token {operand} dtype must be {expected}, got {actual}")]
    SampleTokenDtype {
        operand: &'static str,
        expected: DType,
        actual: DType,
    },

    #[error("sample_token {operand} shape {actual:?} does not match the expected {expected:?}")]
    SampleTokenShape {
        operand: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}
