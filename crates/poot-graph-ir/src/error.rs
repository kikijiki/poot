//! Typed errors for the graph IR.

use std::num::NonZeroU32;

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
    /// The outputs table (result, witnesses, exports, carried-state pairs).
    Outputs,
}

/// A typed failure of a tracer primitive, an `ops` composition, or a staged append: the one error the
/// whole tracer path returns instead of panicking.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum BuilderError {
    /// A request would exceed the caller's [`crate::TraceLimits`]; nothing was allocated or appended.
    #[error(transparent)]
    Budget(#[from] crate::budget::TraceBudgetExceeded),
    #[error("handle {value} was minted by another Builder and cannot be used with this one")]
    ForeignHandle { value: ValueId },
    #[error("literal {value} does not fit an I32 literal")]
    LiteralOutOfRange { value: usize },
    #[error("value {value} is not in the builder's value table")]
    UnknownValue { value: ValueId },
    #[error("input {name:?} declares a range but is {dtype}; a range bounds I32 words")]
    RangeNotI32 { name: String, dtype: DType },
    #[error(
        "duplicate tag {tag:?} for slot {slot:?}: a per-name binder would bind both ids from one buffer"
    )]
    DuplicateSlotTag { slot: Slot, tag: String },
    /// A composition's own precondition on its operands or parameters does not hold.
    #[error("{operation}: {requirement}")]
    Requirement {
        operation: &'static str,
        requirement: Requirement,
    },
    /// The finished graph is not well formed, or does not carry what a resumed graph carried.
    #[error("the finished graph is invalid: {0}")]
    Invalid(Box<crate::graph::EditError>),
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
    #[error("staged equation result {value} has non-Device storage {storage:?}")]
    EquationResultStorage { value: ValueId, storage: Storage },
    #[error("staged equation {equation} ({operation}) references invalid operand {value}")]
    InvalidOperand {
        equation: usize,
        operation: String,
        value: ValueId,
    },
    #[error("staged equation {equation} ({operation}) uses {value} before its definition")]
    OperandNotDefined {
        equation: usize,
        operation: String,
        value: ValueId,
    },
    #[error("staged equation {equation} ({operation}) redefines {value}")]
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
    #[error("staged Device value {value} has no equation definition")]
    MissingDefinition { value: ValueId },
    #[error("append result {value} is not one produced staged value")]
    InvalidResult { value: ValueId },
    #[error("append plan does not declare a result")]
    MissingResult,
    #[error("append plan already declares result {value}")]
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
    #[error("packed linear id at row {index} must not be empty")]
    EmptyPackedLinearId { index: usize },
    #[error("packed block-diagonal out={out} is not divisible by blocks={blocks}")]
    PackedBlockDiagonalBlocks { out: usize, blocks: usize },
    #[error("operand {value} has type {actual}, expected {expected}")]
    OperandType {
        value: ValueId,
        expected: TensorType,
        actual: TensorType,
    },
}

/// A precondition a composition in [`crate::ops`] states about its operands or parameters.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum Requirement {
    #[error("operand dtype must be {expected}, found {found}")]
    Dtype { expected: DType, found: DType },
    #[error("operand shape must end in [low, high], found {found:?}")]
    TrailingWordPair { found: Vec<usize> },
    #[error("operand must have at least {at_least} axes, found shape {found:?}")]
    MinRank { at_least: usize, found: Vec<usize> },
    #[error("operand must have exactly {expected} axes, found shape {found:?}")]
    Rank { expected: usize, found: Vec<usize> },
    #[error("operand axis {axis} must be {expected}, found shape {found:?}")]
    Axis {
        axis: usize,
        expected: usize,
        found: Vec<usize>,
    },
    #[error("operand shapes must match, found {left:?} and {right:?}")]
    ShapesMatch { left: Vec<usize>, right: Vec<usize> },
    #[error("needs at least one expert")]
    NoExperts,
    #[error("needs 1 <= k <= E, got k={k}, E={experts}")]
    TopKRange { k: usize, experts: usize },
    #[error("is the decode (one token) path, got {tokens} tokens")]
    DecodeTokens { tokens: usize },
    #[error("last two axes must be square, found shape {found:?}")]
    SquareTrailingAxes { found: Vec<usize> },
    #[error("block size {size} must be a power of two")]
    PowerOfTwo { size: usize },
    #[error("{value} must be a multiple of {of}")]
    Multiple { value: usize, of: usize },
    #[error("needs one entry per section ({expected}), got {found}")]
    SectionCount { expected: usize, found: usize },
    #[error("section widths must sum to {expected}, got {found}")]
    SectionWidths { expected: usize, found: usize },
    #[error("window must be {expected:?}, found {found:?}")]
    WindowShape {
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("{what} must be at least 1")]
    AtLeastOne { what: &'static str },
    #[error("operand must be a rank >= 1 tensor")]
    Scalar,
}

impl From<crate::graph::EditError> for BuilderError {
    fn from(error: crate::graph::EditError) -> Self {
        BuilderError::Invalid(Box::new(error))
    }
}

/// A structural or binding-table failure of a graph draft, reported by [`crate::GraphEdit::commit`] as
/// [`crate::EditError::Invalid`].
///
/// Value ids are kept as fields so callers can classify malformed graphs without parsing diagnostics.
/// Equation errors also retain the operation name and inference source needed to locate the failure.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum GraphValidationError {
    #[error("the graph declares no primary result")]
    MissingResult,
    #[error("the graph declares more than one primary result")]
    MultipleResults,
    #[error("validation output {value} out of range for {value_count} values")]
    ValidationValueOutOfRange { value: ValueId, value_count: usize },
    #[error("duplicate validation id {id:?}")]
    DuplicateValidationId { id: crate::graph::ValidationId },
    #[error("validation {id:?} has an empty name")]
    EmptyValidationName { id: crate::graph::ValidationId },
    #[error("duplicate validation name {name:?}")]
    DuplicateValidationName { name: String },
    #[error("validation {id:?} value {value} has dtype {actual}; expected f32")]
    ValidationDtype {
        id: crate::graph::ValidationId,
        value: ValueId,
        actual: DType,
    },
    #[error("validation {id:?} value {value} has zero lanes")]
    EmptyValidationValue {
        id: crate::graph::ValidationId,
        value: ValueId,
    },
    #[error("validation {id:?} value {value} shape {shape:?} overflows usize")]
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
    #[error("input {value} out of range for {value_count} values")]
    InputOutOfRange { value: ValueId, value_count: usize },
    #[error("duplicate input binder {value}")]
    DuplicateInput { value: ValueId },
    #[error("input binder {value} has Device storage")]
    DeviceInput { value: ValueId },
    #[error("const table {value} out of range for {value_count} values")]
    ConstOutOfRange { value: ValueId, value_count: usize },
    #[error("duplicate const table entry {value}")]
    DuplicateConst { value: ValueId },
    #[error("const table {value} is not an input binder")]
    ConstNotInput { value: ValueId },
    #[error("const table {value} has {storage:?} storage; expected Const or State")]
    ConstStorageMismatch { value: ValueId, storage: Storage },
    #[error("slot table {value} out of range for {value_count} values")]
    SlotOutOfRange { value: ValueId, value_count: usize },
    #[error("duplicate slot table entry {value}")]
    DuplicateSlot { value: ValueId },
    #[error("slot table {value} is not an input binder")]
    SlotNotInput { value: ValueId },
    #[error("slot table {value} kind {table_slot:?} does not match storage {storage:?}")]
    SlotStorageMismatch {
        value: ValueId,
        table_slot: Slot,
        storage: Storage,
    },
    #[error("input binder {value} with {storage:?} storage is missing from const table")]
    MissingConstBinding { value: ValueId, storage: Storage },
    #[error("input binder {value} with {slot:?} slot storage is missing from slot table")]
    MissingSlotBinding { value: ValueId, slot: Slot },
    #[error("eqn {equation}: operand {value} used before def")]
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
    #[error("eqn {equation} ({operation}): output {value} out of range for {value_count} values")]
    EquationOutputOutOfRange {
        equation: usize,
        operation: String,
        value: ValueId,
        value_count: usize,
    },
    #[error("eqn {equation} ({operation}): output {value} redefines an existing SSA value")]
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
    #[error("output {value} not defined")]
    OutputNotDefined { value: ValueId },
    #[error("validation {id:?} output {value} not defined")]
    ValidationOutputNotDefined {
        id: crate::graph::ValidationId,
        value: ValueId,
    },
    #[error("export {id:?} value {value} out of range for {value_count} values")]
    ExportValueOutOfRange {
        id: crate::graph::BoundaryId,
        value: ValueId,
        value_count: usize,
    },
    #[error("export id {id:?} declared twice")]
    DuplicateExportId { id: crate::graph::BoundaryId },
    #[error("export {id:?} value {value} not defined")]
    ExportNotDefined {
        id: crate::graph::BoundaryId,
        value: ValueId,
    },
    #[error("state pair ({state_input}, {state_output}) out of range for {value_count} values")]
    StatePairOutOfRange {
        state_input: ValueId,
        state_output: ValueId,
        value_count: usize,
    },
    #[error("duplicate state destination {value}")]
    DuplicateStateDestination { value: ValueId },
    #[error("state_in {value} has {storage:?} storage; expected State")]
    StateInputStorageMismatch { value: ValueId, storage: Storage },
    #[error("state_in {value} is not an input binder")]
    StateInputNotInput { value: ValueId },
    #[error("state_out {value} not defined")]
    StateOutputNotDefined { value: ValueId },
    #[error(
        "state pair ({state_input}, {state_output}) type mismatch: {input_type} vs {output_type}"
    )]
    StateTypeMismatch {
        state_input: ValueId,
        state_output: ValueId,
        input_type: TensorType,
        output_type: TensorType,
    },
    #[error("state input {value} has no StateRole")]
    MissingStateRole { value: ValueId },
    #[error("{value} has a StateRole but {storage:?} storage; expected State")]
    UnexpectedStateRole { value: ValueId, storage: Storage },
    #[error(
        "state input {value} declares Positional axis {axis} out of range for rank-{rank} value"
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
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
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

    #[error("all_gather of {world} ranks overflows axis {axis} of length {len}")]
    AllGatherOverflow {
        axis: usize,
        len: usize,
        world: NonZeroU32,
    },

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

    /// An operand has fewer axes than the op reads: it has a last axis (or a leading one) to take and
    /// none exists.
    #[error("{op} operand {operand} has rank {actual}; it needs rank >= {min}")]
    RankTooLow {
        op: &'static str,
        operand: &'static str,
        min: usize,
        actual: usize,
    },

    /// An operand's rank must be exactly `expected`, or equal another operand's.
    #[error("{op} operand {operand} has rank {actual}; expected rank {expected}")]
    OperandRank {
        op: &'static str,
        operand: &'static str,
        expected: usize,
        actual: usize,
    },

    /// One operand's extent on `dim` does not agree with the extent the op's other operands fix.
    #[error("{op} operand {operand} has extent {actual} on dim {dim}; expected {expected}")]
    OperandDim {
        op: &'static str,
        operand: &'static str,
        dim: usize,
        expected: usize,
        actual: usize,
    },

    /// A fused region's operand names something the region does not define.
    #[error("fused region operand: {0}")]
    FusedOperand(FusedOperandReason),

    /// A fused region's `pack` list is not a set of unit-last-axis locals of one type.
    #[error("fused region pack member local {local} has type {actual}; expected {expected}")]
    FusedPack {
        local: usize,
        expected: Box<TensorType>,
        actual: Box<TensorType>,
    },

    /// A contraction epilogue's region result is not the head's shape (the epilogue never reshapes).
    #[error("contraction epilogue region result {region:?} is not the head result {head:?}")]
    EpilogueShape {
        head: Vec<usize>,
        region: Vec<usize>,
    },

    #[error("dtype mismatch: {a:?} vs {b:?}")]
    Dtype { a: TensorType, b: TensorType },

    #[error("iota of {len} elements is not an exact {dtype} range")]
    IotaRange { len: usize, dtype: DType },

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

    #[error("packed stack {role:?} operand must be {expected:?}, got {actual:?}")]
    PackedStackOperand {
        role: SourceRole,
        expected: Box<TensorType>,
        actual: Box<TensorType>,
    },

    #[error("an indexed packed contraction needs at least one expert")]
    PackedStackExperts,

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

    #[error(
        "a contraction epilogue's region must read the head result (local 0) and carry no pack (packed: {packed})"
    )]
    ContractionEpilogueRegion { packed: bool },

    #[error("dense contraction weight must be rank-2 [N, K] in checkpoint order, got {actual:?}")]
    DenseContractionWeightRank { actual: TensorType },

    #[error("grouped dense contraction: {rows} weight rows do not divide into {groups} groups")]
    GroupedDenseContractionGroups { groups: usize, rows: usize },

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
        "attention q must be rank-4 [B, Hq, Lq, D] with n_rep={n_rep} dividing Hq, got {actual:?}"
    )]
    AttentionQuery { n_rep: usize, actual: Vec<usize> },

    #[error("attention {operand} must be {expected:?}, got {actual:?}")]
    AttentionOperand {
        operand: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("attention {operand} dtype {actual} must match q's {expected}")]
    AttentionDtype {
        operand: &'static str,
        expected: DType,
        actual: DType,
    },

    #[error("scan: {0}")]
    Scan(ScanReason),

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

/// Why a fused region's operand or output reference is invalid ([`ShapeError::FusedOperand`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FusedOperandReason {
    #[error("local {local} is not defined here (only locals 0..{defined} exist)")]
    LocalNotDefined { local: usize, defined: usize },
    #[error("pack lane reads external input {input} of a region with {inputs} inputs")]
    PackLaneInput { input: usize, inputs: usize },
    #[error("pack lane {lane} is outside the last axis (width {width}) of external input {input}")]
    PackLaneWidth {
        input: usize,
        lane: usize,
        width: usize,
    },
    #[error("a row region has no pack-lane operand")]
    PackLaneInRowRegion,
}

/// Why operands are not a scan of their family ([`ShapeError::Scan`]).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ScanReason {
    #[error("every operand is F32")]
    NotF32,
    #[error("operand 0 must be [1,H,L,W], got {shape:?}")]
    LeadOperand { shape: Vec<usize> },
    #[error("extents {shape:?} must be nonzero and chunk {chunk} a power of two")]
    Extents { shape: Vec<usize>, chunk: usize },
    #[error("the state has no rows")]
    NoStateRows,
    #[error("{name} must be {want:?}, got {got:?}")]
    Operand {
        name: &'static str,
        want: Vec<usize>,
        got: Vec<usize>,
    },
}
