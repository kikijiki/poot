//! Typed errors for the graph IR.

use std::num::NonZeroU32;

use crate::graph::{Slot, Storage, ValueId};
use crate::types::{DType, TensorType};
use poot_quant::SourceRole;

/// A named graph storage class used in transactional append diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuilderValueNamespace {
    /// Input binders (`Const`, `State` and `Slot` values) by name.
    Input,
    /// Constants staged by name.
    Constant,
    /// Slot inputs by name.
    Slot,
    /// The result value of an equation.
    EquationResult,
    /// Any value carrying a name, regardless of how it was introduced.
    NamedValue,
}

/// A graph collection whose checked growth or reservation failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuilderCollection {
    /// The value table.
    Values,
    /// The input binder list.
    Inputs,
    /// The constant binder list.
    Constants,
    /// The slot binder list.
    Slots,
    /// The equation list.
    Equations,
    /// The bytes of staged and stored names.
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
    /// A handle minted by a different builder was passed to this one.
    #[error("handle {value} was minted by another Builder and cannot be used with this one")]
    ForeignHandle {
        /// The foreign handle's value id.
        value: ValueId,
    },
    /// A `usize` (an axis, extent or index) does not fit the `i32` an inline literal carries.
    #[error("literal {value} does not fit an I32 literal")]
    LiteralOutOfRange {
        /// The value that did not fit.
        value: usize,
    },
    /// A value id is not in the builder's value table.
    #[error("value {value} is not in the builder's value table")]
    UnknownValue {
        /// The unknown value id.
        value: ValueId,
    },
    /// An input declared an index range but its dtype is not I32.
    #[error("input {name:?} declares a range but is {dtype}; a range bounds I32 words")]
    RangeNotI32 {
        /// The input's name.
        name: String,
        /// The dtype it was declared with.
        dtype: DType,
    },
    /// Two occurrences of one slot kind were declared under the same tag, which a per-name binder could not tell apart.
    #[error(
        "duplicate tag {tag:?} for slot {slot:?}: a per-name binder would bind both ids from one buffer"
    )]
    DuplicateSlotTag {
        /// The slot kind.
        slot: Slot,
        /// The repeated tag.
        tag: String,
    },
    /// A composition's own precondition on its operands or parameters does not hold.
    #[error("{operation}: {requirement}")]
    Requirement {
        /// The composition that stated the precondition.
        operation: &'static str,
        /// The precondition that does not hold.
        requirement: Requirement,
    },
    /// The finished graph is not well formed, or does not carry what a resumed graph carried.
    #[error("the finished graph is invalid: {0}")]
    Invalid(Box<crate::graph::EditError>),
    /// An append plan was committed to a builder other than the one that created it.
    #[error("append plan belongs to a different Builder")]
    ForeignBuilder,
    /// A name must not be empty.
    #[error("{namespace:?} name must not be empty")]
    EmptyName {
        /// The kind of name that was empty.
        namespace: BuilderValueNamespace,
    },
    /// A name is already taken in a namespace that shares it.
    #[error("name {name:?} requested for {requested:?} collides with existing {existing:?}")]
    NameCollision {
        /// The colliding name.
        name: String,
        /// The namespace the name was requested for.
        requested: BuilderValueNamespace,
        /// The namespace that already holds the name.
        existing: BuilderValueNamespace,
    },
    /// A staged input was declared with `Device` storage, which an input binder cannot have.
    #[error("staged input {name:?} has Device storage")]
    DeviceInput {
        /// The staged input's name.
        name: String,
    },
    /// A staged equation result has a storage other than `Device`.
    #[error("staged equation result {value} has non-Device storage {storage:?}")]
    EquationResultStorage {
        /// The staged result value.
        value: ValueId,
        /// The storage it was declared with.
        storage: Storage,
    },
    /// A staged equation operand is not a value of the base graph or the plan.
    #[error("staged equation {equation} ({operation}) references invalid operand {value}")]
    InvalidOperand {
        /// The index of the staged equation within the plan.
        equation: usize,
        /// The name of the equation's operation.
        operation: String,
        /// The invalid operand.
        value: ValueId,
    },
    /// A staged equation reads a value before the equation that defines it.
    #[error("staged equation {equation} ({operation}) uses {value} before its definition")]
    OperandNotDefined {
        /// The index of the staged equation within the plan.
        equation: usize,
        /// The name of the equation's operation.
        operation: String,
        /// The operand used too early.
        value: ValueId,
    },
    /// A staged equation defines a value that already has a definition.
    #[error("staged equation {equation} ({operation}) redefines {value}")]
    ValueRedefined {
        /// The index of the staged equation within the plan.
        equation: usize,
        /// The name of the equation's operation.
        operation: String,
        /// The value defined twice.
        value: ValueId,
    },
    /// Shape inference refused a staged equation.
    #[error("staged equation {equation} ({operation}): {source}")]
    Inference {
        /// The index of the staged equation within the plan.
        equation: usize,
        /// The name of the equation's operation.
        operation: String,
        /// The inference failure.
        #[source]
        source: ShapeError,
    },
    /// A staged equation's stored result type differs from the type inferred from its operands.
    #[error("staged equation {equation} ({operation}) stores {stored}, inferred {inferred}")]
    OutputTypeMismatch {
        /// The index of the staged equation within the plan.
        equation: usize,
        /// The name of the equation's operation.
        operation: String,
        /// The type the result was staged with.
        stored: TensorType,
        /// The type inference computed.
        inferred: TensorType,
    },
    /// A staged `Device` value has no equation that defines it.
    #[error("staged Device value {value} has no equation definition")]
    MissingDefinition {
        /// The undefined value.
        value: ValueId,
    },
    /// The declared append result is not a value this plan produced.
    #[error("append result {value} is not one produced staged value")]
    InvalidResult {
        /// The declared value.
        value: ValueId,
    },
    /// The plan finished without declaring a result.
    #[error("append plan does not declare a result")]
    MissingResult,
    /// A second result was declared on a plan that already has one.
    #[error("append plan already declares result {value}")]
    ResultAlreadyDeclared {
        /// The result declared first.
        value: ValueId,
    },
    /// Growing a collection by `additional` entries overflows `usize`.
    #[error("checked {collection:?} growth overflow: current={current}, additional={additional}")]
    SizeOverflow {
        /// The collection that would overflow.
        collection: BuilderCollection,
        /// Its current length.
        current: usize,
        /// The requested growth.
        additional: usize,
    },
    /// The allocator refused to reserve room in a collection.
    #[error("could not reserve {additional} entries in {collection:?}")]
    Reserve {
        /// The collection that could not grow.
        collection: BuilderCollection,
        /// The number of entries requested.
        additional: usize,
    },
    /// The builder changed after the plan was created, so the plan's snapshot no longer matches it.
    #[error("append plan is stale: expected generation {expected}, actual {actual}")]
    StalePlan {
        /// The builder generation the plan was created at.
        expected: u64,
        /// The builder's current generation.
        actual: u64,
    },
    /// The plan was already committed and cannot be committed again.
    #[error("append plan has already been committed")]
    ConsumedPlan,
    /// The builder's generation counter cannot advance.
    #[error("builder generation overflow")]
    GenerationOverflow,
    /// An exact I32 input's shape has an element count that overflows `usize`.
    #[error("exact I32 input {name:?} shape {shape:?} overflows its element count")]
    ElementCountOverflow {
        /// The input's name.
        name: String,
        /// The declared shape.
        shape: Vec<usize>,
    },
    /// An exact I32 slot was declared with an empty tag.
    #[error("exact I32 slot {slot:?} tag must not be empty")]
    EmptySlotTag {
        /// The slot kind.
        slot: Slot,
    },
    /// An operand has a type other than the one the composition requires.
    #[error("operand {value} has type {actual}, expected {expected}")]
    OperandType {
        /// The operand.
        value: ValueId,
        /// The required type.
        expected: TensorType,
        /// The operand's type.
        actual: TensorType,
    },
}

/// A precondition a composition in [`crate::ops`] states about its operands or parameters.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum Requirement {
    /// An operand has the wrong dtype.
    #[error("operand dtype must be {expected}, found {found}")]
    Dtype {
        /// The required dtype.
        expected: DType,
        /// The operand's dtype.
        found: DType,
    },
    /// The last axis is not the `[low, high]` pair of a two-limb u64 value.
    #[error("operand shape must end in [low, high], found {found:?}")]
    TrailingWordPair {
        /// The operand's shape.
        found: Vec<usize>,
    },
    /// The operand has too few axes.
    #[error("operand must have at least {at_least} axes, found shape {found:?}")]
    MinRank {
        /// The minimum rank.
        at_least: usize,
        /// The operand's shape.
        found: Vec<usize>,
    },
    /// The operand's rank is not the required one.
    #[error("operand must have exactly {expected} axes, found shape {found:?}")]
    Rank {
        /// The required rank.
        expected: usize,
        /// The operand's shape.
        found: Vec<usize>,
    },
    /// An axis has an extent other than the required one.
    #[error("operand axis {axis} must be {expected}, found shape {found:?}")]
    Axis {
        /// The axis checked.
        axis: usize,
        /// The required extent.
        expected: usize,
        /// The operand's shape.
        found: Vec<usize>,
    },
    /// Two operands must share a shape and do not.
    #[error("operand shapes must match, found {left:?} and {right:?}")]
    ShapesMatch {
        /// The first operand's shape.
        left: Vec<usize>,
        /// The second operand's shape.
        right: Vec<usize>,
    },
    /// The composition needs at least one expert and has none.
    #[error("needs at least one expert")]
    NoExperts,
    /// The routed `k` is outside `1..=experts`.
    #[error("needs 1 <= k <= E, got k={k}, E={experts}")]
    TopKRange {
        /// The requested number of experts per token.
        k: usize,
        /// The number of experts available.
        experts: usize,
    },
    /// The composition is the single-token decode path and was given more than one token.
    #[error("is the decode (one token) path, got {tokens} tokens")]
    DecodeTokens {
        /// The token count supplied.
        tokens: usize,
    },
    /// The last two axes must be equal and are not.
    #[error("last two axes must be square, found shape {found:?}")]
    SquareTrailingAxes {
        /// The operand's shape.
        found: Vec<usize>,
    },
    /// A block size must be a power of two.
    #[error("block size {size} must be a power of two")]
    PowerOfTwo {
        /// The refused size.
        size: usize,
    },
    /// One extent must be a multiple of another.
    #[error("{value} must be a multiple of {of}")]
    Multiple {
        /// The extent that was checked.
        value: usize,
        /// The divisor it must be a multiple of.
        of: usize,
    },
    /// The section list does not have one entry per section.
    #[error("needs one entry per section ({expected}), got {found}")]
    SectionCount {
        /// The number of sections.
        expected: usize,
        /// The number of entries supplied.
        found: usize,
    },
    /// The section widths do not sum to the rotated width.
    #[error("section widths must sum to {expected}, got {found}")]
    SectionWidths {
        /// The width they must sum to.
        expected: usize,
        /// The sum supplied.
        found: usize,
    },
    /// A convolution window or carried state has the wrong shape.
    #[error("window must be {expected:?}, found {found:?}")]
    WindowShape {
        /// The required shape.
        expected: Vec<usize>,
        /// The supplied shape.
        found: Vec<usize>,
    },
    /// A count or extent must be at least 1.
    #[error("{what} must be at least 1")]
    AtLeastOne {
        /// Which parameter was zero.
        what: &'static str,
    },
    /// The operand is a scalar where a tensor of rank 1 or more is required.
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
    /// The graph declares no primary result.
    #[error("the graph declares no primary result")]
    MissingResult,
    /// The graph declares more than one primary result.
    #[error("the graph declares more than one primary result")]
    MultipleResults,
    /// A validation witness names a value id past the value table.
    #[error("validation output {value} out of range for {value_count} values")]
    ValidationValueOutOfRange {
        /// The witness's value id.
        value: ValueId,
        /// The number of values in the graph.
        value_count: usize,
    },
    /// Two witnesses share one validation id.
    #[error("duplicate validation id {id:?}")]
    DuplicateValidationId {
        /// The repeated id.
        id: crate::graph::ValidationId,
    },
    /// A witness has an empty name.
    #[error("validation {id:?} has an empty name")]
    EmptyValidationName {
        /// The witness's id.
        id: crate::graph::ValidationId,
    },
    /// Two witnesses share one name.
    #[error("duplicate validation name {name:?}")]
    DuplicateValidationName {
        /// The repeated name.
        name: String,
    },
    /// A witness value is not F32.
    #[error("validation {id:?} value {value} has dtype {actual}; expected f32")]
    ValidationDtype {
        /// The witness's id.
        id: crate::graph::ValidationId,
        /// The witness's value.
        value: ValueId,
        /// The value's dtype.
        actual: DType,
    },
    /// A witness value has zero lanes, so it checks nothing.
    #[error("validation {id:?} value {value} has zero lanes")]
    EmptyValidationValue {
        /// The witness's id.
        id: crate::graph::ValidationId,
        /// The witness's value.
        value: ValueId,
    },
    /// A witness value's element count overflows `usize`.
    #[error("validation {id:?} value {value} shape {shape:?} overflows usize")]
    ValidationShapeOverflow {
        /// The witness's id.
        id: crate::graph::ValidationId,
        /// The witness's value.
        value: ValueId,
        /// The witness value's shape.
        shape: Vec<usize>,
    },
    /// The packet's total lane count overflows `usize`.
    #[error("validation packet lane accounting overflowed")]
    ValidationPacketLaneOverflow,
    /// The packet's byte size overflows `usize`.
    #[error("validation packet byte accounting overflowed")]
    ValidationPacketByteOverflow,
    /// The packet exceeds the validation packet byte budget.
    #[error("validation packet is {byte_len} bytes; maximum is {max_bytes}")]
    ValidationPacketTooLarge {
        /// The packet's size in bytes.
        byte_len: usize,
        /// The budget.
        max_bytes: usize,
    },
    /// An input binder names a value id past the value table.
    #[error("input {value} out of range for {value_count} values")]
    InputOutOfRange {
        /// The binder's value id.
        value: ValueId,
        /// The number of values in the graph.
        value_count: usize,
    },
    /// A value is listed twice in the input binder table.
    #[error("duplicate input binder {value}")]
    DuplicateInput {
        /// The repeated binder.
        value: ValueId,
    },
    /// An input binder has `Device` storage.
    #[error("input binder {value} has Device storage")]
    DeviceInput {
        /// The binder.
        value: ValueId,
    },
    /// A const table entry names a value id past the value table.
    #[error("const table {value} out of range for {value_count} values")]
    ConstOutOfRange {
        /// The entry's value id.
        value: ValueId,
        /// The number of values in the graph.
        value_count: usize,
    },
    /// A value is listed twice in the const table.
    #[error("duplicate const table entry {value}")]
    DuplicateConst {
        /// The repeated entry.
        value: ValueId,
    },
    /// A const table entry is not an input binder.
    #[error("const table {value} is not an input binder")]
    ConstNotInput {
        /// The entry.
        value: ValueId,
    },
    /// A const table entry has storage other than `Const` or `State`.
    #[error("const table {value} has {storage:?} storage; expected Const or State")]
    ConstStorageMismatch {
        /// The entry.
        value: ValueId,
        /// Its storage.
        storage: Storage,
    },
    /// A slot table entry names a value id past the value table.
    #[error("slot table {value} out of range for {value_count} values")]
    SlotOutOfRange {
        /// The entry's value id.
        value: ValueId,
        /// The number of values in the graph.
        value_count: usize,
    },
    /// A value is listed twice in the slot table.
    #[error("duplicate slot table entry {value}")]
    DuplicateSlot {
        /// The repeated entry.
        value: ValueId,
    },
    /// A slot table entry is not an input binder.
    #[error("slot table {value} is not an input binder")]
    SlotNotInput {
        /// The entry.
        value: ValueId,
    },
    /// A slot table entry's kind disagrees with its value's storage.
    #[error("slot table {value} kind {table_slot:?} does not match storage {storage:?}")]
    SlotStorageMismatch {
        /// The entry.
        value: ValueId,
        /// The kind the slot table records.
        table_slot: Slot,
        /// The value's storage.
        storage: Storage,
    },
    /// A `Const` or `State` input binder is missing from the const table.
    #[error("input binder {value} with {storage:?} storage is missing from const table")]
    MissingConstBinding {
        /// The binder.
        value: ValueId,
        /// Its storage.
        storage: Storage,
    },
    /// A `Slot` input binder is missing from the slot table.
    #[error("input binder {value} with {slot:?} slot storage is missing from slot table")]
    MissingSlotBinding {
        /// The binder.
        value: ValueId,
        /// Its slot kind.
        slot: Slot,
    },
    /// An equation reads a value before any equation or binder defines it.
    #[error("eqn {equation}: operand {value} used before def")]
    EquationOperandNotDefined {
        /// The index of the equation.
        equation: usize,
        /// The operand used too early.
        value: ValueId,
    },
    /// A binary equation's two value operands have different dtypes.
    #[error(
        "eqn {equation} ({operation}): two value operands have different dtypes ({left} vs {right})"
    )]
    BinaryValueDtypeMismatch {
        /// The index of the equation.
        equation: usize,
        /// The name of the equation's operation.
        operation: String,
        /// The first operand's dtype.
        left: DType,
        /// The second operand's dtype.
        right: DType,
    },
    /// An equation's output id is past the value table.
    #[error("eqn {equation} ({operation}): output {value} out of range for {value_count} values")]
    EquationOutputOutOfRange {
        /// The index of the equation.
        equation: usize,
        /// The name of the equation's operation.
        operation: String,
        /// The output id.
        value: ValueId,
        /// The number of values in the graph.
        value_count: usize,
    },
    /// An equation's output was already defined by a binder or an earlier equation.
    #[error("eqn {equation} ({operation}): output {value} redefines an existing SSA value")]
    EquationOutputRedefined {
        /// The index of the equation.
        equation: usize,
        /// The name of the equation's operation.
        operation: String,
        /// The output defined twice.
        value: ValueId,
    },
    /// Shape inference refused an equation.
    #[error("eqn {equation} ({operation}): {source}")]
    EquationInference {
        /// The index of the equation.
        equation: usize,
        /// The name of the equation's operation.
        operation: String,
        /// The inference failure.
        #[source]
        source: ShapeError,
    },
    /// An equation's stored result type differs from the type inferred from its operands.
    #[error("eqn {equation} ({operation}): stored aval {stored} != inferred {inferred}")]
    EquationOutputTypeMismatch {
        /// The index of the equation.
        equation: usize,
        /// The name of the equation's operation.
        operation: String,
        /// The type stored in the value table.
        stored: TensorType,
        /// The type inference computed.
        inferred: TensorType,
    },
    /// The primary result is never defined.
    #[error("output {value} not defined")]
    OutputNotDefined {
        /// The result value.
        value: ValueId,
    },
    /// A witness value is never defined.
    #[error("validation {id:?} output {value} not defined")]
    ValidationOutputNotDefined {
        /// The witness's id.
        id: crate::graph::ValidationId,
        /// The witness's value.
        value: ValueId,
    },
    /// An export names a value id past the value table.
    #[error("export {id:?} value {value} out of range for {value_count} values")]
    ExportValueOutOfRange {
        /// The export's boundary id.
        id: crate::graph::BoundaryId,
        /// The value id.
        value: ValueId,
        /// The number of values in the graph.
        value_count: usize,
    },
    /// Two exports share one boundary id.
    #[error("export id {id:?} declared twice")]
    DuplicateExportId {
        /// The repeated id.
        id: crate::graph::BoundaryId,
    },
    /// An exported value is never defined.
    #[error("export {id:?} value {value} not defined")]
    ExportNotDefined {
        /// The export's boundary id.
        id: crate::graph::BoundaryId,
        /// The value.
        value: ValueId,
    },
    /// A carried-state pair names a value id past the value table.
    #[error("state pair ({state_input}, {state_output}) out of range for {value_count} values")]
    StatePairOutOfRange {
        /// The pair's input value id.
        state_input: ValueId,
        /// The pair's output value id.
        state_output: ValueId,
        /// The number of values in the graph.
        value_count: usize,
    },
    /// Two state pairs name the same input, so one buffer would be overwritten twice per step.
    #[error("duplicate state destination {value}")]
    DuplicateStateDestination {
        /// The input that appears in two pairs.
        value: ValueId,
    },
    /// A state pair's input does not have `State` storage.
    #[error("state_in {value} has {storage:?} storage; expected State")]
    StateInputStorageMismatch {
        /// The input.
        value: ValueId,
        /// Its storage.
        storage: Storage,
    },
    /// A state pair's input is not an input binder.
    #[error("state_in {value} is not an input binder")]
    StateInputNotInput {
        /// The input.
        value: ValueId,
    },
    /// A state pair's output value is never defined.
    #[error("state_out {value} not defined")]
    StateOutputNotDefined {
        /// The output.
        value: ValueId,
    },
    /// A state pair's input and output have different types.
    #[error(
        "state pair ({state_input}, {state_output}) type mismatch: {input_type} vs {output_type}"
    )]
    StateTypeMismatch {
        /// The pair's input value.
        state_input: ValueId,
        /// The pair's output value.
        state_output: ValueId,
        /// The input's type.
        input_type: TensorType,
        /// The output's type.
        output_type: TensorType,
    },
    /// A `State` input carries no declared role.
    #[error("state input {value} has no StateRole")]
    MissingStateRole {
        /// The input.
        value: ValueId,
    },
    /// A value that is not `State` carries a state role.
    #[error("{value} has a StateRole but {storage:?} storage; expected State")]
    UnexpectedStateRole {
        /// The value.
        value: ValueId,
        /// Its storage.
        storage: Storage,
    },
    /// A `Positional` role names an axis past the value's rank.
    #[error(
        "state input {value} declares Positional axis {axis} out of range for rank-{rank} value"
    )]
    StateRoleAxis {
        /// The `State` input.
        value: ValueId,
        /// The declared axis.
        axis: u8,
        /// The input's rank.
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
    /// The two shapes have no numpy broadcast (or, for `Broadcast`, the source would have to shrink).
    #[error("cannot broadcast shapes {a:?} and {b:?}")]
    Broadcast {
        /// The first shape (the source for `Broadcast`).
        a: Vec<usize>,
        /// The second shape (the target for `Broadcast`).
        b: Vec<usize>,
    },

    /// The target shape holds a different number of elements than the source.
    #[error("reshape changes element count: {from:?} ({from_numel}) -> {to:?} ({to_numel})")]
    Reshape {
        /// The source shape.
        from: Vec<usize>,
        /// The requested shape.
        to: Vec<usize>,
        /// The source element count.
        from_numel: usize,
        /// The requested element count.
        to_numel: usize,
    },

    /// A shape's element count overflows `usize`.
    #[error("reshape element count overflows usize for shape {shape:?}")]
    ReshapeElementCountOverflow {
        /// The shape.
        shape: Vec<usize>,
    },

    /// An axis index is not below the operand's rank.
    #[error("axis {axis} out of range for rank-{rank} tensor")]
    Axis {
        /// The axis requested.
        axis: usize,
        /// The operand's rank.
        rank: usize,
    },

    /// Multiplying the gather axis by the world size overflows `usize`.
    #[error("all_gather of {world} ranks overflows axis {axis} of length {len}")]
    AllGatherOverflow {
        /// The gather axis.
        axis: usize,
        /// The axis extent before gathering.
        len: usize,
        /// The number of ranks.
        world: NonZeroU32,
    },

    /// The permutation's length differs from the operand's rank.
    #[error("transpose perm {perm:?} does not match rank {rank}")]
    Perm {
        /// The permutation.
        perm: Vec<usize>,
        /// The operand's rank.
        rank: usize,
    },

    /// The slice bounds are not within the axis.
    #[error("slice {start}..{end} out of range for axis of length {len}")]
    Slice {
        /// The first index taken.
        start: usize,
        /// One past the last index taken.
        end: usize,
        /// The axis extent.
        len: usize,
    },

    /// Two parts disagree on an axis other than the concatenation axis.
    #[error("concat: non-axis dim {dim} mismatch ({a} vs {b})")]
    Concat {
        /// The disagreeing axis.
        dim: usize,
        /// The first part's extent on it.
        a: usize,
        /// The other part's extent on it.
        b: usize,
    },

    /// A matmul operand has fewer than two axes.
    #[error("matmul needs rank >= 2 operands, got {a:?} and {b:?}")]
    MatMulRank {
        /// The left operand's shape.
        a: Vec<usize>,
        /// The right operand's shape.
        b: Vec<usize>,
    },

    /// The left operand's last axis differs from the right operand's second-to-last.
    #[error("matmul contracting dim mismatch: {a:?} x {b:?}")]
    MatMulContract {
        /// The left operand's shape.
        a: Vec<usize>,
        /// The right operand's shape.
        b: Vec<usize>,
    },

    /// The update does not fit inside the operand at the given axis.
    #[error(
        "dynamic_update_slice: update {update:?} does not fit operand {operand:?} on axis {axis}"
    )]
    UpdateSlice {
        /// The operand's shape.
        operand: Vec<usize>,
        /// The update's shape.
        update: Vec<usize>,
        /// The axis the update is written along.
        axis: usize,
    },

    /// The operation was given a different number of inputs than it takes.
    #[error("op expected {expected} inputs, got {got}")]
    Arity {
        /// The operation's operand count.
        expected: usize,
        /// The number of inputs supplied.
        got: usize,
    },

    /// An operand has fewer axes than the op reads: it has a last axis (or a leading one) to take and
    /// none exists.
    #[error("{op} operand {operand} has rank {actual}; it needs rank >= {min}")]
    RankTooLow {
        /// The operation's name.
        op: &'static str,
        /// The operand's role name.
        operand: &'static str,
        /// The least rank the operation accepts.
        min: usize,
        /// The operand's rank.
        actual: usize,
    },

    /// An operand's rank must be exactly `expected`, or equal another operand's.
    #[error("{op} operand {operand} has rank {actual}; expected rank {expected}")]
    OperandRank {
        /// The operation's name.
        op: &'static str,
        /// The operand's role name.
        operand: &'static str,
        /// The required rank.
        expected: usize,
        /// The operand's rank.
        actual: usize,
    },

    /// One operand's extent on `dim` does not agree with the extent the op's other operands fix.
    #[error("{op} operand {operand} has extent {actual} on dim {dim}; expected {expected}")]
    OperandDim {
        /// The operation's name.
        op: &'static str,
        /// The operand's role name.
        operand: &'static str,
        /// The axis checked.
        dim: usize,
        /// The extent the other operands fix.
        expected: usize,
        /// The operand's extent.
        actual: usize,
    },

    /// A fused region's operand names something the region does not define.
    #[error("fused region operand: {0}")]
    FusedOperand(FusedOperandReason),

    /// A fused region's `pack` list is not a set of unit-last-axis locals of one type.
    #[error("fused region pack member local {local} has type {actual}; expected {expected}")]
    FusedPack {
        /// The pack member's local index.
        local: usize,
        /// The type the pack requires at this member: the first member's shape with a unit last axis, or the first member's type.
        expected: Box<TensorType>,
        /// The member's type.
        actual: Box<TensorType>,
    },

    /// A contraction epilogue's region result is not the head's shape (the epilogue never reshapes).
    #[error("contraction epilogue region result {region:?} is not the head result {head:?}")]
    EpilogueShape {
        /// The head contraction's result shape.
        head: Vec<usize>,
        /// The region's result shape.
        region: Vec<usize>,
    },

    /// Two operands that must share a dtype do not.
    #[error("dtype mismatch: {a:?} vs {b:?}")]
    Dtype {
        /// The first operand's type.
        a: TensorType,
        /// The second operand's type.
        b: TensorType,
    },

    /// An iota of this length is not exactly representable in the dtype.
    #[error("iota of {len} elements is not an exact {dtype} range")]
    IotaRange {
        /// The requested length.
        len: usize,
        /// The requested dtype.
        dtype: DType,
    },

    /// A block top-k keeps `k` of the `blocks` blocks of `ratio` positions: `ratio >= 1`, `1 <= k <= blocks`.
    #[error(
        "block top-k needs ratio >= 1 and 1 <= k <= blocks, got ratio {ratio}, k {k}, blocks {blocks}"
    )]
    BlockTopK {
        /// The positions each block covers.
        ratio: usize,
        /// The blocks kept per query.
        k: usize,
        /// The blocks the scores rank.
        blocks: usize,
    },

    /// The operation does not accept the dtype.
    #[error("{op} does not support {dtype}")]
    DtypeOp {
        /// The operation's name.
        op: &'static str,
        /// The refused dtype.
        dtype: DType,
    },

    /// `Cast` admits every other `(from, to)` pair; these nine have no evaluator definition (card 555): `BF16`/`F16` -> `I32`/`I8`, `I32`/`I8` -> `E4M3FN`, `E4M3FN` -> `I32`/`I8`, and
    /// `F32` -> `I8`.
    #[error("cast {from} -> {to} is not admitted")]
    CastUnsupported {
        /// The source dtype.
        from: DType,
        /// The target dtype.
        to: DType,
    },

    /// The two matmul operand dtypes are not an admitted pairing.
    #[error(
        "matmul operand dtypes {a} (a) and {b} (b) are not an allowed pairing (same arithmetic dtype, or F32 a with BF16/F16 b)"
    )]
    MatMulOperandDtype {
        /// The left operand's dtype.
        a: DType,
        /// The right operand's dtype.
        b: DType,
    },

    /// A packed dequant source operand does not have the type the descriptor derives for its role.
    #[error(
        "packed dequant {role:?} operand {field} mismatch: expected {expected:?}, got {actual:?}"
    )]
    PackedDequantOperand {
        /// The source role.
        role: SourceRole,
        /// Which part of the type disagreed.
        field: &'static str,
        /// The type the descriptor requires.
        expected: Box<TensorType>,
        /// The operand's type.
        actual: Box<TensorType>,
    },

    /// A stacked packed source operand does not have the `[experts, slab_bytes]` type its role requires.
    #[error("packed stack {role:?} operand must be {expected:?}, got {actual:?}")]
    PackedStackOperand {
        /// The source role.
        role: SourceRole,
        /// The required type.
        expected: Box<TensorType>,
        /// The operand's type.
        actual: Box<TensorType>,
    },

    /// An indexed packed contraction has an empty expert stack.
    #[error("an indexed packed contraction needs at least one expert")]
    PackedStackExperts,

    /// The activation's last axis is not the descriptor's `K`.
    #[error("packed contraction activation must end in K={expected_k}, got {actual:?}")]
    PackedContractionActivation {
        /// The descriptor's `K`.
        expected_k: usize,
        /// The activation's type.
        actual: TensorType,
    },

    /// A packed contraction operand that must be F32 is not.
    #[error("packed contraction {operand} must use F32, got {actual}")]
    PackedContractionDtype {
        /// The operand's role name.
        operand: &'static str,
        /// Its dtype.
        actual: DType,
    },

    /// The block count is zero or does not divide the output width.
    #[error("packed contraction blocks={blocks} does not evenly divide out={out}")]
    PackedContractionBlocks {
        /// The output width.
        out: usize,
        /// The block count.
        blocks: usize,
    },

    /// The dense weight dtype is not one the contraction admits.
    #[error(
        "dense contraction weight dtype {actual} is not admitted (see DENSE_CONTRACTION_WEIGHT_DTYPES)"
    )]
    DenseContractionWeightDtype {
        /// The weight's dtype.
        actual: DType,
    },

    /// A dense contraction operand has the wrong dtype.
    #[error("dense contraction {operand} dtype is wrong: got {actual}")]
    DenseContractionDtype {
        /// The operand's role name.
        operand: &'static str,
        /// Its dtype.
        actual: DType,
    },

    /// The epilogue region does not read the head result as local 0, or carries a pack.
    #[error(
        "a contraction epilogue's region must read the head result (local 0) and carry no pack (packed: {packed})"
    )]
    ContractionEpilogueRegion {
        /// Whether the region has a non-empty pack.
        packed: bool,
    },

    /// The dense weight is not rank 2.
    #[error("dense contraction weight must be rank-2 [N, K] in checkpoint order, got {actual:?}")]
    DenseContractionWeightRank {
        /// The weight's type.
        actual: TensorType,
    },

    /// The weight rows do not divide evenly into the groups.
    #[error("grouped dense contraction: {rows} weight rows do not divide into {groups} groups")]
    GroupedDenseContractionGroups {
        /// The group count.
        groups: usize,
        /// The weight's row count.
        rows: usize,
    },

    /// The activation's last axis is not the weight's `K`.
    #[error("dense contraction activation must end in K={expected_k}, got {actual:?}")]
    DenseContractionActivation {
        /// The weight's `K`.
        expected_k: usize,
        /// The activation's type.
        actual: TensorType,
    },

    /// The gather table's dtype is not one the row gather admits.
    #[error(
        "dense row gather source dtype {actual} is not admitted (see DENSE_ROW_GATHER_SOURCE_DTYPES)"
    )]
    DenseRowGatherSourceDtype {
        /// The table's dtype.
        actual: DType,
    },

    /// A row gather operand has the wrong dtype.
    #[error("dense row gather {operand} dtype is wrong: got {actual}")]
    DenseRowGatherDtype {
        /// The operand's role name.
        operand: &'static str,
        /// Its dtype.
        actual: DType,
    },

    /// The gather table is not rank 2.
    #[error("dense row gather table must be rank-2 [V, R], got {actual:?}")]
    DenseRowGatherTableRank {
        /// The table's type.
        actual: TensorType,
    },

    /// `k` exceeds the extent of the axis it selects from.
    #[error("arg_top_k: k={k} exceeds the expert axis extent {extent}")]
    TopK {
        /// The requested count.
        k: usize,
        /// The axis extent.
        extent: usize,
    },

    /// The query is not rank-4 `[B, Hq, Lq, D]` with `n_rep` dividing `Hq`.
    #[error(
        "attention q must be rank-4 [B, Hq, Lq, D] with n_rep={n_rep} dividing Hq, got {actual:?}"
    )]
    AttentionQuery {
        /// The query heads per key/value head.
        n_rep: usize,
        /// The query's shape.
        actual: Vec<usize>,
    },

    /// An attention operand's shape is not the one the query fixes.
    #[error("attention {operand} must be {expected:?}, got {actual:?}")]
    AttentionOperand {
        /// The operand's role name.
        operand: &'static str,
        /// The required shape.
        expected: Vec<usize>,
        /// The operand's shape.
        actual: Vec<usize>,
    },

    /// A selected attention keeps `k >= 1` blocks of `ratio >= 1` positions per query.
    #[error("selected attention needs ratio >= 1 and k >= 1, got ratio {ratio}, k {k}")]
    SelectedAttention {
        /// The positions each block covers.
        ratio: usize,
        /// The blocks each query reads.
        k: usize,
    },

    /// An attention operand's dtype differs from the query's.
    #[error("attention {operand} dtype {actual} must match q's {expected}")]
    AttentionDtype {
        /// The operand's role name.
        operand: &'static str,
        /// The query's dtype.
        expected: DType,
        /// The operand's dtype.
        actual: DType,
    },

    /// The operands are not a scan of their family.
    #[error("scan: {0}")]
    Scan(ScanReason),

    /// The rotary width is odd, zero or past the head dimension.
    #[error("rope rot={rot} must be even and within 1..=D for x {x:?}")]
    RopeWidth {
        /// The requested rotary width.
        rot: usize,
        /// The rotated input's shape.
        x: Vec<usize>,
    },

    /// A cos/sin table cannot broadcast into the rotated input without growing it.
    #[error(
        "rope {operand} {actual:?} must broadcast into the rotated x {rotated:?} without growing it"
    )]
    RopeTable {
        /// The table's role name.
        operand: &'static str,
        /// The rotated slice's shape.
        rotated: Vec<usize>,
        /// The table's shape.
        actual: Vec<usize>,
    },

    /// A rope table's dtype differs from the input's.
    #[error("rope {operand} dtype {actual} must match x's {expected}")]
    RopeDtype {
        /// The table's role name.
        operand: &'static str,
        /// The input's dtype.
        expected: DType,
        /// The table's dtype.
        actual: DType,
    },

    /// The seed is not I32.
    #[error("random_uniform seed must be I32, got {actual}")]
    RandomUniformSeedDtype {
        /// The seed's dtype.
        actual: DType,
    },

    /// A sampling operand has the wrong dtype.
    #[error("sample_token {operand} dtype must be {expected}, got {actual}")]
    SampleTokenDtype {
        /// The operand's role name.
        operand: &'static str,
        /// The required dtype.
        expected: DType,
        /// The operand's dtype.
        actual: DType,
    },

    /// A sampling operand's shape does not match the logits it samples.
    #[error("sample_token {operand} shape {actual:?} does not match the expected {expected:?}")]
    SampleTokenShape {
        /// The operand's role name.
        operand: &'static str,
        /// The required shape.
        expected: Vec<usize>,
        /// The operand's shape.
        actual: Vec<usize>,
    },
}

/// Why a fused region's operand or output reference is invalid ([`ShapeError::FusedOperand`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FusedOperandReason {
    /// A step reads a local that is not defined at that point.
    #[error("local {local} is not defined here (only locals 0..{defined} exist)")]
    LocalNotDefined {
        /// The local index read.
        local: usize,
        /// How many locals exist there.
        defined: usize,
    },
    /// A pack lane names an external input past the region's inputs.
    #[error("pack lane reads external input {input} of a region with {inputs} inputs")]
    PackLaneInput {
        /// The input index.
        input: usize,
        /// The region's input count.
        inputs: usize,
    },
    /// A pack lane indexes past the last axis of its external input.
    #[error("pack lane {lane} is outside the last axis (width {width}) of external input {input}")]
    PackLaneWidth {
        /// The input index.
        input: usize,
        /// The lane index.
        lane: usize,
        /// The input's last-axis extent.
        width: usize,
    },
    /// A row region has no pack-lane operand, but one was used.
    #[error("a row region has no pack-lane operand")]
    PackLaneInRowRegion,
}

/// Why operands are not a scan of their family ([`ShapeError::Scan`]).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ScanReason {
    /// Every scan operand must be F32.
    #[error("every operand is F32")]
    NotF32,
    /// The first operand is not `[1, H, L, W]`.
    #[error("operand 0 must be [1,H,L,W], got {shape:?}")]
    LeadOperand {
        /// The operand's shape.
        shape: Vec<usize>,
    },
    /// An extent is zero or the chunk is not a power of two.
    #[error("extents {shape:?} must be nonzero and chunk {chunk} a power of two")]
    Extents {
        /// The first operand's shape.
        shape: Vec<usize>,
        /// The chunk length.
        chunk: usize,
    },
    /// The carried state has no rows.
    #[error("the state has no rows")]
    NoStateRows,
    /// An operand's shape is not the one the family fixes.
    #[error("{name} must be {want:?}, got {got:?}")]
    Operand {
        /// The operand's role name.
        name: &'static str,
        /// The required shape.
        want: Vec<usize>,
        /// The supplied shape.
        got: Vec<usize>,
    },
}
