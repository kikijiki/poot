//! The model's input schema (ADR-0117): what a model's programs read, carry and require, and the
//! request values that satisfy it.
//!
//! A model returns a [`ModelSchema`] beside its tracer. The driver binds every input of a traced
//! program by *kind*, never by family:
//!
//! - a **step kind** ([`StepInput`]): a value the driver derives from the step's geometry and its own
//!   bookkeeping, the same for every family. The set is closed because each member is a driver
//!   mechanism; its slot key is fixed here ([`StepInput::slot_key`]);
//! - a **request kind** ([`RequestInput`], named by an [`InputId`]): a value a request supplies, produced
//!   by family code (the family's front end). The set is open: a family declares as many as it needs, and
//!   one generic routine of the driver binds them all from [`InputValue`] data.
//!
//! A [`Presence`] says whether a request may omit a request input a program reads, and a request
//! input's *cadence* ([`Cadence`]), fixed by its form, says which steps of a sequence bind it. The
//! checks a request value must pass live here too ([`ModelSchema::check_request`]), so the driver runs
//! them once at admission and the step path binds without validating.

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_graph_ir::{Slot, SlotKey, StateRole};
use poot_tensor::{DType, HostTensor};

use crate::model::{FamilyKey, ModelError, ModelOutput};

/// `Slot::Activation` tag of the real-token mask ([`StepInput::RealTokens`]), `[rows, tokens]` I32:
/// 1 for a fed token, 0 for padding.
pub const REAL_TOKENS_TAG: &str = "attention_mask";
/// `Slot::SlotMap` tag of a paged step's read map ([`StepInput::KvReadMap`]).
const KV_READ_TAG: &str = "read";
/// `Slot::WriteMap` tag of a paged step's write map ([`StepInput::KvWriteMap`]).
const KV_WRITE_TAG: &str = "write";
/// `Slot::Activation` tag of a paged step's `[rows]` state read index ([`StepInput::StateReadRows`]).
pub const STATE_READ_TAG: &str = "state_read";
/// `Slot::WriteMap` tag of a paged step's `[rows]` state write map ([`StepInput::StateWriteRows`]).
pub const STATE_WRITE_TAG: &str = "state_write";

/// The most request inputs a schema declares: a step's carried set is a 64-bit mask.
pub const MAX_REQUEST_INPUTS: usize = 64;

/// A family's own name for one of its traced programs. Opaque to the driver, which reads the
/// programs through [`ModelSchema::step`], [`ModelSchema::encode`] and [`ModelSchema::draft`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProgramId(&'static str);

impl ProgramId {
    /// The id the standard builders give a model's `step` program.
    pub const STEP: Self = Self("step");
    /// The id the standard builders give a model's `encode` program.
    pub const ENCODE: Self = Self("encode");
    /// The id the standard builders give a model's `draft` program.
    pub const DRAFT: Self = Self("draft");

    /// An id displaying as `name`.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// The display name.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for ProgramId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// The identity of one request input inside a schema. A family declares it once (a `const`) and uses
/// it for its slot, its schema entry and the [`RequestInputs`] its front end builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InputId(&'static str);

impl InputId {
    /// An id displaying as `name`.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// The display name, for refusals and receipts.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for InputId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// A value the driver derives from a step's geometry and its own bookkeeping. Closed: each member is a
/// driver mechanism, and every family declares them through the shared step component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StepInput {
    /// The step's token ids, `[rows, tokens]`.
    Tokens,
    /// The absolute position of each token, `[rows, tokens]`.
    Positions,
    /// 1 for a fed token, 0 for padding, `[rows, tokens]`.
    RealTokens,
    /// A paged step's read map: each row's logical position to its pool slot, `[rows, capacity]`.
    KvReadMap,
    /// A paged step's write map: each flat token's pool slot, or -1 for padding, `[rows * tokens]`.
    KvWriteMap,
    /// The recurrent state pool row each step row continues, `[rows]`.
    StateReadRows,
    /// The recurrent state pool row each step row's result is written to, or -1, `[rows]`.
    StateWriteRows,
}

impl StepInput {
    /// The key a program declares this input under.
    pub fn slot_key(self) -> SlotKey {
        match self {
            StepInput::Tokens => SlotKey::new(Slot::Token, None),
            StepInput::Positions => SlotKey::new(Slot::Pos, None),
            StepInput::RealTokens => SlotKey::new(Slot::Activation, Some(REAL_TOKENS_TAG)),
            StepInput::KvReadMap => SlotKey::new(Slot::SlotMap, Some(KV_READ_TAG)),
            StepInput::KvWriteMap => SlotKey::new(Slot::WriteMap, Some(KV_WRITE_TAG)),
            StepInput::StateReadRows => SlotKey::new(Slot::Activation, Some(STATE_READ_TAG)),
            StepInput::StateWriteRows => SlotKey::new(Slot::WriteMap, Some(STATE_WRITE_TAG)),
        }
    }

    /// The step kind `key` is declared as, if it is one.
    pub fn of_slot(key: &SlotKey) -> Option<Self> {
        const ALL: [StepInput; 7] = [
            StepInput::Tokens,
            StepInput::Positions,
            StepInput::RealTokens,
            StepInput::KvReadMap,
            StepInput::KvWriteMap,
            StepInput::StateReadRows,
            StepInput::StateWriteRows,
        ];
        ALL.into_iter().find(|kind| kind.slot_key() == *key)
    }
}

/// How a column's positions are laid out in a [`Per::Token`] input's slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// `[rows * tokens, width]`: one `width`-wide vector per token.
    TokenMajor,
    /// `[width, rows * tokens]`: one `rows * tokens`-long column per axis.
    AxisMajor,
}

/// What a padding column or element of a request input binds. Padding writes nothing and its outputs
/// are dropped, so the value only has to be finite and in range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PadValue {
    /// An I32 pad.
    I32(i32),
    /// An F32 pad.
    F32(f32),
}

impl PadValue {
    fn dtype(self) -> DType {
        match self {
            PadValue::I32(_) => DType::I32,
            PadValue::F32(_) => DType::F32,
        }
    }
}

/// How a request input's rows are shaped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Per {
    /// One `width`-wide vector per sequence position.
    ///
    /// `presence: None` (dense): cadence [`Cadence::Every`]; the slot is in every variant of the
    /// program. `presence: Some(slot)` (sparse): cadence [`Cadence::Covers`]; the slot and its I32
    /// presence companion are in the variant a step runs when some row's fed window holds a position the
    /// value covers.
    Token {
        /// Values per position.
        width: usize,
        /// How the positions sit in the slot.
        layout: Layout,
        /// The I32 `[rows, tokens]` slot that is 1 where the value covers the position.
        presence: Option<SlotKey>,
    },
    /// One tensor per row: cadence [`Cadence::AtStart`]; the slot is in the variant a step runs when it
    /// feeds position 0 of some row (every step of an `encode` program does). The slot is
    /// `[rows, extent, ..trailing]` with the leading axis padded to the step's row extent, or
    /// `[rows, ..trailing]` when `padded` is false.
    Row {
        /// The value's axes after the leading one.
        trailing: Vec<usize>,
        /// Whether the value has a leading axis padded to the step's row extent.
        padded: bool,
        /// The I32 `[rows, extent]` slot that is 1 per real element of a padded value.
        valid: Option<SlotKey>,
    },
}

/// Which steps of a sequence bind a request input, fixed by its form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cadence {
    /// Every step: the slot is in every variant of the program.
    Every,
    /// The steps whose fed window holds a position the value covers.
    Covers,
    /// Only the step that feeds a row's position 0.
    AtStart,
}

/// The form of one request input: its element type, its row shape and its pad.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestForm {
    dtype: DType,
    per: Per,
    pad: PadValue,
}

impl RequestForm {
    /// A form of `dtype` elements, shaped `per`, padded with `pad`.
    pub fn new(dtype: DType, per: Per, pad: PadValue) -> Self {
        Self { dtype, per, pad }
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The row shape.
    pub fn per(&self) -> &Per {
        &self.per
    }

    /// The pad value.
    pub fn pad(&self) -> PadValue {
        self.pad
    }

    /// The cadence the form fixes.
    pub fn cadence(&self) -> Cadence {
        match &self.per {
            Per::Token { presence: None, .. } => Cadence::Every,
            Per::Token {
                presence: Some(_), ..
            } => Cadence::Covers,
            Per::Row { .. } => Cadence::AtStart,
        }
    }
}

/// One request input a schema declares.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestInput {
    id: InputId,
    slot: SlotKey,
    form: RequestForm,
}

impl RequestInput {
    /// The input `id`, declared by a program as `slot`, in `form`.
    pub fn new(id: InputId, slot: SlotKey, form: RequestForm) -> Self {
        Self { id, slot, form }
    }

    /// The input's id.
    pub fn id(&self) -> InputId {
        self.id
    }

    /// The slot a program declares the value under.
    pub fn slot(&self) -> &SlotKey {
        &self.slot
    }

    /// The input's form.
    pub fn form(&self) -> &RequestForm {
        &self.form
    }
}

/// What a request value is at a position its support does not cover (and what a defaulted input binds
/// when a request omits it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fill {
    /// The form's pad value; a sparse input's presence is 0 there.
    Pad,
    /// Every axis at the token's absolute position (I32 only).
    Positions,
    /// `from[axis] + step * (position - support_end)` on each axis (I32 only): a sequence that
    /// continues from where its prompt's values end.
    Advance {
        /// The value of each axis at the first position past the support.
        from: Vec<i32>,
        /// The increment per further position.
        step: i32,
    },
}

/// Whether a request may omit a request input a program reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Presence {
    /// Admission refuses a request without it.
    Required,
    /// A request without it binds the fill (cadence [`Cadence::Every`] only).
    Defaulted(Fill),
    /// A request may omit it (cadence [`Cadence::Covers`] or [`Cadence::AtStart`] only).
    Optional,
}

/// A carried state a program declares: what a traced graph's state pairs are checked against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateSpec {
    name: &'static str,
    role: StateRole,
}

impl StateSpec {
    /// A state group `name` whose graph values are declared with `role`.
    pub const fn new(name: &'static str, role: StateRole) -> Self {
        Self { name, role }
    }

    /// The KV cache of an attention layer stack: positional, indexed by the live position.
    pub const fn kv() -> Self {
        Self::new("kv", StateRole::Positional { axis: 2 })
    }

    /// A state that folds every token into its value (a conv window, a GDN or SSM matrix).
    pub const fn recurrent(name: &'static str) -> Self {
        Self::new(name, StateRole::Recurrent)
    }

    /// The hidden-state carry a later program of the same driver step reads.
    pub const fn hand_off() -> Self {
        Self::new("hidden_carry", StateRole::Handoff)
    }

    /// The group's name, for refusals and receipts.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// The role the group's graph values carry.
    pub const fn role(&self) -> StateRole {
        self.role
    }
}

/// One traced entry point of a model: what it returns, which request inputs it reads and which states
/// it carries.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramSpec {
    id: ProgramId,
    output: ModelOutput,
    reads: Vec<(InputId, Presence)>,
    states: Vec<StateSpec>,
    positions: Option<NonZeroUsize>,
}

impl ProgramSpec {
    /// A program `id` returning `output`, reading no request input and carrying no state.
    pub fn new(id: ProgramId, output: ModelOutput) -> Self {
        Self {
            id,
            output,
            reads: Vec::new(),
            states: Vec::new(),
            positions: None,
        }
    }

    /// This program always traces exactly `positions` positions per row (a vision tower of a fixed
    /// grid), whatever its rows carry.
    #[must_use]
    pub fn with_positions(mut self, positions: NonZeroUsize) -> Self {
        self.positions = Some(positions);
        self
    }

    /// This program also reads request input `input` with `presence`.
    #[must_use]
    pub fn reading(mut self, input: InputId, presence: Presence) -> Self {
        self.reads.push((input, presence));
        self
    }

    /// This program also carries `state`.
    #[must_use]
    pub fn carrying(mut self, state: StateSpec) -> Self {
        self.states.push(state);
        self
    }

    /// The program's id.
    pub fn id(&self) -> ProgramId {
        self.id
    }

    /// What the program returns per row.
    pub fn output(&self) -> ModelOutput {
        self.output
    }

    /// The request inputs the program reads, with whether a request may omit each.
    pub fn reads(&self) -> &[(InputId, Presence)] {
        &self.reads
    }

    /// The states the program carries.
    pub fn states(&self) -> &[StateSpec] {
        &self.states
    }

    /// The positions per row the program traces, when it fixes them.
    pub fn positions(&self) -> Option<NonZeroUsize> {
        self.positions
    }

    /// How the program treats `input`: `None` when it does not read it.
    pub fn presence_of(&self, input: InputId) -> Option<&Presence> {
        self.reads
            .iter()
            .find(|(read, _)| *read == input)
            .map(|(_, presence)| presence)
    }
}

/// The set of carriable request inputs (cadence [`Cadence::Covers`] or [`Cadence::AtStart`]) a step
/// binds, as a mask over a schema's inputs in declaration order. It is part of a prepared entry's key:
/// a step runs the program variant that declares exactly these slots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct InputSet(u64);

impl InputSet {
    /// No carriable input.
    pub const EMPTY: Self = Self(0);

    /// Whether the set is empty.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The set with bit `index` added.
    #[must_use]
    pub fn with(self, index: usize) -> Self {
        Self(self.0 | 1 << index)
    }

    /// Whether bit `index` is set.
    pub fn contains(self, index: usize) -> bool {
        self.0 >> index & 1 == 1
    }
}

/// How the driver binds one slot of a program, resolved against a schema.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlotBinding<'a> {
    /// A step kind.
    Step(StepInput),
    /// A request input's value, presence companion or valid companion.
    Request(&'a RequestInput, RequestPart),
}

/// Which slot of a request input is meant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestPart {
    /// The value itself.
    Value,
    /// A sparse input's presence companion.
    Presence,
    /// A padded row input's valid companion.
    Valid,
}

/// Why a schema is malformed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SchemaError {
    /// More request inputs than a carried set holds.
    #[error("{count} request inputs; a schema declares at most {MAX_REQUEST_INPUTS}")]
    TooManyInputs {
        /// The inputs declared.
        count: usize,
    },
    /// Two request inputs share an id.
    #[error("request input {0} is declared twice")]
    DuplicateInput(InputId),
    /// Two declared slots (inputs, presence or valid companions, step kinds) share a key.
    #[error("slot {0} is declared twice")]
    DuplicateSlot(String),
    /// A program reads an input the schema does not declare.
    #[error("program {program} reads undeclared input {input}")]
    UnknownInput {
        /// The program.
        program: ProgramId,
        /// The undeclared input.
        input: InputId,
    },
    /// A program reads one input twice.
    #[error("program {program} reads input {input} twice")]
    ReadTwice {
        /// The program.
        program: ProgramId,
        /// The input read twice.
        input: InputId,
    },
    /// Two programs share an id.
    #[error("two programs are named {0}")]
    DuplicateProgram(ProgramId),
    /// A pad, fill, presence or valid slot of another type than the form needs.
    #[error("input {input}: {what}")]
    Form {
        /// The input.
        input: InputId,
        /// What is wrong with the form.
        what: &'static str,
    },
    /// A `Defaulted` input whose cadence is not `Every`.
    #[error("input {input} is defaulted but is not read at every step")]
    DefaultedNotEvery {
        /// The input.
        input: InputId,
    },
    /// An `Optional` input whose cadence is `Every`: a program variant without the slot cannot exist.
    #[error("input {input} is optional but is read at every step")]
    OptionalEvery {
        /// The input.
        input: InputId,
    },
    /// A defaulted input's fill that its form cannot express.
    #[error("input {input}: the fill does not fit the form")]
    FillForm {
        /// The input.
        input: InputId,
    },
}

/// The model-owned description of what its programs read, carry and require. Built once with the
/// model and immutable.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSchema {
    step: Option<ProgramSpec>,
    encode: Option<ProgramSpec>,
    draft: Option<ProgramSpec>,
    inputs: Vec<RequestInput>,
}

/// Builds a [`ModelSchema`], checking it once.
#[derive(Clone, Debug, Default)]
pub struct ModelSchemaBuilder {
    step: Option<ProgramSpec>,
    encode: Option<ProgramSpec>,
    draft: Option<ProgramSpec>,
    inputs: Vec<RequestInput>,
}

impl ModelSchema {
    /// A schema under construction.
    pub fn builder() -> ModelSchemaBuilder {
        ModelSchemaBuilder::default()
    }

    /// The program that continues a sequence's carried state and returns logits; `None` for a model
    /// that serves no sequence.
    pub fn step(&self) -> Option<&ProgramSpec> {
        self.step.as_ref()
    }

    /// The stateless pass over whole rows (hidden states or scores per position); `None` for a model
    /// that offers none.
    pub fn encode(&self) -> Option<&ProgramSpec> {
        self.encode.as_ref()
    }

    /// The draft head over the hidden-state carry; `None` for a model that has none.
    pub fn draft(&self) -> Option<&ProgramSpec> {
        self.draft.as_ref()
    }

    /// The program named `id`.
    pub fn program(&self, id: ProgramId) -> Option<&ProgramSpec> {
        [&self.step, &self.encode, &self.draft]
            .into_iter()
            .flatten()
            .find(|program| program.id() == id)
    }

    /// Every request input the schema declares, in declaration order.
    pub fn inputs(&self) -> &[RequestInput] {
        &self.inputs
    }

    /// The request input `id`.
    pub fn input(&self, id: InputId) -> Option<&RequestInput> {
        self.inputs.iter().find(|input| input.id == id)
    }

    /// The bit `id` takes in an [`InputSet`].
    pub fn index_of(&self, id: InputId) -> Option<usize> {
        self.inputs.iter().position(|input| input.id == id)
    }

    /// The ids in `set`, in declaration order.
    pub fn ids_of(&self, set: InputSet) -> Vec<InputId> {
        self.inputs
            .iter()
            .enumerate()
            .filter(|(index, _)| set.contains(*index))
            .map(|(_, input)| input.id)
            .collect()
    }

    /// The carriable inputs `program` reads (cadence `Covers` or `AtStart`), as a set.
    pub fn carriable(&self, program: &ProgramSpec) -> InputSet {
        program
            .reads()
            .iter()
            .filter_map(|(id, _)| {
                let index = self.index_of(*id)?;
                (self.inputs[index].form.cadence() != Cadence::Every).then_some(index)
            })
            .fold(InputSet::EMPTY, InputSet::with)
    }

    /// How a slot of a traced program is bound, if it is a step kind or a request input of this
    /// schema.
    pub fn binding_of(&self, key: &SlotKey) -> Option<SlotBinding<'_>> {
        if let Some(kind) = StepInput::of_slot(key) {
            return Some(SlotBinding::Step(kind));
        }
        self.inputs.iter().find_map(|input| {
            if input.slot == *key {
                return Some(SlotBinding::Request(input, RequestPart::Value));
            }
            match &input.form.per {
                Per::Token {
                    presence: Some(slot),
                    ..
                } if slot == key => Some(SlotBinding::Request(input, RequestPart::Presence)),
                Per::Row {
                    valid: Some(slot), ..
                } if slot == key => Some(SlotBinding::Request(input, RequestPart::Valid)),
                _ => None,
            }
        })
    }

    /// Check `inputs`, a request's values, against `program` before the request holds anything:
    /// every id is declared and read by the program, none is given twice, every `Required` input is
    /// present, and each value has its form. `extent` is the number of positions a dense value covers
    /// and a sparse one lies inside: the prompt's token count.
    pub fn check_request(
        &self,
        program: &ProgramSpec,
        inputs: &RequestInputs,
        extent: usize,
    ) -> Result<(), InputError> {
        let mut seen = BTreeSet::new();
        for (id, value) in inputs.iter() {
            let input = self.input(id).ok_or(InputError::Unknown { id })?;
            if program.presence_of(id).is_none() {
                return Err(InputError::NotRead {
                    id,
                    program: program.id,
                });
            }
            if !seen.insert(id) {
                return Err(InputError::Duplicate { id });
            }
            check_value(input, value, extent)?;
        }
        for (id, presence) in program.reads() {
            if *presence == Presence::Required && !seen.contains(id) {
                return Err(InputError::Missing { id: *id });
            }
        }
        Ok(())
    }
}

impl ModelSchemaBuilder {
    /// [`Self::build`], reporting a defect as `family`'s [`ModelError::Schema`].
    pub fn build_for(self, family: FamilyKey) -> Result<ModelSchema, ModelError> {
        self.build()
            .map_err(|source| ModelError::Schema { family, source })
    }

    /// The model's `step` program.
    #[must_use]
    pub fn step(mut self, program: ProgramSpec) -> Self {
        self.step = Some(program);
        self
    }

    /// The model's `encode` program.
    #[must_use]
    pub fn encode(mut self, program: ProgramSpec) -> Self {
        self.encode = Some(program);
        self
    }

    /// The model's `draft` program.
    #[must_use]
    pub fn draft(mut self, program: ProgramSpec) -> Self {
        self.draft = Some(program);
        self
    }

    /// A request input a program of the model reads.
    #[must_use]
    pub fn input(mut self, input: RequestInput) -> Self {
        self.inputs.push(input);
        self
    }

    /// Check the schema: no id or slot twice, every read declared, every form and presence coherent.
    pub fn build(self) -> Result<ModelSchema, SchemaError> {
        let Self {
            step,
            encode,
            draft,
            inputs,
        } = self;
        if inputs.len() > MAX_REQUEST_INPUTS {
            return Err(SchemaError::TooManyInputs {
                count: inputs.len(),
            });
        }
        let mut slots: Vec<SlotKey> = Vec::new();
        let mut claim = |key: &SlotKey| {
            if StepInput::of_slot(key).is_some() || slots.contains(key) {
                return Err(SchemaError::DuplicateSlot(key.to_string()));
            }
            slots.push(key.clone());
            Ok(())
        };
        for (i, input) in inputs.iter().enumerate() {
            if inputs[..i].iter().any(|earlier| earlier.id == input.id) {
                return Err(SchemaError::DuplicateInput(input.id));
            }
            claim(&input.slot)?;
            check_form(input)?;
            match &input.form.per {
                Per::Token {
                    presence: Some(slot),
                    ..
                }
                | Per::Row {
                    valid: Some(slot), ..
                } => claim(slot)?,
                _ => {}
            }
        }
        let schema = ModelSchema {
            step,
            encode,
            draft,
            inputs,
        };
        let mut ids = Vec::new();
        for program in [&schema.step, &schema.encode, &schema.draft]
            .into_iter()
            .flatten()
        {
            if ids.contains(&program.id) {
                return Err(SchemaError::DuplicateProgram(program.id));
            }
            ids.push(program.id);
            for (i, (id, presence)) in program.reads.iter().enumerate() {
                let input = schema.input(*id).ok_or(SchemaError::UnknownInput {
                    program: program.id,
                    input: *id,
                })?;
                if program.reads[..i].iter().any(|(earlier, _)| earlier == id) {
                    return Err(SchemaError::ReadTwice {
                        program: program.id,
                        input: *id,
                    });
                }
                let cadence = input.form.cadence();
                match presence {
                    Presence::Required => {}
                    Presence::Defaulted(fill) => {
                        if cadence != Cadence::Every {
                            return Err(SchemaError::DefaultedNotEvery { input: *id });
                        }
                        check_fill(input, fill)?;
                    }
                    Presence::Optional => {
                        if cadence == Cadence::Every {
                            return Err(SchemaError::OptionalEvery { input: *id });
                        }
                    }
                }
            }
        }
        Ok(schema)
    }
}

fn check_form(input: &RequestInput) -> Result<(), SchemaError> {
    let form_error = |what| SchemaError::Form {
        input: input.id,
        what,
    };
    let form = &input.form;
    if !matches!(form.dtype, DType::I32 | DType::F32) {
        return Err(form_error("a request input is I32 or F32"));
    }
    if form.pad.dtype() != form.dtype {
        return Err(form_error("the pad value has another type than the input"));
    }
    if let Per::Token { width: 0, .. } = form.per {
        return Err(form_error("a token input is at least one value wide"));
    }
    if let Per::Row {
        valid: Some(_),
        padded: false,
        ..
    } = form.per
    {
        return Err(form_error("only a padded row input has a valid companion"));
    }
    Ok(())
}

fn check_fill(input: &RequestInput, fill: &Fill) -> Result<(), SchemaError> {
    let Per::Token { width, .. } = &input.form.per else {
        return Err(SchemaError::FillForm { input: input.id });
    };
    let fits = match fill {
        Fill::Pad => true,
        Fill::Positions => input.form.dtype == DType::I32,
        Fill::Advance { from, .. } => input.form.dtype == DType::I32 && from.len() == *width,
    };
    if fits {
        Ok(())
    } else {
        Err(SchemaError::FillForm { input: input.id })
    }
}

/// The positions a token value covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Support {
    /// Positions `0..len`.
    Dense {
        /// The number of positions covered.
        len: usize,
    },
    /// These positions, strictly ascending.
    Sparse(Vec<usize>),
}

impl Support {
    /// How many positions the support covers.
    pub fn len(&self) -> usize {
        match self {
            Support::Dense { len } => *len,
            Support::Sparse(positions) => positions.len(),
        }
    }

    /// Whether the support covers nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The index into the value's data of `position`, if the support covers it.
    pub fn index_of(&self, position: usize) -> Option<usize> {
        match self {
            Support::Dense { len } => (position < *len).then_some(position),
            Support::Sparse(positions) => positions.binary_search(&position).ok(),
        }
    }

    /// Whether any position of `start..end` is covered.
    pub fn intersects(&self, start: usize, end: usize) -> bool {
        match self {
            Support::Dense { len } => start < (*len).min(end),
            Support::Sparse(positions) => {
                let first = positions.partition_point(|&p| p < start);
                positions.get(first).is_some_and(|&p| p < end)
            }
        }
    }

    /// The position just past the last one covered: where an [`Fill::Advance`] starts counting.
    pub fn end(&self) -> usize {
        match self {
            Support::Dense { len } => *len,
            Support::Sparse(positions) => positions.last().map_or(0, |&p| p + 1),
        }
    }
}

/// Per-token values over sequence positions.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenValues {
    support: Support,
    data: HostTensor,
    beyond: Fill,
}

impl TokenValues {
    /// Values for positions `0..len`: `data` is `[len, width]`.
    pub fn dense(data: HostTensor) -> Result<Self, ValueError> {
        let len = data.shape().first().copied().ok_or(ValueError::Shape)?;
        Self::new(Support::Dense { len }, data)
    }

    /// Values at `positions` (strictly ascending): `data` is `[positions.len(), width]`.
    pub fn sparse(positions: Vec<usize>, data: HostTensor) -> Result<Self, ValueError> {
        if positions.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ValueError::Unordered);
        }
        Self::new(Support::Sparse(positions), data)
    }

    fn new(support: Support, data: HostTensor) -> Result<Self, ValueError> {
        if data.shape().len() != 2 || data.shape()[0] != support.len() {
            return Err(ValueError::Shape);
        }
        if !matches!(data.dtype(), DType::I32 | DType::F32) {
            return Err(ValueError::Dtype);
        }
        Ok(Self {
            support,
            data,
            beyond: Fill::Pad,
        })
    }

    /// These values with `beyond` at the positions the support does not cover.
    #[must_use]
    pub fn with_beyond(mut self, beyond: Fill) -> Self {
        self.beyond = beyond;
        self
    }

    /// The positions covered.
    pub fn support(&self) -> &Support {
        &self.support
    }

    /// The values, `[support.len(), width]`.
    pub fn data(&self) -> &HostTensor {
        &self.data
    }

    /// What a position outside the support holds.
    pub fn beyond(&self) -> &Fill {
        &self.beyond
    }

    /// Values per position.
    pub fn width(&self) -> usize {
        self.data.shape()[1]
    }
}

/// A request input's value.
#[derive(Clone, Debug, PartialEq)]
pub enum InputValue {
    /// Per-token values over sequence positions.
    Tokens(TokenValues),
    /// One per-row tensor.
    Row(HostTensor),
}

/// Why a value cannot be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ValueError {
    /// The data is not `[support.len(), width]`.
    #[error("the data is not [support, width]")]
    Shape,
    /// The data is neither I32 nor F32.
    #[error("a request value is I32 or F32")]
    Dtype,
    /// Sparse positions that are not strictly ascending.
    #[error("sparse positions must be strictly ascending")]
    Unordered,
}

/// The request values a sequence or an encode row carries. Immutable and cheap to clone: the
/// children of a forked sequence share their parent's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RequestInputs(Arc<Vec<(InputId, InputValue)>>);

impl RequestInputs {
    /// The inputs of a request that carries none.
    pub fn none() -> Self {
        Self::default()
    }

    /// These inputs plus `id`'s `value`. A second value for one id is refused at admission.
    #[must_use]
    pub fn with(mut self, id: InputId, value: InputValue) -> Self {
        Arc::make_mut(&mut self.0).push((id, value));
        self
    }

    /// The value of `id`, if the request carries one.
    pub fn get(&self, id: InputId) -> Option<&InputValue> {
        self.0
            .iter()
            .find(|(held, _)| *held == id)
            .map(|(_, value)| value)
    }

    /// Every id and value, in the order given.
    pub fn iter(&self) -> impl Iterator<Item = (InputId, &InputValue)> {
        self.0.iter().map(|(id, value)| (*id, value))
    }

    /// Whether the request carries nothing.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Why admission refuses a request's values.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InputError {
    /// The schema does not declare the input.
    #[error("the model declares no input {id}")]
    Unknown {
        /// The undeclared id.
        id: InputId,
    },
    /// The program does not read the input.
    #[error("program {program} does not read input {id}")]
    NotRead {
        /// The id.
        id: InputId,
        /// The program.
        program: ProgramId,
    },
    /// The request gives the input twice.
    #[error("input {id} is given twice")]
    Duplicate {
        /// The id.
        id: InputId,
    },
    /// A required input is absent.
    #[error("input {id} is required and absent")]
    Missing {
        /// The id.
        id: InputId,
    },
    /// A value whose type, width or trailing shape is not its form's.
    #[error("input {id}: {what}")]
    Form {
        /// The id.
        id: InputId,
        /// What differs from the form.
        what: &'static str,
    },
    /// A dense support whose length is not the prompt's, or a sparse position outside the prompt.
    #[error("input {id}: {what}")]
    Support {
        /// The id.
        id: InputId,
        /// What is wrong with the support.
        what: &'static str,
    },
}

fn check_value(input: &RequestInput, value: &InputValue, extent: usize) -> Result<(), InputError> {
    let id = input.id;
    let form = &input.form;
    let form_error = |what| InputError::Form { id, what };
    match (&form.per, value) {
        (Per::Token { width, .. }, InputValue::Tokens(values)) => {
            if values.data.dtype() != form.dtype {
                return Err(form_error("the values have another type than the input"));
            }
            if values.width() != *width {
                return Err(form_error("the values have another width than the input"));
            }
            match &values.beyond {
                Fill::Pad => {}
                Fill::Positions if form.dtype == DType::I32 => {}
                Fill::Advance { from, .. } if form.dtype == DType::I32 && from.len() == *width => {}
                _ => {
                    return Err(form_error(
                        "the fill beyond the support does not fit the input",
                    ));
                }
            }
            match &values.support {
                Support::Dense { len } if *len != extent => Err(InputError::Support {
                    id,
                    what: "a dense value covers a different number of positions than the prompt",
                }),
                Support::Sparse(positions) if positions.last().is_some_and(|&p| p >= extent) => {
                    Err(InputError::Support {
                        id,
                        what: "a sparse position lies outside the prompt",
                    })
                }
                _ => Ok(()),
            }
        }
        (
            Per::Row {
                trailing, padded, ..
            },
            InputValue::Row(tensor),
        ) => {
            if tensor.dtype() != form.dtype {
                return Err(form_error("the tensor has another type than the input"));
            }
            let shape = tensor.shape();
            let rest = if *padded {
                shape
                    .get(1..)
                    .ok_or(form_error("the tensor has no leading axis"))?
            } else {
                shape
            };
            if rest != trailing.as_slice() {
                return Err(form_error("the tensor's trailing axes are not the input's"));
            }
            Ok(())
        }
        (Per::Token { .. }, InputValue::Row(_)) => {
            Err(form_error("a per-token input was given a row tensor"))
        }
        (Per::Row { .. }, InputValue::Tokens(_)) => {
            Err(form_error("a per-row input was given per-token values"))
        }
    }
}

/// Why a traced graph and the schema disagree about a program.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StateMismatch {
    /// The graph carries a state of a role the program does not declare.
    #[error("the graph carries a {role:?} state the program does not declare")]
    Undeclared {
        /// The undeclared role.
        role: StateRole,
    },
    /// The program declares a state the graph does not carry.
    #[error("the program declares state {name}, which the graph does not carry")]
    Missing {
        /// The declared group.
        name: &'static str,
    },
}

impl ProgramSpec {
    /// Check the roles of a traced graph's state pairs against the declared states: every pair has a
    /// declared group of its kind (positional, blocked, recurrent or handoff; the axis of a positional
    /// cache follows the layout and is not compared), and every declared group has a pair. A hand-off
    /// carry exists only when the step's shape asks for one, so a declared hand-off group may be absent;
    /// the driver checks it against the shape.
    pub fn check_states(
        &self,
        graph: impl IntoIterator<Item = StateRole>,
    ) -> Result<(), StateMismatch> {
        use std::mem::discriminant;
        let roles: Vec<StateRole> = graph.into_iter().collect();
        for &role in &roles {
            if !self
                .states
                .iter()
                .any(|state| discriminant(&state.role) == discriminant(&role))
            {
                return Err(StateMismatch::Undeclared { role });
            }
        }
        for state in &self.states {
            let present = roles
                .iter()
                .any(|role| discriminant(role) == discriminant(&state.role));
            if !present && state.role != StateRole::Handoff {
                return Err(StateMismatch::Missing { name: state.name });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN_TYPE: InputId = InputId::new("types");
    const IMAGE: InputId = InputId::new("image");
    const SOURCE: InputId = InputId::new("source");

    fn dense(id: InputId, tag: &str, width: usize) -> RequestInput {
        RequestInput::new(
            id,
            SlotKey::new(Slot::Activation, Some(tag)),
            RequestForm::new(
                DType::I32,
                Per::Token {
                    width,
                    layout: Layout::TokenMajor,
                    presence: None,
                },
                PadValue::I32(0),
            ),
        )
    }

    fn sparse(id: InputId, tag: &str) -> RequestInput {
        RequestInput::new(
            id,
            SlotKey::new(Slot::Activation, Some(tag)),
            RequestForm::new(
                DType::F32,
                Per::Token {
                    width: 2,
                    layout: Layout::TokenMajor,
                    presence: Some(SlotKey::new(Slot::Activation, Some("image_at"))),
                },
                PadValue::F32(0.0),
            ),
        )
    }

    fn row(id: InputId, tag: &str) -> RequestInput {
        RequestInput::new(
            id,
            SlotKey::new(Slot::Activation, Some(tag)),
            RequestForm::new(
                DType::I32,
                Per::Row {
                    trailing: vec![],
                    padded: true,
                    valid: Some(SlotKey::new(Slot::Activation, Some("source_valid"))),
                },
                PadValue::I32(0),
            ),
        )
    }

    fn step(reads: &[(InputId, Presence)]) -> ProgramSpec {
        reads.iter().cloned().fold(
            ProgramSpec::new(ProgramId::STEP, ModelOutput::Logits { vocab: 4 }),
            |program, (id, presence)| program.reading(id, presence),
        )
    }

    fn schema() -> ModelSchema {
        ModelSchema::builder()
            .input(dense(TOKEN_TYPE, "types", 1))
            .input(sparse(IMAGE, "image"))
            .input(row(SOURCE, "source"))
            .step(step(&[
                (TOKEN_TYPE, Presence::Defaulted(Fill::Pad)),
                (IMAGE, Presence::Optional),
                (SOURCE, Presence::Required),
            ]))
            .build()
            .unwrap()
    }

    fn ints(values: Vec<i32>) -> HostTensor {
        HostTensor::i32(vec![values.len(), 1], values)
    }

    /// A schema is checked once at construction: a duplicate id or slot, an unknown or repeated read, a
    /// defaulted input that is not read at every step, an optional input that is, a pad of another type
    /// and a fill the form cannot express are each a typed `SchemaError`. Mutation: drop one check in
    /// `ModelSchemaBuilder::build`; its case below builds and the row goes red.
    #[test]
    fn a_malformed_schema_is_refused_with_its_typed_cause() {
        let build = |inputs: Vec<RequestInput>, reads: &[(InputId, Presence)]| {
            let builder = inputs
                .into_iter()
                .fold(ModelSchema::builder(), ModelSchemaBuilder::input);
            builder.step(step(reads)).build().unwrap_err()
        };
        assert_eq!(
            build(
                vec![dense(TOKEN_TYPE, "a", 1), dense(TOKEN_TYPE, "b", 1)],
                &[]
            ),
            SchemaError::DuplicateInput(TOKEN_TYPE)
        );
        assert!(matches!(
            build(vec![dense(TOKEN_TYPE, "a", 1), dense(IMAGE, "a", 1)], &[]),
            SchemaError::DuplicateSlot(_)
        ));
        assert!(matches!(
            build(vec![dense(TOKEN_TYPE, "token", 1)], &[(IMAGE, Presence::Optional)]),
            SchemaError::UnknownInput { input, .. } if input == IMAGE
        ));
        assert!(matches!(
            build(
                vec![dense(TOKEN_TYPE, "a", 1)],
                &[
                    (TOKEN_TYPE, Presence::Required),
                    (TOKEN_TYPE, Presence::Required)
                ]
            ),
            SchemaError::ReadTwice { .. }
        ));
        assert_eq!(
            build(
                vec![sparse(IMAGE, "image")],
                &[(IMAGE, Presence::Defaulted(Fill::Pad))]
            ),
            SchemaError::DefaultedNotEvery { input: IMAGE }
        );
        assert_eq!(
            build(
                vec![dense(TOKEN_TYPE, "a", 1)],
                &[(TOKEN_TYPE, Presence::Optional)]
            ),
            SchemaError::OptionalEvery { input: TOKEN_TYPE }
        );
        assert_eq!(
            build(
                vec![dense(TOKEN_TYPE, "a", 2)],
                &[(
                    TOKEN_TYPE,
                    Presence::Defaulted(Fill::Advance {
                        from: vec![0],
                        step: 1
                    })
                )]
            ),
            SchemaError::FillForm { input: TOKEN_TYPE }
        );
        let mut mispadded = dense(TOKEN_TYPE, "a", 1);
        mispadded.form.pad = PadValue::F32(0.0);
        assert!(matches!(
            build(vec![mispadded], &[]),
            SchemaError::Form { .. }
        ));
    }

    /// A request is checked against the program that serves it: an undeclared id, an id the program
    /// does not read, a repeated id, a missing required input, a value of the wrong type or width, a
    /// dense value that does not cover the prompt and a sparse position outside it are each typed.
    /// Mutation: drop one arm of `check_request` or `check_value`; its case goes red.
    #[test]
    fn a_request_is_checked_against_the_program_that_serves_it() {
        let schema = schema();
        let program = schema.step().unwrap();
        let source = InputValue::Row(HostTensor::i32(vec![2], vec![5, 6]));
        let ok = RequestInputs::none().with(SOURCE, source.clone());
        assert_eq!(schema.check_request(program, &ok, 4), Ok(()));

        let check = |inputs: RequestInputs| schema.check_request(program, &inputs, 4);
        let unknown = InputId::new("other");
        assert_eq!(
            check(ok.clone().with(unknown, source.clone())),
            Err(InputError::Unknown { id: unknown })
        );
        assert_eq!(
            check(ok.clone().with(SOURCE, source.clone())),
            Err(InputError::Duplicate { id: SOURCE })
        );
        assert_eq!(
            check(RequestInputs::none()),
            Err(InputError::Missing { id: SOURCE })
        );
        let types =
            |len: usize| InputValue::Tokens(TokenValues::dense(ints(vec![0; len])).unwrap());
        assert!(matches!(
            check(ok.clone().with(TOKEN_TYPE, types(3))),
            Err(InputError::Support { id, .. }) if id == TOKEN_TYPE
        ));
        assert_eq!(check(ok.clone().with(TOKEN_TYPE, types(4))), Ok(()));
        let floats = InputValue::Tokens(
            TokenValues::dense(HostTensor::f32(vec![4, 1], vec![0.0; 4])).unwrap(),
        );
        assert!(matches!(
            check(ok.clone().with(TOKEN_TYPE, floats)),
            Err(InputError::Form { id, .. }) if id == TOKEN_TYPE
        ));
        let rows = |positions: Vec<usize>| {
            let n = positions.len();
            InputValue::Tokens(
                TokenValues::sparse(positions, HostTensor::f32(vec![n, 2], vec![0.0; 2 * n]))
                    .unwrap(),
            )
        };
        assert_eq!(check(ok.clone().with(IMAGE, rows(vec![1, 3]))), Ok(()));
        assert!(matches!(
            check(ok.clone().with(IMAGE, rows(vec![1, 4]))),
            Err(InputError::Support { id, .. }) if id == IMAGE
        ));
        // A per-row value given to a per-token input (and the reverse) is a form defect.
        assert!(matches!(
            check(ok.clone().with(IMAGE, source)),
            Err(InputError::Form { id, .. }) if id == IMAGE
        ));
    }

    /// An input a program does not read is refused for that program even when the schema declares it.
    #[test]
    fn an_input_the_program_does_not_read_is_refused() {
        let schema = ModelSchema::builder()
            .input(dense(TOKEN_TYPE, "types", 1))
            .step(step(&[]))
            .encode(
                ProgramSpec::new(ProgramId::ENCODE, ModelOutput::Hidden { width: 2 })
                    .reading(TOKEN_TYPE, Presence::Defaulted(Fill::Pad)),
            )
            .build()
            .unwrap();
        let inputs = RequestInputs::none().with(
            TOKEN_TYPE,
            InputValue::Tokens(TokenValues::dense(ints(vec![0, 1])).unwrap()),
        );
        assert_eq!(
            schema.check_request(schema.step().unwrap(), &inputs, 2),
            Err(InputError::NotRead {
                id: TOKEN_TYPE,
                program: ProgramId::STEP
            })
        );
        assert_eq!(
            schema.check_request(schema.encode().unwrap(), &inputs, 2),
            Ok(())
        );
    }

    /// The carriable inputs of a program are exactly those whose cadence is not `Every`, and a slot
    /// resolves to its step kind, its input, or its presence and valid companions.
    #[test]
    fn carriable_inputs_and_slot_bindings_follow_the_declared_forms() {
        let schema = schema();
        let carriable = schema.carriable(schema.step().unwrap());
        let index = |id| schema.index_of(id).unwrap();
        assert!(!carriable.contains(index(TOKEN_TYPE)));
        assert!(carriable.contains(index(IMAGE)) && carriable.contains(index(SOURCE)));
        assert_eq!(schema.ids_of(carriable), vec![IMAGE, SOURCE]);

        let bind = |key: SlotKey| schema.binding_of(&key);
        assert!(matches!(
            bind(StepInput::Positions.slot_key()),
            Some(SlotBinding::Step(StepInput::Positions))
        ));
        assert!(matches!(
            bind(SlotKey::new(Slot::Activation, Some("image"))),
            Some(SlotBinding::Request(input, RequestPart::Value)) if input.id() == IMAGE
        ));
        assert!(matches!(
            bind(SlotKey::new(Slot::Activation, Some("image_at"))),
            Some(SlotBinding::Request(input, RequestPart::Presence)) if input.id() == IMAGE
        ));
        assert!(matches!(
            bind(SlotKey::new(Slot::Activation, Some("source_valid"))),
            Some(SlotBinding::Request(input, RequestPart::Valid)) if input.id() == SOURCE
        ));
        assert!(bind(SlotKey::new(Slot::Activation, Some("unknown"))).is_none());
    }

    /// Every step kind's slot key maps back to it, and no two kinds share a key.
    #[test]
    fn step_kinds_round_trip_through_their_slot_keys() {
        let kinds = [
            StepInput::Tokens,
            StepInput::Positions,
            StepInput::RealTokens,
            StepInput::KvReadMap,
            StepInput::KvWriteMap,
            StepInput::StateReadRows,
            StepInput::StateWriteRows,
        ];
        for kind in kinds {
            assert_eq!(StepInput::of_slot(&kind.slot_key()), Some(kind));
        }
        let keys: BTreeSet<String> = kinds.iter().map(|k| k.slot_key().to_string()).collect();
        assert_eq!(keys.len(), kinds.len());
    }

    /// A traced graph's state roles must be the program's declared groups, by kind: an undeclared
    /// kind and a declared group the graph lacks are refused; a declared hand-off may be absent, and a
    /// positional cache's axis follows the layout and is not compared.
    #[test]
    fn a_graphs_states_are_checked_against_the_declared_groups() {
        let program = ProgramSpec::new(ProgramId::STEP, ModelOutput::Logits { vocab: 4 })
            .carrying(StateSpec::kv())
            .carrying(StateSpec::recurrent("ssm"))
            .carrying(StateSpec::hand_off());
        let kv0 = StateRole::Positional { axis: 0 };
        assert_eq!(program.check_states([kv0, StateRole::Recurrent]), Ok(()));
        assert_eq!(
            program.check_states([StateRole::Positional { axis: 2 }, StateRole::Recurrent]),
            Ok(())
        );
        assert_eq!(
            program.check_states([kv0]),
            Err(StateMismatch::Missing { name: "ssm" })
        );
        let blocked = StateRole::Blocked {
            axis: 1,
            ratio: std::num::NonZeroU32::MIN,
        };
        assert_eq!(
            program.check_states([kv0, StateRole::Recurrent, blocked]),
            Err(StateMismatch::Undeclared { role: blocked })
        );
    }

    /// Request inputs are shared by reference: cloning is cheap and a later `with` leaves the original
    /// untouched.
    #[test]
    fn request_inputs_are_immutable_shared_values() {
        let one =
            RequestInputs::none().with(SOURCE, InputValue::Row(HostTensor::i32(vec![1], vec![1])));
        let two = one.clone().with(
            TOKEN_TYPE,
            InputValue::Row(HostTensor::i32(vec![1], vec![0])),
        );
        assert_eq!(one.iter().count(), 1);
        assert_eq!(two.iter().count(), 2);
        assert!(RequestInputs::none().is_empty());
        assert!(one.get(SOURCE).is_some() && one.get(TOKEN_TYPE).is_none());
    }
}
