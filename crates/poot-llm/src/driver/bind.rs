//! Binding by kind (ADR-0117): the one place the driver turns what a step knows into the values a
//! prepared program's slots take.
//!
//! At prepare every slot of a traced program is resolved once ([`resolve`]) to a [`Binder`]:
//!
//! - a **step kind** ([`StepInput`]): a column the driver computes once per step from its own geometry
//!   and bookkeeping ([`StepColumns`]): tokens, positions, the real-token mask, the KV maps and the
//!   recurrent state rows;
//! - a **request kind**: a value the request supplies and the model's schema declares, bound by one
//!   generic routine ([`bind_request`]) that slices the sequence's [`InputValue`] by each row's fed
//!   window and pads the rest;
//! - a **pass kind** ([`PassInput`]): an input a driver-side graph addition declared (the LoRA index,
//!   the sampling suffix's tables, a beam reorder's row index).
//!
//! No arm here names a family, a family tag or a model-specific input: a model that declares a new
//! request input binds through the same routine without an edit to this file. A slot nothing binds is
//! refused at prepare ([`Unsupported::UnboundSlot`]), never at step time.

use std::sync::Arc;

use poot_graph_ir::ops::sampling::{HISTORY_WRITE_TAG, SamplerInput};
use poot_graph_ir::{ForwardSlotMap, Graph, Slot, SlotKey};
use poot_models::io::{
    Cadence, Fill, InputId, InputSet, InputValue, Layout, ModelSchema, PadValue, Per, Presence,
    ProgramSpec, RequestInput, RequestInputs, RequestPart, SlotBinding, StepInput, TokenValues,
};
use poot_tensor::{DType, HostTensor};

use super::error::Unsupported;

/// An input a driver-side graph addition declares: the model never sees these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PassInput {
    /// The LoRA pass's per-row adapter index, `[rows]`.
    LoraIndex,
    /// The sampling suffix's per-row seed, `[rows]` I32.
    SamplerSeed,
    /// The sampling suffix's per-row parameters, F32.
    SamplerParams,
    /// The sampling suffix's per-row top-k, I32.
    SamplerTopK,
    /// An adjusted head's per-row `(repetition, presence, frequency)` penalties, `[rows, 3]` F32.
    SamplerPenalty,
    /// An adjusted head's additive guided-decoding mask, `[rows, vocab]` F32.
    SamplerMask,
    /// An adjusted head's history pool row each step row reads, `[rows]` I32.
    HistoryRead,
    /// The history pool row each step row writes (`-1`: none), a checked forward write map.
    HistoryWrite,
    /// A beam reorder's row index: the row each state row takes, `[rows]` I32.
    BeamRows,
}

impl PassInput {
    /// The slot a pass declares this input as.
    pub(super) fn slot_key(self) -> SlotKey {
        match self {
            PassInput::LoraIndex => SlotKey::new(Slot::Owned(poot_graph_plan::LORA_IDX), None),
            PassInput::SamplerSeed => SamplerInput::Seed.key(),
            PassInput::SamplerParams => SamplerInput::Params.key(),
            PassInput::SamplerTopK => SamplerInput::TopK.key(),
            PassInput::SamplerPenalty => SamplerInput::Penalty.key(),
            PassInput::SamplerMask => SamplerInput::Mask.key(),
            PassInput::HistoryRead => SamplerInput::HistoryRead.key(),
            PassInput::HistoryWrite => SlotKey::new(Slot::WriteMap, Some(HISTORY_WRITE_TAG)),
            PassInput::BeamRows => super::reorder::beam_rows_slot(),
        }
    }

    fn of_slot(key: &SlotKey) -> Option<Self> {
        [
            PassInput::LoraIndex,
            PassInput::SamplerSeed,
            PassInput::SamplerParams,
            PassInput::SamplerTopK,
            PassInput::SamplerPenalty,
            PassInput::SamplerMask,
            PassInput::HistoryRead,
            PassInput::HistoryWrite,
            PassInput::BeamRows,
        ]
        .into_iter()
        .find(|pass| pass.slot_key() == *key)
    }
}

/// A request input as one program reads it: the schema entry, which of its slots this is, and what a
/// request that omits it binds.
#[derive(Debug)]
pub(super) struct BoundRequest {
    input: RequestInput,
    part: RequestPart,
    /// The fill a request without the input binds, for a program that defaults it.
    default: Option<Fill>,
}

impl BoundRequest {
    /// Whether this slot is the value slot of request input `id`.
    pub(super) fn is_value_of(&self, id: InputId) -> bool {
        self.input.id() == id && self.part == RequestPart::Value
    }
}

/// How one slot of an entry is bound.
#[derive(Clone, Debug)]
pub(super) enum Binder {
    Step(StepInput),
    Request(Arc<BoundRequest>),
    Pass(PassInput),
}

/// One slot of a prepared program: its key, the shape a bound value must have, and its binder.
#[derive(Clone, Debug)]
pub(super) struct BoundSlot {
    pub(super) key: SlotKey,
    pub(super) shape: Vec<usize>,
    pub(super) binder: Binder,
}

impl BoundSlot {
    /// The slot a driver pass added to an entry after resolution.
    pub(super) fn pass(pass: PassInput, shape: Vec<usize>) -> Self {
        Self {
            key: pass.slot_key(),
            shape,
            binder: Binder::Pass(pass),
        }
    }
}

/// Resolve every slot of `graph`. The slots in `model_slots` were declared by the model's own trace and
/// bind through its `schema` as a step kind or a request input of `program`; every other slot was
/// added by a driver pass and binds as a [`PassInput`]. A slot nothing binds is a defect of the
/// family's wiring (or of a pass), refused here.
pub(super) fn resolve(
    graph: &Graph,
    model_slots: &[SlotKey],
    schema: &ModelSchema,
    program: &ProgramSpec,
) -> Result<Vec<BoundSlot>, Unsupported> {
    let unbound = |key: &SlotKey| Unsupported::UnboundSlot {
        program: program.id(),
        key: key.to_string(),
    };
    slots_of(graph, |key| {
        if model_slots.contains(key) {
            return match schema.binding_of(key).ok_or_else(|| unbound(key))? {
                SlotBinding::Step(kind) => Ok(Binder::Step(kind)),
                SlotBinding::Request(input, part) => {
                    // A program binds only the inputs it declares it reads.
                    let presence = program
                        .presence_of(input.id())
                        .ok_or_else(|| unbound(key))?;
                    let default = match presence {
                        Presence::Defaulted(fill) => Some(fill.clone()),
                        Presence::Required | Presence::Optional => None,
                    };
                    Ok(Binder::Request(Arc::new(BoundRequest {
                        input: input.clone(),
                        part,
                        default,
                    })))
                }
            };
        }
        PassInput::of_slot(key)
            .map(Binder::Pass)
            .ok_or_else(|| unbound(key))
    })
}

/// Resolve the slots of a graph the driver built itself (a rollback copy or a beam reorder): every one
/// is a [`PassInput`].
pub(super) fn resolve_passes(
    graph: &Graph,
    program: poot_models::io::ProgramId,
) -> Result<Vec<BoundSlot>, Unsupported> {
    slots_of(graph, |key| {
        PassInput::of_slot(key)
            .map(Binder::Pass)
            .ok_or_else(|| Unsupported::UnboundSlot {
                program,
                key: key.to_string(),
            })
    })
}

fn slots_of(
    graph: &Graph,
    mut binder: impl FnMut(&SlotKey) -> Result<Binder, Unsupported>,
) -> Result<Vec<BoundSlot>, Unsupported> {
    graph
        .slots()
        .iter()
        .map(|&(value, _)| {
            let meta = graph.meta(value);
            let key = meta
                .slot_key()
                .expect("a slot declared through the builder carries its key")
                .clone();
            Ok(BoundSlot {
                binder: binder(&key)?,
                shape: meta.aval.shape.clone(),
                key,
            })
        })
        .collect()
}

/// The slot keys a graph declares.
pub(super) fn slot_keys(graph: &Graph) -> Vec<SlotKey> {
    graph
        .slots()
        .iter()
        .filter_map(|&(value, _)| graph.meta(value).slot_key().cloned())
        .collect()
}

/// The columns a step computes once from its geometry and bookkeeping, whatever the model: the values
/// of the closed step kinds and of the pass kinds. A column an entry does not declare is simply unread.
#[derive(Debug, Default)]
pub(super) struct StepColumns {
    pub(super) tokens: Vec<i32>,
    pub(super) pos: Vec<i32>,
    pub(super) real: Vec<i32>,
    pub(super) kv_read: Vec<i32>,
    /// Only a checked map reaches the bind.
    pub(super) kv_write: ForwardSlotMap,
    pub(super) state_read: Vec<i32>,
    pub(super) state_write: ForwardSlotMap,
    pub(super) lora: Vec<i32>,
    pub(super) seed: Vec<i32>,
    pub(super) params: Vec<f32>,
    pub(super) top_k: Vec<i32>,
    pub(super) penalty: Vec<f32>,
    pub(super) history_read: Vec<i32>,
    /// Only a checked map reaches the bind.
    pub(super) history_write: ForwardSlotMap,
    pub(super) beam_rows: Vec<i32>,
}

/// Where a row's fed tokens sit among the step's `tokens` columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Side {
    /// The last `fed` columns: a serving step, whose padding is masked on the left.
    Left,
    /// The first `fed` columns: an encode step, whose padding follows the row.
    Right,
}

/// One live row of a step as the request inputs see it: the values its sequence or encode row carries
/// and the window of positions it feeds.
#[derive(Clone, Copy, Debug)]
pub(super) struct RowFeed<'a> {
    pub(super) inputs: &'a RequestInputs,
    /// The absolute position of the first fed token.
    pub(super) start: usize,
    /// Tokens (or, for a row input, positions) the row feeds this step.
    pub(super) fed: usize,
}

/// The rows of one step, for the request binder: the entry's row and token extents, where a row's
/// tokens sit and what each live row feeds. Rows beyond the live ones bind pad.
#[derive(Clone, Copy, Debug)]
pub(super) struct RequestFrame<'a> {
    pub(super) rows: usize,
    pub(super) tokens: usize,
    pub(super) side: Side,
    pub(super) feeds: &'a [RowFeed<'a>],
}

impl RequestFrame<'static> {
    /// The frame of an entry that binds no request input: a rollback copy or a row reorder.
    pub(super) const fn empty() -> Self {
        Self {
            rows: 0,
            tokens: 0,
            side: Side::Left,
            feeds: &[],
        }
    }
}

impl RequestFrame<'_> {
    /// The column (within a row's `tokens`) of the `j`-th fed token of a row feeding `fed` tokens.
    fn column(&self, fed: usize, j: usize) -> usize {
        match self.side {
            Side::Left => self.tokens - fed + j,
            Side::Right => j,
        }
    }
}

/// The carriable inputs of `program` (cadence `Covers` or `AtStart`) that some row of the step binds:
/// a `Covers` input whose value covers a position some row feeds, an `AtStart` input some row feeds
/// position 0 of. The step runs the program variant that declares exactly these.
pub(super) fn carried(
    schema: &ModelSchema,
    program: &ProgramSpec,
    feeds: &[RowFeed<'_>],
) -> InputSet {
    let mut set = InputSet::EMPTY;
    for (id, _) in program.reads() {
        let Some(index) = schema.index_of(*id) else {
            continue;
        };
        let cadence = schema.inputs()[index].form().cadence();
        let binds = |feed: &RowFeed<'_>| match (cadence, feed.inputs.get(*id)) {
            (Cadence::Covers, Some(InputValue::Tokens(values))) => values
                .support()
                .intersects(feed.start, feed.start + feed.fed),
            (Cadence::AtStart, Some(InputValue::Row(_))) => feed.start == 0 && feed.fed > 0,
            _ => false,
        };
        if cadence != Cadence::Every && feeds.iter().any(binds) {
            set = set.with(index);
        }
    }
    set
}

/// Bind one request-kind slot: the value, presence or valid companion of `request`, in the declared
/// `shape`, over the rows of `frame`. The values were checked at admission, so this validates nothing.
pub(super) fn bind_request(
    request: &BoundRequest,
    shape: &[usize],
    frame: &RequestFrame<'_>,
) -> HostTensor {
    let form = request.input.form();
    match form.per() {
        Per::Token { width, layout, .. } => match request.part {
            RequestPart::Presence => presence(request, shape, frame),
            _ => token_values(request, *width, *layout, shape, frame),
        },
        Per::Row { padded, .. } => match request.part {
            RequestPart::Valid => valid(request, shape, frame),
            _ => row_values(request, *padded, shape, frame),
        },
    }
}

/// A column of F32 or I32 elements, starting as the form's pad.
enum Cells {
    I32(Vec<i32>),
    F32(Vec<f32>),
}

impl Cells {
    fn padded(dtype: DType, pad: PadValue, len: usize) -> Self {
        match (dtype, pad) {
            (DType::I32, PadValue::I32(pad)) => Cells::I32(vec![pad; len]),
            (DType::F32, PadValue::F32(pad)) => Cells::F32(vec![pad; len]),
            _ => unreachable!("a schema checks that the pad has the input's type"),
        }
    }

    fn into_tensor(self, shape: &[usize]) -> HostTensor {
        match self {
            Cells::I32(words) => HostTensor::i32(shape.to_vec(), words),
            Cells::F32(values) => HostTensor::f32(shape.to_vec(), values),
        }
    }
}

/// The value of one cell the request gives, or `None` where the pad stays.
enum Scalar {
    I32(i32),
    F32(f32),
}

/// The value of `fill` at `position` on `axis`, for a support ending at `end`; `None` binds the pad.
fn fill_scalar(fill: &Fill, position: usize, end: usize, axis: usize) -> Option<Scalar> {
    let at = i32::try_from(position).expect("a position is bounded by the cache capacity");
    match fill {
        Fill::Pad => None,
        Fill::Positions => Some(Scalar::I32(at)),
        Fill::Advance { from, step } => {
            let end = i32::try_from(end).expect("a support is bounded by the prompt");
            Some(Scalar::I32(from[axis] + step * (at - end)))
        }
    }
}

/// The value at `position` on `axis`: the support's own where it covers, else the value's fill beyond
/// it; with no value, the program's default.
fn cell(
    value: Option<&TokenValues>,
    default: Option<&Fill>,
    position: usize,
    axis: usize,
) -> Option<Scalar> {
    match value {
        Some(values) => match values.support().index_of(position) {
            Some(index) => {
                let at = index * values.width() + axis;
                let data = values.data();
                match (data.as_i32(), data.as_f32()) {
                    (Some(words), _) => Some(Scalar::I32(words[at])),
                    (_, Some(floats)) => Some(Scalar::F32(floats[at])),
                    _ => unreachable!("a token value is I32 or F32"),
                }
            }
            None => fill_scalar(values.beyond(), position, values.support().end(), axis),
        },
        None => default.and_then(|fill| fill_scalar(fill, position, 0, axis)),
    }
}

fn token_value<'a>(request: &BoundRequest, feed: &'a RowFeed<'_>) -> Option<&'a TokenValues> {
    match feed.inputs.get(request.input.id()) {
        Some(InputValue::Tokens(values)) => Some(values),
        _ => None,
    }
}

fn token_values(
    request: &BoundRequest,
    width: usize,
    layout: Layout,
    shape: &[usize],
    frame: &RequestFrame<'_>,
) -> HostTensor {
    let form = request.input.form();
    let columns = frame.rows * frame.tokens;
    let mut cells = Cells::padded(form.dtype(), form.pad(), columns * width);
    for (i, feed) in frame.feeds.iter().enumerate() {
        let value = token_value(request, feed);
        for j in 0..feed.fed {
            let column = i * frame.tokens + frame.column(feed.fed, j);
            let position = feed.start + j;
            for axis in 0..width {
                let at = match layout {
                    Layout::TokenMajor => column * width + axis,
                    Layout::AxisMajor => axis * columns + column,
                };
                match (
                    &mut cells,
                    cell(value, request.default.as_ref(), position, axis),
                ) {
                    (Cells::I32(words), Some(Scalar::I32(v))) => words[at] = v,
                    (Cells::F32(floats), Some(Scalar::F32(v))) => floats[at] = v,
                    (_, None) => {}
                    _ => unreachable!("admission checks the value's type against the form"),
                }
            }
        }
    }
    cells.into_tensor(shape)
}

fn presence(request: &BoundRequest, shape: &[usize], frame: &RequestFrame<'_>) -> HostTensor {
    let mut flags = vec![0; frame.rows * frame.tokens];
    for (i, feed) in frame.feeds.iter().enumerate() {
        let Some(values) = token_value(request, feed) else {
            continue;
        };
        for j in 0..feed.fed {
            if values.support().index_of(feed.start + j).is_some() {
                flags[i * frame.tokens + frame.column(feed.fed, j)] = 1;
            }
        }
    }
    HostTensor::i32(shape.to_vec(), flags)
}

/// The tensor a row feed binds for a per-row input: only the step that feeds position 0 binds it.
fn row_tensor<'a>(request: &BoundRequest, feed: &'a RowFeed<'_>) -> Option<&'a HostTensor> {
    match feed.inputs.get(request.input.id()) {
        Some(InputValue::Row(tensor)) if feed.start == 0 && feed.fed > 0 => Some(tensor),
        _ => None,
    }
}

fn row_values(
    request: &BoundRequest,
    padded: bool,
    shape: &[usize],
    frame: &RequestFrame<'_>,
) -> HostTensor {
    let form = request.input.form();
    let per_row: usize = shape[1..].iter().product();
    let mut cells = Cells::padded(form.dtype(), form.pad(), frame.rows * per_row);
    for (i, feed) in frame.feeds.iter().enumerate() {
        let Some(tensor) = row_tensor(request, feed) else {
            continue;
        };
        debug_assert!(
            tensor.numel() <= per_row && (padded || tensor.numel() == per_row),
            "admission and the driver's extent fit a row tensor to its slot"
        );
        let at = i * per_row;
        match (&mut cells, tensor.as_i32(), tensor.as_f32()) {
            (Cells::I32(words), Some(source), _) => {
                words[at..at + source.len()].copy_from_slice(source);
            }
            (Cells::F32(floats), _, Some(source)) => {
                floats[at..at + source.len()].copy_from_slice(source);
            }
            _ => unreachable!("admission checks the tensor's type against the form"),
        }
    }
    cells.into_tensor(shape)
}

fn valid(request: &BoundRequest, shape: &[usize], frame: &RequestFrame<'_>) -> HostTensor {
    let extent = shape[1];
    let mut flags = vec![0; frame.rows * extent];
    for (i, feed) in frame.feeds.iter().enumerate() {
        if let Some(tensor) = row_tensor(request, feed) {
            let real = tensor.shape().first().copied().unwrap_or(0).min(extent);
            flags[i * extent..i * extent + real].fill(1);
        }
    }
    HostTensor::i32(shape.to_vec(), flags)
}

#[cfg(test)]
mod tests {
    use poot_models::io::{
        InputId, ModelSchema, PadValue, Presence, ProgramId, ProgramSpec, RequestForm,
    };
    use poot_models::model::ModelOutput;

    use super::*;

    const POSITIONS: InputId = InputId::new("axes");
    const SPARSE: InputId = InputId::new("sparse");
    const ROWS: InputId = InputId::new("rows");

    fn token_input(id: InputId, tag: &str, width: usize, layout: Layout) -> RequestInput {
        RequestInput::new(
            id,
            SlotKey::new(Slot::Activation, Some(tag)),
            RequestForm::new(
                DType::I32,
                Per::Token {
                    width,
                    layout,
                    presence: None,
                },
                PadValue::I32(-1),
            ),
        )
    }

    fn sparse_input() -> RequestInput {
        RequestInput::new(
            SPARSE,
            SlotKey::new(Slot::Activation, Some("sparse")),
            RequestForm::new(
                DType::F32,
                Per::Token {
                    width: 2,
                    layout: Layout::TokenMajor,
                    presence: Some(SlotKey::new(Slot::Activation, Some("sparse_at"))),
                },
                PadValue::F32(0.0),
            ),
        )
    }

    fn row_input(padded: bool) -> RequestInput {
        RequestInput::new(
            ROWS,
            SlotKey::new(Slot::Activation, Some("rows")),
            RequestForm::new(
                DType::I32,
                Per::Row {
                    trailing: vec![],
                    padded,
                    valid: padded.then(|| SlotKey::new(Slot::Activation, Some("rows_valid"))),
                },
                PadValue::I32(0),
            ),
        )
    }

    fn schema() -> ModelSchema {
        ModelSchema::builder()
            .input(token_input(POSITIONS, "axes", 3, Layout::AxisMajor))
            .input(sparse_input())
            .input(row_input(true))
            .step(
                ProgramSpec::new(ProgramId::STEP, ModelOutput::Logits { vocab: 4 })
                    .reading(
                        POSITIONS,
                        Presence::Defaulted(Fill::Advance {
                            from: vec![10, 20, 30],
                            step: 1,
                        }),
                    )
                    .reading(SPARSE, Presence::Optional)
                    .reading(ROWS, Presence::Optional),
            )
            .build()
            .unwrap()
    }

    fn bound(schema: &ModelSchema, id: InputId, part: RequestPart) -> BoundRequest {
        let program = schema.step().unwrap();
        let input = schema.input(id).unwrap().clone();
        let default = match program.presence_of(id) {
            Some(Presence::Defaulted(fill)) => Some(fill.clone()),
            _ => None,
        };
        BoundRequest {
            input,
            part,
            default,
        }
    }

    fn dense_axes(len: usize) -> TokenValues {
        // Position `p` on axis `a` is `100 * a + p`; beyond the prompt each axis continues from 7.
        let data: Vec<i32> = (0..len)
            .flat_map(|p| (0..3).map(move |a| 100 * a + p as i32))
            .collect();
        TokenValues::dense(HostTensor::i32(vec![len, 3], data))
            .unwrap()
            .with_beyond(Fill::Advance {
                from: vec![7, 7, 7],
                step: 1,
            })
    }

    /// A dense axis-major value binds the prompt's own positions, then its continuation, and pads the
    /// left of a serving column; a row without the value binds the program's default alone.
    #[test]
    fn dense_values_continue_past_the_prompt_and_pad_the_left() {
        let schema = schema();
        let request = bound(&schema, POSITIONS, RequestPart::Value);
        let with = RequestInputs::none().with(POSITIONS, InputValue::Tokens(dense_axes(3)));
        let without = RequestInputs::none();
        // Two live rows over a 2-row entry of 3 columns: the first feeds positions 2..4 (one prompt
        // position and one continuation), the second feeds position 5 from a sequence with nothing.
        let feeds = [
            RowFeed {
                inputs: &with,
                start: 2,
                fed: 2,
            },
            RowFeed {
                inputs: &without,
                start: 5,
                fed: 1,
            },
        ];
        let frame = RequestFrame {
            rows: 2,
            tokens: 3,
            side: Side::Left,
            feeds: &feeds,
        };
        let tensor = bind_request(&request, &[3, 6], &frame);
        let words = tensor.as_i32().unwrap();
        // Axis-major `[3, rows * tokens]`; columns are row 0's three then row 1's three.
        let axis = |a: usize| &words[a * 6..(a + 1) * 6];
        // Row 0: column 0 pads (-1), column 1 is position 2 (`100 * a + 2`), column 2 is position 3,
        // which continues from 7 at the support's end (3): 7.
        assert_eq!(axis(0)[..3], [-1, 2, 7]);
        assert_eq!(axis(1)[..3], [-1, 102, 7]);
        assert_eq!(axis(2)[..3], [-1, 202, 7]);
        // Row 1 has no value: the default `Advance` from [10, 20, 30] at support end 0 puts position 5
        // at `from + 5` on each axis, in its last column.
        assert_eq!(axis(0)[3..], [-1, -1, 15]);
        assert_eq!(axis(1)[3..], [-1, -1, 25]);
        assert_eq!(axis(2)[3..], [-1, -1, 35]);
    }

    /// A sparse input binds its rows where the value covers the fed positions and says so in the
    /// presence column; the other positions keep the pad and presence 0.
    #[test]
    fn sparse_values_bind_covered_positions_with_presence() {
        let schema = schema();
        let value = TokenValues::sparse(
            vec![1, 3],
            HostTensor::f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        let inputs = RequestInputs::none().with(SPARSE, InputValue::Tokens(value));
        let feeds = [RowFeed {
            inputs: &inputs,
            start: 0,
            fed: 4,
        }];
        let frame = RequestFrame {
            rows: 1,
            tokens: 4,
            side: Side::Left,
            feeds: &feeds,
        };
        let values = bind_request(
            &bound(&schema, SPARSE, RequestPart::Value),
            &[1, 4, 2],
            &frame,
        );
        assert_eq!(
            values.as_f32().unwrap(),
            [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0]
        );
        let at = bind_request(
            &bound(&schema, SPARSE, RequestPart::Presence),
            &[1, 4],
            &frame,
        );
        assert_eq!(at.as_i32().unwrap(), [0, 1, 0, 1]);
    }

    /// A per-row input binds on the step that feeds position 0 only, pads the leading axis to the
    /// slot's extent and marks the real elements; a row that starts later binds pad with valid 0.
    #[test]
    fn row_values_bind_at_the_start_with_a_valid_mask() {
        let schema = schema();
        let value = InputValue::Row(HostTensor::i32(vec![2], vec![5, 6]));
        let starting = RequestInputs::none().with(ROWS, value.clone());
        let later = RequestInputs::none().with(ROWS, value);
        let feeds = [
            RowFeed {
                inputs: &starting,
                start: 0,
                fed: 1,
            },
            RowFeed {
                inputs: &later,
                start: 4,
                fed: 1,
            },
        ];
        let frame = RequestFrame {
            rows: 3,
            tokens: 1,
            side: Side::Left,
            feeds: &feeds,
        };
        let values = bind_request(&bound(&schema, ROWS, RequestPart::Value), &[3, 3], &frame);
        assert_eq!(values.as_i32().unwrap(), [5, 6, 0, 0, 0, 0, 0, 0, 0]);
        let valid = bind_request(&bound(&schema, ROWS, RequestPart::Valid), &[3, 3], &frame);
        assert_eq!(valid.as_i32().unwrap(), [1, 1, 0, 0, 0, 0, 0, 0, 0]);
    }

    /// The carried set holds a `Covers` input when a fed window meets its support and an `AtStart` input
    /// when a row feeds position 0, and neither otherwise.
    #[test]
    fn the_carried_set_follows_each_inputs_cadence() {
        let schema = schema();
        let program = schema.step().unwrap();
        let sparse = RequestInputs::none().with(
            SPARSE,
            InputValue::Tokens(
                TokenValues::sparse(vec![6], HostTensor::f32(vec![1, 2], vec![0.0; 2])).unwrap(),
            ),
        );
        let row =
            RequestInputs::none().with(ROWS, InputValue::Row(HostTensor::i32(vec![1], vec![1])));
        let feed = |inputs, start, fed| RowFeed { inputs, start, fed };
        let index = |id| schema.index_of(id).unwrap();
        let held = |feeds: &[RowFeed<'_>]| carried(&schema, program, feeds);
        assert!(
            held(&[feed(&sparse, 0, 4)]).is_empty(),
            "window misses the position"
        );
        assert!(held(&[feed(&sparse, 4, 4)]).contains(index(SPARSE)));
        assert!(held(&[feed(&row, 0, 2)]).contains(index(ROWS)));
        assert!(
            held(&[feed(&row, 2, 2)]).is_empty(),
            "a later piece binds no row input"
        );
    }
}
