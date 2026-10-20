//! The shared residual pair every standard decoder repeats (spec 999, ADR-0105): [`standard_layer`]
//! (`x += mixer(norm(x)); x += ffn(norm(x))`) and [`standard_stack`] (embedding, one
//! `layer_scope` per layer, final norm, head). A family's tracer is a short call of these with its
//! own components in the closures; neither names a family.
//!
//! [`Step`] carries what one traced step shares across its layers: its shape, the token and
//! position inputs, and the state pairs the layers' components carry (an attention cache). The
//! step's inputs are typed by role: `Slot::Token` and `Slot::Pos`, both `[rows, tokens]` I32, the
//! positions absolute, so prefill and decode are one body continuing from `Pos`.
//!
//! A recurrent component (a conv window, a GDN or SSM matrix) keeps one state row per sequence:
//! [`Step::recurrent_state`] hands it the rows this step continues and [`Step::carry_recurrent`] writes
//! them back. Under [`KvLayout::Paged`] the state is a pool of one row per sequence the pool serves,
//! addressed through two index inputs (`state_read`, a `Slot::Activation`; `state_write`, a
//! `Slot::WriteMap`), so a sequence
//! keeps its row whichever step row it occupies; under [`KvLayout::Contiguous`] the one row is the step's.
//!
//! A step whose shape carries a [`crate::model::HiddenCarry`] also writes its final hidden state at every position
//! into a [`StateRole::Handoff`] pool of the same layout ([`Step::carry_hidden`]): the value a later
//! program of the driver step reads, never rolled back.

use std::cell::RefCell;

use poot_graph_ir::{
    BinOp, Builder, BuilderError, Graph, IndexRange, Outputs, Slot, StateRole, TensorType, Traced,
    ValidationId, ops,
};
use poot_quant::weights::{NormRole, WeightId, WeightMap, WeightRole};
use poot_tensor::DType;

use super::embed::{EMBEDS, embeds_input};
use super::embed::{EmbedInputs, Embedding};
use super::head::{Head, HeadKind};
use super::linear::{Weight, WeightError};
use super::norm::{NormParams, NormWeights, norm, norm_with};
use super::rope::{MROPE, mrope_input};
use crate::io::{
    Fill, ModelSchema, ModelSchemaBuilder, Presence, ProgramId, ProgramSpec, STATE_READ_TAG,
    STATE_WRITE_TAG, StateSpec,
};
use crate::model::WiringError;
use crate::model::{
    ConfigReason, FamilyKey, KvLayout, LogitRows, ModelError, ModelOutput, ShapeReason, StepShape,
    TraceError,
};
use poot_graph_ir::Requirement;

/// The state name of the hidden-state carry ([`Step::carry_hidden`]): one f32
/// `[seqs, positions, width]` value, the same aval in every program of an executable.
pub const HIDDEN_CARRY_STATE: &str = "hidden_carry";

/// A component parameter that cannot configure it, naming the field (spec 999: parameters are
/// validated once, at construction, so tracing with them does not panic).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("parameter {field}: {reason:?}")]
pub struct ParamError {
    /// The config field or parameter name.
    pub field: &'static str,
    /// Why the value was refused.
    pub reason: ConfigReason,
}

impl ParamError {
    /// A refusal of `field` for `reason`.
    pub const fn new(field: &'static str, reason: ConfigReason) -> Self {
        Self { field, reason }
    }

    /// The config error `family` reports for this parameter.
    pub fn for_family(self, family: FamilyKey) -> ModelError {
        ModelError::Config {
            family,
            field: self.field,
            reason: self.reason,
        }
    }
}

/// One traced step: its shape, its token and position inputs, and the state pairs its components
/// carry to the next step.
#[derive(Debug)]
pub struct Step {
    shape: StepShape,
    tokens: Traced,
    pos: Traced,
    mrope: Option<Traced>,
    embeds: Option<EmbedInputs>,
    head: HeadKind,
    paged: Option<PagedMaps>,
    /// The recurrent state pool's index inputs, declared by the first recurrent component.
    state_rows: RefCell<Option<StateRows>>,
    state: RefCell<Vec<(Traced, Traced)>>,
    validations: RefCell<Vec<(ValidationId, String, Traced)>>,
}

/// The hidden-state carry a step declared: the pooled [`StateRole::Handoff`] value `[seqs, positions,
/// width]` and the rows this step's sequences continue.
#[derive(Clone, Copy, Debug)]
pub struct HiddenCarryState {
    state: RecurrentState,
    positions: usize,
    width: usize,
}

/// `(tokens, width)` of a hidden state about to be carried: `[rows, tokens, width]` F32.
fn hidden_dims(b: &Builder, hidden: Traced) -> Result<(usize, usize), TraceError> {
    let aval = b.aval(hidden)?;
    let [_, tokens, width] = aval.shape[..] else {
        return Err(BuilderError::Requirement {
            operation: "hidden-state carry",
            requirement: Requirement::Rank {
                expected: 3,
                found: aval.shape,
            },
        }
        .into());
    };
    if aval.dtype != DType::F32 {
        return Err(BuilderError::Requirement {
            operation: "hidden-state carry",
            requirement: Requirement::Dtype {
                expected: DType::F32,
                found: aval.dtype,
            },
        }
        .into());
    }
    Ok((tokens, width))
}

/// The two slot maps of a [`KvLayout::Paged`] step, both I32 inputs.
#[derive(Clone, Copy, Debug)]
pub struct PagedMaps {
    /// `Slot::SlotMap` `[rows, capacity]`: row `r`'s logical position to the pool slot holding its K/V.
    pub read: Traced,
    /// `Slot::WriteMap` `[rows * tokens]`: the pool slot the K/V of flat token `r * tokens + i` is
    /// written to, or -1 for a padding token. Each slot is named at most once (a `ForwardSlotMap`).
    pub write: Traced,
}

/// The two index inputs of a paged step's recurrent state pool, both I32: `read` a `Slot::Activation`,
/// `write` a `Slot::WriteMap`.
#[derive(Clone, Copy, Debug)]
struct StateRows {
    /// `[rows]`: the pool row each step row continues. A masked row (beyond the live ones) reads row 0.
    read: Traced,
    /// `Slot::WriteMap` `[rows]`: the pool row each step row's result is written to. A masked row is
    /// -1, so its result is dropped.
    write: Traced,
}

/// One recurrent state a component continues: the carried value and the rows this step reads.
#[derive(Clone, Copy, Debug)]
pub struct RecurrentState {
    pool: Traced,
    current: Traced,
}

impl RecurrentState {
    /// The state the step's rows continue from, `[rows, ..row_shape]`.
    pub fn current(&self) -> Traced {
        self.current
    }
}

impl Step {
    /// Declare the step's inputs on `b`: the token ids and their absolute positions, both
    /// `[rows, tokens]` I32, and under a paged layout the two slot maps.
    pub fn new(b: &Builder, shape: StepShape) -> Result<Self, TraceError> {
        let ids = TensorType::new(vec![shape.rows.get(), shape.tokens.get()], DType::I32);
        let paged = match shape.kv {
            KvLayout::Contiguous => None,
            KvLayout::Paged { pool_slots, .. } => Some(PagedMaps {
                read: b.read_map(
                    "read",
                    shape.rows.get(),
                    shape.capacity.get(),
                    pool_slots.get(),
                )?,
                write: b.write_map(
                    "write",
                    shape.rows.get() * shape.tokens.get(),
                    pool_slots.get(),
                )?,
            }),
        };
        Ok(Self {
            shape,
            tokens: b.slot(Slot::Token, ids.clone())?,
            // A position is below the cache capacity: declared here so the range is a checked bind
            // fact, which proves the `I32 -> F32` casts of the position arithmetic exact. Token ids
            // stay undeclared: a family bounds them in-graph (`guard_index_bounds`, a witness).
            pos: b.slot_ranged(
                Slot::Pos,
                ids,
                IndexRange::below(shape.capacity.get()).unwrap_or(IndexRange::I32),
            )?,
            mrope: None,
            embeds: None,
            head: HeadKind::Logits,
            paged,
            state_rows: RefCell::new(None),
            state: RefCell::new(Vec::new()),
            validations: RefCell::new(Vec::new()),
        })
    }

    /// This step with caller-computed embedding rows over some positions ([`EmbedInputs`]): the two
    /// extra inputs are declared after the token and position inputs. `width` is the model's embedding
    /// width. A family calls it for a step that carries [`EMBEDS`](super::embed::EMBEDS).
    pub fn with_embeds(mut self, b: &Builder, width: usize) -> Result<Self, TraceError> {
        self.embeds = Some(EmbedInputs::declare(
            b,
            self.shape.rows.get(),
            self.shape.tokens.get(),
            width,
        )?);
        Ok(self)
    }

    /// A step whose attention rotates by sectioned positions: `Slot::MropePosition`, `[3, rows * tokens]`
    /// I32 (temporal, height, width, axis-major), declared after the token, position and embedding
    /// inputs. `Slot::Pos` still carries the absolute positions the cache and the mask use.
    pub fn with_mrope(mut self, b: &Builder) -> Result<Self, TraceError> {
        let n = self.shape.rows.get() * self.shape.tokens.get();
        self.mrope = Some(b.slot(Slot::MropePosition, TensorType::new(vec![3, n], DType::I32))?);
        Ok(self)
    }

    /// The sectioned positions of a [`Self::with_mrope`] step.
    pub fn mrope(&self) -> Option<Traced> {
        self.mrope
    }

    /// A step that returns the final hidden states instead of logits ([`HeadKind::Hidden`]).
    pub fn returning_hidden(mut self) -> Self {
        self.head = HeadKind::Hidden;
        self
    }

    /// What the step returns.
    pub fn head(&self) -> HeadKind {
        self.head
    }

    /// The embedding rows scattered over the token embeddings; `None` for a plain token step.
    pub fn embeds(&self) -> Option<EmbedInputs> {
        self.embeds
    }

    /// The slot maps of a paged step; `None` under [`KvLayout::Contiguous`].
    pub fn paged(&self) -> Option<PagedMaps> {
        self.paged
    }

    /// The step's shape: rows, tokens and cache capacity.
    pub fn shape(&self) -> StepShape {
        self.shape
    }

    /// The token ids of the step, `[rows, tokens]` I32.
    pub fn tokens(&self) -> Traced {
        self.tokens
    }

    /// The absolute position of every token, `[rows, tokens]` I32.
    pub fn pos(&self) -> Traced {
        self.pos
    }

    /// Carry `input`'s value to the next step as `output`.
    pub fn carry(&self, input: Traced, output: Traced) {
        self.state.borrow_mut().push((input, output));
    }

    /// The recurrent state `name`, one `row_shape` row of `dtype` per sequence, as this step's rows
    /// continue it (`[rows, ..row_shape]`). The caller computes the next rows and hands them to
    /// [`Self::carry_recurrent`]. Under a paged layout the carried value is the pool `[seqs, ..row_shape]`
    /// and the rows are gathered from it by the `state_read` input.
    pub fn recurrent_state(
        &self,
        b: &Builder,
        name: &str,
        dtype: DType,
        row_shape: &[usize],
    ) -> Result<RecurrentState, TraceError> {
        self.pooled_state(b, name, dtype, row_shape, StateRole::Recurrent)
    }

    /// The pooled per-sequence state `name` declared with `role`: [`Self::recurrent_state`] and the
    /// hidden-state carry differ only in the role the value is rolled back by.
    fn pooled_state(
        &self,
        b: &Builder,
        name: &str,
        dtype: DType,
        row_shape: &[usize],
        role: StateRole,
    ) -> Result<RecurrentState, TraceError> {
        let rows = self.shape.rows.get();
        let pooled = |leading: usize| {
            let mut shape = vec![leading];
            shape.extend_from_slice(row_shape);
            Ok::<_, TraceError>(b.state_input(name, TensorType::new(shape, dtype), role)?)
        };
        match self.shape.kv {
            KvLayout::Contiguous => {
                let pool = pooled(rows)?;
                Ok(RecurrentState {
                    pool,
                    current: pool,
                })
            }
            KvLayout::Paged { seqs, .. } => {
                let pool = pooled(seqs.get())?;
                let current = b.gather(pool, 0, self.state_rows(b, seqs.get())?.read)?;
                Ok(RecurrentState { pool, current })
            }
        }
    }

    /// Carry `next` (`[rows, ..row_shape]`) as the new value of `state`'s rows. Under a paged layout
    /// row `r` is written into the pool row `state_write[r]` names (a masked row, -1, writes nothing) and
    /// every other pool row passes through.
    pub fn carry_recurrent(
        &self,
        b: &Builder,
        state: RecurrentState,
        next: Traced,
    ) -> Result<(), TraceError> {
        let carried = match self.shape.kv {
            KvLayout::Contiguous => next,
            KvLayout::Paged { seqs, .. } => {
                let rows = self.state_rows(b, seqs.get())?;
                b.scatter_update(state.pool, next, rows.write)?
            }
        };
        self.carry(state.pool, carried);
        Ok(())
    }

    /// The hidden-state carry the step's shape names, declared once with rows of `width`; `None` for
    /// a step that names none. A step reads it ([`Self::carried_hidden`]) and writes it
    /// ([`Self::carry_hidden`], [`Self::carry_last_hidden`]) through the one handle.
    pub fn hidden_carry(
        &self,
        b: &Builder,
        width: usize,
    ) -> Result<Option<HiddenCarryState>, TraceError> {
        let Some(carry) = self.shape.hidden else {
            return Ok(None);
        };
        let positions = carry.positions.get();
        let state = self.pooled_state(
            b,
            HIDDEN_CARRY_STATE,
            DType::F32,
            &[positions, width],
            StateRole::Handoff,
        )?;
        Ok(Some(HiddenCarryState {
            state,
            positions,
            width,
        }))
    }

    /// The first `tokens` rows of `carry` as this step's rows continue them, `[rows, tokens, width]`
    /// F32: the hidden states of the window or chunk an earlier program of the driver step wrote at
    /// the carry's first positions.
    pub fn carried_hidden(
        &self,
        b: &Builder,
        carry: &HiddenCarryState,
    ) -> Result<Traced, TraceError> {
        Ok(b.slice(
            carry.state.current,
            1,
            0,
            self.shape.tokens.get().min(carry.positions),
        )?)
    }

    /// Write `hidden`, the step's final hidden state `[rows, tokens, width]` F32 at every position,
    /// into the hidden-state carry the step's shape asks for ([`StepShape::hidden`]); a step that
    /// asks for none writes nothing. The carry holds the step's positions at `0..tokens` of its
    /// `positions` axis and keeps the rest: rows `tokens..positions` still hold what earlier steps
    /// wrote, so a reader of the carry must know the length of the last window or chunk and read only
    /// that many rows.
    /// Call it before any last-token slice: the head may stay last-only, the carry does not.
    pub fn carry_hidden(&self, b: &Builder, hidden: Traced) -> Result<(), TraceError> {
        let (tokens, width) = hidden_dims(b, hidden)?;
        let Some(carry) = self.hidden_carry(b, width)? else {
            return Ok(());
        };
        if tokens != self.shape.tokens.get() {
            return Err(WiringError::HiddenCarryRows {
                expected: self.shape.tokens.get(),
                found: tokens,
            }
            .into());
        }
        let next = if tokens == carry.positions {
            hidden
        } else {
            let kept = b.slice(carry.state.current, 1, tokens, carry.positions)?;
            b.concat(1, &[hidden, kept])?
        };
        self.carry_recurrent(b, carry.state, next)?;
        Ok(())
    }

    /// Write the last position of `hidden` (`[rows, tokens, width]` F32) into row 0 of `carry` and
    /// keep rows `1..positions`: the row a following one-token step of a chain reads. The
    /// draft program's write, the layer recycling its own output.
    pub fn carry_last_hidden(
        &self,
        b: &Builder,
        carry: &HiddenCarryState,
        hidden: Traced,
    ) -> Result<(), TraceError> {
        let (tokens, width) = hidden_dims(b, hidden)?;
        if width != carry.width {
            return Err(WiringError::HiddenCarryRows {
                expected: carry.width,
                found: width,
            }
            .into());
        }
        let last = b.slice(hidden, 1, tokens - 1, tokens)?;
        let next = if carry.positions == 1 {
            last
        } else {
            let kept = b.slice(carry.state.current, 1, 1, carry.positions)?;
            b.concat(1, &[last, kept])?
        };
        self.carry_recurrent(b, carry.state, next)?;
        Ok(())
    }

    /// The pool's index inputs, declared once per step.
    fn state_rows(&self, b: &Builder, seqs: usize) -> Result<StateRows, TraceError> {
        let mut declared = self.state_rows.borrow_mut();
        if let Some(rows) = *declared {
            return Ok(rows);
        }
        let rows = StateRows {
            read: b.slot_named(
                Slot::Activation,
                STATE_READ_TAG,
                TensorType::new(vec![self.shape.rows.get()], DType::I32),
            )?,
            write: b.write_map(STATE_WRITE_TAG, self.shape.rows.get(), seqs)?,
        };
        *declared = Some(rows);
        Ok(rows)
    }

    /// Declare `value`, a bounded F32 witness, as validation root `id` named `name`: the executor
    /// checks it every step before any output is read back.
    pub fn validate(&self, id: ValidationId, name: String, value: Traced) {
        self.validations.borrow_mut().push((id, name, value));
    }

    /// The step's graph: `out` plus every carried state pair, in carry order, and every declared
    /// validation root, in declaration order.
    pub fn finish(self, b: Builder, out: Traced) -> Result<Graph, BuilderError> {
        let outputs = self.validations.into_inner().into_iter().fold(
            Outputs::new(out).with_state(&self.state.into_inner()),
            |outputs, (id, name, value)| outputs.with_witness(id, name, value),
        );
        b.finish(outputs)
    }
}

/// The two norms of a standard layer, and for a sandwich layer the two norms applied to each
/// block's output ([`LayerNorms::sandwich`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerNorms {
    /// The norm before (or, under [`NormPlacement::Post`], after) the token mixer.
    pub attn: NormWeights,
    /// The norm before (or after) the feed-forward block.
    pub ffn: NormWeights,
    post: Option<[Weight; 2]>,
}

impl LayerNorms {
    /// Resolves layer `layer`'s two unbiased norm scales from `map`; a missing or mis-shaped tensor
    /// is a typed [`WeightError`].
    pub fn new(map: &WeightMap, layer: usize, width: usize) -> Result<Self, WeightError> {
        Self::read(map, layer, width, false)
    }

    /// LayerNorms that carry a bias each (BLOOM).
    pub fn biased(map: &WeightMap, layer: usize, width: usize) -> Result<Self, WeightError> {
        Self::read(map, layer, width, true)
    }

    fn read(
        map: &WeightMap,
        layer: usize,
        width: usize,
        biased: bool,
    ) -> Result<Self, WeightError> {
        let weight = |role| {
            Weight::new(
                map,
                WeightId::layer(layer, WeightRole::Norm(role)),
                &[width],
            )
        };
        let norm = |scale, bias| {
            Ok::<_, WeightError>(NormWeights {
                scale: weight(scale)?,
                bias: if biased { Some(weight(bias)?) } else { None },
            })
        };
        Ok(Self {
            attn: norm(NormRole::Attn, NormRole::AttnBias)?,
            ffn: norm(NormRole::Ffn, NormRole::FfnBias)?,
            post: None,
        })
    }

    /// The four norms of a [`NormPlacement::Sandwich`] layer: `Attn` and `Ffn` before the blocks,
    /// `PostAttn` and `PostFfn` after them.
    pub fn sandwich(map: &WeightMap, layer: usize, width: usize) -> Result<Self, WeightError> {
        let post = |role| {
            Weight::new(
                map,
                WeightId::layer(layer, WeightRole::Norm(role)),
                &[width],
            )
        };
        Ok(Self {
            post: Some([post(NormRole::PostAttn)?, post(NormRole::PostFfn)?]),
            ..Self::new(map, layer, width)?
        })
    }
}

/// The scalar a layer multiplies its output by (Gemma 4's `layer_scalar`): a learned `[1]` weight,
/// applied after the layer's last residual add.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerScale(Weight);

impl LayerScale {
    /// Resolves layer `layer`'s `[1]` scalar from `map`; a missing or mis-shaped tensor is a typed
    /// [`WeightError`].
    pub fn new(map: &WeightMap, layer: usize) -> Result<Self, WeightError> {
        Weight::new(map, WeightId::layer(layer, WeightRole::LayerScale), &[1]).map(Self)
    }

    /// `x` times the scalar.
    pub fn apply(&self, b: &Builder, x: Traced) -> Result<Traced, TraceError> {
        Ok(b.binary(BinOp::Mul, x, self.0.declare_f32(b)?)?)
    }
}

/// Where a layer's two norms sit around its mixer and feed-forward blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormPlacement {
    /// `x += block(norm(x))`: normalize the block's input.
    Pre,
    /// `x += norm(block(x))`: normalize the block's output (OLMo 2). The layer's two norm
    /// weights are the post-mixer and post-feed-forward norms.
    Post,
    /// `x += post(block(pre(x)))`: both (Gemma 2 and 3). Needs [`LayerNorms::sandwich`].
    Sandwich,
}

/// How a standard layer combines its blocks: the norm, where it sits and the scale applied to each
/// block's output before it joins the residual.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerParams {
    norm: NormParams,
    placement: NormPlacement,
    residual_scale: Option<f32>,
}

impl LayerParams {
    /// Pre-norm placement with no residual scale.
    pub fn new(norm: NormParams) -> Self {
        Self {
            norm,
            placement: NormPlacement::Pre,
            residual_scale: None,
        }
    }

    /// Where the layer's norms sit.
    pub fn with_placement(self, placement: NormPlacement) -> Self {
        Self { placement, ..self }
    }

    /// Scale each block's output by `scale` before the residual add (Granite's multiplier).
    pub fn with_residual_scale(self, scale: f32) -> Result<Self, ParamError> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(ParamError::new(
                "residual_scale",
                ConfigReason::NotFinitePositive,
            ));
        }
        Ok(Self {
            residual_scale: Some(scale),
            ..self
        })
    }

    /// The norm the layer applies.
    pub fn norm(&self) -> NormParams {
        self.norm
    }
}

/// `x += mixer(norm(x)); x += ffn(norm(x))`, or with [`NormPlacement::Post`]
/// `x += norm(mixer(x)); x += norm(ffn(x))`, or with [`NormPlacement::Sandwich`] both norms around
/// each block, each block's output scaled by the residual scale. A `Sandwich` placement over
/// norms not built by [`LayerNorms::sandwich`] is a caller bug and panics.
pub fn standard_layer(
    b: &Builder,
    x: Traced,
    norms: &LayerNorms,
    p: &LayerParams,
    mixer: impl FnOnce(&Builder, Traced) -> Result<Traced, TraceError>,
    ffn: impl FnOnce(&Builder, BlockInput) -> Result<Traced, TraceError>,
) -> Result<Traced, TraceError> {
    let post = |i: usize| norms.post.as_ref().map(|post| &post[i]);
    let x = residual_block(b, x, &norms.attn, post(0), p, |b, input| {
        mixer(b, input.block)
    })?;
    residual_block(b, x, &norms.ffn, post(1), p, ffn)
}

/// What a block of a standard layer reads: the block's normalized input (the raw residual under
/// [`NormPlacement::Post`]) and the residual stream it joins. Most blocks read only `block`; Gemma
/// 4's routed feed-forward also routes on `residual` and normalizes it with its own norm.
#[derive(Clone, Copy, Debug)]
pub struct BlockInput {
    /// The value the block computes over.
    pub block: Traced,
    /// The residual stream the block's output is added to.
    pub residual: Traced,
}

fn residual_block(
    b: &Builder,
    x: Traced,
    w: &NormWeights,
    post: Option<&Weight>,
    p: &LayerParams,
    block: impl FnOnce(&Builder, BlockInput) -> Result<Traced, TraceError>,
) -> Result<Traced, TraceError> {
    let input = |block| BlockInput { block, residual: x };
    let out = match p.placement {
        NormPlacement::Pre => block(b, input(norm_with(b, x, w, p.norm)?))?,
        NormPlacement::Post => norm_with(b, block(b, input(x))?, w, p.norm)?,
        NormPlacement::Sandwich => {
            let post = post.ok_or(WiringError::SandwichWithoutPostNorms)?;
            norm(
                b,
                block(b, input(norm_with(b, x, w, p.norm)?))?,
                post,
                p.norm,
            )?
        }
    };
    let out = match p.residual_scale {
        Some(scale) => b.binary_scalar(BinOp::Mul, out, Builder::f32(scale))?,
        None => out,
    };
    Ok(b.binary(BinOp::Add, x, out)?)
}

/// The model-level weights around the layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackWeights {
    /// The token embedding table.
    pub embed: Embedding,
    /// The norm applied after the last layer.
    pub final_norm: NormWeights,
    /// A LayerNorm over the embedded tokens, before the first layer (BLOOM).
    pub embed_norm: Option<NormWeights>,
    /// The output projection.
    pub head: Head,
}

impl StackWeights {
    /// Resolves the embedding, the unbiased final norm and the head from `map`; a missing or
    /// mis-shaped tensor is a typed [`WeightError`].
    pub fn new(map: &WeightMap, vocab: usize, width: usize) -> Result<Self, WeightError> {
        Ok(Self {
            embed: Embedding::new(map, vocab, width)?,
            final_norm: NormWeights {
                scale: Weight::new(map, WeightId::model(WeightRole::FinalNorm), &[width])?,
                bias: None,
            },
            embed_norm: None,
            head: Head::new(map, vocab, width)?,
        })
    }

    /// The same weights with the final norm's bias read from `map`.
    pub fn with_final_norm_bias(
        mut self,
        map: &WeightMap,
        width: usize,
    ) -> Result<Self, WeightError> {
        let id = WeightId::model(WeightRole::FinalNormBias);
        self.final_norm.bias = Some(Weight::new(map, id, &[width])?);
        Ok(self)
    }

    /// The same weights with an embedding LayerNorm (scale and bias) read from `map`.
    pub fn with_embed_norm(mut self, map: &WeightMap, width: usize) -> Result<Self, WeightError> {
        let weight = |role| Weight::new(map, WeightId::model(role), &[width]);
        self.embed_norm = Some(NormWeights {
            scale: weight(WeightRole::EmbedNorm)?,
            bias: Some(weight(WeightRole::EmbedNormBias)?),
        });
        Ok(self)
    }
}

/// How the stack scales around its layers: the final norm, an embedding multiplier and a divisor on
/// the logits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StackParams {
    norm: NormParams,
    embed_scale: Option<f32>,
    logit_divisor: Option<f32>,
    softcap: Option<f32>,
}

impl StackParams {
    /// The final norm with no embedding multiplier, logit divisor or softcap.
    pub fn new(norm: NormParams) -> Self {
        Self {
            norm,
            embed_scale: None,
            logit_divisor: None,
            softcap: None,
        }
    }

    /// Multiply the embedded tokens by `scale` (Gemma's `sqrt(width)`, Granite's multiplier).
    pub fn with_embed_scale(self, scale: f32) -> Result<Self, ParamError> {
        positive("embed_scale", scale)?;
        Ok(Self {
            embed_scale: Some(scale),
            ..self
        })
    }

    /// Divide the logits by `divisor` (Granite's `logits_scaling`).
    pub fn with_logit_divisor(self, divisor: f32) -> Result<Self, ParamError> {
        positive("logit_divisor", divisor)?;
        Ok(Self {
            logit_divisor: Some(divisor),
            ..self
        })
    }

    /// `cap * tanh(logits / cap)` on the logits, after the divisor (Gemma 2's final softcap).
    pub fn with_softcap(self, cap: f32) -> Result<Self, ParamError> {
        positive("final_softcap", cap)?;
        Ok(Self {
            softcap: Some(cap),
            ..self
        })
    }

    /// The final norm.
    pub fn norm(&self) -> NormParams {
        self.norm
    }
}

fn positive(field: &'static str, value: f32) -> Result<(), ParamError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ParamError::new(field, ConfigReason::NotFinitePositive))
    }
}

/// Embed the step's tokens, run `layer` over `layers` layers (each inside its `layer_scope`),
/// apply the final norm and the head. Returns `[rows, tokens, vocab]` logits, or only the last
/// token's (`[rows, 1, vocab]`) under [`LogitRows::Last`].
pub fn standard_stack(
    b: &Builder,
    step: &Step,
    w: &StackWeights,
    p: &StackParams,
    layers: usize,
    mut layer: impl FnMut(&Builder, usize, Traced) -> Result<Traced, TraceError>,
) -> Result<Traced, TraceError> {
    let mut x = w.embed.embed(b, step.tokens())?;
    if let Some(scale) = p.embed_scale {
        x = b.binary_scalar(BinOp::Mul, x, Builder::f32(scale))?;
    }
    if let Some(embed_norm) = &w.embed_norm {
        x = norm_with(b, x, embed_norm, p.norm)?;
    }
    // The caller's rows are the layer input itself: they replace the embedded tokens, after the
    // embedding's own scale and norm.
    if let Some(embeds) = step.embeds() {
        x = embeds.scatter(b, x)?;
    }
    for l in 0..layers {
        let _scope = b.layer_scope(l);
        x = layer(b, l, x)?;
    }
    let tokens = step.shape().tokens.get();
    let last_only = step.shape().logits == LogitRows::Last && tokens > 1;
    let x = if step.shape().hidden.is_some() {
        // The carry takes every position's normed state; only then does the head narrow to the last.
        let normed = norm_with(b, x, &w.final_norm, p.norm)?;
        step.carry_hidden(b, normed)?;
        if last_only {
            b.slice(normed, 1, tokens - 1, tokens)?
        } else {
            normed
        }
    } else {
        if last_only {
            x = b.slice(x, 1, tokens - 1, tokens)?;
        }
        norm_with(b, x, &w.final_norm, p.norm)?
    };
    if step.head() == HeadKind::Hidden {
        return Ok(x);
    }
    let logits = w.head.logits(b, x)?;
    let logits = match p.logit_divisor {
        Some(divisor) => b.binary_scalar(BinOp::Div, logits, Builder::f32(divisor))?,
        None => logits,
    };
    match p.softcap {
        Some(cap) => Ok(ops::softcap(b, logits, cap)?),
        None => Ok(logits),
    }
}

/// Refuse a step shape the dense decoder body cannot trace: a capacity above the model's maximum
/// positions, more new tokens than the cache holds, or a hidden-state carry narrower than the step.
pub fn check_step(
    family: FamilyKey,
    shape: StepShape,
    max_positions: usize,
) -> Result<(), TraceError> {
    let refuse = |reason| {
        Err(TraceError::ShapeUnsupported {
            family,
            shape,
            reason,
        })
    };
    if shape.capacity.get() > max_positions {
        return refuse(ShapeReason::CapacityAboveMax { max: max_positions });
    }
    if shape.tokens > shape.capacity {
        return refuse(ShapeReason::TokensAboveCapacity);
    }
    if let Some(carry) = shape.hidden
        && carry.positions < shape.tokens
    {
        return refuse(ShapeReason::HiddenCarryNarrow {
            positions: carry.positions.get(),
            tokens: shape.tokens.get(),
        });
    }
    Ok(())
}

/// [`check_step`] for a family's draft program: the layer's step is checked as a prefill over the same
/// cache shape, and it must name the hidden-state carry it reads.
pub fn check_draft_step(
    family: FamilyKey,
    shape: StepShape,
    max_positions: usize,
) -> Result<(), TraceError> {
    check_step(family, shape, max_positions)?;
    if shape.hidden.is_none() {
        return Err(TraceError::ShapeUnsupported {
            family,
            shape,
            reason: ShapeReason::DraftWithoutCarry,
        });
    }
    Ok(())
}

/// The `draft` program of a model with a multi-token-prediction layer: logits over `vocab` from the
/// layer's own KV cache, reading and writing the hidden-state carry.
pub fn draft_program(vocab: usize) -> ProgramSpec {
    ProgramSpec::new(ProgramId::DRAFT, ModelOutput::Logits { vocab })
        .carrying(StateSpec::kv())
        .carrying(StateSpec::hand_off())
}

/// The schema of a decoder whose prompt may supply embedding rows in place of some tokens' own
/// ([`EMBEDS`], optional) and that also offers a decoder embedding pass (`encode`, returning hidden
/// states). A `sectioned` model rotates by [`MROPE`] positions, defaulted to the text positions.
pub fn embeddable_decoder_schema(
    vocab: usize,
    width: usize,
    sectioned: bool,
) -> ModelSchemaBuilder {
    let mut step = decoder_program(vocab).reading(EMBEDS, Presence::Optional);
    let mut encode = ProgramSpec::new(ProgramId::ENCODE, ModelOutput::Hidden { width })
        .carrying(StateSpec::kv());
    let mut schema = ModelSchema::builder().input(embeds_input(width));
    if sectioned {
        let positions = Presence::Defaulted(Fill::Positions);
        step = step.reading(MROPE, positions.clone());
        encode = encode.reading(MROPE, positions);
        schema = schema.input(mrope_input());
    }
    schema.step(step).encode(encode)
}

/// The `step` program of a standard decoder: logits over `vocab`, carrying a KV cache and, when a
/// step's shape asks for one, the hidden-state carry. A family adds the request inputs it reads and
/// the further states it carries.
pub fn decoder_program(vocab: usize) -> ProgramSpec {
    ProgramSpec::new(ProgramId::STEP, ModelOutput::Logits { vocab })
        .carrying(StateSpec::kv())
        .carrying(StateSpec::hand_off())
}

/// The CPU-oracle harness the component tests share: a fixture weight store keyed by
/// [`WeightId::const_name`], a binder from graph consts to its entries, and f64 comparisons.
#[cfg(test)]
pub(crate) mod oracle {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    use poot_eval::{EvalOptions, Value};
    use poot_graph_ir::{Graph, PackedSourceName, Slot, Storage, ValueId};
    use poot_quant::weights::{
        DenseWeight, WeightEntry, WeightId, WeightKey, WeightMap, WeightStore, WeightView,
    };
    use poot_quant::{PackedComponentRef, PackedPayload, PackedStack, PackedStackComponentRef};
    use poot_tensor::DType;
    use poot_tensor::HostTensor;

    use crate::model::{HiddenCarry, LogitRows, StepShape, StepSpec};

    /// Deterministic values in `[-0.5, 0.5)`.
    pub(crate) fn random(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn bf16_bits(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    /// A packed payload's logical `[rows, K]` values.
    fn decoded(payload: &PackedPayload) -> Vec<f32> {
        let [rows, k] = payload.weight().shape();
        let mut out = vec![0.0; rows * k];
        for (row, chunk) in out.chunks_mut(k).enumerate() {
            payload.decode_row(row, chunk).unwrap();
        }
        out
    }

    #[derive(Clone, Debug)]
    enum Stored {
        F32(Vec<f32>),
        Bf16(Vec<u16>),
        E4m3(Vec<u8>),
        Packed(Arc<PackedPayload>),
        /// The experts of one projection, each its own packed payload, viewed as one stack.
        PackedStack(Vec<Arc<PackedPayload>>),
    }

    /// A fixture's weights: each id stored under its const name, or tied to another id's entry.
    #[derive(Clone, Debug)]
    pub(crate) struct Fx {
        entries: BTreeMap<WeightId, (Vec<usize>, Stored)>,
        ties: BTreeMap<WeightId, WeightId>,
        seed: u64,
    }

    impl Fx {
        pub(crate) fn new(seed: u64) -> Self {
            Self {
                entries: BTreeMap::new(),
                ties: BTreeMap::new(),
                seed,
            }
        }

        fn next_seed(&mut self) -> u64 {
            self.seed += 1;
            self.seed
        }

        /// A random F32 weight.
        pub(crate) fn f32(&mut self, id: WeightId, shape: &[usize]) {
            let n = shape.iter().product();
            let values = random(n, self.next_seed());
            self.set(id, shape, values);
        }

        /// A random BF16 weight.
        pub(crate) fn bf16(&mut self, id: WeightId, shape: &[usize]) {
            let n = shape.iter().product();
            let bits = random(n, self.next_seed())
                .into_iter()
                .map(bf16_bits)
                .collect();
            self.entries
                .insert(id, (shape.to_vec(), Stored::Bf16(bits)));
        }

        /// An E4M3FN weight with the given codes (finite ones: `0x7f` and `0xff` are NaN).
        pub(crate) fn e4m3(&mut self, id: WeightId, shape: &[usize], codes: Vec<u8>) {
            assert_eq!(codes.len(), shape.iter().product::<usize>());
            assert!(codes.iter().all(|code| code & 0x7f != 0x7f), "finite codes");
            self.entries
                .insert(id, (shape.to_vec(), Stored::E4m3(codes)));
        }

        /// An F32 weight with the given values.
        pub(crate) fn set(&mut self, id: WeightId, shape: &[usize], values: Vec<f32>) {
            assert_eq!(values.len(), shape.iter().product::<usize>());
            self.entries
                .insert(id, (shape.to_vec(), Stored::F32(values)));
        }

        /// A packed weight.
        pub(crate) fn packed(&mut self, id: WeightId, payload: PackedPayload) {
            let shape = payload.weight().shape().to_vec();
            self.entries
                .insert(id, (shape, Stored::Packed(Arc::new(payload))));
        }

        /// A stack of packed experts (one payload per expert, in expert order) as the one weight `id`.
        pub(crate) fn packed_stack(&mut self, id: WeightId, experts: Vec<PackedPayload>) {
            let [out, k] = experts[0].weight().shape();
            let shape = vec![experts.len(), out, k];
            let owners = experts.into_iter().map(Arc::new).collect();
            self.entries
                .insert(id, (shape, Stored::PackedStack(owners)));
        }

        /// Map `id` onto `target`'s stored entry (a tied head).
        pub(crate) fn tie(&mut self, id: WeightId, target: WeightId) {
            self.ties.insert(id, target);
        }

        /// The weight's logical values (a BF16 or packed entry decoded).
        pub(crate) fn values(&self, id: WeightId) -> Vec<f32> {
            let id = self.ties.get(&id).copied().unwrap_or(id);
            match &self.entries[&id].1 {
                Stored::F32(values) => values.clone(),
                Stored::Bf16(bits) => bits
                    .iter()
                    .map(|&b| f32::from_bits(u32::from(b) << 16))
                    .collect(),
                Stored::E4m3(codes) => codes
                    .iter()
                    .map(|&code| poot_quant::scalar::e4m3fn_to_f32(code))
                    .collect(),
                Stored::Packed(payload) => decoded(payload),
                Stored::PackedStack(experts) => experts.iter().flat_map(|e| decoded(e)).collect(),
            }
        }

        /// Add a different random offset to every element of `id`, kept in its stored dtype.
        pub(crate) fn perturb(&mut self, id: WeightId) {
            let (shape, _) = self.entries[&id].clone();
            let seed = self.next_seed();
            let values: Vec<f32> = self
                .values(id)
                .iter()
                .zip(random(shape.iter().product(), seed))
                .map(|(v, r)| v + 0.25 + r)
                .collect();
            let stored = match self.entries[&id].1 {
                Stored::Bf16(_) => Stored::Bf16(values.into_iter().map(bf16_bits).collect()),
                Stored::F32(_) => Stored::F32(values),
                Stored::E4m3(_) | Stored::Packed(_) | Stored::PackedStack(_) => {
                    panic!("perturb an F32 or BF16 weight")
                }
            };
            self.entries.insert(id, (shape, stored));
        }

        pub(crate) fn store(&self) -> WeightStore {
            let mut sb = WeightStore::builder();
            for (id, (shape, stored)) in &self.entries {
                let entry = match stored {
                    Stored::F32(values) => WeightEntry::Dense(
                        DenseWeight::try_new(
                            DType::F32,
                            shape.clone(),
                            values
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<u8>>()
                                .into(),
                        )
                        .unwrap(),
                    ),
                    Stored::Bf16(bits) => WeightEntry::Dense(
                        DenseWeight::try_new(
                            DType::BF16,
                            shape.clone(),
                            bits.iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<u8>>()
                                .into(),
                        )
                        .unwrap(),
                    ),
                    Stored::E4m3(codes) => WeightEntry::Dense(
                        DenseWeight::try_new(DType::E4M3FN, shape.clone(), codes.clone().into())
                            .unwrap(),
                    ),
                    Stored::Packed(payload) => WeightEntry::Packed(Arc::clone(payload)),
                    Stored::PackedStack(experts) => {
                        for (expert, payload) in experts.iter().enumerate() {
                            sb.insert(
                                format!("{}.{expert}", id.const_name()),
                                WeightEntry::Packed(Arc::clone(payload)),
                            )
                            .unwrap();
                        }
                        continue;
                    }
                };
                sb.insert(id.const_name(), entry).unwrap();
            }
            sb.build()
        }

        pub(crate) fn map(&self) -> WeightMap {
            let store = self.store();
            let mut map = WeightMap::builder(&store);
            for (id, (_, stored)) in &self.entries {
                let view = match stored {
                    Stored::PackedStack(experts) => WeightView::ExpertStack(
                        (0..experts.len())
                            .map(|e| WeightKey::new(format!("{}.{e}", id.const_name())))
                            .collect(),
                    ),
                    _ => WeightView::Stored(WeightKey::new(id.const_name())),
                };
                map.map(*id, view).unwrap();
            }
            for (id, target) in &self.ties {
                map.map(*id, WeightView::Stored(WeightKey::new(target.const_name())))
                    .unwrap();
            }
            map.build()
        }

        /// Evaluate `g` on the oracle with this fixture's weights; see [`eval`].
        pub(crate) fn eval(
            &self,
            g: &Graph,
            consts: &[(&str, &[f32])],
            slots: &[(Slot, &[i32])],
            state: &[&[f32]],
        ) -> (Vec<f32>, Vec<Vec<f32>>) {
            eval(g, &self.store(), &self.map(), consts, slots, state)
        }

        /// [`Fx::eval`] with `named` slots (`Slot::SlotMap` `"read"`, `Slot::WriteMap` `"write"`) bound by
        /// their key spelling.
        pub(crate) fn eval_named(
            &self,
            g: &Graph,
            consts: &[(&str, &[f32])],
            slots: &[(Slot, &[i32])],
            named: &[(&str, &[i32])],
            state: &[&[f32]],
        ) -> (Vec<f32>, Vec<Vec<f32>>) {
            eval_named(g, &self.store(), &self.map(), consts, slots, named, state)
        }
    }

    /// Evaluate `g` on the CPU oracle. Every weight const is bound from `store` through `map` by its
    /// [`WeightId::const_name`] (packed carriers by their [`PackedSourceName`]); other consts by name
    /// from `consts`; slots by kind; state inputs in [`Graph::state`] order. Returns the output and
    /// every state output, as f32.
    pub(crate) fn eval(
        g: &Graph,
        store: &WeightStore,
        map: &WeightMap,
        consts: &[(&str, &[f32])],
        slots: &[(Slot, &[i32])],
        state: &[&[f32]],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        eval_named(g, store, map, consts, slots, &[], state)
    }

    /// [`eval`] with `named` slots: a tagged slot (`Builder::slot_named`) takes the entry whose name
    /// is its key spelling (`"slotmap.read"`) before falling back to its kind's entry.
    pub(crate) fn eval_named(
        g: &Graph,
        store: &WeightStore,
        map: &WeightMap,
        consts: &[(&str, &[f32])],
        slots: &[(Slot, &[i32])],
        named: &[(&str, &[i32])],
        state: &[&[f32]],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        eval_slots(g, store, map, consts, slots, (named, &[]), state)
    }

    /// Named slots by key spelling: I32 ones (a [`Slot::SlotMap`]'s tags) and F32 ones (a
    /// `Slot::Activation` of pixels or embedding rows).
    pub(crate) type NamedSlots<'a> = (&'a [(&'a str, &'a [i32])], &'a [(&'a str, &'a [f32])]);

    /// [`eval_named`] with F32 named slots as well, bound by the same key spelling.
    pub(crate) fn eval_slots(
        g: &Graph,
        store: &WeightStore,
        map: &WeightMap,
        consts: &[(&str, &[f32])],
        slots: &[(Slot, &[i32])],
        (named, floats): NamedSlots<'_>,
        state: &[&[f32]],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let ids: HashMap<String, WeightId> =
            map.iter().map(|(id, _, _)| (id.const_name(), id)).collect();
        // The weight's own entry: the stored one, or the rows of a fused or stacked view cut from
        // the store (as the real binders read it).
        let entry = |id: WeightId| -> WeightEntry {
            map.materialize(id, store)
                .unwrap_or_else(|e| panic!("{id}: {e}"))
        };
        assert_eq!(state.len(), g.state().count(), "one value per state input");
        // One stack per packed expert-stack id, shared by every source role that names it.
        let mut stacks: HashMap<WeightId, Arc<PackedStack>> = HashMap::new();
        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        // The harness carries every state as f32; an integer state (a token history) is exact
        // below 2^24 and is bound at its own dtype.
        for (i, (si, _)) in g.state().enumerate() {
            let aval = g.aval(si);
            let tensor = match aval.dtype {
                DType::I32 => HostTensor::i32(
                    aval.shape.clone(),
                    state[i].iter().map(|&v| v as i32).collect(),
                ),
                _ => HostTensor::f32(aval.shape.clone(), state[i].to_vec()),
            };
            inputs.insert(si, tensor.into());
        }
        for &id in g.inputs() {
            let meta = g.meta(id);
            let shape = meta.aval.shape.clone();
            let value: Value = match &meta.storage {
                Storage::State | Storage::Computed(_) => continue,
                Storage::Slot(_)
                    if floats.iter().any(|(n, _)| Some(*n) == meta.name.as_deref()) =>
                {
                    let (_, values) = floats
                        .iter()
                        .find(|(n, _)| Some(*n) == meta.name.as_deref())
                        .expect("matched above");
                    HostTensor::f32(shape, values.to_vec()).into()
                }
                Storage::Slot(slot) => {
                    let by_name = named
                        .iter()
                        .find(|(n, _)| Some(*n) == meta.name.as_deref())
                        .map(|(_, ints)| *ints);
                    let ints = by_name.unwrap_or_else(|| {
                        slots
                            .iter()
                            .find(|(s, _)| s == slot)
                            .map(|(_, ints)| *ints)
                            .unwrap_or_else(|| panic!("no value for slot {slot:?}"))
                    });
                    HostTensor::i32(shape, ints.to_vec()).into()
                }
                Storage::Const => {
                    let name = meta.name.as_deref().expect("a const has a name");
                    if let Some((_, values)) = consts.iter().find(|(n, _)| *n == name) {
                        HostTensor::f32(shape, values.to_vec()).into()
                    } else if let Some(&wid) = ids.get(name) {
                        match entry(wid) {
                            // An index table (a hash router's): its words as they are stored.
                            WeightEntry::Dense(dense) if dense.dtype() == DType::I32 => {
                                let words = dense
                                    .bytes()
                                    .as_slice()
                                    .as_chunks::<4>()
                                    .0
                                    .iter()
                                    .map(|w| i32::from_le_bytes(*w))
                                    .collect();
                                HostTensor::i32(shape, words).into()
                            }
                            // E4M3FN codes: the oracle's carrier keeps them packed, not widened.
                            WeightEntry::Dense(dense) if dense.dtype() == DType::E4M3FN => {
                                Value::Host(
                                    poot_eval::fp8::e4m3fn_tensor(
                                        dense.shape().to_vec(),
                                        dense.bytes().as_slice().to_vec(),
                                    )
                                    .unwrap(),
                                )
                            }
                            WeightEntry::Dense(dense) => {
                                let mut one = WeightStore::builder();
                                one.insert("w", WeightEntry::Dense(dense)).unwrap();
                                Value::Host(
                                    poot_eval::materialize_dense(&one.build(), "w").unwrap(),
                                )
                            }
                            WeightEntry::Packed(_) => {
                                panic!("{name} is packed but the graph declares it as a const")
                            }
                        }
                    } else if let Some(source) = PackedSourceName::parse(name) {
                        let wid = ids[source.linear_id()];
                        if matches!(map.view(wid), Some(WeightView::ExpertStack(_))) {
                            let stack = stacks.entry(wid).or_insert_with(|| {
                                Arc::new(
                                    map.materialize_stack(wid, store)
                                        .unwrap_or_else(|e| panic!("{wid}: {e}")),
                                )
                            });
                            Value::PackedStack(PackedStackComponentRef::new(
                                Arc::clone(stack),
                                source.role(),
                            ))
                        } else {
                            let WeightEntry::Packed(payload) = entry(wid) else {
                                panic!("{name} names a dense weight")
                            };
                            let role = payload
                                .weight()
                                .sources()
                                .into_iter()
                                .find(|&role| {
                                    PackedSourceName::new(source.linear_id(), role) == source
                                })
                                .unwrap();
                            Value::Packed(PackedComponentRef::new(Arc::clone(&payload), role))
                        }
                    } else {
                        panic!("no value for const {name}")
                    }
                }
                other => panic!("unexpected input storage {other:?}"),
            };
            inputs.insert(id, value);
        }
        let eval = poot_eval::eval(g, &inputs, EvalOptions::default())
            .unwrap_or_else(|e| panic!("oracle: {e}"));
        let dense = |v: Value| {
            let host = v.into_host().unwrap();
            match host.as_i32() {
                Some(words) => words.iter().map(|&w| w as f32).collect(),
                None => host.as_f32().unwrap().to_vec(),
            }
        };
        (
            dense(eval.output),
            eval.state.into_iter().map(dense).collect(),
        )
    }

    /// One step of `model` on the oracle: trace its `step` program at `rows = 1` with logits at every token (the last one for a single token),
    /// `tokens.len()` tokens from absolute position `start` over a cache of `capacity`; bind its
    /// packed weights through the packed-weight transform fed by the model's map; evaluate over
    /// `state` (zeros when empty).
    /// Returns the logits and the carried state.
    pub(crate) fn run_step(
        model: &dyn crate::model::Model,
        store: &WeightStore,
        capacity: usize,
        tokens: &[i32],
        start: i32,
        state: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        // A one-token step is the decode step, whose shape asks for the last position's logits (the
        // only position it has): a family with a prefill granule refuses it otherwise.
        let shape = |tokens, capacity| {
            let logits = if tokens == 1 {
                LogitRows::Last
            } else {
                LogitRows::All
            };
            step_shape(tokens, capacity, logits, None)
        };
        run_shape(
            model,
            store,
            &StepSpec::step(shape(tokens.len(), capacity)),
            tokens,
            start,
            state,
        )
    }

    /// The `rows = 1` contiguous shape of a step of `tokens` tokens over `capacity` positions.
    pub(crate) fn step_shape(
        tokens: usize,
        capacity: usize,
        logits: LogitRows,
        hidden: Option<HiddenCarry>,
    ) -> StepShape {
        StepShape::new(
            std::num::NonZeroUsize::MIN,
            std::num::NonZeroUsize::new(tokens).unwrap(),
            std::num::NonZeroUsize::new(capacity).unwrap(),
        )
        .with_logits(logits)
        .with_hidden(hidden)
    }

    /// [`run_step`] over a given `spec` (program, shape, carried inputs): the carried state is
    /// `model`'s, in carry order, the hidden-state carry last when the shape asks for one.
    pub(crate) fn run_shape(
        model: &dyn crate::model::Model,
        store: &WeightStore,
        spec: &StepSpec,
        tokens: &[i32],
        start: i32,
        state: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        run_shape_under(
            poot_graph_ir::TraceLimits::for_tests(),
            model,
            store,
            spec,
            tokens,
            start,
            state,
        )
    }

    /// [`run_shape`] with the step traced under the caller's `limits`.
    pub(crate) fn run_shape_under(
        limits: poot_graph_ir::TraceLimits,
        model: &dyn crate::model::Model,
        store: &WeightStore,
        spec: &StepSpec,
        tokens: &[i32],
        start: i32,
        state: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        assert_eq!(
            spec.shape.tokens.get(),
            tokens.len(),
            "one token per position"
        );
        let g = model.trace(spec, limits).unwrap();
        let formats = poot_graph_plan::WeightFormats::from_weight_map(model.weights());
        let g = poot_graph_plan::bind_packed_weights(&g, &formats).unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state()
            .map(|(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let state: Vec<&[f32]> = if state.is_empty() { &zeros } else { state }
            .iter()
            .map(Vec::as_slice)
            .collect();
        let pos: Vec<i32> = (start..).take(tokens.len()).collect();
        // A sectioned model rotates every token at its text position on all three axes, which is
        // plain RoPE (what the driver binds for a prompt with no image).
        let mrope = pos.repeat(3);
        eval(
            &g,
            store,
            model.weights(),
            &[],
            &[
                (Slot::Token, tokens),
                (Slot::Pos, &pos),
                (Slot::MropePosition, &mrope),
            ],
            &state,
        )
    }

    /// Every element of `got` within `tol` (absolute, plus `tol` relative) of the f64 reference;
    /// NaN fails.
    pub(crate) fn assert_matches_f64(got: &[f32], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "element count");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let g = f64::from(g);
            assert!(
                (g - w).abs() <= tol * (1.0 + w.abs()),
                "element {i}: got {g}, reference {w}"
            );
        }
    }

    /// A perturbation of `id` changed the output somewhere.
    pub(crate) fn assert_differs(perturbed: &[f32], base: &[f32], id: WeightId) {
        assert_eq!(perturbed.len(), base.len());
        let changed = perturbed
            .iter()
            .zip(base)
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();
        assert!(changed > 0, "the output does not depend on {id}");
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use poot_quant::weights::{AttnRole, FfnRole};

    use super::super::norm::NormKind;
    use super::oracle::{self, Fx};
    use super::*;

    fn rms(x: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
        let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        x.iter()
            .zip(w)
            .map(|(v, &w)| v * inv * f64::from(w))
            .collect()
    }

    /// SC-001: `standard_layer` over two stand-in linear closures against
    /// `x + m(rms(x)) ; x + f(rms(x))` in f64; each norm weight changes the output.
    #[test]
    fn standard_layer_matches_an_f64_pre_norm_residual_pair() {
        let (t, w) = (3, 6);
        let eps = 1e-5;
        let attn = WeightId::layer(0, WeightRole::Norm(NormRole::Attn));
        let ffn = WeightId::layer(0, WeightRole::Norm(NormRole::Ffn));
        let mix = WeightId::layer(0, WeightRole::Attn(AttnRole::O));
        let feed = WeightId::layer(0, WeightRole::Ffn(FfnRole::Down));
        let mut fx = Fx::new(3);
        fx.f32(attn, &[w]);
        fx.f32(ffn, &[w]);
        fx.f32(mix, &[w, w]);
        fx.f32(feed, &[w, w]);
        let x = oracle::random(t * w, 5);
        let run = |fx: &Fx| {
            let map = fx.map();
            let norms = LayerNorms::new(&map, 0, w).unwrap();
            let (m, f) = (
                Weight::new(&map, mix, &[w, w]).unwrap(),
                Weight::new(&map, feed, &[w, w]).unwrap(),
            );
            let b = Builder::for_tests();
            let xs = b.constant("x", TensorType::f32(vec![t, w])).unwrap();
            let y = standard_layer(
                &b,
                xs,
                &norms,
                &LayerParams::new(NormParams::new(eps as f32).unwrap()),
                |b, h| super::super::linear::linear(b, h, &m, None),
                |b, BlockInput { block: h, .. }| super::super::linear::linear(b, h, &f, None),
            )
            .unwrap();
            fx.eval(&b.finish(y).unwrap(), &[("x", &x)], &[], &[]).0
        };
        let got = run(&fx);
        let mat = |id, v: &[f64]| -> Vec<f64> {
            let m = fx.values(id);
            (0..w)
                .map(|o| (0..w).map(|j| v[j] * f64::from(m[o * w + j])).sum())
                .collect()
        };
        let eps = f64::from(eps as f32);
        let mut want = Vec::new();
        for row in x.chunks(w) {
            let x0: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            let m = mat(mix, &rms(&x0, &fx.values(attn), eps));
            let x1: Vec<f64> = x0.iter().zip(&m).map(|(a, b)| a + b).collect();
            let f = mat(feed, &rms(&x1, &fx.values(ffn), eps));
            want.extend(x1.iter().zip(&f).map(|(a, b)| a + b));
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        for id in [attn, ffn] {
            let mut p = fx.clone();
            p.perturb(id);
            oracle::assert_differs(&run(&p), &got, id);
        }
    }

    /// SC-001: `standard_stack` with an identity layer is `head(rms(embed(tokens)))` in f64, at
    /// every position and (under `LogitRows::Last`) at the last; each model-level weight matters.
    #[test]
    fn standard_stack_matches_an_f64_embed_norm_head_at_every_and_the_last_position() {
        let (vocab, w, t) = (7, 4, 3);
        let eps = 1e-6f32;
        let embed = WeightId::model(WeightRole::Embed);
        let norm = WeightId::model(WeightRole::FinalNorm);
        let head = WeightId::model(WeightRole::Head);
        let mut fx = Fx::new(9);
        fx.bf16(embed, &[vocab, w]);
        fx.f32(norm, &[w]);
        fx.f32(head, &[vocab, w]);
        let tokens = [4, 0, 6];
        let run = |fx: &Fx, logits: LogitRows| {
            let map = fx.map();
            let sw = StackWeights::new(&map, vocab, w).unwrap();
            let shape = StepShape::new(
                NonZeroUsize::MIN,
                NonZeroUsize::new(t).unwrap(),
                NonZeroUsize::new(8).unwrap(),
            )
            .with_logits(logits);
            let b = Builder::for_tests();
            let step = Step::new(&b, shape).unwrap();
            let out = standard_stack(
                &b,
                &step,
                &sw,
                &StackParams::new(NormParams::new(eps).unwrap()),
                2,
                |_, _, x| Ok(x),
            )
            .unwrap();
            let g = step.finish(b, out).unwrap();
            assert_eq!(g.state().count(), 0);
            fx.eval(
                &g,
                &[],
                &[(Slot::Token, &tokens), (Slot::Pos, &[0, 1, 2])],
                &[],
            )
            .0
        };
        let (e, n, h) = (fx.values(embed), fx.values(norm), fx.values(head));
        let mut want = Vec::new();
        for &tok in &tokens {
            let x: Vec<f64> = e[tok as usize * w..][..w]
                .iter()
                .map(|&v| v.into())
                .collect();
            let y = rms(&x, &n, f64::from(eps));
            want.extend(
                (0..vocab).map(|v| (0..w).map(|j| y[j] * f64::from(h[v * w + j])).sum::<f64>()),
            );
        }
        let all = run(&fx, LogitRows::All);
        oracle::assert_matches_f64(&all, &want, 1e-5);
        oracle::assert_matches_f64(&run(&fx, LogitRows::Last), &want[(t - 1) * vocab..], 1e-5);
        for id in [embed, norm, head] {
            let mut p = fx.clone();
            p.perturb(id);
            oracle::assert_differs(&run(&p, LogitRows::All), &all, id);
        }
    }

    /// POOT-1109 SC-001 (standard stack): a step that asks for the hidden-state carry writes the closing
    /// norm's output at every position into a `Handoff` pool of the asked `positions`, before the
    /// head narrows to the last one; a following narrower step rewrites its own positions and keeps the
    /// rest. The head's logits are the carry-less step's. Mutation: write the pre-norm state
    /// (`carry_hidden(b, x)` before the norm); every element of the first assertion moves.
    #[test]
    fn the_hidden_carry_holds_the_post_norm_state_at_every_position_before_the_last_slice() {
        use crate::model::HiddenCarry;
        let (vocab, w, positions) = (7, 4, 5);
        let eps = 1e-6f32;
        let embed = WeightId::model(WeightRole::Embed);
        let norm = WeightId::model(WeightRole::FinalNorm);
        let head = WeightId::model(WeightRole::Head);
        let mut fx = Fx::new(11);
        fx.bf16(embed, &[vocab, w]);
        fx.f32(norm, &[w]);
        fx.f32(head, &[vocab, w]);
        let run = |tokens: &[i32], start: i32, hidden: bool, state: &[Vec<f32>]| {
            let map = fx.map();
            let sw = StackWeights::new(&map, vocab, w).unwrap();
            let shape = StepShape::new(
                NonZeroUsize::MIN,
                NonZeroUsize::new(tokens.len()).unwrap(),
                NonZeroUsize::new(8).unwrap(),
            )
            .with_hidden(hidden.then(|| HiddenCarry {
                positions: NonZeroUsize::new(positions).unwrap(),
            }));
            let b = Builder::for_tests();
            let step = Step::new(&b, shape).unwrap();
            let out = standard_stack(
                &b,
                &step,
                &sw,
                &StackParams::new(NormParams::new(eps).unwrap()),
                2,
                |_, _, x| Ok(x),
            )
            .unwrap();
            let g = step.finish(b, out).unwrap();
            let pos: Vec<i32> = (start..).take(tokens.len()).collect();
            let zeros = vec![vec![0.0; positions * w]];
            let state: Vec<&[f32]> = if state.is_empty() { &zeros } else { state }
                .iter()
                .map(Vec::as_slice)
                .collect();
            let state = if hidden { state } else { Vec::new() };
            let roles: Vec<_> = g.state_pairs().map(|pair| pair.role).collect();
            assert_eq!(
                roles,
                if hidden {
                    vec![StateRole::Handoff]
                } else {
                    vec![]
                },
                "the carry is the one carried value, and a step that asks for none carries none"
            );
            fx.eval(&g, &[], &[(Slot::Token, tokens), (Slot::Pos, &pos)], &state)
        };
        let (e, n) = (fx.values(embed), fx.values(norm));
        let normed = |token: i32| -> Vec<f64> {
            let x: Vec<f64> = e[token as usize * w..][..w]
                .iter()
                .map(|&v| v.into())
                .collect();
            rms(&x, &n, f64::from(eps))
        };

        let first = [4, 0, 6];
        let (logits, state) = run(&first, 0, true, &[]);
        let mut want: Vec<f64> = first.iter().flat_map(|&t| normed(t)).collect();
        want.resize(positions * w, 0.0);
        oracle::assert_matches_f64(&state[0], &want, 1e-5);
        let (plain, no_state) = run(&first, 0, false, &[]);
        assert!(no_state.is_empty());
        oracle::assert_matches_f64(
            &logits,
            &plain.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            1e-6,
        );

        // A two-token window after it: positions 0..2 are the window's, 2..5 stay what they were.
        let second = [2, 5];
        let (_, after) = run(&second, 3, true, &state);
        let mut want: Vec<f64> = second.iter().flat_map(|&t| normed(t)).collect();
        want.extend(state[0][2 * w..].iter().map(|&v| f64::from(v)));
        oracle::assert_matches_f64(&after[0], &want, 1e-5);
    }

    /// POOT-1109 SC-003: a hidden-state carry with fewer positions than the step has tokens is a typed
    /// shape refusal before any graph is built. Mutation: drop the `HiddenCarryNarrow` return in
    /// `check_step`; the step traces and the assertion is red.
    #[test]
    fn a_carry_narrower_than_the_step_is_refused_before_tracing() {
        use crate::model::HiddenCarry;
        let family = FamilyKey::new("fixture");
        let carry = |positions| {
            Some(HiddenCarry {
                positions: NonZeroUsize::new(positions).unwrap(),
            })
        };
        let mut shape = oracle::step_shape(4, 8, LogitRows::Last, None);
        assert!(check_step(family, shape, 8).is_ok());
        shape.hidden = carry(3);
        assert!(matches!(
            check_step(family, shape, 8),
            Err(TraceError::ShapeUnsupported {
                reason: ShapeReason::HiddenCarryNarrow {
                    positions: 3,
                    tokens: 4
                },
                ..
            })
        ));
        shape.hidden = carry(4);
        assert!(check_step(family, shape, 8).is_ok());
    }

    /// SC-001: the post-norm placement with a residual scale against its f64 definition
    /// `x += s * rms(mixer(x)); x += s * rms(ffn(x))`; the pre-norm output differs, and each norm
    /// weight matters.
    #[test]
    fn post_norm_placement_and_residual_scale_match_an_f64_reference() {
        let (t, w) = (3, 6);
        let (eps, scale) = (1e-5f32, 0.5f32);
        let attn = WeightId::layer(0, WeightRole::Norm(NormRole::Attn));
        let ffn = WeightId::layer(0, WeightRole::Norm(NormRole::Ffn));
        let mix = WeightId::layer(0, WeightRole::Attn(AttnRole::O));
        let feed = WeightId::layer(0, WeightRole::Ffn(FfnRole::Down));
        let mut fx = Fx::new(13);
        fx.f32(attn, &[w]);
        fx.f32(ffn, &[w]);
        fx.f32(mix, &[w, w]);
        fx.f32(feed, &[w, w]);
        let x = oracle::random(t * w, 15);
        let run = |fx: &Fx, placement| {
            let map = fx.map();
            let norms = LayerNorms::new(&map, 0, w).unwrap();
            let (m, f) = (
                Weight::new(&map, mix, &[w, w]).unwrap(),
                Weight::new(&map, feed, &[w, w]).unwrap(),
            );
            let params = LayerParams::new(NormParams::new(eps).unwrap())
                .with_placement(placement)
                .with_residual_scale(scale)
                .unwrap();
            let b = Builder::for_tests();
            let xs = b.constant("x", TensorType::f32(vec![t, w])).unwrap();
            let y = standard_layer(
                &b,
                xs,
                &norms,
                &params,
                |b, h| super::super::linear::linear(b, h, &m, None),
                |b, BlockInput { block: h, .. }| super::super::linear::linear(b, h, &f, None),
            )
            .unwrap();
            fx.eval(&b.finish(y).unwrap(), &[("x", &x)], &[], &[]).0
        };
        let got = run(&fx, NormPlacement::Post);
        let mat = |id, v: &[f64]| -> Vec<f64> {
            let m = fx.values(id);
            (0..w)
                .map(|o| (0..w).map(|j| v[j] * f64::from(m[o * w + j])).sum())
                .collect()
        };
        let eps = f64::from(eps);
        let mut want = Vec::new();
        for row in x.chunks(w) {
            let x0: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            let m = rms(&mat(mix, &x0), &fx.values(attn), eps);
            let x1: Vec<f64> = x0.iter().zip(&m).map(|(a, b)| a + 0.5 * b).collect();
            let f = rms(&mat(feed, &x1), &fx.values(ffn), eps);
            want.extend(x1.iter().zip(&f).map(|(a, b)| a + 0.5 * b));
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        assert_ne!(got, run(&fx, NormPlacement::Pre), "the placements differ");
        for id in [attn, ffn] {
            let mut p = fx.clone();
            p.perturb(id);
            oracle::assert_differs(&run(&p, NormPlacement::Post), &got, id);
        }
    }

    /// SC-001: the stack's embedding multiplier, logit divisor and final softcap against
    /// `cap * tanh((head(rms(s * embed(tokens))) / d) / cap)` in f64; each option changes the output.
    #[test]
    fn stack_scales_match_an_f64_reference() {
        let (vocab, w, t) = (7, 4, 2);
        let eps = 1e-6f32;
        let (embed_scale, divisor, cap) = (2.0f32, 4.0f32, 0.5f32);
        let embed = WeightId::model(WeightRole::Embed);
        let norm_id = WeightId::model(WeightRole::FinalNorm);
        let head = WeightId::model(WeightRole::Head);
        let mut fx = Fx::new(19);
        fx.bf16(embed, &[vocab, w]);
        fx.f32(norm_id, &[w]);
        fx.f32(head, &[vocab, w]);
        let tokens = [4, 1];
        let run = |params: StackParams| {
            let map = fx.map();
            let sw = StackWeights::new(&map, vocab, w).unwrap();
            let shape = StepShape::new(
                NonZeroUsize::MIN,
                NonZeroUsize::new(t).unwrap(),
                NonZeroUsize::new(8).unwrap(),
            )
            .with_logits(LogitRows::All);
            let b = Builder::for_tests();
            let step = Step::new(&b, shape).unwrap();
            let out = standard_stack(&b, &step, &sw, &params, 1, |_, _, x| Ok(x)).unwrap();
            fx.eval(
                &step.finish(b, out).unwrap(),
                &[],
                &[(Slot::Token, &tokens), (Slot::Pos, &[0, 1])],
                &[],
            )
            .0
        };
        let base = StackParams::new(NormParams::new(eps).unwrap());
        let scaled = base
            .with_embed_scale(embed_scale)
            .unwrap()
            .with_logit_divisor(divisor)
            .unwrap()
            .with_softcap(cap)
            .unwrap();
        let (e, n, h) = (fx.values(embed), fx.values(norm_id), fx.values(head));
        let mut want = Vec::new();
        for &tok in &tokens {
            let x: Vec<f64> = e[tok as usize * w..][..w]
                .iter()
                .map(|&v| f64::from(v) * f64::from(embed_scale))
                .collect();
            let y = rms(&x, &n, f64::from(eps));
            want.extend((0..vocab).map(|v| {
                let logit = (0..w).map(|j| y[j] * f64::from(h[v * w + j])).sum::<f64>()
                    / f64::from(divisor);
                f64::from(cap) * (logit / f64::from(cap)).tanh()
            }));
        }
        let got = run(scaled);
        oracle::assert_matches_f64(&got, &want, 1e-5);
        let plain = run(base);
        for option in [
            base.with_embed_scale(embed_scale).unwrap(),
            base.with_logit_divisor(divisor).unwrap(),
            base.with_softcap(cap).unwrap(),
        ] {
            assert_ne!(run(option), plain, "{option:?} changes the logits");
        }
    }

    /// SC-001: the sandwich placement against its f64 definition
    /// `x += post(mixer(pre(x))); x += post(ffn(pre(x)))` with the `1 + w` norm kind; each of the
    /// four norm weights matters.
    #[test]
    fn sandwich_placement_matches_an_f64_reference() {
        let (t, w) = (3, 6);
        let eps = 1e-5f32;
        let roles = [
            NormRole::Attn,
            NormRole::PostAttn,
            NormRole::Ffn,
            NormRole::PostFfn,
        ];
        let id = |r| WeightId::layer(0, WeightRole::Norm(r));
        let mix = WeightId::layer(0, WeightRole::Attn(AttnRole::O));
        let feed = WeightId::layer(0, WeightRole::Ffn(FfnRole::Down));
        let mut fx = Fx::new(23);
        for r in roles {
            fx.f32(id(r), &[w]);
        }
        fx.f32(mix, &[w, w]);
        fx.f32(feed, &[w, w]);
        let x = oracle::random(t * w, 25);
        let run = |fx: &Fx| {
            let map = fx.map();
            let norms = LayerNorms::sandwich(&map, 0, w).unwrap();
            let (m, f) = (
                Weight::new(&map, mix, &[w, w]).unwrap(),
                Weight::new(&map, feed, &[w, w]).unwrap(),
            );
            let params = LayerParams::new(
                NormParams::of(super::super::norm::NormKind::RmsPlusOne, eps).unwrap(),
            )
            .with_placement(NormPlacement::Sandwich);
            let b = Builder::for_tests();
            let xs = b.constant("x", TensorType::f32(vec![t, w])).unwrap();
            let y = standard_layer(
                &b,
                xs,
                &norms,
                &params,
                |b, h| super::super::linear::linear(b, h, &m, None),
                |b, BlockInput { block: h, .. }| super::super::linear::linear(b, h, &f, None),
            )
            .unwrap();
            fx.eval(&b.finish(y).unwrap(), &[("x", &x)], &[], &[]).0
        };
        let got = run(&fx);
        let mat = |id, v: &[f64]| -> Vec<f64> {
            let m = fx.values(id);
            (0..w)
                .map(|o| (0..w).map(|j| v[j] * f64::from(m[o * w + j])).sum())
                .collect()
        };
        let one_plus = |r| -> Vec<f32> { fx.values(id(r)).iter().map(|v| 1.0 + v).collect() };
        let eps = f64::from(eps);
        let mut want = Vec::new();
        for row in x.chunks(w) {
            let x0: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            let m = rms(
                &mat(mix, &rms(&x0, &one_plus(NormRole::Attn), eps)),
                &one_plus(NormRole::PostAttn),
                eps,
            );
            let x1: Vec<f64> = x0.iter().zip(&m).map(|(a, b)| a + b).collect();
            let f = rms(
                &mat(feed, &rms(&x1, &one_plus(NormRole::Ffn), eps)),
                &one_plus(NormRole::PostFfn),
                eps,
            );
            want.extend(x1.iter().zip(&f).map(|(a, b)| a + b));
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        for r in roles {
            let mut p = fx.clone();
            p.perturb(id(r));
            oracle::assert_differs(&run(&p), &got, id(r));
        }
    }

    #[test]
    fn a_non_positive_scale_is_refused_by_field() {
        let norm = NormParams::new(1e-6).unwrap();
        assert_eq!(
            StackParams::new(norm).with_embed_scale(0.0),
            Err(ParamError::new(
                "embed_scale",
                ConfigReason::NotFinitePositive
            ))
        );
        assert_eq!(
            LayerParams::new(norm).with_residual_scale(f32::NAN),
            Err(ParamError::new(
                "residual_scale",
                ConfigReason::NotFinitePositive
            ))
        );
    }

    fn layer_norm(x: &[f64], scale: &[f32], bias: &[f32], eps: f64) -> Vec<f64> {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let inv = 1.0 / (var + eps).sqrt();
        (0..x.len())
            .map(|i| (x[i] - mean) * inv * f64::from(scale[i]) + f64::from(bias[i]))
            .collect()
    }

    /// SC-001: a layer with biased LayerNorms (`x += m(ln(x)); x += f(ln(x))`) against f64; each
    /// norm scale and bias changes the output.
    #[test]
    fn a_layer_with_biased_layer_norms_matches_an_f64_reference() {
        let (t, w) = (3, 6);
        let eps = 1e-5f32;
        let norm_id = |role| WeightId::layer(0, WeightRole::Norm(role));
        let mix = WeightId::layer(0, WeightRole::Attn(AttnRole::O));
        let feed = WeightId::layer(0, WeightRole::Ffn(FfnRole::Down));
        let mut fx = Fx::new(23);
        let roles = [
            NormRole::Attn,
            NormRole::AttnBias,
            NormRole::Ffn,
            NormRole::FfnBias,
        ];
        for role in roles {
            fx.f32(norm_id(role), &[w]);
        }
        fx.f32(mix, &[w, w]);
        fx.f32(feed, &[w, w]);
        let x = oracle::random(t * w, 25);
        let run = |fx: &Fx| {
            let map = fx.map();
            let norms = LayerNorms::biased(&map, 0, w).unwrap();
            let (m, f) = (
                Weight::new(&map, mix, &[w, w]).unwrap(),
                Weight::new(&map, feed, &[w, w]).unwrap(),
            );
            let params = LayerParams::new(NormParams::of(NormKind::Layer, eps).unwrap());
            let b = Builder::for_tests();
            let xs = b.constant("x", TensorType::f32(vec![t, w])).unwrap();
            let y = standard_layer(
                &b,
                xs,
                &norms,
                &params,
                |b, h| super::super::linear::linear(b, h, &m, None),
                |b, BlockInput { block: h, .. }| super::super::linear::linear(b, h, &f, None),
            )
            .unwrap();
            fx.eval(&b.finish(y).unwrap(), &[("x", &x)], &[], &[]).0
        };
        let got = run(&fx);
        let mat = |id, v: &[f64]| -> Vec<f64> {
            let m = fx.values(id);
            (0..w)
                .map(|o| (0..w).map(|j| v[j] * f64::from(m[o * w + j])).sum())
                .collect()
        };
        let eps = f64::from(eps);
        let v = |role| fx.values(norm_id(role));
        let mut want = Vec::new();
        for row in x.chunks(w) {
            let x0: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            let m = mat(
                mix,
                &layer_norm(&x0, &v(NormRole::Attn), &v(NormRole::AttnBias), eps),
            );
            let x1: Vec<f64> = x0.iter().zip(&m).map(|(a, b)| a + b).collect();
            let f = mat(
                feed,
                &layer_norm(&x1, &v(NormRole::Ffn), &v(NormRole::FfnBias), eps),
            );
            want.extend(x1.iter().zip(&f).map(|(a, b)| a + b));
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        for role in roles {
            let mut p = fx.clone();
            p.perturb(norm_id(role));
            oracle::assert_differs(&run(&p), &got, norm_id(role));
        }
    }

    /// SC-001: a stack with an embedding LayerNorm and a biased final LayerNorm,
    /// `head(ln_f(ln_e(embed(tokens))))`, against f64; each of the four norm weights matters.
    #[test]
    fn an_embed_norm_and_a_biased_final_norm_match_an_f64_reference() {
        let (vocab, w, t) = (7, 4, 2);
        let eps = 1e-5f32;
        let ids = [
            WeightId::model(WeightRole::Embed),
            WeightId::model(WeightRole::FinalNorm),
            WeightId::model(WeightRole::FinalNormBias),
            WeightId::model(WeightRole::EmbedNorm),
            WeightId::model(WeightRole::EmbedNormBias),
            WeightId::model(WeightRole::Head),
        ];
        let [embed, fnorm, fbias, enorm, ebias, head] = ids;
        let mut fx = Fx::new(29);
        fx.bf16(embed, &[vocab, w]);
        for id in [fnorm, fbias, enorm, ebias] {
            fx.f32(id, &[w]);
        }
        fx.f32(head, &[vocab, w]);
        let tokens = [5, 2];
        let run = |fx: &Fx| {
            let map = fx.map();
            let sw = StackWeights::new(&map, vocab, w)
                .unwrap()
                .with_final_norm_bias(&map, w)
                .unwrap()
                .with_embed_norm(&map, w)
                .unwrap();
            let shape = StepShape::new(
                NonZeroUsize::MIN,
                NonZeroUsize::new(t).unwrap(),
                NonZeroUsize::new(8).unwrap(),
            )
            .with_logits(LogitRows::All);
            let b = Builder::for_tests();
            let step = Step::new(&b, shape).unwrap();
            let params = StackParams::new(NormParams::of(NormKind::Layer, eps).unwrap());
            let out = standard_stack(&b, &step, &sw, &params, 1, |_, _, x| Ok(x)).unwrap();
            fx.eval(
                &step.finish(b, out).unwrap(),
                &[],
                &[(Slot::Token, &tokens), (Slot::Pos, &[0, 1])],
                &[],
            )
            .0
        };
        let got = run(&fx);
        let (e, h) = (fx.values(embed), fx.values(head));
        let mut want = Vec::new();
        for &tok in &tokens {
            let x: Vec<f64> = e[tok as usize * w..][..w]
                .iter()
                .map(|&v| v.into())
                .collect();
            let x = layer_norm(&x, &fx.values(enorm), &fx.values(ebias), f64::from(eps));
            let y = layer_norm(&x, &fx.values(fnorm), &fx.values(fbias), f64::from(eps));
            want.extend(
                (0..vocab).map(|v| (0..w).map(|j| y[j] * f64::from(h[v * w + j])).sum::<f64>()),
            );
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        for id in [fnorm, fbias, enorm, ebias] {
            let mut p = fx.clone();
            p.perturb(id);
            oracle::assert_differs(&run(&p), &got, id);
        }
    }
}
