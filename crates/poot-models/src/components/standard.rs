//! The shared residual pair every standard decoder repeats (spec 999, ADR-0105): [`standard_layer`]
//! (`x += mixer(norm(x)); x += ffn(norm(x))`) and [`standard_stack`] (embedding, one
//! `layer_scope` per layer, final norm, head). A family's tracer is a short call of these with its
//! own components in the closures; neither names a family.
//!
//! [`Step`] carries what one traced step shares across its layers: its shape, the token and
//! position inputs, and the state pairs the layers' components carry (an attention cache). The
//! step's inputs are typed by role: `Slot::Token` and `Slot::Pos`, both `[rows, tokens]` I32, the
//! positions absolute, so prefill and decode are one body continuing from `Pos`.

use std::cell::RefCell;

use poot_graph_ir::{BinOp, Builder, Graph, Slot, TensorType, Traced, ValidationOutputs, ops};
use poot_quant::weights::{NormRole, WeightId, WeightMap, WeightRole};
use poot_tensor::DType;

use super::embed::Embedding;
use super::head::Head;
use super::linear::{Weight, WeightError};
use super::norm::{NormParams, NormWeights, norm, norm_with};
use crate::model::{
    ConfigReason, FamilyKey, KvLayout, LogitRows, ModelError, Phase, ShapeReason, StepShape,
    TraceError,
};

/// A component parameter that cannot configure it, naming the field (spec 999: parameters are
/// validated once, at construction, so tracing with them does not panic).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("parameter {field}: {reason:?}")]
pub struct ParamError {
    pub field: &'static str,
    pub reason: ConfigReason,
}

impl ParamError {
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
    paged: Option<PagedMaps>,
    state: RefCell<Vec<(Traced, Traced)>>,
}

/// The two slot maps of a [`KvLayout::Paged`] step, both I32 `Slot::SlotMap` inputs.
#[derive(Clone, Copy, Debug)]
pub struct PagedMaps {
    /// `[rows, capacity]`: row `r`'s logical position to the pool slot holding its K/V.
    pub read: Traced,
    /// `[pool_slots]`: pool slot to the flat `[rows * tokens]` index of the token whose K/V this step
    /// writes there, or -1 to keep the slot as it is.
    pub write: Traced,
}

impl Step {
    /// Declare the step's inputs on `b`: the token ids and their absolute positions, both
    /// `[rows, tokens]` I32, and under a paged layout the two slot maps.
    pub fn new(b: &Builder, shape: StepShape) -> Self {
        let ids = TensorType::new(vec![shape.rows.get(), shape.tokens.get()], DType::I32);
        let paged = match shape.kv {
            KvLayout::Contiguous => None,
            KvLayout::Paged { pool_slots } => Some(PagedMaps {
                read: b.slot_named(
                    Slot::SlotMap,
                    "read",
                    TensorType::new(vec![shape.rows.get(), shape.capacity.get()], DType::I32),
                ),
                write: b.slot_named(
                    Slot::SlotMap,
                    "write",
                    TensorType::new(vec![pool_slots.get()], DType::I32),
                ),
            }),
        };
        Self {
            shape,
            tokens: b.slot(Slot::Token, ids.clone()),
            pos: b.slot(Slot::Pos, ids),
            paged,
            state: RefCell::new(Vec::new()),
        }
    }

    /// The slot maps of a paged step; `None` under [`KvLayout::Contiguous`].
    pub fn paged(&self) -> Option<PagedMaps> {
        self.paged
    }

    pub fn shape(&self) -> StepShape {
        self.shape
    }

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

    /// The step's graph: `out` plus every carried state pair, in carry order.
    pub fn finish(self, b: Builder, out: Traced) -> Graph<ValidationOutputs> {
        b.finish_with_state(out, &self.state.into_inner())
            .with_validations(Vec::new())
    }
}

/// The two norms of a standard layer, and for a sandwich layer the two norms applied to each
/// block's output ([`LayerNorms::sandwich`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerNorms {
    pub attn: NormWeights,
    pub ffn: NormWeights,
    post: Option<[Weight; 2]>,
}

impl LayerNorms {
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
            Ok(NormWeights {
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
    pub fn new(norm: NormParams) -> Self {
        Self {
            norm,
            placement: NormPlacement::Pre,
            residual_scale: None,
        }
    }

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
    mixer: impl FnOnce(&Builder, Traced) -> Traced,
    ffn: impl FnOnce(&Builder, Traced) -> Traced,
) -> Traced {
    let post = |i: usize| norms.post.as_ref().map(|post| &post[i]);
    let x = residual_block(b, x, &norms.attn, post(0), p, mixer);
    residual_block(b, x, &norms.ffn, post(1), p, ffn)
}

fn residual_block(
    b: &Builder,
    x: Traced,
    w: &NormWeights,
    post: Option<&Weight>,
    p: &LayerParams,
    block: impl FnOnce(&Builder, Traced) -> Traced,
) -> Traced {
    let out = match p.placement {
        NormPlacement::Pre => block(b, norm_with(b, x, w, p.norm)),
        NormPlacement::Post => norm_with(b, block(b, x), w, p.norm),
        NormPlacement::Sandwich => {
            let post = post.expect("a Sandwich layer is built with LayerNorms::sandwich");
            norm(b, block(b, norm_with(b, x, w, p.norm)), post, p.norm)
        }
    };
    let out = match p.residual_scale {
        Some(scale) => b.binary_scalar(BinOp::Mul, out, Builder::f32(scale)),
        None => out,
    };
    b.binary(BinOp::Add, x, out)
}

/// The model-level weights around the layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackWeights {
    pub embed: Embedding,
    pub final_norm: NormWeights,
    /// A LayerNorm over the embedded tokens, before the first layer (BLOOM).
    pub embed_norm: Option<NormWeights>,
    pub head: Head,
}

impl StackWeights {
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
    mut layer: impl FnMut(&Builder, usize, Traced) -> Traced,
) -> Traced {
    let mut x = w.embed.embed(b, step.tokens());
    if let Some(scale) = p.embed_scale {
        x = b.binary_scalar(BinOp::Mul, x, Builder::f32(scale));
    }
    if let Some(embed_norm) = &w.embed_norm {
        x = norm_with(b, x, embed_norm, p.norm);
    }
    for l in 0..layers {
        let _scope = b.layer_scope(l);
        x = layer(b, l, x);
    }
    let tokens = step.shape().tokens.get();
    if step.shape().logits == LogitRows::Last && tokens > 1 {
        x = b.slice(x, 1, tokens - 1, tokens);
    }
    let x = norm_with(b, x, &w.final_norm, p.norm);
    let logits = w.head.logits(b, x);
    let logits = match p.logit_divisor {
        Some(divisor) => b.binary_scalar(BinOp::Div, logits, Builder::f32(divisor)),
        None => logits,
    };
    match p.softcap {
        Some(cap) => ops::softcap(b, logits, cap),
        None => logits,
    }
}

/// Refuse a step shape the dense decoder body cannot trace: a multi-token decode, a capacity above the model's maximum positions or more new tokens than the cache holds.
pub fn check_step(
    family: FamilyKey,
    phase: Phase,
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
    if phase == Phase::Decode && shape.tokens.get() != 1 {
        return refuse(ShapeReason::DecodeTokens);
    }
    if shape.capacity.get() > max_positions {
        return refuse(ShapeReason::CapacityAboveMax { max: max_positions });
    }
    if shape.tokens > shape.capacity {
        return refuse(ShapeReason::TokensAboveCapacity);
    }
    Ok(())
}

/// The CPU-oracle harness the component tests share: a fixture weight store keyed by
/// [`WeightId::const_name`], a binder from graph consts to its entries, and f64 comparisons.
#[cfg(test)]
pub(crate) mod oracle {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    use poot_eval::{EvalBudget, EvalOptions, Value};
    use poot_graph_ir::{Graph, PackedSourceName, Slot, Storage, ValidationChannel};
    use poot_quant::weights::{
        DenseWeight, WeightEntry, WeightId, WeightKey, WeightMap, WeightStore, WeightView,
    };
    use poot_quant::{PackedComponentRef, PackedPayload};
    use poot_tensor::DType;
    use poot_tensor::HostTensor;

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

    #[derive(Clone, Debug)]
    enum Stored {
        F32(Vec<f32>),
        Bf16(Vec<u16>),
        Packed(Arc<PackedPayload>),
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
                Stored::Packed(payload) => {
                    let [rows, k] = payload.weight().shape();
                    let mut out = vec![0.0; rows * k];
                    for (row, chunk) in out.chunks_mut(k).enumerate() {
                        payload.decode_row(row, chunk).unwrap();
                    }
                    out
                }
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
                Stored::Packed(_) => panic!("perturb a dense weight"),
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
                    Stored::Packed(payload) => WeightEntry::Packed(Arc::clone(payload)),
                };
                sb.insert(id.const_name(), entry).unwrap();
            }
            sb.build()
        }

        pub(crate) fn map(&self) -> WeightMap {
            let store = self.store();
            let mut map = WeightMap::builder(&store);
            for id in self.entries.keys() {
                map.map(*id, WeightView::Stored(WeightKey::new(id.const_name())))
                    .unwrap();
            }
            for (id, target) in &self.ties {
                map.map(*id, WeightView::Stored(WeightKey::new(target.const_name())))
                    .unwrap();
            }
            map.build()
        }

        /// Evaluate `g` on the oracle with this fixture's weights; see [`eval`].
        pub(crate) fn eval<V: ValidationChannel>(
            &self,
            g: &Graph<V>,
            consts: &[(&str, &[f32])],
            slots: &[(Slot, &[i32])],
            state: &[&[f32]],
        ) -> (Vec<f32>, Vec<Vec<f32>>) {
            eval(g, &self.store(), &self.map(), consts, slots, state)
        }

        /// [`Fx::eval`] with `named` slots (a [`Slot::SlotMap`]'s `"read"`/`"write"` tags) bound by
        /// their key spelling.
        pub(crate) fn eval_named<V: ValidationChannel>(
            &self,
            g: &Graph<V>,
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
    pub(crate) fn eval<V: ValidationChannel>(
        g: &Graph<V>,
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
    pub(crate) fn eval_named<V: ValidationChannel>(
        g: &Graph<V>,
        store: &WeightStore,
        map: &WeightMap,
        consts: &[(&str, &[f32])],
        slots: &[(Slot, &[i32])],
        named: &[(&str, &[i32])],
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
        assert_eq!(state.len(), g.state.len(), "one value per state input");
        let mut inputs: HashMap<usize, Value> = HashMap::new();
        for (i, &(si, _)) in g.state.iter().enumerate() {
            let shape = g.aval(si).shape.clone();
            inputs.insert(si, HostTensor::f32(shape, state[i].to_vec()).into());
        }
        for &id in &g.inputs {
            let meta = g.meta(id);
            let shape = meta.aval.shape.clone();
            let value: Value = match &meta.storage {
                Storage::State | Storage::Computed(_) => continue,
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
                        let WeightEntry::Packed(payload) = entry(wid) else {
                            panic!("{name} names a dense weight")
                        };
                        let role = payload
                            .weight()
                            .sources()
                            .into_iter()
                            .find(|&role| PackedSourceName::new(source.linear_id(), role) == source)
                            .unwrap();
                        Value::Packed(PackedComponentRef::new(Arc::clone(&payload), role))
                    } else {
                        panic!("no value for const {name}")
                    }
                }
                other => panic!("unexpected input storage {other:?}"),
            };
            inputs.insert(id, value);
        }
        let eval = poot_eval::eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap_or_else(|e| panic!("oracle: {e}"));
        let dense = |v: Value| v.into_host().unwrap().as_f32().unwrap().to_vec();
        (
            dense(eval.output),
            eval.state.into_iter().map(dense).collect(),
        )
    }

    /// One step of `model` on the oracle: trace `phase` at `rows = 1` with logits at every token,
    /// `tokens.len()` tokens from absolute position `start` over a cache of `capacity`; bind its
    /// packed weights through the packed-weight transform fed by the model's map; evaluate over
    /// `state` (zeros when empty).
    /// Returns the logits and the carried state.
    pub(crate) fn run_step(
        model: &dyn crate::model::Model,
        store: &WeightStore,
        phase: crate::model::Phase,
        capacity: usize,
        tokens: &[i32],
        start: i32,
        state: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        use std::num::NonZeroUsize;
        let shape = crate::model::StepShape {
            rows: NonZeroUsize::MIN,
            tokens: NonZeroUsize::new(tokens.len()).unwrap(),
            capacity: NonZeroUsize::new(capacity).unwrap(),
            kv: crate::model::KvLayout::Contiguous,
            logits: crate::model::LogitRows::All,
        };
        let g = model.trace(phase, shape).unwrap();
        let formats = poot_graph_plan::WeightFormats::from_weight_map(model.weights());
        let g = poot_graph_plan::bind_packed_weights(&g, &formats).unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state
            .iter()
            .map(|&(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let state: Vec<&[f32]> = if state.is_empty() { &zeros } else { state }
            .iter()
            .map(Vec::as_slice)
            .collect();
        let pos: Vec<i32> = (start..).take(tokens.len()).collect();
        eval(
            &g,
            store,
            model.weights(),
            &[],
            &[(Slot::Token, tokens), (Slot::Pos, &pos)],
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
    use crate::model::KvLayout;

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
            let b = Builder::new();
            let xs = b.constant("x", TensorType::f32(vec![t, w]));
            let y = standard_layer(
                &b,
                xs,
                &norms,
                &LayerParams::new(NormParams::new(eps as f32).unwrap()),
                |b, h| super::super::linear::linear(b, h, &m, None),
                |b, h| super::super::linear::linear(b, h, &f, None),
            );
            fx.eval(&b.finish(y), &[("x", &x)], &[], &[]).0
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
            let shape = StepShape {
                rows: NonZeroUsize::MIN,
                tokens: NonZeroUsize::new(t).unwrap(),
                capacity: NonZeroUsize::new(8).unwrap(),
                kv: KvLayout::Contiguous,
                logits,
            };
            let b = Builder::new();
            let step = Step::new(&b, shape);
            let out = standard_stack(
                &b,
                &step,
                &sw,
                &StackParams::new(NormParams::new(eps).unwrap()),
                2,
                |_, _, x| x,
            );
            let g = step.finish(b, out);
            assert_eq!(g.state.len(), 0);
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
            let b = Builder::new();
            let xs = b.constant("x", TensorType::f32(vec![t, w]));
            let y = standard_layer(
                &b,
                xs,
                &norms,
                &params,
                |b, h| super::super::linear::linear(b, h, &m, None),
                |b, h| super::super::linear::linear(b, h, &f, None),
            );
            fx.eval(&b.finish(y), &[("x", &x)], &[], &[]).0
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
            let shape = StepShape {
                rows: NonZeroUsize::MIN,
                tokens: NonZeroUsize::new(t).unwrap(),
                capacity: NonZeroUsize::new(8).unwrap(),
                kv: KvLayout::Contiguous,
                logits: LogitRows::All,
            };
            let b = Builder::new();
            let step = Step::new(&b, shape);
            let out = standard_stack(&b, &step, &sw, &params, 1, |_, _, x| x);
            fx.eval(
                &step.finish(b, out),
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
            let b = Builder::new();
            let xs = b.constant("x", TensorType::f32(vec![t, w]));
            let y = standard_layer(
                &b,
                xs,
                &norms,
                &params,
                |b, h| super::super::linear::linear(b, h, &m, None),
                |b, h| super::super::linear::linear(b, h, &f, None),
            );
            fx.eval(&b.finish(y), &[("x", &x)], &[], &[]).0
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
            let b = Builder::new();
            let xs = b.constant("x", TensorType::f32(vec![t, w]));
            let y = standard_layer(
                &b,
                xs,
                &norms,
                &params,
                |b, h| super::super::linear::linear(b, h, &m, None),
                |b, h| super::super::linear::linear(b, h, &f, None),
            );
            fx.eval(&b.finish(y), &[("x", &x)], &[], &[]).0
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
            let shape = StepShape {
                rows: NonZeroUsize::MIN,
                tokens: NonZeroUsize::new(t).unwrap(),
                capacity: NonZeroUsize::new(8).unwrap(),
                kv: KvLayout::Contiguous,
                logits: LogitRows::All,
            };
            let b = Builder::new();
            let step = Step::new(&b, shape);
            let params = StackParams::new(NormParams::of(NormKind::Layer, eps).unwrap());
            let out = standard_stack(&b, &step, &sw, &params, 1, |_, _, x| x);
            fx.eval(
                &step.finish(b, out),
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
