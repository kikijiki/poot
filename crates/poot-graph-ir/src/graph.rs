//! The flat SSA graph: a value table + ordered eqns (the open-form jaxpr analog: binders + body, no data).

use std::collections::HashSet;
use std::fmt;

use crate::error::GraphValidationError;
use crate::op::OpKind;
use crate::rope_table::{self, RopeSpec, RopeTable};
use crate::types::{DType, Scalar, TensorType};

/// An SSA value: an index into [`Graph::values`] (the jaxpr `Var`, an arena index not object identity).
pub type ValueId = usize;

/// A decoder-layer tag on an equation: the layer whose loop emitted it, as reported by
/// [`crate::Builder::layer_scope`].
///
/// Layer tags are the graph-side image of a layer-pipeline placement: `poot-llm`'s
/// `exact_partition` records `after_layer` cuts between devices, and
/// `poot-graph-plan`'s `split_stages_after_layers` maps those cuts onto this graph through the tag,
/// so no caller has to infer a layer from a name. `None` means "emitted outside any layer scope" -
/// the graph-wide preamble (slots, guards, masks), the head, or an equation a transform
/// synthesized. The tag is data, not semantics: an evaluator or planner that does not split by
/// layer ignores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerIndex(pub usize);

/// Stable model-adapter key for one bounded validation witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValidationId(pub u32);

/// One named validation root. The value remains an ordinary F32 graph value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationOutput {
    pub id: ValidationId,
    pub name: String,
    pub value: ValueId,
}

pub const MAX_VALIDATION_PACKET_BYTES: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationPacketEntry {
    pub id: ValidationId,
    pub name: String,
    pub first_lane: usize,
    pub lane_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationPacketLayout {
    pub entries: Vec<ValidationPacketEntry>,
    pub lane_count: usize,
    pub byte_len: usize,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[error("validation {name} ({id:?}) lane {lane} failed with f32 bits 0x{observed_bits:08x}")]
pub struct ExecutionValidationFailure {
    pub id: ValidationId,
    pub name: String,
    pub lane: usize,
    pub observed_bits: u32,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ValidationPacketError {
    #[error("validation packet has {actual} lanes; expected {expected}")]
    Length { expected: usize, actual: usize },
    #[error(transparent)]
    Failure(#[from] ExecutionValidationFailure),
}

/// An operand: an SSA value reference or an inline literal (the jaxpr `Atom`).
#[derive(Clone, Copy, Debug)]
pub enum Operand {
    Value(ValueId),
    Lit(Scalar),
}

/// Runtime inputs. Most variants are graph-architecture.md PerTokenSlots: `Token`/`Pos`/`SeqLen` are scalars,
/// and `Mask` is the additive attention mask over the fixed-capacity cache (G3d), host-filled per token.
/// `Activation` is an explicitly named standalone boundary, not a model-engine slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Slot {
    Token,
    /// A caller-supplied dense activation for a standalone graph boundary. Unlike [`Slot::Token`], it is not a
    /// token id and has no model-engine binding convention. Declare it with [`crate::builder::Builder::slot_named`]
    /// and bind it by value id or name through a caller-owned contract, such as the Runner-adjacent resident E4M3
    /// linear handoff.
    Activation,
    Pos,
    /// Card 152 / spec 267: one atomically-bound temporal/height/width position tuple for sectioned multimodal RoPE.
    /// Shape `[3,L]` for prefill, `[3]` for scalar decode, or `[3,B]` for statically batched decode, axis-major in
    /// the order temporal, height, width. One packed kind keeps a binder from updating only a subset of the axes;
    /// model tracers split it with Slice/Reshape before `rope_sectioned`. Per-prompt/per-step data, never a `Const`.
    MropePosition,
    SeqLen,
    /// Additive attention mask `[cap]`: 0 for valid key slots `t <= pos`, large-negative for `t > pos`. The
    /// constant-shape masked decode (G3d) uses it instead of slicing `[0..=pos]`.
    Mask,
    /// Paged-KV slot map `[cap]` (i32): logical key position `t` -> the flat physical slot in the block pool holding
    /// its K/V. The paged decode graph (spec 045) gathers the physical-order K/V cache by this to reconstruct
    /// logical order before masked attention; the physical write slot is derived in-graph as
    /// `gather_scalar(slot_map, pos)`. Per-step-varying, so a Slot.
    SlotMap,
    /// Card 188: the GDN recurrent-state pool's row map `[batch]` (i32): row `r`'s target GDN pool slot this step
    /// (the read/write-back index of the GDN pool's device mirror; see
    /// `poot_models::qwen3next::qwen3next_gdn_decode_batched_pool`). A distinct variant from [`Slot::SlotMap`]:
    /// reusing `Slot::SlotMap` at shape `[batch]` is safe for `poot-eval`'s per-id CPU binder but not for an engine
    /// binder that resolves a whole `Slot` kind from one caller-supplied buffer (e.g. `poot-llm`'s
    /// `bind_decode_batched`), which would stamp the same data onto both differently-shaped ids. A dedicated
    /// variant lets a binder give this map its own bind path; `Slot` is matched exhaustively at every binder site.
    GdnSlotMap,
    /// The current token's already-embedded input row `[hidden]`, host-computed per step (a single-row dequant, not
    /// a GPU gather). Lets a decode graph skip declaring the full `[vocab,hidden]` embedding table as a GPU-resident
    /// Const, needed when its dense-f32 size exceeds a backend's max single-buffer size (gemma4's `hidden=5376`
    /// gives ~5.6 GB, over wgpu's ~2 GB `max_buffer_size` here).
    TokenEmbed,
    /// Spec 248 (epic 129 B8): a batched LoRA decode graph's per-row adapter selection `[batch]` (f32,
    /// `IndexedMatMul`'s idx convention, see its doc comment on why idx is always f32): row `m`'s value is the
    /// `poot_load::lora::LoraAdapterPool` index (1-based; `LoraAdapterPool::NO_ADAPTER` = 0) whose stacked `A`/`B`
    /// that row's `lora_linear_batched` call uses this step. A distinct variant from `Slot::SlotMap`/
    /// `Slot::GdnSlotMap` for the reason given at `GdnSlotMap`: an engine binder resolves one whole `Slot` kind
    /// from one buffer per bind call. Per-step-varying (requests and adapters change as the batch admits/evicts),
    /// so a `Slot` rebound every step, never a `Const` (const-cache staleness).
    LoraIdx,
    /// Spec 266: an MoE layer's expert-pool assignment table `[E]` (f32, `IndexedMatMul`'s idx convention): entry
    /// `e` is the pool slot currently holding global expert `e`, or `-1` when
    /// not resident in that layer's pool this step. An in-graph `Gather` translates the router's global expert ids
    /// into pool-slot ids with this table, so the unmodified `IndexedMatMul` op addresses a smaller
    /// `[P, ...]` pool weight instead of the full `[E, ...]` stack.
    ///
    /// Unlike every other `Slot` kind, one graph contains many of these (one per MoE layer, each with different
    /// content in the same step). Slot binders resolve a whole kind from one buffer (see `GdnSlotMap`), which would
    /// stamp the same bytes onto every layer's table. So this kind must be declared through
    /// [`crate::builder::Builder::slot_named`], giving each occurrence a distinct [`ValueMeta::name`], and its
    /// binders must resolve per name (as `Const` does), never per kind. Per-step-varying (a refill moves experts
    /// between slots), so a `Slot`, never a `Const`: the const cache keys by name + element count and would serve a
    /// stale table.
    ExpertPoolMap,
    /// Card 551a (R-551a-4): the sampler suffix's per-step inputs, declared through
    /// [`crate::builder::Builder::slot_named`] with one of the three fixed tags `seed`, `params` or
    /// `top_k` (never the untagged [`crate::builder::Builder::slot`] form - a graph's sampler suffix
    /// needs up to three distinctly-shaped buffers of this one kind, so each occurrence must carry its
    /// own name, as [`Slot::ExpertPoolMap`] does for the same reason). `seed` is per-row I32 noise
    /// state for [`crate::ops::sampling::sample_head`]'s `RandomUniform`; `params` is the per-row
    /// `[inv_temp, floor_offset, noise_scale]` (+ `top_p` for `GumbelTopKTopP`) F32 table;
    /// `top_k` is the per-row I32 threshold for the two `TopK` rules. Per-step-varying (sampling
    /// parameters can change every request), so a `Slot`, never a `Const`.
    Sampler,
}

impl Slot {
    /// This variant's key fragment: the one spelling every [`Storage::Slot`] input name
    /// ([`SlotKey`]) is built from, hand-written per variant rather than taken from `Debug`
    /// (R466-013). `Slot`'s derived `Debug` happens to agree with this today, but nothing here
    /// depends on that: a renamed variant changes this match, and only this match, never a bound
    /// buffer's key silently drifting with whatever the derive emits.
    pub(crate) const fn key_label(&self) -> &'static str {
        match self {
            Slot::Token => "token",
            Slot::Activation => "activation",
            Slot::Pos => "pos",
            Slot::MropePosition => "mropeposition",
            Slot::SeqLen => "seqlen",
            Slot::Mask => "mask",
            Slot::SlotMap => "slotmap",
            Slot::GdnSlotMap => "gdnslotmap",
            Slot::TokenEmbed => "tokenembed",
            Slot::LoraIdx => "loraidx",
            Slot::ExpertPoolMap => "expertpoolmap",
            Slot::Sampler => "sampler",
        }
    }
}

/// How a value is stored. `Const` = a weight/table bound at graph close; `Computed` = a constant whose
/// contents the compiler computes from its own definition, so no checkpoint or binder supplies it;
/// `Slot` = a per-token-varying input routed through a stable device buffer; `Device` = an intermediate;
/// `State` = a persistent buffer carried across decode steps (the KV cache), fed in and overwritten by
/// its paired `state_out` (see [`Graph::state`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Storage {
    Device,
    Const,
    Computed(ComputedConst),
    Slot(Slot),
    State,
}

/// A constant the compiler computes, not a checkpoint binder. The `fold_iota` transform turns a nullary
/// [`crate::OpKind::Iota`] into one of these, so the planner never lowers an iota kernel and no name is
/// bound: a binder materializes the value from this payload exactly like a slot from its [`Slot`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ComputedConst {
    /// The F32 range `[0, 1, .., len)`, shape `[len]`.
    Iota { len: usize },
    /// One table of a RoPE pair, shape `[positions, spec.dim()]`, from
    /// [`crate::rope_table::rope_tables`]. `positions` is the entry's capacity, not the model's
    /// maximum context.
    Rope {
        table: RopeTable,
        positions: u32,
        spec: RopeSpec,
    },
    /// The ALiBi (Press et al.) per-head slopes of a model with `heads` attention heads, shape
    /// `[heads]`: `2^(-8 (i + 1) / n)` for `n` the largest power of two not above `heads`, then the
    /// odd-indexed slopes of the `2n` series for the remaining heads (BLOOM's and MPT's rule).
    AlibiSlopes { heads: usize },
}

fn alibi_series(n: usize) -> impl Iterator<Item = f32> {
    let start = 2f32.powf(-8.0 / n as f32);
    (0..n).map(move |i| start.powi(i as i32 + 1))
}

impl ComputedConst {
    /// The exact elements of the constant, in row-major order, in the F32 dtype the graph declares.
    ///
    /// `len` is bounded by the graph shape it materializes into, so a length above `i32::MAX` cannot be
    /// built (and the `as f32` casts below never lose an integer for a realizable value).
    pub fn values_f32(self) -> Vec<f32> {
        match self {
            ComputedConst::Iota { len } => (0..len).map(|i| i as f32).collect(),
            ComputedConst::Rope {
                table,
                positions,
                spec,
            } => {
                let tables = rope_table::rope_tables(
                    spec.dim(),
                    positions as usize,
                    spec.theta(),
                    &spec.flavor(),
                    None,
                );
                match table {
                    RopeTable::Cos => tables.cos,
                    RopeTable::Sin => tables.sin,
                }
            }
            ComputedConst::AlibiSlopes { heads } => {
                if heads == 0 {
                    return Vec::new();
                }
                let below = if heads.is_power_of_two() {
                    heads
                } else {
                    heads.next_power_of_two() / 2
                };
                let mut slopes: Vec<f32> = alibi_series(below).collect();
                slopes.extend(alibi_series(2 * below).step_by(2).take(heads - below));
                slopes
            }
        }
    }

    /// The shape of the value this constant materializes into.
    pub fn shape(self) -> Vec<usize> {
        match self {
            ComputedConst::Iota { len } => vec![len],
            ComputedConst::Rope {
                positions, spec, ..
            } => vec![positions as usize, spec.dim()],
            ComputedConst::AlibiSlopes { heads } => vec![heads],
        }
    }
}

/// The write pattern of one state pair's per-step overwrite (plan-562-565.md section 3).
///
/// Declared once, on the state input's [`ValueMeta`], at [`crate::builder::Builder::state_input`]. The
/// graph identity does not include it: it changes no kernel, buffer, or binding, so `poot-gpu`'s cache
/// key (hashed from `Graph::state` and the storage tag) is untouched by a role change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StateRole {
    /// Written at the live position along `axis`; a stale write past the committed length is
    /// harmless. The `[1, Hkv, cap, D]` architecture KV caches carry the position on axis 2.
    Positional { axis: u8 },
    /// Folds every token into its value: conv windows, GDN/KDA/SSM matrices, ring buffers written at
    /// `pos % w`, and the pooled `[pool_slots, hkv, d]` caches (position on axis 0). The safe default
    /// every existing state pair is traced with; `Positional` is declared only by `components::attention`
    /// and later family cards.
    Recurrent,
}

/// One `(state_in, state_out)` pair from [`Graph::state`], with the input's declared [`StateRole`] read
/// alongside it. Built by [`Graph::state_pairs`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatePair {
    pub input: ValueId,
    pub output: ValueId,
    pub role: StateRole,
}

#[derive(Clone, Debug)]
pub struct ValueMeta {
    pub aval: TensorType,
    pub storage: Storage,
    pub name: Option<String>,
    /// The structured identity of a [`Storage::Slot`] input, set once at construction
    /// ([`crate::builder::Builder::slot`], `slot_named`, `exact_i32_slot`) and never rebuilt from
    /// `name` (R466-013): `name` is the wire spelling existing binders read, `key` is the typed
    /// fact those binders will move to (Cards 550/564). `None` for every other storage (their
    /// identity is `name` itself, Keep-list section 1) and for a `Slot` value a caller assembled
    /// directly instead of through `Builder` (a malformed-graph test fixture).
    pub(crate) key: Option<SlotKey>,
    /// A [`Storage::State`] input's declared [`StateRole`], set once at construction
    /// ([`ValueMeta::new_state`], [`crate::builder::Builder::state_input`]) and read by
    /// [`Graph::state_pairs`]/[`Graph::state_role`]. `None` for every other storage;
    /// [`Graph::validate`] checks both directions (a `State` value with no role, and any other value
    /// with one).
    pub(crate) state_role: Option<StateRole>,
}

impl ValueMeta {
    /// A `ValueMeta` with no structured slot key and no state role: every storage but a
    /// `Builder`-constructed `Slot` or `State`. `key` and `state_role` are private fields so a
    /// literal cannot set them from outside this crate; test fixtures across the workspace that
    /// build a `Graph` by hand (bypassing `Builder`) call this instead of a struct literal.
    pub fn new(aval: TensorType, storage: Storage, name: Option<String>) -> Self {
        debug_assert!(
            storage != Storage::State,
            "ValueMeta::new: a Storage::State value must carry a StateRole - use ValueMeta::new_state"
        );
        ValueMeta {
            aval,
            storage,
            name,
            key: None,
            state_role: None,
        }
    }

    /// A `Storage::State` input's `ValueMeta`, carrying its declared `role`. The only constructor
    /// that can produce a `State` meta (R-644-1): [`Graph::validate`] rejects a `Storage::State`
    /// value built any other way (it would carry no role).
    pub fn new_state(aval: TensorType, name: Option<String>, role: StateRole) -> Self {
        ValueMeta {
            aval,
            storage: Storage::State,
            name,
            key: None,
            state_role: Some(role),
        }
    }

    /// The structured [`SlotKey`] of a [`Storage::Slot`] input built through [`crate::builder::Builder`]
    /// (Card 546a, Z9): the executor binder's primary key. `None` for every other storage and for a
    /// `Slot` value a caller assembled directly instead of through `Builder` (see the field doc).
    pub fn slot_key(&self) -> Option<&SlotKey> {
        self.key.as_ref()
    }
}

/// The structured identity of one [`Storage::Slot`] input: its typed [`Slot`] role, and - when
/// [`crate::builder::Builder::slot_named`] disambiguates a same-kind occurrence (spec 266's
/// per-layer expert pool table, spec 149's named activation) - the caller's opaque tag.
///
/// Built once, from typed data, at the point [`crate::builder::Builder`] declares the input
/// (R466-013): never re-derived from `Debug`-formatting the role, and never recovered later by
/// parsing the wire name ([`ValueMeta::name`]) back apart - that name is [`SlotKey`]'s
/// [`fmt::Display`] output for the binders that still read it, a one-way spelling, not a second
/// source of truth. Equality compares typed fields only (`Slot`'s own derived equality, and the tag
/// as an opaque string nothing here parses further).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SlotKey {
    role: Slot,
    tag: Option<String>,
}

impl SlotKey {
    /// A key for `role`, with `tag` set for a [`crate::builder::Builder::slot_named`] occurrence.
    pub fn new(role: Slot, tag: Option<&str>) -> Self {
        SlotKey {
            role,
            tag: tag.map(str::to_string),
        }
    }

    /// The slot role this key identifies (Card 546a, Z9): the executor binder's primary key, read
    /// without parsing [`SlotKey`]'s [`fmt::Display`] wire spelling back apart.
    pub fn role(&self) -> Slot {
        self.role
    }
}

impl fmt::Display for SlotKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.role.key_label())?;
        if let Some(tag) = &self.tag {
            write!(f, ".{tag}")?;
        }
        Ok(())
    }
}

/// The structured identity of one graph input binder: which role it fills, as typed data
/// (R466-013, R479-005). `Const`/`State` carry their caller-given name (Keep-list section 1:
/// typed constant names are kept); `Slot` carries the [`ValueMeta::key`] [`Builder`] stored at
/// construction. Two inputs compare equal here iff they fill the identical role - never by
/// `ValueId` (allocator order, coincidentally reused across independently built graphs) and never
/// by `Debug`-formatting or parsing the role back out of a string.
///
/// [`Builder`]: crate::builder::Builder
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum InputKey {
    Const(String),
    State(String),
    Slot(SlotKey),
}

/// One SSA instruction: `out = op(inputs)`. Multi-output ops are reserved (single output for now).
#[derive(Clone, Debug)]
pub struct Eqn {
    pub op: OpKind,
    pub inputs: Vec<Operand>,
    pub out: ValueId,
    /// The decoder layer that emitted this equation, set by [`crate::Builder::layer_scope`] for
    /// every equation emitted inside the scope. `None` outside a scope, and `None` for an equation
    /// a transform synthesized with a fresh `out` (the source equation's own `out` keeps its tag).
    pub layer: Option<LayerIndex>,
}

mod sealed {
    use super::ValidationOutput;

    /// Proof that a call comes from inside this crate. The type cannot be named or built elsewhere, so a generic
    /// caller that sees [`Sealed::outputs_mut`] through a [`super::ValidationChannel`] bound cannot call it.
    pub struct Internal;

    pub trait Sealed {
        /// Mutable declarations for canonical result remapping. The declaration order cannot change.
        fn outputs_mut(&mut self, _: Internal) -> &mut [ValidationOutput];

        /// Drop declarations the predicate rejects, keeping the survivors' order. In-crate transforms
        /// that split a graph use it to keep each piece's witnesses with the piece that defines them.
        /// Like [`Sealed::outputs_mut`], the token keeps it out of generic callers' reach, so a caller
        /// outside this crate still cannot empty a validation-bearing channel.
        fn retain_outputs(&mut self, _: Internal, keep: &mut dyn FnMut(&ValidationOutput) -> bool);
    }
}

/// The validation result channel of a [`Graph`].
///
/// The trait is sealed: [`NoValidations`] and [`ValidationOutputs`] are its only implementations. A graph's
/// channel is part of its type, so an executor typed on [`Graph`] (`Graph<NoValidations>`) cannot receive a
/// validation-bearing graph; there is no conversion from `Graph<ValidationOutputs>` to `Graph`.
///
/// Declarations are read-only outside this crate:
///
/// ```
/// use poot_graph_ir::{ValidationChannel, ValidationOutput};
///
/// fn declarations<V: ValidationChannel>(channel: &V) -> &[ValidationOutput] {
///     channel.outputs()
/// }
/// ```
///
/// Only [`Graph::remap_results`] retargets them, so a generic caller cannot:
///
/// ```compile_fail,E0061
/// use poot_graph_ir::ValidationChannel;
///
/// fn retarget<V: ValidationChannel>(channel: &mut V) {
///     channel.outputs_mut()[0].value = 0;
/// }
/// ```
///
/// and neither can a caller holding a concrete validation-bearing graph:
///
/// ```compile_fail,E0599
/// use poot_graph_ir::{Graph, ValidationChannel};
///
/// let mut graph = Graph::default().with_validations(Vec::new());
/// let _declarations = graph.validations.outputs_mut();
/// ```
pub trait ValidationChannel: sealed::Sealed + Clone + fmt::Debug {
    /// Validation declarations in declaration order.
    fn outputs(&self) -> &[ValidationOutput];
}

/// The channel of an ordinary graph: no validation outputs can be represented.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoValidations;

impl sealed::Sealed for NoValidations {
    fn outputs_mut(&mut self, _: sealed::Internal) -> &mut [ValidationOutput] {
        &mut []
    }

    fn retain_outputs(
        &mut self,
        _: sealed::Internal,
        _: &mut dyn FnMut(&ValidationOutput) -> bool,
    ) {
    }
}

impl ValidationChannel for NoValidations {
    fn outputs(&self) -> &[ValidationOutput] {
        &[]
    }
}

/// The channel of a validation-bearing graph: ordered, bounded F32 witnesses checked before result or state
/// publication.
///
/// The list is private and has no `Default`, so code outside this module cannot empty a validation-bearing
/// graph's declarations. It is created only by
/// [`Graph::with_validations`] and the validating builder finish methods. Only [`Graph::remap_results`] edits a
/// declaration in place, and it cannot add, remove, or reorder one.
///
/// Neither the list nor an empty replacement is reachable:
///
/// ```compile_fail,E0616
/// # use poot_graph_ir::Graph;
/// let mut graph = Graph::default().with_validations(Vec::new());
/// graph.validations.0.clear();
/// ```
///
/// ```compile_fail,E0277
/// # use poot_graph_ir::Graph;
/// let mut graph = Graph::default().with_validations(Vec::new());
/// let _declarations = std::mem::take(&mut graph.validations);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationOutputs(Vec<ValidationOutput>);

impl sealed::Sealed for ValidationOutputs {
    fn outputs_mut(&mut self, _: sealed::Internal) -> &mut [ValidationOutput] {
        &mut self.0
    }

    fn retain_outputs(
        &mut self,
        _: sealed::Internal,
        keep: &mut dyn FnMut(&ValidationOutput) -> bool,
    ) {
        self.0.retain(keep);
    }
}

impl ValidationChannel for ValidationOutputs {
    fn outputs(&self) -> &[ValidationOutput] {
        &self.0
    }
}

/// A flat SSA primitive tensor-op graph.
///
/// `V` is the validation channel. `Graph` alone means `Graph<NoValidations>`, the type every device and
/// specialized executor accepts. A validation-bearing `Graph<ValidationOutputs>` does not coerce to it:
///
/// ```compile_fail,E0308
/// use poot_graph_ir::{Builder, Graph, TensorType};
///
/// fn unsupported_executor(_: &Graph) {}
///
/// let b = Builder::new();
/// let output = b.constant("output", TensorType::f32([1]));
/// let graph = b.finish_with_state(output, &[]).with_validations(vec![]);
/// unsupported_executor(&graph);
/// ```
///
/// and has no implicit conversion:
///
/// ```compile_fail,E0277
/// use poot_graph_ir::{Builder, Graph, TensorType};
///
/// let b = Builder::new();
/// let output = b.constant("output", TensorType::f32([1]));
/// let graph = b.finish_with_state(output, &[]).with_validations(vec![]);
/// let _ordinary: Graph = graph.into();
/// ```
#[derive(Clone, Debug)]
pub struct Graph<V: ValidationChannel = NoValidations> {
    pub values: Vec<ValueMeta>,
    /// All input binders (consts + slots), in declaration order.
    pub inputs: Vec<ValueId>,
    /// The subset of `inputs` that are constants (weights/tables), for binding at close.
    pub consts: Vec<ValueId>,
    /// The per-token-varying inputs and their slot kind.
    pub slots: Vec<(ValueId, Slot)>,
    pub eqns: Vec<Eqn>,
    /// The primary output (the next-token logits).
    pub output: ValueId,
    /// The validation channel: none for `Graph`, ordered bounded F32 witnesses for
    /// `Graph<ValidationOutputs>`.
    pub validations: V,
    /// Input-output alias pairs `(state_in, state_out)` carried across decode steps: `state_in` is a
    /// `Storage::State` input binder and `state_out` a produced (or pass-through) value of identical type. The
    /// executor feeds each cache buffer in as `state_in` and overwrites it with `state_out` (the jax
    /// `donate_argnums` / XLA `input_output_aliases` contract). Empty for stateless graphs.
    pub state: Vec<(ValueId, ValueId)>,
}

impl Default for Graph {
    fn default() -> Self {
        Graph {
            values: Vec::new(),
            inputs: Vec::new(),
            consts: Vec::new(),
            slots: Vec::new(),
            eqns: Vec::new(),
            output: 0,
            validations: NoValidations,
            state: Vec::new(),
        }
    }
}

impl Graph {
    /// Attach validation declarations. The result is not validated; call [`Graph::validate`].
    pub fn with_validations(self, validations: Vec<ValidationOutput>) -> Graph<ValidationOutputs> {
        let Graph {
            values,
            inputs,
            consts,
            slots,
            eqns,
            output,
            validations: NoValidations,
            state,
        } = self;
        Graph {
            values,
            inputs,
            consts,
            slots,
            eqns,
            output,
            validations: ValidationOutputs(validations),
            state,
        }
    }
}

impl<V: ValidationChannel> Graph<V> {
    /// Validation declarations in declaration order; always empty for `Graph`.
    pub fn validation_outputs(&self) -> &[ValidationOutput] {
        self.validations.outputs()
    }

    pub fn aval(&self, id: ValueId) -> &TensorType {
        &self.values[id].aval
    }
    pub fn meta(&self, id: ValueId) -> &ValueMeta {
        &self.values[id]
    }

    /// `id`'s structured [`InputKey`], or `None` when `id` is not an input binder
    /// ([`Storage::Device`]), an input binder with no declared name (`Const`/`State`), or a `Slot`
    /// value a caller assembled directly instead of through [`crate::builder::Builder`] (so
    /// [`ValueMeta::key`] was never set - a malformed or synthetic graph, never a real trace).
    /// Reads [`ValueMeta::key`] as `Builder` stored it; never re-derives it by parsing
    /// [`ValueMeta::name`] (R466-013).
    pub(crate) fn input_key(&self, id: ValueId) -> Option<InputKey> {
        let meta = self.meta(id);
        match meta.storage {
            Storage::Const => meta.name.clone().map(InputKey::Const),
            Storage::Computed(_) => None,
            Storage::State => meta.name.clone().map(InputKey::State),
            Storage::Slot(_) => meta.key.clone().map(InputKey::Slot),
            Storage::Device => None,
        }
    }

    /// `input`'s declared [`StateRole`], or `None` when `input` is not a `Storage::State` value.
    pub fn state_role(&self, input: ValueId) -> Option<StateRole> {
        self.values.get(input).and_then(|meta| meta.state_role)
    }

    /// [`Graph::state`]'s pairs, each with its input's declared [`StateRole`] read from
    /// [`ValueMeta::state_role`]. A validated graph gives every pair a role (`Graph::validate`
    /// rejects a `Storage::State` input with none); an unvalidated or hand-built graph's missing
    /// role reads as [`StateRole::Recurrent`], the safe default every state pair traced before
    /// `components::attention` carries.
    pub fn state_pairs(&self) -> impl Iterator<Item = StatePair> + '_ {
        self.state.iter().map(move |&(input, output)| StatePair {
            input,
            output,
            role: self.state_role(input).unwrap_or(StateRole::Recurrent),
        })
    }

    pub fn observable_values(&self) -> impl Iterator<Item = ValueId> + '_ {
        std::iter::once(self.output)
            .chain(self.validation_outputs().iter().map(|output| output.value))
    }

    pub fn liveness_roots(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.observable_values()
            .chain(self.state.iter().map(|&(_, output)| output))
    }

    pub fn pinned_values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.liveness_roots()
            .chain(self.state.iter().map(|&(input, _)| input))
    }

    pub fn remap_results(&mut self, mut remap: impl FnMut(ValueId) -> ValueId) {
        self.output = remap(self.output);
        for validation in sealed::Sealed::outputs_mut(&mut self.validations, sealed::Internal) {
            validation.value = remap(validation.value);
        }
        for (_, output) in &mut self.state {
            *output = remap(*output);
        }
    }

    /// Drop the declarations the predicate rejects, keeping the survivors in declaration order.
    ///
    /// A split graph keeps exactly the witnesses it still defines (`poot-graph-plan`'s `split_stages`,
    /// card 626, is the one caller outside this crate). The real seal is `sealed::Internal`: it
    /// cannot be named or built outside this crate, so no fully generic
    /// `ValidationChannel`-bound caller can reach `Sealed::retain_outputs` directly and shed a
    /// channel's declarations - only this concrete, crate-owned method can.
    pub fn retain_validations(&mut self, mut keep: impl FnMut(&ValidationOutput) -> bool) {
        sealed::Sealed::retain_outputs(&mut self.validations, sealed::Internal, &mut keep);
    }

    /// Validate basic SSA and binding-table well-formedness. Cheap structural check.
    pub fn validate(&self) -> Result<(), GraphValidationError> {
        let n = self.values.len();
        if self.output >= n {
            return Err(GraphValidationError::OutputNotDefined { value: self.output });
        }
        let validations = self.validation_outputs();
        let mut validation_ids = HashSet::with_capacity(validations.len());
        let mut validation_names = HashSet::with_capacity(validations.len());
        for validation in validations {
            if validation.value >= n {
                return Err(GraphValidationError::ValidationValueOutOfRange {
                    value: validation.value,
                    value_count: n,
                });
            }
            if !validation_ids.insert(validation.id) {
                return Err(GraphValidationError::DuplicateValidationId { id: validation.id });
            }
            if validation.name.is_empty() {
                return Err(GraphValidationError::EmptyValidationName { id: validation.id });
            }
            if !validation_names.insert(validation.name.as_str()) {
                return Err(GraphValidationError::DuplicateValidationName {
                    name: validation.name.clone(),
                });
            }
        }
        // The packet layout checks each validation value's F32 dtype, static shape, and payload bound.
        ValidationPacketLayout::for_outputs(&self.values, validations)?;
        for &(state_input, state_output) in &self.state {
            if state_input >= n || state_output >= n {
                return Err(GraphValidationError::StatePairOutOfRange {
                    state_input,
                    state_output,
                    value_count: n,
                });
            }
        }
        // Input binders are defined up front. Binding tables are an executor-facing contract, so validate them before
        // equation inspection can follow malformed metadata.
        let mut input_bindings = vec![false; n];
        for &id in &self.inputs {
            if id >= n {
                return Err(GraphValidationError::InputOutOfRange {
                    value: id,
                    value_count: n,
                });
            }
            if input_bindings[id] {
                return Err(GraphValidationError::DuplicateInput { value: id });
            }
            if self.values[id].storage == Storage::Device {
                return Err(GraphValidationError::DeviceInput { value: id });
            }
            input_bindings[id] = true;
        }

        let mut const_bindings = vec![false; n];
        for &id in &self.consts {
            if id >= n {
                return Err(GraphValidationError::ConstOutOfRange {
                    value: id,
                    value_count: n,
                });
            }
            if const_bindings[id] {
                return Err(GraphValidationError::DuplicateConst { value: id });
            }
            if !input_bindings[id] {
                return Err(GraphValidationError::ConstNotInput { value: id });
            }
            if !matches!(
                self.values[id].storage,
                Storage::Const | Storage::Computed(_) | Storage::State
            ) {
                return Err(GraphValidationError::ConstStorageMismatch {
                    value: id,
                    storage: self.values[id].storage,
                });
            }
            const_bindings[id] = true;
        }

        let mut slot_bindings = vec![false; n];
        for &(id, slot) in &self.slots {
            if id >= n {
                return Err(GraphValidationError::SlotOutOfRange {
                    value: id,
                    value_count: n,
                });
            }
            if slot_bindings[id] {
                return Err(GraphValidationError::DuplicateSlot { value: id });
            }
            if !input_bindings[id] {
                return Err(GraphValidationError::SlotNotInput { value: id });
            }
            if self.values[id].storage != Storage::Slot(slot) {
                return Err(GraphValidationError::SlotStorageMismatch {
                    value: id,
                    table_slot: slot,
                    storage: self.values[id].storage,
                });
            }
            slot_bindings[id] = true;
        }

        for &id in &self.inputs {
            match self.values[id].storage {
                Storage::Const | Storage::Computed(_) | Storage::State if !const_bindings[id] => {
                    return Err(GraphValidationError::MissingConstBinding {
                        value: id,
                        storage: self.values[id].storage,
                    });
                }
                Storage::Slot(slot) if !slot_bindings[id] => {
                    return Err(GraphValidationError::MissingSlotBinding { value: id, slot });
                }
                Storage::Device => unreachable!("Device inputs reject above"),
                Storage::Const | Storage::Computed(_) | Storage::State | Storage::Slot(_) => {}
            }
        }

        // Every `Storage::State` value carries a `StateRole`, and no other value does (R-644-1): checked over
        // every value, not just the pairs in `self.state`, so an orphaned state binder (rejected below by
        // `StateInputNotInput`) still gets a role check first, in id order.
        for (id, meta) in self.values.iter().enumerate() {
            match (meta.storage, meta.state_role) {
                (Storage::State, None) => {
                    return Err(GraphValidationError::MissingStateRole { value: id });
                }
                (Storage::State, Some(StateRole::Positional { axis })) => {
                    let rank = meta.aval.rank();
                    if usize::from(axis) >= rank {
                        return Err(GraphValidationError::StateRoleAxis {
                            value: id,
                            axis,
                            rank,
                        });
                    }
                }
                (Storage::State, Some(StateRole::Recurrent)) => {}
                (_, Some(_)) => {
                    return Err(GraphValidationError::UnexpectedStateRole {
                        value: id,
                        storage: meta.storage,
                    });
                }
                (_, None) => {}
            }
        }

        // State metadata is an executor-facing table like inputs, consts, and slots. Validate every property
        // independent of equation production before walking equations, so malformed state declarations take
        // deterministic precedence over unrelated equation defects.
        let mut state_destinations = vec![false; n];
        for &(state_input, state_output) in &self.state {
            if state_destinations[state_input] {
                return Err(GraphValidationError::DuplicateStateDestination { value: state_input });
            }
            state_destinations[state_input] = true;
            if self.values[state_input].storage != Storage::State {
                return Err(GraphValidationError::StateInputStorageMismatch {
                    value: state_input,
                    storage: self.values[state_input].storage,
                });
            }
            if !input_bindings[state_input] {
                return Err(GraphValidationError::StateInputNotInput { value: state_input });
            }
            if self.values[state_input].aval != self.values[state_output].aval {
                return Err(GraphValidationError::StateTypeMismatch {
                    state_input,
                    state_output,
                    input_type: self.values[state_input].aval.clone(),
                    output_type: self.values[state_output].aval.clone(),
                });
            }
        }

        // Result definedness is table metadata: a root needs an input binder or a producing equation. Check it before
        // semantic equation inspection so a missing producer takes precedence over unrelated equation defects. The
        // producer table is indexed by equation outputs, so their ids are range-checked first; a validator must return
        // a clean error on a malformed `eqn.out` (e.g. from a buggy transform pass), never panic. The walk below
        // still proves SSA order.
        let mut bound_or_produced = input_bindings.clone();
        for (i, eqn) in self.eqns.iter().enumerate() {
            if eqn.out >= n {
                return Err(GraphValidationError::EquationOutputOutOfRange {
                    equation: i,
                    operation: eqn.op.name(),
                    value: eqn.out,
                    value_count: n,
                });
            }
            bound_or_produced[eqn.out] = true;
        }
        if !bound_or_produced[self.output] {
            return Err(GraphValidationError::OutputNotDefined { value: self.output });
        }
        for validation in validations {
            if !bound_or_produced[validation.value] {
                return Err(GraphValidationError::ValidationOutputNotDefined {
                    id: validation.id,
                    value: validation.value,
                });
            }
        }
        for &(_, state_output) in &self.state {
            if !bound_or_produced[state_output] {
                return Err(GraphValidationError::StateOutputNotDefined {
                    value: state_output,
                });
            }
        }

        let mut defined = input_bindings;
        for (i, eqn) in self.eqns.iter().enumerate() {
            let mut in_types = Vec::with_capacity(eqn.inputs.len());
            for op in &eqn.inputs {
                match op {
                    Operand::Value(v) => {
                        if *v >= n || !defined[*v] {
                            return Err(GraphValidationError::EquationOperandNotDefined {
                                equation: i,
                                value: *v,
                            });
                        }
                        in_types.push(self.values[*v].aval.clone());
                    }
                    Operand::Lit(s) => in_types.push(s.ty()),
                }
            }
            if let [Operand::Value(a), Operand::Value(b)] = eqn.inputs.as_slice()
                && matches!(eqn.op, crate::op::OpKind::Binary(_))
                && crate::op::check_binary_value_dtypes(&in_types[0], &in_types[1]).is_err()
            {
                return Err(GraphValidationError::BinaryValueDtypeMismatch {
                    equation: i,
                    operation: eqn.op.name(),
                    left: self.values[*a].aval.dtype,
                    right: self.values[*b].aval.dtype,
                });
            }
            if defined[eqn.out] {
                return Err(GraphValidationError::EquationOutputRedefined {
                    equation: i,
                    operation: eqn.op.name(),
                    value: eqn.out,
                });
            }
            let inferred = eqn.op.infer(&in_types).map_err(|source| {
                GraphValidationError::EquationInference {
                    equation: i,
                    operation: eqn.op.name(),
                    source,
                }
            })?;
            let stored = &self.values[eqn.out].aval;
            if inferred != *stored && !is_matmul_f32_accumulate_widening(&eqn.op, &inferred, stored)
            {
                return Err(GraphValidationError::EquationOutputTypeMismatch {
                    equation: i,
                    operation: eqn.op.name(),
                    stored: stored.clone(),
                    inferred,
                });
            }
            defined[eqn.out] = true;
        }
        Ok(())
    }
}

impl ValidationPacketLayout {
    /// Checked packet layout of a graph's validation declarations. An ordinary `Graph` has the empty layout.
    pub fn for_graph<V: ValidationChannel>(graph: &Graph<V>) -> Result<Self, GraphValidationError> {
        Self::for_outputs(&graph.values, graph.validation_outputs())
    }

    fn for_outputs(
        values: &[ValueMeta],
        validations: &[ValidationOutput],
    ) -> Result<Self, GraphValidationError> {
        let mut entries = Vec::with_capacity(validations.len());
        let mut lane_count = 0usize;
        for validation in validations {
            let aval = values.get(validation.value).ok_or(
                GraphValidationError::ValidationValueOutOfRange {
                    value: validation.value,
                    value_count: values.len(),
                },
            )?;
            if aval.aval.dtype != DType::F32 {
                return Err(GraphValidationError::ValidationDtype {
                    id: validation.id,
                    value: validation.value,
                    actual: aval.aval.dtype,
                });
            }
            let lanes = aval
                .aval
                .shape
                .iter()
                .try_fold(1usize, |lanes, &extent| lanes.checked_mul(extent))
                .ok_or_else(|| GraphValidationError::ValidationShapeOverflow {
                    id: validation.id,
                    value: validation.value,
                    shape: aval.aval.shape.clone(),
                })?;
            if lanes == 0 {
                return Err(GraphValidationError::EmptyValidationValue {
                    id: validation.id,
                    value: validation.value,
                });
            }
            let first_lane = lane_count;
            lane_count = lane_count
                .checked_add(lanes)
                .ok_or(GraphValidationError::ValidationPacketLaneOverflow)?;
            entries.push(ValidationPacketEntry {
                id: validation.id,
                name: validation.name.clone(),
                first_lane,
                lane_count: lanes,
            });
        }
        let byte_len = lane_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(GraphValidationError::ValidationPacketByteOverflow)?;
        if byte_len > MAX_VALIDATION_PACKET_BYTES {
            return Err(GraphValidationError::ValidationPacketTooLarge {
                byte_len,
                max_bytes: MAX_VALIDATION_PACKET_BYTES,
            });
        }
        Ok(Self {
            entries,
            lane_count,
            byte_len,
        })
    }

    pub fn validate_f32_bits(&self, bits: &[u32]) -> Result<(), ValidationPacketError> {
        if bits.len() != self.lane_count {
            return Err(ValidationPacketError::Length {
                expected: self.lane_count,
                actual: bits.len(),
            });
        }
        for entry in &self.entries {
            for lane in 0..entry.lane_count {
                let observed_bits = bits[entry.first_lane + lane];
                if observed_bits & 0x7fff_ffff != 0 {
                    return Err(ExecutionValidationFailure {
                        id: entry.id,
                        name: entry.name.clone(),
                        lane,
                        observed_bits,
                    }
                    .into());
                }
            }
        }
        Ok(())
    }
}

/// Whether a `MatMul`/`MatMulBias` eqn's stored output aval is a legitimate F32-accumulate widening of its
/// `infer`red (narrower-operand) type, rather than a type error.
///
/// `OpKind::MatMul::infer` reports the dtype of operand A, so a plain matmul type-checks normally. But backends
/// dispatch mixed-precision matmuls where both operands are narrowed (F16 or BF16) while the eqn's output aval
/// stays F32: tensor-core/cooperative-matrix units accumulate in f32 regardless of operand width (see
/// `DType::BF16`/`DType::F16`). Producers: a mixed-bf16 matmul cast insertion (AMD WMMA, spec 130 FR-004:
/// both operands cast to bf16, output f32) and the SPIR-V cooperative-matrix arm's test fixtures (card 154: F16
/// operands, F32 output; RADV coopmat config 13 has no f16-accumulate mode). Rejecting this shape would make
/// `run_resident` unusable for either backend's mixed-precision matmul.
///
/// Only MatMul/MatMulBias get this exception: every other op's `infer` must match the stored aval exactly.
fn is_matmul_f32_accumulate_widening(
    op: &OpKind,
    inferred: &TensorType,
    stored: &TensorType,
) -> bool {
    matches!(op, OpKind::MatMul | OpKind::MatMulBias)
        && inferred.shape == stored.shape
        && stored.dtype == DType::F32
        && matches!(inferred.dtype, DType::F16 | DType::BF16)
}

impl<V: ValidationChannel> fmt::Display for Graph<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "graph {{")?;
        writeln!(f, "  // slots (per-token-varying inputs):")?;
        for (id, slot) in &self.slots {
            // The structured key (role + tag), not `{slot:?}` alone: two `slot_named` occurrences of
            // one kind (spec 266's per-layer expert pool table) would otherwise print identically.
            let key = match self.input_key(*id) {
                Some(InputKey::Slot(key)) => key,
                _ => SlotKey::new(*slot, None),
            };
            writeln!(f, "    v{id} : {} = slot {key}", self.values[*id].aval)?;
        }
        writeln!(
            f,
            "  // consts: {} weight/table binders (elided)",
            self.consts.len()
        )?;
        writeln!(f, "  // body ({} eqns):", self.eqns.len())?;
        for eqn in &self.eqns {
            let outm = &self.values[eqn.out];
            write!(f, "    v{} : {} = {} (", eqn.out, outm.aval, eqn.op.name())?;
            for (i, inp) in eqn.inputs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match inp {
                    Operand::Value(id) => write!(f, "v{id}")?,
                    Operand::Lit(s) => write!(f, "{s}")?,
                }
            }
            match &outm.name {
                Some(name) => writeln!(f, ")   // {name}")?,
                None => writeln!(f, ")")?,
            }
        }
        writeln!(f, "  output: v{} : {}", self.output, self.aval(self.output))?;
        let validations = self.validation_outputs();
        if !validations.is_empty() {
            writeln!(f, "  // validations (in declaration order):")?;
            for validation in validations {
                writeln!(
                    f,
                    "  validation {:?} {:?}: v{} : {}",
                    validation.id,
                    validation.name,
                    validation.value,
                    self.aval(validation.value)
                )?;
            }
        }
        if !self.state.is_empty() {
            writeln!(f, "  // state (carried cache, in -> out):")?;
            for pair in self.state_pairs() {
                writeln!(
                    f,
                    "    v{} : {} <- v{}  [{:?}]",
                    pair.input, self.values[pair.input].aval, pair.output, pair.role
                )?;
            }
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::Builder;
    use crate::op::{BinOp, OpKind};
    use crate::types::{Scalar, TensorType};
    use crate::{
        GraphValidationError, Operand, Slot, StateRole, Storage, ValidationId, ValidationOutput,
        ValidationPacketError, ValidationPacketLayout, ValueId, ValueMeta,
    };
    use poot_tensor::DType;

    /// The ALiBi slopes of a power-of-two head count are `2^(-8 (i + 1) / n)`; a head count between
    /// powers takes the closest lower power's series, then every other slope of the doubled series
    /// (BLOOM's published rule, checked against hand-computed values).
    #[test]
    fn alibi_slopes_follow_the_published_series() {
        let slopes = |heads| super::ComputedConst::AlibiSlopes { heads }.values_f32();
        let close = |got: Vec<f32>, want: Vec<f64>| {
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert!((f64::from(*g) - w).abs() < 1e-7, "{got:?} vs {want:?}");
            }
        };
        close(slopes(4), (1..=4).map(|i| 2f64.powi(-2 * i)).collect());
        close(slopes(8), (1..=8).map(|i| 2f64.powi(-i)).collect());
        let mut twelve: Vec<f64> = (1..=8).map(|i| 2f64.powi(-i)).collect();
        twelve.extend([0.5, 1.5, 2.5, 3.5].map(|e| 2f64.powf(-e)));
        close(slopes(12), twelve);
        assert_eq!(
            super::ComputedConst::AlibiSlopes { heads: 12 }.shape(),
            vec![12]
        );
    }

    #[derive(Clone, Copy)]
    struct BindingFixtureIds {
        weight: ValueId,
        first_pool: ValueId,
        first_state: ValueId,
        output: ValueId,
    }

    fn binding_fixture() -> (super::Graph, BindingFixtureIds) {
        let b = Builder::new();
        let weight = b.constant("weight", TensorType::f32(vec![1]));
        let first_pool = b.slot_named(Slot::ExpertPoolMap, "layer_0", TensorType::f32(vec![1]));
        let _second_pool = b.slot_named(Slot::ExpertPoolMap, "layer_1", TensorType::f32(vec![1]));
        let first_state = b.state_input(
            "first_state",
            TensorType::f32(vec![1]),
            StateRole::Recurrent,
        );
        let second_state = b.state_input(
            "second_state",
            TensorType::f32(vec![1]),
            StateRole::Recurrent,
        );
        let output = b.binary(BinOp::Add, first_state, second_state);
        let graph = b.finish_with_state(
            output,
            &[(first_state, second_state), (second_state, first_state)],
        );
        (
            graph,
            BindingFixtureIds {
                weight: weight.id,
                first_pool: first_pool.id,
                first_state: first_state.id,
                output: output.id,
            },
        )
    }

    #[test]
    fn validate_accepts_a_well_formed_graph() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4]));
        let out = b.binary(BinOp::Mul, x, x);
        let g = b.finish(out);
        g.validate().expect("a well-formed graph must validate");
    }

    #[test]
    fn validation_layout_is_ordered_and_bounded() {
        let b = Builder::new();
        let primary = b.constant("primary", TensorType::f32([1]));
        let pair = b.constant("pair", TensorType::f32([2]));
        let scalar = b.constant("scalar", TensorType::f32([]));
        let graph = crate::test_support::finish_with_validations(
            b,
            primary,
            &[
                (ValidationId(7), "pair-check", pair),
                (ValidationId(9), "scalar-check", scalar),
            ],
        )
        .unwrap();

        let layout = ValidationPacketLayout::for_graph(&graph).unwrap();
        assert_eq!(layout.lane_count, 3);
        assert_eq!(layout.byte_len, 12);
        assert_eq!(layout.entries[0].id, ValidationId(7));
        assert_eq!(layout.entries[0].name, "pair-check");
        assert_eq!(layout.entries[0].first_lane, 0);
        assert_eq!(layout.entries[0].lane_count, 2);
        assert_eq!(layout.entries[1].id, ValidationId(9));
        assert_eq!(layout.entries[1].name, "scalar-check");
        assert_eq!(layout.entries[1].first_lane, 2);
        assert_eq!(layout.entries[1].lane_count, 1);
        assert_eq!(
            layout.validate_f32_bits(&[0, 0]),
            Err(ValidationPacketError::Length {
                expected: 3,
                actual: 2,
            })
        );
        assert!(matches!(
            layout.validate_f32_bits(&[1.0f32.to_bits(), 0, 1.0f32.to_bits()]),
            Err(ValidationPacketError::Failure(
                super::ExecutionValidationFailure {
                    id: ValidationId(7),
                    lane: 0,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn validation_bits_accept_only_signed_zero() {
        let b = Builder::new();
        let primary = b.constant("primary", TensorType::f32([1]));
        let witness = b.constant("witness", TensorType::f32([2]));
        let graph = crate::test_support::finish_with_validations(
            b,
            primary,
            &[(ValidationId(7), "check", witness)],
        )
        .unwrap();
        let layout = ValidationPacketLayout::for_graph(&graph).unwrap();

        layout
            .validate_f32_bits(&[0.0f32.to_bits(), (-0.0f32).to_bits()])
            .unwrap();
        for bits in [
            1.0f32.to_bits(),
            f32::INFINITY.to_bits(),
            f32::NAN.to_bits(),
        ] {
            assert_eq!(
                layout.validate_f32_bits(&[0, bits]),
                Err(ValidationPacketError::Failure(
                    super::ExecutionValidationFailure {
                        id: ValidationId(7),
                        name: "check".into(),
                        lane: 1,
                        observed_bits: bits,
                    }
                ))
            );
        }
    }

    #[test]
    fn validation_declarations_and_packet_bound_are_checked() {
        let b = Builder::new();
        let primary = b.constant("primary", TensorType::f32([1]));
        let mut graph = b.finish(primary).with_validations(vec![ValidationOutput {
            id: ValidationId(1),
            name: "range".into(),
            value: usize::MAX,
        }]);
        assert!(matches!(
            graph.validate(),
            Err(GraphValidationError::ValidationValueOutOfRange {
                value: usize::MAX,
                ..
            })
        ));

        graph.validations.0[0].value = graph.values.len();
        graph.values.push(ValueMeta {
            aval: TensorType::f32([1]),
            storage: Storage::Device,
            name: None,
            key: None,
            state_role: None,
        });
        assert!(matches!(
            graph.validate(),
            Err(GraphValidationError::ValidationOutputNotDefined {
                id: ValidationId(1),
                ..
            })
        ));

        let b = Builder::new();
        let primary = b.constant("primary", TensorType::f32([1]));
        let witness = b.constant("witness", TensorType::f32([1]));
        let cases = [
            (ValidationId(1), "", witness.id),
            (ValidationId(1), "same", witness.id),
        ];
        let mut graph = b.finish(primary).with_validations(
            cases
                .into_iter()
                .map(|(id, name, value)| ValidationOutput {
                    id,
                    name: name.into(),
                    value,
                })
                .collect(),
        );
        assert!(matches!(
            graph.validate(),
            Err(GraphValidationError::EmptyValidationName {
                id: ValidationId(1)
            })
        ));

        graph.validations.0[0].name = "same".into();
        assert!(matches!(
            graph.validate(),
            Err(GraphValidationError::DuplicateValidationId {
                id: ValidationId(1)
            })
        ));
        graph.validations.0[1].id = ValidationId(2);
        assert!(matches!(
            graph.validate(),
            Err(GraphValidationError::DuplicateValidationName { ref name }) if name == "same"
        ));

        let b = Builder::new();
        let primary = b.constant("primary", TensorType::f32([1]));
        let witness = b.constant("invalid", TensorType::new([1], DType::I32));
        assert!(matches!(
            crate::test_support::finish_with_validations(b, primary, &[(ValidationId(1), "invalid", witness)]),
            Err(GraphValidationError::ValidationDtype {
                id: ValidationId(1),
                value,
                actual: DType::I32,
            }) if value == witness.id
        ));

        let b = Builder::new();
        let primary = b.constant("primary", TensorType::f32([1]));
        let first = b.constant("first", TensorType::f32([usize::MAX]));
        let second = b.constant("second", TensorType::f32([1]));
        assert!(matches!(
            crate::test_support::finish_with_validations(
                b,
                primary,
                &[
                    (ValidationId(1), "first", first),
                    (ValidationId(2), "second", second),
                ],
            ),
            Err(GraphValidationError::ValidationPacketLaneOverflow)
        ));

        let b = Builder::new();
        let primary = b.constant("primary", TensorType::f32([1]));
        let witness = b.constant(
            "byte-overflow",
            TensorType::f32([usize::MAX / std::mem::size_of::<f32>() + 1]),
        );
        assert!(matches!(
            crate::test_support::finish_with_validations(
                b,
                primary,
                &[(ValidationId(1), "byte-overflow", witness)]
            ),
            Err(GraphValidationError::ValidationPacketByteOverflow)
        ));

        let b = Builder::new();
        let primary = b.constant("primary", TensorType::f32([1]));
        let witness = b.constant("invalid", TensorType::f32([0]));
        assert!(matches!(
            crate::test_support::finish_with_validations(b, primary, &[(ValidationId(1), "invalid", witness)]),
            Err(GraphValidationError::EmptyValidationValue {
                id: ValidationId(1),
                value,
            }) if value == witness.id
        ));

        let b = Builder::new();
        let primary = b.constant("primary", TensorType::f32([1]));
        let witness = b.constant("invalid", TensorType::f32([usize::MAX, 2]));
        assert!(matches!(
            crate::test_support::finish_with_validations(b, primary, &[(ValidationId(1), "invalid", witness)]),
            Err(GraphValidationError::ValidationShapeOverflow {
                id: ValidationId(1),
                value,
                ..
            }) if value == witness.id
        ));

        let b = Builder::new();
        let primary = b.constant("primary", TensorType::f32([1]));
        // Spec 375 fixes the payload at 4,096 bytes: 1,024 f32 lanes pass and 1,025 do not. Literal lane
        // counts keep this test sensitive to a change of the bound itself.
        let witness = b.constant("bounded", TensorType::f32([1024]));
        crate::test_support::finish_with_validations(
            b,
            primary,
            &[(ValidationId(1), "bounded", witness)],
        )
        .unwrap();

        let b = Builder::new();
        let primary = b.constant("primary", TensorType::f32([1]));
        let witness = b.constant("too-large", TensorType::f32([1025]));
        assert!(matches!(
            crate::test_support::finish_with_validations(
                b,
                primary,
                &[(ValidationId(1), "too-large", witness)]
            ),
            Err(GraphValidationError::ValidationPacketTooLarge {
                byte_len: 4100,
                max_bytes: 4096,
            })
        ));
    }

    #[test]
    fn validate_accepts_named_same_kind_slots_and_valid_state_topologies() {
        let (graph, _) = binding_fixture();
        assert_eq!(
            graph
                .slots
                .iter()
                .filter(|(_, slot)| *slot == Slot::ExpertPoolMap)
                .count(),
            2,
            "fixture must exercise two separately named slots of one kind"
        );
        assert_ne!(
            graph.meta(graph.slots[0].0).name,
            graph.meta(graph.slots[1].0).name,
            "the same-kind slots must remain distinguishable by name"
        );
        graph
            .validate()
            .expect("same-kind named slots and a state swap are valid");

        for (label, state) in [
            ("identity", [(0, 0), (1, 1)]),
            ("shared source", [(0, 0), (1, 0)]),
        ] {
            let b = Builder::new();
            let first = b.state_input("first", TensorType::f32(vec![1]), StateRole::Recurrent);
            let second = b.state_input("second", TensorType::f32(vec![1]), StateRole::Recurrent);
            let output = b.binary(BinOp::Add, first, second);
            let values = [first, second];
            let state = state.map(|(destination, source)| (values[destination], values[source]));
            b.finish_with_state(output, &state)
                .validate()
                .unwrap_or_else(|error| panic!("{label} state topology must validate: {error}"));
        }
    }

    #[test]
    fn validate_rejects_inconsistent_binding_and_state_tables() {
        #[derive(Clone, Copy, Debug)]
        enum Corruption {
            ConstIdOutOfRange,
            SlotIdOutOfRange,
            DuplicateInput,
            DuplicateConst,
            DuplicateSlot,
            ConstMissingFromInputs,
            SlotMissingFromInputs,
            ConstStorageMismatch,
            SlotStorageMismatch,
            SlotKindMismatch,
            MissingConstBinding,
            MissingStateBinding,
            MissingSlotBinding,
            DeviceInput,
            ConflictingStateDestination,
        }

        let cases = [
            Corruption::ConstIdOutOfRange,
            Corruption::SlotIdOutOfRange,
            Corruption::DuplicateInput,
            Corruption::DuplicateConst,
            Corruption::DuplicateSlot,
            Corruption::ConstMissingFromInputs,
            Corruption::SlotMissingFromInputs,
            Corruption::ConstStorageMismatch,
            Corruption::SlotStorageMismatch,
            Corruption::SlotKindMismatch,
            Corruption::MissingConstBinding,
            Corruption::MissingStateBinding,
            Corruption::MissingSlotBinding,
            Corruption::DeviceInput,
            Corruption::ConflictingStateDestination,
        ];

        for corruption in cases {
            let (mut graph, ids) = binding_fixture();
            let expected = match corruption {
                Corruption::ConstIdOutOfRange => {
                    graph.consts.push(usize::MAX);
                    GraphValidationError::ConstOutOfRange {
                        value: usize::MAX,
                        value_count: graph.values.len(),
                    }
                }
                Corruption::SlotIdOutOfRange => {
                    graph.slots.push((usize::MAX, Slot::ExpertPoolMap));
                    GraphValidationError::SlotOutOfRange {
                        value: usize::MAX,
                        value_count: graph.values.len(),
                    }
                }
                Corruption::DuplicateInput => {
                    graph.inputs.push(ids.weight);
                    GraphValidationError::DuplicateInput { value: ids.weight }
                }
                Corruption::DuplicateConst => {
                    graph.consts.push(ids.weight);
                    GraphValidationError::DuplicateConst { value: ids.weight }
                }
                Corruption::DuplicateSlot => {
                    graph.slots.push((ids.first_pool, Slot::ExpertPoolMap));
                    GraphValidationError::DuplicateSlot {
                        value: ids.first_pool,
                    }
                }
                Corruption::ConstMissingFromInputs => {
                    graph.inputs.retain(|&id| id != ids.weight);
                    GraphValidationError::ConstNotInput { value: ids.weight }
                }
                Corruption::SlotMissingFromInputs => {
                    graph.inputs.retain(|&id| id != ids.first_pool);
                    GraphValidationError::SlotNotInput {
                        value: ids.first_pool,
                    }
                }
                Corruption::ConstStorageMismatch => {
                    graph.consts.push(ids.first_pool);
                    GraphValidationError::ConstStorageMismatch {
                        value: ids.first_pool,
                        storage: Storage::Slot(Slot::ExpertPoolMap),
                    }
                }
                Corruption::SlotStorageMismatch => {
                    graph.slots.push((ids.weight, Slot::ExpertPoolMap));
                    GraphValidationError::SlotStorageMismatch {
                        value: ids.weight,
                        table_slot: Slot::ExpertPoolMap,
                        storage: Storage::Const,
                    }
                }
                Corruption::SlotKindMismatch => {
                    graph
                        .slots
                        .iter_mut()
                        .find(|(id, _)| *id == ids.first_pool)
                        .unwrap()
                        .1 = Slot::Token;
                    GraphValidationError::SlotStorageMismatch {
                        value: ids.first_pool,
                        table_slot: Slot::Token,
                        storage: Storage::Slot(Slot::ExpertPoolMap),
                    }
                }
                Corruption::MissingConstBinding => {
                    graph.consts.retain(|&id| id != ids.weight);
                    GraphValidationError::MissingConstBinding {
                        value: ids.weight,
                        storage: Storage::Const,
                    }
                }
                Corruption::MissingStateBinding => {
                    graph.consts.retain(|&id| id != ids.first_state);
                    GraphValidationError::MissingConstBinding {
                        value: ids.first_state,
                        storage: Storage::State,
                    }
                }
                Corruption::MissingSlotBinding => {
                    graph.slots.retain(|&(id, _)| id != ids.first_pool);
                    GraphValidationError::MissingSlotBinding {
                        value: ids.first_pool,
                        slot: Slot::ExpertPoolMap,
                    }
                }
                Corruption::DeviceInput => {
                    graph.inputs.push(ids.output);
                    GraphValidationError::DeviceInput { value: ids.output }
                }
                Corruption::ConflictingStateDestination => {
                    graph.state.push((ids.first_state, ids.first_state));
                    GraphValidationError::DuplicateStateDestination {
                        value: ids.first_state,
                    }
                }
            };

            assert_eq!(graph.validate(), Err(expected), "{corruption:?}");
        }
    }

    #[test]
    fn validate_rejects_out_of_range_graph_input_id() {
        let b = Builder::new();
        let output = b.constant("output", TensorType::f32(vec![1]));
        let mut graph = b.finish(output);
        graph.inputs.push(usize::MAX);

        assert_eq!(
            graph.validate(),
            Err(GraphValidationError::InputOutOfRange {
                value: usize::MAX,
                value_count: graph.values.len(),
            })
        );
    }

    #[test]
    fn result_tables_precede_unrelated_equation_errors() {
        #[derive(Clone, Copy, Debug)]
        enum ResultCorruption {
            DuplicateStateDestination,
            StateStorage,
            StateType,
            OutputNotDefined,
            ValidationOutputNotDefined,
            StateOutputNotDefined,
        }

        for corruption in [
            ResultCorruption::DuplicateStateDestination,
            ResultCorruption::StateStorage,
            ResultCorruption::StateType,
            ResultCorruption::OutputNotDefined,
            ResultCorruption::ValidationOutputNotDefined,
            ResultCorruption::StateOutputNotDefined,
        ] {
            let (graph, ids) = binding_fixture();
            let mut graph = graph.with_validations(Vec::new());
            graph.eqns[0].inputs[0] = Operand::Value(usize::MAX);
            // An in-range, correctly typed value with neither an input binder nor a producer.
            let undefined = graph.values.len();
            graph.values.push(ValueMeta {
                aval: TensorType::f32([1]),
                storage: Storage::Device,
                name: None,
                key: None,
                state_role: None,
            });
            let expected = match corruption {
                ResultCorruption::DuplicateStateDestination => {
                    graph.state.push(graph.state[0]);
                    GraphValidationError::DuplicateStateDestination {
                        value: ids.first_state,
                    }
                }
                ResultCorruption::StateStorage => {
                    // Clear the role too (R-644-1: a non-State value carries none), isolating this
                    // corruption to the storage-table mismatch the row below still exercises.
                    graph.values[ids.first_state].storage = Storage::Const;
                    graph.values[ids.first_state].state_role = None;
                    GraphValidationError::StateInputStorageMismatch {
                        value: ids.first_state,
                        storage: Storage::Const,
                    }
                }
                ResultCorruption::StateType => {
                    let state_output = graph.state[0].1;
                    graph.values[state_output].aval = TensorType::f32([2]);
                    GraphValidationError::StateTypeMismatch {
                        state_input: ids.first_state,
                        state_output,
                        input_type: TensorType::f32([1]),
                        output_type: TensorType::f32([2]),
                    }
                }
                ResultCorruption::OutputNotDefined => {
                    graph.output = undefined;
                    GraphValidationError::OutputNotDefined { value: undefined }
                }
                ResultCorruption::ValidationOutputNotDefined => {
                    graph.validations.0.push(ValidationOutput {
                        id: ValidationId(12),
                        name: "undefined".into(),
                        value: undefined,
                    });
                    GraphValidationError::ValidationOutputNotDefined {
                        id: ValidationId(12),
                        value: undefined,
                    }
                }
                ResultCorruption::StateOutputNotDefined => {
                    graph.state[0].1 = undefined;
                    GraphValidationError::StateOutputNotDefined { value: undefined }
                }
            };

            assert_eq!(graph.validate(), Err(expected), "{corruption:?}");
        }
    }

    #[test]
    fn validate_rejects_state_input_absent_from_inputs_and_consts() {
        let b = Builder::new();
        let state = b.state_input("state", TensorType::f32(vec![1]), StateRole::Recurrent);
        let output = b.constant("output", TensorType::f32(vec![1]));
        let mut graph = b.finish_with_state(output, &[(state, output)]);
        graph.inputs.retain(|&id| id != state.id);
        graph.consts.retain(|&id| id != state.id);

        assert_eq!(
            graph.validate(),
            Err(GraphValidationError::StateInputNotInput { value: state.id })
        );
    }

    #[test]
    fn validate_rejects_positional_axis_out_of_range_for_rank() {
        let b = Builder::new();
        let state = b.state_input(
            "state",
            TensorType::f32(vec![1, 2, 3, 4]),
            StateRole::Positional { axis: 4 },
        );
        let output = b.constant("output", TensorType::f32(vec![1]));
        let graph = b.finish_with_state(output, &[(state, state)]);

        assert_eq!(
            graph.validate(),
            Err(GraphValidationError::StateRoleAxis {
                value: state.id,
                axis: 4,
                rank: 4,
            })
        );
    }

    #[test]
    fn validate_rejects_out_of_range_eqn_output_without_panicking() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4]));
        let out = b.binary(BinOp::Mul, x, x);
        let mut g = b.finish(out);
        // Corrupt an equation's output id to point past the value table. validate() must return an Err, not panic on
        // the out-of-bounds `self.values[eqn.out]` index.
        let bad = g.values.len() + 5;
        g.eqns.last_mut().unwrap().out = bad;
        assert_eq!(
            g.validate(),
            Err(GraphValidationError::EquationOutputOutOfRange {
                equation: 0,
                operation: "mul".into(),
                value: bad,
                value_count: g.values.len(),
            })
        );
    }

    #[test]
    fn validate_rejects_a_duplicate_ssa_producer() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4]));
        let internal = b.binary(BinOp::Mul, x, x);
        let out = b.binary(BinOp::Mul, internal, internal);
        let mut g = b.finish(out);
        g.eqns.push(g.eqns[0].clone());

        assert!(matches!(
            g.validate(),
            Err(GraphValidationError::EquationOutputRedefined {
                equation: 2,
                value,
                ..
            }) if value == internal.id
        ));
    }

    #[test]
    fn validate_rejects_state_const_missing_from_input_binders() {
        let b = Builder::new();
        let state = b.state_input(
            "state",
            TensorType::new(vec![2, 5], DType::E4M3FN),
            StateRole::Recurrent,
        );
        let replacement = b.constant("replacement", TensorType::new(vec![2, 5], DType::E4M3FN));
        let mut g = b.finish_with_state(replacement, &[(state, replacement)]);
        g.inputs.retain(|&id| id != state.id);

        assert_eq!(
            g.validate(),
            Err(GraphValidationError::ConstNotInput { value: state.id })
        );
    }

    #[test]
    fn validate_preserves_literal_first_binary_contract() {
        let b = Builder::new();
        let float = b.constant("float", TensorType::f32(vec![2]));
        let out = b.binary_scalar(BinOp::Add, float, Scalar::F32(1.0));
        let mut g = b.finish(out);
        g.eqns[0].inputs.swap(0, 1);
        // Card 149 deliberately makes literal-first Binary graph-valid. Consumers with a narrower operand
        // grammar must reject it locally without changing Graph's general validity contract.
        g.values[out.id].aval = OpKind::Binary(BinOp::Add)
            .infer(&[Scalar::F32(1.0).ty(), TensorType::f32(vec![2])])
            .unwrap();
        g.validate()
            .expect("literal-first Binary remains graph-valid");
    }

    #[test]
    fn validate_rejects_mismatched_binary_value_operands_including_scalars() {
        let b = Builder::new();
        let float = b.constant("float", TensorType::scalar(DType::F32));
        let int = b.constant("int", TensorType::scalar(DType::I32));
        let out = b.binary_scalar(BinOp::Add, float, Scalar::F32(1.0));
        let mut g = b.finish(out);
        g.eqns[0].inputs = vec![Operand::Value(float.id), Operand::Value(int.id)];

        assert_eq!(
            g.validate(),
            Err(GraphValidationError::BinaryValueDtypeMismatch {
                equation: 0,
                operation: "add".into(),
                left: DType::F32,
                right: DType::I32,
            })
        );
    }

    /// Card 529 (R466-011, SC-001): `infer` rejects `Add` on a packed (I8) or storage-only (E4M3FN)
    /// operand pair, so `Graph::validate` (which calls `infer` on every eqn) does too. Both operands
    /// share a dtype here, so `BinaryValueDtypeMismatch`'s equality check never fires; this is `infer`'s
    /// own arithmetic-class rejection at work. The baseline (before Card 529) accepted this graph.
    #[test]
    fn validate_rejects_arithmetic_on_a_packed_or_storage_only_dtype() {
        for dtype in [DType::I8, DType::E4M3FN] {
            let b = Builder::new();
            let float = b.constant("float", TensorType::f32(vec![2]));
            let packed = b.constant("packed", TensorType::new(vec![2], dtype));
            let out = b.binary_scalar(BinOp::Add, float, Scalar::F32(1.0));
            let mut g = b.finish(out);
            g.eqns[0].inputs = vec![Operand::Value(packed.id), Operand::Value(packed.id)];

            assert_eq!(
                g.validate(),
                Err(GraphValidationError::EquationInference {
                    equation: 0,
                    operation: "add".into(),
                    source: crate::error::ShapeError::DtypeOp {
                        op: "Binary",
                        dtype
                    },
                }),
                "{dtype:?}"
            );
        }
    }

    /// Card 528b (R466-013), SC-001. Two graphs built by the identical deterministic call
    /// sequence key their `Token` input equal; a graph differing in one role (`Pos` instead of
    /// `Token`) keys differently. Asserted on `Graph::input_key`, never on the two graphs' raw
    /// `ValueId`s (which coincide below only because both graphs allocate their first value at id
    /// 0 - an allocator fact, not a role fact).
    ///
    /// Mutation (observed red, then reverted): `Graph::input_key` changed to
    /// `Some(InputKey::Const(format!("v{id}")))` unconditionally, ignoring `meta.storage` -
    /// identity by `ValueId` (and its `Debug`-ish rendering) instead of by role. `g1`'s `Token`
    /// key and `g3`'s `Pos` key both became `Some(Const("v0"))` at the shared id 0, so the
    /// "differing role" assertion failed: `assertion \`left != right\` failed: a graph differing
    /// in one role must key differently, even at the same ValueId` (`left`/`right`:
    /// `Some(Const("v0"))`). Restored, re-run green.
    #[test]
    fn input_key_is_structured_not_debug_formatted_and_not_by_value_id() {
        let token_graph = || {
            let b = Builder::new();
            let t = b.slot(Slot::Token, TensorType::scalar(DType::I32));
            b.finish(t)
        };
        let g1 = token_graph();
        let g2 = token_graph();
        let token_id = g1.slots[0].0;
        assert_eq!(
            token_id, g2.slots[0].0,
            "both graphs allocate their Token slot at the same id; the key assertion below must not \
             be able to pass merely because the ids happen to coincide"
        );
        assert_eq!(
            g1.input_key(token_id),
            g2.input_key(token_id),
            "two graphs built the same way must key their Token input equal"
        );

        let b = Builder::new();
        let p = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
        let g3 = b.finish(p);
        assert_eq!(
            g3.slots[0].0, token_id,
            "same allocated id, a different role"
        );
        assert_ne!(
            g1.input_key(token_id),
            g3.input_key(token_id),
            "a graph differing in one role must key differently, even at the same ValueId"
        );
    }

    /// Card 528b (R466-013), SC-001. `Builder::slot_named` lets one graph hold several inputs of
    /// the same `Slot` kind (spec 266: one `ExpertPoolMap` table per MoE layer); the structured key
    /// must tell them apart by their typed tag, not collapse them to their shared role.
    ///
    /// Mutation (observed red, then reverted): `Graph::input_key`'s `Storage::Slot` arm changed to
    /// `Some(InputKey::Slot(SlotKey::new(role, None)))` unconditionally, discarding the parsed tag -
    /// the pre-528b shape, where every occurrence of a kind shared one label
    /// (`format!("{slot:?}").to_lowercase()`) and nothing but the raw name string (never a typed
    /// key) told two occurrences apart. `key0 == key1` failed
    /// (`assertion \`left != right\` failed`, both sides `Some(Slot(SlotKey { role: ExpertPoolMap,
    /// tag: None }))`). Restored, re-run green.
    #[test]
    fn input_key_distinguishes_same_role_tagged_slots() {
        let b = Builder::new();
        let layer0 = b.slot_named(Slot::ExpertPoolMap, "layer_0", TensorType::f32(vec![4]));
        let layer1 = b.slot_named(Slot::ExpertPoolMap, "layer_1", TensorType::f32(vec![4]));
        let out = b.binary(BinOp::Add, layer0, layer0);
        let g = b.finish(out);

        let key0 = g.input_key(layer0.id).expect("declared slot input");
        let key1 = g.input_key(layer1.id).expect("declared slot input");
        assert_ne!(
            key0, key1,
            "same role, different Builder::slot_named tag, must key differently"
        );
    }

    /// Card 528b (R466-013 F1). `Graph::input_key` must read the [`ValueMeta::key`] `Builder` set
    /// at construction, never rebuild it by parsing [`ValueMeta::name`] back apart: corrupting the
    /// wire name after the fact must not change the structured key at all.
    ///
    /// Mutation (observed red, then reverted): `input_key`'s `Storage::Slot` arm changed back to
    /// parsing `meta.name` (the pre-review shape: `meta.name.as_deref().and_then(|name|
    /// parse_slot_key(role, name)).unwrap_or_else(|| SlotKey::new(role, None))`, `parse_slot_key`
    /// temporarily restored alongside it). `after` no longer equalled `original`: `assertion
    /// \`left == right\` failed: input_key must come from the value's stored key, never by
    /// parsing \`name\` / left: Some(Slot(SlotKey { role: ExpertPoolMap, tag: None })) / right:
    /// Some(Slot(SlotKey { role: ExpertPoolMap, tag: Some("layer_0") }))` - the corrupted name no
    /// longer parsed to the real tag, so the reintroduced parse silently lost it. Restored, re-run
    /// green.
    #[test]
    fn input_key_reads_the_stored_key_never_the_name_string() {
        let b = Builder::new();
        let layer0 = b.slot_named(Slot::ExpertPoolMap, "layer_0", TensorType::f32(vec![4]));
        let mut g = b.finish(layer0);
        let original = g.input_key(layer0.id).expect("declared slot input");

        // Corrupt the wire name a parser would read; the typed key lives elsewhere and must be
        // unaffected.
        g.values[layer0.id].name = Some("not.a.recoverable.name".to_string());
        let after = g.input_key(layer0.id);

        assert_eq!(
            after,
            Some(original),
            "input_key must come from the value's stored key, never by parsing `name`"
        );
    }
}
