//! The flat SSA graph: a value table + ordered eqns (the open-form jaxpr analog: binders + body, no data).

use std::collections::HashSet;
use std::fmt;
use std::num::NonZeroU32;

use poot_tensor::HostTensor;

use crate::error::GraphValidationError;
use crate::op::OpKind;
use crate::range::IndexRange;
use crate::rope_table::{self, RopeSpec, RopeTable};
use crate::types::{DType, Scalar, TensorType};

/// An SSA value: an index into [`Graph::values`] (the jaxpr `Var`, an arena index not object identity).
///
/// A newtype over `u32`, so an arbitrary `usize` (an equation index, a dimension, an element count)
/// cannot be passed where a value is meant, and a value cannot be used to index anything but a
/// [`ValueTable`].
///
/// ```compile_fail,E0308
/// use poot_graph_ir::ValueId;
///
/// fn takes_value(_: ValueId) {}
/// takes_value(3usize);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(u32);

impl ValueId {
    /// The value at table position `index`.
    pub const fn new(index: u32) -> ValueId {
        ValueId(index)
    }

    /// The position of this value in [`Graph::values`].
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The value at table position `index`, or `None` when `index` is past the `u32` id space.
    pub fn from_index(index: usize) -> Option<ValueId> {
        u32::try_from(index).ok().map(ValueId)
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl fmt::Debug for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// A table indexed by [`ValueId`]: the value table of a [`Graph`], and every per-value side table a
/// pass keeps beside it. The only way to index it is with a `ValueId`, and the only way to grow it
/// is [`ValueTable::push`], which mints the id of the new entry.
#[derive(Clone, PartialEq, Eq)]
pub struct ValueTable<T> {
    entries: Vec<T>,
}

impl<T> ValueTable<T> {
    /// An empty table.
    pub const fn new() -> Self {
        ValueTable {
            entries: Vec::new(),
        }
    }

    /// An empty table with room for `capacity` entries, so a known-size fill does not reallocate.
    pub fn with_capacity(capacity: usize) -> Self {
        ValueTable {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// A table of `len` copies of `fill`, one per value id `0..len`.
    pub fn filled(len: usize, fill: T) -> Self
    where
        T: Clone,
    {
        ValueTable {
            entries: vec![fill; len],
        }
    }

    /// The number of entries, which is also the next id's index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The id the next [`ValueTable::push`] mints, or `None` when the `u32` id space is full.
    pub fn next_id(&self) -> Option<ValueId> {
        ValueId::from_index(self.entries.len())
    }

    /// Append `entry` and return the id it is addressed by.
    ///
    /// # Panics
    /// When the table already holds `2^32` entries: callers that stage ids from outside check
    /// [`ValueTable::next_id`] first.
    pub fn push(&mut self, entry: T) -> ValueId {
        let id = self
            .next_id()
            .expect("a value table holds at most 2^32 entries");
        self.entries.push(entry);
        id
    }

    /// [`ValueTable::push`] that hands `entry` back when the `u32` id space is full.
    pub fn try_push(&mut self, entry: T) -> Result<ValueId, T> {
        match self.next_id() {
            Some(id) => {
                self.entries.push(entry);
                Ok(id)
            }
            None => Err(entry),
        }
    }

    /// The entry `id` addresses, or `None` when `id` is past the end.
    pub fn get(&self, id: ValueId) -> Option<&T> {
        self.entries.get(id.index())
    }

    /// The entry `id` addresses, mutably, or `None` when `id` is past the end.
    pub fn get_mut(&mut self, id: ValueId) -> Option<&mut T> {
        self.entries.get_mut(id.index())
    }

    /// Whether `id` addresses an entry of this table.
    pub fn contains(&self, id: ValueId) -> bool {
        id.index() < self.entries.len()
    }

    /// Every entry, in id order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.entries.iter()
    }

    /// Every entry, mutably, in id order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.entries.iter_mut()
    }

    /// Every entry with the id it is addressed by, in id order.
    pub fn iter_enumerated(&self) -> impl ExactSizeIterator<Item = (ValueId, &T)> + '_ {
        self.entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (ValueId(index as u32), entry))
    }

    /// Every id of this table, in order.
    pub fn ids(&self) -> impl ExactSizeIterator<Item = ValueId> + use<T> {
        (0..self.entries.len() as u32).map(ValueId)
    }

    /// Drop every entry from index `len` on; ids at or past `len` stop addressing anything.
    pub fn truncate(&mut self, len: usize) {
        self.entries.truncate(len);
    }

    /// Reserve room for `additional` more entries, reporting allocation failure instead of aborting.
    pub fn try_reserve(
        &mut self,
        additional: usize,
    ) -> Result<(), std::collections::TryReserveError> {
        self.entries.try_reserve(additional)
    }
}

impl<T> Default for ValueTable<T> {
    fn default() -> Self {
        ValueTable::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for ValueTable<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&self.entries).finish()
    }
}

impl<T> std::ops::Index<ValueId> for ValueTable<T> {
    type Output = T;

    fn index(&self, id: ValueId) -> &T {
        &self.entries[id.index()]
    }
}

impl<T> std::ops::IndexMut<ValueId> for ValueTable<T> {
    fn index_mut(&mut self, id: ValueId) -> &mut T {
        &mut self.entries[id.index()]
    }
}

impl<T> FromIterator<T> for ValueTable<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        ValueTable {
            entries: iter.into_iter().collect(),
        }
    }
}

impl<T> IntoIterator for ValueTable<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a ValueTable<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

/// A decoder-layer tag on an equation: the layer whose loop emitted it, as reported by
/// [`crate::Builder::layer_scope`].
///
/// Layer tags are the graph-side image of a layer-pipeline placement: a placement records
/// `after_layer` cuts between devices, and the stage split maps those cuts onto this graph through
/// the tag ([`Graph::layer_runs`] is the one query),
/// so no caller has to infer a layer from a name. `None` means "emitted outside any layer scope" -
/// the graph-wide preamble (slots, guards, masks), the head, or an equation a transform
/// synthesized. The tag is data, not semantics: an evaluator or planner that does not split by
/// layer ignores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerIndex(pub usize);

/// The equation range of every decoder layer of a graph, as [`Graph::layer_runs`] reads it from the
/// [`LayerIndex`] tags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerRuns {
    /// Number of layers: the tagged region holds layer ids `0..layers`. Zero for a graph with no
    /// layer tags.
    pub layers: usize,
    /// One `[start, end)` equation range per layer, in layer order; `runs.len() == layers`.
    pub runs: Vec<std::ops::Range<usize>>,
}

/// Why a graph's layer tags are not one run per layer in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LayerTagError {
    /// An equation without a layer tag sits between the first and last tagged equations.
    #[error(
        "equation {equation} is untagged between the first and last layer-tagged equation; a \
         layer-cut split will not guess which side it belongs to"
    )]
    Untagged {
        /// The index of the untagged equation.
        equation: usize,
    },
    /// Layer tags are not one contiguous run per layer in ascending order.
    #[error(
        "equation {equation} is tagged layer {layer}, breaking the one-run-per-layer order the \
         layer cuts map onto (expected layer {expected})"
    )]
    NonContiguous {
        /// The index of the equation that breaks the order.
        equation: usize,
        /// The layer it is tagged with.
        layer: usize,
        /// The layer the order required at that point.
        expected: usize,
    },
}

/// Stable model-adapter key for one bounded validation witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValidationId(pub u32);

/// Names one value a graph hands to something outside it (a later stage of a split program). The
/// name is stable across every pass that renumbers or merges values; the value it resolves to is
/// [`Graph::exports`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BoundaryId(pub ValueId);

impl BoundaryId {
    /// The [`SlotKey`] a consuming stage binds this boundary under: [`Slot::Activation`] tagged with
    /// the boundary, so it never collides with the stage's own activation input.
    pub fn slot_key(self) -> SlotKey {
        SlotKey::new(
            Slot::Activation,
            Some(&format!("boundary.{}", self.0.index())),
        )
    }
}

/// One exported value: a liveness root the passes keep alive and remap, beside the primary output.
/// Several exports may resolve to one value (CSE merges them); each keeps its own id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Export {
    /// The name the consuming side resolves the value by.
    pub id: BoundaryId,
    /// The exported value.
    pub value: ValueId,
}

/// The most bytes one validation packet may occupy: the budget the packet layout is checked against.
pub const MAX_VALIDATION_PACKET_BYTES: usize = 4096;

/// Where one validation witness sits in the packet a program returns: its lanes are the f32 words
/// `first_lane..first_lane + lane_count`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationPacketEntry {
    /// The witness's stable key.
    pub id: ValidationId,
    /// The witness's published name.
    pub name: String,
    /// Index of the witness's first lane in the packet.
    pub first_lane: usize,
    /// Number of f32 lanes the witness occupies.
    pub lane_count: usize,
}

/// The layout of one validation packet: every witness's lanes, in declaration order, laid end to end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationPacketLayout {
    /// One entry per witness, in lane order.
    pub entries: Vec<ValidationPacketEntry>,
    /// Total f32 lanes of the packet.
    pub lane_count: usize,
    /// Total bytes of the packet; bounded by [`MAX_VALIDATION_PACKET_BYTES`].
    pub byte_len: usize,
}

/// A validation witness lane that held anything but a signed zero when a program ran: a witness reads zero
/// when every check passed.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[error("validation {name} ({id:?}) lane {lane} failed with f32 bits 0x{observed_bits:08x}")]
pub struct ExecutionValidationFailure {
    /// The failing witness's stable key.
    pub id: ValidationId,
    /// The failing witness's published name.
    pub name: String,
    /// The lane of the witness that failed.
    pub lane: usize,
    /// The f32 bit pattern the lane held.
    pub observed_bits: u32,
}

/// Why a returned validation packet is not a clean pass.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ValidationPacketError {
    /// The packet's lane count differs from the layout's.
    #[error("validation packet has {actual} lanes; expected {expected}")]
    Length {
        /// Lanes the layout declares.
        expected: usize,
        /// Lanes the packet held.
        actual: usize,
    },
    /// A witness lane failed.
    #[error(transparent)]
    Failure(#[from] ExecutionValidationFailure),
}

/// An operand: an SSA value reference or an inline literal (the jaxpr `Atom`).
#[derive(Clone, Copy, Debug)]
pub enum Operand {
    /// A value of the graph, by id.
    Value(ValueId),
    /// An inline scalar literal, broadcast as a rank-0 value.
    Lit(Scalar),
}

/// Runtime inputs. Most variants are graph-architecture.md PerTokenSlots: `Token` and `Pos` are scalars. The
/// attention mask is not a slot: it is an in-graph computation over `Pos` and an iota.
/// `Activation` is an explicitly named standalone boundary, not a model-engine slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Slot {
    /// The current step's token id, a scalar per token row.
    Token,
    /// A caller-supplied dense activation for a standalone graph boundary. Unlike [`Slot::Token`], it is not a
    /// token id and has no model-engine binding convention. Declare it with [`crate::builder::Builder::slot_named`]
    /// and bind it by value id or name through a caller-owned contract, such as the driver-adjacent resident E4M3
    /// linear handoff.
    Activation,
    /// The current step's position id, a scalar per token row; the attention mask is computed from it in-graph.
    Pos,
    /// Card 152 / spec 267: one atomically-bound temporal/height/width position tuple for sectioned multimodal RoPE.
    /// Shape `[3,N]` for the `N = rows * tokens` positions of a step (`[3,L]` for prefill, `[3,B]` for statically
    /// batched decode), axis-major in the order temporal, height, width. One packed kind keeps a binder from updating only a subset of the axes;
    /// model tracers split it with Slice/Reshape before `rope_sectioned`. Per-prompt/per-step data, never a `Const`.
    MropePosition,
    /// Paged-KV read map `[rows, cap]` (i32): row `r`'s logical key position `t` -> the flat physical slot in the
    /// block pool holding its K/V. Attention reads the pool through it in place ([`crate::KeyLayout::Paged`]);
    /// a consumer that is not that attention gathers the logical view by it ([`crate::ops::paged_view`]).
    /// Declare it with [`crate::builder::Builder::read_map`], whose range `[0, pool_slots)` is a checked bind
    /// fact: the kernel's unchecked pool read stands on it. Per-step-varying, so a Slot.
    SlotMap,
    /// A forward write map `[N]` (i32) of an in-place row write ([`crate::OpKind::ScatterUpdate`]): for each
    /// of the step's `N` new rows, the pool slot it is written to, or `-1` for a row that writes nothing. One
    /// thread owns one source row, so two rows naming one slot would race; the kind is bound only as a
    /// [`crate::ForwardSlotMap`], whose constructor refuses a repeated or out-of-pool slot. Declare it with
    /// [`crate::builder::Builder::slot_named_ranged`] over `[-1, pool_slots)`; several occurrences (a KV pool's
    /// and a recurrent pool's) are told apart by tag.
    WriteMap,
    /// The current token's already-embedded input row `[hidden]`, host-computed per step (a single-row dequant, not
    /// a GPU gather). Lets a decode graph skip declaring the full `[vocab,hidden]` embedding table as a GPU-resident
    /// Const, needed when its dense-f32 size exceeds a backend's max single-buffer size (gemma4's `hidden=5376`
    /// gives ~5.6 GB, over wgpu's ~2 GB `max_buffer_size` here).
    TokenEmbed,
    /// Spec 248 (epic 129 B8): a batched LoRA decode graph's per-row adapter selection `[batch]` (I32, the index
    /// dtype `IndexedMatMul` reads): row `m`'s value is the
    /// adapter index (1-based; 0 is the no-adapter row) whose stacked `A`/`B` the LoRA compile pass applies to
    /// that row this step. A distinct variant from `Slot::SlotMap`: an engine binder resolves one whole `Slot` kind
    /// from one buffer per bind call. Per-step-varying (requests and adapters change as the batch admits/evicts),
    /// so a `Slot` rebound every step, never a `Const` (const-cache staleness).
    LoraIdx,
    /// Spec 266: an MoE layer's expert-pool assignment table `[E]` (I32, the index dtype `Gather` reads): entry
    /// `e` is the pool slot currently holding global expert `e`, or `-1` when
    /// not resident in that layer's pool this step. An in-graph `Gather` translates the router's global expert ids
    /// into pool-slot ids with this table, so the unmodified `IndexedMatMul` op addresses a smaller
    /// `[P, ...]` pool weight instead of the full `[E, ...]` stack.
    ///
    /// Unlike every other `Slot` kind, one graph contains many of these (one per MoE layer, each with different
    /// content in the same step). Slot binders resolve a whole kind from one buffer (a `Slot::SlotMap` at two shapes), which would
    /// stamp the same bytes onto every layer's table. So this kind must be declared through
    /// [`crate::builder::Builder::slot_named`], giving each occurrence a distinct [`ValueMeta::name`], and its
    /// binders must resolve per name (as `Const` does), never per kind. Per-step-varying (a refill moves experts
    /// between slots), so a `Slot`, never a `Const`: the const cache keys by name + element count and would serve a
    /// stale table.
    ExpertPoolMap,
    /// An expert table paged through a pool (POOT-746): one `[P, ..]` slot per pooled routed table
    /// (a dense `[E, ..]` stack or one packed source role of it), declared by the pool pass in place of
    /// the table's `Const` and tagged with that const's name. Unlike every other slot it is not written
    /// from the step's inputs: it is resident for the executable's life and the expert-residency
    /// manager refills one row at a time at a sync point. Several occurrences exist in one graph, one
    /// per table, so the key is its tag, never the kind.
    ExpertPool,
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
            Slot::SlotMap => "slotmap",
            Slot::WriteMap => "writemap",
            Slot::TokenEmbed => "tokenembed",
            Slot::LoraIdx => "loraidx",
            Slot::ExpertPoolMap => "expertpoolmap",
            Slot::ExpertPool => "expertpool",
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
    /// An intermediate living in a device buffer the planner assigns.
    Device,
    /// A weight or table bound at graph close, uploaded once.
    Const,
    /// A constant the compiler materializes from its own definition; nothing binds it.
    Computed(ComputedConst),
    /// A per-step input routed through a stable device buffer.
    Slot(Slot),
    /// A persistent buffer carried across steps and overwritten by its paired `state_out`.
    State,
}

impl Storage {
    /// Whether the buffer holds a weight's stored bytes in the lane its readers' storage plan chose: a
    /// `Const` uploaded once, the resident [`Slot::ExpertPool`] a host refills row by row from the
    /// same stored bytes, or the [`Slot::TokenEmbed`] rows a host gathers from an embedding table too
    /// large for one device buffer and encodes in the storage the plan chose. Lane decisions made for a
    /// weight (a packed BF16 table, a packed contraction source) apply to all three.
    pub fn holds_weight_bytes(self) -> bool {
        matches!(
            self,
            Storage::Const | Storage::Slot(Slot::ExpertPool | Slot::TokenEmbed)
        )
    }
}

/// A constant the compiler computes, not a checkpoint binder. The `fold_iota` transform turns a nullary
/// [`crate::OpKind::Iota`] into one of these, so the planner never lowers an iota kernel and no name is
/// bound: a binder materializes the value from this payload exactly like a slot from its [`Slot`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ComputedConst {
    /// The range `[0, 1, .., len)`, shape `[len]`, in `dtype` (F32 or I32; an I32 range is exact for every
    /// `len` a graph shape can hold, an F32 one only below 2^24).
    Iota {
        /// The number of elements.
        len: usize,
        /// The element dtype, F32 or I32.
        dtype: DType,
    },
    /// One table of a RoPE pair, shape `[positions, spec.dim()]`, from
    /// [`crate::rope_table::rope_tables`]. `positions` is the entry's capacity, not the model's
    /// maximum context.
    Rope {
        /// Which of the pair this constant is.
        table: RopeTable,
        /// The number of rows: the highest position the table serves, plus one.
        positions: u32,
        /// The rotary width, base and scaling the table derives from.
        spec: RopeSpec,
    },
    /// The ALiBi (Press et al.) per-head slopes of a model with `heads` attention heads, shape
    /// `[heads]`: `2^(-8 (i + 1) / n)` for `n` the largest power of two not above `heads`, then the
    /// odd-indexed slopes of the `2n` series for the remaining heads (BLOOM's and MPT's rule).
    AlibiSlopes {
        /// The model's attention head count.
        heads: usize,
    },
}

fn alibi_series(n: usize) -> impl Iterator<Item = f32> {
    let start = 2f32.powf(-8.0 / n as f32);
    (0..n).map(move |i| start.powi(i as i32 + 1))
}

impl ComputedConst {
    /// The dtype the graph declares for the value this constant materializes into.
    pub fn dtype(self) -> DType {
        match self {
            ComputedConst::Iota { dtype, .. } => dtype,
            ComputedConst::Rope { .. } | ComputedConst::AlibiSlopes { .. } => DType::F32,
        }
    }

    /// The exact constant as a host tensor of [`Self::dtype`] and [`Self::shape`]: the one materialization
    /// every binder and oracle shares.
    ///
    /// # Panics
    /// An `Iota` whose dtype is neither F32 nor I32, or whose `len` does not fit an `i32`: the graph
    /// shape rule ([`crate::OpKind::Iota`]) rejects both before a constant exists.
    pub fn tensor(self) -> HostTensor {
        match self {
            ComputedConst::Iota {
                len,
                dtype: DType::I32,
            } => HostTensor::i32(
                vec![len],
                (0..len)
                    .map(|i| i32::try_from(i).expect("an I32 iota length fits i32"))
                    .collect(),
            ),
            ComputedConst::Iota { dtype, .. } if dtype != DType::F32 => {
                panic!("an iota constant is F32 or I32, got {dtype:?}")
            }
            _ => HostTensor::f32(self.shape(), self.values_f32()),
        }
    }

    /// The elements of an F32 constant, in row-major order.
    ///
    /// `len` is bounded by the graph shape it materializes into (the `as f32` cast of an iota never
    /// loses an integer below 2^24, the F32 iota's checked bound).
    fn values_f32(self) -> Vec<f32> {
        match self {
            ComputedConst::Iota { len, .. } => (0..len).map(|i| i as f32).collect(),
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

    /// The rank of [`Self::shape`], known without allocating it.
    pub(crate) fn rank(self) -> usize {
        match self {
            ComputedConst::Iota { .. } | ComputedConst::AlibiSlopes { .. } => 1,
            ComputedConst::Rope { .. } => 2,
        }
    }

    /// The shape of the value this constant materializes into.
    pub fn shape(self) -> Vec<usize> {
        match self {
            ComputedConst::Iota { len, .. } => vec![len],
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
/// graph identity does not include it: it changes no kernel, buffer, or binding, so `poot-wgpu-device`'s cache
/// key (hashed from `Graph::state` and the storage tag) is untouched by a role change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StateRole {
    /// Written at the live position along `axis`; a stale write past the committed length is
    /// harmless. The `[1, Hkv, cap, D]` architecture KV caches carry the position on axis 2.
    Positional {
        /// The axis along which the live position indexes the buffer.
        axis: u8,
    },
    /// Written per block of `ratio` consecutive positions along `axis`: position `p` lives at index
    /// `p / ratio`, so the axis holds `capacity / ratio` entries.
    /// Like [`StateRole::Positional`] an entry past the committed length is harmless, because a
    /// reader hides it by position. The QSA block-key cache is the one user.
    Blocked {
        /// The axis along which a position's block indexes the buffer.
        axis: u8,
        /// The positions each entry along `axis` covers.
        ratio: NonZeroU32,
    },
    /// Folds every token into its value: conv windows, GDN/KDA/SSM matrices, ring buffers written at
    /// `pos % w`, and the pooled `[pool_slots, hkv, d]` caches (position on axis 0). The safe default
    /// every existing state pair is traced with; `Positional` is declared only by `components::attention`
    /// and later family cards.
    Recurrent,
    /// A value one program writes and a later program of the same driver step reads (the hidden-state
    /// carry of POOT-1109): never rolled back, so a snapshot skips it and a restore leaves it alone,
    /// and it leaves a graph's residency `KvOnly` when nothing else is `Recurrent`. A writer
    /// overwrites only the rows its step carries (the first `tokens` of the carry's position axis);
    /// the rest keep what earlier steps wrote, so a reader must know how many rows the last writer
    /// covered and read no more.
    Handoff,
}

/// One `(state_in, state_out)` pair from [`Graph::state`], with the input's declared [`StateRole`] read
/// alongside it. Built by [`Graph::state_pairs`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatePair {
    /// The `State` input binder the step reads.
    pub input: ValueId,
    /// The value the step overwrites the state with.
    pub output: ValueId,
    /// The role declared on `input`.
    pub role: StateRole,
}

/// What a graph records about one SSA value: its abstract type, how it is stored, and the optional
/// name and construction-time facts (slot key, state role, declared range) binders and passes read.
#[derive(Clone, Debug)]
pub struct ValueMeta {
    /// The value's shape and dtype, computed host-side at construction.
    pub aval: TensorType,
    /// How the value is stored.
    pub storage: Storage,
    /// The value's name: the lookup key of a `Const`, the wire spelling of a `Slot`, a label otherwise.
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
    /// The range a [`Storage::Slot`] or [`Storage::Const`] input declares for its I32 words, set once
    /// at construction ([`crate::builder::Builder::slot_ranged`],
    /// [`crate::builder::Builder::constant_ranged`]) and checked by the binder against the bound
    /// words, so [`crate::value_ranges`] may treat it as a fact. `None` for every other value.
    pub(crate) range: Option<IndexRange>,
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
            range: None,
        }
    }

    /// The input binder a consuming stage declares for the exported value `id`: a keyed
    /// [`Slot::Activation`] ([`BoundaryId::slot_key`]), named by that key's wire spelling like every
    /// other keyed slot.
    pub fn boundary_input(aval: TensorType, id: BoundaryId) -> Self {
        let key = id.slot_key();
        ValueMeta {
            aval,
            storage: Storage::Slot(Slot::Activation),
            name: Some(key.to_string()),
            key: Some(key),
            state_role: None,
            range: None,
        }
    }

    /// A `Storage::State` input's `ValueMeta`, carrying its declared `role`. The only constructor
    /// that can produce a `State` meta (R-644-1): [`Graph::validate`] rejects a `Storage::State`
    /// value built any other way (it would carry no role).
    #[cfg(any(test, feature = "test-support"))]
    pub fn new_state(aval: TensorType, name: Option<String>, role: StateRole) -> Self {
        ValueMeta {
            aval,
            storage: Storage::State,
            name,
            key: None,
            state_role: Some(role),
            range: None,
        }
    }

    /// The range this input declares for its I32 words, if any. The binders check it at bind;
    /// [`crate::value_ranges`] reads it as proven.
    pub fn range(&self) -> Option<IndexRange> {
        self.range
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
    /// The operation this equation applies.
    pub op: OpKind,
    /// The operands, in the operation's operand order.
    pub inputs: Vec<Operand>,
    /// The value this equation defines.
    pub out: ValueId,
    /// The decoder layer that emitted this equation, set by [`crate::Builder::layer_scope`] for
    /// every equation emitted inside the scope. `None` outside a scope, and `None` for an equation
    /// a transform synthesized with a fresh `out` (the source equation's own `out` keeps its tag).
    pub layer: Option<LayerIndex>,
}

/// What one entry of a graph's output list publishes.
///
/// A graph has one output list, private to [`Graph`], and every observable or carried value is an
/// entry of it: the primary result, the validation witnesses, the values handed to other stages, and
/// the next value of each carried-state buffer. There is no second encoding to keep in step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputKind {
    /// The primary output (the next-token logits). Exactly one per graph.
    Result,
    /// A bounded F32 validation witness, checked before the result or state is published. The value
    /// remains an ordinary F32 graph value.
    Witness {
        /// The witness's stable key.
        id: ValidationId,
        /// The witness's published name.
        name: String,
    },
    /// A value handed to another stage of a split program, named by `BoundaryId`.
    Export(BoundaryId),
    /// The value a [`Storage::State`] input is overwritten with at the end of the step (the jax
    /// `donate_argnums` / XLA `input_output_aliases` contract).
    StateNext {
        /// The `State` input this value overwrites.
        input: ValueId,
    },
}

/// One entry of a graph's output list: what it publishes, and the value it publishes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphOutput {
    /// What the entry publishes.
    pub kind: OutputKind,
    /// The published value.
    pub value: ValueId,
}

/// One validation witness, borrowed from the output list it is declared in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Witness<'g> {
    /// The witness's stable key.
    pub id: ValidationId,
    /// The witness's published name.
    pub name: &'g str,
    /// The F32 value the witness checks.
    pub value: ValueId,
}

/// Why an edit was refused. The graph the edit was opened from is untouched; no refused draft
/// becomes a [`Graph`].
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum EditError {
    /// The draft is not a well-formed graph: a dangling operand, an operand or output type that
    /// disagrees with its operation's inference, a malformed binding or state table, ...
    #[error("the edit leaves an invalid graph: {0}")]
    Invalid(#[from] GraphValidationError),
    /// The edit changed the type of the primary result: a rewrite preserves what the graph computes.
    #[error("the edit changes the result type from {before} to {after}")]
    ResultRetyped {
        /// The result type of the graph the edit was opened from.
        before: TensorType,
        /// The result type of the draft.
        after: TensorType,
    },
    /// The edit lost a validation witness: a graph that publishes fewer checks than the one it
    /// replaces would run unverified.
    #[error("the edit drops validation witness {id:?} ({name:?})")]
    WitnessDropped {
        /// The dropped witness's key.
        id: ValidationId,
        /// The dropped witness's name.
        name: String,
    },
    /// The edit declares a witness the graph it was opened from does not carry.
    #[error("the edit adds validation witness {id:?} ({name:?})")]
    WitnessAdded {
        /// The added witness's key.
        id: ValidationId,
        /// The added witness's name.
        name: String,
    },
    /// The edit names a witness as added that is not a new witness of the draft: absent from it, or
    /// already carried by the graph it was opened from.
    #[error("witness {id:?} is named as added but is not a new witness of the edit")]
    WitnessNotAdded {
        /// The witness the edit named as added.
        id: ValidationId,
    },
    /// The edit lost an export: a later stage would wait for a value no stage produces.
    #[error("the edit drops export {id:?}")]
    ExportDropped {
        /// The dropped export's boundary id.
        id: BoundaryId,
    },
    /// The edit lost a carried-state pair: the buffer would stop being carried across steps.
    #[error("the edit drops the carried-state pair of input {input}")]
    StateDropped {
        /// The `State` input that lost its pair.
        input: ValueId,
    },
    /// A piece of a split claims a witness the split graph does not carry.
    #[error("witness {id:?} is not a witness of the split graph")]
    WitnessNotClaimable {
        /// The witness the piece claimed.
        id: ValidationId,
    },
}

/// The interface of the graph an edit was opened from: what a rewrite must preserve.
#[derive(Clone, Debug)]
struct Interface {
    result: TensorType,
    witnesses: Vec<(ValidationId, String)>,
    exports: Vec<BoundaryId>,
    state_inputs: Vec<ValueId>,
}

/// A flat SSA primitive tensor-op graph.
///
/// A `Graph` is always well formed: its tables are private, so the only ways to obtain one are
/// [`Builder::finish`](crate::Builder::finish) and [`GraphEdit::commit`], and both check the whole
/// graph first (SSA order, operand and output types, binding and state tables, one result, bounded
/// F32 witnesses). A pass or executor that holds a `Graph` need not re-validate it, and cannot
/// change it in place: it edits through [`Graph::edit`], and the result is checked again.
///
/// ```compile_fail,E0616
/// use poot_graph_ir::{Builder, Graph, TensorType};
///
/// let b = Builder::new();
/// let out = b.constant("out", TensorType::f32([1]));
/// let graph: Graph = b.finish(out).unwrap();
/// let _ = graph.outputs;
/// ```
///
/// ```compile_fail,E0616
/// use poot_graph_ir::{Builder, Graph, TensorType};
///
/// let b = Builder::new();
/// let out = b.constant("out", TensorType::f32([1]));
/// let mut graph: Graph = b.finish(out).unwrap();
/// graph.eqns.clear();
/// ```
///
/// ```compile_fail,E0599
/// use poot_graph_ir::Graph;
///
/// let _ = Graph::default();
/// ```
#[derive(Clone, Debug)]
pub struct Graph {
    values: ValueTable<ValueMeta>,
    /// All input binders (consts + slots), in declaration order.
    inputs: Vec<ValueId>,
    /// The subset of `inputs` that are constants (weights/tables), for binding at close.
    consts: Vec<ValueId>,
    /// The per-token-varying inputs and their slot kind.
    slots: Vec<(ValueId, Slot)>,
    eqns: Vec<Eqn>,
    /// Every published value: one [`OutputKind::Result`], then witnesses, exports and carried-state
    /// next values, each in declaration order within its kind.
    outputs: Vec<GraphOutput>,
}

impl Graph {
    /// The table of every value's [`ValueMeta`], addressed by [`ValueId`].
    pub fn values(&self) -> &ValueTable<ValueMeta> {
        &self.values
    }

    /// All input binders (consts + slots), in declaration order.
    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    /// The subset of [`Graph::inputs`] that are constants (weights/tables), for binding at close.
    pub fn consts(&self) -> &[ValueId] {
        &self.consts
    }

    /// The per-token-varying inputs and their slot kind.
    pub fn slots(&self) -> &[(ValueId, Slot)] {
        &self.slots
    }

    /// The equations in SSA order: each reads only inputs and the outputs of earlier equations.
    pub fn eqns(&self) -> &[Eqn] {
        &self.eqns
    }

    /// The primary result (the next-token logits).
    pub fn result(&self) -> ValueId {
        self.outputs
            .iter()
            .find(|output| output.kind == OutputKind::Result)
            .map(|output| output.value)
            .expect("a Graph always has one Result output: `check` refuses a draft without one")
    }

    /// Validation witnesses in declaration order.
    pub fn witnesses(&self) -> impl Iterator<Item = Witness<'_>> + '_ {
        self.outputs.iter().filter_map(|output| match &output.kind {
            OutputKind::Witness { id, name } => Some(Witness {
                id: *id,
                name,
                value: output.value,
            }),
            _ => None,
        })
    }

    /// Values handed to other stages, each a liveness root that [`GraphEdit::remap_results`]
    /// rewrites. Empty for a graph that is not one stage of a split program.
    pub fn exports(&self) -> impl Iterator<Item = Export> + '_ {
        self.outputs.iter().filter_map(|output| match &output.kind {
            OutputKind::Export(id) => Some(Export {
                id: *id,
                value: output.value,
            }),
            _ => None,
        })
    }

    /// The carried-state pairs `(state_in, state_out)` in declaration order: `state_in` is a
    /// `Storage::State` input binder and `state_out` a produced (or pass-through) value of
    /// identical type. The executor feeds each cache buffer in as `state_in` and overwrites it with
    /// `state_out`. Empty for stateless graphs.
    pub fn state(&self) -> impl Iterator<Item = (ValueId, ValueId)> + '_ {
        self.outputs.iter().filter_map(|output| match output.kind {
            OutputKind::StateNext { input } => Some((input, output.value)),
            _ => None,
        })
    }

    /// The one tag query: where each decoder layer's equations sit. The tagged region (first to
    /// last tagged equation) must hold no untagged equation and one run per layer id, `0, 0, .., 1,
    /// 1, ..` in order; untagged equations before it (preamble) and after it (head) are expected and
    /// belong to no run. A graph with no tags has zero layers.
    pub fn layer_runs(&self) -> Result<LayerRuns, LayerTagError> {
        let tags: Vec<Option<LayerIndex>> = self.eqns.iter().map(|eqn| eqn.layer).collect();
        let (Some(first), Some(last)) = (
            tags.iter().position(Option::is_some),
            tags.iter().rposition(Option::is_some),
        ) else {
            return Ok(LayerRuns {
                layers: 0,
                runs: Vec::new(),
            });
        };
        let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
        for (equation, tag) in tags.iter().enumerate().take(last + 1).skip(first) {
            let Some(LayerIndex(layer)) = *tag else {
                return Err(LayerTagError::Untagged { equation });
            };
            let current = runs.len().checked_sub(1);
            if current == Some(layer) {
                runs[layer].end = equation + 1;
                continue;
            }
            let next = runs.len();
            if layer != next {
                return Err(LayerTagError::NonContiguous {
                    equation,
                    layer,
                    expected: match current {
                        Some(current) if layer < current => current,
                        _ => next,
                    },
                });
            }
            runs.push(equation..equation + 1);
        }
        Ok(LayerRuns {
            layers: runs.len(),
            runs,
        })
    }

    /// The abstract type of `id`. Panics when `id` is not a value of this graph.
    pub fn aval(&self, id: ValueId) -> &TensorType {
        &self.values[id].aval
    }
    /// The [`ValueMeta`] of `id`. Panics when `id` is not a value of this graph.
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
    /// [`ValueMeta::state_role`]. A graph gives every pair a role (the edit check rejects a
    /// `Storage::State` input with none).
    pub fn state_pairs(&self) -> impl Iterator<Item = StatePair> + '_ {
        self.state().map(move |(input, output)| StatePair {
            input,
            output,
            role: self.state_role(input).unwrap_or(StateRole::Recurrent),
        })
    }

    /// The values a caller can observe: the result, the witnesses and the exports.
    pub fn observable_values(&self) -> impl Iterator<Item = ValueId> + '_ {
        std::iter::once(self.result())
            .chain(self.witnesses().map(|witness| witness.value))
            .chain(self.exports().map(|export| export.value))
    }

    /// The values that must stay alive for the graph to be complete: every observable value and the next value of each carried-state pair.
    pub fn liveness_roots(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.observable_values()
            .chain(self.state().map(|(_, output)| output))
    }

    /// The liveness roots plus every `State` input binder: the values that must survive any rewrite.
    pub fn pinned_values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.liveness_roots()
            .chain(self.state().map(|(input, _)| input))
    }

    /// Open this graph for editing. The returned draft is a copy: nothing changes until
    /// [`GraphEdit::commit`] checks it and returns the new graph.
    pub fn edit(&self) -> GraphEdit {
        self.clone().into_edit()
    }

    /// Open this graph for editing, consuming it. See [`Graph::edit`].
    pub fn into_edit(self) -> GraphEdit {
        let interface = Interface {
            result: self.aval(self.result()).clone(),
            witnesses: self
                .witnesses()
                .map(|witness| (witness.id, witness.name.to_owned()))
                .collect(),
            exports: self.exports().map(|export| export.id).collect(),
            state_inputs: self.state().map(|(input, _)| input).collect(),
        };
        GraphEdit {
            draft: self,
            interface: Some(interface),
        }
    }

    /// Open this graph as the start of a new program: the draft has every table of this graph and no
    /// baseline, so it may publish a different result (a sampler suffix over the logits) and is
    /// checked for well-formedness only. A rewrite that must preserve what the graph computes uses
    /// [`Graph::edit`].
    pub(crate) fn into_draft(self) -> GraphEdit {
        GraphEdit {
            draft: self,
            interface: None,
        }
    }
}

/// A draft of a graph under edit: the one way to change a [`Graph`].
///
/// A draft reads like the graph it holds (it derefs to [`Graph`]) and is changed through the typed
/// operations below. It is never executed or planned: [`GraphEdit::commit`] checks the whole draft
/// and either returns the new [`Graph`] or a typed [`EditError`] naming what is wrong.
///
/// A draft opened from a graph ([`Graph::edit`]) is a rewrite: it must preserve that graph's
/// interface. The result keeps its type, every witness and export stays declared, and every carried
/// state pair stays carried; losing a witness is [`EditError::WitnessDropped`]. A new draft
/// ([`GraphEdit::new`], and the one a [`Builder`](crate::Builder) fills) has no such baseline. A
/// graph split into pieces accounts its witnesses through a [`WitnessLedger`] instead.
///
/// ```compile_fail,E0616
/// use poot_graph_ir::{Builder, TensorType};
///
/// let b = Builder::new();
/// let out = b.constant("out", TensorType::f32([1]));
/// let graph = b.finish(out).unwrap();
/// let edit = graph.edit();
/// let _ = edit.draft;
/// ```
#[derive(Clone, Debug)]
pub struct GraphEdit {
    draft: Graph,
    interface: Option<Interface>,
}

impl std::ops::Deref for GraphEdit {
    type Target = Graph;

    fn deref(&self) -> &Graph {
        &self.draft
    }
}

impl Default for GraphEdit {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphEdit {
    /// An empty draft with no baseline: no values, no equations and no result yet.
    pub fn new() -> Self {
        GraphEdit {
            draft: Graph {
                values: ValueTable::new(),
                inputs: Vec::new(),
                consts: Vec::new(),
                slots: Vec::new(),
                eqns: Vec::new(),
                outputs: Vec::new(),
            },
            interface: None,
        }
    }

    /// The draft's value table, to push new values or retype existing ones.
    pub fn values_mut(&mut self) -> &mut ValueTable<ValueMeta> {
        &mut self.draft.values
    }

    /// The draft's equation list, in SSA order.
    pub fn eqns_mut(&mut self) -> &mut Vec<Eqn> {
        &mut self.draft.eqns
    }

    /// The draft's input binder list, in declaration order.
    pub fn inputs_mut(&mut self) -> &mut Vec<ValueId> {
        &mut self.draft.inputs
    }

    /// The draft's constant binder list, the subset of the inputs that are weights or tables.
    pub fn consts_mut(&mut self) -> &mut Vec<ValueId> {
        &mut self.draft.consts
    }

    /// The draft's slot binder list, each input with its [`Slot`] kind.
    pub fn slots_mut(&mut self) -> &mut Vec<(ValueId, Slot)> {
        &mut self.draft.slots
    }

    /// Reserve room for `additional` output declarations (result, witnesses, exports, state pairs).
    pub fn try_reserve_outputs(
        &mut self,
        additional: usize,
    ) -> Result<(), std::collections::TryReserveError> {
        self.draft.outputs.try_reserve(additional)
    }

    /// Declare the primary result, replacing the previous one.
    pub fn set_result(&mut self, value: ValueId) {
        match self
            .draft
            .outputs
            .iter_mut()
            .find(|output| output.kind == OutputKind::Result)
        {
            Some(output) => output.value = value,
            None => self.draft.outputs.insert(
                0,
                GraphOutput {
                    kind: OutputKind::Result,
                    value,
                },
            ),
        }
    }

    /// Declare a validation witness after the ones already declared.
    pub fn push_witness(&mut self, id: ValidationId, name: impl Into<String>, value: ValueId) {
        let at = self
            .draft
            .outputs
            .iter()
            .rposition(|output| {
                matches!(output.kind, OutputKind::Result | OutputKind::Witness { .. })
            })
            .map_or(0, |last| last + 1);
        self.draft.outputs.insert(
            at,
            GraphOutput {
                kind: OutputKind::Witness {
                    id,
                    name: name.into(),
                },
                value,
            },
        );
    }

    /// Declare an export after the ones already declared.
    pub fn push_export(&mut self, export: Export) {
        let at = self
            .draft
            .outputs
            .iter()
            .rposition(|output| !matches!(output.kind, OutputKind::StateNext { .. }))
            .map_or(0, |last| last + 1);
        self.draft.outputs.insert(
            at,
            GraphOutput {
                kind: OutputKind::Export(export.id),
                value: export.value,
            },
        );
    }

    /// Declare a carried-state pair after the ones already declared.
    pub fn push_state(&mut self, input: ValueId, output: ValueId) {
        self.draft.outputs.push(GraphOutput {
            kind: OutputKind::StateNext { input },
            value: output,
        });
    }

    /// Retarget the next value of the state pair of `input`. Returns whether `input` is carried.
    pub fn set_state_next(&mut self, input: ValueId, value: ValueId) -> bool {
        match self
            .draft
            .outputs
            .iter_mut()
            .find(|output| output.kind == (OutputKind::StateNext { input }))
        {
            Some(output) => {
                output.value = value;
                true
            }
            None => false,
        }
    }

    /// Retarget every output value (result, witnesses, exports and state next values) through
    /// `remap`. The declarations themselves keep their identity and order.
    pub fn remap_results(&mut self, mut remap: impl FnMut(ValueId) -> ValueId) {
        for output in &mut self.draft.outputs {
            output.value = remap(output.value);
        }
    }

    /// Drop the witnesses the predicate rejects, keeping the survivors in declaration order. A
    /// rewrite that drops a witness is refused at [`GraphEdit::commit`]; a split accounts the
    /// dropped ones through a [`WitnessLedger`].
    #[cfg(any(test, feature = "test-support"))]
    pub fn retain_witnesses(&mut self, mut keep: impl FnMut(Witness<'_>) -> bool) {
        self.draft.outputs.retain(|output| match &output.kind {
            OutputKind::Witness { id, name } => keep(Witness {
                id: *id,
                name,
                value: output.value,
            }),
            _ => true,
        });
    }

    /// Remove the carried-state pairs the predicate rejects (`(input, output)`), keeping the
    /// survivors in declaration order. A rewrite that drops a pair is refused at
    /// [`GraphEdit::commit`].
    pub fn retain_state(&mut self, mut keep: impl FnMut(ValueId, ValueId) -> bool) {
        self.draft.outputs.retain(|output| match output.kind {
            OutputKind::StateNext { input } => keep(input, output.value),
            _ => true,
        });
    }

    /// Remove every export. A rewrite that drops one is refused at [`GraphEdit::commit`].
    #[cfg(any(test, feature = "test-support"))]
    pub fn clear_exports(&mut self) {
        self.draft
            .outputs
            .retain(|output| !matches!(output.kind, OutputKind::Export(_)));
    }

    /// Turn the named `Const` input `value` into the resident [`Slot::ExpertPool`] of a pool of `rows`
    /// rows: the leading axis shrinks to `rows`, the value keeps its id and dtype, and its identity is
    /// the key tagged with the const's own name, which is returned. The const's name is therefore
    /// still how the table is found in the weight map; only its role in the graph changes.
    ///
    /// # Panics
    ///
    /// When `value` is not a named `Const` input with a leading axis.
    pub fn rebind_const_as_pool(&mut self, value: ValueId, rows: usize) -> SlotKey {
        let meta = &mut self.draft.values[value];
        assert_eq!(
            meta.storage,
            Storage::Const,
            "rebind_const_as_pool: {value} is not a const"
        );
        let tag = meta
            .name
            .clone()
            .expect("rebind_const_as_pool: a const input carries its name");
        let leading = meta
            .aval
            .shape
            .first_mut()
            .expect("rebind_const_as_pool: a pooled table has a leading expert axis");
        *leading = rows;
        let key = SlotKey::new(Slot::ExpertPool, Some(&tag));
        meta.storage = Storage::Slot(Slot::ExpertPool);
        meta.name = Some(key.to_string());
        meta.key = Some(key.clone());
        self.draft.consts.retain(|&c| c != value);
        self.draft.slots.push((value, Slot::ExpertPool));
        key
    }

    /// Declare a new keyed `slot` input of type `aval`, tagged `tag` like
    /// [`crate::builder::Builder::slot_named`], and return its value. For a transform that rewrites a
    /// finished graph and so has no builder: the key is built here, from typed data.
    ///
    /// # Panics
    ///
    /// When a slot of this kind and tag is already declared.
    pub fn declare_slot(&mut self, slot: Slot, tag: &str, aval: TensorType) -> ValueId {
        let key = SlotKey::new(slot, Some(tag));
        let name = key.to_string();
        assert!(
            !self
                .draft
                .values
                .iter()
                .any(|v| v.storage == Storage::Slot(slot) && v.name.as_deref() == Some(&name)),
            "declare_slot: duplicate tag {tag:?} for slot {slot:?}"
        );
        let id = self.draft.values.push(ValueMeta {
            aval,
            storage: Storage::Slot(slot),
            name: Some(name),
            key: Some(key),
            state_role: None,
            range: None,
        });
        self.draft.inputs.push(id);
        self.draft.slots.push((id, slot));
        id
    }

    /// Check the whole draft and return it as a [`Graph`].
    ///
    /// # Errors
    ///
    /// [`EditError::Invalid`] when the draft is not well formed; for a rewrite, the typed
    /// conservation error ([`EditError::WitnessDropped`], [`EditError::ResultRetyped`], ...) when it
    /// loses part of the interface of the graph it was opened from.
    pub fn commit(self) -> Result<Graph, EditError> {
        self.commit_adding_witnesses(&[])
    }

    /// [`GraphEdit::commit`] for a rewrite that declares witnesses of its own (a pass that publishes a
    /// failure of its own making, such as the expert pool's row guard): each id in `added` must be a
    /// witness the draft carries and the graph it was opened from did not. Every other witness is still
    /// conserved, and an added witness not named here is still [`EditError::WitnessAdded`].
    pub fn commit_adding_witnesses(self, added: &[ValidationId]) -> Result<Graph, EditError> {
        self.draft.check()?;
        if let Some(interface) = &self.interface {
            self.draft.conserves(interface, added)?;
        }
        Ok(self.draft)
    }

    /// [`GraphEdit::commit`] for one piece of a split: the piece may carry fewer witnesses than the
    /// graph the split started from, and must claim each one it carries from `ledger`. The piece
    /// still has to be well formed.
    pub fn commit_piece(self, ledger: &mut WitnessLedger) -> Result<Graph, EditError> {
        self.draft.check()?;
        ledger.claim(&self.draft)?;
        Ok(self.draft)
    }
}

impl Graph {
    /// Whether `self`, a rewrite of the graph `interface` describes, still publishes all of it.
    fn conserves(&self, interface: &Interface, added: &[ValidationId]) -> Result<(), EditError> {
        let after = self.aval(self.result());
        if *after != interface.result {
            return Err(EditError::ResultRetyped {
                before: interface.result.clone(),
                after: after.clone(),
            });
        }
        for (id, name) in &interface.witnesses {
            if !self.witnesses().any(|witness| witness.id == *id) {
                return Err(EditError::WitnessDropped {
                    id: *id,
                    name: name.clone(),
                });
            }
        }
        for id in added {
            if !self.witnesses().any(|witness| witness.id == *id)
                || interface.witnesses.iter().any(|(known, _)| known == id)
            {
                return Err(EditError::WitnessNotAdded { id: *id });
            }
        }
        for witness in self.witnesses() {
            if !interface.witnesses.iter().any(|(id, _)| *id == witness.id)
                && !added.contains(&witness.id)
            {
                return Err(EditError::WitnessAdded {
                    id: witness.id,
                    name: witness.name.to_owned(),
                });
            }
        }
        for id in &interface.exports {
            if !self.exports().any(|export| export.id == *id) {
                return Err(EditError::ExportDropped { id: *id });
            }
        }
        for input in &interface.state_inputs {
            if !self.state().any(|(state_input, _)| state_input == *input) {
                return Err(EditError::StateDropped { input: *input });
            }
        }
        Ok(())
    }
}

/// The accounting of a graph's witnesses across the pieces it is split into.
///
/// A split gives each piece only the witnesses it defines, so no piece conserves the whole set and
/// [`GraphEdit::commit`] cannot be the check. The ledger is: every piece [claims](Self::claim) what
/// it carries, and [`WitnessLedger::finish`] fails closed when some witness of the split graph was
/// claimed by no piece, so a split cannot silently shed a check.
#[derive(Debug)]
pub struct WitnessLedger {
    /// The witnesses no piece has claimed yet.
    pending: Vec<(ValidationId, String)>,
    /// Every witness the split graph carried.
    known: Vec<ValidationId>,
}

impl WitnessLedger {
    /// A ledger holding every witness of `graph`, none claimed yet.
    pub fn of(graph: &Graph) -> Self {
        WitnessLedger {
            pending: graph
                .witnesses()
                .map(|witness| (witness.id, witness.name.to_owned()))
                .collect(),
            known: graph.witnesses().map(|witness| witness.id).collect(),
        }
    }

    /// Record that `piece` carries its witnesses. A witness may be carried by several pieces (a
    /// check whose value a later stage reads runs in each), but never one the split graph lacks.
    fn claim(&mut self, piece: &Graph) -> Result<(), EditError> {
        for witness in piece.witnesses() {
            if !self.known.contains(&witness.id) {
                return Err(EditError::WitnessNotClaimable { id: witness.id });
            }
            self.pending.retain(|(id, _)| *id != witness.id);
        }
        Ok(())
    }

    /// Succeed when every witness the ledger was opened with was claimed by some piece.
    pub fn finish(self) -> Result<(), EditError> {
        match self.pending.into_iter().next() {
            Some((id, name)) => Err(EditError::WitnessDropped { id, name }),
            None => Ok(()),
        }
    }
}

impl Graph {
    /// Check basic SSA and binding-table well-formedness. Cheap structural check.
    fn check(&self) -> Result<(), GraphValidationError> {
        let result = self
            .outputs
            .iter()
            .filter(|o| o.kind == OutputKind::Result)
            .count();
        if result == 0 {
            return Err(GraphValidationError::MissingResult);
        }
        if result > 1 {
            return Err(GraphValidationError::MultipleResults);
        }
        let output = self.result();
        let n = self.values.len();
        if !self.values.contains(output) {
            return Err(GraphValidationError::OutputNotDefined { value: output });
        }
        let validations: Vec<Witness<'_>> = self.witnesses().collect();
        let exports: Vec<Export> = self.exports().collect();
        let state: Vec<(ValueId, ValueId)> = self.state().collect();
        let mut validation_ids = HashSet::with_capacity(validations.len());
        let mut validation_names = HashSet::with_capacity(validations.len());
        for validation in &validations {
            if !self.values.contains(validation.value) {
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
            if !validation_names.insert(validation.name) {
                return Err(GraphValidationError::DuplicateValidationName {
                    name: validation.name.to_owned(),
                });
            }
        }
        // The packet layout checks each validation value's F32 dtype, static shape, and payload bound.
        ValidationPacketLayout::for_outputs(&self.values, validations.iter().copied())?;
        for &(state_input, state_output) in &state {
            if !self.values.contains(state_input) || !self.values.contains(state_output) {
                return Err(GraphValidationError::StatePairOutOfRange {
                    state_input,
                    state_output,
                    value_count: n,
                });
            }
        }
        let mut export_ids = HashSet::with_capacity(exports.len());
        for export in &exports {
            if !self.values.contains(export.value) {
                return Err(GraphValidationError::ExportValueOutOfRange {
                    id: export.id,
                    value: export.value,
                    value_count: n,
                });
            }
            if !export_ids.insert(export.id) {
                return Err(GraphValidationError::DuplicateExportId { id: export.id });
            }
        }
        // Input binders are defined up front. Binding tables are an executor-facing contract, so validate them before
        // equation inspection can follow malformed metadata.
        let mut input_bindings = ValueTable::filled(n, false);
        for &id in &self.inputs {
            if !self.values.contains(id) {
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

        let mut const_bindings = ValueTable::filled(n, false);
        for &id in &self.consts {
            if !self.values.contains(id) {
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

        let mut slot_bindings = ValueTable::filled(n, false);
        for &(id, slot) in &self.slots {
            if !self.values.contains(id) {
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
        for (id, meta) in self.values.iter_enumerated() {
            match (meta.storage, meta.state_role) {
                (Storage::State, None) => {
                    return Err(GraphValidationError::MissingStateRole { value: id });
                }
                (
                    Storage::State,
                    Some(StateRole::Positional { axis } | StateRole::Blocked { axis, .. }),
                ) => {
                    let rank = meta.aval.rank();
                    if usize::from(axis) >= rank {
                        return Err(GraphValidationError::StateRoleAxis {
                            value: id,
                            axis,
                            rank,
                        });
                    }
                }
                (Storage::State, Some(StateRole::Recurrent | StateRole::Handoff)) => {}
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
        let mut state_destinations = ValueTable::filled(n, false);
        for &(state_input, state_output) in &state {
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
            if !self.values.contains(eqn.out) {
                return Err(GraphValidationError::EquationOutputOutOfRange {
                    equation: i,
                    operation: eqn.op.name(),
                    value: eqn.out,
                    value_count: n,
                });
            }
            bound_or_produced[eqn.out] = true;
        }
        if !bound_or_produced[output] {
            return Err(GraphValidationError::OutputNotDefined { value: output });
        }
        for validation in &validations {
            if !bound_or_produced[validation.value] {
                return Err(GraphValidationError::ValidationOutputNotDefined {
                    id: validation.id,
                    value: validation.value,
                });
            }
        }
        for export in &exports {
            if !bound_or_produced[export.value] {
                return Err(GraphValidationError::ExportNotDefined {
                    id: export.id,
                    value: export.value,
                });
            }
        }
        for &(_, state_output) in &state {
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
                        if !self.values.contains(*v) || !defined[*v] {
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
    /// Checked packet layout of a graph's validation declarations. A graph with no witnesses has the
    /// empty layout.
    pub fn for_graph(graph: &Graph) -> Self {
        Self::for_outputs(&graph.values, graph.witnesses())
            .expect("a Graph's witnesses were checked when it was built")
    }

    fn for_outputs<'g>(
        values: &ValueTable<ValueMeta>,
        validations: impl Iterator<Item = Witness<'g>>,
    ) -> Result<Self, GraphValidationError> {
        let mut entries = Vec::new();
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
                name: validation.name.to_owned(),
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

    /// Check the f32 bit patterns of a returned packet against this layout. Fails on a lane-count mismatch or on the first failing witness lane.
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

/// Whether a `MatMul` eqn's stored output aval is a legitimate F32-accumulate widening of its
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
/// Only MatMul gets this exception: every other op's `infer` must match the stored aval exactly.
fn is_matmul_f32_accumulate_widening(
    op: &OpKind,
    inferred: &TensorType,
    stored: &TensorType,
) -> bool {
    matches!(op, OpKind::MatMul)
        && inferred.shape == stored.shape
        && stored.dtype == DType::F32
        && matches!(inferred.dtype, DType::F16 | DType::BF16)
}

impl fmt::Display for Graph {
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
            writeln!(f, "    {id} : {} = slot {key}", self.values[*id].aval)?;
        }
        writeln!(
            f,
            "  // consts: {} weight/table binders (elided)",
            self.consts.len()
        )?;
        writeln!(f, "  // body ({} eqns):", self.eqns.len())?;
        for eqn in &self.eqns {
            let outm = &self.values[eqn.out];
            write!(f, "    {} : {} = {} (", eqn.out, outm.aval, eqn.op.name())?;
            for (i, inp) in eqn.inputs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match inp {
                    Operand::Value(id) => write!(f, "{id}")?,
                    Operand::Lit(s) => write!(f, "{s}")?,
                }
            }
            match &outm.name {
                Some(name) => writeln!(f, ")   // {name}")?,
                None => writeln!(f, ")")?,
            }
        }
        writeln!(
            f,
            "  output: {} : {}",
            self.result(),
            self.aval(self.result())
        )?;
        let mut validations = self.witnesses().peekable();
        if validations.peek().is_some() {
            writeln!(f, "  // validations (in declaration order):")?;
            for validation in validations {
                writeln!(
                    f,
                    "  validation {:?} {:?}: {} : {}",
                    validation.id,
                    validation.name,
                    validation.value,
                    self.aval(validation.value)
                )?;
            }
        }
        if self.state().next().is_some() {
            writeln!(f, "  // state (carried cache, in -> out):")?;
            for pair in self.state_pairs() {
                writeln!(
                    f,
                    "    {} : {} <- {}  [{:?}]",
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
        EditError, GraphEdit, GraphValidationError, Operand, Slot, StateRole, Storage,
        ValidationId, ValidationPacketError, ValidationPacketLayout, ValueId, ValueMeta,
    };
    use poot_tensor::DType;
    use std::num::NonZeroU32;

    /// The typed validity error a draft's commit refuses with.
    fn commit_invalid(draft: GraphEdit) -> GraphValidationError {
        match draft.commit() {
            Err(EditError::Invalid(error)) => error,
            other => panic!("expected an invalid draft, got {other:?}"),
        }
    }

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
        let b = Builder::for_tests();
        let weight = b.constant("weight", TensorType::f32(vec![1])).unwrap();
        let first_pool = b
            .slot_named(
                Slot::ExpertPoolMap,
                "layer_0",
                TensorType::new(vec![1], DType::I32),
            )
            .unwrap();
        let _second_pool = b
            .slot_named(
                Slot::ExpertPoolMap,
                "layer_1",
                TensorType::new(vec![1], DType::I32),
            )
            .unwrap();
        let first_state = b
            .state_input(
                "first_state",
                TensorType::f32(vec![1]),
                StateRole::Recurrent,
            )
            .unwrap();
        let second_state = b
            .state_input(
                "second_state",
                TensorType::f32(vec![1]),
                StateRole::Recurrent,
            )
            .unwrap();
        let output = b.binary(BinOp::Add, first_state, second_state).unwrap();
        let graph = b
            .finish(
                crate::Outputs::new(output)
                    .with_state(&[(first_state, second_state), (second_state, first_state)]),
            )
            .unwrap();
        (
            graph,
            BindingFixtureIds {
                weight: weight.id(),
                first_pool: first_pool.id(),
                first_state: first_state.id(),
                output: output.id(),
            },
        )
    }

    #[test]
    fn validate_accepts_a_well_formed_graph() {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
        let out = b.binary(BinOp::Mul, x, x).unwrap();
        let g = b.finish(out).unwrap();
        crate::test_support::verdict(&g).expect("a well-formed graph must validate");
    }

    #[test]
    fn validation_layout_is_ordered_and_bounded() {
        let b = Builder::for_tests();
        let primary = b.constant("primary", TensorType::f32([1])).unwrap();
        let pair = b.constant("pair", TensorType::f32([2])).unwrap();
        let scalar = b.constant("scalar", TensorType::f32([])).unwrap();
        let graph = crate::test_support::graph_fixtures::finish_with_validations(
            b,
            primary,
            &[
                (ValidationId(7), "pair-check", pair),
                (ValidationId(9), "scalar-check", scalar),
            ],
        )
        .unwrap();

        let layout = ValidationPacketLayout::for_graph(&graph);
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
        let b = Builder::for_tests();
        let primary = b.constant("primary", TensorType::f32([1])).unwrap();
        let witness = b.constant("witness", TensorType::f32([2])).unwrap();
        let graph = crate::test_support::graph_fixtures::finish_with_validations(
            b,
            primary,
            &[(ValidationId(7), "check", witness)],
        )
        .unwrap();
        let layout = ValidationPacketLayout::for_graph(&graph);

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
        let b = Builder::for_tests();
        let primary = b.constant("primary", TensorType::f32([1])).unwrap();
        let mut graph = b.finish(primary).unwrap().into_draft();
        graph.push_witness(ValidationId(1), "range", ValueId::new(u32::MAX));
        assert!(matches!(
            commit_invalid(graph.clone()),
            GraphValidationError::ValidationValueOutOfRange { value, .. }
                if value == ValueId::new(u32::MAX)
        ));

        let undefined = ValueId::from_index(graph.values().len()).unwrap();
        graph.values_mut().push(ValueMeta {
            aval: TensorType::f32([1]),
            storage: Storage::Device,
            name: None,
            key: None,
            state_role: None,
            range: None,
        });
        graph.remap_results(|value| {
            if value == ValueId::new(u32::MAX) {
                undefined
            } else {
                value
            }
        });
        assert!(matches!(
            commit_invalid(graph),
            GraphValidationError::ValidationOutputNotDefined {
                id: ValidationId(1),
                ..
            }
        ));

        let declare = |cases: [(ValidationId, &str); 2]| {
            let mut draft = GraphEdit::new();
            let value = draft.values_mut().push(ValueMeta::new(
                TensorType::f32([1]),
                Storage::Device,
                None,
            ));
            draft.set_result(value);
            for (id, name) in cases {
                draft.push_witness(id, name, value);
            }
            draft
        };
        let mut graph = declare([(ValidationId(1), ""), (ValidationId(1), "same")]);
        assert!(matches!(
            commit_invalid(graph.clone()),
            GraphValidationError::EmptyValidationName {
                id: ValidationId(1)
            }
        ));

        graph = declare([(ValidationId(1), "same"), (ValidationId(1), "same")]);
        assert!(matches!(
            commit_invalid(graph.clone()),
            GraphValidationError::DuplicateValidationId {
                id: ValidationId(1)
            }
        ));
        graph = declare([(ValidationId(1), "same"), (ValidationId(2), "same")]);
        assert!(matches!(
            commit_invalid(graph),
            GraphValidationError::DuplicateValidationName { ref name } if name == "same"
        ));

        let b = Builder::for_tests();
        let primary = b.constant("primary", TensorType::f32([1])).unwrap();
        let witness = b
            .constant("invalid", TensorType::new([1], DType::I32))
            .unwrap();
        assert!(matches!(
            crate::test_support::graph_fixtures::finish_with_validations(b, primary, &[(ValidationId(1), "invalid", witness)]),
            Err(GraphValidationError::ValidationDtype {
                id: ValidationId(1),
                value,
                actual: DType::I32,
            }) if value == witness.id()
        ));

        let b = Builder::for_tests();
        let primary = b.constant("primary", TensorType::f32([1])).unwrap();
        let first = b.constant("first", TensorType::f32([usize::MAX])).unwrap();
        let second = b.constant("second", TensorType::f32([1])).unwrap();
        assert!(matches!(
            crate::test_support::graph_fixtures::finish_with_validations(
                b,
                primary,
                &[
                    (ValidationId(1), "first", first),
                    (ValidationId(2), "second", second),
                ],
            ),
            Err(GraphValidationError::ValidationPacketLaneOverflow)
        ));

        let b = Builder::for_tests();
        let primary = b.constant("primary", TensorType::f32([1])).unwrap();
        let witness = b
            .constant(
                "byte-overflow",
                TensorType::f32([usize::MAX / std::mem::size_of::<f32>() + 1]),
            )
            .unwrap();
        assert!(matches!(
            crate::test_support::graph_fixtures::finish_with_validations(
                b,
                primary,
                &[(ValidationId(1), "byte-overflow", witness)]
            ),
            Err(GraphValidationError::ValidationPacketByteOverflow)
        ));

        let b = Builder::for_tests();
        let primary = b.constant("primary", TensorType::f32([1])).unwrap();
        let witness = b.constant("invalid", TensorType::f32([0])).unwrap();
        assert!(matches!(
            crate::test_support::graph_fixtures::finish_with_validations(b, primary, &[(ValidationId(1), "invalid", witness)]),
            Err(GraphValidationError::EmptyValidationValue {
                id: ValidationId(1),
                value,
            }) if value == witness.id()
        ));

        let b = Builder::for_tests();
        let primary = b.constant("primary", TensorType::f32([1])).unwrap();
        let witness = b
            .constant("invalid", TensorType::f32([usize::MAX, 2]))
            .unwrap();
        assert!(matches!(
            crate::test_support::graph_fixtures::finish_with_validations(b, primary, &[(ValidationId(1), "invalid", witness)]),
            Err(GraphValidationError::ValidationShapeOverflow {
                id: ValidationId(1),
                value,
                ..
            }) if value == witness.id()
        ));

        let b = Builder::for_tests();
        let primary = b.constant("primary", TensorType::f32([1])).unwrap();
        // Spec 375 fixes the payload at 4,096 bytes: 1,024 f32 lanes pass and 1,025 do not. Literal lane
        // counts keep this test sensitive to a change of the bound itself.
        let witness = b.constant("bounded", TensorType::f32([1024])).unwrap();
        crate::test_support::graph_fixtures::finish_with_validations(
            b,
            primary,
            &[(ValidationId(1), "bounded", witness)],
        )
        .unwrap();

        let b = Builder::for_tests();
        let primary = b.constant("primary", TensorType::f32([1])).unwrap();
        let witness = b.constant("too-large", TensorType::f32([1025])).unwrap();
        assert!(matches!(
            crate::test_support::graph_fixtures::finish_with_validations(
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
                .slots()
                .iter()
                .filter(|(_, slot)| *slot == Slot::ExpertPoolMap)
                .count(),
            2,
            "fixture must exercise two separately named slots of one kind"
        );
        assert_ne!(
            graph.meta(graph.slots()[0].0).name,
            graph.meta(graph.slots()[1].0).name,
            "the same-kind slots must remain distinguishable by name"
        );
        crate::test_support::verdict(&graph)
            .expect("same-kind named slots and a state swap are valid");

        for (label, state) in [
            ("identity", [(0, 0), (1, 1)]),
            ("shared source", [(0, 0), (1, 0)]),
        ] {
            let b = Builder::for_tests();
            let first = b
                .state_input("first", TensorType::f32(vec![1]), StateRole::Recurrent)
                .unwrap();
            let second = b
                .state_input("second", TensorType::f32(vec![1]), StateRole::Recurrent)
                .unwrap();
            let output = b.binary(BinOp::Add, first, second).unwrap();
            let values = [first, second];
            let state = state.map(|(destination, source)| (values[destination], values[source]));
            let graph = b
                .finish(crate::Outputs::new(output).with_state(&state))
                .unwrap();
            crate::test_support::verdict(&graph)
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
            let (graph, ids) = binding_fixture();
            let mut graph = graph.into_draft();
            let expected = match corruption {
                Corruption::ConstIdOutOfRange => {
                    graph.consts_mut().push(ValueId::new(u32::MAX));
                    GraphValidationError::ConstOutOfRange {
                        value: ValueId::new(u32::MAX),
                        value_count: graph.values().len(),
                    }
                }
                Corruption::SlotIdOutOfRange => {
                    graph
                        .slots_mut()
                        .push((ValueId::new(u32::MAX), Slot::ExpertPoolMap));
                    GraphValidationError::SlotOutOfRange {
                        value: ValueId::new(u32::MAX),
                        value_count: graph.values().len(),
                    }
                }
                Corruption::DuplicateInput => {
                    graph.inputs_mut().push(ids.weight);
                    GraphValidationError::DuplicateInput { value: ids.weight }
                }
                Corruption::DuplicateConst => {
                    graph.consts_mut().push(ids.weight);
                    GraphValidationError::DuplicateConst { value: ids.weight }
                }
                Corruption::DuplicateSlot => {
                    graph
                        .slots_mut()
                        .push((ids.first_pool, Slot::ExpertPoolMap));
                    GraphValidationError::DuplicateSlot {
                        value: ids.first_pool,
                    }
                }
                Corruption::ConstMissingFromInputs => {
                    graph.inputs_mut().retain(|&id| id != ids.weight);
                    GraphValidationError::ConstNotInput { value: ids.weight }
                }
                Corruption::SlotMissingFromInputs => {
                    graph.inputs_mut().retain(|&id| id != ids.first_pool);
                    GraphValidationError::SlotNotInput {
                        value: ids.first_pool,
                    }
                }
                Corruption::ConstStorageMismatch => {
                    graph.consts_mut().push(ids.first_pool);
                    GraphValidationError::ConstStorageMismatch {
                        value: ids.first_pool,
                        storage: Storage::Slot(Slot::ExpertPoolMap),
                    }
                }
                Corruption::SlotStorageMismatch => {
                    graph.slots_mut().push((ids.weight, Slot::ExpertPoolMap));
                    GraphValidationError::SlotStorageMismatch {
                        value: ids.weight,
                        table_slot: Slot::ExpertPoolMap,
                        storage: Storage::Const,
                    }
                }
                Corruption::SlotKindMismatch => {
                    graph
                        .slots_mut()
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
                    graph.consts_mut().retain(|&id| id != ids.weight);
                    GraphValidationError::MissingConstBinding {
                        value: ids.weight,
                        storage: Storage::Const,
                    }
                }
                Corruption::MissingStateBinding => {
                    graph.consts_mut().retain(|&id| id != ids.first_state);
                    GraphValidationError::MissingConstBinding {
                        value: ids.first_state,
                        storage: Storage::State,
                    }
                }
                Corruption::MissingSlotBinding => {
                    graph.slots_mut().retain(|&(id, _)| id != ids.first_pool);
                    GraphValidationError::MissingSlotBinding {
                        value: ids.first_pool,
                        slot: Slot::ExpertPoolMap,
                    }
                }
                Corruption::DeviceInput => {
                    graph.inputs_mut().push(ids.output);
                    GraphValidationError::DeviceInput { value: ids.output }
                }
                Corruption::ConflictingStateDestination => {
                    graph.push_state(ids.first_state, ids.first_state);
                    GraphValidationError::DuplicateStateDestination {
                        value: ids.first_state,
                    }
                }
            };

            assert_eq!(
                crate::test_support::verdict(&graph),
                Err(expected),
                "{corruption:?}"
            );
        }
    }

    #[test]
    fn validate_rejects_out_of_range_graph_input_id() {
        let b = Builder::for_tests();
        let output = b.constant("output", TensorType::f32(vec![1])).unwrap();
        let mut graph = b.finish(output).unwrap().into_draft();
        graph.inputs_mut().push(ValueId::new(u32::MAX));

        assert_eq!(
            crate::test_support::verdict(&graph),
            Err(GraphValidationError::InputOutOfRange {
                value: ValueId::new(u32::MAX),
                value_count: graph.values().len(),
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
            let mut graph = graph.into_draft();
            graph.eqns_mut()[0].inputs[0] = Operand::Value(ValueId::new(u32::MAX));
            // An in-range, correctly typed value with neither an input binder nor a producer.
            let undefined = ValueId::from_index(graph.values().len()).unwrap();
            graph.values_mut().push(ValueMeta {
                aval: TensorType::f32([1]),
                storage: Storage::Device,
                name: None,
                key: None,
                state_role: None,
                range: None,
            });
            let expected = match corruption {
                ResultCorruption::DuplicateStateDestination => {
                    let first = graph.state().next().unwrap();
                    graph.push_state(first.0, first.1);
                    GraphValidationError::DuplicateStateDestination {
                        value: ids.first_state,
                    }
                }
                ResultCorruption::StateStorage => {
                    // Clear the role too (R-644-1: a non-State value carries none), isolating this
                    // corruption to the storage-table mismatch the row below still exercises.
                    graph.values_mut()[ids.first_state].storage = Storage::Const;
                    graph.values_mut()[ids.first_state].state_role = None;
                    GraphValidationError::StateInputStorageMismatch {
                        value: ids.first_state,
                        storage: Storage::Const,
                    }
                }
                ResultCorruption::StateType => {
                    let state_output = graph.state().next().unwrap().1;
                    graph.values_mut()[state_output].aval = TensorType::f32([2]);
                    GraphValidationError::StateTypeMismatch {
                        state_input: ids.first_state,
                        state_output,
                        input_type: TensorType::f32([1]),
                        output_type: TensorType::f32([2]),
                    }
                }
                ResultCorruption::OutputNotDefined => {
                    graph.set_result(undefined);
                    GraphValidationError::OutputNotDefined { value: undefined }
                }
                ResultCorruption::ValidationOutputNotDefined => {
                    graph.push_witness(ValidationId(12), "undefined", undefined);
                    GraphValidationError::ValidationOutputNotDefined {
                        id: ValidationId(12),
                        value: undefined,
                    }
                }
                ResultCorruption::StateOutputNotDefined => {
                    let input = graph.state().next().unwrap().0;
                    graph.set_state_next(input, undefined);
                    GraphValidationError::StateOutputNotDefined { value: undefined }
                }
            };

            assert_eq!(
                crate::test_support::verdict(&graph),
                Err(expected),
                "{corruption:?}"
            );
        }
    }

    #[test]
    fn validate_rejects_state_input_absent_from_inputs_and_consts() {
        let b = Builder::for_tests();
        let state = b
            .state_input("state", TensorType::f32(vec![1]), StateRole::Recurrent)
            .unwrap();
        let output = b.constant("output", TensorType::f32(vec![1])).unwrap();
        let mut graph = b
            .finish(crate::Outputs::new(output).with_state(&[(state, output)]))
            .unwrap()
            .into_draft();
        graph.inputs_mut().retain(|&id| id != state.id());
        graph.consts_mut().retain(|&id| id != state.id());

        assert_eq!(
            crate::test_support::verdict(&graph),
            Err(GraphValidationError::StateInputNotInput { value: state.id() })
        );
    }

    #[test]
    fn validate_rejects_positional_axis_out_of_range_for_rank() {
        let b = Builder::for_tests();
        let state = b
            .state_input(
                "state",
                TensorType::f32(vec![1, 2, 3, 4]),
                StateRole::Positional { axis: 4 },
            )
            .unwrap();
        let output = b.constant("output", TensorType::f32(vec![1])).unwrap();
        let refused = b.finish(crate::Outputs::new(output).with_state(&[(state, state)]));

        assert_eq!(
            refused.err(),
            Some(crate::BuilderError::from(EditError::Invalid(
                GraphValidationError::StateRoleAxis {
                    value: state.id(),
                    axis: 4,
                    rank: 4,
                }
            )))
        );
    }

    #[test]
    fn validate_rejects_blocked_axis_out_of_range_for_rank() {
        let b = Builder::for_tests();
        let state = b
            .state_input(
                "state",
                TensorType::f32(vec![1, 2, 3, 4]),
                StateRole::Blocked {
                    axis: 4,
                    ratio: NonZeroU32::new(2).unwrap(),
                },
            )
            .unwrap();
        let output = b.constant("output", TensorType::f32(vec![1])).unwrap();
        let refused = b.finish(crate::Outputs::new(output).with_state(&[(state, state)]));

        assert_eq!(
            refused.err(),
            Some(crate::BuilderError::from(EditError::Invalid(
                GraphValidationError::StateRoleAxis {
                    value: state.id(),
                    axis: 4,
                    rank: 4,
                }
            )))
        );
    }

    #[test]
    fn validate_rejects_out_of_range_eqn_output_without_panicking() {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
        let out = b.binary(BinOp::Mul, x, x).unwrap();
        let mut g = b.finish(out).unwrap().into_draft();
        // Corrupt an equation's output id to point past the value table. validate() must return an Err, not panic on
        // the out-of-bounds `self.values[eqn.out]` index.
        let bad = ValueId::from_index(g.values().len() + 5).unwrap();
        g.eqns_mut().last_mut().unwrap().out = bad;
        assert_eq!(
            crate::test_support::verdict(&g),
            Err(GraphValidationError::EquationOutputOutOfRange {
                equation: 0,
                operation: "mul".into(),
                value: bad,
                value_count: g.values().len(),
            })
        );
    }

    #[test]
    fn validate_rejects_a_duplicate_ssa_producer() {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
        let internal = b.binary(BinOp::Mul, x, x).unwrap();
        let out = b.binary(BinOp::Mul, internal, internal).unwrap();
        let mut g = b.finish(out).unwrap().into_draft();
        let first = g.eqns()[0].clone();
        g.eqns_mut().push(first);

        assert!(matches!(
            crate::test_support::verdict(&g),
            Err(GraphValidationError::EquationOutputRedefined {
                equation: 2,
                value,
                ..
            }) if value == internal.id()
        ));
    }

    #[test]
    fn validate_rejects_state_const_missing_from_input_binders() {
        let b = Builder::for_tests();
        let state = b
            .state_input(
                "state",
                TensorType::new(vec![2, 5], DType::E4M3FN),
                StateRole::Recurrent,
            )
            .unwrap();
        let replacement = b
            .constant("replacement", TensorType::new(vec![2, 5], DType::E4M3FN))
            .unwrap();
        let mut g = b
            .finish(crate::Outputs::new(replacement).with_state(&[(state, replacement)]))
            .unwrap()
            .into_draft();
        g.inputs_mut().retain(|&id| id != state.id());

        assert_eq!(
            crate::test_support::verdict(&g),
            Err(GraphValidationError::ConstNotInput { value: state.id() })
        );
    }

    #[test]
    fn validate_preserves_literal_first_binary_contract() {
        let b = Builder::for_tests();
        let float = b.constant("float", TensorType::f32(vec![2])).unwrap();
        let out = b
            .binary_scalar(BinOp::Add, float, Scalar::F32(1.0))
            .unwrap();
        let mut g = b.finish(out).unwrap().into_draft();
        g.eqns_mut()[0].inputs.swap(0, 1);
        // Card 149 deliberately makes literal-first Binary graph-valid. Consumers with a narrower operand
        // grammar must reject it locally without changing Graph's general validity contract.
        g.values_mut()[out.id()].aval = OpKind::Binary(BinOp::Add)
            .infer(&[Scalar::F32(1.0).ty(), TensorType::f32(vec![2])])
            .unwrap();
        crate::test_support::verdict(&g).expect("literal-first Binary remains graph-valid");
    }

    #[test]
    fn validate_rejects_mismatched_binary_value_operands_including_scalars() {
        let b = Builder::for_tests();
        let float = b.constant("float", TensorType::scalar(DType::F32)).unwrap();
        let int = b.constant("int", TensorType::scalar(DType::I32)).unwrap();
        let out = b
            .binary_scalar(BinOp::Add, float, Scalar::F32(1.0))
            .unwrap();
        let mut g = b.finish(out).unwrap().into_draft();
        g.eqns_mut()[0].inputs = vec![Operand::Value(float.id()), Operand::Value(int.id())];

        assert_eq!(
            crate::test_support::verdict(&g),
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
            let b = Builder::for_tests();
            let float = b.constant("float", TensorType::f32(vec![2])).unwrap();
            let packed = b
                .constant("packed", TensorType::new(vec![2], dtype))
                .unwrap();
            let out = b
                .binary_scalar(BinOp::Add, float, Scalar::F32(1.0))
                .unwrap();
            let mut g = b.finish(out).unwrap().into_draft();
            g.eqns_mut()[0].inputs = vec![Operand::Value(packed.id()), Operand::Value(packed.id())];

            assert_eq!(
                crate::test_support::verdict(&g),
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
            let b = Builder::for_tests();
            let t = b.slot(Slot::Token, TensorType::scalar(DType::I32)).unwrap();
            b.finish(t).unwrap()
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

        let b = Builder::for_tests();
        let p = b.slot(Slot::Pos, TensorType::scalar(DType::I32)).unwrap();
        let g3 = b.finish(p).unwrap();
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
        let b = Builder::for_tests();
        let layer0 = b
            .slot_named(Slot::ExpertPoolMap, "layer_0", TensorType::f32(vec![4]))
            .unwrap();
        let layer1 = b
            .slot_named(Slot::ExpertPoolMap, "layer_1", TensorType::f32(vec![4]))
            .unwrap();
        let out = b.binary(BinOp::Add, layer0, layer0).unwrap();
        let g = b.finish(out).unwrap();

        let key0 = g.input_key(layer0.id()).expect("declared slot input");
        let key1 = g.input_key(layer1.id()).expect("declared slot input");
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
        let b = Builder::for_tests();
        let layer0 = b
            .slot_named(Slot::ExpertPoolMap, "layer_0", TensorType::f32(vec![4]))
            .unwrap();
        let mut g = b.finish(layer0).unwrap().into_draft();
        let original = g.input_key(layer0.id()).expect("declared slot input");

        // Corrupt the wire name a parser would read; the typed key lives elsewhere and must be
        // unaffected.
        g.values_mut()[layer0.id()].name = Some("not.a.recoverable.name".to_string());
        let after = g.input_key(layer0.id());

        assert_eq!(
            after,
            Some(original),
            "input_key must come from the value's stored key, never by parsing `name`"
        );
    }
}
