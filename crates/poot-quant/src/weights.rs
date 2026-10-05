//! `WeightStore`: the one load output every checkpoint reader produces (ADR-0103 decisions 1 and
//! 3, Card 540a). A store owns each tensor's bytes exactly as the checkpoint stored them - never
//! widened to f32 - keyed by the name the reader assigned it.
//!
//! Two kinds of entry, in two different type spaces:
//!
//! - [`DenseWeight`]: one contiguous [`StoredBytes`] buffer for one *stored tensor*, named by its
//!   [`poot_tensor::DType`] (the checkpoint's own element dtype: `F32`, `F16`, `BF16`, `E4M3FN`, `I64`,
//!   ...) and logical shape. `DType` is not [`format::WeightFormat`]: a `WeightFormat` names
//!   a quantization scheme (and every scheme but the three plain floats needs a `Scale` operand),
//!   while a stored tensor may be any dtype a checkpoint header names, including ones with no
//!   quantization meaning at all (`I64` index buffers, a standalone `F8_E4M3` tensor with no
//!   paired scale). The store checks once, at construction, that the buffer's length matches the
//!   logical shape times the dtype's element width; nothing downstream re-derives it. A dense
//!   entry's own numeric interpretation (is this `F8_E4M3` tensor a weight, and against what
//!   scale?) stays with its consumer; the store only proves the bytes are the right length.
//! - [`WeightEntry::Packed`]: an [`Arc<PackedPayload>`] (Card 627), the mixed loader's
//!   authoritative source bytes for one packed weight (one [`StoredBytes`] per
//!   [`PackedWeight::sources`] role, built by range reads with streamed upload), named by its
//!   [`format::WeightFormat`] quantization scheme.
//!
//! [`StoredBytes`] is the one byte-ownership and content-identity type both entries build on: a
//! dense tensor's whole buffer, or one packed source's buffer, own their bytes and memoize their
//! fingerprint the same way.
//!
//! Beside the store sits the [`WeightMap`] (Card 562a): a model's table from a typed
//! [`WeightId`] (`(layer, role)`) to a [`WeightView`] over store entries and the [`WeightHandle`] a
//! tracer sees (format and logical shape, no bytes). It lives here, not in `poot-models`, so the
//! executor contract and `poot-graph-plan` read one weight identity without depending on the model
//! families. [`WeightId::const_name`] is the one graph const name of a mapped weight.

use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::collections::btree_map;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{PackedPayload, StoredBytes};

/// The name a loader assigned one stored weight: the checkpoint tensor name for a dense entry, or
/// the loader's linear id (built from that weight's several source tensors - `weight` and `scale`
/// today) for a packed entry. Cheap to clone; two keys with equal string content are equal and
/// order the same, so a [`WeightStore`] can use this as a map key and look entries up by `&str`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WeightKey(Arc<str>);

impl WeightKey {
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WeightKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for WeightKey {
    fn from(name: &str) -> Self {
        Self(Arc::from(name))
    }
}

impl From<String> for WeightKey {
    fn from(name: String) -> Self {
        Self(Arc::from(name))
    }
}

impl From<Arc<str>> for WeightKey {
    fn from(name: Arc<str>) -> Self {
        Self(name)
    }
}

impl Borrow<str> for WeightKey {
    fn borrow(&self) -> &str {
        &self.0
    }
}

/// The identity of one payload a [`WeightStore`] owns: assigned when [`WeightStoreBuilder::insert`]
/// takes the entry, never reused within the process, and carried unchanged when another store
/// shares that entry ([`WeightStoreBuilder::extend_from`]). Two entries with the same name and shape
/// in different stores (a base and an adapter that replaces one weight) have different generations;
/// one entry shared by two stores has one. It is an explicit counter, not an allocation address, so a
/// freed payload's identity is never handed to a later one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StoreGeneration(u64);

impl StoreGeneration {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// One store slot: an entry and the generation it was inserted under.
#[derive(Clone, Debug)]
struct Owned {
    generation: StoreGeneration,
    entry: WeightEntry,
}

/// Typed failures building a [`DenseWeight`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DenseWeightError {
    /// `shape`'s element count, or that count times `dtype`'s stored element width, overflows
    /// `usize`.
    ShapeOverflow {
        dtype: poot_tensor::DType,
        shape: Vec<usize>,
    },
    /// `bytes.len()` does not equal `shape`'s element count times `dtype`'s stored element width.
    ByteLength {
        dtype: poot_tensor::DType,
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for DenseWeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow { dtype, shape } => {
                write!(
                    f,
                    "{dtype:?} shape {shape:?} overflows usize byte accounting"
                )
            }
            Self::ByteLength {
                dtype,
                shape,
                expected,
                actual,
            } => write!(
                f,
                "{dtype:?} shape {shape:?} needs {expected} stored bytes, got {actual}"
            ),
        }
    }
}

impl std::error::Error for DenseWeightError {}

/// One stored tensor's bytes, exactly as the checkpoint stored them (ADR-0103 decision 1: never
/// widened), named by its [`poot_tensor::DType`] and logical shape. Any stored dtype is accepted:
/// a dense entry is a fact about how the checkpoint stored these bytes, not about whether or how
/// they quantize a weight (that reading, if any, is the consumer's).
#[derive(Clone, Debug, PartialEq)]
pub struct DenseWeight {
    dtype: poot_tensor::DType,
    shape: Vec<usize>,
    bytes: StoredBytes,
}

impl DenseWeight {
    /// Checks that `bytes.len()` is exactly `shape`'s element count times `dtype`'s stored element
    /// width. A loader that reads the wrong range gets a typed error at construction rather than a
    /// store that silently reports the wrong footprint (card 540a SC-002).
    pub fn try_new(
        dtype: poot_tensor::DType,
        shape: Vec<usize>,
        bytes: Arc<[u8]>,
    ) -> Result<Self, DenseWeightError> {
        let element_bytes = dtype.byte_size();
        let overflow = || DenseWeightError::ShapeOverflow {
            dtype,
            shape: shape.clone(),
        };
        let elements = shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, extent| acc.checked_mul(extent))
            .ok_or_else(overflow)?;
        let expected = elements.checked_mul(element_bytes).ok_or_else(overflow)?;
        if bytes.len() != expected {
            return Err(DenseWeightError::ByteLength {
                dtype,
                shape,
                expected,
                actual: bytes.len(),
            });
        }
        Ok(Self {
            dtype,
            shape,
            bytes: StoredBytes::new(bytes),
        })
    }

    pub fn dtype(&self) -> poot_tensor::DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn bytes(&self) -> &StoredBytes {
        &self.bytes
    }
}

/// One stored weight: a dense stored tensor ([`poot_tensor::DType`]-named), or a packed weight's
/// authoritative source bytes ([`PackedPayload`], [`format::WeightFormat`]-named). The two live in
/// different type spaces (see the module docs), so a consumer that needs the dtype or the
/// quantization scheme matches the variant; [`Self::shape`] and [`Self::stored_bytes`] are the two
/// facts every entry reports uniformly.
#[derive(Clone, Debug, PartialEq)]
pub enum WeightEntry {
    Dense(DenseWeight),
    Packed(Arc<PackedPayload>),
}

impl WeightEntry {
    /// This weight's logical shape: the full shape for a dense entry, `[out, K]` for a packed one.
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Self::Dense(dense) => dense.shape().to_vec(),
            Self::Packed(payload) => payload.weight().shape().to_vec(),
        }
    }

    /// The stored byte footprint: a dense entry's one buffer length, or a packed weight's total
    /// source bytes (every [`PackedWeight::sources`] role summed). Always the stored footprint,
    /// never a widened decode (ADR-0103 decision 1, card 540a SC-001/SC-002).
    pub fn stored_bytes(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.bytes().len(),
            Self::Packed(payload) => payload.weight().total_source_bytes(),
        }
    }
}

/// [`WeightStoreBuilder::insert`] found a key already present.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DuplicateWeightKey(pub WeightKey);

impl fmt::Display for DuplicateWeightKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "duplicate weight key: {}", self.0)
    }
}

impl std::error::Error for DuplicateWeightKey {}

/// Builds a [`WeightStore`] entry by entry, rejecting a duplicate key with a typed error instead of
/// silently overwriting a loader's earlier read.
#[derive(Debug, Default)]
pub struct WeightStoreBuilder {
    entries: BTreeMap<WeightKey, Owned>,
}

impl WeightStoreBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts one entry. Fails if `key` is already present; a loader assigns each tensor (or each
    /// packed weight's linear id) one name, so a collision is a loader bug, not a merge.
    pub fn insert(
        &mut self,
        key: impl Into<WeightKey>,
        entry: WeightEntry,
    ) -> Result<(), DuplicateWeightKey> {
        match self.entries.entry(key.into()) {
            btree_map::Entry::Occupied(occupied) => Err(DuplicateWeightKey(occupied.key().clone())),
            btree_map::Entry::Vacant(vacant) => {
                vacant.insert(Owned {
                    generation: StoreGeneration::next(),
                    entry,
                });
                Ok(())
            }
        }
    }

    /// Shares every entry of `base` into this store under its own key, keeping each entry's
    /// existing generation: the entries are the same payloads, so an executor that holds both
    /// stores uploads each once. Fails on the first key already present.
    pub fn extend_from(&mut self, base: &WeightStore) -> Result<(), DuplicateWeightKey> {
        for (key, owned) in &base.entries {
            match self.entries.entry(key.clone()) {
                btree_map::Entry::Occupied(occupied) => {
                    return Err(DuplicateWeightKey(occupied.key().clone()));
                }
                btree_map::Entry::Vacant(vacant) => {
                    vacant.insert(owned.clone());
                }
            }
        }
        Ok(())
    }

    pub fn build(self) -> WeightStore {
        WeightStore {
            entries: self.entries,
        }
    }
}

/// The one output of a checkpoint load (ADR-0103 decisions 1 and 3): every tensor the reader saw,
/// keyed by the name it assigned it, each holding its bytes exactly as the checkpoint stored them.
/// Built once through [`WeightStoreBuilder`], then read-only and shared: the design note for Cards
/// 546a/562/563a puts one `Arc<WeightStore>` in `ModelHandle`, shared by every driver.
#[derive(Clone, Debug, Default)]
pub struct WeightStore {
    entries: BTreeMap<WeightKey, Owned>,
}

/// Content equality: the same keys holding equal entries. Generations are identity, not content.
impl PartialEq for WeightStore {
    fn eq(&self, other: &Self) -> bool {
        self.entries.len() == other.entries.len()
            && self
                .entries
                .iter()
                .zip(&other.entries)
                .all(|((ka, a), (kb, b))| ka == kb && a.entry == b.entry)
    }
}

impl WeightStore {
    pub fn builder() -> WeightStoreBuilder {
        WeightStoreBuilder::new()
    }

    pub fn get(&self, key: &str) -> Option<&WeightEntry> {
        self.entries.get(key).map(|owned| &owned.entry)
    }

    /// The generation `key`'s entry was inserted under: the identity a residency cache keys the
    /// entry's bytes by.
    pub fn generation(&self, key: &str) -> Option<StoreGeneration> {
        self.entries.get(key).map(|owned| owned.generation)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &WeightKey> {
        self.entries.keys()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&WeightKey, &WeightEntry)> {
        self.entries.iter().map(|(key, owned)| (key, &owned.entry))
    }

    /// Removes and returns one entry, releasing the store's own copy of its bytes once the caller
    /// drops the returned value (its `StoredBytes`/`PackedPayload` is `Arc`-backed, so this frees
    /// the underlying allocation exactly when nothing else - including a materialized on-demand
    /// view the caller already took and is still holding - references it). A consumer that reads
    /// every entry exactly once (a checkpoint loader building a transform-baked weight map, e.g.)
    /// uses this instead of `get` so the store's resident footprint shrinks as it drains, rather
    /// than staying fully resident until the whole store is dropped (card 540b's M4 RSS gate).
    pub fn remove(&mut self, key: &str) -> Option<WeightEntry> {
        self.entries.remove(key).map(|owned| owned.entry)
    }

    /// Every entry's [`WeightEntry::stored_bytes`], summed: the store's whole stored footprint,
    /// wired into a loader's own byte accounting (the mixed loader's `MixedLoadReport`) instead of
    /// that reader keeping a second, possibly-disagreeing count.
    pub fn total_stored_bytes(&self) -> usize {
        self.entries
            .values()
            .map(|owned| owned.entry.stored_bytes())
            .sum()
    }
}

/// What a weight is for, independent of any checkpoint's tensor names. Families map
/// their checkpoint names onto these roles; a family card adds the sub-roles it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WeightRole {
    Embed,
    Head,
    FinalNorm,
    Attn(AttnRole),
    Ffn(FfnRole),
    Norm(NormRole),
    /// The bias of a model-level LayerNorm (the final norm's; a layer norm's bias is
    /// [`NormRole::AttnBias`] or [`NormRole::FfnBias`]).
    FinalNormBias,
    /// A LayerNorm over the embedded tokens, before the first layer (BLOOM), and its bias.
    EmbedNorm,
    EmbedNormBias,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttnRole {
    Q,
    K,
    V,
    O,
    QBias,
    KBias,
    VBias,
    QNorm,
    KNorm,
    /// The output projection's bias.
    OBias,
    /// A fused `[(heads + 2 * kv_heads) * head_dim, width]` QKV projection whose rows interleave
    /// per head (`q_h, k_h, v_h` for each head; BLOOM), and its bias. A family that has one reads
    /// it instead of `Q`/`K`/`V`.
    Qkv,
    QkvBias,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FfnRole {
    Gate,
    Up,
    Down,
    /// The biases of a plain MLP's two projections.
    UpBias,
    DownBias,
}

/// The per-layer norms around a block (the model-level output norm is [`WeightRole::FinalNorm`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NormRole {
    Attn,
    Ffn,
    /// The norms applied to a block's output before it joins the residual (a sandwich layer's
    /// post-attention and post-feed-forward norms).
    PostAttn,
    PostFfn,
    /// The biases of a LayerNorm's [`NormRole::Attn`] and [`NormRole::Ffn`] scales.
    AttnBias,
    FfnBias,
}

impl WeightRole {
    /// The role's stable spelling inside [`WeightId::const_name`]. Written out per role, never
    /// derived from `Debug`: the name enters kernel caches and plan summaries, so renaming a variant
    /// must not silently rename a graph const.
    pub const fn spelling(self) -> &'static str {
        match self {
            WeightRole::Embed => "embed",
            WeightRole::Head => "head",
            WeightRole::FinalNorm => "final_norm",
            WeightRole::FinalNormBias => "final_norm_bias",
            WeightRole::EmbedNorm => "embed_norm",
            WeightRole::EmbedNormBias => "embed_norm_bias",
            WeightRole::Attn(role) => match role {
                AttnRole::Q => "attn.q",
                AttnRole::K => "attn.k",
                AttnRole::V => "attn.v",
                AttnRole::O => "attn.o",
                AttnRole::QBias => "attn.q_bias",
                AttnRole::KBias => "attn.k_bias",
                AttnRole::VBias => "attn.v_bias",
                AttnRole::QNorm => "attn.q_norm",
                AttnRole::KNorm => "attn.k_norm",
                AttnRole::OBias => "attn.o_bias",
                AttnRole::Qkv => "attn.qkv",
                AttnRole::QkvBias => "attn.qkv_bias",
            },
            WeightRole::Ffn(role) => match role {
                FfnRole::Gate => "ffn.gate",
                FfnRole::Up => "ffn.up",
                FfnRole::Down => "ffn.down",
                FfnRole::UpBias => "ffn.up_bias",
                FfnRole::DownBias => "ffn.down_bias",
            },
            WeightRole::Norm(role) => match role {
                NormRole::Attn => "norm.attn",
                NormRole::Ffn => "norm.ffn",
                NormRole::PostAttn => "norm.post_attn",
                NormRole::PostFfn => "norm.post_ffn",
                NormRole::AttnBias => "norm.attn_bias",
                NormRole::FfnBias => "norm.ffn_bias",
            },
        }
    }
}

/// The typed identity of one model weight: its layer (`None` for a model-level weight) and role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WeightId {
    pub layer: Option<usize>,
    pub role: WeightRole,
}

impl WeightId {
    pub const fn model(role: WeightRole) -> Self {
        Self { layer: None, role }
    }

    pub const fn layer(layer: usize, role: WeightRole) -> Self {
        Self {
            layer: Some(layer),
            role,
        }
    }

    /// The graph const name of this weight: `w.l3.attn.q` for a layer weight, `w.embed` for a
    /// model-level one. Every tracer and every binder spells a mapped weight this way.
    pub fn const_name(&self) -> String {
        match self.layer {
            Some(layer) => format!("w.l{layer}.{}", self.role.spelling()),
            None => format!("w.{}", self.role.spelling()),
        }
    }
}

impl fmt::Display for WeightId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.const_name())
    }
}

/// How a mapped weight reads its bytes out of the store: byte operations on stored entries, never a
/// decode (ADR-0103: tensors stay as stored).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightView {
    /// One stored entry, as stored.
    Stored(WeightKey),
    /// Several entries stacked along the leading (row) axis, in order (split `gate`/`up`, expert
    /// stacks).
    RowStack(Vec<WeightKey>),
    /// A leading-axis row range of one entry (a slice of a fused `qkv`).
    RowRange {
        key: WeightKey,
        rows: std::ops::Range<usize>,
    },
}

/// The stored format of a mapped weight (the driver path's only source of weight
/// formats). A dense entry names its stored dtype; a packed one its quantized `[out, K]` weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandleFormat {
    Dense(poot_tensor::DType),
    Packed(crate::PackedWeight),
}

/// What a tracer may see of a weight: its stored format and logical shape. No bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightHandle {
    pub format: HandleFormat,
    pub shape: Vec<usize>,
}

impl WeightHandle {
    fn of(entry: &WeightEntry) -> Self {
        match entry {
            WeightEntry::Dense(dense) => Self {
                format: HandleFormat::Dense(dense.dtype()),
                shape: dense.shape().to_vec(),
            },
            WeightEntry::Packed(payload) => Self {
                format: HandleFormat::Packed(payload.weight()),
                shape: payload.weight().shape().to_vec(),
            },
        }
    }
}

/// Why a [`WeightMap`] entry could not be built or materialized. Raised when a family builds its
/// map, before any trace or compile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightMapError {
    /// The view names a key the store does not hold.
    Missing { id: WeightId, key: WeightKey },
    /// `id` was mapped twice.
    Duplicate { id: WeightId },
    /// [`WeightMap::materialize`] was asked for an `id` the map does not hold.
    Unmapped { id: WeightId },
    /// A [`WeightView::RowStack`] with no parts.
    EmptyStack { id: WeightId },
    /// A row view over a rank-0 dense entry: there is no row axis.
    NoRowAxis { id: WeightId, key: WeightKey },
    /// A [`WeightView::RowRange`] that is empty or reaches past the entry's rows.
    RowRangeOutOfBounds {
        id: WeightId,
        rows: std::ops::Range<usize>,
        available: usize,
    },
    /// [`WeightView::RowStack`] parts that do not share one format and one row shape.
    MixedParts {
        id: WeightId,
        first: Box<WeightHandle>,
        other: Box<WeightHandle>,
    },
    /// The packed payload refuses the row view (a planar format whose rows are not byte runs).
    Packed {
        id: WeightId,
        source: Box<crate::PackedWeightError>,
    },
}

impl fmt::Display for WeightMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { id, key } => write!(f, "weight {id}: the store has no entry {key}"),
            Self::Duplicate { id } => write!(f, "weight {id} is mapped twice"),
            Self::Unmapped { id } => write!(f, "weight {id} is not in the map"),
            Self::EmptyStack { id } => {
                write!(f, "weight {id}: a row stack needs at least one part")
            }
            Self::NoRowAxis { id, key } => {
                write!(f, "weight {id}: entry {key} is a scalar and has no rows")
            }
            Self::RowRangeOutOfBounds {
                id,
                rows,
                available,
            } => write!(
                f,
                "weight {id}: rows {rows:?} are not a non-empty range of the entry's {available} rows"
            ),
            Self::MixedParts { id, first, other } => write!(
                f,
                "weight {id}: row-stack parts must share one format and row shape: {first:?} vs {other:?}"
            ),
            Self::Packed { id, source } => write!(f, "weight {id}: {source}"),
        }
    }
}

impl std::error::Error for WeightMapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Packed { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

/// A model's weights: `WeightId` -> (store view, handle). Built once by a family from its name table
/// through [`WeightMapBuilder`], which checks every view against the store; owned by the model. It
/// holds no bytes: [`Self::materialize`] reads a view out of the store when a binder needs it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WeightMap {
    entries: BTreeMap<WeightId, (WeightView, WeightHandle)>,
}

impl WeightMap {
    pub fn builder(store: &WeightStore) -> WeightMapBuilder<'_> {
        WeightMapBuilder {
            store,
            map: WeightMap::default(),
        }
    }

    pub fn handle(&self, id: WeightId) -> Option<&WeightHandle> {
        self.entries.get(&id).map(|(_, handle)| handle)
    }

    pub fn view(&self, id: WeightId) -> Option<&WeightView> {
        self.entries.get(&id).map(|(view, _)| view)
    }

    pub fn iter(&self) -> impl Iterator<Item = (WeightId, &WeightView, &WeightHandle)> {
        self.entries
            .iter()
            .map(|(id, (view, handle))| (*id, view, handle))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `id`'s bytes as one store-shaped entry: the stored entry itself for
    /// [`WeightView::Stored`], or the stacked / sliced rows copied byte for byte (a packed block
    /// format's rows are whole runs of blocks, so no decode is involved). The result's handle is
    /// the map's handle for `id`.
    pub fn materialize(
        &self,
        id: WeightId,
        store: &WeightStore,
    ) -> Result<WeightEntry, WeightMapError> {
        let (view, _) = self
            .entries
            .get(&id)
            .ok_or(WeightMapError::Unmapped { id })?;
        materialize_view(id, view, store)
    }
}

/// Builds a [`WeightMap`] against one store, checking each view as it is added.
#[derive(Debug)]
pub struct WeightMapBuilder<'s> {
    store: &'s WeightStore,
    map: WeightMap,
}

impl WeightMapBuilder<'_> {
    /// Add `id` read through `view`. Fails if `id` is already mapped, the view names a missing
    /// entry, or the view cannot be cut from the stored entries.
    pub fn map(&mut self, id: WeightId, view: WeightView) -> Result<&mut Self, WeightMapError> {
        if self.map.entries.contains_key(&id) {
            return Err(WeightMapError::Duplicate { id });
        }
        let handle = view_handle(id, &view, self.store)?;
        self.map.entries.insert(id, (view, handle));
        Ok(self)
    }

    pub fn build(self) -> WeightMap {
        self.map
    }
}

fn stored<'s>(
    id: WeightId,
    key: &WeightKey,
    store: &'s WeightStore,
) -> Result<&'s WeightEntry, WeightMapError> {
    store
        .get(key.as_str())
        .ok_or_else(|| WeightMapError::Missing {
            id,
            key: key.clone(),
        })
}

fn row_count(id: WeightId, key: &WeightKey, entry: &WeightEntry) -> Result<usize, WeightMapError> {
    entry
        .shape()
        .first()
        .copied()
        .ok_or_else(|| WeightMapError::NoRowAxis {
            id,
            key: key.clone(),
        })
}

fn check_row_range(
    id: WeightId,
    key: &WeightKey,
    entry: &WeightEntry,
    rows: &std::ops::Range<usize>,
) -> Result<(), WeightMapError> {
    let available = row_count(id, key, entry)?;
    if rows.is_empty() || rows.end > available {
        return Err(WeightMapError::RowRangeOutOfBounds {
            id,
            rows: rows.clone(),
            available,
        });
    }
    Ok(())
}

fn packed_error(id: WeightId) -> impl FnOnce(crate::PackedWeightError) -> WeightMapError {
    move |source| WeightMapError::Packed {
        id,
        source: Box::new(source),
    }
}

/// The handle `view` materializes into, checked against the store without copying a dense byte.
/// A packed view is checked by the payload operation [`materialize_view`] runs, so the map and the
/// bytes follow one rule.
fn view_handle(
    id: WeightId,
    view: &WeightView,
    store: &WeightStore,
) -> Result<WeightHandle, WeightMapError> {
    match view {
        WeightView::Stored(key) => Ok(WeightHandle::of(stored(id, key, store)?)),
        WeightView::RowRange { key, rows } => {
            let entry = stored(id, key, store)?;
            check_row_range(id, key, entry, rows)?;
            match entry {
                WeightEntry::Dense(dense) => {
                    let mut shape = dense.shape().to_vec();
                    shape[0] = rows.len();
                    Ok(WeightHandle {
                        format: HandleFormat::Dense(dense.dtype()),
                        shape,
                    })
                }
                WeightEntry::Packed(_) => Ok(WeightHandle::of(&materialize_view(id, view, store)?)),
            }
        }
        WeightView::RowStack(parts) => {
            let (first_key, rest) = parts
                .split_first()
                .ok_or(WeightMapError::EmptyStack { id })?;
            let first_entry = stored(id, first_key, store)?;
            let first = WeightHandle::of(first_entry);
            let mut rows = row_count(id, first_key, first_entry)?;
            for key in rest {
                let entry = stored(id, key, store)?;
                let other = WeightHandle::of(entry);
                let same_format = match (first.format, other.format) {
                    (HandleFormat::Dense(a), HandleFormat::Dense(b)) => a == b,
                    (HandleFormat::Packed(a), HandleFormat::Packed(b)) => a.format() == b.format(),
                    _ => false,
                };
                if !same_format || other.shape.get(1..) != first.shape.get(1..) {
                    return Err(WeightMapError::MixedParts {
                        id,
                        first: Box::new(first),
                        other: Box::new(other),
                    });
                }
                rows += row_count(id, key, entry)?;
            }
            match first.format {
                HandleFormat::Dense(dtype) => {
                    let mut shape = first.shape;
                    shape[0] = rows;
                    Ok(WeightHandle {
                        format: HandleFormat::Dense(dtype),
                        shape,
                    })
                }
                HandleFormat::Packed(_) => {
                    Ok(WeightHandle::of(&materialize_view(id, view, store)?))
                }
            }
        }
    }
}

fn materialize_view(
    id: WeightId,
    view: &WeightView,
    store: &WeightStore,
) -> Result<WeightEntry, WeightMapError> {
    match view {
        WeightView::Stored(key) => Ok(stored(id, key, store)?.clone()),
        WeightView::RowRange { key, rows } => match stored(id, key, store)? {
            entry @ WeightEntry::Dense(dense) => {
                check_row_range(id, key, entry, rows)?;
                let row_bytes = dense.bytes().len() / dense.shape()[0];
                let bytes = &dense.bytes().as_slice()[rows.start * row_bytes..rows.end * row_bytes];
                let mut shape = dense.shape().to_vec();
                shape[0] = rows.len();
                Ok(WeightEntry::Dense(DenseWeight {
                    dtype: dense.dtype(),
                    shape,
                    bytes: StoredBytes::new(Arc::from(bytes)),
                }))
            }
            entry @ WeightEntry::Packed(payload) => {
                check_row_range(id, key, entry, rows)?;
                let rows: Vec<usize> = rows.clone().collect();
                PackedPayload::gather_rows(&[payload.as_ref()], &rows)
                    .map(|sliced| WeightEntry::Packed(Arc::new(sliced)))
                    .map_err(packed_error(id))
            }
        },
        WeightView::RowStack(parts) => {
            let entries = parts
                .iter()
                .map(|key| {
                    let entry = stored(id, key, store)?;
                    row_count(id, key, entry)?;
                    Ok(entry)
                })
                .collect::<Result<Vec<_>, _>>()?;
            match entries.first() {
                None => Err(WeightMapError::EmptyStack { id }),
                Some(WeightEntry::Dense(first)) => {
                    let mut shape = first.shape().to_vec();
                    shape[0] = 0;
                    let mut bytes = Vec::new();
                    for entry in &entries {
                        let WeightEntry::Dense(dense) = entry else {
                            return Err(mixed(id, entries[0], entry));
                        };
                        if dense.dtype() != first.dtype() || dense.shape()[1..] != shape[1..] {
                            return Err(mixed(id, entries[0], entry));
                        }
                        shape[0] += dense.shape()[0];
                        bytes.extend_from_slice(dense.bytes().as_slice());
                    }
                    Ok(WeightEntry::Dense(DenseWeight {
                        dtype: first.dtype(),
                        shape,
                        bytes: StoredBytes::new(Arc::from(bytes)),
                    }))
                }
                Some(WeightEntry::Packed(_)) => {
                    let payloads = entries
                        .iter()
                        .map(|entry| match entry {
                            WeightEntry::Packed(payload) => Ok(payload.as_ref()),
                            WeightEntry::Dense(_) => Err(mixed(id, entries[0], entry)),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    PackedPayload::concat_rows(&payloads)
                        .map(|stacked| WeightEntry::Packed(Arc::new(stacked)))
                        .map_err(packed_error(id))
                }
            }
        }
    }
}

fn mixed(id: WeightId, first: &WeightEntry, other: &WeightEntry) -> WeightMapError {
    WeightMapError::MixedParts {
        id,
        first: Box::new(WeightHandle::of(first)),
        other: Box::new(WeightHandle::of(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::WeightFormat;
    use crate::{PackedWeight, SourceRole};
    use poot_tensor::DType;

    fn bytes(n: usize) -> Arc<[u8]> {
        vec![0u8; n].into()
    }

    #[test]
    fn dense_weight_accepts_every_stored_dtype_including_i64_and_standalone_e4m3() {
        // I64 and a standalone (unpaired) F8_E4M3 tensor are dense stored
        // tensors, not WeightFormat quantization schemes; DenseWeight must accept both.
        let f32 = DenseWeight::try_new(DType::F32, vec![2, 3], bytes(2 * 3 * 4)).unwrap();
        assert_eq!(f32.dtype(), DType::F32);
        assert_eq!(f32.shape(), &[2, 3]);
        assert_eq!(f32.bytes().len(), 24);

        let i64_indices = DenseWeight::try_new(DType::I64, vec![10], bytes(10 * 8)).unwrap();
        assert_eq!(i64_indices.dtype(), DType::I64);

        let standalone_e4m3 = DenseWeight::try_new(DType::E4M3FN, vec![4, 4], bytes(16)).unwrap();
        assert_eq!(standalone_e4m3.dtype(), DType::E4M3FN);
    }

    #[test]
    fn dense_weight_rejects_a_byte_length_mismatch() {
        // Guards SC-002: a caller handing widened f32 bytes for a bf16 tensor must be rejected
        // here, not silently accepted. Verified red by disabling the length check and green by
        // restoring it (poot-quant/src/weights.rs, worker-rules 7).
        let error = DenseWeight::try_new(DType::BF16, vec![2, 3], bytes(2 * 3 * 4)).unwrap_err();
        assert_eq!(
            error,
            DenseWeightError::ByteLength {
                dtype: DType::BF16,
                shape: vec![2, 3],
                expected: 12,
                actual: 24,
            }
        );
    }

    #[test]
    fn dense_weight_rejects_shape_overflow() {
        let error = DenseWeight::try_new(DType::F32, vec![usize::MAX, 2], bytes(1)).unwrap_err();
        assert!(matches!(error, DenseWeightError::ShapeOverflow { .. }));
    }

    fn packed_entry(shape: [usize; 2]) -> Arc<PackedPayload> {
        let weight = PackedWeight::try_new(WeightFormat::Q4_0, shape).unwrap();
        let block_bytes = weight.source_bytes(SourceRole::Blocks);
        Arc::new(
            PackedPayload::try_new(weight, [(SourceRole::Blocks, bytes(block_bytes))]).unwrap(),
        )
    }

    #[test]
    fn store_reports_its_entries_uniformly_across_dense_and_packed() {
        let mut builder = WeightStore::builder();
        builder
            .insert(
                "embed_tokens.weight",
                WeightEntry::Dense(
                    DenseWeight::try_new(DType::F32, vec![4, 8], bytes(4 * 8 * 4)).unwrap(),
                ),
            )
            .unwrap();
        builder
            .insert(
                "norm.weight",
                WeightEntry::Dense(
                    DenseWeight::try_new(DType::BF16, vec![8], bytes(8 * 2)).unwrap(),
                ),
            )
            .unwrap();
        builder
            .insert(
                "layers.0.mlp.down_proj",
                WeightEntry::Packed(packed_entry([1, 32])),
            )
            .unwrap();
        let store = builder.build();

        assert_eq!(store.len(), 3);
        assert!(!store.is_empty());
        assert!(store.contains("embed_tokens.weight"));
        assert!(!store.contains("missing"));

        let norm = store.get("norm.weight").unwrap();
        assert_eq!(norm.shape(), vec![8]);
        assert_eq!(norm.stored_bytes(), 16);

        let packed = store.get("layers.0.mlp.down_proj").unwrap();
        assert_eq!(packed.shape(), vec![1, 32]);
        assert_eq!(packed.stored_bytes(), 18); // Q4_0: one 18-byte block per 32 K values.

        // Guards SC-002: total_stored_bytes sums the stored footprint of every entry (128 + 16 +
        // 18 = 162), never a widened decode.
        assert_eq!(store.total_stored_bytes(), 4 * 8 * 4 + 8 * 2 + 18);
    }

    #[test]
    fn builder_rejects_a_duplicate_key() {
        let mut builder = WeightStore::builder();
        let entry =
            || WeightEntry::Dense(DenseWeight::try_new(DType::F32, vec![1], bytes(4)).unwrap());
        builder.insert("w", entry()).unwrap();
        let error = builder.insert("w", entry()).unwrap_err();
        assert_eq!(error, DuplicateWeightKey(WeightKey::from("w")));
    }

    #[test]
    fn weight_key_equal_content_is_equal_and_looks_up_by_str() {
        let a = WeightKey::from("layers.0.q_proj.weight");
        let b = WeightKey::from("layers.0.q_proj.weight".to_string());
        assert_eq!(a, b);
        assert_eq!(a.as_str(), "layers.0.q_proj.weight");

        let mut builder = WeightStore::builder();
        builder
            .insert(
                a.clone(),
                WeightEntry::Dense(DenseWeight::try_new(DType::F32, vec![1], bytes(4)).unwrap()),
            )
            .unwrap();
        let store = builder.build();
        assert!(store.get(a.as_str()).is_some());
    }

    #[test]
    fn const_names_are_the_stable_per_role_spellings() {
        let cases = [
            (WeightId::model(WeightRole::Embed), "w.embed"),
            (WeightId::model(WeightRole::Head), "w.head"),
            (WeightId::model(WeightRole::FinalNorm), "w.final_norm"),
            (
                WeightId::model(WeightRole::FinalNormBias),
                "w.final_norm_bias",
            ),
            (WeightId::model(WeightRole::EmbedNorm), "w.embed_norm"),
            (
                WeightId::model(WeightRole::EmbedNormBias),
                "w.embed_norm_bias",
            ),
            (
                WeightId::layer(1, WeightRole::Attn(AttnRole::OBias)),
                "w.l1.attn.o_bias",
            ),
            (
                WeightId::layer(1, WeightRole::Attn(AttnRole::Qkv)),
                "w.l1.attn.qkv",
            ),
            (
                WeightId::layer(1, WeightRole::Attn(AttnRole::QkvBias)),
                "w.l1.attn.qkv_bias",
            ),
            (
                WeightId::layer(1, WeightRole::Ffn(FfnRole::UpBias)),
                "w.l1.ffn.up_bias",
            ),
            (
                WeightId::layer(1, WeightRole::Ffn(FfnRole::DownBias)),
                "w.l1.ffn.down_bias",
            ),
            (
                WeightId::layer(1, WeightRole::Norm(NormRole::PostAttn)),
                "w.l1.norm.post_attn",
            ),
            (
                WeightId::layer(1, WeightRole::Norm(NormRole::PostFfn)),
                "w.l1.norm.post_ffn",
            ),
            (
                WeightId::layer(1, WeightRole::Norm(NormRole::AttnBias)),
                "w.l1.norm.attn_bias",
            ),
            (
                WeightId::layer(1, WeightRole::Norm(NormRole::FfnBias)),
                "w.l1.norm.ffn_bias",
            ),
            (
                WeightId::layer(3, WeightRole::Attn(AttnRole::Q)),
                "w.l3.attn.q",
            ),
            (
                WeightId::layer(0, WeightRole::Attn(AttnRole::K)),
                "w.l0.attn.k",
            ),
            (
                WeightId::layer(0, WeightRole::Attn(AttnRole::V)),
                "w.l0.attn.v",
            ),
            (
                WeightId::layer(0, WeightRole::Attn(AttnRole::O)),
                "w.l0.attn.o",
            ),
            (
                WeightId::layer(1, WeightRole::Attn(AttnRole::QBias)),
                "w.l1.attn.q_bias",
            ),
            (
                WeightId::layer(1, WeightRole::Attn(AttnRole::KBias)),
                "w.l1.attn.k_bias",
            ),
            (
                WeightId::layer(1, WeightRole::Attn(AttnRole::VBias)),
                "w.l1.attn.v_bias",
            ),
            (
                WeightId::layer(2, WeightRole::Attn(AttnRole::QNorm)),
                "w.l2.attn.q_norm",
            ),
            (
                WeightId::layer(2, WeightRole::Attn(AttnRole::KNorm)),
                "w.l2.attn.k_norm",
            ),
            (
                WeightId::layer(12, WeightRole::Ffn(FfnRole::Gate)),
                "w.l12.ffn.gate",
            ),
            (
                WeightId::layer(12, WeightRole::Ffn(FfnRole::Up)),
                "w.l12.ffn.up",
            ),
            (
                WeightId::layer(12, WeightRole::Ffn(FfnRole::Down)),
                "w.l12.ffn.down",
            ),
            (
                WeightId::layer(4, WeightRole::Norm(NormRole::Attn)),
                "w.l4.norm.attn",
            ),
            (
                WeightId::layer(4, WeightRole::Norm(NormRole::Ffn)),
                "w.l4.norm.ffn",
            ),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for (id, name) in cases {
            assert_eq!(id.const_name(), name);
            assert!(seen.insert(name), "{name} is spelled twice");
        }
    }

    /// `rows` x `cols` F32 whose element `[r, c]` is `100 r + c`, so every row's bytes differ.
    fn ramp(rows: usize, cols: usize, base: usize) -> WeightEntry {
        let values: Vec<u8> = (0..rows * cols)
            .flat_map(|i| (((base + i / cols) * 100 + i % cols) as f32).to_le_bytes())
            .collect();
        WeightEntry::Dense(
            DenseWeight::try_new(DType::F32, vec![rows, cols], values.into()).unwrap(),
        )
    }

    /// Q4_0 `[rows, 32]` (one 18-byte block per row): `d = 1.0` (f16 `0x3c00`) and every quant byte
    /// set to `base + row`, so each row's bytes are distinct and finite.
    fn q4_rows(rows: usize, base: u8) -> WeightEntry {
        let weight = PackedWeight::try_new(WeightFormat::Q4_0, [rows, 32]).unwrap();
        let bytes: Vec<u8> = (0..rows)
            .flat_map(|r| {
                let mut block = vec![0x00, 0x3c];
                block.extend(std::iter::repeat_n(base + r as u8, 16));
                block
            })
            .collect();
        WeightEntry::Packed(Arc::new(
            PackedPayload::try_new(weight, [(SourceRole::Blocks, bytes.into())]).unwrap(),
        ))
    }

    fn entry_bytes(entry: &WeightEntry) -> Vec<u8> {
        match entry {
            WeightEntry::Dense(dense) => dense.bytes().as_slice().to_vec(),
            WeightEntry::Packed(payload) => payload.bytes(SourceRole::Blocks).to_vec(),
        }
    }

    fn view_store() -> WeightStore {
        let mut b = WeightStore::builder();
        b.insert("qkv", ramp(6, 2, 0)).unwrap();
        b.insert("gate", ramp(2, 2, 10)).unwrap();
        b.insert("up", ramp(3, 2, 20)).unwrap();
        b.insert("wide", ramp(2, 3, 30)).unwrap();
        b.insert("pqkv", q4_rows(6, 0x10)).unwrap();
        b.insert("pgate", q4_rows(2, 0x40)).unwrap();
        b.insert("pup", q4_rows(1, 0x50)).unwrap();
        b.insert(
            "scalar",
            WeightEntry::Dense(DenseWeight::try_new(DType::F32, vec![], bytes(4)).unwrap()),
        )
        .unwrap();
        b.build()
    }

    const Q: WeightId = WeightId::layer(0, WeightRole::Attn(AttnRole::Q));
    const K: WeightId = WeightId::layer(0, WeightRole::Attn(AttnRole::K));
    const GATE_UP: WeightId = WeightId::layer(0, WeightRole::Ffn(FfnRole::Gate));

    #[test]
    fn dense_row_views_materialize_the_stored_rows_byte_for_byte() {
        let store = view_store();
        let mut b = WeightMap::builder(&store);
        b.map(
            K,
            WeightView::RowRange {
                key: "qkv".into(),
                rows: 2..4,
            },
        )
        .unwrap();
        b.map(
            GATE_UP,
            WeightView::RowStack(vec!["gate".into(), "up".into()]),
        )
        .unwrap();
        b.map(Q, WeightView::Stored("qkv".into())).unwrap();
        let map = b.build();

        let k = map.materialize(K, &store).unwrap();
        assert_eq!(WeightHandle::of(&k), *map.handle(K).unwrap());
        assert_eq!(map.handle(K).unwrap().shape, vec![2, 2]);
        assert_eq!(
            entry_bytes(&k),
            entry_bytes(&ramp(2, 2, 2)),
            "rows 2..4 of qkv"
        );

        let gate_up = map.materialize(GATE_UP, &store).unwrap();
        assert_eq!(map.handle(GATE_UP).unwrap().shape, vec![5, 2]);
        assert_eq!(WeightHandle::of(&gate_up), *map.handle(GATE_UP).unwrap());
        let mut want = entry_bytes(&ramp(2, 2, 10));
        want.extend(entry_bytes(&ramp(3, 2, 20)));
        assert_eq!(entry_bytes(&gate_up), want);

        assert_eq!(
            map.materialize(Q, &store).unwrap(),
            *store.get("qkv").unwrap()
        );
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn packed_row_views_cut_whole_blocks_without_a_decode() {
        let store = view_store();
        let mut b = WeightMap::builder(&store);
        b.map(
            K,
            WeightView::RowRange {
                key: "pqkv".into(),
                rows: 2..5,
            },
        )
        .unwrap();
        b.map(
            GATE_UP,
            WeightView::RowStack(vec!["pgate".into(), "pup".into()]),
        )
        .unwrap();
        let map = b.build();

        let k = map.materialize(K, &store).unwrap();
        let handle = map.handle(K).unwrap();
        assert_eq!(WeightHandle::of(&k), *handle);
        assert_eq!(
            handle.format,
            HandleFormat::Packed(PackedWeight::try_new(WeightFormat::Q4_0, [3, 32]).unwrap())
        );
        assert_eq!(
            entry_bytes(&k),
            entry_bytes(&q4_rows(3, 0x12)),
            "rows 2..5 of pqkv"
        );

        let gate_up = map.materialize(GATE_UP, &store).unwrap();
        assert_eq!(WeightHandle::of(&gate_up), *map.handle(GATE_UP).unwrap());
        assert_eq!(map.handle(GATE_UP).unwrap().shape, vec![3, 32]);
        let mut want = entry_bytes(&q4_rows(2, 0x40));
        want.extend(entry_bytes(&q4_rows(1, 0x50)));
        assert_eq!(entry_bytes(&gate_up), want);
    }

    #[test]
    fn invalid_views_are_typed_refusals_at_map_time() {
        let store = view_store();
        let mut b = WeightMap::builder(&store);
        let err = |b: &mut WeightMapBuilder<'_>, view| b.map(Q, view).map(|_| ()).unwrap_err();

        assert_eq!(
            err(&mut b, WeightView::Stored("absent".into())),
            WeightMapError::Missing {
                id: Q,
                key: "absent".into()
            }
        );
        assert_eq!(
            err(
                &mut b,
                WeightView::RowRange {
                    key: "qkv".into(),
                    rows: 4..7
                }
            ),
            WeightMapError::RowRangeOutOfBounds {
                id: Q,
                rows: 4..7,
                available: 6
            }
        );
        assert!(matches!(
            err(
                &mut b,
                WeightView::RowRange {
                    key: "qkv".into(),
                    rows: 3..3
                }
            ),
            WeightMapError::RowRangeOutOfBounds { .. }
        ));
        assert_eq!(
            err(
                &mut b,
                WeightView::RowRange {
                    key: "scalar".into(),
                    rows: 0..1
                }
            ),
            WeightMapError::NoRowAxis {
                id: Q,
                key: "scalar".into()
            }
        );
        assert_eq!(
            err(&mut b, WeightView::RowStack(Vec::new())),
            WeightMapError::EmptyStack { id: Q }
        );
        for parts in [["gate", "wide"], ["gate", "pgate"]] {
            let view = WeightView::RowStack(parts.iter().map(|&p| p.into()).collect());
            assert!(
                matches!(err(&mut b, view), WeightMapError::MixedParts { .. }),
                "{parts:?}"
            );
        }

        b.map(Q, WeightView::Stored("qkv".into())).unwrap();
        assert_eq!(
            err(&mut b, WeightView::Stored("gate".into())),
            WeightMapError::Duplicate { id: Q }
        );
        let map = b.build();
        assert_eq!(map.len(), 1, "refused views leave no entry");
        assert_eq!(
            map.materialize(K, &store),
            Err(WeightMapError::Unmapped { id: K })
        );
    }

    #[test]
    fn materialize_rechecks_the_view_against_the_store_it_is_given() {
        let store = view_store();
        let mut b = WeightMap::builder(&store);
        b.map(
            K,
            WeightView::RowRange {
                key: "qkv".into(),
                rows: 4..6,
            },
        )
        .unwrap();
        let map = b.build();
        let mut shorter = WeightStore::builder();
        shorter.insert("qkv", ramp(3, 2, 0)).unwrap();
        assert_eq!(
            map.materialize(K, &shorter.build()),
            Err(WeightMapError::RowRangeOutOfBounds {
                id: K,
                rows: 4..6,
                available: 3
            })
        );
    }
}
