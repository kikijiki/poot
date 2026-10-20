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
//!   [`PackedWeight::sources`] role, a view of the checkpoint's file mapping), named by its
//!   [`format::WeightFormat`] quantization scheme.
//!
//! [`StoredBytes`] is the one byte-ownership type both entries build on: a dense tensor's whole
//! buffer, or one packed source's buffer, is a view of a file mapping (or an owned region for bytes a
//! loader derives or a fixture builds), checked for in-place change with [`WeightStore::check_sources`].
//!
//! Beside the store sits the [`WeightMap`] (Card 562a): a model's table from a typed
//! [`WeightId`] (`(layer, role)`) to a [`WeightView`] over store entries and the [`WeightHandle`] a
//! tracer sees (format and logical shape, no bytes). It lives here, not in `poot-models`, so the
//! executor contract and `poot-graph-plan` read one weight identity without depending on the model
//! families. [`WeightId::const_name`] is the one graph const name of a mapped weight.

use std::borrow::{Borrow, Cow};
use std::collections::btree_map;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(any(test, feature = "test-support"))]
use crate::PackedStack;
#[cfg(any(test, feature = "test-support"))]
use crate::SourceRole;
use crate::{PackedPayload, PackedWeight, SourceChanged, StoredBytes};

/// The name a loader assigned one stored weight: the checkpoint tensor name for a dense entry, or
/// the loader's linear id (built from that weight's several source tensors - `weight` and `scale`
/// today) for a packed entry. Cheap to clone; two keys with equal string content are equal and
/// order the same, so a [`WeightStore`] can use this as a map key and look entries up by `&str`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WeightKey(Arc<str>);

impl WeightKey {
    /// A key for `name`; an `Arc<str>` is shared, not copied.
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self(name.into())
    }

    /// The name as a string slice.
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
        /// The stored element dtype.
        dtype: poot_tensor::DType,
        /// The logical shape whose element count or byte size overflowed.
        shape: Vec<usize>,
    },
    /// `bytes.len()` does not equal `shape`'s element count times `dtype`'s stored element width.
    ByteLength {
        /// The stored element dtype.
        dtype: poot_tensor::DType,
        /// The logical shape the buffer was supposed to fill.
        shape: Vec<usize>,
        /// Byte length `shape` and `dtype` require.
        expected: usize,
        /// Length of the buffer that was supplied.
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
        bytes: StoredBytes,
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
            bytes,
        })
    }

    /// The checkpoint's own element dtype of this tensor.
    pub fn dtype(&self) -> poot_tensor::DType {
        self.dtype
    }

    /// The logical shape, one extent per axis (empty for a scalar).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The tensor's bytes as the checkpoint stored them; their length is `shape`'s element count times `dtype`'s element width.
    pub fn bytes(&self) -> &StoredBytes {
        &self.bytes
    }
}

/// A dense tensor's stored dtype has no f32 decode (an integer or boolean tensor, or f64).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UndecodableDtype {
    /// The stored dtype that has no f32 decode.
    pub dtype: poot_tensor::DType,
}

impl fmt::Display for UndecodableDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stored dtype {:?} has no f32 decode", self.dtype)
    }
}

impl std::error::Error for UndecodableDtype {}

impl DenseWeight {
    /// This tensor's stored bytes widened to row-major f32 (the CPU oracle's payload): F32, BF16, F16,
    /// the unscaled E4M3FN codes and E8M0 scale bytes decode through [`crate::scalar`]; any other stored
    /// dtype is a typed error. A quantized linear's codes and scale are packed together as a
    /// [`PackedPayload`] and decoded by the format descriptor, never here.
    pub fn decode_f32(&self) -> Result<Vec<f32>, UndecodableDtype> {
        use crate::scalar;
        use poot_tensor::DType;

        let raw = self.bytes.as_slice();
        match self.dtype {
            DType::F32 => Ok(raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect()),
            DType::BF16 => Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|word| scalar::bf16_to_f32(u16::from_le_bytes(*word)))
                .collect()),
            DType::F16 => Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|word| scalar::f16_to_f32(u16::from_le_bytes(*word)))
                .collect()),
            DType::E4M3FN => Ok(raw.iter().map(|&b| scalar::e4m3fn_to_f32(b)).collect()),
            DType::E8M0 => Ok(raw.iter().map(|&b| scalar::e8m0_to_f32(b)).collect()),
            dtype => Err(UndecodableDtype { dtype }),
        }
    }
}

/// One stored weight: a dense stored tensor ([`poot_tensor::DType`]-named), or a packed weight's
/// authoritative source bytes ([`PackedPayload`], [`format::WeightFormat`]-named). The two live in
/// different type spaces (see the module docs), so a consumer that needs the dtype or the
/// quantization scheme matches the variant; [`Self::shape`] and [`Self::stored_bytes`] are the two
/// facts every entry reports uniformly.
#[derive(Clone, Debug, PartialEq)]
pub enum WeightEntry {
    /// A stored tensor named by its checkpoint dtype.
    Dense(DenseWeight),
    /// A packed weight's source bytes, shared by `Arc` with whoever else reads the payload.
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
    /// An empty builder.
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

    /// Freezes the entries inserted so far into a read-only store.
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
    /// An empty [`WeightStoreBuilder`].
    pub fn builder() -> WeightStoreBuilder {
        WeightStoreBuilder::new()
    }

    /// The entry stored under `key`, if any.
    pub fn get(&self, key: &str) -> Option<&WeightEntry> {
        self.entries.get(key).map(|owned| &owned.entry)
    }

    /// The generation `key`'s entry was inserted under: the identity a residency cache keys the
    /// entry's bytes by.
    pub fn generation(&self, key: &str) -> Option<StoreGeneration> {
        self.entries.get(key).map(|owned| owned.generation)
    }

    /// Whether the store holds an entry under `key`.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every key, in key order.
    pub fn keys(&self) -> impl Iterator<Item = &WeightKey> {
        self.entries.keys()
    }

    /// Every `(key, entry)` pair, in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&WeightKey, &WeightEntry)> {
        self.entries.iter().map(|(key, owned)| (key, &owned.entry))
    }

    /// [`StoredBytes::check_unchanged`] once per distinct region of the store's bytes (one `fstat` per
    /// mapped checkpoint file, never per tensor): `Ok` when no file mapped under this store changed
    /// since it was mapped. A loader runs it before publishing the store and an executor before a bind
    /// batch (ADR-0103 amendment: the identity check at its named boundaries).
    pub fn check_sources(&self) -> Result<(), SourceChanged> {
        let mut checked: Vec<&StoredBytes> = Vec::new();
        for owned in self.entries.values() {
            let views: Vec<&StoredBytes> = match &owned.entry {
                WeightEntry::Dense(dense) => vec![dense.bytes()],
                WeightEntry::Packed(payload) => payload.stored().collect(),
            };
            for view in views {
                // Owned regions never change; only a mapped region is checked, once per file.
                if view.is_mapped() && !checked.iter().any(|seen| seen.same_region(view)) {
                    view.check_unchanged()?;
                    checked.push(view);
                }
            }
        }
        Ok(())
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
    #[cfg(any(test, feature = "test-support"))]
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
    /// The token embedding table.
    Embed,
    /// The output head: the projection from the final hidden state to the output logits.
    Head,
    /// The norm scale applied after the last layer, before the head.
    FinalNorm,
    /// A projection, bias or norm of a layer's attention.
    Attn(AttnRole),
    /// A projection or bias of a layer's dense feed-forward.
    Ffn(FfnRole),
    /// A per-layer norm scale or bias.
    Norm(NormRole),
    /// A weight of a layer's routed mixture-of-experts block.
    Moe(MoeRole),
    /// The multi-head latent attention projections beside [`AttnRole::Q`] and [`AttnRole::O`].
    Mla(MlaRole),
    /// A Mamba2 mixer's weights.
    Mamba2(Mamba2Role),
    /// A Gated DeltaNet mixer's weights.
    Gdn(GdnRole),
    /// A Kimi Delta Attention (KDA) mixer's weights.
    Kda(KdaRole),
    /// The DeepSeek sparse-attention lightning indexer's weights.
    Dsa(DsaRole),
    /// A multi-token-prediction layer's input fusion and closing norm (GLM's `eh_proj`, `enorm`,
    /// `hnorm` and `shared_head.norm`); the layer's own block reads the ordinary roles.
    Mtp(MtpRole),
    /// The block indexer of Qwen Sparse Attention.
    Qsa(QsaRole),
    /// One Hyper-Connections gate around a block (Qwen4Exp's `*_hyper_connection` modules), and the
    /// model-level mixer that collapses the streams before the head.
    Hyper(HyperSite, HyperRole),
    /// The per-layer-embedding (n-gram) block's weights.
    Ple(PleRole),
    /// DeepSeek-V4's manifold-constrained hyper-connection (mHC) mixing weights.
    Mhc(MhcRole),
    /// DeepSeek-V4's attention-window compressors (the layer's and its indexer's).
    Compress(CompressRole),
    /// The bias of a model-level LayerNorm (the final norm's; a layer norm's bias is
    /// [`NormRole::AttnBias`] or [`NormRole::FfnBias`]).
    FinalNormBias,
    /// A LayerNorm over the embedded tokens, before the first layer (BLOOM), and its bias.
    EmbedNorm,
    /// The bias of [`WeightRole::EmbedNorm`].
    EmbedNormBias,
    /// The scalar a layer multiplies its output by (Gemma 4's `layer_scalar`), `[1]`.
    LayerScale,
    /// A learned absolute position table `[positions, width]` (BERT, a vision tower), and BERT's
    /// token-type table `[types, width]`.
    PosEmbed,
    /// BERT's token-type table `[types, width]`.
    TypeEmbed,
    /// BERT's `[CLS]` pooler projection `[width, width]` and its bias.
    Pooler,
    /// The bias of [`WeightRole::Pooler`].
    PoolerBias,
    /// The bias of the output head (a classifier's), `[out]`.
    HeadBias,
    /// A vision tower's patch embedding, the stored convolution `[width, channels, patch, patch]`,
    /// and its bias.
    Patch,
    /// The bias of [`WeightRole::Patch`].
    PatchBias,
    /// A modality connector's projection `[text_width, vision_width * scale^2]`.
    Connector,
    /// A two-layer connector's first projection's bias, and its second projection and bias (a
    /// Qwen2.5-VL patch merger's `mlp.0` bias, `mlp.2` and its bias).
    ConnectorBias,
    /// The second projection of a two-layer connector.
    ConnectorOut,
    /// The bias of [`WeightRole::ConnectorOut`].
    ConnectorOutBias,
    /// The stacked LoRA adapter tensors the compile-time LoRA pass reads beside the base weights.
    Lora(LoraRole),
}

/// A projection a LoRA adapter can target: the four attention projections and the three gated-MLP ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoraTarget {
    /// The attention query projection ([`AttnRole::Q`]).
    AttnQ,
    /// The attention key projection ([`AttnRole::K`]).
    AttnK,
    /// The attention value projection ([`AttnRole::V`]).
    AttnV,
    /// The attention output projection ([`AttnRole::O`]).
    AttnO,
    /// The gated MLP's gate projection ([`FfnRole::Gate`]).
    FfnGate,
    /// The gated MLP's up projection ([`FfnRole::Up`]).
    FfnUp,
    /// The gated MLP's down projection ([`FfnRole::Down`]).
    FfnDown,
}

impl LoraTarget {
    /// Every targetable projection.
    pub const ALL: [LoraTarget; 7] = [
        LoraTarget::AttnQ,
        LoraTarget::AttnK,
        LoraTarget::AttnV,
        LoraTarget::AttnO,
        LoraTarget::FfnGate,
        LoraTarget::FfnUp,
        LoraTarget::FfnDown,
    ];

    /// The base weight role this projection is.
    pub const fn weight_role(self) -> WeightRole {
        match self {
            LoraTarget::AttnQ => WeightRole::Attn(AttnRole::Q),
            LoraTarget::AttnK => WeightRole::Attn(AttnRole::K),
            LoraTarget::AttnV => WeightRole::Attn(AttnRole::V),
            LoraTarget::AttnO => WeightRole::Attn(AttnRole::O),
            LoraTarget::FfnGate => WeightRole::Ffn(FfnRole::Gate),
            LoraTarget::FfnUp => WeightRole::Ffn(FfnRole::Up),
            LoraTarget::FfnDown => WeightRole::Ffn(FfnRole::Down),
        }
    }

    /// The projection `role` is, if it is one an adapter can target.
    pub fn of(role: WeightRole) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|target| target.weight_role() == role)
    }
}

/// Which stacked adapter tensor a [`WeightRole::Lora`] weight holds. `A` is `[adapters + 1, in, r]` and
/// `B` is `[adapters + 1, r, out]` for the projection `target` of a layer, both in `x @ W` orientation
/// and zero-padded to a common rank; row 0 of each is the no-adapter row. `Scale` is the model-level
/// `[adapters + 1]` per-adapter `lora_alpha / r`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoraRole {
    /// The stacked `[adapters + 1, in, r]` factor of the adapter for this projection.
    A(LoraTarget),
    /// The stacked `[adapters + 1, r, out]` factor of the adapter for this projection.
    B(LoraTarget),
    /// The model-level `[adapters + 1]` per-adapter `lora_alpha / r`.
    Scale,
}

impl LoraRole {
    const fn spelling(self) -> &'static str {
        match self {
            LoraRole::A(LoraTarget::AttnQ) => "lora.a.attn.q",
            LoraRole::A(LoraTarget::AttnK) => "lora.a.attn.k",
            LoraRole::A(LoraTarget::AttnV) => "lora.a.attn.v",
            LoraRole::A(LoraTarget::AttnO) => "lora.a.attn.o",
            LoraRole::A(LoraTarget::FfnGate) => "lora.a.ffn.gate",
            LoraRole::A(LoraTarget::FfnUp) => "lora.a.ffn.up",
            LoraRole::A(LoraTarget::FfnDown) => "lora.a.ffn.down",
            LoraRole::B(LoraTarget::AttnQ) => "lora.b.attn.q",
            LoraRole::B(LoraTarget::AttnK) => "lora.b.attn.k",
            LoraRole::B(LoraTarget::AttnV) => "lora.b.attn.v",
            LoraRole::B(LoraTarget::AttnO) => "lora.b.attn.o",
            LoraRole::B(LoraTarget::FfnGate) => "lora.b.ffn.gate",
            LoraRole::B(LoraTarget::FfnUp) => "lora.b.ffn.up",
            LoraRole::B(LoraTarget::FfnDown) => "lora.b.ffn.down",
            LoraRole::Scale => "lora.scale",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// The weights of one layer's attention: its projections, biases and QK norms.
pub enum AttnRole {
    /// The query projection.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The output projection.
    O,
    /// The query projection's bias.
    QBias,
    /// The key projection's bias.
    KBias,
    /// The value projection's bias.
    VBias,
    /// The norm scale applied to the query (QK-norm).
    QNorm,
    /// The norm scale applied to the key (QK-norm).
    KNorm,
    /// The output projection's bias.
    OBias,
    /// A fused `[(heads + 2 * kv_heads) * head_dim, width]` QKV projection, and its bias. The row
    /// order is the family's: BLOOM interleaves per head (`q_h, k_h, v_h` for each head), a Qwen2.5-VL
    /// vision block stacks `q | k | v`. A family that has one reads it instead of `Q`/`K`/`V`.
    Qkv,
    /// The bias of the fused QKV projection ([`AttnRole::Qkv`]).
    QkvBias,
    /// The per-query-head attention sink logits (gpt-oss), `[heads]`.
    Sinks,
    /// The first stage of a grouped low-rank output projection (DeepSeek-V4's `wo_a`): the stored
    /// `[groups * rank, heads * head_dim / groups]` matrix, read as `groups` block-diagonal
    /// projections. The second stage is [`AttnRole::O`].
    OA,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// The projections and biases of one layer's dense feed-forward block.
pub enum FfnRole {
    /// The gate projection of a gated MLP.
    Gate,
    /// The up projection of a gated or plain MLP.
    Up,
    /// The down projection of a gated or plain MLP.
    Down,
    /// The biases of a plain MLP's two projections, and of a gated MLP's gate (a Qwen2.5-VL vision
    /// block's three).
    UpBias,
    /// The down projection's bias.
    DownBias,
    /// The gate projection's bias.
    GateBias,
}

/// The projection of a routed expert a weight holds. `GateUp` is the one fused gate and up projection
/// a checkpoint stores as a single tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertProj {
    /// The gate projection of one expert.
    Gate,
    /// The up projection of one expert.
    Up,
    /// The gate and up projections fused into one tensor.
    GateUp,
    /// The down projection of one expert.
    Down,
}

impl ExpertProj {
    const fn spelling(self) -> &'static str {
        match self {
            ExpertProj::Gate => "gate",
            ExpertProj::Up => "up",
            ExpertProj::GateUp => "gate_up",
            ExpertProj::Down => "down",
        }
    }
}

/// The weights of a routed mixture-of-experts block. A checkpoint stores one projection of every expert
/// as a stack ([`MoeRole::Stack`]), whether as one tensor or as per-expert tensors; per-expert packed
/// weights (each one `[out, K]` matrix) are viewed as one stack ([`WeightView::ExpertStack`]) without
/// being copied together.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MoeRole {
    /// The router projection, `[experts, width]`.
    Router,
    /// The router's bias, `[experts]`.
    RouterBias,
    /// The scale on the router's normalized input, `[width]` (Gemma 4).
    RouterScale,
    /// The scale on each expert's routing weight, `[experts]` (Gemma 4's `per_expert_scale`).
    ExpertScale,
    /// One projection of every expert: the stored stack `[experts, out, K]` / `[experts, K, out]`, or
    /// per-expert tensors, dense or packed, stacked into `[experts, out, K]`
    /// ([`WeightView::ExpertStack`]).
    Stack(ExpertProj),
    /// The bias of every expert's projection, `[experts, out]`.
    StackBias(ExpertProj),
    /// One projection of the always-active shared expert, `[out, K]`.
    Shared(ExpertProj),
    /// A hash router's token-to-experts table, `[vocab, top_k]` I32 (DeepSeek-V4's `tid2eid`).
    HashTable,
    /// The shared expert's scalar gate: one logit per token, `[1, width]` (or `[width]`), through a
    /// sigmoid (Qwen3-Next).
    SharedGate,
}

/// The weights of multi-head latent attention (DeepSeek). A plain query is [`AttnRole::Q`]; the
/// output projection is [`AttnRole::O`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MlaRole {
    /// The low-rank query's down projection `[q_lora_rank, width]`, its norm scale and its up
    /// projection `[heads * (nope + rope), q_lora_rank]`.
    QA,
    /// The norm scale over the query latent.
    QANorm,
    /// The query up projection out of the latent.
    QB,
    /// The shared KV latent's down projection `[kv_lora_rank + rope, width]` and its norm scale.
    KvA,
    /// The norm scale over the KV latent.
    KvANorm,
    /// HF's one up projection `[heads * (nope + v), kv_lora_rank]`, viewed per head in the graph.
    KvB,
    /// llama.cpp's split up projections, stored per head: the key's `[heads, kv_lora_rank, nope]`
    /// and the value's `[heads, v, kv_lora_rank]`.
    KB,
    /// llama.cpp's per-head value up projection.
    VB,
}

/// The weights of a Mamba2 mixer layer (Nemotron-H). The layer's pre-mixer norm is
/// [`NormRole::Attn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mamba2Role {
    /// The three row ranges of the fused input projection: the output gate `z`, the conv input
    /// `xBC` and the timestep `dt`.
    Z,
    /// The conv-input (`xBC`) rows of the fused input projection.
    Xbc,
    /// The timestep (`dt`) rows of the fused input projection.
    Dt,
    /// The bias added to `dt`.
    DtBias,
    /// The depthwise conv kernel `[K, channels]` and its bias.
    Conv,
    /// The depthwise conv's bias.
    ConvBias,
    /// `log(-A)` per head and the skip `D` per head.
    ALog,
    /// The per-head skip connection `D`.
    D,
    /// The gated RMSNorm scale before the output projection, and the output projection.
    GateNorm,
    /// The output projection.
    Out,
}

/// The weights of a Gated DeltaNet mixer layer (Qwen3-Next). The layer's pre-mixer norm is
/// [`NormRole::Attn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GdnRole {
    /// The mixed query, key and value projection `[2 * key_dim + value_dim, width]` and the output
    /// gate `z` projection `[value_dim, width]`, each separate (a GGUF; Qwen3.5).
    Qkv,
    /// The output gate `z` projection.
    Z,
    /// The per-value-head beta and decay (`alpha`) projections `[value_heads, width]`, separate.
    Beta,
    /// The per-value-head decay (`alpha`) projection.
    Alpha,
    /// The fused projections of an HF `qwen3_next` checkpoint, whose rows group per key head:
    /// `[key_heads * (2 * head_dim + 2 * value_heads / key_heads * head_dim), width]` for `q|k|v|z`
    /// and `[2 * value_heads, width]` for `b|a`.
    Qkvz,
    /// The fused `b` and `a` (beta and alpha) projection of an HF checkpoint.
    Ba,
    /// The bias added to `alpha` before its softplus, `[value_heads]`.
    DtBias,
    /// The depthwise conv kernel over the mixed channels, `[channels, 1, K]` (or `[channels, K]`).
    Conv,
    /// The per-head decay factor, `[value_heads]`: `A_log` (HF; the decay is `-exp(A_log)`) or the
    /// decay itself, already negative (GGUF's `ssm_a`).
    A,
    /// The gated RMSNorm scale over each value head's `head_dim`, and the output projection.
    GateNorm,
    /// The output projection.
    Out,
}

/// The weights of a Kimi Delta Attention mixer layer (GLM-5.3-Flash). The layer's pre-mixer norm is
/// [`NormRole::Attn`]. `heads * head_dim` is the width of each of the query, key and value
/// projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KdaRole {
    /// The query, key and value projections `[heads * head_dim, width]`.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The depthwise conv kernels over the query, key and value channels, `[heads * head_dim, 1, K]`.
    ConvQ,
    /// The depthwise conv kernel over the key channels.
    ConvK,
    /// The depthwise conv kernel over the value channels.
    ConvV,
    /// The low-rank forget-gate projections: `FA` `[gate_rank, width]` and `FB`
    /// `[heads * head_dim, gate_rank]`.
    FA,
    /// The forget gate's up projection out of the low-rank bottleneck.
    FB,
    /// The per-head beta projection `[heads, width]`.
    Beta,
    /// The low-rank output-gate projections, shaped as `FA` and `FB`.
    GA,
    /// The output gate's up projection out of the low-rank bottleneck.
    GB,
    /// The per-head log decay rate `[heads]` and the bias added to the forget gate's argument,
    /// `[heads * head_dim]`.
    ALog,
    /// The bias added to the forget gate's argument.
    DtBias,
    /// The RMSNorm scale over each head's `head_dim`, and the output projection `[width, heads *
    /// head_dim]`.
    ONorm,
    /// The output projection.
    Out,
}

/// The weights a multi-token-prediction layer adds around an ordinary block: its input fuses the
/// normalized embedding of the next token with the normalized hidden state of the position, and the
/// shared head reads the block's output through its own norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MtpRole {
    /// The fusing projection `[width, 2 * width]` over `concat(enorm(embed), hnorm(hidden))`.
    EhProj,
    /// The RMSNorm scale `[width]` of the next token's embedding (`enorm`).
    ENorm,
    /// The RMSNorm scale `[width]` of the position's hidden state (`hnorm`).
    HNorm,
    /// The RMSNorm scale `[width]` applied to the block's output before the shared head.
    HeadNorm,
}

/// The lightning indexer of DeepSeek sparse attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DsaRole {
    /// The query projection from the low-rank query `[heads * dim, q_lora_rank]`.
    Q,
    /// The key projection `[dim, width]`, its LayerNorm scale and bias.
    K,
    /// The LayerNorm scale over the key.
    KNorm,
    /// The LayerNorm bias over the key.
    KNormBias,
    /// The per-head weights projection `[heads, width]`.
    Weights,
}

/// The block indexer of Qwen Sparse Attention: its query and key projections (read from one fused
/// checkpoint tensor, `[(heads + 1) * dim, width]`, query rows first) and the scales of the two
/// norms over each head's `dim`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QsaRole {
    /// The query projection `[heads * dim, width]`.
    Q,
    /// The shared key projection `[dim, width]`.
    K,
    /// The scale of the norm over each query head's `dim`.
    QNorm,
    /// The scale of the norm over the key's `dim`.
    KNorm,
}

/// Where a Hyper-Connections gate sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HyperSite {
    /// Around a layer's mixer (attention or Gated DeltaNet).
    Attn,
    /// Around a layer's feed-forward.
    Mlp,
    /// The model-level mixer after the last layer (a model-level weight).
    Final,
}

/// One Hyper-Connections gate's weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HyperRole {
    /// The grouped RMSNorm scale over the `streams * width` state, `[streams * width]`.
    Norm,
    /// The low-rank gate projections `[rank, streams * width]` and `[streams * width, rank]`.
    MixDown,
    /// The low-rank gate's up projection.
    MixUp,
    /// The per-stream injection weights `[streams, streams * width]` (not on the final mixer).
    Inject,
}

/// The weights of a per-layer-embedding block: the n-gram embedding table and the projections,
/// norms and short conv that gate it into the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PleRole {
    /// The n-gram embedding table `[rows, head_dim]`.
    Table,
    /// One row-range shard of the table, `[shard_rows, head_dim]`, in logical row order: shard `i`
    /// holds the rows after shards `0..i`. A table is stored as `Table` or as shards, never both.
    TableShard(u32),
    /// The scalar a sharded table's rows are multiplied by after the lookup, `[]`.
    TableScale,
    /// The key projection `[streams * width, embed_dim]` and value projection `[width, embed_dim]`.
    Key,
    /// The value projection.
    Value,
    /// The grouped norm scales over the key, the query (the stream) and the gated value.
    KeyNorm,
    /// The grouped norm scale over the query (the stream).
    QueryNorm,
    /// The grouped norm scale over the gated value.
    ConvNorm,
    /// The dilated depthwise conv kernel `[streams * width, 1, K]`.
    Conv,
}

/// The weights of one mHC site. A layer has an `Attn` and an `Ffn` site and the model one `Head`
/// collapse; each site is a projection `[mix, streams * width]`, a base `[mix]` and a scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MhcRole {
    /// The attention site's projection `[mix, streams * width]`.
    AttnFn,
    /// The attention site's base `[mix]`.
    AttnBase,
    /// The attention site's scale.
    AttnScale,
    /// The feed-forward site's projection `[mix, streams * width]`.
    FfnFn,
    /// The feed-forward site's base `[mix]`.
    FfnBase,
    /// The feed-forward site's scale.
    FfnScale,
    /// The model-level head collapse's projection `[mix, streams * width]`.
    HeadFn,
    /// The model-level head collapse's base `[mix]`.
    HeadBase,
    /// The model-level head collapse's scale.
    HeadScale,
}

/// The weights of an attention-window compressor: the key/value and gate projections of the layer's
/// compressed branch, the learned position bias added to the gate, and the norm of the pooled
/// entry. The `Indexer` rows are the same four for the lightning indexer's own compressor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompressRole {
    /// The key/value projection of the layer's compressed branch.
    Kv,
    /// The gate projection of the layer's compressed branch.
    Gate,
    /// The learned position bias added to the compressor's gate.
    PositionBias,
    /// The norm scale of the pooled entry.
    Norm,
    /// The key/value projection of the lightning indexer's compressor.
    IndexerKv,
    /// The gate projection of the lightning indexer's compressor.
    IndexerGate,
    /// The position bias added to the indexer compressor's gate.
    IndexerPositionBias,
    /// The norm scale of the indexer compressor's pooled entry.
    IndexerNorm,
}

/// The per-layer norms around a block (the model-level output norm is [`WeightRole::FinalNorm`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NormRole {
    /// The norm before the layer's mixer (attention or recurrent).
    Attn,
    /// The norm before the layer's feed-forward.
    Ffn,
    /// The norms applied to a block's output before it joins the residual (a sandwich layer's
    /// post-attention and post-feed-forward norms).
    PostAttn,
    /// The norm applied to the feed-forward's output before it joins the residual.
    PostFfn,
    /// The biases of a LayerNorm's [`NormRole::Attn`] and [`NormRole::Ffn`] scales.
    AttnBias,
    /// The bias of the [`NormRole::Ffn`] LayerNorm.
    FfnBias,
    /// The norm before a routed block that sits beside a dense feed-forward (Gemma 4's MoE
    /// layers): the dense block reads [`NormRole::Ffn`], the routed experts read this one.
    PreRouted,
    /// The norms applied to the dense and the routed block's outputs before they are summed.
    PostDense,
    /// The norm applied to the routed block's output before the sum.
    PostRouted,
}

impl WeightRole {
    /// The role's stable spelling inside [`WeightId::const_name`]. Written out per role, never
    /// derived from `Debug`: the name enters kernel caches and plan summaries, so renaming a variant
    /// must not silently rename a graph const.
    pub fn spelling(self) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            WeightRole::Moe(role) => return role.spelling(),
            WeightRole::Mla(role) => match role {
                MlaRole::QA => "mla.q_a",
                MlaRole::QANorm => "mla.q_a_norm",
                MlaRole::QB => "mla.q_b",
                MlaRole::KvA => "mla.kv_a",
                MlaRole::KvANorm => "mla.kv_a_norm",
                MlaRole::KvB => "mla.kv_b",
                MlaRole::KB => "mla.k_b",
                MlaRole::VB => "mla.v_b",
            },
            WeightRole::Mamba2(role) => match role {
                Mamba2Role::Z => "mamba2.z",
                Mamba2Role::Xbc => "mamba2.xbc",
                Mamba2Role::Dt => "mamba2.dt",
                Mamba2Role::DtBias => "mamba2.dt_bias",
                Mamba2Role::Conv => "mamba2.conv",
                Mamba2Role::ConvBias => "mamba2.conv_bias",
                Mamba2Role::ALog => "mamba2.a_log",
                Mamba2Role::D => "mamba2.d",
                Mamba2Role::GateNorm => "mamba2.gate_norm",
                Mamba2Role::Out => "mamba2.out",
            },
            WeightRole::Gdn(role) => match role {
                GdnRole::Qkv => "gdn.qkv",
                GdnRole::Z => "gdn.z",
                GdnRole::Beta => "gdn.beta",
                GdnRole::Alpha => "gdn.alpha",
                GdnRole::Qkvz => "gdn.qkvz",
                GdnRole::Ba => "gdn.ba",
                GdnRole::DtBias => "gdn.dt_bias",
                GdnRole::Conv => "gdn.conv",
                GdnRole::A => "gdn.a",
                GdnRole::GateNorm => "gdn.gate_norm",
                GdnRole::Out => "gdn.out",
            },
            WeightRole::Kda(role) => match role {
                KdaRole::Q => "kda.q",
                KdaRole::K => "kda.k",
                KdaRole::V => "kda.v",
                KdaRole::ConvQ => "kda.conv_q",
                KdaRole::ConvK => "kda.conv_k",
                KdaRole::ConvV => "kda.conv_v",
                KdaRole::FA => "kda.f_a",
                KdaRole::FB => "kda.f_b",
                KdaRole::Beta => "kda.beta",
                KdaRole::GA => "kda.g_a",
                KdaRole::GB => "kda.g_b",
                KdaRole::ALog => "kda.a_log",
                KdaRole::DtBias => "kda.dt_bias",
                KdaRole::ONorm => "kda.o_norm",
                KdaRole::Out => "kda.out",
            },
            WeightRole::Mtp(role) => match role {
                MtpRole::EhProj => "mtp.eh_proj",
                MtpRole::ENorm => "mtp.enorm",
                MtpRole::HNorm => "mtp.hnorm",
                MtpRole::HeadNorm => "mtp.head_norm",
            },
            WeightRole::Dsa(role) => match role {
                DsaRole::Q => "dsa.q",
                DsaRole::K => "dsa.k",
                DsaRole::KNorm => "dsa.k_norm",
                DsaRole::KNormBias => "dsa.k_norm_bias",
                DsaRole::Weights => "dsa.weights",
            },
            WeightRole::Qsa(role) => match role {
                QsaRole::Q => "qsa.q",
                QsaRole::K => "qsa.k",
                QsaRole::QNorm => "qsa.q_norm",
                QsaRole::KNorm => "qsa.k_norm",
            },
            WeightRole::Hyper(site, role) => {
                return Cow::Owned(format!(
                    "hyper.{}.{}",
                    match site {
                        HyperSite::Attn => "attn",
                        HyperSite::Mlp => "mlp",
                        HyperSite::Final => "final",
                    },
                    match role {
                        HyperRole::Norm => "norm",
                        HyperRole::MixDown => "mix_down",
                        HyperRole::MixUp => "mix_up",
                        HyperRole::Inject => "inject",
                    }
                ));
            }
            WeightRole::Ple(role) => match role {
                PleRole::Table => "ple.table",
                PleRole::TableShard(shard) => {
                    return Cow::Owned(format!("ple.table.shard_{shard}"));
                }
                PleRole::TableScale => "ple.table_scale",
                PleRole::Key => "ple.key",
                PleRole::Value => "ple.value",
                PleRole::KeyNorm => "ple.key_norm",
                PleRole::QueryNorm => "ple.query_norm",
                PleRole::ConvNorm => "ple.conv_norm",
                PleRole::Conv => "ple.conv",
            },
            WeightRole::Mhc(role) => match role {
                MhcRole::AttnFn => "mhc.attn_fn",
                MhcRole::AttnBase => "mhc.attn_base",
                MhcRole::AttnScale => "mhc.attn_scale",
                MhcRole::FfnFn => "mhc.ffn_fn",
                MhcRole::FfnBase => "mhc.ffn_base",
                MhcRole::FfnScale => "mhc.ffn_scale",
                MhcRole::HeadFn => "mhc.head_fn",
                MhcRole::HeadBase => "mhc.head_base",
                MhcRole::HeadScale => "mhc.head_scale",
            },
            WeightRole::Compress(role) => match role {
                CompressRole::Kv => "compress.kv",
                CompressRole::Gate => "compress.gate",
                CompressRole::PositionBias => "compress.position_bias",
                CompressRole::Norm => "compress.norm",
                CompressRole::IndexerKv => "compress.indexer_kv",
                CompressRole::IndexerGate => "compress.indexer_gate",
                CompressRole::IndexerPositionBias => "compress.indexer_position_bias",
                CompressRole::IndexerNorm => "compress.indexer_norm",
            },
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
                AttnRole::Sinks => "attn.sinks",
                AttnRole::OA => "attn.o_a",
            },
            WeightRole::Ffn(role) => match role {
                FfnRole::Gate => "ffn.gate",
                FfnRole::Up => "ffn.up",
                FfnRole::Down => "ffn.down",
                FfnRole::UpBias => "ffn.up_bias",
                FfnRole::DownBias => "ffn.down_bias",
                FfnRole::GateBias => "ffn.gate_bias",
            },
            WeightRole::Norm(role) => match role {
                NormRole::Attn => "norm.attn",
                NormRole::Ffn => "norm.ffn",
                NormRole::PostAttn => "norm.post_attn",
                NormRole::PostFfn => "norm.post_ffn",
                NormRole::AttnBias => "norm.attn_bias",
                NormRole::FfnBias => "norm.ffn_bias",
                NormRole::PreRouted => "norm.pre_routed",
                NormRole::PostDense => "norm.post_dense",
                NormRole::PostRouted => "norm.post_routed",
            },
            WeightRole::LayerScale => "layer_scale",
            WeightRole::PosEmbed => "pos_embed",
            WeightRole::TypeEmbed => "type_embed",
            WeightRole::Pooler => "pooler",
            WeightRole::PoolerBias => "pooler_bias",
            WeightRole::HeadBias => "head_bias",
            WeightRole::Patch => "patch",
            WeightRole::PatchBias => "patch_bias",
            WeightRole::Connector => "connector",
            WeightRole::ConnectorBias => "connector_bias",
            WeightRole::ConnectorOut => "connector_out",
            WeightRole::ConnectorOutBias => "connector_out_bias",
            WeightRole::Lora(role) => role.spelling(),
        })
    }
}

impl MoeRole {
    fn spelling(self) -> Cow<'static, str> {
        match self {
            MoeRole::Router => Cow::Borrowed("moe.router"),
            MoeRole::RouterBias => Cow::Borrowed("moe.router_bias"),
            MoeRole::RouterScale => Cow::Borrowed("moe.router_scale"),
            MoeRole::ExpertScale => Cow::Borrowed("moe.expert_scale"),
            MoeRole::Stack(proj) => Cow::Owned(format!("moe.stack.{}", proj.spelling())),
            MoeRole::StackBias(proj) => Cow::Owned(format!("moe.stack_bias.{}", proj.spelling())),
            MoeRole::Shared(proj) => Cow::Owned(format!("moe.shared.{}", proj.spelling())),
            MoeRole::HashTable => Cow::Borrowed("moe.hash_table"),
            MoeRole::SharedGate => Cow::Borrowed("moe.shared_gate"),
        }
    }
}

/// The typed identity of one model weight: its layer (`None` for a model-level weight) and role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WeightId {
    /// The layer index, or `None` for a model-level weight.
    pub layer: Option<usize>,
    /// What the weight is for.
    pub role: WeightRole,
}

impl WeightId {
    /// The identity of a model-level weight (no layer).
    pub const fn model(role: WeightRole) -> Self {
        Self { layer: None, role }
    }

    /// The identity of the `role` weight of layer `layer`.
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
    /// Entries of one format and shape stacked along a new leading axis, in order: the
    /// `[experts, out, K]` stack of per-expert tensors. Dense parts are copied into one dense entry
    /// ([`WeightMap::materialize`]); packed parts stay their own payloads and are read together as a
    /// [`PackedStack`] ([`WeightMap::materialize_stack`]), whose device table pads every expert's
    /// source to word strides ([`crate::PackedWeight::slab_bytes`]).
    ExpertStack(Vec<WeightKey>),
    /// A leading-axis row range of one entry (a slice of a fused `qkv`).
    RowRange {
        /// The entry the rows are cut from.
        key: WeightKey,
        /// The half-open row range of the entry's leading axis.
        rows: std::ops::Range<usize>,
    },
    /// An index table stored as I64, read as I32 (DeepSeek-V4's `tid2eid`): every value is
    /// range-checked when the map is built, so a table that does not fit I32 is refused before any
    /// trace. The one view whose bytes are derived rather than cut from the store.
    IndexNarrow(WeightKey),
    /// A range `cols` of axis `axis` (at least 1) of one entry: the share a tensor-parallel rank holds of a
    /// row-parallel weight (the input columns of a checkpoint `[out, K]` matrix) or of a stacked adapter
    /// factor. A packed `[out, K]` payload is cut on whole encoding blocks of its `K` axis only: a range
    /// off a block boundary, or of a planar format, is refused when the view is built.
    BlockColumns {
        /// The entry the columns are cut from.
        key: WeightKey,
        /// The axis cut; at least 1.
        axis: usize,
        /// The half-open range along `axis`.
        cols: std::ops::Range<usize>,
    },
}

impl WeightView {
    /// The store keys this view reads.
    fn keys(&self) -> &[WeightKey] {
        match self {
            Self::Stored(key)
            | Self::IndexNarrow(key)
            | Self::RowRange { key, .. }
            | Self::BlockColumns { key, .. } => std::slice::from_ref(key),
            Self::RowStack(keys) | Self::ExpertStack(keys) => keys,
        }
    }
}

/// The stored format of a mapped weight (the driver path's only source of weight
/// formats). A dense entry names its stored dtype; a packed one its quantized `[out, K]` weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandleFormat {
    /// A dense entry: its stored dtype.
    Dense(poot_tensor::DType),
    /// A packed entry: its quantized `[out, K]` weight.
    Packed(crate::PackedWeight),
}

/// What a tracer may see of a weight: its stored format and logical shape. No bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightHandle {
    /// The stored format of the weight.
    pub format: HandleFormat,
    /// The logical shape: the entry's full shape when dense, `[out, K]` when packed.
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
    Missing {
        /// The weight being mapped.
        id: WeightId,
        /// The key the view names that the store lacks.
        key: WeightKey,
    },
    /// `id` was mapped twice.
    Duplicate {
        /// The weight that was mapped twice.
        id: WeightId,
    },
    /// [`WeightMap::materialize`] was asked for an `id` the map does not hold.
    Unmapped {
        /// The weight that is not in the map.
        id: WeightId,
    },
    /// A [`WeightView::RowStack`] with no parts.
    EmptyStack {
        /// The weight being mapped.
        id: WeightId,
    },
    /// A row view over a rank-0 dense entry: there is no row axis.
    NoRowAxis {
        /// The weight being mapped.
        id: WeightId,
        /// The rank-0 entry.
        key: WeightKey,
    },
    /// A [`WeightView::RowRange`] that is empty or reaches past the entry's rows.
    RowRangeOutOfBounds {
        /// The weight being mapped.
        id: WeightId,
        /// The requested row range.
        rows: std::ops::Range<usize>,
        /// Rows the entry has.
        available: usize,
    },
    /// [`WeightView::RowStack`] parts that do not share one format and one row shape.
    MixedParts {
        /// The weight being mapped.
        id: WeightId,
        /// The first part's handle, which the others must match.
        first: Box<WeightHandle>,
        /// The first part that differs.
        other: Box<WeightHandle>,
    },
    /// The packed payload refuses the row view (a planar format whose rows are not byte runs).
    Packed {
        /// The weight being mapped.
        id: WeightId,
        /// The packed payload's own refusal.
        source: Box<crate::PackedWeightError>,
    },
    /// A [`WeightView::IndexNarrow`] over an entry that is not a dense I64 or I32 table.
    NotAnIndexTable {
        /// The weight being mapped.
        id: WeightId,
        /// The entry the view names.
        key: WeightKey,
        /// What the entry actually is.
        found: Box<WeightHandle>,
    },
    /// A [`WeightView::IndexNarrow`] table holding a value outside I32: element `index` (row-major)
    /// of `key` is `value`.
    IndexOutOfRange {
        /// The weight being mapped.
        id: WeightId,
        /// The index table's entry.
        key: WeightKey,
        /// Row-major element index of the offending value.
        index: usize,
        /// The value that does not fit I32.
        value: i64,
    },
    /// A [`WeightView::ExpertStack`] over an entry whose format (dense dtype or packed format) or
    /// shape differs from the first part's.
    NotStackable {
        /// The weight being mapped.
        id: WeightId,
        /// The entry that differs from the first part.
        key: WeightKey,
        /// The first part's handle, which the others must match.
        first: Box<WeightHandle>,
        /// The handle of the differing entry.
        other: Box<WeightHandle>,
    },
    /// [`WeightMap::materialize`] over a packed [`WeightView::ExpertStack`]: its experts are separate
    /// payloads, read with [`WeightMap::materialize_stack`], never one entry.
    PackedStackIsNotOneEntry {
        /// The packed expert stack that was asked for as one entry.
        id: WeightId,
    },
    /// [`WeightMap::materialize_stack`] over a view that is not a packed expert stack.
    NotAPackedStack {
        /// The weight whose view is not a packed expert stack.
        id: WeightId,
    },
    /// A [`WeightView::BlockColumns`] over axis 0 (that is [`WeightView::RowRange`]) or past the entry's
    /// rank.
    ColumnAxis {
        /// The weight being mapped.
        id: WeightId,
        /// The entry the view cuts.
        key: WeightKey,
        /// The requested axis.
        axis: usize,
        /// The entry's rank.
        rank: usize,
    },
    /// A [`WeightView::BlockColumns`] range that is empty or reaches past its axis.
    ColumnRangeOutOfBounds {
        /// The weight being mapped.
        id: WeightId,
        /// The axis cut.
        axis: usize,
        /// The requested column range.
        cols: std::ops::Range<usize>,
        /// Extent of `axis`.
        available: usize,
    },
    /// A column range of a packed weight that does not start and end on whole encoding blocks of
    /// `block_values` values, so a block would be cut in two.
    ColumnsOffBlock {
        /// The weight being mapped.
        id: WeightId,
        /// The packed entry the view cuts.
        key: WeightKey,
        /// The requested column range.
        cols: std::ops::Range<usize>,
        /// Values per encoding block; the range must start and end on multiples of it.
        block_values: usize,
    },
    /// A column range of a packed weight whose format is not stored as one run of blocks per row (a
    /// planar format), or of a packed axis other than its `K`.
    ColumnsOfPlanar {
        /// The weight being mapped.
        id: WeightId,
        /// The packed entry the view cuts.
        key: WeightKey,
    },
    /// [`WeightMap::rank_view`] over an axis whose extent `world` does not divide.
    ShareNotDivisible {
        /// The weight being cut.
        id: WeightId,
        /// The axis being divided.
        axis: usize,
        /// Extent of `axis`.
        extent: usize,
        /// Number of ranks `extent` must divide by.
        world: u32,
    },
    /// [`WeightMap::rank_view`] over a weight whose view cannot be cut further on `axis`: a stack, an
    /// index table, or a range already cut on another axis.
    ShareOfView {
        /// The weight being cut.
        id: WeightId,
        /// The axis the cut was requested on.
        axis: usize,
    },
    /// [`WeightMap::check_store`] over a store whose entries read, for `id`, a different handle than the
    /// map recorded: the map was built against another store.
    StoreMismatch {
        /// The weight whose view no longer reads what the map recorded.
        id: WeightId,
        /// What the map recorded.
        mapped: Box<WeightHandle>,
        /// What the store's entries give the view now.
        stored: Box<WeightHandle>,
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
            Self::NotAnIndexTable { id, key, found } => write!(
                f,
                "weight {id}: entry {key} is {found:?}, not an I64 or I32 index table"
            ),
            Self::IndexOutOfRange {
                id,
                key,
                index,
                value,
            } => write!(
                f,
                "weight {id}: entry {key} element {index} is {value}, outside the I32 range"
            ),
            Self::NotStackable {
                id,
                key,
                first,
                other,
            } => write!(
                f,
                "weight {id}: entry {key} cannot join an expert stack of {first:?}: it is {other:?}"
            ),
            Self::PackedStackIsNotOneEntry { id } => write!(
                f,
                "weight {id}: a packed expert stack is one payload per expert, not one entry (read it with materialize_stack)"
            ),
            Self::NotAPackedStack { id } => {
                write!(f, "weight {id} is not a packed expert stack")
            }
            Self::ColumnAxis {
                id,
                key,
                axis,
                rank,
            } => write!(
                f,
                "weight {id}: entry {key} of rank {rank} has no column axis {axis} (axis 0 is the rows)"
            ),
            Self::ColumnRangeOutOfBounds {
                id,
                axis,
                cols,
                available,
            } => write!(
                f,
                "weight {id}: range {cols:?} is not a non-empty range of axis {axis}'s {available} values"
            ),
            Self::ColumnsOffBlock {
                id,
                key,
                cols,
                block_values,
            } => write!(
                f,
                "weight {id}: columns {cols:?} of packed entry {key} are not whole {block_values}-value blocks"
            ),
            Self::ColumnsOfPlanar { id, key } => write!(
                f,
                "weight {id}: packed entry {key} is not stored as one run of blocks per row, so its columns cannot be cut"
            ),
            Self::ShareNotDivisible {
                id,
                axis,
                extent,
                world,
            } => write!(
                f,
                "weight {id}: axis {axis} of extent {extent} does not divide over {world} ranks"
            ),
            Self::ShareOfView { id, axis } => write!(
                f,
                "weight {id}: its view cannot be cut to a rank's share of axis {axis}"
            ),
            Self::StoreMismatch { id, mapped, stored } => write!(
                f,
                "weight {id}: the map recorded {mapped:?} but the store gives its view {stored:?}"
            ),
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
    /// Stored tensors the family recognises and does not read.
    known: BTreeSet<WeightKey>,
}

/// How a checkpoint's stored tensors stand against a model's [`WeightMap`]: every store key is read by
/// the model, known to it and left unread, or not recognised at all.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// Keys some mapped weight reads.
    pub loaded: BTreeSet<WeightKey>,
    /// Keys the family claims without reading (a multi-token-prediction head it does not trace).
    pub known: BTreeSet<WeightKey>,
    /// Keys the family neither reads nor claims.
    pub unknown: BTreeSet<WeightKey>,
}

impl WeightMap {
    /// A builder over `store` that starts from an empty map.
    pub fn builder(store: &WeightStore) -> WeightMapBuilder<'_> {
        WeightMapBuilder {
            store,
            map: WeightMap::default(),
        }
    }

    /// A builder over `store` that starts from this map's weights, for a map that adds weights beside
    /// them (an adapter generation's stacked LoRA tensors). `store` must hold every entry this map's
    /// views read, as a store built with [`WeightStoreBuilder::extend_from`] over the original does.
    pub fn extending<'s>(&self, store: &'s WeightStore) -> WeightMapBuilder<'s> {
        WeightMapBuilder {
            store,
            map: self.clone(),
        }
    }

    /// What a tracer may see of `id`: its stored format and logical shape; `None` when `id` is unmapped.
    pub fn handle(&self, id: WeightId) -> Option<&WeightHandle> {
        self.entries.get(&id).map(|(_, handle)| handle)
    }

    /// How `id` reads its bytes out of the store; `None` when `id` is unmapped.
    pub fn view(&self, id: WeightId) -> Option<&WeightView> {
        self.entries.get(&id).map(|(view, _)| view)
    }

    /// Every mapped weight with its view and handle, in [`WeightId`] order.
    pub fn iter(&self) -> impl Iterator<Item = (WeightId, &WeightView, &WeightHandle)> {
        self.entries
            .iter()
            .map(|(id, (view, handle))| (*id, view, handle))
    }

    /// Number of mapped weights.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no weight is mapped.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check that `store` is the one this map was built against: every view still reads entries of
    /// `store`, and reads the handle the map recorded. A map built against another store fails with the
    /// weight, the key or the two handles, so a binder never meets a missing entry or a short one.
    pub fn check_store(&self, store: &WeightStore) -> Result<(), WeightMapError> {
        for (&id, (view, mapped)) in &self.entries {
            let stored = view_handle(id, view, store)?;
            if &stored != mapped {
                return Err(WeightMapError::StoreMismatch {
                    id,
                    mapped: Box::new(mapped.clone()),
                    stored: Box::new(stored),
                });
            }
        }
        Ok(())
    }

    /// Where every key of `store` stands: read by a mapped weight, claimed as known, or unknown.
    /// A key a weight reads is loaded even if it is also claimed.
    pub fn report(&self, store: &WeightStore) -> LoadReport {
        let loaded: BTreeSet<WeightKey> = self
            .entries
            .values()
            .flat_map(|(view, _)| view.keys().iter())
            .cloned()
            .collect();
        let mut report = LoadReport::default();
        for key in store.keys() {
            let part = if loaded.contains(key) {
                &mut report.loaded
            } else if self.known.contains(key) {
                &mut report.known
            } else {
                &mut report.unknown
            };
            part.insert(key.clone());
        }
        report
    }

    /// `id`'s bytes as one store-shaped entry: the stored entry itself for
    /// [`WeightView::Stored`], or the stacked / sliced rows copied byte for byte (a packed block
    /// format's rows are whole runs of blocks, so no decode is involved). The result's handle is
    /// the map's handle for `id`.
    #[cfg(any(test, feature = "test-support"))]
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

    /// The experts of a packed [`WeightView::ExpertStack`] `id`, each its own payload, as one
    /// [`PackedStack`]. Fails for any other view.
    #[cfg(any(test, feature = "test-support"))]
    pub fn materialize_stack(
        &self,
        id: WeightId,
        store: &WeightStore,
    ) -> Result<PackedStack, WeightMapError> {
        let (view, _) = self
            .entries
            .get(&id)
            .ok_or(WeightMapError::Unmapped { id })?;
        let WeightView::ExpertStack(parts) = view else {
            return Err(WeightMapError::NotAPackedStack { id });
        };
        let owners = parts
            .iter()
            .map(|key| match stored(id, key, store)? {
                WeightEntry::Packed(payload) => Ok(Arc::clone(payload)),
                WeightEntry::Dense(_) => Err(WeightMapError::NotAPackedStack { id }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        PackedStack::try_new(owners).map_err(packed_error(id))
    }

    /// This map as rank `rank` of `world` tensor-parallel ranks reads it: each weight of `splits`, named
    /// with the axis of its mapped logical shape that is divided, is cut to the rank's equal share of that
    /// axis; every other weight is read whole. Axis 0 is a [`WeightView::RowRange`], any other axis a
    /// [`WeightView::BlockColumns`], so no byte is copied until a binder reads the view.
    ///
    /// Every cut is checked against `store` like a family's own view: an extent `world` does not divide,
    /// a packed share off a block boundary, and a view that cannot be cut further are typed errors.
    pub fn rank_view(
        &self,
        store: &WeightStore,
        rank: usize,
        world: std::num::NonZeroU32,
        splits: &[(WeightId, usize)],
    ) -> Result<WeightMap, WeightMapError> {
        let mut view = self.clone();
        for &(id, axis) in splits {
            let (current, handle) = self
                .entries
                .get(&id)
                .ok_or(WeightMapError::Unmapped { id })?;
            let extent = *handle
                .shape
                .get(axis)
                .ok_or(WeightMapError::ShareOfView { id, axis })?;
            let world_n = world.get() as usize;
            if !extent.is_multiple_of(world_n) {
                return Err(WeightMapError::ShareNotDivisible {
                    id,
                    axis,
                    extent,
                    world: world.get(),
                });
            }
            let share = extent / world_n;
            let range = rank * share..(rank + 1) * share;
            let refused = || WeightMapError::ShareOfView { id, axis };
            let cut = match (current, axis) {
                (WeightView::Stored(key), 0) => WeightView::RowRange {
                    key: key.clone(),
                    rows: range,
                },
                (WeightView::Stored(key), _) => WeightView::BlockColumns {
                    key: key.clone(),
                    axis,
                    cols: range,
                },
                (WeightView::RowRange { key, rows }, 0) => WeightView::RowRange {
                    key: key.clone(),
                    rows: rows.start + range.start..rows.start + range.end,
                },
                (
                    WeightView::BlockColumns {
                        key,
                        axis: cut_axis,
                        cols,
                    },
                    _,
                ) if *cut_axis == axis => WeightView::BlockColumns {
                    key: key.clone(),
                    axis,
                    cols: cols.start + range.start..cols.start + range.end,
                },
                _ => return Err(refused()),
            };
            let handle = view_handle(id, &cut, store)?;
            view.entries.insert(id, (cut, handle));
        }
        Ok(view)
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

    /// Claim the stored tensor `key` as known: the family recognises it and reads no weight from
    /// it, so [`WeightMap::report`] lists it as known, not unknown.
    pub fn known(&mut self, key: WeightKey) -> &mut Self {
        self.map.known.insert(key);
        self
    }

    /// Finishes the map.
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

/// Check that `cols` is a non-empty range of axis `axis` (at least 1) of `entry`, and, for a packed
/// entry, that it is the `K` axis cut on whole blocks of a block format.
fn check_columns(
    id: WeightId,
    key: &WeightKey,
    entry: &WeightEntry,
    axis: usize,
    cols: &std::ops::Range<usize>,
) -> Result<(), WeightMapError> {
    let shape = entry.shape();
    if axis == 0 || axis >= shape.len() {
        return Err(WeightMapError::ColumnAxis {
            id,
            key: key.clone(),
            axis,
            rank: shape.len(),
        });
    }
    let extent = shape[axis];
    if cols.is_empty() || cols.end > extent {
        return Err(WeightMapError::ColumnRangeOutOfBounds {
            id,
            axis,
            cols: cols.clone(),
            available: extent,
        });
    }
    if let WeightEntry::Packed(payload) = entry {
        let planar = || WeightMapError::ColumnsOfPlanar {
            id,
            key: key.clone(),
        };
        let crate::format::Storage::Blocks(layout) = payload.weight().format().descriptor().storage
        else {
            return Err(planar());
        };
        if axis != 1 {
            return Err(planar());
        }
        if !cols.start.is_multiple_of(layout.values) || !cols.end.is_multiple_of(layout.values) {
            return Err(WeightMapError::ColumnsOffBlock {
                id,
                key: key.clone(),
                cols: cols.clone(),
                block_values: layout.values,
            });
        }
    }
    Ok(())
}

/// What a checked [`WeightView::BlockColumns`] cuts: `outer` runs of `extent` values on its axis, each
/// value `inner` elements wide, `extent_product` elements in all; for a packed entry also the encoding
/// block it must fall on.
#[cfg(any(test, feature = "test-support"))]
struct ColumnCut {
    outer: usize,
    extent: usize,
    inner: usize,
    extent_product: usize,
    block_values: Option<usize>,
    block_bytes: Option<usize>,
}

/// [`check_columns`], then the geometry of the cut.
#[cfg(any(test, feature = "test-support"))]
fn column_cut(
    id: WeightId,
    key: &WeightKey,
    entry: &WeightEntry,
    axis: usize,
    cols: &std::ops::Range<usize>,
) -> Result<ColumnCut, WeightMapError> {
    check_columns(id, key, entry, axis, cols)?;
    let shape = entry.shape();
    let layout = match entry {
        WeightEntry::Packed(payload) => match payload.weight().format().descriptor().storage {
            crate::format::Storage::Blocks(layout) => Some(layout),
            crate::format::Storage::Planar(_) => None,
        },
        WeightEntry::Dense(_) => None,
    };
    Ok(ColumnCut {
        outer: shape[..axis].iter().product(),
        extent: shape[axis],
        inner: shape[axis + 1..].iter().product(),
        extent_product: shape.iter().product::<usize>().max(1),
        block_values: layout.map(|layout| layout.values),
        block_bytes: layout.map(|layout| layout.bytes),
    })
}

fn packed_error(id: WeightId) -> impl FnOnce(crate::PackedWeightError) -> WeightMapError {
    move |source| WeightMapError::Packed {
        id,
        source: Box::new(source),
    }
}

fn packed_handle(weight: PackedWeight) -> WeightHandle {
    WeightHandle {
        shape: weight.shape().to_vec(),
        format: HandleFormat::Packed(weight),
    }
}

/// The handle `view` materializes into, checked against the store without copying a byte. A packed
/// view is checked by the payload's shape-only rule ([`PackedPayload::gathered_weight`],
/// [`PackedPayload::concatenated_weight`]) that the operation [`materialize_view`] runs first, so the
/// map and the bytes follow one rule and a check costs no gather.
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
                WeightEntry::Packed(payload) => {
                    PackedPayload::gathered_weight(&[payload.as_ref()], rows.len())
                        .map(packed_handle)
                        .map_err(packed_error(id))
                }
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
                    // `same_format` above admitted only packed parts.
                    let payloads: Vec<&PackedPayload> = parts
                        .iter()
                        .filter_map(|key| match store.get(key.as_str()) {
                            Some(WeightEntry::Packed(payload)) => Some(payload.as_ref()),
                            _ => None,
                        })
                        .collect();
                    PackedPayload::concatenated_weight(&payloads)
                        .map(packed_handle)
                        .map_err(packed_error(id))
                }
            }
        }
        WeightView::BlockColumns { key, axis, cols } => {
            let entry = stored(id, key, store)?;
            check_columns(id, key, entry, *axis, cols)?;
            match entry {
                WeightEntry::Dense(dense) => {
                    let mut shape = dense.shape().to_vec();
                    shape[*axis] = cols.len();
                    Ok(WeightHandle {
                        format: HandleFormat::Dense(dense.dtype()),
                        shape,
                    })
                }
                WeightEntry::Packed(payload) => {
                    let weight = payload.weight();
                    let shape = [weight.shape()[0], cols.len()];
                    PackedWeight::try_new(weight.format(), shape)
                        .map(|weight| WeightHandle {
                            format: HandleFormat::Packed(weight),
                            shape: shape.to_vec(),
                        })
                        .map_err(packed_error(id))
                }
            }
        }
        WeightView::IndexNarrow(key) => {
            let shape = narrowed_index_table(id, key, store)?.shape().to_vec();
            Ok(WeightHandle {
                format: HandleFormat::Dense(poot_tensor::DType::I32),
                shape,
            })
        }
        WeightView::ExpertStack(parts) => {
            let first_key = parts.first().ok_or(WeightMapError::EmptyStack { id })?;
            let first = WeightHandle::of(stored(id, first_key, store)?);
            for key in parts {
                let other = WeightHandle::of(stored(id, key, store)?);
                if other != first {
                    return Err(WeightMapError::NotStackable {
                        id,
                        key: key.clone(),
                        first: Box::new(first),
                        other: Box::new(other),
                    });
                }
            }
            let mut shape = vec![parts.len()];
            shape.extend_from_slice(&first.shape);
            Ok(WeightHandle {
                format: first.format,
                shape,
            })
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
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
                let bytes = dense
                    .bytes()
                    .slice(rows.start * row_bytes..rows.end * row_bytes)
                    .expect("check_row_range keeps the rows inside the entry");
                let mut shape = dense.shape().to_vec();
                shape[0] = rows.len();
                Ok(WeightEntry::Dense(DenseWeight {
                    dtype: dense.dtype(),
                    shape,
                    bytes,
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
                        bytes: StoredBytes::from(bytes),
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
        WeightView::BlockColumns { key, axis, cols } => {
            let entry = stored(id, key, store)?;
            let cut = column_cut(id, key, entry, *axis, cols)?;
            match entry {
                WeightEntry::Dense(dense) => {
                    let source = dense.bytes().as_slice();
                    let unit = source.len() / cut.extent_product;
                    let inner = cut.inner * unit;
                    let mut bytes = Vec::with_capacity(cut.outer * cols.len() * inner);
                    for outer in 0..cut.outer {
                        let row = outer * cut.extent;
                        bytes.extend_from_slice(
                            &source[(row + cols.start) * inner..(row + cols.end) * inner],
                        );
                    }
                    let mut shape = dense.shape().to_vec();
                    shape[*axis] = cols.len();
                    Ok(WeightEntry::Dense(DenseWeight {
                        dtype: dense.dtype(),
                        shape,
                        bytes: StoredBytes::from(bytes),
                    }))
                }
                WeightEntry::Packed(payload) => {
                    let weight = payload.weight();
                    let [rows, _] = weight.shape();
                    let layout = cut
                        .block_values
                        .zip(cut.block_bytes)
                        .expect("a packed column cut names its block layout");
                    let [_, row_bytes] = weight.source_shape(SourceRole::Blocks);
                    let (first, last) = (
                        cols.start / layout.0 * layout.1,
                        cols.end / layout.0 * layout.1,
                    );
                    let source = payload.bytes(SourceRole::Blocks);
                    let mut bytes = Vec::with_capacity(rows * (last - first));
                    for row in 0..rows {
                        bytes.extend_from_slice(
                            &source[row * row_bytes + first..row * row_bytes + last],
                        );
                    }
                    let sliced = PackedWeight::try_new(weight.format(), [rows, cols.len()])
                        .map_err(packed_error(id))?;
                    PackedPayload::try_new(sliced, [(SourceRole::Blocks, StoredBytes::from(bytes))])
                        .map(|sliced| WeightEntry::Packed(Arc::new(sliced)))
                        .map_err(packed_error(id))
                }
            }
        }
        WeightView::IndexNarrow(key) => {
            narrowed_index_table(id, key, store).map(WeightEntry::Dense)
        }
        WeightView::ExpertStack(parts) => {
            // The handle checks every part is of one format and one shape.
            let handle = view_handle(id, view, store)?;
            let HandleFormat::Dense(dtype) = handle.format else {
                return Err(WeightMapError::PackedStackIsNotOneEntry { id });
            };
            let mut bytes = Vec::new();
            for key in parts {
                if let WeightEntry::Dense(dense) = stored(id, key, store)? {
                    bytes.extend_from_slice(dense.bytes().as_slice());
                }
            }
            Ok(WeightEntry::Dense(DenseWeight {
                dtype,
                shape: handle.shape,
                bytes: StoredBytes::from(bytes),
            }))
        }
    }
}

/// `key`'s I64 (or already I32) index table as an I32 table: every value checked against the I32
/// range, the first one outside it named by element index.
fn narrowed_index_table(
    id: WeightId,
    key: &WeightKey,
    store: &WeightStore,
) -> Result<DenseWeight, WeightMapError> {
    use poot_tensor::DType;
    let entry = stored(id, key, store)?;
    let WeightEntry::Dense(dense) = entry else {
        return Err(not_an_index_table(id, key, entry));
    };
    let bytes = dense.bytes().as_slice();
    let narrowed: Vec<u8> = match dense.dtype() {
        DType::I32 => bytes.to_vec(),
        DType::I64 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for (index, word) in bytes.as_chunks::<8>().0.iter().enumerate() {
                let value = i64::from_le_bytes(*word);
                let narrow = i32::try_from(value).map_err(|_| WeightMapError::IndexOutOfRange {
                    id,
                    key: key.clone(),
                    index,
                    value,
                })?;
                out.extend_from_slice(&narrow.to_le_bytes());
            }
            out
        }
        _ => return Err(not_an_index_table(id, key, entry)),
    };
    Ok(DenseWeight {
        dtype: DType::I32,
        shape: dense.shape().to_vec(),
        bytes: StoredBytes::from(narrowed),
    })
}

fn not_an_index_table(id: WeightId, key: &WeightKey, entry: &WeightEntry) -> WeightMapError {
    WeightMapError::NotAnIndexTable {
        id,
        key: key.clone(),
        found: Box::new(WeightHandle::of(entry)),
    }
}

#[cfg(any(test, feature = "test-support"))]
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

    fn bytes(n: usize) -> StoredBytes {
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
            (
                WeightId::layer(2, WeightRole::Lora(LoraRole::A(LoraTarget::AttnQ))),
                "w.l2.lora.a.attn.q",
            ),
            (
                WeightId::layer(2, WeightRole::Lora(LoraRole::B(LoraTarget::FfnDown))),
                "w.l2.lora.b.ffn.down",
            ),
            (
                WeightId::model(WeightRole::Lora(LoraRole::Scale)),
                "w.lora.scale",
            ),
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
                WeightId::layer(1, WeightRole::Attn(AttnRole::Sinks)),
                "w.l1.attn.sinks",
            ),
            (
                WeightId::layer(2, WeightRole::Moe(MoeRole::Router)),
                "w.l2.moe.router",
            ),
            (
                WeightId::layer(2, WeightRole::Moe(MoeRole::RouterBias)),
                "w.l2.moe.router_bias",
            ),
            (
                WeightId::layer(2, WeightRole::Moe(MoeRole::Stack(ExpertProj::GateUp))),
                "w.l2.moe.stack.gate_up",
            ),
            (
                WeightId::layer(2, WeightRole::Moe(MoeRole::Stack(ExpertProj::Down))),
                "w.l2.moe.stack.down",
            ),
            (
                WeightId::layer(2, WeightRole::Moe(MoeRole::StackBias(ExpertProj::Up))),
                "w.l2.moe.stack_bias.up",
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
                WeightId::layer(1, WeightRole::Attn(AttnRole::OA)),
                "w.l1.attn.o_a",
            ),
            (
                WeightId::layer(1, WeightRole::Moe(MoeRole::HashTable)),
                "w.l1.moe.hash_table",
            ),
            (
                WeightId::layer(2, WeightRole::Mhc(MhcRole::AttnFn)),
                "w.l2.mhc.attn_fn",
            ),
            (
                WeightId::layer(2, WeightRole::Mhc(MhcRole::AttnBase)),
                "w.l2.mhc.attn_base",
            ),
            (
                WeightId::layer(2, WeightRole::Mhc(MhcRole::AttnScale)),
                "w.l2.mhc.attn_scale",
            ),
            (
                WeightId::layer(2, WeightRole::Mhc(MhcRole::FfnFn)),
                "w.l2.mhc.ffn_fn",
            ),
            (
                WeightId::layer(2, WeightRole::Mhc(MhcRole::FfnBase)),
                "w.l2.mhc.ffn_base",
            ),
            (
                WeightId::layer(2, WeightRole::Mhc(MhcRole::FfnScale)),
                "w.l2.mhc.ffn_scale",
            ),
            (
                WeightId::model(WeightRole::Mhc(MhcRole::HeadFn)),
                "w.mhc.head_fn",
            ),
            (
                WeightId::model(WeightRole::Mhc(MhcRole::HeadBase)),
                "w.mhc.head_base",
            ),
            (
                WeightId::model(WeightRole::Mhc(MhcRole::HeadScale)),
                "w.mhc.head_scale",
            ),
            (
                WeightId::layer(3, WeightRole::Compress(CompressRole::Kv)),
                "w.l3.compress.kv",
            ),
            (
                WeightId::layer(3, WeightRole::Compress(CompressRole::Gate)),
                "w.l3.compress.gate",
            ),
            (
                WeightId::layer(3, WeightRole::Compress(CompressRole::PositionBias)),
                "w.l3.compress.position_bias",
            ),
            (
                WeightId::layer(3, WeightRole::Compress(CompressRole::Norm)),
                "w.l3.compress.norm",
            ),
            (
                WeightId::layer(3, WeightRole::Compress(CompressRole::IndexerKv)),
                "w.l3.compress.indexer_kv",
            ),
            (
                WeightId::layer(3, WeightRole::Compress(CompressRole::IndexerGate)),
                "w.l3.compress.indexer_gate",
            ),
            (
                WeightId::layer(3, WeightRole::Compress(CompressRole::IndexerPositionBias)),
                "w.l3.compress.indexer_position_bias",
            ),
            (
                WeightId::layer(3, WeightRole::Compress(CompressRole::IndexerNorm)),
                "w.l3.compress.indexer_norm",
            ),
            (
                WeightId::layer(1, WeightRole::Ple(PleRole::TableShard(7))),
                "w.l1.ple.table.shard_7",
            ),
            (
                WeightId::layer(1, WeightRole::Ple(PleRole::TableScale)),
                "w.l1.ple.table_scale",
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

    /// A store of `entries`.
    fn store_of(entries: Vec<(&str, WeightEntry)>) -> WeightStore {
        let mut b = WeightStore::builder();
        for (key, entry) in entries {
            b.insert(key, entry).unwrap();
        }
        b.build()
    }

    /// The map of `views` over `store`.
    fn map_of(store: &WeightStore, views: Vec<(WeightId, WeightView)>) -> WeightMap {
        let mut b = WeightMap::builder(store);
        for (id, view) in views {
            b.map(id, view).unwrap();
        }
        b.build()
    }

    fn row_range(key: &str, rows: std::ops::Range<usize>) -> WeightView {
        WeightView::RowRange {
            key: key.into(),
            rows,
        }
    }

    fn row_stack(keys: &[&str]) -> WeightView {
        WeightView::RowStack(keys.iter().map(|key| WeightKey::from(*key)).collect())
    }

    /// The `StoreMismatch` shapes `map.check_store(forged)` refuses with.
    fn mismatch_shapes(map: &WeightMap, forged: &WeightStore) -> (Vec<usize>, Vec<usize>) {
        match map.check_store(forged) {
            Err(WeightMapError::StoreMismatch { id, mapped, stored }) => {
                assert_eq!(id, K);
                (mapped.shape, stored.shape)
            }
            other => panic!("expected StoreMismatch, got {other:?}"),
        }
    }

    /// The store the map was built over passes, for every view kind the check re-derives.
    ///
    /// Mutation: make the comparison in `check_store` always refuse (`if true || &stored != mapped`);
    /// this row goes red with `Err(StoreMismatch ..)`.
    #[test]
    fn check_store_accepts_the_store_the_map_was_built_over() {
        let store = view_store();
        let map = map_of(
            &store,
            vec![
                (Q, WeightView::Stored("qkv".into())),
                (K, row_range("pqkv", 1..4)),
                (GATE_UP, row_stack(&["pgate", "pup"])),
            ],
        );
        assert_eq!(map.check_store(&store), Ok(()));
    }

    /// A store lacking a key a `RowRange` or `RowStack` view reads is `Missing`, naming the weight and key.
    ///
    /// Mutation: in `check_store`, skip a view `view_handle` refuses (`let Ok(stored) = .. else {
    /// continue }`); this row goes red with `left: Ok(()) right: Err(Missing ..)`.
    #[test]
    fn check_store_refuses_a_store_missing_a_key_a_row_view_reads() {
        let store = view_store();
        let map = map_of(
            &store,
            vec![
                (K, row_range("qkv", 1..3)),
                (GATE_UP, row_stack(&["gate", "up"])),
            ],
        );
        let without = |dropped: &str| {
            let mut b = WeightStore::builder();
            for key in ["qkv", "gate", "up"] {
                if key != dropped {
                    b.insert(key, ramp(6, 2, 0)).unwrap();
                }
            }
            b.build()
        };
        assert_eq!(
            map.check_store(&without("qkv")),
            Err(WeightMapError::Missing {
                id: K,
                key: "qkv".into()
            })
        );
        assert_eq!(
            map.check_store(&without("up")),
            Err(WeightMapError::Missing {
                id: GATE_UP,
                key: "up".into()
            })
        );
    }

    /// A smaller dense entry: a `RowRange` whose rows it no longer holds is `RowRangeOutOfBounds`; one
    /// still inside it but narrower is `StoreMismatch` of the two shapes; a `RowStack` part with fewer
    /// rows changes the stacked total, also `StoreMismatch`.
    ///
    /// Mutation: skip the `stored != mapped` comparison in `check_store`; the narrower and the shorter
    /// stack load (`expected StoreMismatch, got Ok(())`) and these assertions go red.
    #[test]
    fn check_store_refuses_a_smaller_dense_entry() {
        let built = store_of(vec![("a", ramp(6, 2, 0)), ("b", ramp(3, 2, 20))]);
        let ranged = map_of(&built, vec![(K, row_range("a", 2..4))]);
        assert_eq!(
            ranged.check_store(&store_of(vec![("a", ramp(3, 2, 0))])),
            Err(WeightMapError::RowRangeOutOfBounds {
                id: K,
                rows: 2..4,
                available: 3
            })
        );
        assert_eq!(
            mismatch_shapes(&ranged, &store_of(vec![("a", ramp(6, 1, 0))])),
            (vec![2, 2], vec![2, 1])
        );
        let stacked = map_of(&built, vec![(K, row_stack(&["a", "b"]))]);
        assert_eq!(
            mismatch_shapes(
                &stacked,
                &store_of(vec![("a", ramp(6, 2, 0)), ("b", ramp(2, 2, 20))])
            ),
            (vec![9, 2], vec![8, 2])
        );
    }

    /// A packed entry of another `K` than the map recorded, under a `RowRange` or a `RowStack`: the
    /// shape-only handle differs, so `StoreMismatch` names `[rows, 32]` against `[rows, 64]`.
    ///
    /// Mutation: compare only the row counts in `check_store` (`stored.shape[0] != mapped.shape[0]`); the
    /// forged width is not seen (`expected StoreMismatch, got Ok(())`).
    #[test]
    fn check_store_refuses_a_packed_entry_of_another_width() {
        let built = store_of(vec![("a", q4_rows(4, 0x10)), ("b", q4_rows(2, 0x40))]);
        let wide = |rows| WeightEntry::Packed(packed_entry([rows, 64]));
        let ranged = map_of(&built, vec![(K, row_range("a", 1..3))]);
        assert_eq!(
            mismatch_shapes(&ranged, &store_of(vec![("a", wide(4))])),
            (vec![2, 32], vec![2, 64])
        );
        let stacked = map_of(&built, vec![(K, row_stack(&["a", "b"]))]);
        assert_eq!(
            mismatch_shapes(&stacked, &store_of(vec![("a", wide(4)), ("b", wide(2))])),
            (vec![6, 32], vec![6, 64])
        );
    }

    /// An I64 table of `words` stored under `key`.
    fn i64_table_store(key: &str, shape: Vec<usize>, words: &[i64]) -> WeightStore {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut builder = WeightStore::builder();
        builder
            .insert(
                key,
                WeightEntry::Dense(DenseWeight::try_new(DType::I64, shape, bytes.into()).unwrap()),
            )
            .unwrap();
        builder.build()
    }

    /// An I64 index table read through `IndexNarrow` is an I32 table of the same shape and values
    /// (negatives and the I32 bounds included), and the handle says so.
    #[test]
    fn index_narrow_reads_an_i64_table_as_i32() {
        let words = [0i64, 7, -3, i64::from(i32::MAX), i64::from(i32::MIN), 42];
        let store = i64_table_store("t", vec![3, 2], &words);
        let id = WeightId::layer(0, WeightRole::Moe(MoeRole::HashTable));
        let mut builder = WeightMap::builder(&store);
        builder
            .map(id, WeightView::IndexNarrow(WeightKey::from("t")))
            .unwrap();
        let map = builder.build();
        let handle = map.handle(id).unwrap();
        assert_eq!(handle.format, HandleFormat::Dense(DType::I32));
        assert_eq!(handle.shape, vec![3, 2]);
        let WeightEntry::Dense(narrowed) = map.materialize(id, &store).unwrap() else {
            panic!("a narrowed table is dense")
        };
        assert_eq!(
            (narrowed.dtype(), narrowed.shape()),
            (DType::I32, &[3, 2][..])
        );
        let got: Vec<i32> = narrowed
            .bytes()
            .as_slice()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| i32::from_le_bytes(*w))
            .collect();
        assert_eq!(got, words.map(|w| w as i32));
    }

    /// A word outside I32, above or below, is refused when the map is built, naming the table and
    /// the element; so is a table that is not integer-valued. Mutation: narrow with `as i32`; the
    /// out-of-range word wraps, `map` succeeds and the row goes red.
    #[test]
    fn index_narrow_refuses_a_word_outside_i32_at_build() {
        let id = WeightId::layer(2, WeightRole::Moe(MoeRole::HashTable));
        for (words, index, value) in [
            (
                [1i64, 2, i64::from(i32::MAX) + 1, 4],
                2,
                i64::from(i32::MAX) + 1,
            ),
            (
                [i64::from(i32::MIN) - 1, 2, 3, 4],
                0,
                i64::from(i32::MIN) - 1,
            ),
        ] {
            let store = i64_table_store("t", vec![2, 2], &words);
            let error = WeightMap::builder(&store)
                .map(id, WeightView::IndexNarrow(WeightKey::from("t")))
                .map(|_| ())
                .unwrap_err();
            assert_eq!(
                error,
                WeightMapError::IndexOutOfRange {
                    id,
                    key: WeightKey::from("t"),
                    index,
                    value,
                }
            );
        }
        let mut builder = WeightStore::builder();
        builder
            .insert(
                "f",
                WeightEntry::Dense(DenseWeight::try_new(DType::F32, vec![2], bytes(8)).unwrap()),
            )
            .unwrap();
        let store = builder.build();
        let error = WeightMap::builder(&store)
            .map(id, WeightView::IndexNarrow(WeightKey::from("f")))
            .map(|_| ())
            .unwrap_err();
        assert!(
            matches!(error, WeightMapError::NotAnIndexTable { .. }),
            "{error:?}"
        );
    }

    fn block_entry(format: WeightFormat, shape: [usize; 2], _seed: u64) -> WeightEntry {
        let weight = PackedWeight::try_new(format, shape).unwrap();
        let block_bytes = weight.source_bytes(SourceRole::Blocks);
        WeightEntry::Packed(Arc::new(
            PackedPayload::try_new(weight, [(SourceRole::Blocks, bytes(block_bytes))]).unwrap(),
        ))
    }

    /// The experts of a routed projection are packed `[out, K]` weights, each its own payload: the map views
    /// them as one `[experts, out, K]` stack with the experts' packed format (the handle a tracer declares the
    /// stack by), `materialize` refuses to make one entry of them, and `materialize_stack` returns the experts
    /// as one `PackedStack` whose bytes are the stored ones, expert for expert.
    #[test]
    fn packed_experts_are_viewed_as_one_stack_that_stays_separate_payloads() {
        let mut store = WeightStore::builder();
        for e in 0..3 {
            store
                .insert(format!("e{e}"), block_entry(WeightFormat::Q8_0, [3, 32], e))
                .unwrap();
        }
        let store = store.build();
        let keys: Vec<WeightKey> = (0..3)
            .map(|e| WeightKey::from(format!("e{e}").as_str()))
            .collect();
        let mut builder = WeightMap::builder(&store);
        builder.map(Q, WeightView::ExpertStack(keys)).unwrap();
        let map = builder.build();

        let handle = map.handle(Q).unwrap();
        assert_eq!(
            handle.format,
            HandleFormat::Packed(PackedWeight::try_new(WeightFormat::Q8_0, [3, 32]).unwrap())
        );
        assert_eq!(handle.shape, [3, 3, 32]);
        assert_eq!(
            map.materialize(Q, &store).unwrap_err(),
            WeightMapError::PackedStackIsNotOneEntry { id: Q }
        );
        let stack = map.materialize_stack(Q, &store).unwrap();
        assert_eq!(stack.experts(), 3);
        for e in 0..3 {
            let WeightEntry::Packed(stored) = store.get(&format!("e{e}")).unwrap() else {
                unreachable!()
            };
            assert!(
                Arc::ptr_eq(stack.owner(e), stored),
                "expert {e} stays the stored payload, not a copy"
            );
        }
    }

    /// Parts of one stack must share a format and a shape: a different packed format, a different shape, a
    /// dense part among packed ones, and an empty stack are each typed refusals at map time, and
    /// `materialize_stack` refuses a view that is not a packed stack.
    #[test]
    fn a_packed_expert_stack_with_unlike_parts_is_a_typed_refusal() {
        let mut store = WeightStore::builder();
        store
            .insert("q8", block_entry(WeightFormat::Q8_0, [3, 32], 1))
            .unwrap();
        store
            .insert("q8b", block_entry(WeightFormat::Q8_0, [3, 32], 2))
            .unwrap();
        store
            .insert("q4", block_entry(WeightFormat::Q4_0, [3, 32], 3))
            .unwrap();
        store
            .insert("tall", block_entry(WeightFormat::Q8_0, [4, 32], 4))
            .unwrap();
        store
            .insert(
                "dense",
                WeightEntry::Dense(
                    DenseWeight::try_new(DType::F32, vec![3, 32], bytes(3 * 32 * 4)).unwrap(),
                ),
            )
            .unwrap();
        let store = store.build();
        let stack = |parts: &[&str]| {
            WeightView::ExpertStack(parts.iter().map(|p| WeightKey::from(*p)).collect())
        };
        for (label, parts) in [
            ("format", &["q8", "q4"][..]),
            ("shape", &["q8", "tall"][..]),
            ("dense part", &["q8", "dense"][..]),
            ("dense first", &["dense", "q8"][..]),
        ] {
            let error = WeightMap::builder(&store)
                .map(Q, stack(parts))
                .map(|_| ())
                .unwrap_err();
            assert!(
                matches!(error, WeightMapError::NotStackable { id: Q, .. }),
                "{label}: {error:?}"
            );
        }
        assert_eq!(
            WeightMap::builder(&store)
                .map(Q, stack(&[]))
                .map(|_| ())
                .unwrap_err(),
            WeightMapError::EmptyStack { id: Q }
        );
        let mut builder = WeightMap::builder(&store);
        builder.map(Q, stack(&["q8", "q8b"])).unwrap();
        builder.map(K, WeightView::Stored("q8".into())).unwrap();
        let map = builder.build();
        assert_eq!(
            map.materialize_stack(K, &store).unwrap_err(),
            WeightMapError::NotAPackedStack { id: K }
        );
    }

    /// `rows` rows of two Q4_0 blocks each (K = 64), every block's quants distinct, so a column cut
    /// that takes the wrong block, or any block of the wrong row, changes the bytes.
    fn q4_wide(rows: usize) -> WeightEntry {
        let weight = PackedWeight::try_new(WeightFormat::Q4_0, [rows, 64]).unwrap();
        let bytes: Vec<u8> = (0..rows * 2)
            .flat_map(|block| {
                let mut bytes = vec![0x00, 0x3c];
                bytes.extend(std::iter::repeat_n(0x10 + block as u8, 16));
                bytes
            })
            .collect();
        WeightEntry::Packed(Arc::new(
            PackedPayload::try_new(weight, [(SourceRole::Blocks, bytes.into())]).unwrap(),
        ))
    }

    fn column_store() -> WeightStore {
        let mut b = WeightStore::builder();
        b.insert("dense", ramp(3, 4, 0)).unwrap();
        b.insert("packed", q4_wide(3)).unwrap();
        b.insert("stack", ramp(6, 2, 40)).unwrap();
        b.build()
    }

    /// A dense column range keeps each row's own columns, in row order.
    #[test]
    fn dense_column_views_cut_every_row_at_the_same_columns() {
        let store = column_store();
        let mut b = WeightMap::builder(&store);
        b.map(
            Q,
            WeightView::BlockColumns {
                key: "dense".into(),
                axis: 1,
                cols: 1..3,
            },
        )
        .unwrap();
        let map = b.build();
        assert_eq!(map.handle(Q).unwrap().shape, vec![3, 2]);
        let cut = map.materialize(Q, &store).unwrap();
        assert_eq!(WeightHandle::of(&cut), *map.handle(Q).unwrap());
        let whole = entry_bytes(store.get("dense").unwrap());
        let want: Vec<u8> = (0..3)
            .flat_map(|row| whole[(row * 4 + 1) * 4..(row * 4 + 3) * 4].to_vec())
            .collect();
        assert_eq!(entry_bytes(&cut), want);
    }

    /// A packed column range is whole blocks of every row, never a decode.
    #[test]
    fn packed_column_views_cut_whole_blocks_of_every_row() {
        let store = column_store();
        let mut b = WeightMap::builder(&store);
        b.map(
            Q,
            WeightView::BlockColumns {
                key: "packed".into(),
                axis: 1,
                cols: 32..64,
            },
        )
        .unwrap();
        let map = b.build();
        assert_eq!(
            map.handle(Q).unwrap().format,
            HandleFormat::Packed(PackedWeight::try_new(WeightFormat::Q4_0, [3, 32]).unwrap())
        );
        let cut = map.materialize(Q, &store).unwrap();
        assert_eq!(WeightHandle::of(&cut), *map.handle(Q).unwrap());
        let whole = entry_bytes(store.get("packed").unwrap());
        // 18-byte blocks, two per row: the second block of each of three rows.
        let want: Vec<u8> = (0..3)
            .flat_map(|row| whole[(row * 2 + 1) * 18..(row * 2 + 2) * 18].to_vec())
            .collect();
        assert_eq!(entry_bytes(&cut), want);
    }

    /// A packed range off a block boundary is refused when the view is built, and so are the axes a
    /// column range cannot be.
    #[test]
    fn column_views_refuse_a_cut_block_and_the_wrong_axis() {
        let store = column_store();
        let view = |key: &str, axis, cols| WeightView::BlockColumns {
            key: key.into(),
            axis,
            cols,
        };
        let err = |view| {
            WeightMap::builder(&store)
                .map(Q, view)
                .map(|_| ())
                .unwrap_err()
        };
        assert_eq!(
            err(view("packed", 1, 16..48)),
            WeightMapError::ColumnsOffBlock {
                id: Q,
                key: "packed".into(),
                cols: 16..48,
                block_values: 32
            }
        );
        assert_eq!(
            err(view("dense", 0, 0..1)),
            WeightMapError::ColumnAxis {
                id: Q,
                key: "dense".into(),
                axis: 0,
                rank: 2
            }
        );
        assert!(matches!(
            err(view("dense", 1, 2..5)),
            WeightMapError::ColumnRangeOutOfBounds { available: 4, .. }
        ));
        assert!(matches!(
            err(view("packed", 0, 0..1)),
            WeightMapError::ColumnAxis { .. }
        ));
    }

    /// A rank's view of a map: axis 0 is the rank's rows, another axis its columns, composed with a view
    /// that is already a row range; a share the world does not divide is refused by name.
    #[test]
    fn rank_views_give_each_rank_its_equal_share_of_the_named_axis() {
        let store = column_store();
        let mut b = WeightMap::builder(&store);
        b.map(Q, WeightView::Stored("dense".into())).unwrap();
        b.map(K, WeightView::Stored("packed".into())).unwrap();
        b.map(
            GATE_UP,
            WeightView::RowRange {
                key: "stack".into(),
                rows: 2..6,
            },
        )
        .unwrap();
        let map = b.build();
        let world = std::num::NonZeroU32::new(2).unwrap();
        let splits = [(Q, 1), (K, 1), (GATE_UP, 0)];
        let whole = |key: &str| entry_bytes(store.get(key).unwrap());
        for rank in 0..2 {
            let view = map.rank_view(&store, rank, world, &splits).unwrap();
            assert_eq!(
                view.view(Q),
                Some(&WeightView::BlockColumns {
                    key: "dense".into(),
                    axis: 1,
                    cols: rank * 2..rank * 2 + 2
                })
            );
            assert_eq!(view.handle(Q).unwrap().shape, vec![3, 2]);
            let dense = entry_bytes(&view.materialize(Q, &store).unwrap());
            let want: Vec<u8> = (0..3)
                .flat_map(|row| {
                    let start = (row * 4 + rank * 2) * 4;
                    whole("dense")[start..start + 8].to_vec()
                })
                .collect();
            assert_eq!(dense, want, "rank {rank}'s dense columns");
            let packed = entry_bytes(&view.materialize(K, &store).unwrap());
            let want: Vec<u8> = (0..3)
                .flat_map(|row| whole("packed")[(row * 2 + rank) * 18..][..18].to_vec())
                .collect();
            assert_eq!(packed, want, "rank {rank}'s packed blocks");
            assert_eq!(
                view.view(GATE_UP),
                Some(&WeightView::RowRange {
                    key: "stack".into(),
                    rows: 2 + rank * 2..4 + rank * 2
                })
            );
        }
        let three = std::num::NonZeroU32::new(3).unwrap();
        assert_eq!(
            map.rank_view(&store, 0, three, &[(Q, 1)]).unwrap_err(),
            WeightMapError::ShareNotDivisible {
                id: Q,
                axis: 1,
                extent: 4,
                world: 3
            }
        );
        // Three ranks of a 64-value row would cut a 32-value block in two.
        assert!(matches!(
            map.rank_view(&store, 0, std::num::NonZeroU32::new(4).unwrap(), &[(K, 1)])
                .unwrap_err(),
            WeightMapError::ColumnsOffBlock { .. }
        ));
        assert_eq!(
            map.rank_view(&store, 0, world, &[(GATE_UP, 1)])
                .unwrap_err(),
            WeightMapError::ShareOfView {
                id: GATE_UP,
                axis: 1
            }
        );
    }

    /// `DenseWeight::decode_f32` widens each decodable stored dtype through the scalar decoders and
    /// refuses every other dtype with a typed error (the loader's former `decode` contract).
    #[test]
    fn dense_weight_decodes_the_float_dtypes_and_refuses_the_rest() {
        use crate::scalar::f32_to_f16;
        use poot_tensor::DType;

        let dense = |dtype, bytes: Vec<u8>, len| {
            DenseWeight::try_new(dtype, vec![len], StoredBytes::from(bytes)).unwrap()
        };
        let f32s = dense(
            DType::F32,
            [1.5f32, -2.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            2,
        );
        assert_eq!(f32s.decode_f32().unwrap(), vec![1.5, -2.0]);
        let bf16s = dense(
            DType::BF16,
            [0x3f80u16, 0xc000]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            2,
        );
        assert_eq!(bf16s.decode_f32().unwrap(), vec![1.0, -2.0]);
        let f16s = dense(
            DType::F16,
            [f32_to_f16(0.5), f32_to_f16(-3.0)]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            2,
        );
        assert_eq!(f16s.decode_f32().unwrap(), vec![0.5, -3.0]);
        let e4m3 = dense(DType::E4M3FN, vec![0x38, 0xb8], 2);
        assert_eq!(e4m3.decode_f32().unwrap(), vec![1.0, -1.0]);
        let e8m0 = dense(DType::E8M0, vec![127, 128], 2);
        assert_eq!(e8m0.decode_f32().unwrap(), vec![1.0, 2.0]);
        let ints = dense(DType::I32, vec![0; 4], 1);
        assert_eq!(
            ints.decode_f32(),
            Err(UndecodableDtype { dtype: DType::I32 })
        );
    }
}
