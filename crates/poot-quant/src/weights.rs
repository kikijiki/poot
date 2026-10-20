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

use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::collections::btree_map;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{PackedPayload, SourceChanged, StoredBytes};

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
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UndecodableDtype {
    /// The stored dtype that has no f32 decode.
    pub dtype: poot_tensor::DType,
}

#[cfg(any(test, feature = "test-support"))]
impl fmt::Display for UndecodableDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stored dtype {:?} has no f32 decode", self.dtype)
    }
}

#[cfg(any(test, feature = "test-support"))]
impl std::error::Error for UndecodableDtype {}

#[cfg(any(test, feature = "test-support"))]
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
