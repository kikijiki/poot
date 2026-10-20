//! The [`WeightMap`]: a model's table from a typed [`WeightId`] to a [`WeightView`] over
//! [`WeightStore`] entries and the [`WeightHandle`] a tracer sees (format and logical shape, no bytes).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use poot_quant::weights::{DenseWeight, DenseWeightError, WeightEntry, WeightKey, WeightStore};
use poot_quant::{PackedPayload, PackedWeight, StoredBytes};
#[cfg(any(test, feature = "test-support"))]
use poot_quant::{PackedStack, SourceRole};

use crate::role::WeightId;

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
    /// source to word strides ([`poot_quant::PackedWeight::slab_bytes`]).
    ExpertStack(Vec<WeightKey>),
    /// A leading-axis row range of one entry (a slice of a fused `qkv`).
    RowRange {
        /// The entry the rows are cut from.
        key: WeightKey,
        /// The half-open row range of the entry's leading axis.
        rows: std::ops::Range<usize>,
    },
    /// The leading-axis rows of one entry in the order `order` names: row `i` of the view is stored row
    /// `order[i]`. Nothing is copied until a binder reads the view, which lays each row's bytes down in
    /// that order (llama.cpp's Q/K rows stored permuted for its interleaved RoPE, read back in the
    /// half-split order the model traces). Every index is a row of the entry; an index may repeat or be
    /// left out, so a view of a subset of the rows is one too.
    RowOrder {
        /// The entry the rows are read from.
        key: WeightKey,
        /// For each row of the view, the stored row it reads.
        order: Arc<[usize]>,
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
            | Self::RowOrder { key, .. }
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
    Packed(poot_quant::PackedWeight),
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
    /// A [`WeightView::RowOrder`] that is empty or names a row the entry does not have.
    RowOrderOutOfBounds {
        /// The weight being mapped.
        id: WeightId,
        /// The first stored row the order names past the entry (`0` for an empty order).
        row: usize,
        /// Rows the entry has.
        available: usize,
    },
    /// An entry whose `rows` are not a whole number of heads of an even `head_dim` rows each, so a per-head
    /// row order cannot be built over it.
    HeadRows {
        /// The weight being mapped.
        id: WeightId,
        /// Rows the entry has.
        rows: usize,
        /// Rows per head.
        head_dim: usize,
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
        source: Box<poot_quant::PackedWeightError>,
    },
    /// The bytes a view derived are not the length its shape and dtype require.
    Derived {
        /// The weight being mapped.
        id: WeightId,
        /// The dense entry's own refusal.
        source: Box<DenseWeightError>,
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
            Self::RowOrderOutOfBounds { id, row, available } => write!(
                f,
                "weight {id}: a row order must name at least one row of the entry's {available} rows; row {row} is past them"
            ),
            Self::HeadRows { id, rows, head_dim } => write!(
                f,
                "weight {id}: {rows} rows are not a whole number of heads of {head_dim} (an even count) rows"
            ),
            Self::MixedParts { id, first, other } => write!(
                f,
                "weight {id}: row-stack parts must share one format and row shape: {first:?} vs {other:?}"
            ),
            Self::Packed { id, source } => write!(f, "weight {id}: {source}"),
            Self::Derived { id, source } => write!(f, "weight {id}: derived bytes: {source}"),
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
            Self::Derived { source, .. } => Some(source.as_ref()),
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
                (WeightView::RowOrder { key, order }, 0) => WeightView::RowOrder {
                    key: key.clone(),
                    order: order[range].into(),
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

fn check_row_order(
    id: WeightId,
    key: &WeightKey,
    entry: &WeightEntry,
    order: &[usize],
) -> Result<(), WeightMapError> {
    let available = row_count(id, key, entry)?;
    match order.iter().find(|&&row| row >= available) {
        Some(&row) => Err(WeightMapError::RowOrderOutOfBounds { id, row, available }),
        None if order.is_empty() => Err(WeightMapError::RowOrderOutOfBounds {
            id,
            row: 0,
            available,
        }),
        None => Ok(()),
    }
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
        let poot_quant::format::Storage::Blocks(layout) =
            payload.weight().format().descriptor().storage
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
            poot_quant::format::Storage::Blocks(layout) => Some(layout),
            poot_quant::format::Storage::Planar(_) => None,
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

fn packed_error(id: WeightId) -> impl FnOnce(poot_quant::PackedWeightError) -> WeightMapError {
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
        WeightView::RowOrder { key, order } => {
            let entry = stored(id, key, store)?;
            check_row_order(id, key, entry, order)?;
            match entry {
                WeightEntry::Dense(dense) => {
                    let mut shape = dense.shape().to_vec();
                    shape[0] = order.len();
                    Ok(WeightHandle {
                        format: HandleFormat::Dense(dense.dtype()),
                        shape,
                    })
                }
                WeightEntry::Packed(payload) => {
                    PackedPayload::gathered_weight(&[payload.as_ref()], order.len())
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
                derived(id, dense.dtype(), shape, bytes).map(WeightEntry::Dense)
            }
            entry @ WeightEntry::Packed(payload) => {
                check_row_range(id, key, entry, rows)?;
                let rows: Vec<usize> = rows.clone().collect();
                PackedPayload::gather_rows(&[payload.as_ref()], &rows)
                    .map(|sliced| WeightEntry::Packed(Arc::new(sliced)))
                    .map_err(packed_error(id))
            }
        },
        WeightView::RowOrder { key, order } => {
            let entry = stored(id, key, store)?;
            check_row_order(id, key, entry, order)?;
            match entry {
                WeightEntry::Dense(dense) => {
                    let row_bytes = dense.bytes().len() / dense.shape()[0];
                    let source = dense.bytes().as_slice();
                    let mut bytes = Vec::with_capacity(order.len() * row_bytes);
                    for &row in order.iter() {
                        bytes.extend_from_slice(&source[row * row_bytes..(row + 1) * row_bytes]);
                    }
                    let mut shape = dense.shape().to_vec();
                    shape[0] = order.len();
                    derived(id, dense.dtype(), shape, StoredBytes::from(bytes))
                        .map(WeightEntry::Dense)
                }
                WeightEntry::Packed(payload) => {
                    PackedPayload::gather_rows(&[payload.as_ref()], order)
                        .map(|gathered| WeightEntry::Packed(Arc::new(gathered)))
                        .map_err(packed_error(id))
                }
            }
        }
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
                    derived(id, first.dtype(), shape, StoredBytes::from(bytes))
                        .map(WeightEntry::Dense)
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
                    derived(id, dense.dtype(), shape, StoredBytes::from(bytes))
                        .map(WeightEntry::Dense)
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
            derived(id, dtype, handle.shape, StoredBytes::from(bytes)).map(WeightEntry::Dense)
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
    derived(
        id,
        DType::I32,
        dense.shape().to_vec(),
        StoredBytes::from(narrowed),
    )
}

/// The dense entry a view derives from stored bytes. The view computes `bytes` from `shape` and `dtype`, so
/// the length check can only fail on a bug in the view's own arithmetic, and says which weight it was.
fn derived(
    id: WeightId,
    dtype: poot_tensor::DType,
    shape: Vec<usize>,
    bytes: StoredBytes,
) -> Result<DenseWeight, WeightMapError> {
    DenseWeight::try_new(dtype, shape, bytes).map_err(|source| WeightMapError::Derived {
        id,
        source: Box::new(source),
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
    use crate::role::*;
    use poot_quant::format::WeightFormat;
    use poot_quant::{PackedWeight, SourceRole};
    use poot_tensor::DType;

    fn bytes(n: usize) -> StoredBytes {
        vec![0u8; n].into()
    }

    fn packed_entry(shape: [usize; 2]) -> Arc<PackedPayload> {
        let weight = PackedWeight::try_new(WeightFormat::Q4_0, shape).unwrap();
        let block_bytes = weight.source_bytes(SourceRole::Blocks);
        Arc::new(
            PackedPayload::try_new(weight, [(SourceRole::Blocks, bytes(block_bytes))]).unwrap(),
        )
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
}
