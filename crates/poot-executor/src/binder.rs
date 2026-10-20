//! The one role-keyed binder (Card 546a, Z9): matches a step's host inputs to an entry's declared
//! slots by [`SlotKey`], checks each one's shape, element count and dtype lane against the slot's
//! declared [`TensorType`] (Card 620 folded in, R-546-11), and encodes the bytes into the slot's
//! planned device lane. A check here is the contract's only gate: a test may not forge the state it
//! guards, so every row drives this function with inputs where the two sides differ.
//!
//! Weights bind here too (Card 564), through the executable's weight map: a const's name is a [`WeightId::const_name`] (or a [`PackedSourceName`] over
//! one), the id's [`WeightView`] names byte runs of store entries, and those runs - never a const
//! name - are the weight's residency identity, so two ids viewing one entry share one upload.
//! A packed expert stack ([`WeightView::ExpertStack`] over packed entries) binds one const per source role:
//! the experts' sources back to back, each padded to whole `u32` words ([`PackedWeight::slab_bytes`]) so a
//! kernel finds expert `e` at word `e * slab_words`.
//! A `<name>.chunkN` const (legalize's split of an oversized weight, [`Chunk`]) binds the rows of
//! its parent weight `<name>` that the chunks before it leave off, as a row range of the parent's
//! own spans: the parent is never read or uploaded whole. A chunk of a packed expert stack's source (legalize's
//! split of a stack over the buffer limit) is a run of whole experts, padded to the slab stride like the whole.
//! A hosted embed gather ([`Slot::TokenEmbed`](poot_graph_ir::Slot::TokenEmbed)) reads its rows
//! through the same resolution.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use poot_graph_ir::{
    IndexRange, PackedSourceName, SlotKey, TensorType, ValueId, packed_stack_source_type,
};
use poot_quant::SourceRole;
use poot_quant::weights::{StoreGeneration, WeightEntry, WeightKey, WeightStore};
use poot_target::BufferStorage;
use poot_tensor::{DType, HostView};
use poot_weights::{HandleFormat, WeightId, WeightMap, WeightView};

use crate::error::{BindError, LoadError};
use crate::lanes::{EncodeError, Encoding, first_non_finite_f16};

/// One entry slot: the structured key a step input matches by, its declared type, and the device
/// lane the planner chose for it.
pub(crate) struct BoundSlot {
    pub key: SlotKey,
    pub aval: TensorType,
    pub storage: BufferStorage,
    /// Index into the entry's `locals`.
    pub buffer: usize,
    /// The range the graph declares for this slot's I32 words ([`poot_graph_ir::ValueMeta::range`]):
    /// checked at every bind, so the planner and the graph's consumers may rely on it.
    pub range: Option<IndexRange>,
}

/// Check one step input against its matched slot and encode it into `slot.storage`'s lane.
///
/// Shape is checked first (SC-011: a multi-dimensional mismatch that may still agree on total
/// element count, Card 620's regression), then element count independent of shape (SC-012: a flat
/// buffer whose total length disagrees even when a weaker check might only compare a shape prefix),
/// then the dtype lane (SC-006): never converts a host dtype the slot does not declare.
pub(crate) fn check_and_encode(
    slot: &BoundSlot,
    shape: &[usize],
    view: HostView<'_>,
) -> Result<Vec<u8>, BindError> {
    if shape != slot.aval.shape.as_slice() {
        return Err(BindError::Shape {
            key: slot.key.clone(),
            expected: slot.aval.shape.clone(),
            got: shape.to_vec(),
        });
    }
    let expected = slot.aval.numel();
    if view.elems() != expected {
        return Err(BindError::ElementCount {
            key: slot.key.clone(),
            expected,
            got: view.elems(),
        });
    }
    if view.dtype() != slot.aval.dtype {
        return Err(BindError::Lane {
            key: slot.key.clone(),
            expected: slot.aval.dtype,
            got: view.dtype(),
        });
    }
    if let Some(range) = slot.range {
        // The lane check above made this an I32 view: its bytes are little-endian words.
        for (index, word) in view.bytes().as_chunks::<4>().0.iter().enumerate() {
            let value = i32::from_le_bytes(*word);
            if !range.contains(value) {
                return Err(BindError::Range {
                    key: slot.key.clone(),
                    index,
                    value,
                    range,
                });
            }
        }
    }
    crate::lanes::encode_host(view.dtype(), view.bytes(), slot.storage).ok_or_else(|| {
        BindError::Lane {
            key: slot.key.clone(),
            expected: slot.aval.dtype,
            got: view.dtype(),
        }
    })
}

/// One run of stored bytes: the byte range `bytes` of store entry `key` (of its source `role`, for a
/// packed entry), as inserted under store generation `generation`. A weight's spans, with its
/// planned storage, are its residency identity: what it uploads, never the const name or `WeightId`
/// that reached it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StoredSpan {
    pub key: WeightKey,
    pub generation: StoreGeneration,
    pub role: Option<SourceRole>,
    pub bytes: Range<usize>,
    /// The bytes are an I64 table read as I32 ([`WeightView::IndexNarrow`]): [`WeightWriter`]
    /// narrows each word. The map range-checked the table against this very entry when it was built.
    pub narrowed: bool,
}

/// How a weight const's stored bytes are encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StoredAs {
    /// Dense elements of this dtype, encoded into the planned lane.
    Dense(DType),
    /// One packed source's raw bytes, uploaded as `u32` words (Card 642).
    PackedSource,
    /// One source role of an expert-stacked packed table: one span per expert, each padded with zeros to
    /// `slab_bytes` (a whole number of words) before the upload, so the experts' slabs sit at fixed word strides.
    PackedStackSource { slab_bytes: usize },
}

/// What one weight const reads: its spans back to back, and how they are stored.
#[derive(Clone, Debug)]
pub(crate) struct ConstBytes {
    pub spans: Vec<StoredSpan>,
    pub stored: StoredAs,
}

impl ConstBytes {
    /// The length of the stored stream (a [`WeightWriter`]'s source), from the spans alone: a planning pass
    /// sizes an upload without reading the store.
    pub(crate) fn stored_len(&self) -> usize {
        match self.stored {
            // Every expert's run is padded to the slab stride.
            StoredAs::PackedStackSource { slab_bytes } => self.spans.len() * slab_bytes,
            StoredAs::Dense(_) | StoredAs::PackedSource => self
                .spans
                .iter()
                .map(|span| {
                    // A narrowed I64 table reads back as I32: half the bytes.
                    if span.narrowed {
                        span.bytes.len() / 2
                    } else {
                        span.bytes.len()
                    }
                })
                .sum(),
        }
    }
}

/// The stored bytes `span` names, as the store holds them (an I64 table is still I64 here).
fn stored_bytes<'s>(store: &'s WeightStore, span: &StoredSpan) -> &'s [u8] {
    let stored = match (store.get(span.key.as_str()), span.role) {
        (Some(WeightEntry::Dense(dense)), None) => dense.bytes().as_slice(),
        (Some(WeightEntry::Packed(payload)), Some(role)) => payload.bytes(role),
        _ => unreachable!("a span is built from the store it reads"),
    };
    &stored[span.bytes.clone()]
}

impl StoredSpan {
    /// The bytes this span contributes to the stored stream: a narrowed I64 table is half as long.
    fn data_len(&self) -> usize {
        if self.narrowed {
            self.bytes.len() / 2
        } else {
            self.bytes.len()
        }
    }
}

/// One weight const on its way to the device: its spans, back to back (a stacked source's runs each
/// followed by zeros to the slab stride, so expert `e` starts at byte `e * slab_bytes`), landed in the planned
/// lane by [`lanes::encode_range`](crate::lanes::encode_range). It reads the store's views straight into
/// the buffers a [`Device::write_with`](crate::Device::write_with) hands out: no intermediate copy of the
/// weight is built (dmap.md section 6.1).
pub(crate) struct WeightWriter {
    /// The store the spans read, shared with the binder: the writer borrows nothing, so the engine builds it
    /// once per weight and keeps it across the device allocation.
    store: Arc<WeightStore>,
    spans: Vec<StoredSpan>,
    /// Each span's first byte in the stored stream.
    starts: Vec<usize>,
    encoding: Encoding,
    shape: Vec<usize>,
    /// The stored stream's length without a stack's last slab padding.
    data_len: usize,
}

impl WeightWriter {
    /// The bytes the device is handed (the encoded length; what the lane holds past it is its zeros).
    pub(crate) fn len(&self) -> usize {
        crate::lanes::encoded_len(self.encoding, &self.shape, self.data_len)
    }

    /// A packed-F16 weight holding an infinity or NaN is refused before any byte is written: the
    /// generated bodies decode the packed binary16 lane over its finite domain only (Card 1007).
    pub(crate) fn check_encodable(&self) -> Result<(), EncodeError> {
        if self.encoding != Encoding::PackedWordsFiniteF16 {
            return Ok(());
        }
        let mut base = 0;
        for span in &self.spans {
            if let Some((index, bits)) = first_non_finite_f16(stored_bytes(&self.store, span)) {
                return Err(EncodeError::NonFiniteF16 {
                    index: base + index,
                    bits,
                });
            }
            base += span.bytes.len() / 2;
        }
        Ok(())
    }

    /// Encoded bytes `range` into `out` (`out.len() == range.len()`): any sub-range, since every encoding
    /// is element- or row-local. Writes every byte of `out`.
    pub(crate) fn fill(&self, range: Range<usize>, out: &mut [u8]) {
        crate::lanes::encode_range(
            self.encoding,
            &self.shape,
            self.data_len,
            range,
            &mut |stored, into| self.read_stream(stored, into),
            out,
        );
    }

    /// Stored-stream bytes `range` into `out`; a slab gap is zeros.
    fn read_stream(&self, range: Range<usize>, out: &mut [u8]) {
        out.fill(0);
        let first = self
            .starts
            .partition_point(|&start| start <= range.start)
            .saturating_sub(1);
        for (span, &start) in self.spans.iter().zip(&self.starts).skip(first) {
            if start >= range.end {
                break;
            }
            let end = start + span.data_len();
            let (lo, hi) = (range.start.max(start), range.end.min(end));
            if lo >= hi {
                continue;
            }
            let into = &mut out[lo - range.start..hi - range.start];
            let stored = stored_bytes(&self.store, span);
            if span.narrowed {
                // Output word `w` of the I32 stream is stored word `w` of the I64 table.
                for (offset, byte) in (lo - start..hi - start).zip(into.iter_mut()) {
                    let (word, lane) = (offset / 4, offset % 4);
                    let wide =
                        i64::from_le_bytes(stored[word * 8..word * 8 + 8].try_into().unwrap());
                    let value = i32::try_from(wide)
                        .expect("WeightMapBuilder::map range-checked this table against the store");
                    *byte = value.to_le_bytes()[lane];
                }
            } else {
                into.copy_from_slice(&stored[lo - start..hi - start]);
            }
        }
    }
}

/// Where a `<base>.chunkN` const sits in its parent weight `base`: the parent's leading-axis rows
/// the const holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Chunk {
    pub base: String,
    pub rows: Range<usize>,
}

/// `name` as `(base, N)` when it is spelled `<base>.chunkN`, the name legalize gives chunk `N` of an
/// oversized weight `base`.
fn chunk_name(name: &str) -> Option<(&str, usize)> {
    let (base, index) = name.rsplit_once(".chunk")?;
    let parsed: usize = index.parse().ok()?;
    (!base.is_empty() && parsed.to_string() == index).then_some((base, parsed))
}

/// The [`Chunk`] of every chunk const among `consts` (`(value, name, declared shape)`): a parent's
/// chunks tile its rows in index order, each as tall as its declared leading axis. Chunk indices
/// that are not dense from 0 are a graph no binder can place, refused by name.
pub(crate) fn chunks<'a>(
    consts: impl IntoIterator<Item = (ValueId, &'a str, &'a [usize])>,
) -> Result<HashMap<ValueId, Chunk>, LoadError> {
    let mut by_base: HashMap<&str, Vec<(usize, ValueId, usize)>> = HashMap::new();
    for (value, name, shape) in consts {
        if let Some((base, index)) = chunk_name(name) {
            let height = shape.first().copied().unwrap_or(0);
            by_base
                .entry(base)
                .or_default()
                .push((index, value, height));
        }
    }
    let mut placed = HashMap::new();
    for (base, mut members) in by_base {
        members.sort_unstable();
        let mut start = 0;
        for (position, (index, value, height)) in members.into_iter().enumerate() {
            if index != position {
                return Err(LoadError::Unbound {
                    value,
                    name: format!("{base}.chunk{index}: chunk {position} of {base} is missing"),
                });
            }
            let rows = start..start + height;
            start = rows.end;
            placed.insert(
                value,
                Chunk {
                    base: base.to_string(),
                    rows,
                },
            );
        }
    }
    Ok(placed)
}

/// One part of a view: a whole stored entry, or a run of its leading-axis rows.
#[derive(Clone, Debug)]
enum Part {
    Whole(WeightKey),
    Rows(WeightKey, Range<usize>),
    /// The rows of an entry in the order given, one byte run per row.
    Ordered(WeightKey, Arc<[usize]>),
    /// A whole I64 index table read as I32.
    Narrowed(WeightKey),
    /// The range `cols` of axis `axis` (at least 1) of an entry: one byte run per outer index.
    Columns(WeightKey, usize, Range<usize>),
}

/// A graph name resolved to the store: the parts its view reads, in order, the format and logical
/// shape they make together, and the packed source role a [`PackedSourceName`] selected.
struct Located {
    parts: Vec<Part>,
    format: HandleFormat,
    shape: Vec<usize>,
    role: Option<SourceRole>,
    /// The expert count when the view is an [`WeightView::ExpertStack`] (`parts` is one per expert).
    experts: Option<usize>,
}

/// One logical row run of a hosted embed table: rows `rows` of store entry `key`.
#[derive(Clone, Debug)]
struct RowRun {
    key: WeightKey,
    rows: Range<usize>,
}

/// Where a hosted embed gather ([`Slot::TokenEmbed`](poot_graph_ir::Slot::TokenEmbed)) reads its
/// rows: the embed view's row runs in order, the row width, and how a row is stored.
#[derive(Clone, Debug)]
pub(crate) struct EmbedRows {
    name: String,
    runs: Vec<RowRun>,
    width: usize,
    format: HandleFormat,
}

impl EmbedRows {
    /// The table's row width: a slot of `n` elements holds `n / width` tokens.
    pub(crate) fn width(&self) -> usize {
        self.width
    }
}

/// An executable's weights: its store and the map its graph consts bind through, each mapped weight
/// under its [`WeightId::const_name`].
pub(crate) struct WeightBinder {
    store: Arc<WeightStore>,
    map: Arc<WeightMap>,
    ids: HashMap<String, WeightId>,
}

impl WeightBinder {
    pub(crate) fn new(store: Arc<WeightStore>, map: Arc<WeightMap>) -> Self {
        let ids = map.iter().map(|(id, _, _)| (id.const_name(), id)).collect();
        Self { store, map, ids }
    }

    /// The store's mapped checkpoint files are still the files that were loaded (one `fstat` per
    /// file): the boundary before a bind batch reads the store's views.
    pub(crate) fn check_sources(&self) -> Result<(), LoadError> {
        Ok(self.store.check_sources()?)
    }

    fn entry(&self, value: ValueId, name: &str, key: &str) -> Result<&WeightEntry, LoadError> {
        self.store.get(key).ok_or_else(|| LoadError::Unbound {
            value,
            name: name.to_string(),
        })
    }

    /// Resolve graph name `name` (of input `value`) to its view over the store. Only a mapped id
    /// binds: a name that is a store key but no const name of the map is [`LoadError::Unbound`]
    /// (SC-003).
    fn locate(&self, value: ValueId, name: &str) -> Result<Located, LoadError> {
        let unbound = || LoadError::Unbound {
            value,
            name: name.to_string(),
        };
        let (id, role) = match self.ids.get(name) {
            Some(&id) => (id, None),
            None => {
                let source = PackedSourceName::parse(name).ok_or_else(unbound)?;
                let id = *self.ids.get(source.linear_id()).ok_or_else(unbound)?;
                (id, Some(source.role()))
            }
        };
        let (Some(view), Some(handle)) = (self.map.view(id), self.map.handle(id)) else {
            return Err(unbound());
        };
        let parts = match view {
            WeightView::Stored(key) => vec![Part::Whole(key.clone())],
            WeightView::RowStack(keys) | WeightView::ExpertStack(keys) => {
                keys.iter().cloned().map(Part::Whole).collect()
            }
            WeightView::RowRange { key, rows } => {
                vec![Part::Rows(key.clone(), rows.clone())]
            }
            WeightView::RowOrder { key, order } => {
                vec![Part::Ordered(key.clone(), Arc::clone(order))]
            }
            WeightView::IndexNarrow(key) => vec![Part::Narrowed(key.clone())],
            WeightView::BlockColumns { key, axis, cols } => {
                vec![Part::Columns(key.clone(), *axis, cols.clone())]
            }
        };
        let experts = matches!(view, WeightView::ExpertStack(_)).then_some(parts.len());
        Ok(Located {
            parts,
            format: handle.format,
            shape: handle.shape.clone(),
            role,
            experts,
        })
    }

    /// Resolve chunk const `name` (of input `value`, declared `aval`) to the rows `chunk.rows` of its
    /// parent: the parent view's parts, cut to those logical rows. Only a dense parent splits (legalize
    /// only splits a dense const), and the chunk must be as tall as its rows and as wide as the parent.
    fn locate_chunk(
        &self,
        value: ValueId,
        name: &str,
        aval: &TensorType,
        chunk: &Chunk,
    ) -> Result<Located, LoadError> {
        let parent = self.locate(value, &chunk.base)?;
        // A packed expert stack's source is `[experts, slab_bytes]`: its leading axis is the experts, one part
        // each, so a chunk is a run of whole experts.
        if let (HandleFormat::Packed(weight), Some(role), Some(experts)) =
            (parent.format, parent.role, parent.experts)
        {
            let stack = packed_stack_source_type(weight, experts, role);
            if chunk.rows.end > experts
                || aval.shape.first() != Some(&chunk.rows.len())
                || aval.shape[1..] != stack.shape[1..]
            {
                return Err(LoadError::WeightShape {
                    name: name.to_string(),
                    stored: stack.shape,
                    declared: aval.shape.clone(),
                });
            }
            return Ok(Located {
                parts: parent.parts[chunk.rows.clone()].to_vec(),
                experts: Some(chunk.rows.len()),
                ..parent
            });
        }
        let HandleFormat::Dense(_) = parent.format else {
            return Err(LoadError::UnboundPackedRole {
                value,
                name: name.to_string(),
            });
        };
        let parent_rows = parent.shape.first().copied().unwrap_or(0);
        let shape_error = || LoadError::WeightShape {
            name: name.to_string(),
            stored: parent.shape.clone(),
            declared: aval.shape.clone(),
        };
        if parent.role.is_some()
            || chunk.rows.end > parent_rows
            || aval.shape.first() != Some(&chunk.rows.len())
            || aval.shape[1..] != parent.shape[1..]
        {
            return Err(shape_error());
        }
        let mut parts = Vec::new();
        let mut held = 0;
        for part in &parent.parts {
            let (key, rows) = match part {
                Part::Rows(key, rows) => (key, rows.clone()),
                Part::Whole(key) => {
                    let total = self.entry(value, name, key.as_str())?.shape()[0];
                    (key, 0..total)
                }
                // A narrowed table is read whole: a chunk of it would split a derived buffer.
                Part::Narrowed(_) => return Err(shape_error()),
                // An ordered view is a gather of rows, not a row range.
                Part::Ordered(..) => return Err(shape_error()),
                // A column cut is a strided set of runs, not a row range.
                Part::Columns(..) => return Err(shape_error()),
            };
            let span = held..held + rows.len();
            held = span.end;
            let start = chunk.rows.start.max(span.start);
            let end = chunk.rows.end.min(span.end);
            if start < end {
                let first = rows.start + (start - span.start);
                parts.push(Part::Rows(key.clone(), first..first + (end - start)));
            }
        }
        Ok(Located {
            parts,
            format: parent.format,
            shape: aval.shape.clone(),
            role: None,
            experts: None,
        })
    }

    /// The stored bytes const `name` (input `value`, declared `aval`) binds: a dense const reads its
    /// view's elements as stored, a [`PackedSourceName`] const one source of a packed view.
    pub(crate) fn const_bytes(
        &self,
        value: ValueId,
        name: &str,
        aval: &TensorType,
        chunk: Option<&Chunk>,
    ) -> Result<ConstBytes, LoadError> {
        let located = match chunk {
            Some(chunk) => self.locate_chunk(value, name, aval, chunk)?,
            None => self.locate(value, name)?,
        };
        let wrong_role = || LoadError::UnboundPackedRole {
            value,
            name: name.to_string(),
        };
        let stored = match (located.format, located.role) {
            (HandleFormat::Dense(dtype), None) => {
                if located.shape != aval.shape {
                    return Err(LoadError::WeightShape {
                        name: name.to_string(),
                        stored: located.shape,
                        declared: aval.shape.clone(),
                    });
                }
                StoredAs::Dense(dtype)
            }
            (HandleFormat::Packed(weight), Some(role)) if weight.sources().contains(&role) => {
                match located.experts {
                    None => StoredAs::PackedSource,
                    Some(experts) => {
                        let expected = packed_stack_source_type(weight, experts, role);
                        if *aval != expected {
                            return Err(LoadError::WeightShape {
                                name: name.to_string(),
                                stored: expected.shape,
                                declared: aval.shape.clone(),
                            });
                        }
                        StoredAs::PackedStackSource {
                            slab_bytes: weight.slab_bytes(role),
                        }
                    }
                }
            }
            _ => return Err(wrong_role()),
        };
        let spans = located
            .parts
            .iter()
            .map(|part| self.spans(value, name, part, located.role))
            .collect::<Result<Vec<_>, _>>()?
            .concat();
        // A stacked source's length was checked against its padded type above; each expert's own bytes were
        // checked against the store entry by `span`.
        if stored == StoredAs::PackedSource {
            let len: usize = spans.iter().map(|span| span.bytes.len()).sum();
            if len != aval.numel() {
                return Err(LoadError::WeightShape {
                    name: name.to_string(),
                    stored: vec![len],
                    declared: aval.shape.clone(),
                });
            }
        }
        Ok(ConstBytes { spans, stored })
    }

    /// `part`'s byte runs in its entry (in source `role` of a packed one). A row run is whole stored
    /// rows: a dense row's elements, or a block format's run of `K / block_values` blocks (a planar
    /// source has no byte rows, and a map refuses a row range of one). A column range is one run per
    /// outer index: a dense entry's elements, or a block format's blocks of each row.
    fn spans(
        &self,
        value: ValueId,
        name: &str,
        part: &Part,
        role: Option<SourceRole>,
    ) -> Result<Vec<StoredSpan>, LoadError> {
        let key = match part {
            Part::Whole(key)
            | Part::Rows(key, _)
            | Part::Ordered(key, _)
            | Part::Narrowed(key)
            | Part::Columns(key, ..) => key,
        };
        let entry = self.entry(value, name, key.as_str())?;
        let (len, row_bytes) = match (entry, role) {
            (WeightEntry::Dense(dense), None) => {
                let len = dense.bytes().len();
                (len, dense.shape().first().map(|&rows| len / rows.max(1)))
            }
            (WeightEntry::Packed(payload), Some(role))
                if payload.weight().sources().contains(&role) =>
            {
                let weight = payload.weight();
                let row_bytes = (role == SourceRole::Blocks).then(|| weight.source_shape(role)[1]);
                (weight.source_bytes(role), row_bytes)
            }
            _ => {
                return Err(LoadError::UnboundPackedRole {
                    value,
                    name: name.to_string(),
                });
            }
        };
        let runs: Vec<Range<usize>> = match part {
            Part::Whole(_) | Part::Narrowed(_) => std::iter::once(0..len).collect(),
            Part::Rows(_, rows) => {
                let row_bytes =
                    row_bytes.ok_or(LoadError::Unimplemented("a row range of a planar source"))?;
                std::iter::once(rows.start * row_bytes..rows.end * row_bytes).collect()
            }
            Part::Ordered(_, order) => {
                let row_bytes =
                    row_bytes.ok_or(LoadError::Unimplemented("a row order of a planar source"))?;
                order
                    .iter()
                    .map(|&row| row * row_bytes..(row + 1) * row_bytes)
                    .collect()
            }
            Part::Columns(_, axis, cols) => match (entry, role) {
                (WeightEntry::Dense(dense), None) => {
                    let shape = dense.shape();
                    let numel: usize = shape.iter().product::<usize>().max(1);
                    let unit = len / numel;
                    let outer: usize = shape[..*axis].iter().product();
                    let extent = shape[*axis];
                    let inner: usize = shape[*axis + 1..].iter().product::<usize>() * unit;
                    (0..outer)
                        .map(|o| {
                            let row = o * extent;
                            (row + cols.start) * inner..(row + cols.end) * inner
                        })
                        .collect()
                }
                (WeightEntry::Packed(payload), Some(SourceRole::Blocks)) => {
                    let weight = payload.weight();
                    let poot_quant::format::Storage::Blocks(layout) =
                        weight.format().descriptor().storage
                    else {
                        return Err(LoadError::Unimplemented(
                            "a column range of a planar source",
                        ));
                    };
                    let row_bytes = row_bytes.expect("a blocks source has byte rows");
                    let first = cols.start / layout.values * layout.bytes;
                    let last = cols.end / layout.values * layout.bytes;
                    (0..weight.shape()[0])
                        .map(|row| row * row_bytes + first..row * row_bytes + last)
                        .collect()
                }
                _ => {
                    return Err(LoadError::Unimplemented(
                        "a column range of a planar source",
                    ));
                }
            },
        };
        let generation = self
            .store
            .generation(key.as_str())
            .ok_or_else(|| LoadError::Unbound {
                value,
                name: name.to_string(),
            })?;
        runs.into_iter()
            .map(|bytes| {
                if bytes.end > len {
                    return Err(LoadError::WeightShape {
                        name: name.to_string(),
                        stored: vec![len],
                        declared: vec![bytes.end],
                    });
                }
                Ok(StoredSpan {
                    key: key.clone(),
                    generation,
                    role,
                    bytes,
                    narrowed: matches!(part, Part::Narrowed(_)),
                })
            })
            .collect()
    }

    /// Every word of the I32 const `name` (stored as `bytes`) lies in `range`, the range its graph
    /// declares ([`poot_graph_ir::Builder::constant_ranged`]): the check that makes the declaration a
    /// fact the graph's consumers may rely on, as [`check_and_encode`] is for a slot. The first word
    /// outside is refused by its row-major index, with the range.
    pub(crate) fn check_range(
        &self,
        name: &str,
        bytes: &ConstBytes,
        range: IndexRange,
    ) -> Result<(), LoadError> {
        let StoredAs::Dense(dtype) = bytes.stored else {
            return Err(LoadError::Unimplemented(
                "a declared range on a packed weight source",
            ));
        };
        if dtype != DType::I32 {
            return Err(LoadError::WeightDtype {
                name: name.to_string(),
                stored: dtype,
                declared: DType::I32,
            });
        }
        let mut base = 0;
        for span in &bytes.spans {
            let stored = stored_bytes(&self.store, span);
            // A narrowed span is an I64 table whose map already range-checked every word into I32.
            let words: Box<dyn Iterator<Item = i32> + '_> = if span.narrowed {
                Box::new(stored.as_chunks::<8>().0.iter().map(|word| {
                    i32::try_from(i64::from_le_bytes(*word))
                        .expect("WeightMapBuilder::map range-checked this table against the store")
                }))
            } else {
                Box::new(
                    stored
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|word| i32::from_le_bytes(*word)),
                )
            };
            let mut len = 0;
            for (index, value) in words.enumerate() {
                len += 1;
                if !range.contains(value) {
                    return Err(LoadError::WeightRange {
                        name: name.to_string(),
                        index: base + index,
                        value,
                        range,
                    });
                }
            }
            base += len;
        }
        Ok(())
    }

    /// The writer that lands `bytes` (a const of logical `shape`, planned in `storage`) in its device
    /// lane, a byte range at a time. A dense const with no encoding into `storage` is
    /// [`EncodeError::NoEncoding`].
    pub(crate) fn writer(
        &self,
        bytes: &ConstBytes,
        shape: &[usize],
        storage: BufferStorage,
    ) -> Result<WeightWriter, EncodeError> {
        let encoding = match bytes.stored {
            StoredAs::Dense(dtype) => {
                crate::lanes::encoding_of(dtype, storage).ok_or(EncodeError::NoEncoding)?
            }
            // A packed source is its stored bytes as little-endian `u32` words (dquant.md D9).
            StoredAs::PackedSource | StoredAs::PackedStackSource { .. } => Encoding::PackedWords,
        };
        let mut starts = Vec::with_capacity(bytes.spans.len());
        let mut next = 0;
        for span in &bytes.spans {
            starts.push(next);
            next += match bytes.stored {
                StoredAs::PackedStackSource { slab_bytes } => slab_bytes,
                _ => span.data_len(),
            };
        }
        // A stack's last slab is padded to the stride: that padding is the allocation's zeros, not data.
        let data_len = bytes
            .spans
            .last()
            .map_or(0, |last| starts[starts.len() - 1] + last.data_len());
        Ok(WeightWriter {
            store: Arc::clone(&self.store),
            spans: bytes.spans.clone(),
            starts,
            encoding,
            shape: shape.to_vec(),
            data_len,
        })
    }

    /// Where the hosted embed gather named `name` (the `Slot::TokenEmbed` input `value`, declared
    /// `aval`, planned in `storage`) reads its rows: the embed view's `[rows, width]` table. A dense
    /// table's rows encode into `storage` as stored; a packed one's decode to F32 rows.
    pub(crate) fn embed_rows(
        &self,
        value: ValueId,
        name: &str,
        aval: &TensorType,
        storage: BufferStorage,
    ) -> Result<EmbedRows, LoadError> {
        let located = self.locate(value, name)?;
        let shape_error = || LoadError::WeightShape {
            name: name.to_string(),
            stored: located.shape.clone(),
            declared: aval.shape.clone(),
        };
        let &[_, width] = located.shape.as_slice() else {
            return Err(shape_error());
        };
        if located.role.is_some() || width == 0 || !aval.numel().is_multiple_of(width) {
            return Err(shape_error());
        }
        let encodes = match located.format {
            HandleFormat::Dense(dtype) => {
                // The finite-F16 lane is refused: it is the one encoding that can fail on a row's data, and
                // `embed` has no per-token error for it. (No hosted embed slot is planned in it.)
                aval.dtype == dtype
                    && crate::lanes::encoding_of(dtype, storage).is_some_and(|encoding| {
                        encoding != crate::lanes::Encoding::PackedWordsFiniteF16
                    })
            }
            HandleFormat::Packed(_) => {
                aval.dtype == DType::F32
                    && crate::lanes::encode_host(DType::F32, &[], storage).is_some()
            }
        };
        if !encodes {
            return Err(LoadError::WeightFormat {
                name: name.to_string(),
                stored: format!("{:?}", located.format),
                planned: storage,
            });
        }
        let runs = located
            .parts
            .into_iter()
            .map(|part| match part {
                Part::Rows(key, rows) => Ok(RowRun { key, rows }),
                Part::Whole(key) => {
                    let rows = self.entry(value, name, key.as_str())?.shape()[0];
                    Ok(RowRun { key, rows: 0..rows })
                }
                // An index table is no embedding, and a column cut is no row run.
                Part::Narrowed(_) | Part::Ordered(..) | Part::Columns(..) => Err(shape_error()),
            })
            .collect::<Result<Vec<_>, LoadError>>()?;
        Ok(EmbedRows {
            name: name.to_string(),
            runs,
            width,
            format: located.format,
        })
    }

    /// The embed rows of `tokens`, in order, encoded into `storage`.
    pub(crate) fn embed(
        &self,
        rows: &EmbedRows,
        tokens: &[i32],
        storage: BufferStorage,
    ) -> Result<Vec<u8>, BindError> {
        let total: usize = rows.runs.iter().map(|run| run.rows.len()).sum();
        let mut dense = Vec::new();
        let mut decoded = Vec::new();
        for &token in tokens {
            let mut row = usize::try_from(token)
                .ok()
                .filter(|&row| row < total)
                .ok_or_else(|| BindError::TokenOutOfRange {
                    name: rows.name.clone(),
                    token,
                    rows: total,
                })?;
            let run = rows
                .runs
                .iter()
                .find(|run| {
                    let held = run.rows.len();
                    if row < held {
                        return true;
                    }
                    row -= held;
                    false
                })
                .expect("a row below the total lies in some run");
            let stored_row = run.rows.start + row;
            match self.store.get(run.key.as_str()) {
                Some(WeightEntry::Dense(entry)) => {
                    let bytes = entry.bytes().as_slice();
                    let row_bytes = bytes.len() / entry.shape()[0];
                    dense.extend_from_slice(
                        &bytes[stored_row * row_bytes..(stored_row + 1) * row_bytes],
                    );
                }
                Some(WeightEntry::Packed(payload)) => {
                    let start = decoded.len();
                    decoded.resize(start + rows.width, 0.0f32);
                    // `load_weights` matched the payload to the map's `[rows, width]`, and `embed_rows` cut the
                    // runs inside its rows, so the decode cannot leave the payload's shape.
                    // Guarded by `WeightMap::check_store`'s tests (`check_store_refuses_a_packed_entry_of_another_width`).
                    payload
                        .decode_row(stored_row, &mut decoded[start..])
                        .expect("a mapped row of a checked payload decodes");
                }
                None => unreachable!("load_weights checked every mapped key against this store"),
            }
        }
        // `embed_rows` admitted this lane for these rows: a dense table's encoding exists and cannot fail on
        // its data (the finite-F16 lane is not admitted), and a packed table's F32 rows land verbatim.
        Ok(match rows.format {
            HandleFormat::Dense(dtype) => {
                crate::lanes::encode_stored(dtype, &[rows.width], &dense, storage)
                    .expect("embed_rows admitted a lane whose encoding cannot fail on data")
            }
            HandleFormat::Packed(_) => {
                crate::lanes::encode_host(DType::F32, bytemuck::cast_slice(&decoded), storage)
                    .expect("embed_rows admitted a lane that holds F32 rows")
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use poot_quant::format::WeightFormat;
    use poot_quant::weights::{WeightEntry, WeightStore};
    use poot_quant::{PackedPayload, PackedWeight, SourceRole};
    use poot_weights::{MoeRole, WeightMap, WeightRole, WeightView};

    use super::*;

    /// The stored stream of a packed const: its spans back to back, a stack's runs padded to the slab stride
    /// (what [`WeightWriter`] reads, whole).
    fn stored_stream(binder: &WeightBinder, bytes: &ConstBytes) -> Vec<u8> {
        let writer = binder
            .writer(bytes, &[], BufferStorage::i32())
            .expect("a packed source has an encoding");
        let mut out = vec![0u8; bytes.stored_len()];
        writer.read_stream(0..out.len(), &mut out);
        out
    }

    /// Three Q8_0 experts of `out = 3, K = 32`: 102 bytes each (3 rows of one 34-byte block), not a whole
    /// number of `u32` words, viewed as the stack `id`.
    fn q8_stack() -> (WeightBinder, WeightId, Vec<Arc<PackedPayload>>) {
        let id = WeightId::layer(
            0,
            WeightRole::Moe(MoeRole::Stack(poot_weights::ExpertProj::Gate)),
        );
        let experts: Vec<Arc<PackedPayload>> = (0..3)
            .map(|e| {
                Arc::new(poot_quant::test_support::packed::random_payload(
                    WeightFormat::Q8_0,
                    [3, 32],
                    17 + e,
                ))
            })
            .collect();
        let mut store = WeightStore::builder();
        for (e, expert) in experts.iter().enumerate() {
            store
                .insert(format!("gate.{e}"), WeightEntry::Packed(Arc::clone(expert)))
                .unwrap();
        }
        let store = Arc::new(store.build());
        let mut map = WeightMap::builder(&store);
        map.map(
            id,
            WeightView::ExpertStack(
                (0..3)
                    .map(|e| WeightKey::from(format!("gate.{e}").as_str()))
                    .collect(),
            ),
        )
        .unwrap();
        let map = Arc::new(map.build());
        (WeightBinder::new(store, map), id, experts)
    }

    /// SC-002: a per-expert Q8_0 table with `out = 3, K = 32` (102 bytes per expert, not word-aligned) binds as
    /// one const per source role whose bytes are the experts back to back, each padded with zeros to the word
    /// stride (104 bytes), so expert `e` starts at byte `104 * e`. The padded type is the graph's declared
    /// carrier type; the declared per-expert type is refused by name. Mutation: drop the per-expert padding in
    /// [`WeightWriter::read_stream`]; the table is 306 bytes, expert 1 starts at byte 102 and every offset check goes
    /// red.
    #[test]
    fn an_unaligned_expert_table_binds_one_const_with_word_padded_slabs() {
        let (binder, id, experts) = q8_stack();
        let weight = PackedWeight::try_new(WeightFormat::Q8_0, [3, 32]).unwrap();
        let role = SourceRole::Blocks;
        assert_eq!(weight.source_bytes(role), 102);
        assert_eq!(weight.slab_bytes(role), 104);
        let name = PackedSourceName::new(&id.const_name(), role);
        let declared = packed_stack_source_type(weight, 3, role);
        let bytes = binder
            .const_bytes(ValueId::new(0), name.as_str(), &declared, None)
            .expect("the stacked source binds");
        assert_eq!(
            bytes.stored,
            StoredAs::PackedStackSource { slab_bytes: 104 }
        );
        assert_eq!(
            bytes.spans.len(),
            3,
            "one run per expert, one const per role"
        );
        let table = stored_stream(&binder, &bytes);
        assert_eq!(table.len(), 3 * 104);
        assert_eq!(bytes.stored_len(), table.len());
        for (e, expert) in experts.iter().enumerate() {
            let slab = &table[e * 104..(e + 1) * 104];
            assert_eq!(&slab[..102], expert.bytes(role), "expert {e}");
            assert_eq!(&slab[102..], [0, 0], "expert {e}'s word padding");
        }

        let per_expert = poot_graph_ir::packed_source_type(weight, role);
        assert!(matches!(
            binder.const_bytes(ValueId::new(0), name.as_str(), &per_expert, None),
            Err(LoadError::WeightShape { .. })
        ));
    }

    /// POOT-1100 LP-4: a stack's writer fills any chunking of its range with the transport layout of
    /// `packed_source_words` over the slab-padded stream: each expert at its `slab_bytes` stride, the word
    /// padding between slabs zero, however the destination range is cut. Every chunk is handed pre-filled with
    /// garbage, so a gap the writer leaves unwritten shows.
    /// Mutation: drop the `out.fill(0)` that opens `WeightWriter::read_stream`; the gaps keep the garbage and
    /// this row goes red.
    #[test]
    fn a_stack_writer_fills_any_chunking_with_zero_gaps() {
        let (binder, id, experts) = q8_stack();
        let weight = PackedWeight::try_new(WeightFormat::Q8_0, [3, 32]).unwrap();
        let role = SourceRole::Blocks;
        let name = PackedSourceName::new(&id.const_name(), role);
        let declared = packed_stack_source_type(weight, 3, role);
        let bytes = binder
            .const_bytes(ValueId::new(0), name.as_str(), &declared, None)
            .expect("the stacked source binds");
        let writer = binder
            .writer(&bytes, &[], BufferStorage::i32())
            .expect("a packed source has an encoding");
        let transport: Vec<u8> = poot_quant::packed_source_words(&stored_stream(&binder, &bytes))
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let len = writer.len();
        assert!(len <= transport.len());
        assert!(
            transport[len..].iter().all(|&b| b == 0),
            "what the writer leaves out is the allocation's zeros"
        );
        for chunk in [1usize, 3, 7, 104, 1000] {
            let mut out = vec![0xAAu8; len];
            let mut at = 0;
            while at < len {
                let n = chunk.min(len - at);
                writer.fill(at..at + n, &mut out[at..at + n]);
                at += n;
            }
            assert_eq!(out, transport[..len], "chunks of {chunk}");
        }
        for (e, expert) in experts.iter().enumerate() {
            let mut slab = vec![0xAAu8; 104];
            writer.fill(e * 104..(e + 1) * 104, &mut slab);
            assert_eq!(&slab[..102], expert.bytes(role), "expert {e}");
            assert_eq!(&slab[102..], [0, 0], "expert {e}'s word padding");
        }
    }

    /// POOT-1100: an I64 table read through `IndexNarrow` is written as the I32 table it narrows to, in any
    /// chunking, including a chunk that starts in the middle of a word (negatives and the I32 bounds included).
    /// Mutation: write the low byte of every narrowed word at lane 0 regardless of `offset % 4` (drop the `lane`
    /// index in `read_stream`); every chunking that cuts a word goes red.
    #[test]
    fn a_narrowed_index_table_is_written_as_i32_in_any_chunking() {
        let words = [0i64, 7, -3, i64::from(i32::MAX), i64::from(i32::MIN), 42];
        let raw: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut store = WeightStore::builder();
        store
            .insert(
                "t",
                WeightEntry::Dense(
                    poot_quant::weights::DenseWeight::try_new(DType::I64, vec![3, 2], raw.into())
                        .unwrap(),
                ),
            )
            .unwrap();
        let store = Arc::new(store.build());
        let id = WeightId::layer(0, WeightRole::Moe(MoeRole::HashTable));
        let mut map = WeightMap::builder(&store);
        map.map(id, WeightView::IndexNarrow(WeightKey::from("t")))
            .unwrap();
        let binder = WeightBinder::new(Arc::clone(&store), Arc::new(map.build()));
        let declared = TensorType::new(vec![3, 2], DType::I32);
        let bytes = binder
            .const_bytes(ValueId::new(0), &id.const_name(), &declared, None)
            .expect("the narrowed table binds");
        let writer = binder
            .writer(&bytes, &[3, 2], BufferStorage::i32())
            .expect("an I32 table lands verbatim");
        let expected: Vec<u8> = words
            .iter()
            .flat_map(|&w| (w as i32).to_le_bytes())
            .collect();
        assert_eq!(writer.len(), expected.len());
        for chunk in [1usize, 3, 4, 5, 24] {
            let mut out = vec![0xAAu8; expected.len()];
            let mut at = 0;
            while at < out.len() {
                let n = chunk.min(out.len() - at);
                writer.fill(at..at + n, &mut out[at..at + n]);
                at += n;
            }
            assert_eq!(out, expected, "chunks of {chunk}");
        }
    }

    /// SC-004 (card 1067): a `{source}.chunkN` const of a packed expert stack (legalize's split of a table over the
    /// buffer limit) binds the run of whole experts the chunks before it leave off, padded to the slab stride like
    /// the whole table: chunk 0 of a 3-expert stack split 2 + 1 is experts 0 and 1, chunk 1 is expert 2 alone and
    /// starts at byte 0 of its own buffer. A chunk whose declared height disagrees with the experts it is
    /// placed over, or that runs past the stack, is refused by name.
    /// Mutation: in `locate_chunk` start the run one expert early (`chunk.rows.start.saturating_sub(1)`). Red:
    /// `assertion `left == right` failed: chunk 1  left: 208  right: 104` (the chunk holds two experts, not one).
    #[test]
    fn a_chunk_of_an_expert_stack_binds_its_run_of_experts() {
        let (binder, id, experts) = q8_stack();
        let weight = PackedWeight::try_new(WeightFormat::Q8_0, [3, 32]).unwrap();
        let role = SourceRole::Blocks;
        let parent = PackedSourceName::new(&id.const_name(), role).to_string();
        let chunk_names = [format!("{parent}.chunk0"), format!("{parent}.chunk1")];
        let shape_of = |experts: usize| packed_stack_source_type(weight, experts, role);
        let declared = [shape_of(2), shape_of(1)];
        let placed = chunks(chunk_names.iter().zip(&declared).enumerate().map(
            |(value, (name, aval))| {
                (
                    ValueId::new(value as u32),
                    name.as_str(),
                    aval.shape.as_slice(),
                )
            },
        ))
        .unwrap();
        for (index, (name, aval)) in chunk_names.iter().zip(&declared).enumerate() {
            let index = ValueId::new(index as u32);
            let bytes = binder
                .const_bytes(index, name, aval, placed.get(&index))
                .expect("the chunk binds");
            assert_eq!(
                bytes.stored,
                StoredAs::PackedStackSource { slab_bytes: 104 }
            );
            let table = stored_stream(&binder, &bytes);
            let first = [0, 2][index.index()];
            assert_eq!(table.len(), aval.shape[0] * 104, "chunk {index}");
            for (offset, expert) in experts[first..first + aval.shape[0]].iter().enumerate() {
                let slab = &table[offset * 104..(offset + 1) * 104];
                assert_eq!(
                    &slab[..102],
                    expert.bytes(role),
                    "chunk {index}, expert {}",
                    first + offset
                );
                assert_eq!(&slab[102..], [0, 0]);
            }
        }

        // A chunk declared taller than the experts left past its start is refused, never clamped.
        let tall = shape_of(2);
        let past_the_end = Chunk {
            base: parent.clone(),
            rows: 2..4,
        };
        assert!(matches!(
            binder.const_bytes(ValueId::new(0), &chunk_names[1], &tall, Some(&past_the_end)),
            Err(LoadError::WeightShape { .. })
        ));
    }
}
