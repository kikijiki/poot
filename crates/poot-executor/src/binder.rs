//! The one role-keyed binder (Card 546a, Z9): matches a step's host inputs to an entry's declared
//! slots by [`SlotKey`], checks each one's shape, element count and dtype lane against the slot's
//! declared [`TensorType`] (Card 620 folded in, R-546-11), and encodes the bytes into the slot's
//! planned device lane. A check here is the contract's only gate: a test may not forge the state it
//! guards, so every row drives this function with inputs where the two sides differ.
//!
//! Weights bind here too (Card 564), by the executable's [`WeightSource`]: under
//! [`WeightSource::Map`] a const's name is a [`WeightId::const_name`] (or a [`PackedSourceName`] over
//! one), the id's [`WeightView`] names byte runs of store entries, and those runs - never a const
//! name - are the weight's residency identity, so two ids viewing one entry share one upload.
//! A `<name>.chunkN` const (legalize's split of an oversized weight, [`Chunk`]) binds the rows of
//! its parent weight `<name>` that the chunks before it leave off, as a row range of the parent's
//! own spans: the parent is never read or uploaded whole.
//! [`WeightSource::ConstNames`] is the name-equality rule held for POOT-739. A hosted embed gather
//! ([`Slot::TokenEmbed`](poot_graph_ir::Slot::TokenEmbed)) reads its rows through the same
//! resolution.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use poot_graph_ir::{PackedSourceName, SlotKey, TensorType, ValueId};
use poot_quant::SourceRole;
use poot_quant::weights::{
    HandleFormat, StoreGeneration, WeightEntry, WeightId, WeightKey, WeightMap, WeightStore,
    WeightView,
};
use poot_target::BufferStorage;
use poot_tensor::{DType, HostView};

use crate::WeightSource;
use crate::error::{BindError, LoadError};

/// One entry slot: the structured key a step input matches by, its declared type, and the device
/// lane the planner chose for it.
pub(crate) struct BoundSlot {
    pub key: SlotKey,
    pub aval: TensorType,
    pub storage: BufferStorage,
    /// Index into the entry's `locals`.
    pub buffer: usize,
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
}

/// How a weight const's stored bytes are encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StoredAs {
    /// Dense elements of this dtype, encoded into the planned lane.
    Dense(DType),
    /// One packed source's raw bytes, uploaded as `u32` words (Card 642).
    PackedSource,
}

/// What one weight const reads: its spans back to back, and how they are stored.
#[derive(Clone, Debug)]
pub(crate) struct ConstBytes {
    pub spans: Vec<StoredSpan>,
    pub stored: StoredAs,
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
}

/// A graph name resolved to the store: the parts its view reads, in order, the format and logical
/// shape they make together, and the packed source role a [`PackedSourceName`] selected.
struct Located {
    parts: Vec<Part>,
    format: HandleFormat,
    shape: Vec<usize>,
    role: Option<SourceRole>,
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
    pub fn width(&self) -> usize {
        self.width
    }
}

/// An executable's weights: its store and the rule its graph consts bind by.
pub(crate) struct WeightBinder {
    store: Arc<WeightStore>,
    rule: Rule,
}

enum Rule {
    /// Each mapped weight under its [`WeightId::const_name`].
    Map {
        map: Arc<WeightMap>,
        ids: HashMap<String, WeightId>,
    },
    ConstNames,
}

impl WeightBinder {
    pub fn new(store: Arc<WeightStore>, source: WeightSource) -> Self {
        let rule = match source {
            WeightSource::Map(map) => {
                let ids = map.iter().map(|(id, _, _)| (id.const_name(), id)).collect();
                Rule::Map { map, ids }
            }
            WeightSource::ConstNames => Rule::ConstNames,
        };
        Self { store, rule }
    }

    fn entry(&self, value: ValueId, name: &str, key: &str) -> Result<&WeightEntry, LoadError> {
        self.store.get(key).ok_or_else(|| LoadError::Unbound {
            value,
            name: name.to_string(),
        })
    }

    /// Resolve graph name `name` (of input `value`) to its view over the store. Under
    /// [`Rule::Map`] only a mapped id binds: a name that is a store key but no const name of the map
    /// is [`LoadError::Unbound`] (SC-003).
    fn locate(&self, value: ValueId, name: &str) -> Result<Located, LoadError> {
        let unbound = || LoadError::Unbound {
            value,
            name: name.to_string(),
        };
        match &self.rule {
            Rule::Map { map, ids } => {
                let (id, role) = match ids.get(name) {
                    Some(&id) => (id, None),
                    None => {
                        let source = PackedSourceName::parse(name).ok_or_else(unbound)?;
                        let id = *ids.get(source.linear_id()).ok_or_else(unbound)?;
                        (id, Some(source.role()))
                    }
                };
                let (Some(view), Some(handle)) = (map.view(id), map.handle(id)) else {
                    return Err(unbound());
                };
                let parts = match view {
                    WeightView::Stored(key) => vec![Part::Whole(key.clone())],
                    WeightView::RowStack(keys) => keys.iter().cloned().map(Part::Whole).collect(),
                    WeightView::RowRange { key, rows } => {
                        vec![Part::Rows(key.clone(), rows.clone())]
                    }
                };
                Ok(Located {
                    parts,
                    format: handle.format,
                    shape: handle.shape.clone(),
                    role,
                })
            }
            Rule::ConstNames => {
                let (key, role) = match PackedSourceName::parse(name) {
                    Some(source) => (source.linear_id().to_string(), Some(source.role())),
                    None => (name.to_string(), None),
                };
                let (format, shape) = match self.entry(value, name, &key)? {
                    WeightEntry::Dense(dense) => {
                        (HandleFormat::Dense(dense.dtype()), dense.shape().to_vec())
                    }
                    WeightEntry::Packed(payload) => (
                        HandleFormat::Packed(payload.weight()),
                        payload.weight().shape().to_vec(),
                    ),
                };
                Ok(Located {
                    parts: vec![Part::Whole(WeightKey::from(key))],
                    format,
                    shape,
                    role,
                })
            }
        }
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
        })
    }

    /// The stored bytes const `name` (input `value`, declared `aval`) binds: a dense const reads its
    /// view's elements as stored, a [`PackedSourceName`] const one source of a packed view.
    pub fn const_bytes(
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
                StoredAs::PackedSource
            }
            _ => return Err(wrong_role()),
        };
        let spans = located
            .parts
            .iter()
            .map(|part| self.span(value, name, part, located.role))
            .collect::<Result<Vec<_>, _>>()?;
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

    /// `part`'s byte run in its entry (in source `role` of a packed one). A row run is whole stored
    /// rows: a dense row's elements, or a block format's run of `K / block_values` blocks (a planar
    /// source has no byte rows, and a map refuses a row range of one).
    fn span(
        &self,
        value: ValueId,
        name: &str,
        part: &Part,
        role: Option<SourceRole>,
    ) -> Result<StoredSpan, LoadError> {
        let key = match part {
            Part::Whole(key) | Part::Rows(key, _) => key,
        };
        let (len, row_bytes) = match (self.entry(value, name, key.as_str())?, role) {
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
        let bytes = match part {
            Part::Whole(_) => 0..len,
            Part::Rows(_, rows) => {
                let row_bytes =
                    row_bytes.ok_or(LoadError::Unimplemented("a row range of a planar source"))?;
                rows.start * row_bytes..rows.end * row_bytes
            }
        };
        if bytes.end > len {
            return Err(LoadError::WeightShape {
                name: name.to_string(),
                stored: vec![len],
                declared: vec![bytes.end],
            });
        }
        let generation = self
            .store
            .generation(key.as_str())
            .ok_or_else(|| LoadError::Unbound {
                value,
                name: name.to_string(),
            })?;
        Ok(StoredSpan {
            key: key.clone(),
            generation,
            role,
            bytes,
        })
    }

    /// `spans`' stored bytes, back to back.
    pub fn read(&self, spans: &[StoredSpan]) -> Vec<u8> {
        let mut out = Vec::with_capacity(spans.iter().map(|span| span.bytes.len()).sum());
        for span in spans {
            let stored = match (self.store.get(span.key.as_str()), span.role) {
                (Some(WeightEntry::Dense(dense)), None) => dense.bytes().as_slice(),
                (Some(WeightEntry::Packed(payload)), Some(role)) => payload.bytes(role),
                _ => unreachable!("a span is built from the store it reads"),
            };
            out.extend_from_slice(&stored[span.bytes.clone()]);
        }
        out
    }

    /// Where the hosted embed gather named `name` (the `Slot::TokenEmbed` input `value`, declared
    /// `aval`, planned in `storage`) reads its rows: the embed view's `[rows, width]` table. A dense
    /// table's rows encode into `storage` as stored; a packed one's decode to F32 rows.
    pub fn embed_rows(
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
                aval.dtype == dtype && crate::lanes::encode_stored(dtype, &[], storage).is_ok()
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
    pub fn embed(
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
                    payload
                        .decode_row(stored_row, &mut decoded[start..])
                        .map_err(|source| BindError::RowDecode {
                            name: rows.name.clone(),
                            source,
                        })?;
                }
                None => unreachable!("embed rows are resolved against the store they read"),
            }
        }
        let encoded = match rows.format {
            HandleFormat::Dense(dtype) => crate::lanes::encode_stored(dtype, &dense, storage).ok(),
            HandleFormat::Packed(_) => {
                crate::lanes::encode_host(DType::F32, bytemuck::cast_slice(&decoded), storage)
            }
        };
        encoded.ok_or_else(|| BindError::TokenEmbedEncode {
            name: rows.name.clone(),
            planned: storage,
        })
    }
}
