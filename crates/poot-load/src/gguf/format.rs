//! GGUF metadata, tensor table, and container I/O.

use std::collections::{HashMap, HashSet};

use poot_quant::StoredBytes;
use poot_quant::format::WeightFormat;

use crate::{GgufSplitError, LoadError};

/// A GGUF metadata value (the 13 GGUF value types; `Array` holds a homogeneous list, read only for the
/// tables and short per-layer lists, see `GgufIndex::metadata`).
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    /// An unsigned 8-bit integer.
    U8(u8),
    /// A signed 8-bit integer.
    I8(i8),
    /// An unsigned 16-bit integer.
    U16(u16),
    /// A signed 16-bit integer.
    I16(i16),
    /// An unsigned 32-bit integer.
    U32(u32),
    /// A signed 32-bit integer.
    I32(i32),
    /// An unsigned 64-bit integer.
    U64(u64),
    /// A signed 64-bit integer.
    I64(i64),
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit float.
    F64(f64),
    /// A boolean.
    Bool(bool),
    /// A UTF-8 string.
    Str(String),
    /// A homogeneous list of values.
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Best-effort integer view (any unsigned/signed scalar) - for hyperparams stored across int widths.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::I8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::I16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::I32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I64(v) => Some(*v as u64),
            _ => None,
        }
    }
    /// The value as f32 when it is an `F32` or `F64` (the latter narrowed); `None` for every other type.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    /// The string when the value is a `Str`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s),
            _ => None,
        }
    }
    /// The boolean when the value is a `Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    /// The elements when the value is an `Array`.
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// One tensor's info from the GGUF table. `dims` are ggml `ne` order (fastest-varying first, i.e. the
/// REVERSE of the logical row-major shape); `offset` is relative to the tensor-data section start.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Extents in ggml `ne` order, fastest-varying first.
    pub dims: Vec<u64>,
    /// The ggml tensor type id (see [`crate::gguf::weight_format_of`]).
    pub ggml_type: u32,
    /// Byte offset of the tensor's data from the start of its shard's tensor-data section.
    pub offset: u64,
    /// Which shard of the set holds the tensor: always 0 for a single file.
    pub shard: usize,
}

/// The GGUF value-type id of an array.
const GGUF_TYPE_ARRAY: u32 = 9;

/// `general.alignment` when a header does not declare one (the GGUF spec's default).
const DEFAULT_ALIGNMENT: usize = 32;

/// The metadata arrays of vocabulary size that readers consume, with the element types each may
/// carry (GGUF type ids: 4 u32, 5 i32, 6 f32, 8 string): the tokenizer tables. Any other array
/// longer than [`MAX_LAYER_ARRAY_LEN`] is bounds-checked and skipped, never stored, and the small
/// per-layer arrays are held to [`MAX_LAYER_ARRAY_TOTAL`] elements per file, so no header makes the
/// parser build a `GgufValue` per byte of the file.
const STORED_ARRAYS: [(&str, &[u32]); 4] = [
    ("tokenizer.ggml.tokens", &[8]),
    ("tokenizer.ggml.merges", &[8]),
    ("tokenizer.ggml.scores", &[6]),
    ("tokenizer.ggml.token_type", &[4, 5]),
];

/// Most elements a stored array may declare (4M, about 16x the largest real vocabulary), bounding
/// the decoded size independently of the file's size.
pub(crate) const MAX_STORED_ARRAY_LEN: usize = 1 << 22;

/// Longest array stored under a non-tokenizer key: the per-layer tables (`sliding_window_pattern`,
/// `compress_ratios`, per-layer head counts) have one element per layer, and no model has 4096 layers.
const MAX_LAYER_ARRAY_LEN: usize = 4096;

/// Elements all non-tokenizer arrays of one file may store between them.
const MAX_LAYER_ARRAY_TOTAL: usize = 1 << 16;

/// Maximum tensor rank accepted from a GGUF header (ggml caps at GGML_MAX_DIMS = 4). Bounds
/// `Vec::with_capacity(n_dims)` against a crafted multi-GB allocation.
pub(crate) const MAX_TENSOR_DIMS: usize = 8;

/// Maximum tensor element count accepted from a GGUF header (~1.1e12): ~500x the largest real
/// tensor (~2-3e9 for a 405B-class model) and far below usize-multiply-wrap and `vec!`
/// capacity-overflow thresholds.
pub(crate) const MAX_TENSOR_ELEMS: usize = 1 << 40;

/// A little-endian byte cursor over the GGUF file.
pub(crate) struct Cursor<'a> {
    pub(crate) b: &'a [u8],
    pub(crate) p: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], LoadError> {
        let end = self
            .p
            .checked_add(n)
            .ok_or_else(|| gerr("offset overflow"))?;
        if end > self.b.len() {
            return Err(gerr("unexpected end of file"));
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    pub(crate) fn u32(&mut self) -> Result<u32, LoadError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, LoadError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// A GGUF string: u64 length + UTF-8 bytes.
    pub(crate) fn gstr(&mut self) -> Result<String, LoadError> {
        let n = self.u64()? as usize;
        let raw = self.take(n)?;
        String::from_utf8(raw.to_vec()).map_err(|_| gerr("non-utf8 string"))
    }
    /// An array's element type id and count, the count refused when the remaining bytes cannot hold
    /// it: each element occupies at least its fixed width (a string at least its length prefix), so
    /// a header cannot claim more elements than the file has bytes for. Nested arrays are refused.
    fn array_header(&mut self) -> Result<(u32, usize), LoadError> {
        let elem_ty = self.u32()?;
        if elem_ty == 9 {
            return Err(gerr("nested GGUF arrays are not supported"));
        }
        let n = usize::try_from(self.u64()?).map_err(|_| gerr("array length overflows usize"))?;
        let min_width = match elem_ty {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            _ => 8,
        };
        let remaining = self.b.len() - self.p;
        if n.checked_mul(min_width).is_none_or(|need| need > remaining) {
            return Err(gerr("array length exceeds the bytes remaining"));
        }
        Ok((elem_ty, n))
    }

    /// The `n` elements of type `elem_ty` that follow an [`Self::array_header`], decoded.
    fn array_elements(&mut self, elem_ty: u32, n: usize) -> Result<GgufValue, LoadError> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.value(elem_ty)?);
        }
        Ok(GgufValue::Array(v))
    }

    /// Step over the `n` elements of type `elem_ty` that follow an [`Self::array_header`] without
    /// decoding or storing them.
    fn skip_elements(&mut self, elem_ty: u32, n: usize) -> Result<(), LoadError> {
        match elem_ty {
            8 => {
                for _ in 0..n {
                    let len = usize::try_from(self.u64()?)
                        .map_err(|_| gerr("string length overflows usize"))?;
                    self.take(len)?;
                }
            }
            0 | 1 | 7 => drop(self.take(n)?),
            2 | 3 => drop(self.take(n * 2)?),
            4..=6 => drop(self.take(n * 4)?),
            10..=12 => drop(self.take(n * 8)?),
            other => return Err(gerr(&format!("unknown GGUF value type {other}"))),
        }
        Ok(())
    }

    /// The array metadata value for `key` at the cursor, or `None` for one no reader consumes, which
    /// is stepped over unstored (see [`STORED_ARRAYS`]). `layer_budget` counts the elements
    /// stored under small per-layer keys across the whole header and is spent here.
    pub(crate) fn array(
        &mut self,
        key: &str,
        layer_budget: &mut usize,
    ) -> Result<Option<GgufValue>, LoadError> {
        let (elem_ty, n) = self.array_header()?;
        if let Some((_, allowed)) = STORED_ARRAYS.iter().find(|(stored, _)| *stored == key) {
            if !allowed.contains(&elem_ty) {
                return Err(gerr(&format!(
                    "{key} has array element type {elem_ty}, expected one of {allowed:?}"
                )));
            }
            if n > MAX_STORED_ARRAY_LEN {
                return Err(gerr(&format!(
                    "{key} has {n} elements, more than the {MAX_STORED_ARRAY_LEN} a tokenizer table holds"
                )));
            }
            return self.array_elements(elem_ty, n).map(Some);
        }
        if n > MAX_LAYER_ARRAY_LEN {
            self.skip_elements(elem_ty, n)?;
            return Ok(None);
        }
        *layer_budget = layer_budget.checked_sub(n).ok_or_else(|| {
            gerr(&format!(
                "{key}: metadata arrays outside the tokenizer tables declare more than \
                 {MAX_LAYER_ARRAY_TOTAL} elements"
            ))
        })?;
        self.array_elements(elem_ty, n).map(Some)
    }

    /// A typed scalar or string given its GGUF value-type id.
    pub(crate) fn value(&mut self, ty: u32) -> Result<GgufValue, LoadError> {
        Ok(match ty {
            0 => GgufValue::U8(self.take(1)?[0]),
            1 => GgufValue::I8(self.take(1)?[0] as i8),
            2 => GgufValue::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => GgufValue::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.u32()? as i32),
            6 => GgufValue::F32(f32::from_bits(self.u32()?)),
            7 => GgufValue::Bool(self.take(1)?[0] != 0),
            8 => GgufValue::Str(self.gstr()?),
            9 => return Err(gerr("an array is read through Cursor::array or skip_array")),
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(i64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            12 => GgufValue::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => return Err(gerr(&format!("unknown GGUF value type {other}"))),
        })
    }
}

pub(crate) fn gerr(msg: &str) -> LoadError {
    LoadError::SafeTensors(format!("gguf: {msg}"))
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn gguf_gstr_bytes(s: &str) -> Vec<u8> {
    let mut v = (s.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(s.as_bytes());
    v
}

/// Serialize a [`GgufValue`] to (type-id, body-bytes) for the writer. Supports the types
/// `Cursor::value` accepts; arrays must be homogeneous.
#[cfg(any(test, feature = "test-support"))]
pub(crate) fn gguf_encode_value(val: &GgufValue) -> (u32, Vec<u8>) {
    match val {
        GgufValue::U32(x) => (4, x.to_le_bytes().to_vec()),
        GgufValue::F32(x) => (6, x.to_le_bytes().to_vec()),
        GgufValue::Bool(b) => (7, vec![*b as u8]),
        GgufValue::Str(s) => (8, gguf_gstr_bytes(s)),
        GgufValue::U64(x) => (10, x.to_le_bytes().to_vec()),
        GgufValue::Array(items) => {
            let mut b = gguf_encode_value(&items[0]).0.to_le_bytes().to_vec(); // homogeneous elem type
            b.extend((items.len() as u64).to_le_bytes());
            for it in items {
                b.extend(gguf_encode_value(it).1);
            }
            (9, b)
        }
        other => panic!("gguf writer: unsupported value {other:?}"),
    }
}

/// Build a GGUF v3 byte buffer from metadata KVs and tensors `(name, ggml-ne-dims, ggml_type,
/// raw_bytes)`, the inverse of [`GgufIndex::parse`] plus [`crate::gguf::read_gguf`]. Tensor data offsets are aligned to 32. A
/// fixture utility (poot is not a converter) for building small in-memory GGUFs in tests.
#[cfg(any(test, feature = "test-support"))]
pub fn write_gguf(
    kvs: &[(&str, GgufValue)],
    tensors: &[(&str, Vec<u64>, u32, Vec<u8>)],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend((3u32).to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs {
        out.extend(gguf_gstr_bytes(k));
        let (ty, body) = gguf_encode_value(v);
        out.extend(ty.to_le_bytes());
        out.extend(body);
    }
    let mut offsets = Vec::new();
    let mut off = 0u64;
    for (_, _, _, data) in tensors {
        offsets.push(off);
        off = (off + data.len() as u64).div_ceil(32) * 32;
    }
    for (i, (name, dims, ty, _)) in tensors.iter().enumerate() {
        out.extend(gguf_gstr_bytes(name));
        out.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(ty.to_le_bytes());
        out.extend(offsets[i].to_le_bytes());
    }
    while !out.len().is_multiple_of(32) {
        out.push(0); // pad to data_start alignment
    }
    let data_start = out.len();
    for (i, (_, _, _, data)) in tensors.iter().enumerate() {
        while out.len() - data_start < offsets[i] as usize {
            out.push(0);
        }
        out.extend_from_slice(data);
    }
    out
}

/// The result of parsing a GGUF header and tensor directory (magic through the last tensor info,
/// data start aligned per `general.alignment`): the one binary-format parse [`GgufIndex::parse`]
/// (header and directory only, dquant.md 8.1) builds on.
struct ParsedHeader {
    version: u32,
    metadata: HashMap<String, GgufValue>,
    tensors: HashMap<String, GgufTensorInfo>,
    data_start: usize,
}

fn parse_header(bytes: &[u8]) -> Result<ParsedHeader, LoadError> {
    let mut c = Cursor { b: bytes, p: 0 };
    if c.take(4)? != b"GGUF" {
        return Err(gerr("bad magic (not a GGUF file)"));
    }
    let version = c.u32()?;
    if version != 2 && version != 3 {
        return Err(gerr(&format!("unsupported GGUF version {version}")));
    }
    let tensor_count = c.u64()? as usize;
    let kv_count = c.u64()? as usize;

    let mut metadata = HashMap::new();
    let mut seen_keys = HashSet::new();
    let mut layer_budget = MAX_LAYER_ARRAY_TOTAL;
    for _ in 0..kv_count {
        let key = c.gstr()?;
        let ty = c.u32()?;
        if !seen_keys.insert(key.clone()) {
            return Err(LoadError::DuplicateGgufMetadataKey { key });
        }
        if ty == GGUF_TYPE_ARRAY {
            if let Some(array) = c.array(&key, &mut layer_budget)? {
                metadata.insert(key, array);
            }
        } else {
            metadata.insert(key, c.value(ty)?);
        }
    }

    let mut tensors = HashMap::new();
    for _ in 0..tensor_count {
        let name = c.gstr()?;
        let n_dims = c.u32()? as usize;
        // A malformed GGUF can declare a rank up to u32::MAX, and `Vec::with_capacity(n_dims)`
        // would allocate multiple GB before the cursor bounds check catches the truncation.
        if n_dims > MAX_TENSOR_DIMS {
            return Err(gerr("tensor rank exceeds maximum (malformed GGUF)"));
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(c.u64()?);
        }
        // Reject an element count that overflows `usize` or exceeds `MAX_TENSOR_ELEMS` here, so
        // downstream `dequant`/`pack_*`/range reads never see a product that panics `vec![0.0;
        // numel]` or wraps a byte-length computation past a length guard.
        let numel = dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| gerr("tensor element count overflows usize (malformed GGUF)"))?;
        if numel > MAX_TENSOR_ELEMS {
            return Err(gerr(
                "tensor element count exceeds maximum (malformed GGUF)",
            ));
        }
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        let info = GgufTensorInfo {
            dims,
            ggml_type,
            offset,
            shard: 0,
        };
        if tensors.insert(name.clone(), info).is_some() {
            return Err(LoadError::DuplicateTensorName {
                name,
                scope: "within one GGUF directory".to_string(),
            });
        }
    }

    // Tensor data begins at the cursor, padded up to general.alignment (default 32).
    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(value) => value
            .as_u64()
            .filter(|a| a.is_power_of_two())
            .and_then(|a| usize::try_from(a).ok())
            .ok_or_else(|| LoadError::InvalidGgufAlignment {
                value: format!("{value:?}"),
            })?,
    };
    let data_start =
        c.p.checked_next_multiple_of(alignment)
            .ok_or_else(|| gerr("tensor data start overflows usize"))?;
    if data_start > bytes.len() {
        return Err(gerr("tensor data start past end of file"));
    }

    Ok(ParsedHeader {
        version,
        metadata,
        tensors,
        data_start,
    })
}

/// Every ggml tensor type id a GGUF file can name (0..=39, ggml.h's `ggml_type`), mapped to the
/// [`WeightFormat`] poot reads it as, or `None` when poot has no packed or dense representation for
/// it: a ggml id ggml itself no longer assigns (a removed legacy variant, e.g. `Q4_2`), or a real
/// ggml scheme with no packed path in `poot-quant` (`Q8_1`, `Q8_K`, the IQ2_XXS/IQ2_S/IQ1 grid formats,
/// `TQ1_0`/`TQ2_0`, or a plain integer/`F64` tensor type). A quantized tensor of a `None` type is
/// refused by name at load ([`crate::LoadError::UnsupportedGgufType`]), never loaded dense silently
/// (ADR-0103 decision 2). Pinned by a literal table (card 543 SC-004): every entry is written out, not
/// derived, so it can be checked against `ggml.h` line by line.
pub fn weight_format_of(ggml_type: u32) -> Option<WeightFormat> {
    const TABLE: [Option<WeightFormat>; 40] = [
        Some(WeightFormat::F32),     // 0 GGML_TYPE_F32
        Some(WeightFormat::F16),     // 1 GGML_TYPE_F16
        Some(WeightFormat::Q4_0),    // 2 GGML_TYPE_Q4_0
        Some(WeightFormat::Q4_1),    // 3 GGML_TYPE_Q4_1
        None,                        // 4 GGML_TYPE_Q4_2 (removed)
        None,                        // 5 GGML_TYPE_Q4_3 (removed)
        Some(WeightFormat::Q5_0),    // 6 GGML_TYPE_Q5_0
        Some(WeightFormat::Q5_1),    // 7 GGML_TYPE_Q5_1
        Some(WeightFormat::Q8_0),    // 8 GGML_TYPE_Q8_0
        None,                        // 9 GGML_TYPE_Q8_1 (no packed path)
        Some(WeightFormat::Q2_K),    // 10 GGML_TYPE_Q2_K
        Some(WeightFormat::Q3_K),    // 11 GGML_TYPE_Q3_K
        Some(WeightFormat::Q4_K),    // 12 GGML_TYPE_Q4_K
        Some(WeightFormat::Q5_K),    // 13 GGML_TYPE_Q5_K
        Some(WeightFormat::Q6_K),    // 14 GGML_TYPE_Q6_K
        None,                        // 15 GGML_TYPE_Q8_K (no packed path)
        None,                        // 16 GGML_TYPE_IQ2_XXS (no packed path)
        Some(WeightFormat::Iq2_Xs),  // 17 GGML_TYPE_IQ2_XS
        Some(WeightFormat::Iq3_Xxs), // 18 GGML_TYPE_IQ3_XXS
        None,                        // 19 GGML_TYPE_IQ1_S (no packed path)
        Some(WeightFormat::Iq4_Nl),  // 20 GGML_TYPE_IQ4_NL
        Some(WeightFormat::Iq3_S),   // 21 GGML_TYPE_IQ3_S
        None,                        // 22 GGML_TYPE_IQ2_S (no packed path)
        Some(WeightFormat::Iq4_Xs),  // 23 GGML_TYPE_IQ4_XS
        None,                        // 24 GGML_TYPE_I8 (not a weight format)
        None,                        // 25 GGML_TYPE_I16 (not a weight format)
        None,                        // 26 GGML_TYPE_I32 (not a weight format)
        None,                        // 27 GGML_TYPE_I64 (not a weight format)
        None,                        // 28 GGML_TYPE_F64 (not a weight format)
        None,                        // 29 GGML_TYPE_IQ1_M (no packed path)
        Some(WeightFormat::Bf16),    // 30 GGML_TYPE_BF16
        None,                        // 31 GGML_TYPE_Q4_0_4_4 (removed)
        None,                        // 32 GGML_TYPE_Q4_0_4_8 (removed)
        None,                        // 33 GGML_TYPE_Q4_0_8_8 (removed)
        None,                        // 34 GGML_TYPE_TQ1_0 (no packed path)
        None,                        // 35 GGML_TYPE_TQ2_0 (no packed path)
        None,                        // 36 GGML_TYPE_IQ4_NL_4_4 (removed)
        None,                        // 37 GGML_TYPE_IQ4_NL_4_8 (removed)
        None,                        // 38 GGML_TYPE_IQ4_NL_8_8 (removed)
        Some(WeightFormat::Mxfp4),   // 39 GGML_TYPE_MXFP4
    ];
    TABLE.get(ggml_type as usize).copied().flatten()
}

/// A GGUF checkpoint's header and tensor directory, with no tensor data read (dquant.md 8.1): the
/// metadata key-value table and the `(name, ggml type, ne, data offset)` of every tensor. A split
/// checkpoint (llama.cpp's `-0000N-of-0000M.gguf` sets) is one index over its shards in order: the
/// metadata is the first shard's, every tensor names the shard that holds it, and the index keeps the
/// mappings it was parsed from, so the bytes a plan slices are the ones the directory describes.
/// [`crate::gguf::read_gguf`] slices each tensor's bytes out of those mappings using
/// [`Self::data_start`]; nothing here copies file contents.
pub struct GgufIndex {
    /// The GGUF format version the first shard declares.
    pub version: u32,
    /// Metadata key-value table of the first shard, keyed by the dotted key (`general.architecture`, ...).
    /// Arrays are stored for the tokenizer tables and for short per-layer tables; a longer array under
    /// any other key is validated and left out (`STORED_ARRAYS`).
    pub metadata: HashMap<String, GgufValue>,
    /// Tensor directory of every shard, keyed by tensor name.
    pub tensors: HashMap<String, GgufTensorInfo>,
    shards: Vec<Shard>,
}

/// One shard of a checkpoint: its mapping and where its tensor data begins.
struct Shard {
    file: StoredBytes,
    data_start: usize,
}

impl GgufIndex {
    /// Parse a single-file checkpoint: a header and tensor directory from `file`, a mapping of a
    /// checkpoint or an in-memory buffer (a fixture). Only the header pages are touched, so it also
    /// works on a partial download. A file that declares itself one shard of a split set is refused
    /// here, as `parse_shards` refuses any set of the wrong size.
    pub fn parse(file: &StoredBytes) -> Result<Self, LoadError> {
        Self::parse_shards(std::slice::from_ref(file))
    }

    /// Parse a split checkpoint from its `files`, in shard order. With more than one file they must
    /// be exactly the set the first one declares (`split.count`), each in its declared position
    /// (`split.no`, the first at 0), and no tensor may appear in two shards; a violation is a typed
    /// [`LoadError::GgufSplit`]. A single file that declares any split key is held to the same rules;
    /// one with none needs no split keys.
    pub fn parse_shards(files: &[StoredBytes]) -> Result<Self, LoadError> {
        let first = files
            .first()
            .ok_or(LoadError::GgufSplit(GgufSplitError::NoShards))?;
        let header = parse_header(first.as_slice())?;
        let mut index = Self {
            version: header.version,
            metadata: header.metadata,
            tensors: header.tensors,
            shards: vec![Shard {
                file: first.clone(),
                data_start: header.data_start,
            }],
        };
        // A file that carries any split key is one shard of a set, so the first shard must declare
        // exactly the number of files given and sit at position 0, whether or not more files follow.
        let declared_no = index.get("split.no").and_then(GgufValue::as_u64);
        if files.len() > 1 || index.get("split.count").is_some() || declared_no.is_some() {
            let declared = index.split_key("split.count");
            if declared != Some(files.len()) {
                return Err(LoadError::GgufSplit(GgufSplitError::Count {
                    declared,
                    given: files.len(),
                }));
            }
            if declared_no != Some(0) {
                return Err(LoadError::GgufSplit(GgufSplitError::Order {
                    position: 0,
                    declared: declared_no,
                }));
            }
        }
        for (position, file) in files.iter().enumerate().skip(1) {
            let header = parse_header(file.as_slice())?;
            let declared = header.metadata.get("split.no").and_then(GgufValue::as_u64);
            if declared != Some(position as u64) {
                return Err(LoadError::GgufSplit(GgufSplitError::Order {
                    position,
                    declared,
                }));
            }
            for (name, mut info) in header.tensors {
                info.shard = position;
                if let Some(first) = index.tensors.insert(name.clone(), info) {
                    return Err(LoadError::GgufSplit(GgufSplitError::DuplicateTensor {
                        tensor: name,
                        shards: (first.shard, position),
                    }));
                }
            }
            index.shards.push(Shard {
                file: file.clone(),
                data_start: header.data_start,
            });
        }
        Ok(index)
    }

    fn split_key(&self, key: &str) -> Option<usize> {
        self.get(key)
            .and_then(GgufValue::as_u64)
            .and_then(|n| usize::try_from(n).ok())
    }

    /// The tensor count the checkpoint declares for the whole split set (`split.tensors.count`), if
    /// it declares one. A directory that lists fewer tensors is missing shards.
    pub(crate) fn declared_tensors(&self) -> Option<usize> {
        self.split_key("split.tensors.count")
    }

    /// The mappings of the shards, in order.
    pub(crate) fn files(&self) -> Vec<StoredBytes> {
        self.shards.iter().map(|s| s.file.clone()).collect()
    }

    /// The metadata value for `key`, if the file has one.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// `general.architecture` (e.g. "qwen2", "llama").
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(GgufValue::as_str)
    }

    /// The byte offset, from the start of shard `shard`, where its tensor data begins.
    /// `GgufTensorInfo::offset` is relative to this.
    pub(crate) fn data_start(&self, shard: usize) -> usize {
        self.shards[shard].data_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(
        kvs: &[(&str, GgufValue)],
        tensors: &[(&str, Vec<u64>, u32, Vec<u8>)],
    ) -> Result<GgufIndex, LoadError> {
        GgufIndex::parse(&StoredBytes::from(write_gguf(kvs, tensors)))
    }

    fn f32_tensor(name: &'static str) -> (&'static str, Vec<u64>, u32, Vec<u8>) {
        (name, vec![2], 0, vec![0u8; 8])
    }

    /// POOT-1196 SC-003: `general.alignment` of 0 and of 3 is a typed refusal naming the value, not
    /// a divide-by-zero panic or a silently wrong data start.
    #[test]
    fn alignment_must_be_a_nonzero_power_of_two() {
        for bad in [0u32, 3] {
            let error = parse(&[("general.alignment", GgufValue::U32(bad))], &[])
                .err()
                .unwrap_or_else(|| panic!("alignment {bad} must be refused"));
            assert!(
                matches!(&error, LoadError::InvalidGgufAlignment { value } if value == &format!("U32({bad})")),
                "alignment {bad}: {error}"
            );
        }
        let wrong_type = parse(&[("general.alignment", GgufValue::Str("32".into()))], &[]);
        assert!(matches!(
            wrong_type,
            Err(LoadError::InvalidGgufAlignment { .. })
        ));
        parse(&[("general.alignment", GgufValue::U32(32))], &[]).expect("32 is a valid alignment");
    }

    /// POOT-1196 SC-004: a tensor name or a metadata key listed twice is refused by name.
    #[test]
    fn repeated_tensor_names_and_metadata_keys_are_refused() {
        let error = parse(&[], &[f32_tensor("w"), f32_tensor("w")])
            .err()
            .expect("two tensors named w");
        assert!(
            matches!(&error, LoadError::DuplicateTensorName { name, .. } if name == "w"),
            "{error}"
        );
        let error = parse(
            &[
                ("general.name", GgufValue::Str("a".into())),
                ("general.name", GgufValue::Str("b".into())),
            ],
            &[],
        )
        .err()
        .expect("general.name twice");
        assert!(
            matches!(&error, LoadError::DuplicateGgufMetadataKey { key } if key == "general.name"),
            "{error}"
        );
    }

    /// A header with one metadata array `key` of `n` declared elements of type `elem_ty`, whose bytes are `elements`.
    fn header_with_array(key: &str, elem_ty: u32, elements: &[u8], n: u64) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend(3u32.to_le_bytes());
        header.extend(0u64.to_le_bytes());
        header.extend(1u64.to_le_bytes());
        header.extend(gguf_gstr_bytes(key));
        header.extend(9u32.to_le_bytes());
        header.extend(elem_ty.to_le_bytes());
        header.extend(n.to_le_bytes());
        header.extend_from_slice(elements);
        header.resize(header.len().next_multiple_of(32), 0); // the (empty) data section's alignment
        header
    }

    /// An array whose declared length the remaining bytes cannot hold is refused before decoding,
    /// stored or skipped.
    #[test]
    fn array_length_beyond_the_file_is_refused() {
        for key in ["k", "tokenizer.ggml.scores"] {
            let elem = if key == "k" { 4 } else { 6 };
            let header = header_with_array(key, elem, &[], 1 << 40);
            let error = GgufIndex::parse(&StoredBytes::from(header))
                .err()
                .expect("huge array");
            assert!(error.to_string().contains("array length"), "{key}: {error}");
        }
    }

    /// An array longer than any per-layer table under a non-tokenizer key is walked and dropped: it
    /// costs no per-element memory (the key is simply absent) while its bytes are still
    /// bounds-checked. A short per-layer array is stored.
    #[test]
    fn a_long_unread_array_is_skipped_not_stored() {
        let long = MAX_LAYER_ARRAY_LEN + 1;
        let bytes = header_with_array("some.table", 0, &vec![7u8; long], long as u64);
        let index = GgufIndex::parse(&StoredBytes::from(bytes)).unwrap();
        assert!(index.get("some.table").is_none());
        let per_layer = header_with_array("some.layer_kinds", 0, &[1, 0, 1], 3);
        let index = GgufIndex::parse(&StoredBytes::from(per_layer)).unwrap();
        assert_eq!(
            index
                .get("some.layer_kinds")
                .and_then(GgufValue::as_array)
                .map(<[_]>::len),
            Some(3)
        );
        // Its bytes must still be there.
        let truncated = header_with_array("some.table", 0, &[1, 2, 3], 100_000);
        assert!(GgufIndex::parse(&StoredBytes::from(truncated)).is_err());
    }

    /// Many short arrays cannot add up to a per-byte inflation: past the per-file element budget
    /// the header is refused.
    #[test]
    fn short_arrays_share_one_element_budget() {
        let arrays = MAX_LAYER_ARRAY_TOTAL / MAX_LAYER_ARRAY_LEN + 1;
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend(3u32.to_le_bytes());
        header.extend(0u64.to_le_bytes());
        header.extend((arrays as u64).to_le_bytes());
        for i in 0..arrays {
            header.extend(gguf_gstr_bytes(&format!("some.table{i}")));
            header.extend(9u32.to_le_bytes());
            header.extend(0u32.to_le_bytes());
            header.extend((MAX_LAYER_ARRAY_LEN as u64).to_le_bytes());
            header.extend(vec![0u8; MAX_LAYER_ARRAY_LEN]);
        }
        header.resize(header.len().next_multiple_of(32), 0);
        let error = GgufIndex::parse(&StoredBytes::from(header))
            .err()
            .expect("over budget");
        assert!(error.to_string().contains("some.table16"), "{error}");
    }

    /// A stored table longer than the cap is refused however many bytes back it.
    #[test]
    fn a_tokenizer_table_beyond_the_cap_is_refused() {
        let n = MAX_STORED_ARRAY_LEN + 1;
        let bytes = header_with_array("tokenizer.ggml.scores", 6, &vec![0u8; 4 * n], n as u64);
        let error = GgufIndex::parse(&StoredBytes::from(bytes))
            .err()
            .expect("over the cap");
        assert!(
            error.to_string().contains("tokenizer.ggml.scores"),
            "{error}"
        );
    }

    /// The tokenizer tables are stored, and only with the element type the readers decode; a
    /// file-size `u8` array under a tokenizer key is refused instead of inflated.
    #[test]
    fn tokenizer_tables_are_stored_with_their_element_type() {
        let mut e = Vec::new();
        for s in ["ab", "c"] {
            e.extend(gguf_gstr_bytes(s));
        }
        let ok = header_with_array("tokenizer.ggml.tokens", 8, &e, 2);
        let index = GgufIndex::parse(&StoredBytes::from(ok)).unwrap();
        assert_eq!(
            index
                .get("tokenizer.ggml.tokens")
                .and_then(GgufValue::as_array)
                .map(<[_]>::len),
            Some(2)
        );
        let bytes = header_with_array("tokenizer.ggml.tokens", 0, &vec![7u8; 4096], 4096);
        let error = GgufIndex::parse(&StoredBytes::from(bytes))
            .err()
            .expect("u8 tokens");
        assert!(
            error.to_string().contains("tokenizer.ggml.tokens"),
            "{error}"
        );
    }
}
