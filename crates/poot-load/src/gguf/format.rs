//! GGUF metadata, tensor table, and container I/O.

use std::collections::HashMap;

use poot_quant::format::WeightFormat;

use crate::LoadError;

/// A GGUF metadata value (the 13 GGUF value types; `Array` holds a homogeneous list).
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
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
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
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
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

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
    /// A typed value given its GGUF value-type id (recursing for arrays).
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
            9 => {
                let elem_ty = self.u32()?;
                if elem_ty == 9 {
                    return Err(gerr("nested GGUF arrays are not supported"));
                }
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(elem_ty)?);
                }
                GgufValue::Array(v)
            }
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

pub(crate) fn gguf_gstr_bytes(s: &str) -> Vec<u8> {
    let mut v = (s.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(s.as_bytes());
    v
}

/// Serialize a [`GgufValue`] to (type-id, body-bytes) for the writer. Supports the types
/// `Cursor::value` accepts; arrays must be homogeneous.
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
/// raw_bytes)`, the inverse of [`GgufIndex::from_bytes`] plus [`crate::gguf::read_gguf`]. Tensor data offsets are aligned to 32. A
/// fixture utility (poot is not a converter) for building small in-memory GGUFs in tests.
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
/// data start aligned per `general.alignment`): the one binary-format parse [`GgufIndex::open`] and
/// [`GgufIndex::from_bytes`] (header and directory only, dquant.md 8.1) build on.
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
    for _ in 0..kv_count {
        let key = c.gstr()?;
        let ty = c.u32()?;
        let val = c.value(ty)?;
        metadata.insert(key, val);
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
        tensors.insert(
            name,
            GgufTensorInfo {
                dims,
                ggml_type,
                offset,
            },
        );
    }

    // Tensor data begins at the cursor, padded up to general.alignment (default 32).
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;
    let data_start = c.p.div_ceil(alignment) * alignment;
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

/// Reads at most 256 MB from the start of `path`: enough for any realistic GGUF header, including
/// large tokenizer tables, without reading tensor data ([`GgufIndex::open`]).
fn read_bounded_header(path: &std::path::Path) -> Result<Vec<u8>, LoadError> {
    use std::io::Read;
    let mut f =
        std::fs::File::open(path).map_err(|e| gerr(&format!("open {}: {e}", path.display())))?;
    let file_len = f
        .metadata()
        .map_err(|e| gerr(&format!("stat {}: {e}", path.display())))?
        .len() as usize;
    // Tokenizer tables are the largest header part (typically 5-20 MB for 200k-vocab models).
    const MAX_HEADER: usize = 256 * 1024 * 1024;
    let read_len = file_len.min(MAX_HEADER);
    let mut bytes = vec![0u8; read_len];
    f.read_exact(&mut bytes)
        .map_err(|e| gerr(&format!("read header {}: {e}", path.display())))?;
    Ok(bytes)
}

/// Every ggml tensor type id a GGUF file can name (0..=39, ggml.h's `ggml_type`), mapped to the
/// [`WeightFormat`] poot reads it as, or `None` when poot has no packed or dense representation for
/// it: a ggml id ggml itself no longer assigns (a removed legacy variant, e.g. `Q4_2`), or a real
/// ggml scheme with no packed path in `poot-quant` (`Q8_1`, `Q8_K`, the IQ2/IQ3/IQ1 grid formats,
/// `TQ1_0`/`TQ2_0`, or a plain integer/`F64` tensor type). A quantized tensor of a `None` type is
/// refused by name at load ([`crate::LoadError::UnsupportedGgufType`]), never loaded dense silently
/// (ADR-0103 decision 2). Pinned by a literal table (card 543 SC-004): every entry is written out, not
/// derived, so it can be checked against `ggml.h` line by line.
pub fn weight_format_of(ggml_type: u32) -> Option<WeightFormat> {
    const TABLE: [Option<WeightFormat>; 40] = [
        Some(WeightFormat::F32),    // 0 GGML_TYPE_F32
        Some(WeightFormat::F16),    // 1 GGML_TYPE_F16
        Some(WeightFormat::Q4_0),   // 2 GGML_TYPE_Q4_0
        Some(WeightFormat::Q4_1),   // 3 GGML_TYPE_Q4_1
        None,                       // 4 GGML_TYPE_Q4_2 (removed)
        None,                       // 5 GGML_TYPE_Q4_3 (removed)
        Some(WeightFormat::Q5_0),   // 6 GGML_TYPE_Q5_0
        Some(WeightFormat::Q5_1),   // 7 GGML_TYPE_Q5_1
        Some(WeightFormat::Q8_0),   // 8 GGML_TYPE_Q8_0
        None,                       // 9 GGML_TYPE_Q8_1 (no packed path)
        Some(WeightFormat::Q2_K),   // 10 GGML_TYPE_Q2_K
        Some(WeightFormat::Q3_K),   // 11 GGML_TYPE_Q3_K
        Some(WeightFormat::Q4_K),   // 12 GGML_TYPE_Q4_K
        Some(WeightFormat::Q5_K),   // 13 GGML_TYPE_Q5_K
        Some(WeightFormat::Q6_K),   // 14 GGML_TYPE_Q6_K
        None,                       // 15 GGML_TYPE_Q8_K (no packed path)
        None,                       // 16 GGML_TYPE_IQ2_XXS (no packed path)
        None,                       // 17 GGML_TYPE_IQ2_XS (no packed path)
        None,                       // 18 GGML_TYPE_IQ3_XXS (no packed path)
        None,                       // 19 GGML_TYPE_IQ1_S (no packed path)
        Some(WeightFormat::Iq4_Nl), // 20 GGML_TYPE_IQ4_NL
        None,                       // 21 GGML_TYPE_IQ3_S (no packed path)
        None,                       // 22 GGML_TYPE_IQ2_S (no packed path)
        Some(WeightFormat::Iq4_Xs), // 23 GGML_TYPE_IQ4_XS
        None,                       // 24 GGML_TYPE_I8 (not a weight format)
        None,                       // 25 GGML_TYPE_I16 (not a weight format)
        None,                       // 26 GGML_TYPE_I32 (not a weight format)
        None,                       // 27 GGML_TYPE_I64 (not a weight format)
        None,                       // 28 GGML_TYPE_F64 (not a weight format)
        None,                       // 29 GGML_TYPE_IQ1_M (no packed path)
        Some(WeightFormat::Bf16),   // 30 GGML_TYPE_BF16
        None,                       // 31 GGML_TYPE_Q4_0_4_4 (removed)
        None,                       // 32 GGML_TYPE_Q4_0_4_8 (removed)
        None,                       // 33 GGML_TYPE_Q4_0_8_8 (removed)
        None,                       // 34 GGML_TYPE_TQ1_0 (no packed path)
        None,                       // 35 GGML_TYPE_TQ2_0 (no packed path)
        None,                       // 36 GGML_TYPE_IQ4_NL_4_4 (removed)
        None,                       // 37 GGML_TYPE_IQ4_NL_4_8 (removed)
        None,                       // 38 GGML_TYPE_IQ4_NL_8_8 (removed)
        Some(WeightFormat::Mxfp4),  // 39 GGML_TYPE_MXFP4
    ];
    TABLE.get(ggml_type as usize).copied().flatten()
}

/// A GGUF file's header and tensor directory, with no tensor data read (dquant.md 8.1): the metadata
/// key-value table and the `(name, ggml type, ne, data offset)` of every tensor, bounded the same way
/// as the 256 MB header read cap ([`read_bounded_header`]). [`crate::gguf::read_gguf`] range-reads each tensor's bytes from a
/// separately opened [`std::fs::File`] using [`Self::data_start`], so nothing here keeps file
/// contents resident once parsing returns.
pub struct GgufIndex {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    data_start: usize,
}

impl GgufIndex {
    /// Header and directory only (bounded read; no tensor data).
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let header = parse_header(&read_bounded_header(path)?)?;
        Ok(Self {
            version: header.version,
            metadata: header.metadata,
            tensors: header.tensors,
            data_start: header.data_start,
        })
    }

    /// Parse a GGUF header and tensor directory from an in-memory buffer, for a fixture or test that
    /// already has the whole (small) GGUF in memory. `bytes` may hold only the header and tensor
    /// table plus the tensor data (`crate::gguf::read_gguf` then range-reads directly from the same
    /// buffer via its [`crate::gguf::GgufSource`] impl for `[u8]`); it never needs to be a real file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let header = parse_header(bytes)?;
        Ok(Self {
            version: header.version,
            metadata: header.metadata,
            tensors: header.tensors,
            data_start: header.data_start,
        })
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// `general.architecture` (e.g. "qwen2", "llama").
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(GgufValue::as_str)
    }

    /// The byte offset, from the start of the file, where tensor data begins. `GgufTensorInfo::offset`
    /// is relative to this.
    pub(crate) fn data_start(&self) -> usize {
        self.data_start
    }
}
