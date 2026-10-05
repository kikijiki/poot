//! `poot-tensor`: the one [`DType`], covering every dtype a checkpoint header or a graph value can
//! name. A rank-0 leaf crate (X1): no dependency on any other poot crate, so every
//! other crate may depend on it with no cycle.
//!
//! The crate also holds the typed host carrier: [`HostTensor`] (dtype, shape and a payload of the
//! dtype's storage class) and the borrowed byte [`HostView`] every executor contract takes.
//!
//! `poot_tensor::DType` and `poot_quant::StoredDtype` both named one overlapping concept (an
//! element's storage dtype) with two different variant sets and two different method names
//! (`byte_size` vs `element_width_bytes`). This crate merges them into one type, imported directly
//! by every former user of either (no alias, no re-export shim): `poot-graph-ir` keeps
//! `DTypeClass`, the graph-only classification of a [`DType`] (arithmetic, index, storage-only,
//! packed), since that classification is graph semantics, not a tensor-storage fact.

use std::fmt;

mod host;

#[cfg(feature = "read-counter")]
pub use host::read_counter;
pub use host::{CarrierError, HostData, HostTensor, StorageClass};

/// Every dtype a checkpoint header or a graph value can name. `Display`/`Eq`/`Hash` let it sit in
/// map keys (residency, state-sharing) and error messages uniformly.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DType {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    /// bfloat16 (spec 024). Storage is bf16; arithmetic accumulates in f32.
    BF16,
    F32,
    F64,
    /// OCP E4M3FN FP8 storage (spec 149 Stage 1): a logical storage dtype, not an arithmetic type.
    E4M3FN,
    /// OCP E8M0 FP8 scale storage: exponent-only, no arithmetic meaning of its own.
    E8M0,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DType::Bool => "bool",
            DType::I8 => "i8",
            DType::U8 => "u8",
            DType::I16 => "i16",
            DType::U16 => "u16",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::U64 => "u64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::E4M3FN => "e4m3fn",
            DType::E8M0 => "e8m0",
        })
    }
}

impl Default for DType {
    /// F32 (the full-precision compute path). Lets `DType` sit in structs that derive `Default`;
    /// a new `DType` field stays f32 unless set explicitly.
    fn default() -> Self {
        DType::F32
    }
}

/// A checkpoint header, or another caller, named a dtype string this crate does not recognize.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownDtype(pub String);

impl fmt::Display for UnknownDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown dtype {:?}", self.0)
    }
}

impl std::error::Error for UnknownDtype {}

impl DType {
    /// Parse a checkpoint header's `dtype` string (safetensors spelling, e.g. `"BF16"`,
    /// `"F8_E4M3"`). A string that names no dtype this crate recognizes is a typed error, never a
    /// silent `None` (former `StoredDtype::parse`).
    pub fn parse(dtype: &str) -> Result<Self, UnknownDtype> {
        Ok(match dtype {
            "BOOL" => Self::Bool,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "I64" => Self::I64,
            "U64" => Self::U64,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "F8_E4M3" => Self::E4M3FN,
            "F8_E8M0" => Self::E8M0,
            other => return Err(UnknownDtype(other.to_string())),
        })
    }

    /// Bytes one element of this dtype occupies in storage (former `DType::byte_size` and
    /// `StoredDtype::element_width_bytes`, now the one width table every reader and every graph
    /// consumer reads, so it cannot drift between the two former copies).
    pub const fn byte_size(self) -> usize {
        match self {
            DType::Bool | DType::I8 | DType::U8 | DType::E4M3FN | DType::E8M0 => 1,
            DType::I16 | DType::U16 | DType::F16 | DType::BF16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 => 8,
        }
    }

    /// Whether `infer` accepts this dtype for an index-role operand (`Gather`, `Scatter`,
    /// `ScatterUpdate`'s `inv`, `IndexedMatMul`'s `idx`, `DynamicUpdateSlice`'s runtime index, and
    /// friends): R466-010. F32 stays accepted too because indices are F32 by evaluator convention
    /// until Card 558b switches them to I32 everywhere. Every other dtype is rejected.
    pub fn is_index_operand(self) -> bool {
        matches!(self, DType::F32 | DType::I32)
    }
}

/// A borrowed, byte-level view of one host-resident tensor, typed by [`DType`] (X1): the contract's
/// `StepInputs` carries these, so a step input is bound and dtype-checked without an owned host
/// copy. `HostView` never interprets its bytes itself; typed accessors are a checked reinterpret of
/// the same borrowed bytes, refusing a mismatched dtype rather than converting.
#[derive(Clone, Copy, Debug)]
pub struct HostView<'a> {
    dtype: DType,
    elems: usize,
    bytes: &'a [u8],
}

/// `HostView::new` was given a byte slice whose length does not equal `elems * dtype.byte_size()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostViewLengthError {
    pub dtype: DType,
    pub elems: usize,
    pub expected_bytes: usize,
    pub actual_bytes: usize,
}

impl fmt::Display for HostViewLengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} x {} needs {} bytes, got {}",
            self.dtype, self.elems, self.expected_bytes, self.actual_bytes
        )
    }
}

impl std::error::Error for HostViewLengthError {}

impl<'a> HostView<'a> {
    /// Checks that `bytes.len()` is exactly `elems * dtype.byte_size()` before borrowing it.
    pub fn new(dtype: DType, elems: usize, bytes: &'a [u8]) -> Result<Self, HostViewLengthError> {
        let expected_bytes = elems.saturating_mul(dtype.byte_size());
        if bytes.len() != expected_bytes {
            return Err(HostViewLengthError {
                dtype,
                elems,
                expected_bytes,
                actual_bytes: bytes.len(),
            });
        }
        Ok(Self {
            dtype,
            elems,
            bytes,
        })
    }

    /// A view over bytes a [`HostTensor`] already checked against `dtype` and `elems`.
    pub(crate) fn checked(dtype: DType, elems: usize, bytes: &'a [u8]) -> Self {
        debug_assert_eq!(bytes.len(), elems * dtype.byte_size());
        Self {
            dtype,
            elems,
            bytes,
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn elems(&self) -> usize {
        self.elems
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// The view's elements as `f32`, if its dtype is `F32`; `None` otherwise (never a silent
    /// widening of another dtype).
    pub fn as_f32(&self) -> Option<&'a [f32]> {
        (self.dtype == DType::F32).then(|| bytemuck::cast_slice(self.bytes))
    }

    /// The view's elements as `i32`, if its dtype is `I32`.
    pub fn as_i32(&self) -> Option<&'a [i32]> {
        (self.dtype == DType::I32).then(|| bytemuck::cast_slice(self.bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trips_every_safetensors_spelling() {
        let cases: &[(&str, DType)] = &[
            ("BOOL", DType::Bool),
            ("I8", DType::I8),
            ("U8", DType::U8),
            ("I16", DType::I16),
            ("U16", DType::U16),
            ("I32", DType::I32),
            ("U32", DType::U32),
            ("I64", DType::I64),
            ("U64", DType::U64),
            ("F16", DType::F16),
            ("BF16", DType::BF16),
            ("F32", DType::F32),
            ("F64", DType::F64),
            ("F8_E4M3", DType::E4M3FN),
            ("F8_E8M0", DType::E8M0),
        ];
        for (text, dtype) in cases {
            assert_eq!(DType::parse(text).unwrap(), *dtype, "parse({text})");
        }
    }

    #[test]
    fn parse_rejects_an_unrecognized_spelling() {
        assert_eq!(
            DType::parse("COMPLEX64").unwrap_err(),
            UnknownDtype("COMPLEX64".to_string())
        );
    }

    #[test]
    fn byte_size_matches_every_known_width() {
        let cases: &[(DType, usize)] = &[
            (DType::Bool, 1),
            (DType::I8, 1),
            (DType::U8, 1),
            (DType::I16, 2),
            (DType::U16, 2),
            (DType::I32, 4),
            (DType::U32, 4),
            (DType::I64, 8),
            (DType::U64, 8),
            (DType::F16, 2),
            (DType::BF16, 2),
            (DType::F32, 4),
            (DType::F64, 8),
            (DType::E4M3FN, 1),
            (DType::E8M0, 1),
        ];
        for (dtype, size) in cases {
            assert_eq!(dtype.byte_size(), *size, "{dtype:?}.byte_size()");
        }
    }

    #[test]
    fn host_view_refuses_a_length_mismatch() {
        let bytes = [0u8; 8];
        let err = HostView::new(DType::F32, 3, &bytes).unwrap_err();
        assert_eq!(err.expected_bytes, 12);
        assert_eq!(err.actual_bytes, 8);
    }

    #[test]
    fn host_view_typed_accessors_refuse_the_wrong_dtype() {
        let bytes = 7i32.to_le_bytes();
        let view = HostView::new(DType::I32, 1, &bytes).unwrap();
        assert_eq!(view.as_i32(), Some(&[7i32][..]));
        assert_eq!(view.as_f32(), None);
    }
}
