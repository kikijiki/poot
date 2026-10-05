//! The typed host carrier: [`HostTensor`] is one owned, shape-carrying host tensor whose dtype is
//! part of its type. The payload ([`HostData`]) is a storage class chosen by the dtype, so a BF16
//! tensor holds 16-bit words, an I32 tensor holds `i32` words and no tensor holds a second,
//! widened copy of its own values. The one constructor that takes an arbitrary payload refuses a
//! payload whose class or length does not fit the dtype ([`CarrierError`]).
//!
//! Words are little-endian in memory: [`HostTensor::view`] hands the payload out as bytes without a
//! copy, which is the layout every device upload and every checkpoint file uses.

use std::borrow::Cow;
use std::sync::Arc;

use crate::{DType, HostView};

#[cfg(not(target_endian = "little"))]
compile_error!("poot-tensor hands payload words out as little-endian bytes without a copy");

/// The element class a dtype's payload is stored as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StorageClass {
    /// 4-byte IEEE floats: `F32`.
    F32,
    /// 4-byte signed words: `I32`.
    I32,
    /// 2-byte words, bit patterns of a half-width float: `F16` and `BF16`.
    Half,
    /// Raw little-endian bytes: every other dtype (E4M3FN and packed words included).
    Bytes,
}

impl DType {
    /// The storage class a [`HostTensor`] of this dtype keeps its payload in.
    pub const fn storage_class(self) -> StorageClass {
        match self {
            DType::F32 => StorageClass::F32,
            DType::I32 => StorageClass::I32,
            DType::F16 | DType::BF16 => StorageClass::Half,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::I16
            | DType::U16
            | DType::U32
            | DType::I64
            | DType::U64
            | DType::F64
            | DType::E4M3FN
            | DType::E8M0 => StorageClass::Bytes,
        }
    }
}

/// A tensor payload, shared by `Arc` so a clone never copies the elements. The variant is the
/// storage class ([`DType::storage_class`]); the tensor's dtype says how to read it.
#[derive(Clone, Debug, PartialEq)]
pub enum HostData {
    F32(Arc<[f32]>),
    I32(Arc<[i32]>),
    /// Half-width float bit patterns (`F16` or `BF16`).
    Half(Arc<[u16]>),
    Bytes(Arc<[u8]>),
}

impl HostData {
    pub fn class(&self) -> StorageClass {
        match self {
            HostData::F32(_) => StorageClass::F32,
            HostData::I32(_) => StorageClass::I32,
            HostData::Half(_) => StorageClass::Half,
            HostData::Bytes(_) => StorageClass::Bytes,
        }
    }

    /// The payload's size in bytes.
    pub fn byte_len(&self) -> usize {
        match self {
            HostData::F32(words) => std::mem::size_of_val(&**words),
            HostData::I32(words) => std::mem::size_of_val(&**words),
            HostData::Half(words) => std::mem::size_of_val(&**words),
            HostData::Bytes(bytes) => bytes.len(),
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            HostData::F32(words) => bytemuck::cast_slice(words),
            HostData::I32(words) => bytemuck::cast_slice(words),
            HostData::Half(words) => bytemuck::cast_slice(words),
            HostData::Bytes(bytes) => bytes,
        }
    }
}

/// A [`HostTensor`] could not be built or read as asked.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CarrierError {
    #[error("{dtype} is stored as {expected:?}, got a {got:?} payload")]
    StorageClass {
        dtype: DType,
        expected: StorageClass,
        got: StorageClass,
    },
    #[error(
        "{dtype} tensor of shape {shape:?} needs {expected_bytes} payload bytes, got {actual_bytes}"
    )]
    PayloadLength {
        dtype: DType,
        shape: Vec<usize>,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    #[error("shape {shape:?} overflows the element count")]
    ShapeOverflow { shape: Vec<usize> },
    #[error("{dtype} has no f32 widening")]
    NotWidenable { dtype: DType },
    #[error("i32 word {value} at element {index} is outside f32's exact integer range")]
    I32NotExact { index: usize, value: i32 },
}

/// The largest magnitude below which every integer is exactly an f32 (2^24).
const F32_EXACT_INTEGER_MAX: i32 = 1 << 24;

/// One owned host tensor: dtype, shape and a payload of the dtype's storage class. Fields are
/// private; every constructor keeps the payload consistent with the dtype and shape.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    dtype: DType,
    shape: Vec<usize>,
    data: HostData,
}

fn element_count(shape: &[usize]) -> Result<usize, CarrierError> {
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| CarrierError::ShapeOverflow {
            shape: shape.to_vec(),
        })
}

impl HostTensor {
    /// Build a tensor from a payload, checking that its storage class is the dtype's and that it
    /// holds exactly `shape`'s elements.
    pub fn new(dtype: DType, shape: Vec<usize>, data: HostData) -> Result<Self, CarrierError> {
        let expected = dtype.storage_class();
        if data.class() != expected {
            return Err(CarrierError::StorageClass {
                dtype,
                expected,
                got: data.class(),
            });
        }
        let expected_bytes = element_count(&shape)?.saturating_mul(dtype.byte_size());
        if data.byte_len() != expected_bytes {
            return Err(CarrierError::PayloadLength {
                dtype,
                shape,
                expected_bytes,
                actual_bytes: data.byte_len(),
            });
        }
        Ok(Self { dtype, shape, data })
    }

    /// Build a tensor from the little-endian bytes of its elements (a checkpoint tensor, or a
    /// [`HostView`]'s bytes), decoding them into the dtype's storage class.
    pub fn from_le_bytes(
        dtype: DType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<Self, CarrierError> {
        let data = match dtype.storage_class() {
            StorageClass::F32 => HostData::F32(
                bytes
                    .chunks_exact(4)
                    .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                    .collect(),
            ),
            StorageClass::I32 => HostData::I32(
                bytes
                    .chunks_exact(4)
                    .map(|word| i32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                    .collect(),
            ),
            StorageClass::Half => HostData::Half(
                bytes
                    .chunks_exact(2)
                    .map(|word| u16::from_le_bytes([word[0], word[1]]))
                    .collect(),
            ),
            StorageClass::Bytes => HostData::Bytes(Arc::from(bytes)),
        };
        // A trailing partial word is dropped by `chunks_exact`; `new` then sees a short payload.
        Self::new(dtype, shape, data)
    }

    /// An F32 tensor. Panics if `data` does not hold `shape`'s elements.
    pub fn f32(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self::f32_arc(shape, data.into())
    }

    /// An F32 tensor over an already-shared payload. Panics if it does not hold `shape`'s elements.
    pub fn f32_arc(shape: Vec<usize>, data: Arc<[f32]>) -> Self {
        Self::of_class(DType::F32, shape, HostData::F32(data))
    }

    /// An I32 tensor. Panics if `words` does not hold `shape`'s elements.
    pub fn i32(shape: Vec<usize>, words: Vec<i32>) -> Self {
        Self::i32_arc(shape, words.into())
    }

    /// An I32 tensor over an already-shared payload. Panics if it does not hold `shape`'s elements.
    pub fn i32_arc(shape: Vec<usize>, words: Arc<[i32]>) -> Self {
        Self::of_class(DType::I32, shape, HostData::I32(words))
    }

    /// A BF16 tensor from its bit patterns. Panics if `words` does not hold `shape`'s elements.
    pub fn bf16(shape: Vec<usize>, words: Vec<u16>) -> Self {
        Self::bf16_arc(shape, words.into())
    }

    /// A BF16 tensor over already-shared bit patterns. Panics if they do not hold `shape`'s elements.
    pub fn bf16_arc(shape: Vec<usize>, words: Arc<[u16]>) -> Self {
        Self::of_class(DType::BF16, shape, HostData::Half(words))
    }

    /// An F16 tensor from its bit patterns. Panics if `words` does not hold `shape`'s elements.
    pub fn f16(shape: Vec<usize>, words: Vec<u16>) -> Self {
        Self::f16_arc(shape, words.into())
    }

    /// An F16 tensor over already-shared bit patterns. Panics if they do not hold `shape`'s elements.
    pub fn f16_arc(shape: Vec<usize>, words: Arc<[u16]>) -> Self {
        Self::of_class(DType::F16, shape, HostData::Half(words))
    }

    /// A rank-0 F32 tensor.
    pub fn scalar(value: f32) -> Self {
        Self::f32(Vec::new(), vec![value])
    }

    /// An F32 tensor of zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let count = shape.iter().product();
        Self::f32(shape, vec![0.0f32; count])
    }

    fn of_class(dtype: DType, shape: Vec<usize>, data: HostData) -> Self {
        Self::new(dtype, shape, data).unwrap_or_else(|error| panic!("HostTensor: {error}"))
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn data(&self) -> &HostData {
        &self.data
    }

    /// The borrowed byte-level view of this tensor, the form the executor contract takes.
    pub fn view(&self) -> HostView<'_> {
        HostView::checked(self.dtype, self.numel(), self.data.bytes())
    }

    /// The elements as `f32`, if the dtype is `F32` (never a widening of another dtype).
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            HostData::F32(values) => Some(values),
            _ => None,
        }
    }

    /// The elements as `i32`, if the dtype is `I32`.
    pub fn as_i32(&self) -> Option<&[i32]> {
        match &self.data {
            HostData::I32(words) => Some(words),
            _ => None,
        }
    }

    /// The bit patterns of an `F16` or `BF16` tensor.
    pub fn as_half(&self) -> Option<&[u16]> {
        match &self.data {
            HostData::Half(words) => Some(words),
            _ => None,
        }
    }

    /// The same payload under a new shape of the same element count (the allocation is shared).
    pub fn reshaped(&self, shape: Vec<usize>) -> Result<Self, CarrierError> {
        Self::new(self.dtype, shape, self.data.clone())
    }

    /// The explicit widening to `f32`: borrowed for `F32`, an exact decode for `BF16`, `F16` and
    /// `E4M3FN` (one exception: both E4M3FN NaN bytes, `0x7f` and `0xff`, decode to the single positive
    /// canonical NaN, so the sign of an E4M3FN NaN is not preserved), and an `i32 as f32` cast for `I32` words whose magnitude is at most 2^24 (every
    /// such integer is an exact f32). Every other dtype, and an I32 word beyond that range, is a
    /// typed error; nothing else in this crate widens.
    pub fn to_f32(&self) -> Result<Cow<'_, [f32]>, CarrierError> {
        #[cfg(feature = "read-counter")]
        read_counter::record();
        match (&self.data, self.dtype) {
            (HostData::F32(values), _) => Ok(Cow::Borrowed(values)),
            (HostData::Half(words), DType::BF16) => Ok(Cow::Owned(
                words.iter().map(|&bits| bf16_to_f32(bits)).collect(),
            )),
            (HostData::Half(words), _) => Ok(Cow::Owned(
                words.iter().map(|&bits| f16_to_f32(bits)).collect(),
            )),
            (HostData::I32(words), _) => {
                if let Some((index, &value)) = words
                    .iter()
                    .enumerate()
                    .find(|(_, word)| word.unsigned_abs() > F32_EXACT_INTEGER_MAX as u32)
                {
                    return Err(CarrierError::I32NotExact { index, value });
                }
                Ok(Cow::Owned(words.iter().map(|&word| word as f32).collect()))
            }
            (HostData::Bytes(bytes), DType::E4M3FN) => Ok(Cow::Owned(
                bytes.iter().map(|&byte| e4m3fn_to_f32(byte)).collect(),
            )),
            (HostData::Bytes(_), dtype) => Err(CarrierError::NotWidenable { dtype }),
        }
    }
}

/// IEEE-754 binary16 to f32 (exact; subnormals, infinities and NaN payloads carry through).
const fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;
    match exponent {
        0 => {
            let magnitude = mantissa as f32 * (1.0 / 16_777_216.0);
            f32::from_bits(magnitude.to_bits() | sign)
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
    }
}

/// bfloat16 to f32: bf16 is the high half of an f32.
const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// OCP E4M3FN to f32: bias 7, no infinities; `0x7f` and `0xff` both decode to the one canonical
/// positive quiet NaN (`0x7fc0_0000`).
const fn e4m3fn_to_f32(byte: u8) -> f32 {
    let sign = ((byte & 0x80) as u32) << 24;
    let exponent = ((byte >> 3) & 0x0f) as u32;
    let mantissa = (byte & 0x07) as u32;
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::from_bits(0x7fc0_0000);
    }
    if exponent == 0 {
        let magnitude = mantissa as f32 * (1.0 / 512.0);
        return f32::from_bits(magnitude.to_bits() | sign);
    }
    f32::from_bits(sign | ((exponent + 120) << 23) | (mantissa << 20))
}

/// A count of [`HostTensor::to_f32`] calls, for the test that proves a bind path never widens.
/// Process-wide, so a test that asserts on it is red-capable only while each test runs in its own
/// process (nextest's isolation); under plain `cargo test` other threads' reads would race it.
#[cfg(feature = "read-counter")]
pub mod read_counter {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static READS: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn record() {
        READS.fetch_add(1, Ordering::Relaxed);
    }

    /// `to_f32` calls so far in this process.
    pub fn reads() -> usize {
        READS.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_DTYPES: [DType; 15] = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::E4M3FN,
        DType::E8M0,
    ];

    /// SC-001: a payload whose storage class does not fit the dtype, or whose length does not fit
    /// the shape, is a typed error from the one checked constructor.
    #[test]
    fn a_payload_that_does_not_fit_the_dtype_is_a_typed_error() {
        let f32_payload = HostData::F32(Arc::from([1.0f32, 2.0]));
        assert_eq!(
            HostTensor::new(DType::BF16, vec![2], f32_payload.clone()).unwrap_err(),
            CarrierError::StorageClass {
                dtype: DType::BF16,
                expected: StorageClass::Half,
                got: StorageClass::F32,
            }
        );
        assert_eq!(
            HostTensor::new(DType::I32, vec![2], f32_payload.clone()).unwrap_err(),
            CarrierError::StorageClass {
                dtype: DType::I32,
                expected: StorageClass::I32,
                got: StorageClass::F32,
            }
        );
        assert!(matches!(
            HostTensor::new(DType::E4M3FN, vec![2], f32_payload.clone()),
            Err(CarrierError::StorageClass { .. })
        ));
        assert_eq!(
            HostTensor::new(DType::F32, vec![3], f32_payload).unwrap_err(),
            CarrierError::PayloadLength {
                dtype: DType::F32,
                shape: vec![3],
                expected_bytes: 12,
                actual_bytes: 8,
            }
        );
        assert_eq!(
            HostTensor::new(DType::U16, vec![2], HostData::Bytes(Arc::from([0u8; 3]))).unwrap_err(),
            CarrierError::PayloadLength {
                dtype: DType::U16,
                shape: vec![2],
                expected_bytes: 4,
                actual_bytes: 3,
            }
        );
        assert!(
            HostTensor::new(
                DType::BF16,
                vec![2],
                HostData::Half(Arc::from([0x3f80u16, 0x8000]))
            )
            .is_ok()
        );
    }

    /// Literal element words for `count` elements of `dtype`, with a NaN payload and `-0.0` where the
    /// dtype has them, as little-endian bytes.
    fn literal_bytes(dtype: DType) -> (Vec<usize>, Vec<u8>) {
        match dtype {
            DType::F32 => (
                vec![4],
                [0x7fc0_0001u32, 0x8000_0000, 0x3f80_0000, 0xffff_ffff]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect(),
            ),
            DType::BF16 => (
                vec![4],
                [0x7fc1u16, 0x8000, 0x3f80, 0xffff]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect(),
            ),
            DType::F16 => (
                vec![4],
                [0x7e01u16, 0x8000, 0x3c00, 0xffff]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect(),
            ),
            DType::F64 => (
                vec![2],
                [0x7ff8_0000_0000_0001u64, 0x8000_0000_0000_0000]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect(),
            ),
            DType::I32 => (
                vec![4],
                [i32::MIN, -1, 0, i32::MAX]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect(),
            ),
            other => {
                let elems = 4;
                let bytes = (0..elems * other.byte_size())
                    .map(|i| (i as u8).wrapping_mul(37).wrapping_add(0x81))
                    .collect();
                (vec![elems], bytes)
            }
        }
    }

    /// SC-002: every dtype round-trips `HostTensor -> HostView -> bytes -> HostTensor` bitwise from
    /// literal words, NaN payloads and `-0.0` included, and the payload holds exactly those words.
    #[test]
    fn every_dtype_round_trips_through_a_view_bitwise() {
        for dtype in ALL_DTYPES {
            let (shape, bytes) = literal_bytes(dtype);
            let tensor = HostTensor::from_le_bytes(dtype, shape.clone(), &bytes)
                .unwrap_or_else(|error| panic!("{dtype}: {error}"));
            assert_eq!(tensor.dtype(), dtype);
            assert_eq!(tensor.data().class(), dtype.storage_class(), "{dtype}");
            assert_eq!(
                tensor.data().byte_len(),
                shape[0] * dtype.byte_size(),
                "{dtype}: payload width"
            );
            let view = tensor.view();
            assert_eq!(view.dtype(), dtype);
            assert_eq!(view.elems(), shape[0]);
            assert_eq!(view.bytes(), bytes.as_slice(), "{dtype}: view bytes");
            let again = HostTensor::from_le_bytes(dtype, shape, view.bytes()).unwrap();
            assert_eq!(
                again.view().bytes(),
                bytes.as_slice(),
                "{dtype}: second trip"
            );
        }
        // The words themselves, not just the bytes: NaN payloads survive in the typed payload.
        let (shape, bytes) = literal_bytes(DType::F32);
        let f32s = HostTensor::from_le_bytes(DType::F32, shape, &bytes).unwrap();
        let bits: Vec<u32> = f32s.as_f32().unwrap().iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, [0x7fc0_0001, 0x8000_0000, 0x3f80_0000, 0xffff_ffff]);
        let (shape, bytes) = literal_bytes(DType::BF16);
        let bf16s = HostTensor::from_le_bytes(DType::BF16, shape, &bytes).unwrap();
        assert_eq!(bf16s.as_half().unwrap(), [0x7fc1, 0x8000, 0x3f80, 0xffff]);
    }

    #[test]
    fn to_f32_widens_exactly_and_refuses_what_it_cannot() {
        let bf16 = HostTensor::bf16(vec![3], vec![0x3f80u16, 0x8000, 0x7fc1]);
        let widened = bf16.to_f32().unwrap();
        assert_eq!(widened[0].to_bits(), 0x3f80_0000);
        assert_eq!(widened[1].to_bits(), 0x8000_0000);
        assert_eq!(widened[2].to_bits(), 0x7fc1_0000);
        let f16 = HostTensor::f16(vec![2], vec![0x3c00u16, 0xc000]);
        assert_eq!(f16.to_f32().unwrap().as_ref(), [1.0, -2.0]);
        let e4m3 = HostTensor::new(
            DType::E4M3FN,
            vec![2],
            HostData::Bytes(Arc::from([0x38u8, 0xb8])),
        )
        .unwrap();
        assert_eq!(e4m3.to_f32().unwrap().as_ref(), [1.0, -1.0]);
        let words = HostTensor::i32(vec![2], vec![16_777_216, -16_777_216]);
        assert_eq!(
            words.to_f32().unwrap().as_ref(),
            [16_777_216.0, -16_777_216.0]
        );
        assert_eq!(
            HostTensor::i32(vec![2], vec![1, 16_777_217])
                .to_f32()
                .unwrap_err(),
            CarrierError::I32NotExact {
                index: 1,
                value: 16_777_217
            }
        );
        assert_eq!(
            HostTensor::new(DType::U8, vec![1], HostData::Bytes(Arc::from([1u8])))
                .unwrap()
                .to_f32()
                .unwrap_err(),
            CarrierError::NotWidenable { dtype: DType::U8 }
        );
    }

    #[test]
    fn reshape_shares_the_payload_and_rechecks_the_element_count() {
        let tensor = HostTensor::f32(vec![2, 3], vec![0.0f32; 6]);
        let flat = tensor.reshaped(vec![6]).unwrap();
        let (HostData::F32(a), HostData::F32(b)) = (tensor.data(), flat.data()) else {
            panic!("F32 payloads")
        };
        assert!(Arc::ptr_eq(a, b));
        assert!(matches!(
            tensor.reshaped(vec![5]),
            Err(CarrierError::PayloadLength { .. })
        ));
    }
}
