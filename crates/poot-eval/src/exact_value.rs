//! Typed exact-value carrier: BF16/F32 checkpoint owner views, the exact-I32 word view, and the one
//! `ExactValue` enum every walk path over a non-f32-tensor carrier matches on.

use poot_tensor::DType;
use std::sync::Arc;

use crate::EvalError;
use crate::exact_bf16::ExactBf16TensorView;
use crate::exact_dense::{DenseOwnerTensorView, ExactDenseError};
use poot_tensor::HostTensor;

/// Row-major authoritative I32 words (spec 048): a checkpoint-derived or equation-derived view, never
/// the lossy `i32 as f32` mirror. Moved here from the deleted `exact_i32.rs`: the walk's
/// I32 handling is one path over `Value::Host`/`Value::Owner(ExactValue::I32)` (`operand.rs`), not a
/// separate lane.
#[derive(Clone, Debug)]
pub struct ExactI32TensorView {
    shape: Vec<usize>,
    words: Arc<[i32]>,
}

impl PartialEq for ExactI32TensorView {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && Arc::ptr_eq(&self.words, &other.words)
    }
}

impl Eq for ExactI32TensorView {}

impl ExactI32TensorView {
    pub fn try_from_words(shape: Vec<usize>, words: Arc<[i32]>) -> Result<Self, EvalError> {
        let expected: usize = shape.iter().product();
        if expected != words.len() {
            return Err(EvalError::unsupported(
                "exact_i32_view",
                format!(
                    "I32 shape {shape:?} expected {expected} words, got {}",
                    words.len()
                ),
            ));
        }
        Ok(Self { shape, words })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.words.len()
    }

    pub fn i32_words(&self) -> &[i32] {
        &self.words
    }

    pub fn word_owner(&self) -> &Arc<[i32]> {
        &self.words
    }

    /// The same authoritative words under a new shape (card 383): a view, so the word allocation is
    /// shared and the element count is re-checked against the new shape.
    pub(crate) fn reshaped(&self, shape: Vec<usize>) -> Result<Self, EvalError> {
        Self::try_from_words(shape, Arc::clone(&self.words))
    }
}

/// Exact CPU storage whose authoritative payload does not use the legacy f32 tensor lane.
///
/// New exact kinds are added here and handled exhaustively inside `poot-eval`. The enum is
/// `#[non_exhaustive]`, so adding a kind never requires a source edit in another crate. A reverse
/// dependency may match the kinds it executes, but it must reject every other kind with a fallback arm:
///
/// ```
/// use poot_eval::ExactValue;
///
/// fn admits(value: &ExactValue) -> bool {
///     match value {
///         ExactValue::Dense(_) | ExactValue::I32(_) => true,
///         _ => false,
///     }
/// }
/// ```
///
/// A match that names every current kind and omits the fallback does not compile:
///
/// ```compile_fail,E0004
/// use poot_eval::ExactValue;
///
/// fn admits(value: &ExactValue) -> bool {
///     match value {
///         ExactValue::Dense(_) | ExactValue::I32(_) => true,
///         ExactValue::Bf16(_) => false,
///     }
/// }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ExactValue {
    Dense(DenseOwnerTensorView),
    Bf16(ExactBf16TensorView),
    I32(ExactI32TensorView),
}

impl Eq for ExactValue {}

impl ExactValue {
    pub fn dtype(&self) -> DType {
        match self {
            Self::Dense(view) => view.dtype(),
            Self::Bf16(_) => DType::BF16,
            Self::I32(_) => DType::I32,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Dense(view) => view.shape(),
            Self::Bf16(view) => view.shape(),
            Self::I32(view) => view.shape(),
        }
    }

    pub fn numel(&self) -> usize {
        match self {
            Self::Dense(view) => view.numel(),
            Self::Bf16(view) => view.numel(),
            Self::I32(view) => view.numel(),
        }
    }

    pub fn physical_bytes(&self) -> usize {
        match self {
            Self::Dense(view) => view.source_byte_len(),
            Self::Bf16(view) => view.physical_bytes(),
            Self::I32(view) => view.numel().saturating_mul(std::mem::size_of::<i32>()),
        }
    }

    /// Decode this value into an ordinary F32 [`HostTensor`] for the one-equation dense bridge's operand
    /// loop (activations, norm weights). Never called for [`Self::I32`], which reads through
    /// [`ExactI32TensorView::i32_words`].
    ///
    /// Do not call this from `Gather`'s data operand or any site whose source may be a large
    /// checkpoint table read by few rows: it decodes the whole source (Qwen3.5's `embed.weight` is
    /// 248,320 x 5,120, ~5GB as F32). `ops::gather::gather_owner` exists for that case.
    pub(crate) fn materialize_dense(&self) -> HostTensor {
        match self {
            Self::Dense(view) => {
                let numel: usize = view.shape().iter().product();
                let data: Vec<f32> = match view.dtype() {
                    DType::BF16 => {
                        let bf16 = ExactBf16TensorView::from_owner_view(view).expect(
                            "DenseOwnerTensorView::new only ever pairs dtype BF16 with an owner of \
                             ExactSourceKind::Bf16",
                        );
                        (0..numel).map(|flat| bf16.value(flat)).collect()
                    }
                    DType::F32 => view
                        .bytes()
                        .chunks_exact(4)
                        .map(|word| f32::from_le_bytes(word.try_into().expect("chunks_exact(4)")))
                        .collect(),
                    other => unreachable!(
                        "DenseOwnerTensorView::new only admits BF16 or F32, got {other:?}"
                    ),
                };
                HostTensor::f32(view.shape().to_vec(), data)
            }
            Self::Bf16(view) => {
                let data: Vec<f32> = (0..view.numel()).map(|flat| view.value(flat)).collect();
                HostTensor::f32(view.shape().to_vec(), data)
            }
            Self::I32(_) => unreachable!("materialize_dense is never called for ExactValue::I32"),
        }
    }

    pub(crate) fn reshape(&self, shape: Vec<usize>) -> Result<Self, EvalError> {
        match self {
            Self::Dense(view) => view.reshape(shape).map(Self::Dense).map_err(Into::into),
            Self::Bf16(view) => view.reshape(shape).map(Self::Bf16).map_err(Into::into),
            // A view over authoritative words: the word allocation is shared, so this is a metadata
            // relabel, not a re-materialization (card 383).
            Self::I32(view) => view.reshaped(shape).map(Self::I32),
        }
    }

    /// `Transpose` of an exact value: only a BF16 owner/derived view supports it (the Spec 376 table);
    /// an I32 or dense-owner carrier has no permuted-stride form.
    pub(crate) fn transpose(&self, perm: &[usize]) -> Result<crate::Value, EvalError> {
        match self {
            Self::Bf16(view) => Ok(crate::Value::Owner(Self::Bf16(view.transpose(perm)))),
            Self::Dense(_) | Self::I32(_) => Err(self.unsupported("Transpose")),
        }
    }

    pub(crate) fn unsupported(&self, consumer: &'static str) -> EvalError {
        match self {
            Self::Dense(_) => ExactDenseError::ClosedSurface { consumer }.into(),
            Self::Bf16(_) => {
                EvalError::unsupported(consumer, "exact BF16 carrier does not support this op")
            }
            Self::I32(_) => {
                EvalError::unsupported(consumer, "exact I32 carrier does not support this op")
            }
        }
    }
}

impl From<DenseOwnerTensorView> for ExactValue {
    fn from(value: DenseOwnerTensorView) -> Self {
        Self::Dense(value)
    }
}

impl From<ExactBf16TensorView> for ExactValue {
    fn from(value: ExactBf16TensorView) -> Self {
        Self::Bf16(value)
    }
}

impl From<ExactI32TensorView> for ExactValue {
    fn from(value: ExactI32TensorView) -> Self {
        Self::I32(value)
    }
}
