//! Exact BF16 values: the Card 370 word carrier `walk.rs` evaluates the Spec 376 equations over.
//!
//! A BF16 element is one 16-bit word, decoded with the Card 370 rule. [`crate::resolve::classify`]
//! names the closed set of equations a BF16-touching value may appear in (the admission table, moved
//! there alongside `resolve`'s other admission logic); `walk.rs` refuses
//! every other one before any equation runs (this file carries no walk of its
//! own, and now no admission table either - only the carrier).

use poot_tensor::DType;
use std::sync::Arc;

use crate::exact_dense::{DenseOwnerTensorView, bf16_word, decode_bf16_word};

/// Row-major BF16 words, possibly permuted, over a Card 370 owner or over words an equation derived.
#[derive(Clone, Debug)]
pub struct ExactBf16TensorView {
    words: Bf16Words,
    shape: Vec<usize>,
    /// Physical word stride of each logical axis.
    strides: Vec<usize>,
    /// Whether `strides` are the canonical row-major strides of `shape`, fixed when the view is built.
    contiguous: bool,
}

#[derive(Clone, Debug)]
enum Bf16Words {
    /// Authoritative checkpoint bytes, two little-endian bytes per word. `word_offset` is the
    /// element index where this view starts inside the owner (nonzero for a row-sliced view such
    /// as a vocab chunk of the LM head).
    Owner {
        owner: Arc<ExactSourceOwner>,
        word_offset: usize,
    },
    /// Words produced by an evaluated equation.
    Derived(Arc<[u16]>),
}

use poot_load::packed_safetensors::{ExactSourceKind, ExactSourceOwner};

impl PartialEq for ExactBf16TensorView {
    fn eq(&self, other: &Self) -> bool {
        let same_words = match (&self.words, &other.words) {
            (
                Bf16Words::Owner {
                    owner: left,
                    word_offset: lo,
                },
                Bf16Words::Owner {
                    owner: right,
                    word_offset: ro,
                },
            ) => Arc::ptr_eq(left, right) && lo == ro,
            (Bf16Words::Derived(left), Bf16Words::Derived(right)) => Arc::ptr_eq(left, right),
            _ => false,
        };
        same_words && self.shape == other.shape && self.strides == other.strides
    }
}

impl Eq for ExactBf16TensorView {}

impl ExactBf16TensorView {
    /// Alias a validated Card 370 BF16 owner view. The owner bytes are not copied or decoded.
    /// A row-sliced view starts at its byte offset (converted to a word offset).
    pub fn from_owner_view(view: &DenseOwnerTensorView) -> Result<Self, ExactBf16Error> {
        if view.owner().kind() != ExactSourceKind::Bf16 {
            return Err(ExactBf16Error::OwnerDtype {
                dtype: view.dtype(),
            });
        }
        let word_offset = view.byte_offset() / 2;
        Self::canonical(
            Bf16Words::Owner {
                owner: Arc::clone(view.owner()),
                word_offset,
            },
            view.shape().to_vec(),
        )
    }

    /// A view over words an equation just derived (a gather's selected rows): no owner, no checkpoint
    /// bytes, just this allocation.
    pub(crate) fn from_derived(
        words: Arc<[u16]>,
        shape: Vec<usize>,
    ) -> Result<Self, ExactBf16Error> {
        Self::canonical(Bf16Words::Derived(words), shape)
    }

    /// A row-major view of `words` with canonical strides.
    fn canonical(words: Bf16Words, shape: Vec<usize>) -> Result<Self, ExactBf16Error> {
        Ok(Self {
            words,
            strides: canonical_strides(&shape)?,
            shape,
            contiguous: true,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Bytes of the word storage this view reads, shared by every alias of that storage.
    pub fn physical_bytes(&self) -> usize {
        match &self.words {
            Bf16Words::Owner { owner, .. } => owner.bytes().len(),
            Bf16Words::Derived(words) => words.len().saturating_mul(2),
        }
    }

    /// The word at row-major logical element `flat`.
    pub fn word(&self, flat: usize) -> u16 {
        let physical = if self.contiguous {
            flat
        } else {
            let mut rest = flat;
            let mut physical = 0;
            for axis in (0..self.shape.len()).rev() {
                physical += (rest % self.shape[axis]) * self.strides[axis];
                rest /= self.shape[axis];
            }
            physical
        };
        match &self.words {
            Bf16Words::Owner { owner, word_offset } => {
                let bytes = owner.bytes();
                let index = word_offset + physical;
                bf16_word([bytes[2 * index], bytes[2 * index + 1]])
            }
            Bf16Words::Derived(words) => words[physical],
        }
    }

    /// Decoded f32 value at row-major logical element `flat`.
    pub fn value(&self, flat: usize) -> f32 {
        decode_bf16_word(self.word(flat))
    }

    /// Contiguous alias with another logical shape.
    pub(crate) fn reshape(&self, shape: Vec<usize>) -> Result<Self, ExactBf16Error> {
        let numel =
            crate::ops::element_count(&shape).map_err(|_| ExactBf16Error::ArithmeticOverflow {
                field: "element count",
            })?;
        if numel != self.numel() {
            return Err(ExactBf16Error::ReshapeElementCount {
                source_shape: self.shape.clone(),
                target_shape: shape,
            });
        }
        if !self.contiguous {
            return Err(ExactBf16Error::NonContiguousReshape {
                source_shape: self.shape.clone(),
                target_shape: shape,
            });
        }
        Self::canonical(self.words.clone(), shape)
    }

    pub(crate) fn transpose(&self, perm: &[usize]) -> Self {
        let shape = perm
            .iter()
            .map(|&axis| self.shape[axis])
            .collect::<Vec<_>>();
        let strides = perm
            .iter()
            .map(|&axis| self.strides[axis])
            .collect::<Vec<_>>();
        let mut stride = 1;
        let mut contiguous = true;
        for (extent, actual) in shape.iter().zip(&strides).rev() {
            contiguous &= *extent == 1 || *actual == stride;
            stride *= extent;
        }
        Self {
            words: self.words.clone(),
            shape,
            strides,
            contiguous,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExactBf16Error {
    #[error("exact BF16 view requires a BF16 owner, got {dtype}")]
    OwnerDtype { dtype: DType },
    #[error("BF16 reshape {source_shape:?} -> {target_shape:?} changes the element count")]
    ReshapeElementCount {
        source_shape: Vec<usize>,
        target_shape: Vec<usize>,
    },
    #[error("BF16 reshape {source_shape:?} -> {target_shape:?} needs a contiguous alias")]
    NonContiguousReshape {
        source_shape: Vec<usize>,
        target_shape: Vec<usize>,
    },
    #[error("exact BF16 arithmetic overflowed {field}")]
    ArithmeticOverflow { field: &'static str },
}

/// Row-major strides of `shape`, routed through the one lane-neutral overflow check
/// (`ops::row_major_strides`) instead of restating it here.
pub(crate) fn canonical_strides(shape: &[usize]) -> Result<Vec<usize>, ExactBf16Error> {
    crate::ops::row_major_strides(shape).map_err(|_| ExactBf16Error::ArithmeticOverflow {
        field: "element count",
    })
}
