//! Exact BF16 values: the Card 370 word carrier `walk.rs` evaluates the Spec 376 equations over.
//!
//! A BF16 element is one 16-bit word, decoded with the Card 370 rule. [`crate::resolve::classify`]
//! names the closed set of equations a BF16-touching value may appear in (the admission table, moved
//! there alongside `resolve`'s other admission logic); `walk.rs` refuses every other one before any
//! equation runs (this file carries no walk of its own, and now no admission table either - only
//! the carrier).

use poot_quant::scalar::bf16_to_f32;
use std::sync::Arc;

/// Row-major BF16 words, possibly permuted, over words an equation derived.
#[derive(Clone, Debug)]
pub struct ExactBf16TensorView {
    words: Arc<[u16]>,
    shape: Vec<usize>,
    /// Physical word stride of each logical axis.
    strides: Vec<usize>,
    /// Whether `strides` are the canonical row-major strides of `shape`, fixed when the view is built.
    contiguous: bool,
}

impl PartialEq for ExactBf16TensorView {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.words, &other.words)
            && self.shape == other.shape
            && self.strides == other.strides
    }
}

impl Eq for ExactBf16TensorView {}

impl ExactBf16TensorView {
    /// A view over words an equation just derived (a gather's selected rows, a widened host
    /// tensor): just this allocation.
    pub(crate) fn from_derived(
        words: Arc<[u16]>,
        shape: Vec<usize>,
    ) -> Result<Self, ExactBf16Error> {
        Self::canonical(words, shape)
    }

    /// A row-major view of `words` with canonical strides.
    fn canonical(words: Arc<[u16]>, shape: Vec<usize>) -> Result<Self, ExactBf16Error> {
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
        self.words.len().saturating_mul(2)
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
        self.words[physical]
    }

    /// Decoded f32 value at row-major logical element `flat`.
    pub fn value(&self, flat: usize) -> f32 {
        bf16_to_f32(self.word(flat))
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
