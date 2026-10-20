//! Core value types: dtypes, abstract values (shape+dtype, no data), and inline scalars.
//!
//! `DType` moved to [`poot_tensor::DType`]: it is the one dtype covering every
//! stored and graph dtype, imported directly here with no alias. [`DTypeClass`] stays: it is graph
//! semantics (which ops admit which dtype), not a tensor-storage fact, so it keeps classifying
//! `poot_tensor::DType` from this crate.

use std::fmt;

// Crate-internal only (not re-exported from `lib.rs`): every external caller imports
// `poot_tensor::DType` directly, with no alias through this crate.
pub(crate) use poot_tensor::DType;

/// Card 529 (R466-011): the class `infer` checks a dtype against, replacing the ad hoc per-op dtype
/// ladders that used to re-derive this by name. Every [`DType`] this crate's ops see has exactly one
/// class; there is no context-dependent overlap (a `Packed` value is never also an `Index`). The
/// dtypes `poot_tensor::DType` added for checkpoint storage only (`Bool`, `U8`, `I16`, `U16`, `U32`,
/// `I64`, `U64`, `F64`, `E8M0`) classify `StorageOnly`, the same as `E4M3FN`: none of them has
/// arithmetic meaning in a traced graph today.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DTypeClass {
    /// F32, BF16, F16: participate in elementwise arithmetic, reductions and matmul.
    Arithmetic,
    /// I32: the exact-integer dtype (position arithmetic, hash bit ops, an index/count value). Stays
    /// admitted wherever arithmetic is, and is the one dtype `infer` accepts for an index-role operand.
    Index,
    /// A logical storage dtype with no arithmetic meaning of its own (`E4M3FN`, `E8M0`, and every
    /// checkpoint-only integer/float width). Values enter or leave computation only through `Cast`;
    /// every other arithmetic op rejects it.
    StorageOnly,
    /// I8: an opaque raw-byte carrier (quantized-KV codes, `PackedDequant`'s validated payload).
    /// Arithmetic ops reject it the same way they reject `StorageOnly`; only `Cast`, `PackedDequant` and
    /// the movement ops (which never read a value's bit pattern) admit it.
    Packed,
}

/// `.class()` method-call syntax for [`DType`], kept local to this crate (the classification is
/// graph semantics; `poot-tensor` stays a rank-0 leaf with no knowledge of it).
pub trait DTypeClassExt {
    /// The class of this dtype.
    fn class(self) -> DTypeClass;
}

impl DTypeClassExt for DType {
    fn class(self) -> DTypeClass {
        match self {
            DType::F32 | DType::BF16 | DType::F16 => DTypeClass::Arithmetic,
            DType::I32 => DTypeClass::Index,
            DType::I8 => DTypeClass::Packed,
            DType::Bool
            | DType::U8
            | DType::I16
            | DType::U16
            | DType::U32
            | DType::I64
            | DType::U64
            | DType::F64
            | DType::E4M3FN
            | DType::E8M0 => DTypeClass::StorageOnly,
        }
    }
}

/// An abstract value (the jaxpr `ShapedArray` analog): shape + dtype, never any data. Every SSA value
/// carries one, computed host-side by [`crate::op::OpKind::infer`].
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TensorType {
    /// The extents, outermost axis first; empty for a scalar.
    pub shape: Vec<usize>,
    /// The element storage dtype.
    pub dtype: DType,
}

impl TensorType {
    /// A type of `shape` and `dtype`.
    pub fn new(shape: impl Into<Vec<usize>>, dtype: DType) -> Self {
        TensorType {
            shape: shape.into(),
            dtype,
        }
    }
    /// An F32 type of `shape`.
    pub fn f32(shape: impl Into<Vec<usize>>) -> Self {
        TensorType::new(shape, DType::F32)
    }
    /// A BF16 type of `shape`.
    pub fn bf16(shape: impl Into<Vec<usize>>) -> Self {
        TensorType::new(shape, DType::BF16)
    }
    /// An F16 type of `shape`.
    pub fn f16(shape: impl Into<Vec<usize>>) -> Self {
        TensorType::new(shape, DType::F16)
    }
    /// A rank-0 type of `dtype`.
    pub fn scalar(dtype: DType) -> Self {
        TensorType::new(Vec::new(), dtype)
    }
    /// The number of axes.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    /// The number of elements: the product of the extents, 1 for a scalar.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[", self.dtype)?;
        for (i, d) in self.shape.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// An inline scalar constant (the jaxpr `Literal`): eps, a scale, 1/N, an index bound, etc.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scalar {
    /// A 32-bit float literal.
    F32(f32),
    /// A 32-bit signed integer literal.
    I32(i32),
}

impl Scalar {
    /// The bytes of the literal's inline payload: what the tracer's literal budget counts.
    pub fn byte_len(&self) -> u64 {
        match self {
            Scalar::F32(_) => std::mem::size_of::<f32>() as u64,
            Scalar::I32(_) => std::mem::size_of::<i32>() as u64,
        }
    }

    /// The dtype the literal has: F32 or I32.
    pub fn dtype(&self) -> DType {
        match self {
            Scalar::F32(_) => DType::F32,
            Scalar::I32(_) => DType::I32,
        }
    }
    /// The rank-0 type of the literal.
    pub fn ty(&self) -> TensorType {
        TensorType::scalar(self.dtype())
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::F32(v) => write!(f, "{v}"),
            Scalar::I32(v) => write!(f, "{v}"),
        }
    }
}

/// The operands of an equation as [`crate::op::OpKind::infer`] reads them: their types, and which of them are
/// inline literals rather than values. A literal is an immediate the lowering bakes in, so it adopts the
/// dtype of the value it meets; a value of the same shape and dtype is a real tensor and must agree exactly.
/// `infer` is the only place that rule is stated, because it is the only place that sees both facts.
///
/// A list of value types converts into operands with no literal (`ins.into()`): that is every operand list
/// outside an equation's own `inputs`, such as a pass probing a candidate op.
#[derive(Clone, Copy)]
pub struct Operands<'a> {
    types: &'a [TensorType],
    is_literal: &'a dyn Fn(usize) -> bool,
}

/// The literal predicate of a list of values: nothing is a literal.
fn no_literals(_: usize) -> bool {
    false
}

impl<'a> Operands<'a> {
    /// The operand `types` of an equation, operand `i` being a literal exactly when `is_literal(i)`.
    pub fn new(types: &'a [TensorType], is_literal: &'a dyn Fn(usize) -> bool) -> Self {
        Self { types, is_literal }
    }

    /// The operand types, in operand order.
    pub fn types(&self) -> &'a [TensorType] {
        self.types
    }

    /// Whether operand `index` is an inline literal.
    pub fn is_literal(&self, index: usize) -> bool {
        (self.is_literal)(index)
    }
}

impl<'a> From<&'a [TensorType]> for Operands<'a> {
    fn from(types: &'a [TensorType]) -> Self {
        Self::new(types, &no_literals)
    }
}

impl<'a> From<&'a Vec<TensorType>> for Operands<'a> {
    fn from(types: &'a Vec<TensorType>) -> Self {
        types.as_slice().into()
    }
}

impl<'a, const N: usize> From<&'a [TensorType; N]> for Operands<'a> {
    fn from(types: &'a [TensorType; N]) -> Self {
        types.as_slice().into()
    }
}

/// numpy broadcast of two shapes (right-aligned, leading-1 padding). The shared rule behind binary ops
/// and matmul batch dims.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    for i in 0..n {
        let da = if i + a.len() < n {
            1
        } else {
            a[i + a.len() - n]
        };
        let db = if i + b.len() < n {
            1
        } else {
            b[i + b.len() - n]
        };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_byte_sizes() {
        // spec 048: I8 KV storage is 1 byte (the ~4x win vs F32). Display round-trips the new variant.
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::BF16.byte_size(), 2);
        assert_eq!(DType::F16.byte_size(), 2);
        assert_eq!(DType::I32.byte_size(), 4);
        assert_eq!(DType::I8.byte_size(), 1);
        assert_eq!(DType::E4M3FN.byte_size(), 1);
        assert_eq!(DType::I8.to_string(), "i8");
        assert_eq!(DType::F16.to_string(), "f16");
        assert_eq!(DType::E4M3FN.to_string(), "e4m3fn");
    }

    #[test]
    fn broadcast_basic() {
        assert_eq!(
            broadcast_shapes(&[1, 1, 896], &[896]),
            Some(vec![1, 1, 896])
        );
        assert_eq!(
            broadcast_shapes(&[1, 14, 1, 64], &[64]),
            Some(vec![1, 14, 1, 64])
        );
        assert_eq!(
            broadcast_shapes(&[1, 1, 896], &[1, 1, 1]),
            Some(vec![1, 1, 896])
        );
        assert_eq!(broadcast_shapes(&[2, 3], &[4, 3]), None);
        assert_eq!(broadcast_shapes(&[], &[5]), Some(vec![5]));
    }

    #[test]
    fn broadcast_edge_cases() {
        // Commutative: broadcasting is symmetric in a/b.
        assert_eq!(broadcast_shapes(&[5], &[]), Some(vec![5]));
        assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[1, 4], &[3, 1]), Some(vec![3, 4]));
        // Scalar (empty shape) broadcasts against anything; scalar+scalar stays scalar.
        assert_eq!(broadcast_shapes(&[], &[]), Some(vec![]));
        assert_eq!(broadcast_shapes(&[], &[2, 3, 4]), Some(vec![2, 3, 4]));
        // Right-alignment with differing ranks: the shorter operand's missing leading dims are 1.
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[4]), Some(vec![2, 3, 4]));
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[3, 1]), Some(vec![2, 3, 4]));
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[3, 4]), Some(vec![2, 3, 4]));
        // A size-0 dim is preserved (1 broadcasts INTO 0, per numpy); 0 vs a non-1/-0 size is incompatible.
        assert_eq!(broadcast_shapes(&[0], &[1]), Some(vec![0]));
        assert_eq!(broadcast_shapes(&[1], &[0]), Some(vec![0]));
        assert_eq!(broadcast_shapes(&[0], &[0]), Some(vec![0]));
        assert_eq!(broadcast_shapes(&[0], &[2]), None);
        // Incompatible non-1 dims anywhere in the aligned tail -> None.
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[2, 9, 4]), None);
        assert_eq!(broadcast_shapes(&[5, 2], &[5, 3]), None);
    }
}
