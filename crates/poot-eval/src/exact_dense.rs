//! Exact BF16/F32 checkpoint owners for the typed CPU value boundary.

use poot_tensor::DType;
use std::collections::HashMap;
use std::sync::Arc;

use poot_graph_ir::{
    Graph, GraphValidationError, OpKind, Operand, Storage, ValidationChannel, ValueId,
};
use poot_load::packed_safetensors::{
    AuthenticatedArtifactIdentity, AuthenticatedTensorDescriptor, ExactSourceKind, ExactSourceOwner,
};

use crate::exact_value::ExactValue;
use crate::resolve;
use crate::{EvalError, Value};

/// A contiguous logical view over the whole of one canonical exact-source owner, retained by `Arc`
/// identity ([`DenseOwnerTensorView::new`]).
#[derive(Clone, Debug)]
pub struct DenseOwnerTensorView {
    owner: Arc<ExactSourceOwner>,
    dtype: DType,
    shape: Vec<usize>,
    numel: usize,
    /// Byte offset into `owner.bytes()` where this view's payload starts (0 for a whole-owner view).
    byte_offset: usize,
    /// Graph constant name this view binds to when it differs from the owner descriptor name
    /// (vocab-chunked `lm_head.weight.chunk*` consts). `None` matches the descriptor name.
    bind_name: Option<String>,
}

impl PartialEq for DenseOwnerTensorView {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
            && self.shape == other.shape
            && self.byte_offset == other.byte_offset
    }
}

impl Eq for DenseOwnerTensorView {}

impl DenseOwnerTensorView {
    /// Validate and retain a canonical BF16 or F32 source owner without decoding it.
    pub fn new(owner: Arc<ExactSourceOwner>) -> Result<Self, ExactDenseError> {
        let (dtype, expected_source_dtype) = dense_kind(owner.kind())?;
        let descriptor = owner.descriptor();
        if descriptor.dtype() != expected_source_dtype {
            return Err(ExactDenseError::OwnerDtypeMismatch {
                kind: owner.kind(),
                descriptor_dtype: descriptor.dtype().to_string(),
            });
        }
        let shape = descriptor.shape().to_vec();
        let numel = crate::ops::element_count(&shape).map_err(|_| {
            ExactDenseError::ElementCountOverflow {
                shape: shape.clone(),
            }
        })?;
        let expected = checked_source_bytes(dtype, numel)?;
        let actual = owner.bytes().len();
        if actual != expected {
            return Err(ExactDenseError::ByteLengthMismatch {
                dtype,
                shape,
                expected,
                actual,
            });
        }
        Ok(Self {
            owner,
            dtype,
            shape,
            numel,
            byte_offset: 0,
            bind_name: None,
        })
    }

    pub fn owner(&self) -> &Arc<ExactSourceOwner> {
        &self.owner
    }

    pub fn artifact(&self) -> &AuthenticatedArtifactIdentity {
        self.owner.artifact()
    }

    pub fn descriptor(&self) -> &AuthenticatedTensorDescriptor {
        self.owner.descriptor()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub const fn numel(&self) -> usize {
        self.numel
    }

    /// Byte offset of this view inside the owner's payload.
    pub const fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// The graph constant name this view binds to: the explicit bind name when set (a vocab-chunk
    /// const), otherwise the owner descriptor name.
    pub fn bind_name(&self) -> &str {
        self.bind_name
            .as_deref()
            .unwrap_or_else(|| self.descriptor().name())
    }

    /// This view's payload bytes inside the owner. Whole-owner views return the full owner bytes.
    pub fn bytes(&self) -> &[u8] {
        let end = self.byte_offset + self.view_byte_len();
        &self.owner.bytes()[self.byte_offset..end]
    }

    /// Bytes this view occupies in the owner (`numel * dtype` width).
    pub fn view_byte_len(&self) -> usize {
        self.numel * self.dtype.byte_size()
    }

    /// Source bytes of this view (not the whole owner). For a whole-owner view this equals
    /// `owner.bytes().len()`; for a row slice it is the slice's span.
    pub fn source_byte_len(&self) -> usize {
        self.view_byte_len()
    }

    /// Create a contiguous alias with a different logical shape and unchanged source byte order.
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, ExactDenseError> {
        let actual = crate::ops::element_count(&shape).map_err(|_| {
            ExactDenseError::ElementCountOverflow {
                shape: shape.clone(),
            }
        })?;
        if actual != self.numel {
            return Err(ExactDenseError::ReshapeElementCount {
                source_shape: self.shape.clone(),
                target_shape: shape,
                expected: self.numel,
                actual,
            });
        }
        Ok(Self {
            owner: Arc::clone(&self.owner),
            dtype: self.dtype,
            shape,
            numel: self.numel,
            byte_offset: self.byte_offset,
            bind_name: self.bind_name.clone(),
        })
    }
}

/// Fail-closed input checks for every owner-backed exact-dense binding:
/// each bound view's metadata must match its graph constant, and a BF16/F32 owner view not consumed
/// by an admitted Spec 376 equation (`resolve::classify`) - and not the graph's own output - is an
/// unconsumed non-output binding, not a silent pass.
pub(crate) fn validate_exact_dense_bindings<V: ValidationChannel>(
    g: &Graph<V>,
    inputs: &HashMap<ValueId, Value>,
) -> Result<(), EvalError> {
    for (&value_id, value) in inputs {
        let Value::Owner(ExactValue::Dense(view)) = value else {
            continue;
        };
        validate_exact_dense_binding_metadata(g, value_id, view)?;
    }
    let graph_consumer = |value_id: ValueId, consumer: String| -> EvalError {
        let meta = g.meta(value_id);
        ExactDenseError::GraphConsumer {
            value_id,
            dtype: meta.aval.dtype,
            shape: meta.aval.shape.clone(),
            storage: meta.storage,
            consumer,
        }
        .into()
    };
    let mut consumed = std::collections::HashSet::new();
    for eqn in &g.eqns {
        // An owner-backed dense view also feeds a plain `Gather` (any residency dtype, card 396's
        // lazy decode): `evaluate_gather`'s own `ExactValue::Dense` arm reads it directly whenever the
        // index is I32, regardless of the Spec 376 BF16-table admission `classify` checks for.
        let is_i32_indexed_gather_data = matches!(eqn.op, OpKind::Gather { .. })
            && matches!(
                eqn.inputs.get(1),
                Some(Operand::Value(index)) if g.aval(*index).dtype == DType::I32
            );
        for (position, operand) in eqn.inputs.iter().enumerate() {
            let Operand::Value(value_id) = operand else {
                continue;
            };
            let Some(Value::Owner(ExactValue::Dense(view))) = inputs.get(value_id) else {
                continue;
            };
            let admitted = (view.dtype() == DType::BF16 && resolve::classify(g, eqn).is_some())
                || (position == 0 && is_i32_indexed_gather_data);
            if !admitted {
                return Err(graph_consumer(*value_id, eqn.op.name()));
            }
            consumed.insert(*value_id);
        }
    }
    for (&value_id, value) in inputs {
        if matches!(value, Value::Owner(ExactValue::Dense(_)))
            && value_id != g.output
            && !consumed.contains(&value_id)
        {
            return Err(graph_consumer(
                value_id,
                "non-output exact binding".to_string(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_exact_dense_binding_metadata<V: ValidationChannel>(
    graph: &Graph<V>,
    value_id: ValueId,
    view: &DenseOwnerTensorView,
) -> Result<(), ExactDenseError> {
    let meta = graph
        .values
        .get(value_id)
        .ok_or(ExactDenseError::ValueIdOutOfRange {
            value_id,
            value_count: graph.values.len(),
        })?;
    let graph_name = meta
        .name
        .as_deref()
        .ok_or(ExactDenseError::MissingGraphName { value_id })?;
    if graph_name != view.bind_name() {
        return Err(ExactDenseError::NameMismatch {
            value_id,
            graph_name: graph_name.to_string(),
            owner_name: view.bind_name().to_string(),
        });
    }
    if meta.storage != Storage::Const {
        return Err(ExactDenseError::StorageMismatch {
            value_id,
            name: graph_name.to_string(),
            actual: meta.storage,
        });
    }
    let (owner_dtype, descriptor_dtype) = dense_kind(view.owner.kind())?;
    if view.descriptor().dtype() != descriptor_dtype {
        return Err(ExactDenseError::OwnerDtypeMismatch {
            kind: view.owner.kind(),
            descriptor_dtype: view.descriptor().dtype().to_string(),
        });
    }
    if owner_dtype != view.dtype || meta.aval.dtype != view.dtype {
        return Err(ExactDenseError::DtypeMismatch {
            value_id,
            name: graph_name.to_string(),
            graph_dtype: meta.aval.dtype,
            owner_dtype,
        });
    }
    // A whole-owner view covers every owner element; a row slice covers a subrange of axis 0 of a
    // rank-2 owner (the vocab-split LM head), so the element counts differ by design.
    let owner_shape = view.descriptor().shape();
    let owner_numel = crate::ops::element_count(owner_shape).map_err(|_| {
        ExactDenseError::ElementCountOverflow {
            shape: owner_shape.to_vec(),
        }
    })?;
    let whole_owner = view.byte_offset == 0
        && view
            .bind_name
            .as_deref()
            .is_none_or(|name| name == view.descriptor().name());
    if whole_owner {
        if owner_numel != view.numel {
            return Err(ExactDenseError::OwnerElementCountMismatch {
                value_id,
                owner_shape: view.descriptor().shape().to_vec(),
                view_shape: view.shape.clone(),
            });
        }
    } else {
        let owner_shape = view.descriptor().shape();
        if owner_shape.len() != 2 || view.shape.len() != 2 || owner_shape[1] != view.shape[1] {
            return Err(ExactDenseError::OwnerElementCountMismatch {
                value_id,
                owner_shape: owner_shape.to_vec(),
                view_shape: view.shape.clone(),
            });
        }
        if view.shape[0] > owner_shape[0] || view.numel > owner_numel {
            return Err(ExactDenseError::OwnerElementCountMismatch {
                value_id,
                owner_shape: owner_shape.to_vec(),
                view_shape: view.shape.clone(),
            });
        }
    }
    if meta.aval.shape != view.shape {
        return Err(ExactDenseError::ShapeMismatch {
            value_id,
            name: graph_name.to_string(),
            graph_shape: meta.aval.shape.clone(),
            owner_shape: view.shape.clone(),
        });
    }
    let expected = checked_source_bytes(view.dtype, view.numel)?;
    if expected != view.source_byte_len() {
        return Err(ExactDenseError::ByteLengthMismatch {
            dtype: view.dtype,
            shape: view.shape.clone(),
            expected,
            actual: view.source_byte_len(),
        });
    }
    Ok(())
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExactDenseError {
    #[error("exact dense view does not admit Card 359 source kind {kind:?}")]
    UnsupportedSourceKind { kind: ExactSourceKind },
    #[error("exact source kind {kind:?} disagrees with descriptor dtype {descriptor_dtype}")]
    OwnerDtypeMismatch {
        kind: ExactSourceKind,
        descriptor_dtype: String,
    },
    #[error("exact dense shape element count overflowed for {shape:?}")]
    ElementCountOverflow { shape: Vec<usize> },
    #[error("exact dense {dtype} byte count overflowed for {numel} elements")]
    ByteCountOverflow { dtype: DType, numel: usize },
    #[error(
        "exact dense {dtype} source for shape {shape:?} has {actual} bytes, expected {expected}"
    )]
    ByteLengthMismatch {
        dtype: DType,
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },
    #[error(
        "exact dense reshape {source_shape:?} -> {target_shape:?} changes element count {expected} -> {actual}"
    )]
    ReshapeElementCount {
        source_shape: Vec<usize>,
        target_shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },
    #[error("exact dense binding graph is invalid: {0}")]
    InvalidGraph(#[from] GraphValidationError),
    #[error("exact dense binding id v{value_id} is outside graph value count {value_count}")]
    ValueIdOutOfRange {
        value_id: ValueId,
        value_count: usize,
    },
    #[error("exact dense binding v{value_id} has no graph name")]
    MissingGraphName { value_id: ValueId },
    #[error(
        "exact dense binding v{value_id} graph name {graph_name} does not match owner name {owner_name}"
    )]
    NameMismatch {
        value_id: ValueId,
        graph_name: String,
        owner_name: String,
    },
    #[error("exact dense binding v{value_id} ({name}) requires Const storage, got {actual:?}")]
    StorageMismatch {
        value_id: ValueId,
        name: String,
        actual: Storage,
    },
    #[error(
        "exact dense binding v{value_id} ({name}) graph dtype {graph_dtype} does not match owner dtype {owner_dtype}"
    )]
    DtypeMismatch {
        value_id: ValueId,
        name: String,
        graph_dtype: DType,
        owner_dtype: DType,
    },
    #[error(
        "exact dense binding v{value_id} ({name}) graph shape {graph_shape:?} does not match owner shape {owner_shape:?}"
    )]
    ShapeMismatch {
        value_id: ValueId,
        name: String,
        graph_shape: Vec<usize>,
        owner_shape: Vec<usize>,
    },
    #[error(
        "exact dense binding v{value_id} owner shape {owner_shape:?} and view shape {view_shape:?} have different element counts"
    )]
    OwnerElementCountMismatch {
        value_id: ValueId,
        owner_shape: Vec<usize>,
        view_shape: Vec<usize>,
    },
    #[error("exact dense value cannot be consumed directly by {consumer}")]
    ClosedSurface { consumer: &'static str },
    #[error(
        "exact dense graph value v{value_id} dtype {dtype} shape {shape:?} storage {storage:?} cannot be consumed by {consumer}"
    )]
    GraphConsumer {
        value_id: ValueId,
        dtype: DType,
        shape: Vec<usize>,
        storage: Storage,
        consumer: String,
    },
}

fn dense_kind(kind: ExactSourceKind) -> Result<(DType, &'static str), ExactDenseError> {
    match kind {
        ExactSourceKind::Bf16 => Ok((DType::BF16, "BF16")),
        ExactSourceKind::F32 => Ok((DType::F32, "F32")),
        ExactSourceKind::E4m3 | ExactSourceKind::I64 => {
            Err(ExactDenseError::UnsupportedSourceKind { kind })
        }
    }
}

fn checked_source_bytes(dtype: DType, numel: usize) -> Result<usize, ExactDenseError> {
    numel
        .checked_mul(dtype.byte_size())
        .ok_or(ExactDenseError::ByteCountOverflow { dtype, numel })
}

/// BF16 word order: two little-endian bytes.
pub(crate) fn bf16_word(bytes: [u8; 2]) -> u16 {
    u16::from_le_bytes(bytes)
}

/// BF16 decode: the word is the top half of the f32 bits.
pub(crate) fn decode_bf16_word(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}
