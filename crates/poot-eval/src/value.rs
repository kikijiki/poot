//! Typed executor values: the one carrier the walk binds, evaluates and publishes.

use poot_tensor::{DType, HostTensor};

use crate::{
    Cow, EvalError, ExactI32TensorView, ExactValue, PackedComponentRef, PackedStackComponentRef,
    fp8,
};

/// The executor value: a typed host tensor, an exact word-carrier value, or a packed component.
///
/// Every executor entry and the CPU oracle bind `HashMap<ValueId, Value>` built directly; the
/// `Tensor`-map adapter `widen_map` is gone (card 545a) and must not come back:
///
/// ```compile_fail,E0432
/// use poot_eval::widen_map;
/// ```
///
/// (Red-proven: restoring a `pub fn widen_map` makes this doctest compile, which fails it.)
///
/// The eight former entry points (`eval_value`, `eval_value_with_state`,
/// `eval_value_with_state_observed`, `eval_exact_i32`, `eval_exact_i32_with_state`, and the dense-only
/// `eval`/`eval_with_state`/`eval_all`) are gone too, replaced by the one walk:
///
/// ```compile_fail,E0432
/// use poot_eval::eval_value;
/// ```
///
/// ```compile_fail,E0432
/// use poot_eval::eval_with_state;
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A typed host tensor: its dtype says how to read the payload (BF16 words, I32 words, E4M3FN
    /// bytes, f32 values), and no tensor carries a second, widened copy of itself.
    Host(HostTensor),
    /// An exact word-carrier value (derived BF16 words, exact I32 words).
    Exact(ExactValue),
    Packed(PackedComponentRef),
    /// One source role of an expert-stacked packed table (an indexed packed contraction's carrier).
    PackedStack(PackedStackComponentRef),
}

impl Value {
    pub fn shape(&self) -> Cow<'_, [usize]> {
        match self {
            Value::Host(t) => Cow::Borrowed(t.shape()),
            Value::Exact(value) => Cow::Borrowed(value.shape()),
            Value::Packed(component) => {
                Cow::Owned(component.weight().source_shape(component.role()).to_vec())
            }
            Value::PackedStack(component) => Cow::Owned(vec![
                component.stack().experts(),
                component.weight().slab_bytes(component.role()),
            ]),
        }
    }

    /// The host tensor this value is, or `None` for a packed component or an exact
    /// value: the typed accessor a host-side reader of a weight uses, so it can never reinterpret
    /// packed source bytes as numbers.
    #[cfg(any(test, feature = "test-support"))]
    pub fn as_host(&self) -> Option<&HostTensor> {
        match self {
            Value::Host(tensor) => Some(tensor),
            Value::Exact(_) | Value::Packed(_) | Value::PackedStack(_) => None,
        }
    }

    /// The host tensor this value is, typed `EvalError::NotHost` for any other carrier: the one
    /// accessor a caller of [`crate::eval`] uses once it knows its graph publishes a host output.
    pub fn into_host(self) -> Result<HostTensor, EvalError> {
        match self {
            Value::Host(tensor) => Ok(tensor),
            Value::Exact(_) => Err(EvalError::NotHost {
                what: "exact value",
            }),
            Value::Packed(_) => Err(EvalError::NotHost {
                what: "packed component",
            }),
            Value::PackedStack(_) => Err(EvalError::NotHost {
                what: "packed stack component",
            }),
        }
    }

    pub fn numel(&self) -> usize {
        match self {
            Value::Exact(value) => value.numel(),
            _ => self.shape().iter().product(),
        }
    }

    /// Count bytes in the value's current physical host/executor lane. For E4M3FN this is the normative
    /// packed-row device allocation, including tail padding. It deliberately differs from `numel()`.
    pub fn physical_bytes(&self) -> usize {
        match self {
            Value::Host(t) if t.dtype() == DType::E4M3FN => {
                fp8::PortableE4m3FnLayout::for_shape(t.shape()).device_bytes
            }
            Value::Host(t) => t.data().byte_len(),
            Value::Exact(value) => value.physical_bytes(),
            Value::Packed(component) => component.bytes().len(),
            Value::PackedStack(component) => {
                component.stack().experts() * component.weight().slab_bytes(component.role())
            }
        }
    }
}

impl From<HostTensor> for Value {
    fn from(tensor: HostTensor) -> Self {
        Value::Host(tensor)
    }
}

impl From<ExactValue> for Value {
    fn from(value: ExactValue) -> Self {
        Value::Exact(value)
    }
}

impl From<ExactI32TensorView> for Value {
    fn from(value: ExactI32TensorView) -> Self {
        Value::Exact(ExactValue::I32(value))
    }
}

impl From<PackedStackComponentRef> for Value {
    fn from(value: PackedStackComponentRef) -> Self {
        Value::PackedStack(value)
    }
}

impl From<PackedComponentRef> for Value {
    fn from(value: PackedComponentRef) -> Self {
        Value::Packed(value)
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum PackedEvalError {
    #[error("packed component cannot be consumed by {operation}")]
    WrongConsumer { operation: &'static str },
    #[error("packed binding {field} mismatch")]
    Binding { field: &'static str },
    #[error("packed weight and scale bindings do not share one Arc<PackedPayload>")]
    OwnerIdentity,
    #[error("packed row decode failed at row {row}: {detail}")]
    Decode { row: usize, detail: String },
    #[error("packed CPU oracle arithmetic overflowed {field}")]
    OracleArithmeticOverflow { field: &'static str },
    #[error("packed CPU oracle {resource} budget exceeded: {requested} > {limit}")]
    OracleLimit {
        resource: &'static str,
        requested: usize,
        limit: usize,
    },
}
