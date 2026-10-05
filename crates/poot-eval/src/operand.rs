//! Reading one equation's operands out of the walk's `Vec<Option<Value>>` environment.
//!
//! Every helper here names the carrier it needs and fails closed (`EvalError::Input` or
//! `EvalError::UseBeforeDef`) when the bound value is a different one: no helper silently widens a
//! carrier it was not asked to read.

use poot_tensor::DType;
use std::borrow::Cow;
use std::sync::Arc;

use poot_graph_ir::{Eqn, Operand as IrOperand, Scalar, ValueId};

use crate::exact_bf16::ExactBf16TensorView;
use crate::exact_value::ExactValue;
use crate::{EvalError, Value};
use poot_tensor::{HostData, HostTensor};

/// The bound value of environment slot `id`, or [`EvalError::UseBeforeDef`].
pub(crate) fn env_value(env: &[Option<Value>], id: ValueId) -> Result<&Value, EvalError> {
    env.get(id)
        .and_then(Option::as_ref)
        .ok_or(EvalError::UseBeforeDef(id))
}

/// `eqn.inputs[position]` as a graph value id; a literal there is [`EvalError::Unsupported`] (every
/// op admitting a literal operand reads it through [`i32_words`]/[`f32_literal`] instead).
fn operand_value(eqn: &Eqn, position: usize) -> Result<ValueId, EvalError> {
    match eqn.inputs.get(position) {
        Some(IrOperand::Value(id)) => Ok(*id),
        _ => Err(EvalError::Unsupported {
            eqn: eqn.out,
            op: crate::resolve::op_label(&eqn.op),
            detail: format!("operand {position} is a literal, not a graph value"),
        }),
    }
}

/// The host tensor operand at `position`: the one carrier the plain dense ops read. Which dtype it
/// must be is each op's own check (a read of its F32 lane, its I32 words, ...), never a silent
/// conversion here.
pub(crate) fn host_tensor<'a>(
    eqn: &Eqn,
    position: usize,
    env: &'a [Option<Value>],
) -> Result<&'a HostTensor, EvalError> {
    let id = operand_value(eqn, position)?;
    match env_value(env, id)? {
        Value::Host(tensor) => Ok(tensor),
        other => Err(EvalError::Input {
            value: id,
            expected: poot_graph_ir::TensorType::new(Vec::new(), DType::F32),
            got: other.carrier_name(),
        }),
    }
}

/// The authoritative I32 words of operand `position`, from a literal, an I32 (or I8) host tensor, or
/// an exact-I32 view - the one definition every I32-consuming op shares (never an f32 value).
pub(crate) fn i32_words<'a>(
    eqn: &Eqn,
    position: usize,
    env: &'a [Option<Value>],
) -> Result<(Cow<'a, [i32]>, Vec<usize>), EvalError> {
    match eqn.inputs.get(position) {
        Some(IrOperand::Lit(Scalar::I32(value))) => Ok((Cow::Owned(vec![*value]), Vec::new())),
        Some(IrOperand::Lit(Scalar::F32(_))) | None => Err(EvalError::Unsupported {
            eqn: eqn.out,
            op: crate::resolve::op_label(&eqn.op),
            detail: format!("operand {position} has no I32 words"),
        }),
        Some(IrOperand::Value(id)) => value_i32_words(*id, env),
    }
}

pub(crate) fn value_i32_words(
    id: ValueId,
    env: &[Option<Value>],
) -> Result<(Cow<'_, [i32]>, Vec<usize>), EvalError> {
    match env_value(env, id)? {
        Value::Owner(ExactValue::I32(view)) => {
            Ok((Cow::Borrowed(view.i32_words()), view.shape().to_vec()))
        }
        Value::Host(tensor) => Ok((host_i32_words(tensor, id)?, tensor.shape().to_vec())),
        other => Err(EvalError::Input {
            value: id,
            expected: poot_graph_ir::TensorType::new(Vec::new(), DType::I32),
            got: other.carrier_name(),
        }),
    }
}

/// The exact I32 words of a host tensor: its own words for `I32`, the sign-extended bytes for `I8`
/// (a graph value of that dtype is held as bytes), a typed refusal for every other dtype.
pub(crate) fn host_i32_words(
    tensor: &HostTensor,
    id: ValueId,
) -> Result<Cow<'_, [i32]>, EvalError> {
    match (tensor.dtype(), tensor.data()) {
        (DType::I32, HostData::I32(words)) => Ok(Cow::Borrowed(words)),
        (DType::I8, HostData::Bytes(bytes)) => Ok(Cow::Owned(
            bytes.iter().map(|&byte| i32::from(byte as i8)).collect(),
        )),
        _ => Err(EvalError::Input {
            value: id,
            expected: poot_graph_ir::TensorType::new(tensor.shape().to_vec(), DType::I32),
            got: "Value::Host that is not an I32 or I8 tensor",
        }),
    }
}

/// The BF16 operand at `position`: a derived or owner-aliased exact view, or a BF16 host tensor's own
/// words (shared, never decoded through f32).
pub(crate) fn bf16_view(
    eqn: &Eqn,
    position: usize,
    env: &[Option<Value>],
) -> Result<ExactBf16TensorView, EvalError> {
    let id = operand_value(eqn, position)?;
    match env_value(env, id)? {
        Value::Owner(ExactValue::Bf16(view)) => Ok(view.clone()),
        Value::Owner(ExactValue::Dense(view)) => Ok(ExactBf16TensorView::from_owner_view(view)?),
        Value::Host(tensor) if tensor.dtype() == DType::BF16 => {
            let HostData::Half(words) = tensor.data() else {
                unreachable!("a BF16 HostTensor holds Half words")
            };
            Ok(ExactBf16TensorView::from_derived(
                Arc::clone(words),
                tensor.shape().to_vec(),
            )?)
        }
        other => Err(EvalError::Input {
            value: id,
            expected: poot_graph_ir::TensorType::new(Vec::new(), DType::BF16),
            got: other.carrier_name(),
        }),
    }
}

/// The E4M3FN operand at `position`: the raw-byte tensor, never decoded here.
pub(crate) fn e4m3_tensor<'a>(
    eqn: &Eqn,
    position: usize,
    env: &'a [Option<Value>],
) -> Result<&'a HostTensor, EvalError> {
    let id = operand_value(eqn, position)?;
    match env_value(env, id)? {
        Value::Host(tensor) if tensor.dtype() == DType::E4M3FN => Ok(tensor),
        other => Err(EvalError::Input {
            value: id,
            expected: poot_graph_ir::TensorType::new(Vec::new(), DType::E4M3FN),
            got: other.carrier_name(),
        }),
    }
}

/// The generic [`Value`] operand at `position`, for ops the storage-aware walk dispatches by the
/// bound carrier itself (`Reshape`, `Gather`, `Concat`, ...) rather than by a declared dtype.
pub(crate) fn value_operand<'a>(
    eqn: &Eqn,
    position: usize,
    env: &'a [Option<Value>],
) -> Result<&'a Value, EvalError> {
    env_value(env, operand_value(eqn, position)?)
}

impl Value {
    /// A short, stable name of this carrier for an [`EvalError::Input`] mismatch.
    pub(crate) fn carrier_name(&self) -> &'static str {
        match self {
            Value::Host(_) => "Value::Host",
            Value::Owner(ExactValue::Dense(_)) => "Value::Owner(ExactValue::Dense)",
            Value::Owner(ExactValue::Bf16(_)) => "Value::Owner(ExactValue::Bf16)",
            Value::Owner(ExactValue::I32(_)) => "Value::Owner(ExactValue::I32)",
            Value::Packed(_) => "Value::Packed",
        }
    }
}

/// Allocate a fresh, fallibly reserved `Arc<[T]>`, filled in place by `value_at`: the one allocation
/// site every walk-owned buffer uses, so a reservation failure never leaves a half-written allocation
/// and a filled allocation never pays for an extra `Vec` copy (554c).
pub(crate) fn initialized_arc_slice<T: Copy>(
    elements: usize,
    mut value_at: impl FnMut(usize) -> T,
) -> Arc<[T]> {
    let mut storage = Arc::<[T]>::new_uninit_slice(elements);
    let slots = Arc::get_mut(&mut storage).expect("a fresh Arc slice is uniquely owned");
    for (index, slot) in slots.iter_mut().enumerate() {
        slot.write(value_at(index));
    }
    // SAFETY: every slot of this fresh allocation is written exactly once in the loop above, and
    // `T: Copy` has no `Drop`, so an unwinding `value_at` cannot strand an initialized value.
    unsafe { storage.assume_init() }
}

/// Like [`initialized_arc_slice`], but `value_at` may fail (an I32 word op with no total semantics
/// for some input, e.g. `RemU` by zero). The slots filled before a failure need no cleanup: an
/// unwritten `MaybeUninit<T>` slot drops to nothing, so the partially-filled allocation is simply
/// dropped along with the early `Err`.
pub(crate) fn try_initialized_arc_slice<T: Copy, E>(
    elements: usize,
    mut value_at: impl FnMut(usize) -> Result<T, E>,
) -> Result<Arc<[T]>, E> {
    let mut storage = Arc::<[T]>::new_uninit_slice(elements);
    let slots = Arc::get_mut(&mut storage).expect("a fresh Arc slice is uniquely owned");
    for (index, slot) in slots.iter_mut().enumerate() {
        slot.write(value_at(index)?);
    }
    // SAFETY: this point is reached only after the loop above has written every slot; an early
    // failure returns before here via `?`.
    Ok(unsafe { storage.assume_init() })
}
