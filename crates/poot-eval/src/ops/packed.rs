//! Packed-quant ops: `PackedDequant` and `PackedRowGather`, and the shared carrier
//! binding contract (card 642) every packed equation and its executors' pre-upload check share.
//!
//! Oracle allocation limits were a fixed const (`PACKED_ORACLE_MAX_ELEMENTS`/`BYTES`); `walk.rs` now
//! charges every packed output against the caller's `EvalBudget` instead, so the limit is an option,
//! not a hardcoded wall (deval A5).

use poot_graph_ir::{Graph, OpKind, Operand, ValidationChannel, ValueId};
use poot_quant::{PackedComponentRef, PackedPayload, PackedStack, SourceRole};

use crate::value::PackedEvalError;
use poot_tensor::{HostData, HostTensor};

use crate::{EvalError, Value};

pub(crate) fn checked_packed_oracle_product(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, EvalError> {
    left.checked_mul(right)
        .ok_or(PackedEvalError::OracleArithmeticOverflow { field }.into())
}

/// Bind a `PackedDequant`/`PackedRowGather` equation's carrier operands by role and return the
/// shared payload owner every source resolves to.
///
/// Operand `i` binds `descriptor.sources()[i]`'s role (no positional slots): a block format has one
/// carrier, a registered E4M3/E2M1 cell has two, and GPTQ/AWQ have three or four. Every carrier must
/// share one `Arc<PackedPayload>` owner and one linear id, so this decodes only through that one
/// `poot-quant` payload - never a second, ad hoc decode.
pub(crate) fn packed_components<'a, V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &poot_graph_ir::Eqn,
    env: &'a [Option<Value>],
    descriptor: poot_quant::PackedWeight,
) -> Result<&'a std::sync::Arc<PackedPayload>, EvalError> {
    let owner = bind_packed_carriers(g, eqn, descriptor, |value_id| {
        env[value_id]
            .as_ref()
            .map(Some)
            .ok_or(EvalError::UseBeforeDef(value_id))
    })?;
    Ok(owner.expect("every carrier was bound, so there is a first component"))
}

/// The shared binding contract of [`packed_components`]. `bound`
/// resolves a carrier to its bound value, `None` when it is not supplied (skipped). Returns the
/// payload owner of the first supplied carrier, `None` when none was supplied.
fn bind_packed_carriers<'a, V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &poot_graph_ir::Eqn,
    descriptor: poot_quant::PackedWeight,
    bound: impl Fn(ValueId) -> Result<Option<&'a Value>, EvalError>,
) -> Result<Option<&'a std::sync::Arc<PackedPayload>>, EvalError> {
    let carrier_ids = match eqn.op {
        OpKind::PackedDequant { .. } => eqn.inputs.as_slice(),
        OpKind::PackedRowGather { .. } => &eqn.inputs[..eqn.inputs.len().saturating_sub(1)],
        _ => unreachable!("bind_packed_carriers called for another operation"),
    };
    let roles = descriptor.sources();
    if carrier_ids.len() != roles.len() {
        return Err(PackedEvalError::Binding { field: "operands" }.into());
    }
    let mut components: Vec<(&PackedComponentRef, ValueId, SourceRole)> =
        Vec::with_capacity(roles.len());
    for (operand, role) in carrier_ids.iter().zip(roles.iter().copied()) {
        let Operand::Value(value_id) = operand else {
            return Err(PackedEvalError::Binding { field: "operands" }.into());
        };
        if g.meta(*value_id).storage != poot_graph_ir::Storage::Const {
            return Err(PackedEvalError::Binding {
                field: "source.storage",
            }
            .into());
        }
        let Some(bound) = bound(*value_id)? else {
            continue;
        };
        let Value::Packed(component) = bound else {
            return Err(PackedEvalError::Binding {
                field: "source.storage",
            }
            .into());
        };
        if component.role() != role {
            return Err(PackedEvalError::Binding {
                field: "source.role",
            }
            .into());
        }
        if component.weight() != descriptor {
            return Err(PackedEvalError::Binding {
                field: "descriptor",
            }
            .into());
        }
        if component.bytes().len() != descriptor.source_bytes(role) {
            return Err(PackedEvalError::Binding {
                field: "source.byte_length",
            }
            .into());
        }
        components.push((component, *value_id, role));
    }
    let Some(&(owner_component, _, _)) = components.first() else {
        return Ok(None);
    };
    if components[1..]
        .iter()
        .any(|(component, _, _)| !component.same_owner(owner_component))
    {
        return Err(PackedEvalError::OwnerIdentity.into());
    }
    let source_name = |value_id: ValueId, role: SourceRole| {
        g.meta(value_id)
            .name
            .as_deref()
            .and_then(crate::PackedSourceName::parse)
            .filter(|name| name.role() == role)
    };
    let mut linear_id: Option<String> = None;
    for &(_, value_id, role) in &components {
        let name = source_name(value_id, role).ok_or(PackedEvalError::Binding { field: "name" })?;
        match &linear_id {
            None => linear_id = Some(name.linear_id().to_string()),
            Some(existing) if existing != name.linear_id() => {
                return Err(PackedEvalError::Binding { field: "name" }.into());
            }
            _ => {}
        }
    }
    Ok(Some(owner_component.owner()))
}

pub(crate) fn evaluate_packed_dequant(
    owner: &PackedPayload,
    descriptor: poot_quant::PackedWeight,
) -> Result<HostTensor, EvalError> {
    let [out, k] = descriptor.shape();
    let logical_values = checked_packed_oracle_product(out, k, "logical_values")?;
    let mut values = vec![0.0; logical_values];
    if k > 0 {
        for (row, decoded) in values.chunks_exact_mut(k).enumerate() {
            decode_packed_row(owner, row, decoded)?;
        }
    }
    Ok(HostTensor::f32(vec![out, k], values))
}

/// One logical weight row through the payload's row decoder (`O(K)`, dquant.md R5).
fn decode_packed_row(owner: &PackedPayload, row: usize, out: &mut [f32]) -> Result<(), EvalError> {
    owner.decode_row(row, out).map_err(|error| {
        PackedEvalError::Decode {
            row,
            detail: error.to_string(),
        }
        .into()
    })
}

/// The row ids operand of a `PackedRowGather` (its last input), dense F32 or I32.
pub(crate) fn packed_row_ids<'a>(
    eqn: &poot_graph_ir::Eqn,
    env: &'a [Option<Value>],
) -> Result<&'a HostTensor, EvalError> {
    let Some(Operand::Value(ids)) = eqn.inputs.last() else {
        return Err(PackedEvalError::Binding { field: "operands" }.into());
    };
    match env[*ids].as_ref().ok_or(EvalError::UseBeforeDef(*ids))? {
        Value::Host(ids) => Ok(ids),
        _ => Err(PackedEvalError::Binding {
            field: "row_ids.storage",
        }
        .into()),
    }
}

/// `out[r, :] = decode(ids[r], :)`: a packed weight's rows, each through the payload's row decoder
/// (card 545a). Not capped: it decodes only the rows it gathers.
pub(crate) fn evaluate_packed_row_gather(
    ids: &HostTensor,
    owner: &PackedPayload,
    descriptor: poot_quant::PackedWeight,
) -> Result<HostTensor, EvalError> {
    let [out, k] = descriptor.shape();
    // An f32 id must be a finite whole number (a NaN or fractional id names no row; it is refused
    // like an out-of-range one, never rounded to a neighbour or to row 0).
    let range = || -> EvalError {
        PackedEvalError::Binding {
            field: "row_ids.range",
        }
        .into()
    };
    let id_values: Vec<i64> = match ids.data() {
        HostData::I32(ints) => ints.iter().map(|&id| i64::from(id)).collect(),
        HostData::F32(data) => data
            .iter()
            .map(|&id| {
                (id.is_finite() && id.fract() == 0.0)
                    .then_some(id as i64)
                    .ok_or_else(range)
            })
            .collect::<Result<_, _>>()?,
        _ => return Err(range()),
    };
    let rows: Vec<usize> = id_values
        .into_iter()
        .map(|id| {
            usize::try_from(id)
                .ok()
                .filter(|&row| row < out)
                .ok_or_else(range)
        })
        .collect::<Result<_, _>>()?;
    let elements = checked_packed_oracle_product(rows.len(), k, "rows_k")?;
    let mut values = vec![0.0; elements];
    if k > 0 {
        for (row, decoded) in rows.iter().zip(values.chunks_exact_mut(k)) {
            decode_packed_row(owner, *row, decoded)?;
        }
    }
    let mut shape = ids.shape().to_vec();
    shape.push(k);
    Ok(HostTensor::f32(shape, values))
}

/// Bind an indexed `PackedContraction`'s stacked carriers (operands `2..`) by role and return the one
/// [`PackedStack`] every source resolves to. The stacked counterpart of [`packed_components`]: each
/// carrier must be a [`Value::PackedStack`] of its role, of `descriptor` and `experts` experts, over one
/// shared stack, and named after one stack id.
pub(crate) fn packed_stack_components<'a, V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &poot_graph_ir::Eqn,
    env: &'a [Option<Value>],
    descriptor: poot_quant::PackedWeight,
    experts: usize,
) -> Result<&'a std::sync::Arc<PackedStack>, EvalError> {
    let roles = descriptor.sources();
    let carriers = eqn.inputs.get(2..).unwrap_or_default();
    if carriers.len() != roles.len() {
        return Err(PackedEvalError::Binding { field: "operands" }.into());
    }
    let mut first: Option<&poot_quant::PackedStackComponentRef> = None;
    let mut linear_id: Option<String> = None;
    for (operand, role) in carriers.iter().zip(roles.iter().copied()) {
        let Operand::Value(value_id) = operand else {
            return Err(PackedEvalError::Binding { field: "operands" }.into());
        };
        if g.meta(*value_id).storage != poot_graph_ir::Storage::Const {
            return Err(PackedEvalError::Binding {
                field: "source.storage",
            }
            .into());
        }
        let Some(Value::PackedStack(component)) = env[*value_id].as_ref() else {
            return Err(PackedEvalError::Binding {
                field: "source.storage",
            }
            .into());
        };
        if component.role() != role {
            return Err(PackedEvalError::Binding {
                field: "source.role",
            }
            .into());
        }
        if component.weight() != descriptor {
            return Err(PackedEvalError::Binding {
                field: "descriptor",
            }
            .into());
        }
        if component.stack().experts() != experts {
            return Err(PackedEvalError::Binding { field: "experts" }.into());
        }
        if first.is_some_and(|first| !first.same_owner(component)) {
            return Err(PackedEvalError::OwnerIdentity.into());
        }
        first.get_or_insert(component);
        let name = g
            .meta(*value_id)
            .name
            .as_deref()
            .and_then(crate::PackedSourceName::parse)
            .filter(|name| name.role() == role)
            .ok_or(PackedEvalError::Binding { field: "name" })?;
        match &linear_id {
            None => linear_id = Some(name.linear_id().to_string()),
            Some(existing) if existing != name.linear_id() => {
                return Err(PackedEvalError::Binding { field: "name" }.into());
            }
            _ => {}
        }
    }
    first
        .map(|component| component.stack())
        .ok_or(PackedEvalError::Binding { field: "operands" }.into())
}

/// `out[m, n] = sum over k of x[m, k] * decode(expert ids[m], n, k)`: the indexed packed contraction, evaluated
/// by decoding only the experts the ids select. Rows are grouped by expert; each distinct selected expert is
/// decoded once (`charge` is called per decode, so the caller's budget and observer see exactly the experts
/// read, never the whole table), and the unselected experts' bytes are never touched. Accumulation is the
/// sequential `k` sum of the dense `IndexedMatMul` oracle over the same decoded weight, bit for bit.
pub(crate) fn evaluate_packed_indexed(
    x: &HostTensor,
    ids: &HostTensor,
    stack: &PackedStack,
    out_shape: &[usize],
    eqn: ValueId,
    mut charge: impl FnMut() -> Result<(), EvalError>,
) -> Result<HostTensor, EvalError> {
    let [out, k] = stack.weight().shape();
    let m = x.shape()[0];
    let x_view = x.to_f32()?;
    let selected: Vec<usize> = super::movement::index_values(ids)?
        .into_iter()
        .enumerate()
        .map(|(position, value)| super::index_rule::index_at(value, position, stack.experts(), eqn))
        .collect::<Result<_, _>>()?;
    let mut by_expert: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (row, &expert) in selected.iter().enumerate() {
        by_expert.entry(expert).or_default().push(row);
    }
    let mut data = vec![0.0f32; checked_packed_oracle_product(m, out, "rows_out")?];
    let mut weight = vec![0.0f32; checked_packed_oracle_product(out, k, "logical_values")?];
    for (expert, rows) in by_expert {
        charge()?;
        for (row, decoded) in weight.chunks_exact_mut(k).enumerate() {
            stack.decode_row(expert, row, decoded).map_err(|error| {
                EvalError::from(PackedEvalError::Decode {
                    row,
                    detail: error.to_string(),
                })
            })?;
        }
        for mm in rows {
            for nn in 0..out {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += x_view[mm * k + kk] * weight[nn * k + kk];
                }
                data[mm * out + nn] = acc;
            }
        }
    }
    Ok(HostTensor::f32(out_shape.to_vec(), data))
}
