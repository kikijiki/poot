//! The one evaluator walk: bind, preflight every equation's admission, evaluate each in
//! order, gate on the validation packet, publish. Replaces the five walks this crate used to carry
//! (`eval_env_unchecked`, the storage walk, the one-equation bridge, the exact-I32 walk, the BF16
//! table): one function, one environment (`Value`), admission decided only by an equation's own
//! operands, budgets and observers as options (not separate walks).

use poot_tensor::{DType, HostData, HostTensor};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use poot_graph_ir::decompose::{Decomposition, decompose};
use poot_graph_ir::{
    Eqn, Graph, OpClass, OpKind, Operand as IrOperand, TensorType, ValidationChannel, ValueId,
};

use crate::cast_authority::{CastAuthority, ExactI32CastRole, validate_cast_source};
use crate::exact_bf16::{self, ExactBf16TensorView};
use crate::exact_dense;
use crate::exact_value::ExactValue;
use crate::observer::EvalObserver;
use crate::operand::{e4m3_tensor, env_value, host_tensor, i32_words, value_i32_words};
use crate::ops::elementwise::I32Operand;
use crate::resolve::Bf16Equation;
use crate::{EvalError, Value, ops, resolve};

/// A caller-stated ceiling on the packed/cast/Spec-376-BF16 arms' materialization (see
/// [`EvalObserver::allocation`]'s doc for exactly which equations charge against it) and on every intermediate of a composite's decomposition (Card 556: a decomposed prefill
/// attention materializes `[1, Hq, L, L]` scores behind a `[1, Hq, L, D]` result) - not a per-equation
/// ceiling over the whole walk. `EvalOptions::new` takes one explicitly: there is no `Default`, so no
/// caller evaluates unbounded by omission.
#[derive(Clone, Copy, Debug)]
pub struct EvalBudget {
    max_work_elements: Option<u64>,
    max_alloc_bytes: Option<u64>,
}

impl EvalBudget {
    /// No ceiling: every allocation this walk would otherwise charge against a budget proceeds.
    pub const UNBOUNDED: Self = EvalBudget {
        max_work_elements: None,
        max_alloc_bytes: None,
    };

    pub const fn bounded(max_work_elements: u64, max_alloc_bytes: u64) -> Self {
        EvalBudget {
            max_work_elements: Some(max_work_elements),
            max_alloc_bytes: Some(max_alloc_bytes),
        }
    }
}

/// Options for one [`eval`] call. The builder methods take and return `self`, so no field is public
/// and no caller can skip stating a budget.
pub struct EvalOptions<'a> {
    budget: EvalBudget,
    observer: Option<&'a mut dyn EvalObserver>,
    keep_environment: bool,
    cast_authority: Option<&'a mut dyn CastAuthority>,
    /// The composite equation whose decomposition is being evaluated, if any: every charge made inside
    /// it is attributed to that equation, the one the caller's graph names.
    composite: Option<ValueId>,
}

impl<'a> EvalOptions<'a> {
    pub fn new(budget: EvalBudget) -> Self {
        EvalOptions {
            budget,
            observer: None,
            keep_environment: false,
            cast_authority: None,
            composite: None,
        }
    }

    pub fn observer(mut self, observer: &'a mut dyn EvalObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Keep the full per-value environment on [`Evaluation::environment`] (replaces `eval_all`).
    pub fn keep_environment(mut self) -> Self {
        self.keep_environment = true;
        self
    }

    /// Authorize exact-I32 `Cast(I32 -> F32)` equations; deleted with this file by Card
    /// 558b. Without it, every such cast still passes X3's range check; it just never gets a role, so
    /// `CastAuthorityError::MissingCastAuthorization` fires the moment the graph has one.
    pub fn cast_authority(mut self, authority: &'a mut dyn CastAuthority) -> Self {
        self.cast_authority = Some(authority);
        self
    }

    fn charge(
        &mut self,
        eqn: ValueId,
        idx: usize,
        dtype: DType,
        elements: usize,
        bytes: usize,
    ) -> Result<(), EvalError> {
        let eqn = self.composite.unwrap_or(eqn);
        if let Some(limit) = self.budget.max_work_elements
            && elements as u64 > limit
        {
            return Err(EvalError::Budget {
                eqn,
                resource: "elements",
                needed: elements,
                limit: limit as usize,
            });
        }
        if let Some(limit) = self.budget.max_alloc_bytes
            && bytes as u64 > limit
        {
            return Err(EvalError::Budget {
                eqn,
                resource: "bytes",
                needed: bytes,
                limit: limit as usize,
            });
        }
        if let Some(observer) = self.observer.as_deref_mut() {
            observer.allocation(idx, eqn, dtype, bytes);
        }
        Ok(())
    }
}

/// The result of one [`eval`] call.
#[derive(Debug, Clone)]
pub struct Evaluation {
    pub output: Value,
    /// State outputs in [`Graph::state`] order.
    pub state: Vec<Value>,
    /// The full per-value environment, present only when [`EvalOptions::keep_environment`] was set.
    pub environment: Option<Vec<Option<Value>>>,
}

/// Evaluate `g`. `inputs` must bind every graph input (consts and slots; a computed constant, such as
/// a folded `Iota`, materializes from its own definition and needs no entry).
///
/// Bind, preflight every input binding ([`preflight_bindings`]), evaluate each equation in
/// declaration order, gate the result against the graph's validation packet, publish. No separate
/// walk for I32, BF16 or E4M3FN: every equation's evaluation depends only on its own operands,
/// never on an unrelated value elsewhere in the graph. Card 555: there
/// is no separate per-equation admission preflight ahead of this loop any more - `infer` already
/// restricts every op's admitted dtype combination to exactly what the match in
/// [`evaluate_equation`] implements for it, so an unadmitted shape fails there, the first time (and
/// only time) that equation is evaluated, not in an earlier, now-vacuous pass.
pub fn eval<V: ValidationChannel>(
    g: &Graph<V>,
    inputs: &HashMap<ValueId, Value>,
    mut opts: EvalOptions<'_>,
) -> Result<Evaluation, EvalError> {
    g.validate()?;
    preflight_bindings(g, inputs)?;

    let mut env: Vec<Option<Value>> = vec![None; g.values.len()];
    bind_inputs(g, inputs, &mut env)?;

    let producers = producers(g);
    let mut decompositions = Decompositions::default();
    for (idx, eqn) in g.eqns.iter().enumerate() {
        let value = evaluate_equation(
            g,
            eqn,
            &env,
            &producers,
            idx,
            &mut opts,
            &mut decompositions,
        )?;
        let value = store_as_declared(value, g.aval(eqn.out).dtype)?;
        check_published(g, eqn, &value)?;
        if let Some(observer) = opts.observer.as_deref_mut() {
            observer.value(idx, eqn.out, &value);
        }
        env[eqn.out] = Some(value);
    }

    validate_environment(g, &env)?;

    let output = publish_value(&env, g.output)?;
    let state = g
        .state
        .iter()
        .map(|&(state_in, state_out)| {
            let value = publish_value(&env, state_out)?;
            check_state_storage(g, state_in, state_out, &value)?;
            Ok(value)
        })
        .collect::<Result<Vec<_>, EvalError>>()?;
    let environment = opts.keep_environment.then(|| {
        #[cfg(test)]
        publication_probe::record();
        env.clone()
    });

    Ok(Evaluation {
        output,
        state,
        environment,
    })
}

/// The index of the equation that produces each value, if any.
fn producers<V: ValidationChannel>(g: &Graph<V>) -> Vec<Option<usize>> {
    let mut producers: Vec<Option<usize>> = vec![None; g.values.len()];
    for (index, eqn) in g.eqns.iter().enumerate() {
        producers[eqn.out] = Some(index);
    }
    producers
}

/// Copy one value out of the completed environment for publication: the one place every output and
/// state value leaves the walk, always after [`validate_environment`] has gated it.
fn publish_value(env: &[Option<Value>], value: ValueId) -> Result<Value, EvalError> {
    #[cfg(test)]
    publication_probe::record();
    env.get(value)
        .and_then(Option::as_ref)
        .cloned()
        .ok_or(EvalError::UseBeforeDef(value))
}

/// The validation-packet gate (ADR-0090/ADR-0101): every declared validation root must be a complete
/// dense F32 tensor of its declared shape; an empty validation set is a no-op.
fn validate_environment<V: ValidationChannel>(
    g: &Graph<V>,
    env: &[Option<Value>],
) -> Result<(), EvalError> {
    let validations = g.validation_outputs();
    if validations.is_empty() {
        return Ok(());
    }
    let layout = poot_graph_ir::ValidationPacketLayout::for_graph(g)?;
    let mut bits = Vec::with_capacity(layout.lane_count);
    for validation in validations {
        let not_dense = || EvalError::ValidationValueNotHost {
            value: validation.value,
        };
        let tensor = match env.get(validation.value).and_then(Option::as_ref) {
            Some(Value::Host(tensor)) => tensor,
            _ => return Err(not_dense()),
        };
        let data = tensor.as_f32().ok_or_else(not_dense)?;
        let expected = &g.aval(validation.value).shape;
        if tensor.shape() != expected.as_slice() {
            return Err(EvalError::ValidationShape {
                value: validation.value,
                expected: expected.clone(),
                actual: tensor.shape().to_vec(),
            });
        }
        bits.extend(data.iter().map(|value| value.to_bits()));
    }
    layout
        .validate_f32_bits(&bits)
        .map_err(|error| match error {
            poot_graph_ir::ValidationPacketError::Failure(failure) => {
                EvalError::Validation(failure)
            }
            poot_graph_ir::ValidationPacketError::Length { expected, actual } => {
                EvalError::ValidationPacketLength { expected, actual }
            }
        })
}

fn check_state_storage<V: ValidationChannel>(
    g: &Graph<V>,
    state_in: ValueId,
    state_out: ValueId,
    value: &Value,
) -> Result<(), EvalError> {
    let expected = g.aval(state_in);
    let carrier_matches = match value {
        Value::Host(tensor) => tensor.dtype() == expected.dtype,
        Value::Owner(ExactValue::Bf16(_)) => expected.dtype == DType::BF16,
        Value::Owner(_) | Value::Packed(_) => false,
    };
    if !carrier_matches || value.shape().as_ref() != expected.shape.as_slice() {
        return Err(EvalError::Input {
            value: state_out,
            expected: expected.clone(),
            got: value.carrier_name(),
        });
    }
    Ok(())
}

/// Store a `Value::Host` F32 result as the equation's declared dtype: every dense op computes in f32,
/// so a BF16/F16-declared result is narrowed once, here, as the GPU kernels do on store. Any other
/// value passes through unchanged.
fn store_as_declared(value: Value, declared: DType) -> Result<Value, EvalError> {
    match value {
        Value::Host(tensor) => Ok(Value::Host(ops::cast::store_as(tensor, declared)?)),
        other => Ok(other),
    }
}

/// A coarse published-value check against the equation's declared `out` aval: shape, and a dtype
/// consistent with the carrier. A host tensor must be of the declared dtype (its payload then holds
/// the declared element count by construction); `debug_assert`s the stronger claim, since a release
/// build should fail closed on a typed error rather than publish a mismatched carrier, but never
/// panic on a build the caller ships.
fn check_published<V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &Eqn,
    value: &Value,
) -> Result<(), EvalError> {
    let expected = g.aval(eqn.out);
    let shape_ok = value.shape().as_ref() == expected.shape.as_slice();
    let class_ok = match value {
        Value::Host(tensor) => tensor.dtype() == expected.dtype,
        Value::Owner(ExactValue::Bf16(_)) => expected.dtype == DType::BF16,
        Value::Owner(ExactValue::I32(_)) => expected.dtype == DType::I32,
        Value::Owner(ExactValue::Dense(_)) => {
            matches!(expected.dtype, DType::BF16 | DType::F32)
        }
        Value::Packed(_) => false,
    };
    debug_assert!(
        shape_ok && class_ok,
        "eqn v{} ({}) published {:?} for declared {expected}",
        eqn.out,
        resolve::op_label(&eqn.op),
        value.shape()
    );
    if shape_ok && class_ok {
        Ok(())
    } else {
        Err(EvalError::Unsupported {
            eqn: eqn.out,
            op: resolve::op_label(&eqn.op),
            detail: format!("published a carrier inconsistent with declared {expected}"),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Bind
// ---------------------------------------------------------------------------------------------

/// Fail-closed checks every bound input must pass before any equation runs: a host tensor must be of
/// exactly the declared dtype and shape (a payload cannot disagree with its dtype, so no narrowing or
/// widening happens at bind), and a non-output owner-backed exact-dense view must be consumed by an
/// equation that actually reads it.
fn preflight_bindings<V: ValidationChannel>(
    g: &Graph<V>,
    inputs: &HashMap<ValueId, Value>,
) -> Result<(), EvalError> {
    for &id in &g.inputs {
        let aval = g.aval(id);
        if let poot_graph_ir::Storage::Computed(_) = g.meta(id).storage {
            continue;
        }
        let value = inputs.get(&id).ok_or(EvalError::MissingInput(id))?;
        if value.shape().as_ref() != aval.shape.as_slice() {
            return Err(EvalError::Input {
                value: id,
                expected: aval.clone(),
                got: value.carrier_name(),
            });
        }
        match (aval.dtype, value) {
            (dtype, Value::Host(tensor)) if tensor.dtype() == dtype => {}
            (_, Value::Host(_)) => {
                return Err(EvalError::Input {
                    value: id,
                    expected: aval.clone(),
                    got: "Value::Host of a different dtype",
                });
            }
            (DType::F32 | DType::BF16 | DType::F16, Value::Owner(ExactValue::Dense(view)))
                if view.dtype() == aval.dtype => {}
            (DType::BF16, Value::Owner(ExactValue::Bf16(_))) => {}
            (DType::I32, Value::Owner(ExactValue::I32(_))) => {}
            (DType::I8, Value::Packed(_)) => {}
            (_, other) => {
                return Err(EvalError::Input {
                    value: id,
                    expected: aval.clone(),
                    got: other.carrier_name(),
                });
            }
        }
    }
    // Card 396: a BF16/F32 owner view may feed only a Spec 376 BF16 equation (via `classify`), else it
    // is a refused `GraphConsumer`; every owner view not consumed by one, and not the graph's output,
    // is an unconsumed non-output binding.
    exact_dense::validate_exact_dense_bindings(g, inputs)
}

/// Bind every graph input into `env`, exactly as bound: [`preflight_bindings`] has already checked
/// each host tensor's dtype and shape against the graph's declaration.
fn bind_inputs<V: ValidationChannel>(
    g: &Graph<V>,
    inputs: &HashMap<ValueId, Value>,
    env: &mut [Option<Value>],
) -> Result<(), EvalError> {
    for &id in &g.inputs {
        if let poot_graph_ir::Storage::Computed(computed) = g.meta(id).storage {
            env[id] = Some(Value::Host(HostTensor::f32(
                computed.shape(),
                computed.values_f32(),
            )));
            continue;
        }
        env[id] = Some(inputs.get(&id).ok_or(EvalError::MissingInput(id))?.clone());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Evaluate: one equation
// ---------------------------------------------------------------------------------------------

fn touches_e4m3<V: ValidationChannel>(g: &Graph<V>, eqn: &Eqn) -> bool {
    g.aval(eqn.out).dtype == DType::E4M3FN
        || eqn.inputs.iter().any(|operand| match operand {
            IrOperand::Value(id) => g.aval(*id).dtype == DType::E4M3FN,
            IrOperand::Lit(_) => false,
        })
}

fn evaluate_equation<V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &Eqn,
    env: &[Option<Value>],
    producers: &[Option<usize>],
    idx: usize,
    opts: &mut EvalOptions<'_>,
    decompositions: &mut Decompositions,
) -> Result<Value, EvalError> {
    if eqn.op.class() == OpClass::Composite {
        return evaluate_composite(g, eqn, env, idx, opts, decompositions);
    }
    match eqn.op {
        OpKind::PackedDequant { descriptor } => {
            let owner = ops::packed_components(g, eqn, env, descriptor)?;
            let [out, k] = descriptor.shape();
            opts.charge(eqn.out, idx, DType::F32, out * k, out * k * 4)?;
            Ok(Value::Host(ops::evaluate_packed_dequant(
                owner, descriptor,
            )?))
        }
        OpKind::PackedContraction { descriptor, blocks } => {
            let [IrOperand::Value(activation_id), ..] = eqn.inputs.as_slice() else {
                return Err(EvalError::Unsupported {
                    eqn: eqn.out,
                    op: "packed_contraction",
                    detail: "malformed PackedContraction operands".into(),
                });
            };
            let activation = host_tensor(eqn, 0, env)?;
            let _ = activation_id;
            let owner = ops::packed_components(g, eqn, env, descriptor)?;
            let elements: usize = g.aval(eqn.out).shape.iter().product();
            opts.charge(eqn.out, idx, DType::F32, elements, elements * 4)?;
            Ok(Value::Host(ops::evaluate_packed_contraction(
                activation, owner, descriptor, blocks,
            )?))
        }
        OpKind::PackedRowGather { descriptor } => {
            let ids = ops::packed_row_ids(eqn, env)?;
            let owner = ops::packed_components(g, eqn, env, descriptor)?;
            let elements: usize = g.aval(eqn.out).shape.iter().product();
            opts.charge(eqn.out, idx, DType::F32, elements, elements * 4)?;
            Ok(Value::Host(ops::evaluate_packed_row_gather(
                ids, owner, descriptor,
            )?))
        }
        OpKind::Cast { to } => evaluate_cast(g, eqn, to, env, producers, idx, opts),
        OpKind::Gather { axis } => evaluate_gather(g, eqn, axis, env, idx, opts),
        _ => {
            if let Some(equation) = resolve::classify(g, eqn) {
                return evaluate_bf16(g, eqn, equation, env, idx, opts);
            }
            if touches_e4m3(g, eqn) {
                return evaluate_e4m3(g, eqn, env);
            }
            evaluate_dense(g, eqn, env, idx, opts)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Composite: evaluated through its decomposition (ADR-0101 tier 1, Card 556)
// ---------------------------------------------------------------------------------------------

/// The decompositions one [`eval`] call has built, by operand types and then op: a qwen2 decode graph
/// holds hundreds of `Fused` equations, most of them one of a few regions over the same types, so each
/// distinct (op, operand types) is decomposed once per call.
#[derive(Default)]
struct Decompositions {
    by_operand_types: HashMap<Vec<TensorType>, Vec<(OpKind, Rc<Decomposition>)>>,
}

impl Decompositions {
    fn get(&mut self, op: &OpKind, operands: Vec<TensorType>) -> Option<Rc<Decomposition>> {
        if let Some(built) = self.by_operand_types.get(&operands)
            && let Some((_, decomposition)) = built.iter().find(|(built_op, _)| built_op == op)
        {
            return Some(Rc::clone(decomposition));
        }
        let decomposition = Rc::new(decompose(op, &operands)?);
        self.by_operand_types
            .entry(operands)
            .or_default()
            .push((op.clone(), Rc::clone(&decomposition)));
        Some(decomposition)
    }
}

/// Evaluate a composite equation from its decomposition with this same walk: bind the equation's
/// operand values (as they are, so each primitive reads exactly what the replaced chain read) to the
/// decomposition's operand placeholders, evaluate every primitive equation in order, and return the
/// decomposition's output. Each intermediate is charged to the budget, attributed to the composite
/// equation; the observer sees only the composite's own value, which the caller publishes. An error a
/// primitive raises is attributed to the composite equation too ([`attributed_to`]).
fn evaluate_composite<V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &Eqn,
    env: &[Option<Value>],
    idx: usize,
    opts: &mut EvalOptions<'_>,
    decompositions: &mut Decompositions,
) -> Result<Value, EvalError> {
    let operand_types = eqn
        .inputs
        .iter()
        .map(|operand| match operand {
            IrOperand::Value(id) => g.aval(*id).clone(),
            IrOperand::Lit(scalar) => scalar.ty(),
        })
        .collect();
    let decomposition =
        decompositions
            .get(&eqn.op, operand_types)
            .ok_or_else(|| EvalError::Unsupported {
                eqn: eqn.out,
                op: resolve::op_label(&eqn.op),
                detail: "the composite has no decomposition for its operand types".into(),
            })?;
    let inner = &decomposition.graph;
    let mut inner_env: Vec<Option<Value>> = vec![None; inner.values.len()];
    for (&placeholder, operand) in decomposition.operands.iter().zip(&eqn.inputs) {
        inner_env[placeholder] = Some(match operand {
            IrOperand::Value(id) => env_value(env, *id)?.clone(),
            IrOperand::Lit(poot_graph_ir::Scalar::F32(v)) => Value::Host(HostTensor::scalar(*v)),
            IrOperand::Lit(poot_graph_ir::Scalar::I32(v)) => {
                Value::Host(HostTensor::i32(Vec::new(), vec![*v]))
            }
        });
    }
    let inner_producers = producers(inner);
    let outermost = opts.composite.is_none();
    if outermost {
        opts.composite = Some(eqn.out);
    }
    let result = (|| {
        for inner_eqn in &inner.eqns {
            let declared = inner.aval(inner_eqn.out);
            // Every dense equation computes into a 4-byte-per-element F32 or I32 buffer before the
            // declared-dtype store.
            let elements = ops::element_count(&declared.shape)?;
            opts.charge(
                inner_eqn.out,
                idx,
                declared.dtype,
                elements,
                elements.saturating_mul(4),
            )?;
            let value = evaluate_equation(
                inner,
                inner_eqn,
                &inner_env,
                &inner_producers,
                idx,
                opts,
                decompositions,
            )?;
            let value = store_as_declared(value, declared.dtype)?;
            check_published(inner, inner_eqn, &value)?;
            inner_env[inner_eqn.out] = Some(value);
        }
        inner_env[inner.output]
            .take()
            .ok_or(EvalError::UseBeforeDef(inner.output))
    })();
    if outermost {
        opts.composite = None;
    }
    result.map_err(|error| attributed_to(eqn, error))
}

/// `error`, raised inside `composite`'s decomposition, with every equation or value id it carries
/// replaced by the composite equation's: an id from the decomposition's own graph names nothing in the
/// caller's graph. An `Unsupported` keeps the primitive's label in its detail. The remaining variants
/// carry no id a decomposition can raise (decompositions hold no `Cast` the cast authority judges,
/// no packed, BF16-table or E4M3 equation, no validation output).
fn attributed_to(composite: &Eqn, error: EvalError) -> EvalError {
    let eqn = composite.out;
    match error {
        EvalError::Unsupported { op, detail, .. } => EvalError::Unsupported {
            eqn,
            op: resolve::op_label(&composite.op),
            detail: format!("{op} in its decomposition: {detail}"),
        },
        EvalError::Cast(mut fault) => {
            fault.eqn = eqn;
            EvalError::Cast(fault)
        }
        EvalError::Index(mut fault) => {
            fault.eqn = eqn;
            EvalError::Index(fault)
        }
        EvalError::Budget {
            resource,
            needed,
            limit,
            ..
        } => EvalError::Budget {
            eqn,
            resource,
            needed,
            limit,
        },
        EvalError::UseBeforeDef(_) => EvalError::UseBeforeDef(eqn),
        EvalError::Input { expected, got, .. } => EvalError::Input {
            value: eqn,
            expected,
            got,
        },
        other => other,
    }
}

// ---------------------------------------------------------------------------------------------
// Cast (X3, X9, X7/cast authority)
// ---------------------------------------------------------------------------------------------

fn evaluate_cast<V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &Eqn,
    to: DType,
    env: &[Option<Value>],
    producers: &[Option<usize>],
    idx: usize,
    opts: &mut EvalOptions<'_>,
) -> Result<Value, EvalError> {
    let input_id = match eqn.inputs.first() {
        Some(IrOperand::Value(id)) => *id,
        _ => {
            return Err(EvalError::Unsupported {
                eqn: eqn.out,
                op: "cast",
                detail: format!("cast to {to} has a malformed operand"),
            });
        }
    };
    let from = g.aval(input_id).dtype;
    let out_shape = &g.aval(eqn.out).shape;
    // The exact-integer range check every I32/I8 -> float cast shares (X3).
    let check_exact_range = |words: &[i32]| -> Result<(), EvalError> {
        for (index, &value) in words.iter().enumerate() {
            if !(-ops::cast::F32_EXACT_I32_MAX..=ops::cast::F32_EXACT_I32_MAX).contains(&value) {
                return Err(crate::CastFault {
                    eqn: eqn.out,
                    from,
                    to,
                    index,
                    value: crate::CastOperand::I32(value),
                }
                .into());
            }
        }
        Ok(())
    };
    match (from, to) {
        (DType::F32, DType::E4M3FN) | (DType::E4M3FN, DType::F32) => {
            ops::cast::cast_value(from, to, env_value(env, input_id)?.clone())
        }
        (DType::BF16, DType::F32) => {
            if let Value::Host(tensor) = env_value(env, input_id)? {
                // An explicit (non-owner) BF16 value: its words widen exactly; no owner table to
                // materialize, so no budget charge.
                return Ok(Value::Host(HostTensor::f32(
                    out_shape.clone(),
                    tensor.to_f32()?.into_owned(),
                )));
            }
            let view = crate::operand::bf16_view(eqn, 0, env)?;
            let elements = view.numel();
            opts.charge(eqn.out, idx, DType::F32, elements, elements * 4)?;
            let data = crate::operand::initialized_arc_slice(elements, |flat| view.value(flat));
            Ok(Value::Host(HostTensor::f32_arc(out_shape.clone(), data)))
        }
        // F16 has no owner/exact carrier (unlike BF16): its words widen exactly.
        (DType::F16, DType::F32) => {
            let tensor = host_tensor(eqn, 0, env)?;
            Ok(Value::Host(HostTensor::f32(
                out_shape.clone(),
                tensor.to_f32()?.into_owned(),
            )))
        }
        (DType::I32, DType::F32) => {
            let (words, _) = i32_words(eqn, 0, env)?;
            let role = match opts.cast_authority.as_deref_mut() {
                Some(authority) => {
                    let role = authority.role(eqn.out).ok_or(
                        crate::cast_authority::CastAuthorityError::MissingCastAuthorization {
                            cast: eqn.out,
                        },
                    )?;
                    validate_cast_source(g, producers, eqn.out, input_id, role)?;
                    if words.len() > role.selected_element_ceiling() {
                        return Err(
                            crate::cast_authority::CastAuthorityError::SelectedElementLimit {
                                cast: eqn.out,
                                actual: words.len(),
                                limit: role.selected_element_ceiling(),
                            }
                            .into(),
                        );
                    }
                    Some(role)
                }
                None => None,
            };
            check_exact_range(&words)?;
            if let Some(ExactI32CastRole::HostCheckedSelector(bounds)) = role {
                for (index, &value) in words.iter().enumerate() {
                    if value < 0 || value as usize >= bounds.expert_count {
                        return Err(crate::cast_authority::CastAuthorityError::ExpertIdRange {
                            cast: eqn.out,
                            index,
                            value,
                            experts: bounds.expert_count,
                        }
                        .into());
                    }
                }
            }
            opts.charge(eqn.out, idx, DType::F32, words.len(), words.len() * 4)?;
            let data = crate::operand::initialized_arc_slice(words.len(), |i| words[i] as f32);
            Ok(Value::Host(HostTensor::f32_arc(out_shape.clone(), data)))
        }
        (DType::F32, DType::I32) => {
            let tensor = host_tensor(eqn, 0, env)?;
            let view = tensor.to_f32()?;
            for (index, &value) in view.iter().enumerate() {
                if !value.is_finite() || value.fract() != 0.0 || value.abs() > i32::MAX as f32 {
                    return Err(crate::CastFault {
                        eqn: eqn.out,
                        from,
                        to,
                        index,
                        value: crate::CastOperand::F32(value),
                    }
                    .into());
                }
            }
            let words: Vec<i32> = view.iter().map(|&v| v as i32).collect();
            Ok(Value::Host(HostTensor::i32(out_shape.clone(), words)))
        }
        (DType::I32, DType::I8) => {
            let (words, _) = i32_words(eqn, 0, env)?;
            for (index, &value) in words.iter().enumerate() {
                if !(-128..=127).contains(&value) {
                    return Err(crate::CastFault {
                        eqn: eqn.out,
                        from,
                        to,
                        index,
                        value: crate::CastOperand::I32(value),
                    }
                    .into());
                }
            }
            let bytes: Vec<u8> = words.iter().map(|&word| word as i8 as u8).collect();
            Ok(Value::Host(HostTensor::new(
                DType::I8,
                out_shape.clone(),
                HostData::Bytes(bytes.into()),
            )?))
        }
        (from, to) if from == to => Ok(env_value(env, input_id)?.clone()),
        (DType::F32, DType::BF16 | DType::F16)
        | (DType::BF16, DType::F16)
        | (DType::F16, DType::BF16) => {
            let tensor = host_tensor(eqn, 0, env)?;
            Ok(Value::Host(ops::cast::cast_float(tensor, to)?))
        }
        // I8 -> F32 is an exact widen (I8's range never approaches F32's 2^24 exact-integer bound), and
        // I8 -> I32 is the same words, relabeled.
        (DType::I8, DType::F32) => {
            let (words, _) = i32_words(eqn, 0, env)?;
            let data = crate::operand::initialized_arc_slice(words.len(), |i| words[i] as f32);
            Ok(Value::Host(HostTensor::f32_arc(out_shape.clone(), data)))
        }
        (DType::I8, DType::I32) => {
            let (words, _) = i32_words(eqn, 0, env)?;
            Ok(Value::Host(HostTensor::i32(
                out_shape.clone(),
                words.into_owned(),
            )))
        }
        // I32/I8 -> BF16/F16: via F32 (deval 4), composed from the two single steps already defined:
        // X3's I32->F32 exact-integer range fault, then the half-width narrowing. I8 never approaches
        // the 2^24 bound, so the fault never fires for it.
        (DType::I32 | DType::I8, DType::BF16 | DType::F16) => {
            let (words, _) = i32_words(eqn, 0, env)?;
            check_exact_range(&words)?;
            let values: Vec<f32> = words.iter().map(|&word| word as f32).collect();
            Ok(Value::Host(ops::cast::narrow_f32(
                out_shape.clone(),
                &values,
                to,
            )?))
        }
        // BF16/F16 <-> E4M3FN: via F32 (deval 4), composed from the float-precision widen/narrow above
        // and the E4M3 codec (`fp8::encode_e4m3fn`/`poot_quant::scalar::e4m3fn_to_f32`) `cast_value`
        // already uses for the direct F32 <-> E4M3FN pair.
        (DType::BF16, DType::E4M3FN) => {
            let bytes: Vec<u8> = if let Value::Host(tensor) = env_value(env, input_id)? {
                tensor
                    .to_f32()?
                    .iter()
                    .map(|&v| crate::fp8::encode_e4m3fn(v))
                    .collect()
            } else {
                let view = crate::operand::bf16_view(eqn, 0, env)?;
                (0..view.numel())
                    .map(|i| crate::fp8::encode_e4m3fn(view.value(i)))
                    .collect()
            };
            Ok(Value::Host(crate::fp8::e4m3fn_tensor(
                out_shape.clone(),
                bytes,
            )?))
        }
        (DType::F16, DType::E4M3FN) => {
            let tensor = host_tensor(eqn, 0, env)?;
            let bytes: Vec<u8> = tensor
                .to_f32()?
                .iter()
                .map(|&v| crate::fp8::encode_e4m3fn(v))
                .collect();
            Ok(Value::Host(crate::fp8::e4m3fn_tensor(
                out_shape.clone(),
                bytes,
            )?))
        }
        (DType::E4M3FN, DType::BF16 | DType::F16) => {
            let table = e4m3_tensor(eqn, 0, env)?;
            let values = table.to_f32()?;
            Ok(Value::Host(ops::cast::narrow_f32(
                out_shape.clone(),
                &values,
                to,
            )?))
        }
        (from, to) => Err(EvalError::Unsupported {
            eqn: eqn.out,
            op: "cast",
            detail: format!("cast {from} -> {to} is not wired"),
        }),
    }
}

// ---------------------------------------------------------------------------------------------
// Gather: data-carrier-aware (dense f32 index; exact-I32 index over I32/dense/owner data; E4M3FN)
// ---------------------------------------------------------------------------------------------

fn evaluate_gather<V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &Eqn,
    axis: usize,
    env: &[Option<Value>],
    idx: usize,
    opts: &mut EvalOptions<'_>,
) -> Result<Value, EvalError> {
    let (data_id, index_id) = match eqn.inputs.as_slice() {
        [IrOperand::Value(data), IrOperand::Value(index)] => (*data, *index),
        _ => {
            return Err(EvalError::Unsupported {
                eqn: eqn.out,
                op: "gather",
                detail: "malformed Gather operands".into(),
            });
        }
    };
    if let Some(equation) = resolve::classify(g, eqn) {
        return evaluate_bf16(g, eqn, equation, env, idx, opts);
    }
    let index_shape = g.aval(index_id).shape.clone();
    let out_shape = g.aval(eqn.out).shape.clone();
    // Read the index lazily, straight from whichever lane it is bound in - no intermediate
    // `Vec<IndexValue>` the size of `index` (card 554c allocation bound).
    let index_words = if g.aval(index_id).dtype == DType::I32 {
        Some(value_i32_words(index_id, env)?.0)
    } else {
        None
    };
    let index_data = if index_words.is_none() {
        Some(host_tensor(eqn, 1, env)?.to_f32()?)
    } else {
        None
    };
    let (index_len, index_at_pos): (
        usize,
        Box<dyn Fn(usize) -> ops::index_rule::IndexValue + '_>,
    ) = match (&index_words, &index_data) {
        (Some(words), _) => (
            words.len(),
            Box::new(|p: usize| ops::index_rule::IndexValue::I32(words[p])),
        ),
        (None, Some(data)) => (
            data.len(),
            Box::new(|p: usize| ops::index_rule::IndexValue::F32(data[p])),
        ),
        (None, None) => unreachable!("exactly one of index_words/index_data is populated"),
    };
    match env_value(env, data_id)? {
        Value::Host(table) => Ok(Value::Host(ops::gather::gather_host(
            table,
            index_at_pos,
            index_len,
            &index_shape,
            axis,
            eqn.out,
        )?)),
        Value::Owner(ExactValue::I32(view)) => {
            let words = view.i32_words();
            let out = ops::gather::gather(
                |i| words[i],
                view.shape(),
                index_at_pos,
                index_len,
                &index_shape,
                axis,
                eqn.out,
            )?;
            Ok(Value::Host(HostTensor::i32_arc(out_shape, out)))
        }
        Value::Owner(exact @ (ExactValue::Dense(_) | ExactValue::Bf16(_))) => {
            let out = ops::gather::gather_owner(
                exact,
                index_at_pos,
                index_len,
                &index_shape,
                axis,
                eqn.out,
            )?;
            Ok(Value::Host(HostTensor::f32_arc(out_shape, out)))
        }
        Value::Packed(_) => Err(EvalError::Packed(crate::PackedEvalError::WrongConsumer {
            operation: "Gather",
        })),
    }
}

// ---------------------------------------------------------------------------------------------
// BF16 table (Spec 376): Widen, Gather/RowGatherWiden, Reshape, Transpose, Weight{MatMul,Contraction}
// ---------------------------------------------------------------------------------------------

fn evaluate_bf16<V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &Eqn,
    equation: Bf16Equation<'_>,
    env: &[Option<Value>],
    idx: usize,
    opts: &mut EvalOptions<'_>,
) -> Result<Value, EvalError> {
    let out_shape = &g.aval(eqn.out).shape;
    match equation {
        Bf16Equation::Widen => {
            let source = crate::operand::bf16_view(eqn, 0, env)?;
            opts.charge(eqn.out, idx, DType::F32, source.numel(), source.numel() * 4)?;
            let data =
                crate::operand::initialized_arc_slice(source.numel(), |flat| source.value(flat));
            Ok(Value::Host(HostTensor::f32_arc(out_shape.clone(), data)))
        }
        equation @ (Bf16Equation::Gather { .. } | Bf16Equation::RowGatherWiden) => {
            let axis = match equation {
                Bf16Equation::Gather { axis } => axis,
                _ => 0,
            };
            let table = crate::operand::bf16_view(eqn, 0, env)?;
            let (rows, index_shape) = i32_words(eqn, 1, env)?;
            let extent = table.shape()[axis];
            let rows: Vec<usize> = rows
                .iter()
                .enumerate()
                .map(|(position, &value)| {
                    ops::index_rule::index_at(
                        ops::index_rule::IndexValue::I32(value),
                        position,
                        extent,
                        eqn.out,
                    )
                })
                .collect::<Result<_, _>>()?;
            let data_strides = exact_bf16::canonical_strides(table.shape())?;
            let index_strides = exact_bf16::canonical_strides(&index_shape)?;
            let source_flat = |flat: usize| {
                crate::gather_source_flat(
                    flat,
                    out_shape,
                    &data_strides,
                    &index_strides,
                    axis,
                    |p| rows[p],
                )
            };
            let elements: usize = out_shape.iter().product();
            match equation {
                Bf16Equation::RowGatherWiden => {
                    opts.charge(eqn.out, idx, DType::F32, elements, elements * 4)?;
                    let data = crate::operand::initialized_arc_slice(elements, |flat| {
                        table.value(source_flat(flat))
                    });
                    Ok(Value::Host(HostTensor::f32_arc(out_shape.clone(), data)))
                }
                _ => {
                    opts.charge(eqn.out, idx, DType::BF16, elements, elements * 2)?;
                    let words: Arc<[u16]> =
                        crate::operand::initialized_arc_slice(elements, |flat| {
                            table.word(source_flat(flat))
                        });
                    let view = ExactBf16TensorView::from_derived(words, out_shape.clone())?;
                    Ok(Value::Owner(ExactValue::Bf16(view)))
                }
            }
        }
        Bf16Equation::Reshape { shape } => Ok(Value::Owner(ExactValue::Bf16(
            crate::operand::bf16_view(eqn, 0, env)?.reshape(shape.to_vec())?,
        ))),
        Bf16Equation::Transpose { perm } => Ok(Value::Owner(ExactValue::Bf16(
            crate::operand::bf16_view(eqn, 0, env)?.transpose(perm),
        ))),
        equation @ (Bf16Equation::WeightMatMul | Bf16Equation::WeightContraction) => {
            let activation = host_tensor(eqn, 0, env)?;
            let activation_values = activation.to_f32()?;
            let elements: usize = out_shape.iter().product();
            opts.charge(eqn.out, idx, DType::F32, elements, elements * 4)?;
            let mut data = crate::operand::initialized_arc_slice(elements, |_| 0.0f32);
            // The weight is a BF16 host tensor or an owner-backed view; both read through the same
            // exact BF16 view, never widened ahead of the contraction.
            let weight = crate::operand::bf16_view(eqn, 1, env)?;
            let weight = match equation {
                Bf16Equation::WeightContraction => weight.transpose(&[1, 0]),
                _ => weight,
            };
            ops::contraction::matmul_accumulate(
                activation.shape(),
                |flat| activation_values[flat],
                weight.shape(),
                |flat| weight.value(flat),
                out_shape,
                Arc::get_mut(&mut data).expect("a fresh allocation is uniquely owned"),
            );
            Ok(Value::Host(HostTensor::f32_arc(out_shape.clone(), data)))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// E4M3FN (storage-only; Cast is handled in `evaluate_cast`)
// ---------------------------------------------------------------------------------------------

fn evaluate_e4m3<V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &Eqn,
    env: &[Option<Value>],
) -> Result<Value, EvalError> {
    match &eqn.op {
        OpKind::DenseRowGather {
            source: DType::E4M3FN,
        } => {
            let table = e4m3_tensor(eqn, 0, env)?;
            let index = host_tensor(eqn, 1, env)?;
            Ok(Value::Host(ops::index::dense_row_gather_e4m3(
                table,
                index,
                &g.aval(eqn.out).shape,
                eqn.out,
            )?))
        }
        OpKind::Reshape { shape } => {
            crate::operand::value_operand(eqn, 0, env)?.reshape(shape.clone())
        }
        OpKind::Transpose { perm } => crate::operand::value_operand(eqn, 0, env)?.transpose(perm),
        OpKind::Slice { axis, start, end } => {
            crate::operand::value_operand(eqn, 0, env)?.slice(*axis, *start, *end)
        }
        OpKind::Broadcast { shape } => {
            crate::operand::value_operand(eqn, 0, env)?.broadcast(shape.clone())
        }
        OpKind::Concat { axis } => {
            let values = eqn
                .inputs
                .iter()
                .filter_map(|input| match input {
                    IrOperand::Value(id) => Some(*id),
                    IrOperand::Lit(_) => None,
                })
                .map(|id| env_value(env, id))
                .collect::<Result<Vec<_>, _>>()?;
            Value::concat(&values, *axis)
        }
        OpKind::DynamicUpdateSlice { axis } => {
            let (operand_id, update_id, index) = match eqn.inputs.as_slice() {
                [
                    IrOperand::Value(operand),
                    IrOperand::Value(update),
                    IrOperand::Lit(poot_graph_ir::Scalar::I32(index)),
                ] if *index >= 0 => (*operand, *update, *index as usize),
                [
                    IrOperand::Value(operand),
                    IrOperand::Value(update),
                    IrOperand::Value(index_id),
                ] => {
                    let index_tensor = host_tensor(eqn, 2, env)?;
                    let Some(&runtime_index) = index_tensor.as_f32().and_then(|data| data.first())
                    else {
                        return Err(EvalError::Unsupported {
                            eqn: eqn.out,
                            op: "dynamic_update_slice",
                            detail: "the runtime index needs a dense f32 payload, checked before \
                                     execution"
                                .into(),
                        });
                    };
                    let index = crate::fp8::validate_dynamic_update_runtime_index(
                        &g.aval(*operand).shape,
                        &g.aval(*update).shape,
                        *axis,
                        runtime_index,
                        eqn.out,
                    )?;
                    let _ = index_id;
                    (*operand, *update, index)
                }
                _ => {
                    return Err(EvalError::Unsupported {
                        eqn: eqn.out,
                        op: "dynamic_update_slice",
                        detail:
                            "expected a nonnegative I32 literal or runtime dense-f32 scalar index"
                                .into(),
                    });
                }
            };
            let update = env_value(env, update_id)?.clone();
            env_value(env, operand_id)?.dynamic_update_slice(&update, index, *axis)
        }
        OpKind::ScatterUpdate => {
            let [
                IrOperand::Value(base_id),
                IrOperand::Value(src_id),
                IrOperand::Value(inverse_id),
            ] = eqn.inputs.as_slice()
            else {
                return Err(EvalError::Unsupported {
                    eqn: eqn.out,
                    op: "scatter_update",
                    detail: "malformed ScatterUpdate operands".into(),
                });
            };
            let src = env_value(env, *src_id)?.clone();
            let inverse = host_tensor(eqn, 2, env)?;
            let _ = inverse_id;
            env_value(env, *base_id)?.scatter_update(&src, inverse, eqn.out)
        }
        other => Err(EvalError::Unsupported {
            eqn: eqn.out,
            op: resolve::op_label(other),
            detail: format!("e4m3fn value execution is not wired for {other:?}"),
        }),
    }
}

// ---------------------------------------------------------------------------------------------
// Dense (F32 and I32 words over HostTensor; the common case)
// ---------------------------------------------------------------------------------------------

/// A `HostTensor` over any I32-bearing operand: a host tensor as-is, or a zero-copy alias of an
/// exact-I32 view's word `Arc` (a huge checkpoint-derived table costs one `Arc` clone, not a copy).
fn i32_tensor(id: ValueId, env: &[Option<Value>]) -> Result<HostTensor, EvalError> {
    match env_value(env, id)? {
        Value::Host(tensor) => Ok(tensor.clone()),
        Value::Owner(ExactValue::I32(view)) => Ok(HostTensor::i32_arc(
            view.shape().to_vec(),
            Arc::clone(view.word_owner()),
        )),
        other => Err(EvalError::Input {
            value: id,
            expected: poot_graph_ir::TensorType::new(Vec::new(), DType::I32),
            got: other.carrier_name(),
        }),
    }
}

fn evaluate_dense<V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &Eqn,
    env: &[Option<Value>],
    idx: usize,
    opts: &mut EvalOptions<'_>,
) -> Result<Value, EvalError> {
    let out_shape = g.aval(eqn.out).shape.clone();
    let out_dtype = g.aval(eqn.out).dtype;
    let get = |operand: &IrOperand| -> Result<HostTensor, EvalError> {
        match operand {
            IrOperand::Value(id) => {
                if g.aval(*id).dtype == DType::I32 {
                    i32_tensor(*id, env)
                } else {
                    dense_tensor_or_materialize(*id, env)
                }
            }
            IrOperand::Lit(poot_graph_ir::Scalar::F32(v)) => Ok(HostTensor::scalar(*v)),
            IrOperand::Lit(poot_graph_ir::Scalar::I32(v)) => {
                Ok(HostTensor::i32(Vec::new(), vec![*v]))
            }
        }
    };
    let result = match &eqn.op {
        OpKind::Unary(u) => {
            let x = get(&eqn.inputs[0])?;
            if out_dtype == DType::I32 {
                let operand = I32Operand {
                    shape: x.shape(),
                    words: ops::i32_lane(&x, "unary")?,
                };
                HostTensor::i32_arc(out_shape, ops::elementwise::unary_i32(*u, operand)?)
            } else {
                ops::elementwise::unary_f32(*u, &x)?
            }
        }
        OpKind::Select => {
            let operands = [
                get(&eqn.inputs[0])?,
                get(&eqn.inputs[1])?,
                get(&eqn.inputs[2])?,
            ];
            let words = operands
                .iter()
                .map(|t| ops::i32_lane(t, "select"))
                .collect::<Result<Vec<_>, _>>()?;
            let operand = |i: usize| I32Operand {
                shape: operands[i].shape(),
                words: words[i],
            };
            let selected =
                ops::elementwise::select_i32(operand(0), operand(1), operand(2), &out_shape)?;
            HostTensor::i32_arc(out_shape, selected)
        }
        OpKind::Binary(bop) => {
            let a = get(&eqn.inputs[0])?;
            let b = get(&eqn.inputs[1])?;
            if out_dtype == DType::I32 {
                let words = ops::elementwise::binary_i32(
                    *bop,
                    I32Operand {
                        shape: a.shape(),
                        words: ops::i32_lane(&a, "binary")?,
                    },
                    I32Operand {
                        shape: b.shape(),
                        words: ops::i32_lane(&b, "binary")?,
                    },
                    &out_shape,
                )?;
                HostTensor::i32_arc(out_shape, words)
            } else {
                ops::elementwise::binary_f32(*bop, &a, &b, &out_shape)?
            }
        }
        OpKind::Reduce { op, axis, keepdim } => {
            ops::reduce::reduce(*op, &get(&eqn.inputs[0])?, *axis, *keepdim)?
        }
        OpKind::Broadcast { shape } => ops::movement::broadcast_to(&get(&eqn.inputs[0])?, shape)?,
        OpKind::Reshape { shape } => ops::movement::reshape(&get(&eqn.inputs[0])?, shape)?,
        OpKind::Transpose { perm } => ops::movement::transpose(&get(&eqn.inputs[0])?, perm)?,
        OpKind::Slice { axis, start, end } => {
            ops::movement::slice(&get(&eqn.inputs[0])?, *axis, *start, *end)?
        }
        OpKind::Concat { axis } => {
            let parts = eqn.inputs.iter().map(get).collect::<Result<Vec<_>, _>>()?;
            let refs: Vec<&HostTensor> = parts.iter().collect();
            ops::movement::concat(&refs, *axis)?
        }
        OpKind::Scatter { axis } => {
            debug_assert_eq!(*axis, 0, "scatter is axis-0 only");
            let src = get(&eqn.inputs[0])?;
            let index = get(&eqn.inputs[1])?;
            ops::movement::scatter_axis0(&src, &index, eqn.out)?
        }
        OpKind::ScatterUpdate => {
            let base = get(&eqn.inputs[0])?;
            let src = get(&eqn.inputs[1])?;
            let inverse = get(&eqn.inputs[2])?;
            ops::movement::scatter_update(&base, &src, &inverse, &out_shape, eqn.out)?
        }
        OpKind::ArgTopK { k } => ops::index::arg_top_k(&get(&eqn.inputs[0])?, *k, eqn.out)?,
        OpKind::PackI8 => ops::index::pack_i8(&get(&eqn.inputs[0])?, &out_shape)?,
        OpKind::UnpackI8 { len } => ops::index::unpack_i8(&get(&eqn.inputs[0])?, *len, &out_shape)?,
        OpKind::Iota { len } => ops::index::iota(*len),
        OpKind::MatMul => {
            let a = get(&eqn.inputs[0])?;
            let b = get(&eqn.inputs[1])?;
            ops::contraction::matmul_to(&a, &b, &out_shape)?
        }
        OpKind::IndexedMatMul => {
            let x = get(&eqn.inputs[0])?;
            let w = get(&eqn.inputs[1])?;
            let idx = get(&eqn.inputs[2])?;
            ops::contraction::indexed_matmul(&x, &w, &idx, &out_shape, eqn.out)?
        }
        OpKind::DynamicUpdateSlice { axis } => {
            let operand = get(&eqn.inputs[0])?;
            let update = get(&eqn.inputs[1])?;
            let index_t = get(&eqn.inputs[2])?;
            let index_value = match index_t.data() {
                HostData::I32(words) => ops::index_rule::IndexValue::I32(words[0]),
                HostData::F32(data) => ops::index_rule::IndexValue::F32(data[0]),
                _ => {
                    return Err(EvalError::Unsupported {
                        eqn: eqn.out,
                        op: "dynamic_update_slice",
                        detail: format!("the runtime index is F32 or I32, got {}", index_t.dtype()),
                    });
                }
            };
            let starts = operand.shape()[*axis] - update.shape()[*axis] + 1;
            let index = ops::index_rule::index_at(index_value, 0, starts, eqn.out)?;
            ops::movement::dynamic_update_slice(&operand, &update, index, *axis)?
        }
        OpKind::AllReduce { .. } | OpKind::AllGather { .. } => {
            ops::collective::single_rank_identity(get(&eqn.inputs[0])?)
        }
        // Card 1007: an F16 weight reads through the same definition; `matmul` widens each element exactly.
        OpKind::DenseContraction {
            weight: DType::F32 | DType::F16,
        } => {
            let a = get(&eqn.inputs[0])?;
            let w = get(&eqn.inputs[1])?;
            ops::contraction::dense_contraction(&a, &w, &out_shape)?
        }
        OpKind::DenseContraction { .. } | OpKind::DenseRowGather { .. } => {
            return Err(EvalError::Unsupported {
                eqn: eqn.out,
                op: resolve::op_label(&eqn.op),
                detail: format!(
                    "{out_dtype} is a dense narrow-float operation; needs the storage-aware exact BF16 binding"
                ),
            });
        }
        OpKind::RandomUniform { cols } => {
            let seed = get(&eqn.inputs[0])?;
            ops::sampling::random_uniform(&seed, *cols)?
        }
        OpKind::SampleToken { rule } => {
            let logits = get(&eqn.inputs[0])?;
            let mut rest = eqn.inputs[1..].iter().map(get);
            let (noise, params, top_k) = match rule {
                poot_graph_ir::op::SampleRule::Greedy => (None, None, None),
                poot_graph_ir::op::SampleRule::Gumbel => (
                    Some(rest.next().unwrap()?),
                    Some(rest.next().unwrap()?),
                    None,
                ),
                poot_graph_ir::op::SampleRule::GumbelTopK
                | poot_graph_ir::op::SampleRule::GumbelTopKTopP => (
                    Some(rest.next().unwrap()?),
                    Some(rest.next().unwrap()?),
                    Some(rest.next().unwrap()?),
                ),
            };
            ops::sampling::sample_token(
                *rule,
                &logits,
                noise.as_ref(),
                params.as_ref(),
                top_k.as_ref(),
            )?
        }
        OpKind::Cast { .. }
        | OpKind::Gather { .. }
        | OpKind::PackedDequant { .. }
        | OpKind::PackedContraction { .. }
        | OpKind::PackedRowGather { .. } => {
            unreachable!("handled before evaluate_dense in evaluate_equation")
        }
        OpKind::MatMulBias
        | OpKind::Fused(_)
        | OpKind::FusedRow(_)
        | OpKind::FlashAttentionDecode { .. }
        | OpKind::FlashAttentionPrefill { .. }
        | OpKind::Rope { .. } => {
            unreachable!("a composite evaluates through its decomposition (evaluate_composite)")
        }
    };
    let _ = idx;
    let _ = opts;
    Ok(Value::Host(result))
}

/// An F32-lane reading of operand `id`: a plain `Value::Host`, or an owner-backed exact BF16/F32
/// view materialized once (card 396's "one-equation dense bridge", folded into the common dense path
/// now that every equation dispatches on its own operands, not a graph-wide switch). Never the large
/// embedding-table case: a `Gather`/`DenseRowGather` data operand reads its owner lazily instead
/// (`evaluate_gather`), so this is reached only for an ordinary activation-sized operand (a norm
/// weight, a small table) bound as an owner view instead of eagerly loaded.
fn dense_tensor_or_materialize(
    id: ValueId,
    env: &[Option<Value>],
) -> Result<HostTensor, EvalError> {
    match env_value(env, id)? {
        Value::Host(tensor) => Ok(tensor.clone()),
        Value::Owner(exact @ (ExactValue::Dense(_) | ExactValue::Bf16(_))) => {
            Ok(exact.materialize_dense())
        }
        other => Err(EvalError::Input {
            value: id,
            expected: poot_graph_ir::TensorType::new(Vec::new(), DType::F32),
            got: other.carrier_name(),
        }),
    }
}

/// Test-only counter of calls to the publication helpers on this thread, so a test can prove a failed
/// validation publishes nothing (ADR-0090). Internal single-equation evaluation (the storage-aware
/// E4M3FN movement arms) bypasses this; only the walk's own output/state/environment publication at
/// the end of [`eval`] is counted.
#[cfg(test)]
pub(crate) mod publication_probe {
    use std::cell::Cell;

    thread_local! {
        static PUBLICATIONS: Cell<usize> = const { Cell::new(0) };
    }

    pub(crate) fn record() {
        PUBLICATIONS.with(|count| count.set(count.get() + 1));
    }

    /// Return the count since the last call and reset it.
    pub(crate) fn take() -> usize {
        PUBLICATIONS.with(|count| count.replace(0))
    }
}
