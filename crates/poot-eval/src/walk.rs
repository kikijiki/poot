//! The one evaluator walk: bind, preflight every equation's admission, evaluate each in
//! order, gate on the validation packet, publish. Replaces the five walks this crate used to carry
//! (`eval_env_unchecked`, the storage walk, the one-equation bridge, the exact-I32 walk, the BF16
//! table): one function, one environment (`Value`), admission decided only by an equation's own
//! operands, budgets and observers as options (not separate walks).

use poot_tensor::{DType, HostData, HostTensor};
use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::rc::Rc;
use std::sync::Arc;

use poot_graph_ir::TraceLimits;
use poot_graph_ir::decompose::{Decomposition, decompose};
use poot_graph_ir::{
    ComputedConst, Eqn, Graph, OpClass, OpKind, Operand as IrOperand, ScanAttrs, TensorType,
    ValueId, ValueTable,
};

use crate::exact_bf16::{self, ExactBf16TensorView};
use crate::exact_value::ExactValue;
use crate::observer::{EvalObserver, Publication};
use crate::operand::{e4m3_tensor, env_value, host_tensor, i32_words, value_i32_words};
use crate::ops::elementwise::{I32Operand, LaneOperand, SelectLane};
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

    /// A budget with finite ceilings, for tests that need the budget refusal itself.
    #[cfg(any(test, feature = "test-support"))]
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
    /// The composite equation whose decomposition is being evaluated, if any: every charge made inside
    /// it is attributed to that equation, the one the caller's graph names.
    composite: Option<ValueId>,
    /// Set while one equation of a decomposition evaluates: the decomposition loop has already charged
    /// that equation's output, so an arm's own charge for it is not a second allocation.
    charged_by_composite: bool,
}

impl<'a> EvalOptions<'a> {
    /// Options with `budget`, no observer, and no kept environment.
    pub fn new(budget: EvalBudget) -> Self {
        EvalOptions {
            budget,
            observer: None,
            keep_environment: false,
            composite: None,
            charged_by_composite: false,
        }
    }

    /// Attach an observer that sees each allocation the walk charges.
    pub fn observer(mut self, observer: &'a mut dyn EvalObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Keep the full per-value environment on [`Evaluation::environment`] (replaces `eval_all`).
    pub fn keep_environment(mut self) -> Self {
        self.keep_environment = true;
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
        if self.charged_by_composite {
            return Ok(());
        }
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
    /// The graph's output value.
    pub output: Value,
    /// State outputs in [`Graph::state`] order.
    pub state: Vec<Value>,
    /// The full per-value environment, present only when [`EvalOptions::keep_environment`] was set.
    pub environment: Option<ValueTable<Option<Value>>>,
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
pub fn eval(
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
    mut opts: EvalOptions<'_>,
) -> Result<Evaluation, EvalError> {
    preflight_bindings(g, inputs)?;

    let mut env: ValueTable<Option<Value>> = ValueTable::filled(g.values().len(), None);
    bind_inputs(g, inputs, &mut env)?;

    let mut decompositions = Decompositions::default();
    let releases = (!opts.keep_environment).then(|| releases_after_each_equation(g));
    for (idx, eqn) in g.eqns().iter().enumerate() {
        let value = evaluate_equation(g, eqn, &env, idx, &mut opts, &mut decompositions)?;
        let value = store_as_declared(value, g.aval(eqn.out).dtype)?;
        check_published(g, eqn, &value)?;
        if let Some(observer) = opts.observer.as_deref_mut() {
            observer.value(idx, eqn.out, &value);
        }
        env[eqn.out] = Some(value);
        for &dead in releases.iter().flat_map(|releases| &releases[idx]) {
            env[dead] = None;
        }
    }

    validate_environment(g, &env)?;

    let observer = &mut opts.observer;
    notify_published(observer, Publication::Output(g.result()));
    let output = publish_value(&env, g.result())?;
    let state = g
        .state()
        .map(|(state_in, state_out)| {
            notify_published(observer, Publication::State(state_out));
            let value = publish_value(&env, state_out)?;
            check_state_storage(g, state_in, state_out, &value)?;
            Ok(value)
        })
        .collect::<Result<Vec<_>, EvalError>>()?;
    let environment = opts.keep_environment.then(|| {
        notify_published(&mut opts.observer, Publication::Environment);
        env.clone()
    });

    Ok(Evaluation {
        output,
        state,
        environment,
    })
}

/// For each equation, the values nothing reads after it: those it read for the last time and its own result
/// when no later equation reads that. The walk drops them from the environment, so a graph that dequantizes
/// every weight in turn (the CPU oracle over a 27B checkpoint) holds one weight at a time, not all of them. A
/// liveness root (the result, a witness or export, the next value of a carried state) is never released.
pub(crate) fn releases_after_each_equation(g: &Graph) -> Vec<Vec<ValueId>> {
    let mut last_read: HashMap<ValueId, usize> = HashMap::new();
    for (idx, eqn) in g.eqns().iter().enumerate() {
        for operand in &eqn.inputs {
            if let IrOperand::Value(value) = operand {
                last_read.insert(*value, idx);
            }
        }
    }
    let roots: std::collections::HashSet<ValueId> = g.liveness_roots().collect();
    let mut releases = vec![Vec::new(); g.eqns().len()];
    for (idx, eqn) in g.eqns().iter().enumerate() {
        if !last_read.contains_key(&eqn.out) && !roots.contains(&eqn.out) {
            releases[idx].push(eqn.out);
        }
    }
    for (value, idx) in last_read {
        if !roots.contains(&value) {
            releases[idx].push(value);
        }
    }
    releases
}

/// Tell the caller's observer, if any, what the walk is about to copy out.
fn notify_published(observer: &mut Option<&mut dyn EvalObserver>, publication: Publication) {
    if let Some(observer) = observer.as_deref_mut() {
        observer.published(publication);
    }
}

/// Copy one value out of the completed environment for publication: the one place every output and
/// state value leaves the walk, always after [`validate_environment`] has gated it.
fn publish_value(env: &ValueTable<Option<Value>>, value: ValueId) -> Result<Value, EvalError> {
    env.get(value)
        .and_then(Option::as_ref)
        .cloned()
        .ok_or(EvalError::UseBeforeDef(value))
}

/// The validation-packet gate (ADR-0090/ADR-0101): every declared validation root must be a complete
/// dense F32 tensor of its declared shape; an empty validation set is a no-op.
fn validate_environment(g: &Graph, env: &ValueTable<Option<Value>>) -> Result<(), EvalError> {
    let mut validations = g.witnesses().peekable();
    if validations.peek().is_none() {
        return Ok(());
    }
    let layout = poot_graph_ir::ValidationPacketLayout::for_graph(g);
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

fn check_state_storage(
    g: &Graph,
    state_in: ValueId,
    state_out: ValueId,
    value: &Value,
) -> Result<(), EvalError> {
    let expected = g.aval(state_in);
    let carrier_matches = match value {
        Value::Host(tensor) => tensor.dtype() == expected.dtype,
        Value::Exact(ExactValue::Bf16(_)) => expected.dtype == DType::BF16,
        Value::Exact(_) | Value::Packed(_) | Value::PackedStack(_) => false,
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
fn check_published(g: &Graph, eqn: &Eqn, value: &Value) -> Result<(), EvalError> {
    let expected = g.aval(eqn.out);
    let shape_ok = value.shape().as_ref() == expected.shape.as_slice();
    let class_ok = match value {
        Value::Host(tensor) => tensor.dtype() == expected.dtype,
        Value::Exact(ExactValue::Bf16(_)) => expected.dtype == DType::BF16,
        Value::Exact(ExactValue::I32(_)) => expected.dtype == DType::I32,
        Value::Packed(_) | Value::PackedStack(_) => false,
    };
    debug_assert!(
        shape_ok && class_ok,
        "eqn {} ({}) published {:?} for declared {expected}",
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
/// widening happens at bind), and a non-output exact-dense view must be consumed by an
/// equation that actually reads it.
fn preflight_bindings(g: &Graph, inputs: &HashMap<ValueId, Value>) -> Result<(), EvalError> {
    for &id in g.inputs() {
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
            (DType::BF16, Value::Exact(ExactValue::Bf16(_))) => {}
            (DType::I32, Value::Exact(ExactValue::I32(_))) => {}
            (DType::I8, Value::Packed(_) | Value::PackedStack(_)) => {}
            (_, other) => {
                return Err(EvalError::Input {
                    value: id,
                    expected: aval.clone(),
                    got: other.carrier_name(),
                });
            }
        }
        if let Some(range) = g.meta(id).range() {
            check_declared_range(id, value, range)?;
        }
    }
    Ok(())
}

/// A slot's declared range is a fact its consumers use ([`poot_graph_ir::value_ranges`]), so the bound
/// words are checked against it before any equation runs, the bind check a device executor makes
/// (POOT-726).
fn check_declared_range(
    id: ValueId,
    value: &Value,
    range: poot_graph_ir::IndexRange,
) -> Result<(), EvalError> {
    let words = match value {
        Value::Exact(ExactValue::I32(view)) => std::borrow::Cow::Borrowed(view.i32_words()),
        Value::Host(tensor) => crate::operand::host_i32_words(tensor, id)?,
        other => {
            return Err(EvalError::Input {
                value: id,
                expected: TensorType::new(Vec::new(), DType::I32),
                got: other.carrier_name(),
            });
        }
    };
    match words.iter().position(|&word| !range.contains(word)) {
        Some(index) => Err(crate::SlotRangeFault {
            value: id,
            index,
            word: words[index],
            range,
        }
        .into()),
        None => Ok(()),
    }
}

/// Bind every graph input into `env`, exactly as bound: [`preflight_bindings`] has already checked
/// each host tensor's dtype and shape against the graph's declaration.
fn bind_inputs(
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
    env: &mut ValueTable<Option<Value>>,
) -> Result<(), EvalError> {
    for &id in g.inputs() {
        if let poot_graph_ir::Storage::Computed(computed) = g.meta(id).storage {
            env[id] = Some(Value::Host(computed.tensor()));
            continue;
        }
        env[id] = Some(inputs.get(&id).ok_or(EvalError::MissingInput(id))?.clone());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Evaluate: one equation
// ---------------------------------------------------------------------------------------------

fn touches_e4m3(g: &Graph, eqn: &Eqn) -> bool {
    g.aval(eqn.out).dtype == DType::E4M3FN
        || eqn.inputs.iter().any(|operand| match operand {
            IrOperand::Value(id) => g.aval(*id).dtype == DType::E4M3FN,
            IrOperand::Lit(_) => false,
        })
}

fn evaluate_equation(
    g: &Graph,
    eqn: &Eqn,
    env: &ValueTable<Option<Value>>,
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
        OpKind::PackedContraction {
            descriptor,
            rows: poot_graph_ir::ContractionRows::Indexed { experts },
        } => {
            let [x, ids] = [&eqn.inputs[0], &eqn.inputs[1]].map(|operand| match operand {
                IrOperand::Value(id) => *id,
                IrOperand::Lit(_) => unreachable!("infer admits value operands only"),
            });
            let (x, ids) = (
                dense_tensor_or_materialize(x, env)?,
                dense_tensor_or_materialize(ids, env)?,
            );
            let stack = ops::packed_stack_components(g, eqn, env, descriptor, experts)?;
            let [out, k] = descriptor.shape();
            let out_shape = g.aval(eqn.out).shape.clone();
            let outputs: usize = out_shape.iter().product();
            opts.charge(eqn.out, idx, DType::F32, outputs, outputs * 4)?;
            Ok(Value::Host(ops::evaluate_packed_indexed(
                &x,
                &ids,
                stack,
                &out_shape,
                eqn.out,
                // One decoded `[out, K]` expert at a time: charged per expert actually read.
                || opts.charge(eqn.out, idx, DType::F32, out * k, out * k * 4),
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
        OpKind::Cast { to } => evaluate_cast(g, eqn, to, env, idx, opts),
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

/// What the decomposition of one composite equation may construct. The largest a real graph needs is a
/// scan unrolled over its chunks (a few thousand equations for a 4096-token prefill); these caps are far
/// above that and far below what exhausts a host. `max_tokens` is the bound a model trace applies to its
/// step and means nothing to a decomposition, which has no steps.
const DECOMPOSITION_LIMITS: TraceLimits = TraceLimits {
    max_tokens: NonZeroUsize::MIN,
    max_values: NonZeroUsize::new(1 << 20).unwrap(),
    max_eqns: NonZeroUsize::new(1 << 20).unwrap(),
    max_literal_bytes: NonZeroU64::new(1 << 26).unwrap(),
    max_owned_bytes: NonZeroU64::new(1 << 30).unwrap(),
};

/// The decompositions one [`eval`] call has built, by operand types and then op: a qwen2 decode graph
/// holds hundreds of `Fused` equations, most of them one of a few regions over the same types, so each
/// distinct (op, operand types) is decomposed once per call.
#[derive(Default)]
struct Decompositions {
    by_operand_types: HashMap<Vec<TensorType>, Vec<(OpKind, Rc<Decomposition>)>>,
}

impl Decompositions {
    /// The decomposition of `op` over `operands`: `Ok(None)` when the operation has none.
    fn get(
        &mut self,
        op: &OpKind,
        operands: Vec<TensorType>,
    ) -> Result<Option<Rc<Decomposition>>, poot_graph_ir::BuilderError> {
        if let Some(built) = self.by_operand_types.get(&operands)
            && let Some((_, decomposition)) = built.iter().find(|(built_op, _)| built_op == op)
        {
            return Ok(Some(Rc::clone(decomposition)));
        }
        let Some(decomposition) = decompose(op, &operands, DECOMPOSITION_LIMITS)? else {
            return Ok(None);
        };
        let decomposition = Rc::new(decomposition);
        self.by_operand_types
            .entry(operands)
            .or_default()
            .push((op.clone(), Rc::clone(&decomposition)));
        Ok(Some(decomposition))
    }
}

/// Evaluate a composite equation from its decomposition with this same walk: bind the equation's
/// operand values (as they are, so each primitive reads exactly what the replaced chain read) to the
/// decomposition's operand placeholders, evaluate every primitive equation in order, and return the
/// decomposition's output. Each intermediate is charged to the budget, attributed to the composite
/// equation; the observer sees only the composite's own value, which the caller publishes. An error a
/// primitive raises is attributed to the composite equation too ([`attributed_to`]).
fn evaluate_composite(
    g: &Graph,
    eqn: &Eqn,
    env: &ValueTable<Option<Value>>,
    idx: usize,
    opts: &mut EvalOptions<'_>,
    decompositions: &mut Decompositions,
) -> Result<Value, EvalError> {
    let operand_types: Vec<TensorType> = eqn
        .inputs
        .iter()
        .map(|operand| match operand {
            IrOperand::Value(id) => g.aval(*id).clone(),
            IrOperand::Lit(scalar) => scalar.ty(),
        })
        .collect();
    let bound = eqn
        .inputs
        .iter()
        .map(|operand| {
            Ok(match operand {
                IrOperand::Value(id) => env_value(env, *id)?.clone(),
                IrOperand::Lit(poot_graph_ir::Scalar::F32(v)) => {
                    Value::Host(HostTensor::scalar(*v))
                }
                IrOperand::Lit(poot_graph_ir::Scalar::I32(v)) => {
                    Value::Host(HostTensor::i32(Vec::new(), vec![*v]))
                }
            })
        })
        .collect::<Result<Vec<_>, EvalError>>()?;
    let outermost = opts.composite.is_none();
    if outermost {
        opts.composite = Some(eqn.out);
    }
    let oversized_block_top_k = match &eqn.op {
        OpKind::BlockTopK(attrs) => attrs
            .layout(&operand_types)
            .ok()
            .filter(|layout| block_top_k_pairs(layout) > MAX_BLOCK_TOP_K_PAIRS),
        _ => None,
    };
    let result = match &eqn.op {
        OpKind::Scan(attrs) => {
            evaluate_scan(attrs, eqn, &operand_types, bound, idx, opts, decompositions)
        }
        OpKind::BlockTopK(_) if oversized_block_top_k.is_some() => Err(EvalError::Unsupported {
            eqn: eqn.out,
            op: resolve::op_label(&eqn.op),
            detail: oversized_block_top_k.map_or_else(String::new, |layout| {
                format!(
                    "the oracle's definition ranks every pair of blocks (quadratic); {} query rows of {} blocks exceed its {MAX_BLOCK_TOP_K_PAIRS}-pair bound",
                    layout.rows * layout.tokens,
                    layout.blocks,
                )
            }),
        }),
        OpKind::SelectedAttention(attrs)
            if attrs.layout(&operand_types).is_ok_and(|layout| {
                selected_block_pairs(attrs, &layout) > MAX_SELECTED_BLOCK_PAIRS
            }) =>
        {
            Err(EvalError::Unsupported {
                eqn: eqn.out,
                op: resolve::op_label(&eqn.op),
                detail: format!(
                    "the oracle's definition compares every selected block id with every block for every query; this selection exceeds its {MAX_SELECTED_BLOCK_PAIRS}-comparison bound"
                ),
            })
        }
        op => decompositions
            .get(op, operand_types)
            .map_err(|source| EvalError::Decomposition {
                eqn: eqn.out,
                op: resolve::op_label(op),
                source: Box::new(source),
            })
            .and_then(|decomposition| {
                decomposition.ok_or_else(|| EvalError::Unsupported {
                    eqn: eqn.out,
                    op: resolve::op_label(op),
                    detail: "the composite has no decomposition for its operand types".into(),
                })
            })
            .and_then(|decomposition| {
                run_decomposition(&decomposition, bound, idx, opts, decompositions)
            }),
    };
    if outermost {
        opts.composite = None;
    }
    result.map_err(|error| attributed_to(eqn, error))
}

/// The most block pairs a `BlockTopK` decomposition may rank at once: its definition compares every block
/// with every other for every query, so its tensors grow with the square of the block count. A graph past
/// the bound is refused rather than allocated (8192 blocks of one query is 2^26 pairs).
const MAX_BLOCK_TOP_K_PAIRS: usize = 1 << 27;

/// The most `(selected id, block)` comparisons a `SelectedAttention` decomposition may make at once: its mask is
/// built from an indicator of every query's `K` ids against every block.
const MAX_SELECTED_BLOCK_PAIRS: usize = 1 << 27;

/// The comparisons the mask of a selected attention makes: `batch * lq * k * blocks`, saturating.
fn selected_block_pairs(
    attrs: &poot_graph_ir::SelectedAttentionAttrs,
    layout: &poot_graph_ir::SelectedAttentionLayout,
) -> usize {
    (layout.batch * layout.lq)
        .saturating_mul(attrs.k)
        .saturating_mul(layout.lk.div_ceil(attrs.ratio))
}

/// The pairs a block top-k ranks: `queries * blocks^2`, saturating.
fn block_top_k_pairs(layout: &poot_graph_ir::BlockTopKLayout) -> usize {
    (layout.rows * layout.tokens).saturating_mul(layout.blocks.saturating_mul(layout.blocks))
}

/// Evaluate `decomposition` over `bound`, the values of its operand placeholders in operand order: every
/// primitive equation in order, each intermediate charged to the budget, the decomposition's output
/// returned.
fn run_decomposition(
    decomposition: &Decomposition,
    bound: Vec<Value>,
    idx: usize,
    opts: &mut EvalOptions<'_>,
    decompositions: &mut Decompositions,
) -> Result<Value, EvalError> {
    let inner = &decomposition.graph;
    let mut inner_env: ValueTable<Option<Value>> = ValueTable::filled(inner.values().len(), None);
    for (&placeholder, value) in decomposition.operands.iter().zip(bound) {
        inner_env[placeholder] = Some(value);
    }
    for inner_eqn in inner.eqns() {
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
        opts.charged_by_composite = true;
        let value = evaluate_equation(inner, inner_eqn, &inner_env, idx, opts, decompositions);
        opts.charged_by_composite = false;
        let value = value?;
        let value = store_as_declared(value, declared.dtype)?;
        check_published(inner, inner_eqn, &value)?;
        inner_env[inner_eqn.out] = Some(value);
    }
    inner_env[inner.result()]
        .take()
        .ok_or(EvalError::UseBeforeDef(inner.result()))
}

/// Evaluate a `Scan` one chunk at a time with the state carried between chunks: each window of
/// `attrs.chunk` positions (the last one shorter) is the same scan over that window, seeded with the
/// previous window's `s_out`, through its own decomposition. The oracle holds one chunk body at a
/// time, so it has no sequence-length ceiling and the working set does not grow with the sequence; the
/// packed result is `y` windows followed by the final state, the layout the full scan declares.
fn evaluate_scan(
    attrs: &ScanAttrs,
    eqn: &Eqn,
    types: &[TensorType],
    operands: Vec<Value>,
    idx: usize,
    opts: &mut EvalOptions<'_>,
    decompositions: &mut Decompositions,
) -> Result<Value, EvalError> {
    let layout = attrs
        .layout(types)
        .map_err(|error| EvalError::Unsupported {
            eqn: eqn.out,
            op: "scan",
            detail: error.to_string(),
        })?;
    let sequence = attrs.family.sequence_operands();
    let state_at = attrs.family.operands() - 1;
    let mut state = operands[state_at].clone();
    let mut parts = Vec::with_capacity(layout.len.div_ceil(attrs.chunk) + 1);
    for start in (0..layout.len).step_by(attrs.chunk) {
        let end = (start + attrs.chunk).min(layout.len);
        let mut window_types = types.to_vec();
        let mut window = Vec::with_capacity(operands.len());
        for (position, value) in operands.iter().enumerate() {
            window.push(if sequence.contains(&position) {
                window_types[position].shape[2] = end - start;
                value.slice(2, start, end)?
            } else if position == state_at {
                state.clone()
            } else {
                value.clone()
            });
        }
        let decomposition = decompositions
            .get(&eqn.op, window_types)
            .map_err(|source| EvalError::Decomposition {
                eqn: eqn.out,
                op: "scan",
                source: Box::new(source),
            })?
            .ok_or_else(|| EvalError::Unsupported {
                eqn: eqn.out,
                op: "scan",
                detail: "the scan has no decomposition for its chunk window".into(),
            })?;
        let packed = run_decomposition(&decomposition, window, idx, opts, decompositions)?;
        let len = end - start;
        parts.push(packed.slice(2, 0, len)?);
        state = packed.slice(2, len, len + layout.state_rows)?;
    }
    parts.push(state);
    Value::concat(&parts.iter().collect::<Vec<_>>(), 2)
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
// Cast (X3, X9)
// ---------------------------------------------------------------------------------------------

fn evaluate_cast(
    g: &Graph,
    eqn: &Eqn,
    to: DType,
    env: &ValueTable<Option<Value>>,
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
                // A plain host BF16 value: its words widen exactly; no exact view to
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
        // F16 has no exact carrier (unlike BF16): its words widen exactly.
        (DType::F16, DType::F32) => {
            let tensor = host_tensor(eqn, 0, env)?;
            Ok(Value::Host(HostTensor::f32(
                out_shape.clone(),
                tensor.to_f32()?.into_owned(),
            )))
        }
        (DType::I32, DType::F32) => {
            let (words, _) = i32_words(eqn, 0, env)?;
            check_exact_range(&words)?;
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
// Gather: data-carrier-aware (I32 index over I32/dense/exact data; E4M3FN)
// ---------------------------------------------------------------------------------------------

fn evaluate_gather(
    g: &Graph,
    eqn: &Eqn,
    axis: usize,
    env: &ValueTable<Option<Value>>,
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
    // Read the index lazily, straight from its I32 words - no intermediate `Vec<IndexValue>` the size of
    // `index` (card 554c allocation bound).
    let (index_words, _) = value_i32_words(index_id, env)?;
    let index_len = index_words.len();
    let index_at_pos = |p: usize| ops::index_rule::IndexValue::I32(index_words[p]);
    match env_value(env, data_id)? {
        Value::Host(table) => Ok(Value::Host(ops::gather::gather_host(
            table,
            index_at_pos,
            index_len,
            &index_shape,
            axis,
            eqn.out,
        )?)),
        Value::Exact(ExactValue::I32(view)) => {
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
        // A `Value::Exact(Bf16)` is only ever bound to, or produced for, a BF16-typed value
        // (`preflight_bindings` and the Spec 376 equations), `Gather` infers only I32 ids, and
        // `classify` routes every such `Gather` through `evaluate_bf16` above.
        Value::Exact(ExactValue::Bf16(_)) => unreachable!(
            "a Gather of a BF16 table is the Spec 376 Gather equation, classified before this match"
        ),
        Value::Packed(_) | Value::PackedStack(_) => {
            Err(EvalError::Packed(crate::PackedEvalError::WrongConsumer {
                operation: "Gather",
            }))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// BF16 table (Spec 376): Widen, Gather/RowGatherWiden, Reshape, Transpose, Weight{MatMul,Contraction}
// ---------------------------------------------------------------------------------------------

fn evaluate_bf16(
    g: &Graph,
    eqn: &Eqn,
    equation: Bf16Equation<'_>,
    env: &ValueTable<Option<Value>>,
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
        Bf16Equation::Gather { axis } => {
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
            opts.charge(eqn.out, idx, DType::BF16, elements, elements * 2)?;
            let words: Arc<[u16]> = crate::operand::initialized_arc_slice(elements, |flat| {
                table.word(source_flat(flat))
            });
            let view = ExactBf16TensorView::from_derived(words, out_shape.clone())?;
            Ok(Value::Exact(ExactValue::Bf16(view)))
        }
        Bf16Equation::Reshape { shape } => Ok(Value::Exact(ExactValue::Bf16(
            crate::operand::bf16_view(eqn, 0, env)?.reshape(shape.to_vec())?,
        ))),
        Bf16Equation::Transpose { perm } => Ok(Value::Exact(ExactValue::Bf16(
            crate::operand::bf16_view(eqn, 0, env)?.transpose(perm),
        ))),
        Bf16Equation::WeightMatMul => {
            let activation = host_tensor(eqn, 0, env)?;
            let activation_values = activation.to_f32()?;
            let elements: usize = out_shape.iter().product();
            opts.charge(eqn.out, idx, DType::F32, elements, elements * 4)?;
            let mut data = crate::operand::initialized_arc_slice(elements, |_| 0.0f32);
            // The weight is a BF16 host tensor or an exact view; both read through the same
            // exact BF16 view, never widened ahead of the contraction.
            let weight = crate::operand::bf16_view(eqn, 1, env)?;
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

fn evaluate_e4m3(
    g: &Graph,
    eqn: &Eqn,
    env: &ValueTable<Option<Value>>,
) -> Result<Value, EvalError> {
    match &eqn.op {
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
                    let (words, _) = value_i32_words(*index_id, env)?;
                    let Some(&runtime_index) = words.first() else {
                        return Err(EvalError::Unsupported {
                            eqn: eqn.out,
                            op: "dynamic_update_slice",
                            detail: "the runtime index needs an I32 word, checked before execution"
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
                    (*operand, *update, index)
                }
                _ => {
                    return Err(EvalError::Unsupported {
                        eqn: eqn.out,
                        op: "dynamic_update_slice",
                        detail: "expected a nonnegative I32 literal or runtime I32 scalar index"
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
                IrOperand::Value(slots_id),
            ] = eqn.inputs.as_slice()
            else {
                return Err(EvalError::Unsupported {
                    eqn: eqn.out,
                    op: "scatter_update",
                    detail: "malformed ScatterUpdate operands".into(),
                });
            };
            let src = env_value(env, *src_id)?.clone();
            let slots = host_tensor(eqn, 2, env)?;
            let _ = slots_id;
            env_value(env, *base_id)?.scatter_update(&src, slots, eqn.out)
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
fn i32_tensor(id: ValueId, env: &ValueTable<Option<Value>>) -> Result<HostTensor, EvalError> {
    match env_value(env, id)? {
        Value::Host(tensor) => Ok(tensor.clone()),
        Value::Exact(ExactValue::I32(view)) => Ok(HostTensor::i32_arc(
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

fn evaluate_dense(
    g: &Graph,
    eqn: &Eqn,
    env: &ValueTable<Option<Value>>,
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
    // The operand a site that only moves elements reads: an exact BF16 view (what a reshape or
    // transpose of a BF16 value is) materializes as its own two-byte words, never as the F32 values
    // arithmetic reads, so a NaN payload reaches the output unchanged and the parts of a concat agree.
    let stored = |operand: &IrOperand| match operand {
        IrOperand::Value(id) if g.aval(*id).dtype != DType::I32 => {
            stored_tensor_or_materialize(*id, env)
        }
        other => get(other),
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
                stored(&eqn.inputs[1])?,
                stored(&eqn.inputs[2])?,
            ];
            let condition = I32Operand {
                shape: operands[0].shape(),
                words: ops::i32_lane(&operands[0], "select")?,
            };
            match out_dtype {
                DType::I32 => {
                    let words = ops::elementwise::select(
                        condition,
                        select_lanes(&operands[1], out_dtype)?,
                        select_lanes(&operands[2], out_dtype)?,
                        &out_shape,
                    )?;
                    HostTensor::i32_arc(out_shape, words)
                }
                DType::F32 => {
                    let words = ops::elementwise::select(
                        condition,
                        select_lanes(&operands[1], out_dtype)?,
                        select_lanes(&operands[2], out_dtype)?,
                        &out_shape,
                    )?;
                    HostTensor::f32_arc(out_shape, words)
                }
                half @ (DType::F16 | DType::BF16) => {
                    let words = ops::elementwise::select(
                        condition,
                        select_lanes(&operands[1], out_dtype)?,
                        select_lanes(&operands[2], out_dtype)?,
                        &out_shape,
                    )?;
                    if half == DType::F16 {
                        HostTensor::f16_arc(out_shape, words)
                    } else {
                        HostTensor::bf16_arc(out_shape, words)
                    }
                }
                other => {
                    return Err(EvalError::unsupported(
                        "select",
                        format!("Select has no {other} lane"),
                    ));
                }
            }
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
        OpKind::Broadcast { shape } => {
            ops::movement::broadcast_to(&stored(&eqn.inputs[0])?, shape)?
        }
        OpKind::Reshape { shape } => ops::movement::reshape(&stored(&eqn.inputs[0])?, shape)?,
        OpKind::Transpose { perm } => ops::movement::transpose(&stored(&eqn.inputs[0])?, perm)?,
        OpKind::Slice { axis, start, end } => {
            ops::movement::slice(&stored(&eqn.inputs[0])?, *axis, *start, *end)?
        }
        OpKind::Concat { axis } => {
            let parts = eqn
                .inputs
                .iter()
                .map(stored)
                .collect::<Result<Vec<_>, _>>()?;
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
            let slots = get(&eqn.inputs[2])?;
            ops::movement::scatter_update(&base, &src, &slots, &out_shape, eqn.out)?
        }
        OpKind::ArgTopK { k } => ops::index::arg_top_k(&get(&eqn.inputs[0])?, *k, eqn.out)?,
        OpKind::PackI8 => ops::index::pack_i8(&get(&eqn.inputs[0])?, &out_shape)?,
        OpKind::UnpackI8 { len } => ops::index::unpack_i8(&get(&eqn.inputs[0])?, *len, &out_shape)?,
        OpKind::Iota { len, dtype } => ComputedConst::Iota {
            len: *len,
            dtype: *dtype,
        }
        .tensor(),
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
                _ => {
                    return Err(EvalError::Unsupported {
                        eqn: eqn.out,
                        op: "dynamic_update_slice",
                        detail: format!("the runtime index is I32, got {}", index_t.dtype()),
                    });
                }
            };
            let starts = operand.shape()[*axis] - update.shape()[*axis] + 1;
            let index = ops::index_rule::index_at(index_value, 0, starts, eqn.out)?;
            ops::movement::dynamic_update_slice(&operand, &update, index, *axis)?
        }
        OpKind::AllReduce { .. } => ops::collective::single_rank_identity(get(&eqn.inputs[0])?),
        OpKind::AllGather { world, .. } => {
            ops::collective::all_gather(get(&eqn.inputs[0])?, *world, eqn.out)?
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
        | OpKind::PackedRowGather { .. }
        | OpKind::PackedContraction {
            rows: poot_graph_ir::ContractionRows::Indexed { .. },
            ..
        } => {
            unreachable!("handled before evaluate_dense in evaluate_equation")
        }
        OpKind::PackedContraction {
            rows: poot_graph_ir::ContractionRows::Blocks(_),
            ..
        }
        | OpKind::DenseContraction { .. }
        | OpKind::GroupedDenseContraction { .. }
        | OpKind::DenseRowGather { .. }
        | OpKind::ContractionEpilogue { .. }
        | OpKind::Fused(_)
        | OpKind::FusedRow(_)
        | OpKind::Attention(_)
        | OpKind::Rope { .. }
        | OpKind::Scan(_)
        | OpKind::BlockTopK(_)
        | OpKind::SelectedAttention(_) => {
            unreachable!("a composite evaluates through its decomposition (evaluate_composite)")
        }
    };
    let _ = idx;
    let _ = opts;
    Ok(Value::Host(result))
}

/// An F32-lane reading of operand `id`: a plain `Value::Host`, or an exact BF16/F32
/// view materialized once (card 396's "one-equation dense bridge", folded into the common dense path
/// now that every equation dispatches on its own operands, not a graph-wide switch). Never the large
/// embedding-table case: a `Gather`/`DenseRowGather` data operand reads the exact view lazily
/// instead (`evaluate_gather`), so this is reached only for an ordinary activation-sized operand (a
/// norm weight, a small table) bound as an exact view instead of eagerly loaded.
fn dense_tensor_or_materialize(
    id: ValueId,
    env: &ValueTable<Option<Value>>,
) -> Result<HostTensor, EvalError> {
    match env_value(env, id)? {
        Value::Host(tensor) => Ok(tensor.clone()),
        Value::Exact(exact @ ExactValue::Bf16(_)) => Ok(exact.materialize_dense()),
        other => Err(EvalError::Input {
            value: id,
            expected: poot_graph_ir::TensorType::new(Vec::new(), DType::F32),
            got: other.carrier_name(),
        }),
    }
}

/// [`dense_tensor_or_materialize`] keeping the value's stored dtype and bits: an exact BF16 view
/// materializes as BF16 words.
fn stored_tensor_or_materialize(
    id: ValueId,
    env: &ValueTable<Option<Value>>,
) -> Result<HostTensor, EvalError> {
    match env_value(env, id)? {
        Value::Exact(exact @ ExactValue::Bf16(_)) => Ok(exact.materialize_stored()),
        _ => dense_tensor_or_materialize(id, env),
    }
}

/// A `Select` branch's lanes in the result's lane type, or the typed refusal naming the mismatch.
fn select_lanes<W: SelectLane>(
    tensor: &HostTensor,
    out_dtype: DType,
) -> Result<LaneOperand<'_, W>, EvalError> {
    W::lanes_of(tensor)
        .map(|words| LaneOperand {
            shape: tensor.shape(),
            words,
        })
        .ok_or_else(|| {
            EvalError::unsupported(
                "select",
                format!(
                    "a {out_dtype} Select branch carries {} lanes",
                    tensor.dtype()
                ),
            )
        })
}
