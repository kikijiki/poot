//! The tracer: a `Builder` that mints `ValueId`s and appends eqns while a model forward runs with type-only
//! `Traced` handles (the `DynamicJaxprTrace` analog). Interior mutability lets ops take `&self`.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::error::{BuilderAppendError, BuilderCollection, BuilderValueNamespace};
use crate::graph::{
    ComputedConst, Eqn, Graph, LayerIndex, Operand, Slot, SlotKey, StateRole, Storage, ValueId,
    ValueMeta,
};
use crate::op::{OpKind, SampleRule};
#[cfg(test)]
use crate::types::DType;
use crate::types::{Scalar, TensorType};
use poot_quant::PackedWeight;

/// A type-only tensor handle: a `ValueId` with no data. There is no `Deref`/`as_slice`/`to_scalar`, so a host
/// read of a traced tensor cannot be written (graph-architecture.md Q1). Static shape is available via
/// [`Builder::aval`] for host-side static decisions (`n_rep`, the rope half-split, etc.).
#[derive(Clone, Copy, Debug)]
pub struct Traced {
    pub id: ValueId,
}

/// Checked accounting for a prepared transactional append.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuilderAppendAccounting {
    pub staged_values: usize,
    pub staged_equations: usize,
    pub packed_rows: usize,
    pub checked_name_bytes: usize,
}

/// A complete graph fragment staged against one builder generation.
///
/// Values and equations are private so callers cannot bypass checked id allocation or inference. Construct a
/// fragment with [`BuilderAppendPlan::input`], [`BuilderAppendPlan::input_result`],
/// [`BuilderAppendPlan::equation`], and [`BuilderAppendPlan::declare_result`].
#[derive(Debug)]
pub struct BuilderAppendPlan {
    builder_identity: Rc<()>,
    generation: u64,
    base_value_count: usize,
    base_types: Vec<TensorType>,
    values: Vec<ValueMeta>,
    equations: Vec<Eqn>,
    result: Option<DeclaredAppendResult>,
    packed_rows: usize,
    checked_name_bytes: usize,
    /// The builder's layer tag when the plan was created. A plan stages its equations after the
    /// caller may have left the scope, so the tag is snapshotted here (see [`Builder::append_plan`]).
    layer: Option<LayerIndex>,
}

#[derive(Clone, Copy, Debug)]
enum DeclaredAppendResult {
    StagedInput(ValueId),
    EquationOutput(ValueId),
}

impl DeclaredAppendResult {
    fn value(self) -> ValueId {
        match self {
            Self::StagedInput(value) | Self::EquationOutput(value) => value,
        }
    }
}

impl BuilderAppendPlan {
    fn next_value_id(&self) -> Result<ValueId, BuilderAppendError> {
        self.base_value_count.checked_add(self.values.len()).ok_or(
            BuilderAppendError::SizeOverflow {
                collection: BuilderCollection::Values,
                current: self.base_value_count,
                additional: self.values.len(),
            },
        )
    }

    fn add_name_bytes(&mut self, name: &str) -> Result<(), BuilderAppendError> {
        self.checked_name_bytes = self.checked_name_bytes.checked_add(name.len()).ok_or(
            BuilderAppendError::SizeOverflow {
                collection: BuilderCollection::NameBytes,
                current: self.checked_name_bytes,
                additional: name.len(),
            },
        )?;
        Ok(())
    }

    /// Stage a named input binder for use by later equations. Use [`Self::input_result`] when the plan-owned input
    /// is the append result. Device values must be created by [`Self::equation`].
    pub fn input(
        &mut self,
        name: impl Into<String>,
        aval: TensorType,
        storage: Storage,
    ) -> Result<Traced, BuilderAppendError> {
        let name = name.into();
        if storage == Storage::Device {
            return Err(BuilderAppendError::DeviceInput { name });
        }
        debug_assert!(
            storage != Storage::State,
            "BuilderAppendPlan::input: no append-plan caller stages a State input; \
             Builder::state_input builds one directly"
        );
        let id = self.next_value_id()?;
        self.add_name_bytes(&name)?;
        self.values.push(ValueMeta {
            aval,
            storage,
            name: Some(name),
            // This generic staging method knows only the already-formatted wire name; a `Slot`
            // staged here (rather than through `Builder::slot`/`slot_named`, which build the typed
            // `SlotKey` up front) would need the caller to patch it in once `commit_append`
            // returns a real `ValueId`.
            key: None,
            state_role: None,
        });
        Ok(Traced { id })
    }

    /// Atomically stage a named non-Device input binder and declare that plan-owned input as the result.
    pub fn input_result(
        &mut self,
        name: impl Into<String>,
        aval: TensorType,
        storage: Storage,
    ) -> Result<Traced, BuilderAppendError> {
        if let Some(result) = self.result {
            return Err(BuilderAppendError::ResultAlreadyDeclared {
                value: result.value(),
            });
        }
        let result = self.input(name, aval, storage)?;
        self.result = Some(DeclaredAppendResult::StagedInput(result.id));
        Ok(result)
    }

    /// Stage one inferred Device equation result.
    ///
    /// This accepts any [`OpKind`], composites included ([`OpKind::class`]), so an oracle or planner test
    /// can stage a composite directly. A tracer must not use it for a composite: tracers state primitives
    /// and `ops::` compositions, and `compile`'s passes form the composites. The guard is the corpus check
    /// `every_corpus_trace_is_primitive_before_compile` (`poot-graph-plan`'s plan-summary test).
    pub fn equation(
        &mut self,
        op: OpKind,
        operands: Vec<Operand>,
    ) -> Result<Traced, BuilderAppendError> {
        self.equation_named(op, operands, None)
    }

    /// Stage one inferred Device equation result with an optional diagnostic name. Like
    /// [`Self::equation`], it accepts composites for tests; a tracer must not stage one.
    pub fn equation_named(
        &mut self,
        op: OpKind,
        operands: Vec<Operand>,
        name: Option<String>,
    ) -> Result<Traced, BuilderAppendError> {
        let equation = self.equations.len();
        let operation = op.name();
        let mut in_types = Vec::with_capacity(operands.len());
        for operand in &operands {
            match operand {
                Operand::Lit(value) => in_types.push(value.ty()),
                Operand::Value(value) => {
                    let Some(aval) = self.type_of(*value) else {
                        return Err(BuilderAppendError::InvalidOperand {
                            equation,
                            operation,
                            value: *value,
                        });
                    };
                    in_types.push(aval.clone());
                }
            }
        }
        checked_op_sizes(&op, &in_types)?;
        if let [Operand::Value(_), Operand::Value(_)] = operands.as_slice()
            && matches!(op, OpKind::Binary(_))
        {
            crate::op::check_binary_value_dtypes(&in_types[0], &in_types[1]).map_err(|source| {
                BuilderAppendError::Inference {
                    equation,
                    operation: operation.clone(),
                    source,
                }
            })?;
        }
        let aval = op
            .infer(&in_types)
            .map_err(|source| BuilderAppendError::Inference {
                equation,
                operation: operation.clone(),
                source,
            })?;
        let id = self.next_value_id()?;
        if let Some(name) = name.as_deref() {
            self.add_name_bytes(name)?;
        }
        self.values.push(ValueMeta {
            aval,
            storage: Storage::Device,
            name,
            key: None,
            state_role: None,
        });
        self.equations.push(Eqn {
            op,
            inputs: operands,
            out: id,
            layer: self.layer,
        });
        Ok(Traced { id })
    }

    /// Declare a staged equation output as the result. Input results must be staged and declared together with
    /// [`Self::input_result`]; preflight enforces this because `Traced` carries only a public numeric id.
    pub fn declare_result(&mut self, result: Traced) -> Result<(), BuilderAppendError> {
        if let Some(value) = self.result {
            return Err(BuilderAppendError::ResultAlreadyDeclared {
                value: value.value(),
            });
        }
        self.result = Some(DeclaredAppendResult::EquationOutput(result.id));
        Ok(())
    }

    pub fn type_of(&self, value: ValueId) -> Option<&TensorType> {
        if value < self.base_value_count {
            self.base_types.get(value)
        } else {
            self.values
                .get(value.checked_sub(self.base_value_count)?)
                .map(|meta| &meta.aval)
        }
    }
}

/// A fully checked append. Neither `Clone` nor reusable.
#[derive(Debug)]
pub struct PreparedBuilderAppend {
    builder_identity: Rc<()>,
    generation: u64,
    base_value_count: usize,
    accounting: BuilderAppendAccounting,
    staged_inputs: usize,
    staged_constants: usize,
    staged_slots: usize,
    result: PreparedAppendResult,
    plan: Option<BuilderAppendPlan>,
}

impl PreparedBuilderAppend {
    pub fn accounting(&self) -> BuilderAppendAccounting {
        self.accounting
    }
}

#[derive(Debug)]
enum PreparedAppendResult {
    StagedInput(ValueId),
    EquationOutput(ValueId),
}

impl PreparedAppendResult {
    fn value(&self) -> ValueId {
        match *self {
            Self::StagedInput(value) | Self::EquationOutput(value) => value,
        }
    }
}

/// The `MatMulDequant` family is deleted (card 545b): `Builder` has no `matmul_dequant` or
/// `indexed_matmul_dequant` method, and `poot_graph_ir` exports no `QuantScheme`. `PackedDequant` /
/// `PackedContraction` (built through [`crate::ops::packed_linear`] and friends) are the only quantized
/// primitives left.
///
/// ```compile_fail,E0599
/// use poot_graph_ir::builder::Builder;
///
/// let b = Builder::new();
/// let _ = b.matmul_dequant();
/// ```
///
/// ```compile_fail,E0599
/// use poot_graph_ir::builder::Builder;
///
/// let b = Builder::new();
/// let _ = b.indexed_matmul_dequant();
/// ```
///
/// ```compile_fail,E0432
/// use poot_graph_ir::QuantScheme;
/// ```
///
/// Card 557: a tracer emits primitives and `ops::` compositions only. The bias epilogue and flash
/// attention are composites `compile`'s passes form ([`crate::op::OpKind::class`]), and the kernel choice
/// behind them is the planner's, so `Builder` has no method for either. Each call below uses the deleted
/// method's own signature, so re-adding one makes its example compile.
///
/// ```compile_fail,E0599
/// use poot_graph_ir::{Builder, TensorType};
///
/// let b = Builder::new();
/// let x = b.constant("x", TensorType::f32(vec![2, 4]));
/// let w = b.constant("w", TensorType::f32(vec![4, 3]));
/// let bias = b.constant("bias", TensorType::f32(vec![3]));
/// let _ = b.matmul_bias(x, w, bias);
/// ```
///
/// ```compile_fail,E0599
/// use poot_graph_ir::{Builder, TensorType};
///
/// let b = Builder::new();
/// let q = b.constant("q", TensorType::f32(vec![1, 2, 1, 4]));
/// let k = b.constant("k", TensorType::f32(vec![1, 2, 8, 4]));
/// let v = b.constant("v", TensorType::f32(vec![1, 2, 8, 4]));
/// let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, 8]));
/// let _ = b.flash_attention_decode(q, k, v, mask, 1, 0.5);
/// ```
///
/// ```compile_fail,E0599
/// use poot_graph_ir::{Builder, TensorType};
///
/// let b = Builder::new();
/// let q = b.constant("q", TensorType::f32(vec![1, 2, 4, 4]));
/// let k = b.constant("k", TensorType::f32(vec![1, 2, 4, 4]));
/// let v = b.constant("v", TensorType::f32(vec![1, 2, 4, 4]));
/// let mask = b.constant("mask", TensorType::f32(vec![1, 1, 4, 4]));
/// let _ = b.flash_attention_prefill(q, k, v, mask, 1, 0.5);
/// ```
///
/// ```compile_fail,E0599
/// use poot_graph_ir::{Builder, TensorType};
///
/// let b = Builder::new();
/// let q = b.constant("q", TensorType::f32(vec![1, 2, 4, 4]));
/// let k = b.constant("k", TensorType::f32(vec![1, 2, 4, 4]));
/// let v = b.constant("v", TensorType::f32(vec![1, 2, 4, 4]));
/// let mask = b.constant("mask", TensorType::f32(vec![1, 1, 4, 4]));
/// let _ = b.flash_attention_prefill_softcap(q, k, v, mask, 1, 0.5, Some(50.0));
/// ```
pub struct Builder {
    identity: Rc<()>,
    g: RefCell<Graph>,
    generation: Cell<u64>,
    /// The layer tag every equation emitted right now carries. Shared with the [`LayerScope`]s this
    /// builder hands out, so a scope restores the previous tag on drop without borrowing the
    /// builder (a tracer holds the guard while its `&mut self` layer method runs).
    layer: Rc<Cell<Option<LayerIndex>>>,
    #[cfg(test)]
    reservation_failure: Cell<Option<BuilderCollection>>,
}

/// The result of [`Builder::resume`]: a fresh builder over a previously finished graph, plus that
/// graph's old primary output and state pairs, rehydrated as [`Traced`] so a caller can append new
/// equations that read them and finish the graph again.
pub struct Resumed {
    pub builder: Builder,
    pub out: Traced,
    pub state: Vec<(Traced, Traced)>,
}

/// The active [`Builder::layer_scope`]: tags every equation emitted while alive with one layer and
/// restores the previous tag on drop.
///
/// Scopes nest (a layer scope inside a graph-wide one restores it correctly) and must be dropped in
/// stack order, which every `?`, `return` and block exit already guarantees.
#[derive(Debug)]
pub struct LayerScope {
    layer: Rc<Cell<Option<LayerIndex>>>,
    previous: Option<LayerIndex>,
}

impl Drop for LayerScope {
    fn drop(&mut self) {
        self.layer.set(self.previous);
    }
}

fn checked_growth(
    collection: BuilderCollection,
    current: usize,
    additional: usize,
) -> Result<usize, BuilderAppendError> {
    current
        .checked_add(additional)
        .ok_or(BuilderAppendError::SizeOverflow {
            collection,
            current,
            additional,
        })
}

fn reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    collection: BuilderCollection,
) -> Result<(), BuilderAppendError> {
    values
        .try_reserve(additional)
        .map_err(|_| BuilderAppendError::Reserve {
            collection,
            additional,
        })
}

fn checked_op_sizes(op: &OpKind, inputs: &[TensorType]) -> Result<(), BuilderAppendError> {
    match op {
        OpKind::Concat { axis } if inputs.first().is_some_and(|ty| *axis < ty.rank()) => {
            let mut total = 0usize;
            for ty in inputs {
                if *axis < ty.rank() {
                    total = checked_growth(BuilderCollection::Values, total, ty.shape[*axis])?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn staged_namespace(
    plan: &BuilderAppendPlan,
    value: ValueId,
    meta: &ValueMeta,
) -> BuilderValueNamespace {
    match meta.storage {
        Storage::Const | Storage::Computed(_) | Storage::State => BuilderValueNamespace::Constant,
        Storage::Slot(_) => BuilderValueNamespace::Slot,
        Storage::Device if plan.equations.iter().any(|eqn| eqn.out == value) => {
            BuilderValueNamespace::EquationResult
        }
        Storage::Device => BuilderValueNamespace::NamedValue,
    }
}

fn live_name_namespace(g: &Graph, name: &str) -> Option<BuilderValueNamespace> {
    let value = g
        .values
        .iter()
        .position(|meta| meta.name.as_deref() == Some(name))?;
    if g.consts.contains(&value) {
        Some(BuilderValueNamespace::Constant)
    } else if g.slots.iter().any(|&(id, _)| id == value) {
        Some(BuilderValueNamespace::Slot)
    } else if g.eqns.iter().any(|eqn| eqn.out == value) {
        Some(BuilderValueNamespace::EquationResult)
    } else if g.inputs.contains(&value) {
        Some(BuilderValueNamespace::Input)
    } else {
        Some(BuilderValueNamespace::NamedValue)
    }
}

fn validate_append_result(
    plan: &BuilderAppendPlan,
    live_value_count: usize,
    defined: &[bool],
) -> Result<PreparedAppendResult, BuilderAppendError> {
    let declared = plan.result.ok_or(BuilderAppendError::MissingResult)?;
    let value = declared.value();
    let Some(offset) = value.checked_sub(live_value_count) else {
        return Err(BuilderAppendError::InvalidResult { value });
    };
    let Some(meta) = plan.values.get(offset) else {
        return Err(BuilderAppendError::InvalidResult { value });
    };
    if !defined.get(offset).copied().unwrap_or(false) {
        return Err(BuilderAppendError::InvalidResult { value });
    }
    match (declared, meta.storage) {
        (DeclaredAppendResult::EquationOutput(_), Storage::Device) => {
            Ok(PreparedAppendResult::EquationOutput(value))
        }
        (
            DeclaredAppendResult::StagedInput(_),
            Storage::Const | Storage::Computed(_) | Storage::Slot(_) | Storage::State,
        ) => Ok(PreparedAppendResult::StagedInput(value)),
        _ => Err(BuilderAppendError::InvalidResult { value }),
    }
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            identity: Rc::new(()),
            g: RefCell::new(Graph::default()),
            generation: Cell::new(0),
            layer: Rc::new(Cell::new(None)),
            #[cfg(test)]
            reservation_failure: Cell::new(None),
        }
    }

    /// Tag every equation emitted while the returned guard is alive with decoder layer `layer`.
    ///
    /// A model tracer calls this once around its layer loop (or once per iteration), never per op:
    ///
    /// ```ignore
    /// for layer in 0..layers {
    ///     let _layer = b.layer_scope(layer);
    ///     x = trace_block(&b, x, layer)?;
    /// }
    /// ```
    ///
    /// The tag is what `poot-graph-plan`'s `split_stages_after_layers` maps a placement's
    /// `after_layer` cut onto. It covers equations only: constants, slots and state binders created
    /// inside the scope stay untagged, because a layer cut never falls between a binder and its
    /// readers.
    pub fn layer_scope(&self, layer: usize) -> LayerScope {
        let previous = self.layer.get();
        self.layer.set(Some(LayerIndex(layer)));
        LayerScope {
            layer: Rc::clone(&self.layer),
            previous,
        }
    }

    /// The layer tag the next emitted equation will carry.
    fn current_layer(&self) -> Option<LayerIndex> {
        self.layer.get()
    }

    /// Finalize: set the primary output and return the graph.
    pub fn finish(self, out: Traced) -> Graph {
        let mut g = self.g.into_inner();
        g.output = out.id;
        g
    }

    /// Finalize with carried state: set the primary output and the `(state_in, state_out)` alias pairs
    /// (the KV cache). `state` is `(the State input binder, the updated value)` per cache buffer.
    pub fn finish_with_state(self, out: Traced, state: &[(Traced, Traced)]) -> Graph {
        let mut g = self.g.into_inner();
        g.output = out.id;
        g.state = state.iter().map(|(si, so)| (si.id, so.id)).collect();
        g
    }

    /// Reopen a finished graph for one more append (card 551a): `finish`/
    /// `finish_with_state` consume `self`, so extending an already-finished graph (card 551b appends a
    /// sampler suffix after the Runner's per-step graph) needs a fresh `Builder` over `g`'s own live
    /// values/eqns. The returned builder is a new identity (never aliasing whatever builder produced
    /// `g`), so [`Builder::append_plan`]/[`Builder::preflight_append`] work unchanged on it; `out` and
    /// `state` rehydrate `g`'s old output and state pairs as [`Traced`] so new equations can read them
    /// and a later `finish`/`finish_with_state` call can re-close over the (possibly unchanged) state
    /// pairs. `Builder`'s live storage is always a plain `Graph` (`NoValidations`), and `g`'s type here
    /// is that same plain `Graph`, not the generic `Graph<V>`: a validated `Graph<ValidationOutputs>`
    /// has no implicit conversion down to it (see [`Graph`]'s own doc), so it cannot be passed to
    /// `resume` at all - a caller holding one must consume its validations first.
    pub fn resume(g: Graph) -> Resumed {
        let out = Traced { id: g.output };
        let state = g
            .state
            .iter()
            .map(|&(si, so)| (Traced { id: si }, Traced { id: so }))
            .collect();
        Resumed {
            builder: Builder {
                identity: Rc::new(()),
                g: RefCell::new(g),
                generation: Cell::new(0),
                layer: Rc::new(Cell::new(None)),
                #[cfg(test)]
                reservation_failure: Cell::new(None),
            },
            out,
            state,
        }
    }

    pub fn aval(&self, t: Traced) -> TensorType {
        self.g.borrow().values[t.id].aval.clone()
    }

    /// The mutation generation used to invalidate prepared transactional appends.
    pub fn generation(&self) -> u64 {
        self.generation.get()
    }

    /// Start a staged append against the builder's current immutable generation.
    pub fn append_plan(&self, packed_rows: usize) -> BuilderAppendPlan {
        let g = self.g.borrow();
        BuilderAppendPlan {
            builder_identity: Rc::clone(&self.identity),
            generation: self.generation.get(),
            base_value_count: g.values.len(),
            base_types: g.values.iter().map(|meta| meta.aval.clone()).collect(),
            values: Vec::new(),
            equations: Vec::new(),
            result: None,
            packed_rows,
            checked_name_bytes: 0,
            layer: self.layer.get(),
        }
    }

    /// Validate an entire staged fragment without changing the live graph.
    pub fn preflight_append(
        &self,
        plan: BuilderAppendPlan,
    ) -> Result<PreparedBuilderAppend, BuilderAppendError> {
        if !Rc::ptr_eq(&plan.builder_identity, &self.identity) {
            return Err(BuilderAppendError::ForeignBuilder);
        }
        let g = self.g.borrow();
        let actual_generation = self.generation.get();
        if plan.generation != actual_generation || plan.base_value_count != g.values.len() {
            return Err(BuilderAppendError::StalePlan {
                expected: plan.generation,
                actual: actual_generation,
            });
        }

        checked_growth(BuilderCollection::Values, g.values.len(), plan.values.len())?;
        checked_growth(
            BuilderCollection::Equations,
            g.eqns.len(),
            plan.equations.len(),
        )?;

        let mut staged_inputs = 0usize;
        let mut staged_constants = 0usize;
        let mut staged_slots = 0usize;
        let mut staged_names: HashMap<&str, BuilderValueNamespace> = HashMap::new();
        for (offset, meta) in plan.values.iter().enumerate() {
            let value = plan.base_value_count + offset;
            let namespace = staged_namespace(&plan, value, meta);
            if meta.storage != Storage::Device {
                staged_inputs =
                    staged_inputs
                        .checked_add(1)
                        .ok_or(BuilderAppendError::SizeOverflow {
                            collection: BuilderCollection::Inputs,
                            current: staged_inputs,
                            additional: 1,
                        })?;
                match meta.storage {
                    Storage::Const | Storage::Computed(_) | Storage::State => {
                        staged_constants =
                            checked_growth(BuilderCollection::Constants, staged_constants, 1)?;
                    }
                    Storage::Slot(_) => {
                        staged_slots = checked_growth(BuilderCollection::Slots, staged_slots, 1)?;
                    }
                    Storage::Device => {}
                }
            }
            if let Some(name) = meta.name.as_deref() {
                if name.is_empty() {
                    return Err(BuilderAppendError::EmptyName { namespace });
                }
                if let Some(existing) = staged_names.insert(name, namespace) {
                    return Err(BuilderAppendError::NameCollision {
                        name: name.to_string(),
                        requested: namespace,
                        existing,
                    });
                }
                if let Some(existing) = live_name_namespace(&g, name) {
                    return Err(BuilderAppendError::NameCollision {
                        name: name.to_string(),
                        requested: namespace,
                        existing,
                    });
                }
            }
        }
        checked_growth(BuilderCollection::Inputs, g.inputs.len(), staged_inputs)?;
        checked_growth(
            BuilderCollection::Constants,
            g.consts.len(),
            staged_constants,
        )?;
        checked_growth(BuilderCollection::Slots, g.slots.len(), staged_slots)?;

        let total_values = g.values.len() + plan.values.len();
        let mut defined = vec![false; plan.values.len()];
        for (offset, meta) in plan.values.iter().enumerate() {
            if meta.storage != Storage::Device {
                defined[offset] = true;
            }
        }
        for (equation, eqn) in plan.equations.iter().enumerate() {
            let operation = eqn.op.name();
            let mut in_types = Vec::with_capacity(eqn.inputs.len());
            for operand in &eqn.inputs {
                match operand {
                    Operand::Lit(value) => in_types.push(value.ty()),
                    Operand::Value(value) if *value < g.values.len() => {
                        in_types.push(g.values[*value].aval.clone());
                    }
                    Operand::Value(value) if *value >= total_values => {
                        return Err(BuilderAppendError::InvalidOperand {
                            equation,
                            operation,
                            value: *value,
                        });
                    }
                    Operand::Value(value) => {
                        let offset = *value - g.values.len();
                        if !defined[offset] {
                            return Err(BuilderAppendError::OperandNotDefined {
                                equation,
                                operation,
                                value: *value,
                            });
                        }
                        in_types.push(plan.values[offset].aval.clone());
                    }
                }
            }
            if eqn.out < g.values.len() || eqn.out >= total_values {
                return Err(BuilderAppendError::InvalidOperand {
                    equation,
                    operation,
                    value: eqn.out,
                });
            }
            let output_offset = eqn.out - g.values.len();
            if defined[output_offset] {
                return Err(BuilderAppendError::ValueRedefined {
                    equation,
                    operation,
                    value: eqn.out,
                });
            }
            let output = &plan.values[output_offset];
            if output.storage != Storage::Device {
                return Err(BuilderAppendError::EquationResultStorage {
                    value: eqn.out,
                    storage: output.storage,
                });
            }
            checked_op_sizes(&eqn.op, &in_types)?;
            if let [Operand::Value(_), Operand::Value(_)] = eqn.inputs.as_slice()
                && matches!(eqn.op, OpKind::Binary(_))
            {
                crate::op::check_binary_value_dtypes(&in_types[0], &in_types[1]).map_err(
                    |source| BuilderAppendError::Inference {
                        equation,
                        operation: operation.clone(),
                        source,
                    },
                )?;
            }
            let inferred =
                eqn.op
                    .infer(&in_types)
                    .map_err(|source| BuilderAppendError::Inference {
                        equation,
                        operation: operation.clone(),
                        source,
                    })?;
            if inferred != output.aval {
                return Err(BuilderAppendError::OutputTypeMismatch {
                    equation,
                    operation,
                    stored: output.aval.clone(),
                    inferred,
                });
            }
            defined[output_offset] = true;
        }
        for (offset, meta) in plan.values.iter().enumerate() {
            if meta.storage == Storage::Device && !defined[offset] {
                return Err(BuilderAppendError::MissingDefinition {
                    value: g.values.len() + offset,
                });
            }
        }
        let result = validate_append_result(&plan, g.values.len(), &defined)?;

        let accounting = BuilderAppendAccounting {
            staged_values: plan.values.len(),
            staged_equations: plan.equations.len(),
            packed_rows: plan.packed_rows,
            checked_name_bytes: plan.checked_name_bytes,
        };
        Ok(PreparedBuilderAppend {
            builder_identity: Rc::clone(&plan.builder_identity),
            generation: actual_generation,
            base_value_count: g.values.len(),
            accounting,
            staged_inputs,
            staged_constants,
            staged_slots,
            result,
            plan: Some(plan),
        })
    }

    /// Reserve all live collections and publish a preflighted fragment as one generation.
    pub fn commit_append(
        &self,
        prepared: &mut PreparedBuilderAppend,
    ) -> Result<ValueId, BuilderAppendError> {
        if !Rc::ptr_eq(&prepared.builder_identity, &self.identity) {
            return Err(BuilderAppendError::ForeignBuilder);
        }
        if prepared.plan.is_none() {
            return Err(BuilderAppendError::ConsumedPlan);
        }
        let actual_generation = self.generation.get();
        if prepared.generation != actual_generation {
            return Err(BuilderAppendError::StalePlan {
                expected: prepared.generation,
                actual: actual_generation,
            });
        }
        let next_generation = actual_generation
            .checked_add(1)
            .ok_or(BuilderAppendError::GenerationOverflow)?;
        let mut g = self.g.borrow_mut();
        if prepared.base_value_count != g.values.len() {
            return Err(BuilderAppendError::StalePlan {
                expected: prepared.generation,
                actual: actual_generation,
            });
        }
        self.reserve_append_collection(
            &mut g.values,
            prepared.accounting.staged_values,
            BuilderCollection::Values,
        )?;
        self.reserve_append_collection(
            &mut g.inputs,
            prepared.staged_inputs,
            BuilderCollection::Inputs,
        )?;
        self.reserve_append_collection(
            &mut g.consts,
            prepared.staged_constants,
            BuilderCollection::Constants,
        )?;
        self.reserve_append_collection(
            &mut g.slots,
            prepared.staged_slots,
            BuilderCollection::Slots,
        )?;
        self.reserve_append_collection(
            &mut g.eqns,
            prepared.accounting.staged_equations,
            BuilderCollection::Equations,
        )?;

        // Presence was checked before reservation; keep the invariant typed if this code is rearranged, and leave the
        // plan reusable when an earlier reservation fails.
        let plan = prepared
            .plan
            .take()
            .ok_or(BuilderAppendError::ConsumedPlan)?;
        let result = prepared.result.value();
        for (offset, meta) in plan.values.into_iter().enumerate() {
            let id = prepared.base_value_count + offset;
            match meta.storage {
                Storage::Const | Storage::Computed(_) | Storage::State => {
                    g.inputs.push(id);
                    g.consts.push(id);
                }
                Storage::Slot(slot) => {
                    g.inputs.push(id);
                    g.slots.push((id, slot));
                }
                Storage::Device => {}
            }
            g.values.push(meta);
        }
        g.eqns.extend(plan.equations);
        self.generation.set(next_generation);
        Ok(result)
    }

    fn reserve_append_collection<T>(
        &self,
        values: &mut Vec<T>,
        additional: usize,
        collection: BuilderCollection,
    ) -> Result<(), BuilderAppendError> {
        #[cfg(test)]
        if self.reservation_failure.get() == Some(collection) {
            self.reservation_failure.set(None);
            return Err(BuilderAppendError::Reserve {
                collection,
                additional,
            });
        }
        reserve(values, additional, collection)
    }

    fn push_value(
        &self,
        aval: TensorType,
        storage: Storage,
        name: Option<String>,
        key: Option<SlotKey>,
        state_role: Option<StateRole>,
    ) -> ValueId {
        let mut g = self.g.borrow_mut();
        let id = g.values.len();
        g.values.push(ValueMeta {
            aval,
            storage,
            name,
            key,
            state_role,
        });
        self.generation.set(
            self.generation
                .get()
                .checked_add(1)
                .expect("Builder generation exhausted"),
        );
        id
    }

    // input binders

    /// A constant input (weight / table), bound at graph close. Not per-token.
    pub fn constant(&self, name: &str, aval: TensorType) -> Traced {
        let id = self.push_value(aval, Storage::Const, Some(name.to_string()), None, None);
        {
            let mut g = self.g.borrow_mut();
            g.consts.push(id);
            g.inputs.push(id);
        }
        Traced { id }
    }

    /// A constant the binder materializes from its own payload (a derived table such as RoPE's):
    /// no name, no weight. Declared F32 in [`ComputedConst::shape`].
    pub fn computed(&self, computed: ComputedConst) -> Traced {
        let aval = TensorType::f32(computed.shape());
        let id = self.push_value(aval, Storage::Computed(computed), None, None, None);
        {
            let mut g = self.g.borrow_mut();
            g.consts.push(id);
            g.inputs.push(id);
        }
        Traced { id }
    }

    /// A per-token-varying input (token / pos / seq_len), routed through a stable device buffer.
    pub fn slot(&self, slot: Slot, aval: TensorType) -> Traced {
        let key = SlotKey::new(slot, None);
        let name = key.to_string();
        let id = self.push_value(aval, Storage::Slot(slot), Some(name), Some(key), None);
        {
            let mut g = self.g.borrow_mut();
            g.slots.push((id, slot));
            g.inputs.push(id);
        }
        Traced { id }
    }

    /// A runtime input whose name carries a caller-supplied disambiguator, so one graph can hold several inputs of
    /// the same [`Slot`] kind with different per-call content (spec 266: one expert-pool assignment table per MoE
    /// layer; spec 149: a named standalone activation). [`Builder::slot`] names every occurrence of a kind
    /// identically ([`Slot::key_label`], R466-013: a hand-written label, never `Debug`), which model-engine binders
    /// rely on to resolve a whole `Slot` kind from one caller-supplied buffer (see [`Slot::GdnSlotMap`]). A named
    /// input has no such implicit buffer. The name is `"{kind}.{tag}"` (built by [`crate::graph::SlotKey`]), and its
    /// binder must dispatch on [`crate::graph::ValueMeta::name`] as the `Const` path does, not on the kind alone.
    ///
    /// `tag` must be non-empty (an empty tag would stage the untagged label plus a bare trailing
    /// `.`, indistinguishable in spelling from a typo) and unique within the graph for a given kind;
    /// both are asserted (tracing is host-side and one-shot), the same rejection
    /// [`Builder::exact_i32_slot`] gives a caller-supplied empty tag.
    pub fn slot_named(&self, slot: Slot, tag: &str, aval: TensorType) -> Traced {
        assert!(
            !tag.is_empty(),
            "Builder::slot_named: an empty tag for slot {slot:?} is not a valid disambiguator - \
             pass a tag or use Builder::slot for the untagged occurrence"
        );
        let key = SlotKey::new(slot, Some(tag));
        let name = key.to_string();
        {
            let g = self.g.borrow();
            assert!(
                !g.values
                    .iter()
                    .any(|v| v.storage == Storage::Slot(slot) && v.name.as_deref() == Some(&name)),
                "Builder::slot_named: duplicate tag {tag:?} for slot {slot:?} - a per-name binder would \
                 bind both ids from one buffer, the collision slot_named exists to prevent"
            );
        }
        let id = self.push_value(aval, Storage::Slot(slot), Some(name), Some(key), None);
        {
            let mut g = self.g.borrow_mut();
            g.slots.push((id, slot));
            g.inputs.push(id);
        }
        Traced { id }
    }

    /// A persistent state input (a KV-cache buffer), carried across decode steps. A stable input binder like a
    /// constant, but mutable: overwritten each step by the paired `state_out` passed to [`Builder::finish_with_state`].
    /// Registered as a const binder for buffer binding but storage-tagged `State`. `role` declares how the
    /// executor's per-step overwrite behaves: [`StateRole::Recurrent`] folds every token in;
    /// [`StateRole::Positional`] writes only at the live position along an axis.
    ///
    /// The role is a required third argument, not a default (SC-002):
    ///
    /// ```compile_fail,E0061
    /// use poot_graph_ir::{Builder, TensorType};
    ///
    /// let b = Builder::new();
    /// let _cache = b.state_input("cache", TensorType::f32([1]));
    /// ```
    pub fn state_input(&self, name: &str, aval: TensorType, role: StateRole) -> Traced {
        let id = self.push_value(
            aval,
            Storage::State,
            Some(name.to_string()),
            None,
            Some(role),
        );
        {
            let mut g = self.g.borrow_mut();
            g.consts.push(id);
            g.inputs.push(id);
        }
        Traced { id }
    }

    // primitive emitters
    //
    // Each infers the output type host-side and appends an eqn. An infer failure is a trace-time programmer error
    // (incompatible shapes), surfaced as a panic with context like `abstract_eval` in JAX; the typed `ShapeError`
    // is what `OpKind::infer` and its tests return.

    fn emit(&self, op: OpKind, operands: Vec<Operand>, in_types: &[TensorType]) -> Traced {
        let aval = op
            .infer(in_types)
            .unwrap_or_else(|e| panic!("trace-time shape error in {}: {e}", op.name()));
        let out = self.push_value(aval, Storage::Device, None, None, None);
        self.g.borrow_mut().eqns.push(Eqn {
            op,
            inputs: operands,
            out,
            layer: self.current_layer(),
        });
        Traced { id: out }
    }

    pub fn unary(&self, op: crate::op::UnOp, x: Traced) -> Traced {
        let t = self.aval(x);
        self.emit(OpKind::Unary(op), vec![Operand::Value(x.id)], &[t])
    }

    pub fn binary(&self, op: crate::op::BinOp, a: Traced, b: Traced) -> Traced {
        let (ta, tb) = (self.aval(a), self.aval(b));
        // OpKind::Binary::infer takes its output dtype from `ins[0]` alone and cannot distinguish this two-value call
        // from `binary_scalar` (where a differing literal dtype is legitimate, baked as an immediate). Here both
        // operands are real buffers read by the planner's broadcast-value kernel at one element type (`fty(odt)`); a
        // dtype mismatch would reinterpret the other operand's bytes at the wrong width, so check it here (R466-012,
        // the one shared rule: `crate::op::check_binary_value_dtypes`).
        if let Err(e) = crate::op::check_binary_value_dtypes(&ta, &tb) {
            panic!("trace-time dtype error in Binary({op:?}): {e}");
        }
        self.emit(
            OpKind::Binary(op),
            vec![Operand::Value(a.id), Operand::Value(b.id)],
            &[ta, tb],
        )
    }

    /// Pack int8 codes into i32 words, 4 per word along the last axis (spec 048). Output is `I32` with last dim
    /// `ceil(L/4)`.
    pub fn pack_i8(&self, x: Traced) -> Traced {
        let tx = self.aval(x);
        self.emit(OpKind::PackI8, vec![Operand::Value(x.id)], &[tx])
    }

    /// Unpack `len` int8 codes from i32 words (inverse of [`Builder::pack_i8`]). Output is `F32` with last dim `len`.
    pub fn unpack_i8(&self, codes: Traced, len: usize) -> Traced {
        let tc = self.aval(codes);
        self.emit(
            OpKind::UnpackI8 { len },
            vec![Operand::Value(codes.id)],
            &[tc],
        )
    }

    /// Binary op with an inline scalar literal as the second operand (broadcasts as a 0-d value).
    pub fn binary_scalar(&self, op: crate::op::BinOp, a: Traced, s: Scalar) -> Traced {
        let ta = self.aval(a);
        let ts = s.ty();
        self.emit(
            OpKind::Binary(op),
            vec![Operand::Value(a.id), Operand::Lit(s)],
            &[ta, ts],
        )
    }

    /// I32 `Select(cond, if_true, if_false)` with wrapping `if_false + cond * (if_true - if_false)`.
    pub fn select(&self, cond: Traced, if_true: Traced, if_false: Traced) -> Traced {
        let (tc, tt, tf) = (self.aval(cond), self.aval(if_true), self.aval(if_false));
        self.emit(
            OpKind::Select,
            vec![
                Operand::Value(cond.id),
                Operand::Value(if_true.id),
                Operand::Value(if_false.id),
            ],
            &[tc, tt, tf],
        )
    }

    pub fn reduce(&self, op: crate::op::RedOp, x: Traced, axis: usize, keepdim: bool) -> Traced {
        let t = self.aval(x);
        self.emit(
            OpKind::Reduce { op, axis, keepdim },
            vec![Operand::Value(x.id)],
            &[t],
        )
    }

    pub fn broadcast(&self, x: Traced, shape: Vec<usize>) -> Traced {
        let t = self.aval(x);
        self.emit(
            OpKind::Broadcast { shape },
            vec![Operand::Value(x.id)],
            &[t],
        )
    }

    /// The F32 range `[0, 1, .., len)`, shape `[len]` (R466-019): a computed range, not a bound constant.
    /// The dtype stays F32 until Card 558b switches indices and ranges to I32 end to end. The compiler
    /// folds it into a [`crate::Storage::Computed`] graph constant before planning.
    pub fn iota(&self, len: usize) -> Traced {
        self.emit(OpKind::Iota { len }, Vec::new(), &[])
    }

    pub fn reshape(&self, x: Traced, shape: Vec<usize>) -> Traced {
        let t = self.aval(x);
        self.emit(OpKind::Reshape { shape }, vec![Operand::Value(x.id)], &[t])
    }

    /// Cast `x` to storage dtype `to` (same shape). f32 -> bf16 rounds to bf16 precision; a widening or same-dtype
    /// cast is the identity. Feeds bf16 operands into the tensor-core matmul (card 045).
    pub fn cast(&self, x: Traced, to: crate::types::DType) -> Traced {
        let t = self.aval(x);
        self.emit(OpKind::Cast { to }, vec![Operand::Value(x.id)], &[t])
    }

    pub fn transpose(&self, x: Traced, perm: Vec<usize>) -> Traced {
        let t = self.aval(x);
        self.emit(OpKind::Transpose { perm }, vec![Operand::Value(x.id)], &[t])
    }

    pub fn slice(&self, x: Traced, axis: usize, start: usize, end: usize) -> Traced {
        let t = self.aval(x);
        self.emit(
            OpKind::Slice { axis, start, end },
            vec![Operand::Value(x.id)],
            &[t],
        )
    }

    pub fn concat(&self, axis: usize, parts: &[Traced]) -> Traced {
        let in_types: Vec<TensorType> = parts.iter().map(|p| self.aval(*p)).collect();
        let operands: Vec<Operand> = parts.iter().map(|p| Operand::Value(p.id)).collect();
        self.emit(OpKind::Concat { axis }, operands, &in_types)
    }

    /// Gather rows of `data` along `axis` by `index`. A scalar index drops the axis (embedding / RoPE
    /// row); a vector index `[L]` replaces it with `L` (a whole prompt's embeddings).
    pub fn gather(&self, data: Traced, axis: usize, index: Traced) -> Traced {
        let (td, ti) = (self.aval(data), self.aval(index));
        self.emit(
            OpKind::Gather { axis },
            vec![Operand::Value(data.id), Operand::Value(index.id)],
            &[td, ti],
        )
    }

    /// Batched argtop-k index extraction (spec 136): `rank[..,E] -> [..,k]` expert ids, `out[..,r]` = the index `i`
    /// with `rank[..,i] == r`. Generalizes `scatter(iota, rank)[0..k]` to leading dims and feeds `IndexedMatMul`'s
    /// per-row `idx` operand. The rank producer owns score ordering and tie-breaking; this op only inverts a
    /// permutation. See [`OpKind::ArgTopK`].
    pub fn arg_top_k(&self, rank: Traced, k: usize) -> Traced {
        let tr = self.aval(rank);
        self.emit(OpKind::ArgTopK { k }, vec![Operand::Value(rank.id)], &[tr])
    }

    /// Exact uniform noise from an integer seed (card 551a, R472-007): `seed` is any-shape I32; output
    /// is `seed`'s shape with one trailing `cols` axis appended. See [`OpKind::RandomUniform`] and
    /// [`crate::ops::sampling`].
    pub fn random_uniform(&self, seed: Traced, cols: usize) -> Traced {
        let ts = self.aval(seed);
        self.emit(
            OpKind::RandomUniform { cols },
            vec![Operand::Value(seed.id)],
            &[ts],
        )
    }

    /// Pick one token per row (card 551a, R472-007, R-551a-2): `rule` fixes the operand list (see
    /// [`SampleRule`]) - `noise`/`params` are required for every rule but `Greedy`, `top_k` only for the
    /// two `TopK` rules. A caller that passes the wrong combination for `rule` panics at trace time via
    /// [`OpKind::infer`], the same contract every other primitive emitter gives. See
    /// [`crate::ops::sampling::sample_head`] for the composed suffix (noise generation included).
    pub fn sample_token(
        &self,
        rule: SampleRule,
        logits: Traced,
        noise: Option<Traced>,
        params: Option<Traced>,
        top_k: Option<Traced>,
    ) -> Traced {
        let mut operands = vec![Operand::Value(logits.id)];
        let mut types = vec![self.aval(logits)];
        for t in [noise, params, top_k].into_iter().flatten() {
            operands.push(Operand::Value(t.id));
            types.push(self.aval(t));
        }
        self.emit(OpKind::SampleToken { rule }, operands, &types)
    }

    /// Scatter rows of `src` along axis 0 by a `[N]` permutation `index` (inverse of axis-0 gather):
    /// `out[index[j], ..] = src[j, ..]`, same shape as `src`. See [`OpKind::Scatter`].
    pub fn scatter(&self, src: Traced, index: Traced) -> Traced {
        let (ts, ti) = (self.aval(src), self.aval(index));
        self.emit(
            OpKind::Scatter { axis: 0 },
            vec![Operand::Value(src.id), Operand::Value(index.id)],
            &[ts, ti],
        )
    }

    /// Scatter-update (card 059): `out[p] = inv[p] >= 0 ? src[inv[p]] : base[p]`. `base [POOL,..rest]`,
    /// `src [N,..rest]`, `[POOL]` inverse map `inv` (the src row that writes slot `p`, or `-1`). Writes the N mapped
    /// rows and passes the rest through: the scattered analog of `dynamic_update_slice`. See
    /// [`OpKind::ScatterUpdate`].
    pub fn scatter_update(&self, base: Traced, src: Traced, inv: Traced) -> Traced {
        let (tb, ts, ti) = (self.aval(base), self.aval(src), self.aval(inv));
        self.emit(
            OpKind::ScatterUpdate,
            vec![
                Operand::Value(base.id),
                Operand::Value(src.id),
                Operand::Value(inv.id),
            ],
            &[tb, ts, ti],
        )
    }

    /// Gather with a scalar index (the common decode case): drops `axis`.
    pub fn gather_scalar(&self, data: Traced, axis: usize, index: Traced) -> Traced {
        self.gather(data, axis, index)
    }

    /// Write `update` into a copy of `operand` at the static index `index` along `axis` (the KV-cache slot write).
    /// The index is an inline literal, so the decode graph is pos-specialized.
    pub fn dynamic_update_slice(
        &self,
        operand: Traced,
        update: Traced,
        index: usize,
        axis: usize,
    ) -> Traced {
        let (to, tu) = (self.aval(operand), self.aval(update));
        let idx = Scalar::I32(index as i32);
        self.emit(
            OpKind::DynamicUpdateSlice { axis },
            vec![
                Operand::Value(operand.id),
                Operand::Value(update.id),
                Operand::Lit(idx),
            ],
            &[to, tu, idx.ty()],
        )
    }

    /// Write `update` into a copy of `operand` at a runtime `index` (a scalar value, e.g. the `Pos` slot) along
    /// `axis` (G3d). Like [`Builder::dynamic_update_slice`] but the index is a graph value read at run time, so the
    /// decode graph is constant across positions and one capture replays every token. `index` must be an
    /// empty-shape scalar (`[]`); `infer` rejects `[1]` (card 204: reshape a `[1]` row map to `[]` first).
    pub fn dynamic_update_slice_dyn(
        &self,
        operand: Traced,
        update: Traced,
        index: Traced,
        axis: usize,
    ) -> Traced {
        let (to, tu, ti) = (self.aval(operand), self.aval(update), self.aval(index));
        self.emit(
            OpKind::DynamicUpdateSlice { axis },
            vec![
                Operand::Value(operand.id),
                Operand::Value(update.id),
                Operand::Value(index.id),
            ],
            &[to, tu, ti],
        )
    }

    pub fn matmul(&self, a: Traced, b: Traced) -> Traced {
        let (ta, tb) = (self.aval(a), self.aval(b));
        self.emit(
            OpKind::MatMul,
            vec![Operand::Value(a.id), Operand::Value(b.id)],
            &[ta, tb],
        )
    }

    /// Emit the backend-neutral packed-dequant semantic primitive. The compiler-only `PackedContraction`,
    /// `DenseContraction` and `DenseRowGather` have no builder method: they exist only so a recognizer can rewrite a
    /// composition the tracer already expressed.
    ///
    /// `sources` are the carrier constants in [`PackedWeight::sources`] role order: one for a block
    /// format (GGUF), one per planar operand otherwise (AWQ three, GPTQ four).
    pub fn packed_dequant(&self, sources: &[Traced], descriptor: PackedWeight) -> Traced {
        let types: Vec<TensorType> = sources.iter().map(|&source| self.aval(source)).collect();
        self.emit(
            OpKind::PackedDequant { descriptor },
            sources
                .iter()
                .map(|source| Operand::Value(source.id))
                .collect(),
            &types,
        )
    }

    /// Indexed matmul (card 088): `out[m,n] = sum_k x[m,k] * W[idx[m],k,n]`. `x [M,K]`, stacked weight `w [E,K,N]`,
    /// per-row expert ids `idx [M]` (f32) -> `[M,N]`. Each row contracts against its own expert `W[idx[m]]`: a
    /// gather-free MoE GEMM. See [`OpKind::IndexedMatMul`].
    pub fn indexed_matmul(&self, x: Traced, w: Traced, idx: Traced) -> Traced {
        let (tx, tw, tidx) = (self.aval(x), self.aval(w), self.aval(idx));
        self.emit(
            OpKind::IndexedMatMul,
            vec![
                Operand::Value(x.id),
                Operand::Value(w.id),
                Operand::Value(idx.id),
            ],
            &[tx, tw, tidx],
        )
    }

    /// AllReduce collective (card 049a, tensor-parallel): `op`-reduces a same-shaped partial tensor across all ranks
    /// and returns the full result in the same shape. `axis` is the shard axis (metadata for the multi-device
    /// ring-allreduce lowering, card 049b).
    ///
    /// In a single-rank (world_size=1) graph the node evaluates as the identity. The multi-rank test harness sums
    /// per-rank evals outside the graph to simulate the collective.
    pub fn all_reduce(&self, x: Traced, op: crate::op::RedOp, axis: usize) -> Traced {
        let t = self.aval(x);
        self.emit(
            OpKind::AllReduce { op, axis },
            vec![Operand::Value(x.id)],
            &[t],
        )
    }

    /// AllGather collective (card 049a, column-parallel matmul): each rank holds a slice of the output along `axis`;
    /// the collective concatenates slices across ranks, growing `axis` by world_size.
    ///
    /// `axis` is the gather axis (the output N/feature axis in column-parallel matmul). In a single-rank
    /// (world_size=1) graph the node evaluates as the identity. The multi-rank lowering (card 049b) concatenates
    /// per-rank output slices; the test harness simulates this by concatenating per-rank evals outside the graph.
    pub fn all_gather(&self, x: Traced, axis: usize) -> Traced {
        let t = self.aval(x);
        self.emit(OpKind::AllGather { axis }, vec![Operand::Value(x.id)], &[t])
    }

    // convenience scalar makers (kept here so models do not import types directly).
    pub fn f32(v: f32) -> Scalar {
        Scalar::F32(v)
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ShapeError;
    use crate::op::BinOp;

    #[derive(Debug, PartialEq, Eq)]
    struct State {
        values: usize,
        inputs: usize,
        constants: usize,
        slots: usize,
        equations: usize,
        output: ValueId,
        state: usize,
        generation: u64,
        names: Vec<Option<String>>,
        types: Vec<TensorType>,
        storage: Vec<Storage>,
    }

    fn state(builder: &Builder) -> State {
        let graph = builder.g.borrow();
        State {
            values: graph.values.len(),
            inputs: graph.inputs.len(),
            constants: graph.consts.len(),
            slots: graph.slots.len(),
            equations: graph.eqns.len(),
            output: graph.output,
            state: graph.state.len(),
            generation: builder.generation(),
            names: graph.values.iter().map(|meta| meta.name.clone()).collect(),
            types: graph.values.iter().map(|meta| meta.aval.clone()).collect(),
            storage: graph.values.iter().map(|meta| meta.storage).collect(),
        }
    }

    fn valid_plan(
        builder: &Builder,
        input: Traced,
    ) -> Result<BuilderAppendPlan, BuilderAppendError> {
        let mut plan = builder.append_plan(0);
        let output = plan.equation(
            OpKind::Cast { to: DType::F32 },
            vec![Operand::Value(input.id)],
        )?;
        plan.declare_result(output)?;
        Ok(plan)
    }

    #[test]
    fn transactional_input_constant_can_be_result() -> Result<(), BuilderAppendError> {
        let builder = Builder::new();
        let mut plan = builder.append_plan(0);
        let result = plan.input_result(
            "shared.table",
            TensorType::new(vec![2, 3], DType::BF16),
            Storage::Const,
        )?;

        let mut prepared = builder.preflight_append(plan)?;
        assert_eq!(
            prepared.accounting(),
            BuilderAppendAccounting {
                staged_values: 1,
                staged_equations: 0,
                packed_rows: 0,
                checked_name_bytes: 12,
            }
        );
        assert_eq!(builder.commit_append(&mut prepared)?, 0);

        let graph = builder.finish(result);
        assert_eq!(graph.output, 0);
        assert_eq!(graph.inputs, vec![0]);
        assert_eq!(graph.consts, vec![0]);
        assert!(graph.slots.is_empty());
        assert!(graph.eqns.is_empty());
        assert_eq!(graph.values[0].storage, Storage::Const);
        assert_eq!(
            graph.values[0].aval,
            TensorType::new(vec![2, 3], DType::BF16)
        );
        assert!(graph.validate().is_ok());
        Ok(())
    }

    #[test]
    fn transactional_input_slot_can_be_result() -> Result<(), BuilderAppendError> {
        let builder = Builder::new();
        let mut plan = builder.append_plan(0);
        let result = plan.input_result(
            "slot_map.route",
            TensorType::new(vec![4], DType::I8),
            Storage::Slot(Slot::SlotMap),
        )?;

        let mut prepared = builder.preflight_append(plan)?;
        assert_eq!(builder.commit_append(&mut prepared)?, 0);

        let graph = builder.finish(result);
        assert_eq!(graph.output, 0);
        assert_eq!(graph.inputs, vec![0]);
        assert!(graph.consts.is_empty());
        assert_eq!(graph.slots, vec![(0, Slot::SlotMap)]);
        assert!(graph.eqns.is_empty());
        assert_eq!(graph.values[0].storage, Storage::Slot(Slot::SlotMap));
        assert_eq!(graph.values[0].aval, TensorType::new(vec![4], DType::I8));
        assert!(graph.validate().is_ok());
        Ok(())
    }

    #[test]
    fn transactional_input_result_rejections_are_atomic() -> Result<(), BuilderAppendError> {
        {
            let builder = Builder::new();
            let existing = builder.constant("existing", TensorType::f32(vec![1]));
            let mut plan = builder.append_plan(0);
            plan.declare_result(existing)?;
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderAppendError::InvalidResult { value }) if value == existing.id
            ));
            assert_eq!(state(&builder), before);
        }

        {
            let builder = Builder::new();
            let mut plan = builder.append_plan(0);
            plan.input("staged", TensorType::f32(vec![1]), Storage::Const)?;
            plan.declare_result(Traced { id: usize::MAX })?;
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderAppendError::InvalidResult { value }) if value == usize::MAX
            ));
            assert_eq!(state(&builder), before);
        }

        {
            let builder = Builder::new();
            let input = builder.constant("input", TensorType::f32(vec![1]));
            let mut plan = builder.append_plan(0);
            let output = plan.equation(
                OpKind::Cast { to: DType::F32 },
                vec![Operand::Value(input.id)],
            )?;
            plan.declare_result(output)?;
            let staged_values = plan.values.len();
            let checked_name_bytes = plan.checked_name_bytes;
            assert!(matches!(
                plan.input_result("late", TensorType::f32(vec![1]), Storage::Const),
                Err(BuilderAppendError::ResultAlreadyDeclared { value }) if value == output.id
            ));
            assert_eq!(plan.values.len(), staged_values);
            assert_eq!(plan.checked_name_bytes, checked_name_bytes);
            assert!(matches!(
                plan.result,
                Some(DeclaredAppendResult::EquationOutput(value)) if value == output.id
            ));
        }

        {
            let foreign = Builder::new();
            let mut foreign_plan = foreign.append_plan(0);
            let foreign_input =
                foreign_plan.input("foreign", TensorType::f32(vec![1]), Storage::Const)?;

            let builder = Builder::new();
            let mut plan = builder.append_plan(0);
            let local_input = plan.input("local", TensorType::f32(vec![1]), Storage::Const)?;
            assert_eq!(foreign_input.id, local_input.id);
            plan.declare_result(foreign_input)?;
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderAppendError::InvalidResult { value }) if value == local_input.id
            ));
            assert_eq!(state(&builder), before);
        }

        {
            let builder = Builder::new();
            let mut plan = builder.append_plan(0);
            let local_input = plan.input("local", TensorType::f32(vec![1]), Storage::Const)?;
            plan.declare_result(Traced { id: local_input.id })?;
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderAppendError::InvalidResult { value }) if value == local_input.id
            ));
            assert_eq!(state(&builder), before);
        }

        {
            let builder = Builder::new();
            let mut plan = builder.append_plan(0);
            let value = plan.next_value_id()?;
            plan.values.push(ValueMeta {
                aval: TensorType::f32(vec![1]),
                storage: Storage::Device,
                name: None,
                key: None,
                state_role: None,
            });
            plan.declare_result(Traced { id: value })?;
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderAppendError::MissingDefinition { value: actual }) if actual == value
            ));
            assert_eq!(state(&builder), before);
        }

        {
            let origin = Builder::new();
            let foreign = Builder::new();
            let mut plan = origin.append_plan(0);
            plan.input_result("origin", TensorType::f32(vec![1]), Storage::Const)?;
            let origin_before = state(&origin);
            let foreign_before = state(&foreign);
            assert!(matches!(
                foreign.preflight_append(plan),
                Err(BuilderAppendError::ForeignBuilder)
            ));
            assert_eq!(state(&origin), origin_before);
            assert_eq!(state(&foreign), foreign_before);

            let mut plan = origin.append_plan(0);
            plan.input_result("origin", TensorType::f32(vec![1]), Storage::Const)?;
            let mut prepared = origin.preflight_append(plan)?;
            assert_eq!(
                foreign.commit_append(&mut prepared),
                Err(BuilderAppendError::ForeignBuilder)
            );
            assert!(prepared.plan.is_some());
            assert_eq!(state(&origin), origin_before);
            assert_eq!(state(&foreign), foreign_before);
        }

        {
            let builder = Builder::new();
            let mut plan = builder.append_plan(0);
            let stale_result =
                plan.input_result("stale", TensorType::f32(vec![1]), Storage::Const)?;
            assert_eq!(stale_result.id, 0);
            builder.constant("intervening", TensorType::f32(vec![1]));
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderAppendError::StalePlan { .. })
            ));
            assert_eq!(state(&builder), before);

            let mut next = builder.append_plan(0);
            let next_result =
                next.input_result("next", TensorType::f32(vec![1]), Storage::Const)?;
            assert_eq!(next_result.id, 1);
            let mut prepared = builder.preflight_append(next)?;
            assert_eq!(builder.commit_append(&mut prepared)?, 1);
        }

        {
            let builder = Builder::new();
            builder.slot(Slot::Activation, TensorType::f32(vec![1]));
            let mut plan = builder.append_plan(0);
            plan.input_result(
                "activation",
                TensorType::f32(vec![1]),
                Storage::Slot(Slot::Activation),
            )?;
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderAppendError::NameCollision {
                    name,
                    requested: BuilderValueNamespace::Slot,
                    existing: BuilderValueNamespace::Slot,
                }) if name == "activation"
            ));
            assert_eq!(state(&builder), before);

            let mut next = builder.append_plan(0);
            let next_result = next.input_result(
                "activation.fresh",
                TensorType::f32(vec![1]),
                Storage::Slot(Slot::Activation),
            )?;
            assert_eq!(next_result.id, 1);
            let mut prepared = builder.preflight_append(next)?;
            assert_eq!(builder.commit_append(&mut prepared)?, 1);
        }

        Ok(())
    }

    #[test]
    fn transactional_input_result_reservation_is_atomic() -> Result<(), BuilderAppendError> {
        let cases: [(Storage, &[BuilderCollection]); 2] = [
            (
                Storage::Const,
                &[
                    BuilderCollection::Values,
                    BuilderCollection::Inputs,
                    BuilderCollection::Constants,
                ],
            ),
            (
                Storage::Slot(Slot::SlotMap),
                &[
                    BuilderCollection::Values,
                    BuilderCollection::Inputs,
                    BuilderCollection::Slots,
                ],
            ),
        ];

        for (storage, collections) in cases {
            for &collection in collections {
                let builder = Builder::new();
                let mut plan = builder.append_plan(0);
                let result = plan.input_result("result", TensorType::f32(vec![1]), storage)?;
                let mut prepared = builder.preflight_append(plan)?;
                let before = state(&builder);
                builder.reservation_failure.set(Some(collection));
                assert_eq!(
                    builder.commit_append(&mut prepared),
                    Err(BuilderAppendError::Reserve {
                        collection,
                        additional: 1,
                    })
                );
                assert_eq!(state(&builder), before);
                assert!(prepared.plan.is_some());
                assert_eq!(prepared.result.value(), result.id);
                assert_eq!(builder.commit_append(&mut prepared)?, result.id);
                assert_eq!(result.id, 0);
            }
        }
        Ok(())
    }

    fn inject_name(builder: &Builder, namespace: BuilderValueNamespace, name: &str) {
        let mut graph = builder.g.borrow_mut();
        let id = graph.values.len();
        let storage = match namespace {
            BuilderValueNamespace::Slot => Storage::Slot(Slot::Activation),
            BuilderValueNamespace::EquationResult | BuilderValueNamespace::NamedValue => {
                Storage::Device
            }
            BuilderValueNamespace::Input | BuilderValueNamespace::Constant => Storage::State,
        };
        graph.values.push(ValueMeta {
            aval: TensorType::f32(vec![1]),
            storage,
            name: Some(name.to_string()),
            key: None,
            state_role: (storage == Storage::State).then_some(StateRole::Recurrent),
        });
        match namespace {
            BuilderValueNamespace::Input => graph.inputs.push(id),
            BuilderValueNamespace::Constant => {
                graph.inputs.push(id);
                graph.consts.push(id);
            }
            BuilderValueNamespace::Slot => {
                graph.inputs.push(id);
                graph.slots.push((id, Slot::Activation));
            }
            BuilderValueNamespace::EquationResult => {
                let source = 0;
                graph.eqns.push(Eqn {
                    op: OpKind::Cast { to: DType::F32 },
                    inputs: vec![Operand::Value(source)],
                    out: id,
                    layer: None,
                });
            }
            BuilderValueNamespace::NamedValue => {}
        }
    }

    #[test]
    fn builder_append_preflight_checks_every_name_namespace() -> Result<(), BuilderAppendError> {
        for existing in [
            BuilderValueNamespace::Input,
            BuilderValueNamespace::Constant,
            BuilderValueNamespace::Slot,
            BuilderValueNamespace::EquationResult,
            BuilderValueNamespace::NamedValue,
        ] {
            let builder = Builder::new();
            builder.constant("source", TensorType::f32(vec![1]));
            inject_name(&builder, existing, "taken");
            let mut plan = builder.append_plan(0);
            plan.input("taken", TensorType::f32(vec![1]), Storage::Const)?;
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderAppendError::NameCollision {
                    name,
                    requested: BuilderValueNamespace::Constant,
                    existing: actual,
                }) if name == "taken" && actual == existing
            ));
        }

        let builder = Builder::new();
        let mut plan = builder.append_plan(0);
        plan.input("duplicate", TensorType::f32(vec![1]), Storage::Const)?;
        plan.input("duplicate", TensorType::f32(vec![1]), Storage::Const)?;
        assert!(matches!(
            builder.preflight_append(plan),
            Err(BuilderAppendError::NameCollision {
                name,
                requested: BuilderValueNamespace::Constant,
                existing: BuilderValueNamespace::Constant,
            }) if name == "duplicate"
        ));
        Ok(())
    }

    #[test]
    fn builder_preflight_rejects_typed_collisions() -> Result<(), BuilderAppendError> {
        builder_append_preflight_checks_every_name_namespace()
    }

    #[test]
    fn builder_append_preflight_errors_are_typed() -> Result<(), BuilderAppendError> {
        let builder = Builder::new();
        let input = builder.constant("input", TensorType::f32(vec![2, 3]));
        let mut plan = valid_plan(&builder, input)?;
        plan.equations[0].op = OpKind::Reshape { shape: vec![5] };
        assert!(matches!(
            builder.preflight_append(plan),
            Err(BuilderAppendError::Inference {
                equation: 0,
                operation,
                source: ShapeError::Reshape { .. },
            }) if operation == "reshape [5]"
        ));
        Ok(())
    }

    #[test]
    fn builder_append_rejects_foreign_builder_before_mutation() -> Result<(), BuilderAppendError> {
        let origin = Builder::new();
        let origin_input = origin.constant("origin.input", TensorType::f32(vec![2, 3]));
        let foreign = Builder::new();
        foreign.constant("foreign.input", TensorType::new(vec![7], DType::I32));
        assert_eq!(origin.generation(), foreign.generation());
        assert_eq!(
            origin.g.borrow().values.len(),
            foreign.g.borrow().values.len()
        );
        assert_ne!(state(&origin).names, state(&foreign).names);
        assert_ne!(state(&origin).types, state(&foreign).types);

        let plan = valid_plan(&origin, origin_input)?;
        let origin_before = state(&origin);
        let foreign_before = state(&foreign);
        assert!(matches!(
            foreign.preflight_append(plan),
            Err(BuilderAppendError::ForeignBuilder)
        ));
        assert_eq!(state(&origin), origin_before);
        assert_eq!(state(&foreign), foreign_before);

        let plan = valid_plan(&origin, origin_input)?;
        let mut prepared = origin.preflight_append(plan)?;
        assert!(matches!(
            foreign.commit_append(&mut prepared),
            Err(BuilderAppendError::ForeignBuilder)
        ));
        assert!(prepared.plan.is_some());
        assert_eq!(state(&origin), origin_before);
        assert_eq!(state(&foreign), foreign_before);
        Ok(())
    }

    #[test]
    fn builder_append_failure_is_atomic() -> Result<(), BuilderAppendError> {
        let builder = Builder::new();
        let input = builder.constant("input", TensorType::f32(vec![2, 3]));
        let mut plan = valid_plan(&builder, input)?;
        plan.equations[0].inputs[0] = Operand::Value(usize::MAX);
        let before = state(&builder);
        assert!(matches!(
            builder.preflight_append(plan),
            Err(BuilderAppendError::InvalidOperand {
                equation: 0,
                value: usize::MAX,
                ..
            })
        ));
        assert_eq!(state(&builder), before);
        Ok(())
    }

    #[test]
    fn builder_append_prepared_generation_is_enforced() -> Result<(), BuilderAppendError> {
        let builder = Builder::new();
        let input = builder.constant("input", TensorType::f32(vec![2, 3]));
        let plan = valid_plan(&builder, input)?;
        let mut prepared = builder.preflight_append(plan)?;
        builder.constant("intervening", TensorType::f32(vec![1]));
        let before = state(&builder);
        assert!(matches!(
            builder.commit_append(&mut prepared),
            Err(BuilderAppendError::StalePlan { .. })
        ));
        assert_eq!(state(&builder), before);

        let plan = valid_plan(&builder, input)?;
        let mut prepared = builder.preflight_append(plan)?;
        let output = builder.commit_append(&mut prepared)?;
        let after_success = state(&builder);
        assert!(output < after_success.values);
        assert_eq!(
            builder.commit_append(&mut prepared),
            Err(BuilderAppendError::ConsumedPlan)
        );
        assert_eq!(state(&builder), after_success);
        Ok(())
    }

    #[test]
    fn builder_append_reserves_every_collection_before_mutation() -> Result<(), BuilderAppendError>
    {
        for collection in [
            BuilderCollection::Values,
            BuilderCollection::Inputs,
            BuilderCollection::Constants,
            BuilderCollection::Slots,
            BuilderCollection::Equations,
        ] {
            let builder = Builder::new();
            builder.constant("base", TensorType::f32(vec![1]));
            let mut plan = builder.append_plan(0);
            let constant = plan.input("constant", TensorType::f32(vec![1]), Storage::Const)?;
            let slot = plan.input(
                "slot",
                TensorType::f32(vec![1]),
                Storage::Slot(Slot::Activation),
            )?;
            let output = plan.equation(
                OpKind::Binary(crate::op::BinOp::Add),
                vec![Operand::Value(constant.id), Operand::Value(slot.id)],
            )?;
            plan.declare_result(output)?;
            let mut prepared = builder.preflight_append(plan)?;
            let before = state(&builder);
            builder.reservation_failure.set(Some(collection));
            assert!(matches!(
                builder.commit_append(&mut prepared),
                Err(BuilderAppendError::Reserve {
                    collection: actual,
                    additional,
                }) if actual == collection && additional > 0
            ));
            assert_eq!(state(&builder), before, "{collection:?} reserved too late");
            assert!(prepared.plan.is_some(), "{collection:?} consumed the plan");
            assert_eq!(builder.commit_append(&mut prepared)?, output.id);
        }
        Ok(())
    }

    #[test]
    fn builder_append_is_atomic() -> Result<(), BuilderAppendError> {
        let builder = Builder::new();
        let input = builder.constant("input", TensorType::f32(vec![2, 3]));
        let mut plan = builder.append_plan(7);
        let constant = plan.input("weight", TensorType::f32(vec![3, 4]), Storage::Const)?;
        let output = plan.equation(
            OpKind::MatMul,
            vec![Operand::Value(input.id), Operand::Value(constant.id)],
        )?;
        plan.declare_result(output)?;
        let mut prepared = builder.preflight_append(plan)?;
        assert_eq!(
            prepared.accounting(),
            BuilderAppendAccounting {
                staged_values: 2,
                staged_equations: 1,
                packed_rows: 7,
                checked_name_bytes: 6,
            }
        );
        let generation = builder.generation();
        assert_eq!(builder.commit_append(&mut prepared)?, output.id);
        assert_eq!(builder.generation(), generation + 1);
        let graph = builder.finish(output);
        assert_eq!(graph.values.len(), 3);
        assert_eq!(graph.inputs, vec![input.id, constant.id]);
        assert_eq!(graph.consts, vec![input.id, constant.id]);
        assert_eq!(graph.eqns.len(), 1);
        assert_eq!(graph.eqns[0].out, output.id);
        Ok(())
    }

    #[test]
    fn exact_i32_builder_append_is_transactional() -> Result<(), BuilderAppendError> {
        let builder = Builder::new();
        let table = crate::test_support::i32_constant(&builder, "routing.table", vec![2, 3])?;
        let graph = builder.finish(table);
        assert_eq!(
            graph.aval(table.id),
            &TensorType::new(vec![2, 3], DType::I32)
        );
        assert_eq!(graph.meta(table.id).storage, Storage::Const);
        assert!(graph.eqns.is_empty());
        assert!(graph.validate().is_ok());

        let builder = Builder::new();
        crate::test_support::i32_constant(&builder, "taken", vec![1])?;
        let before = state(&builder);
        assert!(matches!(
            crate::test_support::i32_constant(&builder, "overflow", vec![usize::MAX, 2]),
            Err(BuilderAppendError::ElementCountOverflow { name, .. }) if name == "overflow"
        ));
        assert_eq!(state(&builder), before);
        assert!(matches!(
            crate::test_support::i32_constant(&builder, "taken", vec![1]),
            Err(BuilderAppendError::NameCollision { .. })
        ));
        assert_eq!(state(&builder), before);

        Ok(())
    }

    /// Card 529 (R466-012, SC-003): `Builder::binary`, `BuilderAppendPlan::equation_named`,
    /// `Builder::preflight_append` and `Graph::validate` all check a `Binary` op's two operands through
    /// one shared rule, `crate::op::check_binary_value_dtypes`. A mismatched F32/I32 pair (never a
    /// literal on either side, so no coercion carve-out applies) turns every one of the four red.
    #[test]
    fn binary_dtype_rule_rejects_the_same_mismatch_at_every_call_site() {
        // 1. Builder::binary panics at trace time.
        let builder = Builder::new();
        let float = builder.constant("float", TensorType::f32(vec![2]));
        let int = builder.constant("int", TensorType::new(vec![2], DType::I32));
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            builder.binary(BinOp::Add, float, int)
        }));
        assert!(
            panicked.is_err(),
            "Builder::binary must panic on mismatched dtypes"
        );

        // 2. BuilderAppendPlan::equation_named (via `equation`), which checks eagerly at staging time.
        let builder = Builder::new();
        let float = builder.constant("float", TensorType::f32(vec![2]));
        let int = builder.constant("int", TensorType::new(vec![2], DType::I32));
        let mut plan = builder.append_plan(0);
        let err = plan
            .equation(
                OpKind::Binary(BinOp::Add),
                vec![Operand::Value(float.id), Operand::Value(int.id)],
            )
            .expect_err("equation_named must reject the mismatched pair");
        assert!(matches!(
            err,
            BuilderAppendError::Inference {
                source: ShapeError::Dtype { .. },
                ..
            }
        ));

        // 3. Builder::preflight_append: a plan staged with matching operands, then corrupted post-hoc
        // to reference the mismatched value (mirroring Graph::validate's own test below), so it is
        // preflight's own check that fires, not equation_named's (which already ran once at staging).
        let builder = Builder::new();
        let float_a = builder.constant("float_a", TensorType::f32(vec![2]));
        let float_b = builder.constant("float_b", TensorType::f32(vec![2]));
        let int = builder.constant("int", TensorType::new(vec![2], DType::I32));
        let mut plan = builder.append_plan(0);
        let result = plan
            .equation(
                OpKind::Binary(BinOp::Add),
                vec![Operand::Value(float_a.id), Operand::Value(float_b.id)],
            )
            .unwrap();
        plan.declare_result(result).unwrap();
        plan.equations[0].inputs = vec![Operand::Value(float_a.id), Operand::Value(int.id)];
        let err = builder
            .preflight_append(plan)
            .expect_err("preflight_append must reject the corrupted, mismatched pair");
        assert!(matches!(
            err,
            BuilderAppendError::Inference {
                source: ShapeError::Dtype { .. },
                ..
            }
        ));

        // 4. Graph::validate: see `graph::tests::validate_rejects_mismatched_binary_value_operands_including_scalars`.
    }

    /// Card 551a: `Builder::resume` lets a finished graph take a head. The
    /// resumed builder's `out`/`state` rehydrate the old graph's values, a new equation can read `out`,
    /// and a second `finish_with_state` over the SAME state pair ids carries them through unchanged
    /// (SC-003/SC-006's "every state pair is unchanged" clause - the state pairs this test checks are
    /// the `(ValueId, ValueId)` identities `Graph::state` carries, not their eval output).
    #[test]
    fn resume_reopens_a_finished_graph_and_keeps_its_state_pairs() {
        let b = Builder::new();
        let logits = b.constant("logits", TensorType::f32(vec![4]));
        let cache_in = b.state_input("cache", TensorType::f32(vec![4]), StateRole::Recurrent);
        let g = b.finish_with_state(logits, &[(cache_in, cache_in)]);
        assert_eq!(g.output, logits.id);
        assert_eq!(g.state, vec![(cache_in.id, cache_in.id)]);

        let resumed = Builder::resume(g);
        assert_eq!(resumed.out.id, logits.id);
        let resumed_state_ids: Vec<(ValueId, ValueId)> = resumed
            .state
            .iter()
            .map(|(si, so)| (si.id, so.id))
            .collect();
        assert_eq!(resumed_state_ids, vec![(cache_in.id, cache_in.id)]);

        // Append an equation reading the resumed `out` - proves `append_plan`/`preflight_append` still
        // work on the new builder identity.
        let doubled = resumed
            .builder
            .binary(crate::op::BinOp::Add, resumed.out, resumed.out);
        let g2 = resumed.builder.finish_with_state(doubled, &resumed.state);
        assert_eq!(g2.output, doubled.id);
        assert_eq!(
            g2.state,
            vec![(cache_in.id, cache_in.id)],
            "the state pair must survive resume + a second finish_with_state unchanged"
        );
    }
}
