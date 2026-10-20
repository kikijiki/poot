//! The tracer: a `Builder` that mints `ValueId`s and appends eqns while a model forward runs with type-only
//! `Traced` handles (the `DynamicJaxprTrace` analog). Interior mutability lets ops take `&self`.
//!
//! Every primitive is fallible and budgeted: it returns a typed [`BuilderError`], and it reserves the
//! requested values, equations, literal bytes and owned storage from the caller's [`TraceLimits`]
//! ([`crate::budget`]) before it allocates or appends. A refused request changes nothing.

use std::cell::{Cell, RefCell};
#[cfg(any(test, feature = "test-support"))]
use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::budget::{AllocationSite, Charge, HeldCharge, Meter, TraceLimits};
#[cfg(any(test, feature = "test-support"))]
use crate::error::BuilderValueNamespace;
use crate::error::{BuilderCollection, BuilderError, Requirement};
#[cfg(any(test, feature = "test-support"))]
use crate::graph::ValueTable;
use crate::graph::{
    BoundaryId, ComputedConst, EditError, Eqn, Export, Graph, GraphEdit, GraphOutput, LayerIndex,
    Operand, Slot, SlotKey, StateRole, Storage, ValidationId, ValueId, ValueMeta,
};
use crate::op::{OpKind, SampleRule};
use crate::range::IndexRange;
use crate::types::{DType, Scalar, TensorType};
use crate::weight_ref::WeightRef;
use poot_quant::PackedWeight;

/// The identity of one [`Builder`] (or [`BuilderAppendPlan`] staged against it): a process-unique
/// token every handle it mints carries, so a handle used with another builder is recognized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BuilderBrand(NonZeroU64);

impl BuilderBrand {
    fn mint() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let minted = NEXT.fetch_add(1, Ordering::Relaxed);
        // `1 + minted` is nonzero by construction (the counter would need 2^64 builders to wrap).
        BuilderBrand(NonZeroU64::MIN.saturating_add(minted))
    }
}

/// A type-only tensor handle: a `ValueId` with no data, branded by the builder that minted it. There is no
/// `Deref`/`as_slice`/`to_scalar`, so a host read of a traced tensor cannot be written
/// (graph-architecture.md Q1). Static shape is available via [`Builder::aval`] for host-side static
/// decisions (`n_rep`, the rope half-split, etc.).
///
/// A handle is only meaningful to its own builder: every [`Builder`] method that takes one panics when
/// it comes from another builder (a tracer bug, not a recoverable condition). The transactional append
/// path ([`BuilderAppendPlan`]) names values by [`ValueId`] and reports a foreign plan as
/// [`BuilderError::ForeignBuilder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Traced {
    id: ValueId,
    brand: BuilderBrand,
}

impl Traced {
    /// The value this handle names in its builder's graph.
    pub fn id(self) -> ValueId {
        self.id
    }
}

/// Checked accounting for a prepared transactional append.
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuilderAppendAccounting {
    /// Values (input binders and equation results) the plan staged.
    pub staged_values: usize,
    /// Equations the plan staged.
    pub staged_equations: usize,
    /// Packed weight rows the plan's constants carry; checked against the packed-row limit before the append commits.
    pub packed_rows: usize,
    /// Bytes of staged names whose length was checked and charged.
    pub checked_name_bytes: usize,
}

/// A complete graph fragment staged against one builder generation.
///
/// Values and equations are private so callers cannot bypass checked id allocation or inference. Construct a
/// fragment with [`BuilderAppendPlan::input`], [`BuilderAppendPlan::input_result`],
/// [`BuilderAppendPlan::equation`], and [`BuilderAppendPlan::declare_result`].
///
/// A plan charges the builder's budget for the type snapshot it took and for everything it stages, and
/// gives the charge back when it is dropped without committing.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug)]
pub struct BuilderAppendPlan {
    brand: BuilderBrand,
    generation: u64,
    base_value_count: usize,
    base_types: ValueTable<TensorType>,
    values: Vec<ValueMeta>,
    equations: Vec<Eqn>,
    result: Option<DeclaredAppendResult>,
    packed_rows: usize,
    checked_name_bytes: usize,
    /// The builder's layer tag when the plan was created. A plan stages its equations after the
    /// caller may have left the scope, so the tag is snapshotted here (see [`Builder::append_plan`]).
    layer: Option<LayerIndex>,
    /// Everything this plan charged the builder's budget for.
    held: HeldCharge,
    /// The part of `held` that is the type snapshot: it dies with the plan even when the staged
    /// storage becomes the graph's.
    snapshot: Charge,
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug)]
enum DeclaredAppendResult {
    StagedInput(ValueId),
    EquationOutput(ValueId),
}

#[cfg(any(test, feature = "test-support"))]
impl DeclaredAppendResult {
    fn value(self) -> ValueId {
        match self {
            Self::StagedInput(value) => value,
            Self::EquationOutput(value) => value,
        }
    }
}

const WORD: u64 = std::mem::size_of::<usize>() as u64;

/// `a * b` as owned bytes, or the overflow refusal at `site`.
fn times(
    meter: &Meter,
    site: AllocationSite,
    count: usize,
    each: usize,
) -> Result<u64, BuilderError> {
    (count as u64)
        .checked_mul(each as u64)
        .ok_or_else(|| meter.size_overflow(site).into())
}

fn plus(meter: &Meter, site: AllocationSite, a: u64, b: u64) -> Result<u64, BuilderError> {
    a.checked_add(b)
        .ok_or_else(|| meter.size_overflow(site).into())
}

/// The owned bytes of one value record: its inline [`ValueMeta`], the heap shape list of its type and its
/// name.
fn value_record_bytes(
    meter: &Meter,
    site: AllocationSite,
    rank: usize,
    name_len: usize,
) -> Result<u64, BuilderError> {
    let shape = times(meter, site, rank, std::mem::size_of::<usize>())?;
    let inline = std::mem::size_of::<ValueMeta>() as u64;
    plus(
        meter,
        site,
        plus(meter, site, inline, shape)?,
        name_len as u64,
    )
}

/// The owned bytes of one equation record: its inline [`Eqn`], its operand list and its operation's
/// attribute storage.
fn eqn_record_bytes(
    meter: &Meter,
    site: AllocationSite,
    op: &OpKind,
    operands: usize,
) -> Result<u64, BuilderError> {
    let list = times(meter, site, operands, std::mem::size_of::<Operand>())?;
    let attributes = op
        .owned_heap_bytes()
        .ok_or_else(|| BuilderError::from(meter.size_overflow(site)))?;
    plus(
        meter,
        site,
        plus(meter, site, std::mem::size_of::<Eqn>() as u64, list)?,
        attributes,
    )
}

/// What one equation costs before it is appended: the budget request and the clone of its operand types
/// that inference reads.
struct EquationPrice {
    literal: u64,
    /// The inline and heap bytes of the cloned operand types.
    scratch: u64,
    /// An upper bound of the output type's rank, known before inference allocates it.
    rank_bound: usize,
}

/// Why an equation could not be priced: an operand the table does not hold, or a size computation that
/// does not fit (the refusal names the overflowing request, not a wrapped or clamped number).
enum PriceRefusal {
    Operand(ValueId),
    Overflow(crate::budget::TraceBudgetExceeded),
}

impl PriceRefusal {
    fn into_error(self, equation: usize, operation: String) -> BuilderError {
        match self {
            Self::Operand(value) => BuilderError::InvalidOperand {
                equation,
                operation,
                value,
            },
            Self::Overflow(budget) => budget.into(),
        }
    }
}

/// An upper bound of an output type's rank: the operand ranks plus one word-sized rank per attribute
/// word, plus one. `None` when an input is already an overflowed size or the sum does not fit.
pub(crate) fn rank_bound(rank_sum: usize, attribute_bytes: Option<u64>) -> Option<usize> {
    usize::try_from(attribute_bytes? / WORD)
        .ok()
        .and_then(|ranks| rank_sum.checked_add(ranks))
        .and_then(|bound| bound.checked_add(1))
}

/// Price an equation from borrowed operand types, allocating nothing. Every sum and product is checked:
/// an overflow is [`Attempted::Overflow`] at `site`, never a saturated number a limit might admit.
fn price_equation<'t>(
    meter: &Meter,
    site: AllocationSite,
    op: &OpKind,
    operands: &[Operand],
    mut type_of: impl FnMut(ValueId) -> Option<&'t TensorType>,
) -> Result<EquationPrice, PriceRefusal> {
    let over = || PriceRefusal::Overflow(meter.size_overflow(site));
    let mut literal = 0u64;
    let mut scratch = 0u64;
    let mut rank_sum = 0usize;
    let type_inline = std::mem::size_of::<TensorType>() as u64;
    for operand in operands {
        match operand {
            Operand::Lit(value) => {
                literal = literal.checked_add(value.byte_len()).ok_or_else(over)?;
                scratch = scratch.checked_add(type_inline).ok_or_else(over)?;
            }
            Operand::Value(value) => {
                let ty = type_of(*value).ok_or(PriceRefusal::Operand(*value))?;
                rank_sum = rank_sum.checked_add(ty.shape.len()).ok_or_else(over)?;
                let shape_bytes = (ty.shape.len() as u64).checked_mul(WORD).ok_or_else(over)?;
                scratch = scratch
                    .checked_add(type_inline)
                    .and_then(|bytes| bytes.checked_add(shape_bytes))
                    .ok_or_else(over)?;
            }
        }
    }
    let rank_bound = rank_bound(rank_sum, op.owned_heap_bytes()).ok_or_else(over)?;
    Ok(EquationPrice {
        literal,
        scratch,
        rank_bound,
    })
}

/// The types an equation's operands have, cloned: inference reads them. The budget reserved their size
/// ([`EquationPrice::scratch`]) before this runs.
fn clone_operand_types<'t>(
    op: &OpKind,
    operands: &[Operand],
    equation: usize,
    mut type_of: impl FnMut(ValueId) -> Option<&'t TensorType>,
) -> Result<Vec<TensorType>, BuilderError> {
    let mut types = Vec::new();
    reserve(operands.len(), BuilderCollection::Values, |n| {
        types.try_reserve_exact(n)
    })?;
    for operand in operands {
        types.push(match operand {
            Operand::Lit(value) => value.ty(),
            Operand::Value(value) => {
                type_of(*value)
                    .cloned()
                    .ok_or_else(|| BuilderError::InvalidOperand {
                        equation,
                        operation: op.name(),
                        value: *value,
                    })?
            }
        });
    }
    Ok(types)
}

/// The output type of an equation over `in_types`, with the dtype rule two value operands of a binary
/// operation obey.
fn infer_equation(
    op: &OpKind,
    operands: &[Operand],
    in_types: &[TensorType],
    equation: usize,
) -> Result<TensorType, BuilderError> {
    checked_op_sizes(op, in_types)?;
    if let [Operand::Value(_), Operand::Value(_)] = operands
        && matches!(op, OpKind::Binary(_))
    {
        crate::op::check_binary_value_dtypes(&in_types[0], &in_types[1]).map_err(|source| {
            BuilderError::Inference {
                equation,
                operation: op.name(),
                source,
            }
        })?;
    }
    op.infer(in_types)
        .map_err(|source| BuilderError::Inference {
            equation,
            operation: op.name(),
            source,
        })
}

#[cfg(any(test, feature = "test-support"))]
impl BuilderAppendPlan {
    fn next_value_id(&self) -> Result<ValueId, BuilderError> {
        self.staged_id(self.values.len())
    }

    /// The id the staged value at `offset` takes when this plan commits.
    fn staged_id(&self, offset: usize) -> Result<ValueId, BuilderError> {
        self.base_value_count
            .checked_add(offset)
            .and_then(ValueId::from_index)
            .ok_or(BuilderError::SizeOverflow {
                collection: BuilderCollection::Values,
                current: self.base_value_count,
                additional: offset,
            })
    }

    /// The handle of a value this plan staged.
    fn handle(&self, id: ValueId) -> Traced {
        Traced {
            id,
            brand: self.brand,
        }
    }

    fn add_name_bytes(&mut self, name: &str) -> Result<(), BuilderError> {
        self.checked_name_bytes =
            self.checked_name_bytes
                .checked_add(name.len())
                .ok_or(BuilderError::SizeOverflow {
                    collection: BuilderCollection::NameBytes,
                    current: self.checked_name_bytes,
                    additional: name.len(),
                })?;
        Ok(())
    }

    /// Reserve `want` for this plan, or refuse it and keep every earlier holding.
    fn charge(&mut self, want: Charge, site: AllocationSite) -> Result<(), BuilderError> {
        self.held.reserve(want, site)?;
        Ok(())
    }

    /// Stage a named input binder for use by later equations. Use [`Self::input_result`] when the plan-owned input
    /// is the append result. Device values must be created by [`Self::equation`].
    pub fn input(
        &mut self,
        name: impl Into<String>,
        aval: TensorType,
        storage: Storage,
    ) -> Result<Traced, BuilderError> {
        let name = name.into();
        if storage == Storage::Device {
            return Err(BuilderError::DeviceInput { name });
        }
        debug_assert!(
            storage != Storage::State,
            "BuilderAppendPlan::input: no append-plan caller stages a State input; \
             Builder::state_input builds one directly"
        );
        let id = self.next_value_id()?;
        let record = value_record_bytes(
            self.held.meter(),
            AllocationSite::StagedValue,
            aval.shape.len(),
            0,
        )?;
        let name_bytes = name.len() as u64;
        // A refusal at any step gives the earlier steps back: a refused input leaves the plan's
        // holdings as they were.
        let mut taken = Charge::default();
        let staged = (|| {
            let want = Charge {
                values: 1,
                owned: record,
                ..Charge::default()
            };
            self.charge(want, AllocationSite::StagedValue)?;
            taken = taken.checked_add(want).unwrap_or(taken);
            let want = Charge::owned(name_bytes);
            self.charge(want, AllocationSite::StagedName)?;
            taken = taken.checked_add(want).unwrap_or(taken);
            reserve(1, BuilderCollection::Values, |n| self.values.try_reserve(n))?;
            self.add_name_bytes(&name)
        })();
        if let Err(error) = staged {
            self.held.release(taken);
            return Err(error);
        }
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
            weight: None,
            range: None,
        });
        Ok(self.handle(id))
    }

    /// Stage the constant `weight` names, carrying its [`WeightRef`] ([`Builder::weight`] for an append
    /// plan): a packed carrier a fixture stages as already bound.
    #[cfg(any(test, feature = "test-support"))]
    pub fn weight_input(
        &mut self,
        weight: WeightRef,
        aval: TensorType,
    ) -> Result<Traced, BuilderError> {
        let handle = self.input(weight.name(), aval, Storage::Const)?;
        self.values
            .last_mut()
            .expect("input staged the value just now")
            .weight = Some(weight);
        Ok(handle)
    }

    /// Atomically stage a named non-Device input binder and declare that plan-owned input as the result.
    pub fn input_result(
        &mut self,
        name: impl Into<String>,
        aval: TensorType,
        storage: Storage,
    ) -> Result<Traced, BuilderError> {
        if let Some(result) = self.result {
            return Err(BuilderError::ResultAlreadyDeclared {
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
    pub fn equation(&mut self, op: OpKind, operands: Vec<Operand>) -> Result<Traced, BuilderError> {
        self.equation_named(op, operands, None)
    }

    /// Stage one inferred Device equation result with an optional diagnostic name. Like
    /// [`Self::equation`], it accepts composites for tests; a tracer must not stage one.
    pub fn equation_named(
        &mut self,
        op: OpKind,
        operands: Vec<Operand>,
        name: Option<String>,
    ) -> Result<Traced, BuilderError> {
        let equation = self.equations.len();
        let price = price_equation(
            self.held.meter(),
            AllocationSite::StagedEquation,
            &op,
            &operands,
            |value| self.type_of(value),
        )
        .map_err(|refusal| refusal.into_error(equation, op.name()))?;
        let id = self.next_value_id()?;
        let site = AllocationSite::StagedEquation;
        let record = eqn_record_bytes(self.held.meter(), site, &op, operands.len())?;
        let out_bound = times(
            self.held.meter(),
            AllocationSite::StagedInference,
            price.rank_bound,
            std::mem::size_of::<usize>(),
        )?;
        let value_record = value_record_bytes(self.held.meter(), site, 0, 0)?;
        let name_bytes = name.as_ref().map_or(0, |name| name.len()) as u64;

        // Everything this call charges, in the order it is reserved; a refusal at any step gives the
        // earlier steps back, so a refused equation leaves the plan's holdings as they were.
        let mut taken = Charge::default();
        let result = (|| {
            let want = Charge {
                values: 1,
                eqns: 1,
                literal: price.literal,
                owned: plus(self.held.meter(), site, record, value_record)?,
            };
            self.charge(want, site)?;
            taken = taken.checked_add(want).unwrap_or(taken);
            if name_bytes > 0 {
                let want = Charge::owned(name_bytes);
                self.charge(want, AllocationSite::StagedName)?;
                taken = taken.checked_add(want).unwrap_or(taken);
            }
            let scratch = Charge::owned(plus(
                self.held.meter(),
                AllocationSite::StagedInference,
                price.scratch,
                out_bound,
            )?);
            self.charge(scratch, AllocationSite::StagedInference)?;
            taken = taken.checked_add(scratch).unwrap_or(taken);
            let in_types =
                clone_operand_types(&op, &operands, equation, |value| self.type_of(value))?;
            let aval = infer_equation(&op, &operands, &in_types, equation)?;
            drop(in_types);
            // Settle the inference scratch: the operand clones are gone and the output keeps its
            // actual shape list, not the bound.
            let actual = times(
                self.held.meter(),
                AllocationSite::InferenceResult,
                aval.shape.len(),
                std::mem::size_of::<usize>(),
            )?;
            let settled = price
                .scratch
                .saturating_add(out_bound.saturating_sub(actual));
            self.held.release(Charge::owned(settled));
            taken.owned = taken.owned.saturating_sub(settled);
            if actual > out_bound {
                let extra = Charge::owned(actual - out_bound);
                self.charge(extra, AllocationSite::InferenceResult)?;
                taken = taken.checked_add(extra).unwrap_or(taken);
            }
            self.add_name_bytes(name.as_deref().unwrap_or(""))?;
            reserve(1, BuilderCollection::Values, |n| self.values.try_reserve(n))?;
            reserve(1, BuilderCollection::Equations, |n| {
                self.equations.try_reserve(n)
            })?;
            Ok(aval)
        })();
        let aval = match result {
            Ok(aval) => aval,
            Err(error) => {
                self.held.release(taken);
                return Err(error);
            }
        };
        self.values.push(ValueMeta {
            aval,
            storage: Storage::Device,
            name,
            key: None,
            state_role: None,
            weight: None,
            range: None,
        });
        self.equations.push(Eqn {
            op,
            inputs: operands,
            out: id,
            layer: self.layer,
        });
        Ok(self.handle(id))
    }

    /// Declare a staged equation output as the result. Input results must be staged and declared together with
    /// [`Self::input_result`]; preflight enforces this because a handle names a value, not how it was staged.
    ///
    /// A handle another builder minted is [`BuilderError::ForeignBuilder`]: this is the transactional
    /// path, where the typed error replaces the tracer's panic.
    pub fn declare_result(&mut self, result: Traced) -> Result<(), BuilderError> {
        if result.brand != self.brand {
            return Err(BuilderError::ForeignBuilder);
        }
        if let Some(value) = self.result {
            return Err(BuilderError::ResultAlreadyDeclared {
                value: value.value(),
            });
        }
        self.result = Some(DeclaredAppendResult::EquationOutput(result.id));
        Ok(())
    }

    /// The type of `value`: a value the base graph already held, or one this plan staged. `None` when the id is in neither.
    pub fn type_of(&self, value: ValueId) -> Option<&TensorType> {
        match value.index().checked_sub(self.base_value_count) {
            None => self.base_types.get(value),
            Some(offset) => self.values.get(offset).map(|meta| &meta.aval),
        }
    }
}

/// A fully checked append. Neither `Clone` nor reusable.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug)]
pub struct PreparedBuilderAppend {
    brand: BuilderBrand,
    generation: u64,
    base_value_count: usize,
    accounting: BuilderAppendAccounting,
    staged_inputs: usize,
    staged_constants: usize,
    staged_slots: usize,
    result: PreparedAppendResult,
    plan: Option<BuilderAppendPlan>,
}

#[cfg(any(test, feature = "test-support"))]
impl PreparedBuilderAppend {
    /// The checked counts this append staged, as validated by `prepare`.
    pub fn accounting(&self) -> BuilderAppendAccounting {
        self.accounting
    }
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Debug)]
enum PreparedAppendResult {
    StagedInput(ValueId),
    EquationOutput(ValueId),
}

#[cfg(any(test, feature = "test-support"))]
impl PreparedAppendResult {
    fn value(&self) -> ValueId {
        match *self {
            Self::StagedInput(value) => value,
            Self::EquationOutput(value) => value,
        }
    }
}

/// The `MatMulDequant` family is deleted (card 545b): `Builder` has no `matmul_dequant` or
/// `indexed_matmul_dequant` method, and `poot_graph_ir` exports no `QuantScheme`. `PackedDequant` /
/// `PackedContraction` (built through `packed_graphs::packed_linear` and friends) are the only quantized
/// primitives left.
///
/// ```compile_fail,E0599
/// use poot_graph_ir::builder::Builder;
/// # use poot_graph_ir::TraceLimits;
/// # use std::num::{NonZeroU64, NonZeroUsize};
/// # let n = NonZeroUsize::MIN;
/// # let limits = TraceLimits { max_tokens: n, max_values: n, max_eqns: n, max_literal_bytes: NonZeroU64::MIN, max_owned_bytes: NonZeroU64::MIN };
///
/// let b = Builder::new(limits);
/// let _ = b.matmul_dequant();
/// ```
///
/// ```compile_fail,E0599
/// use poot_graph_ir::builder::Builder;
/// # use poot_graph_ir::TraceLimits;
/// # use std::num::{NonZeroU64, NonZeroUsize};
/// # let n = NonZeroUsize::MIN;
/// # let limits = TraceLimits { max_tokens: n, max_values: n, max_eqns: n, max_literal_bytes: NonZeroU64::MIN, max_owned_bytes: NonZeroU64::MIN };
///
/// let b = Builder::new(limits);
/// let _ = b.indexed_matmul_dequant();
/// ```
///
/// ```compile_fail,E0432
/// use poot_graph_ir::QuantScheme;
/// ```
///
/// Card 557: a tracer emits primitives and `ops::` compositions only. The contraction epilogue and flash
/// attention are composites `compile`'s passes form ([`crate::op::OpKind::class`]), and the kernel choice
/// behind them is the planner's, so `Builder` has no method for either. Each call below uses the deleted
/// method's own signature, so re-adding one makes its example compile.
///
/// ```compile_fail,E0599
/// use poot_graph_ir::{Builder, TensorType};
/// # use poot_graph_ir::TraceLimits;
/// # use std::num::{NonZeroU64, NonZeroUsize};
/// # let n = NonZeroUsize::MIN;
/// # let limits = TraceLimits { max_tokens: n, max_values: n, max_eqns: n, max_literal_bytes: NonZeroU64::MIN, max_owned_bytes: NonZeroU64::MIN };
///
/// let b = Builder::new(limits);
/// let x = b.constant("x", TensorType::f32(vec![2, 4]));
/// let w = b.constant("w", TensorType::f32(vec![4, 3]));
/// let bias = b.constant("bias", TensorType::f32(vec![3]));
/// let _ = b.matmul_bias(x, w, bias);
/// ```
///
/// ```compile_fail,E0599
/// use poot_graph_ir::{Builder, TensorType};
/// # use poot_graph_ir::TraceLimits;
/// # use std::num::{NonZeroU64, NonZeroUsize};
/// # let n = NonZeroUsize::MIN;
/// # let limits = TraceLimits { max_tokens: n, max_values: n, max_eqns: n, max_literal_bytes: NonZeroU64::MIN, max_owned_bytes: NonZeroU64::MIN };
///
/// let b = Builder::new(limits);
/// let q = b.constant("q", TensorType::f32(vec![1, 2, 1, 4]));
/// let k = b.constant("k", TensorType::f32(vec![1, 2, 8, 4]));
/// let v = b.constant("v", TensorType::f32(vec![1, 2, 8, 4]));
/// let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, 8]));
/// let _ = b.flash_attention_decode(q, k, v, mask, 1, 0.5);
/// ```
///
/// ```compile_fail,E0599
/// use poot_graph_ir::{Builder, TensorType};
/// # use poot_graph_ir::TraceLimits;
/// # use std::num::{NonZeroU64, NonZeroUsize};
/// # let n = NonZeroUsize::MIN;
/// # let limits = TraceLimits { max_tokens: n, max_values: n, max_eqns: n, max_literal_bytes: NonZeroU64::MIN, max_owned_bytes: NonZeroU64::MIN };
///
/// let b = Builder::new(limits);
/// let q = b.constant("q", TensorType::f32(vec![1, 2, 4, 4]));
/// let k = b.constant("k", TensorType::f32(vec![1, 2, 4, 4]));
/// let v = b.constant("v", TensorType::f32(vec![1, 2, 4, 4]));
/// let mask = b.constant("mask", TensorType::f32(vec![1, 1, 4, 4]));
/// let _ = b.flash_attention_prefill(q, k, v, mask, 1, 0.5);
/// ```
///
/// ```compile_fail,E0599
/// use poot_graph_ir::{Builder, TensorType};
/// # use poot_graph_ir::TraceLimits;
/// # use std::num::{NonZeroU64, NonZeroUsize};
/// # let n = NonZeroUsize::MIN;
/// # let limits = TraceLimits { max_tokens: n, max_values: n, max_eqns: n, max_literal_bytes: NonZeroU64::MIN, max_owned_bytes: NonZeroU64::MIN };
///
/// let b = Builder::new(limits);
/// let q = b.constant("q", TensorType::f32(vec![1, 2, 4, 4]));
/// let k = b.constant("k", TensorType::f32(vec![1, 2, 4, 4]));
/// let v = b.constant("v", TensorType::f32(vec![1, 2, 4, 4]));
/// let mask = b.constant("mask", TensorType::f32(vec![1, 1, 4, 4]));
/// let _ = b.flash_attention_prefill_softcap(q, k, v, mask, 1, 0.5, Some(50.0));
/// ```
pub struct Builder {
    brand: BuilderBrand,
    g: RefCell<GraphEdit>,
    /// The state inputs of the graph this builder resumed (none for a new one): every one must be
    /// carried again by [`Builder::finish`].
    carried: Vec<ValueId>,
    generation: Cell<u64>,
    /// The layer tag every equation emitted right now carries. Shared with the [`LayerScope`]s this
    /// builder hands out, so a scope restores the previous tag on drop without borrowing the
    /// builder (a tracer holds the guard while its `&mut self` layer method runs).
    layer: Rc<Cell<Option<LayerIndex>>>,
    /// The caller's budget and what this trace has charged against it so far.
    meter: Rc<Meter>,
    #[cfg(any(test, feature = "test-support"))]
    reservation_failure: Cell<Option<BuilderCollection>>,
}

/// The result of [`Builder::resume`]: a fresh builder over a previously finished graph, plus
/// everything that graph published, rehydrated as [`Traced`] so a caller can append new equations
/// that read it and finish the graph again.
pub struct Resumed {
    /// A builder over the finished graph, ready to append equations that read its values.
    pub builder: Builder,
    /// What the finished graph published, as handles of `builder`.
    pub outputs: Outputs,
}

/// A validation witness declared at [`Builder::finish`].
pub struct WitnessDeclaration {
    /// The witness's identity in the graph's validation table.
    pub id: ValidationId,
    /// The human-readable name the witness is published under.
    pub name: String,
    /// The value whose contents the witness exposes.
    pub value: Traced,
}

/// Everything a graph publishes, as the handles of the builder that finishes it: the primary
/// result, the carried-state pairs, the validation witnesses and the exports.
///
/// A bare [`Traced`] converts to the outputs of a stateless graph with that result.
pub struct Outputs {
    /// The graph's primary result.
    pub result: Traced,
    /// `(the State input binder, the updated value)` per cache buffer.
    pub state: Vec<(Traced, Traced)>,
    /// The validation witnesses the graph exposes alongside its result.
    pub witnesses: Vec<WitnessDeclaration>,
    /// The exported values, each under its boundary id.
    pub exports: Vec<(BoundaryId, Traced)>,
}

impl From<Traced> for Outputs {
    fn from(result: Traced) -> Self {
        Outputs::new(result)
    }
}

impl Outputs {
    /// The outputs of a stateless graph whose primary result is `result`.
    pub fn new(result: Traced) -> Self {
        Outputs {
            result,
            state: Vec::new(),
            witnesses: Vec::new(),
            exports: Vec::new(),
        }
    }

    /// Carry `state` pairs `(state_in, state_out)` across steps (the KV cache).
    #[must_use]
    pub fn with_state(mut self, state: &[(Traced, Traced)]) -> Self {
        self.state.extend_from_slice(state);
        self
    }

    /// Declare a bounded F32 validation witness, checked before the result or state is published.
    #[must_use]
    pub fn with_witness(
        mut self,
        id: ValidationId,
        name: impl Into<String>,
        value: Traced,
    ) -> Self {
        self.witnesses.push(WitnessDeclaration {
            id,
            name: name.into(),
            value,
        });
        self
    }

    /// Hand `value` to another stage of a split program under `id`.
    #[must_use]
    pub fn with_export(mut self, id: BoundaryId, value: Traced) -> Self {
        self.exports.push((id, value));
        self
    }
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
) -> Result<usize, BuilderError> {
    current
        .checked_add(additional)
        .ok_or(BuilderError::SizeOverflow {
            collection,
            current,
            additional,
        })
}

fn reserve(
    additional: usize,
    collection: BuilderCollection,
    try_reserve: impl FnOnce(usize) -> Result<(), std::collections::TryReserveError>,
) -> Result<(), BuilderError> {
    try_reserve(additional).map_err(|_| BuilderError::Reserve {
        collection,
        additional,
    })
}

fn checked_op_sizes(op: &OpKind, inputs: &[TensorType]) -> Result<(), BuilderError> {
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

#[cfg(any(test, feature = "test-support"))]
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

#[cfg(any(test, feature = "test-support"))]
fn live_name_namespace(g: &Graph, name: &str) -> Option<BuilderValueNamespace> {
    let (value, _) = g
        .values()
        .iter_enumerated()
        .find(|(_, meta)| meta.name.as_deref() == Some(name))?;
    if g.consts().contains(&value) {
        Some(BuilderValueNamespace::Constant)
    } else if g.slots().iter().any(|&(id, _)| id == value) {
        Some(BuilderValueNamespace::Slot)
    } else if g.eqns().iter().any(|eqn| eqn.out == value) {
        Some(BuilderValueNamespace::EquationResult)
    } else if g.inputs().contains(&value) {
        Some(BuilderValueNamespace::Input)
    } else {
        Some(BuilderValueNamespace::NamedValue)
    }
}

#[cfg(any(test, feature = "test-support"))]
fn validate_append_result(
    plan: &BuilderAppendPlan,
    live_value_count: usize,
    defined: &[bool],
) -> Result<PreparedAppendResult, BuilderError> {
    let declared = plan.result.ok_or(BuilderError::MissingResult)?;
    let value = declared.value();
    let Some(offset) = value.index().checked_sub(live_value_count) else {
        return Err(BuilderError::InvalidResult { value });
    };
    let Some(meta) = plan.values.get(offset) else {
        return Err(BuilderError::InvalidResult { value });
    };
    if !defined.get(offset).copied().unwrap_or(false) {
        return Err(BuilderError::InvalidResult { value });
    }
    match (declared, meta.storage) {
        (DeclaredAppendResult::EquationOutput(_), Storage::Device) => {
            Ok(PreparedAppendResult::EquationOutput(value))
        }
        (
            DeclaredAppendResult::StagedInput(_),
            Storage::Const | Storage::Computed(_) | Storage::Slot(_) | Storage::State,
        ) => Ok(PreparedAppendResult::StagedInput(value)),
        _ => Err(BuilderError::InvalidResult { value }),
    }
}

/// Which binder tables an input value joins besides the input list.
#[derive(Clone, Copy)]
enum InputTable {
    Const,
    Slot(Slot),
}

impl Builder {
    /// A builder whose every allocation and append is charged against `limits`. There is no
    /// unlimited builder: the caller states what the trace may cost.
    pub fn new(limits: TraceLimits) -> Self {
        Builder {
            brand: BuilderBrand::mint(),
            g: RefCell::new(GraphEdit::new()),
            carried: Vec::new(),
            generation: Cell::new(0),
            layer: Rc::new(Cell::new(None)),
            meter: Rc::new(Meter::new(limits)),
            #[cfg(any(test, feature = "test-support"))]
            reservation_failure: Cell::new(None),
        }
    }

    /// The limits this builder charges against.
    #[cfg(any(test, feature = "test-support"))]
    pub fn limits(&self) -> TraceLimits {
        self.meter.limits()
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
    /// The tag is what `poot-graph-plan`'s `StageAssignment::from_sites` maps a placement's
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

    /// Finalize: declare `outputs` and return the checked graph.
    ///
    /// Passing a bare [`Traced`] declares it as the primary result with nothing else published. A
    /// carried-state cache, validation witnesses and exports are declared on [`Outputs`].
    ///
    /// # Errors
    ///
    /// [`BuilderError::Budget`] when the declarations do not fit the budget,
    /// [`BuilderError::ForeignHandle`] when a handle in `outputs` was minted by another builder, and
    /// [`BuilderError::Invalid`] when the finished graph is not well formed: a state pair whose input
    /// is not a `State` input or whose types differ, a witness that is not a bounded F32 value, a
    /// duplicate witness or export id.
    pub fn finish(self, outputs: impl Into<Outputs>) -> Result<Graph, BuilderError> {
        let outputs = outputs.into();
        let meter = &self.meter;
        let output_entry = std::mem::size_of::<GraphOutput>();
        let mut held = HeldCharge::new(Rc::clone(&self.meter));
        let state_bytes = times(
            meter,
            AllocationSite::FinalState,
            outputs.state.len(),
            std::mem::size_of::<(ValueId, ValueId)>() + output_entry,
        )?;
        held.reserve(Charge::owned(state_bytes), AllocationSite::FinalState)?;
        let mut witness_bytes = times(
            meter,
            AllocationSite::FinalValidation,
            outputs.witnesses.len(),
            std::mem::size_of::<(ValidationId, String, ValueId)>() + output_entry,
        )?;
        for witness in &outputs.witnesses {
            witness_bytes = plus(
                meter,
                AllocationSite::FinalValidation,
                witness_bytes,
                witness.name.len() as u64,
            )?;
        }
        held.reserve(
            Charge::owned(witness_bytes),
            AllocationSite::FinalValidation,
        )?;
        let export_bytes = times(
            meter,
            AllocationSite::FinalExports,
            outputs.exports.len(),
            std::mem::size_of::<Export>() + output_entry,
        )?;
        held.reserve(Charge::owned(export_bytes), AllocationSite::FinalExports)?;

        let result = self.id(outputs.result)?;
        let mut state: Vec<(ValueId, ValueId)> = Vec::new();
        reserve(outputs.state.len(), BuilderCollection::Outputs, |n| {
            state.try_reserve_exact(n)
        })?;
        for &(input, next) in &outputs.state {
            state.push((self.id(input)?, self.id(next)?));
        }
        let mut witnesses: Vec<(ValidationId, String, ValueId)> = Vec::new();
        reserve(outputs.witnesses.len(), BuilderCollection::Outputs, |n| {
            witnesses.try_reserve_exact(n)
        })?;
        for witness in outputs.witnesses {
            let value = self.id(witness.value)?;
            witnesses.push((witness.id, witness.name, value));
        }
        let mut exports: Vec<Export> = Vec::new();
        reserve(outputs.exports.len(), BuilderCollection::Outputs, |n| {
            exports.try_reserve_exact(n)
        })?;
        for &(id, value) in &outputs.exports {
            exports.push(Export {
                id,
                value: self.id(value)?,
            });
        }
        if let Some(&input) = self
            .carried
            .iter()
            .find(|input| !state.iter().any(|&(declared, _)| declared == **input))
        {
            return Err(EditError::StateDropped { input }.into());
        }
        let declarations = 1 + state.len() + witnesses.len() + exports.len();
        let mut draft = self.g.into_inner();
        reserve(declarations, BuilderCollection::Outputs, |n| {
            draft.try_reserve_outputs(n)
        })?;
        draft.set_result(result);
        for (id, name, value) in witnesses {
            draft.push_witness(id, name, value);
        }
        for export in exports {
            draft.push_export(export);
        }
        for (input, next) in state {
            draft.push_state(input, next);
        }
        Ok(draft.commit()?)
    }

    /// Reopen a finished graph for one more append (card 551a): `finish` consumes `self`, so
    /// extending an already-finished graph (card 551b appends a sampler suffix after the driver's
    /// per-step graph) needs a fresh `Builder` over `g`'s own live values/eqns. The returned builder
    /// is a new identity (never aliasing whatever builder produced `g`), so
    /// [`Builder::append_plan`]/[`Builder::preflight_append`] work unchanged on it.
    ///
    /// `outputs` rehydrates `g`'s result and carried-state pairs as [`Traced`], so new equations can
    /// read them and [`Builder::finish`] re-declares them, changed or not. Its witnesses and exports
    /// stay declared as they were, and a finish that does not hand back every carried-state pair is
    /// refused, so extending a graph cannot silently stop carrying a cache.
    ///
    /// The graph's own values, equations, literals and storage are charged against `limits` up
    /// front: a graph that already exceeds them is refused ([`BuilderError::Budget`]).
    pub fn resume(g: Graph, limits: TraceLimits) -> Result<Resumed, BuilderError> {
        let meter = Rc::new(Meter::new(limits));
        meter.reserve(graph_charge(&meter, &g)?, AllocationSite::ResumedGraph)?;
        let result = g.result();
        let state: Vec<(ValueId, ValueId)> = g.state().collect();
        let mut draft = g.into_draft();
        draft.retain_state(|_, _| false);
        let builder = Builder {
            brand: BuilderBrand::mint(),
            g: RefCell::new(draft),
            carried: state.iter().map(|&(input, _)| input).collect(),
            generation: Cell::new(0),
            layer: Rc::new(Cell::new(None)),
            meter,
            #[cfg(any(test, feature = "test-support"))]
            reservation_failure: Cell::new(None),
        };
        let outputs = Outputs {
            result: builder.handle(result),
            state: state
                .iter()
                .map(|&(input, next)| (builder.handle(input), builder.handle(next)))
                .collect(),
            witnesses: Vec::new(),
            exports: Vec::new(),
        };
        Ok(Resumed { builder, outputs })
    }

    /// The handle for `id`, a value this builder just minted.
    pub(crate) fn handle(&self, id: ValueId) -> Traced {
        Traced {
            id,
            brand: self.brand,
        }
    }

    /// The value `t` names, or [`BuilderError::ForeignHandle`] when another builder minted it.
    fn id(&self, t: Traced) -> Result<ValueId, BuilderError> {
        if t.brand != self.brand {
            return Err(BuilderError::ForeignHandle { value: t.id });
        }
        Ok(t.id)
    }

    /// The operand naming the value `t`.
    fn operand(&self, t: Traced) -> Result<Operand, BuilderError> {
        Ok(Operand::Value(self.id(t)?))
    }

    /// The static type of `t`, for host-side static decisions (`n_rep`, the rope half-split, ...).
    pub fn aval(&self, t: Traced) -> Result<TensorType, BuilderError> {
        let id = self.id(t)?;
        self.g
            .borrow()
            .values()
            .get(id)
            .map(|meta| meta.aval.clone())
            .ok_or(BuilderError::UnknownValue { value: id })
    }

    /// The mutation generation used to invalidate prepared transactional appends.
    pub fn generation(&self) -> u64 {
        self.generation.get()
    }

    /// Start a staged append against the builder's current immutable generation.
    ///
    /// The plan snapshots every value type of the builder; the snapshot is charged against the budget
    /// before it is cloned and given back when the plan is dropped or committed.
    #[cfg(any(test, feature = "test-support"))]
    pub fn append_plan(&self, packed_rows: usize) -> Result<BuilderAppendPlan, BuilderError> {
        let g = self.g.borrow();
        let site = AllocationSite::PlanSnapshot;
        let mut snapshot = times(
            &self.meter,
            site,
            g.values().len(),
            std::mem::size_of::<TensorType>(),
        )?;
        for meta in g.values().iter() {
            snapshot = plus(
                &self.meter,
                site,
                snapshot,
                times(
                    &self.meter,
                    site,
                    meta.aval.shape.len(),
                    std::mem::size_of::<usize>(),
                )?,
            )?;
        }
        let mut held = HeldCharge::new(Rc::clone(&self.meter));
        let snapshot_charge = Charge::owned(snapshot);
        held.reserve(snapshot_charge, site)?;
        let mut base_types = Vec::new();
        reserve(g.values().len(), BuilderCollection::Values, |n| {
            base_types.try_reserve_exact(n)
        })?;
        for meta in g.values().iter() {
            let mut shape = Vec::new();
            reserve(meta.aval.shape.len(), BuilderCollection::Values, |n| {
                shape.try_reserve_exact(n)
            })?;
            shape.extend_from_slice(&meta.aval.shape);
            base_types.push(TensorType {
                shape,
                dtype: meta.aval.dtype,
            });
        }
        Ok(BuilderAppendPlan {
            brand: self.brand,
            generation: self.generation.get(),
            base_value_count: g.values().len(),
            base_types: base_types.into_iter().collect(),
            values: Vec::new(),
            equations: Vec::new(),
            result: None,
            packed_rows,
            checked_name_bytes: 0,
            layer: self.layer.get(),
            held,
            snapshot: snapshot_charge,
        })
    }

    /// Validate an entire staged fragment without changing the live graph.
    #[cfg(any(test, feature = "test-support"))]
    pub fn preflight_append(
        &self,
        plan: BuilderAppendPlan,
    ) -> Result<PreparedBuilderAppend, BuilderError> {
        if plan.brand != self.brand {
            return Err(BuilderError::ForeignBuilder);
        }
        let g = self.g.borrow();
        let actual_generation = self.generation.get();
        if plan.generation != actual_generation || plan.base_value_count != g.values().len() {
            return Err(BuilderError::StalePlan {
                expected: plan.generation,
                actual: actual_generation,
            });
        }

        checked_growth(
            BuilderCollection::Values,
            g.values().len(),
            plan.values.len(),
        )?;
        checked_growth(
            BuilderCollection::Equations,
            g.eqns().len(),
            plan.equations.len(),
        )?;

        // The tables this validation keeps: a defined flag per staged value and a name map entry per
        // staged name. Transient: given back when the validation ends.
        let mut tables = HeldCharge::new(Rc::clone(&self.meter));
        let table_bytes = plus(
            &self.meter,
            AllocationSite::PreflightTables,
            times(
                &self.meter,
                AllocationSite::PreflightTables,
                plan.values.len(),
                std::mem::size_of::<bool>(),
            )?,
            times(
                &self.meter,
                AllocationSite::PreflightTables,
                plan.values.len(),
                std::mem::size_of::<(&str, BuilderValueNamespace)>() + std::mem::size_of::<u64>(),
            )?,
        )?;
        tables.reserve(Charge::owned(table_bytes), AllocationSite::PreflightTables)?;

        let mut staged_inputs = 0usize;
        let mut staged_constants = 0usize;
        let mut staged_slots = 0usize;
        let mut staged_names: HashMap<&str, BuilderValueNamespace> = HashMap::new();
        reserve(plan.values.len(), BuilderCollection::NameBytes, |n| {
            staged_names.try_reserve(n)
        })?;
        for (offset, meta) in plan.values.iter().enumerate() {
            let value = plan.staged_id(offset)?;
            let namespace = staged_namespace(&plan, value, meta);
            if meta.storage != Storage::Device {
                staged_inputs = staged_inputs
                    .checked_add(1)
                    .ok_or(BuilderError::SizeOverflow {
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
                    return Err(BuilderError::EmptyName { namespace });
                }
                if let Some(existing) = staged_names.insert(name, namespace) {
                    return Err(BuilderError::NameCollision {
                        name: name.to_string(),
                        requested: namespace,
                        existing,
                    });
                }
                if let Some(existing) = live_name_namespace(&g, name) {
                    return Err(BuilderError::NameCollision {
                        name: name.to_string(),
                        requested: namespace,
                        existing,
                    });
                }
            }
        }
        checked_growth(BuilderCollection::Inputs, g.inputs().len(), staged_inputs)?;
        checked_growth(
            BuilderCollection::Constants,
            g.consts().len(),
            staged_constants,
        )?;
        checked_growth(BuilderCollection::Slots, g.slots().len(), staged_slots)?;

        let total_values = g.values().len() + plan.values.len();
        let live_values = g.values().len();
        let mut defined: Vec<bool> = Vec::new();
        reserve(plan.values.len(), BuilderCollection::Values, |n| {
            defined.try_reserve_exact(n)
        })?;
        defined.extend(
            plan.values
                .iter()
                .map(|meta| meta.storage != Storage::Device),
        );
        for (equation, eqn) in plan.equations.iter().enumerate() {
            let operation = eqn.op.name();
            for operand in &eqn.inputs {
                let Operand::Value(value) = operand else {
                    continue;
                };
                if g.values().contains(*value) {
                    continue;
                }
                if value.index() >= total_values {
                    return Err(BuilderError::InvalidOperand {
                        equation,
                        operation,
                        value: *value,
                    });
                }
                if !defined[value.index() - live_values] {
                    return Err(BuilderError::OperandNotDefined {
                        equation,
                        operation,
                        value: *value,
                    });
                }
            }
            let mut scratch = HeldCharge::new(Rc::clone(&self.meter));
            let staged_type = |value: ValueId| {
                if g.values().contains(value) {
                    g.values().get(value).map(|meta| &meta.aval)
                } else {
                    value
                        .index()
                        .checked_sub(live_values)
                        .and_then(|offset| plan.values.get(offset))
                        .map(|meta| &meta.aval)
                }
            };
            let price = price_equation(
                &self.meter,
                AllocationSite::StagedInference,
                &eqn.op,
                &eqn.inputs,
                staged_type,
            )
            .map_err(|refusal| refusal.into_error(equation, operation.clone()))?;
            if eqn.out.index() < live_values || eqn.out.index() >= total_values {
                return Err(BuilderError::InvalidOperand {
                    equation,
                    operation,
                    value: eqn.out,
                });
            }
            let output_offset = eqn.out.index() - live_values;
            if defined[output_offset] {
                return Err(BuilderError::ValueRedefined {
                    equation,
                    operation,
                    value: eqn.out,
                });
            }
            let output = &plan.values[output_offset];
            if output.storage != Storage::Device {
                return Err(BuilderError::EquationResultStorage {
                    value: eqn.out,
                    storage: output.storage,
                });
            }
            scratch.reserve(
                Charge::owned(price.scratch),
                AllocationSite::StagedInference,
            )?;
            let in_types = clone_operand_types(&eqn.op, &eqn.inputs, equation, staged_type)?;
            let inferred = infer_equation(&eqn.op, &eqn.inputs, &in_types, equation)?;
            if inferred != output.aval {
                return Err(BuilderError::OutputTypeMismatch {
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
                return Err(BuilderError::MissingDefinition {
                    value: plan.staged_id(offset)?,
                });
            }
        }
        let result = validate_append_result(&plan, g.values().len(), &defined)?;

        let accounting = BuilderAppendAccounting {
            staged_values: plan.values.len(),
            staged_equations: plan.equations.len(),
            packed_rows: plan.packed_rows,
            checked_name_bytes: plan.checked_name_bytes,
        };
        Ok(PreparedBuilderAppend {
            brand: plan.brand,
            generation: actual_generation,
            base_value_count: g.values().len(),
            accounting,
            staged_inputs,
            staged_constants,
            staged_slots,
            result,
            plan: Some(plan),
        })
    }

    /// Reserve all live collections and publish a preflighted fragment as one generation. The result is
    /// the handle of the value the plan declared as its result.
    ///
    /// The plan's staged storage was charged when it was staged; it becomes the graph's, and only the
    /// type snapshot is given back. A refused commit changes neither the graph nor the generation, and
    /// the plan keeps every charge it holds.
    #[cfg(any(test, feature = "test-support"))]
    pub fn commit_append(
        &self,
        prepared: &mut PreparedBuilderAppend,
    ) -> Result<Traced, BuilderError> {
        if prepared.brand != self.brand {
            return Err(BuilderError::ForeignBuilder);
        }
        if prepared.plan.is_none() {
            return Err(BuilderError::ConsumedPlan);
        }
        let actual_generation = self.generation.get();
        if prepared.generation != actual_generation {
            return Err(BuilderError::StalePlan {
                expected: prepared.generation,
                actual: actual_generation,
            });
        }
        let next_generation = actual_generation
            .checked_add(1)
            .ok_or(BuilderError::GenerationOverflow)?;
        let mut g = self.g.borrow_mut();
        // The value ids of the whole fragment fit before any of it is appended.
        if let Some(last) = prepared.accounting.staged_values.checked_sub(1)
            && g.values()
                .len()
                .checked_add(last)
                .and_then(ValueId::from_index)
                .is_none()
        {
            return Err(BuilderError::SizeOverflow {
                collection: BuilderCollection::Values,
                current: g.values().len(),
                additional: prepared.accounting.staged_values,
            });
        }
        if prepared.base_value_count != g.values().len() {
            return Err(BuilderError::StalePlan {
                expected: prepared.generation,
                actual: actual_generation,
            });
        }
        self.reserve_collection(
            prepared.accounting.staged_values,
            BuilderCollection::Values,
            |n| g.values_mut().try_reserve(n),
        )?;
        self.reserve_collection(prepared.staged_inputs, BuilderCollection::Inputs, |n| {
            g.inputs_mut().try_reserve(n)
        })?;
        self.reserve_collection(
            prepared.staged_constants,
            BuilderCollection::Constants,
            |n| g.consts_mut().try_reserve(n),
        )?;
        self.reserve_collection(prepared.staged_slots, BuilderCollection::Slots, |n| {
            g.slots_mut().try_reserve(n)
        })?;
        self.reserve_collection(
            prepared.accounting.staged_equations,
            BuilderCollection::Equations,
            |n| g.eqns_mut().try_reserve(n),
        )?;

        // Presence was checked before reservation; keep the invariant typed if this code is rearranged, and leave the
        // plan reusable when an earlier reservation fails.
        let plan = prepared.plan.take().ok_or(BuilderError::ConsumedPlan)?;
        let result = prepared.result.value();
        let BuilderAppendPlan {
            values,
            equations,
            mut held,
            snapshot,
            ..
        } = plan;
        for meta in values {
            let storage = meta.storage;
            let id = g.values_mut().push(meta);
            match storage {
                Storage::Const | Storage::Computed(_) | Storage::State => {
                    g.inputs_mut().push(id);
                    g.consts_mut().push(id);
                }
                Storage::Slot(slot) => {
                    g.inputs_mut().push(id);
                    g.slots_mut().push((id, slot));
                }
                Storage::Device => {}
            }
        }
        g.eqns_mut().extend(equations);
        // The snapshot dies with the plan; the staged values and equations are the graph's now.
        held.release(snapshot);
        held.keep();
        self.generation.set(next_generation);
        Ok(self.handle(result))
    }

    fn reserve_collection(
        &self,
        additional: usize,
        collection: BuilderCollection,
        try_reserve: impl FnOnce(usize) -> Result<(), std::collections::TryReserveError>,
    ) -> Result<(), BuilderError> {
        #[cfg(any(test, feature = "test-support"))]
        if self.reservation_failure.get() == Some(collection) {
            self.reservation_failure.set(None);
            return Err(BuilderError::Reserve {
                collection,
                additional,
            });
        }
        reserve(additional, collection, try_reserve)
    }

    /// Charge one input binder, reserve the tables it joins and append it. `rank`, `name_len` and
    /// `extra_bytes` (a slot tag's copy) describe the storage `make_meta` allocates; the whole is
    /// reserved before `make_meta` runs, so a refusal precedes the name's allocation.
    fn append_input(
        &self,
        site: AllocationSite,
        rank: usize,
        name_len: usize,
        extra_bytes: u64,
        table: InputTable,
        make_meta: impl FnOnce() -> ValueMeta,
    ) -> Result<ValueId, BuilderError> {
        let next_generation = self
            .generation
            .get()
            .checked_add(1)
            .ok_or(BuilderError::GenerationOverflow)?;
        let record = value_record_bytes(&self.meter, site, rank, name_len)?;
        let entries = std::mem::size_of::<ValueId>()
            + match table {
                InputTable::Const => std::mem::size_of::<ValueId>(),
                InputTable::Slot(_) => std::mem::size_of::<(ValueId, Slot)>(),
            };
        let owned = plus(
            &self.meter,
            site,
            plus(&self.meter, site, record, extra_bytes)?,
            entries as u64,
        )?;
        let mut held = HeldCharge::new(Rc::clone(&self.meter));
        held.reserve(
            Charge {
                values: 1,
                owned,
                ..Charge::default()
            },
            site,
        )?;
        let id = {
            let mut g = self.g.borrow_mut();
            self.reserve_collection(1, BuilderCollection::Values, |n| {
                g.values_mut().try_reserve(n)
            })?;
            self.reserve_collection(1, BuilderCollection::Inputs, |n| {
                g.inputs_mut().try_reserve(n)
            })?;
            match table {
                InputTable::Const => {
                    self.reserve_collection(1, BuilderCollection::Constants, |n| {
                        g.consts_mut().try_reserve(n)
                    })?
                }
                InputTable::Slot(_) => {
                    self.reserve_collection(1, BuilderCollection::Slots, |n| {
                        g.slots_mut().try_reserve(n)
                    })?
                }
            }
            let current = g.values().len();
            let id =
                g.values_mut()
                    .try_push(make_meta())
                    .map_err(|_| BuilderError::SizeOverflow {
                        collection: BuilderCollection::Values,
                        current,
                        additional: 1,
                    })?;
            g.inputs_mut().push(id);
            match table {
                InputTable::Const => g.consts_mut().push(id),
                InputTable::Slot(slot) => g.slots_mut().push((id, slot)),
            }
            id
        };
        self.generation.set(next_generation);
        held.keep();
        Ok(id)
    }

    // input binders

    /// A constant input (weight / table), bound at graph close. Not per-token.
    pub fn constant(&self, name: &str, aval: TensorType) -> Result<Traced, BuilderError> {
        self.declare_constant(name, aval, None, None)
    }

    /// A model weight input, bound at graph close: the constant `weight` names, at `aval`. Its
    /// [`WeightRef`] is recorded on the value ([`ValueMeta::weight`]), and its name is the
    /// reference's spelling ([`WeightRef::name`]).
    pub fn weight(&self, weight: WeightRef, aval: TensorType) -> Result<Traced, BuilderError> {
        self.declare_constant(&weight.name(), aval, Some(weight), None)
    }

    /// [`Builder::constant`] for an I32 table whose words are declared to lie in `range`: a hash
    /// router's token-to-experts table in `[0, experts)`.
    ///
    /// The declaration is a fact the graph's consumers rely on ([`crate::value_ranges`]: a `Gather`
    /// of the table inherits it, so an `I32 -> F32` cast of a gathered word is proven exact), so the
    /// weight binder checks every stored word against it and refuses a weight with a word outside it.
    ///
    /// # Errors
    /// [`BuilderError::RangeNotI32`] when `aval` is not I32: a range is a fact about integer words.
    pub fn constant_ranged(
        &self,
        name: &str,
        aval: TensorType,
        range: IndexRange,
    ) -> Result<Traced, BuilderError> {
        if aval.dtype != DType::I32 {
            return Err(BuilderError::RangeNotI32 {
                name: name.to_string(),
                dtype: aval.dtype,
            });
        }
        self.declare_constant(name, aval, None, Some(range))
    }

    fn declare_constant(
        &self,
        name: &str,
        aval: TensorType,
        weight: Option<WeightRef>,
        range: Option<IndexRange>,
    ) -> Result<Traced, BuilderError> {
        let id = self.append_input(
            AllocationSite::InputRecord,
            aval.shape.len(),
            name.len(),
            0,
            InputTable::Const,
            || ValueMeta {
                aval,
                storage: Storage::Const,
                name: Some(name.to_string()),
                key: None,
                state_role: None,
                weight,
                range,
            },
        )?;
        Ok(self.handle(id))
    }

    /// A constant the binder materializes from its own payload (a derived table such as RoPE's):
    /// no name, no weight. Declared in [`ComputedConst::dtype`] and [`ComputedConst::shape`].
    pub fn computed(&self, computed: ComputedConst) -> Result<Traced, BuilderError> {
        let id = self.append_input(
            AllocationSite::InputRecord,
            computed.rank(),
            0,
            0,
            InputTable::Const,
            || ValueMeta {
                aval: TensorType::new(computed.shape(), computed.dtype()),
                storage: Storage::Computed(computed),
                name: None,
                key: None,
                state_role: None,
                weight: None,
                range: None,
            },
        )?;
        Ok(self.handle(id))
    }

    /// A per-token-varying input (token / pos / seq_len), routed through a stable device buffer.
    pub fn slot(&self, slot: Slot, aval: TensorType) -> Result<Traced, BuilderError> {
        self.declare_slot(slot, None, aval, None)
    }

    /// [`Builder::slot`] for an I32 input whose words are declared to lie in `range`: `Token` in
    /// `[0, vocab)`, `Pos` in `[0, capacity)`.
    ///
    /// The declaration is a fact the graph's consumers rely on ([`crate::value_ranges`]: an
    /// `I32 -> F32` cast of the slot is proven exact from it), so every binder checks the bound words
    /// against it and refuses a step with a word outside it.
    ///
    /// # Errors
    /// [`BuilderError::RangeNotI32`] when `aval` is not I32.
    pub fn slot_ranged(
        &self,
        slot: Slot,
        aval: TensorType,
        range: IndexRange,
    ) -> Result<Traced, BuilderError> {
        self.declare_slot(slot, None, aval, Some(range))
    }

    /// A runtime input whose name carries a caller-supplied disambiguator, so one graph can hold several inputs of
    /// the same [`Slot`] kind with different per-call content (spec 266: one expert-pool assignment table per MoE
    /// layer; spec 149: a named standalone activation). [`Builder::slot`] names every occurrence of a kind
    /// identically ([`Slot::key_label`], R466-013: a hand-written label, never `Debug`), which model-engine binders
    /// rely on to resolve a whole `Slot` kind from one caller-supplied buffer (see [`Slot::ExpertPoolMap`]). A named
    /// input has no such implicit buffer. The name is `"{kind}.{tag}"` (built by [`crate::graph::SlotKey`]), and its
    /// binder must dispatch on [`crate::graph::ValueMeta::name`] as the `Const` path does, not on the kind alone.
    ///
    /// `tag` must be non-empty (an empty tag would stage the untagged label plus a bare trailing
    /// `.`, indistinguishable in spelling from a typo) and unique within the graph for a given kind:
    /// [`BuilderError::EmptySlotTag`] and [`BuilderError::DuplicateSlotTag`], the same rejection
    /// [`Builder::exact_i32_slot`] gives a caller-supplied empty tag.
    pub fn slot_named(
        &self,
        slot: Slot,
        tag: &str,
        aval: TensorType,
    ) -> Result<Traced, BuilderError> {
        if tag.is_empty() {
            return Err(BuilderError::EmptySlotTag { slot });
        }
        self.declare_slot(slot, Some(tag), aval, None)
    }

    /// [`Builder::slot_named`] for an I32 input with a declared `range` ([`Builder::slot_ranged`]).
    pub fn slot_named_ranged(
        &self,
        slot: Slot,
        tag: &str,
        aval: TensorType,
        range: IndexRange,
    ) -> Result<Traced, BuilderError> {
        if tag.is_empty() {
            return Err(BuilderError::EmptySlotTag { slot });
        }
        self.declare_slot(slot, Some(tag), aval, Some(range))
    }

    fn declare_slot(
        &self,
        slot: Slot,
        tag: Option<&str>,
        aval: TensorType,
        range: Option<IndexRange>,
    ) -> Result<Traced, BuilderError> {
        if range.is_some() && aval.dtype != DType::I32 {
            return Err(BuilderError::RangeNotI32 {
                name: slot.key_label().to_string(),
                dtype: aval.dtype,
            });
        }
        let label = slot.key_label();
        if let Some(tag) = tag
            && self.g.borrow().values().iter().any(|v| {
                v.storage == Storage::Slot(slot)
                    && v.name.as_deref().is_some_and(|name| {
                        name.strip_prefix(label)
                            .and_then(|rest| rest.strip_prefix('.'))
                            == Some(tag)
                    })
            })
        {
            return Err(BuilderError::DuplicateSlotTag {
                slot,
                tag: tag.to_string(),
            });
        }
        // `SlotKey` spells `"{label}"` or `"{label}.{tag}"`.
        let tag_len = tag.map_or(0, str::len);
        let name_len = label.len() + tag.map_or(0, |_| 1 + tag_len);
        let id = self.append_input(
            AllocationSite::InputRecord,
            aval.shape.len(),
            name_len,
            tag_len as u64,
            InputTable::Slot(slot),
            || {
                let key = SlotKey::new(slot, tag);
                ValueMeta {
                    aval,
                    storage: Storage::Slot(slot),
                    name: Some(key.to_string()),
                    key: Some(key),
                    state_role: None,
                    weight: None,
                    range,
                }
            },
        )?;
        Ok(self.handle(id))
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
    /// # use poot_graph_ir::TraceLimits;
    /// # use std::num::{NonZeroU64, NonZeroUsize};
    /// # let n = NonZeroUsize::MIN;
    /// # let limits = TraceLimits { max_tokens: n, max_values: n, max_eqns: n, max_literal_bytes: NonZeroU64::MIN, max_owned_bytes: NonZeroU64::MIN };
    ///
    /// let b = Builder::new(limits);
    /// let _cache = b.state_input("cache", TensorType::f32([1]));
    /// ```
    pub fn state_input(
        &self,
        name: &str,
        aval: TensorType,
        role: StateRole,
    ) -> Result<Traced, BuilderError> {
        let id = self.append_input(
            AllocationSite::InputRecord,
            aval.shape.len(),
            name.len(),
            0,
            InputTable::Const,
            || ValueMeta {
                aval,
                storage: Storage::State,
                name: Some(name.to_string()),
                key: None,
                state_role: Some(role),
                weight: None,
                range: None,
            },
        )?;
        Ok(self.handle(id))
    }

    // primitive emitters
    //
    // Each infers the output type host-side and appends an eqn. An infer failure is a trace-time error
    // (incompatible shapes), returned as a typed `BuilderError::Inference` like `abstract_eval` in JAX; the
    // typed `ShapeError` is what `OpKind::infer` returns.

    /// Charge one equation, infer its type and append it. Every request is reserved before the
    /// storage it covers is allocated or appended; a refused or failed equation changes nothing.
    pub(crate) fn emit(&self, op: OpKind, operands: Vec<Operand>) -> Result<Traced, BuilderError> {
        let equation = self.g.borrow().eqns().len();
        let site = AllocationSite::EquationRecord;
        let next_generation = self
            .generation
            .get()
            .checked_add(1)
            .ok_or(BuilderError::GenerationOverflow)?;
        let price = {
            let g = self.g.borrow();
            price_equation(&self.meter, site, &op, &operands, |value| {
                g.values().get(value).map(|meta| &meta.aval)
            })
        }
        .map_err(|refusal| refusal.into_error(equation, op.name()))?;
        let record = eqn_record_bytes(&self.meter, site, &op, operands.len())?;
        let value_record = value_record_bytes(&self.meter, site, 0, 0)?;
        let out_bound = times(
            &self.meter,
            AllocationSite::InferenceResult,
            price.rank_bound,
            std::mem::size_of::<usize>(),
        )?;
        let mut held = HeldCharge::new(Rc::clone(&self.meter));
        held.reserve(
            Charge {
                values: 1,
                eqns: 1,
                literal: price.literal,
                owned: plus(&self.meter, site, record, value_record)?,
            },
            site,
        )?;
        held.reserve(
            Charge::owned(price.scratch),
            AllocationSite::InferenceOperands,
        )?;
        held.reserve(Charge::owned(out_bound), AllocationSite::InferenceResult)?;
        let aval = {
            let g = self.g.borrow();
            let in_types = clone_operand_types(&op, &operands, equation, |value| {
                g.values().get(value).map(|meta| &meta.aval)
            })?;
            infer_equation(&op, &operands, &in_types, equation)?
        };
        // Settle the scratch: the operand clones are gone and the output keeps its actual shape list,
        // not the bound.
        let actual = times(
            &self.meter,
            AllocationSite::InferenceResult,
            aval.shape.len(),
            std::mem::size_of::<usize>(),
        )?;
        held.release(Charge::owned(
            price
                .scratch
                .saturating_add(out_bound.saturating_sub(actual)),
        ));
        if actual > out_bound {
            held.reserve(
                Charge::owned(actual - out_bound),
                AllocationSite::InferenceResult,
            )?;
        }
        let out = {
            let mut g = self.g.borrow_mut();
            self.reserve_collection(1, BuilderCollection::Values, |n| {
                g.values_mut().try_reserve(n)
            })?;
            self.reserve_collection(1, BuilderCollection::Equations, |n| {
                g.eqns_mut().try_reserve(n)
            })?;
            let out = g
                .values_mut()
                .try_push(ValueMeta {
                    aval,
                    storage: Storage::Device,
                    name: None,
                    key: None,
                    state_role: None,
                    weight: None,
                    range: None,
                })
                .map_err(|_| BuilderError::SizeOverflow {
                    collection: BuilderCollection::Values,
                    current: g.values().len(),
                    additional: 1,
                })?;
            g.eqns_mut().push(Eqn {
                op,
                inputs: operands,
                out,
                layer: self.current_layer(),
            });
            out
        };
        self.generation.set(next_generation);
        held.keep();
        Ok(self.handle(out))
    }

    /// Elementwise unary `op` over `x`; the output has `x`'s type.
    pub fn unary(&self, op: crate::op::UnOp, x: Traced) -> Result<Traced, BuilderError> {
        self.emit(OpKind::Unary(op), vec![self.operand(x)?])
    }

    /// Elementwise binary `op` over two values with numpy broadcasting; both operands must share one dtype.
    pub fn binary(
        &self,
        op: crate::op::BinOp,
        a: Traced,
        b: Traced,
    ) -> Result<Traced, BuilderError> {
        // `OpKind::Binary::infer` takes its output dtype from `ins[0]` alone and cannot distinguish this two-value call
        // from `binary_scalar` (where a differing literal dtype is legitimate, baked as an immediate). Here both
        // operands are real buffers read by the planner's broadcast-value kernel at one element type (`fty(odt)`); a
        // dtype mismatch would reinterpret the other operand's bytes at the wrong width, so `emit` checks it for a
        // binary equation of two values (R466-012, the one shared rule: `crate::op::check_binary_value_dtypes`).
        self.emit(OpKind::Binary(op), vec![self.operand(a)?, self.operand(b)?])
    }

    /// Pack int8 codes into i32 words, 4 per word along the last axis (spec 048). Output is `I32` with last dim
    /// `ceil(L/4)`.
    #[cfg(any(test, feature = "test-support"))]
    pub fn pack_i8(&self, x: Traced) -> Result<Traced, BuilderError> {
        self.emit(OpKind::PackI8, vec![self.operand(x)?])
    }

    /// Unpack `len` int8 codes from i32 words (inverse of [`Builder::pack_i8`]). Output is `F32` with last dim `len`.
    #[cfg(any(test, feature = "test-support"))]
    pub fn unpack_i8(&self, codes: Traced, len: usize) -> Result<Traced, BuilderError> {
        self.emit(OpKind::UnpackI8 { len }, vec![self.operand(codes)?])
    }

    /// Binary op with an inline scalar literal as the second operand (broadcasts as a 0-d value).
    pub fn binary_scalar(
        &self,
        op: crate::op::BinOp,
        a: Traced,
        s: Scalar,
    ) -> Result<Traced, BuilderError> {
        self.emit(OpKind::Binary(op), vec![self.operand(a)?, Operand::Lit(s)])
    }

    /// `Select(cond, if_true, if_false)`: the bit pattern of `if_true` where the I32 condition word is
    /// nonzero, else of `if_false`. The branches share one I32 or float dtype. A true select, so a NaN or
    /// Inf in the unselected branch cannot reach the result.
    ///
    /// Any nonzero condition word picks `if_true`, not only 1. Audited at POOT-1033: no production caller
    /// relied on a condition outside {0, 1} (`index_guard`, `ple`'s equality flag, the bf16 binary cond and
    /// the exact-I32 fixtures all feed `GeU`/`Ge`/`And` flags); two tests that pinned the old arithmetic on a
    /// wider condition word were re-pinned to this definition.
    pub fn select(
        &self,
        cond: Traced,
        if_true: Traced,
        if_false: Traced,
    ) -> Result<Traced, BuilderError> {
        self.emit(
            OpKind::Select,
            vec![
                self.operand(cond)?,
                self.operand(if_true)?,
                self.operand(if_false)?,
            ],
        )
    }

    /// Reduce `x` over `axis` with `op`. `keepdim` keeps the reduced axis as extent 1 instead of removing it.
    pub fn reduce(
        &self,
        op: crate::op::RedOp,
        x: Traced,
        axis: usize,
        keepdim: bool,
    ) -> Result<Traced, BuilderError> {
        self.emit(OpKind::Reduce { op, axis, keepdim }, vec![self.operand(x)?])
    }

    /// Broadcast `x` to `shape` (numpy semantics: trailing axes align, extent-1 axes expand).
    pub fn broadcast(&self, x: Traced, shape: Vec<usize>) -> Result<Traced, BuilderError> {
        self.emit(OpKind::Broadcast { shape }, vec![self.operand(x)?])
    }

    /// The F32 range `[0, 1, .., len)`, shape `[len]` (R466-019): a computed range, not a bound constant.
    /// The compiler folds it into a [`crate::Storage::Computed`] graph constant before planning.
    pub fn iota(&self, len: usize) -> Result<Traced, BuilderError> {
        self.emit(
            OpKind::Iota {
                len,
                dtype: DType::F32,
            },
            Vec::new(),
        )
    }

    /// The exact I32 range `[0, 1, .., len)`, shape `[len]`: the integer sibling of [`Self::iota`], for
    /// positions and bounds that must not round-trip through F32.
    pub fn iota_i32(&self, len: usize) -> Result<Traced, BuilderError> {
        self.emit(
            OpKind::Iota {
                len,
                dtype: DType::I32,
            },
            Vec::new(),
        )
    }

    /// A recurrence scan (POOT-634): ONE `OpKind::Scan` equation over `operands` (the family's operand
    /// order), returned as its three typed values. The two projections are slices of the packed result
    /// at the rows its [`crate::ScanLayout`] names.
    pub fn scan(
        &self,
        attrs: crate::op::ScanAttrs,
        operands: &[Traced],
    ) -> Result<ScanResult, BuilderError> {
        let operands = operands
            .iter()
            .map(|&t| self.operand(t))
            .collect::<Result<Vec<_>, _>>()?;
        let op = OpKind::Scan(attrs);
        let equation = self.g.borrow().eqns().len();
        // The layout reads the operand types: charge their clone like any inference scratch.
        let price = {
            let g = self.g.borrow();
            price_equation(
                &self.meter,
                AllocationSite::InferenceOperands,
                &op,
                &operands,
                |value| g.values().get(value).map(|meta| &meta.aval),
            )
        }
        .map_err(|refusal| refusal.into_error(equation, op.name()))?;
        let mut scratch = HeldCharge::new(Rc::clone(&self.meter));
        scratch.reserve(
            Charge::owned(price.scratch),
            AllocationSite::InferenceOperands,
        )?;
        let layout = {
            let g = self.g.borrow();
            let in_types = clone_operand_types(&op, &operands, equation, |value| {
                g.values().get(value).map(|meta| &meta.aval)
            })?;
            attrs
                .layout(&in_types)
                .map_err(|source| BuilderError::Inference {
                    equation,
                    operation: op.name(),
                    source,
                })?
        };
        drop(scratch);
        let packed = self.emit(op, operands)?;
        let (y, s_out) = project_scan(self, packed, &layout)?;
        Ok(ScanResult { packed, y, s_out })
    }

    /// The block selection of a sparse-attention indexer (POOT-1143): ONE `OpKind::BlockTopK` equation over
    /// `scores` `[R, 1, T, NB]` and the queries' positions `pos` `[R, T]`, returned as its typed
    /// values. The two projections are slices of the packed result at the columns its
    /// [`crate::BlockTopKLayout`] names.
    pub fn block_top_k(
        &self,
        attrs: crate::op::BlockTopKAttrs,
        scores: Traced,
        pos: Traced,
    ) -> Result<BlockTopKResult, BuilderError> {
        let layout = {
            let (scores, pos) = (self.aval(scores)?, self.aval(pos)?);
            attrs
                .layout(&[scores, pos])
                .map_err(|source| BuilderError::Inference {
                    equation: self.g.borrow().eqns().len(),
                    operation: OpKind::BlockTopK(attrs).name(),
                    source,
                })?
        };
        let packed = self.emit(
            OpKind::BlockTopK(attrs),
            vec![self.operand(scores)?, self.operand(pos)?],
        )?;
        let (ids, keep) = (layout.ids(), layout.keep());
        Ok(BlockTopKResult {
            packed,
            ids: self.slice(packed, 3, ids.start, ids.end)?,
            keep: self.slice(packed, 3, keep.start, keep.end)?,
        })
    }

    /// Attention over the blocks a selection made (POOT-1143): ONE `OpKind::SelectedAttention` equation over
    /// `q`, the cached `k` and `v`, the selected block `ids` `[B, 1, Lq, K]` and the queries' positions `pos`
    /// `[B, Lq]`. Output `[B, Hq, Lq, Dv]`.
    pub fn selected_attention(
        &self,
        attrs: crate::op::SelectedAttentionAttrs,
        q: Traced,
        k: Traced,
        v: Traced,
        ids: Traced,
        pos: Traced,
    ) -> Result<Traced, BuilderError> {
        self.emit(
            OpKind::SelectedAttention(attrs),
            vec![
                self.operand(q)?,
                self.operand(k)?,
                self.operand(v)?,
                self.operand(ids)?,
                self.operand(pos)?,
            ],
        )
    }

    /// Reinterpret `x` with `shape`, which must hold the same number of elements.
    pub fn reshape(&self, x: Traced, shape: Vec<usize>) -> Result<Traced, BuilderError> {
        self.emit(OpKind::Reshape { shape }, vec![self.operand(x)?])
    }

    /// Cast `x` to storage dtype `to` (same shape). f32 -> bf16 rounds to bf16 precision; a widening or same-dtype
    /// cast is the identity. Feeds bf16 operands into the tensor-core matmul (card 045).
    pub fn cast(&self, x: Traced, to: crate::types::DType) -> Result<Traced, BuilderError> {
        self.emit(OpKind::Cast { to }, vec![self.operand(x)?])
    }

    /// Permute `x`'s axes: output axis `i` is input axis `perm[i]`.
    pub fn transpose(&self, x: Traced, perm: Vec<usize>) -> Result<Traced, BuilderError> {
        self.emit(OpKind::Transpose { perm }, vec![self.operand(x)?])
    }

    /// Take `start..end` along `axis` of `x`, keeping the other axes whole.
    pub fn slice(
        &self,
        x: Traced,
        axis: usize,
        start: usize,
        end: usize,
    ) -> Result<Traced, BuilderError> {
        self.emit(OpKind::Slice { axis, start, end }, vec![self.operand(x)?])
    }

    /// Concatenate `parts` along `axis`; the parts agree on dtype and on every other axis.
    pub fn concat(&self, axis: usize, parts: &[Traced]) -> Result<Traced, BuilderError> {
        let mut operands = Vec::new();
        reserve(parts.len(), BuilderCollection::Values, |n| {
            operands.try_reserve_exact(n)
        })?;
        for &part in parts {
            operands.push(self.operand(part)?);
        }
        self.emit(OpKind::Concat { axis }, operands)
    }

    /// Gather rows of `data` along `axis` by `index`. A scalar index drops the axis (embedding / RoPE
    /// row); a vector index `[L]` replaces it with `L` (a whole prompt's embeddings).
    pub fn gather(&self, data: Traced, axis: usize, index: Traced) -> Result<Traced, BuilderError> {
        self.emit(
            OpKind::Gather { axis },
            vec![self.operand(data)?, self.operand(index)?],
        )
    }

    /// Batched argtop-k index extraction (spec 136): `rank[..,E] -> [..,k]` expert ids, `out[..,r]` = the index `i`
    /// with `rank[..,i] == r`. Generalizes `scatter(iota, rank)[0..k]` to leading dims and feeds `IndexedMatMul`'s
    /// per-row `idx` operand. The rank producer owns score ordering and tie-breaking; this op only inverts a
    /// permutation. See [`OpKind::ArgTopK`].
    pub fn arg_top_k(&self, rank: Traced, k: usize) -> Result<Traced, BuilderError> {
        self.emit(OpKind::ArgTopK { k }, vec![self.operand(rank)?])
    }

    /// Exact uniform noise from an integer seed (card 551a, R472-007): `seed` is any-shape I32; output
    /// is `seed`'s shape with one trailing `cols` axis appended. See [`OpKind::RandomUniform`] and
    /// [`crate::ops::sampling`].
    pub fn random_uniform(&self, seed: Traced, cols: usize) -> Result<Traced, BuilderError> {
        self.emit(OpKind::RandomUniform { cols }, vec![self.operand(seed)?])
    }

    /// Pick one token per row (card 551a, R472-007, R-551a-2): `rule` fixes the operand list (see
    /// [`SampleRule`]) - `noise`/`params` are required for every rule but `Greedy`, `top_k` only for the
    /// two `TopK` rules. A caller that passes the wrong combination for `rule` gets
    /// [`BuilderError::Inference`] from [`OpKind::infer`], the same contract every other primitive emitter
    /// gives. See [`crate::ops::sampling::sample_head`] for the composed suffix (noise generation included).
    pub fn sample_token(
        &self,
        rule: SampleRule,
        logits: Traced,
        noise: Option<Traced>,
        params: Option<Traced>,
        top_k: Option<Traced>,
    ) -> Result<Traced, BuilderError> {
        let mut operands = vec![self.operand(logits)?];
        for t in [noise, params, top_k].into_iter().flatten() {
            operands.push(self.operand(t)?);
        }
        self.emit(OpKind::SampleToken { rule }, operands)
    }

    /// Scatter rows of `src` along axis 0 by a `[N]` permutation `index` (inverse of axis-0 gather):
    /// `out[index[j], ..] = src[j, ..]`, same shape as `src`. See [`OpKind::Scatter`].
    pub fn scatter(&self, src: Traced, index: Traced) -> Result<Traced, BuilderError> {
        self.emit(
            OpKind::Scatter { axis: 0 },
            vec![self.operand(src)?, self.operand(index)?],
        )
    }

    /// Scatter-update: `out = base` with `src[t]` written at pool slot `slots[t]` (`-1`: row `t` writes
    /// nothing). `base [POOL,..rest]`, `src [N,..rest]`, `[N]` forward map `slots`. Writes the N named rows and
    /// passes the rest through: the scattered analog of `dynamic_update_slice`. See [`OpKind::ScatterUpdate`]
    /// and, for declaring the map, [`Builder::write_map`].
    pub fn scatter_update(
        &self,
        base: Traced,
        src: Traced,
        slots: Traced,
    ) -> Result<Traced, BuilderError> {
        self.emit(
            OpKind::ScatterUpdate,
            vec![
                self.operand(base)?,
                self.operand(src)?,
                self.operand(slots)?,
            ],
        )
    }

    /// Declare the forward write map `tag` of `rows` source rows over a pool of `pool_slots` slots: a
    /// [`Slot::WriteMap`] I32 `[rows]` input whose words are declared in `[-1, pool_slots)`, so the bound
    /// range is a checked fact. The input is bound as a [`crate::ForwardSlotMap`].
    pub fn write_map(
        &self,
        tag: &str,
        rows: usize,
        pool_slots: usize,
    ) -> Result<Traced, BuilderError> {
        let range = i64::try_from(pool_slots)
            .ok()
            .and_then(|pool| IndexRange::new(-1, pool))
            .ok_or(BuilderError::Requirement {
                operation: "write map",
                requirement: Requirement::AtLeastOne { what: "pool slots" },
            })?;
        self.slot_named_ranged(
            Slot::WriteMap,
            tag,
            TensorType::new(vec![rows], DType::I32),
            range,
        )
    }

    /// Declare the read map `tag` of a paged KV pool of `pool_slots` slots: a [`Slot::SlotMap`] I32
    /// `[rows, capacity]` input mapping row `r`'s logical position to the pool slot holding its K/V. Its
    /// words are declared in `[0, pool_slots)`, so the bound range is a checked fact: a word outside the pool
    /// is refused at the bind, and attention reads the pool through the map in place on the strength of it.
    pub fn read_map(
        &self,
        tag: &str,
        rows: usize,
        capacity: usize,
        pool_slots: usize,
    ) -> Result<Traced, BuilderError> {
        let range = IndexRange::below(pool_slots).ok_or(BuilderError::Requirement {
            operation: "read map",
            requirement: Requirement::AtLeastOne { what: "pool slots" },
        })?;
        self.slot_named_ranged(
            Slot::SlotMap,
            tag,
            TensorType::new(vec![rows, capacity], DType::I32),
            range,
        )
    }

    /// Gather with a scalar index (the common decode case): drops `axis`.
    pub fn gather_scalar(
        &self,
        data: Traced,
        axis: usize,
        index: Traced,
    ) -> Result<Traced, BuilderError> {
        self.gather(data, axis, index)
    }

    /// Write `update` into a copy of `operand` at the static index `index` along `axis` (the KV-cache slot write).
    /// The index is an inline literal, so the decode graph is pos-specialized.
    ///
    /// # Errors
    /// [`BuilderError::LiteralOutOfRange`] when `index` does not fit the `I32` literal.
    #[cfg(any(test, feature = "test-support"))]
    pub fn dynamic_update_slice(
        &self,
        operand: Traced,
        update: Traced,
        index: usize,
        axis: usize,
    ) -> Result<Traced, BuilderError> {
        let index =
            i32::try_from(index).map_err(|_| BuilderError::LiteralOutOfRange { value: index })?;
        self.emit(
            OpKind::DynamicUpdateSlice { axis },
            vec![
                self.operand(operand)?,
                self.operand(update)?,
                Operand::Lit(Scalar::I32(index)),
            ],
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
    ) -> Result<Traced, BuilderError> {
        self.emit(
            OpKind::DynamicUpdateSlice { axis },
            vec![
                self.operand(operand)?,
                self.operand(update)?,
                self.operand(index)?,
            ],
        )
    }

    /// Batched matrix product `a[.., M, K] x b[.., K, N] -> [.., M, N]`, batch dims broadcasting.
    pub fn matmul(&self, a: Traced, b: Traced) -> Result<Traced, BuilderError> {
        self.emit(OpKind::MatMul, vec![self.operand(a)?, self.operand(b)?])
    }

    /// Emit the backend-neutral packed-dequant semantic primitive. The compiler-only `PackedContraction`,
    /// `DenseContraction` and `DenseRowGather` have no builder method: they exist only so a recognizer can rewrite a
    /// composition the tracer already expressed.
    ///
    /// `sources` are the carrier constants in [`PackedWeight::sources`] role order: one for a block
    /// format (GGUF), one per planar operand otherwise (AWQ three, GPTQ four).
    pub fn packed_dequant(
        &self,
        sources: &[Traced],
        descriptor: PackedWeight,
    ) -> Result<Traced, BuilderError> {
        let mut operands = Vec::new();
        reserve(sources.len(), BuilderCollection::Values, |n| {
            operands.try_reserve_exact(n)
        })?;
        for &source in sources {
            operands.push(self.operand(source)?);
        }
        self.emit(OpKind::PackedDequant { descriptor }, operands)
    }

    /// Indexed matmul (card 088): `out[m,n] = sum_k x[m,k] * W[idx[m],k,n]`. `x [M,K]`, stacked weight `w [E,K,N]`,
    /// per-row expert ids `idx [M]` (f32) -> `[M,N]`. Each row contracts against its own expert `W[idx[m]]`: a
    /// gather-free MoE GEMM. See [`OpKind::IndexedMatMul`].
    pub fn indexed_matmul(
        &self,
        x: Traced,
        w: Traced,
        idx: Traced,
    ) -> Result<Traced, BuilderError> {
        self.emit(
            OpKind::IndexedMatMul,
            vec![self.operand(x)?, self.operand(w)?, self.operand(idx)?],
        )
    }

    /// AllReduce collective (tensor-parallel): `op`-reduces a same-shaped partial tensor across all ranks and
    /// returns the full result in the same shape. `axis` is the shard axis.
    ///
    /// A collective is a cut point between stage programs: `compile` refuses one inside a stage graph.
    pub fn all_reduce(
        &self,
        x: Traced,
        op: crate::op::RedOp,
        axis: usize,
    ) -> Result<Traced, BuilderError> {
        self.emit(OpKind::AllReduce { op, axis }, vec![self.operand(x)?])
    }

    /// AllGather collective (column-parallel matmul): each of `world` ranks holds a slice of the output along
    /// `axis`; the collective concatenates the slices, so the result's `axis` is `world` times `x`'s.
    ///
    /// `axis` is the gather axis (the output N/feature axis in column-parallel matmul). A collective is a cut
    /// point between stage programs: `compile` refuses one inside a stage graph.
    pub fn all_gather(
        &self,
        x: Traced,
        axis: usize,
        world: NonZeroU32,
    ) -> Result<Traced, BuilderError> {
        self.emit(OpKind::AllGather { axis, world }, vec![self.operand(x)?])
    }

    // convenience scalar makers (kept here so models do not import types directly).
    /// An F32 scalar literal for [`Builder::binary_scalar`] and the other literal-taking calls.
    pub fn f32(v: f32) -> Scalar {
        Scalar::F32(v)
    }
}

/// What a graph's own values, equations, literals and tables cost: the charge a resumed builder takes
/// over.
fn graph_charge(meter: &Meter, g: &Graph) -> Result<Charge, BuilderError> {
    let site = AllocationSite::ResumedGraph;
    let mut owned = 0u64;
    let mut literal = 0u64;
    for meta in g.values().iter() {
        let name_len = meta.name.as_ref().map_or(0, String::len)
            + meta.key.as_ref().map_or(0, |key| key.to_string().len());
        owned = plus(
            meter,
            site,
            owned,
            value_record_bytes(meter, site, meta.aval.shape.len(), name_len)?,
        )?;
    }
    for eqn in g.eqns() {
        owned = plus(
            meter,
            site,
            owned,
            eqn_record_bytes(meter, site, &eqn.op, eqn.inputs.len())?,
        )?;
        for operand in &eqn.inputs {
            if let Operand::Lit(value) = operand {
                literal = plus(meter, site, literal, value.byte_len())?;
            }
        }
    }
    let tables = times(
        meter,
        site,
        g.inputs().len() + g.consts().len(),
        std::mem::size_of::<ValueId>(),
    )?;
    let slots = times(
        meter,
        site,
        g.slots().len(),
        std::mem::size_of::<(ValueId, Slot)>(),
    )?;
    let outputs = times(
        meter,
        site,
        1 + g.witnesses().count() + g.exports().count() + g.state().count(),
        std::mem::size_of::<GraphOutput>(),
    )?;
    let owned = plus(
        meter,
        site,
        plus(meter, site, owned, tables)?,
        plus(meter, site, slots, outputs)?,
    )?;
    Ok(Charge {
        values: g.values().len() as u64,
        eqns: g.eqns().len() as u64,
        literal,
        owned,
    })
}

#[cfg(any(test, feature = "test-support"))]
impl Builder {
    /// A builder under [`TraceLimits::for_tests`]: for a fixture that is not about the limits.
    pub fn for_tests() -> Self {
        Builder::new(TraceLimits::for_tests())
    }

    /// What this trace has charged and not given back: the exact pre-existing charge a test records
    /// before a controlled allocation.
    pub fn charged(&self) -> crate::budget::Charged {
        self.meter.charged()
    }

    /// A reader of this trace's ledger that outlives the builder (a `finish` consumes it): every
    /// reservation the budget granted with the site that asked for it, and every charge given back.
    pub fn ledger(&self) -> crate::budget::Ledger {
        crate::budget::Ledger {
            meter: Rc::clone(&self.meter),
        }
    }
}

/// The three values of a traced scan: the packed result its one equation defines and the two
/// projections of it.
#[derive(Clone, Copy, Debug)]
pub struct ScanResult {
    /// The packed `[1, H, L + S, W]` result.
    pub packed: Traced,
    /// The sequence output `[1, H, L, W]`.
    pub y: Traced,
    /// The carried state `[1, H, S, W]`.
    pub s_out: Traced,
}

/// The values of a traced block selection: the packed result its one equation defines and the two
/// projections of it.
#[derive(Clone, Copy, Debug)]
pub struct BlockTopKResult {
    /// The packed I32 `[R, 1, T, K + NB]` result.
    pub packed: Traced,
    /// The selected block ids `[R, 1, T, K]`, ascending, `-1` after the last selected.
    pub ids: Traced,
    /// The `0`/`1` keep indicator of each block `[R, 1, T, NB]`.
    pub keep: Traced,
}

/// `(y, s_out)` as the slices of `packed` at the layout's rows.
pub(crate) fn project_scan(
    b: &Builder,
    packed: Traced,
    layout: &crate::op::ScanLayout,
) -> Result<(Traced, Traced), BuilderError> {
    let (y, s) = (layout.y_rows(), layout.state_rows());
    Ok((
        b.slice(packed, 2, y.start, y.end)?,
        b.slice(packed, 2, s.start, s.end)?,
    ))
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
        state: usize,
        generation: u64,
        names: Vec<Option<String>>,
        types: Vec<TensorType>,
        storage: Vec<Storage>,
    }

    fn state(builder: &Builder) -> State {
        let graph = builder.g.borrow();
        State {
            values: graph.values().len(),
            inputs: graph.inputs().len(),
            constants: graph.consts().len(),
            slots: graph.slots().len(),
            equations: graph.eqns().len(),
            state: graph.state().count(),
            generation: builder.generation(),
            names: graph
                .values()
                .iter()
                .map(|meta| meta.name.clone())
                .collect(),
            types: graph
                .values()
                .iter()
                .map(|meta| meta.aval.clone())
                .collect(),
            storage: graph.values().iter().map(|meta| meta.storage).collect(),
        }
    }

    /// A builder with one `[2]` F32 constant, `name`d; its value id is 0 in every builder, so a handle
    /// from one is a valid-looking id in another.
    fn builder_with_constant(name: &str) -> (Builder, Traced) {
        let b = Builder::for_tests();
        let x = b.constant(name, TensorType::f32(vec![2])).unwrap();
        (b, x)
    }

    /// SC-001: a handle used with the builder that did not mint it panics at first use. Both handles
    /// are value 0, so without the brand check the foreign one is silently read as the local value.
    #[test]
    fn a_foreign_handle_in_an_emitter_is_a_typed_error_and_appends_nothing() {
        let (a, foreign) = builder_with_constant("a");
        let (b, local) = builder_with_constant("b");
        assert_eq!(foreign.id(), local.id());
        drop(a);
        let before = state(&b);
        assert_eq!(
            b.binary(crate::op::BinOp::Add, local, foreign),
            Err(BuilderError::ForeignHandle {
                value: foreign.id()
            })
        );
        assert_eq!(state(&b), before);
    }

    #[test]
    fn a_foreign_handle_cannot_finish_a_graph() {
        let (a, foreign) = builder_with_constant("a");
        let (b, _local) = builder_with_constant("b");
        drop(a);
        assert_eq!(
            b.finish(foreign).err(),
            Some(BuilderError::ForeignHandle {
                value: foreign.id()
            })
        );
    }

    #[test]
    fn a_foreign_handle_has_no_aval() {
        let (a, foreign) = builder_with_constant("a");
        let (b, _local) = builder_with_constant("b");
        drop(a);
        assert_eq!(
            b.aval(foreign),
            Err(BuilderError::ForeignHandle {
                value: foreign.id()
            })
        );
    }

    fn valid_plan(builder: &Builder, input: Traced) -> Result<BuilderAppendPlan, BuilderError> {
        let mut plan = builder.append_plan(0).unwrap();
        let output = plan.equation(
            OpKind::Cast { to: DType::F32 },
            vec![Operand::Value(input.id)],
        )?;
        plan.declare_result(output)?;
        Ok(plan)
    }

    #[test]
    fn transactional_input_constant_can_be_result() -> Result<(), BuilderError> {
        let builder = Builder::for_tests();
        let mut plan = builder.append_plan(0).unwrap();
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
        assert_eq!(builder.commit_append(&mut prepared)?.id(), ValueId::new(0));

        let graph = builder.finish(result).unwrap();
        assert_eq!(graph.result(), ValueId::new(0));
        assert_eq!(graph.inputs(), vec![ValueId::new(0)]);
        assert_eq!(graph.consts(), vec![ValueId::new(0)]);
        assert!(graph.slots().is_empty());
        assert!(graph.eqns().is_empty());
        assert_eq!(graph.values()[ValueId::new(0)].storage, Storage::Const);
        assert_eq!(
            graph.values()[ValueId::new(0)].aval,
            TensorType::new(vec![2, 3], DType::BF16)
        );
        assert!(crate::test_support::verdict(&graph).is_ok());
        Ok(())
    }

    #[test]
    fn transactional_input_slot_can_be_result() -> Result<(), BuilderError> {
        let builder = Builder::for_tests();
        let mut plan = builder.append_plan(0).unwrap();
        let result = plan.input_result(
            "slot_map.route",
            TensorType::new(vec![4], DType::I8),
            Storage::Slot(Slot::SlotMap),
        )?;

        let mut prepared = builder.preflight_append(plan)?;
        assert_eq!(builder.commit_append(&mut prepared)?.id(), ValueId::new(0));

        let graph = builder.finish(result).unwrap();
        assert_eq!(graph.result(), ValueId::new(0));
        assert_eq!(graph.inputs(), vec![ValueId::new(0)]);
        assert!(graph.consts().is_empty());
        assert_eq!(graph.slots(), vec![(ValueId::new(0), Slot::SlotMap)]);
        assert!(graph.eqns().is_empty());
        assert_eq!(
            graph.values()[ValueId::new(0)].storage,
            Storage::Slot(Slot::SlotMap)
        );
        assert_eq!(
            graph.values()[ValueId::new(0)].aval,
            TensorType::new(vec![4], DType::I8)
        );
        assert!(crate::test_support::verdict(&graph).is_ok());
        Ok(())
    }

    #[test]
    fn transactional_input_result_rejections_are_atomic() -> Result<(), BuilderError> {
        {
            let builder = Builder::for_tests();
            let existing = builder
                .constant("existing", TensorType::f32(vec![1]))
                .unwrap();
            let mut plan = builder.append_plan(0).unwrap();
            plan.declare_result(existing)?;
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderError::InvalidResult { value }) if value == existing.id
            ));
            assert_eq!(state(&builder), before);
        }

        {
            let builder = Builder::for_tests();
            let mut plan = builder.append_plan(0).unwrap();
            plan.input("staged", TensorType::f32(vec![1]), Storage::Const)?;
            plan.declare_result(plan.handle(ValueId::new(u32::MAX)))?;
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderError::InvalidResult { value }) if value == ValueId::new(u32::MAX)
            ));
            assert_eq!(state(&builder), before);
        }

        {
            let builder = Builder::for_tests();
            let input = builder.constant("input", TensorType::f32(vec![1])).unwrap();
            let mut plan = builder.append_plan(0).unwrap();
            let output = plan.equation(
                OpKind::Cast { to: DType::F32 },
                vec![Operand::Value(input.id)],
            )?;
            plan.declare_result(output)?;
            let staged_values = plan.values.len();
            let checked_name_bytes = plan.checked_name_bytes;
            assert!(matches!(
                plan.input_result("late", TensorType::f32(vec![1]), Storage::Const),
                Err(BuilderError::ResultAlreadyDeclared { value }) if value == output.id
            ));
            assert_eq!(plan.values.len(), staged_values);
            assert_eq!(plan.checked_name_bytes, checked_name_bytes);
            assert!(matches!(
                plan.result,
                Some(DeclaredAppendResult::EquationOutput(value)) if value == output.id
            ));
        }

        {
            let foreign = Builder::for_tests();
            let mut foreign_plan = foreign.append_plan(0).unwrap();
            let foreign_input =
                foreign_plan.input("foreign", TensorType::f32(vec![1]), Storage::Const)?;

            let builder = Builder::for_tests();
            let mut plan = builder.append_plan(0).unwrap();
            let local_input = plan.input("local", TensorType::f32(vec![1]), Storage::Const)?;
            // Same numeric id, different builder: the brand, not the number, tells them apart.
            assert_eq!(foreign_input.id(), local_input.id());
            assert_eq!(
                plan.declare_result(foreign_input),
                Err(BuilderError::ForeignBuilder)
            );
            assert!(plan.result.is_none());
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderError::MissingResult)
            ));
            assert_eq!(state(&builder), before);
        }

        {
            let builder = Builder::for_tests();
            let mut plan = builder.append_plan(0).unwrap();
            let local_input = plan.input("local", TensorType::f32(vec![1]), Storage::Const)?;
            plan.declare_result(local_input)?;
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderError::InvalidResult { value }) if value == local_input.id
            ));
            assert_eq!(state(&builder), before);
        }

        {
            let builder = Builder::for_tests();
            let mut plan = builder.append_plan(0).unwrap();
            let value = plan.next_value_id()?;
            plan.values.push(ValueMeta {
                aval: TensorType::f32(vec![1]),
                storage: Storage::Device,
                name: None,
                key: None,
                state_role: None,
                weight: None,
                range: None,
            });
            plan.declare_result(plan.handle(value))?;
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderError::MissingDefinition { value: actual }) if actual == value
            ));
            assert_eq!(state(&builder), before);
        }

        {
            let origin = Builder::for_tests();
            let foreign = Builder::for_tests();
            let mut plan = origin.append_plan(0).unwrap();
            plan.input_result("origin", TensorType::f32(vec![1]), Storage::Const)?;
            let origin_before = state(&origin);
            let foreign_before = state(&foreign);
            assert!(matches!(
                foreign.preflight_append(plan),
                Err(BuilderError::ForeignBuilder)
            ));
            assert_eq!(state(&origin), origin_before);
            assert_eq!(state(&foreign), foreign_before);

            let mut plan = origin.append_plan(0).unwrap();
            plan.input_result("origin", TensorType::f32(vec![1]), Storage::Const)?;
            let mut prepared = origin.preflight_append(plan)?;
            assert_eq!(
                foreign.commit_append(&mut prepared),
                Err(BuilderError::ForeignBuilder)
            );
            assert!(prepared.plan.is_some());
            assert_eq!(state(&origin), origin_before);
            assert_eq!(state(&foreign), foreign_before);
        }

        {
            let builder = Builder::for_tests();
            let mut plan = builder.append_plan(0).unwrap();
            let stale_result =
                plan.input_result("stale", TensorType::f32(vec![1]), Storage::Const)?;
            assert_eq!(stale_result.id(), ValueId::new(0));
            builder
                .constant("intervening", TensorType::f32(vec![1]))
                .unwrap();
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderError::StalePlan { .. })
            ));
            assert_eq!(state(&builder), before);

            let mut next = builder.append_plan(0).unwrap();
            let next_result =
                next.input_result("next", TensorType::f32(vec![1]), Storage::Const)?;
            assert_eq!(next_result.id(), ValueId::new(1));
            let mut prepared = builder.preflight_append(next)?;
            assert_eq!(builder.commit_append(&mut prepared)?.id(), ValueId::new(1));
        }

        {
            let builder = Builder::for_tests();
            builder
                .slot(Slot::Activation, TensorType::f32(vec![1]))
                .unwrap();
            let mut plan = builder.append_plan(0).unwrap();
            plan.input_result(
                "activation",
                TensorType::f32(vec![1]),
                Storage::Slot(Slot::Activation),
            )?;
            let before = state(&builder);
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderError::NameCollision {
                    name,
                    requested: BuilderValueNamespace::Slot,
                    existing: BuilderValueNamespace::Slot,
                }) if name == "activation"
            ));
            assert_eq!(state(&builder), before);

            let mut next = builder.append_plan(0).unwrap();
            let next_result = next.input_result(
                "activation.fresh",
                TensorType::f32(vec![1]),
                Storage::Slot(Slot::Activation),
            )?;
            assert_eq!(next_result.id(), ValueId::new(1));
            let mut prepared = builder.preflight_append(next)?;
            assert_eq!(builder.commit_append(&mut prepared)?.id(), ValueId::new(1));
        }

        Ok(())
    }

    #[test]
    fn transactional_input_result_reservation_is_atomic() -> Result<(), BuilderError> {
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
                let builder = Builder::for_tests();
                let mut plan = builder.append_plan(0).unwrap();
                let result = plan.input_result("result", TensorType::f32(vec![1]), storage)?;
                let mut prepared = builder.preflight_append(plan)?;
                let before = state(&builder);
                builder.reservation_failure.set(Some(collection));
                assert_eq!(
                    builder.commit_append(&mut prepared),
                    Err(BuilderError::Reserve {
                        collection,
                        additional: 1,
                    })
                );
                assert_eq!(state(&builder), before);
                assert!(prepared.plan.is_some());
                assert_eq!(prepared.result.value(), result.id);
                assert_eq!(builder.commit_append(&mut prepared)?, result);
                assert_eq!(result.id(), ValueId::new(0));
            }
        }
        Ok(())
    }

    fn inject_name(builder: &Builder, namespace: BuilderValueNamespace, name: &str) {
        let mut graph = builder.g.borrow_mut();
        let storage = match namespace {
            BuilderValueNamespace::Slot => Storage::Slot(Slot::Activation),
            BuilderValueNamespace::EquationResult | BuilderValueNamespace::NamedValue => {
                Storage::Device
            }
            BuilderValueNamespace::Input | BuilderValueNamespace::Constant => Storage::State,
        };
        let id = graph.values_mut().push(ValueMeta {
            aval: TensorType::f32(vec![1]),
            storage,
            name: Some(name.to_string()),
            key: None,
            state_role: (storage == Storage::State).then_some(StateRole::Recurrent),
            weight: None,
            range: None,
        });
        match namespace {
            BuilderValueNamespace::Input => graph.inputs_mut().push(id),
            BuilderValueNamespace::Constant => {
                graph.inputs_mut().push(id);
                graph.consts_mut().push(id);
            }
            BuilderValueNamespace::Slot => {
                graph.inputs_mut().push(id);
                graph.slots_mut().push((id, Slot::Activation));
            }
            BuilderValueNamespace::EquationResult => {
                let source = ValueId::new(0);
                graph.eqns_mut().push(Eqn {
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
    fn builder_append_preflight_checks_every_name_namespace() -> Result<(), BuilderError> {
        for existing in [
            BuilderValueNamespace::Input,
            BuilderValueNamespace::Constant,
            BuilderValueNamespace::Slot,
            BuilderValueNamespace::EquationResult,
            BuilderValueNamespace::NamedValue,
        ] {
            let builder = Builder::for_tests();
            builder
                .constant("source", TensorType::f32(vec![1]))
                .unwrap();
            inject_name(&builder, existing, "taken");
            let mut plan = builder.append_plan(0).unwrap();
            plan.input("taken", TensorType::f32(vec![1]), Storage::Const)?;
            assert!(matches!(
                builder.preflight_append(plan),
                Err(BuilderError::NameCollision {
                    name,
                    requested: BuilderValueNamespace::Constant,
                    existing: actual,
                }) if name == "taken" && actual == existing
            ));
        }

        let builder = Builder::for_tests();
        let mut plan = builder.append_plan(0).unwrap();
        plan.input("duplicate", TensorType::f32(vec![1]), Storage::Const)?;
        plan.input("duplicate", TensorType::f32(vec![1]), Storage::Const)?;
        assert!(matches!(
            builder.preflight_append(plan),
            Err(BuilderError::NameCollision {
                name,
                requested: BuilderValueNamespace::Constant,
                existing: BuilderValueNamespace::Constant,
            }) if name == "duplicate"
        ));
        Ok(())
    }

    #[test]
    fn builder_preflight_rejects_typed_collisions() -> Result<(), BuilderError> {
        builder_append_preflight_checks_every_name_namespace()
    }

    #[test]
    fn builder_append_preflight_errors_are_typed() -> Result<(), BuilderError> {
        let builder = Builder::for_tests();
        let input = builder
            .constant("input", TensorType::f32(vec![2, 3]))
            .unwrap();
        let mut plan = valid_plan(&builder, input)?;
        plan.equations[0].op = OpKind::Reshape { shape: vec![5] };
        assert!(matches!(
            builder.preflight_append(plan),
            Err(BuilderError::Inference {
                equation: 0,
                operation,
                source: ShapeError::Reshape { .. },
            }) if operation == "reshape [5]"
        ));
        Ok(())
    }

    #[test]
    fn builder_append_rejects_foreign_builder_before_mutation() -> Result<(), BuilderError> {
        let origin = Builder::for_tests();
        let origin_input = origin
            .constant("origin.input", TensorType::f32(vec![2, 3]))
            .unwrap();
        let foreign = Builder::for_tests();
        foreign
            .constant("foreign.input", TensorType::new(vec![7], DType::I32))
            .unwrap();
        assert_eq!(origin.generation(), foreign.generation());
        assert_eq!(
            origin.g.borrow().values().len(),
            foreign.g.borrow().values().len()
        );
        assert_ne!(state(&origin).names, state(&foreign).names);
        assert_ne!(state(&origin).types, state(&foreign).types);

        let plan = valid_plan(&origin, origin_input)?;
        let origin_before = state(&origin);
        let foreign_before = state(&foreign);
        assert!(matches!(
            foreign.preflight_append(plan),
            Err(BuilderError::ForeignBuilder)
        ));
        assert_eq!(state(&origin), origin_before);
        assert_eq!(state(&foreign), foreign_before);

        let plan = valid_plan(&origin, origin_input)?;
        let mut prepared = origin.preflight_append(plan)?;
        assert!(matches!(
            foreign.commit_append(&mut prepared),
            Err(BuilderError::ForeignBuilder)
        ));
        assert!(prepared.plan.is_some());
        assert_eq!(state(&origin), origin_before);
        assert_eq!(state(&foreign), foreign_before);
        Ok(())
    }

    #[test]
    fn builder_append_failure_is_atomic() -> Result<(), BuilderError> {
        let builder = Builder::for_tests();
        let input = builder
            .constant("input", TensorType::f32(vec![2, 3]))
            .unwrap();
        let mut plan = valid_plan(&builder, input)?;
        plan.equations[0].inputs[0] = Operand::Value(ValueId::new(u32::MAX));
        let before = state(&builder);
        assert!(matches!(
            builder.preflight_append(plan),
            Err(BuilderError::InvalidOperand {
                equation: 0,
                value,
                ..
            }) if value == ValueId::new(u32::MAX)
        ));
        assert_eq!(state(&builder), before);
        Ok(())
    }

    #[test]
    fn builder_append_prepared_generation_is_enforced() -> Result<(), BuilderError> {
        let builder = Builder::for_tests();
        let input = builder
            .constant("input", TensorType::f32(vec![2, 3]))
            .unwrap();
        let plan = valid_plan(&builder, input)?;
        let mut prepared = builder.preflight_append(plan)?;
        builder
            .constant("intervening", TensorType::f32(vec![1]))
            .unwrap();
        let before = state(&builder);
        assert!(matches!(
            builder.commit_append(&mut prepared),
            Err(BuilderError::StalePlan { .. })
        ));
        assert_eq!(state(&builder), before);

        let plan = valid_plan(&builder, input)?;
        let mut prepared = builder.preflight_append(plan)?;
        let output = builder.commit_append(&mut prepared)?;
        let after_success = state(&builder);
        assert!(output.id().index() < after_success.values);
        assert_eq!(
            builder.commit_append(&mut prepared),
            Err(BuilderError::ConsumedPlan)
        );
        assert_eq!(state(&builder), after_success);
        Ok(())
    }

    #[test]
    fn builder_append_reserves_every_collection_before_mutation() -> Result<(), BuilderError> {
        for collection in [
            BuilderCollection::Values,
            BuilderCollection::Inputs,
            BuilderCollection::Constants,
            BuilderCollection::Slots,
            BuilderCollection::Equations,
        ] {
            let builder = Builder::for_tests();
            builder.constant("base", TensorType::f32(vec![1])).unwrap();
            let mut plan = builder.append_plan(0).unwrap();
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
                Err(BuilderError::Reserve {
                    collection: actual,
                    additional,
                }) if actual == collection && additional > 0
            ));
            assert_eq!(state(&builder), before, "{collection:?} reserved too late");
            assert!(prepared.plan.is_some(), "{collection:?} consumed the plan");
            assert_eq!(builder.commit_append(&mut prepared)?, output);
        }
        Ok(())
    }

    #[test]
    fn builder_append_is_atomic() -> Result<(), BuilderError> {
        let builder = Builder::for_tests();
        let input = builder
            .constant("input", TensorType::f32(vec![2, 3]))
            .unwrap();
        let mut plan = builder.append_plan(7).unwrap();
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
        assert_eq!(builder.commit_append(&mut prepared)?, output);
        assert_eq!(builder.generation(), generation + 1);
        let graph = builder.finish(output).unwrap();
        assert_eq!(graph.values().len(), 3);
        assert_eq!(graph.inputs(), vec![input.id, constant.id]);
        assert_eq!(graph.consts(), vec![input.id, constant.id]);
        assert_eq!(graph.eqns().len(), 1);
        assert_eq!(graph.eqns()[0].out, output.id);
        Ok(())
    }

    #[test]
    fn exact_i32_builder_append_is_transactional() -> Result<(), BuilderError> {
        let builder = Builder::for_tests();
        let table = crate::test_support::graph_fixtures::i32_constant(
            &builder,
            "routing.table",
            vec![2, 3],
        )?;
        let graph = builder.finish(table).unwrap();
        assert_eq!(
            graph.aval(table.id),
            &TensorType::new(vec![2, 3], DType::I32)
        );
        assert_eq!(graph.meta(table.id).storage, Storage::Const);
        assert!(graph.eqns().is_empty());
        assert!(crate::test_support::verdict(&graph).is_ok());

        let builder = Builder::for_tests();
        crate::test_support::graph_fixtures::i32_constant(&builder, "taken", vec![1])?;
        let before = state(&builder);
        assert!(matches!(
            crate::test_support::graph_fixtures::i32_constant(&builder, "overflow", vec![usize::MAX, 2]),
            Err(BuilderError::ElementCountOverflow { name, .. }) if name == "overflow"
        ));
        assert_eq!(state(&builder), before);
        assert!(matches!(
            crate::test_support::graph_fixtures::i32_constant(&builder, "taken", vec![1]),
            Err(BuilderError::NameCollision { .. })
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
        let builder = Builder::for_tests();
        let float = builder.constant("float", TensorType::f32(vec![2])).unwrap();
        let int = builder
            .constant("int", TensorType::new(vec![2], DType::I32))
            .unwrap();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            builder.binary(BinOp::Add, float, int).unwrap()
        }));
        assert!(
            panicked.is_err(),
            "Builder::binary must panic on mismatched dtypes"
        );

        // 2. BuilderAppendPlan::equation_named (via `equation`), which checks eagerly at staging time.
        let builder = Builder::for_tests();
        let float = builder.constant("float", TensorType::f32(vec![2])).unwrap();
        let int = builder
            .constant("int", TensorType::new(vec![2], DType::I32))
            .unwrap();
        let mut plan = builder.append_plan(0).unwrap();
        let err = plan
            .equation(
                OpKind::Binary(BinOp::Add),
                vec![Operand::Value(float.id), Operand::Value(int.id)],
            )
            .expect_err("equation_named must reject the mismatched pair");
        assert!(matches!(
            err,
            BuilderError::Inference {
                source: ShapeError::Dtype { .. },
                ..
            }
        ));

        // 3. Builder::preflight_append: a plan staged with matching operands, then corrupted post-hoc
        // to reference the mismatched value (mirroring Graph::validate's own test below), so it is
        // preflight's own check that fires, not equation_named's (which already ran once at staging).
        let builder = Builder::for_tests();
        let float_a = builder
            .constant("float_a", TensorType::f32(vec![2]))
            .unwrap();
        let float_b = builder
            .constant("float_b", TensorType::f32(vec![2]))
            .unwrap();
        let int = builder
            .constant("int", TensorType::new(vec![2], DType::I32))
            .unwrap();
        let mut plan = builder.append_plan(0).unwrap();
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
            BuilderError::Inference {
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
        let b = Builder::for_tests();
        let logits = b.constant("logits", TensorType::f32(vec![4])).unwrap();
        let cache_in = b
            .state_input("cache", TensorType::f32(vec![4]), StateRole::Recurrent)
            .unwrap();
        let g = b
            .finish(crate::Outputs::new(logits).with_state(&[(cache_in, cache_in)]))
            .unwrap();
        assert_eq!(g.result(), logits.id);
        assert_eq!(
            g.state().collect::<Vec<_>>(),
            vec![(cache_in.id, cache_in.id)]
        );

        let resumed = Builder::resume(g, TraceLimits::for_tests()).unwrap();
        assert_eq!(resumed.outputs.result.id, logits.id);
        let resumed_state_ids: Vec<(ValueId, ValueId)> = resumed
            .outputs
            .state
            .iter()
            .map(|(si, so)| (si.id, so.id))
            .collect();
        assert_eq!(resumed_state_ids, vec![(cache_in.id, cache_in.id)]);

        // Append an equation reading the resumed `out` - proves `append_plan`/`preflight_append` still
        // work on the new builder identity.
        let doubled = resumed
            .builder
            .binary(
                crate::op::BinOp::Add,
                resumed.outputs.result,
                resumed.outputs.result,
            )
            .unwrap();
        let g2 = resumed
            .builder
            .finish(crate::Outputs::new(doubled).with_state(&resumed.outputs.state))
            .unwrap();
        assert_eq!(g2.result(), doubled.id);
        assert_eq!(
            g2.state().collect::<Vec<_>>(),
            vec![(cache_in.id, cache_in.id)],
            "the state pair must survive resume + a second finish_with_state unchanged"
        );
    }

    impl Builder {
        /// The handle of the value with index `index` of this builder: a test that rebuilds the handles
        /// of a graph it declared through setup code.
        pub(crate) fn handle_for_tests(&self, index: usize) -> Traced {
            let id =
                ValueId::from_index(index).expect("a test names a value index of the id space");
            self.handle(id)
        }

        /// Replace the limits this trace charges against, keeping everything charged so far: a test that
        /// sets up state under generous limits and then narrows them to a controlled boundary.
        pub(crate) fn set_limits_for_tests(&self, limits: TraceLimits) {
            self.meter.set_limits(limits);
        }

        /// Make the next commit-time reservation of `collection` fail as if the allocator refused it.
        pub(crate) fn fail_next_reservation(&self, collection: BuilderCollection) {
            self.reservation_failure.set(Some(collection));
        }
    }
}
