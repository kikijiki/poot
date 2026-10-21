//! `apply_lora`: the LoRA mode transform (POOT-579, dmodel.md ruling M2/R4).
//!
//! A model traces once, with no adapter in it. A caller that serves adapters hands `compile` a
//! [`LoraPlan`] and this pass adds, to every contraction of a targeted projection,
//!
//! ```text
//! y + gather(lora.scale, idx) * IndexedMatMul(IndexedMatMul(x, lora.a, idx), lora.b, idx)
//! ```
//!
//! where `idx` is the one `LORA_IDX` input `[rows]` (I32, the index dtype),
//! expanded to one entry per token. The adapters of one set are stacked on a leading axis beside the
//! base weights, in the `x @ W` orientation (`A [adapters + 1, in, r]`, `B [adapters + 1, r, out]`,
//! `scale [adapters + 1]`), zero-padded to a common rank `r`; entry 0 is the no-adapter row, whose `A`
//! and scale are zero so its correction is an exact zero. Each stacked tensor is a graph const named by
//! its [`WeightId`] with a [`WeightRole::Lora`] role, so the executor binds adapters through the same
//! weight map as the base weights.
//!
//! The transform runs before anything else in [`crate::compile`], on the traced graph: every later pass
//! (and the planner) sees an ordinary graph that happens to contain `IndexedMatMul`s, and no tracer, no
//! executor and no per-stage compile knows about adapters. The inserted equations carry the layer tag of
//! the contraction they correct.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use poot_graph_ir::builder::Resumed;
use poot_graph_ir::graph::{Eqn, Graph, Operand, Storage, ValueId, ValueMeta};
use poot_graph_ir::op::{BinOp, OpKind};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Builder, IndexRange, OwnedSlot, Slot, ValueTable, WeightRef};
use poot_tensor::DType;
use poot_weights::{LoraRole, LoraTarget, WeightId, WeightRole};

use super::GraphBudget;
use crate::ExpansionError;

/// The stage name an expansion refusal reports.
const STAGE: &str = "apply_lora";

/// The one slot of an adapter-stacked graph: the per-row adapter selection `[rows]` (I32, the index dtype
/// `IndexedMatMul` reads). Row `m`'s value is the 1-based adapter index (0 is the no-adapter row) whose stacked
/// `A`/`B` this pass applies to that row this step. It varies per step (requests and adapters change as the
/// batch admits and evicts), so it is a slot rebound every step, never a `Const` (const-cache staleness).
pub const LORA_IDX: OwnedSlot = OwnedSlot::i32_tensor("loraidx");

/// What one LoRA transform inserts: which projections are adapted, and the shape of the adapter stack
/// the program is compiled against. The stack's contents are bound at run time; only its shape is a
/// compile-time fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraPlan {
    /// The step's row count: the extent of the `LORA_IDX` input, one adapter index per row.
    pub rows: NonZeroUsize,
    /// Adapters in the stack, not counting the reserved no-adapter entry 0.
    pub adapters: NonZeroUsize,
    /// The common rank every adapter is zero-padded to.
    pub rank: NonZeroUsize,
    /// The dtype the stacked factors are stored and uploaded in (F32, F16 or BF16). A narrow factor is
    /// widened to F32 on the device by the correction itself, never on the host.
    pub factor_dtype: DType,
    /// The projections the adapters correct. Each must occur in the graph.
    pub projections: Vec<LoraTarget>,
}

/// Why [`apply_lora`] refused a graph.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LoraError {
    /// The plan names a projection no contraction of the graph reads: an adapter for it would be
    /// silently ignored.
    #[error("the graph has no {0:?} projection for the adapters to correct")]
    NoProjection(LoraTarget),
    /// The plan adapts no projection at all.
    #[error("the LoRA plan names no projection")]
    NoTargets,
    /// The plan's factor dtype is not a float the correction can contract.
    #[error("the LoRA factors are {0:?}; the correction contracts F32, F16 or BF16 factors")]
    FactorDtype(DType),
    /// A contraction's activation is not F32: the stacked factors are, and `IndexedMatMul` contracts
    /// operands of one dtype.
    #[error("{weight}: the activation is {dtype:?}; the LoRA correction contracts F32 activations")]
    Activation {
        /// Name of the contracted weight.
        weight: String,
        /// The activation's dtype.
        dtype: DType,
    },
    /// A contraction's activation rows are not a whole number of tokens per step row.
    #[error("{weight}: {elements} activation rows do not split over {rows} step rows")]
    Rows {
        /// Name of the contracted weight.
        weight: String,
        /// Number of activation rows the contraction reads.
        elements: usize,
        /// Number of rows in one step.
        rows: usize,
    },
    /// The adapter stack has more entries than the `i32` words of an adapter index.
    #[error("an adapter stack of {entries} entries is past the i32 adapter indices")]
    StackNotIndexable {
        /// Number of entries in the adapter stack.
        entries: usize,
    },
    /// Growing the graph for the correction would pass the compile's limits.
    #[error(transparent)]
    Expansion(#[from] ExpansionError),
    /// The graph editor refused a rewrite the pass issued.
    #[error(transparent)]
    Edit(Box<poot_graph_ir::EditError>),
    /// The tracer refused a declaration: it does not fit the compile's graph limits, or the graph
    /// already held a value of that name.
    #[error(transparent)]
    Build(Box<poot_graph_ir::BuilderError>),
}

impl From<poot_graph_ir::BuilderError> for LoraError {
    fn from(error: poot_graph_ir::BuilderError) -> Self {
        Self::Build(Box::new(error))
    }
}

impl From<poot_graph_ir::EditError> for LoraError {
    fn from(error: poot_graph_ir::EditError) -> Self {
        Self::Edit(Box::new(error))
    }
}

/// One contraction of an adapted projection.
struct Site {
    eqn: usize,
    x: ValueId,
    layer: usize,
    target: LoraTarget,
    weight: String,
    in_dim: usize,
    out_dim: usize,
}

/// Insert the LoRA correction of `plan` after every contraction of a targeted projection of `g`.
pub(crate) fn apply_lora(
    g: &Graph,
    plan: &LoraPlan,
    budget: &GraphBudget,
) -> Result<Graph, LoraError> {
    if plan.projections.is_empty() {
        return Err(LoraError::NoTargets);
    }
    if !matches!(plan.factor_dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(LoraError::FactorDtype(plan.factor_dtype));
    }
    let sites = sites(g, plan);
    for &target in &plan.projections {
        if !sites.iter().any(|site| site.target == target) {
            return Err(LoraError::NoProjection(target));
        }
    }
    let rows = plan.rows.get();
    let stack = plan.adapters.get() + 1;
    let rank = plan.rank.get();
    for site in &sites {
        let x = g.aval(site.x);
        if x.dtype != DType::F32 {
            return Err(LoraError::Activation {
                weight: site.weight.clone(),
                dtype: x.dtype,
            });
        }
        let elements: usize = x.shape[..x.shape.len() - 1].iter().product();
        if !elements.is_multiple_of(rows) {
            return Err(LoraError::Rows {
                weight: site.weight.clone(),
                elements,
                rows,
            });
        }
    }
    budget.reserve(STAGE, g, 4 + sites.len() * 14, sites.len() * 13 + 3)?;

    // The adapter inputs: the one index slot, the scale vector, and the stacked factors of each
    // adapted (layer, projection). Declared through a builder so the slot carries its typed key.
    let Resumed { builder, outputs } = Builder::resume(g.clone(), budget.trace_limits())?;
    let b = &builder;
    // The adapter index selects one of `stack` entries; the binder checks every word against the range, so
    // the `IndexedMatMul`s below are proven to read inside the stack (`compile` requires it).
    let entries =
        IndexRange::below(stack).ok_or(LoraError::StackNotIndexable { entries: stack })?;
    let idx = b
        .slot_ranged(
            Slot::Owned(LORA_IDX),
            TensorType::new(vec![rows], DType::I32),
            entries,
        )?
        .id();
    let scale = b
        .weight(
            WeightRef::whole(WeightId::model(WeightRole::Lora(LoraRole::Scale))),
            TensorType::f32(vec![stack]),
        )?
        .id();
    let mut factors: HashMap<(usize, LoraTarget), (ValueId, ValueId)> = HashMap::new();
    for site in &sites {
        if factors.contains_key(&(site.layer, site.target)) {
            continue;
        }
        let a = WeightId::layer(site.layer, WeightRole::Lora(LoraRole::A(site.target)));
        let bb = WeightId::layer(site.layer, WeightRole::Lora(LoraRole::B(site.target)));
        let factor_a = b
            .weight(
                WeightRef::whole(a),
                TensorType::new(vec![stack, site.in_dim, rank], plan.factor_dtype),
            )?
            .id();
        let factor_b = b
            .weight(
                WeightRef::whole(bb),
                TensorType::new(vec![stack, rank, site.out_dim], plan.factor_dtype),
            )?
            .id();
        factors.insert((site.layer, site.target), (factor_a, factor_b));
    }
    // The builder keeps the witnesses and exports `g` declared, and refuses a finish that does not
    // carry every state pair again.
    let declared = builder.finish(outputs)?;
    let mut edit = declared.edit();
    let originals = std::mem::take(edit.eqns_mut());

    let mut emitter = Emitter {
        values: edit.values_mut(),
        eqns: Vec::with_capacity(originals.len() + sites.len() * 11),
        budget,
        layer: None,
    };
    // A narrow factor is widened on the device once per step, before any contraction reads it: the
    // correction contracts F32 operands only.
    if plan.factor_dtype != DType::F32 {
        for (a, bb) in factors.values_mut() {
            *a = emitter.val(OpKind::Cast { to: DType::F32 }, vec![*a])?;
            *bb = emitter.val(OpKind::Cast { to: DType::F32 }, vec![*bb])?;
        }
    }
    let mut site_of: HashMap<usize, &Site> = sites.iter().map(|site| (site.eqn, site)).collect();
    // The per-token index for a step of `tokens` tokens per row, emitted once per width.
    let mut token_idx: HashMap<usize, ValueId> = HashMap::new();
    for (e, eqn) in originals.into_iter().enumerate() {
        let Some(site) = site_of.remove(&e) else {
            emitter.eqns.push(eqn);
            continue;
        };
        let Eqn {
            op,
            inputs,
            out: y,
            layer,
        } = eqn;
        emitter.layer = layer;
        // The contraction keeps its operands but writes a fresh value; the corrected result takes the
        // original id, so every reader (and the graph's output, state and witnesses) is untouched.
        let base_meta = emitter.values[y].clone();
        let base = emitter.push_value(ValueMeta::new(
            base_meta.aval.clone(),
            Storage::Device,
            None,
        ))?;
        emitter.eqns.push(Eqn {
            op,
            inputs,
            out: base,
            layer,
        });

        let x_shape = declared_shape(&emitter, site.x);
        let m: usize = x_shape[..x_shape.len() - 1].iter().product();
        let tokens = m / rows;
        let per_token = match token_idx.get(&tokens) {
            Some(&value) => value,
            None => {
                let value = if tokens == 1 {
                    idx
                } else {
                    let column = emitter.val(
                        OpKind::Reshape {
                            shape: vec![rows, 1],
                        },
                        vec![idx],
                    )?;
                    let wide = emitter.val(
                        OpKind::Broadcast {
                            shape: vec![rows, tokens],
                        },
                        vec![column],
                    )?;
                    emitter.val(OpKind::Reshape { shape: vec![m] }, vec![wide])?
                };
                token_idx.insert(tokens, value);
                value
            }
        };
        let (a, bb) = factors[&(site.layer, site.target)];
        let x2 = if x_shape.len() == 2 {
            site.x
        } else {
            emitter.val(
                OpKind::Reshape {
                    shape: vec![m, site.in_dim],
                },
                vec![site.x],
            )?
        };
        let xa = emitter.val(OpKind::IndexedMatMul, vec![x2, a, per_token])?;
        let xab = emitter.val(OpKind::IndexedMatMul, vec![xa, bb, per_token])?;
        let row_scale = emitter.val(OpKind::Gather { axis: 0 }, vec![scale, per_token])?;
        let scale_col = emitter.val(OpKind::Reshape { shape: vec![m, 1] }, vec![row_scale])?;
        let scaled = emitter.val(OpKind::Binary(BinOp::Mul), vec![xab, scale_col])?;
        let correction = emitter.val(
            OpKind::Reshape {
                shape: base_meta.aval.shape.clone(),
            },
            vec![scaled],
        )?;
        emitter.budget.push_eqn(
            STAGE,
            &mut emitter.eqns,
            Eqn {
                op: OpKind::Binary(BinOp::Add),
                inputs: vec![Operand::Value(base), Operand::Value(correction)],
                out: y,
                layer,
            },
        )?;
    }
    let eqns = emitter.eqns;
    *edit.eqns_mut() = eqns;
    Ok(edit.commit()?)
}

fn declared_shape(emitter: &Emitter<'_>, value: ValueId) -> Vec<usize> {
    emitter.values[value].aval.shape.clone()
}

/// The contractions of `g` whose weight is a targeted projection, in equation order.
fn sites(g: &Graph, plan: &LoraPlan) -> Vec<Site> {
    let mut producer: ValueTable<Option<usize>> = ValueTable::filled(g.values().len(), None);
    for (e, eqn) in g.eqns().iter().enumerate() {
        producer[eqn.out] = Some(e);
    }
    let mut found = Vec::new();
    for (e, eqn) in g.eqns().iter().enumerate() {
        let (true, [Operand::Value(x), Operand::Value(w)]) =
            (eqn.op.is_plain_matmul(), eqn.inputs.as_slice())
        else {
            continue;
        };
        let Some(id) = weight_id(g, &producer, *w) else {
            continue;
        };
        let (Some(layer), Some(target)) = (id.layer, LoraTarget::of(id.role)) else {
            continue;
        };
        if !plan.projections.contains(&target) {
            continue;
        }
        let &[in_dim, out_dim] = g.aval(*w).shape.as_slice() else {
            continue;
        };
        found.push(Site {
            eqn: e,
            x: *x,
            layer,
            target,
            weight: id.const_name(),
            in_dim,
            out_dim,
        });
    }
    found
}

/// The mapped weight a contraction's right operand is: the const itself, a transpose of it, or the
/// `PackedDequant` of its source tensors, whichever way the tracer spelled the linear. The identity
/// is the [`WeightRef`] the tracer declared on the constant, never its name.
fn weight_id(
    g: &Graph,
    producer: &ValueTable<Option<usize>>,
    operand: ValueId,
) -> Option<WeightId> {
    let through = match producer[operand].map(|e| &g.eqns()[e]) {
        Some(Eqn {
            op: OpKind::Transpose { perm },
            inputs,
            ..
        }) if perm.as_slice() == [1, 0] => match inputs.as_slice() {
            [Operand::Value(inner)] => *inner,
            _ => return None,
        },
        _ => operand,
    };
    let source = match producer[through].map(|e| &g.eqns()[e]) {
        None => through,
        Some(Eqn {
            op: OpKind::PackedDequant { .. },
            inputs,
            ..
        }) => match inputs.first() {
            Some(&Operand::Value(source)) => source,
            _ => return None,
        },
        Some(_) => return None,
    };
    Some(g.meta(source).weight()?.id())
}

/// Appends values and equations to the graph under rewrite, each through the budget, with each
/// output type computed by [`OpKind::infer`] rather than written down.
struct Emitter<'a> {
    values: &'a mut ValueTable<ValueMeta>,
    eqns: Vec<Eqn>,
    budget: &'a GraphBudget,
    layer: Option<poot_graph_ir::graph::LayerIndex>,
}

impl Emitter<'_> {
    fn push_value(&mut self, meta: ValueMeta) -> Result<ValueId, ExpansionError> {
        self.budget.push_value(STAGE, self.values, meta)
    }

    fn val(&mut self, op: OpKind, inputs: Vec<ValueId>) -> Result<ValueId, ExpansionError> {
        let types: Vec<TensorType> = inputs
            .iter()
            .map(|&input| self.values[input].aval.clone())
            .collect();
        let aval = op
            .infer(&types)
            .unwrap_or_else(|error| panic!("apply_lora: internal shape error in {op:?}: {error}"));
        let out = self.push_value(ValueMeta::new(aval, Storage::Device, None))?;
        self.budget.push_eqn(
            STAGE,
            &mut self.eqns,
            Eqn {
                op,
                inputs: inputs.into_iter().map(Operand::Value).collect(),
                out,
                layer: self.layer,
            },
        )?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Builder, Graph, Slot, TensorType};
    use poot_target::Backend;
    use poot_test_util::device_caps::default_caps_for;

    use super::*;
    use crate::{CompileError, CompileLimits, CompileOptions, Target, compile};

    const ROWS: usize = 4;

    fn target() -> Target {
        Target::Gpu {
            backend: Backend::SpirvVulkan,
            caps: default_caps_for(Backend::SpirvVulkan),
        }
    }

    fn options() -> CompileOptions {
        CompileOptions::new(CompileLimits::STANDARD)
    }

    /// SC-003: a plan naming a projection the graph lacks is refused by `compile` with the typed error.
    /// Mutation: skip the `NoProjection` check; the plan compiles and its adapters are silently ignored.
    #[test]
    fn a_projection_the_graph_lacks_is_a_typed_compile_refusal() {
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(vec![ROWS, 1, 4]))
            .unwrap();
        let w = b
            .weight(
                WeightRef::whole(WeightId::layer(
                    0,
                    WeightRole::Attn(poot_weights::AttnRole::Q),
                )),
                TensorType::f32(vec![4, 4]),
            )
            .unwrap();
        let y = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let g = b.finish(y).unwrap();
        let wanted = LoraPlan {
            rows: NonZeroUsize::new(ROWS).unwrap(),
            adapters: NonZeroUsize::MIN,
            rank: NonZeroUsize::MIN,
            factor_dtype: poot_tensor::DType::F32,
            projections: vec![LoraTarget::AttnQ, LoraTarget::FfnDown],
        };
        let mut with_plan = options();
        with_plan.lora = Some(wanted.clone());
        match compile(&g, &target(), &with_plan) {
            Err(CompileError::Lora(error)) => {
                assert_eq!(*error, LoraError::NoProjection(LoraTarget::FfnDown));
            }
            other => panic!(
                "expected NoProjection(FfnDown), got {:?}",
                other.map(|_| ())
            ),
        }
        // The same graph under a plan for what it has compiles.
        let present = LoraPlan {
            projections: vec![LoraTarget::AttnQ],
            ..wanted
        };
        let mut with_plan = options();
        with_plan.lora = Some(present);
        compile(&g, &target(), &with_plan).expect("a plan for a present projection compiles");
    }

    /// A site is found by the weight identity the tracer declared, not by the constant's name: a constant
    /// spelled `w.l0.attn.q` with no declared identity is no projection.
    #[test]
    fn a_constant_named_like_a_projection_without_its_identity_is_no_site() {
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(vec![ROWS, 1, 4]))
            .unwrap();
        let w = b
            .constant("w.l0.attn.q", TensorType::f32(vec![4, 4]))
            .unwrap();
        let y = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let g = b.finish(y).unwrap();
        assert_eq!(
            refused(&g, q_plan(ROWS, vec![LoraTarget::AttnQ])),
            LoraError::NoProjection(LoraTarget::AttnQ)
        );
    }

    /// A one-projection graph `x [elements, 1, 4] @ q^T` with `x` and `q` in `dtype`.
    fn q_graph(elements: usize, dtype: DType) -> Graph {
        let b = Builder::for_tests();
        let x = b
            .slot_named(
                Slot::Activation,
                "x",
                TensorType::new(vec![elements, 1, 4], dtype),
            )
            .unwrap();
        let w = b
            .weight(
                WeightRef::whole(WeightId::layer(
                    0,
                    WeightRole::Attn(poot_weights::AttnRole::Q),
                )),
                TensorType::new(vec![4, 4], dtype),
            )
            .unwrap();
        let y = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        b.finish(y).unwrap()
    }

    fn q_plan(rows: usize, projections: Vec<LoraTarget>) -> LoraPlan {
        LoraPlan {
            rows: NonZeroUsize::new(rows).unwrap(),
            adapters: NonZeroUsize::MIN,
            rank: NonZeroUsize::MIN,
            factor_dtype: poot_tensor::DType::F32,
            projections,
        }
    }

    fn refused(g: &Graph, plan: LoraPlan) -> LoraError {
        let mut with_plan = options();
        with_plan.lora = Some(plan);
        match compile(g, &target(), &with_plan) {
            Err(CompileError::Lora(error)) => *error,
            other => panic!("expected a typed LoRA refusal, got {:?}", other.map(|_| ())),
        }
    }

    /// A plan that adapts no projection is refused by `compile`; the same graph under a plan naming one
    /// compiles. Mutation: delete the empty check; with nothing to adapt the pass passes and `compile`
    /// returns `Ok`.
    #[test]
    fn a_plan_with_no_projection_is_a_typed_compile_refusal() {
        let g = q_graph(2, DType::F32);
        assert_eq!(refused(&g, q_plan(2, Vec::new())), LoraError::NoTargets);
        let mut with_plan = options();
        with_plan.lora = Some(q_plan(2, vec![LoraTarget::AttnQ]));
        compile(&g, &target(), &with_plan).expect("a plan naming a present projection compiles");
    }

    /// A bf16 activation is refused with the weight named; the F32 twin of the same graph compiles.
    #[test]
    fn a_non_f32_activation_is_a_typed_compile_refusal() {
        let error = refused(&q_graph(2, DType::BF16), q_plan(2, vec![LoraTarget::AttnQ]));
        assert_eq!(
            error,
            LoraError::Activation {
                weight: "w.l0.attn.q".to_string(),
                dtype: DType::BF16
            }
        );
    }

    /// Three activation rows do not split over two step rows; two do.
    #[test]
    fn activation_rows_that_do_not_split_over_the_step_rows_are_a_typed_compile_refusal() {
        let error = refused(&q_graph(3, DType::F32), q_plan(2, vec![LoraTarget::AttnQ]));
        assert_eq!(
            error,
            LoraError::Rows {
                weight: "w.l0.attn.q".to_string(),
                elements: 3,
                rows: 2
            }
        );
        let mut with_plan = options();
        with_plan.lora = Some(q_plan(3, vec![LoraTarget::AttnQ]));
        compile(&q_graph(3, DType::F32), &target(), &with_plan).expect("3 rows over 3 step rows");
    }

    /// POOT-1258 SC-001 (the program side): a plan whose factors are stored BF16 declares the stacked `A`/`B`
    /// consts as BF16, so the executor uploads them as stored, and widens them with a device `Cast` that
    /// reads the BF16 const. An F32 plan declares F32 consts and no cast of them. Mutation: declare the
    /// factors `TensorType::f32` whatever the plan says; the BF16 row finds F32 consts.
    #[test]
    fn narrow_factors_are_declared_in_their_stored_dtype_and_widened_on_the_device() {
        let g = q_graph(2, DType::F32);
        let lora_consts = |dtype: DType| {
            let mut plan = q_plan(2, vec![LoraTarget::AttnQ]);
            plan.factor_dtype = dtype;
            let mut with_plan = options();
            with_plan.lora = Some(plan);
            let program = compile(&g, &target(), &with_plan).expect("the plan compiles");
            let graph = program.graph();
            let consts: Vec<(ValueId, DType)> = graph
                .values()
                .iter_enumerated()
                .filter(|(_, meta)| {
                    meta.weight().is_some_and(|w| {
                        matches!(
                            w.id().role,
                            WeightRole::Lora(LoraRole::A(_) | LoraRole::B(_))
                        )
                    })
                })
                .map(|(id, meta)| (id, meta.aval.dtype))
                .collect();
            let widened = graph
                .eqns()
                .iter()
                .filter(|eqn| {
                    matches!(eqn.op, OpKind::Cast { to: DType::F32 })
                        && matches!(eqn.inputs.as_slice(), [Operand::Value(input)]
                            if consts.iter().any(|(id, _)| id == input))
                })
                .count();
            (consts, widened)
        };
        let (bf16, widened) = lora_consts(DType::BF16);
        assert_eq!(bf16.len(), 2, "one A and one B const");
        assert!(
            bf16.iter().all(|&(_, dtype)| dtype == DType::BF16),
            "{bf16:?}"
        );
        assert_eq!(widened, 2, "each BF16 factor is widened by a device Cast");
        let (f32s, widened) = lora_consts(DType::F32);
        assert!(
            f32s.iter().all(|&(_, dtype)| dtype == DType::F32),
            "{f32s:?}"
        );
        assert_eq!(widened, 0, "F32 factors are contracted as they are");
    }

    /// A factor dtype the correction cannot contract (an integer) is the typed refusal.
    #[test]
    fn a_non_float_factor_dtype_is_a_typed_compile_refusal() {
        let mut plan = q_plan(2, vec![LoraTarget::AttnQ]);
        plan.factor_dtype = DType::I32;
        assert_eq!(
            refused(&q_graph(2, DType::F32), plan),
            LoraError::FactorDtype(DType::I32)
        );
    }
}
