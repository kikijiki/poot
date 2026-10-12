//! `apply_lora`: the LoRA mode transform (POOT-579, dmodel.md ruling M2/R4).
//!
//! A model traces once, with no adapter in it. A caller that serves adapters hands `compile` a
//! [`LoraPlan`] and this pass adds, to every contraction of a targeted projection,
//!
//! ```text
//! y + gather(lora.scale, idx) * IndexedMatMul(IndexedMatMul(x, lora.a, idx), lora.b, idx)
//! ```
//!
//! where `idx` is the one [`Slot::LoraIdx`] input `[rows]` (I32, the index dtype),
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
use poot_graph_ir::{Builder, IndexRange, PackedSourceName, Slot, ValueTable};
use poot_quant::weights::{LoraRole, LoraTarget, WeightId, WeightRole};
use poot_tensor::DType;

use super::GraphBudget;
use crate::ExpansionError;

/// The stage name an expansion refusal reports.
const STAGE: &str = "apply_lora";

/// What one LoRA transform inserts: which projections are adapted, and the shape of the adapter stack
/// the program is compiled against. The stack's contents are bound at run time; only its shape is a
/// compile-time fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraPlan {
    /// The step's row count: the extent of the [`Slot::LoraIdx`] input, one adapter index per row.
    pub rows: NonZeroUsize,
    /// Adapters in the stack, not counting the reserved no-adapter entry 0.
    pub adapters: NonZeroUsize,
    /// The common rank every adapter is zero-padded to.
    pub rank: NonZeroUsize,
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
    budget.reserve(STAGE, g, 4 + sites.len() * 12, sites.len() * 11 + 3)?;

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
            Slot::LoraIdx,
            TensorType::new(vec![rows], DType::I32),
            entries,
        )?
        .id();
    let scale = b
        .constant(
            &WeightId::model(WeightRole::Lora(LoraRole::Scale)).const_name(),
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
            .constant(
                &a.const_name(),
                TensorType::f32(vec![stack, site.in_dim, rank]),
            )?
            .id();
        let factor_b = b
            .constant(
                &bb.const_name(),
                TensorType::f32(vec![stack, rank, site.out_dim]),
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
        let (OpKind::MatMul, [Operand::Value(x), Operand::Value(w)]) =
            (&eqn.op, eqn.inputs.as_slice())
        else {
            continue;
        };
        let Some(weight) = weight_name(g, &producer, *w) else {
            continue;
        };
        let Some((layer, target)) = projection(&weight, &plan.projections) else {
            continue;
        };
        let &[in_dim, out_dim] = g.aval(*w).shape.as_slice() else {
            continue;
        };
        found.push(Site {
            eqn: e,
            x: *x,
            layer,
            target,
            weight,
            in_dim,
            out_dim,
        });
    }
    found
}

/// The const name of the weight a contraction's right operand is: the const itself, a transpose of it,
/// or the `PackedDequant` of its source tensors, whichever way the tracer spelled the linear.
fn weight_name(
    g: &Graph,
    producer: &ValueTable<Option<usize>>,
    operand: ValueId,
) -> Option<String> {
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
    match producer[through].map(|e| &g.eqns()[e]) {
        None => {
            let meta = g.meta(through);
            (meta.storage == Storage::Const)
                .then(|| meta.name.clone())
                .flatten()
        }
        Some(Eqn {
            op: OpKind::PackedDequant { .. },
            inputs,
            ..
        }) => {
            let Some(&Operand::Value(source)) = inputs.first() else {
                return None;
            };
            let name = g.meta(source).name.as_deref()?;
            Some(PackedSourceName::parse(name)?.linear_id().to_string())
        }
        Some(_) => None,
    }
}

/// `(layer, target)` when `name` is the const name of a layer weight whose role is one of `targets`.
fn projection(name: &str, targets: &[LoraTarget]) -> Option<(usize, LoraTarget)> {
    let rest = name.strip_prefix("w.l")?;
    let (layer, role) = rest.split_once('.')?;
    let layer = layer.parse().ok()?;
    targets
        .iter()
        .copied()
        .find(|target| target.weight_role().spelling() == role)
        .map(|target| (layer, target))
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
    use crate::production_plan::Submission;
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
        CompileOptions::new(Submission::Replay, CompileLimits::STANDARD)
    }

    /// SC-004: a plan naming a projection the graph lacks is refused by `compile` with the typed error.
    #[test]
    fn a_projection_the_graph_lacks_is_a_typed_compile_refusal() {
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(vec![ROWS, 1, 4]))
            .unwrap();
        let w = b
            .constant(
                &WeightId::layer(0, WeightRole::Attn(poot_quant::weights::AttnRole::Q))
                    .const_name(),
                TensorType::f32(vec![4, 4]),
            )
            .unwrap();
        let y = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let g = b.finish(y).unwrap();
        let wanted = LoraPlan {
            rows: NonZeroUsize::new(ROWS).unwrap(),
            adapters: NonZeroUsize::MIN,
            rank: NonZeroUsize::MIN,
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
            .constant(
                &WeightId::layer(0, WeightRole::Attn(poot_quant::weights::AttnRole::Q))
                    .const_name(),
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
}
