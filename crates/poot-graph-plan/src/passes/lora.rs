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
use poot_graph_ir::graph::{Eqn, Graph, Operand, Storage, ValidationChannel, ValueId, ValueMeta};
use poot_graph_ir::op::{BinOp, OpKind};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Builder, PackedSourceName, Slot, ValueTable};
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
    Activation { weight: String, dtype: DType },
    /// A contraction's activation rows are not a whole number of tokens per step row.
    #[error("{weight}: {elements} activation rows do not split over {rows} step rows")]
    Rows {
        weight: String,
        elements: usize,
        rows: usize,
    },
    #[error(transparent)]
    Expansion(#[from] ExpansionError),
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
pub fn apply_lora<V: ValidationChannel>(
    g: &Graph<V>,
    plan: &LoraPlan,
    budget: &GraphBudget,
) -> Result<Graph<V>, LoraError> {
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
    let Graph {
        values,
        inputs,
        consts,
        slots,
        eqns,
        output,
        state,
        validations,
        exports,
    } = g.clone();
    let plain = Graph {
        values,
        inputs,
        consts,
        slots,
        eqns,
        output,
        validations: poot_graph_ir::NoValidations,
        exports,
        state,
    };
    let Resumed {
        builder,
        out: traced,
        state: carried,
    } = Builder::resume(plain);
    let b = &builder;
    let idx = b
        .slot(Slot::LoraIdx, TensorType::new(vec![rows], DType::I32))
        .id();
    let scale = b
        .constant(
            &WeightId::model(WeightRole::Lora(LoraRole::Scale)).const_name(),
            TensorType::f32(vec![stack]),
        )
        .id();
    let mut factors: HashMap<(usize, LoraTarget), (ValueId, ValueId)> = HashMap::new();
    for site in &sites {
        factors.entry((site.layer, site.target)).or_insert_with(|| {
            let a = WeightId::layer(site.layer, WeightRole::Lora(LoraRole::A(site.target)));
            let bb = WeightId::layer(site.layer, WeightRole::Lora(LoraRole::B(site.target)));
            (
                b.constant(
                    &a.const_name(),
                    TensorType::f32(vec![stack, site.in_dim, rank]),
                )
                .id(),
                b.constant(
                    &bb.const_name(),
                    TensorType::f32(vec![stack, rank, site.out_dim]),
                )
                .id(),
            )
        });
    }
    let mut declared = builder.finish_with_state(traced, &carried);
    let originals = std::mem::take(&mut declared.eqns);

    let mut emitter = Emitter {
        values: &mut declared.values,
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
    let Graph {
        values,
        inputs,
        consts,
        slots,
        output,
        state,
        exports,
        ..
    } = declared;
    Ok(Graph {
        values,
        inputs,
        consts,
        slots,
        eqns,
        output,
        validations,
        exports,
        state,
    })
}

fn declared_shape(emitter: &Emitter<'_>, value: ValueId) -> Vec<usize> {
    emitter.values[value].aval.shape.clone()
}

/// The contractions of `g` whose weight is a targeted projection, in equation order.
fn sites<V: ValidationChannel>(g: &Graph<V>, plan: &LoraPlan) -> Vec<Site> {
    let mut producer: ValueTable<Option<usize>> = ValueTable::filled(g.values.len(), None);
    for (e, eqn) in g.eqns.iter().enumerate() {
        producer[eqn.out] = Some(e);
    }
    let mut found = Vec::new();
    for (e, eqn) in g.eqns.iter().enumerate() {
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
fn weight_name<V: ValidationChannel>(
    g: &Graph<V>,
    producer: &ValueTable<Option<usize>>,
    operand: ValueId,
) -> Option<String> {
    let through = match producer[operand].map(|e| &g.eqns[e]) {
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
    match producer[through].map(|e| &g.eqns[e]) {
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
    use std::collections::HashMap;

    use poot_eval::{EvalBudget, EvalOptions, Value, eval};
    use poot_executor_parity::dense::{Dense, Family, plain, step};
    use poot_graph_ir::Graph;
    use poot_graph_ir::{Builder, Slot, Storage, TensorType, ValueId};
    use poot_models::model::{LogitRows, Phase};
    use poot_target::Backend;
    use poot_tensor::HostTensor;
    use poot_test_util::device_caps::default_caps_for;

    use super::*;
    use crate::{CompileError, CompileLimits, CompileOptions, Submission, Target, compile};

    const ROWS: usize = 4;
    const RANK: usize = 2;

    fn target() -> Target {
        Target {
            backend: Backend::SpirvVulkan,
            caps: default_caps_for(Backend::SpirvVulkan),
        }
    }

    fn options() -> CompileOptions {
        CompileOptions::new(Submission::Replay, CompileLimits::STANDARD)
    }

    /// Deterministic fill in [-1, 1) keyed by `seed`.
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn name_seed(name: &str) -> u64 {
        name.bytes().fold(0xcbf2_9ce4_8422_2325, |acc, byte| {
            (acc ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// The base value of weight const `name`: the same for every graph that declares it.
    fn base_weight(name: &str, numel: usize) -> Vec<f32> {
        fill(numel, name_seed(name))
            .into_iter()
            .map(|v| v * 0.3)
            .collect()
    }

    /// The tiny qwen2 the LoRA rows trace through the registry.
    fn qwen2() -> poot_executor_parity::weight_map::MappedModel {
        Dense::new(Family::Qwen2)
            .vocab(32)
            .dims(16, 32, 2)
            .heads(4, 2)
            .head_dim(4)
            .max_positions(16)
            .f32_model()
    }

    fn decode(rows: usize) -> Graph {
        plain(
            qwen2()
                .model
                .trace(Phase::Decode, step(rows, 1, 8, LogitRows::Last))
                .unwrap(),
        )
    }

    /// `(in, out)` of every targeted projection of the tiny qwen2.
    fn dims(target: LoraTarget) -> (usize, usize) {
        match target {
            LoraTarget::AttnQ => (16, 16),
            LoraTarget::AttnK | LoraTarget::AttnV => (16, 8),
            LoraTarget::AttnO => (16, 16),
            LoraTarget::FfnGate | LoraTarget::FfnUp => (16, 32),
            LoraTarget::FfnDown => (32, 16),
        }
    }

    /// One adapter: `A [in, r]`, `B [r, out]` per (layer, projection) and its scale.
    struct Adapter {
        scale: f32,
        a: HashMap<(usize, LoraTarget), Vec<f32>>,
        b: HashMap<(usize, LoraTarget), Vec<f32>>,
    }

    fn adapter(seed: u64, scale: f32) -> Adapter {
        let (mut a, mut b) = (HashMap::new(), HashMap::new());
        for layer in 0..2 {
            for (i, target) in LoraTarget::ALL.into_iter().enumerate() {
                let (in_dim, out_dim) = dims(target);
                let s = seed + (layer * 7 + i) as u64 * 11;
                a.insert((layer, target), fill(in_dim * RANK, s));
                b.insert((layer, target), fill(RANK * out_dim, s + 1));
            }
        }
        Adapter { scale, a, b }
    }

    fn plan() -> LoraPlan {
        LoraPlan {
            rows: NonZeroUsize::new(ROWS).unwrap(),
            adapters: NonZeroUsize::new(2).unwrap(),
            rank: NonZeroUsize::new(RANK).unwrap(),
            projections: LoraTarget::ALL.to_vec(),
        }
    }

    /// The stacked consts of `adapters`, entry 0 all zeros, by const name.
    fn stacked(adapters: &[&Adapter]) -> HashMap<String, Vec<f32>> {
        let mut out = HashMap::new();
        let mut scale = vec![0.0];
        scale.extend(adapters.iter().map(|adapter| adapter.scale));
        out.insert(
            WeightId::model(WeightRole::Lora(LoraRole::Scale)).const_name(),
            scale,
        );
        for layer in 0..2 {
            for target in LoraTarget::ALL {
                let (in_dim, out_dim) = dims(target);
                let mut a = vec![0.0; in_dim * RANK];
                let mut b = vec![0.0; RANK * out_dim];
                for adapter in adapters {
                    a.extend(&adapter.a[&(layer, target)]);
                    b.extend(&adapter.b[&(layer, target)]);
                }
                out.insert(
                    WeightId::layer(layer, WeightRole::Lora(LoraRole::A(target))).const_name(),
                    a,
                );
                out.insert(
                    WeightId::layer(layer, WeightRole::Lora(LoraRole::B(target))).const_name(),
                    b,
                );
            }
        }
        out
    }

    /// The base weights with `adapter` merged in: `W + scale * (A B)^T` (the stored `[out, in]` layout).
    fn merged(adapter: &Adapter) -> HashMap<String, Vec<f32>> {
        let mut out = HashMap::new();
        for layer in 0..2 {
            for target in LoraTarget::ALL {
                let (in_dim, out_dim) = dims(target);
                let name = WeightId::layer(layer, target.weight_role()).const_name();
                let mut w = base_weight(&name, in_dim * out_dim);
                let (a, b) = (&adapter.a[&(layer, target)], &adapter.b[&(layer, target)]);
                for i in 0..in_dim {
                    for o in 0..out_dim {
                        let delta: f32 = (0..RANK)
                            .map(|k| a[i * RANK + k] * b[k * out_dim + o])
                            .sum();
                        w[o * in_dim + i] += adapter.scale * delta;
                    }
                }
                out.insert(name, w);
            }
        }
        out
    }

    /// Evaluate `g` on the oracle for one decode step: `tokens` per row, `lora` rows when it declares
    /// the slot, consts from `overrides` or the base fill.
    fn run(
        g: &Graph,
        overrides: &HashMap<String, Vec<f32>>,
        tokens: &[i32],
        lora: &[i32],
    ) -> Vec<f32> {
        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        for &id in &g.inputs {
            let aval = g.aval(id).clone();
            let meta = g.meta(id);
            let tensor = match meta.storage {
                Storage::Slot(Slot::Token) => HostTensor::i32(aval.shape, tokens.to_vec()),
                Storage::Slot(Slot::Pos) => HostTensor::i32(aval.shape, vec![3; tokens.len()]),
                Storage::Slot(Slot::LoraIdx) => HostTensor::i32(aval.shape, lora.to_vec()),
                Storage::Slot(other) => panic!("unexpected slot {other:?}"),
                Storage::Computed(c) => c.tensor(),
                Storage::State => HostTensor::f32(aval.shape.clone(), vec![0.0; aval.numel()]),
                Storage::Const => {
                    let name = meta.name.as_deref().expect("a const has a name");
                    let data = overrides
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| base_weight(name, aval.numel()));
                    HostTensor::f32(aval.shape, data)
                }
                Storage::Device => unreachable!(),
            };
            inputs.insert(id, Value::from(tensor));
        }
        eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
            .as_f32()
            .unwrap()
            .to_vec()
    }

    fn compiled(g: &Graph, lora: Option<LoraPlan>) -> Graph {
        let mut options = options();
        options.lora = lora;
        compile(g, &target(), &options)
            .unwrap_or_else(|e| panic!("compile: {e}"))
            .graph()
            .clone()
    }

    const TOKENS: [i32; ROWS] = [5, 9, 3, 11];

    /// Row `r`'s slice of `[ROWS, 1, vocab]` logits.
    fn row(logits: &[f32], r: usize) -> &[f32] {
        let vocab = logits.len() / ROWS;
        &logits[r * vocab..(r + 1) * vocab]
    }

    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                "{what}: logit {i}: got {g}, merged-weight reference {w}"
            );
        }
    }

    /// SC-003: qwen2 decodes four rows under two adapters on alternate rows (and one no-adapter row);
    /// each row's logits equal the dense graph with that adapter merged into the weights, and the
    /// no-adapter row equals the unadapted program bit for bit.
    #[test]
    fn lora_rows_equal_the_per_row_merged_weight_graph_and_index_zero_is_the_base() {
        let g = decode(ROWS);
        let (first, second) = (adapter(100, 2.0), adapter(200, 3.0));
        let program = compiled(&g, Some(plan()));
        let base = compiled(&g, None);
        let consts = stacked(&[&first, &second]);
        let idx = [1, 2, 0, 2];
        let adapted = run(&program, &consts, &TOKENS, &idx);

        let reference = |adapter: &Adapter| run(&base, &merged(adapter), &TOKENS, &[]);
        let (with_first, with_second) = (reference(&first), reference(&second));
        assert_close(
            row(&adapted, 0),
            row(&with_first, 0),
            "row 0 under adapter 1",
        );
        assert_close(
            row(&adapted, 1),
            row(&with_second, 1),
            "row 1 under adapter 2",
        );
        assert_close(
            row(&adapted, 3),
            row(&with_second, 3),
            "row 3 under adapter 2",
        );

        let plain_run = run(&base, &HashMap::new(), &TOKENS, &[]);
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(row(&adapted, 2)),
            bits(row(&plain_run, 2)),
            "LoraIdx 0 must equal the no-adapter program bit for bit"
        );
        // The adapters visibly change the output: a correction that were all zero would pass the rows above.
        assert!(
            row(&adapted, 0)
                .iter()
                .zip(row(&plain_run, 0))
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "adapter 1 changes row 0's logits"
        );
    }

    /// SC-004: a plan naming a projection the graph lacks is refused by `compile` with the typed error.
    #[test]
    fn a_projection_the_graph_lacks_is_a_typed_compile_refusal() {
        let b = Builder::new();
        let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![ROWS, 1, 4]));
        let w = b.constant(
            &WeightId::layer(0, WeightRole::Attn(poot_quant::weights::AttnRole::Q)).const_name(),
            TensorType::f32(vec![4, 4]),
        );
        let y = b.matmul(x, b.transpose(w, vec![1, 0]));
        let g = b.finish(y);
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
        let b = Builder::new();
        let x = b.slot_named(
            Slot::Activation,
            "x",
            TensorType::new(vec![elements, 1, 4], dtype),
        );
        let w = b.constant(
            &WeightId::layer(0, WeightRole::Attn(poot_quant::weights::AttnRole::Q)).const_name(),
            TensorType::new(vec![4, 4], dtype),
        );
        let y = b.matmul(x, b.transpose(w, vec![1, 0]));
        b.finish(y)
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
