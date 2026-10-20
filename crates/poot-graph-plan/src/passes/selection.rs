//! `form_selection`: the block selection of a sparse-attention indexer and the attention over it, formed from
//! the primitive compositions a tracer states (POOT-1236).
//!
//! A tracer writes [`ops::block_top_k_packed`] and [`ops::selected_attention`] as ordinary equations; it
//! cannot emit [`OpKind::BlockTopK`] or [`OpKind::SelectedAttention`] (architecture rule 3: the composites
//! are a pass's output). This pass recognizes the two compositions and replaces their root equation, in place,
//! with the composite whose decomposition they are, so a device runs the linear radix select and the
//! block-masked streaming attention instead of the quadratic rank chain and the dense mask.
//!
//! Recognition is by the decomposition itself, never by a hand-written copy of its shape: a candidate root
//! fixes the composite's parameters (the block ratio, the number of kept blocks, the operand types), the
//! definition is built over placeholders of those types with the same `ops::` function the tracer calls,
//! normalized the way `compile` normalizes the traced graph, and unified with the graph from the candidate
//! root. Every equation of the definition must match one of the graph, so a near-miss composition (a different
//! tie rule, a changed mask) is left alone. The members the composite absorbed stay until `dce` drops them.
//! The pass is required, never reverted: the composition is quadratic in the block count and has no device
//! lowering, so a target that cannot plan the composite refuses it with its own typed refusal.
//!
//! - `BlockTopK`: the root is the `Concat` of the ids and the keep indicator ([`ops::block_top_k_packed`]).
//! - `SelectedAttention`: the root is an additive, capacity-bound `Attention` the flash pass formed, whose
//!   mask is [`ops::selected_block_mask`] over the selected block ids and the query positions.

use std::collections::{HashMap, HashSet};

use poot_graph_ir::graph::{EditError, Eqn, Graph, Operand, Storage, ValueId};
use poot_graph_ir::op::{
    AttentionAttrs, BinOp, BlockTopKAttrs, KeyLayout, KvBound, MaskOperand, OpKind,
    SelectedAttentionAttrs, SinkOperand,
};
use poot_graph_ir::types::{Scalar, TensorType};
use poot_graph_ir::{BuilderError, Outputs, TraceLimits, Traced, ops};
use poot_tensor::DType;

use super::guard_witness::producer_map;
use super::{canonicalize_with_declines, cse, fold_iota, with_eqns};

/// The most equations a definition built for recognition may hold. The compositions are a few hundred
/// equations whatever the tensor extents; this is far above that and far below what exhausts a host.
const DEFINITION_LIMITS: TraceLimits = TraceLimits {
    max_tokens: std::num::NonZeroUsize::MIN,
    max_values: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    max_eqns: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    max_literal_bytes: std::num::NonZeroU64::new(1 << 20).unwrap(),
    max_owned_bytes: std::num::NonZeroU64::new(1 << 28).unwrap(),
};

/// The most equations a search for a candidate's parameters visits before giving up: the mask of a selected
/// attention is under a hundred equations, so a search that reaches this left the mask for the model.
const SEARCH_BUDGET: usize = 512;

/// Replace each recognized block selection and selected attention with its composite.
pub(crate) fn form_selection(g: &Graph) -> Result<Graph, EditError> {
    let producers = producer_map(g);
    let mut formed = false;
    let eqns: Vec<Eqn> = g
        .eqns()
        .iter()
        .map(|eqn| match recognize(g, &producers, eqn) {
            Some(composite) => {
                formed = true;
                composite
            }
            None => eqn.clone(),
        })
        .collect();
    if formed {
        with_eqns(g, eqns)
    } else {
        Ok(g.clone())
    }
}

/// `eqn` as the composite its composition defines, or `None` when it is not the root of one.
fn recognize(g: &Graph, producers: &HashMap<ValueId, usize>, eqn: &Eqn) -> Option<Eqn> {
    match &eqn.op {
        OpKind::Concat { axis: 3 } => block_top_k(g, producers, eqn),
        OpKind::Attention(attrs) => selected_attention(g, producers, eqn, attrs),
        _ => None,
    }
}

/// A definition built over placeholders, normalized as `compile` normalizes a traced graph.
struct Definition {
    graph: Graph,
    root: ValueId,
    /// The placeholder of each operand, in operand order.
    operands: Vec<ValueId>,
    producers: HashMap<ValueId, usize>,
}

impl Definition {
    /// The graph `build` states over placeholders of `operands`' types. `None` when the types do not admit
    /// the definition (the typing rules refuse them).
    fn new(
        operands: &[TensorType],
        build: impl FnOnce(&poot_graph_ir::Builder, &[Traced]) -> Result<Traced, BuilderError>,
    ) -> Option<Self> {
        let b = poot_graph_ir::Builder::new(DEFINITION_LIMITS);
        let placeholders: Vec<Traced> = operands
            .iter()
            .enumerate()
            .map(|(position, ty)| b.constant(&format!("operand.{position}"), ty.clone()))
            .collect::<Result<_, _>>()
            .ok()?;
        let result = build(&b, &placeholders).ok()?;
        let root = result.id();
        let graph = b.finish(Outputs::new(result)).ok()?;
        // The order `compile` runs: canonicalize, cse, fold_iota. The root is an output, so it keeps its id.
        let (graph, _) = canonicalize_with_declines(&graph).ok()?;
        let graph = fold_iota(&cse(&graph).ok()?).ok()?;
        let producers = producer_map(&graph);
        Some(Self {
            graph,
            root,
            operands: placeholders.iter().map(|p| p.id()).collect(),
            producers,
        })
    }

    /// The graph values the placeholders stand for when this definition is the computation of `root` in `g`,
    /// in operand order, or `None` when some equation of the definition has no match.
    fn unify(
        &self,
        g: &Graph,
        producers: &HashMap<ValueId, usize>,
        root: ValueId,
    ) -> Option<Vec<ValueId>> {
        let mut unifier = Unifier {
            definition: self,
            g,
            producers,
            bound: HashMap::new(),
        };
        unifier.value(self.root, root).then(|| {
            self.operands
                .iter()
                .map(|placeholder| unifier.bound[placeholder])
                .collect()
        })
    }
}

/// The unification of a definition's values with a graph's.
struct Unifier<'a> {
    definition: &'a Definition,
    g: &'a Graph,
    producers: &'a HashMap<ValueId, usize>,
    /// The graph value each definition value stands for, once matched.
    bound: HashMap<ValueId, ValueId>,
}

impl Unifier<'_> {
    /// Whether the definition's value `d` is the graph's value `t`: same type, and either a placeholder (any
    /// value), the same compiler-computed constant, or an equation of the same op over matching operands.
    fn value(&mut self, d: ValueId, t: ValueId) -> bool {
        if let Some(&bound) = self.bound.get(&d) {
            return bound == t;
        }
        let definition = &self.definition.graph;
        if definition.aval(d) != self.g.aval(t) {
            return false;
        }
        let matched = match self.definition.producers.get(&d) {
            None if self.definition.operands.contains(&d) => true,
            None => match (definition.meta(d).storage, self.g.meta(t).storage) {
                (Storage::Computed(expected), Storage::Computed(found)) => expected == found,
                _ => false,
            },
            Some(&at) => {
                let expected = &definition.eqns()[at];
                self.producers.get(&t).is_some_and(|&found| {
                    let found = &self.g.eqns()[found];
                    expected.op == found.op
                        && expected.inputs.len() == found.inputs.len()
                        && expected
                            .inputs
                            .iter()
                            .zip(&found.inputs)
                            .all(|(d, t)| self.operand(*d, *t))
                })
            }
        };
        if matched {
            self.bound.insert(d, t);
        }
        matched
    }

    fn operand(&mut self, d: Operand, t: Operand) -> bool {
        match (d, t) {
            (Operand::Value(d), Operand::Value(t)) => self.value(d, t),
            (Operand::Lit(d), Operand::Lit(t)) => same_literal(d, t),
            _ => false,
        }
    }
}

/// Literals are the same when their bits are: a NaN literal equals itself, `-0.0` is not `0.0`.
fn same_literal(a: Scalar, b: Scalar) -> bool {
    match (a, b) {
        (Scalar::F32(a), Scalar::F32(b)) => a.to_bits() == b.to_bits(),
        (Scalar::I32(a), Scalar::I32(b)) => a == b,
        _ => false,
    }
}

/// The producing equation of `value`, if it is an intermediate.
fn eqn_of<'g>(
    g: &'g Graph,
    producers: &HashMap<ValueId, usize>,
    value: ValueId,
) -> Option<&'g Eqn> {
    producers.get(&value).map(|&at| &g.eqns()[at])
}

/// The value operand `index` of `value`'s producing equation, when that equation is a `Binary(op)`.
fn binary_input(
    g: &Graph,
    producers: &HashMap<ValueId, usize>,
    value: ValueId,
    op: BinOp,
    index: usize,
) -> Option<Operand> {
    let eqn = eqn_of(g, producers, value)?;
    (eqn.op == OpKind::Binary(op))
        .then(|| eqn.inputs.get(index).copied())
        .flatten()
}

/// A `BlockTopK` from the `Concat` that ends [`ops::block_top_k_packed`]: its ratio is read off the
/// eligibility `Ge(pos, first + (ratio - 1))` of its keep indicator, then the whole definition is checked.
fn block_top_k(g: &Graph, producers: &HashMap<ValueId, usize>, root: &Eqn) -> Option<Eqn> {
    let [Operand::Value(ids), Operand::Value(keep_i32)] = root.inputs[..] else {
        return None;
    };
    let out = g.aval(root.out);
    let &[rows, 1, tokens, width] = out.shape.as_slice() else {
        return None;
    };
    if out.dtype != DType::I32 {
        return None;
    }
    let k = *g.aval(ids).shape.get(3)?;
    let blocks = width.checked_sub(k)?;
    // keep = Mul(top-k mask, eligible); eligible = Ge(pos, last); last = Add(first, ratio - 1).
    let keep = match eqn_of(g, producers, keep_i32)? {
        Eqn {
            op: OpKind::Cast { .. },
            inputs,
            ..
        } => match inputs[..] {
            [Operand::Value(keep)] => keep,
            _ => return None,
        },
        _ => return None,
    };
    let Operand::Value(eligible) = binary_input(g, producers, keep, BinOp::Mul, 1)? else {
        return None;
    };
    let Operand::Value(last) = binary_input(g, producers, eligible, BinOp::Ge, 1)? else {
        return None;
    };
    let (_, Scalar::F32(ratio_less_one)) = value_and_scalar(&eqn_of(g, producers, last)?.inputs)?
    else {
        return None;
    };
    let ratio = block_ratio(ratio_less_one + 1.0)?;

    let operands = [
        TensorType::f32(vec![rows, 1, tokens, blocks]),
        TensorType::new(vec![rows, tokens], DType::I32),
    ];
    let definition = Definition::new(&operands, |b, ops_in| {
        ops::block_top_k_packed(b, ops_in[0], ops_in[1], ratio, k)
    })?;
    let bound = definition.unify(g, producers, root.out)?;
    let attrs = BlockTopKAttrs { ratio, k };
    composite(
        g,
        root,
        OpKind::BlockTopK(attrs),
        bound.into_iter().map(Operand::Value).collect(),
    )
}

/// A `SelectedAttention` from the additive attention whose mask is [`ops::selected_block_mask`]: the block ids
/// and the positions are the two integers the mask casts to F32, its ratio is the multiplier of its block
/// iota, and the whole mask is then checked.
fn selected_attention(
    g: &Graph,
    producers: &HashMap<ValueId, usize>,
    root: &Eqn,
    attrs: &AttentionAttrs,
) -> Option<Eqn> {
    if attrs.mask != MaskOperand::Additive
        || attrs.key != KeyLayout::Single
        || attrs.bound != KvBound::Capacity
        || attrs.sink != SinkOperand::Absent
    {
        return None;
    }
    let [
        Operand::Value(q),
        Operand::Value(k),
        Operand::Value(v),
        Operand::Value(mask),
    ] = root.inputs[..]
    else {
        return None;
    };
    let mask_type = g.aval(mask);
    let &[rows, 1, lq, lk] = mask_type.shape.as_slice() else {
        return None;
    };
    let Search { integers, ratios } = search_mask(g, producers, mask)?;
    let [ids, pos] = integers[..] else {
        return None;
    };
    let [ratio] = ratios[..] else {
        return None;
    };
    // The integers sort by rank: the ids are `[B, 1, Lq, K]`, the positions `[B, Lq]`.
    let (ids, pos) = if g.aval(ids).rank() == 4 {
        (ids, pos)
    } else {
        (pos, ids)
    };
    let &[_, _, _, selected] = g.aval(ids).shape.as_slice() else {
        return None;
    };
    if g.aval(pos).shape != [rows, lq] {
        return None;
    }
    let operands = [g.aval(ids).clone(), g.aval(pos).clone()];
    let definition = Definition::new(&operands, |b, ops_in| {
        ops::selected_block_mask(b, ops_in[0], ops_in[1], ratio, lk)
    })?;
    let bound = definition.unify(g, producers, mask)?;
    let selected_attrs = SelectedAttentionAttrs {
        n_rep: attrs.n_rep,
        scale: attrs.scale,
        softcap: attrs.softcap,
        ratio,
        k: selected,
    };
    composite(
        g,
        root,
        OpKind::SelectedAttention(selected_attrs),
        [q, k, v, bound[0], bound[1]]
            .into_iter()
            .map(Operand::Value)
            .collect(),
    )
}

/// What a search of a mask's definition found: the integer values it casts to F32 and the multipliers of its
/// block iotas.
struct Search {
    integers: Vec<ValueId>,
    ratios: Vec<usize>,
}

/// Walk the equations `mask` is computed from, stopping at the I32-to-F32 casts (their sources are the
/// operands of the definition) and at values nothing computes.
fn search_mask(g: &Graph, producers: &HashMap<ValueId, usize>, mask: ValueId) -> Option<Search> {
    let mut seen = HashSet::new();
    let mut pending = vec![mask];
    let mut integers = Vec::new();
    let mut ratios = Vec::new();
    while let Some(value) = pending.pop() {
        if !seen.insert(value) {
            continue;
        }
        if seen.len() > SEARCH_BUDGET {
            return None;
        }
        let Some(eqn) = eqn_of(g, producers, value) else {
            continue;
        };
        if let OpKind::Cast { to: DType::F32 } = eqn.op
            && let [Operand::Value(source)] = eqn.inputs[..]
            && g.aval(source).dtype == DType::I32
        {
            if !integers.contains(&source) {
                integers.push(source);
            }
            continue;
        }
        if eqn.op == OpKind::Binary(BinOp::Mul)
            && let Some((iota, Scalar::F32(ratio))) = value_and_scalar(&eqn.inputs)
            && matches!(
                g.meta(iota).storage,
                Storage::Computed(poot_graph_ir::ComputedConst::Iota { .. })
            )
            && let Some(ratio) = block_ratio(ratio)
            && !ratios.contains(&ratio)
        {
            ratios.push(ratio);
        }
        pending.extend(eqn.inputs.iter().filter_map(|operand| match operand {
            Operand::Value(input) => Some(*input),
            Operand::Lit(_) => None,
        }));
    }
    Some(Search { integers, ratios })
}

/// `inputs` as `(value, scalar literal)`, the literal in either position.
fn value_and_scalar(inputs: &[Operand]) -> Option<(ValueId, Scalar)> {
    match inputs {
        [Operand::Value(value), Operand::Lit(scalar)]
        | [Operand::Lit(scalar), Operand::Value(value)] => Some((*value, *scalar)),
        _ => None,
    }
}

/// `ratio` as a block ratio: a positive whole number.
fn block_ratio(ratio: f32) -> Option<usize> {
    (ratio >= 1.0 && ratio.fract() == 0.0 && ratio < (1 << 24) as f32).then_some(ratio as usize)
}

/// `op` over `inputs` as the replacement of `root`, when it types over them exactly as `root` did.
fn composite(g: &Graph, root: &Eqn, op: OpKind, inputs: Vec<Operand>) -> Option<Eqn> {
    let types: Vec<TensorType> = inputs
        .iter()
        .map(|operand| match operand {
            Operand::Value(value) => Some(g.aval(*value).clone()),
            Operand::Lit(_) => None,
        })
        .collect::<Option<_>>()?;
    (op.infer(&types).ok()? == *g.aval(root.out)).then_some(Eqn {
        op,
        inputs,
        out: root.out,
        layer: root.layer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::passes_without_target;
    use poot_eval::{EvalOptions, Value, eval};
    use poot_graph_ir::test_support::block_top_k::{grid_scores, reference_row};
    use poot_graph_ir::test_support::selected_attention::{Case, cases};
    use poot_graph_ir::{Builder, IndexRange, Slot};
    use poot_tensor::HostTensor;

    const RATIO: usize = 2;
    const KEPT: usize = 3;
    const BLOCKS: usize = 9;
    const POSITIONS: [i32; 6] = [3, 5, 9, 1, 7, 17];

    /// The traced composition over 2 rows of 3 queries: its graph and the inputs it runs on.
    fn selection_graph() -> (Graph, HashMap<ValueId, Value>) {
        let b = Builder::for_tests();
        let scores = b
            .constant("scores", TensorType::f32(vec![2, 1, 3, BLOCKS]))
            .unwrap();
        let pos = b
            .slot_ranged(
                Slot::Pos,
                TensorType::new(vec![2, 3], DType::I32),
                IndexRange::new(0, 18).unwrap(),
            )
            .unwrap();
        let packed = ops::block_top_k_packed(&b, scores, pos, RATIO, KEPT).unwrap();
        let graph = b.finish(Outputs::new(packed)).unwrap();
        let rows: Vec<f32> = (0..6)
            .flat_map(|q| grid_scores(BLOCKS, 1 << 20, q))
            .collect();
        let inputs = HashMap::from([
            (
                scores.id(),
                Value::from(HostTensor::f32(vec![2, 1, 3, BLOCKS], rows)),
            ),
            (
                pos.id(),
                Value::from(HostTensor::i32(vec![2, 3], POSITIONS.to_vec())),
            ),
        ]);
        (graph, inputs)
    }

    fn count(graph: &Graph, is: impl Fn(&OpKind) -> bool) -> usize {
        graph.eqns().iter().filter(|eqn| is(&eqn.op)).count()
    }

    fn run(graph: &Graph, inputs: &HashMap<ValueId, Value>) -> HostTensor {
        eval(graph, inputs, EvalOptions::default())
            .unwrap()
            .output
            .into_host()
            .expect("a dense result")
    }

    /// The composition `ops::block_top_k_packed` states becomes one `BlockTopK` of its ratio and kept
    /// blocks, the rank chain goes with it, and the result is the oracle's bit for bit. Mutation: have
    /// `form_selection` return the graph unchanged; the graph holds no `BlockTopK` and this row goes red.
    #[test]
    fn the_block_selection_composition_is_formed_into_the_op() {
        let (traced, inputs) = selection_graph();
        assert_eq!(count(&traced, |op| matches!(op, OpKind::BlockTopK(_))), 0);
        let formed = passes_without_target(&traced).unwrap();
        let ops: Vec<_> = formed
            .eqns()
            .iter()
            .filter_map(|eqn| match &eqn.op {
                OpKind::BlockTopK(attrs) => Some(*attrs),
                _ => None,
            })
            .collect();
        assert_eq!(
            ops,
            [BlockTopKAttrs {
                ratio: RATIO,
                k: KEPT
            }]
        );
        assert_eq!(count(&formed, |op| matches!(op, OpKind::ArgTopK { .. })), 0);
        assert_eq!(
            run(&formed, &inputs),
            run(&traced, &inputs),
            "the formed op is the composition's result"
        );
        // The plain selection of the independent reference agrees with both.
        let scores = inputs
            .values()
            .find_map(|v| v.as_host().and_then(|t| t.as_f32().map(<[f32]>::to_vec)))
            .expect("the scores input");
        let want: Vec<i32> = (0..6)
            .flat_map(|q| {
                reference_row(
                    &scores[q * BLOCKS..(q + 1) * BLOCKS],
                    POSITIONS[q],
                    RATIO,
                    KEPT,
                )
            })
            .collect();
        assert_eq!(run(&formed, &inputs).as_i32().unwrap(), want);
    }

    /// A composition that differs from the definition by one operation is not the op: `keep` combined by an
    /// `Add` where the definition multiplies stays the primitive chain. Mutation: have `Unifier::value` accept
    /// any op for a producer; the near-miss is formed and this row goes red.
    #[test]
    fn a_near_miss_block_selection_is_left_alone() {
        let (traced, _) = selection_graph();
        let producers = producer_map(&traced);
        let root = traced
            .eqns()
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Concat { axis: 3 }))
            .expect("the packed result");
        let Operand::Value(keep_i32) = root.inputs[1] else {
            panic!("the keep indicator is a value")
        };
        let Operand::Value(keep) = traced.eqns()[producers[&keep_i32]].inputs[0] else {
            panic!("the cast reads the keep mask")
        };
        let mut edit = traced.edit();
        let at = producers[&keep];
        assert_eq!(edit.eqns_mut()[at].op, OpKind::Binary(BinOp::Mul));
        edit.eqns_mut()[at].op = OpKind::Binary(BinOp::Add);
        let near_miss = edit.commit().unwrap();
        let formed = passes_without_target(&near_miss).unwrap();
        assert_eq!(count(&formed, |op| matches!(op, OpKind::BlockTopK(_))), 0);
        assert!(count(&formed, |op| matches!(op, OpKind::ArgTopK { .. })) > 0);
    }

    /// The traced graph of `case`'s attention: `ops::selected_attention` over ranged id and position slots,
    /// and the inputs it runs on.
    fn attention_graph(case: &Case) -> (Graph, HashMap<ValueId, Value>) {
        let (batch, hq, lq, lk, d, hkv) =
            (case.batch, case.hq, case.lq, case.lk, case.d, case.hkv());
        let b = Builder::for_tests();
        let q = b
            .constant("q", TensorType::f32(vec![batch, hq, lq, d]))
            .unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![batch, hkv, lk, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![batch, hkv, lk, d]))
            .unwrap();
        let ids = b
            .slot_named_ranged(
                Slot::Token,
                "ids",
                TensorType::new(vec![batch, 1, lq, case.k], DType::I32),
                IndexRange::new(-1, lk.div_ceil(case.ratio) as i64).unwrap(),
            )
            .unwrap();
        let pos = b
            .slot_ranged(
                Slot::Pos,
                TensorType::new(vec![batch, lq], DType::I32),
                IndexRange::below(lk).unwrap(),
            )
            .unwrap();
        let out = ops::selected_attention(
            &b,
            q,
            k,
            v,
            ids,
            pos,
            case.n_rep,
            case.scale(),
            None,
            case.ratio,
        )
        .unwrap();
        let graph = b.finish(Outputs::new(out)).unwrap();
        let [qv, kv, vv] = case.operands(21);
        let inputs = HashMap::from([
            (
                q.id(),
                Value::from(HostTensor::f32(vec![batch, hq, lq, d], qv)),
            ),
            (
                k.id(),
                Value::from(HostTensor::f32(vec![batch, hkv, lk, d], kv)),
            ),
            (
                v.id(),
                Value::from(HostTensor::f32(vec![batch, hkv, lk, d], vv)),
            ),
            (
                ids.id(),
                Value::from(HostTensor::i32(
                    vec![batch, 1, lq, case.k],
                    case.ids.clone(),
                )),
            ),
            (
                pos.id(),
                Value::from(HostTensor::i32(vec![batch, lq], case.pos.clone())),
            ),
        ]);
        (graph, inputs)
    }

    /// The attention `ops::selected_attention` states becomes one `SelectedAttention` with the case's ratio and
    /// block count, for every selection case, and computes what the composition computed. Mutation: have
    /// `form_selection` return the graph unchanged; no case forms the op and this row goes red.
    #[test]
    fn the_selected_attention_composition_is_formed_into_the_op() {
        for case in cases() {
            let (traced, inputs) = attention_graph(&case);
            let formed = passes_without_target(&traced).unwrap();
            let attrs: Vec<_> = formed
                .eqns()
                .iter()
                .filter_map(|eqn| match &eqn.op {
                    OpKind::SelectedAttention(attrs) => Some(attrs.clone()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                attrs,
                [SelectedAttentionAttrs {
                    n_rep: case.n_rep,
                    scale: case.scale(),
                    softcap: None,
                    ratio: case.ratio,
                    k: case.k,
                }],
                "{}",
                case.label
            );
            assert_eq!(
                count(&formed, |op| matches!(op, OpKind::Attention(_))),
                0,
                "{}: the dense attention is gone",
                case.label
            );
            assert_eq!(
                run(&formed, &inputs),
                run(&traced, &inputs),
                "{}: the formed op is the composition's result",
                case.label
            );
        }
    }

    /// A mask that differs from the definition by one operation is not a selection: the block indicators
    /// combined by `Add` where the definition takes their `Max` stays a dense attention. Mutation: as above.
    #[test]
    fn a_near_miss_selected_attention_is_left_alone() {
        let case = cases().remove(0);
        let (traced, _) = attention_graph(&case);
        let mut edit = traced.edit();
        let max = edit
            .eqns_mut()
            .iter_mut()
            .find(|eqn| eqn.op == OpKind::Binary(BinOp::Max))
            .expect("the mask takes the larger of the selected and the tail blocks");
        max.op = OpKind::Binary(BinOp::Add);
        let near_miss = edit.commit().unwrap();
        let formed = passes_without_target(&near_miss).unwrap();
        assert_eq!(
            count(&formed, |op| matches!(op, OpKind::SelectedAttention(_))),
            0
        );
        assert_eq!(count(&formed, |op| matches!(op, OpKind::Attention(_))), 1);
    }
}
