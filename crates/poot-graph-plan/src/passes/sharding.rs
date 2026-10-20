//! Tensor-parallel sharding (POOT-749, ADR-0031's `shard_map` semantics over the ordinary traced graph).
//!
//! [`shard`] rewrites one traced graph into the program every rank of a tensor-parallel group runs:
//! the same equations over per-rank shapes, with an `AllReduce(Sum)` where ranks hold partial sums and
//! an `AllGather` where a consumer needs the whole axis. It never reads a family name: how a weight is
//! split comes from its [`WeightRole`] ([`ShardRule::Megatron`]) and everything else is propagated.
//!
//! # Layouts
//!
//! Every value of the sharded graph is, relative to the value of the whole graph, in one of four
//! layouts:
//!
//! | layout | each rank holds |
//! | ------ | --------------- |
//! | `Whole` | the whole value (replicated; computed redundantly) |
//! | `Split(a)` | its equal slice of the value along axis `a` |
//! | `Partial` | an addend: the value is the sum over ranks |
//! | `Undecided` | a carried-state input no equation has given a layout yet |
//!
//! [`ShardRule::Megatron`] cuts the checkpoint `[out, in]` weights of Q, K, V, gate and up on `out`
//! (column-parallel) and of O and down on `in` (row-parallel); embedding, head and norms are replicated.
//! Propagation is a closed rule set: pointwise ops, reductions, broadcast, transposes, reshapes that
//! keep the split axis a leading factor of one output axis (`Hq*D -> [Hq, D]`), slices and
//! concatenations off the split axis, row gathers, batched matmul on the head axis, the indexed matmul
//! of an adapter stack, and cache writes (a slice update, or the paged pool's scatter by slot map, which
//! divides by KV heads so each rank's pool holds its own heads). A contraction over a split axis yields
//! `Partial`; a partial is added to partials and scaled by whole values without a collective, and is
//! reduced (`AllReduce(Sum)`) at its first consumer that is not linear in it, so a LoRA correction and
//! its base projection share one reduction. Any other op that reads a split value is
//! [`ShardError::Unpropagatable`]: a typed refusal, never a guess.
//!
//! # Lowering to ranks
//!
//! The sharded graph keeps its collectives as equations, which `compile` plans nowhere (a collective is
//! a cut between stage programs). [`expand_ranks`] lowers them: it unrolls the graph once per rank into
//! one rank-annotated graph in which `AllReduce(Sum)` is, on every rank, the left fold of the partials
//! of ranks `0..world` in rank order (`world - 1` adds) and `AllGather` a `Concat` in rank order,
//! orders the equations by (segment between collectives, rank), and names the cuts. The ordinary stage
//! split then makes every rank's equations its own stage on its own device; the operand a rank reads
//! from a peer is a boundary value that crosses devices, carried by one point-to-point row. The fold
//! is the same adds of the same operands in the same order on every rank, so ranks stay bit-identical.
//!
//! Each rank reads every peer's partial once: a collective is one stage boundary whatever the world,
//! and moves `world * (world - 1)` vectors. A tree or a recursive-doubling exchange moves fewer bytes
//! at four ranks and costs a stage boundary per level; on the fixture of the `tensor_parallel` test
//! (two layers, four reductions) recursive doubling measured 30 stages and 27 copies against 16 and
//! 39, so the fold is kept for decode-sized vectors.

use std::collections::HashMap;
use std::num::NonZeroU32;

use poot_graph_ir::analysis::NumericsProperty;
use poot_graph_ir::graph::{Eqn, Graph, LayerIndex, Operand, StateRole, Storage, ValueId};
use poot_graph_ir::op::{BinOp, OpKind, RedOp, UnOp};
use poot_graph_ir::{PackedSourceName, TensorType, ValueMeta, ValueTable, packed_source_type};
use poot_quant::format::Storage as FormatStorage;
use poot_quant::weights::{
    AttnRole, FfnRole, HandleFormat, LoraRole, LoraTarget, WeightId, WeightMap, WeightRole,
};
use poot_quant::{PackedWeight, SourceRole};

/// How the weights of a model are divided over the ranks of a tensor-parallel group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardRule {
    /// The one family-neutral rule: column-parallel Q/K/V/gate/up (and their biases and adapter `B`
    /// factors), row-parallel O/down (and their adapter `A` factors), everything else replicated.
    Megatron,
}

/// Why a graph cannot be sharded. Raised before any stage is compiled.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ShardError {
    /// Equation `eqn` (`op`) reads a value split on `axis` and no propagation rule covers it.
    #[error("equation {eqn} ({op}) has no sharding rule for an operand split on axis {axis}")]
    Unpropagatable {
        /// Index of the equation.
        eqn: usize,
        /// Name of the equation's op.
        op: String,
        /// The axis the operand is split on.
        axis: usize,
    },
    /// The graph reads weight `weight` and the rule has no split for its role: multi-head latent
    /// attention, routed experts, mixers.
    #[error("weight {weight} ({role:?}) has no {rule:?} sharding rule")]
    NoRule {
        /// The weight no rule covers.
        weight: WeightId,
        /// The weight's role.
        role: WeightRole,
        /// The rule that has no split for the role.
        rule: ShardRule,
    },
    /// An extent the split divides does not divide over the ranks.
    #[error("{what}: extent {extent} does not divide over {world} ranks")]
    NotDivisible {
        /// What was being divided, for the message.
        what: String,
        /// The extent that must divide.
        extent: usize,
        /// The number of ranks.
        world: u32,
    },
    /// A packed weight's share would cut an encoding block, or the format has no per-row blocks.
    #[error(
        "weight {weight}: splitting {format:?} {shape:?} on axis {axis} cuts an encoding block"
    )]
    IllegalEncodingSplit {
        /// The weight being split.
        weight: WeightId,
        /// The weight's packed format.
        format: poot_quant::format::WeightFormat,
        /// The weight's logical `[out, in]` shape.
        shape: [usize; 2],
        /// The axis the rule splits.
        axis: usize,
    },
    /// A const names a weight of the map in a form the map does not store it in (a packed source of a
    /// dense weight, or a dense const of a packed one): its share cannot be sized.
    #[error("const {name} reads weight {weight} in a form it is not stored in")]
    WeightForm {
        /// The weight the const reads.
        weight: WeightId,
        /// Name of the const.
        name: String,
    },
    /// A carried value is written with a layout it was not read with, or never decided.
    #[error("carried state {name}: written {written:?} but read {read:?}")]
    StateLayout {
        /// Name of the carried value.
        name: String,
        /// The layout it is read with.
        read: String,
        /// The layout it is written with.
        written: String,
    },
    /// A validation witness, the output or an export is not the same on every rank.
    #[error("{what} is split across ranks")]
    SplitResult {
        /// What is split across ranks.
        what: &'static str,
    },
    /// The final stage of the lowered program does not hold the graph's output.
    #[error("the output is not computed by the final stage of the rank program")]
    OutputNotLast,
    /// A rewrite produced a graph the edit check refused: a defect of the pass, never of the input.
    #[error(transparent)]
    Edit(Box<poot_graph_ir::EditError>),
}

impl From<poot_graph_ir::EditError> for ShardError {
    fn from(error: poot_graph_ir::EditError) -> Self {
        Self::Edit(Box::new(error))
    }
}

/// A sharded graph: the program every rank runs.
#[derive(Clone, Debug)]
pub struct ShardedGraph {
    /// The traced graph over per-rank shapes, with its collectives as equations.
    pub graph: Graph,
    /// Each weight the rule cuts and the axis of its mapped logical shape that is divided.
    pub weights: Vec<(WeightId, usize)>,
    /// The result of every collective equation, in equation order.
    pub collectives: Vec<ValueId>,
    /// The number of ranks the graph was sharded for.
    pub world: NonZeroU32,
}

impl ShardedGraph {
    /// The numerics this transform declares. An `AllReduce(Sum)` adds the ranks' partial sums in rank
    /// order, a different summation order from the contraction one device runs over the whole axis, so a
    /// graph with one is [`NumericsProperty::Reassociating`]; a column-parallel cut and an `AllGather` only
    /// move values and stay exact.
    pub fn numerics(&self) -> NumericsProperty {
        let reduces = self
            .graph
            .eqns()
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::AllReduce { .. }));
        if reduces {
            NumericsProperty::Reassociating
        } else {
            NumericsProperty::Exact
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    Whole,
    Split(usize),
    Partial,
    Undecided,
}

impl Layout {
    fn describe(self) -> String {
        match self {
            Layout::Whole => "whole".into(),
            Layout::Split(axis) => format!("split on axis {axis}"),
            Layout::Partial => "partial".into(),
            Layout::Undecided => "undecided".into(),
        }
    }
}

/// How a rule divides one weight.
enum Share {
    /// Cut on this axis of the weight's mapped logical shape.
    Axis(usize),
    /// Every rank reads it whole.
    Whole,
    /// The rule has no answer for the role.
    NoRule,
}

/// How `rule` divides a weight of `role`.
fn weight_share(rule: ShardRule, role: WeightRole) -> Share {
    let ShardRule::Megatron = rule;
    let column_parallel = |target: LoraTarget| {
        matches!(
            target,
            LoraTarget::AttnQ
                | LoraTarget::AttnK
                | LoraTarget::AttnV
                | LoraTarget::FfnGate
                | LoraTarget::FfnUp
        )
    };
    let axis_if = |cut: bool, axis: usize| if cut { Share::Axis(axis) } else { Share::Whole };
    match role {
        WeightRole::Attn(AttnRole::Q | AttnRole::K | AttnRole::V) => Share::Axis(0),
        WeightRole::Attn(AttnRole::QBias | AttnRole::KBias | AttnRole::VBias) => Share::Axis(0),
        WeightRole::Attn(AttnRole::O) => Share::Axis(1),
        WeightRole::Attn(AttnRole::OBias | AttnRole::QNorm | AttnRole::KNorm) => Share::Whole,
        WeightRole::Ffn(FfnRole::Gate | FfnRole::Up) => Share::Axis(0),
        WeightRole::Ffn(FfnRole::UpBias | FfnRole::GateBias) => Share::Axis(0),
        WeightRole::Ffn(FfnRole::Down) => Share::Axis(1),
        WeightRole::Ffn(FfnRole::DownBias) => Share::Whole,
        WeightRole::Embed
        | WeightRole::Head
        | WeightRole::FinalNorm
        | WeightRole::FinalNormBias
        | WeightRole::Norm(_) => Share::Whole,
        // Stacked `[adapters, in, r]` / `[adapters, r, out]` factors of the `x @ W` orientation: the
        // factor that touches the split side of its projection is cut, the other is whole.
        WeightRole::Lora(LoraRole::A(target)) => axis_if(!column_parallel(target), 1),
        WeightRole::Lora(LoraRole::B(target)) => axis_if(column_parallel(target), 2),
        WeightRole::Lora(LoraRole::Scale) => Share::Whole,
        _ => Share::NoRule,
    }
}

/// Rewrite `g` for `world` ranks under `rule`: per-rank weights and shapes, one collective where ranks
/// must agree. `weights` supplies each const's role and stored format.
pub fn shard(
    g: &Graph,
    weights: &WeightMap,
    rule: ShardRule,
    world: NonZeroU32,
) -> Result<ShardedGraph, ShardError> {
    if world.get() == 1 {
        return Ok(ShardedGraph {
            graph: g.clone(),
            weights: Vec::new(),
            collectives: Vec::new(),
            world,
        });
    }
    // A carried cache takes the layout of what is written into it, which only its writers know and
    // which every reader of it must agree with: a first walk finds it, the second applies it.
    let mut probe = Sharder::new(g, weights, rule, world, None);
    probe.walk()?;
    let states = probe.decided;
    let mut sharder = Sharder::new(g, weights, rule, world, Some(states));
    sharder.walk()?;
    sharder.finalize()
}

struct Sharder<'a> {
    g: &'a Graph,
    rule: ShardRule,
    world: usize,
    world_n: NonZeroU32,
    /// Const name of every mapped weight, and the weight it names.
    ids: HashMap<String, WeightId>,
    weights: &'a WeightMap,
    values: ValueTable<ValueMeta>,
    layout: ValueTable<Layout>,
    eqns: Vec<Eqn>,
    /// The `AllReduce(Sum)` of a partial, once emitted: later readers share it.
    reduced: HashMap<ValueId, ValueId>,
    /// The `AllGather` of a split value, once emitted.
    gathered: HashMap<ValueId, ValueId>,
    collectives: Vec<ValueId>,
    cut: Vec<(WeightId, usize)>,
    /// The layout of each carried-state input: `None` while probing (every cache is `Undecided`),
    /// the probe's findings when applying.
    states: Option<HashMap<ValueId, Layout>>,
    /// What the probe found: the layout of the value written into each cache.
    decided: HashMap<ValueId, Layout>,
    /// The carried-state input an `Undecided` value is derived from.
    origin: HashMap<ValueId, ValueId>,
    /// The layer and source index of the equation being rewritten.
    layer: Option<LayerIndex>,
    index: usize,
}

type Rewrite = Result<(OpKind, Vec<Operand>, Layout), ShardError>;

impl<'a> Sharder<'a> {
    fn new(
        g: &'a Graph,
        weights: &'a WeightMap,
        rule: ShardRule,
        world: NonZeroU32,
        states: Option<HashMap<ValueId, Layout>>,
    ) -> Self {
        let ids = weights
            .iter()
            .map(|(id, _, _)| (id.const_name(), id))
            .collect();
        Self {
            g,
            rule,
            world: world.get() as usize,
            world_n: world,
            ids,
            weights,
            values: g.values().clone(),
            layout: ValueTable::filled(g.values().len(), Layout::Whole),
            eqns: Vec::with_capacity(g.eqns().len()),
            reduced: HashMap::new(),
            gathered: HashMap::new(),
            collectives: Vec::new(),
            cut: Vec::new(),
            states,
            decided: HashMap::new(),
            origin: HashMap::new(),
            layer: None,
            index: 0,
        }
    }

    fn walk(&mut self) -> Result<(), ShardError> {
        for &input in self.g.inputs() {
            self.bind_input(input)?;
        }
        for (index, eqn) in self.g.eqns().iter().enumerate() {
            self.index = index;
            self.layer = eqn.layer;
            let (op, inputs, layout) = self.rewrite(eqn)?;
            self.finish(eqn, op, inputs, layout)?;
        }
        Ok(())
    }

    fn weight_of(&self, name: &str) -> Option<(WeightId, Option<SourceRole>)> {
        match self.ids.get(name) {
            Some(&id) => Some((id, None)),
            None => {
                let source = PackedSourceName::parse(name)?;
                let id = *self.ids.get(source.linear_id())?;
                Some((id, Some(source.role())))
            }
        }
    }

    /// The layout of an input binder, and its per-rank type when the rule cuts it.
    fn bind_input(&mut self, input: ValueId) -> Result<(), ShardError> {
        let meta = self.g.meta(input);
        let layout = match meta.storage {
            Storage::Const => {
                let Some(name) = meta.name.as_deref() else {
                    return Ok(());
                };
                let Some((id, source)) = self.weight_of(name) else {
                    // A table the family computes, or a weight the map does not name: read whole.
                    return Ok(());
                };
                let axis = match weight_share(self.rule, id.role) {
                    Share::Axis(axis) => axis,
                    Share::Whole => return Ok(()),
                    Share::NoRule => {
                        return Err(ShardError::NoRule {
                            weight: id,
                            role: id.role,
                            rule: self.rule,
                        });
                    }
                };
                self.cut_weight(input, name, id, source, axis)?;
                Layout::Split(axis)
            }
            // The sampler's history is the same on every rank: the suffix draws from logits every rank holds.
            Storage::State if self.g.state_role(input) == Some(StateRole::Sampler) => Layout::Whole,
            Storage::State => match &self.states {
                None => {
                    self.origin.insert(input, input);
                    Layout::Undecided
                }
                Some(states) => {
                    let layout = states.get(&input).copied().unwrap_or(Layout::Whole);
                    if let Layout::Split(axis) = layout {
                        let aval = &mut self.values[input].aval;
                        let extent = aval.shape[axis];
                        if !extent.is_multiple_of(self.world) {
                            return Err(ShardError::NotDivisible {
                                what: format!(
                                    "carried state {}",
                                    self.g.meta(input).name.as_deref().unwrap_or("?")
                                ),
                                extent,
                                world: self.world_n.get(),
                            });
                        }
                        aval.shape[axis] = extent / self.world;
                    }
                    layout
                }
            },
            _ => Layout::Whole,
        };
        self.layout[input] = layout;
        Ok(())
    }

    fn cut_weight(
        &mut self,
        input: ValueId,
        name: &str,
        id: WeightId,
        source: Option<SourceRole>,
        axis: usize,
    ) -> Result<(), ShardError> {
        let handle = self
            .weights
            .handle(id)
            .expect("the id was read from this map");
        let world = self.world;
        let extent = *handle
            .shape
            .get(axis)
            .expect("a rule's axis lies inside the weight's shape");
        if !extent.is_multiple_of(world) {
            return Err(ShardError::NotDivisible {
                what: format!("weight {id} axis {axis}"),
                extent,
                world: self.world_n.get(),
            });
        }
        let aval = match (handle.format, source) {
            (HandleFormat::Dense(_), None) => {
                let mut aval = self.g.aval(input).clone();
                aval.shape[axis] /= world;
                aval
            }
            (HandleFormat::Packed(weight), Some(role)) => {
                let mut shape = weight.shape();
                shape[axis] /= world;
                let illegal = || ShardError::IllegalEncodingSplit {
                    weight: id,
                    format: weight.format(),
                    shape: weight.shape(),
                    axis,
                };
                let FormatStorage::Blocks(_) = weight.format().descriptor().storage else {
                    return Err(illegal());
                };
                // A share that is not whole blocks is not a packed weight of that format.
                let local = PackedWeight::try_new(weight.format(), shape).map_err(|_| illegal())?;
                packed_source_type(local, role)
            }
            _ => {
                return Err(ShardError::WeightForm {
                    weight: id,
                    name: name.to_string(),
                });
            }
        };
        self.values[input].aval = aval;
        if !self.cut.iter().any(|&(seen, _)| seen == id) {
            self.cut.push((id, axis));
        }
        Ok(())
    }

    fn operand_layout(&self, operand: Operand) -> Layout {
        match operand {
            Operand::Value(value) => self.layout[value],
            Operand::Lit(_) => Layout::Whole,
        }
    }

    fn operand_shape(&self, operand: Operand) -> Vec<usize> {
        match operand {
            Operand::Value(value) => self.values[value].aval.shape.clone(),
            Operand::Lit(_) => Vec::new(),
        }
    }

    fn refuse(&self, eqn: &Eqn, axis: usize) -> ShardError {
        ShardError::Unpropagatable {
            eqn: self.index,
            op: eqn.op.name(),
            axis,
        }
    }

    /// The first split axis among `layouts`, for an error naming it.
    fn first_axis(layouts: &[Layout]) -> usize {
        layouts
            .iter()
            .find_map(|layout| match layout {
                Layout::Split(axis) => Some(*axis),
                _ => None,
            })
            .unwrap_or(0)
    }

    fn push_value(&mut self, aval: TensorType) -> ValueId {
        self.layout.push(Layout::Whole);
        self.values
            .push(ValueMeta::new(aval, Storage::Device, None))
    }

    /// The `AllReduce(Sum)` of partial `value`: emitted once, in front of its first non-linear reader.
    fn all_reduce(&mut self, value: ValueId) -> ValueId {
        if let Some(&reduced) = self.reduced.get(&value) {
            return reduced;
        }
        let aval = self.values[value].aval.clone();
        let axis = aval.shape.len().saturating_sub(1);
        let out = self.push_value(aval);
        self.eqns.push(Eqn {
            op: OpKind::AllReduce {
                op: RedOp::Sum,
                axis,
            },
            inputs: vec![Operand::Value(value)],
            out,
            layer: self.layer,
        });
        self.collectives.push(out);
        self.reduced.insert(value, out);
        out
    }

    /// The `AllGather` of `value`, split on `axis`.
    fn all_gather(&mut self, value: ValueId, axis: usize) -> ValueId {
        if let Some(&gathered) = self.gathered.get(&value) {
            return gathered;
        }
        let mut aval = self.values[value].aval.clone();
        aval.shape[axis] *= self.world;
        let out = self.push_value(aval);
        self.eqns.push(Eqn {
            op: OpKind::AllGather {
                axis,
                world: self.world_n,
            },
            inputs: vec![Operand::Value(value)],
            out,
            layer: self.layer,
        });
        self.collectives.push(out);
        self.gathered.insert(value, out);
        out
    }

    /// `operand` as a value every rank holds whole: a partial is reduced, a split value gathered.
    fn whole(&mut self, operand: Operand) -> Operand {
        let Operand::Value(value) = operand else {
            return operand;
        };
        Operand::Value(match self.layout[value] {
            Layout::Partial => self.all_reduce(value),
            Layout::Split(axis) => self.all_gather(value, axis),
            Layout::Whole | Layout::Undecided => value,
        })
    }

    /// `operand` with partial sums reduced, other layouts kept.
    fn settled(&mut self, operand: Operand) -> Operand {
        match self.operand_layout(operand) {
            Layout::Partial => self.whole(operand),
            _ => operand,
        }
    }

    fn rewrite(&mut self, eqn: &Eqn) -> Rewrite {
        use OpKind as K;
        let inputs = eqn.inputs.clone();
        match &eqn.op {
            K::Unary(op) => self.pointwise(eqn, inputs, matches!(op, UnOp::Neg)),
            K::Binary(op) => {
                let linear = match op {
                    BinOp::Add | BinOp::Sub => Linear::Additive,
                    BinOp::Mul => Linear::Scaled,
                    BinOp::Div => Linear::Divided,
                    _ => Linear::None,
                };
                self.binary(eqn, inputs, linear)
            }
            K::Select | K::Cast { .. } => self.pointwise(eqn, inputs, false),
            K::Reduce { op, axis, keepdim } => self.reduce(eqn, inputs, *op, *axis, *keepdim),
            K::Broadcast { shape } => self.broadcast(eqn, inputs, shape),
            K::Reshape { shape } => self.reshape(eqn, inputs, shape),
            K::Transpose { perm } => {
                let layout = match self.operand_layout(inputs[0]) {
                    Layout::Split(axis) => {
                        Layout::Split(perm.iter().position(|&p| p == axis).expect("a permutation"))
                    }
                    other => other,
                };
                Ok((eqn.op.clone(), inputs, layout))
            }
            K::Slice { axis, .. } => match self.operand_layout(inputs[0]) {
                Layout::Split(split) if split == *axis => Err(self.refuse(eqn, split)),
                layout => Ok((eqn.op.clone(), inputs, layout)),
            },
            K::Concat { axis } => self.concat(eqn, inputs, *axis),
            K::Iota { .. } => Ok((eqn.op.clone(), inputs, Layout::Whole)),
            K::Gather { axis } => self.gather(eqn, inputs, *axis),
            K::MatMul => self.matmul(eqn, inputs),
            K::IndexedMatMul => self.indexed_matmul(eqn, inputs),
            K::DynamicUpdateSlice { axis } => self.update_slice(eqn, inputs, *axis),
            K::ScatterUpdate => self.scatter_update(eqn, inputs),
            K::PackedDequant { descriptor } => self.packed_dequant(eqn, inputs, *descriptor),
            K::AllReduce { .. } | K::AllGather { .. } => Err(self.refuse(eqn, 0)),
            _ => self.replicated(eqn, inputs),
        }
    }

    /// An op with no sharding rule of its own: it runs replicated when every operand is whole (partials
    /// reduced first), and is otherwise unpropagatable.
    fn replicated(&mut self, eqn: &Eqn, inputs: Vec<Operand>) -> Rewrite {
        let inputs: Vec<Operand> = inputs
            .into_iter()
            .map(|operand| self.settled(operand))
            .collect();
        let layouts: Vec<Layout> = inputs.iter().map(|&o| self.operand_layout(o)).collect();
        if layouts
            .iter()
            .any(|l| matches!(l, Layout::Split(_) | Layout::Undecided))
        {
            return Err(self.refuse(eqn, Self::first_axis(&layouts)));
        }
        Ok((eqn.op.clone(), inputs, Layout::Whole))
    }

    /// One-operand-class pointwise op (`linear` is whether it commutes with a sum over ranks).
    fn pointwise(&mut self, eqn: &Eqn, inputs: Vec<Operand>, linear: bool) -> Rewrite {
        let mut inputs = inputs;
        if !(linear && inputs.len() == 1) {
            for operand in &mut inputs {
                *operand = self.settled(*operand);
            }
        }
        self.align(eqn, inputs)
    }

    fn binary(&mut self, eqn: &Eqn, mut inputs: Vec<Operand>, linear: Linear) -> Rewrite {
        let layouts: Vec<Layout> = inputs.iter().map(|&o| self.operand_layout(o)).collect();
        let partial = |i: usize| layouts[i] == Layout::Partial;
        let whole = |i: usize| layouts[i] == Layout::Whole;
        // A partial stays a partial through an operation that is linear in it, with the other
        // operand a value every rank holds identically.
        let keeps_partial = match linear {
            Linear::Additive => partial(0) && partial(1),
            Linear::Scaled => (partial(0) && whole(1)) || (whole(0) && partial(1)),
            Linear::Divided => partial(0) && whole(1),
            Linear::None => false,
        };
        if keeps_partial {
            let (op, inputs, _) = self.align(eqn, inputs)?;
            return Ok((op, inputs, Layout::Partial));
        }
        for operand in &mut inputs {
            *operand = self.settled(*operand);
        }
        self.align(eqn, inputs)
    }

    /// Pointwise broadcasting: every split operand splits the output on one axis and every other
    /// operand is broadcast along it.
    fn align(&mut self, eqn: &Eqn, inputs: Vec<Operand>) -> Rewrite {
        let layouts: Vec<Layout> = inputs.iter().map(|&o| self.operand_layout(o)).collect();
        if layouts.contains(&Layout::Undecided) {
            return Err(self.refuse(eqn, 0));
        }
        let shapes: Vec<Vec<usize>> = inputs.iter().map(|&o| self.operand_shape(o)).collect();
        let rank = shapes.iter().map(Vec::len).max().unwrap_or(0);
        let mut split_out: Option<usize> = None;
        for (layout, shape) in layouts.iter().zip(&shapes) {
            if let Layout::Split(axis) = layout {
                let out_axis = axis + (rank - shape.len());
                match split_out {
                    Some(seen) if seen != out_axis => return Err(self.refuse(eqn, out_axis)),
                    _ => split_out = Some(out_axis),
                }
            }
        }
        if let Some(axis) = split_out {
            for (layout, shape) in layouts.iter().zip(&shapes) {
                if matches!(layout, Layout::Split(_)) {
                    continue;
                }
                let offset = rank - shape.len();
                // A whole operand must be broadcast along the split axis: absent or extent 1.
                if axis >= offset && shape[axis - offset] != 1 {
                    return Err(self.refuse(eqn, axis));
                }
            }
        }
        let layout = match split_out {
            Some(axis) => Layout::Split(axis),
            None if layouts.contains(&Layout::Partial) => Layout::Partial,
            None => Layout::Whole,
        };
        Ok((eqn.op.clone(), inputs, layout))
    }

    fn reduce(
        &mut self,
        eqn: &Eqn,
        inputs: Vec<Operand>,
        op: RedOp,
        axis: usize,
        keepdim: bool,
    ) -> Rewrite {
        let operand = inputs[0];
        match self.operand_layout(operand) {
            Layout::Partial if op == RedOp::Sum => Ok((eqn.op.clone(), inputs, Layout::Partial)),
            Layout::Partial => {
                let whole = self.whole(operand);
                Ok((eqn.op.clone(), vec![whole], Layout::Whole))
            }
            Layout::Split(split) if split == axis => {
                if op == RedOp::Sum {
                    Ok((eqn.op.clone(), inputs, Layout::Partial))
                } else {
                    // A maximum needs the whole axis.
                    let whole = self.whole(operand);
                    Ok((eqn.op.clone(), vec![whole], Layout::Whole))
                }
            }
            Layout::Split(split) => {
                let moved = if keepdim || axis > split {
                    split
                } else {
                    split - 1
                };
                Ok((eqn.op.clone(), inputs, Layout::Split(moved)))
            }
            Layout::Undecided => Err(self.refuse(eqn, 0)),
            Layout::Whole => Ok((eqn.op.clone(), inputs, Layout::Whole)),
        }
    }

    fn broadcast(&mut self, eqn: &Eqn, inputs: Vec<Operand>, target: &[usize]) -> Rewrite {
        let layout = self.operand_layout(inputs[0]);
        let Layout::Split(axis) = layout else {
            if layout == Layout::Undecided {
                return Err(self.refuse(eqn, 0));
            }
            return Ok((eqn.op.clone(), inputs, layout));
        };
        let local = self.operand_shape(inputs[0]);
        let out_axis = axis + (target.len() - local.len());
        // The split extent must carry over unexpanded: the global extent is `local * world`.
        if target[out_axis] != local[axis] * self.world {
            return Err(self.refuse(eqn, axis));
        }
        let mut shape = target.to_vec();
        shape[out_axis] /= self.world;
        Ok((OpKind::Broadcast { shape }, inputs, Layout::Split(out_axis)))
    }

    fn reshape(&mut self, eqn: &Eqn, inputs: Vec<Operand>, target: &[usize]) -> Rewrite {
        let layout = self.operand_layout(inputs[0]);
        let Layout::Split(axis) = layout else {
            if layout == Layout::Undecided {
                return Err(self.refuse(eqn, 0));
            }
            return Ok((eqn.op.clone(), inputs, layout));
        };
        // The split axis's leading axes' product must reappear as a prefix of the target, and the
        // target axis that starts the split tail must divide over the ranks: then each rank's
        // contiguous chunk of the tail is its slice of that axis.
        let local = self.operand_shape(inputs[0]);
        let prefix: usize = local[..axis].iter().product();
        let mut running = 1usize;
        for (out_axis, &extent) in target.iter().enumerate() {
            if running == prefix && extent > 1 {
                // The first axis of the split tail: a chunk that is not a whole number of its entries
                // (a head count that does not divide over the ranks) is no slice of the axis.
                if !extent.is_multiple_of(self.world) {
                    return Err(ShardError::NotDivisible {
                        what: format!(
                            "equation {} ({}) result axis {out_axis}",
                            self.index,
                            eqn.op.name()
                        ),
                        extent,
                        world: self.world_n.get(),
                    });
                }
                let mut shape = target.to_vec();
                shape[out_axis] /= self.world;
                // Leading unit axes before `out_axis` do not change the prefix product, so the first
                // divisible axis is the one the chunk lies on.
                return Ok((OpKind::Reshape { shape }, inputs, Layout::Split(out_axis)));
            }
            running *= extent;
            if running > prefix {
                break;
            }
        }
        Err(self.refuse(eqn, axis))
    }

    fn concat(&mut self, eqn: &Eqn, inputs: Vec<Operand>, axis: usize) -> Rewrite {
        let mut inputs: Vec<Operand> = inputs;
        let layouts: Vec<Layout> = inputs.iter().map(|&o| self.operand_layout(o)).collect();
        if layouts.iter().all(|l| *l == Layout::Partial) {
            return Ok((eqn.op.clone(), inputs, Layout::Partial));
        }
        for operand in &mut inputs {
            *operand = self.settled(*operand);
        }
        let layouts: Vec<Layout> = inputs.iter().map(|&o| self.operand_layout(o)).collect();
        let first = layouts[0];
        if layouts.contains(&Layout::Undecided) || layouts.iter().any(|l| *l != first) {
            return Err(self.refuse(eqn, Self::first_axis(&layouts)));
        }
        if first == Layout::Split(axis) {
            return Err(self.refuse(eqn, axis));
        }
        Ok((eqn.op.clone(), inputs, first))
    }

    fn gather(&mut self, eqn: &Eqn, inputs: Vec<Operand>, axis: usize) -> Rewrite {
        let inputs: Vec<Operand> = inputs.into_iter().map(|o| self.settled(o)).collect();
        let (data, index) = (
            self.operand_layout(inputs[0]),
            self.operand_layout(inputs[1]),
        );
        let index_rank = self.operand_shape(inputs[1]).len();
        let layout = match (data, index) {
            (Layout::Whole, Layout::Whole) => Layout::Whole,
            (Layout::Split(split), Layout::Whole) if split < axis => Layout::Split(split),
            (Layout::Split(split), Layout::Whole) if split > axis => {
                Layout::Split(split - 1 + index_rank)
            }
            (Layout::Whole, Layout::Split(split)) => Layout::Split(axis + split),
            _ => return Err(self.refuse(eqn, Self::first_axis(&[data, index]))),
        };
        Ok((eqn.op.clone(), inputs, layout))
    }

    fn matmul(&mut self, eqn: &Eqn, inputs: Vec<Operand>) -> Rewrite {
        let (a, b) = (inputs[0], inputs[1]);
        let (la, lb) = (self.operand_layout(a), self.operand_layout(b));
        let (sa, sb) = (self.operand_shape(a), self.operand_shape(b));
        let (ra, rb) = (sa.len(), sb.len());
        let rank = ra.max(rb);
        // Batch axes line up from the right, like every broadcast: `a`'s axis `i` is output axis
        // `i + rank - ra`, and a batch axis is any axis before the last two.
        let batch = |axis: usize, operand_rank: usize| axis + 2 < operand_rank;
        let layout = match (la, lb) {
            (Layout::Whole, Layout::Whole) => Layout::Whole,
            (Layout::Partial, Layout::Whole) | (Layout::Whole, Layout::Partial) => Layout::Partial,
            // Column-parallel: the weight's output axis is split.
            (Layout::Whole, Layout::Split(j)) if j == rb - 1 => Layout::Split(rank - 1),
            // Row-parallel: both sides split on the contracted axis.
            (Layout::Split(i), Layout::Split(j)) if i == ra - 1 && j + 2 == rb => Layout::Partial,
            // Batched (heads): both sides split on the same batch axis.
            (Layout::Split(i), Layout::Split(j))
                if batch(i, ra) && batch(j, rb) && i + rank - ra == j + rank - rb =>
            {
                Layout::Split(i + rank - ra)
            }
            // One side split on a batch axis the other broadcasts along.
            (Layout::Split(i), Layout::Whole) if batch(i, ra) => {
                let out = i + rank - ra;
                if out >= rank - rb && sb[out - (rank - rb)] != 1 {
                    return Err(self.refuse(eqn, i));
                }
                Layout::Split(out)
            }
            (Layout::Whole, Layout::Split(j)) if batch(j, rb) => {
                let out = j + rank - rb;
                if out >= rank - ra && sa[out - (rank - ra)] != 1 {
                    return Err(self.refuse(eqn, j));
                }
                Layout::Split(out)
            }
            // Rows of the activation split, the weight whole.
            (Layout::Split(i), Layout::Whole) if i + 2 == ra => Layout::Split(rank - 2),
            (Layout::Partial, Layout::Partial) => {
                let a = self.whole(a);
                return self.matmul(eqn, vec![a, b]);
            }
            _ => return Err(self.refuse(eqn, Self::first_axis(&[la, lb]))),
        };
        Ok((eqn.op.clone(), inputs, layout))
    }

    fn indexed_matmul(&mut self, eqn: &Eqn, inputs: Vec<Operand>) -> Rewrite {
        let index = self.settled(inputs[2]);
        if self.operand_layout(index) != Layout::Whole {
            return Err(self.refuse(eqn, 0));
        }
        let (x, w) = (inputs[0], inputs[1]);
        let layout = match (self.operand_layout(x), self.operand_layout(w)) {
            (Layout::Whole, Layout::Whole) => Layout::Whole,
            (Layout::Whole, Layout::Split(2)) => Layout::Split(1),
            (Layout::Split(1), Layout::Split(1)) => Layout::Partial,
            (Layout::Partial, Layout::Whole) => Layout::Partial,
            (lx, lw) => return Err(self.refuse(eqn, Self::first_axis(&[lx, lw]))),
        };
        Ok((eqn.op.clone(), vec![x, w, index], layout))
    }

    /// The cache an undecided `operand` derives from takes the layout of what is written into it; two
    /// writes with different layouts are a refusal.
    fn decide_cache(&mut self, operand: Operand, decided: Layout) -> Result<(), ShardError> {
        let root = match operand {
            Operand::Value(value) => self.origin[&value],
            Operand::Lit(_) => unreachable!("an undecided operand is a value"),
        };
        if self
            .decided
            .insert(root, decided)
            .is_some_and(|seen| seen != decided)
        {
            return Err(ShardError::StateLayout {
                name: self.g.meta(root).name.clone().unwrap_or_default(),
                read: "written with two layouts".into(),
                written: decided.describe(),
            });
        }
        Ok(())
    }

    fn update_slice(&mut self, eqn: &Eqn, inputs: Vec<Operand>, axis: usize) -> Rewrite {
        let mut inputs = inputs;
        for operand in &mut inputs {
            *operand = self.settled(*operand);
        }
        for &position in &inputs[2..] {
            if self.operand_layout(position) != Layout::Whole {
                return Err(self.refuse(eqn, 0));
            }
        }
        let (target, update) = (
            self.operand_layout(inputs[0]),
            self.operand_layout(inputs[1]),
        );
        let layout = match (target, update) {
            (Layout::Undecided, decided @ (Layout::Whole | Layout::Split(_))) => {
                if let Layout::Split(split) = decided
                    && split == axis
                {
                    return Err(self.refuse(eqn, split));
                }
                self.decide_cache(inputs[0], decided)?;
                decided
            }
            (Layout::Whole, Layout::Whole) => Layout::Whole,
            (Layout::Split(a), Layout::Split(b)) if a == b && a != axis => Layout::Split(a),
            (lt, lu) => return Err(self.refuse(eqn, Self::first_axis(&[lt, lu]))),
        };
        Ok((eqn.op.clone(), inputs, layout))
    }

    /// The paged KV write: `out = base`, then `out[slots[t], ..] = src[t, ..]` for each `slots[t] >= 0`.
    /// Rows are indexed on axis 0 by the whole-valued slot map, so the write divides over ranks on any
    /// trailing axis (the KV heads): each rank scatters its own heads' rows into its own pool, and the
    /// pool takes the layout of what is written into it exactly like a contiguous cache.
    fn scatter_update(&mut self, eqn: &Eqn, inputs: Vec<Operand>) -> Rewrite {
        let mut inputs = inputs;
        for operand in &mut inputs {
            *operand = self.settled(*operand);
        }
        if self.operand_layout(inputs[2]) != Layout::Whole {
            return Err(self.refuse(eqn, 0));
        }
        let (base, src) = (
            self.operand_layout(inputs[0]),
            self.operand_layout(inputs[1]),
        );
        let layout = match (base, src) {
            (Layout::Undecided, decided @ (Layout::Whole | Layout::Split(_))) => {
                if decided == Layout::Split(0) {
                    return Err(self.refuse(eqn, 0));
                }
                self.decide_cache(inputs[0], decided)?;
                decided
            }
            (Layout::Whole, Layout::Whole) => Layout::Whole,
            (Layout::Split(a), Layout::Split(b)) if a == b && a != 0 => Layout::Split(a),
            (lb, ls) => return Err(self.refuse(eqn, Self::first_axis(&[lb, ls]))),
        };
        Ok((eqn.op.clone(), inputs, layout))
    }

    fn packed_dequant(
        &mut self,
        eqn: &Eqn,
        inputs: Vec<Operand>,
        descriptor: PackedWeight,
    ) -> Rewrite {
        let layouts: Vec<Layout> = inputs.iter().map(|&o| self.operand_layout(o)).collect();
        match layouts.first() {
            Some(&Layout::Split(axis)) => {
                let mut shape = descriptor.shape();
                shape[axis] /= self.world;
                let local = PackedWeight::try_new(descriptor.format(), shape)
                    .map_err(|_| self.refuse(eqn, axis))?;
                Ok((
                    OpKind::PackedDequant { descriptor: local },
                    inputs,
                    Layout::Split(axis),
                ))
            }
            _ => Ok((eqn.op.clone(), inputs, Layout::Whole)),
        }
    }

    /// Record the rewritten equation: its per-rank result type and layout.
    fn finish(
        &mut self,
        eqn: &Eqn,
        op: OpKind,
        inputs: Vec<Operand>,
        layout: Layout,
    ) -> Result<(), ShardError> {
        let mut aval = self.g.aval(eqn.out).clone();
        if let Layout::Split(axis) = layout {
            let extent = aval.shape[axis];
            if !extent.is_multiple_of(self.world) {
                return Err(ShardError::NotDivisible {
                    what: format!(
                        "equation {} ({}) result axis {axis}",
                        self.index,
                        eqn.op.name()
                    ),
                    extent,
                    world: self.world_n.get(),
                });
            }
            aval.shape[axis] = extent / self.world;
        }
        if layout == Layout::Undecided
            && let Some(root) = inputs.iter().find_map(|operand| match operand {
                Operand::Value(value) => self.origin.get(value).copied(),
                Operand::Lit(_) => None,
            })
        {
            self.origin.insert(eqn.out, root);
        }
        self.values[eqn.out].aval = aval;
        self.layout[eqn.out] = layout;
        self.eqns.push(Eqn {
            op,
            inputs,
            out: eqn.out,
            layer: eqn.layer,
        });
        Ok(())
    }

    fn finalize(mut self) -> Result<ShardedGraph, ShardError> {
        // The output is a value every rank holds.
        let output = self.g.result();
        let mut new_output = output;
        if matches!(self.layout[output], Layout::Partial | Layout::Split(_)) {
            self.layer = None;
            new_output = match self.whole(Operand::Value(output)) {
                Operand::Value(value) => value,
                Operand::Lit(_) => unreachable!("an output is a value"),
            };
        }
        // Carried state is written with the layout it is read with.
        let mut state: Vec<(ValueId, ValueId)> = self.g.state().collect();
        for pair in &mut state {
            let (read, written) = (self.layout[pair.0], self.layout[pair.1]);
            let name = self.g.meta(pair.0).name.clone().unwrap_or_default();
            let read = if read == Layout::Undecided {
                Layout::Whole
            } else {
                read
            };
            if read == Layout::Whole && written == Layout::Partial {
                pair.1 = match self.whole(Operand::Value(pair.1)) {
                    Operand::Value(value) => value,
                    Operand::Lit(_) => unreachable!("a state output is a value"),
                };
                continue;
            }
            if read != written && !(read == Layout::Whole && written == Layout::Undecided) {
                return Err(ShardError::StateLayout {
                    name,
                    read: read.describe(),
                    written: written.describe(),
                });
            }
        }
        for declaration in self.g.witnesses() {
            if self.layout[declaration.value] != Layout::Whole {
                return Err(ShardError::SplitResult {
                    what: "a validation witness",
                });
            }
        }
        for export in self.g.exports() {
            if self.layout[export.value] != Layout::Whole {
                return Err(ShardError::SplitResult { what: "an export" });
            }
        }
        let mut graph = self.g.edit();
        *graph.values_mut() = self.values;
        *graph.eqns_mut() = self.eqns;
        graph.set_result(new_output);
        for (input, next) in state {
            graph.set_state_next(input, next);
        }
        Ok(ShardedGraph {
            graph: graph.commit()?,
            weights: self.cut,
            collectives: self.collectives,
            world: self.world_n,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Linear {
    /// `a + b`, `a - b`: partial only with a partial on both sides.
    Additive,
    /// `a * b`: partial with one partial side.
    Scaled,
    /// `a / b`: partial with a partial numerator.
    Divided,
    None,
}

/// The rank program [`expand_ranks`] lowers a sharded graph to: one graph holding every rank's
/// equations, and where its stages begin.
#[derive(Debug)]
pub(crate) struct RankProgram {
    pub graph: Graph,
    /// The equation each stage after the first starts at, strictly increasing.
    pub cuts: Vec<usize>,
    /// The rank each stage runs on, one entry per stage (`cuts.len() + 1`).
    pub stage_ranks: Vec<usize>,
}

/// Lower the collectives of `sharded` to the point-to-point exchanges the stage split derives, and
/// unroll the graph once per rank. See the module docs.
pub(crate) fn expand_ranks(sharded: &ShardedGraph) -> Result<RankProgram, ShardError> {
    let g = &sharded.graph;
    let world = sharded.world.get() as usize;

    let mut values = g.values().clone();
    let mut inputs = g.inputs().to_vec();
    let mut consts = g.consts().to_vec();
    // Rank `r >= 1`'s copy of each value it computes or carries; rank 0 keeps the original ids and
    // names, and consts, slots and computed constants are shared by every rank.
    let mut copies: Vec<HashMap<ValueId, ValueId>> = vec![HashMap::new(); world];
    let mut state: Vec<(ValueId, ValueId)> = g.state().collect();
    // The sampler's history lives on rank 0 alone: the suffix is computed once, from logits rank 0 holds,
    // and the other ranks' copy of it is dead.
    let ranked = |state_in: ValueId| g.state_role(state_in) != Some(StateRole::Sampler);
    for (rank, rank_copies) in copies.iter_mut().enumerate().skip(1) {
        for (state_in, _) in g.state().filter(|&(state_in, _)| ranked(state_in)) {
            let mut meta = g.meta(state_in).clone();
            meta.name = meta.name.map(|name| format!("{name}.tp{rank}"));
            let carried = values.push(meta);
            inputs.push(carried);
            consts.push(carried);
            rank_copies.insert(state_in, carried);
        }
    }
    let at = |copies: &[HashMap<ValueId, ValueId>], rank: usize, value: ValueId| {
        copies[rank].get(&value).copied().unwrap_or(value)
    };

    // Each emitted equation with the segment (collectives at or before it) and rank it belongs to.
    struct Placed {
        segment: usize,
        rank: usize,
        eqn: Eqn,
    }
    let mut emitted: Vec<Placed> = Vec::with_capacity(g.eqns().len() * world);
    let mut segment = 0usize;
    for eqn in g.eqns() {
        let collective = matches!(eqn.op, OpKind::AllReduce { .. } | OpKind::AllGather { .. });
        if collective {
            segment += 1;
        }
        for rank in 0..world {
            let out = if rank == 0 {
                eqn.out
            } else {
                let fresh = values.push(g.meta(eqn.out).clone());
                copies[rank].insert(eqn.out, fresh);
                fresh
            };
            match &eqn.op {
                OpKind::AllReduce { .. } => {
                    let Operand::Value(partial) = eqn.inputs[0] else {
                        unreachable!("a collective reads a value")
                    };
                    // Every rank folds the partials in rank order, so all ranks hold the same bits.
                    let mut sum = at(&copies, 0, partial);
                    for from in 1..world {
                        let next = at(&copies, from, partial);
                        let target = if from + 1 == world {
                            out
                        } else {
                            values.push(g.meta(eqn.out).clone())
                        };
                        emitted.push(Placed {
                            segment,
                            rank,
                            eqn: Eqn {
                                op: OpKind::Binary(BinOp::Add),
                                inputs: vec![Operand::Value(sum), Operand::Value(next)],
                                out: target,
                                layer: eqn.layer,
                            },
                        });
                        sum = target;
                    }
                }
                OpKind::AllGather { axis, .. } => {
                    let Operand::Value(partial) = eqn.inputs[0] else {
                        unreachable!("a collective reads a value")
                    };
                    emitted.push(Placed {
                        segment,
                        rank,
                        eqn: Eqn {
                            op: OpKind::Concat { axis: *axis },
                            inputs: (0..world)
                                .map(|from| Operand::Value(at(&copies, from, partial)))
                                .collect(),
                            out,
                            layer: eqn.layer,
                        },
                    });
                }
                op => emitted.push(Placed {
                    segment,
                    rank,
                    eqn: Eqn {
                        op: op.clone(),
                        inputs: eqn
                            .inputs
                            .iter()
                            .map(|operand| match *operand {
                                Operand::Value(value) => Operand::Value(at(&copies, rank, value)),
                                lit @ Operand::Lit(_) => lit,
                            })
                            .collect(),
                        out,
                        layer: eqn.layer,
                    },
                }),
            }
        }
    }
    // Within a segment the ranks run highest first, so rank 0, whose copy of the output the program
    // returns, is last.
    emitted.sort_by_key(|placed| (placed.segment, std::cmp::Reverse(placed.rank)));

    // Liveness: the output, every rank's carried state, the witnesses and the exports.
    let mut live: ValueTable<bool> = ValueTable::filled(values.len(), false);
    live[g.result()] = true;
    for rank in 0..world {
        for (state_in, state_out) in g.state() {
            if rank == 0 || ranked(state_in) {
                live[at(&copies, rank, state_out)] = true;
            }
        }
    }
    for declaration in g.witnesses() {
        live[declaration.value] = true;
    }
    for export in g.exports() {
        live[export.value] = true;
    }
    let mut keep = vec![false; emitted.len()];
    for (index, placed) in emitted.iter().enumerate().rev() {
        if live[placed.eqn.out] {
            keep[index] = true;
            for operand in &placed.eqn.inputs {
                if let Operand::Value(value) = *operand {
                    live[value] = true;
                }
            }
        }
    }

    let mut eqns = Vec::new();
    let mut cuts = Vec::new();
    let mut stage_ranks: Vec<usize> = Vec::new();
    for (placed, kept) in emitted.into_iter().zip(keep) {
        if !kept {
            continue;
        }
        if stage_ranks.last() != Some(&placed.rank) {
            if !eqns.is_empty() {
                cuts.push(eqns.len());
            }
            stage_ranks.push(placed.rank);
        }
        eqns.push(placed.eqn);
    }
    let defines_output = eqns
        .iter()
        .rposition(|eqn| eqn.out == g.result())
        .is_none_or(|position| position >= cuts.last().copied().unwrap_or(0));
    if !defines_output {
        return Err(ShardError::OutputNotLast);
    }

    for rank in 1..world {
        state.extend(g.state().filter(|&(state_in, _)| ranked(state_in)).map(
            |(state_in, state_out)| (at(&copies, rank, state_in), at(&copies, rank, state_out)),
        ));
    }
    let mut graph = g.edit();
    *graph.values_mut() = values;
    *graph.inputs_mut() = inputs;
    *graph.consts_mut() = consts;
    *graph.eqns_mut() = eqns;
    graph.retain_state(|_, _| false);
    for (input, next) in state {
        graph.push_state(input, next);
    }
    Ok(RankProgram {
        graph: graph.commit()?,
        cuts,
        stage_ranks,
    })
}
