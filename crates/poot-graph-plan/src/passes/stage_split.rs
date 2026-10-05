//! ADR-0099 S2: the backend-neutral stage split - one traced graph in, one graph per stage plus
//! the values that cross between them.
//!
//! # Two ways to say where the cuts are
//!
//! [`StageAssignment`] (from [`StageAssignment::from_stage_per_equation`] or
//! [`StageAssignment::from_cuts`]) is a stage index per traced equation, validated as
//! **contiguous**: stage `s` owns one non-empty run of the equation list, the runs appear in order
//! (`0, 0, .., 1, 1, .., 2, ..`), and every stage id is used.
//!
//! [`split_stages_after_layers`] is the placement-facing entry: it takes `after_layer` cut indices -
//! exactly what `poot_llm::components::exact_partition::StageBoundary::after_layer` records - and
//! maps each cut onto the first equation tagged with the next layer. The tags come from
//! [`poot_graph_ir::Builder::layer_scope`], which tracers wrap around their layer loop, so the cut list
//! needs no name inference and no per-family equation index. What a family supplies is the placement
//! it already supplied; the tracer's own loop reports where its layers are.
//!
//! Layer tags are checked, not trusted: an untagged equation between the first and last tagged one
//! is [`StageSplitError::UntaggedLayerEquation`], layer ids that are not one run per layer in order
//! are [`StageSplitError::NonContiguousLayer`], and a cut with no layer after it is
//! [`StageSplitError::LayerCutOutOfRange`]. Untagged equations *outside* the layer region - the
//! graph-wide preamble and the head - are expected and belong to the first and last stage.
//!
//! # What crosses
//!
//! A boundary value is an equation result read by an equation in a different stage. It becomes an
//! explicit output of the producing stage ([`StageGraph::outputs`]) and an explicit input of every
//! consuming stage, retagged as [`Slot::Activation`] there. Graph input binders (constants, slots,
//! carried state) are **not** boundary values: each stage binds the ones it reads itself, which is
//! how a replicated weight or a per-replay slot reaches a device without a transfer.
//!
//! Carried state stays whole: a `(state_in, state_out)` pair whose equations span two stages is
//! [`StageSplitError::StatePairSplit`]. Each stage's graph keeps only the pairs it owns, so a KV
//! cache buffer is bound on the device that writes it.
//!
//! Validation witnesses travel with the piece of graph that defines them: each stage keeps the
//! declarations whose value it binds or produces, so a stage graph is still CPU-checkable on its own
//! witnesses.
//!
//! `byte_len` comes from the value's [`TensorType`] (`numel * dtype.byte_size()`, checked) - never a
//! literal size.
//!
//! # Which transforms keep the tags
//!
//! A transform that rewrites or relocates an equation - `cse`,
//! `dce`, fuse, tiling, flash/rope rewrites, dtype and packed rewrites,
//! `to_mixed_bf16`, `lower_nonlast_reduces`, `dtype_widen` - copies `Eqn::layer`, because those all
//! carry the equation's `out` (or substitute for it) and a substitute computation belongs to the
//! same layer. An equation with no source - emitted outside a [`poot_graph_ir::Builder::layer_scope`], or
//! built by a fixture - is untagged, so a layer-cut split either sees exactly the layers the tracer
//! wrote or refuses with a typed error. It never guesses a cut, and no pass silently moves an
//! equation across a layer.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use poot_graph_ir::Operand;
use poot_graph_ir::error::GraphValidationError;
use poot_graph_ir::graph::ValueId;
use poot_graph_ir::graph::{
    Eqn, Graph, LayerIndex, NoValidations, Slot, Storage, ValidationChannel,
};
use poot_graph_ir::types::TensorType;

/// A stage assignment over the traced equation list: one contiguous, non-empty run of equations per
/// stage, runs in stage order.
///
/// Build it with [`StageAssignment::from_stage_per_equation`] (one index per equation) or
/// [`StageAssignment::from_cuts`] (the first equation of each stage after the first).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StageAssignment {
    stage_of_eqn: Vec<usize>,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-745 (formerly 581a)")
)]
impl StageAssignment {
    /// Assign each equation a stage index. Rejects a sequence that is not `0, 0, .., 1, 1, ..`: a
    /// stage must be one run, runs must be in order and adjacent, and the first equation must be in
    /// stage 0.
    pub(crate) fn from_stage_per_equation(stage_of_eqn: &[usize]) -> Result<Self, StageSplitError> {
        validate_stage_sequence(stage_of_eqn)?;
        Ok(Self {
            stage_of_eqn: stage_of_eqn.to_vec(),
        })
    }

    /// Cut points in the traced equation order: `cuts[i]` is the first equation of stage `i + 1`, so
    /// `eqn_count` equations become `cuts.len() + 1` stages. This is the form a layer-pipeline
    /// placement (`after_layer` cuts) maps to once the tracer reports each layer's first equation.
    pub(crate) fn from_cuts(cuts: &[usize], eqn_count: usize) -> Result<Self, StageSplitError> {
        let mut previous = 0;
        for (index, &cut) in cuts.iter().enumerate() {
            if cut == 0 || cut >= eqn_count {
                return Err(StageSplitError::CutOutOfRange {
                    index,
                    cut,
                    eqn_count,
                });
            }
            if cut <= previous {
                return Err(StageSplitError::CutOutOfOrder {
                    index,
                    cut,
                    previous,
                });
            }
            previous = cut;
        }
        let stage_of_eqn: Vec<usize> = (0..eqn_count)
            .map(|equation| cuts.iter().filter(|&&cut| cut <= equation).count())
            .collect();
        Self::from_stage_per_equation(&stage_of_eqn)
    }

    /// Cut at a placement's layer boundaries: `after_layers` holds one `StageBoundary::after_layer`
    /// per boundary, and each cut lands on the first equation tagged with the next layer.
    ///
    /// This is the entry a caller holding `exact_partition`'s plan uses:
    ///
    /// ```ignore
    /// let assignment = StageAssignment::from_layer_cuts(&graph, &plan.after_layer_cuts())?;
    /// ```
    ///
    /// Layer tags come from [`poot_graph_ir::Builder::layer_scope`]. Untagged equations *inside* the layer
    /// region, layer ids that do not run `0, 0, .., 1, 1, ..`, and a cut with no layer after it are
    /// typed errors; untagged equations before the first layer (preamble) and after the last
    /// (head) are expected and stay with the first and last stage.
    pub(crate) fn from_layer_cuts<V: ValidationChannel>(
        g: &Graph<V>,
        after_layers: &[usize],
    ) -> Result<Self, StageSplitError> {
        let layer_of_eqn: Vec<Option<LayerIndex>> = g.eqns.iter().map(|eqn| eqn.layer).collect();
        let first_tagged = layer_of_eqn.iter().position(Option::is_some);
        let last_tagged = layer_of_eqn.iter().rposition(Option::is_some);

        // One contiguous tagged region, one run per layer id in ascending order starting at 0.
        let mut layers = 0usize;
        if let (Some(first), Some(last)) = (first_tagged, last_tagged) {
            for (offset, tag) in layer_of_eqn[first..=last].iter().enumerate() {
                if tag.is_none() {
                    return Err(StageSplitError::UntaggedLayerEquation {
                        equation: first + offset,
                    });
                }
            }
            let first_layer = layer_of_eqn[first]
                .expect("the first tagged equation is tagged")
                .0;
            if first_layer != 0 {
                return Err(StageSplitError::NonContiguousLayer {
                    equation: first,
                    layer: first_layer,
                    expected: 0,
                });
            }
            let mut current = 0usize;
            for (offset, tag) in layer_of_eqn[first..=last].iter().enumerate() {
                let layer = tag.expect("the tagged region has no hole").0;
                if layer == current {
                    continue;
                }
                if layer != current + 1 {
                    return Err(StageSplitError::NonContiguousLayer {
                        equation: first + offset,
                        layer,
                        expected: if layer < current {
                            current
                        } else {
                            current + 1
                        },
                    });
                }
                current = layer;
            }
            layers = current + 1;
        }

        let mut previous_cut: Option<usize> = None;
        for (index, &cut) in after_layers.iter().enumerate() {
            if cut >= layers.saturating_sub(1) {
                return Err(StageSplitError::LayerCutOutOfRange { cut, layers });
            }
            if let Some(previous) = previous_cut
                && cut <= previous
            {
                return Err(StageSplitError::CutOutOfOrder {
                    index,
                    cut,
                    previous,
                });
            }
            previous_cut = Some(cut);
        }

        let last_stage = after_layers.len();
        let stage_of_eqn: Vec<usize> = layer_of_eqn
            .iter()
            .enumerate()
            .map(|(equation, layer)| match layer {
                // Preamble before the first tagged layer starts stage 0; head after the last one
                // lands in the final stage; a tagged equation sits after every cut below its layer.
                None => {
                    if first_tagged.is_some_and(|first| equation < first) {
                        0
                    } else {
                        last_stage
                    }
                }
                Some(layer) => after_layers.iter().filter(|&&cut| cut < layer.0).count(),
            })
            .collect();
        Self::from_stage_per_equation(&stage_of_eqn)
    }

    /// Number of equations this assignment covers.
    pub(crate) fn equation_count(&self) -> usize {
        self.stage_of_eqn.len()
    }

    /// Number of stages: the highest assigned stage index plus one.
    pub(crate) fn stage_count(&self) -> usize {
        self.stage_of_eqn
            .last()
            .map(|&stage| stage + 1)
            .unwrap_or(0)
    }

    /// The stage that owns equation `equation`. Indexes the equation list, so it panics out of range
    /// the same way `graph.eqns[equation]` does.
    pub(crate) fn stage_of(&self, equation: usize) -> usize {
        self.stage_of_eqn[equation]
    }

    /// One `[start, end)` equation range per stage, in stage order; together the ranges partition
    /// `0..len()`.
    pub(crate) fn stage_ranges(&self) -> Vec<Range<usize>> {
        let mut ranges = Vec::with_capacity(self.stage_count());
        let mut start = 0;
        for equation in 1..self.stage_of_eqn.len() {
            if self.stage_of_eqn[equation] != self.stage_of_eqn[equation - 1] {
                ranges.push(start..equation);
                start = equation;
            }
        }
        ranges.push(start..self.stage_of_eqn.len());
        ranges
    }
}

fn validate_stage_sequence(stage_of_eqn: &[usize]) -> Result<(), StageSplitError> {
    let Some(&first) = stage_of_eqn.first() else {
        return Err(StageSplitError::EmptyAssignment);
    };
    if first != 0 {
        return Err(StageSplitError::NonContiguousStage {
            equation: 0,
            stage: first,
            expected: 0,
        });
    }
    let mut current = 0;
    for (equation, &stage) in stage_of_eqn.iter().enumerate().skip(1) {
        if stage == current {
            continue;
        }
        if stage != current + 1 {
            return Err(StageSplitError::NonContiguousStage {
                equation,
                stage,
                expected: if stage < current {
                    current
                } else {
                    current + 1
                },
            });
        }
        current = stage;
    }
    Ok(())
}

/// One value that crosses a stage boundary, in the order the producing equations appear.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryDescriptor {
    /// The value's id in the split graph and in every stage that defines or reads it.
    pub value: ValueId,
    /// The value's dtype and shape, cloned from the producing graph.
    pub aval: TensorType,
    /// `aval.numel() * aval.dtype.byte_size()`, checked - the transfer size of this value.
    pub byte_len: usize,
    pub producing_stage: usize,
    /// Ascending, deduplicated stages that read the value.
    pub consuming_stages: Vec<usize>,
}

/// One stage of a split: its graph, and the boundary values it produces as explicit outputs.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-745 (formerly 581a)")
)]
#[derive(Clone, Debug)]
pub(crate) struct StageGraph<V: ValidationChannel = NoValidations> {
    #[cfg_attr(
        test,
        expect(
            dead_code,
            reason = "held for POOT-745 (formerly 581a) (its staged compile reads it)"
        )
    )]
    pub stage: usize,
    pub graph: Graph<V>,
    /// The crossing values this stage defines, in producing-equation order. Each is an input of the
    /// stages in [`BoundaryDescriptor::consuming_stages`].
    pub outputs: Vec<ValueId>,
}

/// The result of [`split_stages`]: one graph per stage plus the values that cross between them.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-745 (formerly 581a)")
)]
#[derive(Clone, Debug)]
pub(crate) struct StageSplit<V: ValidationChannel = NoValidations> {
    pub stages: Vec<StageGraph<V>>,
    pub boundaries: Vec<BoundaryDescriptor>,
}

/// Split a traced graph into one graph per stage, following `assignment`.
///
/// Every equation lands in exactly one stage; each crossing value becomes an explicit output of its
/// producing stage and an explicit input of its consumers; carried state stays with one stage. Each
/// returned stage graph passes [`Graph::validate`].
pub(crate) fn split_stages<V: ValidationChannel>(
    g: &Graph<V>,
    assignment: &StageAssignment,
) -> Result<StageSplit<V>, StageSplitError> {
    g.validate()?;
    if assignment.equation_count() != g.eqns.len() {
        return Err(StageSplitError::AssignmentLength {
            assigned: assignment.equation_count(),
            eqn_count: g.eqns.len(),
        });
    }
    let stage_count = assignment.stage_count();
    let ranges = assignment.stage_ranges();

    // Which stage defines each value, and which stages read it. A value with no defining equation is
    // a graph input binder: every stage that reads it binds it itself, so it never crosses.
    let mut producer: Vec<Option<usize>> = vec![None; g.values.len()];
    let mut readers: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); g.values.len()];
    for (equation, eqn) in g.eqns.iter().enumerate() {
        let stage = assignment.stage_of(equation);
        producer[eqn.out] = Some(stage);
        for operand in &eqn.inputs {
            if let Operand::Value(value) = operand {
                readers[*value].insert(stage);
            }
        }
    }

    let mut consuming_stages: BTreeMap<ValueId, BTreeSet<usize>> = BTreeMap::new();
    for (value, stages) in readers.iter().enumerate() {
        let Some(producing) = producer[value] else {
            continue;
        };
        for &stage in stages {
            if stage != producing {
                consuming_stages.entry(value).or_default().insert(stage);
            }
        }
    }

    let pair_stage = state_pair_stages(g, &producer, &readers)?;

    let mut boundaries = Vec::new();
    for eqn in &g.eqns {
        if !consuming_stages.contains_key(&eqn.out) {
            continue;
        }
        let aval = g.aval(eqn.out).clone();
        let byte_len = tensor_byte_len(&aval, eqn.out)?;
        boundaries.push(BoundaryDescriptor {
            value: eqn.out,
            aval,
            byte_len,
            // A boundary value is by construction defined by an equation.
            producing_stage: producer[eqn.out].expect("a boundary value has a producing equation"),
            consuming_stages: consuming_stages[&eqn.out].iter().copied().collect(),
        });
    }

    let mut stage_boundary_inputs = vec![Vec::new(); stage_count];
    for (&value, stages) in &consuming_stages {
        for &stage in stages {
            stage_boundary_inputs[stage].push(value);
        }
    }
    let mut stage_state_pairs = vec![Vec::new(); stage_count];
    for (pair, &owner) in pair_stage.iter().enumerate() {
        stage_state_pairs[owner].push(pair);
    }
    let mut stage_outputs = vec![Vec::new(); stage_count];
    for boundary in &boundaries {
        stage_outputs[boundary.producing_stage].push(boundary.value);
    }

    let mut input_binder = vec![false; g.values.len()];
    for &value in &g.inputs {
        input_binder[value] = true;
    }

    let mut stages = Vec::with_capacity(stage_count);
    for (stage, range) in ranges.iter().enumerate() {
        let final_stage = stage + 1 == stage_count;

        // What this stage binds: the graph inputs its own equations read, the carried state it owns,
        // the values earlier stages hand it, and (on the final stage) the graph output when that is
        // itself a binder.
        let mut bound: BTreeSet<ValueId> = BTreeSet::new();
        for eqn in &g.eqns[range.clone()] {
            for operand in &eqn.inputs {
                if let Operand::Value(value) = operand
                    && input_binder[*value]
                {
                    bound.insert(*value);
                }
            }
        }
        for &pair in &stage_state_pairs[stage] {
            let (state_input, state_output) = g.state[pair];
            bound.insert(state_input);
            if input_binder[state_output] {
                bound.insert(state_output);
            }
        }
        bound.extend(stage_boundary_inputs[stage].iter().copied());
        if final_stage && input_binder[g.output] {
            bound.insert(g.output);
        }

        let mut inputs: Vec<ValueId> = g
            .inputs
            .iter()
            .copied()
            .filter(|value| bound.contains(value))
            .collect();
        inputs.extend(stage_boundary_inputs[stage].iter().copied());

        // A boundary value is an equation result, never an input binder (validation rejects an
        // equation that redefines one), so retagging it as this stage's activation input is safe.
        let mut values = g.values.clone();
        for &value in &stage_boundary_inputs[stage] {
            values[value].storage = Storage::Slot(Slot::Activation);
        }
        let consts: Vec<ValueId> = g
            .consts
            .iter()
            .copied()
            .filter(|value| bound.contains(value))
            .collect();
        let mut slots: Vec<(ValueId, Slot)> = g
            .slots
            .iter()
            .copied()
            .filter(|(value, _)| bound.contains(value))
            .collect();
        slots.extend(
            stage_boundary_inputs[stage]
                .iter()
                .map(|&value| (value, Slot::Activation)),
        );

        let eqns: Vec<Eqn> = g.eqns[range.clone()].to_vec();
        let state: Vec<(ValueId, ValueId)> = stage_state_pairs[stage]
            .iter()
            .map(|&pair| g.state[pair])
            .collect();
        let output = if final_stage {
            g.output
        } else {
            eqns.iter()
                .rev()
                .map(|eqn| eqn.out)
                .find(|value| consuming_stages.contains_key(value))
                .ok_or(StageSplitError::StageWithoutExit { stage })?
        };

        let mut graph = Graph {
            values,
            inputs,
            consts,
            slots,
            eqns,
            output,
            validations: g.validations.clone(),
            state,
        };

        // Witnesses stay with the piece of graph that defines them.
        let mut defined = vec![false; graph.values.len()];
        for &value in &graph.inputs {
            defined[value] = true;
        }
        for eqn in &graph.eqns {
            defined[eqn.out] = true;
        }
        if final_stage && !defined[g.output] {
            return Err(StageSplitError::FinalStageMissingOutput {
                stage,
                output: g.output,
            });
        }
        graph.retain_validations(|declaration| defined[declaration.value]);
        graph
            .validate()
            .map_err(|source| StageSplitError::InvalidStage { stage, source })?;

        stages.push(StageGraph {
            stage,
            graph,
            outputs: std::mem::take(&mut stage_outputs[stage]),
        });
    }

    Ok(StageSplit { stages, boundaries })
}

/// The single stage that owns each carried-state pair, or a typed error when a pair spans stages.
///
/// A pair is owned by the equations that touch it: the producer of `state_out` and every reader of
/// `state_in`/`state_out` must agree. A pair no equation touches is inert and stays with stage 0.
fn state_pair_stages<V: ValidationChannel>(
    g: &Graph<V>,
    producer: &[Option<usize>],
    readers: &[BTreeSet<usize>],
) -> Result<Vec<usize>, StageSplitError> {
    let mut owners = Vec::with_capacity(g.state.len());
    for (pair, &(state_input, state_output)) in g.state.iter().enumerate() {
        let mut stages = BTreeSet::new();
        if let Some(stage) = producer[state_output] {
            stages.insert(stage);
        }
        stages.extend(readers[state_input].iter().copied());
        stages.extend(readers[state_output].iter().copied());
        let mut sorted = stages.iter().copied();
        match (sorted.next(), sorted.next()) {
            (None, _) => owners.push(0),
            (Some(stage), None) => owners.push(stage),
            (Some(first_stage), Some(second_stage)) => {
                return Err(StageSplitError::StatePairSplit {
                    pair,
                    state_input,
                    state_output,
                    first_stage,
                    second_stage,
                });
            }
        }
    }
    Ok(owners)
}

/// Split a traced graph at a placement's layer boundaries.
///
/// `after_layers` is one `StageBoundary::after_layer` per boundary, in order: cut `k` separates the
/// equations tagged with layer `k` from those tagged `k + 1`. The tags come from
/// [`poot_graph_ir::Builder::layer_scope`], which the model tracers wrap around their layer loop, so this
/// call carries no per-family equation index and infers nothing from names. Everything else - the
/// stage graphs, the crossing set, the descriptors - is [`split_stages`].
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-745 (formerly 581a)")
)]
pub(crate) fn split_stages_after_layers<V: ValidationChannel>(
    g: &Graph<V>,
    after_layers: &[usize],
) -> Result<StageSplit<V>, StageSplitError> {
    split_stages(g, &StageAssignment::from_layer_cuts(g, after_layers)?)
}

fn tensor_byte_len(aval: &TensorType, value: ValueId) -> Result<usize, StageSplitError> {
    let numel = aval
        .shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or(StageSplitError::ByteLenOverflow { value })?;
    numel
        .checked_mul(aval.dtype.byte_size())
        .ok_or(StageSplitError::ByteLenOverflow { value })
}

/// Typed failure of [`split_stages`]. Value ids and stage indices are kept as fields so a caller can
/// classify a bad split without parsing diagnostics.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum StageSplitError {
    #[error("input graph is invalid: {0}")]
    InvalidGraph(#[from] GraphValidationError),
    #[error("stage assignment covers {assigned} equations but the graph has {eqn_count}")]
    AssignmentLength { assigned: usize, eqn_count: usize },
    #[error("stage assignment is empty")]
    EmptyAssignment,
    #[error(
        "equation {equation} assigns stage {stage}, which is not one contiguous run per stage in \
         order (expected stage {expected})"
    )]
    NonContiguousStage {
        equation: usize,
        stage: usize,
        expected: usize,
    },
    #[error("cut {index} is {cut}, outside the 1..{eqn_count} equation range")]
    CutOutOfRange {
        index: usize,
        cut: usize,
        eqn_count: usize,
    },
    #[error("cut {index} is {cut}, not after the previous cut {previous}")]
    CutOutOfOrder {
        index: usize,
        cut: usize,
        previous: usize,
    },
    #[error(
        "equation {equation} is untagged between the first and last layer-tagged equation; a \
         layer-cut split will not guess which side it belongs to"
    )]
    UntaggedLayerEquation { equation: usize },
    #[error(
        "equation {equation} is tagged layer {layer}, breaking the one-run-per-layer order the \
         layer cuts map onto (expected layer {expected})"
    )]
    NonContiguousLayer {
        equation: usize,
        layer: usize,
        expected: usize,
    },
    #[error(
        "after_layer cut {cut} has no layer after it: the graph tags {layers} layer(s), so a cut \
         past the last layer would produce an empty stage"
    )]
    LayerCutOutOfRange { cut: usize, layers: usize },
    #[error("byte length of v{value} overflows usize")]
    ByteLenOverflow { value: ValueId },
    #[error(
        "state pair {pair} (v{state_input} -> v{state_output}) spans stages {first_stage} and \
         {second_stage}; carried state must stay with one stage"
    )]
    StatePairSplit {
        pair: usize,
        state_input: ValueId,
        state_output: ValueId,
        first_stage: usize,
        second_stage: usize,
    },
    #[error("stage {stage} produces no boundary value, so it cannot feed a later stage")]
    StageWithoutExit { stage: usize },
    #[error(
        "stage {stage} is the final stage but neither defines nor binds the graph output v{output}"
    )]
    FinalStageMissingOutput { stage: usize, output: ValueId },
    #[error("stage {stage} graph is invalid: {source}")]
    InvalidStage {
        stage: usize,
        #[source]
        source: GraphValidationError,
    },
}
