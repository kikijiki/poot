//! Test-only reconstructions of `poot-graph-ir`'s `Builder` convenience methods card 671 deleted
//! (`finish_with_validations`, `finish_with_state_and_validations`: dead in production, moved to
//! `poot_test_util::graph_fixtures`), shared by this crate's own `#[cfg(test)]` modules.
//!
//! Duplicated here rather than imported from `poot-test-util`: that crate's `graph-fixtures`
//! feature regular-depends on `poot-graph-plan` (for its own graph fixtures), so this crate's own
//! unit tests dev-depending on it back would compile two mismatched instances of `poot-graph-ir`'s
//! `Builder`/`Traced`/`Graph` (the same reasoning as `poot-graph-ir`'s and `poot-eval`'s
//! equivalents - see their commit messages). Every function here is built only from
//! `poot-graph-ir`'s already-public `Builder`/`Graph` methods, exactly as
//! `poot_test_util::graph_fixtures` builds its own copy.

use poot_graph_ir::{Builder, Graph, GraphValidationError, Traced, ValidationId};

pub(crate) fn finish_with_validations(
    b: Builder,
    out: Traced,
    validations: &[(ValidationId, &str, Traced)],
) -> Result<Graph, GraphValidationError> {
    finish_with_state_and_validations(b, out, &[], validations)
}

pub(crate) fn finish_with_state_and_validations(
    b: Builder,
    out: Traced,
    state: &[(Traced, Traced)],
    validations: &[(ValidationId, &str, Traced)],
) -> Result<Graph, GraphValidationError> {
    let outputs = validations.iter().fold(
        poot_graph_ir::Outputs::new(out).with_state(state),
        |outputs, &(id, name, value)| outputs.with_witness(id, name, value),
    );
    b.finish(outputs).map_err(|error| match error {
        poot_graph_ir::EditError::Invalid(error) => error,
        other => panic!("a fresh builder has no baseline to break: {other}"),
    })
}

// ---- stage splits ---------------------------------------------------------------------------

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{BoundaryId, ComputedConst, SlotKey, Storage, ValueId, ValueMeta};

use crate::passes::StageSplit;
use crate::{CutSite, StageAssignment, StageSplitError, split_stages};

/// Cut `g` before each listed equation.
pub(crate) fn assignment_before(
    g: &Graph,
    cuts: &[usize],
) -> Result<StageAssignment, StageSplitError> {
    let sites: Vec<CutSite> = cuts.iter().map(|&cut| CutSite::Before(cut)).collect();
    StageAssignment::from_sites(g, &sites)
}

/// Cut `g` after each listed layer.
pub(crate) fn assignment_after_layers(
    g: &Graph,
    layers: &[usize],
) -> Result<StageAssignment, StageSplitError> {
    let sites: Vec<CutSite> = layers
        .iter()
        .map(|&layer| CutSite::AfterLayer(layer))
        .collect();
    StageAssignment::from_sites(g, &sites)
}

/// Split `g` before each listed equation.
pub(crate) fn split_before(g: &Graph, cuts: &[usize]) -> Result<StageSplit, StageSplitError> {
    split_stages(g, &assignment_before(g, cuts)?)
}

/// Split `g` after each listed layer.
pub(crate) fn split_after_layers(
    g: &Graph,
    layers: &[usize],
) -> Result<StageSplit, StageSplitError> {
    split_stages(g, &assignment_after_layers(g, layers)?)
}

/// What names an input binder across a split: stage graphs renumber their values, so a binder is
/// identified by what the uncut graph declared, never by its id.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Identity {
    Named(String),
    Slot(SlotKey),
    Computed(ComputedConst),
}

fn identity(meta: &ValueMeta) -> Identity {
    match meta.storage {
        Storage::Const | Storage::State => {
            Identity::Named(meta.name.clone().expect("a const or state binder is named"))
        }
        Storage::Slot(_) => Identity::Slot(
            meta.slot_key()
                .cloned()
                .expect("a slot binder built through the Builder is keyed"),
        ),
        Storage::Computed(payload) => Identity::Computed(payload),
        Storage::Device => unreachable!("a device value is never an input binder"),
    }
}

/// The result of evaluating stage graphs one after another on the CPU oracle.
pub(crate) struct Chain {
    /// The final stage's output.
    pub output: Value,
    /// Carried state per uncut pair, in the uncut graph's pair order.
    pub state: Vec<Value>,
    /// Every boundary value as its producer computed it.
    pub boundaries: HashMap<BoundaryId, Value>,
    /// Every validation witness by id, as the stage that defines it computed it.
    pub witnesses: HashMap<ValidationId, Value>,
}

/// Evaluate `stages` one after another: each stage is bound to the uncut `inputs` it reads (matched by
/// declared identity) and to the boundary values earlier stages exported (matched by
/// [`BoundaryId::slot_key`]), and every export is read back from the stage's environment.
pub(crate) fn chain(graph: &Graph, stages: &[&Graph], inputs: &HashMap<ValueId, Value>) -> Chain {
    let mut by_identity: HashMap<Identity, Value> = graph
        .inputs()
        .iter()
        .filter_map(|&id| {
            inputs
                .get(&id)
                .map(|value| (identity(graph.meta(id)), value.clone()))
        })
        .collect();
    let mut boundaries = HashMap::new();
    let mut witnesses = HashMap::new();
    let mut state_by_name: HashMap<String, Value> = HashMap::new();
    let mut output = None;
    for (index, &g) in stages.iter().enumerate() {
        let stage_inputs: HashMap<ValueId, Value> = g
            .inputs()
            .iter()
            .map(|&id| {
                // A computed constant is its own payload; a compile pass may have introduced
                // one the uncut graph never declared.
                if let Storage::Computed(payload) = g.meta(id).storage {
                    return (id, Value::from(payload.tensor()));
                }
                let identity = identity(g.meta(id));
                let value = by_identity
                    .get(&identity)
                    .unwrap_or_else(|| panic!("stage {index}: {identity:?} is unbound"));
                (id, value.clone())
            })
            .collect();
        let result = eval(
            g,
            &stage_inputs,
            EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
        )
        .unwrap_or_else(|error| panic!("stage {index} did not evaluate: {error}"));
        let environment = result.environment.expect("the environment was kept");
        for export in g.exports() {
            let value = environment[export.value]
                .clone()
                .unwrap_or_else(|| panic!("stage {index} lacks boundary {:?}", export.id));
            by_identity.insert(Identity::Slot(export.id.slot_key()), value.clone());
            boundaries.insert(export.id, value);
        }
        for declaration in g.witnesses() {
            let value = environment[declaration.value]
                .clone()
                .unwrap_or_else(|| panic!("stage {index} lacks witness {:?}", declaration.id));
            witnesses.insert(declaration.id, value);
        }
        for ((state_input, _), value) in g.state().zip(result.state) {
            let name = g.meta(state_input).name.clone().expect("state is named");
            state_by_name.insert(name, value);
        }
        if index + 1 == stages.len() {
            output = Some(result.output);
        }
    }
    let state = graph
        .state()
        .map(|(state_input, _)| {
            let name = graph
                .meta(state_input)
                .name
                .as_ref()
                .expect("state is named");
            state_by_name
                .remove(name)
                .unwrap_or_else(|| panic!("no stage owns carried state {name}"))
        })
        .collect();
    Chain {
        output: output.expect("the final stage ran"),
        state,
        boundaries,
        witnesses,
    }
}

// ---- the qwen2 decode fixture ---------------------------------------------------------------

use poot_executor_parity::dense::{Dense, Family, step};
use poot_models::model::{LogitRows, Phase};
use poot_tensor::HostTensor;

/// Deterministic pseudo-random fill in [-1, 1) (xorshift, no rng dependency).
pub(crate) fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Layers of [`qwen2_decode`]'s default fixture.
pub(crate) const LAYERS: usize = 3;

/// The tiny qwen2 decode graph (GQA 4 over 2) of `layers` layers over a six-position cache.
pub(crate) fn qwen2_decode(layers: usize) -> Graph {
    qwen2_decode_rows(layers, 1)
}

/// [`qwen2_decode`] stepping `rows` sequences at once.
pub(crate) fn qwen2_decode_rows(layers: usize, rows: usize) -> Graph {
    let model = Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, layers)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(32)
        .f32_model();
    model
        .model
        .trace(Phase::Decode, step(rows, 1, 6, LogitRows::Last))
        .expect("the qwen2 fixture traces a decode step")
}

/// Every input binder of `g` bound: token 5 at position 3, name-seeded weights, and a nonzero KV
/// cache per state pair (a zero cache would let a lost state input cancel).
pub(crate) fn bind_decode(g: &Graph) -> HashMap<ValueId, Value> {
    let mut inputs = HashMap::new();
    for &id in g.inputs() {
        let meta = g.meta(id);
        let shape = meta.aval.shape.clone();
        let numel = shape.iter().product::<usize>().max(1);
        let tensor = match meta.storage {
            Storage::Slot(poot_graph_ir::Slot::Token) => HostTensor::i32(shape, vec![5; numel]),
            Storage::Slot(poot_graph_ir::Slot::Pos) => HostTensor::i32(shape, vec![3; numel]),
            // Adapter 1 on the first row, no adapter on the rest.
            Storage::Slot(poot_graph_ir::Slot::LoraIdx) => {
                HostTensor::i32(shape, (0..numel).map(|row| i32::from(row == 0)).collect())
            }
            Storage::Slot(poot_graph_ir::Slot::Activation) => {
                HostTensor::f32(shape, fill(numel, 31).iter().map(|v| v * 0.5).collect())
            }
            Storage::Slot(other) => panic!("unexpected slot {other:?}"),
            Storage::Const => {
                let name = meta.name.as_deref().expect("a constant is named");
                let seed = name.bytes().fold(1469598103934665603u64, |h, c| {
                    (h ^ u64::from(c)).wrapping_mul(1099511628211)
                });
                let data = fill(numel, seed).iter().map(|v| v * 0.1).collect();
                HostTensor::f32(shape, data)
            }
            Storage::Computed(computed) => computed.tensor(),
            Storage::State => continue,
            Storage::Device => unreachable!("an input binder is never a device value"),
        };
        inputs.insert(id, Value::from(tensor));
    }
    for (pair, (state_input, _)) in g.state().enumerate() {
        let shape = g.aval(state_input).shape.clone();
        let numel = shape.iter().product::<usize>().max(1);
        inputs.insert(
            state_input,
            Value::from(HostTensor::f32(shape, fill(numel, 7000 + pair as u64))),
        );
    }
    inputs
}

/// An f32 value's bit pattern, for exact comparison.
pub(crate) fn bits(value: &Value) -> Vec<u32> {
    value
        .as_host()
        .expect("a dense value")
        .as_f32()
        .expect("an f32 value")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

// A GPU target fixture's backend and capabilities, for this crate's own tests: they know which arm they
// built. Duplicated for the same reason as the builders above (`poot_test_util::graph_fixtures` has its own
// copy over the other `poot-graph-plan` instance). Production code matches `Target` or asks `Target::gpu`.

pub(crate) fn gpu_caps(target: &crate::Target) -> poot_target::DeviceCaps {
    match target {
        crate::Target::Gpu { caps, .. } => *caps,
        crate::Target::Aie { .. } => {
            panic!("{target:?} is an AIE target: it has no GPU DeviceCaps")
        }
    }
}

pub(crate) fn gpu_caps_mut(target: &mut crate::Target) -> &mut poot_target::DeviceCaps {
    match target {
        crate::Target::Gpu { caps, .. } => caps,
        crate::Target::Aie { .. } => panic!("an AIE target has no GPU DeviceCaps"),
    }
}
