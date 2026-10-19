//! Chained evaluation of stage graphs on the CPU oracle, for the planner tests that cut a decode graph and
//! compare the chain with the uncut graph (ADR-0101 tier 1): used by poot-graph-plan's own unit tests and
//! by the model-tracing staged tests in poot-models. It sits here, not in poot-graph-plan's `test_support`,
//! because evaluating needs the oracle and poot-graph-plan never depends on it outside its own tests.

use std::collections::HashMap;

use poot_graph_ir::{
    BoundaryId, ComputedConst, Graph, SlotKey, Storage, ValidationId, ValueId, ValueMeta,
};
use poot_tensor::HostTensor;
use poot_test_util::fill;

use crate::{EvalOptions, Value, eval};

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
pub struct Chain {
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
pub fn chain(graph: &Graph, stages: &[&Graph], inputs: &HashMap<ValueId, Value>) -> Chain {
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
        let result = eval(g, &stage_inputs, EvalOptions::default().keep_environment())
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

/// Every input binder of `g` bound: token 5 at position 3, name-seeded weights, and a nonzero KV
/// cache per state pair (a zero cache would let a lost state input cancel).
pub fn bind_decode(g: &Graph) -> HashMap<ValueId, Value> {
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
pub fn bits(value: &Value) -> Vec<u32> {
    value
        .as_host()
        .expect("a dense value")
        .as_f32()
        .expect("an f32 value")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}
