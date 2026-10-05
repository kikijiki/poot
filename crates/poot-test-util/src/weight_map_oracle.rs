//! The CPU oracle for a graph whose weights bind through a [`WeightMap`] (Card 564): a const named by
//! a [`WeightId::const_name`](poot_quant::weights::WeightId::const_name) reads
//! [`WeightMap::materialize`] - the map's own row copies, independent of the executor binder's
//! byte-run arithmetic - and a [`PackedSourceName`] over one reads that source of the materialized
//! payload.

use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, Value, eval, materialize_dense};
use poot_graph_ir::{Graph, PackedSourceName, SlotKey, Storage, ValidationChannel};
use poot_quant::PackedComponentRef;
use poot_quant::weights::{WeightEntry, WeightId, WeightMap, WeightStore};
use poot_tensor::HostTensor;

use crate::StepFixture;

/// Evaluate `g` on the CPU: weights through `map` over `store`, slots by key from `slots`, state
/// inputs from `state` in [`Graph::state`] order. Returns the output and every state output.
/// Panics on a const the map does not name: the oracle has no fallback a binder could hide behind.
pub fn eval_mapped<V: ValidationChannel>(
    g: &Graph<V>,
    store: &WeightStore,
    map: &WeightMap,
    slots: &[(SlotKey, HostTensor)],
    state: &[HostTensor],
) -> (HostTensor, Vec<HostTensor>) {
    let ids: HashMap<String, WeightId> =
        map.iter().map(|(id, _, _)| (id.const_name(), id)).collect();
    let materialize = |id: WeightId| map.materialize(id, store).unwrap();
    assert_eq!(state.len(), g.state.len(), "one value per state input");
    let mut inputs: HashMap<usize, Value> = HashMap::new();
    for (&(input, _), value) in g.state.iter().zip(state) {
        inputs.insert(input, value.clone().into());
    }
    for &id in &g.inputs {
        let meta = g.meta(id);
        let value: Value = match meta.storage {
            Storage::State => continue,
            Storage::Computed(c) => HostTensor::f32(c.shape(), c.values_f32()).into(),
            Storage::Slot(_) => {
                let key = meta.slot_key().expect("a builder slot has a key");
                let (_, tensor) = slots
                    .iter()
                    .find(|(k, _)| k == key)
                    .unwrap_or_else(|| panic!("no value for slot {key}"));
                tensor.clone().into()
            }
            Storage::Const => {
                let name = meta.name.as_deref().expect("a const has a name");
                if let Some(&wid) = ids.get(name) {
                    let WeightEntry::Dense(dense) = materialize(wid) else {
                        panic!("{name} is packed but the graph declares it dense")
                    };
                    let mut one = WeightStore::builder();
                    one.insert("w", WeightEntry::Dense(dense)).unwrap();
                    Value::Host(materialize_dense(&one.build(), "w").unwrap())
                } else {
                    let source = PackedSourceName::parse(name)
                        .unwrap_or_else(|| panic!("the map names no const {name}"));
                    let WeightEntry::Packed(payload) = materialize(ids[source.linear_id()]) else {
                        panic!("{name} names a dense weight")
                    };
                    Value::Packed(PackedComponentRef::new(Arc::clone(&payload), source.role()))
                }
            }
            Storage::Device => unreachable!("a graph input is never a device intermediate"),
        };
        inputs.insert(id, value);
    }
    let evaluation = eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap_or_else(|e| panic!("oracle: {e}"));
    let host = |v: Value| v.into_host().expect("a dense tensor");
    (
        host(evaluation.output),
        evaluation.state.into_iter().map(host).collect(),
    )
}

/// The CPU oracle for the parity and coverage rows, as a per-step callback threading state across calls
/// exactly as `poot_executor_parity::run_parity` steps the device (`poot-executor-parity` never depends on
/// `poot-eval`, so the caller computes each step's expected output). Weights bind from `store` by const name:
/// a const named as a packed source ([`PackedSourceName`]) over a packed entry binds that entry's component
/// of the named role, every other const the store entry as its stored dtype (an F16 or BF16 weight binds as
/// its stored words). This is the one definition every backend's parity tests share.
pub fn oracle_for<'a, V: ValidationChannel>(
    graph: &'a Graph<V>,
    store: &'a WeightStore,
) -> impl FnMut(&[StepFixture]) -> HostTensor + 'a {
    let mut state: HashMap<String, HostTensor> = HashMap::new();
    let packed: HashMap<usize, PackedComponentRef> = graph
        .inputs
        .iter()
        .filter_map(|&id| {
            let source = PackedSourceName::parse(graph.meta(id).name.as_deref()?)?;
            let WeightEntry::Packed(owner) = store.get(source.linear_id())? else {
                return None;
            };
            Some((
                id,
                PackedComponentRef::new(Arc::clone(owner), source.role()),
            ))
        })
        .collect();
    move |step: &[StepFixture]| {
        let mut inputs: HashMap<usize, Value> = HashMap::new();
        for &id in &graph.inputs {
            let meta = graph.meta(id);
            let value = match meta.storage {
                Storage::Const => match packed.get(&id) {
                    Some(component) => Value::Packed(component.clone()),
                    None => Value::from(
                        materialize_dense(store, meta.name.as_deref().unwrap()).unwrap(),
                    ),
                },
                Storage::State => Value::from(
                    state
                        .entry(meta.name.clone().unwrap())
                        .or_insert_with(|| HostTensor::zeros(meta.aval.shape.clone()))
                        .clone(),
                ),
                Storage::Computed(c) => Value::from(HostTensor::f32(c.shape(), c.values_f32())),
                Storage::Slot(_) => {
                    let key = meta.slot_key().unwrap();
                    let v = step.iter().find(|v| &v.key == key).unwrap();
                    Value::from(v.tensor.clone())
                }
                Storage::Device => unreachable!("a graph input is never a device intermediate"),
            };
            inputs.insert(id, value);
        }
        let evaluation = eval(graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
        for (&(si, _), t) in graph.state.iter().zip(evaluation.state) {
            state.insert(
                graph.meta(si).name.clone().unwrap(),
                t.into_host().expect("dense state tensor"),
            );
        }
        evaluation.output.into_host().expect("dense output")
    }
}
