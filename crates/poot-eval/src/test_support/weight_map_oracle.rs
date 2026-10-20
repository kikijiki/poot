//! The CPU oracle for a graph whose weights bind through a [`WeightMap`] (Card 564): a const named by
//! a [`WeightId::const_name`](poot_weights::WeightId::const_name) reads
//! [`WeightMap::materialize`] - the map's own row copies, independent of the executor binder's
//! byte-run arithmetic - and a [`PackedSourceName`] over one reads that source of the materialized
//! payload.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{EvalOptions, Value, eval, materialize_dense};
use poot_graph_ir::{Graph, PackedSourceName, SlotKey, Storage, ValueId};
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_quant::{PackedComponentRef, PackedPayload, PackedStack, PackedStackComponentRef};
use poot_tensor::{DType, HostTensor};
use poot_weights::{WeightId, WeightMap, WeightView};

use super::StepFixture;

/// Evaluate `g` on the CPU: weights through `map` over `store`, slots by key from `slots`, state
/// inputs from `state` in [`Graph::state`] order. Returns the output and every state output.
/// Panics on a const the map does not name: the oracle has no fallback a binder could hide behind.
pub fn eval_mapped(
    g: &Graph,
    store: &WeightStore,
    map: &WeightMap,
    slots: &[(SlotKey, HostTensor)],
    state: &[HostTensor],
) -> (HostTensor, Vec<HostTensor>) {
    let ids: HashMap<String, WeightId> =
        map.iter().map(|(id, _, _)| (id.const_name(), id)).collect();
    let materialize = |id: WeightId| map.materialize(id, store).unwrap();
    // One stack per packed expert-stack id, shared by every source role that names it.
    let mut stacks: HashMap<WeightId, Arc<PackedStack>> = HashMap::new();
    assert_eq!(state.len(), g.state().count(), "one value per state input");
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for ((input, _), value) in g.state().zip(state) {
        inputs.insert(input, value.clone().into());
    }
    for &id in g.inputs() {
        let meta = g.meta(id);
        let value: Value = match meta.storage {
            Storage::State => continue,
            Storage::Computed(c) => c.tensor().into(),
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
                    if dense.dtype() == DType::I32 {
                        // An index table (a hash router's): its words as stored.
                        let words = dense
                            .bytes()
                            .as_slice()
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|w| i32::from_le_bytes(*w))
                            .collect();
                        HostTensor::i32(dense.shape().to_vec(), words).into()
                    } else {
                        let mut one = WeightStore::builder();
                        one.insert("w", WeightEntry::Dense(dense)).unwrap();
                        Value::Host(materialize_dense(&one.build(), "w").unwrap())
                    }
                } else {
                    let source = PackedSourceName::parse(name)
                        .unwrap_or_else(|| panic!("the map names no const {name}"));
                    let wid = ids[source.linear_id()];
                    if matches!(map.view(wid), Some(WeightView::ExpertStack(_))) {
                        let stack = stacks.entry(wid).or_insert_with(|| {
                            Arc::new(map.materialize_stack(wid, store).unwrap())
                        });
                        Value::PackedStack(PackedStackComponentRef::new(
                            Arc::clone(stack),
                            source.role(),
                        ))
                    } else {
                        let WeightEntry::Packed(payload) = materialize(wid) else {
                            panic!("{name} names a dense weight")
                        };
                        Value::Packed(PackedComponentRef::new(Arc::clone(&payload), source.role()))
                    }
                }
            }
            Storage::Device => unreachable!("a graph input is never a device intermediate"),
        };
        inputs.insert(id, value);
    }
    let evaluation =
        eval(g, &inputs, EvalOptions::default()).unwrap_or_else(|e| panic!("oracle: {e}"));
    let host = |v: Value| v.into_host().expect("a dense tensor");
    (
        host(evaluation.output),
        evaluation.state.into_iter().map(host).collect(),
    )
}

/// The CPU oracle for the parity and coverage rows, as a per-step callback threading state across calls
/// exactly as `poot_executor_parity::run_parity` steps the device (`poot-executor-parity` never depends on
/// `poot-eval`, so the caller computes each step's expected output). The weight-map and pipeline fixtures it
/// serves are `poot_models::test_support`'s. Weights bind from `store` by const name:
/// a const named as a packed source ([`PackedSourceName`]) over a packed entry binds that entry's component
/// of the named role, every other const the store entry as its stored dtype (an F16 or BF16 weight binds as
/// its stored words). This is the one definition every backend's parity tests share.
pub fn oracle_for<'a>(
    graph: &'a Graph,
    store: &'a WeightStore,
) -> impl FnMut(&[StepFixture]) -> HostTensor + 'a {
    let mut state: HashMap<String, HostTensor> = HashMap::new();
    let mut stacks: HashMap<String, Arc<PackedStack>> = HashMap::new();
    let packed: HashMap<ValueId, Value> = graph
        .inputs()
        .iter()
        .filter_map(|&id| {
            let source = PackedSourceName::parse(graph.meta(id).name.as_deref()?)?;
            if let Some(WeightEntry::Packed(owner)) = store.get(source.linear_id()) {
                return Some((
                    id,
                    Value::Packed(PackedComponentRef::new(Arc::clone(owner), source.role())),
                ));
            }
            // An expert stack: the store holds its experts as `{id}.0`, `{id}.1`, ..
            let owners = expert_owners(store, source.linear_id())?;
            let stack = stacks
                .entry(source.linear_id().to_string())
                .or_insert_with(|| Arc::new(PackedStack::try_new(owners).unwrap()));
            Some((
                id,
                Value::PackedStack(PackedStackComponentRef::new(
                    Arc::clone(stack),
                    source.role(),
                )),
            ))
        })
        .collect();
    move |step: &[StepFixture]| {
        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        for &id in graph.inputs() {
            let meta = graph.meta(id);
            let value = match meta.storage {
                Storage::Const => match packed.get(&id) {
                    Some(component) => component.clone(),
                    None => Value::from(
                        materialize_dense(store, meta.name.as_deref().unwrap()).unwrap(),
                    ),
                },
                Storage::State => Value::from(
                    state
                        .entry(meta.name.clone().unwrap())
                        .or_insert_with(|| zero_state(&meta.aval))
                        .clone(),
                ),
                Storage::Computed(c) => Value::from(c.tensor()),
                Storage::Slot(_) => {
                    let key = meta.slot_key().unwrap();
                    let v = step.iter().find(|v| &v.key == key).unwrap();
                    Value::from(v.tensor.clone())
                }
                Storage::Device => unreachable!("a graph input is never a device intermediate"),
            };
            inputs.insert(id, value);
        }
        let evaluation = eval(graph, &inputs, EvalOptions::default()).unwrap();
        for ((si, _), t) in graph.state().zip(evaluation.state) {
            state.insert(
                graph.meta(si).name.clone().unwrap(),
                t.into_host().expect("dense state tensor"),
            );
        }
        evaluation.output.into_host().expect("dense output")
    }
}

/// A carried state's value before the first step: zeros of the state's own dtype.
fn zero_state(aval: &poot_graph_ir::TensorType) -> HostTensor {
    let shape = aval.shape.clone();
    match aval.dtype {
        DType::F32 => HostTensor::zeros(shape),
        DType::F16 => {
            let count = shape.iter().product();
            HostTensor::f16(shape, vec![0u16; count])
        }
        other => panic!("the oracle has no zero state for {other:?}"),
    }
}

/// The packed experts a store holds for stack `stack_id`: the entries `{stack_id}.0`, `{stack_id}.1`, ..
/// (the key spelling a GGUF load gives a rank-3 expert tensor), or `None` when it holds no `{stack_id}.0`.
pub fn expert_owners(store: &WeightStore, stack_id: &str) -> Option<Vec<Arc<PackedPayload>>> {
    let mut owners = Vec::new();
    while let Some(WeightEntry::Packed(owner)) = store.get(&format!("{stack_id}.{}", owners.len()))
    {
        owners.push(Arc::clone(owner));
    }
    (!owners.is_empty()).then_some(owners)
}
