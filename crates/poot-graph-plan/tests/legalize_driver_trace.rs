//! POOT-1017: `legalize` on the graph the DRIVER traces (`Model::trace` over a registry-built qwen2),
//! driven through `compile`. That tracer reads every dense weight as `Transpose(const [out, in])`
//! (`components::linear`: `x @ w^T` over the checkpoint's own layout), the lm_head included, so an
//! oversized weight is never a bare `MatMul` operand: `legalize` must split the weight behind the
//! transpose, or `compile` refuses the head (`OversizedConstant`) under a `max_buffer_bytes` below it.
//!
//! The reference is `weight_map_oracle::eval_mapped` over the unsplit trace; the compiled program is
//! evaluated on the same weights, each `<name>.chunkN` const bound as the matching row block of `<name>`
//! and the hosted `Slot::TokenEmbed` as the token's embedding row.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use poot_eval::{EvalBudget, EvalOptions, Value, eval, materialize_dense};
use poot_graph_ir::{Slot, SlotKey, Storage};
use poot_graph_plan::{
    CompileError, CompileLimits, CompileOptions, FusionPolicy, LegalizeError, Submission, Target,
    compile,
};
use poot_models::model::{KvLayout, LogitRows, Phase, StepShape};
use poot_models::registry::{RawConfig, Registry};
use poot_quant::weights::{WeightEntry, WeightMap, WeightStore};
use poot_target::{Backend, DeviceCaps};
use poot_tensor::{DType, HostTensor};
use poot_test_util::weight_map_oracle::eval_mapped;

const CAPACITY: usize = 8;
const TOKEN: i32 = 11;
const POS: i32 = 3;

/// Below the fixture's lm_head (48 x 64 BF16 = 6144 bytes) and its embed table, and also below its
/// 8192-byte q/o projections and 12288-byte FFN weights: every one of those splits into row chunks.
const LIMIT: u64 = 4096;

fn caps(max_buffer_bytes: u64) -> DeviceCaps {
    DeviceCaps {
        max_buffer_bytes,
        ..DeviceCaps::wgpu_rdna3_igpu()
    }
}

fn options() -> CompileOptions {
    CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: CompileLimits::STANDARD,
    }
}

struct Driver {
    graph: poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs>,
    store: WeightStore,
    map: WeightMap,
}

/// The registry's qwen2 fixture, traced as one decode step exactly as the driver traces it.
fn driver_trace() -> Driver {
    let registry = Registry::builtin().unwrap();
    let fixture = (registry.entries()[0].fixture)();
    let raw = RawConfig::HfJson {
        config: &fixture.config,
        generation: fixture.generation.as_ref(),
    };
    let model = registry.build(&raw, &fixture.store).unwrap();
    let shape = StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::MIN,
        capacity: NonZeroUsize::new(CAPACITY).unwrap(),
        kv: KvLayout::Contiguous,
        logits: LogitRows::Last,
    };
    Driver {
        graph: model.trace(Phase::Decode, shape).unwrap(),
        store: fixture.store,
        map: model.weights().clone(),
    }
}

fn slot_values() -> Vec<(SlotKey, HostTensor)> {
    vec![
        (
            SlotKey::new(Slot::Token, None),
            HostTensor::i32(vec![1, 1], vec![TOKEN]),
        ),
        (
            SlotKey::new(Slot::Pos, None),
            HostTensor::i32(vec![1, 1], vec![POS]),
        ),
    ]
}

fn dense_weight(map: &WeightMap, store: &WeightStore, name: &str) -> HostTensor {
    let (id, _, _) = map
        .iter()
        .find(|(id, _, _)| id.const_name() == name)
        .unwrap_or_else(|| panic!("the map names no weight {name}"));
    let WeightEntry::Dense(dense) = map.materialize(id, store).unwrap() else {
        panic!("{name} is packed")
    };
    let mut one = WeightStore::builder();
    one.insert("w", WeightEntry::Dense(dense)).unwrap();
    materialize_dense(&one.build(), "w").unwrap()
}

fn rows_of(t: &HostTensor, from: usize, rows: usize) -> HostTensor {
    let width = t.shape()[1];
    let range = from * width..(from + rows) * width;
    let shape = vec![rows, width];
    match t.dtype() {
        DType::BF16 => HostTensor::bf16(shape, t.as_half().unwrap()[range].to_vec()),
        DType::F32 => HostTensor::f32(shape, t.as_f32().unwrap()[range].to_vec()),
        other => panic!("unexpected weight dtype {other:?}"),
    }
}

fn argmax(logits: &HostTensor) -> usize {
    logits
        .to_f32()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

/// SC-001: a driver compile under a `max_buffer_bytes` below the head succeeds with the unsplit
/// trace's logits and greedy token, and no constant of the program exceeds the limit.
///
/// Mutation (recorded): making `transposed_consts` return an empty map leaves the transposed head
/// unsplit, so the `compile` below refuses it.
#[test]
fn driver_compile_splits_transposed_weights_under_a_small_buffer_limit() {
    let d = driver_trace();
    let state: Vec<HostTensor> = d
        .graph
        .state
        .iter()
        .map(|&(si, _)| HostTensor::zeros(d.graph.aval(si).shape.clone()))
        .collect();
    let (dense, _) = eval_mapped(&d.graph, &d.store, &d.map, &slot_values(), &state);

    let target = Target {
        backend: Backend::SpirvVulkan,
        caps: caps(LIMIT),
    };
    let program = compile(&d.graph, &target, &options())
        .expect("every oversized weight splits and the embed hosts");
    let g = program.graph();

    // per split weight: (chunks bound so far, rows bound so far)
    let mut next: HashMap<String, (usize, usize)> = HashMap::new();
    let mut inputs: HashMap<usize, Value> = HashMap::new();
    for &(si, _) in &g.state {
        inputs.insert(si, Value::from(HostTensor::zeros(g.aval(si).shape.clone())));
    }
    for &id in &g.inputs {
        let meta = g.meta(id);
        let name = meta.name.as_deref();
        let value = match meta.storage {
            Storage::State => continue,
            Storage::Computed(c) => HostTensor::f32(c.shape(), c.values_f32()),
            Storage::Slot(Slot::TokenEmbed) => {
                let embed = dense_weight(&d.map, &d.store, name.expect("a hosted embed is named"));
                let row = rows_of(&embed, TOKEN as usize, 1);
                row.reshaped(meta.aval.shape.clone()).unwrap()
            }
            Storage::Slot(_) => {
                let key = meta.slot_key().unwrap();
                slot_values().into_iter().find(|(k, _)| k == key).unwrap().1
            }
            Storage::Const => {
                let name = name.unwrap();
                match name.split_once(".chunk") {
                    Some((base, idx)) => {
                        let (chunks, at) = next.entry(base.to_string()).or_insert((0, 0));
                        assert_eq!(idx.parse::<usize>().unwrap(), *chunks, "chunks in order");
                        *chunks += 1;
                        let rows = meta.aval.shape[0];
                        let chunk = rows_of(&dense_weight(&d.map, &d.store, base), *at, rows);
                        *at += rows;
                        chunk
                    }
                    None => dense_weight(&d.map, &d.store, name),
                }
            }
            Storage::Device => unreachable!("a program input is never a device intermediate"),
        };
        inputs.insert(id, Value::from(value));
    }
    for &id in &g.consts {
        let bytes = g.aval(id).numel() as u64 * g.aval(id).dtype.byte_size() as u64;
        assert!(
            bytes <= LIMIT,
            "{:?} is {bytes} bytes, over the limit",
            g.meta(id).name
        );
    }
    let (head, &(_, rows)) = next
        .iter()
        .find(|(n, _)| n.contains("head") || n.contains("lm"))
        .expect("the lm_head is among the split weights");
    assert_eq!(
        rows,
        dense_weight(&d.map, &d.store, head).shape()[0],
        "the head's chunks partition its rows once"
    );

    let split = eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    poot_test_util::assert_close_rel(&split.to_f32().unwrap(), &dense.to_f32().unwrap(), 1e-5);
    assert_eq!(argmax(&split), argmax(&dense), "identical greedy token");
}

/// A limit under one row of the head (hidden 64 x 2 bytes = 128) cannot be met by any row split: the
/// head stays whole and `compile` refuses it by name, before any device allocation.
#[test]
fn driver_compile_refuses_a_head_no_row_split_can_fit() {
    let d = driver_trace();
    let target = Target {
        backend: Backend::SpirvVulkan,
        caps: caps(100),
    };
    let err = compile(&d.graph, &target, &options()).expect_err("one head row is over 100 bytes");
    let CompileError::Legalize(refusal) = err else {
        panic!("expected a legalize refusal, got {err}")
    };
    assert!(
        matches!(
            *refusal,
            LegalizeError::OversizedConstant { limit: 100, .. }
        ),
        "got {refusal:?}"
    );
}
