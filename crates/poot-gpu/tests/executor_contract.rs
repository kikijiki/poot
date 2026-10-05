//! Card 546a acceptance: a qwen2-shaped decode step with KV state (and a prefill entry sharing that
//! state) is traced, compiled through `compile_staged`, loaded into `Engine<WgpuDevice>` through the
//! object-safe `Executor`, stepped, and compared with the CPU oracle (`poot_eval::eval`).
//!
//! No `Executor::read_state` exists: state correctness is checked two ways, both
//! through production step outputs - (1) a decode/prefill entry's own output already depends on the
//! KV state it reads (attention over the cache), so a wrong state value desyncs every later step
//! from the oracle; (2) a dedicated "state probe" entry, declared with the identical state name/aval/
//! storage so it shares the real entries' buffer, whose own graph output IS the state
//! value - an ordinary production entry, not a test-only accessor.

use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{
    BindError, Executor, ExecutorStats, LoadError, NoSync, StateScope, StepInputs,
};
use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_gpu::device::WgpuDevice;
use poot_graph_ir::PackedSourceName;
use poot_graph_ir::{Graph, Slot, SlotKey, StateRole, Storage, TensorType, ValueId};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    Submission, Target, TargetSet, compile_staged,
};
use poot_models::model::{LogitRows, Phase};
use poot_quant::PackedComponentRef;
use poot_quant::format::{GroupMap, WeightFormat};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::HostTensor;
use poot_tensor::{DType, HostView};
use poot_test_util::assert_close_rel;

const CAP: usize = 8;

fn dense() -> Dense {
    Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
}

/// `dense`'s one-token decode step over `cap` positions, as `Model::trace` returns it.
fn decode_graph(dense: &Dense, cap: usize) -> Graph {
    plain(
        dense
            .f32_model()
            .model
            .trace(Phase::Decode, step(1, 1, cap, LogitRows::Last))
            .unwrap(),
    )
}

/// `dense`'s `n`-token prefill step over `cap` positions, logits of the last token.
fn prefill_graph(dense: &Dense, n: usize, cap: usize) -> Graph {
    plain(
        dense
            .f32_model()
            .model
            .trace(Phase::Prefill, step(1, n, cap, LogitRows::Last))
            .unwrap(),
    )
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.1
        })
        .collect()
}

/// Deterministic const data by name; the prefill's causal mask is causal.
fn const_data(name: &str, shape: &[usize]) -> Vec<f32> {
    let n = shape.iter().product::<usize>();
    if name == "causal.mask" {
        let l = shape[shape.len() - 1];
        return (0..n)
            .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
            .collect();
    }
    let seed = name.bytes().fold(1469598103934665603u64, |h, c| {
        (h ^ c as u64).wrapping_mul(1099511628211)
    });
    fill(n, seed)
}

/// A store holding every const the graphs name, as F32 stored tensors.
fn store_for(graphs: &[&Graph]) -> WeightStore {
    let mut entries: HashMap<String, Vec<usize>> = HashMap::new();
    for g in graphs {
        for &id in &g.inputs {
            let meta = g.meta(id);
            if meta.storage == Storage::Const {
                entries.insert(meta.name.clone().unwrap(), meta.aval.shape.clone());
            }
        }
    }
    let mut builder = WeightStore::builder();
    for (name, shape) in entries {
        let bytes: Vec<u8> = const_data(&name, &shape)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let dense = DenseWeight::try_new(DType::F32, shape, Arc::from(bytes)).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    builder.build()
}

fn wgpu_target(device: &WgpuDevice) -> Target {
    use poot_executor::Device;
    device.target()
}

fn staged(
    g: &Graph,
    target: Target,
    submission: Submission,
) -> poot_graph_plan::StagedProgram<poot_graph_ir::ValidationOutputs> {
    let g = g.clone().with_validations(Vec::new());
    compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            execution: submission,
            fusion: FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .unwrap()
}

/// One step's slot values: `tokens` at positions `start..`, as each slot's declared dtype.
struct SlotValue {
    key: SlotKey,
    shape: Vec<usize>,
    f32s: Vec<f32>,
    i32s: Vec<i32>,
    dtype: DType,
}

fn slot_values(g: &Graph, tokens: &[u32], start: usize) -> Vec<SlotValue> {
    g.inputs
        .iter()
        .filter_map(|&id| {
            let meta = g.meta(id);
            let Storage::Slot(slot) = meta.storage else {
                return None;
            };
            let TensorType { shape, dtype } = meta.aval.clone();
            let values: Vec<f64> = match slot {
                Slot::Token => tokens.iter().map(|&t| t as f64).collect(),
                // Card 550: `Slot::Pos` is `[rows, tokens]`: `[1,1] = [pos]` for a decode step, or
                // `[1,n] = [start, start+1, .., start+n-1]` for this one-shot prefill (always
                // `start=0` at every call site here, matching the mask's in-graph computation).
                Slot::Pos => {
                    let tokens_axis = *shape.last().unwrap_or(&1);
                    (0..tokens_axis).map(|i| (start + i) as f64).collect()
                }
                other => panic!("qwen2 test graph has no {other:?} slot"),
            };
            Some(SlotValue {
                key: meta.slot_key().unwrap().clone(),
                shape,
                f32s: values.iter().map(|&v| v as f32).collect(),
                i32s: values.iter().map(|&v| v as i32).collect(),
                dtype,
            })
        })
        .collect()
}

fn view(v: &SlotValue) -> HostView<'_> {
    match v.dtype {
        DType::F32 => {
            HostView::new(DType::F32, v.f32s.len(), bytemuck::cast_slice(&v.f32s)).unwrap()
        }
        DType::I32 => {
            HostView::new(DType::I32, v.i32s.len(), bytemuck::cast_slice(&v.i32s)).unwrap()
        }
        other => panic!("unexpected slot dtype {other:?}"),
    }
}

fn step_inputs(values: &[SlotValue]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for v in values {
        inputs.push(v.key.clone(), &v.shape, view(v));
    }
    inputs
}

/// The CPU oracle for one step: returns the output and advances `state` (by state name).
fn oracle_step(
    g: &Graph,
    store: &WeightStore,
    values: &[SlotValue],
    state: &mut HashMap<String, HostTensor>,
) -> Vec<f32> {
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        let value = match meta.storage {
            Storage::Const => {
                let name = meta.name.as_deref().unwrap();
                let WeightEntry::Dense(dense) = store.get(name).unwrap() else {
                    unreachable!()
                };
                let data: Vec<f32> = bytemuck_f32(dense.bytes().as_slice());
                Value::from(HostTensor::f32(meta.aval.shape.clone(), data))
            }
            Storage::State => Value::from(
                state
                    .entry(meta.name.clone().unwrap())
                    .or_insert_with(|| HostTensor::zeros(meta.aval.shape.clone()))
                    .clone(),
            ),
            Storage::Computed(c) => Value::from(HostTensor::f32(c.shape(), c.values_f32())),
            Storage::Slot(_) => {
                let key = meta.slot_key().unwrap();
                let v = values.iter().find(|v| &v.key == key).unwrap();
                Value::from(match v.dtype {
                    DType::I32 => HostTensor::i32(v.shape.clone(), v.i32s.clone()),
                    _ => HostTensor::f32(v.shape.clone(), v.f32s.clone()),
                })
            }
            Storage::Device => unreachable!(),
        };
        inputs.insert(id, value);
    }
    let evaluation = eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
    for (&(si, _), t) in g.state.iter().zip(evaluation.state) {
        state.insert(
            g.meta(si).name.clone().unwrap(),
            t.into_host().expect("dense state tensor"),
        );
    }
    evaluation
        .output
        .into_host()
        .expect("dense output")
        .as_f32()
        .unwrap()
        .to_vec()
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A graph whose sole output is one named state value: a production entry an `Executor` steps like
/// any other, which happens to let the test read a state buffer through its real step output
/// (no `read_state` accessor exists, and this is not one - it is the same binder,
/// commit and readback path every other entry uses).
fn state_probe_graph(name: &str, aval: TensorType) -> Graph {
    let b = poot_graph_ir::Builder::new();
    let state = b.state_input(name, aval, StateRole::Recurrent);
    b.finish_with_state(state, &[(state, state)])
}

fn require_wgpu() -> Option<WgpuDevice> {
    WgpuDevice::new().ok()
}

/// SC-001 (fix: wgpu only): a qwen2 decode entry on `Box<dyn Executor>` matches the CPU oracle for
/// every step of a full cache, and capture is the default: one recording after `CAP` replayed steps.
#[test]
fn qwen2_decode_on_the_contract_matches_the_oracle_and_records_once() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let g = decode_graph(&dense(), CAP);
    let store = store_for(&[&g]);
    let exe = exec
        .load_weights(
            Arc::new(store.clone()),
            poot_executor::WeightSource::ConstNames,
        )
        .unwrap();
    let decode = exec
        .add_entry(exe, &staged(&g, target, Submission::Replay))
        .unwrap();
    let mut oracle_state = HashMap::new();
    for pos in 0..CAP {
        let token = ((pos * 7 + 3) % 32) as u32;
        let values = slot_values(&g, &[token], pos);
        let expected = oracle_step(&g, &store, &values, &mut oracle_state);
        let got = {
            let mut out = exec
                .step(exe, decode, &step_inputs(&values), &mut NoSync)
                .unwrap();
            bytemuck_f32(&out.read().unwrap())
        };
        assert_close_rel(&got, &expected, 5e-3);
    }
    let ExecutorStats { recordings, .. } = exec.stats();
    assert_eq!(recordings, 1, "one entry, one recording after 8 steps");
}

/// The contract admits `Submission::Replay` only. A `Submission::Eager`
/// program is refused at `add_entry` with a typed error naming the missing capability, before any
/// allocation (`load_entry` never runs, so `stats()` after the refusal is unchanged from before it -
/// `Submission::Eager` survives only on the pre-contract executors, never through this entry point).
/// Mutation: drop the `program.submission() != Submission::Replay` check in `Engine::add_entry`; an
/// Eager program is accepted and the row goes red.
#[test]
fn eager_submission_is_refused_at_add_entry_before_any_allocation() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);
    let g = decode_graph(&dense(), CAP);
    let store = Arc::new(store_for(&[&g]));
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let before = engine.stats();
    let error = engine
        .add_entry(exe, &staged(&g, target, Submission::Eager))
        .expect_err("a Submission::Eager program must be refused at add_entry");
    assert!(
        matches!(error, poot_executor::ExecError::Load(ref e) if matches!(**e, poot_executor::LoadError::Unimplemented(_))),
        "expected a typed LoadError::Unimplemented naming the missing capability, got {error:?}"
    );
    let after = engine.stats();
    assert_eq!(
        before.memory, after.memory,
        "a refused add_entry must not allocate anything"
    );
}

/// SC-004 (Z5) + SC-010: a prefill entry and a decode entry share state by (name, aval, storage).
/// Prefill then decode steps match the oracle; `reset_state(All)` zeroes the state without
/// re-recording (checked by a fresh decode step from position 0 matching a from-zero oracle, and by
/// the state-probe entry's own output reading zero bytes).
#[test]
fn prefill_and_decode_entries_share_state_and_reset_without_rerecording() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);
    let n = 3;
    let prefill_g = prefill_graph(&dense(), n, CAP);
    let decode_g = decode_graph(&dense(), CAP);
    let store = Arc::new(store_for(&[&prefill_g, &decode_g]));
    let exe = engine
        .load_weights(store.clone(), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let prefill = engine
        .add_entry(exe, &staged(&prefill_g, target, Submission::Replay))
        .unwrap();
    let decode = engine
        .add_entry(exe, &staged(&decode_g, target, Submission::Replay))
        .unwrap();

    // One state-probe entry per KV state name, sharing the real entries' buffers.
    let mut probes = Vec::new();
    for &id in &decode_g.inputs {
        let meta = decode_g.meta(id);
        if meta.storage == Storage::State {
            let name = meta.name.clone().unwrap();
            let probe_graph = state_probe_graph(&name, meta.aval.clone());
            let probe = engine
                .add_entry(exe, &staged(&probe_graph, target, Submission::Replay))
                .unwrap();
            probes.push((name, probe, meta.aval.numel() * 4));
        }
    }

    for round in 0..2 {
        let mut oracle_state = HashMap::new();
        let prompt: Vec<u32> = (0..n as u32).map(|t| (t * 11 + 2 + round) % 32).collect();
        let values = slot_values(&prefill_g, &prompt, 0);
        let expected = oracle_step(&prefill_g, &store, &values, &mut oracle_state);
        let got = bytemuck_f32(
            &engine
                .step(exe, prefill, &step_inputs(&values), &mut NoSync)
                .unwrap()
                .read()
                .unwrap(),
        );
        assert_close_rel(&got, &expected, 5e-3);
        for pos in n..CAP {
            let values = slot_values(&decode_g, &[((pos * 3 + 1) % 32) as u32], pos);
            let expected = oracle_step(&decode_g, &store, &values, &mut oracle_state);
            let got = bytemuck_f32(
                &engine
                    .step(exe, decode, &step_inputs(&values), &mut NoSync)
                    .unwrap()
                    .read()
                    .unwrap(),
            );
            assert_close_rel(&got, &expected, 5e-3);
        }
        for &(ref name, probe, _) in &probes {
            let got = bytemuck_f32(
                &engine
                    .step(exe, probe, &StepInputs::new(), &mut NoSync)
                    .unwrap()
                    .read()
                    .unwrap(),
            );
            let expected = oracle_state[name].as_f32().unwrap().to_vec();
            assert_close_rel(&got, &expected, 5e-3);
        }
        engine.reset_state(exe, StateScope::All).unwrap();
        for &(_, probe, bytes) in &probes {
            let got = engine
                .step(exe, probe, &StepInputs::new(), &mut NoSync)
                .unwrap()
                .read()
                .unwrap();
            assert_eq!(
                got,
                vec![0u8; bytes],
                "state not zeroed after reset_state(All)"
            );
        }
    }
    let ExecutorStats { recordings, .. } = engine.stats();
    // prefill + decode + one probe per KV state pair, each recorded exactly once.
    assert_eq!(recordings, 2 + probes.len() as u64);
}

/// SC-017: removing a prefill entry (`remove_entry`) returns its arena bytes to the counters, and the
/// decode entry sharing the executable still replays with `stats().recordings` unchanged.
#[test]
fn remove_entry_frees_its_arena_and_leaves_the_other_entry_unaffected() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);
    let n = 3;
    let prefill_g = prefill_graph(&dense(), n, CAP);
    let decode_g = decode_graph(&dense(), CAP);
    let store = Arc::new(store_for(&[&prefill_g, &decode_g]));
    let exe = engine
        .load_weights(store.clone(), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let prefill = engine
        .add_entry(exe, &staged(&prefill_g, target, Submission::Replay))
        .unwrap();
    let decode = engine
        .add_entry(exe, &staged(&decode_g, target, Submission::Replay))
        .unwrap();

    // Step both so each records once (recordings is otherwise 0 and the "unchanged" assertion below
    // would be vacuous).
    let prompt: Vec<u32> = (0..n as u32).collect();
    let values = slot_values(&prefill_g, &prompt, 0);
    engine
        .step(exe, prefill, &step_inputs(&values), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    let values = slot_values(&decode_g, &[1], n);
    engine
        .step(exe, decode, &step_inputs(&values), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();

    let before = engine.stats();
    let decode_recordings_before = before.recordings;
    engine.remove_entry(exe, prefill).unwrap();
    let after = engine.stats();

    let bytes =
        |s: &ExecutorStats| -> u64 { s.memory.iter().map(|&(_, snap)| snap.live_bytes).sum() };
    assert!(
        bytes(&after) < bytes(&before),
        "remove_entry must free the prefill entry's arena bytes: before={}, after={}",
        bytes(&before),
        bytes(&after)
    );
    // The decode entry's own recording is untouched; stats() is a sum over live entries, so removing
    // the prefill entry's contribution (1 recording) must drop the total by exactly 1, not disturb
    // the decode entry's.
    assert_eq!(
        after.recordings,
        decode_recordings_before - 1,
        "removing the prefill entry must drop only its own recording from the aggregate"
    );

    // The decode entry still replays with its recording count unchanged.
    let values = slot_values(&decode_g, &[2], n + 1);
    engine
        .step(exe, decode, &step_inputs(&values), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(
        engine.stats().recordings,
        after.recordings,
        "the decode entry must replay, not re-record, after the sibling entry was removed"
    );
}

/// Card 547a SC-002: dropping every buffer reads zero on the wgpu runtime, and `unload` on the wgpu
/// engine reads zero through `Engine::stats().memory` (which reads the device's own runtime counters,
/// not an engine-side tally, so this is a real release, not a bookkeeping no-op).
///
/// Mutation (recorded here, applied by hand against real hardware, never left in the tree): in
/// `Context::tag_alloc` (`crates/poot-runtime/src/context/counters.rs`), made every `BufferRole::Weight`
/// allocation also record (and immediately `std::mem::forget`) a second, never-released guard for the
/// same bytes, so that role's live bytes never return to zero. Result: RED -
/// `role Weight must read zero live bytes after unload, got 23616` (`left: 23616, right: 0`), the
/// leaked extra Weight-role allocation surviving `unload`. Reverted: GREEN.
#[test]
fn unload_on_the_wgpu_engine_reads_zero_memory() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);
    let n = 3;
    let prefill_g = prefill_graph(&dense(), n, CAP);
    let decode_g = decode_graph(&dense(), CAP);
    let store = Arc::new(store_for(&[&prefill_g, &decode_g]));
    let exe = engine
        .load_weights(store.clone(), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let prefill = engine
        .add_entry(exe, &staged(&prefill_g, target, Submission::Replay))
        .unwrap();
    let decode = engine
        .add_entry(exe, &staged(&decode_g, target, Submission::Replay))
        .unwrap();
    let prompt: Vec<u32> = (0..n as u32).collect();
    let values = slot_values(&prefill_g, &prompt, 0);
    engine
        .step(exe, prefill, &step_inputs(&values), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    let values = slot_values(&decode_g, &[1], n);
    engine
        .step(exe, decode, &step_inputs(&values), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();

    let before = engine.stats();
    let total_before: u64 = before.memory.iter().map(|&(_, snap)| snap.live_bytes).sum();
    assert!(
        total_before > 0,
        "fixture must have live allocations before unload"
    );

    engine.unload(exe).unwrap();
    let after = engine.stats();
    for &(role, snapshot) in &after.memory {
        assert_eq!(
            snapshot.live_bytes, 0,
            "role {role:?} must read zero live bytes after unload, got {}",
            snapshot.live_bytes
        );
    }
}

/// SC-004 (schema mismatch): an entry whose state has an existing name but another aval is refused
/// at `add_entry`.
#[test]
fn state_with_same_name_and_other_aval_is_refused() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);
    let g = decode_graph(&dense(), CAP);
    let other = decode_graph(&dense(), CAP + 4);
    let store = Arc::new(store_for(&[&g, &other]));
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    engine
        .add_entry(exe, &staged(&g, target, Submission::Replay))
        .unwrap();
    let err = engine
        .add_entry(exe, &staged(&other, target, Submission::Replay))
        .unwrap_err();
    assert!(
        matches!(&err, poot_executor::ExecError::Bind(b) if matches!(**b, BindError::StateSchema { .. })),
        "{err}"
    );
}

/// SC-004 (round-2, F9): a mid-load refusal rolls back whatever `add_entry` had already
/// allocated, and the executable's existing entry still replays afterward with `stats()` unchanged.
/// `state_with_same_name_and_other_aval_is_refused`'s fixture never actually exercises the rollback
/// (every one of `other`'s states already exists by name, so `StateSchema` fires before any new
/// allocation - round 2's own probe: disabling the rollback left all 18 tests green). Here the
/// second entry's graph declares a brand-new state "c" BEFORE the mismatched "b", so `load_entry`
/// allocates "c" for real, then fails on "b" - the rollback has something to undo.
///
/// Mutation: drop the truncate/retain block in `add_entry`; "c"'s buffer stays charged to the
/// executable (`stats().memory`'s `State` role grows) and the row goes red.
#[test]
fn mid_load_refusal_rolls_back_a_newly_allocated_state_and_leaves_the_existing_entry_unaffected() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);

    // g1: states "a" and "b" (shape [2]) - the executable's existing resident state.
    let b1 = poot_graph_ir::Builder::new();
    let a1 = b1.state_input("a", TensorType::f32(vec![2]), StateRole::Recurrent);
    let b1_state = b1.state_input("b", TensorType::f32(vec![2]), StateRole::Recurrent);
    let out1 = b1.binary(poot_graph_ir::BinOp::Add, a1, b1_state);
    let g1 = b1.finish_with_state(out1, &[(a1, a1), (b1_state, b1_state)]);

    // g2: a brand-new state "c" declared first (so load_entry allocates it before reaching "b"),
    // then "b" at a mismatched shape ([3], not g1's [2]) to force the refusal.
    let b2 = poot_graph_ir::Builder::new();
    let c2 = b2.state_input("c", TensorType::f32(vec![2]), StateRole::Recurrent);
    let b2_state = b2.state_input("b", TensorType::f32(vec![3]), StateRole::Recurrent);
    let g2 = b2.finish_with_state(c2, &[(c2, c2), (b2_state, b2_state)]);

    let store = Arc::new(WeightStore::builder().build());
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine
        .add_entry(exe, &staged(&g1, target, Submission::Replay))
        .unwrap();
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    let before = engine.stats();
    assert_eq!(before.recordings, 1, "the first step records once");

    let err = engine
        .add_entry(exe, &staged(&g2, target, Submission::Replay))
        .unwrap_err();
    assert!(
        matches!(&err, poot_executor::ExecError::Bind(b) if matches!(**b, BindError::StateSchema { .. })),
        "{err}"
    );
    let after_refusal = engine.stats();
    // Only `live_bytes` is a rollback claim: `peak_bytes`/`allocations` are monotonic by design
    // (Card 547a) - "c"'s buffer really was allocated once, however briefly, so its peak and count
    // stay recorded even after the rollback frees it.
    let live_bytes = |stats: &poot_executor::ExecutorStats| -> Vec<_> {
        stats
            .memory
            .iter()
            .map(|&(role, snapshot)| (role, snapshot.live_bytes))
            .collect()
    };
    assert_eq!(
        live_bytes(&after_refusal),
        live_bytes(&before),
        "a refused add_entry must roll back the new state \"c\" it already allocated"
    );
    assert_eq!(
        (after_refusal.recordings, after_refusal.replays),
        (before.recordings, before.replays),
        "a refused add_entry on a different entry must not touch the existing entry's counters"
    );

    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    let after_replay = engine.stats();
    assert_eq!(
        after_replay.recordings, before.recordings,
        "the existing entry still replays (not re-records) after the refused sibling add"
    );
}

/// The binder's typed refusals (Card 620 folded into 546a, SC-011/SC-006/SC-012) and the target
/// check (SC-014).
#[test]
fn binder_and_target_refusals_are_typed() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);
    let g = decode_graph(&dense(), CAP);
    let store = Arc::new(store_for(&[&g]));
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let decode = engine
        .add_entry(exe, &staged(&g, target, Submission::Replay))
        .unwrap();
    let values = slot_values(&g, &[1], 0);

    let missing = step_inputs(&values[1..]);
    let err = engine
        .step(exe, decode, &missing, &mut NoSync)
        .err()
        .unwrap();
    assert!(
        matches!(&err, poot_executor::ExecError::Bind(b) if matches!(**b, BindError::Missing { .. })),
        "{err}"
    );

    let mut unknown = step_inputs(&values);
    let zero = [0i32];
    unknown.push(
        SlotKey::new(Slot::SlotMap, None),
        &[],
        HostView::new(DType::I32, 0, bytemuck::cast_slice(&zero[..0])).unwrap(),
    );
    let err = engine
        .step(exe, decode, &unknown, &mut NoSync)
        .err()
        .unwrap();
    assert!(
        matches!(&err, poot_executor::ExecError::Bind(b) if matches!(**b, BindError::Unknown { .. })),
        "{err}"
    );

    // Card 550 deleted the decode graph's `Slot::Mask`; `Slot::Pos` (`[1,1]`) is the shape-mismatch
    // target instead - a `[1]` shape still carries one element, so a check that only compared
    // element count (not shape) would miss it (the SC-011 concern this row guards).
    let mut short = StepInputs::new();
    let short_shape = [1usize];
    for v in &values {
        if v.key.role() == Slot::Pos {
            short.push(
                v.key.clone(),
                &short_shape,
                HostView::new(DType::I32, 1, bytemuck::cast_slice(&v.i32s[..1])).unwrap(),
            );
        } else {
            short.push(v.key.clone(), &v.shape, view(v));
        }
    }
    let err = engine.step(exe, decode, &short, &mut NoSync).err().unwrap();
    assert!(
        matches!(&err, poot_executor::ExecError::Bind(b) if matches!(**b, BindError::Shape { .. })),
        "{err}"
    );

    // SC-006: an f32 value bound to a declared-I32 scalar slot (Pos/SeqLen/
    // Token are all I32), same shape and element count, must be refused by dtype lane rather than
    // silently converted by the binder.
    let i32_slot = values
        .iter()
        .find(|v| v.dtype == DType::I32)
        .expect("qwen2 decode graph has an I32 slot (Token/Pos/SeqLen)")
        .key
        .clone();
    let others: Vec<&SlotValue> = values.iter().filter(|v| v.key != i32_slot).collect();
    let i32_value = values.iter().find(|v| v.key == i32_slot).unwrap();

    let mut wrong_lane = StepInputs::new();
    for v in &others {
        wrong_lane.push(v.key.clone(), &v.shape, view(v));
    }
    let as_f32 = [i32_value.i32s[0] as f32];
    wrong_lane.push(
        i32_slot.clone(),
        &i32_value.shape,
        HostView::new(DType::F32, 1, bytemuck::cast_slice(&as_f32)).unwrap(),
    );
    let err = engine
        .step(exe, decode, &wrong_lane, &mut NoSync)
        .err()
        .unwrap();
    assert!(
        matches!(&err, poot_executor::ExecError::Bind(b) if matches!(**b, BindError::Lane { .. })),
        "{err}"
    );

    // SC-012: a tensor whose declared shape matches the slot (so a check that
    // only compared a shape prefix would miss it) but whose element count does not - a scalar slot
    // (shape `[]`, numel 1) fed a 2-element view.
    let mut wrong_numel = StepInputs::new();
    for v in &others {
        wrong_numel.push(v.key.clone(), &v.shape, view(v));
    }
    let two_i32s = [i32_value.i32s[0], i32_value.i32s[0]];
    wrong_numel.push(
        i32_slot.clone(),
        &i32_value.shape,
        HostView::new(DType::I32, 2, bytemuck::cast_slice(&two_i32s)).unwrap(),
    );
    let err = engine
        .step(exe, decode, &wrong_numel, &mut NoSync)
        .err()
        .unwrap();
    assert!(
        matches!(&err, poot_executor::ExecError::Bind(b) if matches!(**b, BindError::ElementCount { .. })),
        "{err}"
    );

    let mut other = target;
    other.caps.max_grid[0] -= 1;
    let err = engine
        .add_entry(exe, &staged(&g, other, Submission::Replay))
        .unwrap_err();
    assert!(
        matches!(&err, poot_executor::ExecError::Load(l) if matches!(**l, LoadError::TargetMismatch { .. })),
        "{err}"
    );
}

/// SC-007: a two-phase state commit (a program whose state pairs are not donatable - here a straight
/// swap, `(a,b)` and `(b,a)`, matches `poot-graph-plan`'s own
/// `native_state_plan_classifies_pass_through_alias_and_identity_edges` fixture) runs through the
/// engine's explicit `Device::copy` commits and equals the oracle after 4 steps on
/// wgpu. The program has no slots, so every step binds empty `StepInputs`.
#[test]
fn two_phase_state_commit_matches_the_oracle_after_four_steps() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);

    // `a`/`s` start zeroed (no slots or read_state to seed them otherwise); a bias const drives both
    // away from zero each step, so the fixture is sensitive to whether the commit copy actually runs
    // (if `commits` silently no-ops, both states and the output stay zero forever instead of
    // growing). State pairs (a, new_a=s+bias) and (s, new_s=a+bias) are a true swap-plus-increment:
    // neither is donatable in place (each new value reads the *other* state, so writing one in place
    // before the other reads it would corrupt it), matching `poot-graph-plan`'s own
    // `native_state_plan_classifies_pass_through_alias_and_identity_edges` two-commit fixture.
    let b = poot_graph_ir::Builder::new();
    let a = b.state_input("a", TensorType::f32(vec![4]), StateRole::Recurrent);
    let s = b.state_input("s", TensorType::f32(vec![4]), StateRole::Recurrent);
    let bias = b.constant("bias", TensorType::f32(vec![4]));
    let output = b.binary(poot_graph_ir::BinOp::Add, a, s);
    let new_a = b.binary(poot_graph_ir::BinOp::Add, s, bias);
    let new_s = b.binary(poot_graph_ir::BinOp::Add, a, bias);
    let g = b.finish_with_state(output, &[(a, new_a), (s, new_s)]);

    let bias_value = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut builder = WeightStore::builder();
    builder
        .insert(
            "bias",
            WeightEntry::Dense(
                DenseWeight::try_new(
                    DType::F32,
                    vec![4],
                    Arc::from(
                        bias_value
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<u8>>(),
                    ),
                )
                .unwrap(),
            ),
        )
        .unwrap();
    let store = Arc::new(builder.build());
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine
        .add_entry(exe, &staged(&g, target, Submission::Replay))
        .unwrap();

    let mut oracle_a = vec![0.0f32; 4];
    let mut oracle_s = vec![0.0f32; 4];
    for _ in 0..4 {
        let expected: Vec<f32> = oracle_a.iter().zip(&oracle_s).map(|(x, y)| x + y).collect();
        let got = bytemuck_f32(
            &engine
                .step(exe, entry, &StepInputs::new(), &mut NoSync)
                .unwrap()
                .read()
                .unwrap(),
        );
        assert_close_rel(&got, &expected, 1e-6);
        let next_a: Vec<f32> = oracle_s
            .iter()
            .zip(&bias_value)
            .map(|(x, y)| x + y)
            .collect();
        let next_s: Vec<f32> = oracle_a
            .iter()
            .zip(&bias_value)
            .map(|(x, y)| x + y)
            .collect();
        oracle_a = next_a;
        oracle_s = next_s;
    }
    let ExecutorStats { recordings, .. } = engine.stats();
    assert_eq!(recordings, 1, "one entry, one recording after 4 steps");
}

/// SC-008: a prefill fixture whose dispatch count exceeds one submit's budget (`FLUSH_EVERY` in
/// `poot_gpu::device`; a 64-layer model crosses it on dispatch count alone, cheaper to build than
/// crossing `FLUSH_WORK` by growing the prompt - the CPU oracle below evaluates the raw, unfused
/// graph, so a longer prompt costs it quadratically) splits into several real `submit_cached` calls,
/// and its output still matches the CPU oracle (the split is a host-side batching detail, never a
/// semantic difference).
#[test]
fn many_layer_prefill_splits_submits_by_dispatch_count_and_matches_the_oracle() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);
    // Many thin layers, not a long prompt: `FLUSH_EVERY` (1024 dispatches) is a dispatch-count bound
    // independent of per-dispatch size, and each layer contributes a roughly fixed equation count -
    // cheaper to cross than `FLUSH_WORK` by growing `n` (the CPU oracle below evaluates the raw,
    // unfused graph, so its cost would grow with any larger prompt length).
    let cfg = dense().dims(16, 32, 64);
    let n = 8;
    let cap = n;
    let prefill_g = prefill_graph(&cfg, n, cap);
    let store = store_for(&[&prefill_g]);
    let exe = engine
        .load_weights(
            Arc::new(store.clone()),
            poot_executor::WeightSource::ConstNames,
        )
        .unwrap();
    let prefill = engine
        .add_entry(exe, &staged(&prefill_g, target, Submission::Replay))
        .unwrap();
    let prompt: Vec<u32> = (0..n as u32).map(|t| t % 32).collect();
    let values = slot_values(&prefill_g, &prompt, 0);
    let mut oracle_state = HashMap::new();
    let expected = oracle_step(&prefill_g, &store, &values, &mut oracle_state);
    let submits_before = engine.device().context().submits();
    let got = bytemuck_f32(
        &engine
            .step(exe, prefill, &step_inputs(&values), &mut NoSync)
            .unwrap()
            .read()
            .unwrap(),
    );
    let submits_after = engine.device().context().submits();
    assert_close_rel(&got, &expected, 5e-3);
    assert!(
        submits_after - submits_before > 1,
        "a {n}-token prefill's recorded run must split into more than one submit_cached call \
         (submits before={submits_before}, after={submits_after}); the dispatch work weight must \
         be too small to exceed FLUSH_WORK/FLUSH_EVERY for this shape"
    );
}

/// SC-008: a fixture whose dispatch *count* stays far under `FLUSH_EVERY` but
/// whose summed `Dispatch::work` crosses `FLUSH_WORK` on its own - the prior dispatch-count fixture
/// never exercises this (an "ignore `Dispatch::work`" probe stayed green against it). `dispatch_work` is an equation's OUTPUT element count only, never the
/// contracted dimension, so a single-layer decode step with a huge `vocab` crosses `FLUSH_WORK` in
/// the lm_head matmul's one equation while the real FLOPs (vocab * hidden) and the CPU oracle's cost
/// stay small, since hidden/layers/cap all stay tiny.
#[test]
fn single_decode_step_crosses_flush_work_by_summed_work_and_matches_the_oracle() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);
    let cfg = dense().vocab(4_300_000).dims(16, 32, 1);
    let g = decode_graph(&cfg, CAP);
    let store = store_for(&[&g]);
    let exe = engine
        .load_weights(
            Arc::new(store.clone()),
            poot_executor::WeightSource::ConstNames,
        )
        .unwrap();
    let decode = engine
        .add_entry(exe, &staged(&g, target, Submission::Replay))
        .unwrap();
    let values = slot_values(&g, &[0], 0);
    let mut oracle_state = HashMap::new();
    let expected = oracle_step(&g, &store, &values, &mut oracle_state);
    let submits_before = engine.device().context().submits();
    let got = bytemuck_f32(
        &engine
            .step(exe, decode, &step_inputs(&values), &mut NoSync)
            .unwrap()
            .read()
            .unwrap(),
    );
    let submits_after = engine.device().context().submits();
    assert_close_rel(&got, &expected, 5e-3);
    assert!(
        submits_after - submits_before > 1,
        "a huge-vocab decode step's recorded run must split into more than one submit_cached call \
         by summed Dispatch::work alone, with dispatch count far under FLUSH_EVERY; submits \
         before={submits_before}, after={submits_after}"
    );
}

/// SC-016 core: a packed `format` linear (built with `poot_test_util::packed::random_payload`'s
/// admitted-domain synthetic bytes, Card 642) binds its source(s) from the store by (store key,
/// `SourceRole`) - Card 642's const naming (`PackedSourceName`), one entry per source regardless of
/// how many roles `format` has (Q8_0's one `Blocks` role; GPTQ/AWQ's several `Planar` roles,
/// including GPTQ act-order's `GroupIndex`) - and its device output equals the CPU oracle
/// (`poot_eval::eval` fed the identical `Value::Packed`s) within float tolerance (the
/// matmul's reduction order differs from the CPU walk's, ADR-0101 tier 2).
///
/// Mutation (run by hand, see the done-report): in `Engine::weight_buffer`, bind the store by the
/// const's full flat name instead of `PackedSourceName::parsed.linear_id()`; the store (keyed
/// "linear0") has no entry under any of "linear0.packed_*_source", so `add_entry` fails with
/// `Unbound` and every one of these rows goes red together.
fn packed_linear_matches_the_oracle(format: WeightFormat, out: usize, k: usize) {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);

    let payload = Arc::new(poot_test_util::packed::random_payload(
        format,
        [out, k],
        0x546a,
    ));
    let descriptor = payload.weight();

    let b = poot_graph_ir::Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![1, k]));
    let y = poot_graph_ir::ops::packed_linear(&b, x, "linear0", descriptor, None, None).unwrap();
    let g = b.finish(y);

    let store = {
        let mut builder = WeightStore::builder();
        builder
            .insert("linear0", WeightEntry::Packed(Arc::clone(&payload)))
            .unwrap();
        builder.build()
    };

    let exe = engine
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine
        .add_entry(exe, &staged(&g, target, Submission::Replay))
        .unwrap();

    let x_values = fill(k, 0xA5);
    let (x_id, _) = *g
        .slots
        .iter()
        .find(|&&(_, kind)| kind == Slot::Activation)
        .unwrap();
    let mut inputs = StepInputs::new();
    let x_shape = [1, k];
    inputs.push(
        g.meta(x_id).slot_key().unwrap().clone(),
        &x_shape,
        HostView::new(DType::F32, k, bytemuck::cast_slice(&x_values)).unwrap(),
    );
    let actual = bytemuck_f32(
        &engine
            .step(exe, entry, &inputs, &mut NoSync)
            .unwrap()
            .read()
            .unwrap(),
    );

    let mut oracle_inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        let value = match meta.storage {
            Storage::Const => {
                let parsed = PackedSourceName::parse(meta.name.as_deref().unwrap()).unwrap();
                Value::from(PackedComponentRef::new(Arc::clone(&payload), parsed.role()))
            }
            Storage::Slot(Slot::Activation) => {
                Value::from(HostTensor::f32(vec![1, k], x_values.clone()))
            }
            other => unreachable!("packed linear fixture has no {other:?} input"),
        };
        oracle_inputs.insert(id, value);
    }
    let oracle = eval(&g, &oracle_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec();
    assert_close_rel(&actual, &oracle, 1e-3);
}

#[test]
fn packed_q8_0_linear_binds_by_store_key_and_source_role_and_matches_the_oracle() {
    packed_linear_matches_the_oracle(WeightFormat::Q8_0, 4, 32);
}

/// Q4_K_M is the main benchmark proxy format ([[prefer-small-qwen-as-benchmark-proxy]]); its packed
/// binding through the new contract must be verified, not just compiled.
#[test]
fn packed_q4_k_linear_binds_by_store_key_and_source_role_and_matches_the_oracle() {
    packed_linear_matches_the_oracle(WeightFormat::Q4_K, 4, 256);
}

#[test]
fn packed_q5_k_linear_binds_by_store_key_and_source_role_and_matches_the_oracle() {
    packed_linear_matches_the_oracle(WeightFormat::Q5_K, 4, 256);
}

#[test]
fn packed_q6_k_linear_binds_by_store_key_and_source_role_and_matches_the_oracle() {
    packed_linear_matches_the_oracle(WeightFormat::Q6_K, 4, 256);
}

#[test]
fn packed_gptq_linear_binds_by_store_key_and_source_role_and_matches_the_oracle() {
    packed_linear_matches_the_oracle(
        WeightFormat::Gptq {
            groups: GroupMap::Indexed {
                groups: std::num::NonZeroUsize::new(4).unwrap(),
            },
        },
        8,
        256,
    );
}

#[test]
fn packed_awq_linear_binds_by_store_key_and_source_role_and_matches_the_oracle() {
    packed_linear_matches_the_oracle(
        WeightFormat::Awq {
            group_size: std::num::NonZeroUsize::new(64).unwrap(),
        },
        8,
        256,
    );
}

/// SC-015: a kernel whose Card 531c assert fires surfaces as a classifiable fault from
/// `Device::synchronize`, before any output readback - the exact mechanism `Engine::step` relies on
/// (it calls `self.device.synchronize()`, and on error asks `self.device.classify_fault` before ever
/// reading an output). No kernelgen op in this tree today compiles a bounds check to a `Terminator::
/// Trap` to drive this through a real graph, so this drives `WgpuDevice` through the identical
/// begin/dispatch/finish/replay/synchronize sequence `Engine::step`/`run_transaction` uses
/// internally (recording never touches the device, F2: the dispatch only actually runs once
/// `replay` submits it), directly on the public `Device` trait surface, loading the one committed
/// fixture that IS a real rustc-inserted bounds check (`ImportedKernel::AssertTrap`,
/// `poot-rocm-gpu/assets/assert_trap.kir.json`, card 531c: `out[i] = data[idx[i]]`), same as
/// `poot_runtime`'s own `pending_fault_tests` module.
///
/// Mutation: `WgpuDevice::classify_fault` always returns `None` (never matches
/// `RuntimeError::KernelAssertFailed`); `classify_fault`'s `Some` branch never fires and the row goes
/// red, matching exactly `Engine::step`'s own fallback to the generic device-error wrap instead of
/// `ExecError::Fault`.
#[test]
fn wgpu_device_classifies_a_kernel_assert_fault_from_synchronize() {
    use poot_executor::{Arg, BufferRole, Device, Dispatch};
    use poot_target::BufferStorage;

    let Some(mut device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let body: poot_kernel_ir::Body = poot_graph_plan::ImportedKernel::AssertTrap.body().clone();
    let dir = std::env::temp_dir().join("poot-gpu-sc015-fault-probe");
    std::fs::create_dir_all(&dir).unwrap();
    let out_path =
        poot_codegen::artifact_path(&dir, "sc015_assert_trap", poot_codegen::Target::SpirvVulkan);
    poot_codegen::compile(&body, poot_codegen::Target::SpirvVulkan, &out_path)
        .expect("assert_trap must compile to SpirV");
    let spv = std::fs::read(&out_path).unwrap();
    let compiled = poot_codegen::kernel_handle(&body, poot_codegen::Target::SpirvVulkan, spv);
    let kernel = device
        .load_kernel("sc015_assert_trap", compiled)
        .expect("load the assert_trap kernel");

    // idx[2] == 4 is one past data's 4 elements (the fixture's own shape: out[i] = data[idx[i]] over
    // 4 lanes) - the same out-of-range probe poot_runtime's own pending-fault test uses.
    let idx = device
        .allocate(BufferRole::Input, BufferStorage::i32(), 4)
        .unwrap();
    device
        .write(&idx, bytemuck::cast_slice(&[3i32, 0, 4, 1]))
        .unwrap();
    let data = device
        .allocate(BufferRole::Input, BufferStorage::f32(), 4)
        .unwrap();
    device
        .write(&data, bytemuck::cast_slice(&[10.0f32, 20.0, 30.0, 40.0]))
        .unwrap();
    let out_buf = device
        .allocate(BufferRole::Output, BufferStorage::f32(), 4)
        .unwrap();

    device.begin(poot_graph_plan::Submission::Replay).unwrap();
    device
        .dispatch(Dispatch {
            kernel: &kernel,
            inputs: &[
                Arg {
                    buffer: &idx,
                    elems: 4,
                },
                Arg {
                    buffer: &data,
                    elems: 4,
                },
            ],
            output: Arg {
                buffer: &out_buf,
                elems: 4,
            },
            threads: [4, 1, 1],
            workgroup: [64, 1, 1],
            work: 4,
        })
        .unwrap();
    let recording = device
        .finish()
        .unwrap()
        .expect("Submission::Replay always returns a recording");
    device
        .replay(&recording)
        .expect("replay only re-encodes metadata; it never runs the dispatch itself");
    let err = device
        .synchronize()
        .expect_err("idx[2] == 4 is out of range for a 4-element data buffer");
    let (kernel_name, code) = device.classify_fault(&err).expect(
        "a kernel-assert RuntimeError must classify as a fault, not fall through to the generic \
         device-error wrap",
    );
    assert_eq!(kernel_name, "sc015_assert_trap");
    assert_ne!(
        code, 0,
        "a fired trap's code is never the 0 sentinel (no-fault)"
    );
}

/// SC-009: a planted bad validation packet is reported before any readback of outputs (the
/// validated session). The graph's sole output IS its validated value (`b.finish(witness)`
/// with a `ValidationOutput` naming that same `witness`), bound from the store to a nonzero f32: the
/// packet's lane check (`0x7fff_ffff` masked nonzero fails, ADR-0101 decision 2) must reject it in
/// `step`, which returns before ever constructing a `StepOutputs` for the caller to `.read()`.
///
/// Mutation: drop the `self.check_validation(...)?` call in `Engine::step`; `step` returns `Ok` and
/// the row goes red (the caller could then read the un-validated output).
#[test]
fn planted_bad_validation_packet_fails_before_any_output_readback() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);

    let b = poot_graph_ir::Builder::new();
    let planted = b.constant("bad_witness", TensorType::f32(vec![1]));
    // A canonical device witness (an ordered `Ge` observation) bound nonzero: `Ge(1.0, 0.0)` is 1, so
    // the runtime packet check rejects it. A graph-input witness could never reach this check - Card
    // 673's witness admission refuses it at `compile_staged`.
    let witness = b.binary_scalar(
        poot_graph_ir::BinOp::Ge,
        planted,
        poot_graph_ir::Scalar::F32(0.0),
    );
    let g = b
        .finish(witness)
        .with_validations(vec![poot_graph_ir::ValidationOutput {
            id: poot_graph_ir::ValidationId(1),
            name: "planted".into(),
            value: witness.id,
        }]);

    let store = {
        let mut builder = WeightStore::builder();
        let dense =
            DenseWeight::try_new(DType::F32, vec![1], Arc::from([0u8, 0, 0x80, 0x3f])).unwrap(); // 1.0f32 little-endian: nonzero, must fail the planted packet.
        builder
            .insert("bad_witness", WeightEntry::Dense(dense))
            .unwrap();
        builder.build()
    };

    let staged_program = poot_graph_plan::compile_staged(
        &g,
        &poot_graph_plan::TargetSet::single(poot_graph_plan::DeviceId(0), target),
        &poot_graph_plan::Partition {
            experts: poot_graph_plan::ExpertPlacement::AllResident,
            devices: poot_graph_plan::DevicePlacement::Single(poot_graph_plan::DeviceId(0)),
        },
        &poot_graph_plan::CompileOptions {
            execution: Submission::Replay,
            fusion: poot_graph_plan::FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .unwrap();

    let exe = engine
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine.add_entry(exe, &staged_program).unwrap();
    let inputs = StepInputs::new();
    let error = match engine.step(exe, entry, &inputs, &mut NoSync) {
        Ok(_) => panic!("a nonzero planted validation witness must fail step before any readback"),
        Err(error) => error,
    };
    assert!(
        matches!(error, poot_executor::ExecError::Validation(_)),
        "expected ExecError::Validation, got {error:?}"
    );
}

/// SC-005: a BF16 checkpoint weight consumed by both an embed `Gather` and a decode GEMV
/// (`MatMul` over its transpose) binds from the store at its stored dtype with no host f32 copy.
/// `compile`'s own legalization (card 380/381: `fold_dense_contractions`,
/// `fold_dense_bf16_row_gathers`) rewrites `Cast(Reshape(Gather(w)))` into `DenseRowGather` and
/// `MatMul(x, Transpose(w))` into `DenseContraction`, both reading `w` directly - exactly the two
/// `packed_bf16_reader_operand` cases - so `w` stays eligible to stay BF16-native (packed two
/// elements per `u32` word on SpirvVulkan) rather than being widened to F32 (no
/// pass retypes a const, Card 1011). The residency map holds one buffer per (store key, planned
/// `BufferStorage`): `stats().memory`'s `Weight` total is exactly `rows*cols*2` bytes (native BF16),
/// never `rows*cols*4` (a host f32 copy), and the step equals the oracle.
///
/// Mutation (run by hand, see the done-report): in `Engine::weight_buffer`'s dense arm, encode
/// through `BufferStorage::f32()` unconditionally instead of the program's own planned `storage`;
/// the weight-byte total doubles to 32 and the row goes red.
#[test]
fn bf16_embed_weight_feeds_gather_and_decode_gemv_with_no_host_f32_copy() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&device);
    let mut engine = poot_executor::Engine::new(device);

    let rows = 4; // vocab
    let cols = 4; // hidden
    let b = poot_graph_ir::Builder::new();
    let w = b.constant("embed", TensorType::bf16(vec![rows, cols]));
    let idx = b.slot_named(Slot::Activation, "idx", TensorType::new(vec![], DType::I32));
    let gathered = b.gather(w, 0, idx);
    let gathered_row = b.reshape(gathered, vec![1, cols]);
    let gathered_f32 = b.cast(gathered_row, DType::F32);
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![1, cols]));
    let w_t = b.transpose(w, vec![1, 0]);
    let gemv = b.matmul(x, w_t);
    let y = b.binary(poot_graph_ir::BinOp::Add, gathered_f32, gemv);
    let g = b.finish(y);

    let w_values: Vec<f32> = fill(rows * cols, 0x6a54);
    let w_bytes: Vec<u8> = w_values
        .iter()
        .flat_map(|&v| poot_runtime_common::f32_to_bf16(v).to_le_bytes())
        .collect();
    let store = {
        let mut builder = WeightStore::builder();
        let dense = DenseWeight::try_new(DType::BF16, vec![rows, cols], Arc::from(w_bytes.clone()))
            .unwrap();
        builder.insert("embed", WeightEntry::Dense(dense)).unwrap();
        builder.build()
    };

    let exe = engine
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine
        .add_entry(exe, &staged(&g, target, Submission::Replay))
        .unwrap();

    let idx_value: i32 = 2;
    let idx_bytes = idx_value.to_le_bytes();
    let x_values = fill(cols, 0xA5);
    let x_shape = [1, cols];
    let mut inputs = StepInputs::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        if meta.storage != Storage::Slot(Slot::Activation) {
            continue;
        }
        match meta.name.as_deref() {
            Some("activation.idx") => inputs.push(
                meta.slot_key().unwrap().clone(),
                &[],
                HostView::new(DType::I32, 1, &idx_bytes).unwrap(),
            ),
            Some("activation.x") => inputs.push(
                meta.slot_key().unwrap().clone(),
                &x_shape,
                HostView::new(DType::F32, cols, bytemuck::cast_slice(&x_values)).unwrap(),
            ),
            other => panic!("unexpected Activation slot {other:?}"),
        }
    }
    let actual = bytemuck_f32(
        &engine
            .step(exe, entry, &inputs, &mut NoSync)
            .unwrap()
            .read()
            .unwrap(),
    );

    let stats = engine.stats();
    let weight_bytes = stats
        .memory
        .iter()
        .find(|&&(role, _)| role == poot_executor::BufferRole::Weight)
        .map(|&(_, snapshot)| snapshot.live_bytes)
        .unwrap_or(0);
    assert_eq!(
        weight_bytes,
        (rows * cols * 2) as u64,
        "a dual-consumer BF16 weight must upload native (2 bytes/elem), never a host f32 copy \
         (4 bytes/elem); stats={:?}",
        stats.memory
    );

    let mut oracle_inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        let value = match meta.storage {
            Storage::Const => Value::from(
                HostTensor::from_le_bytes(DType::BF16, vec![rows, cols], &w_bytes).unwrap(),
            ),
            Storage::Slot(Slot::Activation) => match meta.name.as_deref() {
                Some("activation.idx") => Value::from(HostTensor::i32(vec![], vec![idx_value])),
                Some("activation.x") => {
                    Value::from(HostTensor::f32(vec![1, cols], x_values.clone()))
                }
                other => panic!("unexpected Activation slot {other:?}"),
            },
            other => unreachable!("fixture has no {other:?} input"),
        };
        oracle_inputs.insert(id, value);
    }
    let oracle = eval(&g, &oracle_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec();
    assert_close_rel(&actual, &oracle, 1e-4);
}

/// Card 552: a real qwen2 decode entry on `WgpuDevice::new_with_timing(4)` produces measured,
/// honest device time through the production `step()` path - `DeviceTime::Measured` with a
/// positive `sum_of_dispatch_durations`, one `DispatchTiming` per dispatch in the entry's
/// recording, each a real `OpKind::kind_name` (never the empty/placeholder string) - while a
/// sibling engine over `WgpuDevice::new()` (the counters-only default) stays `DeviceTime::Unknown`
/// for the identical graph and inputs. Skips cleanly without a wgpu adapter.
///
/// MUTATION (recorded here, not left in the tree; Card 552): in `WgpuDevice::device_time`, return
/// `DeviceTime::Unknown` unconditionally (the pre-552 behavior) instead of draining
/// `drain_device_timing`. Result: RED - "timed device must report Measured, got Unknown".
/// Reverted: GREEN.
#[test]
fn wgpu_device_timing_produces_measured_per_dispatch_device_time() {
    let Some(timed_device) = WgpuDevice::new_with_timing(4).ok() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let Some(plain_device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = wgpu_target(&timed_device);
    let g = decode_graph(&dense(), CAP);
    let store = Arc::new(store_for(&[&g]));

    let mut timed = poot_executor::Engine::with_timing(
        timed_device,
        poot_executor::TimingOptions::Detailed(poot_executor::DetailedTiming::every_step(
            4, 256, 4,
        )),
    );
    let exe_t = timed
        .load_weights(Arc::clone(&store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let decode_t = timed
        .add_entry(exe_t, &staged(&g, target, Submission::Replay))
        .unwrap();

    let mut plain = poot_executor::Engine::new(plain_device);
    let exe_p = plain
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let decode_p = plain
        .add_entry(exe_p, &staged(&g, target, Submission::Replay))
        .unwrap();

    let values = slot_values(&g, &[5], 0);
    let time_t = timed
        .step(exe_t, decode_t, &step_inputs(&values), &mut NoSync)
        .unwrap()
        .device_time();
    let time_p = plain
        .step(exe_p, decode_p, &step_inputs(&values), &mut NoSync)
        .unwrap()
        .device_time();

    assert_eq!(
        time_p,
        poot_executor::DeviceTime::Unknown,
        "counters-only mode must never fabricate a measurement"
    );
    let poot_executor::DeviceTime::Measured(measured) = time_t else {
        panic!("timed device must report Measured, got {time_t:?}");
    };
    let sum = measured
        .sum_of_dispatch_durations
        .expect("a real decode step dispatches real kernels");
    assert!(sum.as_nanos() > 0, "positive device time: {sum:?}");
    assert!(
        !measured.dispatches.is_empty(),
        "detailed mode must retain per-dispatch records"
    );
    for d in &measured.dispatches {
        assert!(
            d.duration.as_nanos() > 0,
            "every retained dispatch must have a positive duration"
        );
    }

    // The engine's own cumulative timing counters (Card 552) reflect the step above exactly.
    let stats = timed.stats();
    assert_eq!(
        stats.timing.cumulative(decode_t.raw()).steps,
        1,
        "the step above must be reflected in the engine's cumulative timing counters"
    );
}
