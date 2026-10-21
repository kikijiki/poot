//! Runs tiny-config Qwen2 step graphs (`Model::trace` of the registry's qwen2 family) on the wgpu
//! executor and checks them against the CPU eager reference. Skips without a Vulkan adapter.
//!
//! Weights are data: every const is filled from its name, so the device and the oracle read the same
//! values. Each graph is the family's own (`poot_models::test_support::dense::Dense`), traced for a
//! [`StepShape`]; a paged step reads one shared KV pool through the `Slot::SlotMap` input `"read"` and
//! writes it through the forward `Slot::WriteMap` input `"write"` (one pool slot per new token).

use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalOptions, Value, eval};
use poot_executor::{Device, Executor, NoSync, StepInputs};
use poot_graph_ir::{ForwardSlotMap, Graph, Slot, SlotKey, Storage, ValueId, ValueTable};
use poot_graph_plan::{
    CompileOptions, Target, multi_device::topology::DeviceId, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::StagedProgram, staged::TargetSet,
    staged::compile_staged,
};
use poot_models::model::{LogitRows, StepShape, StepSpec};
use poot_models::test_support::dense::{Dense, Family, paged_step, step};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::{DType, HostTensor, HostView};
use poot_test_util::{assert_close_rel, max_abs_error};
use poot_wgpu_device::device::WgpuDevice;

/// Compile `g` for `target` through the executor contract's one staged program.
fn staged(g: &Graph, target: Target) -> StagedProgram {
    let g = g.clone();
    compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("compile the qwen2 graph")
}

fn f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

struct SlotBind {
    key: SlotKey,
    shape: Vec<usize>,
    dtype: DType,
    bytes: Vec<u8>,
    /// The checked map of a `Slot::WriteMap` input, which binds only through it.
    map: Option<ForwardSlotMap>,
}

fn step_inputs(binds: &[SlotBind]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for b in binds {
        if let Some(map) = &b.map {
            inputs.push_write_map(b.key.clone(), map);
            continue;
        }
        let elems = b.shape.iter().product();
        inputs.push(
            b.key.clone(),
            &b.shape,
            HostView::new(b.dtype, elems, &b.bytes).unwrap(),
        );
    }
    inputs
}

fn tensor_bytes(dtype: DType, t: &HostTensor) -> Vec<u8> {
    if dtype == DType::I32 {
        t.as_i32()
            .expect("an I32-declared input needs an I32 HostTensor")
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect()
    } else {
        t.as_f32()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }
}

/// Split a `HashMap<ValueId, Value>` bind into the contract's two bind mechanisms (`Storage::Const` ->
/// a named [`WeightStore`] entry, `Storage::Slot` -> a [`StepInputs`] row); `Storage::State`/
/// `Storage::Computed` are left for the contract to seed itself.
fn split_consts_and_slots_v(
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> (WeightStore, Vec<SlotBind>) {
    let mut builder = WeightStore::builder();
    let mut slots = Vec::new();
    for &id in g.inputs() {
        let m = g.meta(id);
        let dtype = m.aval.dtype;
        match m.storage {
            Storage::Const => {
                let name = m.name().expect("named const").into_owned();
                let t = inputs
                    .get(&id)
                    .and_then(Value::as_host)
                    .unwrap_or_else(|| panic!("missing const input {id:?}"));
                let bytes = tensor_bytes(dtype, t);
                let dense =
                    DenseWeight::try_new(dtype, t.shape().to_vec(), StoredBytes::from(bytes))
                        .unwrap();
                builder.insert(name, WeightEntry::Dense(dense)).unwrap();
            }
            Storage::Slot(_) => {
                let t = inputs
                    .get(&id)
                    .and_then(Value::as_host)
                    .unwrap_or_else(|| panic!("missing slot input {id:?}"));
                let map = (m.slot_key().unwrap().role() == Slot::WriteMap).then(|| {
                    let pool = m
                        .range()
                        .expect("a write map declares its pool")
                        .hi_exclusive();
                    ForwardSlotMap::new(t.as_i32().unwrap().to_vec(), pool as usize)
                        .expect("the feed's write map is distinct slots of the pool")
                });
                slots.push(SlotBind {
                    key: m.slot_key().unwrap().clone(),
                    shape: t.shape().to_vec(),
                    dtype,
                    bytes: tensor_bytes(dtype, t),
                    map,
                });
            }
            Storage::State | Storage::Computed(_) => {}
            Storage::Device => unreachable!(),
        }
    }
    (builder.build(), slots)
}

/// The executor-contract replacement for the deleted `GpuExecutor::run`/`run_resident`/`run_resident_kv` on a
/// one-shot graph (`Value`-typed bind): `add_entry`/`step` through `Engine<WgpuDevice>`, torn back
/// down afterward. A graph with state zero-seeds it automatically on first use (Card 546a).
fn run_once_contract_v(
    exec: &mut dyn Executor,
    target: Target,
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> Vec<f32> {
    let (store, slot_binds) = split_consts_and_slots_v(g, inputs);
    let (bound, map) = ConstWeights::bind(g, &store);
    let program = staged(&bound, target);
    let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let bytes = exec
        .step(exe, entry, &step_in, &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    f32_bytes(&bytes)
}

/// A graph whose sole output IS one named state value, for reading a carried state buffer back through
/// a real step output (same pattern as `wgpu_executor_contract.rs`'s `state_probe_graph` - no `read_state`
/// accessor exists).
fn state_probe_graph(name: &str, aval: poot_graph_ir::TensorType) -> Graph {
    let b = poot_graph_ir::Builder::for_tests();
    let state = b
        .state_input(name, aval, poot_graph_ir::StateRole::Recurrent)
        .unwrap();
    b.finish(poot_graph_ir::Outputs::new(state).with_state(&[(state, state)]))
        .unwrap()
}

/// Compares raw contract output bytes against the CPU oracle `Tensor`, within relative tolerance.
/// `step().read()` returns flat bytes with no shape of its own, so `g`'s declared output aval is the
/// only source of the shape the comparison checks (review 546b-b1b F1): a length-only check would
/// miss a same-numel shape mismatch.
fn assert_close_bytes(g: &Graph, got: &[f32], cpu: &HostTensor, label: &str) {
    assert_eq!(g.aval(g.result()).shape, cpu.shape(), "{label}: shape");
    assert_eq!(got.len(), cpu.as_f32().unwrap().len(), "{label}: length");
    assert_close_rel(got, cpu.as_f32().unwrap(), 5e-3);
    eprintln!(
        "{label}: GPU matches CPU (max abs diff {:.2e})",
        max_abs_error(got, cpu.as_f32().unwrap())
    );
}

/// Serializes GPU access within this test binary: the Intel Arc Vulkan driver segfaults when several tests use
/// wgpu devices concurrently. Poison-tolerant. Hold the guard for the whole test.
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.1
        })
        .collect()
}

/// The deterministic weight of the const `name`: the same values for every graph that names it.
fn weight(name: &str, shape: &[usize]) -> HostTensor {
    let seed = name.bytes().fold(1469598103934665603u64, |h, c| {
        (h ^ c as u64).wrapping_mul(1099511628211)
    });
    HostTensor::f32(
        shape.to_vec(),
        fill(shape.iter().product::<usize>().max(1), seed),
    )
}

/// The tiny qwen2 every row here traces (4 query heads sharing 2 key-value heads of width 4).
fn tiny() -> Dense {
    Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
}

/// `dense`'s step graph at `shape`, as an ordinary graph.
fn trace(dense: &Dense, shape: StepShape) -> Graph {
    dense
        .f32_model()
        .model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap()
}

/// One step's slot values: the new tokens and their absolute positions (`[rows, tokens]`, row major)
/// and, under a paged layout, the two slot maps.
#[derive(Clone, Default)]
struct Feed {
    tokens: Vec<i32>,
    pos: Vec<i32>,
    read: Vec<i32>,
    write: Vec<i32>,
}

impl Feed {
    fn new(tokens: Vec<i32>, pos: Vec<i32>) -> Self {
        Self {
            tokens,
            pos,
            ..Self::default()
        }
    }

    /// The paged slot maps: `read` is `[rows, capacity]`, `write` is `[rows * tokens]` (one pool slot per
    /// new token, -1 for padding).
    fn paged(mut self, read: Vec<i32>, write: Vec<i32>) -> Self {
        (self.read, self.write) = (read, write);
        self
    }
}

/// Binds every input of `g`: slots from `feed`, consts by name, computed constants as they are, and
/// the carried state zero-filled when `seed_state` (the CPU oracle needs it; the contract seeds its
/// own).
fn bind(g: &Graph, feed: &Feed, seed_state: bool) -> HashMap<ValueId, Value> {
    let mut m = HashMap::new();
    for &id in g.inputs() {
        let meta = g.meta(id);
        let shape = meta.aval.shape.clone();
        let t = match meta.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(shape, feed.tokens.clone()),
            Storage::Slot(Slot::Pos) => HostTensor::i32(shape, feed.pos.clone()),
            Storage::Slot(Slot::SlotMap) => match meta.name().as_deref() {
                Some("slotmap.read") => HostTensor::i32(shape, feed.read.clone()),
                other => panic!("unexpected slot map {other:?}"),
            },
            Storage::Slot(Slot::WriteMap) => match meta.name().as_deref() {
                Some("writemap.write") => HostTensor::i32(shape, feed.write.clone()),
                other => panic!("unexpected write map {other:?}"),
            },
            Storage::Slot(other) => panic!("unexpected slot {other:?}"),
            Storage::Const => {
                assert_eq!(meta.aval.dtype, DType::F32, "an F32 model's const");
                weight(&meta.name().expect("a named const"), &shape)
            }
            Storage::Computed(c) => c.tensor(),
            Storage::State if seed_state => HostTensor::zeros(shape),
            Storage::State => continue,
            Storage::Device => unreachable!(),
        };
        m.insert(id, t.into());
    }
    m
}

/// One CPU step: the output and the next state, from `inputs` (state excluded) and the carried
/// `state` (empty on the first step: zeros).
fn cpu_step(
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
    state: &[HostTensor],
) -> (HostTensor, Vec<HostTensor>) {
    let mut cpu_inputs = inputs.clone();
    for (ci, (si, _)) in g.state().enumerate() {
        let carried = state
            .get(ci)
            .cloned()
            .unwrap_or_else(|| HostTensor::zeros(g.aval(si).shape.clone()));
        cpu_inputs.insert(si, carried.into());
    }
    let r = eval(g, &cpu_inputs, EvalOptions::default()).unwrap();
    let out = r.output.into_host().unwrap();
    let next = r
        .state
        .into_iter()
        .map(|v| v.into_host().unwrap())
        .collect();
    (out, next)
}

/// `g` replayed on one entry across `feeds`, the KV state carried by the contract and by the CPU
/// oracle, each step's device logits equal to the oracle's.
fn assert_replay_matches_cpu(g: &Graph, feeds: &[Feed], label: &str) {
    let _gpu_guard = gpu_lock();
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let (store, _) = split_consts_and_slots_v(g, &bind(g, &feeds[0], false));
    let (bound, map) = ConstWeights::bind(g, &store);
    let (store, map) = (Arc::new(store), Arc::new(map));
    let program = staged(&bound, target);
    let mut cpu_state: Vec<HostTensor> = Vec::new();
    let mut loaded: Option<(poot_executor::ExecutableId, poot_executor::EntryId)> = None;
    for (i, feed) in feeds.iter().enumerate() {
        let inputs = bind(g, feed, false);
        let (cpu_logits, next) = cpu_step(g, &inputs, &cpu_state);
        cpu_state = next;
        let (_, slot_binds) = split_consts_and_slots_v(g, &inputs);
        let (exe, entry) = *loaded.get_or_insert_with(|| {
            let exe = exec
                .load_weights(Arc::clone(&store), Arc::clone(&map))
                .unwrap();
            (exe, exec.add_entry(exe, &program).unwrap())
        });
        let gpu_logits = f32_bytes(
            &exec
                .step(exe, entry, &step_inputs(&slot_binds), &mut NoSync)
                .unwrap()
                .read()
                .unwrap(),
        );
        assert_close_bytes(g, &gpu_logits, &cpu_logits, &format!("{label} step {i}"));
    }
    if let Some((exe, entry)) = loaded {
        exec.remove_entry(exe, entry).unwrap();
        exec.unload(exe).unwrap();
    }
}

/// A one-token decode step at position `pos` over a cache of 8 positions, and its feed.
fn tiny_decode_graph() -> (Graph, Feed) {
    let g = trace(&tiny(), step(1, 1, 8, LogitRows::Last));
    (g, Feed::new(vec![5], vec![3]))
}

#[test]
fn qwen2_decode_gpu_matches_cpu() {
    let _gpu_guard = gpu_lock();
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let (g, feed) = tiny_decode_graph();
    // `GpuExecutor::run` (the host round-trip path this test used to drive) is deleted outright
    // (`executor/run.rs`); the contract has one path, the device-resident one every other row here
    // already exercises.
    let (cpu, _) = cpu_step(&g, &bind(&g, &feed, false), &[]);
    let got = run_once_contract_v(&mut *exec, target, &g, &bind(&g, &feed, false));
    assert_close_bytes(&g, &got, &cpu, "qwen2 tiny decode on GPU");
}

#[test]
fn flash_attention_pass_prefill_gpu_matches_cpu() {
    // card 038: the flash-attention pass inside `compile` rewrites a raw prefill trace's materialized attention into
    // Attention, and the rewritten graph runs on the wgpu executor with the same logits as the
    // un-rewritten prefill on CPU. Pipeline: canonicalize -> cse -> fold_iota -> rope_fusion -> flash_attention -> dce
    // -> GPU (the mask is a graph computation over `Slot::Pos`, so the flash pass needs it folded).
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::{canonicalize, cse, dce, flash_attention_capped, fold_iota, rope_fusion};
    let _gpu_guard = gpu_lock();
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let tokens = [5, 9, 2, 14, 7, 1];
    let l = tokens.len();
    // cap == the prompt: the flash pass recognises the square causal prefill, not a prompt over a longer cache.
    let g = trace(&tiny(), step(1, l, l, LogitRows::Last));
    let feed = Feed::new(tokens.to_vec(), (0..l as i32).collect());
    let fused = dce(&flash_attention_capped(
        &rope_fusion(&fold_iota(&cse(&canonicalize(&g).unwrap()).unwrap()).unwrap())
            .unwrap()
            .0,
        None,
    )
    .unwrap()
    .0)
    .unwrap();
    assert!(
        fused
            .eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Attention(_))),
        "the flash pass must rewrite the prefill attention"
    );
    let (cpu, _) = cpu_step(&g, &bind(&g, &feed, false), &[]);
    let got = run_once_contract_v(&mut *exec, target, &fused, &bind(&fused, &feed, false));
    assert_close_bytes(
        &fused,
        &got,
        &cpu,
        "flash_attention pass prefill on GPU vs materialized on CPU",
    );
}

#[test]
fn prefill_runs_a_flash_fused_compiled_graph_and_matches_cpu() {
    // card 183/535b: the caller `compile`s the graph with full fusion (flash-attention included) before
    // running it. Uses a 64-token prefill (large enough that N^2 attention is not degenerate) and
    // checks GPU logits against a zero-seeded-state CPU reference on the unoptimized graph.
    let _gpu_guard = gpu_lock();
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let n = 64usize;
    let dense = tiny().max_positions(128);
    let g = trace(&dense, step(1, n, n, LogitRows::Last));
    let feed = Feed::new(
        (0..n).map(|i| (i % dense.vocab) as i32).collect(),
        (0..n as i32).collect(),
    );
    let (cpu, _) = cpu_step(&g, &bind(&g, &feed, false), &[]);
    let got = run_once_contract_v(&mut *exec, target, &g, &bind(&g, &feed, false));
    assert_eq!(g.aval(g.result()).shape, cpu.shape());
    assert_eq!(got.len(), cpu.as_f32().unwrap().len());
    assert_close_rel(&got, cpu.as_f32().unwrap(), 5e-3);
    eprintln!(
        "prefill flash-fused GPU vs CPU: max abs diff = {:.2e}",
        max_abs_error(&got, cpu.as_f32().unwrap())
    );
}

/// A prefill of `n` tokens over a paged pool whose logical position `p < n` sits at pool slot
/// `slot_map[p]` writes the same K/V as the contiguous prefill, and gives the same logits. Compares the
/// filled caches, not just the logits (which use the fresh k/v directly and match regardless of the
/// write): on the CPU oracle, then on the device against the oracle's pool and the contiguous run.
fn assert_paged_prefill_matches_contiguous(slot_map: &[usize], cap: usize, pool: usize) {
    let _gpu_guard = gpu_lock();
    let dense = tiny();
    let n = slot_map.len();
    let (hkv, d) = (dense.kv_heads, dense.head_dim.unwrap());
    let g_cont = trace(&dense, step(1, n, cap, LogitRows::Last));
    let g_paged = trace(&dense, paged_step(1, n, cap, pool, LogitRows::Last));
    let tokens: Vec<i32> = (0..n).map(|j| (j % dense.vocab) as i32).collect();
    let pos: Vec<i32> = (0..n as i32).collect();
    // position p -> slot_map[p]; positions past the prompt read slot 0 (they are masked).
    let read: Vec<i32> = (0..cap)
        .map(|p| slot_map.get(p).map_or(0, |&s| s as i32))
        .collect();
    // token p of the prompt writes the slot of position p.
    let write: Vec<i32> = slot_map.iter().map(|&s| s as i32).collect();
    let feed_c = Feed::new(tokens.clone(), pos.clone());
    let feed_p = Feed::new(tokens, pos).paged(read, write);

    let env = |g: &Graph, feed: &Feed| -> ValueTable<Option<HostTensor>> {
        eval(
            g,
            &bind(g, feed, true),
            EvalOptions::default().keep_environment(),
        )
        .unwrap()
        .environment
        .unwrap()
        .into_iter()
        .map(|slot| slot.map(|v| v.into_host().unwrap()))
        .collect()
    };
    let (env_c, env_p) = (env(&g_cont, &feed_c), env(&g_paged, &feed_p));
    assert_eq!(g_cont.state().count(), g_paged.state().count());
    // each layer: contiguous [1,Hkv,cap,D] position p vs the pool [pool,Hkv,D] slot slot_map[p].
    for ((_, so_c), (_, so_p)) in g_cont.state().zip(g_paged.state()) {
        let cc = env_c[so_c].as_ref().unwrap().as_f32().unwrap();
        let cp = env_p[so_p].as_ref().unwrap().as_f32().unwrap();
        for (p, &slot) in slot_map.iter().enumerate() {
            for hh in 0..hkv {
                for dd in 0..d {
                    let cont_v = cc[(hh * cap + p) * d + dd];
                    let pooled = cp[(slot * hkv + hh) * d + dd];
                    assert!(
                        (cont_v - pooled).abs() < 1e-5,
                        "pos {p} slot {slot} h{hh} d{dd}: contiguous {cont_v} vs paged {pooled}"
                    );
                }
            }
        }
    }
    // unused pool slots stay zero (passed through from the zero-seeded base).
    let pool0 = env_p[g_paged.state().next().unwrap().1]
        .as_ref()
        .unwrap()
        .as_f32()
        .unwrap();
    for slot in (0..pool).filter(|s| !slot_map.contains(s)) {
        for x in &pool0[slot * hkv * d..(slot + 1) * hkv * d] {
            assert_eq!(*x, 0.0, "unused slot {slot} must stay zero");
        }
    }
    let logit_diff = max_abs_error(
        env_p[g_paged.result()].as_ref().unwrap().as_f32().unwrap(),
        env_c[g_cont.result()].as_ref().unwrap().as_f32().unwrap(),
    );
    assert!(
        logit_diff < 1e-5,
        "paged prefill logits diverge: {logit_diff}"
    );

    // GPU: the paged prefill lowers and runs on the contract; its logits match the contiguous run and
    // its filled pool buffers match the CPU pool. The engine hands these buffers to the paged decode,
    // so a GPU-only write bug would produce garbage decode. A state-probe entry per carried pair reads
    // the filled cache back (same pattern as `wgpu_executor_contract.rs`).
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let cont = run_once_contract_v(&mut *exec, target, &g_cont, &bind(&g_cont, &feed_c, false));
    let inputs = bind(&g_paged, &feed_p, false);
    let (store, slot_binds) = split_consts_and_slots_v(&g_paged, &inputs);
    let (bound, map) = ConstWeights::bind(&g_paged, &store);
    let program = staged(&bound, target);
    let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let paged = f32_bytes(
        &exec
            .step(exe, entry, &step_inputs(&slot_binds), &mut NoSync)
            .unwrap()
            .read()
            .unwrap(),
    );
    let d = max_abs_error(&paged, &cont);
    assert!(
        d < 1e-3,
        "GPU paged prefill logits diverge from contiguous: {d}"
    );
    for (i, (si, so)) in g_paged.state().enumerate() {
        let cpu = env_p[so].as_ref().unwrap();
        let name = g_paged.meta(si).name().expect("named state").into_owned();
        let probe = state_probe_graph(&name, g_paged.aval(si).clone());
        let probe_entry = exec.add_entry(exe, &staged(&probe, target)).unwrap();
        let got = f32_bytes(
            &exec
                .step(exe, probe_entry, &StepInputs::new(), &mut NoSync)
                .unwrap()
                .read()
                .unwrap(),
        );
        exec.remove_entry(exe, probe_entry).unwrap();
        let max_abs = max_abs_error(&got, cpu.as_f32().unwrap());
        assert!(
            max_abs < 1e-4,
            "GPU paged prefill cache {i} diverges from CPU: max_abs={max_abs}"
        );
    }
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
}

#[test]
fn prefill_kv_paged_matches_contiguous() {
    // card 059: a paged prefill with an identity map must fill the KV pool exactly as the contiguous
    // prefill fills its cache.
    assert_paged_prefill_matches_contiguous(&[0, 1, 2, 3, 4], 8, 8);
}

#[test]
fn prefill_kv_shared_pool_matches_contiguous() {
    // card 059: a paged prefill writes the N positions into the [pool,Hkv,D] cache at global slots
    // given by a permutation slot map, passing unused slots through. Each position p's KV at
    // slot_map[p] must equal the contiguous prefill's position-p KV.
    assert_paged_prefill_matches_contiguous(&[2, 0, 5, 7, 3], 8, 8);
}

#[test]
fn qwen2_decode_gpu_resident_matches_cpu() {
    let _gpu_guard = gpu_lock();
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let (g, feed) = tiny_decode_graph();
    let (cpu, _) = cpu_step(&g, &bind(&g, &feed, false), &[]);
    let got = run_once_contract_v(&mut *exec, target, &g, &bind(&g, &feed, false));
    assert_close_bytes(&g, &got, &cpu, "qwen2 tiny decode on GPU (device resident)");
}

/// G5a: the fused decode graph (pointwise chains synthesized into single kernels) runs on the Arc and matches
/// the un-fused graph within a tight fp tolerance, with fewer dispatches. Fusion moves intermediates from f32
/// global buffers to registers (~1 ULP rounding difference), so the result is not bit-identical.
#[test]
fn fused_decode_matches_unfused_on_gpu() {
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::{cse, fuse};
    let _gpu_guard = gpu_lock();
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    let (g, feed) = tiny_decode_graph();
    let base = cse(&g).unwrap(); // the executor's normal pre-pass; fusion groups on top of it.
    let fg = fuse(&base).unwrap();
    let n_fused = fg
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::Fused(_)))
        .count();
    let n_row = fg
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::FusedRow(_)))
        .count();
    assert!(n_fused > 0, "pointwise fusion should fire");
    assert!(n_row > 0, "reduction-rooted fusion should fire");
    assert!(
        fg.eqns().len() < base.eqns().len(),
        "fewer dispatches after fusion: {} -> {}",
        base.eqns().len(),
        fg.eqns().len()
    );
    eprintln!(
        "fused decode: {} eqns -> {} ({} Fused, {} FusedRow regions)",
        base.eqns().len(),
        fg.eqns().len(),
        n_fused,
        n_row
    );

    let unfused_gpu = run_once_contract_v(&mut *exec, target, &base, &bind(&base, &feed, false));
    let fused_gpu = run_once_contract_v(&mut *exec, target, &fg, &bind(&fg, &feed, false));
    // Fusion keeps region intermediates in registers instead of f32 global buffers, so they can round ~1 ULP
    // differently: the fused result equals the unfused one only within a tight fp tolerance (~9e-10 max_abs
    // measured on 4/32 elems).
    assert_eq!(
        fused_gpu.len(),
        unfused_gpu.len(),
        "fused/unfused output length"
    );
    let max_abs = max_abs_error(&fused_gpu, &unfused_gpu);
    assert!(
        max_abs <= 1e-5,
        "fused GPU result must match un-fused within fp tolerance: max_abs={max_abs:.3e}"
    );
    // and both match the CPU reference.
    let (cpu, _) = cpu_step(&g, &bind(&g, &feed, false), &[]);
    assert_close_bytes(&g, &fused_gpu, &cpu, "fused qwen2 tiny decode on GPU");
}

// `const_cache_warms_after_first_run` deleted (Card 546b): it tested GpuExecutor's own
// per-name const-cache warmth counter (`const_uploads()`). The contract loads an executable's weights
// exactly once at `add_entry` (`Engine::weight_buffer`'s residency map), never re-checked per `step`,
// so there is no "does the second run re-upload" decision left to observe - it is structurally true by
// construction, not a runtime cache-hit outcome - and no equivalent counter exists on `Executor`/
// `ExecutorStats`. This is the "decode-cache warmth" case the contract excludes.

// `kernel_disk_cache_persists_across_executors` deleted (Card 546b): it tested
// the deleted `GpuExecutor::kernel_compiles()`, a counter with no equivalent on the `Executor` trait or
// `ExecutorStats` (verified: `poot-executor`'s `kernel_cache` field is private, with no compile-count
// accessor). The underlying on-disk `poot_codegen::KernelCache` sharing this test exercised is the
// SAME cache `Engine` uses (`Engine::with_timing`'s `kernel_cache: poot_codegen::KernelCache::open(..)`)
// and is independently tested at the `poot_codegen::cache` level (persistence-across-instances,
// epoch invalidation), so the underlying claim is not left untested - only the observability of the deleted
// `GpuExecutor` is gone.

/// Paged decode graph (spec 045) runs on the wgpu executor and matches the CPU reference with a non-identity
/// (reverse) slot map. The K/V gather is composed from on-device primitives (transpose + axis-0 gather +
/// transpose), so there is no host fallback.
#[test]
fn paged_decode_kv_gpu_matches_cpu() {
    let (cap, tokens) = (8usize, [5, 9, 2, 14, 7]);
    // non-identity layout: logical t -> physical slot cap-1-t. The graph writes logical pos `p` at slot_map[p] and
    // reads by gathering slot_map[t], so it must reconstruct logical order regardless.
    let read: Vec<i32> = (0..cap).map(|t| (cap - 1 - t) as i32).collect();
    let g = trace(&tiny(), paged_step(1, 1, cap, cap, LogitRows::Last));
    let feeds: Vec<Feed> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &token)| {
            let write = vec![(cap - 1 - pos) as i32];
            Feed::new(vec![token], vec![pos as i32]).paged(read.clone(), write)
        })
        .collect();
    assert_replay_matches_cpu(&g, &feeds, "paged kv decode GPU");
}

/// `batch` rows of a paged decode, each with its own pool region and a per-row non-identity (reverse)
/// slot map, advancing one token per step over device-resident KV: the per-row scatter/gather lowers
/// on-device.
#[test]
fn batched_paged_decode_gpu_matches_cpu() {
    let (batch, cap, n_steps) = (2usize, 8usize, 4usize);
    let vocab = tiny().vocab;
    let g = trace(
        &tiny(),
        paged_step(batch, 1, cap, batch * cap, LogitRows::Last),
    );
    // row r owns slots [r*cap, r*cap+cap), reversed.
    let slot = |r: usize, t: usize| (r * cap + (cap - 1 - t)) as i32;
    let read: Vec<i32> = (0..batch)
        .flat_map(|r| (0..cap).map(move |t| slot(r, t)))
        .collect();
    let feeds: Vec<Feed> = (0..n_steps)
        .map(|s| {
            let tokens = (0..batch)
                .map(|r| (((s * batch + r) * 5 + 2) % vocab) as i32)
                .collect();
            let write: Vec<i32> = (0..batch).map(|r| slot(r, s)).collect();
            Feed::new(tokens, vec![s as i32; batch]).paged(read.clone(), write)
        })
        .collect();
    assert_replay_matches_cpu(&g, &feeds, "batched paged decode GPU");
}

/// Shared-pool batched decode (card 046b): all rows share one [P,Hkv,D] pool, the slot map holds global slots.
/// Runs on the wgpu executor and matches CPU. B=2 rows interleaved in a pool of `B*n_steps` slots (< B*cap).
#[test]
fn batched_shared_pool_decode_gpu_matches_cpu() {
    let (batch, cap, n_steps) = (2usize, 8usize, 4usize);
    let pool_slots = batch * n_steps; // 8 < B*cap = 16
    let vocab = tiny().vocab;
    let g = trace(
        &tiny(),
        paged_step(batch, 1, cap, pool_slots, LogitRows::Last),
    );
    // global interleaved slot map [B, cap]: (row r, pos t) -> t*B + r for used positions, else slot 0.
    let read: Vec<i32> = (0..batch)
        .flat_map(|r| {
            (0..cap).map(move |t| {
                if t < n_steps {
                    (t * batch + r) as i32
                } else {
                    0
                }
            })
        })
        .collect();
    let feeds: Vec<Feed> = (0..n_steps)
        .map(|s| {
            let tokens = (0..batch)
                .map(|r| (((s * batch + r) * 5 + 2) % vocab) as i32)
                .collect();
            let write: Vec<i32> = (0..batch).map(|r| (s * batch + r) as i32).collect();
            Feed::new(tokens, vec![s as i32; batch]).paged(read.clone(), write)
        })
        .collect();
    assert_replay_matches_cpu(&g, &feeds, "batched shared-pool decode GPU");
}

/// G3d-3: the constant-shape masked decode graph (built once) runs device-resident with the KV cache carried
/// across steps and matches the CPU reference step for step.
#[test]
fn qwen2_masked_decode_gpu_matches_cpu() {
    let tokens = [5, 9, 2, 14, 7];
    // the decode graph is built once (depends only on cap) and replayed for every position.
    let g = trace(&tiny(), step(1, 1, tokens.len(), LogitRows::Last));
    let feeds: Vec<Feed> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &token)| Feed::new(vec![token], vec![pos as i32]))
        .collect();
    assert_replay_matches_cpu(&g, &feeds, "masked decode GPU");
}

/// POOT-1209: decode attention is bounded by the live length `pos + 1`. A 24-position cache stepped through
/// every position, so the first (`pos` 0, a length of one), 15 and the last (`capacity - 1`, a full length)
/// are all compared with the CPU oracle, each step reading what the earlier ones wrote. The CPU oracle runs the
/// same graph, so this row proves the device's live-bound attention kernel; the length itself is pinned
/// against an independent f64 reference by `components::attention`'s
/// `a_decode_token_attends_to_exactly_its_live_prefix_at_the_first_a_middle_and_the_last_position`.
#[test]
fn qwen2_live_bound_decode_gpu_matches_cpu_at_the_first_a_middle_and_the_last_position() {
    const CAP: usize = 24;
    let dense = Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(CAP);
    let g = trace(&dense, step(1, 1, CAP, LogitRows::Last));
    let feeds: Vec<Feed> = (0..CAP)
        .map(|pos| Feed::new(vec![(pos as i32 * 7 + 3) % 32], vec![pos as i32]))
        .collect();
    assert_replay_matches_cpu(&g, &feeds, "live-bound decode GPU");
}

/// The real Qwen2.5-0.5B architecture (24 layers, hidden 896, 14 heads over 2 key-value heads), its
/// vocabulary cut to 4096 (the vocabulary does not change the graph's structure). The checkpoint is
/// zeros: these rows read the graph, not its values.
fn real_dense() -> Dense {
    Dense::new(Family::Qwen2)
        .vocab(4096)
        .dims(896, 4864, 24)
        .heads(14, 2)
        .head_dim(64)
        .max_positions(32768)
}

/// `dense`'s decode graph over `cap` positions, from a zero checkpoint.
fn real_trace(dense: &Dense, shape: StepShape) -> Graph {
    dense
        .zeroed_model(DType::BF16)
        .model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap()
}

/// G5 dispatch-reduction receipt on the real Qwen2.5-0.5B architecture (weights-free: dispatch count depends
/// only on graph structure). Counts the kernels that dispatch (Plan::Dispatches) before vs after fusion and
/// asserts a material reduction.
#[test]
fn fusion_dispatch_reduction_on_real_qwen2_config() {
    use poot_graph_plan::test_support::plan_eqn;
    use poot_graph_plan::{Plan, cse, fuse};
    use poot_target::Backend;

    fn count_dispatches(g: &poot_graph_ir::Graph) -> usize {
        g.eqns()
            .iter()
            .filter(|eqn| {
                matches!(
                    plan_eqn(
                        g,
                        eqn,
                        Backend::SpirvVulkan,
                        &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan)
                    ),
                    Ok(Plan::Dispatches(_))
                )
            })
            .count()
    }

    let cap = 64;
    let base = cse(&real_trace(&real_dense(), step(1, 1, cap, LogitRows::Last))).unwrap();
    let fused = fuse(&base).unwrap();

    let before = count_dispatches(&base);
    let after = count_dispatches(&fused);
    let n_fused = fused
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, poot_graph_ir::op::OpKind::Fused(_)))
        .count();
    let n_row = fused
        .eqns()
        .iter()
        .filter(|e| matches!(e.op, poot_graph_ir::op::OpKind::FusedRow(_)))
        .count();
    eprintln!(
        "real Qwen2.5-0.5B decode dispatches/token: {before} -> {after} after fusion \
         ({n_fused} Fused + {n_row} FusedRow regions)"
    );
    assert!(
        after < before,
        "fusion must cut the real-model dispatch count: {before} -> {after}"
    );
    // sanity: reduction-rooted fusion fires on every layer's norms + softmax (24 layers).
    assert!(n_row >= 24, "expected >= 24 FusedRow regions, got {n_row}");

    // breakdown of the post-fusion dispatching eqns by op kind, to see whether a fusion/aliasing win is left.
    let mut by_kind: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for eqn in fused.eqns() {
        if matches!(
            plan_eqn(
                &fused,
                eqn,
                Backend::SpirvVulkan,
                &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan)
            ),
            Ok(Plan::Dispatches(_))
        ) {
            let kind = eqn.op.name();
            let head = kind.split_whitespace().next().unwrap_or(&kind).to_string();
            *by_kind.entry(head).or_default() += 1;
        }
    }
    let mut v: Vec<_> = by_kind.into_iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    eprintln!("dispatching-eqn breakdown: {v:?}");

    // distinguish residual adds (output = hidden) from rope/mask adds (head_dim / cap): are the standalone adds
    // inherent (multi-use residual) or a missed fusion chain.
    let mut add_shapes: std::collections::BTreeMap<Vec<usize>, usize> = Default::default();
    for eqn in fused.eqns() {
        if eqn.op.name().starts_with("add")
            && matches!(
                plan_eqn(
                    &fused,
                    eqn,
                    Backend::SpirvVulkan,
                    &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan)
                ),
                Ok(Plan::Dispatches(_))
            )
        {
            *add_shapes
                .entry(fused.aval(eqn.out).shape.clone())
                .or_default() += 1;
        }
    }
    eprintln!("standalone add output shapes: {add_shapes:?}");
}

/// Card 550 SC-006 (wgpu leg): a chunked prefill starting at a NONZERO position
/// equals the same tokens run one decode step at a time, on real wgpu hardware through the executor
/// contract - the hardware sibling of poot-eval's
/// `chunked_prefill_at_nonzero_start_matches_stepped_decode` (CPU oracle only). Two independent
/// executables (so each starts from the same zero state) are both seeded identically by prefilling the
/// first `s` positions, then diverge: one continues as a second chunk-prefill replay of `[s, s+l)`, the
/// other steps a paged decode entry `l` times over the same range; both read the pool through the
/// same non-contiguous `global_slots` map, so this also proves the chunk's write/read addressing
/// matches decode's, not just its mask.
#[test]
fn chunked_prefill_at_nonzero_start_matches_stepped_decode_gpu() {
    let _gpu_guard = gpu_lock();
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

    let dense = tiny().dims(16, 24, 2);
    let s = 3usize; // nonzero chunk start
    let l = 3usize; // chunk width (== s, so the prefill entry's shape is reused for both halves)
    let cap = s + l;
    let pool_slots = cap + 4;
    let global_slots: Vec<usize> = (0..cap).map(|i| cap + 3 - i).collect(); // non-contiguous
    let tokens: Vec<i32> = vec![5, 11, 3, 20, 8, 1];
    let read: Vec<i32> = global_slots.iter().map(|&p| p as i32).collect();
    // the chunk of `len` tokens from position `start` writes its token `i` at the slot of `start + i`.
    let feed = |start: usize, len: usize| -> Feed {
        let write: Vec<i32> = (0..len).map(|i| global_slots[start + i] as i32).collect();
        Feed::new(
            tokens[start..start + len].to_vec(),
            (start as i32..(start + len) as i32).collect(),
        )
        .paged(read.clone(), write)
    };

    let g_prefill = trace(&dense, paged_step(1, l, cap, pool_slots, LogitRows::Last));
    let g_decode = trace(&dense, paged_step(1, 1, cap, pool_slots, LogitRows::Last));
    // One merged store (by const name) for both graphs, loaded fresh per path below so each path
    // starts from independent zero state.
    let merged_store = || -> WeightStore {
        let mut builder = WeightStore::builder();
        let mut seen = std::collections::HashSet::new();
        for g in [&g_prefill, &g_decode] {
            for &id in g.inputs() {
                let meta = g.meta(id);
                if meta.storage != Storage::Const {
                    continue;
                }
                let name = meta.name().unwrap().into_owned();
                if !seen.insert(name.clone()) {
                    continue;
                }
                let data = weight(&name, &meta.aval.shape);
                let bytes = tensor_bytes(DType::F32, &data);
                let dense = DenseWeight::try_new(
                    DType::F32,
                    meta.aval.shape.clone(),
                    StoredBytes::from(bytes),
                )
                .unwrap();
                builder.insert(name, WeightEntry::Dense(dense)).unwrap();
            }
        }
        builder.build()
    };
    let merged = merged_store();
    let mut weights = ConstWeights::new(&merged);
    let (named_prefill, named_decode) = (weights.graph(&g_prefill), weights.graph(&g_decode));
    let weight_map = Arc::new(weights.map());
    let staged_prefill = staged(&named_prefill, target);
    let staged_decode = staged(&named_decode, target);
    let slots_of = |g: &Graph, feed: &Feed| split_consts_and_slots_v(g, &bind(g, feed, false)).1;

    // --- path A: fill [0,s), then a second chunk-prefill replay fills [s, s+l) - logits_chunk. ---
    let exe_a = exec
        .load_weights(Arc::new(merged_store()), Arc::clone(&weight_map))
        .unwrap();
    let entry_a = exec.add_entry(exe_a, &staged_prefill).unwrap();
    exec.step(
        exe_a,
        entry_a,
        &step_inputs(&slots_of(&g_prefill, &feed(0, s))),
        &mut NoSync,
    )
    .unwrap()
    .read()
    .unwrap();
    let logits_chunk = f32_bytes(
        &exec
            .step(
                exe_a,
                entry_a,
                &step_inputs(&slots_of(&g_prefill, &feed(s, l))),
                &mut NoSync,
            )
            .unwrap()
            .read()
            .unwrap(),
    );
    exec.remove_entry(exe_a, entry_a).unwrap();
    exec.unload(exe_a).unwrap();

    // --- path B: fill [0,s) the same way, then step a paged decode entry l times over [s, s+l),
    // sharing the pool by name with the SAME executable's prefill entry. ---
    let exe_b = exec
        .load_weights(Arc::new(merged_store()), Arc::clone(&weight_map))
        .unwrap();
    let entry_b_prefill = exec.add_entry(exe_b, &staged_prefill).unwrap();
    exec.step(
        exe_b,
        entry_b_prefill,
        &step_inputs(&slots_of(&g_prefill, &feed(0, s))),
        &mut NoSync,
    )
    .unwrap()
    .read()
    .unwrap();
    exec.remove_entry(exe_b, entry_b_prefill).unwrap();
    let entry_b_decode = exec.add_entry(exe_b, &staged_decode).unwrap();
    let mut logits_decode = Vec::new();
    for i in 0..l {
        logits_decode = f32_bytes(
            &exec
                .step(
                    exe_b,
                    entry_b_decode,
                    &step_inputs(&slots_of(&g_decode, &feed(s + i, 1))),
                    &mut NoSync,
                )
                .unwrap()
                .read()
                .unwrap(),
        );
    }
    exec.remove_entry(exe_b, entry_b_decode).unwrap();
    exec.unload(exe_b).unwrap();

    let max_diff = max_abs_error(&logits_chunk, &logits_decode);
    assert!(
        max_diff < 1e-3,
        "chunk prefill at nonzero start vs stepped decode (wgpu): last-position logits differ: \
         max abs err {max_diff}"
    );
}
