//! Card 548 acceptance for `RocmDevice`: the ROCm/HSA implementation of the executor contract's
//! `poot_executor::Device`. SC-001/SC-005/SC-006/SC-007/SC-008/SC-010/SC-011/SC-012 (SC-002/SC-003's
//! parity-table half live in `tests/parity.rs`; SC-004, a packed E4M3-per-channel linear fixture,
//! lives in `tests/packed_serving.rs::packed_e4m3_per_channel_decode_through_the_contract_is_bit_exact`;
//! SC-009 is the required-device-lane gate, not a test here).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::StepInputs;
use poot_executor::{Arg, Device, DeviceTime, Dispatch, Engine, Executor, ExecutorStats, NoSync};
use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_graph_ir::{BinOp, Builder, Graph, StateRole, TensorType};
use poot_graph_plan::{ImportedKernel, Submission};
use poot_kernel_ir::Body;
use poot_models::model::{LogitRows, Phase};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_rocm_gpu::RocmBuffer;
use poot_rocm_gpu::device::RocmDevice;
use poot_runtime_common::{BufferRole, DeviceBackend, MemoryCounterSnapshot};
use poot_target::BufferStorage;
use poot_tensor::DType;
use poot_tensor::HostTensor;
use poot_test_util::device_skip::open_or_skip;
use poot_test_util::{assert_close_rel, max_abs_error};

const CAP: usize = 8;

fn require_rocm() -> Option<RocmDevice> {
    open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
}

const VOCAB: usize = 32;

/// The qwen2 decode graph of the fixture over `CAP` positions: Model::trace of the registry's qwen2
/// family at 16 hidden, 4 heads over 2 key-value heads, F32 consts.
fn decode_graph() -> Graph {
    let m = Dense::new(Family::Qwen2)
        .vocab(VOCAB)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
        .f32_model();
    plain(
        m.model
            .trace(Phase::Decode, step(1, 1, CAP, LogitRows::Last))
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

fn store_for(g: &Graph) -> Arc<WeightStore> {
    let mut entries: HashMap<String, Vec<usize>> = HashMap::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        if meta.storage == poot_graph_ir::Storage::Const {
            entries.insert(meta.name.clone().unwrap(), meta.aval.shape.clone());
        }
    }
    let mut builder = WeightStore::builder();
    for (name, shape) in entries {
        let n = shape.iter().product::<usize>();
        let seed = name.bytes().fold(1469598103934665603u64, |h, c| {
            (h ^ c as u64).wrapping_mul(1099511628211)
        });
        let bytes: Vec<u8> = fill(n, seed).iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense = DenseWeight::try_new(DType::F32, shape, Arc::from(bytes)).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    Arc::new(builder.build())
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn decode_staged(
    g: &Graph,
    target: poot_graph_plan::Target,
) -> poot_graph_plan::StagedProgram<poot_graph_ir::ValidationOutputs> {
    let g = g.clone().with_validations(Vec::new());
    poot_graph_plan::compile_staged(
        &g,
        &poot_graph_plan::TargetSet::single(poot_graph_plan::DeviceId(0), target),
        &poot_graph_plan::Partition {
            experts: poot_graph_plan::ExpertPlacement::AllResident,
            devices: poot_graph_plan::DevicePlacement::Single(poot_graph_plan::DeviceId(0)),
        },
        &poot_graph_plan::CompileOptions {
            execution: poot_graph_plan::Submission::Replay,
            fusion: poot_graph_plan::FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .unwrap()
}

fn decode_slot_inputs(g: &Graph, token: u32, pos: usize, _cap: usize) -> StepInputs<'static> {
    // Leak the encoded byte buffers for the step's lifetime (test-only convenience; avoids a
    // borrow-lifetime struct just to feed `StepInputs`, which borrows its byte slices).
    let mut inputs = StepInputs::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        let poot_graph_ir::Storage::Slot(slot) = meta.storage else {
            continue;
        };
        let key = meta.slot_key().unwrap().clone();
        let shape: &'static [usize] = Box::leak(meta.aval.shape.clone().into_boxed_slice());
        // Card 621/188 Inc 6c: a slot's bytes must respect its *declared* dtype, never a hardcoded
        // f32 - an I32-declared index slot (Token/Pos/SeqLen on some archs) fed f32 bits reinterprets
        // as a huge wrong index, corrupting a gather/KV-cache write (an out-of-bounds device access,
        // not a value mismatch; mirrors `poot_executor_parity::step_fixtures`'s own dtype switch).
        // Card 550: `Slot::Pos` is now `[1,1]`, not a bare scalar; the mask is a graph computation
        // over `Slot::Pos` and `iota`, so the decode graph declares no `Slot::Mask`.
        let values: Vec<f64> = match slot {
            poot_graph_ir::Slot::Token => vec![token as f64],
            poot_graph_ir::Slot::Pos => vec![pos as f64],
            poot_graph_ir::Slot::SeqLen => vec![(pos + 1) as f64],
            other => panic!("decode fixture has no {other:?} slot"),
        };
        let bytes: &'static [u8] = match meta.aval.dtype {
            DType::I32 => Box::leak(
                values
                    .iter()
                    .flat_map(|&v| (v as i32).to_le_bytes())
                    .collect::<Vec<u8>>()
                    .into_boxed_slice(),
            ),
            _ => Box::leak(
                values
                    .iter()
                    .flat_map(|&v| (v as f32).to_le_bytes())
                    .collect::<Vec<u8>>()
                    .into_boxed_slice(),
            ),
        };
        let elems = shape.iter().product();
        let view = poot_executor::HostView::new(meta.aval.dtype, elems, bytes).unwrap();
        inputs.push(key, shape, view);
    }
    inputs
}

/// SC-001: a qwen2 decode entry on `Engine<RocmDevice>` matches the CPU oracle for every position of
/// a full cache, and capture is the default: one recording after `CAP` replayed steps. Mutation:
/// reverse the recorded AQL batch's dispatch order in `RocmDevice::replay` (a stand-in for "replay
/// reuses stale/wrong packets" - the card's generic phrasing for the shared capture-once contract,
/// already proven once by Card 546a's own suite; this row demonstrates it holds through ROCm's real
/// AQL submission path specifically). Observed: reversing `dispatches` without reversing the
/// parallel `elements` schema list desynced the two, and the device returned
/// `KernelArgs(Count { expected: 3, actual: 4 })` from the very first (now out-of-order, wrongly
/// schema-checked) replayed dispatch; restoring in-order replay returned the row to green.
///
/// the row also asks for generated tokens, not just logits, compared against the
/// CPU oracle's own greedy choice (ADR-0101) - every step now feeds the ORACLE's own greedy
/// (argmax) token forward into both the device and the oracle (real autoregressive generation, same
/// schedule as `Runner::generate_kv_gpu_cached`'s production loop), and asserts the device's own
/// argmax agrees with the oracle's at every position, on top of the existing numeric closeness
/// check. There is no frozen pre-card baseline to compare bit-exactly against (548 is this backend's
/// first executor-contract implementation, not a rewrite of a prior ROCm-on-the-contract one), so
/// "bit-identical to its pre-card output" is necessarily the live CPU oracle, as every other backend
/// row in this family (`poot-gpu`'s own `qwen2_decode_gpu_matches_cpu`) already does.
#[test]
fn qwen2_decode_on_the_contract_matches_the_oracle_and_records_once() {
    let Some(device) = require_rocm() else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let g = decode_graph();
    let store = store_for(&g);
    let exe = engine
        .load_weights(store.clone(), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine.add_entry(exe, &decode_staged(&g, target)).unwrap();

    let mut state: HashMap<String, HostTensor> = HashMap::new();
    let mut device_tokens: Vec<u32> = Vec::with_capacity(CAP);
    let mut oracle_tokens: Vec<u32> = Vec::with_capacity(CAP);
    let mut token = (3u32) % VOCAB as u32;
    for pos in 0..CAP {
        let inputs = decode_slot_inputs(&g, token, pos, CAP);
        let got = bytemuck_f32(
            &engine
                .step(exe, entry, &inputs, &mut NoSync)
                .unwrap()
                .read()
                .unwrap(),
        );
        let expected = oracle_decode_step(&g, &store, &mut state, token, pos, CAP);
        assert_close_rel(&got, &expected, 1e-3);
        let device_next = argmax(&got);
        let oracle_next = argmax(&expected);
        device_tokens.push(device_next);
        oracle_tokens.push(oracle_next);
        assert_eq!(
            device_next, oracle_next,
            "SC-001: greedy token at pos {pos} must agree with the CPU oracle's argmax \
             (device_tokens so far: {device_tokens:?}, oracle_tokens so far: {oracle_tokens:?})"
        );
        token = oracle_next; // real autoregressive generation, driven by the oracle's own greedy choice
    }
    assert_eq!(
        device_tokens, oracle_tokens,
        "SC-001: the full generated token sequence must match the CPU oracle's greedy tokens"
    );
    let ExecutorStats { recordings, .. } = engine.stats();
    assert_eq!(recordings, 1, "one entry, one recording after CAP steps");
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best as u32
}

fn oracle_decode_step(
    g: &Graph,
    store: &WeightStore,
    state: &mut HashMap<String, HostTensor>,
    token: u32,
    pos: usize,
    _cap: usize,
) -> Vec<f32> {
    let mut inputs: HashMap<poot_graph_ir::ValueId, Value> = HashMap::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        let value = match meta.storage {
            poot_graph_ir::Storage::Const => {
                let name = meta.name.as_deref().unwrap();
                let WeightEntry::Dense(dense) = store.get(name).unwrap() else {
                    unreachable!()
                };
                Value::from(HostTensor::f32(
                    meta.aval.shape.clone(),
                    bytemuck_f32(dense.bytes().as_slice()),
                ))
            }
            poot_graph_ir::Storage::State => Value::from(
                state
                    .entry(meta.name.clone().unwrap())
                    .or_insert_with(|| HostTensor::zeros(meta.aval.shape.clone()))
                    .clone(),
            ),
            poot_graph_ir::Storage::Slot(slot) => {
                // Card 550: see `decode_slot_inputs`'s identical arm.
                let values: Vec<f64> = match slot {
                    poot_graph_ir::Slot::Token => vec![token as f64],
                    poot_graph_ir::Slot::Pos => vec![pos as f64],
                    poot_graph_ir::Slot::SeqLen => vec![(pos + 1) as f64],
                    other => panic!("no {other:?} slot"),
                };
                match meta.aval.dtype {
                    DType::I32 => Value::from(HostTensor::i32(
                        meta.aval.shape.clone(),
                        values.iter().map(|&v| v as i32).collect(),
                    )),
                    _ => Value::from(HostTensor::f32(
                        meta.aval.shape.clone(),
                        values.iter().map(|&v| v as f32).collect(),
                    )),
                }
            }
            poot_graph_ir::Storage::Computed(computed) => {
                Value::from(HostTensor::f32(computed.shape(), computed.values_f32()))
            }
            other => panic!("decode fixture has no {other:?} input kind"),
        };
        inputs.insert(id, value);
    }
    let evaluation = eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
    for (&(si, _), t) in g.state.iter().zip(evaluation.state) {
        state.insert(g.meta(si).name.clone().unwrap(), t.into_host().unwrap());
    }
    evaluation
        .output
        .into_host()
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec()
}

/// SC-005: ROCm profiling of a fixture reports `Unknown` device time, never host wall. Mutation:
/// have `RocmDevice::device_time` return `DeviceTime::Measured` with a wall-clock span (restoring
/// the pre-contract behavior `record_dispatch(name, wall, Some(wall))` once reported); the fixture
/// then reports a number and this row goes red.
#[test]
fn rocm_device_time_is_always_unknown() {
    let Some(mut device) = require_rocm() else {
        return;
    };
    device.begin(poot_graph_plan::Submission::Replay).unwrap();
    let recording = device.finish().unwrap().expect("Replay always records");
    device.replay(&recording).unwrap();
    device.synchronize().unwrap();
    assert_eq!(
        device.device_time(),
        DeviceTime::Unknown,
        "ROCm has no HSA timestamp binding yet (Card 552's SC-001, moved here)"
    );
}

/// SC-006: the same fixture reads equal bytes by role on the wgpu and ROCm runtime counters (moved
/// from Card 547a, R-546-7). The row's actual claim is cross-backend agreement,
/// not just "ROCm's own Weight/State roles are self-consistently nonzero" - this now opens a wgpu
/// `Engine<WgpuDevice>` on the identical graph/store and compares `Weight`/`State` live-byte counts
/// per role between the two backends, not just within ROCm. Mutation: drop the `Weight` role's
/// increment in `RocmContext::allocate` (charge every allocation as `Activation` instead); ROCm's
/// `Weight` row then reports 0 live bytes while wgpu's still reports the real byte count, and this
/// row goes red on the cross-backend comparison (not just the `> 0` check).
#[test]
fn memory_counters_charge_the_roles_a_decode_entry_actually_allocates() {
    let Some(device) = require_rocm() else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let g = decode_graph();
    let store = store_for(&g);
    let exe = engine
        .load_weights(Arc::clone(&store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let _entry = engine.add_entry(exe, &decode_staged(&g, target)).unwrap();
    let memory = engine.stats().memory;
    let rocm_weight: u64 = memory
        .iter()
        .find(|(role, _)| *role == BufferRole::Weight)
        .map(|(_, s): &(BufferRole, MemoryCounterSnapshot)| s.live_bytes)
        .unwrap_or(0);
    assert!(
        rocm_weight > 0,
        "a decode entry with named consts must charge the Weight role: {memory:?}"
    );
    let rocm_state: u64 = memory
        .iter()
        .find(|(role, _)| *role == BufferRole::State)
        .map(|(_, s)| s.live_bytes)
        .unwrap_or(0);
    assert!(
        rocm_state > 0,
        "a KV-cache decode entry must charge the State role: {memory:?}"
    );

    let Some(wgpu_device) = open_or_skip(DeviceBackend::Wgpu, poot_gpu::device::WgpuDevice::new())
    else {
        return;
    };
    let wgpu_target = wgpu_device.target();
    let mut wgpu_engine = Engine::new(wgpu_device);
    let wgpu_exe = wgpu_engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let _wgpu_entry = wgpu_engine
        .add_entry(wgpu_exe, &decode_staged(&g, wgpu_target))
        .unwrap();
    let wgpu_memory = wgpu_engine.stats().memory;
    for role in [BufferRole::Weight, BufferRole::State] {
        let rocm_bytes = memory
            .iter()
            .find(|(r, _)| *r == role)
            .map(|(_, s)| s.live_bytes)
            .unwrap_or(0);
        let wgpu_bytes = wgpu_memory
            .iter()
            .find(|(r, _)| *r == role)
            .map(|(_, s)| s.live_bytes)
            .unwrap_or(0);
        assert_eq!(
            rocm_bytes, wgpu_bytes,
            "SC-006: {role:?} must charge the same live bytes on ROCm and wgpu for the identical \
             decode entry: rocm={memory:?} wgpu={wgpu_memory:?}"
        );
    }
}

/// SC-007: with a 600s typed `wait_timeout` (the default) and a normal step, the step succeeds -
/// the env var this replaces is read nowhere any more. Mutation: reintroduce
/// `std::env::var("POOT_GPU_WAIT_TIMEOUT_SECS")` inside `wait_completion_bounded`/`synchronize`/
/// `wait_for_space` with `POOT_GPU_WAIT_TIMEOUT_SECS=0` set in the environment; the step then times
/// out immediately and this row goes red.
#[test]
fn wait_timeout_is_typed_construction_state_not_an_env_read() {
    // SAFETY: nextest runs ROCm device tests with `--test-threads=1` (AGENTS.md); no concurrent env
    // access in this process during this test.
    unsafe {
        std::env::set_var("POOT_GPU_WAIT_TIMEOUT_SECS", "0");
    }
    let result = (|| {
        let device = require_rocm()?;
        let target = device.target();
        let mut engine = Engine::new(device);
        let g = decode_graph();
        let store = store_for(&g);
        let exe = engine
            .load_weights(store, poot_executor::WeightSource::ConstNames)
            .unwrap();
        let entry = engine.add_entry(exe, &decode_staged(&g, target)).unwrap();
        let inputs = decode_slot_inputs(&g, 0, 0, CAP);
        Some(engine.step(exe, entry, &inputs, &mut NoSync).is_ok())
    })();
    // SAFETY: see above.
    unsafe {
        std::env::remove_var("POOT_GPU_WAIT_TIMEOUT_SECS");
    }
    if let Some(ok) = result {
        assert!(
            ok,
            "POOT_GPU_WAIT_TIMEOUT_SECS=0 must have no effect: the bound is RocmContextOptions::wait_timeout (600s default), read nowhere from the environment"
        );
    }
}

/// SC-008: a two-phase state commit (a swap-plus-increment with no in-place-donatable state, mirrors
/// `poot-gpu`'s own fixture) matches the oracle on ROCm through the generated copy kernel. Mutation:
/// drop the `device.copy(at(entry.output), at(preserve))` snapshot call in `Engine::walk` (shared
/// code, already proven by Card 546a) - for this row's ROCm-specific teeth, the mutation applied was
/// `RocmDevice::copy` returning `Ok(())` without dispatching the copy kernel (the commit silently
/// no-ops): both states and the output then stayed zero forever instead of growing, and restoring
/// the real dispatch returned the row to green.
#[test]
fn two_phase_state_commit_matches_the_oracle_after_four_steps_on_rocm() {
    let Some(device) = require_rocm() else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);

    let b = Builder::new();
    let a = b.state_input("a", TensorType::f32(vec![4]), StateRole::Recurrent);
    let s = b.state_input("s", TensorType::f32(vec![4]), StateRole::Recurrent);
    let bias = b.constant("bias", TensorType::f32(vec![4]));
    let output = b.binary(BinOp::Add, a, s);
    let new_a = b.binary(BinOp::Add, s, bias);
    let new_s = b.binary(BinOp::Add, a, bias);
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
    let entry = engine.add_entry(exe, &decode_staged(&g, target)).unwrap();

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
}

/// Card 548: `Device::copy` dispatches the generated `copy_words` kernel, whose
/// `&[u32]` parameters take a WORD count, not the buffer's logical element count - the two disagree
/// for any element narrower than 4 bytes (bf16/f16: 2 bytes, elem_count is 2x words; i8/raw bytes: 1
/// byte, elem_count is 4x words; `poot-target::ElementKind` has nothing wider than 4 bytes today, so
/// there is no over-4-byte case to drive). No graph in this crate ever copies a <4-byte
/// state/output buffer (every carried state here is F32), so this is the first test to drive
/// `Device::copy` on one. `WORDS` is deliberately not a multiple of the copy kernel's workgroup size
/// (64), so `padded_launch` really does add extra padding threads for every format here (`words %
/// 64 != 0`); `dst_sentinel`, a same-size buffer allocated right after `dst`, is written with a
/// known pattern and re-read after the copy, to catch those padding threads landing past `dst`'s
/// real extent.
///
/// This test does NOT prove the length is correct (it cannot be made to - see
/// below) - it is real, additional coverage of `Device::copy` round-tripping every byte on every
/// narrow dtype, which nothing exercised before this card. The actual length proof is
/// `copy_word_count`'s own unit test in `device.rs` (`copy_word_count_is_the_byte_capacity_never_
/// the_element_count`): a pure function, mutation-tested there with no device at all. Tried and
/// reverted here (not committed): `build_kernarg`'s per-call-site explicit length back to
/// `buf.elem_count()` for both slots. On real gfx1151 hardware this did NOT corrupt `dst_sentinel`
/// or crash: the bug's maximum overrun is bounded by one workgroup (<=63 words = <=252 bytes), and
/// `BufferRole::Activation`'s coarse-grained pool allocates in page granularity, so that overrun
/// stays inside `dst`'s own page's slack in practice (observed: unmutated and mutated runs both pass
/// byte-for-byte) - exactly why the length needs its own pure-function unit test instead.
#[test]
fn copy_round_trips_every_narrow_element_width_with_no_sentinel_corruption() {
    let Some(mut device) = require_rocm() else {
        return;
    };
    const WORDS: usize = 100; // 100 % 64 == 36, so padded_launch adds 28 padding threads
    for (name, storage, bytes_per_elem) in [
        ("bf16", BufferStorage::bf16(), 2usize),
        ("f16", BufferStorage::f16(), 2usize),
        ("raw_bytes (i8)", BufferStorage::raw_bytes(), 1usize),
    ] {
        let elems = WORDS * 4 / bytes_per_elem;
        let src_data: Vec<u8> = (0..elems * bytes_per_elem)
            .map(|i| (i as u64).wrapping_mul(0x9E37_79B9) as u8)
            .collect();
        let sentinel_pattern = vec![0xABu8; elems * bytes_per_elem];
        let src = device
            .allocate(BufferRole::Activation, storage, elems)
            .unwrap_or_else(|e| panic!("{name}: allocate src: {e}"));
        let dst = device
            .allocate(BufferRole::Activation, storage, elems)
            .unwrap_or_else(|e| panic!("{name}: allocate dst: {e}"));
        let dst_sentinel = device
            .allocate(BufferRole::Activation, storage, elems)
            .unwrap_or_else(|e| panic!("{name}: allocate dst_sentinel: {e}"));
        device
            .write(&src, &src_data)
            .unwrap_or_else(|e| panic!("{name}: write src: {e}"));
        device
            .write(&dst_sentinel, &sentinel_pattern)
            .unwrap_or_else(|e| panic!("{name}: write dst_sentinel: {e}"));
        device.begin(Submission::Replay).unwrap();
        device
            .copy(&src, &dst)
            .unwrap_or_else(|e| panic!("{name}: copy: {e}"));
        let recording = device.finish().unwrap().expect("Replay always records");
        device
            .replay(&recording)
            .unwrap_or_else(|e| panic!("{name}: replay: {e}"));
        device.synchronize().unwrap();
        let mut out = vec![0u8; elems * bytes_per_elem];
        device
            .read(&dst, &mut out)
            .unwrap_or_else(|e| panic!("{name}: read dst: {e}"));
        assert_eq!(
            out, src_data,
            "{name}: copy must round-trip every real byte"
        );
        let mut sentinel_out = vec![0u8; elems * bytes_per_elem];
        device
            .read(&dst_sentinel, &mut sentinel_out)
            .unwrap_or_else(|e| panic!("{name}: read dst_sentinel: {e}"));
        assert_eq!(
            sentinel_out, sentinel_pattern,
            "{name}: copy must not touch the next allocation's bytes"
        );
    }
}

/// SC-010/SC-011/SC-012: a real dispatch forced to time out (tiny `wait_timeout`) through
/// `RocmDevice`'s production methods leaves the device poisoned, its data allocations retained
/// (not released) with their role-counter bytes still charged, and its code object alive - exactly
/// the Card 602 `TimeoutLeaks`/`DevicePoison` contract proven here through the new `Device` trait's
/// own entry points (`begin`/`dispatch`/`finish`/`replay`/`synchronize`), not a raw
/// `RocmContext`/`HsacoModule` call. Mutation: skip `self.timeouts.poison()` in
/// `wait_completion_bounded`'s timeout branch (the mutation `poot-rocm-runtime`'s own
/// `submissions_after_a_timed_out_completion_wait_are_refused` already proves red/green for); this
/// row's own evidence is specific to driving that path through `RocmDevice::replay`/`synchronize`.
#[test]
fn a_timed_out_dispatch_through_the_device_trait_retains_its_allocations_and_poisons() {
    let Some(ctx) = open_or_skip(
        DeviceBackend::Rocm,
        poot_rocm_runtime::RocmContext::new_with_options(poot_rocm_runtime::RocmContextOptions {
            wait_timeout: Duration::from_secs(2),
        }),
    ) else {
        return;
    };
    let mut device = RocmDevice::from_context(ctx).expect("RocmDevice over the 2s-bound context");
    let target = device.target();
    let arch = match target.backend {
        poot_target::Backend::AmdGcn(arch) => arch,
        other => panic!("RocmDevice::target must report AmdGcn, got {other:?}"),
    };

    // A real dispatch, deliberately sized to run far longer than the 2s bound (a finite busy-spin,
    // not a fake/forged state): the production path proves the timeout for real, on real hardware,
    // through `RocmDevice`'s own `Device` methods - never a private `RocmContext`/queue internal.
    let codegen_target = poot_codegen::Target::AmdGcn(arch);
    let body: Body = ImportedKernel::SpinBusy.body().clone();
    let tmp = poot_codegen::kernel_cache_root("rocm-sc011", codegen_target);
    let out_path = poot_codegen::artifact_path(&tmp, "spin_busy_sc011", codegen_target);
    poot_codegen::compile(&body, codegen_target, &out_path).expect("spin_busy must compile");
    let bytes = std::fs::read(&out_path).unwrap();
    let compiled = poot_codegen::kernel_handle(&body, codegen_target, bytes);
    let kernel = device
        .load_kernel("spin_busy_sc011", compiled)
        .expect("load the spin_busy kernel");

    let n_buf = device
        .allocate(BufferRole::Activation, BufferStorage::i32(), 1)
        .unwrap();
    // 4e9 dependent-chain iterations: far beyond the 2s bound on any GPU this box runs, while still
    // finite (the dispatch really does complete, eventually, if left alone).
    device
        .write(&n_buf, &4_000_000_000u32.to_le_bytes())
        .unwrap();
    let out_buf: RocmBuffer = device
        .allocate(BufferRole::Activation, BufferStorage::f32(), 1)
        .unwrap();
    let live_before = device
        .memory()
        .into_iter()
        .find(|(role, _)| *role == BufferRole::Activation)
        .map(|(_, s)| s.live_bytes)
        .unwrap_or(0);
    assert!(live_before > 0, "the probe output buffer must be charged");

    device.begin(Submission::Replay).unwrap();
    device
        .dispatch(Dispatch {
            kernel: &kernel,
            inputs: &[Arg {
                buffer: &n_buf,
                elems: 1,
            }],
            output: Arg {
                buffer: &out_buf,
                elems: 1,
            },
            threads: [1, 1, 1],
            workgroup: [1, 1, 1],
            work: 1,
        })
        .unwrap();
    let recording = device.finish().unwrap().expect("Replay always records");
    let t0 = std::time::Instant::now();
    let err = device
        .replay(&recording)
        .expect_err("the 4e9-iteration spin must outlast the 2s wait_timeout bound");
    eprintln!(
        "a_timed_out_dispatch_through_the_device_trait_retains_its_allocations_and_poisons: \
         observed {err} after {:?}",
        t0.elapsed()
    );
    device.abort().expect_err(
        "a poisoned RocmContext cannot establish a reusable state; abort must report it so Engine poisons too",
    );
    assert!(
        device.context().is_poisoned(),
        "the context must be poisoned after the completion wait times out"
    );

    drop(out_buf);
    let live_after = device
        .memory()
        .into_iter()
        .find(|(role, _)| *role == BufferRole::Activation)
        .map(|(_, s)| s.live_bytes)
        .unwrap_or(0);
    assert_eq!(
        live_after, live_before,
        "SC-011: the probe buffer's allocation must stay charged after its last handle drops, \
         because the device never proved the hung submission's completion"
    );
}

/// SC-012 (the ordinary-path counterpart of SC-011): a step that actually completes releases its
/// locals' role-counter bytes once `remove_entry`/`unload` drops them - completion was proven
/// synchronously (ROCm's `replay` blocks on the real completion signal), so nothing stays charged.
/// Mutation: in `RocmAlloc::drop` (poot-rocm-runtime), leak unconditionally instead of only when
/// `self.poison.is_poisoned()`; this row's allocation then stays charged after `unload` and goes red.
#[test]
fn an_ordinary_unload_with_completion_proven_releases_its_allocations() {
    let Some(device) = require_rocm() else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let g = decode_graph();
    let store = store_for(&g);
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine.add_entry(exe, &decode_staged(&g, target)).unwrap();
    let inputs = decode_slot_inputs(&g, 0, 0, CAP);
    engine
        .step(exe, entry, &inputs, &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    let live_loaded = engine.stats().memory;
    assert!(
        live_loaded.iter().any(|(_, s)| s.live_bytes > 0),
        "a stepped entry must have live allocations: {live_loaded:?}"
    );
    engine.unload(exe).unwrap();
    let live_unloaded = engine.stats().memory;
    for (role, snapshot) in &live_unloaded {
        assert_eq!(
            snapshot.live_bytes, 0,
            "SC-012: role {role:?} must read 0 live bytes once the only executable unloads \
             after a completed (non-poisoned) step: {live_unloaded:?}"
        );
    }
}

/// Card 548 SC-009: the I32-index capability (spike 562 F-9: a `Slot::Pos` that
/// is both a `Gather` index and feeds another use) does exist on ROCm - `poot-graph-plan`'s planner
/// has no backend-specific admission for it, and `poot_test_util::i32_slot_gather`'s shared harness
/// is backend-generic (wgpu's own row, `poot-gpu/tests/graph/gather_scatter.rs`, drives the same
/// three variants through `run_resident_kv`). The two state-carrying variants (v1/v2) need a
/// non-zero initial cache; the executor contract has no explicit "seed state" entry point (state
/// always starts zero-seeded within an executable), so this primes each one
/// through a one-shot entry on the same executable that writes the exact seed bytes into that named
/// state before the real variant's entry reads it - the same "prefill writes, decode reads" sharing
/// by (name, aval, storage) every other state-carrying ROCm test here already relies on, not a new
/// escape hatch.
#[test]
fn i32_slot_gather_variants_compiled_match_cpu_oracle_on_rocm() {
    let Some(device) = require_rocm() else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    poot_test_util::i32_slot_gather::run_all("rocm", |variant| {
        let base_inputs = variant.inputs(&variant.graph);
        let mut builder = WeightStore::builder();
        for &id in &variant.graph.inputs {
            let meta = variant.graph.meta(id);
            if meta.storage != poot_graph_ir::Storage::Const {
                continue;
            }
            let name = meta.name.as_deref().unwrap().to_string();
            let tensor = base_inputs[&id]
                .clone()
                .into_host()
                .map_err(|e| format!("{name}: {e}"))?;
            let bytes: Vec<u8> = tensor
                .as_f32()
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            builder
                .insert(
                    name.as_str(),
                    WeightEntry::Dense(
                        DenseWeight::try_new(DType::F32, tensor.shape().to_vec(), Arc::from(bytes))
                            .map_err(|e| format!("{name}: {e:?}"))?,
                    ),
                )
                .map_err(|e| format!("{name}: {e:?}"))?;
        }
        let state_seeds = variant.state();
        for (i, seed) in state_seeds.iter().enumerate() {
            let (si, _) = variant.graph.state[i];
            let shape = variant.graph.meta(si).aval.shape.clone();
            let bytes: Vec<u8> = seed.iter().flat_map(|v| v.to_le_bytes()).collect();
            builder
                .insert(
                    format!("seed_{i}").as_str(),
                    WeightEntry::Dense(
                        DenseWeight::try_new(DType::F32, shape, Arc::from(bytes))
                            .map_err(|e| format!("seed_{i}: {e:?}"))?,
                    ),
                )
                .map_err(|e| format!("seed_{i}: {e:?}"))?;
        }
        let store = Arc::new(builder.build());
        let exe = engine
            .load_weights(store, poot_executor::WeightSource::ConstNames)
            .map_err(|e| format!("load_weights: {e}"))?;

        for (i, &(si, _)) in variant.graph.state.iter().enumerate() {
            let meta = variant.graph.meta(si);
            let name = meta.name.as_deref().unwrap().to_string();
            let shape = meta.aval.shape.clone();
            let b = Builder::new();
            let state_in =
                b.state_input(&name, TensorType::f32(shape.clone()), StateRole::Recurrent);
            let seed_const = b.constant(&format!("seed_{i}"), TensorType::f32(shape));
            let seed_g = b.finish_with_state(seed_const, &[(state_in, seed_const)]);
            let seed_entry = engine
                .add_entry(exe, &decode_staged(&seed_g, target))
                .map_err(|e| format!("seed {name}: add_entry: {e}"))?;
            engine
                .step(exe, seed_entry, &StepInputs::new(), &mut NoSync)
                .map_err(|e| format!("seed {name}: step: {e}"))?
                .read()
                .map_err(|e| format!("seed {name}: read: {e}"))?;
            engine
                .remove_entry(exe, seed_entry)
                .map_err(|e| format!("seed {name}: remove_entry: {e}"))?;
        }

        let mut slot_inputs = StepInputs::new();
        for &id in &variant.graph.inputs {
            let meta = variant.graph.meta(id);
            if !matches!(meta.storage, poot_graph_ir::Storage::Slot(_)) {
                continue;
            }
            let key = meta.slot_key().unwrap().clone();
            let shape: &'static [usize] = Box::leak(meta.aval.shape.clone().into_boxed_slice());
            let tensor = base_inputs[&id]
                .clone()
                .into_host()
                .map_err(|e| format!("slot {key:?}: {e}"))?;
            let bytes: &'static [u8] = match meta.aval.dtype {
                DType::I32 => Box::leak(
                    tensor
                        .as_i32()
                        .expect("I32-declared slot carries I32 words")
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>()
                        .into_boxed_slice(),
                ),
                _ => Box::leak(
                    tensor
                        .as_f32()
                        .unwrap()
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>()
                        .into_boxed_slice(),
                ),
            };
            let elems = shape.iter().product();
            let view = poot_executor::HostView::new(meta.aval.dtype, elems, bytes).unwrap();
            slot_inputs.push(key, shape, view);
        }

        let real_entry = engine
            .add_entry(exe, &decode_staged(&variant.graph, target))
            .map_err(|e| format!("add_entry: {e}"))?;
        let bytes = engine
            .step(exe, real_entry, &slot_inputs, &mut NoSync)
            .map_err(|e| format!("step: {e}"))?
            .read()
            .map_err(|e| format!("read: {e}"))?;
        let out_shape = variant.graph.aval(variant.graph.output).shape.clone();
        engine.unload(exe).map_err(|e| format!("unload: {e}"))?;
        Ok(HostTensor::f32(out_shape, bytemuck_f32(&bytes)))
    });
}

/// Card 550 SC-003: the wgpu-side `sliding_window_mask_from_pos_gpu_matches_cpu`
/// (`poot-gpu/tests/stress.rs`)'s fixture replayed on `Engine<RocmDevice>`: a family-neutral
/// attention graph whose mask is `causal_mask_from_pos` with a real window (`w < l`), fed
/// `Slot::Pos` as a step input through the one executor contract, matching the CPU oracle.
#[test]
fn sliding_window_mask_from_pos_rocm_matches_cpu() {
    let Some(device) = require_rocm() else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);

    let (hq, hkv, l, window) = (2usize, 1usize, 6usize, 3usize);
    let n_rep = hq / hkv;
    let d = 4usize;
    let scale = 1.0 / (d as f32).sqrt();
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
    let pos = b.slot(
        poot_graph_ir::Slot::Pos,
        TensorType::new(vec![1, l], DType::I32),
    );
    let mask = poot_graph_ir::ops::causal_mask_from_pos(&b, pos, l, Some(window));
    let out = poot_graph_ir::ops::attention_prefill(&b, q, k, v, n_rep, scale, mask);
    let (qi, ki, vi, posi) = (q.id, k.id, v.id, pos.id);
    let g = b.finish(out);

    let fill = |seed: u64, count: usize| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    };
    let qd = fill(20001, hq * l * d);
    let kd = fill(20002, hkv * l * d);
    let vd = fill(20003, hkv * l * d);
    let posd: Vec<i32> = (0..l as i32).collect();

    let mut const_inputs: HashMap<usize, Value> = HashMap::new();
    const_inputs.insert(qi, HostTensor::f32(vec![1, hq, l, d], qd.clone()).into());
    const_inputs.insert(ki, HostTensor::f32(vec![1, hkv, l, d], kd.clone()).into());
    const_inputs.insert(vi, HostTensor::f32(vec![1, hkv, l, d], vd.clone()).into());
    const_inputs.insert(posi, HostTensor::i32(vec![1, l], posd.clone()).into());
    let cpu = eval(&g, &const_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let mut builder = WeightStore::builder();
    for (name, shape, data) in [
        ("q", vec![1, hq, l, d], &qd),
        ("k", vec![1, hkv, l, d], &kd),
        ("v", vec![1, hkv, l, d], &vd),
    ] {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        builder
            .insert(
                name,
                WeightEntry::Dense(
                    DenseWeight::try_new(DType::F32, shape, Arc::from(bytes)).unwrap(),
                ),
            )
            .unwrap();
    }
    let store = Arc::new(builder.build());
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine.add_entry(exe, &decode_staged(&g, target)).unwrap();

    let pos_key = g.meta(posi).slot_key().unwrap().clone();
    let pos_shape = [1usize, l];
    let run_at = |engine: &mut Engine<RocmDevice>, positions: &[i32]| -> Vec<f32> {
        let pos_bytes: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut step_in = StepInputs::new();
        step_in.push(
            pos_key.clone(),
            &pos_shape,
            poot_executor::HostView::new(DType::I32, l, &pos_bytes).unwrap(),
        );
        let bytes = engine
            .step(exe, entry, &step_in, &mut NoSync)
            .unwrap()
            .read()
            .unwrap();
        bytemuck_f32(&bytes)
    };
    let got = run_at(&mut engine, &posd);

    // Sensitivity half of the mutation note (see the wgpu sibling test): the mask must actually come
    // from `Slot::Pos`. Collapse every row to position 2 and confirm the output moves.
    let shifted: Vec<i32> = vec![2; l];
    let got_shifted = run_at(&mut engine, &shifted);
    let cpu_shifted = {
        let mut shifted_inputs = const_inputs.clone();
        shifted_inputs.insert(posi, HostTensor::i32(vec![1, l], shifted).into());
        eval(&g, &shifted_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    engine.unload(exe).unwrap();

    poot_test_util::assert_close_rel(&got, cpu.as_f32().unwrap(), 1e-4);
    poot_test_util::assert_close_rel(&got_shifted, cpu_shifted.as_f32().unwrap(), 1e-4);
    assert!(
        max_abs_error(&got, &got_shifted) > 1e-3,
        "the fixture's output did not change when Slot::Pos changed; its mask may not be reading Pos"
    );
}
