//! Dev-only table-driven parity fixtures for the executor contract: one
//! table of fixture programs, run through the object-safe `Executor` on whichever `Device` the
//! caller built, with the oracle passed in by the caller rather than depended on here - this crate
//! takes `poot-models` (for the fixture graphs) but never `poot-eval`, so `poot-executor` stays free
//! of it transitively too.
//!
//! SC-002: for each fixture, every step's replayed output (the entry's first step, which records
//! then replays in one call, and every later pure-replay step) matches an independently-computed
//! CPU oracle, within float tolerance (ADR-0101 tier 1).
//!
//! The oracle is supplied by the caller as a per-step callback (`run_parity`'s `oracle` parameter):
//! this crate never depends on `poot-eval`, so the caller (`poot-gpu/src/tests/parity.rs`, which
//! already does) computes each step's expected bytes from `fixture.graph`/`fixture.store` and feeds
//! them back in the same order `run_parity` steps the device, including the final repeat of step
//! 0's inputs. Comparing every step against an independent computation (not just a later replay of
//! the same inputs against an earlier one, as this crate's previous self-consistency check did)
//! catches a `Device::replay` that silently drops the dispatch writing the primary output: both
//! reads of a never-written buffer agreed with each other but not with the oracle. Verified by
//! hand: with `WgpuDevice::replay` mutated to drop its last recorded item, the old self-consistency
//! check stayed green (false negative) while comparing against the oracle goes red.

//! Card 607 adds the device-coverage fixtures the deleted ROCm probes carried, one module per family:
//! [`routing`] (`ArgTopK` partial workgroups, tied MoE routing), [`alibi`] (ALiBi over the computed
//! slope constant), [`attention`] (per-head ALiBi flash attention, multi-head and multi-query
//! decode), [`packed`] (a packed linear per scheme and the
//! canonical expert chain, at real weight magnitude), [`capture`] (the GDN, MoE and bias toy graphs),
//! [`gdn`] (the chunked prefill) and [`launch`] (the tiled region, WMMA and a grid past 255
//! workgroups). Besides [`run_parity`] they use [`run_outputs`], for rows that assert values the oracle
//! could share a fault with, and [`run_capture_replay`], for the bit-for-bit replay check.

pub mod alibi;
pub mod attention;
pub mod bf16_cast;
pub mod capture;
pub mod dense;
pub mod gdn;
pub mod launch;
pub mod packed;
pub mod routing;
pub mod weight_map;

use std::collections::HashMap;
use std::sync::Arc;

use crate::dense::{Dense, Family, plain, step};
use poot_executor::{Executor, NoSync, StateScope, StepInputs};
use poot_graph_ir::{Graph, Slot, Storage, TensorType};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_models::model::{LogitRows, Phase};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::{DType, HostTensor};
use poot_test_util::StepFixture;

/// One parity fixture: a graph, the weight store it binds, and the input values for its first few
/// steps (enough to exercise the record-then-replay boundary at least twice).
pub struct Fixture {
    pub name: &'static str,
    pub graph: Graph,
    pub store: WeightStore,
    pub steps: Vec<Vec<StepFixture>>,
    /// The compile policy the fixture runs under: most fixtures run `Full`; a fixture for a kernel choice the
    /// `MoeHangGuard` policy changes names it.
    pub fusion: FusionPolicy,
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

fn store_for(g: &Graph) -> WeightStore {
    store_with(g, |_, _| {})
}

/// [`store_for`] with each named constant's data rewritten in place by `adjust(name, data)`.
fn store_with(g: &Graph, adjust: impl Fn(&str, &mut [f32])) -> WeightStore {
    let mut entries: HashMap<String, Vec<usize>> = HashMap::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        if meta.storage == Storage::Const {
            entries.insert(meta.name.clone().unwrap(), meta.aval.shape.clone());
        }
    }
    let mut builder = WeightStore::builder();
    for (name, shape) in entries {
        let mut data = const_data(&name, &shape);
        adjust(&name, &mut data);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense = DenseWeight::try_new(DType::F32, shape, Arc::from(bytes)).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    builder.build()
}

fn step_fixtures(g: &Graph, tokens: &[u32], start: usize) -> Vec<StepFixture> {
    let pos = start + tokens.len() - 1;
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
                // `[1,n] = [start, .., start+n-1]` for this one-shot prefill (always `start=0` at
                // every call site here, matching the mask's in-graph computation).
                Slot::Pos => {
                    let tokens_axis = *shape.last().unwrap_or(&1);
                    (0..tokens_axis).map(|i| (start + i) as f64).collect()
                }
                Slot::SeqLen => vec![(pos + 1) as f64],
                other => panic!("fixture graph has no {other:?} slot"),
            };
            let tensor = match dtype {
                DType::I32 => HostTensor::i32(shape, values.iter().map(|&v| v as i32).collect()),
                _ => HostTensor::f32(shape, values.iter().map(|&v| v as f32).collect()),
            };
            Some(StepFixture {
                key: meta.slot_key().unwrap().clone(),
                tensor,
            })
        })
        .collect()
}

/// The qwen2 dimensions of the decode and prefill fixtures.
fn fixture_dense(family: Family) -> Dense {
    Dense::new(family)
        .vocab(32)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
}

/// A qwen2 decode fixture: `cap` decode steps at increasing positions, enough to see at least one
/// recorded step followed by a replayed one with different inputs, plus a repeat of step 0's inputs
/// later to test record-vs-replay bit-exactness directly.
pub fn qwen2_decode_fixture(cap: usize) -> Fixture {
    decode_fixture(
        "qwen2_decode",
        &fixture_dense(Family::Qwen2),
        cap,
        store_for,
    )
}

/// A decode fixture of `dense` over `cap` positions: `cap` one-token steps, the store from `store`.
pub(crate) fn decode_fixture(
    name: &'static str,
    dense: &Dense,
    cap: usize,
    store: impl Fn(&Graph) -> WeightStore,
) -> Fixture {
    let m = dense.f32_model();
    let g = plain(
        m.model
            .trace(Phase::Decode, step(1, 1, cap, LogitRows::Last))
            .unwrap(),
    );
    let store = store(&g);
    let steps = (0..cap)
        .map(|pos| step_fixtures(&g, &[((pos * 7 + 3) % 32) as u32], pos))
        .collect();
    Fixture {
        name,
        graph: g,
        store,
        steps,
        fusion: FusionPolicy::Full,
    }
}

/// A prefill fixture of `dense`: one `n`-token forward over `cap` positions.
fn prefill_fixture(
    name: &'static str,
    dense: &Dense,
    n: usize,
    cap: usize,
    store: impl Fn(&Graph) -> WeightStore,
) -> Fixture {
    let m = dense.f32_model();
    let g = plain(
        m.model
            .trace(Phase::Prefill, step(1, n, cap, LogitRows::Last))
            .unwrap(),
    );
    let store = store(&g);
    let prompt: Vec<u32> = (0..n as u32).map(|t| (t * 11 + 2) % 32).collect();
    let steps = vec![step_fixtures(&g, &prompt, 0)];
    Fixture {
        name,
        graph: g,
        store,
        steps,
        fusion: FusionPolicy::Full,
    }
}

/// A qwen2 prefill fixture: one `n`-token forward.
pub fn qwen2_prefill_fixture(n: usize, cap: usize) -> Fixture {
    prefill_fixture(
        "qwen2_prefill",
        &fixture_dense(Family::Qwen2),
        n,
        cap,
        store_for,
    )
}

/// The Gemma2/Grok attention-logit softcap the softcap fixtures trace (Card 557 SC-004).
pub const ATTN_LOGIT_SOFTCAP: f32 = 50.0;

/// The dimensions of the softcap fixtures: a Gemma 2 whose window no position reaches and whose
/// final logits are not capped, so the attention-logit softcap is the only cap in the graph.
fn softcap_dense(softcap: Option<f32>) -> Dense {
    fixture_dense(Family::Gemma2)
        .with("attn_logit_softcapping", softcap)
        .with("final_logit_softcapping", serde_json::Value::Null)
        .with("sliding_window", 1 << 20)
}

/// The softcap fixtures' weights. Softmax is shift-invariant and `50 * tanh(s / 50)` is not: large Q
/// and K projections put the scores where tanh compresses them, so a dropped softcap moves the
/// softmax weights. Small scores leave the softcap the identity; either way a dropped softcap would
/// be invisible. The path from the attention output to the logits (V and O projections, the norms,
/// the LM head) is scaled so the logits are order one and the difference reaches them above the
/// parity tolerance.
fn softcap_weights(name: &str, data: &mut [f32]) {
    let gain = if name.ends_with(".attn.q") || name.ends_with(".attn.k") {
        30.0
    } else if name.contains("norm")
        || name.ends_with(".attn.v")
        || name.ends_with(".attn.o")
        || name == "w.head"
    {
        10.0
    } else {
        1.0
    };
    data.iter_mut().for_each(|value| *value *= gain);
}

/// Card 557 SC-004: [`qwen2_decode_fixture`] over a Gemma 2 body traced with `softcap` as its
/// attention-logit softcap. A softcapped decode stays the attention decomposition
/// (`FlashAttentionDecode` has no softcap), with the softcap in the graph.
pub fn qwen2_softcap_decode_fixture(cap: usize, softcap: Option<f32>) -> Fixture {
    decode_fixture("qwen2_softcap_decode", &softcap_dense(softcap), cap, |g| {
        store_with(g, softcap_weights)
    })
}

/// Card 557 SC-004: [`qwen2_prefill_fixture`] over a Gemma 2 body traced with `softcap`. `compile`
/// fuses a softcapped prefill into `FlashAttentionPrefill { softcap }`, which the planner gives the
/// imported flash prefill kernel (the generated region prefill has no softcap).
pub fn qwen2_softcap_prefill_fixture(n: usize, cap: usize, softcap: Option<f32>) -> Fixture {
    prefill_fixture(
        "qwen2_softcap_prefill",
        &softcap_dense(softcap),
        n,
        cap,
        |g| store_with(g, softcap_weights),
    )
}

/// Card 645: a projection against an F32 weight held in checkpoint `[N, K]` orientation,
/// `matmul(x, transpose(w))` with `x` an `[m, k]` activation slot. `compile` folds the pair into one
/// `DenseContraction` that reads `w` as stored, so no device `transpose` runs. Two steps with different
/// activations, so the replayed step is checked against the oracle too.
///
/// Card 1007: with `weight = DType::F16`, `w` is an F16 checkpoint weight (finite binary16 words), which
/// `compile` stores packed two elements per `u32` word for the same contraction bodies to decode.
fn checkpoint_projection_fixture(
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    weight: DType,
    fusion: FusionPolicy,
) -> Fixture {
    let b = poot_graph_ir::Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::new(vec![n, k], weight));
    let out = b.matmul(x, b.transpose(w, vec![1, 0]));
    let graph = b.finish(out);
    let key = graph.meta(x.id).slot_key().unwrap().clone();
    let steps = (0..2u64)
        .map(|step| {
            vec![StepFixture {
                key: key.clone(),
                tensor: HostTensor::f32(vec![m, k], fill(m * k, 0x5EED + step)),
            }]
        })
        .collect();
    let store = match weight {
        DType::F16 => f16_weight_store("w", vec![n, k]),
        _ => store_for(&graph),
    };
    Fixture {
        name,
        graph,
        store,
        steps,
        fusion,
    }
}

/// A store holding one F16 weight `name` of `shape`: finite binary16 words of both signs with magnitudes in
/// `[2^-8, 2^-1)`, generated as bits (so no f32 rounding step decides them).
fn f16_weight_store(name: &str, shape: Vec<usize>) -> WeightStore {
    let numel = shape.iter().product::<usize>();
    let mut s = 0x0F16_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let bytes: Vec<u8> = (0..numel)
        .flat_map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let r = (s >> 32) as u16;
            let exponent = 7 + (r >> 10) % 7;
            ((r & 0x8000) | (exponent << 10) | (r & 0x03ff)).to_le_bytes()
        })
        .collect();
    let mut builder = WeightStore::builder();
    let dense = DenseWeight::try_new(DType::F16, shape, Arc::from(bytes)).unwrap();
    builder
        .insert(name.to_string(), WeightEntry::Dense(dense))
        .unwrap();
    builder.build()
}

/// Card 645 SC-001: [`checkpoint_projection_fixture`] at the decode GEMV (`M = 1`), a few rows (`M = 4`) and a
/// prefill (`M = 33`, not a multiple of the 16-row tile), with `K = 200` (past one 128-lane workgroup and
/// not a multiple of the K-unroll) and `N = 37` (not a multiple of the 8-column tile); plus a `K` that is a
/// multiple of the K-unroll, and one prefill under `FusionPolicy::MoeHangGuard`, which keeps a matmul off the
/// generated tiled GEMM and so runs the one-thread-per-output `[N, K]` kernel.
pub fn checkpoint_projection_fixtures() -> Vec<Fixture> {
    checkpoint_projection_set(
        [
            "checkpoint_projection_m1",
            "checkpoint_projection_m4",
            "checkpoint_projection_m33",
            "checkpoint_projection_m33_k64",
            "checkpoint_projection_m33_moe_hang_guard",
        ],
        DType::F32,
    )
}

/// Card 1007: the same table over an F16 `[N, K]` weight: the decode GEMV (`M = 1`), the tiled GEMM (`M = 4`,
/// `33`, and a `K` that is a multiple of the K-unroll), and the one-thread-per-output kernel under
/// `FusionPolicy::MoeHangGuard`, each reading the weight's packed binary16 words. `N * K` is odd in the
/// `K = 200, N = 37` rows, so the last packed word carries one element and a zero pad.
pub fn checkpoint_projection_f16_fixtures() -> Vec<Fixture> {
    checkpoint_projection_set(
        [
            "checkpoint_projection_f16_m1",
            "checkpoint_projection_f16_m4",
            "checkpoint_projection_f16_m33",
            "checkpoint_projection_f16_m33_k64",
            "checkpoint_projection_f16_m33_moe_hang_guard",
        ],
        DType::F16,
    )
}

/// The shared projection table, named `[m1, m4, m33, m33_k64, m33_moe_hang_guard]`.
fn checkpoint_projection_set(names: [&'static str; 5], weight: DType) -> Vec<Fixture> {
    let [m1, m4, m33, m33_k64, moe] = names;
    vec![
        checkpoint_projection_fixture(m1, 1, 200, 37, weight, FusionPolicy::Full),
        checkpoint_projection_fixture(m4, 4, 200, 37, weight, FusionPolicy::Full),
        checkpoint_projection_fixture(m33, 33, 200, 37, weight, FusionPolicy::Full),
        checkpoint_projection_fixture(m33_k64, 33, 64, 40, weight, FusionPolicy::Full),
        checkpoint_projection_fixture(moe, 33, 200, 37, weight, FusionPolicy::MoeHangGuard),
    ]
}

/// Card 1007: `fixture`'s program, as compiled for `target`, holds exactly one `DenseContraction` and no
/// `Transpose`, so a device row shows the contraction it ran is the one the fold made, not only that the
/// output matched (an unfolded `Transpose` + `MatMul` computes the same values).
pub fn assert_one_dense_contraction(fixture: &Fixture, target: Target) {
    let ops = planned_ops(fixture, target);
    let count = |prefix: &str| ops.iter().filter(|op| op.starts_with(prefix)).count();
    assert_eq!(
        (count("dense_contraction"), count("transpose")),
        (1, 0),
        "{}: {ops:?}",
        fixture.name
    );
}

/// The names of the ops `fixture`'s program holds once compiled for `target`, in plan order: what
/// the device will be asked to run, so a row can show it exercises the op it names (a flash kernel,
/// a contraction) and not only that its output matched.
pub fn planned_ops(fixture: &Fixture, target: Target) -> Vec<String> {
    staged(fixture, target, Submission::Replay)
        .stages()
        .flat_map(|(_, _, program)| program.planned().map(|(eqn, _)| eqn.op.name()))
        .collect()
}

/// A weight store of named dense tensors, each held as its own dtype (a BF16 tensor as its stored
/// words, an I32 tensor as its words), for a fixture whose constants carry chosen data rather than
/// [`store_for`]'s hash-seeded fill.
fn store_from_tensors(entries: Vec<(&str, HostTensor)>) -> WeightStore {
    let mut builder = WeightStore::builder();
    for (name, tensor) in entries {
        let dense = DenseWeight::try_new(
            tensor.dtype(),
            tensor.shape().to_vec(),
            Arc::from(tensor.view().bytes()),
        )
        .unwrap();
        builder
            .insert(name.to_string(), WeightEntry::Dense(dense))
            .unwrap();
    }
    builder.build()
}

/// A fixture whose inputs are all named constants of chosen data: `steps` identical empty steps, so
/// the entry records on the first and replays on every later one.
fn const_fixture(
    name: &'static str,
    graph: Graph,
    consts: Vec<(&str, HostTensor)>,
    steps: usize,
) -> Fixture {
    Fixture {
        name,
        graph,
        store: store_from_tensors(consts),
        steps: (0..steps).map(|_| Vec::new()).collect(),
        fusion: FusionPolicy::Full,
    }
}

fn view(step: &[StepFixture]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for v in step {
        inputs.push(v.key.clone(), v.tensor.shape(), v.tensor.view());
    }
    inputs
}

/// Compile `fixture.graph` for `target` under `submission`, single-stage, single-device (Card
/// 546a's only shape).
pub fn staged(
    fixture: &Fixture,
    target: Target,
    submission: Submission,
) -> StagedProgram<poot_graph_ir::ValidationOutputs> {
    let g = fixture.graph.clone().with_validations(Vec::new());
    compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            execution: submission,
            fusion: fixture.fusion,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .unwrap()
}

/// `oracle` is called once per device step, in the exact order `run_parity` steps the device
/// (`fixture.steps[0]`, then `fixture.steps[1..]`, then a repeat of `fixture.steps[0]`), so a
/// stateful oracle computation can thread its own state the same way the device does. It returns
/// that step's expected output tensor.
pub type Oracle<'a> = dyn FnMut(&[StepFixture]) -> HostTensor + 'a;

/// Two F32 tensors agree in dtype and shape and within relative tolerance `rel` elementwise (CPU vs.
/// GPU float ops are not bit-identical, so an oracle comparison needs tolerance; record-then-replay
/// bit-exactness is a separate, narrower claim this function does not make). A NaN never agrees.
fn tensors_close(a: &HostTensor, b: &HostTensor, rel: f32) -> bool {
    a.dtype() == b.dtype()
        && a.shape() == b.shape()
        && match (a.to_f32(), b.to_f32()) {
            (Ok(x), Ok(y)) => x
                .iter()
                .zip(y.iter())
                .all(|(&x, &y)| (x - y).abs() <= rel * y.abs().max(1.0)),
            _ => false,
        }
}

/// One named `Storage::Const` row for [`run_once`]: the checkpoint-style name a traced graph's
/// const input declares (`meta.name`) and its typed host tensor (matching [`StepFixture`]'s
/// `Storage::Slot` row shape).
pub struct ConstFixture {
    pub name: &'static str,
    pub tensor: HostTensor,
}

/// Compiles and runs `g` once through `exec` at `target`, then tears the entry and executable back
/// down: the executor-contract replacement for the deleted `GpuExecutor::run` (Card 546b), for a
/// test that builds its own graph and inputs rather than driving one of this crate's named fixtures.
/// `consts` becomes a one-off [`WeightStore`] (built and loaded here, one entry per row); `slots`
/// becomes the step's [`StepInputs`] via [`view`]. Splitting `g`'s old `HashMap<ValueId, Value>`
/// bind into these two row sets is the only change a caller makes - the graph, its op lowering and
/// its expected output are unchanged. For example:
///
/// ```ignore
/// // before (GpuExecutor, deleted):
/// let mut gpu = poot_gpu::GpuExecutor::new()?;
/// let got = gpu.run(&g, &inputs)?; // inputs: HashMap<ValueId, poot_eval::Value>, Const rows and
///                                  // Slot rows mixed together by ValueId.
///
/// // after (the executor contract, via this crate):
/// let device = poot_gpu::device::WgpuDevice::new()?;
/// let target = device.target();
/// let mut exec: Box<dyn poot_executor::Executor> = Box::new(poot_executor::Engine::new(device));
/// let got = poot_executor_parity::run_once(&mut *exec, target, &g, &consts, &slots)?;
/// // consts: the old `inputs` rows at a `Storage::Const` id, each a `ConstFixture { name, tensor }`
/// //         (`name` is that id's `meta.name`, the same name `inputs` was keyed by).
/// // slots:  the old `inputs` rows at a `Storage::Slot` id, each a `StepFixture { key, tensor }`
/// //         (`key` is that id's `meta.slot_key()`, as `slot_step_inputs` already builds for the
/// //         production contract).
/// ```
///
/// Single step, single entry: a test that needs more than one step (a decode loop, a prefill-then-
/// decode handoff) keeps its own `add_entry`/`step` loop against `exec` directly, as [`run_parity`]
/// does, rather than calling `run_once` per step (each call tears the entry down, so carried state
/// would not survive between calls).
pub fn run_once(
    exec: &mut dyn Executor,
    target: Target,
    g: &Graph,
    consts: &[ConstFixture],
    slots: &[StepFixture],
) -> Result<HostTensor, String> {
    let mut builder = WeightStore::builder();
    for c in consts {
        let dense = DenseWeight::try_new(
            c.tensor.dtype(),
            c.tensor.shape().to_vec(),
            Arc::from(c.tensor.view().bytes()),
        )
        .map_err(|e| format!("run_once: const `{}`: {e}", c.name))?;
        builder
            .insert(c.name, WeightEntry::Dense(dense))
            .map_err(|e| format!("run_once: const `{}`: {e}", c.name))?;
    }
    let exe = exec
        .load_weights(
            Arc::new(builder.build()),
            poot_executor::WeightSource::ConstNames,
        )
        .map_err(|e| format!("run_once: load_weights: {e}"))?;
    let g = g.clone().with_validations(Vec::new());
    let staged = compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .map_err(|e| format!("run_once: compile_staged: {e}"))?;
    let entry = exec
        .add_entry(exe, &staged)
        .map_err(|e| format!("run_once: add_entry: {e}"))?;
    let output = exec
        .step(exe, entry, &view(slots), &mut NoSync)
        .map_err(|e| format!("run_once: step: {e}"))?
        .to_host()
        .map_err(|e| format!("run_once: read: {e}"))?;
    exec.remove_entry(exe, entry)
        .map_err(|e| format!("run_once: remove_entry: {e}"))?;
    exec.unload(exe)
        .map_err(|e| format!("run_once: unload: {e}"))?;
    Ok(output)
}

/// SC-002: load `fixture` onto `exec` at `target`, step through every declared step, and assert
/// that each step's replayed output matches `oracle`'s independently-computed expectation for that
/// same step, within relative tolerance - including a final repeat of step 0's inputs, so a
/// pure-replay step (no new recording) is checked against the oracle too, not only the recording
/// step (ADR-0101 tier 1).
pub fn run_parity(
    exec: &mut dyn Executor,
    target: Target,
    fixture: &Fixture,
    oracle: &mut Oracle<'_>,
) -> Result<(), String> {
    let exe = exec
        .load_weights(
            Arc::new(fixture.store.clone()),
            poot_executor::WeightSource::ConstNames,
        )
        .map_err(|e| format!("{}: load_weights: {e}", fixture.name))?;
    let entry = exec
        .add_entry(exe, &staged(fixture, target, Submission::Replay))
        .map_err(|e| format!("{}: add_entry: {e}", fixture.name))?;
    for (i, step) in fixture.steps.iter().enumerate() {
        let expected = oracle(step);
        let got = exec
            .step(exe, entry, &view(step), &mut NoSync)
            .map_err(|e| format!("{}: step {i}: {e}", fixture.name))?
            .to_host()
            .map_err(|e| format!("{}: read {i}: {e}", fixture.name))?;
        if !tensors_close(&got, &expected, 5e-3) {
            return Err(format!(
                "{}: step {i}'s replayed output does not match the oracle ({} {:?} vs {} {:?})",
                fixture.name,
                got.dtype(),
                got.shape(),
                expected.dtype(),
                expected.shape()
            ));
        }
    }
    let first = &fixture.steps[0];
    let expected = oracle(first);
    let got = exec
        .step(exe, entry, &view(first), &mut NoSync)
        .map_err(|e| format!("{}: repeat step: {e}", fixture.name))?
        .to_host()
        .map_err(|e| format!("{}: repeat read: {e}", fixture.name))?;
    if !tensors_close(&got, &expected, 5e-3) {
        return Err(format!(
            "{}: the repeated replay step does not match the oracle ({} {:?} vs {} {:?})",
            fixture.name,
            got.dtype(),
            got.shape(),
            expected.dtype(),
            expected.shape()
        ));
    }
    exec.unload(exe)
        .map_err(|e| format!("{}: unload: {e}", fixture.name))?;
    Ok(())
}

/// Load `fixture` onto `exec` at `target`, step through every declared step, and return each step's
/// output: for a row that asserts the values themselves (the ids an ArgTopK selects, the experts a
/// tied route picks) rather than only agreement with an oracle.
pub fn run_outputs(
    exec: &mut dyn Executor,
    target: Target,
    fixture: &Fixture,
) -> Result<Vec<HostTensor>, String> {
    let exe = exec
        .load_weights(
            Arc::new(fixture.store.clone()),
            poot_executor::WeightSource::ConstNames,
        )
        .map_err(|e| format!("{}: load_weights: {e}", fixture.name))?;
    let entry = exec
        .add_entry(exe, &staged(fixture, target, Submission::Replay))
        .map_err(|e| format!("{}: add_entry: {e}", fixture.name))?;
    let mut outputs = Vec::with_capacity(fixture.steps.len());
    for (i, step) in fixture.steps.iter().enumerate() {
        outputs.push(
            exec.step(exe, entry, &view(step), &mut NoSync)
                .map_err(|e| format!("{}: step {i}: {e}", fixture.name))?
                .to_host()
                .map_err(|e| format!("{}: read {i}: {e}", fixture.name))?,
        );
    }
    exec.unload(exe)
        .map_err(|e| format!("{}: unload: {e}", fixture.name))?;
    Ok(outputs)
}

/// Capture-and-replay repeatability. The entry's first step records and replays in one call; every
/// later step is a pure replay. This steps `fixture` once, comparing each step to `oracle` within
/// tolerance (as [`run_parity`] does, so a replay that drops a state commit diverges from the
/// oracle), then resets the carried state and steps the same inputs again: this second pass runs
/// every step, the first included, as a pure replay of the recording, and each step's output must
/// equal the first pass's bit for bit (the first pass's first step is the recording step's own
/// execution). A replay that reads stale or half-written state differs between the two passes.
pub fn run_capture_replay(
    exec: &mut dyn Executor,
    target: Target,
    fixture: &Fixture,
    oracle: &mut Oracle<'_>,
) -> Result<(), String> {
    let exe = exec
        .load_weights(
            Arc::new(fixture.store.clone()),
            poot_executor::WeightSource::ConstNames,
        )
        .map_err(|e| format!("{}: load_weights: {e}", fixture.name))?;
    let entry = exec
        .add_entry(exe, &staged(fixture, target, Submission::Replay))
        .map_err(|e| format!("{}: add_entry: {e}", fixture.name))?;
    let mut first_pass = Vec::with_capacity(fixture.steps.len());
    for (i, step) in fixture.steps.iter().enumerate() {
        let expected = oracle(step);
        let got = exec
            .step(exe, entry, &view(step), &mut NoSync)
            .map_err(|e| format!("{}: step {i}: {e}", fixture.name))?
            .to_host()
            .map_err(|e| format!("{}: read {i}: {e}", fixture.name))?;
        if !tensors_close(&got, &expected, 5e-3) {
            return Err(format!(
                "{}: step {i}'s output does not match the oracle ({} {:?} vs {} {:?})",
                fixture.name,
                got.dtype(),
                got.shape(),
                expected.dtype(),
                expected.shape()
            ));
        }
        first_pass.push(got);
    }
    exec.reset_state(exe, StateScope::All)
        .map_err(|e| format!("{}: reset_state: {e}", fixture.name))?;
    for (i, (step, recorded)) in fixture.steps.iter().zip(&first_pass).enumerate() {
        let replayed = exec
            .step(exe, entry, &view(step), &mut NoSync)
            .map_err(|e| format!("{}: replay step {i}: {e}", fixture.name))?
            .to_host()
            .map_err(|e| format!("{}: replay read {i}: {e}", fixture.name))?;
        let (recorded, replayed) = (recorded.view().bytes(), replayed.view().bytes());
        if recorded != replayed {
            let first = recorded
                .iter()
                .zip(replayed)
                .position(|(a, b)| a != b)
                .unwrap_or(recorded.len().min(replayed.len()));
            return Err(format!(
                "{}: replay step {i} is not bit-identical to the first pass (first differing byte {first} of {})",
                fixture.name,
                recorded.len()
            ));
        }
    }
    exec.unload(exe)
        .map_err(|e| format!("{}: unload: {e}", fixture.name))?;
    Ok(())
}
