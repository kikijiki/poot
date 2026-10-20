//! Dev-only table-driven parity fixtures for the executor contract: one
//! table of fixture programs, run through the object-safe `Executor` on whichever `Device` the
//! caller built, with the oracle passed in by the caller rather than depended on here. The crate
//! sits below `poot-models` (the fixtures that trace a model live in `poot_models::test_support`)
//! and takes no model; `poot-executor` stays free of the oracle transitively too.
//!
//! SC-002: for each fixture, every step's replayed output (the entry's first step, which records
//! then replays in one call, and every later pure-replay step) matches an independently-computed
//! CPU oracle, within float tolerance (ADR-0101 tier 1).
//!
//! The oracle is supplied by the caller as a per-step callback (`run_parity`'s `oracle` parameter):
//! this crate never depends on `poot-eval`, so the caller (`poot-wgpu-device/src/tests/parity.rs`, which
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
pub mod block_top_k;
pub mod capture;
pub mod copy_probe;
pub mod coresident;
pub mod device_sampling;
pub mod dispatch_length;
pub mod echo;
pub mod gdn;
pub mod i32_gather;
pub mod launch;
pub mod live_attention;
pub mod lora;
pub mod packed;
pub mod routing;
pub mod sample_token;
pub mod scan;
pub mod select;
pub mod selected_attention;
pub mod state_row;
pub mod strided_read;
pub mod write_probe;

use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::test_support::StepFixture;
use poot_executor::{Executor, NoSync, StateScope, StepInputs};
use poot_graph_ir::analysis::Tier2Class;
use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_graph_ir::{ForwardSlotMap, Graph, Slot, Storage, TensorType};
use poot_graph_plan::{
    CompileOptions, Target, multi_device::topology::DeviceId, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::StagedProgram, staged::TargetSet,
    staged::compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::{DType, HostTensor};

/// One parity fixture: a graph, the weight store it binds, and the input values for its first few
/// steps (enough to exercise the record-then-replay boundary at least twice).
pub struct Fixture {
    /// Fixture name, used in assertion messages and to pick a row out of a table.
    pub name: &'static str,
    /// The traced program the executor compiles.
    pub graph: Graph,
    /// The weight store binding the graph's `Storage::Const` inputs by name.
    pub store: WeightStore,
    /// Per-step input rows: `steps[i]` binds the slot inputs of step `i`, in step order.
    pub steps: Vec<Vec<StepFixture>>,
}

/// The magnitude of the default fixture weights and activations. Small enough that a toy model stays out
/// of saturation, large enough that its outputs sit well above the signal floor of their class
/// ([`signal_floor`]), so a wrong device cannot hide inside the tolerance.
const FILL_SCALE: f32 = 0.25;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * FILL_SCALE
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

/// The deterministic weight store for `g`: one dense F32 entry per named `Storage::Const` input, filled by a fixed seed derived from the constant's name (the causal mask is generated, not random).
pub fn store_for(g: &Graph) -> WeightStore {
    store_with(g, |_, _| {})
}

/// [`store_for`] with each named constant's data rewritten in place by `adjust(name, data)`.
pub fn store_with(g: &Graph, adjust: impl Fn(&str, &mut [f32])) -> WeightStore {
    let mut entries: HashMap<String, Vec<usize>> = HashMap::new();
    for &id in g.inputs() {
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
        let dense = DenseWeight::try_new(DType::F32, shape, StoredBytes::from(bytes)).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    builder.build()
}

/// The slot inputs for one step of `g`: `Token` rows from `tokens` and `Pos` rows counting up from `start`. Panics on any other slot kind.
pub fn step_fixtures(g: &Graph, tokens: &[u32], start: usize) -> Vec<StepFixture> {
    g.inputs()
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
) -> Fixture {
    let b = poot_graph_ir::Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![m, k]))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![n, k], weight))
        .unwrap();
    let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    let graph = b.finish(out).unwrap();
    let key = graph.meta(x.id()).slot_key().unwrap().clone();
    let steps = (0..2u64)
        .map(|step| {
            vec![StepFixture {
                key: key.clone(),
                tensor: HostTensor::f32(vec![m, k], fill(m * k, 0x5EED + step)),
            }]
        })
        .collect();
    let store = match weight {
        DType::F16 | DType::BF16 => narrow_weight_store("w", vec![n, k], weight),
        _ => store_for(&graph),
    };
    Fixture {
        name,
        graph,
        store,
        steps,
    }
}

/// A store holding one 2-byte float weight `name` of `shape` and `dtype` (F16 or BF16): finite words of both
/// signs with magnitudes in `[2^-8, 2^-1)`, generated as bits (so no f32 rounding step decides them).
fn narrow_weight_store(name: &str, shape: Vec<usize>, dtype: DType) -> WeightStore {
    let numel = shape.iter().product::<usize>();
    let mut s = 0x0F16_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let bytes: Vec<u8> = (0..numel)
        .flat_map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let r = (s >> 32) as u16;
            let word = match dtype {
                DType::F16 => {
                    let exponent = 7 + (r >> 10) % 7;
                    (r & 0x8000) | (exponent << 10) | (r & 0x03ff)
                }
                DType::BF16 => {
                    let exponent = 119 + (r >> 7) % 7;
                    (r & 0x8000) | (exponent << 7) | (r & 0x007f)
                }
                other => unreachable!("not a 2-byte float weight: {other:?}"),
            };
            word.to_le_bytes()
        })
        .collect();
    let mut builder = WeightStore::builder();
    let dense = DenseWeight::try_new(dtype, shape, StoredBytes::from(bytes)).unwrap();
    builder
        .insert(name.to_string(), WeightEntry::Dense(dense))
        .unwrap();
    builder.build()
}

/// Card 645 SC-001: [`checkpoint_projection_fixture`] at the decode GEMV (`M = 1`), a few rows (`M = 4`) and a
/// prefill (`M = 33`, not a multiple of the 16-row tile), with `K = 200` (past one 128-lane workgroup and
/// not a multiple of the K-unroll) and `N = 37` (not a multiple of the 8-column tile); plus a `K` that is a
/// multiple of the K-unroll.
pub fn checkpoint_projection_fixtures() -> Vec<Fixture> {
    checkpoint_projection_set(
        [
            "checkpoint_projection_m1",
            "checkpoint_projection_m4",
            "checkpoint_projection_m33",
            "checkpoint_projection_m33_k64",
        ],
        DType::F32,
    )
}

/// Card 1007: the same table over an F16 `[N, K]` weight: the decode GEMV (`M = 1`), the tiled GEMM (`M = 4`,
/// `33`, and a `K` that is a multiple of the K-unroll), each reading the weight's packed binary16 words. `N * K` is odd in the
/// `K = 200, N = 37` rows, so the last packed word carries one element and a zero pad.
pub fn checkpoint_projection_f16_fixtures() -> Vec<Fixture> {
    checkpoint_projection_set(
        [
            "checkpoint_projection_f16_m1",
            "checkpoint_projection_f16_m4",
            "checkpoint_projection_f16_m33",
            "checkpoint_projection_f16_m33_k64",
        ],
        DType::F16,
    )
}

/// Card 1024: the same table over a BF16 `[N, K]` weight: the decode GEMV (`M = 1`), the generated tiled GEMM
/// (`M = 4`, `33`, and a `K` that is a multiple of the K-unroll), each reading the weight's packed bfloat16 words. `N * K` is odd in the
/// `K = 200, N = 37` rows, so the last packed word carries one element and a zero pad.
pub fn checkpoint_projection_bf16_fixtures() -> Vec<Fixture> {
    checkpoint_projection_set(
        [
            "checkpoint_projection_bf16_m1",
            "checkpoint_projection_bf16_m4",
            "checkpoint_projection_bf16_m33",
            "checkpoint_projection_bf16_m33_k64",
        ],
        DType::BF16,
    )
}

/// POOT-1048: a grouped projection against a BF16 weight held as one checkpoint `[groups * N, K]` matrix, read as
/// `groups` blocks of rows: `matmul(x, transpose(reshape(w, [groups, N, K]), [0, 2, 1]))` with `x` a
/// `[groups, m, k]` activation slot (DeepSeek V4's `wo_a`). `compile` folds the chain into one
/// `GroupedDenseContraction` that reads `w` as stored, so no device `transpose` runs and the weight stays BF16
/// resident. Two steps with different activations, so the replayed step is checked against the oracle too.
fn grouped_projection_fixture(
    name: &'static str,
    groups: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Fixture {
    let b = poot_graph_ir::Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![groups, m, k]))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![groups * n, k], DType::BF16))
        .unwrap();
    let blocks = b.reshape(w, vec![groups, n, k]).unwrap();
    let out = b
        .matmul(x, b.transpose(blocks, vec![0, 2, 1]).unwrap())
        .unwrap();
    let graph = b.finish(out).unwrap();
    let key = graph.meta(x.id()).slot_key().unwrap().clone();
    let steps = (0..2u64)
        .map(|step| {
            vec![StepFixture {
                key: key.clone(),
                tensor: HostTensor::f32(vec![groups, m, k], fill(groups * m * k, 0x6A0B + step)),
            }]
        })
        .collect();
    Fixture {
        name,
        graph,
        store: narrow_weight_store("w", vec![groups * n, k], DType::BF16),
        steps,
    }
}

/// POOT-1048: [`grouped_projection_fixture`] with three groups at a decode row (`M = 1`), a few rows (`M = 5`) and a
/// prefill (`M = 33`), `K = 71` (past one 64-lane run and odd) and `N = 37`, so `groups * N * K` is odd and the
/// weight's last packed bfloat16 word carries one element and a zero pad; plus a single group, which the
/// grouped body must read as the ungrouped one.
pub fn grouped_projection_bf16_fixtures() -> Vec<Fixture> {
    vec![
        grouped_projection_fixture("grouped_projection_bf16_g3_m1", 3, 1, 71, 37),
        grouped_projection_fixture("grouped_projection_bf16_g3_m5", 3, 5, 71, 37),
        grouped_projection_fixture("grouped_projection_bf16_g3_m33", 3, 33, 71, 37),
        grouped_projection_fixture("grouped_projection_bf16_g1_m5", 1, 5, 71, 37),
    ]
}

/// POOT-1048: `fixture`'s program, as compiled for `target`, holds exactly one `GroupedDenseContraction` and
/// neither a `Transpose` nor a `Cast`, so a device row shows the contraction it ran is the one the fold made and
/// that the weight was read as stored (an unfolded chain computes the same values).
pub fn assert_one_grouped_dense_contraction(fixture: &Fixture, target: Target) {
    let ops = planned_ops(fixture, target);
    let count = |prefix: &str| ops.iter().filter(|op| op.starts_with(prefix)).count();
    assert_eq!(
        (
            count("grouped_dense_contraction"),
            count("transpose"),
            count("cast")
        ),
        (1, 0, 0),
        "{}: {ops:?}",
        fixture.name
    );
}

/// The shared projection table, named `[m1, m4, m33, m33_k64]`.
fn checkpoint_projection_set(names: [&'static str; 4], weight: DType) -> Vec<Fixture> {
    let [m1, m4, m33, m33_k64] = names;
    vec![
        checkpoint_projection_fixture(m1, 1, 200, 37, weight),
        checkpoint_projection_fixture(m4, 4, 200, 37, weight),
        checkpoint_projection_fixture(m33, 33, 200, 37, weight),
        checkpoint_projection_fixture(m33_k64, 33, 64, 40, weight),
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
    staged(fixture, target)
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
            StoredBytes::from(tensor.view().bytes().to_vec()),
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
    }
}

/// One step's fixture values prepared for an executor: every tensor is a raw view, except a
/// [`Slot::WriteMap`] one, which only binds as a checked [`ForwardSlotMap`]. The map is checked against the
/// pool `g` declares for the slot (the upper end of its declared range), so a fixture with a repeated or
/// out-of-pool slot is refused here, by the one constructor, before any executor sees it.
struct BoundStep<'a> {
    values: &'a [StepFixture],
    maps: Vec<(poot_graph_ir::SlotKey, ForwardSlotMap)>,
}

impl<'a> BoundStep<'a> {
    fn new(g: &Graph, values: &'a [StepFixture]) -> Result<Self, String> {
        let mut maps = Vec::new();
        for v in values.iter().filter(|v| v.key.role() == Slot::WriteMap) {
            let pool = g
                .inputs()
                .iter()
                .map(|&id| g.meta(id))
                .find(|meta| meta.slot_key() == Some(&v.key))
                .and_then(|meta| meta.range())
                .map(|range| range.hi_exclusive() as usize)
                .ok_or_else(|| format!("write map {} declares no pool range", v.key))?;
            let words = v
                .tensor
                .as_i32()
                .ok_or_else(|| format!("write map {} is not I32", v.key))?;
            let map = ForwardSlotMap::new(words.to_vec(), pool)
                .map_err(|e| format!("write map {}: {e}", v.key))?;
            maps.push((v.key.clone(), map));
        }
        Ok(Self { values, maps })
    }

    fn inputs(&self) -> StepInputs<'_> {
        let mut inputs = StepInputs::new();
        for v in self
            .values
            .iter()
            .filter(|v| v.key.role() != Slot::WriteMap)
        {
            inputs.push(v.key.clone(), v.tensor.shape(), v.tensor.view());
        }
        for (key, map) in &self.maps {
            inputs.push_write_map(key.clone(), map);
        }
        inputs
    }
}

/// `fixture`'s graph with each const named by a weight id, and the map over `fixture.store` that
/// binds those names (see [`poot_graph_ir::test_support::const_weights::ConstWeights`]). Deterministic: every
/// call names the same consts the same way, so [`staged`] and [`fixture_weights`] agree.
fn bound(fixture: &Fixture) -> (Graph, poot_quant::weights::WeightMap) {
    let mut weights = ConstWeights::new(&fixture.store);
    let graph = weights.graph(&fixture.graph);
    (graph, weights.map())
}

/// How `fixture`'s executable binds its store: load `Arc::new(fixture.store.clone())` with this and
/// add the entry [`staged`] compiles. The oracle keeps reading `fixture.graph` and `fixture.store`.
pub fn fixture_weights(fixture: &Fixture) -> Arc<poot_quant::weights::WeightMap> {
    Arc::new(bound(fixture).1)
}

/// Compile `fixture.graph` for `target`, single-stage, single-device (Card 546a's only shape).
pub fn staged(fixture: &Fixture, target: Target) -> StagedProgram {
    let g = bound(fixture).0;
    compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .unwrap()
}

/// `oracle` is called once per device step, in the exact order `run_parity` steps the device
/// (`fixture.steps[0]`, then `fixture.steps[1..]`, then a repeat of `fixture.steps[0]`), so a
/// stateful oracle computation can thread its own state the same way the device does. It returns
/// that step's expected output tensor.
pub type Oracle<'a> = dyn FnMut(&[StepFixture]) -> HostTensor + 'a;

/// How many tolerances the largest oracle element of a compared step must exceed.
const SIGNAL_MARGIN: f32 = 10.0;

/// The magnitude the largest oracle element of a step must exceed for `class`. A step whose every element
/// is below it passes however wrong the device is, because the tolerance is absolute below 1; such a
/// fixture proves nothing and is refused rather than compared.
fn signal_floor(class: Tier2Class) -> f32 {
    class.tolerance() * SIGNAL_MARGIN
}

/// Compare one device step output with the oracle's within the tolerance of the program's `class`
/// (ADR-0101 decision 2): same dtype and shape, and every element within
/// `class.tolerance() * max(|oracle|, 1)`. CPU and device float ops are not bit-identical, so an
/// oracle comparison needs the tolerance; record-then-replay bit-exactness is a separate, narrower
/// claim this function does not make. A NaN never agrees.
///
/// The oracle output must also carry signal: its largest element must exceed
/// [`signal_floor`]. Below the floor every element is inside the tolerance of zero, so a
/// device that wrote nothing would pass; such a fixture is refused, not compared.
fn compare_step(class: Tier2Class, got: &HostTensor, want: &HostTensor) -> Result<(), String> {
    if got.dtype() != want.dtype() || got.shape() != want.shape() {
        return Err(format!(
            "{} {:?} vs the oracle's {} {:?}",
            got.dtype(),
            got.shape(),
            want.dtype(),
            want.shape()
        ));
    }
    let (Ok(got), Ok(want)) = (got.to_f32(), want.to_f32()) else {
        return Err("the output does not read as f32".to_string());
    };
    let signal = want.iter().fold(0.0f32, |most, v| most.max(v.abs()));
    let floor = signal_floor(class);
    if signal.partial_cmp(&floor) != Some(std::cmp::Ordering::Greater) {
        return Err(format!(
            "the oracle's largest element is {signal}, not above the {class:?} signal floor {floor}: \
             the fixture cannot show a wrong device"
        ));
    }
    let tolerance = class.tolerance();
    poot_test_util::check_close_unit_floor(&got, &want, tolerance)
        .map_err(|worst| format!("{worst} outside the {class:?} tolerance {tolerance}"))
}

/// One named `Storage::Const` row for [`run_once`]: the checkpoint-style name a traced graph's
/// const input declares (`meta.name`) and its typed host tensor (matching [`StepFixture`]'s
/// `Storage::Slot` row shape).
pub struct ConstFixture {
    /// The checkpoint-style name of the `Storage::Const` input this row binds.
    pub name: &'static str,
    /// The constant's value, typed to match the input's aval.
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
/// let mut gpu = poot_wgpu_device::GpuExecutor::new()?; // deleted
/// let got = gpu.run(&g, &inputs)?; // inputs: HashMap<ValueId, poot_eval::Value>, Const rows and
///                                  // Slot rows mixed together by ValueId.
///
/// // after (the executor contract, via this crate):
/// let device = poot_wgpu_device::device::WgpuDevice::new()?;
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
            StoredBytes::from(c.tensor.view().bytes().to_vec()),
        )
        .map_err(|e| format!("run_once: const `{}`: {e}", c.name))?;
        builder
            .insert(c.name, WeightEntry::Dense(dense))
            .map_err(|e| format!("run_once: const `{}`: {e}", c.name))?;
    }
    let store = builder.build();
    let mut weights = ConstWeights::new(&store);
    let g = weights.graph(g);
    let exe = exec
        .load_weights(Arc::new(store.clone()), Arc::new(weights.map()))
        .map_err(|e| format!("run_once: load_weights: {e}"))?;
    let staged = compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .map_err(|e| format!("run_once: compile_staged: {e}"))?;
    let entry = exec
        .add_entry(exe, &staged)
        .map_err(|e| format!("run_once: add_entry: {e}"))?;
    let output = exec
        .step(
            exe,
            entry,
            &BoundStep::new(&g, slots)?.inputs(),
            &mut NoSync,
        )
        .map_err(|e| format!("run_once: step: {e}"))?
        .to_host()
        .map_err(|e| format!("run_once: read: {e}"))?;
    exec.remove_entry(exe, entry)
        .map_err(|e| format!("run_once: remove_entry: {e}"))?;
    exec.unload(exe)
        .map_err(|e| format!("run_once: unload: {e}"))?;
    Ok(output)
}

/// The softcapped trace's oracle differs from the uncapped trace's at some step by more than `5e-2`, relative
/// to `max(|uncapped|, 1)` (ten times the parity tolerance): a fixture whose softcap the oracle cannot see
/// shows no device applying it. Panics on a NaN in either oracle output.
pub fn assert_softcap_is_visible(softcapped: &Fixture, uncapped: &Fixture) {
    use poot_eval::test_support::weight_map_oracle::oracle_for;
    let (mut capped_oracle, mut plain_oracle) = (
        oracle_for(&softcapped.graph, &softcapped.store),
        oracle_for(&uncapped.graph, &uncapped.store),
    );
    let largest = softcapped
        .steps
        .iter()
        .zip(&uncapped.steps)
        .map(|(capped_step, plain_step)| {
            let capped = capped_oracle(capped_step);
            let plain = plain_oracle(plain_step);
            poot_test_util::max_unit_floor_error(
                capped.as_f32().expect("F32 oracle output"),
                plain.as_f32().expect("F32 oracle output"),
            )
        })
        .max_by(f32::total_cmp)
        .expect("a fixture has at least one step");
    assert!(
        largest > 5e-2,
        "{}: the softcap barely changes the oracle (max relative difference {largest:.2e}), so this \
         fixture cannot show a device applying it",
        softcapped.name
    );
}

/// SC-002: load `fixture` onto `exec` at `target`, step through every declared step, and assert
/// that each step's replayed output matches `oracle`'s independently-computed expectation for that
/// same step - including a final repeat of step 0's inputs, so a pure-replay step (no new recording)
/// is checked against the oracle too, not only the recording step (ADR-0101 tier 2).
///
/// The tolerance is not a parameter: it is [`Tier2Class::tolerance`] of the class the compiled program
/// publishes ([`StagedProgram::numerics_class`]), so a row cannot pick a looser bound than its program's
/// passes allow. A step whose oracle output carries no signal above that tolerance is refused.
///
/// ```compile_fail
/// fn with_a_tolerance(
///     exec: &mut dyn poot_executor::Executor,
///     target: poot_graph_plan::Target,
///     fixture: &poot_executor_parity::Fixture,
///     oracle: &mut poot_executor_parity::Oracle<'_>,
/// ) {
///     let _ = poot_executor_parity::run_parity(exec, target, fixture, oracle, 1e-2);
/// }
/// ```
pub fn run_parity(
    exec: &mut dyn Executor,
    target: Target,
    fixture: &Fixture,
    oracle: &mut Oracle<'_>,
) -> Result<(), String> {
    let exe = exec
        .load_weights(Arc::new(fixture.store.clone()), fixture_weights(fixture))
        .map_err(|e| format!("{}: load_weights: {e}", fixture.name))?;
    let program = staged(fixture, target);
    let class = program.numerics_class();
    let entry = exec
        .add_entry(exe, &program)
        .map_err(|e| format!("{}: add_entry: {e}", fixture.name))?;
    for (i, step) in fixture.steps.iter().enumerate() {
        let expected = oracle(step);
        let got = exec
            .step(
                exe,
                entry,
                &BoundStep::new(&fixture.graph, step)?.inputs(),
                &mut NoSync,
            )
            .map_err(|e| format!("{}: step {i}: {e}", fixture.name))?
            .to_host()
            .map_err(|e| format!("{}: read {i}: {e}", fixture.name))?;
        compare_step(class, &got, &expected).map_err(|e| {
            format!(
                "{}: step {i}'s replayed output does not match the oracle: {e}",
                fixture.name
            )
        })?;
    }
    let first = &fixture.steps[0];
    let expected = oracle(first);
    let got = exec
        .step(
            exe,
            entry,
            &BoundStep::new(&fixture.graph, first)?.inputs(),
            &mut NoSync,
        )
        .map_err(|e| format!("{}: repeat step: {e}", fixture.name))?
        .to_host()
        .map_err(|e| format!("{}: repeat read: {e}", fixture.name))?;
    compare_step(class, &got, &expected).map_err(|e| {
        format!(
            "{}: the repeated replay step does not match the oracle: {e}",
            fixture.name
        )
    })?;
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
        .load_weights(Arc::new(fixture.store.clone()), fixture_weights(fixture))
        .map_err(|e| format!("{}: load_weights: {e}", fixture.name))?;
    let entry = exec
        .add_entry(exe, &staged(fixture, target))
        .map_err(|e| format!("{}: add_entry: {e}", fixture.name))?;
    let mut outputs = Vec::with_capacity(fixture.steps.len());
    for (i, step) in fixture.steps.iter().enumerate() {
        outputs.push(
            exec.step(
                exe,
                entry,
                &BoundStep::new(&fixture.graph, step)?.inputs(),
                &mut NoSync,
            )
            .map_err(|e| format!("{}: step {i}: {e}", fixture.name))?
            .to_host()
            .map_err(|e| format!("{}: read {i}: {e}", fixture.name))?,
        );
    }
    exec.unload(exe)
        .map_err(|e| format!("{}: unload: {e}", fixture.name))?;
    Ok(outputs)
}

/// The device time of one step of `name`: its device span, first dispatch start to last dispatch end. The span
/// is the one measure every backend reports; PTX replays a captured graph bracketed by one event pair and has
/// no per-dispatch duration, so a receipt keyed on the dispatch sum would find none there. `Err` when the
/// device reports no timestamps at all: a receipt never falls back to host wall.
fn step_device_time(
    name: &str,
    device_time: &poot_executor::DeviceTime,
) -> Result<std::time::Duration, String> {
    let poot_executor::DeviceTime::Measured(measured) = device_time else {
        return Err(format!("{name}: the device reports no timestamps"));
    };
    measured
        .device_span
        .ok_or_else(|| format!("{name}: no device span"))
}

/// Load `fixture` onto `exec` and step its first step `repeats` times, returning each step's device span
/// ([`step_device_time`]). A step is the fixture's one op, so the span is that kernel's time (and any copy the
/// plan adds). `Err` when the device reports no timestamps: a receipt never falls back to host wall.
pub fn run_device_times(
    exec: &mut dyn Executor,
    target: Target,
    fixture: &Fixture,
    repeats: usize,
) -> Result<Vec<std::time::Duration>, String> {
    let exe = exec
        .load_weights(Arc::new(fixture.store.clone()), fixture_weights(fixture))
        .map_err(|e| format!("{}: load_weights: {e}", fixture.name))?;
    let entry = exec
        .add_entry(exe, &staged(fixture, target))
        .map_err(|e| format!("{}: add_entry: {e}", fixture.name))?;
    let step = fixture
        .steps
        .first()
        .ok_or_else(|| format!("{}: no step", fixture.name))?;
    let mut times = Vec::with_capacity(repeats);
    for i in 0..repeats {
        let out = exec
            .step(
                exe,
                entry,
                &BoundStep::new(&fixture.graph, step)?.inputs(),
                &mut NoSync,
            )
            .map_err(|e| format!("{}: step {i}: {e}", fixture.name))?;
        times.push(step_device_time(fixture.name, &out.device_time())?);
        drop(out);
    }
    exec.unload(exe)
        .map_err(|e| format!("{}: unload: {e}", fixture.name))?;
    Ok(times)
}

/// A perf receipt: the device time of each fixture's first step, `rounds` times over, the fixtures visited in
/// an order that reverses every round (A B, B A, ...) with `repeats` steps each visit, so a drift of the box
/// between visits lands on every fixture alike (single numbers on this box move about 30% between holds).
/// Returns one sample list per fixture, in the order given; the first step of every visit (a cold cache) is
/// dropped.
pub fn receipt(
    exec: &mut dyn Executor,
    target: Target,
    fixtures: &[&Fixture],
    rounds: usize,
    repeats: usize,
) -> Result<Vec<Vec<std::time::Duration>>, String> {
    let mut samples = vec![Vec::new(); fixtures.len()];
    for round in 0..rounds {
        let order: Vec<usize> = if round % 2 == 0 {
            (0..fixtures.len()).collect()
        } else {
            (0..fixtures.len()).rev().collect()
        };
        for i in order {
            let times = run_device_times(exec, target, fixtures[i], repeats + 1)?;
            samples[i].extend(times.into_iter().skip(1));
        }
    }
    Ok(samples)
}

/// Capture-and-replay repeatability. The entry's first step records and replays in one call; every
/// later step is a pure replay. This steps `fixture` once, comparing each step to `oracle` within
/// the program's class tolerance (as [`run_parity`] does, so a replay that drops a state commit diverges from the
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
        .load_weights(Arc::new(fixture.store.clone()), fixture_weights(fixture))
        .map_err(|e| format!("{}: load_weights: {e}", fixture.name))?;
    let program = staged(fixture, target);
    let class = program.numerics_class();
    let entry = exec
        .add_entry(exe, &program)
        .map_err(|e| format!("{}: add_entry: {e}", fixture.name))?;
    let mut first_pass = Vec::with_capacity(fixture.steps.len());
    for (i, step) in fixture.steps.iter().enumerate() {
        let expected = oracle(step);
        let got = exec
            .step(
                exe,
                entry,
                &BoundStep::new(&fixture.graph, step)?.inputs(),
                &mut NoSync,
            )
            .map_err(|e| format!("{}: step {i}: {e}", fixture.name))?
            .to_host()
            .map_err(|e| format!("{}: read {i}: {e}", fixture.name))?;
        compare_step(class, &got, &expected).map_err(|e| {
            format!(
                "{}: step {i}'s output does not match the oracle: {e}",
                fixture.name
            )
        })?;
        first_pass.push(got);
    }
    exec.reset_state(exe, StateScope::All)
        .map_err(|e| format!("{}: reset_state: {e}", fixture.name))?;
    for (i, (step, recorded)) in fixture.steps.iter().zip(&first_pass).enumerate() {
        let replayed = exec
            .step(
                exe,
                entry,
                &BoundStep::new(&fixture.graph, step)?.inputs(),
                &mut NoSync,
            )
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

#[cfg(test)]
mod device_time_tests {
    use std::time::Duration;

    use poot_executor::{DeviceTime, MeasuredDeviceTime};

    use super::step_device_time;

    /// PTX reports a graph-replay span and no per-dispatch durations: the receipt reads that span.
    ///
    /// Mutation: read `sum_of_dispatch_durations` instead of `device_span`; this fails with
    /// `Err("row: no device span")` (the card 1170 failure shape).
    #[test]
    fn a_span_only_measurement_yields_the_span() {
        let ptx_shaped = DeviceTime::Measured(MeasuredDeviceTime {
            sum_of_dispatch_durations: None,
            device_span: Some(Duration::from_micros(7)),
            dispatches: Vec::new(),
        });
        assert_eq!(
            step_device_time("row", &ptx_shaped),
            Ok(Duration::from_micros(7))
        );
    }

    #[test]
    fn unknown_device_time_is_refused_not_fabricated() {
        let err = step_device_time("row", &DeviceTime::Unknown).unwrap_err();
        assert!(err.contains("no timestamps"), "{err}");
    }
}

#[cfg(test)]
mod compare_tests {
    use poot_graph_ir::analysis::Tier2Class;
    use poot_graph_ir::{BinOp, Builder, TensorType, UnOp};
    use poot_target::Backend;
    use poot_tensor::HostTensor;
    use poot_test_util::device_caps::default_caps_for;

    use super::*;

    /// A wgpu-shaped target: the class is read off the compiled program, which needs no device.
    fn target() -> Target {
        Target::Gpu {
            backend: Backend::SpirvVulkan,
            caps: default_caps_for(Backend::SpirvVulkan),
        }
    }

    /// `exp(x * s) + bias` over constants: a float pointwise chain `compile` fuses into one region, so the
    /// program publishes the reassociating class.
    fn fused_chain() -> Fixture {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![4, 8])).unwrap();
        let s = b.constant("s", TensorType::f32(vec![4, 8])).unwrap();
        let bias = b.constant("bias", TensorType::f32(vec![4, 8])).unwrap();
        let scaled = b.binary(BinOp::Mul, x, s).unwrap();
        let e = b.unary(UnOp::Exp, scaled).unwrap();
        let out = b.binary(BinOp::Add, e, bias).unwrap();
        let graph = b.finish(out).unwrap();
        let store = store_for(&graph);
        Fixture {
            name: "fused_chain",
            graph,
            store,
            steps: vec![Vec::new()],
        }
    }

    /// The oracle's output: every element at magnitude one, past the signal floor of every class.
    fn oracle_output() -> HostTensor {
        HostTensor::f32(vec![4, 8], vec![1.0; 32])
    }

    /// `want` with every element moved by `multiple` of `tolerance` (the elements are one, so the bound
    /// `tolerance * max(|want|, 1)` is `tolerance`).
    fn perturbed(want: &HostTensor, tolerance: f32, multiple: f32) -> HostTensor {
        let moved = want
            .as_f32()
            .unwrap()
            .iter()
            .map(|v| v + multiple * tolerance)
            .collect();
        HostTensor::f32(want.shape().to_vec(), moved)
    }

    /// SC-002: the class a parity row compares within is the one its compiled program publishes, and a
    /// device result 2x that class's published tolerance off the oracle fails while one 0.5x off passes.
    /// The published reassociating tolerance is pinned here as the ADR-0101 table's number.
    ///
    /// Mutation: make `Tier2Class::tolerance` return a constant `1e-2`; the 2x row passes and this goes
    /// red on `a result 2x the tolerance off the oracle must fail`.
    #[test]
    fn a_reassociating_program_is_compared_within_its_class_tolerance() {
        const PUBLISHED_REASSOCIATING: f32 = 1e-3;
        let program = staged(&fused_chain(), target());
        let class = program.numerics_class();
        assert_eq!(
            class,
            Tier2Class::Reassociating,
            "the chain must fuse, or this row proves nothing about the reassociating class"
        );
        let want = oracle_output();
        let far = perturbed(&want, PUBLISHED_REASSOCIATING, 2.0);
        let err = compare_step(class, &far, &want)
            .expect_err("a result 2x the tolerance off the oracle must fail");
        assert!(err.contains("outside the Reassociating tolerance"), "{err}");
        compare_step(
            class,
            &perturbed(&want, PUBLISHED_REASSOCIATING, 0.5),
            &want,
        )
        .expect("a result 0.5x the tolerance off the oracle must pass");
    }

    /// SC-002, through `run_parity` itself: the real engine, compile and `run_parity` run over a device
    /// that answers with the oracle moved by `multiple` of the reassociating tolerance. The wiring from the
    /// program's class to the comparison is what fails here: a `run_parity` that compared within the
    /// `NarrowStorage` bound (5x looser) would pass the 2x row.
    ///
    /// Mutation: in `run_parity` replace `program.numerics_class()` with `Tier2Class::NarrowStorage`; the
    /// 2x row passes and this goes red on `a device 2x the reassociating tolerance off must fail`.
    #[test]
    fn run_parity_compares_within_the_class_of_the_program_it_ran() {
        const PUBLISHED_REASSOCIATING: f32 = 1e-3;
        let run = |multiple: f32| {
            let (mut exec, target, answer) = crate::echo::echo_executor();
            let want = oracle_output();
            let mut oracle = |_: &[StepFixture]| {
                answer.set(&perturbed(&want, PUBLISHED_REASSOCIATING, multiple));
                want.clone()
            };
            run_parity(exec.as_mut(), target, &fused_chain(), &mut oracle)
        };
        let err = run(2.0).expect_err("a device 2x the reassociating tolerance off must fail");
        assert!(err.contains("outside the Reassociating tolerance"), "{err}");
        run(0.5).expect("a device 0.5x the tolerance off must pass");
    }

    /// The comparator's negative rows (R2-C5-006): each way a device output can disagree with the oracle is
    /// refused, and the matching output passes.
    ///
    /// Mutation: in `poot_test_util::Worst::of` drop the `error.is_nan()` arm (the NaN-blind form); the NaN row
    /// passes and this goes red on `an all-NaN output must fail`.
    #[test]
    fn compare_step_refuses_nan_wrong_dtype_wrong_shape_and_empty() {
        let class = Tier2Class::Reassociating;
        let want = oracle_output();
        compare_step(class, &want, &want).expect("the oracle's own output must agree");

        let nan = HostTensor::f32(want.shape().to_vec(), vec![f32::NAN; want.numel()]);
        let err = compare_step(class, &nan, &want).expect_err("an all-NaN output must fail");
        assert!(err.contains("actual NaN"), "{err}");

        let wrong_shape = HostTensor::f32(vec![want.numel()], vec![0.0; want.numel()]);
        let err = compare_step(class, &wrong_shape, &want)
            .expect_err("a differently shaped output must fail");
        assert!(err.contains("vs the oracle's"), "{err}");

        let wrong_dtype = HostTensor::i32(want.shape().to_vec(), vec![0; want.numel()]);
        let err = compare_step(class, &wrong_dtype, &want)
            .expect_err("a differently typed output must fail");
        assert!(err.contains("vs the oracle's"), "{err}");

        let empty = HostTensor::f32(vec![0], Vec::new());
        assert!(
            compare_step(class, &empty, &empty).is_err(),
            "two empty outputs must be refused"
        );
    }

    /// SC-003: an oracle output below the class's signal floor is refused, however close the device is,
    /// because a device that wrote nothing would agree with it.
    ///
    /// Mutation: delete the signal check in `compare_step`; the all-zero output passes and this goes red on
    /// `an all-zero oracle output must be refused`.
    #[test]
    fn an_oracle_output_with_no_signal_is_refused() {
        let class = Tier2Class::Reassociating;
        let zeros = HostTensor::f32(vec![4, 8], vec![0.0; 32]);
        let err = compare_step(class, &zeros, &zeros)
            .expect_err("an all-zero oracle output must be refused");
        assert!(err.contains("signal floor"), "{err}");
        let faint = HostTensor::f32(vec![4, 8], vec![signal_floor(class) * 0.5; 32]);
        assert!(compare_step(class, &faint, &faint).is_err());
        compare_step(class, &oracle_output(), &oracle_output())
            .expect("an output with signal agrees with itself");
    }
}
