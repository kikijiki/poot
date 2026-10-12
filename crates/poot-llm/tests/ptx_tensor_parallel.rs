//! POOT-749 on two NVIDIA devices (and POOT-1102 on four): a tensor-parallel decode agrees with the
//! single-device decode.
//!
//! The same fixture is loaded twice: once as one program on one device through an ordinary
//! [`Engine`], and once as a [`Coordinator`] over two [`PtxDevice`]s whose weights are cut by
//! [`poot_graph_plan::shard`] (rank `r` binds its [`poot_quant::weights::WeightMap::rank_view`]). Both
//! decode the same tokens and every logit is compared. A partial sum crosses devices as a peer copy
//! recorded in the producing stage's CUDA graph and is added to the rank's own partial by an ordinary
//! add of the consuming stage, so a skipped reduction, a dropped copy or a wrong share changes the
//! logits.
//!
//! Skips cleanly with fewer NVIDIA GPUs than a row needs, and fails loudly under
//! `POOT_REQUIRE_PTX_2GPU=1` (two) or `POOT_REQUIRE_PTX_4GPU=1` (the four-way row).

use std::sync::Arc;

use poot_executor::{Coordinator, Device, Engine, Executor, NoSync};
use poot_executor_parity::{TIER2, assert_close};
use poot_graph_ir::{IndexRange, Slot};
use poot_graph_plan::{
    CompileLimits, CompileOptions, ShardRule, multi_device::topology::DeviceId,
    production_plan::Submission, staged::DevicePlacement, staged::ExpertPlacement,
    staged::Partition, staged::StagedProgram, staged::TargetSet, staged::compile_staged,
};
use poot_models::model::Phase;
use poot_models::test_support::dense::{Dense, Family};
use poot_models::test_support::weight_map::{MappedModel, Step, decode_steps, trace};
use poot_ptx_device::PtxDevice;
use poot_runtime_common::DeviceBackend;

const TOKENS: [i32; 6] = [3, 11, 7, 29, 5, 17];

/// The cache capacity every row traces with: at least the longest run any row drives (24 steps), so
/// every position stays inside the `Pos` slot's declared range `[0, CAPACITY)`.
const CAPACITY: usize = 32;

/// Token-by-token decode of `ids` from position 0, refusing a run that would leave the traced
/// capacity (a position past it is rejected at bind, and the row would test the rejection).
fn decode(ids: &[i32]) -> Vec<Step> {
    assert!(
        ids.len() <= CAPACITY,
        "{} steps exceed the traced capacity {CAPACITY}",
        ids.len()
    );
    decode_steps(ids)
}

/// NVIDIA devices `0..count`, or a skip: `None` without device 0, and without a later one unless
/// `require` names an environment variable that makes the absence a failure.
fn open_devices(count: usize, require: &str) -> Option<Vec<PtxDevice>> {
    let first =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::open(0))?;
    let mut devices = vec![first];
    for ordinal in 1..count {
        match PtxDevice::open(ordinal) {
            Ok(device) => devices.push(device),
            Err(error) => {
                assert!(
                    std::env::var_os(require).is_none(),
                    "{count} NVIDIA devices are required ({require}) and device {ordinal} is \
                     unavailable: {error}"
                );
                eprintln!("SKIP: NVIDIA device {ordinal} is unavailable: {error}");
                return None;
            }
        }
    }
    Some(devices)
}

fn open_two() -> Option<Vec<PtxDevice>> {
    open_devices(2, "POOT_REQUIRE_PTX_2GPU")
}

fn options() -> CompileOptions {
    CompileOptions::new(Submission::Replay, CompileLimits::STANDARD)
}

fn logits<E: Executor>(
    exec: &mut E,
    exe: poot_executor::ExecutableId,
    entry: poot_executor::EntryId,
    steps: &[Step],
) -> Vec<Vec<f32>> {
    steps
        .iter()
        .map(|step| {
            exec.step(exe, entry, &step.inputs(), &mut NoSync)
                .expect("a step")
                .to_host()
                .expect("logits read")
                .as_f32()
                .expect("f32 logits")
                .to_vec()
        })
        .collect()
}

fn single_device(mapped: &MappedModel, steps: &[Step]) -> Vec<Vec<f32>> {
    let device = PtxDevice::open(0).unwrap();
    let target = device.target();
    let mut engine = Engine::new(device);
    let exe = engine
        .load_weights(Arc::clone(&mapped.store), Arc::clone(&mapped.map))
        .unwrap();
    let g = trace(&*mapped.model, &mapped.map, Phase::Decode, 1, CAPACITY);
    let program = compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &options(),
    )
    .expect("the fixture compiles on one device");
    let entry = engine.add_entry(exe, &program).unwrap();
    logits(&mut engine, exe, entry, steps)
}

/// A tensor-parallel coordinator that has prepared the fixture's decode entry on both devices.
struct Group {
    coordinator: Coordinator<PtxDevice>,
    exe: poot_executor::ExecutableId,
    entry: poot_executor::EntryId,
    program: StagedProgram,
}

fn tensor_parallel(devices: Vec<PtxDevice>, mapped: &MappedModel) -> Group {
    let ids: Vec<DeviceId> = (0..devices.len() as u32).map(DeviceId).collect();
    let set = TargetSet::new(
        ids.iter()
            .zip(&devices)
            .map(|(&id, device)| (id, device.target()))
            .collect(),
    )
    .unwrap();
    let g = trace(&*mapped.model, &mapped.map, Phase::Decode, 1, CAPACITY);
    let program = compile_staged(
        &g,
        &set,
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Tensor {
                rule: ShardRule::Megatron,
                devices: ids.clone(),
                weights: Arc::clone(&mapped.map),
            },
        },
        &options(),
    )
    .unwrap_or_else(|error| panic!("compile_staged: {error}"));
    let mut coordinator = Coordinator::new(ids.into_iter().zip(devices).collect()).unwrap();
    let exe = coordinator
        .load_tensor_weights(
            Arc::clone(&mapped.store),
            Arc::clone(&mapped.map),
            program.tensor().expect("a tensor-parallel program"),
        )
        .unwrap();
    let entry = coordinator.add_entry(exe, &program).unwrap();
    Group {
        coordinator,
        exe,
        entry,
        program,
    }
}

fn assert_tensor_parallel_matches_single(
    devices: Vec<PtxDevice>,
    mapped: &MappedModel,
    label: &str,
) {
    let steps = decode(&TOKENS);
    let want = single_device(mapped, &steps);
    assert!(
        want.windows(2).any(|pair| pair[0] != pair[1]),
        "{label}: constant logits prove nothing"
    );
    let world = devices.len();
    let mut group = tensor_parallel(devices, mapped);
    assert!(
        group.program.transfers().len() > 1,
        "{label}: the ranks exchange partial sums"
    );
    let got = logits(&mut group.coordinator, group.exe, group.entry, &steps);
    for (step, (got, want)) in got.iter().zip(&want).enumerate() {
        assert_close(got, want, TIER2, &format!("{label}: step {step}"));
    }
    let counters = group.coordinator.counters();
    assert_eq!(
        counters.devices.len(),
        world,
        "{label}: one counter per rank"
    );
    for device in &counters.devices {
        assert!(
            device.sessions >= 1 && device.launches >= 1 && device.transfers >= 1,
            "{label}: device {:?} took part and sent: {device:?}",
            device.device
        );
    }
}

/// SC-003 (qwen2): TP on two GPUs matches single-GPU decode (ADR-0101 tier 2).
///
/// Mutation: skip the `AllReduce` after layer 0's `o_proj` (`Sharder::all_reduce` hands the partial
/// on as whole); the ranks never exchange it and the program is refused at compile.
#[test]
fn qwen2_tensor_parallel_decode_matches_single_device_on_ptx() {
    let Some(devices) = open_two() else { return };
    assert_tensor_parallel_matches_single(devices, &Dense::new(Family::Qwen2).model(), "qwen2");
}

/// POOT-1102 SC-001: TP on four GPUs matches single-GPU decode (ADR-0101 tier 2). Each reduction
/// sums four partials that crossed three peer copies into every rank, so a dropped hop or a peer
/// pair that cannot copy changes the logits or fails the step.
///
/// Mutation: flip the sign of the third partial's hop in `expand_ranks`'s `AllReduce` fold (`Sub`
/// for `from == 2`); every reduction is wrong and the logits leave the tier-2 bound.
#[test]
fn qwen2_four_way_tensor_parallel_decode_matches_single_device_on_ptx() {
    let Some(devices) = open_devices(4, "POOT_REQUIRE_PTX_4GPU") else {
        return;
    };
    let mapped = Dense::new(Family::Qwen2).heads(4, 4).model();
    assert_tensor_parallel_matches_single(devices, &mapped, "qwen2 x4");
}

/// SC-003 (a non-qwen2 dense family): Llama has no QKV bias and a gated MLP of its own spelling.
#[test]
fn llama_tensor_parallel_decode_matches_single_device_on_ptx() {
    let Some(devices) = open_two() else { return };
    assert_tensor_parallel_matches_single(devices, &Dense::new(Family::Llama).model(), "llama");
}

/// SC-003 (a second non-qwen2 family): Qwen3 normalizes each head's query and key, an explicit head
/// dimension that is not `hidden / heads`.
#[test]
fn qwen3_tensor_parallel_decode_matches_single_device_on_ptx() {
    let Some(devices) = open_two() else { return };
    assert_tensor_parallel_matches_single(devices, &Dense::new(Family::Qwen3).model(), "qwen3");
}

/// SC-004 on the device: over sixteen unchanged steps each rank has one recording session and
/// instantiates no native graph after the first step, and the weights read from the store do not move.
///
/// Mutation: rebuild the executable per token; the bytes read from the store grow every step and the
/// graph counter moves.
#[test]
fn sixteen_tensor_parallel_steps_upload_the_weight_shares_once_on_ptx() {
    let Some(devices) = open_two() else { return };
    let mapped = Dense::new(Family::Qwen2).dims(64, 96, 3).model();
    let steps = decode(&(0..16).map(|i| (i * 5 + 1) % 48).collect::<Vec<_>>());
    let mut group = tensor_parallel(devices, &mapped);
    logits(&mut group.coordinator, group.exe, group.entry, &steps[..1]);
    let before = group.coordinator.counters();
    let read = group.coordinator.stats().weight_bytes_read;
    assert!(read > 0, "the first step uploaded the shares");
    logits(&mut group.coordinator, group.exe, group.entry, &steps[1..]);
    let after = group.coordinator.counters();
    assert_eq!(
        group.coordinator.stats().weight_bytes_read,
        read,
        "no weight is read from the store again"
    );
    for (b, a) in before.devices.iter().zip(&after.devices) {
        assert_eq!(
            a.sessions, 1,
            "device {:?}: one recording session",
            a.device
        );
        assert_eq!(
            a.native.graph_instantiations, b.native.graph_instantiations,
            "device {:?}: no native graph after preparation",
            a.device
        );
        assert!(a.launches > b.launches);
    }
}

/// Model-free twin of the long-run rows on a device-less box: the fixture traced over
/// [`CAPACITY`] declares its `Pos` slot `[0, CAPACITY)`, and every position the 16-step
/// run drives is inside it, so the device rows exercise the programs, not the bind rejection.
///
/// Mutation: `CAPACITY = 8` (the old shared capacity); `decode` refuses the run with "N steps exceed
/// the traced capacity 8" (with that guard removed, the range check names position 8).
#[test]
fn the_long_run_stays_inside_the_traced_position_range() {
    let mapped = Dense::new(Family::Qwen2).dims(64, 96, 3).model();
    let g = trace(&*mapped.model, &mapped.map, Phase::Decode, 1, CAPACITY);
    let pos = g
        .slots()
        .iter()
        .find(|(_, slot)| *slot == Slot::Pos)
        .map(|(id, _)| g.meta(*id).range())
        .expect("a decode trace has a Pos slot")
        .expect("the production tracer declares the Pos range");
    assert_eq!(pos, IndexRange::below(CAPACITY).unwrap());
    for step in &decode(&(0..16).map(|i| (i * 5 + 1) % 48).collect::<Vec<_>>()) {
        let last = step.start + step.ids.len() - 1;
        assert!(
            pos.contains(last as i32),
            "position {last} is outside the declared Pos range {pos:?}"
        );
    }
}
