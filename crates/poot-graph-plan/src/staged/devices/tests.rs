//! POOT-747: what `compile_staged` produces and refuses when the stages sit on distinct devices. Every
//! row is host-only: `compile_staged` compiles for a target and never opens a device, so a PTX
//! `TargetSet` plans without a GPU, and a refusal is shown by the planner's own compile counter staying
//! put.

use std::collections::HashMap;

use poot_graph_ir::{BinOp, Builder, Graph, RedOp, Slot, StateRole, TensorType};
use poot_target::{AmdArch, Backend};
use poot_tensor::DType;
use poot_test_util::device_caps::default_caps_for;

use crate::CollectiveKind;
use crate::multi_device::{AccountingError, PlacementError};
use crate::planner::per_eqn_plans_inside_compile;
use crate::staged::{
    DevicePlacement, ExpertPlacement, Partition, StagedCompileError, TargetSet, compile_staged,
};
use crate::test_support::{LAYERS, qwen2_decode};
use crate::{Capability, CompileLimits, CompileOptions, DeviceId, StageId, Submission, Target};

const GIB: u64 = 1 << 30;

fn target(backend: Backend, budget: u64) -> Target {
    let mut caps = default_caps_for(backend);
    caps.memory_budget_bytes = Some(budget);
    Target { backend, caps }
}

fn options() -> CompileOptions {
    CompileOptions::new(Submission::Replay, CompileLimits::STANDARD)
}

/// A pipeline over devices `0..stages`, cut after layers `0..stages - 1`.
fn pipeline(stages: usize) -> Partition {
    Partition {
        experts: ExpertPlacement::AllResident,
        devices: DevicePlacement::Pipeline {
            after_layers: (0..stages - 1).collect(),
            devices: (0..stages as u32).map(DeviceId).collect(),
        },
    }
}

/// One device per stage, each with `backend` and its own budget.
fn targets(backend: Backend, budgets: &[u64]) -> TargetSet {
    TargetSet::new(
        budgets
            .iter()
            .enumerate()
            .map(|(index, &budget)| (DeviceId(index as u32), target(backend, budget)))
            .collect(),
    )
    .expect("distinct devices")
}

/// Layer 0 on device 0 reads `w0` and hands `t0` to layers 1 and 2; layer 1 contracts it against a
/// bigger weight and layer 2 reads both `t0` and layer 1's `t1`. Every device holds a different
/// amount: 16, 512 and 128 weight bytes.
fn three_device_fixture() -> Graph {
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4]));
    let (t0, t1);
    {
        let _layer = b.layer_scope(0);
        let w0 = b.constant("w0", TensorType::f32(vec![4]));
        t0 = b.binary(BinOp::Mul, x, w0);
    }
    {
        let _layer = b.layer_scope(1);
        let w1 = b.constant("w1", TensorType::f32(vec![4, 32]));
        let row = b.reshape(t0, vec![1, 4]);
        t1 = b.matmul(row, w1);
    }
    let out;
    {
        let _layer = b.layer_scope(2);
        let w2 = b.constant("w2", TensorType::f32(vec![1, 32]));
        let total = b.reduce(RedOp::Sum, t0, 0, true);
        let shifted = b.binary(BinOp::Add, t1, w2);
        out = b.binary(BinOp::Mul, shifted, total);
    }
    b.finish(out)
}

fn compile_on(
    graph: &Graph,
    targets: &TargetSet,
    partition: &Partition,
) -> Result<crate::StagedProgram<poot_graph_ir::NoValidations>, StagedCompileError> {
    compile_staged(graph, targets, partition, &options())
}

/// SC-001: qwen2's decode step cut after layer 0 on two PTX devices, compiled with no device, has two
/// stages, one `PointToPoint` row per crossing value and no link: every stage input a boundary names is
/// bound by exactly one transfer, whose row is in the plan.
///
/// Mutation: make `plan_stage_communication` skip its second row. The boundary that row carried is
/// bound by neither a link nor a row, and `compile_staged` names it (`UnboundBoundary`).
#[test]
fn a_qwen2_pipeline_on_two_ptx_devices_binds_every_boundary_by_a_row() {
    let graph = qwen2_decode(LAYERS);
    let program = compile_on(&graph, &targets(Backend::Nvptx, &[GIB, GIB]), &pipeline(2))
        .unwrap_or_else(|error| panic!("a 2-device PTX pipeline compiles host-only: {error}"));

    let stages: Vec<_> = program.stages().collect();
    assert_eq!(stages.len(), 2);
    assert_eq!(stages[0].1, DeviceId(0));
    assert_eq!(stages[1].1, DeviceId(1));

    let boundaries = crate::test_support::split_after_layers(&graph, &[0])
        .expect("the graph splits after layer 0")
        .boundaries;
    assert!(!boundaries.is_empty(), "the cut crosses values");
    assert!(program.links().is_empty(), "no hop stays on one device");
    assert_eq!(program.transfers().len(), boundaries.len());
    let rows = &program.comm().rows;
    assert_eq!(rows.len(), boundaries.len(), "one row per boundary");
    for row in rows {
        assert_eq!(row.kind, CollectiveKind::PointToPoint);
        assert_eq!(row.participants, [DeviceId(0), DeviceId(1)]);
    }

    // Every boundary input of the consuming stage is some transfer's import, and no row is shared.
    let consumer = stages[1].2;
    let imported: Vec<_> = program.transfers().iter().map(|t| t.import).collect();
    for boundary in &boundaries {
        let input = consumer
            .boundary_input(boundary.id)
            .unwrap_or_else(|| panic!("stage 1 has no input for boundary {:?}", boundary.id));
        assert!(
            imported.contains(&input),
            "boundary {:?} is bound by no transfer",
            boundary.id
        );
    }
    let mut row_ids: Vec<_> = program.transfers().iter().map(|t| t.row).collect();
    row_ids.sort();
    row_ids.dedup();
    assert_eq!(row_ids.len(), rows.len(), "each transfer owns its row");
}

/// SC-004: a 2-device pipeline on wgpu or on ROCm is refused with `Collective(PointToPoint)` naming the
/// value that would cross, and no stage is compiled: the planner's compile counter does not move. The
/// same graph on PTX moves it, so the counter is live.
///
/// Mutation: move `admit_links` after the stage compile loop in `compile_staged`; the counter moves for
/// wgpu and ROCm and the zero-work assertion goes red.
#[test]
fn a_two_device_pipeline_on_wgpu_or_rocm_is_refused_before_any_stage_is_compiled() {
    let graph = qwen2_decode(LAYERS);
    let before = per_eqn_plans_inside_compile();
    compile_on(&graph, &targets(Backend::Nvptx, &[GIB, GIB]), &pipeline(2))
        .expect("PTX carries the boundary");
    assert!(
        per_eqn_plans_inside_compile() > before,
        "a PTX compile plans equations, so the counter is live"
    );

    for backend in [Backend::SpirvVulkan, Backend::AmdGcn(AmdArch::gfx1151())] {
        let before = per_eqn_plans_inside_compile();
        let error = compile_on(&graph, &targets(backend, &[GIB, GIB]), &pipeline(2))
            .map(|_| ())
            .expect_err("no link carries the boundary");
        let StagedCompileError::Refused(refusal) = error else {
            panic!("{backend:?}: expected a refusal, got {error:?}");
        };
        assert_eq!(
            refusal.missing,
            Capability::Collective(CollectiveKind::PointToPoint),
            "{backend:?}"
        );
        assert_eq!(refusal.target, backend);
        assert!(
            graph.eqns.iter().any(|eqn| eqn.out == refusal.eqn),
            "the refusal names an equation of the uncut graph"
        );
        assert_eq!(
            per_eqn_plans_inside_compile(),
            before,
            "{backend:?}: a stage was compiled before the refusal"
        );
    }
}

/// SC-004 (control): the same two wgpu stages on one device compile, so the refusal above is the
/// device crossing and not the cut.
#[test]
fn the_same_cut_on_one_wgpu_device_compiles() {
    let graph = qwen2_decode(LAYERS);
    let partition = Partition {
        experts: ExpertPlacement::AllResident,
        devices: DevicePlacement::Pipeline {
            after_layers: vec![0],
            devices: vec![DeviceId(0), DeviceId(0)],
        },
    };
    let program = compile_on(&graph, &targets(Backend::SpirvVulkan, &[GIB]), &partition)
        .expect("one device carries its own boundary");
    assert!(program.comm().rows.is_empty());
    assert!(program.transfers().is_empty());
    assert!(!program.links().is_empty());
}

/// SC-006: a three-device pipeline with a boundary two stages read and weights of different sizes.
/// Every hop has its export and import and the bytes its storage plans; each device's footprint is the
/// weights it holds, the arena its stages retain and the boundary buffers delivered to it.
///
/// Mutations: omit the second consumer's transfer (`devices::communication`'s row for `t0` to stage 2)
/// and the map is incomplete; omit device 2's arena from `footprints` and its footprint is short.
#[test]
fn a_multi_consumer_boundary_has_a_complete_map_and_an_exact_per_device_footprint() {
    let graph = three_device_fixture();
    let program = compile_on(
        &graph,
        &targets(Backend::Nvptx, &[GIB, GIB, GIB]),
        &pipeline(3),
    )
    .unwrap_or_else(|error| panic!("the fixture compiles: {error}"));
    let stages: Vec<_> = program.stages().collect();
    assert_eq!(stages.len(), 3);

    // t0 is read by stage 1 and stage 2, t1 by stage 2: three hops, three rows.
    let transfers = program.transfers();
    assert_eq!(transfers.len(), 3, "t0 -> 1, t0 -> 2, t1 -> 2");
    assert_eq!(program.comm().rows.len(), 3);
    let t0_hops: Vec<_> = transfers
        .iter()
        .filter(|t| t.from == StageId(0))
        .map(|t| t.to)
        .collect();
    assert_eq!(t0_hops, [StageId(1), StageId(2)], "both consumers of t0");
    for transfer in transfers {
        let producer = stages[transfer.from.0 as usize].2;
        let consumer = stages[transfer.to.0 as usize].2;
        let export = producer
            .exports()
            .iter()
            .find(|export| export.id == transfer.id)
            .expect("the producer exports the boundary");
        assert_eq!(export.value, transfer.export);
        assert_eq!(
            consumer.boundary_input(transfer.id),
            Some(transfer.import),
            "the consumer's import of boundary {:?}",
            transfer.id
        );
        assert_eq!(
            producer.storage().storage(export.value).buffer_storage(),
            transfer.storage
        );
        assert_eq!(
            consumer.storage().storage(transfer.import).buffer_storage(),
            transfer.storage
        );
        let row = program
            .comm()
            .rows
            .iter()
            .find(|row| row.id == transfer.row)
            .expect("the transfer's row exists");
        assert_eq!(row.byte_len as u64, transfer.bytes);
    }
    // t0 is 4 f32 (16 bytes), t1 is [1, 32] f32 (128 bytes).
    let bytes: Vec<u64> = transfers.iter().map(|t| t.bytes).collect();
    assert_eq!(bytes, [16, 16, 128]);

    // Footprints: weights are 16 / 512 / 128 bytes; arenas come from the programs themselves.
    let footprint: HashMap<DeviceId, _> = program.footprints().iter().copied().collect();
    assert_eq!(footprint.len(), 3);
    let arena = |stage: usize| stages[stage].2.buffer_plan().arena_bytes() as u64;
    assert_eq!(footprint[&DeviceId(0)].dense_constants, 16);
    assert_eq!(footprint[&DeviceId(1)].dense_constants, 512);
    assert_eq!(footprint[&DeviceId(2)].dense_constants, 128);
    // Stage 0 also holds the activation slot x (16 bytes); only the consumers receive boundaries.
    assert_eq!(footprint[&DeviceId(0)].activations, 16 + arena(0));
    assert_eq!(footprint[&DeviceId(1)].activations, arena(1));
    assert_eq!(footprint[&DeviceId(2)].activations, arena(2));
    assert_eq!(footprint[&DeviceId(0)].communication, 0);
    assert_eq!(footprint[&DeviceId(1)].communication, 16);
    assert_eq!(footprint[&DeviceId(2)].communication, 16 + 128);
    for (device, categories) in &footprint {
        assert_eq!(categories.state, 0, "{device:?}");
    }
}

/// SC-006, POOT-725: an I32 value one PTX device computes and exports and another reads only as a gather
/// index crosses the cut in one representation (dense I32 words): the cut compiles and the boundary is a
/// transfer. Before, the consumer's float-mirror read made the schemas differ and the cut was refused.
///
/// Mutation: make `value_storage_kind` store an I32 value no `Binary` consumes as F32; the schemas differ
/// again and this goes red with a `LinkSchema` refusal.
#[test]
fn an_i32_boundary_read_only_as_an_index_crosses_devices() {
    let b = Builder::new();
    let token = b.slot(Slot::Token, TensorType::new(vec![4], DType::I32));
    let table = b.constant("table", TensorType::f32(vec![8, 4]));
    let (ids, rows);
    {
        let _layer = b.layer_scope(0);
        ids = b.binary(BinOp::Add, token, token);
        rows = b.gather(table, 0, ids);
    }
    let k_in = b.state_input("k", TensorType::f32(vec![4, 4]), StateRole::Recurrent);
    let (k_out, out);
    {
        let _layer = b.layer_scope(1);
        k_out = b.binary(BinOp::Add, k_in, rows);
        out = b.gather(table, 0, ids);
    }
    let graph = b.finish_with_state(out, &[(k_in, k_out)]);
    let before = per_eqn_plans_inside_compile();
    let program = compile_on(&graph, &targets(Backend::Nvptx, &[GIB, GIB]), &pipeline(2))
        .unwrap_or_else(|error| panic!("an I32 boundary has one representation: {error}"));
    assert!(
        program
            .transfers()
            .iter()
            .any(|transfer| transfer.aval.dtype == DType::I32),
        "the ids cross as a transfer"
    );
    assert!(per_eqn_plans_inside_compile() > before);
}

/// SC-003: a device whose footprint exceeds its budget refuses the placement, naming the device, the
/// bytes it needs and the bytes it has. The budget is the heaviest device's exact footprint minus one.
///
/// Mutation: drop the `admit_capacity` call in `compile_staged`; the placement is admitted and this
/// goes red.
#[test]
fn a_device_over_its_budget_refuses_the_placement() {
    let graph = three_device_fixture();
    let roomy = compile_on(
        &graph,
        &targets(Backend::Nvptx, &[GIB, GIB, GIB]),
        &pipeline(3),
    )
    .expect("roomy devices admit the fixture");
    let totals: Vec<u64> = roomy
        .footprints()
        .iter()
        .map(|(_, categories)| categories.checked_total().unwrap())
        .collect();
    let (heaviest, &needed) = totals
        .iter()
        .enumerate()
        .max_by_key(|&(_, total)| *total)
        .unwrap();
    // Exactly enough is admitted; one byte less is not.
    let mut budgets = vec![GIB; 3];
    budgets[heaviest] = needed;
    compile_on(&graph, &targets(Backend::Nvptx, &budgets), &pipeline(3))
        .expect("a device with exactly its footprint fits");
    budgets[heaviest] = needed - 1;
    match compile_on(&graph, &targets(Backend::Nvptx, &budgets), &pipeline(3)) {
        Err(StagedCompileError::Capacity(AccountingError::CapacityExceeded {
            device,
            required,
            usable,
        })) => {
            assert_eq!(device, DeviceId(heaviest as u32));
            assert_eq!(required, needed);
            assert_eq!(usable, needed - 1);
        }
        other => panic!("expected CapacityExceeded, got {:?}", other.map(|_| ())),
    }
}

/// A device with no measured budget is refused, not read as unlimited.
#[test]
fn a_device_with_an_unknown_budget_is_refused() {
    let graph = three_device_fixture();
    let mut devices = targets(Backend::Nvptx, &[GIB, GIB, GIB]);
    let unknown = Target {
        backend: Backend::Nvptx,
        caps: default_caps_for(Backend::Nvptx),
    };
    assert_eq!(unknown.caps.memory_budget_bytes, None);
    devices = TargetSet::new(vec![
        (DeviceId(0), devices.devices()[0].1),
        (DeviceId(1), unknown),
        (DeviceId(2), devices.devices()[2].1),
    ])
    .unwrap();
    match compile_on(&graph, &devices, &pipeline(3)) {
        Err(StagedCompileError::UnknownBudget(DeviceId(1))) => {}
        other => panic!("expected UnknownBudget(1), got {:?}", other.map(|_| ())),
    }
}

/// SC-003: one store key bound to two different tensors in two stages is a duplicate owner, refused with
/// the owner and both rows; the same key and type in two stages is one replicated owner and compiles.
///
/// Mutation: remove the `DuplicateOwner` check from `validate_placement_set`; the conflicting graph
/// compiles and this goes red.
#[test]
fn one_store_key_naming_two_different_tensors_is_a_duplicate_owner() {
    let build = |second: Vec<usize>| {
        let b = Builder::new();
        let x = b.slot(Slot::Activation, TensorType::f32(vec![4]));
        let hidden;
        {
            let _layer = b.layer_scope(0);
            let w = b.constant("w", TensorType::f32(vec![4]));
            hidden = b.binary(BinOp::Mul, x, w);
        }
        let out;
        {
            let _layer = b.layer_scope(1);
            let w = b.constant("w", TensorType::f32(second));
            out = b.binary(BinOp::Add, hidden, w);
        }
        b.finish(out)
    };
    let devices = targets(Backend::Nvptx, &[GIB, GIB]);

    let tied =
        compile_on(&build(vec![4]), &devices, &pipeline(2)).expect("a tied weight is shared");
    // One owner, replicated on both devices: both account its 16 bytes.
    let footprint: HashMap<DeviceId, _> = tied.footprints().iter().copied().collect();
    assert_eq!(footprint[&DeviceId(0)].dense_constants, 16);
    assert_eq!(footprint[&DeviceId(1)].dense_constants, 16);

    match compile_on(&build(vec![1]), &devices, &pipeline(2)) {
        Err(StagedCompileError::Placement(PlacementError::DuplicateOwner { indices, .. })) => {
            assert_eq!(indices.len(), 2)
        }
        other => panic!("expected DuplicateOwner, got {:?}", other.map(|_| ())),
    }
}

/// The derivation builds no `MissingOwner` or `OffStage` placement: every weight of a real model, cut
/// over three devices, is placed by `place_weights` and admitted, and each device's footprint holds the
/// weights its own stages read. These two variants are structural invariants of that derivation, not
/// `compile_staged` refusals; they are checked, with placements the derivation would never build, at
/// `validate_stage_placement` (`multi_device_stage_placement_rejects_an_owner_placed_off_its_stage`).
#[test]
fn the_weight_derivation_of_a_real_model_places_every_owner_on_its_stages() {
    let graph = qwen2_decode(LAYERS);
    let program = compile_on(
        &graph,
        &targets(Backend::Nvptx, &[GIB, GIB, GIB]),
        &pipeline(3),
    )
    .expect("every weight is placed on a stage that reads it");
    for (device, categories) in program.footprints() {
        let stage = program
            .stages()
            .find(|(_, stage_device, _)| stage_device == device)
            .expect("each device runs a stage");
        let weights: u64 = stage
            .2
            .graph()
            .consts
            .iter()
            .filter(|&&value| {
                stage.2.graph().meta(value).storage == poot_graph_ir::Storage::Const
                    && stage.2.graph().meta(value).name.is_some()
            })
            .map(|&value| {
                crate::staged::devices::value_bytes(
                    &stage.2.graph().aval(value).clone(),
                    stage.2.storage().storage(value).buffer_storage(),
                )
            })
            .sum();
        // The footprint also holds the computed tables (rope), so it is at least the named weights.
        assert!(
            categories.dense_constants >= weights,
            "{device:?} holds the weights its stage reads"
        );
    }
}
