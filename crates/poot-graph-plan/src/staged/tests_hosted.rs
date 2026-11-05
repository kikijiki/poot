//! POOT-1423: what `compile_staged` produces for a program with hosted gather sites, planned on every GPU
//! backend and checked on the CPU oracle (a device-indexed site chained stage by stage equals the unhosted
//! graph). The executor and the device lanes check the same programs run.
use std::collections::HashMap;

use poot_eval::test_support::stage_chain::{bits, chain};
use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::{
    BinOp, Builder, Graph, Scalar, Slot, SlotKey, Storage, TensorType, Traced, ValueId,
    packed_source_type,
};
use poot_quant::PackedWeight;
use poot_quant::format::WeightFormat;
use poot_target::{AmdArch, Backend};
use poot_tensor::{DType, HostTensor};
use poot_test_util::device_caps::default_caps_for;
use poot_weights::{WeightId, WeightRole};

use super::*;
use crate::CompileLimits;
use poot_graph_ir::WeightRef;

const EMBED: WeightId = WeightId::model(WeightRole::Embed);

fn target_for(backend: Backend) -> Target {
    Target::Gpu {
        backend,
        caps: default_caps_for(backend),
    }
}

fn gpu_backends() -> [Backend; 3] {
    [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ]
}

fn compile_for(g: &Graph, backend: Backend) -> Result<StagedProgram, StagedCompileError> {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target_for(backend)),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(CompileLimits::STANDARD),
    )
}

/// Flash-Next's per-layer table: `[320_001_536, 160]` IQ4_NL (26.8 GiB), read by one gather whose ids are the
/// token plus a head offset (a value the device computes). `device_index` false gathers by the token itself.
fn ple_like(rows: usize, device_index: bool) -> Graph {
    let descriptor = PackedWeight::try_new(WeightFormat::Iq4_Nl, [rows, 160]).unwrap();
    let b = Builder::for_tests();
    let token = b
        .slot(Slot::Token, TensorType::new(vec![4], DType::I32))
        .unwrap();
    let ids = if device_index {
        b.binary_scalar(BinOp::Add, token, Scalar::I32(3)).unwrap()
    } else {
        token
    };
    let sources: Vec<Traced> = descriptor
        .sources()
        .into_iter()
        .map(|role| {
            b.weight(
                WeightRef::whole(EMBED).with_source(role),
                packed_source_type(descriptor, role),
            )
            .unwrap()
        })
        .collect();
    let table = b.packed_dequant(&sources, descriptor).unwrap();
    let rows = b.gather(table, 0, ids).unwrap();
    let scale = b.constant("scale", TensorType::f32(vec![160])).unwrap();
    let scale = b.broadcast(scale, vec![4, 160]).unwrap();
    let out = b.binary(BinOp::Mul, rows, scale).unwrap();
    b.finish(out).unwrap()
}

/// SC-001/SC-002: a packed table far past what one buffer, or one kernel's 32-bit bit offsets, can address
/// plans on every GPU backend, because it is never a device constant: the stage graphs hold no constant of
/// the table, and the site is a device-indexed gather with one host sync.
///
/// Mutation: skip `hosted::apply` in `compile_staged`; the table stays a device constant and Vulkan refuses it
/// (`PackedSourceAddressing`), ROCm and PTX refuse it as oversized.
#[test]
fn a_table_past_every_device_limit_plans_on_every_backend_as_a_hosted_site() {
    let g = ple_like(320_001_536, true);
    for backend in gpu_backends() {
        let program = compile_for(&g, backend)
            .unwrap_or_else(|error| panic!("{backend:?}: the hosted table plans: {error}"));
        for (_, _, stage) in program.stages() {
            let graph = stage.graph();
            assert!(
                graph.consts().iter().all(|&id| {
                    graph
                        .meta(id)
                        .name()
                        .is_none_or(|name| !name.starts_with("w.embed"))
                }),
                "{backend:?}: no constant of the table reaches a stage"
            );
        }
        let [site] = program.hosted() else {
            panic!("{backend:?}: one hosted site")
        };
        assert!(
            matches!(site.index, IndexSource::Device(_)),
            "{backend:?}: {:?}",
            site.index
        );
        assert_eq!(program.stages().count(), 2, "{backend:?}: cut at the sync");
        let [sync] = program.sync_points() else {
            panic!("{backend:?}: one sync")
        };
        assert!(
            matches!(&sync.action, SyncAction::GatherRows(gather) if gather == site),
            "{backend:?}"
        );
        assert_eq!(sync.after, StageId(0));
        assert!(program.host_fed());
    }
}

/// A step-indexed site (the gather reads the token slot itself) is gathered at bind time: one stage, no sync,
/// and the program still reports itself host-fed, which is what keeps a driver from chaining it.
///
/// Mutation: make `StagedProgram::host_fed` ignore `hosted`; the driver would offer a chain the executor refuses.
#[test]
fn a_step_indexed_site_adds_no_cut_but_makes_the_program_host_fed() {
    let g = ple_like(100_000, false);
    for backend in gpu_backends() {
        let program =
            compile_for(&g, backend).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
        assert_eq!(program.stages().count(), 1, "{backend:?}");
        assert!(program.sync_points().is_empty(), "{backend:?}");
        let [site] = program.hosted() else {
            panic!("{backend:?}: one hosted site")
        };
        assert_eq!(
            site.index,
            IndexSource::Step(SlotKey::new(Slot::Token, None))
        );
        assert!(program.host_fed(), "{backend:?}");
        assert!(
            program.takes_input(&SlotKey::new(Slot::Token, None)),
            "{backend:?}: a step still binds the token the host gathers by"
        );
    }
}

/// A program with no gather-only table is not host-fed: the driver may chain it.
#[test]
fn a_program_with_no_hosted_site_is_not_host_fed() {
    let b = Builder::for_tests();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![4])).unwrap();
    let out = b.binary(BinOp::Mul, x, w).unwrap();
    let program = compile_for(&b.finish(out).unwrap(), Backend::SpirvVulkan).unwrap();
    assert!(!program.host_fed());
    assert!(program.hosted().is_empty());
}

/// A hosted gather on a device-computed index in a program spread over two devices is refused typed: the sync
/// reads one device's ids and writes that device's rows.
#[test]
fn a_device_indexed_site_across_devices_is_refused() {
    let g = ple_like(4096, true);
    let targets = TargetSet::new(vec![
        (DeviceId(0), target_for(Backend::SpirvVulkan)),
        (DeviceId(1), target_for(Backend::SpirvVulkan)),
    ])
    .unwrap();
    let error = compile_staged(
        &g,
        &targets,
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Pipeline {
                after_layers: vec![0],
                devices: vec![DeviceId(0), DeviceId(1)],
            },
        },
        &CompileOptions::new(CompileLimits::STANDARD),
    )
    .expect_err("the sync cannot span two devices");
    assert!(
        matches!(
            error,
            StagedCompileError::Unsupported(Unsupported::HostedIndexAcrossDevices)
        ),
        "{error}"
    );
}

/// A dense table over a device-computed index: `ids = token + 3`, rows gathered from `[vocab, width]`.
fn dense_device_indexed(vocab: usize, width: usize, tokens: usize) -> Graph {
    let b = Builder::for_tests();
    let token = b
        .slot(Slot::Token, TensorType::new(vec![tokens], DType::I32))
        .unwrap();
    let ids = b.binary_scalar(BinOp::Add, token, Scalar::I32(3)).unwrap();
    let table = b
        .weight(WeightRef::whole(EMBED), TensorType::f32(vec![vocab, width]))
        .unwrap();
    let rows = b.gather(table, 0, ids).unwrap();
    let scale = b.constant("scale", TensorType::f32(vec![width])).unwrap();
    let scale = b.broadcast(scale, vec![tokens, width]).unwrap();
    let out = b.binary(BinOp::Mul, rows, scale).unwrap();
    b.finish(out).unwrap()
}

/// The rows `rows` of the unhosted graph's table input, in order: what the executor copies at a sync.
fn gather_rows(table: &Value, ids: &[i32]) -> HostTensor {
    let host = table.as_host().expect("a dense table");
    let width = host.shape()[1];
    let words = host.as_f32().expect("an F32 table");
    let rows: Vec<f32> = ids
        .iter()
        .flat_map(|&id| words[id as usize * width..(id as usize + 1) * width].to_vec())
        .collect();
    HostTensor::f32(vec![ids.len(), width], rows)
}

/// SC-003 (oracle half): a program with a device-indexed hosted site, its stages chained on the CPU oracle
/// with the rows a host would copy at the sync (the ids the first stage exported), equals the unhosted graph
/// bit for bit.
///
/// Mutation: gather row `id + 1` in `gather_rows`' caller (an off-by-one row at the sync); the output differs.
#[test]
fn a_device_indexed_hosted_program_equals_the_unhosted_oracle_bit_for_bit() {
    let g = dense_device_indexed(32, 6, 4);
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in g.inputs() {
        let meta = g.meta(id);
        let value = match meta.storage {
            Storage::Slot(Slot::Token) => {
                HostTensor::i32(meta.aval.shape.clone(), vec![5, 0, 9, 12])
            }
            Storage::Const => {
                let seed = meta.name().unwrap().bytes().map(u64::from).sum::<u64>();
                let data = poot_test_util::fill(meta.aval.numel(), seed);
                HostTensor::f32(meta.aval.shape.clone(), data)
            }
            other => panic!("unexpected input {other:?}"),
        };
        inputs.insert(id, Value::from(value));
    }
    let whole = eval(&g, &inputs, EvalOptions::default()).expect("the unhosted graph evaluates");

    let program = compile_for(&g, Backend::SpirvVulkan).unwrap();
    let stages: Vec<&Graph> = program.stages().map(|(_, _, p)| p.graph()).collect();
    assert_eq!(stages.len(), 2);
    let table = inputs
        .iter()
        .find(|&(&id, _)| g.meta(id).name().as_deref() == Some("w.embed"))
        .map(|(_, value)| value.clone())
        .unwrap();
    let [site] = program.hosted() else {
        panic!("one site")
    };
    let mut supply = |key: &SlotKey, boundaries: &HashMap<poot_graph_ir::BoundaryId, Value>| {
        if key != &site.rows[0].slot {
            return None;
        }
        let IndexSource::Device(boundary) = &site.index else {
            panic!("a device-indexed site")
        };
        let ids = boundaries[boundary]
            .as_host()
            .unwrap()
            .as_i32()
            .unwrap()
            .to_vec();
        Some(Value::from(gather_rows(&table, &ids)))
    };
    let chained = chain(&g, &stages, &inputs, &mut supply);
    assert_eq!(bits(&chained.output), bits(&whole.output));
}

/// The typed refusal a device refuses with when it cannot hold a table that is not hosted (a tied head also
/// reads it): hosting changes nothing for such a table, and the planner still answers for it.
#[test]
fn a_tied_table_past_the_limit_is_still_a_typed_refusal() {
    let descriptor = PackedWeight::try_new(WeightFormat::Iq4_Nl, [320_001_536, 160]).unwrap();
    let b = Builder::for_tests();
    let token = b
        .slot(Slot::Token, TensorType::new(vec![4], DType::I32))
        .unwrap();
    let sources: Vec<Traced> = descriptor
        .sources()
        .into_iter()
        .map(|role| {
            b.weight(
                WeightRef::whole(EMBED).with_source(role),
                packed_source_type(descriptor, role),
            )
            .unwrap()
        })
        .collect();
    let table = b.packed_dequant(&sources, descriptor).unwrap();
    let rows = b.gather(table, 0, token).unwrap();
    let out = b
        .matmul(rows, b.transpose(table, vec![1, 0]).unwrap())
        .unwrap();
    let error = compile_for(&b.finish(out).unwrap(), Backend::SpirvVulkan)
        .expect_err("a tied table past the addressing is not hosted and not plannable");
    assert!(
        matches!(error, StagedCompileError::Compile(_)),
        "the compile of the stage answers for the table: {error}"
    );
    assert!(
        !matches!(error, StagedCompileError::Hosted(_)),
        "hosting is not what refused it"
    );
}

/// A tied model: the embedding gather and a head matmul read two consts (`w.embed`, `w.head`) of one stored
/// tensor, as the tracers declare them. The graph alone cannot tell.
fn tied_model() -> Graph {
    let b = Builder::for_tests();
    let token = b
        .slot(Slot::Token, TensorType::new(vec![1], DType::I32))
        .unwrap();
    let embed = b
        .weight(WeightRef::whole(EMBED), TensorType::f32(vec![64, 8]))
        .unwrap();
    let head = b
        .weight(
            WeightRef::whole(WeightId::model(WeightRole::Head)),
            TensorType::f32(vec![64, 8]),
        )
        .unwrap();
    let row = b.gather(embed, 0, token).unwrap();
    let out = b
        .matmul(row, b.transpose(head, vec![1, 0]).unwrap())
        .unwrap();
    b.finish(out).unwrap()
}

/// The weight map of the tied model: both ids view the one stored entry `t`.
fn tied_aliases(tied: bool) -> poot_weights::WeightAliases {
    use poot_quant::weights::{DenseWeight, WeightEntry, WeightKey, WeightStore};
    let entry = |bytes: Vec<u8>| {
        WeightEntry::Dense(
            DenseWeight::try_new(
                DType::F32,
                vec![64, 8],
                poot_quant::StoredBytes::from(bytes),
            )
            .unwrap(),
        )
    };
    let mut store = WeightStore::builder();
    store.insert("t", entry(vec![0; 64 * 8 * 4])).unwrap();
    store.insert("u", entry(vec![0; 64 * 8 * 4])).unwrap();
    let store = store.build();
    let mut map = poot_weights::WeightMap::builder(&store);
    map.map(
        EMBED,
        poot_weights::WeightView::Stored(WeightKey::from("t")),
    )
    .unwrap();
    let head = WeightId::model(WeightRole::Head);
    map.map(
        head,
        poot_weights::WeightView::Stored(WeightKey::from(if tied { "t" } else { "u" })),
    )
    .unwrap();
    map.build().aliases()
}

/// Alias-aware hosting: with the weight map's aliases a tied embedding is NOT hosted (the device holds its
/// bytes under `w.head` anyway, and hosting would only unchain the decode); an untied table with the same graph
/// still is. Without the aliases the planner cannot know, so it hosts.
///
/// Mutation: ignore the aliases in `host_gather_tables` (skip the `retain`); the tied program hosts and
/// reports itself host-fed.
#[test]
fn a_tied_embedding_is_not_hosted_and_an_untied_one_is() {
    let g = tied_model();
    let compile = |aliases| {
        compile_staged(
            &g,
            &TargetSet::single(DeviceId(0), target_for(Backend::SpirvVulkan)),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(CompileLimits::STANDARD).with_aliases(aliases),
        )
        .expect("compiles")
    };
    let tied = compile(tied_aliases(true));
    assert!(
        tied.hosted().is_empty() && !tied.host_fed(),
        "a tied table stays resident and chainable"
    );
    // The tied program is the one `compile` makes of the traced graph: nothing was rewritten.
    let plain = crate::compile(
        &g,
        &target_for(Backend::SpirvVulkan),
        &CompileOptions::new(CompileLimits::STANDARD),
    )
    .unwrap();
    let (_, _, stage) = tied.stages().next().unwrap();
    assert_eq!(stage.planned().count(), plain.planned().count());
    assert_eq!(
        stage.passes().collect::<Vec<_>>(),
        plain.passes().collect::<Vec<_>>()
    );
    let untied = compile(tied_aliases(false));
    assert_eq!(untied.hosted().len(), 1, "an untied table is hosted");
    assert!(untied.host_fed());
}
