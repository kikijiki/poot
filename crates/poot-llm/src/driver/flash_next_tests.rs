//! POOT-1423 SC-001 on the real Qwen3.8-Flash-Next tensor directory (arch `qwen4exp`): the decode step planned on
//! every GPU backend hosts its per-layer table. The checkpoint is the three sparse shard files of
//! [`poot_models::test_support::flash_next`]: no tensor byte is read.

use std::num::NonZeroUsize;

use poot_graph_plan::multi_device::topology::DeviceId;
use poot_graph_plan::staged::{
    DevicePlacement, ExpertPlacement, Partition, SyncAction, TargetSet, compile_staged,
};
use poot_graph_plan::{CompileLimits, CompileOptions, IndexSource, Target, WeightFormats};
use poot_models::model::{StepShape, StepSpec};
use poot_models::registry::Registry;
use poot_models::test_support::flash_next::SparseSet;
use poot_target::{Backend, DeviceCaps};

use super::{LoadOptions, ModelHandle};

/// POOT-1423 SC-001: on the real tensor directory (no tensor byte), the compiled programs of the decode step on
/// ROCm, PTX and wgpu/Vulkan hold no constant of `w.l1.ple.table` (the 26.82 GiB per-layer table,
/// `per_layer_token_embd.weight`) and exactly one device-indexed hosted gather names it: the ids are the n-gram
/// hash the device computes from the token and the history state, exported by the first stage and read at one
/// host sync. The vocabulary table is hosted by token at bind time.
///
/// Mutation: drop the `Storage::Device` arm of the index classification in `hosted_gather`; the per-layer
/// table is no longer hosted, its constant is back in a stage and the first assertion goes red.
#[test]
fn flash_next_per_layer_table_is_a_hosted_device_indexed_gather_on_every_backend() {
    let set = SparseSet::create();
    let registry = Registry::builtin().unwrap();
    let handle = ModelHandle::load(&set.paths[0], &registry, &LoadOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    let model = handle.model();
    let shape = StepShape::new(
        NonZeroUsize::MIN,
        NonZeroUsize::MIN,
        NonZeroUsize::new(256).unwrap(),
    );
    let graph = model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let graph = poot_graph_plan::bind_packed_weights(
        &graph,
        &WeightFormats::from_weight_map(model.weights()),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let targets = [
        ("ROCm", Backend::AmdGcn(poot_target::AmdArch::gfx1151())),
        ("PTX", Backend::Nvptx),
        ("wgpu", Backend::SpirvVulkan),
    ];
    for (name, backend) in targets {
        let caps = DeviceCaps {
            memory_budget_bytes: Some(128 << 30),
            ..poot_test_util::device_caps::default_caps_for(backend)
        };
        let program = compile_staged(
            &graph,
            &TargetSet::single(DeviceId(0), Target::Gpu { backend, caps }),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(CompileLimits::STANDARD),
        )
        .unwrap_or_else(|e| panic!("{name}: Flash-Next decode must compile: {e}"));
        for (_, _, stage) in program.stages() {
            let g = stage.graph();
            assert!(
                g.consts().iter().all(|&id| {
                    g.meta(id)
                        .name()
                        .is_none_or(|n| !n.starts_with("w.l1.ple.table"))
                }),
                "{name}: no stage holds a constant of the per-layer table"
            );
        }
        let device_indexed: Vec<_> = program
            .hosted()
            .iter()
            .filter(|site| matches!(site.index, IndexSource::Device(_)))
            .collect();
        let [site] = device_indexed[..] else {
            panic!(
                "{name}: exactly one device-indexed hosted gather, got {}",
                device_indexed.len()
            )
        };
        assert!(
            site.rows
                .iter()
                .all(|rows| rows.source.starts_with("w.l1.ple.table")),
            "{name}: the device-indexed site is the per-layer table: {:?}",
            site.rows
        );
        // SC-006: a decode token gathers 16 rows (two n-gram sizes x 8 heads) of 90 stored IQ4_NL bytes each: the
        // slot holds exactly the 1,440 stored bytes, never decoded rows, and the table's 26.82 GiB stay mapped.
        let [rows] = site.rows.as_slice() else {
            panic!("{name}: IQ4_NL has one source")
        };
        assert_eq!(
            (rows.aval.shape.clone(), rows.aval.dtype),
            (vec![16, 90], poot_tensor::DType::I8),
            "{name}: one decode token's rows as stored"
        );
        assert_eq!(rows.aval.numel(), 1_440, "{name}: bytes per PLE token");
        assert_eq!(
            site.table_bytes,
            320_001_536 * 90,
            "{name}: the mapped table"
        );
        assert!(
            matches!(program.sync_points(), [point]
                if matches!(&point.action, SyncAction::GatherRows(gather) if gather == site)),
            "{name}: one host sync copies the per-layer rows"
        );
        assert!(
            program
                .hosted()
                .iter()
                .any(|site| matches!(site.index, IndexSource::Step(_))),
            "{name}: the vocabulary table is hosted by token"
        );
    }
}
