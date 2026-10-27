//! POOT-582: Qwen3-MoE with its routed experts paged through a pool smaller than the table
//! (`Driver::with_expert_pool`, ADR-0072), through the production path: the registry's fixture written as a
//! checkpoint, loaded by `ModelHandle::load`, compiled by `compile_staged` with the full pass pipeline, run
//! by the executor's staged walk with the residency manager refilling pool rows at each sync point.
//!
//! Pooling is structural (`pool_experts` finds the family's `components::moe` sites), so this file holds
//! the proof rows only: the family has no pool type and no pool entry point of its own.

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use poot_graph_plan::{
    CompileLimits, CompileOptions, WeightFormats, bind_packed_weights, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::StageId, staged::TargetSet,
    staged::compile_staged,
};
use poot_llm::driver::{
    BackendChoice, Driver, DriverOptions, GenerateRequest, Layout, LoadOptions, ModelHandle,
    PreparedSetLimits, Serving, open_executor, program_retention,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::model::{StepShape, StepSpec};
use poot_models::registry::{RawConfig, Registry};
use poot_runtime_common::DeviceBackend;
use poot_target::{AmdArch, Backend, DeviceCaps};
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const FAMILY: &str = "qwen3moe";
const VOCAB: usize = 48;
/// An end-of-sequence id outside the vocabulary: the run never ends early on a sampled token.
const NO_EOS: u32 = 999;
const CAPACITY: usize = 32;
/// The fixture has eight experts and routes two per token: four pool rows are half the table and serve
/// two tokens per step, so a prompt runs as chunks and the manager evicts.
const POOL_ROWS: usize = 4;
const EXPERTS: usize = 8;
const _: () = assert!(POOL_ROWS < EXPERTS);

const PROMPT: [u32; 7] = [3, 17, 40, 8, 25, 11, 2];
/// The family's recorded 24 greedy tokens for [`PROMPT`] (POOT-738: the pre-driver engine's CPU generate
/// on this fixture, equal to the unpooled driver on wgpu and ROCm).
const TOKENS: [u32; 24] = [
    28, 28, 28, 28, 28, 38, 27, 15, 34, 15, 42, 18, 15, 16, 38, 27, 44, 21, 28, 38, 40, 15, 38, 40,
];

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

/// A word-level tokenizer over the fixtures' 48-word vocabulary.
fn tokenizer() -> Tokenizer {
    let mut words: Vec<String> = ('a'..='z').map(String::from).collect();
    words.extend(('0'..='9').map(String::from));
    words.extend((0..VOCAB - words.len()).map(|i| format!("p{i}")));
    let vocab = words
        .into_iter()
        .enumerate()
        .map(|(i, w)| (w, i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("a".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
}

/// The family's registry fixture, with its end of sequence unreachable, written as a checkpoint
/// directory and loaded.
fn load() -> (Arc<ModelHandle>, poot_test_util::UniqueTempPath) {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == FAMILY)
        .unwrap_or_else(|| panic!("no builtin family {FAMILY}"));
    let fixture = poot_models::test_support::family_fixtures::base(entry.family);
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    let dir = poot_test_util::unique_temp_path("pooled-qwen3moe");
    write_checkpoint(&dir, &config, &fixture.store);
    let handle = ModelHandle::load(&dir, &registry, &LoadOptions::default()).unwrap();
    (Arc::new(handle), dir)
}

fn write_checkpoint(
    dir: &Path,
    config: &serde_json::Value,
    store: &poot_quant::weights::WeightStore,
) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    poot_quant::test_support::safetensors::write(&dir.join("model.safetensors"), store);
}

fn options() -> DriverOptions {
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    DriverOptions {
        kv_dtype: None,
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(CAPACITY),
        prefill_chunk: nz(8),
        trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(64)),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
        },
        charge: program_retention,
        draft: poot_llm::driver::DraftPolicy::Off,
        hidden_window: None,
    }
}

fn generate(driver: &mut Driver<Serving>, prompt: &[u32], max_new: usize) -> Vec<u32> {
    driver
        .generate(
            GenerateRequest {
                prompt: prompt.to_vec(),
                max_new,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: false,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap()
        .tokens
}

/// SC-001, tokens: the family decodes its recorded literals through a pool of [`POOL_ROWS`] `< E` rows
/// per selection; the pool evicts (`evictions > 0`) and the executor walked the sync points.
/// Mutation: in `ExpertResidency`, skip the refill of a miss (answer the rows without writing the
/// expert); the pool serves another expert's bytes and the tokens diverge from the literals.
fn pooled_tokens_equal_the_recorded_literals(choice: BackendChoice, backend: DeviceBackend) {
    let (handle, dir) = load();
    let Some(executor) =
        poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice))
    else {
        return;
    };
    let mut driver = Driver::new(handle, executor, options())
        .unwrap()
        .with_expert_pool(nz(POOL_ROWS))
        .unwrap()
        .serve(Layout::Contiguous);
    let got = generate(&mut driver, &PROMPT, 24);
    assert_eq!(got, TOKENS, "pooled {FAMILY} on {choice:?}");
    let counters = driver.stats().residency;
    eprintln!("{FAMILY} residency counters on {choice:?}: {counters:?}");
    assert!(
        counters.refills > 0 && counters.evictions > 0,
        "{counters:?}"
    );
    let stats = driver.executor_stats();
    assert!(stats.syncs > 0 && stats.sync_read_bytes > 0, "{stats:?}");
    drop(dir);
}

#[test]
fn pooled_qwen3moe_tokens_equal_the_recorded_literals_wgpu() {
    pooled_tokens_equal_the_recorded_literals(BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

/// Without an AMD GPU the device does not open and the row skips.
#[test]
fn pooled_qwen3moe_tokens_equal_the_recorded_literals_rocm() {
    pooled_tokens_equal_the_recorded_literals(BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn pooled_qwen3moe_tokens_equal_the_recorded_literals_ptx() {
    pooled_tokens_equal_the_recorded_literals(BackendChoice::Ptx, DeviceBackend::Ptx);
}

/// The contraction-epilogue and pointwise fusion passes ran in `what`.
fn assert_fusion_passes<'a>(label: &str, what: &str, passes: impl Iterator<Item = &'a str>) {
    let ran: Vec<&str> = passes.collect();
    for pass in ["fuse_contraction_epilogues", "fuse"] {
        assert!(
            ran.contains(&pass),
            "{label}: {FAMILY} {what} skipped {pass}; it ran {ran:?}"
        );
    }
}

/// SC-001, pass list: `compile_staged` for each production preset, with the inputs the driver compiles
/// with, runs the contraction-epilogue and pointwise fusion passes both over the resident program and
/// in the pooled stages around each sync point (the router's stage and the stage that reads the pooled
/// rows). (`tile_matmuls` is gone: the planner makes that choice; these are the passes that took its
/// place.) Mutation: in `compile`, skip `fuse_contraction_epilogues` for a graph with a routed-expert
/// site; the resident program's pass list lacks it and the row goes red naming it.
#[test]
fn pooled_qwen3moe_stages_run_the_contraction_and_pointwise_passes() {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == FAMILY)
        .unwrap_or_else(|| panic!("no builtin family {FAMILY}"));
    let fixture = poot_models::test_support::family_fixtures::base(entry.family);
    let raw = RawConfig::HfJson {
        config: &fixture.config,
        generation: None,
        preprocessor: None,
    };
    let model = (entry.build)(&raw, &fixture.store).unwrap();
    let shape = StepShape::new(NonZeroUsize::MIN, NonZeroUsize::MIN, nz(CAPACITY));
    let graph = model
        .model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let formats = WeightFormats::from_weight_map(model.model.weights());
    let graph = bind_packed_weights(&graph, &formats).unwrap();
    let presets = [
        ("wgpu", Backend::SpirvVulkan, DeviceCaps::wgpu_rdna3_igpu()),
        (
            "rocm",
            Backend::AmdGcn(AmdArch::gfx1151()),
            DeviceCaps::rocm_default(),
        ),
        ("ptx", Backend::Nvptx, DeviceCaps::ptx_default()),
    ];
    for (label, backend, caps) in presets {
        let target = poot_graph_plan::Target::Gpu { backend, caps };
        let device = poot_graph_plan::multi_device::topology::DeviceId(0);
        let compile = |experts| {
            compile_staged(
                &graph,
                &TargetSet::single(device, target),
                &Partition {
                    experts,
                    devices: DevicePlacement::Single(device),
                },
                &CompileOptions::new(CompileLimits::STANDARD),
            )
            .unwrap_or_else(|e| panic!("{label}: {FAMILY} compiles: {e}"))
        };
        // Resident, the routed selection is in the one program `compile` receives.
        let resident = compile(ExpertPlacement::AllResident);
        let (_, _, program) = resident.stages().next().unwrap();
        assert_fusion_passes(label, "the resident program", program.passes());
        let staged = compile(ExpertPlacement::Pooled {
            slots_per_layer: POOL_ROWS,
        });
        assert!(
            !staged.sync_points().is_empty(),
            "{label}: the pooled program is cut at the pool's sync points"
        );
        // Each sync point cuts the step at a routed selection: the stage before it holds the router, the
        // stage after it reads the pooled rows (the routed experts' contractions). Both must have run
        // the contraction and pointwise fusion passes.
        for sync in staged.sync_points() {
            for stage in [sync.after, StageId(sync.after.0 + 1)] {
                let (_, _, program) = staged
                    .stages()
                    .find(|(id, _, _)| *id == stage)
                    .unwrap_or_else(|| panic!("{label}: no stage {stage:?} around {sync:?}"));
                assert_fusion_passes(label, &format!("pooled stage {stage:?}"), program.passes());
            }
        }
    }
}
