//! POOT-1228 SC-002: the second out-of-tree family ([`parallel`]: parallel residual and per-head
//! interleaved fused QKV) serves a row with no production edit: this crate reaches the product through
//! its public surface only.
//!
//! - Host half (model-free): the family registers, loads from a written checkpoint, compiles Prefill and
//!   Decode for the wgpu, ROCm and PTX reference targets, greedy-decodes the pinned tokens on the CPU
//!   oracle (and the option that makes it parallel decides them), and a driver over a fake device
//!   prepares and generates under the common planner.
//! - Device half: `Driver::generate` on a real device equals the oracle's tokens.

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use poot_executor::Engine;
use poot_graph_plan::{
    CompileLimits, CompileOptions, Target, WeightFormats, bind_packed_weights, compile,
};
use poot_llm::driver::{
    BackendChoice, Driver, DriverOptions, GenerateRequest, Layout, LoadOptions, ModelHandle,
    open_executor,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::model::{Model, StepSpec};
use poot_models::registry::{Fixture, Registry};
use poot_runtime_common::DeviceBackend;
use poot_target::{AmdArch, Backend};
use poot_test_util::device_caps::default_caps_for;

use crate::parallel::{self, FAMILY, VOCAB};
use crate::{Counting, FakeDevice, driver_options, oracle_tokens, shape, write_checkpoint};

/// The prompt, as token ids.
const PROMPT: [u32; 6] = [3, 17, 8, 29, 5, 21];
/// How many tokens every row generates.
const NEW_TOKENS: usize = 8;
/// The greedy tokens of [`PROMPT`] on the CPU oracle, pinned: the fixture's weights are fixed bytes, so
/// a different token here is a different model.
const PINNED: [u32; NEW_TOKENS] = [35, 29, 29, 20, 29, 19, 29, 19];

fn registry() -> Registry {
    Registry::builtin_plus(&[parallel::ENTRY]).unwrap()
}

fn load(dir: &Path) -> ModelHandle {
    ModelHandle::load(dir, &registry(), &LoadOptions::default()).unwrap()
}

fn parallel_model(fixture: &Fixture) -> Box<dyn Model> {
    registry()
        .build(&fixture.raw(), &fixture.store)
        .unwrap()
        .model
}

fn request() -> GenerateRequest {
    GenerateRequest {
        prompt: PROMPT.to_vec(),
        max_new: NEW_TOKENS,
        sampler: Sampler::greedy(),
        stops: Vec::new(),
        ignore_eos: true,
    }
}

/// The oracle's greedy tokens after [`PROMPT`] on `fixture`, each with its lead over the runner-up.
fn oracle(fixture: &Fixture) -> Vec<(u32, f32)> {
    oracle_tokens(
        &*parallel_model(fixture),
        &fixture.store,
        &PROMPT,
        NEW_TOKENS,
        |p| p,
    )
}

/// The family registers through the public registry surface, loads from a written checkpoint through
/// `ModelHandle::load`, and its Prefill and Decode graphs compile for the wgpu, ROCm and PTX reference
/// targets with the interleaved fused QKV weight read whole.
#[test]
fn the_parallel_family_registers_loads_and_compiles_for_every_reference_target() {
    let fixture = parallel::fixture();
    let dir = poot_test_util::unique_temp_path("neox_parallel_checkpoint");
    write_checkpoint(&dir, &fixture);
    let handle = load(&dir);
    assert_eq!(handle.config().family, FAMILY);
    assert_eq!(handle.config().vocab, VOCAB);

    let model = registry().build(&fixture.raw(), handle.store()).unwrap();
    let targets = [
        (
            "wgpu",
            Target::Gpu {
                backend: Backend::SpirvVulkan,
                caps: default_caps_for(Backend::SpirvVulkan),
            },
        ),
        (
            "ROCm",
            Target::Gpu {
                backend: Backend::AmdGcn(AmdArch::gfx1151()),
                caps: default_caps_for(Backend::AmdGcn(AmdArch::gfx1151())),
            },
        ),
        (
            "PTX",
            Target::Gpu {
                backend: Backend::Nvptx,
                caps: default_caps_for(Backend::Nvptx),
            },
        ),
    ];
    let options = CompileOptions::new(CompileLimits::STANDARD);
    let formats = WeightFormats::from_weight_map(model.model.weights());
    for (name, target) in &targets {
        for (step, tokens) in [("prefill", 4), ("decode", 1)] {
            let graph = model
                .model
                .trace(
                    &StepSpec::step(shape(tokens)),
                    poot_graph_ir::TraceLimits::for_tests(),
                )
                .unwrap();
            let graph = bind_packed_weights(&graph, &formats).unwrap();
            compile(&graph, target, &options)
                .unwrap_or_else(|e| panic!("{step} does not compile for {name}: {e}"));
        }
    }
}

/// The oracle's 8 greedy tokens equal the pinned list, each leading its runner-up by a margin no
/// device rounding closes; the same checkpoint with the sequential residual (`use_parallel_residual`
/// off) decodes other tokens, so the option is what the pin sees. Mutation: drop the parallel option
/// from the family's layer (`NormPlacement::Parallel`); the tokens differ from the pin.
#[test]
fn the_oracle_decodes_the_pinned_tokens_and_the_parallel_residual_decides_them() {
    let fixture = parallel::fixture();
    let tokens = oracle(&fixture);
    for (i, &(token, lead)) in tokens.iter().enumerate() {
        assert!(lead > 0.05, "token {i} ({token}) leads by only {lead}");
    }
    let tokens: Vec<u32> = tokens.into_iter().map(|(token, _)| token).collect();
    assert_eq!(tokens, PINNED, "the oracle's greedy tokens");

    let mut sequential = parallel::fixture_config();
    sequential["use_parallel_residual"] = false.into();
    let other: Vec<u32> = oracle(&parallel::fixture_of(sequential))
        .into_iter()
        .map(|(token, _)| token)
        .collect();
    assert_ne!(
        other, PINNED,
        "the sequential residual decodes the parallel residual's tokens"
    );
}

/// A driver over the family on the fake device prepares its entries through the common planner and
/// generates: the real driver, engine and planner, no family branch anywhere.
#[test]
fn the_driver_prepares_and_generates_the_parallel_family_with_no_family_branch() {
    let dir = poot_test_util::unique_temp_path("neox_parallel_driver_checkpoint");
    write_checkpoint(&dir, &parallel::fixture());
    let loads = Rc::new(Cell::new(0));
    let executor = Counting {
        inner: Engine::new(FakeDevice::default()),
        loads: loads.clone(),
    };
    let options: DriverOptions = driver_options(4, 64);
    let mut driver = Driver::new(Arc::new(load(&dir)), Box::new(executor), options)
        .unwrap()
        .serve(Layout::Contiguous);
    let generation = driver
        .generate(request(), &mut |_: u32, _: &str| {
            GenerationControl::Continue(())
        })
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(generation.tokens.len(), NEW_TOKENS);
    assert!(loads.get() > 0, "the driver loaded no executable");
}

/// `Driver::generate` on `choice`'s device over the loaded checkpoint: the tokens equal the oracle's,
/// which are the pinned list. The device skips cleanly when absent (and fails when the lane requires it).
fn assert_generate_matches_the_oracle(choice: BackendChoice, backend: DeviceBackend) {
    let Some(executor) =
        poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice))
    else {
        return;
    };
    let fixture = parallel::fixture();
    let dir = poot_test_util::unique_temp_path("neox_parallel_device_checkpoint");
    write_checkpoint(&dir, &fixture);
    let oracle: Vec<u32> = oracle(&fixture)
        .into_iter()
        .map(|(token, _)| token)
        .collect();
    assert_eq!(oracle, PINNED);
    let mut driver = Driver::new(Arc::new(load(&dir)), executor, driver_options(8, 64))
        .unwrap()
        .serve(Layout::Contiguous);
    let generation = driver
        .generate(request(), &mut |_: u32, _: &str| {
            GenerationControl::Continue(())
        })
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        generation.tokens, oracle,
        "the device's greedy tokens differ from the oracle's"
    );
}

/// The device half on wgpu. Mutation: bind position 0 at every step of the driver's decode input; the
/// device's tokens diverge from the oracle's.
#[test]
fn parallel_generate_equals_the_oracle_tokens_wgpu() {
    assert_generate_matches_the_oracle(BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

/// The device half on ROCm (without an AMD GPU the device does not open and the row skips).
#[test]
fn parallel_generate_equals_the_oracle_tokens_rocm() {
    assert_generate_matches_the_oracle(BackendChoice::Rocm, DeviceBackend::Rocm);
}

/// The device half on PTX, in the pod batch.
#[test]
fn parallel_generate_equals_the_oracle_tokens_ptx() {
    assert_generate_matches_the_oracle(BackendChoice::Ptx, DeviceBackend::Ptx);
}
