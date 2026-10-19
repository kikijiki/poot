//! The traced-model rows of the raw-Vulkan parity table (SC-001): the qwen2 decode and prefill fixtures, their
//! softcapped twins and the head-layout decodes run through the executor contract and every replayed step
//! matches the CPU oracle within tolerance (ADR-0101 tier 1), plus the sixteen-replayed-steps capture row. The
//! fixtures trace `poot-models`, which sits above the device crate, so these rows live here; the model-free
//! table is `poot-vulkan-device/tests/parity.rs`.

use std::sync::Arc;

use poot_eval::test_support::StepFixture;
use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::{Device, Engine, Executor, ExecutorStats, NoSync, StepInputs};
use poot_executor_parity::Fixture;
use poot_graph_plan::Target;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_test_util::assert_close_rel;
use poot_vulkan_device::VulkanDevice;

/// The raw-Vulkan executor and the target it compiles for, or `None` (a skip) without a device.
fn vulkan_executor() -> Option<(Box<dyn Executor>, Target)> {
    let device = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new())?;
    let target = device.target();
    Some((Box::new(Engine::new(device)), target))
}

fn vulkan() -> Option<VulkanDevice> {
    open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new())
}

fn step_inputs(step: &[StepFixture]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for value in step {
        inputs.push(value.key.clone(), value.tensor.shape(), value.tensor.view());
    }
    inputs
}

/// Run `fixture` through the parity runner against the oracle's per-step expectation.
fn assert_matches_oracle(exec: &mut dyn Executor, target: Target, fixture: &Fixture) {
    let mut oracle = oracle_for(&fixture.graph, &fixture.store);
    poot_executor_parity::run_parity(exec, target, fixture, &mut oracle)
        .unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn parity_table_matches_the_oracle_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));

    let fixtures = [
        poot_models::test_support::fixtures::qwen2_decode_fixture(8),
        poot_models::test_support::fixtures::qwen2_prefill_fixture(3, 8),
    ];
    for fixture in &fixtures {
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// Card 557 SC-004 (dkernel K1): a qwen2-class decode and prefill traced with
/// `attn_logit_softcap = Some(50.0)` and compiled for raw Vulkan match `eval(g)` of that same softcapped
/// trace within the parity tolerance, at every replayed step. The fixtures' scores reach the softcap
/// regime: the softcapped oracle differs from the uncapped trace's by far more than that tolerance, so
/// the device output cannot match it without applying the softcap. Reproduction: the pre-card flash
/// tracer arms (`trace_*_flash`) dropped the softcap. Mutation: make the qwen2 tracers call attention
/// without `cfg.attn_logit_softcap`; the softcapped and uncapped traces coincide and this row goes red.
#[test]
fn softcapped_qwen2_matches_the_oracle_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));

    let softcap = Some(poot_models::test_support::fixtures::ATTN_LOGIT_SOFTCAP);
    let pairs = [
        (
            poot_models::test_support::fixtures::qwen2_softcap_decode_fixture(8, softcap),
            poot_models::test_support::fixtures::qwen2_softcap_decode_fixture(8, None),
        ),
        (
            poot_models::test_support::fixtures::qwen2_softcap_prefill_fixture(3, 8, softcap),
            poot_models::test_support::fixtures::qwen2_softcap_prefill_fixture(3, 8, None),
        ),
    ];
    for (softcapped, uncapped) in &pairs {
        let mut oracle = oracle_for(&softcapped.graph, &softcapped.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, softcapped, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
        poot_executor_parity::assert_softcap_is_visible(softcapped, uncapped);
    }
}

/// A qwen2 decode over the multi-head (`2 x 2`) and multi-query (`2 x 1`) query/key-value head layouts matches
/// the oracle at every step. (The grouped-query layout, `4 x 2`, is the parity table's `qwen2_decode` row.)
#[test]
fn decode_over_multi_head_and_multi_query_layouts_matches_the_oracle_on_vulkan() {
    let Some((mut exec, target)) = vulkan_executor() else {
        return;
    };
    for fixture in &poot_models::test_support::fixtures::head_layout_decode_fixtures(8) {
        assert_matches_oracle(exec.as_mut(), target, fixture);
    }
}

/// SC-005: a qwen2 decode entry on `Engine<VulkanDevice>` matches the CPU oracle at every one of 16
/// replayed steps, and capture is the default: `stats()` shows one recording, and the device itself one
/// native command-buffer recording, after the 16 steps.
///
/// Mutation: record every step (`needs_recording` in `Engine::step` always true); `recordings` is 16 and
/// the row goes red.
#[test]
fn sixteen_replayed_steps_record_once() {
    const STEPS: usize = 16;
    let Some(device) = vulkan() else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let fixture = poot_models::test_support::fixtures::qwen2_decode_fixture(STEPS);
    assert_eq!(fixture.steps.len(), STEPS);
    let mut oracle = oracle_for(&fixture.graph, &fixture.store);
    let exe = engine
        .load_weights(
            Arc::new(fixture.store.clone()),
            poot_executor_parity::fixture_weights(&fixture),
        )
        .unwrap();
    let entry = engine
        .add_entry(exe, &poot_executor_parity::staged(&fixture, target))
        .unwrap();
    for (i, step) in fixture.steps.iter().enumerate() {
        let want = oracle(step);
        let got = engine
            .step(exe, entry, &step_inputs(step), &mut NoSync)
            .unwrap()
            .to_host()
            .unwrap();
        assert_close_rel(got.as_f32().unwrap(), want.as_f32().unwrap(), 5e-3);
        let ExecutorStats {
            recordings,
            replays,
            ..
        } = engine.stats();
        assert_eq!(recordings, 1, "step {i}: one entry records once");
        assert_eq!(replays, i as u64 + 1, "step {i}: every step replays");
    }
    assert_eq!(
        engine.device().context().graphs_recorded(),
        1,
        "the device recorded one native command-buffer graph for {STEPS} steps"
    );
}
