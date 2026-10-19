//! The traced-model rows of the ROCm parity table (SC-002/SC-003): the qwen2 decode and prefill fixtures and
//! their softcapped twins run on ROCm through the executor contract and every replayed step matches the CPU
//! oracle within tolerance (ADR-0101 tier 1). The fixtures trace `poot-models`, which sits above the device
//! crate, so these rows live here; the model-free table is `poot-rocm-device/tests/parity.rs`.
#![cfg(feature = "rocm")]

use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::{Device, Engine, Executor};
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;

#[test]
fn parity_table_matches_the_oracle_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
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
/// `attn_logit_softcap = Some(50.0)` and compiled for ROCm match `eval(g)` of that same softcapped
/// trace within the parity tolerance, at every replayed step. The fixtures' scores reach the softcap
/// regime: the softcapped oracle differs from the uncapped trace's by far more than that tolerance, so
/// the device output cannot match it without applying the softcap. Reproduction: the pre-card flash
/// tracer arms (`trace_*_flash`) dropped the softcap. Mutation: make the qwen2 tracers call attention
/// without `cfg.attn_logit_softcap`; the softcapped and uncapped traces coincide and this row goes red.
#[test]
fn softcapped_qwen2_matches_the_oracle_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
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
