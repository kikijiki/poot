// Moved in-crate (card 622): references a pub(crate)-only item (poot has no users, tests are not consumers).

//! Card 594 SC-001: the unpooled MoE families keep generating through the `Runner` after the pool engines and the
//! pool attachment are gone. The tiny untrained Mixtral and Qwen3-MoE fixtures decode a fixed prompt greedily, on the
//! CPU oracle and on wgpu, and every generated token id must equal the ids the same fixtures produced before the card
//! (pinned literals recorded on the commit before Card 594). A dropped or zeroed *routed* expert changes the greedy
//! path of these random-weight models, so the comparison fails on it. An expert the fixture never routes to is
//! invisible: expert 0 of the single sparse Qwen3-MoE layer is never selected on this prompt, so zeroing it
//! leaves those rows green (Mixtral's rows do see it). The mutation that turns all four rows red drops the
//! top-1 selected expert in `moe_grouped_prep`.
//!
//! wgpu decodes through a carried-KV path, so the unpooled `L == 1` decode graphs run: Qwen3-MoE through
//! `generate_kv_gpu_prefilled` (one batched prefill, then single-token decode) and Mixtral through
//! `generate_kv_gpu_cached` (Mixtral has no fixed-KV prefill, so every token is a masked single-token
//! step - `decode_arch::DecodeArch` dispatches it the same exhaustive way as every other arch, Card
//! 546b). The CPU oracle re-prefills every step.
//!
//! Each test skips when its fixture (under `POOT_MODELS_DIR`) or, for wgpu, a device is absent, as the other
//! checkpoint tests in this crate do; `POOT_REQUIRE_WGPU=1` turns a missing device into a failure.

use std::ops::ControlFlow;

use super::*;

const PROMPT: &str = "The capital of France is";
const MAX_NEW: usize = 8;

/// Prompt ids followed by the generated ids of `Runner::generate` on the `qwen3-moe-tiny` fixture, before Card 594.
const QWEN3_MOE_TINY_TOKENS: &[u32] = &[
    785, 6722, 315, 9625, 374, 54372, 100616, 5963, 115027, 119759, 59590, 87794, 24744,
];
/// Prompt ids followed by the generated ids of `Runner::generate` on the `mixtral-tiny` fixture, before Card 594.
const MIXTRAL_TINY_TOKENS: &[u32] = &[
    415, 5565, 302, 4843, 349, 17121, 17121, 29566, 294, 18144, 18882, 28500, 31232,
];

fn fixture(checkpoint: poot_test_util::Checkpoint) -> Option<Runner> {
    let dir = poot_test_util::model_path(checkpoint)?;
    Some(Runner::load(&dir).unwrap_or_else(|e| panic!("load {checkpoint}: {e}")))
}

fn qwen3_moe_tiny() -> Option<Runner> {
    fixture(poot_test_util::checkpoint!("qwen3-moe-tiny"))
}

fn mixtral_tiny() -> Option<Runner> {
    fixture(poot_test_util::checkpoint!("mixtral-tiny"))
}

#[test]
fn qwen3_moe_tiny_cpu_greedy_tokens_match_the_pre_card_594_ids() {
    let Some(runner) = qwen3_moe_tiny() else {
        return;
    };
    let toks = runner
        .generate(PROMPT, MAX_NEW, |_| ControlFlow::Continue(()))
        .expect("qwen3-moe-tiny CPU generate");
    eprintln!("qwen3-moe-tiny cpu tokens: {toks:?}");
    assert_eq!(toks, QWEN3_MOE_TINY_TOKENS, "qwen3-moe-tiny CPU greedy ids");
}

#[test]
fn qwen3_moe_tiny_wgpu_greedy_tokens_match_the_pre_card_594_ids() {
    let Some(runner) = qwen3_moe_tiny() else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no wgpu device ({e}); skipping");
            return;
        }
    };
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let toks = runner
        .generate_kv_gpu_prefilled(PROMPT, MAX_NEW, &mut engine, exe, |_| {
            ControlFlow::Continue(())
        })
        .expect("qwen3-moe-tiny wgpu generate");
    eprintln!("qwen3-moe-tiny wgpu tokens: {toks:?}");
    assert_eq!(
        toks, QWEN3_MOE_TINY_TOKENS,
        "qwen3-moe-tiny wgpu greedy ids"
    );
}

#[test]
fn mixtral_tiny_cpu_greedy_tokens_match_the_pre_card_594_ids() {
    let Some(runner) = mixtral_tiny() else {
        return;
    };
    let toks = runner
        .generate(PROMPT, MAX_NEW, |_| ControlFlow::Continue(()))
        .expect("mixtral-tiny CPU generate");
    eprintln!("mixtral-tiny cpu tokens: {toks:?}");
    assert_eq!(toks, MIXTRAL_TINY_TOKENS, "mixtral-tiny CPU greedy ids");
}

#[test]
fn mixtral_tiny_wgpu_greedy_tokens_match_the_pre_card_594_ids() {
    let Some(runner) = mixtral_tiny() else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no wgpu device ({e}); skipping");
            return;
        }
    };
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let toks = runner
        .generate_kv_gpu_cached(PROMPT, MAX_NEW, &mut engine, exe, |_| {
            ControlFlow::Continue(())
        })
        .expect("mixtral-tiny wgpu generate");
    eprintln!("mixtral-tiny wgpu tokens: {toks:?}");
    assert_eq!(toks, MIXTRAL_TINY_TOKENS, "mixtral-tiny wgpu greedy ids");
}
