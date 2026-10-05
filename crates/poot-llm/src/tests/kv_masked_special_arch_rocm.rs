// Moved in-crate (card 622): references a pub(crate)-only item (poot has no users, tests are not consumers).

//! ROCm/HSA real-hardware parity for `Runner::generate_kv_gpu_masked_rocm`
//! (`crates/poot-llm/src/backends/rocm_vulkan.rs`), the ROCm capture/replay twin of update 0683's
//! `Runner::generate_kv_gpu_masked` (wgpu, `crates/poot-llm/src/backends/gpu_generate.rs`). That update covered
//! the seven "special tracer" archs (BLOOM, MPT, SmolLM3, Mixtral dense-and-packed-quant, OlmoE,
//! gpt-oss, DeepSeek-V2) on wgpu only; this is the ROCm half of its "no ROCm or PTX analogue"
//! follow-on (docs/updates/0683 "not done").
//!
//! Mirrors `coherence.rs`'s `mixtral_tiny_gpu_kv_masked_matches_cpu_and_beats_reprefill_growth`/
//! `gptoss_tiny_gpu_kv_masked_matches_cpu`, but drives `generate_kv_gpu_masked_rocm` against
//! `poot_rocm_gpu::RocmGraphExecutor` instead of `poot_gpu::GpuExecutor`, and compares against the
//! same CPU KV-cached reference (`Runner::generate_kv_masked`) plus the CPU re-prefill reference
//! (`Runner::generate`), using the tiny untrained `mixtral-tiny` and `gptoss-tiny` safetensors
//! fixtures under `POOT_MODELS_DIR`.
//!
//! Structured like `card255_bloom_mpt_rocm_reprefill.rs`/`card135d_mixtral_rocm_reprefill.rs`/
//! `card135d_gptoss_rocm_reprefill.rs` (runtime skip on a missing model or ROCm, not `#[ignore]`), in
//! its own file because `coherence.rs` is not `rocm`-gated and must compile without the feature.
//!
//! Run one at a time (the Strix Halo HSA firmware supports one active HSA queue across all
//! processes; a second concurrent `RocmGraphExecutor::new()` corrupts the first queue's ring buffer):
//! `cargo test -p poot-llm --features rocm --release --test kv_masked_special_arch_rocm --
//! --test-threads=1 --exact <test_name> --nocapture`.

use super::*;

#[test]
fn mixtral_tiny_rocm_kv_masked_matches_cpu() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("mixtral-tiny")) else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP mixtral_tiny_rocm_kv_masked_matches_cpu (ROCm unavailable: {e})");
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load(&dir).expect("load mixtral-tiny");
    let exe = runner.load_on(&mut rocm).expect("load_on");
    assert!(
        runner.decode_arch().unwrap() == crate::core::decode_arch::DecodeArch::Mixtral,
        "mixtral-tiny classifies as Mixtral"
    );
    let prompt = "The capital of France is";
    let max_new = 24;

    let mut rocm_steps: Vec<std::time::Duration> = Vec::new();
    let mut last = std::time::Instant::now();
    let t0 = last;
    let rocm_kv_toks = runner
        .generate_kv_gpu_masked_rocm(prompt, max_new, &mut rocm, exe, |_| {
            let now = std::time::Instant::now();
            rocm_steps.push(now - last);
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("mixtral-tiny ROCm KV-cached masked decode");
    let rocm_kv_elapsed = t0.elapsed();

    let cpu_kv_toks = runner
        .generate_kv_masked(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("mixtral-tiny CPU KV-cached masked decode");

    eprintln!(
        "mixtral-tiny rocm_kv={:?} cpu_kv={:?}",
        runner.decode(&rocm_kv_toks).unwrap(),
        runner.decode(&cpu_kv_toks).unwrap()
    );
    assert_eq!(
        rocm_kv_toks, cpu_kv_toks,
        "ROCm KV-cached masked decode (MoE routing under a real, non-growing KV cache) must match the \
         CPU KV-cached eager reference"
    );

    // Re-prefill reference: same model and prompt, smaller max_new (this path's per-step cost grows with
    // position, per update 0673). It only needs to show its per-step cost is not flat.
    let reprefill_max_new = 8;
    let mut reprefill_steps: Vec<std::time::Duration> = Vec::new();
    let mut last = std::time::Instant::now();
    let reprefill_toks = runner
        .generate(prompt, reprefill_max_new, |_| {
            let now = std::time::Instant::now();
            reprefill_steps.push(now - last);
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("mixtral-tiny CPU re-prefill generate");
    assert_eq!(
        &rocm_kv_toks[..reprefill_toks.len()],
        &reprefill_toks[..],
        "CPU re-prefill greedy output must agree with the KV-cached paths on the tokens both produce"
    );

    eprintln!(
        "mixtral-tiny ROCm KV-cached: {max_new} tokens in {rocm_kv_elapsed:.2?} (first step {:.2?}, \
         last step {:.2?})",
        rocm_steps.first().copied().unwrap_or_default(),
        rocm_steps.last().copied().unwrap_or_default(),
    );
    eprintln!(
        "mixtral-tiny CPU re-prefill: {reprefill_max_new} tokens (first step {:.2?}, last step {:.2?})",
        reprefill_steps.first().copied().unwrap_or_default(),
        reprefill_steps.last().copied().unwrap_or_default(),
    );
}

/// gpt-oss twin of [`mixtral_tiny_rocm_kv_masked_matches_cpu`] above. Proves
/// `generate_kv_gpu_masked_rocm`'s dispatch to `trace_gptoss_decode_kv_masked` (attention sinks +
/// alternating sliding-window layers + biased-clamped-GLU, under a GPU-resident KV cache
/// captured/replayed on ROCm) is token-for-token identical to the CPU KV-cached reference on gfx1151.
#[test]
fn gptoss_tiny_rocm_kv_masked_matches_cpu() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("gptoss-tiny")) else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP gptoss_tiny_rocm_kv_masked_matches_cpu (ROCm unavailable: {e})");
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load(&dir).expect("load gptoss-tiny");
    let exe = runner.load_on(&mut rocm).expect("load_on");
    assert!(
        runner.decode_arch().unwrap() == crate::core::decode_arch::DecodeArch::GptOss,
        "gptoss-tiny classifies as GptOss"
    );
    let prompt = "The capital of France is";
    let max_new = 24;

    let t0 = std::time::Instant::now();
    let rocm_kv_toks = runner
        .generate_kv_gpu_masked_rocm(prompt, max_new, &mut rocm, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("gptoss-tiny ROCm KV-cached masked decode");
    let rocm_kv_elapsed = t0.elapsed();

    let cpu_kv_toks = runner
        .generate_kv_masked(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("gptoss-tiny CPU KV-cached masked decode");

    eprintln!(
        "gptoss-tiny rocm_kv={:?} cpu_kv={:?} ({max_new} tokens in {rocm_kv_elapsed:.2?})",
        runner.decode(&rocm_kv_toks).unwrap(),
        runner.decode(&cpu_kv_toks).unwrap()
    );
    assert_eq!(
        rocm_kv_toks, cpu_kv_toks,
        "ROCm KV-cached masked decode (attention sinks + sliding-window + biased-clamped-GLU under a \
         real, non-growing KV cache) must match the CPU KV-cached eager reference"
    );

    let reprefill_max_new = 6;
    let reprefill_toks = runner
        .generate(prompt, reprefill_max_new, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("gptoss-tiny CPU re-prefill generate");
    assert_eq!(
        &rocm_kv_toks[..reprefill_toks.len()],
        &reprefill_toks[..],
        "CPU re-prefill greedy output must agree with the KV-cached paths on the tokens both produce"
    );
}
