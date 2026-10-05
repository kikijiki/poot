// Moved in-crate (card 622): references a pub(crate)-only item (poot has no users, tests are not consumers).

//! DeepSeek-V2 ROCm re-prefill matches the CPU eager reference on real hardware.
//!
//! `Runner::generate_rocm_reprefill` (`crates/poot-llm/src/backends/rocm_vulkan.rs`) already dispatches
//! `self.deepseek2` to `trace_deepseek2_prefill`, like the wgpu/PTX paths. Mirrors
//! `gptoss_tiny_rocm_reprefill_matches_cpu` (`card135d_gptoss_rocm_reprefill.rs`).
//!
//! This exercises what the other ROCm rounds do not: the MLA compressed shared latent cache
//! (`[1,1,cap,kv_lora_rank]` + `[1,1,cap,qk_rope_head_dim]`, decompressed through `kv_b_proj` each step)
//! and interleaved-pair RoPE (`rope_interleave_apply`).
//!
//! `yujiepan/deepseek-v2-tiny-random` is untrained, so the test checks exact token equality with the
//! CPU reference, not coherent text.
//!
//! The untied lm_head (`vocab=102400, hidden=8`, `out_numel * K = 8.192e5`) sits between ROCm sizes
//! already observed not to hang (Mixtral N=32000, OlmoE N=50280, gpt-oss N=201088).
//!
//! Skips at runtime if the model or ROCm is missing. Kept out of `coherence.rs` because that file is
//! not `rocm`-gated.
//!
//! Run alone (Strix Halo HSA firmware supports one active queue across all processes):
//! `cargo test -p poot-llm --features rocm --release --test card135d_deepseek2_rocm_reprefill --
//! --test-threads=1 --nocapture`.

use super::*;

#[test]
fn deepseek2_tiny_rocm_reprefill_matches_cpu() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("deepseek2-tiny"))
    else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP deepseek2_tiny_rocm_reprefill_matches_cpu (ROCm unavailable: {e})");
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load(&dir).expect("load deepseek2-tiny");
    let exe = runner.load_on(&mut rocm).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV2
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let rocm_toks = runner
        .generate_rocm_reprefill(prompt, max_new, &mut rocm, exe, |piece| {
            let now = std::time::Instant::now();
            eprintln!(
                "  step +{:.2?} (total {:.2?}): {piece:?}",
                now - last,
                now - t0
            );
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("deepseek2-tiny ROCm re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("deepseek2-tiny CPU generate");
    eprintln!(
        "deepseek2-tiny rocm={:?} cpu={:?}",
        runner.decode(&rocm_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        rocm_toks, cpu_toks,
        "ROCm MLA (compressed-latent cache + interleaved-pair RoPE) re-prefill decode must match the CPU \
         eager reference"
    );
}

/// GGUF-loaded DeepSeek-V2 (`Runner::load_gguf`) ROCm re-prefill matches the CPU reference.
/// `generate_rocm_reprefill` dispatches on `self.deepseek2.is_some()` whatever loader set it, so this
/// mirrors `deepseek2_tiny_rocm_reprefill_matches_cpu` with the GGUF loader (see also
/// `deepseek2_tiny_gguf_gpu_reprefill_matches_cpu` in `coherence.rs`).
#[test]
fn deepseek2_tiny_gguf_rocm_reprefill_matches_cpu() {
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "deepseek2-tiny/deepseek2-tiny-f32.gguf"
    )) else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "SKIP deepseek2_tiny_gguf_rocm_reprefill_matches_cpu (ROCm unavailable: {e})"
            );
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load_gguf(&gguf_path).expect("load deepseek2-tiny gguf");
    let exe = runner.load_on(&mut rocm).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV2
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let rocm_toks = runner
        .generate_rocm_reprefill(prompt, max_new, &mut rocm, exe, |piece| {
            let now = std::time::Instant::now();
            eprintln!(
                "  step +{:.2?} (total {:.2?}): {piece:?}",
                now - last,
                now - t0
            );
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("deepseek2-tiny gguf ROCm re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("deepseek2-tiny gguf CPU generate");
    eprintln!(
        "deepseek2-tiny gguf rocm={:?} cpu={:?}",
        runner.decode(&rocm_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        rocm_toks, cpu_toks,
        "GGUF-loaded ROCm MLA (compressed-latent cache + interleaved-pair RoPE) re-prefill decode \
         must match the GGUF-loaded CPU eager reference"
    );
}
