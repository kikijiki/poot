// Moved in-crate (card 622): references a pub(crate)-only item (poot has no users, tests are not consumers).

//! gpt-oss ROCm re-prefill matches the CPU eager reference on real hardware.
//!
//! `Runner::generate_rocm_reprefill` (`crates/poot-llm/src/backends/rocm_vulkan.rs`) already dispatches
//! `self.gpt_oss` to `trace_gptoss_prefill`, like the wgpu/PTX paths. Mirrors
//! `olmoe_tiny_rocm_reprefill_matches_cpu` (`card135d_olmoe_rocm_reprefill.rs`).
//!
//! This exercises attention sinks (a per-query-head learned logit in the softmax denominator but not
//! the weighted-V sum, via `attention_prefill_with_sink`) and alternating sliding-window/full layers,
//! which the other ROCm rounds do not.
//!
//! The tied lm_head (`vocab=201088, hidden=32`, `out_numel * K = 6.435e6`) is the largest `N` tried on
//! ROCm so far (Mixtral N=32000 and OlmoE N=50280 did not hang).
//!
//! Skips at runtime if the model or ROCm is missing. Kept out of `coherence.rs` because that file is
//! not `rocm`-gated.
//!
//! Run alone (Strix Halo HSA firmware supports one active queue across all processes):
//! `cargo test -p poot-llm --features rocm --release --test card135d_gptoss_rocm_reprefill --
//! --test-threads=1 --nocapture`.

use super::*;

#[test]
fn gptoss_tiny_rocm_reprefill_matches_cpu() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("gptoss-tiny")) else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP gptoss_tiny_rocm_reprefill_matches_cpu (ROCm unavailable: {e})");
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load(&dir).expect("load gptoss-tiny");
    let exe = runner.load_on(&mut rocm).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::GptOss
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
        .expect("gptoss-tiny ROCm re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("gptoss-tiny CPU generate");
    eprintln!(
        "gptoss-tiny rocm={:?} cpu={:?}",
        runner.decode(&rocm_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        rocm_toks, cpu_toks,
        "ROCm attention-sink MoE re-prefill decode must match the CPU eager reference"
    );
}

/// GGUF-loaded gpt-oss (`Runner::load_gguf`) ROCm re-prefill matches the CPU reference.
/// `generate_rocm_reprefill` dispatches on `self.gpt_oss.is_some()` whatever loader set it, so this
/// mirrors `gptoss_tiny_rocm_reprefill_matches_cpu` with the GGUF loader (see also
/// `gptoss_tiny_gguf_gpu_reprefill_matches_cpu` in `coherence.rs`).
#[test]
fn gptoss_tiny_gguf_rocm_reprefill_matches_cpu() {
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "gptoss-tiny/gptoss-tiny-f32.gguf"
    )) else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP gptoss_tiny_gguf_rocm_reprefill_matches_cpu (ROCm unavailable: {e})");
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load_gguf(&gguf_path).expect("load gptoss-tiny gguf");
    let exe = runner.load_on(&mut rocm).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::GptOss
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
        .expect("gptoss-tiny gguf ROCm re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("gptoss-tiny gguf CPU generate");
    eprintln!(
        "gptoss-tiny gguf rocm={:?} cpu={:?}",
        runner.decode(&rocm_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        rocm_toks, cpu_toks,
        "GGUF-loaded ROCm attention-sink MoE re-prefill decode must match the GGUF-loaded CPU \
         eager reference"
    );
}
