// Moved in-crate (card 622): references a pub(crate)-only item (poot has no users, tests are not consumers).

//! Mixtral ROCm re-prefill matches the CPU eager reference on real hardware.
//!
//! `Runner::generate_rocm_reprefill` (`crates/poot-llm/src/backends/rocm_vulkan.rs`) already dispatches
//! `self.mixtral` to `trace_mixtral_prefill`, like the wgpu/PTX paths.
//!
//! Every layer routes through top-2 MoE gating (`IndexedMatMul` expert dispatch), unlike the dense-FFN
//! archs. Shape `vocab=32000, hidden=1024` (product ~3.28e7) is below BLOOM's hanging case (~2.57e8);
//! see card 258 and `mixtral_tiny_gpu_reprefill_matches_cpu` in `coherence.rs`.
//!
//! Skips at runtime if the model or ROCm is missing. Kept out of `coherence.rs` because that file is
//! not `rocm`-gated.
//!
//! Run alone (Strix Halo HSA firmware supports one active queue across all processes):
//! `cargo test -p poot-llm --features rocm --release --test card135d_mixtral_rocm_reprefill --
//! --test-threads=1 --nocapture`.

use super::*;

#[test]
fn mixtral_tiny_rocm_reprefill_matches_cpu() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("mixtral-tiny")) else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP mixtral_tiny_rocm_reprefill_matches_cpu (ROCm unavailable: {e})");
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load(&dir).expect("load mixtral-tiny");
    let exe = runner.load_on(&mut rocm).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Mixtral
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
        .expect("mixtral-tiny ROCm re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("mixtral-tiny CPU generate");
    eprintln!(
        "mixtral-tiny rocm={:?} cpu={:?}",
        runner.decode(&rocm_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        rocm_toks, cpu_toks,
        "ROCm MoE re-prefill decode must match the CPU eager reference"
    );
}

/// GGUF-loaded Mixtral (`Runner::load_gguf`) ROCm re-prefill matches the CPU reference.
/// `generate_rocm_reprefill` dispatches on `self.mixtral.is_some()` whatever loader set it, so this
/// mirrors `mixtral_tiny_rocm_reprefill_matches_cpu` with the GGUF loader (see also
/// `mixtral_tiny_gguf_gpu_reprefill_matches_cpu` in `coherence.rs`).
#[test]
#[ignore = "held for POOT-738: a Mixtral GGUF names `general.architecture = llama`, a registered family, so the Runner refuses it and the driver's llama family does not carry experts yet"]
fn mixtral_tiny_gguf_rocm_reprefill_matches_cpu() {
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "mixtral-tiny/mixtral-tiny-f32.gguf"
    )) else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP mixtral_tiny_gguf_rocm_reprefill_matches_cpu (ROCm unavailable: {e})");
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load_gguf(&gguf_path).expect("load mixtral-tiny gguf");
    let exe = runner.load_on(&mut rocm).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Mixtral
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
        .expect("mixtral-tiny gguf ROCm re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("mixtral-tiny gguf CPU generate");
    eprintln!(
        "mixtral-tiny gguf rocm={:?} cpu={:?}",
        runner.decode(&rocm_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        rocm_toks, cpu_toks,
        "GGUF-loaded ROCm MoE re-prefill decode must match the GGUF-loaded CPU eager reference"
    );
}
