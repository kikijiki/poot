// Moved in-crate (card 622): references a pub(crate)-only item (poot has no users, tests are not consumers).

//! OlmoE ROCm re-prefill matches the CPU eager reference on real hardware.
//!
//! `Runner::generate_rocm_reprefill` (`crates/poot-llm/src/backends/rocm_vulkan.rs`) already dispatches
//! `self.olmoe` to `trace_olmoe_prefill`, like the wgpu/PTX paths. Mirrors
//! `mixtral_tiny_rocm_reprefill_matches_cpu` (`card135d_mixtral_rocm_reprefill.rs`).
//!
//! OlmoE uses a non-renormalized top-8-of-64 router (`norm_topk_prob: false`) via its own
//! `olmoe_ffn`/`olmoe_top_k_gate` composition, not the shared `poot_graph_ir::ops::moe` op Mixtral uses.
//!
//! The untied lm_head (`vocab=50280, hidden=32`, `out_numel * K = 1.609e6`) is far below BLOOM's hanging
//! case (~2.569e8).
//!
//! Skips at runtime if the model or ROCm is missing. Kept out of `coherence.rs` because that file is
//! not `rocm`-gated.
//!
//! Run alone (Strix Halo HSA firmware supports one active queue across all processes):
//! `cargo test -p poot-llm --features rocm --release --test card135d_olmoe_rocm_reprefill --
//! --test-threads=1 --nocapture`.

use super::*;

#[test]
fn olmoe_tiny_rocm_reprefill_matches_cpu() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("olmoe-tiny")) else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP olmoe_tiny_rocm_reprefill_matches_cpu (ROCm unavailable: {e})");
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load(&dir).expect("load olmoe-tiny");
    let exe = runner.load_on(&mut rocm).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Olmoe
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
        .expect("olmoe-tiny ROCm re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("olmoe-tiny CPU generate");
    eprintln!(
        "olmoe-tiny rocm={:?} cpu={:?}",
        runner.decode(&rocm_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        rocm_toks, cpu_toks,
        "ROCm non-renormalized MoE re-prefill decode must match the CPU eager reference"
    );
}

/// GGUF-loaded OlmoE (`Runner::load_gguf`) ROCm re-prefill matches the CPU reference.
/// `generate_rocm_reprefill` dispatches on `self.olmoe.is_some()` whatever loader set it, so this
/// mirrors `olmoe_tiny_rocm_reprefill_matches_cpu` with the GGUF loader (see also
/// `olmoe_tiny_gguf_gpu_reprefill_matches_cpu` in `coherence.rs`).
#[test]
fn olmoe_tiny_gguf_rocm_reprefill_matches_cpu() {
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "olmoe-tiny/olmoe-tiny-f32.gguf"
    )) else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP olmoe_tiny_gguf_rocm_reprefill_matches_cpu (ROCm unavailable: {e})");
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load_gguf(&gguf_path).expect("load olmoe-tiny gguf");
    let exe = runner.load_on(&mut rocm).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Olmoe
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
        .expect("olmoe-tiny gguf ROCm re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("olmoe-tiny gguf CPU generate");
    eprintln!(
        "olmoe-tiny gguf rocm={:?} cpu={:?}",
        runner.decode(&rocm_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        rocm_toks, cpu_toks,
        "GGUF-loaded ROCm non-renormalized MoE re-prefill decode must match the GGUF-loaded CPU \
         eager reference"
    );
}
