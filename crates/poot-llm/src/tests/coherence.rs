//! Coherence checks for the Runner's per-backend generation entry points (the MoE and hybrid families
//! still on the Runner; the dense families' coherence rows run through the driver in
//! `driver/device_tests.rs`). In-crate as `#[cfg(test)]` so `generate_ptx_reprefill` and friends stay
//! `pub(crate)`. Every real-checkpoint test here still needs POOT_MODELS_DIR and skips (passes) when
//! it is unset or the model is absent, exactly as it did as an integration test; assertions, mutation
//! evidence and #[ignore] reasons are unchanged from their original tests/coherence.rs home.

use super::*;

#[test]
#[ignore = "loads granite-moe-1b + generates on the Arc GPU via re-prefill (slow MoE); run with --ignored"]
fn granite_moe_generates_on_gpu() {
    // Can a MoE model run on the GPU at all? The cached resident decode rejects the host-fallback
    // TopKGate, but the re-prefill path (`generate_gpu_reprefill` -> `run`) host-falls-back the tiny gate
    // while the expert FFN matmuls run on-device. Coherent output means MoE-on-GPU works on the slow path.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("granite-moe-1b"))
    else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load granite-moe-1b");
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let toks = runner
        .generate_gpu_reprefill("The capital of France is", 5, &mut engine, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    let text = runner.decode(&toks).unwrap();
    eprintln!("granite-moe on GPU: {text:?}");
    assert!(
        text.to_lowercase().contains("paris"),
        "granite-moe GPU continuation was: {text:?}"
    );
}

#[test]
#[ignore = "loads optimum-intel-internal-testing/tiny-mixtral (~945MB f32 safetensors, randomly \
            initialized/untrained per its own model card) + generates on real PTX/NVIDIA hardware via the \
            generic re-prefill GPU path, compared token-for-token against the CPU eager reference; run \
            with --ignored --release"]
fn mixtral_tiny_ptx_reprefill_matches_cpu() {
    // Mixtral PTX/NVIDIA parity check, mirroring `smollm3_3b_ptx_reprefill_matches_cpu`/
    // `mptk_1b_ptx_reprefill_matches_cpu`/`bloom_560m_ptx_reprefill_matches_cpu`: `generate_ptx_reprefill`
    // routes to `trace_mixtral_prefill` when `self.mixtral.is_some()`. `tiny-mixtral` is randomly
    // initialized (its model card says so), so this checks no coherence, only that PTX dispatch of the
    // top-2 MoE routing (topk gating -> gather-free indexed_matmul expert MLP -> weighted combine) yields
    // the exact same tokens as the CPU eager reference.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("mixtral-tiny")) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load mixtral-tiny");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Mixtral
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("mixtral-tiny PTX re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("mixtral-tiny CPU generate");
    eprintln!(
        "mixtral-tiny ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "PTX MoE re-prefill decode must match the CPU eager reference"
    );
}

#[test]
#[ignore = "loads optimum-intel-internal-testing/tiny-mixtral (~945MB f32 safetensors, randomly \
            initialized/untrained per its own model card) + generates on wgpu/Vulkan (this box's RADV \
            iGPU) via the generic re-prefill GPU path, compared token-for-token against the CPU eager \
            reference; run with --ignored --release"]
fn mixtral_tiny_gpu_reprefill_matches_cpu() {
    // Mixtral wgpu/RADV parity check (card 135d), mirroring `mixtral_tiny_ptx_reprefill_matches_cpu`
    // above and the smollm3/mptk/bloom wgpu tests. `generate_gpu_reprefill` routes to
    // `trace_mixtral_prefill` when `self.mixtral.is_some()`. This is the safetensors path
    // (`Runner::load`); GGUF is covered by `mixtral_tiny_gguf_generates_finite_output` below.
    // `tiny-mixtral` is randomly initialized, so this checks only that wgpu dispatch of the top-2 MoE
    // routing yields the exact same tokens as the CPU eager reference.
    //
    // Card 258 watchdog risk is low: the untied lm_head is `vocab=32000, hidden=1024`
    // (`out_numel * K = 3.277e7`), an order of magnitude below BLOOM's hanging `~2.569e8` and SmolLM3's
    // non-hanging `~2.628e8`, and its `N=32000` is far smaller than either (250880, 128256).
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("mixtral-tiny")) else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load mixtral-tiny");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Mixtral
    );
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let prompt = "The capital of France is";
    let max_new = 12;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let gpu_toks = runner
        .generate_gpu_reprefill(prompt, max_new, &mut engine, exe, |piece| {
            let now = std::time::Instant::now();
            eprintln!(
                "  step +{:.2?} (total {:.2?}): {piece:?}",
                now - last,
                now - t0
            );
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("mixtral-tiny GPU re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("mixtral-tiny CPU generate");
    eprintln!(
        "mixtral-tiny gpu={:?} cpu={:?}",
        runner.decode(&gpu_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "wgpu MoE re-prefill decode must match the CPU eager reference"
    );
}

#[test]
#[ignore = "loads hf-tiny-v2/tiny-random-OlmoeForCausalLM (randomly initialized/untrained per its own model \
            card) + generates on real PTX/NVIDIA hardware via the generic re-prefill GPU path, compared \
            token-for-token against the CPU eager reference; run with --ignored --release"]
fn olmoe_tiny_ptx_reprefill_matches_cpu() {
    // OlmoE PTX/NVIDIA parity check (card 135d, spec 262), mirroring
    // `mixtral_tiny_ptx_reprefill_matches_cpu`: `generate_ptx_reprefill` routes to `trace_olmoe_prefill`
    // when `self.olmoe.is_some()`. Unlike Mixtral, OlmoE's checkpoint sets `norm_topk_prob: false`, so its
    // `olmoe_ffn`/`olmoe_top_k_gate` composition differs from the shared `poot_graph_ir::ops::moe` op,
    // not just in routing math. `tiny-random-OlmoeForCausalLM` is untrained, so this checks only that PTX
    // dispatch of the top-8 non-renormalized routing (softmax-all -> topk -> no renormalize -> gather-free
    // indexed_matmul expert MLP -> weighted combine) yields the exact same tokens as the CPU eager
    // reference.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("olmoe-tiny")) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load olmoe-tiny");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Olmoe
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("olmoe-tiny PTX re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("olmoe-tiny CPU generate");
    eprintln!(
        "olmoe-tiny ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "PTX non-renormalized MoE re-prefill decode must match the CPU eager reference"
    );
}

#[test]
#[ignore = "loads hf-tiny-v2/tiny-random-OlmoeForCausalLM (randomly initialized/untrained per its own model \
            card) + generates on wgpu/Vulkan (this box's RADV iGPU) via the generic re-prefill GPU path, \
            compared token-for-token against the CPU eager reference; run with --ignored --release"]
fn olmoe_tiny_gpu_reprefill_matches_cpu() {
    // OlmoE wgpu/RADV parity check (card 135d, spec 262), mirroring
    // `olmoe_tiny_ptx_reprefill_matches_cpu` above and the mixtral/smollm3/mptk/bloom wgpu tests.
    // `generate_gpu_reprefill` routes to `trace_olmoe_prefill` when `self.olmoe.is_some()`. This is the
    // safetensors path (`Runner::load`); GGUF PTX parity is covered separately (update 0645). Same
    // `norm_topk_prob: false` and untrained-fixture caveats as the PTX test: this checks only that wgpu
    // dispatch of the top-8 non-renormalized routing yields the exact same tokens as the CPU eager
    // reference.
    //
    // Card 258 watchdog risk is low: the untied lm_head is `vocab=50280, hidden=32`
    // (`out_numel * K = 1.609e6`), two orders of magnitude below Mixtral-tiny's `3.277e7` and far below
    // BLOOM's hanging `~2.569e8` and SmolLM3's non-hanging `~2.628e8`. Its `N=50280` is above
    // Mixtral-tiny's `N=32000` but far below SmolLM3's non-hanging `N=128256` and BLOOM's hanging
    // `N=250880`, so the product-dominant and `N`-dominant hypotheses both predict low risk.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("olmoe-tiny")) else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load olmoe-tiny");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Olmoe
    );
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let prompt = "The capital of France is";
    let max_new = 12;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let gpu_toks = runner
        .generate_gpu_reprefill(prompt, max_new, &mut engine, exe, |piece| {
            let now = std::time::Instant::now();
            eprintln!(
                "  step +{:.2?} (total {:.2?}): {piece:?}",
                now - last,
                now - t0
            );
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("olmoe-tiny GPU re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("olmoe-tiny CPU generate");
    eprintln!(
        "olmoe-tiny gpu={:?} cpu={:?}",
        runner.decode(&gpu_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "wgpu non-renormalized MoE re-prefill decode must match the CPU eager reference"
    );
}

#[test]
#[ignore = "loads tiny-random/gpt-oss (randomly initialized/untrained per its own model card) + generates \
            on real PTX/NVIDIA hardware via the generic re-prefill GPU path, compared token-for-token \
            against the CPU eager reference; run with --ignored --release"]
fn gptoss_tiny_ptx_reprefill_matches_cpu() {
    // gpt-oss PTX/NVIDIA parity check (card 135d, spec 263), mirroring
    // `olmoe_tiny_ptx_reprefill_matches_cpu`: `generate_ptx_reprefill` routes to `trace_gptoss_prefill`
    // when `self.gpt_oss.is_some()`. The router is algebraically identical to Mixtral's (shared
    // `top_k_gate` op), but this checkpoint also exercises attention sinks (a per-query-head learned logit
    // that competes in the softmax row max/denominator but never contributes to the weighted-V output;
    // `attention_prefill_with_sink`/`attention_masked_with_sink`) and alternating sliding-window/full
    // attention layers (`GptOssParams::layer_is_sliding`, from the checkpoint's `layer_types`).
    // `tiny-random/gpt-oss` is untrained, so this checks only that PTX dispatch of biased-GQA +
    // attention-sink softmax + alternating-window masking + biased-router top-k MoE with the
    // clamped-sigmoid-GLU expert MLP yields the exact same tokens as the CPU eager reference.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("gptoss-tiny")) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load gptoss-tiny");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::GptOss
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("gptoss-tiny PTX re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("gptoss-tiny CPU generate");
    eprintln!(
        "gptoss-tiny ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "PTX attention-sink MoE re-prefill decode must match the CPU eager reference"
    );
}

#[test]
#[ignore = "loads tiny-random/gpt-oss (randomly initialized/untrained per its own model card) + generates \
            on wgpu/Vulkan (this box's RADV iGPU) via the generic re-prefill GPU path, compared \
            token-for-token against the CPU eager reference; run with --ignored --release"]
fn gptoss_tiny_gpu_reprefill_matches_cpu() {
    // gpt-oss wgpu/RADV parity check (card 135d, spec 263), mirroring
    // `gptoss_tiny_ptx_reprefill_matches_cpu` above and the olmoe/mixtral/smollm3/mptk/bloom wgpu tests.
    // `generate_gpu_reprefill` routes to `trace_gptoss_prefill` when `self.gpt_oss.is_some()`. This is the
    // safetensors path (`Runner::load`); GGUF PTX parity is covered separately (update 0645).
    // `tiny-random/gpt-oss` is untrained, so this checks only that wgpu dispatch of attention sinks and
    // alternating sliding-window/full-attention layers, on top of the top-k MoE routing Mixtral's wgpu
    // test covers, yields the exact same tokens as the CPU eager reference. It confirms
    // `attention_prefill_with_sink`/`attention_masked_with_sink` and the sliding-window mask on RADV.
    //
    // Card 258 watchdog risk: the TIED lm_head (same tied-transpose-then-matmul shape as BLOOM's hanging
    // case) is `vocab=201088, hidden=32` (`out_numel * K = 6.435e6`). By the product-dominant hypothesis
    // that is low risk (two orders below Mixtral-tiny's `3.277e7`, ~40x below BLOOM's `~2.569e8` and
    // SmolLM3's `~2.628e8`). By the `N`-dominant hypothesis (BLOOM `N=250880` hangs; SmolLM3
    // `N=128256`, OlmoE `N=50280`, Mixtral `N=32000` do not) `N=201088` is the largest non-BLOOM `N`
    // tested (~80.1% of BLOOM's, ~1.57x SmolLM3's), so this is the highest-risk case of the sweep. It
    // ran clean (update 0649), so the hang threshold lies above `201088`.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("gptoss-tiny")) else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load gptoss-tiny");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::GptOss
    );
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let prompt = "The capital of France is";
    let max_new = 12;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let gpu_toks = runner
        .generate_gpu_reprefill(prompt, max_new, &mut engine, exe, |piece| {
            let now = std::time::Instant::now();
            eprintln!(
                "  step +{:.2?} (total {:.2?}): {piece:?}",
                now - last,
                now - t0
            );
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("gptoss-tiny GPU re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("gptoss-tiny CPU generate");
    eprintln!(
        "gptoss-tiny gpu={:?} cpu={:?}",
        runner.decode(&gpu_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "wgpu attention-sink MoE re-prefill decode must match the CPU eager reference"
    );
}

#[test]
#[ignore = "loads yujiepan/deepseek-v2-tiny-random (randomly initialized/untrained per its own model card) \
            + generates on real PTX/NVIDIA hardware via the generic re-prefill GPU path, compared \
            token-for-token against the CPU eager reference; run with --ignored --release"]
fn deepseek2_tiny_ptx_reprefill_matches_cpu() {
    // DeepSeek-V2 PTX/NVIDIA parity check (card 135d, spec 264), mirroring
    // `gptoss_tiny_ptx_reprefill_matches_cpu`: `generate_ptx_reprefill` routes to
    // `trace_deepseek2_prefill` when `self.deepseek2.is_some()`. DeepSeek-V2's headline mechanism is
    // Multi-head Latent Attention (MLA): a compressed low-rank KV latent shared across heads, with a
    // narrower cache than other archs' per-head `[1,Hkv,cap,head_dim]` (`[1,1,cap,kv_lora_rank]` +
    // `[1,1,cap,qk_rope_head_dim]`, decompressed through `kv_b_proj` every decode step), plus a
    // decoupled-RoPE slice with DeepSeek's interleaved-pair convention (`rope_interleave_apply`, pairing
    // `(x[2i],x[2i+1])`) rather than poot's usual half-split pairing. `yujiepan/deepseek-v2-tiny-random`
    // is untrained, so this checks only that PTX dispatch of the compressed-latent cache and
    // interleaved-pair RoPE, on top of the routed-plus-shared-expert MoE MLP with a dense-layer prefix,
    // yields the exact same tokens as the CPU eager reference.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("deepseek2-tiny"))
    else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load deepseek2-tiny");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV2
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("deepseek2-tiny PTX re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("deepseek2-tiny CPU generate");
    eprintln!(
        "deepseek2-tiny ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "PTX compressed-latent-cache + interleaved-RoPE MLA re-prefill decode must match the CPU eager \
         reference"
    );
}

#[test]
#[ignore = "loads yujiepan/deepseek-v2-tiny-random (randomly initialized/untrained per its own model card) \
            + generates on wgpu/Vulkan (this box's RADV iGPU) via the generic re-prefill GPU path, \
            compared token-for-token against the CPU eager reference; run with --ignored --release"]
fn deepseek2_tiny_gpu_reprefill_matches_cpu() {
    // DeepSeek-V2 wgpu/RADV parity check (card 135d, spec 264), mirroring
    // `deepseek2_tiny_ptx_reprefill_matches_cpu` above and the gptoss/olmoe/mixtral wgpu tests.
    // `generate_gpu_reprefill` routes to `trace_deepseek2_prefill` when `self.deepseek2.is_some()`. MLA
    // and the interleaved-pair decoupled RoPE are described at the PTX test.
    // `yujiepan/deepseek-v2-tiny-random` is untrained, so this checks only that wgpu dispatch of the
    // compressed-latent cache and interleaved-pair RoPE, on top of the routed-plus-shared-expert MoE MLP
    // with a dense-layer prefix, yields the exact same tokens as the CPU eager reference.
    //
    // Card 258 watchdog risk is low: the untied lm_head is `vocab=102400, hidden=8`
    // (`out_numel * K = 8.192e5`), the lowest product of any calibration point, and `N=102400` sits below
    // the bracket between the non-hanging `N=201088` (gpt-oss) and the hanging `N=250880` (BLOOM), above
    // OlmoE's `50280` and below SmolLM3's `128256`. Both hypotheses predict low risk; it fills the `N`
    // calibration gap between `50280` and `128256`.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("deepseek2-tiny"))
    else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load deepseek2-tiny");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV2
    );
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let prompt = "The capital of France is";
    let max_new = 12;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let gpu_toks = runner
        .generate_gpu_reprefill(prompt, max_new, &mut engine, exe, |piece| {
            let now = std::time::Instant::now();
            eprintln!(
                "  step +{:.2?} (total {:.2?}): {piece:?}",
                now - last,
                now - t0
            );
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("deepseek2-tiny GPU re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("deepseek2-tiny CPU generate");
    eprintln!(
        "deepseek2-tiny gpu={:?} cpu={:?}",
        runner.decode(&gpu_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "wgpu compressed-latent-cache + interleaved-RoPE MLA re-prefill decode must match the CPU eager \
         reference"
    );
}

#[test]
#[ignore = "loads a REAL llama.cpp-converted GGUF of olmoe-tiny under POOT_MODELS_DIR through Runner::load_gguf, \
            generates on real PTX/NVIDIA hardware via the generic re-prefill GPU path, compared \
            token-for-token against the GGUF-loaded CPU eager reference; run with --ignored --release"]
fn olmoe_tiny_gguf_ptx_reprefill_matches_cpu() {
    // PTX-parity follow-on for the GGUF loading path, mirroring
    // `mixtral_tiny_gguf_ptx_reprefill_matches_cpu`; `olmoe_tiny_ptx_reprefill_matches_cpu` covers the
    // safetensors-loaded runner. Checks that the GGUF-populated `olmoe` field's non-renormalized top-8 MoE
    // routing (the `olmoe_ffn`/`olmoe_top_k_gate` composition, not the shared `moe` op) dispatches
    // correctly through PTX/CUDA graphs.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "olmoe-tiny/olmoe-tiny-f32.gguf"
    )) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load olmoe-tiny gguf");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Olmoe
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("olmoe-tiny gguf PTX re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("olmoe-tiny gguf CPU generate");
    eprintln!(
        "olmoe-tiny gguf ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "GGUF-loaded PTX non-renormalized MoE re-prefill decode must match the GGUF-loaded CPU eager \
         reference"
    );
}

#[test]
#[ignore = "loads a REAL llama.cpp-converted GGUF of olmoe-tiny under POOT_MODELS_DIR through Runner::load_gguf, \
            generates on wgpu/Vulkan (this box's RADV iGPU) via the generic re-prefill GPU path, compared \
            token-for-token against the GGUF-loaded CPU eager reference; run with --ignored --release"]
fn olmoe_tiny_gguf_gpu_reprefill_matches_cpu() {
    // wgpu/RADV follow-on for the GGUF loading path, combining
    // `olmoe_tiny_gguf_ptx_reprefill_matches_cpu` (GGUF load) with
    // `olmoe_tiny_gpu_reprefill_matches_cpu` (wgpu dispatch via `generate_gpu_reprefill`), as
    // `mixtral_tiny_gguf_gpu_reprefill_matches_cpu` does for Mixtral. Checks that the GGUF-populated
    // `olmoe` field's non-renormalized top-8 MoE routing (softmax-all -> topk -> no renormalize ->
    // gather-free indexed_matmul expert MLP -> weighted combine; the `olmoe_ffn`/`olmoe_top_k_gate`
    // composition, not the shared `moe` op) dispatches correctly through wgpu/Vulkan.
    //
    // Card 258 watchdog risk: OlmoE-tiny's untied lm_head is `vocab=50280, hidden=32`
    // (`out_numel * K = 1.609e6`, `N=50280`), the same shapes as the safetensors checkpoint, which ran
    // clean, well below the `N=201088`-`250880` hang bracket.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "olmoe-tiny/olmoe-tiny-f32.gguf"
    )) else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load olmoe-tiny gguf");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Olmoe
    );
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let prompt = "The capital of France is";
    let max_new = 12;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let gpu_toks = runner
        .generate_gpu_reprefill(prompt, max_new, &mut engine, exe, |piece| {
            let now = std::time::Instant::now();
            eprintln!(
                "  step +{:.2?} (total {:.2?}): {piece:?}",
                now - last,
                now - t0
            );
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("olmoe-tiny gguf GPU re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("olmoe-tiny gguf CPU generate");
    eprintln!(
        "olmoe-tiny gguf gpu={:?} cpu={:?}",
        runner.decode(&gpu_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "GGUF-loaded wgpu non-renormalized MoE re-prefill decode must match the GGUF-loaded CPU eager \
         reference"
    );
}

#[test]
#[ignore = "loads a REAL llama.cpp-converted GGUF of gptoss-tiny under POOT_MODELS_DIR through Runner::load_gguf, \
            generates on wgpu/Vulkan (this box's RADV iGPU) via the generic re-prefill GPU path, compared \
            token-for-token against the GGUF-loaded CPU eager reference; run with --ignored --release"]
fn gptoss_tiny_gguf_gpu_reprefill_matches_cpu() {
    // wgpu/RADV follow-on for the GGUF loading path, combining
    // `gptoss_tiny_gguf_ptx_reprefill_matches_cpu` (GGUF load) with
    // `gptoss_tiny_gpu_reprefill_matches_cpu` (wgpu dispatch via `generate_gpu_reprefill`), as the mixtral
    // and olmoe equivalents do. gpt-oss has more attention-mechanism surface: attention sinks (a
    // per-query-head learned logit competing in the softmax row max/denominator but never contributing to
    // the weighted-V output) and alternating sliding-window/full-attention layers, on top of the
    // biased-router top-k MoE. Checks that the GGUF-populated `gpt_oss` field's
    // `attention_prefill_with_sink`/`attention_masked_with_sink` and sliding-window mask dispatch
    // correctly through wgpu/Vulkan.
    //
    // Card 258 watchdog risk: the TIED lm_head is `vocab=201088, hidden=32` (`out_numel * K = 6.435e6`,
    // `N=201088`), the same shapes as the safetensors checkpoint. That was the highest-risk case of the
    // safetensors wgpu sweep (`N` ~80.1% of BLOOM's hanging `N=250880`, same tied-lm_head code path) and
    // ran clean (update 0649).
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "gptoss-tiny/gptoss-tiny-f32.gguf"
    )) else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load gptoss-tiny gguf");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::GptOss
    );
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let prompt = "The capital of France is";
    let max_new = 12;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let gpu_toks = runner
        .generate_gpu_reprefill(prompt, max_new, &mut engine, exe, |piece| {
            let now = std::time::Instant::now();
            eprintln!(
                "  step +{:.2?} (total {:.2?}): {piece:?}",
                now - last,
                now - t0
            );
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("gptoss-tiny gguf GPU re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("gptoss-tiny gguf CPU generate");
    eprintln!(
        "gptoss-tiny gguf gpu={:?} cpu={:?}",
        runner.decode(&gpu_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "GGUF-loaded wgpu attention-sink MoE re-prefill decode must match the GGUF-loaded CPU eager \
         reference"
    );
}

#[test]
#[ignore = "loads a REAL llama.cpp-converted GGUF of gptoss-tiny under POOT_MODELS_DIR through Runner::load_gguf, \
            generates on real PTX/NVIDIA hardware via the generic re-prefill GPU path, compared \
            token-for-token against the GGUF-loaded CPU eager reference; run with --ignored --release"]
fn gptoss_tiny_gguf_ptx_reprefill_matches_cpu() {
    // PTX-parity follow-on for the GGUF loading path, mirroring
    // `olmoe_tiny_gguf_ptx_reprefill_matches_cpu`; `gptoss_tiny_ptx_reprefill_matches_cpu` covers the
    // safetensors-loaded runner. Checks that the GGUF-populated `gpt_oss` field's attention-sink softmax +
    // alternating sliding-window/full masking + biased-router MoE dispatches correctly through PTX/CUDA
    // graphs.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "gptoss-tiny/gptoss-tiny-f32.gguf"
    )) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load gptoss-tiny gguf");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::GptOss
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("gptoss-tiny gguf PTX re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("gptoss-tiny gguf CPU generate");
    eprintln!(
        "gptoss-tiny gguf ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "GGUF-loaded PTX attention-sink MoE re-prefill decode must match the GGUF-loaded CPU eager \
         reference"
    );
}

#[test]
#[ignore = "loads a REAL llama.cpp-converted GGUF of deepseek2-tiny under POOT_MODELS_DIR through Runner::load_gguf, \
            generates on real PTX/NVIDIA hardware via the generic re-prefill GPU path, compared \
            token-for-token against the GGUF-loaded CPU eager reference; run with --ignored --release"]
fn deepseek2_tiny_gguf_ptx_reprefill_matches_cpu() {
    // PTX-parity follow-on for the GGUF loading path, mirroring
    // `gptoss_tiny_gguf_ptx_reprefill_matches_cpu`; `deepseek2_tiny_ptx_reprefill_matches_cpu` covers the
    // safetensors-loaded runner. Checks that the GGUF-populated `deepseek2` field's compressed-latent MLA
    // cache and interleaved-pair RoPE dispatch correctly through PTX/CUDA graphs.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "deepseek2-tiny/deepseek2-tiny-f32.gguf"
    )) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load deepseek2-tiny gguf");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV2
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("deepseek2-tiny gguf PTX re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("deepseek2-tiny gguf CPU generate");
    eprintln!(
        "deepseek2-tiny gguf ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "GGUF-loaded PTX compressed-latent-cache + interleaved-RoPE MLA re-prefill decode must match the \
         GGUF-loaded CPU eager reference"
    );
}

#[test]
#[ignore = "loads a REAL llama.cpp-converted GGUF of deepseek2-tiny under POOT_MODELS_DIR through Runner::load_gguf, \
            generates on wgpu/Vulkan (this box's RADV iGPU) via the generic re-prefill GPU path, compared \
            token-for-token against the GGUF-loaded CPU eager reference; run with --ignored --release"]
fn deepseek2_tiny_gguf_gpu_reprefill_matches_cpu() {
    // wgpu/RADV follow-on for the GGUF loading path, combining
    // `deepseek2_tiny_gguf_ptx_reprefill_matches_cpu` (GGUF load) with
    // `deepseek2_tiny_gpu_reprefill_matches_cpu` (wgpu dispatch via `generate_gpu_reprefill`), as the
    // mixtral, olmoe and gpt-oss equivalents do. `generate_gpu_reprefill` dispatches
    // `self.deepseek2.is_some()` to `trace_deepseek2_prefill` regardless of which loader populated the
    // field. Checks that the GGUF-populated `deepseek2` field's compressed-latent MLA cache
    // (`[1,1,cap,kv_lora_rank]` + `[1,1,cap,qk_rope_head_dim]`, decompressed through `kv_b_proj` every
    // decode step) and interleaved-pair RoPE (`rope_interleave_apply`) dispatch correctly through
    // wgpu/Vulkan.
    //
    // Card 258 watchdog risk: deepseek2-tiny's untied lm_head is `vocab=102400, hidden=8`
    // (`out_numel * K = 8.192e5`, `N=102400`), the same shapes as the safetensors checkpoint, which ran
    // clean (update 0650). `N=102400` is well below the hang bracket (`201088`-`250880`), above OlmoE's
    // `50280` and below SmolLM3's `128256`.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "deepseek2-tiny/deepseek2-tiny-f32.gguf"
    )) else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load deepseek2-tiny gguf");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV2
    );
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let prompt = "The capital of France is";
    let max_new = 12;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let gpu_toks = runner
        .generate_gpu_reprefill(prompt, max_new, &mut engine, exe, |piece| {
            let now = std::time::Instant::now();
            eprintln!(
                "  step +{:.2?} (total {:.2?}): {piece:?}",
                now - last,
                now - t0
            );
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("deepseek2-tiny gguf GPU re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("deepseek2-tiny gguf CPU generate");
    eprintln!(
        "deepseek2-tiny gguf gpu={:?} cpu={:?}",
        runner.decode(&gpu_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "GGUF-loaded wgpu compressed-latent-cache + interleaved-RoPE MLA re-prefill decode must match the \
         GGUF-loaded CPU eager reference"
    );
}

#[test]
#[ignore = "loads a real ~14GB bf16 safetensors checkpoint (allenai/OLMoE-1B-7B-0924-Instruct) + generates \
            on real PTX/NVIDIA hardware via the generic re-prefill GPU path, compared token-for-token \
            against the CPU eager reference; run with --ignored --release --nocapture"]
fn olmoe_1b_7b_ptx_reprefill_matches_cpu() {
    // Real-scale companion to `olmoe_tiny_ptx_reprefill_matches_cpu`: checks the non-renormalized
    // top-8-of-64 MoE routing chain (softmax-all -> topk -> no renormalize -> gather-free indexed_matmul
    // expert MLP -> weighted combine) executes identically on real GPU hardware at the checkpoint's full
    // scale (16 layers, hidden 2048, 64 experts), where CPU/PTX evaluator drift is more likely to flip an
    // argmax than at toy dimensions.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("olmoe-1b-7b")) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load olmoe-1b-7b");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Olmoe
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("olmoe-1b-7b PTX re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("olmoe-1b-7b CPU generate");
    eprintln!(
        "olmoe-1b-7b ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "PTX non-renormalized MoE re-prefill decode must match the CPU eager reference at real checkpoint \
         scale"
    );
}

#[test]
#[ignore = "loads a real ~31GB bf16 safetensors checkpoint (deepseek-ai/DeepSeek-V2-Lite-Chat) + generates \
            on CPU; run with --ignored --release --nocapture"]
fn deepseek2_lite_greedy_is_coherent() {
    // DeepSeek-V2 real-trained-checkpoint coherence check (card 135d, spec 264). Earlier rounds verified
    // the MLA tracer/loader and PTX-vs-CPU parity only against `yujiepan/deepseek-v2-tiny-random`
    // (2 layers, hidden 8, randomly initialized), which shows self-consistency but not correct semantics
    // on real weights (the gap the bloom/smollm3/olmoe real-checkpoint tests close).
    // `deepseek-ai/DeepSeek-V2-Lite-Chat` (27 layers, hidden 2048, 64 routed experts top-6 + 2 shared,
    // `q_lora_rank: null`, the plain-`q_proj` MLA branch the tiny fixture's `q_lora_rank: Some(2)` does
    // not exercise) is the instruction-tuned variant, chosen over the base model (as
    // `olmoe_1b_7b_greedy_is_coherent` prefers the Instruct release) for a cleaner greedy completion. Its
    // config.json has `topk_method: "greedy"`/`n_group: 1`/`topk_group: 1`, so `group_limited_greedy` is
    // a no-op and the plain-greedy-only router (`poot_models::deepseek2` module doc, item 6) is exactly
    // correct here. The checkpoint also needs real YaRN RoPE scaling (`rope_scaling.type: "yarn"`,
    // `factor: 40`, `mscale`/`mscale_all_dim: 0.707`, used by both real DeepSeek-V2 configs;
    // `poot_models::deepseek2::DeepseekV2Yarn`, `poot_llm::runner::deepseek2_yarn_params`), without which
    // a coherence check is not decisive.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("deepseek2-lite"))
    else {
        return;
    };
    // This test only calls the CPU eager `generate()` path (no GPU bind), so it uses
    // `Runner::load`, which skips the native bf16 residency bytes only a GPU zero-copy
    // bind reads, cutting peak host RAM during load by ~1/3 with identical CPU-eager numerics (the source
    // dtype is recovered via a header-only probe, so `Qwen2Config::proj_dtype` and `poot_eval`'s
    // dtype-narrowing rounding are unaffected).
    let runner = Runner::load(&dir).expect("load deepseek2-lite");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV2
    );
    let prompt = "The capital of France is";
    let toks = runner
        .generate(prompt, 20, |_| std::ops::ControlFlow::Continue(()))
        .expect("deepseek2-lite generate");
    let text = runner.decode(&toks).expect("decode");
    eprintln!("deepseek2-lite greedy: {text:?}");
    assert!(
        text.to_lowercase().contains("paris"),
        "deepseek2-lite continuation was: {text:?}"
    );
}

#[test]
#[ignore = "loads a real ~42GB bf16 safetensors checkpoint (unsloth/gpt-oss-20b-BF16, upconverted from the \
            real openai/gpt-oss-20b native MXFP4 release) + generates on the CPU eager oracle. CONFIRMED on \
            real RunPod hardware (NVIDIA RTX 6000 Ada, single GPU, this round's third attempt) to cost \
            ~1384s (~23 MINUTES) for the FIRST generated token alone, with each further step costing MORE \
            (CPU-eager re-prefills the whole growing sequence from scratch every step) - so even this test's \
            reduced 2-token budget is a REAL 1-2+ HOUR run, not a quick check. Do not run this with \
            --ignored --release --nocapture unless you intend to spend that wall-clock time (and, \
            per this round's own task brief, check in before renting hardware for it - see this test's own \
            doc comment, 'Attempt history', for the full story of why this is documented instead of run)"]
fn gptoss_20b_greedy_is_coherent() {
    // gpt-oss real-trained-checkpoint coherence check (card 135d, spec 263). Earlier rounds verified the
    // tracer, safetensors loader, PTX parity and GGUF loading only against `tiny-random/gpt-oss`
    // (2 layers, hidden 32, randomly initialized), which shows self-consistency but not semantic
    // correctness on real weights.
    //
    // Finding (header-only probes via `gguf_dump_tensors` and a safetensors-header Range request): the
    // real `openai/gpt-oss-20b` release and every GGUF conversion found on Hugging Face
    // (`ggml-org/gpt-oss-20b-GGUF`'s native `MXFP4.gguf`, `unsloth/gpt-oss-20b-GGUF`'s `Q4_K_M`/`F16`,
    // `bartowski/openai_gpt-oss-20b-GGUF`'s `Q4_K_M` and `bf16.gguf`) keep the MoE expert tensors
    // (`ffn_{gate,up,down}_exps.weight`, most of the parameters) in ggml type 39 (MXFP4) whatever the
    // file's quant label, because the source is natively MXFP4 (`quantization_config.quant_method:
    // "mxfp4"`; `modules_to_not_convert` lists only attention/router/embed/lm_head).
    // `poot_load::gguf::Gguf::dequant` has no MXFP4 arm (`crates/poot-load/src/gguf.rs` returns
    // `unsupported ggml type {other}` for type 39), so no real gpt-oss-20b/120b GGUF loads through this
    // CPU path. Unlike `mixtral_8x7b_greedy_is_coherent`'s first finding (a naming gap with a compatible
    // alternative), there is no alternative GGUF.
    //
    // Workaround: the real safetensors checkpoint is also natively MXFP4 and poot's safetensors path has
    // no MXFP4 dequantizer, but `unsloth/gpt-oss-20b-BF16` is a genuine upconversion (the maintainers
    // say "These are up converted for now", following OpenAI's upcast-to-bf16 recommendation; its
    // `model.safetensors.index.json` reports `total_size: 41829514368` for `total_parameters:
    // 20914757184`, exactly 2 bytes/param of real bf16 data). Its per-layer
    // `mlp.experts.gate_up_proj [32, 2880, 5760]` / `down_proj [32, 2880, 2880]` names and shapes match
    // the tiny fixture's crosswalk at real scale.
    //
    // Memory: gpt-oss-20b has ~20.9B params (24 layers, hidden 2880, 32 experts top-4, intermediate
    // 2880, from the real config.json). Full f32 CPU-oracle residency (`Runner::load`,
    // ~4 bytes/param, the lever that fit DeepSeek-V2-Lite's 15.7B under a single-GPU pod's ~175GiB host
    // RAM in docs/updates/0669) is `20.9e9 * 4 = ~83.6GB`, comfortably under that ceiling (Mixtral's
    // 46.7B lands at ~187GB).
    //
    // Also fixed: `read_tokenizer_config_chat` (`chat.rs`) only read the embedded `chat_template` field of
    // `tokenizer_config.json`, but the real `openai/gpt-oss-20b` repo (and `unsloth/gpt-oss-20b-BF16`)
    // ship a standalone `chat_template.jinja`. Without a fallback, `Runner::render_chat_value` silently
    // degraded to the hardcoded ChatML `ChatFormat`, rendering a harmony-format reasoning model through
    // the wrong template. Fixed generally with a fallback read and a regression test
    // (`chat::read_tokenizer_config_chat_tests`).
    //
    // Attempt history: three RunPod attempts, none reaching a coherence verdict, and none practical with
    // the CPU-eager implementation at this scale.
    // 1. `max_new = 40` with a no-op `on_token` callback ran 6+ hours (~$6.30 of single-GPU pod time)
    //    with no way to tell "slowly computing" from "hung" except sampling `ps` CPU-time over SSH.
    //    Killed without a verdict.
    // 2. Added the calibration pass + per-step ETA logging (see `calib_start` below; the reusable fix).
    //    It measured one token at 1384.2s (~23 minutes) on an RTX 6000 Ada pod, with later steps costing
    //    more (CPU-eager re-prefill cost grows with sequence length). The code then ran a separate full
    //    `max_new = 20` generate anyway, re-paying the first-step cost. Killed without a verdict.
    // 3. Now one bounded `generate` call (no separate throwaway calibration pass) with `max_new = 2`;
    //    decoding whatever that call produces is the artifact. Worst case is roughly 2 * 23+ minutes,
    //    and it was not run again (no further RunPod time without checking in).
    //
    // Conclusion: gpt-oss-20b coherence is unverified, not because of a poot correctness bug but because
    // `Runner::generate`'s CPU-eager oracle path (full re-prefill every step, no KV cache, no BLAS/SIMD)
    // does not scale to a 20B+ checkpoint with nontrivial context (the harmony system preamble alone is
    // a few hundred tokens) within a practical single-session budget. Follow-on: a GPU-dispatch generate
    // path (`generate_kv_gpu_packed`, or a wgpu/PTX path, not the CPU oracle) with a real KV cache.
    // The checkpoint is e.g. `hf download unsloth/gpt-oss-20b-BF16 --local-dir ...`; the real
    // openai/gpt-oss-20b repo's native MXFP4 safetensors will NOT work (see this test's own doc comment).
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("gptoss-20b")) else {
        return;
    };
    // CPU eager `generate()` only (no GPU bind), so it uses the same `load_cpu_oracle_only` lever as
    // `deepseek2_lite_greedy_is_coherent` above, skipping the bf16 residency bytes only a GPU zero-copy
    // bind reads.
    let runner = Runner::load(&dir).expect("load gpt-oss-20b");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::GptOss
    );
    // Instruct/reasoning-tuned checkpoint using OpenAI's "harmony" chat format (channels:
    // analysis/commentary/final): a raw prompt would degenerate, so render through the checkpoint's own
    // jinja template.
    //
    // `max_new = 2`, deliberately tiny (see "Attempt history" above): one bounded `generate` call whose
    // tokens are the artifact. At ~23+ min/step this is a 1-2+ hour run, not a smoke test, hence
    // `#[ignore]`. The harmony channel-header tokens alone would consume most of a 2-token budget, so
    // "paris" is not asserted; the assertion only checks well-formed, non-empty, non-degenerate output,
    // i.e. that loading and one forward pass through the real weights works. A multi-token coherence
    // verdict needs a GPU-dispatch generate path (see `gptoss_20b_ptx_kv_masked_is_coherent`).
    let max_new = 2usize;
    let rc = runner.render_chat_value(
        &serde_json::json!([{
            "role": "user",
            "content": "What is the capital of France? Answer in one short sentence.",
        }]),
        None,
    );
    // Observability: every generated token's callback prints elapsed wall time, step count and a running
    // ETA for the remaining budget, so progress (or a hang) is visible in a log tail instead of behind a
    // silent callback.
    let gen_start = std::time::Instant::now();
    let step = std::cell::RefCell::new(0usize);
    let toks = runner
        .generate(&rc.prompt, max_new, |_piece| {
            let mut n = step.borrow_mut();
            *n += 1;
            let elapsed = gen_start.elapsed().as_secs_f64();
            let avg_secs_per_step = elapsed / *n as f64;
            let remaining = max_new.saturating_sub(*n);
            eprintln!(
                "gpt-oss-20b progress: step {}/{max_new}, {elapsed:.1}s elapsed, {avg_secs_per_step:.1}s/step \
                 avg, ETA {:.1} min for the remaining {remaining} step(s) (re-prefill cost GROWS with \
                 sequence length, so this ETA is a lower bound, not later steps' actual cost)",
                *n,
                avg_secs_per_step * remaining as f64 / 60.0
            );
            std::ops::ControlFlow::Continue(())
        })
        .expect("gpt-oss-20b generate");
    let text = runner.decode(&toks).expect("decode");
    eprintln!(
        "gpt-oss-20b greedy, {max_new}-token budget (harmony format, NOT a coherence verdict): {text:?}"
    );
    assert!(
        !text.trim().is_empty(),
        "gpt-oss-20b produced empty output: {text:?}"
    );
}

/// The real gpt-oss-20b coherence verdict. Loads `ggml-org/gpt-oss-20b-GGUF`'s
/// `gpt-oss-20b-MXFP4.gguf` (12.1 GB, `openai/gpt-oss-20b`'s native MXFP4 routed experts) through
/// `Runner::load_gguf`'s packed native-quant resident MoE loader (update 0689), then generates through
/// the PTX/NVIDIA GPU-resident KV-cached decode (`Runner::generate_kv_gpu_masked_ptx`, update 0686)
/// using the MXFP4 GPU kernels of update 0692.
///
/// Why this shape:
///
/// - **MXFP4, not a BF16 upconversion.** Every real gpt-oss release keeps its routed experts natively
///   MXFP4. The BF16 path (`unsloth/gpt-oss-20b-BF16`, update 0673) needs ~73.6 GiB GPU-resident
///   (wgpu's `upload` is f32-only), so it never got a GPU run; the packed MXFP4 path is an order of
///   magnitude smaller (see the VRAM math below).
/// - **PTX, not wgpu.** wgpu/Vulkan does not initialize on a RunPod NVIDIA compute pod (update 0691):
///   the host does not load the `nvidia_drm` kernel module (`/proc/modules`), so no `/dev/dri` render
///   node exists. `PtxGraphExecutor` needs only `libcuda.so.1`, and `generate_kv_gpu_masked_ptx` has a
///   RunPod precedent (update 0686's RTX 3090 run of `gptoss_tiny_ptx_kv_masked_matches_cpu`).
/// - **KV-cached decode, not `Runner::generate`.** CPU-eager re-prefill re-runs the whole prompt each
///   step and measured ~23 minutes per token on this checkpoint, growing with position (update 0673).
///   `generate_kv_gpu_masked_ptx` replays a constant-shape captured CUDA graph per token over a
///   fixed-capacity masked KV cache: flat per-step cost, so a small calibration run predicts a longer
///   one.
///
/// GPU-resident VRAM, from this file's tensor table (24 layers, hidden 2880, intermediate 2880, 32
/// experts, vocab 201088, 64 q heads / 8 kv heads, head_dim 64). Non-expert tensors are Q8_0 on disk
/// but poot's gpt-oss packed loader keeps everything outside the routed experts dense f32 (4
/// bytes/elem resident):
///
/// - routed experts, packed: 19.11e9 elems (`2*2880*2880*32` gate+up plus `2880*2880*32` down, per
///   layer, times 24) at MXFP4's 4 bits/elem in i32 words = 9.56 GB, plus compact E8M0 scale carriers
///   (a 2880-wide row has 90 exponent bytes padded to 92, ~0.61 GB total). **10.17 GB.**
/// - `token_embd` + `output`, dequantized Q8_0 -> f32: 2 * 201088 * 2880 * 4 = **4.63 GB.**
/// - attention q/k/v/o (+ biases), dequantized Q8_0 -> f32: 24 * 26.55e6 * 4 = **2.55 GB.**
/// - router, per-expert biases, norms, sinks: **~0.03 GB.**
///
/// Total ~17.4 GB: a 24GB card is tight (CUDA context and fragmentation eat into the ~6.6 GB left), a
/// 48GB-class card has headroom, and it is ~4x below the BF16 path's ~73.6 GiB. Host RAM peaks higher
/// than VRAM during load: `Gguf::load` reads the whole 12.1 GB file into a `Vec<u8>` before the
/// packed/dense tensors are built, so peak RSS is ~30 GB. `Tensor` payloads are `Arc`-shared, so
/// `const_inputs`' clone into the capture map does not double that.
///
/// Coherence bar: gpt-oss speaks the "harmony" format and always emits a chain-of-thought `analysis`
/// channel before the `final` channel with the user-visible answer, so "Paris" does not appear in the
/// first few tokens even when the model is coherent; coherent English in the analysis text is already
/// a real verdict. The strict `paris` assert arms only at a budget large enough for the final channel
/// to plausibly be reached; below that the test prints the sample and asserts only non-emptiness. Set
/// `POOT_GPTOSS_20B_MAX_NEW` (default 3) for a calibration pass with real per-step timing before the
/// full run.
#[test]
#[ignore = "loads the REAL ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf (12.1GB on disk, ~17.4GB \
            GPU-resident) through Runner::load_gguf's PACKED MXFP4 resident MoE loader, then generates on \
            real PTX/NVIDIA hardware via Runner::generate_kv_gpu_masked_ptx. Needs a 24GB-class GPU at \
            minimum (48GB-class for real headroom) and ~40GB host RAM. Set POOT_GPTOSS_20B_MAX_NEW to 2-3 \
            for a calibration-only pass with real per-step timing/ETA BEFORE committing to the full \
            budget. Run via the poot-orchestrator exec route with the \
            ghcr.io/kikijiki/poot-bench image (it carries the LLVM 22 `llc` poot's PTX kernelgen shells \
            out to); --ignored --release --nocapture."]
fn gptoss_20b_ptx_kv_masked_is_coherent() {
    // The GGUF is e.g. `hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf`.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf"
    )) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let load_start = std::time::Instant::now();
    let runner = Runner::load_gguf(&gguf_path).expect("load gpt-oss-20b MXFP4 gguf");
    let exe = runner.load_on(&mut ptx).expect("load_on");
    eprintln!(
        "gpt-oss-20b MXFP4 load: {:.1?} (packed native-quant resident MoE loader)",
        load_start.elapsed()
    );
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::GptOss
    );
    assert_eq!(
        runner
            .weight_formats()
            .get("model.layers.0.mlp.experts.gate_up_proj")
            .map(|packed| packed.weight.format()),
        Some(poot_quant::format::WeightFormat::Mxfp4),
        "a real gpt-oss release's routed experts are MXFP4 and load packed (card 545a)"
    );

    // Instruct model with a non-ChatML prompt format: render through the checkpoint's own harmony
    // template (`gptoss_20b_gguf_harmony_chat_template_renders` above is the cheap precondition check).
    let rc = runner.render_chat_value(
        &serde_json::json!([{"role": "user", "content": "What is the capital of France?"}]),
        None,
    );
    eprintln!(
        "gpt-oss-20b harmony prompt ({} chars): {rc:?}",
        rc.prompt.len()
    );

    // Calibration first: the default budget is tiny and every step prints elapsed/rate/ETA, so a log
    // tail shows live progress or a hang. Per-step cost is flat (captured CUDA graph over a
    // fixed-capacity KV cache), so the calibration rate predicts a longer run. Step 1 also carries
    // one-time capture/JIT cost, so the ETA uses the running average, which converges toward the
    // steady-state rate.
    let max_new: usize = std::env::var("POOT_GPTOSS_20B_MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let gen_start = std::time::Instant::now();
    let step = std::cell::RefCell::new(0usize);
    let toks = runner
        .generate_kv_gpu_cached(&rc.prompt, max_new, &mut ptx, exe, |piece| {
            let mut n = step.borrow_mut();
            *n += 1;
            let elapsed = gen_start.elapsed().as_secs_f64();
            let avg = elapsed / *n as f64;
            let remaining = max_new.saturating_sub(*n);
            eprintln!(
                "gpt-oss-20b PTX KV-cached progress: step {}/{max_new} {piece:?}, {elapsed:.1}s elapsed, \
                 {avg:.2}s/step avg, ETA {:.1}s for the remaining {remaining} step(s) (first step \
                 includes one-time CUDA graph capture + PTX JIT; flat per-step cost expected after that)",
                *n,
                avg * remaining as f64
            );
            std::ops::ControlFlow::Continue(())
        })
        .expect("gpt-oss-20b PTX KV-cached masked decode");
    let total = gen_start.elapsed();
    eprintln!(
        "gpt-oss-20b PTX KV-cached: {max_new} tokens in {total:.2?} ({:.2}s/token avg) - a 64-token run \
         would cost about {:.1}s at this measured per-token rate",
        total.as_secs_f64() / max_new.max(1) as f64,
        (total.as_secs_f64() / max_new.max(1) as f64) * 64.0
    );
    let text = runner.decode(&toks).expect("decode");
    eprintln!("gpt-oss-20b PTX KV-cached greedy ({max_new} tokens), FULL text: {text:?}");
    // The generated tail alone (the prompt is a long harmony preamble, so the full text buries it).
    let prompt_len = runner.encode(&rc.prompt).expect("encode prompt").len();
    let generated = runner
        .decode(&toks[prompt_len.min(toks.len())..])
        .expect("decode generated tail");
    eprintln!("gpt-oss-20b PTX KV-cached greedy ({max_new} tokens), GENERATED ONLY: {generated:?}");
    assert!(!text.trim().is_empty(), "decoded text must be non-empty");
    // Harmony always emits an `analysis` channel first, so the user-visible answer needs a real budget to
    // reach - see this test's own doc comment. Only arm the strict bar once that budget is present.
    if max_new >= 64 {
        assert!(
            text.to_lowercase().contains("paris"),
            "gpt-oss-20b PTX KV-cached continuation was: {text:?}"
        );
    }
}

#[test]
#[ignore = "loads a real ~31GB bf16 safetensors checkpoint (deepseek-ai/DeepSeek-V2-Lite-Chat) + generates \
            on real PTX/NVIDIA hardware via the generic re-prefill GPU path, compared token-for-token \
            against the CPU eager reference; run with --ignored --release --nocapture"]
fn deepseek2_lite_ptx_reprefill_matches_cpu() {
    // Real-scale companion to `deepseek2_tiny_ptx_reprefill_matches_cpu` above: checks the
    // compressed-latent-cache + interleaved-pair-RoPE MLA machinery, plus real YaRN scaling, executes
    // identically on real GPU hardware at the checkpoint's full scale (27 layers, hidden 2048,
    // `kv_lora_rank: 512`, `qk_rope_head_dim: 64`, 64 routed experts), where CPU/PTX evaluator drift is
    // more likely to flip an argmax than at toy dimensions.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("deepseek2-lite"))
    else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load deepseek2-lite");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV2
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("deepseek2-lite PTX re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("deepseek2-lite CPU generate");
    eprintln!(
        "deepseek2-lite ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "PTX compressed-latent-cache + interleaved-RoPE + real-YaRN MLA re-prefill decode must match the \
         CPU eager reference at real checkpoint scale"
    );
}

/// PTX/NVIDIA twin of `mixtral_tiny_gpu_kv_masked_matches_cpu_and_beats_reprefill_growth` (deleted here,
/// `generate_kv_gpu_masked` is test-only and does not migrate; this PTX twin drives its own
/// `generate_kv_gpu_masked_ptx`, outside this card's scope).
/// Drives `Runner::generate_kv_gpu_masked_ptx` (`crates/poot-llm/src/backends/ptx.rs`), the
/// captured-CUDA-graph twin of `generate_kv_gpu_masked`, against the same mixtral-tiny checkpoint and
/// the same CPU KV-cached / CPU re-prefill references. Needs NVIDIA hardware: `#[ignore]`, run via the
/// RunPod orchestrator, like `mixtral_tiny_ptx_reprefill_matches_cpu`.
#[test]
#[ignore = "loads optimum-intel-internal-testing/tiny-mixtral (~945MB f32 safetensors, randomly \
            initialized/untrained per its own model card) + generates on real PTX/NVIDIA hardware via the \
            new GPU KV-cached masked-decode path, compared token-for-token against the CPU KV-cached \
            reference and the CPU re-prefill reference; run with --ignored --release"]
fn mixtral_tiny_ptx_kv_masked_matches_cpu_and_beats_reprefill_growth() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("mixtral-tiny")) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load mixtral-tiny");
    let exe = runner.load_on(&mut ptx).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Mixtral
    );
    let prompt = "The capital of France is";
    let max_new = 24;

    let mut ptx_steps: Vec<std::time::Duration> = Vec::new();
    let mut last = std::time::Instant::now();
    let t0 = last;
    let ptx_kv_toks = runner
        .generate_kv_gpu_cached(prompt, max_new, &mut ptx, exe, |_| {
            let now = std::time::Instant::now();
            ptx_steps.push(now - last);
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("mixtral-tiny PTX KV-cached masked decode");
    let ptx_kv_elapsed = t0.elapsed();

    let cpu_kv_toks = runner
        .generate_kv_masked(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("mixtral-tiny CPU KV-cached masked decode");

    eprintln!(
        "mixtral-tiny ptx_kv={:?} cpu_kv={:?}",
        runner.decode(&ptx_kv_toks).unwrap(),
        runner.decode(&cpu_kv_toks).unwrap()
    );
    assert_eq!(
        ptx_kv_toks, cpu_kv_toks,
        "PTX KV-cached masked decode (MoE routing under a real, non-growing KV cache) must match the CPU \
         KV-cached eager reference"
    );

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
        &ptx_kv_toks[..reprefill_toks.len()],
        &reprefill_toks[..],
        "CPU re-prefill greedy output must agree with the KV-cached paths on the tokens both produce"
    );

    eprintln!(
        "mixtral-tiny PTX KV-cached: {max_new} tokens in {ptx_kv_elapsed:.2?} (first step {:.2?}, last \
         step {:.2?})",
        ptx_steps.first().copied().unwrap_or_default(),
        ptx_steps.last().copied().unwrap_or_default(),
    );
    eprintln!(
        "mixtral-tiny CPU re-prefill: {reprefill_max_new} tokens (first step {:.2?}, last step {:.2?})",
        reprefill_steps.first().copied().unwrap_or_default(),
        reprefill_steps.last().copied().unwrap_or_default(),
    );
}

/// gpt-oss twin of [`mixtral_tiny_ptx_kv_masked_matches_cpu_and_beats_reprefill_growth`] above.
/// Proves `generate_kv_gpu_masked_ptx`'s dispatch to `trace_gptoss_decode_kv_masked` on PTX/NVIDIA via
/// a captured CUDA graph is token-for-token identical to the CPU KV-cached reference.
#[test]
#[ignore = "loads a REAL tiny gpt-oss checkpoint (~models/gptoss-tiny, f32 safetensors, randomly \
            initialized/untrained) + generates on real PTX/NVIDIA hardware via the new GPU KV-cached \
            masked-decode path, compared token-for-token against the CPU KV-cached reference; run with \
            --ignored --release"]
fn gptoss_tiny_ptx_kv_masked_matches_cpu() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("gptoss-tiny")) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load gptoss-tiny");
    let exe = runner.load_on(&mut ptx).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::GptOss
    );
    let prompt = "The capital of France is";
    let max_new = 24;

    let t0 = std::time::Instant::now();
    let ptx_kv_toks = runner
        .generate_kv_gpu_cached(prompt, max_new, &mut ptx, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("gptoss-tiny PTX KV-cached masked decode");
    let ptx_kv_elapsed = t0.elapsed();

    let cpu_kv_toks = runner
        .generate_kv_masked(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("gptoss-tiny CPU KV-cached masked decode");

    eprintln!(
        "gptoss-tiny ptx_kv={:?} cpu_kv={:?} ({max_new} tokens in {ptx_kv_elapsed:.2?})",
        runner.decode(&ptx_kv_toks).unwrap(),
        runner.decode(&cpu_kv_toks).unwrap()
    );
    assert_eq!(
        ptx_kv_toks, cpu_kv_toks,
        "PTX KV-cached masked decode (attention sinks + sliding-window + biased-clamped-GLU under a real, \
         non-growing KV cache) must match the CPU KV-cached eager reference"
    );

    let reprefill_max_new = 6;
    let reprefill_toks = runner
        .generate(prompt, reprefill_max_new, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("gptoss-tiny CPU re-prefill generate");
    assert_eq!(
        &ptx_kv_toks[..reprefill_toks.len()],
        &reprefill_toks[..],
        "CPU re-prefill greedy output must agree with the KV-cached paths on the tokens both produce"
    );
}

/// DeepSeek-V3 GGUF PTX/NVIDIA parity. The loader (update 0791) added `self.deepseek3` dispatch arms
/// mirroring `self.deepseek2` (`generate.rs`/`gpu_generate.rs`/`ptx.rs`/`rocm_vulkan.rs`/`batched.rs`),
/// but the GPU dispatch was only compile-checked ("Not
/// done" item 1). No real DeepSeek-V3 checkpoint exists to test against (item 2), so this reuses the
/// synthetic-GGUF fixture shape of `deepseek3_load_tests`/
/// `deepseek3_synthetic_gguf_matches_hand_rolled_reference` in `runner.rs` (3 layers, layer 0 dense,
/// layers 1-2 routed, real group limiting with `n_group=3`/`topk_group=1` on 6 experts, plus the
/// V3-only `exp_probs_b.bias` selection-correction tensor), written to a `.gguf` FILE so
/// `Runner::load_gguf` loads it like a real checkpoint. It mirrors
/// `deepseek2_tiny_gguf_ptx_reprefill_matches_cpu` above (same MLA attention, special-tracer family and
/// `generate_ptx_reprefill` path).
///
/// Unlike the loader unit test (which binds 4 raw token ids), this goes through the real text encode
/// path, exercising the fixture's 6-token byte-level-BPE vocab (`["a".."f"]`, no `tokenizer.ggml.model`
/// so it defaults to BPE, no `tokenizer.ggml.eos_token_id` so it defaults to 100_001, unreachable from
/// a 6-way vocab, so `max_new` always runs to completion). `deepseek2.context_length` is 32 (the loader
/// test uses 8) because re-prefill grows the sequence past the prompt each step and the RoPE table is
/// sized off `context_length`. Tensor values are fresh random fill generated in GGUF wire shape; the
/// loader's transpose/reconstruct correctness is the other test's job, this one needs a well-formed,
/// non-degenerate checkpoint to compare PTX against CPU on.
fn write_synthetic_deepseek3_gguf() -> poot_test_util::UniqueTempPath {
    use poot_load::gguf::{GgufValue, write_gguf};

    let (h, hq, r, kv_rank, nope, rope_d, vd) =
        (8usize, 2usize, 4usize, 4usize, 2usize, 2usize, 2usize);
    let qk_head_dim = nope + rope_d;
    let layers = 3usize;
    let vocab = 6usize;
    let max_pos = 32usize;
    let eps = 1e-5f32;
    let rope_theta = 10_000.0f32;
    let first_k_dense_replace = 1usize;
    let dense_inter = 6usize;
    let (n_routed_experts, top_k, n_group, topk_group, moe_inter, n_shared_experts) =
        (6usize, 2usize, 3usize, 1usize, 4usize, 1usize);
    let routed_scaling_factor = 1.7f32;
    let shared_inter = moe_inter * n_shared_experts;

    // Same xorshift fill this codebase's `deepseek3_load_tests` (runner.rs) uses, so magnitudes match a
    // proven-non-degenerate fixture.
    fn fill(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }
    let wgt = |seed: u64, n: usize| -> Vec<f32> { fill(seed, n).iter().map(|v| v * 0.1).collect() };
    let gamma = |seed: u64, n: usize| -> Vec<f32> {
        fill(seed, n).iter().map(|v| 1.0 + v * 0.05).collect()
    };
    let bias_vec =
        |seed: u64, n: usize| -> Vec<f32> { fill(seed, n).iter().map(|v| v * 0.5).collect() };
    use poot_test_util::f32_bytes;

    let mut seed = 1u64;
    let mut next_seed = || {
        seed = seed.wrapping_add(0x9E37_79B9);
        seed
    };

    const F32: u32 = 0;
    let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    tensors.push((
        "token_embd.weight".to_string(),
        vec![h as u64, vocab as u64],
        F32,
        f32_bytes(&wgt(next_seed(), h * vocab)),
    ));
    for li in 0..layers {
        let blk = |s: &str| format!("blk.{li}.{s}");
        tensors.push((
            blk("attn_norm.weight"),
            vec![h as u64],
            F32,
            f32_bytes(&gamma(next_seed(), h)),
        ));
        tensors.push((
            blk("ffn_norm.weight"),
            vec![h as u64],
            F32,
            f32_bytes(&gamma(next_seed(), h)),
        ));
        tensors.push((
            blk("attn_q_a.weight"),
            vec![h as u64, r as u64],
            F32,
            f32_bytes(&wgt(next_seed(), h * r)),
        ));
        tensors.push((
            blk("attn_q_a_norm.weight"),
            vec![r as u64],
            F32,
            f32_bytes(&gamma(next_seed(), r)),
        ));
        tensors.push((
            blk("attn_q_b.weight"),
            vec![r as u64, (hq * qk_head_dim) as u64],
            F32,
            f32_bytes(&wgt(next_seed(), r * hq * qk_head_dim)),
        ));
        tensors.push((
            blk("attn_kv_a_mqa.weight"),
            vec![h as u64, (kv_rank + rope_d) as u64],
            F32,
            f32_bytes(&wgt(next_seed(), h * (kv_rank + rope_d))),
        ));
        tensors.push((
            blk("attn_kv_a_norm.weight"),
            vec![kv_rank as u64],
            F32,
            f32_bytes(&gamma(next_seed(), kv_rank)),
        ));
        tensors.push((
            blk("attn_k_b.weight"),
            vec![nope as u64, kv_rank as u64, hq as u64],
            F32,
            f32_bytes(&wgt(next_seed(), nope * kv_rank * hq)),
        ));
        tensors.push((
            blk("attn_v_b.weight"),
            vec![kv_rank as u64, vd as u64, hq as u64],
            F32,
            f32_bytes(&wgt(next_seed(), kv_rank * vd * hq)),
        ));
        tensors.push((
            blk("attn_output.weight"),
            vec![(hq * vd) as u64, h as u64],
            F32,
            f32_bytes(&wgt(next_seed(), hq * vd * h)),
        ));
        if li < first_k_dense_replace {
            tensors.push((
                blk("ffn_gate.weight"),
                vec![h as u64, dense_inter as u64],
                F32,
                f32_bytes(&wgt(next_seed(), h * dense_inter)),
            ));
            tensors.push((
                blk("ffn_up.weight"),
                vec![h as u64, dense_inter as u64],
                F32,
                f32_bytes(&wgt(next_seed(), h * dense_inter)),
            ));
            tensors.push((
                blk("ffn_down.weight"),
                vec![dense_inter as u64, h as u64],
                F32,
                f32_bytes(&wgt(next_seed(), dense_inter * h)),
            ));
        } else {
            tensors.push((
                blk("ffn_gate_inp.weight"),
                vec![h as u64, n_routed_experts as u64],
                F32,
                f32_bytes(&wgt(next_seed(), h * n_routed_experts)),
            ));
            tensors.push((
                blk("exp_probs_b.bias"),
                vec![n_routed_experts as u64],
                F32,
                f32_bytes(&bias_vec(next_seed(), n_routed_experts)),
            ));
            tensors.push((
                blk("ffn_gate_exps.weight"),
                vec![h as u64, moe_inter as u64, n_routed_experts as u64],
                F32,
                f32_bytes(&wgt(next_seed(), h * moe_inter * n_routed_experts)),
            ));
            tensors.push((
                blk("ffn_up_exps.weight"),
                vec![h as u64, moe_inter as u64, n_routed_experts as u64],
                F32,
                f32_bytes(&wgt(next_seed(), h * moe_inter * n_routed_experts)),
            ));
            tensors.push((
                blk("ffn_down_exps.weight"),
                vec![moe_inter as u64, h as u64, n_routed_experts as u64],
                F32,
                f32_bytes(&wgt(next_seed(), moe_inter * h * n_routed_experts)),
            ));
            tensors.push((
                blk("ffn_gate_shexp.weight"),
                vec![h as u64, shared_inter as u64],
                F32,
                f32_bytes(&wgt(next_seed(), h * shared_inter)),
            ));
            tensors.push((
                blk("ffn_up_shexp.weight"),
                vec![h as u64, shared_inter as u64],
                F32,
                f32_bytes(&wgt(next_seed(), h * shared_inter)),
            ));
            tensors.push((
                blk("ffn_down_shexp.weight"),
                vec![shared_inter as u64, h as u64],
                F32,
                f32_bytes(&wgt(next_seed(), shared_inter * h)),
            ));
        }
    }
    tensors.push((
        "output_norm.weight".to_string(),
        vec![h as u64],
        F32,
        f32_bytes(&gamma(next_seed(), h)),
    ));
    tensors.push((
        "output.weight".to_string(),
        vec![h as u64, vocab as u64],
        F32,
        f32_bytes(&wgt(next_seed(), h * vocab)),
    ));

    let tokenizer_tokens: Vec<GgufValue> = ["a", "b", "c", "d", "e", "f"]
        .iter()
        .map(|s| GgufValue::Str(s.to_string()))
        .collect();
    let kvs: Vec<(&str, GgufValue)> = vec![
        (
            "general.architecture",
            GgufValue::Str("deepseek2".to_string()),
        ),
        ("deepseek2.embedding_length", GgufValue::U32(h as u32)),
        ("deepseek2.block_count", GgufValue::U32(layers as u32)),
        ("deepseek2.attention.head_count", GgufValue::U32(hq as u32)),
        ("deepseek2.attention.q_lora_rank", GgufValue::U32(r as u32)),
        (
            "deepseek2.attention.kv_lora_rank",
            GgufValue::U32(kv_rank as u32),
        ),
        (
            "deepseek2.attention.key_length_mla",
            GgufValue::U32(qk_head_dim as u32),
        ),
        (
            "deepseek2.rope.dimension_count",
            GgufValue::U32(rope_d as u32),
        ),
        (
            "deepseek2.attention.value_length_mla",
            GgufValue::U32(vd as u32),
        ),
        (
            "deepseek2.attention.layer_norm_rms_epsilon",
            GgufValue::F32(eps),
        ),
        ("deepseek2.context_length", GgufValue::U32(max_pos as u32)),
        ("deepseek2.rope.freq_base", GgufValue::F32(rope_theta)),
        (
            "deepseek2.feed_forward_length",
            GgufValue::U32(dense_inter as u32),
        ),
        (
            "deepseek2.expert_count",
            GgufValue::U32(n_routed_experts as u32),
        ),
        ("deepseek2.expert_used_count", GgufValue::U32(top_k as u32)),
        (
            "deepseek2.expert_feed_forward_length",
            GgufValue::U32(moe_inter as u32),
        ),
        (
            "deepseek2.expert_shared_count",
            GgufValue::U32(n_shared_experts as u32),
        ),
        (
            "deepseek2.leading_dense_block_count",
            GgufValue::U32(first_k_dense_replace as u32),
        ),
        (
            "deepseek2.expert_weights_scale",
            GgufValue::F32(routed_scaling_factor),
        ),
        (
            "deepseek2.expert_group_count",
            GgufValue::U32(n_group as u32),
        ),
        (
            "deepseek2.expert_group_used_count",
            GgufValue::U32(topk_group as u32),
        ),
        ("tokenizer.ggml.tokens", GgufValue::Array(tokenizer_tokens)),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("a b".into())]),
        ),
    ];
    let tensors_ref: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = tensors
        .iter()
        .map(|(n, d, t, b)| (n.as_str(), d.clone(), *t, b.clone()))
        .collect();
    let bytes = write_gguf(&kvs, &tensors_ref);
    let path = poot_test_util::unique_temp_path("poot_llm_deepseek3_synthetic_ptx_fixture.gguf");
    std::fs::write(&path, &bytes).expect("write synthetic deepseek3 gguf fixture");
    path
}

#[test]
#[ignore = "builds a fully SYNTHETIC deepseek3-style GGUF in-process (no real DeepSeek-V3 checkpoint exists \
            anywhere to download - see docs/updates/0791-deepseek3-gguf-loader.md), loads it through \
            Runner::load_gguf, generates on real PTX/NVIDIA hardware via the generic re-prefill GPU path, \
            compared token-for-token against the CPU eager reference; run with --ignored --release"]
fn deepseek3_synthetic_ptx_reprefill_matches_cpu() {
    let path = write_synthetic_deepseek3_gguf();
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load_gguf(&path).expect("load synthetic deepseek3 gguf");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV3
    );
    // Uses all 6 vocab entries (this fixture's byte-level BPE has no live merges - "ab" was never added
    // to the vocab, so each ASCII char stays its own token, ids 0..6 in declaration order).
    let prompt = "abcdef";
    let max_new = 10;
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("synthetic deepseek3 PTX re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("synthetic deepseek3 CPU generate");
    eprintln!(
        "deepseek3-synthetic ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "GGUF-loaded PTX re-prefill decode for a synthetic DeepSeek-V3-style checkpoint (group-limited \
         MoE routing + MLA attention + the V3-only exp_probs_b.bias selection-correction term) must match \
         the GGUF-loaded CPU eager reference"
    );
}

#[test]
#[ignore = "held for POOT-738: a Mixtral GGUF names `general.architecture = llama`, a registered family, so the Runner refuses it and the driver's llama family does not carry experts yet"]
fn mixtral_8x7b_greedy_is_coherent() {
    // Real trained Mixtral coherence check. Earlier Mixtral verification (tracer, both loaders,
    // PTX/wgpu/ROCm parity) used only `optimum-intel-internal-testing/tiny-mixtral`, a randomly
    // initialized 2-layer fixture; that shows self-consistency but not that poot's top-2 MoE routing and
    // per-expert SwiGLU produce coherent text on real weights (the gap
    // `bloom_560m_greedy_is_coherent`/`olmoe_1b_7b_greedy_is_coherent`/`deepseek2_lite_greedy_is_coherent`
    // close for their archs). `mistralai/Mixtral-8x7B-Instruct-v0.1` (46.7B params, 32 layers, 8 experts
    // top-2, `norm_topk_prob` router) is used as a Q4_K_M GGUF because the bf16 safetensors are ~90GB.
    // poot does not run it quantized end to end; see the findings below.
    //
    // Finding 1 (naming): `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` (`mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf`)
    // fails to load (`dequant blk.0.ffn_gate_exps.weight: gguf: missing tensor
    // blk.0.ffn_gate_exps.weight`). Confirmed via a header-only dump of the first 256MB (HTTP Range
    // request): that conversion (~Dec 2023) uses the old per-expert tensors (`blk.N.ffn_gate.{0..7}.weight`,
    // etc.), not the merged 3D `blk.N.ffn_{gate,up,down}_exps.weight` layout that `gguf_weights`'s Mixtral
    // detection exclusively recognizes. Any pre-2024 conversion with the per-expert layout is
    // incompatible. Worked around with `mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF`'s Q4_K_M (merged
    // `_exps` layout). Follow-on: teach `gguf_weights`'s Mixtral arm the per-expert layout, as the
    // `qwen3moe` arm already copes with more than one layout.
    //
    // Finding 2: Mixtral's GGUF loader has no native-quant (Q4_0/Q8_0/K-quant) resident loading path
    // (`gguf_quant`'s packed constants in `Runner::load_gguf` are gated to `arch == "qwen2" | "qwen3"`).
    // Every Mixtral GGUF tensor is dequantized to F32 on load (`poot_load::gguf::Gguf::dequant` returns a
    // whole-tensor `Vec<f32>`): ~46.7e9 * 4 bytes = ~187GB of f32 weights resident in host RAM, on top of
    // the ~26-28GB raw file bytes that `Gguf` also keeps resident (`bytes: Vec<u8>`). This is the
    // memory-wall class documented in `docs/updates/0633`/`0638`/`0669` for DeepSeek-V2-Lite, ~3x worse;
    // that model needed `Runner::load` to reach ~60-63GB and still barely fit a
    // single-GPU RunPod pod's host RAM ceiling (`187999997952` bytes for an `NVIDIA L40S` SECURE pod).
    // Mixtral-8x7B's ~186.8GB floor is within ~1GB of that ceiling, with no bf16-residency lever (GGUF
    // tensors carry no native bf16/f16 bytes) and no native-quant resident path.
    //
    // Confirmed on hardware: run via `poot-orchestrator exec` on a single RunPod `NVIDIA L40S` SECURE pod
    // (cgroup `memory.max` = `187999997952`) with the `mradermacher` checkpoint, the process was SIGKILLed
    // (exit 137, cgroup OOM) during `Runner::load_gguf`'s dequant, before generation started, so this test
    // yields no coherence verdict. The fix (extending `gguf_quant`'s packed-constant resident loading
    // beyond qwen2/qwen3 to Mixtral/MoE archs) is a cross-cutting project (card 097/134 scope); do not
    // attempt it without checking in.
    // The GGUF must be a MODERN-format conversion (e.g. `hf download
    // mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf`), not
    // TheBloke's repo, whose older per-expert tensor layout this loader does not recognize (see the
    // findings above). Even with a compatible GGUF this test is confirmed to OOM-kill on a normal
    // single-GPU pod's host RAM (the second finding).
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "mixtral-8x7b-instruct-gguf/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
    )) else {
        return;
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load mixtral-8x7b-instruct Q4_K_M gguf");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Mixtral
    );
    // Instruct-tuned checkpoint: a raw prompt degenerates on poot and llama.cpp alike, so render through
    // the checkpoint's own chat template.
    let rc = runner.render_chat_value(
        &serde_json::json!([{
            "role": "user",
            "content": "What is the capital of France? Answer in one short sentence.",
        }]),
        None,
    );
    let toks = runner
        .generate(&rc.prompt, 40, |_| std::ops::ControlFlow::Continue(()))
        .expect("mixtral-8x7b-instruct generate");
    let text = runner.decode(&toks).expect("decode");
    eprintln!("mixtral-8x7b-instruct greedy: {text:?}");
    assert!(
        text.to_lowercase().contains("paris"),
        "mixtral-8x7b-instruct continuation was: {text:?}"
    );
}

#[test]
#[ignore = "held for POOT-738: a Mixtral GGUF names `general.architecture = llama`, a registered family, so the Runner refuses it and the driver's llama family does not carry experts yet"]
fn mixtral_8x7b_ptx_kv_masked_is_coherent() {
    // PTX twin of `mixtral_8x7b_gpu_kv_masked_is_coherent` above; see its doc comment for the VRAM
    // derivation. PTX uploads the same packed Q6_K/Q8_0 constants via `Runner::const_inputs` +
    // `PtxGraphExecutor::capture_decode`, so the same ~62.79GB (Q6_K) / ~57.16GB (Q8_0) total applies and
    // an 80GB-class GPU is required. CUDA has no 2GiB per-buffer cap, so the wgpu buffer-size paragraph
    // does not apply.
    //
    // Why this test exists: docs/updates/0691 root-caused a blocker for the wgpu path on RunPod:
    // `vulkaninfo` fails with `ERROR_INCOMPATIBLE_DRIVER` because the RunPod NVIDIA compute host does not
    // load the `nvidia_drm` kernel module (`/dev/dri` absent, `cap_sys_module` denied in the container).
    // PTX/CUDA needs only `libcuda.so.1`, and `generate_kv_gpu_masked_ptx` (update 0686) mirrors
    // `generate_kv_gpu_masked`'s per-arch dispatch, including the `mixtral_quant_gpu_supported`
    // Q6_K/Q8_0 guard.
    //
    // Host RAM matters as well as VRAM (update 0693). `Gguf::load` is `Self::from_bytes(std::fs::read(
    // path)?)` and `Gguf` owns a `bytes: Vec<u8>`, so the whole 38,380,817,824-byte file (35.75 GiB)
    // stays in host RAM for all of `Runner::load_gguf` (`g` is a local dropped when the function
    // returns, after every packed constant is built). Load-time peak = file bytes + ~62.79 GB
    // packed+dense constants + transient per-layer packing buffers (`gate_q` 469.8 MB, `up_q` 469.8 MB,
    // `stack_expert_rows`'s gate||up copy 939.5 MB, `down_q` 469.8 MB, freed each layer) = 38.38 + 62.79
    // + 2.4 = ~104 GB. Budget >= 110 GB; an A100-SXM-80GB pod advertises ~117 GB, so headroom is only
    // ~13 GB. `Runner::const_inputs`' `.cloned()` adds no second copy (`Tensor` is
    // `Arc<[f32]>`/`Arc<[i32]>`-backed), and the packed expert constants use `Tensor::int_packed`, which
    // omits the `i32 as f32` mirror `Tensor::int` would add (that would double the 56.37 GB expert
    // footprint). The 38.38 GB file half is avoidable with a streaming/mmap GGUF reader, a core-loader
    // change out of scope here.
    // The GGUF must be a MODERN-format conversion (e.g. `hf download
    // mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF Mixtral-8x7B-Instruct-v0.1.Q6_K.gguf`), not
    // TheBloke's repo, whose older per-expert tensor layout this loader does not recognize.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "mixtral-8x7b-instruct-gguf/mixtral-8x7b-instruct-v0.1.Q6_K.gguf"
    )) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load mixtral-8x7b-instruct Q6_K gguf");
    let exe = runner.load_on(&mut ptx).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Mixtral
    );
    assert!(
        runner
            .weight_formats()
            .get("model.layers.0.block_sparse_moe.experts.gate_up_proj.weight")
            .is_some(),
        "this checkpoint's routed experts should load packed (card 545a)"
    );

    // Instruct-tuned checkpoint: a raw prompt degenerates on poot and llama.cpp alike, so render through
    // the checkpoint's own chat template.
    let rc = runner.render_chat_value(
        &serde_json::json!([{
            "role": "user",
            "content": "What is the capital of France? Answer in one short sentence.",
        }]),
        None,
    );

    // Calibration first (see the wgpu test's rationale: flat per-step KV-cached decode). PTX's
    // captured-CUDA-graph first step also includes one-time capture/JIT cost (update 0686); the ETA print
    // uses the running average, so it converges toward the steady-state rate instead of staying skewed
    // by step 1.
    let max_new: usize = std::env::var("POOT_MIXTRAL_8X7B_MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let gen_start = std::time::Instant::now();
    let step = std::cell::RefCell::new(0usize);
    let toks = runner
        .generate_kv_gpu_cached(&rc.prompt, max_new, &mut ptx, exe, |_piece| {
            let mut n = step.borrow_mut();
            *n += 1;
            let elapsed = gen_start.elapsed().as_secs_f64();
            let avg_secs_per_step = elapsed / *n as f64;
            let remaining = max_new.saturating_sub(*n);
            eprintln!(
                "mixtral-8x7b-instruct PTX KV-cached progress: step {}/{max_new}, {elapsed:.1}s elapsed, \
                 {avg_secs_per_step:.2}s/step avg, ETA {:.1}s for the remaining {remaining} step(s) (first \
                 step includes one-time CUDA graph capture; flat per-step cost expected after that)",
                *n,
                avg_secs_per_step * remaining as f64
            );
            std::ops::ControlFlow::Continue(())
        })
        .expect("mixtral-8x7b-instruct PTX KV-cached masked decode");
    let total = gen_start.elapsed();
    eprintln!(
        "mixtral-8x7b-instruct PTX KV-cached: {max_new} tokens in {total:.2?} ({:.2}s/token avg) - a \
         reasonable 20-40 token coherence check would cost about {:.1}s at this measured per-token rate",
        total.as_secs_f64() / max_new.max(1) as f64,
        (total.as_secs_f64() / max_new.max(1) as f64) * 30.0
    );
    let text = runner.decode(&toks).expect("decode");
    eprintln!("mixtral-8x7b-instruct PTX KV-cached greedy ({max_new} tokens): {text:?}");
    assert!(!text.trim().is_empty(), "decoded text must be non-empty");
    // Assert the coherence bar ("paris") only once max_new is large enough for the answer to plausibly
    // appear, as in the wgpu test.
    if max_new >= 20 {
        assert!(
            text.to_lowercase().contains("paris"),
            "mixtral-8x7b-instruct PTX KV-cached continuation was: {text:?}"
        );
    }
}

#[test]
#[ignore = "held for POOT-738: a Mixtral GGUF names `general.architecture = llama`, a registered family, so the Runner refuses it and the driver's llama family does not carry experts yet"]
fn mixtral_tiny_gguf_ptx_reprefill_matches_cpu() {
    // PTX-parity follow-on for the GGUF loading path: `mixtral_tiny_ptx_reprefill_matches_cpu` covers the
    // safetensors-loaded runner. This loads the same llama.cpp-converted GGUF as
    // `mixtral_tiny_gguf_generates_finite_output` via `Runner::load_gguf` and compares
    // `generate_ptx_reprefill` against `generate` on the GGUF-loaded runner, checking that the
    // GGUF-populated `mixtral` field's top-2 MoE routing dispatches correctly through PTX/CUDA graphs.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "mixtral-tiny/mixtral-tiny-f32.gguf"
    )) else {
        return;
    };
    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!("no PTX GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load mixtral-tiny gguf");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Mixtral
    );
    let prompt = "The capital of France is";
    let max_new = 12;
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("mixtral-tiny gguf PTX re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("mixtral-tiny gguf CPU generate");
    eprintln!(
        "mixtral-tiny gguf ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "GGUF-loaded PTX MoE re-prefill decode must match the GGUF-loaded CPU eager reference"
    );
}

#[test]
#[ignore = "held for POOT-738: a Mixtral GGUF names `general.architecture = llama`, a registered family, so the Runner refuses it and the driver's llama family does not carry experts yet"]
fn mixtral_tiny_gguf_gpu_reprefill_matches_cpu() {
    // wgpu/RADV follow-on for the GGUF loading path, combining
    // `mixtral_tiny_gguf_ptx_reprefill_matches_cpu` (GGUF load) with
    // `mixtral_tiny_gpu_reprefill_matches_cpu` (wgpu dispatch via `generate_gpu_reprefill`, otherwise
    // only run against the safetensors-loaded runner). Checks that the GGUF-populated `mixtral` field's
    // top-2 MoE routing (topk gating -> gather-free indexed_matmul expert MLP -> weighted combine)
    // dispatches correctly through wgpu/Vulkan.
    //
    // Card 258 watchdog risk: Mixtral-tiny's untied lm_head is `vocab=32000, hidden=1024`
    // (`out_numel * K = 3.277e7`, `N=32000`), the same shapes as the safetensors checkpoint, which ran
    // clean (update 0647, ~0.6-1.0s/step, no ring timeout), well below the `N=201088`-`250880` hang
    // bracket.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "mixtral-tiny/mixtral-tiny-f32.gguf"
    )) else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load mixtral-tiny gguf");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::Mixtral
    );
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let prompt = "The capital of France is";
    let max_new = 12;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let gpu_toks = runner
        .generate_gpu_reprefill(prompt, max_new, &mut engine, exe, |piece| {
            let now = std::time::Instant::now();
            eprintln!(
                "  step +{:.2?} (total {:.2?}): {piece:?}",
                now - last,
                now - t0
            );
            last = now;
            std::ops::ControlFlow::Continue(())
        })
        .expect("mixtral-tiny gguf GPU re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("mixtral-tiny gguf CPU generate");
    eprintln!(
        "mixtral-tiny gguf gpu={:?} cpu={:?}",
        runner.decode(&gpu_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "GGUF-loaded wgpu MoE re-prefill decode must match the GGUF-loaded CPU eager reference"
    );
}
