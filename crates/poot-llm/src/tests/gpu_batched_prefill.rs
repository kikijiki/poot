//! Spec 023 sub-step 2: the wgpu/Arc batched-prefill GPU path.
//!
//! The batched-prefill GPU decode (`generate_kv_gpu_prefilled`: one multi-token prefill forward fills
//! the KV cache for positions [0,N), then single-token cached decode continues from pos=N) must
//! produce the same greedy token ids as the token-by-token GPU path (`generate_kv_gpu_cached`, which
//! fills the cache one position at a time).
//!
//! Marked #[ignore]: loads a ~2GB f32 model and needs a GPU. Run with:
//!   cargo nextest run -p poot-llm gpu_batched_prefill_matches_token_by_token --run-ignored all
//! Needs qwen2.5-0.5b under POOT_MODELS_DIR. Skips (passes) if the model or the GPU is absent.

use std::path::PathBuf;

use super::*;

fn model() -> Option<PathBuf> {
    poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b"))
}

/// The Arc Vulkan driver segfaults under concurrent device use; serialize the GPU tests in this
/// binary (nextest runs each test in its own process, but a merge into one binary must stay serial).
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
#[ignore = "loads ~2GB f32 + runs prefill+decode on the GPU (slow); run with --run-ignored all"]
fn gpu_batched_prefill_matches_token_by_token() {
    let _gpu_guard = gpu_lock();
    let Some(dir) = model() else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load model");
    // a multi-token prompt so the prefill batches several positions.
    let prompt = "The capital of France is Paris and the capital of Italy is";
    let max_new = 8;

    // batched prefill GPU path: one prefill forward fills the cache, decode continues from pos=N,
    // both on Card 546a/546b's executor contract.
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let t0 = std::time::Instant::now();
    let prefilled = runner
        .generate_kv_gpu_prefilled(prompt, max_new, &mut engine, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("batched-prefill GPU generate");
    let prefilled_ms = t0.elapsed().as_secs_f64() * 1e3;

    // reference: the token-by-token GPU path (cached single-token decode for every prompt position),
    // on a separate executable so the two cache histories never alias.
    let mut engine2 = poot_executor::Engine::new(poot_gpu::device::WgpuDevice::new().unwrap());
    let exe2 = runner.load_on(&mut engine2).unwrap();
    let t1 = std::time::Instant::now();
    let token_by_token = runner
        .generate_kv_gpu_cached(prompt, max_new, &mut engine2, exe2, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("token-by-token GPU generate");
    let tbt_ms = t1.elapsed().as_secs_f64() * 1e3;

    eprintln!(
        "batched prefill: {:?} ({prefilled_ms:.0}ms) vs token-by-token ({tbt_ms:.0}ms)",
        runner.decode(&prefilled).unwrap()
    );
    assert_eq!(
        prefilled, token_by_token,
        "batched-prefill greedy ids must equal the token-by-token GPU path"
    );
}

/// The PTX prefill garbles at n=864 (correct at n=5). Both backends share the kernelgen and neither
/// prefill was tested at large N, so reproduce on the Arc: a large prompt through the wgpu batched
/// prefill vs the CPU oracle's first token. If the Arc also diverges, the bug is in the shared
/// kernelgen (a large-shape Body); if it agrees, it is PTX-specific.
#[test]
#[ignore = "loads ~2GB f32 + a large-N prefill on the GPU + a CPU oracle forward (slow)"]
fn gpu_prefill_large_n_vs_cpu_oracle() {
    let _gpu_guard = gpu_lock();
    let Some(dir) = model() else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load model");
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    // Repetitive prompt; POOT_PREFILL_REPEAT sets the size (default 30 ~= 420 tokens). The PTX prefill
    // broke at n=864 (~62 repeats); run that here to see whether the Arc agrees (PTX-specific) or also
    // garbles (shared kernelgen). The CPU oracle forward is very slow at large n, so it only runs for
    // n <= 500; above that, inspect the printed wgpu continuation (coherent English vs degenerate).
    // POOT_PREFILL_PROMPT_FILE pins the prompt to a file (so the wgpu scan matches the PTX prompt exactly
    // for the PTX-vs-wgpu per-eqn diff); otherwise a repeated sentence sized by POOT_PREFILL_REPEAT.
    let prompt = match std::env::var("POOT_PREFILL_PROMPT_FILE").ok() {
        Some(f) => std::fs::read_to_string(&f)
            .expect("read prompt file")
            .trim_end()
            .to_string(),
        None => {
            let repeat: usize = std::env::var("POOT_PREFILL_REPEAT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30);
            "The history of artificial intelligence began in antiquity with myths and stories. "
                .repeat(repeat)
        }
    };
    let n = runner.encode(&prompt).expect("encode").len();

    let gpu_ids = runner
        .generate_kv_gpu_prefilled(&prompt, 6, &mut engine, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("wgpu prefill");
    let gpu_first = gpu_ids.get(n).copied();
    eprintln!(
        "n={n} wgpu_first={gpu_first:?} | wgpu_text={:?}",
        runner.decode(&gpu_ids[n..]).unwrap_or_default()
    );

    // The whole generated continuation, not only the first token: the decode steps after the large prefill
    // attend over the K/V it wrote (the carried state), so a corrupted cache changes a later token even when
    // the first one is right. The reference is the wgpu token-by-token path, which builds the same cache one
    // position at a time and stays cheap at any n (unlike the CPU oracle below). `Runner` exposes no
    // prefill logits or KV state to a test, so the generated ids are the widest comparison available here.
    // Card 546a's executor contract drives this reference path.
    let mut engine = poot_executor::Engine::new(poot_gpu::device::WgpuDevice::new().unwrap());
    let exe = runner.load_on(&mut engine).unwrap();
    let token_by_token = runner
        .generate_kv_gpu_cached(&prompt, 6, &mut engine, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("wgpu token-by-token");
    assert_eq!(
        gpu_ids, token_by_token,
        "wgpu batched-prefill continuation must equal the wgpu token-by-token continuation at n={n}"
    );

    if n <= 500 {
        // CPU oracle: one full prefill forward (next_token = argmax of the last-position logits).
        let cpu_first = runner.next_token(&gpu_ids[..n]).expect("cpu next_token");
        eprintln!("cpu_first={cpu_first}");
        assert_eq!(
            gpu_first,
            Some(cpu_first),
            "wgpu prefill first token must match the CPU oracle at n={n} (if this FAILS, the large-N bug is \
             in the shared kernelgen, not PTX-specific)"
        );
    }
}
