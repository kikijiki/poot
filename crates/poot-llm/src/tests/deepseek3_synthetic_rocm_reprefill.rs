// Moved in-crate (card 622): references a pub(crate)-only item (poot has no users, tests are not consumers).

//! DeepSeek-V3 ROCm/AMD parity. The GGUF loader (update 0791) added `self.deepseek3` dispatch arms on
//! every backend, mirroring `self.deepseek2`, and update 0793 verified PTX/NVIDIA on a rented L40S.
//! `Runner::generate_rocm_reprefill` (`crates/poot-llm/src/backends/rocm_vulkan.rs`) already has a
//! `self.deepseek3.is_some()` arm dispatching to `trace_deepseek3_prefill(dp.cfg, dp.moe,
//! tokens.len())`, alongside the `deepseek2`/`bloom`/`mpt`/`smollm3`/`mixtral`/`olmoe`/`gpt_oss` arms,
//! so only this test was needed.
//!
//! No real DeepSeek-V3 checkpoint exists (update 0791 "not done" item 2), so this reuses the
//! synthetic-GGUF fixture of `write_synthetic_deepseek3_gguf`/
//! `deepseek3_synthetic_ptx_reprefill_matches_cpu` (`coherence.rs`): 3 layers, layer 0 dense, layers
//! 1-2 routed, `n_group=3`/`topk_group=1` group limiting on 6 experts, and the V3-only
//! `exp_probs_b.bias` selection-correction tensor. It is duplicated here because `coherence.rs` is not
//! `rocm`-gated and must compile without the feature (same reason as
//! `card135d_deepseek2_rocm_reprefill.rs`).
//!
//! The checkpoint is synthetic (random fill), so no coherent text is expected. The test checks that
//! ROCm dispatch of DeepSeek-V3's MLA attention (compressed-latent cache, interleaved-pair RoPE),
//! group-limited MoE routing and the `exp_probs_b.bias` term yields the exact same tokens as the CPU
//! eager reference, on MI300X hardware.
//!
//! Run alone (HSA on the dev box allows one active queue across processes; irrelevant on a dedicated
//! pod, but kept for consistency with the other `cardNNN_*_rocm_reprefill.rs` files):
//! `cargo test -p poot-llm --features rocm --release --test deepseek3_synthetic_rocm_reprefill --
//! --ignored --test-threads=1 --nocapture`.

use super::*;

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

    // Same xorshift fill as `deepseek3_load_tests` (runner.rs) and `write_synthetic_deepseek3_gguf`
    // (coherence.rs), so magnitudes match a non-degenerate fixture.
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
    let path = poot_test_util::unique_temp_path("poot_llm_deepseek3_synthetic_rocm_fixture.gguf");
    std::fs::write(&path, &bytes).expect("write synthetic deepseek3 gguf fixture");
    path
}

#[test]
#[ignore = "builds a fully SYNTHETIC deepseek3-style GGUF in-process (no real DeepSeek-V3 checkpoint \
            exists anywhere to download - see docs/updates/0791-deepseek3-gguf-loader.md), loads it \
            through Runner::load_gguf, generates on real ROCm/AMD hardware via the generic re-prefill \
            GPU path, compared token-for-token against the CPU eager reference; run with --ignored \
            --release"]
fn deepseek3_synthetic_rocm_reprefill_matches_cpu() {
    let path = write_synthetic_deepseek3_gguf();
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no ROCm GPU ({e}); skipping");
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load_gguf(&path).expect("load synthetic deepseek3 gguf");
    let exe = runner.load_on(&mut rocm).expect("load_on");
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV3
    );
    // Uses all 6 vocab entries (this fixture's byte-level BPE has no merges, so each ASCII char is its
    // own token, ids 0..6 in declaration order).
    let prompt = "abcdef";
    let max_new = 10;
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
        .expect("synthetic deepseek3 ROCm re-prefill generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("synthetic deepseek3 CPU generate");
    eprintln!(
        "deepseek3-synthetic rocm={:?} cpu={:?}",
        runner.decode(&rocm_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        rocm_toks, cpu_toks,
        "GGUF-loaded ROCm re-prefill decode for a synthetic DeepSeek-V3-style checkpoint \
         (group-limited MoE routing + MLA attention + the V3-only exp_probs_b.bias \
         selection-correction term) must match the GGUF-loaded CPU eager reference"
    );
}
