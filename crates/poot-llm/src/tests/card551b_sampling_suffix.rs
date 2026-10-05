//! Card 551b acceptance: the Runner's sampling suffix
//! (`driver::suffix`) and the one step-and-pick (`core::decode_step::Runner::pick_step`)
//! every decode loop now calls.
//!
//! - **SC-001**: a seeded dense decode through the Runner gives tokens equal to the shared
//!   `SampleToken`/`RandomUniform` semantics evaluated on that same device's read-back logits, step
//!   by step, on wgpu and ROCm (PTX joins the next pod batch), for a plain `Gumbel` rule and
//!   (review F3) a `GumbelTopKTopP` rule (`top_p < 1.0`). Demonstrated without a second, independent
//!   oracle: the same scenario run twice with identically-seeded `Sampler`s, once through the device
//!   suffix (`rule_of` admits it) and once forced onto the host path by a no-op out-of-vocab
//!   `logit_bias` entry (adds 0 to a token id outside the vocab, so `adjusts_logits()` is true and
//!   `rule_of` returns `None`, but no logit value changes) - both consume exactly one
//!   `Sampler::next_device_seed` draw per *committed* decode step (never on a
//!   `generate_kv_gpu_cached_sampled` prompt-replay throwaway step), so an identical seed must give
//!   an identical token sequence if, and only if, the host and device paths share one semantics.
//! - **SC-002**: a sampled row at a real temperature gives a recorded token sequence that differs
//!   from the greedy sequence on the same fixture/seed (the sampler is actually applied).
//! - **SC-003**: a dense decode whose logits are forced NaN at the first step returns the typed
//!   sampler fault (`value: None` - the device readback never carries the offending value, only the
//!   index, R-551a-2) for both the greedy and the sampled device-suffix path, on wgpu and ROCm.
//! - **SC-004**: a request the suffix cannot express (a non-zero `logit_bias`) takes the host path
//!   and still gives the token of the shared semantics on its *transformed* logits (the sampler's
//!   bias-adjusted row), demonstrated the same way as SC-001's host leg but with a bias that
//!   actually changes a logit.
//!
//! Every row uses the same tiny synthetic granitemoe GGUF fixture (model-free, no `POOT_MODELS_DIR`
//! checkpoint; a family still on the Runner): one layer, two experts, `hidden=8`, `vocab=8`,
//! deterministic pseudo-random weights (card `post_prefill_prepare.rs`'s fixture, same shapes). `POOT_REQUIRE_WGPU=1`/`POOT_REQUIRE_ROCM=1`
//! turn a missing device into a failure rather than a skip (required-device-case lists, card 670).

use super::*;
use crate::core::sampler::Sampler;
use poot_runtime_common::DeviceBackend;
use std::collections::HashMap;
use std::ops::ControlFlow;

/// The tiny synthetic granitemoe fixture (one layer, `hidden=8`, `vocab=8`): `post_prefill_prepare.rs`'s
/// fixture shape, under this card's own file name so a corrupted-weight mutation here never aliases
/// that file's temp path.
fn runner() -> Runner {
    use poot_load::gguf::{GgufValue, write_gguf};
    // granitemoe: a family still on the Runner (POOT-738), with two experts.
    let (h, inter, vocab, experts) = (8usize, 16usize, 8usize, 2usize);
    let mut seed = 0x005a_5b1b_c551_u64;
    let mut tensors = Vec::new();
    for (name, shape) in [
        ("token_embd.weight", vec![h, vocab]),
        ("output.weight", vec![h, vocab]),
        ("output_norm.weight", vec![h]),
        ("blk.0.attn_q.weight", vec![h, h]),
        ("blk.0.attn_k.weight", vec![h, h]),
        ("blk.0.attn_v.weight", vec![h, h]),
        ("blk.0.attn_output.weight", vec![h, h]),
        ("blk.0.attn_norm.weight", vec![h]),
        ("blk.0.ffn_norm.weight", vec![h]),
        ("blk.0.ffn_gate_inp.weight", vec![h, experts]),
        ("blk.0.ffn_gate_exps.weight", vec![h, inter, experts]),
        ("blk.0.ffn_up_exps.weight", vec![h, inter, experts]),
        ("blk.0.ffn_down_exps.weight", vec![inter, h, experts]),
    ] {
        let data: Vec<u8> = (0..shape.iter().product::<usize>())
            .flat_map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let v = if name == "token_embd.weight" || name == "output.weight" {
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0
                } else if name.contains("norm") {
                    1.0
                } else {
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6
                };
                v.to_le_bytes()
            })
            .collect();
        tensors.push((name, shape.into_iter().map(|n| n as u64).collect(), 0, data));
    }
    let kvs = vec![
        ("general.architecture", GgufValue::Str("granitemoe".into())),
        ("granitemoe.embedding_length", GgufValue::U32(h as u32)),
        ("granitemoe.block_count", GgufValue::U32(1)),
        ("granitemoe.attention.head_count", GgufValue::U32(1)),
        ("granitemoe.attention.head_count_kv", GgufValue::U32(1)),
        (
            "granitemoe.feed_forward_length",
            GgufValue::U32(inter as u32),
        ),
        ("granitemoe.context_length", GgufValue::U32(64)),
        ("granitemoe.expert_count", GgufValue::U32(experts as u32)),
        ("granitemoe.expert_used_count", GgufValue::U32(1)),
        ("granitemoe.embedding_scale", GgufValue::F32(1.0)),
        ("granitemoe.attention.scale", GgufValue::F32(0.35355338)),
        ("granitemoe.residual_scale", GgufValue::F32(1.0)),
        ("granitemoe.logit_scale", GgufValue::F32(1.0)),
        ("tokenizer.ggml.eos_token_id", GgufValue::U32(99)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "c", "d", "e", "f", "g", "ab"]
                    .into_iter()
                    .map(|s| GgufValue::Str(s.into()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("a b".into())]),
        ),
    ];
    let path = poot_test_util::unique_temp_path("card551b-sampling-suffix.gguf");
    std::fs::write(&path, write_gguf(&kvs, &tensors)).unwrap();
    Runner::load_gguf(&path).unwrap()
}

/// `logit_bias` keyed at an id outside the fixture's 8-token vocab adds nothing to any real logit
/// (`Sampler::adjust`'s `logits.get_mut(tok)` is `None` for it) but still makes `adjusts_logits()`
/// true, forcing `suffix::rule_of` to return `None` (the host path) - SC-001's "same
/// scenario, host path" leg.
fn no_op_host_forcing_bias() -> HashMap<u32, f32> {
    HashMap::from([(999_999u32, 0.0)])
}

/// SC-001's device-vs-host-oracle comparison, shared by the plain-Gumbel and `top_p < 1.0` legs and
/// by every backend (wgpu/ROCm): runs the same seeded scenario twice, once through the device
/// suffix and once forced onto the host path by [`no_op_host_forcing_bias`], and asserts the token
/// sequences agree. `make_device` builds a fresh device per call (the skip check already confirmed
/// one is available).
fn assert_sc001_device_matches_host<D: poot_executor::Device>(
    label: &str,
    runner: &Runner,
    device: D,
    make_second_device: impl FnOnce() -> D,
    sampler_factory: impl Fn(u64) -> Sampler,
    seed: u64,
) {
    let prompt = "cde";
    let max_new = 6;

    let mut device_engine = poot_executor::Engine::new(device);
    let device_exe = runner.load_on(&mut device_engine).unwrap();
    let mut device_sampler = sampler_factory(seed);
    let device_tokens = runner
        .generate_kv_gpu_cached_sampled(
            prompt,
            max_new,
            &mut device_engine,
            device_exe,
            &mut device_sampler,
            &[],
            |_| ControlFlow::Continue(()),
        )
        .unwrap();

    let mut host_engine = poot_executor::Engine::new(make_second_device());
    let host_exe = runner.load_on(&mut host_engine).unwrap();
    let mut host_sampler = sampler_factory(seed).with_logit_bias(no_op_host_forcing_bias());
    assert!(
        !host_sampler.is_simple_temperature(),
        "the no-op bias must force the host path"
    );
    let host_tokens = runner
        .generate_kv_gpu_cached_sampled(
            prompt,
            max_new,
            &mut host_engine,
            host_exe,
            &mut host_sampler,
            &[],
            |_| ControlFlow::Continue(()),
        )
        .unwrap();

    assert_eq!(
        device_tokens, host_tokens,
        "SC-001 ({label}): the device suffix (Sample(rule) entry) and the host pick (Logits entry \
         + Sampler::pick, forced by a no-op bias) must agree token-for-token on identical read-back \
         logits given the same seed"
    );
    eprintln!("SC-001 {label}: device={device_tokens:?} host={host_tokens:?}");
}

#[test]
fn sc001_device_suffix_matches_host_oracle_on_same_readback_logits_wgpu() {
    let Some(device) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Wgpu,
        poot_gpu::device::WgpuDevice::new(),
    ) else {
        return;
    };
    assert_sc001_device_matches_host(
        "wgpu/Gumbel",
        &runner(),
        device,
        || poot_gpu::device::WgpuDevice::new().unwrap(),
        |seed| Sampler::new(0.8, 0, 1.0, seed),
        0xC0FFEE,
    );
}

/// Card 551b: the ROCm sibling of the wgpu SC-001 row above (same fixture, same
/// comparison), closing the gap the prior round's commit message overstated.
#[cfg(feature = "rocm")]
#[test]
fn sc001_device_suffix_matches_host_oracle_on_same_readback_logits_rocm() {
    let Some(device) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Rocm,
        poot_rocm_gpu::device::RocmDevice::new(),
    ) else {
        return;
    };
    assert_sc001_device_matches_host(
        "rocm/Gumbel",
        &runner(),
        device,
        || poot_rocm_gpu::device::RocmDevice::new().unwrap(),
        |seed| Sampler::new(0.8, 0, 1.0, seed),
        0xC0FFEE,
    );
}

/// Card 551b: a `top_p < 1.0` leg of SC-001 (every other row above uses `top_p = 1.0`,
/// which the card text itself flags as the untested tier - "tier 2 for top-p", ADR-0114), with
/// min-p and top-k both inactive so top-p alone does the truncating: the regime card 677 fixed
/// (`poot-eval`'s `sample_one_row` used to start its pre-top-p floor at `-inf` whenever neither
/// min-p nor top-k had already raised it, pinning the bisection and silently disabling top-p).
/// Exercises the quantized top-p bisection on both the device and the host.
#[test]
fn sc001_device_suffix_matches_host_oracle_with_top_p_wgpu() {
    let Some(device) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Wgpu,
        poot_gpu::device::WgpuDevice::new(),
    ) else {
        return;
    };
    assert_sc001_device_matches_host(
        "wgpu/GumbelTopKTopP",
        &runner(),
        device,
        || poot_gpu::device::WgpuDevice::new().unwrap(),
        |seed| Sampler::new(0.8, 0, 0.9, seed),
        0xC0FFEE,
    );
}

/// ROCm sibling of the `top_p < 1.0` leg above.
#[cfg(feature = "rocm")]
#[test]
fn sc001_device_suffix_matches_host_oracle_with_top_p_rocm() {
    let Some(device) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Rocm,
        poot_rocm_gpu::device::RocmDevice::new(),
    ) else {
        return;
    };
    assert_sc001_device_matches_host(
        "rocm/GumbelTopKTopP",
        &runner(),
        device,
        || poot_rocm_gpu::device::RocmDevice::new().unwrap(),
        |seed| Sampler::new(0.8, 0, 0.9, seed),
        0xC0FFEE,
    );
}

#[test]
fn sc002_sampled_temperature_actually_perturbs_the_pick_wgpu() {
    let Some(device) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Wgpu,
        poot_gpu::device::WgpuDevice::new(),
    ) else {
        return;
    };
    let runner = runner();
    let prompt = "cde";
    let max_new = 6;

    let mut greedy_engine = poot_executor::Engine::new(device);
    let greedy_exe = runner.load_on(&mut greedy_engine).unwrap();
    let greedy_tokens = runner
        .generate_kv_gpu_cached(prompt, max_new, &mut greedy_engine, greedy_exe, |_| {
            ControlFlow::Continue(())
        })
        .unwrap();

    let mut sampled_engine =
        poot_executor::Engine::new(poot_gpu::device::WgpuDevice::new().unwrap());
    let sampled_exe = runner.load_on(&mut sampled_engine).unwrap();
    let mut sampler = Sampler::new(1.5, 0, 1.0, 0x5EED_u64);
    let sampled_tokens = runner
        .generate_kv_gpu_cached_sampled(
            prompt,
            max_new,
            &mut sampled_engine,
            sampled_exe,
            &mut sampler,
            &[],
            |_| ControlFlow::Continue(()),
        )
        .unwrap();

    eprintln!("SC-002 wgpu: greedy={greedy_tokens:?} sampled(T=1.5)={sampled_tokens:?}");
    assert_ne!(
        sampled_tokens, greedy_tokens,
        "SC-002: a T=1.5 sampled row must differ from the greedy row on this fixture/seed \
         (recorded literals: greedy={greedy_tokens:?}, sampled={sampled_tokens:?})"
    );
}

/// Poisons the embedding table's row for `runner.encode(prompt)[0]` with NaN (the GGUF loader
/// translates `token_embd.weight` into this same internal name safetensors checkpoints use), so the
/// very first decode step's embedding (and therefore every downstream logit) is non-finite - the
/// SC-003 reproduction every row below shares.
fn poison_first_prompt_token_embedding(runner: &mut Runner, prompt: &str) {
    let first = runner.encode(prompt).unwrap()[0];
    let embed = runner.weights["model.embed_tokens.weight"].clone();
    let dense = embed.as_host().expect("dense weight");
    let hidden = runner.cfg.hidden;
    let mut data = dense.as_f32().unwrap().to_vec();
    data[first as usize * hidden..(first as usize + 1) * hidden].fill(f32::NAN);
    runner.weights.insert(
        "model.embed_tokens.weight".to_string(),
        poot_eval::Value::from(poot_tensor::HostTensor::f32(dense.shape().to_vec(), data)),
    );
}

/// `result` is Card 601's typed sampler fault with a device-readback-shaped payload: the index is
/// known, the value is not (R-551a-2 - the suffix's `[rows,2]` readback carries only the index).
fn assert_device_non_finite_fault<T: std::fmt::Debug>(result: Result<T>) {
    match result {
        Err(RunnerError::Sampler(fault)) => match *fault {
            SamplerFault::NonFiniteLogit { value, .. } => {
                assert_eq!(
                    value, None,
                    "a device suffix readback must never carry the non-finite value, only the index"
                );
            }
            other => panic!("expected NonFiniteLogit, got {other:?}"),
        },
        other => panic!("expected a typed sampler fault, got {other:?}"),
    }
}

#[test]
fn sc003_nan_logits_are_a_typed_fault_greedy_and_sampled_wgpu() {
    let Some(device) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Wgpu,
        poot_gpu::device::WgpuDevice::new(),
    ) else {
        return;
    };
    let prompt = "cde";
    let mut runner = runner();
    poison_first_prompt_token_embedding(&mut runner, prompt);

    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let greedy_result =
        runner.generate_kv_gpu_cached(prompt, 3, &mut engine, exe, |_| ControlFlow::Continue(()));
    assert_device_non_finite_fault(greedy_result);

    let mut sampled_engine =
        poot_executor::Engine::new(poot_gpu::device::WgpuDevice::new().unwrap());
    let sampled_exe = runner.load_on(&mut sampled_engine).unwrap();
    let mut sampler = Sampler::new(0.8, 0, 1.0, 1);
    let sampled_result = runner.generate_kv_gpu_cached_sampled(
        prompt,
        3,
        &mut sampled_engine,
        sampled_exe,
        &mut sampler,
        &[],
        |_| ControlFlow::Continue(()),
    );
    assert_device_non_finite_fault(sampled_result);
}

#[cfg(feature = "rocm")]
#[test]
fn sc003_nan_logits_are_a_typed_fault_greedy_and_sampled_rocm() {
    let Some(device) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Rocm,
        poot_rocm_gpu::device::RocmDevice::new(),
    ) else {
        return;
    };
    let prompt = "cde";
    let mut runner = runner();
    poison_first_prompt_token_embedding(&mut runner, prompt);

    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let greedy_result =
        runner.generate_kv_gpu_cached(prompt, 3, &mut engine, exe, |_| ControlFlow::Continue(()));
    assert_device_non_finite_fault(greedy_result);

    let mut sampled_engine =
        poot_executor::Engine::new(poot_rocm_gpu::device::RocmDevice::new().unwrap());
    let sampled_exe = runner.load_on(&mut sampled_engine).unwrap();
    let mut sampler = Sampler::new(0.8, 0, 1.0, 1);
    let sampled_result = runner.generate_kv_gpu_cached_sampled(
        prompt,
        3,
        &mut sampled_engine,
        sampled_exe,
        &mut sampler,
        &[],
        |_| ControlFlow::Continue(()),
    );
    assert_device_non_finite_fault(sampled_result);
}

#[test]
fn sc004_a_host_only_bias_still_gives_the_shared_semantics_token_on_its_transformed_logits() {
    // No GPU, no Runner: a request the suffix cannot express (R-551b-1's `rule_of` gate - a real
    // `logit_bias` forces the host path) must still give the token of the shared `SampleToken`
    // semantics on its *transformed* logits (R-551b-2). The independent reference below computes
    // that token through `suffix::eval_sample_token` directly, from a seed drawn by its
    // own freshly-built `Sampler` - a different call path than `Sampler::pick`'s, so a regression
    // in `pick_from` (e.g. reverting to the deleted xorshift draw) makes `pick`'s answer diverge
    // from it instead of trivially agreeing by construction.
    let logits = [0.3f32, -0.2, 1.5, 0.9, -0.7, 2.0, 0.1, -1.0];
    let temp = 1.2f32;
    let seed = 0x5EED_u64;
    let forbidden = 5u32; // the raw argmax (logits[5] = 2.0)
    let bias = HashMap::from([(forbidden, -1.0e6)]);

    let mut adjusted = logits;
    adjusted[forbidden as usize] += -1.0e6;
    let mut reference_sampler = Sampler::new(temp, 0, 1.0, seed).with_logit_bias(bias.clone());
    let device_seed = reference_sampler.next_device_seed();
    let rule = crate::driver::suffix::temperature_rule(0, 1.0);
    let (expected_token, non_finite) =
        crate::driver::suffix::eval_sample_token(rule, &adjusted, device_seed, temp, 0.0, 0, 1.0)
            .unwrap();
    assert_eq!(non_finite, -1, "the adjusted row has no non-finite logit");

    let mut real_sampler = Sampler::new(temp, 0, 1.0, seed).with_logit_bias(bias);
    assert!(
        !real_sampler.is_simple_temperature(),
        "a real bias forces the host path"
    );
    let got = real_sampler.pick(&logits).unwrap();

    assert_eq!(
        got, expected_token as usize,
        "SC-004: Sampler::pick (the host path, forced by a real bias) must give the token the \
         shared SampleToken semantics give on the bias-adjusted logits: got {got}, expected \
         {expected_token}"
    );
    assert_ne!(
        got, forbidden as usize,
        "the -1e6 bias must exclude the raw argmax from the draw"
    );
    eprintln!(
        "SC-004: bias-adjusted draw={got} (raw argmax={forbidden} excluded), shared-semantics \
         reference={expected_token}"
    );
}
