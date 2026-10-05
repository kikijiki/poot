use super::*;
use poot_runtime_common::DeviceBackend;
use std::ops::ControlFlow;

fn runner() -> Runner {
    use poot_load::gguf::{GgufValue, write_gguf};
    // granitemoe: a family still on the Runner (POOT-738), with two experts.
    let (h, inter, vocab, experts) = (8usize, 16usize, 8usize, 2usize);
    let mut seed = 0x123456789abcdefu64;
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
            .flat_map(|i| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let v = if name == "token_embd.weight" || name == "output.weight" {
                    if i / h == i % h { 3.0 } else { 0.0 }
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
    let path = poot_test_util::unique_temp_path("post-prefill-prepare.gguf");
    std::fs::write(&path, write_gguf(&kvs, &tensors)).unwrap();
    Runner::load_gguf(&path).unwrap()
}

#[test]
fn post_prefill_prepare_production_aba_and_stop_boundaries_gpu() {
    let Some(device) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Wgpu,
        poot_gpu::device::WgpuDevice::new(),
    ) else {
        return;
    };
    let mut runner = runner();
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let mut reference_engine =
        poot_executor::Engine::new(poot_gpu::device::WgpuDevice::new().unwrap());
    let reference_exe = runner.load_on(&mut reference_engine).unwrap();
    for text_path in [false, true] {
        let mut trajectories = Vec::new();
        for prompt in ["cde", "efg", "cde"] {
            let context = runner.encode(prompt).unwrap();
            assert_eq!(context.len(), 3);
            let mut calls = 0;
            let mut sink = |_: &str| {
                calls += 1;
                ControlFlow::Continue(())
            };
            let ids = if text_path {
                runner.generate_kv_gpu_prefilled(prompt, 5, &mut engine, exe, &mut sink)
            } else {
                runner.generate_kv_gpu_prefilled_tokens(
                    &context,
                    5,
                    &mut engine,
                    exe,
                    false,
                    &mut sink,
                )
            }
            .unwrap();
            // Every one of the 5 requested tokens reaches on_token exactly once (no decode step
            // runs before the prefill-derived token is delivered, R-546-3's entry-removal
            // discipline leaks no extra step either way).
            assert_eq!(calls, 5);
            assert_eq!(ids.len(), context.len() + 5);
            // Same executable, repeated calls: the shared carried K/V state must
            // reproduce the identical continuation for the identical prompt ("cde" first vs
            // third), and a different prompt ("efg") must diverge from it.
            let parent = runner
                .generate_kv_gpu_cached(prompt, 5, &mut reference_engine, reference_exe, |_| {
                    ControlFlow::Continue(())
                })
                .unwrap();
            assert_eq!(ids, parent);
            trajectories.push(ids);
        }
        assert_eq!(trajectories[0], trajectories[2]);
        assert_ne!(
            trajectories[0][3..],
            trajectories[1][3..],
            "A/B continuations must differ"
        );
        eprintln!("production A/B/A text={text_path}: {trajectories:?}");
        // max_new=0 preserves the existing prefill-only contract; one token never prepares decode.
        // cap (= context.len() + max_new) varies across these rows, so each gets its own
        // executable (a state name shared across differing cap/aval is a schema
        // error, not a resize).
        for (max_new, stop, eos) in [
            (0, false, false),
            (1, false, false),
            (5, true, false),
            (5, false, true),
        ] {
            let mut row_engine =
                poot_executor::Engine::new(poot_gpu::device::WgpuDevice::new().unwrap());
            let row_exe = runner.load_on(&mut row_engine).unwrap();
            let context = runner.encode("cde").unwrap();
            if eos {
                runner.eos = trajectories[0][context.len()];
            }
            let mut calls = 0;
            let mut sink = |_: &str| {
                calls += 1;
                if stop {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            };
            let ids = if text_path {
                runner.generate_kv_gpu_prefilled(
                    "cde",
                    max_new,
                    &mut row_engine,
                    row_exe,
                    &mut sink,
                )
            } else {
                runner.generate_kv_gpu_prefilled_tokens(
                    &context,
                    max_new,
                    &mut row_engine,
                    row_exe,
                    false,
                    &mut sink,
                )
            }
            .unwrap();
            let expected = usize::from(max_new > 0 && !eos);
            assert_eq!(calls, expected);
            assert_eq!(ids.len(), context.len() + expected);
            runner.eos = 99;
        }
    }
}
