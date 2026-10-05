//! Card 246 (epic 129 B7): `Runner::load`'s qwen3_moe safetensors path: loading, the per-layer
//! dense/MoE `Qwen3MoeParams`, and the expert-fusion step in `build_weights` (`fuse_qwen3_moe_experts`).
//! Internal (`#[cfg(test)] mod` here, not `tests/coherence.rs`) to reach the private `qwen3_moe`/`cfg`
//! fields and the `pub(crate) bind` binder, which an external integration test cannot.

use super::super::*;

use poot_models::qwen3moe::trace_qwen3_moe_prefill;

/// Proves `Runner::load`'s weight-layout crosswalk (transpose + per-expert fuse) is correct:
/// `poot_models::qwen3moe`'s `cpu_oracle` test fused this same `yujiepan/qwen3-moe-tiny-random`
/// checkpoint by hand (bypassing `Runner`) and checked it against an independent pure-numpy
/// reimplementation of HF's `modeling_qwen3_moe.py`. This reuses those golden values (same tokens,
/// same checkpoint) through the production `Runner::load` + `build_weights` path, so a transpose/fuse
/// bug shows as a different argmax/logit set, not a rounding difference. Needs `qwen3-moe-tiny` under
/// POOT_MODELS_DIR.
#[test]
fn qwen3_moe_tiny_checkpoint_runner_load_matches_tracer_golden_values() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen3-moe-tiny"))
    else {
        return;
    };
    let runner = Runner::load(&dir).expect("load qwen3-moe-tiny via Runner");
    assert_eq!(runner.arch, "qwen3_moe");
    let mp = runner
        .qwen3_moe
        .clone()
        .expect("Runner::load must set qwen3_moe params for a qwen3_moe checkpoint");
    // Sanity: this checkpoint's decoder_sparse_step=2 must produce a mixed dense/MoE pattern (else
    // `fuse_qwen3_moe_experts` is not exercised), as in poot_models::qwen3moe's real-checkpoint test.
    assert!(
        mp.sparse_layer.iter().any(|&s| s) && mp.sparse_layer.iter().any(|&s| !s),
        "expected a mixed dense/MoE layer pattern, got {:?}",
        mp.sparse_layer
    );

    let tokens = [1u32, 2, 3, 4];
    let g = trace_qwen3_moe_prefill(runner.cfg, mp, tokens.len());
    let inputs = runner
        .bind(&g, &tokens)
        .expect("bind qwen3-moe prefill graph");
    let out = crate::core::cpu_oracle::cpu_eval(&g, &inputs).expect("cpu eval");
    assert_eq!(out.shape(), vec![1, 1, runner.cfg.vocab]);
    assert!(
        out.as_f32().unwrap().iter().all(|v| v.is_finite()),
        "logits must be finite"
    );

    // Same golden values as `poot_models::qwen3moe::tests::cpu_oracle::
    // qwen3_moe_tiny_random_checkpoint_matches_numpy_reference` (see its doc for provenance: an independent
    // numpy reimplementation of modeling_qwen3_moe.py, checked against transformers v4.53.3).
    let golden: [(usize, f32); 8] = [
        (27959, -8.709_933),
        (68261, -8.360_13),
        (182, 7.893_251),
        (53765, -7.761_153),
        (34591, -7.591_176_5),
        (586, -7.377_918_7),
        (120483, -7.355_760_6),
        (131148, 7.283_245_6),
    ];
    let mut argmax_id = 0usize;
    let mut argmax_val = f32::NEG_INFINITY;
    for (i, &v) in out.as_f32().unwrap().iter().enumerate() {
        if v > argmax_val {
            argmax_val = v;
            argmax_id = i;
        }
    }
    assert_eq!(
        argmax_id, 182,
        "Runner-loaded argmax should match the tracer's own real-checkpoint golden value \
             (got value {argmax_val})"
    );
    for (id, expect) in golden {
        let got = out.as_f32().unwrap()[id];
        assert!(
            (got - expect).abs() < 0.01,
            "token {id}: Runner-loaded={got} golden={expect} (diff {})",
            (got - expect).abs()
        );
    }
}

/// Card 247: `Runner::load_gguf`'s `qwen3moe` arm, cross-checked against `Runner::load`'s verified
/// safetensors path on the same model (executor equivalence, per AGENTS.md's verification stack; not a
/// second golden-value derivation).
///
/// `yujiepan/qwen3-moe-tiny-random` cannot be used: it sets `decoder_sparse_step: 2` (mixed dense/MoE
/// layers), which llama.cpp's `convert_hf_to_gguf.py` cannot represent for `qwen3moe` (it raises
/// `ValueError: Can not map tensor 'model.layers.0.mlp.down_proj.weight'`, since
/// `MODEL_TENSORS[MODEL_ARCH.QWEN3MOE]` in gguf-py/gguf/constants.py has no dense FFN tensor names), and
/// `src/models/qwen3moe.cpp`'s `load_arch_tensors` creates `ffn_gate_inp`/`ffn_{gate,up,down}_exps` for
/// every layer. llama.cpp's qwen3moe is MoE-every-layer only, matching every real release
/// (Qwen3-30B-A3B, Qwen3-235B-A22B: `decoder_sparse_step: 1`, `mlp_only_layers: []`).
///
/// So this uses a small self-authored fully-sparse `qwen3_moe` checkpoint (real
/// `transformers.Qwen3MoeForCausalLM`, fixed seed, tiny dims, `decoder_sparse_step: 1`,
/// `mlp_only_layers: []`), converted with the real `convert_hf_to_gguf.py`, then loaded through both
/// `Runner::load` (safetensors) and `Runner::load_gguf` and compared. A transpose/stacking bug in the
/// GGUF `qwen3moe` arm (`gguf_weights` in `gguf.rs`) would show as a materially different forward pass,
/// since both read bit-identical underlying weights.
#[test]
fn qwen3_moe_gguf_matches_safetensors_on_self_authored_sparse_checkpoint() {
    // See this test's doc comment for how to regenerate the fixture.
    let Some(st_dir) =
        poot_test_util::model_path(poot_test_util::checkpoint!("qwen3-moe-tiny-sparse"))
    else {
        return;
    };
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "qwen3-moe-tiny-sparse-gguf/model.gguf"
    )) else {
        return;
    };
    let st_runner = Runner::load(&st_dir).expect("load qwen3-moe-tiny-sparse safetensors");
    let gguf_runner = Runner::load_gguf(&gguf_path).expect("load qwen3-moe-tiny-sparse gguf");
    assert_eq!(st_runner.arch, "qwen3_moe");
    assert_eq!(gguf_runner.arch, "qwen3moe");

    let st_mp = st_runner
        .qwen3_moe
        .clone()
        .expect("safetensors Runner must set qwen3_moe params");
    let gguf_mp = gguf_runner
        .qwen3_moe
        .clone()
        .expect("gguf Runner must set qwen3_moe params (card 247)");
    assert_eq!(st_mp.n_experts, gguf_mp.n_experts);
    assert_eq!(st_mp.top_k, gguf_mp.top_k);
    assert_eq!(st_mp.inter, gguf_mp.inter);
    assert_eq!(st_mp.sparse_layer, gguf_mp.sparse_layer);
    // This fixture is fully sparse (the only layout llama.cpp's GGUF tooling supports); check the
    // per-layer tensor-presence detection found that, not a vacuously empty result.
    assert!(
        !gguf_mp.sparse_layer.is_empty() && gguf_mp.sparse_layer.iter().all(|&s| s),
        "expected every layer sparse, got {:?}",
        gguf_mp.sparse_layer
    );

    let tokens = [3u32, 7, 11, 2];
    let st_g = trace_qwen3_moe_prefill(st_runner.cfg, st_mp, tokens.len());
    let st_inputs = st_runner
        .bind(&st_g, &tokens)
        .expect("bind safetensors prefill graph");
    let st_out =
        crate::core::cpu_oracle::cpu_eval(&st_g, &st_inputs).expect("cpu eval (safetensors)");

    let gguf_g = trace_qwen3_moe_prefill(gguf_runner.cfg, gguf_mp, tokens.len());
    let gguf_inputs = gguf_runner
        .bind(&gguf_g, &tokens)
        .expect("bind gguf prefill graph");
    let gguf_out =
        crate::core::cpu_oracle::cpu_eval(&gguf_g, &gguf_inputs).expect("cpu eval (gguf)");

    assert_eq!(st_out.shape(), gguf_out.shape());
    assert!(st_out.as_f32().unwrap().iter().all(|v| v.is_finite()));
    assert!(gguf_out.as_f32().unwrap().iter().all(|v| v.is_finite()));
    // Safetensors vs gguf last-position logits: a divergence here is a real transpose/fuse bug in the new
    // qwen3moe GGUF arm, not float noise (both paths are f32, no quantization involved).
    poot_test_util::assert_close(st_out.as_f32().unwrap(), gguf_out.as_f32().unwrap(), 1e-3);
    // argmax must agree exactly, not just be numerically close.
    let argmax = |data: &[f32]| -> usize {
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap()
    };
    assert_eq!(
        argmax(st_out.as_f32().unwrap()),
        argmax(gguf_out.as_f32().unwrap()),
        "safetensors and gguf loaders must agree on the argmax token"
    );
}
