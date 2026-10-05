use super::*;

#[test]
pub(crate) fn qwen3_moe_prefill_validates_mixed_dense_and_sparse_layers() {
    let cfg = qwen3_moe_cfg();
    // alternating dense/MoE layers, like the decoder_sparse_step=2 pattern of the real tiny checkpoint.
    let mp = moe_params(&cfg, vec![false, true, false, true]);
    let g = trace_qwen3_moe_prefill(cfg, mp, 5);
    g.validate()
        .expect("qwen3-moe prefill graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
}

#[test]
pub(crate) fn qwen3_moe_decode_kv_masked_validates_all_sparse() {
    let cfg = qwen3_moe_cfg();
    let mp = moe_params(&cfg, vec![true; cfg.layers]);
    let g = trace_qwen3_moe_decode_kv_masked(cfg, mp, 16);
    g.validate()
        .expect("qwen3-moe kv decode graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
    assert_eq!(g.state.len(), 2 * cfg.layers);
    for (si, _so) in &g.state {
        assert_eq!(g.aval(*si).shape, vec![1, cfg.n_kv_heads, 16, cfg.head_dim]);
        assert_eq!(g.values[*si].storage, Storage::State);
    }
}

#[test]
pub(crate) fn qwen3_moe_prefill_kv_carries_same_state_shape_as_decode() {
    let cfg = qwen3_moe_cfg();
    let (n, cap) = (5, 16);
    let mp = moe_params(&cfg, vec![false, true, true, false]);
    let g = trace_qwen3_moe_prefill_kv(cfg, mp, n, cap);
    g.validate()
        .expect("qwen3-moe kv prefill graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
    assert_eq!(g.state.len(), 2 * cfg.layers);
    for (si, _so) in &g.state {
        assert_eq!(
            g.aval(*si).shape,
            vec![1, cfg.n_kv_heads, cap, cfg.head_dim]
        );
        assert_eq!(g.values[*si].storage, Storage::State);
    }
}

#[test]
#[should_panic(expected = "sparse_layer")]
pub(crate) fn sparse_layer_length_mismatch_panics_not_silently_wrong() {
    let cfg = qwen3_moe_cfg();
    // bypass moe_params' length check to exercise the tracer's guard directly.
    let mp = Qwen3MoeParams {
        n_experts: 6,
        top_k: 2,
        inter: 12,
        sparse_layer: vec![true; cfg.layers - 1], // one short
    };
    let _ = trace_qwen3_moe_prefill(cfg, mp, 3);
}
