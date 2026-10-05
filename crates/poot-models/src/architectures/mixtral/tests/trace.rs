use super::*;

#[test]
pub(crate) fn mixtral_prefill_validates_no_bias_and_uses_moe_every_layer() {
    let cfg = tiny_cfg();
    let mp = tiny_mp();
    let g = trace_mixtral_prefill(cfg, mp, 5);
    g.validate().expect("mixtral prefill graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);

    // FR-002: no bias constant anywhere.
    for id in &g.inputs {
        if let Storage::Const = g.values[*id].storage {
            let name = g.values[*id].name.as_deref().unwrap_or("");
            assert!(
                !name.ends_with(".bias"),
                "Mixtral has no bias, found {name}"
            );
        }
    }
    // Every layer declares a router constant: the MoE branch is unconditional (FR-001).
    let router_count = g
        .inputs
        .iter()
        .filter(|id| {
            matches!(g.values[**id].storage, Storage::Const)
                && g.values[**id]
                    .name
                    .as_deref()
                    .is_some_and(|n| n.ends_with("block_sparse_moe.gate.weight"))
        })
        .count();
    assert_eq!(router_count, cfg.layers, "one router per layer, no skips");
}

#[test]
pub(crate) fn mixtral_decode_kv_masked_validates_state_and_mask_shapes() {
    let cfg = tiny_cfg();
    let mp = tiny_mp();
    let cap = 16;
    let g = trace_mixtral_decode_kv_masked(cfg, mp, cap);
    g.validate().expect("mixtral decode graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
    assert_eq!(g.state.len(), 2 * cfg.layers);
    for (si, _so) in &g.state {
        assert_eq!(
            g.aval(*si).shape,
            vec![1, cfg.n_kv_heads, cap, cfg.head_dim]
        );
        assert_eq!(g.values[*si].storage, Storage::State);
    }
    let mask_id = g
        .inputs
        .iter()
        .find(|id| matches!(g.values[**id].storage, Storage::Slot(Slot::Mask)))
        .expect("Slot::Mask present");
    assert_eq!(g.values[*mask_id].aval.shape, vec![cap]);
}
