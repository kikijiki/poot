//! Card 210 correctness gate, kept when the Qwen3-Next private runtime was deleted (card 595): the
//! zero-GDN-slot trace (`qwen3next_zero_gdn_slot_trace`) plans every eqn on the AMD backend (the planner has
//! no host plan; an unplannable eqn is a typed refusal, Card 626). CPU-only (no GPU or `rocm` feature): it builds the
//! trace with a tiny synthetic `Qwen3NextConfig` and classifies every eqn with `poot_graph_plan::plan_eqn`
//! (cf. `card188_prefill_plan_classify.rs`).

#[test]
fn card210_zero_gdn_slot_trace_is_plan_host_free() {
    use poot_models::qwen3next::{Qwen3NextConfig, qwen3next_zero_gdn_slot_trace};
    use poot_target::AmdArch;
    use poot_target::Backend;
    use poot_test_util::graph_fixtures::plan_eqn;

    let cfg = Qwen3NextConfig {
        vocab: 256,
        hidden: 64,
        n_layers: 8,
        full_attention_interval: 4,
        eps: 1e-6,
        max_pos: 4096,
        rotary_dim: 16,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 32,
        gdn_num_k_heads: 4,
        gdn_num_v_heads: 8,
        gdn_head_dim: 16,
        conv_k: 4,
        n_experts: 4,
        top_k: 2,
        expert_inter: 32,
        shared_inter: 32,
    };
    // Sanity: this cfg exercises BOTH branches of qwen3next_zero_gdn_slot_trace.
    assert!(cfg.is_attn_layer(3), "layer 3 expected full-attention");
    assert!(cfg.is_attn_layer(7), "layer 7 expected full-attention");
    assert!(!cfg.is_attn_layer(0), "layer 0 expected GDN");
    let attn_layers = (0..cfg.n_layers).filter(|&i| cfg.is_attn_layer(i)).count();
    assert_eq!(attn_layers, 2, "expected exactly 2 full-attention layers");

    let pool_slots = 4;
    let gdn_n_slots = 2;
    let slot = 1;
    let g = qwen3next_zero_gdn_slot_trace(&cfg, pool_slots, gdn_n_slots, slot);
    assert!(
        !g.eqns.is_empty(),
        "graph must have eqns to classify anything"
    );

    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    for eqn in &g.eqns {
        plan_eqn(
            &g,
            eqn,
            backend,
            &poot_test_util::device_caps::default_caps_for(backend),
        )
        .unwrap_or_else(|e| {
            panic!(
                "card210_zero_gdn_slot_trace_is_plan_host_free: plan_eqn failed on op={}: {e}",
                eqn.op.name()
            )
        });
    }
}
