use crate::core::runner::Runner;

// Real-checkpoint smoke test for llama GGUF packed loading (card 545a): the local Llama-3.2-1B-Instruct
// Q4_K_M GGUF (the matching safetensors are HF-gated, so only the GGUF is needed) keeps every projection
// packed, recorded in the Runner's `WeightFormats` with its carriers in the weight map. CPU-only
// (Runner::load_gguf, no generate loop); skips if the model is absent from POOT_MODELS_DIR.
#[test]
fn llama_3_2_1b_q4_k_m_loads_every_projection_packed() {
    let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    )) else {
        return;
    };
    let runner = Runner::load_gguf(&path).expect("load the real llama-3.2-1b Q4_K_M GGUF");
    let formats = runner.weight_formats();
    for i in 0..runner.cfg.layers {
        for proj in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            let name = format!("model.layers.{i}.{proj}.weight");
            let packed = formats
                .get(&name)
                .unwrap_or_else(|| panic!("{name} should be stored packed"));
            // Q4_K_M mixes Q4_K (most projections) with Q6_K (attn_v/ffn_down on some layers).
            assert!(
                matches!(
                    packed.weight.format(),
                    poot_quant::format::WeightFormat::Q4_K | poot_quant::format::WeightFormat::Q6_K
                ),
                "{name}: {:?}",
                packed.weight.format()
            );
            for linear_id in &packed.linear_ids {
                let carrier = poot_graph_ir::packed_source::PackedSourceName::new(
                    linear_id,
                    poot_quant::SourceRole::Blocks,
                );
                assert!(
                    runner.weights.contains_key(carrier.as_ref()),
                    "missing packed carrier {carrier}"
                );
            }
        }
    }
}
