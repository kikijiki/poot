//! DeepSeek-V3.2 (DSA, spec 277) checkpoint loader: a real-name-aware per-layer safetensors weight loader for
//! `poot_models::deepseek32`, following `nemotron_h_load.rs`. Unlike Nemotron-H, DeepSeek-V3.2 uses the shared
//! `poot_load::Qwen2HfConfig` that `is_deepseek2()`/`is_deepseek3()` already use in `runner.rs`: it is
//! architecturally DeepSeek-V3.2-Exp plus DSA, so every MLA/MoE config field is unchanged and DSA adds three
//! fields (`index_n_heads`/`index_head_dim`/`index_topk`, on `Qwen2HfConfig`) plus the indexer's four
//! per-layer tensors.
//!
//! **config.json** (`huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json`):
//! `model_type: "deepseek_v32"` (distinct from V3's `"deepseek_v3"`, see [`Qwen2HfConfig::is_deepseek32`]),
//! `architectures: ["DeepseekV32ForCausalLM"]`, the MLA/MoE fields `is_deepseek3()` already reads
//! (`q_lora_rank: 1536`, `kv_lora_rank: 512`, `qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`,
//! `v_head_dim: 128`, `n_routed_experts: 256`, `n_shared_experts: 1`, `num_experts_per_tok: 8`,
//! `moe_intermediate_size: 2048`, `n_group: 8`, `topk_group: 4`, `first_k_dense_replace: 3`,
//! `routed_scaling_factor: 2.5`, YaRN `rope_scaling`), plus the DSA fields at the same top level:
//! `index_n_heads: 64`, `index_head_dim: 128`, `index_topk: 2048`.
//!
//! **Tensor names** (`.../DeepSeek-V3.2-Exp/raw/main/model.safetensors.index.json`): every per-layer MLA/MoE
//! name is identical to DeepSeek-V3's (`self_attn.q_a_proj.weight`, `self_attn.q_a_layernorm.weight`,
//! `self_attn.q_b_proj.weight`, `self_attn.kv_a_proj_with_mqa.weight`, `self_attn.kv_a_layernorm.weight`,
//! `self_attn.kv_b_proj.weight`, `self_attn.o_proj.weight`, `mlp.{gate,up,down}_proj.weight` on dense layers,
//! and on MoE layers `mlp.gate.weight`, `mlp.gate.e_score_correction_bias`,
//! `mlp.experts.{e}.{gate,up,down}_proj.weight`, `mlp.shared_experts.{gate,up,down}_proj.weight`), plus the
//! four DSA indexer tensors documented in `poot_models::deepseek32`: `self_attn.indexer.wq_b.weight`,
//! `self_attn.indexer.wk.weight`, `self_attn.indexer.weights_proj.weight`, `self_attn.indexer.k_norm.{weight,bias}`.
//! The real checkpoint is FP8-quantized: every `.weight` except `weights_proj.weight`/`k_norm.*` has a sibling
//! `.weight_scale_inv` (`[ceil(out/128), ceil(in/128)]`, `weight_block_size: [128, 128]` in the
//! `quantization_config`).
//!
//! **Block FP8 stays packed (Card 654).** Every FP8 linear (`F8_E4M3` weight with an F32, BF16
//! or E8M0 `[ceil(out/128), ceil(in/128)]` `.weight_scale_inv` sibling, `weight_block_size: [128, 128]`; e.g.
//! `model.layers.3.mlp.experts.0.gate_proj.weight` is `F8_E4M3` `[2048, 7168]` with a `[16, 56]` scale) is packed
//! by [`poot_load::safetensors::pack_quantized_linears`] into one `E4m3Block128` owner straight from its
//! checkpoint bytes, never dequantized on the host; a non-finite scale (E8M0 `0xff`) is refused there. Every
//! weight is then placed through [`crate::checkpoint::place`]: a projection as its stored owner (packed or
//! dense), each routed expert's `gate_proj`/`up_proj` pair row-concatenated into one `gate||up` owner
//! ([`StoredRows::concat`], which refuses FP8 operands whose row count is not a multiple of 128, so no scale
//! block straddles the boundary), and the experts stacked one owner per expert. The traced graph then binds the
//! packed owners through `Runner::bind_storage`. A checkpoint with no FP8 sibling loads dense as before. Verified
//! against synthetic FP8 fixtures only; loading a real multi-hundred-GB checkpoint end to end is out of scope
//! (see `specs/277-deepseek-v3.2-dsa/spec.md`).
//!
//! **Reused, not re-derived**: [`crate::deepseek2_yarn_params`] (`pub(crate)` for this cross-file reuse) and
//! `checkpoint::place`'s [`WeightPlacement`] (the HF-[out,in] -> tracer-[in,out] projection placement every
//! loader in this crate shares). Every transform follows `runner.rs`'s `is_deepseek3` weight-building block, plus
//! the four indexer tensors. `poot_models::deepseek32` already has full top-level (embed -> layers -> final norm
//! -> lm_head) entry points, `trace_deepseek32_dsa_prefill`/`trace_deepseek32_dsa_decode`, and the loader binds
//! directly to their constant names.
//!
//! **`Runner` wiring.** `Runner::load_impl` detects `hf.is_deepseek32()` right after the standard
//! `Qwen2HfConfig` parse (a real `deepseek_v32` config.json parses through it, unlike BLOOM/MPT) and
//! dispatches to an early-return `Runner::load_deepseek32_impl`, which calls [`deepseek32_config_from_hf`]/
//! [`build_deepseek32_weights`] (see the `deepseek32` field's doc on `Runner`, `crates/poot-llm/src/runner.rs`).
//! `Runner::stateless_prefill_graph` (CPU re-prefill) and `Runner::generate_kv_masked` (fixed-KV masked
//! decode) have `deepseek32` arms calling `trace_deepseek32_dsa_prefill`/`_decode`; `Runner::bind`/
//! `bind_decode` resolve the graph's named constants against `self.weights` (including
//! `index.rope.cos`/`index.rope.sin`), and
//! carry `Storage::State` generically over `g.state`'s length, which covers DSA's extra indexer key cache.
//! `Runner::decode_arch` classifies a DSA `Runner` as `DecodeArch::DeepseekV32`, and the three
//! `generate_*reprefill` GPU paths bail explicitly. Tokenizer and
//! chat-template loading are unchanged (`load_deepseek32_impl` reuses `Tokenizer::from_file`/
//! `read_tokenizer_config_chat`; DSA changes attention only). Out of scope: GPU dispatch, the batched/pooled
//! decode engines, and real-checkpoint loading (see the spec's Out-of-scope section).

use poot_eval::{Value, materialize_dense};
use poot_graph_plan::WeightFormats;
use poot_load::{QuantKind, QuantScheme, Qwen2HfConfig, safetensors};
use poot_models::deepseek2::{DeepseekV2Config, deepseek2_rope_tables_interleaved};
use poot_models::deepseek3::DeepseekV3MoeParams;
use poot_models::deepseek32::{DsaConfig, dsa_indexer_rope_tables};
use poot_tensor::HostTensor;
use std::collections::HashMap;

use crate::checkpoint::place::{StoredRows, WeightPlacement};
use crate::core::runner::deepseek2_yarn_params;
use crate::error::{OptionExt, Result, ResultExt};
use poot_quant::weights::WeightStore;

/// Build `poot_models::deepseek32`'s three config structs from an already-parsed real `Qwen2HfConfig` (mirrors
/// `runner.rs`'s `is_deepseek3()` block in `Runner::load_impl` for the shared MLA/MoE fields). Requires
/// `hf.is_deepseek32()`'s three DSA fields and `q_lora_rank` to be `Some` (real DeepSeek-V3.2 always sets both:
/// DSA's indexer `wq_b` consumes MLA's `qr`, which only exists on the low-rank-query branch, see
/// `poot_models::deepseek32`'s module doc).
pub(crate) fn deepseek32_config_from_hf(
    hf: &Qwen2HfConfig,
) -> Result<(DeepseekV2Config, DeepseekV3MoeParams, DsaConfig)> {
    let q_lora_rank = hf.q_lora_rank.context(
        "deepseek_v32 config missing q_lora_rank (DSA's own indexer wq_b requires it - \
         real checkpoints always set it)",
    )?;
    let cfg = DeepseekV2Config {
        vocab: hf.vocab_size,
        hidden: hf.hidden_size,
        layers: hf.num_hidden_layers,
        n_heads: hf.num_attention_heads,
        q_lora_rank: Some(q_lora_rank),
        kv_lora_rank: hf
            .kv_lora_rank
            .context("deepseek_v32 config missing kv_lora_rank")?,
        qk_nope_head_dim: hf
            .qk_nope_head_dim
            .context("deepseek_v32 config missing qk_nope_head_dim")?,
        qk_rope_head_dim: hf
            .qk_rope_head_dim
            .context("deepseek_v32 config missing qk_rope_head_dim")?,
        v_head_dim: hf
            .v_head_dim
            .context("deepseek_v32 config missing v_head_dim")?,
        eps: hf.rms_norm_eps,
        max_pos: hf.max_position_embeddings,
        rope_theta: hf.effective_rope_theta()?,
        yarn: deepseek2_yarn_params(hf),
    };
    let n_group = hf.n_group.unwrap_or(1);
    let topk_group = hf.topk_group.unwrap_or(n_group);
    let mp = DeepseekV3MoeParams {
        n_routed_experts: hf
            .n_routed_experts
            .context("deepseek_v32 config missing n_routed_experts")?,
        top_k: hf
            .num_experts_per_tok
            .context("deepseek_v32 config missing num_experts_per_tok")?,
        moe_inter: hf
            .moe_intermediate_size
            .context("deepseek_v32 config missing moe_intermediate_size")?,
        n_shared_experts: hf.n_shared_experts.unwrap_or(0),
        dense_inter: hf.intermediate_size,
        first_k_dense_replace: hf.first_k_dense_replace.unwrap_or(1),
        n_group,
        topk_group,
        routed_scaling_factor: hf.routed_scaling_factor.unwrap_or(1.0),
    };
    let dcfg = DsaConfig {
        index_n_heads: hf
            .index_n_heads
            .context("deepseek_v32 config missing index_n_heads")?,
        index_head_dim: hf
            .index_head_dim
            .context("deepseek_v32 config missing index_head_dim")?,
        index_topk: hf
            .index_topk
            .context("deepseek_v32 config missing index_topk")?,
    };
    Ok((cfg, mp, dcfg))
}

/// Load and place every DeepSeek-V3.2 weight from `store` (real on-disk names, see the module doc) under
/// `poot_models::deepseek32::trace_deepseek32_dsa_prefill`/`trace_deepseek32_dsa_decode`'s constant names, with
/// the packed-storage record `Runner::bind_storage` puts onto every traced graph. Unlike `nemotron_h_load.rs`
/// these are the real HF names unchanged (both tracers declare `b.constant("self_attn.kv_a_proj_with_mqa.weight",
/// ...)` etc. directly), so this loader only places stored weights (and concatenates each routed expert's
/// `gate||up`); it does not rename or re-split anything as the Mamba fused `in_proj` did for Nemotron-H.
///
/// Every FP8 linear is packed straight from its checkpoint bytes ([`safetensors::pack_quantized_linears`]) and
/// placed packed; `fp8_block` (the config's `weight_block_size`) must be `(128, 128)`, the one block the packed
/// `E4m3Block128` format describes. The DSA config is not needed: loading needs no indexer shape numbers (an
/// on-disk tensor is placed whole).
pub(crate) fn build_deepseek32_weights(
    store: &WeightStore,
    cfg: &DeepseekV2Config,
    mp: &DeepseekV3MoeParams,
    fp8_block: (usize, usize),
) -> Result<(HashMap<String, Value>, WeightFormats)> {
    if fp8_block != (128, 128) {
        bail!(
            "deepseek_v32: weight_block_size {fp8_block:?} is not [128, 128], the one FP8 block size the \
             packed E4m3Block128 format describes"
        );
    }
    cfg.q_lora_rank.context(
        "deepseek32_load: DSA requires q_lora_rank (real checkpoints always set it - see \
         deepseek32_config_from_hf)",
    )?;
    let store = safetensors::pack_quantized_linears(
        store,
        QuantScheme {
            kind: QuantKind::Fp8,
            group_size: 128,
        },
    )
    .context("deepseek_v32: pack FP8 linears")?;
    let read = |name: &str| -> Result<StoredRows> {
        StoredRows::read(&store, name).with_context(|| format!("deepseek_v32: {name}"))
    };
    let dense = |name: &str| -> Result<HostTensor> {
        materialize_dense(&store, name).with_context(|| format!("deepseek_v32: missing {name}"))
    };
    let mut w = WeightPlacement::default();

    w.table(
        "model.embed_tokens.weight",
        read("model.embed_tokens.weight")?,
    )?;
    w.dense("model.norm.weight", dense("model.norm.weight")?);
    w.projection("lm_head.weight", read("lm_head.weight")?)?;

    // Rope tables: computed host-side, not loaded from the checkpoint, as in `runner.rs`'s `is_deepseek2()`/
    // `is_deepseek3()` loading block (see its `rope.cos`/`rope.sin` override). The indexer's table reuses the same
    // `cfg.yarn`/`rope_theta` and width `cfg.qk_rope_head_dim` (not `dcfg.index_head_dim`; see
    // `dsa_indexer_rope_tables`'s doc for why).
    let (cos, sin) = deepseek2_rope_tables_interleaved(
        cfg.max_pos,
        cfg.qk_rope_head_dim,
        cfg.rope_theta,
        cfg.yarn.as_ref(),
    );
    w.dense(
        "rope.cos",
        HostTensor::f32(vec![cfg.max_pos, cfg.qk_rope_head_dim / 2], cos),
    );
    w.dense(
        "rope.sin",
        HostTensor::f32(vec![cfg.max_pos, cfg.qk_rope_head_dim / 2], sin),
    );
    let (idx_cos, idx_sin) = dsa_indexer_rope_tables(
        cfg.max_pos,
        cfg.qk_rope_head_dim,
        cfg.rope_theta,
        cfg.yarn.as_ref(),
    );
    w.dense(
        "index.rope.cos",
        HostTensor::f32(vec![cfg.max_pos, cfg.qk_rope_head_dim], idx_cos),
    );
    w.dense(
        "index.rope.sin",
        HostTensor::f32(vec![cfg.max_pos, cfg.qk_rope_head_dim], idx_sin),
    );

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        // Norm weights, biases and the router's correction bias: 1D, placed as stored.
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.kv_a_layernorm.weight",
            // DSA indexer k_norm: a LayerNorm (weight+bias).
            "self_attn.indexer.k_norm.weight",
            "self_attn.indexer.k_norm.bias",
        ] {
            w.dense(p(suffix), dense(&p(suffix))?);
        }
        // MLA attention (names/shapes identical to plain DeepSeek-V3, see `runner.rs`'s `is_deepseek3()` block) and
        // the DSA indexer (real names, see the module doc). `wq_b`'s input is `qr` (MLA's low-rank query), a
        // tracer-side fact; the weight is an `[out,in]` HF linear like the rest, real shape
        // `[index_n_heads*index_head_dim, q_lora_rank]`. `weights_proj.weight` is the one indexer linear shipped in
        // BF16 (no FP8 scale sibling), so it stays dense. `kv_a_proj_with_mqa.weight` is the one name that does not
        // end in "proj.weight"; listing every projection here means no suffix match can miss it (card 135d).
        for suffix in [
            "self_attn.q_a_proj.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.indexer.wq_b.weight",
            "self_attn.indexer.wk.weight",
            "self_attn.indexer.weights_proj.weight",
        ] {
            w.projection(p(suffix), read(&p(suffix))?)?;
        }

        if mp.is_moe_layer(li) {
            w.projection(p("mlp.gate.weight"), read(&p("mlp.gate.weight"))?)?;
            w.dense(
                p("mlp.gate.e_score_correction_bias"),
                dense(&p("mlp.gate.e_score_correction_bias"))?,
            );
            // Routed experts: one `gate||up` owner per expert (`[2I, H]` stored rows, the tracer's `[E, H, 2I]`
            // stack with gate's columns first) and one `down` owner per expert, each stored owner shared as read.
            let mut gate_up = Vec::with_capacity(mp.n_routed_experts);
            let mut down = Vec::with_capacity(mp.n_routed_experts);
            for e in 0..mp.n_routed_experts {
                let expert = |s: &str| p(&format!("mlp.experts.{e}.{s}"));
                let gate = read(&expert("gate_proj.weight"))?;
                let up = read(&expert("up_proj.weight"))?;
                gate_up.push(
                    StoredRows::concat(&[&gate, &up])
                        .with_context(|| format!("deepseek_v32: {}", expert("gate||up")))?,
                );
                down.push(read(&expert("down_proj.weight"))?);
            }
            w.experts(p("mlp.experts.gate_up_proj.weight"), gate_up)?;
            w.experts(p("mlp.experts.down_proj.weight"), down)?;
            if mp.n_shared_experts > 0 {
                for suffix in [
                    "mlp.shared_experts.gate_proj.weight",
                    "mlp.shared_experts.up_proj.weight",
                    "mlp.shared_experts.down_proj.weight",
                ] {
                    w.projection(p(suffix), read(&p(suffix))?)?;
                }
            }
        } else {
            for suffix in [
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                w.projection(p(suffix), read(&p(suffix))?)?;
            }
        }
    }
    Ok(w.finish())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::core::cpu_oracle::{cpu_eval, cpu_eval_with_state};
    use crate::core::graphs::token_slot;
    use crate::core::runner::Runner;

    use std::collections::HashMap;
    use std::sync::Arc;

    use poot_eval::Value;
    use poot_graph_ir::packed_source::PackedSourceName;
    use poot_graph_ir::{Graph, Slot, Storage, ValueId};
    use poot_graph_plan::{PackedLayout, bind_packed_weights};
    use poot_models::deepseek32::{trace_deepseek32_dsa_decode, trace_deepseek32_dsa_prefill};
    use poot_quant::format::{ScaleEncoding, WeightFormat};
    use poot_quant::{OperandRole, PackedPayload, PackedWeightError, SourceRole};
    use poot_test_util::seed_of;

    type Loaded = (HashMap<String, Value>, WeightFormats);

    fn loader_const(loaded: &Loaded, name: &str) -> Value {
        loaded
            .0
            .get(name)
            .unwrap_or_else(|| panic!("missing loaded weight {name}"))
            .clone()
    }

    /// `graph` with the loaded packed storage placed on it, as `Runner::bind_storage` does.
    fn bound(graph: &Graph, loaded: &Loaded) -> Graph {
        bind_packed_weights(graph, &loaded.1).expect("bind packed storage")
    }

    /// The whole-model prefill logits of `loaded` for tokens `0..seq_len` on the CPU oracle (the Runner's
    /// packed-claiming entry point), a `[1, 1, vocab]` tensor.
    fn prefill_logits(
        loaded: &Loaded,
        (cfg, mp, dcfg): &(DeepseekV2Config, DeepseekV3MoeParams, DsaConfig),
        seq_len: usize,
    ) -> HostTensor {
        let graph = bound(
            &trace_deepseek32_dsa_prefill(*cfg, *dcfg, *mp, seq_len),
            loaded,
        );
        let mut cmask = vec![0.0f32; seq_len * seq_len];
        for t in 0..seq_len {
            for j in 0..seq_len {
                if j > t {
                    cmask[t * seq_len + j] = -1.0e9;
                }
            }
        }
        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        for &id in &graph.inputs {
            let meta = graph.meta(id);
            let v = match meta.storage {
                Storage::Slot(Slot::Token) => Value::from(token_slot(
                    &meta.aval,
                    &(0..seq_len as u32).collect::<Vec<_>>(),
                )),
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().expect("mask slot without a name");
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    Value::from(HostTensor::f32(vec![1, 1, seq_len, seq_len], cmask.clone()))
                }
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    loader_const(loaded, name)
                }
                other => panic!("unexpected storage {other:?} in deepseek32 dsa prefill graph"),
            };
            inputs.insert(id, v);
        }
        let logits = cpu_eval(&graph, &inputs).expect("eval deepseek32 dsa prefill graph");
        assert_eq!(logits.shape(), vec![1, 1, cfg.vocab]);
        logits
    }

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.3
            })
            .collect()
    }

    fn ln_gamma(n: usize, seed: u64) -> Vec<f32> {
        fill(n, seed).iter().map(|v| 1.0 + v * 0.2).collect()
    }

    /// One safetensors tensor: name, dtype, shape and stored bytes.
    type RawTensor = (String, &'static str, Vec<usize>, Vec<u8>);

    fn f32_le(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn write_safetensors_bytes(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let raw: Vec<RawTensor> = tensors
            .iter()
            .map(|(name, shape, values)| (name.clone(), "F32", shape.clone(), f32_le(values)))
            .collect();
        write_raw_safetensors_bytes(&raw)
    }

    fn write_raw_safetensors_bytes(tensors: &[RawTensor]) -> Vec<u8> {
        let mut data = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            let end = data.len();
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::with_capacity(8 + header_bytes.len() + data.len());
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        out
    }

    /// A real-shaped `Qwen2HfConfig`, deserialized from a JSON literal using the real field names/values of
    /// `huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json` (module doc), with only
    /// `num_hidden_layers`/`hidden_size`/etc scaled down for a fast test. The routed dimensions use `E=6`,
    /// `n_group=2`, and `group_size=3` so no requested iota aliases another.
    fn real_shaped_config_json() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DeepseekV32ForCausalLM"],
            "model_type": "deepseek_v32",
            "hidden_size": 8,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 10,
            "max_position_embeddings": 16,
            "intermediate_size": 6,
            "rms_norm_eps": 1e-6,
            "q_lora_rank": 5,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 3,
            "qk_rope_head_dim": 2,
            "v_head_dim": 3,
            "index_n_heads": 2,
            "index_head_dim": 4,
            "index_topk": 3,
            "n_routed_experts": 6,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 3,
            "n_group": 2,
            "topk_group": 1,
            "first_k_dense_replace": 1,
            "routed_scaling_factor": 1.0,
            "rope_theta": 10000.0,
            "eos_token_id": 1,
            "rope_scaling": {
                "type": "yarn",
                "factor": 40,
                "original_max_position_embeddings": 4,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0
            }
        })
    }

    #[test]
    fn config_from_hf_parses_real_field_names_and_dsa_fields() {
        let raw = real_shaped_config_json();
        let hf: Qwen2HfConfig = serde_json::from_value(raw).expect("parse Qwen2HfConfig");
        assert!(hf.is_deepseek32());
        assert!(
            !hf.is_deepseek3(),
            "deepseek_v32 must not also match is_deepseek3"
        );

        let (cfg, mp, dcfg) = deepseek32_config_from_hf(&hf).expect("deepseek32_config_from_hf");
        assert_eq!(cfg.vocab, 10);
        assert_eq!(cfg.hidden, 8);
        assert_eq!(cfg.layers, 3);
        assert_eq!(cfg.n_heads, 2);
        assert_eq!(cfg.q_lora_rank, Some(5));
        assert_eq!(cfg.kv_lora_rank, 4);
        assert_eq!(cfg.qk_nope_head_dim, 3);
        assert_eq!(cfg.qk_rope_head_dim, 2);
        assert_eq!(cfg.v_head_dim, 3);
        assert!(
            cfg.yarn.is_some(),
            "real config's YaRN block must be threaded through"
        );

        assert_eq!(mp.n_routed_experts, 6);
        assert_eq!(mp.top_k, 1);
        assert_eq!(mp.moe_inter, 3);
        assert_eq!(mp.n_shared_experts, 1);
        assert_eq!(mp.dense_inter, 6);
        assert_eq!(mp.first_k_dense_replace, 1);
        assert_eq!(mp.n_group, 2);
        assert_eq!(mp.topk_group, 1);

        assert_eq!(dcfg.index_n_heads, 2);
        assert_eq!(dcfg.index_head_dim, 4);
        assert_eq!(dcfg.index_topk, 3);
    }

    #[test]
    fn config_from_hf_rejects_missing_q_lora_rank() {
        let mut raw = real_shaped_config_json();
        raw.as_object_mut().unwrap().remove("q_lora_rank");
        let hf: Qwen2HfConfig = serde_json::from_value(raw).expect("parse Qwen2HfConfig");
        assert!(deepseek32_config_from_hf(&hf).is_err());
    }

    #[test]
    fn config_from_hf_rejects_missing_dsa_fields() {
        let mut raw = real_shaped_config_json();
        raw.as_object_mut().unwrap().remove("index_topk");
        let hf: Qwen2HfConfig = serde_json::from_value(raw).expect("parse Qwen2HfConfig");
        assert!(deepseek32_config_from_hf(&hf).is_err());
    }

    /// Build a real-named, real-layout synthetic safetensors buffer for `cfg`/`mp`'s full stack (layer 0 dense,
    /// layers 1-2 MoE per `first_k_dense_replace: 1`, so both the dense MLP and the routed+shared-expert MoE paths
    /// are exercised, plus the DSA indexer tensors on every layer, in one checkpoint).
    fn write_synthetic_deepseek32_checkpoint_bytes(
        cfg: &DeepseekV2Config,
        mp: &DeepseekV3MoeParams,
        dcfg: &DsaConfig,
    ) -> Vec<u8> {
        let h = cfg.hidden;
        let hq = cfg.n_heads;
        let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
        let qk_head_dim = nope + rope_d;
        let kv_rank = cfg.kv_lora_rank;
        let q_lora_rank = cfg.q_lora_rank.unwrap();
        let (hi, di) = (dcfg.index_n_heads, dcfg.index_head_dim);
        let e = mp.n_routed_experts;

        let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        t.push((
            "model.embed_tokens.weight".into(),
            vec![cfg.vocab, h],
            fill(cfg.vocab * h, seed_of("embed")),
        ));
        t.push((
            "model.norm.weight".into(),
            vec![h],
            ln_gamma(h, seed_of("norm_f")),
        ));
        t.push((
            "lm_head.weight".into(),
            vec![cfg.vocab, h],
            fill(cfg.vocab * h, seed_of("lm_head")),
        ));

        for li in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let sf = |s: &str| format!("l{li}.{s}");
            t.push((
                p("input_layernorm.weight"),
                vec![h],
                ln_gamma(h, seed_of(&sf("ln1"))),
            ));
            t.push((
                p("post_attention_layernorm.weight"),
                vec![h],
                ln_gamma(h, seed_of(&sf("ln2"))),
            ));

            // real HF [out,in] layout throughout.
            t.push((
                p("self_attn.q_a_proj.weight"),
                vec![q_lora_rank, h],
                fill(q_lora_rank * h, seed_of(&sf("qa"))),
            ));
            t.push((
                p("self_attn.q_a_layernorm.weight"),
                vec![q_lora_rank],
                ln_gamma(q_lora_rank, seed_of(&sf("qaln"))),
            ));
            t.push((
                p("self_attn.q_b_proj.weight"),
                vec![hq * qk_head_dim, q_lora_rank],
                fill(hq * qk_head_dim * q_lora_rank, seed_of(&sf("qb"))),
            ));
            t.push((
                p("self_attn.kv_a_proj_with_mqa.weight"),
                vec![kv_rank + rope_d, h],
                fill((kv_rank + rope_d) * h, seed_of(&sf("kva"))),
            ));
            t.push((
                p("self_attn.kv_a_layernorm.weight"),
                vec![kv_rank],
                ln_gamma(kv_rank, seed_of(&sf("kvaln"))),
            ));
            t.push((
                p("self_attn.kv_b_proj.weight"),
                vec![hq * (nope + vd), kv_rank],
                fill(hq * (nope + vd) * kv_rank, seed_of(&sf("kvb"))),
            ));
            t.push((
                p("self_attn.o_proj.weight"),
                vec![h, hq * vd],
                fill(h * hq * vd, seed_of(&sf("o"))),
            ));

            t.push((
                p("self_attn.indexer.wq_b.weight"),
                vec![hi * di, q_lora_rank],
                fill(hi * di * q_lora_rank, seed_of(&sf("iwqb"))),
            ));
            t.push((
                p("self_attn.indexer.wk.weight"),
                vec![di, h],
                fill(di * h, seed_of(&sf("iwk"))),
            ));
            t.push((
                p("self_attn.indexer.weights_proj.weight"),
                vec![hi, h],
                fill(hi * h, seed_of(&sf("iww"))),
            ));
            t.push((
                p("self_attn.indexer.k_norm.weight"),
                vec![di],
                ln_gamma(di, seed_of(&sf("iknw"))),
            ));
            t.push((
                p("self_attn.indexer.k_norm.bias"),
                vec![di],
                fill(di, seed_of(&sf("iknb"))),
            ));

            if mp.is_moe_layer(li) {
                t.push((
                    p("mlp.gate.weight"),
                    vec![e, h],
                    fill(e * h, seed_of(&sf("gate"))),
                ));
                t.push((
                    p("mlp.gate.e_score_correction_bias"),
                    vec![e],
                    fill(e, seed_of(&sf("gate_bias"))),
                ));
                for ei in 0..e {
                    let sfe = |s: &str| format!("l{li}.e{ei}.{s}");
                    t.push((
                        p(&format!("mlp.experts.{ei}.gate_proj.weight")),
                        vec![mp.moe_inter, h],
                        fill(mp.moe_inter * h, seed_of(&sfe("g"))),
                    ));
                    t.push((
                        p(&format!("mlp.experts.{ei}.up_proj.weight")),
                        vec![mp.moe_inter, h],
                        fill(mp.moe_inter * h, seed_of(&sfe("u"))),
                    ));
                    t.push((
                        p(&format!("mlp.experts.{ei}.down_proj.weight")),
                        vec![h, mp.moe_inter],
                        fill(h * mp.moe_inter, seed_of(&sfe("d"))),
                    ));
                }
                if mp.n_shared_experts > 0 {
                    let shared_w = mp.moe_inter * mp.n_shared_experts;
                    t.push((
                        p("mlp.shared_experts.gate_proj.weight"),
                        vec![shared_w, h],
                        fill(shared_w * h, seed_of(&sf("sg"))),
                    ));
                    t.push((
                        p("mlp.shared_experts.up_proj.weight"),
                        vec![shared_w, h],
                        fill(shared_w * h, seed_of(&sf("su"))),
                    ));
                    t.push((
                        p("mlp.shared_experts.down_proj.weight"),
                        vec![h, shared_w],
                        fill(h * shared_w, seed_of(&sf("sd"))),
                    ));
                }
            } else {
                t.push((
                    p("mlp.gate_proj.weight"),
                    vec![mp.dense_inter, h],
                    fill(mp.dense_inter * h, seed_of(&sf("dg"))),
                ));
                t.push((
                    p("mlp.up_proj.weight"),
                    vec![mp.dense_inter, h],
                    fill(mp.dense_inter * h, seed_of(&sf("du"))),
                ));
                t.push((
                    p("mlp.down_proj.weight"),
                    vec![h, mp.dense_inter],
                    fill(h * mp.dense_inter, seed_of(&sf("dd"))),
                ));
            }
        }
        write_safetensors_bytes(&t)
    }

    fn tiny_stack() -> (DeepseekV2Config, DeepseekV3MoeParams, DsaConfig) {
        let raw = real_shaped_config_json();
        let hf: Qwen2HfConfig = serde_json::from_value(raw).expect("parse Qwen2HfConfig");
        deepseek32_config_from_hf(&hf).expect("deepseek32_config_from_hf")
    }

    #[test]
    fn synthetic_checkpoint_loads_every_expected_name() {
        let (cfg, mp, dcfg) = tiny_stack();
        let bytes = write_synthetic_deepseek32_checkpoint_bytes(&cfg, &mp, &dcfg);
        let st = poot_load::safetensors::load_weight_store_bytes(&bytes)
            .expect("parse synthetic deepseek32 checkpoint");
        let loaded =
            build_deepseek32_weights(&st, &cfg, &mp, (128, 128)).expect("build_deepseek32_weights");

        for name in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "rope.cos",
            "rope.sin",
            "index.rope.cos",
            "index.rope.sin",
        ] {
            assert!(loaded.0.contains_key(name), "loader did not produce {name}");
        }
        for li in 0..cfg.layers {
            for suf in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_a_proj.weight",
                "self_attn.q_a_layernorm.weight",
                "self_attn.q_b_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight",
                "self_attn.kv_a_layernorm.weight",
                "self_attn.kv_b_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.indexer.wq_b.weight",
                "self_attn.indexer.wk.weight",
                "self_attn.indexer.weights_proj.weight",
                "self_attn.indexer.k_norm.weight",
                "self_attn.indexer.k_norm.bias",
            ] {
                let name = format!("model.layers.{li}.{suf}");
                assert!(
                    loaded.0.contains_key(&name),
                    "loader did not produce {name}"
                );
            }
            if mp.is_moe_layer(li) {
                for suf in [
                    "mlp.gate.weight",
                    "mlp.gate.e_score_correction_bias",
                    "mlp.experts.gate_up_proj.weight",
                    "mlp.experts.down_proj.weight",
                    "mlp.shared_experts.gate_proj.weight",
                    "mlp.shared_experts.up_proj.weight",
                    "mlp.shared_experts.down_proj.weight",
                ] {
                    let name = format!("model.layers.{li}.{suf}");
                    assert!(
                        loaded.0.contains_key(&name),
                        "loader did not produce {name}"
                    );
                }
            } else {
                for suf in [
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ] {
                    let name = format!("model.layers.{li}.{suf}");
                    assert!(
                        loaded.0.contains_key(&name),
                        "loader did not produce {name}"
                    );
                }
            }
        }
    }

    /// Loader-to-tracer wiring proof: load a real-named/real-shaped synthetic checkpoint, trace the whole model via
    /// `trace_deepseek32_dsa_prefill` (a full embed -> layers -> norm -> lm_head entry point), and evaluate through
    /// `poot-eval`'s CPU oracle. A coherence check (finite, non-degenerate, correctly-shaped logits), not a numeric
    /// match against a reference: the per-layer DSA/MLA/MoE math is verified by `poot_models::deepseek32`'s
    /// `cpu_oracle` tests. This proves loading and tracing connect without a wiring bug (a wrong transpose axis, a
    /// missing constant, a shape mismatch).
    #[test]
    fn synthetic_checkpoint_prefill_end_to_end_via_trace_deepseek32_dsa_prefill() {
        let stack = tiny_stack();
        let (cfg, mp, dcfg) = &stack;
        let bytes = write_synthetic_deepseek32_checkpoint_bytes(cfg, mp, dcfg);
        let st = poot_load::safetensors::load_weight_store_bytes(&bytes)
            .expect("parse synthetic deepseek32 checkpoint");
        let loaded =
            build_deepseek32_weights(&st, cfg, mp, (128, 128)).expect("build_deepseek32_weights");
        assert!(
            loaded.1.is_empty(),
            "a checkpoint with no FP8 sibling loads dense"
        );

        let logits = prefill_logits(&loaded, &stack, 4);
        assert!(
            logits.as_f32().unwrap().iter().all(|v| v.is_finite()),
            "logits must be finite: {:?}",
            logits.as_f32().unwrap()
        );
        assert!(
            logits
                .as_f32()
                .unwrap()
                .iter()
                .any(|&v| v != logits.as_f32().unwrap()[0]),
            "logits must not be degenerate (all-equal)"
        );
    }

    /// The decode counterpart of the prefill test above, driven step by step with cache carry-over via
    /// `poot_eval::eval_with_state` (decode is the harder case, see `poot_models::deepseek32`'s module doc). Same
    /// coherence bar (finite, non-degenerate, correctly-shaped logits at every step); the per-step DSA/MLA math is
    /// verified elsewhere (see the prefill test's doc).
    #[test]
    fn synthetic_checkpoint_decode_end_to_end_via_trace_deepseek32_dsa_decode() {
        let (cfg, mp, dcfg) = tiny_stack();
        let bytes = write_synthetic_deepseek32_checkpoint_bytes(&cfg, &mp, &dcfg);
        let st = poot_load::safetensors::load_weight_store_bytes(&bytes)
            .expect("parse synthetic deepseek32 checkpoint");
        let loaded =
            build_deepseek32_weights(&st, &cfg, &mp, (128, 128)).expect("build_deepseek32_weights");

        let cap = 5usize;
        let tokens = [3u32, 7, 1, 9, 5];
        let graph = bound(&trace_deepseek32_dsa_decode(cfg, dcfg, mp, cap), &loaded);

        let mut caches: Vec<HostTensor> = graph
            .state
            .iter()
            .map(|&(si, _)| HostTensor::zeros(graph.aval(si).shape.clone()))
            .collect();

        for (pos, &tok) in tokens.iter().enumerate() {
            let mut inputs: HashMap<ValueId, Value> = HashMap::new();
            for &id in &graph.inputs {
                let meta = graph.meta(id);
                let v = match meta.storage {
                    Storage::Slot(Slot::Token) => Value::from(token_slot(&meta.aval, &[tok])),
                    Storage::Slot(Slot::Pos) => {
                        Value::from(HostTensor::i32(vec![], vec![pos as i32]))
                    }
                    Storage::Slot(Slot::SeqLen) => {
                        Value::from(HostTensor::i32(vec![], vec![(pos + 1) as i32]))
                    }
                    Storage::Slot(Slot::Mask) => {
                        let cap_n = meta.aval.shape.iter().product::<usize>();
                        let m: Vec<f32> = (0..cap_n)
                            .map(|k| if k <= pos { 0.0 } else { -1.0e9 })
                            .collect();
                        Value::from(HostTensor::f32(meta.aval.shape.clone(), m))
                    }
                    Storage::State => continue,
                    Storage::Const => {
                        let name = meta.name.as_deref().expect("const without a name");
                        loader_const(&loaded, name)
                    }
                    other => panic!("unexpected storage {other:?} in deepseek32 dsa decode graph"),
                };
                inputs.insert(id, v);
            }
            for (ci, &(si, _)) in graph.state.iter().enumerate() {
                inputs.insert(si, Value::from(caches[ci].clone()));
            }

            let (logits, new_caches) =
                cpu_eval_with_state(&graph, &inputs).expect("eval deepseek32 dsa decode graph");
            caches = new_caches;

            assert_eq!(logits.shape(), vec![1, 1, cfg.vocab]);
            assert!(
                logits.as_f32().unwrap().iter().all(|v| v.is_finite()),
                "step {pos}: logits must be finite: {:?}",
                logits.as_f32().unwrap()
            );
            assert!(
                logits
                    .as_f32()
                    .unwrap()
                    .iter()
                    .any(|&v| v != logits.as_f32().unwrap()[0]),
                "step {pos}: logits must not be degenerate (all-equal)"
            );
        }
    }

    /// Write `bytes` (from [`write_synthetic_deepseek32_checkpoint_bytes`]) to `path`: the on-disk counterpart of
    /// the two tests above, which parse the same bytes in memory via `SafeTensors::load_bytes`.
    /// `Runner::load` need a real `model.safetensors` file (they read
    /// `config.json`/`model.safetensors`/`tokenizer.json` off `dir`), unlike this module's other tests, which call
    /// `build_deepseek32_weights` on an in-memory `SafeTensors`.
    fn write_safetensors_file(path: &std::path::Path, bytes: &[u8]) {
        std::fs::write(path, bytes).expect("write model.safetensors fixture");
    }

    /// Build a tiny real-shaped DeepSeek-V3.2 DSA checkpoint directory (config.json + model.safetensors +
    /// tokenizer.json) under `std::env::temp_dir()`: the on-disk counterpart of
    /// [`tiny_stack`]/[`write_synthetic_deepseek32_checkpoint_bytes`], with the same HF field/tensor names and
    /// `real_shaped_config_json` config, so `Runner::load` can load it. Mirrors `bloom_load.rs`'s
    /// `write_tiny_bloom_checkpoint`. Returns the directory path.
    pub(crate) fn write_tiny_deepseek32_checkpoint() -> poot_test_util::UniqueTempPath {
        let (cfg, mp, dcfg) = tiny_stack();
        let bytes = write_synthetic_deepseek32_checkpoint_bytes(&cfg, &mp, &dcfg);
        write_deepseek32_checkpoint_dir(&real_shaped_config_json(), &bytes)
    }

    /// A checkpoint directory holding `config` (a [`real_shaped_config_json`] variant), the safetensors `bytes`
    /// and a WordLevel tokenizer over the config's vocabulary.
    fn write_deepseek32_checkpoint_dir(
        config: &serde_json::Value,
        bytes: &[u8],
    ) -> poot_test_util::UniqueTempPath {
        let vocab = config["vocab_size"].as_u64().expect("vocab_size") as usize;
        let dir = poot_test_util::unique_temp_path("poot_deepseek32_loader_fixture");
        std::fs::create_dir_all(&dir).expect("create fixture dir");

        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(config).unwrap(),
        )
        .expect("write config.json");

        write_safetensors_file(&dir.join("model.safetensors"), bytes);

        // A minimal tokenizers::Tokenizer (WordLevel, "t0".."t{vocab-1}"), as in `bloom_load.rs`'s fixture, saved
        // to disk so `Runner::load`'s `Tokenizer::from_file` path is exercised.
        let vocab_map: std::collections::HashMap<String, u32> =
            (0..vocab).map(|i| (format!("t{i}"), i as u32)).collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab_map)
            .unk_token("t0".to_string())
            .build()
            .expect("build wordlevel model");
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        tok.save(dir.join("tokenizer.json"), false)
            .expect("save tokenizer.json");

        dir
    }

    /// End to end through the full production `Runner::load` path (unlike the tests above, which call
    /// `deepseek32_config_from_hf`/`build_deepseek32_weights` directly): `Runner::load_impl`'s
    /// `hf.is_deepseek32()` detection, `Runner::load_deepseek32_impl`'s config/weight-map construction, the
    /// `deepseek32` arms of `Runner::stateless_prefill_graph` and `Runner::generate_kv_masked`, and
    /// `Runner::bind`/`bind_decode`'s generic constant/state resolution all connect. The per-layer DSA/MLA/MoE math
    /// is verified by this module's tests above and `poot_models::deepseek32`'s `cpu_oracle` tests; this proves the
    /// wiring, the coherence bar of `bloom_load.rs`'s `runner_load_detects_bloom_and_generates_finite_output`.
    #[test]
    fn runner_load_deepseek32_and_generates_finite_output() {
        let dir = write_tiny_deepseek32_checkpoint();
        let runner = Runner::load(&dir).expect("load synthetic deepseek_v32 checkpoint");

        assert_eq!(runner.arch, "deepseek_v32");
        assert_eq!(
            runner.decode_arch().unwrap(),
            crate::core::decode_arch::DecodeArch::DeepseekV32
        );
        assert!(
            runner.deepseek2.is_none() && runner.deepseek3.is_none(),
            "a deepseek_v32 checkpoint must not also set deepseek2/deepseek3"
        );
        let dp = runner
            .deepseek32
            .expect("Runner::load must set deepseek32 params for a deepseek_v32 checkpoint");
        assert_eq!(dp.cfg.q_lora_rank, Some(5));
        assert_eq!(dp.cfg.kv_lora_rank, 4);
        assert_eq!(dp.moe.n_routed_experts, 6);
        assert_eq!(dp.dsa.index_n_heads, 2);
        assert_eq!(dp.dsa.index_head_dim, 4);
        assert_eq!(dp.dsa.index_topk, 3);

        // The production entry point (Runner::generate -> generate_sampled's deepseek32 arm -> stateless_prefill_graph, re-prefilling the growing context each step).
        let tokens = runner
            .generate("t1 t2 t3", 2, |_| std::ops::ControlFlow::Continue(()))
            .expect("generate (re-prefill) on synthetic deepseek_v32 checkpoint");
        assert!(
            tokens.len() >= 3,
            "at least the 3 prompt tokens should come back"
        );

        // The fixed-KV masked decode entry point (Runner::generate_kv_masked's deepseek32 arm ->
        // trace_deepseek32_dsa_decode, carrying MLA's two caches plus the indexer's third across steps): shows
        // `Runner::bind_decode`'s generic `Storage::State`/`Storage::Const` handling covers DSA's extra state tensor
        // without DSA-specific binder code.
        let kv_tokens = runner
            .generate_kv_masked("t1 t2 t3", 2, |_| std::ops::ControlFlow::Continue(()))
            .expect("generate_kv_masked on synthetic deepseek_v32 checkpoint");
        assert!(
            kv_tokens.len() >= 3,
            "at least the 3 prompt tokens should come back from the KV-masked decode path"
        );

        // A direct finite/non-degenerate numeric check on the production-loaded weights via the same dispatch
        // `generate`/`generate_sampled` use (`stateless_prefill_graph`), like `bloom_load.rs`'s fixture.
        let tokens3 = runner.encode("t1 t2 t3").expect("encode prompt");
        assert_eq!(tokens3.len(), 3);
        let g = runner.stateless_prefill_graph(tokens3.len()).unwrap();
        let inputs = runner
            .bind(&g, &tokens3)
            .expect("bind deepseek32 prefill graph via Runner::bind");
        let logits =
            crate::core::cpu_oracle::cpu_eval(&g, &inputs).expect("eval deepseek32 prefill graph");
        assert_eq!(logits.shape(), vec![1, 1, dp.cfg.vocab]);
        assert!(
            logits.as_f32().unwrap().iter().all(|v| v.is_finite()),
            "logits must be finite: {:?}",
            logits.as_f32().unwrap()
        );
        assert!(
            logits
                .as_f32()
                .unwrap()
                .iter()
                .any(|&v| v != logits.as_f32().unwrap()[0]),
            "logits must not be degenerate (all-equal)"
        );
    }

    /// A small representable-value lookup table (OCP E4M3FN codes with their exact values, as in `poot_load`'s
    /// `fp8_e4m3_decode` unit test), cycled to fill a synthetic FP8 weight's codes without a general f32->fp8
    /// encoder.
    fn fp8_e4m3_fixture_byte(i: usize) -> (u8, f32) {
        const TABLE: [(u8, f32); 6] = [
            (0x38, 1.0),
            (0x40, 2.0),
            (0x3C, 1.5),
            (0xB8, -1.0),
            (0x30, 0.5),
            (0x00, 0.0),
        ];
        TABLE[i % TABLE.len()]
    }

    /// The `.weight_scale_inv` of one FP8 fixture tensor: one scale per 128x128 block of its
    /// `[ceil(out/128), ceil(in/128)]` grid, row-major.
    #[derive(Clone, Copy, Debug)]
    enum BlockScale {
        /// F32 scale `base * (1 + block)`: every block its own value.
        F32(f32),
        /// One E8M0 byte (`2^(byte - 127)`, `0xff` is NaN) for every block.
        E8m0(u8),
    }

    impl BlockScale {
        fn value(self, block: usize) -> f32 {
            match self {
                Self::F32(base) => base * (1 + block) as f32,
                Self::E8m0(byte) => poot_quant::scalar::e8m0_to_f32(byte),
            }
        }

        /// The safetensors dtype and bytes of a `blocks`-block scale tensor.
        fn stored(self, blocks: usize) -> (&'static str, Vec<u8>) {
            match self {
                Self::F32(_) => (
                    "F32",
                    (0..blocks)
                        .flat_map(|b| self.value(b).to_le_bytes())
                        .collect(),
                ),
                Self::E8m0(byte) => ("F8_E8M0", vec![byte; blocks]),
            }
        }
    }

    /// The value element `[o, k]` of an `[out, k_dim]` FP8 fixture tensor decodes to: its code's value times
    /// its block's scale, computed here from the fixture table, not through `poot-quant`.
    fn fp8_fixture_value(o: usize, k: usize, k_dim: usize, scale: BlockScale) -> f32 {
        let block = (o / 128) * k_dim.div_ceil(128) + k / 128;
        fp8_e4m3_fixture_byte(o * k_dim + k).1 * scale.value(block)
    }

    /// Re-encode `f32_bytes` (a [`write_synthetic_deepseek32_checkpoint_bytes`] checkpoint) so every `targets`
    /// tensor is DeepSeek-V3.2 block FP8: `F8_E4M3` codes from the fixture table plus a `.weight_scale_inv`
    /// sibling of the given scale. Returns that FP8 checkpoint and the dense F32 checkpoint of the values it
    /// decodes to (every other tensor identical in both), the reference an FP8 load must reproduce.
    fn quantize_to_fp8(f32_bytes: &[u8], targets: &[(String, BlockScale)]) -> (Vec<u8>, Vec<u8>) {
        let store = poot_load::safetensors::load_weight_store_bytes(f32_bytes)
            .expect("parse synthetic f32 checkpoint");
        let mut fp8: Vec<RawTensor> = Vec::new();
        let mut reference: Vec<RawTensor> = Vec::new();
        for key in store.keys() {
            let name = key.as_str();
            let rt = materialize_dense(&store, name).expect("materialize synthetic tensor");
            let Some(&(_, scale)) = targets.iter().find(|(target, _)| target == name) else {
                let raw = (
                    name.to_string(),
                    "F32",
                    rt.shape().to_vec(),
                    f32_le(rt.as_f32().unwrap()),
                );
                fp8.push(raw.clone());
                reference.push(raw);
                continue;
            };
            let [out, k_dim] = rt.shape()[..] else {
                panic!("FP8 target {name} is not 2D: {:?}", rt.shape())
            };
            let codes = (0..out * k_dim)
                .map(|i| fp8_e4m3_fixture_byte(i).0)
                .collect();
            fp8.push((name.to_string(), "F8_E4M3", rt.shape().to_vec(), codes));
            let blocks = [out.div_ceil(128), k_dim.div_ceil(128)];
            let (dtype, scale_bytes) = scale.stored(blocks[0] * blocks[1]);
            let prefix = name
                .strip_suffix(".weight")
                .expect("FP8 target is a .weight");
            fp8.push((
                format!("{prefix}.weight_scale_inv"),
                dtype,
                blocks.to_vec(),
                scale_bytes,
            ));
            let values: Vec<f32> = (0..out)
                .flat_map(|o| (0..k_dim).map(move |k| fp8_fixture_value(o, k, k_dim, scale)))
                .collect();
            reference.push((
                name.to_string(),
                "F32",
                rt.shape().to_vec(),
                f32_le(&values),
            ));
        }
        (
            write_raw_safetensors_bytes(&fp8),
            write_raw_safetensors_bytes(&reference),
        )
    }

    /// Every routed expert's `gate_proj`/`up_proj`/`down_proj` on every MoE layer, each with its own F32 block
    /// scales.
    fn routed_expert_targets(
        cfg: &DeepseekV2Config,
        mp: &DeepseekV3MoeParams,
    ) -> Vec<(String, BlockScale)> {
        (0..cfg.layers)
            .filter(|&li| mp.is_moe_layer(li))
            .flat_map(|li| {
                (0..mp.n_routed_experts).flat_map(move |e| {
                    [("gate_proj", 0.03), ("up_proj", 0.05), ("down_proj", 0.02)].map(
                        |(tensor, base)| {
                            (
                                format!("model.layers.{li}.mlp.experts.{e}.{tensor}.weight"),
                                BlockScale::F32(base + 0.01 * e as f32),
                            )
                        },
                    )
                })
            })
            .collect()
    }

    /// [`real_shaped_config_json`] with a routed (and shared) expert width of `moe_inter`, as a DeepSeek-V3.2
    /// FP8 checkpoint's config (`quant_method: "fp8"`, `weight_block_size: [128, 128]`).
    fn fp8_moe_config_json(moe_inter: usize) -> serde_json::Value {
        let mut raw = real_shaped_config_json();
        raw["moe_intermediate_size"] = serde_json::json!(moe_inter);
        raw["quantization_config"] = serde_json::json!({
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        });
        raw
    }

    fn stack_of(raw: serde_json::Value) -> (DeepseekV2Config, DeepseekV3MoeParams, DsaConfig) {
        let hf: Qwen2HfConfig = serde_json::from_value(raw).expect("parse Qwen2HfConfig");
        deepseek32_config_from_hf(&hf).expect("deepseek32_config_from_hf")
    }

    /// All-dense variant of `tiny_stack` (`first_k_dense_replace` covers every layer): no MoE tensors, so the
    /// non-expert FP8 test stays on the projections (routed experts have their own tests below).
    fn tiny_all_dense_stack() -> (DeepseekV2Config, DeepseekV3MoeParams, DsaConfig) {
        let mut raw = real_shaped_config_json();
        raw["first_k_dense_replace"] = serde_json::json!(3);
        stack_of(raw)
    }

    fn load(bytes: &[u8], cfg: &DeepseekV2Config, mp: &DeepseekV3MoeParams) -> Result<Loaded> {
        let st = poot_load::safetensors::load_weight_store_bytes(bytes)
            .expect("parse synthetic deepseek32 checkpoint");
        build_deepseek32_weights(&st, cfg, mp, (128, 128))
    }

    /// The layout and the owners (one per linear id, in order) `loaded` placed for packed constant `name`.
    fn placed_owners(loaded: &Loaded, name: &str) -> (PackedLayout, Vec<Arc<PackedPayload>>) {
        let packed = loaded
            .1
            .get(name)
            .unwrap_or_else(|| panic!("{name} should be placed packed"));
        let owners = packed
            .linear_ids
            .iter()
            .map(|linear_id| {
                let carrier =
                    PackedSourceName::new(linear_id, SourceRole::Planar(OperandRole::Codes));
                let Some(Value::Packed(component)) = loaded.0.get(carrier.as_ref()) else {
                    panic!("{carrier} is not a packed carrier");
                };
                Arc::clone(component.owner())
            })
            .collect();
        (packed.layout, owners)
    }

    /// Every row of `payload`, decoded through the CPU oracle's one packed read.
    fn decoded_rows(payload: &PackedPayload) -> Vec<Vec<f32>> {
        let [out, k] = payload.weight().shape();
        (0..out)
            .map(|row| {
                let mut values = vec![0.0f32; k];
                payload.decode_row(row, &mut values).expect("decode row");
                values
            })
            .collect()
    }

    /// Assert `payload` is an `E4m3Block128` weight with `scale` encoding whose rows are `expected` bit for bit.
    fn assert_rows(
        payload: &PackedPayload,
        encoding: ScaleEncoding,
        expected: &[Vec<f32>],
        what: &str,
    ) {
        assert_eq!(
            payload.weight().format(),
            WeightFormat::E4m3Block128 { scale: encoding },
            "{what}"
        );
        let got = decoded_rows(payload);
        assert_eq!(got.len(), expected.len(), "{what} rows");
        let bits = |row: &[f32]| row.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        for (row, (got, want)) in got.iter().zip(expected).enumerate() {
            assert!(
                bits(got) == bits(want),
                "{what} row {row}: {got:?} vs reference {want:?}"
            );
        }
    }

    /// The rows fixture tensor `[out, k_dim]` decodes to under `scale`.
    fn fixture_rows(out: usize, k_dim: usize, scale: BlockScale) -> Vec<Vec<f32>> {
        (0..out)
            .map(|o| {
                (0..k_dim)
                    .map(|k| fp8_fixture_value(o, k, k_dim, scale))
                    .collect()
            })
            .collect()
    }

    /// ADR-0101: every logit within `1e-4` of the dense reference (relative above 1); NaN always fails.
    fn assert_logits_match(got: &HostTensor, want: &HostTensor, what: &str) {
        assert_eq!(got.shape(), want.shape(), "{what}");
        for (i, (&g, &w)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(want.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "{what} logit {i}: {g} vs dense reference {w}"
            );
        }
    }

    /// The `PackedWeightError` somewhere in `error`'s source chain.
    fn packed_weight_error(error: &crate::RunnerError) -> Option<&PackedWeightError> {
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(error);
        while let Some(current) = cause {
            if let Some(packed) = current.downcast_ref::<PackedWeightError>() {
                return Some(packed);
            }
            cause = current.source();
        }
        None
    }

    /// Card 654: the non-expert FP8 linears (MLA projections and `lm_head`, with an F32 and an E8M0
    /// `.weight_scale_inv`) load through `pack_quantized_linears`/`checkpoint::place` as one `E4m3Block128`
    /// owner each, placed as `[in, out]` columns, decoding to `e4m3(q) * scale`, and the whole model's prefill
    /// logits match the dense checkpoint of the same values.
    #[test]
    fn fp8_linears_load_packed_and_match_the_dense_reference() {
        let stack = tiny_all_dense_stack();
        let (cfg, mp, dcfg) = &stack;
        let f32_bytes = write_synthetic_deepseek32_checkpoint_bytes(cfg, mp, dcfg);
        let targets = [
            (
                "model.layers.0.self_attn.q_a_proj.weight".to_string(),
                BlockScale::F32(0.3),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".to_string(),
                BlockScale::E8m0(125),
            ),
            ("lm_head.weight".to_string(), BlockScale::F32(0.2)),
        ];
        let (fp8, reference) = quantize_to_fp8(&f32_bytes, &targets);
        let loaded = load(&fp8, cfg, mp).expect("load the FP8 checkpoint");
        for (name, scale) in &targets {
            let (layout, owners) = placed_owners(&loaded, name);
            assert_eq!(layout, PackedLayout::Columns, "{name}");
            let [owner] = &owners[..] else {
                panic!("{name}: {} owners", owners.len())
            };
            let [out, k_dim] = owner.weight().shape();
            let encoding = match scale {
                BlockScale::F32(_) => ScaleEncoding::F32,
                BlockScale::E8m0(_) => ScaleEncoding::E8m0,
            };
            assert_rows(owner, encoding, &fixture_rows(out, k_dim, *scale), name);
        }
        let dense = load(&reference, cfg, mp).expect("load the dense reference");
        assert!(dense.1.is_empty());
        assert_logits_match(
            &prefill_logits(&loaded, &stack, 4),
            &prefill_logits(&dense, &stack, 4),
            "FP8 projections",
        );
    }

    /// SC-001 (Card 654): routed experts whose `gate_proj`/`up_proj` have 256 rows (two 128-row
    /// scale blocks each, every block its own F32 scale) load through `pack_quantized_linears` and
    /// `checkpoint::place`: every expert's fused `gate||up` owner decodes, row for row, to the un-fused gate
    /// rows then up rows (`e4m3(q) * scale[o/128, k/128]`, each tensor on its own block grid), `down_proj` is
    /// placed per expert as stored, and the production `Runner` load of the FP8 checkpoint gives the prefill
    /// logits of the dense checkpoint with the same values. Mutation: concatenating `up`'s scale blocks
    /// ahead of `gate`'s puts them on `gate`'s block grid, and the fused rows diverge at the first block.
    #[test]
    fn fp8_routed_experts_fuse_gate_up_packed_row_for_row() {
        let config = fp8_moe_config_json(256);
        let stack = stack_of(config.clone());
        let (cfg, mp, dcfg) = &stack;
        let h = cfg.hidden;
        let inter = mp.moe_inter;
        let targets = routed_expert_targets(cfg, mp);
        let (fp8, reference) = quantize_to_fp8(
            &write_synthetic_deepseek32_checkpoint_bytes(cfg, mp, dcfg),
            &targets,
        );
        let loaded = load(&fp8, cfg, mp).expect("load the FP8 checkpoint");
        let scale_of = |name: &str| {
            targets
                .iter()
                .find(|(target, _)| target == name)
                .map(|&(_, scale)| scale)
                .unwrap_or_else(|| panic!("{name} is not an FP8 target"))
        };
        for li in (0..cfg.layers).filter(|&li| mp.is_moe_layer(li)) {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let (layout, gate_up) = placed_owners(&loaded, &p("mlp.experts.gate_up_proj.weight"));
            assert_eq!(layout, PackedLayout::StackedColumns);
            let (layout, down) = placed_owners(&loaded, &p("mlp.experts.down_proj.weight"));
            assert_eq!(layout, PackedLayout::StackedColumns);
            assert_eq!(
                (gate_up.len(), down.len()),
                (mp.n_routed_experts, mp.n_routed_experts)
            );
            for e in 0..mp.n_routed_experts {
                let expert = |t: &str| p(&format!("mlp.experts.{e}.{t}.weight"));
                let mut fused = fixture_rows(inter, h, scale_of(&expert("gate_proj")));
                fused.extend(fixture_rows(inter, h, scale_of(&expert("up_proj"))));
                assert_rows(
                    &gate_up[e],
                    ScaleEncoding::F32,
                    &fused,
                    &format!("layer {li} expert {e} gate||up"),
                );
                assert_rows(
                    &down[e],
                    ScaleEncoding::F32,
                    &fixture_rows(h, inter, scale_of(&expert("down_proj"))),
                    &format!("layer {li} expert {e} down"),
                );
            }
        }

        // The production load: `Runner::load_deepseek32_impl` packs, places and records the formats that
        // `Runner::bind_storage` puts onto the traced prefill graph; the dense twin checkpoint is the reference.
        let fp8_dir = write_deepseek32_checkpoint_dir(&config, &fp8);
        let dense_dir = write_deepseek32_checkpoint_dir(&config, &reference);
        let fp8_runner = Runner::load(&fp8_dir).expect("Runner loads the FP8 checkpoint");
        let dense_runner = Runner::load(&dense_dir).expect("Runner loads the dense reference");
        assert!(!fp8_runner.formats.is_empty() && dense_runner.formats.is_empty());
        let logits = |runner: &Runner| {
            let tokens = runner.encode("t1 t2 t3").expect("encode prompt");
            let g = runner
                .stateless_prefill_graph(tokens.len())
                .expect("prefill graph");
            let inputs = runner.bind(&g, &tokens).expect("bind prefill graph");
            cpu_eval(&g, &inputs).expect("eval prefill graph")
        };
        assert_logits_match(
            &logits(&fp8_runner),
            &logits(&dense_runner),
            "FP8 routed experts",
        );
    }

    /// SC-002 (Card 654): an FP8 routed expert whose `gate_proj`/`up_proj` have 100 rows cannot be fused
    /// without a 128x128 scale block covering both, so the load is refused with the typed
    /// `PackedWeightError::RowConcatUnaligned` naming the scale operand, before any byte is copied. Mutation:
    /// dropping `poot-quant`'s row-count check lets the load succeed, rows 100..128 of `up` decoding under
    /// `gate`'s scale block.
    #[test]
    fn fp8_routed_expert_gate_up_off_the_128_row_grid_is_refused() {
        let (cfg, mp, dcfg) = stack_of(fp8_moe_config_json(100));
        let (fp8, _) = quantize_to_fp8(
            &write_synthetic_deepseek32_checkpoint_bytes(&cfg, &mp, &dcfg),
            &routed_expert_targets(&cfg, &mp),
        );
        let error = load(&fp8, &cfg, &mp).expect_err("a 100-row FP8 gate||up must be refused");
        assert_eq!(
            packed_weight_error(&error),
            Some(&PackedWeightError::RowConcatUnaligned {
                format: WeightFormat::E4m3Block128 {
                    scale: ScaleEncoding::F32
                },
                role: OperandRole::Scale,
                rows: 100,
                block_rows: 128,
            }),
            "{error}"
        );
    }

    /// SC-003 (Card 654): an E8M0 `0xff` (NaN) block scale on one routed expert's `gate_proj` is
    /// refused at load with the typed `PackedWeightError::NonFiniteField`, never widened into NaN weights.
    /// Mutation: dropping `poot-quant`'s payload finiteness check lets the load succeed with a NaN expert.
    #[test]
    fn fp8_routed_expert_nan_e8m0_scale_is_refused_at_load() {
        let (cfg, mp, dcfg) = stack_of(fp8_moe_config_json(128));
        // Every routed expert E8M0-scaled (one encoding per expert stack), one of them poisoned.
        let mut targets: Vec<(String, BlockScale)> = routed_expert_targets(&cfg, &mp)
            .into_iter()
            .map(|(name, _)| (name, BlockScale::E8m0(122)))
            .collect();
        let poisoned = "model.layers.1.mlp.experts.2.gate_proj.weight";
        targets
            .iter_mut()
            .find(|(name, _)| name == poisoned)
            .expect("the poisoned expert is a target")
            .1 = BlockScale::E8m0(0xff);
        let (fp8, _) = quantize_to_fp8(
            &write_synthetic_deepseek32_checkpoint_bytes(&cfg, &mp, &dcfg),
            &targets,
        );
        let error = load(&fp8, &cfg, &mp).expect_err("a NaN E8M0 scale must be refused");
        assert!(
            matches!(
                packed_weight_error(&error),
                Some(PackedWeightError::NonFiniteField {
                    format: WeightFormat::E4m3Block128 {
                        scale: ScaleEncoding::E8m0
                    },
                    operand: OperandRole::Scale,
                    ..
                })
            ),
            "{error}"
        );
        assert!(error.to_string().contains(poisoned), "{error}");
    }
}
