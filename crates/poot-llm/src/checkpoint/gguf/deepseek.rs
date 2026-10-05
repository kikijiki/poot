use poot_eval::{Value, materialize_dense};
use poot_graph_plan::WeightFormats;
use poot_load::gguf::GgufIndex;
use poot_quant::weights::WeightStore;
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::permute::gather_from;
use super::{gguf_deepseek2_yarn_params, gguf_u32};
use crate::checkpoint::place::{StoredRows, WeightPlacement};
use crate::error::{OptionExt, Result, ResultExt};

/// Build a [`poot_models::deepseek2::DeepseekV2Config`] from a DeepSeek-V2 GGUF's `deepseek2.*` metadata
/// (llama.cpp's `conversion/deepseek.py::DeepseekV2Model.set_gguf_parameters`,
/// `src/models/deepseek2.cpp::load_arch_hparams`/`load_arch_tensors`):
/// - `deepseek2.attention.kv_lora_rank`, `deepseek2.attention.q_lora_rank` (optional: present only when the
///   checkpoint compresses Q; DeepSeek-V2-Lite's GGUF omits it, as the safetensors path's
///   `hf.q_lora_rank: Option<usize>`).
/// - `deepseek2.attention.key_length_mla` (= `qk_nope_head_dim + qk_rope_head_dim`, the per-head Q/K width
///   after decompression) and `deepseek2.rope.dimension_count` (= `qk_rope_head_dim`; DeepSeek's meaning of
///   this key differs from other archs' full-rotary width): `qk_nope_head_dim` is their difference, since
///   the GGUF has no such key.
/// - `deepseek2.attention.value_length_mla` (= `v_head_dim`), independent of `key_length_mla`.
pub(crate) fn gguf_config_deepseek2(
    g: &GgufIndex,
) -> Result<poot_models::deepseek2::DeepseekV2Config> {
    let k = |s: &str| format!("deepseek2.{s}");
    let vocab = g
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .context("deepseek2 gguf missing tokenizer.ggml.tokens")?
        .len();
    let max_pos = gguf_u32(g, &k("context_length"))? as usize;
    let key_length_mla = gguf_u32(g, &k("attention.key_length_mla"))? as usize;
    let qk_rope_head_dim = gguf_u32(g, &k("rope.dimension_count"))? as usize;
    let qk_nope_head_dim = key_length_mla
        .checked_sub(qk_rope_head_dim)
        .context("deepseek2 gguf: key_length_mla < rope.dimension_count")?;
    Ok(poot_models::deepseek2::DeepseekV2Config {
        vocab,
        hidden: gguf_u32(g, &k("embedding_length"))? as usize,
        layers: gguf_u32(g, &k("block_count"))? as usize,
        n_heads: gguf_u32(g, &k("attention.head_count"))? as usize,
        q_lora_rank: g
            .get(&k("attention.q_lora_rank"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        kv_lora_rank: gguf_u32(g, &k("attention.kv_lora_rank"))? as usize,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim: gguf_u32(g, &k("attention.value_length_mla"))? as usize,
        eps: g
            .get(&k("attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6),
        max_pos,
        rope_theta: g
            .get(&k("rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0),
        yarn: gguf_deepseek2_yarn_params(g, max_pos),
    })
}

/// Build a [`poot_models::deepseek2::DeepseekV2MoeParams`] from a DeepSeek-V2 GGUF's `deepseek2.*` MoE
/// metadata: the `expert_count`/`expert_used_count`/`expert_feed_forward_length` keys every MoE arch uses,
/// plus DeepSeek's `leading_dense_block_count`/`expert_shared_count`/`expert_weights_scale`
/// (`src/models/deepseek2.cpp::load_arch_hparams`). `dense_inter` (the dense-layer FFN width,
/// `deepseek2.feed_forward_length`) is passed in, as the safetensors path's `dmoe.dense_inter = cfg.inter`.
///
/// `deepseek2.expert_group_count`/`deepseek2.expert_group_used_count` (`n_group`/`topk_group`), the keys
/// [`gguf_config_deepseek3_moe`] reads, default to `1`/`n_group` when absent (the group-limiting no-op
/// shape, see `poot_models::deepseek2::deepseek2_router_gate`). Real non-Lite DeepSeek-V2 conversions set
/// them (`~/models/deepseek2-tiny`: `expert_group_count: 8`, `expert_group_used_count: 3`).
pub(crate) fn gguf_config_deepseek2_moe(
    g: &GgufIndex,
    dense_inter: usize,
) -> Result<poot_models::deepseek2::DeepseekV2MoeParams> {
    let k = |s: &str| format!("deepseek2.{s}");
    let n_group = gguf_u32(g, &k("expert_group_count")).unwrap_or(1) as usize;
    let topk_group = gguf_u32(g, &k("expert_group_used_count")).unwrap_or(n_group as u32) as usize;
    Ok(poot_models::deepseek2::DeepseekV2MoeParams {
        n_routed_experts: gguf_u32(g, &k("expert_count"))? as usize,
        top_k: gguf_u32(g, &k("expert_used_count"))? as usize,
        moe_inter: gguf_u32(g, &k("expert_feed_forward_length"))? as usize,
        n_shared_experts: gguf_u32(g, &k("expert_shared_count")).unwrap_or(0) as usize,
        dense_inter,
        first_k_dense_replace: gguf_u32(g, &k("leading_dense_block_count")).unwrap_or(1) as usize,
        n_group,
        topk_group,
        routed_scaling_factor: g
            .get(&k("expert_weights_scale"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1.0),
    })
}

/// Reconstruct DeepSeek-V2's `self_attn.kv_b_proj.weight` (poot's
/// `[kv_lora_rank, n_head*(qk_nope_head_dim+v_head_dim)]` matmul-convention layout: the un-absorbed per-head
/// K(nope)/V decompression weight `trace_deepseek2_prefill`/`trace_deepseek2_decode_kv_masked` bind, as
/// the safetensors path's generic `proj.weight` transpose loop produces) from llama.cpp's GGUF tensors.
///
/// llama.cpp's conversion does not carry the combined HF `kv_b_proj.weight` for a current GGUF (only old
/// pre-split files have it, as `attn_kv_b`; `src/models/deepseek2.cpp::load_arch_tensors`'s `is_mla` branch).
/// `conversion/deepseek.py::DeepseekV2Model.modify_tensors` splits it into two tensors for llama.cpp's MLA
/// weight-absorption optimization. poot does not absorb weights (see `poot_models::deepseek2` module docs,
/// item 4), so this un-splits them:
/// - `attn_k_b.weight`: the per-head nope-K weight, transposed (`k_b = kv_b[..,:qk_nope_head_dim,:]
///   .transpose(1,2)`) to ggml `ne=[qk_nope_head_dim,kv_lora_rank,n_head]`, i.e. the stored tensor's
///   row-major `[n_head,kv_lora_rank,qk_nope_head_dim]`. Each per-head `[kv_lora_rank,qk_nope_head_dim]`
///   slice is already in poot's `[in,out]` orientation.
/// - `attn_v_b.weight`: the per-head V weight, untransposed (`v_b = kv_b[..,qk_nope_head_dim:,:]`), ggml
///   `ne=[kv_lora_rank,v_head_dim,n_head]`, row-major `[n_head,v_head_dim,kv_lora_rank]`. Each per-head slice
///   needs a transpose to `[kv_lora_rank, v_head_dim]` before going into the V column block.
///
/// The ggml shapes are from `src/models/deepseek2.cpp`'s `create_tensor(tn(LLM_TENSOR_ATTN_K_B, ...),
/// {n_embd_head_qk_nope, kv_lora_rank, n_head}, ...)` / `create_tensor(tn(LLM_TENSOR_ATTN_V_B, ...),
/// {kv_lora_rank, n_embd_head_v_mla, n_head}, ...)`.
fn reconstruct_deepseek2_kv_b(
    k_b: &HostTensor,
    v_b: &HostTensor,
    n_head: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
) -> HostTensor {
    debug_assert_eq!(k_b.shape(), vec![n_head, kv_lora_rank, qk_nope_head_dim]);
    debug_assert_eq!(v_b.shape(), vec![n_head, v_head_dim, kv_lora_rank]);
    let out_per_head = qk_nope_head_dim + v_head_dim;
    let out = n_head * out_per_head;
    // Output `[r, h*out_per_head + c]` is head `h`'s `k_b[h, r, c]` for the nope columns and the
    // transposed `v_b[h, c, r]` for the value columns; one gather in the checkpoint's own dtype.
    gather_from(
        &[k_b, v_b],
        vec![kv_lora_rank, out],
        (0..kv_lora_rank).flat_map(move |r| {
            (0..n_head).flat_map(move |h| {
                let k_base = h * kv_lora_rank * qk_nope_head_dim;
                let v_base = h * v_head_dim * kv_lora_rank;
                (0..qk_nope_head_dim)
                    .map(move |c| (0, k_base + r * qk_nope_head_dim + c))
                    .chain((0..v_head_dim).map(move |c| (1, v_base + c * kv_lora_rank + r)))
            })
        }),
    )
}

/// Build the HF-named weight map `trace_deepseek2_prefill`/`trace_deepseek2_decode_kv_masked` bind from a
/// DeepSeek-V2 GGUF, under the constant names the safetensors path (`runner.rs`'s `build_weights`
/// deepseek2 block) uses. Unlike mixtral/olmoe/qwen3moe (qwen2-shaped attention, handled by the generic
/// `gguf_weights`), DeepSeek-V2's MLA attention does not map onto `Qwen2Config`'s
/// `n_heads`/`n_kv_heads`/`head_dim`/`rotary_dim`, so this is a dedicated builder like
/// `gguf_bloom_weights`/`gguf_mpt_weights`.
pub(crate) fn gguf_deepseek2_weights(
    g: &GgufIndex,
    store: &WeightStore,
    cfg: &poot_models::deepseek2::DeepseekV2Config,
    mp: &poot_models::deepseek2::DeepseekV2MoeParams,
) -> Result<(HashMap<String, Value>, WeightFormats)> {
    let mut p = WeightPlacement::default();
    let deq = |name: &str| -> Result<HostTensor> {
        materialize_dense(store, name)
            .with_context(|| format!("deepseek2 gguf: materialize {name}"))
    };
    let rd = |name: &str| StoredRows::read(store, name);
    let experts = |name: &str| StoredRows::experts(store, name);

    p.table("model.embed_tokens.weight", rd("token_embd.weight")?)?;

    for li in 0..cfg.layers {
        let hf = |s: &str| format!("model.layers.{li}.{s}");
        let blk = |s: &str| format!("blk.{li}.{s}");

        p.dense(hf("input_layernorm.weight"), deq(&blk("attn_norm.weight"))?);
        p.dense(
            hf("post_attention_layernorm.weight"),
            deq(&blk("ffn_norm.weight"))?,
        );

        // MLA's Q side: low-rank-compressed (attn_q_a/attn_q_a_norm/attn_q_b) when `q_lora_rank.is_some()` (real
        // V2/V3 and the tiny test fixture), else a plain attn_q (real DeepSeek-V2-Lite); see
        // `DeepseekV2Config::q_lora_rank`.
        if cfg.q_lora_rank.is_some() {
            p.projection(
                hf("self_attn.q_a_proj.weight"),
                rd(&blk("attn_q_a.weight"))?,
            )?;
            p.dense(
                hf("self_attn.q_a_layernorm.weight"),
                deq(&blk("attn_q_a_norm.weight"))?,
            );
            p.projection(
                hf("self_attn.q_b_proj.weight"),
                rd(&blk("attn_q_b.weight"))?,
            )?;
        } else {
            p.projection(hf("self_attn.q_proj.weight"), rd(&blk("attn_q.weight"))?)?;
        }

        // The shared low-rank KV latent compression (kv_a_proj_with_mqa + kv_a_layernorm): a plain 2-D `[out,in]`
        // GGUF tensor and a plain 1-D norm.
        p.projection(
            hf("self_attn.kv_a_proj_with_mqa.weight"),
            rd(&blk("attn_kv_a_mqa.weight"))?,
        )?;
        p.dense(
            hf("self_attn.kv_a_layernorm.weight"),
            deq(&blk("attn_kv_a_norm.weight"))?,
        );
        // The MLA decompression weight, reconstructed from llama.cpp's split attn_k_b/attn_v_b tensors; see
        // `reconstruct_deepseek2_kv_b`.
        let k_b = deq(&blk("attn_k_b.weight"))?;
        let v_b = deq(&blk("attn_v_b.weight"))?;
        p.dense(
            hf("self_attn.kv_b_proj.weight"),
            reconstruct_deepseek2_kv_b(
                &k_b,
                &v_b,
                cfg.n_heads,
                cfg.kv_lora_rank,
                cfg.qk_nope_head_dim,
                cfg.v_head_dim,
            ),
        );
        p.projection(
            hf("self_attn.o_proj.weight"),
            rd(&blk("attn_output.weight"))?,
        )?;

        // Per-layer dense-vs-routed+shared MoE MLP split (`mp.first_k_dense_replace`), the same threshold as the
        // safetensors path's deepseek2 block in `runner.rs`.
        if li < mp.first_k_dense_replace {
            p.projection(hf("mlp.gate_proj.weight"), rd(&blk("ffn_gate.weight"))?)?;
            p.projection(hf("mlp.up_proj.weight"), rd(&blk("ffn_up.weight"))?)?;
            p.projection(hf("mlp.down_proj.weight"), rd(&blk("ffn_down.weight"))?)?;
        } else {
            p.projection(hf("mlp.gate.weight"), rd(&blk("ffn_gate_inp.weight"))?)?;
            let gate = experts(&blk("ffn_gate_exps.weight"))?; // E x [I,H]
            let up = experts(&blk("ffn_up_exps.weight"))?; // E x [I,H]
            let gate_up = gate
                .iter()
                .zip(&up)
                .map(|(gate, up)| {
                    StoredRows::gather(&[gate, up], &(0..2 * gate.rows()).collect::<Vec<_>>())
                })
                .collect::<Result<Vec<_>>>()?;
            p.experts(hf("mlp.experts.gate_up_proj.weight"), gate_up)?; // [E,H,2I] (gate||up)
            p.experts(
                hf("mlp.experts.down_proj.weight"),
                experts(&blk("ffn_down_exps.weight"))?,
            )?; // [E,I,H]
            if mp.n_shared_experts > 0 {
                // the always-active shared expert(s): one combined plain SwiGLU MLP (no extra gate), a plain 2-D GGUF
                // tensor per projection, not stacked per expert (llama.cpp writes it this way for every shared-expert MoE
                // arch; `LLM_TENSOR_FFN_*_SHEXP` in `src/llama-arch.cpp`).
                p.projection(
                    hf("mlp.shared_experts.gate_proj.weight"),
                    rd(&blk("ffn_gate_shexp.weight"))?,
                )?;
                p.projection(
                    hf("mlp.shared_experts.up_proj.weight"),
                    rd(&blk("ffn_up_shexp.weight"))?,
                )?;
                p.projection(
                    hf("mlp.shared_experts.down_proj.weight"),
                    rd(&blk("ffn_down_shexp.weight"))?,
                )?;
            }
        }
    }

    p.dense("model.norm.weight".to_string(), deq("output_norm.weight")?);
    let lm_head = if g.tensors.contains_key("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    p.projection("lm_head.weight", rd(lm_head)?)?;

    // DeepSeek's interleaved-pair RoPE table (half the width of the generic half-split table other archs'
    // `gguf_rope_tables` produce: one angle per pair), as the safetensors path's override in `runner.rs`'s
    // `build_weights` (see "deepseek2_rope_tables_interleaved").
    let (cos, sin) = poot_models::deepseek2::deepseek2_rope_tables_interleaved(
        cfg.max_pos,
        cfg.qk_rope_head_dim,
        cfg.rope_theta,
        cfg.yarn.as_ref(),
    );
    let table_shape = vec![cfg.max_pos, cfg.qk_rope_head_dim / 2];
    p.dense(
        "rope.cos".to_string(),
        HostTensor::f32(table_shape.clone(), cos),
    );
    p.dense("rope.sin".to_string(), HostTensor::f32(table_shape, sin));

    Ok(p.finish())
}

/// DeepSeek-V2 and DeepSeek-V3 checkpoints share the llama.cpp GGUF architecture string `"deepseek2"`
/// (`DeepseekV2Model` is registered for both `DeepseekV2ForCausalLM` and `DeepseekV3ForCausalLM` and always
/// sets `model_arch = gguf.MODEL_ARCH.DEEPSEEK2`). The discriminator is `blk.N.exp_probs_b.bias`
/// (llama.cpp's `TENSOR_NOT_REQUIRED` tensor for the `noaux_tc` selection bias, `e_score_correction_bias`
/// in HF): V2's softmax router never has it, V3 always does since
/// [`poot_models::deepseek3::deepseek3_router_gate`] requires it. Metadata such as
/// `deepseek2.expert_group_count` is not a safe discriminator: non-Lite DeepSeek-V2 also sets
/// `n_group`/`topk_group` (see `poot_models::deepseek2`).
pub(crate) fn gguf_deepseek2_is_v3_style(g: &GgufIndex) -> bool {
    g.tensors
        .keys()
        .any(|name| name.ends_with(".exp_probs_b.bias"))
}

/// Build a [`poot_models::deepseek3::DeepseekV3MoeParams`] from a DeepSeek-V3-style `"deepseek2"` GGUF's
/// metadata: the key convention [`gguf_config_deepseek2_moe`] reads, into a struct with the two
/// group-routing fields. `deepseek2.expert_group_count`/`deepseek2.expert_group_used_count`
/// (`n_group`/`topk_group`; llama.cpp's generic `TextModel.set_gguf_parameters` writes them from the HF
/// config) default to `1`/`n_group` when absent, the no-op shape documented on
/// [`poot_models::deepseek3::deepseek3_router_gate`].
pub(crate) fn gguf_config_deepseek3_moe(
    g: &GgufIndex,
    dense_inter: usize,
) -> Result<poot_models::deepseek3::DeepseekV3MoeParams> {
    let k = |s: &str| format!("deepseek2.{s}");
    let n_group = gguf_u32(g, &k("expert_group_count")).unwrap_or(1) as usize;
    let topk_group = gguf_u32(g, &k("expert_group_used_count")).unwrap_or(n_group as u32) as usize;
    Ok(poot_models::deepseek3::DeepseekV3MoeParams {
        n_routed_experts: gguf_u32(g, &k("expert_count"))? as usize,
        top_k: gguf_u32(g, &k("expert_used_count"))? as usize,
        moe_inter: gguf_u32(g, &k("expert_feed_forward_length"))? as usize,
        n_shared_experts: gguf_u32(g, &k("expert_shared_count")).unwrap_or(0) as usize,
        dense_inter,
        first_k_dense_replace: gguf_u32(g, &k("leading_dense_block_count")).unwrap_or(1) as usize,
        n_group,
        topk_group,
        routed_scaling_factor: g
            .get(&k("expert_weights_scale"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1.0),
    })
}

/// Build the HF-named weight map `poot_models::deepseek3::trace_deepseek3_prefill`/
/// `trace_deepseek3_decode_kv_masked` bind from a DeepSeek-V3-style `"deepseek2"` GGUF. A near copy of
/// [`gguf_deepseek2_weights`] (attention, dense FFN, routed and shared expert stacks, and the RoPE table
/// override use identical name-mapping/dequant logic; see `poot_models::deepseek3` module docs), duplicated
/// rather than parameterized like the other per-arch `gguf_*_weights` functions. The one addition: the
/// routed-MoE branch also reads `blk.N.exp_probs_b.bias` (a plain `[n_routed_experts]` vector, no
/// transpose) and inserts it as `mlp.gate.e_score_correction_bias`, the name
/// `poot_models::deepseek3::deepseek3_moe_ffn` binds.
pub(crate) fn gguf_deepseek3_weights(
    g: &GgufIndex,
    store: &WeightStore,
    cfg: &poot_models::deepseek2::DeepseekV2Config,
    mp: &poot_models::deepseek3::DeepseekV3MoeParams,
) -> Result<(HashMap<String, Value>, WeightFormats)> {
    let mut p = WeightPlacement::default();
    let deq = |name: &str| -> Result<HostTensor> {
        materialize_dense(store, name)
            .with_context(|| format!("deepseek2 (v3-style) gguf: materialize {name}"))
    };
    let rd = |name: &str| StoredRows::read(store, name);
    let experts = |name: &str| StoredRows::experts(store, name);

    p.table("model.embed_tokens.weight", rd("token_embd.weight")?)?;

    for li in 0..cfg.layers {
        let hf = |s: &str| format!("model.layers.{li}.{s}");
        let blk = |s: &str| format!("blk.{li}.{s}");

        p.dense(hf("input_layernorm.weight"), deq(&blk("attn_norm.weight"))?);
        p.dense(
            hf("post_attention_layernorm.weight"),
            deq(&blk("ffn_norm.weight"))?,
        );

        if cfg.q_lora_rank.is_some() {
            p.projection(
                hf("self_attn.q_a_proj.weight"),
                rd(&blk("attn_q_a.weight"))?,
            )?;
            p.dense(
                hf("self_attn.q_a_layernorm.weight"),
                deq(&blk("attn_q_a_norm.weight"))?,
            );
            p.projection(
                hf("self_attn.q_b_proj.weight"),
                rd(&blk("attn_q_b.weight"))?,
            )?;
        } else {
            p.projection(hf("self_attn.q_proj.weight"), rd(&blk("attn_q.weight"))?)?;
        }

        p.projection(
            hf("self_attn.kv_a_proj_with_mqa.weight"),
            rd(&blk("attn_kv_a_mqa.weight"))?,
        )?;
        p.dense(
            hf("self_attn.kv_a_layernorm.weight"),
            deq(&blk("attn_kv_a_norm.weight"))?,
        );
        let k_b = deq(&blk("attn_k_b.weight"))?;
        let v_b = deq(&blk("attn_v_b.weight"))?;
        p.dense(
            hf("self_attn.kv_b_proj.weight"),
            reconstruct_deepseek2_kv_b(
                &k_b,
                &v_b,
                cfg.n_heads,
                cfg.kv_lora_rank,
                cfg.qk_nope_head_dim,
                cfg.v_head_dim,
            ),
        );
        p.projection(
            hf("self_attn.o_proj.weight"),
            rd(&blk("attn_output.weight"))?,
        )?;

        if li < mp.first_k_dense_replace {
            p.projection(hf("mlp.gate_proj.weight"), rd(&blk("ffn_gate.weight"))?)?;
            p.projection(hf("mlp.up_proj.weight"), rd(&blk("ffn_up.weight"))?)?;
            p.projection(hf("mlp.down_proj.weight"), rd(&blk("ffn_down.weight"))?)?;
        } else {
            p.projection(hf("mlp.gate.weight"), rd(&blk("ffn_gate_inp.weight"))?)?;
            p.dense(
                hf("mlp.gate.e_score_correction_bias"),
                deq(&blk("exp_probs_b.bias"))?,
            );
            let gate = experts(&blk("ffn_gate_exps.weight"))?; // E x [I,H]
            let up = experts(&blk("ffn_up_exps.weight"))?; // E x [I,H]
            let gate_up = gate
                .iter()
                .zip(&up)
                .map(|(gate, up)| {
                    StoredRows::gather(&[gate, up], &(0..2 * gate.rows()).collect::<Vec<_>>())
                })
                .collect::<Result<Vec<_>>>()?;
            p.experts(hf("mlp.experts.gate_up_proj.weight"), gate_up)?; // [E,H,2I] (gate||up)
            p.experts(
                hf("mlp.experts.down_proj.weight"),
                experts(&blk("ffn_down_exps.weight"))?,
            )?; // [E,I,H]
            if mp.n_shared_experts > 0 {
                p.projection(
                    hf("mlp.shared_experts.gate_proj.weight"),
                    rd(&blk("ffn_gate_shexp.weight"))?,
                )?;
                p.projection(
                    hf("mlp.shared_experts.up_proj.weight"),
                    rd(&blk("ffn_up_shexp.weight"))?,
                )?;
                p.projection(
                    hf("mlp.shared_experts.down_proj.weight"),
                    rd(&blk("ffn_down_shexp.weight"))?,
                )?;
            }
        }
    }

    p.dense("model.norm.weight".to_string(), deq("output_norm.weight")?);
    let lm_head = if g.tensors.contains_key("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    p.projection("lm_head.weight", rd(lm_head)?)?;

    let (cos, sin) = poot_models::deepseek2::deepseek2_rope_tables_interleaved(
        cfg.max_pos,
        cfg.qk_rope_head_dim,
        cfg.rope_theta,
        cfg.yarn.as_ref(),
    );
    let table_shape = vec![cfg.max_pos, cfg.qk_rope_head_dim / 2];
    p.dense(
        "rope.cos".to_string(),
        HostTensor::f32(table_shape.clone(), cos),
    );
    p.dense("rope.sin".to_string(), HostTensor::f32(table_shape, sin));

    Ok(p.finish())
}
