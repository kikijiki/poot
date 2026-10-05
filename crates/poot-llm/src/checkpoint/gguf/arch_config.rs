use poot_eval::{Value, materialize_dense};
use poot_graph_plan::WeightFormats;
use poot_load::gguf::GgufIndex;
use poot_models::qwen2::Qwen2Config;
use poot_quant::weights::WeightStore;
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::experts::interleave_experts_last;
use super::rope::{gguf_rope_tables, rope_tables};
use crate::checkpoint::place::{StoredRows, WeightPlacement, interleave_rows, unpermute_qk_rows};
use crate::error::{Result, ResultExt};

/// Build the HF-named weight map the qwen2-shaped tracers (and every other
/// qwen2-shaped arch's tracer) expect, from a GGUF-loaded [`WeightStore`] (dquant.md 8.1), with the
/// [`WeightFormats`] of the weights the file stores packed (card 545a). A packed
/// tensor stays packed - its payload shared, its loader row ops done on the stored blocks - and binds
/// onto the traced graph through `bind_packed_weights`; a dense tensor loads as before.
pub(crate) fn gguf_weights(
    g: &GgufIndex,
    store: &WeightStore,
    cfg: &Qwen2Config,
    arch: &str,
) -> Result<(HashMap<String, Value>, WeightFormats)> {
    let mut p = WeightPlacement::default();
    let deq = |name: &str| -> Result<HostTensor> {
        materialize_dense(store, name).with_context(|| format!("materialize {name}"))
    };
    let rd = |name: &str| StoredRows::read(store, name);
    let experts = |name: &str| StoredRows::experts(store, name);
    // gate||up per expert: each expert's gate rows then its up rows (`interleave` for gpt-oss: even
    // rows gate, odd rows up).
    let fuse_gate_up = |gate: Vec<StoredRows>, up: Vec<StoredRows>, interleave: bool| {
        gate.iter()
            .zip(&up)
            .map(|(gate, up)| {
                let n = gate.rows();
                let rows: Vec<usize> = if interleave {
                    interleave_rows(n)
                } else {
                    (0..2 * n).collect()
                };
                StoredRows::gather(&[gate, up], &rows)
            })
            .collect::<Result<Vec<_>>>()
    };

    // token embedding: GGUF [vocab,hidden] (ne reversed), kept as the gather source (no transpose).
    p.table("model.embed_tokens.weight", rd("token_embd.weight")?)?;
    for i in 0..cfg.layers {
        let hf = |s: &str| format!("model.layers.{i}.{s}");
        let blk = |s: &str| format!("blk.{i}.{s}");
        // llama-arch GGUFs store q/k row-permuted for ggml's rope; un-permute to HF order (qwen2 is not).
        // smollm3's converter class (`SmolLM3Model(LlamaModel)`) overrides nothing but `model_arch`, so it inherits
        // `LlamaModel`'s `undo_permute = True` and gets the same q/k row-permute as plain llama (`conversion/llama.py`
        // gates the permute on `self.undo_permute`; unlike `Llama4Model`/`ApertusModel`, `SmolLM3Model` never sets
        // it to False).
        let qk = |rows: StoredRows, heads: usize| {
            if arch == "llama" || arch == "smollm3" {
                StoredRows::gather(&[&rows], &unpermute_qk_rows(rows.rows(), heads))
            } else {
                Ok(rows)
            }
        };
        if arch == "phi3" {
            // phi3 fuses q|k|v into one `attn_qkv` tensor (dequant -> [q_dim+2*kv_dim, in]); split by rows. phi3
            // GGUFs store q/k in HF order (no ggml rope permute), so no unpermute.
            let q_dim = cfg.n_heads * cfg.head_dim;
            let kv_dim = cfg.n_kv_heads * cfg.head_dim;
            let qkv = rd(&blk("attn_qkv.weight"))?;
            p.projection(hf("self_attn.q_proj.weight"), qkv.slice(0, q_dim)?)?;
            p.projection(
                hf("self_attn.k_proj.weight"),
                qkv.slice(q_dim, q_dim + kv_dim)?,
            )?;
            p.projection(
                hf("self_attn.v_proj.weight"),
                qkv.slice(q_dim + kv_dim, q_dim + 2 * kv_dim)?,
            )?;
        } else {
            p.projection(
                hf("self_attn.q_proj.weight"),
                qk(rd(&blk("attn_q.weight"))?, cfg.n_heads)?,
            )?;
            p.projection(
                hf("self_attn.k_proj.weight"),
                qk(rd(&blk("attn_k.weight"))?, cfg.n_kv_heads)?,
            )?;
            p.projection(hf("self_attn.v_proj.weight"), rd(&blk("attn_v.weight"))?)?;
        }
        p.projection(
            hf("self_attn.o_proj.weight"),
            rd(&blk("attn_output.weight"))?,
        )?;
        if arch == "gpt-oss" {
            // gpt-oss's attention has a bias on q/k/v and o (`attention_bias: true`), unlike qwen2's q/k/v-only bias
            // (the only other GGUF arch marked `qkv_bias`). The generic `cfg.qkv_bias` block below reads only
            // `attn_{q,k,v}.bias`, and no other arch's GGUF needs an o_proj bias, so it is handled here. llama.cpp's
            // `src/models/openai-moe.cpp` `load_arch_tensors` requires it (`layer.wo_b`, not `TENSOR_NOT_REQUIRED`).
            p.dense(hf("self_attn.o_proj.bias"), deq(&blk("attn_output.bias"))?);
            // Attention sinks (see `poot_models::gpt_oss`'s module docs, point 1): one learned scalar per query head,
            // competing in the softmax row-max/denominator but excluded from the weighted-V sum. llama.cpp's
            // `conversion/gpt_oss.py` `filter_tensors` appends ".weight" to the sinks name (`if "sinks" in name:
            // name += ".weight"`), so the GGUF tensor is ".weight"-suffixed although it is a raw `nn.Parameter` in
            // safetensors; `src/models/openai-moe.cpp` loads it as `layer.attn_sinks` and passes it to `build_attn`. poot's
            // `attention_prefill_with_sink`/`attention_masked_with_sink` implement the same mechanic.
            p.dense(hf("self_attn.sinks"), deq(&blk("attn_sinks.weight"))?);
        }
        if cfg.qkv_bias {
            p.dense(hf("self_attn.q_proj.bias"), deq(&blk("attn_q.bias"))?);
            p.dense(hf("self_attn.k_proj.bias"), deq(&blk("attn_k.bias"))?);
            p.dense(hf("self_attn.v_proj.bias"), deq(&blk("attn_v.bias"))?);
        }
        if arch == "olmo2" {
            // OLMo2 is post-norm: no input/pre layernorms (the projections read the raw residual). Norm the attention
            // and MLP outputs before each residual add. (Full-projection QK-norm below.)
            p.dense(
                hf("post_attention_layernorm.weight"),
                deq(&blk("post_attention_norm.weight"))?,
            );
            p.dense(
                hf("post_feedforward_layernorm.weight"),
                deq(&blk("post_ffw_norm.weight"))?,
            );
        } else if arch == "gemma3" {
            // gemma3's 4-norm sandwich: input (attn_norm), post_attention_norm, ffn_norm=pre-FF, post_ffw_norm.
            // (qwen2/llama/qwen3 map ffn_norm to the single post-attention norm, the 2-norm convention below.)
            p.dense(hf("input_layernorm.weight"), deq(&blk("attn_norm.weight"))?);
            p.dense(
                hf("post_attention_layernorm.weight"),
                deq(&blk("post_attention_norm.weight"))?,
            );
            p.dense(
                hf("pre_feedforward_layernorm.weight"),
                deq(&blk("ffn_norm.weight"))?,
            );
            p.dense(
                hf("post_feedforward_layernorm.weight"),
                deq(&blk("post_ffw_norm.weight"))?,
            );
        } else if arch == "gpt-oss" {
            // gpt-oss is a standard 2-norm pre-norm block (input_layernorm before attention, one more before the FFN,
            // as in the generic `else` branch below), but its GGUF names the second norm `post_attention_norm`: the same
            // tensor name olmo2's arm above reads (`blk.N.post_attention_norm.weight`), with a different role (pre-FFN,
            // not post-residual). Confirmed by llama.cpp's `src/models/openai-moe.cpp` (`build_norm(inpL, attn_norm)`
            // before attention, `build_norm(ffn_inp, attn_post_norm)` before the MoE FFN) and `gguf-py`'s
            // `MODEL_TENSOR.ATTN_POST_NORM: "blk.{bid}.post_attention_norm"`, shared by both archs despite different
            // norm placement.
            p.dense(hf("input_layernorm.weight"), deq(&blk("attn_norm.weight"))?);
            p.dense(
                hf("post_attention_layernorm.weight"),
                deq(&blk("post_attention_norm.weight"))?,
            );
        } else {
            p.dense(hf("input_layernorm.weight"), deq(&blk("attn_norm.weight"))?);
            p.dense(
                hf("post_attention_layernorm.weight"),
                deq(&blk("ffn_norm.weight"))?,
            );
        }
        if cfg.qk_norm {
            // QK-norm before RoPE: per-head (over head_dim) for gemma3/qwen3, full-projection (over q_dim/kv_dim) for
            // olmo2; the GGUF tensor is sized to match the tracer, so copy as-is.
            p.dense(
                hf("self_attn.q_norm.weight"),
                deq(&blk("attn_q_norm.weight"))?,
            );
            p.dense(
                hf("self_attn.k_norm.weight"),
                deq(&blk("attn_k_norm.weight"))?,
            );
        }
        if arch == "granitemoe" {
            // granitemoe: a top-k expert mixture, not a dense MLP. GGUF carries the experts as 3D
            // `ffn_{gate,up,down}_exps` ([E, out, in] after ne-reversal) plus the `ffn_gate_inp` router. The moe op
            // wants router [H,E], input_linear [E,H,2I] (gate||up fused, [in,out]), output_linear [E,I,H].
            p.projection(
                hf("block_sparse_moe.router.layer.weight"),
                rd(&blk("ffn_gate_inp.weight"))?,
            )?; // [E,H] -> [H,E]
            let gate = experts(&blk("ffn_gate_exps.weight"))?; // E x [I,H]
            let up = experts(&blk("ffn_up_exps.weight"))?; // E x [I,H]
            p.experts(
                hf("block_sparse_moe.input_linear.weight"),
                fuse_gate_up(gate, up, false)?,
            )?; // [E,H,2I] (gate||up)
            p.experts(
                hf("block_sparse_moe.output_linear.weight"),
                experts(&blk("ffn_down_exps.weight"))?,
            )?; // [E,I,H]
        } else if arch == "qwen3moe" && g.tensors.contains_key(&blk("ffn_gate_inp.weight")) {
            // qwen3moe's routed layers. Same GGUF tensor shapes/naming as granitemoe above (llama.cpp uses one MoE
            // tensor convention across every MoE arch: `blk.{bid}.ffn_gate_inp`/`ffn_{gate,up,down}_exps` in
            // gguf-py's TENSOR_NAMES), mapped to the names `poot_models::qwen3moe`'s tracer binds (see its module
            // docs; `runner.rs`'s `fuse_qwen3_moe_experts` is the safetensors analog). Gated on the router tensor's
            // presence rather than every layer, because qwen3moe's HF config allows per-layer dense/MoE mixing
            // (decoder_sparse_step/mlp_only_layers) that the GGUF carries no metadata for; a layer without
            // `ffn_gate_inp` falls through to the dense branch below.
            p.projection(hf("mlp.gate.weight"), rd(&blk("ffn_gate_inp.weight"))?)?; // [E,H] -> [H,E]
            let gate = experts(&blk("ffn_gate_exps.weight"))?; // E x [I,H]
            let up = experts(&blk("ffn_up_exps.weight"))?; // E x [I,H]
            p.experts(
                hf("mlp.experts.gate_up_proj.weight"),
                fuse_gate_up(gate, up, false)?,
            )?; // [E,H,2I] (gate||up)
            p.experts(
                hf("mlp.experts.down_proj.weight"),
                experts(&blk("ffn_down_exps.weight"))?,
            )?; // [E,I,H]
        } else if arch == "llama" && g.tensors.contains_key(&blk("ffn_gate_inp.weight")) {
            // Mixtral. It has no GGUF architecture string of its own: llama.cpp's `conversion/llama.py`
            // `LlamaModel` also registers `MixtralForCausalLM`, so it converts under `general.architecture = "llama"`
            // like dense Llama/SmolLM2. The only structural signal is the router tensor. A real conversion of
            // `optimum-intel-internal-testing/tiny-mixtral` carries `llama.expert_count`/`llama.expert_used_count` and
            // the same `ffn_gate_inp`/`ffn_{gate,up,down}_exps` 3D-stacked-expert convention as granitemoe/qwen3moe
            // above (`LlamaModel.modify_tensors` merges Mixtral's per-expert `w1`/`w2`/`w3` tensors into them). Same
            // shape crosswalk as those arms, mapped to `poot_models::mixtral`'s `block_sparse_moe.*` names (see
            // `runner.rs`'s safetensors `fuse_mixtral_experts`).
            //
            // The merged 3D layout is what current `convert_hf_to_gguf.py` emits, but older files differ:
            // `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` (the most-downloaded Mixtral-8x7B GGUF) stores each expert as
            // its own 2D tensor, `blk.N.ffn_{gate,up,down}.{0..n_experts-1}.weight` (confirmed from a header-only dump
            // of the real file). The layout is detected by the presence of the merged `ffn_gate_exps.weight`, and the
            // per-expert tensors are read as the same per-expert rows the merged layout splits into, as the
            // `qwen3moe` GGUF arm does for multiple layouts.
            p.projection(
                hf("block_sparse_moe.gate.weight"),
                rd(&blk("ffn_gate_inp.weight"))?,
            )?; // [E,H] -> [H,E]
            // The old per-expert layout reads the same experts from their own tensors.
            let stem_experts = |stem: &str| {
                if g.tensors.contains_key(&blk(&format!("{stem}_exps.weight"))) {
                    experts(&blk(&format!("{stem}_exps.weight")))
                } else {
                    let n_experts = super::gguf_u32(g, "llama.expert_count")? as usize;
                    (0..n_experts)
                        .map(|e| rd(&blk(&format!("{stem}.{e}.weight"))))
                        .collect()
                }
            };
            p.experts(
                hf("block_sparse_moe.experts.gate_up_proj.weight"),
                fuse_gate_up(stem_experts("ffn_gate")?, stem_experts("ffn_up")?, false)?,
            )?; // [E,H,2I] (gate||up)
            p.experts(
                hf("block_sparse_moe.experts.down_proj.weight"),
                stem_experts("ffn_down")?,
            )?; // [E,I,H]
        } else if arch == "olmoe" {
            // OlmoE. Unlike Mixtral it has its own GGUF architecture string ("olmoe"; llama.cpp's
            // `conversion/olmo.py` `OlmoeModel`, `MODEL_ARCH.OLMOE`). A real `convert_hf_to_gguf.py --outtype f32`
            // conversion of the `~/models/olmoe-tiny` fixture carries `general.architecture = olmoe`,
            // `olmoe.expert_count`/`olmoe.expert_used_count`, and the same `ffn_gate_inp`/`ffn_{gate,up,down}_exps`
            // 3D-stacked-expert convention as the arms above (`blk.0.ffn_gate_exps.weight` shape `{32, 16, 8}` =
            // `[H,I,E]` ne-reversed to `[E,I,H]`). Every OlmoE layer routes (llama.cpp's `src/models/olmoe.cpp` creates
            // the MoE tensors unconditionally and errors if `n_expert`/`n_expert_used` are 0), so no router-presence
            // gate is needed, unlike qwen3moe. Target names are qwen3moe's `mlp.*` names, not granitemoe's/Mixtral's
            // (`poot_models::olmoe` reuses qwen3moe's convention; `runner.rs`'s `fuse_qwen3_moe_experts` is the
            // safetensors analog).
            p.projection(hf("mlp.gate.weight"), rd(&blk("ffn_gate_inp.weight"))?)?; // [E,H] -> [H,E]
            let gate = experts(&blk("ffn_gate_exps.weight"))?; // E x [I,H]
            let up = experts(&blk("ffn_up_exps.weight"))?; // E x [I,H]
            p.experts(
                hf("mlp.experts.gate_up_proj.weight"),
                fuse_gate_up(gate, up, false)?,
            )?; // [E,H,2I] (gate||up)
            p.experts(
                hf("mlp.experts.down_proj.weight"),
                experts(&blk("ffn_down_exps.weight"))?,
            )?; // [E,I,H]
        } else if arch == "gpt-oss" {
            // gpt-oss has its own GGUF arch string "gpt-oss" (hyphen, unlike the HF `model_type` "gpt_oss";
            // `conversion/gpt_oss.py` `GptOssModel`, `MODEL_ARCH_NAMES[MODEL_ARCH.GPT_OSS] == "gpt-oss"`).
            //
            // Router: the same `ffn_gate_inp` tensor as every other MoE arm, but gpt-oss's router also has a bias
            // (`router_logits = F.linear(hidden_states, self.weight, self.bias)`, see `poot_models::gpt_oss`), the only
            // MoE arch here with one (`src/models/openai-moe.cpp`: `layer.ffn_gate_inp_b`).
            p.projection(hf("mlp.router.weight"), rd(&blk("ffn_gate_inp.weight"))?)?; // [E,H] -> [H,E]
            p.dense(hf("mlp.router.bias"), deq(&blk("ffn_gate_inp.bias"))?); // [E]

            // Experts. `poot_models::gpt_oss`'s safetensors path reads the native layout: one already-fused per-layer
            // `gate_up_proj` with gate/up interleaved even/odd along the last axis (`gate, up = gate_up[..., ::2],
            // gate_up[..., 1::2]`, module docs point 4). llama.cpp's GGUF conversion (`conversion/gpt_oss.py`
            // `GptOssModel.modify_tensors`, dequantized branch) de-interleaves that (`data_torch[:, ::2, :]`,
            // `data_torch[:, 1::2, :]`) into separate gate/up tensors written as the standard `ffn_gate_exps`/
            // `ffn_up_exps` 3D convention every other MoE arch uses. This arm therefore re-interleaves gate/up
            // (the stored rows interleaved, even rows gate and odd rows up, the mirror of the other arms'
            // concatenation) to match `poot_models::gpt_oss::gptoss_ffn`'s deinterleave-by-reshape. A plain
            // concatenation would give a shape-valid but wrong tensor (every other output channel's gate/up
            // weights swapped). A real gpt-oss checkpoint's experts are MXFP4 and stay packed: the interleave
            // moves whole stored rows.
            let gate = experts(&blk("ffn_gate_exps.weight"))?; // E x [I,H]
            let up = experts(&blk("ffn_up_exps.weight"))?; // E x [I,H]
            p.experts(
                hf("mlp.experts.gate_up_proj"), // NOT ".weight"-suffixed - matches the safetensors
                // path's raw-nn.Parameter naming (see poot_models::gpt_oss's module docs).
                fuse_gate_up(gate, up, true)?, // [E,H,2I] (even=gate,odd=up)
            )?;
            p.experts(
                hf("mlp.experts.down_proj"),
                experts(&blk("ffn_down_exps.weight"))?,
            )?; // [E,I,H]

            // Expert biases: gpt-oss has real per-expert biases on both projections (`poot_models::gpt_oss` module
            // docs; none of Mixtral/OlmoE/qwen3-moe/granitemoe do), required by the runtime
            // (`src/models/openai-moe.cpp`: `ffn_gate_exps_b`/`ffn_up_exps_b`/`ffn_down_exps_b`). The gate/up biases
            // are 2D `[E,I]` (no H axis, so no `transpose_experts`; interleave directly as for the weights); the down
            // bias is `[E,H]`, matching `poot_models::gpt_oss`'s `w_out_b` as-is.
            let gate_b = deq(&blk("ffn_gate_exps.bias"))?; // [E,I]
            let up_b = deq(&blk("ffn_up_exps.bias"))?; // [E,I]
            p.dense(
                hf("mlp.experts.gate_up_proj_bias"),
                interleave_experts_last(&gate_b, &up_b), // [E,I]+[E,I] -> [E,2I] interleaved
            );
            p.dense(
                hf("mlp.experts.down_proj_bias"),
                deq(&blk("ffn_down_exps.bias"))?,
            ); // [E,H]
        } else {
            if arch == "phi3" {
                // phi3 fuses gate|up into one `ffn_up` tensor (`[2*inter, in]`); split by rows.
                let gu = rd(&blk("ffn_up.weight"))?;
                p.projection(hf("mlp.gate_proj.weight"), gu.slice(0, cfg.inter)?)?;
                p.projection(
                    hf("mlp.up_proj.weight"),
                    gu.slice(cfg.inter, 2 * cfg.inter)?,
                )?;
            } else {
                p.projection(hf("mlp.gate_proj.weight"), rd(&blk("ffn_gate.weight"))?)?;
                p.projection(hf("mlp.up_proj.weight"), rd(&blk("ffn_up.weight"))?)?;
            }
            p.projection(hf("mlp.down_proj.weight"), rd(&blk("ffn_down.weight"))?)?;
        }
    }
    p.dense("model.norm.weight".to_string(), deq("output_norm.weight")?);
    // lm_head: a separate `output.weight` if present, else tied to the embedding (read `[hidden,vocab]`
    // by the tracer). A packed tied table is the embedding's own payload, shared, never copied.
    let lm_head = if g.tensors.contains_key("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    p.projection("lm_head.weight", rd(lm_head)?)?;
    let (cos, sin) = gguf_rope_tables(g, store, cfg, arch)?;
    p.dense("rope.cos".to_string(), cos);
    p.dense("rope.sin".to_string(), sin);
    if arch == "gemma3" {
        // Unlike the safetensors path, do not fold the gemma `(1 + weight)` here: llama.cpp already adds 1 to
        // every gemma `*norm.weight` at conversion time, so the GGUF tensors carry the folded value.
        //
        // Local (sliding-window) layers use a separate, lower RoPE base (gemma-3 default 10000; the GGUF omits the
        // key). The global table above came from gemma3.rope.freq_base (1e6).
        let (lc, ls) = rope_tables(cfg, 10000.0, None, None);
        p.dense("rope_local.cos".to_string(), lc);
        p.dense("rope_local.sin".to_string(), ls);
    }
    Ok(p.finish())
}
