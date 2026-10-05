use super::exact_trace::{
    Qwen4ExpExactPackedSources, Qwen4ExpExactTraceError, declare_routed_dense, exact_ple_embeddings,
};
use super::*;

/// How the whole-model tracer sources the routed FFN and the PLE embedding for one layer.
/// One scaffolding function serves both the dense synthetic path (spec 282) and Card 449 H0b's
/// exact packed path; only these two hooks differ.
#[derive(Clone, Copy)]
pub(crate) enum Qwen4ExpWholeModelMode<'a> {
    /// Dense fused MoE tensors + step-input `{layer}.ple.ngram_ids` + dense n-gram table gather.
    Dense,
    /// Card 362 packed FFN + option-D in-graph n-gram ids (history State) + Card 363 sharded E4M3 PLE.
    #[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-571"))]
    Exact(&'a Qwen4ExpExactPackedSources),
}

/// Shared whole-model prefill scaffolding (spec 282): embed, per-layer hybrid stack, mixer, head.
/// `mode` selects only the FFN and PLE sourcing; attention/GDN/HC and the constant names are one
/// definition. Returns `Result` so the exact path can surface packed-FFN errors; the dense path
/// cannot fail and unwraps.
///
/// `cache_len` is the fixed KV/idx cache capacity for the exact path's carried state (Card 449 H0b): `None` keeps the dense L-shaped caches with no idx/PLE-conv pairs; `Some(cap)` sizes
/// QSA `k`/`v`/`idx` caches at `cap` (>= `seq_len`, multiple of the compress ratio) and emits the
/// PLE conv-cache pair so prefill's `graph.state` layout matches decode for prefill-then-decode
/// chaining. Attention still runs at `seq_len` (real prompt length; no token padding).
pub(crate) fn trace_qwen38_prefill_impl(
    mcfg: &Qwen4ExpModelConfig,
    seq_len: usize,
    mode: Qwen4ExpWholeModelMode<'_>,
    cache_len: Option<usize>,
) -> Result<Graph, Qwen4ExpExactTraceError> {
    assert_eq!(
        seq_len % mcfg.qcfg.index_compress_ratio,
        0,
        "seq_len must be an exact multiple of index_compress_ratio (spec 282)"
    );
    if let Qwen4ExpWholeModelMode::Exact(sources) = mode {
        if sources.layers.len() != mcfg.layer_is_full.len() {
            return Err(Qwen4ExpExactTraceError::LayerCount {
                expected: mcfg.layer_is_full.len(),
                actual: sources.layers.len(),
            });
        }
        let cap = cache_len.unwrap_or(seq_len);
        assert!(
            cap >= seq_len && cap.is_multiple_of(mcfg.qcfg.index_compress_ratio),
            "exact prefill cache_len must be >= seq_len and a multiple of index_compress_ratio"
        );
    }
    let kv_len = match mode {
        Qwen4ExpWholeModelMode::Dense => seq_len,
        Qwen4ExpWholeModelMode::Exact(_) => cache_len.unwrap_or(seq_len),
    };
    let exact_layout = match mode {
        Qwen4ExpWholeModelMode::Dense => false,
        Qwen4ExpWholeModelMode::Exact(_) => true,
    };
    let b = Builder::new();
    let h = mcfg.cfg.hidden;
    let l = seq_len;
    let c = mcfg.qcfg.index_compress_ratio;
    let nb = l / c;
    let rot = mcfg.cfg.rotary_dim;
    let max_blocks = mcfg.max_pos / c;
    let (hc_count, hc_lowrank) = (mcfg.hc_count, mcfg.hc_lowrank);

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![mcfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens);
    let emb = b.reshape(emb, vec![1, l, h]);
    let mut x = qwen38_widen_embedding(&b, emb, hc_count);

    let cos = b.constant("rope.cos", TensorType::f32(vec![mcfg.max_pos, rot]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![mcfg.max_pos, rot]));
    let block_cos = b.constant("qsa.block_rope.cos", TensorType::f32(vec![max_blocks, rot]));
    let block_sin = b.constant("qsa.block_rope.sin", TensorType::f32(vec![max_blocks, rot]));
    let causal_mask = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));
    let block_eligible = b.slot_named(
        Slot::Activation,
        "qsa.block_eligible",
        TensorType::f32(vec![1, 1, l, nb]),
    );
    let tail_mask = b.slot_named(
        Slot::Activation,
        "qsa.tail_mask",
        TensorType::f32(vec![1, 1, l, l]),
    );
    let chunk = mcfg.chunk;
    let tril_iota = b.iota(chunk);
    let tril_rows = b.broadcast(b.reshape(tril_iota, vec![chunk, 1]), vec![chunk, chunk]);
    let tril_cols = b.broadcast(b.reshape(tril_iota, vec![1, chunk]), vec![chunk, chunk]);
    let tril_next = b.binary_scalar(BinOp::Add, tril_cols, Scalar::F32(1.0));
    let tril_incl = b.reshape(
        b.binary(BinOp::Ge, tril_rows, tril_cols),
        vec![1, 1, chunk, chunk],
    );
    let tril_strict = b.reshape(
        b.binary(BinOp::Ge, tril_rows, tril_next),
        vec![1, 1, chunk, chunk],
    );

    let (gk, gv, ghd, gck) = (
        mcfg.gdn.num_k_heads,
        mcfg.gdn.num_v_heads,
        mcfg.gdn.head_dim,
        mcfg.gdn.conv_k,
    );
    let key_dim = gk * ghd;
    let value_dim = gv * ghd;
    let conv_dim = 2 * key_dim + value_dim;

    let mut state: Vec<(Traced, Traced)> = Vec::new();
    for (li, &is_full) in mcfg.layer_is_full.iter().enumerate() {
        let _layer_scope = b.layer_scope(li);
        let p = format!("layers.{li}");
        // N-gram/PLE injection at the front of the layer, before `attn_hyper_connection`.
        if let Some(ple) = mcfg.ple.as_ref()
            && let Some(ple_layer_index) = ple.ple_layer_index(li)
        {
            let tables = qwen38_ngram_tables(ple, ple_layer_index);
            match mode {
                Qwen4ExpWholeModelMode::Dense => {
                    let ngram_ids = b.slot_named(
                        Slot::Activation,
                        &format!("{p}.ple.ngram_ids"),
                        TensorType::new(vec![l, ple.ngram_heads()], DType::I32),
                    );
                    let ple_out = qwen38_ple_prefill(
                        &b, x, ngram_ids, &p, ple, &tables, hc_count, h, mcfg.eps,
                    );
                    x = b.binary(BinOp::Add, x, ple_out);
                }
                Qwen4ExpWholeModelMode::Exact(sources) => {
                    let history = b.state_input(
                        &format!("{p}.ple.ngram_history"),
                        TensorType::new(vec![1, ple.context_len()], DType::I32),
                        StateRole::Recurrent,
                    );
                    let (ngram_ids, history_out, _token_guard) =
                        qwen38_ngram_ids_graph(&b, tokens, history, ple, &tables);
                    state.push((history, history_out));
                    let emb_ngram = exact_ple_embeddings(&b, ngram_ids, sources, ple);
                    if exact_layout {
                        let (ple_out, cc_out) = qwen38_ple_prefill_with_embeddings_and_cache(
                            &b, x, emb_ngram, &p, ple, hc_count, h, mcfg.eps,
                        );
                        let cc_in = b.state_input(
                            &format!("{p}.ple.conv_cache"),
                            TensorType::f32(vec![1, ple.conv_state_len(), hc_count * h]),
                            StateRole::Recurrent,
                        );
                        state.push((cc_in, cc_out));
                        x = b.binary(BinOp::Add, x, ple_out);
                    } else {
                        let ple_out = qwen38_ple_prefill_with_embeddings(
                            &b, x, emb_ngram, &p, ple, hc_count, h, mcfg.eps,
                        );
                        x = b.binary(BinOp::Add, x, ple_out);
                    }
                }
            }
        }
        let (mixed_input, inj_w) = qwen38_gated_residual(
            &b,
            x,
            &format!("{p}.attn_hyper_connection"),
            hc_count,
            h,
            hc_lowrank,
            mcfg.eps,
        );
        let normed = mixed_input;

        let mix = if is_full {
            let kc_in = b.state_input(
                &format!("{p}.k_cache"),
                TensorType::f32(vec![1, mcfg.cfg.n_kv_heads, kv_len, mcfg.cfg.head_dim]),
                StateRole::Recurrent,
            );
            let vc_in = b.state_input(
                &format!("{p}.v_cache"),
                TensorType::f32(vec![1, mcfg.cfg.n_kv_heads, kv_len, mcfg.cfg.head_dim]),
                StateRole::Recurrent,
            );
            if exact_layout {
                let idxc_in = b.state_input(
                    &format!("{p}.idx_k_cache"),
                    TensorType::f32(vec![1, 1, kv_len, mcfg.qcfg.index_head_dim]),
                    StateRole::Recurrent,
                );
                let (out, kc_out, vc_out, raw_k) = qsa_attention_prefill_layer_with_raw_k(
                    &b,
                    normed,
                    &p,
                    mcfg.cfg,
                    mcfg.qcfg,
                    l,
                    cos,
                    sin,
                    block_cos,
                    block_sin,
                    causal_mask,
                    block_eligible,
                    tail_mask,
                    kc_in,
                    vc_in,
                );
                // Write the prompt's raw keys into the capacity-sized idx cache at slot 0 (XLA
                // DynamicUpdateSlice: update extent L fits capacity kv_len).
                let idxc_out = b.dynamic_update_slice(idxc_in, raw_k, 0, 2);
                state.push((kc_in, kc_out));
                state.push((vc_in, vc_out));
                state.push((idxc_in, idxc_out));
                out
            } else {
                let (out, kc_out, vc_out) = qsa_attention_prefill_layer(
                    &b,
                    normed,
                    &p,
                    mcfg.cfg,
                    mcfg.qcfg,
                    l,
                    cos,
                    sin,
                    block_cos,
                    block_sin,
                    causal_mask,
                    block_eligible,
                    tail_mask,
                    kc_in,
                    vc_in,
                );
                state.push((kc_in, kc_out));
                state.push((vc_in, vc_out));
                out
            }
        } else {
            let w_qkv = b.constant(
                &format!("{p}.linear_attn.in_proj_qkv.weight"),
                TensorType::f32(vec![h, conv_dim]),
            );
            let w_gate = b.constant(
                &format!("{p}.linear_attn.in_proj_z.weight"),
                TensorType::f32(vec![h, value_dim]),
            );
            let w_conv = b.constant(
                &format!("{p}.linear_attn.conv1d.weight"),
                TensorType::f32(vec![gck, conv_dim]),
            );
            let w_beta = b.constant(
                &format!("{p}.linear_attn.in_proj_b.weight"),
                TensorType::f32(vec![h, gv]),
            );
            let w_alpha = b.constant(
                &format!("{p}.linear_attn.in_proj_a.weight"),
                TensorType::f32(vec![h, gv]),
            );
            let dt_bias = b.constant(
                &format!("{p}.linear_attn.dt_bias"),
                TensorType::f32(vec![gv]),
            );
            let ssm_a = b.constant(&format!("{p}.linear_attn.ssm_a"), TensorType::f32(vec![gv]));
            let norm_w = b.constant(
                &format!("{p}.linear_attn.norm.weight"),
                TensorType::f32(vec![ghd]),
            );
            let w_out = b.constant(
                &format!("{p}.linear_attn.out_proj.weight"),
                TensorType::f32(vec![value_dim, h]),
            );
            let s_in = b.state_input(
                &format!("{p}.ssm_state"),
                TensorType::f32(vec![1, gv, ghd, ghd]),
                StateRole::Recurrent,
            );
            let (out, cc_out, s_out) = qwen3next_gdn_prefill_block(
                &b,
                normed,
                w_qkv,
                w_gate,
                w_conv,
                w_beta,
                w_alpha,
                dt_bias,
                ssm_a,
                norm_w,
                w_out,
                s_in,
                tril_incl,
                tril_strict,
                gk,
                gv,
                ghd,
                gck,
                mcfg.chunk,
                mcfg.eps,
                GDN_HEAD_ORDER,
            );
            let cc_ph = b.state_input(
                &format!("{p}.conv_cache"),
                TensorType::f32(vec![1, gck - 1, conv_dim]),
                StateRole::Recurrent,
            );
            state.push((cc_ph, cc_out));
            state.push((s_in, s_out));
            out
        };
        x = qwen38_gated_residual_inject(&b, x, mix, inj_w);

        let (mixed_input2, inj_w2) = qwen38_gated_residual(
            &b,
            x,
            &format!("{p}.mlp_hyper_connection"),
            hc_count,
            h,
            hc_lowrank,
            mcfg.eps,
        );
        let ffn_out = match mode {
            Qwen4ExpWholeModelMode::Dense => qwen38_moe_ffn(
                &b,
                mixed_input2,
                &p,
                h,
                mcfg.moe_n_experts,
                mcfg.moe_top_k,
                mcfg.moe_inter,
                mcfg.ffn_inter,
            ),
            Qwen4ExpWholeModelMode::Exact(sources) => {
                let packed = &sources.layers[li];
                let dense = declare_routed_dense(&b, packed);
                qwen38_packed_grouped_moe_ffn(
                    &b,
                    mixed_input2,
                    dense[0],
                    dense[1],
                    dense[2],
                    dense[3],
                    dense[4],
                    mcfg.moe_top_k,
                    &packed.tables,
                )?
            }
        };
        x = qwen38_gated_residual_inject(&b, x, ffn_out, inj_w2);
    }

    let xn = qwen38_gated_residual_mixer(
        &b,
        x,
        "hyper_connection_mixer",
        hc_count,
        h,
        hc_lowrank,
        mcfg.eps,
    );
    let last = b.slice(xn, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, mcfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    Ok(b.finish_with_state(logits, &state))
}

/// Shared whole-model decode scaffolding (spec 282): fixed-KV-capacity decode, one token per call.
/// See [`trace_qwen38_prefill_impl`] for the mode split.
pub(crate) fn trace_qwen38_decode_impl(
    mcfg: &Qwen4ExpModelConfig,
    cap: usize,
    mode: Qwen4ExpWholeModelMode<'_>,
) -> Result<Graph, Qwen4ExpExactTraceError> {
    let c = mcfg.qcfg.index_compress_ratio;
    assert_eq!(
        cap % c,
        0,
        "cap must be an exact multiple of index_compress_ratio (spec 282)"
    );
    if let Qwen4ExpWholeModelMode::Exact(sources) = mode
        && sources.layers.len() != mcfg.layer_is_full.len()
    {
        return Err(Qwen4ExpExactTraceError::LayerCount {
            expected: mcfg.layer_is_full.len(),
            actual: sources.layers.len(),
        });
    }
    let b = Builder::new();
    let h = mcfg.cfg.hidden;
    let rot = mcfg.cfg.rotary_dim;
    let nb = cap / c;
    let max_blocks = mcfg.max_pos / c;
    let (hc_count, hc_lowrank) = (mcfg.hc_count, mcfg.hc_lowrank);

    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let _seq_len = b.slot(Slot::SeqLen, TensorType::scalar(DType::I32));
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![cap]));
    let mask = b.reshape(mask, vec![1, 1, 1, cap]);

    let iota_full = b.iota(mcfg.max_pos);
    let pos_f = b.gather_scalar(iota_full, 0, pos_slot);

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![mcfg.vocab, h]),
    );
    let x0 = b.gather_scalar(embed, 0, token);
    let x0 = b.reshape(x0, vec![1, 1, h]);
    let mut x = qwen38_widen_embedding(&b, x0, hc_count);

    let cos = b.constant("rope.cos", TensorType::f32(vec![mcfg.max_pos, rot]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![mcfg.max_pos, rot]));
    let block_cos = b.constant("qsa.block_rope.cos", TensorType::f32(vec![max_blocks, rot]));
    let block_sin = b.constant("qsa.block_rope.sin", TensorType::f32(vec![max_blocks, rot]));
    let block_positions = b.slot_named(
        Slot::Activation,
        "qsa.block_positions",
        TensorType::f32(vec![nb]),
    );

    let (gk, gv, ghd, gck) = (
        mcfg.gdn.num_k_heads,
        mcfg.gdn.num_v_heads,
        mcfg.gdn.head_dim,
        mcfg.gdn.conv_k,
    );
    let key_dim = gk * ghd;
    let value_dim = gv * ghd;
    let conv_dim = 2 * key_dim + value_dim;

    let mut state: Vec<(Traced, Traced)> = Vec::new();
    for (li, &is_full) in mcfg.layer_is_full.iter().enumerate() {
        let _layer_scope = b.layer_scope(li);
        let p = format!("layers.{li}");
        if let Some(ple) = mcfg.ple.as_ref()
            && let Some(ple_layer_index) = ple.ple_layer_index(li)
        {
            let tables = qwen38_ngram_tables(ple, ple_layer_index);
            match mode {
                Qwen4ExpWholeModelMode::Dense => {
                    let ngram_ids = b.slot_named(
                        Slot::Activation,
                        &format!("{p}.ple.ngram_ids"),
                        TensorType::new(vec![1, ple.ngram_heads()], DType::I32),
                    );
                    let cc_in = b.state_input(
                        &format!("{p}.ple.conv_cache"),
                        TensorType::f32(vec![1, ple.conv_state_len(), hc_count * h]),
                        StateRole::Recurrent,
                    );
                    let (ple_out, cc_out) = qwen38_ple_decode(
                        &b, x, ngram_ids, cc_in, &p, ple, &tables, hc_count, h, mcfg.eps,
                    );
                    state.push((cc_in, cc_out));
                    x = b.binary(BinOp::Add, x, ple_out);
                }
                Qwen4ExpWholeModelMode::Exact(sources) => {
                    let token_ids = b.reshape(token, vec![1]);
                    let history = b.state_input(
                        &format!("{p}.ple.ngram_history"),
                        TensorType::new(vec![1, ple.context_len()], DType::I32),
                        StateRole::Recurrent,
                    );
                    let (ngram_ids, history_out, _token_guard) =
                        qwen38_ngram_ids_graph(&b, token_ids, history, ple, &tables);
                    state.push((history, history_out));
                    let emb_ngram = exact_ple_embeddings(&b, ngram_ids, sources, ple);
                    let cc_in = b.state_input(
                        &format!("{p}.ple.conv_cache"),
                        TensorType::f32(vec![1, ple.conv_state_len(), hc_count * h]),
                        StateRole::Recurrent,
                    );
                    let (ple_out, cc_out) = qwen38_ple_decode_with_embeddings(
                        &b, x, emb_ngram, cc_in, &p, ple, hc_count, h, mcfg.eps,
                    );
                    state.push((cc_in, cc_out));
                    x = b.binary(BinOp::Add, x, ple_out);
                }
            }
        }
        let (mixed_input, inj_w) = qwen38_gated_residual(
            &b,
            x,
            &format!("{p}.attn_hyper_connection"),
            hc_count,
            h,
            hc_lowrank,
            mcfg.eps,
        );
        let normed = mixed_input;

        let mix = if is_full {
            let kc_in = b.state_input(
                &format!("{p}.k_cache"),
                TensorType::f32(vec![1, mcfg.cfg.n_kv_heads, cap, mcfg.cfg.head_dim]),
                StateRole::Recurrent,
            );
            let vc_in = b.state_input(
                &format!("{p}.v_cache"),
                TensorType::f32(vec![1, mcfg.cfg.n_kv_heads, cap, mcfg.cfg.head_dim]),
                StateRole::Recurrent,
            );
            let idxc_in = b.state_input(
                &format!("{p}.idx_k_cache"),
                TensorType::f32(vec![1, 1, cap, mcfg.qcfg.index_head_dim]),
                StateRole::Recurrent,
            );
            let (out, kc_out, vc_out, idxc_out) = qsa_attention_decode_layer(
                &b,
                normed,
                &p,
                mcfg.cfg,
                mcfg.qcfg,
                cap,
                pos_slot,
                pos_f,
                cos,
                sin,
                block_cos,
                block_sin,
                block_positions,
                mask,
                kc_in,
                vc_in,
                idxc_in,
            );
            state.push((kc_in, kc_out));
            state.push((vc_in, vc_out));
            state.push((idxc_in, idxc_out));
            out
        } else {
            let w_qkv = b.constant(
                &format!("{p}.linear_attn.in_proj_qkv.weight"),
                TensorType::f32(vec![h, conv_dim]),
            );
            let w_gate = b.constant(
                &format!("{p}.linear_attn.in_proj_z.weight"),
                TensorType::f32(vec![h, value_dim]),
            );
            let w_conv = b.constant(
                &format!("{p}.linear_attn.conv1d.weight"),
                TensorType::f32(vec![gck, conv_dim]),
            );
            let w_beta = b.constant(
                &format!("{p}.linear_attn.in_proj_b.weight"),
                TensorType::f32(vec![h, gv]),
            );
            let w_alpha = b.constant(
                &format!("{p}.linear_attn.in_proj_a.weight"),
                TensorType::f32(vec![h, gv]),
            );
            let dt_bias = b.constant(
                &format!("{p}.linear_attn.dt_bias"),
                TensorType::f32(vec![gv]),
            );
            let ssm_a = b.constant(&format!("{p}.linear_attn.ssm_a"), TensorType::f32(vec![gv]));
            let norm_w = b.constant(
                &format!("{p}.linear_attn.norm.weight"),
                TensorType::f32(vec![ghd]),
            );
            let w_out = b.constant(
                &format!("{p}.linear_attn.out_proj.weight"),
                TensorType::f32(vec![value_dim, h]),
            );
            let cc_in = b.state_input(
                &format!("{p}.conv_cache"),
                TensorType::f32(vec![1, gck - 1, conv_dim]),
                StateRole::Recurrent,
            );
            let s_in = b.state_input(
                &format!("{p}.ssm_state"),
                TensorType::f32(vec![1, gv, ghd, ghd]),
                StateRole::Recurrent,
            );
            let (out, cc_out, s_out) = qwen3next_gdn_block(
                &b,
                normed,
                w_qkv,
                w_gate,
                w_conv,
                w_beta,
                w_alpha,
                dt_bias,
                ssm_a,
                norm_w,
                w_out,
                cc_in,
                s_in,
                gk,
                gv,
                ghd,
                gck,
                mcfg.eps,
                GDN_HEAD_ORDER,
            );
            state.push((cc_in, cc_out));
            state.push((s_in, s_out));
            out
        };
        x = qwen38_gated_residual_inject(&b, x, mix, inj_w);

        let (mixed_input2, inj_w2) = qwen38_gated_residual(
            &b,
            x,
            &format!("{p}.mlp_hyper_connection"),
            hc_count,
            h,
            hc_lowrank,
            mcfg.eps,
        );
        let ffn_out = match mode {
            Qwen4ExpWholeModelMode::Dense => qwen38_moe_ffn(
                &b,
                mixed_input2,
                &p,
                h,
                mcfg.moe_n_experts,
                mcfg.moe_top_k,
                mcfg.moe_inter,
                mcfg.ffn_inter,
            ),
            Qwen4ExpWholeModelMode::Exact(sources) => {
                let packed = &sources.layers[li];
                let dense = declare_routed_dense(&b, packed);
                qwen38_packed_indexed_moe_ffn(
                    &b,
                    mixed_input2,
                    dense[0],
                    dense[1],
                    dense[2],
                    dense[3],
                    dense[4],
                    mcfg.moe_top_k,
                    &packed.tables,
                )?
            }
        };
        x = qwen38_gated_residual_inject(&b, x, ffn_out, inj_w2);
    }

    let xn = qwen38_gated_residual_mixer(
        &b,
        x,
        "hyper_connection_mixer",
        hc_count,
        h,
        hc_lowrank,
        mcfg.eps,
    );
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, mcfg.vocab]));
    let logits = linear(&b, xn, lm_head, None);
    Ok(b.finish_with_state(logits, &state))
}

/// Whole-model prefill trace (spec 282): embed (widened `hc_count`-fold, [`qwen38_widen_embedding`]),
/// then a per-layer hybrid stack of `linear_attention` (GDN, [`qwen3next_gdn_prefill_block`] semantics in
/// HF head order) and `full_attention` (QSA, [`qsa_attention_prefill_layer`]) layers, each sub-layer
/// wrapped in the Hyper-Connections [`qwen38_gated_residual`]/[`qwen38_gated_residual_inject`] pair
/// (attention/GDN, then MLP) with the routed-MoE + shared-expert FFN ([`qwen38_moe_ffn`]), then the
/// [`qwen38_gated_residual_mixer`] collapse to one stream and `lm_head`. `seq_len` must be an exact
/// multiple of `mcfg.qcfg.index_compress_ratio` (QSA's block-pooling reshape, `qsa_combined_mask`'s
/// `nb * c == l` assert). `mcfg.chunk` is not a length constraint: the GDN prefill block's chunked
/// recurrence (`gdn_prefill_chunked`, FR-005) zero-pads `seq_len` to whole `chunk` tiles and truncates
/// back, as in the dense Qwen3-Next prefill tracer (exercised at non-multiple-of-16 lengths, e.g.
/// `poot-llm`'s `card188_prefill_plan_classify` test at `L=40`).
///
/// # No plain per-block pre-norm
///
/// The real checkpoint (`Qwen4ExpTextDecoderLayer.forward` in `modeling_qwen4_exp.py`) has no plain
/// `input_layernorm`/`post_attention_layernorm` tensor: normalization is folded into Hyper-Connections'
/// `hc_norm` (`Qwen4ExpTextGatedResidual.hc_norm`, a grouped RMSNorm over the `hc_count * hidden`
/// multi-stream state; see [`qwen38_grouped_rmsnorm`]). So [`qwen38_gated_residual`]'s `hc_norm` step is
/// the pre-attention/pre-mlp normalization, and there is no final norm before `lm_head` either:
/// `Qwen4ExpTextModel.forward` feeds `hyper_connection_mixer`'s output straight to `lm_head`, with no
/// `model.norm` module.
///
/// # RMSNorm `1 + weight` convention (a loader concern)
///
/// `Qwen4ExpTextRMSNorm` (used for `q_norm`/`k_norm`/the indexer's `q_layernorm`/`k_layernorm`/every
/// `hc_norm`) computes `x_normalized * (1.0 + weight)`, zero-initialized, not the plain
/// `x_normalized * weight` of [`qwen38_grouped_rmsnorm`]/`poot_graph_ir::ops::rmsnorm`. Weight
/// materialization applies the `+1.0` once, so the tracer reads it from the weight.
pub fn trace_qwen38_prefill(mcfg: &Qwen4ExpModelConfig, seq_len: usize) -> Graph {
    trace_qwen38_prefill_impl(mcfg, seq_len, Qwen4ExpWholeModelMode::Dense, None)
        .expect("dense whole-model prefill has no failure mode")
}

/// Whole-model fixed-KV-capacity decode trace (spec 282), the follow-up to [`trace_qwen38_prefill`]. Same
/// shape: embed a single token, widened `hc_count`-fold ([`qwen38_widen_embedding`]), then a per-layer
/// hybrid stack of `linear_attention` (GDN, [`qwen3next_gdn_block`] semantics in HF head order) and
/// `full_attention` (QSA, [`qsa_attention_decode_layer`]) layers, each sub-layer wrapped in the same
/// [`qwen38_gated_residual`]/[`qwen38_gated_residual_inject`] pair as prefill, the same routed-MoE +
/// shared-expert FFN, [`qwen38_gated_residual_mixer`] collapse, and `lm_head`. Every mixer call is the
/// decode function (per-layer state threaded via `Builder::state_input`/`dynamic_update_slice_dyn`) and
/// `x` is a single token (`[1,1,H]`), so one traced graph serves every step (`Slot::Pos`/`Slot::Mask`
/// vary per call, not the graph). Unlike [`trace_qwen38_prefill`], this has no `seq_len % chunk == 0`
/// re-prefill assert (the growing-length constraint [`crate::deepseek4`]'s `V4LayerKind` schedule
/// describes), since `cap` is chosen once by the caller.
///
/// Each QSA layer carries a third state pair beyond `k_cache`/`v_cache` (the indexer's raw-key history,
/// see [`qsa_attention_decode_layer`]) that prefill's QSA layers never declare (prefill pools block keys
/// from the in-hand prompt), so `Graph::state` has `3 * (#QSA layers) + 2 * (#GDN layers)` entries.
///
/// `cap` must be an exact multiple of `mcfg.qcfg.index_compress_ratio` (QSA's block tiling, as
/// [`trace_qwen38_prefill`]'s `seq_len`). `mcfg.chunk` plays no role: GDN's decode step
/// ([`qwen3next_gdn_block`]) is a plain per-token recurrence, not the chunked prefill.
pub fn trace_qwen38_decode(mcfg: &Qwen4ExpModelConfig, cap: usize) -> Graph {
    trace_qwen38_decode_impl(mcfg, cap, Qwen4ExpWholeModelMode::Dense)
        .expect("dense whole-model decode has no failure mode")
}
