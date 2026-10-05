use super::*;

/// Trace one Qwen3-Next decode token into a flat primitive graph (card 135c integration milestone).
///
/// Composes the four merged block compositions into the full model, following the top-level decode
/// graph in section 2:
///
/// ```text
/// x = embed[token]
/// for li in 0..n_layers:
///     inpSA = x
///     n     = rmsnorm(x, attn_norm[li])                 # input norm
///     mix   = is_attn(li) ? gated_attention(n, ...) : gdn_block(n, ...)
///     x     = inpSA + mix                                # attn residual
///     ffres = x
///     f     = rmsnorm(x, post_attention_norm[li])       # pre-FFN norm
///     x     = ffres + moe_ffn(f, ...)                    # FFN residual
/// x      = rmsnorm(x, model.norm)
/// logits = x @ lm_head
/// ```
///
/// Both norms are pre-norms (RMSNorm before the sublayer, residual added after). The heterogeneous
/// per-layer state is threaded per layer: each full-attention layer carries a
/// `(k_cache, v_cache)` pair; each GDN layer carries a `(conv_cache, ssm_state)` pair. All pairs are
/// registered with `finish_with_state` so the runtime feeds each layer's updated buffers back on the
/// next step.
///
/// `pos` is the new token's position (the prior context length); the graph is pos-specialized (the full-attention KV scatter index and valid-prefix slice `[0..=pos]` are baked). `cap` is
/// the fixed KV capacity (`cap > pos`). RoPE uses the `Pos` slot for its table gather. All weights are f32
/// constants referenced by name; a loader binds them.
pub fn qwen3next_decode_trace(cfg: &Qwen3NextConfig, pos: usize, cap: usize) -> Graph {
    assert!(cap > pos, "cache capacity {cap} must exceed pos {pos}");
    let b = Builder::new();
    let h = cfg.hidden;

    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32));

    // Partial NeoX RoPE tables (full-attention layers only).
    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let x0 = b.gather_scalar(embed, 0, token); // [hidden]
    let mut x = b.reshape(x0, vec![1, 1, h]); // [1,1,hidden]

    // (state_in, state_out) pairs, heterogeneous per layer, accumulated across all layers.
    let mut state: Vec<(Traced, Traced)> = Vec::new();

    for li in 0..cfg.n_layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        // 1. Input RMSNorm (pre-norm). Named attn_norm in the checkpoint for both layer types.
        let attn_norm = b.constant(&p("attn_norm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, attn_norm, cfg.eps);

        // 2. Mixer: full-attention or Gated DeltaNet, per the layer schedule.
        let mixer_out = if cfg.is_attn_layer(li) {
            let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
            // attn_q carries query|gate interleaved per head: out = nh * 2 * head_dim.
            let wq = b.constant(&p("attn_q.weight"), TensorType::f32(vec![h, nh * 2 * hd]));
            let wk = b.constant(&p("attn_k.weight"), TensorType::f32(vec![h, nkv * hd]));
            let wv = b.constant(&p("attn_v.weight"), TensorType::f32(vec![h, nkv * hd]));
            let wo = b.constant(&p("attn_output.weight"), TensorType::f32(vec![nh * hd, h]));
            let qn = b.constant(&p("attn_q_norm.weight"), TensorType::f32(vec![hd]));
            let kn = b.constant(&p("attn_k_norm.weight"), TensorType::f32(vec![hd]));
            let kc = b.state_input(
                &p("kv.k_cache"),
                TensorType::f32(vec![1, nkv, cap, hd]),
                StateRole::Recurrent,
            );
            let vc = b.state_input(
                &p("kv.v_cache"),
                TensorType::f32(vec![1, nkv, cap, hd]),
                StateRole::Recurrent,
            );
            let (out, kc_out, vc_out) = qwen3next_gated_attention(
                &b, normed, wq, wk, wv, wo, qn, kn, cos, sin, pos_slot, kc, vc, nh, nkv, hd, pos,
                cfg.eps,
            );
            state.push((kc, kc_out));
            state.push((vc, vc_out));
            out
        } else {
            let (hk, hv, hd, ck) = (
                cfg.gdn_num_k_heads,
                cfg.gdn_num_v_heads,
                cfg.gdn_head_dim,
                cfg.conv_k,
            );
            let key_dim = hk * hd;
            let value_dim = hv * hd;
            let conv_dim = 2 * key_dim + value_dim;
            let w_qkv = b.constant(&p("attn_qkv.weight"), TensorType::f32(vec![h, conv_dim]));
            let w_gate = b.constant(&p("attn_gate.weight"), TensorType::f32(vec![h, value_dim]));
            let w_conv = b.constant(&p("ssm_conv1d.weight"), TensorType::f32(vec![ck, conv_dim]));
            let w_beta = b.constant(&p("ssm_beta.weight"), TensorType::f32(vec![h, hv]));
            let w_alpha = b.constant(&p("ssm_alpha.weight"), TensorType::f32(vec![h, hv]));
            let dt_bias = b.constant(&p("ssm_dt.bias"), TensorType::f32(vec![hv]));
            let ssm_a = b.constant(&p("ssm_a"), TensorType::f32(vec![hv]));
            let norm_w = b.constant(&p("ssm_norm.weight"), TensorType::f32(vec![hd]));
            let w_out = b.constant(&p("ssm_out.weight"), TensorType::f32(vec![value_dim, h]));
            let cc = b.state_input(
                &p("gdn.conv_cache"),
                TensorType::f32(vec![1, ck - 1, conv_dim]),
                StateRole::Recurrent,
            );
            let si = b.state_input(
                &p("gdn.ssm_state"),
                TensorType::f32(vec![1, hv, hd, hd]),
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
                cc,
                si,
                hk,
                hv,
                hd,
                ck,
                cfg.eps,
                GdnHeadOrder::Tiled,
            );
            state.push((cc, cc_out));
            state.push((si, s_out));
            out
        };

        // 3. Attention residual.
        x = b.binary(BinOp::Add, x, mixer_out);

        // 4. Pre-FFN RMSNorm (named post_attention_norm, but applied to the FFN input).
        let post_norm = b.constant(&p("post_attention_norm.weight"), TensorType::f32(vec![h]));
        let ff_in = rmsnorm(&b, x, post_norm, cfg.eps);

        // 5. MoE FFN + shared expert.
        let (e, k, i, si_) = (cfg.n_experts, cfg.top_k, cfg.expert_inter, cfg.shared_inter);
        let router = b.constant(&p("ffn_gate_inp.weight"), TensorType::f32(vec![h, e]));
        let w_in = b.constant(&p("ffn.w_in"), TensorType::f32(vec![e, h, 2 * i]));
        let w_out_moe = b.constant(&p("ffn.w_out"), TensorType::f32(vec![e, i, h]));
        let sg = b.constant(&p("ffn_gate_shexp.weight"), TensorType::f32(vec![h, si_]));
        let su = b.constant(&p("ffn_up_shexp.weight"), TensorType::f32(vec![h, si_]));
        let sd = b.constant(&p("ffn_down_shexp.weight"), TensorType::f32(vec![si_, h]));
        let sgi = b.constant(&p("ffn_gate_inp_shexp.weight"), TensorType::f32(vec![h, 1]));
        let ffn_out =
            qwen3next_moe_ffn(&b, ff_in, router, w_in, w_out_moe, sg, su, sd, sgi, e, k, i);

        // 6. FFN residual.
        x = b.binary(BinOp::Add, x, ffn_out);
    }

    // Final RMSNorm + lm_head.
    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None); // [1,1,vocab]

    b.finish_with_state(logits, &state)
}

/// Card 188 Increment 4 (spec `specs/188-qwen3next-batching/spec.md`): the batched continuous-decode
/// counterpart of [`qwen3next_decode_trace`]. `batch` concurrent sequences share one shared-pool attention KV cache
/// (the `scatter_shared_pool`/`gather_shared_pool` mechanism of `poot-models/src/qwen2.rs` and
/// `gemma4_decode_trace_batched_shared_pool`) and one GDN state pool ([`qwen3next_gdn_decode_batched_pool`]). The
/// grouped MoE FFN (`ops::moe`/`moe_grouped_prep`) is reused unmodified with `batch` as the row axis (FR-006).
///
/// Per-layer structure mirrors [`qwen3next_decode_trace`] (same `is_attn_layer` split, weight names, RMSNorm/residual
/// order), generalized to `[batch, 1, H]` rows:
///
/// - **Attention layers**: same projections/QK-norm/interleaved-Q|gate-split/output-gate as
///   [`qwen3next_gated_attention`], but K/Q RoPE uses [`poot_graph_ir::ops::rope_batched`] (per-row `pos_slot`),
///   the KV cache is a shared pool addressed by the `[batch,cap]` `Slot::SlotMap` via
///   `scatter_shared_pool`/`gather_shared_pool`, and attention is [`poot_graph_ir::ops::attention_masked`] (an
///   explicit `[batch,1,1,cap]` additive mask). Unwritten or inactive pool positions are masked out regardless of
///   their stale contents, as in [`crate::gemma4::gemma4_decode_trace_batched_shared_pool`].
/// - **GDN layers**: [`qwen3next_gdn_decode_batched_pool`] end to end (gather from the GDN pool at each row's
///   `gdn_slot_map` entry, run the batch-generic [`qwen3next_gdn_block`], scatter back). Unlike attention, a
///   reused GDN pool row is not protected by any mask, so a newly admitted sequence's row must be zeroed (or
///   seeded, e.g. by [`qwen3next_gdn_prefill_into_slot`]) before its first batched-decode step.
/// - **MoE FFN**: `ff_in [batch,1,H]` is reshaped to `[1,batch,H]` so `batch` is the row axis `L` that
///   [`qwen3next_moe_ffn`]/`ops::moe` dispatch on (`L=1` -> `moe_sparse`, `L>1` -> `moe_grouped`), then reshaped
///   back. No new MoE code (FR-006).
///
/// The graph declares two slot-map kinds: `Slot::SlotMap` `[batch,cap]` i32 for the attention pool and
/// `Slot::GdnSlotMap` `[batch]` i32 for the GDN pool row map. Sharing one kind is safe for `poot-eval`'s per-id CPU
/// binder but not for a binder that resolves a whole `Slot` kind from one caller-supplied buffer, which would stamp
/// the same data onto both differently-shaped ids.
///
/// `batch` rows share the embed table via a plain `[batch]` `Slot::Token` + axis-0 `gather`, as in
/// [`qwen3next_decode_trace`] (not gemma4's `Slot::TokenEmbed` host-embed workaround). A 35B deployment may need
/// that workaround if `[vocab,hidden]` exceeds a backend's max buffer size.
///
/// ## Arguments
///
/// - `cfg`: the same [`Qwen3NextConfig`] driving [`qwen3next_decode_trace`].
/// - `batch`: number of concurrent decode rows in this step.
/// - `cap`: attention KV logical capacity per row (same meaning as the single-seq tracer's `cap`).
/// - `pool_slots`: physical row count of the shared attention KV pool (`pool_slots <= batch*cap`; tight sharing is
///   safe because of the slot map, as in gemma4's tracer).
/// - `gdn_n_slots`: physical row count of the GDN state pool (>= the number of concurrently resident sequences,
///   independent of `batch`/`cap`/`pool_slots`, since GDN state does not grow).
///
/// Returns a `Graph` with `Slot::Token/Pos/Mask/SlotMap(x2)` inputs and per-layer `(kv or gdn)` state
/// pairs wired via `finish_with_state`, producing `logits [batch, 1, vocab]`.
pub fn qwen3next_decode_trace_batched_shared_pool(
    cfg: &Qwen3NextConfig,
    batch: usize,
    cap: usize,
    pool_slots: usize,
    gdn_n_slots: usize,
) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;

    let token = b.slot(Slot::Token, TensorType::new(vec![batch], DType::I32));
    let pos_slot = b.slot(Slot::Pos, TensorType::new(vec![batch], DType::I32));
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![batch, cap]));
    let mask4 = b.reshape(mask, vec![batch, 1, 1, cap]);
    // Attention shared-pool slot map: (row, logical cap position) -> global pool row.
    let slot_map = b.slot(Slot::SlotMap, TensorType::new(vec![batch, cap], DType::I32));
    // GDN pool row map: row -> this row's GDN pool slot. A distinct Slot kind from the attention shared-pool map
    // (card 188 Increment 5; see the doc comment).
    let gdn_slot_map = b.slot(Slot::GdnSlotMap, TensorType::new(vec![batch], DType::I32));

    // Partial NeoX RoPE tables (full-attention layers only).
    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let x0 = b.gather(embed, 0, token); // [batch, h]
    let mut x = b.reshape(x0, vec![batch, 1, h]); // [batch,1,hidden]

    // (state_in, state_out) pairs, heterogeneous per layer, accumulated across all layers.
    let mut state: Vec<(Traced, Traced)> = Vec::new();

    for li in 0..cfg.n_layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        // 1. Input RMSNorm (pre-norm). Named attn_norm in the checkpoint for both layer types.
        let attn_norm = b.constant(&p("attn_norm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, attn_norm, cfg.eps);

        // 2. Mixer: full-attention (batched shared-pool) or Gated DeltaNet (batched GDN pool).
        let mixer_out = if cfg.is_attn_layer(li) {
            let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
            let n_rep = nh / nkv;
            let q_dim = nh * hd;
            let scale = 1.0 / (hd as f32).sqrt();
            // attn_q carries query|gate interleaved per head: out = nh * 2 * head_dim.
            let wq = b.constant(&p("attn_q.weight"), TensorType::f32(vec![h, nh * 2 * hd]));
            let wk = b.constant(&p("attn_k.weight"), TensorType::f32(vec![h, nkv * hd]));
            let wv = b.constant(&p("attn_v.weight"), TensorType::f32(vec![h, nkv * hd]));
            let wo = b.constant(&p("attn_output.weight"), TensorType::f32(vec![nh * hd, h]));
            let qn = b.constant(&p("attn_q_norm.weight"), TensorType::f32(vec![hd]));
            let kn = b.constant(&p("attn_k_norm.weight"), TensorType::f32(vec![hd]));

            // Project Q (interleaved query|gate), K, V - mirrors qwen3next_gated_attention exactly,
            // generalized to [batch, ...] rows.
            let qg_flat = linear(&b, normed, wq, None); // [batch, 1, nh*2*hd]
            let qg = b.reshape(qg_flat, vec![batch, 1, nh, 2 * hd]);
            let q_raw = b.slice(qg, 3, 0, hd); // [batch, 1, nh, hd]
            let gate_raw = b.slice(qg, 3, hd, 2 * hd); // [batch, 1, nh, hd]
            let k_flat = linear(&b, normed, wk, None); // [batch, 1, nkv*hd]
            let v_flat = linear(&b, normed, wv, None); // [batch, 1, nkv*hd]

            // QK-norm (Qwen3-style, per-head, before RoPE).
            let q_normed = rmsnorm(&b, q_raw, qn, cfg.eps);
            let k_shaped = b.reshape(k_flat, vec![batch, 1, nkv, hd]);
            let k_normed = rmsnorm(&b, k_shaped, kn, cfg.eps);
            let v_shaped = b.reshape(v_flat, vec![batch, 1, nkv, hd]);

            // Transpose to [batch, heads, seq=1, head_dim] for RoPE/attention.
            let q4 = b.transpose(q_normed, vec![0, 2, 1, 3]);
            let k4 = b.transpose(k_normed, vec![0, 2, 1, 3]);
            let v4 = b.transpose(v_shaped, vec![0, 2, 1, 3]);

            // Batched partial NeoX RoPE (per-row position via pos_slot).
            let q4 = rope_batched(&b, q4, cos, sin, pos_slot, batch);
            let k4 = rope_batched(&b, k4, cos, sin, pos_slot, batch);

            // Shared-pool KV cache: write this row's new k/v at its GLOBAL slot
            // slot_map[row][pos[row]], then gather each row's cap logical slots back (existing
            // qwen2 helpers, reused verbatim).
            let kcache = b.state_input(
                &p("kv.k_cache"),
                TensorType::f32(vec![pool_slots, nkv, hd]),
                StateRole::Recurrent,
            );
            let vcache = b.state_input(
                &p("kv.v_cache"),
                TensorType::f32(vec![pool_slots, nkv, hd]),
                StateRole::Recurrent,
            );
            let kcache_out = scatter_shared_pool(&b, kcache, k4, pos_slot, slot_map, batch);
            let vcache_out = scatter_shared_pool(&b, vcache, v4, pos_slot, slot_map, batch);
            state.push((kcache, kcache_out));
            state.push((vcache, vcache_out));
            let kread = gather_shared_pool(&b, kcache_out, slot_map, batch, cap, nkv, hd);
            let vread = gather_shared_pool(&b, vcache_out, slot_map, batch, cap, nkv, hd);

            let attn = attention_masked(&b, q4, kread, vread, n_rep, scale, mask4); // [batch,nh,1,hd]

            // Output gate: sigmoid(gate) applied before o-projection.
            let gate4 = b.transpose(gate_raw, vec![0, 2, 1, 3]); // [batch, nh, 1, hd]
            let gate_sig = sigmoid(&b, gate4);
            let attn_gated = b.binary(BinOp::Mul, attn, gate_sig);

            // Flatten and output projection.
            let attn_back = b.transpose(attn_gated, vec![0, 2, 1, 3]); // [batch, 1, nh, hd]
            let attn_flat = b.reshape(attn_back, vec![batch, 1, q_dim]);
            linear(&b, attn_flat, wo, None) // [batch, 1, H]
        } else {
            let (hk, hv, hd, ck) = (
                cfg.gdn_num_k_heads,
                cfg.gdn_num_v_heads,
                cfg.gdn_head_dim,
                cfg.conv_k,
            );
            let key_dim = hk * hd;
            let value_dim = hv * hd;
            let conv_dim = 2 * key_dim + value_dim;
            let w_qkv = b.constant(&p("attn_qkv.weight"), TensorType::f32(vec![h, conv_dim]));
            let w_gate = b.constant(&p("attn_gate.weight"), TensorType::f32(vec![h, value_dim]));
            let w_conv = b.constant(&p("ssm_conv1d.weight"), TensorType::f32(vec![ck, conv_dim]));
            let w_beta = b.constant(&p("ssm_beta.weight"), TensorType::f32(vec![h, hv]));
            let w_alpha = b.constant(&p("ssm_alpha.weight"), TensorType::f32(vec![h, hv]));
            let dt_bias = b.constant(&p("ssm_dt.bias"), TensorType::f32(vec![hv]));
            let ssm_a = b.constant(&p("ssm_a"), TensorType::f32(vec![hv]));
            let norm_w = b.constant(&p("ssm_norm.weight"), TensorType::f32(vec![hd]));
            let w_out = b.constant(&p("ssm_out.weight"), TensorType::f32(vec![value_dim, h]));
            let conv_pool = b.state_input(
                &p("gdn.conv_cache"),
                TensorType::f32(vec![gdn_n_slots, ck - 1, conv_dim]),
                StateRole::Recurrent,
            );
            let ssm_pool = b.state_input(
                &p("gdn.ssm_state"),
                TensorType::f32(vec![gdn_n_slots, hv, hd, hd]),
                StateRole::Recurrent,
            );
            let (out, conv_pool_out, ssm_pool_out) = qwen3next_gdn_decode_batched_pool(
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
                conv_pool,
                ssm_pool,
                gdn_slot_map,
                hk,
                hv,
                hd,
                ck,
                cfg.eps,
                batch,
                GdnHeadOrder::Tiled,
            );
            state.push((conv_pool, conv_pool_out));
            state.push((ssm_pool, ssm_pool_out));
            out
        };

        // 3. Attention residual.
        x = b.binary(BinOp::Add, x, mixer_out);

        // 4. Pre-FFN RMSNorm.
        let post_norm = b.constant(&p("post_attention_norm.weight"), TensorType::f32(vec![h]));
        let ff_in = rmsnorm(&b, x, post_norm, cfg.eps);

        // 5. MoE FFN + shared expert, reused verbatim (FR-006): reshape [batch,1,H] -> [1,batch,H] so
        //    `moe()`'s internal L=shape[-2] dispatch sees L=batch (moe_grouped/moe_grouped_prep for
        //    batch>1, moe_sparse for batch=1 - identical to qwen3next_decode_trace at batch=1, FR-008).
        let (e, k, i, si_) = (cfg.n_experts, cfg.top_k, cfg.expert_inter, cfg.shared_inter);
        let router = b.constant(&p("ffn_gate_inp.weight"), TensorType::f32(vec![h, e]));
        let w_in = b.constant(&p("ffn.w_in"), TensorType::f32(vec![e, h, 2 * i]));
        let w_out_moe = b.constant(&p("ffn.w_out"), TensorType::f32(vec![e, i, h]));
        let sg = b.constant(&p("ffn_gate_shexp.weight"), TensorType::f32(vec![h, si_]));
        let su = b.constant(&p("ffn_up_shexp.weight"), TensorType::f32(vec![h, si_]));
        let sd = b.constant(&p("ffn_down_shexp.weight"), TensorType::f32(vec![si_, h]));
        let sgi = b.constant(&p("ffn_gate_inp_shexp.weight"), TensorType::f32(vec![h, 1]));
        let ff_in_l = b.reshape(ff_in, vec![1, batch, h]);
        let ffn_out_l = qwen3next_moe_ffn(
            &b, ff_in_l, router, w_in, w_out_moe, sg, su, sd, sgi, e, k, i,
        );
        let ffn_out = b.reshape(ffn_out_l, vec![batch, 1, h]);

        // 6. FFN residual.
        x = b.binary(BinOp::Add, x, ffn_out);
    }

    // Final RMSNorm + lm_head.
    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None); // [batch,1,vocab]

    b.finish_with_state(logits, &state)
}
