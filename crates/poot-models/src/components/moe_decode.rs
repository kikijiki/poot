//! Shared MoE-capable batched shared-pool decode tracer (spec 249 "Decode counterpart").
//! [`crate::moe_prefill::trace_moe_prefill_kv_shared_pool`] covers admission for granitemoe/qwen3_moe; this
//! covers the per-iteration decode step `batch_engine_loop` runs for admitted sequences (the qwen2/llama-only
//! [`crate::qwen2::trace_decode_kv_masked_batched_shared_pool_ext`] had no MoE equivalent).
//!
//! Same "swap the FFN block, keep the block scaffolding" pattern as [`crate::moe_prefill`]: an `n_slots`-wide,
//! one-new-token-per-row batched decode over the `KvLayout::SharedPool` arm of
//! `crate::qwen2::trace_decode_kv_masked_batched_impl`: a `[B]` token vector, a `[B]` position vector, each row
//! addressing its own logical-to-physical `[B,cap]` [`poot_graph_ir::Slot::SlotMap`] against one shared
//! `[pool_slots,Hkv,D]` K/V pool per layer, reusing [`crate::components::scatter_shared_pool`]/
//! [`crate::components::gather_shared_pool`] (also used by `moe_prefill` and gemma4's batched decode). Parameterized by
//! an FFN closure and Granite's scalar knobs as `moe_prefill`, so
//! `granite::trace_granite_decode_kv_masked_batched_shared_pool` and
//! `qwen3moe::trace_qwen3_moe_decode_kv_masked_batched_shared_pool` are thin wrappers. Weight names match
//! `trace_granite_decode_kv_masked`/`trace_qwen3_moe_decode_kv_masked`, so loaded weights bind to either tracer,
//! and `n_slots` batched steps over a row's token stream must equal that row's sequence of single-sequence
//! contiguous decode steps (the differential CPU-oracle test in `poot-eval`, as
//! `gemma4_batched_decode_matches_sequential_single_seq`).

use poot_graph_ir::ops::{attention_masked, linear, rmsnorm, rope_batched};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, TensorType, Traced};
use poot_tensor::DType;

use crate::components::{gather_shared_pool, scatter_shared_pool};
use crate::qwen2::Qwen2Config;

/// Trace a batched shared-pool decode for an MoE-capable arch (granite/granitemoe, qwen3_moe): the generic core
/// [`crate::granite::trace_granite_decode_kv_masked_batched_shared_pool`] and
/// [`crate::qwen3moe::trace_qwen3_moe_decode_kv_masked_batched_shared_pool`] call, with their own FFN closure and
/// scalar knobs (same meanings as [`crate::moe_prefill::trace_moe_prefill_kv_shared_pool`]). `cap` is each row's
/// logical capacity (the `[cap]` window its `Slot::SlotMap` row addresses); `batch` is the number of concurrently
/// decoding rows; `pool_slots` is the shared physical pool size (possibly far smaller than `batch * cap`). `ffn`
/// is `(builder, normalized hidden [1,batch,H], layer index) -> mlp output [1,batch,H]` (pre `residual_mult`, pre
/// residual add), the same `[1,N,H]` contract as `moe_prefill`'s closure (`N = batch` here); this tracer reshapes
/// its `[batch,1,H]` operand to `[1,batch,H]` around the call (see the comment at the call site).
#[allow(clippy::too_many_arguments)]
pub fn trace_moe_decode_kv_masked_batched_shared_pool<F>(
    cfg: Qwen2Config,
    qk_norm: bool,
    embed_mult: f32,
    attn_scale: f32,
    residual_mult: f32,
    logits_mult: f32,
    cap: usize,
    batch: usize,
    pool_slots: usize,
    ffn: F,
) -> Graph
where
    F: Fn(&Builder, Traced, usize) -> Traced,
{
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;

    let token = b.slot(Slot::Token, TensorType::new(vec![batch], DType::I32));
    let pos_slot = b.slot(Slot::Pos, TensorType::new(vec![batch], DType::I32));
    let _seq_len = b.slot(Slot::SeqLen, TensorType::scalar(DType::I32));
    // Per-row additive mask over the shared pool's [cap] logical window (broadcast over Hq and the q axis).
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![batch, cap]));
    let mask = b.reshape(mask, vec![batch, 1, 1, cap]);
    // The [B, cap] forward logical->physical slot map (row r's logical position t -> global pool slot
    // slot_map[r][t]), as qwen2's SharedPool batched decode, so `Runner::bind_decode_batched` binds this graph
    // unchanged.
    let slot_map = b.slot(Slot::SlotMap, TensorType::new(vec![batch, cap], DType::I32));

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
    let x0 = b.gather(embed, 0, token); // [B, h]
    let x0 = b.reshape(x0, vec![batch, 1, h]);
    // Granite scales the embeddings by embedding_multiplier; qwen3-moe passes 1.0 (an exact no-op for finite f32),
    // so this stays one code path.
    let mut x = b.binary_scalar(BinOp::Mul, x0, Builder::f32(embed_mult));

    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(2 * cfg.layers);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let wq = b.constant(
            &p("self_attn.q_proj.weight"),
            TensorType::f32(vec![h, q_dim]),
        );
        let wk = b.constant(
            &p("self_attn.k_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let wv = b.constant(
            &p("self_attn.v_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![q_dim, h]),
        );

        let q = linear(&b, normed, wq, None);
        let k = linear(&b, normed, wk, None);
        let v = linear(&b, normed, wv, None);

        let mut q = b.reshape(q, vec![batch, 1, hq, d]);
        let mut k = b.reshape(k, vec![batch, 1, hkv, d]);
        let v = b.reshape(v, vec![batch, 1, hkv, d]);
        if qk_norm {
            let qn = b.constant(&p("self_attn.q_norm.weight"), TensorType::f32(vec![d]));
            let kn = b.constant(&p("self_attn.k_norm.weight"), TensorType::f32(vec![d]));
            q = rmsnorm(&b, q, qn, cfg.eps);
            k = rmsnorm(&b, k, kn, cfg.eps);
        }
        let q = b.transpose(q, vec![0, 2, 1, 3]); // [B, Hq, 1, D]
        let k = b.transpose(k, vec![0, 2, 1, 3]); // [B, Hkv, 1, D]
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        let q = rope_batched(&b, q, cos, sin, pos_slot, batch);
        let k = rope_batched(&b, k, cos, sin, pos_slot, batch);

        // Shared-pool cache write and logical-order read, as the `KvLayout::SharedPool` arm of
        // `trace_decode_kv_masked_batched_impl`. Each row writes its new K/V at global slot
        // `slot_map[row][pos[row]]` in the one shared pool buffer, then gathers its `cap` logical slots for attention.
        let kcache = b.state_input(
            &p("kv.k_cache"),
            TensorType::f32(vec![pool_slots, hkv, d]),
            StateRole::Recurrent,
        );
        let vcache = b.state_input(
            &p("kv.v_cache"),
            TensorType::f32(vec![pool_slots, hkv, d]),
            StateRole::Recurrent,
        );
        let kcache_out = scatter_shared_pool(&b, kcache, k, pos_slot, slot_map, batch);
        let vcache_out = scatter_shared_pool(&b, vcache, v, pos_slot, slot_map, batch);
        state.push((kcache, kcache_out));
        state.push((vcache, vcache_out));

        let kread = gather_shared_pool(&b, kcache_out, slot_map, batch, cap, hkv, d);
        let vread = gather_shared_pool(&b, vcache_out, slot_map, batch, cap, hkv, d);

        let attn = attention_masked(&b, q, kread, vread, n_rep, attn_scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [B, 1, Hq, D]
        let attn = b.reshape(attn, vec![batch, 1, q_dim]);
        let attn = linear(&b, attn, wo, None);
        let attn = b.binary_scalar(BinOp::Mul, attn, Builder::f32(residual_mult));
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps); // [B,1,H]
        // `ops::moe` (`moe_sparse`/`moe_grouped`) hardcodes a single leading batch-1 axis (`x[1,L,H]`). MoE
        // routing/FFN has no cross-token interaction, so B independent single-token rows and one sequence of B
        // tokens are the same computation: present the rows to `ffn` as `[1,L,H]` with L = batch rather than
        // extending the shared op. This keeps `ffn`'s contract identical to `trace_moe_prefill_kv_shared_pool`'s,
        // so the granite/qwen3moe decode wrapper closures match their prefill siblings.
        let normed_flat = b.reshape(normed, vec![1, batch, h]);
        let m = ffn(&b, normed_flat, li);
        let m = b.reshape(m, vec![batch, 1, h]);
        let m = b.binary_scalar(BinOp::Mul, m, Builder::f32(residual_mult));
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None); // [B, 1, vocab]
    let logits = b.binary_scalar(BinOp::Mul, logits, Builder::f32(logits_mult));
    b.finish_with_state(logits, &state)
}
