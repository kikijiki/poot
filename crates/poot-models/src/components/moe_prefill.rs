//! Shared MoE-capable shared-pool prefill tracer (spec 249). The paged serving path
//! ([`crate::qwen2::trace_prefill_kv_shared_pool_ext`]) had no MoE equivalent, so granitemoe and qwen3_moe could
//! not use `Runner::prefill_shared_pool_step`: their contiguous prefill tracers
//! (`trace_granite_prefill_kv`/`trace_qwen3_moe_prefill_kv`) use a different cache layout (`[1,Hkv,cap,D]` at
//! slot 0, versus the pool's `[pool,Hkv,D]` at arbitrary physical slots).
//!
//! The two MoE archs' contiguous tracers ([`crate::granite::trace_granite_prefill_kv`],
//! [`crate::qwen3moe::trace_qwen3_moe_prefill_kv`]) are near-identical to qwen2 dense's (`Contiguous` arm of
//! `crate::qwen2::trace_prefill_kv_impl`): a pre-norm GQA block (`attention_prefill` + `rope_prefill`) differing
//! in the FFN/routing composition (dense swiglu vs [`poot_graph_ir::ops::moe`], with different per-expert weight
//! names) and, for Granite, four scalar multipliers (`embed_mult`/`attn_mult`/`residual_mult`/`logits_scale`).
//! This module is one generic tracer parameterized by an FFN closure (as `crate::qwen3moe::ffn`) plus the scalar
//! knobs, which are identity at qwen3-moe's call site (`embed_mult=1.0`, `residual_mult=1.0`, `logits_mult=1.0`;
//! `qk_norm` differs per arch).
//!
//! The write/read mechanics are those of the `PrefillKv::SharedPool` arm of `crate::qwen2::trace_prefill_kv_impl`:
//! scatter the `n` fresh K/V rows into a `[pool,Hkv,D]` shared cache by the `Slot::SlotMap` inverse map
//! (`inv[physical] = logical`, or -1 to pass another sequence's/zero row through), while attention reads the
//! fresh K/V directly. The cache write never affects the output, so the logits must be bit-identical to the
//! contiguous prefill tracer for the same weights/tokens regardless of physical slot layout (the differential
//! CPU-oracle test in `poot-eval`'s `moe_paged_prefill` module)

use poot_graph_ir::ops::{attention_prefill, linear, rmsnorm, rope_prefill};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, TensorType, Traced};
use poot_tensor::DType;

use crate::qwen2::Qwen2Config;

/// Trace a shared-pool paged prefill for an MoE-capable arch (granite/granitemoe, qwen3_moe): the generic core
/// [`crate::granite::trace_granite_prefill_kv_shared_pool`] and
/// [`crate::qwen3moe::trace_qwen3_moe_prefill_kv_shared_pool`] call, with their own FFN closure and scalar knobs.
/// `qk_norm` toggles the qwen3-style per-head RMSNorm on Q/K before RoPE (granite: `false`, qwen3-moe: `true`);
/// `embed_mult`/`residual_mult`/`logits_mult` are Granite's scalar multipliers (`1.0` for qwen3-moe);
/// `attn_scale` is the softmax scale (Granite's `attention_multiplier`, or `1/sqrt(head_dim)`). `ffn` is
/// `(builder, normalized hidden [1,n,H], layer index) -> mlp output [1,n,H]` (pre `residual_mult`, pre residual
/// add), the piece that differs between the archs. `n` is the prompt length; `pool` is the shared physical pool
/// size (`pool >= n`).
#[allow(clippy::too_many_arguments)]
pub fn trace_moe_prefill_kv_shared_pool<F>(
    cfg: Qwen2Config,
    qk_norm: bool,
    embed_mult: f32,
    attn_scale: f32,
    residual_mult: f32,
    logits_mult: f32,
    n: usize,
    pool: usize,
    ffn: F,
) -> Graph
where
    F: Fn(&Builder, Traced, usize) -> Traced,
{
    assert!(
        pool >= n,
        "shared pool size {pool} must be at least the prompt length {n}"
    );
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let l = n;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    // The [pool] inverse slot map: inv[physical] = the prompt position whose K/V lands there, or -1 to pass another
    // sequence's (or zero) row through. Same convention as `trace_prefill_kv_shared_pool_ext`, so
    // `Runner::bind_prefill_kv_paged` binds this graph unchanged.
    let inv = b.slot(Slot::SlotMap, TensorType::new(vec![pool], DType::I32));
    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let mask = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens);
    let x0 = b.reshape(emb, vec![1, l, h]);
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

        let mut q = b.reshape(q, vec![1, l, hq, d]);
        let mut k = b.reshape(k, vec![1, l, hkv, d]);
        let v = b.reshape(v, vec![1, l, hkv, d]);
        if qk_norm {
            let qn = b.constant(&p("self_attn.q_norm.weight"), TensorType::f32(vec![d]));
            let kn = b.constant(&p("self_attn.k_norm.weight"), TensorType::f32(vec![d]));
            q = rmsnorm(&b, q, qn, cfg.eps);
            k = rmsnorm(&b, k, kn, cfg.eps);
        }
        let q = b.transpose(q, vec![0, 2, 1, 3]); // [1,Hq,l,D]
        let k = b.transpose(k, vec![0, 2, 1, 3]); // [1,Hkv,l,D]
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        let q = rope_prefill(&b, q, cos, sin, l);
        let k = rope_prefill(&b, k, cos, sin, l);

        // Shared-pool cache write: scatter this call's [1,Hkv,l,D] K/V into the [pool,Hkv,D] cache at global physical
        // slots via the inverse map, passing other sequences' slots through (as qwen2's `PrefillKv::SharedPool` arm).
        let kcache = b.state_input(
            &p("kv.k_cache"),
            TensorType::f32(vec![pool, hkv, d]),
            StateRole::Recurrent,
        );
        let vcache = b.state_input(
            &p("kv.v_cache"),
            TensorType::f32(vec![pool, hkv, d]),
            StateRole::Recurrent,
        );
        let write_cache = |cache: Traced, newkv: Traced| -> Traced {
            let base = b.reshape(cache, vec![pool, kv_dim]); // [pool, Hkv*D]
            let src = b.reshape(b.transpose(newkv, vec![2, 1, 0, 3]), vec![l, kv_dim]); // [l, Hkv*D]
            let updated = b.scatter_update(base, src, inv);
            b.reshape(updated, vec![pool, hkv, d])
        };
        let kcache_out = write_cache(kcache, k);
        let vcache_out = write_cache(vcache, v);
        state.push((kcache, kcache_out));
        state.push((vcache, vcache_out));

        // Attention reads the fresh k/v directly; the cache write above only feeds carried state.
        let attn = attention_prefill(&b, q, k, v, n_rep, attn_scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [1,l,Hq,D]
        let attn = b.reshape(attn, vec![1, l, q_dim]);
        let attn = linear(&b, attn, wo, None);
        let attn = b.binary_scalar(BinOp::Mul, attn, Builder::f32(residual_mult));
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = ffn(&b, normed, li);
        let m = b.binary_scalar(BinOp::Mul, m, Builder::f32(residual_mult));
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    let logits = b.binary_scalar(BinOp::Mul, logits, Builder::f32(logits_mult));
    b.finish_with_state(logits, &state)
}
