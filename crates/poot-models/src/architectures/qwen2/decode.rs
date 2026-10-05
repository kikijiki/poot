//! The legacy qwen2 decode tracers, held for POOT-739: the VLM caption path decodes through
//! `trace_decode_kv_masked`, and the mRoPE rows (`trace_decode_mrope`, the batched mRoPE decode and the
//! plain `trace_decode` they are compared with) are POOT-739's. No Runner dense path and no driver path
//! traces them: the qwen2 family's `Model` traces its own steps. POOT-739 deletes this module.

use super::*;

mod wrappers;
pub use wrappers::*;

/// Trace one decode token. `pos` is the new token's position; the prior KV length is `pos`, so after
/// append the KV length is `seq_len = pos + 1`. Returns the traced graph (logits as output).
///
/// The KV cache grows (cached_k/v are const state of length `pos`, new k/v are concatenated).
pub fn trace_decode(cfg: Qwen2Config, pos: usize) -> Graph {
    // Single-token capture path is qwen2-only (biased, no QK-norm); qwen3 decode uses the fixed-KV
    // tracers. Guard against silently tracing a wrong graph.
    assert!(
        cfg.qkv_bias && !cfg.qk_norm,
        "trace_decode currently supports only the Qwen2 variant (bias, no QK-norm); use trace_prefill"
    );
    let b = Builder::new();
    let h = cfg.hidden;
    let d = cfg.head_dim;
    let hq = cfg.n_heads;
    let hkv = cfg.n_kv_heads;
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();

    // the two per-token-varying inputs (growing KV, no mask: the concat output dim is `pos`/`cap`
    // itself, static at trace time, so no `Slot::SeqLen` is needed here - card 550 removes it).
    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32));

    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );

    // legalize (card 523a) hosts this gather off-device when the target's buffer limit needs it.
    let embed = b.constant("embed_tokens", TensorType::f32(vec![cfg.vocab, h]));
    let x0 = b.gather_scalar(embed, 0, token); // [hidden]
    let mut x = b.reshape(x0, vec![1, 1, h]); // [1,1,hidden]

    for li in 0..cfg.layers {
        let name = |s: &str| format!("layers.{li}.{s}");

        // attention block
        let ln1 = b.constant(&name("input_layernorm"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let wq = b.constant(&name("q_proj.w"), TensorType::f32(vec![h, q_dim]));
        let bq = b.constant(&name("q_proj.b"), TensorType::f32(vec![q_dim]));
        let wk = b.constant(&name("k_proj.w"), TensorType::f32(vec![h, kv_dim]));
        let bk = b.constant(&name("k_proj.b"), TensorType::f32(vec![kv_dim]));
        let wv = b.constant(&name("v_proj.w"), TensorType::f32(vec![h, kv_dim]));
        let bv = b.constant(&name("v_proj.b"), TensorType::f32(vec![kv_dim]));
        let wo = b.constant(&name("o_proj.w"), TensorType::f32(vec![q_dim, h]));

        let q = linear(&b, normed, wq, Some(bq));
        let k = linear(&b, normed, wk, Some(bk));
        let v = linear(&b, normed, wv, Some(bv));

        let q = b.reshape(q, vec![1, 1, hq, d]);
        let q = b.transpose(q, vec![0, 2, 1, 3]); // [1,Hq,1,D]
        let k = b.reshape(k, vec![1, 1, hkv, d]);
        let k = b.transpose(k, vec![0, 2, 1, 3]);
        let v = b.reshape(v, vec![1, 1, hkv, d]);
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        let q = rope(&b, q, cos, sin, pos_slot);
        let k = rope(&b, k, cos, sin, pos_slot);

        let cached_k = b.constant(&name("kv.cached_k"), TensorType::f32(vec![1, hkv, pos, d]));
        let cached_v = b.constant(&name("kv.cached_v"), TensorType::f32(vec![1, hkv, pos, d]));
        let k = b.concat(2, &[cached_k, k]); // [1,Hkv,seq_len,D]
        let v = b.concat(2, &[cached_v, v]);

        let attn = attention_softcap(&b, q, k, v, n_rep, scale, cfg.attn_logit_softcap); // [1,Hq,1,D]
        let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [1,1,Hq,D]
        let attn = b.reshape(attn, vec![1, 1, q_dim]);
        let attn = linear(&b, attn, wo, None); // [1,1,hidden]
        x = b.binary(BinOp::Add, x, attn); // residual

        // MLP block
        let ln2 = b.constant(&name("post_attention_layernorm"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let wg = b.constant(&name("gate_proj.w"), TensorType::f32(vec![h, cfg.inter]));
        let wu = b.constant(&name("up_proj.w"), TensorType::f32(vec![h, cfg.inter]));
        let wd = b.constant(&name("down_proj.w"), TensorType::f32(vec![cfg.inter, h]));
        let gate = linear(&b, normed, wg, None);
        let up = linear(&b, normed, wu, None);
        let act = swiglu(&b, gate, up);
        let mlp = linear(&b, act, wd, None);
        x = b.binary(BinOp::Add, x, mlp); // residual
    }

    let norm = b.constant("model.norm", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head (tied)", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None); // [1,1,vocab]
    let logits = match cfg.final_logit_softcap {
        Some(c) => softcap(&b, logits, c),
        None => logits,
    };

    b.finish(logits)
}

/// Split one axis-major `Slot::MropePosition` value into the three operands consumed by
/// `rope_sectioned`. `axis_shape=[]` is scalar decode (`packed=[3]`); `axis_shape=[L]` is prefill
/// (`packed=[3,L]`); `axis_shape=[B]` is static batched decode (`packed=[3,B]`).
pub(crate) fn split_mrope_position_axes(
    b: &Builder,
    packed: Traced,
    axis_shape: &[usize],
) -> [Traced; 3] {
    std::array::from_fn(|axis| {
        let one_axis = b.slice(packed, 0, axis, axis + 1);
        b.reshape(one_axis, axis_shape.to_vec())
    })
}

/// Trace one Qwen2-VL/2.5-VL text-tower decode token with sectioned ("mRoPE") position embedding (spec
/// 267, card 152). Identical to [`trace_decode`] (see [`Qwen2Config::mrope_section`]) except RoPE gathers
/// three per-token position ids (temporal/height/width) via [`rope_sectioned`] instead of one via [`rope`].
///
/// `pos_t`/`pos_h`/`pos_w` are this token's three position ids (HF's `get_rope_index`); `kv_len` is the
/// prior cached sequence length (like `trace_decode`'s `pos`, it bakes the `cached_k`/`cached_v` const
/// shape). For an all-text prompt every axis advances together (`pos_t == pos_h == pos_w == kv_len`), and
/// the output is then bit-exact to `trace_decode` (checked by `qwen2_pipeline`'s CPU-oracle test).
///
/// Position ids are one packed `Slot::MropePosition` input shaped `[3]` (temporal/height/width).
/// [`split_mrope_position_axes`] decomposes it with ordinary Slice/Reshape ops, so [`rope_sectioned`]
/// stays the only mRoPE math definition. GPU capture/replay slot-buffer updates are a separate
/// executor-equivalence gate; this tracer and the host binder are CPU-oracle plumbing.
pub fn trace_decode_mrope(cfg: Qwen2Config, kv_len: usize) -> Graph {
    let sections = cfg
        .mrope_section
        .expect("trace_decode_mrope needs cfg.mrope_section (Some([t,h,w]))");
    assert!(
        cfg.qkv_bias && !cfg.qk_norm,
        "trace_decode_mrope currently supports only the Qwen2-VL text-tower variant (bias, no QK-norm), \
         matching HF's Qwen2VLAttention (modeling_qwen2_vl.py) - see Qwen2Config::mrope_section's doc \
         comment"
    );
    let b = Builder::new();
    let h = cfg.hidden;
    let d = cfg.head_dim;
    let hq = cfg.n_heads;
    let hkv = cfg.n_kv_heads;
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();

    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let positions = b.slot(Slot::MropePosition, TensorType::new(vec![3], DType::I32));
    let [pos_t, pos_h, pos_w] = split_mrope_position_axes(&b, positions, &[]);

    let embed = b.constant("embed_tokens", TensorType::f32(vec![cfg.vocab, h]));
    let x0 = b.gather_scalar(embed, 0, token); // [hidden]
    let mut x = b.reshape(x0, vec![1, 1, h]); // [1,1,hidden]

    for li in 0..cfg.layers {
        let name = |s: &str| format!("layers.{li}.{s}");

        let ln1 = b.constant(&name("input_layernorm"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let wq = b.constant(&name("q_proj.w"), TensorType::f32(vec![h, q_dim]));
        let bq = b.constant(&name("q_proj.b"), TensorType::f32(vec![q_dim]));
        let wk = b.constant(&name("k_proj.w"), TensorType::f32(vec![h, kv_dim]));
        let bk = b.constant(&name("k_proj.b"), TensorType::f32(vec![kv_dim]));
        let wv = b.constant(&name("v_proj.w"), TensorType::f32(vec![h, kv_dim]));
        let bv = b.constant(&name("v_proj.b"), TensorType::f32(vec![kv_dim]));
        let wo = b.constant(&name("o_proj.w"), TensorType::f32(vec![q_dim, h]));

        let q = linear(&b, normed, wq, Some(bq));
        let k = linear(&b, normed, wk, Some(bk));
        let v = linear(&b, normed, wv, Some(bv));

        let q = b.reshape(q, vec![1, 1, hq, d]);
        let q = b.transpose(q, vec![0, 2, 1, 3]); // [1,Hq,1,D]
        let k = b.reshape(k, vec![1, 1, hkv, d]);
        let k = b.transpose(k, vec![0, 2, 1, 3]);
        let v = b.reshape(v, vec![1, 1, hkv, d]);
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        let q = rope_sectioned(&b, q, cos, sin, &[pos_t, pos_h, pos_w], &sections);
        let k = rope_sectioned(&b, k, cos, sin, &[pos_t, pos_h, pos_w], &sections);

        let cached_k = b.constant(
            &name("kv.cached_k"),
            TensorType::f32(vec![1, hkv, kv_len, d]),
        );
        let cached_v = b.constant(
            &name("kv.cached_v"),
            TensorType::f32(vec![1, hkv, kv_len, d]),
        );
        let k = b.concat(2, &[cached_k, k]); // [1,Hkv,seq_len,D]
        let v = b.concat(2, &[cached_v, v]);

        let attn = attention_softcap(&b, q, k, v, n_rep, scale, cfg.attn_logit_softcap); // [1,Hq,1,D]
        let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [1,1,Hq,D]
        let attn = b.reshape(attn, vec![1, 1, q_dim]);
        let attn = linear(&b, attn, wo, None); // [1,1,hidden]
        x = b.binary(BinOp::Add, x, attn); // residual

        let ln2 = b.constant(&name("post_attention_layernorm"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let wg = b.constant(&name("gate_proj.w"), TensorType::f32(vec![h, cfg.inter]));
        let wu = b.constant(&name("up_proj.w"), TensorType::f32(vec![h, cfg.inter]));
        let wd = b.constant(&name("down_proj.w"), TensorType::f32(vec![cfg.inter, h]));
        let gate = linear(&b, normed, wg, None);
        let up = linear(&b, normed, wu, None);
        let act = swiglu(&b, gate, up);
        let mlp = linear(&b, act, wd, None);
        x = b.binary(BinOp::Add, x, mlp); // residual
    }

    let norm = b.constant("model.norm", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head (tied)", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None); // [1,1,vocab]
    let logits = match cfg.final_logit_softcap {
        Some(c) => softcap(&b, logits, c),
        None => logits,
    };

    b.finish(logits)
}
fn trace_decode_kv_masked_impl(cfg: Qwen2Config, cap: usize) -> Graph {
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();

    // `Slot::Pos` is the `[rows, tokens]` I32 absolute-position input (card 550): one row,
    // one token for plain single-sequence decode. `pos_slot` is its first (only) column, the scalar the
    // rest of this tracer (RoPE, the KV write offset) already used before the card.
    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, 1], DType::I32));
    let pos_slot = b.reshape(pos, vec![]);
    // Additive attention mask over the full cache, built in-graph from `pos` against `iota(cap)` (card
    // 550): 0 for valid slots, -1e9 past pos (and outside the window, if any). RoPE path: one row
    // broadcast over heads. ALiBi (spec 254): one row per head carrying the linear distance bias plus
    // visibility; `attention_masked_softcap`'s additive add is the same.
    let mask = if cfg.alibi {
        let slopes = b.constant("alibi.slopes", TensorType::f32(vec![hq]));
        alibi_mask_from_pos(&b, pos, cap, cfg.sliding_window, slopes, hq) // [1,hq,1,cap]
    } else {
        causal_mask_from_pos(&b, pos, cap, cfg.sliding_window) // [1,1,1,cap]
    };

    // ALiBi replaces RoPE (see `cfg.alibi`), so the rope.cos/rope.sin tables are declared only on the RoPE path.
    let rope_tables = (!cfg.alibi).then(|| {
        (
            b.constant(
                "rope.cos",
                TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
            ),
            b.constant(
                "rope.sin",
                TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
            ),
        )
    });

    // legalize (card 523a) hosts this gather off-device when the target's buffer limit needs it.
    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let x0 = b.gather_scalar(embed, 0, token);
    let mut x = b.reshape(x0, vec![1, 1, h]);

    let mut state: Vec<(poot_graph_ir::Traced, poot_graph_ir::Traced)> =
        Vec::with_capacity(2 * cfg.layers);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let wq = b.constant(
            &p("self_attn.q_proj.weight"),
            TensorType::new(vec![h, q_dim], cfg.proj_dtype),
        );
        let wk = b.constant(
            &p("self_attn.k_proj.weight"),
            TensorType::new(vec![h, kv_dim], cfg.proj_dtype),
        );
        let wv = b.constant(
            &p("self_attn.v_proj.weight"),
            TensorType::new(vec![h, kv_dim], cfg.proj_dtype),
        );
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::new(vec![q_dim, h], cfg.proj_dtype),
        );
        let (bq, bk, bv) = if cfg.qkv_bias {
            (
                Some(b.constant(&p("self_attn.q_proj.bias"), TensorType::f32(vec![q_dim]))),
                Some(b.constant(&p("self_attn.k_proj.bias"), TensorType::f32(vec![kv_dim]))),
                Some(b.constant(&p("self_attn.v_proj.bias"), TensorType::f32(vec![kv_dim]))),
            )
        } else {
            (None, None, None)
        };

        let q = linear(&b, normed, wq, bq);
        let k = linear(&b, normed, wk, bk);
        let v = linear(&b, normed, wv, bv);

        let mut q = b.reshape(q, vec![1, 1, hq, d]);
        let mut k = b.reshape(k, vec![1, 1, hkv, d]);
        let v = b.reshape(v, vec![1, 1, hkv, d]);
        if cfg.qk_norm {
            let qn = b.constant(&p("self_attn.q_norm.weight"), TensorType::f32(vec![d]));
            let kn = b.constant(&p("self_attn.k_norm.weight"), TensorType::f32(vec![d]));
            q = rmsnorm(&b, q, qn, cfg.eps);
            k = rmsnorm(&b, k, kn, cfg.eps);
        }
        let q = b.transpose(q, vec![0, 2, 1, 3]);
        let k = b.transpose(k, vec![0, 2, 1, 3]);
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        // ALiBi (cfg.alibi): no rotation; the per-head linear bias lives in `mask`.
        let (q, k) = match rope_tables {
            Some((cos, sin)) => (
                rope(&b, q, cos, sin, pos_slot),
                rope(&b, k, cos, sin, pos_slot),
            ),
            None => (q, k),
        };

        // carried caches; scatter the new token's k/v at the runtime slot `pos` (axis 2).
        let kcache = b.state_input(
            &p("kv.k_cache"),
            TensorType::f32(vec![1, hkv, cap, d]),
            StateRole::Recurrent,
        );
        let vcache = b.state_input(
            &p("kv.v_cache"),
            TensorType::f32(vec![1, hkv, cap, d]),
            StateRole::Recurrent,
        );
        let kread = b.dynamic_update_slice_dyn(kcache, k, pos_slot, 2);
        let vread = b.dynamic_update_slice_dyn(vcache, v, pos_slot, 2);
        state.push((kcache, kread));
        state.push((vcache, vread));

        // attend over the full cache with the additive mask (constant shape; masked slots get ~0 weight).
        // `compile`'s `flash_attention_capped` forms the flash op from this chain where it fits.
        let attn = attention_masked_softcap(
            &b,
            q,
            kread,
            vread,
            n_rep,
            scale,
            mask,
            cfg.attn_logit_softcap,
        );
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, 1, q_dim]);
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let wg = b.constant(
            &p("mlp.gate_proj.weight"),
            TensorType::new(vec![h, cfg.inter], cfg.proj_dtype),
        );
        let wu = b.constant(
            &p("mlp.up_proj.weight"),
            TensorType::new(vec![h, cfg.inter], cfg.proj_dtype),
        );
        let wd = b.constant(
            &p("mlp.down_proj.weight"),
            TensorType::new(vec![cfg.inter, h], cfg.proj_dtype),
        );
        let gate = linear(&b, normed, wg, None);
        let up = linear(&b, normed, wu, None);
        let act = swiglu(&b, gate, up);
        let mlp = linear(&b, act, wd, None);
        x = b.binary(BinOp::Add, x, mlp);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None);
    let logits = match cfg.final_logit_softcap {
        Some(c) => softcap(&b, logits, c),
        None => logits,
    };

    b.finish_with_state(logits, &state)
}

/// The adapter-pool shape the Runner's LoRA pool reports (`n_adapters` real adapters, one shared `rank`
/// across the seven projections and every layer; a smaller module rank is zero-padded, which is exact).
/// The tracers that consumed it are deleted; the Runner's pool still returns it until POOT-739 deletes
/// the Runner.
#[derive(Clone, Copy, Debug)]
pub struct LoraBatchedSpec {
    pub n_adapters: usize,
    pub rank: usize,
}

fn trace_decode_kv_masked_batched_impl(cfg: Qwen2Config, cap: usize, batch: usize) -> Graph {
    assert!(
        cfg.mrope_section.is_none() || (!cfg.alibi && cfg.qkv_bias && !cfg.qk_norm),
        "batched mRoPE decode supports only the Qwen2-VL text-tower variant (RoPE, bias, no QK-norm)"
    );
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();

    let token = b.slot(Slot::Token, TensorType::new(vec![batch], DType::I32));
    // `[batch, 1]`: one row per sequence, one token each (card 550). `pos_slot` is the
    // `[batch]` flat column every pre-card consumer here (RoPE, the KV scatter helpers) already expected.
    let pos = b.slot(Slot::Pos, TensorType::new(vec![batch, 1], DType::I32));
    let pos_slot = b.reshape(pos, vec![batch]);
    // Absolute token positions drive KV addressing and masks. mRoPE positions can differ after a visual
    // prompt, so Q/K rotation gets a separate axis-major value bound atomically with them.
    let mrope_positions = cfg.mrope_section.map(|_| {
        let packed = b.slot(
            Slot::MropePosition,
            TensorType::new(vec![3, batch], DType::I32),
        );
        split_mrope_position_axes(&b, packed, &[batch])
    });
    // per-row additive mask over the full cache, built in-graph from `pos` against `iota(cap)`. RoPE path:
    // [B,1,1,cap] (broadcast over Hq and the q axis). ALiBi (spec 254/255): [B,hq,1,cap] (one bias row per
    // head per row).
    let mask = if cfg.alibi {
        let slopes = b.constant("alibi.slopes", TensorType::f32(vec![hq]));
        alibi_mask_from_pos(&b, pos, cap, cfg.sliding_window, slopes, hq) // [batch,hq,1,cap]
    } else {
        causal_mask_from_pos(&b, pos, cap, cfg.sliding_window) // [batch,1,1,cap]
    };
    // ALiBi replaces RoPE, so the rope.cos/rope.sin tables are declared only on the RoPE path.
    let rope_tables = (!cfg.alibi).then(|| {
        (
            b.constant(
                "rope.cos",
                TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
            ),
            b.constant(
                "rope.sin",
                TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
            ),
        )
    });

    // legalize (card 523a) hosts this gather off-device when the target's buffer limit needs it.
    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let x0 = b.gather(embed, 0, token); // [B, h]
    let mut x = b.reshape(x0, vec![batch, 1, h]);

    let mut state: Vec<(poot_graph_ir::Traced, poot_graph_ir::Traced)> =
        Vec::with_capacity(2 * cfg.layers);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let (bq, bk, bv) = if cfg.qkv_bias {
            (
                Some(b.constant(&p("self_attn.q_proj.bias"), TensorType::f32(vec![q_dim]))),
                Some(b.constant(&p("self_attn.k_proj.bias"), TensorType::f32(vec![kv_dim]))),
                Some(b.constant(&p("self_attn.v_proj.bias"), TensorType::f32(vec![kv_dim]))),
            )
        } else {
            (None, None, None)
        };

        let proj = |x, prefix: &str, in_dim, out_dim, bias| {
            let w = b.constant(
                &p(&format!("{prefix}.weight")),
                TensorType::f32(vec![in_dim, out_dim]),
            );
            linear(&b, x, w, bias)
        };

        let q = proj(normed, "self_attn.q_proj", h, q_dim, bq);
        let k = proj(normed, "self_attn.k_proj", h, kv_dim, bk);
        let v = proj(normed, "self_attn.v_proj", h, kv_dim, bv);

        let mut q = b.reshape(q, vec![batch, 1, hq, d]);
        let mut k = b.reshape(k, vec![batch, 1, hkv, d]);
        let v = b.reshape(v, vec![batch, 1, hkv, d]);
        if cfg.qk_norm {
            let qn = b.constant(&p("self_attn.q_norm.weight"), TensorType::f32(vec![d]));
            let kn = b.constant(&p("self_attn.k_norm.weight"), TensorType::f32(vec![d]));
            q = rmsnorm(&b, q, qn, cfg.eps);
            k = rmsnorm(&b, k, kn, cfg.eps);
        }
        let q = b.transpose(q, vec![0, 2, 1, 3]); // [B, Hq, 1, D]
        let k = b.transpose(k, vec![0, 2, 1, 3]); // [B, Hkv, 1, D]
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        // ALiBi (cfg.alibi): no rotation; the per-head linear bias lives in `mask`.
        let (q, k) = match (rope_tables, cfg.mrope_section, mrope_positions) {
            (Some((cos, sin)), Some(sections), Some(positions)) => (
                rope_sectioned_batched(&b, q, cos, sin, &positions, &sections, batch),
                rope_sectioned_batched(&b, k, cos, sin, &positions, &sections, batch),
            ),
            (Some((cos, sin)), None, None) => (
                rope_batched(&b, q, cos, sin, pos_slot, batch),
                rope_batched(&b, k, cos, sin, pos_slot, batch),
            ),
            (None, None, None) => (q, k),
            _ => unreachable!("mRoPE positions are declared exactly when sections are configured"),
        };

        // carried per-row caches; scatter each row's new k/v into its own runtime slot `pos[row]` (axis 2):
        // B unrolled single-row writes, concatenated back.
        let kcache = b.state_input(
            &p("kv.k_cache"),
            TensorType::f32(vec![batch, hkv, cap, d]),
            StateRole::Recurrent,
        );
        let vcache = b.state_input(
            &p("kv.v_cache"),
            TensorType::f32(vec![batch, hkv, cap, d]),
            StateRole::Recurrent,
        );
        let (kread, vread) = (
            scatter_per_row(&b, kcache, k, pos_slot, batch),
            scatter_per_row(&b, vcache, v, pos_slot, batch),
        );
        state.push((kcache, kread));
        state.push((vcache, vread));
        let attn = attention_masked_softcap(
            &b,
            q,
            kread,
            vread,
            n_rep,
            scale,
            mask,
            cfg.attn_logit_softcap,
        );
        let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [B, 1, Hq, D]
        let attn = b.reshape(attn, vec![batch, 1, q_dim]);
        let attn = proj(attn, "self_attn.o_proj", q_dim, h, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let gate = proj(normed, "mlp.gate_proj", h, cfg.inter, None);
        let up = proj(normed, "mlp.up_proj", h, cfg.inter, None);
        let act = swiglu(&b, gate, up);
        let mlp = proj(act, "mlp.down_proj", cfg.inter, h, None);
        x = b.binary(BinOp::Add, x, mlp);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None); // [B, 1, vocab]
    let logits = match cfg.final_logit_softcap {
        Some(c) => softcap(&b, logits, c),
        None => logits,
    };

    b.finish_with_state(logits, &state)
}

/// Scatter each batch row's `update[b]` into `cache[b]` at that row's runtime position `pos[b]` along axis 2
/// (the cache-slot axis). `cache[B,Hkv,cap,D]`, `update[B,Hkv,1,D]`, `pos[B]`. B is fixed at trace time, so
/// this unrolls to B single-row `dynamic_update_slice` writes concatenated back to `[B,...]`.
fn scatter_per_row(
    b: &Builder,
    cache: poot_graph_ir::Traced,
    update: poot_graph_ir::Traced,
    pos: poot_graph_ir::Traced,
    batch: usize,
) -> poot_graph_ir::Traced {
    let rows: Vec<poot_graph_ir::Traced> = (0..batch)
        .map(|row| {
            let cache_row = b.slice(cache, 0, row, row + 1); // [1, Hkv, cap, D]
            let update_row = b.slice(update, 0, row, row + 1); // [1, Hkv, 1, D]
            // DUS needs a rank-0 runtime index; reshape this row's [1] pos slice to a scalar.
            let pos_row = b.reshape(b.slice(pos, 0, row, row + 1), vec![]);
            b.dynamic_update_slice_dyn(cache_row, update_row, pos_row, 2)
        })
        .collect();
    // fold to a left-leaning tree of 2-input concats: the GPU lowers only 2-input concat (a >2-input concat
    // is host-routed, which the GPU-resident cached-decode path forbids). Concat is associative.
    rows.into_iter()
        .reduce(|acc, r| b.concat(0, &[acc, r]))
        .expect("batch >= 1")
}
