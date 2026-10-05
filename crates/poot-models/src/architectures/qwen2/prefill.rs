//! The legacy qwen2 prefill tracers, held for POOT-739: `trace_prefill_kv` and `trace_prefill_kv_embeds`
//! (the VLM caption path and its rows), `trace_prefill_mrope` and `trace_qwen2_5_vl_prefill_kv_embeds`
//! (the mRoPE rows) and `trace_prefill`, the plain tracer those rows are compared with. The qwen2
//! family's `Model` traces its own steps. POOT-739 deletes this module.

use super::decode::split_mrope_position_axes;

use super::*;

mod wrappers;
pub use wrappers::*;

fn trace_prefill_impl(cfg: Qwen2Config, seq_len: usize) -> Graph {
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();
    let l = seq_len;

    // the prompt token ids (one per position), a [L] vector input.
    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    // this one-shot prefill always starts at position 0 (card 550): `Slot::Pos` is `[1,l]`,
    // bound as `[0,1,..,l-1]`, feeding the in-graph mask below (RoPE still uses the static `0..l` table
    // slice, unaffected: it already equals a gather by these same positions).
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, l], DType::I32));
    // ALiBi (card 255): no rotation; the per-head linear bias lives in `mask`.
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
    // RoPE path: [1,1,l,l], broadcast over every head. ALiBi: [1,hq,l,l], one bias row per head per query
    // row. Built in-graph from `pos` against `iota(l)` (card 550).
    let mask = if cfg.alibi {
        let slopes = b.constant("alibi.slopes", TensorType::f32(vec![hq]));
        alibi_mask_from_pos(&b, pos, l, cfg.sliding_window, slopes, hq) // [1,hq,l,l]
    } else {
        causal_mask_from_pos(&b, pos, l, cfg.sliding_window) // [1,1,l,l]
    };

    let dense_weight = |prefix: &str, in_dim: usize, out_dim: usize| {
        b.constant(
            &format!("{prefix}.weight"),
            TensorType::new(vec![in_dim, out_dim], cfg.proj_dtype),
        )
    };

    // legalize (card 523a) hosts this gather off-device when the target's buffer limit needs it.
    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens); // [L, hidden]
    let mut x = b.reshape(emb, vec![1, l, h]); // [1,L,hidden]
    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let wk = dense_weight(&p("self_attn.k_proj"), h, kv_dim);
        let wv = dense_weight(&p("self_attn.v_proj"), h, kv_dim);
        let wo = dense_weight(&p("self_attn.o_proj"), q_dim, h);
        // Qwen2 has q/k/v bias; Qwen3 does not.
        let (bq, bk, bv) = if cfg.qkv_bias {
            (
                Some(b.constant(&p("self_attn.q_proj.bias"), TensorType::f32(vec![q_dim]))),
                Some(b.constant(&p("self_attn.k_proj.bias"), TensorType::f32(vec![kv_dim]))),
                Some(b.constant(&p("self_attn.v_proj.bias"), TensorType::f32(vec![kv_dim]))),
            )
        } else {
            (None, None, None)
        };

        let wq = dense_weight(&p("self_attn.q_proj"), h, q_dim);
        let q = linear(&b, normed, wq, bq); // [1,L,q_dim]
        let k = linear(&b, normed, wk, bk);
        let v = linear(&b, normed, wv, bv);

        let mut q = b.reshape(q, vec![1, l, hq, d]);
        let mut k = b.reshape(k, vec![1, l, hkv, d]);
        let v = b.reshape(v, vec![1, l, hkv, d]);
        // Qwen3 per-head QK-norm (RMSNorm over head_dim) before RoPE.
        if cfg.qk_norm {
            let qn = b.constant(&p("self_attn.q_norm.weight"), TensorType::f32(vec![d]));
            let kn = b.constant(&p("self_attn.k_norm.weight"), TensorType::f32(vec![d]));
            q = rmsnorm(&b, q, qn, cfg.eps);
            k = rmsnorm(&b, k, kn, cfg.eps);
        }
        let q = b.transpose(q, vec![0, 2, 1, 3]); // [1,Hq,L,D]
        let k = b.transpose(k, vec![0, 2, 1, 3]); // [1,Hkv,L,D]
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        // ALiBi (cfg.alibi): no rotation - the fixed per-head linear bias lives entirely in `mask`.
        let (q, k) = match rope_tables {
            Some((cos, sin)) => (
                rope_prefill(&b, q, cos, sin, l),
                rope_prefill(&b, k, cos, sin, l),
            ),
            None => (q, k),
        };

        // `compile`'s `flash_attention_capped` forms the flash op from this chain where it fits.
        let attn =
            attention_prefill_softcap(&b, q, k, v, n_rep, scale, mask, cfg.attn_logit_softcap); // [1,Hq,L,D]
        let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [1,L,Hq,D]
        let attn = b.reshape(attn, vec![1, l, q_dim]);
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let wg = dense_weight(&p("mlp.gate_proj"), h, cfg.inter);
        let wu = dense_weight(&p("mlp.up_proj"), h, cfg.inter);
        let wd = dense_weight(&p("mlp.down_proj"), cfg.inter, h);
        let gate = linear(&b, normed, wg, None);
        let up = linear(&b, normed, wu, None);
        let act = swiglu(&b, gate, up);
        let mlp = linear(&b, act, wd, None);
        x = b.binary(BinOp::Add, x, mlp);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps); // [1,L,hidden]
    // only the last position's logits matter for greedy next-token.
    let last = b.slice(x, 1, l - 1, l); // [1,1,hidden]
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None); // [1,1,vocab]
    let logits = match cfg.final_logit_softcap {
        Some(c) => softcap(&b, logits, c),
        None => logits,
    };

    b.finish(logits)
}

/// Qwen2-VL/2.5-VL text-tower prefill with sectioned ("mRoPE") position embedding (spec 267, card 152).
/// The text tower is plain Qwen2 (see [`Qwen2Config::mrope_section`]);
/// the only difference from [`trace_prefill`] is RoPE: instead of the implicit `0..L` positions, HF's
/// `apply_multimodal_rotary_pos_emb` gathers three per-token position ids (temporal/height/width, from
/// `get_rope_index`) and partitions the frequency axis by `cfg.mrope_section`, which is
/// [`rope_sectioned`]'s parameterization (decomposition proof in `poot-eval`'s `mrope.rs`).
///
/// Position ids are one explicit per-prompt `Slot::MropePosition` input shaped `[3,L]`, since mRoPE
/// height/width ids are not contiguous within an image/video grid. [`split_mrope_position_axes`]
/// decomposes it with ordinary graph primitives and feeds the shared `rope_sectioned`.
///
/// For an all-text prompt all three axes are the same running counter `0..L`, and the output is
/// bit-exact to [`trace_prefill`] (checked by `qwen2_pipeline`'s CPU-oracle test).
pub fn trace_prefill_mrope(cfg: Qwen2Config, seq_len: usize) -> Graph {
    let sections = cfg
        .mrope_section
        .expect("trace_prefill_mrope needs cfg.mrope_section (Some([t,h,w]))");
    assert!(
        cfg.qkv_bias && !cfg.qk_norm,
        "trace_prefill_mrope currently supports only the Qwen2-VL text-tower variant (bias, no QK-norm), \
         matching HF's Qwen2VLAttention (modeling_qwen2_vl.py) - see Qwen2Config::mrope_section's doc \
         comment"
    );
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();
    let l = seq_len;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let positions = b.slot(Slot::MropePosition, TensorType::new(vec![3, l], DType::I32));
    let [pos_t, pos_h, pos_w] = split_mrope_position_axes(&b, positions, &[l]);
    // The causal mask follows linear sequence order (card 550), not the mrope temporal/height/width
    // RoPE angle positions above: this one-shot prefill always starts at position 0.
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, l], DType::I32));
    let mask = causal_mask_from_pos(&b, pos, l, cfg.sliding_window);

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens); // [L, hidden]
    let mut x = b.reshape(emb, vec![1, l, h]); // [1,L,hidden]

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
        let bq = b.constant(&p("self_attn.q_proj.bias"), TensorType::f32(vec![q_dim]));
        let bk = b.constant(&p("self_attn.k_proj.bias"), TensorType::f32(vec![kv_dim]));
        let bv = b.constant(&p("self_attn.v_proj.bias"), TensorType::f32(vec![kv_dim]));

        let q = linear(&b, normed, wq, Some(bq)); // [1,L,q_dim]
        let k = linear(&b, normed, wk, Some(bk));
        let v = linear(&b, normed, wv, Some(bv));

        let q = b.reshape(q, vec![1, l, hq, d]);
        let k = b.reshape(k, vec![1, l, hkv, d]);
        let v = b.reshape(v, vec![1, l, hkv, d]);
        let q = b.transpose(q, vec![0, 2, 1, 3]); // [1,Hq,L,D]
        let k = b.transpose(k, vec![0, 2, 1, 3]); // [1,Hkv,L,D]
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        let q = rope_sectioned(&b, q, cos, sin, &[pos_t, pos_h, pos_w], &sections);
        let k = rope_sectioned(&b, k, cos, sin, &[pos_t, pos_h, pos_w], &sections);

        let attn =
            attention_prefill_softcap(&b, q, k, v, n_rep, scale, mask, cfg.attn_logit_softcap); // [1,Hq,L,D]
        let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [1,L,Hq,D]
        let attn = b.reshape(attn, vec![1, l, q_dim]);
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
    let x = rmsnorm(&b, x, norm, cfg.eps); // [1,L,hidden]
    // only the last position's logits matter for greedy next-token.
    let last = b.slice(x, 1, l - 1, l); // [1,1,hidden]
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None); // [1,1,vocab]
    let logits = match cfg.final_logit_softcap {
        Some(c) => softcap(&b, logits, c),
        None => logits,
    };

    b.finish(logits)
}

fn trace_prefill_kv_impl(
    cfg: Qwen2Config,
    n: usize,
    cap: usize,
    from_embeds: bool,
    mrope_sections: Option<[usize; 3]>,
) -> Graph {
    assert!(
        cap >= n,
        "cache capacity {cap} must be at least the prompt length {n}"
    );
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();
    let l = n;

    // the prompt token ids (one per position), a [N] vector input (as trace_prefill).
    // VLM path (spec 049): take input embeddings directly instead of token ids + embedding gather.
    let tokens = if from_embeds {
        None
    } else {
        Some(b.slot(Slot::Token, TensorType::new(vec![l], DType::I32)))
    };
    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    // this one-shot prefill always starts at position 0 (card 550): the cache write always fills from slot 0.
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, l], DType::I32));
    let mask = causal_mask_from_pos(&b, pos, l, cfg.sliding_window);
    let mrope_positions = mrope_sections.map(|_| {
        let packed = b.slot(Slot::MropePosition, TensorType::new(vec![3, l], DType::I32));
        split_mrope_position_axes(&b, packed, &[l])
    });

    let mut x = if from_embeds {
        // the spliced image+text embeddings, provided per-request.
        let emb = b.slot_named(
            Slot::Activation,
            "vlm.input_embeds",
            TensorType::f32(vec![l, h]),
        );
        b.reshape(emb, vec![1, l, h])
    } else {
        // legalize (card 523a) hosts this gather off-device when the target's buffer limit needs it.
        let embed = b.constant(
            "model.embed_tokens.weight",
            TensorType::f32(vec![cfg.vocab, h]),
        );
        let emb = b.gather(embed, 0, tokens.unwrap()); // [N, hidden]
        b.reshape(emb, vec![1, l, h]) // [1,N,hidden]
    };

    // (state_in, state_out) cache pairs, accumulated across layers (same order/names as the masked decode).
    let mut state: Vec<(poot_graph_ir::Traced, poot_graph_ir::Traced)> =
        Vec::with_capacity(2 * cfg.layers);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let proj = |x, prefix: &str, in_dim, out_dim, bias| {
            let w = b.constant(
                &p(&format!("{prefix}.weight")),
                TensorType::new(vec![in_dim, out_dim], cfg.proj_dtype),
            );
            linear(&b, x, w, bias)
        };
        let (bq, bk, bv) = if cfg.qkv_bias {
            (
                Some(b.constant(&p("self_attn.q_proj.bias"), TensorType::f32(vec![q_dim]))),
                Some(b.constant(&p("self_attn.k_proj.bias"), TensorType::f32(vec![kv_dim]))),
                Some(b.constant(&p("self_attn.v_proj.bias"), TensorType::f32(vec![kv_dim]))),
            )
        } else {
            (None, None, None)
        };

        let q = proj(normed, "self_attn.q_proj", h, q_dim, bq); // [1,N,q_dim]
        let k = proj(normed, "self_attn.k_proj", h, kv_dim, bk);
        let v = proj(normed, "self_attn.v_proj", h, kv_dim, bv);

        let mut q = b.reshape(q, vec![1, l, hq, d]);
        let mut k = b.reshape(k, vec![1, l, hkv, d]);
        let v = b.reshape(v, vec![1, l, hkv, d]);
        if cfg.qk_norm {
            let qn = b.constant(&p("self_attn.q_norm.weight"), TensorType::f32(vec![d]));
            let kn = b.constant(&p("self_attn.k_norm.weight"), TensorType::f32(vec![d]));
            q = rmsnorm(&b, q, qn, cfg.eps);
            k = rmsnorm(&b, k, kn, cfg.eps);
        }
        let q = b.transpose(q, vec![0, 2, 1, 3]); // [1,Hq,N,D]
        let k = b.transpose(k, vec![0, 2, 1, 3]); // [1,Hkv,N,D]
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        let (q, k) = match (mrope_sections, mrope_positions) {
            (Some(sections), Some([pos_t, pos_h, pos_w])) => (
                rope_sectioned(&b, q, cos, sin, &[pos_t, pos_h, pos_w], &sections),
                rope_sectioned(&b, k, cos, sin, &[pos_t, pos_h, pos_w], &sections),
            ),
            _ => (
                rope_prefill(&b, q, cos, sin, l),
                rope_prefill(&b, k, cos, sin, l),
            ),
        };

        // fill the carried cache: scatter the whole [1,Hkv,N,D] block into the zero-seeded [1,Hkv,cap,D]
        // cache at slot 0 (axis 2 = the seq axis). Attention reads the fresh k/v directly.
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
        let write_cache = |cache: Traced, newkv: Traced| b.dynamic_update_slice(cache, newkv, 0, 2);
        let kcache_out = write_cache(kcache, k);
        let vcache_out = write_cache(vcache, v);
        state.push((kcache, kcache_out));
        state.push((vcache, vcache_out));

        // causal attention over the prefix [0..N) using k/v directly (equal to reading the filled prefix;
        // slots past N are never read since attention is over N keys).
        let attn =
            attention_prefill_softcap(&b, q, k, v, n_rep, scale, mask, cfg.attn_logit_softcap); // [1,Hq,N,D]
        let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [1,N,Hq,D]
        let attn = b.reshape(attn, vec![1, l, q_dim]);
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
    let x = rmsnorm(&b, x, norm, cfg.eps); // [1,N,hidden]
    let last = b.slice(x, 1, l - 1, l); // [1,1,hidden]
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None); // [1,1,vocab]

    b.finish_with_state(logits, &state)
}
