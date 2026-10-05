#[cfg(test)]
use super::*;

/// Trace a single V4 DECODE step for a mixed per-layer `layer_types` schedule. `schedule` gives one
/// [`V4LayerKind`] per layer (typically from [`parse_compress_ratios`]); each layer dispatches to the per-layer
/// body its single-attention-type tracer uses ([`deepseek4_sliding_layer_decode`],
/// [`deepseek4_hca_layer_decode`], [`deepseek4_csa_layer_decode`]), like
/// `crate::nemotron_h::trace_nemotron_h_hybrid_stack_prefill`.
///
/// The local sliding-window K==V branch and `local_mask` are built once, since every V4 layer keeps the local
/// branch. The rope tables (`"main"`/`rope_theta` for sliding layers, `"compress"`/`compress_rope_theta` for
/// CSA/HCA) and each compression kind's window-position/causal-validity constants are built only if that kind
/// appears in `schedule`. `cap` must be a multiple of `hca_compress_rate` when an HCA layer is present and of
/// `csa_compress_rate` when a CSA layer is present.
#[cfg(test)]
pub(crate) fn trace_deepseek4_hybrid_stack_decode(
    cfg: DeepseekV4Config,
    cap: usize,
    schedule: &[V4LayerKind],
) -> Result<Graph<ValidationOutputs>, DeepseekV4MoeError> {
    assert_eq!(
        schedule.len(),
        cfg.layers,
        "trace_deepseek4_hybrid_stack_decode: schedule length must equal cfg.layers"
    );
    let has_sliding = schedule.contains(&V4LayerKind::Sliding);
    let has_hca = schedule.contains(&V4LayerKind::Hca);
    let has_csa = schedule.contains(&V4LayerKind::Csa);
    if has_hca {
        assert!(
            cap.is_multiple_of(cfg.hca_compress_rate) && cfg.hca_compress_rate > 0,
            "trace_deepseek4_hybrid_stack_decode: cap must be an exact multiple of hca_compress_rate \
when the schedule includes an HCA layer"
        );
    }
    if has_csa {
        assert!(
            cap.is_multiple_of(cfg.csa_compress_rate) && cfg.csa_compress_rate > 0,
            "trace_deepseek4_hybrid_stack_decode: cap must be an exact multiple of csa_compress_rate \
when the schedule includes a CSA layer"
        );
    }

    let b = Builder::new();
    let (h, hc, rd) = (cfg.hidden, cfg.hc_mult, cfg.qk_rope_head_dim);

    // `[1]`, not a scalar: Card 372c's guard needs an axis to reduce, and Card 371's exact lane has no I32
    // reshape.
    let token = b.slot(Slot::Token, TensorType::new(vec![1], DType::I32));
    let (plan, dims, tokens, mut moe) =
        deepseek4_moe_preamble(&b, &cfg, schedule, token, V4MoePhase::Decode)?;
    let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let _seq_len = b.slot(Slot::SeqLen, TensorType::scalar(DType::I32));
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![cap]));
    let mask = b.reshape(mask, vec![1, 1, 1, cap]);

    let iota_full = b.iota(cfg.max_pos);
    let iota_cap = b.slice(iota_full, 0, 0, cap);
    let pos_f = b.gather_scalar(iota_full, 0, pos_slot);
    let local_mask =
        gemma4_local_window_floor(&b, mask, pos_f, iota_cap, cfg.sliding_window, 1, cap);

    let main_rope = has_sliding.then(|| {
        (
            b.constant(V4_ROPE_COS, TensorType::f32(vec![cfg.max_pos, rd / 2])),
            b.constant(V4_ROPE_SIN, TensorType::f32(vec![cfg.max_pos, rd / 2])),
        )
    });
    let compress_rope = (has_hca || has_csa).then(|| {
        (
            b.constant(
                V4_COMPRESS_ROPE_COS,
                TensorType::f32(vec![cfg.max_pos, rd / 2]),
            ),
            b.constant(
                V4_COMPRESS_ROPE_SIN,
                TensorType::f32(vec![cfg.max_pos, rd / 2]),
            ),
        )
    });

    let hca_info = has_hca.then(|| {
        let cr = cfg.hca_compress_rate;
        let n_windows = cap / cr;
        let win_positions = b.slot_named(
            Slot::Activation,
            V4_HCA_WINDOW_POSITIONS,
            TensorType::f32(vec![n_windows]),
        );
        let hca_mask = hca_decode_validity_mask(&b, pos_f, win_positions, cr);
        let hca_mask = b.reshape(hca_mask, vec![1, 1, 1, n_windows]);
        let combined_mask = b.concat(3, &[local_mask, hca_mask]);
        (combined_mask, win_positions, n_windows)
    });
    let csa_info = has_csa.then(|| {
        let cr = cfg.csa_compress_rate;
        let n_windows = cap / cr;
        let win_positions = b.slot_named(
            Slot::Activation,
            V4_CSA_WINDOW_POSITIONS,
            TensorType::f32(vec![n_windows]),
        );
        let csa_causal_mask = hca_decode_validity_mask(&b, pos_f, win_positions, cr);
        let csa_causal_mask = b.reshape(csa_causal_mask, vec![1, 1, 1, n_windows]);
        (csa_causal_mask, win_positions, n_windows)
    });

    let embed = v4_source(&b, plan.top_dense(V4TopDense::Embedding)?);
    let x0 = b.cast(b.gather(embed, 0, tokens), DType::F32);
    let x0 = b.reshape(x0, vec![1, 1, 1, h]);
    let mut x = b.broadcast(x0, vec![1, 1, hc, h]);

    let mut state: Vec<(Traced, Traced)> = Vec::new();

    for (li, kind) in schedule.iter().enumerate() {
        let _layer_scope = b.layer_scope(li);
        let (new_x, pairs) = match kind {
            V4LayerKind::Sliding => {
                let (cos, sin) = main_rope.expect("has_sliding implies main_rope is built");
                deepseek4_sliding_layer_decode(
                    &b,
                    &cfg,
                    li,
                    x,
                    cap,
                    pos_slot,
                    local_mask,
                    cos,
                    sin,
                    &plan,
                    dims,
                    &mut moe.context(),
                )?
            }
            V4LayerKind::Hca => {
                let (cos, sin) = compress_rope.expect("has_hca implies compress_rope is built");
                let (combined_mask, win_idx, n_windows) =
                    hca_info.expect("has_hca implies hca_info is built");
                deepseek4_hca_layer_decode(
                    &b,
                    &cfg,
                    li,
                    x,
                    cap,
                    pos_slot,
                    cos,
                    sin,
                    combined_mask,
                    win_idx,
                    n_windows,
                    &plan,
                    dims,
                    &mut moe.context(),
                )?
            }
            V4LayerKind::Csa => {
                let (cos, sin) = compress_rope.expect("has_csa implies compress_rope is built");
                let (csa_causal_mask, win_idx, n_windows) =
                    csa_info.expect("has_csa implies csa_info is built");
                deepseek4_csa_layer_decode(
                    &b,
                    &cfg,
                    li,
                    x,
                    cap,
                    pos_slot,
                    cos,
                    sin,
                    local_mask,
                    csa_causal_mask,
                    win_idx,
                    n_windows,
                    &plan,
                    dims,
                    &mut moe.context(),
                )?
            }
        };
        x = new_x;
        state.extend(pairs);
    }

    let hh_fn = v4_mhc_projection(&b, plan.top_dense(V4TopDense::HcHeadProjection)?);
    let hh_base = v4_source_f32(&b, plan.top_dense(V4TopDense::HcHeadBase)?);
    let hh_scale = v4_source_f32(&b, plan.top_dense(V4TopDense::HcHeadScale)?);
    let collapsed_final = hyper_head(
        &b, x, hh_fn, hh_base, hh_scale, hc, h, 1, cfg.eps, cfg.hc_eps,
    );

    let norm = v4_source_f32(&b, plan.top_dense(V4TopDense::FinalNorm)?);
    let xf = rmsnorm(&b, collapsed_final, norm, cfg.eps);
    let logits = v4_source_linear(&b, xf, plan.top_dense(V4TopDense::Head)?);
    moe.finish(b, logits, &state)
}

/// [`trace_deepseek4_hybrid_stack_decode`]'s PREFILL twin. The universal `[1,1,l,l]` windowed-causal `mask` is
/// built internally, as are `hca.block_bias`/`csa.block_bias` (`[1,1,l,n_windows]`), only for kinds present in
/// `schedule`. `seq_len` need not be a multiple of any active rate: the graph is padded internally (see
/// [`deepseek4_prefill_pad_len`] for why that cannot change a real position's output).
#[cfg(test)]
pub(crate) fn trace_deepseek4_hybrid_stack_prefill(
    cfg: DeepseekV4Config,
    seq_len: usize,
    schedule: &[V4LayerKind],
) -> Result<Graph<ValidationOutputs>, DeepseekV4MoeError> {
    assert_eq!(
        schedule.len(),
        cfg.layers,
        "trace_deepseek4_hybrid_stack_prefill: schedule length must equal cfg.layers"
    );
    let has_sliding = schedule.contains(&V4LayerKind::Sliding);
    let has_hca = schedule.contains(&V4LayerKind::Hca);
    let has_csa = schedule.contains(&V4LayerKind::Csa);
    // `Runner::generate`'s CPU re-prefill loop re-traces at the growing token count every step, so requiring
    // `seq_len` to be a multiple of the active rates would panic after one step on the real schedule (CSA rate
    // 4, HCA rate 128). Pad to the next valid length (see `deepseek4_prefill_pad_len`) and slice the final
    // logits at the real last token.
    let l = deepseek4_prefill_pad_len(seq_len, &cfg, has_hca, has_csa);

    let b = Builder::new();
    let (h, hc, rd) = (cfg.hidden, cfg.hc_mult, cfg.qk_rope_head_dim);

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let (plan, dims, tokens, mut moe) =
        deepseek4_moe_preamble(&b, &cfg, schedule, tokens, V4MoePhase::Prefill)?;
    let mask = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));

    let main_rope = has_sliding.then(|| {
        (
            b.constant(V4_ROPE_COS, TensorType::f32(vec![cfg.max_pos, rd / 2])),
            b.constant(V4_ROPE_SIN, TensorType::f32(vec![cfg.max_pos, rd / 2])),
        )
    });
    let compress_rope = (has_hca || has_csa).then(|| {
        (
            b.constant(
                V4_COMPRESS_ROPE_COS,
                TensorType::f32(vec![cfg.max_pos, rd / 2]),
            ),
            b.constant(
                V4_COMPRESS_ROPE_SIN,
                TensorType::f32(vec![cfg.max_pos, rd / 2]),
            ),
        )
    });

    let hca_info = has_hca.then(|| {
        let cr = cfg.hca_compress_rate;
        let n_windows = l / cr;
        let block_bias = b.slot_named(
            Slot::Activation,
            V4_HCA_BLOCK_BIAS,
            TensorType::f32(vec![1, 1, l, n_windows]),
        );
        let win_idx = b.slot_named(
            Slot::Activation,
            V4_HCA_WINDOW_POSITIONS,
            TensorType::f32(vec![n_windows]),
        );
        (block_bias, win_idx, n_windows)
    });
    let csa_info = has_csa.then(|| {
        let cr = cfg.csa_compress_rate;
        let n_windows = l / cr;
        let block_bias = b.slot_named(
            Slot::Activation,
            V4_CSA_BLOCK_BIAS,
            TensorType::f32(vec![1, 1, l, n_windows]),
        );
        let win_idx = b.slot_named(
            Slot::Activation,
            V4_CSA_WINDOW_POSITIONS,
            TensorType::f32(vec![n_windows]),
        );
        (block_bias, win_idx, n_windows)
    });

    let embed = v4_source(&b, plan.top_dense(V4TopDense::Embedding)?);
    let emb = b.cast(b.gather(embed, 0, tokens), DType::F32); // [L, hidden]
    let x0 = b.reshape(emb, vec![1, l, 1, h]);
    let mut x = b.broadcast(x0, vec![1, l, hc, h]);

    for (li, kind) in schedule.iter().enumerate() {
        let _layer_scope = b.layer_scope(li);
        x = match kind {
            V4LayerKind::Sliding => {
                let (cos, sin) = main_rope.expect("has_sliding implies main_rope is built");
                deepseek4_sliding_layer_prefill(
                    &b,
                    &cfg,
                    li,
                    x,
                    l,
                    mask,
                    cos,
                    sin,
                    &plan,
                    dims,
                    &mut moe.context(),
                )?
            }
            V4LayerKind::Hca => {
                let (cos, sin) = compress_rope.expect("has_hca implies compress_rope is built");
                let (block_bias, win_idx, n_windows) =
                    hca_info.expect("has_hca implies hca_info is built");
                let combined_mask = b.concat(3, &[mask, block_bias]);
                deepseek4_hca_layer_prefill(
                    &b,
                    &cfg,
                    li,
                    x,
                    l,
                    cos,
                    sin,
                    combined_mask,
                    win_idx,
                    n_windows,
                    &plan,
                    dims,
                    &mut moe.context(),
                )?
            }
            V4LayerKind::Csa => {
                let (cos, sin) = compress_rope.expect("has_csa implies compress_rope is built");
                let (block_bias, win_idx, n_windows) =
                    csa_info.expect("has_csa implies csa_info is built");
                deepseek4_csa_layer_prefill(
                    &b,
                    &cfg,
                    li,
                    x,
                    l,
                    cos,
                    sin,
                    mask,
                    block_bias,
                    win_idx,
                    n_windows,
                    &plan,
                    dims,
                    &mut moe.context(),
                )?
            }
        };
    }

    let hh_fn = v4_mhc_projection(&b, plan.top_dense(V4TopDense::HcHeadProjection)?);
    let hh_base = v4_source_f32(&b, plan.top_dense(V4TopDense::HcHeadBase)?);
    let hh_scale = v4_source_f32(&b, plan.top_dense(V4TopDense::HcHeadScale)?);
    let collapsed_final = hyper_head(
        &b, x, hh_fn, hh_base, hh_scale, hc, h, l, cfg.eps, cfg.hc_eps,
    );

    let norm = v4_source_f32(&b, plan.top_dense(V4TopDense::FinalNorm)?);
    let xf = rmsnorm(&b, collapsed_final, norm, cfg.eps);
    // Real last token, not the padded one.
    let last = b.slice(xf, 1, seq_len - 1, seq_len);
    let logits = v4_source_linear(&b, last, plan.top_dense(V4TopDense::Head)?);
    moe.finish(b, logits, &[])
}
