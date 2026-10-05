use super::*;
use poot_graph_ir::Operand;
use std::collections::HashMap;

fn assert_same_operand(label: &str, left: Operand, right: Operand) {
    match (left, right) {
        (Operand::Value(left), Operand::Value(right)) => assert_eq!(left, right, "{label}"),
        (Operand::Lit(left), Operand::Lit(right)) => assert_eq!(left, right, "{label}"),
        (left, right) => panic!("{label}: operand kind differs: {left:?} != {right:?}"),
    }
}

fn assert_same_graph(label: &str, wrapper: &Graph, expected: &Graph) {
    wrapper
        .validate()
        .unwrap_or_else(|error| panic!("{label}: wrapper graph is invalid: {error}"));
    expected
        .validate()
        .unwrap_or_else(|error| panic!("{label}: expected graph is invalid: {error}"));
    assert_eq!(wrapper.inputs, expected.inputs, "{label}: inputs");
    assert_eq!(wrapper.consts, expected.consts, "{label}: constants");
    assert_eq!(wrapper.slots, expected.slots, "{label}: slots");
    assert_eq!(wrapper.output, expected.output, "{label}: output id");
    assert_eq!(wrapper.state, expected.state, "{label}: state order");
    assert_eq!(
        wrapper.values.len(),
        expected.values.len(),
        "{label}: values"
    );
    for (index, (left, right)) in wrapper.values.iter().zip(&expected.values).enumerate() {
        assert_eq!(left.aval, right.aval, "{label}: value {index} type");
        assert_eq!(
            left.storage, right.storage,
            "{label}: value {index} storage"
        );
        assert_eq!(left.name, right.name, "{label}: value {index} name");
    }
    assert_eq!(
        wrapper.eqns.len(),
        expected.eqns.len(),
        "{label}: equations"
    );
    for (index, (left, right)) in wrapper.eqns.iter().zip(&expected.eqns).enumerate() {
        assert_eq!(left.op, right.op, "{label}: equation {index} op");
        assert_eq!(left.out, right.out, "{label}: equation {index} output");
        assert_eq!(
            left.inputs.len(),
            right.inputs.len(),
            "{label}: equation {index} input count"
        );
        for (operand, (&left, &right)) in left.inputs.iter().zip(&right.inputs).enumerate() {
            assert_same_operand(
                &format!("{label}: equation {index} operand {operand}"),
                left,
                right,
            );
        }
    }
}

// Test-only reconstruction of the parent wrapper graph from ordinary primitives. It must not call the extracted
// split/projected helpers, whose mutations are the regressions this side of the comparison must expose.
#[allow(clippy::too_many_arguments)]
fn expected_gated_attention(
    b: &Builder,
    x: Traced,
    wq: Traced,
    wk: Traced,
    wv: Traced,
    wo: Traced,
    q_norm_w: Traced,
    k_norm_w: Traced,
    cos: Traced,
    sin: Traced,
    position_or_mask: Traced,
    k_cache_in: Traced,
    v_cache_in: Traced,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    prefill: bool,
    eps: f32,
) -> (Traced, Traced, Traced) {
    let seq = b.aval(x).shape[1];
    let n_rep = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_dim = n_heads * head_dim;

    let qg_flat = linear(b, x, wq, None);
    let qg = b.reshape(qg_flat, vec![1, seq, n_heads, 2 * head_dim]);
    let q_raw = b.slice(qg, 3, 0, head_dim);
    let gate_raw = b.slice(qg, 3, head_dim, 2 * head_dim);
    let k_flat = linear(b, x, wk, None);
    let v_flat = linear(b, x, wv, None);

    let q_normed = rmsnorm(b, q_raw, q_norm_w, eps);
    let k_shaped = b.reshape(k_flat, vec![1, seq, n_kv_heads, head_dim]);
    let k_normed = rmsnorm(b, k_shaped, k_norm_w, eps);
    let v_shaped = b.reshape(v_flat, vec![1, seq, n_kv_heads, head_dim]);
    let q4 = b.transpose(q_normed, vec![0, 2, 1, 3]);
    let k4 = b.transpose(k_normed, vec![0, 2, 1, 3]);
    let v4 = b.transpose(v_shaped, vec![0, 2, 1, 3]);

    let (attn, k_cache_out, v_cache_out) = if prefill {
        let q4 = rope_prefill(b, q4, cos, sin, seq);
        let k4 = rope_prefill(b, k4, cos, sin, seq);
        let k_cache_out = b.dynamic_update_slice(k_cache_in, k4, 0, 2);
        let v_cache_out = b.dynamic_update_slice(v_cache_in, v4, 0, 2);
        let attn = attention_prefill(b, q4, k4, v4, n_rep, scale, position_or_mask);
        (attn, k_cache_out, v_cache_out)
    } else {
        let q4 = rope(b, q4, cos, sin, position_or_mask);
        let k4 = rope(b, k4, cos, sin, position_or_mask);
        let k_cache_out = b.dynamic_update_slice(k_cache_in, k4, 1, 2);
        let v_cache_out = b.dynamic_update_slice(v_cache_in, v4, 1, 2);
        let k_valid = b.slice(k_cache_out, 2, 0, 2);
        let v_valid = b.slice(v_cache_out, 2, 0, 2);
        let attn = attention(b, q4, k_valid, v_valid, n_rep, scale);
        (attn, k_cache_out, v_cache_out)
    };

    let gate4 = b.transpose(gate_raw, vec![0, 2, 1, 3]);
    let gate_sig = sigmoid(b, gate4);
    let attn_gated = b.binary(BinOp::Mul, attn, gate_sig);
    let attn_back = b.transpose(attn_gated, vec![0, 2, 1, 3]);
    let attn_flat = b.reshape(attn_back, vec![1, seq, q_dim]);
    let out = linear(b, attn_flat, wo, None);
    (out, k_cache_out, v_cache_out)
}

// Same independent parent-graph oracle for GDN. Test-only; production keeps one implementation in the projected core.
#[allow(clippy::too_many_arguments)]
fn expected_gdn(
    b: &Builder,
    x: Traced,
    w_qkv: Traced,
    w_gate: Traced,
    w_conv: Traced,
    w_beta: Traced,
    w_alpha: Traced,
    dt_bias: Traced,
    ssm_a: Traced,
    norm_w: Traced,
    w_out: Traced,
    conv_cache_in: Traced,
    s_in: Traced,
    tril_incl: Traced,
    tril_strict: Traced,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
    conv_k: usize,
    chunk: usize,
    prefill: bool,
    eps: f32,
) -> (Traced, Traced, Traced) {
    let batch = b.aval(x).shape[0];
    let seq = b.aval(x).shape[1];
    let n_rep = num_v_heads / num_k_heads;
    let key_dim = num_k_heads * head_dim;
    let value_dim = num_v_heads * head_dim;
    let conv_dim = 2 * key_dim + value_dim;

    let qkv_mixed = linear(b, x, w_qkv, None);
    let z = linear(b, x, w_gate, None);
    let beta_raw = linear(b, x, w_beta, None);
    let beta = sigmoid(b, beta_raw);
    let alpha_raw = linear(b, x, w_alpha, None);
    let alpha = softplus(b, b.binary(BinOp::Add, alpha_raw, dt_bias));
    let g = b.binary(BinOp::Mul, alpha, ssm_a);

    let (o, conv_cache_out, s_out) = if prefill {
        let conv_out = causal_conv1d_prefill(b, qkv_mixed, w_conv, conv_k);
        let conv_out = poot_graph_ir::ops::silu(b, conv_out);
        let first = b.slice(qkv_mixed, 1, 0, 1);
        let zero_slice = b.binary_scalar(BinOp::Mul, first, Scalar::F32(0.0));
        let zeros = b.broadcast(zero_slice, vec![1, conv_k - 1, conv_dim]);
        let x_pad = b.concat(1, &[zeros, qkv_mixed]);
        let conv_cache_out = b.slice(x_pad, 1, seq, seq + conv_k - 1);

        let q_flat = b.slice(conv_out, 2, 0, key_dim);
        let k_flat = b.slice(conv_out, 2, key_dim, 2 * key_dim);
        let v_flat = b.slice(conv_out, 2, 2 * key_dim, conv_dim);
        let q = b.reshape(q_flat, vec![1, seq, num_k_heads, head_dim]);
        let q = b.transpose(q, vec![0, 2, 1, 3]);
        let k = b.reshape(k_flat, vec![1, seq, num_k_heads, head_dim]);
        let k = b.transpose(k, vec![0, 2, 1, 3]);
        let v = b.reshape(v_flat, vec![1, seq, num_v_heads, head_dim]);
        let v = b.transpose(v, vec![0, 2, 1, 3]);
        let q = l2_norm_last(b, q, eps);
        let k = l2_norm_last(b, k, eps);
        let g = b.reshape(g, vec![1, seq, num_v_heads, 1]);
        let g = b.transpose(g, vec![0, 2, 1, 3]);
        let beta = b.reshape(beta, vec![1, seq, num_v_heads, 1]);
        let beta = b.transpose(beta, vec![0, 2, 1, 3]);
        let (o, s_out) =
            gdn_prefill_chunked(b, q, k, v, g, beta, s_in, tril_incl, tril_strict, chunk);
        (o, conv_cache_out, s_out)
    } else {
        let (conv_out, conv_cache_out) =
            causal_conv1d_decode(b, qkv_mixed, w_conv, conv_cache_in, conv_k);
        let conv_out = poot_graph_ir::ops::silu(b, conv_out);
        let q_flat = b.slice(conv_out, 2, 0, key_dim);
        let k_flat = b.slice(conv_out, 2, key_dim, 2 * key_dim);
        let v_flat = b.slice(conv_out, 2, 2 * key_dim, conv_dim);
        let q = b.reshape(q_flat, vec![batch, num_k_heads, 1, head_dim]);
        let k = b.reshape(k_flat, vec![batch, num_k_heads, 1, head_dim]);
        let v = b.reshape(v_flat, vec![batch, num_v_heads, 1, head_dim]);
        let q = l2_norm_last(b, q, eps);
        let k = l2_norm_last(b, k, eps);
        let q = repeat_kv_tiled(b, q, n_rep);
        let k = repeat_kv_tiled(b, k, n_rep);
        let g = b.reshape(g, vec![batch, num_v_heads, 1, 1]);
        let beta = b.reshape(beta, vec![batch, num_v_heads, 1, 1]);
        let (o, s_out) = gated_delta_net_decode(b, q, k, v, g, beta, s_in);
        (o, conv_cache_out, s_out)
    };

    let o_norm = rmsnorm(b, o, norm_w, eps);
    let o_gated = if prefill {
        let z_shaped = b.reshape(z, vec![1, seq, num_v_heads, head_dim]);
        let z_shaped = b.transpose(z_shaped, vec![0, 2, 1, 3]);
        let silu_z = poot_graph_ir::ops::silu(b, z_shaped);
        b.binary(BinOp::Mul, o_norm, silu_z)
    } else {
        let z_shaped = b.reshape(z, vec![batch, num_v_heads, 1, head_dim]);
        let silu_z = poot_graph_ir::ops::silu(b, z_shaped);
        b.binary(BinOp::Mul, o_norm, silu_z)
    };
    let o_flat = if prefill {
        let o_back = b.transpose(o_gated, vec![0, 2, 1, 3]);
        b.reshape(o_back, vec![1, seq, value_dim])
    } else {
        b.reshape(o_gated, vec![batch, 1, value_dim])
    };
    let out = linear(b, o_flat, w_out, None);
    (out, conv_cache_out, s_out)
}

fn gated_attention_graph(prefill: bool, expected: bool) -> Graph {
    let (h, n_heads, n_kv_heads, head_dim, cap) = (6, 2, 1, 2, 3);
    let seq = if prefill { 2 } else { 1 };
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, seq, h]));
    let wq = b.constant("wq", TensorType::f32(vec![h, n_heads * 2 * head_dim]));
    let wk = b.constant("wk", TensorType::f32(vec![h, n_kv_heads * head_dim]));
    let wv = b.constant("wv", TensorType::f32(vec![h, n_kv_heads * head_dim]));
    let wo = b.constant("wo", TensorType::f32(vec![n_heads * head_dim, h]));
    let q_norm = b.constant("q_norm", TensorType::f32(vec![head_dim]));
    let k_norm = b.constant("k_norm", TensorType::f32(vec![head_dim]));
    let cos = b.constant("cos", TensorType::f32(vec![cap, head_dim]));
    let sin = b.constant("sin", TensorType::f32(vec![cap, head_dim]));
    let position_or_mask = if prefill {
        b.constant("mask", TensorType::f32(vec![1, 1, seq, seq]))
    } else {
        b.constant("pos", TensorType::scalar(DType::I32))
    };
    let k_cache = b.state_input(
        "k_cache",
        TensorType::f32(vec![1, n_kv_heads, cap, head_dim]),
        StateRole::Recurrent,
    );
    let v_cache = b.state_input(
        "v_cache",
        TensorType::f32(vec![1, n_kv_heads, cap, head_dim]),
        StateRole::Recurrent,
    );

    let (out, k_out, v_out) = match (prefill, expected) {
        (false, false) => qwen3next_gated_attention(
            &b,
            x,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            cos,
            sin,
            position_or_mask,
            k_cache,
            v_cache,
            n_heads,
            n_kv_heads,
            head_dim,
            1,
            1e-6,
        ),
        (true, false) => qwen3next_gated_attention_prefill(
            &b,
            x,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            cos,
            sin,
            position_or_mask,
            k_cache,
            v_cache,
            n_heads,
            n_kv_heads,
            head_dim,
            1e-6,
        ),
        (_, true) => expected_gated_attention(
            &b,
            x,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            cos,
            sin,
            position_or_mask,
            k_cache,
            v_cache,
            n_heads,
            n_kv_heads,
            head_dim,
            prefill,
            1e-6,
        ),
    };
    b.finish_with_state(out, &[(k_cache, k_out), (v_cache, v_out)])
}

fn gdn_graph(prefill: bool, expected: bool) -> Graph {
    let (h, num_k_heads, num_v_heads, head_dim, conv_k, chunk) = (6, 1, 2, 2, 2, 2);
    let seq = if prefill { 2 } else { 1 };
    let key_dim = num_k_heads * head_dim;
    let value_dim = num_v_heads * head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, seq, h]));
    let w_qkv = b.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let w_gate = b.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let w_conv = b.constant("w_conv", TensorType::f32(vec![conv_k, conv_dim]));
    let w_beta = b.constant("w_beta", TensorType::f32(vec![h, num_v_heads]));
    let w_alpha = b.constant("w_alpha", TensorType::f32(vec![h, num_v_heads]));
    let dt_bias = b.constant("dt_bias", TensorType::f32(vec![num_v_heads]));
    let ssm_a = b.constant("ssm_a", TensorType::f32(vec![num_v_heads]));
    let norm_w = b.constant("norm_w", TensorType::f32(vec![head_dim]));
    let w_out = b.constant("w_out", TensorType::f32(vec![value_dim, h]));
    let conv_cache = b.state_input(
        "conv_cache",
        TensorType::f32(vec![1, conv_k - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let ssm_state = b.state_input(
        "ssm_state",
        TensorType::f32(vec![1, num_v_heads, head_dim, head_dim]),
        StateRole::Recurrent,
    );
    let tril_incl = b.constant("tril_incl", TensorType::f32(vec![1, 1, chunk, chunk]));
    let tril_strict = b.constant("tril_strict", TensorType::f32(vec![1, 1, chunk, chunk]));

    let (out, conv_out, state_out) = if expected {
        expected_gdn(
            &b,
            x,
            w_qkv,
            w_gate,
            w_conv,
            w_beta,
            w_alpha,
            dt_bias,
            ssm_a,
            norm_w,
            w_out,
            conv_cache,
            ssm_state,
            tril_incl,
            tril_strict,
            num_k_heads,
            num_v_heads,
            head_dim,
            conv_k,
            chunk,
            prefill,
            1e-6,
        )
    } else if prefill {
        qwen3next_gdn_prefill_block(
            &b,
            x,
            w_qkv,
            w_gate,
            w_conv,
            w_beta,
            w_alpha,
            dt_bias,
            ssm_a,
            norm_w,
            w_out,
            ssm_state,
            tril_incl,
            tril_strict,
            num_k_heads,
            num_v_heads,
            head_dim,
            conv_k,
            chunk,
            1e-6,
            GdnHeadOrder::Tiled,
        )
    } else {
        qwen3next_gdn_block(
            &b,
            x,
            w_qkv,
            w_gate,
            w_conv,
            w_beta,
            w_alpha,
            dt_bias,
            ssm_a,
            norm_w,
            w_out,
            conv_cache,
            ssm_state,
            num_k_heads,
            num_v_heads,
            head_dim,
            conv_k,
            1e-6,
            GdnHeadOrder::Tiled,
        )
    };
    b.finish_with_state(out, &[(conv_cache, conv_out), (ssm_state, state_out)])
}

fn invalid_attention_wrapper_graph(prefill: bool) -> Graph {
    let (h, n_heads, n_kv_heads, head_dim, cap) = (4, 3, 2, 2, 3);
    let seq = if prefill { 2 } else { 1 };
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, seq, h]));
    let wq = b.constant("wq", TensorType::f32(vec![h, n_heads * 2 * head_dim]));
    let wk = b.constant("wk", TensorType::f32(vec![h, n_kv_heads * head_dim]));
    let wv = b.constant("wv", TensorType::f32(vec![h, n_kv_heads * head_dim]));
    let wo = b.constant("wo", TensorType::f32(vec![n_heads * head_dim, h]));
    let q_norm = b.constant("q_norm", TensorType::f32(vec![head_dim]));
    let k_norm = b.constant("k_norm", TensorType::f32(vec![head_dim]));
    let cos = b.constant("cos", TensorType::f32(vec![cap, head_dim]));
    let sin = b.constant("sin", TensorType::f32(vec![cap, head_dim]));
    let position_or_mask = if prefill {
        b.constant("mask", TensorType::f32(vec![1, 1, seq, seq]))
    } else {
        b.constant("pos", TensorType::scalar(DType::I32))
    };
    let k_cache = b.state_input(
        "k_cache",
        TensorType::f32(vec![1, n_kv_heads, cap, head_dim]),
        StateRole::Recurrent,
    );
    let v_cache = b.state_input(
        "v_cache",
        TensorType::f32(vec![1, n_kv_heads, cap, head_dim]),
        StateRole::Recurrent,
    );

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if prefill {
            let _ = qwen3next_gated_attention_prefill(
                &b,
                x,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                cos,
                sin,
                position_or_mask,
                k_cache,
                v_cache,
                n_heads,
                n_kv_heads,
                head_dim,
                1e-6,
            );
        } else {
            let _ = qwen3next_gated_attention(
                &b,
                x,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                cos,
                sin,
                position_or_mask,
                k_cache,
                v_cache,
                n_heads,
                n_kv_heads,
                head_dim,
                1,
                1e-6,
            );
        }
    }));
    assert!(result.is_err(), "invalid attention heads were accepted");
    b.finish(x)
}

fn invalid_gdn_wrapper_graph(prefill: bool) -> Graph {
    let (h, num_k_heads, num_v_heads, head_dim, conv_k, chunk) = (4, 2, 3, 2, 2, 2);
    let seq = if prefill { 2 } else { 1 };
    let key_dim = num_k_heads * head_dim;
    let value_dim = num_v_heads * head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, seq, h]));
    let w_qkv = b.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let w_gate = b.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let w_conv = b.constant("w_conv", TensorType::f32(vec![conv_k, conv_dim]));
    let w_beta = b.constant("w_beta", TensorType::f32(vec![h, num_v_heads]));
    let w_alpha = b.constant("w_alpha", TensorType::f32(vec![h, num_v_heads]));
    let dt_bias = b.constant("dt_bias", TensorType::f32(vec![num_v_heads]));
    let ssm_a = b.constant("ssm_a", TensorType::f32(vec![num_v_heads]));
    let norm_w = b.constant("norm_w", TensorType::f32(vec![head_dim]));
    let w_out = b.constant("w_out", TensorType::f32(vec![value_dim, h]));
    let conv_cache = b.state_input(
        "conv_cache",
        TensorType::f32(vec![1, conv_k - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let ssm_state = b.state_input(
        "ssm_state",
        TensorType::f32(vec![1, num_v_heads, head_dim, head_dim]),
        StateRole::Recurrent,
    );
    let tril_incl = b.constant("tril_incl", TensorType::f32(vec![1, 1, chunk, chunk]));
    let tril_strict = b.constant("tril_strict", TensorType::f32(vec![1, 1, chunk, chunk]));

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if prefill {
            let _ = qwen3next_gdn_prefill_block(
                &b,
                x,
                w_qkv,
                w_gate,
                w_conv,
                w_beta,
                w_alpha,
                dt_bias,
                ssm_a,
                norm_w,
                w_out,
                ssm_state,
                tril_incl,
                tril_strict,
                num_k_heads,
                num_v_heads,
                head_dim,
                conv_k,
                chunk,
                1e-6,
                GdnHeadOrder::Tiled,
            );
        } else {
            let _ = qwen3next_gdn_block(
                &b,
                x,
                w_qkv,
                w_gate,
                w_conv,
                w_beta,
                w_alpha,
                dt_bias,
                ssm_a,
                norm_w,
                w_out,
                conv_cache,
                ssm_state,
                num_k_heads,
                num_v_heads,
                head_dim,
                conv_k,
                1e-6,
                GdnHeadOrder::Tiled,
            );
        }
    }));
    assert!(result.is_err(), "invalid GDN heads were accepted");
    b.finish(x)
}

#[test]
fn qwen3next_projection_wrappers_preserve_graphs() {
    for (label, wrapper, expected) in [
        (
            "attention decode",
            gated_attention_graph(false, false),
            gated_attention_graph(false, true),
        ),
        (
            "attention prefill",
            gated_attention_graph(true, false),
            gated_attention_graph(true, true),
        ),
        (
            "GDN decode",
            gdn_graph(false, false),
            gdn_graph(false, true),
        ),
        ("GDN prefill", gdn_graph(true, false), gdn_graph(true, true)),
    ] {
        assert_same_graph(label, &wrapper, &expected);
    }
}

#[test]
fn gdn_head_order_pairs_value_heads_with_source_key_heads() {
    // Two key heads expanded to six value heads. Tiled pairs value head j with key head j % 2;
    // grouped (HF `repeat_interleave`) pairs it with key head j / 3.
    for (order, expected) in [
        (
            GdnHeadOrder::Tiled,
            [10.0_f32, 20.0, 10.0, 20.0, 10.0, 20.0],
        ),
        (
            GdnHeadOrder::Grouped,
            [10.0_f32, 10.0, 10.0, 20.0, 20.0, 20.0],
        ),
    ] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 2, 1, 1]));
        let expanded = order.expand(&b, x, 3);
        let graph = b.finish(expanded);
        let inputs: HashMap<_, poot_eval::Value> = HashMap::from([(
            x.id,
            poot_tensor::HostTensor::f32(vec![1, 2, 1, 1], vec![10.0, 20.0]).into(),
        )]);
        let got = poot_eval::eval(
            &graph,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("head expansion evaluates")
        .output
        .into_host()
        .expect("dense output");
        assert_eq!(got.shape(), [1, 6, 1, 1], "{order:?}");
        assert_eq!(got.as_f32().unwrap(), expected.as_slice(), "{order:?}");
    }
}

#[test]
fn qwen3next_public_wrappers_validate_heads_before_graph_mutation() {
    for (label, graph) in [
        ("attention decode", invalid_attention_wrapper_graph(false)),
        ("attention prefill", invalid_attention_wrapper_graph(true)),
        ("GDN decode", invalid_gdn_wrapper_graph(false)),
        ("GDN prefill", invalid_gdn_wrapper_graph(true)),
    ] {
        assert!(
            graph.eqns.is_empty(),
            "{label} mutated the graph before rejecting invalid heads"
        );
    }
}
