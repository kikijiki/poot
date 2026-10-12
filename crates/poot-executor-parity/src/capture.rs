//! The capture-and-replay toy graphs: one decode step over the three op families a captured decode
//! needs to hold across steps. Masked full attention over a fixed-capacity KV cache (RoPE at the
//! `Pos` slot), one Gated-DeltaNet layer (a causal conv1d feeding the delta-rule recurrence), and one
//! mixture-of-experts layer (the `ArgTopK` router feeding indexed expert matmuls). The carried state
//! is the two KV caches, the conv window and the recurrent matrix.

use poot_graph_ir::ops::{
    attention_masked_softcap, causal_conv1d_decode, causal_mask_from_pos, gated_delta_net_decode,
    linear, moe_grouped, rmsnorm, rope, sigmoid, silu, softplus,
};
use poot_graph_ir::{BinOp, Builder, IndexRange, Slot, StateRole, TensorType, UnOp};
use poot_tensor::{DType, HostTensor};

use poot_eval::test_support::StepFixture;

use crate::{Fixture, store_with};

/// Hidden size, which is also the attention head dim (one head), the GDN `HV * DG` and the MoE
/// intermediate width.
const H: usize = 32;
/// KV cache capacity, and the number of decode steps: the run fills the cache exactly.
const CAP: usize = 6;
const MAX_POS: usize = 16;
const VOCAB: usize = 8;
/// GDN conv kernel width, value heads, and per-head state dim.
const GK: usize = 4;
const HV: usize = 2;
const DG: usize = H / HV;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const EPS: f32 = 1.0e-5;

/// The GDN and MoE toy: no projection bias.
pub fn gdn_moe_toy_fixture() -> Fixture {
    toy_fixture("capture_gdn_moe_toy", false)
}

/// The bias toy: the same program with biased Q/K/V projections, so the matmul-plus-bias contraction
/// runs inside the recorded program on every step (the qwen2 `qkv_bias` shape).
pub fn bias_toy_fixture() -> Fixture {
    toy_fixture("capture_bias_toy", true)
}

fn toy_fixture(name: &'static str, bias: bool) -> Fixture {
    let b = Builder::for_tests();
    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32)).unwrap();
    let pos = b
        .slot_ranged(
            Slot::Pos,
            TensorType::scalar(DType::I32),
            IndexRange::below(CAP).expect("a nonempty cache"),
        )
        .unwrap();
    let mask4 = causal_mask_from_pos(&b, b.reshape(pos, vec![1, 1]).unwrap(), CAP, None).unwrap();

    let embed = b
        .constant("toy.embed", TensorType::f32(vec![VOCAB, H]))
        .unwrap();
    let mut x = b
        .reshape(b.gather_scalar(embed, 0, token).unwrap(), vec![1, 1, H])
        .unwrap();

    // Masked full attention, one head of dim H, RoPE and a fixed-capacity KV cache.
    let attn_norm = b.constant("attn.norm", TensorType::f32(vec![H])).unwrap();
    let xn = rmsnorm(&b, x, attn_norm, EPS).unwrap();
    let projection = |tag: &str| {
        let w = b
            .constant(&format!("attn.w{tag}"), TensorType::f32(vec![H, H]))
            .unwrap();
        let bias = bias.then(|| {
            b.constant(&format!("attn.b{tag}"), TensorType::f32(vec![H]))
                .unwrap()
        });
        b.reshape(linear(&b, xn, w, bias).unwrap(), vec![1, 1, 1, H])
            .unwrap()
    };
    let (q, k, v) = (projection("q"), projection("k"), projection("v"));
    let wo = b.constant("attn.wo", TensorType::f32(vec![H, H])).unwrap();
    let cos = b
        .constant("attn.rope.cos", TensorType::f32(vec![MAX_POS, H]))
        .unwrap();
    let sin = b
        .constant("attn.rope.sin", TensorType::f32(vec![MAX_POS, H]))
        .unwrap();
    let (q, k) = (
        rope(&b, q, cos, sin, pos).unwrap(),
        rope(&b, k, cos, sin, pos).unwrap(),
    );
    let cache = |name: &str| {
        b.state_input(
            name,
            TensorType::f32(vec![1, 1, CAP, H]),
            StateRole::Positional { axis: 2 },
        )
        .unwrap()
    };
    let (kc, vc) = (cache("attn.k_cache"), cache("attn.v_cache"));
    let kc_out = b.dynamic_update_slice_dyn(kc, k, pos, 2).unwrap();
    let vc_out = b.dynamic_update_slice_dyn(vc, v, pos, 2).unwrap();
    let scale = 1.0 / (H as f32).sqrt();
    let attn = attention_masked_softcap(&b, q, kc_out, vc_out, 1, scale, mask4, None).unwrap();
    let attn = b.reshape(attn, vec![1, 1, H]).unwrap();
    x = b
        .binary(BinOp::Add, x, linear(&b, attn, wo, None).unwrap())
        .unwrap();

    // One Gated-DeltaNet layer: causal conv1d, then the delta-rule recurrence.
    let gdn_norm = b.constant("gdn.norm", TensorType::f32(vec![H])).unwrap();
    let xn2 = rmsnorm(&b, x, gdn_norm, EPS).unwrap();
    let conv_w = b
        .constant("gdn.conv_w", TensorType::f32(vec![GK, H]))
        .unwrap();
    let conv_cache = b
        .state_input(
            "gdn.conv_cache",
            TensorType::f32(vec![1, GK - 1, H]),
            StateRole::Recurrent,
        )
        .unwrap();
    let (conv, conv_cache_out) = causal_conv1d_decode(&b, xn2, conv_w, conv_cache, GK).unwrap();
    let conv = silu(&b, conv).unwrap();
    let w_gdn_in = b
        .constant("gdn.w_in", TensorType::f32(vec![H, 3 * H + 2 * HV]))
        .unwrap();
    let qkvgb = linear(&b, conv, w_gdn_in, None).unwrap();
    let head_rows = |lo: usize, hi: usize, last: usize| {
        b.reshape(b.slice(qkvgb, 2, lo, hi).unwrap(), vec![1, HV, 1, last])
            .unwrap()
    };
    let (q_raw, k_raw, v_raw) = (
        head_rows(0, H, DG),
        head_rows(H, 2 * H, DG),
        head_rows(2 * H, 3 * H, DG),
    );
    let g = b
        .unary(
            UnOp::Neg,
            softplus(&b, head_rows(3 * H, 3 * H + HV, 1)).unwrap(),
        )
        .unwrap();
    let beta = sigmoid(&b, head_rows(3 * H + HV, 3 * H + 2 * HV, 1)).unwrap();
    let ssm = b
        .state_input(
            "gdn.ssm_state",
            TensorType::f32(vec![1, HV, DG, DG]),
            StateRole::Recurrent,
        )
        .unwrap();
    let (gdn_o, ssm_out) = gated_delta_net_decode(&b, q_raw, k_raw, v_raw, g, beta, ssm).unwrap();
    let w_gdn_out = b
        .constant("gdn.w_out", TensorType::f32(vec![H, H]))
        .unwrap();
    let gdn_o = b.reshape(gdn_o, vec![1, 1, H]).unwrap();
    x = b
        .binary(BinOp::Add, x, linear(&b, gdn_o, w_gdn_out, None).unwrap())
        .unwrap();

    // One MoE layer: the `ArgTopK` router feeding indexed expert matmuls.
    let moe_norm = b.constant("moe.norm", TensorType::f32(vec![H])).unwrap();
    let xn3 = rmsnorm(&b, x, moe_norm, EPS).unwrap();
    let router = b
        .constant("moe.router", TensorType::f32(vec![H, EXPERTS]))
        .unwrap();
    let w_in = b
        .constant("moe.w_in", TensorType::f32(vec![EXPERTS, H, 2 * H]))
        .unwrap();
    let w_out = b
        .constant("moe.w_out", TensorType::f32(vec![EXPERTS, H, H]))
        .unwrap();
    let moe = moe_grouped(&b, xn3, router, w_in, w_out, EXPERTS, TOP_K, H).unwrap();
    x = b
        .binary(BinOp::Add, x, b.reshape(moe, vec![1, 1, H]).unwrap())
        .unwrap();

    // Final norm and unembedding.
    let final_norm = b.constant("final.norm", TensorType::f32(vec![H])).unwrap();
    let xf = b
        .reshape(rmsnorm(&b, x, final_norm, EPS).unwrap(), vec![1, H])
        .unwrap();
    let unembed = b
        .constant("unembed", TensorType::f32(vec![H, VOCAB]))
        .unwrap();
    let logits = linear(&b, xf, unembed, None).unwrap();

    let graph = b
        .finish(poot_graph_ir::Outputs::new(logits).with_state(&[
            (kc, kc_out),
            (vc, vc_out),
            (conv_cache, conv_cache_out),
            (ssm, ssm_out),
        ]))
        .unwrap();
    let store = store_with(&graph, toy_weights);
    let key = |id| graph.meta(id).slot_key().unwrap().clone();
    let steps = (0..CAP)
        .map(|step| {
            vec![
                StepFixture {
                    key: key(token.id()),
                    tensor: HostTensor::i32(vec![], vec![(step % VOCAB) as i32]),
                },
                StepFixture {
                    key: key(pos.id()),
                    tensor: HostTensor::i32(vec![], vec![step as i32]),
                },
            ]
        })
        .collect();
    Fixture {
        name,
        graph,
        store,
        steps,
    }
}

/// The toy's weights: the default hash-seeded fill is `+-0.1`, scaled per weight so the GDN state and
/// the MoE activations stay bounded over the six steps while the router still separates experts.
/// Norm weights are one, and the RoPE tables are real (angle growing with position, slower on the
/// higher frequencies) so the `Pos` slot reaches the attention scores.
fn toy_weights(name: &str, data: &mut [f32]) {
    let scale = match name {
        "toy.embed" | "moe.router" => 3.0,
        "unembed" | "gdn.w_out" => 6.0,
        "gdn.conv_w" => 5.0,
        "gdn.w_in" => 3.0,
        _ if name.starts_with("attn.w") || name.starts_with("moe.w") => 1.5,
        _ => 1.0,
    };
    match name {
        "attn.norm" | "gdn.norm" | "moe.norm" | "final.norm" => data.fill(1.0),
        "attn.rope.cos" | "attn.rope.sin" => {
            let half = H / 2;
            for (i, value) in data.iter_mut().enumerate() {
                let (position, column) = (i / H, i % H);
                let angle = position as f32 * 0.5 / (1 + column % half) as f32;
                *value = if name.ends_with("cos") {
                    angle.cos()
                } else {
                    angle.sin()
                };
            }
        }
        _ => data.iter_mut().for_each(|value| *value *= scale),
    }
}
