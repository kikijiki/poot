//! Gated delta net decode, qwen3next MoE FFN / gated attention / GDN block / decode trace references.

use crate::{EvalBudget, EvalError, EvalOptions, Value};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::gated_delta_net_decode;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Slot, StateRole, ValueId};
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::{assert_close_rel, max_abs_error, seed_of};

/// card 135c: gated delta-net decode oracle. The composition matches a hand-rolled recurrence per head run over L decode
/// steps. H=2 heads, D=4 per-head dim.
///
/// Reference: `delta-net-base.cpp` `build_delta_net_autoregressive` (doc sec 3.4):
///   q      = q / sqrt(D)
///   decay  = exp(g)
///   S      = S * decay
///   kv     = k @ S         (= S^T k)
///   delta  = beta * (v - kv)
///   S      = S + outer(k, delta)
///   o      = q @ S         (= S^T q)
#[test]
fn gated_delta_net_decode_matches_recurrence() {
    let (h, d, l) = (2usize, 4usize, 5usize);

    // deterministic synthetic inputs
    let qv = fill(h * l * d, 11);
    let kv = fill(h * l * d, 22);
    let vv = fill(h * l * d, 33);
    // g is the forget-gate arg; keep it negative so exp(g) is in (0,1).
    let gv: Vec<f32> = (0..h * l).map(|i| -0.3 - 0.1 * (i % 5) as f32).collect();
    // beta in (0,1)
    let betav: Vec<f32> = (0..h * l).map(|i| 0.5 + 0.1 * (i % 4) as f32).collect();

    // --- build the graph (one decode step, carries state) ---
    let b = Builder::new();
    let q_in = b.constant("q", TensorType::f32(vec![1, h, 1, d]));
    let k_in = b.constant("k", TensorType::f32(vec![1, h, 1, d]));
    let v_in = b.constant("v", TensorType::f32(vec![1, h, 1, d]));
    let g_in = b.constant("g", TensorType::f32(vec![1, h, 1, 1]));
    let bt_in = b.constant("beta", TensorType::f32(vec![1, h, 1, 1]));
    let s_in = b.state_input("s", TensorType::f32(vec![1, h, d, d]), StateRole::Recurrent);
    let (o, s_out) = gated_delta_net_decode(&b, q_in, k_in, v_in, g_in, bt_in, s_in);
    // check shape inference before b is consumed
    assert_eq!(b.aval(o).shape, vec![1, h, 1, d], "output shape");
    assert_eq!(b.aval(s_out).shape, vec![1, h, d, d], "state shape");
    let g = b.finish_with_state(o, &[(s_in, s_out)]);

    // --- run L decode steps ---
    let mut state = HostTensor::f32(vec![1, h, d, d], vec![0.0f32; h * d * d]);
    let mut got_outs = vec![0.0f32; h * l * d];

    for t in 0..l {
        let qt: Vec<f32> = (0..h * d)
            .map(|i| qv[(i / d * l + t) * d + i % d])
            .collect();
        let kt: Vec<f32> = (0..h * d)
            .map(|i| kv[(i / d * l + t) * d + i % d])
            .collect();
        let vt: Vec<f32> = (0..h * d)
            .map(|i| vv[(i / d * l + t) * d + i % d])
            .collect();
        let gt: Vec<f32> = (0..h).map(|hh| gv[hh * l + t]).collect();
        let btt: Vec<f32> = (0..h).map(|hh| betav[hh * l + t]).collect();

        let mut inp = HashMap::new();
        inp.insert(q_in.id, HostTensor::f32(vec![1, h, 1, d], qt));
        inp.insert(k_in.id, HostTensor::f32(vec![1, h, 1, d], kt));
        inp.insert(v_in.id, HostTensor::f32(vec![1, h, 1, d], vt));
        inp.insert(g_in.id, HostTensor::f32(vec![1, h, 1, 1], gt));
        inp.insert(bt_in.id, HostTensor::f32(vec![1, h, 1, 1], btt));
        inp.insert(s_in.id, state.clone());

        let (ot, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inp
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .unwrap();
        state = new_states.into_iter().next().unwrap();
        assert_eq!(ot.shape(), vec![1, h, 1, d]);
        for hh in 0..h {
            for dd in 0..d {
                got_outs[(hh * l + t) * d + dd] = ot.as_f32().unwrap()[hh * d + dd];
            }
        }
    }

    // --- hand-rolled reference recurrence ---
    let mut want_outs = vec![0.0f32; h * l * d];
    let mut ref_s = vec![0.0f32; h * d * d]; // S[h, i, j] = ref_s[h*d*d + i*d + j]

    for t in 0..l {
        for hh in 0..h {
            let q_h: Vec<f32> = (0..d)
                .map(|dd| qv[(hh * l + t) * d + dd] / (d as f32).sqrt())
                .collect();
            let k_h: Vec<f32> = (0..d).map(|dd| kv[(hh * l + t) * d + dd]).collect();
            let v_h: Vec<f32> = (0..d).map(|dd| vv[(hh * l + t) * d + dd]).collect();
            let g_h = gv[hh * l + t];
            let beta_h = betav[hh * l + t];

            let decay = g_h.exp();
            // decay the state
            for elem in ref_s[hh * d * d..(hh + 1) * d * d].iter_mut() {
                *elem *= decay;
            }
            // kv = k @ S  (= S^T k)
            let mut kv_h = vec![0.0f32; d];
            for j in 0..d {
                for i in 0..d {
                    kv_h[j] += k_h[i] * ref_s[hh * d * d + i * d + j];
                }
            }
            // delta = beta * (v - kv)
            let delta_h: Vec<f32> = (0..d).map(|j| beta_h * (v_h[j] - kv_h[j])).collect();
            // S += outer(k, delta)
            for i in 0..d {
                for j in 0..d {
                    ref_s[hh * d * d + i * d + j] += k_h[i] * delta_h[j];
                }
            }
            // o = q @ S
            for j in 0..d {
                let mut acc = 0.0f32;
                for i in 0..d {
                    acc += q_h[i] * ref_s[hh * d * d + i * d + j];
                }
                want_outs[(hh * l + t) * d + j] = acc;
            }
        }
    }

    assert_close_rel(&got_outs, &want_outs, 1e-5);
}

/// Card 188 / SC-001: `gated_delta_net_decode` (ops.rs:666-697) reads every shape off its inputs and never hardcodes a
/// leading batch dim (its `matmul` calls treat leading `[B,H,...]` dims as batch dims, `docs/graph-architecture.md` sec
/// 3.2). This CPU oracle proves it is batch-generic: pack `B` independent rows (distinct nonzero q/k/v/g/beta and distinct
/// nonzero starting state per row) into one call and check every row's `(o, s_out)` is bit-identical to an independent
/// B=1 call on that row (no cross-row leakage, no batch-order dependence). Swept at `B in {2,4,8}`.
/// One row's (q, k, v, g, beta, s_in) for [`gated_delta_net_decode_batch_generalizes_from_batch1`].
type GdnRow = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

#[test]
fn gated_delta_net_decode_batch_generalizes_from_batch1() {
    let (h, d) = (2usize, 3usize);

    for &b_batch in &[2usize, 4, 8] {
        // per-row distinct q/k/v/g/beta/s_in (row index folds into every seed so no two rows collide).
        let rows: Vec<GdnRow> = (0..b_batch)
            .map(|row| {
                let seed = row as u64 * 13;
                let q = fill(h * d, 1000 + seed);
                let k = fill(h * d, 2000 + seed);
                let v = fill(h * d, 3000 + seed);
                // g negative so decay = exp(g) is in (0,1).
                let g: Vec<f32> = fill(h, 4000 + seed)
                    .iter()
                    .map(|x| -x.abs() - 0.1)
                    .collect();
                // beta in (0,1).
                let beta: Vec<f32> = fill(h, 5000 + seed)
                    .iter()
                    .map(|x| 0.1 + x.abs() * 0.5)
                    .collect();
                let s_in = fill(h * d * d, 6000 + seed);
                (q, k, v, g, beta, s_in)
            })
            .collect();

        // --- B=1 reference: one independent call per row. ---
        let single: Vec<(Vec<f32>, Vec<f32>)> = rows
            .iter()
            .map(|(q, k, v, g, beta, s_in)| {
                let b = Builder::new();
                let q_in = b.constant("q", TensorType::f32(vec![1, h, 1, d]));
                let k_in = b.constant("k", TensorType::f32(vec![1, h, 1, d]));
                let v_in = b.constant("v", TensorType::f32(vec![1, h, 1, d]));
                let g_in = b.constant("g", TensorType::f32(vec![1, h, 1, 1]));
                let bt_in = b.constant("beta", TensorType::f32(vec![1, h, 1, 1]));
                let s0 =
                    b.state_input("s", TensorType::f32(vec![1, h, d, d]), StateRole::Recurrent);
                let (o, s_out) = gated_delta_net_decode(&b, q_in, k_in, v_in, g_in, bt_in, s0);
                let g_graph = b.finish_with_state(o, &[(s0, s_out)]);

                let mut inp = HashMap::new();
                inp.insert(q_in.id, HostTensor::f32(vec![1, h, 1, d], q.clone()));
                inp.insert(k_in.id, HostTensor::f32(vec![1, h, 1, d], k.clone()));
                inp.insert(v_in.id, HostTensor::f32(vec![1, h, 1, d], v.clone()));
                inp.insert(g_in.id, HostTensor::f32(vec![1, h, 1, 1], g.clone()));
                inp.insert(bt_in.id, HostTensor::f32(vec![1, h, 1, 1], beta.clone()));
                inp.insert(s0.id, HostTensor::f32(vec![1, h, d, d], s_in.clone()));
                let (o_t, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
                    let values: HashMap<ValueId, Value> = inp
                        .iter()
                        .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                        .collect();
                    let evaluation =
                        crate::eval(&g_graph, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
                    let state = evaluation
                        .state
                        .into_iter()
                        .map(Value::into_host)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((evaluation.output.into_host()?, state))
                })()
                .unwrap();
                let s_t = new_states.into_iter().next().unwrap();
                (
                    o_t.as_f32().unwrap().to_vec(),
                    s_t.as_f32().unwrap().to_vec(),
                )
            })
            .collect();

        // --- batched call: all B rows packed into ONE call. ---
        let b = Builder::new();
        let q_in = b.constant("q", TensorType::f32(vec![b_batch, h, 1, d]));
        let k_in = b.constant("k", TensorType::f32(vec![b_batch, h, 1, d]));
        let v_in = b.constant("v", TensorType::f32(vec![b_batch, h, 1, d]));
        let g_in = b.constant("g", TensorType::f32(vec![b_batch, h, 1, 1]));
        let bt_in = b.constant("beta", TensorType::f32(vec![b_batch, h, 1, 1]));
        let s0 = b.state_input(
            "s",
            TensorType::f32(vec![b_batch, h, d, d]),
            StateRole::Recurrent,
        );
        let (o, s_out) = gated_delta_net_decode(&b, q_in, k_in, v_in, g_in, bt_in, s0);
        assert_eq!(b.aval(o).shape, vec![b_batch, h, 1, d], "batched o shape");
        assert_eq!(
            b.aval(s_out).shape,
            vec![b_batch, h, d, d],
            "batched s_out shape"
        );
        let g_graph = b.finish_with_state(o, &[(s0, s_out)]);

        let flat = |pick: fn(&GdnRow) -> &Vec<f32>| -> Vec<f32> {
            rows.iter().flat_map(|r| pick(r).clone()).collect()
        };
        let mut inp = HashMap::new();
        inp.insert(
            q_in.id,
            HostTensor::f32(vec![b_batch, h, 1, d], flat(|r| &r.0)),
        );
        inp.insert(
            k_in.id,
            HostTensor::f32(vec![b_batch, h, 1, d], flat(|r| &r.1)),
        );
        inp.insert(
            v_in.id,
            HostTensor::f32(vec![b_batch, h, 1, d], flat(|r| &r.2)),
        );
        inp.insert(
            g_in.id,
            HostTensor::f32(vec![b_batch, h, 1, 1], flat(|r| &r.3)),
        );
        inp.insert(
            bt_in.id,
            HostTensor::f32(vec![b_batch, h, 1, 1], flat(|r| &r.4)),
        );
        inp.insert(
            s0.id,
            HostTensor::f32(vec![b_batch, h, d, d], flat(|r| &r.5)),
        );
        let (o_t, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inp
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation =
                crate::eval(&g_graph, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .unwrap();
        let s_t = new_states.into_iter().next().unwrap();

        let (o_stride, s_stride) = (h * d, h * d * d);
        let mut max_abs_o = 0.0f32;
        let mut max_abs_s = 0.0f32;
        for (row, (want_o, want_s)) in single.iter().enumerate() {
            let got_o = &o_t.as_f32().unwrap()[row * o_stride..(row + 1) * o_stride];
            let got_s = &s_t.as_f32().unwrap()[row * s_stride..(row + 1) * s_stride];
            assert_close_rel(got_o, want_o, 1e-5);
            assert_close_rel(got_s, want_s, 1e-5);
            max_abs_o = max_abs_o.max(max_abs_error(got_o, want_o));
            max_abs_s = max_abs_s.max(max_abs_error(got_s, want_s));
        }
        eprintln!(
            "gated_delta_net_decode_batch_generalizes_from_batch1: B={b_batch} max_abs(o)={max_abs_o} max_abs(s_out)={max_abs_s}"
        );
        assert!(
            max_abs_o < 1e-6 && max_abs_s < 1e-6,
            "B={b_batch}: batched row should be numerically identical to the B=1 call, \
             got max_abs_o={max_abs_o} max_abs_s={max_abs_s}"
        );
    }
}

/// Card 135c: Qwen3-Next MoE FFN + shared-expert block (CPU oracle). Tiny config: hidden=8, n_experts=4, top_k=2, inter=8,
/// seq_len=1. Uses `moe()` for routing plus a sigmoid-gated dense shared expert.
///
/// Router normalization (section 5.2): softmax(all n_experts logits) -> top_k ->
/// renormalize. This equals poot's `top_k_gate`: mask non-top-k to -1e30, softmax, so only top-k contribute.
#[test]
fn qwen3next_moe_ffn_matches_reference() {
    use poot_models::qwen3next::qwen3next_moe_ffn;

    let (h, e, k, i) = (8usize, 4usize, 2usize, 8usize);
    let l = 1usize; // decode step (sparse path)

    // --- synthetic inputs (deterministic) ---
    let x_data = fill(l * h, 1);
    let router_data = fill(h * e, 2);
    // w_in: [E, H, 2*I] -- fused gate||up per expert
    let w_in_data = fill(e * h * 2 * i, 3);
    // w_out: [E, I, H]
    let w_out_data = fill(e * i * h, 4);
    // shared expert weights
    let sg_data = fill(h * i, 5); // [H, I]
    let su_data = fill(h * i, 6); // [H, I]
    let sd_data = fill(i * h, 7); // [I, H]
    // shared gate input: [H, 1]
    let sgi_data = fill(h, 8);

    // --- build the graph ---
    // All constants are referenced by name and bound after the graph is built. `moe_sparse` computes its
    // own index range with `iota` (card 558a), an equation the oracle evaluates directly, so it needs no
    // name-binding entry.
    let b = Builder::new();
    let x_t = b.constant("x", TensorType::f32(vec![1, l, h]));
    let router_t = b.constant("router", TensorType::f32(vec![h, e]));
    let w_in_t = b.constant("w_in", TensorType::f32(vec![e, h, 2 * i]));
    let w_out_t = b.constant("w_out", TensorType::f32(vec![e, i, h]));
    let sg_t = b.constant("sg", TensorType::f32(vec![h, i]));
    let su_t = b.constant("su", TensorType::f32(vec![h, i]));
    let sd_t = b.constant("sd", TensorType::f32(vec![i, h]));
    let sgi_t = b.constant("sgi", TensorType::f32(vec![h, 1]));

    let out = qwen3next_moe_ffn(
        &b, x_t, router_t, w_in_t, w_out_t, sg_t, su_t, sd_t, sgi_t, e, k, i,
    );
    assert_eq!(b.aval(out).shape, vec![1, l, h], "output shape");
    let g = b.finish(out);

    // Build a name->data map for the graph's named constants.
    let mut named: HashMap<String, Vec<f32>> = HashMap::new();
    named.insert("x".into(), x_data.clone());
    named.insert("router".into(), router_data.clone());
    named.insert("w_in".into(), w_in_data.clone());
    named.insert("w_out".into(), w_out_data.clone());
    named.insert("sg".into(), sg_data.clone());
    named.insert("su".into(), su_data.clone());
    named.insert("sd".into(), sd_data.clone());
    named.insert("sgi".into(), sgi_data.clone());

    let inputs: HashMap<ValueId, poot_tensor::HostTensor> = g
        .inputs
        .iter()
        .map(|&id| {
            let name = g.meta(id).name.as_deref().unwrap();
            let data = named[name].clone();
            (
                id,
                poot_tensor::HostTensor::f32(g.aval(id).shape.clone(), data),
            )
        })
        .collect();

    let got = (|| -> Result<HostTensor, EvalError> {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?
            .output
            .into_host()
    })()
    .unwrap();
    assert_eq!(got.shape(), vec![1, l, h]);

    // --- hand-coded reference (plain Rust, matches section 5 of qwen35moe-forward-pass.md) ---

    // 1. Router: softmax over all n_experts, top-k, renormalize.
    let xm: Vec<f32> = x_data.clone(); // [L=1, H] -- same as x_data since l=1

    // logits[e] = x @ router_w[:, e] -- [L=1, H] @ [H, E] -> [E]
    let mut logits = vec![0.0f32; e];
    for ei in 0..e {
        for hi in 0..h {
            logits[ei] += xm[hi] * router_data[hi * e + ei];
        }
    }

    // softmax over all experts
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_l: Vec<f32> = logits.iter().map(|&v| (v - max_l).exp()).collect();
    let sum_exp: f32 = exp_l.iter().sum();
    let probs: Vec<f32> = exp_l.iter().map(|&v| v / sum_exp).collect();

    // top-k by index (highest prob = highest logit)
    let mut order: Vec<usize> = (0..e).collect();
    order.sort_by(|&a, &c| probs[c].partial_cmp(&probs[a]).unwrap());
    let top: Vec<usize> = order[..k].to_vec();

    // renormalize
    let top_sum: f32 = top.iter().map(|&ei| probs[ei]).sum();
    let gate_w: Vec<f32> = top.iter().map(|&ei| probs[ei] / top_sum).collect();

    // per-expert SwiGLU, weighted sum
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let mut moe_out_ref = vec![0.0f32; h];
    for (rank, &ex) in top.iter().enumerate() {
        // w_in[ex]: [H, 2*I] (contiguous block for this expert)
        let base_in = ex * h * 2 * i;
        let mut gu = vec![0.0f32; 2 * i];
        for ii in 0..2 * i {
            for hi in 0..h {
                gu[ii] += xm[hi] * w_in_data[base_in + hi * 2 * i + ii];
            }
        }
        let act: Vec<f32> = (0..i).map(|ii| silu(gu[ii]) * gu[i + ii]).collect();

        // w_out[ex]: [I, H]
        let base_out = ex * i * h;
        for hi in 0..h {
            let o: f32 = (0..i)
                .map(|ii| act[ii] * w_out_data[base_out + ii * h + hi])
                .sum();
            moe_out_ref[hi] += gate_w[rank] * o;
        }
    }

    // 2. Shared expert SwiGLU.
    // sg_data [H, I]: gate proj; su_data [H, I]: up proj
    let mut sg_out = vec![0.0f32; i];
    let mut su_out = vec![0.0f32; i];
    for ii in 0..i {
        for hi in 0..h {
            sg_out[ii] += xm[hi] * sg_data[hi * i + ii];
            su_out[ii] += xm[hi] * su_data[hi * i + ii];
        }
    }
    let shexp_act: Vec<f32> = (0..i).map(|ii| silu(sg_out[ii]) * su_out[ii]).collect();

    // sd_data [I, H]: down proj
    let mut shexp_out = vec![0.0f32; h];
    for hi in 0..h {
        for ii in 0..i {
            shexp_out[hi] += shexp_act[ii] * sd_data[ii * h + hi];
        }
    }

    // 3. Scalar sigmoid gate.
    // sgi_data [H, 1] -> dot with x -> scalar, sigmoid
    let g_raw: f32 = (0..h).map(|hi| xm[hi] * sgi_data[hi]).sum();
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let g_gate = sigmoid(g_raw);

    // 4. Combine: moe_out + g * shexp.
    let want: Vec<f32> = (0..h)
        .map(|hi| moe_out_ref[hi] + g_gate * shexp_out[hi])
        .collect();

    // shape is [1, L=1, H] -- got.data is flat [H]
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}

/// Card 135c: Qwen3-Next gated full-attention decode step (CPU oracle). Tiny config: n_q_heads=2, n_kv_heads=1,
/// head_dim=4, hidden=8, rotary_dim=2, pos=0, cap=4.
///
/// At pos=0 the RoPE cos/sin tables are [1,1] / [0,0] (identity rotation), keeping the reference simple while exercising
/// the per-head query/gate interleave split, QK-norm, GQA attention (n_rep=2), the sigmoid output gate, and the
/// o-projection.
///
/// Reference (section 4):
///   Qg = x @ wq -> [n_heads * 2 * head_dim]; split per-head into q and gate.
///   q_normed[h] = rmsnorm(q[h], q_norm_w); k_normed = rmsnorm(k, k_norm_w).
///   RoPE at pos=0: identity.
///   Attention (single key at pos=0, softmax([score]) = 1): attn[h] = v for each query head.
///   attn_gated[h] = attn[h] * sigmoid(gate[h]).
///   out = flatten(attn_gated) @ wo.
#[test]
fn qwen3next_gated_attention_matches_reference() {
    use poot_graph_ir::Storage;
    use poot_models::qwen3next::qwen3next_gated_attention;

    let (nh, nkv, hd, h_dim) = (2usize, 1usize, 4usize, 8usize); // n_q_heads, n_kv_heads, head_dim, hidden
    let rotary_dim = 2usize;
    let max_pos = 2usize;
    let cap = 4usize;
    let pos_idx = 0usize; // first decode step
    let eps = 1e-6f32;

    // Synthetic weights (deterministic).
    let x_data = fill(h_dim, 1); // [H=8]
    let wq_data = fill(h_dim * nh * 2 * hd, 2); // [H=8, n_heads*2*head_dim=16]
    let wk_data = fill(h_dim * nkv * hd, 3); // [H=8, n_kv_heads*head_dim=4]
    let wv_data = fill(h_dim * nkv * hd, 4); // [H=8, 4]
    let wo_data = fill(nh * hd * h_dim, 5); // [n_heads*head_dim=8, H=8]
    let q_norm_w_data = fill(hd, 6); // [head_dim=4]
    let k_norm_w_data = fill(hd, 7); // [head_dim=4]
    // RoPE tables [max_pos=2, rotary_dim=2]: pos=0 row is [1,1]/[0,0] (identity rotation).
    let cos_data: Vec<f32> = vec![1.0, 1.0, 0.5403, 0.8776];
    let sin_data: Vec<f32> = vec![0.0, 0.0, 0.8415, 0.4794];

    // Build the graph.
    let b = Builder::new();
    let x_t = b.constant("x", TensorType::f32(vec![1, 1, h_dim]));
    let wq_t = b.constant("wq", TensorType::f32(vec![h_dim, nh * 2 * hd]));
    let wk_t = b.constant("wk", TensorType::f32(vec![h_dim, nkv * hd]));
    let wv_t = b.constant("wv", TensorType::f32(vec![h_dim, nkv * hd]));
    let wo_t = b.constant("wo", TensorType::f32(vec![nh * hd, h_dim]));
    let qn_t = b.constant("qn", TensorType::f32(vec![hd]));
    let kn_t = b.constant("kn", TensorType::f32(vec![hd]));
    let cos_t = b.constant("cos", TensorType::f32(vec![max_pos, rotary_dim]));
    let sin_t = b.constant("sin", TensorType::f32(vec![max_pos, rotary_dim]));
    let pos_t = b.constant("pos", TensorType::f32(vec![])); // scalar, value=0 at eval
    let kc_t = b.state_input(
        "k_cache",
        TensorType::f32(vec![1, nkv, cap, hd]),
        StateRole::Recurrent,
    );
    let vc_t = b.state_input(
        "v_cache",
        TensorType::f32(vec![1, nkv, cap, hd]),
        StateRole::Recurrent,
    );

    let (out, kc_out, vc_out) = qwen3next_gated_attention(
        &b, x_t, wq_t, wk_t, wv_t, wo_t, qn_t, kn_t, cos_t, sin_t, pos_t, kc_t, vc_t, nh, nkv, hd,
        pos_idx, eps,
    );
    assert_eq!(b.aval(out).shape, vec![1, 1, h_dim], "output shape");
    let g = b.finish_with_state(out, &[(kc_t, kc_out), (vc_t, vc_out)]);

    // Bind all inputs.
    let mut named: HashMap<String, Vec<f32>> = HashMap::new();
    named.insert("x".into(), x_data.clone());
    named.insert("wq".into(), wq_data.clone());
    named.insert("wk".into(), wk_data.clone());
    named.insert("wv".into(), wv_data.clone());
    named.insert("wo".into(), wo_data.clone());
    named.insert("qn".into(), q_norm_w_data.clone());
    named.insert("kn".into(), k_norm_w_data.clone());
    named.insert("cos".into(), cos_data.clone());
    named.insert("sin".into(), sin_data.clone());
    named.insert("pos".into(), vec![0.0f32]); // scalar pos=0

    let mut inputs: HashMap<ValueId, poot_tensor::HostTensor> = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                let data = named[name].clone();
                inputs.insert(
                    id,
                    poot_tensor::HostTensor::f32(g.aval(id).shape.clone(), data),
                );
            }
            Storage::State => {} // bound below
            _ => panic!("unexpected storage in test graph"),
        }
    }
    // Initial KV caches are zeros.
    for &(si, _) in &g.state {
        inputs.insert(si, poot_tensor::HostTensor::zeros(g.aval(si).shape.clone()));
    }

    let (got, _new_caches) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();
    assert_eq!(got.shape(), vec![1, 1, h_dim]);

    // --- Hand-coded reference ---

    // matmul helper: [M, K] @ [K, N] -> [M, N] (row-major).
    let mm = |a: &[f32], a_rows: usize, a_cols: usize, b: &[f32], b_cols: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; a_rows * b_cols];
        for i in 0..a_rows {
            for j in 0..b_cols {
                for kk in 0..a_cols {
                    out[i * b_cols + j] += a[i * a_cols + kk] * b[kk * b_cols + j];
                }
            }
        }
        out
    };

    // rmsnorm helper for a single vector.
    let rms = |v: &[f32], w: &[f32]| -> Vec<f32> {
        let n = v.len();
        let mean_sq: f32 = v.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let scale_rms = 1.0 / (mean_sq + eps).sqrt();
        v.iter().zip(w).map(|(x, wi)| x * scale_rms * wi).collect()
    };

    let sigmoid_scalar = |v: f32| 1.0 / (1.0 + (-v).exp());

    // 1. Project Q (interleaved), K, V.
    let x_flat = &x_data[..]; // [H=8]
    let qg_flat = mm(x_flat, 1, h_dim, &wq_data, nh * 2 * hd); // [1, 16]
    // Per-head split: [n_heads=2, 2*head_dim=8] -- q = first 4, gate = last 4 of each head block.
    let q_raw: Vec<Vec<f32>> = (0..nh)
        .map(|hh| qg_flat[hh * 2 * hd..hh * 2 * hd + hd].to_vec())
        .collect();
    let gate_raw: Vec<Vec<f32>> = (0..nh)
        .map(|hh| qg_flat[hh * 2 * hd + hd..hh * 2 * hd + 2 * hd].to_vec())
        .collect();

    let k_vec = mm(x_flat, 1, h_dim, &wk_data, nkv * hd); // [1, 4]
    let v_vec = mm(x_flat, 1, h_dim, &wv_data, nkv * hd); // [1, 4]

    // 2. QK-norm (RMSNorm over head_dim=4, no bias).
    let q_normed: Vec<Vec<f32>> = q_raw.iter().map(|q| rms(q, &q_norm_w_data)).collect();
    let k_normed = rms(&k_vec, &k_norm_w_data);

    // 3. RoPE at pos=0: identity (cos=[1,1], sin=[0,0] -> no rotation).

    // 4. GQA attention at pos=0. Single key in valid prefix [0..=0].
    //    n_rep=2: both query heads attend the same (only) KV head.
    //    softmax of a single score = 1.0, so attn[h] = v for each h.
    let scale = 1.0 / (hd as f32).sqrt();
    let attn: Vec<Vec<f32>> = (0..nh)
        .map(|hh| {
            // score (unused for the result; kept to check the attention formula)
            let _score: f32 = q_normed[hh]
                .iter()
                .zip(&k_normed)
                .map(|(qi, ki)| qi * ki)
                .sum::<f32>()
                * scale;
            // single key -> softmax is 1.0 -> attn = v
            v_vec.clone()
        })
        .collect();

    // 5. Output gate: attn * sigmoid(gate) elementwise per head.
    let attn_gated: Vec<Vec<f32>> = (0..nh)
        .map(|hh| {
            attn[hh]
                .iter()
                .zip(&gate_raw[hh])
                .map(|(&ai, &gi)| ai * sigmoid_scalar(gi))
                .collect()
        })
        .collect();

    // 6. Flatten attn_gated [n_heads=2, head_dim=4] -> [8], then o-projection.
    let attn_flat: Vec<f32> = attn_gated.into_iter().flatten().collect();
    let want = mm(&attn_flat, 1, nh * hd, &wo_data, h_dim); // [1, H=8]

    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}
/// Card 135c: Qwen3-Next GDN (Gated DeltaNet) linear-attention block CPU oracle. Tiny config: hidden=8, num_k_heads=2,
/// num_v_heads=4, head_dim=4, conv_k=4. Derived: key_dim=8, value_dim=16, conv_dim=32.
///
/// The conv cache and GDN recurrent state start zeroed. With a zero state the delta-net output per head simplifies to
///   o_h[j] = dot(q_scaled_h, k_h) * beta_h * v_h[j]
/// (the rank-1 update S = outer(k, beta*(v-0)) applied to q_scaled).
///
/// Reference (section 3):
///   qkv_mixed = x @ w_qkv;  z = x @ w_gate
///   beta = sigmoid(x @ w_beta);  alpha = softplus((x @ w_alpha) + dt_bias);  g = alpha * ssm_a
///   conv_out = silu(depthwise_conv(qkv_mixed, w_conv, zeroed_cache, K=4))
///   q,k,v = split(conv_out); q,k = l2_norm_per_head(q,k)
///   q,k = repeat_heads(q,k, n_rep=2)
///   o = gated_delta_net_decode(q, k, v, g, beta, zero_state)
///   o = rmsnorm_per_head(o, norm_w) * silu(z_reshaped)
///   cur = o_flat @ w_out
#[test]
fn qwen3next_gdn_block_matches_reference() {
    use poot_graph_ir::Storage;
    use poot_models::qwen3next::{GdnHeadOrder, qwen3next_gdn_block};

    let (h, hk, hv, hd) = (8usize, 2usize, 4usize, 4usize); // hidden, num_k_heads, num_v_heads, head_dim
    let key_dim = hk * hd; // 8
    let value_dim = hv * hd; // 16
    let conv_dim = 2 * key_dim + value_dim; // 32
    let conv_k = 4usize;
    let _n_rep = hv / hk; // 2 (tiled repeat uses hh % hk, not n_rep)
    let eps = 1e-6f32;

    // Synthetic weights (deterministic, each with a distinct seed).
    let x_data = fill(h, 1);
    let w_qkv_data = fill(h * conv_dim, 2); // [H, conv_dim]
    let w_gate_data = fill(h * value_dim, 3); // [H, value_dim]
    let w_conv_data = fill(conv_k * conv_dim, 4); // [K, conv_dim]
    let w_beta_data = fill(h * hv, 5); // [H, H_v]
    let w_alpha_data = fill(h * hv, 6); // [H, H_v]
    let dt_bias_data = fill(hv, 7); // [H_v]
    let ssm_a_data = fill(hv, 8); // [H_v]  (may be pos or neg; sign matters for real use)
    let norm_w_data = fill(hd, 9); // [head_dim]
    let w_out_data = fill(value_dim * h, 10); // [value_dim, H]

    // Build the graph.
    let b = Builder::new();
    let x_t = b.constant("x", TensorType::f32(vec![1, 1, h]));
    let wqkv_t = b.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let wg_t = b.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let wc_t = b.constant("w_conv", TensorType::f32(vec![conv_k, conv_dim]));
    let wb_t = b.constant("w_beta", TensorType::f32(vec![h, hv]));
    let wa_t = b.constant("w_alpha", TensorType::f32(vec![h, hv]));
    let dtb_t = b.constant("dt_bias", TensorType::f32(vec![hv]));
    let sa_t = b.constant("ssm_a", TensorType::f32(vec![hv]));
    let nw_t = b.constant("norm_w", TensorType::f32(vec![hd]));
    let wo_t = b.constant("w_out", TensorType::f32(vec![value_dim, h]));
    let cc_t = b.state_input(
        "conv_cache",
        TensorType::f32(vec![1, conv_k - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let si_t = b.state_input(
        "ssm_state",
        TensorType::f32(vec![1, hv, hd, hd]),
        StateRole::Recurrent,
    );

    let (out, cc_out, s_out) = qwen3next_gdn_block(
        &b,
        x_t,
        wqkv_t,
        wg_t,
        wc_t,
        wb_t,
        wa_t,
        dtb_t,
        sa_t,
        nw_t,
        wo_t,
        cc_t,
        si_t,
        hk,
        hv,
        hd,
        conv_k,
        eps,
        GdnHeadOrder::Tiled,
    );
    assert_eq!(b.aval(out).shape, vec![1, 1, h], "output shape");
    assert_eq!(
        b.aval(cc_out).shape,
        vec![1, conv_k - 1, conv_dim],
        "conv cache shape"
    );
    assert_eq!(b.aval(s_out).shape, vec![1, hv, hd, hd], "state shape");
    let g = b.finish_with_state(out, &[(cc_t, cc_out), (si_t, s_out)]);

    // Bind constants.
    let mut named: HashMap<String, Vec<f32>> = HashMap::new();
    named.insert("x".into(), x_data.clone());
    named.insert("w_qkv".into(), w_qkv_data.clone());
    named.insert("w_gate".into(), w_gate_data.clone());
    named.insert("w_conv".into(), w_conv_data.clone());
    named.insert("w_beta".into(), w_beta_data.clone());
    named.insert("w_alpha".into(), w_alpha_data.clone());
    named.insert("dt_bias".into(), dt_bias_data.clone());
    named.insert("ssm_a".into(), ssm_a_data.clone());
    named.insert("norm_w".into(), norm_w_data.clone());
    named.insert("w_out".into(), w_out_data.clone());

    let mut inputs: HashMap<ValueId, poot_tensor::HostTensor> = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                let data = named[name].clone();
                inputs.insert(
                    id,
                    poot_tensor::HostTensor::f32(g.aval(id).shape.clone(), data),
                );
            }
            Storage::State => {}
            _ => panic!("unexpected storage"),
        }
    }
    // Initial states are zeros.
    for &(si, _) in &g.state {
        inputs.insert(si, poot_tensor::HostTensor::zeros(g.aval(si).shape.clone()));
    }

    let (got, _states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();
    assert_eq!(got.shape(), vec![1, 1, h]);

    // --- Hand-coded reference ---

    // matmul helper: A[M, K] @ B[K, N] -> C[M, N] (row-major).
    let mm = |a: &[f32], a_rows: usize, a_cols: usize, b_: &[f32], b_cols: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; a_rows * b_cols];
        for i in 0..a_rows {
            for j in 0..b_cols {
                for kk in 0..a_cols {
                    out[i * b_cols + j] += a[i * a_cols + kk] * b_[kk * b_cols + j];
                }
            }
        }
        out
    };

    let silu_f = |v: f32| v / (1.0 + (-v).exp());
    let sigmoid_f = |v: f32| 1.0 / (1.0 + (-v).exp());
    let softplus_f = |v: f32| (1.0f32 + v.exp()).ln();

    // rmsnorm over a slice (mean of squares denominator).
    let rms = |v: &[f32], w: &[f32]| -> Vec<f32> {
        let n = v.len() as f32;
        let mean_sq: f32 = v.iter().map(|x| x * x).sum::<f32>() / n;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        v.iter().zip(w).map(|(vi, wi)| vi * scale * wi).collect()
    };

    // l2_norm over a slice (sum of squares, no mean, no weight).
    let l2 = |v: &[f32]| -> Vec<f32> {
        let ss: f32 = v.iter().map(|x| x * x).sum();
        let norm = (ss + eps).sqrt();
        v.iter().map(|x| x / norm).collect()
    };

    // 1. Projections.
    let x_flat = &x_data[..]; // [H]
    let qkv_mixed = mm(x_flat, 1, h, &w_qkv_data, conv_dim); // [conv_dim]
    let z_flat = mm(x_flat, 1, h, &w_gate_data, value_dim); // [value_dim]
    let beta_raw = mm(x_flat, 1, h, &w_beta_data, hv); // [H_v]
    let beta_ref: Vec<f32> = beta_raw.iter().map(|&v| sigmoid_f(v)).collect();
    let alpha_raw = mm(x_flat, 1, h, &w_alpha_data, hv); // [H_v]
    let alpha_ref: Vec<f32> = (0..hv)
        .map(|i| softplus_f(alpha_raw[i] + dt_bias_data[i]))
        .collect();
    let _g_ref: Vec<f32> = (0..hv).map(|i| alpha_ref[i] * ssm_a_data[i]).collect();

    // 2. Causal conv1d (zeroed cache: first K-1 rows are 0, last row = qkv_mixed).
    //    With zeroed cache: conv_out[c] = w_conv[(K-1)*conv_dim + c] * qkv_mixed[c].
    //    (The first K-1 window rows are 0 from the zeroed cache.)
    let conv_out_raw: Vec<f32> = (0..conv_dim)
        .map(|c| {
            // window[k, c] = 0 for k < K-1, qkv_mixed[c] for k = K-1
            w_conv_data[(conv_k - 1) * conv_dim + c] * qkv_mixed[c]
        })
        .collect();
    let conv_out_ref: Vec<f32> = conv_out_raw.iter().map(|&v| silu_f(v)).collect();

    // 3. Split into Q, K, V (flat slices, then reshape to [heads, head_dim]).
    let q_flat: Vec<f32> = conv_out_ref[..key_dim].to_vec(); // [key_dim]
    let k_flat: Vec<f32> = conv_out_ref[key_dim..2 * key_dim].to_vec(); // [key_dim]
    let v_flat: Vec<f32> = conv_out_ref[2 * key_dim..].to_vec(); // [value_dim]

    // 4. L2 normalize Q and K per head ([hk, hd] and [hk, hd]).
    let q_heads: Vec<Vec<f32>> = (0..hk)
        .map(|hh| l2(&q_flat[hh * hd..(hh + 1) * hd]))
        .collect();
    let k_heads: Vec<Vec<f32>> = (0..hk)
        .map(|hh| l2(&k_flat[hh * hd..(hh + 1) * hd]))
        .collect();
    let v_heads: Vec<Vec<f32>> = (0..hv)
        .map(|hh| v_flat[hh * hd..(hh + 1) * hd].to_vec())
        .collect();

    // 5. GQA broadcast: repeat q,k from hk to hv heads. Qwen3-Next uses the tiled repeat (v-head h <- k-head h % hk,
    // llama.cpp ggml_repeat_4d), not blocked (h / n_rep).
    let q_v: Vec<Vec<f32>> = (0..hv).map(|hh| q_heads[hh % hk].clone()).collect();
    let k_v: Vec<Vec<f32>> = (0..hv).map(|hh| k_heads[hh % hk].clone()).collect();

    // 6. Gated delta-net decode (zeroed state).
    //    With S=0: S_new = outer(k_h, beta_h * v_h); o_h = q_scaled_h @ S_new.
    let scale = 1.0 / (hd as f32).sqrt();
    let mut o_heads: Vec<Vec<f32>> = Vec::with_capacity(hv);
    for hh in 0..hv {
        let q_scaled: Vec<f32> = q_v[hh].iter().map(|&v| v * scale).collect();
        let beta_h = beta_ref[hh];
        // delta = beta_h * (v_h - S^T k = v_h - 0) = beta_h * v_h
        let delta: Vec<f32> = v_heads[hh].iter().map(|&v| beta_h * v).collect();
        // S_new[i,j] = k_h[i] * delta[j]  (outer product; S starts at 0)
        // o_h[j] = sum_i q_scaled[i] * S_new[i,j] = sum_i q_scaled[i] * k_h[i] * delta[j]
        //        = dot(q_scaled, k_h) * delta[j]
        let dot_qk: f32 = q_scaled.iter().zip(&k_v[hh]).map(|(a, b)| a * b).sum();
        let o_h: Vec<f32> = delta.iter().map(|&d| dot_qk * d).collect();
        o_heads.push(o_h);
    }

    // 7. Gated RMSNorm: rmsnorm(o, norm_w) * silu(z_shaped).
    //    z_shaped [hv, hd] = reshape(z_flat [value_dim]).
    let z_heads: Vec<Vec<f32>> = (0..hv)
        .map(|hh| z_flat[hh * hd..(hh + 1) * hd].to_vec())
        .collect();
    let o_gated: Vec<f32> = (0..hv)
        .flat_map(|hh| {
            let o_norm = rms(&o_heads[hh], &norm_w_data);
            o_norm
                .iter()
                .zip(&z_heads[hh])
                .map(|(&on, &zi)| on * silu_f(zi))
                .collect::<Vec<f32>>()
        })
        .collect(); // [value_dim]

    // 8. Out-projection.
    let want = mm(&o_gated, 1, value_dim, &w_out_data, h); // [H]

    assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
}

/// Card 135c integration: the full Qwen3-Next (qwen35moe) decode tracer end to end. Traces `qwen3next_decode_trace` for a
/// tiny synthetic config whose layer schedule has both block types (`full_attention_interval=2`, `n_layers=4` => Gated
/// DeltaNet at layers 0,2 and gated full-attention at layers 1,3), decodes several tokens while threading all
/// heterogeneous carried state (attn layers carry k/v caches; GDN layers carry conv + recurrent state), and checks the
/// logits at every step against a plain-Rust reference composing the same four blocks.
///
/// Graph and reference use the same by-name weights. RoPE tables are the identity (cos=1, sin=0) so the reference need not
/// re-implement rope (covered by `qwen3next_gated_attention_matches_reference`); everything else (embed gather, both
/// mixer types, the per-layer pre-norms, residuals, MoE + shared expert, final norm, lm_head, growing KV cache, GDN
/// conv/recurrent state carry) is exercised across steps.
#[test]
fn qwen3next_decode_trace_matches_reference() {
    use poot_graph_ir::Storage;
    use poot_models::qwen3next::{Qwen3NextConfig, qwen3next_decode_trace};

    // --- tiny-synthetic config (both layer types present) ---
    let cfg = Qwen3NextConfig {
        vocab: 6,
        hidden: 8,
        n_layers: 4,
        full_attention_interval: 2, // attn at li=1,3 ; GDN at li=0,2
        eps: 1e-6,
        max_pos: 8,
        rotary_dim: 4,
        // full-attention dims
        n_heads: 4,
        n_kv_heads: 2, // n_rep = 2 (genuine GQA)
        head_dim: 4,
        // GDN dims
        gdn_num_k_heads: 2,
        gdn_num_v_heads: 4, // n_rep = 2
        gdn_head_dim: 4,
        conv_k: 4,
        // MoE dims
        n_experts: 4,
        top_k: 2,
        expert_inter: 6,
        shared_inter: 6,
    };
    let cap = 4usize;
    let tokens = [1usize, 3, 2]; // decode 3 tokens
    let eps = cfg.eps;

    let (h, e, k_top, ei, si_) = (
        cfg.hidden,
        cfg.n_experts,
        cfg.top_k,
        cfg.expert_inter,
        cfg.shared_inter,
    );

    // --- deterministic weights, keyed by the exact names the tracer uses (single source) ---
    let mut named: HashMap<String, Vec<f32>> = HashMap::new();
    let put = |named: &mut HashMap<String, Vec<f32>>, name: &str, numel: usize| {
        named.insert(name.to_string(), fill(numel, seed_of(name)));
    };

    // globals
    put(&mut named, "model.embed_tokens.weight", cfg.vocab * h);
    put(&mut named, "model.norm.weight", h);
    put(&mut named, "lm_head.weight", h * cfg.vocab);
    named.insert("rope.cos".into(), vec![1.0; cfg.max_pos * cfg.rotary_dim]);
    named.insert("rope.sin".into(), vec![0.0; cfg.max_pos * cfg.rotary_dim]);

    let ahd = cfg.head_dim;
    let (nh, nkv) = (cfg.n_heads, cfg.n_kv_heads);
    let ghd = cfg.gdn_head_dim;
    let (hk, hv, ck) = (cfg.gdn_num_k_heads, cfg.gdn_num_v_heads, cfg.conv_k);
    let key_dim = hk * ghd;
    let value_dim = hv * ghd;
    let conv_dim = 2 * key_dim + value_dim;

    for li in 0..cfg.n_layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        put(&mut named, &p("attn_norm.weight"), h);
        put(&mut named, &p("post_attention_norm.weight"), h);
        // MoE (both layer types)
        put(&mut named, &p("ffn_gate_inp.weight"), h * e);
        put(&mut named, &p("ffn.w_in"), e * h * 2 * ei);
        put(&mut named, &p("ffn.w_out"), e * ei * h);
        put(&mut named, &p("ffn_gate_shexp.weight"), h * si_);
        put(&mut named, &p("ffn_up_shexp.weight"), h * si_);
        put(&mut named, &p("ffn_down_shexp.weight"), si_ * h);
        put(&mut named, &p("ffn_gate_inp_shexp.weight"), h);
        if cfg.is_attn_layer(li) {
            put(&mut named, &p("attn_q.weight"), h * nh * 2 * ahd);
            put(&mut named, &p("attn_k.weight"), h * nkv * ahd);
            put(&mut named, &p("attn_v.weight"), h * nkv * ahd);
            put(&mut named, &p("attn_output.weight"), nh * ahd * h);
            put(&mut named, &p("attn_q_norm.weight"), ahd);
            put(&mut named, &p("attn_k_norm.weight"), ahd);
        } else {
            put(&mut named, &p("attn_qkv.weight"), h * conv_dim);
            put(&mut named, &p("attn_gate.weight"), h * value_dim);
            put(&mut named, &p("ssm_conv1d.weight"), ck * conv_dim);
            put(&mut named, &p("ssm_beta.weight"), h * hv);
            put(&mut named, &p("ssm_alpha.weight"), h * hv);
            put(&mut named, &p("ssm_dt.bias"), hv);
            // ssm_a stored negative so exp(alpha*ssm_a) is a decay in (0,1) (matches the real model).
            let a = fill(hv, seed_of(&p("ssm_a")))
                .iter()
                .map(|v| -v.abs())
                .collect();
            named.insert(p("ssm_a"), a);
            put(&mut named, &p("ssm_norm.weight"), ghd);
            put(&mut named, &p("ssm_out.weight"), value_dim * h);
        }
    }

    // --- plain-Rust reference helpers ---
    // row-vector @ matrix: x[K] @ w[K,N] -> [N].
    let mm = |x: &[f32], w: &[f32], n: usize| -> Vec<f32> {
        let kk = x.len();
        let mut out = vec![0.0f32; n];
        for j in 0..n {
            for i in 0..kk {
                out[j] += x[i] * w[i * n + j];
            }
        }
        out
    };
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let sp = |x: f32| (1.0f32 + x.exp()).ln();
    let rms = |v: &[f32], w: &[f32]| -> Vec<f32> {
        let n = v.len() as f32;
        let ms: f32 = v.iter().map(|x| x * x).sum::<f32>() / n;
        let sc = 1.0 / (ms + eps).sqrt();
        v.iter().zip(w).map(|(x, wi)| x * sc * wi).collect()
    };
    let l2 = |v: &[f32]| -> Vec<f32> {
        let ss: f32 = v.iter().map(|x| x * x).sum();
        let nrm = (ss + eps).sqrt();
        v.iter().map(|x| x / nrm).collect()
    };

    // MoE FFN + shared expert reference (section 5). x: [H] -> [H].
    let g = |name: &str| named[name].clone();
    let moe_ffn_ref = |x: &[f32], li: usize| -> Vec<f32> {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        let router = g(&p("ffn_gate_inp.weight")); // [H, E]
        let w_in = g(&p("ffn.w_in")); // [E, H, 2I]
        let w_out = g(&p("ffn.w_out")); // [E, I, H]
        // routed: softmax over all E, top-k, renormalize.
        let logits = mm(x, &router, e);
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let expv: Vec<f32> = logits.iter().map(|&v| (v - mx).exp()).collect();
        let se: f32 = expv.iter().sum();
        let probs: Vec<f32> = expv.iter().map(|&v| v / se).collect();
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &c| probs[c].partial_cmp(&probs[a]).unwrap());
        let top: Vec<usize> = order[..k_top].to_vec();
        let tsum: f32 = top.iter().map(|&ex| probs[ex]).sum();
        let gw: Vec<f32> = top.iter().map(|&ex| probs[ex] / tsum).collect();
        let mut moe_out = vec![0.0f32; h];
        for (rank, &ex) in top.iter().enumerate() {
            let base_in = ex * h * 2 * ei;
            let mut gu = vec![0.0f32; 2 * ei];
            for ii in 0..2 * ei {
                for hi in 0..h {
                    gu[ii] += x[hi] * w_in[base_in + hi * 2 * ei + ii];
                }
            }
            let act: Vec<f32> = (0..ei).map(|ii| silu(gu[ii]) * gu[ei + ii]).collect();
            let base_out = ex * ei * h;
            for hi in 0..h {
                let o: f32 = (0..ei)
                    .map(|ii| act[ii] * w_out[base_out + ii * h + hi])
                    .sum();
                moe_out[hi] += gw[rank] * o;
            }
        }
        // shared expert (dense SwiGLU, sigmoid scalar gate, ADD).
        let sgw = g(&p("ffn_gate_shexp.weight"));
        let suw = g(&p("ffn_up_shexp.weight"));
        let sdw = g(&p("ffn_down_shexp.weight"));
        let sgi = g(&p("ffn_gate_inp_shexp.weight"));
        let sgo = mm(x, &sgw, si_);
        let suo = mm(x, &suw, si_);
        let sact: Vec<f32> = (0..si_).map(|ii| silu(sgo[ii]) * suo[ii]).collect();
        let shexp = mm(&sact, &sdw, h);
        let graw: f32 = (0..h).map(|hi| x[hi] * sgi[hi]).sum();
        let gate = sig(graw);
        (0..h).map(|hi| moe_out[hi] + gate * shexp[hi]).collect()
    };

    // --- reference carried state ---
    // attn layers: growing per-position k/v (each [nkv*ahd]); GDN layers: conv cache + recurrent state.
    let mut ref_kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.n_layers];
    let mut ref_vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.n_layers];
    let mut ref_conv: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|_| vec![0.0f32; (ck - 1) * conv_dim])
        .collect();
    let mut ref_ssm: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|_| vec![0.0f32; hv * ghd * ghd])
        .collect();

    // --- graph carried state, in g.state order ---
    let g0 = qwen3next_decode_trace(&cfg, 0, cap);
    let mut caches: Vec<poot_tensor::HostTensor> = g0
        .state
        .iter()
        .map(|&(sid, _)| poot_tensor::HostTensor::zeros(g0.aval(sid).shape.clone()))
        .collect();

    for (step, &tok) in tokens.iter().enumerate() {
        // ---- graph decode for this position ----
        let graph = qwen3next_decode_trace(&cfg, step, cap);
        let mut inputs: HashMap<ValueId, poot_tensor::HostTensor> = HashMap::new();
        for &id in &graph.inputs {
            let m = graph.meta(id);
            match m.storage {
                Storage::Slot(Slot::Token) => {
                    inputs.insert(id, poot_tensor::HostTensor::i32(vec![], vec![tok as i32]));
                }
                Storage::Slot(Slot::Pos) => {
                    inputs.insert(id, poot_tensor::HostTensor::i32(vec![], vec![step as i32]));
                }
                Storage::Const => {
                    let name = m.name.as_deref().unwrap();
                    inputs.insert(
                        id,
                        poot_tensor::HostTensor::f32(
                            graph.aval(id).shape.clone(),
                            named[name].clone(),
                        ),
                    );
                }
                Storage::State => {}
                other => panic!("unexpected storage {other:?} in qwen3next decode trace"),
            }
        }
        for (ci, &(sid, _)) in graph.state.iter().enumerate() {
            inputs.insert(sid, caches[ci].clone());
        }
        let (got, new_caches) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&graph, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .unwrap();
        caches = new_caches;
        assert_eq!(got.shape(), vec![1, 1, cfg.vocab], "logits shape");

        // ---- plain-Rust reference for this position ----
        // embed gather
        let emb = &named["model.embed_tokens.weight"];
        let mut x: Vec<f32> = emb[tok * h..(tok + 1) * h].to_vec();

        for li in 0..cfg.n_layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            // 1. input norm
            let normed = rms(&x, &g(&p("attn_norm.weight")));

            // 2. mixer
            let mix: Vec<f32> = if cfg.is_attn_layer(li) {
                let wq = g(&p("attn_q.weight"));
                let wk = g(&p("attn_k.weight"));
                let wv = g(&p("attn_v.weight"));
                let wo = g(&p("attn_output.weight"));
                let qn = g(&p("attn_q_norm.weight"));
                let kn = g(&p("attn_k_norm.weight"));
                let qg = mm(&normed, &wq, nh * 2 * ahd);
                // per-head query|gate split
                let qh: Vec<Vec<f32>> = (0..nh)
                    .map(|hh| qg[hh * 2 * ahd..hh * 2 * ahd + ahd].to_vec())
                    .collect();
                let gateh: Vec<Vec<f32>> = (0..nh)
                    .map(|hh| qg[hh * 2 * ahd + ahd..hh * 2 * ahd + 2 * ahd].to_vec())
                    .collect();
                let kvec = mm(&normed, &wk, nkv * ahd);
                let vvec = mm(&normed, &wv, nkv * ahd);
                // QK-norm per head; rope identity.
                let q_normed: Vec<Vec<f32>> = qh.iter().map(|q| rms(q, &qn)).collect();
                let mut k_flat = Vec::with_capacity(nkv * ahd);
                for kk in 0..nkv {
                    k_flat.extend(rms(&kvec[kk * ahd..(kk + 1) * ahd], &kn));
                }
                // append to caches at this position.
                ref_kc[li].push(k_flat);
                ref_vc[li].push(vvec);
                let scale = 1.0 / (ahd as f32).sqrt();
                let n_rep = nh / nkv;
                let s_len = ref_kc[li].len();
                let mut attn_flat = vec![0.0f32; nh * ahd];
                for hh in 0..nh {
                    let kvh = hh / n_rep;
                    // scores over all cached positions
                    let mut scores = vec![0.0f32; s_len];
                    for t in 0..s_len {
                        let kt = &ref_kc[li][t][kvh * ahd..(kvh + 1) * ahd];
                        let dot: f32 = q_normed[hh].iter().zip(kt).map(|(a, b)| a * b).sum();
                        scores[t] = dot * scale;
                    }
                    let mxs = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let ex: Vec<f32> = scores.iter().map(|&v| (v - mxs).exp()).collect();
                    let den: f32 = ex.iter().sum();
                    let prob: Vec<f32> = ex.iter().map(|&v| v / den).collect();
                    // weighted sum of values -> head output, then output gate.
                    for j in 0..ahd {
                        let mut o = 0.0f32;
                        for t in 0..s_len {
                            o += prob[t] * ref_vc[li][t][kvh * ahd + j];
                        }
                        attn_flat[hh * ahd + j] = o * sig(gateh[hh][j]);
                    }
                }
                mm(&attn_flat, &wo, h)
            } else {
                let wqkv = g(&p("attn_qkv.weight"));
                let wgate = g(&p("attn_gate.weight"));
                let wconv = g(&p("ssm_conv1d.weight"));
                let wbeta = g(&p("ssm_beta.weight"));
                let walpha = g(&p("ssm_alpha.weight"));
                let dtb = g(&p("ssm_dt.bias"));
                let ssm_a = g(&p("ssm_a"));
                let nw = g(&p("ssm_norm.weight"));
                let wout = g(&p("ssm_out.weight"));

                let qkv = mm(&normed, &wqkv, conv_dim);
                let z = mm(&normed, &wgate, value_dim);
                let beta: Vec<f32> = mm(&normed, &wbeta, hv).iter().map(|&v| sig(v)).collect();
                let alpha: Vec<f32> = mm(&normed, &walpha, hv)
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| sp(v + dtb[i]))
                    .collect();
                let gg: Vec<f32> = (0..hv).map(|i| alpha[i] * ssm_a[i]).collect();

                // causal depthwise conv over the concatenated qkv channels, then silu.
                let cache = &ref_conv[li]; // [(K-1)*conv_dim]
                let mut conv_out = vec![0.0f32; conv_dim];
                for c in 0..conv_dim {
                    let mut acc = 0.0f32;
                    for jw in 0..ck {
                        let val = if jw < ck - 1 {
                            cache[jw * conv_dim + c]
                        } else {
                            qkv[c]
                        };
                        acc += wconv[jw * conv_dim + c] * val;
                    }
                    conv_out[c] = silu(acc);
                }
                // update conv cache: drop oldest column -> [cache[1..], qkv].
                let mut new_cache = vec![0.0f32; (ck - 1) * conv_dim];
                for pos in 0..ck - 1 {
                    for c in 0..conv_dim {
                        new_cache[pos * conv_dim + c] = if pos + 1 < ck - 1 {
                            cache[(pos + 1) * conv_dim + c]
                        } else {
                            qkv[c]
                        };
                    }
                }
                ref_conv[li] = new_cache;

                // split, l2-norm q/k per head, GQA repeat.
                let q_flat = &conv_out[0..key_dim];
                let k_flat = &conv_out[key_dim..2 * key_dim];
                let v_flat = &conv_out[2 * key_dim..];
                let q_heads: Vec<Vec<f32>> = (0..hk)
                    .map(|hh| l2(&q_flat[hh * ghd..(hh + 1) * ghd]))
                    .collect();
                let k_heads: Vec<Vec<f32>> = (0..hk)
                    .map(|hh| l2(&k_flat[hh * ghd..(hh + 1) * ghd]))
                    .collect();
                let _n_rep = hv / hk;
                let scale = 1.0 / (ghd as f32).sqrt();

                // per v-head gated delta-net recurrence with carried state S[hh] ([ghd,ghd]).
                let mut o_all = vec![0.0f32; value_dim];
                for hh in 0..hv {
                    // TILED GDN repeat (v-head h <- k-head h % hk), not blocked (h / n_rep).
                    let qv: Vec<f32> = q_heads[hh % hk].iter().map(|&v| v * scale).collect();
                    let kvh = &k_heads[hh % hk];
                    let vvh = &v_flat[hh * ghd..(hh + 1) * ghd];
                    let decay = gg[hh].exp();
                    let base = hh * ghd * ghd;
                    // S *= decay
                    let mut s: Vec<f32> = (0..ghd * ghd)
                        .map(|idx| ref_ssm[li][base + idx] * decay)
                        .collect();
                    // kv[j] = sum_i S[i,j]*k[i]
                    let mut kvv = vec![0.0f32; ghd];
                    for j in 0..ghd {
                        for i in 0..ghd {
                            kvv[j] += s[i * ghd + j] * kvh[i];
                        }
                    }
                    // delta[j] = beta*(v[j]-kv[j]); S[i,j] += k[i]*delta[j]
                    let delta: Vec<f32> = (0..ghd).map(|j| beta[hh] * (vvh[j] - kvv[j])).collect();
                    for i in 0..ghd {
                        for j in 0..ghd {
                            s[i * ghd + j] += kvh[i] * delta[j];
                        }
                    }
                    // o[j] = sum_i q_scaled[i]*S_new[i,j]
                    for j in 0..ghd {
                        let mut o = 0.0f32;
                        for i in 0..ghd {
                            o += qv[i] * s[i * ghd + j];
                        }
                        o_all[hh * ghd + j] = o;
                    }
                    // write back state
                    ref_ssm[li][base..(ghd * ghd + base)].copy_from_slice(&s[..(ghd * ghd)]);
                }
                // gated output rmsnorm: rmsnorm(o_head, nw) * silu(z_head); flatten; out-proj.
                let mut o_gated = vec![0.0f32; value_dim];
                for hh in 0..hv {
                    let on = rms(&o_all[hh * ghd..(hh + 1) * ghd], &nw);
                    for j in 0..ghd {
                        o_gated[hh * ghd + j] = on[j] * silu(z[hh * ghd + j]);
                    }
                }
                mm(&o_gated, &wout, h)
            };

            // 3. attention residual
            for hi in 0..h {
                x[hi] += mix[hi];
            }
            // 4. pre-FFN norm + 5. MoE FFN
            let ff_in = rms(&x, &g(&p("post_attention_norm.weight")));
            let ff = moe_ffn_ref(&ff_in, li);
            // 6. FFN residual
            for hi in 0..h {
                x[hi] += ff[hi];
            }
        }

        // final norm + lm_head
        let xn = rms(&x, &g("model.norm.weight"));
        let want = mm(&xn, &g("lm_head.weight"), cfg.vocab);

        assert_close_rel(got.as_f32().unwrap(), &want, 2e-4);
    }
}
