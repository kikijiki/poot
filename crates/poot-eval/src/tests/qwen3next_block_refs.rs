//! The gated delta-net decode op against a hand-rolled recurrence.

use crate::{EvalBudget, EvalError, EvalOptions, Value};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::gated_delta_net_decode;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{StateRole, ValueId};
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::{assert_close_rel, max_abs_error};

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
