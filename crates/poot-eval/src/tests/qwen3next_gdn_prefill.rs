//! The chunked gated delta-net core against the sequential one on realistic inputs, and the Mamba2 core paths.

use crate::{EvalBudget, EvalError, EvalOptions, Value};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::graph::StateRole;
use poot_graph_ir::ops::{gated_delta_net_decode, gdn_prefill_chunked, mamba2_ssd_decode};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{ValueId, ValueTable};
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::{assert_close_rel, max_abs_error};

/// Diagnostic (card 158): feed the chunked core (`gdn_prefill_chunked`) and the sequential core
/// (`gated_delta_net_decode`) the same core inputs a real GDN block produces (q/k = l2norm of conv(proj),
/// v = silu(conv(proj)), g = softplus(proj)*ssm_a, beta = sigmoid(proj)) at real dims, and report per-head
/// state error with each head's decay magnitude. Sweeps dt magnitude to test the cumulative-log-decay
/// cancellation hypothesis. Prints, asserts nothing.
#[test]
fn gdn_core_on_realistic_frontend_inputs_per_head_real_dims() {
    use poot_graph_ir::op::{BinOp, RedOp, UnOp};
    use poot_graph_ir::ops::{causal_conv1d_prefill, linear, sigmoid, softplus};
    use poot_graph_ir::types::Scalar;

    let (h, hk, hv, d, ck, l) = (2048usize, 16usize, 32usize, 128usize, 4usize, 115usize);
    let key_dim = hk * d;
    let value_dim = hv * d;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6f32;
    let ssm_a_d: Vec<f32> = (0..hv)
        .map(|i| -0.02 - 72.0 * (i as f32 / (hv - 1) as f32))
        .collect();

    for &dt_scale in &[1.0f32, 8.0] {
        // Front-end graph outputting q(H_k), k(H_k), v(H_v), g(H_v per pos), beta(H_v per pos).
        let fb = Builder::for_tests();
        let x = fb.constant("x", TensorType::f32(vec![1, l, h])).unwrap();
        let w_qkv = fb
            .constant("w_qkv", TensorType::f32(vec![h, conv_dim]))
            .unwrap();
        let w_conv = fb
            .constant("w_conv", TensorType::f32(vec![ck, conv_dim]))
            .unwrap();
        let w_beta = fb.constant("w_beta", TensorType::f32(vec![h, hv])).unwrap();
        let w_alpha = fb
            .constant("w_alpha", TensorType::f32(vec![h, hv]))
            .unwrap();
        let dt_bias = fb.constant("dt_bias", TensorType::f32(vec![hv])).unwrap();
        let ssm_a = fb.constant("ssm_a", TensorType::f32(vec![hv])).unwrap();
        let qkv_mixed = linear(&fb, x, w_qkv, None).unwrap();
        let conv_out = poot_graph_ir::ops::silu(
            &fb,
            causal_conv1d_prefill(&fb, qkv_mixed, w_conv, ck).unwrap(),
        )
        .unwrap();
        let q_flat = fb.slice(conv_out, 2, 0, key_dim).unwrap();
        let k_flat = fb.slice(conv_out, 2, key_dim, 2 * key_dim).unwrap();
        let v_flat = fb.slice(conv_out, 2, 2 * key_dim, conv_dim).unwrap();
        let q = fb
            .transpose(
                fb.reshape(q_flat, vec![1, l, hk, d]).unwrap(),
                vec![0, 2, 1, 3],
            )
            .unwrap();
        let k = fb
            .transpose(
                fb.reshape(k_flat, vec![1, l, hk, d]).unwrap(),
                vec![0, 2, 1, 3],
            )
            .unwrap();
        let v = fb
            .transpose(
                fb.reshape(v_flat, vec![1, l, hv, d]).unwrap(),
                vec![0, 2, 1, 3],
            )
            .unwrap();
        // l2 norm of q,k over the last axis.
        let l2 = |t| {
            let sq = fb.binary(BinOp::Mul, t, t).unwrap();
            let ss = fb.reduce(RedOp::Sum, sq, 3, true).unwrap();
            let nrm = fb
                .unary(
                    UnOp::Sqrt,
                    fb.binary_scalar(BinOp::Add, ss, Scalar::F32(eps)).unwrap(),
                )
                .unwrap();
            fb.binary(BinOp::Div, t, nrm).unwrap()
        };
        let q = l2(q);
        let k = l2(k);
        let beta = sigmoid(&fb, linear(&fb, x, w_beta, None).unwrap()).unwrap();
        let alpha = softplus(
            &fb,
            fb.binary(BinOp::Add, linear(&fb, x, w_alpha, None).unwrap(), dt_bias)
                .unwrap(),
        )
        .unwrap();
        let g = fb.binary(BinOp::Mul, alpha, ssm_a).unwrap();
        let g4 = fb
            .transpose(fb.reshape(g, vec![1, l, hv, 1]).unwrap(), vec![0, 2, 1, 3])
            .unwrap(); // [1,Hv,L,1]
        let beta4 = fb
            .transpose(
                fb.reshape(beta, vec![1, l, hv, 1]).unwrap(),
                vec![0, 2, 1, 3],
            )
            .unwrap();
        // Multiple outputs via eval_all and reading ids.
        let fg = fb.finish(q).unwrap(); // primary irrelevant; use eval_all to read every tap.
        let mut fin = HashMap::new();
        fin.insert(x.id(), HostTensor::f32(vec![1, l, h], fill(l * h, 111)));
        fin.insert(
            w_qkv.id(),
            HostTensor::f32(vec![h, conv_dim], fill(h * conv_dim, 112)),
        );
        fin.insert(
            w_conv.id(),
            HostTensor::f32(vec![ck, conv_dim], fill(ck * conv_dim, 114)),
        );
        fin.insert(w_beta.id(), HostTensor::f32(vec![h, hv], fill(h * hv, 115)));
        // dt_scale scales w_alpha to raise/lower the softplus argument (dt magnitude).
        fin.insert(
            w_alpha.id(),
            HostTensor::f32(
                vec![h, hv],
                fill(h * hv, 116)
                    .into_iter()
                    .map(|x| x * dt_scale)
                    .collect(),
            ),
        );
        fin.insert(dt_bias.id(), HostTensor::f32(vec![hv], fill(hv, 117)));
        fin.insert(ssm_a.id(), HostTensor::f32(vec![hv], ssm_a_d.clone()));
        let env = (|| -> Result<ValueTable<Option<HostTensor>>, EvalError> {
            let values: HashMap<ValueId, Value> = fin
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let environment = crate::eval(
                &fg,
                &values,
                EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
            )?
            .environment
            .expect("keep_environment was set");
            environment
                .into_iter()
                .map(|slot| slot.map(Value::into_host).transpose())
                .collect()
        })()
        .unwrap();
        let read = |t: poot_graph_ir::Traced| env[t.id()].clone().unwrap();
        let (qd, kd, vd, gd, betad) = (
            read(q).as_f32().unwrap().to_vec(),
            read(k).as_f32().unwrap().to_vec(),
            read(v).as_f32().unwrap().to_vec(),
            read(g4).as_f32().unwrap().to_vec(),
            read(beta4).as_f32().unwrap().to_vec(),
        );
        // g magnitude report.
        let gmin = gd.iter().cloned().fold(f32::INFINITY, f32::min);
        let gmax = gd.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Tiled repeat of q,k H_k->H_v for the sequential reference.
        let tile = |src: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; hv * l * d];
            for h2 in 0..hv {
                let hk2 = h2 % hk;
                out[h2 * l * d..(h2 + 1) * l * d]
                    .copy_from_slice(&src[hk2 * l * d..(hk2 + 1) * l * d]);
            }
            out
        };
        let qv = tile(&qd);
        let kv = tile(&kd);

        // Sequential core (chunk-independent reference), computed first so the chunk sweep can diff against it.
        let db = Builder::for_tests();
        let q_in = db
            .constant("q", TensorType::f32(vec![1, hv, 1, d]))
            .unwrap();
        let k_in = db
            .constant("k", TensorType::f32(vec![1, hv, 1, d]))
            .unwrap();
        let v_in = db
            .constant("v", TensorType::f32(vec![1, hv, 1, d]))
            .unwrap();
        let g_in = db
            .constant("g", TensorType::f32(vec![1, hv, 1, 1]))
            .unwrap();
        let bt_in = db
            .constant("beta", TensorType::f32(vec![1, hv, 1, 1]))
            .unwrap();
        let ds_in = db
            .state_input(
                "s",
                TensorType::f32(vec![1, hv, d, d]),
                StateRole::Recurrent,
            )
            .unwrap();
        let (_o, s_out) =
            gated_delta_net_decode(&db, q_in, k_in, v_in, g_in, bt_in, ds_in).unwrap();
        let dg = db
            .finish(poot_graph_ir::Outputs::new(_o).with_state(&[(ds_in, s_out)]))
            .unwrap();
        let mut state = HostTensor::f32(vec![1, hv, d, d], vec![0.0; hv * d * d]);
        for t in 0..l {
            let qt: Vec<f32> = (0..hv * d)
                .map(|i| qv[(i / d * l + t) * d + i % d])
                .collect();
            let kt: Vec<f32> = (0..hv * d)
                .map(|i| kv[(i / d * l + t) * d + i % d])
                .collect();
            let vt: Vec<f32> = (0..hv * d)
                .map(|i| vd[(i / d * l + t) * d + i % d])
                .collect();
            let gt: Vec<f32> = (0..hv).map(|hh| gd[hh * l + t]).collect();
            let btt: Vec<f32> = (0..hv).map(|hh| betad[hh * l + t]).collect();
            let mut inp = HashMap::new();
            inp.insert(q_in.id(), HostTensor::f32(vec![1, hv, 1, d], qt));
            inp.insert(k_in.id(), HostTensor::f32(vec![1, hv, 1, d], kt));
            inp.insert(v_in.id(), HostTensor::f32(vec![1, hv, 1, d], vt));
            inp.insert(g_in.id(), HostTensor::f32(vec![1, hv, 1, 1], gt));
            inp.insert(bt_in.id(), HostTensor::f32(vec![1, hv, 1, 1], btt));
            inp.insert(ds_in.id(), state.clone());
            state = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
                let values: HashMap<ValueId, Value> = inp
                    .iter()
                    .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                    .collect();
                let evaluation =
                    crate::eval(&dg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })()
            .unwrap()
            .1
            .into_iter()
            .next()
            .unwrap();
        }
        let want_state = state.as_f32().unwrap().to_vec();

        // Chunk-size sweep of the chunked core on the same inputs. Cancellation in the within-chunk cumulative
        // log-decay prefix sum G_i (tril_incl @ g) scales with the chunk's cumulative magnitude, so a smaller
        // chunk should reduce the error if cancellation is the cause.
        eprintln!("--- dt_scale={dt_scale} g_range=[{gmin:.2},{gmax:.2}] ---");
        for &c in &[64usize, 32, 16, 8] {
            let mut ti = vec![0.0f32; c * c];
            let mut ts = vec![0.0f32; c * c];
            for i in 0..c {
                for j in 0..c {
                    if j <= i {
                        ti[i * c + j] = 1.0;
                    }
                    if j < i {
                        ts[i * c + j] = 1.0;
                    }
                }
            }
            let cb = Builder::for_tests();
            let cq = cb
                .constant("q", TensorType::f32(vec![1, hk, l, d]))
                .unwrap();
            let ckk = cb
                .constant("k", TensorType::f32(vec![1, hk, l, d]))
                .unwrap();
            let cv = cb
                .constant("v", TensorType::f32(vec![1, hv, l, d]))
                .unwrap();
            let cg = cb
                .constant("g", TensorType::f32(vec![1, hv, l, 1]))
                .unwrap();
            let cbeta = cb
                .constant("beta", TensorType::f32(vec![1, hv, l, 1]))
                .unwrap();
            let cs = cb
                .state_input(
                    "s",
                    TensorType::f32(vec![1, hv, d, d]),
                    StateRole::Recurrent,
                )
                .unwrap();
            let cti = cb
                .constant("ti", TensorType::f32(vec![1, 1, c, c]))
                .unwrap();
            let cts = cb
                .constant("ts", TensorType::f32(vec![1, 1, c, c]))
                .unwrap();
            let (co, cso) =
                gdn_prefill_chunked(&cb, cq, ckk, cv, cg, cbeta, cs, cti, cts, c).unwrap();
            let cgraph = cb
                .finish(poot_graph_ir::Outputs::new(co).with_state(&[(cs, cso)]))
                .unwrap();
            let mut cin = HashMap::new();
            cin.insert(cq.id(), HostTensor::f32(vec![1, hk, l, d], qd.clone()));
            cin.insert(ckk.id(), HostTensor::f32(vec![1, hk, l, d], kd.clone()));
            cin.insert(cv.id(), HostTensor::f32(vec![1, hv, l, d], vd.clone()));
            cin.insert(cg.id(), HostTensor::f32(vec![1, hv, l, 1], gd.clone()));
            cin.insert(
                cbeta.id(),
                HostTensor::f32(vec![1, hv, l, 1], betad.clone()),
            );
            cin.insert(
                cs.id(),
                HostTensor::f32(vec![1, hv, d, d], vec![0.0; hv * d * d]),
            );
            cin.insert(cti.id(), HostTensor::f32(vec![1, 1, c, c], ti));
            cin.insert(cts.id(), HostTensor::f32(vec![1, 1, c, c], ts));
            let (_go, gss) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
                let values: HashMap<ValueId, Value> = cin
                    .iter()
                    .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                    .collect();
                let evaluation =
                    crate::eval(&cgraph, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })()
            .unwrap();
            let got_state = gss.into_iter().next().unwrap().as_f32().unwrap().to_vec();
            let mut worst = (0usize, 0.0f32);
            for hh in 0..hv {
                let head = hh * d * d..(hh + 1) * d * d;
                // A probe that prints rather than asserts, so a non-finite chunked state is reported as an infinite
                // error instead of failing the test. At dt_scale=8 the front end's `ops::softplus` (`ln(1 + exp(x))`)
                // overflows for x > 88.7, g becomes -inf, and the chunked cumulative-decay sum yields NaN: the NaN this
                // sweep printed as 0 while the error was a NaN-blind fold (R482-002).
                let he = if got_state[head.clone()].iter().all(|v| v.is_finite()) {
                    max_abs_error(&got_state[head.clone()], &want_state[head])
                } else {
                    f32::INFINITY
                };
                if he > worst.1 {
                    worst = (hh, he);
                }
            }
            eprintln!(
                "  chunk={c:2}: WORST head {} state_err={:.3e}",
                worst.0, worst.1
            );
        }
    }
}

#[test]
fn mamba_core_path_decode_loop_matches_reference() {
    // Mamba2 SSM core path: causal conv1d -> SiLU -> SSD with Delta = softplus(dt), run step by step with conv
    // cache and SSM state carried through the State mechanism, must equal a hand-rolled Mamba reference.
    // Hq=2 heads, P=4 head dim (conv channels = Hq*P = 8), N=3 state, K=4 conv.
    use poot_graph_ir::ops::{causal_conv1d_decode, mamba2_ssd_decode, softplus};
    let (l, hq, pdim, n, kk) = (4usize, 2usize, 4usize, 3usize, 4usize);
    let ch = hq * pdim; // conv channels = 8
    let xcv = |t: usize, c: usize| ((t * 3 + c) % 7) as f32 * 0.1 - 0.3; // conv input [L, ch]
    let bv = |t: usize, h: usize, i: usize| ((t + h * 2 + i) % 5) as f32 * 0.1 - 0.2; // B [L,Hq,N]
    let cvv = |t: usize, h: usize, i: usize| ((t * 2 + h + i) % 6) as f32 * 0.1 - 0.25; // C
    let dtv = |t: usize, h: usize| 0.1 + 0.15 * ((t + h) % 3) as f32; // dt pre-softplus [L,Hq]
    let wcv = |k: usize, c: usize| ((k + c) % 5) as f32 * 0.1 - 0.2; // conv kernel [K, ch]
    let av = |h: usize| -0.5 - 0.2 * h as f32;
    let dv = |h: usize| 0.2 + 0.1 * h as f32;
    let silu = |x: f32| x / (1.0 + (-x).exp());

    // Decode graph (one step), carrying conv cache + SSM state.
    let b = Builder::for_tests();
    let xc = b.constant("xc", TensorType::f32(vec![1, 1, ch])).unwrap(); // conv input
    let wc = b.constant("wc", TensorType::f32(vec![kk, ch])).unwrap(); // conv kernel
    let bb = b.constant("b", TensorType::f32(vec![1, hq, 1, n])).unwrap();
    let cc = b.constant("c", TensorType::f32(vec![1, hq, 1, n])).unwrap();
    let dt = b
        .constant("dt", TensorType::f32(vec![1, hq, 1, 1]))
        .unwrap(); // pre-softplus
    let ap = b.constant("a", TensorType::f32(vec![1, hq, 1, 1])).unwrap();
    let dsk = b.constant("d", TensorType::f32(vec![1, hq, 1, 1])).unwrap();
    let conv_cache = b
        .state_input(
            "cv",
            TensorType::f32(vec![1, kk - 1, ch]),
            StateRole::Recurrent,
        )
        .unwrap();
    let h_in = b
        .state_input(
            "h",
            TensorType::f32(vec![1, hq, n, pdim]),
            StateRole::Recurrent,
        )
        .unwrap();
    // conv -> silu, reshape [1,1,ch] -> [1,Hq,1,P]
    let (conv_out, conv_cache_out) = causal_conv1d_decode(&b, xc, wc, conv_cache, kk).unwrap();
    let xact = poot_graph_ir::ops::silu(&b, conv_out).unwrap();
    let xheads = b.reshape(xact, vec![1, hq, 1, pdim]).unwrap();
    let delta = softplus(&b, dt).unwrap(); // Delta = softplus(dt)
    let (y, h_out) = mamba2_ssd_decode(&b, xheads, bb, cc, delta, ap, dsk, h_in).unwrap();
    let g = b
        .finish(
            poot_graph_ir::Outputs::new(y)
                .with_state(&[(conv_cache, conv_cache_out), (h_in, h_out)]),
        )
        .unwrap();

    let bind_scalars = |inp: &mut HashMap<ValueId, HostTensor>| {
        inp.insert(
            wc.id(),
            HostTensor::f32(
                vec![kk, ch],
                (0..kk * ch).map(|i| wcv(i / ch, i % ch)).collect(),
            ),
        );
        inp.insert(
            ap.id(),
            HostTensor::f32(vec![1, hq, 1, 1], (0..hq).map(av).collect()),
        );
        inp.insert(
            dsk.id(),
            HostTensor::f32(vec![1, hq, 1, 1], (0..hq).map(dv).collect()),
        );
    };

    let mut cv_cache = HostTensor::f32(vec![1, kk - 1, ch], vec![0.0f32; (kk - 1) * ch]);
    let mut h_state = HostTensor::f32(vec![1, hq, n, pdim], vec![0.0f32; hq * n * pdim]);
    let mut got = vec![0.0f32; hq * l * pdim];
    for t in 0..l {
        let mut inp = HashMap::new();
        inp.insert(
            xc.id(),
            HostTensor::f32(vec![1, 1, ch], (0..ch).map(|c| xcv(t, c)).collect()),
        );
        inp.insert(
            bb.id(),
            HostTensor::f32(
                vec![1, hq, 1, n],
                (0..hq * n).map(|i| bv(t, i / n, i % n)).collect(),
            ),
        );
        inp.insert(
            cc.id(),
            HostTensor::f32(
                vec![1, hq, 1, n],
                (0..hq * n).map(|i| cvv(t, i / n, i % n)).collect(),
            ),
        );
        inp.insert(
            dt.id(),
            HostTensor::f32(vec![1, hq, 1, 1], (0..hq).map(|h| dtv(t, h)).collect()),
        );
        bind_scalars(&mut inp);
        inp.insert(conv_cache.id(), cv_cache.clone());
        inp.insert(h_in.id(), h_state.clone());
        let (yt, new) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
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
        cv_cache = new[0].clone();
        h_state = new[1].clone();
        for h in 0..hq {
            for p in 0..pdim {
                got[(h * l + t) * pdim + p] = yt.as_f32().unwrap()[h * pdim + p];
            }
        }
    }

    // Hand reference: conv -> silu per channel, then SSD per head with Delta=softplus(dt).
    // Conv channel c maps to (head = c/P, p = c%P).
    let mut want = vec![0.0f32; hq * l * pdim];
    let mut state = vec![0.0f32; hq * n * pdim];
    for t in 0..l {
        // conv + silu, per channel.
        let mut xc_act = vec![0.0f32; ch];
        for (c, slot) in xc_act.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..kk {
                let src = t as isize - (kk as isize - 1) + k as isize;
                if src >= 0 {
                    acc += wcv(k, c) * xcv(src as usize, c);
                }
            }
            *slot = silu(acc);
        }
        for h in 0..hq {
            let delta_h = (1.0 + dtv(t, h).exp()).ln(); // softplus
            let a_bar = (delta_h * av(h)).exp();
            for nn in 0..n {
                let b_bar = delta_h * bv(t, h, nn);
                for p in 0..pdim {
                    let xval = xc_act[h * pdim + p];
                    state[(h * n + nn) * pdim + p] =
                        a_bar * state[(h * n + nn) * pdim + p] + b_bar * xval;
                }
            }
            for p in 0..pdim {
                let mut acc = 0.0f32;
                for nn in 0..n {
                    acc += cvv(t, h, nn) * state[(h * n + nn) * pdim + p];
                }
                want[(h * l + t) * pdim + p] = acc + dv(h) * xc_act[h * pdim + p];
            }
        }
    }
    assert_close_rel(&got, &want, 1e-4);
}

#[test]
fn mamba2_ssd_decode_matches_recurrence() {
    // Mamba2 / SSD recurrence (h_t = exp(Delta*A)*h_{t-1} + (Delta*B)^T x_t; y = C@h + D*x) run step by step
    // via the State mechanism must equal the hand-rolled SSM recurrence. Hq=2 heads, N=3, P=4, L=4.
    let (l, hq, n, p) = (4usize, 2usize, 3usize, 4usize);
    let xv = |h: usize, t: usize, i: usize| ((h * 3 + t * 2 + i) % 7) as f32 * 0.1 - 0.3;
    let bv = |h: usize, t: usize, i: usize| ((h * 5 + t + i * 2) % 5) as f32 * 0.1 - 0.2;
    let cv = |h: usize, t: usize, i: usize| ((h + t * 3 + i) % 6) as f32 * 0.1 - 0.25;
    let dlt = |h: usize, t: usize| 0.3 + 0.2 * ((h + t) % 3) as f32; // timestep in (0.3, 0.7)
    let av = |h: usize| -0.5 - 0.3 * h as f32; // learned A (negative -> Abar in (0,1))
    let dv = |h: usize| 0.2 + 0.1 * h as f32; // skip D

    // build the decode graph (one step), carry h.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, hq, 1, p])).unwrap();
    let bin = b.constant("b", TensorType::f32(vec![1, hq, 1, n])).unwrap();
    let cin = b.constant("c", TensorType::f32(vec![1, hq, 1, n])).unwrap();
    let delta = b
        .constant("dl", TensorType::f32(vec![1, hq, 1, 1]))
        .unwrap();
    let a_param = b.constant("a", TensorType::f32(vec![1, hq, 1, 1])).unwrap();
    let d_skip = b.constant("d", TensorType::f32(vec![1, hq, 1, 1])).unwrap();
    let h_in = b
        .state_input(
            "h",
            TensorType::f32(vec![1, hq, n, p]),
            StateRole::Recurrent,
        )
        .unwrap();
    let (y, h_out) = mamba2_ssd_decode(&b, x, bin, cin, delta, a_param, d_skip, h_in).unwrap();
    let g = b
        .finish(poot_graph_ir::Outputs::new(y).with_state(&[(h_in, h_out)]))
        .unwrap();
    let ad: Vec<f32> = (0..hq).map(av).collect();
    let dd: Vec<f32> = (0..hq).map(dv).collect();

    let mut caches = vec![HostTensor::f32(vec![1, hq, n, p], vec![0.0f32; hq * n * p])];
    let mut got = vec![0.0f32; hq * l * p];
    for t in 0..l {
        let xt: Vec<f32> = (0..hq)
            .flat_map(|h| (0..p).map(move |i| xv(h, t, i)))
            .collect();
        let bt: Vec<f32> = (0..hq)
            .flat_map(|h| (0..n).map(move |i| bv(h, t, i)))
            .collect();
        let ct: Vec<f32> = (0..hq)
            .flat_map(|h| (0..n).map(move |i| cv(h, t, i)))
            .collect();
        let dlt_t: Vec<f32> = (0..hq).map(|h| dlt(h, t)).collect();
        let mut inp = HashMap::new();
        inp.insert(x.id(), HostTensor::f32(vec![1, hq, 1, p], xt));
        inp.insert(bin.id(), HostTensor::f32(vec![1, hq, 1, n], bt));
        inp.insert(cin.id(), HostTensor::f32(vec![1, hq, 1, n], ct));
        inp.insert(delta.id(), HostTensor::f32(vec![1, hq, 1, 1], dlt_t));
        inp.insert(a_param.id(), HostTensor::f32(vec![1, hq, 1, 1], ad.clone()));
        inp.insert(d_skip.id(), HostTensor::f32(vec![1, hq, 1, 1], dd.clone()));
        inp.insert(h_in.id(), caches[0].clone());
        let (yt, new) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
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
        caches = new;
        for h in 0..hq {
            for i in 0..p {
                got[(h * l + t) * p + i] = yt.as_f32().unwrap()[h * p + i];
            }
        }
    }

    // hand SSM recurrence per head.
    let mut want = vec![0.0f32; hq * l * p];
    for h in 0..hq {
        let mut state = vec![0.0f32; n * p]; // h [N, P]
        for t in 0..l {
            let a_bar = (dlt(h, t) * av(h)).exp();
            for nn in 0..n {
                let b_bar = dlt(h, t) * bv(h, t, nn);
                for pp in 0..p {
                    state[nn * p + pp] = a_bar * state[nn * p + pp] + b_bar * xv(h, t, pp);
                }
            }
            for pp in 0..p {
                let mut acc = 0.0f32;
                for nn in 0..n {
                    acc += cv(h, t, nn) * state[nn * p + pp];
                }
                want[(h * l + t) * p + pp] = acc + dv(h) * xv(h, t, pp);
            }
        }
    }
    assert_close_rel(&got, &want, 1e-5);
}
