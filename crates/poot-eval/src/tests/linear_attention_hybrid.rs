//! Linear-attention fuzz, hybrid model loops, decay masks, mamba2 block, layer-level decode-vs-prefill.

use crate::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::builder::{Builder, Traced};
use poot_graph_ir::graph::{Operand, StateRole};
use poot_graph_ir::op::{BinOp, OpKind};
use poot_graph_ir::ops::{
    attention, attention_prefill, causal_conv1d_decode, linear, linear_attention_decode,
    linear_attention_prefill, rmsnorm,
};
use poot_graph_ir::types::TensorType;
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::{assert_close_rel, max_abs_error, max_abs_error_f64};

/// Stage one `FlashAttentionPrefill { softcap: None }` over `[q, k, v, mask]`. Card 557: `Builder` has no
/// method for a composite (`compile`'s `flash_attention_capped` forms it from the traced chain), so a test
/// of the op's own eval stages it through an append plan, as `e4m3_per_channel.rs` stages a
/// `PackedContraction`.
fn stage_flash_prefill(
    b: &Builder,
    [q, k, v, mask]: [Traced; 4],
    n_rep: usize,
    scale: f32,
) -> Traced {
    let mut plan = b.append_plan(0);
    let out = plan
        .equation(
            OpKind::FlashAttentionPrefill {
                n_rep,
                scale,
                softcap: None,
            },
            [q, k, v, mask]
                .iter()
                .map(|t| Operand::Value(t.id))
                .collect(),
        )
        .expect("stage FlashAttentionPrefill");
    plan.declare_result(out).expect("declare result");
    let mut prepared = b.preflight_append(plan).expect("preflight append");
    Traced {
        id: b.commit_append(&mut prepared).expect("commit append"),
    }
}

#[test]
fn linear_attention_decode_fuzz_matches_recurrence() {
    // card 035: independent-oracle shape fuzzer for the linear-attention decode family (scalar + gated). Sweeps random
    // Hkv / n_rep / D_k / D_v / L (rectangular state, GQA repeat) and runs the multi-step decode loop carrying S against
    // a from-scratch recurrence. GPU==CPU cannot catch a state-shape derivation bug (same decomposition); an independent
    // reference can. 40 seeds.
    use poot_graph_ir::ops::{linear_attention_decode, linear_attention_decode_gated};
    let mut s: u64 = 0xA17F_C0DE_1234_5678;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut covered = (0, 0); // (scalar, gated) seeds run
    for _seed in 0..40u32 {
        let hkv = 1 + (rng() % 2) as usize; // 1..=2
        let n_rep = 1 + (rng() % 3) as usize; // 1..=3
        let hq = hkv * n_rep;
        let dk = 2 + (rng() % 5) as usize; // 2..=6
        let dv = 2 + (rng() % 5) as usize; // 2..=6
        let l = 2 + (rng() % 4) as usize; // 2..=5
        let gated = rng() % 2 == 0;
        let decay = 0.7 + 0.1 * (rng() % 3) as f32; // scalar variant: 0.7 / 0.8 / 0.9
        let rf = |r: &mut dyn FnMut() -> u64| ((r() % 1000) as f32 / 1000.0) * 1.6 - 0.8; // value in (-0.8, 0.8)
        // per-step random q/k/v; per-head/D_k gate (gated variant), in (0.3, 0.95).
        let mut q = vec![vec![0.0f32; hq * dk]; l];
        let mut k = vec![vec![0.0f32; hkv * dk]; l];
        let mut v = vec![vec![0.0f32; hkv * dv]; l];
        let mut gate = vec![vec![0.0f32; hq * dk]; l];
        for t in 0..l {
            for x in q[t].iter_mut() {
                *x = rf(&mut rng);
            }
            for x in k[t].iter_mut() {
                *x = rf(&mut rng);
            }
            for x in v[t].iter_mut() {
                *x = rf(&mut rng);
            }
            for x in gate[t].iter_mut() {
                *x = 0.3 + 0.65 * (rng() % 1000) as f32 / 1000.0;
            }
        }

        // build the one-step decode graph carrying S [1,Hq,D_k,D_v].
        let b = Builder::new();
        let qc = b.constant("q", TensorType::f32(vec![1, hq, 1, dk]));
        let kc = b.constant("k", TensorType::f32(vec![1, hkv, 1, dk]));
        let vc = b.constant("v", TensorType::f32(vec![1, hkv, 1, dv]));
        let s_in = b.state_input(
            "s",
            TensorType::f32(vec![1, hq, dk, dv]),
            StateRole::Recurrent,
        );
        let gc = b.constant("g", TensorType::f32(vec![1, hq, dk, 1]));
        let (o, s_out) = if gated {
            linear_attention_decode_gated(&b, qc, kc, vc, n_rep, gc, s_in)
        } else {
            linear_attention_decode(&b, qc, kc, vc, n_rep, decay, s_in)
        };
        let g = b.finish_with_state(o, &[(s_in, s_out)]);

        let mut state = HostTensor::f32(vec![1, hq, dk, dv], vec![0.0f32; hq * dk * dv]);
        let mut got = vec![0.0f32; l * hq * dv];
        for t in 0..l {
            let mut inp = HashMap::new();
            inp.insert(
                qc.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, dk], q[t].clone())),
            );
            inp.insert(
                kc.id,
                Value::from(HostTensor::f32(vec![1, hkv, 1, dk], k[t].clone())),
            );
            inp.insert(
                vc.id,
                Value::from(HostTensor::f32(vec![1, hkv, 1, dv], v[t].clone())),
            );
            if gated {
                // gate broadcasts over D_v: shape [1,Hq,D_k,1].
                let gv: Vec<f32> = (0..hq * dk).map(|i| gate[t][i]).collect();
                inp.insert(gc.id, Value::from(HostTensor::f32(vec![1, hq, dk, 1], gv)));
            } else {
                inp.insert(
                    gc.id,
                    Value::from(HostTensor::f32(vec![1, hq, dk, 1], vec![0.0f32; hq * dk])),
                );
            }
            inp.insert(s_in.id, Value::from(state.clone()));
            let (ot, new) = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| {
                    let state = r
                        .state
                        .into_iter()
                        .map(|v| {
                            v.into_host()
                                .expect("linear_attention_hybrid state is dense")
                        })
                        .collect::<Vec<_>>();
                    (
                        r.output
                            .into_host()
                            .expect("linear_attention_hybrid tests evaluate dense graphs"),
                        state,
                    )
                })
                .unwrap();
            state = new[0].clone();
            got[t * hq * dv..(t + 1) * hq * dv].copy_from_slice(ot.as_f32().unwrap());
        }

        // reference recurrence: S[h][i][j] = gate_or_decay * S + k[kv][i]*v[kv][j]; o[h][j] = sum_i q[h][i]*S[h][i][j].
        let mut want = vec![0.0f32; l * hq * dv];
        let mut sref = vec![0.0f32; hq * dk * dv];
        for t in 0..l {
            for h in 0..hq {
                let kv = h / n_rep;
                for i in 0..dk {
                    let decw = if gated { gate[t][h * dk + i] } else { decay };
                    for j in 0..dv {
                        let idx = (h * dk + i) * dv + j;
                        sref[idx] = decw * sref[idx] + k[t][kv * dk + i] * v[t][kv * dv + j];
                    }
                }
                for j in 0..dv {
                    let mut acc = 0.0f32;
                    for i in 0..dk {
                        acc += q[t][h * dk + i] * sref[(h * dk + i) * dv + j];
                    }
                    want[t * hq * dv + h * dv + j] = acc;
                }
            }
        }
        assert_close_rel(&got, &want, 1e-4);
        if gated {
            covered.1 += 1;
        } else {
            covered.0 += 1;
        }
    }
    assert!(
        covered.0 > 0 && covered.1 > 0,
        "both variants exercised: {covered:?}"
    );
}

#[test]
fn mamba2_ssd_decode_fuzz_matches_recurrence() {
    // card 035: independent-oracle shape fuzzer for the Mamba2 / SSD decode step. Sweeps random Hq / N / P / L against a
    // from-scratch SSD recurrence (Abar=exp(Delta*A), Bbar=Delta*B, state update, C@state + D*x). Guards the [N,P] state
    // shape and the Delta/A/B/C/D broadcasts. 40 seeds.
    use poot_graph_ir::ops::mamba2_ssd_decode;
    let mut s: u64 = 0x5EED_1357_9BDF_2468;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let rf = |r: &mut dyn FnMut() -> u64| ((r() % 1000) as f32 / 1000.0) * 1.2 - 0.6;
    for _seed in 0..40u32 {
        let hq = 1 + (rng() % 3) as usize; // 1..=3
        let n = 2 + (rng() % 4) as usize; // 2..=5
        let p = 2 + (rng() % 4) as usize; // 2..=5
        let l = 2 + (rng() % 4) as usize; // 2..=5
        let a_par: Vec<f32> = (0..hq)
            .map(|h| -0.4 - 0.3 * h as f32 - 0.1 * (rng() % 3) as f32)
            .collect();
        let d_par: Vec<f32> = (0..hq).map(|_| 0.1 + 0.2 * (rng() % 4) as f32).collect();
        let mut xs = vec![vec![0.0f32; hq * p]; l];
        let mut bs = vec![vec![0.0f32; hq * n]; l];
        let mut cs = vec![vec![0.0f32; hq * n]; l];
        let mut dts = vec![vec![0.0f32; hq]; l];
        for t in 0..l {
            for x in xs[t].iter_mut() {
                *x = rf(&mut rng);
            }
            for x in bs[t].iter_mut() {
                *x = rf(&mut rng);
            }
            for x in cs[t].iter_mut() {
                *x = rf(&mut rng);
            }
            for x in dts[t].iter_mut() {
                *x = 0.1 + 0.5 * (rng() % 1000) as f32 / 1000.0;
            }
        }

        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, hq, 1, p]));
        let bin = b.constant("b", TensorType::f32(vec![1, hq, 1, n]));
        let cin = b.constant("c", TensorType::f32(vec![1, hq, 1, n]));
        let delta = b.constant("dl", TensorType::f32(vec![1, hq, 1, 1]));
        let a = b.constant("a", TensorType::f32(vec![1, hq, 1, 1]));
        let dsk = b.constant("d", TensorType::f32(vec![1, hq, 1, 1]));
        let h_in = b.state_input(
            "h",
            TensorType::f32(vec![1, hq, n, p]),
            StateRole::Recurrent,
        );
        let (y, h_out) = mamba2_ssd_decode(&b, x, bin, cin, delta, a, dsk, h_in);
        let g = b.finish_with_state(y, &[(h_in, h_out)]);

        let mut hstate = HostTensor::f32(vec![1, hq, n, p], vec![0.0f32; hq * n * p]);
        let mut got = vec![0.0f32; l * hq * p];
        for t in 0..l {
            let mut inp = HashMap::new();
            inp.insert(
                x.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, p], xs[t].clone())),
            );
            inp.insert(
                bin.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, n], bs[t].clone())),
            );
            inp.insert(
                cin.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, n], cs[t].clone())),
            );
            inp.insert(
                delta.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, 1], dts[t].clone())),
            );
            inp.insert(
                a.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, 1], a_par.clone())),
            );
            inp.insert(
                dsk.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, 1], d_par.clone())),
            );
            inp.insert(h_in.id, Value::from(hstate.clone()));
            let (yt, new) = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| {
                    let state = r
                        .state
                        .into_iter()
                        .map(|v| {
                            v.into_host()
                                .expect("linear_attention_hybrid state is dense")
                        })
                        .collect::<Vec<_>>();
                    (
                        r.output
                            .into_host()
                            .expect("linear_attention_hybrid tests evaluate dense graphs"),
                        state,
                    )
                })
                .unwrap();
            hstate = new[0].clone();
            got[t * hq * p..(t + 1) * hq * p].copy_from_slice(yt.as_f32().unwrap());
        }

        let mut want = vec![0.0f32; l * hq * p];
        let mut state = vec![0.0f32; hq * n * p];
        for t in 0..l {
            for h in 0..hq {
                let a_bar = (dts[t][h] * a_par[h]).exp();
                for nn in 0..n {
                    let b_bar = dts[t][h] * bs[t][h * n + nn];
                    for pp in 0..p {
                        let idx = (h * n + nn) * p + pp;
                        state[idx] = a_bar * state[idx] + b_bar * xs[t][h * p + pp];
                    }
                }
                for pp in 0..p {
                    let mut acc = 0.0f32;
                    for nn in 0..n {
                        acc += cs[t][h * n + nn] * state[(h * n + nn) * p + pp];
                    }
                    want[t * hq * p + h * p + pp] = acc + d_par[h] * xs[t][h * p + pp];
                }
            }
        }
        assert_close_rel(&got, &want, 1e-4);
    }
}

#[test]
fn mamba2_ssd_prefill_matches_decode_loop() {
    // `mamba2_ssd_prefill` (`decay_mask_from_gates` + `linear_attention_prefill`) shape fuzzer. Ground truth is
    // `mamba2_ssd_decode` (verified by `mamba2_ssd_decode_fuzz_matches_recurrence`) run `L` times from a zero state.
    // Checks both the per-position `y` and the final SSM state `h_out`. Random Hq/N/P/L, 30 seeds.
    use poot_graph_ir::ops::{mamba2_ssd_decode, mamba2_ssd_prefill};
    let mut s: u64 = 0x9A55_3311_7788_BBCC;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let rf = |r: &mut dyn FnMut() -> u64| ((r() % 1000) as f32 / 1000.0) * 1.2 - 0.6;
    for _seed in 0..30u32 {
        let hq = 1 + (rng() % 3) as usize; // 1..=3
        let n = 2 + (rng() % 4) as usize; // 2..=5
        let p = 2 + (rng() % 4) as usize; // 2..=5
        let l = 2 + (rng() % 5) as usize; // 2..=6
        let a_par: Vec<f32> = (0..hq)
            .map(|h| -0.4 - 0.3 * h as f32 - 0.1 * (rng() % 3) as f32)
            .collect();
        let d_par: Vec<f32> = (0..hq).map(|_| 0.1 + 0.2 * (rng() % 4) as f32).collect();
        // step-major: xs[t] is [Hq*P], bs[t]/cs[t] are [Hq*N], dts[t] is [Hq].
        let mut xs = vec![vec![0.0f32; hq * p]; l];
        let mut bs = vec![vec![0.0f32; hq * n]; l];
        let mut cs = vec![vec![0.0f32; hq * n]; l];
        let mut dts = vec![vec![0.0f32; hq]; l];
        for t in 0..l {
            for x in xs[t].iter_mut() {
                *x = rf(&mut rng);
            }
            for x in bs[t].iter_mut() {
                *x = rf(&mut rng);
            }
            for x in cs[t].iter_mut() {
                *x = rf(&mut rng);
            }
            for x in dts[t].iter_mut() {
                *x = 0.1 + 0.5 * (rng() % 1000) as f32 / 1000.0;
            }
        }

        // (1) decode loop, L steps from a zero state (ground truth).
        let bd = Builder::new();
        let xd = bd.constant("x", TensorType::f32(vec![1, hq, 1, p]));
        let bind = bd.constant("b", TensorType::f32(vec![1, hq, 1, n]));
        let cind = bd.constant("c", TensorType::f32(vec![1, hq, 1, n]));
        let deltad = bd.constant("dl", TensorType::f32(vec![1, hq, 1, 1]));
        let ad = bd.constant("a", TensorType::f32(vec![1, hq, 1, 1]));
        let dskd = bd.constant("d", TensorType::f32(vec![1, hq, 1, 1]));
        let h_ind = bd.state_input(
            "h",
            TensorType::f32(vec![1, hq, n, p]),
            StateRole::Recurrent,
        );
        let (yd, h_outd) = mamba2_ssd_decode(&bd, xd, bind, cind, deltad, ad, dskd, h_ind);
        let gd = bd.finish_with_state(yd, &[(h_ind, h_outd)]);

        let mut hstate = HostTensor::f32(vec![1, hq, n, p], vec![0.0f32; hq * n * p]);
        let mut want_y = vec![0.0f32; l * hq * p]; // step-major [L,Hq,P]
        for t in 0..l {
            let mut inp = HashMap::new();
            inp.insert(
                xd.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, p], xs[t].clone())),
            );
            inp.insert(
                bind.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, n], bs[t].clone())),
            );
            inp.insert(
                cind.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, n], cs[t].clone())),
            );
            inp.insert(
                deltad.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, 1], dts[t].clone())),
            );
            inp.insert(
                ad.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, 1], a_par.clone())),
            );
            inp.insert(
                dskd.id,
                Value::from(HostTensor::f32(vec![1, hq, 1, 1], d_par.clone())),
            );
            inp.insert(h_ind.id, Value::from(hstate.clone()));
            let (yt, new) = eval(&gd, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| {
                    let state = r
                        .state
                        .into_iter()
                        .map(|v| {
                            v.into_host()
                                .expect("linear_attention_hybrid state is dense")
                        })
                        .collect::<Vec<_>>();
                    (
                        r.output
                            .into_host()
                            .expect("linear_attention_hybrid tests evaluate dense graphs"),
                        state,
                    )
                })
                .unwrap();
            hstate = new[0].clone();
            want_y[t * hq * p..(t + 1) * hq * p].copy_from_slice(yt.as_f32().unwrap());
        }
        let want_h = hstate.as_f32().unwrap().to_vec(); // [1,Hq,N,P] head-major already

        // (2) one-shot prefill graph over all L positions (head-major [1,Hq,L,*] layout).
        let bp = Builder::new();
        let xp = bp.constant("x", TensorType::f32(vec![1, hq, l, p]));
        let binp = bp.constant("b", TensorType::f32(vec![1, hq, l, n]));
        let cinp = bp.constant("c", TensorType::f32(vec![1, hq, l, n]));
        let deltap = bp.constant("dl", TensorType::f32(vec![1, hq, l, 1]));
        let ap = bp.constant("a", TensorType::f32(vec![1, hq, 1, 1]));
        let dskp = bp.constant("d", TensorType::f32(vec![1, hq, 1, 1]));
        let trilp = bp.constant("tr", TensorType::f32(vec![1, hq, l, l]));
        let (yp, h_outp) = mamba2_ssd_prefill(&bp, xp, binp, cinp, deltap, ap, dskp, trilp);
        let gy = bp.finish(yp);
        let mut gh = gy.clone();
        gh.output = h_outp.id;

        // Reindex the step-major generated data into head-major [Hq,L,*] for the prefill inputs.
        let mut x_hm = vec![0.0f32; hq * l * p];
        let mut b_hm = vec![0.0f32; hq * l * n];
        let mut c_hm = vec![0.0f32; hq * l * n];
        let mut d_hm = vec![0.0f32; hq * l];
        for h in 0..hq {
            for t in 0..l {
                x_hm[(h * l + t) * p..(h * l + t) * p + p]
                    .copy_from_slice(&xs[t][h * p..h * p + p]);
                b_hm[(h * l + t) * n..(h * l + t) * n + n]
                    .copy_from_slice(&bs[t][h * n..h * n + n]);
                c_hm[(h * l + t) * n..(h * l + t) * n + n]
                    .copy_from_slice(&cs[t][h * n..h * n + n]);
                d_hm[h * l + t] = dts[t][h];
            }
        }
        let mut tril = vec![0.0f32; hq * l * l];
        for h in 0..hq {
            for t in 0..l {
                for j in 0..=t {
                    tril[(h * l + t) * l + j] = 1.0;
                }
            }
        }

        let mut inp = HashMap::new();
        inp.insert(xp.id, Value::from(HostTensor::f32(vec![1, hq, l, p], x_hm)));
        inp.insert(
            binp.id,
            Value::from(HostTensor::f32(vec![1, hq, l, n], b_hm)),
        );
        inp.insert(
            cinp.id,
            Value::from(HostTensor::f32(vec![1, hq, l, n], c_hm)),
        );
        inp.insert(
            deltap.id,
            Value::from(HostTensor::f32(vec![1, hq, l, 1], d_hm)),
        );
        inp.insert(
            ap.id,
            Value::from(HostTensor::f32(vec![1, hq, 1, 1], a_par.clone())),
        );
        inp.insert(
            dskp.id,
            Value::from(HostTensor::f32(vec![1, hq, 1, 1], d_par.clone())),
        );
        inp.insert(
            trilp.id,
            Value::from(HostTensor::f32(vec![1, hq, l, l], tril)),
        );

        let got_y_hm = eval(&gy, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("linear_attention_hybrid tests evaluate dense graphs");
        let got_h = eval(&gh, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("linear_attention_hybrid tests evaluate dense graphs");

        // got_y_hm is head-major [1,Hq,L,P]; reindex to step-major [L,Hq,P] to compare against want_y.
        let mut got_y = vec![0.0f32; l * hq * p];
        for h in 0..hq {
            for t in 0..l {
                got_y[t * hq * p + h * p..t * hq * p + h * p + p].copy_from_slice(
                    &got_y_hm.as_f32().unwrap()[(h * l + t) * p..(h * l + t) * p + p],
                );
            }
        }

        assert_close_rel(&got_y, &want_y, 1e-4);
        assert_close_rel(got_h.as_f32().unwrap(), &want_h, 1e-4);
    }
}

#[test]
fn causal_conv1d_decode_fuzz_matches_reference() {
    // card 035: independent-oracle shape fuzzer for the depthwise causal conv1d decode step (the op that had the non-last-axis
    // keepdim reduce shape bug; this op now uses a last-axis reduce). Sweeps random channels C / kernel K / steps L, runs
    // the decode loop carrying the (K-1)-wide ring buffer against a from-scratch windowed-conv reference. 40 seeds.
    use poot_graph_ir::ops::causal_conv1d_decode;
    let mut s: u64 = 0xC0FF_EE15_F00D_BA77;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let rf = |r: &mut dyn FnMut() -> u64| ((r() % 1000) as f32 / 1000.0) * 1.4 - 0.7;
    for _seed in 0..40u32 {
        let c = 1 + (rng() % 8) as usize; // 1..=8 channels
        let kk = 2 + (rng() % 4) as usize; // 2..=5 kernel
        let l = 2 + (rng() % 5) as usize; // 2..=6 steps
        let wv: Vec<f32> = (0..kk * c).map(|_| rf(&mut rng)).collect(); // [K, C]
        let mut xs = vec![vec![0.0f32; c]; l];
        for row in xs.iter_mut() {
            for x in row.iter_mut() {
                *x = rf(&mut rng);
            }
        }

        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, c]));
        let w = b.constant("w", TensorType::f32(vec![kk, c]));
        let cache = b.state_input(
            "cv",
            TensorType::f32(vec![1, kk - 1, c]),
            StateRole::Recurrent,
        );
        let (out, new_cache) = causal_conv1d_decode(&b, x, w, cache, kk);
        let g = b.finish_with_state(out, &[(cache, new_cache)]);

        let mut cv = HostTensor::f32(vec![1, kk - 1, c], vec![0.0f32; (kk - 1) * c]);
        let mut got = vec![0.0f32; l * c];
        for t in 0..l {
            let mut inp = HashMap::new();
            inp.insert(
                x.id,
                Value::from(HostTensor::f32(vec![1, 1, c], xs[t].clone())),
            );
            inp.insert(w.id, Value::from(HostTensor::f32(vec![kk, c], wv.clone())));
            inp.insert(cache.id, Value::from(cv.clone()));
            let (ot, new) = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| {
                    let state = r
                        .state
                        .into_iter()
                        .map(|v| {
                            v.into_host()
                                .expect("linear_attention_hybrid state is dense")
                        })
                        .collect::<Vec<_>>();
                    (
                        r.output
                            .into_host()
                            .expect("linear_attention_hybrid tests evaluate dense graphs"),
                        state,
                    )
                })
                .unwrap();
            cv = new[0].clone();
            got[t * c..(t + 1) * c].copy_from_slice(ot.as_f32().unwrap());
        }

        // reference: out[t][ch] = sum_k w[k][ch] * window[k][ch], window = last K of x history (zero-padded).
        let mut want = vec![0.0f32; l * c];
        for t in 0..l {
            for ch in 0..c {
                let mut acc = 0.0f32;
                for k in 0..kk {
                    let idx = t as isize - (kk as isize - 1) + k as isize;
                    if idx >= 0 {
                        acc += wv[k * c + ch] * xs[idx as usize][ch];
                    }
                }
                want[t * c + ch] = acc;
            }
        }
        assert_close_rel(&got, &want, 1e-4);
    }
}

#[test]
fn causal_conv1d_decode_batch_generalizes_from_batch1() {
    // card 188: `causal_conv1d_decode` used to hardcode a batch=1 reshape, which would panic at trace time for B>1.
    // (a) B=1 matches a from-scratch hand-computed causal-conv reference.
    // (b) B=3, with distinct nonzero starting caches and distinct input sequences in ONE batched call, gives per row
    //     exactly the B=1 result for that row (no cross-row leakage from the `wk` broadcast multiply) and exactly the
    //     hand reference.
    let (k, c, l, b_batch) = (3usize, 5usize, 4usize, 3usize);
    let wd = fill(k * c, 100); // [K, C], a shared model-weight kernel - NOT batched, same for every row.

    // per-row distinct input sequences and distinct nonzero starting caches.
    let xs: Vec<Vec<Vec<f32>>> = (0..b_batch)
        .map(|row| {
            let flat = fill(l * c, 200 + row as u64);
            (0..l).map(|t| flat[t * c..(t + 1) * c].to_vec()).collect()
        })
        .collect();
    let init_caches: Vec<Vec<f32>> = (0..b_batch)
        .map(|row| fill((k - 1) * c, 300 + row as u64))
        .collect();

    // hand reference: independent causal conv per row, treating the row's starting cache as the K-1 prior inputs before
    // x[row][0] (oldest first, matching `concat(cache, x)` window order). out[t][ch] = sum_kk w[kk][ch] * history[t+kk][ch].
    let hand_ref = |row: usize| -> Vec<Vec<f32>> {
        let mut history: Vec<Vec<f32>> = (0..k - 1)
            .map(|j| init_caches[row][j * c..(j + 1) * c].to_vec())
            .collect();
        history.extend(xs[row].iter().cloned());
        (0..l)
            .map(|t| {
                (0..c)
                    .map(|ch| {
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            acc += wd[kk * c + ch] * history[t + kk][ch];
                        }
                        acc
                    })
                    .collect()
            })
            .collect()
    };
    let want: Vec<Vec<Vec<f32>>> = (0..b_batch).map(hand_ref).collect();

    // --- (a) B=1 regression check: one row at a time through causal_conv1d_decode alone. ---
    let single_outputs: Vec<Vec<Vec<f32>>> = (0..b_batch)
        .map(|row| {
            let b = Builder::new();
            let x = b.constant("x", TensorType::f32(vec![1, 1, c]));
            let w = b.constant("w", TensorType::f32(vec![k, c]));
            let cache = b.state_input(
                "cache",
                TensorType::f32(vec![1, k - 1, c]),
                StateRole::Recurrent,
            );
            let (out, new_cache) = causal_conv1d_decode(&b, x, w, cache, k);
            let g = b.finish_with_state(out, &[(cache, new_cache)]);

            let mut cv = HostTensor::f32(vec![1, k - 1, c], init_caches[row].clone());
            let mut got = vec![vec![0.0f32; c]; l];
            for (t, out_row) in got.iter_mut().enumerate() {
                let mut inp = HashMap::new();
                inp.insert(
                    x.id,
                    Value::from(HostTensor::f32(vec![1, 1, c], xs[row][t].clone())),
                );
                inp.insert(w.id, Value::from(HostTensor::f32(vec![k, c], wd.clone())));
                inp.insert(cache.id, Value::from(cv.clone()));
                let (o_t, new) = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                    .map(|r| {
                        let state = r
                            .state
                            .into_iter()
                            .map(|v| {
                                v.into_host()
                                    .expect("linear_attention_hybrid state is dense")
                            })
                            .collect::<Vec<_>>();
                        (
                            r.output
                                .into_host()
                                .expect("linear_attention_hybrid tests evaluate dense graphs"),
                            state,
                        )
                    })
                    .unwrap();
                cv = new[0].clone();
                *out_row = o_t.as_f32().unwrap().to_vec();
            }
            got
        })
        .collect();
    for row in 0..b_batch {
        for t in 0..l {
            assert_close_rel(&single_outputs[row][t], &want[row][t], 1e-5);
        }
    }

    // --- (b) B=3 batched call: the same 3 rows, packed together into ONE call per step. ---
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![b_batch, 1, c]));
    let w = b.constant("w", TensorType::f32(vec![k, c]));
    let cache = b.state_input(
        "cache",
        TensorType::f32(vec![b_batch, k - 1, c]),
        StateRole::Recurrent,
    );
    let (out, new_cache) = causal_conv1d_decode(&b, x, w, cache, k);
    assert_eq!(
        b.aval(out).shape,
        vec![b_batch, 1, c],
        "batched out shape [B,1,C]"
    );
    assert_eq!(
        b.aval(new_cache).shape,
        vec![b_batch, k - 1, c],
        "batched new_cache shape [B,K-1,C]"
    );
    let g = b.finish_with_state(out, &[(cache, new_cache)]);

    let mut cv_flat: Vec<f32> = (0..b_batch)
        .flat_map(|row| init_caches[row].clone())
        .collect();
    let mut got_batched = vec![vec![vec![0.0f32; c]; l]; b_batch];
    for t in 0..l {
        let mut inp = HashMap::new();
        let x_flat: Vec<f32> = (0..b_batch).flat_map(|row| xs[row][t].clone()).collect();
        inp.insert(
            x.id,
            Value::from(HostTensor::f32(vec![b_batch, 1, c], x_flat)),
        );
        inp.insert(w.id, Value::from(HostTensor::f32(vec![k, c], wd.clone())));
        inp.insert(
            cache.id,
            Value::from(HostTensor::f32(vec![b_batch, k - 1, c], cv_flat.clone())),
        );
        let (o_t, new) = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| {
                        v.into_host()
                            .expect("linear_attention_hybrid state is dense")
                    })
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention_hybrid tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();
        cv_flat = new[0].as_f32().unwrap().to_vec();
        for (row, slot) in got_batched.iter_mut().enumerate().take(b_batch) {
            slot[t] = o_t.as_f32().unwrap()[row * c..(row + 1) * c].to_vec();
        }
    }
    let mut max_abs_diff_vs_single = 0.0f32;
    for row in 0..b_batch {
        for t in 0..l {
            assert_close_rel(&got_batched[row][t], &want[row][t], 1e-5);
            assert_close_rel(&got_batched[row][t], &single_outputs[row][t], 1e-6);
            max_abs_diff_vs_single = max_abs_diff_vs_single
                .max(max_abs_error(&got_batched[row][t], &single_outputs[row][t]));
        }
    }
    // batched vs single-row B=1 should be exact (same ops, same fp order per row), not just within `assert_close_rel`'s relative
    // tolerance. Printed so a `--nocapture` run shows the observed max_abs.
    eprintln!(
        "causal_conv1d_decode_batch_generalizes_from_batch1: max_abs(batched - single) = {max_abs_diff_vs_single}"
    );
    assert!(
        max_abs_diff_vs_single < 1e-6,
        "batched row should be numerically identical to the B=1 call, got max_abs={max_abs_diff_vs_single}"
    );
}

#[test]
fn decay_mask_from_gates_fuzz_matches_cumprod() {
    // card 035: shape fuzzer for the in-graph decay mask (decay_mask_from_gates: log gates -> tril-matmul cumsum -> exp)
    // against a from-scratch cumprod reference. Guards the cumsum-as-matmul and [H,L,L] broadcast. 40 seeds.
    use poot_graph_ir::ops::decay_mask_from_gates;
    let mut s: u64 = 0xDEAD_BEEF_CAFE_F00D;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for _seed in 0..40u32 {
        let hh = 1 + (rng() % 3) as usize; // 1..=3
        let l = 1 + (rng() % 6) as usize; // 1..=6
        let gd: Vec<f32> = (0..hh * l)
            .map(|_| 0.3 + 0.65 * (rng() % 1000) as f32 / 1000.0)
            .collect();
        let mut tril = vec![0.0f32; hh * l * l];
        for h in 0..hh {
            for t in 0..l {
                for j in 0..=t {
                    tril[(h * l + t) * l + j] = 1.0;
                }
            }
        }
        let b = Builder::new();
        let gates = b.constant("g", TensorType::f32(vec![1, hh, 1, l]));
        let tr = b.constant("tr", TensorType::f32(vec![1, hh, l, l]));
        let m = decay_mask_from_gates(&b, gates, tr, hh, l);
        let g = b.finish(m);
        let mut inp = HashMap::new();
        inp.insert(
            gates.id,
            Value::from(HostTensor::f32(vec![1, hh, 1, l], gd.clone())),
        );
        inp.insert(tr.id, Value::from(HostTensor::f32(vec![1, hh, l, l], tril)));
        let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("linear_attention_hybrid tests evaluate dense graphs");

        let mut want = vec![0.0f32; hh * l * l];
        for h in 0..hh {
            for t in 0..l {
                for j in 0..=t {
                    let mut p = 1.0f32;
                    for k in (j + 1)..=t {
                        p *= gd[h * l + k];
                    }
                    want[(h * l + t) * l + j] = p;
                }
            }
        }
        assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
    }
}

// Exact per-channel gated linear-attention recurrence, the independent oracle for both prefill fuzzers:
// state[a][c] = g[t][a]*state[a][c] + k[kv][t][a]*v[kv][t][c]; o[t][c] = sum_a q[t][a]*state[a][c]. GQA via
// kv = h/n_rep. q/k/v/g are flat [head, L, D]; returns [Hq, L, D_v].
#[allow(clippy::too_many_arguments)]
fn gated_linattn_recurrence(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    hq: usize,
    n_rep: usize,
    l: usize,
    dk: usize,
    dv: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; hq * l * dv];
    for h in 0..hq {
        let kv = h / n_rep;
        let mut state = vec![0.0f32; dk * dv];
        for t in 0..l {
            for a in 0..dk {
                let gta = g[(h * l + t) * dk + a];
                for c in 0..dv {
                    state[a * dv + c] = gta * state[a * dv + c]
                        + k[(kv * l + t) * dk + a] * v[(kv * l + t) * dv + c];
                }
            }
            for c in 0..dv {
                let mut acc = 0.0f32;
                for a in 0..dk {
                    acc += q[(h * l + t) * dk + a] * state[a * dv + c];
                }
                out[(h * l + t) * dv + c] = acc;
            }
        }
    }
    out
}

#[test]
fn flash_prefill_matches_materialized_attention_prefill() {
    // card 038: the fused flash-prefill op (FlashAttentionPrefill, online softmax, never materializing [1,Hq,L,L])
    // must equal the materialized `attention_prefill` (softmax(scale*QK^T + mask) @ V), its definition.
    // Direct vs online softmax are different algorithms. Random Hkv / n_rep / D / L (multiple of the block), causal mask. 30 seeds.
    let mut s: u64 = 0xF1A5_4E11_2233_4455;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let rf = |r: &mut dyn FnMut() -> u64| ((r() % 1000) as f32 / 1000.0) * 1.4 - 0.7;
    for _seed in 0..30u32 {
        let hkv = 1 + (rng() % 2) as usize;
        let n_rep = 1 + (rng() % 3) as usize;
        let hq = hkv * n_rep;
        let d = 2 + (rng() % 6) as usize; // 2..=7
        let block = 2 + (rng() % 3) as usize; // 2..=4
        let n_blocks = 1 + (rng() % 4) as usize; // 1..=4
        let l = block * n_blocks;
        let scale = 0.3 + 0.2 * (rng() % 4) as f32; // not always 1/sqrt(D)
        let qd: Vec<f32> = (0..hq * l * d).map(|_| rf(&mut rng)).collect();
        let kd: Vec<f32> = (0..hkv * l * d).map(|_| rf(&mut rng)).collect();
        let vd: Vec<f32> = (0..hkv * l * d).map(|_| rf(&mut rng)).collect();
        // additive causal mask [1,1,L,L]: 0 if j<=t else -1e9.
        let mask: Vec<f32> = (0..l * l)
            .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
            .collect();

        let build = |sel: u8| -> HostTensor {
            let b = Builder::new();
            let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
            let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
            let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
            let m = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
            let o = match sel {
                0 => attention_prefill(&b, q, k, v, n_rep, scale, m),
                _ => stage_flash_prefill(&b, [q, k, v, m], n_rep, scale),
            };
            let g = b.finish(o);
            let mut inp = HashMap::new();
            inp.insert(
                q.id,
                Value::from(HostTensor::f32(vec![1, hq, l, d], qd.clone())),
            );
            inp.insert(
                k.id,
                Value::from(HostTensor::f32(vec![1, hkv, l, d], kd.clone())),
            );
            inp.insert(
                v.id,
                Value::from(HostTensor::f32(vec![1, hkv, l, d], vd.clone())),
            );
            inp.insert(
                m.id,
                Value::from(HostTensor::f32(vec![1, 1, l, l], mask.clone())),
            );
            eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .expect("linear_attention_hybrid tests evaluate dense graphs")
        };
        let reference = build(0);
        let fused = build(1);
        assert_close_rel(fused.as_f32().unwrap(), reference.as_f32().unwrap(), 5e-4);
    }
}

/// Real-magnitude hardening of `flash_prefill_matches_materialized_attention_prefill`, which uses tiny
/// synthetic Q/K/V (`[-0.7, 0.7]`), head_dim 2..=7 and a small scale. Real attention runs at head_dim=128 with an
/// "attention sink" outlier token whose raw score can be an order of magnitude above the rest of the row. This builds
/// Q/K/V at head_dim=128, Hkv=2, n_rep=4, scale 1/sqrt(128), L=128 (4 blocks of 32) with a 20x sink key at
/// position 0, and checks the flash-fused form against the materialized invariant and an
/// independent f64 ground truth. The max-shifted online softmax has no cancellation mechanism, so this stays tight
/// (see `softmax_stays_accurate_at_extreme_score_magnitude`); it is a regression guard.
#[test]
fn flash_attention_matches_f64_reference_at_real_head_dim_and_sink_outlier() {
    let (hkv, n_rep, d, block, n_blocks) = (2usize, 4usize, 128usize, 32usize, 4usize);
    let hq = hkv * n_rep;
    let l = block * n_blocks; // 128
    let scale = 1.0 / (d as f32).sqrt(); // real 1/sqrt(head_dim) scaling

    let mut qd = fill(hq * l * d, 0xA11CE);
    let mut kd = fill(hkv * l * d, 0xB0B);
    let vd = fill(hkv * l * d, 0xCAFE);
    // amplify every head's KEY at position 0 into an "attention sink" outlier (~20x), so several rows see one raw score
    // an order of magnitude past the rest of the row.
    for h in 0..hkv {
        for j in 0..d {
            kd[(h * l) * d + j] *= 20.0;
        }
    }
    // also widen Q/K past the small oracle's [-0.7,0.7].
    for v in qd.iter_mut() {
        *v *= 3.0;
    }
    for v in kd.iter_mut() {
        *v *= 3.0;
    }

    // additive causal mask [1,1,L,L].
    let mask: Vec<f32> = (0..l * l)
        .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
        .collect();

    let build = |sel: u8| -> HostTensor {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
        let m = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
        let o = match sel {
            0 => attention_prefill(&b, q, k, v, n_rep, scale, m),
            _ => stage_flash_prefill(&b, [q, k, v, m], n_rep, scale),
        };
        let g = b.finish(o);
        let mut inp = HashMap::new();
        inp.insert(
            q.id,
            Value::from(HostTensor::f32(vec![1, hq, l, d], qd.clone())),
        );
        inp.insert(
            k.id,
            Value::from(HostTensor::f32(vec![1, hkv, l, d], kd.clone())),
        );
        inp.insert(
            v.id,
            Value::from(HostTensor::f32(vec![1, hkv, l, d], vd.clone())),
        );
        inp.insert(
            m.id,
            Value::from(HostTensor::f32(vec![1, 1, l, l], mask.clone())),
        );
        eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("linear_attention_hybrid tests evaluate dense graphs")
    };
    let materialized = build(0);
    let fused = build(1);
    // absolute (not relative) comparison: output values near zero (rows dominated by the sink) make `close`'s relative
    // tolerance meaningless; the f64 ground-truth check below is the accuracy oracle.
    assert!(
        max_abs_error(fused.as_f32().unwrap(), materialized.as_f32().unwrap()) < 1e-3,
        "flash-fused vs materialized attention at real dims + sink outlier"
    );

    // independent f64 ground truth: causal softmax(scale*QK^T)@V per (head, row), GQA-repeated.
    let mut want = vec![0.0f64; hq * l * d];
    for hqi in 0..hq {
        let hk = hqi / n_rep;
        for i in 0..l {
            let mut scores = vec![0.0f64; i + 1];
            for j in 0..=i {
                let mut dot = 0.0f64;
                for c in 0..d {
                    dot += (qd[(hqi * l + i) * d + c] as f64) * (kd[(hk * l + j) * d + c] as f64);
                }
                scores[j] = dot * (scale as f64);
            }
            let m64 = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|&s| (s - m64).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for c in 0..d {
                let mut acc = 0.0f64;
                for j in 0..=i {
                    acc += (exps[j] / sum) * (vd[(hk * l + j) * d + c] as f64);
                }
                want[(hqi * l + i) * d + c] = acc;
            }
        }
    }
    let worst_materialized = max_abs_error_f64(materialized.as_f32().unwrap(), &want);
    let worst_fused = max_abs_error_f64(fused.as_f32().unwrap(), &want);
    eprintln!(
        "flash attention at real head_dim=128 + sink outlier vs f64 ground truth: \
         materialized={worst_materialized:e} fused={worst_fused:e}"
    );
    assert!(
        worst_fused < 1e-3,
        "flash-fused attention vs f64 ground truth at real dims + sink outlier: err={worst_fused:e}"
    );
}

#[test]
fn linear_attention_prefill_chunked_fuzz_matches_recurrence() {
    // card 035: independent-oracle shape fuzzer for the chunked gated linear-attention prefill (per-chunk slice/state-carry/
    // concat, the Lambda gated carry, Q'/K' beta scaling). Sweeps random chunk size / count / Hkv / n_rep / D_k / D_v
    // with a data-dependent per-channel gate; beta is the within-chunk cumulative gate product; compared against the
    // exact recurrence. Gates are mild (0.6..0.95, <=4 chunks) so K/beta stays well-conditioned. 36 seeds.
    use poot_graph_ir::ops::linear_attention_prefill_chunked;
    let mut s: u64 = 0x9111_2233_4455_6677;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let rf = |r: &mut dyn FnMut() -> u64| ((r() % 1000) as f32 / 1000.0) * 1.2 - 0.6;
    for _seed in 0..36u32 {
        let c = 2 + (rng() % 3) as usize; // chunk size 2..=4
        let n_chunks = 2 + (rng() % 3) as usize; // 2..=4 chunks
        let l = c * n_chunks;
        let hkv = 1 + (rng() % 2) as usize;
        let n_rep = 1 + (rng() % 2) as usize;
        let hq = hkv * n_rep;
        let dk = 2 + (rng() % 4) as usize; // 2..=5
        let dv = 2 + (rng() % 4) as usize;
        let qd: Vec<f32> = (0..hq * l * dk).map(|_| rf(&mut rng)).collect();
        let kd: Vec<f32> = (0..hkv * l * dk).map(|_| rf(&mut rng)).collect();
        let vd: Vec<f32> = (0..hkv * l * dv).map(|_| rf(&mut rng)).collect();
        // per (head, position, channel) gate in (0.6, 0.95).
        let gd: Vec<f32> = (0..hq * l * dk)
            .map(|_| 0.6 + 0.35 * (rng() % 1000) as f32 / 1000.0)
            .collect();
        // beta = within-chunk cumulative product of the gate (chunk start = (t/c)*c, inclusive of t).
        let mut beta = vec![0.0f32; hq * l * dk];
        for h in 0..hq {
            for a in 0..dk {
                for t in 0..l {
                    let cs = (t / c) * c;
                    let mut pb = 1.0f32;
                    for sidx in cs..=t {
                        pb *= gd[(h * l + sidx) * dk + a];
                    }
                    beta[(h * l + t) * dk + a] = pb;
                }
            }
        }
        let causal_chunk: Vec<f32> = (0..c * c)
            .map(|i| if i % c <= i / c { 1.0 } else { 0.0 })
            .collect();

        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, dk]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, l, dk]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, l, dv]));
        let be = b.constant("be", TensorType::f32(vec![1, hq, l, dk]));
        let ca = b.constant("ca", TensorType::f32(vec![1, 1, c, c]));
        let o = linear_attention_prefill_chunked(&b, q, k, v, n_rep, be, ca, c);
        let g = b.finish(o);
        let mut inp = HashMap::new();
        inp.insert(
            q.id,
            Value::from(HostTensor::f32(vec![1, hq, l, dk], qd.clone())),
        );
        inp.insert(
            k.id,
            Value::from(HostTensor::f32(vec![1, hkv, l, dk], kd.clone())),
        );
        inp.insert(
            v.id,
            Value::from(HostTensor::f32(vec![1, hkv, l, dv], vd.clone())),
        );
        inp.insert(
            be.id,
            Value::from(HostTensor::f32(vec![1, hq, l, dk], beta)),
        );
        inp.insert(
            ca.id,
            Value::from(HostTensor::f32(vec![1, 1, c, c], causal_chunk)),
        );
        let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("linear_attention_hybrid tests evaluate dense graphs");

        let want = gated_linattn_recurrence(&qd, &kd, &vd, &gd, hq, n_rep, l, dk, dv);
        assert_close_rel(got.as_f32().unwrap(), &want, 5e-3);
    }
}

#[test]
fn linear_attention_prefill_gated_fuzz_matches_recurrence() {
    // card 035: independent-oracle shape fuzzer for the non-chunked gated quadratic prefill (Q'=Q*A, K'=K/A, A =
    // whole-sequence cumulative gate, full causal mask) vs the same exact recurrence. L and gate are kept so A does not
    // underflow (the chunked form covers that). 36 seeds.
    use poot_graph_ir::ops::linear_attention_prefill_gated;
    let mut s: u64 = 0x1A2B_3C4D_5E6F_7081;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let rf = |r: &mut dyn FnMut() -> u64| ((r() % 1000) as f32 / 1000.0) * 1.2 - 0.6;
    for _seed in 0..36u32 {
        let l = 3 + (rng() % 8) as usize; // 3..=10
        let hkv = 1 + (rng() % 2) as usize;
        let n_rep = 1 + (rng() % 2) as usize;
        let hq = hkv * n_rep;
        let dk = 2 + (rng() % 4) as usize;
        let dv = 2 + (rng() % 4) as usize;
        let qd: Vec<f32> = (0..hq * l * dk).map(|_| rf(&mut rng)).collect();
        let kd: Vec<f32> = (0..hkv * l * dk).map(|_| rf(&mut rng)).collect();
        let vd: Vec<f32> = (0..hkv * l * dv).map(|_| rf(&mut rng)).collect();
        // mild gate (0.75, 0.97) so A = prod over up to 10 steps stays a normal f32 (>= 0.75^10 ~ 0.056).
        let gd: Vec<f32> = (0..hq * l * dk)
            .map(|_| 0.75 + 0.22 * (rng() % 1000) as f32 / 1000.0)
            .collect();
        // A = whole-sequence cumulative product of the gate (inclusive of t).
        let mut a_cum = vec![0.0f32; hq * l * dk];
        for h in 0..hq {
            for aa in 0..dk {
                let mut pa = 1.0f32;
                for t in 0..l {
                    pa *= gd[(h * l + t) * dk + aa];
                    a_cum[(h * l + t) * dk + aa] = pa;
                }
            }
        }
        let causal: Vec<f32> = (0..l * l)
            .map(|i| if i % l <= i / l { 1.0 } else { 0.0 })
            .collect();

        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, dk]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, l, dk]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, l, dv]));
        let cg = b.constant("cg", TensorType::f32(vec![1, hq, l, dk]));
        let m = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
        let o = linear_attention_prefill_gated(&b, q, k, v, n_rep, cg, m);
        let g = b.finish(o);
        let mut inp = HashMap::new();
        inp.insert(
            q.id,
            Value::from(HostTensor::f32(vec![1, hq, l, dk], qd.clone())),
        );
        inp.insert(
            k.id,
            Value::from(HostTensor::f32(vec![1, hkv, l, dk], kd.clone())),
        );
        inp.insert(
            v.id,
            Value::from(HostTensor::f32(vec![1, hkv, l, dv], vd.clone())),
        );
        inp.insert(
            cg.id,
            Value::from(HostTensor::f32(vec![1, hq, l, dk], a_cum)),
        );
        inp.insert(m.id, Value::from(HostTensor::f32(vec![1, 1, l, l], causal)));
        let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("linear_attention_hybrid tests evaluate dense graphs");

        let want = gated_linattn_recurrence(&qd, &kd, &vd, &gd, hq, n_rep, l, dk, dv);
        assert_close_rel(got.as_f32().unwrap(), &want, 5e-3);
    }
}

#[test]
fn hybrid_model_decode_loop_matches_prefill() {
    // card 038: 2-layer hybrid model (layer 0 linear attention with matrix state S, layer 1 softmax attention with KV cache)
    // end to end. The mixed-state decode loop (carrying S + K/V caches) must equal the prefill of the same model over the
    // whole sequence, using existing ops and synthetic weights. Hq=2, Hkv=1, D=4, H=8, cap=L=4, decay=0.8, scale=0.5.
    let (l, hq, hkv, d) = (4usize, 2usize, 1usize, 4usize);
    let (h, qdim, kvd) = (hq * d, hq * d, hkv * d);
    let n_rep = hq / hkv;
    let (decay, scale, eps) = (0.8f32, 1.0 / (d as f32).sqrt(), 1e-6f32);
    let xd = fill(l * h, 2);
    // weights: per layer {wn, wq, wk, wv, wo}; seeds offset by layer.
    let w = |seed: u64, n: usize| fill(n, seed);
    let lw = |ly: u64| {
        (
            w(20 + ly, h),             // wn
            w(21 + ly * 10, h * qdim), // wq
            w(22 + ly * 10, h * kvd),  // wk
            w(23 + ly * 10, h * kvd),  // wv
            w(24 + ly * 10, qdim * h), // wo
        )
    };
    let (l0, l1) = (lw(0), lw(1));

    let to_heads = |b: &Builder, t: Traced, heads: usize, tt: usize| -> Traced {
        b.transpose(b.reshape(t, vec![1, tt, heads, d]), vec![0, 2, 1, 3])
    };
    let from_heads = |b: &Builder, o: Traced, tt: usize| -> Traced {
        b.reshape(b.transpose(o, vec![0, 2, 1, 3]), vec![1, tt, qdim])
    };
    // constants for one layer's weights; returns the 5 ids in order.
    let mk_w = |b: &Builder, p: &str| -> [Traced; 5] {
        [
            b.constant(&format!("{p}n"), TensorType::f32(vec![h])),
            b.constant(&format!("{p}q"), TensorType::f32(vec![h, qdim])),
            b.constant(&format!("{p}k"), TensorType::f32(vec![h, kvd])),
            b.constant(&format!("{p}v"), TensorType::f32(vec![h, kvd])),
            b.constant(&format!("{p}o"), TensorType::f32(vec![qdim, h])),
        ]
    };
    type LayerW = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    let bind_w = |inp: &mut HashMap<usize, Value>, ids: &[Traced; 5], wts: &LayerW| {
        inp.insert(
            ids[0].id,
            Value::from(HostTensor::f32(vec![h], wts.0.clone())),
        );
        inp.insert(
            ids[1].id,
            Value::from(HostTensor::f32(vec![h, qdim], wts.1.clone())),
        );
        inp.insert(
            ids[2].id,
            Value::from(HostTensor::f32(vec![h, kvd], wts.2.clone())),
        );
        inp.insert(
            ids[3].id,
            Value::from(HostTensor::f32(vec![h, kvd], wts.3.clone())),
        );
        inp.insert(
            ids[4].id,
            Value::from(HostTensor::f32(vec![qdim, h], wts.4.clone())),
        );
    };

    // --- prefill: x[1,L,H] -> linear layer -> softmax layer -> [1,L,H] ---
    let bp = Builder::new();
    let xp = bp.constant("x", TensorType::f32(vec![1, l, h]));
    let w0 = mk_w(&bp, "a"); // layer 0 (linear)
    let w1 = mk_w(&bp, "b"); // layer 1 (softmax)
    let mut dm = vec![0.0f32; l * l]; // linear decay-causal mask
    let mut cm = vec![0.0f32; l * l]; // softmax additive causal mask
    for t in 0..l {
        for j in 0..l {
            if j <= t {
                dm[t * l + j] = decay.powi((t - j) as i32);
            } else {
                cm[t * l + j] = -1.0e9;
            }
        }
    }
    let cmd = bp.constant("dm", TensorType::f32(vec![1, 1, l, l]));
    let cmc = bp.constant("cm", TensorType::f32(vec![1, 1, l, l]));
    // layer 0 (linear)
    let n0 = rmsnorm(&bp, xp, w0[0], eps);
    let q0 = to_heads(&bp, linear(&bp, n0, w0[1], None), hq, l);
    let k0 = to_heads(&bp, linear(&bp, n0, w0[2], None), hkv, l);
    let v0 = to_heads(&bp, linear(&bp, n0, w0[3], None), hkv, l);
    let o0 = linear_attention_prefill(&bp, q0, k0, v0, n_rep, cmd);
    let x1 = bp.binary(
        BinOp::Add,
        xp,
        linear(&bp, from_heads(&bp, o0, l), w0[4], None),
    );
    // layer 1 (softmax)
    let n1 = rmsnorm(&bp, x1, w1[0], eps);
    let q1 = to_heads(&bp, linear(&bp, n1, w1[1], None), hq, l);
    let k1 = to_heads(&bp, linear(&bp, n1, w1[2], None), hkv, l);
    let v1 = to_heads(&bp, linear(&bp, n1, w1[3], None), hkv, l);
    let a1 = attention_prefill(&bp, q1, k1, v1, n_rep, scale, cmc);
    let x2 = bp.binary(
        BinOp::Add,
        x1,
        linear(&bp, from_heads(&bp, a1, l), w1[4], None),
    );
    let gp = bp.finish(x2);
    let mut pin = HashMap::new();
    pin.insert(
        xp.id,
        Value::from(HostTensor::f32(vec![1, l, h], xd.clone())),
    );
    bind_w(&mut pin, &w0, &l0);
    bind_w(&mut pin, &w1, &l1);
    pin.insert(cmd.id, Value::from(HostTensor::f32(vec![1, 1, l, l], dm)));
    pin.insert(cmc.id, Value::from(HostTensor::f32(vec![1, 1, l, l], cm)));
    let want = eval(&gp, &pin, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention_hybrid tests evaluate dense graphs");

    // --- decode loop: rebuild the pos-specialized graph each step, carry [S, Kcache, Vcache] ---
    let mut s_mat = HostTensor::f32(vec![1, hq, d, d], vec![0.0f32; hq * d * d]);
    let mut kc = HostTensor::f32(vec![1, hkv, l, d], vec![0.0f32; hkv * l * d]);
    let mut vc = HostTensor::f32(vec![1, hkv, l, d], vec![0.0f32; hkv * l * d]);
    let mut got = vec![0.0f32; l * h];
    for pos in 0..l {
        let b = Builder::new();
        let xt = b.constant("x", TensorType::f32(vec![1, 1, h]));
        let w0d = mk_w(&b, "a");
        let w1d = mk_w(&b, "b");
        let s_in = b.state_input(
            "s",
            TensorType::f32(vec![1, hq, d, d]),
            StateRole::Recurrent,
        );
        let kc_in = b.state_input(
            "kc",
            TensorType::f32(vec![1, hkv, l, d]),
            StateRole::Recurrent,
        );
        let vc_in = b.state_input(
            "vc",
            TensorType::f32(vec![1, hkv, l, d]),
            StateRole::Recurrent,
        );
        // layer 0 (linear)
        let n0 = rmsnorm(&b, xt, w0d[0], eps);
        let q0 = to_heads(&b, linear(&b, n0, w0d[1], None), hq, 1);
        let k0 = to_heads(&b, linear(&b, n0, w0d[2], None), hkv, 1);
        let v0 = to_heads(&b, linear(&b, n0, w0d[3], None), hkv, 1);
        let (o0, s_out) = linear_attention_decode(&b, q0, k0, v0, n_rep, decay, s_in);
        let x1 = b.binary(
            BinOp::Add,
            xt,
            linear(&b, from_heads(&b, o0, 1), w0d[4], None),
        );
        // layer 1 (softmax over the KV cache up to pos)
        let n1 = rmsnorm(&b, x1, w1d[0], eps);
        let q1 = to_heads(&b, linear(&b, n1, w1d[1], None), hq, 1);
        let k1 = to_heads(&b, linear(&b, n1, w1d[2], None), hkv, 1);
        let v1 = to_heads(&b, linear(&b, n1, w1d[3], None), hkv, 1);
        let kc_out = b.dynamic_update_slice(kc_in, k1, pos, 2);
        let vc_out = b.dynamic_update_slice(vc_in, v1, pos, 2);
        let kv = b.slice(kc_out, 2, 0, pos + 1);
        let vv = b.slice(vc_out, 2, 0, pos + 1);
        let a1 = attention(&b, q1, kv, vv, n_rep, scale);
        let x2 = b.binary(
            BinOp::Add,
            x1,
            linear(&b, from_heads(&b, a1, 1), w1d[4], None),
        );
        let g = b.finish_with_state(x2, &[(s_in, s_out), (kc_in, kc_out), (vc_in, vc_out)]);

        let mut inp = HashMap::new();
        inp.insert(
            xt.id,
            Value::from(HostTensor::f32(
                vec![1, 1, h],
                xd[pos * h..pos * h + h].to_vec(),
            )),
        );
        bind_w(&mut inp, &w0d, &l0);
        bind_w(&mut inp, &w1d, &l1);
        inp.insert(s_in.id, Value::from(s_mat.clone()));
        inp.insert(kc_in.id, Value::from(kc.clone()));
        inp.insert(vc_in.id, Value::from(vc.clone()));
        let (o_t, new) = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| {
                        v.into_host()
                            .expect("linear_attention_hybrid state is dense")
                    })
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention_hybrid tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();
        // states come back in g.state order: [S, Kcache, Vcache].
        s_mat = new[0].clone();
        kc = new[1].clone();
        vc = new[2].clone();
        got[pos * h..pos * h + h].copy_from_slice(o_t.as_f32().unwrap());
    }
    assert_close_rel(&got, want.as_f32().unwrap(), 1e-4);
}

#[test]
fn decay_mask_from_gates_matches_reference() {
    // card 038: build the gated-linear-attention decay-causal mask in the graph from per-step gates,
    // mask[h,t,j] = prod_{j<k<=t} g[h,k] for j<=t (cumsum of log gates -> exp). (a) constant gate g -> the geometric
    // g^(t-j) tril mask; (b) data-dependent per-step gate -> a from-scratch cumprod reference. H=2, L=4.
    use poot_graph_ir::ops::decay_mask_from_gates;
    let (hh, l) = (2usize, 4usize);
    // lower-triangular ones [1,H,L,L].
    let mut tril = vec![0.0f32; hh * l * l];
    for h in 0..hh {
        for t in 0..l {
            for j in 0..=t {
                tril[(h * l + t) * l + j] = 1.0;
            }
        }
    }
    let run = |gv: &[f32]| -> HostTensor {
        let b = Builder::new();
        let gates = b.constant("g", TensorType::f32(vec![1, hh, 1, l]));
        let tr = b.constant("tr", TensorType::f32(vec![1, hh, l, l]));
        let m = decay_mask_from_gates(&b, gates, tr, hh, l);
        let g = b.finish(m);
        let mut inp = HashMap::new();
        inp.insert(
            gates.id,
            Value::from(HostTensor::f32(vec![1, hh, 1, l], gv.to_vec())),
        );
        inp.insert(
            tr.id,
            Value::from(HostTensor::f32(vec![1, hh, l, l], tril.clone())),
        );
        eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("linear_attention_hybrid tests evaluate dense graphs")
    };

    // (a) constant gate per head -> g^(t-j).
    let gc = [0.8f32, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5]; // head 0 = 0.8, head 1 = 0.5
    let got_c = run(&gc);
    let mut want_c = vec![0.0f32; hh * l * l];
    let gconst = [0.8f32, 0.5];
    for h in 0..hh {
        for t in 0..l {
            for j in 0..=t {
                want_c[(h * l + t) * l + j] = gconst[h].powi((t - j) as i32);
            }
        }
    }
    assert_close_rel(got_c.as_f32().unwrap(), &want_c, 1e-5);

    // (b) data-dependent per-step gate -> hand cumprod. g[h,k] in (0,1).
    let gd: Vec<f32> = (0..hh * l).map(|i| 0.4 + 0.1 * ((i % 5) as f32)).collect();
    let got_d = run(&gd);
    let mut want_d = vec![0.0f32; hh * l * l];
    for h in 0..hh {
        for t in 0..l {
            for j in 0..=t {
                let mut p = 1.0f32;
                for k in (j + 1)..=t {
                    p *= gd[h * l + k];
                }
                want_d[(h * l + t) * l + j] = p;
            }
        }
    }
    assert_close_rel(got_d.as_f32().unwrap(), &want_d, 1e-5);
}

#[test]
fn gated_linear_attention_prefill_with_in_graph_mask_matches_reference() {
    // card 038: feed the in-graph decay mask (decay_mask_from_gates) into `linear_attention_prefill` and check against a
    // from-scratch gated linear-attention reference with data-dependent per-step gates. Hq=2, Hkv=1, D=3, L=4.
    use poot_graph_ir::ops::{decay_mask_from_gates, linear_attention_prefill};
    let (l, hq, hkv, d) = (4usize, 2usize, 1usize, 3usize);
    let n_rep = hq / hkv;
    let qd = fill(hq * l * d, 60);
    let kd = fill(hkv * l * d, 61);
    let vd = fill(hkv * l * d, 62);
    // per-step gates per query head, in (0,1).
    let gd: Vec<f32> = (0..hq * l).map(|i| 0.5 + 0.08 * ((i % 4) as f32)).collect();
    let mut tril = vec![0.0f32; hq * l * l];
    for h in 0..hq {
        for t in 0..l {
            for j in 0..=t {
                tril[(h * l + t) * l + j] = 1.0;
            }
        }
    }

    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
    let gates = b.constant("g", TensorType::f32(vec![1, hq, 1, l]));
    let tr = b.constant("tr", TensorType::f32(vec![1, hq, l, l]));
    let mask = decay_mask_from_gates(&b, gates, tr, hq, l);
    let o = linear_attention_prefill(&b, q, k, v, n_rep, mask);
    let g = b.finish(o);
    let mut inp = HashMap::new();
    inp.insert(
        q.id,
        Value::from(HostTensor::f32(vec![1, hq, l, d], qd.clone())),
    );
    inp.insert(
        k.id,
        Value::from(HostTensor::f32(vec![1, hkv, l, d], kd.clone())),
    );
    inp.insert(
        v.id,
        Value::from(HostTensor::f32(vec![1, hkv, l, d], vd.clone())),
    );
    inp.insert(
        gates.id,
        Value::from(HostTensor::f32(vec![1, hq, 1, l], gd.clone())),
    );
    inp.insert(tr.id, Value::from(HostTensor::f32(vec![1, hq, l, l], tril)));
    let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention_hybrid tests evaluate dense graphs");

    // reference: o[h,t] = sum_{j<=t} decay[h,t,j] * (q[h,t] . k[kv,j]) * v[kv,j], decay = prod_{j<k<=t} g.
    let mut want = vec![0.0f32; hq * l * d];
    for h in 0..hq {
        let kv = h / n_rep;
        for t in 0..l {
            for j in 0..=t {
                let mut dot = 0.0f32;
                for e in 0..d {
                    dot += qd[(h * l + t) * d + e] * kd[(kv * l + j) * d + e];
                }
                let mut decay = 1.0f32;
                for kk in (j + 1)..=t {
                    decay *= gd[h * l + kk];
                }
                for e in 0..d {
                    want[(h * l + t) * d + e] += decay * dot * vd[(kv * l + j) * d + e];
                }
            }
        }
    }
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
}

#[test]
fn mamba2_block_decode_loop_matches_reference() {
    // card 038: a full Mamba2 mixer block: in_proj (separate projections for z, x_conv, B, C, dt) -> depthwise causal
    // conv1d -> SiLU -> Delta=softplus(dt) -> SSD -> output gate y*silu(z) -> out_proj -> residual. The decode loop,
    // carrying the conv ring buffer and SSM state, must equal a from-scratch Mamba2 reference. H=8, Hq=2, P=4 (Hq*P=H),
    // N=3 state, K=4 conv. (Separate projections are equivalent to the real fused in_proj-then-split.)
    use poot_graph_ir::ops::{causal_conv1d_decode, mamba2_ssd_decode, softplus};
    let (l, hh, hq, pdim, n, kk) = (4usize, 8usize, 2usize, 4usize, 3usize, 4usize);
    let ch = hq * pdim; // conv channels = inner dim = 8
    let eps_silu = |x: f32| x / (1.0 + (-x).exp());
    // weights (deterministic).
    let w_z = fill(hh * ch, 30);
    let w_xc = fill(hh * ch, 31);
    let w_b = fill(hh * hq * n, 32);
    let w_c = fill(hh * hq * n, 33);
    let w_dt = fill(hh * hq, 34);
    let dt_bias = fill(hq, 35);
    let w_conv = fill(kk * ch, 36);
    let w_out = fill(ch * hh, 37);
    let a_par: Vec<f32> = (0..hq).map(|h| -0.5 - 0.3 * h as f32).collect(); // A < 0 -> Abar in (0,1)
    let d_par: Vec<f32> = (0..hq).map(|h| 0.2 + 0.1 * h as f32).collect();
    let xd = fill(l * hh, 2); // hidden states [L, H]

    // decode graph (one step), carrying conv cache + SSM state.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1, hh]));
    let cz = b.constant("wz", TensorType::f32(vec![hh, ch]));
    let cxc = b.constant("wxc", TensorType::f32(vec![hh, ch]));
    let cb = b.constant("wb", TensorType::f32(vec![hh, hq * n]));
    let cc = b.constant("wc", TensorType::f32(vec![hh, hq * n]));
    let cdt = b.constant("wdt", TensorType::f32(vec![hh, hq]));
    let cdtb = b.constant("dtb", TensorType::f32(vec![hq]));
    let cwconv = b.constant("wconv", TensorType::f32(vec![kk, ch]));
    let ca = b.constant("a", TensorType::f32(vec![1, hq, 1, 1]));
    let cd = b.constant("d", TensorType::f32(vec![1, hq, 1, 1]));
    let cwout = b.constant("wout", TensorType::f32(vec![ch, hh]));
    let conv_cache = b.state_input(
        "cv",
        TensorType::f32(vec![1, kk - 1, ch]),
        StateRole::Recurrent,
    );
    let h_in = b.state_input(
        "h",
        TensorType::f32(vec![1, hq, n, pdim]),
        StateRole::Recurrent,
    );
    // in_proj.
    let z = linear(&b, x, cz, None);
    let xc = linear(&b, x, cxc, None);
    let bproj = linear(&b, x, cb, None);
    let cproj = linear(&b, x, cc, None);
    let dt_raw = linear(&b, x, cdt, Some(cdtb));
    // conv -> silu -> heads.
    let (conv_out, conv_cache_out) = causal_conv1d_decode(&b, xc, cwconv, conv_cache, kk);
    let xact = poot_graph_ir::ops::silu(&b, conv_out);
    let xheads = b.reshape(xact, vec![1, hq, 1, pdim]);
    let delta = softplus(&b, b.reshape(dt_raw, vec![1, hq, 1, 1]));
    let bheads = b.reshape(bproj, vec![1, hq, 1, n]);
    let cheads = b.reshape(cproj, vec![1, hq, 1, n]);
    let (y, h_out) = mamba2_ssd_decode(&b, xheads, bheads, cheads, delta, ca, cd, h_in);
    // output gate y * silu(z), out_proj, residual.
    let zheads = b.reshape(z, vec![1, hq, 1, pdim]);
    let ygated = b.binary(BinOp::Mul, y, poot_graph_ir::ops::silu(&b, zheads));
    let yflat = b.reshape(ygated, vec![1, 1, ch]);
    let out = linear(&b, yflat, cwout, None);
    let result = b.binary(BinOp::Add, x, out);
    let g = b.finish_with_state(result, &[(conv_cache, conv_cache_out), (h_in, h_out)]);

    let bind_w = |inp: &mut HashMap<usize, Value>| {
        inp.insert(
            cz.id,
            Value::from(HostTensor::f32(vec![hh, ch], w_z.clone())),
        );
        inp.insert(
            cxc.id,
            Value::from(HostTensor::f32(vec![hh, ch], w_xc.clone())),
        );
        inp.insert(
            cb.id,
            Value::from(HostTensor::f32(vec![hh, hq * n], w_b.clone())),
        );
        inp.insert(
            cc.id,
            Value::from(HostTensor::f32(vec![hh, hq * n], w_c.clone())),
        );
        inp.insert(
            cdt.id,
            Value::from(HostTensor::f32(vec![hh, hq], w_dt.clone())),
        );
        inp.insert(
            cdtb.id,
            Value::from(HostTensor::f32(vec![hq], dt_bias.clone())),
        );
        inp.insert(
            cwconv.id,
            Value::from(HostTensor::f32(vec![kk, ch], w_conv.clone())),
        );
        inp.insert(
            ca.id,
            Value::from(HostTensor::f32(vec![1, hq, 1, 1], a_par.clone())),
        );
        inp.insert(
            cd.id,
            Value::from(HostTensor::f32(vec![1, hq, 1, 1], d_par.clone())),
        );
        inp.insert(
            cwout.id,
            Value::from(HostTensor::f32(vec![ch, hh], w_out.clone())),
        );
    };

    let mut cv_cache = HostTensor::f32(vec![1, kk - 1, ch], vec![0.0f32; (kk - 1) * ch]);
    let mut h_state = HostTensor::f32(vec![1, hq, n, pdim], vec![0.0f32; hq * n * pdim]);
    let mut got = vec![0.0f32; l * hh];
    for t in 0..l {
        let mut inp = HashMap::new();
        inp.insert(
            x.id,
            Value::from(HostTensor::f32(
                vec![1, 1, hh],
                xd[t * hh..t * hh + hh].to_vec(),
            )),
        );
        bind_w(&mut inp);
        inp.insert(conv_cache.id, Value::from(cv_cache.clone()));
        inp.insert(h_in.id, Value::from(h_state.clone()));
        let (yt, new) = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| {
                        v.into_host()
                            .expect("linear_attention_hybrid state is dense")
                    })
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention_hybrid tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();
        cv_cache = new[0].clone();
        h_state = new[1].clone();
        got[t * hh..t * hh + hh].copy_from_slice(yt.as_f32().unwrap());
    }

    // hand reference: the whole Mamba2 block, carrying xc history (for conv) + SSM state.
    let mm = |x: &[f32], w: &[f32], inn: usize, out: usize| -> Vec<f32> {
        (0..out)
            .map(|j| (0..inn).map(|i| x[i] * w[i * out + j]).sum())
            .collect()
    };
    let mut want = vec![0.0f32; l * hh];
    let mut xc_hist: Vec<Vec<f32>> = Vec::new(); // past xc vectors [ch]
    let mut state = vec![0.0f32; hq * n * pdim];
    for t in 0..l {
        let xt = &xd[t * hh..t * hh + hh];
        let z = mm(xt, &w_z, hh, ch);
        let xc = mm(xt, &w_xc, hh, ch);
        let bproj = mm(xt, &w_b, hh, hq * n);
        let cproj = mm(xt, &w_c, hh, hq * n);
        let mut dt_raw = mm(xt, &w_dt, hh, hq);
        for h in 0..hq {
            dt_raw[h] += dt_bias[h];
        }
        xc_hist.push(xc.clone());
        // conv: window = last K of xc history (zero-padded), then silu.
        let mut xc_act = vec![0.0f32; ch];
        for (c, slot) in xc_act.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..kk {
                let idx = t as isize - (kk as isize - 1) + k as isize;
                if idx >= 0 {
                    acc += w_conv[k * ch + c] * xc_hist[idx as usize][c];
                }
            }
            *slot = eps_silu(acc);
        }
        // SSD per head with Delta = softplus.
        let mut y = vec![0.0f32; ch];
        for h in 0..hq {
            let delta_h = (1.0 + dt_raw[h].exp()).ln();
            let a_bar = (delta_h * a_par[h]).exp();
            for nn in 0..n {
                let b_bar = delta_h * bproj[h * n + nn];
                for p in 0..pdim {
                    let xv = xc_act[h * pdim + p];
                    state[(h * n + nn) * pdim + p] =
                        a_bar * state[(h * n + nn) * pdim + p] + b_bar * xv;
                }
            }
            for p in 0..pdim {
                let mut acc = 0.0f32;
                for nn in 0..n {
                    acc += cproj[h * n + nn] * state[(h * n + nn) * pdim + p];
                }
                let yv = acc + d_par[h] * xc_act[h * pdim + p];
                y[h * pdim + p] = yv * eps_silu(z[h * pdim + p]); // output gate
            }
        }
        let out = mm(&y, &w_out, ch, hh);
        for j in 0..hh {
            want[t * hh + j] = xt[j] + out[j]; // residual
        }
    }
    assert_close_rel(&got, &want, 1e-4);
}

#[test]
fn linear_attention_layer_decode_loop_matches_prefill() {
    // card 038: a full linear-attention sublayer (rmsnorm -> q/k/v projections -> linear_attention -> out projection ->
    // residual), with q/k/v derived from x. The decode loop (carrying matrix state S) must equal the prefill of the same
    // sublayer over the whole sequence. H = Hq*D = 8, Hkv=1, D=4, decay=0.8, L=4.
    let (l, hq, hkv, d) = (4usize, 2usize, 1usize, 4usize);
    let (h, qd_dim, kvd) = (hq * d, hq * d, hkv * d);
    let n_rep = hq / hkv;
    let decay = 0.8f32;
    let eps = 1e-6f32;
    let xd = fill(l * h, 2); // hidden states [L, H]
    let wn = fill(h, 11);
    let wq = fill(h * qd_dim, 12);
    let wk = fill(h * kvd, 13);
    let wv = fill(h * kvd, 14);
    let wo = fill(qd_dim * h, 15);

    // build the q/k/v -> [1,Hq|Hkv, T, D] heads from a normed [1,T,H].
    let to_heads = |b: &Builder, t: Traced, heads: usize, tt: usize| -> Traced {
        let r = b.reshape(t, vec![1, tt, heads, d]);
        b.transpose(r, vec![0, 2, 1, 3]) // [1, heads, T, D]
    };
    let from_heads = |b: &Builder, o: Traced, tt: usize| -> Traced {
        let r = b.transpose(o, vec![0, 2, 1, 3]); // [1, T, Hq, D]
        b.reshape(r, vec![1, tt, qd_dim])
    };

    // --- prefill reference: the sublayer over the whole [1,L,H] sequence ---
    let mut md = vec![0.0f32; l * l];
    for t in 0..l {
        for j in 0..=t {
            md[t * l + j] = decay.powi((t - j) as i32);
        }
    }
    let bp = Builder::new();
    let xp = bp.constant("x", TensorType::f32(vec![1, l, h]));
    let cwn = bp.constant("wn", TensorType::f32(vec![h]));
    let cwq = bp.constant("wq", TensorType::f32(vec![h, qd_dim]));
    let cwk = bp.constant("wk", TensorType::f32(vec![h, kvd]));
    let cwv = bp.constant("wv", TensorType::f32(vec![h, kvd]));
    let cwo = bp.constant("wo", TensorType::f32(vec![qd_dim, h]));
    let cmask = bp.constant("m", TensorType::f32(vec![1, 1, l, l]));
    let normed = rmsnorm(&bp, xp, cwn, eps);
    let q = to_heads(&bp, linear(&bp, normed, cwq, None), hq, l);
    let k = to_heads(&bp, linear(&bp, normed, cwk, None), hkv, l);
    let v = to_heads(&bp, linear(&bp, normed, cwv, None), hkv, l);
    let o = linear_attention_prefill(&bp, q, k, v, n_rep, cmask);
    let attn = linear(&bp, from_heads(&bp, o, l), cwo, None);
    let xout_p = bp.binary(BinOp::Add, xp, attn);
    let gp = bp.finish(xout_p);
    let bindw = |inp: &mut HashMap<usize, Value>, ids: [usize; 5]| {
        inp.insert(ids[0], Value::from(HostTensor::f32(vec![h], wn.clone())));
        inp.insert(
            ids[1],
            Value::from(HostTensor::f32(vec![h, qd_dim], wq.clone())),
        );
        inp.insert(
            ids[2],
            Value::from(HostTensor::f32(vec![h, kvd], wk.clone())),
        );
        inp.insert(
            ids[3],
            Value::from(HostTensor::f32(vec![h, kvd], wv.clone())),
        );
        inp.insert(
            ids[4],
            Value::from(HostTensor::f32(vec![qd_dim, h], wo.clone())),
        );
    };
    let mut pin = HashMap::new();
    pin.insert(
        xp.id,
        Value::from(HostTensor::f32(vec![1, l, h], xd.clone())),
    );
    bindw(&mut pin, [cwn.id, cwq.id, cwk.id, cwv.id, cwo.id]);
    pin.insert(cmask.id, Value::from(HostTensor::f32(vec![1, 1, l, l], md)));
    let want = eval(&gp, &pin, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention_hybrid tests evaluate dense graphs"); // [1,L,H]

    // --- decode loop: the same sublayer one token at a time, carrying S ---
    let bd = Builder::new();
    let xt = bd.constant("x", TensorType::f32(vec![1, 1, h]));
    let dwn = bd.constant("wn", TensorType::f32(vec![h]));
    let dwq = bd.constant("wq", TensorType::f32(vec![h, qd_dim]));
    let dwk = bd.constant("wk", TensorType::f32(vec![h, kvd]));
    let dwv = bd.constant("wv", TensorType::f32(vec![h, kvd]));
    let dwo = bd.constant("wo", TensorType::f32(vec![qd_dim, h]));
    let s_in = bd.state_input(
        "s",
        TensorType::f32(vec![1, hq, d, d]),
        StateRole::Recurrent,
    );
    let nd = rmsnorm(&bd, xt, dwn, eps);
    let qd_ = to_heads(&bd, linear(&bd, nd, dwq, None), hq, 1);
    let kd_ = to_heads(&bd, linear(&bd, nd, dwk, None), hkv, 1);
    let vd_ = to_heads(&bd, linear(&bd, nd, dwv, None), hkv, 1);
    let (od, s_out) = linear_attention_decode(&bd, qd_, kd_, vd_, n_rep, decay, s_in);
    let attnd = linear(&bd, from_heads(&bd, od, 1), dwo, None);
    let xout_d = bd.binary(BinOp::Add, xt, attnd);
    let gd = bd.finish_with_state(xout_d, &[(s_in, s_out)]);

    let mut caches = vec![HostTensor::f32(vec![1, hq, d, d], vec![0.0f32; hq * d * d])];
    let mut got = vec![0.0f32; l * h];
    for t in 0..l {
        let mut inp = HashMap::new();
        inp.insert(
            xt.id,
            Value::from(HostTensor::f32(
                vec![1, 1, h],
                xd[t * h..t * h + h].to_vec(),
            )),
        );
        bindw(&mut inp, [dwn.id, dwq.id, dwk.id, dwv.id, dwo.id]);
        for (ci, &(s, _)) in gd.state.iter().enumerate() {
            inp.insert(s, Value::from(caches[ci].clone()));
        }
        let (o_t, new) = eval(&gd, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| {
                        v.into_host()
                            .expect("linear_attention_hybrid state is dense")
                    })
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention_hybrid tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();
        caches = new;
        got[t * h..t * h + h].copy_from_slice(o_t.as_f32().unwrap());
    }
    assert_close_rel(&got, want.as_f32().unwrap(), 1e-4);
}

#[test]
fn linear_attention_decode_loop_matches_prefill() {
    // card 038: the recurrent state S carried step by step through the State mechanism (`finish_with_state` /
    // `eval_with_state`, the captured-graph-replay path) must give the same per-token outputs as the parallel prefill.
    // Exercises State with a matrix state [1,Hq,D_k,D_v]. L=5, GQA Hq=2/Hkv=1, D_k=3, D_v=4, decay=0.8.
    let (l, hq, hkv, dk, dv) = (5usize, 2usize, 1usize, 3usize, 4usize);
    let n_rep = hq / hkv;
    let decay = 0.8f32;
    let qd = fill(hq * l * dk, 3); // [Hq, L, D_k]
    let kd = fill(hkv * l * dk, 4); // [Hkv, L, D_k]
    let vd = fill(hkv * l * dv, 5); // [Hkv, L, D_v]

    // one-token decode graph, state S carried (s_in -> s_out).
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, dk]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, 1, dk]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, 1, dv]));
    let s_in = b.state_input(
        "s",
        TensorType::f32(vec![1, hq, dk, dv]),
        StateRole::Recurrent,
    );
    let (o, s_out) = linear_attention_decode(&b, q, k, v, n_rep, decay, s_in);
    let (qi, ki, vi) = (q.id, k.id, v.id);
    let gd = b.finish_with_state(o, &[(s_in, s_out)]);

    // run the decode loop, carrying S; collect outputs into the prefill's [Hq, L, D_v] layout.
    let mut caches = vec![HostTensor::f32(
        vec![1, hq, dk, dv],
        vec![0.0f32; hq * dk * dv],
    )];
    let mut got = vec![0.0f32; hq * l * dv];
    for t in 0..l {
        let qt: Vec<f32> = (0..hq)
            .flat_map(|h| qd[(h * l + t) * dk..(h * l + t) * dk + dk].to_vec())
            .collect();
        let kt: Vec<f32> = (0..hkv)
            .flat_map(|h| kd[(h * l + t) * dk..(h * l + t) * dk + dk].to_vec())
            .collect();
        let vt: Vec<f32> = (0..hkv)
            .flat_map(|h| vd[(h * l + t) * dv..(h * l + t) * dv + dv].to_vec())
            .collect();
        let mut inputs = HashMap::new();
        inputs.insert(qi, Value::from(HostTensor::f32(vec![1, hq, 1, dk], qt)));
        inputs.insert(ki, Value::from(HostTensor::f32(vec![1, hkv, 1, dk], kt)));
        inputs.insert(vi, Value::from(HostTensor::f32(vec![1, hkv, 1, dv], vt)));
        for (ci, &(s, _)) in gd.state.iter().enumerate() {
            inputs.insert(s, Value::from(caches[ci].clone()));
        }
        let (o_t, new) = eval(&gd, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| {
                        v.into_host()
                            .expect("linear_attention_hybrid state is dense")
                    })
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention_hybrid tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();
        caches = new;
        for h in 0..hq {
            for c in 0..dv {
                got[(h * l + t) * dv + c] = o_t.as_f32().unwrap()[h * dv + c];
            }
        }
    }

    // prefill reference on the full sequence (the parallel quadratic form, separately verified).
    let mut md = vec![0.0f32; l * l];
    for t in 0..l {
        for j in 0..=t {
            md[t * l + j] = decay.powi((t - j) as i32);
        }
    }
    let bp = Builder::new();
    let pq = bp.constant("q", TensorType::f32(vec![1, hq, l, dk]));
    let pk = bp.constant("k", TensorType::f32(vec![1, hkv, l, dk]));
    let pv = bp.constant("v", TensorType::f32(vec![1, hkv, l, dv]));
    let pm = bp.constant("m", TensorType::f32(vec![1, 1, l, l]));
    let po = linear_attention_prefill(&bp, pq, pk, pv, n_rep, pm);
    let (pqi, pki, pvi, pmi) = (pq.id, pk.id, pv.id, pm.id);
    let gp = bp.finish(po);
    let mut pin = HashMap::new();
    pin.insert(pqi, Value::from(HostTensor::f32(vec![1, hq, l, dk], qd)));
    pin.insert(pki, Value::from(HostTensor::f32(vec![1, hkv, l, dk], kd)));
    pin.insert(pvi, Value::from(HostTensor::f32(vec![1, hkv, l, dv], vd)));
    pin.insert(pmi, Value::from(HostTensor::f32(vec![1, 1, l, l], md)));
    let want = eval(&gp, &pin, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention_hybrid tests evaluate dense graphs");
    assert_close_rel(&got, want.as_f32().unwrap(), 1e-4);
}
