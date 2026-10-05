//! Attention and linear-attention decode/prefill, UT transform inverse, GDN chunked prefill, causal conv1d.

use crate::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::builder::{Builder, Traced};
use poot_graph_ir::graph::StateRole;
use poot_graph_ir::ops::{
    attention, causal_conv1d_decode, causal_conv1d_prefill, gated_delta_net_decode,
    gdn_prefill_chunked, linear_attention_decode, linear_attention_decode_gated,
    linear_attention_prefill, linear_attention_prefill_chunked, linear_attention_prefill_gated,
    unit_lower_triangular_inverse,
};
use poot_graph_ir::types::TensorType;
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::{assert_close_rel, max_abs_error, max_abs_error_f64};

#[test]
fn attention_matches_direct() {
    // decode attention: 1 query token, Hq=2, Hkv=1 (n_rep=2), S=4, D=4.
    let b = Builder::new();
    let (hq, hkv, s, d) = (2usize, 1usize, 4usize, 4usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, s, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, s, d]));
    let out = attention(&b, q, k, v, n_rep, scale);
    let (qi, ki, vi) = (q.id, k.id, v.id);
    let g = b.finish(out);

    let qd = fill(hq * d, 8);
    let kd = fill(hkv * s * d, 9);
    let vd = fill(hkv * s * d, 10);
    let mut inputs = HashMap::new();
    inputs.insert(
        qi,
        Value::from(HostTensor::f32(vec![1, hq, 1, d], qd.clone())),
    );
    inputs.insert(
        ki,
        Value::from(HostTensor::f32(vec![1, hkv, s, d], kd.clone())),
    );
    inputs.insert(
        vi,
        Value::from(HostTensor::f32(vec![1, hkv, s, d], vd.clone())),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");

    // direct: for each query head h, kv head = h / n_rep.
    let mut want = vec![0.0f32; hq * d];
    for h in 0..hq {
        let kvh = h / n_rep;
        let qh = &qd[h * d..h * d + d];
        // scores over S keys.
        let mut scores = vec![0.0f32; s];
        for j in 0..s {
            let kj = &kd[(kvh * s + j) * d..(kvh * s + j) * d + d];
            scores[j] = qh.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|x| (x - m).exp()).collect();
        let denom: f32 = exps.iter().sum();
        for (j, p) in exps.iter().enumerate() {
            let p = p / denom;
            let vj = &vd[(kvh * s + j) * d..(kvh * s + j) * d + d];
            for c in 0..d {
                want[h * d + c] += p * vj[c];
            }
        }
    }
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}

#[test]
fn linear_attention_decode_matches_direct() {
    // card 038: one decode step of linear attention, S_t = decay*S + k^T v, o = q @ S_t. Hq=2, Hkv=1 (n_rep=2),
    // D_k=3, D_v=4 (distinct so a transpose shows), decay=0.9. Checks `o` and `s_out` against a hand recurrence.
    let (hq, hkv, dk, dv) = (2usize, 1usize, 3usize, 4usize);
    let n_rep = hq / hkv;
    let decay = 0.9f32;
    let qd = fill(hq * dk, 3);
    let kd = fill(hkv * dk, 4);
    let vd = fill(hkv * dv, 5);
    let sd = fill(hq * dk * dv, 6);

    // hand recurrence: per query head h, kv head = h / n_rep.
    let mut want_s = vec![0.0f32; hq * dk * dv];
    let mut want_o = vec![0.0f32; hq * dv];
    for h in 0..hq {
        let kvh = h / n_rep;
        for a in 0..dk {
            for c in 0..dv {
                let kv = kd[kvh * dk + a] * vd[kvh * dv + c]; // outer product k^T v
                want_s[(h * dk + a) * dv + c] = decay * sd[(h * dk + a) * dv + c] + kv;
            }
        }
        for c in 0..dv {
            let mut acc = 0.0f32;
            for a in 0..dk {
                acc += qd[h * dk + a] * want_s[(h * dk + a) * dv + c]; // q @ S_t
            }
            want_o[h * dv + c] = acc;
        }
    }

    // `finish` consumes the builder, so build the op once per output (`o`, then `s_out`).
    let eval_out = |pick_state: bool| -> HostTensor {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, dk]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, 1, dk]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, 1, dv]));
        let s = b.constant("s", TensorType::f32(vec![1, hq, dk, dv]));
        let (o, s_out) = linear_attention_decode(&b, q, k, v, n_rep, decay, s);
        let (qi, ki, vi, si) = (q.id, k.id, v.id, s.id);
        let g = b.finish(if pick_state { s_out } else { o });
        let mut inputs = HashMap::new();
        inputs.insert(
            qi,
            Value::from(HostTensor::f32(vec![1, hq, 1, dk], qd.clone())),
        );
        inputs.insert(
            ki,
            Value::from(HostTensor::f32(vec![1, hkv, 1, dk], kd.clone())),
        );
        inputs.insert(
            vi,
            Value::from(HostTensor::f32(vec![1, hkv, 1, dv], vd.clone())),
        );
        inputs.insert(
            si,
            Value::from(HostTensor::f32(vec![1, hq, dk, dv], sd.clone())),
        );
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("linear_attention tests evaluate dense graphs")
    };

    let got_o = eval_out(false);
    assert_eq!(got_o.shape(), vec![1, hq, 1, dv], "output is [1,Hq,1,D_v]");
    assert_close_rel(got_o.as_f32().unwrap(), &want_o, 1e-5);
    let got_s = eval_out(true);
    assert_eq!(
        got_s.shape(),
        vec![1, hq, dk, dv],
        "state is [1,Hq,D_k,D_v]"
    );
    assert_close_rel(got_s.as_f32().unwrap(), &want_s, 1e-5);
}

#[test]
fn linear_attention_decode_gated_matches_direct() {
    // card 038: gated variant with per-key-channel decay `gate[1,Hq,D_k,1]`:
    // S_t[h][a][c] = gate[h][a]*S[h][a][c] + k[a]*v[c].
    let (hq, hkv, dk, dv) = (2usize, 1usize, 3usize, 4usize);
    let n_rep = hq / hkv;
    let qd = fill(hq * dk, 3);
    let kd = fill(hkv * dk, 4);
    let vd = fill(hkv * dv, 5);
    let sd = fill(hq * dk * dv, 6);
    // per-(head, key-channel) gate in (0,1).
    let gd: Vec<f32> = (0..hq * dk)
        .map(|i| 0.5 + 0.4 * ((i * 7 % 5) as f32 / 5.0))
        .collect();

    let mut want_s = vec![0.0f32; hq * dk * dv];
    let mut want_o = vec![0.0f32; hq * dv];
    for h in 0..hq {
        let kvh = h / n_rep;
        for a in 0..dk {
            for c in 0..dv {
                let kv = kd[kvh * dk + a] * vd[kvh * dv + c];
                want_s[(h * dk + a) * dv + c] = gd[h * dk + a] * sd[(h * dk + a) * dv + c] + kv;
            }
        }
        for c in 0..dv {
            let mut acc = 0.0f32;
            for a in 0..dk {
                acc += qd[h * dk + a] * want_s[(h * dk + a) * dv + c];
            }
            want_o[h * dv + c] = acc;
        }
    }

    let eval_out = |pick_state: bool| -> HostTensor {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, dk]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, 1, dk]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, 1, dv]));
        let gate = b.constant("g", TensorType::f32(vec![1, hq, dk, 1]));
        let s = b.constant("s", TensorType::f32(vec![1, hq, dk, dv]));
        let (o, s_out) = linear_attention_decode_gated(&b, q, k, v, n_rep, gate, s);
        let (qi, ki, vi, gi, si) = (q.id, k.id, v.id, gate.id, s.id);
        let g = b.finish(if pick_state { s_out } else { o });
        let mut inputs = HashMap::new();
        inputs.insert(
            qi,
            Value::from(HostTensor::f32(vec![1, hq, 1, dk], qd.clone())),
        );
        inputs.insert(
            ki,
            Value::from(HostTensor::f32(vec![1, hkv, 1, dk], kd.clone())),
        );
        inputs.insert(
            vi,
            Value::from(HostTensor::f32(vec![1, hkv, 1, dv], vd.clone())),
        );
        inputs.insert(
            gi,
            Value::from(HostTensor::f32(vec![1, hq, dk, 1], gd.clone())),
        );
        inputs.insert(
            si,
            Value::from(HostTensor::f32(vec![1, hq, dk, dv], sd.clone())),
        );
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("linear_attention tests evaluate dense graphs")
    };

    let got_o = eval_out(false);
    assert_eq!(got_o.shape(), vec![1, hq, 1, dv]);
    assert_close_rel(got_o.as_f32().unwrap(), &want_o, 1e-5);
    let got_s = eval_out(true);
    assert_eq!(got_s.shape(), vec![1, hq, dk, dv]);
    assert_close_rel(got_s.as_f32().unwrap(), &want_s, 1e-5);
}

#[test]
fn linear_attention_prefill_matches_recurrence() {
    // card 038: the parallel quadratic prefill form o = (decay_mask (.) Q K^T) @ V must equal the sequential decode
    // recurrence run over the prompt. L=5, Hq=2, Hkv=1, D_k=3, D_v=4, decay=0.8.
    let (l, hq, hkv, dk, dv) = (5usize, 2usize, 1usize, 3usize, 4usize);
    let n_rep = hq / hkv;
    let decay = 0.8f32;
    let qd = fill(hq * l * dk, 3);
    let kd = fill(hkv * l * dk, 4);
    let vd = fill(hkv * l * dv, 5);
    // multiplicative decay-causal mask [1,1,L,L]: mask[t][j] = decay^(t-j) for j<=t, else 0.
    let mut md = vec![0.0f32; l * l];
    for t in 0..l {
        for j in 0..=t {
            md[t * l + j] = decay.powi((t - j) as i32);
        }
    }

    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, dk]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, dk]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, dv]));
    let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
    let o = linear_attention_prefill(&b, q, k, v, n_rep, mask);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(o);
    let mut inputs = HashMap::new();
    inputs.insert(
        qi,
        Value::from(HostTensor::f32(vec![1, hq, l, dk], qd.clone())),
    );
    inputs.insert(
        ki,
        Value::from(HostTensor::f32(vec![1, hkv, l, dk], kd.clone())),
    );
    inputs.insert(
        vi,
        Value::from(HostTensor::f32(vec![1, hkv, l, dv], vd.clone())),
    );
    inputs.insert(mi, Value::from(HostTensor::f32(vec![1, 1, l, l], md)));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![1, hq, l, dv]);

    // reference: the sequential recurrence, per query head.
    let mut want = vec![0.0f32; hq * l * dv];
    for h in 0..hq {
        let kvh = h / n_rep;
        let mut state = vec![0.0f32; dk * dv]; // S [D_k, D_v]
        for t in 0..l {
            for a in 0..dk {
                for c in 0..dv {
                    let kt = kd[(kvh * l + t) * dk + a];
                    let vt = vd[(kvh * l + t) * dv + c];
                    state[a * dv + c] = decay * state[a * dv + c] + kt * vt;
                }
            }
            for c in 0..dv {
                let mut acc = 0.0f32;
                for a in 0..dk {
                    acc += qd[(h * l + t) * dk + a] * state[a * dv + c];
                }
                want[(h * l + t) * dv + c] = acc;
            }
        }
    }
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
}

#[test]
fn linear_attention_prefill_per_channel_gate_matches_recurrence() {
    // card 038: per-channel gated prefill (naive form Q'=Q(.)A, K'=K(/)A, binary causal mask) must equal the per-channel
    // gated recurrence. Mild gates (0.7..0.95) and L=4 keep the cumulative product A well-conditioned in f32.
    let (l, hq, hkv, dk, dv) = (4usize, 2usize, 1usize, 3usize, 4usize);
    let n_rep = hq / hkv;
    let qd = fill(hq * l * dk, 3);
    let kd = fill(hkv * l * dk, 4);
    let vd = fill(hkv * l * dv, 5);
    let gate =
        |h: usize, t: usize, a: usize| 0.7 + 0.25 * (((h * 5 + t * 3 + a * 2) % 4) as f32 / 4.0);

    // cumgate A[h][t][a] = prod_{i<=t} g[h][i][a]; binary causal mask[t][j] = 1 for j<=t.
    let mut ad = vec![0.0f32; hq * l * dk];
    for h in 0..hq {
        for a in 0..dk {
            let mut p = 1.0f32;
            for t in 0..l {
                p *= gate(h, t, a);
                ad[(h * l + t) * dk + a] = p;
            }
        }
    }
    let mut md = vec![0.0f32; l * l];
    for t in 0..l {
        for j in 0..=t {
            md[t * l + j] = 1.0;
        }
    }

    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, dk]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, dk]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, dv]));
    let cg = b.constant("a", TensorType::f32(vec![1, hq, l, dk]));
    let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
    let o = linear_attention_prefill_gated(&b, q, k, v, n_rep, cg, mask);
    let (qi, ki, vi, ai, mi) = (q.id, k.id, v.id, cg.id, mask.id);
    let g = b.finish(o);
    let mut inputs = HashMap::new();
    inputs.insert(
        qi,
        Value::from(HostTensor::f32(vec![1, hq, l, dk], qd.clone())),
    );
    inputs.insert(
        ki,
        Value::from(HostTensor::f32(vec![1, hkv, l, dk], kd.clone())),
    );
    inputs.insert(
        vi,
        Value::from(HostTensor::f32(vec![1, hkv, l, dv], vd.clone())),
    );
    inputs.insert(ai, Value::from(HostTensor::f32(vec![1, hq, l, dk], ad)));
    inputs.insert(mi, Value::from(HostTensor::f32(vec![1, 1, l, l], md)));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![1, hq, l, dv]);

    // reference: per-channel gated sequential recurrence, per head.
    let mut want = vec![0.0f32; hq * l * dv];
    for h in 0..hq {
        let kvh = h / n_rep;
        let mut state = vec![0.0f32; dk * dv];
        for t in 0..l {
            for a in 0..dk {
                for c in 0..dv {
                    let kt = kd[(kvh * l + t) * dk + a];
                    let vt = vd[(kvh * l + t) * dv + c];
                    state[a * dv + c] = gate(h, t, a) * state[a * dv + c] + kt * vt;
                }
            }
            for c in 0..dv {
                let mut acc = 0.0f32;
                for a in 0..dk {
                    acc += qd[(h * l + t) * dk + a] * state[a * dv + c];
                }
                want[(h * l + t) * dv + c] = acc;
            }
        }
    }
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
}

#[test]
fn linear_attention_prefill_chunked_beats_naive_under_strong_decay() {
    // card 038: at long L with strong decay the naive per-channel gated prefill (K/A, A = whole-sequence cumulative
    // gate) loses f32 precision, while the chunked form (K/beta, beta cumulative within a chunk) stays accurate.
    // L=64, chunk=8, decay=0.1: A_64 = 1e-64 underflows to zero in f32 (K/A is inf/NaN), while beta spans 8 positions
    // (1e-8, still normal). Asserts chunked is tight and decisively closer than naive.
    let (l, c, hq, hkv, dk, dv) = (64usize, 8usize, 1usize, 1usize, 4usize, 4usize);
    let n_rep = 1;
    let qd = fill(hq * l * dk, 7);
    let kd = fill(hkv * l * dk, 8);
    let vd = fill(hkv * l * dv, 9);
    let decay = 0.1f32; // strong enough that the whole-sequence cumulative product underflows to zero

    let mut full_a = vec![0.0f32; hq * l * dk]; // whole-sequence cumulative gate (naive)
    let mut chunk_b = vec![0.0f32; hq * l * dk]; // per-chunk cumulative gate (chunked)
    for h in 0..hq {
        for a in 0..dk {
            let mut pa = 1.0f32;
            for t in 0..l {
                pa *= decay;
                full_a[(h * l + t) * dk + a] = pa;
                let cs = (t / c) * c;
                let mut pb = 1.0f32;
                for _ in cs..=t {
                    pb *= decay;
                }
                chunk_b[(h * l + t) * dk + a] = pb;
            }
        }
    }
    let causal_full: Vec<f32> = (0..l * l)
        .map(|i| if i % l <= i / l { 1.0 } else { 0.0 })
        .collect();
    let causal_chunk: Vec<f32> = (0..c * c)
        .map(|i| if i % c <= i / c { 1.0 } else { 0.0 })
        .collect();

    let run =
        |build: &dyn Fn(&Builder) -> Traced, extra: &[(&str, Vec<usize>, Vec<f32>)]| -> Vec<f32> {
            let b = Builder::new();
            let o = build(&b);
            let g = b.finish(o);
            // resolve constant ids by name from the graph.
            let mut inputs = HashMap::new();
            for &id in &g.inputs {
                let name = g.values[id].name.clone().unwrap();
                let (shape, data) = match name.as_str() {
                    "q" => (vec![1, hq, l, dk], qd.clone()),
                    "k" => (vec![1, hkv, l, dk], kd.clone()),
                    "v" => (vec![1, hkv, l, dv], vd.clone()),
                    other => {
                        let e = extra.iter().find(|(n, _, _)| *n == other).unwrap();
                        (e.1.clone(), e.2.clone())
                    }
                };
                inputs.insert(id, Value::from(HostTensor::f32(shape, data)));
            }
            eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .expect("linear_attention tests evaluate dense graphs")
                .as_f32()
                .unwrap()
                .to_vec()
        };

    let naive = run(
        &|b| {
            let q = b.constant("q", TensorType::f32(vec![1, hq, l, dk]));
            let k = b.constant("k", TensorType::f32(vec![1, hkv, l, dk]));
            let v = b.constant("v", TensorType::f32(vec![1, hkv, l, dv]));
            let cg = b.constant("cg", TensorType::f32(vec![1, hq, l, dk]));
            let m = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
            linear_attention_prefill_gated(b, q, k, v, n_rep, cg, m)
        },
        &[
            ("cg", vec![1, hq, l, dk], full_a.clone()),
            ("m", vec![1, 1, l, l], causal_full.clone()),
        ],
    );
    let chunked = run(
        &|b| {
            let q = b.constant("q", TensorType::f32(vec![1, hq, l, dk]));
            let k = b.constant("k", TensorType::f32(vec![1, hkv, l, dk]));
            let v = b.constant("v", TensorType::f32(vec![1, hkv, l, dv]));
            let be = b.constant("be", TensorType::f32(vec![1, hq, l, dk]));
            let ca = b.constant("ca", TensorType::f32(vec![1, 1, c, c]));
            linear_attention_prefill_chunked(b, q, k, v, n_rep, be, ca, c)
        },
        &[
            ("be", vec![1, hq, l, dk], chunk_b.clone()),
            ("ca", vec![1, 1, c, c], causal_chunk.clone()),
        ],
    );

    // exact recurrence reference.
    let mut want = vec![0.0f32; hq * l * dv];
    for h in 0..hq {
        let mut state = vec![0.0f32; dk * dv];
        for t in 0..l {
            for a in 0..dk {
                for cc in 0..dv {
                    state[a * dv + cc] = decay * state[a * dv + cc]
                        + kd[(h * l + t) * dk + a] * vd[(h * l + t) * dv + cc];
                }
            }
            for cc in 0..dv {
                let mut acc = 0.0f32;
                for a in 0..dk {
                    acc += qd[(h * l + t) * dk + a] * state[a * dv + cc];
                }
                want[(h * l + t) * dv + cc] = acc;
            }
        }
    }
    // Under strong decay the naive form is non-finite (A underflows to 0, so K/A is inf), and NaN elements can
    // appear in either output. A non-finite element is the worst possible error: `f32::max` alone drops a NaN and
    // reads it as 0, which would hide a NaN naive result and let a NaN chunked result pass.
    let err = |x: &[f32]| -> f32 {
        x.iter()
            .zip(&want)
            .map(|(a, b)| {
                let e = (a - b).abs();
                if e.is_finite() { e } else { f32::INFINITY }
            })
            .fold(0.0f32, f32::max)
    };
    let (chunk_err, naive_err) = (err(&chunked), err(&naive));
    eprintln!("strong-decay L={l}: chunked_err={chunk_err:.2e}, naive_err={naive_err:.2e}");
    assert!(
        chunk_err < 1e-3,
        "chunked must stay accurate: {chunk_err:.2e}"
    );
    assert!(
        naive_err > 10.0 * chunk_err,
        "chunked ({chunk_err:.2e}) must be decisively more accurate than naive ({naive_err:.2e})"
    );
}

#[test]
fn linear_attention_prefill_chunked_matches_recurrence() {
    // card 038: chunked-scan per-channel gated prefill must equal the per-channel gated sequential recurrence.
    // L=8, chunk=4 exercises both inter-chunk state carry and intra-chunk terms. beta is the per-chunk cumulative gate product.
    let (l, c, hq, hkv, dk, dv) = (8usize, 4usize, 2usize, 1usize, 3usize, 4usize);
    let n_rep = hq / hkv;
    let qd = fill(hq * l * dk, 3);
    let kd = fill(hkv * l * dk, 4);
    let vd = fill(hkv * l * dv, 5);
    let gate =
        |h: usize, t: usize, a: usize| 0.6 + 0.3 * (((h * 5 + t * 3 + a * 2) % 4) as f32 / 4.0);

    // chunk_beta[h][t][a] = prod_{m=chunkstart..=t} g[h][m][a] (cumulative from each chunk's start, inclusive).
    let mut bd = vec![0.0f32; hq * l * dk];
    for h in 0..hq {
        for a in 0..dk {
            for t in 0..l {
                let cs = (t / c) * c;
                let mut p = 1.0f32;
                for m in cs..=t {
                    p *= gate(h, m, a);
                }
                bd[(h * l + t) * dk + a] = p;
            }
        }
    }
    let mut cd = vec![0.0f32; c * c]; // binary causal [chunk, chunk]
    for t in 0..c {
        for j in 0..=t {
            cd[t * c + j] = 1.0;
        }
    }

    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, dk]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, dk]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, dv]));
    let beta = b.constant("be", TensorType::f32(vec![1, hq, l, dk]));
    let causal = b.constant("ca", TensorType::f32(vec![1, 1, c, c]));
    let o = linear_attention_prefill_chunked(&b, q, k, v, n_rep, beta, causal, c);
    let (qi, ki, vi, bi, ci) = (q.id, k.id, v.id, beta.id, causal.id);
    let g = b.finish(o);
    let mut inputs = HashMap::new();
    inputs.insert(
        qi,
        Value::from(HostTensor::f32(vec![1, hq, l, dk], qd.clone())),
    );
    inputs.insert(
        ki,
        Value::from(HostTensor::f32(vec![1, hkv, l, dk], kd.clone())),
    );
    inputs.insert(
        vi,
        Value::from(HostTensor::f32(vec![1, hkv, l, dv], vd.clone())),
    );
    inputs.insert(bi, Value::from(HostTensor::f32(vec![1, hq, l, dk], bd)));
    inputs.insert(ci, Value::from(HostTensor::f32(vec![1, 1, c, c], cd)));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![1, hq, l, dv]);

    // reference: per-channel gated sequential recurrence.
    let mut want = vec![0.0f32; hq * l * dv];
    for h in 0..hq {
        let kvh = h / n_rep;
        let mut state = vec![0.0f32; dk * dv];
        for t in 0..l {
            for a in 0..dk {
                for cc in 0..dv {
                    let kt = kd[(kvh * l + t) * dk + a];
                    let vt = vd[(kvh * l + t) * dv + cc];
                    state[a * dv + cc] = gate(h, t, a) * state[a * dv + cc] + kt * vt;
                }
            }
            for cc in 0..dv {
                let mut acc = 0.0f32;
                for a in 0..dk {
                    acc += qd[(h * l + t) * dk + a] * state[a * dv + cc];
                }
                want[(h * l + t) * dv + cc] = acc;
            }
        }
    }
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
}

/// Build a random STRICTLY-lower-triangular `[c,c]` matrix (zero on and above the diagonal) from
/// [`fill`]'s deterministic pseudo-random stream.
fn random_strictly_lower_triangular(c: usize, seed: u64) -> Vec<f32> {
    let raw = fill(c * c, seed);
    let mut out = vec![0.0f32; c * c];
    for i in 0..c {
        for j in 0..i {
            out[i * c + j] = raw[i * c + j];
        }
    }
    out
}

/// UT-transform sub-fn in isolation (no chunk loop, no model). `attn` is a random strictly-lower-triangular
/// `[C,C]` matrix, so `M = I + attn` is unit-lower-triangular. `unit_lower_triangular_inverse` computes
/// `T = M^{-1}` by block-recursive halving; asserts `M @ T == I` within 1e-5.
#[test]
fn ut_transform_inverse_reconstructs_identity() {
    let c = 8usize;
    let attn = random_strictly_lower_triangular(c, 11);

    let b = Builder::new();
    let attn_t = b.constant("attn", TensorType::f32(vec![c, c]));
    let t = unit_lower_triangular_inverse(&b, attn_t);
    assert_eq!(b.aval(t).shape, vec![c, c], "T keeps the [C,C] shape");
    let g = b.finish(t);
    let mut inputs = HashMap::new();
    inputs.insert(
        attn_t.id,
        Value::from(HostTensor::f32(vec![c, c], attn.clone())),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![c, c]);

    // M = I + attn
    let mut m = attn.clone();
    for i in 0..c {
        m[i * c + i] += 1.0;
    }
    let recon = matmul_ref(&m, got.as_f32().unwrap(), c, c, c);
    let mut identity = vec![0.0f32; c * c];
    for i in 0..c {
        identity[i * c + i] = 1.0;
    }
    let err = max_abs_error(&recon, &identity);
    eprintln!("ut_transform_inverse_reconstructs_identity C={c}: max_abs_err={err:.2e}");
    assert!(
        err < 1e-5,
        "(I+attn)@T must reconstruct I within 1e-5, got {err:.2e}"
    );
}

/// Realistic GDN-style strictly-lower-triangular attn `[c,c]`: `attn[i,j] = beta_i * dot(k_i, k_j)` for `j < i`,
/// with L2-normalized `k` rows and `beta_i in (0,1)`. Unlike [`random_strictly_lower_triangular`] this is
/// well-conditioned (`|dot(k_i,k_j)| <= 1`), matching the magnitude range the UT-transform sees in production.
fn realistic_gdn_attn(c: usize, dk: usize, seed: u64) -> Vec<f32> {
    let raw = fill(c * dk, seed);
    let mut k = vec![0.0f32; c * dk];
    for i in 0..c {
        let row = &raw[i * dk..(i + 1) * dk];
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for a in 0..dk {
            k[i * dk + a] = row[a] / norm;
        }
    }
    let beta = fill(c, seed.wrapping_add(1));
    let mut attn = vec![0.0f32; c * c];
    for i in 0..c {
        let beta_i = 0.5 + 0.45 * beta[i]; // (0.05, 0.95) -> a real (0,1) delta-rule weight
        for j in 0..i {
            let dot: f32 = (0..dk).map(|a| k[i * dk + a] * k[j * dk + a]).sum();
            attn[i * c + j] = beta_i * dot;
        }
    }
    attn
}

/// Same as [`ut_transform_inverse_reconstructs_identity`] at `C=64` (llama.cpp's typical chunk size) with a
/// realistic GDN-shaped attn ([`realistic_gdn_attn`]): exercises the full 6-level halving depth, not just `C=8`.
///
/// A raw uniform `[-1,1)` attn at `C=64` is adversarial: `M=I+attn` is badly conditioned (`T` entries reach ~2e3)
/// and f32 rounding gives ~3.7e-4 reconstruction error even though `T` matches an f64 reference. See
/// [`ut_transform_inverse_matches_forward_substitution_under_adversarial_input`].
#[test]
fn ut_transform_inverse_reconstructs_identity_c64() {
    let c = 64usize;
    let attn = realistic_gdn_attn(c, 32, 23);

    let b = Builder::new();
    let attn_t = b.constant("attn", TensorType::f32(vec![c, c]));
    let t = unit_lower_triangular_inverse(&b, attn_t);
    let g = b.finish(t);
    let mut inputs = HashMap::new();
    inputs.insert(
        attn_t.id,
        Value::from(HostTensor::f32(vec![c, c], attn.clone())),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![c, c]);

    let mut m = attn.clone();
    for i in 0..c {
        m[i * c + i] += 1.0;
    }
    let recon = matmul_ref(&m, got.as_f32().unwrap(), c, c, c);
    let mut identity = vec![0.0f32; c * c];
    for i in 0..c {
        identity[i * c + i] = 1.0;
    }
    let err = max_abs_error(&recon, &identity);
    let max_t = got
        .as_f32()
        .unwrap()
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()));
    eprintln!(
        "ut_transform_inverse_reconstructs_identity_c64 C={c}: max_abs_err={err:.2e} max|T|={max_t:.2e}"
    );
    assert!(
        err < 1e-5,
        "(I+attn)@T must reconstruct I within 1e-5, got {err:.2e}"
    );
}

/// f64 forward-substitution reference for inverting a unit-lower-triangular matrix (column-by-column solve
/// `M @ T[:,col] = e_col`), independent of the graph's block-recursive halving. Compares `T` entries directly.
fn unit_lower_triangular_inverse_f64_ref(m: &[f64], c: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; c * c];
    for col in 0..c {
        t[col * c + col] = 1.0 / m[col * c + col];
        for row in (col + 1)..c {
            let mut acc = 0.0f64;
            for kk in col..row {
                acc += m[row * c + kk] * t[kk * c + col];
            }
            t[row * c + col] = -acc / m[row * c + row];
        }
    }
    t
}

/// Cross-checks [`unit_lower_triangular_inverse`] against the f64 forward-substitution reference
/// ([`unit_lower_triangular_inverse_f64_ref`]) under an adversarial raw-uniform `[-1,1)` attn at `C=64`, where
/// `T` entries reach ~2e3 and the `(I+attn)@T==I` check is not a fair 1e-5 gate (see
/// [`ut_transform_inverse_reconstructs_identity_c64`]). Uses a relative tolerance of 1e-4; the reconstruction
/// error there is conditioning, not a wrong `T`.
#[test]
fn ut_transform_inverse_matches_forward_substitution_under_adversarial_input() {
    let c = 64usize;
    let attn = random_strictly_lower_triangular(c, 23);

    let b = Builder::new();
    let attn_t = b.constant("attn", TensorType::f32(vec![c, c]));
    let t = unit_lower_triangular_inverse(&b, attn_t);
    let g = b.finish(t);
    let mut inputs = HashMap::new();
    inputs.insert(
        attn_t.id,
        Value::from(HostTensor::f32(vec![c, c], attn.clone())),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");

    let attn64: Vec<f64> = attn.iter().map(|&v| v as f64).collect();
    let mut m64 = attn64;
    for i in 0..c {
        m64[i * c + i] += 1.0;
    }
    let t64 = unit_lower_triangular_inverse_f64_ref(&m64, c);
    let max_t64 = t64.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));

    let mut worst_rel = 0.0f64;
    for (got_v, ref_v) in got.as_f32().unwrap().iter().zip(&t64) {
        let denom = ref_v.abs().max(1.0); // floor so near-zero entries don't dominate the relative error
        let rel = (*got_v as f64 - ref_v).abs() / denom;
        // `f64::max` would drop a NaN and read it as 0.
        assert!(
            rel.is_finite(),
            "non-finite T entry: got {got_v} vs {ref_v}"
        );
        worst_rel = worst_rel.max(rel);
    }
    eprintln!(
        "ut_transform_inverse_matches_forward_substitution_under_adversarial_input C={c}: max|T_f64ref|={max_t64:.2e} worst_rel_diff={worst_rel:.2e}"
    );
    assert!(
        worst_rel < 1e-4,
        "block-recursive T must match the f64 forward-substitution reference (worst_rel_diff={worst_rel:.2e})"
    );
}

/// Same as [`ut_transform_inverse_reconstructs_identity`] with a batched leading dim `[H,C,C]`: the last-two-axes
/// inverse must compose through a batched `MatMul` without cross-contaminating heads.
#[test]
fn ut_transform_inverse_reconstructs_identity_batched() {
    let (h, c) = (3usize, 8usize);
    let mut attn = vec![0.0f32; h * c * c];
    for head in 0..h {
        let block = random_strictly_lower_triangular(c, 100 + head as u64);
        attn[head * c * c..(head + 1) * c * c].copy_from_slice(&block);
    }

    let b = Builder::new();
    let attn_t = b.constant("attn", TensorType::f32(vec![h, c, c]));
    let t = unit_lower_triangular_inverse(&b, attn_t);
    assert_eq!(b.aval(t).shape, vec![h, c, c]);
    let g = b.finish(t);
    let mut inputs = HashMap::new();
    inputs.insert(
        attn_t.id,
        Value::from(HostTensor::f32(vec![h, c, c], attn.clone())),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![h, c, c]);

    let mut identity = vec![0.0f32; c * c];
    for i in 0..c {
        identity[i * c + i] = 1.0;
    }
    let mut worst = 0.0f32;
    for head in 0..h {
        let a = &attn[head * c * c..(head + 1) * c * c];
        let mut m = a.to_vec();
        for i in 0..c {
            m[i * c + i] += 1.0;
        }
        let t_head = &got.as_f32().unwrap()[head * c * c..(head + 1) * c * c];
        let recon = matmul_ref(&m, t_head, c, c, c);
        let err = max_abs_error(&recon, &identity);
        worst = worst.max(err);
        assert!(
            err < 1e-5,
            "head {head}: (I+attn)@T must reconstruct I within 1e-5, got {err:.2e}"
        );
    }
    eprintln!(
        "ut_transform_inverse_reconstructs_identity_batched H={h} C={c}: worst_max_abs_err={worst:.2e}"
    );
}

/// `gdn_prefill_chunked` must reproduce `l` sequential [`gated_delta_net_decode`] calls exactly (output and final
/// state). The decode op is the llama.cpp-validated oracle; only an external oracle catches a wrong delta-rule
/// chunk formula. `h_k=2, h_v=4` (tiled GQA): the decode reference gets a host-tiled H_v copy of q/k (`hv` reads
/// kv-head `hv % h_k`, matching [`repeat_kv_tiled`]), while `gdn_prefill_chunked` gets raw H_k q/k. `s_in` is a
/// nonzero random state so both state-dependent cross terms (`k_cumdecay @ S_in`, `(Q (.) A) @ S_in`) are
/// exercised. `l` / `chunk` are parameters so the `L % C != 0` test reuses this.
fn check_gdn_prefill_chunked_matches_decode(l: usize, chunk: usize, abs_tol_only: bool) {
    check_gdn_prefill_chunked_matches_decode_ex(l, chunk, 2, 4, 3, 0, 1e-4, abs_tol_only);
}

/// Generalized [`check_gdn_prefill_chunked_matches_decode`] with configurable head/dim shape, a `seed_base`
/// for independent deterministic random streams, and a relative tolerance `tol` (default callers use 1e-4).
#[allow(clippy::too_many_arguments)]
fn check_gdn_prefill_chunked_matches_decode_ex(
    l: usize,
    chunk: usize,
    h_k: usize,
    h_v: usize,
    d: usize,
    seed_base: u64,
    tol: f32,
    abs_tol_only: bool,
) {
    let l2norm = |raw: &mut [f32], rows: usize, dim: usize| {
        for r in 0..rows {
            let row = &mut raw[r * dim..(r + 1) * dim];
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    };

    let mut qd = fill(h_k * l * d, seed_base + 101); // [1,H_k,L,D], L2-normed, UNSCALED (S4 scales internally)
    l2norm(&mut qd, h_k * l, d);
    let mut kd = fill(h_k * l * d, seed_base + 202); // [1,H_k,L,D], L2-normed
    l2norm(&mut kd, h_k * l, d);
    let vd = fill(h_v * l * d, seed_base + 303); // [1,H_v,L,D]
    // g: small negative log-decay (already log-domain; no Log applied).
    let gd: Vec<f32> = (0..h_v * l)
        .map(|i| -0.05 - 0.03 * ((i * 5 + i / l) % 4) as f32)
        .collect();
    // beta in a realistic (0,1) range.
    let betad: Vec<f32> = (0..h_v * l)
        .map(|i| 0.3 + 0.5 * (((i * 3 + i / l) % 4) as f32 / 4.0))
        .collect();
    let s0 = fill(h_v * d * d, seed_base + 404); // nonzero initial state [1,H_v,D,D]

    // Host-side tiled repeat: head hv reads kv-head hv % h_k (`repeat_kv_tiled` convention, not blocked `repeat_kv`).
    let tile = |src: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; h_v * l * d];
        for hv in 0..h_v {
            let hk = hv % h_k;
            out[hv * l * d..(hv + 1) * l * d].copy_from_slice(&src[hk * l * d..(hk + 1) * l * d]);
        }
        out
    };
    let qv = tile(&qd);
    let kv = tile(&kd);

    // --- sequential decode-recurrence reference (H_v heads, state carried across L steps) ---
    let db = Builder::new();
    let q_in = db.constant("q", TensorType::f32(vec![1, h_v, 1, d]));
    let k_in = db.constant("k", TensorType::f32(vec![1, h_v, 1, d]));
    let v_in = db.constant("v", TensorType::f32(vec![1, h_v, 1, d]));
    let g_in = db.constant("g", TensorType::f32(vec![1, h_v, 1, 1]));
    let bt_in = db.constant("beta", TensorType::f32(vec![1, h_v, 1, 1]));
    let ds_in = db.state_input(
        "s",
        TensorType::f32(vec![1, h_v, d, d]),
        StateRole::Recurrent,
    );
    let (o, s_out) = gated_delta_net_decode(&db, q_in, k_in, v_in, g_in, bt_in, ds_in);
    let dg = db.finish_with_state(o, &[(ds_in, s_out)]);

    let mut state = HostTensor::f32(vec![1, h_v, d, d], s0.clone());
    let mut want_o = vec![0.0f32; h_v * l * d];
    for t in 0..l {
        let qt: Vec<f32> = (0..h_v * d)
            .map(|i| qv[(i / d * l + t) * d + i % d])
            .collect();
        let kt: Vec<f32> = (0..h_v * d)
            .map(|i| kv[(i / d * l + t) * d + i % d])
            .collect();
        let vt: Vec<f32> = (0..h_v * d)
            .map(|i| vd[(i / d * l + t) * d + i % d])
            .collect();
        let gt: Vec<f32> = (0..h_v).map(|hh| gd[hh * l + t]).collect();
        let btt: Vec<f32> = (0..h_v).map(|hh| betad[hh * l + t]).collect();

        let mut inp = HashMap::new();
        inp.insert(
            q_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, d], qt)),
        );
        inp.insert(
            k_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, d], kt)),
        );
        inp.insert(
            v_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, d], vt)),
        );
        inp.insert(
            g_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, 1], gt)),
        );
        inp.insert(
            bt_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, 1], btt)),
        );
        inp.insert(ds_in.id, Value::from(state.clone()));

        let (ot, new_states) = eval(&dg, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("linear_attention state is dense"))
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();
        state = new_states.into_iter().next().unwrap();
        assert_eq!(ot.shape(), vec![1, h_v, 1, d]);
        for hh in 0..h_v {
            for dd in 0..d {
                want_o[(hh * l + t) * d + dd] = ot.as_f32().unwrap()[hh * d + dd];
            }
        }
    }
    let want_state = state.as_f32().unwrap().to_vec();

    // --- chunked prefill: raw H_k q/k, H_v v/g/beta, ONE call over all L positions ---
    let c = chunk;
    let mut tril_incl = vec![0.0f32; c * c]; // lower-triangular INCLUDING the diagonal (S2)
    let mut tril_strict = vec![0.0f32; c * c]; // STRICTLY lower-triangular, zero diagonal (S2)
    for i in 0..c {
        for j in 0..c {
            if j <= i {
                tril_incl[i * c + j] = 1.0;
            }
            if j < i {
                tril_strict[i * c + j] = 1.0;
            }
        }
    }

    let cb = Builder::new();
    let cq = cb.constant("q", TensorType::f32(vec![1, h_k, l, d]));
    let ck = cb.constant("k", TensorType::f32(vec![1, h_k, l, d]));
    let cv = cb.constant("v", TensorType::f32(vec![1, h_v, l, d]));
    let cg = cb.constant("g", TensorType::f32(vec![1, h_v, l, 1]));
    let cbeta = cb.constant("beta", TensorType::f32(vec![1, h_v, l, 1]));
    let cs_in = cb.state_input(
        "s",
        TensorType::f32(vec![1, h_v, d, d]),
        StateRole::Recurrent,
    );
    let ctril_incl = cb.constant("tril_incl", TensorType::f32(vec![1, 1, c, c]));
    let ctril_strict = cb.constant("tril_strict", TensorType::f32(vec![1, 1, c, c]));
    let (co, cs_out) = gdn_prefill_chunked(
        &cb,
        cq,
        ck,
        cv,
        cg,
        cbeta,
        cs_in,
        ctril_incl,
        ctril_strict,
        c,
    );
    assert_eq!(
        cb.aval(co).shape,
        vec![1, h_v, l, d],
        "gdn_prefill_chunked output shape [1,H_v,L,D]"
    );
    assert_eq!(
        cb.aval(cs_out).shape,
        vec![1, h_v, d, d],
        "gdn_prefill_chunked state shape [1,H_v,D,D]"
    );
    let cgraph = cb.finish_with_state(co, &[(cs_in, cs_out)]);

    let mut cinp = HashMap::new();
    cinp.insert(cq.id, Value::from(HostTensor::f32(vec![1, h_k, l, d], qd)));
    cinp.insert(ck.id, Value::from(HostTensor::f32(vec![1, h_k, l, d], kd)));
    cinp.insert(cv.id, Value::from(HostTensor::f32(vec![1, h_v, l, d], vd)));
    cinp.insert(cg.id, Value::from(HostTensor::f32(vec![1, h_v, l, 1], gd)));
    cinp.insert(
        cbeta.id,
        Value::from(HostTensor::f32(vec![1, h_v, l, 1], betad)),
    );
    cinp.insert(
        cs_in.id,
        Value::from(HostTensor::f32(vec![1, h_v, d, d], s0)),
    );
    cinp.insert(
        ctril_incl.id,
        Value::from(HostTensor::f32(vec![1, 1, c, c], tril_incl)),
    );
    cinp.insert(
        ctril_strict.id,
        Value::from(HostTensor::f32(vec![1, 1, c, c], tril_strict)),
    );

    let (got_o, got_states) = eval(&cgraph, &cinp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| {
            let state = r
                .state
                .into_iter()
                .map(|v| v.into_host().expect("linear_attention state is dense"))
                .collect::<Vec<_>>();
            (
                r.output
                    .into_host()
                    .expect("linear_attention tests evaluate dense graphs"),
                state,
            )
        })
        .unwrap();
    assert_eq!(got_o.shape(), vec![1, h_v, l, d]);
    let got_state = got_states.into_iter().next().unwrap();
    assert_eq!(got_state.shape(), vec![1, h_v, d, d]);

    let o_err = max_abs_error(got_o.as_f32().unwrap(), &want_o);
    let s_err = max_abs_error(got_state.as_f32().unwrap(), &want_state);
    eprintln!(
        "gdn_prefill_chunked l={l} chunk={c}: output_max_abs_err={o_err:.2e} state_max_abs_err={s_err:.2e}"
    );
    if abs_tol_only {
        // Heavy padding (C=64) leaves many near-zero output elements where the relative `close` bound is
        // meaningless; both errs are ~1e-7. Assert the absolute-`tol` convention the full-model prefill oracle uses.
        assert!(o_err < tol, "output max_abs={o_err:.3e} >= {tol:.1e}");
        assert!(s_err < tol, "state max_abs={s_err:.3e} >= {tol:.1e}");
    } else {
        assert_close_rel(got_o.as_f32().unwrap(), &want_o, tol); // SC-001
        assert_close_rel(got_state.as_f32().unwrap(), &want_state, tol); // SC-002
    }
}

/// SC-001/002: >=2 chunks (`L=8, C=4`), `H_v > H_k` tiled GQA.
#[test]
fn gdn_prefill_chunked_matches_decode_recurrence() {
    check_gdn_prefill_chunked_matches_decode(8, 4, false);
}

/// FR-005: `L % C != 0` (`L=6, C=4`, pad=2) must zero-pad + truncate transparently, matching the exact
/// recurrence over the true `L=6` positions (no residue from the padded tail).
#[test]
fn gdn_prefill_chunked_matches_decode_recurrence_padded() {
    check_gdn_prefill_chunked_matches_decode(6, 4, false);
}

/// Card 158: heavy padding (short prompt, production chunk C=64; the real 35B pads ~6 positions to a 64-chunk).
/// The padded tail must be inert for the state carry on the CPU oracle (output and s_out ~1e-7 vs the recurrence).
#[test]
fn gdn_prefill_chunked_matches_decode_recurrence_heavy_pad() {
    check_gdn_prefill_chunked_matches_decode(6, 64, true);
    check_gdn_prefill_chunked_matches_decode(10, 64, true);
}

/// Card 158: multi-chunk at the production chunk size C=64. Other `gdn_prefill_chunked` oracle tests use toy chunk
/// sizes or a single padded C=64 chunk, so the inter-chunk state carry at C=64 was unchecked. Cases: L=100 (64+36),
/// L=128 (two full chunks), L=130 (64+64+2), L=200 (64+64+64+8), with GQA n_rep=2 and n_rep=1 configs and
/// independent seed streams. Uses `abs_tol_only` because small-magnitude outputs (~1e-3..1e-4) turn ~1e-7 float
/// noise into large relative error. The tight 1e-5 absolute bound would be exceeded by orders of magnitude
/// if the carry were wrong.
#[test]
fn gdn_prefill_chunked_matches_decode_recurrence_multi_chunk_c64() {
    // (h_k, h_v, d): default GQA shape, tiled-repeat n_rep=2.
    check_gdn_prefill_chunked_matches_decode_ex(100, 64, 2, 4, 3, 1_000, 1e-5, true);
    check_gdn_prefill_chunked_matches_decode_ex(128, 64, 2, 4, 3, 2_000, 1e-5, true);
    // GQA n_rep=2 with wider heads/dim, L=130 leaves only 2 real positions in the final chunk.
    check_gdn_prefill_chunked_matches_decode_ex(130, 64, 4, 8, 4, 3_000, 1e-5, true);
    // No GQA repeat (n_rep=1), 4 full-ish chunks.
    check_gdn_prefill_chunked_matches_decode_ex(200, 64, 2, 2, 5, 4_000, 1e-5, true);
}

/// Card 158 diagnostic: sweep decay magnitude at real GDN dims (H_k=16, H_v=32, D=128, C=64, L=115). Real
/// Qwen3-Next feeds g = softplus(dt) * ssm_a with ssm_a down to ~-72 (blk.0), far larger than the small-g
/// synthetic oracle (g in [-0.05, -0.17], D <= 8). Prints errors, asserts nothing.
#[test]
fn gdn_prefill_chunked_decay_magnitude_sweep_real_dims() {
    let (h_k, h_v, d, chunk, l) = (16usize, 32usize, 128usize, 64usize, 115usize);
    let l2norm = |raw: &mut [f32], rows: usize, dim: usize| {
        for r in 0..rows {
            let row = &mut raw[r * dim..(r + 1) * dim];
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    };
    // tiled GQA repeat H_k -> H_v (head hv reads kv-head hv % h_k).
    let tile = |src: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; h_v * l * d];
        for hv in 0..h_v {
            let hk = hv % h_k;
            out[hv * l * d..(hv + 1) * l * d].copy_from_slice(&src[hk * l * d..(hk + 1) * l * d]);
        }
        out
    };

    let mut qd = fill(h_k * l * d, 101);
    l2norm(&mut qd, h_k * l, d);
    let mut kd = fill(h_k * l * d, 202);
    l2norm(&mut kd, h_k * l, d);
    let vd = fill(h_v * l * d, 303);
    let betad: Vec<f32> = (0..h_v * l)
        .map(|i| 0.3 + 0.5 * ((i % 4) as f32 / 4.0))
        .collect();
    let s0 = vec![0.0f32; h_v * d * d]; // fresh prefill starts from zero state (matches the real prefill)
    let qv = tile(&qd);
    let kv = tile(&kd);

    // tril masks (C=64).
    let c = chunk;
    let mut tril_incl = vec![0.0f32; c * c];
    let mut tril_strict = vec![0.0f32; c * c];
    for i in 0..c {
        for j in 0..c {
            if j <= i {
                tril_incl[i * c + j] = 1.0;
            }
            if j < i {
                tril_strict[i * c + j] = 1.0;
            }
        }
    }

    // Build the chunked-prefill graph once (inputs vary per decay level).
    let cb = Builder::new();
    let cq = cb.constant("q", TensorType::f32(vec![1, h_k, l, d]));
    let ck = cb.constant("k", TensorType::f32(vec![1, h_k, l, d]));
    let cv = cb.constant("v", TensorType::f32(vec![1, h_v, l, d]));
    let cg = cb.constant("g", TensorType::f32(vec![1, h_v, l, 1]));
    let cbeta = cb.constant("beta", TensorType::f32(vec![1, h_v, l, 1]));
    let cs_in = cb.state_input(
        "s",
        TensorType::f32(vec![1, h_v, d, d]),
        StateRole::Recurrent,
    );
    let cti = cb.constant("tril_incl", TensorType::f32(vec![1, 1, c, c]));
    let cts = cb.constant("tril_strict", TensorType::f32(vec![1, 1, c, c]));
    let (co, cs_out) = gdn_prefill_chunked(&cb, cq, ck, cv, cg, cbeta, cs_in, cti, cts, c);
    let cgraph = cb.finish_with_state(co, &[(cs_in, cs_out)]);

    // Sequential decode-recurrence reference graph (single step, H_v heads).
    let db = Builder::new();
    let q_in = db.constant("q", TensorType::f32(vec![1, h_v, 1, d]));
    let k_in = db.constant("k", TensorType::f32(vec![1, h_v, 1, d]));
    let v_in = db.constant("v", TensorType::f32(vec![1, h_v, 1, d]));
    let g_in = db.constant("g", TensorType::f32(vec![1, h_v, 1, 1]));
    let bt_in = db.constant("beta", TensorType::f32(vec![1, h_v, 1, 1]));
    let ds_in = db.state_input(
        "s",
        TensorType::f32(vec![1, h_v, d, d]),
        StateRole::Recurrent,
    );
    let (o, s_out) = gated_delta_net_decode(&db, q_in, k_in, v_in, g_in, bt_in, ds_in);
    let dg = db.finish_with_state(o, &[(ds_in, s_out)]);

    eprintln!(
        "=== gdn_prefill_chunked vs sequential recurrence, REAL dims (H_v=32,D=128,C=64,L=115) ==="
    );
    // Sweep both decay and value magnitude: the block feeds v = silu(conv(proj)), which can be O(10-100), and per-head
    // ssm_a spans -0.02..-72.
    for &(g_mean, v_scale) in &[
        (-0.05f32, 1.0f32),
        (-0.5, 1.0),
        (-5.0, 1.0),
        (-0.5, 50.0),
        (-5.0, 50.0),
        (-40.0, 50.0),
    ] {
        let vd: Vec<f32> = vd.iter().map(|x| x * v_scale).collect();
        // per-(head,pos) log-decay around g_mean (all negative), same values fed to both paths.
        let gd: Vec<f32> = (0..h_v * l)
            .map(|i| g_mean * (0.7 + 0.3 * ((i % 5) as f32 / 5.0)))
            .collect();

        // chunked prefill.
        let mut cinp = HashMap::new();
        cinp.insert(
            cq.id,
            Value::from(HostTensor::f32(vec![1, h_k, l, d], qd.clone())),
        );
        cinp.insert(
            ck.id,
            Value::from(HostTensor::f32(vec![1, h_k, l, d], kd.clone())),
        );
        cinp.insert(
            cv.id,
            Value::from(HostTensor::f32(vec![1, h_v, l, d], vd.clone())),
        );
        cinp.insert(
            cg.id,
            Value::from(HostTensor::f32(vec![1, h_v, l, 1], gd.clone())),
        );
        cinp.insert(
            cbeta.id,
            Value::from(HostTensor::f32(vec![1, h_v, l, 1], betad.clone())),
        );
        cinp.insert(
            cs_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, d, d], s0.clone())),
        );
        cinp.insert(
            cti.id,
            Value::from(HostTensor::f32(vec![1, 1, c, c], tril_incl.clone())),
        );
        cinp.insert(
            cts.id,
            Value::from(HostTensor::f32(vec![1, 1, c, c], tril_strict.clone())),
        );
        let (got_o, _) = eval(&cgraph, &cinp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("linear_attention state is dense"))
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();

        // sequential recurrence reference (fresh zero state).
        let mut state = HostTensor::f32(vec![1, h_v, d, d], s0.clone());
        let mut want_o = vec![0.0f32; h_v * l * d];
        for t in 0..l {
            let qt: Vec<f32> = (0..h_v * d)
                .map(|i| qv[(i / d * l + t) * d + i % d])
                .collect();
            let kt: Vec<f32> = (0..h_v * d)
                .map(|i| kv[(i / d * l + t) * d + i % d])
                .collect();
            let vt: Vec<f32> = (0..h_v * d)
                .map(|i| vd[(i / d * l + t) * d + i % d])
                .collect();
            let gt: Vec<f32> = (0..h_v).map(|hh| gd[hh * l + t]).collect();
            let btt: Vec<f32> = (0..h_v).map(|hh| betad[hh * l + t]).collect();
            let mut inp = HashMap::new();
            inp.insert(
                q_in.id,
                Value::from(HostTensor::f32(vec![1, h_v, 1, d], qt)),
            );
            inp.insert(
                k_in.id,
                Value::from(HostTensor::f32(vec![1, h_v, 1, d], kt)),
            );
            inp.insert(
                v_in.id,
                Value::from(HostTensor::f32(vec![1, h_v, 1, d], vt)),
            );
            inp.insert(
                g_in.id,
                Value::from(HostTensor::f32(vec![1, h_v, 1, 1], gt)),
            );
            inp.insert(
                bt_in.id,
                Value::from(HostTensor::f32(vec![1, h_v, 1, 1], btt)),
            );
            inp.insert(ds_in.id, Value::from(state.clone()));
            let (ot, ns) = eval(&dg, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| {
                    let state = r
                        .state
                        .into_iter()
                        .map(|v| v.into_host().expect("linear_attention state is dense"))
                        .collect::<Vec<_>>();
                    (
                        r.output
                            .into_host()
                            .expect("linear_attention tests evaluate dense graphs"),
                        state,
                    )
                })
                .unwrap();
            state = ns.into_iter().next().unwrap();
            for hh in 0..h_v {
                for dd in 0..d {
                    want_o[(hh * l + t) * d + dd] = ot.as_f32().unwrap()[hh * d + dd];
                }
            }
        }
        let err = max_abs_error(got_o.as_f32().unwrap(), &want_o);
        // last-position error.
        let mut last_err = 0.0f32;
        for hh in 0..h_v {
            let last_row = (hh * l + (l - 1)) * d..(hh * l + l) * d;
            last_err = last_err.max(max_abs_error(
                &got_o.as_f32().unwrap()[last_row.clone()],
                &want_o[last_row],
            ));
        }
        eprintln!(
            "g_mean={g_mean:>6} v_scale={v_scale:>5}: chunked-vs-sequential max_abs_error(all)={err:.3e} last_pos={last_err:.3e}"
        );
    }
}

/// Follow-up to `gdn_prefill_chunked_decay_magnitude_sweep_real_dims`: after prefill, does the (tiny, ~1e-6..1e-8)
/// chunked-vs-sequential state divergence grow over further decode steps, or shrink/stay flat?
///
/// Real dims (H_v=32, D=128, C=64, L=115) across the same decay/value levels (g_mean=-0.5 through -40). Builds the
/// two post-prefill states, feeds both the same 10 real-magnitude decode inputs (already GQA-repeated to H_v), and
/// tracks max-abs-err per step. `exp(g) in (0,1)` contracts the recurrent state, so the perturbation should decay
/// geometrically. Diagnostic-first (prints every step), with one soft guard per level: the error at step 10 must not
/// exceed 5x the step-1 error.
#[test]
fn gdn_decode_continuation_error_shrinks_after_chunked_prefill_real_dims() {
    let (h_v, d, chunk, l) = (32usize, 128usize, 64usize, 115usize);
    let l2norm = |raw: &mut [f32], rows: usize, dim: usize| {
        for r in 0..rows {
            let row = &mut raw[r * dim..(r + 1) * dim];
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    };
    for &(g_mean, v_scale) in &[(-0.5f32, 1.0f32), (-5.0, 50.0), (-40.0, 50.0)] {
        gdn_decode_continuation_one_level(h_v, d, chunk, l, g_mean, v_scale, &l2norm);
    }
}

#[allow(clippy::too_many_arguments)]
fn gdn_decode_continuation_one_level(
    h_v: usize,
    d: usize,
    chunk: usize,
    l: usize,
    g_mean: f32,
    v_scale: f32,
    l2norm: &dyn Fn(&mut [f32], usize, usize),
) {
    let mut qd = fill(h_v * l * d, 701);
    l2norm(&mut qd, h_v * l, d);
    let mut kd = fill(h_v * l * d, 702);
    l2norm(&mut kd, h_v * l, d);
    let vd: Vec<f32> = fill(h_v * l * d, 703)
        .into_iter()
        .map(|x| x * v_scale)
        .collect();
    let betad: Vec<f32> = (0..h_v * l)
        .map(|i| 0.3 + 0.5 * ((i % 4) as f32 / 4.0))
        .collect();
    let gd: Vec<f32> = (0..h_v * l)
        .map(|i| g_mean * (0.7 + 0.3 * ((i % 5) as f32 / 5.0)))
        .collect();
    let s0 = vec![0.0f32; h_v * d * d];

    // tril masks (C=64).
    let c = chunk;
    let mut tril_incl = vec![0.0f32; c * c];
    let mut tril_strict = vec![0.0f32; c * c];
    for i in 0..c {
        for j in 0..c {
            if j <= i {
                tril_incl[i * c + j] = 1.0;
            }
            if j < i {
                tril_strict[i * c + j] = 1.0;
            }
        }
    }

    // Chunked-prefill state (H_v heads directly; GQA repeat is covered by the sweep above).
    let cb = Builder::new();
    let cq = cb.constant("q", TensorType::f32(vec![1, h_v, l, d]));
    let ck = cb.constant("k", TensorType::f32(vec![1, h_v, l, d]));
    let cv = cb.constant("v", TensorType::f32(vec![1, h_v, l, d]));
    let cg = cb.constant("g", TensorType::f32(vec![1, h_v, l, 1]));
    let cbeta = cb.constant("beta", TensorType::f32(vec![1, h_v, l, 1]));
    let cs_in = cb.state_input(
        "s",
        TensorType::f32(vec![1, h_v, d, d]),
        StateRole::Recurrent,
    );
    let cti = cb.constant("tril_incl", TensorType::f32(vec![1, 1, c, c]));
    let cts = cb.constant("tril_strict", TensorType::f32(vec![1, 1, c, c]));
    let (_co, cs_out) = gdn_prefill_chunked(&cb, cq, ck, cv, cg, cbeta, cs_in, cti, cts, c);
    let cgraph = cb.finish_with_state(_co, &[(cs_in, cs_out)]);
    let mut cinp = HashMap::new();
    cinp.insert(
        cq.id,
        Value::from(HostTensor::f32(vec![1, h_v, l, d], qd.clone())),
    );
    cinp.insert(
        ck.id,
        Value::from(HostTensor::f32(vec![1, h_v, l, d], kd.clone())),
    );
    cinp.insert(
        cv.id,
        Value::from(HostTensor::f32(vec![1, h_v, l, d], vd.clone())),
    );
    cinp.insert(
        cg.id,
        Value::from(HostTensor::f32(vec![1, h_v, l, 1], gd.clone())),
    );
    cinp.insert(
        cbeta.id,
        Value::from(HostTensor::f32(vec![1, h_v, l, 1], betad.clone())),
    );
    cinp.insert(
        cs_in.id,
        Value::from(HostTensor::f32(vec![1, h_v, d, d], s0.clone())),
    );
    cinp.insert(
        cti.id,
        Value::from(HostTensor::f32(vec![1, 1, c, c], tril_incl)),
    );
    cinp.insert(
        cts.id,
        Value::from(HostTensor::f32(vec![1, 1, c, c], tril_strict)),
    );
    let (_, chunked_states) = eval(&cgraph, &cinp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| {
            let state = r
                .state
                .into_iter()
                .map(|v| v.into_host().expect("linear_attention state is dense"))
                .collect::<Vec<_>>();
            (
                r.output
                    .into_host()
                    .expect("linear_attention tests evaluate dense graphs"),
                state,
            )
        })
        .unwrap();
    let mut state_chunked = chunked_states.into_iter().next().unwrap();

    // Sequential decode-recurrence reference graph, used for the post-prefill "want" state (L steps) and the
    // 10 continuation steps.
    let db = Builder::new();
    let q_in = db.constant("q", TensorType::f32(vec![1, h_v, 1, d]));
    let k_in = db.constant("k", TensorType::f32(vec![1, h_v, 1, d]));
    let v_in = db.constant("v", TensorType::f32(vec![1, h_v, 1, d]));
    let g_in = db.constant("g", TensorType::f32(vec![1, h_v, 1, 1]));
    let bt_in = db.constant("beta", TensorType::f32(vec![1, h_v, 1, 1]));
    let ds_in = db.state_input(
        "s",
        TensorType::f32(vec![1, h_v, d, d]),
        StateRole::Recurrent,
    );
    let (o, s_out) = gated_delta_net_decode(&db, q_in, k_in, v_in, g_in, bt_in, ds_in);
    let dg = db.finish_with_state(o, &[(ds_in, s_out)]);

    let step = |state: &HostTensor,
                qt: Vec<f32>,
                kt: Vec<f32>,
                vt: Vec<f32>,
                gt: Vec<f32>,
                btt: Vec<f32>| {
        let mut inp = HashMap::new();
        inp.insert(
            q_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, d], qt)),
        );
        inp.insert(
            k_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, d], kt)),
        );
        inp.insert(
            v_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, d], vt)),
        );
        inp.insert(
            g_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, 1], gt)),
        );
        inp.insert(
            bt_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, 1], btt)),
        );
        inp.insert(ds_in.id, Value::from(state.clone()));
        let (ot, ns) = eval(&dg, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("linear_attention state is dense"))
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();
        (ot, ns.into_iter().next().unwrap())
    };

    let mut state_seq = HostTensor::f32(vec![1, h_v, d, d], s0);
    for t in 0..l {
        let qt: Vec<f32> = (0..h_v * d)
            .map(|i| qd[(i / d * l + t) * d + i % d])
            .collect();
        let kt: Vec<f32> = (0..h_v * d)
            .map(|i| kd[(i / d * l + t) * d + i % d])
            .collect();
        let vt: Vec<f32> = (0..h_v * d)
            .map(|i| vd[(i / d * l + t) * d + i % d])
            .collect();
        let gt: Vec<f32> = (0..h_v).map(|hh| gd[hh * l + t]).collect();
        let btt: Vec<f32> = (0..h_v).map(|hh| betad[hh * l + t]).collect();
        let (_, ns) = step(&state_seq, qt, kt, vt, gt, btt);
        state_seq = ns;
    }

    let state_err0 = max_abs_error(state_chunked.as_f32().unwrap(), state_seq.as_f32().unwrap());
    eprintln!(
        "gdn_decode_continuation g_mean={g_mean:>6} v_scale={v_scale:>5}: post-prefill state max_abs_err={state_err0:.3e}"
    );

    // 10 further decode steps, SAME inputs fed to both trajectories.
    let mut first_step_err = None;
    let mut last_step_err = 0.0f32;
    for t in 0..10usize {
        let mut qt = fill(h_v * d, 800 + t as u64);
        l2norm(&mut qt, h_v, d);
        let mut kt = fill(h_v * d, 900 + t as u64);
        l2norm(&mut kt, h_v, d);
        let vt: Vec<f32> = fill(h_v * d, 1000 + t as u64)
            .into_iter()
            .map(|x| x * v_scale)
            .collect();
        let gt: Vec<f32> = (0..h_v)
            .map(|hh| g_mean * (0.7 + 0.3 * ((hh % 5) as f32 / 5.0)))
            .collect();
        let btt: Vec<f32> = (0..h_v)
            .map(|hh| 0.3 + 0.5 * ((hh % 4) as f32 / 4.0))
            .collect();

        let (out_chunked, ns_chunked) = step(
            &state_chunked,
            qt.clone(),
            kt.clone(),
            vt.clone(),
            gt.clone(),
            btt.clone(),
        );
        let (out_seq, ns_seq) = step(&state_seq, qt, kt, vt, gt, btt);

        let out_err = max_abs_error(out_chunked.as_f32().unwrap(), out_seq.as_f32().unwrap());
        let st_err = max_abs_error(ns_chunked.as_f32().unwrap(), ns_seq.as_f32().unwrap());
        eprintln!(
            "  decode step {t}: output_max_abs_err={out_err:.3e} state_max_abs_err={st_err:.3e}"
        );
        if first_step_err.is_none() {
            first_step_err = Some(out_err.max(1e-12));
        }
        last_step_err = out_err;

        state_chunked = ns_chunked;
        state_seq = ns_seq;
    }

    let first = first_step_err.unwrap();
    eprintln!(
        "gdn_decode_continuation: step0_err={first:.3e} step9_err={last_step_err:.3e} ratio={:.3}",
        last_step_err / first
    );
    assert!(
        last_step_err <= first * 5.0 + 1e-9,
        "post-prefill state divergence grew unboundedly over 10 decode steps: \
         step0={first:.3e} step9={last_step_err:.3e} (expected shrink/flat, GDN gating contracts state)"
    );
}

/// Card 158: `cumulative_log_decay_scan` (Hillis-Steele doubling additive scan, ~6 rounds at C=64) must be at least
/// as precise as the `tril_incl @ g` matmul cumsum it replaces, at real dims (C=64, H_v=32) and real-shaped decay
/// (ssm_a spanning -0.02..-72, alpha~1, modest per-position variation). Both are compared against an f64 cumulative
/// sum. At the time of the fix: matmul ~1.8e-3 abs error, scan ~3.7e-4. The scan alone does not reach decode's
/// ~1e-5 for the most extreme per-head decay (see `gdn_core_on_realistic_frontend_inputs_per_head_real_dims`; the smaller
/// `GDN_PREFILL_CHUNK` covers the rest).
#[test]
fn cumulative_log_decay_scan_at_least_as_precise_as_matmul_cumsum() {
    let (hv, c) = (32usize, 64usize);
    let ssm_a: Vec<f32> = (0..hv)
        .map(|i| -0.02 - 72.0 * (i as f32 / (hv - 1) as f32))
        .collect();
    let mut gd = vec![0.0f32; hv * c];
    for h in 0..hv {
        for t in 0..c {
            let alpha = 0.7 + 0.3 * (((h * 7 + t * 3) % 5) as f32 / 5.0); // in [0.7,1.0), like softplus(dt)
            gd[h * c + t] = alpha * ssm_a[h];
        }
    }
    // matmul-based cumsum reference.
    let mut ti = vec![0.0f32; c * c];
    for i in 0..c {
        for j in 0..=i {
            ti[i * c + j] = 1.0;
        }
    }
    let mb = Builder::new();
    let mg = mb.constant("g", TensorType::f32(vec![1, hv, c, 1]));
    let mt = mb.constant("ti", TensorType::f32(vec![1, 1, c, c]));
    let mcum = mb.matmul(mt, mg);
    let mgraph = mb.finish(mcum);
    let mut minp = HashMap::new();
    minp.insert(
        mg.id,
        Value::from(HostTensor::f32(vec![1, hv, c, 1], gd.clone())),
    );
    minp.insert(mt.id, Value::from(HostTensor::f32(vec![1, 1, c, c], ti)));
    let matmul_cum = eval(&mgraph, &minp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");

    // Hillis-Steele scan.
    let sb = Builder::new();
    let sg = sb.constant("g", TensorType::f32(vec![1, hv, c, 1]));
    let scum = poot_graph_ir::ops::cumulative_log_decay_scan(&sb, sg, c);
    let sgraph = sb.finish(scum);
    let mut sinp = HashMap::new();
    sinp.insert(
        sg.id,
        Value::from(HostTensor::f32(vec![1, hv, c, 1], gd.clone())),
    );
    let scan_cum = eval(&sgraph, &sinp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");

    // f64 ground truth, per head.
    let mut truth = vec![0.0f64; hv * c];
    for h in 0..hv {
        let mut acc = 0.0f64;
        for t in 0..c {
            acc += gd[h * c + t] as f64;
            truth[h * c + t] = acc;
        }
    }
    let worst_matmul = max_abs_error_f64(matmul_cum.as_f32().unwrap(), &truth);
    let worst_scan = max_abs_error_f64(scan_cum.as_f32().unwrap(), &truth);
    eprintln!(
        "cumlog abs err vs f64 ground truth: matmul={worst_matmul:.3e} hillis-steele-scan={worst_scan:.3e}"
    );
    assert!(
        worst_scan <= worst_matmul * 1.5,
        "Hillis-Steele scan should not be less precise than the matmul cumsum it replaces \
         (scan={worst_scan:.3e}, matmul={worst_matmul:.3e})"
    );
}

/// Card 158: `gdn_prefill_chunked` must stay within decode's float-noise floor (~1e-5) even when one position
/// mid-chunk applies a huge decay (g down to -60) while its neighbors stay modest. A multiplicative
/// absolute-decay-with-division formulation loses the shared reference magnitude once any position's cumulative
/// decay underflows to 0.0 (max_abs_err ~0.46; see `cumulative_log_decay_scan`). H_v=1, H_k=1, D=2, C=8, L=8
/// (single chunk, so any error comes from the intra-chunk decay computation, not the state carry).
#[test]
fn gdn_prefill_chunked_matches_decode_at_extreme_within_chunk_decay_step() {
    // g varies across positions (some large, some small) to stress the ratio computation within one chunk.
    let (h_k, h_v, d, c, l) = (1usize, 1usize, 2usize, 8usize, 8usize);
    let qd = {
        let mut v = fill(h_k * l * d, 501);
        let norm = |row: &mut [f32]| {
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in row.iter_mut() {
                *x /= n;
            }
        };
        for r in 0..h_k * l {
            norm(&mut v[r * d..(r + 1) * d]);
        }
        v
    };
    let kd = {
        let mut v = fill(h_k * l * d, 502);
        let norm = |row: &mut [f32]| {
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in row.iter_mut() {
                *x /= n;
            }
        };
        for r in 0..h_k * l {
            norm(&mut v[r * d..(r + 1) * d]);
        }
        v
    };
    let vd = fill(h_v * l * d, 503);
    // VARIED per-position decay: alternates moderate/large/huge magnitude.
    let gd: Vec<f32> = vec![-0.5, -30.0, -1.0, -60.0, -0.3, -0.2, -45.0, -0.1];
    let betad: Vec<f32> = (0..l).map(|i| 0.3 + 0.5 * ((i % 4) as f32 / 4.0)).collect();
    let s0 = vec![0.0f32; h_v * d * d];

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
    let cb = Builder::new();
    let cq = cb.constant("q", TensorType::f32(vec![1, h_k, l, d]));
    let ck = cb.constant("k", TensorType::f32(vec![1, h_k, l, d]));
    let cv = cb.constant("v", TensorType::f32(vec![1, h_v, l, d]));
    let cg = cb.constant("g", TensorType::f32(vec![1, h_v, l, 1]));
    let cbeta = cb.constant("beta", TensorType::f32(vec![1, h_v, l, 1]));
    let cs = cb.state_input(
        "s",
        TensorType::f32(vec![1, h_v, d, d]),
        StateRole::Recurrent,
    );
    let cti = cb.constant("ti", TensorType::f32(vec![1, 1, c, c]));
    let cts = cb.constant("ts", TensorType::f32(vec![1, 1, c, c]));
    let (co, cso) = gdn_prefill_chunked(&cb, cq, ck, cv, cg, cbeta, cs, cti, cts, c);
    let cgraph = cb.finish_with_state(co, &[(cs, cso)]);
    let mut cin = HashMap::new();
    cin.insert(
        cq.id,
        Value::from(HostTensor::f32(vec![1, h_k, l, d], qd.clone())),
    );
    cin.insert(
        ck.id,
        Value::from(HostTensor::f32(vec![1, h_k, l, d], kd.clone())),
    );
    cin.insert(
        cv.id,
        Value::from(HostTensor::f32(vec![1, h_v, l, d], vd.clone())),
    );
    cin.insert(
        cg.id,
        Value::from(HostTensor::f32(vec![1, h_v, l, 1], gd.clone())),
    );
    cin.insert(
        cbeta.id,
        Value::from(HostTensor::f32(vec![1, h_v, l, 1], betad.clone())),
    );
    cin.insert(
        cs.id,
        Value::from(HostTensor::f32(vec![1, h_v, d, d], s0.clone())),
    );
    cin.insert(cti.id, Value::from(HostTensor::f32(vec![1, 1, c, c], ti)));
    cin.insert(cts.id, Value::from(HostTensor::f32(vec![1, 1, c, c], ts)));
    let (got_o, got_s) = eval(&cgraph, &cin, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| {
            let state = r
                .state
                .into_iter()
                .map(|v| v.into_host().expect("linear_attention state is dense"))
                .collect::<Vec<_>>();
            (
                r.output
                    .into_host()
                    .expect("linear_attention tests evaluate dense graphs"),
                state,
            )
        })
        .unwrap();

    // sequential reference.
    let db = Builder::new();
    let q_in = db.constant("q", TensorType::f32(vec![1, h_v, 1, d]));
    let k_in = db.constant("k", TensorType::f32(vec![1, h_v, 1, d]));
    let v_in = db.constant("v", TensorType::f32(vec![1, h_v, 1, d]));
    let g_in = db.constant("g", TensorType::f32(vec![1, h_v, 1, 1]));
    let bt_in = db.constant("beta", TensorType::f32(vec![1, h_v, 1, 1]));
    let ds_in = db.state_input(
        "s",
        TensorType::f32(vec![1, h_v, d, d]),
        StateRole::Recurrent,
    );
    let (o, s_out) = gated_delta_net_decode(&db, q_in, k_in, v_in, g_in, bt_in, ds_in);
    let dg = db.finish_with_state(o, &[(ds_in, s_out)]);
    let mut state = HostTensor::f32(vec![1, h_v, d, d], s0.clone());
    let mut want_o = vec![0.0f32; l * d];
    for t in 0..l {
        let mut inp = HashMap::new();
        inp.insert(
            q_in.id,
            Value::from(HostTensor::f32(
                vec![1, h_v, 1, d],
                qd[t * d..(t + 1) * d].to_vec(),
            )),
        );
        inp.insert(
            k_in.id,
            Value::from(HostTensor::f32(
                vec![1, h_v, 1, d],
                kd[t * d..(t + 1) * d].to_vec(),
            )),
        );
        inp.insert(
            v_in.id,
            Value::from(HostTensor::f32(
                vec![1, h_v, 1, d],
                vd[t * d..(t + 1) * d].to_vec(),
            )),
        );
        inp.insert(
            g_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, 1], vec![gd[t]])),
        );
        inp.insert(
            bt_in.id,
            Value::from(HostTensor::f32(vec![1, h_v, 1, 1], vec![betad[t]])),
        );
        inp.insert(ds_in.id, Value::from(state.clone()));
        let (ot, ns) = eval(&dg, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("linear_attention state is dense"))
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();
        state = ns.into_iter().next().unwrap();
        want_o[t * d..(t + 1) * d].copy_from_slice(ot.as_f32().unwrap());
    }
    eprintln!("g = {gd:?}");
    eprintln!("got_o  = {:?}", got_o.as_f32().unwrap());
    eprintln!("want_o = {want_o:?}");
    eprintln!("got_s  = {:?}", got_s[0].as_f32().unwrap());
    eprintln!("want_s = {:?}", state.as_f32().unwrap());
    let err = max_abs_error(got_o.as_f32().unwrap(), &want_o);
    let serr = max_abs_error(got_s[0].as_f32().unwrap(), state.as_f32().unwrap());
    eprintln!("output_max_abs_err={err:.3e} state_max_abs_err={serr:.3e}");
    assert!(
        err < 1e-4,
        "output diverged at the mid-chunk decay step: {err:.3e}"
    );
    assert!(
        serr < 1e-4,
        "state diverged at the mid-chunk decay step: {serr:.3e}"
    );
}

/// Is the batched causal conv (`causal_conv1d_prefill`) faithful to L sequential `causal_conv1d_decode` steps at the
/// real GDN conv shape (conv_dim=8192, K=4, L=115)? Decode (llama.cpp-identical) uses the incremental conv-state
/// form; prefill uses a batched shift-multiply-add. Diagnostic: prints the max error, asserts nothing.
#[test]
fn causal_conv1d_prefill_vs_decode_real_dims() {
    let (conv_dim, k, l) = (8192usize, 4usize, 115usize);
    let xd = fill(l * conv_dim, 77); // [1, L, conv_dim]
    let wd = fill(k * conv_dim, 88); // [K, conv_dim]

    // batched prefill conv.
    let pb = Builder::new();
    let px = pb.constant("x", TensorType::f32(vec![1, l, conv_dim]));
    let pw = pb.constant("w", TensorType::f32(vec![k, conv_dim]));
    let po = causal_conv1d_prefill(&pb, px, pw, k);
    let pg = pb.finish(po);
    let mut pin = HashMap::new();
    pin.insert(
        px.id,
        Value::from(HostTensor::f32(vec![1, l, conv_dim], xd.clone())),
    );
    pin.insert(
        pw.id,
        Value::from(HostTensor::f32(vec![k, conv_dim], wd.clone())),
    );
    let got = eval(&pg, &pin, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![1, l, conv_dim]);

    // L sequential decode-conv steps from a fresh zero cache.
    let db = Builder::new();
    let dx = db.constant("x", TensorType::f32(vec![1, 1, conv_dim]));
    let dw = db.constant("w", TensorType::f32(vec![k, conv_dim]));
    let dc = db.state_input(
        "c",
        TensorType::f32(vec![1, k - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let (dout, dcout) = causal_conv1d_decode(&db, dx, dw, dc, k);
    let dgg = db.finish_with_state(dout, &[(dc, dcout)]);
    let mut cache = HostTensor::f32(vec![1, k - 1, conv_dim], vec![0.0f32; (k - 1) * conv_dim]);
    let mut want = vec![0.0f32; l * conv_dim];
    for t in 0..l {
        let xt = xd[t * conv_dim..(t + 1) * conv_dim].to_vec();
        let mut inp = HashMap::new();
        inp.insert(
            dx.id,
            Value::from(HostTensor::f32(vec![1, 1, conv_dim], xt)),
        );
        inp.insert(
            dw.id,
            Value::from(HostTensor::f32(vec![k, conv_dim], wd.clone())),
        );
        inp.insert(dc.id, Value::from(cache.clone()));
        let (ot, ns) = eval(&dgg, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("linear_attention state is dense"))
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();
        cache = ns.into_iter().next().unwrap();
        want[t * conv_dim..(t + 1) * conv_dim].copy_from_slice(ot.as_f32().unwrap());
    }
    let err = max_abs_error(got.as_f32().unwrap(), &want);
    eprintln!(
        "causal_conv1d prefill-vs-decode REAL dims (conv_dim=8192,K=4,L=115): max_abs_err={err:.3e}"
    );
}

#[test]
fn linear_attention_prefill_per_head_gate_matches_recurrence() {
    // card 038: per-head data-dependent gating is covered by `linear_attention_prefill` with a per-head mask[1,Hq,L,L]
    // of cumulative per-head gate products mask[h,t,j] = prod_{j<i<=t} g_h[i] (in [0,1], so stable). It must equal the
    // sequential recurrence S_t = g_h[t]*S + k^T v per head. L=5, Hq=2, Hkv=1, D_k=3, D_v=4. (Per-channel gating needs a scan.)
    let (l, hq, hkv, dk, dv) = (5usize, 2usize, 1usize, 3usize, 4usize);
    let n_rep = hq / hkv;
    let qd = fill(hq * l * dk, 3);
    let kd = fill(hkv * l * dk, 4);
    let vd = fill(hkv * l * dv, 5);
    let gate = |h: usize, t: usize| 0.6 + 0.35 * (((h * 7 + t * 3) % 5) as f32 / 5.0); // per-(head,position)

    // mask[h][t][j] = prod_{i=j+1..=t} g_h[i] for j<=t, else 0.
    let mut md = vec![0.0f32; hq * l * l];
    for h in 0..hq {
        for t in 0..l {
            for j in 0..=t {
                let mut p = 1.0f32;
                for i in (j + 1)..=t {
                    p *= gate(h, i);
                }
                md[(h * l + t) * l + j] = p;
            }
        }
    }

    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, dk]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, dk]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, dv]));
    let mask = b.constant("m", TensorType::f32(vec![1, hq, l, l])); // PER-HEAD mask (not broadcast)
    let o = linear_attention_prefill(&b, q, k, v, n_rep, mask);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(o);
    let mut inputs = HashMap::new();
    inputs.insert(
        qi,
        Value::from(HostTensor::f32(vec![1, hq, l, dk], qd.clone())),
    );
    inputs.insert(
        ki,
        Value::from(HostTensor::f32(vec![1, hkv, l, dk], kd.clone())),
    );
    inputs.insert(
        vi,
        Value::from(HostTensor::f32(vec![1, hkv, l, dv], vd.clone())),
    );
    inputs.insert(mi, Value::from(HostTensor::f32(vec![1, hq, l, l], md)));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![1, hq, l, dv]);

    // reference: per-head sequential recurrence with the per-head scalar gate.
    let mut want = vec![0.0f32; hq * l * dv];
    for h in 0..hq {
        let kvh = h / n_rep;
        let mut state = vec![0.0f32; dk * dv];
        for t in 0..l {
            for a in 0..dk {
                for c in 0..dv {
                    let kt = kd[(kvh * l + t) * dk + a];
                    let vt = vd[(kvh * l + t) * dv + c];
                    state[a * dv + c] = gate(h, t) * state[a * dv + c] + kt * vt;
                }
            }
            for c in 0..dv {
                let mut acc = 0.0f32;
                for a in 0..dk {
                    acc += qd[(h * l + t) * dk + a] * state[a * dv + c];
                }
                want[(h * l + t) * dv + c] = acc;
            }
        }
    }
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
}

#[test]
fn reduce_over_non_last_axis_keepdim_is_correct() {
    // regression: a keepdim reduce over a non-last axis used a skip-axis counter to index the out index, mis-reading
    // every dim after the axis (collapsed to index 0). [2,3,4] summed over axis 1 (keepdim) must equal the hand per-(i,k) sum.
    use poot_graph_ir::op::RedOp;
    let (a, b_, c) = (2usize, 3usize, 4usize);
    let xd: Vec<f32> = (0..a * b_ * c).map(|i| i as f32 * 0.5 - 3.0).collect();
    let bld = Builder::new();
    let x = bld.constant("x", TensorType::f32(vec![a, b_, c]));
    let r = bld.reduce(RedOp::Sum, x, 1, true); // [a,1,c]
    let g = bld.finish(r);
    let mut inp = HashMap::new();
    inp.insert(
        x.id,
        Value::from(HostTensor::f32(vec![a, b_, c], xd.clone())),
    );
    let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![a, 1, c]);
    let mut want = vec![0.0f32; a * c];
    for i in 0..a {
        for k in 0..c {
            for j in 0..b_ {
                want[i * c + k] += xd[(i * b_ + j) * c + k];
            }
        }
    }
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-6);
}

#[test]
fn causal_conv1d_decode_loop_matches_reference() {
    // card 038: depthwise causal conv1d (Mamba's short conv) run step by step, carrying the conv cache through State,
    // must equal a hand-computed causal conv over the sequence. K=4, C=6, L=5. (out[t][c] = sum_k w[k][c] * x[t-(K-1)+k][c], x[<0]=0.)
    let (k, c, l) = (4usize, 6usize, 5usize);
    let xd = fill(l * c, 21);
    let wd = fill(k * c, 22);

    // decode graph: x[1,1,C], cache[1,K-1,C] carried.
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

    let mut caches = vec![HostTensor::f32(
        vec![1, k - 1, c],
        vec![0.0f32; (k - 1) * c],
    )];
    let mut got = vec![0.0f32; l * c];
    for t in 0..l {
        let mut inp = HashMap::new();
        inp.insert(
            x.id,
            Value::from(HostTensor::f32(
                vec![1, 1, c],
                xd[t * c..t * c + c].to_vec(),
            )),
        );
        inp.insert(w.id, Value::from(HostTensor::f32(vec![k, c], wd.clone())));
        inp.insert(cache.id, Value::from(caches[0].clone()));
        let (o_t, new) = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("linear_attention state is dense"))
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();
        caches = new;
        got[t * c..t * c + c].copy_from_slice(o_t.as_f32().unwrap());
    }

    // hand causal conv.
    let mut want = vec![0.0f32; l * c];
    for t in 0..l {
        for ch in 0..c {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let src = t as isize - (k as isize - 1) + kk as isize;
                if src >= 0 {
                    acc += wd[kk * c + ch] * xd[src as usize * c + ch];
                }
            }
            want[t * c + ch] = acc;
        }
    }
    assert_close_rel(&got, &want, 1e-5);
}

/// `causal_conv1d_prefill` (batched causal depthwise conv) must reproduce `L` sequential [`causal_conv1d_decode`]
/// calls exactly, starting from a zero cache (decode is the trusted reference). K=4, C=8, L=6.
#[test]
fn causal_conv1d_prefill_matches_decode() {
    let (k, c, l) = (4usize, 8usize, 6usize);
    let xd = fill(l * c, 31); // [1, L, C]
    let wd = fill(k * c, 32); // [K, C]

    // --- sequential decode reference: L calls, cache carried, starting from a zero cache. ---
    let db = Builder::new();
    let dx = db.constant("x", TensorType::f32(vec![1, 1, c]));
    let dw = db.constant("w", TensorType::f32(vec![k, c]));
    let dcache = db.state_input(
        "cache",
        TensorType::f32(vec![1, k - 1, c]),
        StateRole::Recurrent,
    );
    let (dout, dcache_out) = causal_conv1d_decode(&db, dx, dw, dcache, k);
    let dg = db.finish_with_state(dout, &[(dcache, dcache_out)]);

    let mut cache = HostTensor::f32(vec![1, k - 1, c], vec![0.0f32; (k - 1) * c]);
    let mut want = vec![0.0f32; l * c];
    for t in 0..l {
        let mut inp = HashMap::new();
        inp.insert(
            dx.id,
            Value::from(HostTensor::f32(
                vec![1, 1, c],
                xd[t * c..t * c + c].to_vec(),
            )),
        );
        inp.insert(dw.id, Value::from(HostTensor::f32(vec![k, c], wd.clone())));
        inp.insert(dcache.id, Value::from(cache.clone()));
        let (o_t, new_states) = eval(&dg, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                let state = r
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("linear_attention state is dense"))
                    .collect::<Vec<_>>();
                (
                    r.output
                        .into_host()
                        .expect("linear_attention tests evaluate dense graphs"),
                    state,
                )
            })
            .unwrap();
        cache = new_states.into_iter().next().unwrap();
        want[t * c..t * c + c].copy_from_slice(o_t.as_f32().unwrap());
    }

    // --- batched prefill: ONE call over all L positions. ---
    let pb = Builder::new();
    let px = pb.constant("x", TensorType::f32(vec![1, l, c]));
    let pw = pb.constant("w", TensorType::f32(vec![k, c]));
    let pout = causal_conv1d_prefill(&pb, px, pw, k);
    assert_eq!(
        pb.aval(pout).shape,
        vec![1, l, c],
        "causal_conv1d_prefill output shape [1,L,C]"
    );
    let pg = pb.finish(pout);

    let mut pinp = HashMap::new();
    pinp.insert(px.id, Value::from(HostTensor::f32(vec![1, l, c], xd)));
    pinp.insert(pw.id, Value::from(HostTensor::f32(vec![k, c], wd)));
    let got = eval(&pg, &pinp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("linear_attention tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![1, l, c]);

    let err = max_abs_error(got.as_f32().unwrap(), &want);
    eprintln!("causal_conv1d_prefill_matches_decode: max_abs_err={err:.2e}");
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}
