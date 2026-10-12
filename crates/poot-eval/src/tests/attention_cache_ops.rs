//! Masked attention, softcap, matmul and the in-place slot write, evaluated on the CPU oracle against
//! hand-computed references. (The qwen2 model-trace rows live in `poot-models/tests/oracle_traces`.)

use crate::{EvalBudget, EvalOptions, Value};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::{attention, attention_masked, attention_softcap, softcap};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{StateRole, ValueId};
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::{assert_close_rel, max_abs_error};

#[test]
fn masked_attention_equals_sliced() {
    // G3d FR-001: masked attention over the full fixed-capacity cache == sliced attention over the valid prefix [0..=pos].
    // cap=4, pos=1 -> slots 0,1 valid, slots 2,3 masked (arbitrary stale data).
    let (hq, hkv, cap, d, pos) = (2usize, 1usize, 4usize, 4usize, 1usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();

    let qd = fill(hq * d, 8);
    let kd = fill(hkv * cap * d, 9); // slots past pos hold arbitrary (stale) values
    let vd = fill(hkv * cap * d, 10);
    let mut maskd = vec![0.0f32; cap];
    for m in maskd.iter_mut().skip(pos + 1) {
        *m = -1.0e9; // additive -inf for slots t > pos
    }

    // masked path: full cache + additive mask.
    let masked = {
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hkv, cap, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hkv, cap, d]))
            .unwrap();
        let mask = b
            .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
            .unwrap();
        let out = attention_masked(&b, q, k, v, n_rep, scale, mask).unwrap();
        let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
        let g = b.finish(out).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], qd.clone()));
        inputs.insert(ki, HostTensor::f32(vec![1, hkv, cap, d], kd.clone()));
        inputs.insert(vi, HostTensor::f32(vec![1, hkv, cap, d], vd.clone()));
        inputs.insert(mi, HostTensor::f32(vec![1, 1, 1, cap], maskd.clone()));
        {
            let values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .unwrap()
        }
    };

    // sliced path (G2): attend only the valid prefix [0..=pos].
    let sliced = {
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hkv, cap, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hkv, cap, d]))
            .unwrap();
        let kvalid = b.slice(k, 2, 0, pos + 1).unwrap();
        let vvalid = b.slice(v, 2, 0, pos + 1).unwrap();
        let out = attention(&b, q, kvalid, vvalid, n_rep, scale).unwrap();
        let (qi, ki, vi) = (q.id(), k.id(), v.id());
        let g = b.finish(out).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], qd.clone()));
        inputs.insert(ki, HostTensor::f32(vec![1, hkv, cap, d], kd.clone()));
        inputs.insert(vi, HostTensor::f32(vec![1, hkv, cap, d], vd.clone()));
        {
            let values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .unwrap()
        }
    };

    assert_close_rel(masked.as_f32().unwrap(), sliced.as_f32().unwrap(), 1e-5);
}

#[test]
fn softcap_evaluates_to_c_tanh_and_approaches_identity() {
    // B9 (Gemma2/Grok logit softcapping): softcap(x, c) = c * tanh(x / c), a composition of scalar mul, tanh, scalar mul.
    // Must evaluate elementwise to c*tanh(x/c), including large |x| where the internal tanh must not overflow.
    let n = 8usize;
    let c = 5.0f32;
    // include big magnitudes so |x/c| saturates tanh (stable-tanh no-overflow path).
    let xd: Vec<f32> = fill(n, 123).iter().map(|v| v * 40.0).collect();

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![n])).unwrap();
    let y = softcap(&b, x, c).unwrap();
    let xi = x.id();
    let g = b.finish(y).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![n], xd.clone()));
    let out = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    let want: Vec<f32> = xd.iter().map(|&v| c * (v / c).tanh()).collect();
    assert_close_rel(out.as_f32().unwrap(), &want, 1e-5);

    // As c grows, softcap approaches the identity on modest logits (tanh(x/c) ~ x/c). The internal tanh is an exp-based
    // composition (no `tanh`/`expm1` primitive), so for x/c below ~1e-4 the `1 - exp(-2 x/c)` numerator cancels to 0 in f32;
    // c ~ 1e9 is beyond what it can represent. A cap of 1e4 (~200x the real Gemma2/Grok caps of 30-50) shows the identity
    // limit in the accurate range.
    let big = 1.0e4f32;
    let modest: Vec<f32> = fill(n, 7).iter().map(|v| v * 10.0).collect();
    let b2 = Builder::for_tests();
    let x2 = b2.constant("x", TensorType::f32(vec![n])).unwrap();
    let y2 = softcap(&b2, x2, big).unwrap();
    let xi2 = x2.id();
    let g2 = b2.finish(y2).unwrap();
    let mut in2 = HashMap::new();
    in2.insert(xi2, HostTensor::f32(vec![n], modest.clone()));
    let out2 = {
        let values: HashMap<ValueId, Value> = in2
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g2, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    assert_close_rel(out2.as_f32().unwrap(), &modest, 1e-3);
}

#[test]
fn attention_softcap_matches_reference_and_differs_from_uncapped() {
    // B9: decode attention with attn_logit_softcap = Some(c) must equal a hand-rolled reference that applies c*tanh(scores/c)
    // to the scaled QK^T scores before softmax, and must differ from un-capped attention. Tiny deterministic GQA shapes.
    let (hq, hkv, s, d) = (2usize, 1usize, 3usize, 4usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let cap = 0.2f32; // small cap -> meaningful saturation vs the un-capped scores
    // scale the operands up so the raw scores are large relative to `cap` (guarantees a visible delta).
    let qd: Vec<f32> = fill(hq * d, 11).iter().map(|v| v * 2.0).collect();
    let kd: Vec<f32> = fill(hkv * s * d, 12).iter().map(|v| v * 2.0).collect();
    let vd = fill(hkv * s * d, 13);

    let run = |softcap_c: Option<f32>| {
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hkv, s, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hkv, s, d]))
            .unwrap();
        let out = attention_softcap(&b, q, k, v, n_rep, scale, softcap_c).unwrap();
        let (qi, ki, vi) = (q.id(), k.id(), v.id());
        let g = b.finish(out).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], qd.clone()));
        inputs.insert(ki, HostTensor::f32(vec![1, hkv, s, d], kd.clone()));
        inputs.insert(vi, HostTensor::f32(vec![1, hkv, s, d], vd.clone()));
        {
            let values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .unwrap()
        }
    };
    let capped = run(Some(cap));
    let uncapped = run(None);

    // independent Rust reference: capped attention per query head (GQA maps head h -> kv head h/n_rep).
    let mut want = vec![0.0f32; hq * d];
    for h in 0..hq {
        let kvh = h / n_rep;
        let mut scores = vec![0.0f32; s];
        for (si, sc_out) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for di in 0..d {
                dot += qd[h * d + di] * kd[(kvh * s + si) * d + di];
            }
            let sc = dot * scale;
            *sc_out = cap * (sc / cap).tanh(); // softcap before softmax
        }
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&z| (z - m).exp()).collect();
        let denom: f32 = exps.iter().sum();
        for (di, w) in want[h * d..h * d + d].iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for si in 0..s {
                acc += (exps[si] / denom) * vd[(kvh * s + si) * d + di];
            }
            *w = acc;
        }
    }
    assert_close_rel(capped.as_f32().unwrap(), &want, 1e-5);

    // softcapping changed the attention output (capped != uncapped somewhere).
    let maxdiff = max_abs_error(capped.as_f32().unwrap(), uncapped.as_f32().unwrap());
    assert!(
        maxdiff > 1e-3,
        "attn_logit_softcap should change the attention output (max diff {maxdiff})"
    );
}

#[test]
fn matmul_matches_naive() {
    // [1,1,3]x[3,2] -> [1,1,2]
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![1, 1, 3])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![3, 2])).unwrap();
    let out = b.matmul(a, w).unwrap();
    let (ai, wi) = (a.id(), w.id());
    let g = b.finish(out).unwrap();

    let ad = vec![1.0, 2.0, 3.0];
    let wd = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // row-major [3,2]
    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![1, 1, 3], ad.clone()));
    inputs.insert(wi, HostTensor::f32(vec![3, 2], wd.clone()));
    let got = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    // [1*1+2*2+3*3, 1*4+2*5+3*6] = [14, 32]
    assert_close_rel(got.as_f32().unwrap(), &[14.0, 32.0], 1e-6);
}

#[test]
// `st[0] * 0` / `st[0] * 1` below are the row indices in the stride formula (kept parallel across the two row
// assertions); the explicit form triggers erasing_op/identity_op.
#[allow(clippy::erasing_op, clippy::identity_op)]
fn dynamic_update_slice_writes_one_slot() {
    // a [2,4,3] cache; write a [2,1,3] update at slot index 2 on axis 1, the rest unchanged.
    let b = Builder::for_tests();
    let cache = b
        .state_input(
            "kcache",
            TensorType::f32(vec![2, 4, 3]),
            StateRole::Recurrent,
        )
        .unwrap();
    let upd = b.constant("upd", TensorType::f32(vec![2, 1, 3])).unwrap();
    let written = b.dynamic_update_slice(cache, upd, 2, 1).unwrap();
    // a trivial logits output so the graph is well-formed.
    let logits = b
        .constant("logits", TensorType::f32(vec![1, 1, 4]))
        .unwrap();
    let g = b
        .finish(poot_graph_ir::Outputs::new(logits).with_state(&[(cache, written)]))
        .unwrap();

    let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
    inputs.insert(cache.id(), HostTensor::zeros(vec![2, 4, 3]));
    inputs.insert(
        upd.id(),
        HostTensor::f32(vec![2, 1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    inputs.insert(logits.id(), HostTensor::zeros(vec![1, 1, 4]));

    let (_out, states) = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        (evaluation.output.into_host().unwrap(), state)
    };
    assert_eq!(states.len(), 1);
    let kc = &states[0];
    assert_eq!(kc.shape(), vec![2, 4, 3]);
    // row 0, slot 2 == [1,2,3]; row 1, slot 2 == [4,5,6]; everything else 0.
    let st = crate::strides(kc.shape());
    assert_eq!(
        &kc.as_f32().unwrap()[st[0] * 0 + st[1] * 2..][..3],
        &[1.0, 2.0, 3.0]
    );
    assert_eq!(
        &kc.as_f32().unwrap()[st[0] * 1 + st[1] * 2..][..3],
        &[4.0, 5.0, 6.0]
    );
    assert_eq!(
        kc.as_f32().unwrap().iter().filter(|&&v| v != 0.0).count(),
        6
    );
}
