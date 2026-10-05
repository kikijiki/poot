//! scatter_axis0, ArgTopK, top-k gate, MoE dense/sparse/grouped references.

use crate::ops::movement::scatter_axis0;
use crate::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::ValueId;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, RedOp};
use poot_graph_ir::ops::top_k_gate;
use poot_graph_ir::types::TensorType;
use poot_tensor::DType;
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::assert_close_rel;

#[test]
fn scatter_axis0_inverts_a_permutation() {
    // Scatter (card 034): out[index[j], ..] = src[j, ..]. With src = iota rows and index a permutation, the result is the
    // inverse permutation, the argtop-k index inversion sparse MoE needs.
    let b = Builder::new();
    let src = b.constant("src", TensorType::f32(vec![3, 2]));
    let idx = b.constant("idx", TensorType::f32(vec![3]));
    let s = b.scatter(src, idx);
    let (si, ii) = (src.id, idx.id);
    let g = b.finish(s);

    let mut inputs = HashMap::new();
    inputs.insert(
        si,
        Value::from(HostTensor::f32(
            vec![3, 2],
            vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        )),
    );
    // index = [2, 0, 1]: row 0 -> slot 2, row 1 -> slot 0, row 2 -> slot 1.
    inputs.insert(
        ii,
        Value::from(HostTensor::f32(vec![3], vec![2.0, 0.0, 1.0])),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("moe_gating tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![3, 2]);
    // out[0]=src[1], out[1]=src[2], out[2]=src[0].
    assert_close_rel(
        got.as_f32().unwrap(),
        &[20.0, 21.0, 30.0, 31.0, 10.0, 11.0],
        1e-9,
    );

    // scalar-row case: scatter iota [0,1,2,3] by a reverse permutation -> reversed (its own inverse here).
    let b2 = Builder::new();
    let src2 = b2.constant("s", TensorType::f32(vec![4]));
    let idx2 = b2.constant("i", TensorType::f32(vec![4]));
    let sc = b2.scatter(src2, idx2);
    let (s2, i2) = (src2.id, idx2.id);
    let g2 = b2.finish(sc);
    let mut in2 = HashMap::new();
    in2.insert(
        s2,
        Value::from(HostTensor::f32(vec![4], vec![0.0, 1.0, 2.0, 3.0])),
    );
    in2.insert(
        i2,
        Value::from(HostTensor::f32(vec![4], vec![3.0, 2.0, 1.0, 0.0])),
    );
    let got2 = eval(&g2, &in2, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("moe_gating tests evaluate dense graphs");
    assert_close_rel(got2.as_f32().unwrap(), &[3.0, 2.0, 1.0, 0.0], 1e-9);
}

/// mutants-m4 follow-up: a Scatter index outside `0..N` (past the end, or negative) used to slice
/// out of range and panic, or saturate to row 0 when negative. The oracle refuses it with a typed
/// error (card 555: `EvalError::Index`, through the one index rule) naming the position.
#[test]
fn scatter_refuses_an_index_outside_the_rows() {
    use crate::ops::index_rule::IndexFaultKind;

    for (index, position, kind) in [
        (vec![2.0, 3.0, 0.0], 1, IndexFaultKind::OutOfRange),
        (vec![2.0, 0.0, -1.0], 2, IndexFaultKind::Negative),
        (vec![f32::NAN, 0.0, 1.0], 0, IndexFaultKind::NotFinite),
    ] {
        let b = Builder::new();
        let src = b.constant("src", TensorType::f32(vec![3, 2]));
        let idx = b.constant("idx", TensorType::f32(vec![3]));
        let scattered = b.scatter(src, idx);
        let (si, ii) = (src.id, idx.id);
        let g = b.finish(scattered);
        let mut inputs = HashMap::new();
        inputs.insert(si, Value::from(HostTensor::f32(vec![3, 2], vec![0.0; 6])));
        inputs.insert(ii, Value::from(HostTensor::f32(vec![3], index.clone())));
        match eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).map(|r| {
            r.output
                .into_host()
                .expect("moe_gating tests evaluate dense graphs")
        }) {
            Err(crate::EvalError::Index(fault)) => {
                assert_eq!(fault.position, position, "{index:?}");
                assert_eq!(fault.len, 3, "{index:?}");
                assert_eq!(fault.kind, kind, "{index:?}");
            }
            other => panic!("{index:?}: expected EvalError::Index, got {other:?}"),
        }
    }
}

#[test]
fn arg_top_k_matches_per_row_scatter_iota_rank_slice() {
    // spec 136 P1: ArgTopK batches `scatter(iota,rank)[0..k]` (`moe_sparse`) over leading dims. E=32 and k=4 with a batch
    // of L=6 rows, so leading-dim batching is exercised.
    use poot_graph_ir::op::OpKind;
    let (l, e, k) = (6usize, 32usize, 4usize);

    // This ArgTopK inversion fixture uses distinct per-row logits. Stable tied routing is covered by
    // `deterministic_moe_routes_match_stable_sort_table` below.
    let logits: Vec<f32> = fill(l * e, 7)
        .iter()
        .enumerate()
        .map(|(i, &v)| v + (i % e) as f32 * 1e-4)
        .collect();
    let row0_logits = logits[0..e].to_vec(); // kept for the direction sanity check below

    // rank_i = #{j : logit_j > logit_i}, batched over L rows (the pairwise-Ge composition `moe_sparse`/`top_k_gate` use).
    // p[l,i,j] = logit[l,i], q[l,i,j] = logit[l,j]; gt = 1 - (p>=q); rank[l,i] = sum_j gt[l,i,j].
    let br = Builder::new();
    let lg = br.constant("logits", TensorType::f32(vec![l, e]));
    let p = br.broadcast(br.reshape(lg, vec![l, e, 1]), vec![l, e, e]);
    let qsrc = br.reshape(lg, vec![l, 1, e]);
    let q = br.broadcast(qsrc, vec![l, e, e]);
    let ge = br.binary(BinOp::Ge, p, q);
    let gt = br.binary_scalar(BinOp::Mul, ge, poot_graph_ir::types::Scalar::F32(-1.0));
    let gt = br.binary_scalar(BinOp::Add, gt, poot_graph_ir::types::Scalar::F32(1.0));
    let rank = br.reduce(RedOp::Sum, gt, 2, false); // [L, E]
    let lgi = lg.id;
    let gr = br.finish(rank);
    let mut rank_inputs = HashMap::new();
    rank_inputs.insert(lgi, Value::from(HostTensor::f32(vec![l, e], logits)));
    let rank_t = eval(&gr, &rank_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("moe_gating tests evaluate dense graphs");
    assert_eq!(rank_t.shape(), vec![l, e]);

    // Feed the concrete rank tensor into ArgTopK (its input contract is the rank tensor, not logits).
    let ba = Builder::new();
    let rk = ba.constant("rank", TensorType::f32(vec![l, e]));
    let out = ba.arg_top_k(rk, k);
    assert_eq!(
        ba.aval(out).shape,
        vec![l, k],
        "ArgTopK shape rule: [..,E] -> [..,k]"
    );
    assert_eq!(
        ba.aval(out).dtype,
        DType::F32,
        "ArgTopK ids are F32 (fork c)"
    );
    let rki = rk.id;
    let ga = ba.finish(out);
    assert!(
        matches!(ga.eqns.last().unwrap().op, OpKind::ArgTopK { k: kk } if kk == k),
        "the traced graph carries an ArgTopK eqn"
    );
    let mut a_inputs = HashMap::new();
    a_inputs.insert(rki, Value::from(rank_t.clone()));
    let got = eval(&ga, &a_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("moe_gating tests evaluate dense graphs");
    assert_eq!(got.shape(), vec![l, k]);

    // Reference: per row, `scatter(iota, rank)` (the same `scatter_axis0` the `Scatter` op runs), then the `[0..k]` slice on
    // the last axis (FR-002). Index-exact, no tolerance.
    for row in 0..l {
        let rank_row = HostTensor::f32(
            vec![e],
            rank_t.as_f32().unwrap()[row * e..(row + 1) * e].to_vec(),
        );
        let iota_row = HostTensor::f32(vec![e], (0..e).map(|j| j as f32).collect());
        let inverted = scatter_axis0(&iota_row, &rank_row, 0).unwrap();
        let expected = &inverted.as_f32().unwrap()[0..k];
        let actual = &got.as_f32().unwrap()[row * k..(row + 1) * k];
        assert_eq!(
            actual, expected,
            "row {row}: ArgTopK must equal scatter(iota,rank)[0..k] exactly"
        );
    }

    // Direction sanity (rank 0 = highest-logit expert): the expert at ArgTopK's slot 0 in row 0 must be the argmax of the
    // original logits, confirming the logits -> rank -> ArgTopK pipeline points the right way.
    let argmax = row0_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let row0_rank = &rank_t.as_f32().unwrap()[0..e];
    assert_eq!(
        row0_rank[argmax].round() as usize,
        0,
        "the highest-logit expert has rank 0"
    );
    assert_eq!(
        got.as_f32().unwrap()[0] as usize,
        argmax,
        "ArgTopK[.,0] is the best (rank-0, highest-logit) expert"
    );
}

#[test]
fn ge_indicator_with_broadcast() {
    // The `Ge` primitive (card 061): `a >= b -> 1.0 else 0.0`, `b` broadcast over rows. Exact tie `a == b` gives 1.0.
    // All other comparisons derive from this op.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![2, 3]));
    let y = b.constant("y", TensorType::f32(vec![3]));
    let g = b.binary(BinOp::Ge, x, y);
    let (xi, yi) = (x.id, y.id);
    let g = b.finish(g);

    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        Value::from(HostTensor::f32(
            vec![2, 3],
            vec![0.0, 1.0, 2.0, 3.0, 1.0, 0.5],
        )),
    );
    inputs.insert(
        yi,
        Value::from(HostTensor::f32(vec![3], vec![1.0, 1.0, 1.0])),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("moe_gating tests evaluate dense graphs");
    // row 0: [0>=1, 1>=1, 2>=1] = [0,1,1]; row 1: [3>=1, 1>=1, 0.5>=1] = [1,1,0].
    assert_close_rel(got.as_f32().unwrap(), &[0.0, 1.0, 1.0, 1.0, 1.0, 0.0], 1e-9);
}

#[test]
fn top_k_gate_selects_topk_and_normalizes() {
    // Two rows of 4 expert logits; k=2 keeps the two largest per row, softmaxes over them, zeros the rest.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![2, 4]));
    let g = top_k_gate(&b, x, 2);
    let xi = x.id;
    let g = b.finish(g);

    let xd = vec![1.0, 3.0, 2.0, 0.0, 5.0, 5.0, 1.0, 4.0];
    let mut inputs = HashMap::new();
    inputs.insert(xi, Value::from(HostTensor::f32(vec![2, 4], xd)));

    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("moe_gating tests evaluate dense graphs");

    // row 0: top-2 are indices 1 (3.0) and 2 (2.0); softmax over {3,2}; 0 and 3 are zero.
    let e3 = 1.0f32;
    let e2 = (-1.0f32).exp();
    let (w1, w2) = (e3 / (e3 + e2), e2 / (e3 + e2));
    assert_close_rel(&got.as_f32().unwrap()[0..4], &[0.0, w1, w2, 0.0], 1e-6);
    // row 1: top-2 are indices 0 and 1 (both 5.0); the stable lower-index tie-break keeps exactly {0,1}.
    assert_close_rel(&got.as_f32().unwrap()[4..8], &[0.5, 0.5, 0.0, 0.0], 1e-6);
}

#[test]
fn top_k_gate_composition_matches_sort_reference() {
    // card 061/308: the stable-rank + selected-only gate composition matches the sort-based reference oracle
    // (`crate::top_k_gate`) on distinct logits: 6 rows, E=8, k=3.
    let (rows, e, k) = (6usize, 8usize, 3usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, e]));
    let xi = x.id;
    let gate = top_k_gate(&b, x, k);
    let g = b.finish(gate);

    // distinct pseudo-random logits (a cheap hash, scaled), no ties.
    let xd: Vec<f32> = (0..rows * e)
        .map(|i| (((i * 2654435761) % 997) as f32) / 113.0 - 4.0)
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, Value::from(HostTensor::f32(vec![rows, e], xd.clone())));

    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("moe_gating tests evaluate dense graphs");

    let want = crate::top_k_gate(&HostTensor::f32(vec![rows, e], xd), k);
    assert_close_rel(got.as_f32().unwrap(), want.as_f32().unwrap(), 1e-6);
    // each row: exactly k nonzero weights, summing to 1.
    for r in 0..rows {
        let row = &got.as_f32().unwrap()[r * e..(r + 1) * e];
        assert_eq!(
            row.iter().filter(|&&w| w > 0.0).count(),
            k,
            "row {r} keeps k"
        );
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row {r} sums to 1: {sum}");
    }
}

#[test]
fn moe_matches_reference() {
    // A tiny MoE (E=3, top_k=2, hidden=2, inter=2) on one token vs an independent reference. Tests the dense form directly
    // (`moe` dispatches L=1 to the sparse path, covered by moe_sparse_matches_dense).
    use poot_graph_ir::ops::moe_dense;
    let (e, k, h, i, l) = (3usize, 2usize, 2usize, 2usize, 1usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, l, h]));
    let router = b.constant("router", TensorType::f32(vec![h, e]));
    let w_in = b.constant("w_in", TensorType::f32(vec![e, h, 2 * i]));
    let w_out = b.constant("w_out", TensorType::f32(vec![e, i, h]));
    let out = moe_dense(&b, x, router, w_in, w_out, e, k, i);
    let (xi, ri, wii, woi) = (x.id, router.id, w_in.id, w_out.id);
    let g = b.finish(out);

    let xd = fill(l * h, 1);
    let rd = fill(h * e, 2);
    let wid = fill(e * h * 2 * i, 3);
    let wod = fill(e * i * h, 4);
    let mut inputs = HashMap::new();
    inputs.insert(xi, Value::from(HostTensor::f32(vec![1, l, h], xd.clone())));
    inputs.insert(ri, Value::from(HostTensor::f32(vec![h, e], rd.clone())));
    inputs.insert(
        wii,
        Value::from(HostTensor::f32(vec![e, h, 2 * i], wid.clone())),
    );
    inputs.insert(
        woi,
        Value::from(HostTensor::f32(vec![e, i, h], wod.clone())),
    );

    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("moe_gating tests evaluate dense graphs");

    // reference (single token): router logits, top-k softmax, per-expert swiglu, weighted sum.
    let mut logits = vec![0.0f32; e];
    for (ei, lg) in logits.iter_mut().enumerate() {
        *lg = (0..h).map(|hi| xd[hi] * rd[hi * e + ei]).sum();
    }
    let mut order: Vec<usize> = (0..e).collect();
    order.sort_by(|&a, &c| logits[c].partial_cmp(&logits[a]).unwrap());
    let top = &order[..k];
    let m = top
        .iter()
        .map(|&t| logits[t])
        .fold(f32::NEG_INFINITY, f32::max);
    let denom: f32 = top.iter().map(|&t| (logits[t] - m).exp()).sum();
    let mut want = vec![0.0f32; h];
    let silu = |v: f32| v / (1.0 + (-v).exp());
    for &ex in top {
        let gate = (logits[ex] - m).exp() / denom;
        // gu = x @ w_in[ex]  ([2i]); split gate||up; act = silu(gate)*up ([i]); out = act @ w_out[ex] ([h]).
        let mut gu = vec![0.0f32; 2 * i];
        for (o, guo) in gu.iter_mut().enumerate() {
            *guo = (0..h)
                .map(|hi| xd[hi] * wid[ex * h * 2 * i + hi * 2 * i + o])
                .sum();
        }
        let act: Vec<f32> = (0..i).map(|ii| silu(gu[ii]) * gu[i + ii]).collect();
        for (hi, wv) in want.iter_mut().enumerate() {
            let o: f32 = (0..i)
                .map(|ii| act[ii] * wod[ex * i * h + ii * h + hi])
                .sum();
            *wv += gate * o;
        }
    }
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}

#[test]
fn moe_sparse_matches_dense() {
    // card 034 slice 2: the sparse decode MoE (only the k selected experts) must match the dense `moe` (all E experts,
    // zero outside the top-k) for one token. Distinct router logits; ties are covered by the deterministic table below.
    use poot_graph_ir::ops::{moe_dense, moe_sparse};
    let (e, k, h, i) = (5usize, 2usize, 4usize, 3usize);
    // shared weights, bound by const name into both graphs.
    let mut data: HashMap<String, Vec<f32>> = HashMap::new();
    data.insert("x".into(), fill(h, 1));
    data.insert("router".into(), fill(h * e, 2));
    data.insert("w_in".into(), fill(e * h * 2 * i, 3));
    data.insert("w_out".into(), fill(e * i * h, 4));

    let bind_by_name = |g: &poot_graph_ir::Graph| -> HashMap<ValueId, Value> {
        let mut inputs = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let name = meta.name.as_deref().unwrap();
            inputs.insert(
                id,
                Value::from(HostTensor::f32(meta.aval.shape.clone(), data[name].clone())),
            );
        }
        inputs
    };

    let bd = Builder::new();
    let xd = bd.constant("x", TensorType::f32(vec![1, 1, h]));
    let rd = bd.constant("router", TensorType::f32(vec![h, e]));
    let wid = bd.constant("w_in", TensorType::f32(vec![e, h, 2 * i]));
    let wod = bd.constant("w_out", TensorType::f32(vec![e, i, h]));
    let dense_out = moe_dense(&bd, xd, rd, wid, wod, e, k, i);
    let gd = bd.finish(dense_out);
    let dense = eval(
        &gd,
        &bind_by_name(&gd),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .expect("moe_gating tests evaluate dense graphs");

    let bs = Builder::new();
    let xs = bs.constant("x", TensorType::f32(vec![1, 1, h]));
    let rs = bs.constant("router", TensorType::f32(vec![h, e]));
    let wis = bs.constant("w_in", TensorType::f32(vec![e, h, 2 * i]));
    let wos = bs.constant("w_out", TensorType::f32(vec![e, i, h]));
    let sparse_out = moe_sparse(&bs, xs, rs, wis, wos, e, k, i);
    let gs = bs.finish(sparse_out);
    let sparse = eval(
        &gs,
        &bind_by_name(&gs),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .expect("moe_gating tests evaluate dense graphs");

    assert_eq!(sparse.shape(), dense.shape(), "[1,1,H]");
    assert_close_rel(sparse.as_f32().unwrap(), dense.as_f32().unwrap(), 1e-5);
}

#[test]
fn moe_grouped_matches_dense() {
    // spec 136 P3: the grouped prefill MoE (`M = L*k` flattened rows through the gather-free `indexed_matmul`) must match
    // the dense `moe` for L>1 tokens, as `moe_sparse_matches_dense` does for L=1. E=8, k=2, L=4 exercises the leading-dim
    // batching (rank[L,E], ArgTopK[L,k], gate-selection one-hot, flatten to M=L*k). Distinct router logits; the
    // deterministic table below covers tied routes.
    use poot_graph_ir::op::OpKind;
    use poot_graph_ir::ops::{moe_dense, moe_grouped};
    let (l, e, k, h, i) = (4usize, 8usize, 2usize, 4usize, 3usize);

    // shared weights, bound by const name into both graphs (same convention as moe_sparse_matches_dense).
    let mut data: HashMap<String, Vec<f32>> = HashMap::new();
    data.insert("x".into(), fill(l * h, 1));
    data.insert("router".into(), fill(h * e, 2));
    data.insert("w_in".into(), fill(e * h * 2 * i, 3));
    data.insert("w_out".into(), fill(e * i * h, 4));

    let bind_by_name = |g: &poot_graph_ir::Graph| -> HashMap<ValueId, Value> {
        let mut inputs = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let name = meta.name.as_deref().unwrap();
            inputs.insert(
                id,
                Value::from(HostTensor::f32(meta.aval.shape.clone(), data[name].clone())),
            );
        }
        inputs
    };

    let bd = Builder::new();
    let xd = bd.constant("x", TensorType::f32(vec![1, l, h]));
    let rd = bd.constant("router", TensorType::f32(vec![h, e]));
    let wid = bd.constant("w_in", TensorType::f32(vec![e, h, 2 * i]));
    let wod = bd.constant("w_out", TensorType::f32(vec![e, i, h]));
    let dense_out = moe_dense(&bd, xd, rd, wid, wod, e, k, i);
    let gd = bd.finish(dense_out);
    let dense = eval(
        &gd,
        &bind_by_name(&gd),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .expect("moe_gating tests evaluate dense graphs");

    let bg = Builder::new();
    let xg = bg.constant("x", TensorType::f32(vec![1, l, h]));
    let rg = bg.constant("router", TensorType::f32(vec![h, e]));
    let wig = bg.constant("w_in", TensorType::f32(vec![e, h, 2 * i]));
    let wog = bg.constant("w_out", TensorType::f32(vec![e, i, h]));
    let grouped_out = moe_grouped(&bg, xg, rg, wig, wog, e, k, i);
    let gg = bg.finish(grouped_out);
    let grouped = eval(
        &gg,
        &bind_by_name(&gg),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .expect("moe_gating tests evaluate dense graphs");

    assert_eq!(grouped.shape(), dense.shape(), "[1,L,H]");
    // the summation order differs (dense reduces over E, grouped over k), so use a tolerance, not assert_eq.
    assert_close_rel(grouped.as_f32().unwrap(), dense.as_f32().unwrap(), 1e-5);

    // SC-004 (structural): the IndexedMatMul rows equal M = L*k, not L*E (not a disguised dense pass).
    let mut checked = 0;
    for eqn in &gg.eqns {
        if matches!(eqn.op, OpKind::IndexedMatMul) {
            let x_id = match eqn.inputs[0] {
                poot_graph_ir::Operand::Value(id) => id,
                _ => panic!("IndexedMatMul's x operand must be a value"),
            };
            let m = gg.aval(x_id).shape[0];
            assert_eq!(m, l * k, "IndexedMatMul M dim must be L*k, not L*E");
            checked += 1;
        }
    }
    assert_eq!(
        checked, 2,
        "moe_grouped emits exactly 2 IndexedMatMul eqns (w_in, w_out)"
    );
}

#[derive(Clone, Copy, Debug)]
enum RouteKind {
    Dense,
    Sparse,
    Grouped,
}

fn eval_tied_route(route: RouteKind, scores: &[f32], k: usize, w_in: &[f32], w_out: &[f32]) -> f32 {
    use poot_graph_ir::ops::{moe_dense, moe_grouped, moe_sparse};

    let e = scores.len();
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1]));
    let router = b.constant("router", TensorType::f32(vec![1, e]));
    let mut named = HashMap::from([
        ("x".to_string(), HostTensor::f32(vec![1, 1, 1], vec![1.0])),
        (
            "router".to_string(),
            HostTensor::f32(vec![1, e], scores.to_vec()),
        ),
    ]);

    let wi = b.constant("w_in", TensorType::f32(vec![e, 1, 2]));
    let wo = b.constant("w_out", TensorType::f32(vec![e, 1, 1]));
    named.insert(
        "w_in".to_string(),
        HostTensor::f32(vec![e, 1, 2], w_in.to_vec()),
    );
    named.insert(
        "w_out".to_string(),
        HostTensor::f32(vec![e, 1, 1], w_out.to_vec()),
    );
    let out = match route {
        RouteKind::Dense => moe_dense(&b, x, router, wi, wo, e, k, 1),
        RouteKind::Sparse => moe_sparse(&b, x, router, wi, wo, e, k, 1),
        RouteKind::Grouped => moe_grouped(&b, x, router, wi, wo, e, k, 1),
    };
    let graph = b.finish(out);
    let inputs = graph
        .inputs
        .iter()
        .map(|&id| {
            let name = graph.meta(id).name.as_deref().expect("named fixture input");
            (id, Value::from(named[name].clone()))
        })
        .collect();
    eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("moe_gating tests evaluate dense graphs")
        .as_f32()
        .unwrap()[0]
}

#[test]
fn deterministic_moe_routes_match_stable_sort_table() {
    struct Case {
        name: &'static str,
        scores: [f32; 5],
        k: usize,
        ids: &'static [usize],
    }

    let cases = [
        Case {
            name: "all equal",
            scores: [1.0; 5],
            k: 3,
            ids: &[0, 1, 2],
        },
        Case {
            name: "boundary tie",
            scores: [5.0, 4.0, 4.0, 4.0, 1.0],
            k: 2,
            ids: &[0, 1],
        },
        Case {
            name: "repeated negative",
            scores: [-2.0, -3.0, -2.0, -5.0, -3.0],
            k: 3,
            ids: &[0, 2, 1],
        },
        Case {
            // Frozen deterministic-RNG sample; explicit values keep the expected order independent of graph-adjacent test code.
            name: "fixed-seed random untied",
            scores: [0.71958816, 0.28636825, 0.4453932, -0.65204656, -0.63948715],
            k: 2,
            ids: &[0, 2],
        },
        Case {
            name: "k one",
            scores: [1.0, 9.0, 2.0, 9.0, 0.0],
            k: 1,
            ids: &[1],
        },
        Case {
            name: "k equals experts",
            scores: [f32::MAX, f32::MIN, 3.0, -0.0, f32::MIN],
            k: 5,
            ids: &[0, 2, 3, 1, 4],
        },
        Case {
            name: "NaN ranks below finite",
            scores: [f32::NAN, -3.0, -3.0, 2.0, -4.0],
            k: 3,
            ids: &[3, 1, 2],
        },
        Case {
            name: "all NaN",
            scores: [f32::NAN; 5],
            k: 2,
            ids: &[0, 1],
        },
        Case {
            name: "positive infinity tie",
            scores: [f32::INFINITY, 3.0, f32::INFINITY, 2.0, 1.0],
            k: 2,
            ids: &[0, 2],
        },
        Case {
            name: "negative infinity selected",
            scores: [
                4.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                3.0,
                f32::NEG_INFINITY,
            ],
            k: 4,
            ids: &[0, 3, 1, 2],
        },
        Case {
            name: "canonical endpoints",
            scores: [f32::MAX, f32::MIN, 0.0, f32::MIN, f32::MAX],
            k: 4,
            ids: &[0, 4, 2, 1],
        },
        Case {
            name: "signed zero tie",
            scores: [-0.0, 0.0, -0.0, 0.0, -1.0],
            k: 3,
            ids: &[0, 1, 2],
        },
    ];

    // Every expert computes a different scalar FFN output, so a wrong selection changes the final result.
    let w_in: Vec<f32> = (0..5)
        .flat_map(|expert| [0.25 + expert as f32 * 0.1, 1.0 + expert as f32 * 0.2])
        .collect();
    let w_out: Vec<f32> = (0..5).map(|expert| 0.7 + expert as f32 * 0.3).collect();
    let expert_out: Vec<f32> = (0..5)
        .map(|expert| {
            let gate = w_in[expert * 2];
            let up = w_in[expert * 2 + 1];
            (gate / (1.0 + (-gate).exp())) * up * w_out[expert]
        })
        .collect();

    for case in cases {
        let score_tensor = HostTensor::f32(vec![1, 5], case.scores.to_vec());
        let oracle_ids = crate::top_k_ids(&score_tensor, case.k);
        let oracle_mask = crate::top_k_mask(&score_tensor, case.k);
        let oracle = crate::top_k_gate(&score_tensor, case.k);
        let selected: Vec<usize> = oracle_ids
            .as_f32()
            .unwrap()
            .iter()
            .map(|&id| id as usize)
            .collect();
        assert_eq!(selected, case.ids, "{}: independent oracle ids", case.name);
        let selected_from_mask: Vec<usize> = oracle_mask
            .as_f32()
            .unwrap()
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| (selected == 1.0).then_some(i))
            .collect();
        let mut expected_set = case.ids.to_vec();
        expected_set.sort_unstable();
        assert_eq!(
            selected_from_mask, expected_set,
            "{}: independent oracle mask",
            case.name
        );
        assert_eq!(selected.len(), case.k, "{}: exactly k ids", case.name);
        assert_eq!(
            oracle_mask
                .as_f32()
                .unwrap()
                .iter()
                .filter(|&&v| v == 1.0)
                .count(),
            case.k,
            "{}: exactly k selected mask entries",
            case.name
        );
        for (id, (&mask, &weight)) in oracle_mask
            .as_f32()
            .unwrap()
            .iter()
            .zip(oracle.as_f32().unwrap().iter())
            .enumerate()
        {
            if mask == 0.0 {
                assert_eq!(weight, 0.0, "{}: unselected expert {id}", case.name);
            }
        }
        let sum: f32 = oracle.as_f32().unwrap().iter().sum();
        assert!((sum - 1.0).abs() <= 1e-6, "{}: gate sum {sum}", case.name);

        let b = Builder::new();
        let scores = b.constant("scores", TensorType::f32(vec![1, 5]));
        let rank = poot_graph_ir::ops::stable_descending_rank(&b, scores);
        let mask = poot_graph_ir::ops::top_k_keep_mask(&b, rank, case.k);
        let gate = top_k_gate(&b, scores, case.k);
        let route = b.concat(1, &[mask, gate]);
        let graph = b.finish(route);
        let inputs = graph
            .inputs
            .iter()
            .copied()
            .map(|id| {
                let name = graph.meta(id).name.as_deref().expect("named gate input");
                let tensor = if name == "scores" {
                    HostTensor::f32(vec![1, 5], case.scores.to_vec())
                } else {
                    panic!("unexpected gate input {name}")
                };
                (id, Value::from(tensor))
            })
            .collect();
        let got_gate = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("moe_gating tests evaluate dense graphs");
        assert_close_rel(
            &got_gate.as_f32().unwrap()[..5],
            oracle_mask.as_f32().unwrap(),
            0.0,
        );
        assert_close_rel(
            &got_gate.as_f32().unwrap()[5..],
            oracle.as_f32().unwrap(),
            1e-6,
        );
        assert_eq!(
            got_gate.as_f32().unwrap()[..5]
                .iter()
                .filter(|&&v| v == 1.0)
                .count(),
            case.k,
            "{}: graph selects exactly k mask entries",
            case.name
        );
        if case.name == "canonical endpoints" {
            assert_eq!(
                oracle_mask.as_f32().unwrap()[1],
                1.0,
                "endpoint expert is selected"
            );
            assert_eq!(
                oracle.as_f32().unwrap()[1],
                0.0,
                "selected endpoint weight underflows"
            );
            assert_eq!(
                got_gate.as_f32().unwrap()[1],
                1.0,
                "graph endpoint mask remains selected"
            );
            assert_eq!(
                got_gate.as_f32().unwrap()[6],
                0.0,
                "graph endpoint weight underflows"
            );
        }

        let b = Builder::new();
        let xm = b.constant("x", TensorType::f32(vec![1, 1]));
        let logits = b.constant("scores", TensorType::f32(vec![1, 5]));
        let (_, ids, weights) = poot_graph_ir::ops::moe_grouped_prep(&b, xm, logits, 5, case.k);
        let route = b.concat(0, &[ids, weights]);
        let (xm_id, logits_id) = (xm.id, logits.id);
        let graph = b.finish(route);
        let inputs: HashMap<_, _> = graph
            .inputs
            .iter()
            .copied()
            .map(|id| {
                let tensor = if id == xm_id {
                    HostTensor::f32(vec![1, 1], vec![1.0])
                } else if id == logits_id {
                    HostTensor::f32(vec![1, 5], case.scores.to_vec())
                } else {
                    panic!("unexpected grouped prep input {id}")
                };
                (id, Value::from(tensor))
            })
            .collect();
        let got_route = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("moe_gating tests evaluate dense graphs");
        let got_ids: Vec<usize> = got_route.as_f32().unwrap()[..case.k]
            .iter()
            .map(|&id| id as usize)
            .collect();
        assert_eq!(got_ids, case.ids, "{}: stable graph ids", case.name);
        let mut distinct = got_ids.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), case.k, "{}: ids are distinct", case.name);
        let expected_weights: Vec<f32> = case
            .ids
            .iter()
            .map(|&id| oracle.as_f32().unwrap()[id])
            .collect();
        assert_close_rel(
            &got_route.as_f32().unwrap()[case.k..],
            &expected_weights,
            1e-6,
        );

        let want: f32 = oracle
            .as_f32()
            .unwrap()
            .iter()
            .zip(&expert_out)
            .map(|(weight, output)| weight * output)
            .sum();
        for route in [RouteKind::Dense, RouteKind::Sparse, RouteKind::Grouped] {
            let got = eval_tied_route(route, &case.scores, case.k, &w_in, &w_out);
            assert!(
                (got - want).abs() <= 1e-6,
                "{} {route:?}: {got} vs independent {want}",
                case.name
            );
        }
    }

    for invalid_k in [0, 6] {
        let result = std::panic::catch_unwind(|| {
            let b = Builder::new();
            let scores = b.constant("scores", TensorType::f32(vec![1, 5]));
            let _ = top_k_gate(&b, scores, invalid_k);
        });
        assert!(
            result.is_err(),
            "k={invalid_k} must fail graph construction"
        );
    }
}
