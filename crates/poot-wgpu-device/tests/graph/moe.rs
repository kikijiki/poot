use super::*;

#[test]
fn moe_sparse_gpu_matches_dense() {
    // The sparse decode MoE (stable rank + ArgTopK + k expert FFNs) lowers on the GPU and matches the dense `moe`: stable rank,
    // rank inversion, gather-by-index and k FFNs are all GPU-native.
    use poot_graph_ir::ops::{moe_dense, moe_sparse};
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (e, k, h, i) = (5usize, 2usize, 4usize, 3usize);
    let fill = |n: usize, seed: u64| -> Vec<f32> {
        (0..n)
            .map(|j| (((j as u64 * 2654435761 + seed * 40503) % 1009) as f32) / 137.0 - 3.5)
            .collect()
    };
    let mut data: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    data.insert("x".into(), fill(h, 1));
    data.insert("router".into(), fill(h * e, 2));
    data.insert("w_in".into(), fill(e * h * 2 * i, 3));
    data.insert("w_out".into(), fill(e * i * h, 4));
    let bind = |g: &poot_graph_ir::Graph| -> HashMap<poot_graph_ir::ValueId, HostTensor> {
        let mut inputs = HashMap::new();
        for &id in g.inputs() {
            let meta = g.meta(id);
            let name = meta.name.as_deref().unwrap();
            inputs.insert(
                id,
                HostTensor::f32(meta.aval.shape.clone(), data[name].clone()),
            );
        }
        inputs
    };

    let bd = Builder::for_tests();
    let (xd, rd) = (
        bd.constant("x", TensorType::f32(vec![1, 1, h])).unwrap(),
        bd.constant("router", TensorType::f32(vec![h, e])).unwrap(),
    );
    let wid = bd
        .constant("w_in", TensorType::f32(vec![e, h, 2 * i]))
        .unwrap();
    let wod = bd
        .constant("w_out", TensorType::f32(vec![e, i, h]))
        .unwrap();
    let dout = moe_dense(&bd, xd, rd, wid, wod, e, k, i).unwrap();
    let gd = bd.finish(dout).unwrap();
    let dense = run_via_contract(&mut gpu, target, &gd, &bind(&gd));

    let bs = Builder::for_tests();
    let (xs, rs) = (
        bs.constant("x", TensorType::f32(vec![1, 1, h])).unwrap(),
        bs.constant("router", TensorType::f32(vec![h, e])).unwrap(),
    );
    let wis = bs
        .constant("w_in", TensorType::f32(vec![e, h, 2 * i]))
        .unwrap();
    let wos = bs
        .constant("w_out", TensorType::f32(vec![e, i, h]))
        .unwrap();
    let sout = moe_sparse(&bs, xs, rs, wis, wos, e, k, i).unwrap();
    let gs = bs.finish(sout).unwrap();
    let sparse = run_via_contract(&mut gpu, target, &gs, &bind(&gs));

    assert_eq!(sparse.shape(), dense.shape());
    for (j, (a, b)) in sparse
        .as_f32()
        .unwrap()
        .iter()
        .zip(dense.as_f32().unwrap().iter())
        .enumerate()
    {
        let tol = 1e-4 * b.abs().max(1e-3);
        assert!(
            (a - b).abs() <= tol,
            "sparse != dense MoE on GPU at {j}: {a} vs {b}"
        );
    }
}

/// The decode sparse MoE (`k` rows, each routed to its own expert) against the CPU oracle at tier 2 over a
/// model-shaped graph whose experts are well separated: expert `e`'s weights are scaled by `e + 1` and carry their
/// own pattern, so a row that reads another row's expert moves the logits by far more than the tolerance.
///
/// Mutation (`emit_row_expert`): read `ids[0]` for every row; the second and third selected experts are replaced by
/// the first and the output goes red.
#[test]
fn moe_sparse_decode_with_separated_experts_matches_the_oracle() {
    use poot_graph_ir::ops::moe_sparse;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (e, k, h, i) = (8usize, 3usize, 16usize, 12usize);
    let fill = |n: usize, seed: u64| -> Vec<f32> {
        (0..n)
            .map(|j| (((j as u64 * 2654435761 + seed * 40503) % 1009) as f32) / 505.0 - 1.0)
            .collect()
    };
    let scaled = |per_expert: usize, seed: u64| -> Vec<f32> {
        (0..e)
            .flat_map(|expert| {
                fill(per_expert, seed + expert as u64 * 7)
                    .into_iter()
                    .map(move |v| v * (expert + 1) as f32)
            })
            .collect()
    };
    let mut data: HashMap<String, Vec<f32>> = HashMap::new();
    data.insert("x".into(), fill(h, 1));
    data.insert("router".into(), fill(h * e, 2));
    data.insert("w_in".into(), scaled(h * 2 * i, 3));
    data.insert("w_out".into(), scaled(i * h, 4));
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 1, h])).unwrap();
    let router = b.constant("router", TensorType::f32(vec![h, e])).unwrap();
    let w_in = b
        .constant("w_in", TensorType::f32(vec![e, h, 2 * i]))
        .unwrap();
    let w_out = b.constant("w_out", TensorType::f32(vec![e, i, h])).unwrap();
    let out = moe_sparse(&b, x, router, w_in, w_out, e, k, i).unwrap();
    let g = b.finish(out).unwrap();
    let inputs: HashMap<_, HostTensor> = g
        .inputs()
        .iter()
        .map(|&id| {
            let meta = g.meta(id);
            let name = meta.name.as_deref().unwrap();
            (
                id,
                HostTensor::f32(meta.aval.shape.clone(), data[name].clone()),
            )
        })
        .collect();
    let eval_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let want = eval(&g, &eval_inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_via_contract(&mut gpu, target, &g, &inputs);
    assert_eq!(got.shape(), want.shape());
    let scale = want
        .as_f32()
        .unwrap()
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(scale > 1.0, "the fixture must produce a signal: {scale}");
    for (j, (a, b)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(want.as_f32().unwrap())
        .enumerate()
    {
        assert!(
            (a - b).abs() <= 1e-4 * scale,
            "decode MoE on the device differs from the oracle at {j}: {a} vs {b}"
        );
    }
}

#[test]
fn moe_grouped_gpu_matches_cpu() {
    // The grouped prefill MoE (stable batched rank, ArgTopK, exact selected-only gate, B1 gate-selection one-hot, flatten to
    // M=L*k, gather-free `indexed_matmul` FFN twice, transpose-to-last-axis reduce over k) lowers on the GPU and matches the
    // CPU eval oracle for a real prefill batch (L=4, so routing is batched over tokens, unlike the L=1 form
    // `moe_sparse_gpu_matches_dense` covers). Executor equivalence; the CPU-only decomposition check is
    // `moe_grouped_matches_dense` in poot-eval.
    use poot_graph_ir::op::OpKind;
    use poot_graph_ir::ops::moe_grouped;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (l, e, k, h, i) = (4usize, 8usize, 2usize, 16usize, 8usize);
    let fill = |n: usize, seed: u64| -> Vec<f32> {
        (0..n)
            .map(|j| (((j as u64 * 2654435761 + seed * 40503) % 1009) as f32) / 137.0 - 3.5)
            .collect()
    };
    let mut data: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    data.insert("x".into(), fill(l * h, 1));
    data.insert("router".into(), fill(h * e, 2));
    data.insert("w_in".into(), fill(e * h * 2 * i, 3));
    data.insert("w_out".into(), fill(e * i * h, 4));
    let bind = |g: &poot_graph_ir::Graph| -> HashMap<poot_graph_ir::ValueId, HostTensor> {
        let mut inputs = HashMap::new();
        for &id in g.inputs() {
            let meta = g.meta(id);
            let name = meta.name.as_deref().unwrap();
            inputs.insert(
                id,
                HostTensor::f32(meta.aval.shape.clone(), data[name].clone()),
            );
        }
        inputs
    };

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, l, h])).unwrap();
    let router = b.constant("router", TensorType::f32(vec![h, e])).unwrap();
    let w_in = b
        .constant("w_in", TensorType::f32(vec![e, h, 2 * i]))
        .unwrap();
    let w_out = b.constant("w_out", TensorType::f32(vec![e, i, h])).unwrap();
    let out = moe_grouped(&b, x, router, w_in, w_out, e, k, i).unwrap();
    let g = b.finish(out).unwrap();

    // SC-004 (structural, on the graph run below): IndexedMatMul's M dim must be L*k, not L*E.
    let mut checked = 0;
    for eqn in g.eqns() {
        if matches!(eqn.op, OpKind::IndexedMatMul) {
            let x_id = match eqn.inputs[0] {
                poot_graph_ir::Operand::Value(id) => id,
                _ => panic!("IndexedMatMul's x operand must be a value"),
            };
            let m = g.aval(x_id).shape[0];
            assert_eq!(m, l * k, "IndexedMatMul M dim must be L*k, not L*E");
            checked += 1;
        }
    }
    assert_eq!(checked, 2, "moe_grouped emits exactly 2 IndexedMatMul eqns");

    let inputs = bind(&g);
    let eval_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_via_contract(&mut gpu, target, &g, &inputs);
    assert_eq!(got.shape(), vec![1, l, h]);
    assert_eq!(got.shape(), cpu.shape());
    for (j, (a, b)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        let tol = 1e-4 * b.abs().max(1e-3);
        assert!(
            (a - b).abs() <= tol,
            "moe_grouped GPU != CPU at {j}: {a} vs {b}"
        );
    }
}

#[test]
fn stable_moe_ties_gpu_match_independent_oracle() {
    // Card 308 device gate: score -> stable rank -> ArgTopK + exact selected-only gate on tied and non-finite rows. Separate
    // from `arg_top_k_gpu_matches_cpu`, which isolates the rank-inversion kernel with an already-valid permutation.
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let e = 5usize;
    let cases = [
        ("all equal", [1.0; 5], 3usize),
        ("NaN low", [f32::NAN, -3.0, -3.0, 2.0, -4.0], 3),
        (
            "positive infinity",
            [f32::INFINITY, 3.0, f32::INFINITY, 2.0, 1.0],
            2,
        ),
        (
            "negative infinity",
            [4.0, f32::NEG_INFINITY, f32::NEG_INFINITY, 3.0, -5.0],
            4,
        ),
        (
            "canonical endpoints",
            [f32::MAX, f32::MIN, 0.0, f32::MIN, f32::MAX],
            4,
        ),
        ("signed zero", [-0.0, 0.0, -0.0, 0.0, -1.0], 3),
        ("k one", [1.0, 9.0, 2.0, 9.0, 0.0], 1),
        (
            "k equals experts",
            [f32::MAX, f32::MIN, 3.0, -0.0, f32::MIN],
            5,
        ),
    ];

    for (name, scores_data, k) in cases {
        let b = Builder::for_tests();
        let scores = b.constant("scores", TensorType::f32(vec![1, e])).unwrap();
        let rank = poot_graph_ir::ops::stable_descending_rank(&b, scores).unwrap();
        let ids = b.arg_top_k(rank, k).unwrap();
        let mask = poot_graph_ir::ops::top_k_keep_mask(&b, rank, k).unwrap();
        let gate = poot_graph_ir::ops::top_k_gate(&b, scores, k).unwrap();
        // The ids are I32; the concatenated probe row is F32 (ids below 2^24 widen exactly).
        let ids = b.cast(ids, poot_tensor::DType::F32).unwrap();
        let out = b.concat(1, &[ids, mask, gate]).unwrap();
        let graph = b.finish(out).unwrap();
        let inputs: HashMap<_, _> = graph
            .inputs()
            .iter()
            .copied()
            .map(|id| {
                let input_name = graph
                    .meta(id)
                    .name
                    .as_deref()
                    .expect("named tie probe input");
                let tensor = if input_name == "scores" {
                    HostTensor::f32(vec![1, e], scores_data.to_vec())
                } else {
                    panic!("unexpected tie probe input {input_name}")
                };
                (id, tensor)
            })
            .collect();
        let scores_tensor = HostTensor::f32(vec![1, e], scores_data.to_vec());
        let ids_oracle = poot_eval::top_k_ids(&scores_tensor, k);
        let mask_oracle = poot_eval::top_k_mask(&scores_tensor, k);
        let gate_oracle = poot_eval::top_k_gate(&scores_tensor, k);

        let got = run_via_contract(&mut gpu, target, &graph, &inputs);
        assert_eq!(got.shape(), vec![1, k + 2 * e], "{name}: shape");
        assert_eq!(
            &got.as_f32().unwrap()[..k],
            ids_oracle.as_f32().unwrap(),
            "{name}: selected ids"
        );
        assert_eq!(
            &got.as_f32().unwrap()[k..k + e],
            mask_oracle.as_f32().unwrap(),
            "{name}: selection mask"
        );
        assert_eq!(
            got.as_f32().unwrap()[k..k + e]
                .iter()
                .filter(|&&v| v == 1.0)
                .count(),
            k,
            "{name}: exactly k mask entries"
        );
        for (i, (&actual, &expected)) in got.as_f32().unwrap()[k + e..]
            .iter()
            .zip(gate_oracle.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(actual.is_finite(), "{name} weight {i}: non-finite");
            assert!(
                (actual - expected).abs() <= 1e-5,
                "{name} weight {i}: gpu {actual} vs oracle {expected}"
            );
            if mask_oracle.as_f32().unwrap()[i] == 0.0 {
                assert_eq!(actual, 0.0, "{name} weight {i}: unselected");
            }
        }
    }
}
