use super::*;

#[test]
#[ignore = "perf microbench: times dense vs sparse MoE on the GPU; run with --ignored --nocapture"]
fn moe_sparse_vs_dense_decode_timing() {
    // Receipt: is the sparse MoE (k of E expert FFNs) faster than dense on the Arc at granite-moe-1b's MoE dims (E=32, k=8,
    // H=1024, I=512, one token)? Sparse saves 24 experts' matmuls but adds a gather of the 8 selected experts' weight rows; on
    // a memory-bound iGPU that can offset the FLOP win, so this measures rather than asserts.
    use poot_graph_ir::ops::{moe_dense, moe_sparse};
    use std::time::Instant;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (e, k, h, i) = (32usize, 8usize, 1024usize, 512usize);
    let f = |n: usize, seed: u64| -> Vec<f32> {
        (0..n)
            .map(|j| (((j as u64 * 2654435761 + seed * 40503) % 2003) as f32) / 290.0 - 3.5)
            .collect()
    };
    let mut data: HashMap<String, Vec<f32>> = HashMap::new();
    data.insert("x".into(), f(h, 1));
    data.insert("router".into(), f(h * e, 2));
    data.insert("w_in".into(), f(e * h * 2 * i, 3));
    data.insert("w_out".into(), f(e * i * h, 4));
    let bind = |g: &poot_graph_ir::Graph| -> HashMap<poot_graph_ir::ValueId, HostTensor> {
        g.inputs
            .iter()
            .map(|&id| {
                let m = g.meta(id);
                (
                    id,
                    HostTensor::f32(
                        m.aval.shape.clone(),
                        data[m.name.as_deref().unwrap()].clone(),
                    ),
                )
            })
            .collect()
    };
    let build = |sparse: bool| {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, h]));
        let r = b.constant("router", TensorType::f32(vec![h, e]));
        let wi = b.constant("w_in", TensorType::f32(vec![e, h, 2 * i]));
        let wo = b.constant("w_out", TensorType::f32(vec![e, i, h]));
        let out = if sparse {
            moe_sparse(&b, x, r, wi, wo, e, k, i)
        } else {
            moe_dense(&b, x, r, wi, wo, e, k, i)
        };
        b.finish(out)
    };
    let gd = build(false);
    let gs = build(true);
    let (id, is) = (bind(&gd), bind(&gs));
    let iters = 40;
    // Resident entry (consts uploaded once, intermediates on-device, only the output read back), the
    // realistic decode path. Per-call `run_via_contract` is dominated by host transfers and per-call
    // compile/teardown (~64 ms/fwd at these dims), which masks the kernel difference.
    let (exe_d, entry_d) = add_resident_entry(&mut gpu, target, &gd, &id);
    let (exe_s, entry_s) = add_resident_entry(&mut gpu, target, &gs, &is);
    let d0 = step_resident(&mut gpu, exe_d, entry_d, &gd);
    let s0 = step_resident(&mut gpu, exe_s, entry_s, &gs);
    // Sparse uses the indexed LDS-GEMV (tree-ish reduction), dense sums serially, so they differ by the usual f32
    // reduction-order amount (~2e-3 relative), not bit-exactly. A loose sanity bound; the exact oracles are
    // `indexed_gemv_lds_large_matches_cpu` and `moe_sparse_gpu_matches_dense`.
    for (a, b) in s0.as_f32().unwrap().iter().zip(d0.as_f32().unwrap().iter()) {
        assert!(
            (a - b).abs() <= 1e-2 * b.abs() + 1e-2,
            "sparse {a} vs dense {b}"
        );
    }
    let t = Instant::now();
    for _ in 0..iters {
        let _ = step_resident(&mut gpu, exe_d, entry_d, &gd);
    }
    let dense_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let t = Instant::now();
    for _ in 0..iters {
        let _ = step_resident(&mut gpu, exe_s, entry_s, &gs);
    }
    let sparse_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    gpu.remove_entry(exe_d, entry_d).unwrap();
    gpu.remove_entry(exe_s, entry_s).unwrap();
    gpu.unload(exe_d).unwrap();
    gpu.unload(exe_s).unwrap();
    eprintln!(
        "MoE decode RESIDENT (E={e}, k={k}, H={h}, I={i}): dense {dense_ms:.3} ms/fwd, sparse {sparse_ms:.3} \
         ms/fwd, speedup {:.2}x",
        dense_ms / sparse_ms
    );
    // Perf-regression guard (a ratio, tolerant of this box's ~2x absolute noise): sparse decode must stay at least as fast as
    // dense (k of E experts via the indexed LDS-GEMV). A regression to the naive indexed kernel or a broken sparse dispatch
    // trips this.
    assert!(
        sparse_ms < dense_ms,
        "sparse MoE decode regressed: sparse {sparse_ms:.3} >= dense {dense_ms:.3} ms/fwd"
    );
}

#[test]
#[ignore = "perf measurement; run with --ignored --nocapture"]
fn moe_prefill_dense_timing() {
    // Baseline for MoE prefill (L>1): `moe()` uses moe_dense, whose per-expert matmuls are batched (weight rank-3 [E,H,2I]) and
    // take the naive `matmul_batched` (the tiled GEMM gate requires a rank-2 shared weight). Measures the headroom for a
    // batched tiled GEMM.
    use poot_graph_ir::ops::moe;
    use std::time::Instant;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (e, k, h, i, l) = (32usize, 8usize, 1024usize, 512usize, 64usize);
    let f = |n: usize, seed: u64| -> Vec<f32> {
        (0..n)
            .map(|j| (((j as u64 * 2654435761 + seed * 40503) % 2003) as f32) / 290.0 - 3.5)
            .collect()
    };
    let mut data: HashMap<String, Vec<f32>> = HashMap::new();
    data.insert("x".into(), f(l * h, 1));
    data.insert("router".into(), f(h * e, 2));
    data.insert("w_in".into(), f(e * h * 2 * i, 3));
    data.insert("w_out".into(), f(e * i * h, 4));
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, l, h]));
    let r = b.constant("router", TensorType::f32(vec![h, e]));
    let wi = b.constant("w_in", TensorType::f32(vec![e, h, 2 * i]));
    let wo = b.constant("w_out", TensorType::f32(vec![e, i, h]));
    let out = moe(&b, x, r, wi, wo, e, k, i); // L>1 -> moe_dense
    let g = b.finish(out);
    let inp: HashMap<poot_graph_ir::ValueId, HostTensor> = g
        .inputs
        .iter()
        .map(|&id| {
            let m = g.meta(id);
            (
                id,
                HostTensor::f32(
                    m.aval.shape.clone(),
                    data[m.name.as_deref().unwrap()].clone(),
                ),
            )
        })
        .collect();
    let iters = 20;
    let (exe, entry) = add_resident_entry(&mut gpu, target, &g, &inp);
    step_resident(&mut gpu, exe, entry, &g); // warmup
    let t = Instant::now();
    for _ in 0..iters {
        let _ = step_resident(&mut gpu, exe, entry, &g);
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    gpu.remove_entry(exe, entry).unwrap();
    gpu.unload(exe).unwrap();
    eprintln!(
        "MoE PREFILL dense RESIDENT (E={e}, k={k}, H={h}, I={i}, L={l}): {ms:.3} ms/fwd \
         (batched tiled GEMM since update 0174; was 13.58 naive)"
    );
}

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
    let bind = |g: &poot_graph_ir::Graph| -> HashMap<usize, HostTensor> {
        let mut inputs = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let name = meta.name.as_deref().unwrap();
            inputs.insert(
                id,
                HostTensor::f32(meta.aval.shape.clone(), data[name].clone()),
            );
        }
        inputs
    };

    let bd = Builder::new();
    let (xd, rd) = (
        bd.constant("x", TensorType::f32(vec![1, 1, h])),
        bd.constant("router", TensorType::f32(vec![h, e])),
    );
    let wid = bd.constant("w_in", TensorType::f32(vec![e, h, 2 * i]));
    let wod = bd.constant("w_out", TensorType::f32(vec![e, i, h]));
    let dout = moe_dense(&bd, xd, rd, wid, wod, e, k, i);
    let gd = bd.finish(dout);
    let dense = run_via_contract(&mut gpu, target, &gd, &bind(&gd));

    let bs = Builder::new();
    let (xs, rs) = (
        bs.constant("x", TensorType::f32(vec![1, 1, h])),
        bs.constant("router", TensorType::f32(vec![h, e])),
    );
    let wis = bs.constant("w_in", TensorType::f32(vec![e, h, 2 * i]));
    let wos = bs.constant("w_out", TensorType::f32(vec![e, i, h]));
    let sout = moe_sparse(&bs, xs, rs, wis, wos, e, k, i);
    let gs = bs.finish(sout);
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
    let bind = |g: &poot_graph_ir::Graph| -> HashMap<usize, HostTensor> {
        let mut inputs = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let name = meta.name.as_deref().unwrap();
            inputs.insert(
                id,
                HostTensor::f32(meta.aval.shape.clone(), data[name].clone()),
            );
        }
        inputs
    };

    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, l, h]));
    let router = b.constant("router", TensorType::f32(vec![h, e]));
    let w_in = b.constant("w_in", TensorType::f32(vec![e, h, 2 * i]));
    let w_out = b.constant("w_out", TensorType::f32(vec![e, i, h]));
    let out = moe_grouped(&b, x, router, w_in, w_out, e, k, i);
    let g = b.finish(out);

    // SC-004 (structural, on the graph run below): IndexedMatMul's M dim must be L*k, not L*E.
    let mut checked = 0;
    for eqn in &g.eqns {
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
    let cpu = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
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
        let b = Builder::new();
        let scores = b.constant("scores", TensorType::f32(vec![1, e]));
        let rank = poot_graph_ir::ops::stable_descending_rank(&b, scores);
        let ids = b.arg_top_k(rank, k);
        let mask = poot_graph_ir::ops::top_k_keep_mask(&b, rank, k);
        let gate = poot_graph_ir::ops::top_k_gate(&b, scores, k);
        let out = b.concat(1, &[ids, mask, gate]);
        let graph = b.finish(out);
        let inputs: HashMap<_, _> = graph
            .inputs
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
