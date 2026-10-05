use super::*;

#[test]
fn gpu_ut_transform_matches_cpu_c64() {
    // Block-recursive UT-transform `unit_lower_triangular_inverse` inside the GDN delta rule at the real chunk size C=64
    // (6 levels of halving 64->32->...->1). The biggest suspect: 64x64 is exactly the >64-lane LDS cap / tiled-GEMM-race
    // regime. CPU-oracle coverage: tests.rs `ut_transform_inverse_reconstructs_identity_c64`.
    use poot_graph_ir::ops::unit_lower_triangular_inverse;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let c = 64usize;
    let attn = realistic_gdn_attn_c64(c, 32, 23);

    let b = Builder::new();
    let attn_t = b.constant("attn", TensorType::f32(vec![c, c]));
    let t = unit_lower_triangular_inverse(&b, attn_t);
    let ai = attn_t.id;
    let g = b.finish(t);
    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![c, c], attn));

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
    assert_eq!(got.shape(), cpu.shape());
    assert_prefill_op_gpu_matches_cpu(
        "unit_lower_triangular_inverse",
        "T",
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        3e-3,
        1e-3,
    );
}

#[test]
fn gpu_attention_prefill_matches_cpu_l64() {
    // Batched masked full-attention prefill (`[1,1,L,L]` additive causal mask + GQA `repeat_kv`) at L=64, the per-chunk prefill
    // length of the attention layers interleaved with GDN in Qwen3-Next. Hq=4, Hkv=2 (n_rep=2), D=8. This is
    // `attention_prefill`, not the decode-only `attention_masked`.
    use poot_graph_ir::ops::attention_prefill;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, d, l) = (4usize, 2usize, 8usize, 64usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let qd: Vec<f32> = (0..hq * l * d).map(|i| (i as f32) * 0.003 - 0.4).collect();
    let kd: Vec<f32> = (0..hkv * l * d).map(|i| (i as f32) * 0.004 - 0.3).collect();
    let vd: Vec<f32> = (0..hkv * l * d).map(|i| (i as f32) * 0.005 - 0.2).collect();
    // additive causal mask [1,1,L,L]: 0 on/below the diagonal, large-negative above (the qwen3next
    // gated-attention-prefill convention).
    let mask_data: Vec<f32> = (0..l * l)
        .map(|idx| {
            let (i, j) = (idx / l, idx % l);
            if j <= i { 0.0 } else { -1.0e30 }
        })
        .collect();

    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, l, l]));
    let out = attention_prefill(&b, q, k, v, n_rep, scale, mask);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(qi, HostTensor::f32(vec![1, hq, l, d], qd));
    inputs.insert(ki, HostTensor::f32(vec![1, hkv, l, d], kd));
    inputs.insert(vi, HostTensor::f32(vec![1, hkv, l, d], vd));
    inputs.insert(mi, HostTensor::f32(vec![1, 1, l, l], mask_data));

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
    assert_eq!(got.shape(), cpu.shape());
    assert_prefill_op_gpu_matches_cpu(
        "attention_prefill",
        "out",
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        5e-3,
        1e-3,
    );
}

#[test]
fn gpu_attention_prefill_matches_cpu_past_matmul_grid_cap() {
    // Card 181 Bug 2: wgpu batched prefill failed at long ISL with `GridCap([229376,1,1])` in the `Q @ K^T` MatMul itself
    // (`scores = [1,Hq,L,L]`). At Hq=1 the batched-tiled-GEMM tile count (`ceil(L/16)*ceil(L/8)`) crosses the device's
    // known-miscompile ceiling (`caps.known_miscompiles.tiled_gemm_max_workgroups`, 2^15 on the RADV default, card 095)
    // well before the output element count crosses the wg-bump's `65535*256` ceiling, so
    // `is_batched_tiled_gemm` declines and the eqn falls to the naive `matmul_batched_dt`, which had no 2-D grid fold. L=4160
    // (Hq=1, D=8) makes `scores` 17,305,600 elements: `is_batched_tiled_gemm` sees 135,200 tiles (> 32768, declines) while
    // `is_elementwise_2d` sees 17,305,600 > 16,776,960 (folds), the regime of the real Qwen2.5-0.5B ISL=2048 repro
    // (`14*2048*2048 = 58,720,256` elements, 458,752 tiles). Smallest synthetic shape that reproduces the naive-matmul
    // grid-cap path.
    use poot_graph_ir::ops::attention_prefill;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, d, l) = (1usize, 1usize, 8usize, 4160usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let qd: Vec<f32> = (0..hq * l * d)
        .map(|i| ((i as f32) * 0.0000123).sin() * 0.4)
        .collect();
    let kd: Vec<f32> = (0..hkv * l * d)
        .map(|i| ((i as f32) * 0.0000271).cos() * 0.3)
        .collect();
    let vd: Vec<f32> = (0..hkv * l * d)
        .map(|i| ((i as f32) * 0.0000417).sin() * 0.2)
        .collect();
    // additive causal mask [1,1,L,L]: 0 on/below the diagonal, large-negative above.
    let mask_data: Vec<f32> = (0..l * l)
        .map(|idx| {
            let (i, j) = (idx / l, idx % l);
            if j <= i { 0.0 } else { -1.0e30 }
        })
        .collect();

    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, l, l]));
    let out = attention_prefill(&b, q, k, v, n_rep, scale, mask);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(qi, HostTensor::f32(vec![1, hq, l, d], qd));
    inputs.insert(ki, HostTensor::f32(vec![1, hkv, l, d], kd));
    inputs.insert(vi, HostTensor::f32(vec![1, hkv, l, d], vd));
    inputs.insert(mi, HostTensor::f32(vec![1, 1, l, l], mask_data));

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
    assert_eq!(got.shape(), cpu.shape());
    assert_prefill_op_gpu_matches_cpu(
        "attention_prefill_past_matmul_grid_cap",
        "out",
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        5e-3,
        1e-3,
    );
}

#[test]
fn gpu_gdn_prefill_chunked_matches_cpu_c64() {
    // GDN chunked delta-rule prefill at the real chunk size C=64, L=72 (2 chunks: 64 real positions, then 8 real + 56 padded): crosses a chunk boundary and the zero-padded tail together.
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    check_gdn_prefill_chunked_gpu(&mut gpu, target, 72, 64, 3);
}

#[test]
fn gpu_gdn_prefill_chunked_matches_cpu_heavy_pad() {
    // Card 158's real-35B condition (the 35B pads ~6 real positions to a full 64-chunk): L=6, C=64 -> one chunk with 58 padded
    // zero positions. The CPU oracle showed the padding is inert for the sequential reference (poot-eval's
    // `gdn_prefill_chunked_matches_decode_recurrence_heavy_pad`); this checks the GPU execution of the heavily padded chunk
    // (mostly-zero rows/cols through the 64x64 UT-inverse and matmuls).
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    check_gdn_prefill_chunked_gpu(&mut gpu, target, 6, 64, 3);
}

#[test]
fn gpu_gdn_prefill_chunked_matches_cpu_real_d() {
    // Card 158: the f32 GDN-prefill GPU tests above (`_c64`, `_heavy_pad`) use a tiny per-head dim (D=3); the real Qwen3-Next
    // `gdn_head_dim` is 128 (35B config). A D-dependent GPU bug (LDS-array bound, lane-count assumption, loop-tiling constant)
    // would be invisible at D=3. L=64=chunk (one full chunk, no padding) isolates D from padding, covered by `_heavy_pad`.
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    check_gdn_prefill_chunked_gpu(&mut gpu, target, 64, 64, 128);
}

#[test]
fn gpu_attention_prefill_matches_cpu_real_d() {
    // Card 158: `gpu_attention_prefill_matches_cpu_l64` uses D=8; the real Qwen3-Next full-attention head_dim is 256. D=128
    // already crosses the plausible LDS/lane-width power-of-two boundaries and keeps the run under the iGPU time budget.
    // Hq=4, Hkv=2 (n_rep=2), L=64 (the real per-chunk prefill length).
    use poot_graph_ir::ops::attention_prefill;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, d, l) = (4usize, 2usize, 128usize, 64usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let qd: Vec<f32> = (0..hq * l * d).map(|i| (i as f32) * 0.0007 - 0.4).collect();
    let kd: Vec<f32> = (0..hkv * l * d)
        .map(|i| (i as f32) * 0.0009 - 0.3)
        .collect();
    let vd: Vec<f32> = (0..hkv * l * d)
        .map(|i| (i as f32) * 0.0011 - 0.2)
        .collect();
    // additive causal mask [1,1,L,L]: 0 on/below the diagonal, large-negative above (the qwen3next
    // gated-attention-prefill convention).
    let mask_data: Vec<f32> = (0..l * l)
        .map(|idx| {
            let (i, j) = (idx / l, idx % l);
            if j <= i { 0.0 } else { -1.0e30 }
        })
        .collect();

    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, l, l]));
    let out = attention_prefill(&b, q, k, v, n_rep, scale, mask);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(qi, HostTensor::f32(vec![1, hq, l, d], qd));
    inputs.insert(ki, HostTensor::f32(vec![1, hkv, l, d], kd));
    inputs.insert(vi, HostTensor::f32(vec![1, hkv, l, d], vd));
    inputs.insert(mi, HostTensor::f32(vec![1, 1, l, l], mask_data));

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
    assert_eq!(got.shape(), cpu.shape());
    assert_prefill_op_gpu_matches_cpu(
        "attention_prefill_real_d",
        "out",
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        5e-3,
        1e-3,
    );
}
