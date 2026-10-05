use super::*;

#[test]
fn linear_attention_decode_gpu_matches_cpu() {
    // Linear-attention decode composition (S_t = decay*S + k^T v; o = q @ S_t) is pure primitives (transpose, two matmuls, a
    // scaled add) and dispatches through the same imported kernels as softmax attention. Output and carried state must match
    // CPU eval.
    use poot_graph_ir::ops::linear_attention_decode;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, dk, dv) = (4usize, 2usize, 5usize, 6usize);
    let n_rep = hq / hkv;
    let qd: Vec<f32> = (0..hq * dk).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let kd: Vec<f32> = (0..hkv * dk).map(|i| (i as f32) * 0.07 - 0.2).collect();
    let vd: Vec<f32> = (0..hkv * dv).map(|i| (i as f32) * 0.05 - 0.1).collect();
    let sd: Vec<f32> = (0..hq * dk * dv)
        .map(|i| (i as f32) * 0.013 - 0.3)
        .collect();
    for pick_state in [false, true] {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, dk]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, 1, dk]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, 1, dv]));
        let s = b.constant("s", TensorType::f32(vec![1, hq, dk, dv]));
        let (o, s_out) = linear_attention_decode(&b, q, k, v, n_rep, 0.9, s);
        let (qi, ki, vi, si) = (q.id, k.id, v.id, s.id);
        let g = b.finish(if pick_state { s_out } else { o });
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, dk], qd.clone()));
        inputs.insert(ki, HostTensor::f32(vec![1, hkv, 1, dk], kd.clone()));
        inputs.insert(vi, HostTensor::f32(vec![1, hkv, 1, dv], vd.clone()));
        inputs.insert(si, HostTensor::f32(vec![1, hq, dk, dv], sd.clone()));
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
        assert_eq!(got.shape(), cpu.shape(), "pick_state={pick_state}");
        for (i, (x, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (x - c).abs() < 1e-4,
                "linear-attn (state={pick_state}) elem {i}: gpu {x} vs cpu {c}"
            );
        }
    }
}

#[test]
fn causal_conv1d_decode_gpu_matches_cpu() {
    // Depthwise causal conv1d decode step, GPU == CPU. Pure primitives (concat, transpose, broadcast multiply, last-axis reduce, slice). K=4, C=6.
    use poot_graph_ir::ops::causal_conv1d_decode;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (k, c) = (4usize, 6usize);
    let xv: Vec<f32> = (0..c).map(|i| i as f32 * 0.1 - 0.3).collect();
    let wv: Vec<f32> = (0..k * c).map(|i| i as f32 * 0.05 - 0.4).collect();
    let cv: Vec<f32> = (0..(k - 1) * c).map(|i| i as f32 * 0.03 - 0.2).collect();
    for pick_state in [false, true] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, c]));
        let w = b.constant("w", TensorType::f32(vec![k, c]));
        let cache = b.constant("cache", TensorType::f32(vec![1, k - 1, c]));
        let (out, new_cache) = causal_conv1d_decode(&b, x, w, cache, k);
        let (xi, wi, ci) = (x.id, w.id, cache.id);
        let g = b.finish(if pick_state { new_cache } else { out });
        let mut inputs = HashMap::new();
        inputs.insert(xi, HostTensor::f32(vec![1, 1, c], xv.clone()));
        inputs.insert(wi, HostTensor::f32(vec![k, c], wv.clone()));
        inputs.insert(ci, HostTensor::f32(vec![1, k - 1, c], cv.clone()));
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
        assert_eq!(got.shape(), cpu.shape(), "pick_state={pick_state}");
        for (i, (xx, cc)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (xx - cc).abs() < 1e-4,
                "conv (state={pick_state}) elem {i}: gpu {xx} vs cpu {cc}"
            );
        }
    }
}

#[test]
fn in_graph_decay_mask_gated_linear_attention_gpu_matches_cpu() {
    // In-graph data-dependent decay mask (decay_mask_from_gates: log gates -> tril-matmul cumsum -> exp of the difference)
    // feeding `linear_attention_prefill`, fused into one graph, GPU == CPU. Exercises Log + matmul + broadcast sub + exp
    // together. Hq=2, Hkv=1, D=3, L=4. Checks both the mask and the attention output.
    use poot_graph_ir::ops::{decay_mask_from_gates, linear_attention_prefill};
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (l, hq, hkv, d) = (4usize, 2usize, 1usize, 3usize);
    let n_rep = hq / hkv;
    let qd: Vec<f32> = (0..hq * l * d).map(|i| i as f32 * 0.05 - 0.3).collect();
    let kd: Vec<f32> = (0..hkv * l * d).map(|i| i as f32 * 0.04 - 0.2).collect();
    let vd: Vec<f32> = (0..hkv * l * d).map(|i| i as f32 * 0.06 - 0.25).collect();
    let gd: Vec<f32> = (0..hq * l).map(|i| 0.5 + 0.08 * ((i % 4) as f32)).collect();
    let mut tril = vec![0.0f32; hq * l * l];
    for h in 0..hq {
        for t in 0..l {
            for j in 0..=t {
                tril[(h * l + t) * l + j] = 1.0;
            }
        }
    }
    for pick_mask in [true, false] {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
        let gates = b.constant("g", TensorType::f32(vec![1, hq, 1, l]));
        let tr = b.constant("tr", TensorType::f32(vec![1, hq, l, l]));
        let ids = [q.id, k.id, v.id, gates.id, tr.id];
        let mask = decay_mask_from_gates(&b, gates, tr, hq, l);
        let o = linear_attention_prefill(&b, q, k, v, n_rep, mask);
        let g = b.finish(if pick_mask { mask } else { o });
        let mut inputs = HashMap::new();
        inputs.insert(ids[0], HostTensor::f32(vec![1, hq, l, d], qd.clone()));
        inputs.insert(ids[1], HostTensor::f32(vec![1, hkv, l, d], kd.clone()));
        inputs.insert(ids[2], HostTensor::f32(vec![1, hkv, l, d], vd.clone()));
        inputs.insert(ids[3], HostTensor::f32(vec![1, hq, 1, l], gd.clone()));
        inputs.insert(ids[4], HostTensor::f32(vec![1, hq, l, l], tril.clone()));
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
        assert_eq!(got.shape(), cpu.shape(), "pick_mask={pick_mask}");
        for (i, (xx, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (xx - c).abs() < 1e-4,
                "decay-mask GLA (mask={pick_mask}) elem {i}: gpu {xx} vs cpu {c}"
            );
        }
    }
}

#[test]
fn mamba2_ssd_decode_gpu_matches_cpu() {
    // Mamba2 / SSD decode step (discretization exp(Delta*A), Delta*B + gated linear-attn step + D*x skip), GPU == CPU. Pure primitives. Hq=2, N=3, P=4.
    use poot_graph_ir::ops::mamba2_ssd_decode;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, n, p) = (2usize, 3usize, 4usize);
    let xd: Vec<f32> = (0..hq * p).map(|i| i as f32 * 0.07 - 0.3).collect();
    let bd: Vec<f32> = (0..hq * n).map(|i| i as f32 * 0.05 - 0.2).collect();
    let cd: Vec<f32> = (0..hq * n).map(|i| i as f32 * 0.06 - 0.25).collect();
    let dld: Vec<f32> = (0..hq).map(|i| 0.3 + 0.2 * i as f32).collect();
    let ad: Vec<f32> = (0..hq).map(|i| -0.5 - 0.3 * i as f32).collect();
    let dd: Vec<f32> = (0..hq).map(|i| 0.2 + 0.1 * i as f32).collect();
    let hd: Vec<f32> = (0..hq * n * p).map(|i| i as f32 * 0.01 - 0.1).collect();
    for pick_state in [false, true] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, hq, 1, p]));
        let bin = b.constant("b", TensorType::f32(vec![1, hq, 1, n]));
        let cin = b.constant("c", TensorType::f32(vec![1, hq, 1, n]));
        let delta = b.constant("dl", TensorType::f32(vec![1, hq, 1, 1]));
        let a_param = b.constant("a", TensorType::f32(vec![1, hq, 1, 1]));
        let d_skip = b.constant("d", TensorType::f32(vec![1, hq, 1, 1]));
        let h_in = b.constant("h", TensorType::f32(vec![1, hq, n, p]));
        let (y, h_out) = mamba2_ssd_decode(&b, x, bin, cin, delta, a_param, d_skip, h_in);
        let (xi, bi, ci, dli, ai, di, hi) = (
            x.id, bin.id, cin.id, delta.id, a_param.id, d_skip.id, h_in.id,
        );
        let g = b.finish(if pick_state { h_out } else { y });
        let mut inputs = HashMap::new();
        inputs.insert(xi, HostTensor::f32(vec![1, hq, 1, p], xd.clone()));
        inputs.insert(bi, HostTensor::f32(vec![1, hq, 1, n], bd.clone()));
        inputs.insert(ci, HostTensor::f32(vec![1, hq, 1, n], cd.clone()));
        inputs.insert(dli, HostTensor::f32(vec![1, hq, 1, 1], dld.clone()));
        inputs.insert(ai, HostTensor::f32(vec![1, hq, 1, 1], ad.clone()));
        inputs.insert(di, HostTensor::f32(vec![1, hq, 1, 1], dd.clone()));
        inputs.insert(hi, HostTensor::f32(vec![1, hq, n, p], hd.clone()));
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
        assert_eq!(got.shape(), cpu.shape(), "pick_state={pick_state}");
        for (i, (xx, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (xx - c).abs() < 1e-4,
                "ssd (state={pick_state}) elem {i}: gpu {xx} vs cpu {c}"
            );
        }
    }
}

#[test]
fn mamba_core_path_gpu_matches_cpu() {
    // Full Mamba SSM core fused into one graph on the GPU: causal conv1d -> SiLU -> reshape to heads -> SSD with
    // Delta = softplus(dt), with two carried states. Each of the three outputs (y, conv cache, SSM state) must match CPU.
    // Hq=2, P=4 (conv channels = 8), N=3, K=4.
    use poot_graph_ir::ops::{causal_conv1d_decode, mamba2_ssd_decode, softplus};
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, pdim, n, kk) = (2usize, 4usize, 3usize, 4usize);
    let ch = hq * pdim;
    let xcv: Vec<f32> = (0..ch).map(|i| i as f32 * 0.07 - 0.3).collect();
    let wcv: Vec<f32> = (0..kk * ch).map(|i| i as f32 * 0.02 - 0.2).collect();
    let cvv: Vec<f32> = (0..(kk - 1) * ch).map(|i| i as f32 * 0.03 - 0.15).collect();
    let bd: Vec<f32> = (0..hq * n).map(|i| i as f32 * 0.05 - 0.2).collect();
    let cd: Vec<f32> = (0..hq * n).map(|i| i as f32 * 0.06 - 0.25).collect();
    let dtd: Vec<f32> = (0..hq).map(|i| 0.1 + 0.15 * i as f32).collect();
    let ad: Vec<f32> = (0..hq).map(|i| -0.5 - 0.2 * i as f32).collect();
    let dd: Vec<f32> = (0..hq).map(|i| 0.2 + 0.1 * i as f32).collect();
    let hd: Vec<f32> = (0..hq * n * pdim).map(|i| i as f32 * 0.01 - 0.1).collect();
    // 0 = y, 1 = conv cache, 2 = SSM state
    for pick in 0..3u8 {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, ch]));
        let wc = b.constant("wc", TensorType::f32(vec![kk, ch]));
        let cache = b.constant("cache", TensorType::f32(vec![1, kk - 1, ch]));
        let bb = b.constant("b", TensorType::f32(vec![1, hq, 1, n]));
        let cc = b.constant("c", TensorType::f32(vec![1, hq, 1, n]));
        let dt = b.constant("dt", TensorType::f32(vec![1, hq, 1, 1]));
        let ap = b.constant("a", TensorType::f32(vec![1, hq, 1, 1]));
        let dsk = b.constant("d", TensorType::f32(vec![1, hq, 1, 1]));
        let h_in = b.constant("h", TensorType::f32(vec![1, hq, n, pdim]));
        let ids = [
            x.id, wc.id, cache.id, bb.id, cc.id, dt.id, ap.id, dsk.id, h_in.id,
        ];
        let (conv_out, conv_cache_out) = causal_conv1d_decode(&b, x, wc, cache, kk);
        let xact = poot_graph_ir::ops::silu(&b, conv_out);
        let xheads = b.reshape(xact, vec![1, hq, 1, pdim]);
        let delta = softplus(&b, dt);
        let (y, h_out) = mamba2_ssd_decode(&b, xheads, bb, cc, delta, ap, dsk, h_in);
        let g = b.finish(match pick {
            0 => y,
            1 => conv_cache_out,
            _ => h_out,
        });
        let mut inputs = HashMap::new();
        inputs.insert(ids[0], HostTensor::f32(vec![1, 1, ch], xcv.clone()));
        inputs.insert(ids[1], HostTensor::f32(vec![kk, ch], wcv.clone()));
        inputs.insert(ids[2], HostTensor::f32(vec![1, kk - 1, ch], cvv.clone()));
        inputs.insert(ids[3], HostTensor::f32(vec![1, hq, 1, n], bd.clone()));
        inputs.insert(ids[4], HostTensor::f32(vec![1, hq, 1, n], cd.clone()));
        inputs.insert(ids[5], HostTensor::f32(vec![1, hq, 1, 1], dtd.clone()));
        inputs.insert(ids[6], HostTensor::f32(vec![1, hq, 1, 1], ad.clone()));
        inputs.insert(ids[7], HostTensor::f32(vec![1, hq, 1, 1], dd.clone()));
        inputs.insert(ids[8], HostTensor::f32(vec![1, hq, n, pdim], hd.clone()));
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
        assert_eq!(got.shape(), cpu.shape(), "pick={pick}");
        for (i, (xx, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (xx - c).abs() < 1e-4,
                "mamba core (pick={pick}) elem {i}: gpu {xx} vs cpu {c}"
            );
        }
    }
}

#[test]
fn mamba2_block_gpu_matches_cpu() {
    // Full Mamba2 mixer block fused into one graph: in_proj (z/x_conv/B/C/dt) -> causal conv1d -> SiLU -> softplus(dt) -> SSD ->
    // gate y*silu(z) -> out_proj -> residual (~10 matmuls); GPU == CPU. H=8, Hq=2, P=4, N=3, K=4. Checks block result, conv
    // cache and SSM state.
    use poot_graph_ir::op::BinOp;
    use poot_graph_ir::ops::{causal_conv1d_decode, linear, mamba2_ssd_decode, softplus};
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hh, hq, pdim, n, kk) = (8usize, 2usize, 4usize, 3usize, 4usize);
    let ch = hq * pdim;
    let f = |k: u64, m: usize| -> Vec<f32> {
        (0..m)
            .map(|i| ((i as u64 * 7 + k) % 11) as f32 * 0.07 - 0.35)
            .collect()
    };
    let xv = f(1, hh);
    let (wz, wxc) = (f(2, hh * ch), f(3, hh * ch));
    let (wb, wc) = (f(4, hh * hq * n), f(5, hh * hq * n));
    let (wdt, dtb) = (f(6, hh * hq), f(7, hq));
    let wconv = f(8, kk * ch);
    let wout = f(9, ch * hh);
    let av: Vec<f32> = (0..hq).map(|h| -0.5 - 0.3 * h as f32).collect();
    let dv: Vec<f32> = (0..hq).map(|h| 0.2 + 0.1 * h as f32).collect();
    let cvd = f(10, (kk - 1) * ch);
    let hd = f(11, hq * n * pdim);
    for pick in 0..3u8 {
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
        let cache = b.constant("cv", TensorType::f32(vec![1, kk - 1, ch]));
        let h_in = b.constant("h", TensorType::f32(vec![1, hq, n, pdim]));
        let ids = [
            x.id, cz.id, cxc.id, cb.id, cc.id, cdt.id, cdtb.id, cwconv.id, ca.id, cd.id, cwout.id,
            cache.id, h_in.id,
        ];
        let z = linear(&b, x, cz, None);
        let xc = linear(&b, x, cxc, None);
        let bproj = linear(&b, x, cb, None);
        let cproj = linear(&b, x, cc, None);
        let dt_raw = linear(&b, x, cdt, Some(cdtb));
        let (conv_out, conv_cache_out) = causal_conv1d_decode(&b, xc, cwconv, cache, kk);
        let xact = poot_graph_ir::ops::silu(&b, conv_out);
        let xheads = b.reshape(xact, vec![1, hq, 1, pdim]);
        let delta = softplus(&b, b.reshape(dt_raw, vec![1, hq, 1, 1]));
        let bheads = b.reshape(bproj, vec![1, hq, 1, n]);
        let cheads = b.reshape(cproj, vec![1, hq, 1, n]);
        let (y, h_out) = mamba2_ssd_decode(&b, xheads, bheads, cheads, delta, ca, cd, h_in);
        let zheads = b.reshape(z, vec![1, hq, 1, pdim]);
        let ygated = b.binary(BinOp::Mul, y, poot_graph_ir::ops::silu(&b, zheads));
        let yflat = b.reshape(ygated, vec![1, 1, ch]);
        let out = linear(&b, yflat, cwout, None);
        let result = b.binary(BinOp::Add, x, out);
        let g = b.finish(match pick {
            0 => result,
            1 => conv_cache_out,
            _ => h_out,
        });
        let mut inputs = HashMap::new();
        inputs.insert(ids[0], HostTensor::f32(vec![1, 1, hh], xv.clone()));
        inputs.insert(ids[1], HostTensor::f32(vec![hh, ch], wz.clone()));
        inputs.insert(ids[2], HostTensor::f32(vec![hh, ch], wxc.clone()));
        inputs.insert(ids[3], HostTensor::f32(vec![hh, hq * n], wb.clone()));
        inputs.insert(ids[4], HostTensor::f32(vec![hh, hq * n], wc.clone()));
        inputs.insert(ids[5], HostTensor::f32(vec![hh, hq], wdt.clone()));
        inputs.insert(ids[6], HostTensor::f32(vec![hq], dtb.clone()));
        inputs.insert(ids[7], HostTensor::f32(vec![kk, ch], wconv.clone()));
        inputs.insert(ids[8], HostTensor::f32(vec![1, hq, 1, 1], av.clone()));
        inputs.insert(ids[9], HostTensor::f32(vec![1, hq, 1, 1], dv.clone()));
        inputs.insert(ids[10], HostTensor::f32(vec![ch, hh], wout.clone()));
        inputs.insert(ids[11], HostTensor::f32(vec![1, kk - 1, ch], cvd.clone()));
        inputs.insert(ids[12], HostTensor::f32(vec![1, hq, n, pdim], hd.clone()));
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
        assert_eq!(got.shape(), cpu.shape(), "pick={pick}");
        for (i, (xx, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (xx - c).abs() < 1e-4,
                "mamba2 block (pick={pick}) elem {i}: gpu {xx} vs cpu {c}"
            );
        }
    }
}

#[test]
fn linear_attention_decode_gated_gpu_matches_cpu() {
    // Gated variant: the per-key-channel gate [1,Hq,D_k,1] broadcasts over D_v in the state update. Same primitives as the
    // scalar form plus a broadcast multiply; GPU must match CPU.
    use poot_graph_ir::ops::linear_attention_decode_gated;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, dk, dv) = (4usize, 2usize, 5usize, 6usize);
    let n_rep = hq / hkv;
    let qd: Vec<f32> = (0..hq * dk).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let kd: Vec<f32> = (0..hkv * dk).map(|i| (i as f32) * 0.07 - 0.2).collect();
    let vd: Vec<f32> = (0..hkv * dv).map(|i| (i as f32) * 0.05 - 0.1).collect();
    let sd: Vec<f32> = (0..hq * dk * dv)
        .map(|i| (i as f32) * 0.013 - 0.3)
        .collect();
    let gd: Vec<f32> = (0..hq * dk)
        .map(|i| 0.5 + 0.4 * ((i * 7 % 5) as f32 / 5.0))
        .collect();
    for pick_state in [false, true] {
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
        inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, dk], qd.clone()));
        inputs.insert(ki, HostTensor::f32(vec![1, hkv, 1, dk], kd.clone()));
        inputs.insert(vi, HostTensor::f32(vec![1, hkv, 1, dv], vd.clone()));
        inputs.insert(gi, HostTensor::f32(vec![1, hq, dk, 1], gd.clone()));
        inputs.insert(si, HostTensor::f32(vec![1, hq, dk, dv], sd.clone()));
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
        assert_eq!(got.shape(), cpu.shape(), "pick_state={pick_state}");
        for (i, (x, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (x - c).abs() < 1e-4,
                "gated linear-attn (state={pick_state}) elem {i}: gpu {x} vs cpu {c}"
            );
        }
    }
}

#[test]
fn linear_attention_prefill_gpu_matches_cpu() {
    // Parallel quadratic prefill form ((decay_mask (.) Q K^T) @ V), GPU == CPU. Pure primitives (rank-4 batched matmuls as in
    // softmax prefill, plus a masked multiply).
    use poot_graph_ir::ops::linear_attention_prefill;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (l, hq, hkv, dk, dv) = (6usize, 4usize, 2usize, 5usize, 6usize);
    let n_rep = hq / hkv;
    let decay = 0.8f32;
    let qd: Vec<f32> = (0..hq * l * dk).map(|i| (i as f32) * 0.03 - 0.4).collect();
    let kd: Vec<f32> = (0..hkv * l * dk).map(|i| (i as f32) * 0.02 - 0.2).collect();
    let vd: Vec<f32> = (0..hkv * l * dv)
        .map(|i| (i as f32) * 0.017 - 0.1)
        .collect();
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
    inputs.insert(qi, HostTensor::f32(vec![1, hq, l, dk], qd));
    inputs.insert(ki, HostTensor::f32(vec![1, hkv, l, dk], kd));
    inputs.insert(vi, HostTensor::f32(vec![1, hkv, l, dv], vd));
    inputs.insert(mi, HostTensor::f32(vec![1, 1, l, l], md));
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
    assert_eq!(got.shape(), vec![1, hq, l, dv]);
    for (i, (x, c)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (x - c).abs() < 1e-4,
            "linear-attn prefill elem {i}: gpu {x} vs cpu {c}"
        );
    }
}

#[test]
fn linear_attention_prefill_gated_gpu_matches_cpu() {
    // Per-channel gated prefill (Q'=Q(.)A, K'=K(/)A, binary causal mask), GPU == CPU. Pure primitives (multiply, divide, two
    // matmuls, masked multiply). Mild gates keep K/A precise.
    use poot_graph_ir::ops::linear_attention_prefill_gated;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (l, hq, hkv, dk, dv) = (4usize, 4usize, 2usize, 5usize, 6usize);
    let n_rep = hq / hkv;
    let qd: Vec<f32> = (0..hq * l * dk).map(|i| (i as f32) * 0.03 - 0.4).collect();
    let kd: Vec<f32> = (0..hkv * l * dk).map(|i| (i as f32) * 0.02 - 0.2).collect();
    let vd: Vec<f32> = (0..hkv * l * dv)
        .map(|i| (i as f32) * 0.017 - 0.1)
        .collect();
    let gate =
        |h: usize, t: usize, a: usize| 0.7 + 0.25 * (((h * 5 + t * 3 + a * 2) % 4) as f32 / 4.0);
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
    inputs.insert(qi, HostTensor::f32(vec![1, hq, l, dk], qd));
    inputs.insert(ki, HostTensor::f32(vec![1, hkv, l, dk], kd));
    inputs.insert(vi, HostTensor::f32(vec![1, hkv, l, dv], vd));
    inputs.insert(ai, HostTensor::f32(vec![1, hq, l, dk], ad));
    inputs.insert(mi, HostTensor::f32(vec![1, 1, l, l], md));
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
    assert_eq!(got.shape(), vec![1, hq, l, dv]);
    for (i, (x, c)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (x - c).abs() < 1e-4,
            "gated prefill elem {i}: gpu {x} vs cpu {c}"
        );
    }
}

#[test]
fn linear_attention_prefill_chunked_gpu_matches_cpu() {
    // Chunked-scan per-channel gated prefill, GPU == CPU. The chunk loop is unrolled at trace time (slices, per-chunk matmuls,
    // carried-state multiply, concat). L=8, chunk=4.
    use poot_graph_ir::ops::linear_attention_prefill_chunked;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (l, c, hq, hkv, dk, dv) = (8usize, 4usize, 4usize, 2usize, 5usize, 6usize);
    let n_rep = hq / hkv;
    let qd: Vec<f32> = (0..hq * l * dk).map(|i| (i as f32) * 0.02 - 0.4).collect();
    let kd: Vec<f32> = (0..hkv * l * dk)
        .map(|i| (i as f32) * 0.015 - 0.2)
        .collect();
    let vd: Vec<f32> = (0..hkv * l * dv)
        .map(|i| (i as f32) * 0.012 - 0.1)
        .collect();
    let gate =
        |h: usize, t: usize, a: usize| 0.6 + 0.3 * (((h * 5 + t * 3 + a * 2) % 4) as f32 / 4.0);
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
    let mut cd = vec![0.0f32; c * c];
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
    inputs.insert(qi, HostTensor::f32(vec![1, hq, l, dk], qd));
    inputs.insert(ki, HostTensor::f32(vec![1, hkv, l, dk], kd));
    inputs.insert(vi, HostTensor::f32(vec![1, hkv, l, dv], vd));
    inputs.insert(bi, HostTensor::f32(vec![1, hq, l, dk], bd));
    inputs.insert(ci, HostTensor::f32(vec![1, 1, c, c], cd));
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
    assert_eq!(got.shape(), vec![1, hq, l, dv]);
    for (i, (x, cc)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (x - cc).abs() < 1e-4,
            "chunked prefill elem {i}: gpu {x} vs cpu {cc}"
        );
    }
}

#[test]
fn gpu_causal_conv1d_prefill_matches_cpu() {
    // Batched depthwise causal conv1d of the GDN block's prefill path, GPU vs CPU. K=4, C=8 channels, L=16. Pure primitives
    // (concat/slice/broadcast-multiply/reduce).
    use poot_graph_ir::ops::causal_conv1d_prefill;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (k, c, l) = (4usize, 8usize, 16usize);
    let xd: Vec<f32> = (0..l * c).map(|i| (i as f32) * 0.013 - 0.4).collect();
    let wd: Vec<f32> = (0..k * c).map(|i| (i as f32) * 0.02 - 0.3).collect();

    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, l, c]));
    let w = b.constant("w", TensorType::f32(vec![k, c]));
    let out = causal_conv1d_prefill(&b, x, w, k);
    let (xi, wi) = (x.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![1, l, c], xd));
    inputs.insert(wi, HostTensor::f32(vec![k, c], wd));

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
        "causal_conv1d_prefill",
        "out",
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        3e-3,
        1e-3,
    );
}

#[test]
fn linear_attention_prefill_chunked_gpu_fuzz_matches_cpu() {
    // Device-path shape fuzzer for the chunked gated linear-attention prefill, the backend-codegen companion to the CPU
    // composition-logic fuzzer (0308). The chunked op emits a different kernel mix than the single-shot prefill (per-chunk
    // slices, a concat over per-chunk outputs, carried-state matmuls), so a GPU==CPU sweep over random chunk size / chunk count
    // / n_rep / D_k / D_v exercises those kernels across shapes. Values are arbitrary (semantic correctness is the CPU
    // fuzzer's job); this checks gpu.run == eval. 16 seeds, small dims to bound per-shape kernel compiles.
    use poot_graph_ir::ops::linear_attention_prefill_chunked;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let fill = |seed: u64, count: usize, lo: f32, hi: f32| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                lo + ((s >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
            })
            .collect()
    };
    for seed in 0u64..16 {
        let mut s = seed
            .wrapping_mul(0x100000001B3)
            .wrapping_add(0x9E3779B97F4A7C15)
            | 1;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let c = 2 + (rng() % 2) as usize; // chunk size 2..=3
        let n_chunks = 2 + (rng() % 2) as usize; // 2..=3 chunks
        let l = c * n_chunks;
        let n_rep = 1 + (rng() % 2) as usize;
        let hkv = 1usize;
        let hq = hkv * n_rep;
        let dk = 2 + (rng() % 3) as usize; // 2..=4
        let dv = 2 + (rng() % 3) as usize;

        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, dk]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, l, dk]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, l, dv]));
        let be = b.constant("be", TensorType::f32(vec![1, hq, l, dk]));
        let ca = b.constant("ca", TensorType::f32(vec![1, 1, c, c]));
        let ids = [q.id, k.id, v.id, be.id, ca.id];
        let o = linear_attention_prefill_chunked(&b, q, k, v, n_rep, be, ca, c);
        let g = b.finish(o);
        let causal_chunk: Vec<f32> = (0..c * c)
            .map(|i| if i % c <= i / c { 1.0 } else { 0.0 })
            .collect();
        let mut inputs = HashMap::new();
        inputs.insert(
            ids[0],
            HostTensor::f32(vec![1, hq, l, dk], fill(seed, hq * l * dk, -0.6, 0.6)),
        );
        inputs.insert(
            ids[1],
            HostTensor::f32(
                vec![1, hkv, l, dk],
                fill(seed ^ 0xA5, hkv * l * dk, -0.6, 0.6),
            ),
        );
        inputs.insert(
            ids[2],
            HostTensor::f32(
                vec![1, hkv, l, dv],
                fill(seed ^ 0x5A, hkv * l * dv, -0.6, 0.6),
            ),
        );
        // beta strictly positive (it divides K); arbitrary values are fine for a gpu==cpu check.
        inputs.insert(
            ids[3],
            HostTensor::f32(vec![1, hq, l, dk], fill(seed ^ 0x33, hq * l * dk, 0.3, 1.0)),
        );
        inputs.insert(ids[4], HostTensor::f32(vec![1, 1, c, c], causal_chunk));

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
        assert_eq!(got.shape(), vec![1, hq, l, dv], "seed {seed}");
        for (i, (x, cv)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (x - cv).abs() <= 1e-3 + 1e-3 * cv.abs(),
                "chunked GPU (seed {seed}: c={c},nc={n_chunks},hq={hq},dk={dk},dv={dv}) elem {i}: gpu {x} vs cpu {cv}"
            );
        }
    }
}

#[test]
fn mamba2_ssd_decode_gpu_fuzz_matches_cpu() {
    // Device-path shape fuzzer for the Mamba2 / SSD decode step, the companion to the CPU recurrence fuzzer (0307). Sweeps
    // random Hq / N / P, asserting gpu.run == CPU eval (exp discretization, gated linear-attn update, D*x skip). 24 seeds.
    // Picks both the output and the carried state.
    use poot_graph_ir::ops::mamba2_ssd_decode;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let fill = |seed: u64, count: usize, lo: f32, hi: f32| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                lo + ((s >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
            })
            .collect()
    };
    for seed in 0u64..24 {
        let mut s = seed
            .wrapping_mul(0x100000001B3)
            .wrapping_add(0x9E3779B97F4A7C15)
            | 1;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let hq = 1 + (rng() % 3) as usize; // 1..=3
        let n = 2 + (rng() % 4) as usize; // 2..=5
        let p = 2 + (rng() % 4) as usize; // 2..=5
        for pick_state in [false, true] {
            let b = Builder::new();
            let x = b.constant("x", TensorType::f32(vec![1, hq, 1, p]));
            let bin = b.constant("b", TensorType::f32(vec![1, hq, 1, n]));
            let cin = b.constant("c", TensorType::f32(vec![1, hq, 1, n]));
            let dl = b.constant("dl", TensorType::f32(vec![1, hq, 1, 1]));
            let a = b.constant("a", TensorType::f32(vec![1, hq, 1, 1]));
            let dsk = b.constant("d", TensorType::f32(vec![1, hq, 1, 1]));
            let h_in = b.constant("h", TensorType::f32(vec![1, hq, n, p]));
            let ids = [x.id, bin.id, cin.id, dl.id, a.id, dsk.id, h_in.id];
            let (y, h_out) = mamba2_ssd_decode(&b, x, bin, cin, dl, a, dsk, h_in);
            let g = b.finish(if pick_state { h_out } else { y });
            let mut inputs = HashMap::new();
            inputs.insert(
                ids[0],
                HostTensor::f32(vec![1, hq, 1, p], fill(seed, hq * p, -0.6, 0.6)),
            );
            inputs.insert(
                ids[1],
                HostTensor::f32(vec![1, hq, 1, n], fill(seed ^ 0x11, hq * n, -0.6, 0.6)),
            );
            inputs.insert(
                ids[2],
                HostTensor::f32(vec![1, hq, 1, n], fill(seed ^ 0x22, hq * n, -0.6, 0.6)),
            );
            inputs.insert(
                ids[3],
                HostTensor::f32(vec![1, hq, 1, 1], fill(seed ^ 0x33, hq, 0.1, 0.6)),
            );
            inputs.insert(
                ids[4],
                HostTensor::f32(vec![1, hq, 1, 1], fill(seed ^ 0x44, hq, -1.0, -0.3)),
            );
            inputs.insert(
                ids[5],
                HostTensor::f32(vec![1, hq, 1, 1], fill(seed ^ 0x55, hq, 0.1, 0.5)),
            );
            inputs.insert(
                ids[6],
                HostTensor::f32(vec![1, hq, n, p], fill(seed ^ 0x66, hq * n * p, -0.3, 0.3)),
            );

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
            assert_eq!(
                got.shape(),
                cpu.shape(),
                "seed {seed} pick_state={pick_state}"
            );
            for (i, (x, cv)) in got
                .as_f32()
                .unwrap()
                .iter()
                .zip(cpu.as_f32().unwrap().iter())
                .enumerate()
            {
                assert!(
                    (x - cv).abs() <= 1e-4 + 1e-4 * cv.abs(),
                    "ssd GPU (seed {seed}: hq={hq},n={n},p={p},state={pick_state}) elem {i}: gpu {x} vs cpu {cv}"
                );
            }
        }
    }
}
