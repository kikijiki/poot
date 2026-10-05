use super::*;

#[test]
fn rmsnorm_gpu_matches_cpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let n = 16usize;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let out = rmsnorm(&b, x, w, 1e-6);
    let (xi, wi) = (x.id, w.id);
    let g = b.finish(out);

    let xd: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let wd: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![1, 1, n], xd));
    inputs.insert(wi, HostTensor::f32(vec![n], wd));

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
    for (i, (a, b)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        let tol = 1e-5 * b.abs().max(1e-3);
        assert!((a - b).abs() <= tol, "elem {i}: gpu {a} vs cpu {b}");
    }
}

#[test]
fn rope_fused_gpu_matches_cpu_decomposition() {
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::{cse, fold_iota, rope_fusion};
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    // full rotary (rot==d) and partial rotary (rot<d): the fused `Rope` op on GPU must match the un-fused
    // rotate-half decomposition run on the CPU oracle, bit-approximately (float-reorder tolerance).
    for (hq, d, rot) in [(3usize, 8usize, 8usize), (3, 8, 4)] {
        let (g, xi, ci, si) = build_rope_decomposition(hq, d, rot);
        let opt = rope_fusion(&fold_iota(&cse(&g)));
        assert!(
            opt.eqns.iter().any(|e| matches!(e.op, OpKind::Rope { .. })),
            "cse, fold_iota then rope_fusion must fuse the rope chain (hq={hq} d={d} rot={rot})"
        );
        let xd: Vec<f32> = (0..hq * d).map(|i| (i as f32) * 0.03 - 0.5).collect();
        let cd: Vec<f32> = (0..rot).map(|i| 0.2 + (i as f32) * 0.05).collect();
        let sd: Vec<f32> = (0..rot).map(|i| -0.1 + (i as f32) * 0.04).collect();
        let mut inputs = HashMap::new();
        inputs.insert(xi, HostTensor::f32(vec![1, hq, 1, d], xd));
        inputs.insert(ci, HostTensor::f32(vec![rot], cd));
        inputs.insert(si, HostTensor::f32(vec![rot], sd));

        let eval_inputs: HashMap<_, Value> = inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect();
        let cpu = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap(); // the un-fused decomposition = reference
        let got = run_via_contract(&mut gpu, target, &opt, &inputs); // the fused Rope kernel
        assert_eq!(got.shape(), cpu.shape(), "rope shape (d={d} rot={rot})");
        for (i, (a, b)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            let tol = 1e-5 * b.abs().max(1e-3);
            assert!(
                (a - b).abs() <= tol,
                "rope elem {i} (d={d} rot={rot}): gpu {a} vs cpu {b}"
            );
        }
    }
}

#[test]
fn ge_comparison_gpu_matches_cpu() {
    // `Ge` (a >= b -> 1.0 else 0.0) lowers on the GPU and matches CPU, both standalone (broadcast binary) and inside a fused
    // pointwise chain (`ge(a,b) * c`), where the i1->f32 cast matters.
    use poot_graph_ir::BinOp;
    use poot_graph_plan::{cse, fuse};

    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let n = 16usize;
    // a crosses b; index 6 is an EXACT tie (a == b) -> ge must be 1.0 (>=, not >).
    let ad: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 2.0).collect();
    let bd: Vec<f32> = vec![1.0; n];
    let cd: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 1.0).collect();

    // standalone broadcast `ge(a, b)` (exercises binary_broadcast_dt).
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1, 1, n]));
    let bb = b.constant("b", TensorType::f32(vec![1, 1, n]));
    let ge = b.binary(BinOp::Ge, a, bb);
    let (ai, bi) = (a.id, bb.id);
    let g = b.finish(ge);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![1, 1, n], ad.clone()));
    inputs.insert(bi, HostTensor::f32(vec![1, 1, n], bd.clone()));
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
    for (i, (x, y)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!((x - y).abs() < 1e-6, "ge elem {i}: gpu {x} vs cpu {y}");
        assert!(*x == 0.0 || *x == 1.0, "ge elem {i} not 0/1: {x}");
    }
    assert_eq!(
        cpu.as_f32().unwrap()[6],
        1.0,
        "exact tie a==b must be 1.0 (>=)"
    );

    // fused `ge(a,b) * c` (exercises the fused-scalar comparison path + the i1->f32 cast).
    let b2 = Builder::new();
    let a2 = b2.constant("a", TensorType::f32(vec![1, 1, n]));
    let b2b = b2.constant("b", TensorType::f32(vec![1, 1, n]));
    let c2 = b2.constant("c", TensorType::f32(vec![1, 1, n]));
    let masked = b2.binary(BinOp::Mul, b2.binary(BinOp::Ge, a2, b2b), c2);
    let (a2i, b2i, c2i) = (a2.id, b2b.id, c2.id);
    let g2 = fuse(&cse(&b2.finish(masked)));

    let mut in2 = HashMap::new();
    in2.insert(a2i, HostTensor::f32(vec![1, 1, n], ad));
    in2.insert(b2i, HostTensor::f32(vec![1, 1, n], bd));
    in2.insert(c2i, HostTensor::f32(vec![1, 1, n], cd.clone()));
    let eval_in2: HashMap<_, Value> = in2
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu2 = eval(&g2, &eval_in2, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got2 = run_via_contract(&mut gpu, target, &g2, &in2);
    assert_eq!(got2.shape(), cpu2.shape());
    for (i, (x, y)) in got2
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu2.as_f32().unwrap().iter())
        .enumerate()
    {
        let tol = 1e-5 * y.abs().max(1e-3);
        assert!((x - y).abs() <= tol, "ge*c elem {i}: gpu {x} vs cpu {y}");
        // ground truth: c[i] where a[i] >= 1.0, else 0.0.
        let want = if i >= 6 { cd[i] } else { 0.0 };
        assert!((y - want).abs() < 1e-5, "ge*c ref elem {i}: {y} vs {want}");
    }
}

#[test]
fn layernorm_gpu_matches_cpu() {
    // The LayerNorm composition (mean-subtract, var, scale+bias) lowers on the GPU and matches CPU.
    use poot_graph_ir::ops::layernorm;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let n = 24usize;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 3, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let bias = b.constant("b", TensorType::f32(vec![n]));
    let out = layernorm(&b, x, w, bias, 1e-6);
    let (xi, wi, bi) = (x.id, w.id, bias.id);
    let g = b.finish(out);

    let xd: Vec<f32> = (0..3 * n)
        .map(|i| (i as f32 * 0.21).sin() * 2.0 - 0.3)
        .collect();
    let wd: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.02).collect();
    let bd: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.1).collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![1, 3, n], xd));
    inputs.insert(wi, HostTensor::f32(vec![n], wd));
    inputs.insert(bi, HostTensor::f32(vec![n], bd));
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
    for (i, (a, c)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        let tol = 1e-5 * c.abs().max(1e-3);
        assert!(
            (a - c).abs() <= tol,
            "layernorm elem {i}: gpu {a} vs cpu {c}"
        );
    }
}

#[test]
fn round_and_quantize_composition_gpu_matches_cpu() {
    // `Round` lowers on the GPU, and the int8 quantize composition (div by per-token scale, round, clamp via max/neg) matches CPU on-device.
    use poot_graph_ir::op::{BinOp, RedOp, UnOp};
    use poot_graph_ir::types::Scalar;

    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (rows, cols) = (4usize, 8usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, cols]));
    // per-token scale = absmax/127 (|x| = max(x, -x), last-axis reduce-max).
    let neg = b.unary(UnOp::Neg, x);
    let absx = b.binary(BinOp::Max, x, neg);
    let absmax = b.reduce(RedOp::Max, absx, 1, true);
    let scale = b.binary_scalar(BinOp::Mul, absmax, Scalar::F32(1.0 / 127.0));
    // codes = clamp(round(x/scale), -127, 127), clamp via max/neg: lo = max(q,-127);
    // hi = min(lo,127) = -max(-lo,-127).
    let q = b.binary(BinOp::Div, x, scale);
    let r = b.unary(UnOp::Round, q);
    let lo = b.binary_scalar(BinOp::Max, r, Scalar::F32(-127.0));
    let neg_lo = b.unary(UnOp::Neg, lo);
    let m = b.binary_scalar(BinOp::Max, neg_lo, Scalar::F32(-127.0));
    let codes = b.unary(UnOp::Neg, m);
    let xi = x.id;
    let g = b.finish(codes);

    let xd: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.37).sin() * 3.0)
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![rows, cols], xd));
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
    for (i, (a, c)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (a - c).abs() < 1e-4,
            "quantize elem {i}: gpu {a} vs cpu {c}"
        );
        assert!(
            *a >= -127.0 && *a <= 127.0,
            "code {i} out of int8 range: {a}"
        );
        assert_eq!(*a, a.round(), "code {i} not integral: {a}");
    }
}

/// Exact (erf) GELU, `ops::gelu_erf` over the `Erf` primitive. The GPU expands `Erf` as an abs-folded Abramowitz-Stegun
/// polynomial; CPU eval uses libm erff, so a tight tolerance shows the polynomial is accurate. Also asserts it differs from the tanh-approx GELU and reproduces
/// gelu(1)=0.8413, gelu(-1)=-0.1587, gelu(2)=1.9545.
#[test]
fn gelu_erf_gpu_matches_true_erf_cpu() {
    use poot_graph_ir::ops::{gelu, gelu_erf};
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let n = 64usize;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![n]));
    let xi = x.id;
    let y = gelu_erf(&b, x);
    let g = b.finish(y);
    // a spread spanning the interesting range (both signs, near zero, out to +-6).
    let xd: Vec<f32> = (0..n).map(|i| (i as f32 - 31.5) * (6.0 / 31.5)).collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![n], xd.clone()));
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
    for (i, (a, c)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (a - c).abs() < 3e-4,
            "gelu_erf elem {i} (x={}): gpu {a} vs cpu {c}",
            xd[i]
        );
    }
    // diverges from tanh-approx GELU on the same input (proves it is the exact form, not the tanh one).
    let gt = {
        let bb = Builder::new();
        let xx = bb.constant("x", TensorType::f32(vec![n]));
        let id = xx.id;
        let yy = gelu(&bb, xx);
        let gg = bb.finish(yy);
        let mut ii = HashMap::new();
        ii.insert(id, Value::from(HostTensor::f32(vec![n], xd.clone())));
        eval(&gg, &ii, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    let max_div = max_abs_error(gt.as_f32().unwrap(), cpu.as_f32().unwrap());
    assert!(
        max_div > 1e-4,
        "erf and tanh GELU should differ; max divergence {max_div}"
    );
    // known reference values (torch nn.functional.gelu, exact).
    for (xv, want) in [(1.0f32, 0.8413447), (-1.0, -0.15865526), (2.0, 1.9544997)] {
        let bb = Builder::new();
        let xx = bb.constant("x", TensorType::f32(vec![1]));
        let id = xx.id;
        let yy = gelu_erf(&bb, xx);
        let gg = bb.finish(yy);
        let mut ii = HashMap::new();
        ii.insert(id, HostTensor::f32(vec![1], vec![xv]));
        let got = run_via_contract(&mut gpu, target, &gg, &ii);
        assert!(
            (got.as_f32().unwrap()[0] - want).abs() < 3e-4,
            "gelu_erf({xv}) = {} want {want}",
            got.as_f32().unwrap()[0]
        );
    }
}

#[test]
fn log_and_softplus_gpu_matches_cpu() {
    // The Log primitive (SPIR-V llvm.log.f32; NVPTX lg2*ln2) + softplus = ln(1+exp(x)) (Mamba's Delta): GPU == CPU == the host
    // reference. NVPTX parity is a pod check; this is the wgpu receipt.
    use poot_graph_ir::op::UnOp;
    use poot_graph_ir::ops::softplus;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let xd: Vec<f32> = (0..16).map(|i| (i as f32) * 0.4 - 2.0).collect(); // -2.0 .. 4.0
    // ln on strictly-positive inputs.
    let posd: Vec<f32> = (0..16).map(|i| 0.1 + (i as f32) * 0.3).collect();
    let b = Builder::new();
    let xp = b.constant("x", TensorType::f32(vec![16]));
    let pp = b.constant("p", TensorType::f32(vec![16]));
    let sp = softplus(&b, xp);
    let lg = b.unary(UnOp::Log, pp);
    let out = b.binary(poot_graph_ir::op::BinOp::Add, sp, lg); // one graph exercising both
    let (xi, pi) = (xp.id, pp.id);
    let g = b.finish(out);
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![16], xd.clone()));
    inputs.insert(pi, HostTensor::f32(vec![16], posd.clone()));
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
    // independent host reference.
    for i in 0..16 {
        let want = (1.0 + xd[i].exp()).ln() + posd[i].ln();
        assert!(
            (cpu.as_f32().unwrap()[i] - want).abs() < 1e-4
                && (got.as_f32().unwrap()[i] - want).abs() < 1e-4,
            "elem {i}: gpu {} cpu {} want {want}",
            got.as_f32().unwrap()[i],
            cpu.as_f32().unwrap()[i]
        );
    }
}
