use super::*;

use poot_graph_plan::{ImportedKernel, KernelChoice};
use poot_kernelgen::{ContractionSpec, KernelRequest};

/// ADR-0113: a test checks the production kernel choice instead of calling the deleted test-only predicates
/// `is_decode_gemv`/`is_tiled_gemm`. A plan split into chunks has one choice per chunk.
fn leaf_choices(choice: &KernelChoice) -> Vec<&KernelChoice> {
    match choice {
        KernelChoice::Chunked(chunks) => chunks.iter().collect(),
        other => vec![other],
    }
}

/// The shipped tiled GEMM (shared or batched weight, with or without bias).
fn is_tiled_gemm_kernel(choice: &KernelChoice) -> bool {
    matches!(
        choice,
        KernelChoice::Imported {
            kernel: ImportedKernel::TiledGemm
                | ImportedKernel::TiledGemmBias
                | ImportedKernel::TiledGemmBatched,
            ..
        }
    )
}

/// The generated tiled GEMM (`kg::tiled_region`), the planner's choice for an M>1 F32 matmul against a rank-2
/// weight (Card 557).
fn is_generated_tiled_gemm(choice: &KernelChoice) -> bool {
    matches!(
        choice,
        KernelChoice::Generated(KernelRequest::Contraction(
            ContractionSpec::TiledRegion { .. }
        ))
    )
}

/// The decode GEMV: a shipped coalesced body, or one range of a watchdog-split generated one.
fn is_decode_gemv_kernel(choice: &KernelChoice) -> bool {
    matches!(
        choice,
        KernelChoice::Imported {
            kernel: ImportedKernel::GemvCoalesced
                | ImportedKernel::GemvCoalescedBias
                | ImportedKernel::GemvBatchedCoalesced
                | ImportedKernel::GemvBatchedCoalescedBias
                | ImportedKernel::GemvCoalescedBf16
                | ImportedKernel::GemvCoalescedBiasBf16
                | ImportedKernel::GemvBatchedCoalescedBf16
                | ImportedKernel::GemvBatchedCoalescedBiasBf16,
            ..
        } | KernelChoice::Generated(KernelRequest::Contraction(
            ContractionSpec::GemvChunk { .. }
        ))
    )
}

#[test]
fn tiled_gemm_large_n_under_tile_threshold_matches_cpu() {
    // Large-N tiled GEMM (N=174000) with a tile count under the failure threshold (M=16 -> 21750 tiles) is GPU==CPU. Shows the
    // card-095 corruption is driven by tile count (occupancy race above ~30-65k workgroups), not large-N index arithmetic.
    use poot_graph_ir::op::OpKind;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (m, k, n) = (16usize, 16usize, 174_000usize);
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![m, k]));
    let wt = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(a, wt);
    let g = b.finish(out);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .unwrap();
    let (plan, choice) = plan_eqn_with_choice(
        &g,
        eqn,
        poot_target::Backend::SpirvVulkan,
        &poot_test_util::device_caps::default_caps_for(poot_target::Backend::SpirvVulkan),
    )
    .unwrap();
    assert!(
        leaf_choices(&choice)
            .into_iter()
            .all(is_generated_tiled_gemm),
        "expected tiled path, got {plan:?} / {choice:?}"
    );
    let r = |s: usize, num: usize| -> Vec<f32> {
        (0..num)
            .map(|i| ((i + s) as f32 * 0.0007).sin() * 0.1)
            .collect()
    };
    let mut inp = HashMap::new();
    inp.insert(a.id, HostTensor::f32(vec![m, k], r(1, m * k)));
    inp.insert(wt.id, HostTensor::f32(vec![k, n], r(7, k * n)));
    let eval_inp: HashMap<_, Value> = inp
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_via_contract(&mut gpu, target, &g, &inp);
    // large-N tiled GEMM under the tile-count threshold must be GPU==CPU
    assert_close(got.as_f32().unwrap(), cpu.as_f32().unwrap(), 1e-3);
}

#[test]
fn tiled_gemm_tile_count_threshold_gated_and_correct() {
    // Card 095: the wgpu tiled GEMM (`kernelgen::tiled_gemm_dt`) miscompiles on RADV at >= 2^15 = 32768 dispatched workgroups
    // (output tiles): every element is wrong and non-deterministic; at 32767 it is exact. Card 096 chunks larger GEMMs into
    // sub-2^15 dispatches. This pins both sides:
    //  (a) just under (M=16, N=262136 -> 32767 tiles): single-dispatch tiled path;
    //  (b) just over (M=16, N=262144 -> 32768 tiles): chunked tiled path;
    //  both GPU==CPU and deterministic across runs.
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::Plan;
    use poot_target::Backend;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let k = 16usize;
    let r = |s: usize, num: usize| -> Vec<f32> {
        (0..num)
            .map(|i| ((i + s) as f32 * 0.0007).sin() * 0.1)
            .collect()
    };
    // (m, n, expect_chunked): tiles = ceil(m/16)*ceil(n/8). 262136 -> 32767 (single), 262144 -> 32768 (chunked).
    let cases = [
        (16usize, 262_136usize, false),
        (16usize, 262_144usize, true),
    ];
    for (m, n, expect_chunked) in cases {
        let b = Builder::new();
        let a = b.constant("a", TensorType::f32(vec![m, k]));
        let wt = b.constant("w", TensorType::f32(vec![k, n]));
        let out = b.matmul(a, wt);
        let g = b.finish(out);
        let eqn = g
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul))
            .unwrap();
        // both take the tiled path; the plan is single Compute vs ComputeChunks.
        let (plan, choice) = plan_eqn_with_choice(
            &g,
            eqn,
            Backend::SpirvVulkan,
            &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
        )
        .unwrap();
        assert!(
            leaf_choices(&choice)
                .into_iter()
                .all(is_generated_tiled_gemm),
            "m={m} n={n}: should take the tiled path, got {plan:?} / {choice:?}"
        );
        let chunked = matches!(plan, Plan::ComputeChunks(_));
        assert_eq!(
            chunked,
            expect_chunked,
            "m={m} n={n} (tiles={}): single-vs-chunked dispatch wrong",
            m.div_ceil(16) * n.div_ceil(8)
        );
        let mut inp = HashMap::new();
        inp.insert(a.id, HostTensor::f32(vec![m, k], r(1, m * k)));
        inp.insert(wt.id, HostTensor::f32(vec![k, n], r(7, k * n)));
        let eval_inp: HashMap<_, Value> = inp
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect();
        let cpu = eval(&g, &eval_inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();
        // run twice: GPU==CPU AND deterministic (the bug's signature was non-deterministic corruption).
        let g0 = run_via_contract(&mut gpu, target, &g, &inp);
        let g1 = run_via_contract(&mut gpu, target, &g, &inp);
        let nondet = g0
            .as_f32()
            .unwrap()
            .iter()
            .zip(g1.as_f32().unwrap().iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_close(g0.as_f32().unwrap(), cpu.as_f32().unwrap(), 1e-3);
        assert_eq!(
            nondet, 0,
            "m={m} n={n}: {nondet} non-deterministic elements across runs"
        );
    }
}

#[test]
fn large_elementwise_readback_matches_cpu() {
    // A large (~8.4M element, ~33 MB) elementwise op via the eager gpu.run path reads back GPU==CPU. Isolated card 095: large-buffer
    // readback and the eager path are fine, so the tiled-GEMM failure is in the kernel's compute.
    use poot_graph_ir::op::BinOp as B2;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let n = 48 * 174_000usize; // same element count as the failing tiled test's output
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![n]));
    let y = b.binary(B2::Mul, x, x); // x*x, one thread per element
    let g = b.finish(y);
    let xd: Vec<f32> = (0..n).map(|i| ((i % 1000) as f32) * 0.001).collect();
    let mut inp = HashMap::new();
    inp.insert(x.id, HostTensor::f32(vec![n], xd.clone()));
    let eval_inp: HashMap<_, Value> = inp
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_via_contract(&mut gpu, target, &g, &inp);
    assert_close(got.as_f32().unwrap(), cpu.as_f32().unwrap(), 1e-4);
}

#[test]
fn lm_head_gemv_2d_grid_matches_cpu() {
    // The decode GEMV lays its output over a 2-D workgroup grid (`col = GroupY*x_groups + GroupX`), so a large-vocab lm_head
    // (out cols > the 65535 per-dim cap) uses the LDS-GEMV instead of the naive kernel. A [1,K] @ [K,N] projection with
    // N > 65535 must lower GPU == CPU (exercising the Y-grid spill and over-dispatch tail guard).
    use poot_graph_ir::op::OpKind;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (k, n) = (48usize, 70_000usize); // n > 65535 -> the grid spills onto Y
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1, k]));
    let wt = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(a, wt);
    let g = b.finish(out);
    // sanity: this is the decode-GEMV path (M=1, rank-2 weight, large N).
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .unwrap();
    let (plan, choice) = plan_eqn_with_choice(
        &g,
        eqn,
        poot_target::Backend::SpirvVulkan,
        &poot_test_util::device_caps::default_caps_for(poot_target::Backend::SpirvVulkan),
    )
    .unwrap();
    assert!(
        leaf_choices(&choice).into_iter().all(is_decode_gemv_kernel),
        "large-N decode matmul should take the 2-D GEMV path, got {plan:?} / {choice:?}"
    );
    let r = |s: usize, num: usize| -> Vec<f32> {
        (0..num)
            .map(|i| ((i + s) as f32 * 0.001).sin() * 0.1)
            .collect()
    };
    let mut inp = HashMap::new();
    inp.insert(a.id, HostTensor::f32(vec![1, k], r(1, k)));
    inp.insert(wt.id, HostTensor::f32(vec![k, n], r(7, k * n)));
    let eval_inp: HashMap<_, Value> = inp
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_via_contract(&mut gpu, target, &g, &inp);
    assert_eq!(got.shape(), vec![1, n]);
    for (i, (a, c)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap())
        .enumerate()
    {
        assert!(
            (a - c).abs() < 1e-3,
            "lm_head gemv elem {i}: gpu {a} vs cpu {c}"
        );
    }
}

#[test]
fn decode_gemv_chunked_at_bloom_lm_head_n_matches_cpu() {
    // Card 258: BLOOM's tied lm_head has `vocab=250880` (over `DECODE_GEMV_CHUNK_TRIGGER`), which hung real wgpu/RADV hardware
    // at K=1024. This uses the real vocab (N=250880) with a
    // small K (~32MB weight) to check the `Plan::ComputeChunks` split (`decode_gemv_plan`/`kernelgen::gemv_lds`'s
    // `elem_offset`) bit-for-bit against the CPU oracle. N, not K, is the watchdog hazard axis, so it is safe to run
    // unconditionally (unlike the `#[ignore]`d real-BLOOM-scale diagnostics in `cross_backend.rs`).
    use poot_graph_ir::op::OpKind;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (k, n) = (32usize, 250_880usize); // BLOOM's real tied lm_head vocab, a small K
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1, k]));
    let wt = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(a, wt);
    let g = b.finish(out);
    // sanity: this is the decode-GEMV path, over the chunk trigger - plan_eqn must actually take the new
    // ComputeChunks branch, not silently stay a single dispatch.
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .unwrap();
    let (plan, choice) = plan_eqn_with_choice(
        &g,
        eqn,
        poot_target::Backend::SpirvVulkan,
        &poot_test_util::device_caps::default_caps_for(poot_target::Backend::SpirvVulkan),
    )
    .unwrap();
    assert!(
        leaf_choices(&choice).into_iter().all(is_decode_gemv_kernel),
        "large-N decode matmul should take the decode-GEMV path, got {plan:?} / {choice:?}"
    );
    match &plan {
        poot_graph_plan::Plan::ComputeChunks(chunks) => assert!(
            chunks.len() > 1,
            "BLOOM-scale N should split into >1 watchdog-safety chunk"
        ),
        other => panic!("BLOOM-scale N should plan as Plan::ComputeChunks: {other:?}"),
    }
    let r = |s: usize, num: usize| -> Vec<f32> {
        (0..num)
            .map(|i| ((i + s) as f32 * 0.001).sin() * 0.1)
            .collect()
    };
    let mut inp = HashMap::new();
    inp.insert(a.id, HostTensor::f32(vec![1, k], r(1, k)));
    inp.insert(wt.id, HostTensor::f32(vec![k, n], r(7, k * n)));
    let eval_inp: HashMap<_, Value> = inp
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_via_contract(&mut gpu, target, &g, &inp);
    assert_eq!(got.shape(), vec![1, n]);
    assert_close(got.as_f32().unwrap(), cpu.as_f32().unwrap(), 1e-3);
}

#[test]
#[ignore = "perf measurement (card 248): IndexedMatMul at LoRA-scale tiny r vs MoE-scale K; run with \
            --ignored --nocapture --test-threads=1"]
fn card248_indexed_matmul_lora_tiny_r_vs_moe_scale() {
    // Card 248: `lora_linear_batched` (crates/poot-graph-ir/src/ops.rs) dispatches `indexed_matmul` twice per targeted
    // projection: `xa = x[M,in] @ A_stacked[E,in,r]` (K=hidden, N=r) then `xab = xa[M,r] @ B_stacked[E,r,out]` (K=r,
    // N=hidden). The second call's K is the LoRA rank (typically 2-64), far below the MoE `indexed_matmul` K (hundreds to
    // thousands).
    //
    // On wgpu both shapes take `indexed_gemv_lds` (crates/poot-kernelgen/src/gemv.rs); `is_indexed_gemv`
    // (crates/poot-graph-plan/src/predicates.rs) gates only on `out_numel = M*N <= 65535` and backend/dtype. The kernel runs
    // one workgroup of GEMV_WIDTH=128 lanes per output element; each lane strides `j+=128` over K, writes its partial to LDS,
    // and lane 0 serially sums all 128 slots after a barrier, a fixed cost independent of K. For K < 128 the lanes `K..127`
    // do no multiply-adds but the barrier and 128-wide reduction still run. This measures whether that fixed cost shows in
    // wall-clock time, with the same methodology as `igpu_matmul_latency_vs_npu`.
    //
    // Three sweeps, all M*N <= 65535 so the same kernel is used throughout:
    //   1. dispatch-2 shape (K varies, N=896, M=8): per-workgroup lane utilization, K from LoRA rank through the 128 boundary
    //      to MoE-scale K.
    //   2. dispatch-1 shape (K=896, N varies small, M=8): K is large but N=r is tiny, so the workgroup count (M*N) is tiny:
    //      an occupancy question.
    //   3. batch scaling (K=8 vs K=1024, N=896, M varies): does a larger batch narrow the K=8 vs K=1024 gap?
    use std::time::Instant;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };

    let f = |sz: usize, seed: u64| -> Vec<f32> {
        (0..sz)
            .map(|i| {
                (((i as u64)
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed * 40503))
                    % 1009) as f32
                    / 500.0
                    - 1.0
            })
            .collect()
    };
    let build = |m: usize,
                 k: usize,
                 n: usize,
                 e: usize|
     -> (
        poot_graph_ir::Graph,
        HashMap<poot_graph_ir::ValueId, HostTensor>,
    ) {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![m, k]));
        let w = b.constant("w", TensorType::f32(vec![e, k, n]));
        let idx = b.constant("idx", TensorType::f32(vec![m]));
        let out = b.indexed_matmul(x, w, idx);
        let g = b.finish(out);
        let mut inp: HashMap<poot_graph_ir::ValueId, HostTensor> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let name = meta.name.as_deref().unwrap();
            let t = match name {
                "x" => HostTensor::f32(vec![m, k], f(m * k, 1)),
                "w" => HostTensor::f32(vec![e, k, n], f(e * k * n, 2)),
                "idx" => HostTensor::f32(vec![m], (0..m).map(|i| (i % e) as f32).collect()),
                other => panic!("unexpected input {other}"),
            };
            inp.insert(id, t);
        }
        assert!(
            m * n <= 65535,
            "sweep shape M={m} N={n} exceeds the wgpu is_indexed_gemv out_numel cap - would silently \
             switch kernels mid-sweep"
        );
        (g, inp)
    };

    const WARMUP: usize = 5;
    const ITERS: usize = 100;
    let measure = |exec: &mut dyn Executor, m: usize, k: usize, n: usize, e: usize| -> (f64, f64) {
        let (g, inp) = build(m, k, n, e);
        let (exe, entry) = add_resident_entry(exec, target, &g, &inp);
        for _ in 0..WARMUP {
            step_resident(exec, exe, entry, &g);
        }
        let mut lat_us: Vec<f64> = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            step_resident(exec, exe, entry, &g);
            lat_us.push(t.elapsed().as_secs_f64() * 1e6);
        }
        exec.remove_entry(exe, entry).unwrap();
        exec.unload(exe).unwrap();
        lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_us = lat_us[ITERS / 2];
        let ops = 2.0 * m as f64 * k as f64 * n as f64;
        let gflops = ops / (median_us * 1e-6) / 1e9;
        (median_us, gflops)
    };

    // --- Sweep 1: dispatch-2 shape, K varies, N=896 (qwen2.5-0.5b hidden), M=8, E=8 ---
    eprintln!(
        "\n[card248] Sweep 1 (dispatch-2 shape xa@B: K varies, N=896, M=8, E=8) - per-workgroup lane \
         utilization"
    );
    let (m1, n1, e1) = (8usize, 896usize, 8usize);
    let ks1: [(&str, usize); 10] = [
        ("r=2", 2),
        ("r=4", 4),
        ("r=8", 8),
        ("r=16", 16),
        ("r=32", 32),
        ("r=64", 64),
        ("r=128=GEMV_WIDTH", 128),
        ("r=256", 256),
        ("K=896 (qwen2.5-0.5b hidden, dense-equiv)", 896),
        ("K=1024 (mixtral-tiny hidden, MoE-scale)", 1024),
    ];
    let mut sweep1: Vec<(usize, f64, f64)> = Vec::new();
    for (label, k) in ks1 {
        let (median_us, gflops) = measure(&mut gpu, m1, k, n1, e1);
        eprintln!("  K={k:>5} ({label:<42}) median={median_us:>9.1} us  GFLOPS={gflops:>7.3}");
        sweep1.push((k, median_us, gflops));
    }
    let tiny = sweep1.iter().find(|&&(k, ..)| k == 8).unwrap();
    let moe = sweep1.iter().find(|&&(k, ..)| k == 1024).unwrap();
    eprintln!(
        "  => r=8 (LoRA-typical) median {:.1} us vs K=1024 (MoE-scale) median {:.1} us: {:.2}x wall time \
         for {:.0}x fewer FLOPs (K ratio) => {:.1}x lower GFLOPS/s at tiny r",
        tiny.1,
        moe.1,
        tiny.1 / moe.1,
        moe.0 as f64 / tiny.0 as f64,
        moe.2 / tiny.2
    );

    // --- Sweep 2: dispatch-1 shape, K=896 fixed, N varies (=r), M=8, E=8 ---
    eprintln!(
        "\n[card248] Sweep 2 (dispatch-1 shape x@A: K=896 fixed, N=r varies, M=8, E=8) - total workgroup \
         count (M*N) occupancy"
    );
    let ns2: [usize; 6] = [2, 8, 16, 32, 64, 896];
    for n in ns2 {
        let (median_us, gflops) = measure(&mut gpu, m1, 896, n, e1);
        eprintln!(
            "  N={n:>4} (workgroups={:>6}) median={median_us:>9.1} us  GFLOPS={gflops:>7.3}",
            m1 * n
        );
    }

    // --- Sweep 3: batch-size scaling at K=8 (LoRA-typical) vs K=1024 (MoE-scale), N=896, E=8 ---
    eprintln!(
        "\n[card248] Sweep 3 (batch scaling: N=896, E=8, K=8 vs K=1024) - does a bigger concurrently-served \
         batch narrow the gap?"
    );
    let ms3: [usize; 4] = [1, 8, 32, 64]; // M*896 <= 65535 for every M here (cap at M=64)
    for m in ms3 {
        let (lo_us, lo_gf) = measure(&mut gpu, m, 8, 896, 8);
        let (hi_us, hi_gf) = measure(&mut gpu, m, 1024, 896, 8);
        eprintln!(
            "  M={m:>3}: K=8 median={lo_us:>9.1} us ({lo_gf:>6.3} GFLOPS)   K=1024 median={hi_us:>9.1} us \
             ({hi_gf:>7.3} GFLOPS)   ratio={:.2}x",
            lo_us / hi_us
        );
    }
}

#[test]
#[ignore = "perf measurement (card 248): device-time (GPU timestamp) breakdown, isolating host-submit \
            overhead from actual kernel device time at LoRA-scale vs MoE-scale K; run with --ignored \
            --nocapture --test-threads=1"]
fn card248_indexed_matmul_device_time_lora_vs_moe_scale() {
    // The sweep above (card248_indexed_matmul_lora_tiny_r_vs_moe_scale) times one replayed `step` per
    // shape, so each measurement includes a host submit and wait that a real decode step (one command
    // buffer, one submit+wait per token) amortizes across many dispatches. This uses
    // `WgpuDevice::new_with_timing` (Card 552's typed per-dispatch device-time path, the contract
    // replacement for `GpuExecutor::new_profiled`/`profiler`/`profile_report`) to isolate on-device
    // kernel time at sweep 1's two decisive shapes: K=8 (LoRA-typical rank) and K=1024 (mixtral-tiny
    // MoE hidden), M=8, N=896, E=8.
    use std::time::Instant;
    let _gpu_guard = gpu_lock();
    let Some(device) = poot_test_util::device_skip::open_or_skip(
        poot_runtime_common::DeviceBackend::Wgpu,
        poot_gpu::device::WgpuDevice::new_with_timing(4),
    ) else {
        return;
    };
    let target = device.target();
    let mut gpu = Engine::new(device);
    let f = |sz: usize, seed: u64| -> Vec<f32> {
        (0..sz)
            .map(|i| {
                (((i as u64)
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed * 40503))
                    % 1009) as f32
                    / 500.0
                    - 1.0
            })
            .collect()
    };
    let e = 8usize;
    let build = |m: usize,
                 k: usize,
                 n: usize|
     -> (
        poot_graph_ir::Graph,
        HashMap<poot_graph_ir::ValueId, HostTensor>,
    ) {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![m, k]));
        let w = b.constant("w", TensorType::f32(vec![e, k, n]));
        let idx = b.constant("idx", TensorType::f32(vec![m]));
        let out = b.indexed_matmul(x, w, idx);
        let g = b.finish(out);
        let mut inp: HashMap<poot_graph_ir::ValueId, HostTensor> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let name = meta.name.as_deref().unwrap();
            let t = match name {
                "x" => HostTensor::f32(vec![m, k], f(m * k, 1)),
                "w" => HostTensor::f32(vec![e, k, n], f(e * k * n, 2)),
                "idx" => HostTensor::f32(vec![m], (0..m).map(|i| (i % e) as f32).collect()),
                other => panic!("unexpected input {other}"),
            };
            inp.insert(id, t);
        }
        (g, inp)
    };
    let iters = 50u32;
    // Only one real dispatch per step (the graph is a single `indexed_matmul`), so the step's own
    // `sum_of_dispatch_durations` (Card 552) is that kernel's device time - no per-name filter needed.
    let measure = |exec: &mut dyn Executor, m: usize, k: usize, n: usize| -> (f64, f64) {
        let (g, inp) = build(m, k, n);
        let (exe, entry) = add_resident_entry(exec, target, &g, &inp);
        step_resident(exec, exe, entry, &g); // warmup: records the entry
        let t0 = Instant::now();
        let mut dev_total = std::time::Duration::ZERO;
        let empty = poot_executor::StepInputs::new();
        for _ in 0..iters {
            let mut out = exec
                .step(exe, entry, &empty, &mut poot_executor::NoSync)
                .expect("timed");
            if let poot_executor::DeviceTime::Measured(m) = out.device_time() {
                dev_total += m.sum_of_dispatch_durations.unwrap_or_default();
            }
            out.read().expect("timed read");
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let dev_ms = dev_total.as_secs_f64() * 1e3 / iters as f64;
        exec.remove_entry(exe, entry).unwrap();
        exec.unload(exe).unwrap();
        (wall_ms, dev_ms)
    };

    eprintln!("\n[card248_indexed_matmul_device_time_lora_vs_moe_scale]");

    // Dispatch-2 shape (xa@B: K varies, N=hidden=896 fixed, M=8, E=8).
    eprintln!("  -- dispatch-2 shape (K varies, N=896, M=8) --");
    let (wall_lo, dev_lo) = measure(&mut gpu, 8, 8, 896);
    let (wall_hi, dev_hi) = measure(&mut gpu, 8, 1024, 896);
    eprintln!(
        "  K=8    (LoRA-typical): wall {wall_lo:.4} ms/op (host submit+wait), device {dev_lo:.4} ms/op \
         (GPU timestamp)"
    );
    eprintln!(
        "  K=1024 (MoE-scale):    wall {wall_hi:.4} ms/op (host submit+wait), device {dev_hi:.4} ms/op \
         (GPU timestamp)"
    );
    let host_frac_lo = (1.0 - dev_lo / wall_lo) * 100.0;
    eprintln!(
        "  => at K=8: {host_frac_lo:.1}% of WALL time is host-submit overhead, not device execution \
         ({wall_lo:.4} ms wall - {dev_lo:.4} ms device = {:.4} ms host-side)",
        wall_lo - dev_lo
    );
    let expected_linear_dev_lo = dev_hi * (8.0 / 1024.0);
    eprintln!(
        "  => DEVICE time only: K=8 is {dev_lo:.4} ms; pure-FLOP-linear scaling from K=1024's {dev_hi:.4} \
         ms would predict {expected_linear_dev_lo:.4} ms (dev_hi * 8/1024) - measured is {:.2}x that \
         prediction, i.e. a real but modest fixed device-side floor (~{:.4} ms) beyond pure compute scaling",
        dev_lo / expected_linear_dev_lo,
        dev_lo - expected_linear_dev_lo
    );

    // Dispatch-1 shape (x@A: K=hidden=896, N=r tiny vs N=896, M=8, E=8): does device time also floor out when the workgroup
    // count (M*N) is tiny though K is large and every lane is used?
    eprintln!("  -- dispatch-1 shape (K=896 fixed, N varies, M=8) --");
    let (wall_n8, dev_n8) = measure(&mut gpu, 8, 896, 8);
    let (wall_n896, dev_n896) = measure(&mut gpu, 8, 896, 896);
    eprintln!("  N=8   (r=8, workgroups=64):   wall {wall_n8:.4} ms/op, device {dev_n8:.4} ms/op");
    eprintln!(
        "  N=896 (dense-equiv, workgroups=7168): wall {wall_n896:.4} ms/op, device {dev_n896:.4} ms/op"
    );
    let expected_linear_dev_n8 = dev_n896 * (64.0 / 7168.0);
    eprintln!(
        "  => DEVICE time only: N=8 is {dev_n8:.4} ms; pure-workgroup-count-linear scaling from N=896's \
         {dev_n896:.4} ms would predict {expected_linear_dev_n8:.4} ms (dev_n896 * 64/7168) - measured is \
         {:.2}x that prediction",
        dev_n8 / expected_linear_dev_n8
    );
}

#[test]
fn concat_n_gpu_matches_cpu() {
    // A >2-input concat lowers to the general GPU kernel (`kernelgen::concat_n_dt`) instead of the host fallback. Three inputs
    // of different axis-1 sizes (2,1,3) concatenated along axis 1 -> [2,6,3] exercise the segment-selection guard chain and
    // per-input strides. Asserts the plan is a Compute kernel (not Host, else GPU==CPU would pass via the host path) and the
    // output equals CPU eval.
    use poot_graph_ir::op::OpKind;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let b = Builder::new();
    let p0 = b.constant("p0", TensorType::f32(vec![2, 2, 3]));
    let p1 = b.constant("p1", TensorType::f32(vec![2, 1, 3]));
    let p2 = b.constant("p2", TensorType::f32(vec![2, 3, 3]));
    let out = b.concat(1, &[p0, p1, p2]);
    let (i0, i1, i2) = (p0.id, p1.id, p2.id);
    let g = b.finish(out);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::Concat { .. }))
        .unwrap();
    let (plan, _) = plan_eqn_with_choice(
        &g,
        eqn,
        poot_target::Backend::SpirvVulkan,
        &poot_test_util::device_caps::default_caps_for(poot_target::Backend::SpirvVulkan),
    )
    .unwrap();
    assert!(
        matches!(plan, poot_graph_plan::Plan::Compute { .. }),
        ">2-input concat must lower to a kernel, not the host fallback"
    );
    let mk = |off: f32, n: usize| -> Vec<f32> { (0..n).map(|i| i as f32 + off).collect() };
    let mut inputs = HashMap::new();
    inputs.insert(i0, HostTensor::f32(vec![2, 2, 3], mk(0.0, 12)));
    inputs.insert(i1, HostTensor::f32(vec![2, 1, 3], mk(100.0, 6)));
    inputs.insert(i2, HostTensor::f32(vec![2, 3, 3], mk(200.0, 18)));
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
    assert_eq!(got.shape(), vec![2, 6, 3]);
    assert_eq!(got.shape(), cpu.shape());
    for (i, (x, y)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!((x - y).abs() < 1e-6, "concatN elem {i}: gpu {x} vs cpu {y}");
    }
}

#[test]
fn concat_n_rank4_axis2_gpu_matches_cpu() {
    // The GDN chunk-output concat at L>chunk (card 158 follow-up): N=5 tensors [1,4,32,8] concatenated along axis 2 ->
    // [1,4,160,8]. `concat_n_gpu_matches_cpu` covers N=3/rank-3/axis-1; this covers N=5, rank 4, a middle axis, and the deeper
    // segment-guard chain.
    use poot_graph_ir::op::OpKind;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let b = Builder::new();
    let n = 5usize;
    let ps: Vec<_> = (0..n)
        .map(|k| b.constant(&format!("p{k}"), TensorType::f32(vec![1, 4, 32, 8])))
        .collect();
    let out = b.concat(2, &ps);
    let ids: Vec<_> = ps.iter().map(|p| p.id).collect();
    let g = b.finish(out);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::Concat { .. }))
        .unwrap();
    let (plan, _) = plan_eqn_with_choice(
        &g,
        eqn,
        poot_target::Backend::SpirvVulkan,
        &poot_test_util::device_caps::default_caps_for(poot_target::Backend::SpirvVulkan),
    )
    .unwrap();
    assert!(
        matches!(plan, poot_graph_plan::Plan::Compute { .. }),
        ">2-input concat must lower to a kernel"
    );
    let mut inputs = HashMap::new();
    for (k, id) in ids.iter().enumerate() {
        let base = (k as f32) * 1000.0;
        let data: Vec<f32> = (0..(4 * 32 * 8)).map(|i| base + i as f32).collect();
        inputs.insert(*id, HostTensor::f32(vec![1, 4, 32, 8], data));
    }
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
    assert_eq!(got.shape(), vec![1, 4, 160, 8]);
    assert_eq!(got.shape(), cpu.shape());
    for (i, (x, y)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (x - y).abs() < 1e-6,
            "concatN rank4 axis2 elem {i}: gpu {x} vs cpu {y}"
        );
    }
}

#[test]
fn indexed_matmul_gpu_matches_cpu() {
    // The `IndexedMatMul` graph op (gather-free MoE GEMM) lowers on the GPU and matches the CPU oracle.
    // out[m,n] = sum_k x[m,k] * W[idx[m],k,n]; the two rows route to experts 2 and 0, so each contracts against a different
    // expert block. Tracer-emittable op, plan route and eager eval agree (one definition, both backends).
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (e, m, k, n) = (3usize, 2usize, 4usize, 3usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![e, k, n]));
    let idx = b.constant("idx", TensorType::f32(vec![m]));
    let out = b.indexed_matmul(x, w, idx);
    let (xi, wi, ii) = (x.id, w.id, idx.id);
    let g = b.finish(out);

    let xd: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.3 - 1.0).collect();
    let wd: Vec<f32> = (0..e * k * n).map(|i| (i as f32) * 0.07 - 1.0).collect();
    let idxd: Vec<f32> = vec![2.0, 0.0];
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![m, k], xd));
    inputs.insert(wi, HostTensor::f32(vec![e, k, n], wd));
    inputs.insert(ii, HostTensor::f32(vec![m], idxd));

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
    assert_eq!(got.shape(), vec![m, n]);
    assert_eq!(got.shape(), cpu.shape());
    for (i, (a, b)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-5,
            "indexed_matmul elem {i}: gpu {a} vs cpu {b}"
        );
    }
}

#[test]
fn indexed_gemv_lds_large_matches_cpu() {
    // The indexed LDS-GEMV at a realistic K, exercising the K-loop, the LDS reduction and the per-row expert offset (the small
    // `indexed_matmul_gpu_matches_cpu`, K=4, does not). Relative tolerance: the gemv reduces in a different order than eval's
    // serial sum, so they agree to ~f32 reduction-order precision, not bit-exactly.
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (e, m, k, n) = (4usize, 3usize, 256usize, 48usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![e, k, n]));
    let idx = b.constant("idx", TensorType::f32(vec![m]));
    let out = b.indexed_matmul(x, w, idx);
    let (xi, wi, ii) = (x.id, w.id, idx.id);
    let g = b.finish(out);

    let xd: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.05 - 0.3).collect();
    let wd: Vec<f32> = (0..e * k * n)
        .map(|i| ((i % 11) as f32) * 0.04 - 0.2)
        .collect();
    let idxd: Vec<f32> = vec![2.0, 0.0, 3.0]; // distinct experts, incl. the last
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![m, k], xd));
    inputs.insert(wi, HostTensor::f32(vec![e, k, n], wd));
    inputs.insert(ii, HostTensor::f32(vec![m], idxd.clone()));

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
    assert_eq!(got.shape(), vec![m, n]);
    for (i, (a, b)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() <= 1e-3 * b.abs() + 1e-4,
            "indexed_gemv elem {i} (row {}, expert {}): gpu {a} vs cpu {b}",
            i / n,
            idxd[i / n]
        );
    }
}

#[test]
fn tiled_gemm_gpu_matches_cpu() {
    // The wgpu prefill tiled GEMM (`kernelgen::tiled_gemm_dt`, routed by `is_tiled_gemm` for M>1 shared-weight matmuls). Dims
    // are not multiples of the 16-wide tile (M=18, K=33, N=20), so every boundary guard (row<M, col<N, k<K) is exercised in the
    // partial last tile of each dim. Must equal the CPU eval oracle.
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (m, k, n) = (18usize, 33usize, 20usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(x, w);
    let (xi, wi) = (x.id, w.id);
    let g = b.finish(out);

    // small, well-conditioned values so the f32 dot is near-exact.
    let xd: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
    let wd: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.08 - 0.16).collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![m, k], xd));
    inputs.insert(wi, HostTensor::f32(vec![k, n], wd));

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
    assert_eq!(got.shape(), vec![m, n]);
    assert_eq!(got.shape(), cpu.shape());
    for (i, (a, b)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-5,
            "tiled_gemm elem {i} (row {}, col {}): gpu {a} vs cpu {b}",
            i / n,
            i % n
        );
    }
}

#[test]
fn tiled_gemm_random_shapes_gpu_matches_cpu() {
    // The tiled GEMM (`kernelgen::tiled_gemm_dt`) is the most intricate wgpu kernel (LDS tiles, an in-loop barrier,
    // 2-rows-per-lane coarsening) and the prefill workhorse; it had a real RADV miscompile (card 095), and its partial-tile
    // handling (row<M, col<N, k<K guards) is covered by one fixed shape. Fuzz random M/K/N (mostly partial tiles in all three
    // dims, under the 2^15-tile gate so the tiled path is taken, asserted via the compiled kernel choice) and require
    // `gpu.run` == CPU eval. Half the seeds trace a bias linear, which `compile` fuses into `MatMulBias` and the planner
    // lowers to the imported tiled GEMM with the bias epilogue; the other half are bare matmuls, which the planner lowers
    // to the generated tiled GEMM (Card 557). Small well-conditioned values keep the f32 dot near-exact.
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::{CompileOptions, FusionPolicy, Plan, Submission, compile};
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let fill = |seed: u64, count: usize| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 0.4 - 0.2
            })
            .collect()
    };
    let mut tiled_count = 0usize;
    for seed in 0u64..40 {
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
        let m = 2 + (rng() % 39) as usize; // 2..40 (M>1 for the tiled path)
        let k = 1 + (rng() % 40) as usize; // 1..40
        let n = 1 + (rng() % 40) as usize; // 1..40
        let with_bias = rng().is_multiple_of(2);
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![m, k]));
        let w = b.constant("w", TensorType::f32(vec![k, n]));
        let mut inputs = HashMap::new();
        inputs.insert(x.id, HostTensor::f32(vec![m, k], fill(seed * 7 + 1, m * k)));
        inputs.insert(w.id, HostTensor::f32(vec![k, n], fill(seed * 7 + 2, k * n)));
        let out = if with_bias {
            let bias = b.constant("bias", TensorType::f32(vec![n]));
            inputs.insert(bias.id, HostTensor::f32(vec![n], fill(seed * 7 + 3, n)));
            poot_graph_ir::ops::linear(&b, x, w, Some(bias))
        } else {
            b.matmul(x, w)
        };
        let g = b.finish(out);
        // require the tiled path (else we'd be fuzzing the naive kernel, not the tiled one under test), on the
        // graph `compile` actually plans.
        let program = compile(
            &g,
            &spirv_fixture_target(),
            &CompileOptions {
                execution: Submission::Replay,
                fusion: FusionPolicy::Full,
                limits: poot_graph_plan::CompileLimits::STANDARD,
            },
        )
        .unwrap();
        let (eqn, plan) = program
            .planned()
            .find(|(e, _)| matches!(e.op, OpKind::MatMul | OpKind::MatMulBias))
            .unwrap();
        let choice = program.kernel_choice(eqn);
        if with_bias {
            assert!(
                matches!(eqn.op, OpKind::MatMulBias),
                "seed {seed}: compile should fuse the bias add into MatMulBias"
            );
            // The non-chunked imported tiled GEMM is dispatched as `ComputeMeta` (the `[M,K,N]` dims buffer).
            match plan {
                Plan::ComputeMeta { meta, .. } => assert_eq!(
                    meta,
                    &vec![m as u32, k as u32, n as u32, 0],
                    "seed {seed}: dims [M,K,N]"
                ),
                _ => panic!("seed {seed}: non-chunked tiled GEMM should be a ComputeMeta plan"),
            }
            assert!(
                is_tiled_gemm_kernel(choice),
                "seed {seed} (M={m},K={k},N={n}): expected the imported tiled path, got {choice:?}"
            );
        } else {
            assert!(
                matches!(
                    choice,
                    KernelChoice::Generated(KernelRequest::Contraction(
                        ContractionSpec::TiledRegion { .. }
                    ))
                ),
                "seed {seed} (M={m},K={k},N={n}): expected the generated tiled path, got {choice:?}"
            );
        }
        tiled_count += 1;

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
        assert_eq!(got.shape(), cpu.shape(), "seed {seed}: shape");
        for (i, (a, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (a - c).abs() < 1e-4,
                "seed {seed} (M={m},K={k},N={n}) elem {i} (row {}, col {}): gpu {a} vs cpu {c}",
                i / n,
                i % n
            );
        }
    }
    eprintln!(
        "tiled-gemm shape fuzz: {tiled_count}/40 random shapes took the tiled path, all GPU==CPU"
    );
    assert_eq!(
        tiled_count, 40,
        "all small M>1 shapes should take the tiled path"
    );
}

#[test]
fn synthesized_tiled_gemm_choice_matches_cpu() {
    // A bare M>1 MatMul compiles to the synthesized element-by-element tiled GEMM (the default): confirm the planner's
    // recorded choice is the generated tiled-GEMM request and the path runs GPU==CPU. The planner's contraction choice
    // owns the tiling decision (Card 557), and the Body is generated by `kg::tiled_region`, not hand-authored.
    use poot_graph_ir::op::OpKind;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let fill = |seed: u64, count: usize| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 0.4 - 0.2
            })
            .collect()
    };
    for (seed, (m, k, n)) in [(2usize, 16usize, 8usize), (17, 32, 24), (6, 40, 33)]
        .into_iter()
        .enumerate()
    {
        let seed = seed as u64;
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![m, k]));
        let w = b.constant("w", TensorType::f32(vec![k, n]));
        let mut inputs = HashMap::new();
        inputs.insert(x.id, HostTensor::f32(vec![m, k], fill(seed * 7 + 1, m * k)));
        inputs.insert(w.id, HostTensor::f32(vec![k, n], fill(seed * 7 + 2, k * n)));
        let out = b.matmul(x, w);
        let g = b.finish(out);

        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::MatMul)
        });
        assert!(
            matches!(
                choice,
                KernelChoice::Generated(KernelRequest::Contraction(
                    ContractionSpec::TiledRegion { .. }
                ))
            ),
            "expected the synthesized tiled_region choice, got {choice:?}"
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
        assert_eq!(got.shape(), cpu.shape());
        for (a, c) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
        {
            assert!(
                (a - c).abs() < 1e-4,
                "M={m} K={k} N={n}: gpu {a} vs cpu {c}"
            );
        }
    }
    eprintln!("synthesized tiled GEMM (the planner's contraction choice) runs GPU==CPU");
}

#[test]
fn chunked_large_tiled_gemm_gpu_matches_cpu() {
    // Card 096: a tiled GEMM whose tile count crosses the RADV 2^15-workgroup miscompile threshold is dispatched as several
    // sub-2^15 chunks (each a baked-offset kernel variant covering a disjoint tile range) instead of the slow naive kernel.
    // M=544 (34 row-tiles, 2*GEMM_TILE=16 rows each) x N=8192 (1024 col-tiles, GEMM_TILE=8 cols each) = 34816 tiles > 32768
    // -> 2 chunks. Assert the plan is ComputeChunks and `gpu.run == CPU eval` (the chunks reconstruct the full result).
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::Plan;
    use poot_target::Backend;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (m, k, n) = (544usize, 32usize, 8192usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(x, w);
    let g = b.finish(out);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .unwrap();
    let n_chunks = match plan_eqn_with_choice(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
    )
    .unwrap()
    {
        (Plan::ComputeChunks(c), _) => c.len(),
        _ => panic!("expected a ComputeChunks plan for a 34816-tile GEMM"),
    };
    assert_eq!(n_chunks, 2, "34816 tiles / 32767-per-chunk = 2 chunks");

    let xd: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
    let wd: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.08 - 0.16).collect();
    let mut inputs = HashMap::new();
    inputs.insert(x.id, HostTensor::f32(vec![m, k], xd));
    inputs.insert(w.id, HostTensor::f32(vec![k, n], wd));
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
    assert_eq!(got.shape(), vec![m, n]);
    for (i, (a, c)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (a - c).abs() < 1e-4,
            "elem {i} (row {}, col {}): gpu {a} vs cpu {c}",
            i / n,
            i % n
        );
    }
    let max_abs = max_abs_error(got.as_f32().unwrap(), cpu.as_f32().unwrap());
    // Card 546b: the old host-roundtrip `run()` and device-resident `run_resident_kv` were two
    // separate executor code paths, so a >= 2^15-tile chunked GEMM needed checking on both. The
    // contract has one execution path (`add_entry`/`step`, via `run_via_contract` above), already
    // GPU-resident across replays, so there is no second path left to separately re-check here.
    eprintln!(
        "chunked large tiled GEMM: {m}x{k}x{n} = 34816 tiles -> 2 chunks; GPU==CPU (max_abs {max_abs:.2e})"
    );
}

#[test]
fn batched_tiled_gemm_random_shapes_gpu_matches_cpu() {
    // Batched companion to the tiled-GEMM shape fuzzer (0235): `tiled_gemm_dt` with b_count>1, the MoE-per-expert and
    // prefill-attention workhorse, whose per-batch stride/offset logic the single-GEMM fuzzer never exercises. 32 random seeds
    // cover rank-3 `[E,M,K]@[E,K,N]` and rank-4 `[B,H,M,K]@[B,H,K,N]` with random batch counts and mostly-partial-tile M/K/N
    // (under the 2^15-tile gate). Each asserts the kernel choice is the batched tiled GEMM (so the batched tiled kernel is under
    // test) and `gpu.run == CPU eval`.
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::Plan;
    use poot_target::Backend;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let fill = |seed: u64, count: usize| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 0.4 - 0.2
            })
            .collect()
    };
    let mut batched_count = 0usize;
    for seed in 0u64..32 {
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
        let rank4 = rng().is_multiple_of(2);
        let m = 2 + (rng() % 18) as usize; // 2..20 (M>1)
        let k = 1 + (rng() % 34) as usize; // 1..35
        let n = 1 + (rng() % 18) as usize; // 1..20
        // batch dims (product > 1 so the BATCHED path is taken, not the rank-2 tiled).
        let (xshape, wshape) = if rank4 {
            let bd = 1 + (rng() % 2) as usize; // 1..2
            let hd = 2 + (rng() % 3) as usize; // 2..4 -> B*H in 2..8
            (vec![bd, hd, m, k], vec![bd, hd, k, n])
        } else {
            let e = 2 + (rng() % 4) as usize; // 2..5 experts
            (vec![e, m, k], vec![e, k, n])
        };
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(xshape.clone()));
        let w = b.constant("w", TensorType::f32(wshape.clone()));
        let out = b.matmul(x, w);
        let mut inputs = HashMap::new();
        inputs.insert(
            x.id,
            HostTensor::f32(xshape.clone(), fill(seed * 7 + 1, xshape.iter().product())),
        );
        inputs.insert(
            w.id,
            HostTensor::f32(wshape.clone(), fill(seed * 7 + 2, wshape.iter().product())),
        );
        let g = b.finish(out);
        let eqn = g
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul))
            .unwrap();
        // the batched-weight tiled GEMM is the imported kernel, dispatched as `ComputeMeta` (the `[E,M,K,N]` dims buffer) rather than a kernelgen `Compute`.
        let choice = match plan_eqn_with_choice(
            &g,
            eqn,
            Backend::SpirvVulkan,
            &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
        )
        .unwrap()
        {
            (Plan::ComputeMeta { meta, .. }, choice) => {
                assert_eq!(meta.len(), 4, "seed {seed}: batched dims [E,M,K,N]");
                choice
            }
            _ => panic!("seed {seed}: batched tiled GEMM should be a ComputeMeta plan"),
        };
        assert!(
            matches!(
                choice,
                KernelChoice::Imported {
                    kernel: ImportedKernel::TiledGemmBatched,
                    ..
                }
            ),
            "seed {seed} (rank4={rank4}, x={xshape:?}): expected the batched tiled path, got {choice:?}"
        );
        batched_count += 1;

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
        assert_eq!(got.shape(), cpu.shape(), "seed {seed}: shape");
        for (i, (a, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (a - c).abs() <= 1e-3 * c.abs() + 1e-4,
                "seed {seed} (x={xshape:?}) elem {i}: gpu {a} vs cpu {c}"
            );
        }
    }
    eprintln!(
        "batched-tiled-gemm shape fuzz: {batched_count}/32 random shapes took the batched tiled path, all GPU==CPU"
    );
    assert_eq!(
        batched_count, 32,
        "all batch>1 shapes should take the batched tiled path"
    );
}

#[test]
fn tiled_gemm_bias_gpu_matches_cpu() {
    // The tiled GEMM with the fused bias epilogue (`MatMulBias`, the q/k/v projections at M>1), formed by `compile` from a
    // traced bias linear. Tile-unaligned (M=20, K=16, N=33): K is exactly one tile, M and N straddle.
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (m, k, n) = (20usize, 16usize, 33usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let bias = b.constant("bias", TensorType::f32(vec![n]));
    let out = poot_graph_ir::ops::linear(&b, x, w, Some(bias));
    let (xi, wi, bi) = (x.id, w.id, bias.id);
    let g = b.finish(out);
    let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
        matches!(op, OpKind::MatMulBias)
    });
    assert!(
        leaf_choices(&choice).into_iter().all(is_tiled_gemm_kernel),
        "the fused bias linear should take the imported tiled GEMM: {choice:?}"
    );

    let xd: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32) * 0.05 - 0.2).collect();
    let wd: Vec<f32> = (0..k * n).map(|i| ((i % 6) as f32) * 0.07 - 0.14).collect();
    let bd: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.1).collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![m, k], xd));
    inputs.insert(wi, HostTensor::f32(vec![k, n], wd));
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
    assert_eq!(got.shape(), vec![m, n]);
    assert_eq!(got.shape(), cpu.shape());
    for (i, (a, b)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-5,
            "tiled_gemm_bias elem {i} (row {}, col {}): gpu {a} vs cpu {b}",
            i / n,
            i % n
        );
    }
}

#[test]
fn batched_tiled_gemm_gpu_matches_cpu() {
    // The batched tiled GEMM (`tiled_gemm_dt` with b_count>1, routed by `is_batched_tiled_gemm`): the MoE-prefill per-expert
    // GEMM A[E,M,K] @ B[E,K,N] -> [E,M,N]. Dims are tile-unaligned (E=3, M=18, K=33, N=20) so the per-batch offsets and the
    // boundary masks are exercised.
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (e, m, k, n) = (3usize, 18usize, 33usize, 20usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![e, m, k]));
    let w = b.constant("w", TensorType::f32(vec![e, k, n]));
    let out = b.matmul(x, w);
    let (xi, wi) = (x.id, w.id);
    let g = b.finish(out);

    let xd: Vec<f32> = (0..e * m * k)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();
    let wd: Vec<f32> = (0..e * k * n)
        .map(|i| ((i % 5) as f32) * 0.08 - 0.16)
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![e, m, k], xd));
    inputs.insert(wi, HostTensor::f32(vec![e, k, n], wd));

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
    assert_eq!(got.shape(), vec![e, m, n]);
    for (idx, (a, bb)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (a - bb).abs() <= 1e-3 * bb.abs() + 1e-4,
            "batched_tiled_gemm elem {idx} (batch {}, row {}, col {}): gpu {a} vs cpu {bb}",
            idx / (m * n),
            (idx % (m * n)) / n,
            idx % n
        );
    }
}

#[test]
fn rank4_batched_tiled_gemm_gpu_matches_cpu() {
    // The prefill attention shape: a rank-4 batched GEMM A[B,H,M,K] @ B[B,H,K,N] -> [B,H,M,N], batched over B*H heads (the
    // `is_batched_tiled_gemm` rank-N generalization routes it through the tiled kernel with b_count = B*H). Tile-unaligned dims
    // (B=2, H=3 -> 6 batches, M=7, K=33, N=20) exercise the per-batch offsets and boundary masks. Mirrors `Q @ K^T` /
    // `scores @ V` (the K^T is a separate Transpose in-graph, so the MatMul itself is plain).
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (bb, h, m, k, n) = (2usize, 3usize, 7usize, 33usize, 20usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![bb, h, m, k]));
    let w = b.constant("w", TensorType::f32(vec![bb, h, k, n]));
    let out = b.matmul(x, w);
    let (xi, wi) = (x.id, w.id);
    let g = b.finish(out);

    let xd: Vec<f32> = (0..bb * h * m * k)
        .map(|i| ((i % 11) as f32) * 0.07 - 0.35)
        .collect();
    let wd: Vec<f32> = (0..bb * h * k * n)
        .map(|i| ((i % 5) as f32) * 0.08 - 0.16)
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![bb, h, m, k], xd));
    inputs.insert(wi, HostTensor::f32(vec![bb, h, k, n], wd));

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
    assert_eq!(got.shape(), vec![bb, h, m, n]);
    for (idx, (a, c)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (a - c).abs() <= 1e-3 * c.abs() + 1e-4,
            "rank4_batched_tiled_gemm elem {idx} (batch {}, row {}, col {}): gpu {a} vs cpu {c}",
            idx / (m * n),
            (idx % (m * n)) / n,
            idx % n
        );
    }
}

#[test]
fn nvptx_prefill_tiled_gemm_plan_selection() {
    // Card 557: the M>1 shared-weight F32 projection matmul takes the generated tiled GEMM (`kg::tiled_region`, a
    // single `Plan::Compute`) on NVPTX as on wgpu. The chunked case (a large GEMM that would trip the RADV 2^15 cap on
    // wgpu) stays a single dispatch on NVPTX: NVPTX has no cap, and the PTX executor `unreachable!`s on ComputeChunks.
    // Pure planner test (no GPU).
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::Plan;
    use poot_target::Backend;
    let plan_for = |m: usize, n: usize, backend| {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![m, 64]));
        let w = b.constant("w", TensorType::f32(vec![64, n]));
        let out = b.matmul(x, w);
        let g = b.finish(out);
        let eqn = g
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul))
            .unwrap();
        plan_eqn_with_choice(
            &g,
            eqn,
            backend,
            &poot_test_util::device_caps::default_caps_for(backend),
        )
        .unwrap()
    };
    // small M>1 GEMM -> the generated tiled GEMM, one Compute, on both backends.
    let is_tiled_region = |choice: &KernelChoice| {
        matches!(
            choice,
            KernelChoice::Generated(KernelRequest::Contraction(
                ContractionSpec::TiledRegion { .. }
            ))
        )
    };
    assert!(
        matches!(plan_for(8, 8, Backend::Nvptx), (Plan::Compute { .. }, choice) if is_tiled_region(&choice))
    );
    assert!(
        matches!(plan_for(8, 8, Backend::SpirvVulkan), (Plan::Compute { .. }, choice) if is_tiled_region(&choice))
    );
    // a GEMM large enough to chunk on wgpu: ComputeChunks on SpirvVulkan, but a single Compute on NVPTX.
    let big_n = 8 * 40_000; // groups = ceil(8/16)*ceil(N/8) = 1 * 40000 > 2^15 (RADV chunk threshold)
    assert!(matches!(
        plan_for(8, big_n, Backend::SpirvVulkan),
        (Plan::ComputeChunks(_), KernelChoice::Chunked(_))
    ));
    assert!(
        matches!(plan_for(8, big_n, Backend::Nvptx), (Plan::Compute { .. }, choice) if is_tiled_region(&choice))
    );
}
