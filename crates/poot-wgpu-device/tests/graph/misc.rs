use super::*;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;

// iGPU (wgpu/RADV) f32 matmul latency and GFLOPS probe, comparable to a Peano/IRON AIE-core (XDNA2
// NPU) probe measured previously at the same small shape.
//
// Shapes: small M=32, K=256, N=128 (the NPU's 17.8 GFLOPS reference); prefill M=128, K=896, N=896 (Qwen2.5-0.5B linear
// at L=128). Protocol: 5 warmup dispatches (inputs resident), then 100 timed dispatches via a resident entry. Reports
// min/median/p95/max us and GFLOPS = 2*M*K*N / median_us / 1e3.
//
// Run: cargo test -p poot-wgpu-device igpu_matmul_latency_vs_npu -- --ignored --nocapture --test-threads=1
//
// A sub-0.05s pass without output means no Vulkan adapter was found and the test skipped.

#[test]
fn vram_budget_bytes_reports_a_plausible_value() {
    // The device VRAM budget via the Vulkan runtime (VK_EXT_memory_budget heap_budget): engine-queried,
    // vendor/OS-agnostic, no sysfs/CLI. Sanity bound only; verifies the wgpu-hal/ash interop works. Tests
    // `poot_wgpu_runtime::Context` directly (card 546b: the `GpuExecutor` pass-throughs are gone with the rest
    // of the pre-contract executor).
    let _gpu_guard = gpu_lock();
    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, poot_wgpu_runtime::Context::new()) else {
        return;
    };
    let total = ctx.vram_budget_bytes().expect(
        "VK_EXT_memory_budget heap_budget should return a value on the RADV Vulkan backend",
    );
    eprintln!(
        "device VRAM budget: {:.1} MiB",
        total as f64 / (1024.0 * 1024.0)
    );
    assert!(
        total > 256 * 1024 * 1024 && total < 256u64 * 1024 * 1024 * 1024,
        "device VRAM budget implausible: {total} bytes"
    );
}

#[test]
fn masked_attention_gpu_matches_cpu() {
    // G3d-1 on the Arc (executor equivalence): masked decode attention over the full fixed-capacity cache
    // runs on poot's GPU kernels and matches the CPU eval. cap=4, mask hides slots 2,3.
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, cap, d) = (2usize, 1usize, 4usize, 4usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
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

    let fill =
        |n: usize, seed: f32| -> Vec<f32> { (0..n).map(|i| (i as f32) * 0.1 + seed).collect() };
    let mut inputs = HashMap::new();
    inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], fill(hq * d, -0.4)));
    inputs.insert(
        ki,
        HostTensor::f32(vec![1, hkv, cap, d], fill(hkv * cap * d, -0.7)),
    );
    inputs.insert(
        vi,
        HostTensor::f32(vec![1, hkv, cap, d], fill(hkv * cap * d, 0.3)),
    );
    inputs.insert(
        mi,
        HostTensor::f32(vec![1, 1, 1, cap], vec![0.0, 0.0, -1.0e9, -1.0e9]),
    );

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
    assert_eq!(got.shape(), cpu.shape());
    for (i, (a, b)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        let tol = 5e-3 * b.abs().max(1e-3);
        assert!((a - b).abs() <= tol, "elem {i}: gpu {a} vs cpu {b}");
    }
}

#[test]
fn swiglu_gpu_matches_cpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let n = 12usize;
    let b = Builder::for_tests();
    let gate = b.constant("gate", TensorType::f32(vec![1, 1, n])).unwrap();
    let up = b.constant("up", TensorType::f32(vec![1, 1, n])).unwrap();
    let out = poot_graph_ir::ops::swiglu(&b, gate, up).unwrap();
    let (gi, ui) = (gate.id(), up.id());
    let g = b.finish(out).unwrap();

    let gd: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 1.5).collect();
    let ud: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 0.2).collect();
    let mut inputs = HashMap::new();
    inputs.insert(gi, HostTensor::f32(vec![1, 1, n], gd));
    inputs.insert(ui, HostTensor::f32(vec![1, 1, n], ud));

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
    for (i, (a, b)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        let tol = 1e-4 * b.abs().max(1e-3);
        assert!((a - b).abs() <= tol, "elem {i}: gpu {a} vs cpu {b}");
    }
}

#[test]
fn fused_gpu_matches_cpu_over_random_graphs() {
    // GPU analogue of poot-eval's fusion fuzzer (oracle stage 2 on the device path): over random pointwise + reduction-region
    // graphs, `gpu.run(fuse(cse(g)))` (which dispatches the synthesized Fused/FusedRow kernels, one per region) must match the
    // CPU eager eval of the unfused graph within a tight f32 tolerance, generalizing the single hand-written
    // `fused_decode_matches_unfused_on_gpu`. Tolerance, not bit-exact: a reduction's GPU sum order differs from the CPU's. The
    // shape varies per seed (row counts and widths, including partial widths like 7/33 that stress the strided reduce loop) and
    // one leaf is a row-broadcast input ([..,1]), exercising the fused kernel's broadcast-stride path. Each seed is one
    // (shape, structure), so the unique-kernel count (hence disk-cached llc compiles) stays about one per seed.
    use poot_graph_ir::op::{BinOp, RedOp, UnOp};
    use poot_graph_plan::{cse, fuse};
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let shapes: [Vec<usize>; 5] = [
        vec![1, 4, 8],
        vec![2, 3, 7], // 6 rows, width 7 (partial workgroup width)
        vec![1, 8, 16],
        vec![1, 6, 33], // width 33 -> a multi-pass strided reduce
        vec![3, 5, 5],
    ];
    let fill = |seed: u64, count: usize| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    };
    let mut fused_total = 0usize;
    for seed in 0u64..30 {
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
        let shape = shapes[(seed as usize) % shapes.len()].clone();
        let axis = shape.len() - 1;
        let numel: usize = shape.iter().product();
        let mut bshape = shape.clone();
        bshape[axis] = 1; // a row-broadcast leaf (constant across the reduced axis).
        let bnumel: usize = bshape.iter().product();
        let b = Builder::for_tests();
        let full: Vec<_> = (0..3)
            .map(|i| {
                b.constant(&format!("x{i}"), TensorType::f32(shape.clone()))
                    .unwrap()
            })
            .collect();
        let bcast = b.constant("xb", TensorType::f32(bshape.clone())).unwrap();
        let mut acc = full[(rng() as usize) % 3];
        let steps = 4 + (rng() % 5) as usize; // 4..8 ops (small -> bounded fused-kernel compiles).
        for _ in 0..steps {
            match rng() % 5 {
                0 => {
                    let op = [UnOp::Neg, UnOp::Tanh, UnOp::Erf][(rng() as usize) % 3];
                    acc = b.unary(op, acc).unwrap();
                }
                1 | 2 => {
                    let other = full[(rng() as usize) % 3];
                    let op = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Max][(rng() as usize) % 4];
                    acc = b.binary(op, acc, other).unwrap();
                }
                3 => {
                    // binary against the row-broadcast leaf (broadcasts to the full row).
                    let op = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Max][(rng() as usize) % 4];
                    acc = b.binary(op, acc, bcast).unwrap();
                }
                _ => {
                    let op = if rng().is_multiple_of(2) {
                        RedOp::Sum
                    } else {
                        RedOp::Max
                    };
                    let r = b.reduce(op, acc, axis, true).unwrap();
                    acc = b.broadcast(r, shape.clone()).unwrap();
                }
            }
        }
        let g = b.finish(acc).unwrap();
        let mut inputs = HashMap::new();
        for (i, t) in full.iter().enumerate() {
            inputs.insert(
                t.id(),
                HostTensor::f32(shape.clone(), fill(seed * 7 + i as u64 + 1, numel)),
            );
        }
        inputs.insert(
            bcast.id(),
            HostTensor::f32(bshape.clone(), fill(seed * 7 + 101, bnumel)),
        );
        let fg = fuse(&cse(&g).unwrap()).unwrap();
        if fg.eqns().len() < g.eqns().len() {
            fused_total += 1;
        }
        let eval_inputs: HashMap<_, Value> = inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect();
        let cpu = eval(&g, &eval_inputs, EvalOptions::default())
            .unwrap()
            .output
            .into_host()
            .unwrap();
        let got = run_via_contract(&mut gpu, target, &fg, &inputs);
        assert_eq!(got.shape(), cpu.shape(), "seed {seed}: shape");
        for (i, (a, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            let tol = 2e-4 * c.abs().max(1e-3);
            assert!(
                (a - c).abs() <= tol,
                "seed {seed} elem {i}: fused-gpu {a} vs cpu {c} ({} -> {} eqns)",
                g.eqns().len(),
                fg.eqns().len()
            );
        }
    }
    eprintln!("fused-gpu fuzz: 30 random graphs, {fused_total} exercised fusion");
    assert!(
        fused_total >= 22,
        "most random graphs should fuse (got {fused_total}/30)"
    );
}

#[test]
fn movement_gpu_matches_cpu_over_random_graphs() {
    // GPU analogue of poot-eval's movement fuzzer (update 0205): the fuzzer above is pointwise + reduction only, so the device
    // movement kernels (reshape/transpose/broadcast/slice/concat) are tested individually but never in random combination with
    // the fusion boundaries. Random graphs interleave movement ops with pointwise, then `gpu.run(fuse(cse(g)))` must match the
    // CPU eager eval within a tight tolerance (pointwise on the GPU may contract to FMA, so not bit-exact). Small shapes and
    // short chains bound the unique-kernel count. Runs only existing kernels, no new SPIR-V.
    use poot_graph_ir::op::{BinOp, UnOp};
    use poot_graph_plan::{cse, fuse};
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    fn numel(s: &[usize]) -> usize {
        s.iter().product()
    }
    let fill = |seed: u64, count: usize| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    };
    let mut had_movement = 0usize;
    for seed in 0u64..24 {
        let mut s = seed.wrapping_mul(0x100000001B3).wrapping_add(0x9E3779B9) | 1;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let b = Builder::for_tests();
        // 2-3 small leaves (rank 2-3, dims 2-4), at least one with a size-1 axis (enables broadcast).
        let mut pool: Vec<(poot_graph_ir::builder::Traced, Vec<usize>)> = Vec::new();
        let mut inputs = HashMap::new();
        let n_leaves = 2 + (rng() % 2) as usize;
        for k in 0..n_leaves {
            let rank = 2 + (rng() % 2) as usize;
            let mut shape: Vec<usize> = (0..rank).map(|_| 2 + (rng() % 3) as usize).collect();
            if k == 0 {
                shape[rng() as usize % rank] = 1; // a broadcastable leaf
            }
            let t = b
                .constant(&format!("x{k}"), TensorType::f32(shape.clone()))
                .unwrap();
            inputs.insert(
                t.id(),
                HostTensor::f32(shape.clone(), fill(seed * 7 + k as u64, numel(&shape))),
            );
            pool.push((t, shape));
        }
        let mut movement_here = false;
        let steps = 3 + (rng() % 3) as usize; // 3..5 ops
        for _ in 0..steps {
            let (v, vs) = pool[rng() as usize % pool.len()].clone();
            let (nv, ns) = match rng() % 7 {
                0 => (b.unary(UnOp::Neg, v).unwrap(), vs.clone()),
                1 => {
                    let partner = pool
                        .iter()
                        .find(|(_, p)| *p == vs)
                        .map(|(t, _)| *t)
                        .unwrap_or(v);
                    let op = [BinOp::Add, BinOp::Sub, BinOp::Mul][rng() as usize % 3];
                    (b.binary(op, v, partner).unwrap(), vs.clone())
                }
                2 => {
                    // reshape to another factorization of the same element count.
                    movement_here = true;
                    let n = numel(&vs);
                    let mut rem = n;
                    let r = 1 + (rng() % 2) as usize;
                    let mut dims = Vec::new();
                    for _ in 0..r {
                        let divs: Vec<usize> =
                            (1..=rem).filter(|d| rem.is_multiple_of(*d)).collect();
                        let d = divs[rng() as usize % divs.len()];
                        dims.push(d);
                        rem /= d;
                    }
                    dims.push(rem.max(1));
                    (b.reshape(v, dims.clone()).unwrap(), dims)
                }
                3 if vs.len() >= 2 => {
                    movement_here = true;
                    // a random permutation.
                    let mut perm: Vec<usize> = (0..vs.len()).collect();
                    for i in (1..perm.len()).rev() {
                        perm.swap(i, rng() as usize % (i + 1));
                    }
                    let ns = perm.iter().map(|&p| vs[p]).collect::<Vec<_>>();
                    (b.transpose(v, perm).unwrap(), ns)
                }
                4 => {
                    // broadcast a size-1 axis, if any.
                    if let Some(ax) = vs.iter().position(|&d| d == 1) {
                        movement_here = true;
                        let mut ns = vs.clone();
                        ns[ax] = 2 + (rng() % 2) as usize;
                        (b.broadcast(v, ns.clone()).unwrap(), ns)
                    } else {
                        (b.unary(UnOp::Neg, v).unwrap(), vs.clone())
                    }
                }
                5 => {
                    movement_here = true;
                    let ax = rng() as usize % vs.len();
                    let len = vs[ax];
                    let start = rng() as usize % len;
                    let end = start + 1 + rng() as usize % (len - start);
                    let mut ns = vs.clone();
                    ns[ax] = end - start;
                    (b.slice(v, ax, start, end).unwrap(), ns)
                }
                _ => {
                    movement_here = true;
                    let ax = rng() as usize % vs.len();
                    let mut ns = vs.clone();
                    ns[ax] = vs[ax] * 2;
                    (b.concat(ax, &[v, v]).unwrap(), ns)
                }
            };
            if numel(&ns) <= 512 {
                pool.push((nv, ns));
            }
        }
        if movement_here {
            had_movement += 1;
        }
        let (out, out_shape) = pool.last().unwrap().clone();
        let g = b.finish(out).unwrap();
        let eval_inputs: HashMap<_, Value> = inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect();
        let cpu = eval(&g, &eval_inputs, EvalOptions::default())
            .unwrap()
            .output
            .into_host()
            .unwrap();
        let got = run_via_contract(&mut gpu, target, &fuse(&cse(&g).unwrap()).unwrap(), &inputs);
        assert_eq!(
            got.shape(),
            cpu.shape(),
            "seed {seed}: shape (out {out_shape:?})"
        );
        for (i, (a, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            let tol = 2e-4 * c.abs().max(1e-3);
            assert!(
                (a - c).abs() <= tol,
                "seed {seed} elem {i}: gpu {a} vs cpu {c} (out shape {out_shape:?})"
            );
        }
    }
    eprintln!("movement-gpu fuzz: 24 random graphs, {had_movement} had a movement op");
    assert!(
        had_movement >= 18,
        "most graphs should contain a movement op (got {had_movement}/24)"
    );
}
