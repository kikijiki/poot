use super::*;
use poot_graph_plan::KernelChoice;
use poot_kernelgen::{KernelRequest, MovementOp, MovementSpec};
use poot_quant::StoredBytes;

/// The generated movement a choice names, and whether its plan is one dispatch with no metadata buffer.
fn movement_of(plan: &poot_graph_plan::Plan, choice: &KernelChoice) -> MovementSpec {
    assert!(
        matches!(single_meta(plan), Some([])),
        "a movement is one dispatch with no metadata buffer: {plan:?}"
    );
    match choice {
        KernelChoice::Generated(KernelRequest::Movement(spec)) => spec.clone(),
        other => panic!("expected the generated movement kernel: {other:?}"),
    }
}

#[test]
fn scatter_axis0_gpu_matches_cpu() {
    // `Scatter` (write-side inverse of axis-0 gather) lowers on the GPU and matches CPU: `out[index[j], :] = src[j, :]` with a
    // reverse-permutation index (a real reorder). Used by argtop-k expert-index inversion.
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (n, rest) = (4usize, 3usize);
    let b = Builder::for_tests();
    let src = b.constant("src", TensorType::f32(vec![n, rest])).unwrap();
    let idx = b
        .constant("idx", TensorType::new(vec![n], poot_tensor::DType::I32))
        .unwrap();
    let out = b.scatter(src, idx).unwrap();
    let (si, ii) = (src.id(), idx.id());
    let g = b.finish(out).unwrap();

    let srcd: Vec<f32> = (0..n * rest).map(|i| i as f32 + 1.0).collect();
    let idxd: Vec<i32> = (0..n).map(|j| (n - 1 - j) as i32).collect(); // reverse permutation
    let mut inputs = HashMap::new();
    inputs.insert(si, HostTensor::f32(vec![n, rest], srcd));
    inputs.insert(ii, HostTensor::i32(vec![n], idxd));

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
        assert!((a - b).abs() < 1e-6, "scatter elem {i}: gpu {a} vs cpu {b}");
    }
    // ground truth: out[n-1-j] = src[j], i.e. the rows reversed.
    assert_eq!(
        &got.as_f32().unwrap()[0..3],
        &[10.0, 11.0, 12.0],
        "out[0] == src[3]"
    );
    assert_eq!(
        &got.as_f32().unwrap()[9..12],
        &[1.0, 2.0, 3.0],
        "out[3] == src[0]"
    );
}

#[test]
fn arg_top_k_gpu_matches_cpu() {
    // Naive scalar `ArgTopK` body (one thread per output element, serial branchless scan over E) lowers on GPU and matches the
    // CPU oracle. Executor equivalence (decomposition correctness is SC-001 in poot-eval); the summation order is fixed, so
    // this is bit-exact.
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (l, e, k) = (3usize, 16usize, 4usize);
    let b = Builder::for_tests();
    let rank = b.constant("rank", TensorType::f32(vec![l, e])).unwrap();
    let out = b.arg_top_k(rank, k).unwrap();
    let ri = rank.id();
    let g = b.finish(out).unwrap();

    // rank[row,i] = (i*7 + row*3) % e: a permutation of 0..e per row (gcd(7,e)=1 for e=16), with a
    // distinct per-row shift so the L rows are not all the same permutation (batching over leading dims
    // is actually exercised, not just repeated).
    let rankd: Vec<f32> = (0..l)
        .flat_map(|row| (0..e).map(move |i| ((i * 7 + row * 3) % e) as f32))
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(ri, HostTensor::f32(vec![l, e], rankd));

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
    assert_eq!(got.shape(), vec![l, k]);
    assert_eq!(
        got.as_i32().unwrap(),
        cpu.as_i32().unwrap(),
        "arg_top_k ids"
    );
    // ground truth for row 0: rank[0,i] = (7*i) % 16 -> rank 0 at i=0, rank 1 at i=7 (7*7=49%16=1),
    // rank 2 at i=14 (7*14=98%16=2), rank 3 at i=5 (7*5=35%16=3).
    assert_eq!(
        &got.as_i32().unwrap()[0..4],
        &[0, 7, 14, 5],
        "row 0 top-4 expert ids"
    );
}

#[test]
fn gather_nonaxis0_gpu_matches_cpu() {
    // Non-axis-0 gather lowers to a GPU kernel instead of the host fallback, keeping the op on-device; on f32 it
    // is the generated movement kernel. Two cases on a [3,4,5] table: (a) axis=1 with a vector index [2] ->
    // out [3,2,5]; (b) axis=2 with a scalar index -> out [3,4]. Each asserts the plan is the generated kernel (not Host, else
    // GPU==CPU would pass via the host path) and output == CPU eval.
    use poot_graph_ir::op::OpKind;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let datad: Vec<f32> = (0..3 * 4 * 5).map(|i| i as f32 * 0.5 - 7.0).collect();

    // (a) axis=1, vector index [2] selecting rows 3 and 1 of the size-4 axis.
    {
        let b = Builder::for_tests();
        let data = b.constant("data", TensorType::f32(vec![3, 4, 5])).unwrap();
        let idx = b
            .constant("idx", TensorType::new(vec![2], poot_tensor::DType::I32))
            .unwrap();
        let out = b.gather(data, 1, idx).unwrap();
        let (di, ii) = (data.id(), idx.id());
        let g = b.finish(out).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::Gather { .. }))
            .unwrap();
        let (plan, choice) = plan_eqn_with_choice(
            &g,
            eqn,
            poot_target::Backend::SpirvVulkan,
            &poot_test_util::device_caps::default_caps_for(poot_target::Backend::SpirvVulkan),
        )
        .unwrap();
        // The gather is the generated movement kernel, not a host fallback.
        let spec = movement_of(&plan, &choice);
        assert!(
            matches!(
                spec.op,
                MovementOp::Gather {
                    axis: 1,
                    index_numel: 2,
                    ..
                }
            ),
            "{spec:?}"
        );
        let mut inputs = HashMap::new();
        inputs.insert(di, HostTensor::f32(vec![3, 4, 5], datad.clone()));
        inputs.insert(ii, HostTensor::i32(vec![2], vec![3, 1]));
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
        assert_eq!(got.shape(), vec![3, 2, 5]);
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
                "axis1 gather elem {i}: gpu {x} vs cpu {y}"
            );
        }
    }

    // (b) axis=2 (last), scalar index -> drops the axis, out [3,4].
    {
        let b = Builder::for_tests();
        let data = b.constant("data", TensorType::f32(vec![3, 4, 5])).unwrap();
        let idx = b
            .constant("idx", TensorType::new(vec![], poot_tensor::DType::I32))
            .unwrap();
        let out = b.gather(data, 2, idx).unwrap();
        let (di, ii) = (data.id(), idx.id());
        let g = b.finish(out).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::Gather { .. }))
            .unwrap();
        let (plan, choice) = plan_eqn_with_choice(
            &g,
            eqn,
            poot_target::Backend::SpirvVulkan,
            &poot_test_util::device_caps::default_caps_for(poot_target::Backend::SpirvVulkan),
        )
        .unwrap();
        // The scalar-index gather over the last axis is the generated movement kernel too.
        let spec = movement_of(&plan, &choice);
        assert!(
            matches!(
                spec.op,
                MovementOp::Gather {
                    axis: 2,
                    index_numel: 1,
                    ..
                }
            ),
            "{spec:?}"
        );
        let mut inputs = HashMap::new();
        inputs.insert(di, HostTensor::f32(vec![3, 4, 5], datad.clone()));
        inputs.insert(ii, HostTensor::i32(vec![], vec![2]));
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
        assert_eq!(got.shape(), vec![3, 4]);
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
                "axis2 gather elem {i}: gpu {x} vs cpu {y}"
            );
        }
    }
}

#[test]
fn gather_nonaxis0_plan_selection() {
    // A non-axis-0 gather is the one generated movement kernel on every backend and dtype: the request
    // names the table's storage (an I32 table moves its words as I32) and the gather axis, and the
    // dispatch binds no metadata buffer. No GPU.
    use poot_graph_ir::op::OpKind;
    use poot_target::Backend;
    use poot_tensor::DType;
    let plan_for = |dt: DType, backend| {
        let b = Builder::for_tests();
        let data = b
            .constant("data", TensorType::new(vec![3, 4, 5], dt))
            .unwrap();
        let idx = b
            .constant("idx", TensorType::new(vec![2], poot_tensor::DType::I32))
            .unwrap();
        let out = b.gather(data, 1, idx).unwrap(); // axis 1: inner = 5, axis_len = 4
        let g = b.finish(out).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::Gather { .. }))
            .unwrap();
        plan_eqn_with_choice(
            &g,
            eqn,
            backend,
            &poot_test_util::device_caps::default_caps_for(backend),
        )
        .unwrap()
    };
    for (dt, backend, carrier) in [
        (DType::F32, Backend::SpirvVulkan, poot_kernel_ir::Ty::F32),
        (DType::F32, Backend::Nvptx, poot_kernel_ir::Ty::F32),
        (DType::I32, Backend::SpirvVulkan, poot_kernel_ir::Ty::I32),
    ] {
        let (plan_, choice_) = plan_for(dt, backend);
        let spec = movement_of(&plan_, &choice_);
        assert_eq!(
            spec.storage,
            poot_kernelgen::Storage::Elements { carrier },
            "{dt:?} {backend:?}"
        );
        assert!(
            matches!(
                &spec.op,
                MovementOp::Gather { in_shape, axis: 1, index_numel: 2, .. } if in_shape == &[3, 4, 5]
            ),
            "{dt:?} {backend:?}: {spec:?}"
        );
        assert_eq!(spec.out_shape, [3, 2, 5]);
    }
}
#[test]
fn gather_nonaxis0_gpu_matches_cpu_over_random_shapes() {
    // Property fuzzer for the imported non-axis-0 gather (`gather_axis`, metadata launch `[inner, axis_len]`), which the axis-0-only
    // gather-composition fuzzer does not cover. Random data rank (3-4), gather axis (>=1) and vector-index size and positions
    // stress the in-kernel `inner`/`axis_len`/`idx_numel` derivation and the meta buffer. Gather is a pure copy, so `gpu.run`
    // must equal CPU eval bit-for-bit.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
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
        let rank = 3 + (rng() % 2) as usize; // 3 or 4
        let dims: Vec<usize> = (0..rank).map(|_| 2 + (rng() % 4) as usize).collect();
        let axis = 1 + (rng() % (rank as u64 - 1)) as usize; // non-axis-0
        let idx_size = 1 + (rng() % 3) as usize;
        let axis_len = dims[axis];
        let idxd: Vec<i32> = (0..idx_size)
            .map(|_| (rng() % axis_len as u64) as i32)
            .collect();
        let data_numel: usize = dims.iter().product();
        let datad: Vec<f32> = {
            let mut v = s.wrapping_add(0x9E3779B97F4A7C15) | 1;
            (0..data_numel)
                .map(|_| {
                    v ^= v << 13;
                    v ^= v >> 7;
                    v ^= v << 17;
                    ((v >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };

        let b = Builder::for_tests();
        let data = b.constant("data", TensorType::f32(dims.clone())).unwrap();
        let idx = b
            .constant(
                "idx",
                TensorType::new(vec![idx_size], poot_tensor::DType::I32),
            )
            .unwrap();
        let out = b.gather(data, axis, idx).unwrap();
        let (di, ii) = (data.id(), idx.id());
        let g = b.finish(out).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(di, HostTensor::f32(dims.clone(), datad));
        inputs.insert(ii, HostTensor::i32(vec![idx_size], idxd));

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
        assert_eq!(
            got.as_f32().unwrap(),
            cpu.as_f32().unwrap(),
            "seed {seed} (dims={dims:?}, axis={axis}, idx_size={idx_size}): gather is a copy, must be bit-exact"
        );
    }
}

#[test]
fn scatter_update_gpu_matches_cpu() {
    // The `ScatterUpdate` graph op (the multi-token paged KV write) lowers on the GPU and matches the CPU oracle.
    // out[p] = inv[p]>=0 ? src[inv[p]] : base[p]: POOL=4 rows of rest=2, src=2 rows, inv maps src row 1 -> slot 0 and src row 0
    // -> slot 2; slots 1,3 keep base. Tracer-emittable op, plan route and eager eval agree (one definition,
    // shared by both backends).
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (pool, n, rest) = (4usize, 2usize, 2usize);
    let b = Builder::for_tests();
    let base = b
        .constant("base", TensorType::f32(vec![pool, rest]))
        .unwrap();
    let src = b.constant("src", TensorType::f32(vec![n, rest])).unwrap();
    let inv = b
        .constant("inv", TensorType::new(vec![pool], poot_tensor::DType::I32))
        .unwrap();
    let out = b.scatter_update(base, src, inv).unwrap();
    let (bi, si, ii) = (base.id(), src.id(), inv.id());
    let g = b.finish(out).unwrap();

    let based: Vec<f32> = (0..pool * rest).map(|i| 100.0 + i as f32).collect();
    let srcd: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let invd: Vec<i32> = vec![1, -1, 0, -1];
    let mut inputs = HashMap::new();
    inputs.insert(bi, HostTensor::f32(vec![pool, rest], based));
    inputs.insert(si, HostTensor::f32(vec![n, rest], srcd));
    inputs.insert(ii, HostTensor::i32(vec![pool], invd));

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
    assert_eq!(got.shape(), vec![pool, rest]);
    assert_eq!(got.shape(), cpu.shape());
    for (i, (a, b)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-6,
            "scatter_update elem {i}: gpu {a} vs cpu {b}"
        );
    }
    // ground truth: slot0=src1=[3,4], slot1=base=[102,103], slot2=src0=[1,2], slot3=base=[106,107].
    assert_eq!(
        got.as_f32().unwrap(),
        &[3.0, 4.0, 102.0, 103.0, 1.0, 2.0, 106.0, 107.0]
    );
}

#[test]
fn gather_axis0_plan_selection() {
    // The axis-0 embedding gather is the one generated movement kernel on every backend and dtype, with the
    // table's row width implied by its shape rather than a metadata buffer. Pure planner test (no GPU).
    use poot_graph_ir::op::OpKind;
    use poot_target::Backend;
    let plan_for = |dt: TensorType, backend| {
        let b = Builder::for_tests();
        // data [table_rows=5, rest=8]; index [rows=3]; out [3, 8].
        let data = b.constant("data", dt).unwrap();
        let idx = b
            .constant("idx", TensorType::new(vec![3], poot_tensor::DType::I32))
            .unwrap();
        let out = b.gather(data, 0, idx).unwrap();
        let g = b.finish(out).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::Gather { axis: 0 }))
            .unwrap();
        plan_eqn_with_choice(
            &g,
            eqn,
            backend,
            &poot_test_util::device_caps::default_caps_for(backend),
        )
        .unwrap()
    };
    for (table, backend) in [
        (TensorType::f32(vec![5, 8]), Backend::SpirvVulkan),
        (TensorType::f32(vec![5, 8]), Backend::Nvptx),
        (
            TensorType::new(vec![5, 8], poot_tensor::DType::I32),
            Backend::SpirvVulkan,
        ),
    ] {
        let (plan_, choice_) = plan_for(table, backend);
        let spec = movement_of(&plan_, &choice_);
        assert!(
            matches!(
                &spec.op,
                MovementOp::Gather { in_shape, axis: 0, index_numel: 3, .. } if in_shape == &[5, 8]
            ),
            "{backend:?}: {spec:?}"
        );
        assert_eq!(spec.out_shape, [3, 8]);
    }
}
#[test]
fn scatter_update_plan_selection() {
    // The paged-KV scatter-update is the one generated movement kernel on every backend and dtype: a whole
    // dispatch below the watchdog budget, no metadata buffer. Pure planner test.
    use poot_graph_ir::op::OpKind;
    use poot_target::Backend;
    use poot_tensor::DType;
    // base [4,3] pool, src [2,3], inv [4].
    let plan_for = |dt: DType, backend| {
        let b = Builder::for_tests();
        let base = b.constant("base", TensorType::new(vec![4, 3], dt)).unwrap();
        let src = b.constant("src", TensorType::new(vec![2, 3], dt)).unwrap();
        let inv = b
            .constant("inv", TensorType::new(vec![4], poot_tensor::DType::I32))
            .unwrap();
        let out = b.scatter_update(base, src, inv).unwrap();
        let g = b.finish(out).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::ScatterUpdate))
            .unwrap();
        plan_eqn_with_choice(
            &g,
            eqn,
            backend,
            &poot_test_util::device_caps::default_caps_for(backend),
        )
        .unwrap()
    };
    for (dt, backend) in [
        (DType::F32, Backend::SpirvVulkan),
        (DType::F32, Backend::Nvptx),
        (DType::I32, Backend::SpirvVulkan),
    ] {
        let (plan_, choice_) = plan_for(dt, backend);
        let spec = movement_of(&plan_, &choice_);
        assert!(
            matches!(
                &spec.op,
                MovementOp::ScatterRows { src_shape, window: None, .. } if src_shape == &[2, 3]
            ),
            "{dt:?} {backend:?}: {spec:?}"
        );
        assert_eq!(spec.out_shape, [4, 3]);
    }
}
#[test]
fn scatter_axis0_plan_selection() {
    // The axis-0 permutation scatter is the one generated movement kernel on every backend and dtype, one
    // thread per source element. Pure planner test (no GPU).
    use poot_graph_ir::op::OpKind;
    use poot_target::Backend;
    let plan_for = |dt: TensorType, backend| {
        let b = Builder::for_tests();
        // src [rows=4, rest=3]; index [4]; out [4, 3] (a permutation).
        let src = b.constant("src", dt).unwrap();
        let idx = b
            .constant("idx", TensorType::new(vec![4], poot_tensor::DType::I32))
            .unwrap();
        let out = b.scatter(src, idx).unwrap();
        let g = b.finish(out).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::Scatter { axis: 0 }))
            .unwrap();
        plan_eqn_with_choice(
            &g,
            eqn,
            backend,
            &poot_test_util::device_caps::default_caps_for(backend),
        )
        .unwrap()
    };
    for (table, backend) in [
        (TensorType::f32(vec![4, 3]), Backend::SpirvVulkan),
        (TensorType::f32(vec![4, 3]), Backend::Nvptx),
        (
            TensorType::new(vec![4, 3], poot_tensor::DType::I32),
            Backend::SpirvVulkan,
        ),
    ] {
        let (plan_, choice_) = plan_for(table, backend);
        let spec = movement_of(&plan_, &choice_);
        assert!(
            matches!(
                &spec.op,
                MovementOp::PermuteRows { src_shape, .. } if src_shape == &[4, 3]
            ),
            "{backend:?}: {spec:?}"
        );
        assert_eq!(spec.threads(), 12);
    }
}
#[test]
fn index_remap_layout_plan_selection() {
    // Transpose, slice and broadcast are the one generated movement kernel on every backend and dtype:
    // the request names the layout change and the dispatch binds no metadata buffer. Pure planner test
    // (no GPU).
    use poot_graph_ir::op::OpKind;
    use poot_target::Backend;
    let plan_op = |build: &dyn Fn(&Builder) -> poot_graph_ir::builder::Traced,
                   want: fn(&OpKind) -> bool,
                   backend| {
        let b = Builder::for_tests();
        let out = build(&b);
        let g = b.finish(out).unwrap();
        let eqn = g.eqns().iter().find(|e| want(&e.op)).unwrap();
        plan_eqn_with_choice(
            &g,
            eqn,
            backend,
            &poot_test_util::device_caps::default_caps_for(backend),
        )
        .unwrap()
    };
    let f32t = |b: &Builder, sh: Vec<usize>| b.constant("x", TensorType::f32(sh)).unwrap();
    let i32t = |b: &Builder, sh: Vec<usize>| {
        b.constant("x", TensorType::new(sh, poot_tensor::DType::I32))
            .unwrap()
    };
    let is_t = |o: &OpKind| matches!(o, OpKind::Transpose { .. });
    let is_s = |o: &OpKind| matches!(o, OpKind::Slice { .. });
    let is_b = |o: &OpKind| matches!(o, OpKind::Broadcast { .. });

    for backend in [Backend::SpirvVulkan, Backend::Nvptx] {
        // TRANSPOSE [2,3,4] perm [2,0,1] -> [4,2,3].
        let (plan_, choice_) = plan_op(
            &|b| b.transpose(f32t(b, vec![2, 3, 4]), vec![2, 0, 1]).unwrap(),
            is_t,
            backend,
        );
        let spec = movement_of(&plan_, &choice_);
        assert!(
            matches!(
                &spec.op,
                MovementOp::Transpose { in_shape, perm } if in_shape == &[2, 3, 4] && perm == &[2, 0, 1]
            ),
            "{backend:?}: {spec:?}"
        );
        assert_eq!(spec.out_shape, [4, 2, 3]);

        // SLICE [4,5] axis 1 start 1 end 4 -> [4,3].
        let (plan_, choice_) = plan_op(
            &|b| b.slice(f32t(b, vec![4, 5]), 1, 1, 4).unwrap(),
            is_s,
            backend,
        );
        let spec = movement_of(&plan_, &choice_);
        assert!(
            matches!(
                &spec.op,
                MovementOp::Slice { in_shape, axis: 1, start: 1 } if in_shape == &[4, 5]
            ),
            "{backend:?}: {spec:?}"
        );

        // BROADCAST [3,1] -> [3,4].
        let (plan_, choice_) = plan_op(
            &|b| b.broadcast(f32t(b, vec![3, 1]), vec![3, 4]).unwrap(),
            is_b,
            backend,
        );
        let spec = movement_of(&plan_, &choice_);
        assert!(
            matches!(&spec.op, MovementOp::Broadcast { in_shape } if in_shape == &[3, 1]),
            "{backend:?}: {spec:?}"
        );
    }
    // An I32 value moves as I32 words: a transpose of a constant I32 value is an I32 carrier whatever
    // reads it (there is no f32 lane for an I32 value).
    let (plan_, choice_) = plan_op(
        &|b| b.transpose(i32t(b, vec![2, 3, 4]), vec![2, 0, 1]).unwrap(),
        is_t,
        Backend::SpirvVulkan,
    );
    assert_eq!(
        movement_of(&plan_, &choice_).storage,
        poot_kernelgen::Storage::Elements {
            carrier: poot_kernel_ir::Ty::I32
        }
    );
}
#[test]
fn gather_composition_gpu_matches_cpu_over_random_graphs() {
    // The gather coverage gap: the fused and movement fuzzers above exclude gather/scatter, yet gather is everywhere in real
    // graphs (embedding / KV-cache / MoE-router lookups). Random graphs start from an embedding-style axis-0 gather and
    // interleave more gathers with pointwise + reduce regions, then `gpu.run(fuse(cse(g)))` must match the CPU eager eval. This
    // exercises the axis-0 gather kernel as a producer feeding fused regions and as a consumer of them (a gather whose `data` is
    // a fused result). `acc` stays rank-2 `[rows, D]` (D fixed, only the row count changes via gather) so every step stays
    // shape-valid.
    use poot_graph_ir::op::{BinOp, RedOp, UnOp};
    use poot_graph_plan::{cse, fuse};
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
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    };
    let mut fused_total = 0usize;
    let mut chained_gathers = 0usize;
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
        let d = 3 + (rng() % 3) as usize; // fixed last dim, 3..5
        let v = 5 + (rng() % 4) as usize; // table rows, 5..8
        let b = Builder::for_tests();
        let mut inputs = HashMap::new();
        let table = b.constant("table", TensorType::f32(vec![v, d])).unwrap();
        inputs.insert(
            table.id(),
            HostTensor::f32(vec![v, d], fill(seed * 7 + 1, v * d)),
        );
        let bcast = b.constant("bcast", TensorType::f32(vec![1, d])).unwrap(); // row-broadcast leaf
        inputs.insert(
            bcast.id(),
            HostTensor::f32(vec![1, d], fill(seed * 7 + 2, d)),
        );

        // the initial embedding-style gather: idx [k0] of valid rows into the [v, d] table.
        let k0 = 2 + (rng() % 4) as usize;
        let i0: Vec<i32> = (0..k0).map(|_| (rng() as usize % v) as i32).collect();
        let idx0 = b
            .constant("idx0", TensorType::new(vec![k0], poot_tensor::DType::I32))
            .unwrap();
        inputs.insert(idx0.id(), HostTensor::i32(vec![k0], i0));
        let mut acc = b.gather(table, 0, idx0).unwrap(); // [k0, d]
        let mut rows = k0;

        let mut gctr = 1usize;
        let mut had_chain = false;
        let steps = 4 + (rng() % 3) as usize; // 4..6 ops
        for _ in 0..steps {
            match rng() % 5 {
                0 => {
                    let op = [UnOp::Neg, UnOp::Tanh, UnOp::Erf][(rng() as usize) % 3];
                    acc = b.unary(op, acc).unwrap();
                }
                1 => {
                    let op = [BinOp::Add, BinOp::Sub, BinOp::Mul][(rng() as usize) % 3];
                    acc = b.binary(op, acc, acc).unwrap(); // self-binary keeps the shape
                }
                2 => {
                    let op = [BinOp::Add, BinOp::Sub, BinOp::Mul][(rng() as usize) % 3];
                    acc = b.binary(op, acc, bcast).unwrap(); // [1, d] broadcasts over the rows
                }
                3 => {
                    let rop = if rng().is_multiple_of(2) {
                        RedOp::Sum
                    } else {
                        RedOp::Max
                    };
                    let r = b.reduce(rop, acc, 1, true).unwrap(); // [rows, 1]
                    acc = b.broadcast(r, vec![rows, d]).unwrap();
                }
                _ => {
                    // a second gather along axis 0 - its `data` is the (possibly fused) current acc.
                    let kn = 2 + (rng() % 4) as usize;
                    let iv: Vec<i32> = (0..kn).map(|_| (rng() as usize % rows) as i32).collect();
                    let idxn = b
                        .constant(
                            &format!("idx{gctr}"),
                            TensorType::new(vec![kn], poot_tensor::DType::I32),
                        )
                        .unwrap();
                    inputs.insert(idxn.id(), HostTensor::i32(vec![kn], iv));
                    gctr += 1;
                    acc = b.gather(acc, 0, idxn).unwrap();
                    rows = kn;
                    had_chain = true;
                }
            }
        }
        if had_chain {
            chained_gathers += 1;
        }
        let g = b.finish(acc).unwrap();
        let fg = fuse(&cse(&g).unwrap()).unwrap();
        if fg.eqns().len() < g.eqns().len() {
            fused_total += 1;
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
    eprintln!(
        "gather-composition fuzz: 24 random graphs, {fused_total} fused, {chained_gathers} chained a 2nd gather"
    );
    assert!(
        fused_total >= 12,
        "many gather graphs should still fuse their pointwise regions (got {fused_total}/24)"
    );
    assert!(
        chained_gathers >= 6,
        "several graphs should chain a second gather (got {chained_gathers}/24)"
    );
}

#[test]
fn scatter_composition_gpu_matches_cpu_over_random_graphs() {
    // Write-side companion to the gather fuzzer: scatter (the axis-0 row reorder, `out[idx[j]] = src[j]`) is the other op the
    // movement/fused fuzzers exclude, and its codegen writes to computed row addresses, which is more error-prone than a read.
    // Random graphs interleave scatter (with a fresh random permutation index, so writes never collide) with pointwise +
    // reduce, then `gpu.run(fuse(cse(g)))` must match the CPU eager eval. scatter preserves the `[N, D]` shape, and consumes
    // the (possibly fused) current acc as `src`, exercising scatter as a fused-region consumer.
    use poot_graph_ir::op::{BinOp, RedOp, UnOp};
    use poot_graph_plan::{cse, fuse};
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
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    };
    let mut fused_total = 0usize;
    let mut scattered = 0usize;
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
        let nn = 3 + (rng() % 4) as usize; // rows 3..6
        let d = 3 + (rng() % 3) as usize; // cols 3..5
        let b = Builder::for_tests();
        let mut inputs = HashMap::new();
        let mut acc = b.constant("x", TensorType::f32(vec![nn, d])).unwrap();
        inputs.insert(
            acc.id(),
            HostTensor::f32(vec![nn, d], fill(seed * 7 + 1, nn * d)),
        );
        let bcast = b.constant("bcast", TensorType::f32(vec![1, d])).unwrap();
        inputs.insert(
            bcast.id(),
            HostTensor::f32(vec![1, d], fill(seed * 7 + 2, d)),
        );

        let mut gctr = 0usize;
        let mut did_scatter = false;
        let steps = 4 + (rng() % 3) as usize; // 4..6
        for _ in 0..steps {
            match rng() % 4 {
                0 => {
                    let op = [UnOp::Neg, UnOp::Tanh, UnOp::Erf][(rng() as usize) % 3];
                    acc = b.unary(op, acc).unwrap();
                }
                1 => {
                    let op = [BinOp::Add, BinOp::Sub, BinOp::Mul][(rng() as usize) % 3];
                    acc = b.binary(op, acc, acc).unwrap();
                }
                2 => {
                    // a reduce over the cols then broadcast back keeps [N, D] and adds a fused-reduce region.
                    let rop = if rng().is_multiple_of(2) {
                        RedOp::Sum
                    } else {
                        RedOp::Max
                    };
                    let r = b.reduce(rop, acc, 1, true).unwrap();
                    acc = b.broadcast(r, vec![nn, d]).unwrap();
                }
                _ => {
                    // scatter acc's rows by a random permutation (Fisher-Yates) - no write collisions.
                    let mut perm: Vec<i32> = (0..nn as i32).collect();
                    for i in (1..nn).rev() {
                        perm.swap(i, (rng() as usize) % (i + 1));
                    }
                    let idx = b
                        .constant(
                            &format!("idx{gctr}"),
                            TensorType::new(vec![nn], poot_tensor::DType::I32),
                        )
                        .unwrap();
                    inputs.insert(idx.id(), HostTensor::i32(vec![nn], perm));
                    gctr += 1;
                    acc = b.scatter(acc, idx).unwrap();
                    did_scatter = true;
                }
            }
        }
        if did_scatter {
            scattered += 1;
        }
        let g = b.finish(acc).unwrap();
        let fg = fuse(&cse(&g).unwrap()).unwrap();
        if fg.eqns().len() < g.eqns().len() {
            fused_total += 1;
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
    eprintln!(
        "scatter-composition fuzz: 24 random graphs, {fused_total} fused, {scattered} scattered"
    );
    assert!(
        scattered >= 12,
        "most graphs should contain a scatter (got {scattered}/24)"
    );
}

/// Spike 562 F-9 (card 643): an I32 `Slot::Pos` that is a Gather index and also a Cast source, a
/// DynamicUpdateSlice start, or both ([`poot_eval::test_support::i32_slot_gather`]), driven through `compile` (the
/// production graph-to-program entry) and the resident executor, bit for bit against the CPU oracle.
///
/// POOT-725: no I32 value has a second lane, so the Gather index, the Cast source and the DUS start read the
/// slot as `Slice<i32>`; v3 writes at a runtime I32 index from a named `Slot::Activation` input.
///
/// Mutation (never left in the tree): encode an I32 slot's words as f32 values in `lanes::encode_host` (the
/// old mirror lane). Observed: v0 `wgpu elem 0: 1077936100 vs cpu 3.518228`, v1 `wgpu elem 24: 0.9320391 vs
/// cpu 0.5182281`, v2 `wgpu elem 24: 0.9320391 vs cpu 3.518228` and v3 `wgpu elem 32: 0.9995736 vs cpu 0`
/// (the write misses its offset).
#[test]
fn i32_slot_gather_variants_compiled_match_cpu_bit_exact() {
    use poot_eval::test_support::i32_slot_gather::run_all;
    use poot_executor::StepInputs;
    use poot_graph_ir::StateRole;
    use poot_graph_ir::test_support::const_weights::ConstWeights;
    use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};

    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };

    // Card 546b: the contract admits `Submission::Replay` only (`staged_replay`), never the `Submission::Eager`
    // of the deleted `GpuExecutor::run_resident_kv` + raw `&[DeviceBuffer]` state. The bug this
    // row guards (an I32 `Slot::Pos` read as both a `Gather` index and a `Cast`/`DynamicUpdateSlice`
    // source, `planner/cast.rs`'s mirror-lane `Plan::Alias`) is a `compile`-time lane decision, not an
    // execution-model one, so `Submission::Replay` exercises the identical planner path. The Dus/
    // CastDus variants carry a seeded (non-zero) `Storage::State` "cache": since `add_entry` only ever
    // zero-allocates a fresh state buffer (no seed-by-name upload exists on the contract), a tiny
    // priming entry writes the seed into that shared, name-keyed state buffer first (Z5: two entries on
    // one executable share state by name) before the real variant's entry reads it.
    run_all("wgpu", |variant| {
        // `compile()` (not `staged_replay`'s own `compile_staged`) so `graph` stays `Graph`,
        // matching `variant.inputs`'s signature; `staged_replay` below compiles it a second time into a
        // `StagedProgram` for the engine, mirroring `legalize.rs`'s already-landed split test (compile
        // once to inspect/bind, `run_via_contract`/`staged_replay` compiles again to execute).
        let options = poot_graph_plan::CompileOptions::new(
            poot_graph_plan::production_plan::Submission::Replay,
            poot_graph_plan::CompileLimits::STANDARD,
        );
        let program = poot_graph_plan::compile(&variant.graph, &target, &options)
            .map_err(|e| format!("compile: {e}"))?;
        let graph = program.graph().clone();
        let bound = variant.inputs(&graph);

        let mut store = WeightStore::builder();
        let mut slots = StepInputs::new();
        let mut slot_bytes: Vec<(poot_graph_ir::SlotKey, Vec<usize>, DType, Vec<u8>)> = Vec::new();
        let mut state_seed: Option<(String, Vec<usize>)> = None;
        for &id in graph.inputs() {
            let meta = graph.meta(id);
            let Some(Value::Host(t)) = bound.get(&id) else {
                return Err(format!("{id:?}: expected a dense input"));
            };
            match meta.storage {
                Storage::Const => {
                    let name = meta.name.clone().ok_or("const without a name")?;
                    let bytes: Vec<u8> = t
                        .as_f32()
                        .unwrap()
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    let dense = DenseWeight::try_new(
                        DType::F32,
                        t.shape().to_vec(),
                        StoredBytes::from(bytes),
                    )
                    .map_err(|e| e.to_string())?;
                    store
                        .insert(name, WeightEntry::Dense(dense))
                        .map_err(|e| e.to_string())?;
                }
                Storage::Slot(_) => {
                    let key = meta.slot_key().ok_or("slot without a SlotKey")?.clone();
                    let bytes: Vec<u8> = if meta.aval.dtype == DType::I32 {
                        t.as_i32()
                            .ok_or("I32 slot without I32 words")?
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect()
                    } else {
                        t.as_f32()
                            .unwrap()
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect()
                    };
                    slot_bytes.push((key, t.shape().to_vec(), meta.aval.dtype, bytes));
                }
                Storage::State => {
                    let name = meta.name.clone().ok_or("state without a name")?;
                    let seed_bytes: Vec<u8> = t
                        .as_f32()
                        .unwrap()
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    let dense = DenseWeight::try_new(
                        DType::F32,
                        t.shape().to_vec(),
                        StoredBytes::from(seed_bytes),
                    )
                    .map_err(|e| e.to_string())?;
                    store
                        .insert(format!("{name}.seed"), WeightEntry::Dense(dense))
                        .map_err(|e| e.to_string())?;
                    state_seed = Some((name, t.shape().to_vec()));
                }
                Storage::Computed(_) => {}
                other => return Err(format!("{id:?}: unexpected storage {other:?}")),
            }
        }
        for (key, shape, dtype, bytes) in &slot_bytes {
            let elems = shape.iter().product();
            let view = poot_tensor::HostView::new(*dtype, elems, bytes)
                .map_err(|e| format!("slot view: {e}"))?;
            slots.push(key.clone(), shape, view);
        }

        let store = store.build();
        let mut weights = ConstWeights::new(&store);
        let staged = staged_replay(&weights.graph(&graph), target);
        let prime = state_seed.as_ref().map(|(name, shape)| {
            let b = Builder::for_tests();
            let seed = b
                .constant(&format!("{name}.seed"), TensorType::f32(shape.clone()))
                .unwrap();
            let cache = b
                .state_input(name, TensorType::f32(shape.clone()), StateRole::Recurrent)
                .unwrap();
            weights.graph(
                &b.finish(poot_graph_ir::Outputs::new(seed).with_state(&[(cache, seed)]))
                    .unwrap(),
            )
        });
        let map = Arc::new(weights.map());
        let exe = gpu
            .load_weights(Arc::new(store), map)
            .map_err(|e| e.to_string())?;

        if let Some(prime) = &prime {
            let prime_staged = staged_replay(prime, target);
            let prime_entry = gpu
                .add_entry(exe, &prime_staged)
                .map_err(|e| format!("prime add_entry: {e}"))?;
            gpu.step(
                exe,
                prime_entry,
                &StepInputs::new(),
                &mut poot_executor::NoSync,
            )
            .map_err(|e| format!("prime step: {e}"))?;
            gpu.remove_entry(exe, prime_entry)
                .map_err(|e| format!("prime remove_entry: {e}"))?;
        }

        let entry = gpu
            .add_entry(exe, &staged)
            .map_err(|e| format!("add_entry: {e}"))?;
        let bytes = gpu
            .step(exe, entry, &slots, &mut poot_executor::NoSync)
            .map_err(|e| format!("step: {e}"))?
            .read()
            .map_err(|e| format!("read: {e}"))?;
        gpu.remove_entry(exe, entry)
            .map_err(|e| format!("remove_entry: {e}"))?;
        gpu.unload(exe).map_err(|e| format!("unload: {e}"))?;

        let out = graph.aval(graph.result());
        Ok(HostTensor::f32(
            out.shape.clone(),
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ))
    });
}
