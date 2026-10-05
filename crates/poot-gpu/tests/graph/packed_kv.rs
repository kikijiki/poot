use super::*;
use poot_graph_plan::{ImportedKernel, KernelChoice};

/// Whether a plan's choice is the shipped kernel `kernel`.
fn is_imported(choice: &KernelChoice, kernel: ImportedKernel) -> bool {
    matches!(choice, KernelChoice::Imported { kernel: k, .. } if *k == kernel)
}

#[test]
fn nvptx_decode_gemv_plan_selection() {
    // Card 044 PTX swap: on the NVPTX path the B=1 decode GEMV is the imported kernel (a plain Plan::Compute), while batched B>1
    // stays on kernelgen (NVPTX has no stable-buffer dims path for ComputeMeta yet). Pure planner test (no GPU); also checks
    // wgpu is unchanged.
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::Plan;
    use poot_target::Backend;
    let (k, n) = (4usize, 8usize);
    let plan_for = |bsz: usize, backend| {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![bsz, 1, k]));
        let w = b.constant("w", TensorType::f32(vec![k, n]));
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
    // B=1 on NVPTX -> the coalesced imported GEMV (Plan::Compute, key "matmul:gemv:coalesced").
    match plan_for(1, Backend::Nvptx) {
        (Plan::Compute { .. }, choice) => {
            assert!(
                is_imported(&choice, ImportedKernel::GemvCoalesced),
                "B=1 NVPTX -> coalesced gemv: {choice:?}"
            )
        }
        _ => panic!("B=1 NVPTX decode GEMV should be a Compute plan"),
    }
    // B>1 on NVPTX -> the imported batched gemv ComputeMeta (the executor binds the dims=[B] buffer).
    match plan_for(2, Backend::Nvptx) {
        (Plan::ComputeMeta { meta, .. }, choice) => {
            assert!(
                is_imported(&choice, ImportedKernel::GemvBatchedCoalesced),
                "B>1 NVPTX -> batched gemv ComputeMeta: {choice:?}"
            );
            assert_eq!(meta, vec![2], "dims = [B]");
        }
        _ => panic!("B>1 NVPTX decode GEMV should be a ComputeMeta plan now"),
    }
    // wgpu unchanged: B=1 coalesced Compute, B>1 ComputeMeta (the batched coalesced gemv).
    assert!(
        matches!(plan_for(1, Backend::SpirvVulkan), (Plan::Compute { .. }, choice) if is_imported(&choice, ImportedKernel::GemvCoalesced))
    );
    assert!(matches!(
        plan_for(2, Backend::SpirvVulkan),
        (Plan::ComputeMeta { .. }, _)
    ));
}

#[test]
fn decode_gemv_random_shapes_gpu_matches_cpu() {
    // The decode GEMV (`gemv_lds`, routed by `is_decode_gemv` for M=1 shared-weight matmuls) is the most-run GPU kernel: the
    // q/k/v/o/gate/up/down projections at M=1 fire on every decode token. Its LDS strided reduction over K and per-output-column
    // 2-D grid (`col = GroupY*x_groups + GroupX`, so a large-N lm_head spills onto Y) had only a couple of fixed-shape tests.
    // Fuzz over random K/N, with a few seeds at N > 65535 to stress the Y-spill, asserting the plan key is `matmul:gemv...` and
    // `gpu.run == CPU`. Half the seeds use the fused-bias variant.
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::Plan;
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
    let mut gemv_count = 0usize;
    let mut spilled = 0usize;
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
        let k = 1 + (rng() % 200) as usize; // 1..200
        // every 8th seed uses a large N (> 65535) that spills the gemv grid onto the Y dim (the lm_head case).
        let n = if seed.is_multiple_of(8) {
            65_536 + (rng() % 4000) as usize
        } else {
            1 + (rng() % 400) as usize
        };
        let with_bias = rng().is_multiple_of(2);
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, k]));
        let w = b.constant("w", TensorType::f32(vec![k, n]));
        let mut inputs = HashMap::new();
        inputs.insert(x.id, HostTensor::f32(vec![1, 1, k], fill(seed * 7 + 1, k)));
        inputs.insert(w.id, HostTensor::f32(vec![k, n], fill(seed * 7 + 2, k * n)));
        let out = if with_bias {
            let bias = b.constant("bias", TensorType::f32(vec![n]));
            inputs.insert(bias.id, HostTensor::f32(vec![n], fill(seed * 7 + 3, n)));
            poot_graph_ir::ops::linear(&b, x, w, Some(bias))
        } else {
            b.matmul(x, w)
        };
        let g = b.finish(out);
        // Card 557: `compile` fuses the traced bias add into `MatMulBias`; plan and choice are read off the
        // compiled program.
        let program = compile_full(&g, &spirv_fixture_target());
        let (eqn, plan) = program
            .planned()
            .find(|(e, _)| matches!(e.op, OpKind::MatMul | OpKind::MatMulBias))
            .unwrap();
        assert_eq!(
            matches!(eqn.op, OpKind::MatMulBias),
            with_bias,
            "seed {seed}: compile should fuse exactly the traced bias add"
        );
        let choice = match plan {
            Plan::Compute { .. } => program.kernel_choice(eqn),
            _ => panic!("seed {seed}: decode GEMV should be a Compute plan"),
        };
        assert!(
            is_imported(choice, ImportedKernel::GemvCoalesced)
                || is_imported(choice, ImportedKernel::GemvCoalescedBias),
            "seed {seed} (K={k},N={n}): expected the decode GEMV path, got {choice:?}"
        );
        gemv_count += 1;
        if n > 65535 {
            spilled += 1;
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
                "seed {seed} (K={k},N={n}) col {i}: gpu {a} vs cpu {c}"
            );
        }
    }
    eprintln!(
        "decode-gemv shape fuzz: {gemv_count}/32 took the gemv path ({spilled} Y-spilled), all GPU==CPU"
    );
    assert_eq!(
        gemv_count, 32,
        "all M=1 shared-weight shapes should take the gemv path"
    );
    assert!(
        spilled >= 3,
        "several seeds should exercise the N>65535 grid spill"
    );
}

#[test]
fn batched_decode_gemv_gpu_matches_cpu() {
    // Card 044: the batched decode GEMV (B>1 rows, one token each, shared weight; continuous batching). `is_decode_gemv` routes
    // it like B=1, but the imported batched kernel needs `dims = [B]`, so the plan is `Plan::ComputeMeta` and the executor binds
    // an extra u32 buffer. Fuzz over random B/K/N, with and without fused bias, asserting the plan key is
    // `matmul:gemv_batched...` and `gpu.run == CPU`; covers the ComputeMeta planner emit and the executor's metadata-buffer
    // binding.
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::Plan;
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
    let mut batched = 0usize;
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
        let bsz = 2 + (rng() % 5) as usize; // B in 2..6
        let k = 1 + (rng() % 200) as usize;
        let n = 1 + (rng() % 400) as usize;
        let with_bias = rng().is_multiple_of(2);
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![bsz, 1, k]));
        let w = b.constant("w", TensorType::f32(vec![k, n]));
        let mut inputs = HashMap::new();
        inputs.insert(
            x.id,
            HostTensor::f32(vec![bsz, 1, k], fill(seed * 7 + 1, bsz * k)),
        );
        inputs.insert(w.id, HostTensor::f32(vec![k, n], fill(seed * 7 + 2, k * n)));
        let out = if with_bias {
            let bias = b.constant("bias", TensorType::f32(vec![n]));
            inputs.insert(bias.id, HostTensor::f32(vec![n], fill(seed * 7 + 3, n)));
            poot_graph_ir::ops::linear(&b, x, w, Some(bias))
        } else {
            b.matmul(x, w)
        };
        let g = b.finish(out);
        // Card 557: `compile` fuses the traced bias add into `MatMulBias`; plan and choice are read off the
        // compiled program.
        let program = compile_full(&g, &spirv_fixture_target());
        let (eqn, plan) = program
            .planned()
            .find(|(e, _)| matches!(e.op, OpKind::MatMul | OpKind::MatMulBias))
            .unwrap();
        assert_eq!(
            matches!(eqn.op, OpKind::MatMulBias),
            with_bias,
            "seed {seed}: compile should fuse exactly the traced bias add"
        );
        let (choice, meta) = match plan {
            Plan::ComputeMeta { meta, .. } => (program.kernel_choice(eqn), meta),
            _ => panic!("seed {seed}: batched GEMV should be ComputeMeta"),
        };
        assert!(
            is_imported(choice, ImportedKernel::GemvBatchedCoalesced)
                || is_imported(choice, ImportedKernel::GemvBatchedCoalescedBias),
            "seed {seed} (B={bsz},K={k},N={n}): expected the batched GEMV path, got {choice:?}"
        );
        assert_eq!(
            meta,
            &vec![bsz as u32],
            "seed {seed}: dims metadata must be [B]"
        );
        batched += 1;

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
                "seed {seed} (B={bsz},K={k},N={n}) elem {i}: gpu {a} vs cpu {c}"
            );
        }
    }
    eprintln!(
        "batched-decode-gemv fuzz: {batched}/24 took the ComputeMeta batched path, all GPU==CPU"
    );
    assert_eq!(
        batched, 24,
        "all B>1 M=1 shared-weight shapes should be ComputeMeta"
    );
}

/// Decode bf16 GEMV on device: F32 activation x BF16 weight plans the packed-u32 body, binds via
/// `bind_resident`'s packed lane, and matches a CPU oracle run on the exact-widened f32 weights.
#[test]
fn decode_bf16_gemv_gpu_matches_cpu() {
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::Plan;
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
    // Card 370 RNE round + exact widen (same rule as the kernel's `from_bits(bits << 16)`).
    let f32_to_bf16 = |x: f32| -> u16 {
        let bits = x.to_bits();
        if x.is_nan() {
            return ((bits >> 16) as u16) | 0x0040;
        }
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    };
    let widen = |b: u16| f32::from_bits((b as u32) << 16);

    let mut ran = 0usize;
    for (seed, with_bias) in [(1u64, false), (2, true), (3, false), (4, true)] {
        let (k, n) = (7 + (seed as usize % 5), 40 + (seed as usize % 17));
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, k]));
        let w = b.constant("w", TensorType::new(vec![k, n], DType::BF16));
        let out = if with_bias {
            let bias = b.constant("bias", TensorType::f32(vec![n]));
            poot_graph_ir::ops::linear(&b, x, w, Some(bias))
        } else {
            b.matmul(x, w)
        };
        let g = b.finish(out);
        // Card 557: `compile` fuses the traced bias add into `MatMulBias`; plan and choice are read off the
        // compiled program.
        let program = compile_full(&g, &spirv_fixture_target());
        let (eqn, plan) = program
            .planned()
            .find(|(e, _)| matches!(e.op, OpKind::MatMul | OpKind::MatMulBias))
            .unwrap();
        assert_eq!(
            matches!(eqn.op, OpKind::MatMulBias),
            with_bias,
            "seed {seed}: compile should fuse exactly the traced bias add"
        );
        let choice = match plan {
            Plan::Compute { .. } => program.kernel_choice(eqn),
            other => panic!("expected Compute, got {other:?}"),
        };
        let expected = if with_bias {
            ImportedKernel::GemvCoalescedBiasBf16
        } else {
            ImportedKernel::GemvCoalescedBf16
        };
        assert!(
            is_imported(choice, expected),
            "seed {seed}: expected the bf16 GEMV, got {choice:?}"
        );

        let x_f = fill(seed * 7 + 1, k);
        let w_f = fill(seed * 7 + 2, k * n);
        let bias_f = fill(seed * 7 + 3, n);
        let w_bf: Vec<u16> = w_f.iter().map(|&x| f32_to_bf16(x)).collect();
        let w_widened: Vec<f32> = w_bf.iter().map(|&b| widen(b)).collect();

        // GPU: BF16-declared weight with native bytes (HostTensor::bf16 words).
        let mut inputs: HashMap<poot_graph_ir::ValueId, HostTensor> = HashMap::new();
        inputs.insert(x.id, HostTensor::f32(vec![1, 1, k], x_f.clone()));
        inputs.insert(w.id, HostTensor::bf16(vec![k, n], w_bf.clone()));
        if with_bias {
            let bias_id = g
                .values
                .iter()
                .enumerate()
                .find(|(_, v)| v.name.as_deref() == Some("bias"))
                .map(|(i, _)| i)
                .unwrap();
            inputs.insert(bias_id, HostTensor::f32(vec![n], bias_f.clone()));
        }

        // CPU oracle: same graph shape with F32 weight holding the exact widenings.
        let b2 = Builder::new();
        let x2 = b2.constant("x", TensorType::f32(vec![1, 1, k]));
        let w2 = b2.constant("w", TensorType::f32(vec![k, n]));
        let out2 = if with_bias {
            let bias = b2.constant("bias", TensorType::f32(vec![n]));
            poot_graph_ir::ops::linear(&b2, x2, w2, Some(bias))
        } else {
            b2.matmul(x2, w2)
        };
        let g2 = b2.finish(out2);
        let mut inputs2 = HashMap::new();
        inputs2.insert(
            x2.id,
            Value::from(HostTensor::f32(vec![1, 1, k], x_f.clone())),
        );
        inputs2.insert(w2.id, Value::from(HostTensor::f32(vec![k, n], w_widened)));
        if with_bias {
            let bias_id = g2
                .values
                .iter()
                .enumerate()
                .find(|(_, v)| v.name.as_deref() == Some("bias"))
                .map(|(i, _)| i)
                .unwrap();
            inputs2.insert(
                bias_id,
                Value::from(HostTensor::f32(vec![n], bias_f.clone())),
            );
        }

        let cpu = eval(&g2, &inputs2, EvalOptions::new(EvalBudget::UNBOUNDED))
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
                (a - c).abs() <= 1e-5 + 1e-5 * c.abs(),
                "seed {seed} (K={k},N={n},bias={with_bias}) elem {i}: gpu {a} vs cpu {c}"
            );
        }
        ran += 1;
    }
    eprintln!("decode bf16 GEMV device: {ran}/4 shapes GPU==CPU");
    assert_eq!(ran, 4);
}
