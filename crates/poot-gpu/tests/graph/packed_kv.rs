use super::*;
use poot_graph_plan::{ImportedKernel, KernelChoice};

/// Whether a plan's choice is the shipped kernel `kernel`.
fn is_imported(choice: &KernelChoice, kernel: ImportedKernel) -> bool {
    matches!(choice, KernelChoice::Imported { kernel: k, .. } if *k == kernel)
}

#[test]
fn nvptx_decode_gemv_plan_selection() {
    // Card 044 PTX swap: on the NVPTX path the B=1 decode GEMV is the imported kernel (a plain single dispatch), while batched B>1
    // stays on kernelgen (NVPTX has no stable-buffer dims path for metadata launches yet). Pure planner test (no GPU); also checks
    // wgpu is unchanged.
    use poot_graph_ir::op::OpKind;
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
    // B=1 on NVPTX -> the coalesced imported GEMV (single dispatch, key "matmul:gemv:coalesced").
    let (plan_, choice_) = plan_for(1, Backend::Nvptx);
    match (single_meta(&plan_), choice_) {
        (Some([]), choice) => {
            assert!(
                is_imported(&choice, ImportedKernel::GemvCoalesced),
                "B=1 NVPTX -> coalesced gemv: {choice:?}"
            )
        }
        _ => panic!("B=1 NVPTX decode GEMV should be a Compute plan"),
    }
    // B>1 on NVPTX -> the imported batched gemv metadata launch (the executor binds the dims=[B] buffer).
    let (plan_, choice_) = plan_for(2, Backend::Nvptx);
    match (single_meta(&plan_), choice_) {
        (Some(meta @ [_, ..]), choice) => {
            assert!(
                is_imported(&choice, ImportedKernel::GemvBatchedCoalesced),
                "B>1 NVPTX -> batched gemv metadata launch: {choice:?}"
            );
            assert_eq!(meta, vec![2], "dims = [B]");
        }
        _ => panic!("B>1 NVPTX decode GEMV should be a metadata launch now"),
    }
    // wgpu unchanged: B=1 coalesced single dispatch, B>1 metadata launch (the batched coalesced gemv).
    assert!(
        matches!(plan_for(1, Backend::SpirvVulkan), (plan_, choice) if single_meta(&plan_).is_some_and(|meta| meta.is_empty()) && is_imported(&choice, ImportedKernel::GemvCoalesced))
    );
    assert!(matches!(
        plan_for(2, Backend::SpirvVulkan),
        (plan_, _) if single_meta(&plan_).is_some_and(|meta| !meta.is_empty())
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
        // Plan and choice are read off the compiled program. The decode GEMV has no epilogue lowering, so
        // `compile` leaves the traced bias add as its own equation.
        let program = compile_full(&g, &spirv_fixture_target());
        let (eqn, plan) = program
            .planned()
            .find(|(e, _)| matches!(e.op, OpKind::MatMul))
            .unwrap();
        assert_eq!(
            program
                .planned()
                .filter(|(e, _)| matches!(e.op, OpKind::Binary(_)))
                .count(),
            usize::from(with_bias),
            "seed {seed}: the traced bias add stays one equation beside the GEMV"
        );
        let choice = match single_meta(plan) {
            Some([]) => program.kernel_choice(eqn),
            _ => panic!("seed {seed}: decode GEMV should be one dispatch without metadata"),
        };
        assert!(
            is_imported(choice, ImportedKernel::GemvCoalesced),
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
    // it like B=1, but the imported batched kernel needs `dims = [B]`, so the plan is a metadata launch and the executor binds
    // an extra u32 buffer. Fuzz over random B/K/N, with and without fused bias, asserting the plan key is
    // `matmul:gemv_batched...` and `gpu.run == CPU`; covers the metadata launchner emit and the executor's metadata-buffer
    // binding.
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
        // Plan and choice are read off the compiled program. The decode GEMV has no epilogue lowering, so
        // `compile` leaves the traced bias add as its own equation.
        let program = compile_full(&g, &spirv_fixture_target());
        let (eqn, plan) = program
            .planned()
            .find(|(e, _)| matches!(e.op, OpKind::MatMul))
            .unwrap();
        assert_eq!(
            program
                .planned()
                .filter(|(e, _)| matches!(e.op, OpKind::Binary(_)))
                .count(),
            usize::from(with_bias),
            "seed {seed}: the traced bias add stays one equation beside the GEMV"
        );
        let (choice, meta) = match single_meta(plan) {
            Some(meta @ [_, ..]) => (program.kernel_choice(eqn), meta),
            _ => panic!("seed {seed}: batched GEMV should be a metadata launch"),
        };
        assert!(
            is_imported(choice, ImportedKernel::GemvBatchedCoalesced),
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
        "batched-decode-gemv fuzz: {batched}/24 took the metadata-launch batched path, all GPU==CPU"
    );
    assert_eq!(
        batched, 24,
        "all B>1 M=1 shared-weight shapes should be a metadata launch"
    );
}
