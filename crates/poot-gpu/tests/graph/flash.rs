use super::*;
use poot_graph_ir::ops::{attention_prefill, attention_prefill_softcap};
use poot_graph_plan::{ImportedKernel, KernelChoice};
use poot_kernelgen::{AttentionSpec, ContractionSpec, KernelRequest};

/// Whether a choice is the generated region flash decode (`kg::flash_region_decode`), the planner's
/// choice for every decode within `FLASH_LDS_CAP` (Card 557).
fn is_region_decode(choice: &KernelChoice) -> bool {
    matches!(
        choice,
        KernelChoice::Generated(KernelRequest::Attention(AttentionSpec::RegionDecode { .. }))
    )
}

/// Whether a choice is the generated region flash prefill (`kg::flash_region_prefill`).
fn is_region_prefill(choice: &KernelChoice) -> bool {
    matches!(
        choice,
        KernelChoice::Generated(KernelRequest::Attention(
            AttentionSpec::RegionPrefill { .. }
        ))
    )
}

/// Whether a choice is the generated tiled GEMM (`kg::tiled_region`), one dispatch or chunked.
fn is_tiled_region(choice: &KernelChoice) -> bool {
    match choice {
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::TiledRegion {
            ..
        })) => true,
        KernelChoice::Chunked(chunks) => chunks.iter().all(is_tiled_region),
        _ => false,
    }
}

#[test]
fn flash_decode_gpu_matches_cpu() {
    // The traced decode attention chain, which `compile` fuses into `FlashAttentionDecode` and the planner
    // lowers to the generated region flash decode (D within the LDS cap), matches CPU eval of the same
    // graph (`attention_masked`). GQA with Hq=4/Hkv=2 (n_rep=2), cap=3 with the last position masked
    // out, D=4, scale=0.5.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, cap, d) = (4usize, 2usize, 3usize, 4usize);
    let n_rep = hq / hkv;
    let scale = 0.5f32;
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
    let y = attention_masked(&b, q, k, v, n_rep, scale, mask);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(y);
    let choice = compiled_choice(&g, &target, |op| {
        matches!(op, OpKind::FlashAttentionDecode { .. })
    });
    assert!(is_region_decode(&choice), "{choice:?}");
    let qd: Vec<f32> = (0..hq * d).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let kd: Vec<f32> = (0..hkv * cap * d)
        .map(|i| (i as f32) * 0.05 - 0.2)
        .collect();
    let vd: Vec<f32> = (0..hkv * cap * d)
        .map(|i| (i as f32) * 0.07 - 0.1)
        .collect();
    let md: Vec<f32> = vec![0.0, 0.0, -1.0e9]; // the last KV position is masked out
    let mut inputs = HashMap::new();
    inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], qd));
    inputs.insert(ki, HostTensor::f32(vec![1, hkv, cap, d], kd));
    inputs.insert(vi, HostTensor::f32(vec![1, hkv, cap, d], vd));
    inputs.insert(mi, HostTensor::f32(vec![1, 1, 1, cap], md));
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
    assert_eq!(got.shape(), vec![1, hq, 1, d]);
    for (i, (x, c)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!((x - c).abs() < 1e-4, "flash elem {i}: gpu {x} vs cpu {c}");
    }
}

#[test]
fn synthesized_flash_region_decode_matches_cpu() {
    // A traced decode attention compiles to `FlashAttentionDecode`, which the planner lowers via the generated
    // `kg::flash_region_decode` (online softmax, o[D] in LDS; Card 557: the choice is the planner's). Confirm
    // the compiled choice is the synthesized kernel and `gpu.run == eval` (== attention_masked) over a couple
    // of head dims, with the last KV position masked out.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    for &(hq, hkv, cap, d) in &[(4usize, 2usize, 5usize, 4usize), (8, 8, 7, 64)] {
        let n_rep = hq / hkv;
        let scale = 1.0 / (d as f32).sqrt();
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
        let y = attention_masked(&b, q, k, v, n_rep, scale, mask);
        let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
        let g = b.finish(y);

        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::FlashAttentionDecode { .. })
        });
        assert!(
            is_region_decode(&choice),
            "expected the synthesized flash_region_decode choice, got {choice:?}"
        );

        let fill = |seed: u64, n: usize| -> Vec<f32> {
            let mut s = seed | 1;
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                })
                .collect()
        };
        let mut md = vec![0.0f32; cap];
        md[cap - 1] = -1.0e9; // mask out the last KV position
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], fill(1, hq * d)));
        inputs.insert(
            ki,
            HostTensor::f32(vec![1, hkv, cap, d], fill(2, hkv * cap * d)),
        );
        inputs.insert(
            vi,
            HostTensor::f32(vec![1, hkv, cap, d], fill(3, hkv * cap * d)),
        );
        inputs.insert(mi, HostTensor::f32(vec![1, 1, 1, cap], md));
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
        assert_eq!(got.shape(), vec![1, hq, 1, d]);
        for (i, (x, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (x - c).abs() < 1e-4,
                "hq={hq} d={d} elem {i}: gpu {x} vs cpu {c}"
            );
        }
    }
    eprintln!("synthesized flash region decode (graph path) runs GPU==CPU");
}

#[test]
fn synthesized_flash_region_decode_batched_matches_cpu() {
    // The synthesized flash decode handles batched (B>1) decode: `kg::flash_region_decode` decodes `gi = batch*Hq + head`
    // (qbase/kvbase/maskbase offset by batch), so in-flight-batched serving decode uses the generated kernel, not the imported
    // `flash_decode.rs`. Compile a traced B>1 attention, confirm the planner chose the synthesized kernel, and
    // check `gpu.run == eval` (== attention_masked per batch row) over random shapes. B=1 is covered by
    // `synthesized_flash_region_decode_matches_cpu`.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let fill = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 1.4 - 0.7
            })
            .collect()
    };
    for seed in 0u64..12 {
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
        let bsz = 2 + (rng() % 3) as usize;
        let hkv = 1 + (rng() % 2) as usize;
        let n_rep = 1 + (rng() % 2) as usize;
        let hq = hkv * n_rep;
        let d = [2usize, 4, 8][(rng() % 3) as usize];
        let cap = 1 + (rng() % 5) as usize;
        let scale = 0.2 + (rng() % 20) as f32 * 0.04;
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![bsz, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![bsz, hkv, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![bsz, hkv, cap, d]));
        let mask = b.constant("m", TensorType::f32(vec![bsz, 1, 1, cap]));
        let ids = [q.id, k.id, v.id, mask.id];
        let y = attention_masked(&b, q, k, v, n_rep, scale, mask);
        let g = b.finish(y);

        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::FlashAttentionDecode { .. })
        });
        assert!(
            is_region_decode(&choice),
            "expected the synthesized batched flash_region_decode choice, got {choice:?}"
        );

        let md: Vec<f32> = (0..bsz * cap)
            .map(|i| {
                if i % cap == 0 {
                    0.0
                } else {
                    -1.0e9 * ((i % 2) as f32)
                }
            })
            .collect();
        let mut inputs = HashMap::new();
        inputs.insert(
            ids[0],
            HostTensor::f32(vec![bsz, hq, 1, d], fill(seed, bsz * hq * d)),
        );
        inputs.insert(
            ids[1],
            HostTensor::f32(
                vec![bsz, hkv, cap, d],
                fill(seed ^ 0xA5, bsz * hkv * cap * d),
            ),
        );
        inputs.insert(
            ids[2],
            HostTensor::f32(
                vec![bsz, hkv, cap, d],
                fill(seed ^ 0x5A, bsz * hkv * cap * d),
            ),
        );
        inputs.insert(ids[3], HostTensor::f32(vec![bsz, 1, 1, cap], md));
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
        assert_eq!(got.shape(), vec![bsz, hq, 1, d], "seed {seed}");
        for (i, (x, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (x - c).abs() <= 1e-3 + 1e-3 * c.abs(),
                "synth batched flash (seed {seed} B={bsz},hq={hq},d={d},cap={cap}) elem {i}: gpu {x} vs cpu {c}"
            );
        }
    }
    eprintln!("synthesized flash region decode (BATCHED graph path) runs GPU==CPU");
}

#[test]
fn synthesized_flash_region_prefill_matches_cpu() {
    // A traced prefill attention compiles to `FlashAttentionPrefill`, which the planner lowers via the generated
    // `kg::flash_region_prefill` (online softmax, o[D] in LDS, 2-D Hq*L grid) instead of the imported
    // `flash_prefill.rs` (Card 557: the choice is the planner's). Confirm the compiled choice is the synthesized
    // kernel and `gpu.run == eval` (== attention_prefill) with a causal mask.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    for &(hq, hkv, l, d) in &[(4usize, 2usize, 6usize, 4usize), (6, 6, 9, 16)] {
        let n_rep = hq / hkv;
        let scale = 1.0 / (d as f32).sqrt();
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
        let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
        let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
        let o = attention_prefill(&b, q, k, v, n_rep, scale, mask);
        let g = b.finish(o);

        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::FlashAttentionPrefill { .. })
        });
        assert!(
            is_region_prefill(&choice),
            "expected the synthesized flash_region_prefill choice, got {choice:?}"
        );

        let fill = |seed: u64, n: usize| -> Vec<f32> {
            let mut s = seed | 1;
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                })
                .collect()
        };
        // additive causal mask: 0 if j <= row else -1e9.
        let maskd: Vec<f32> = (0..l * l)
            .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
            .collect();
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, l, d], fill(1, hq * l * d)));
        inputs.insert(
            ki,
            HostTensor::f32(vec![1, hkv, l, d], fill(2, hkv * l * d)),
        );
        inputs.insert(
            vi,
            HostTensor::f32(vec![1, hkv, l, d], fill(3, hkv * l * d)),
        );
        inputs.insert(mi, HostTensor::f32(vec![1, 1, l, l], maskd));
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
        assert_eq!(got.shape(), vec![1, hq, l, d]);
        for (i, (x, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (x - c).abs() <= 1e-3 + 1e-3 * c.abs(),
                "hq={hq} l={l} d={d} elem {i}: gpu {x} vs cpu {c}"
            );
        }
    }
    eprintln!("synthesized flash region prefill (graph path) runs GPU==CPU");
}

#[test]
fn flash_region_prefill_d512_matches_cpu() {
    // Card 185: gemma4's global attention layers use head_dim D=512, above the FLASH_LDS_CAP=256 of the imported
    // flash-decode/flash-prefill kernels (fixed `[f32;256]` LDS array). The synthesized flash-prefill kernel
    // (`kg::flash_region_prefill`) sizes its LDS `o[D]` scratch dynamically, with its own
    // FLASH_PREFILL_LDS_CAP=512 (2KB, well under the per-workgroup LDS budget on both backends). Regression test for that cap:
    // the planner must choose the region prefill for a D=512 B=1 prefill (not reject it or leave it on the
    // fixed-256 imported kernel), and the kernel must be correct (gpu.run == eval). Small L/Hq keeps the run
    // fast; varied random inputs so an indexing bug or near-tie shows in max_abs/max_rel.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, l, d) = (2usize, 2usize, 8usize, 512usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
    let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let o = attention_prefill(&b, q, k, v, n_rep, scale, mask);
    let g = b.finish(o);

    let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
        matches!(op, OpKind::FlashAttentionPrefill { .. })
    });
    assert!(
        is_region_prefill(&choice),
        "the B=1 D=512 prefill should choose the synthesized flash_region_prefill (cap 512), got {choice:?}"
    );

    let fill = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    };
    // additive causal mask: 0 if j <= row else -1e9.
    let maskd: Vec<f32> = (0..l * l)
        .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(qi, HostTensor::f32(vec![1, hq, l, d], fill(1, hq * l * d)));
    inputs.insert(
        ki,
        HostTensor::f32(vec![1, hkv, l, d], fill(2, hkv * l * d)),
    );
    inputs.insert(
        vi,
        HostTensor::f32(vec![1, hkv, l, d], fill(3, hkv * l * d)),
    );
    inputs.insert(mi, HostTensor::f32(vec![1, 1, l, l], maskd));
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
    assert_eq!(got.shape(), vec![1, hq, l, d]);
    for (i, (x, c)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        let abs_err = (x - c).abs();
        assert!(
            abs_err <= 1e-3 + 1e-3 * c.abs(),
            "D=512 hq={hq} l={l} d={d} elem {i}: gpu {x} vs cpu {c}"
        );
    }
    eprintln!(
        "synthesized flash region prefill D=512 (card 185) runs GPU==CPU: max_abs={}",
        max_abs_error(got.as_f32().unwrap(), cpu.as_f32().unwrap())
    );
}

#[test]
fn synthesized_production_kernels_random_shapes_match_cpu() {
    // The synthesized kernels are the planner's production choice (Card 557: tiled GEMM for an M>1 F32 matmul,
    // region flash decode/prefill within the LDS caps). This fuzzes the synthesized path over the shape space,
    // since a synthesized-only shape-dependent bug (partial tiles, ragged KV, a coarsening edge) would slip past the
    // fixed-shape `synthesized_*` tests. gpu.run == eval (== the materialized op definition) for dense tiled
    // GEMM, flash decode and flash prefill over random shapes.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let fill = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 0.6 - 0.3
            })
            .collect()
    };
    let rng_for = |seed: u64| {
        let mut s = seed.wrapping_mul(0x100000001B3) | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    };

    // dense tiled GEMM (the generated `tiled_region` choice): random partial-tile M/K/N.
    for seed in 0u64..30 {
        let mut rng = rng_for(seed);
        let (m, k, n) = (
            2 + (rng() % 39) as usize,
            1 + (rng() % 40) as usize,
            1 + (rng() % 40) as usize,
        );
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![m, k]));
        let w = b.constant("w", TensorType::f32(vec![k, n]));
        let out = b.matmul(x, w);
        let g = b.finish(out);
        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::MatMul)
        });
        assert!(
            is_tiled_region(&choice),
            "seed {seed}: an M>1 matmul should choose the tiled GEMM, got {choice:?}"
        );
        let mut inputs = HashMap::new();
        inputs.insert(x.id, HostTensor::f32(vec![m, k], fill(seed * 7 + 1, m * k)));
        inputs.insert(w.id, HostTensor::f32(vec![k, n], fill(seed * 7 + 2, k * n)));
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
        for (i, (a, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (a - c).abs() < 1e-4,
                "GEMM seed {seed} (M={m},K={k},N={n}) elem {i}: gpu {a} vs cpu {c}"
            );
        }
    }

    // flash decode + flash prefill (the region choices): random GQA / D / KV length.
    for seed in 0u64..24 {
        let mut rng = rng_for(seed ^ 0xF1A);
        let hkv = 1 + (rng() % 3) as usize;
        let n_rep = 1 + (rng() % 3) as usize;
        let hq = hkv * n_rep;
        let d = [2usize, 4, 8, 16, 24, 64][(rng() % 6) as usize];
        let cap = 1 + (rng() % 16) as usize; // decode KV length / prefill query rows
        let scale = 0.2 + (rng() % 30) as f32 * 0.03;
        let prefill = seed.is_multiple_of(2);

        let b = Builder::new();
        if prefill {
            let (l, _) = (cap.max(2), 0);
            let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
            let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
            let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
            let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
            let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
            let o = attention_prefill(&b, q, k, v, n_rep, scale, mask);
            let g = b.finish(o);
            let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
                matches!(op, OpKind::FlashAttentionPrefill { .. })
            });
            assert!(
                is_region_prefill(&choice),
                "seed {seed}: prefill should choose the region prefill, got {choice:?}"
            );
            let maskd: Vec<f32> = (0..l * l)
                .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
                .collect();
            let mut inputs = HashMap::new();
            inputs.insert(
                qi,
                HostTensor::f32(vec![1, hq, l, d], fill(seed + 1, hq * l * d)),
            );
            inputs.insert(
                ki,
                HostTensor::f32(vec![1, hkv, l, d], fill(seed + 2, hkv * l * d)),
            );
            inputs.insert(
                vi,
                HostTensor::f32(vec![1, hkv, l, d], fill(seed + 3, hkv * l * d)),
            );
            inputs.insert(mi, HostTensor::f32(vec![1, 1, l, l], maskd));
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
            for (i, (x, c)) in got
                .as_f32()
                .unwrap()
                .iter()
                .zip(cpu.as_f32().unwrap().iter())
                .enumerate()
            {
                assert!(
                    (x - c).abs() <= 1e-3 + 1e-3 * c.abs(),
                    "prefill seed {seed} (hq={hq},l={l},d={d}) elem {i}: gpu {x} vs cpu {c}"
                );
            }
        } else {
            let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
            let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
            let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
            let mask = b.constant("m", TensorType::f32(vec![1, 1, 1, cap]));
            let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
            let o = attention_masked(&b, q, k, v, n_rep, scale, mask);
            let g = b.finish(o);
            let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
                matches!(op, OpKind::FlashAttentionDecode { .. })
            });
            assert!(
                is_region_decode(&choice),
                "seed {seed}: decode should choose the region decode, got {choice:?}"
            );
            // mask: a random KV position masked out (exercise the masked online softmax).
            let mut maskd = vec![0.0f32; cap];
            if cap > 1 {
                maskd[(seed as usize) % cap] = -1.0e9;
            }
            let mut inputs = HashMap::new();
            inputs.insert(
                qi,
                HostTensor::f32(vec![1, hq, 1, d], fill(seed + 1, hq * d)),
            );
            inputs.insert(
                ki,
                HostTensor::f32(vec![1, hkv, cap, d], fill(seed + 2, hkv * cap * d)),
            );
            inputs.insert(
                vi,
                HostTensor::f32(vec![1, hkv, cap, d], fill(seed + 3, hkv * cap * d)),
            );
            inputs.insert(mi, HostTensor::f32(vec![1, 1, 1, cap], maskd));
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
            for (i, (x, c)) in got
                .as_f32()
                .unwrap()
                .iter()
                .zip(cpu.as_f32().unwrap().iter())
                .enumerate()
            {
                assert!(
                    (x - c).abs() <= 1e-3 + 1e-3 * c.abs(),
                    "decode seed {seed} (hq={hq},cap={cap},d={d}) elem {i}: gpu {x} vs cpu {c}"
                );
            }
        }
    }
    eprintln!(
        "synthesized production kernels (GEMM + flash decode/prefill) match CPU over random shapes"
    );
}

#[test]
fn flash_decode_batched_gpu_matches_cpu() {
    // Batched flash decode (q [B,Hq,1,D], k/v [B,Hkv,cap,D], mask [B,1,1,cap]; the planner's region decode, one
    // workgroup per (batch,head)) on the wgpu executor == CPU eval (== attention_masked per batch row). Random
    // shapes, compiled from the traced decomposition. NVPTX parity was ptx-graph-check
    // --decode-batched-flash-check, retired with the rest of that CLI's device modes (card 549) - no numeric
    // NVPTX flash receipt replaces it.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let fill = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 1.4 - 0.7
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
        let bsz = 2 + (rng() % 3) as usize;
        let hkv = 1 + (rng() % 2) as usize;
        let n_rep = 1 + (rng() % 2) as usize;
        let hq = hkv * n_rep;
        let d = [2usize, 4, 8][(rng() % 3) as usize];
        let cap = 1 + (rng() % 5) as usize;
        let scale = 0.2 + (rng() % 20) as f32 * 0.04;
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![bsz, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![bsz, hkv, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![bsz, hkv, cap, d]));
        let mask = b.constant("m", TensorType::f32(vec![bsz, 1, 1, cap]));
        let ids = [q.id, k.id, v.id, mask.id];
        let y = attention_masked(&b, q, k, v, n_rep, scale, mask);
        let g = b.finish(y);
        let md: Vec<f32> = (0..bsz * cap)
            .map(|i| {
                if i % cap == 0 {
                    0.0
                } else {
                    -1.0e9 * ((i % 2) as f32)
                }
            })
            .collect();
        let mut inputs = HashMap::new();
        inputs.insert(
            ids[0],
            HostTensor::f32(vec![bsz, hq, 1, d], fill(seed, bsz * hq * d)),
        );
        inputs.insert(
            ids[1],
            HostTensor::f32(
                vec![bsz, hkv, cap, d],
                fill(seed ^ 0xA5, bsz * hkv * cap * d),
            ),
        );
        inputs.insert(
            ids[2],
            HostTensor::f32(
                vec![bsz, hkv, cap, d],
                fill(seed ^ 0x5A, bsz * hkv * cap * d),
            ),
        );
        inputs.insert(ids[3], HostTensor::f32(vec![bsz, 1, 1, cap], md));
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
        assert_eq!(got.shape(), vec![bsz, hq, 1, d], "seed {seed}");
        for (i, (x, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (x - c).abs() <= 1e-3 + 1e-3 * c.abs(),
                "batched flash (seed {seed} B={bsz},hq={hq},d={d},cap={cap}) elem {i}: gpu {x} vs cpu {c}"
            );
        }
    }
}

#[test]
fn flash_decode_plan_selection() {
    // A traced decode attention with D within the LDS cap compiles to the generated region flash decode on
    // both backends (Card 557: the planner's choice). No GPU.
    //
    // This pins the request contract between the planner and `kg::flash_region_decode`: the shape, the
    // GQA ratio, the scale as its bits (0.5 -> 0x3f000000, not always 1/sqrt(D)) and the mask layout. Card
    // 259: the broadcast `[1,1,1,cap]` and the per-head `[1,Hq,1,cap]` (ALiBi) masks are different
    // requests, since a wrong mask stride is a silently wrong mask read on the shared path every non-ALiBi
    // arch uses.
    use poot_graph_ir::builder::Builder;
    use poot_target::Backend;
    let choice_for_mask = |mask_heads: usize, backend| {
        let (hq, hkv, cap, d) = (4usize, 2usize, 3usize, 4usize);
        let scale = 0.5f32;
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, mask_heads, 1, cap]));
        let y = attention_masked(&b, q, k, v, hq / hkv, scale, mask);
        let target = Target {
            backend,
            caps: poot_test_util::device_caps::default_caps_for(backend),
        };
        compiled_choice(&b.finish(y), &target, |op| {
            matches!(op, OpKind::FlashAttentionDecode { .. })
        })
    };
    for backend in [Backend::SpirvVulkan, Backend::Nvptx] {
        for (mask_heads, per_head) in [(1usize, false), (4, true)] {
            match choice_for_mask(mask_heads, backend) {
                KernelChoice::Generated(KernelRequest::Attention(
                    AttentionSpec::RegionDecode {
                        bsz,
                        hq,
                        n_rep,
                        cap,
                        d,
                        scale_bits,
                        mask_per_head,
                        ..
                    },
                )) => {
                    assert_eq!(
                        (bsz, hq, n_rep, cap, d, scale_bits),
                        (1, 4, 2, 3, 4, 0.5f32.to_bits()),
                        "{backend:?}"
                    );
                    assert_eq!(
                        mask_per_head, per_head,
                        "{backend:?} mask heads {mask_heads}"
                    );
                }
                other => panic!("{backend:?}: expected the generated region decode, got {other:?}"),
            }
        }
    }
}

#[test]
fn flash_decode_gpu_matches_cpu_over_random_shapes() {
    // Property fuzzer for flash decode as the planner lowers it (the generated region decode, Card 557), compiled from
    // the traced decomposition. Over random shapes asserts `gpu.run` == CPU eval
    // (== `attention_masked`), varying GQA ratio (n_rep), KV length (cap), head dim (D), the softmax scale
    // (exercising the `f32::from_bits` meta path with values that are not 1/sqrt(D)), and a random causal-style mask
    // (position 0 always kept, so no row is fully masked). Tolerance, not bit-exact: exp/softmax sum order rounds in f32.
    use poot_graph_ir::builder::Builder;
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
        let hkv = [1usize, 2, 4][(rng() % 3) as usize];
        let n_rep = 1 + (rng() % 3) as usize;
        let hq = hkv * n_rep;
        let cap = 1 + (rng() % 8) as usize;
        let d = [2usize, 4, 8, 16][(rng() % 4) as usize];
        // a scale that is NOT 1/sqrt(d) (so the bits really matter): a small positive value.
        let scale = 0.05 + (rng() % 200) as f32 * 0.01;

        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
        let y = attention_masked(&b, q, k, v, n_rep, scale, mask);
        let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
        let g = b.finish(y);

        // mask: position 0 always kept; each other position randomly masked (additive -1e9).
        let maskd: Vec<f32> = (0..cap)
            .map(|t| {
                if t == 0 || rng() % 2 == 0 {
                    0.0
                } else {
                    -1.0e9
                }
            })
            .collect();
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], fill(seed, hq * d)));
        inputs.insert(
            ki,
            HostTensor::f32(vec![1, hkv, cap, d], fill(seed ^ 0xA5, hkv * cap * d)),
        );
        inputs.insert(
            vi,
            HostTensor::f32(vec![1, hkv, cap, d], fill(seed ^ 0x5A, hkv * cap * d)),
        );
        inputs.insert(mi, HostTensor::f32(vec![1, 1, 1, cap], maskd));

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
        assert_eq!(got.shape(), vec![1, hq, 1, d], "seed {seed}");
        for (i, (x, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (x - c).abs() <= 1e-3 + 1e-3 * c.abs(),
                "seed {seed} (hq={hq},hkv={hkv},cap={cap},d={d},scale={scale}) elem {i}: gpu {x} vs cpu {c}"
            );
        }
    }
}

#[test]
fn flash_prefill_gpu_2d_grid_over_65535_workgroups() {
    // Flash prefill with Hq*L > 65535 (the wgpu gridDim.x cap) runs via the 2-D workgroup grid
    // (gi = GroupY*x_groups + GroupX). Hq=512, L=130 -> Hq*L = 66560 (y_groups=2); the planner chooses the
    // generated region prefill (B=1, D=4, no softcap). gpu.run == CPU eval (== attention_prefill). Card 557:
    // `compile`'s `flash_attention_capped` declines a prefill wider than the target's grid, so the composite
    // is staged directly, as a pass would form it.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, l, d) = (512usize, 512usize, 130usize, 4usize); // Hq*L = 66560 > 65535
    let n_rep = hq / hkv;
    let scale = 0.3f32;
    let fill = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 1.2 - 0.6
            })
            .collect()
    };
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
    let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
    let ids = [q.id, k.id, v.id, mask.id];
    let mut plan = b.append_plan(0);
    let o = plan
        .equation(
            OpKind::FlashAttentionPrefill {
                n_rep,
                scale,
                softcap: None,
            },
            ids.map(poot_graph_ir::Operand::Value).to_vec(),
        )
        .expect("stage the flash prefill");
    plan.declare_result(o).expect("declare the result");
    let mut prepared = b.preflight_append(plan).expect("preflight");
    let id = b.commit_append(&mut prepared).expect("commit");
    let g = b.finish(poot_graph_ir::Traced { id });
    assert!(
        is_region_prefill(&compiled_choice(&g, &target, |op| {
            matches!(op, OpKind::FlashAttentionPrefill { .. })
        })),
        "the over-cap flash prefill plans the generated region prefill"
    );
    let maskd: Vec<f32> = (0..l * l)
        .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(
        ids[0],
        HostTensor::f32(vec![1, hq, l, d], fill(1, hq * l * d)),
    );
    inputs.insert(
        ids[1],
        HostTensor::f32(vec![1, hkv, l, d], fill(2, hkv * l * d)),
    );
    inputs.insert(
        ids[2],
        HostTensor::f32(vec![1, hkv, l, d], fill(3, hkv * l * d)),
    );
    inputs.insert(ids[3], HostTensor::f32(vec![1, 1, l, l], maskd));
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
    assert_eq!(got.shape(), vec![1, hq, l, d]);
    // spot-check all elements within tolerance (online vs direct softmax).
    let max_abs = max_abs_error(got.as_f32().unwrap(), cpu.as_f32().unwrap());
    assert!(
        max_abs <= 2e-3,
        "2-D grid flash prefill max abs diff {max_abs}"
    );
}

#[test]
fn decomposed_prefill_attention_over_65535_query_groups_matches_cpu() {
    // The traced prefill attention chain at Hq*L = 66560 > 65535: `flash_attention_capped` declines the shape
    // on a 65535-wide grid, so the unfused softmax chain reaches the executor. gpu.run == CPU eval
    // (== attention_prefill). Card 1006.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, l, d) = (512usize, 130usize, 4usize);
    let fill = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 1.2 - 0.6
            })
            .collect()
    };
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hq, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hq, l, d]));
    let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
    let ids = [q.id, k.id, v.id, mask.id];
    let o = attention_prefill(&b, q, k, v, 1, 0.3, mask);
    let g = b.finish(o);
    assert!(
        !compile_full(&g, &target)
            .graph()
            .eqns
            .iter()
            .any(|e| matches!(e.op, OpKind::FlashAttentionPrefill { .. })),
        "the over-grid prefill stays the decomposed chain"
    );
    let maskd: Vec<f32> = (0..l * l)
        .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(
        ids[0],
        HostTensor::f32(vec![1, hq, l, d], fill(1, hq * l * d)),
    );
    inputs.insert(
        ids[1],
        HostTensor::f32(vec![1, hq, l, d], fill(2, hq * l * d)),
    );
    inputs.insert(
        ids[2],
        HostTensor::f32(vec![1, hq, l, d], fill(3, hq * l * d)),
    );
    inputs.insert(ids[3], HostTensor::f32(vec![1, 1, l, l], maskd));
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
    assert_eq!(got.shape(), vec![1, hq, l, d]);
    let max_abs = max_abs_error(got.as_f32().unwrap(), cpu.as_f32().unwrap());
    assert!(
        max_abs <= 2e-3,
        "decomposed over-grid prefill attention max abs diff {max_abs}"
    );
}

#[test]
fn flash_prefill_gpu_matches_cpu_over_random_shapes() {
    // Flash prefill (FlashAttentionPrefill, one workgroup per (head, query row), online softmax, running output o[D] in
    // LDS) on the wgpu executor == CPU eval (== attention_prefill_softcap). Both kernels the planner chooses between are
    // covered (Card 557): odd seeds carry an attention-logit softcap, which only the imported `flash_prefill.rs`
    // implements, so they take it; even seeds take the generated region prefill. Sweeps random Hkv / n_rep / D / L
    // with a causal mask; D <= 256 and Hq*L <= 65535. 24 seeds.
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
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 1.4 - 0.7
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
        let hkv = 1 + (rng() % 2) as usize;
        let n_rep = 1 + (rng() % 3) as usize;
        let hq = hkv * n_rep;
        let d = [2usize, 4, 8, 16, 24][(rng() % 5) as usize];
        let l = 2 + (rng() % 14) as usize; // 2..=15 query rows
        let scale = 0.2 + (rng() % 30) as f32 * 0.03; // not always 1/sqrt(D)

        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
        let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
        let ids = [q.id, k.id, v.id, mask.id];
        let softcap = (!seed.is_multiple_of(2)).then_some(30.0f32);
        let o = attention_prefill_softcap(&b, q, k, v, n_rep, scale, mask, softcap);
        let g = b.finish(o);
        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::FlashAttentionPrefill { .. })
        });
        let imported = matches!(
            choice,
            KernelChoice::Imported {
                kernel: ImportedKernel::FlashPrefill,
                ..
            }
        );
        assert_eq!(
            (imported, is_region_prefill(&choice)),
            (softcap.is_some(), softcap.is_none()),
            "seed {seed} softcap {softcap:?}: {choice:?}"
        );
        // additive causal mask: 0 if j<=t else -1e9.
        let maskd: Vec<f32> = (0..l * l)
            .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
            .collect();
        let mut inputs = HashMap::new();
        inputs.insert(
            ids[0],
            HostTensor::f32(vec![1, hq, l, d], fill(seed, hq * l * d)),
        );
        inputs.insert(
            ids[1],
            HostTensor::f32(vec![1, hkv, l, d], fill(seed ^ 0xA5, hkv * l * d)),
        );
        inputs.insert(
            ids[2],
            HostTensor::f32(vec![1, hkv, l, d], fill(seed ^ 0x5A, hkv * l * d)),
        );
        inputs.insert(ids[3], HostTensor::f32(vec![1, 1, l, l], maskd));

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
        assert_eq!(got.shape(), vec![1, hq, l, d], "seed {seed}");
        for (i, (x, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (x - c).abs() <= 1e-3 + 1e-3 * c.abs(),
                "flash prefill GPU (seed {seed}: hq={hq},hkv={hkv},d={d},l={l},scale={scale},softcap={softcap:?}) elem {i}: gpu {x} vs cpu {c}"
            );
        }
    }
}

#[test]
fn synthesized_tiled_region_element_by_element_matches_cpu() {
    // A bare M>1 F32 MatMul compiles to the synthesizer's tile-IR Body generator `kg::tiled_region` (Card 557: the
    // planner's contraction choice), which generates the LDS-staged tiled GEMM from the (M,K,N,tile) descriptor
    // instead of reusing the hand-authored kernel. Confirm the compiled choice is the synthesized kernel and it
    // runs GPU==CPU over partial-tile shapes in all three dims (RADV safety and masking). Foundation for the dequant
    // load and online-softmax accumulation variants.
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
    // partial tiles in all dims (GEMM_TILE=8, coarsened 2 rows/lane -> 16-row tiles): M not a mult of 16,
    // N not a mult of 8, K not a mult of 8; plus an exact-tile case and a tall/narrow one.
    for (seed, (m, k, n)) in [
        (2usize, 16usize, 8usize),
        (17, 32, 24),
        (6, 40, 33),
        (16, 8, 8),
        (33, 5, 41),
    ]
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
            "expected the single-dispatch synthesized tiled_region choice, got {choice:?}"
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
    eprintln!("synthesized tiled GEMM (element-by-element kg::tiled_region) runs GPU==CPU");
}

#[test]
fn synthesized_tiled_region_perf_is_competitive_with_imported() {
    // The synthesized tiled GEMM (`kg::tiled_region`, the default) must not be materially slower than the imported coarsened
    // kernel (same structure, so perf should be ~equal). Time the same GEMM compiled under `FusionPolicy::MoeHangGuard`
    // (which keeps it off the generated tiled GEMM: the imported coarsened kernel, `is_tiled_gemm`) vs `FusionPolicy::Full`
    // (the synthesized default, Card 557's contraction choice) over many iterations in one run (the box is noisy; the
    // ratio is the signal). Assert the synthesized path is within 2x (a generous sanity bound, not a perf claim).
    use poot_graph_plan::FusionPolicy;
    use std::time::Instant;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (m, k, n) = (128usize, 2048usize, 2048usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(x, w);
    let g = b.finish(out);
    let choice_under = |fusion: FusionPolicy| {
        let program = staged_replay_with(&g, target, fusion);
        let stage = program.stages().next().expect("single stage").2;
        let eqn = stage
            .graph()
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul))
            .expect("the matmul");
        stage.kernel_choice(eqn).clone()
    };
    let synth_choice = choice_under(FusionPolicy::Full);
    assert!(
        is_tiled_region(&synth_choice),
        "Full should choose the synthesized tiled GEMM: {synth_choice:?}"
    );
    let imported_choice = choice_under(FusionPolicy::MoeHangGuard);
    assert!(
        !is_tiled_region(&imported_choice),
        "MoeHangGuard should leave the matmul on the imported tiled GEMM: {imported_choice:?}"
    );
    let xd: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
    let wd: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.08 - 0.16).collect();
    let mut inputs = HashMap::new();
    inputs.insert(x.id, HostTensor::f32(vec![m, k], xd));
    inputs.insert(w.id, HostTensor::f32(vec![k, n], wd));

    // Card 546b: times the entry's replayed `step` only (not `run_via_contract`'s per-call
    // compile/add_entry/remove_entry/unload, which would charge planning overhead equally to both
    // graphs and dilute the real kernel-time ratio this test measures). `add_entry`'s own first
    // `step` records; that call is the warmup, and only the following replayed steps are timed.
    let time = |exec: &mut dyn Executor, fusion: FusionPolicy| -> f64 {
        let (exe, entry) = add_resident_entry_with(exec, target, &g, &inputs, fusion);
        step_resident(exec, exe, entry, &g); // warmup: records the entry
        let iters = 30;
        let t0 = Instant::now();
        for _ in 0..iters {
            step_resident(exec, exe, entry, &g);
        }
        let dt = t0.elapsed().as_secs_f64() / iters as f64;
        exec.remove_entry(exe, entry).unwrap();
        exec.unload(exe).unwrap();
        dt
    };
    let t_imported = time(&mut gpu, FusionPolicy::MoeHangGuard);
    let t_synth = time(&mut gpu, FusionPolicy::Full);
    let ratio = t_synth / t_imported.max(1e-9);
    eprintln!(
        "tiled GEMM {m}x{k}x{n}: imported {:.3}ms vs synthesized {:.3}ms (ratio {ratio:.2}x)",
        t_imported * 1e3,
        t_synth * 1e3
    );
    assert!(
        ratio < 2.0,
        "synthesized tiled GEMM should be within 2x of imported (got {ratio:.2}x); same structure, so a big \
         gap signals a codegen regression"
    );
}

#[test]
fn synthesized_tiled_region_chunked_large_matches_cpu() {
    // The synthesized tiled GEMM (`kg::tiled_region`) also chunks a >= 2^15-tile GEMM into sub-2^15 dispatches (each a baked
    // `tile_offset` variant), so the synthesized path handles any size on RADV. Same 544x8192 shape as the imported chunked
    // test (34816 tiles -> 2 chunks); the planner's contraction choice (Card 557) takes the synthesized tiled GEMM, and
    // this asserts the compiled choice is chunked and `gpu.run == CPU eval`.
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
    let n_chunks = match compiled_choice(&g, &spirv_fixture_target(), |op| {
        matches!(op, OpKind::MatMul)
    }) {
        KernelChoice::Chunked(choices) => {
            assert!(
                choices.iter().all(|choice| matches!(
                    choice,
                    KernelChoice::Generated(KernelRequest::Contraction(
                        ContractionSpec::TiledRegion { .. }
                    ))
                )),
                "chunks should be the synthesized tiled_region kernel: {choices:?}"
            );
            choices.len()
        }
        other => {
            panic!("expected a chunked choice for a 34816-tile synthesized GEMM, got {other:?}")
        }
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
    eprintln!("synthesized chunked tiled GEMM: {m}x{k}x{n} -> 2 chunks, GPU==CPU");
}

/// Card 259: real-wgpu dispatch of the per-head flash-attention mask, the layout ALiBi needs. Covers every flash kernel
/// the planner chooses (Card 557) in one GPU session: the synthesized region decode and region prefill (strides baked
/// into the generated Body) and the imported flash prefill, chosen for a softcapped prefill (per-head stride rides as
/// the `dims[8]` metadata word). Each is checked against the CPU oracle, which is checked against a hand-rolled ALiBi
/// forward pass in `poot-eval`'s `alibi_attention` tests.
///
/// The mask is a real ALiBi mask (visibility plus `-slope[h]*(pos-t)`) and every head has a different slope, so a kernel that
/// dropped the head stride would score every head with head 0's row. The final per-path assertion pins that: the per-head GPU
/// result must differ from a SEPARATE graph, statically typed with a broadcast (`Hm=1`) mask and bound to head 0's row,
/// run on the same q/k/v. Card 531c: this used to substitute a smaller mask tensor into the per-head graph at runtime,
/// which the planner (still typed `Hm=Hq` from the graph's static shape) indexed with per-head strides — an out-of-bounds
/// read that the old silently-erased `Assert` hid. Comparing two properly-typed graphs instead keeps the same coverage
/// without relying on undefined behavior.
#[test]
fn flash_per_head_alibi_mask_gpu_matches_cpu() {
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, d) = (4usize, 2usize, 4usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let slopes = [0.25f32, 0.0625, 0.015625, 0.00390625];
    let fill = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    };
    // Per-head ALiBi decode mask row [hq, cap] and prefill plane stack [hq, l, l].
    let decode_mask = |cap: usize, pos: usize| -> Vec<f32> {
        (0..hq)
            .flat_map(|h| {
                (0..cap).map(move |t| {
                    if t <= pos {
                        -slopes[h] * (pos as f32 - t as f32)
                    } else {
                        -1.0e9
                    }
                })
            })
            .collect()
    };
    let prefill_mask = |l: usize| -> Vec<f32> {
        (0..hq)
            .flat_map(|h| {
                (0..l).flat_map(move |i| {
                    (0..l).map(move |j| {
                        if j <= i {
                            -slopes[h] * (i as f32 - j as f32)
                        } else {
                            -1.0e30
                        }
                    })
                })
            })
            .collect()
    };

    // Compare gpu.run vs the CPU oracle, then run a SEPARATE graph whose mask is statically typed
    // broadcast (Hm=1, head 0's slab only) and require a material difference from the per-head result
    // (the dropped-head-stride mutation). The broadcast run binds a tensor that matches its own graph's
    // declared mask shape exactly: unlike an out-of-bounds substitution into the per-head graph, this
    // never reads past the mask buffer it was actually given.
    let check = |label: &str,
                 g: &poot_graph_ir::Graph,
                 inputs: &HashMap<poot_graph_ir::ValueId, HostTensor>,
                 gb: &poot_graph_ir::Graph,
                 bcast_inputs: &HashMap<poot_graph_ir::ValueId, HostTensor>,
                 gpu: &mut dyn Executor| {
        let eval_inputs: HashMap<_, Value> = inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect();
        let cpu = eval(g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();
        let got = run_via_contract(gpu, target, g, inputs);
        assert_eq!(got.shape(), cpu.shape(), "{label}: shape");
        for (i, (x, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!((x - c).abs() < 1e-4, "{label} elem {i}: gpu {x} vs cpu {c}");
        }
        let flat = run_via_contract(gpu, target, gb, bcast_inputs);
        let max_diff = max_abs_error(flat.as_f32().unwrap(), got.as_f32().unwrap());
        assert!(
            max_diff > 1e-4,
            "{label}: the per-head mask must not collapse to head 0's slab on the GPU; \
             max_diff={max_diff:.2e}"
        );
        eprintln!(
            "card 259 {label}: per-head ALiBi flash GPU == CPU, head-stride mutation diff {max_diff:.3e}"
        );
    };

    // Each traced graph's compiled choice must be the kernel the label names (Card 557: the planner's choice).
    let assert_choice =
        |label: &str, g: &poot_graph_ir::Graph, want: &dyn Fn(&KernelChoice) -> bool| {
            let choice = compiled_choice(g, &spirv_fixture_target(), |op| {
                matches!(
                    op,
                    OpKind::FlashAttentionDecode { .. } | OpKind::FlashAttentionPrefill { .. }
                )
            });
            assert!(want(&choice), "{label}: unexpected choice {choice:?}");
        };
    let is_imported_prefill = |choice: &KernelChoice| {
        matches!(
            choice,
            KernelChoice::Imported {
                kernel: ImportedKernel::FlashPrefill,
                ..
            }
        )
    };

    // (1) the synthesized region decode, the planner's only decode choice within the LDS cap.
    {
        let (cap, pos) = (6usize, 3usize);
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, hq, 1, cap]));
        let y = attention_masked(&b, q, k, v, n_rep, scale, mask);
        let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
        let g = b.finish(y);
        let md = decode_mask(cap, pos);
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], fill(1, hq * d)));
        inputs.insert(
            ki,
            HostTensor::f32(vec![1, hkv, cap, d], fill(2, hkv * cap * d)),
        );
        inputs.insert(
            vi,
            HostTensor::f32(vec![1, hkv, cap, d], fill(3, hkv * cap * d)),
        );
        inputs.insert(mi, HostTensor::f32(vec![1, hq, 1, cap], md.clone()));
        let head0 = md[..cap].to_vec();

        // A separate graph whose mask is statically typed broadcast ([1,1,1,cap], Hm=1): the planner
        // resolves it to the broadcast stride (0, cap), so binding head 0's slab here is in-bounds by
        // construction, unlike substituting a smaller buffer into the per-head graph above.
        let bb = Builder::new();
        let bq = bb.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let bk = bb.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
        let bv = bb.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
        let bmask = bb.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
        let by = attention_masked(&bb, bq, bk, bv, n_rep, scale, bmask);
        let (bqi, bki, bvi, bmi) = (bq.id, bk.id, bv.id, bmask.id);
        let gb = bb.finish(by);
        let mut bcast_inputs = HashMap::new();
        bcast_inputs.insert(bqi, HostTensor::f32(vec![1, hq, 1, d], fill(1, hq * d)));
        bcast_inputs.insert(
            bki,
            HostTensor::f32(vec![1, hkv, cap, d], fill(2, hkv * cap * d)),
        );
        bcast_inputs.insert(
            bvi,
            HostTensor::f32(vec![1, hkv, cap, d], fill(3, hkv * cap * d)),
        );
        bcast_inputs.insert(bmi, HostTensor::f32(vec![1, 1, 1, cap], head0));

        assert_choice("synthesized flash region decode", &g, &is_region_decode);
        assert_choice(
            "synthesized flash region decode (bcast)",
            &gb,
            &is_region_decode,
        );
        check(
            "synthesized flash region decode",
            &g,
            &inputs,
            &gb,
            &bcast_inputs,
            &mut gpu,
        );
    }

    // (2) the synthesized region prefill (no softcap) + (3) the imported flash prefill, which the planner
    // chooses for a softcapped prefill (the region prefill has no softcap). The large cap keeps the softcap
    // nearly linear over these scores, so the per-head mask difference still shows.
    for (label, softcap) in [
        ("synthesized flash region prefill", None),
        ("imported flash prefill", Some(50.0f32)),
    ] {
        let l = 6usize;
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
        let mask = b.constant("m", TensorType::f32(vec![1, hq, l, l]));
        let y = attention_prefill_softcap(&b, q, k, v, n_rep, scale, mask, softcap);
        let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
        let g = b.finish(y);
        let md = prefill_mask(l);
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, l, d], fill(4, hq * l * d)));
        inputs.insert(
            ki,
            HostTensor::f32(vec![1, hkv, l, d], fill(5, hkv * l * d)),
        );
        inputs.insert(
            vi,
            HostTensor::f32(vec![1, hkv, l, d], fill(6, hkv * l * d)),
        );
        inputs.insert(mi, HostTensor::f32(vec![1, hq, l, l], md.clone()));
        let head0 = md[..l * l].to_vec();

        // A separate graph whose mask is statically typed broadcast ([1,1,l,l], Hm=1), mirroring the
        // decode block above.
        let bb = Builder::new();
        let bq = bb.constant("q", TensorType::f32(vec![1, hq, l, d]));
        let bk = bb.constant("k", TensorType::f32(vec![1, hkv, l, d]));
        let bv = bb.constant("v", TensorType::f32(vec![1, hkv, l, d]));
        let bmask = bb.constant("m", TensorType::f32(vec![1, 1, l, l]));
        let by = attention_prefill_softcap(&bb, bq, bk, bv, n_rep, scale, bmask, softcap);
        let (bqi, bki, bvi, bmi) = (bq.id, bk.id, bv.id, bmask.id);
        let gb = bb.finish(by);
        let mut bcast_inputs = HashMap::new();
        bcast_inputs.insert(bqi, HostTensor::f32(vec![1, hq, l, d], fill(4, hq * l * d)));
        bcast_inputs.insert(
            bki,
            HostTensor::f32(vec![1, hkv, l, d], fill(5, hkv * l * d)),
        );
        bcast_inputs.insert(
            bvi,
            HostTensor::f32(vec![1, hkv, l, d], fill(6, hkv * l * d)),
        );
        bcast_inputs.insert(bmi, HostTensor::f32(vec![1, 1, l, l], head0));

        let want: &dyn Fn(&KernelChoice) -> bool = if softcap.is_some() {
            &is_imported_prefill
        } else {
            &is_region_prefill
        };
        assert_choice(label, &g, want);
        assert_choice(label, &gb, want);
        check(label, &g, &inputs, &gb, &bcast_inputs, &mut gpu);
    }
}
