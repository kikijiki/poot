use super::*;
use poot_graph_ir::AttentionAttrs;
use poot_graph_ir::ops::{attention_prefill, attention_prefill_softcap};
use poot_graph_plan::KernelChoice;
use poot_kernelgen as kg;
use poot_kernelgen::{AttentionSpec, ContractionSpec, KernelRequest, MaskKind, ScoreTransform};

/// The attention request a choice generated, if it generated one.
fn generated_attention(choice: &KernelChoice) -> Option<&AttentionSpec> {
    match choice {
        KernelChoice::Generated(KernelRequest::Attention(spec)) => Some(spec),
        _ => None,
    }
}

/// Whether a choice is the generated attention kernel over one query row per sequence (decode).
fn is_attention_decode(choice: &KernelChoice) -> bool {
    generated_attention(choice).is_some_and(|spec| spec.lq == 1)
}

/// Whether a choice is the generated attention kernel over several query rows (prefill).
fn is_attention_prefill(choice: &KernelChoice) -> bool {
    generated_attention(choice).is_some_and(|spec| spec.lq > 1)
}

/// Whether a choice is the generated attention kernel carrying an attention-logit softcap.
fn is_softcapped_attention(choice: &KernelChoice) -> bool {
    generated_attention(choice)
        .is_some_and(|spec| matches!(spec.score, ScoreTransform::ScaleSoftcap { .. }))
}

/// Whether a choice is the register-tiled GEMM, one dispatch or split.
fn is_register_tiled(choice: &KernelChoice) -> bool {
    matches!(
        choice,
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(_)))
    )
}

#[test]
fn flash_decode_gpu_matches_cpu() {
    // The traced decode attention chain, which `compile` fuses into `Attention` and the planner
    // lowers to the generated attention kernel, matches CPU eval of the same graph (`attention_masked`). GQA with Hq=4/Hkv=2 (n_rep=2), cap=3 with the last position masked
    // out, D=4, scale=0.5.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, cap, d) = (4usize, 2usize, 3usize, 4usize);
    let n_rep = hq / hkv;
    let scale = 0.5f32;
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
    let y = attention_masked(&b, q, k, v, n_rep, scale, mask).unwrap();
    let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
    let g = b.finish(y).unwrap();
    let choice = compiled_choice(&g, &target, |op| matches!(op, OpKind::Attention(_)));
    assert!(is_attention_decode(&choice), "{choice:?}");
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
    let cpu = eval(&g, &eval_inputs, EvalOptions::default())
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
fn attention_decode_matches_cpu() {
    // A traced decode attention compiles to `Attention`, which the planner lowers to the generated attention
    // kernel (online softmax, partials in LDS). Confirm the compiled choice is that kernel and `gpu.run == eval` (== attention_masked) over a couple
    // of head dims, with the last KV position masked out.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    for &(hq, hkv, cap, d) in &[(4usize, 2usize, 5usize, 4usize), (8, 8, 7, 64)] {
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
        let y = attention_masked(&b, q, k, v, n_rep, scale, mask).unwrap();
        let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
        let g = b.finish(y).unwrap();

        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::Attention(_))
        });
        assert!(
            is_attention_decode(&choice),
            "expected the generated attention decode, got {choice:?}"
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
        let cpu = eval(&g, &eval_inputs, EvalOptions::default())
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

/// Every row walks its keys in several passes of the kernel the device plans for it (a key per lane for the
/// streaming and split schedules, a staged key block for the tiled one), so the running maximum, denominator and
/// output are rescaled between passes: a kernel that skipped the rescale, or walked only its first pass, passes
/// only a fixture one pass covers. The key count is derived from that pass: a probe of the row's shape is
/// planned on the device, the row takes three and a half of its passes (the last one partial), and the planned
/// row is checked to still need at least three. The rows: decode at `B = 1` (64 wide), decode at D = 512 (the
/// head width SPIR-V used to refuse), batched decode at `B = 4` (D = 128) and a causal prefill (D = 64, the
/// tiled schedule), each against the CPU oracle with a mask that hides a different tail per batch row.
///
/// Mutation: in `attention/tiled.rs`, end the key-block walk after its first block (`count_to` over
/// `keys.first..keys.first + bc`); the prefill row goes red against the oracle (112 keys over 32-key blocks on a
/// 128-lane tiling: max abs diff 0.149).
#[test]
fn attention_over_many_key_tiles_matches_cpu() {
    use poot_graph_ir::builder::Builder;
    // (batch, hq, hkv, prefill, d): a decode row has one query; a prefill row has as many queries as keys.
    let rows: [(usize, usize, usize, bool, usize); 4] = [
        (1, 4, 2, false, 64),
        (1, 2, 1, false, 512),
        (4, 4, 2, false, 128),
        (1, 2, 2, true, 64),
    ];
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
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
    for (batch, hq, hkv, prefill, d) in rows {
        let n_rep = hq / hkv;
        let scale = 1.0 / (d as f32).sqrt();
        // The traced chain over `lk` keys, its operand ids, and the keys of one pass of the kernel the device
        // plans for it.
        let traced = |lk: usize| {
            let lq = if prefill { lk } else { 1 };
            let b = Builder::for_tests();
            let q = b
                .constant("q", TensorType::f32(vec![batch, hq, lq, d]))
                .unwrap();
            let k = b
                .constant("k", TensorType::f32(vec![batch, hkv, lk, d]))
                .unwrap();
            let v = b
                .constant("v", TensorType::f32(vec![batch, hkv, lk, d]))
                .unwrap();
            let mask = b
                .constant("mask", TensorType::f32(vec![batch, 1, lq, lk]))
                .unwrap();
            let ids = [q.id(), k.id(), v.id(), mask.id()];
            let y = attention_masked(&b, q, k, v, n_rep, scale, mask).unwrap();
            let g = b.finish(y).unwrap();
            let choice = compiled_choice(&g, &target, |op| matches!(op, OpKind::Attention(_)));
            let schedule = generated_attention(&choice)
                .unwrap_or_else(|| {
                    panic!("d={d}: expected the generated attention, got {choice:?}")
                })
                .schedule;
            let pass = match schedule {
                kg::AttentionSchedule::Tiled { bc, .. } => bc,
                _ => schedule.lanes(),
            };
            (g, ids, schedule, pass)
        };
        let (_, _, _, probe_pass) = traced(256);
        let lk = 3 * probe_pass + probe_pass / 2;
        let (g, ids, schedule, pass) = traced(lk);
        let lq = if prefill { lk } else { 1 };
        assert!(
            lk > 3 * pass,
            "d={d}: {lk} keys must take at least three passes of {pass} ({schedule:?})"
        );
        if prefill {
            assert!(
                matches!(schedule, kg::AttentionSchedule::Tiled { .. }),
                "the prefill row plans the tiled schedule, got {schedule:?}"
            );
        }

        // Batch row `bi` shows its first `lk - 7 * bi` keys (decode), or the causal prefix (prefill).
        let md: Vec<f32> = (0..batch * lq * lk)
            .map(|i| {
                let (bi, row, key) = (i / (lq * lk), (i / lk) % lq, i % lk);
                let visible = if lq == 1 {
                    key < lk - 7 * bi
                } else {
                    key <= row
                };
                if visible { 0.0 } else { -1.0e9 }
            })
            .collect();
        let mut inputs = HashMap::new();
        inputs.insert(
            ids[0],
            HostTensor::f32(vec![batch, hq, lq, d], fill(1, batch * hq * lq * d)),
        );
        inputs.insert(
            ids[1],
            HostTensor::f32(vec![batch, hkv, lk, d], fill(2, batch * hkv * lk * d)),
        );
        inputs.insert(
            ids[2],
            HostTensor::f32(vec![batch, hkv, lk, d], fill(3, batch * hkv * lk * d)),
        );
        inputs.insert(ids[3], HostTensor::f32(vec![batch, 1, lq, lk], md));
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
        assert_eq!(got.shape(), vec![batch, hq, lq, d]);
        let max_abs = max_abs_error(got.as_f32().unwrap(), cpu.as_f32().unwrap());
        assert!(
            max_abs <= 1e-3,
            "B={batch} Hq={hq} Lq={lq} Lk={lk} D={d} ({schedule:?}): max abs diff {max_abs}"
        );
    }
}

/// Prefill into a cache: `1 < Lq < Lk`. Three new query rows attend an 8-slot cache whose first five slots
/// hold the prefix, so row `i` sees keys `0..=5 + i`. The traced chain goes through the matcher, the planner
/// and the device, and is compared with the oracle. A kernel that indexed the mask row with `Lq` instead of
/// `Lk` reads the wrong plane offset and goes red.
#[test]
fn attention_prefill_into_a_cache_matches_cpu() {
    use poot_graph_ir::builder::Builder;
    let (hq, hkv, lq, lk, d) = (4usize, 2usize, 3usize, 8usize, 16usize);
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
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
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, hq, lq, d]))
        .unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, lk, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hkv, lk, d]))
        .unwrap();
    let mask = b
        .constant("m", TensorType::f32(vec![1, 1, lq, lk]))
        .unwrap();
    let ids = [q.id(), k.id(), v.id(), mask.id()];
    let y = attention_prefill(&b, q, k, v, hq / hkv, 0.25, mask).unwrap();
    let g = b.finish(y).unwrap();
    let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
        matches!(op, OpKind::Attention(_))
    });
    let spec = generated_attention(&choice)
        .unwrap_or_else(|| panic!("expected the generated attention, got {choice:?}"));
    assert_eq!((spec.lq, spec.lk), (lq, lk), "{choice:?}");
    let md: Vec<f32> = (0..lq * lk)
        .map(|i| if i % lk <= 5 + i / lk { 0.0 } else { -1.0e9 })
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(
        ids[0],
        HostTensor::f32(vec![1, hq, lq, d], fill(1, hq * lq * d)),
    );
    inputs.insert(
        ids[1],
        HostTensor::f32(vec![1, hkv, lk, d], fill(2, hkv * lk * d)),
    );
    inputs.insert(
        ids[2],
        HostTensor::f32(vec![1, hkv, lk, d], fill(3, hkv * lk * d)),
    );
    inputs.insert(ids[3], HostTensor::f32(vec![1, 1, lq, lk], md));
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
    assert_eq!(got.shape(), vec![1, hq, lq, d]);
    let max_abs = max_abs_error(got.as_f32().unwrap(), cpu.as_f32().unwrap());
    assert!(
        max_abs <= 1e-4,
        "prefill into a cache: max abs diff {max_abs}"
    );
}

#[test]
fn attention_batched_decode_matches_cpu() {
    // The generated attention kernel handles batched (B>1) decode: the flat workgroup index is
    // `(batch*Hq + head)*Lq + row`, with the q/k/v/mask offsets following the batch. Compile a traced B>1
    // attention, confirm the planner chose the generated kernel, and check `gpu.run == eval`
    // (== attention_masked per batch row) over random shapes. B=1 is covered by
    // `attention_decode_matches_cpu`.
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
        let b = Builder::for_tests();
        let q = b
            .constant("q", TensorType::f32(vec![bsz, hq, 1, d]))
            .unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![bsz, hkv, cap, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![bsz, hkv, cap, d]))
            .unwrap();
        let mask = b
            .constant("m", TensorType::f32(vec![bsz, 1, 1, cap]))
            .unwrap();
        let ids = [q.id(), k.id(), v.id(), mask.id()];
        let y = attention_masked(&b, q, k, v, n_rep, scale, mask).unwrap();
        let g = b.finish(y).unwrap();

        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::Attention(_))
        });
        assert!(
            is_attention_decode(&choice),
            "expected the generated attention decode, got {choice:?}"
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
        let cpu = eval(&g, &eval_inputs, EvalOptions::default())
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
fn attention_prefill_matches_cpu() {
    // A traced prefill attention compiles to `Attention`, which the planner lowers to the generated attention
    // kernel (online softmax, partials in LDS, one workgroup per (head, row) on a 2-D grid). Confirm the
    // compiled choice is that kernel and `gpu.run == eval` (== attention_prefill) with a causal mask.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    for &(hq, hkv, l, d) in &[(4usize, 2usize, 6usize, 4usize), (6, 6, 9, 16)] {
        let n_rep = hq / hkv;
        let scale = 1.0 / (d as f32).sqrt();
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hkv, l, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hkv, l, d]))
            .unwrap();
        let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l])).unwrap();
        let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
        let o = attention_prefill(&b, q, k, v, n_rep, scale, mask).unwrap();
        let g = b.finish(o).unwrap();

        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::Attention(_))
        });
        assert!(
            is_attention_prefill(&choice),
            "expected the generated attention prefill, got {choice:?}"
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
        let cpu = eval(&g, &eval_inputs, EvalOptions::default())
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
fn attention_prefill_d512_matches_cpu() {
    // Card 185: gemma4's global attention layers use head_dim D=512. The generated attention kernel sizes its
    // LDS partials from the device's workgroup memory (a few lanes at D=512), so the planner must choose it
    // for a D=512 prefill (not reject it), and the kernel must be correct (gpu.run == eval). Small L/Hq keeps the run
    // fast; varied random inputs so an indexing bug or near-tie shows in max_abs/max_rel.
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, l, d) = (2usize, 2usize, 8usize, 512usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d])).unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, l, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hkv, l, d]))
        .unwrap();
    let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l])).unwrap();
    let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
    let o = attention_prefill(&b, q, k, v, n_rep, scale, mask).unwrap();
    let g = b.finish(o).unwrap();

    let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
        matches!(op, OpKind::Attention(_))
    });
    assert!(
        is_attention_prefill(&choice),
        "the B=1 D=512 prefill should choose the generated attention, got {choice:?}"
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
    let cpu = eval(&g, &eval_inputs, EvalOptions::default())
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

    // dense tiled GEMM (the register-tiled choice): random partial-tile M/K/N.
    for seed in 0u64..30 {
        let mut rng = rng_for(seed);
        let (m, k, n) = (
            2 + (rng() % 39) as usize,
            1 + (rng() % 40) as usize,
            1 + (rng() % 40) as usize,
        );
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
        let out = b.matmul(x, w).unwrap();
        let g = b.finish(out).unwrap();
        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::MatMul { .. })
        });
        assert!(
            is_register_tiled(&choice),
            "seed {seed}: an M>1 matmul should choose the tiled GEMM, got {choice:?}"
        );
        let mut inputs = HashMap::new();
        inputs.insert(
            x.id(),
            HostTensor::f32(vec![m, k], fill(seed * 7 + 1, m * k)),
        );
        inputs.insert(
            w.id(),
            HostTensor::f32(vec![k, n], fill(seed * 7 + 2, k * n)),
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

        let b = Builder::for_tests();
        if prefill {
            let (l, _) = (cap.max(2), 0);
            let q = b.constant("q", TensorType::f32(vec![1, hq, l, d])).unwrap();
            let k = b
                .constant("k", TensorType::f32(vec![1, hkv, l, d]))
                .unwrap();
            let v = b
                .constant("v", TensorType::f32(vec![1, hkv, l, d]))
                .unwrap();
            let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l])).unwrap();
            let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
            let o = attention_prefill(&b, q, k, v, n_rep, scale, mask).unwrap();
            let g = b.finish(o).unwrap();
            let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
                matches!(op, OpKind::Attention(_))
            });
            assert!(
                is_attention_prefill(&choice),
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
            let cpu = eval(&g, &eval_inputs, EvalOptions::default())
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
            let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
            let k = b
                .constant("k", TensorType::f32(vec![1, hkv, cap, d]))
                .unwrap();
            let v = b
                .constant("v", TensorType::f32(vec![1, hkv, cap, d]))
                .unwrap();
            let mask = b
                .constant("m", TensorType::f32(vec![1, 1, 1, cap]))
                .unwrap();
            let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
            let o = attention_masked(&b, q, k, v, n_rep, scale, mask).unwrap();
            let g = b.finish(o).unwrap();
            let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
                matches!(op, OpKind::Attention(_))
            });
            assert!(
                is_attention_decode(&choice),
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
            let cpu = eval(&g, &eval_inputs, EvalOptions::default())
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
    // shapes, compiled from the traced decomposition. No numeric NVPTX flash receipt exists
    // (the old `--decode-batched-flash-check` CLI mode was retired with the rest of that CLI, card 549).
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
        let b = Builder::for_tests();
        let q = b
            .constant("q", TensorType::f32(vec![bsz, hq, 1, d]))
            .unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![bsz, hkv, cap, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![bsz, hkv, cap, d]))
            .unwrap();
        let mask = b
            .constant("m", TensorType::f32(vec![bsz, 1, 1, cap]))
            .unwrap();
        let ids = [q.id(), k.id(), v.id(), mask.id()];
        let y = attention_masked(&b, q, k, v, n_rep, scale, mask).unwrap();
        let g = b.finish(y).unwrap();
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
        let cpu = eval(&g, &eval_inputs, EvalOptions::default())
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
    // This pins the request contract between the planner and the attention generator: the shape, the
    // GQA ratio, the scale as its bits (0.5 -> 0x3f000000, not always 1/sqrt(D)) and the mask layout. Card
    // 259: the broadcast `[1,1,1,cap]` and the per-head `[1,Hq,1,cap]` (ALiBi) masks are different
    // requests, since a wrong mask stride is a silently wrong mask read on the shared path every non-ALiBi
    // arch uses.
    use poot_graph_ir::builder::Builder;
    use poot_target::Backend;
    let choice_for_mask = |mask_heads: usize, backend| {
        let (hq, hkv, cap, d) = (4usize, 2usize, 3usize, 4usize);
        let scale = 0.5f32;
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hkv, cap, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hkv, cap, d]))
            .unwrap();
        let mask = b
            .constant("mask", TensorType::f32(vec![1, mask_heads, 1, cap]))
            .unwrap();
        let y = attention_masked(&b, q, k, v, hq / hkv, scale, mask).unwrap();
        let target = Target::Gpu {
            backend,
            caps: poot_test_util::device_caps::default_caps_for(backend),
        };
        compiled_choice(&b.finish(y).unwrap(), &target, |op| {
            matches!(op, OpKind::Attention(_))
        })
    };
    for backend in [Backend::SpirvVulkan, Backend::Nvptx] {
        for (mask_heads, per_head) in [(1usize, false), (4, true)] {
            let choice = choice_for_mask(mask_heads, backend);
            let Some(spec) = generated_attention(&choice) else {
                panic!("{backend:?}: expected the generated attention, got {choice:?}");
            };
            assert_eq!(
                (spec.batch, spec.hq, spec.n_rep, spec.lq, spec.lk, spec.dk),
                (1, 4, 2, 1, 3, 4),
                "{backend:?}"
            );
            assert_eq!(
                spec.score,
                ScoreTransform::Scale {
                    scale_bits: 0.5f32.to_bits()
                },
                "{backend:?}"
            );
            assert_eq!(
                spec.mask == MaskKind::PerHead,
                per_head,
                "{backend:?} mask heads {mask_heads}"
            );
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
        let y = attention_masked(&b, q, k, v, n_rep, scale, mask).unwrap();
        let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
        let g = b.finish(y).unwrap();

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
        let cpu = eval(&g, &eval_inputs, EvalOptions::default())
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
    // generated region prefill (B=1, D=4, no softcap). gpu.run == CPU eval (== attention_prefill). The
    // composite is staged directly; the traced chain through `compile` is the next test.
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
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d])).unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, l, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hkv, l, d]))
        .unwrap();
    let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l])).unwrap();
    let ids = [q.id(), k.id(), v.id(), mask.id()];
    let mut plan = b.append_plan(0).unwrap();
    let o = plan
        .equation(
            OpKind::Attention(AttentionAttrs::new(n_rep, scale, None)),
            ids.map(poot_graph_ir::Operand::Value).to_vec(),
        )
        .expect("stage the flash prefill");
    plan.declare_result(o).expect("declare the result");
    let mut prepared = b.preflight_append(plan).expect("preflight");
    let out = b.commit_append(&mut prepared).expect("commit");
    let g = b.finish(out).unwrap();
    assert!(
        is_attention_prefill(&compiled_choice(&g, &target, |op| {
            matches!(op, OpKind::Attention(_))
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
    let cpu = eval(&g, &eval_inputs, EvalOptions::default())
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
fn traced_prefill_attention_over_65535_query_groups_fuses_and_matches_cpu() {
    // The traced prefill attention chain at Hq*L = 66560 > 65535: `compile` fuses it to the flash kernel,
    // which folds its rows onto a 2-D grid (POOT-1308: declining it kept the whole `[Hq, L, capacity]`
    // scores, 3.8 GB at an 8192-token prefill). gpu.run == CPU eval (== attention_prefill).
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
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d])).unwrap();
    let k = b.constant("k", TensorType::f32(vec![1, hq, l, d])).unwrap();
    let v = b.constant("v", TensorType::f32(vec![1, hq, l, d])).unwrap();
    let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l])).unwrap();
    let ids = [q.id(), k.id(), v.id(), mask.id()];
    let o = attention_prefill(&b, q, k, v, 1, 0.3, mask).unwrap();
    let g = b.finish(o).unwrap();
    assert!(
        compile_full(&g, &target)
            .graph()
            .eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Attention(_))),
        "the over-grid prefill fuses onto the folded grid"
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
    let cpu = eval(&g, &eval_inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_via_contract(&mut gpu, target, &g, &inputs);
    assert_eq!(got.shape(), vec![1, hq, l, d]);
    let max_abs = max_abs_error(got.as_f32().unwrap(), cpu.as_f32().unwrap());
    assert!(
        max_abs <= 2e-3,
        "fused over-grid prefill attention max abs diff {max_abs}"
    );
}

#[test]
fn flash_prefill_gpu_matches_cpu_over_random_shapes() {
    // Prefill attention (one workgroup per (head, query row), online softmax, partials in LDS) on the wgpu
    // executor == CPU eval (== attention_prefill_softcap). Odd seeds carry an attention-logit softcap, even
    // seeds none: one generated kernel, two score transforms. Sweeps random Hkv / n_rep / D / L
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

        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hkv, l, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hkv, l, d]))
            .unwrap();
        let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l])).unwrap();
        let ids = [q.id(), k.id(), v.id(), mask.id()];
        let softcap = (!seed.is_multiple_of(2)).then_some(30.0f32);
        let o = attention_prefill_softcap(&b, q, k, v, n_rep, scale, mask, softcap).unwrap();
        let g = b.finish(o).unwrap();
        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::Attention(_))
        });
        assert!(
            is_attention_prefill(&choice),
            "seed {seed} softcap {softcap:?}: {choice:?}"
        );
        assert_eq!(
            is_softcapped_attention(&choice),
            softcap.is_some(),
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
        let cpu = eval(&g, &eval_inputs, EvalOptions::default())
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
fn synthesized_tiled_gemm_matches_cpu_over_partial_tiles() {
    // A bare M>1 F32 MatMul compiles to the register-tiled GEMM (Card 557: the planner's contraction choice).
    // Confirm the compiled choice is that kernel and it runs GPU==CPU over partial-tile shapes in all three dims
    // (RADV safety and masking).
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
    // partial tiles in all dims: M not a multiple of 16, N not a multiple of 32, K not a multiple of the 16-value
    // stage; plus an exact-tile case and a tall/narrow one.
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
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            x.id(),
            HostTensor::f32(vec![m, k], fill(seed * 7 + 1, m * k)),
        );
        inputs.insert(
            w.id(),
            HostTensor::f32(vec![k, n], fill(seed * 7 + 2, k * n)),
        );
        let out = b.matmul(x, w).unwrap();
        let g = b.finish(out).unwrap();

        let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
            matches!(op, OpKind::MatMul { .. })
        });
        assert!(
            matches!(
                choice,
                KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(_)))
            ),
            "expected the register-tiled GEMM choice, got {choice:?}"
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
    eprintln!("register-tiled GEMM runs GPU==CPU");
}

/// Card 259: real-wgpu dispatch of the per-head attention mask, the layout ALiBi needs. Covers the generated
/// attention kernel in all three forms in one GPU session: decode, prefill and softcapped prefill (the mask
/// plane stride is baked into the generated Body). Each is checked against the CPU oracle, which is checked against a hand-rolled ALiBi
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
        let cpu = eval(g, &eval_inputs, EvalOptions::default())
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
                matches!(op, OpKind::Attention(_))
            });
            assert!(want(&choice), "{label}: unexpected choice {choice:?}");
        };
    let is_softcapped_prefill =
        |choice: &KernelChoice| is_attention_prefill(choice) && is_softcapped_attention(choice);

    // (1) the synthesized region decode, the planner's only decode choice within the LDS cap.
    {
        let (cap, pos) = (6usize, 3usize);
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hkv, cap, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hkv, cap, d]))
            .unwrap();
        let mask = b
            .constant("mask", TensorType::f32(vec![1, hq, 1, cap]))
            .unwrap();
        let y = attention_masked(&b, q, k, v, n_rep, scale, mask).unwrap();
        let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
        let g = b.finish(y).unwrap();
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
        let bb = Builder::for_tests();
        let bq = bb
            .constant("q", TensorType::f32(vec![1, hq, 1, d]))
            .unwrap();
        let bk = bb
            .constant("k", TensorType::f32(vec![1, hkv, cap, d]))
            .unwrap();
        let bv = bb
            .constant("v", TensorType::f32(vec![1, hkv, cap, d]))
            .unwrap();
        let bmask = bb
            .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
            .unwrap();
        let by = attention_masked(&bb, bq, bk, bv, n_rep, scale, bmask).unwrap();
        let (bqi, bki, bvi, bmi) = (bq.id(), bk.id(), bv.id(), bmask.id());
        let gb = bb.finish(by).unwrap();
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

        assert_choice("generated attention decode", &g, &is_attention_decode);
        assert_choice(
            "generated attention decode (bcast)",
            &gb,
            &is_attention_decode,
        );
        check(
            "generated attention decode",
            &g,
            &inputs,
            &gb,
            &bcast_inputs,
            &mut gpu,
        );
    }

    // (2) the prefill and (3) the softcapped prefill, one generated kernel with a different score transform.
    // The large cap keeps the softcap nearly linear over these scores, so the per-head mask difference still
    // shows.
    for (label, softcap) in [
        ("generated attention prefill", None),
        ("generated attention softcapped prefill", Some(50.0f32)),
    ] {
        let l = 6usize;
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, d])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, hkv, l, d]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, hkv, l, d]))
            .unwrap();
        let mask = b.constant("m", TensorType::f32(vec![1, hq, l, l])).unwrap();
        let y = attention_prefill_softcap(&b, q, k, v, n_rep, scale, mask, softcap).unwrap();
        let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
        let g = b.finish(y).unwrap();
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
        let bb = Builder::for_tests();
        let bq = bb
            .constant("q", TensorType::f32(vec![1, hq, l, d]))
            .unwrap();
        let bk = bb
            .constant("k", TensorType::f32(vec![1, hkv, l, d]))
            .unwrap();
        let bv = bb
            .constant("v", TensorType::f32(vec![1, hkv, l, d]))
            .unwrap();
        let bmask = bb.constant("m", TensorType::f32(vec![1, 1, l, l])).unwrap();
        let by = attention_prefill_softcap(&bb, bq, bk, bv, n_rep, scale, bmask, softcap).unwrap();
        let (bqi, bki, bvi, bmi) = (bq.id(), bk.id(), bv.id(), bmask.id());
        let gb = bb.finish(by).unwrap();
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
            &is_softcapped_prefill
        } else {
            &is_attention_prefill
        };
        assert_choice(label, &g, want);
        assert_choice(label, &gb, want);
        check(label, &g, &inputs, &gb, &bcast_inputs, &mut gpu);
    }
}

/// Run `g` on the device and on the CPU oracle with the same operands and return the worst absolute
/// difference; the operands are deterministic fills, the `mask` the caller's.
fn device_vs_oracle(
    gpu: &mut Engine<WgpuDevice>,
    target: poot_graph_plan::Target,
    g: &poot_graph_ir::Graph,
    operands: &[(poot_graph_ir::ValueId, Vec<usize>, u32)],
    mask: (poot_graph_ir::ValueId, Vec<usize>, Vec<f32>),
) -> f32 {
    let mut inputs = HashMap::new();
    for (id, shape, seed) in operands {
        let n = shape.iter().product::<usize>();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + *seed as f32) * 0.37).sin() * 0.5)
            .collect();
        inputs.insert(*id, HostTensor::f32(shape.clone(), data));
    }
    inputs.insert(mask.0, HostTensor::f32(mask.1, mask.2));
    device_vs_oracle_inputs(gpu, target, g, inputs)
}

/// [`device_vs_oracle`] over every input already built: the device's answer against the CPU oracle's.
fn device_vs_oracle_inputs(
    gpu: &mut Engine<WgpuDevice>,
    target: poot_graph_plan::Target,
    g: &poot_graph_ir::Graph,
    inputs: HashMap<poot_graph_ir::ValueId, HostTensor>,
) -> f32 {
    let eval_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(g, &eval_inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_via_contract(gpu, target, g, &inputs);
    assert_eq!(got.shape(), cpu.shape());
    poot_test_util::max_abs_error(got.as_f32().unwrap(), cpu.as_f32().unwrap())
}

/// SC-001 on wgpu, Gemma2: a GQA decode with attention-logit softcapping (scores reach well past the cap) is
/// the generated attention carrying its softcap and matches the CPU oracle within tier 2. Mutation: restore
/// the decode softcap decline in `match_attention`; the chain stays materialized and the choice assertion fails.
#[test]
fn gemma2_softcap_decode_gpu_matches_cpu() {
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, cap, d) = (4usize, 2usize, 40usize, 32usize);
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
    // Scale 4.0 pushes the raw scores far past the cap of 3.0.
    let y =
        poot_graph_ir::ops::attention_masked_softcap(&b, q, k, v, hq / hkv, 4.0, mask, Some(3.0))
            .unwrap();
    let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
    let g = b.finish(y).unwrap();
    let choice = compiled_choice(&g, &target, |op| matches!(op, OpKind::Attention(_)));
    assert!(
        is_attention_decode(&choice) && is_softcapped_attention(&choice),
        "{choice:?}"
    );
    let mut mask_data = vec![0.0; cap];
    mask_data[cap - 7..].fill(-1.0e9);
    let worst = device_vs_oracle(
        &mut gpu,
        target,
        &g,
        &[
            (qi, vec![1, hq, 1, d], 1),
            (ki, vec![1, hkv, cap, d], 2),
            (vi, vec![1, hkv, cap, d], 3),
        ],
        (mi, vec![1, 1, 1, cap], mask_data),
    );
    assert!(
        worst <= 1e-4,
        "device differs from the oracle by {worst:.3e}"
    );
}

/// A sink attention of the shape `rows` queries against `cap` keys (`rows == 1` is a decode): the chain
/// `attention_masked_sinks` traces fuses to the generated attention carrying the per-head sink, and the device
/// matches the CPU oracle within tier 2. Heads are GQA (`Hq = 4`, `Hkv = 2`) so the sink is read per query head.
fn assert_sink_attention_matches_the_oracle(label: &str, rows: usize, cap: usize) {
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, d) = (4usize, 2usize, 32usize);
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, hq, rows, d]))
        .unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, cap, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hkv, cap, d]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, rows, cap]))
        .unwrap();
    let sinks = b.constant("sinks", TensorType::f32(vec![hq])).unwrap();
    let y = poot_graph_ir::ops::attention_masked_sinks(&b, q, k, v, hq / hkv, 1.5, mask, sinks)
        .unwrap();
    let (qi, ki, vi, mi, si) = (q.id(), k.id(), v.id(), mask.id(), sinks.id());
    let g = b.finish(y).unwrap();
    let choice = compiled_choice(&g, &target, |op| matches!(op, OpKind::Attention(_)));
    assert!(
        generated_attention(&choice).is_some_and(|spec| spec.sink == kg::AttentionSink::PerHead),
        "{label}: {choice:?}"
    );
    // Row `i` sees the keys up to `cap - rows + i` (a prefill into a cache; a decode row sees all but its tail).
    let mut mask_data = vec![-1.0e9; rows * cap];
    for i in 0..rows {
        mask_data[i * cap..i * cap + cap - rows + i + 1].fill(0.0);
    }
    let worst = device_vs_oracle(
        &mut gpu,
        target,
        &g,
        &[
            (qi, vec![1, hq, rows, d], 1),
            (ki, vec![1, hkv, cap, d], 2),
            (vi, vec![1, hkv, cap, d], 3),
            (si, vec![hq], 4),
        ],
        (mi, vec![1, 1, rows, cap], mask_data),
    );
    assert!(
        worst <= 1e-4,
        "{label}: device differs from the oracle by {worst:.3e}"
    );
}

/// SC-001 on wgpu, gpt-oss: a GQA decode with a per-head attention sink over more keys than lanes is the
/// generated attention carrying its sink. Mutation: drop the sink from the kernel's initial `(m, l)` state in
/// `attention_body`; the sink's mass vanishes from the normalizer and the device differs from the oracle.
#[test]
fn sink_decode_gpu_matches_cpu() {
    assert_sink_attention_matches_the_oracle("sink decode", 1, 70);
}

/// SC-001 on wgpu, gpt-oss prefill: five query rows into a 40-slot cache, each seeing its own prefix, with the
/// per-head sink. Mutation: as `sink_decode_gpu_matches_cpu`.
#[test]
fn sink_prefill_into_a_cache_gpu_matches_cpu() {
    assert_sink_attention_matches_the_oracle("sink prefill", 5, 40);
}

/// A causal attention of `rows` queries at positions `pos0..pos0 + rows` against `cap` keys under `window`:
/// the mask `causal_mask_from_pos` traces is stated by the positions, the generated attention carries
/// `MaskKind::Causal`, and the device matches the CPU oracle (which evaluates the mask tensor) within tier 2.
/// Keys past the last position and before the window stay hidden by the kernel's own loop bounds.
fn assert_causal_attention_matches_the_oracle(
    label: &str,
    rows: usize,
    cap: usize,
    pos0: i32,
    window: Option<usize>,
) {
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, d) = (4usize, 2usize, 32usize);
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, hq, rows, d]))
        .unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, cap, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hkv, cap, d]))
        .unwrap();
    // A ranged position slot, as the tracers declare it: the I32-to-F32 cast of the mask is proven exact.
    let pos = b
        .slot_ranged(
            poot_graph_ir::Slot::Pos,
            TensorType::new(vec![1, rows], DType::I32),
            poot_graph_ir::IndexRange::below(cap).unwrap(),
        )
        .unwrap();
    let mask = poot_graph_ir::ops::causal_mask_from_pos(&b, pos, cap, window).unwrap();
    let y = poot_graph_ir::ops::attention_masked(&b, q, k, v, hq / hkv, 1.5, mask).unwrap();
    let (qi, ki, vi, pi) = (q.id(), k.id(), v.id(), pos.id());
    let g = b.finish(y).unwrap();
    let choice = compiled_choice(&g, &target, |op| matches!(op, OpKind::Attention(_)));
    assert!(
        generated_attention(&choice)
            .is_some_and(|spec| spec.mask == kg::MaskKind::Causal { window }),
        "{label}: {choice:?}"
    );
    let mut inputs = HashMap::new();
    for (id, shape, seed) in [
        (qi, vec![1, hq, rows, d], 1u32),
        (ki, vec![1, hkv, cap, d], 2),
        (vi, vec![1, hkv, cap, d], 3),
    ] {
        let n = shape.iter().product::<usize>();
        let data = (0..n)
            .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 0.5)
            .collect();
        inputs.insert(id, HostTensor::f32(shape, data));
    }
    inputs.insert(
        pi,
        HostTensor::i32(vec![1, rows], (pos0..pos0 + rows as i32).collect()),
    );
    let worst = device_vs_oracle_inputs(&mut gpu, target, &g, inputs);
    assert!(
        worst <= 1e-4,
        "{label}: device differs from the oracle by {worst:.3e}"
    );
}

/// SC-001 (R2-C7-006) on wgpu: a decode at position 53 of a 70-slot cache, every earlier key visible, then the
/// same under a 16-key window, is the generated attention with a structural causal mask. Mutation: leave the
/// causal row's `key_end` at the capacity in `attention_body`; the window row reads keys the window hides and
/// differs from the oracle.
#[test]
fn causal_decode_gpu_matches_cpu() {
    assert_causal_attention_matches_the_oracle("causal decode", 1, 70, 53, None);
    assert_causal_attention_matches_the_oracle("windowed decode", 1, 70, 53, Some(16));
}

/// SC-001 (R2-C7-006) on wgpu: five query rows at positions 30..35 of a 40-slot cache, each with its own
/// visible prefix and window. Mutation: as `causal_decode_gpu_matches_cpu`.
#[test]
fn causal_prefill_into_a_cache_gpu_matches_cpu() {
    assert_causal_attention_matches_the_oracle("causal prefill", 5, 40, 30, None);
    assert_causal_attention_matches_the_oracle("windowed prefill", 5, 40, 30, Some(8));
}

/// SC-001 on wgpu, DeepSeek-V2 MLA: `dk` 192 (`k_nope` 128 per head plus the one shared 64-wide `k_pe`) and a
/// 128-wide value, a split key, over more keys than lanes. The choice is the generated split-key attention
/// and the device matches the CPU oracle within tier 2. Mutation: drop the split-key recognition in
/// `peel_key`; the chain stays materialized and the choice assertion fails.
#[test]
fn deepseek_v2_mla_split_key_decode_gpu_matches_cpu() {
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, cap, nope, rope, dv) = (2usize, 70usize, 128usize, 64usize, 128usize);
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, hq, 1, nope + rope]))
        .unwrap();
    let k_nope = b
        .constant("k_nope", TensorType::f32(vec![1, hq, cap, nope]))
        .unwrap();
    let k_pe = b
        .constant("k_pe", TensorType::f32(vec![1, 1, cap, rope]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hq, cap, dv]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
        .unwrap();
    let key = b
        .concat(
            3,
            &[k_nope, poot_graph_ir::ops::repeat_kv(&b, k_pe, hq).unwrap()],
        )
        .unwrap();
    let y = poot_graph_ir::ops::attention_masked(&b, q, key, v, 1, 0.07, mask).unwrap();
    let ids = (q.id(), k_nope.id(), k_pe.id(), v.id(), mask.id());
    let g = b.finish(y).unwrap();
    let choice = compiled_choice(&g, &target, |op| matches!(op, OpKind::Attention(_)));
    let spec = generated_attention(&choice).expect("the MLA chain is the generated attention");
    assert_eq!(spec.key, kg::KeyLayout::Split { shared_dim: rope });
    assert_eq!((spec.dk, spec.dv), (nope + rope, dv));
    let mut mask_data = vec![0.0; cap];
    mask_data[cap - 9..].fill(-1.0e9);
    let worst = device_vs_oracle(
        &mut gpu,
        target,
        &g,
        &[
            (ids.0, vec![1, hq, 1, nope + rope], 1),
            (ids.1, vec![1, hq, cap, nope], 2),
            (ids.2, vec![1, 1, cap, rope], 3),
            (ids.3, vec![1, hq, cap, dv], 4),
        ],
        (ids.4, vec![1, 1, 1, cap], mask_data),
    );
    assert!(
        worst <= 1e-4,
        "device differs from the oracle by {worst:.3e}"
    );
}

/// POOT-1313: the gemma-3-1b-it global-attention prefill at 8192 tokens (4 query heads over one key/value
/// head of width 256) completes on the device. As one dispatch it ran about 2 s, the display watchdog reset
/// the context, and the next upload failed ("write_with ... is not word aligned"). The planner now splits it
/// into runs of rows under the budget; this row drives the real compile and executor, requires the split, and
/// checks sampled rows against a host softmax (the full CPU eval is minutes).
///
/// Mutation: set `ATTENTION_CHUNK_WORK` to `u64::MAX`; the plan-split assertion goes red before any launch.
/// (Unsplit, the launch itself loses the device, which can wedge the iGPU, so the launch is not mutated.)
#[test]
fn gemma_global_attention_prefill_at_8192_tokens_runs_split_and_matches_a_host_reference() {
    use poot_graph_ir::builder::Builder;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (hq, hkv, l, d) = (4usize, 1usize, 8192usize, 256usize);
    let scale = 1.0 / (d as f32).sqrt();
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
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d])).unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, l, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hkv, l, d]))
        .unwrap();
    let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l])).unwrap();
    let ids = [q.id(), k.id(), v.id(), mask.id()];
    let o = attention_prefill(&b, q, k, v, hq / hkv, scale, mask).unwrap();
    let g = b.finish(o).unwrap();
    let choice = compiled_choice(&g, &target, |op| matches!(op, OpKind::Attention(_)));
    assert!(
        matches!(
            choice,
            KernelChoice::Generated(KernelRequest::AttentionRows(_))
        ),
        "an 8192-token global prefill plans as runs of rows on a watchdog device: {choice:?}"
    );
    let (qd, kd, vd) = (
        fill(1, hq * l * d),
        fill(2, hkv * l * d),
        fill(3, hkv * l * d),
    );
    let maskd: Vec<f32> = (0..l * l)
        .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(ids[0], HostTensor::f32(vec![1, hq, l, d], qd.clone()));
    inputs.insert(ids[1], HostTensor::f32(vec![1, hkv, l, d], kd.clone()));
    inputs.insert(ids[2], HostTensor::f32(vec![1, hkv, l, d], vd.clone()));
    inputs.insert(ids[3], HostTensor::f32(vec![1, 1, l, l], maskd));
    let got = run_via_contract(&mut gpu, target, &g, &inputs);
    assert_eq!(got.shape(), vec![1, hq, l, d]);
    let got = got.as_f32().unwrap();
    for h in 0..hq {
        for i in [0usize, 4097, l - 1] {
            let qrow = &qd[(h * l + i) * d..][..d];
            let scores: Vec<f64> = (0..=i)
                .map(|j| {
                    let krow = &kd[j * d..][..d];
                    f64::from(scale)
                        * qrow
                            .iter()
                            .zip(krow)
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum::<f64>()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for c in 0..d {
                let want: f64 = weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * f64::from(vd[j * d + c]))
                    .sum::<f64>()
                    / total;
                let have = f64::from(got[(h * l + i) * d + c]);
                assert!(
                    (have - want).abs() < 2e-3,
                    "head {h} row {i} col {c}: gpu {have} vs host {want}"
                );
            }
        }
    }
}
