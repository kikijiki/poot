use super::*;

use poot_graph_plan::{ImportedKernel, KernelChoice};
use poot_kernelgen::{ContractionSpec, KernelRequest};

// ADR-0113: a test checks the production kernel choice instead of calling the deleted test-only predicates
// `is_decode_gemv`/`is_tiled_gemm`.

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
            kernel: ImportedKernel::GemvCoalesced | ImportedKernel::GemvBatchedCoalesced,
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
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![m, k])).unwrap();
    let wt = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let out = b.matmul(a, wt).unwrap();
    let g = b.finish(out).unwrap();
    let eqn = g
        .eqns()
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
        is_generated_tiled_gemm(&choice),
        "expected tiled path, got {plan:?} / {choice:?}"
    );
    let r = |s: usize, num: usize| -> Vec<f32> {
        (0..num)
            .map(|i| ((i + s) as f32 * 0.0007).sin() * 0.1)
            .collect()
    };
    let mut inp = HashMap::new();
    inp.insert(a.id(), HostTensor::f32(vec![m, k], r(1, m * k)));
    inp.insert(wt.id(), HostTensor::f32(vec![k, n], r(7, k * n)));
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
        let b = Builder::for_tests();
        let a = b.constant("a", TensorType::f32(vec![m, k])).unwrap();
        let wt = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
        let out = b.matmul(a, wt).unwrap();
        let g = b.finish(out).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul))
            .unwrap();
        // both take the tiled path; the plan is one dispatch or several.
        let (plan, choice) = plan_eqn_with_choice(
            &g,
            eqn,
            Backend::SpirvVulkan,
            &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
        )
        .unwrap();
        assert!(
            is_generated_tiled_gemm(&choice),
            "m={m} n={n}: should take the tiled path, got {plan:?} / {choice:?}"
        );
        let chunked = plan.dispatches().len() > 1;
        assert_eq!(
            chunked,
            expect_chunked,
            "m={m} n={n} (tiles={}): single-vs-chunked dispatch wrong",
            m.div_ceil(16) * n.div_ceil(8)
        );
        let mut inp = HashMap::new();
        inp.insert(a.id(), HostTensor::f32(vec![m, k], r(1, m * k)));
        inp.insert(wt.id(), HostTensor::f32(vec![k, n], r(7, k * n)));
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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![n])).unwrap();
    let y = b.binary(B2::Mul, x, x).unwrap(); // x*x, one thread per element
    let g = b.finish(y).unwrap();
    let xd: Vec<f32> = (0..n).map(|i| ((i % 1000) as f32) * 0.001).collect();
    let mut inp = HashMap::new();
    inp.insert(x.id(), HostTensor::f32(vec![n], xd.clone()));
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
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![1, k])).unwrap();
    let wt = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let out = b.matmul(a, wt).unwrap();
    let g = b.finish(out).unwrap();
    // sanity: this is the decode-GEMV path (M=1, rank-2 weight, large N).
    let eqn = g
        .eqns()
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
        is_decode_gemv_kernel(&choice),
        "large-N decode matmul should take the 2-D GEMV path, got {plan:?} / {choice:?}"
    );
    let r = |s: usize, num: usize| -> Vec<f32> {
        (0..num)
            .map(|i| ((i + s) as f32 * 0.001).sin() * 0.1)
            .collect()
    };
    let mut inp = HashMap::new();
    inp.insert(a.id(), HostTensor::f32(vec![1, k], r(1, k)));
    inp.insert(wt.id(), HostTensor::f32(vec![k, n], r(7, k * n)));
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
    // small K (~32MB weight) to check the split-dispatch plan (`decode_gemv_plan`/`kernelgen::gemv_lds`'s
    // `elem_offset`) bit-for-bit against the CPU oracle. N, not K, is the watchdog hazard axis, so it is safe to run
    // unconditionally (unlike the `#[ignore]`d real-BLOOM-scale diagnostics in `cross_backend.rs`).
    use poot_graph_ir::op::OpKind;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (k, n) = (32usize, 250_880usize); // BLOOM's real tied lm_head vocab, a small K
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![1, k])).unwrap();
    let wt = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let out = b.matmul(a, wt).unwrap();
    let g = b.finish(out).unwrap();
    // sanity: this is the decode-GEMV path, over the chunk trigger - plan_eqn must actually take the new
    // split-dispatch branch, not silently stay a single dispatch.
    let eqn = g
        .eqns()
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
        is_decode_gemv_kernel(&choice),
        "large-N decode matmul should take the decode-GEMV path, got {plan:?} / {choice:?}"
    );
    assert!(
        plan.dispatches().len() > 1,
        "BLOOM-scale N should split into >1 watchdog-safety chunk, got {plan:?}"
    );
    let r = |s: usize, num: usize| -> Vec<f32> {
        (0..num)
            .map(|i| ((i + s) as f32 * 0.001).sin() * 0.1)
            .collect()
    };
    let mut inp = HashMap::new();
    inp.insert(a.id(), HostTensor::f32(vec![1, k], r(1, k)));
    inp.insert(wt.id(), HostTensor::f32(vec![k, n], r(7, k * n)));
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
    let b = Builder::for_tests();
    let p0 = b.constant("p0", TensorType::f32(vec![2, 2, 3])).unwrap();
    let p1 = b.constant("p1", TensorType::f32(vec![2, 1, 3])).unwrap();
    let p2 = b.constant("p2", TensorType::f32(vec![2, 3, 3])).unwrap();
    let out = b.concat(1, &[p0, p1, p2]).unwrap();
    let (i0, i1, i2) = (p0.id(), p1.id(), p2.id());
    let g = b.finish(out).unwrap();
    let eqn = g
        .eqns()
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
        plan.single()
            .is_some_and(|dispatch| dispatch.launch().meta.is_empty()),
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
    let b = Builder::for_tests();
    let n = 5usize;
    let ps: Vec<_> = (0..n)
        .map(|k| {
            b.constant(&format!("p{k}"), TensorType::f32(vec![1, 4, 32, 8]))
                .unwrap()
        })
        .collect();
    let out = b.concat(2, &ps).unwrap();
    let ids: Vec<_> = ps.iter().map(|p| p.id()).collect();
    let g = b.finish(out).unwrap();
    let eqn = g
        .eqns()
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
        plan.single()
            .is_some_and(|dispatch| dispatch.launch().meta.is_empty()),
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

/// An `IndexedMatMul` fixture: `x [m, k] x W [e, k, n]` with row `r` reading expert `ids[r]`.
struct IndexedCase {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    e: usize,
    ids: Vec<i32>,
}

/// Integer-valued fill in `[-3, 3]`: with `K * 3 * 3` far below `2^24` every partial sum is exact, so every
/// summation order and every fused multiply-add gives the same bits and a wrong expert changes the value.
fn int_fill(seed: u64, count: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 33) % 7) as f32 - 3.0
        })
        .collect()
}

const INDEXED_EXPERTS: usize = 4;
/// `K` is not a multiple of the 8-row tile and `N` not a multiple of its column edge, so the tail masks run.
const INDEXED_K: usize = 36;
const INDEXED_N: usize = 21;

/// Sorted expert ids over `rows` rows, runs of unequal length so tile boundaries (every 8 rows) cut across
/// expert boundaries.
fn sorted_ids(rows: usize) -> Vec<i32> {
    let bounds = [37, 78, 116];
    (0..rows)
        .map(|r| bounds.iter().filter(|&&b| r >= b).count() as i32)
        .collect()
}

/// The same multiset in a deterministic shuffled order (stride 7 is coprime to 160).
fn shuffled_ids(rows: usize) -> Vec<i32> {
    let sorted = sorted_ids(rows);
    (0..rows).map(|r| sorted[(r * 7) % rows]).collect()
}

/// One expert per 8-row tile: the fixture on which a walk that stops after its first step stays green.
fn tile_aligned_ids(rows: usize) -> Vec<i32> {
    (0..rows)
        .map(|r| ((r / 8) % INDEXED_EXPERTS) as i32)
        .collect()
}

fn indexed_cases() -> Vec<IndexedCase> {
    let case = |name, m, ids: Vec<i32>| IndexedCase {
        name,
        m,
        k: INDEXED_K,
        n: INDEXED_N,
        e: INDEXED_EXPERTS,
        ids,
    };
    vec![
        case("decode, unsorted ids with a repeat", 5, vec![2, 0, 3, 2, 1]),
        case("prefill, sorted ids", 160, sorted_ids(160)),
        case("prefill, shuffled ids", 160, shuffled_ids(160)),
        case("prefill, one expert per tile", 160, tile_aligned_ids(160)),
    ]
}

struct IndexedProgram {
    graph: poot_graph_ir::Graph,
    inputs: HashMap<poot_graph_ir::ValueId, HostTensor>,
}

/// `IndexedMatMul(x, w, ids)`, and with `gate` that product times `gate[m]` (the routed gate weighting), every
/// operand a named const.
fn indexed_program(case: &IndexedCase, gate: Option<Vec<f32>>) -> IndexedProgram {
    let IndexedCase {
        m, k, n, e, ids, ..
    } = case;
    let (m, k, n, e) = (*m, *k, *n, *e);
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![e, k, n])).unwrap();
    let idx = b
        .constant_ranged(
            "idx",
            TensorType::new(vec![m], poot_tensor::DType::I32),
            poot_graph_ir::IndexRange::below(e).unwrap(),
        )
        .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(x.id(), HostTensor::f32(vec![m, k], int_fill(1, m * k)));
    inputs.insert(
        w.id(),
        HostTensor::f32(vec![e, k, n], int_fill(2, e * k * n)),
    );
    inputs.insert(idx.id(), HostTensor::i32(vec![m], ids.clone()));
    let mut out = b.indexed_matmul(x, w, idx).unwrap();
    if let Some(gate) = gate {
        let gate_t = b.constant("gate", TensorType::f32(vec![m])).unwrap();
        inputs.insert(gate_t.id(), HostTensor::f32(vec![m], gate));
        let column = b.reshape(gate_t, vec![m, 1]).unwrap();
        out = b
            .binary(poot_graph_ir::op::BinOp::Mul, out, column)
            .unwrap();
    }
    IndexedProgram {
        graph: b.finish(out).unwrap(),
        inputs,
    }
}

/// The CPU oracle over `program`.
fn indexed_oracle(program: &IndexedProgram) -> Result<HostTensor, poot_eval::EvalError> {
    let eval_inputs: HashMap<_, Value> = program
        .inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    eval(
        &program.graph,
        &eval_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|evaluation| evaluation.output.into_host().unwrap())
}

fn bits(t: &HostTensor) -> Vec<u32> {
    t.as_f32().unwrap().iter().map(|v| v.to_bits()).collect()
}

/// SC-001, SC-002: `IndexedMatMul` through `compile` equals the oracle `to_bits` on every element at a decode
/// shape (`M = 5`, the Gemv) and at prefill shapes (`M = 160`, the segmented Tiled) whose tiles cross expert
/// boundaries, with the ids sorted, shuffled and tile-aligned. A wrong expert changes a value by an integer, so
/// the comparison is exact.
///
/// Mutation (`emit_row_slab`): read `ids[(m + 1) % M]`; every case goes red. Mutation (the segmented walk): stop
/// after its first step; the two crossing cases go red and the tile-aligned case stays green (its tiles hold
/// one expert each).
#[test]
fn indexed_matmul_matches_the_oracle_bit_for_bit() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let mut failures = Vec::new();
    for case in indexed_cases() {
        let program = indexed_program(&case, None);
        let want = indexed_oracle(&program).unwrap();
        let got = run_via_contract(&mut gpu, target, &program.graph, &program.inputs);
        assert_eq!(got.shape(), vec![case.m, case.n], "{}", case.name);
        let (got_bits, want_bits) = (bits(&got), bits(&want));
        let wrong: Vec<usize> = (0..got_bits.len())
            .filter(|&i| got_bits[i] != want_bits[i])
            .collect();
        if let Some(&first) = wrong.first() {
            failures.push(format!(
                "{}: {} of {} elements differ; row {} col {} is {} on the device, {} in the oracle",
                case.name,
                wrong.len(),
                got_bits.len(),
                first / case.n,
                first % case.n,
                f32::from_bits(got_bits[first]),
                f32::from_bits(want_bits[first]),
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// POOT-646 SC-001: experts held in checkpoint `[E, N, K]` order and read through `transpose(w, [0, 2, 1])` equal
/// the oracle `to_bits` on every element (`E = 8`): `M = 1` and `M = 5` (the Gemv, ids unsorted with repeats) and a
/// prefill shape (the segmented Tiled, shuffled ids). `compile` folds the transpose into the contraction.
///
/// Mutation (`weight_layout` in `planner/moe.rs`): return `WeightLayout::Kn` for the view; the weight is read as
/// `[E, K, N]` and every case goes red.
#[test]
fn indexed_matmul_checkpoint_order_experts_match_the_oracle_bit_for_bit() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (e, k, n) = (8usize, INDEXED_K, INDEXED_N);
    let prefill: Vec<i32> = (0..160)
        .map(|r| [7, 2, 2, 5, 0, 7, 3][(r * 5) % 7])
        .collect();
    let cases = [
        ("M=1", vec![5]),
        ("M=5, unsorted with repeats", vec![6, 0, 6, 3, 1]),
        ("M=160, shuffled", prefill),
    ];
    let mut failures = Vec::new();
    for (name, ids) in cases {
        let m = ids.len();
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![e, n, k])).unwrap();
        let idx = b
            .constant_ranged(
                "idx",
                TensorType::new(vec![m], poot_tensor::DType::I32),
                poot_graph_ir::IndexRange::below(e).unwrap(),
            )
            .unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(x.id(), HostTensor::f32(vec![m, k], int_fill(1, m * k)));
        inputs.insert(
            w.id(),
            HostTensor::f32(vec![e, n, k], int_fill(2, e * n * k)),
        );
        inputs.insert(idx.id(), HostTensor::i32(vec![m], ids));
        let out = b
            .indexed_matmul(x, b.transpose(w, vec![0, 2, 1]).unwrap(), idx)
            .unwrap();
        let program = IndexedProgram {
            graph: b.finish(out).unwrap(),
            inputs,
        };
        let want = indexed_oracle(&program).unwrap();
        let got = run_via_contract(&mut gpu, target, &program.graph, &program.inputs);
        let (got_bits, want_bits) = (bits(&got), bits(&want));
        if let Some(first) = (0..got_bits.len()).find(|&i| got_bits[i] != want_bits[i]) {
            failures.push(format!(
                "{name}: row {} col {} is {} on the device, {} in the oracle",
                first / n,
                first % n,
                f32::from_bits(got_bits[first]),
                f32::from_bits(want_bits[first]),
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// SC-005: a routed gate weighting runs as one `ContractionEpilogue` over the indexed contraction and its output
/// equals the oracle's bit for bit, at a decode and a prefill shape.
///
/// Mutation (the epilogue's store index): index the epilogue by column instead of the flat output index; the
/// per-row gate is read at the wrong row and the fused output goes red against the oracle.
#[test]
fn routed_gate_weighting_fused_equals_the_oracle_bit_for_bit() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    for case in indexed_cases().into_iter().take(3) {
        // Dyadic gates, so the product with an exact integer sum is exact in both programs.
        let gate: Vec<f32> = (0..case.m)
            .map(|r| [0.5, -2.0, 1.25, 3.0, -0.75][r % 5])
            .collect();
        let program = indexed_program(&case, Some(gate));
        let (exe, entry) = add_resident_entry(&mut gpu, target, &program.graph, &program.inputs);
        let fused = step_resident(&mut gpu, exe, entry, &program.graph);
        let want = indexed_oracle(&program).unwrap();
        assert_eq!(bits(&fused), bits(&want), "{}: fused vs oracle", case.name);
    }
}

/// POOT-646: the routed gate weighting fused over checkpoint-order experts (`ContractionEpilogue { head: Indexed }`
/// reading the `[E, N, K]` weight in place) equals the oracle `to_bits` (`E = 8`, a decode and a prefill shape).
///
/// Mutation: drop the `ContractionEpilogue` arm of `is_strided_capable`; the transpose runs as a copy, the plan
/// row goes red, and this one stays green (the value is the same, only the copy returns).
#[test]
fn routed_gate_weighting_checkpoint_order_equals_the_oracle_bit_for_bit() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (e, k, n) = (8usize, INDEXED_K, INDEXED_N);
    for ids in [vec![6, 0, 6, 3, 1], shuffled_ids(160)] {
        let m = ids.len();
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![e, n, k])).unwrap();
        let idx = b
            .constant_ranged(
                "idx",
                TensorType::new(vec![m], poot_tensor::DType::I32),
                poot_graph_ir::IndexRange::below(e).unwrap(),
            )
            .unwrap();
        let gate_t = b.constant("gate", TensorType::f32(vec![m])).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(x.id(), HostTensor::f32(vec![m, k], int_fill(1, m * k)));
        inputs.insert(
            w.id(),
            HostTensor::f32(vec![e, n, k], int_fill(2, e * n * k)),
        );
        inputs.insert(idx.id(), HostTensor::i32(vec![m], ids));
        // Dyadic gates, so the product with an exact integer sum is exact.
        let gate: Vec<f32> = (0..m)
            .map(|r| [0.5, -2.0, 1.25, 3.0, -0.75][r % 5])
            .collect();
        inputs.insert(gate_t.id(), HostTensor::f32(vec![m], gate));
        let product = b
            .indexed_matmul(x, b.transpose(w, vec![0, 2, 1]).unwrap(), idx)
            .unwrap();
        let out = b
            .binary(
                poot_graph_ir::op::BinOp::Mul,
                product,
                b.reshape(gate_t, vec![m, 1]).unwrap(),
            )
            .unwrap();
        let program = IndexedProgram {
            graph: b.finish(out).unwrap(),
            inputs,
        };
        let want = indexed_oracle(&program).unwrap();
        let got = run_via_contract(&mut gpu, target, &program.graph, &program.inputs);
        assert_eq!(bits(&got), bits(&want), "M={m}");
    }
}

/// POOT-1028: `IndexedMatMul` over a BF16-resident expert stack (a BF16 const, `[E, K, N]` or the checkpoint's
/// `[E, N, K]` behind `transpose(w, [0, 2, 1])`; f32 activation and output) equals the oracle `to_bits` on every
/// element, with and without the routed gate epilogue, at a decode shape (the Gemv) and a prefill shape whose tiles
/// cross expert boundaries (the segmented Tiled). The fill is integer-valued, so each element is exactly a BF16 and
/// the widened sums are exact in any order. The packed lanes are widened in-register: no f32 copy of the stack
/// exists.
///
/// Mutation (`WeightLane::decode_low_half`, `poot-kernelgen/src/helpers.rs`): shift the BF16 bits by 15 instead of
/// 16 when widening; every case goes red.
#[test]
fn indexed_matmul_bf16_experts_match_the_oracle_bit_for_bit() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (e, k, n) = (8usize, INDEXED_K, INDEXED_N);
    let decode = vec![6, 0, 6, 3, 1];
    let prefill: Vec<i32> = (0..160)
        .map(|r| [7, 2, 2, 5, 0, 7, 3][(r * 5) % 7])
        .collect();
    let mut failures = Vec::new();
    for (name, ids) in [("M=5", decode), ("M=160, shuffled", prefill)] {
        for checkpoint_order in [false, true] {
            for gated in [false, true] {
                let m = ids.len();
                let b = Builder::for_tests();
                let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
                let idx = b
                    .constant_ranged(
                        "idx",
                        TensorType::new(vec![m], poot_tensor::DType::I32),
                        poot_graph_ir::IndexRange::below(e).unwrap(),
                    )
                    .unwrap();
                let stored = if checkpoint_order {
                    vec![e, n, k]
                } else {
                    vec![e, k, n]
                };
                let w = b
                    .constant(
                        "w",
                        TensorType::new(stored.clone(), poot_tensor::DType::BF16),
                    )
                    .unwrap();
                let mut inputs = HashMap::new();
                inputs.insert(x.id(), HostTensor::f32(vec![m, k], int_fill(1, m * k)));
                inputs.insert(idx.id(), HostTensor::i32(vec![m], ids.clone()));
                let words = int_fill(2, e * k * n)
                    .iter()
                    .map(|v| (v.to_bits() >> 16) as u16)
                    .collect();
                inputs.insert(w.id(), HostTensor::bf16(stored, words));
                let weight = if checkpoint_order {
                    b.transpose(w, vec![0, 2, 1]).unwrap()
                } else {
                    w
                };
                let mut out = b.indexed_matmul(x, weight, idx).unwrap();
                if gated {
                    let gate_t = b.constant("gate", TensorType::f32(vec![m])).unwrap();
                    // Dyadic gates, so the product with an exact integer sum is exact.
                    let gate: Vec<f32> = (0..m)
                        .map(|r| [0.5, -2.0, 1.25, 3.0, -0.75][r % 5])
                        .collect();
                    inputs.insert(gate_t.id(), HostTensor::f32(vec![m], gate));
                    out = b
                        .binary(
                            poot_graph_ir::op::BinOp::Mul,
                            out,
                            b.reshape(gate_t, vec![m, 1]).unwrap(),
                        )
                        .unwrap();
                }
                let program = IndexedProgram {
                    graph: b.finish(out).unwrap(),
                    inputs,
                };
                let want = indexed_oracle(&program).unwrap();
                let got = run_via_contract(&mut gpu, target, &program.graph, &program.inputs);
                let (got_bits, want_bits) = (bits(&got), bits(&want));
                if let Some(first) = (0..got_bits.len()).find(|&i| got_bits[i] != want_bits[i]) {
                    failures.push(format!(
                        "{name} nk={checkpoint_order} gate={gated}: row {} col {} is {} on the device, {} in the oracle",
                        first / n,
                        first % n,
                        f32::from_bits(got_bits[first]),
                        f32::from_bits(want_bits[first]),
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let out = b.matmul(x, w).unwrap();
    let (xi, wi) = (x.id(), w.id());
    let g = b.finish(out).unwrap();

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
    // `gpu.run` == CPU eval. Half the seeds trace a bias linear, whose add `compile` folds into the matmul as its
    // epilogue; every matmul lowers to the generated tiled GEMM (Card 557). Small well-conditioned
    // values keep the f32 dot near-exact.
    use poot_graph_ir::op::{ContractionHead, OpKind};
    use poot_graph_plan::{CompileOptions, compile, production_plan::Submission};
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
        let out = if with_bias {
            let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
            inputs.insert(bias.id(), HostTensor::f32(vec![n], fill(seed * 7 + 3, n)));
            poot_graph_ir::ops::linear(&b, x, w, Some(bias)).unwrap()
        } else {
            b.matmul(x, w).unwrap()
        };
        let g = b.finish(out).unwrap();
        // require the tiled path (else we'd be fuzzing the naive kernel, not the tiled one under test), on the
        // graph `compile` actually plans.
        let program = compile(
            &g,
            &spirv_fixture_target(),
            &CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD),
        )
        .unwrap();
        let (eqn, _) = program
            .planned()
            .find(|(e, _)| {
                matches!(
                    e.op,
                    OpKind::MatMul
                        | OpKind::ContractionEpilogue {
                            head: ContractionHead::MatMul,
                            ..
                        }
                )
            })
            .unwrap();
        let choice = program.kernel_choice(eqn);
        // The generated tiled GEMM carries the epilogue, so a traced bias add folds into it as one
        // `ContractionEpilogue` and no standalone add remains (POOT-1036).
        assert_eq!(
            matches!(eqn.op, OpKind::ContractionEpilogue { .. }),
            with_bias,
            "seed {seed}: a tiled matmul fuses its bias add exactly when it has one"
        );
        assert!(
            matches!(
                choice,
                KernelChoice::Generated(KernelRequest::Contraction(
                    ContractionSpec::TiledRegion { .. }
                ))
            ),
            "seed {seed} (M={m},K={k},N={n}): expected the generated tiled path, got {choice:?}"
        );
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
    // -> 2 chunks. Assert the plan is split and `gpu.run == CPU eval` (the chunks reconstruct the full result).
    use poot_graph_ir::op::OpKind;
    use poot_target::Backend;
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (m, k, n) = (544usize, 32usize, 8192usize);
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let out = b.matmul(x, w).unwrap();
    let g = b.finish(out).unwrap();
    let eqn = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .unwrap();
    let (plan, _) = plan_eqn_with_choice(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
    )
    .unwrap();
    let n_chunks = plan.dispatches().len();
    assert_eq!(n_chunks, 2, "34816 tiles / 32767-per-chunk = 2 chunks");

    let xd: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
    let wd: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.08 - 0.16).collect();
    let mut inputs = HashMap::new();
    inputs.insert(x.id(), HostTensor::f32(vec![m, k], xd));
    inputs.insert(w.id(), HostTensor::f32(vec![k, n], wd));
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
    // Card 546b: the deleted host-roundtrip `run()` and device-resident `run_resident_kv` were two
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
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(xshape.clone())).unwrap();
        let w = b.constant("w", TensorType::f32(wshape.clone())).unwrap();
        let out = b.matmul(x, w).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            x.id(),
            HostTensor::f32(xshape.clone(), fill(seed * 7 + 1, xshape.iter().product())),
        );
        inputs.insert(
            w.id(),
            HostTensor::f32(wshape.clone(), fill(seed * 7 + 2, wshape.iter().product())),
        );
        let g = b.finish(out).unwrap();
        let eqn = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul))
            .unwrap();
        // the batched-weight tiled GEMM is the imported kernel, dispatched with a metadata buffer (the `[E,M,K,N]` dims buffer) rather than a kernelgen launch with none.
        let (plan, choice) = plan_eqn_with_choice(
            &g,
            eqn,
            Backend::SpirvVulkan,
            &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
        )
        .unwrap();
        match single_meta(&plan) {
            Some(meta @ [_, ..]) => {
                assert_eq!(meta.len(), 4, "seed {seed}: batched dims [E,M,K,N]")
            }
            _ => panic!("seed {seed}: batched tiled GEMM should be a metadata launch"),
        }
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
fn tiled_gemm_then_bias_add_gpu_matches_cpu() {
    // The q/k/v projections at M>1: the generated tiled GEMM with the traced bias add folded in as its epilogue
    // (POOT-1036). Tile-unaligned (M=20, K=16, N=33): K is
    // exactly one tile, M and N straddle.
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (m, k, n) = (20usize, 16usize, 33usize);
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
    let out = poot_graph_ir::ops::linear(&b, x, w, Some(bias)).unwrap();
    let (xi, wi, bi) = (x.id(), w.id(), bias.id());
    let g = b.finish(out).unwrap();
    let choice = compiled_choice(&g, &spirv_fixture_target(), |op| {
        matches!(
            op,
            OpKind::ContractionEpilogue {
                head: poot_graph_ir::op::ContractionHead::MatMul,
                ..
            }
        )
    });
    assert!(
        is_generated_tiled_gemm(&choice),
        "the bias linear's matmul should take the generated tiled GEMM: {choice:?}"
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
            "tiled_gemm then bias add elem {i} (row {}, col {}): gpu {a} vs cpu {b}",
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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![e, m, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![e, k, n])).unwrap();
    let out = b.matmul(x, w).unwrap();
    let (xi, wi) = (x.id(), w.id());
    let g = b.finish(out).unwrap();

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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![bb, h, m, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![bb, h, k, n])).unwrap();
    let out = b.matmul(x, w).unwrap();
    let (xi, wi) = (x.id(), w.id());
    let g = b.finish(out).unwrap();

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
    // single dispatch) on NVPTX as on wgpu. The chunked case (a large GEMM that would trip the RADV 2^15 cap on
    // wgpu) stays a single dispatch on NVPTX: NVPTX has no cap, and the PTX executor `unreachable!`s on split plans.
    // Pure planner test (no GPU).
    use poot_graph_ir::op::OpKind;
    use poot_target::Backend;
    let plan_for = |m: usize, n: usize, backend| {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![m, 64])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![64, n])).unwrap();
        let out = b.matmul(x, w).unwrap();
        let g = b.finish(out).unwrap();
        let eqn = g
            .eqns()
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
        matches!(plan_for(8, 8, Backend::Nvptx), (plan_, choice) if single_meta(&plan_).is_some_and(|meta| meta.is_empty()) && is_tiled_region(&choice))
    );
    assert!(
        matches!(plan_for(8, 8, Backend::SpirvVulkan), (plan_, choice) if single_meta(&plan_).is_some_and(|meta| meta.is_empty()) && is_tiled_region(&choice))
    );
    // a GEMM large enough to chunk on wgpu: split on SpirvVulkan, but a single dispatch on NVPTX.
    let big_n = 8 * 40_000; // groups = ceil(8/16)*ceil(N/8) = 1 * 40000 > 2^15 (RADV chunk threshold)
    assert!(matches!(
        plan_for(8, big_n, Backend::SpirvVulkan),
        (plan, choice) if plan.dispatches().len() > 1 && is_tiled_region(&choice)
    ));
    assert!(
        matches!(plan_for(8, big_n, Backend::Nvptx), (plan_, choice) if single_meta(&plan_).is_some_and(|meta| meta.is_empty()) && is_tiled_region(&choice))
    );
}
