//! Card 727 on wgpu: the generated dense Gemv (SC-002) and a contraction keeping its LDS width through the
//! planner (SC-005), each against an f64 host reference over every output element (ADR-0101 tier 2).

use poot_executor_parity::{Fixture, run_outputs};
use poot_graph_ir::{Builder, OpKind, Slot, TensorType};
use poot_graph_plan::{
    CompileLimits, CompileOptions, FusionPolicy, KernelChoice, Submission, Target, compile,
};
use poot_kernelgen::{ContractionSpec, KernelRequest, Schedule};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::{DType, HostTensor};
use poot_test_util::StepFixture;
use std::sync::Arc;

use super::{gpu_lock, open_engine_or_skip};

/// A seeded xorshift stream.
fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// `count` weight values of `dtype` as stored bytes and as the f32 values they hold: every value nonzero, with
/// magnitude in `[2^-7, 1)`, generated as bits so no rounding step decides them.
fn weight(dtype: DType, count: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut next = stream(seed);
    let mut bytes = Vec::new();
    let mut held = Vec::with_capacity(count);
    for _ in 0..count {
        let r = (next() >> 32) as u32;
        match dtype {
            DType::F16 => {
                let bits = ((r & 0x8000) | ((8 + (r >> 10) % 7) << 10) | (r & 0x03ff)) as u16;
                bytes.extend(bits.to_le_bytes());
                held.push(poot_quant::scalar::f16_to_f32(bits));
            }
            DType::BF16 | DType::F32 => {
                let bits = ((r & 0x8000) | ((120 + (r >> 7) % 7) << 7) | (r & 0x7f)) as u16;
                let value = f32::from_bits(u32::from(bits) << 16);
                if dtype == DType::BF16 {
                    bytes.extend(bits.to_le_bytes());
                } else {
                    bytes.extend(value.to_le_bytes());
                }
                held.push(value);
            }
            other => panic!("no dense weight fixture for {other:?}"),
        }
    }
    (bytes, held)
}

/// `x [1, k] @ w [n, k]^T` with `w` a `dtype` checkpoint-orientation constant, which `compile` folds into one
/// `DenseContraction` at M == 1. The activation's first and last elements are 64: each prescribes a
/// contribution far above the tolerance in the first K run and in the K tail every lane finishes element by
/// element, so a body that drops or misreads either fails the comparison.
struct GemvCase {
    fixture: Fixture,
    activation: Vec<f32>,
    held: Vec<f32>,
    k: usize,
    n: usize,
}

fn gemv_case(dtype: DType, k: usize, n: usize) -> GemvCase {
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![1, k]));
    let w = b.constant("w", TensorType::new(vec![n, k], dtype));
    let out = b.matmul(x, b.transpose(w, vec![1, 0]));
    let graph = b.finish(out);
    let key = graph.meta(x.id).slot_key().unwrap().clone();
    let (bytes, held) = weight(dtype, n * k, (k * 131 + n) as u64);
    let mut builder = WeightStore::builder();
    let dense = DenseWeight::try_new(dtype, vec![n, k], Arc::from(bytes)).unwrap();
    builder
        .insert("w".to_string(), WeightEntry::Dense(dense))
        .unwrap();
    let mut next = stream(k as u64 + 7);
    let mut activation: Vec<f32> = (0..k)
        .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
        .collect();
    activation[0] = 64.0;
    activation[k - 1] = 64.0;
    let steps = vec![vec![StepFixture {
        key,
        tensor: HostTensor::f32(vec![1, k], activation.clone()),
    }]];
    GemvCase {
        fixture: Fixture {
            name: "dense_gemv",
            graph,
            store: builder.build(),
            steps,
            fusion: FusionPolicy::Full,
        },
        activation,
        held,
        k,
        n,
    }
}

const OPTIONS: CompileOptions = CompileOptions {
    execution: Submission::Replay,
    fusion: FusionPolicy::Full,
    limits: CompileLimits::STANDARD,
};

/// Output widths `N` that reach every launch the planner admits on a device with `compute_units` compute units
/// (at least two), none a multiple of its `cols`: `cols` 4 while `ceil(N / 4)` still launches `2 x compute_units`
/// workgroups (a wide `N` and one just past that bound), else 2 (one just short of it).
fn launch_shapes(compute_units: u32) -> [(usize, u32); 3] {
    let cu = compute_units as usize;
    [(128 * cu + 5, 4), (8 * cu + 5, 4), (8 * cu - 7, 2)]
}

/// The Gemv launch `compile` plans for `case` on `target`.
fn planned_schedule(case: &GemvCase, target: Target) -> Schedule {
    let program = compile(&case.fixture.graph, &target, &OPTIONS).expect("the fixture compiles");
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::DenseContraction { .. }))
        .expect("compile folds the projection into one DenseContraction");
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseGemv {
            schedule,
            ..
        })) => *schedule,
        other => panic!("expected the generated dense Gemv, got {other:?}"),
    }
}

/// Every output of `got` against the f64 dot product of `activation` and the held weight rows, within the
/// stated tier-2 tolerance of an f32 reassociated contraction (`1e-4` of the sum of the terms' magnitudes).
/// A non-finite output fails at any tolerance.
fn assert_matches_reference(got: &[f32], case: &GemvCase, what: &str) {
    assert_eq!(got.len(), case.n, "{what}: output length");
    for (col, &value) in got.iter().enumerate() {
        let terms = (0..case.k)
            .map(|i| f64::from(case.activation[i]) * f64::from(case.held[col * case.k + i]));
        let want: f64 = terms.clone().sum();
        let scale: f64 = terms.map(f64::abs).sum();
        assert!(
            value.is_finite() && (f64::from(value) - want).abs() <= 1e-4 * scale,
            "{what}: out[{col}] = {value}, reference {want} (scale {scale})"
        );
    }
}

/// Card 727 SC-002 on wgpu: the generated dense Gemv over an F32, packed-F16 and packed-BF16 `[N, K]` weight,
/// at every launch the planner admits (`cols` 4 and 2, reached by the output width), meets
/// ADR-0101 tier 2 on every output. `K = 1037` leaves a K tail past the last full run at every launch, `K = 515`
/// is odd so packed rows start mid-word, and no `N` is a multiple of its `cols`.
///
/// Mutation: in `contraction::gemv` start every lane's first run one trip late (skip the first K tile), or skip
/// the tail runs; the prescribed contribution is lost and the row goes red.
#[test]
fn the_dense_gemv_meets_tier_2_in_every_lane_and_launch_on_wgpu() {
    let _g = gpu_lock();
    let Some((engine, device_target)) = open_engine_or_skip() else {
        return;
    };
    let mut exec: Box<dyn poot_executor::Executor> = Box::new(engine);
    let target = device_target;
    for (n, cols) in launch_shapes(target.caps.compute_units) {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for k in [1037, 515] {
                let case = gemv_case(dtype, k, n);
                let schedule = planned_schedule(&case, target);
                assert!(
                    matches!(schedule, Schedule::Gemv { cols: c, .. } if c == cols),
                    "N = {n} on {} compute units: {schedule:?}",
                    target.caps.compute_units
                );
                let what = format!("{dtype:?} k={k} n={n} {schedule:?}");
                let outputs = run_outputs(exec.as_mut(), target, &case.fixture)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_matches_reference(outputs[0].as_f32().expect("f32 output"), &case, &what);
            }
        }
    }
}

/// Decode attention `scores @ V` with the scores as a slot: `[1, hq, 1, cap] @ [1, hq, cap, d]`, the shape the
/// planner routes to the generated LDS-reduction attention GEMV.
fn attention_gemv_case(hq: usize, cap: usize, d: usize) -> (Fixture, Vec<f32>, Vec<f32>) {
    let b = Builder::new();
    let scores = b.slot_named(
        Slot::Activation,
        "scores",
        TensorType::f32(vec![1, hq, 1, cap]),
    );
    let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
    let out = b.matmul(scores, v);
    let graph = b.finish(out);
    let key = graph.meta(scores.id).slot_key().unwrap().clone();
    let (bytes, values) = weight(DType::F32, hq * cap * d, 0x5C0);
    let mut builder = WeightStore::builder();
    let dense = DenseWeight::try_new(DType::F32, vec![1, hq, cap, d], Arc::from(bytes)).unwrap();
    builder
        .insert("v".to_string(), WeightEntry::Dense(dense))
        .unwrap();
    let mut next = stream(0xA7);
    let scores: Vec<f32> = (0..hq * cap)
        .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
        .collect();
    let steps = vec![vec![StepFixture {
        key,
        tensor: HostTensor::f32(vec![1, hq, 1, cap], scores.clone()),
    }]];
    let fixture = Fixture {
        name: "attention_gemv",
        graph,
        store: builder.build(),
        steps,
        fusion: FusionPolicy::Full,
    };
    (fixture, scores, values)
}

/// Card 727 SC-005 on wgpu: a contraction whose body sizes its LDS reduction by its baked width
/// (the decode attention GEMV) keeps that width through the planner at an output past `128 x` the X grid cap,
/// where the generic bump would raise a one-thread-per-output body to 256 lanes, and its result
/// equals the f64 reference on every element.
///
/// Mutation: in `kernel_mapping::shape_launch` drop the request-type exemption (`contraction = false`); the
/// workgroup becomes 256 while the body's LDS tree still folds 128 lanes, and the row goes red.
#[test]
fn a_contraction_keeps_its_lds_width_through_the_planner_on_wgpu() {
    let _g = gpu_lock();
    let Some((engine, device_target)) = open_engine_or_skip() else {
        return;
    };
    let mut exec: Box<dyn poot_executor::Executor> = Box::new(engine);
    let target = device_target;
    // Enough output elements that `out_numel / 128` passes the device's X grid cap: the generic bump's
    // trigger. A tiny cache length keeps the run short.
    let (cap, d) = (4usize, 4096usize);
    let hq = 128 * target.caps.max_grid[0] as usize / d + 1;
    let (fixture, scores, values) = attention_gemv_case(hq, cap, d);
    let program = compile(&fixture.graph, &target, &OPTIONS).expect("the fixture compiles");
    let (eqn, plan) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::MatMul))
        .expect("the scores @ V matmul");
    assert!(
        matches!(
            program.kernel_choice(eqn),
            KernelChoice::Generated(KernelRequest::Contraction(
                ContractionSpec::AttnScoresVGemv { width: 128, .. }
            ))
        ),
        "{:?}",
        program.kernel_choice(eqn)
    );
    let (poot_graph_plan::Plan::Compute { body, .. }
    | poot_graph_plan::Plan::ComputeMeta { body, .. }) = plan
    else {
        panic!("a single dispatch: {plan:?}");
    };
    assert_eq!(body.workgroup_size, [128, 1, 1], "the baked LDS width");
    let outputs = run_outputs(exec.as_mut(), target, &fixture).unwrap_or_else(|e| panic!("{e}"));
    let got = outputs[0].as_f32().expect("f32 output");
    assert_eq!(got.len(), hq * d);
    for h in 0..hq {
        for j in 0..d {
            let terms = (0..cap)
                .map(|c| f64::from(scores[h * cap + c]) * f64::from(values[(h * cap + c) * d + j]));
            let want: f64 = terms.clone().sum();
            let scale: f64 = terms.map(f64::abs).sum();
            let value = got[h * d + j];
            assert!(
                value.is_finite() && (f64::from(value) - want).abs() <= 1e-4 * scale,
                "out[{h}, {j}] = {value}, reference {want} (scale {scale})"
            );
        }
    }
}
