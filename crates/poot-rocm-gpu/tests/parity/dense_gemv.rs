//! Card 727 SC-002 on ROCm: the generated dense Gemv against an f64 host reference over every output element
//! (ADR-0101 tier 2). The fixture helpers are the wgpu row's (`poot-gpu/tests/graph/dense_gemv.rs`).

use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::{Fixture, run_outputs};
use poot_graph_ir::{Builder, OpKind, Slot, TensorType};
use poot_graph_plan::{
    CompileLimits, CompileOptions, FusionPolicy, KernelChoice, Submission, Target, compile,
};
use poot_kernelgen::{ContractionSpec, KernelRequest, Schedule};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_rocm_gpu::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_tensor::{DType, HostTensor};
use poot_test_util::StepFixture;
use poot_test_util::device_skip::open_or_skip;
use std::sync::Arc;

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

/// Card 727 SC-002 on ROCm: the generated dense Gemv over an F32, packed-F16 and packed-BF16 `[N, K]` weight,
/// at every launch the planner admits (`cols` 4 and 2, reached by the output width), meets
/// ADR-0101 tier 2 on every output. `K = 1037` leaves a K tail past the last full run at every launch, `K = 515`
/// is odd so packed rows start mid-word, and no `N` is a multiple of its `cols`.
///
/// Mutation: in `contraction::gemv` start every lane's first run one trip late (skip the first K tile), or skip
/// the tail runs; the prescribed contribution is lost and the row goes red.
#[test]
fn the_dense_gemv_meets_tier_2_in_every_lane_and_launch_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
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
