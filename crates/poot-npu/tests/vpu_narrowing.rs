//! POOT-1178: the narrowing contract of the Intel NPU, measured on the device.
//!
//! The NPU computes in FP16, so a program that stores F32 is planned over F16 values with an explicit
//! `Convert` at each region boundary (`narrow_to_device_precision`), under the numerical contract the device's
//! capabilities record. The rows here are the measurement that contract rests on: the Qwen2.5-0.5B MLP and
//! attention-core fragments (the same graphs as `vpu_qwen`, at the real dimensions) are built in F32, run on
//! the device through the ordinary route and compared with the F32 CPU oracle. The error is normwise,
//! `max |npu - oracle| / max |oracle|` over each step's result, which is the quantity the contract bounds; the
//! worst of the steps is printed as a `RECEIPT` and must stay within the recorded bound. Weights are
//! synthetic; no checkpoint is loaded.
//!
//! ```text
//! nix develop .#npu-intel -c cargo nextest run -p poot-npu --features vpu --test vpu_narrowing --test-threads=1 --no-capture
//! ```

#![cfg(feature = "vpu")]

mod vpu_common;

use poot_eval::test_support::StepFixture;
use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::{Device, NoSync};
use poot_executor_parity::{Fixture, fixture_weights, staged};
use poot_graph_ir::{BinOp, Builder, Scalar, Slot, TensorType, Traced, UnOp};
use poot_graph_plan::production_plan::Submission;
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore, WeightStoreBuilder};
use poot_tensor::{DType, HostTensor};

use vpu_common::{npu, slot_key};

const HIDDEN: usize = 896;
const INTER: usize = 4864;
const HEADS: usize = 14;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const TOKENS: usize = 8;

fn f32s(shape: &[usize]) -> TensorType {
    TensorType::f32(shape.to_vec())
}

/// `x @ w^T` for an HF-layout `[out, in]` weight.
fn linear(b: &Builder, x: Traced, w: Traced) -> Traced {
    let wt = b.transpose(w, vec![1, 0]).unwrap();
    b.matmul(x, wt).unwrap()
}

/// A named dense F32 weight whose element `i` is `scale * u` with `u` a deterministic value in about `[-1, 1]`.
fn weight(builder: &mut WeightStoreBuilder, name: &str, shape: &[usize], seed: usize, scale: f32) {
    let numel: usize = shape.iter().product();
    let bytes: Vec<u8> = (0..numel)
        .flat_map(|i| {
            let h = (i.wrapping_mul(2_654_435_761).wrapping_add(seed * 40_503)) % 2003;
            (scale * (h as f32 / 1001.0 - 1.0)).to_le_bytes()
        })
        .collect();
    let dense = DenseWeight::try_new(DType::F32, shape.to_vec(), StoredBytes::from(bytes)).unwrap();
    builder.insert(name, WeightEntry::Dense(dense)).unwrap();
}

/// Three steps of the `[TOKENS, HIDDEN]` activation `x`.
fn x_steps(modulus: usize, stride: usize, divisor: f32) -> Vec<Vec<StepFixture>> {
    (0..3)
        .map(|s| {
            vec![StepFixture {
                key: slot_key("x"),
                tensor: HostTensor::f32(
                    vec![TOKENS, HIDDEN],
                    (0..TOKENS * HIDDEN)
                        .map(|i| ((i * (s + stride)) % modulus) as f32 / divisor - 1.0)
                        .collect(),
                ),
            }]
        })
        .collect()
}

/// The MLP of the block with its residual, `x + down(gate(x) * up(x))`, over F32 values.
fn mlp_fragment() -> Fixture {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", f32s(&[TOKENS, HIDDEN]))
        .unwrap();
    let gate_w = b.constant("gate", f32s(&[INTER, HIDDEN])).unwrap();
    let up_w = b.constant("up", f32s(&[INTER, HIDDEN])).unwrap();
    let down_w = b.constant("down", f32s(&[HIDDEN, INTER])).unwrap();
    let gate = linear(&b, x, gate_w);
    let up = linear(&b, x, up_w);
    let hidden = b.binary(BinOp::Mul, gate, up).unwrap();
    let down = linear(&b, hidden, down_w);
    let out = b.binary(BinOp::Add, down, x).unwrap();
    let graph = b.finish(out).unwrap();
    let mut store = WeightStore::builder();
    weight(&mut store, "gate", &[INTER, HIDDEN], 1, 0.05);
    weight(&mut store, "up", &[INTER, HIDDEN], 2, 0.05);
    weight(&mut store, "down", &[HIDDEN, INTER], 3, 0.02);
    Fixture {
        name: "qwen2_mlp_f32",
        graph,
        store: store.build(),
        steps: x_steps(17, 3, 8.0),
    }
}

/// The attention core with its biases, grouped-query expansion, the score scale, the softmax numerator and the
/// output projection, `o(exp(q k^T / 8) v)`, over F32 values.
fn attention_fragment() -> Fixture {
    let b = Builder::for_tests();
    let (q_width, kv_width) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let group = HEADS / KV_HEADS;
    let x = b
        .slot_named(Slot::Activation, "x", f32s(&[TOKENS, HIDDEN]))
        .unwrap();
    let project = |name: &str, width: usize| {
        let w = b
            .constant(&format!("{name}_w"), f32s(&[width, HIDDEN]))
            .unwrap();
        let bias = b.constant(&format!("{name}_b"), f32s(&[1, width])).unwrap();
        let bias = b.broadcast(bias, vec![TOKENS, width]).unwrap();
        b.binary(BinOp::Add, linear(&b, x, w), bias).unwrap()
    };
    let q = project("q", q_width);
    let k = project("k", kv_width);
    let v = project("v", kv_width);
    let to_heads = |t: Traced, heads: usize| {
        let t = b.reshape(t, vec![1, TOKENS, heads, HEAD_DIM]).unwrap();
        b.transpose(t, vec![0, 2, 1, 3]).unwrap()
    };
    let expand = |t: Traced| {
        let t = b
            .reshape(t, vec![1, KV_HEADS, 1, TOKENS, HEAD_DIM])
            .unwrap();
        let t = b
            .broadcast(t, vec![1, KV_HEADS, group, TOKENS, HEAD_DIM])
            .unwrap();
        b.reshape(t, vec![1, HEADS, TOKENS, HEAD_DIM]).unwrap()
    };
    let q = to_heads(q, HEADS);
    let k = expand(to_heads(k, KV_HEADS));
    let v = expand(to_heads(v, KV_HEADS));
    let kt = b.transpose(k, vec![0, 1, 3, 2]).unwrap();
    let scores = b.matmul(q, kt).unwrap();
    let scores = b
        .binary_scalar(BinOp::Mul, scores, Scalar::F32(0.125))
        .unwrap();
    let weights = b.unary(UnOp::Exp, scores).unwrap();
    let mixed = b.matmul(weights, v).unwrap();
    let mixed = b.transpose(mixed, vec![0, 2, 1, 3]).unwrap();
    let mixed = b.reshape(mixed, vec![TOKENS, q_width]).unwrap();
    let o_w = b.constant("o_w", f32s(&[HIDDEN, q_width])).unwrap();
    let out = linear(&b, mixed, o_w);
    let graph = b.finish(out).unwrap();
    let mut store = WeightStore::builder();
    weight(&mut store, "q_w", &[q_width, HIDDEN], 4, 0.05);
    weight(&mut store, "q_b", &[1, q_width], 5, 0.1);
    weight(&mut store, "k_w", &[kv_width, HIDDEN], 6, 0.05);
    weight(&mut store, "k_b", &[1, kv_width], 7, 0.1);
    weight(&mut store, "v_w", &[kv_width, HIDDEN], 8, 0.05);
    weight(&mut store, "v_b", &[1, kv_width], 9, 0.1);
    weight(&mut store, "o_w", &[HIDDEN, q_width], 10, 0.02);
    Fixture {
        name: "qwen2_attention_core_f32",
        graph,
        store: store.build(),
        steps: x_steps(19, 5, 9.0),
    }
}

/// Run `fixture` on the NPU and return its worst normwise error against the F32 oracle over the steps, and the
/// bound the device's capabilities record; `None` when the device is absent (a printed skip).
fn measure(fixture: &Fixture) -> Option<(f32, f32)> {
    let mut npu = npu()?;
    let target = npu.engine.device().target();
    let poot_graph_plan::Target::Vpu { caps, .. } = target else {
        panic!("the NPU device plans for a Vpu target");
    };
    let recorded = caps.narrowing.expect("the device records its contract");
    let bound = recorded.max_relative_error_ppm as f32 * 1.0e-6;
    let exec = &mut npu.engine;
    let exe = poot_executor::Executor::load_weights(
        exec,
        std::sync::Arc::new(fixture.store.clone()),
        fixture_weights(fixture),
    )
    .unwrap();
    let entry =
        poot_executor::Executor::add_entry(exec, exe, &staged(fixture, target, Submission::Replay))
            .unwrap();
    let mut oracle = oracle_for(&fixture.graph, &fixture.store);
    let mut worst = 0.0f32;
    for (i, step) in fixture.steps.iter().enumerate() {
        let mut inputs = poot_executor::StepInputs::new();
        for v in step {
            inputs.push(v.key.clone(), v.tensor.shape(), v.tensor.view());
        }
        let got = poot_executor::Executor::step(exec, exe, entry, &inputs, &mut NoSync)
            .unwrap()
            .to_host()
            .unwrap();
        let want = oracle(step);
        assert_eq!(
            got.dtype(),
            DType::F32,
            "{} step {i}: the result keeps the program's F32",
            fixture.name
        );
        assert_eq!(
            got.shape(),
            want.shape(),
            "{} step {i}: shape",
            fixture.name
        );
        let (got, want) = (got.to_f32().unwrap(), want.to_f32().unwrap());
        let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        let error = got.iter().zip(&*want).fold(0.0f32, |m, (&g, &w)| {
            if g.is_nan() {
                f32::INFINITY
            } else {
                m.max((g - w).abs())
            }
        });
        worst = worst.max(error / scale);
    }
    poot_executor::Executor::unload(exec, exe).unwrap();
    println!(
        "RECEIPT {}: worst normwise error {worst:.3e} over {} steps; recorded bound {bound:.3e}",
        fixture.name,
        fixture.steps.len()
    );
    Some((worst, bound))
}

/// The F32 fragments run through the narrowed region and stay within the contract the device records.
///
/// Mutation: lower the recorded bound in `NPU4_F32_NARROWING` (poot-npu) below the measured worst; this row
/// fails with the measured error and the bound.
#[test]
fn the_f32_fragments_stay_within_the_recorded_narrowing_contract() {
    for fixture in [mlp_fragment(), attention_fragment()] {
        let Some((worst, bound)) = measure(&fixture) else {
            return;
        };
        assert!(
            worst <= bound,
            "{}: the narrowed region's normwise error {worst:.3e} exceeds the recorded contract {bound:.3e}",
            fixture.name
        );
    }
}

/// `out = x + x` over F32 `[4]`, where `scale` multiplies it: the smallest program whose F32 input and result
/// both cross the narrowing boundary.
fn doubling_fixture() -> Fixture {
    let b = Builder::for_tests();
    let x = b.slot_named(Slot::Activation, "x", f32s(&[4])).unwrap();
    let out = b.binary(BinOp::Add, x, x).unwrap();
    Fixture {
        name: "doubling_f32",
        graph: b.finish(out).unwrap(),
        store: WeightStore::builder().build(),
        steps: Vec::new(),
    }
}

fn run_step(
    npu: &mut vpu_common::Npu,
    exe: poot_executor::ExecutableId,
    entry: poot_executor::EntryId,
    x: [f32; 4],
) -> Result<Vec<f32>, poot_executor::ExecError> {
    let tensor = HostTensor::f32(vec![4], x.to_vec());
    let mut inputs = poot_executor::StepInputs::new();
    inputs.push(slot_key("x"), tensor.shape(), tensor.view());
    match poot_executor::Executor::step(&mut npu.engine, exe, entry, &inputs, &mut NoSync) {
        Ok(mut out) => Ok(out.to_host().unwrap().to_f32().unwrap().to_vec()),
        Err(error) => {
            // A failed step leaves the executable needing a reset before its next one.
            poot_executor::Executor::reset_state(
                &mut npu.engine,
                exe,
                poot_executor::StateScope::All,
            )
            .unwrap();
            Err(error)
        }
    }
}

/// The range the contract records is enforced at the device, on the range edge: an F32 input of 65504 (the
/// largest finite FP16) runs and the result is the F32 sum; an input of 65520, which FP16 rounds to infinity, is
/// a typed `NarrowingRange` error before the replay, as is a NaN; a result that overflows FP16 from in-range
/// inputs (`40000 + 40000`) is the same error after it. Measured on the laptop NPU: that sum comes back as the
/// finite F32 `80000`, not infinity, so the region does not saturate there; the range is what the contract's
/// bound stops holding for. None of them poisons the device: after a reset the next in-range step runs.
///
/// Mutation: drop the `RangeCheck` loop over inputs in `VpuDevice::replay` (poot-npu); 65520 runs and the row
/// fails on the missing error. Drop the loop over results; the out-of-range sum comes back as a plain F32.
#[test]
fn an_f32_value_outside_the_narrowing_range_is_a_typed_error() {
    let fixture = doubling_fixture();
    let Some(mut npu) = npu() else {
        return;
    };
    let target = npu.engine.device().target();
    let poot_graph_plan::Target::Vpu { caps, .. } = target else {
        panic!("the NPU device plans for a Vpu target");
    };
    let contract = caps.narrowing.expect("the device records its contract");
    assert_eq!(contract.max_magnitude, 65_504);
    let exe = poot_executor::Executor::load_weights(
        &mut npu.engine,
        std::sync::Arc::new(fixture.store.clone()),
        fixture_weights(&fixture),
    )
    .unwrap();
    let entry = poot_executor::Executor::add_entry(
        &mut npu.engine,
        exe,
        &staged(&fixture, target, Submission::Replay),
    )
    .unwrap();

    let range_error = |result: Result<Vec<f32>, poot_executor::ExecError>| {
        let poot_executor::ExecError::Device(device) = result.as_ref().expect_err("the value is refused")
        else {
            panic!("a device refusal, got {result:?}");
        };
        let error = device
            .source
            .downcast_ref::<poot_npu::VpuError>()
            .expect("the NPU's own error");
        let poot_npu::VpuError::NarrowingRange {
            role,
            index,
            value,
            max_magnitude,
            ..
        } = error
        else {
            panic!("a NarrowingRange error, got {error}");
        };
        assert_eq!(*max_magnitude, 65_504, "{error}");
        (*role, *index, *value)
    };

    // The edge: 65504 / 2 is the largest value whose double is still finite in FP16.
    let edge = run_step(&mut npu, exe, entry, [1.0, 2.0, 3.0, 32_752.0]).unwrap();
    assert_eq!(edge, [2.0, 4.0, 6.0, 65_504.0]);
    // An input of exactly 65504 is admitted (its double is outside the range: a result error).
    assert_eq!(
        range_error(run_step(&mut npu, exe, entry, [0.0, 0.0, 0.0, 65_504.0])),
        ("result", 3, 131_008.0)
    );
    let (role, index, value) =
        range_error(run_step(&mut npu, exe, entry, [0.0, 65_520.0, 0.0, 0.0]));
    assert_eq!((role, index, value), ("input", 1, 65_520.0));
    let (role, index, value) =
        range_error(run_step(&mut npu, exe, entry, [f32::NAN, 0.0, 0.0, 0.0]));
    assert_eq!((role, index), ("input", 0));
    assert!(value.is_nan());
    assert_eq!(
        range_error(run_step(&mut npu, exe, entry, [40_000.0, 0.0, 0.0, 0.0])),
        ("result", 0, 80_000.0)
    );
    // A refusal is not a failed replay: the device is healthy and the next in-range step runs.
    let after = run_step(&mut npu, exe, entry, [4.0, 5.0, 6.0, 7.0]).unwrap();
    assert_eq!(after, [8.0, 10.0, 12.0, 14.0]);
    poot_executor::Executor::unload(&mut npu.engine, exe).unwrap();
}
