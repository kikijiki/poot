//! The opt-in range detector on a device (POOT-1475, `receipts/1471-narrow-without-range-guards/design.md` section
//! 6.2): a program compiled with `CompileOptions::range_check` computes its range witnesses with the device's own
//! kernels, and a step that leaves f16's range is `ExecError::Validation` naming the first op, its layer and the
//! operand, before any output is read.
//!
//! Three claims, each from the production path (`compile_staged`, `add_entry`, `step`):
//!
//! - past-range blocks fault, and name a contraction's activation;
//! - an in-range block passes with a finite output (no false positive from a device's comparison or reduction);
//! - a softcapped attention whose query is past f16's range publishes a finite output without the detector (the
//!   limit of the logits refusal, design 4.3) and faults naming layer 17's attention query with it.

use std::sync::Arc;

use poot_eval::test_support::StepFixture;
use poot_executor::{Executor, NoSync};
use poot_graph_ir::{
    AttentionAttrs, Builder, MatmulPrecision, OpKind, Operand, OperandPrecision, Slot, TensorType,
};
use poot_graph_plan::Target;
use poot_graph_plan::{CompileLimits, CompileOptions};
use poot_tensor::{DType, HostTensor};

use crate::Fixture;

const HQ: usize = 4;
const HKV: usize = 2;
// Rows enough to fill the device, which a fragment attention tiling requires (a smaller one is refused as narrowed).
const LQ: usize = 512;
const LK: usize = 512;
const D: usize = 64;

/// A causal attention in layer 17 over an f16 cache of positive keys and values, with a score softcap and a positive
/// query of magnitude up to `q_scale` supplied as the step's activation. Every entry is positive, so an overflowed
/// query gives `+inf` scores, never `inf - inf`: the softcap maps them to a finite constant.
fn softcapped_attention(q_scale: f32) -> Fixture {
    let b = Builder::for_tests();
    let q = b
        .slot_named(Slot::Activation, "q", TensorType::f32(vec![1, HQ, LQ, D]))
        .unwrap();
    let kv_type = TensorType::new(vec![1, HKV, LK, D], DType::F16);
    let k = b.constant("k", kv_type.clone()).unwrap();
    let v = b.constant("v", kv_type).unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, LQ, LK]))
        .unwrap();
    let _layer = b.layer_scope(17);
    let mut plan = b.append_plan(0).unwrap();
    let out = plan
        .equation(
            OpKind::Attention(AttentionAttrs {
                precision: OperandPrecision::Exact,
                ..AttentionAttrs::new(HQ / HKV, 1.0e-6, Some(30.0))
            }),
            [q, k, v, mask].map(|t| Operand::Value(t.id())).to_vec(),
        )
        .expect("the attention types");
    plan.declare_result(out).unwrap();
    let mut prepared = b.preflight_append(plan).unwrap();
    let out = b.commit_append(&mut prepared).unwrap();
    let graph = b.finish(out).unwrap();
    let positive = |n: usize, seed: u64, scale: f32| -> Vec<f32> {
        crate::fill(n, seed)
            .into_iter()
            .map(|v| (v.abs() + 0.05) * scale)
            .collect()
    };
    let f16 = |seed| {
        HostTensor::f16(
            vec![1, HKV, LK, D],
            positive(HKV * LK * D, seed, 1.0)
                .into_iter()
                .map(poot_quant::scalar::f32_to_f16)
                .collect::<Vec<u16>>(),
        )
    };
    let causal = (0..LQ)
        .flat_map(|i| (0..LK).map(move |j| if j <= i + LK - LQ { 0.0 } else { -1.0e9 }))
        .collect();
    let key = graph.meta(q.id()).slot_key().unwrap().clone();
    let steps = (0..2u64)
        .map(|step| {
            vec![StepFixture {
                key: key.clone(),
                tensor: HostTensor::f32(
                    vec![1, HQ, LQ, D],
                    positive(HQ * LQ * D, 0x1475 + step, q_scale),
                ),
            }]
        })
        .collect();
    Fixture {
        name: "softcapped_attention_past_range",
        graph,
        store: crate::store_from_tensors(vec![
            ("k", f16(12)),
            ("v", f16(13)),
            ("mask", HostTensor::f32(vec![1, 1, LQ, LK], causal)),
        ]),
        steps,
    }
}

/// The narrow policy with the detector `range_check`. The detector's subject is the narrowed operand, so the narrowing
/// is declared rather than left to the planner's cost model (which keeps a one-wave device exact).
fn narrowed_options(exec: &dyn Executor, range_check: bool) -> CompileOptions {
    CompileOptions {
        declare_precision: true,
        ..exec
            .compile_options(CompileLimits::STANDARD)
            .with_matmul_precision(MatmulPrecision::Narrow)
            .with_range_check(range_check)
    }
}

/// Every step of `fixture` on `exec`, compiled under the narrow policy with the detector `range_check`: each step's
/// output, or the error of the first step that failed.
fn run_steps(
    exec: &mut dyn Executor,
    target: Target,
    fixture: &Fixture,
    range_check: bool,
) -> Result<Vec<HostTensor>, String> {
    let options = narrowed_options(exec, range_check);
    let staged = crate::compile_fixture(fixture, target, &options);
    let exe = exec
        .load_weights(
            Arc::new(fixture.store.clone()),
            crate::fixture_weights(fixture),
        )
        .map_err(|e| format!("{}: load_weights: {e}", fixture.name))?;
    let entry = exec
        .add_entry(exe, &staged)
        .map_err(|e| format!("{}: add_entry: {e}", fixture.name))?;
    let mut outputs = Vec::with_capacity(fixture.steps.len());
    let mut failure = None;
    for (i, step) in fixture.steps.iter().enumerate() {
        let inputs = crate::BoundStep::new(&fixture.graph, step)?;
        match exec.step(exe, entry, &inputs.inputs(), &mut NoSync) {
            Ok(mut output) => outputs.push(
                output
                    .to_host()
                    .map_err(|e| format!("{}: read {i}: {e}", fixture.name))?,
            ),
            Err(e) => {
                failure = Some(format!("{}: step {i}: {e}", fixture.name));
                break;
            }
        }
    }
    exec.unload(exe)
        .map_err(|e| format!("{}: unload: {e}", fixture.name))?;
    failure.map_or(Ok(outputs), Err)
}

/// `fixture` faults under the detector with a validation fault whose message contains every one of `names`.
fn expect_fault(
    exec: &mut dyn Executor,
    target: Target,
    fixture: &Fixture,
    names: &[&str],
) -> Result<(), String> {
    match run_steps(exec, target, fixture, true) {
        Ok(_) => Err(format!(
            "{}: the detector let a step past f16's range pass",
            fixture.name
        )),
        Err(message) if names.iter().all(|name| message.contains(name)) => Ok(()),
        Err(message) => Err(format!(
            "{}: the fault does not name {names:?}: {message}",
            fixture.name
        )),
    }
}

/// Every step of `fixture` runs and publishes only finite values.
fn expect_finite(outputs: Vec<HostTensor>, name: &str) -> Result<(), String> {
    for (step, output) in outputs.iter().enumerate() {
        let values = output
            .to_f32()
            .map_err(|e| format!("{name}: an output does not read as f32: {e}"))?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("{name}: step {step}: a non-finite output"));
        }
    }
    Ok(())
}

/// The detector's three claims on the device `exec` drives (module docs).
pub fn run_range_check(exec: &mut dyn Executor, target: Target) -> Result<(), String> {
    for fixture in crate::f16_range::block_fixtures() {
        expect_fault(exec, target, &fixture, &["f16_range.", ".activation"])?;
    }
    let in_range = crate::f16_range::in_range_block_fixture();
    let outputs = run_steps(exec, target, &in_range, true)?;
    expect_finite(outputs, in_range.name)?;

    let softcapped = softcapped_attention(1.0e6);
    // A device whose fragments cannot run the attention (one wave: radv) holds the declared narrowing and refuses it
    // by name, so no narrowed attention exists there for the detector to watch.
    if let Err(error) =
        crate::try_compile_fixture(&softcapped, target, &narrowed_options(exec, true))
    {
        return if error.contains("NoMatrixUnitSchedule") {
            Ok(())
        } else {
            Err(error)
        };
    }
    let silent = run_steps(exec, target, &softcapped, false)?;
    expect_finite(silent, softcapped.name)
        .map_err(|e| format!("{e}: the softcap was expected to keep the overflow finite"))?;
    expect_fault(
        exec,
        target,
        &softcapped,
        &["f16_range.", ".l17.", ".attention.query"],
    )
}
