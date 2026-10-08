//! POOT-1040 on PTX: a guarded live-attention graph the matcher declines (no declared length witness, so the
//! length is not bounded) stays generic primitives, key ends in exact I32 included, and a live PTX device runs it.

use poot_eval::{EvalBudget, EvalOptions, eval};
use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::live_attention::{compile_live, live_step};
use poot_graph_plan::{KernelChoice, Target};
use poot_kernelgen::KernelRequest;
use poot_ptx_gpu::PtxDevice;
use poot_runtime_common::DeviceBackend;
use poot_test_util::device_skip::open_or_skip;
use poot_test_util::live_attention::{LiveAttentionFixture, LiveAttentionGraph};

/// Whether the plan `built` compiles to for `target` holds a generated attention kernel.
fn plans_attention_kernel(built: &LiveAttentionGraph, target: Target) -> bool {
    compile_live(&built.graph, target)
        .stages()
        .any(|(_, _, stage)| {
            stage.planned().any(|(eqn, _)| {
                matches!(
                    stage.kernel_choice(eqn),
                    KernelChoice::Generated(KernelRequest::Attention(_))
                )
            })
        })
}

/// With NaN in every key and value past the length, the output is finite and within the attention tier of the
/// first-5-key oracle, and the plan has no attention kernel. Mutation: build `LiveKeys::new`'s key ends from an
/// F32 iota plus a `Cast` to I32 (`ops::attention_over_keys`); the plan refuses the F32 -> I32 cast.
#[test]
fn a_declined_live_attention_graph_runs_as_primitives_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let fixture = LiveAttentionFixture {
        batch: 1,
        hq: 2,
        hkv: 1,
        cap: 9,
        dk: 8,
        dv: 8,
        shared_dim: None,
        softcap: None,
        dtype: poot_tensor::DType::F32,
    };
    let built = fixture.build(false);
    let inputs = fixture.inputs(&built, &[5], true);
    assert!(
        !plans_attention_kernel(&built, target),
        "the unproved graph stays generic"
    );
    let got = live_step(&mut *exec, target, &built, &inputs)
        .expect("a declined graph runs on the device");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    let cpu = eval(
        &built.graph,
        &inputs.values(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let want = fixture.oracle(&inputs, &built, &[5]);
    for (label, reference) in [
        ("guarded graph", cpu.as_f32().unwrap()),
        ("first 5 keys", &want[..]),
    ] {
        for (i, (g, w)) in got.iter().zip(reference).enumerate() {
            assert!(
                (g - w).abs() <= 1e-3,
                "vs the {label}: element {i}: device {g} vs oracle {w}"
            );
        }
    }
}

/// POOT-1041 on PTX: the guarded live composite over BF16 and F16 operands (dense and split key, batch rows
/// with different lengths, NaN in every key and value past each length) runs as generic primitives at its own
/// dtype, the two-byte `Select` guarding keys, values and scores. The output is finite and within the two-byte
/// tier of the guarded graph's CPU evaluation and of the first-`len` keys' oracle. Mutation: emit the select
/// as an unconditional `if_true` in `emit_select` (kernelgen `fused.rs`); the poisoned tail reaches the
/// matmuls and the finiteness assertion fails.
#[test]
fn a_narrow_live_attention_graph_runs_on_ptx_and_never_reads_the_poisoned_tail() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for (dtype, tier) in [
        (poot_tensor::DType::BF16, 4e-3),
        (poot_tensor::DType::F16, 1e-3),
    ] {
        for shared_dim in [None, Some(4)] {
            let fixture = LiveAttentionFixture {
                batch: 2,
                hq: 4,
                hkv: 2,
                cap: 37,
                dk: 12,
                dv: 8,
                shared_dim,
                softcap: Some(30.0),
                dtype,
            };
            let label = format!("{dtype:?} shared {shared_dim:?}");
            let built = fixture.build(true);
            let len = [5, 37];
            let inputs = fixture.inputs(&built, &len, true);
            assert!(
                !plans_attention_kernel(&built, target),
                "{label}: the F32 attention kernel must not claim a narrow chain"
            );
            let got = live_step(&mut *exec, target, &built, &inputs)
                .unwrap_or_else(|e| panic!("{label}: a narrow live graph runs on PTX: {e:?}"));
            assert!(
                got.iter().all(|x| x.is_finite()),
                "{label}: poison reached the output: {got:?}"
            );
            let cpu = eval(
                &built.graph,
                &inputs.values(),
                EvalOptions::new(EvalBudget::UNBOUNDED),
            )
            .unwrap()
            .output
            .into_host()
            .unwrap();
            let want = fixture.oracle(&inputs, &built, &len);
            for (reference_label, reference) in [
                ("guarded graph", cpu.as_f32().unwrap()),
                ("first keys", &want[..]),
            ] {
                let worst = got
                    .iter()
                    .zip(reference)
                    .map(|(g, w)| {
                        if (g - w).is_nan() {
                            f32::INFINITY
                        } else {
                            (g - w).abs()
                        }
                    })
                    .fold(0.0, f32::max);
                assert!(
                    worst <= tier,
                    "{label}: vs the {reference_label}: worst difference {worst:.3e}"
                );
            }
        }
    }
}
