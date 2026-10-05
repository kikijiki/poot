//! End-to-end dispatch test for the SPIR-V cooperative-matrix (RADV tensor-core) matmul arm, through
//! the executor contract.
//!
//! `coopmat_dispatch_probe.rs` (poot-kernelgen) dispatches the kernel body by hand and
//! `coopmat_matmul_traces_plans_and_validates` (poot-graph-plan) checks the `Backend::SpirvVulkan`
//! selection, but neither goes through tracer -> plan -> `Engine::step`. This test builds a small F16
//! matmul with `Builder`, runs it through [`poot_executor_parity::run_once`] and compares against the
//! f16-rounded CPU oracle.
//!
//! Shape/dtype follow `matmul_spirv_coopmat_eligible` (card 154): F16 operands, F32 output, K == 16, M/N
//! 16-aligned, no batch dims. `Builder::matmul` infers the output dtype from operand A (F16), so the
//! output is retagged to F32 via `ValueMeta::aval`, as `poot-graph-plan`'s `mixed_matmul_graph_shape`
//! helper does; no checkpoint loader produces an f16-in/f32-out matmul yet.

use poot_runtime_common::DeviceBackend;
use poot_tensor::{DType, HostTensor};

use poot_executor::{Device, Executor};
use poot_executor_parity::{ConstFixture, run_once};
use poot_gpu::device::WgpuDevice;

use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::OpKind;
use poot_graph_ir::types::TensorType;

/// `a`/`w` are declared F16: honest F16 words, the same narrowing `poot_load::gguf::f32_to_f16` already
/// does for the CPU oracle below.
fn f16_tensor(shape: Vec<usize>, data: &[f32]) -> HostTensor {
    HostTensor::f16(
        shape,
        data.iter()
            .map(|&x| poot_load::gguf::f32_to_f16(x))
            .collect(),
    )
}

/// Serialize GPU access (Vulkan segfaults under concurrent device use on some adapters).
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Build `a[1,m,k] @ w[k,n] -> [1,m,n]` with F16 operands, then retag the output to F32 (the shape
/// `matmul_spirv_coopmat_eligible` accepts).
fn f16_in_f32_out_matmul_graph(
    m: usize,
    k: usize,
    n: usize,
) -> (
    poot_graph_ir::Graph,
    poot_graph_ir::ValueId,
    poot_graph_ir::ValueId,
) {
    let bld = Builder::new();
    let a = bld.constant("a", TensorType::new(vec![1, m, k], DType::F16));
    let w = bld.constant("w", TensorType::new(vec![k, n], DType::F16));
    let mm = bld.matmul(a, w);
    let (ai, wi) = (a.id, w.id);
    let mut g = bld.finish(mm);
    let out = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .expect("a matmul eqn")
        .out;
    assert_eq!(
        g.values[out].aval.dtype,
        DType::F16,
        "sanity: Builder::matmul must still infer F16 (operand A's dtype) before the retag"
    );
    g.values[out].aval.dtype = DType::F32;
    (g, ai, wi)
}

/// Trace an F16 matmul, let `plan_eqn` select the `Backend::SpirvVulkan` coopmat arm, and run it through
/// the executor contract on RADV. Multi-tile (32x16x32, 2x2 output tiles), the shape
/// `coopmat_dispatch_probe.rs` already proves at kernel level, so a mismatch here points at plan/bind/exec wiring.
#[test]
fn coopmat_matmul_run_resident_matches_cpu_on_radv() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

    let (m, k, n) = (32usize, 16usize, 32usize);
    let (g, _ai, _wi) = f16_in_f32_out_matmul_graph(m, k, n);

    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.04).collect();
    let w_data: Vec<f32> = (0..k * n).map(|i| ((i % 19) as f32 - 9.0) * 0.04).collect();
    let consts = [
        ConstFixture {
            name: "a",
            tensor: f16_tensor(vec![1, m, k], &a_data),
        },
        ConstFixture {
            name: "w",
            tensor: f16_tensor(vec![k, n], &w_data),
        },
    ];

    let got = match run_once(&mut *exec, target, &g, &consts, &[]) {
        Ok(out) => out.as_f32().expect("the matmul output is F32").to_vec(),
        Err(e) => {
            panic!(
                "run_once failed on the real graph-trace -> plan -> execute coopmat path \
                 (kernel-body dispatch is already proven correct by coopmat_dispatch_probe.rs, so \
                 this is a plan/bind/exec wiring gap, not a coopmat kernel bug): {e}"
            );
        }
    };

    // CPU oracle: round A/W to f16 (the coopmat load narrows), f32-accumulate.
    let round = |x: f32| poot_quant::scalar::f16_to_f32(poot_load::gguf::f32_to_f16(x));
    let ra: Vec<f32> = a_data.iter().map(|&x| round(x)).collect();
    let rw: Vec<f32> = w_data.iter().map(|&x| round(x)).collect();
    let mut max_abs = 0.0f32;
    let mut first_bad = None;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += ra[i * k + kk] * rw[kk * n + j];
            }
            let gv = got[i * n + j];
            let d = (gv - acc).abs();
            max_abs = max_abs.max(d);
            let tol = acc.abs() * 0.02 + 1e-2;
            // A NaN output is a mismatch: `d > tol` alone is false for it.
            if (d.is_nan() || d > tol) && first_bad.is_none() {
                first_bad = Some((i, j, acc, gv));
            }
        }
    }
    if let Some((i, j, want, gv)) = first_bad {
        panic!(
            "run_resident coopmat matmul MISMATCH at [{i},{j}]: want {want:.4} got {gv:.4} \
             (max_abs={max_abs:.3e}) - the real plan->bind->exec path disagrees with the \
             kernel-body-level probe in coopmat_dispatch_probe.rs"
        );
    }
    eprintln!(
        "coopmat run_resident [{m}x{k}]@[{k}x{n}] OK end to end on RADV (max_abs={max_abs:.3e})"
    );
}

/// Checks the graph reaches the coopmat plan arm rather than a fallback that gives the same numbers.
/// Inspects the kernel choice directly; no GPU needed.
#[test]
fn coopmat_matmul_run_resident_graph_selects_coopmat_arm() {
    use poot_graph_plan::Plan;
    use poot_graph_plan::cse;
    use poot_target::Backend;

    let (g, _ai, _wi) = f16_in_f32_out_matmul_graph(32, 16, 32);
    let g = cse(&g);
    let eqn = g
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .expect("a matmul eqn");
    let (plan, choice) = plan_eqn_with_choice(
        &g,
        eqn,
        Backend::SpirvVulkan,
        &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
    )
    .expect("the coopmat-eligible graph must plan on SpirvVulkan");
    assert!(
        matches!(plan, Plan::Compute { .. }),
        "expected Plan::Compute, got {plan:?}"
    );
    assert!(
        matches!(
            choice,
            poot_graph_plan::KernelChoice::Generated(poot_kernelgen::KernelRequest::Contraction(
                poot_kernelgen::ContractionSpec::Coopmat { .. }
            ))
        ),
        "expected run_resident's exec_resident to select the coopmat arm too (same plan_eqn call \
         with the same backend/graph shape): got {choice:?}"
    );
}

/// One equation's plan and the kernel choice behind it, planned for a single device with no strided views
/// (the planner entry point `compile` itself drives per equation). Planner-selection tests assert on the
/// choice, never on the plan's opaque key.
fn plan_eqn_with_choice(
    g: &poot_graph_ir::Graph,
    eqn: &poot_graph_ir::Eqn,
    backend: poot_target::Backend,
    caps: &poot_target::DeviceCaps,
) -> Result<(poot_graph_plan::Plan, poot_graph_plan::KernelChoice), poot_graph_plan::PlanError> {
    poot_graph_plan::plan_eqn_choice_analyzed(
        &poot_graph_plan::ExactI32StorageAnalysis::new(g),
        g,
        eqn,
        backend,
        1,
        &std::collections::HashMap::new(),
        caps,
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .map(|planned| (planned.plan, planned.choice))
}
