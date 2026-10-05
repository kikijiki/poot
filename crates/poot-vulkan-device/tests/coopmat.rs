//! The cooperative-matrix (tensor-core) matmul arm on raw Vulkan, through the executor contract (Card 553,
//! R-553-1): `Context::device_caps` reports matrix hardware only because the device enables
//! `VK_KHR_cooperative_matrix` and the SPIR-V the planner selects for it declares the capability, so a
//! planned coopmat kernel is one the device runs.
//!
//! An F16 `[1, 32, 16] @ [16, 32]` matmul with the output retagged F32 (the shape
//! `matmul_spirv_coopmat_eligible` accepts) is compiled for the device's own measured target, must plan
//! the coopmat arm, and must match the f16-rounded f32-accumulated CPU product. The row skips on a device
//! with no cooperative-matrix hardware (it reports `TensorCoreSupport::None`).
//!
//! Mutation: create the cooperative-matrix pipeline with a required subgroup size of 64 in place of the
//! module's `LocalSize` X of 32 (`Context::build_pipeline`); the 32-lane kernel runs in wave64 and the row
//! goes red on the product (`[2,0]: want 0.0561, got 0.0000`). A device that stops reporting its matrix
//! hardware does not turn this row red, it skips it; `poot-vulkan-runtime`'s `device_caps` row
//! (`raw_vulkan_reports_wmma_class_support_on_this_box`) goes red instead, and the lane's required-case gate
//! counts a skip as a failure.

use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::{ConstFixture, run_once};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::OpKind;
use poot_graph_ir::types::TensorType;
use poot_runtime_common::DeviceBackend;
use poot_target::{Backend, TensorCoreSupport};
use poot_tensor::{DType, HostTensor};
use poot_test_util::device_skip::open_or_skip;
use poot_vulkan_device::VulkanDevice;

fn f16_tensor(shape: Vec<usize>, data: &[f32]) -> HostTensor {
    HostTensor::f16(
        shape,
        data.iter()
            .map(|&x| poot_load::gguf::f32_to_f16(x))
            .collect(),
    )
}

/// `a[1, m, k] @ w[k, n]` over F16 operands with the output retagged F32.
fn f16_in_f32_out_matmul_graph(m: usize, k: usize, n: usize) -> poot_graph_ir::Graph {
    let builder = Builder::new();
    let a = builder.constant("a", TensorType::new(vec![1, m, k], DType::F16));
    let w = builder.constant("w", TensorType::new(vec![k, n], DType::F16));
    let product = builder.matmul(a, w);
    let mut graph = builder.finish(product);
    let out = graph
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .expect("a matmul eqn")
        .out;
    graph.values[out].aval.dtype = DType::F32;
    graph
}

#[test]
fn coopmat_matmul_plans_the_coopmat_arm_and_matches_the_cpu_product_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
        return;
    };
    let target = device.target();
    if target.caps.tensor_core == TensorCoreSupport::None {
        eprintln!("SKIP: this device reports no cooperative-matrix hardware");
        return;
    }
    let (m, k, n) = (32usize, 16usize, 32usize);
    let graph = f16_in_f32_out_matmul_graph(m, k, n);

    // The planner selects the coopmat arm for this target.
    let analysis = poot_graph_plan::ExactI32StorageAnalysis::new(&graph);
    let eqn = graph
        .eqns
        .iter()
        .find(|e| matches!(e.op, OpKind::MatMul))
        .unwrap();
    let planned = poot_graph_plan::plan_eqn_choice_analyzed(
        &analysis,
        &graph,
        eqn,
        Backend::SpirvVulkan,
        1,
        &std::collections::HashMap::new(),
        &target.caps,
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .expect("the coopmat-eligible matmul plans on SpirvVulkan");
    assert!(
        matches!(
            planned.choice,
            poot_graph_plan::KernelChoice::Generated(poot_kernelgen::KernelRequest::Contraction(
                poot_kernelgen::ContractionSpec::Coopmat { .. }
            ))
        ),
        "expected the coopmat arm, got {:?}",
        planned.choice
    );

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
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let out = run_once(&mut *exec, target, &graph, &consts, &[])
        .unwrap_or_else(|e| panic!("the coopmat matmul runs through the contract: {e}"));
    let got = out.as_f32().expect("the matmul output is F32");

    let round = |x: f32| poot_quant::scalar::f16_to_f32(poot_load::gguf::f32_to_f16(x));
    let ra: Vec<f32> = a_data.iter().map(|&x| round(x)).collect();
    let rw: Vec<f32> = w_data.iter().map(|&x| round(x)).collect();
    for i in 0..m {
        for j in 0..n {
            let want: f32 = (0..k).map(|kk| ra[i * k + kk] * rw[kk * n + j]).sum();
            let have = got[i * n + j];
            // A NaN output is a mismatch: `diff > tol` alone is false for it.
            let diff = (have - want).abs();
            assert!(
                !diff.is_nan() && diff <= want.abs() * 0.02 + 1e-2,
                "[{i},{j}]: want {want:.4}, got {have:.4}"
            );
        }
    }
}
