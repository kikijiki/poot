//! SC-001, the large-grid row: a dispatch wider than wgpu's 65535-workgroup grid runs as the one 1-D
//! dispatch the device's measured `DeviceCaps::max_grid` allows, and matches the CPU oracle.
//!
//! Flash prefill over `Hq * L = 512 * 130 = 66560` query groups (the staged `FlashAttentionPrefill`, as a
//! pass would form it). The planner reads `caps.max_grid[0]` to choose between one 1-D launch and the 2-D
//! fold, so this row holds only while the caps are the physical limit: it asserts the program really
//! dispatches more than 65535 workgroups on x, then compares the output to `eval` of the same graph.
//!
//! The refusal side (a grid over the physical limit is `GridCap`, never folded) is the mock row
//! `a_grid_over_the_device_limit_is_refused_not_folded` in `poot-vulkan-runtime`: this device's own x limit
//! is `u32::MAX`, so no launch here can exceed it.

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::{ConstFixture, run_once};
use poot_graph_ir::OpKind;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::types::TensorType;
use poot_graph_plan::{CompileLimits, CompileOptions, FusionPolicy, Plan, Submission, compile};
use poot_runtime_common::DeviceBackend;
use poot_tensor::HostTensor;
use poot_test_util::device_skip::open_or_skip;
use poot_vulkan_device::VulkanDevice;
use std::collections::HashMap;

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 1.2 - 0.6
        })
        .collect()
}

#[test]
fn flash_prefill_over_65535_workgroups_is_one_wide_dispatch_and_matches_the_oracle_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
        return;
    };
    let target = device.target();
    let (hq, l, d) = (512usize, 130usize, 4usize);
    let scale = 0.3f32;
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hq, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hq, l, d]));
    let mask = b.constant("m", TensorType::f32(vec![1, 1, l, l]));
    let ids = [q.id, k.id, v.id, mask.id];
    let mut plan = b.append_plan(0);
    let o = plan
        .equation(
            OpKind::FlashAttentionPrefill {
                n_rep: 1,
                scale,
                softcap: None,
            },
            ids.map(poot_graph_ir::Operand::Value).to_vec(),
        )
        .expect("stage the flash prefill");
    plan.declare_result(o).expect("declare the result");
    let mut prepared = b.preflight_append(plan).expect("preflight");
    let id = b.commit_append(&mut prepared).expect("commit");
    let graph = b.finish(poot_graph_ir::Traced { id });

    // The planner trusts the device's x grid: one dispatch of more than 65535 workgroups.
    let program = compile(
        &graph,
        &target,
        &CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::Full,
            limits: CompileLimits::STANDARD,
        },
    )
    .expect("the flash prefill compiles for this device");
    let widest = program
        .planned()
        .filter_map(|(_, plan)| match plan {
            Plan::Compute { body, grid, .. } | Plan::ComputeMeta { body, grid, .. } => {
                Some(grid[0].div_ceil(body.workgroup_size[0].max(1)))
            }
            _ => None,
        })
        .max()
        .expect("the program dispatches a kernel");
    assert!(
        widest > 65_535,
        "the widest dispatch is {widest} workgroups: the row needs one past wgpu's x grid"
    );

    let mask_data: Vec<f32> = (0..l * l)
        .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
        .collect();
    let tensors = [
        HostTensor::f32(vec![1, hq, l, d], fill(1, hq * l * d)),
        HostTensor::f32(vec![1, hq, l, d], fill(2, hq * l * d)),
        HostTensor::f32(vec![1, hq, l, d], fill(3, hq * l * d)),
        HostTensor::f32(vec![1, 1, l, l], mask_data),
    ];
    let inputs: HashMap<_, Value> = ids
        .iter()
        .zip(&tensors)
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let want = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let [q_t, k_t, v_t, m_t] = tensors;
    let consts = [
        ConstFixture {
            name: "q",
            tensor: q_t,
        },
        ConstFixture {
            name: "k",
            tensor: k_t,
        },
        ConstFixture {
            name: "v",
            tensor: v_t,
        },
        ConstFixture {
            name: "m",
            tensor: m_t,
        },
    ];
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let got = run_once(&mut *exec, target, &graph, &consts, &[])
        .unwrap_or_else(|e| panic!("the wide flash prefill runs through the contract: {e}"));
    assert_eq!(got.shape(), vec![1, hq, l, d]);
    let worst = got
        .as_f32()
        .unwrap()
        .iter()
        .zip(want.as_f32().unwrap())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |worst, diff| {
            if diff.is_nan() {
                f32::INFINITY
            } else {
                worst.max(diff)
            }
        });
    assert!(
        worst <= 2e-3,
        "the wide dispatch differs from the oracle by {worst:.3e}"
    );
}
