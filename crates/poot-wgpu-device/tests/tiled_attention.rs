//! The tiled (FlashAttention-2) prefill attention on RADV with workgroups wider than one wave64 (POOT-1402).
//!
//! The RADV wave64 LDS defect (update 0169) lost the workgroup-memory writes of every wave past the first: a
//! 256-lane LDS-tiled GEMM read stale upper-wave LDS after the barrier. POOT-1403 found it does not reproduce on
//! Strix Halo RADV (Mesa 26.1.3) and deleted the flag that kept every RADV workgroup to 64 lanes, so the tiled
//! attention now plans its widest tiling there too. Every lane of its workgroup stages a share of each query,
//! key and value block into workgroup memory, so a wave whose writes were lost would leave holes in the tiles
//! every other lane reads.
//!
//! Mutation (the defect's own shape, applied once, observed red, restored): in `attention/tiled.rs`, stage zeros
//! from the lanes past the first wave64 (`stage_strided`, which stages every query, key and value tile); this row
//! goes red, the 256-lane kernel 3.361e-1 from the oracle.

use poot_eval::{EvalOptions, Value, eval};
use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::{ConstFixture, run_once};
use poot_graph_ir::OpKind;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::attention_prefill_softcap;
use poot_graph_ir::types::TensorType;
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, compile};
use poot_kernelgen as kg;
use poot_kernelgen::KernelRequest;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_tensor::HostTensor;
use poot_wgpu_device::device::WgpuDevice;
use std::collections::HashMap;

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// A causal prefill of 8 heads (one per key/value head) over 256 rows of width 64 plans, on the RADV caps, the
/// widest tiling: 256 lanes (four waves) over 32-row tiles, 64 row groups for the device's 40 compute units,
/// and a key block much shorter than the 256 keys, so every row walks several staged blocks. It matches the CPU
/// oracle.
#[test]
fn the_tiled_attention_runs_workgroups_wider_than_one_wave64_on_radv() {
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let (hq, hkv, l, d) = (8usize, 8usize, 256usize, 64usize);
    let b = Builder::for_tests();
    let operand = |name, heads| {
        b.constant(name, TensorType::f32(vec![1, heads, l, d]))
            .unwrap()
    };
    let (q, k, v) = (operand("q", hq), operand("k", hkv), operand("v", hkv));
    let m = b.constant("m", TensorType::f32(vec![1, 1, l, l])).unwrap();
    let scale = 1.0 / (d as f32).sqrt();
    let out = attention_prefill_softcap(&b, q, k, v, hq / hkv, scale, m, None).unwrap();
    let graph = b.finish(out).unwrap();
    let mask = (0..l * l)
        .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
        .collect();
    let consts = [
        ("q", vec![1, hq, l, d], fill(1, hq * l * d)),
        ("k", vec![1, hkv, l, d], fill(2, hkv * l * d)),
        ("v", vec![1, hkv, l, d], fill(3, hkv * l * d)),
        ("m", vec![1, 1, l, l], mask),
    ]
    .map(|(name, shape, data)| ConstFixture {
        name,
        tensor: HostTensor::f32(shape, data),
    });
    let ids = [q.id(), k.id(), v.id(), m.id()];

    let target = device.target();
    let program = compile(
        &graph,
        &target,
        &CompileOptions::new(CompileLimits::STANDARD),
    )
    .expect("the attention graph compiles for wgpu");
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::Attention(_)))
        .expect("the chain fuses to one attention equation");
    let KernelChoice::Generated(KernelRequest::Attention(spec)) = program.kernel_choice(eqn) else {
        panic!("expected the generated attention kernel");
    };
    let kg::AttentionSchedule::Tiled { lanes, bc, .. } = spec.schedule else {
        panic!(
            "a long prefill plans the tiled schedule, got {:?}",
            spec.schedule
        );
    };
    assert!(
        lanes > 64,
        "{lanes} lanes: the workgroup is wider than one wave64"
    );
    assert!(l > bc, "{l} keys walk more than one {bc}-key block");

    let inputs: HashMap<_, Value> = ids
        .iter()
        .zip(&consts)
        .map(|(&id, c)| (id, Value::from(c.tensor.clone())))
        .collect();
    let want = eval(&graph, &inputs, EvalOptions::default())
        .expect("the oracle evaluates the graph")
        .output
        .into_host()
        .unwrap();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let got = run_once(&mut *exec, target, &graph, &consts, &[])
        .expect("the attention runs through the contract");
    assert_eq!(got.shape(), want.shape());
    let worst = poot_test_util::max_abs_error(got.as_f32().unwrap(), want.as_f32().unwrap());
    assert!(
        worst <= 1e-3,
        "{lanes}-lane tiled attention differs from the oracle by {worst:.3e}"
    );
}
