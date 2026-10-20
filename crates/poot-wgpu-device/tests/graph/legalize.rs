//! Card 523a: `compile`'s `legalize` pass's split outcome, driven
//! through `compile` (the real graph-to-program entry point) and executed on real wgpu hardware, not a
//! bare `legalize` call and not a CPU-only check. The host outcome's real-hardware evidence is
//! `poot-llm/tests/coherence.rs`'s `qwen2_7b_host_embed_serves_on_wgpu` (a real qwen2.5-7b model through
//! the actual production path, the driver's packed device generate); split has no production model yet
//! (a restored capability, same as before this card touched it), so this drives
//! `compile` directly on a small synthetic oversized weight instead.

use super::*;
use poot_graph_ir::WeightRef;
use poot_graph_plan::test_support::gpu_caps;
use poot_graph_plan::{CompileOptions, Target, compile};
use poot_target::{Backend, DeviceCaps};
use poot_weights::{WeightId, WeightRole};

#[test]
fn compile_split_matmul_matches_cpu_on_real_wgpu_hardware() {
    let _gpu_guard = gpu_lock();
    let Some((mut exec, target)) = open_engine_or_skip() else {
        return;
    };

    let (k, n) = (4usize, 10usize);
    let b = Builder::for_tests();
    // `x` is a broadcast of one real scalar, not a declared `[1, k]` constant: legalize's split rewrite
    // never looks at the activation's storage, only the weight's.
    let x0 = b.constant("x", TensorType::f32(vec![1, 1])).unwrap();
    let x = b.broadcast(x0, vec![1, k]).unwrap();
    // The tracers' `linear`: an `[out, in]` weight read through a transpose.
    let head = WeightId::model(WeightRole::Head);
    let w = b
        .weight(WeightRef::whole(head), TensorType::f32(vec![n, k]))
        .unwrap();
    let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    let g = b.finish(out).unwrap();

    // row_bytes = k*4 = 16 bytes; a 48-byte limit (well under this device's real max_buffer_bytes, and
    // below 4096 so `usable == limit`) forces 3 rows per chunk: row ranges (0,3),(3,6),(6,9),(9,10) - 4
    // chunks, including an uneven final one.
    let split_target = Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: DeviceCaps {
            max_buffer_bytes: 48,
            ..gpu_caps(&target)
        },
    };
    let options = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
    let program =
        compile(&g, &split_target, &options).expect("an oversized matmul weight always splits");
    let gh = program.graph();
    assert!(
        gh.eqns()
            .iter()
            .any(|e| matches!(e.op, OpKind::Concat { .. })),
        "the compiled program must show the split outcome (chunks joined by concat)"
    );

    let w_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01 - 0.05).collect();
    let mut cpu_inputs = HashMap::new();
    for &id in g.inputs() {
        let m = g.meta(id);
        let t = match m.name.as_deref() {
            Some("w.head") => HostTensor::f32(vec![n, k], w_data.clone()),
            Some("x") => HostTensor::f32(vec![1, 1], vec![1.0]),
            other => panic!("unexpected const {other:?}"),
        };
        cpu_inputs.insert(id, Value::from(t));
    }
    let cpu = eval(&g, &cpu_inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();

    // The device binds the compiled program's `w.head.chunkN` consts from the one parent weight `w.head`,
    // stored whole under its own name: the binder places chunk N as the next rows of it.
    let gh_chunks: Vec<_> = gh
        .consts()
        .iter()
        .filter_map(|&id| gh.meta(id).name.as_deref())
        .filter(|name| name.starts_with("w.head.chunk"))
        .collect();
    assert_eq!(
        gh_chunks.len(),
        4,
        "the split program reads four chunk consts"
    );
    let consts = [
        poot_executor_parity::ConstFixture {
            name: "x",
            tensor: HostTensor::f32(vec![1, 1], vec![1.0]),
        },
        poot_executor_parity::ConstFixture {
            name: "w.head",
            tensor: HostTensor::f32(vec![n, k], w_data),
        },
    ];
    let got = poot_executor_parity::run_once(&mut exec, target, gh, &consts, &[])
        .unwrap_or_else(|e| panic!("{e}"));

    assert_close(got.as_f32().unwrap(), cpu.as_f32().unwrap(), 1e-5);
}
