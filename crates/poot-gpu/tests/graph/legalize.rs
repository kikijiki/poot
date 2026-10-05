//! Card 523a: `compile`'s `legalize` pass's split outcome, driven
//! through `compile` (the real graph-to-program entry point) and executed on real wgpu hardware, not a
//! bare `legalize` call and not a CPU-only check. The host outcome's real-hardware evidence is
//! `poot-llm/tests/coherence.rs`'s `qwen2_7b_host_embed_serves_on_wgpu` (a real qwen2.5-7b model through
//! the actual production path, `Runner::generate_kv_gpu_packed`); split has no production model yet
//! (a restored capability, same as before this card touched it), so this drives
//! `compile` directly on a small synthetic oversized weight instead.

use super::*;
use poot_graph_plan::{CompileOptions, FusionPolicy, Submission, Target, compile};
use poot_target::{Backend, DeviceCaps};

#[test]
fn compile_split_matmul_matches_cpu_on_real_wgpu_hardware() {
    let _gpu_guard = gpu_lock();
    let Some((mut exec, target)) = open_engine_or_skip() else {
        return;
    };

    let (k, n) = (4usize, 10usize);
    let b = Builder::new();
    // `x` is a broadcast of one real scalar, not a declared `[1, k]` constant: legalize's split rewrite
    // never looks at the activation's storage, only the weight's.
    let x0 = b.constant("x", TensorType::f32(vec![1, 1]));
    let x = b.broadcast(x0, vec![1, k]);
    // The tracers' `linear`: an `[out, in]` weight read through a transpose.
    let w = b.constant("w", TensorType::f32(vec![n, k]));
    let out = b.matmul(x, b.transpose(w, vec![1, 0]));
    let g = b.finish(out);

    // row_bytes = k*4 = 16 bytes; a 48-byte limit (well under this device's real max_buffer_bytes, and
    // below 4096 so `usable == limit`) forces 3 rows per chunk: row ranges (0,3),(3,6),(6,9),(9,10) - 4
    // chunks, including an uneven final one.
    let split_target = Target {
        backend: Backend::SpirvVulkan,
        caps: DeviceCaps {
            max_buffer_bytes: 48,
            ..target.caps
        },
    };
    let options = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    let program =
        compile(&g, &split_target, &options).expect("an oversized matmul weight always splits");
    let gh = program.graph();
    assert!(
        gh.eqns
            .iter()
            .any(|e| matches!(e.op, OpKind::Concat { .. })),
        "the compiled program must show the split outcome (chunks joined by concat)"
    );

    let w_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01 - 0.05).collect();
    let mut cpu_inputs = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        let t = match m.name.as_deref() {
            Some("w") => HostTensor::f32(vec![n, k], w_data.clone()),
            Some("x") => HostTensor::f32(vec![1, 1], vec![1.0]),
            other => panic!("unexpected const {other:?}"),
        };
        cpu_inputs.insert(id, Value::from(t));
    }
    let cpu = eval(&g, &cpu_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    // The device binds the compiled program's `w.chunkN` consts from the one parent weight `w`, stored
    // whole under its own name: the binder places chunk N as the next rows of `w`.
    let gh_chunks: Vec<_> = gh
        .consts
        .iter()
        .filter_map(|&id| gh.meta(id).name.as_deref())
        .filter(|name| name.starts_with("w.chunk"))
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
            name: "w",
            tensor: HostTensor::f32(vec![n, k], w_data),
        },
    ];
    let got = poot_executor_parity::run_once(&mut exec, target, gh, &consts, &[])
        .unwrap_or_else(|e| panic!("{e}"));

    assert_close(got.as_f32().unwrap(), cpu.as_f32().unwrap(), 1e-5);
}
