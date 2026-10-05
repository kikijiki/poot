//! Criterion benchmarks for per-op GPU dispatch time (card 035).
//!
//! Measures end-to-end dispatch latency (compiled/cached dispatch + download) for key GPU
//! operations, through the executor contract: one `Executor::add_entry` per shape, then
//! `Executor::step` repeatedly inside the timed loop (the replay path - compilation happens once,
//! outside the timing). Results are stored by criterion for cross-run comparison.
//!
//! Run: `cargo bench -p poot-gpu --bench op_dispatch` (inside `nix develop`).
//! Save baseline: `cargo bench -p poot-gpu --bench op_dispatch -- --save-baseline main`
//! Compare:       `cargo bench -p poot-gpu --bench op_dispatch -- --baseline main`
//!
//! Skips silently when no Vulkan adapter is available.

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use poot_executor::{Device as _, EntryId, ExecutableId, Executor as _, NoSync, StepInputs};
use poot_gpu::device::WgpuDevice;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, RedOp, UnOp};
use poot_graph_ir::types::TensorType;
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    Submission, TargetSet, compile_staged,
};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::DType;

fn fill(seed: u64, count: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
    (0..count)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Loads `consts` onto a fresh `WgpuDevice`, compiles `g` and adds one entry; returns the live
/// engine so the caller can `step` it repeatedly (the compiled/cached replay path this benchmark
/// measures) before tearing the entry down.
fn setup(
    g: &poot_graph_ir::Graph,
    consts: &[(&'static str, Vec<usize>, Vec<f32>)],
) -> Option<(poot_executor::Engine<WgpuDevice>, ExecutableId, EntryId)> {
    let device = WgpuDevice::new().ok()?;
    let target = device.target();
    let mut engine = poot_executor::Engine::new(device);
    let mut builder = WeightStore::builder();
    for (name, shape, values) in consts {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense = DenseWeight::try_new(DType::F32, shape.clone(), Arc::from(bytes)).ok()?;
        builder.insert(*name, WeightEntry::Dense(dense)).ok()?;
    }
    let exe = engine
        .load_weights(
            Arc::new(builder.build()),
            poot_executor::WeightSource::ConstNames,
        )
        .ok()?;
    let g = g.clone().with_validations(Vec::new());
    let staged = compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .ok()?;
    let entry = engine.add_entry(exe, &staged).ok()?;
    Some((engine, exe, entry))
}

fn teardown(engine: &mut poot_executor::Engine<WgpuDevice>, exe: ExecutableId, entry: EntryId) {
    let _ = engine.remove_entry(exe, entry);
    let _ = engine.unload(exe);
}

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise");
    for &n in &[32usize, 128, 512, 1024] {
        let b = Builder::new();
        let a = b.constant("a", TensorType::f32(vec![n]));
        let c_val = b.constant("c", TensorType::f32(vec![n]));
        let out = b.binary(BinOp::Add, a, c_val);
        let g = b.finish(out);

        let consts = vec![("a", vec![n], fill(42, n)), ("c", vec![n], fill(99, n))];
        let Some((mut engine, exe, entry)) = setup(&g, &consts) else {
            eprintln!("no GPU; skipping op_dispatch benchmarks");
            return;
        };

        group.bench_with_input(BenchmarkId::new("add", n), &n, |bench, &_n| {
            bench.iter(|| {
                let result = engine
                    .step(exe, entry, black_box(&StepInputs::new()), &mut NoSync)
                    .unwrap()
                    .read()
                    .unwrap();
                black_box(&result);
            });
        });
        teardown(&mut engine, exe, entry);
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    let shapes = [
        (4usize, 8usize, 4usize),
        (8, 16, 8),
        (16, 32, 16),
        (32, 64, 32),
    ];
    for &(m, k, n) in &shapes {
        let b = Builder::new();
        let a = b.constant("a", TensorType::f32(vec![m, k]));
        let w = b.constant("w", TensorType::f32(vec![k, n]));
        let out = b.matmul(a, w);
        let g = b.finish(out);

        let consts = vec![
            ("a", vec![m, k], fill(101, m * k)),
            ("w", vec![k, n], fill(102, k * n)),
        ];
        let Some((mut engine, exe, entry)) = setup(&g, &consts) else {
            return;
        };

        let label = format!("{m}x{k}x{n}");
        group.bench_with_input(BenchmarkId::new("matmul", &label), &label, |bench, _| {
            bench.iter(|| {
                let result = engine
                    .step(exe, entry, black_box(&StepInputs::new()), &mut NoSync)
                    .unwrap()
                    .read()
                    .unwrap();
                black_box(&result);
            });
        });
        teardown(&mut engine, exe, entry);
    }
    group.finish();
}

fn bench_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce");
    for &(rows, cols) in &[(4usize, 32usize), (4, 128), (4, 512)] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![rows, cols]));
        let out = b.reduce(RedOp::Sum, x, 1, false);
        let g = b.finish(out);

        let consts = vec![("x", vec![rows, cols], fill(201, rows * cols))];
        let Some((mut engine, exe, entry)) = setup(&g, &consts) else {
            return;
        };

        let label = format!("{rows}x{cols}");
        group.bench_with_input(BenchmarkId::new("sum", &label), &label, |bench, _| {
            bench.iter(|| {
                let result = engine
                    .step(exe, entry, black_box(&StepInputs::new()), &mut NoSync)
                    .unwrap()
                    .read()
                    .unwrap();
                black_box(&result);
            });
        });
        teardown(&mut engine, exe, entry);
    }
    group.finish();
}

fn bench_rmsnorm(c: &mut Criterion) {
    use poot_graph_ir::ops::rmsnorm;
    let mut group = c.benchmark_group("rmsnorm");
    for &n in &[32usize, 128, 512] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, n]));
        let w = b.constant("w", TensorType::f32(vec![n]));
        let out = rmsnorm(&b, x, w, 1e-6);
        let g = b.finish(out);

        let consts = vec![
            ("x", vec![1, n], fill(301, n)),
            (
                "w",
                vec![n],
                fill(302, n).iter().map(|v| 1.0 + v.abs()).collect(),
            ),
        ];
        let Some((mut engine, exe, entry)) = setup(&g, &consts) else {
            return;
        };

        group.bench_with_input(BenchmarkId::new("rmsnorm", n), &n, |bench, &_n| {
            bench.iter(|| {
                let result = engine
                    .step(exe, entry, black_box(&StepInputs::new()), &mut NoSync)
                    .unwrap()
                    .read()
                    .unwrap();
                black_box(&result);
            });
        });
        teardown(&mut engine, exe, entry);
    }
    group.finish();
}

fn bench_unary(c: &mut Criterion) {
    let mut group = c.benchmark_group("unary");
    for &n in &[64usize, 256, 1024] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![n]));
        let out = b.unary(UnOp::Exp, x);
        let g = b.finish(out);

        let consts = vec![("x", vec![n], fill(401, n))];
        let Some((mut engine, exe, entry)) = setup(&g, &consts) else {
            return;
        };

        group.bench_with_input(BenchmarkId::new("exp", n), &n, |bench, &_n| {
            bench.iter(|| {
                let result = engine
                    .step(exe, entry, black_box(&StepInputs::new()), &mut NoSync)
                    .unwrap()
                    .read()
                    .unwrap();
                black_box(&result);
            });
        });
        teardown(&mut engine, exe, entry);
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_elementwise,
    bench_matmul,
    bench_reduce,
    bench_rmsnorm,
    bench_unary,
);
criterion_main!(benches);
