//! End-to-end: poot-kernel-ir Body -> poot-codegen SPIR-V -> wgpu dispatch on the Arc, checking the
//! result against the CPU computation (the GPU-path executor-equivalence anchor). Skips (passes) if no
//! Vulkan adapter is available.

use std::path::PathBuf;

use poot_codegen::{Target, compile, kernel_handle};
use poot_kernel_ir::fixtures;
use poot_runtime::{CompiledKernel, Context, KernelBuffer};
use poot_test_util::kernel_fixtures::{matmul_kernel, square_kernel};

fn spv_for(body: &poot_kernel_ir::Body, name: &str) -> CompiledKernel {
    let dir: PathBuf = std::env::temp_dir().join("poot-runtime-test");
    std::fs::create_dir_all(&dir).unwrap();
    let out = poot_codegen::artifact_path(&dir, name, Target::SpirvVulkan);
    compile(body, Target::SpirvVulkan, &out).expect("compile spirv");
    let bytes = std::fs::read(&out).unwrap();
    kernel_handle(body, Target::SpirvVulkan, bytes)
}

#[test]
fn add_runs_on_gpu() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let spv = spv_for(&fixtures::add_kernel(), "add");
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&b),
        KernelBuffer::write_f32(a.len()),
    ];
    ctx.dispatch("test", &spv, [64, 1, 1], [a.len() as u32, 1, 1], &mut bufs)
        .expect("dispatch");
    assert_eq!(bufs[2].as_f32(), &[11.0, 22.0, 33.0, 44.0, 55.0]);
}

#[test]
fn square_runs_on_gpu() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let spv = spv_for(&square_kernel(), "square");
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut bufs = [
        KernelBuffer::read_only_f32(&x),
        KernelBuffer::write_f32(x.len()),
    ];
    ctx.dispatch("test", &spv, [64, 1, 1], [x.len() as u32, 1, 1], &mut bufs)
        .expect("dispatch");
    assert_eq!(bufs[1].as_f32(), &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
}

#[test]
fn matmul_runs_on_gpu() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let (m, n, k) = (2usize, 3, 4);
    let spv = spv_for(&matmul_kernel(m, n, k), "matmul");
    let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&b),
        KernelBuffer::write_f32(m * n),
    ];
    // grid: x = N columns, y = M rows; workgroup 16x16 matches the kernel LocalSize.
    ctx.dispatch(
        "test",
        &spv,
        [16, 16, 1],
        [n as u32, m as u32, 1],
        &mut bufs,
    )
    .expect("dispatch");

    let mut want = vec![0.0f32; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += a[r * k + kk] * b[kk * n + col];
            }
            want[r * n + col] = acc;
        }
    }
    assert_eq!(bufs[2].as_f32(), &want[..], "matmul mismatch");
}

// `profiled_dispatch_records_stats` lives in `src/tests.rs` (`profiled_dispatch_tests`), not here: it
// drives `Context::timestamps_available`, `pub(crate)` only, so an external `tests/` binary cannot call it.

#[test]
fn add_device_resident() {
    use poot_kernel_ir::fixtures;
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let spv = spv_for(&fixtures::add_kernel(), "add");
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let da = ctx.upload_f32(&a);
    let db = ctx.upload_f32(&b);
    let dc = ctx.alloc_f32(a.len());
    // chain: keep results on the device, download only at the end.
    ctx.dispatch_dev(
        "test",
        &spv,
        [64, 1, 1],
        [a.len() as u32, 1, 1],
        &[&da, &db],
        &dc,
    )
    .unwrap();
    assert_eq!(
        ctx.download_f32(&dc).unwrap(),
        vec![11.0, 22.0, 33.0, 44.0, 55.0]
    );
}
