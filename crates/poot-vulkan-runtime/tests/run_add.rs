//! P1 acceptance (spec 133 Verification, "Decomposition (P1)"): poot-kernel-ir Body -> poot-codegen
//! SPIR-V (`Target::SpirvVulkan`) -> raw-Vulkan (`ash`) dispatch, checked against the CPU oracle. The
//! required raw-Vulkan lane opens no second backend: inherited wgpu configuration or requirement state
//! must be irrelevant once raw Vulkan opens. Skips (passes) if no Vulkan device is available, unless
//! the required lane selected it.

use std::path::PathBuf;

use poot_codegen::{Target, compile, kernel_handle};
use poot_kernel_ir::fixtures;
use poot_runtime_common::{BufferRole, CompiledKernel};
use poot_test_util::kernel_fixtures::square_kernel;
use poot_vulkan_runtime::{BufferStorage, Context, DeviceBuffer};

/// Upload `data` as f32 through the storage-typed [`Context::alloc_storage`]/[`DeviceBuffer::write_bytes`]
/// primitives (Card 547a: this crate has no `upload_f32`/`alloc_f32`/`read_f32` family, since it has no
/// consumer outside its own tests).
fn upload_f32(ctx: &Context, role: BufferRole, data: &[f32]) -> DeviceBuffer {
    let buf = ctx
        .alloc_storage(role, BufferStorage::f32(), data.len())
        .expect("alloc_storage");
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    buf.write_bytes(&bytes).expect("write_bytes");
    buf
}

fn read_f32(buf: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0u8; buf.elem_count() as usize * 4];
    buf.read_bytes(&mut bytes).expect("read_bytes");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn kernel_for(body: &poot_kernel_ir::Body, name: &str) -> CompiledKernel {
    let dir: PathBuf = std::env::temp_dir().join("poot-vulkan-runtime-test");
    std::fs::create_dir_all(&dir).unwrap();
    let out = poot_codegen::artifact_path(&dir, name, Target::SpirvVulkan);
    compile(body, Target::SpirvVulkan, &out).expect("compile spirv");
    let spv = std::fs::read(&out).unwrap();
    kernel_handle(body, Target::SpirvVulkan, spv)
}

#[test]
fn add_runs_on_gpu_via_raw_vulkan() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device ({e}); skipping");
            return;
        }
    };
    eprintln!("raw-vulkan device: {}", ctx.device_name());

    let kernel = kernel_for(&fixtures::add_kernel(), "add");
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];

    let da = upload_f32(&ctx, BufferRole::Input, &a);
    let db = upload_f32(&ctx, BufferRole::Input, &b);
    let dc = ctx
        .alloc_storage(BufferRole::Output, BufferStorage::f32(), a.len())
        .expect("alloc_storage");

    ctx.dispatch(
        &kernel,
        [64, 1, 1],
        [a.len() as u32, 1, 1],
        &[&da, &db, &dc],
    )
    .expect("dispatch");

    let got = read_f32(&dc);
    let want_cpu: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    assert_eq!(got, want_cpu, "raw-vulkan add vs CPU oracle");
}

#[test]
fn square_runs_on_gpu_via_raw_vulkan() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device ({e}); skipping");
            return;
        }
    };
    let kernel = kernel_for(&square_kernel(), "square");
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dx = upload_f32(&ctx, BufferRole::Input, &x);
    let dy = ctx
        .alloc_storage(BufferRole::Output, BufferStorage::f32(), x.len())
        .expect("alloc_storage");
    ctx.dispatch(&kernel, [64, 1, 1], [x.len() as u32, 1, 1], &[&dx, &dy])
        .expect("dispatch");
    let want: Vec<f32> = x.iter().map(|v| v * v).collect();
    assert_eq!(read_f32(&dy), want);
}
