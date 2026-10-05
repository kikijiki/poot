//! `poot-vulkan-check`: P1 smoke-check binary (spec 133 Verification, "Decomposition (P1)").
//!
//! `--add` compiles poot's `add` kernel to `.spv` via the `poot-codegen` `Target::SpirvVulkan` path,
//! dispatches it through the raw-Vulkan (`ash`) runtime on the local Vulkan device, and checks the
//! result against the CPU oracle (`a + b`) and, if a wgpu adapter is available, against wgpu
//! bit-for-bit. Exits 0 on PASS or an explicit skip when no Vulkan device is present, non-zero on a
//! mismatch or Vulkan error.

use std::path::PathBuf;

use poot_codegen::{Target, compile, kernel_handle};
use poot_kernel_ir::fixtures;
use poot_runtime_common::BufferRole;
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

fn compiled_add(body: &poot_kernel_ir::Body, name: &str) -> poot_runtime_common::CompiledKernel {
    let dir: PathBuf = std::env::temp_dir().join("poot-vulkan-check");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let out = poot_codegen::artifact_path(&dir, name, Target::SpirvVulkan);
    compile(body, Target::SpirvVulkan, &out).expect("compile add kernel to SPIR-V");
    let bytes = std::fs::read(&out).expect("read compiled .spv");
    kernel_handle(body, Target::SpirvVulkan, bytes)
}

fn run_add() -> i32 {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            println!("SKIP: no Vulkan device available ({e})");
            return 0;
        }
    };
    println!("Vulkan device: {}", ctx.device_name());

    let body = fixtures::add_kernel();
    let kernel = compiled_add(&body, "add");
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

    let da = upload_f32(&ctx, BufferRole::Input, &a);
    let db = upload_f32(&ctx, BufferRole::Input, &b);
    let dc = ctx
        .alloc_storage(BufferRole::Output, BufferStorage::f32(), a.len())
        .expect("alloc_storage");

    let device_time = ctx
        .dispatch(
            &kernel,
            [64, 1, 1],
            [a.len() as u32, 1, 1],
            &[&da, &db, &dc],
        )
        .expect("dispatch add kernel");
    if let Some(t) = device_time {
        println!("device dispatch time: {t:?}");
    }

    let got = read_f32(&dc);
    let want: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    if got != want {
        println!("FAIL: raw-vulkan add result {got:?} != CPU oracle {want:?}");
        return 1;
    }
    println!("PASS: raw-vulkan add matches the CPU oracle ({got:?})");

    match poot_runtime::Context::new() {
        Ok(wgpu_ctx) => {
            let mut bufs = [
                poot_runtime::KernelBuffer::read_only_f32(&a),
                poot_runtime::KernelBuffer::read_only_f32(&b),
                poot_runtime::KernelBuffer::write_f32(a.len()),
            ];
            // Same `.spv` bytes, recompiled into a wgpu-side handle: the ABI is identical across the two
            // runtimes (module doc), so this is still a same-artifact cross-check.
            let wgpu_kernel = compiled_add(&body, "add");
            wgpu_ctx
                .dispatch(
                    "add",
                    &wgpu_kernel,
                    [64, 1, 1],
                    [a.len() as u32, 1, 1],
                    &mut bufs,
                )
                .expect("wgpu dispatch");
            if got != bufs[2].as_f32() {
                println!(
                    "FAIL: raw-vulkan add result {got:?} != wgpu result {:?}",
                    bufs[2].as_f32()
                );
                return 1;
            }
            println!("PASS: raw-vulkan add matches wgpu bit-for-bit (same .spv, same ABI)");
        }
        Err(e) => println!("(wgpu cross-check skipped: no adapter - {e})"),
    }

    0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = if args.iter().any(|a| a == "--add") {
        run_add()
    } else {
        eprintln!("usage: poot-vulkan-check --add");
        2
    };
    std::process::exit(code);
}
