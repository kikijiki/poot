//! First RADV dispatch of `wmma_tile` (card 530: the target-neutral single 16x16x16 fragment tile, F16
//! operands / F32 accumulate, lowered here to RADV coopmat config 13), compiled through the real
//! `fix_coopmat_calls` post-process path (wired into `compile()`) and dispatched on the gfx1151 iGPU via
//! `poot_wgpu_runtime::Context::dispatch`, the same raw-KernelBuffer path `f16_matmul_probe.rs` uses. Only the
//! coopmat kernel body and oracle are new.
//!
//! Skips (passes) if no Vulkan adapter or `spirv-val` is absent, same convention as every other probe in this crate.

use std::path::PathBuf;

use poot_codegen::{Target, artifact_path, compile_kernel};
use poot_kernel_ir::test_support::wmma_tile;
use poot_runtime_common::BufferRole;
use poot_target::BufferStorage;
use poot_wgpu_runtime::Context;

fn which(tool: &str) -> bool {
    std::process::Command::new(tool)
        .arg("--version")
        .output()
        .is_ok()
}

/// Encode f32 -> f16 bytes (little-endian), reusing the loader's own f32->f16 narrowing
/// (`poot_quant::scalar::f32_to_f16`), matching `f16_matmul_probe.rs`'s helper.
fn f16_encode(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&f| poot_quant::scalar::f32_to_f16(f).to_le_bytes())
        .collect()
}

#[test]
fn coopmat_tile_dispatches_and_matches_cpu_on_radv() {
    let body = wmma_tile("k");

    // (1) compiles to valid SPIR-V (real `llc` + `fix_coopmat_calls`) and passes spirv-val on RADV.
    let dir: PathBuf = std::env::temp_dir().join("poot-coopmat-dispatch-probe");
    std::fs::create_dir_all(&dir).unwrap();
    let out = artifact_path(&dir, "coopmat_tile", Target::SpirvVulkan);
    if !which("llc") {
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
        return;
    }
    let artifact = compile_kernel(&body, poot_target::Backend::SpirvVulkan, &out)
        .expect("coopmat tile must compile to SPIR-V");
    if which("spirv-val") {
        let v = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&out)
            .output()
            .expect("run spirv-val");
        assert!(
            v.status.success(),
            "spirv-val failed on the coopmat tile kernel:\n{}",
            String::from_utf8_lossy(&v.stderr)
        );
    } else {
        poot_test_util::skip_report("poot-kernelgen", &"spirv-val not on PATH");
    }

    // (2) the real dispatch half: run the compiled coopmat tile on the actual Vulkan adapter and
    // compare against the CPU oracle.
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            poot_test_util::skip_report("poot-kernelgen", &format_args!("no wgpu context: {e}"));
            return;
        }
    };
    let kernel = poot_codegen::kernel_handle(&body, &artifact).unwrap();
    let a_data: Vec<f32> = (0..256).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let b_data: Vec<f32> = (0..256).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
    let a_buf = ctx
        .alloc_storage(BufferRole::Input, BufferStorage::f16(), 256)
        .expect("alloc");
    write_host_bytes(&ctx, &a_buf, 0, &f16_encode(&a_data)).unwrap();
    let b_buf = ctx
        .alloc_storage(BufferRole::Input, BufferStorage::f16(), 256)
        .expect("alloc");
    write_host_bytes(&ctx, &b_buf, 0, &f16_encode(&b_data)).unwrap();
    let out_buf = ctx
        .alloc_storage(BufferRole::Output, BufferStorage::f32(), 256)
        .expect("alloc");
    ctx.dispatch_dev(
        "k",
        &kernel,
        body.workgroup_size,
        [32, 1, 1],
        &[&a_buf, &b_buf],
        &out_buf,
    )
    .expect("real RADV coopmat dispatch");
    let mut got_bytes = vec![0u8; 256 * 4];
    ctx.read_bytes(&out_buf, &mut got_bytes).unwrap();
    let got: Vec<f32> = got_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // CPU oracle: round A/B to f16 (the coopmat load narrows on the way in), f32-accumulate,
    // k=0..16 - matching the NVPTX WMMA sibling probe's oracle shape (`probe_tc_matmul`).
    let round = |x: f32| poot_quant::scalar::f16_to_f32(poot_quant::scalar::f32_to_f16(x));
    let ra: Vec<f32> = a_data.iter().map(|&x| round(x)).collect();
    let rb: Vec<f32> = b_data.iter().map(|&x| round(x)).collect();
    let mut max_abs = 0.0f32;
    let mut first_bad = None;
    for i in 0..16 {
        for j in 0..16 {
            let mut acc = 0.0f32;
            for kk in 0..16 {
                acc += ra[i * 16 + kk] * rb[kk * 16 + j];
            }
            let g = got[i * 16 + j];
            let d = (g - acc).abs();
            max_abs = max_abs.max(d);
            let tol = acc.abs() * 0.02 + 1e-2;
            // `d > tol` is false for a NaN `d`, so a NaN result would pass without the explicit check.
            if (d.is_nan() || d > tol) && first_bad.is_none() {
                first_bad = Some((i, j, acc, g));
            }
        }
    }
    if let Some((i, j, want, g)) = first_bad {
        panic!(
            "coopmat tile MISMATCH at [{i},{j}]: want {want:.4} got {g:.4} (max_abs={max_abs:.3e})"
        );
    }
    eprintln!("coopmat tile [16x16]@[16x16] OK on RADV cooperative matrix (max_abs={max_abs:.3e})");
}

/// Write `bytes` at `offset` of `buf` through the context's one write primitive, `write_with` (a test helper:
/// the contexts have no slice-taking write of their own).
fn write_host_bytes(
    ctx: &poot_wgpu_runtime::Context,
    buf: &poot_wgpu_runtime::DeviceBuffer,
    offset: usize,
    bytes: &[u8],
) -> Result<(), poot_wgpu_runtime::RuntimeError> {
    ctx.write_with(buf, offset, bytes.len(), &mut |range, out| {
        out.copy_from_slice(&bytes[range.start - offset..range.end - offset]);
    })
}
