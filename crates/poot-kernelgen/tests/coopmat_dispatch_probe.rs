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
use poot_kernelgen as kg;
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
        eprintln!("llc not on PATH; skipping the coopmat dispatch probe");
        return;
    }
    let artifact = compile_kernel(&body, poot_codegen::GpuTarget::SpirvVulkan, &out)
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
        eprintln!("spirv-val not on PATH; skipping the validity check");
    }

    // (2) the real dispatch half: run the compiled coopmat tile on the actual Vulkan adapter and
    // compare against the CPU oracle.
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch half");
            return;
        }
    };
    let kernel = poot_codegen::kernel_handle(&body, &artifact).unwrap();
    let a_data: Vec<f32> = (0..256).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let b_data: Vec<f32> = (0..256).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
    let a_buf = ctx.alloc_storage(BufferRole::Input, BufferStorage::f16(), 256);
    write_host_bytes(&ctx, &a_buf, 0, &f16_encode(&a_data)).unwrap();
    let b_buf = ctx.alloc_storage(BufferRole::Input, BufferStorage::f16(), 256);
    write_host_bytes(&ctx, &b_buf, 0, &f16_encode(&b_data)).unwrap();
    let out_buf = ctx.alloc_storage(BufferRole::Output, BufferStorage::f32(), 256);
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

/// Multi-tile case: `matmul_tensorcore_coopmat` (the kernel `poot-graph-plan`'s `Backend::SpirvVulkan` coopmat
/// arm selects, vs `wmma_tile`'s single-tile fixture above) on a 32x32x16 shape (2x2 = 4 independent
/// output tiles, K == 16 exactly). Checks the M/N tile-index decode (`block = tid/32`, `tn`/`tm`) across multiple
/// subgroups, not just the single-tile case.
#[test]
fn coopmat_multi_tile_matmul_dispatches_and_matches_cpu_on_radv() {
    let (m, k, n) = (32usize, 16usize, 32usize);
    let body = kg::matmul_tensorcore_coopmat("k", &[m, n], &[m, k], &[k, n])
        .expect("matmul_tensorcore_coopmat precondition");

    let dir: PathBuf = std::env::temp_dir().join("poot-coopmat-dispatch-probe");
    std::fs::create_dir_all(&dir).unwrap();
    let out = artifact_path(&dir, "coopmat_multi_tile", Target::SpirvVulkan);
    if !which("llc") {
        eprintln!("llc not on PATH; skipping the coopmat multi-tile dispatch probe");
        return;
    }
    let artifact = compile_kernel(&body, poot_codegen::GpuTarget::SpirvVulkan, &out)
        .expect("coopmat multi-tile matmul must compile to SPIR-V");
    if which("spirv-val") {
        let v = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&out)
            .output()
            .expect("run spirv-val");
        assert!(
            v.status.success(),
            "spirv-val failed on the coopmat multi-tile matmul kernel:\n{}",
            String::from_utf8_lossy(&v.stderr)
        );
    } else {
        eprintln!("spirv-val not on PATH; skipping the validity check");
    }

    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch half");
            return;
        }
    };
    let kernel = poot_codegen::kernel_handle(&body, &artifact).unwrap();
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.04).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 19) as f32 - 9.0) * 0.04).collect();
    let a_buf = ctx.alloc_storage(BufferRole::Input, BufferStorage::f16(), m * k);
    write_host_bytes(&ctx, &a_buf, 0, &f16_encode(&a_data)).unwrap();
    let b_buf = ctx.alloc_storage(BufferRole::Input, BufferStorage::f16(), k * n);
    write_host_bytes(&ctx, &b_buf, 0, &f16_encode(&b_data)).unwrap();
    let out_buf = ctx.alloc_storage(BufferRole::Output, BufferStorage::f32(), m * n);
    let tiles_m = m / 16;
    let tiles_n = n / 16;
    let num_tiles = tiles_m * tiles_n;
    let threads = [(num_tiles * 32) as u32, 1, 1];
    ctx.dispatch_dev(
        "k",
        &kernel,
        body.workgroup_size,
        threads,
        &[&a_buf, &b_buf],
        &out_buf,
    )
    .expect("real RADV coopmat multi-tile dispatch");
    let mut got_bytes = vec![0u8; m * n * 4];
    ctx.read_bytes(&out_buf, &mut got_bytes).unwrap();
    let got: Vec<f32> = got_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let round = |x: f32| poot_quant::scalar::f16_to_f32(poot_quant::scalar::f32_to_f16(x));
    let ra: Vec<f32> = a_data.iter().map(|&x| round(x)).collect();
    let rb: Vec<f32> = b_data.iter().map(|&x| round(x)).collect();
    let mut max_abs = 0.0f32;
    let mut first_bad = None;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += ra[i * k + kk] * rb[kk * n + j];
            }
            let g = got[i * n + j];
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
            "coopmat multi-tile matmul MISMATCH at [{i},{j}]: want {want:.4} got {g:.4} (max_abs={max_abs:.3e})"
        );
    }
    eprintln!(
        "coopmat multi-tile matmul [{m}x{k}]@[{k}x{n}] ({num_tiles} tiles) OK on RADV cooperative matrix (max_abs={max_abs:.3e})"
    );
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
