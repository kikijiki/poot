//! Correctness gate for a whole-graph F16 matmul (F16 operands, F16 output): (1) it compiles to valid SPIR-V
//! and passes `spirv-val` on RADV, and (2) its GPU dispatch matches the CPU f32 oracle within f16 tolerance.
//! [`poot_runtime::KernelBuffer`] is raw, dtype-agnostic bytes, so an f16-encoded (2 bytes/elem) buffer
//! dispatches through the same `Context::dispatch` every other kernelgen probe uses (see `tests/run.rs`'s
//! `binary_mul`). `matmul_batched_dt(Ty::F16, ...)` is the dtype-generic serial matmul, the kernel a whole-graph
//! F16 matmul equation lowers to once the `is_decode_gemv`/`is_tiled_gemm`/`is_tileable_matmul` f32-only fast
//! paths are excluded (poot-graph-plan / poot-graph-ir transform tests cover that exclusion).
//!
//! Skips (passes) if no Vulkan adapter or `spirv-val` is absent, same convention as every other probe.

use std::path::PathBuf;

use poot_codegen::{Target, artifact_path, compile};
use poot_kernel_ir::Ty;
use poot_kernelgen as kg;
use poot_runtime::{BufferRole, BufferStorage, Context};

fn which(tool: &str) -> bool {
    std::process::Command::new(tool)
        .arg("--version")
        .output()
        .is_ok()
}

/// Encode f32 -> f16 bytes (little-endian), reusing the loader's own f32->f16 narrowing
/// (`poot_load::gguf::f32_to_f16`) so this probe's precision loss matches what a real f16 checkpoint
/// tensor's bytes would already look like.
fn f16_encode(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&f| poot_load::gguf::f32_to_f16(f).to_le_bytes())
        .collect()
}

/// Decode f16 bytes (little-endian) back to f32, reusing the loader's `f16_to_f32` (the exact IEEE-754
/// half decode the loader uses for native f16 checkpoint bytes).
fn f16_decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| poot_quant::scalar::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// A plain row-major CPU reference matmul: `out[M,N] = a[M,K] @ b[K,N]`.
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[test]
fn f16_matmul_compiles_valid_spirv_and_matches_cpu_on_radv() {
    let (m, k, n) = (4usize, 8usize, 6usize);
    let out_shape = [m, n];
    let a_shape = [m, k];
    let b_shape = [k, n];
    let body = kg::matmul_batched_dt_grid("f16mm", Ty::F16, &out_shape, &a_shape, &b_shape, None);

    // (1) SC-002 half 1: F16 matmul lowers to valid SPIR-V and passes spirv-val on RADV. Build-time
    // only, no GPU hardware needed - every other kernelgen probe in this crate follows this pattern.
    let dir: PathBuf = std::env::temp_dir().join("poot-f16-matmul-probe");
    std::fs::create_dir_all(&dir).unwrap();
    let out = artifact_path(&dir, "f16mm", Target::SpirvVulkan);
    compile(&body, Target::SpirvVulkan, &out).expect("F16 matmul must compile to SPIR-V");
    if which("spirv-val") {
        let v = std::process::Command::new("spirv-val")
            .arg(&out)
            .output()
            .expect("run spirv-val");
        assert!(
            v.status.success(),
            "spirv-val failed on the F16 matmul kernel:\n{}",
            String::from_utf8_lossy(&v.stderr)
        );
    } else {
        eprintln!("spirv-val not on PATH; skipping the validity check");
    }

    // (2) SC-002 half 2: the GPU dispatch matches the CPU f32 oracle within f16 tolerance. This is
    // executor-equivalence, real dispatch - the part that needs an actual Vulkan adapter.
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch half");
            return;
        }
    };
    let spv_bytes = std::fs::read(&out).unwrap();
    let kernel = poot_codegen::kernel_handle(&body, Target::SpirvVulkan, spv_bytes);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.2).collect();
    let a_buf = ctx.alloc_storage(BufferRole::Input, BufferStorage::f16(), m * k);
    ctx.write_bytes(&a_buf, &f16_encode(&a_data)).unwrap();
    let b_buf = ctx.alloc_storage(BufferRole::Input, BufferStorage::f16(), k * n);
    ctx.write_bytes(&b_buf, &f16_encode(&b_data)).unwrap();
    let out_buf = ctx.alloc_storage(BufferRole::Output, BufferStorage::f16(), m * n);
    ctx.dispatch_dev(
        "f16mm",
        &kernel,
        [64, 1, 1],
        [(m * n) as u32, 1, 1],
        &[&a_buf, &b_buf],
        &out_buf,
    )
    .unwrap();
    let mut out_bytes = vec![0u8; 2 * m * n];
    ctx.read_bytes(&out_buf, &mut out_bytes).unwrap();
    let got = f16_decode(&out_bytes);

    // CPU oracle: f32 matmul, narrowed to f16 - the same precision loss a real f16-output kernel
    // incurs on store (mirrors poot-eval's BF16/F16 matmul narrowing, spec 024/135).
    let want_f32 = cpu_matmul(&a_data, &b_data, m, k, n);
    let want: Vec<f32> = want_f32
        .iter()
        .map(|&x| poot_quant::scalar::f16_to_f32(poot_load::gguf::f32_to_f16(x)))
        .collect();

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        // f16 has a 10-bit mantissa (~3 decimal digits); loose but real tolerance for this shape/range.
        let tol = 1e-2 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "elem {i}: gpu {g} vs cpu(f16-rounded) {w}"
        );
    }
}
