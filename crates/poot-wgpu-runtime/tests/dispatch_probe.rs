//! Card 144 / spec 134 P0.75: real dispatch (not just static llc/spirv-val) of the
//! mixed-scalar+vec4-binding SPIR-V shapes from
//! `tests/fixtures/dispatch_probe/`, through `poot_wgpu_runtime::Context` on RADV
//! with readback checks. (Case 1, the vec4-load shape, lives in `src/tests.rs`'s
//! `vec4_probe_tests`: it needs `KernelBuffer::read_write_f32`, `pub(crate)` only.)
//!
//! The fixtures are hand-written raw LLVM IR (not `poot_kernel_ir::Body` +
//! `poot_codegen::emit_llvm_ir`), so the tests shell out to `llc` directly instead of `poot_codegen::compile`. A missing
//! tool fails the test (`poot_test_util::toolchain`); a missing Vulkan adapter is a `POOT-SKIP`, and a failure when
//! `POOT_REQUIRE_WGPU` requires it.
//!
//! Verify with `cargo test -p poot-wgpu-runtime --test dispatch_probe --no-run` (or `--release`); run the
//! dispatch tests only in the serial GPU lane (run command in the module docs at the bottom).

use std::path::PathBuf;
use std::process::Command;

use poot_codegen::{ArgSchema, CompiledKernel, KernelCode};
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_target::ArgAccess;
use poot_target::ElementKind;
use poot_test_util::toolchain::require_tools;
use poot_wgpu_runtime::{Context, DeviceBuffer};

/// Wrap hand-written raw-LLVM-IR SPIR-V (not `poot_kernel_ir::Body` + `poot_codegen`, see the module
/// doc) as a dispatchable [`CompiledKernel`] (card 608's sanctioned "imported kernel" escape hatch: these
/// fixtures are not compiler output, so this test builds the handle itself).
fn kernel_from_spv(spv: &[u8], args: Vec<ArgSchema>) -> CompiledKernel {
    let words: Vec<u32> = spv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // SAFETY: `spv` is `llc`/`spirv-val`-clean SPIR-V (checked by `compile_and_validate`) whose bindings
    // match `args` exactly, by construction of each fixture below.
    unsafe {
        CompiledKernel::new(
            poot_target::TargetId::Gpu(poot_target::Backend::SpirvVulkan),
            "main",
            KernelCode::SpirvWords(words.into_boxed_slice()),
            args,
            false,
        )
    }
}

fn probe_dir() -> PathBuf {
    let d = std::env::temp_dir().join("poot-wgpu-runtime-dispatch-probe");
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Compile raw SPIR-V-target LLVM IR text to a `.spv` via system `llc` (same flags as
/// `poot_codegen::Target::SpirvVulkan::llc_args()`), then validate it with `spirv-val`. Panics with the
/// tool's stderr on failure (a hard gate, not a skip; the caller already required `llc` and `spirv-val`).
fn compile_and_validate(ll_source: &str, tag: &str) -> Vec<u8> {
    let dir = probe_dir();
    let ll_path = dir.join(format!("{tag}.ll"));
    std::fs::write(&ll_path, ll_source).unwrap();
    let spv_path = dir.join(format!("{tag}.spv"));
    let llc = std::env::var("POOT_LLC").unwrap_or_else(|_| "llc".to_string());
    let out = Command::new(&llc)
        .args([
            "-O0",
            "-filetype=obj",
            "--spirv-ext=+SPV_EXT_shader_atomic_float_add",
        ])
        .arg(&ll_path)
        .arg("-o")
        .arg(&spv_path)
        .output()
        .expect("failed to run llc");
    assert!(
        out.status.success(),
        "llc failed for {tag}:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&spv_path).unwrap();
    let v = Command::new("spirv-val")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg(&spv_path)
        .output()
        .expect("failed to run spirv-val");
    assert!(
        v.status.success(),
        "spirv-val failed for {tag}:\n{}",
        String::from_utf8_lossy(&v.stderr)
    );
    bytes
}

/// Case 2, negative half: the same buffer bound at one binding index as both a scalar `[0 x float]`
/// and a vec4 `[0 x <4 x float>]` resource (`C1_mixed_scalar_vec4_one_binding_INVALID.ll`). Needs no
/// GPU: `llc` succeeds but the SPIR-V backend deduplicates same-binding `handlefrombinding` calls to
/// one `OpVariable` (keeping the first element type) and emits an illegal `OpCopyObject`
/// type-mismatch for the other, which `spirv-val` rejects. Regression guard: one binding index cannot
/// carry two differently-typed views through poot's llc pipeline. The dispatching fallback is
/// `mixed_scalar_vec4_two_bindings_alias_one_buffer_runs_on_gpu` below.
#[test]
fn mixed_scalar_vec4_one_binding_is_rejected_by_spirv_val() {
    require_tools(&["llc", "spirv-val"]);
    let ll = include_str!("fixtures/dispatch_probe/C1_mixed_scalar_vec4_one_binding_INVALID.ll");
    let dir = probe_dir();
    let ll_path = dir.join("c1_mixed_one_binding.ll");
    std::fs::write(&ll_path, ll).unwrap();
    let spv_path = dir.join("c1_mixed_one_binding.spv");
    let llc = std::env::var("POOT_LLC").unwrap_or_else(|_| "llc".to_string());
    let llc_out = Command::new(&llc)
        .args([
            "-O0",
            "-filetype=obj",
            "--spirv-ext=+SPV_EXT_shader_atomic_float_add",
        ])
        .arg(&ll_path)
        .arg("-o")
        .arg(&spv_path)
        .output()
        .expect("failed to run llc");
    assert!(
        llc_out.status.success(),
        "expected llc to SUCCEED (it emits invalid-but-well-formed SPIR-V here); got:\n{}",
        String::from_utf8_lossy(&llc_out.stderr)
    );
    let v = Command::new("spirv-val")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg(&spv_path)
        .output()
        .expect("failed to run spirv-val");
    assert!(
        !v.status.success(),
        "expected spirv-val to REJECT the one-binding mixed scalar+vec4 module (review S1 finding); \
         it unexpectedly validated - the finding may no longer hold, re-check the C1 fixture's comment"
    );
    let stderr = String::from_utf8_lossy(&v.stderr);
    assert!(
        stderr.contains("OpCopyObject") || stderr.contains("Result Type"),
        "expected the known OpCopyObject type-mismatch rejection, got a different error:\n{stderr}"
    );
}

/// Case 2, positive half (P1 fallback design): the same GPU buffer bound at two binding indices,
/// binding 0 as scalar `[0 x float]` and binding 1 as vec4 `[0 x <4 x float>]`
/// (`C2_mixed_scalar_vec4_two_bindings_WORKS.ll`), via `dispatch_dev` with the same `DeviceBuffer` as
/// both `ins` entries; the result goes to a separate `out` (binding 2), because an input slot is
/// read-only and WebGPU forbids a writable view of a buffer another binding reads. poot_wgpu_runtime's
/// binding API already supports one logical buffer with two typed views; only poot-codegen needs
/// per-param dtype-view emission for P1.
#[test]
fn mixed_scalar_vec4_two_bindings_alias_one_buffer_runs_on_gpu() {
    require_tools(&["llc", "spirv-val"]);
    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
        return;
    };
    let ll = include_str!("fixtures/dispatch_probe/C2_mixed_scalar_vec4_two_bindings_WORKS.ll");
    let spv = compile_and_validate(ll, "c2_mixed_two_bindings");

    // layout: [0..4) vec4 "body", [4] scalar "tail"; padded to 8 so the buffer is a vec4 multiple. The
    // result lands in a separate one-word output.
    let data = [10.0f32, 20.0, 30.0, 40.0, 50.0, 0.0, 0.0, 0.0];
    let shared: DeviceBuffer = ctx.upload_f32(&data);
    let result: DeviceBuffer = ctx.upload_f32(&[0.0]);
    let kernel = kernel_from_spv(
        &spv,
        vec![
            ArgSchema::new(ElementKind::F32, ArgAccess::Read),
            ArgSchema::new(ElementKind::F32, ArgAccess::Read),
            ArgSchema::new(ElementKind::F32, ArgAccess::Write),
        ],
    );
    ctx.dispatch_dev(
        "c2_mixed_two_bindings",
        &kernel,
        [1, 1, 1],
        [1, 1, 1],
        &[&shared, &shared], // bindings 0 (scalar view) and 1 (vec4 view): the SAME wgpu buffer, read-only
        &result,
    )
    .expect("dispatch");

    let out = ctx.download_f32(&result).expect("readback");
    let vsum = data[0] + data[1] + data[2] + data[3]; // 100
    let want_total = vsum + data[4]; // + tail (50) = 150
    assert_eq!(
        out[0], want_total,
        "a binding-1 vec4 read + a binding-0 scalar read: both views must see the same underlying buffer"
    );
}

/// POOT-1575 (W5 of POOT-1566): an input slot is declared read-only in the bind group layout, so one buffer
/// bound as an input and as the output of one dispatch is a wgpu validation error ("conflicting usages":
/// a read-only and an exclusive writable use of one buffer). Before, every data slot was declared writable
/// and the dispatch passed, which WebGPU's writable-aliasing rule forbids.
///
/// Mutation (red): declare every data slot writable again in `Context::dispatch_dev`'s layout
/// (`storage_entry(i, false)` in `Context::cached_pipeline`); the dispatch no longer panics.
#[test]
fn an_input_bound_as_the_output_of_one_dispatch_is_a_validation_error_on_gpu() {
    require_tools(&["llc", "spirv-val"]);
    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
        return;
    };
    let ll = include_str!("fixtures/dispatch_probe/C2_mixed_scalar_vec4_two_bindings_WORKS.ll");
    let spv = compile_and_validate(ll, "c2_input_as_output");
    let shared: DeviceBuffer = ctx.upload_f32(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 0.0, 0.0, 0.0]);
    let kernel = kernel_from_spv(
        &spv,
        vec![
            ArgSchema::new(ElementKind::F32, ArgAccess::Read),
            ArgSchema::new(ElementKind::F32, ArgAccess::Read),
            ArgSchema::new(ElementKind::F32, ArgAccess::Write),
        ],
    );
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ctx.dispatch_dev(
            "c2_input_as_output",
            &kernel,
            [1, 1, 1],
            [1, 1, 1],
            &[&shared, &shared],
            &shared,
        );
        let _ = ctx.download_f32(&shared);
    }))
    .expect_err("the dispatch must be refused by wgpu validation");
    let message = refused
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| refused.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default();
    assert!(message.contains("conflicting usages"), "{message}");
}
