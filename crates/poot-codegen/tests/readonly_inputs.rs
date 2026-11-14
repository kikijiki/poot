//! A `&[T]` kernel parameter is a read-only buffer on SpirvVulkan, a `&mut [T]` one a writable buffer (POOT-1575).
//! The emitter declares the resource `(..., 12, 0)` for the first and `(..., 12, 1)` for the second, which the
//! SPIR-V backend lowers to a `NonWritable` decoration, so WebGPU's rule on overlapping writable bindings never
//! applies to a dispatch's inputs.

use std::path::PathBuf;
use std::process::Command;

use poot_codegen::{Target, compile, emit_llvm_ir};
use poot_kernel_ir::test_support::u32_vector_load_kernel;
use poot_test_util::toolchain::require_tools;

fn out_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("poot-codegen-test").join(tag);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The `handlefrombinding` call that binds slot `bind`: its result type carries the writable bit.
fn handle_for(ir: &str, bind: u32) -> &str {
    let tail = format!("(i32 0, i32 {bind}, i32 1, i32 0,");
    ir.lines()
        .find(|line| line.contains("handlefrombinding") && line.contains(&tail))
        .unwrap_or_else(|| panic!("no handle for binding {bind}:\n{ir}"))
}

#[test]
fn a_shared_slice_param_is_declared_read_only_and_a_mutable_one_writable() {
    let ir = emit_llvm_ir(&u32_vector_load_kernel(), Target::SpirvVulkan).unwrap();
    let input = handle_for(&ir, 0);
    let output = handle_for(&ir, 1);
    assert!(
        input.contains(", 12, 0)"),
        "input must be read-only: {input}"
    );
    assert!(
        output.contains(", 12, 1)"),
        "output must be writable: {output}"
    );
}

#[test]
fn the_compiled_module_decorates_only_the_input_non_writable() {
    require_tools(&["llc", "spirv-val", "spirv-dis"]);
    let dir = out_dir("readonly-inputs-spirv");
    let out = poot_codegen::artifact_path(&dir, "readonly_inputs", Target::SpirvVulkan);
    compile(&u32_vector_load_kernel(), Target::SpirvVulkan, &out).expect("SpirvVulkan compile");
    let val = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        val.status.success(),
        "spirv-val failed:\n{}",
        String::from_utf8_lossy(&val.stderr)
    );
    let dis = Command::new("spirv-dis").arg(&out).output().unwrap();
    let text = String::from_utf8_lossy(&dis.stdout);
    let non_writable = text.matches("NonWritable").count();
    // The input buffer and the length buffer are read-only; the output buffer is not.
    assert_eq!(non_writable, 2, "NonWritable decorations:\n{text}");
}
