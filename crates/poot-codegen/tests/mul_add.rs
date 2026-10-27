//! `Rvalue::MulAdd` lowers to the target's fused multiply-add on every GPU target, and every kernel states the
//! workgroup it is launched with (`amdgpu-flat-work-group-size` on AMDGCN, `nvvm.maxntid` on NVPTX), so the
//! register allocator budgets for that workgroup and not the 1024-lane default (POOT-1403: an 8x4 register
//! tile spilled 402 VGPRs on gfx1151 under the default). Skips (passes) when the toolchain is absent, as
//! `no_contract.rs` does.

use std::path::PathBuf;
use std::process::Command;

use poot_codegen::{Target, compile, emit_llvm_ir};
use poot_kernel_ir::test_support::mul_add_probe_kernel;

fn have(bin: &str) -> bool {
    Command::new(bin).arg("--version").output().is_ok()
}

fn amd_toolchain_available() -> bool {
    std::env::var("POOT_AMD_CLANG")
        .is_ok_and(|clang| Command::new(clang).arg("--version").output().is_ok())
}

fn out_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join("poot-codegen-test").join(tag);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The fused probe emits one `llvm.fma.f32` call on every GPU target and the unfused one none, so a
/// multiply-add is never left to a backend's contraction default. Mutation: lower `Rvalue::MulAdd` as a plain
/// `fmul` then `fadd` in `emit_mul_add`; no `llvm.fma` call is emitted and this row goes red.
#[test]
fn a_mul_add_emits_one_fused_intrinsic_on_every_gpu_target() {
    for target in [
        Target::SpirvVulkan,
        Target::Nvptx,
        Target::AmdGcn(poot_target::AmdArch::gfx1151()),
    ] {
        for fused in [true, false] {
            let ir = emit_llvm_ir(&mul_add_probe_kernel(fused), target)
                .unwrap_or_else(|e| panic!("{target:?}: {e}"));
            let calls = ir.matches("call float @llvm.fma.f32(").count();
            assert_eq!(calls, usize::from(fused), "{target:?} fused={fused}:\n{ir}");
        }
    }
}

/// The compiled kernels carry the fused instruction: `fma.rn.f32` in the PTX, an `Fma` extended instruction
/// in valid Vulkan SPIR-V, and a `v_fma`/`v_fmac` in the gfx1151 ISA.
#[test]
fn a_mul_add_compiles_to_the_targets_fma_instruction() {
    let body = mul_add_probe_kernel(true);
    let dir = out_dir("mul-add");
    if have("llc") {
        let out = poot_codegen::artifact_path(&dir, "mul_add", Target::Nvptx);
        compile(&body, Target::Nvptx, &out).expect("NVPTX compile");
        let ptx = std::fs::read_to_string(&out).unwrap();
        assert!(ptx.contains("fma.rn.f32"), "{ptx}");
    } else {
        eprintln!("llc not on PATH; skipping the NVPTX half");
    }
    if have("llc") && have("spirv-val") && have("spirv-dis") {
        let out = poot_codegen::artifact_path(&dir, "mul_add", Target::SpirvVulkan);
        compile(&body, Target::SpirvVulkan, &out).expect("SPIR-V compile");
        let val = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            val.status.success(),
            "spirv-val:\n{}",
            String::from_utf8_lossy(&val.stderr)
        );
        let dis = Command::new("spirv-dis").arg(&out).output().unwrap();
        let text = String::from_utf8_lossy(&dis.stdout);
        assert!(text.contains("Fma"), "{text}");
    } else {
        eprintln!("llc/spirv-val/spirv-dis not on PATH; skipping the SPIR-V half");
    }
    if amd_toolchain_available() {
        let target = Target::AmdGcn(poot_target::AmdArch::gfx1151());
        let out = poot_codegen::artifact_path(&dir, "mul_add", target);
        compile(&body, target, &out).expect("AMDGCN compile");
        assert!(std::fs::metadata(&out).unwrap().len() > 0);
    } else {
        eprintln!("POOT_AMD_CLANG not set; skipping the AMDGCN half");
    }
}

/// Every kernel states the largest workgroup it runs: the body's own workgroup size, as
/// `"amdgpu-flat-work-group-size"="1,N"` on AMDGCN and `"nvvm.maxntid"="X,Y,Z"` on NVPTX (which `llc` turns into
/// the PTX `.maxntid` directive). Mutation: drop the attribute from `emit_signature`'s AMDGCN arm; the IR no
/// longer states the bound and this row goes red.
#[test]
fn a_kernel_states_its_workgroup_to_the_register_allocator() {
    let mut body = mul_add_probe_kernel(true);
    body.workgroup_size = [128, 1, 1];
    let amd = emit_llvm_ir(&body, Target::AmdGcn(poot_target::AmdArch::gfx1151())).unwrap();
    assert!(
        amd.contains("\"amdgpu-flat-work-group-size\"=\"1,128\""),
        "{amd}"
    );
    let nvptx = emit_llvm_ir(&body, Target::Nvptx).unwrap();
    assert!(nvptx.contains("\"nvvm.maxntid\"=\"128,1,1\""), "{nvptx}");
    if have("llc") {
        let out = poot_codegen::artifact_path(&out_dir("maxntid"), "maxntid", Target::Nvptx);
        compile(&body, Target::Nvptx, &out).expect("NVPTX compile");
        let ptx = std::fs::read_to_string(&out).unwrap();
        assert!(ptx.contains(".maxntid 128, 1, 1"), "{ptx}");
    }
}
