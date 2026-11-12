//! `Rvalue::SubgroupShuffleXor` lowers on the three GPU targets and is refused by name on the AIE core: NVPTX to
//! `shfl.sync.bfly`, AMDGPU to `ds_bpermute` at the lane id xor the mask (wave32 and wave64), SPIR-V to
//! `OpGroupNonUniformShuffle` with the `GroupNonUniformShuffle` capability declared. What each lowering exchanges is
//! checked on the devices (`subgroup_shuffle_xor_matches_the_interpreter_on_*`); these rows pin the shape of the
//! lowering and that the modules assemble and validate.

use std::path::PathBuf;
use std::process::Command;

use poot_codegen::{Target, capability, compile, emit_llvm_ir};
use poot_kernel_ir::test_support::subgroup_shuffle_xor_kernel;
use poot_kernel_ir::{BlockId, Constant, Local, Operand, Place, Rvalue, Statement, Terminator, Ty};
use poot_target::AmdArch;
use poot_test_util::toolchain::{require_amd_clang, require_tools};

const WORKGROUP: u32 = 128;

fn out_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("poot-codegen-test").join(tag);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn count(text: &str, needle: &str) -> usize {
    text.matches(needle).count()
}

fn kernel(ty: Ty) -> poot_kernel_ir::Body {
    subgroup_shuffle_xor_kernel(ty, WORKGROUP)
}

fn masks() -> usize {
    poot_kernel_ir::test_support::SHUFFLE_MASKS.len()
}

#[test]
fn the_gpu_targets_declare_the_shuffle_and_the_aie_core_does_not() {
    for target in [
        Target::Nvptx,
        Target::SpirvVulkan,
        Target::AmdGcn(AmdArch::gfx1151()),
    ] {
        assert!(capability(target).subgroup_shuffle, "{target:?}");
    }
    assert!(!capability(Target::AieCore).subgroup_shuffle);
    // The AIE core refuses a thread index by name too, so drop it: the body then reaches the shuffle itself.
    let mut body = kernel(Ty::U32);
    body.blocks[0].statements.push(Statement::Assign(
        Place::local(Local { index: 2 }),
        Rvalue::Use(Operand::Const(Constant::Usize(0))),
    ));
    body.blocks[0].terminator = Terminator::Goto {
        target: BlockId { index: 1 },
    };
    let error = emit_llvm_ir(&body, Target::AieCore).unwrap_err();
    assert!(error.to_string().contains("SubgroupShuffleXor"), "{error}");
}

#[test]
fn nvptx_ir_butterflies_with_the_mask_and_an_f32_payload_travels_as_bits() {
    let u32_ir = emit_llvm_ir(&kernel(Ty::U32), Target::Nvptx).unwrap();
    assert_eq!(
        count(&u32_ir, "@llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 "),
        masks(),
        "{u32_ir}"
    );
    for mask in poot_kernel_ir::test_support::SHUFFLE_MASKS {
        assert!(
            u32_ir.contains(&format!(", i32 {mask}, i32 31)")),
            "mask {mask}\n{u32_ir}"
        );
    }
    let f32_ir = emit_llvm_ir(&kernel(Ty::F32), Target::Nvptx).unwrap();
    assert_eq!(count(&f32_ir, "bitcast float"), masks(), "{f32_ir}");
    assert_eq!(count(&f32_ir, "bitcast i32"), masks(), "{f32_ir}");
}

#[test]
fn nvptx_ptx_holds_shfl_sync_bfly() {
    require_tools(&["llc"]);
    let dir = out_dir("subgroup-shuffle-nvptx");
    let body = kernel(Ty::U32);
    let out = poot_codegen::artifact_path(&dir, &body.name, Target::Nvptx);
    compile(&body, Target::Nvptx, &out).unwrap();
    let ptx = std::fs::read_to_string(&out).unwrap();
    assert_eq!(count(&ptx, "shfl.sync.bfly.b32"), masks(), "{ptx}");
}

#[test]
fn the_spirv_module_validates_with_the_shuffle_capability_and_one_shuffle_per_mask() {
    require_tools(&["llc", "spirv-val", "spirv-dis"]);
    let dir = out_dir("subgroup-shuffle-spirv");
    for ty in [Ty::U32, Ty::F32] {
        let body = kernel(ty);
        let out = poot_codegen::artifact_path(&dir, &body.name, Target::SpirvVulkan);
        compile(&body, Target::SpirvVulkan, &out).unwrap();
        let val = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            val.status.success(),
            "spirv-val failed for {}:\n{}",
            body.name,
            String::from_utf8_lossy(&val.stderr)
        );
        let dis = Command::new("spirv-dis").arg(&out).output().unwrap();
        let text = String::from_utf8_lossy(&dis.stdout);
        assert!(
            text.contains("OpCapability GroupNonUniformShuffle"),
            "{text}"
        );
        assert_eq!(count(&text, "OpGroupNonUniformShuffle "), masks(), "{text}");
    }
}

/// Assembly of `body` for `arch` through the AMD clang, to read the instruction the IR selected.
fn amdgcn_asm(body: &poot_kernel_ir::Body, arch: AmdArch, dir: &std::path::Path) -> String {
    let clang = require_amd_clang();
    let target = Target::AmdGcn(arch);
    let ir_path = dir.join(format!("{}-{}.ll", body.name, arch.mcpu()));
    std::fs::write(&ir_path, emit_llvm_ir(body, target).unwrap()).unwrap();
    let asm = Command::new(clang)
        .args(["-target", "amdgcn-amd-amdhsa"])
        .arg(format!("-mcpu={}", arch.mcpu()))
        .args(["-O2", "-x", "ir", "-S", "-o", "-"])
        .arg(&ir_path)
        .output()
        .unwrap();
    assert!(
        asm.status.success(),
        "{}",
        String::from_utf8_lossy(&asm.stderr)
    );
    String::from_utf8(asm.stdout).unwrap()
}

#[test]
fn amdgcn_permutes_at_the_wave_lane_xor_the_mask_on_wave32_and_wave64() {
    let dir = out_dir("subgroup-shuffle-amdgcn");
    let body = kernel(Ty::U32);
    let wave32 = AmdArch::gfx1151();
    let wave64 = AmdArch::new("gfx942", 64);
    let ir32 = emit_llvm_ir(&body, Target::AmdGcn(wave32)).unwrap();
    let ir64 = emit_llvm_ir(&body, Target::AmdGcn(wave64)).unwrap();
    assert_eq!(
        count(&ir32, "@llvm.amdgcn.ds.bpermute(i32 "),
        masks(),
        "{ir32}"
    );
    assert_eq!(
        count(&ir32, "@llvm.amdgcn.mbcnt.hi"),
        0,
        "wave32 counts lanes with mbcnt.lo only\n{ir32}"
    );
    assert!(
        ir64.contains("@llvm.amdgcn.mbcnt.hi(i32 -1, i32 "),
        "{ir64}"
    );
    for (arch, ir) in [(wave32, &ir32), (wave64, &ir64)] {
        let asm = amdgcn_asm(&body, arch, &dir);
        assert!(
            count(&asm, "ds_bpermute_b32") >= 1 && !ir.is_empty(),
            "{}\n{asm}",
            arch.mcpu()
        );
        let hi = count(&asm, "v_mbcnt_hi_u32_b32");
        assert_eq!(
            hi > 0,
            arch.wave == 64,
            "{}: mbcnt_hi in the assembly\n{asm}",
            arch.mcpu()
        );
    }
}

#[test]
fn the_amdgcn_hsaco_assembles() {
    require_amd_clang();
    let dir = out_dir("subgroup-shuffle-amdgcn-hsaco");
    let target = Target::AmdGcn(AmdArch::gfx1151());
    for ty in [Ty::U32, Ty::F32] {
        let body = kernel(ty);
        let out = poot_codegen::artifact_path(&dir, &body.name, target);
        compile(&body, target, &out).unwrap();
        assert!(std::fs::metadata(&out).unwrap().len() > 0);
    }
}
