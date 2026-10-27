//! `Rvalue::ConstantTableRead` lowers to a real constant on every GPU target: an initialized module
//! global in the target's constant space (`addrspace(4)` on AMDGCN, `addrspace(1)` on NVPTX,
//! UniformConstant `addrspace(2)` on OpenCL SPIR-V, the Private storage class `addrspace(10)` on
//! Vulkan SPIR-V), indexed dynamically and loaded as an `i32`. The probe kernel is
//! `out[i] = table[i]`; each text check is followed by a compile through the real toolchain where
//! one is installed, and the Vulkan
//! module must pass `spirv-val`. Skips (passes) when the relevant tool is absent, like the other
//! codegen integration tests.

use std::path::PathBuf;
use std::process::Command;

use poot_codegen::{Target, compile, emit_llvm_ir};
use poot_kernel_ir::{
    BasicBlock, BlockId, Body, Constant, ConstantTableDecl, IndexAxis, Local, LocalDecl, Operand,
    Place, ProjectionElem, Rvalue, Statement, Terminator, Ty,
};

const WORDS: [u32; 4] = [0x0807_0605, 0xdead_beef, 0x8000_0001, 7];

fn local(index: u32) -> Local {
    Local { index }
}

/// `out[i] = table[i]` over four lanes: `_1` is `out: &mut [u32]`, `_2` the lane, `_3` the word.
fn probe() -> Body {
    let decl = |ty| LocalDecl { ty, mutable: true };
    let copy = |index| Operand::Copy(Place::local(local(index)));
    let locals = vec![
        decl(Ty::Unit),
        decl(Ty::Ref {
            mutable: true,
            pointee: Box::new(Ty::Slice(Box::new(Ty::U32))),
        }),
        decl(Ty::Usize),
        decl(Ty::U32),
    ];
    let blocks = vec![
        BasicBlock {
            statements: vec![],
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(local(2)),
                dim: IndexAxis::X,
                target: BlockId { index: 1 },
            },
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(local(3)),
                    Rvalue::ConstantTableRead {
                        idx: copy(2),
                        table: 0,
                    },
                ),
                Statement::Assign(
                    Place {
                        local: local(1),
                        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(local(2))],
                    },
                    Rvalue::Use(copy(3)),
                ),
            ],
            terminator: Terminator::Return,
        },
    ];
    let mut body = Body::new("table_probe", 1, locals, blocks);
    body.workgroup_size = [4, 1, 1];
    body.constant_tables = vec![ConstantTableDecl {
        words: WORDS.to_vec(),
    }];
    body
}

fn have(bin: &str) -> bool {
    Command::new(bin).arg("--version").output().is_ok()
}

fn out_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("poot-codegen-test").join(tag);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The initialized global, in the right space, with the table's words in order (an `i32` is the
/// word's two's-complement value), and the load through it.
fn assert_table(ir: &str, space: u32, kind: &str) {
    let words: Vec<String> = WORDS
        .iter()
        .map(|&word| format!("i32 {}", word as i32))
        .collect();
    let global = format!(
        "@table_probe_ct0 = internal unnamed_addr addrspace({space}) {kind} [4 x i32] [{}], align 16",
        words.join(", ")
    );
    assert!(ir.contains(&global), "missing `{global}` in:\n{ir}");
    assert!(
        ir.contains(&format!(
            "getelementptr inbounds [4 x i32], ptr addrspace({space}) @table_probe_ct0"
        )),
        "the read does not index the table in:\n{ir}"
    );
    assert!(
        ir.contains(&format!("load i32, ptr addrspace({space})")),
        "the read does not load an i32 from the table's space in:\n{ir}"
    );
}

#[test]
fn every_target_declares_the_table_in_its_constant_space() {
    let amd = Target::AmdGcn(poot_target::AmdArch::gfx1151());
    for (target, space, kind) in [
        (Target::SpirvVulkan, 10, "global"),
        (Target::Nvptx, 1, "constant"),
        (amd, 4, "constant"),
    ] {
        let ir = emit_llvm_ir(&probe(), target).unwrap_or_else(|e| panic!("{target:?}: {e}"));
        assert_table(&ir, space, kind);
    }
}

/// An AIE core has no module constants a kernel can index: the table is a typed refusal there, not a
/// silently wrong kernel.
#[test]
fn aie_core_refuses_a_constant_table() {
    // A scalar kernel (no thread index, which an AIE core refuses first): `out[0] = table[0]`.
    let mut body = probe();
    body.blocks = vec![BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(local(2)),
                Rvalue::Use(Operand::Const(Constant::Usize(0))),
            ),
            body.blocks[1].statements[0].clone(),
            body.blocks[1].statements[1].clone(),
        ],
        terminator: Terminator::Return,
    }];
    let error = emit_llvm_ir(&body, Target::AieCore).unwrap_err();
    assert!(
        error.to_string().contains("constant tables on AIE-core"),
        "{error}"
    );
}

#[test]
fn the_vulkan_module_validates() {
    if !have("llc") || !have("spirv-val") {
        eprintln!("llc/spirv-val not on PATH; skipping");
        return;
    }
    let dir = out_dir("constant-table-spirv");
    let out = poot_codegen::artifact_path(&dir, "table_probe", Target::SpirvVulkan);
    compile(&probe(), Target::SpirvVulkan, &out).expect("SpirvVulkan compile");
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
}

#[test]
fn the_nvptx_module_reads_the_table_from_global_memory() {
    if !have("llc") {
        eprintln!("llc not on PATH; skipping");
        return;
    }
    let dir = out_dir("constant-table-nvptx");
    let out = poot_codegen::artifact_path(&dir, "table_probe", Target::Nvptx);
    compile(&probe(), Target::Nvptx, &out).expect("Nvptx compile");
    let ptx = std::fs::read_to_string(&out).unwrap();
    // Sixteen initialized bytes (the words, little-endian), read back with a parallel global load: a
    // `.const` read replays once per distinct lane address (see `constant_table_addrspace`).
    assert!(
        ptx.contains(".global .align 16 .b8 table_probe_ct0[16] = {5, 6, 7, 8, 239, 190, 173, 222, 1, 0, 0, 128, 7"),
        "the table is not an initialized .global in:\n{ptx}"
    );
    assert!(ptx.contains("ld.global.nc.b32"), "{ptx}");
    assert!(!ptx.contains("ld.const"), "{ptx}");
}

#[test]
fn the_amdgcn_module_compiles() {
    if !std::env::var("POOT_AMD_CLANG")
        .is_ok_and(|clang| Command::new(clang).arg("--version").output().is_ok())
    {
        eprintln!("POOT_AMD_CLANG not set or not runnable; skipping");
        return;
    }
    let dir = out_dir("constant-table-amdgcn");
    let target = Target::AmdGcn(poot_target::AmdArch::gfx1151());
    let out = poot_codegen::artifact_path(&dir, "table_probe", target);
    compile(&probe(), target, &out).expect("AmdGcn compile");
    assert!(std::fs::metadata(&out).unwrap().len() > 0);
}
