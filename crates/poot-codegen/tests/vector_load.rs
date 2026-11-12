//! `Rvalue::VectorLoad` of integer lanes lowers on every vector-capable target from the alignment the emitter
//! proved. NVPTX and AMDGPU load `<lanes x i32>` through the param's own pointer, at the stamped alignment, next to
//! the scalar reads of the same param. SpirvVulkan declares a param that is only vector-loaded (with whole-vector
//! alignment) as a vector resource and reads any other param's lanes as scalars. The values each lowering reads are
//! checked on the devices (`u32x4_vector_load_matches_scalar_loads_on_*`); these rows pin the shape of the lowering.

use std::path::PathBuf;
use std::process::Command;

use poot_codegen::{Target, compile, emit_llvm_ir};
use poot_kernel_ir::test_support::{u32_vector_load_kernel, u32_vector_load_mixed_kernel};
use poot_kernel_ir::{Body, Rvalue, Statement};
use poot_test_util::toolchain::{require_amd_clang, require_tools};

fn amd() -> Target {
    Target::AmdGcn(poot_target::AmdArch::gfx1151())
}

fn out_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("poot-codegen-test").join(tag);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn ir(body: &Body, target: Target) -> String {
    emit_llvm_ir(body, target).unwrap_or_else(|e| panic!("{target:?}: {e}"))
}

/// `body` with every vector load claiming `align_bytes`.
fn claiming(mut body: Body, align_bytes: u32) -> Body {
    for statement in body.blocks.iter_mut().flat_map(|b| &mut b.statements) {
        if let Statement::Assign(_, Rvalue::VectorLoad { align_bytes: a, .. }) = statement {
            *a = align_bytes;
        }
    }
    body
}

fn count(text: &str, needle: &str) -> usize {
    text.matches(needle).count()
}

/// The element pointers taken from the scalar `i32` resource bound for param 0 (`%h1`), one per scalar read of it.
fn scalar_input_reads(ir: &str) -> usize {
    count(
        ir,
        "VulkanBuffer_a0i32_12_1t(target(\"spirv.VulkanBuffer\", [0 x i32], 12, 1) %h1,",
    )
}

#[test]
fn nvptx_and_amdgcn_load_the_vector_through_the_params_pointer_at_the_proven_alignment() {
    for target in [Target::Nvptx, amd()] {
        let wide = ir(&u32_vector_load_kernel(), target);
        assert_eq!(
            count(&wide, "load <4 x i32>, ptr addrspace(1)"),
            1,
            "{target:?}\n{wide}"
        );
        assert!(wide.contains(", align 16\n"), "{target:?}\n{wide}");

        // One param read three ways: a u32x4 (align 16), a u32x2 (align 8) and a scalar, all on `input`.
        let mixed = ir(&u32_vector_load_mixed_kernel(), target);
        assert_eq!(
            count(&mixed, "load <4 x i32>, ptr addrspace(1)"),
            1,
            "{mixed}"
        );
        assert_eq!(
            count(&mixed, "load <2 x i32>, ptr addrspace(1)"),
            1,
            "{mixed}"
        );
        assert!(
            mixed.contains("load <2 x i32>, ptr addrspace(1)"),
            "{mixed}"
        );
        assert_eq!(
            count(&mixed, "load i32, ptr addrspace(1)"),
            1,
            "the scalar read stays scalar\n{mixed}"
        );

        // A load that claims only the lane alignment is still a vector value, stamped with what it proved.
        let narrow = ir(&claiming(u32_vector_load_kernel(), 4), target);
        assert!(
            narrow.contains("load <4 x i32>, ptr addrspace(1)") && narrow.contains(", align 4\n"),
            "{target:?}\n{narrow}"
        );
    }
}

/// Mutation: lower `Rvalue::VectorLoad` on NVPTX with the lane alignment (`align 4`) in `emit_vector_load`; llc then
/// splits the vector into scalar loads and the `ld.global.v4` / `v2` counts go to zero.
#[test]
fn nvptx_issues_one_vector_instruction_per_aligned_vector_load() {
    require_tools(&["llc"]);
    let dir = out_dir("vector-load-nvptx");
    let ptx = |body: &Body, name: &str| {
        let out = poot_codegen::artifact_path(&dir, name, Target::Nvptx);
        compile(body, Target::Nvptx, &out).unwrap_or_else(|e| panic!("{name}: {e}"));
        std::fs::read_to_string(&out).unwrap()
    };
    let wide = ptx(&u32_vector_load_kernel(), "u32_vector_load");
    assert_eq!(count(&wide, "ld.global.v4.b32"), 1, "{wide}");
    let mixed = ptx(&u32_vector_load_mixed_kernel(), "u32_vector_load_mixed");
    assert_eq!(count(&mixed, "ld.global.v4.b32"), 1, "{mixed}");
    assert_eq!(count(&mixed, "ld.global.v2.b32"), 1, "{mixed}");
    // A weaker proof cannot become a vector instruction: the same body claiming the lane alignment loads scalars.
    let scalar = ptx(
        &claiming(u32_vector_load_kernel(), 4),
        "u32_vector_load_lane",
    );
    assert_eq!(count(&scalar, "ld.global.v4.b32"), 0, "{scalar}");
}

#[test]
fn spirv_binds_a_vector_only_param_as_a_vector_resource_and_reads_any_other_as_scalars() {
    let vector_resource = "[0 x <4 x i32>]";
    let wide = ir(&u32_vector_load_kernel(), Target::SpirvVulkan);
    assert!(
        wide.contains(vector_resource) && wide.contains("lshr i32"),
        "a vector-only, whole-vector-aligned param is a vector resource read by group\n{wide}"
    );
    assert_eq!(
        count(&wide, "load <4 x i32>, ptr addrspace(11)"),
        1,
        "{wide}"
    );

    // The same param also read as a u32x2 and a scalar cannot be one vector resource: its lanes are scalar loads.
    let mixed = ir(&u32_vector_load_mixed_kernel(), Target::SpirvVulkan);
    assert!(!mixed.contains(vector_resource), "{mixed}");
    assert!(!mixed.contains("[0 x <2 x i32>]"), "{mixed}");
    assert_eq!(
        scalar_input_reads(&mixed),
        4 + 2 + 1,
        "four + two lanes as scalars, plus the scalar read\n{mixed}"
    );

    // A load that proves less than a whole vector cannot divide into a group index either.
    let narrow = ir(&claiming(u32_vector_load_kernel(), 4), Target::SpirvVulkan);
    assert!(!narrow.contains(vector_resource), "{narrow}");
    assert_eq!(scalar_input_reads(&narrow), 4, "{narrow}");
}

#[test]
fn the_vulkan_modules_validate() {
    require_tools(&["llc", "spirv-val"]);
    let dir = out_dir("vector-load-spirv");
    for (body, name) in [
        (u32_vector_load_kernel(), "u32_vector_load"),
        (u32_vector_load_mixed_kernel(), "u32_vector_load_mixed"),
        (
            claiming(u32_vector_load_kernel(), 4),
            "u32_vector_load_lane",
        ),
    ] {
        let out = poot_codegen::artifact_path(&dir, name, Target::SpirvVulkan);
        compile(&body, Target::SpirvVulkan, &out).unwrap_or_else(|e| panic!("{name}: {e}"));
        let val = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            val.status.success(),
            "spirv-val failed for {name}:\n{}",
            String::from_utf8_lossy(&val.stderr)
        );
    }
}

#[test]
fn the_amdgcn_modules_compile() {
    require_amd_clang();
    let dir = out_dir("vector-load-amdgcn");
    let target = amd();
    for (body, name) in [
        (u32_vector_load_kernel(), "u32_vector_load"),
        (u32_vector_load_mixed_kernel(), "u32_vector_load_mixed"),
    ] {
        let out = poot_codegen::artifact_path(&dir, name, target);
        compile(&body, target, &out).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(std::fs::metadata(&out).unwrap().len() > 0);
    }
}

#[test]
fn the_aie_core_refuses_a_vector_load_by_name() {
    let error = emit_llvm_ir(&u32_vector_load_kernel(), Target::AieCore).unwrap_err();
    assert!(error.to_string().contains("Ty::Vec"), "{error}");
}
