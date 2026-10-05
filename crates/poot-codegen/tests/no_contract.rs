//! Card 628 (dquant.md section 7 R1) end-to-end acceptance: `Rvalue::BinaryOpNoContract`
//! lowers to a real per-backend no-contraction mechanism, verified against the actual toolchain
//! (`fixtures::no_contract_probe_kernel`, `out[i] = a[i]*b[i] - c[i]*d[i]`). Skips (passes) if the relevant
//! toolchain is absent, matching `llc.rs`'s convention so `cargo test` outside the devshell stays green.
//!
//! The marker sits on the final **subtract**, not either multiply - a real-hardware finding (RADV/ACO,
//! `poot-runtime`'s `no_contract_tests` module): decorating only the second multiply (`c*d`) leaves ACO
//! free to instead fuse the *other*, undecorated multiply (`a*b`) into the subtract, still diverging from
//! the unfused reference on the same rows, just via a different fused term. Decorating the subtract itself
//! blocks contracting it with either producing multiply (SPIR-V's `NoContraction`: "must not be contracted
//! with... any other operation", applied at the consumer); the same reasoning carries to the constrained
//! intrinsic on AmdGcn (a `STRICT_FSUB` DAG node is not the plain `ISD::FSUB` the FMA-combine pass matches
//! on, regardless of which producer feeds it) - AmdGcn's compiled HSACO is the device's final ISA, so this
//! is a complete backend-level guarantee there. **Nvptx's guarantee is ISA-level, not marker-level**: this
//! crate's `llc` legalizes the constrained intrinsic straight back to the plain PTX instruction, so the
//! marked and unmarked kernels compile to byte-identical `.ptx` (see
//! `nvptx_marked_subtract_emits_constrained_intrinsic_under_strictfp`'s doc comment). What actually blocks
//! `ptxas` (the JIT that runs at kernel launch, outside this crate's toolchain) from contracting either
//! kernel is PTX ISA 9.4 SS9.7.3.3/.4/.5 ("add"/"sub"/"mul", Notes): an add/sub/mul instruction with an
//! explicit rounding modifier (`.rn` here) "is treated conservatively by the code optimizer", whereas one
//! with no modifier "may be optimized aggressively... to use fused-multiply-add instructions" - and `llc`
//! always emits the explicit modifier for ordinary (non-`contract`-flagged) float arithmetic, marked or
//! not. The marker's own mechanism (the constrained intrinsic) stays useful as a hedge against a narrower
//! future regression - `emit_binop`'s plain path ever gaining a `contract` flag would fuse it outright at
//! the `llc` level, while the marked path's constrained call would not (confirmed directly: `contract` on
//! a plain `fmul`/`fsub` pair fuses into `fma.rn.f32`; the same flag on the constrained call does not fuse
//! but does drop the `.rn`, which the marked test's assertion below catches) - not because it changes
//! today's compiled output.
//!
//! SC-001 (NVPTX real-hardware FMA divergence) and the numeric half of SC-002 (RADV/ACO dispatch) need
//! device execution this box/toolchain cannot provide from `poot-codegen` alone (no NVIDIA GPU locally;
//! the JIT step - `ptxas` - runs on the CUDA driver, not in this crate's own toolchain; a real dispatch
//! needs `poot-runtime`'s wgpu harness). SC-002's numeric half is proven in `poot-runtime`'s
//! `no_contract_tests` module (real RADV/ACO dispatch, marked matches the unfused reference bit for bit,
//! unmarked diverges on the same rows); SC-001's NVPTX device row (marked matches the unfused reference)
//! runs on the PTX pod - there is no corresponding "unmarked diverges" acceptance row on Nvptx,
//! since the ISA guarantee above applies to both kernels identically, so that assertion would only ever
//! restate the same guarantee under a different, more misleading name.

use std::path::PathBuf;
use std::process::Command;

use poot_codegen::{Target, compile};
use poot_test_util::kernel_fixtures::no_contract_probe_kernel;

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

/// SC-002: marking the final subtract decorates its compiled `OpFSub` `NoContraction`, inspected in the
/// disassembled module - the mutation (`marked: false`) drops the decoration entirely, the detector this
/// row exists to prove. Both forms must still be valid Vulkan SPIR-V (`spirv-val`): the decoration must not
/// itself break validation.
#[test]
fn spirv_marked_subtract_is_decorated_nocontraction_unmarked_is_not() {
    if !have("llc") || !have("spirv-dis") || !have("spirv-val") {
        eprintln!("llc/spirv-dis/spirv-val not on PATH; skipping");
        return;
    }
    let dir = out_dir("no-contract-spirv");
    for (marked, tag) in [(true, "marked"), (false, "unmarked")] {
        let body = no_contract_probe_kernel(marked);
        let out =
            poot_codegen::artifact_path(&dir, &format!("no_contract_{tag}"), Target::SpirvVulkan);
        let ir = compile(&body, Target::SpirvVulkan, &out)
            .unwrap_or_else(|e| panic!("SpirvVulkan compile ({tag}) failed: {e}"));

        let val = Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            val.status.success(),
            "spirv-val failed for {tag}:\n{}",
            String::from_utf8_lossy(&val.stderr)
        );

        let dis = Command::new("spirv-dis").arg(&out).output().unwrap();
        assert!(dis.status.success(), "spirv-dis failed for {tag}");
        let text = String::from_utf8_lossy(&dis.stdout);
        let no_contraction_count = text.matches("NoContraction").count();
        if marked {
            assert_eq!(
                no_contraction_count, 1,
                "expected exactly one NoContraction decoration for the marked subtract, found \
                 {no_contraction_count}\nemitted LLVM IR:\n{ir}\ndisassembly:\n{text}"
            );
        } else {
            assert_eq!(
                no_contraction_count, 0,
                "expected no NoContraction decoration for the unmarked body (SC-002's mutation), found \
                 {no_contraction_count}\ndisassembly:\n{text}"
            );
        }
    }
}

/// SC-001's IR-level half: marking the final subtract lowers to `llvm.experimental.constrained.fsub.f32`
/// under `strictfp` - mutation-provable (dropping the NVPTX arm of `emit_binop_no_contract` turns this red
/// with no GPU needed, confirmed against card 628's real mutation).
///
/// The artifact-level guarantee this backs is documented, not assumed: PTX ISA 9.4 SS9.7.3.3/.4/.5
/// ("add"/"sub"/"mul", Notes) - "a [add/sub/mul] instruction with an explicit rounding modifier is treated
/// conservatively by the code optimizer. A[n] [add/sub/mul] instruction with no rounding modifier defaults
/// to round-to-nearest-even and may be optimized aggressively... In particular, mul/[add/sub] sequences
/// with no rounding modifiers may be optimized to use fused-multiply-add instructions on the target
/// device." `llc` always emits the explicit `.rn` qualifier for ordinary (non-`contract`-flagged) float
/// add/sub/mul, so the compiled `sub.rn.f32` below is the real, ISA-guaranteed non-contraction proof `ptxas`
/// must honour - not an undocumented default this box happened to observe. That is also why the unmarked
/// kernel doesn't diverge on real NVPTX hardware either (batch #4, RTX 4090/610.43.02, 2026-09-29): `llc`
/// never attaches `contract` to `emit_binop`'s plain float path, so it gets the identical `.rn`-qualified,
/// ISA-protected form. Confirmed with two direct experiments against this box's `llc` (not committed, no
/// GPU involved): (1) attaching the `contract` fast-math flag to a plain `fsub`/`fmul` pair makes `llc`
/// fuse them into a single `fma.rn.f32` outright, dropping both separate instructions; (2) attaching
/// `contract` to the marked op's `llvm.experimental.constrained.fsub.f32` *call* does not make `llc` fuse
/// it (it stays three separate instructions - the constrained form is immune to `llc`'s own DAG-combine
/// FMA formation), but it DOES make `llc` drop the sub's explicit `.rn` (emitting bare `sub.f32`) - the
/// weaker, "may be optimized aggressively" form `ptxas` is free to contract. The byte-identity check below
/// is what catches that second case for the marker's own emission path (mutation confirmed manually
/// 2026-09-29: attaching `contract` to `emit_binop_no_contract`'s Nvptx call turns this test's `sub.rn.f32`
/// assertion red - `assert!(ptx.contains("sub.rn.f32"))` failed, compiled PTX had bare `sub.f32` instead -
/// then reverted, confirmed green again).
#[test]
fn nvptx_marked_subtract_emits_constrained_intrinsic_under_strictfp() {
    if !have("llc") {
        eprintln!("llc not on PATH; skipping");
        return;
    }
    let dir = out_dir("no-contract-nvptx");
    let mut compiled = std::collections::HashMap::new();
    for (marked, tag) in [(true, "marked"), (false, "unmarked")] {
        let body = no_contract_probe_kernel(marked);
        let out = poot_codegen::artifact_path(&dir, &format!("no_contract_{tag}"), Target::Nvptx);
        let ir = compile(&body, Target::Nvptx, &out)
            .unwrap_or_else(|e| panic!("Nvptx compile ({tag}) failed: {e}"));
        let has_constrained = ir.contains("llvm.experimental.constrained.fsub.f32");
        let has_strictfp = ir.contains("strictfp");
        assert_eq!(
            has_constrained, marked,
            "{tag}: expected constrained.fsub.f32 present={marked}, IR:\n{ir}"
        );
        assert_eq!(
            has_strictfp, marked,
            "{tag}: expected strictfp present={marked}, IR:\n{ir}"
        );

        let ptx = std::fs::read_to_string(&out).unwrap();
        assert!(
            !ptx.contains("fma.rn.f32"),
            "{tag}: compiled .ptx already contracted at the llc level (unexpected on this toolchain):\n{ptx}"
        );
        // The real per-instruction no-contraction guarantee (PTX ISA 9.4 SS9.7.3.4 "sub", Notes): an
        // explicit rounding modifier is required for `ptxas` to treat the op conservatively (not
        // contractible). Confirmed mutation-red (see doc comment above): a `contract` flag reaching this
        // call site drops the modifier to bare `sub.f32` without touching `has_constrained`/`has_strictfp`.
        assert!(
            ptx.contains("sub.rn.f32"),
            "{tag}: compiled .ptx lost the explicit .rn rounding modifier on the subtract - this is the \
             PTX-ISA-guaranteed non-contraction proof (SS9.7.3.4 Notes), not just IR-level bookkeeping:\n{ptx}"
        );
        compiled.insert(tag, ptx);
    }
    // Today, marked and unmarked take different codegen paths (constrained intrinsic vs plain fsub) but
    // compile to byte-identical .ptx, because neither path ever attaches `contract`. This is a strictly
    // stronger check than the two assertions above (it would also catch e.g. a changed register/operand
    // order) and pins down exactly how redundant the marker is on Nvptx today.
    assert_eq!(
        compiled["marked"], compiled["unmarked"],
        "marked and unmarked compiled to DIFFERENT .ptx text - the NVPTX marker is no longer inert at \
         the artifact level; update the doc comment above and check whether ptxas-level protection now \
         differs between the two kernels"
    );
}

/// The AMDGCN counterpart of the NVPTX check above, run through the real production toolchain
/// (`$POOT_AMD_CLANG`, the same one `compile()` uses) with a disassembly of the compiled HSACO: unlike
/// NVPTX's PTX (JIT-recompiled by the CUDA driver at kernel launch, outside this crate's toolchain),
/// AMDGCN's HSACO is the device's final ISA - ahead of time, no further JIT - so this is a complete,
/// real-hardware-verifiable proof for this backend, not just an IR-level check. Marking the subtract keeps
/// it and both multiplies as distinct instructions; dropping the mark still does not fuse them either on
/// this llc (no `contract` flag present), which is exactly why the marker's own mechanism (a constrained
/// intrinsic immune to a future default change, not reliance on today's flag-absence default) is the point
/// of this card.
#[test]
fn amdgcn_marked_subtract_stays_unfused_in_the_compiled_isa() {
    if !amd_toolchain_available() {
        eprintln!(
            "no AMD toolchain (POOT_AMD_CLANG) available; skipping AMDGCN no-contraction test"
        );
        return;
    }
    let objdump = {
        let clang = std::env::var("POOT_AMD_CLANG").unwrap();
        PathBuf::from(&clang).parent().unwrap().join("llvm-objdump")
    };
    if !objdump.exists() {
        eprintln!("llvm-objdump not found next to POOT_AMD_CLANG; skipping disassembly check");
        return;
    }
    let dir = out_dir("no-contract-amdgcn");
    let target = Target::AmdGcn(poot_target::AmdArch::gfx1151());
    for (marked, tag) in [(true, "marked"), (false, "unmarked")] {
        let body = no_contract_probe_kernel(marked);
        let out = poot_codegen::artifact_path(&dir, &format!("no_contract_{tag}"), target);
        let ir = compile(&body, target, &out)
            .unwrap_or_else(|e| panic!("AmdGcn compile ({tag}) failed: {e}"));
        assert_eq!(
            ir.contains("llvm.experimental.constrained.fsub.f32"),
            marked,
            "{tag}: expected constrained.fsub.f32 present={marked}, IR:\n{ir}"
        );

        let dump = Command::new(&objdump)
            .arg("-d")
            .arg(&out)
            .output()
            .unwrap_or_else(|e| panic!("llvm-objdump failed: {e}"));
        assert!(dump.status.success(), "llvm-objdump failed for {tag}");
        let asm = String::from_utf8_lossy(&dump.stdout);
        assert!(
            !asm.to_lowercase().contains("fmac") && !asm.to_lowercase().contains("_fma_"),
            "{tag}: compiled HSACO contracted the marked subtract with a producing multiply into a fused \
             instruction:\n{asm}"
        );
    }
}
