//! End-to-end emit + `llc` check: the hand-built `add` and `square` kernels lower to a real `.spv` and
//! `.ptx` via system `llc`. Skips (passes) if `llc` is not on PATH, so `cargo test` outside the devshell
//! stays green.

use std::path::PathBuf;
use std::process::Command;

use poot_codegen::{Target, compile};
use poot_kernel_ir::fixtures;
use poot_test_util::kernel_fixtures::{
    e4m3fn_decode_kernel, e4m3fn_encode_kernel, emit_llvm_ir, gemv_loop_kernel, matmul_kernel,
    square_kernel, vadd_loop_kernel,
};

fn have(bin: &str) -> bool {
    Command::new(bin).arg("--version").output().is_ok()
}

fn out_dir(tag: &str) -> PathBuf {
    // A per-test subdir so concurrent tests do not race on artifact filenames.
    let d = std::env::temp_dir().join("poot-codegen-test").join(tag);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn add_lowers_to_spirv_and_ptx() {
    if !have("llc") {
        eprintln!("llc not on PATH; skipping");
        return;
    }
    let dir = out_dir("lower");
    for (body, name) in [
        (fixtures::add_kernel(), "add"),
        (square_kernel(), "square"),
        (matmul_kernel(2, 3, 4), "matmul"),
    ] {
        for target in [Target::SpirvVulkan, Target::Nvptx] {
            let out = poot_codegen::artifact_path(&dir, name, target);
            let ir =
                compile(&body, target, &out).unwrap_or_else(|e| panic!("{name} {target:?}: {e}"));
            let meta = std::fs::metadata(&out).unwrap();
            assert!(meta.len() > 0, "{name} {target:?}: empty artifact");
            // Sanity checks on the emitted PTX text.
            if target == Target::Nvptx {
                let ptx = std::fs::read_to_string(&out).unwrap();
                assert!(
                    ptx.contains(".visible .entry"),
                    "{name}: ptx missing entry\n{ir}"
                );
            }
        }
    }
}

#[test]
fn exact_i32_arithmetic_lowers_without_float_conversion() {
    if !have("llc") {
        eprintln!("llc not on PATH; skipping exact-I32 lowering");
        return;
    }
    let dir = out_dir("exact-i32");
    let bodies = [
        (
            poot_kernelgen::binary_scalar_i32_grid(
                "exact_i32_add",
                poot_kernel_ir::BinOp::Add,
                1,
                None,
            ),
            "add",
            "add i32",
        ),
        (
            poot_kernelgen::binary_scalar_i32_geu_grid("exact_i32_geu", i32::MIN, None),
            "geu",
            "icmp uge i32",
        ),
        (
            poot_kernelgen::binary_scalar_i32_remu_grid("exact_i32_remu", 0x7fff_ffff, None),
            "remu",
            "urem i32",
        ),
        (
            poot_kernelgen::binary_scalar_i32_grid(
                "exact_i32_and",
                poot_kernel_ir::BinOp::BitAnd,
                -1,
                None,
            ),
            "and",
            "and i32",
        ),
        (
            poot_kernelgen::binary_scalar_i32_grid(
                "exact_i32_shr",
                poot_kernel_ir::BinOp::Shr,
                1,
                None,
            ),
            "shr",
            "lshr i32",
        ),
        (
            poot_kernelgen::unary_i32_clz_grid("exact_i32_clz", None),
            "clz",
            "llvm.ctlz.i32",
        ),
        (
            poot_kernelgen::unary_dt_grid(
                "exact_i32_not",
                poot_kernel_ir::Ty::I32,
                poot_kernel_ir::UnOp::Not,
                None,
            ),
            "not",
            "xor i32",
        ),
        (
            poot_test_util::kernel_fixtures::fused_i32_views(
                "exact_i32_fused_clz",
                &[2],
                &[&[2], &[2]],
                &[
                    poot_kernelgen::Layout::contiguous(&[2]),
                    poot_kernelgen::Layout::contiguous(&[2]),
                ],
                &poot_kernelgen::FusedKernel {
                    n_leaves: 2,
                    steps: vec![
                        poot_kernelgen::FusedStep {
                            op: poot_kernelgen::FusedScalarOp::Binary(
                                poot_kernel_ir::BinOp::BitAnd,
                            ),
                            inputs: vec![
                                poot_kernelgen::FusedInput::Leaf(0),
                                poot_kernelgen::FusedInput::LitI32(-1),
                            ],
                        },
                        poot_kernelgen::FusedStep {
                            op: poot_kernelgen::FusedScalarOp::Clz,
                            inputs: vec![poot_kernelgen::FusedInput::Step(0)],
                        },
                    ],
                    output: poot_kernelgen::FusedInput::Step(1),
                },
            )
            .expect("fused_i32_views precondition"),
            "fused-clz",
            "llvm.ctlz.i32",
        ),
    ];
    let mut targets = vec![Target::SpirvVulkan, Target::Nvptx];
    if amd_toolchain_available() {
        targets.push(Target::AmdGcn(poot_target::AmdArch::gfx1151()));
    }
    for (body, name, opcode) in bodies {
        for &target in &targets {
            let out = poot_codegen::artifact_path(&dir, name, target);
            let ir = compile(&body, target, &out)
                .unwrap_or_else(|error| panic!("{name} {target:?}: {error}"));
            let expected = if opcode == "llvm.ctlz.i32" && target == Target::SpirvVulkan {
                "lshr i32"
            } else if opcode == "urem i32" && target == Target::Nvptx {
                // NVPTX integer remainder lowers as `a - (a/b)*b` (see emit_binop's Rem workaround).
                "udiv i32"
            } else {
                opcode
            };
            assert!(ir.contains(expected), "{name} {target:?}: {ir}");
            if opcode == "llvm.ctlz.i32" && target == Target::SpirvVulkan {
                assert!(!ir.contains("llvm.ctlz"), "{name} {target:?}: {ir}");
            }
            if name == "shr" {
                assert!(!ir.contains("ashr"), "{name} {target:?}: {ir}");
            }
            if name == "remu" {
                assert!(!ir.contains("sitofp"), "{name} {target:?}: {ir}");
                assert!(!ir.contains("srem"), "{name} {target:?}: {ir}");
            }
            assert!(!ir.contains("sitofp"), "{name} {target:?}: {ir}");
            assert!(std::fs::metadata(&out).unwrap().len() > 0);
        }
    }
}

/// Spec 149 Stage 1 SC-004 (CPU-safe compile gate): both format-aware conversion directions lower through
/// the software integer/float path without an LLVM FP8 type or native target instruction.
#[test]
fn e4m3fn_software_conversion_lowers_on_all_gpu_targets() {
    if !have("llc") {
        eprintln!("llc not on PATH; skipping FP8 conversion lowering");
        return;
    }
    let dir = out_dir("e4m3fn");
    for (body, name) in [
        (e4m3fn_encode_kernel(), "e4m3fn_encode"),
        (e4m3fn_decode_kernel(), "e4m3fn_decode"),
        (
            poot_kernelgen::f32_to_e4m3fn_packed("e4m3fn_pack", 5)
                .expect("f32_to_e4m3fn_packed precondition"),
            "e4m3fn_pack",
        ),
        (
            poot_kernelgen::e4m3fn_packed_to_f32("e4m3fn_unpack", 5)
                .expect("e4m3fn_packed_to_f32 precondition"),
            "e4m3fn_unpack",
        ),
        (
            poot_kernelgen::e4m3fn_repack_reshape("e4m3fn_repack", 5, 2)
                .expect("e4m3fn_repack_reshape precondition"),
            "e4m3fn_repack",
        ),
        (
            poot_kernelgen::e4m3fn_transpose_packed("e4m3fn_transpose", &[5, 2], &[2, 5], &[1, 0])
                .expect("e4m3fn_transpose_packed precondition"),
            "e4m3fn_transpose",
        ),
        (
            poot_kernelgen::e4m3fn_slice_packed("e4m3fn_slice", &[2, 3], &[2, 5], 1, 1)
                .expect("e4m3fn_slice_packed precondition"),
            "e4m3fn_slice",
        ),
        (
            poot_kernelgen::e4m3fn_concat_packed(
                "e4m3fn_concat",
                &[2, 6, 3],
                1,
                &[&[2, 2, 3], &[2, 1, 3], &[2, 3, 3]],
            )
            .expect("e4m3fn_concat_packed precondition"),
            "e4m3fn_concat",
        ),
        (
            poot_kernelgen::e4m3fn_gather_packed("e4m3fn_gather", &[2, 2, 5], &[2, 3, 5], 1, &[2])
                .expect("e4m3fn_gather_packed precondition"),
            "e4m3fn_gather",
        ),
        (
            poot_kernelgen::e4m3fn_dynamic_update_slice_packed(
                "e4m3fn_dynamic_update",
                &[2, 4, 5],
                &[2, 2, 5],
                1,
                1,
            )
            .expect("e4m3fn_dynamic_update_slice_packed precondition"),
            "e4m3fn_dynamic_update",
        ),
        (
            poot_kernelgen::e4m3fn_dynamic_update_slice_dynamic_packed(
                "e4m3fn_dynamic_update_runtime",
                &[2, 5],
                &[2, 2],
                1,
            )
            .expect("e4m3fn_dynamic_update_slice_dynamic_packed precondition"),
            "e4m3fn_dynamic_update_runtime",
        ),
        (
            poot_kernelgen::e4m3fn_broadcast_packed("e4m3fn_broadcast", &[2, 3, 5], &[2, 1, 5])
                .expect("e4m3fn_broadcast_packed precondition"),
            "e4m3fn_broadcast",
        ),
        (
            poot_kernelgen::e4m3fn_scatter_update_packed(
                "e4m3fn_scatter_update",
                &[4, 2, 5],
                &[3, 2, 5],
            )
            .expect("e4m3fn_scatter_update_packed precondition"),
            "e4m3fn_scatter_update",
        ),
        (
            poot_kernelgen::e4m3fn_scatter_update_packed("e4m3fn_scatter_update_rank1", &[5], &[3])
                .expect("e4m3fn_scatter_update_packed precondition"),
            "e4m3fn_scatter_update_rank1",
        ),
    ] {
        for target in [Target::SpirvVulkan, Target::Nvptx] {
            let out = poot_codegen::artifact_path(&dir, name, target);
            compile(&body, target, &out).unwrap_or_else(|e| panic!("{name} {target:?}: {e}"));
            assert!(std::fs::metadata(&out).unwrap().len() > 0);
        }
        if amd_toolchain_available() {
            let target = Target::AmdGcn(poot_target::AmdArch::gfx1151());
            let out = poot_codegen::artifact_path(&dir, name, target);
            compile(&body, target, &out).unwrap_or_else(|e| panic!("{name} {target:?}: {e}"));
            assert!(std::fs::metadata(&out).unwrap().len() > 0);
        } else {
            eprintln!("no AMD toolchain; skipping {name} AMDGPU lowering");
        }
    }
}

/// Card 227: `poot_codegen::compile` must be safe under concurrent same-body compiles. `out` is
/// content-addressed, so threads compiling the same body would otherwise write the same `.ll` and `out`
/// and publish a corrupt artifact (the CUDA_ERROR_INVALID_IMAGE / HSA_STATUS_ERROR_INVALID_CODE_OBJECT
/// class seen in tensor-parallel serving). Each compile now gets a unique `.ll`/temp-object scratch
/// (process-global atomic nonce) and atomic-renames onto `out`. This drives many threads at the same
/// `out` and asserts every compile succeeds and the artifact is well-formed SPIR-V. Skips if `llc` is
/// absent.
#[test]
fn concurrent_same_body_compiles_do_not_corrupt_the_shared_artifact() {
    if !have("llc") {
        eprintln!("llc not on PATH; skipping concurrent-compile race test");
        return;
    }
    let dir = out_dir("race");
    // One content-addressed output path shared by every thread (the tensor-parallel collision shape).
    let out = poot_codegen::artifact_path(&dir, "add_race", Target::SpirvVulkan);
    let out = std::sync::Arc::new(out);
    // matmul lowers to more IR than `add`, widening the race window.
    const THREADS: usize = 16;
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let out = out.clone();
            std::thread::spawn(move || {
                let body = matmul_kernel(2, 3, 4);
                compile(&body, Target::SpirvVulkan, &out).map(|_| ())
            })
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        h.join()
            .expect("compile thread panicked")
            .unwrap_or_else(|e| panic!("concurrent compile {i} failed: {e}"));
    }
    // The artifact must be well-formed SPIR-V: non-empty, word-aligned, correct magic.
    let bytes = std::fs::read(&*out).expect("published artifact missing");
    assert!(
        !bytes.is_empty(),
        "published SPIR-V is empty (corrupted by the race)"
    );
    assert_eq!(
        bytes.len() % 4,
        0,
        "published SPIR-V not word-aligned (torn write)"
    );
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(
        magic, 0x0723_0203,
        "published artifact is not a SPIR-V module (magic {magic:#x})"
    );
    // No `.ll` / `.tmp` scratch files may remain in the output dir.
    let leftover: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".ll") || n.ends_with(".tmp"))
        .collect();
    assert!(
        leftover.is_empty(),
        "compile left scratch siblings behind: {leftover:?}"
    );
}

/// Resolve a runnable Peano `llc` (`$POOT_AIE_LLC`, an FHS wrapper on NixOS) that knows `aie2p`, or `None`
/// to skip. Peano is a separate llc fork from system LLVM.
fn peano_llc() -> Option<String> {
    let p = std::env::var("POOT_AIE_LLC").ok()?;
    let v = Command::new(&p).arg("--version").output().ok()?;
    if !String::from_utf8_lossy(&v.stdout).contains("aie2p") {
        return None;
    }
    Some(p)
}

/// Lower a body to an AIE2p object via Peano and assert it is an ELF carrying the kernel `symbol`. Peano
/// runs under an FHS sandbox (steam-run) that binds $HOME but not the nix-shell TMPDIR, so the .ll input and
/// .o output must live under $HOME.
fn assert_lowers_to_aie2p(body: &poot_kernel_ir::Body, symbol: &str) {
    let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/poot-codegen-test/aie");
    std::fs::create_dir_all(&dir).unwrap();
    let out = poot_codegen::artifact_path(&dir, symbol, Target::AieCore);
    let ir = compile(body, Target::AieCore, &out)
        .unwrap_or_else(|e| panic!("AIE-core compile failed for {symbol}: {e}"));
    let bytes = std::fs::read(&out).unwrap();
    assert!(!bytes.is_empty(), "empty AIE object\n{ir}");
    // Check the ELF magic and that the kernel symbol survived (IRON links it by C symbol via
    // external_func(link_with=...)).
    assert_eq!(&bytes[0..4], b"\x7fELF", "not an ELF object\n{ir}");
    assert!(
        bytes.windows(symbol.len()).any(|w| w == symbol.as_bytes()),
        "kernel symbol `{symbol}` not found in the AIE object"
    );
}

/// Card 089 / spec 057 (FR-002): poot's LLVM IR for a scalar kernel lowers to an AIE2p core object through
/// Peano (`llc --march=aie2p`). This is the codegen half only; dispatch on the NPU needs XRT + the IRON
/// harness (a separate on-hardware test). `vadd_loop` is the elementwise case; `gemv_loop` covers the
/// nested-loop + f32-accumulator shape of decode matvec. Skips when Peano is absent.
#[test]
fn aie_kernels_lower_to_aie2p_via_peano() {
    if peano_llc().is_none() {
        eprintln!(
            "POOT_AIE_LLC (Peano llc with aie2p) not set/runnable; skipping AIE-core lowering test"
        );
        return;
    }
    assert_lowers_to_aie2p(&vadd_loop_kernel(), "vadd_loop");
    // GEMV (M=8, K=16): outer and inner back-edged loops, an accumulator carried across the inner loop,
    // a multiply, and `i*K + j` flat indexing.
    assert_lowers_to_aie2p(&gemv_loop_kernel(8, 16), "gemv_loop");
}

#[test]
fn add_spirv_validates() {
    if !have("llc") || !have("spirv-val") {
        eprintln!("llc/spirv-val not on PATH; skipping");
        return;
    }
    let dir = out_dir("val");
    let out = poot_codegen::artifact_path(&dir, "add", Target::SpirvVulkan);
    compile(&fixtures::add_kernel(), Target::SpirvVulkan, &out).unwrap();
    let v = Command::new("spirv-val")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        v.status.success(),
        "spirv-val failed:\n{}",
        String::from_utf8_lossy(&v.stderr)
    );
}

/// Card 154 (target-neutral by card 530): the SPIR-V cooperative-matrix WMMA path end to end
/// (`compile()` -> `llc` -> the `fix_coopmat_calls` post-process in `fixup_spirv_barriers` ->
/// `spirv-val`), as `add_spirv_validates` does for the plain kernel. `wmma_tile` uses SPIR-V
/// storage-buffer resource-pointer addressing (`ptr addrspace(11)`), not the LDS-only shape of the
/// `spirv_postprocess` unit tests, so it covers the matmul-dispatch codegen path and not just the
/// rewrite.
#[test]
fn coopmat_tile_spirv_validates() {
    if !have("llc") || !have("spirv-val") {
        eprintln!("llc/spirv-val not on PATH; skipping");
        return;
    }
    let dir = out_dir("coopmat");
    let out = poot_codegen::artifact_path(&dir, "wmma_coopmat_tile", Target::SpirvVulkan);
    let body = poot_test_util::kernel_fixtures::wmma_tile("wmma_coopmat_tile");
    compile(&body, Target::SpirvVulkan, &out).unwrap();
    let v = Command::new("spirv-val")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        v.status.success(),
        "spirv-val failed:\n{}",
        String::from_utf8_lossy(&v.stderr)
    );
}

/// Whether an AMDGPU-capable toolchain is available. `compile()` for `Target::AmdGcn` needs a runnable
/// `$POOT_AMD_CLANG` (the AMD-flavored clang that adds the HSA load headers) and has no fallback.
fn amd_toolchain_available() -> bool {
    std::env::var("POOT_AMD_CLANG")
        .is_ok_and(|clang| Command::new(clang).arg("--version").output().is_ok())
}

/// Card 106 (spec 063, SC-002 + FR-006): the emitted LLVM IR for the elementwise `add` kernel lowers to an
/// AMDGPU ELF HSACO via the ROCm-flavored llc (`-mtriple=amdgcn-amd-amdhsa -mcpu=gfx1151 -O2
/// -filetype=obj`), and the ELF carries `e_machine = EM_AMDGPU (0xe0)`, the gfx1151 `EF_AMDGPU_MACH`, and a
/// kernel symbol named `add`. Skips when the AMD llc is absent. Device-libs linking uses
/// `$POOT_ROCM_DEVICE_LIBS` (set by the flake); if it is absent the test may fail at the llc link step,
/// deliberately, rather than skipping the requirement.
#[test]
fn add_lowers_to_amdgpu() {
    if !amd_toolchain_available() {
        eprintln!("no AMD toolchain (POOT_AMD_CLANG) available; skipping AMDGPU lowering test");
        return;
    }
    let target = Target::AmdGcn(poot_target::AmdArch::gfx1151());
    let dir = out_dir("amdgpu");
    let out = poot_codegen::artifact_path(&dir, "add", target);
    let ir = compile(&fixtures::add_kernel(), target, &out)
        .unwrap_or_else(|e| panic!("AMDGPU compile failed for `add`: {e}"));
    let bytes = std::fs::read(&out).unwrap();
    assert!(!bytes.is_empty(), "empty HSACO\n{ir}");
    // ELF magic.
    assert_eq!(&bytes[0..4], b"\x7fELF", "HSACO is not an ELF object\n{ir}");
    // e_machine is a little-endian u16 at offset 0x12 in the ELF64 header (16-byte e_ident + 2-byte
    // e_type). EM_AMDGPU = 0xe0 = 224 (`amd_hsa_elf.h`).
    let e_machine = u16::from_le_bytes([bytes[0x12], bytes[0x13]]);
    assert_eq!(
        e_machine, 0xe0,
        "e_machine must be EM_AMDGPU (0xe0), got {e_machine:#x}\n{ir}"
    );
    // ELF64 header layout (gABI):
    //   e_ident(16) | e_type(2) | e_machine(2) | e_version(4) | e_entry(8) | e_phoff(8) | e_shoff(8)
    //   | e_flags(4) | e_ehsize(2) | e_phentsize(2) | e_phnum(2) | e_shentsize(2) | e_shnum(2)
    //   | e_shstrndx(2)
    // e_flags is at byte offset 0x30 (`amdgcn-amd-amdhsa` is 64-bit, so ELF64).
    //
    // gfx1151 = 0x4a in the lower byte of e_flags (`BinaryFormat/ELF.h` in the ROCm LLVM 22 fork), the
    // AMDGPU MACH id the LLVM backend and ROCR's `hsa_agent_get_info(ISA)` use. The `0x1151` in the
    // original task text and spec FR-006 was a notational slip.
    let e_flags = u32::from_le_bytes([bytes[0x30], bytes[0x31], bytes[0x32], bytes[0x33]]);
    assert_eq!(
        e_flags & 0xFF,
        0x4a,
        "EF_AMDGPU_MACH must be 0x4a (gfx1151), got e_flags={e_flags:#x}\n{ir}"
    );
    // Only assert the kernel name appears as a string in the ELF (dynamic string table / symbol names);
    // the `STT_AMDGPU_HSA_KERNEL` binding is not checked.
    assert!(
        bytes.windows(3).any(|w| w == b"add"),
        "kernel symbol `add` not found in the HSACO"
    );
}

/// Spec 138 Phase 2 (FR-003/004, SC-004): `Statement::Fence` lowers to an AMDGPU HSACO for both the
/// workgroup-scope kernel (`workgroup_fence_spin_wait`, also dispatched via RADV/Vulkan in
/// `poot-kernelgen`'s tests) and the device-scope kernel (`cross_block_fence_spin_wait_coop`, compile-only
/// here; its runtime handoff needs a cooperative NVPTX grid on a pod). Confirms the ROCm-flavored llc
/// accepts the `fence syncscope("workgroup"/"agent") ...` IR from `Emitter::fence_syncscope`. Skips with no
/// AMD toolchain.
#[test]
fn fence_lowers_to_amdgpu_both_scopes() {
    if !amd_toolchain_available() {
        eprintln!("no AMD toolchain (POOT_AMD_CLANG) available; skipping AMDGCN fence test");
        return;
    }
    let target = Target::AmdGcn(poot_target::AmdArch::gfx1151());
    let dir = out_dir("amdgpu-fence");
    for (name, body) in [
        (
            "wgfence_amd",
            poot_test_util::kernel_fixtures::workgroup_fence_spin_wait("wgfence_amd", 128),
        ),
        (
            "devfence_amd",
            poot_kernelgen::cross_block_fence_spin_wait_coop("devfence_amd"),
        ),
    ] {
        let out = poot_codegen::artifact_path(&dir, name, target);
        let ir = compile(&body, target, &out)
            .unwrap_or_else(|e| panic!("AMDGCN compile failed for {name}: {e}"));
        let bytes = std::fs::read(&out).unwrap();
        assert!(!bytes.is_empty(), "{name}: empty HSACO\n{ir}");
        assert_eq!(
            &bytes[0..4],
            b"\x7fELF",
            "{name}: HSACO is not an ELF object\n{ir}"
        );
    }
}

/// Spec 120 SC-002: building the same kernel for a second gfx target (gfx1100, Navi 31) selects
/// `-mcpu=gfx1100` and produces an HSACO whose `EF_AMDGPU_MACH` follows the threaded target (0x41 for
/// gfx1100, 0x4a for gfx1151). Guards that codegen follows the threaded `AmdArch`, not a `gfx1151`
/// constant. The HSACO is only compiled and ELF-inspected; this box is a gfx1151 iGPU.
#[test]
fn second_amd_target_follows_mach() {
    if !amd_toolchain_available() {
        eprintln!("no AMD toolchain (POOT_AMD_CLANG) available; skipping second-target test");
        return;
    }
    let target = Target::AmdGcn(poot_target::AmdArch::new("gfx1100", 32));
    let dir = out_dir("amdgpu-gfx1100");
    let out = poot_codegen::artifact_path(&dir, "add", target);
    let ir = compile(&fixtures::add_kernel(), target, &out)
        .unwrap_or_else(|e| panic!("gfx1100 compile failed for `add`: {e}"));
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(
        &bytes[0..4],
        b"\x7fELF",
        "gfx1100 HSACO is not an ELF object\n{ir}"
    );
    let e_machine = u16::from_le_bytes([bytes[0x12], bytes[0x13]]);
    assert_eq!(e_machine, 0xe0, "e_machine must be EM_AMDGPU (0xe0)\n{ir}");
    // EF_AMDGPU_MACH_AMDGCN_GFX1100 = 0x041 (`BinaryFormat/ELF.h`), distinct from gfx1151's 0x4a.
    let e_flags = u32::from_le_bytes([bytes[0x30], bytes[0x31], bytes[0x32], bytes[0x33]]);
    assert_eq!(
        e_flags & 0xFF,
        0x41,
        "EF_AMDGPU_MACH must be 0x41 (gfx1100), got e_flags={e_flags:#x}\n{ir}"
    );
}

/// Card 145 follow-on (spec 130 FR-004): the AMD WMMA tensor-core matmul body lowers to the RDNA3 WMMA
/// intrinsic for `Target::AmdGcn(gfx1151)`. This is the body `plan_eqn` selects for a mixed
/// bf16-in/f32-out matmul, the only shape that reaches AMD WMMA (see the plan-side test
/// `amd_wmma_reached_only_by_mixed_bf16_in_f32_out`). `emit_llvm_ir` is pure Rust (no llc, no GPU), so
/// this always runs. It guards that WMMA selection codegens the intrinsic, not a serial fallback or an
/// unresolved extern. With the AMD toolchain present it also lowers the IR through llc to an HSACO;
/// otherwise it skips only that half.
#[test]
fn amd_wmma_matmul_lowers_to_wmma_intrinsic() {
    use poot_kernel_ir::Ty;
    use poot_target::AmdArch;
    let target = Target::AmdGcn(AmdArch::gfx1151());
    // bf16 operands, f32 output: the mixed-precision matmul the plan routes to AMD WMMA.
    let body = poot_kernelgen::matmul_tensorcore("k", Ty::F32, &[16, 16], &[16, 32], &[32, 16])
        .expect("matmul_tensorcore precondition");
    let ir = emit_llvm_ir(&body, target).expect("emit AMD WMMA IR");
    // The RDNA3 WMMA 16x16x16 bf16-in/f32-accumulate intrinsic (emit.rs `emit_wmma_mma_amdgcn`).
    assert!(
        ir.contains("llvm.amdgcn.wmma.f32.16x16x16.bf16"),
        "AMD WMMA matmul must lower to the RDNA3 WMMA intrinsic; IR:\n{ir}"
    );
    // The NVPTX tensor-core intrinsic must not leak into the AMD lowering.
    assert!(
        !ir.contains("nvvm.wmma"),
        "NVPTX wmma intrinsic leaked into AMD IR:\n{ir}"
    );
    // If the AMD toolchain is present, confirm llc accepts the WMMA IR.
    if amd_toolchain_available() {
        let dir = out_dir("amdgpu-wmma");
        let out = poot_codegen::artifact_path(&dir, "wmma_mm", target);
        let ir2 = compile(&body, target, &out).unwrap_or_else(|e| panic!("AMD WMMA compile: {e}"));
        let bytes = std::fs::read(&out).unwrap();
        assert!(!bytes.is_empty(), "empty WMMA HSACO\n{ir2}");
        assert_eq!(
            &bytes[0..4],
            b"\x7fELF",
            "WMMA HSACO is not an ELF object\n{ir2}"
        );
    }
}
