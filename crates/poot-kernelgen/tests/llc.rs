//! End-to-end emit + `llc` check: the hand-built `add` and `square` kernels lower to a real `.spv` and
//! `.ptx` via system `llc`. Skips (passes) if `llc` is not on PATH, so `cargo test` outside the devshell
//! stays green.

use std::path::PathBuf;
use std::process::Command;

use poot_codegen::emit_llvm_ir;
use poot_codegen::{Target, compile};
use poot_kernel_ir::test_support::{
    e4m3fn_encode_kernel, gemv_loop_kernel, matmul_kernel, square_kernel, vadd_loop_kernel,
};
use poot_kernel_ir::{Ty, fixtures};
use poot_kernelgen::{MovementOp, MovementSpec, Storage, UpdateIndex};

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
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
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

/// The exact-I32 body of one `op` step over a `[4]` leaf, with the literal as its second operand when given.
fn one_step_i32(
    name: &str,
    op: poot_kernelgen::FusedScalarOp,
    lit: Option<i32>,
) -> poot_kernel_ir::Body {
    let mut inputs = vec![poot_kernelgen::FusedInput::Leaf(
        poot_kernelgen::LeafId::new(0),
    )];
    inputs.extend(lit.map(poot_kernelgen::FusedInput::LitI32));
    poot_kernelgen::fused_i32_views_grid(
        name,
        &[4],
        &[&[4]],
        &[poot_kernelgen::Layout::contiguous(&[4])],
        &poot_kernelgen::FusedKernel {
            leaf_tys: vec![poot_kernelgen::LeafTy::I32],
            steps: vec![poot_kernelgen::FusedStep { op, inputs }],
            output: poot_kernelgen::FusedInput::Step(poot_kernelgen::StepId::new(0)),
        },
        None,
    )
    .expect("an exact-I32 step generates")
}

#[test]
fn exact_i32_arithmetic_lowers_without_float_conversion() {
    if !have("llc") {
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
        return;
    }
    let dir = out_dir("exact-i32");
    let bodies = [
        (
            one_step_i32(
                "exact_i32_add",
                poot_kernelgen::FusedScalarOp::Binary(poot_kernel_ir::BinOp::Add),
                Some(1),
            ),
            "add",
            "add i32",
        ),
        (
            one_step_i32(
                "exact_i32_geu",
                poot_kernelgen::FusedScalarOp::GeU,
                Some(i32::MIN),
            ),
            "geu",
            "icmp uge i32",
        ),
        (
            one_step_i32(
                "exact_i32_remu",
                poot_kernelgen::FusedScalarOp::RemU,
                Some(0x7fff_ffff),
            ),
            "remu",
            "urem i32",
        ),
        (
            one_step_i32(
                "exact_i32_and",
                poot_kernelgen::FusedScalarOp::Binary(poot_kernel_ir::BinOp::BitAnd),
                Some(-1),
            ),
            "and",
            "and i32",
        ),
        (
            one_step_i32(
                "exact_i32_shr",
                poot_kernelgen::FusedScalarOp::Binary(poot_kernel_ir::BinOp::Shr),
                Some(1),
            ),
            "shr",
            "lshr i32",
        ),
        (
            one_step_i32("exact_i32_clz", poot_kernelgen::FusedScalarOp::Clz, None),
            "clz",
            "llvm.ctlz.i32",
        ),
        (
            one_step_i32(
                "exact_i32_not",
                poot_kernelgen::FusedScalarOp::Unary(poot_kernel_ir::UnOp::Not),
                None,
            ),
            "not",
            "xor i32",
        ),
        (
            poot_kernelgen::fused_i32_views_grid(
                "exact_i32_fused_clz",
                &[2],
                &[&[2], &[2]],
                &[
                    poot_kernelgen::Layout::contiguous(&[2]),
                    poot_kernelgen::Layout::contiguous(&[2]),
                ],
                &poot_kernelgen::FusedKernel {
                    leaf_tys: vec![poot_kernelgen::LeafTy::I32; 2],
                    steps: vec![
                        poot_kernelgen::FusedStep {
                            op: poot_kernelgen::FusedScalarOp::Binary(
                                poot_kernel_ir::BinOp::BitAnd,
                            ),
                            inputs: vec![
                                poot_kernelgen::FusedInput::Leaf(poot_kernelgen::LeafId::new(0)),
                                poot_kernelgen::FusedInput::LitI32(-1),
                            ],
                        },
                        poot_kernelgen::FusedStep {
                            op: poot_kernelgen::FusedScalarOp::Clz,
                            inputs: vec![poot_kernelgen::FusedInput::Step(
                                poot_kernelgen::StepId::new(0),
                            )],
                        },
                    ],
                    output: poot_kernelgen::FusedInput::Step(poot_kernelgen::StepId::new(1)),
                },
                None,
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

/// The packed E4M3FN movement `op` into `out_shape`, as one generated body.
fn packed_movement(name: &str, out_shape: Vec<usize>, op: MovementOp) -> poot_kernel_ir::Body {
    poot_kernelgen::movement_body(
        name,
        &MovementSpec::new(Storage::BYTE_LANES, out_shape, op),
        None,
    )
    .expect("a valid packed movement")
}

/// Spec 149 Stage 1 SC-004 (CPU-safe compile gate): both format-aware conversion directions and every
/// packed movement lower through the software integer/float path without an LLVM FP8 type or native
/// target instruction.
#[test]
fn e4m3fn_software_conversion_lowers_on_all_gpu_targets() {
    if !have("llc") {
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
        return;
    }
    let dir = out_dir("e4m3fn");
    for (body, name) in [
        (e4m3fn_encode_kernel(), "e4m3fn_encode"),
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
            packed_movement(
                "e4m3fn_repack",
                vec![2, 5],
                MovementOp::Reshape {
                    in_shape: vec![5, 2],
                },
            ),
            "e4m3fn_repack",
        ),
        (
            packed_movement(
                "e4m3fn_transpose",
                vec![5, 2],
                MovementOp::Transpose {
                    in_shape: vec![2, 5],
                    perm: vec![1, 0],
                },
            ),
            "e4m3fn_transpose",
        ),
        (
            packed_movement(
                "e4m3fn_slice",
                vec![2, 3],
                MovementOp::Slice {
                    in_shape: vec![2, 5],
                    axis: 1,
                    start: 1,
                },
            ),
            "e4m3fn_slice",
        ),
        (
            packed_movement(
                "e4m3fn_concat",
                vec![2, 6, 3],
                MovementOp::Concat {
                    axis: 1,
                    in_shapes: vec![vec![2, 2, 3], vec![2, 1, 3], vec![2, 3, 3]],
                },
            ),
            "e4m3fn_concat",
        ),
        (
            packed_movement(
                "e4m3fn_gather",
                vec![2, 2, 5],
                MovementOp::Gather {
                    in_shape: vec![2, 3, 5],
                    axis: 1,
                    index_numel: 2,
                },
            ),
            "e4m3fn_gather",
        ),
        (
            packed_movement(
                "e4m3fn_dynamic_update",
                vec![2, 4, 5],
                MovementOp::UpdateSlice {
                    update_shape: vec![2, 2, 5],
                    axis: 1,
                    index: UpdateIndex::Static(1),
                },
            ),
            "e4m3fn_dynamic_update",
        ),
        (
            packed_movement(
                "e4m3fn_dynamic_update_runtime",
                vec![2, 5],
                MovementOp::UpdateSlice {
                    update_shape: vec![2, 2],
                    axis: 1,
                    index: UpdateIndex::Runtime,
                },
            ),
            "e4m3fn_dynamic_update_runtime",
        ),
        (
            packed_movement(
                "e4m3fn_broadcast",
                vec![2, 3, 5],
                MovementOp::Broadcast {
                    in_shape: vec![2, 1, 5],
                },
            ),
            "e4m3fn_broadcast",
        ),
        (
            packed_movement(
                "e4m3fn_scatter_update",
                vec![4, 2, 5],
                MovementOp::ScatterRows {
                    src_shape: vec![3, 2, 5],
                    window: None,
                },
            ),
            "e4m3fn_scatter_update",
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
            poot_test_util::skip_report("poot-kernelgen", &"no AMD toolchain");
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
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
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
/// nested-loop + f32-accumulator shape of decode matvec.
///
/// Ignored, not skipped (POOT-1120): without `POOT_AIE_LLC` it reports as ignored and the row stays
/// unverified; run by exact name under `nix develop .#npu` with `--run-ignored ignored-only`. With the
/// variable unset the compile is the typed missing-toolchain error, so a run never passes vacuously.
#[test]
#[ignore = "needs the pinned Peano llc (POOT_AIE_LLC, `nix develop .#npu`); unverified without it"]
fn aie_kernels_lower_to_aie2p_via_peano() {
    assert_lowers_to_aie2p(&vadd_loop_kernel(), "vadd_loop");
    // GEMV (M=8, K=16): outer and inner back-edged loops, an accumulator carried across the inner loop,
    // a multiply, and `i*K + j` flat indexing.
    assert_lowers_to_aie2p(&gemv_loop_kernel(8, 16), "gemv_loop");
}

#[test]
fn add_spirv_validates() {
    if !have("llc") || !have("spirv-val") {
        poot_test_util::skip_report("poot-kernelgen", &"llc/spirv-val not on PATH");
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
        poot_test_util::skip_report("poot-kernelgen", &"llc/spirv-val not on PATH");
        return;
    }
    let dir = out_dir("coopmat");
    let out = poot_codegen::artifact_path(&dir, "wmma_coopmat_tile", Target::SpirvVulkan);
    let body = poot_kernel_ir::test_support::wmma_tile("wmma_coopmat_tile");
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

/// `wmma_tile`'s matrix work moved behind a branch on `lane < bound` (the thread index along X): the shape of a
/// kernel that selects some lanes of its workgroup to run a collective matrix op. `bound == 32` selects the first
/// whole subgroup; any other bound splits one.
fn wmma_tile_behind_lane_bound(bound: u64) -> poot_kernel_ir::Body {
    use poot_kernel_ir::{
        BasicBlock, BinOp, BlockId, Constant, IndexAxis, LocalDecl, Operand, Place, Rvalue,
        Statement, SwitchTargets, Terminator,
    };
    let mut body = poot_kernel_ir::test_support::wmma_tile("wmma_tile_behind_lane_bound");
    let lane = poot_kernel_ir::Local {
        index: body.locals.len() as u32,
    };
    let selected = poot_kernel_ir::Local {
        index: lane.index + 1,
    };
    for ty in [Ty::Usize, Ty::Bool] {
        body.locals.push(LocalDecl { ty, mutable: true });
    }
    let mut statements = std::mem::take(&mut body.blocks[0].statements);
    let matrix_work = statements.split_off(1); // statements[0] is the tile index assignment
    body.blocks = vec![
        BasicBlock {
            statements,
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(lane),
                dim: IndexAxis::X,
                target: BlockId { index: 1 },
            },
        },
        BasicBlock {
            statements: vec![Statement::Assign(
                Place::local(selected),
                Rvalue::BinaryOp(
                    BinOp::Lt,
                    Operand::Copy(Place::local(lane)),
                    Operand::Const(Constant::Usize(bound)),
                ),
            )],
            terminator: Terminator::SwitchInt {
                discr: Operand::Copy(Place::local(selected)),
                targets: SwitchTargets {
                    branches: vec![(0, BlockId { index: 3 })],
                    otherwise: BlockId { index: 2 },
                },
            },
        },
        BasicBlock {
            statements: matrix_work,
            terminator: Terminator::Return,
        },
        BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        },
    ];
    body.workgroup_size = [64, 1, 1];
    body
}

/// Card 663 SC-002 on the production path: matrix work under a lane-varying branch is refused by `compile()`
/// itself, on every target, before any IR is emitted.
///
/// Mutation: drop the collective-participation check in `Body::verify_divergent_control`
/// (`poot-kernel-ir`); the body compiles and this row goes red.
#[test]
fn matrix_work_under_a_lane_varying_branch_is_refused_before_codegen() {
    use poot_codegen::EmitError;
    use poot_kernel_ir::VerifyErrorKind;
    let body = wmma_tile_behind_lane_bound(16);
    for target in [
        Target::SpirvVulkan,
        Target::Nvptx,
        Target::AmdGcn(poot_target::AmdArch::gfx1151()),
    ] {
        let Err(EmitError::InvalidBody(error)) = emit_llvm_ir(&body, target) else {
            panic!("{target:?}: a body splitting a subgroup must be refused as invalid");
        };
        assert!(
            matches!(
                error.kind,
                VerifyErrorKind::DivergentFragmentParticipation { .. }
            ),
            "{target:?}: {error}"
        );
    }
}

/// Card 663 SC-002 on the production path, roles: `wmma_tile` with the multiply's A and B operands swapped is
/// refused by `emit_llvm_ir` itself on every target, naming the local and the fragment type it needed.
///
/// Mutation: make `Body::expect_fragment` (`poot-kernel-ir`) accept any declaration; the swapped body lowers
/// and this row goes red.
#[test]
fn swapped_matrix_operands_are_refused_before_codegen() {
    use poot_codegen::EmitError;
    use poot_kernel_ir::{FragmentTy, Statement, VerifyErrorKind, WmmaDtype, WmmaMat, WmmaShape};
    let mut body = poot_kernel_ir::test_support::wmma_tile("wmma_tile_swapped_operands");
    let mma = body.blocks[0]
        .statements
        .iter_mut()
        .find_map(|statement| match statement {
            Statement::WmmaMma { a, b, .. } => Some((a, b)),
            _ => None,
        })
        .expect("the tile multiplies");
    std::mem::swap(mma.0, mma.1);
    // `a` now names the B fragment: it is the first operand the verifier checks.
    let offending = *mma.0;
    for target in [
        Target::SpirvVulkan,
        Target::Nvptx,
        Target::AmdGcn(poot_target::AmdArch::gfx1151()),
    ] {
        let Err(EmitError::InvalidBody(error)) = emit_llvm_ir(&body, target) else {
            panic!("{target:?}: swapped A/B operands must be refused as an invalid body");
        };
        assert_eq!(
            error.kind,
            VerifyErrorKind::FragmentTypeMismatch {
                local: offending,
                expected: FragmentTy::matrix(WmmaMat::A, WmmaDtype::F16, WmmaShape::M16N16K16),
                found: Ty::Fragment(FragmentTy::matrix(
                    WmmaMat::B,
                    WmmaDtype::F16,
                    WmmaShape::M16N16K16
                )),
            },
            "{target:?}"
        );
    }
}

/// Card 663 SC-002 positive side: selecting a whole subgroup (`lane < 32` in a 64-lane workgroup, the warp-zero
/// selection) is accepted and lowers to SPIR-V that validates.
#[test]
fn whole_subgroup_selection_lowers_to_valid_spirv() {
    if !have("llc") || !have("spirv-val") {
        poot_test_util::skip_report("poot-kernelgen", &"llc/spirv-val not on PATH");
        return;
    }
    let dir = out_dir("coopmat-warp-select");
    let out = poot_codegen::artifact_path(&dir, "coopmat_warp_select", Target::SpirvVulkan);
    compile(&wmma_tile_behind_lane_bound(32), Target::SpirvVulkan, &out).unwrap();
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

/// POOT-1237: the per-element ops and the batched-weight tiled GEMM are generated bodies, so each lowers on every
/// GPU target without a device: a broadcast binary over strided leaves, a two-byte float step (widen, compute, round
/// back) and the batched tiled GEMM. A two-byte float body needs a native two-byte lane, which SPIR-V has none of
/// (the planner refuses it there before it generates), so it is lowered on NVPTX and AMDGCN only.
///
/// Mutation: in `check_float_region_tys`, type a two-byte step as `F32`; the narrow body no longer generates (its
/// step register and its operands disagree), and this row and `a_two_byte_float_step_computes_in_f32_and_rounds_back`
/// go red.
#[test]
fn generated_pointwise_and_batched_tiled_bodies_lower_on_every_gpu_target() {
    if !have("llc") {
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
        return;
    }
    use poot_kernelgen::{
        ContractionSpec, DenseTiledContraction, DenseTiledWeight, FusedInput, FusedKernel,
        FusedScalarOp, KernelRequest, Layout, LeafId, LeafTy, RegisterTiling, TiledShape,
        WeightLayout, fused_views,
    };
    let leaf = |index| FusedInput::Leaf(LeafId::new(index));
    let (a_shape, b_shape, out_shape) = ([2usize, 1, 4], [3usize, 1], [2usize, 3, 4]);
    let broadcast = |tys: Vec<LeafTy>, op, lit: Option<f32>| {
        let mut inputs = vec![leaf(0), leaf(1)];
        inputs.extend(lit.map(FusedInput::Lit));
        let kernel = FusedKernel::one_step(op, tys, inputs);
        fused_views(
            "k",
            &out_shape,
            &[&a_shape, &b_shape],
            &[Layout::contiguous(&a_shape), Layout::contiguous(&b_shape)],
            &kernel,
            None,
        )
        .expect("a broadcast binary generates")
    };
    let add = FusedScalarOp::Binary(poot_kernel_ir::BinOp::Add);
    // Three `[17, 24] x [24, 21]` products, one `[K, N]` weight slab each: the register-tiled GEMM's blocks.
    let batched_tiled = || {
        let caps = poot_target::DeviceCaps::wgpu_rdna3_igpu();
        let (blocks, m, k, n) = (3usize, 17usize, 24usize, 21usize);
        let tiling = RegisterTiling::choose(
            TiledShape {
                blocks,
                m_per_block: m,
                out_per_block: n,
                k,
                layout: WeightLayout::Kn,
                segments: 1,
            },
            &caps,
        );
        let (tile_rows, tile_cols) = tiling.output_tile();
        let tiles = blocks * m.div_ceil(tile_rows) * n.div_ceil(tile_cols);
        let lanes = tiling.lanes() * tiling.lanes();
        let request =
            KernelRequest::Contraction(ContractionSpec::DenseTiled(DenseTiledContraction {
                weight: DenseTiledWeight::KnF32,
                schedule: tiling.schedule(),
                epilogue: None,
                grid: [(tiles * lanes) as u32, 1, 1],
                meta: vec![k as u32, n as u32, m as u32, tiles as u32, 0],
            }));
        let limits = poot_kernelgen::BodyLimits {
            max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
            max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
        };
        poot_kernelgen::generate(&request, &caps, &limits)
            .expect("a batched tiled GEMM generates")
            .body
    };
    // (name, body, lowers on SPIR-V).
    let bodies = [
        (
            "bcast_f32",
            broadcast(vec![LeafTy::F32; 2], add, None),
            true,
        ),
        (
            "bcast_bf16",
            broadcast(vec![LeafTy::BF16; 2], add, None),
            false,
        ),
        (
            "bcast_f16",
            broadcast(vec![LeafTy::F16; 2], add, None),
            false,
        ),
        ("tiled_batched", batched_tiled(), true),
    ];
    let dir = out_dir("pointwise-lowers");
    let mut targets = vec![Target::SpirvVulkan, Target::Nvptx];
    if amd_toolchain_available() {
        targets.push(Target::AmdGcn(poot_target::AmdArch::gfx1151()));
    }
    for (name, body, on_spirv) in &bodies {
        for &target in &targets {
            if target == Target::SpirvVulkan && !on_spirv {
                continue;
            }
            let out = poot_codegen::artifact_path(&dir, name, target);
            compile(body, target, &out)
                .unwrap_or_else(|error| panic!("{name} {target:?}: {error}"));
            assert!(
                std::fs::metadata(&out).unwrap().len() > 0,
                "{name} {target:?}"
            );
        }
    }
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
        poot_test_util::skip_report("poot-kernelgen", &"no AMD toolchain");
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
        poot_test_util::skip_report("poot-kernelgen", &"no AMD toolchain");
        return;
    }
    let target = Target::AmdGcn(poot_target::AmdArch::gfx1151());
    let dir = out_dir("amdgpu-fence");
    for (name, body) in [
        (
            "wgfence_amd",
            poot_kernel_ir::test_support::workgroup_fence_spin_wait("wgfence_amd", 128),
        ),
        (
            "devfence_amd",
            poot_kernel_ir::test_support::cross_block_fence_spin_wait_coop("devfence_amd"),
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
        poot_test_util::skip_report("poot-kernelgen", &"no AMD toolchain");
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

/// Card 145 follow-on (spec 130 FR-004): a Bf16 fragment tile lowers to the RDNA3 bf16 WMMA intrinsic for
/// `Target::AmdGcn(gfx1151)`. `emit_llvm_ir` is pure Rust (no llc, no GPU), so this always runs. It guards
/// that a Bf16 fragment codegens the intrinsic, not an unresolved extern. With the AMD toolchain present it
/// also lowers the IR through llc to an HSACO; otherwise it skips only that half.
#[test]
fn amd_wmma_bf16_tile_lowers_to_the_bf16_intrinsic() {
    use poot_target::AmdArch;
    let target = Target::AmdGcn(AmdArch::gfx1151());
    let body = poot_kernel_ir::test_support::wmma_tile_of("k", poot_kernel_ir::WmmaDtype::Bf16);
    let ir = emit_llvm_ir(&body, target).expect("emit AMD WMMA IR");
    // The RDNA3 WMMA 16x16x16 bf16-in/f32-accumulate intrinsic (emit.rs `emit_wmma_mma_amdgcn`).
    assert!(
        ir.contains("llvm.amdgcn.wmma.f32.16x16x16.bf16"),
        "a Bf16 fragment tile must lower to the RDNA3 bf16 WMMA intrinsic; IR:\n{ir}"
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
