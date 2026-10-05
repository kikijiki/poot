//! Golden-IR tests for `poot_codegen::emit` (moved from `src/emit.rs`'s own `#[cfg(test)]` module, card
//! 671): that module called `poot_test_util::kernel_fixtures::emit_llvm_ir`, and `poot-test-util` depends on
//! `poot-codegen` (for the real, still-pub `emit_llvm_ir_marked` the golden-IR helper wraps), so building it
//! as part of poot-codegen's own lib-test unit compiled two different instances of `poot-codegen` into one
//! binary (`Target`/`EmitError` each becoming two distinct, mismatched types). An integration test under
//! `tests/` always treats its own crate as an ordinary external dependency, which has no such cycle.

use poot_codegen::{CompileError, EmitError, Target, artifact_path, compile};
use poot_kernel_ir::fixtures;
use poot_target::AmdArch;
use poot_test_util::kernel_fixtures::{
    e4m3fn_decode_kernel, e4m3fn_encode_kernel, emit_llvm_ir, gemv_loop_kernel, vadd_loop_kernel,
    wmma_tile,
};

#[test]
fn add_nvptx_shape() {
    let ir = emit_llvm_ir(&fixtures::add_kernel(), Target::Nvptx).unwrap();
    assert!(ir.contains("define ptx_kernel void @add("));
    assert!(ir.contains("ptr addrspace(1) %p1, i64 %n1"));
    assert!(ir.contains("llvm.nvvm.read.ptx.sreg.tid.x"));
    assert!(ir.contains("getelementptr inbounds float, ptr addrspace(1) %p1"));
    assert!(ir.contains("fadd float"));
    assert!(ir.contains("ret void"));
}

#[test]
fn add_spirv_shape() {
    let ir = emit_llvm_ir(&fixtures::add_kernel(), Target::SpirvVulkan).unwrap();
    assert!(ir.contains("define void @main() #0"));
    assert!(ir.contains("spirv.VulkanBuffer"));
    assert!(ir.contains("llvm.spv.thread.id.i32"));
    assert!(ir.contains("getpointer.p11"));
    assert!(ir.contains("\"hlsl.numthreads\"=\"64,1,1\""));
    // length buffer read-only variant present.
    assert!(ir.contains("a0i32_12_0t"));
}

#[test]
fn exact_i32_pointwise_emits_integer_arithmetic_and_unsigned_comparison() {
    let signed = [
        (poot_kernel_ir::BinOp::Add, "add i32"),
        (poot_kernel_ir::BinOp::Sub, "sub i32"),
        (poot_kernel_ir::BinOp::Mul, "mul i32"),
        (poot_kernel_ir::BinOp::Max, "llvm.smax.i32"),
        (poot_kernel_ir::BinOp::Ge, "icmp sge i32"),
    ];
    let geu = poot_kernelgen::binary_scalar_i32_geu_grid("exact_i32_geu", i32::MIN, None);
    let a_shape = [2, 1];
    let b_shape = [1, 3];
    let out_shape = [2, 3];
    let a_layout = poot_kernelgen::Layout::contiguous(&a_shape);
    let b_layout = poot_kernelgen::Layout::contiguous(&b_shape);
    let broadcast_add = poot_kernelgen::binary_broadcast_dt_views_grid(
        "exact_i32_broadcast_add",
        poot_kernel_ir::BinOp::Add,
        poot_kernel_ir::Ty::I32,
        &out_shape,
        &a_shape,
        &a_layout,
        &b_shape,
        &b_layout,
        None,
    );
    let broadcast_geu = poot_kernelgen::binary_broadcast_i32_geu_views_grid(
        "exact_i32_broadcast_geu",
        &out_shape,
        &a_shape,
        &a_layout,
        &b_shape,
        &b_layout,
        None,
    );
    for target in [
        Target::SpirvVulkan,
        Target::Nvptx,
        Target::AmdGcn(AmdArch::gfx1151()),
    ] {
        for (op, opcode) in signed {
            let body =
                poot_kernelgen::binary_scalar_i32_grid("exact_i32_signed", op, i32::MIN, None);
            let ir = emit_llvm_ir(&body, target).unwrap();
            assert!(ir.contains(opcode), "{op:?} {target:?}: {ir}");
            assert!(!ir.contains(" float"), "{op:?} {target:?}: {ir}");
            assert!(!ir.contains("sitofp"), "{op:?} {target:?}: {ir}");
            assert!(!ir.contains(" nsw "), "{op:?} {target:?}: {ir}");
            assert!(!ir.contains(" nuw "), "{op:?} {target:?}: {ir}");
        }

        let geu_ir = emit_llvm_ir(&geu, target).unwrap();
        assert!(geu_ir.contains("icmp uge i32"), "{target:?}: {geu_ir}");
        assert!(!geu_ir.contains("icmp sge i32"), "{target:?}: {geu_ir}");
        assert!(!geu_ir.contains("fcmp"), "{target:?}: {geu_ir}");

        let broadcast_add_ir = emit_llvm_ir(&broadcast_add, target).unwrap();
        assert!(
            broadcast_add_ir.contains("add i32"),
            "{target:?}: {broadcast_add_ir}"
        );
        assert!(
            !broadcast_add_ir.contains("fadd float"),
            "{target:?}: {broadcast_add_ir}"
        );
        let broadcast_geu_ir = emit_llvm_ir(&broadcast_geu, target).unwrap();
        assert!(
            broadcast_geu_ir.contains("icmp uge i32"),
            "{target:?}: {broadcast_geu_ir}"
        );
        assert!(
            !broadcast_geu_ir.contains("fcmp"),
            "{target:?}: {broadcast_geu_ir}"
        );
    }
}

#[test]
fn e4m3fn_software_conversion_is_target_neutral_ir() {
    for target in [
        Target::SpirvVulkan,
        Target::Nvptx,
        Target::AmdGcn(AmdArch::gfx1151()),
    ] {
        let encode = emit_llvm_ir(&e4m3fn_encode_kernel(), target).unwrap();
        assert!(encode.contains("fcmp uno float"), "{target:?}");
        assert!(encode.contains("select i1"), "{target:?}");
        assert!(!encode.to_lowercase().contains("fp8"), "{target:?}");

        let decode = emit_llvm_ir(&e4m3fn_decode_kernel(), target).unwrap();
        assert!(decode.contains("and i32"), "{target:?}");
        assert!(decode.contains("bitcast i32"), "{target:?}");
        assert!(!decode.to_lowercase().contains("fp8"), "{target:?}");
    }
}

#[test]
fn add_amdgcn_shape() {
    // The IR contract: `X` is the global work-item id, `LocalX` the per-wavefront thread id (v0), `GroupX` the
    // workgroup id. The AMDGPU backend lowers `@llvm.amdgcn.workitem.id.x` to v0, so the global id is assembled in
    // IR as `workgroup_id * workgroup_size + workitem_id`; without that, every thread in a workgroup writes
    // c[0..workgroup_size-1] and the rest of `c` is left untouched. This is the regression test for that
    // landmine.
    let ir = emit_llvm_ir(&fixtures::add_kernel(), Target::AmdGcn(AmdArch::gfx1151())).unwrap();
    // kernel signature + datalayout
    assert!(ir.contains("define amdgpu_kernel void @add("));
    assert!(ir.contains("ptr addrspace(1) %p1, i64 %n1"));
    assert!(ir.contains("amdgcn-amd-amdhsa"));
    // the global X arm composes workgroup_id * workgroup_size + workitem_id
    assert!(ir.contains("llvm.amdgcn.workitem.id.x"));
    assert!(ir.contains("llvm.amdgcn.workgroup.id.x"));
    // The add_kernel fixture uses workgroup_size = [64, 1, 1], so the multiplier in the IR is the literal `64`.
    assert!(
        ir.contains("mul i32 "),
        "global-X must multiply workgroup_id by workgroup_size: {ir}"
    );
    assert!(
        ir.contains(", 64"),
        "global-X multiplier must be the workgroup_size X (64 in the add fixture): {ir}"
    );
    assert!(
        ir.contains("add i32 "),
        "global-X must add workitem_id to the workgroup base: {ir}"
    );
    // and the result is zext'd to i64 (the IR's usize width)
    assert!(ir.contains("zext i32 "));
}

// ---- AIE core (Peano aie2p) target (card 089 / spec 057) -----------------------------------

#[test]
fn vadd_loop_aie_shape() {
    let ir = emit_llvm_ir(&vadd_loop_kernel(), Target::AieCore).unwrap();
    // a plain extern-C-shape function, flat `ptr` args, i32 lengths - no ptx_kernel, no addrspace(1).
    assert!(
        ir.contains("define void @vadd_loop(ptr %p1, i32 %n1, ptr %p2, i32 %n2, ptr %p3, i32 %n3)")
    );
    assert!(!ir.contains("ptx_kernel"));
    assert!(!ir.contains("addrspace(1)"));
    // usize is i32 on the 32-bit core; the loop index compares unsigned.
    assert!(ir.contains("icmp ult i32"));
    // flat-pointer GEP with an i32 index, the elementwise add, the store-back.
    assert!(ir.contains("getelementptr inbounds float, ptr %p1, i32"));
    assert!(ir.contains("fadd float"));
    // sequential loop: a single core walks the tile, so NO SPMD thread-index intrinsics appear.
    assert!(!ir.contains("read.ptx.sreg"));
    assert!(!ir.contains("thread.id"));
    // the length is taken from the scalar arg, not a buffer load.
    assert!(ir.contains("%n3"));
}

#[test]
fn gemv_loop_aie_shape() {
    // The harder nested-loop + accumulator kernel emits cleanly for AIE-core (no thread-index / barrier
    // / LDS), with the f32 accumulate (fmul + fadd) and flat `i*K+j` indexing the matvec needs.
    let ir = emit_llvm_ir(&gemv_loop_kernel(8, 16), Target::AieCore).unwrap();
    assert!(ir.contains("define void @gemv_loop(ptr %p1, i32 %n1"));
    assert!(ir.contains("fmul float"));
    assert!(ir.contains("fadd float"));
    assert!(ir.contains("getelementptr inbounds float, ptr %p1, i32"));
    assert!(!ir.contains("thread.id") && !ir.contains("read.ptx.sreg"));
}

#[test]
fn aie_rejects_thread_index_by_name() {
    // The SPMD elementwise kernel uses thread_index, which a single AIE core cannot provide.
    let err = emit_llvm_ir(&fixtures::add_kernel(), Target::AieCore).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("thread_index on AIE-core"),
        "diagnostic must name the construct: {msg}"
    );
}

#[test]
fn aie_rejects_bf16_by_name() {
    // bf16 is out of the f32-only PoC subset; the diagnostic must name it (FR-004).
    use poot_kernel_ir::*;
    let body = Body::new(
        "bf16_local",
        0,
        vec![
            LocalDecl {
                ty: Ty::Unit,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::BF16,
                mutable: false,
            },
        ],
        vec![BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        }],
    );
    let err = emit_llvm_ir(&body, Target::AieCore).unwrap_err();
    assert!(err.to_string().contains("bf16"), "{err}");
}

#[test]
fn exp_amdgcn_emits_llvm_exp() {
    // MathOp::Exp on AmdGcn emits `llvm.exp.f32` (resolved to __ocml_exp_f32 by clang + device-libs at compile
    // time), the OCML follow-on for card 106. Kernel shape: `y[i] = exp(x[i])`, as poot_test_util's square_kernel.
    use poot_kernel_ir::*;
    let l = |i| Local { index: i };
    let (x, y, i, len, cmp, r) = (l(1), l(2), l(3), l(4), l(5), l(6));
    let elem = |p: Local| Place {
        local: p,
        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(i)],
    };
    let slice_f32 = |m| Ty::Ref {
        mutable: m,
        pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
    };
    let body = Body::new(
        "exp_kernel",
        2,
        vec![
            LocalDecl {
                ty: Ty::Unit,
                mutable: false,
            },
            LocalDecl {
                ty: slice_f32(false),
                mutable: false,
            },
            LocalDecl {
                ty: slice_f32(true),
                mutable: true,
            },
            LocalDecl {
                ty: Ty::Usize,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::Usize,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::Bool,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::F32,
                mutable: false,
            },
        ],
        vec![
            BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(i),
                    dim: IndexAxis::X,
                    target: BlockId { index: 1 },
                },
            },
            BasicBlock {
                statements: vec![
                    Statement::Assign(Place::local(len), Rvalue::Len(Place::local(y))),
                    Statement::Assign(
                        Place::local(cmp),
                        Rvalue::BinaryOp(
                            BinOp::Lt,
                            Operand::Copy(Place::local(i)),
                            Operand::Copy(Place::local(len)),
                        ),
                    ),
                ],
                terminator: Terminator::SwitchInt {
                    discr: Operand::Copy(Place::local(cmp)),
                    targets: SwitchTargets {
                        branches: vec![(0, BlockId { index: 3 })],
                        otherwise: BlockId { index: 2 },
                    },
                },
            },
            BasicBlock {
                statements: vec![
                    Statement::Assign(
                        Place::local(r),
                        Rvalue::MathUnary(MathOp::Exp, Operand::Copy(elem(x))),
                    ),
                    Statement::Assign(elem(y), Rvalue::Use(Operand::Copy(Place::local(r)))),
                ],
                terminator: Terminator::Return,
            },
            BasicBlock {
                statements: vec![],
                terminator: Terminator::Return,
            },
        ],
    );
    let ir = emit_llvm_ir(&body, Target::AmdGcn(AmdArch::gfx1151())).unwrap();
    assert!(
        ir.contains("llvm.exp.f32"),
        "AmdGcn exp must emit llvm.exp.f32 (ocml resolves it): {ir}"
    );
    assert!(
        ir.contains("declare float @llvm.exp.f32"),
        "llvm.exp.f32 must be declared: {ir}"
    );
}

#[test]
fn exp_aie_emits_software_exp() {
    // MathOp::Exp on AieCore emits the software inline sequence (magic rounding + Horner poly + i32 exponent
    // scaling) with no external @expf or llvm.exp.f32 call (Peano's aie2p libm has no expf). Kernel shape:
    // sequential loop y[i] = exp(x[i]), as vadd_loop_kernel.
    use poot_kernel_ir::*;
    let l = |idx| Local { index: idx };
    let (x, y, i, len, cmp, r, i_new) = (l(1), l(2), l(3), l(4), l(5), l(6), l(7));
    let slice_f32 = |m| Ty::Ref {
        mutable: m,
        pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
    };
    let elem_x = Place {
        local: x,
        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(i)],
    };
    let elem_y = Place {
        local: y,
        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(i)],
    };
    let body = Body::new(
        "exp_loop",
        2,
        vec![
            LocalDecl {
                ty: Ty::Unit,
                mutable: false,
            },
            LocalDecl {
                ty: slice_f32(false),
                mutable: false,
            },
            LocalDecl {
                ty: slice_f32(true),
                mutable: true,
            },
            LocalDecl {
                ty: Ty::Usize,
                mutable: true,
            },
            LocalDecl {
                ty: Ty::Usize,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::Bool,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::F32,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::Usize,
                mutable: false,
            },
        ],
        vec![
            // bb0: len = y.len(); i = 0; goto bb1.
            BasicBlock {
                statements: vec![
                    Statement::Assign(Place::local(len), Rvalue::Len(Place::local(y))),
                    Statement::Assign(
                        Place::local(i),
                        Rvalue::Use(Operand::Const(Constant::Usize(0))),
                    ),
                ],
                terminator: Terminator::Goto {
                    target: BlockId { index: 1 },
                },
            },
            // bb1: cmp = i < len; if 0 -> bb3, else -> bb2.
            BasicBlock {
                statements: vec![Statement::Assign(
                    Place::local(cmp),
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        Operand::Copy(Place::local(i)),
                        Operand::Copy(Place::local(len)),
                    ),
                )],
                terminator: Terminator::SwitchInt {
                    discr: Operand::Copy(Place::local(cmp)),
                    targets: SwitchTargets {
                        branches: vec![(0, BlockId { index: 3 })],
                        otherwise: BlockId { index: 2 },
                    },
                },
            },
            // bb2: r = exp(x[i]); y[i] = r; i += 1; back-edge to bb1.
            BasicBlock {
                statements: vec![
                    Statement::Assign(
                        Place::local(r),
                        Rvalue::MathUnary(MathOp::Exp, Operand::Copy(elem_x)),
                    ),
                    Statement::Assign(elem_y, Rvalue::Use(Operand::Copy(Place::local(r)))),
                    Statement::Assign(
                        Place::local(i_new),
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            Operand::Copy(Place::local(i)),
                            Operand::Const(Constant::Usize(1)),
                        ),
                    ),
                    Statement::Assign(
                        Place::local(i),
                        Rvalue::Use(Operand::Copy(Place::local(i_new))),
                    ),
                ],
                terminator: Terminator::Goto {
                    target: BlockId { index: 1 },
                },
            },
            // bb3: return.
            BasicBlock {
                statements: vec![],
                terminator: Terminator::Return,
            },
        ],
    );
    let ir = emit_llvm_ir(&body, Target::AieCore).unwrap();
    // Software exp uses: clamping (fcmp + select), magic rounding (fadd/fsub),
    // Horner poly (fmul + fadd), and 2^n scaling (add i32 / shl i32 / bitcast i32 %.. to float).
    assert!(
        ir.contains("shl i32"),
        "AIE exp must emit shl i32 for 2^n scaling: {ir}"
    );
    // bitcast format is "bitcast i32 %tN to float" (register-qualified, not type-to-type)
    assert!(
        ir.contains("bitcast i32 ") && ir.contains(" to float"),
        "AIE exp must emit 'bitcast i32 %reg to float' for 2^n scaling: {ir}"
    );
    assert!(
        ir.contains("fptosi float"),
        "AIE exp must emit fptosi float for n = round(x*log2e): {ir}"
    );
    // Must NOT call external expf or llvm.exp.f32 (neither exists in Peano's aie2p libm).
    assert!(
        !ir.contains("@expf") && !ir.contains("@__expf"),
        "AIE exp must not emit external expf call: {ir}"
    );
    assert!(
        !ir.contains("llvm.exp.f32"),
        "AIE exp must not use llvm.exp.f32 (unsupported by Peano AIE2p): {ir}"
    );
}

/// Card 530 SC-001: one generator (`wmma_tile`) produces one fragment `Body`, and that same `Body`
/// lowers on every implemented target - AMDGCN WMMA, NVPTX `mma.sync`/`wmma`, and SPIR-V cooperative
/// matrix - with no per-target `Body` variant. Real device evidence (not just this compile-time check)
/// is `poot-kernelgen`'s `coopmat_dispatch_probe` (RADV coopmat) and `poot-rocm-gpu`'s
/// `tests::resident::probe_*_wmma_rocm` (AMDGCN WMMA), both of which dispatch this same generator (or
/// its production `matmul_tensorcore` sibling) on real hardware and compare against the CPU oracle.
#[test]
fn wmma_tile_is_one_body_lowered_on_every_target() {
    let body = wmma_tile("wmma_tile_neutral");
    for (target, marker) in [
        (Target::SpirvVulkan, "__spirv_CooperativeMatrixMulAddKHR"),
        (
            Target::AmdGcn(AmdArch::gfx1151()),
            "llvm.amdgcn.wmma.f32.16x16x16.f16",
        ),
        (Target::Nvptx, "llvm.nvvm.wmma.m16n16k16.mma.row.row"),
    ] {
        let ir = emit_llvm_ir(&body, target)
            .unwrap_or_else(|e| panic!("wmma_tile must lower on {target:?}: {e}"));
        assert!(
            ir.contains(marker),
            "wmma_tile on {target:?} must contain {marker:?}: {ir}"
        );
    }

    // Card 530: the markers above only prove an intrinsic of the right *name* was
    // emitted, not that A and B were not swapped (no layout is asserted on any of the three paths).
    // One A/B-layout fact per target, so the card's own SC-001 mutation ("swap the A/B fragment
    // layout in one target's lowering") can flip a row red:
    //
    // SPIR-V: the coopmat type's Use operand (0 = MatrixA, 1 = MatrixB, spec 154) is baked into the
    // A load's and the B load's result type text, distinct from each other.
    let spirv_ir = emit_llvm_ir(&body, Target::SpirvVulkan).unwrap();
    assert!(
        spirv_ir.contains("target(\"spirv.CooperativeMatrixKHR\", half, 3, 16, 16, 0)"),
        "SPIR-V A fragment must be Use=0 (MatrixA): {spirv_ir}"
    );
    assert!(
        spirv_ir.contains("target(\"spirv.CooperativeMatrixKHR\", half, 3, 16, 16, 1)"),
        "SPIR-V B fragment must be Use=1 (MatrixB): {spirv_ir}"
    );

    // NVPTX: the `load.a`/`load.b` intrinsic names are which the NVVM WMMA fragment table keys A vs
    // B on, both with the F16 dtype token.
    let nvptx_ir = emit_llvm_ir(&body, Target::Nvptx).unwrap();
    assert!(
        nvptx_ir.contains("llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0"),
        "NVPTX A fragment must load via load.a: {nvptx_ir}"
    );
    assert!(
        nvptx_ir.contains("llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0"),
        "NVPTX B fragment must load via load.b: {nvptx_ir}"
    );

    // AMDGCN: `emit_wmma_load_amdgcn`'s row-vs-col per-lane offset arithmetic differs by `which` (see
    // its doc): the A load precomputes `row * stride` exactly once (a register-operand `mul`) and
    // then only adds per-lane constants, while the B load has no such precompute and instead
    // recomputes `j * stride` fresh each of the 15 nonzero lanes (15 literal-operand `mul`s). Splitting
    // on the per-load `workitem.id.x` call (emitted once per `WmmaLoad`, A's call before B's in
    // program order) isolates each load's block; counting `mul i32` occurrences in each catches either
    // half of a swap (A losing its one precompute, or B gaining one it should not have), not just a
    // clean full exchange.
    // `emit_wmma_store_amdgcn` (the store epilogue) also calls `workitem.id.x()` once for its own lane
    // addressing, so the function has 3 such calls total (A load, B load, store); split into 4 parts
    // and stop at the 3rd so `b_block` holds only the B load's own code, not the unrelated store tail.
    let amdgcn_ir = emit_llvm_ir(&body, Target::AmdGcn(AmdArch::gfx1151())).unwrap();
    let lane_call = "call i32 @llvm.amdgcn.workitem.id.x()";
    let mut parts = amdgcn_ir.splitn(4, lane_call);
    let _before = parts.next().expect("text before the first lane call");
    let a_block = parts
        .next()
        .expect("A load's block, up to the B load's lane call");
    let b_block = parts
        .next()
        .expect("B load's block, up to the store's lane call");
    let mul_count = |s: &str| s.matches("mul i32").count();
    assert_eq!(
        mul_count(a_block),
        1,
        "AMDGCN A fragment must precompute row*stride exactly once: {a_block}"
    );
    assert_eq!(
        mul_count(b_block),
        15,
        "AMDGCN B fragment must compute j*stride fresh for each of its 15 nonzero lanes: {b_block}"
    );
}

#[test]
fn wmma_amdgcn_emits_wmma_intrinsic() {
    // `wmma_tile` is card 530's target-neutral F16 fragment body; on AMDGCN it must emit the F16 WMMA
    // intrinsic (gfx1151 wave32: v16f16 A/B, v8f32 C). The Bf16 production shape (v16i16 A/B, the same
    // register packing, different element bits) is `matmul_tensorcore`'s coverage in `tests/llc.rs`'s
    // `amd_wmma_matmul_lowers_to_wmma_intrinsic`.
    let body = wmma_tile("wmma_amdgcn_test");
    let ir = emit_llvm_ir(&body, Target::AmdGcn(AmdArch::gfx1151())).unwrap();
    // Must emit the typed AMDGPU WMMA MMA intrinsic (gfx1151 wave32: v16f16 A/B, v8f32 C).
    assert!(
        ir.contains("llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16"),
        "AmdGcn WMMA must emit llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16: {ir}"
    );
    // Must NOT use NVVM intrinsics.
    assert!(
        !ir.contains("llvm.nvvm.wmma"),
        "AmdGcn WMMA must not use NVVM intrinsics: {ir}"
    );
    // Must use workitem.id.x for lane addressing.
    assert!(
        ir.contains("llvm.amdgcn.workitem.id.x"),
        "AmdGcn WMMA must use workitem.id.x for lane addressing: {ir}"
    );
    // Must declare the typed WMMA intrinsic with v16f16 inputs.
    assert!(
        ir.contains("declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16(<16 x half>, <16 x half>, <8 x float>)"),
        "AmdGcn WMMA must declare the typed intrinsic: {ir}"
    );
    // A/B fragments must be <16 x half> (not <8 x half>).
    assert!(
        ir.contains("<16 x half>"),
        "AmdGcn WMMA must use <16 x half> for A/B fragments: {ir}"
    );
    // Must use lane ID for addressing.
    assert!(
        ir.contains("and i32"),
        "AmdGcn WMMA load must mask lane ID: {ir}"
    );
}

/// Spec 120 SC-003: a CDNA (gfx942, MFMA) target, or any non-RDNA3 AMD family, must reject the WMMA path with a
/// typed `Unsupported` error instead of emitting the RDNA3 intrinsic or hitting a backend "cannot select"
/// crash. RDNA4 (gfx1201) is rejected too.
#[test]
fn wmma_unsupported_amd_family_is_typed_error() {
    for gfx in ["gfx942", "gfx1201", "gfx906"] {
        let body = wmma_tile("wmma_unsupported_test");
        let target = Target::AmdGcn(AmdArch::new(gfx, 64));
        let err = emit_llvm_ir(&body, target)
            .expect_err(&format!("WMMA on {gfx} must be a typed error, not IR"));
        let EmitError::Unsupported(msg) = err else {
            panic!("expected EmitError::Unsupported, got {err:?}")
        };
        assert!(
            msg.contains("unsupported") && msg.contains(gfx),
            "error must name the unsupported target {gfx}: {msg}"
        );
    }
}

/// Card 154: the SPIR-V cooperative-matrix WMMA arms emit `__spirv_CooperativeMatrix*KHR`/
/// `__spirv_CompositeConstruct` builtin calls. `spirv-val` validation of the compiled and fixed output is
/// `tests/llc.rs`'s `coopmat_tile_spirv_validates`.
#[test]
fn coopmat_tile_emits_real_builtin_calls() {
    let body = wmma_tile("wmma_coopmat_test");
    let ir = emit_llvm_ir(&body, Target::SpirvVulkan).unwrap();
    assert!(
        ir.contains("@_Z32__spirv_CooperativeMatrixLoadKHR_1"),
        "missing coopmat A load: {ir}"
    );
    assert!(
        ir.contains("@_Z32__spirv_CooperativeMatrixLoadKHR_2"),
        "missing coopmat B load: {ir}"
    );
    assert!(
        ir.contains("@_Z27__spirv_CompositeConstruct"),
        "missing coopmat zero-accumulator constructor: {ir}"
    );
    assert!(
        ir.contains("@_Z34__spirv_CooperativeMatrixMulAddKHR"),
        "missing coopmat mma: {ir}"
    );
    assert!(
        ir.contains("@_Z33__spirv_CooperativeMatrixStoreKHR"),
        "missing coopmat store: {ir}"
    );
    assert!(
        ir.contains("target(\"spirv.CooperativeMatrixKHR\", half, 3, 16, 16, 0)"),
        "A fragment must be a 16x16x16 Subgroup-scope half coopmat type: {ir}"
    );
    assert!(
        ir.contains("target(\"spirv.CooperativeMatrixKHR\", float, 3, 16, 16, 2)"),
        "accumulator fragment must be a 16x16x16 Subgroup-scope float coopmat type: {ir}"
    );
}

/// Card 530: F16 fragments (`wmma_tile`'s target-neutral dtype) now lower on every implemented target;
/// the one remaining WMMA dtype/target mismatch is Bf16 fragments on SpirvVulkan (RADV has no
/// `VK_KHR_shader_bfloat16`) - a typed error, not silently-wrong codegen.
#[test]
fn wmma_dtype_target_mismatch_is_typed_error() {
    // `matmul_tensorcore` (Bf16 fragments, the NVPTX/AMDGCN production convention) declares BF16-typed
    // slice params, which an earlier check already rejects on SpirvVulkan (bf16 buffers are NVPTX-only;
    // SPIR-V needs SPV_KHR_bfloat16). `wmma_check_target`'s (SpirvVulkan, Bf16) arm is therefore
    // unreachable via this route; assert the mismatch is rejected (by whichever check fires first)
    // rather than pin the message.
    let bf16_body = poot_kernelgen::matmul_tensorcore(
        "wmma_bf16_wrong_target",
        poot_kernel_ir::Ty::F32,
        &[16, 16],
        &[16, 16],
        &[16, 16],
    )
    .expect("matmul_tensorcore precondition");
    let err = emit_llvm_ir(&bf16_body, Target::SpirvVulkan)
        .expect_err("Bf16 fragments on SpirvVulkan must be a typed error");
    let EmitError::Unsupported(msg) = err else {
        panic!("expected EmitError::Unsupported, got {err:?}")
    };
    assert!(
        msg.to_lowercase().contains("bf16"),
        "error must name the bf16 mismatch: {msg}"
    );
}

/// Card 530: a `WmmaLoad` whose `dtype` does not match its tile's declared element
/// type must be a typed error, not a silent bit reinterpretation. Built directly (not through
/// `poot_kernelgen`, which never produces this combination) with `dtype: F16` over a BF16 tile, on
/// AmdGcn - the one target where both dtypes individually pass `wmma_check_target`, so this specific
/// cross-check is what fires, not an unrelated target/dtype rejection.
#[test]
fn wmma_dtype_tile_mismatch_is_typed_error() {
    use poot_kernel_ir::{
        BasicBlock, Body, Constant, Local, LocalDecl, Operand, Place, ProjectionElem, Rvalue,
        Statement, Terminator, Ty, WmmaDtype, WmmaShape,
    };
    let tile_param = Ty::Ref {
        mutable: false,
        pointee: Box::new(Ty::Slice(Box::new(Ty::BF16))),
    };
    let locals = vec![
        LocalDecl {
            ty: Ty::Unit,
            mutable: false,
        },
        LocalDecl {
            ty: tile_param,
            mutable: false,
        },
        LocalDecl {
            ty: Ty::Usize,
            mutable: false,
        },
        LocalDecl {
            ty: Ty::F16,
            mutable: false,
        },
    ];
    let idx = Local { index: 2 };
    let dst = Local { index: 3 };
    let block = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(idx),
                Rvalue::Use(Operand::Const(Constant::Usize(0))),
            ),
            Statement::WmmaLoad {
                which: poot_kernel_ir::WmmaMat::A,
                dtype: WmmaDtype::F16, // the tile is BF16: a deliberate mismatch
                shape: WmmaShape::M16N16K16,
                tile: Place {
                    local: Local { index: 1 },
                    projection: vec![ProjectionElem::Deref, ProjectionElem::Index(idx)],
                },
                stride: 16,
                dst,
            },
        ],
        terminator: Terminator::Return,
    };
    let body = Body::new("wmma_dtype_tile_mismatch", 1, locals, vec![block]);
    let err = emit_llvm_ir(&body, Target::AmdGcn(AmdArch::gfx1151()))
        .expect_err("a dtype/tile-element mismatch must be a typed error");
    let EmitError::Unsupported(msg) = err else {
        panic!("expected EmitError::Unsupported, got {err:?}")
    };
    assert!(
        msg.contains("F16") && msg.contains("BF16"),
        "error must name both the declared dtype and the tile's element type: {msg}"
    );
}

/// R468-005: `_2 = 1.0` where `_2` is not a declared local (baseline: an index panic inside the emitter), and
/// `_2: usize = 1.0f32` (baseline: llc crashed for SPIR-V and compiled silently for NVPTX).
fn malformed_bodies() -> [(
    &'static str,
    poot_kernel_ir::Body,
    poot_kernel_ir::VerifyErrorKind,
); 2] {
    use poot_kernel_ir::{
        BasicBlock, Body, Constant, Local, LocalDecl, Operand, Place, Rvalue, Statement,
        Terminator, Ty, VerifyErrorKind,
    };
    let buffer = Ty::Ref {
        mutable: true,
        pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
    };
    let body = |extra: Option<Ty>, assigned: u32| {
        let locals = [Ty::Unit, buffer.clone()]
            .into_iter()
            .chain(extra)
            .map(|ty| LocalDecl { ty, mutable: true })
            .collect();
        let assign = Statement::Assign(
            Place::local(Local { index: assigned }),
            Rvalue::Use(Operand::Const(Constant::F32(1.0))),
        );
        let block = BasicBlock {
            statements: vec![assign],
            terminator: Terminator::Return,
        };
        Body::new("malformed", 1, locals, vec![block])
    };
    [
        (
            "undeclared local",
            body(None, 9),
            VerifyErrorKind::UndeclaredLocal(Local { index: 9 }),
        ),
        (
            "f32 into a usize local",
            body(Some(Ty::Usize), 2),
            VerifyErrorKind::TypeMismatch {
                what: "assigned value",
                expected: Ty::Usize,
                found: Ty::F32,
            },
        ),
    ]
}

fn every_target() -> [Target; 4] {
    [
        Target::SpirvVulkan,
        Target::Nvptx,
        Target::AieCore,
        Target::AmdGcn(AmdArch::gfx1151()),
    ]
}

#[test]
fn a_malformed_body_is_a_typed_error_on_every_target() {
    for (name, body, expected) in malformed_bodies() {
        for target in every_target() {
            let err = emit_llvm_ir(&body, target)
                .expect_err(&format!("{name} on {target:?} must be rejected"));
            let EmitError::InvalidBody(invalid) = err else {
                panic!("{name} on {target:?}: expected InvalidBody, got {err:?}")
            };
            assert_eq!(invalid.kind, expected, "{name} on {target:?}");
        }
    }
}

/// The same rejection through `compile`, before any tool runs or any file is written.
#[test]
fn compile_rejects_a_malformed_body_before_running_a_tool() {
    let dir = std::env::temp_dir().join(format!("poot-verify-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, body, expected) in malformed_bodies() {
        for target in [Target::SpirvVulkan, Target::Nvptx] {
            let out = artifact_path(&dir, "malformed", target);
            let err = compile(&body, target, &out)
                .expect_err(&format!("{name} on {target:?} must be rejected"));
            let CompileError::Emit(EmitError::InvalidBody(invalid)) = err else {
                panic!("{name} on {target:?}: expected an invalid-body error, got {err:?}")
            };
            assert_eq!(invalid.kind, expected, "{name} on {target:?}");
            assert!(!out.exists(), "{name} on {target:?} published an artifact");
        }
    }
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "scratch left behind"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
