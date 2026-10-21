//! `Target::SpirvOpenCl` (OpenCL-flavoured SPIR-V; no runtime consumes it, an Intel GPU backend is a future card): generated and fixture bodies lower
//! through the real `compile()` path (`llc`, triple `spirv64-unknown-unknown`) to modules `spirv-val` accepts
//! under `--target-env spv1.4` (the backend's output version), and the constructs OpenCL SPIR-V cannot
//! express are typed refusals. Needs `llc` and `spirv-val` (the devshell); a missing tool fails the test, it
//! does not skip it. There is no device row.

use std::path::{Path, PathBuf};
use std::process::Command;

use poot_codegen::{EmitError, Target, compile, emit_llvm_ir};
use poot_kernel_ir::test_support::{matmul_kernel, square_kernel, wg_sum, wmma_tile};
use poot_kernel_ir::{BinOp, Body, Terminator, Ty, fixtures};
use poot_kernelgen as kg;
use poot_kernelgen::reduce_last_dt;

#[path = "support/gather.rs"]
mod gather;
use gather::gather_axis0_dt;

fn out_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir()
        .join("poot-spirv-opencl-test")
        .join(tag);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `compile` `body` for OpenCL SPIR-V and return the artifact path.
fn compile_opencl(dir: &Path, body: &Body) -> PathBuf {
    let out = dir.join(format!("{}.spv", body.name));
    compile(body, Target::SpirvOpenCl, &out)
        .unwrap_or_else(|e| panic!("{} did not compile for SpirvOpenCl: {e}", body.name));
    out
}

fn spirv_val(module: &Path) {
    let v = Command::new("spirv-val")
        .args(["--target-env", "spv1.4"])
        .arg(module)
        .output()
        .expect("spirv-val is on PATH in the devshell");
    assert!(
        v.status.success(),
        "spirv-val rejected {}:\n{}",
        module.display(),
        String::from_utf8_lossy(&v.stderr)
    );
}

fn spirv_dis(module: &Path) -> String {
    let d = Command::new("spirv-dis")
        .arg(module)
        .output()
        .expect("spirv-dis is on PATH in the devshell");
    assert!(
        d.status.success(),
        "spirv-dis failed on {}",
        module.display()
    );
    String::from_utf8(d.stdout).unwrap()
}

fn movement(name: &str, out_shape: Vec<usize>, op: kg::MovementOp) -> Body {
    kg::movement_body(
        name,
        &kg::MovementSpec::new(kg::Storage::Elements { carrier: Ty::F32 }, out_shape, op),
        None,
    )
    .expect("a valid movement request")
}

fn fused_unary(name: &str, op: kg::FusedScalarOp) -> Body {
    kg::fused(
        name,
        &[4],
        &[&[4]],
        &kg::FusedKernel {
            leaf_tys: vec![kg::LeafTy::F32; 1],
            steps: vec![kg::FusedStep {
                op,
                inputs: vec![kg::FusedInput::Leaf(kg::LeafId::new(0))],
            }],
            output: kg::FusedInput::Step(kg::StepId::new(0)),
        },
    )
    .expect("fused precondition")
}

/// The attention generator's decode body: online softmax, LDS partials, barriers and `exp`.
fn attention_decode_body() -> Body {
    let spec = kg::AttentionSpec {
        batch: 1,
        hq: 4,
        n_rep: 2,
        lq: 1,
        lk: 6,
        dk: 8,
        dv: 8,
        key: kg::KeyLayout::Single,
        score: kg::ScoreTransform::Scale {
            scale_bits: 0.35355338f32.to_bits(),
        },
        mask: kg::MaskKind::Broadcast,
        bound: kg::KeyBound::Static,
        sink: kg::AttentionSink::Absent,
        schedule: kg::AttentionSchedule::Streaming { lanes: 4 },
    };
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    };
    let mut body = kg::generate(
        &kg::KernelRequest::Attention(spec),
        &poot_target::DeviceCaps::ptx_default(),
        &limits,
    )
    .expect("the attention request generates")
    .body;
    body.name = "attention_decode".to_string();
    body
}

/// SC-001: ordinary bodies, including the ones with shared memory, barriers, multi-block control flow and
/// transcendental math, emit OpenCL SPIR-V that `spirv-val` accepts.
#[test]
fn opencl_spirv_validates_on_representative_bodies() {
    let dir = out_dir("validates");
    let bodies = [
        fixtures::add_kernel(),
        square_kernel(),
        matmul_kernel(2, 3, 4),
        fused_unary("tanh", kg::FusedScalarOp::Tanh),
        fused_unary("erf", kg::FusedScalarOp::Erf),
        reduce_last_dt("reduce", Ty::F32, BinOp::Add, 4, 0.0),
        movement(
            "bcast",
            vec![2, 3],
            kg::MovementOp::Broadcast { in_shape: vec![3] },
        ),
        gather_axis0_dt("gather", Ty::F32, 2),
        wg_sum("wg_sum", 8),
        attention_decode_body(),
    ];
    for body in &bodies {
        spirv_val(&compile_opencl(&dir, body));
    }
}

/// The module is an OpenCL kernel (the `Kernel` capability, an `OpEntryPoint Kernel`), the workgroup size
/// rides `LocalSize`, and a workgroup barrier is a real `OpControlBarrier` over workgroup scope.
#[test]
fn opencl_module_shape_carries_kernel_workgroup_size_and_barrier() {
    let dir = out_dir("shape");
    let body = wg_sum("wg_sum_shape", 8);
    let [x, y, z] = body.workgroup_size;
    let dis = spirv_dis(&compile_opencl(&dir, &body));
    assert!(dis.contains("OpCapability Kernel"), "{dis}");
    assert!(dis.contains("OpEntryPoint Kernel"), "{dis}");
    assert!(
        dis.contains(&format!(
            "OpExecutionMode %wg_sum_shape LocalSize {x} {y} {z}"
        )),
        "no LocalSize {x} {y} {z}:\n{dis}"
    );
    assert!(dis.contains("OpControlBarrier"), "{dis}");
}

/// The thread-index builtins, the length ABI and the barrier come out of the emitter as the OpenCL forms.
#[test]
fn opencl_ir_uses_flat_global_pointers_and_opencl_builtins() {
    let ir = emit_llvm_ir(&wg_sum("wg_sum_ir", 8), Target::SpirvOpenCl).unwrap();
    assert!(
        ir.contains("target triple = \"spirv64-unknown-unknown\""),
        "{ir}"
    );
    assert!(
        ir.contains("define spir_kernel void @wg_sum_ir(ptr addrspace(1) %p1, i64 %n1"),
        "{ir}"
    );
    assert!(
        ir.contains("tail call i64 @_Z12get_local_idj(i32 0)"),
        "{ir}"
    );
    // Workgroup scope; AcquireRelease (0x8) over WorkgroupMemory (0x100) and CrossWorkgroupMemory (0x200).
    let semantics = 0x8 | 0x100 | 0x200;
    assert!(
        ir.contains(&format!(
            "call void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 {semantics})"
        )),
        "{ir}"
    );
    assert!(!ir.contains("llvm.spv.resource"), "{ir}");
}

fn refusal(body: &Body) -> String {
    match emit_llvm_ir(body, Target::SpirvOpenCl) {
        Err(EmitError::Unsupported(why)) => why,
        other => panic!(
            "{} should be a typed Unsupported refusal, got {other:?}",
            body.name
        ),
    }
}

/// SC-002a: a fault terminator is refused by name. OpenCL SPIR-V has no device trap, and `llvm.trap` lowers to
/// `OpUnreachable`, which is undefined behaviour, so emitting it would be a check that does nothing.
#[test]
fn opencl_refuses_a_trap_terminator_by_name() {
    let mut trapping = fixtures::add_kernel();
    trapping.name = "add_with_trap".into();
    let block = trapping
        .blocks
        .iter_mut()
        .find(|b| matches!(b.terminator, Terminator::Return))
        .expect("add_kernel returns");
    block.terminator = Terminator::Trap { code: 1 };
    assert!(refusal(&trapping).contains("no device trap"));
}

/// SC-002b: a no-contraction marker is refused by name; the backend carries no `NoContraction` decoration.
#[test]
fn opencl_refuses_a_no_contract_marker_by_name() {
    let marked = poot_kernel_ir::test_support::no_contract_probe_kernel(true);
    assert!(refusal(&marked).contains("NoContraction"));
}

/// SC-002c: a tensor-core fragment is refused by name; the Intel DPAS lowering is not implemented.
#[test]
fn opencl_refuses_a_tensor_core_fragment_by_name() {
    let tile = wmma_tile("wmma_tile_refused");
    assert!(refusal(&tile).contains("DPAS"));
}
