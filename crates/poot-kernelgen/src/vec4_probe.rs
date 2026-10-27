//! The `Ty::Vec` exerciser (spec 134 P1): a vec4 load, splat, lanewise add and vec4 store, end to end on the
//! SPIR-V path. Test-only; no request generates it.

use poot_codegen::Target;
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Operand, Place, Rvalue, Statement,
    Terminator, Ty,
};
use poot_wgpu_runtime::KernelBuffer;

use crate::helpers::{copy, elem, ld, local, slice_f32};
use crate::unit_helpers::{ctx, spv, which};

/// Minimal `Ty::Vec` exerciser (spec 134 P1): each thread vec4-loads group `g` from `a`, adds a broadcast
/// constant `add`, and vec4-stores to `out`: `out[4g+l] = a[4g+l] + add`, lane `l` in `0..4`. Exercises
/// `Rvalue::VectorLoad`, `Rvalue::VectorSplat`, `BinaryOp` lanewise on `Ty::Vec`, and
/// `Statement::VectorStore`. No bounds-check block: the dispatch is sized exactly to the buffer.
pub(crate) fn vec4_add_splat(name: &str, add: f32) -> Body {
    // _0 ret, _1 a: &[f32], _2 out: &mut [f32], _3 i (vector-group index), _4 v (loaded vec4),
    // _5 splat (broadcast add), _6 sum (v + splat).
    let locals = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        ld(slice_f32(true), true),
        ld(Ty::Usize, false),
        ld(
            Ty::Vec {
                elem: Box::new(Ty::F32),
                lanes: 4,
            },
            false,
        ),
        ld(
            Ty::Vec {
                elem: Box::new(Ty::F32),
                lanes: 4,
            },
            false,
        ),
        ld(
            Ty::Vec {
                elem: Box::new(Ty::F32),
                lanes: 4,
            },
            false,
        ),
    ];
    let (a, out, i, v, splat, sum) = (local(1), local(2), local(3), local(4), local(5), local(6));
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(v), Rvalue::VectorLoad { place: elem(a, i) }),
            Statement::Assign(
                Place::local(splat),
                Rvalue::VectorSplat(Operand::Const(Constant::F32(add))),
            ),
            Statement::Assign(
                Place::local(sum),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(v)), copy(Place::local(splat))),
            ),
            Statement::VectorStore {
                place: elem(out, i),
                value: copy(Place::local(sum)),
            },
        ],
        terminator: Terminator::Return,
    };
    Body::new(name, 2, locals, vec![bb0, bb1])
}

/// `vec4_add_splat` end to end (SpirvVulkan only), split out of `tests/run.rs` (card 546c: its
/// only other caller, poot-ptx-device's schedule-batch atomicity test, was deleted with the resident
/// E4M3 region's `compile_schedule`/`compiled_batch` stack, which made this an external-only
/// consumer that `pub(crate)` cannot serve): (1) `spirv-val` the module (vec4 buffer resource,
/// splat and vec4 store must be strictly valid, not just RADV-tolerated); (2) dispatch on the
/// Arc/RADV and compare every lane to a scalar CPU reference - one workgroup of 64 threads, each
/// vec4-loading its own 4-float group (256 floats).
#[test]
fn vec4_add_splat_validates_and_matches_cpu() {
    let add = 1.5f32;
    let body = vec4_add_splat("vec4add", add);
    let s = spv(&body, "vec4add");

    // strict SPIR-V validity (skip only if spirv-val is absent).
    let dir = std::env::temp_dir()
        .join("poot-kernelgen-unit-test")
        .join("vec4add");
    let spv_path = poot_codegen::artifact_path(&dir, "vec4add", Target::SpirvVulkan);
    if which("spirv-val") {
        let v = std::process::Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&spv_path)
            .output()
            .expect("run spirv-val");
        assert!(
            v.status.success(),
            "spirv-val failed for vec4_add_splat:\n{}",
            String::from_utf8_lossy(&v.stderr)
        );
    } else {
        poot_test_util::skip_report("poot-kernelgen", &"spirv-val not on PATH");
    }

    let Some(ctx) = ctx() else { return };
    // 64 vec4 groups (one per thread in the single workgroup) * 4 lanes = 256 floats.
    let n = 256usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut bufs = [KernelBuffer::read_only_f32(&a), KernelBuffer::write_f32(n)];
    ctx.dispatch("test", &s, [64, 1, 1], [64, 1, 1], &mut bufs)
        .unwrap();
    let got = bufs[1].as_f32();
    let want: Vec<f32> = a.iter().map(|x| x + add).collect();
    assert_eq!(
        got,
        &want[..],
        "vec4_add_splat: gpu output must match the scalar CPU reference lane-for-lane"
    );
}
