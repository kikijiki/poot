//! Verify each generated kernel on the Arc (compile to SPIR-V, dispatch, compare to the CPU result).
//! Skips (passes) if no Vulkan adapter.

use std::path::PathBuf;

use poot_codegen::emit_llvm_ir;
use poot_codegen::{GpuTarget, Target, compile, compile_kernel};
use poot_kernel_ir::test_support::{wg_sum, workgroup_fence_spin_wait};
use poot_kernel_ir::{
    BasicBlock, BinOp, Body, LocalDecl, MathOp, MemoryOrdering, MemoryScope, Statement, Terminator,
    Ty, UnOp,
};
use poot_kernelgen as kg;
use poot_target::BufferStorage;
use poot_wgpu_runtime::{Context, KernelBuffer};

#[path = "support/attention.rs"]
mod attention;

/// The f32-element movement `op` into `out_shape`.
fn movement(name: &str, out_shape: Vec<usize>, op: kg::MovementOp) -> Body {
    kg::movement_body(
        name,
        &kg::MovementSpec::new(kg::Storage::Elements { carrier: Ty::F32 }, out_shape, op),
        None,
    )
    .expect("a valid movement request")
}

/// The f32 numpy-broadcasting elementwise `c = a op b` over contiguous operands.
fn binary_broadcast(
    name: &str,
    op: BinOp,
    out_shape: &[usize],
    a_shape: &[usize],
    b_shape: &[usize],
) -> Body {
    kg::binary_broadcast_dt_views_grid(
        name,
        op,
        Ty::F32,
        out_shape,
        a_shape,
        &kg::Layout::contiguous(a_shape),
        b_shape,
        &kg::Layout::contiguous(b_shape),
        None,
    )
}

fn spv(body: &Body, name: &str) -> poot_runtime_common::CompiledKernel {
    let dir: PathBuf = std::env::temp_dir().join("poot-kernelgen-test").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let out = poot_codegen::artifact_path(&dir, name, Target::SpirvVulkan);
    let artifact = compile_kernel(body, GpuTarget::SpirvVulkan, &out).expect("compile");
    poot_codegen::kernel_handle(body, &artifact).unwrap()
}

/// The compiled SPIR-V word count of a handle `spv()` built, for callers that just want to assert a
/// non-empty compile (card 608: `CompiledKernel` has no `len`/`is_empty` of its own).
fn spirv_word_count(kernel: &poot_runtime_common::CompiledKernel) -> usize {
    let poot_runtime_common::KernelCode::SpirvWords(words) = kernel.code() else {
        panic!("expected a SpirvVulkan-compiled kernel");
    };
    words.len()
}

/// Compile `body` to NVPTX and return the `.ptx` text. Panics if llc rejects it (an unsupported op / invalid
/// IR for the NVPTX backend) - the local check for the PTX codegen path without a rented GPU.
fn ptx(body: &Body, name: &str) -> String {
    let dir: PathBuf = std::env::temp_dir()
        .join("poot-kernelgen-test")
        .join(format!("{name}_ptx"));
    std::fs::create_dir_all(&dir).unwrap();
    let out = poot_codegen::artifact_path(&dir, name, Target::Nvptx);
    compile(body, Target::Nvptx, &out).expect("compile to NVPTX");
    std::fs::read_to_string(&out).unwrap()
}

/// An attention request over dense f32 operands with the Llama scale `0.5`.
#[expect(clippy::too_many_arguments)]
fn attention_spec(
    batch: usize,
    hq: usize,
    n_rep: usize,
    lq: usize,
    lk: usize,
    d: usize,
    lanes: usize,
    mask: kg::MaskKind,
) -> kg::AttentionSpec {
    kg::AttentionSpec {
        batch,
        hq,
        n_rep,
        lq,
        lk,
        dk: d,
        dv: d,
        key: kg::KeyLayout::Single,
        score: kg::ScoreTransform::Scale {
            scale_bits: 0.5f32.to_bits(),
        },
        mask,
        bound: kg::KeyBound::Static,
        schedule: kg::AttentionSchedule::Streaming { lanes },
    }
}

/// The body and launch of `spec` through the one generation seam, on a wgpu-class device.
fn attention(spec: &kg::AttentionSpec) -> kg::Generated {
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    };
    kg::generate(
        &kg::KernelRequest::Attention(spec.clone()),
        &poot_target::DeviceCaps::wgpu_rdna3_igpu(),
        &limits,
    )
    .expect("the attention request generates")
}

#[test]
fn synthesized_kernels_compile_to_valid_ptx() {
    // Every synthesized kernel the engine uses by default on wgpu (dense tiled GEMM, GGUF/GPTQ/AWQ
    // fused-dequant GEMM, attention) must also lower to NVPTX. A clean `compile(.., Nvptx)` means
    // llc accepted the IR; the `.extern .func` grep catches an unresolved intrinsic (a wrong NVPTX intrinsic name
    // only JIT-fails on a real GPU, see memory `nvptx-lg2-intrinsic-name`), so it must be empty.
    let (m, k, n, ts) = (17usize, 64usize, 24usize, 8usize);
    let kernels: Vec<(&str, Body)> = vec![
        (
            "tiled_region",
            kg::tiled_region(
                "tr",
                m,
                k,
                n,
                ts,
                0,
                kg::WeightLayout::Kn,
                BufferStorage::f32(),
                None,
            )
            .expect("f32 weight storage"),
        ),
        (
            "attention_decode",
            // 3 lanes do not divide 8 keys: exercises the tail-lane (partial-slice) path.
            attention(&attention_spec(
                1,
                4,
                2,
                1,
                8,
                4,
                3,
                kg::MaskKind::Broadcast,
            ))
            .body,
        ),
        (
            "attention_softcapped_prefill",
            attention(&kg::AttentionSpec {
                score: kg::ScoreTransform::ScaleSoftcap {
                    scale_bits: 0.5f32.to_bits(),
                    cap_bits: 30.0f32.to_bits(),
                },
                ..attention_spec(1, 4, 2, 6, 6, 4, 4, kg::MaskKind::PerHead)
            })
            .body,
        ),
        (
            "attn_scores_v_gemv_lds",
            kg::attn_scores_v_gemv_lds("asvg", 300, n, 128, n),
        ),
    ];
    if !which("llc") {
        eprintln!("llc not on PATH; skipping the NVPTX-compile check");
        return;
    }
    for (name, body) in &kernels {
        let asm = ptx(body, name);
        // every callee must be a defined function (`.func name(` / `.visible .func`); an `.extern .func`
        // declaration is an unresolved intrinsic that will JIT-fail on a real NVPTX GPU.
        assert!(
            !asm.contains(".extern .func"),
            "{name}: PTX has an unresolved `.extern .func` (a missing NVPTX intrinsic):\n{}",
            asm.lines()
                .filter(|l| l.contains(".extern .func"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    eprintln!(
        "all {} synthesized kernels compile to valid NVPTX (no unresolved intrinsics)",
        kernels.len()
    );
}

fn ctx() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            None
        }
    }
}

#[test]
fn binary_mul() {
    let Some(ctx) = ctx() else { return };
    let s = spv(&kg::binary("mul", BinOp::Mul), "mul");
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [2.0f32, 3.0, 4.0, 5.0];
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&b),
        KernelBuffer::write_f32(4),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [4, 1, 1], &mut bufs)
        .unwrap();
    assert_eq!(bufs[2].as_f32(), &[2.0, 6.0, 12.0, 20.0]);
}

/// Cross-block producer/consumer handoff on a cooperative grid, with `Fence` pairs bracketing the
/// handoff, must lower to a PTX `fence` with no unresolved intrinsic. Compile-only: RADV cannot prove
/// cross-CTA forward progress, so there is no SpirvVulkan/wgpu counterpart to run.
#[test]
fn cross_block_fence_spin_wait_coop_compiles_to_ptx() {
    let p = ptx(
        &poot_kernel_ir::test_support::cross_block_fence_spin_wait_coop("fencespinwaitc"),
        "fencespinwaitc",
    );
    assert!(
        p.contains("fence") && !p.contains(".extern .func"),
        "the fence-guarded cooperative spin-wait must emit a PTX fence with no unresolved intrinsic:\n{p}"
    );
    assert!(
        p.contains("atom"),
        "the flag handoff must still use an atomic:\n{p}"
    );
}

/// Workgroup-scope fence kernel must still lower to PTX (compile parity; `Statement::Fence` lowers on every
/// target except AieCore).
#[test]
fn workgroup_fence_spin_wait_compiles_to_ptx() {
    let p = ptx(&workgroup_fence_spin_wait("wgfence", 128), "wgfence");
    assert!(
        p.contains("fence") && !p.contains(".extern .func"),
        "the workgroup fence must emit a PTX fence with no unresolved intrinsic:\n{p}"
    );
}

/// `Statement::Fence`'s SPIR-V lowering must be spirv-val-clean for both scopes. An explicit
/// `syncscope("device")`/`syncscope("workgroup")` already emits `OpMemoryBarrier`'s Memory Scope as
/// `Device`(1)/`Workgroup`(2) (see `Emitter::fence_syncscope` in `poot-codegen/src/emit.rs`), so no
/// `spirv_postprocess` rewrite is needed. Mirrors `global_atomic_add_counter_validates_as_spirv`. Skips the
/// assertion (not the test) if spirv-val is absent.
#[test]
fn fence_validates_as_spirv_both_scopes() {
    for (body, name) in [
        (workgroup_fence_spin_wait("wgfence_val", 128), "wgfence_val"),
        (
            poot_kernel_ir::test_support::cross_block_fence_spin_wait_coop("devfence_val"),
            "devfence_val",
        ),
    ] {
        let s = spv(&body, name);
        assert!(
            spirv_word_count(&s) > 0,
            "{name}: compile must produce non-empty SPIR-V"
        );
        let dir = std::env::temp_dir().join("poot-kernelgen-test").join(name);
        let spv_path = poot_codegen::artifact_path(&dir, name, Target::SpirvVulkan);
        if which("spirv-val") {
            let v = std::process::Command::new("spirv-val")
                .arg("--target-env")
                .arg("vulkan1.3")
                .arg(&spv_path)
                .output()
                .expect("run spirv-val");
            assert!(
                v.status.success(),
                "spirv-val failed for {name}:\n{}",
                String::from_utf8_lossy(&v.stderr)
            );
        } else {
            eprintln!("spirv-val not on PATH; skipping the validity check for {name}");
        }
    }
}

/// `workgroup_fence_spin_wait` dispatched as exactly one workgroup on the real Vulkan/RADV device. Thread 0
/// writes LDS `data[0]=42` behind a `Fence{Workgroup,Release}`; every other thread spins on an LDS flag then reads
/// `data[0]` behind a `Fence{Workgroup,Acquire}`. Waves of one workgroup are co-resident, so unlike the cross-CTA
/// case (cooperative NVPTX grid, pod-gated) this is locally verifiable: every `out[tid]` must equal 42.0;
/// without the fence pair's happens-before a consumer could observe a stale (0.0) `data[0]`.
#[test]
fn workgroup_fence_spin_wait_dispatches_on_radv() {
    let Some(ctx) = ctx() else { return };
    const THREADS: u32 = 128;
    let s = spv(
        &workgroup_fence_spin_wait("wgfence_run", THREADS),
        "wgfence_run",
    );
    let mut bufs = [KernelBuffer::write_f32(THREADS as usize)];
    // dispatch size == workgroup size: exactly ONE workgroup, so every thread shares the LDS fence.
    ctx.dispatch("test", &s, [THREADS, 1, 1], [THREADS, 1, 1], &mut bufs)
        .unwrap();
    let got = bufs[0].as_f32();
    let bad: Vec<(usize, f32)> = got
        .iter()
        .enumerate()
        .skip(1) // thread 0 is the producer; it never writes `out[0]`.
        .filter(|&(_, &v)| v != 42.0)
        .map(|(i, &v)| (i, v))
        .collect();
    assert!(
        bad.is_empty(),
        "workgroup-scope fence handoff failed for {} of {} consumer threads (first few: {:?}); the \
         Fence{{Workgroup, Acquire/Release}} pair did not establish the cross-wave happens-before",
        bad.len(),
        THREADS - 1,
        &bad[..bad.len().min(8)]
    );
}

/// `Statement::Fence` is rejected on AieCore with a named diagnostic (a single AIE core has no other core to
/// fence against), like the Barrier/CAS AIE rejections. Pure Rust check via `emit_llvm_ir`, no llc/Peano needed,
/// so it always runs (unlike the `#[ignore]`d real-Peano rows). Uses a minimal single-statement
/// body (Fence, then `Return`): the spin-wait kernels above also use `Terminator::ThreadIndexCall` (itself
/// AieCore-rejected), which would let `.is_err()` pass for the wrong reason.
#[test]
fn fence_rejects_on_aiecore() {
    for (name, scope) in [
        ("workgroup", MemoryScope::Workgroup),
        ("device", MemoryScope::Device),
    ] {
        let locals = vec![LocalDecl {
            ty: Ty::Unit,
            mutable: false,
        }];
        let bb0 = BasicBlock {
            statements: vec![Statement::Fence {
                scope,
                ordering: MemoryOrdering::AcqRel,
            }],
            terminator: Terminator::Return,
        };
        let body = Body::new("fence_only", 0, locals, vec![bb0]);
        let err = emit_llvm_ir(&body, Target::AieCore).expect_err(
            "{name}-scope Fence on AieCore must be a named rejection, not a silently-accepted emit",
        );
        assert!(
            err.to_string().contains("fence"),
            "{name}-scope Fence on AieCore must be rejected BY THE FENCE (not some other unrelated \
             AieCore restriction); got: {err}"
        );
    }
}

#[test]
fn binary_scalar_add() {
    let Some(ctx) = ctx() else { return };
    let s = spv(&kg::binary_scalar("addk", BinOp::Add, 10.0), "addk");
    let a = [1.0f32, 2.0, 3.0];
    let mut bufs = [KernelBuffer::read_only_f32(&a), KernelBuffer::write_f32(3)];
    ctx.dispatch("test", &s, [64, 1, 1], [3, 1, 1], &mut bufs)
        .unwrap();
    assert_eq!(bufs[1].as_f32(), &[11.0, 12.0, 13.0]);
}

#[test]
fn unary_neg() {
    let Some(ctx) = ctx() else { return };
    let s = spv(&kg::unary("neg", UnOp::Neg), "neg");
    let x = [1.0f32, -2.0, 3.0];
    let mut bufs = [KernelBuffer::read_only_f32(&x), KernelBuffer::write_f32(3)];
    ctx.dispatch("test", &s, [64, 1, 1], [3, 1, 1], &mut bufs)
        .unwrap();
    assert_eq!(bufs[1].as_f32(), &[-1.0, 2.0, -3.0]);
}

#[test]
fn broadcast_bias_and_colvec() {
    let Some(ctx) = ctx() else { return };
    // [2,3] + [3] (row bias broadcast over rows)
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias = [10.0f32, 20.0, 30.0];
    let s = spv(
        &binary_broadcast("bcast1", BinOp::Add, &[2, 3], &[2, 3], &[3]),
        "bcast1",
    );
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&bias),
        KernelBuffer::write_f32(6),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [6, 1, 1], &mut bufs)
        .unwrap();
    assert_eq!(bufs[2].as_f32(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

    // [2,3] / [2,1] (per-row scalar broadcast over columns) - the rmsnorm x/den shape.
    let den = [2.0f32, 4.0];
    let s = spv(
        &binary_broadcast("bcast2", BinOp::Div, &[2, 3], &[2, 3], &[2, 1]),
        "bcast2",
    );
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&den),
        KernelBuffer::write_f32(6),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [6, 1, 1], &mut bufs)
        .unwrap();
    assert_eq!(bufs[2].as_f32(), &[0.5, 1.0, 1.5, 1.0, 1.25, 1.5]);
}

#[test]
fn dyn_update_slice_writes_slot() {
    let Some(ctx) = ctx() else { return };
    // operand [2,4,3] (filled 0), update [2,1,3] written at slot idx=2 on axis 1; rest preserved.
    let operand: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let update = [100.0f32, 200.0, 300.0, 400.0, 500.0, 600.0];
    let s = spv(
        &movement(
            "dus",
            vec![2, 4, 3],
            kg::MovementOp::UpdateSlice {
                update_shape: vec![2, 1, 3],
                axis: 1,
                index: kg::UpdateIndex::Static(2),
            },
        ),
        "dus",
    );
    let mut bufs = [
        KernelBuffer::read_only_f32(&operand),
        KernelBuffer::read_only_f32(&update),
        KernelBuffer::write_f32(24),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [24, 1, 1], &mut bufs)
        .unwrap();
    // expected: operand with row0 slot2 (flat 6..9) and row1 slot2 (flat 18..21) overwritten.
    let mut want: Vec<f32> = (0..24).map(|i| i as f32).collect();
    want[6..9].copy_from_slice(&[100.0, 200.0, 300.0]);
    want[18..21].copy_from_slice(&[400.0, 500.0, 600.0]);
    assert_eq!(bufs[2].as_f32(), &want[..]);
}

/// A one-step fused kernel `y = op(x)` over `n` elements: the single synthesizer both standalone and fused
/// transcendentals go through.
fn fused_unary(name: &str, op: kg::FusedScalarOp, n: usize) -> Body {
    let k = kg::FusedKernel {
        leaf_tys: vec![kg::LeafTy::F32; 1],
        steps: vec![kg::FusedStep {
            op,
            inputs: vec![kg::FusedInput::Leaf(0)],
        }],
        output: kg::FusedInput::Step(0),
    };
    kg::fused(name, &[n], &[&[n]], &k).expect("fused precondition")
}

fn run_unary(body: &Body, name: &str, ctx: &Context, x: &[f32]) -> Vec<f32> {
    let s = spv(body, name);
    let mut bufs = [
        KernelBuffer::read_only_f32(x),
        KernelBuffer::write_f32(x.len()),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [x.len() as u32, 1, 1], &mut bufs)
        .unwrap();
    bufs[1].as_f32().to_vec()
}

#[test]
fn tanh_matches_reference_and_saturates_exactly() {
    let Some(ctx) = ctx() else { return };
    let x = [
        0.0f32,
        1e-6,
        -1e-6,
        0.5,
        -0.5,
        1.0,
        -1.0,
        3.0,
        -3.0,
        10.0,
        -10.0,
        88.0,
        -88.0,
        100.0,
        -100.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let got = run_unary(
        &fused_unary("tanh", kg::FusedScalarOp::Tanh, x.len()),
        "tanh",
        &ctx,
        &x,
    );
    for (g, v) in got.iter().zip(&x) {
        assert!(
            (g - v.tanh()).abs() <= 2e-6,
            "tanh({v}): {g} vs {}",
            v.tanh()
        );
    }
    // The overflow-free form saturates exactly; the textbook `(e^x - e^-x) / (e^x + e^-x)` is NaN here.
    for (v, want) in [(100.0f32, 1.0f32), (-100.0, -1.0), (f32::INFINITY, 1.0)] {
        let i = x.iter().position(|&xv| xv == v).unwrap();
        assert_eq!(got[i], want, "tanh({v}) must be exactly {want}");
    }
}

#[test]
fn erf_matches_reference() {
    let Some(ctx) = ctx() else { return };
    // erf values to 10 digits (Abramowitz-Stegun table 7.1).
    let table = [
        (0.0f32, 0.0f32),
        (0.1, 0.112_462_92),
        (0.5, 0.520_499_9),
        (1.0, 0.842_700_8),
        (2.0, 0.995_322_3),
        (-3.0, -0.999_977_9),
        (-0.5, -0.520_499_9),
        (6.0, 1.0),
        (-100.0, -1.0),
        (f32::INFINITY, 1.0),
    ];
    let x: Vec<f32> = table.iter().map(|&(v, _)| v).collect();
    let got = run_unary(
        &fused_unary("erf", kg::FusedScalarOp::Erf, x.len()),
        "erf",
        &ctx,
        &x,
    );
    for (g, (v, want)) in got.iter().zip(&table) {
        assert!((g - want).abs() <= 3e-7, "erf({v}): {g} vs {want}");
    }
}

#[test]
fn matmul_gemv_and_batched() {
    let Some(ctx) = ctx() else { return };
    // gemv: A[1,1,4] @ B[4,3] -> [1,1,3]
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bm: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let s = spv(
        &kg::matmul_batched_dt_grid("mm", Ty::F32, &[1, 1, 3], &[1, 1, 4], &[4, 3], None),
        "mm",
    );
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&bm),
        KernelBuffer::write_f32(3),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [3, 1, 1], &mut bufs)
        .unwrap();
    let mut want = [0.0f32; 3];
    for nn in 0..3 {
        for kk in 0..4 {
            want[nn] += a[kk] * bm[kk * 3 + nn];
        }
    }
    assert_eq!(bufs[2].as_f32(), &want[..]);

    // batched: A[1,2,1,4] @ B[1,2,4,3] -> [1,2,1,3]
    let a2: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let b2: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5).collect();
    let s = spv(
        &kg::matmul_batched_dt_grid(
            "mmb",
            Ty::F32,
            &[1, 2, 1, 3],
            &[1, 2, 1, 4],
            &[1, 2, 4, 3],
            None,
        ),
        "mmb",
    );
    let mut bufs = [
        KernelBuffer::read_only_f32(&a2),
        KernelBuffer::read_only_f32(&b2),
        KernelBuffer::write_f32(6),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [6, 1, 1], &mut bufs)
        .unwrap();
    let mut want2 = [0.0f32; 6];
    for h in 0..2 {
        for nn in 0..3 {
            let mut acc = 0.0;
            for kk in 0..4 {
                acc += a2[h * 4 + kk] * b2[h * 12 + kk * 3 + nn];
            }
            want2[h * 3 + nn] = acc;
        }
    }
    assert_eq!(bufs[2].as_f32(), &want2[..]);
}

/// The epilogue `head + bias` over an f32 head of `out_shape`, the bias of shape `bias_shape`.
fn bias_epilogue(out_shape: &[usize], bias_shape: &[usize]) -> kg::Epilogue {
    kg::Epilogue {
        dt: poot_kernel_ir::Ty::F32,
        out_shape: out_shape.to_vec(),
        leaves: vec![kg::EpilogueLeaf {
            shape: bias_shape.to_vec(),
            storage: None,
        }],
        steps: vec![kg::EpilogueStep {
            op: kg::FusedScalarOp::Binary(poot_kernel_ir::BinOp::Add),
            inputs: vec![kg::EpilogueInput::Head, kg::EpilogueInput::Leaf(0)],
        }],
        output: kg::EpilogueInput::Step(0),
    }
}

#[test]
fn matmul_bias_epilogue() {
    // fused matmul+bias `C[..,m,n] = (A@B)[..,m,n] + bias[n]` matches matmul-then-add. A[1,1,4] @ B[4,3] + bias[3]
    // -> [1,1,3]. Inputs bind a, b, bias; output last.
    let Some(ctx) = ctx() else { return };
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bm: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let bias: Vec<f32> = vec![100.0, 200.0, 300.0];
    let s = spv(
        &kg::matmul_batched_epilogue_dt_grid(
            "mmbias",
            poot_kernel_ir::Ty::F32,
            &[1, 1, 3],
            &[1, 1, 4],
            &[4, 3],
            &bias_epilogue(&[1, 1, 3], &[3]),
            None,
        )
        .expect("the epilogue's leaf storage is valid"),
        "mmbias",
    );
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&bm),
        KernelBuffer::read_only_f32(&bias),
        KernelBuffer::write_f32(3),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [3, 1, 1], &mut bufs)
        .unwrap();
    let mut want = [0.0f32; 3];
    for nn in 0..3 {
        for kk in 0..4 {
            want[nn] += a[kk] * bm[kk * 3 + nn];
        }
        want[nn] += bias[nn];
    }
    assert_eq!(bufs[3].as_f32(), &want[..]);
}

/// The generated indexed contraction for `(m, k, n, experts)` under `schedule`, with the unfolded launch the planner
/// builds (`Gemv`: `ceil(M*N/cols)` workgroups; `Tiled`: one workgroup per `tile x tile` output block) and the
/// `[K, N, E, M, x_groups]` metadata. Returns the compiled kernel and its launch.
fn indexed_kernel(
    schedule: kg::Schedule,
    (m, k, n, experts): (usize, usize, usize, usize),
    name: &str,
) -> (poot_runtime_common::CompiledKernel, kg::Launch, [u32; 3]) {
    use poot_target::DeviceCaps;
    let caps = DeviceCaps::wgpu_rdna3_igpu();
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    let threads = schedule.grid_threads(1, m, n);
    let x_groups = match schedule {
        kg::Schedule::Gemv { .. } => 0,
        kg::Schedule::Tiled { tile } => (threads / (tile.get() * tile.get()) as usize) as u32,
        kg::Schedule::RegisterTiled { .. } => {
            panic!("an indexed contraction is never RegisterTiled")
        }
    };
    let request =
        kg::KernelRequest::Contraction(kg::ContractionSpec::Indexed(kg::IndexedContraction {
            schedule,
            layout: kg::WeightLayout::Kn,
            weight: BufferStorage::f32(),
            epilogue: None,
            grid: [threads as u32, 1, 1],
            meta: vec![k as u32, n as u32, experts as u32, m as u32, x_groups],
        }));
    let generated =
        kg::generate(&request, &caps, &limits).unwrap_or_else(|e| panic!("{name}: {e}"));
    let workgroup = generated.body.workgroup_size;
    (spv(&generated.body, name), generated.launch, workgroup)
}

/// Run the indexed kernel and return its `[M, N]` output.
fn run_indexed(
    ctx: &Context,
    (kernel, launch, workgroup): &(poot_runtime_common::CompiledKernel, kg::Launch, [u32; 3]),
    (x, w, ids): (&[f32], &[f32], &[i32]),
    out_len: usize,
) -> Vec<f32> {
    let meta = launch
        .meta
        .clone()
        .expect("an indexed launch carries metadata");
    let mut bufs = [
        KernelBuffer::read_only_f32(x),
        KernelBuffer::read_only_f32(w),
        KernelBuffer::read_only_i32(ids),
        KernelBuffer::read_only_u32(&meta),
        KernelBuffer::write_f32(out_len),
    ];
    ctx.dispatch("test", kernel, *workgroup, launch.grid, &mut bufs)
        .unwrap();
    bufs[4].as_f32().to_vec()
}

/// `out[m, n] = sum_k x[m, k] * W[ids[m], k, n]`, summed `k` ascending (a NaN for an id outside `0..E`).
fn indexed_oracle(
    (x, w, ids): (&[f32], &[f32], &[i32]),
    (k, n, experts): (usize, usize, usize),
) -> Vec<f32> {
    let mut want = Vec::new();
    for (row, id) in ids.iter().enumerate() {
        for col in 0..n {
            if !(0..experts as i32).contains(id) {
                want.push(f32::NAN);
                continue;
            }
            let expert = *id as usize;
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[row * k + kk] * w[expert * k * n + kk * n + col];
            }
            want.push(acc);
        }
    }
    want
}

#[test]
fn indexed_gemv_selects_expert_in_kernel() {
    // Gather-free indexed GEMV: out[n] = sum_k x[k] * W[e,k,n] where e = idx[0] selects one [K,N] expert from
    // W[E,K,N] in-kernel (no gather copy). E=3, K=4, N=3. Run once per expert id; each must read that
    // expert's rows. Integer-valued operands, so every summation order gives the same bits.
    let Some(ctx) = ctx() else { return };
    let (e, k, n) = (3usize, 4usize, 3usize);
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let w: Vec<f32> = (0..e * k * n).map(|v| (v as f32) - 12.0).collect();
    let gemv = kg::Schedule::Gemv {
        width: 16,
        cols: 2,
        unroll: 2,
    };
    let kernel = indexed_kernel(gemv, (1, k, n, e), "igemv");
    for expert in 0..e {
        let ids = [expert as i32];
        let got = run_indexed(&ctx, &kernel, (&x, &w, &ids), n);
        let want = indexed_oracle((&x, &w, &ids), (k, n, e));
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "expert {expert}"
        );
    }
}

#[test]
fn indexed_matmul_routes_rows_to_experts() {
    // M>1 indexed GEMM on the segmented Tiled schedule: out[m,n] = sum_k x[m,k] * W[idx[m],k,n], each row
    // routed to its own expert in-kernel. E=3, M=3, K=4, N=2; rows pick experts 2, 0, 1, so one tile holds rows
    // of three experts and the kernel reads a different weight block per row.
    let Some(ctx) = ctx() else { return };
    let (e, m, k, n) = (3usize, 3usize, 4usize, 2usize);
    let x: Vec<f32> = (0..m * k).map(|v| (v as f32) - 5.0).collect();
    let w: Vec<f32> = (0..e * k * n).map(|v| (v as f32) - 11.0).collect();
    let ids = [2i32, 0, 1];
    let tiled = kg::Schedule::Tiled {
        tile: kg::TileSize::new(4).unwrap(),
    };
    let kernel = indexed_kernel(tiled, (m, k, n, e), "igemm");
    let got = run_indexed(&ctx, &kernel, (&x, &w, &ids), m * n);
    let want = indexed_oracle((&x, &w, &ids), (k, n, e));
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    );
}

#[test]
fn indexed_bodies_compile_to_valid_spirv_and_nvptx() {
    // The Gemv and the segmented Tiled body, each bare and with an epilogue, build, verify and compile on both
    // SPIR-V and NVPTX.
    use kg::{ContractionSpec, IndexedContraction, KernelRequest, Schedule};
    use poot_target::DeviceCaps;
    let caps = DeviceCaps::wgpu_rdna3_igpu();
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    let tile = Schedule::Tiled {
        tile: kg::TileSize::new(8).unwrap(),
    };
    let gemv = Schedule::Gemv {
        width: 64,
        cols: 4,
        unroll: 4,
    };
    let request = |schedule, epilogue| {
        KernelRequest::Contraction(ContractionSpec::Indexed(IndexedContraction {
            schedule,
            layout: kg::WeightLayout::Kn,
            weight: BufferStorage::f32(),
            epilogue,
            grid: [64, 1, 1],
            meta: vec![16, 8, 4, 8, 0],
        }))
    };
    let requests = [
        ("indexed_gemv_f32", request(gemv, None)),
        ("indexed_tiled_f32", request(tile, None)),
        (
            "indexed_gemv_epilogue",
            request(gemv, Some(chain_epilogue(Ty::F32, &[8, 8]))),
        ),
        (
            "indexed_tiled_epilogue",
            request(tile, Some(chain_epilogue(Ty::F32, &[8, 8]))),
        ),
    ];
    for (name, request) in &requests {
        let generated =
            kg::generate(request, &caps, &limits).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(spirv_word_count(&spv(&generated.body, name)) > 0, "{name}");
    }
    if which("llc") {
        for (name, request) in &requests {
            let generated =
                kg::generate(request, &caps, &limits).unwrap_or_else(|e| panic!("{name}: {e}"));
            let asm = ptx(&generated.body, name);
            assert!(
                !asm.contains(".extern .func"),
                "{name}: an unresolved intrinsic in the NVPTX output"
            );
        }
    } else {
        eprintln!("llc not on PATH; skipping the NVPTX-compile check");
    }
}

#[test]
fn scatter_update_writes_mapped_rows_keeps_rest() {
    // The in-place row write of the paged KV append: out[slots[t]] = src[t] for slots[t] >= 0, into a pool.
    // POOL=4 rows of rest=2; src has 3 rows; src row 0 -> slot 2, row 1 is padding (-1), row 2 -> slot 0;
    // slots 1 and 3 are not named. Only source rows are read. The runtime's writable buffers start zeroed,
    // so the unnamed slots read back zero here; keeping a non-zero base is checked on the KIR interpreter
    // (`packed_contraction_movement`) and on every executor (`state_row`).
    let Some(ctx) = ctx() else { return };
    let (pool, rest) = (4usize, 2usize);
    let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // src[0]=[1,2], src[1]=[3,4], src[2]=[5,6]
    let slots: Vec<i32> = vec![2, -1, 0];
    let s = spv(
        &movement(
            "scu",
            vec![pool, rest],
            kg::MovementOp::ScatterRows {
                src_shape: vec![3, rest],
                window: None,
            },
        ),
        "scu",
    );
    let mut bufs = [
        KernelBuffer::read_only_f32(&src),
        KernelBuffer::read_only_i32(&slots),
        KernelBuffer::write_f32(pool * rest),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [(3 * rest) as u32, 1, 1], &mut bufs)
        .unwrap();
    // out: slot0=src2=[5,6], slot1 unnamed, slot2=src0=[1,2], slot3 unnamed; src row 1 (padding) lands
    // nowhere.
    assert_eq!(
        bufs[2].as_f32(),
        &[5.0, 6.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
        "slot0=src2, slot2=src0, the rest untouched"
    );
}

#[test]
fn gather_axis0_embedding() {
    let Some(ctx) = ctx() else { return };
    // data [3,2], index [2,0,1] -> out [3,2]
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let index = [2i32, 0, 1];
    let s = spv(
        &movement(
            "g",
            vec![3, 2],
            kg::MovementOp::Gather {
                in_shape: vec![3, 2],
                axis: 0,
                index_numel: 3,
            },
        ),
        "g",
    );
    let mut bufs = [
        KernelBuffer::read_only_f32(&data),
        KernelBuffer::read_only_i32(&index),
        KernelBuffer::write_f32(6),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [6, 1, 1], &mut bufs)
        .unwrap();
    assert_eq!(bufs[2].as_f32(), &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn reduce_sum_and_max() {
    let Some(ctx) = ctx() else { return };
    let x = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // 2 rows x 4 cols
    // sum
    let s = spv(
        &kg::reduce_last_dt("rsum", Ty::F32, BinOp::Add, 4, 0.0),
        "rsum",
    );
    let mut bufs = [KernelBuffer::read_only_f32(&x), KernelBuffer::write_f32(2)];
    ctx.dispatch("test", &s, [64, 1, 1], [2, 1, 1], &mut bufs)
        .unwrap();
    assert_eq!(bufs[1].as_f32(), &[10.0, 100.0]);
    // max
    let s = spv(
        &kg::reduce_last_dt("rmax", Ty::F32, BinOp::Max, 4, -1e30),
        "rmax",
    );
    let mut bufs = [KernelBuffer::read_only_f32(&x), KernelBuffer::write_f32(2)];
    ctx.dispatch("test", &s, [64, 1, 1], [2, 1, 1], &mut bufs)
        .unwrap();
    assert_eq!(bufs[1].as_f32(), &[4.0, 40.0]);
}

#[test]
fn fused_chain_with_broadcast_and_literal() {
    // out = (a + b) * s + 1, s a scalar leaf broadcast over the row. Exercises the fused synthesizer: multiple
    // leaves, a broadcast leaf (eff stride 0), register reuse between steps, and an inline literal, all in registers.
    let Some(ctx) = ctx() else { return };
    let k = kg::FusedKernel {
        leaf_tys: vec![kg::LeafTy::F32; 3],
        steps: vec![
            kg::FusedStep {
                op: kg::FusedScalarOp::Binary(BinOp::Add),
                inputs: vec![kg::FusedInput::Leaf(0), kg::FusedInput::Leaf(1)],
            },
            kg::FusedStep {
                op: kg::FusedScalarOp::Binary(BinOp::Mul),
                inputs: vec![kg::FusedInput::Step(0), kg::FusedInput::Leaf(2)],
            },
            kg::FusedStep {
                op: kg::FusedScalarOp::Binary(BinOp::Add),
                inputs: vec![kg::FusedInput::Step(1), kg::FusedInput::Lit(1.0)],
            },
        ],
        output: kg::FusedInput::Step(2),
    };
    let body = kg::fused("fchain", &[4], &[&[4], &[4], &[1]], &k).expect("fused precondition");
    let s = spv(&body, "fchain");
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [1.0f32, 1.0, 1.0, 1.0];
    let sc = [10.0f32];
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&b),
        KernelBuffer::read_only_f32(&sc),
        KernelBuffer::write_f32(4),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [4, 1, 1], &mut bufs)
        .unwrap();
    // (a+b)*10 + 1 = [21, 31, 41, 51]
    assert_eq!(bufs[3].as_f32(), &[21.0, 31.0, 41.0, 51.0]);
}

#[test]
fn fused_swiglu_chain() {
    // swiglu = silu(gate) * up as one fused kernel of primitive steps: silu = x / (1 + exp(-x)).
    let Some(ctx) = ctx() else { return };
    use kg::{FusedInput::*, FusedScalarOp as Op};
    let step = |op, inputs: Vec<kg::FusedInput>| kg::FusedStep { op, inputs };
    let k = kg::FusedKernel {
        leaf_tys: vec![kg::LeafTy::F32; 2],
        steps: vec![
            step(Op::Unary(UnOp::Neg), vec![Leaf(0)]),
            step(Op::Math(MathOp::Exp), vec![Step(0)]),
            step(Op::Binary(BinOp::Add), vec![Step(1), Lit(1.0)]),
            step(Op::Binary(BinOp::Div), vec![Leaf(0), Step(2)]),
            step(Op::Binary(BinOp::Mul), vec![Step(3), Leaf(1)]),
        ],
        output: Step(4),
    };
    let body = kg::fused("swiglu", &[4], &[&[4], &[4]], &k).expect("fused precondition");
    let s = spv(&body, "swiglu");
    let gate = [-2.0f32, -0.5, 0.5, 2.0];
    let up = [1.0f32, 2.0, 3.0, 4.0];
    let mut bufs = [
        KernelBuffer::read_only_f32(&gate),
        KernelBuffer::read_only_f32(&up),
        KernelBuffer::write_f32(4),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [4, 1, 1], &mut bufs)
        .unwrap();
    let want: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    for (got, e) in bufs[2].as_f32().iter().zip(&want) {
        assert!((got - e).abs() <= 1e-5, "swiglu: got {got} want {e}");
    }
}

#[test]
fn lds_workgroup_sum_probe() {
    // LDS + barrier probe: one workgroup of 8 lanes, each writes a[lane] to shared memory, barrier, lane 0 sums
    // it -> out[0]. Validates addrspace(3) globals + WorkgroupLocalWrite/Read + Barrier end to end.
    let Some(ctx) = ctx() else { return };
    let w = 8usize;
    let body = wg_sum("wgsum", w);
    let s = spv(&body, "wgsum");
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut bufs = [KernelBuffer::read_only_f32(&a), KernelBuffer::write_f32(1)];
    ctx.dispatch("test", &s, [w as u32, 1, 1], [w as u32, 1, 1], &mut bufs)
        .unwrap();
    assert_eq!(bufs[1].as_f32(), &[36.0], "workgroup sum of 1..=8");
}

/// Build the one-pass RMSNorm RowKernel (leaves x, w) for `n_cols` columns. Shared by the serial and
/// parallel equivalence test.
fn rmsnorm_kernel(n_cols: usize, eps: f32) -> kg::RowKernel {
    kg::RowKernel {
        n_leaves: 2, // x, w
        leaf_storage: vec![poot_target::BufferStorage::f32(); 2],
        n_cols,
        steps: vec![
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Binary(BinOp::Mul),
                inputs: vec![kg::FusedInput::Leaf(0), kg::FusedInput::Leaf(0)],
            },
            kg::RowOp::Reduce {
                op: kg::RowReduce::Sum,
                input: kg::FusedInput::Step(0),
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Binary(BinOp::Mul),
                inputs: vec![
                    kg::FusedInput::Step(1),
                    kg::FusedInput::Lit(1.0 / n_cols as f32),
                ],
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Binary(BinOp::Add),
                inputs: vec![kg::FusedInput::Step(2), kg::FusedInput::Lit(eps)],
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Math(MathOp::Sqrt),
                inputs: vec![kg::FusedInput::Step(3)],
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Binary(BinOp::Div),
                inputs: vec![kg::FusedInput::Leaf(0), kg::FusedInput::Step(4)],
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Binary(BinOp::Mul),
                inputs: vec![kg::FusedInput::Step(5), kg::FusedInput::Leaf(1)],
            },
        ],
        output: 2 + 6,
    }
}

#[test]
fn tiled_region_epilogue_is_bit_exact_against_the_unfused_chain() {
    // `out = (A @ B + bias) + residual` through the tiled body's epilogue equals the same body without one, followed by
    // the two f32 adds on the host: bit for bit, at a single partial tile (M = 5) and at many (M = 160), in both
    // weight layouts.
    let Some(ctx) = ctx() else { return };
    let (ts, k, n) = (8usize, 32usize, 24usize);
    let fill = |seed: u64, count: usize| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 0.4 - 0.2
            })
            .collect()
    };
    let run = |body: &Body, inputs: &[&[f32]], numel: usize, groups: usize| -> Vec<f32> {
        let s = spv(body, "tiled_region_epilogue");
        let mut bufs: Vec<KernelBuffer> = inputs
            .iter()
            .map(|input| KernelBuffer::read_only_f32(input))
            .collect();
        bufs.push(KernelBuffer::write_f32(numel));
        ctx.dispatch(
            "test",
            &s,
            [(ts * ts) as u32, 1, 1],
            [(groups * ts * ts) as u32, 1, 1],
            &mut bufs,
        )
        .unwrap();
        bufs.pop().unwrap().as_f32().to_vec()
    };
    for m in [5usize, 160] {
        for layout in [kg::WeightLayout::Kn, kg::WeightLayout::Nk] {
            let a = fill(1, m * k);
            let b = fill(2, k * n);
            let bias = fill(3, n);
            let residual = fill(4, m * n);
            let groups = m.div_ceil(2 * ts) * n.div_ceil(ts);
            let epilogue = kg::Epilogue {
                dt: Ty::F32,
                out_shape: vec![m, n],
                leaves: [vec![n], vec![m, n]]
                    .into_iter()
                    .map(|shape| kg::EpilogueLeaf {
                        shape,
                        storage: None,
                    })
                    .collect(),
                steps: vec![
                    kg::EpilogueStep {
                        op: kg::FusedScalarOp::Binary(BinOp::Add),
                        inputs: vec![kg::EpilogueInput::Head, kg::EpilogueInput::Leaf(0)],
                    },
                    kg::EpilogueStep {
                        op: kg::FusedScalarOp::Binary(BinOp::Add),
                        inputs: vec![kg::EpilogueInput::Step(0), kg::EpilogueInput::Leaf(1)],
                    },
                ],
                output: kg::EpilogueInput::Step(1),
            };
            let body = |epilogue| {
                kg::tiled_region(
                    "tiled_region_epilogue",
                    m,
                    k,
                    n,
                    ts,
                    0,
                    layout,
                    BufferStorage::f32(),
                    epilogue,
                )
                .expect("f32 weight storage")
            };
            let fused = run(
                &body(Some(&epilogue)),
                &[&a, &b, &bias, &residual],
                m * n,
                groups,
            );
            let head = run(&body(None), &[&a, &b], m * n, groups);
            let unfused: Vec<f32> = head
                .iter()
                .enumerate()
                .map(|(i, h)| (h + bias[i % n]) + residual[i])
                .collect();
            let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
            assert_eq!(
                bits(&fused),
                bits(&unfused),
                "m={m} {layout:?}: the fused epilogue differs from the unfused chain"
            );
        }
    }
}

#[test]
fn tiled_region_validates_and_matches_cpu_matmul() {
    // `kg::tiled_region` (element-by-element tiled-GEMM generator): (1) compile to SPIR-V and spirv-val (strict
    // check: in-loop barrier + LDS); (2) dispatch on the Arc and compare to a serial CPU matmul over partial-tile
    // shapes.
    let (ts, m, k, n) = (8usize, 17usize, 32usize, 24usize); // partial tiles in M (17), N (24)
    let body = kg::tiled_region(
        "tiled_region",
        m,
        k,
        n,
        ts,
        0,
        kg::WeightLayout::Kn,
        BufferStorage::f32(),
        None,
    )
    .expect("f32 weight storage");
    let s = spv(&body, "tiled_region");

    // strict SPIR-V validity (skip only if spirv-val is absent).
    let dir = std::env::temp_dir()
        .join("poot-kernelgen-test")
        .join("tiled_region");
    let spv_path = poot_codegen::artifact_path(&dir, "tiled_region", Target::SpirvVulkan);
    if which("spirv-val") {
        let v = std::process::Command::new("spirv-val")
            .arg(&spv_path)
            .output()
            .expect("run spirv-val");
        assert!(
            v.status.success(),
            "spirv-val failed:\n{}",
            String::from_utf8_lossy(&v.stderr)
        );
    } else {
        eprintln!("spirv-val not on PATH; skipping the validity check");
    }

    let Some(ctx) = ctx() else { return };
    let fill = |seed: u64, count: usize| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 0.4 - 0.2
            })
            .collect()
    };
    let a = fill(1, m * k);
    let b = fill(2, k * n);
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&b),
        KernelBuffer::write_f32(m * n),
    ];
    // grid: ceil(M/2ts) row-tiles * ceil(N/ts) col-tiles workgroups, ts*ts lanes each (matches the body).
    let groups = m.div_ceil(2 * ts) * n.div_ceil(ts);
    ctx.dispatch(
        "test",
        &s,
        [(ts * ts) as u32, 1, 1],
        [(groups * ts * ts) as u32, 1, 1],
        &mut bufs,
    )
    .unwrap();
    let got = bufs[2].as_f32();
    for i in 0..m {
        for j in 0..n {
            let want: f32 = (0..k).map(|kk| a[i * k + kk] * b[kk * n + j]).sum();
            let g = got[i * n + j];
            assert!(
                (g - want).abs() < 1e-4,
                "tiled_region [{i},{j}]: gpu {g} vs cpu {want}"
            );
        }
    }
}

/// Compile `generated`'s body, run spirv-val over it when available, and dispatch it on `operands` over its
/// own launch; returns the output.
fn run_attention(
    name: &str,
    generated: &kg::Generated,
    operands: [&[f32]; 4],
    out_len: usize,
) -> Option<Vec<f32>> {
    let s_bytes = spv(&generated.body, name);
    let dir = std::env::temp_dir().join("poot-kernelgen-test").join(name);
    let spv_path = poot_codegen::artifact_path(&dir, name, Target::SpirvVulkan);
    if which("spirv-val") {
        let v = std::process::Command::new("spirv-val")
            .arg(&spv_path)
            .output()
            .expect("run spirv-val");
        assert!(
            v.status.success(),
            "spirv-val failed:\n{}",
            String::from_utf8_lossy(&v.stderr)
        );
    } else {
        eprintln!("spirv-val not on PATH; skipping the validity check");
    }
    let ctx = ctx()?;
    let mut bufs = [
        KernelBuffer::read_only_f32(operands[0]),
        KernelBuffer::read_only_f32(operands[1]),
        KernelBuffer::read_only_f32(operands[2]),
        KernelBuffer::read_only_f32(operands[3]),
        KernelBuffer::write_f32(out_len),
    ];
    ctx.dispatch(
        "test",
        &s_bytes,
        generated.body.workgroup_size,
        generated.launch.grid,
        &mut bufs,
    )
    .unwrap();
    Some(bufs[4].as_f32().to_vec())
}

fn attention_inputs(spec: &kg::AttentionSpec) -> [Vec<f32>; 3] {
    let f = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    };
    let hkv = spec.hq / spec.n_rep;
    [
        f(1, spec.batch * spec.hq * spec.lq * spec.dk),
        f(2, spec.batch * hkv * spec.lk * spec.dk),
        f(3, spec.batch * hkv * spec.lk * spec.dv),
    ]
}

#[test]
fn attention_decode_validates_and_matches_cpu() {
    // Decode: `lanes` lanes per (batch, head) workgroup each run an online softmax over a strided slice of
    // the cache, then the LDS combine. 13 keys over 5 lanes is ragged (lanes 3 and 4 take one key fewer).
    // spirv-val is the strict check for an LDS + nested-loop + barrier kernel; the output is compared with
    // the textbook two-pass attention. The combine reassociates the sum, so the tolerance is looser than
    // a serial kernel's.
    let spec = attention_spec(2, 4, 2, 1, 13, 4, 5, kg::MaskKind::Broadcast);
    let [q, k, v] = attention_inputs(&spec);
    let mask: Vec<f32> = (0..2 * 13)
        .map(|i| {
            if i % 13 < 9 {
                0.25 * (i % 13) as f32
            } else {
                -1.0e9
            }
        })
        .collect();
    let generated = attention(&spec);
    let Some(got) = run_attention(
        "attention_decode",
        &generated,
        [&q, &k, &v, &mask],
        2 * 4 * 4,
    ) else {
        return;
    };
    let want = attention::attention_reference(&spec, &q, &k, &v, &mask);
    poot_test_util::assert_close(&got, &want, 1e-4);
}

#[test]
fn attention_prefill_validates_and_matches_cpu() {
    // Prefill: every query row attends to the prompt through a causal mask, one workgroup per (head, row).
    let spec = attention_spec(1, 4, 2, 6, 6, 4, 4, kg::MaskKind::Broadcast);
    let [q, k, v] = attention_inputs(&spec);
    let mut mask = vec![0.0f32; 6 * 6];
    for row in 0..6 {
        for j in row + 1..6 {
            mask[row * 6 + j] = -1.0e9;
        }
    }
    let generated = attention(&spec);
    let Some(got) = run_attention(
        "attention_prefill",
        &generated,
        [&q, &k, &v, &mask],
        4 * 6 * 4,
    ) else {
        return;
    };
    let want = attention::attention_reference(&spec, &q, &k, &v, &mask);
    poot_test_util::assert_close(&got, &want, 1e-4);
}

#[test]
fn attn_scores_v_gemv_lds_matches_naive_batched_matmul() {
    // Decode attention `scores @ V` LDS-parallel GEMV (kg::attn_scores_v_gemv_lds) against the naive
    // one-thread-per-output batched matmul it replaces (kg::matmul_batched_dt), both dispatched on the same random
    // data (GPU vs GPU, so a misreading of the batched-matmul semantics can't hide in two independent references).
    //
    // hq = 12 stands in for Hq after GQA's repeat_kv expanded Hkv=4 (n_rep=3); both operands have matching Hq rows
    // (no GQA math inside the kernel, see attn_scores_v_gemv_lds / is_decode_attn_gemv). cap=300 is not a multiple
    // of w=128 (300 = 2*128 + 44). d=64 is a plausible head dim.
    let (w, hq, cap, d) = (128usize, 12usize, 300usize, 64usize);
    let rows = hq; // batch = 1
    let x_groups = rows * d; // rows*d well under 65535, so a single-row (y=1) 2-D grid: col == GroupX.
    let body = kg::attn_scores_v_gemv_lds("attn_scores_v", cap, d, w, x_groups);
    let spv_bytes = spv(&body, "attn_scores_v");

    let dir = std::env::temp_dir()
        .join("poot-kernelgen-test")
        .join("attn_scores_v");
    let spv_path = poot_codegen::artifact_path(&dir, "attn_scores_v", Target::SpirvVulkan);
    if which("spirv-val") {
        let v = std::process::Command::new("spirv-val")
            .arg(&spv_path)
            .output()
            .expect("run spirv-val");
        assert!(
            v.status.success(),
            "spirv-val failed:\n{}",
            String::from_utf8_lossy(&v.stderr)
        );
    } else {
        eprintln!("spirv-val not on PATH; skipping the validity check");
    }

    let Some(ctx) = ctx() else { return };
    let f = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    };
    let probs = f(101, rows * cap);
    let v = f(202, rows * cap * d);

    let mut bufs = [
        KernelBuffer::read_only_f32(&probs),
        KernelBuffer::read_only_f32(&v),
        KernelBuffer::write_f32(rows * d),
    ];
    // threads = [x_groups*w, 1, 1]; Context divides by the kernel's baked wg [w,1,1] -> x_groups workgroups
    // on X, matching dispatch_grid's is_decode_attn_gemv arm in poot-graph-plan.
    ctx.dispatch(
        "test",
        &spv_bytes,
        [w as u32, 1, 1],
        [(x_groups * w) as u32, 1, 1],
        &mut bufs,
    )
    .unwrap();
    let got = bufs[2].as_f32().to_vec();

    // the naive one-thread-per-output batched matmul: probs[1,hq,1,cap] @ v[1,hq,cap,d] -> [1,hq,1,d].
    let out_shape = [1, hq, 1, d];
    let a_shape = [1, hq, 1, cap];
    let b_shape = [1, hq, cap, d];
    let naive_body =
        kg::matmul_batched_dt_grid("attn_naive", Ty::F32, &out_shape, &a_shape, &b_shape, None);
    let naive_bytes = spv(&naive_body, "attn_scores_v_naive");
    let out_numel = rows * d;
    let mut naive_bufs = [
        KernelBuffer::read_only_f32(&probs),
        KernelBuffer::read_only_f32(&v),
        KernelBuffer::write_f32(out_numel),
    ];
    // matmul_batched_dt's default workgroup_size is [64,1,1] (Body::new's default; matmul_batched_impl never
    // overrides it) and it indexes by the global thread id (IndexAxis::X), guarded by `i < len(out)`.
    ctx.dispatch(
        "test",
        &naive_bytes,
        [64, 1, 1],
        [out_numel as u32, 1, 1],
        &mut naive_bufs,
    )
    .unwrap();
    let want = naive_bufs[2].as_f32();

    let mut max_abs = 0.0f32;
    for i in 0..out_numel {
        let g = got[i];
        let e = want[i];
        let diff = (g - e).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        assert!(
            diff <= 1e-3,
            "attn_scores_v_gemv[{i}]: gpu(lds) {g} vs gpu(naive) {e} diff={diff}"
        );
    }
    eprintln!(
        "synthesized attn scores@V LDS decode GEMV (kg::attn_scores_v_gemv_lds) matches the naive batched \
         matmul on RADV, max_abs={max_abs:.2e}"
    );
}

/// shared with `fused_row_parallel_matches_serial_softmax` and the CI gate test below.
fn softmax_kernel(n_cols: usize) -> kg::RowKernel {
    kg::RowKernel {
        n_leaves: 1,
        leaf_storage: vec![poot_target::BufferStorage::f32()],
        n_cols,
        steps: vec![
            kg::RowOp::Reduce {
                op: kg::RowReduce::Max,
                input: kg::FusedInput::Leaf(0),
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Binary(BinOp::Sub),
                inputs: vec![kg::FusedInput::Leaf(0), kg::FusedInput::Step(0)],
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Math(MathOp::Exp),
                inputs: vec![kg::FusedInput::Step(1)],
            },
            kg::RowOp::Reduce {
                op: kg::RowReduce::Sum,
                input: kg::FusedInput::Step(2),
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Binary(BinOp::Div),
                inputs: vec![kg::FusedInput::Step(2), kg::FusedInput::Step(3)],
            },
        ],
        output: 1 + 4,
    }
}

/// spirv-val CI gate for the hot synthesized kernels the wgpu engine dispatches on RADV, including six that
/// were never run through `spirv-val` before: `gemv_lds` (base f32 decode GEMV), `fused_row_parallel`
/// (RMSNorm/softmax decode row-kernel), `rope_dt`, `arg_top_k_dt`, `matmul_batched_dt` in F32 (the F16
/// instantiation is validated by `f16_matmul_probe.rs`), and the 4 `cast_*` narrow/widen kernels. RADV
/// tolerates some invalid SPIR-V that a strict validator or a WebGPU backend rejects, so a kernel that
/// works here only proves wgpu-on-RADV (see the card 147 atomics bug, caught only by
/// `global_atomic_add_counter_validates_as_spirv`).
///
/// Compile-time only, no GPU dispatch; intentionally redundant with the per-kernel `spirv-val` checks in the
/// dequant/flash/attn tests above, so "is every hot kernel spirv-val'd" has one answer. Every kernel is built
/// at the same small decode-sized shape as `synthesized_kernels_compile_to_valid_ptx`, compiled to
/// `SpirvVulkan`, and run through `spirv-val --target-env vulkan1.3`. All failures are collected before
/// asserting. Skips (passes) if `spirv-val` is absent.
///
/// `cast_f32_to_bf16`/`cast_bf16_to_f32` are not in the must-pass list: bf16 compute is NVPTX-only
/// (`emit.rs`'s `scalar_llty`; the Arc has no `SPV_KHR_bfloat16`), so `compile(.., SpirvVulkan)`
/// must reject them. They are still gated here (silently emitting SPIR-V for bf16 would be as dangerous as an
/// unvalidated kernel), via "compile must reject with the documented reason"
#[test]
fn every_hot_synthesized_kernel_validates_as_spirv() {
    if !which("spirv-val") {
        eprintln!(
            "spirv-val not on PATH; skipping every_hot_synthesized_kernel_validates_as_spirv"
        );
        return;
    }

    // small decode-sized shape shared with `synthesized_kernels_compile_to_valid_ptx` above (n=24, w=128):
    // 256 = one Q4_K/Q5_K/Q6_K/Q2_K/Q3_K super-block (the quant kernels require K a multiple of 256).
    let (n, w) = (24usize, 128usize);
    let eps = 1e-6f32;

    // bf16 casts: the Arc has no SPIR-V bf16 support by design; assert the rejection stays intact rather than
    // spirv-val PASS.
    let nvptx_only_kernels: Vec<(&str, Body)> = vec![
        (
            "cast_f32_to_bf16",
            kg::cast_f32_to_bf16("cast_f32_bf16_gate"),
        ),
        (
            "cast_bf16_to_f32",
            kg::cast_bf16_to_f32("cast_bf16_f32_gate"),
        ),
    ];

    let kernels: Vec<(&str, Body)> = vec![
        (
            "gemv_lds",
            // shape mirrors pootc/tests/import_run.rs's kernelgen-vs-imported gemv_lds parity check.
            kg::gemv_lds(
                "gemv_lds_gate",
                128,
                8,
                128,
                8,
                0,
                kg::WeightLayout::Kn,
                BufferStorage::f32(),
            )
            .expect("f32 weight storage"),
        ),
        (
            "fused_row_parallel (rmsnorm)",
            kg::fused_row_parallel_views(
                "frp_rms_gate",
                &[3, 8],
                &[&[3usize, 8][..], &[8usize][..]],
                &[
                    kg::Layout::contiguous(&[3usize, 8]),
                    kg::Layout::contiguous(&[8usize]),
                ],
                &rmsnorm_kernel(8, eps),
                4,
                1,
            )
            .expect("fused_row_parallel precondition"),
        ),
        (
            "fused_row_parallel (softmax)",
            kg::fused_row_parallel_views(
                "frp_sm_gate",
                &[2, 10],
                &[&[2usize, 10][..]],
                &[kg::Layout::contiguous(&[2usize, 10])],
                &softmax_kernel(10),
                4,
                1,
            )
            .expect("fused_row_parallel precondition"),
        ),
        (
            "attention",
            attention(&attention_spec(
                1,
                4,
                2,
                1,
                8,
                4,
                3,
                kg::MaskKind::Broadcast,
            ))
            .body,
        ),
        (
            "attn_scores_v_gemv_lds",
            kg::attn_scores_v_gemv_lds("asvg_gate", 300, n, w, n),
        ),
        (
            "rope_dt",
            kg::rope_dt("rope_gate", Ty::F32, &[1, 4, 8], &[8], 8).expect("rope_dt precondition"),
        ),
        ("arg_top_k_dt", kg::arg_top_k_dt("argtopk_gate", 8, 4)),
        (
            "matmul_batched_dt (F32)",
            kg::matmul_batched_dt_grid("mmbdt_f32_gate", Ty::F32, &[4, 6], &[4, 8], &[8, 6], None),
        ),
        (
            "matmul_batched_dt (F16)",
            kg::matmul_batched_dt_grid("mmbdt_f16_gate", Ty::F16, &[4, 6], &[4, 8], &[8, 6], None),
        ),
        ("cast_f32_to_f16", kg::cast_f32_to_f16("cast_f32_f16_gate")),
        ("cast_f16_to_f32", kg::cast_f16_to_f32("cast_f16_f32_gate")),
    ];

    let dir = std::env::temp_dir()
        .join("poot-kernelgen-test")
        .join("hot_kernel_spirv_gate");
    std::fs::create_dir_all(&dir).unwrap();

    let mut failures: Vec<String> = Vec::new();
    let mut passed: Vec<&str> = Vec::new();
    for (label, body) in &kernels {
        let out = poot_codegen::artifact_path(&dir, label, Target::SpirvVulkan);
        if let Err(e) = compile(body, Target::SpirvVulkan, &out) {
            failures.push(format!("{label}: FAILED TO COMPILE to SPIR-V: {e}"));
            continue;
        }
        let v = std::process::Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&out)
            .output()
            .expect("run spirv-val");
        if v.status.success() {
            passed.push(label);
        } else {
            failures.push(format!(
                "{label}: spirv-val FAILED:\n{}",
                String::from_utf8_lossy(&v.stderr)
            ));
        }
    }

    // the bf16 casts: compile(.., SpirvVulkan) must still reject them with the documented "NVPTX-only" reason;
    // a silent flip to Ok(..) would mean SPIR-V started emitting bf16 without Vulkan bf16 support.
    let mut nvptx_only_confirmed: Vec<&str> = Vec::new();
    for (label, body) in &nvptx_only_kernels {
        let out = poot_codegen::artifact_path(&dir, label, Target::SpirvVulkan);
        match compile(body, Target::SpirvVulkan, &out) {
            Err(e) if format!("{e}").contains("NVPTX-only") => nvptx_only_confirmed.push(label),
            Err(e) => failures.push(format!(
                "{label}: rejected compile to SpirvVulkan, but NOT for the expected bf16/NVPTX-only \
                 reason (the guard in emit.rs may have changed): {e}"
            )),
            Ok(_) => failures.push(format!(
                "{label}: compile(.., SpirvVulkan) UNEXPECTEDLY SUCCEEDED - bf16 was thought to be \
                 NVPTX-only (no SPV_KHR_bfloat16 on the Arc); if this is now supported, move it into the \
                 must-pass-spirv-val list instead of the nvptx_only_kernels list"
            )),
        }
    }

    let total = kernels.len() + nvptx_only_kernels.len();
    eprintln!(
        "every_hot_synthesized_kernel_validates_as_spirv: {}/{} passed spirv-val --target-env vulkan1.3: [{}]; {}/{} bf16 casts confirmed NVPTX-only (correctly SPIR-V-unsupported): [{}]",
        passed.len(),
        kernels.len(),
        passed.join(", "),
        nvptx_only_confirmed.len(),
        nvptx_only_kernels.len(),
        nvptx_only_confirmed.join(", ")
    );
    assert!(
        failures.is_empty(),
        "{} of {} hot kernels FAILED the gate:\n\n{}",
        failures.len(),
        total,
        failures.join("\n\n")
    );
}

/// Is `tool` on PATH? (small local helper for the optional spirv-val check.)
fn which(tool: &str) -> bool {
    std::process::Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `tanh(head + bias) * 0.5 + residual` over a head of `out_shape` and dtype `dt`: a bias leaf `[N]`, a
/// literal scale and a residual leaf of the head's shape.
fn chain_epilogue(dt: Ty, out_shape: &[usize]) -> kg::Epilogue {
    let n = out_shape[out_shape.len() - 1];
    kg::Epilogue {
        dt,
        out_shape: out_shape.to_vec(),
        leaves: [vec![n], out_shape.to_vec()]
            .into_iter()
            .map(|shape| kg::EpilogueLeaf {
                shape,
                storage: None,
            })
            .collect(),
        steps: vec![
            kg::EpilogueStep {
                op: kg::FusedScalarOp::Binary(BinOp::Add),
                inputs: vec![kg::EpilogueInput::Head, kg::EpilogueInput::Leaf(0)],
            },
            kg::EpilogueStep {
                op: kg::FusedScalarOp::Tanh,
                inputs: vec![kg::EpilogueInput::Step(0)],
            },
            kg::EpilogueStep {
                op: kg::FusedScalarOp::Binary(BinOp::Mul),
                inputs: vec![kg::EpilogueInput::Step(1), kg::EpilogueInput::Lit(0.5)],
            },
            kg::EpilogueStep {
                op: kg::FusedScalarOp::Binary(BinOp::Add),
                inputs: vec![kg::EpilogueInput::Step(2), kg::EpilogueInput::Leaf(1)],
            },
        ],
        output: kg::EpilogueInput::Step(3),
    }
}

#[test]
fn matmul_epilogue_chain_matches_cpu() {
    // `C = tanh(A@B + bias) * 0.5 + residual` in one kernel matches the CPU evaluation of the same chain: A[1,2,3,4] @
    // B[1,1,4,5] (the weight's size-1 batch axis broadcasts), a bias [5] broadcast over every row, a full residual.
    let Some(ctx) = ctx() else { return };
    let a: Vec<f32> = (0..24).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
    let b: Vec<f32> = (0..20).map(|i| ((i % 5) as f32) * 0.2 - 0.4).collect();
    let bias: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5];
    let residual: Vec<f32> = (0..30).map(|i| ((i % 11) as f32) * 0.05 - 0.2).collect();
    let out_shape = [1usize, 2, 3, 5];
    let body = kg::matmul_batched_epilogue_dt_grid(
        "mm_epilogue",
        Ty::F32,
        &out_shape,
        &[1, 2, 3, 4],
        &[1, 1, 4, 5],
        &chain_epilogue(Ty::F32, &out_shape),
        None,
    )
    .expect("the epilogue's leaf storage is valid");
    let s = spv(&body, "mm_epilogue");
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&b),
        KernelBuffer::read_only_f32(&bias),
        KernelBuffer::read_only_f32(&residual),
        KernelBuffer::write_f32(30),
    ];
    ctx.dispatch("test", &s, [64, 1, 1], [30, 1, 1], &mut bufs)
        .unwrap();
    for batch in 0..2 {
        for row in 0..3 {
            for col in 0..5 {
                let dot: f32 = (0..4)
                    .map(|k| a[batch * 12 + row * 4 + k] * b[k * 5 + col])
                    .sum();
                let want = (dot + bias[col]).tanh() * 0.5 + residual[batch * 15 + row * 5 + col];
                let got = bufs[4].as_f32()[batch * 15 + row * 5 + col];
                assert!(
                    (got - want).abs() < 1e-5,
                    "[{batch},{row},{col}]: got {got}, want {want}"
                );
            }
        }
    }
}

#[test]
fn epilogue_bodies_compile_to_valid_spirv_and_nvptx() {
    use kg::{ContractionSpec, KernelRequest, MatmulShapes, Schedule};
    use poot_target::DeviceCaps;
    let caps = DeviceCaps::wgpu_rdna3_igpu();
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    let fold = |numel| kg::Fold {
        numel,
        two_d: false,
        width: 64,
    };
    let requests: Vec<(&str, KernelRequest)> = vec![
        (
            "epilogue_serial_f32",
            KernelRequest::Contraction(ContractionSpec::Serial {
                dt: Ty::F32,
                epilogue: Some(chain_epilogue(Ty::F32, &[2, 3, 5])),
                shapes: MatmulShapes {
                    out_shape: vec![2, 3, 5],
                    a_shape: vec![2, 3, 4],
                    b_shape: vec![2, 4, 5],
                },
                fold: fold(30),
            }),
        ),
        (
            "epilogue_dense_tiled_bf16_weight",
            KernelRequest::Contraction(ContractionSpec::DenseTiled {
                m: 3,
                k: 4,
                n: 5,
                tile: 8,
                offset: 0,
                groups: 1,
                weight: BufferStorage::bf16_packed(),
                epilogue: Some(chain_epilogue(Ty::F32, &[3, 5])),
            }),
        ),
        (
            "epilogue_tiled_region_f32",
            KernelRequest::Contraction(ContractionSpec::TiledRegion {
                m: 3,
                k: 4,
                n: 5,
                tile: 8,
                offset: 0,
                groups: 1,
                epilogue: Some(chain_epilogue(Ty::F32, &[3, 5])),
            }),
        ),
        (
            "epilogue_dense_gemv_f16_weight",
            KernelRequest::Contraction(ContractionSpec::DenseGemv {
                weight: BufferStorage::f16_packed(),
                schedule: Schedule::Gemv {
                    width: 16,
                    cols: 2,
                    unroll: 2,
                },
                k: 8,
                n: 5,
                epilogue: Some(chain_epilogue(Ty::F32, &[1, 5])),
            }),
        ),
    ];
    for (name, request) in &requests {
        let generated =
            kg::generate(request, &caps, &limits).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(spirv_word_count(&spv(&generated.body, name)) > 0, "{name}");
    }
    // The tiled bodies' epilogue stores lower to NVPTX too (llc accepts the IR; the device run is the PTX pod lane).
    if which("llc") {
        let ptx_caps = DeviceCaps::ptx_default();
        for (name, request) in requests.iter().filter(|(name, _)| name.contains("tiled")) {
            let generated =
                kg::generate(request, &ptx_caps, &limits).unwrap_or_else(|e| panic!("{name}: {e}"));
            let asm = ptx(&generated.body, &format!("{name}_nvptx"));
            assert!(
                !asm.contains(".extern .func"),
                "{name}: an unresolved intrinsic in the NVPTX output"
            );
        }
    }
    // The narrow-head bodies hold a BF16 or F16 buffer, which SPIR-V has no scalar type for: NVPTX only here (ROCm
    // compiles and runs them in `poot-rocm-device`'s cross_backend rows).
    if which("llc") {
        for dt in [Ty::BF16, Ty::F16] {
            let body = kg::matmul_batched_epilogue_dt_grid(
                "epilogue_serial_narrow",
                dt.clone(),
                &[2, 3, 5],
                &[2, 3, 4],
                &[2, 4, 5],
                &chain_epilogue(dt.clone(), &[2, 3, 5]),
                None,
            )
            .expect("the epilogue's leaf storage is valid");
            let asm = ptx(&body, &format!("epilogue_serial_{dt:?}"));
            assert!(
                !asm.contains(".extern .func"),
                "{dt:?}: an unresolved intrinsic in the NVPTX output"
            );
        }
    } else {
        eprintln!("llc not on PATH; skipping the NVPTX-compile check");
    }
}
