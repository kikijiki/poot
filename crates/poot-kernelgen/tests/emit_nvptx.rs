//! Emit NVPTX `.ptx` for a representative set of generated kernels to `poot_test_util::ptx_stage_dir()`, so they can be
//! verified on a real NVIDIA GPU via ptx-check (exp ex2.approx, the silu composition, div/rem, cast
//! f32->usize, the index-math kernels). Run inside `nix develop` (needs llc). Asserts only that llc accepts the
//! output; the check is the on-GPU run.

use poot_codegen::{Target, compile};
use poot_kernel_ir::test_support::{matmul_kernel, square_kernel, wg_sum};
use poot_kernel_ir::{BinOp, Ty, fixtures};
use poot_kernelgen as kg;
use poot_kernelgen::reduce_last_dt;

#[path = "support/gather.rs"]
mod gather;
use gather::gather_axis0_dt;

/// The f32-element movement `op` into `out_shape`.
fn movement(name: &str, out_shape: Vec<usize>, op: kg::MovementOp) -> poot_kernel_ir::Body {
    kg::movement_body(
        name,
        &kg::MovementSpec::new(kg::Storage::Elements { carrier: Ty::F32 }, out_shape, op),
        None,
    )
    .expect("a valid movement request")
}

/// The attention generator's decode body through the one generation seam, on a PTX device.
fn attention_decode_body() -> poot_kernel_ir::Body {
    let spec = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
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
    // The PTX entry and the staged file carry the body's name; `ptx-check` runs `attention_decode`.
    body.name = "attention_decode".to_string();
    body
}

#[test]
fn emit_nvptx_kernel_set() {
    fn have_llc() -> bool {
        std::process::Command::new("llc")
            .arg("--version")
            .output()
            .is_ok()
    }
    if !have_llc() {
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
        return;
    }
    let dir = poot_test_util::ptx_stage_dir();
    std::fs::create_dir_all(&dir).unwrap();
    // a pointwise fused region and the two row-wise (reduction-rooted) kernels, whose multi-block control flow is
    // the NVPTX-lowering risk.
    let fused_pointwise = kg::fused(
        "fused_pw",
        &[4],
        &[&[4], &[4]],
        &kg::FusedKernel {
            leaf_tys: vec![kg::LeafTy::F32; 2],
            steps: vec![kg::FusedStep {
                op: kg::FusedScalarOp::Binary(BinOp::Add),
                inputs: vec![
                    kg::FusedInput::Leaf(kg::LeafId::new(0)),
                    kg::FusedInput::Leaf(kg::LeafId::new(1)),
                ],
            }],
            output: kg::FusedInput::Step(kg::StepId::new(0)),
        },
    )
    .expect("fused precondition");
    // `fused_row`/`fused_row_parallel_views` (rmsnorm+softmax row kernels), `reduce_last_lds` and
    // `array_sum_probe` are this crate's own `#[cfg(test)]`-only kernels (card
    // 622: no cross-crate caller), so their NVPTX staging to this same drop point
    // lives in their own in-crate unit tests (`unit_helpers::stage_nvptx`) instead of here.
    let fused_unary = |name: &str, op| {
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
    };
    let bodies = [
        fused_unary("tanh", kg::FusedScalarOp::Tanh),
        fused_unary("erf", kg::FusedScalarOp::Erf),
        reduce_last_dt("reduce", Ty::F32, BinOp::Add, 4, 0.0),
        movement(
            "bcast",
            vec![2, 3],
            kg::MovementOp::Broadcast { in_shape: vec![3] },
        ),
        gather_axis0_dt("gather", Ty::F32, 2),
        movement(
            "dus",
            vec![2, 4, 3],
            kg::MovementOp::UpdateSlice {
                update_shape: vec![2, 1, 3],
                axis: 1,
                index: kg::UpdateIndex::Static(2),
            },
        ),
        fused_pointwise,
        // LDS + barrier: the NVPTX .shared + bar.sync lowering.
        wg_sum("wg_sum", 8),
        // the attention kernel (online softmax, LDS partials): a GQA decode, Hq=4, n_rep=2, 6 keys, D=8, over 4
        // lanes.
        attention_decode_body(),
        // base codegen fixtures, so this one emit also stages everything ptx-check needs (one dir).
        fixtures::add_kernel(),
        square_kernel(),
        matmul_kernel(2, 3, 4),
    ];
    for body in &bodies {
        let out = dir.join(format!("{}.ptx", body.name));
        compile(body, Target::Nvptx, &out).unwrap_or_else(|e| panic!("{}: {e}", body.name));
        assert!(std::fs::metadata(&out).unwrap().len() > 0);
    }
    eprintln!("emitted NVPTX kernels to {}", dir.display());
}

// `private_array_rejected_on_spirv` moved to `array_sum_probe`'s own in-crate unit test (card 622:
// `array_sum_probe` is this crate's own `#[cfg(test)]`-only kernel, no cross-crate caller).
