//! Emit NVPTX `.ptx` for a representative set of generated kernels to /tmp/poot-ptx-stage2, so they can be
//! verified on a real NVIDIA GPU via poot-ptx-check (exp ex2.approx, the silu composition, div/rem, cast
//! f32->usize, the index-math kernels). Run inside `nix develop` (needs llc). Asserts only that llc accepts the
//! output; the check is the on-GPU run.

use std::path::PathBuf;

use poot_codegen::{Target, compile};
use poot_kernel_ir::{BinOp, Ty, fixtures};
use poot_kernelgen as kg;
use poot_test_util::kernel_fixtures::{
    gather_axis0_dt, matmul_kernel, reduce_last, square_kernel, wg_sum,
};

#[test]
fn emit_nvptx_kernel_set() {
    fn have_llc() -> bool {
        std::process::Command::new("llc")
            .arg("--version")
            .output()
            .is_ok()
    }
    if !have_llc() {
        eprintln!("llc not on PATH; skipping");
        return;
    }
    let dir = PathBuf::from("/tmp/poot-ptx-stage2");
    std::fs::create_dir_all(&dir).unwrap();
    // a pointwise fused region and the two row-wise (reduction-rooted) kernels, whose multi-block control flow is
    // the NVPTX-lowering risk.
    let fused_pointwise = kg::fused(
        "fused_pw",
        &[4],
        &[&[4], &[4]],
        &kg::FusedKernel {
            n_leaves: 2,
            steps: vec![kg::FusedStep {
                op: kg::FusedScalarOp::Binary(BinOp::Add),
                inputs: vec![kg::FusedInput::Leaf(0), kg::FusedInput::Leaf(1)],
            }],
            output: kg::FusedInput::Step(0),
        },
    )
    .expect("fused precondition");
    // `fused_row`/`fused_row_parallel_views` (rmsnorm+softmax row kernels), `math_unary_dt` (transcendental
    // exp), `reduce_last_lds` and `array_sum_probe` are this crate's own `#[cfg(test)]`-only kernels (card
    // 622: no cross-crate caller), so their NVPTX staging to this same `/tmp/poot-ptx-stage2` drop point
    // lives in their own in-crate unit tests (`test_support::stage_nvptx`) instead of here.
    let fused_unary = |name: &str, op| {
        kg::fused(
            name,
            &[4],
            &[&[4]],
            &kg::FusedKernel {
                n_leaves: 1,
                steps: vec![kg::FusedStep {
                    op,
                    inputs: vec![kg::FusedInput::Leaf(0)],
                }],
                output: kg::FusedInput::Step(0),
            },
        )
        .expect("fused precondition")
    };
    let bodies = [
        fused_unary("tanh", kg::FusedScalarOp::Tanh),
        fused_unary("erf", kg::FusedScalarOp::Erf),
        reduce_last("reduce", BinOp::Add, 4, 0.0),
        kg::broadcast("bcast", &[2, 3], &[3]),
        gather_axis0_dt("gather", Ty::F32, 2),
        kg::dyn_update_slice_dt("dus", Ty::F32, &[2, 4, 3], 1, 2, 1),
        fused_pointwise,
        // LDS + barrier: the NVPTX .shared + bar.sync lowering.
        wg_sum("wg_sum", 8),
        // the flash-attention decode kernel (online softmax, private o[D]), NVPTX-only. GQA decode shapes: Hq=4,
        // n_rep=2, cap=6, D=8.
        kg::flash_attention_decode("flash_decode", 4, 2, 6, 8, 0.35355338, false),
        // base codegen fixtures, so this one emit also stages everything poot-ptx-check needs (one dir).
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
