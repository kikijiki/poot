//! Card 531b SC-001: one test per generator family, calling the generator directly with a shape or
//! count that violates its own precondition, and checking the typed [`poot_kernelgen::KernelGenError`]
//! it returns. At the baseline (before this card) each of these called `assert!`/`assert_eq!` and
//! panicked inside the planner chain (R484-010); the mutation for each row restores that assert (or
//! removes the early return) so the test observably goes red, then the fix is restored so it is green.

use poot_kernel_ir::Ty;
use poot_kernelgen::KernelGenError;

#[test]
fn wmma_family_rejects_non_divisible_dim_and_non_exact_k() {
    // MUTATION target: matmul_tensorcore's M/N/K %16 check.
    let err = poot_kernelgen::matmul_tensorcore("t", Ty::F32, &[17, 16], &[17, 16], &[16, 16])
        .expect_err("M=17 is not a multiple of 16");
    assert_eq!(
        err,
        KernelGenError::NotDivisible {
            generator: "matmul_tensorcore",
            dim: "M".to_string(),
            value: 17,
            divisor: 16,
        }
    );
    // Same family, the coopmat sibling's exact-K contract (a different KernelGenError variant).
    let err = poot_kernelgen::matmul_tensorcore_coopmat("t", &[16, 16], &[16, 17], &[17, 16])
        .expect_err("K=17 must equal 16 exactly");
    assert_eq!(
        err,
        KernelGenError::NotEqualTo {
            generator: "matmul_tensorcore_coopmat",
            dim: "K".to_string(),
            value: 17,
            expected: 16,
        }
    );
}

#[test]
fn fused_family_rejects_leaf_count_mismatch_and_zero_width() {
    // MUTATION target: fused_views's `leaf_shapes.len() == k.n_leaves` check.
    let kfk = poot_kernelgen::FusedKernel {
        n_leaves: 2,
        steps: vec![],
        output: poot_kernelgen::FusedInput::Leaf(0),
    };
    let layouts = vec![poot_kernelgen::Layout::contiguous(&[4])];
    let err = poot_kernelgen::fused_views("t", &[4], &[&[4]], &layouts, &kfk)
        .expect_err("1 leaf shape given, kernel wants 2");
    assert_eq!(
        err,
        KernelGenError::CountMismatch {
            generator: "fused_views",
            what: "leaf shape count".to_string(),
            expected: 2,
            actual: 1,
        }
    );
    // Same family, the row-parallel sibling's workgroup-width contract.
    let krk = poot_kernelgen::RowKernel {
        n_leaves: 1,
        n_cols: 4,
        steps: vec![],
        output: 0,
    };
    let layouts = vec![poot_kernelgen::Layout::contiguous(&[4])];
    let err = poot_kernelgen::fused_row_parallel_views("t", &[4], &[&[4]], &layouts, &krk, 0, 1)
        .expect_err("workgroup width 0 is invalid");
    assert_eq!(
        err,
        KernelGenError::BelowMinimum {
            generator: "fused_row_parallel_views",
            what: "workgroup width".to_string(),
            value: 0,
            min: 1,
        }
    );
}

#[test]
fn fp8_family_rejects_rank_and_count_mismatches() {
    // MUTATION target: e4m3fn_transpose_packed's out_shape/in_shape rank check.
    let err = poot_kernelgen::e4m3fn_transpose_packed("t", &[2, 3], &[3, 2, 1], &[0, 1])
        .expect_err("out_shape rank 2 != in_shape rank 3");
    assert_eq!(
        err,
        KernelGenError::CountMismatch {
            generator: "e4m3fn_transpose_packed",
            what: "out_shape/in_shape rank".to_string(),
            expected: 3,
            actual: 2,
        }
    );
    // Same family, concat's minimum-input-count contract (a different KernelGenError variant).
    let err = poot_kernelgen::e4m3fn_concat_packed("t", &[2, 3], 1, &[&[2, 3]])
        .expect_err("concat needs at least 2 inputs, got 1");
    assert_eq!(
        err,
        KernelGenError::BelowMinimum {
            generator: "e4m3fn_concat_packed",
            what: "in_shapes count".to_string(),
            value: 1,
            min: 2,
        }
    );
}

#[test]
fn rope_family_rejects_zero_rank_and_odd_rot() {
    // MUTATION target: rope_dt's `r >= 1` rank check.
    let err = poot_kernelgen::rope_dt("t", Ty::F32, &[], &[8], 8)
        .expect_err("x_shape rank 0 has no last (head) dim");
    assert_eq!(
        err,
        KernelGenError::BelowMinimum {
            generator: "rope_dt",
            what: "x_shape rank".to_string(),
            value: 0,
            min: 1,
        }
    );
    // Same family, rot's evenness contract (a different KernelGenError variant).
    let err = poot_kernelgen::rope_dt("t", Ty::F32, &[1, 4, 8], &[8], 7).expect_err("rot=7 is odd");
    assert_eq!(
        err,
        KernelGenError::NotDivisible {
            generator: "rope_dt",
            dim: "rot".to_string(),
            value: 7,
            divisor: 2,
        }
    );
}

#[test]
fn concat_family_rejects_zero_inputs() {
    // MUTATION target: concat_n_dt's `n >= 1` check.
    let err = poot_kernelgen::concat_n_dt("t", Ty::F32, &[0], 0, &[])
        .expect_err("concat needs at least one input");
    assert_eq!(
        err,
        KernelGenError::BelowMinimum {
            generator: "concat_n_dt",
            what: "in_shapes count".to_string(),
            value: 0,
            min: 1,
        }
    );
}
