//! Card 531b SC-001: one test per generator family, calling the generator directly with a shape or
//! count that violates its own precondition, and checking the typed [`poot_kernelgen::KernelGenError`]
//! it returns. At the baseline (before this card) each of these called `assert!`/`assert_eq!` and
//! panicked inside the planner chain (R484-010); the mutation for each row restores that assert (or
//! removes the early return) so the test observably goes red, then the fix is restored so it is green.

use poot_kernel_ir::Ty;
use poot_kernelgen::KernelGenError;

#[test]
fn wmma_family_rejects_non_divisible_dim_and_k_tail() {
    // Mutation target: matmul_tensorcore's M/N/K %16 check.
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
    // Same family, the coopmat sibling: a ragged K tail is refused, never padded (card 663).
    let err = poot_kernelgen::matmul_tensorcore_coopmat("t", &[16, 16], &[16, 17], &[17, 16])
        .expect_err("K=17 is not a multiple of 16");
    assert_eq!(
        err,
        KernelGenError::NotDivisible {
            generator: "matmul_tensorcore_coopmat",
            dim: "K".to_string(),
            value: 17,
            divisor: 16,
        }
    );
}

#[test]
fn fused_family_rejects_leaf_count_mismatch_and_zero_width() {
    // Mutation target: fused_views's `leaf_shapes.len() == k.n_leaves` check.
    let kfk = poot_kernelgen::FusedKernel {
        leaf_tys: vec![poot_kernelgen::LeafTy::F32; 2],
        steps: vec![],
        output: poot_kernelgen::FusedInput::Leaf(0),
    };
    let layouts = vec![poot_kernelgen::Layout::contiguous(&[4])];
    let err = poot_kernelgen::fused_views("t", &[4], &[&[4]], &layouts, &kfk, None)
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
        leaf_storage: vec![poot_target::BufferStorage::f32()],
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
fn movement_family_rejects_rank_and_count_mismatches() {
    // Mutation target: the transpose input-rank check in `MovementSpec::validate`.
    let spec = poot_kernelgen::MovementSpec::new;
    let err = poot_kernelgen::movement_body(
        "t",
        &spec(
            poot_kernelgen::Storage::BYTE_LANES,
            vec![2, 3],
            poot_kernelgen::MovementOp::Transpose {
                in_shape: vec![3, 2, 1],
                perm: vec![0, 1],
            },
        ),
        None,
    )
    .expect_err("out_shape rank 2 != in_shape rank 3");
    assert_eq!(
        err,
        KernelGenError::CountMismatch {
            generator: "data_movement",
            what: "in_shape rank".to_string(),
            expected: 2,
            actual: 3,
        }
    );
    // Same family, concat's extent-sum contract (a different KernelGenError variant).
    let err = poot_kernelgen::movement_body(
        "t",
        &spec(
            poot_kernelgen::Storage::BYTE_LANES,
            vec![2, 4],
            poot_kernelgen::MovementOp::Concat {
                axis: 1,
                in_shapes: vec![vec![2, 3]],
            },
        ),
        None,
    )
    .expect_err("one input of extent 3 does not fill an axis of 4");
    assert_eq!(
        err,
        KernelGenError::NotEqualTo {
            generator: "data_movement",
            dim: "concat extents on axis 1".to_string(),
            value: 3,
            expected: 4,
        }
    );
}

#[test]
fn rope_family_rejects_zero_rank_and_odd_rot() {
    // Mutation target: rope_dt's `r >= 1` rank check.
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
    // Mutation target: the `in_shapes.is_empty()` check in `MovementSpec::validate`.
    let err = poot_kernelgen::movement_body(
        "t",
        &poot_kernelgen::MovementSpec::new(
            poot_kernelgen::Storage::Elements { carrier: Ty::F32 },
            vec![1],
            poot_kernelgen::MovementOp::Concat {
                axis: 0,
                in_shapes: vec![],
            },
        ),
        None,
    )
    .expect_err("concat needs at least one input");
    assert_eq!(
        err,
        KernelGenError::BelowMinimum {
            generator: "data_movement",
            what: "concat inputs".to_string(),
            value: 0,
            min: 1,
        }
    );
}

/// POOT-1033: an I32 leaf of a float region is a select condition and nothing else. Reading it as a float
/// operand (here the left operand of an `Add`) is the typed `IndexLeafOperand`, never a kernel that reads the
/// condition word as float bits. Mutation: drop `check_float_region_tys` from `fused_views`; the body builds and
/// this row goes red.
#[test]
fn float_region_rejects_an_i32_leaf_outside_a_select_condition() {
    use poot_kernelgen::{FusedInput, FusedKernel, FusedScalarOp, FusedStep, LeafTy};
    let read_as_float = FusedKernel {
        leaf_tys: vec![LeafTy::I32, LeafTy::F32],
        steps: vec![FusedStep {
            op: FusedScalarOp::Binary(poot_kernel_ir::BinOp::Add),
            inputs: vec![FusedInput::Leaf(0), FusedInput::Leaf(1)],
        }],
        output: FusedInput::Step(0),
    };
    let layouts = vec![poot_kernelgen::Layout::contiguous(&[4]); 2];
    let err = poot_kernelgen::fused_views("t", &[4], &[&[4], &[4]], &layouts, &read_as_float, None)
        .expect_err("an I32 leaf feeds a float Add");
    assert!(
        matches!(err, KernelGenError::IndexLeafOperand { leaf: 0, .. }),
        "{err:?}"
    );

    // The same leaf as the condition of a select is the one admitted use.
    let as_condition = FusedKernel {
        leaf_tys: vec![LeafTy::I32, LeafTy::F32, LeafTy::F32],
        steps: vec![FusedStep {
            op: FusedScalarOp::Select,
            inputs: vec![
                FusedInput::Leaf(0),
                FusedInput::Leaf(1),
                FusedInput::Leaf(2),
            ],
        }],
        output: FusedInput::Step(0),
    };
    let layouts = vec![poot_kernelgen::Layout::contiguous(&[4]); 3];
    poot_kernelgen::fused_views(
        "t",
        &[4],
        &[&[4], &[4], &[4]],
        &layouts,
        &as_condition,
        None,
    )
    .expect("an I32 condition leaf beside two float branches");

    // And a float leaf has no place in an exact-I32 region.
    let float_leaf = FusedKernel {
        leaf_tys: vec![LeafTy::F32],
        steps: vec![],
        output: FusedInput::Leaf(0),
    };
    let layouts = vec![poot_kernelgen::Layout::contiguous(&[4])];
    let err = poot_kernelgen::fused_i32_views_grid("t", &[4], &[&[4]], &layouts, &float_leaf, None)
        .expect_err("an F32 leaf in an exact-I32 region");
    assert!(
        matches!(
            err,
            KernelGenError::NotEqualTo {
                value: 1,
                expected: 0,
                ..
            }
        ),
        "{err:?}"
    );
}

/// POOT-1041: a two-byte float value is moved, never computed on. As a `Select` branch (and the region output)
/// it generates; read by an arithmetic step, or as the two branches of different types, it is the typed
/// `NarrowOperand` before any body exists. Mutation: drop the two-byte arm of `check_float_region_tys`; the
/// arithmetic row would build a body that adds raw two-byte lanes as F32 and this row goes red.
#[test]
fn a_two_byte_float_is_only_ever_a_select_branch() {
    use poot_kernelgen::{FusedInput, FusedKernel, FusedScalarOp, FusedStep, LeafTy};
    let layouts = vec![poot_kernelgen::Layout::contiguous(&[4]); 3];
    let shapes: [&[usize]; 3] = [&[4], &[4], &[4]];
    let select = |leaf_tys: Vec<LeafTy>| FusedKernel {
        leaf_tys,
        steps: vec![FusedStep {
            op: FusedScalarOp::Select,
            inputs: vec![
                FusedInput::Leaf(0),
                FusedInput::Leaf(1),
                FusedInput::Leaf(2),
            ],
        }],
        output: FusedInput::Step(0),
    };
    for leaf in [LeafTy::BF16, LeafTy::F16] {
        poot_kernelgen::fused_views(
            "t",
            &[4],
            &shapes,
            &layouts,
            &select(vec![LeafTy::I32, leaf, leaf]),
            None,
        )
        .expect("a select moves two-byte branches");
    }
    let mixed = select(vec![LeafTy::I32, LeafTy::BF16, LeafTy::F16]);
    let err = poot_kernelgen::fused_views("t", &[4], &shapes, &layouts, &mixed, None)
        .expect_err("branches of two types");
    assert!(
        matches!(err, KernelGenError::NarrowOperand { .. }),
        "{err:?}"
    );
    let float_and_narrow = select(vec![LeafTy::I32, LeafTy::BF16, LeafTy::F32]);
    let err = poot_kernelgen::fused_views("t", &[4], &shapes, &layouts, &float_and_narrow, None)
        .expect_err("a two-byte and an F32 branch");
    assert!(
        matches!(err, KernelGenError::NarrowOperand { .. }),
        "{err:?}"
    );
    let arithmetic = FusedKernel {
        leaf_tys: vec![LeafTy::F32, LeafTy::BF16],
        steps: vec![FusedStep {
            op: FusedScalarOp::Binary(poot_kernel_ir::BinOp::Add),
            inputs: vec![FusedInput::Leaf(0), FusedInput::Leaf(1)],
        }],
        output: FusedInput::Step(0),
    };
    let err =
        poot_kernelgen::fused_views("t", &[4], &shapes[..2], &layouts[..2], &arithmetic, None)
            .expect_err("a two-byte leaf feeds an Add");
    assert!(
        matches!(err, KernelGenError::NarrowOperand { .. }),
        "{err:?}"
    );
}
