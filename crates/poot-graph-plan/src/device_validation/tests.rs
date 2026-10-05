use poot_graph_ir::op::{BinOp, OpKind, RedOp, UnOp};
use poot_graph_ir::{
    Builder, Eqn, Graph, NoValidations, Operand, Scalar, Storage, TensorType, Traced, ValidationId,
    ValueMeta,
};
use poot_tensor::DType;

use super::*;
use crate::plan_validation_packet;

const MAX: f32 = f32::MAX;

/// `1 - Ge(x, -MAX) * Ge(-x, -MAX)`: one invalid flag per lane, 1 for NaN and both infinities.
fn nonfinite_flags(b: &Builder, x: Traced) -> Traced {
    let not_below = b.binary_scalar(BinOp::Ge, x, Scalar::F32(-MAX));
    let negated = b.unary(UnOp::Neg, x);
    let not_above = b.binary_scalar(BinOp::Ge, negated, Scalar::F32(-MAX));
    let finite = b.binary(BinOp::Mul, not_below, not_above);
    let negative = b.binary_scalar(BinOp::Mul, finite, Scalar::F32(-1.0));
    b.binary_scalar(BinOp::Add, negative, Scalar::F32(1.0))
}

fn rejection(result: Result<(), PlanError>) -> (ValidationId, ValueId, DeviceWitnessRejection) {
    match result {
        Err(PlanError::ValidationWitnessNotCanonical { id, value, reason }) => (id, value, reason),
        other => panic!("expected a noncanonical witness, got {other:?}"),
    }
}

#[test]
fn device_witness_admits_spec_recipes() {
    let b = Builder::new();
    let weight = b.constant("weight", TensorType::f32(vec![4, 3]));
    let activation = b.constant("activation", TensorType::f32(vec![2, 4]));
    let logits = b.matmul(activation, weight);

    // Nonfinite router logits, reduced to one lane.
    let flags = nonfinite_flags(&b, logits);
    let flat = b.reshape(flags, vec![6]);
    let nonfinite = b.reduce(RedOp::Max, flat, 0, true);

    // A nonpositive selected-score sum: Ge(-sum, 0).
    let sum = b.reduce(RedOp::Sum, logits, 1, false);
    let negated = b.unary(UnOp::Neg, sum);
    let nonpositive = b.binary_scalar(BinOp::Ge, negated, Scalar::F32(0.0));

    // Exact I32 selector bounds for 3 experts, cast to F32 before the count is reduced.
    let raw = b.constant("raw", TensorType::new(vec![4], DType::I32));
    let at_least_zero = b.binary_scalar(BinOp::Ge, raw, Scalar::I32(0));
    let too_large = b.binary_scalar(BinOp::Ge, raw, Scalar::I32(3));
    let flipped = b.binary_scalar(BinOp::Mul, at_least_zero, Scalar::I32(-1));
    let below_zero = b.binary_scalar(BinOp::Add, flipped, Scalar::I32(1));
    let either = b.select(at_least_zero, too_large, below_zero);
    let as_f32 = b.cast(either, DType::F32);
    let bounds = b.reduce(RedOp::Sum, as_f32, 0, true);

    let graph = crate::test_support::finish_with_validations(
        b,
        logits,
        &[
            (ValidationId(1), "nonfinite", nonfinite),
            (ValidationId(2), "nonpositive", nonpositive),
            (ValidationId(3), "bounds", bounds),
        ],
    )
    .unwrap();

    let packet = plan_validation_packet(&graph).unwrap();
    admit_device_witnesses(&graph).unwrap();
    assert!(!packet.sources.is_empty());
    assert_eq!(
        packet
            .sources
            .iter()
            .map(|source| source.value)
            .collect::<Vec<_>>(),
        vec![nonfinite.id, nonpositive.id, bounds.id]
    );
}

#[test]
fn device_witness_rejects_noncanonical_heads() {
    type Build = fn(&Builder, Traced, Traced) -> Traced;
    type Expect = fn(&DeviceWitnessRejection, ValueId, ValueId) -> bool;
    let cases: [(&str, Build, Expect); 7] = [
        (
            "exact float difference",
            |b, h, _| b.binary(BinOp::Sub, h, h),
            |reason, _, h| matches!(reason, DeviceWitnessRejection::UnsupportedOp { value, .. } if *value == h),
        ),
        (
            "graph input",
            |_, _, input| input,
            |reason, _, input| matches!(reason, DeviceWitnessRejection::GraphInput { value } if *value == input),
        ),
        (
            "fractional literal",
            |b, h, _| {
                let flag = b.binary_scalar(BinOp::Ge, h, Scalar::F32(0.0));
                b.binary_scalar(BinOp::Mul, flag, Scalar::F32(0.5))
            },
            |reason, witness, _| matches!(reason, DeviceWitnessRejection::Literal { value, .. } if *value == witness),
        ),
        (
            "product above the bound",
            |b, h, _| {
                let flag = b.binary_scalar(BinOp::Ge, h, Scalar::F32(0.0));
                let at_bound = b.binary_scalar(BinOp::Mul, flag, Scalar::F32(16_777_216.0));
                b.binary_scalar(BinOp::Mul, at_bound, Scalar::F32(2.0))
            },
            |reason, witness, _| matches!(reason, DeviceWitnessRejection::Bound { value, bound } if *value == witness && *bound == 1 << 25),
        ),
        (
            "sum over an extent above the bound",
            |b, _, _| {
                let wide = b.constant("wide", TensorType::f32(vec![(1 << 24) + 1]));
                let flag = b.binary_scalar(BinOp::Ge, wide, Scalar::F32(0.0));
                b.reduce(RedOp::Sum, flag, 0, true)
            },
            |reason, witness, _| matches!(reason, DeviceWitnessRejection::Bound { value, bound } if *value == witness && *bound == (1 << 24) + 1),
        ),
        (
            "narrowing cast",
            |b, h, _| {
                let flag = b.binary_scalar(BinOp::Ge, h, Scalar::F32(0.0));
                let narrow = b.cast(flag, DType::BF16);
                b.cast(narrow, DType::F32)
            },
            |reason, witness, _| matches!(reason, DeviceWitnessRejection::CastDtype { value, from: DType::BF16, to: DType::F32 } if *value == witness),
        ),
        (
            "division",
            |b, h, _| {
                let flag = b.binary_scalar(BinOp::Ge, h, Scalar::F32(0.0));
                b.binary(BinOp::Div, flag, flag)
            },
            |reason, witness, _| matches!(reason, DeviceWitnessRejection::UnsupportedOp { value, .. } if *value == witness),
        ),
    ];
    for (name, build, expect) in cases {
        let b = Builder::new();
        let weight = b.constant("weight", TensorType::f32(vec![1, 1]));
        let activation = b.constant("activation", TensorType::f32(vec![1, 1]));
        let h = b.matmul(activation, weight);
        let input = b.constant("input", TensorType::f32(vec![1]));
        let witness = build(&b, h, input);
        let graph = crate::test_support::finish_with_validations(
            b,
            h,
            &[(ValidationId(9), "witness", witness)],
        )
        .unwrap();
        let (id, value, reason) = rejection(admit_device_witnesses(&graph));
        assert_eq!((id, value), (ValidationId(9), witness.id), "{name}");
        let blame = if name == "graph input" {
            input.id
        } else {
            h.id
        };
        assert!(expect(&reason, witness.id, blame), "{name}: {reason:?}");
    }
}

/// An ordinary graph takes the empty-plan path: no walk, and no second `Graph::validate`. A graph
/// that `Graph::validate` would reject still admits (vacuously), which is only possible if the walk
/// never validates it.
#[test]
fn ordinary_graph_plan_is_empty_and_does_not_validate() {
    let malformed = Graph::default();
    assert!(
        malformed.validate().is_err(),
        "an empty graph has no output value, so validation must reject it"
    );
    admit_device_witnesses(&malformed).unwrap();
}

/// The count domain is checked per value, not only where a `Cast` names a dtype. The graph is
/// hand-built because no traced graph can reach a narrow intermediate: the packet forces an F32
/// validation value, and `Cast`, the only dtype-changing admitted producer, rejects a narrow source itself.
#[test]
fn device_witness_rejects_a_narrow_intermediate() {
    let narrow = |shape: Vec<usize>| {
        ValueMeta::new(TensorType::new(shape, DType::BF16), Storage::Device, None)
    };
    let graph: Graph<NoValidations> = Graph {
        values: vec![
            ValueMeta::new(
                TensorType::new(vec![4], DType::BF16),
                Storage::Const,
                Some("x".into()),
            ),
            narrow(vec![4]),
            narrow(vec![4]),
        ],
        inputs: vec![0],
        consts: vec![0],
        slots: Vec::new(),
        eqns: vec![
            Eqn {
                op: OpKind::Binary(BinOp::Ge),
                inputs: vec![Operand::Value(0), Operand::Lit(Scalar::F32(0.0))],
                out: 1,
                layer: None,
            },
            Eqn {
                op: OpKind::Binary(BinOp::Mul),
                inputs: vec![Operand::Value(1), Operand::Value(1)],
                out: 2,
                layer: None,
            },
        ],
        output: 2,
        validations: NoValidations,
        state: Vec::new(),
    };
    let mut walk = WitnessWalk::new(&graph);
    assert_eq!(
        walk.bound(2),
        Err(DeviceWitnessRejection::Dtype {
            value: 2,
            dtype: DType::BF16
        })
    );
}

#[test]
fn device_witness_reports_the_first_declaration() {
    let b = Builder::new();
    let first = b.constant("first", TensorType::f32(vec![1]));
    let second = b.constant("second", TensorType::f32(vec![1]));
    let graph = crate::test_support::finish_with_validations(
        b,
        first,
        &[
            (ValidationId(5), "second", second),
            (ValidationId(2), "first", first),
        ],
    )
    .unwrap();
    let (id, value, _) = rejection(admit_device_witnesses(&graph));
    assert_eq!((id, value), (ValidationId(5), second.id));
}

#[test]
fn device_witness_admits_the_selector_bounds_chain() {
    // Card 372c's exact chain, emitted by the builder helper rather than rebuilt here, so this row
    // tracks the shape the authorization admits.
    let b = Builder::new();
    let raw = b.constant("raw", TensorType::new(vec![4], DType::I32));
    let guard = b.guard_index_bounds(raw, 3).unwrap();
    let graph = crate::test_support::finish_with_validations(
        b,
        guard.guarded,
        &[(ValidationId(0), "bounds", guard.witness)],
    )
    .unwrap();

    let packet = plan_validation_packet(&graph).unwrap();
    admit_device_witnesses(&graph).unwrap();
    assert_eq!(packet.layout.lane_count, 1);
    assert_eq!(packet.layout.byte_len, 4);
    // The walk stops at the unsigned comparison, so the Gather that would produce the raw ids in a real
    // router is never classified. Bound: one flag per lane summed over 4 lanes.
    let mut walk = WitnessWalk::new(&graph);
    assert_eq!(walk.bound(guard.witness.id).unwrap(), 4);
}

#[test]
fn device_witness_rejects_the_chain_above_the_exact_bound() {
    // The same chain over more lanes than an f32 can count exactly. The reduce is what grows the bound,
    // so this is the row that fails if the axis extent stops multiplying it.
    let lanes = (1usize << 24) + 1;
    let b = Builder::new();
    let raw = b.constant("raw", TensorType::new(vec![lanes], DType::I32));
    let guard = b.guard_index_bounds(raw, 3).unwrap();
    let graph = crate::test_support::finish_with_validations(
        b,
        guard.guarded,
        &[(ValidationId(0), "bounds", guard.witness)],
    )
    .unwrap();

    let (id, value, reason) = rejection(admit_device_witnesses(&graph));
    assert_eq!(id, ValidationId(0));
    assert_eq!(value, guard.witness.id);
    assert_eq!(
        reason,
        DeviceWitnessRejection::Bound {
            value: guard.witness.id,
            bound: lanes as u64,
        }
    );
}

/// Direct rows for the checks the production planner cannot reach with today's kernels: a body states
/// something no generator or shipped template emits (a zero extent, a fragment with no declared subgroup,
/// a workgroup that is not a whole number of subgroups, an LDS array of an
/// unmeasurable type). They guard the day a generator or an imported asset does; the production-path rows
/// are in `tests/generated_resources.rs`.
mod resources {
    use poot_kernel_ir::WorkgroupLocalDecl;
    use poot_kernel_ir::{
        BasicBlock, Body, Local, Statement, Terminator, Ty, WmmaDtype, WmmaShape,
    };
    use poot_kernelgen::{FragmentUse, KernelRequirements};
    use poot_target::{DeviceCaps, Queried, SubgroupSupport};

    use crate::device_validation::{ResourceRefusal, validate_kernel_resources};

    fn body(workgroup: [u32; 3], statements: Vec<Statement>) -> Body {
        let mut body = Body::new(
            "k",
            0,
            vec![],
            vec![BasicBlock {
                statements,
                terminator: Terminator::Return,
            }],
        );
        body.workgroup_size = workgroup;
        body
    }

    fn fragment_statement() -> Statement {
        Statement::WmmaZero {
            dtype: WmmaDtype::F16,
            shape: WmmaShape::M16N16K16,
            dst: Local { index: 1 },
        }
    }

    fn caps() -> DeviceCaps {
        DeviceCaps::ptx_default()
    }

    #[test]
    fn a_zero_workgroup_extent_is_refused() {
        let refusal = validate_kernel_resources(
            &body([64, 0, 1], vec![]),
            KernelRequirements::default(),
            &caps(),
        );
        assert_eq!(
            refusal,
            Err(ResourceRefusal::WorkgroupExtentZero { axis: 1 })
        );
    }

    /// A body that uses a fragment but declares no subgroup lanes understates its needs: the measured
    /// body, not the declaration, decides. Mutation: drop the `FragmentWithoutSubgroup` return and the
    /// understated body is admitted (`Ok(())`).
    #[test]
    fn a_fragment_without_a_declared_subgroup_is_refused() {
        let refusal = validate_kernel_resources(
            &body([32, 1, 1], vec![fragment_statement()]),
            KernelRequirements::default(),
            &caps(),
        );
        assert_eq!(
            refusal,
            Err(ResourceRefusal::FragmentWithoutSubgroup {
                fragment: FragmentUse {
                    dtype: WmmaDtype::F16,
                    shape: WmmaShape::M16N16K16,
                },
            })
        );
    }

    #[test]
    fn a_workgroup_that_is_not_whole_subgroups_is_refused() {
        let requirements = KernelRequirements {
            subgroup_lanes: Some(32),
            serial_work: None,
        };
        let refusal = validate_kernel_resources(
            &body([48, 1, 1], vec![fragment_statement()]),
            requirements,
            &caps(),
        );
        assert_eq!(
            refusal,
            Err(ResourceRefusal::WorkgroupNotSubgroupMultiple {
                invocations: 48,
                lanes: 32,
            })
        );
    }

    #[test]
    fn a_workgroup_array_of_unmeasurable_type_is_refused_not_guessed() {
        let mut body = body([64, 1, 1], vec![]);
        body.workgroup_locals = vec![WorkgroupLocalDecl {
            elem_ty: Ty::Bool,
            len: 8,
        }];
        let refusal = validate_kernel_resources(&body, KernelRequirements::default(), &caps());
        assert!(
            matches!(refusal, Err(ResourceRefusal::UnmeasurableLds(_))),
            "{refusal:?}"
        );
    }

    /// A subgroup the device reports in a range admits a declared size inside it and nothing outside.
    #[test]
    fn a_declared_subgroup_must_lie_in_the_reported_range() {
        let mut caps = caps();
        caps.subgroup = Queried::Known(SubgroupSupport::Present {
            min_size: 16,
            max_size: 32,
        });
        let fits = KernelRequirements {
            subgroup_lanes: Some(32),
            serial_work: None,
        };
        let outside = KernelRequirements {
            subgroup_lanes: Some(64),
            serial_work: None,
        };
        let launch = body([64, 1, 1], vec![]);
        assert_eq!(validate_kernel_resources(&launch, fits, &caps), Ok(()));
        assert_eq!(
            validate_kernel_resources(&launch, outside, &caps),
            Err(ResourceRefusal::SubgroupSize {
                lanes: 64,
                min_size: 16,
                max_size: 32,
            })
        );
    }
}
