//! POOT-733 SC-005: the tracer's construction budget. Every limit admits its exact boundary and
//! refuses the first request past it before anything is allocated or appended, and the refusal
//! carries the resource, the limit and the attempted total as typed data.

use std::num::{NonZeroU64, NonZeroUsize};

use crate::budget::{
    AllocationSite, Attempted, BudgetResource, Ledger, LedgerEvent, TraceBudgetExceeded,
};
use crate::error::{BuilderCollection, BuilderError};
use crate::graph::{Operand, Slot, StateRole, Storage, ValidationId};
use crate::op::{BinOp, OpKind, UnOp};
use crate::types::{DType, Scalar, TensorType};
use crate::{Builder, TraceLimits, Traced};

fn limits(values: usize, eqns: usize, literal: u64, owned: u64) -> TraceLimits {
    TraceLimits {
        max_tokens: NonZeroUsize::new(4).unwrap(),
        max_values: NonZeroUsize::new(values).unwrap(),
        max_eqns: NonZeroUsize::new(eqns).unwrap(),
        max_literal_bytes: NonZeroU64::new(literal).unwrap(),
        max_owned_bytes: NonZeroU64::new(owned).unwrap(),
    }
}

/// The limits of the SC-005 fixture with one field narrowed at a time: `max_tokens=4`, `max_values=8`,
/// `max_eqns=6`, `max_literal_bytes=32`; the owned-byte ceiling is out of the way.
fn sc005(values: usize, eqns: usize, literal: u64) -> TraceLimits {
    limits(values, eqns, literal, 1 << 30)
}

fn budget(error: BuilderError) -> TraceBudgetExceeded {
    match error {
        BuilderError::Budget(budget) => budget,
        other => panic!("expected a budget refusal, got {other:?}"),
    }
}

fn vector() -> TensorType {
    TensorType::f32(vec![4])
}

/// What a refused request left behind: nothing in the graph, the generation and the ledger as they
/// were.
fn assert_untouched(b: &Builder, generation: u64, charged: crate::budget::Charged) {
    assert_eq!(
        b.generation(),
        generation,
        "a refused request bumped the generation"
    );
    assert_eq!(b.charged(), charged, "a refused request changed the ledger");
}

#[test]
fn values_admit_the_eighth_and_refuse_the_ninth() {
    let b = Builder::new(sc005(8, 6, 32));
    for index in 0..8 {
        b.constant(&format!("c{index}"), vector()).unwrap();
    }
    let (generation, charged) = (b.generation(), b.charged());
    let refused = budget(b.constant("c8", vector()).unwrap_err());
    assert_eq!(
        (
            refused.resource,
            refused.limit,
            refused.used,
            refused.requested
        ),
        (BudgetResource::Values, 8, 8, 1)
    );
    assert_eq!(refused.attempted, Attempted::Total(9));
    assert_untouched(&b, generation, charged);
}

#[test]
fn equations_admit_the_sixth_and_refuse_the_seventh() {
    let b = Builder::new(sc005(64, 6, 32));
    let mut x = b.constant("x", vector()).unwrap();
    for _ in 0..6 {
        x = b.unary(UnOp::Neg, x).unwrap();
    }
    let (generation, charged) = (b.generation(), b.charged());
    let refused = budget(b.unary(UnOp::Neg, x).unwrap_err());
    assert_eq!(
        (
            refused.resource,
            refused.limit,
            refused.used,
            refused.requested
        ),
        (BudgetResource::Equations, 6, 6, 1)
    );
    assert_eq!(refused.attempted, Attempted::Total(7));
    assert_untouched(&b, generation, charged);
    let graph = b.finish(x).unwrap();
    assert_eq!(graph.eqns().len(), 6, "no seventh equation was appended");
}

#[test]
fn literal_bytes_admit_the_eighth_scalar_and_refuse_the_byte_36() {
    let b = Builder::new(sc005(64, 64, 32));
    let mut x = b.constant("x", vector()).unwrap();
    for _ in 0..8 {
        x = b.binary_scalar(BinOp::Add, x, Scalar::F32(1.0)).unwrap();
    }
    assert_eq!(b.charged().literal_bytes, 32);
    let (generation, charged) = (b.generation(), b.charged());
    let refused = budget(
        b.binary_scalar(BinOp::Add, x, Scalar::F32(1.0))
            .unwrap_err(),
    );
    assert_eq!(
        (
            refused.resource,
            refused.limit,
            refused.used,
            refused.requested
        ),
        (BudgetResource::LiteralBytes, 32, 32, 4)
    );
    assert_eq!(refused.attempted, Attempted::Total(36));
    assert_untouched(&b, generation, charged);
}

#[test]
fn a_refused_equation_appends_nothing_and_a_later_request_still_fits() {
    let b = Builder::new(sc005(64, 64, 4));
    let x = b.constant("x", vector()).unwrap();
    let first = b.binary_scalar(BinOp::Add, x, Scalar::F32(1.0)).unwrap();
    let (generation, charged) = (b.generation(), b.charged());
    budget(
        b.binary_scalar(BinOp::Add, first, Scalar::F32(2.0))
            .unwrap_err(),
    );
    assert_untouched(&b, generation, charged);
    // A request that needs no literal still fits the same builder.
    b.unary(UnOp::Neg, first).unwrap();
}

/// A scenario that ends in controlled metadata allocations: `setup` builds the pre-existing state under
/// the given limits, `act` performs the operation (consuming the builder, so a `finish` can be one) and
/// reports the ledger it charged and its result.
struct Scenario {
    setup: fn(TraceLimits) -> Builder,
    act: fn(Builder) -> (Ledger, Result<(), BuilderError>),
    /// The allocation sites the operation must have reserved at, in the order they are first reserved.
    covers: &'static [AllocationSite],
}

/// One reservation of an operation with the owned bytes the ledger held just before it.
struct Step {
    site: AllocationSite,
    requested: u64,
    before: u64,
}

impl Scenario {
    fn run(&self, owned_ceiling: u64) -> (Vec<LedgerEvent>, usize, Result<(), BuilderError>) {
        let b = (self.setup)(limits(1 << 12, 1 << 12, 1 << 20, 1 << 30));
        // The pre-existing state is built under generous limits (its own transient scratch may peak
        // above what it keeps); only the controlled operation runs under the boundary.
        b.set_limits_for_tests(limits(1 << 12, 1 << 12, 1 << 20, owned_ceiling));
        let events_before = b.ledger().events().len();
        let (ledger, result) = (self.act)(b);
        (ledger.events(), events_before, result)
    }

    /// The reservations the operation makes under generous limits, each with the charge held before it.
    fn steps(&self) -> Vec<Step> {
        let b = (self.setup)(limits(1 << 12, 1 << 12, 1 << 20, 1 << 30));
        let mut running = b.charged().owned_bytes;
        let events_before = b.ledger().events().len();
        let (ledger, result) = (self.act)(b);
        result.expect("the operation fits generous limits");
        let mut steps = Vec::new();
        for event in &ledger.events()[events_before..] {
            match event {
                LedgerEvent::Reserved(reservation) => {
                    steps.push(Step {
                        site: reservation.site,
                        requested: reservation.owned_bytes,
                        before: running,
                    });
                    running += reservation.owned_bytes;
                }
                LedgerEvent::Released(charge) => running -= charge.owned_bytes,
            }
        }
        steps
    }

    /// SC-005's owned-byte row, for every reservation of the operation: record the exact pre-existing
    /// charge and the checked requested size `B`; `B` remaining bytes grant the request and `B - 1`
    /// refuse it with the allocation site named, before the allocation (the ledger holds no entry for
    /// it and nothing after it was reserved).
    fn assert_exact_boundaries(&self) {
        let steps = self.steps();
        for covered in self.covers {
            assert!(
                steps
                    .iter()
                    .any(|step| step.site == *covered && step.requested > 0),
                "the operation made no reservation at {covered:?}: {:?}",
                steps.iter().map(|s| s.site).collect::<Vec<_>>()
            );
        }
        for (index, step) in steps.iter().enumerate().filter(|(_, s)| s.requested > 0) {
            let reserved_before = steps[..index].len();
            let exact = step.before + step.requested;

            let (events, from, result) = self.run(exact - 1);
            let refused = budget(result.expect_err("B - 1 remaining bytes must refuse"));
            assert_eq!(
                refused.resource,
                BudgetResource::OwnedBytes,
                "{:?}",
                step.site
            );
            assert_eq!(refused.site, step.site, "step {index}");
            assert_eq!(refused.limit, exact - 1);
            assert_eq!(
                refused.used, step.before,
                "{:?}: the exact pre-existing charge",
                step.site
            );
            assert_eq!(
                refused.requested, step.requested,
                "{:?}: the requested size",
                step.site
            );
            assert_eq!(refused.attempted, Attempted::Total(exact));
            let granted = events[from..]
                .iter()
                .filter(|event| matches!(event, LedgerEvent::Reserved(_)))
                .count();
            assert_eq!(
                granted, reserved_before,
                "{:?}: something was reserved after the refused request, or the refused one was",
                step.site
            );

            let (events, from, result) = self.run(exact);
            let granted = events[from..]
                .iter()
                .filter(|event| matches!(event, LedgerEvent::Reserved(_)))
                .count();
            assert!(
                granted > index,
                "{:?}: B remaining bytes must grant the request ({granted} granted)",
                step.site
            );
            // Whatever refuses next, if anything does, is a later request: `granted > index`.
            let _ = result;
        }
    }
}

fn with_three_values(limits: TraceLimits) -> Builder {
    let b = Builder::new(limits);
    let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![3, 4])).unwrap();
    b.matmul(x, w).unwrap();
    b
}

fn through_a_plan(
    b: Builder,
    stage: impl FnOnce(&Builder) -> Result<(), BuilderError>,
) -> (Ledger, Result<(), BuilderError>) {
    let ledger = b.ledger();
    let result = stage(&b);
    (ledger, result)
}

#[test]
fn the_plan_snapshot_is_charged_before_it_is_cloned() {
    Scenario {
        setup: with_three_values,
        act: |b| through_a_plan(b, |b| b.append_plan(0).map(drop)),
        covers: &[AllocationSite::PlanSnapshot],
    }
    .assert_exact_boundaries();
}

#[test]
fn a_staged_name_and_value_are_charged_before_they_are_taken() {
    Scenario {
        setup: with_three_values,
        act: |b| {
            through_a_plan(b, |b| {
                let mut plan = b.append_plan(0)?;
                plan.input(
                    "a_staged_input_with_a_long_enough_name",
                    vector(),
                    Storage::Const,
                )
                .map(drop)
            })
        },
        covers: &[
            AllocationSite::PlanSnapshot,
            AllocationSite::StagedValue,
            AllocationSite::StagedName,
        ],
    }
    .assert_exact_boundaries();
}

#[test]
fn a_staged_equation_and_its_inference_metadata_are_charged_before_they_are_built() {
    Scenario {
        setup: with_three_values,
        act: |b| {
            through_a_plan(b, |b| {
                let mut plan = b.append_plan(0)?;
                let x = plan.input("x2", TensorType::f32(vec![2, 3]), Storage::Const)?;
                plan.equation_named(
                    OpKind::Unary(UnOp::Neg),
                    vec![Operand::Value(x.id())],
                    Some("negated_value".to_string()),
                )
                .map(drop)
            })
        },
        covers: &[
            AllocationSite::StagedEquation,
            AllocationSite::StagedName,
            AllocationSite::StagedInference,
        ],
    }
    .assert_exact_boundaries();
}

#[test]
fn a_preflight_charges_its_tables_before_it_builds_them() {
    Scenario {
        setup: with_three_values,
        act: |b| {
            through_a_plan(b, |b| {
                let mut plan = b.append_plan(0)?;
                let x = plan.input("x2", TensorType::f32(vec![2, 3]), Storage::Const)?;
                let y = plan.equation(OpKind::Unary(UnOp::Neg), vec![Operand::Value(x.id())])?;
                plan.declare_result(y)?;
                b.preflight_append(plan).map(drop)
            })
        },
        covers: &[
            AllocationSite::PreflightTables,
            AllocationSite::StagedInference,
        ],
    }
    .assert_exact_boundaries();
}

#[test]
fn an_equation_record_and_its_operand_types_are_charged_before_they_are_cloned() {
    Scenario {
        setup: with_three_values,
        act: |b| {
            through_a_plan(b, |b| {
                let x = b.constant("y", TensorType::f32(vec![2, 3]))?;
                b.unary(UnOp::Exp, x).map(drop)
            })
        },
        covers: &[
            AllocationSite::InputRecord,
            AllocationSite::EquationRecord,
            AllocationSite::InferenceOperands,
            AllocationSite::InferenceResult,
        ],
    }
    .assert_exact_boundaries();
}

#[test]
fn carried_state_witnesses_and_exports_are_charged_before_finish_builds_them() {
    Scenario {
        setup: |limits| {
            let b = Builder::new(limits);
            let state = b.state_input("s", vector(), StateRole::Recurrent).unwrap();
            let next = b.unary(UnOp::Exp, state).unwrap();
            b.reduce(crate::op::RedOp::Sum, next, 0, false).unwrap();
            b
        },
        act: |b| {
            let ledger = b.ledger();
            let result = {
                // The graph's handles are rebuilt from the builder's own ids: value 0 is the state
                // input, 1 its exp, 2 their sum.
                let outputs = crate::Outputs::new(b.handle_for_tests(1))
                    .with_state(&[(b.handle_for_tests(0), b.handle_for_tests(1))])
                    .with_witness(ValidationId(1), "witness", b.handle_for_tests(2))
                    .with_export(
                        crate::BoundaryId(b.handle_for_tests(1).id()),
                        b.handle_for_tests(1),
                    );
                b.finish(outputs).map(drop)
            };
            (ledger, result)
        },
        covers: &[
            AllocationSite::FinalState,
            AllocationSite::FinalValidation,
            AllocationSite::FinalExports,
        ],
    }
    .assert_exact_boundaries();
}

#[test]
fn a_resumed_graph_is_charged_before_it_is_taken_over() {
    let b = Builder::for_tests();
    let x = b.constant("x", vector()).unwrap();
    let y = b.unary(UnOp::Neg, x).unwrap();
    let graph = b.finish(y).unwrap();
    // The graph holds two values and one equation: a builder that allows one value refuses it whole.
    let refused = Builder::resume(graph.clone(), limits(1, 8, 32, 1 << 30))
        .err()
        .unwrap();
    let refused = budget(refused);
    assert_eq!(
        (refused.resource, refused.site, refused.requested),
        (BudgetResource::Values, AllocationSite::ResumedGraph, 2)
    );
    let resumed = Builder::resume(graph, limits(2, 8, 32, 1 << 30)).unwrap();
    assert_eq!(resumed.builder.charged().values, 2);
    assert_eq!(resumed.builder.charged().equations, 1);
}

/// A plan that stages a constant, a slot and one equation, prepared and ready to commit.
fn prepared_plan(b: &Builder) -> (crate::builder::PreparedBuilderAppend, Traced) {
    let mut plan = b.append_plan(0).unwrap();
    let constant = plan
        .input("constant", TensorType::f32(vec![1]), Storage::Const)
        .unwrap();
    let slot = plan
        .input(
            "slot",
            TensorType::f32(vec![1]),
            Storage::Slot(Slot::Activation),
        )
        .unwrap();
    let output = plan
        .equation(
            OpKind::Binary(BinOp::Add),
            vec![Operand::Value(constant.id()), Operand::Value(slot.id())],
        )
        .unwrap();
    plan.declare_result(output).unwrap();
    (b.preflight_append(plan).unwrap(), output)
}

#[test]
fn a_reserve_failure_after_an_earlier_reserve_keeps_the_contents_and_the_charge() {
    let b = Builder::for_tests();
    b.constant("base", TensorType::f32(vec![1])).unwrap();
    let (mut prepared, output) = prepared_plan(&b);
    let (generation, charged) = (b.generation(), b.charged());
    let values_before = {
        // The graph the builder holds, read through a finished copy of a sibling builder is not
        // possible; the value count is the ledger's.
        charged.values
    };
    // The values, inputs and constants reservations succeed first; the equations reservation fails.
    b.fail_next_reservation(BuilderCollection::Equations);
    let refused = b.commit_append(&mut prepared).unwrap_err();
    assert!(matches!(
        refused,
        BuilderError::Reserve {
            collection: BuilderCollection::Equations,
            ..
        }
    ));
    assert_eq!(
        b.generation(),
        generation,
        "a failed commit bumped the generation"
    );
    assert_eq!(
        b.charged(),
        charged,
        "a failed commit gave back capacity an earlier reserve of it retained"
    );
    assert_eq!(b.charged().values, values_before);
    // The plan is still whole and commits once the allocator relents; the snapshot dies with it.
    assert_eq!(b.commit_append(&mut prepared).unwrap(), output);
    assert_eq!(b.generation(), generation + 1);
    let after = b.charged();
    assert_eq!(after.values, charged.values);
    assert!(
        after.owned_bytes < charged.owned_bytes,
        "the committed plan gives its type snapshot back ({} -> {})",
        charged.owned_bytes,
        after.owned_bytes
    );
}

#[test]
fn a_dropped_plan_gives_back_everything_it_charged() {
    let b = Builder::for_tests();
    b.constant("base", TensorType::f32(vec![1])).unwrap();
    let charged = b.charged();
    {
        let mut plan = b.append_plan(0).unwrap();
        plan.input("staged", TensorType::f32(vec![1]), Storage::Const)
            .unwrap();
        assert!(b.charged().owned_bytes > charged.owned_bytes);
        assert_eq!(b.charged().values, charged.values + 1);
    }
    assert_eq!(b.charged(), charged);
}

#[test]
fn a_foreign_dtype_error_releases_what_the_failed_equation_reserved() {
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![2])).unwrap();
    let c = b
        .constant("c", TensorType::new(vec![2], DType::I32))
        .unwrap();
    let charged = b.charged();
    assert!(matches!(
        b.binary(BinOp::Add, a, c),
        Err(BuilderError::Inference { .. })
    ));
    assert_eq!(b.charged(), charged);
}

/// The rank bound that sizes an equation's inference reservation is a checked sum: operands whose
/// ranks plus attribute words do not fit `usize`, or an attribute size that already overflowed `u64`,
/// are no bound at all. Mutation: `checked_add` -> `saturating_add` in `rank_bound`; the first row
/// then yields `Some(usize::MAX)`, a bound no limit would reject for the wrong reason.
#[test]
fn the_rank_bound_refuses_overflow_instead_of_saturating() {
    use crate::builder::rank_bound;

    assert_eq!(rank_bound(2, Some(16)), Some(2 + 2 + 1));
    assert_eq!(rank_bound(usize::MAX - 1, Some(16)), None);
    assert_eq!(rank_bound(usize::MAX, Some(0)), None);
    assert_eq!(
        rank_bound(0, None),
        None,
        "an overflowed attribute size is no bound"
    );
}
