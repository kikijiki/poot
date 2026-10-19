//! POOT-760: the graph container is encapsulated. A `Graph` is only changed through `GraphEdit`, and every
//! commit re-checks the whole graph and what it must keep (the result's type, every witness, every export,
//! every carried-state pair). These rows drive the real entry points: the edit API on a traced model graph,
//! and the production passes over a graph that carries all four kinds of output.

use std::collections::HashSet;

use poot_graph_ir::op::BinOp;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{
    BoundaryId, Builder, EditError, Graph, GraphValidationError, Operand, Outputs, StateRole,
    ValidationId, ValueId, WitnessLedger,
};
use poot_graph_plan::{cse, dce, fuse};
use poot_models::model::{LogitRows, StepSpec};
use poot_models::test_support::dense::{Dense, Family, step};

/// A tiny qwen2 decode step: a real traced graph with carried KV state.
fn qwen2_decode() -> Graph {
    Dense::new(Family::Qwen2)
        .vocab(16)
        .dims(8, 16, 2)
        .heads(2, 1)
        .head_dim(4)
        .max_positions(8)
        .f32_model()
        .model
        .trace(
            &StepSpec::step(step(1, 1, 4, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .expect("the qwen2 decode traces")
}

/// Ids of the fixture's outputs: four equations compute the same value, and each of the four kinds of
/// output reads a different copy, so CSE has to retarget every kind.
struct Fixture {
    graph: Graph,
    primary: ValueId,
    witness_copy: ValueId,
    export_copy: ValueId,
    state_copy: ValueId,
    cache: ValueId,
}

fn fixture() -> Fixture {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32([4])).unwrap();
    let y = b.constant("y", TensorType::f32([4])).unwrap();
    let cache = b
        .state_input("cache", TensorType::f32([4]), StateRole::Recurrent)
        .unwrap();
    let primary = b.binary(BinOp::Add, x, y).unwrap();
    let witness_copy = b.binary(BinOp::Add, x, y).unwrap();
    let export_copy = b.binary(BinOp::Add, x, y).unwrap();
    let state_copy = b.binary(BinOp::Add, x, y).unwrap();
    let graph = b
        .finish(
            Outputs::new(primary)
                .with_state(&[(cache, state_copy)])
                .with_witness(ValidationId(7), "copy", witness_copy)
                .with_export(BoundaryId(export_copy.id()), export_copy),
        )
        .expect("the fixture is well formed");
    Fixture {
        graph,
        primary: primary.id(),
        witness_copy: witness_copy.id(),
        export_copy: export_copy.id(),
        state_copy: state_copy.id(),
        cache: cache.id(),
    }
}

/// SC-001: an invalid edit through the edit API is a typed error, not a graph. A dangling operand, and a
/// rewrite that changes a type, each name what is wrong. Mutation: skip the whole-graph check in
/// `GraphEdit::commit` (`self.draft.check()?`); the dangling operand and the retyped intermediate are
/// accepted and both rows go red.
#[test]
fn an_invalid_edit_is_a_typed_error() {
    let traced = qwen2_decode();

    // A dangling operand: the first equation now reads a value no one defines.
    let mut edit = traced.edit();
    let missing = ValueId::from_index(traced.values().len() + 5).unwrap();
    edit.eqns_mut()[0].inputs[0] = Operand::Value(missing);
    match edit.commit() {
        Err(EditError::Invalid(GraphValidationError::EquationOperandNotDefined {
            equation: 0,
            value,
        })) => assert_eq!(value, missing),
        other => panic!(
            "a dangling operand must be refused as such, got {:?}",
            other.map(|graph| graph.eqns().len())
        ),
    }

    // A shape-changing rewrite: an intermediate's declared type no longer matches what its operation
    // infers from its operands.
    let intermediate = traced.eqns()[0].out;
    let mut edit = traced.edit();
    let mut retyped = edit.values()[intermediate].aval.clone();
    retyped.shape.push(3);
    edit.values_mut()[intermediate].aval = retyped.clone();
    match edit.commit() {
        Err(EditError::Invalid(GraphValidationError::EquationOutputTypeMismatch {
            stored,
            ..
        })) => assert_eq!(stored, retyped),
        other => panic!(
            "a retyped intermediate must be refused as such, got {:?}",
            other.map(|graph| graph.eqns().len())
        ),
    }

    // The result itself keeps its type through a rewrite.
    let result = traced.result();
    let mut edit = traced.edit();
    let mut wider = edit.values()[result].aval.clone();
    wider.shape.push(2);
    edit.values_mut()[result].aval = wider;
    assert!(
        edit.commit().is_err(),
        "a rewrite may not change the type of the result"
    );

    // An untouched edit commits to the same graph.
    let same = traced.edit().commit().expect("an empty edit is valid");
    assert_eq!(same.eqns().len(), traced.eqns().len());
}

/// SC-003: result, witnesses, exports and carried state are kinds of one output list, so a rewrite that
/// moves a value retargets every kind through the one `remap_results`. CSE merges four copies of one
/// computation; each kind of output must land on the survivor, and the passes that follow keep every one.
/// Mutation: make `GraphEdit::remap_results` skip the `Witness` kind in one pass; the witness keeps naming
/// a removed duplicate, `commit` refuses the graph, and this row goes red.
#[test]
fn every_output_kind_is_retargeted_by_one_pass_and_kept_by_the_next() {
    let f = fixture();
    let merged = cse(&f.graph).expect("CSE keeps the witness and every other output");
    assert_eq!(merged.eqns().len(), 1, "four copies of one computation");
    assert_eq!(merged.result(), f.primary);
    let witness = merged.witnesses().next().expect("the witness is kept");
    assert_eq!((witness.id, witness.name), (ValidationId(7), "copy"));
    assert_eq!(witness.value, f.primary, "the witness follows its producer");
    let export = merged.exports().next().expect("the export is kept");
    assert_eq!(export.id, BoundaryId(f.export_copy));
    assert_eq!(export.value, f.primary, "the export follows its producer");
    assert_eq!(
        merged.state().collect::<Vec<_>>(),
        vec![(f.cache, f.primary)],
        "the carried state follows its producer"
    );
    assert_ne!(f.witness_copy, f.primary);
    assert_ne!(f.state_copy, f.primary);

    // DCE and fusion keep the same four outputs and still describe a defined graph.
    for graph in [dce(&merged).unwrap(), fuse(&merged).unwrap()] {
        assert!(graph.witnesses().eq(merged.witnesses()));
        assert!(graph.exports().eq(merged.exports()));
        assert_eq!(
            graph.state().collect::<Vec<_>>(),
            merged.state().collect::<Vec<_>>()
        );
        let produced: HashSet<ValueId> = graph
            .eqns()
            .iter()
            .map(|eqn| eqn.out)
            .chain(graph.inputs().iter().copied())
            .collect();
        assert!(
            graph.liveness_roots().all(|root| produced.contains(&root)),
            "every output names a defined value"
        );
    }
}

/// SC-004: a commit that loses a witness is `EditError::WitnessDropped`, naming it; a split accounts every
/// witness through a `WitnessLedger`, and the ledger fails closed when a piece shed one. Mutation: skip the
/// witness comparison in `GraphEdit::commit` (the loop over `interface.witnesses` in `Graph::conserves`);
/// the first assertion goes red. Mutation: make `WitnessLedger::finish` return `Ok` unconditionally; the
/// second goes red.
#[test]
fn a_commit_that_loses_a_witness_is_refused_and_the_ledger_accounts_every_witness() {
    let f = fixture();

    let mut edit = f.graph.edit();
    edit.retain_witnesses(|_| false);
    match edit.commit() {
        Err(EditError::WitnessDropped { id, name }) => {
            assert_eq!((id, name.as_str()), (ValidationId(7), "copy"));
        }
        other => panic!(
            "a rewrite that sheds a witness must be refused, got {:?}",
            other.map(|graph| graph.eqns().len())
        ),
    }

    // The same applies to the other outputs a rewrite must keep.
    let mut edit = f.graph.edit();
    edit.clear_exports();
    assert!(matches!(
        edit.commit(),
        Err(EditError::ExportDropped { id }) if id == BoundaryId(f.export_copy)
    ));
    let mut edit = f.graph.edit();
    edit.retain_state(|_, _| false);
    assert!(matches!(
        edit.commit(),
        Err(EditError::StateDropped { input }) if input == f.cache
    ));

    // A split: a piece that keeps the witness claims it and the ledger finishes; a piece that sheds it
    // is allowed to exist (a stage need not define every witness) but the ledger then fails closed.
    let mut ledger = WitnessLedger::of(&f.graph);
    f.graph
        .edit()
        .commit_piece(&mut ledger)
        .expect("a piece that keeps the witness");
    ledger.finish().expect("every witness was claimed");

    let mut ledger = WitnessLedger::of(&f.graph);
    let mut shedding = f.graph.edit();
    shedding.retain_witnesses(|_| false);
    shedding
        .commit_piece(&mut ledger)
        .expect("a stage may define no witness");
    match ledger.finish() {
        Err(EditError::WitnessDropped { id, .. }) => assert_eq!(id, ValidationId(7)),
        other => panic!("a split that sheds a witness must fail the ledger, got {other:?}"),
    }
}

/// POOT-1119: a rewrite that declares a witness of its own commits only through
/// `commit_adding_witnesses`, naming exactly the witnesses it added. A plain `commit` refuses the new
/// one, a listed id that is not new (the fixture's own witness) or absent is `WitnessNotAdded`, and every
/// pre-existing witness is still conserved.
///
/// Mutation: make `conserves` ignore `added` for the `WitnessAdded` comparison (always `Ok`); the plain
/// `commit` assertion goes red.
#[test]
fn a_rewrite_adds_a_witness_only_by_naming_it() {
    let f = fixture();
    let new = ValidationId(8);
    let with_new = || {
        let mut edit = f.graph.edit();
        edit.push_witness(new, "added", f.witness_copy);
        edit
    };

    assert!(matches!(
        with_new().commit(),
        Err(EditError::WitnessAdded { id, .. }) if id == new
    ));
    let graph = with_new()
        .commit_adding_witnesses(&[new])
        .expect("a named added witness commits");
    assert_eq!(graph.witnesses().count(), 2);

    // The fixture's own witness is not new; an id the draft does not carry is not a witness at all.
    for listed in [ValidationId(7), ValidationId(99)] {
        assert!(matches!(
            with_new().commit_adding_witnesses(&[new, listed]),
            Err(EditError::WitnessNotAdded { id }) if id == listed
        ));
    }

    // Naming an added witness does not excuse shedding an existing one.
    let mut edit = with_new();
    edit.retain_witnesses(|witness| witness.id == new);
    assert!(matches!(
        edit.commit_adding_witnesses(&[new]),
        Err(EditError::WitnessDropped { id, .. }) if id == ValidationId(7)
    ));
}
