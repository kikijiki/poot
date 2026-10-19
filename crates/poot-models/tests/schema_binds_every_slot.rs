//! The input schema and the traced graphs agree for every registered family (POOT-1224, ADR-0117).
//!
//! The driver resolves every slot of a traced program through the model's schema when it prepares an
//! entry, and refuses a slot nothing binds. This row runs that resolution (`ModelSchema::binding_of`) on
//! the graphs of every registered fixture, so a family whose tracer declares a slot its schema lacks, or
//! whose schema promises a carried input its tracer does not declare, fails here, model-free, before any
//! device sees it.
use poot_graph_ir::{Graph, SlotKey};
use poot_models::io::{Cadence, InputId, ModelSchema, Per, ProgramId, ProgramSpec, SlotBinding};
use poot_models::model::{LogitRows, StepSpec};
use poot_models::registry::Registry;
use poot_models::test_support::dense::step;

#[path = "support/corpus.rs"]
mod corpus;

use corpus::{Fixtured, Shape, fixtured};

fn slot_keys(graph: &Graph) -> Vec<SlotKey> {
    graph
        .slots()
        .iter()
        .map(|&(value, _)| {
            graph
                .meta(value)
                .slot_key()
                .expect("a slot declared through the builder carries its key")
                .clone()
        })
        .collect()
}

/// The slots one input declares in a graph: its value slot and, by form, its presence or valid slot.
fn input_slots(schema: &ModelSchema, id: InputId) -> Vec<SlotKey> {
    let input = schema.input(id).expect("a program reads declared inputs");
    let mut keys = vec![input.slot().clone()];
    match input.form().per() {
        Per::Token {
            presence: Some(slot),
            ..
        } => keys.push(slot.clone()),
        Per::Row {
            valid: Some(slot), ..
        } => keys.push(slot.clone()),
        _ => {}
    }
    keys
}

fn traced(fixture: &Fixtured, program: ProgramId, shape: &Shape, carried: &[InputId]) -> Graph {
    let spec = carried.iter().fold(
        StepSpec::new(
            program,
            step(1, shape.tokens, shape.capacity, LogitRows::Last),
        ),
        |spec, &id| spec.carrying(id),
    );
    fixture
        .model()
        .trace(&spec, poot_graph_ir::TraceLimits::for_tests())
        .unwrap_or_else(|e| panic!("{} {}: {e}", fixture.label, shape.name))
}

fn check_bindings(label: &str, schema: &ModelSchema, program: &ProgramSpec, graph: &Graph) {
    for key in slot_keys(graph) {
        match schema.binding_of(&key) {
            None => panic!("{label}: slot {key} is bound by no step kind or schema input"),
            Some(SlotBinding::Step(_)) => {}
            Some(SlotBinding::Request(input, part)) => {
                assert!(
                    program.presence_of(input.id()).is_some(),
                    "{label}: slot {key} is {part:?} of input {}, which program {} does not read",
                    input.id(),
                    program.id()
                );
            }
        }
    }
}

/// Every slot of every registered family's traced programs is a step kind or an input its program
/// reads, and a carried input's variant declares its slots. Mutation: drop `step.with_embeds(..)` from
/// a family's tracer (qwen2's `trace_into`); its carried variant lacks the embeds slot and this row
/// names it.
#[test]
fn every_slot_of_every_registered_family_binds_through_its_schema() {
    let registry = Registry::builtin().unwrap();
    for fixture in fixtured(&registry) {
        let schema = fixture.model().schema();
        for shape in fixture.shapes() {
            let program = schema.program(shape.program).unwrap_or_else(|| {
                panic!(
                    "{}: the schema has no {} program",
                    fixture.label, shape.program
                )
            });
            let label = format!("{} {}", fixture.label, shape.name);
            check_bindings(
                &label,
                schema,
                program,
                &traced(&fixture, shape.program, shape, &[]),
            );
            // Each carriable input, carried: the variant declares exactly its slots, and still binds.
            for id in schema.ids_of(schema.carriable(program)) {
                let label = format!("{label} carrying {id}");
                let graph = traced(&fixture, shape.program, shape, &[id]);
                check_bindings(&label, schema, program, &graph);
                let keys = slot_keys(&graph);
                for slot in input_slots(schema, id) {
                    assert!(
                        keys.contains(&slot),
                        "{label}: the carried input's slot {slot} is not in the graph"
                    );
                }
            }
        }
    }
}

/// A `step` program's carriable input is absent from the variant that does not carry it, so a step
/// that binds nothing of it runs an entry with no slot for it. Mutation: declare the embedding slots
/// unconditionally in a family's tracer; this row names the family and the slot.
#[test]
fn a_step_variant_without_a_carried_input_has_no_slot_for_it() {
    let registry = Registry::builtin().unwrap();
    let mut checked = 0;
    for fixture in fixtured(&registry) {
        let schema = fixture.model().schema();
        let Some(program) = schema.step() else {
            continue;
        };
        for shape in fixture.shapes() {
            let graph = traced(&fixture, ProgramId::STEP, shape, &[]);
            let keys = slot_keys(&graph);
            for id in schema.ids_of(schema.carriable(program)) {
                let input = schema.input(id).unwrap();
                assert_ne!(input.form().cadence(), Cadence::Every);
                for slot in input_slots(schema, id) {
                    assert!(
                        !keys.contains(&slot),
                        "{} {}: the variant that carries nothing declares {slot}",
                        fixture.label,
                        shape.name
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked > 0,
        "no registered step program has a carriable input"
    );
}
