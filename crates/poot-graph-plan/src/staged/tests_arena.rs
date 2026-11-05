//! POOT-1433: the activation arena of a staged entry, on a real `compile_staged` pipeline. Stages run one
//! after another, so the values dead between them share buffers, and a value that outlives its stage (the
//! boundary a later stage binds) keeps a buffer of its own for the whole entry. The rows read the arena the
//! engine allocates from the stage programs, never a hand-built plan.
use std::collections::HashMap;

use poot_graph_ir::{BinOp, Builder, Graph, Slot, TensorType, UnOp};

use super::tests::{options, pipeline, staged};
use crate::EntryArena;

/// Three layers, each widening the `[64, 256]` residual to `[64, 1024]` and back: the wide temporaries of one
/// layer are dead once its stage has run, the residual crosses each cut.
fn widening_layers() -> Graph {
    let (rows, width) = (64usize, 256usize);
    let b = Builder::for_tests();
    let mut h = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![rows, width]))
        .unwrap();
    for layer in 0..3 {
        let _layer = b.layer_scope(layer);
        let up = b
            .constant(
                &format!("up{layer}"),
                TensorType::f32(vec![width, 4 * width]),
            )
            .unwrap();
        let down = b
            .constant(
                &format!("down{layer}"),
                TensorType::f32(vec![4 * width, width]),
            )
            .unwrap();
        let wide = b.unary(UnOp::Tanh, b.matmul(h, up).unwrap()).unwrap();
        let narrow = b.matmul(wide, down).unwrap();
        h = b.binary(BinOp::Add, h, narrow).unwrap();
    }
    b.finish(h).unwrap()
}

/// Two stages each hold `[64, 1024]` temporaries, but never at once: the entry holds one set of them.
/// Mutation: give every stage buffers of its own (plan each stage into a fresh arena and concatenate);
/// the entry arena equals the sum of the stage arenas and this goes red.
#[test]
fn stages_share_the_buffers_of_dead_values() {
    let program = staged(&widening_layers(), pipeline(&[0, 1]), &options());
    let plans: Vec<_> = program.stages().map(|(_, _, p)| p.buffer_plan()).collect();
    assert_eq!(plans.len(), 3);
    let arena = EntryArena::plan(plans.iter().copied());
    let separate: usize = plans.iter().map(|plan| plan.arena_bytes()).sum();
    let largest = plans.iter().map(|plan| plan.arena_bytes()).max().unwrap();
    assert!(
        arena.bytes() < separate,
        "the entry arena is {} bytes, as much as three arenas of their own ({separate})",
        arena.bytes()
    );
    assert!(
        arena.bytes() >= plans.iter().map(|p| p.live_peak_bytes()).max().unwrap(),
        "the entry arena cannot be under one stage's live peak"
    );
    assert!(
        arena.bytes() < largest * 2,
        "the stages' dead values share, the entry arena stays near one stage's ({largest}): {}",
        arena.bytes()
    );
    // Every slot lands in a buffer of its class large enough for it, and a stage's slots never share one.
    for (stage, plan) in plans.iter().enumerate() {
        let mut seen = HashMap::new();
        for (slot, storage, elems) in plan.slots() {
            let buffer = arena.buffer_of(stage, slot);
            let (buffer_storage, buffer_elems) = arena.buffers()[buffer];
            assert_eq!(buffer_storage, storage, "stage {stage} slot {slot:?}");
            assert!(buffer_elems >= elems, "stage {stage} slot {slot:?}");
            assert_eq!(
                seen.insert(buffer, slot),
                None,
                "stage {stage}: two slots live at once share buffer {buffer}"
            );
        }
    }
}

/// The boundary stage 0 exports is read by stage 1, which also fills shared buffers: its buffer is its own
/// for the whole entry, exactly its size, and no slot of any other stage maps to it.
/// Mutation: treat every slot as dead after its stage (drop the `slot_outlives_stage` branch in
/// `EntryArena::plan`); an export shares a buffer with a later stage's slot and the row names both.
#[test]
fn an_export_keeps_a_buffer_of_its_own_to_the_end_of_the_entry() {
    let program = staged(&widening_layers(), pipeline(&[0, 1]), &options());
    let plans: Vec<_> = program.stages().map(|(_, _, p)| p.buffer_plan()).collect();
    let arena = EntryArena::plan(plans.iter().copied());
    let mut owners: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (stage, plan) in plans.iter().enumerate() {
        for (slot, _, _) in plan.slots() {
            owners
                .entry(arena.buffer_of(stage, slot))
                .or_default()
                .push((stage, slot.index()));
        }
    }
    let mut exports = 0;
    for (stage, (_, _, stage_program)) in program.stages().enumerate() {
        for export in stage_program.exports() {
            let plan = stage_program.buffer_plan();
            let slot = plan.slot(export.value).expect("an export is materialized");
            exports += 1;
            assert!(
                plan.slot_outlives_stage(slot),
                "stage {stage}: the export {:?} sits in a slot that dies with its stage",
                export.id
            );
            let buffer = arena.buffer_of(stage, slot);
            assert_eq!(
                owners[&buffer],
                [(stage, slot.index())],
                "stage {stage}: the export's buffer {buffer} is shared"
            );
            assert_eq!(arena.buffers()[buffer].1, plan.slot_elems(slot));
            // POOT-1450: a whole-buffer transfer carries no storage, so an export's buffer is typed with its own.
            assert_eq!(
                arena.buffers()[buffer].0,
                stage_program
                    .storage()
                    .storage(export.value)
                    .buffer_storage(),
                "stage {stage}: the export's buffer is typed"
            );
        }
    }
    assert!(exports >= 2, "two cuts export the residual: {exports}");
}
