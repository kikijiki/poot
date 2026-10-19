//! POOT-1221 (R2-L04-017): the buffer plan against an independent liveness scan. `plan_buffers` colors
//! disjoint lifetimes onto shared arena slots; if it ever lets two values that are alive at once share a
//! slot, a kernel overwrites an operand another kernel still reads. This file recomputes every value's
//! lifetime from the compiled program's public equations and plans (not from `plan_buffers`'s own
//! tables), then requires that no two occupants of one slot are alive at the same time. Rows run on
//! the three GPU targets (wgpu, ROCm, PTX) over seeded random graphs and a decode chain with scratch.
//! No device. Model-free.
//!
//! The checker is itself a row: a plan whose slot assignment is forged so two live values share a slot
//! must be reported, with both values named.
//!
//! Mutations (applied by hand, shown red, restored; recorded in the POOT-1221 report):
//! BL-1 the checker's liveness compare drops its overlap test (the forged-slot row goes red);
//! BL-2 `plan_buffers` frees a slot when `d <= b` instead of `d < b` (the random-graph rows go red);
//! BL-3 `plan_buffers` ends a scratch's lifetime one equation early (the decode-chain row goes red).

use std::collections::HashMap;

use poot_graph_ir::op::{BinOp, RedOp, UnOp};
use poot_graph_ir::{Builder, Graph, Slot, TensorType, ValueId};
use poot_graph_plan::{
    CompileOptions, Plan, Program, ScratchId, Target, compile, production_plan::Submission,
    value_ids,
};
use poot_target::{AmdArch, Backend};
use poot_test_util::device_caps::default_caps_for;

const OPTIONS: CompileOptions =
    CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD);

fn targets() -> Vec<(&'static str, Target)> {
    [
        ("wgpu", Backend::SpirvVulkan),
        ("rocm", Backend::AmdGcn(AmdArch::gfx1151())),
        ("ptx", Backend::Nvptx),
    ]
    .into_iter()
    .map(|(name, backend)| {
        (
            name,
            Target::Gpu {
                backend,
                caps: default_caps_for(backend),
            },
        )
    })
    .collect()
}

/// A deterministic xorshift stream (the repository's tests carry no rng dependency).
fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed
        .wrapping_mul(0x100000001B3)
        .wrapping_add(0x9E3779B97F4A7C15)
        | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// What occupies a slot: a value or a scratch, with the equation span it must stay valid for.
#[derive(Clone, Debug)]
struct Occupant {
    name: String,
    slot: usize,
    birth: usize,
    death: usize,
}

/// Every occupant of `program`'s arena with its lifetime, recomputed from the program's equations and
/// plans. `slot_of` names the slot a value or scratch was assigned, so a row can hand the scan a forged
/// assignment; production rows pass the plan's own.
///
/// A value is born at the equation that dispatches it and dies at the last equation that reads it or
/// anything that aliases it (`Alias`/`View` share their source's buffer). A value the program hands out
/// (the result, an export, a state-commit copy source or a validation witness) lives to the end. A
/// scratch lives for exactly its equation. Two occupants are alive together when their closed spans
/// intersect: an equation's operands and its output are alive at once.
fn occupants(program: &Program, slot_of: &dyn Fn(Key) -> Option<usize>) -> Vec<Occupant> {
    let equations = program.graph().eqns().len();
    let mut root: HashMap<ValueId, ValueId> = HashMap::new();
    let resolve = |root: &HashMap<ValueId, ValueId>, v: ValueId| *root.get(&v).unwrap_or(&v);
    let mut born: Vec<(usize, ValueId)> = Vec::new();
    let mut scratch: Vec<(usize, ScratchId)> = Vec::new();
    for (index, (eqn, plan)) in program.planned().enumerate() {
        match plan {
            Plan::Alias(source) => {
                let source = resolve(&root, *source);
                root.insert(eqn.out, source);
            }
            Plan::View { src, .. } => {
                let source = resolve(&root, *src);
                root.insert(eqn.out, source);
            }
            Plan::Dispatches(dispatches) => {
                if slot_of(Key::Value(eqn.out)).is_some() {
                    born.push((index, eqn.out));
                }
                scratch.extend(
                    (0..dispatches.scratch().len()).map(|id| (index, ScratchId::new(id as u32))),
                );
            }
            Plan::Graph => {}
        }
    }
    let mut death: HashMap<ValueId, usize> = born.iter().map(|&(i, v)| (v, i)).collect();
    for (index, eqn) in program.graph().eqns().iter().enumerate() {
        for operand in value_ids(eqn) {
            if let Some(d) = death.get_mut(&resolve(&root, operand)) {
                *d = (*d).max(index);
            }
        }
    }
    let handed_out = std::iter::once(program.graph().result())
        .chain(program.graph().exports().map(|export| export.value))
        .chain(
            program
                .state_commit()
                .copies
                .iter()
                .map(|copy| copy.state_output),
        )
        .chain(
            program
                .validation()
                .sources
                .iter()
                .map(|source| source.value),
        );
    for value in handed_out {
        if let Some(d) = death.get_mut(&resolve(&root, value)) {
            *d = equations;
        }
    }
    let mut all: Vec<Occupant> = born
        .iter()
        .map(|&(birth, value)| Occupant {
            name: format!("{value:?}"),
            slot: slot_of(Key::Value(value)).expect("born values have a slot"),
            birth,
            death: death[&value],
        })
        .collect();
    for (eqn, id) in scratch {
        let slot = slot_of(Key::Scratch(eqn, id))
            .unwrap_or_else(|| panic!("scratch {id:?} of equation {eqn} has no slot"));
        all.push(Occupant {
            name: format!("scratch {id:?} of equation {eqn}"),
            slot,
            birth: eqn,
            death: eqn,
        });
    }
    all
}

/// What a slot assignment is asked about.
#[derive(Clone, Copy)]
enum Key {
    Value(ValueId),
    Scratch(usize, ScratchId),
}

/// The plan's own assignment.
fn planned_slot(program: &Program) -> impl Fn(Key) -> Option<usize> + '_ {
    move |key| match key {
        Key::Value(value) => program.buffer_plan().slot(value).map(|slot| slot.index()),
        Key::Scratch(eqn, id) => program
            .buffer_plan()
            .scratch_slot(eqn, id)
            .map(|slot| slot.index()),
    }
}

/// Every pair of occupants that share a slot while both are alive, described.
fn overlaps(occupants: &[Occupant]) -> Vec<String> {
    let mut found = Vec::new();
    for (i, a) in occupants.iter().enumerate() {
        for b in &occupants[i + 1..] {
            if a.slot == b.slot && a.birth <= b.death && b.birth <= a.death {
                found.push(format!(
                    "slot {}: {} alive [{}, {}] and {} alive [{}, {}]",
                    a.slot, a.name, a.birth, a.death, b.name, b.birth, b.death
                ));
            }
        }
    }
    found
}

/// How many slots hold more than one occupant: the reuse the disjointness property is about.
fn shared_slots(occupants: &[Occupant]) -> usize {
    let mut count: HashMap<usize, usize> = HashMap::new();
    for occupant in occupants {
        *count.entry(occupant.slot).or_default() += 1;
    }
    count.values().filter(|&&n| n > 1).count()
}

fn compiled(graph: &Graph, target: &Target) -> Program {
    compile(graph, target, &OPTIONS).expect("the graph compiles")
}

/// A random graph over `[4, 8]` activations: unary and binary steps over any earlier value (so values
/// stay live across many equations), reshape and transpose pairs (aliases and views of a live buffer),
/// a keepdim reduce broadcast back, and a matmul, folded into one result that re-reads several earlier
/// values.
fn random_graph(seed: u64) -> Graph {
    let mut next = stream(seed ^ 0x0B0F_FE12);
    let b = Builder::for_tests();
    let shape = vec![4usize, 8];
    let mut pool = vec![
        b.slot_named(Slot::Activation, "x", TensorType::f32(shape.clone()))
            .unwrap(),
        b.constant("c", TensorType::f32(shape.clone())).unwrap(),
    ];
    let w = b.constant("w", TensorType::f32(vec![8, 8])).unwrap();
    for _ in 0..8 + next() % 16 {
        let pick = |next: &mut dyn FnMut() -> u64, pool: &[poot_graph_ir::Traced]| {
            pool[(next() % pool.len() as u64) as usize]
        };
        let a = pick(&mut next, &pool);
        let produced = match next() % 6 {
            0 => b
                .unary([UnOp::Neg, UnOp::Exp, UnOp::Tanh][(next() % 3) as usize], a)
                .unwrap(),
            1 => {
                let other = pick(&mut next, &pool);
                b.binary(
                    [BinOp::Add, BinOp::Sub, BinOp::Mul][(next() % 3) as usize],
                    a,
                    other,
                )
                .unwrap()
            }
            2 => {
                let flat = b.reshape(a, vec![8, 4]).unwrap();
                b.reshape(flat, shape.clone()).unwrap()
            }
            3 => {
                let t = b.transpose(a, vec![1, 0]).unwrap();
                b.transpose(t, vec![1, 0]).unwrap()
            }
            4 => {
                let sum = b.reduce(RedOp::Sum, a, 1, true).unwrap();
                b.broadcast(sum, shape.clone()).unwrap()
            }
            _ => b.matmul(a, w).unwrap(),
        };
        pool.push(produced);
    }
    let mut acc = *pool.last().unwrap();
    for _ in 0..3 {
        let earlier = pool[(next() % pool.len() as u64) as usize];
        acc = b.binary(BinOp::Add, acc, earlier).unwrap();
    }
    b.finish(acc).unwrap()
}

/// A chain of decode attentions: each layer's split-KV scratch is one more buffer in the same plan.
fn decode_chain(layers: usize) -> Graph {
    let (hq, cap, d) = (2usize, 1024usize, 64usize);
    let b = Builder::for_tests();
    let mut q = b
        .slot_named(Slot::Activation, "q", TensorType::f32(vec![1, hq, 1, d]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
        .unwrap();
    for layer in 0..layers {
        let kv = |name: &str| {
            b.slot_named(
                Slot::Activation,
                &format!("{name}{layer}"),
                TensorType::f32(vec![1, hq, cap, d]),
            )
            .unwrap()
        };
        let (k, v) = (kv("k"), kv("v"));
        q = poot_graph_ir::ops::attention_masked(&b, q, k, v, 1, 0.125, mask).unwrap();
    }
    b.finish(q).unwrap()
}

/// No two values alive at once share a slot, on every target, for 120 seeded graphs each.
#[test]
fn no_two_live_values_share_a_slot_in_random_graphs() {
    let mut shared = 0usize;
    let mut materialized = 0usize;
    for (name, target) in targets() {
        for seed in 0..120u64 {
            let program = compiled(&random_graph(seed), &target);
            assert!(
                program.regions().is_empty(),
                "{name}: a kernel target has no regions"
            );
            let all = occupants(&program, &planned_slot(&program));
            let found = overlaps(&all);
            assert!(
                found.is_empty(),
                "{name} seed {seed}: live values share a slot:\n{}",
                found.join("\n")
            );
            shared += shared_slots(&all);
            materialized += all.len();
        }
    }
    assert!(
        materialized > 1000,
        "only {materialized} occupants: the graphs are too small to test"
    );
    assert!(
        shared > 100,
        "only {shared} shared slots: the property would hold with no reuse at all"
    );
}

/// The decode chain's scratch is colored with the values, and no layer's scratch is alive with a
/// value that shares its slot.
#[test]
fn a_scratch_never_shares_a_slot_with_a_live_value() {
    for (name, target) in targets() {
        for layers in [1, 3, 6] {
            let program = compiled(&decode_chain(layers), &target);
            let all = occupants(&program, &planned_slot(&program));
            assert!(
                all.iter().any(|o| o.name.starts_with("scratch")),
                "{name}: the chain declares no scratch, the row tests nothing"
            );
            let found = overlaps(&all);
            assert!(
                found.is_empty(),
                "{name} x{layers}: live occupants share a slot:\n{}",
                found.join("\n")
            );
        }
    }
}

/// The checker finds an overlap: the plan's assignment with two values that are alive together forced
/// into one slot is reported, naming both. Without this row the two rows above would pass with a scan
/// that never looked.
#[test]
fn the_scan_reports_two_live_values_forced_into_one_slot() {
    let (_, target) = targets().remove(0);
    let program = compiled(&random_graph(3), &target);
    let honest = occupants(&program, &planned_slot(&program));
    assert!(
        overlaps(&honest).is_empty(),
        "the unforced plan is disjoint"
    );

    let (a, b) = honest
        .iter()
        .enumerate()
        .flat_map(|(i, a)| honest[i + 1..].iter().map(move |b| (a, b)))
        .find(|(a, b)| a.slot != b.slot && a.birth <= b.death && b.birth <= a.death)
        .expect("a graph with two values alive together");
    let (a_name, b_name, target_slot) = (a.name.clone(), b.name.clone(), a.slot);

    let forged = |key: Key| {
        let name = match key {
            Key::Value(value) => format!("{value:?}"),
            Key::Scratch(eqn, id) => format!("scratch {id:?} of equation {eqn}"),
        };
        if name == b_name {
            Some(target_slot)
        } else {
            planned_slot(&program)(key)
        }
    };
    let found = overlaps(&occupants(&program, &forged));
    assert!(
        found
            .iter()
            .any(|line| line.contains(&a_name) && line.contains(&b_name)),
        "forcing {a_name} and {b_name} into slot {target_slot} must be reported, got {found:?}"
    );
}
