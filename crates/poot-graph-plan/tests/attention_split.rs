//! Split-KV decode attention through the production planner (POOT-1113): the scratch contract, the buffer
//! plan that colors it, and the fall back to the streaming schedule when the split does not fit. No device:
//! the device rows (`poot-wgpu-device`, `poot-rocm-device`) run the same graphs through each executor.
//!
//! Every row drives `compile` over a traced decode graph; none builds a `DispatchPlan` by hand.

use poot_graph_ir::{AttentionAttrs, Builder, Graph, OpKind, Slot, TensorType};
use poot_graph_plan::{
    CompileError, CompileOptions, DispatchPlan, KernelChoice, Plan, PlanError, Program, ScratchId,
    Sink, Source, Target, compile, production_plan::Submission, value_ids,
};
use poot_kernelgen::{AttentionSchedule, AttentionSpec, KernelRequest};
use poot_target::{AmdArch, Backend, DeviceCaps};
use poot_test_util::device_caps::default_caps_for;

const OPTIONS: CompileOptions =
    CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD);

fn target(backend: Backend, caps: DeviceCaps) -> Target {
    Target::Gpu { backend, caps }
}

fn targets() -> Vec<(&'static str, Target)> {
    [
        ("wgpu", Backend::SpirvVulkan),
        ("rocm", Backend::AmdGcn(AmdArch::gfx1151())),
        ("ptx", Backend::Nvptx),
    ]
    .into_iter()
    .map(|(name, backend)| (name, target(backend, default_caps_for(backend))))
    .collect()
}

/// `layers` decode attentions in a chain (each layer's output is the next layer's query), over `hq` heads
/// of width `d` against a `cap`-key cache each. K and V are activation inputs, so they are not bound by
/// the constant size gate and a device whose buffer limit is below the cache still plans the graph.
fn decode_chain(layers: usize, hq: usize, cap: usize, d: usize) -> Graph {
    let b = Builder::for_tests();
    let mut q = b
        .slot_named(Slot::Activation, "q", TensorType::f32(vec![1, hq, 1, d]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
        .unwrap();
    for layer in 0..layers {
        let k = b
            .slot_named(
                Slot::Activation,
                &format!("k{layer}"),
                TensorType::f32(vec![1, hq, cap, d]),
            )
            .unwrap();
        let v = b
            .slot_named(
                Slot::Activation,
                &format!("v{layer}"),
                TensorType::f32(vec![1, hq, cap, d]),
            )
            .unwrap();
        q = poot_graph_ir::ops::attention_masked(&b, q, k, v, 1, 0.125, mask).unwrap();
    }
    b.finish(q).unwrap()
}

fn compiled(graph: &Graph, target: &Target) -> Program {
    compile(graph, target, &OPTIONS).expect("the decode chain compiles")
}

/// Every attention equation of `program`: its index among the equations, its plan, its request.
fn attentions(
    program: &Program,
) -> Vec<(usize, &poot_graph_ir::Eqn, &DispatchPlan, &AttentionSpec)> {
    program
        .planned()
        .enumerate()
        .filter(|(_, (eqn, _))| matches!(eqn.op, OpKind::Attention(_)))
        .map(|(index, (eqn, plan))| {
            let Plan::Dispatches(dispatches) = plan else {
                panic!("an attention launches: {plan:?}");
            };
            let KernelChoice::Generated(KernelRequest::Attention(spec)) =
                program.kernel_choice(eqn)
            else {
                panic!(
                    "generated attention expected: {:?}",
                    program.kernel_choice(eqn)
                );
            };
            (index, eqn, dispatches, spec)
        })
        .collect()
}

/// A low-head decode over a long cache plans stage one into a scratch and the combine out of it: two
/// launches, one declared scratch of `rows * splits * (dv + 2)` f32 elements, the first dispatch writing it
/// and the second reading it and writing the output. The chain's kernel request is the split schedule.
#[test]
fn a_low_head_decode_plans_two_launches_through_one_scratch() {
    let graph = decode_chain(1, 2, 1024, 64);
    for (name, target) in targets() {
        let program = compiled(&graph, &target);
        let [(_, _, plan, spec)] = attentions(&program)[..] else {
            panic!("{name}: one attention");
        };
        let AttentionSchedule::SplitKv { splits, .. } = spec.schedule else {
            panic!("{name}: a 2-row decode on 40 CUs must split: {spec:?}");
        };
        assert!(splits >= 2, "{name}");
        assert_eq!(plan.len(), 2, "{name}: partials then combine");
        let [decl] = plan.scratch() else {
            panic!("{name}: one scratch: {:?}", plan.scratch());
        };
        assert_eq!(decl.elems(), 2 * splits * (64 + 2), "{name}");
        assert_eq!(decl.storage(), poot_target::BufferStorage::f32(), "{name}");
        let id = ScratchId::new(0);
        assert_eq!(plan[0].binding().write(), Sink::Scratch(id), "{name}");
        assert_eq!(plan[1].binding().reads(), &[Source::Scratch(id)], "{name}");
        assert_eq!(plan[1].binding().write(), Sink::Out, "{name}");
        // The choice names the request the key was digested from: the split schedule's partials stage.
        assert!(
            matches!(
                program.kernel_choice(attentions(&program)[0].1),
                KernelChoice::Generated(_)
            ),
            "{name}"
        );
    }
}

/// The production decode shape: the key bound is the live length `Slot::Pos + 1`, a position slot whose
/// declared range is `0..cap` (every binder checks it), so the guarded composite fuses to the live
/// attention without a validation witness and plans the split schedule. The bank's live cases and the
/// serving decode graphs take this form.
#[test]
fn a_live_decode_over_a_declared_position_slot_plans_the_split_schedule() {
    use poot_graph_ir::IndexRange;
    use poot_graph_ir::ops::{AttentionKeys, attention_live, causal_mask_from_pos};

    let (hq, cap, d) = (2usize, 1024usize, 64usize);
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hq, cap, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hq, cap, d]))
        .unwrap();
    let pos = b
        .slot_ranged(
            Slot::Pos,
            TensorType::new(vec![1, 1], poot_tensor::DType::I32),
            IndexRange::below(cap).expect("a non-empty range"),
        )
        .unwrap();
    let mask = causal_mask_from_pos(&b, pos, cap, None).unwrap();
    let len = b
        .binary_scalar(
            poot_graph_ir::BinOp::Add,
            b.reshape(pos, vec![1]).unwrap(),
            poot_graph_ir::Scalar::I32(1),
        )
        .unwrap();
    let live = attention_live(
        &b,
        q,
        AttentionKeys::Single(k),
        v,
        1,
        0.125,
        mask,
        None,
        len,
    )
    .expect("the live composite builds");
    let graph = b.finish(live.out).unwrap();
    for (name, target) in targets() {
        let program = compiled(&graph, &target);
        let [(_, _, plan, spec)] = attentions(&program)[..] else {
            panic!("{name}: the guarded composite fuses to one attention");
        };
        assert_eq!(spec.bound, poot_kernelgen::KeyBound::Live, "{name}");
        assert!(
            matches!(spec.schedule, AttentionSchedule::SplitKv { .. }),
            "{name}: {spec:?}"
        );
        assert_eq!((plan.len(), plan.scratch().len()), (2, 1), "{name}");
    }
}

/// The rows already fill the device, or the cache is too short for two chunks: the streaming schedule, one
/// launch, no scratch. A decode with `hq >= compute_units` gains nothing from splitting.
#[test]
fn a_decode_that_fills_the_device_or_has_a_short_cache_keeps_the_streaming_schedule() {
    for (name, target) in targets() {
        let Target::Gpu { caps, .. } = &target else {
            unreachable!("targets() lists GPU targets")
        };
        let units = caps.compute_units as usize;
        for (graph, why) in [
            (decode_chain(1, units, 1024, 64), "one row per compute unit"),
            (
                decode_chain(1, 2, 192, 64),
                "192 keys: under two 128-key chunks",
            ),
        ] {
            let program = compiled(&graph, &target);
            let [(_, _, plan, spec)] = attentions(&program)[..] else {
                panic!("{name}: one attention");
            };
            assert!(
                matches!(spec.schedule, AttentionSchedule::Streaming { .. }),
                "{name} {why}: {spec:?}"
            );
            assert_eq!((plan.len(), plan.scratch().len()), (1, 0), "{name} {why}");
        }
    }
}

/// The scratch is one more buffer of the plan the values share: `slot_count` does not grow with the
/// layers (a later layer's scratch takes a slot an earlier one freed), and no layer's scratch shares a slot
/// with its own operands or output.
///
/// Mutation: end a scratch's lifetime one equation early (`death = birth - 1` in `plan_buffers`); the
/// layer's output is then colored onto the scratch's slot and the disjointness assertion goes red.
#[test]
fn the_scratch_is_colored_with_the_values_and_reused_across_layers() {
    let (_, target) = targets().remove(0);
    let counts: Vec<usize> = [3, 6, 9]
        .into_iter()
        .map(|layers| {
            compiled(&decode_chain(layers, 2, 1024, 64), &target)
                .buffer_plan()
                .slot_count()
        })
        .collect();
    assert_eq!(
        counts[0], counts[1],
        "slot_count grew with layers: {counts:?}"
    );
    assert_eq!(
        counts[1], counts[2],
        "slot_count grew with layers: {counts:?}"
    );

    let program = compiled(&decode_chain(4, 2, 1024, 64), &target);
    let plan = program.buffer_plan();
    let mut scratch_slots = Vec::new();
    for (index, eqn, dispatches, _) in attentions(&program) {
        assert_eq!(dispatches.scratch().len(), 1);
        let slot = plan
            .scratch_slot(index, ScratchId::new(0))
            .expect("every declared scratch is colored");
        assert_eq!(
            plan.slot_bytes(slot),
            dispatches.scratch()[0].bytes() as usize,
            "the slot is the scratch's size (no larger occupant shares it)"
        );
        for value in std::iter::once(eqn.out).chain(value_ids(eqn)) {
            assert_ne!(
                plan.slot(value),
                Some(slot),
                "layer {index}: {value:?} shares the scratch's slot"
            );
        }
        scratch_slots.push(slot);
    }
    let mut distinct = scratch_slots.clone();
    distinct.sort_by_key(|slot| slot.index());
    distinct.dedup();
    assert!(
        distinct.len() < scratch_slots.len(),
        "a later layer's scratch reuses an earlier one's slot: {scratch_slots:?}"
    );
}

/// One `Attention` equation staged directly (not recognized from primitives), so a planner refusal is the
/// typed error `compile` returns and not a fusion pass quietly reverting to the decomposed chain.
fn staged_decode(hq: usize, cap: usize, d: usize) -> Graph {
    let b = Builder::for_tests();
    let q = b
        .slot_named(Slot::Activation, "q", TensorType::f32(vec![1, hq, 1, d]))
        .unwrap();
    let k = b
        .slot_named(Slot::Activation, "k", TensorType::f32(vec![1, hq, cap, d]))
        .unwrap();
    let v = b
        .slot_named(Slot::Activation, "v", TensorType::f32(vec![1, hq, cap, d]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
        .unwrap();
    let mut plan = b.append_plan(0).unwrap();
    let out = plan
        .equation(
            OpKind::Attention(AttentionAttrs::new(1, 0.125, None)),
            [q, k, v, mask]
                .map(|operand| poot_graph_ir::Operand::Value(operand.id()))
                .to_vec(),
        )
        .expect("stage the decode attention");
    plan.declare_result(out).expect("declare the result");
    let mut prepared = b.preflight_append(plan).expect("preflight");
    let committed = b.commit_append(&mut prepared).expect("commit");
    b.finish(committed).unwrap()
}

/// A device whose single-buffer ceiling is below the partial-state scratch plans the same decode on the
/// streaming schedule, not a plan it cannot allocate.
///
/// Mutation: remove the `max_buffer_bytes` gate from `attention_splits`; the planner then emits the split
/// plan and `validate_planned_resources` refuses it with `Capability::ScratchBytes` (typed, not an
/// allocation failure at the executor), so `compile` returns `Err` and this row goes red.
#[test]
fn a_buffer_limit_below_the_scratch_falls_back_to_streaming() {
    let graph = staged_decode(2, 1024, 64);
    let wgpu = default_caps_for(Backend::SpirvVulkan);
    let roomy = compiled(&graph, &target(Backend::SpirvVulkan, wgpu));
    let [(_, _, plan, spec)] = attentions(&roomy)[..] else {
        panic!("one attention");
    };
    let scratch_bytes = plan.scratch()[0].bytes();
    assert!(matches!(spec.schedule, AttentionSchedule::SplitKv { .. }));

    let tight = DeviceCaps {
        max_buffer_bytes: scratch_bytes - 1,
        ..wgpu
    };
    let program = match compile(&graph, &target(Backend::SpirvVulkan, tight), &OPTIONS) {
        Ok(program) => program,
        Err(CompileError::Plan(error)) => match *error {
            PlanError::Refused(refusal) => panic!(
                "a refusal ({:?}), not the fall back to streaming",
                refusal.missing
            ),
            other => panic!("{other:?}"),
        },
        Err(other) => panic!("{other:?}"),
    };
    let [(_, _, plan, spec)] = attentions(&program)[..] else {
        panic!("one attention");
    };
    assert!(
        matches!(spec.schedule, AttentionSchedule::Streaming { .. }),
        "{spec:?}"
    );
    assert!(plan.scratch().is_empty());
}

/// Every dispatch of the split plan stays inside the device's grid even where the cap forces the 2-D fold:
/// a device whose X grid is 4 workgroups plans the partials stage's 16 `(row, split)` workgroups as a
/// 4 x 4 grid and the combine's 2 rows as 2 x 1, each within `max_grid`.
///
/// Mutation: fold against `u32::MAX` instead of `caps.max_grid[0]` in `request.rs`'s `fold_within_grid`;
/// the partials launch is then 16 workgroups wide and the assertion names axis 0.
#[test]
fn every_split_dispatch_stays_within_max_grid() {
    let graph = decode_chain(1, 2, 1024, 64);
    let caps = DeviceCaps {
        max_grid: [4, 8, 65_535],
        ..default_caps_for(Backend::SpirvVulkan)
    };
    let max_grid = caps.max_grid;
    let target = target(Backend::SpirvVulkan, caps);
    let program = compiled(&graph, &target);
    let [(_, _, plan, _)] = attentions(&program)[..] else {
        panic!("one attention");
    };
    assert_eq!(plan.len(), 2, "the narrow grid still splits");
    for dispatch in plan.iter() {
        let launch = dispatch.launch();
        let width = dispatch.body().workgroup_size;
        for axis in 0..3 {
            let groups = launch.grid[axis].div_ceil(width[axis].max(1));
            assert!(
                groups <= max_grid[axis],
                "axis {axis}: {groups} workgroups past max_grid {:?}",
                max_grid
            );
        }
    }
}
