//! Backend-neutral planning for native captured-state commit.
//!
//! A graph state transition is simultaneous assignment. Native executors may donate a state input buffer to
//! its producing equation only when overwriting that input cannot change any later graph read, the primary
//! output, or another state source. Every remaining non-identity edge is committed by the backend through a
//! two-phase device copy after equation execution.

use std::collections::HashMap;

use poot_graph_ir::{Graph, OpKind, Operand, ValueId, ValueTable};

use crate::{Plan, PlanError, value_ids};

/// One explicit state assignment that the backend must commit after graph equations finish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateCommitCopy {
    /// The graph input the state is read from.
    pub state_input: ValueId,
    /// The graph value that holds the state's next contents.
    pub state_output: ValueId,
}

/// Native state execution plan. `inplace` maps the materializing equation output to the paired state input;
/// `copies` retains graph value ids so each backend can resolve the final buffers after aliases are bound.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateCommitPlan {
    /// Materializing equation output mapped to the state input it overwrites in place.
    pub inplace: HashMap<ValueId, ValueId>,
    /// Assignments that cannot be made in place and are copied after the equations finish.
    pub copies: Vec<StateCommitCopy>,
    /// The primary output aliases a destination that explicit commit overwrites and must be snapshotted.
    pub preserve_output: bool,
}

fn update_base(eqn: &poot_graph_ir::Eqn) -> Option<ValueId> {
    if !matches!(
        eqn.op,
        OpKind::DynamicUpdateSlice { .. } | OpKind::ScatterUpdate
    ) {
        return None;
    }
    match eqn.inputs.first() {
        Some(Operand::Value(base)) => Some(*base),
        _ => None,
    }
}

/// Derive donation and explicit-commit decisions from the program's own plans, index-aligned with
/// `g.eqns` (`compile` already planned every equation once via `plan_graph`; a second independent
/// planning pass here broke `compile_plans_each_equation_exactly_once`, spike-546 F7). `compile` calls
/// this to fill `Program::state_commit`.
///
/// State outputs are pinned by `compute_views`, so strided views materialize before this boundary (the
/// plans this takes were planned against those views). Planner aliases such as `Reshape` and
/// same-dtype `Cast` are followed to their storage root. Donation is narrow: only DynamicUpdateSlice
/// and ScatterUpdate have an overlapping base/output kernel contract, and for ScatterUpdate it is a
/// precondition: a graph whose scatter cannot be donated is refused with [`PlanError::ScatterNotInPlace`].
pub(crate) fn state_commit_from_plans(
    g: &Graph,
    plans: &[Plan],
) -> Result<StateCommitPlan, PlanError> {
    check_state_shapes(g)?;

    let mut roots: ValueTable<ValueId> = g.values().ids().collect();
    let mut producers: HashMap<ValueId, usize> = HashMap::new();

    for (eqn_idx, (eqn, plan)) in g.eqns().iter().zip(plans).enumerate() {
        match plan {
            Plan::Alias(source) => roots[eqn.out] = roots[*source],
            Plan::View { src, .. } => roots[eqn.out] = roots[*src],
            _ => {
                roots[eqn.out] = eqn.out;
                producers.insert(eqn.out, eqn_idx);
            }
        }
    }

    let commit = commit_from_roots(g, &roots, &producers);
    // `ScatterUpdate`'s kernel writes only the named rows into its output buffer, so the output must be
    // the base's own state buffer: an undonated scatter would leave every unnamed row unwritten.
    if let Some(scatter) = g
        .eqns()
        .iter()
        .find(|eqn| eqn.op == OpKind::ScatterUpdate && !commit.inplace.contains_key(&eqn.out))
    {
        return Err(PlanError::ScatterNotInPlace { value: scatter.out });
    }
    Ok(commit)
}

fn check_state_shapes(g: &Graph) -> Result<(), PlanError> {
    for (state_input, state_output) in g.state() {
        let input = g.aval(state_input);
        let output = g.aval(state_output);
        if input.shape != output.shape || input.dtype != output.dtype {
            return Err(PlanError::BadShape(format!(
                "state {state_input} <- {state_output} has mismatched types: {:?} {:?} <- {:?} {:?}",
                input.shape, input.dtype, output.shape, output.dtype
            )));
        }
    }
    Ok(())
}

/// [`state_commit_from_plans`]'s tail: donation and explicit-commit decisions from each state edge's
/// alias root and producer.
fn commit_from_roots(
    g: &Graph,
    roots: &ValueTable<ValueId>,
    producers: &HashMap<ValueId, usize>,
) -> StateCommitPlan {
    let state_roots: Vec<ValueId> = g
        .state()
        .map(|(_, state_output)| roots[state_output])
        .collect();
    let mut root_counts: HashMap<ValueId, usize> = HashMap::new();
    for &root in &state_roots {
        *root_counts.entry(root).or_default() += 1;
    }

    let output_root = roots[g.result()];
    let mut result = StateCommitPlan::default();
    for (edge_idx, ((state_input, state_output), &state_root)) in
        g.state().zip(&state_roots).enumerate()
    {
        if state_root == state_input {
            continue;
        }

        let can_donate = producers.get(&state_root).is_some_and(|&producer_idx| {
            let producer = &g.eqns()[producer_idx];
            let paired_base = update_base(producer).is_some_and(|base| roots[base] == state_input);
            let independent_update_inputs = value_ids(producer)
                .into_iter()
                .skip(1)
                .all(|value| roots[value] != state_input);
            let unique_state_source = root_counts.get(&state_root) == Some(&1);
            let input_is_not_another_state_source = state_roots
                .iter()
                .enumerate()
                .all(|(other_idx, &root)| other_idx == edge_idx || root != state_input);
            let input_is_not_primary_output = output_root != state_input;
            let input_dead_after_update = g.eqns()[producer_idx + 1..].iter().all(|eqn| {
                value_ids(eqn)
                    .into_iter()
                    .all(|value| roots[value] != state_input)
            });

            paired_base
                && independent_update_inputs
                && unique_state_source
                && input_is_not_another_state_source
                && input_is_not_primary_output
                && input_dead_after_update
        });

        if can_donate {
            result.inplace.insert(state_root, state_input);
        } else {
            result.copies.push(StateCommitCopy {
                state_input,
                state_output,
            });
        }
    }
    result.preserve_output = result
        .copies
        .iter()
        .any(|copy| copy.state_input == output_root);

    result
}

#[cfg(test)]
mod tests {
    use crate::{compute_views, plan_eqn_views};
    use poot_graph_ir::builder::Builder;
    use poot_graph_ir::graph::StateRole;
    use poot_graph_ir::op::BinOp;
    use poot_graph_ir::types::{Scalar, TensorType};
    use poot_target::{Backend, DeviceCaps};
    use poot_test_util::device_caps::default_caps_for;

    use super::*;

    #[test]
    fn native_state_plan_classifies_pass_through_alias_and_identity_edges() {
        let builder = Builder::for_tests();
        let a = builder
            .state_input("a", TensorType::f32(vec![1]), StateRole::Recurrent)
            .unwrap();
        let b = builder
            .state_input("b", TensorType::f32(vec![1]), StateRole::Recurrent)
            .unwrap();
        let output = builder.binary(BinOp::Sub, a, b).unwrap();
        let graph = builder
            .finish(poot_graph_ir::Outputs::new(output).with_state(&[(a, b), (b, a)]))
            .unwrap();
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(plan.inplace.is_empty());
        assert!(!plan.preserve_output);
        assert_eq!(
            plan.copies,
            [
                StateCommitCopy {
                    state_input: a.id(),
                    state_output: b.id(),
                },
                StateCommitCopy {
                    state_input: b.id(),
                    state_output: a.id(),
                },
            ]
        );

        let builder = Builder::for_tests();
        let a = builder
            .state_input("a", TensorType::f32(vec![1]), StateRole::Recurrent)
            .unwrap();
        let b = builder
            .state_input("b", TensorType::f32(vec![1]), StateRole::Recurrent)
            .unwrap();
        let graph = builder
            .finish(poot_graph_ir::Outputs::new(a).with_state(&[(a, b), (b, a)]))
            .unwrap();
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(
            plan.preserve_output,
            "commit overwrites the primary output alias"
        );

        let builder = Builder::for_tests();
        let state = builder
            .state_input("state", TensorType::f32(vec![1, 1]), StateRole::Recurrent)
            .unwrap();
        let reshaped = builder.reshape(state, vec![1, 1]).unwrap();
        let graph = builder
            .finish(poot_graph_ir::Outputs::new(reshaped).with_state(&[(state, reshaped)]))
            .unwrap();
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(plan.inplace.is_empty());
        assert!(plan.copies.is_empty(), "reshape identity needs no copy");

        let builder = Builder::for_tests();
        let a = builder
            .state_input("a", TensorType::f32(vec![1, 2]), StateRole::Recurrent)
            .unwrap();
        let b = builder
            .state_input("b", TensorType::f32(vec![2]), StateRole::Recurrent)
            .unwrap();
        let reshaped = builder.reshape(b, vec![1, 2]).unwrap();
        let graph = builder
            .finish(poot_graph_ir::Outputs::new(a).with_state(&[(a, reshaped), (b, b)]))
            .unwrap();
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(plan.inplace.is_empty());
        assert_eq!(
            plan.copies,
            [StateCommitCopy {
                state_input: a.id(),
                state_output: reshaped.id(),
            }]
        );

        let builder = Builder::for_tests();
        let a = builder
            .state_input("a", TensorType::f32(vec![1]), StateRole::Recurrent)
            .unwrap();
        let b = builder
            .state_input("b", TensorType::f32(vec![1]), StateRole::Recurrent)
            .unwrap();
        let c = builder
            .state_input("c", TensorType::f32(vec![1]), StateRole::Recurrent)
            .unwrap();
        let output = builder.binary(BinOp::Add, a, b).unwrap();
        let graph = builder
            .finish(poot_graph_ir::Outputs::new(output).with_state(&[(a, b), (b, c), (c, a)]))
            .unwrap();
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(plan.inplace.is_empty());
        assert_eq!(plan.copies.len(), 3, "a three-cycle needs three snapshots");

        let builder = Builder::for_tests();
        let a = builder
            .state_input("a", TensorType::f32(vec![1]), StateRole::Recurrent)
            .unwrap();
        let b = builder
            .state_input("b", TensorType::f32(vec![1]), StateRole::Recurrent)
            .unwrap();
        let source = builder
            .state_input("source", TensorType::f32(vec![1]), StateRole::Recurrent)
            .unwrap();
        let output = builder.binary(BinOp::Add, a, b).unwrap();
        let graph = builder
            .finish(poot_graph_ir::Outputs::new(output).with_state(&[
                (a, source),
                (b, source),
                (source, source),
            ]))
            .unwrap();
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(plan.inplace.is_empty());
        assert_eq!(
            plan.copies.len(),
            2,
            "both shared-source destinations need snapshots"
        );
    }

    #[test]
    fn native_state_plan_donates_only_proved_dead_paired_updates() {
        let builder = Builder::for_tests();
        let state = builder
            .state_input("state", TensorType::f32(vec![2]), StateRole::Recurrent)
            .unwrap();
        let update = builder
            .slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![1]))
            .unwrap();
        let updated = builder.dynamic_update_slice(state, update, 1, 0).unwrap();
        let graph = builder
            .finish(poot_graph_ir::Outputs::new(updated).with_state(&[(state, updated)]))
            .unwrap();
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert_eq!(plan.inplace, HashMap::from([(updated.id(), state.id())]));
        assert!(plan.copies.is_empty());

        let builder = Builder::for_tests();
        let state = builder
            .state_input("state", TensorType::f32(vec![2]), StateRole::Recurrent)
            .unwrap();
        let update = builder
            .slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![1]))
            .unwrap();
        let updated = builder.dynamic_update_slice(state, update, 1, 0).unwrap();
        let old_state_sum = builder
            .binary_scalar(BinOp::Mul, state, Scalar::F32(2.0))
            .unwrap();
        let output = builder.binary(BinOp::Add, updated, old_state_sum).unwrap();
        let graph = builder
            .finish(poot_graph_ir::Outputs::new(output).with_state(&[(state, updated)]))
            .unwrap();
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(
            plan.inplace.is_empty(),
            "a later old-state read must disable donation"
        );
        assert_eq!(plan.copies.len(), 1);
    }

    /// The production paged-KV write shape (`write_paged`): the pool is scatter-updated by this step's new
    /// rows, then the step reads the pool back through a gather of the post-write value. The pool is read by
    /// nothing else, so the scatter's output is donated to the pool's own buffer and no copy is committed.
    #[test]
    fn a_paged_pool_write_read_back_after_the_scatter_is_donated_in_place() {
        let builder = Builder::for_tests();
        let pool = builder
            .state_input(
                "pool",
                TensorType::f32(vec![8, 2, 4]),
                StateRole::Positional { axis: 0 },
            )
            .unwrap();
        let new = builder
            .slot(
                poot_graph_ir::Slot::Activation,
                TensorType::f32(vec![3, 2, 4]),
            )
            .unwrap();
        let write = builder.write_map("write", 3, 8).unwrap();
        let read = builder
            .slot_named(
                poot_graph_ir::Slot::SlotMap,
                "read",
                TensorType::new(vec![5], poot_tensor::DType::I32),
            )
            .unwrap();
        let written = builder.scatter_update(pool, new, write).unwrap();
        let viewed = builder.gather(written, 0, read).unwrap();
        let graph = builder
            .finish(poot_graph_ir::Outputs::new(viewed).with_state(&[(pool, written)]))
            .unwrap();
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert_eq!(plan.inplace, HashMap::from([(written.id(), pool.id())]));
        assert!(
            plan.copies.is_empty(),
            "a donated pool needs no commit copy"
        );
    }

    /// A step that also reads the pool as it was before the write cannot donate it: overwriting the pool
    /// would change that read. The scatter kernel writes only the named rows, so an undonated scatter would
    /// leave the rest of its output unwritten: the graph is refused, naming the scatter's output.
    #[test]
    fn a_paged_pool_read_before_the_write_is_refused_not_copied() {
        let builder = Builder::for_tests();
        let pool = builder
            .state_input(
                "pool",
                TensorType::f32(vec![8, 2, 4]),
                StateRole::Positional { axis: 0 },
            )
            .unwrap();
        let new = builder
            .slot(
                poot_graph_ir::Slot::Activation,
                TensorType::f32(vec![3, 2, 4]),
            )
            .unwrap();
        let write = builder.write_map("write", 3, 8).unwrap();
        let read = builder
            .slot_named(
                poot_graph_ir::Slot::SlotMap,
                "read",
                TensorType::new(vec![5], poot_tensor::DType::I32),
            )
            .unwrap();
        let written = builder.scatter_update(pool, new, write).unwrap();
        let stale = builder.gather(pool, 0, read).unwrap();
        let fresh = builder.gather(written, 0, read).unwrap();
        let viewed = builder.binary(BinOp::Add, stale, fresh).unwrap();
        let graph = builder
            .finish(poot_graph_ir::Outputs::new(viewed).with_state(&[(pool, written)]))
            .unwrap();
        match commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)) {
            Err(PlanError::ScatterNotInPlace { value }) => assert_eq!(value, written.id()),
            other => panic!("expected ScatterNotInPlace, got {other:?}"),
        }
    }

    /// A scatter whose base is not a carried state has no buffer to write in place: refused, naming it.
    #[test]
    fn a_scatter_over_a_value_that_is_not_state_is_refused() {
        let builder = Builder::for_tests();
        let base = builder
            .slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![8, 2]))
            .unwrap();
        let new = builder
            .slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![3, 2]))
            .unwrap();
        let write = builder.write_map("write", 3, 8).unwrap();
        let written = builder.scatter_update(base, new, write).unwrap();
        let graph = builder.finish(written).unwrap();
        match commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)) {
            Err(PlanError::ScatterNotInPlace { value }) => assert_eq!(value, written.id()),
            other => panic!("expected ScatterNotInPlace, got {other:?}"),
        }
    }

    /// Card 626 deleted `plan_native_state_commit`, the production executors' own
    /// re-planning entry point: `state_commit_from_plans` (the live `compile` path) takes plans
    /// `plan_graph` already computed instead of planning each equation again. This module's fixtures want
    /// "plan a graph and derive its state-commit plan" in one call, so this re-plans each equation here
    /// (test-only; `compile` itself never does this twice) and routes the result through the real
    /// production function instead of restating `commit_from_roots`'s logic a second time.
    fn commit_for_test(
        g: &Graph,
        backend: Backend,
        caps: &DeviceCaps,
    ) -> Result<StateCommitPlan, PlanError> {
        let views = compute_views(g, backend);
        let plans: Vec<Plan> = g
            .eqns()
            .iter()
            .map(|eqn| {
                plan_eqn_views(
                    g,
                    eqn,
                    backend,
                    &views,
                    caps,
                    &crate::test_support::roomy_body_limits(),
                )
            })
            .collect::<Result<_, _>>()?;
        state_commit_from_plans(g, &plans)
    }
}
