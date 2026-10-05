//! Backend-neutral planning for native captured-state commit.
//!
//! A graph state transition is simultaneous assignment. Native executors may donate a state input buffer to
//! its producing equation only when overwriting that input cannot change any later graph read, the primary
//! output, or another state source. Every remaining non-identity edge is committed by the backend through a
//! two-phase device copy after equation execution.

use std::collections::HashMap;

#[cfg(test)]
use poot_graph_ir::ValidationOutputs;
use poot_graph_ir::{Graph, OpKind, Operand, ValidationChannel, ValueId};

#[cfg(test)]
use poot_target::{Backend, DeviceCaps};

#[cfg(test)]
use crate::{
    ExactI32StorageAnalysis, ValidationPacketPlan, compute_views, plan_eqn_views_analyzed,
    plan_validation_packet,
};
use crate::{Plan, PlanError, value_ids};
#[cfg(test)]
use poot_test_util::device_caps::default_caps_for;

/// One explicit state assignment that the backend must commit after graph equations finish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateCommitCopy {
    pub state_input: ValueId,
    pub state_output: ValueId,
}

/// Native state execution plan. `inplace` maps the materializing equation output to the paired state input;
/// `copies` retains graph value ids so each backend can resolve the final buffers after aliases are bound.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateCommitPlan {
    pub inplace: HashMap<ValueId, ValueId>,
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
/// and ScatterUpdate have an overlapping base/output kernel contract.
///
/// Generic over the validation channel (Card 375c) so a captured backend can plan a
/// `Graph<ValidationOutputs>` like an ordinary `Graph`. A donation admitted here is not necessarily safe to
/// roll back; see `journal_donated_updates` for the statically-recoverable-footprint requirement a
/// native transaction adds.
pub(crate) fn state_commit_from_plans<V: ValidationChannel>(
    g: &Graph<V>,
    plans: &[Plan],
) -> Result<StateCommitPlan, PlanError> {
    check_state_shapes(g)?;

    let mut roots: Vec<ValueId> = (0..g.values.len()).collect();
    let mut producers: HashMap<ValueId, usize> = HashMap::new();

    for (eqn_idx, (eqn, plan)) in g.eqns.iter().zip(plans).enumerate() {
        match plan {
            Plan::Alias(source) => roots[eqn.out] = roots[*source],
            Plan::View { src, .. } => roots[eqn.out] = roots[*src],
            _ => {
                roots[eqn.out] = eqn.out;
                producers.insert(eqn.out, eqn_idx);
            }
        }
    }

    Ok(commit_from_roots(g, &roots, &producers))
}

fn check_state_shapes<V: ValidationChannel>(g: &Graph<V>) -> Result<(), PlanError> {
    for &(state_input, state_output) in &g.state {
        let input = g.aval(state_input);
        let output = g.aval(state_output);
        if input.shape != output.shape || input.dtype != output.dtype {
            return Err(PlanError::BadShape(format!(
                "state v{state_input} <- v{state_output} has mismatched types: {:?} {:?} <- {:?} {:?}",
                input.shape, input.dtype, output.shape, output.dtype
            )));
        }
    }
    Ok(())
}

/// [`state_commit_from_plans`]'s tail: donation and explicit-commit decisions from each state edge's
/// alias root and producer.
fn commit_from_roots<V: ValidationChannel>(
    g: &Graph<V>,
    roots: &[ValueId],
    producers: &HashMap<ValueId, usize>,
) -> StateCommitPlan {
    let state_roots: Vec<ValueId> = g
        .state
        .iter()
        .map(|&(_, state_output)| roots[state_output])
        .collect();
    let mut root_counts: HashMap<ValueId, usize> = HashMap::new();
    for &root in &state_roots {
        *root_counts.entry(root).or_default() += 1;
    }

    let output_root = roots[g.output];
    let mut result = StateCommitPlan::default();
    for (edge_idx, (&(state_input, state_output), &state_root)) in
        g.state.iter().zip(&state_roots).enumerate()
    {
        if state_root == state_input {
            continue;
        }

        let can_donate = producers.get(&state_root).is_some_and(|&producer_idx| {
            let producer = &g.eqns[producer_idx];
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
            let input_dead_after_update = g.eqns[producer_idx + 1..].iter().all(|eqn| {
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

/// The exact overwrite footprint of one donated state edge (Card 375c).
///
/// Fields are private with no constructor outside this module: only `journal_donated_updates`'s
/// `DynamicUpdateSlice` arm builds one (this module's tests build literals as fixtures).
///
/// The update operand has a static shape, so the size of the overwritten region is bounded by that shape
/// at the equation's axis. That is all this type guarantees: it carries no offset, so a native backend
/// must resolve the runtime index operand at replay and use the same offset for the pre-write snapshot
/// and the rollback restore (Card 375d/375e).
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct DonationFootprint {
    /// The materializing equation's output, aliased onto `state_input` by [`StateCommitPlan::inplace`].
    state_root: ValueId,
    /// The paired state input this donation overwrites in place.
    state_input: ValueId,
    /// The axis the donating `DynamicUpdateSlice` writes along.
    axis: usize,
    /// The update operand's static shape: the exact region a rollback must restore.
    shape: Vec<usize>,
}

#[cfg(test)]
impl DonationFootprint {
    /// The paired state input this donation overwrites in place.
    fn state_input(&self) -> ValueId {
        self.state_input
    }
}

/// Build the checked rollback journal for every donation `commit` recorded, or reject the first one whose
/// overwrite region a native transaction cannot statically recover (Card 375c).
///
/// `state_commit_from_plans` donates an edge whenever aliasing is safe for its own two-phase commit
/// (`update_base` grants this to `DynamicUpdateSlice` and `ScatterUpdate`). A native transaction needs the
/// exact bytes a rollback must restore, known without inspecting device data. `DynamicUpdateSlice` gives
/// that (see [`DonationFootprint`]). `ScatterUpdate`'s `inv` operand is a full-length per-base-row selector
/// of runtime data, so any subset of rows may be touched and no static shape smaller than the whole buffer
/// exists; snapshotting the entire state would defeat the donation, so it is rejected with
/// [`PlanError::UnrecoverableDonation`].
#[cfg(test)]
fn journal_donated_updates<V: ValidationChannel>(
    g: &Graph<V>,
    commit: &StateCommitPlan,
) -> Result<Vec<DonationFootprint>, PlanError> {
    let mut donations = commit
        .inplace
        .iter()
        .map(|(&state_root, &state_input)| {
            let producer = g
                .eqns
                .iter()
                .find(|eqn| eqn.out == state_root)
                .expect("state planner only donates materializing equation outputs");
            match &producer.op {
                OpKind::DynamicUpdateSlice { axis } => match producer.inputs.get(1) {
                    Some(Operand::Value(update)) => Ok(DonationFootprint {
                        state_root,
                        state_input,
                        axis: *axis,
                        shape: g.aval(*update).shape.clone(),
                    }),
                    _ => Err(PlanError::BadShape(format!(
                        "state v{state_input} donated DynamicUpdateSlice v{state_root} has no value update operand"
                    ))),
                },
                other => Err(PlanError::UnrecoverableDonation {
                    state_input,
                    op: other.name(),
                }),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    // `commit.inplace` is a `HashMap`; sort for a deterministic journal.
    donations.sort_by_key(|d| d.state_input);
    Ok(donations)
}

/// A backend-neutral plan for one native captured-state transaction (Card 375c): the state-commit plan
/// (donation and explicit two-phase copies), the validation packet plan (Card 375a/375b), and the rollback
/// journal for every donated edge.
///
/// A captured backend (Card 375d PTX, Card 375e ROCm) replays these phases in order, with every address
/// and the packet shape fixed at capture:
///
/// 1. **Compute**: the graph's equations run, including donated in-place writes. A donated write happens
///    only after its [`DonationFootprint`] has been snapshotted.
/// 2. **Materialize the packet**: the compact witness packet is read through its fixed device address.
/// 3. The host waits for the readback, then commits or rolls back, never both and never neither:
///    - **Commit** (every lane passed): snapshot the primary output first if `commit.preserve_output` (it
///      aliases a destination the copies overwrite), run `commit.copies`' simultaneous two-phase swap,
///      then advance the generation.
///    - **Rollback** (some lane failed): restore every entry in `donations` from its pre-write snapshot.
///      `commit.copies` edges need no restoration: their compute wrote a separate `state_output` buffer
///      never aliased to published storage. Generation and published bytes are unchanged.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct NativeStateTransactionPlan {
    commit: StateCommitPlan,
    packet: ValidationPacketPlan,
    donations: Vec<DonationFootprint>,
}

/// Card 626 deleted `plan_native_state_commit`, the production executors' own
/// re-planning entry point: `state_commit_from_plans` (the live `compile` path) takes plans
/// `plan_graph` already computed instead of planning each equation again. This module's fixtures want
/// "plan a graph and derive its state-commit plan" in one call, so this re-plans each equation here
/// (test-only; `compile` itself never does this twice) and routes the result through the real
/// production function instead of restating `commit_from_roots`'s logic a second time.
#[cfg(test)]
fn commit_for_test<V: ValidationChannel>(
    g: &Graph<V>,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<StateCommitPlan, PlanError> {
    let views = compute_views(g, backend);
    let exact_i32 = ExactI32StorageAnalysis::new(g);
    let plans: Vec<Plan> = g
        .eqns
        .iter()
        .map(|eqn| {
            plan_eqn_views_analyzed(
                &exact_i32,
                g,
                eqn,
                backend,
                1,
                &views,
                caps,
                &poot_test_util::graph_fixtures::roomy_body_limits(),
            )
        })
        .collect::<Result<_, _>>()?;
    state_commit_from_plans(g, &plans)
}

/// Plan one native captured-state transaction over a validation-bearing graph (Card 375c).
///
/// Composes [`state_commit_from_plans`], [`journal_donated_updates`], and [`plan_validation_packet`]
/// without re-deriving their analyses; it adds no compute or packet semantics of its own.
#[cfg(test)]
fn plan_native_state_transaction(
    g: &Graph<ValidationOutputs>,
    backend: Backend,
) -> Result<NativeStateTransactionPlan, PlanError> {
    let commit = commit_for_test(g, backend, &default_caps_for(backend))?;
    let donations = journal_donated_updates(g, &commit)?;
    let packet = plan_validation_packet(g)?;
    Ok(NativeStateTransactionPlan {
        commit,
        packet,
        donations,
    })
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::builder::Builder;
    use poot_graph_ir::graph::StateRole;
    use poot_graph_ir::op::BinOp;
    use poot_graph_ir::types::{Scalar, TensorType};

    use super::*;

    #[test]
    fn native_state_plan_classifies_pass_through_alias_and_identity_edges() {
        let builder = Builder::new();
        let a = builder.state_input("a", TensorType::f32(vec![1]), StateRole::Recurrent);
        let b = builder.state_input("b", TensorType::f32(vec![1]), StateRole::Recurrent);
        let output = builder.binary(BinOp::Sub, a, b);
        let graph = builder.finish_with_state(output, &[(a, b), (b, a)]);
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(plan.inplace.is_empty());
        assert!(!plan.preserve_output);
        assert_eq!(
            plan.copies,
            [
                StateCommitCopy {
                    state_input: a.id,
                    state_output: b.id,
                },
                StateCommitCopy {
                    state_input: b.id,
                    state_output: a.id,
                },
            ]
        );

        let builder = Builder::new();
        let a = builder.state_input("a", TensorType::f32(vec![1]), StateRole::Recurrent);
        let b = builder.state_input("b", TensorType::f32(vec![1]), StateRole::Recurrent);
        let graph = builder.finish_with_state(a, &[(a, b), (b, a)]);
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(
            plan.preserve_output,
            "commit overwrites the primary output alias"
        );

        let builder = Builder::new();
        let state = builder.state_input("state", TensorType::f32(vec![1, 1]), StateRole::Recurrent);
        let reshaped = builder.reshape(state, vec![1, 1]);
        let graph = builder.finish_with_state(reshaped, &[(state, reshaped)]);
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(plan.inplace.is_empty());
        assert!(plan.copies.is_empty(), "reshape identity needs no copy");

        let builder = Builder::new();
        let a = builder.state_input("a", TensorType::f32(vec![1, 2]), StateRole::Recurrent);
        let b = builder.state_input("b", TensorType::f32(vec![2]), StateRole::Recurrent);
        let reshaped = builder.reshape(b, vec![1, 2]);
        let graph = builder.finish_with_state(a, &[(a, reshaped), (b, b)]);
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(plan.inplace.is_empty());
        assert_eq!(
            plan.copies,
            [StateCommitCopy {
                state_input: a.id,
                state_output: reshaped.id,
            }]
        );

        let builder = Builder::new();
        let a = builder.state_input("a", TensorType::f32(vec![1]), StateRole::Recurrent);
        let b = builder.state_input("b", TensorType::f32(vec![1]), StateRole::Recurrent);
        let c = builder.state_input("c", TensorType::f32(vec![1]), StateRole::Recurrent);
        let output = builder.binary(BinOp::Add, a, b);
        let graph = builder.finish_with_state(output, &[(a, b), (b, c), (c, a)]);
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(plan.inplace.is_empty());
        assert_eq!(plan.copies.len(), 3, "a three-cycle needs three snapshots");

        let builder = Builder::new();
        let a = builder.state_input("a", TensorType::f32(vec![1]), StateRole::Recurrent);
        let b = builder.state_input("b", TensorType::f32(vec![1]), StateRole::Recurrent);
        let source = builder.state_input("source", TensorType::f32(vec![1]), StateRole::Recurrent);
        let output = builder.binary(BinOp::Add, a, b);
        let graph =
            builder.finish_with_state(output, &[(a, source), (b, source), (source, source)]);
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
        let builder = Builder::new();
        let state = builder.state_input("state", TensorType::f32(vec![2]), StateRole::Recurrent);
        let update = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![1]));
        let updated = builder.dynamic_update_slice(state, update, 1, 0);
        let graph = builder.finish_with_state(updated, &[(state, updated)]);
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert_eq!(plan.inplace, HashMap::from([(updated.id, state.id)]));
        assert!(plan.copies.is_empty());

        let builder = Builder::new();
        let state = builder.state_input("state", TensorType::f32(vec![2]), StateRole::Recurrent);
        let update = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![1]));
        let updated = builder.dynamic_update_slice(state, update, 1, 0);
        let old_state_sum = builder.binary_scalar(BinOp::Mul, state, Scalar::F32(2.0));
        let output = builder.binary(BinOp::Add, updated, old_state_sum);
        let graph = builder.finish_with_state(output, &[(state, updated)]);
        let plan =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(
            plan.inplace.is_empty(),
            "a later old-state read must disable donation"
        );
        assert_eq!(plan.copies.len(), 1);
    }

    /// A `DynamicUpdateSlice` donation's footprint is exactly the update operand's static shape at the
    /// axis the equation names, not the whole state buffer.
    #[test]
    fn journal_records_the_dynamic_update_slice_footprint() {
        let builder = Builder::new();
        let state = builder.state_input("state", TensorType::f32(vec![6, 3]), StateRole::Recurrent);
        let update = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![1, 3]));
        let updated = builder.dynamic_update_slice(state, update, 2, 0);
        let graph = builder.finish_with_state(updated, &[(state, updated)]);
        let commit =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert_eq!(commit.inplace, HashMap::from([(updated.id, state.id)]));

        let donations = journal_donated_updates(&graph, &commit).unwrap();
        assert_eq!(
            donations,
            [DonationFootprint {
                state_root: updated.id,
                state_input: state.id,
                axis: 0,
                shape: vec![1, 3],
            }]
        );
    }

    /// Multi-donation completeness, ordering, and pairing. The other journal tests use zero or one
    /// donation, where `.collect()` cannot drop or duplicate an item and a one-element list is trivially
    /// sorted, so they cannot catch a deleted `sort_by_key`, a collection that keeps only the last entry,
    /// or `axis`/`shape` fields mispaired with the wrong `state_input`. A multi-layer captured model
    /// donates one `DynamicUpdateSlice` per layer, so N > 1 is the normal case.
    ///
    /// Each donation has a distinct shape (update lengths 1..=5 along a fixed axis), and the full sorted
    /// `Vec<DonationFootprint>` is compared against five expected structs. A removed `sort_by_key` is
    /// caught with false-pass chance under 1% (`HashMap` random seed).
    #[test]
    fn journal_is_complete_ordered_and_correctly_paired_across_multiple_donations() {
        let builder = Builder::new();
        let s1 = builder.state_input("s1", TensorType::f32(vec![10]), StateRole::Recurrent);
        let u1 = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![1]));
        let updated1 = builder.dynamic_update_slice(s1, u1, 0, 0);
        let s2 = builder.state_input("s2", TensorType::f32(vec![10]), StateRole::Recurrent);
        let u2 = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![2]));
        let updated2 = builder.dynamic_update_slice(s2, u2, 0, 0);
        let s3 = builder.state_input("s3", TensorType::f32(vec![10]), StateRole::Recurrent);
        let u3 = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![3]));
        let updated3 = builder.dynamic_update_slice(s3, u3, 0, 0);
        let s4 = builder.state_input("s4", TensorType::f32(vec![10]), StateRole::Recurrent);
        let u4 = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![4]));
        let updated4 = builder.dynamic_update_slice(s4, u4, 0, 0);
        let s5 = builder.state_input("s5", TensorType::f32(vec![10]), StateRole::Recurrent);
        let u5 = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![5]));
        let updated5 = builder.dynamic_update_slice(s5, u5, 0, 0);
        let output = builder.binary(
            BinOp::Add,
            builder.binary(
                BinOp::Add,
                builder.binary(BinOp::Add, updated1, updated2),
                builder.binary(BinOp::Add, updated3, updated4),
            ),
            updated5,
        );
        let graph = builder.finish_with_state(
            output,
            &[
                (s1, updated1),
                (s2, updated2),
                (s3, updated3),
                (s4, updated4),
                (s5, updated5),
            ],
        );

        let commit =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert_eq!(
            commit.inplace.len(),
            5,
            "premise: all five independent DynamicUpdateSlice edges must be proved donatable"
        );

        // Not re-sorted here: this checks `journal_donated_updates`'s own ordering guarantee.
        let donations = journal_donated_updates(&graph, &commit).unwrap();

        assert_eq!(
            donations,
            [
                DonationFootprint {
                    state_root: updated1.id,
                    state_input: s1.id,
                    axis: 0,
                    shape: vec![1],
                },
                DonationFootprint {
                    state_root: updated2.id,
                    state_input: s2.id,
                    axis: 0,
                    shape: vec![2],
                },
                DonationFootprint {
                    state_root: updated3.id,
                    state_input: s3.id,
                    axis: 0,
                    shape: vec![3],
                },
                DonationFootprint {
                    state_root: updated4.id,
                    state_input: s4.id,
                    axis: 0,
                    shape: vec![4],
                },
                DonationFootprint {
                    state_root: updated5.id,
                    state_input: s5.id,
                    axis: 0,
                    shape: vec![5],
                },
            ],
            "a dropped, duplicated, or mispaired entry"
        );

        assert!(
            donations
                .windows(2)
                .all(|w| w[0].state_input() < w[1].state_input()),
            "journal must be sorted ascending by state_input, got {:?}",
            donations
                .iter()
                .map(|d| d.state_input())
                .collect::<Vec<_>>()
        );
    }

    /// `ScatterUpdate` stays donation-eligible for `state_commit_from_plans`, but the journal cannot
    /// statically recover which base rows its runtime `inv` map touches, so it is refused rather than
    /// snapshotting the whole buffer.
    #[test]
    fn journal_rejects_a_scatter_update_donation_instead_of_snapshotting_whole_state() {
        let builder = Builder::new();
        let state = builder.state_input("pool", TensorType::f32(vec![4, 2]), StateRole::Recurrent);
        let src = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![2, 2]));
        let inv = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![4]));
        let updated = builder.scatter_update(state, src, inv);
        let graph = builder.finish_with_state(updated, &[(state, updated)]);

        // Premise: the commit plan does donate this edge, as `update_base` grants ScatterUpdate.
        let commit =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert_eq!(
            commit.inplace,
            HashMap::from([(updated.id, state.id)]),
            "premise: state_commit_from_plans still donates a proved-dead ScatterUpdate pairing"
        );

        match journal_donated_updates(&graph, &commit) {
            Err(PlanError::UnrecoverableDonation { state_input, op }) => {
                assert_eq!(state_input, state.id);
                assert_eq!(op, "scatter_update");
            }
            other => panic!("expected UnrecoverableDonation, got {other:?}"),
        }
    }

    /// A graph with no donated edge journals to an empty list (the common case: plain explicit copies).
    #[test]
    fn journal_is_empty_with_no_donation() {
        let builder = Builder::new();
        let a = builder.state_input("a", TensorType::f32(vec![1]), StateRole::Recurrent);
        let b = builder.state_input("b", TensorType::f32(vec![1]), StateRole::Recurrent);
        let graph = builder.finish_with_state(a, &[(a, b), (b, a)]);
        let commit =
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap();
        assert!(commit.inplace.is_empty());
        assert_eq!(journal_donated_updates(&graph, &commit).unwrap(), []);
    }

    /// `plan_native_state_transaction` bundles the state-commit plan, the packet plan, and the donation
    /// journal from one validation-bearing graph, so a caller cannot mismatch them.
    #[test]
    fn native_state_transaction_bundles_commit_packet_and_donation_journal() {
        let builder = Builder::new();
        let state = builder.state_input("state", TensorType::f32(vec![6, 3]), StateRole::Recurrent);
        let update = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![1, 3]));
        let updated = builder.dynamic_update_slice(state, update, 2, 0);
        let witness = builder.binary_scalar(
            BinOp::Ge,
            builder.reduce(poot_graph_ir::RedOp::Sum, updated, 1, false),
            Scalar::F32(0.0),
        );
        let graph = crate::test_support::finish_with_state_and_validations(
            builder,
            updated,
            &[(state, updated)],
            &[(poot_graph_ir::ValidationId(7), "sum_nonneg", witness)],
        )
        .unwrap();

        let transaction = plan_native_state_transaction(&graph, Backend::Nvptx).unwrap();
        assert_eq!(
            transaction.commit,
            commit_for_test(&graph, Backend::Nvptx, &default_caps_for(Backend::Nvptx)).unwrap()
        );
        assert_eq!(transaction.packet, plan_validation_packet(&graph).unwrap());
        assert_eq!(
            transaction.donations,
            [DonationFootprint {
                state_root: updated.id,
                state_input: state.id,
                axis: 0,
                shape: vec![1, 3],
            }]
        );
    }

    /// The transaction plan surfaces the journal's rejection, so no `NativeStateTransactionPlan` widens a
    /// ScatterUpdate donation into a whole-state snapshot.
    #[test]
    fn native_state_transaction_rejects_an_unrecoverable_donation() {
        let builder = Builder::new();
        let state = builder.state_input("pool", TensorType::f32(vec![4, 2]), StateRole::Recurrent);
        let src = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![2, 2]));
        let inv = builder.slot(poot_graph_ir::Slot::Activation, TensorType::f32(vec![4]));
        let updated = builder.scatter_update(state, src, inv);
        let witness = builder.binary_scalar(
            BinOp::Ge,
            builder.reduce(poot_graph_ir::RedOp::Sum, updated, 1, false),
            Scalar::F32(0.0),
        );
        let graph = crate::test_support::finish_with_state_and_validations(
            builder,
            updated,
            &[(state, updated)],
            &[(poot_graph_ir::ValidationId(3), "sum_nonneg", witness)],
        )
        .unwrap();

        match plan_native_state_transaction(&graph, Backend::Nvptx) {
            Err(PlanError::UnrecoverableDonation { state_input, op }) => {
                assert_eq!(state_input, state.id);
                assert_eq!(op, "scatter_update");
            }
            other => panic!("expected UnrecoverableDonation, got {other:?}"),
        }
    }
}
