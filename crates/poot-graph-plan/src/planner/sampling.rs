//! Sampling equation planning (card 551a, R472-007): `RandomUniform` and `SampleToken` lower onto the
//! sampler bodies in `imported::sampling` (Card 447's original consolidation). The stochastic rules
//! dispatch one workgroup of 64 lanes per row (the bodies' own fixed shape; see `finalize`'s
//! `custom_grid` guard in the parent module, which keeps the shared launch postamble from bumping this
//! workgroup size); top-k/top-p/noise stages run inside those bodies (bisection, not a separate dispatch).
//! `RandomUniform` is a plain one-thread-per-output elementwise kernel.
//!
//! `Greedy` is a two-dispatch plan (POOT-1146): a row's `vocab` logits split into [`ARGMAX_CHUNK`]-element
//! chunks, one workgroup per (row, chunk) reduces its chunk to a `(value, index, bad_index)` partial in the
//! equation's scratch, and one workgroup per row reduces the partials to the `[rows, 2]` pair. One
//! 64-lane workgroup scanning a 151,936-wide row is the whole of the old greedy step (about 750 us on
//! gfx1151); the chunks run on every compute unit at once. The partial combine is the sampler's own
//! pick rule (larger value, lower index on a tie), associative and commutative, so the split cannot
//! change the answer.

use super::{ContractionHead, DType, Eqn, ImportedKernel, OpKind, PlanError, Planned, value_ids};
use crate::Capability;
use crate::planner::GraphTables;
use crate::refusal::refusal_at;
use crate::types::{Binding, ScratchDecl, ScratchId, Sink, Source};
use poot_graph_ir::op::SampleRule;
use poot_target::{Backend, BufferStorage, DeviceCaps};

/// Logits one `ArgmaxChunk` workgroup reduces: 16 per lane of its 64. Few enough that a 150K row spreads
/// over about 150 workgroups, many enough that the combine stage stays a handful of steps per lane.
const ARGMAX_CHUNK: usize = 1024;

/// The largest vocabulary whose indices an f32 holds exactly: the partials scratch and the published pair
/// carry each winner's index as an f32 (`2^24` is the last power of two below which every integer is exact).
const ARGMAX_MAX_EXACT_VOCAB: usize = 1 << 24;

/// The greedy pick as two dispatches over one scratch. `ArgmaxChunk` takes the metadata `[vocab, chunk,
/// groups]` on a `groups x rows` grid (rows on Y); `ArgmaxCombine` takes `[groups, vocab]` on a `rows`
/// grid (rows on X). A vocabulary past [`ARGMAX_MAX_EXACT_VOCAB`] is refused with
/// [`Capability::ArgmaxIndexNotExact`] (its indices are not exact f32). A row count or chunk count past the target's grid limit on its axis is refused with
/// [`Capability::DispatchGridAxisLimit`], never launched truncated.
fn greedy(
    g: GraphTables<'_>,
    eqn: &Eqn,
    rows: usize,
    vocab: usize,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    if vocab > ARGMAX_MAX_EXACT_VOCAB {
        return Err(refusal_at(
            g,
            eqn,
            backend,
            Capability::ArgmaxIndexNotExact {
                vocab,
                limit: ARGMAX_MAX_EXACT_VOCAB,
            },
        ));
    }
    // At least one chunk, so an empty vocabulary still publishes its `(0, -1)` pair.
    let groups = vocab.div_ceil(ARGMAX_CHUNK).max(1);
    for (axis, workgroups) in [(0, groups.max(rows)), (1, rows)] {
        let limit = caps.max_grid[axis] as usize;
        if workgroups > limit {
            return Err(refusal_at(
                g,
                eqn,
                backend,
                Capability::DispatchGridAxisLimit {
                    axis,
                    workgroups,
                    limit,
                },
            ));
        }
    }
    let scratch = ScratchId::new(0);
    let chunk = Planned::imported_split_dispatch(
        ImportedKernel::ArgmaxChunk,
        backend,
        vec![vocab as u32, ARGMAX_CHUNK as u32, groups as u32],
        [groups, rows],
    )
    .bound(Binding::new(
        vec![Source::Operand(0)],
        Sink::Scratch(scratch),
    ));
    let combine = Planned::imported_split_dispatch(
        ImportedKernel::ArgmaxCombine,
        backend,
        vec![groups as u32, vocab as u32],
        [rows, 1],
    )
    .bound(Binding::new(vec![Source::Scratch(scratch)], Sink::Out));
    Planned::pipeline(
        vec![chunk, combine],
        vec![ScratchDecl::new(BufferStorage::f32(), rows * groups * 3)],
    )
}

/// Plan one sampling-family equation (`RandomUniform`/`SampleToken`).
#[allow(clippy::too_many_arguments)]
pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    _out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    let ids = value_ids(eqn);
    let planned = match &eqn.op {
        OpKind::RandomUniform { cols } => Planned::imported(
            ImportedKernel::RandomUniform,
            backend,
            Some(vec![*cols as u32]),
            [out_numel as u32, 1, 1],
        )?,
        OpKind::SampleToken { rule } => {
            // out_shape is [.., 2]; rows is every leading-axis element flattened, vocab the logits'
            // last-axis extent (not out_shape's - the output's own last axis is always 2).
            debug_assert_eq!(odt, DType::I32, "SampleToken output is always I32");
            let rows = out_numel / 2;
            let logits_shape = g.aval(ids[0]).shape.clone();
            let vocab = *logits_shape.last().unwrap_or(&0);
            let kernel = match rule {
                SampleRule::Greedy => return greedy(g, eqn, rows, vocab, backend, caps),
                SampleRule::Gumbel => ImportedKernel::SampleGumbelArgmaxBatched,
                SampleRule::GumbelTopK => ImportedKernel::SampleTruncatedGumbelArgmaxBatched,
                SampleRule::GumbelTopKTopP => ImportedKernel::SampleToppGumbelArgmaxBatched,
            };
            Planned::imported(
                kernel,
                backend,
                Some(vec![vocab as u32]),
                [(rows * 64) as u32, 1, 1],
            )?
        }
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | packed_ops!()
        | moe_ops!()
        | elementwise_ops!()
        | matmul_ops!()
        | attention_ops!()
        | scan_ops!()
        | block_top_k_ops!()
        | cast_ops!()
        | movement_ops!() => unreachable!("plan_eqn routes only sampling ops here"),
    };
    Ok(planned)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use poot_graph_ir::{Builder, Graph, TensorType};
    use poot_target::AmdArch;
    use poot_test_util::device_caps::default_caps_for;

    use super::*;
    use crate::{KernelChoice, Plan, Planned, Target};

    fn greedy_graph(rows: usize, vocab: usize) -> Graph {
        let b = Builder::for_tests();
        let logits = b
            .constant("logits", TensorType::f32(vec![rows, vocab]))
            .unwrap();
        let out = b
            .sample_token(SampleRule::Greedy, logits, None, None, None)
            .unwrap();
        b.finish(out).unwrap()
    }

    fn plan(graph: &Graph, backend: Backend) -> Planned {
        let eqn = graph
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::SampleToken { .. }))
            .expect("fixture has a SampleToken equation");
        crate::planner::plan_eqn_compiled(
            graph,
            eqn,
            &Target::Gpu {
                backend,
                caps: default_caps_for(backend),
            },
            &HashMap::new(),
            &crate::test_support::roomy_body_limits(),
        )
        .unwrap_or_else(|e| panic!("{backend:?}: {e}"))
    }

    /// POOT-1146: a greedy pick over `[rows, vocab]` logits plans as the two dispatches the module
    /// documents, on every backend: the chunk stage reads operand 0 and writes the scratch of `rows *
    /// groups` `(value, index, bad_index)` triples on a `groups x rows` grid; the combine stage reads
    /// that scratch and writes the output on a `rows` grid. Rows cover a single chunk, a partial last
    /// chunk, an exact multiple and the production vocabulary.
    ///
    /// Mutation: plan `Greedy` as one dispatch again, or make the group count `vocab / ARGMAX_CHUNK`
    /// (dropping the partial last chunk); the dispatch count or the metadata assertion fails.
    #[test]
    fn greedy_plans_a_chunk_stage_and_a_combine_stage() {
        for backend in [
            Backend::Nvptx,
            Backend::SpirvVulkan,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            for (rows, vocab, groups) in [
                (1usize, 1000usize, 1usize),
                (3, 5000, 5),
                (2, 4096, 4),
                (1, 151_936, 149),
            ] {
                let planned = plan(&greedy_graph(rows, vocab), backend);
                let dispatches = planned.plan.dispatches();
                assert_eq!(dispatches.len(), 2, "{backend:?} {rows}x{vocab}");
                let [chunk, combine] = dispatches else {
                    unreachable!()
                };

                assert!(matches!(
                    chunk.choice(),
                    KernelChoice::Imported {
                        kernel: ImportedKernel::ArgmaxChunk,
                        ..
                    }
                ));
                assert_eq!(
                    chunk.launch().meta,
                    [vocab as u32, ARGMAX_CHUNK as u32, groups as u32],
                    "{backend:?} {rows}x{vocab}: chunk metadata"
                );
                assert_eq!(
                    chunk.launch().grid,
                    [groups as u32 * 64, rows as u32, 1],
                    "{backend:?} {rows}x{vocab}: one workgroup per (row, chunk)"
                );
                assert_eq!(
                    (chunk.binding().reads(), chunk.binding().write()),
                    (&[Source::Operand(0)][..], Sink::Scratch(ScratchId::new(0)))
                );

                assert!(matches!(
                    combine.choice(),
                    KernelChoice::Imported {
                        kernel: ImportedKernel::ArgmaxCombine,
                        ..
                    }
                ));
                assert_eq!(
                    combine.launch().meta,
                    [groups as u32, vocab as u32],
                    "{backend:?} {rows}x{vocab}: combine metadata"
                );
                assert_eq!(
                    combine.launch().grid,
                    [rows as u32 * 64, 1, 1],
                    "{backend:?} {rows}x{vocab}: one workgroup per row"
                );
                assert_eq!(
                    (combine.binding().reads(), combine.binding().write()),
                    (&[Source::Scratch(ScratchId::new(0))][..], Sink::Out)
                );

                let scratch = match &planned.plan {
                    Plan::Dispatches(dispatches) => dispatches.scratch(),
                    other => panic!("expected dispatches, got {other:?}"),
                };
                assert_eq!(scratch.len(), 1);
                assert_eq!(
                    (scratch[0].storage(), scratch[0].elems()),
                    (BufferStorage::f32(), rows * groups * 3),
                    "{backend:?} {rows}x{vocab}: partials scratch"
                );
            }
        }
    }

    /// POOT-1222 SC-003: a vocabulary of `2^24 + 1` is refused for the f32 argmax route with the typed
    /// `ArgmaxIndexNotExact` naming the vocabulary and the limit, on every backend; `2^24` (every index
    /// exact) plans.
    ///
    /// Mutation: delete the `ARGMAX_MAX_EXACT_VOCAB` check in `greedy`; the `2^24 + 1` row plans and the
    /// refusal assertion fails.
    #[test]
    fn greedy_refuses_a_vocabulary_whose_indices_are_not_exact_f32() {
        for backend in [
            Backend::Nvptx,
            Backend::SpirvVulkan,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let plan_vocab = |vocab: usize| {
                let graph = greedy_graph(1, vocab);
                let eqn = graph
                    .eqns()
                    .iter()
                    .find(|e| matches!(e.op, OpKind::SampleToken { .. }))
                    .unwrap();
                let mut caps = default_caps_for(backend);
                caps.max_grid = [u32::MAX; 3];
                crate::planner::plan_eqn_compiled(
                    &graph,
                    eqn,
                    &Target::Gpu { backend, caps },
                    &HashMap::new(),
                    &crate::test_support::roomy_body_limits(),
                )
            };
            plan_vocab(1 << 24).unwrap_or_else(|e| panic!("{backend:?}: 2^24 plans: {e}"));
            match plan_vocab((1 << 24) + 1) {
                Err(crate::PlanError::Refused(refusal)) => assert_eq!(
                    refusal.missing,
                    Capability::ArgmaxIndexNotExact {
                        vocab: (1 << 24) + 1,
                        limit: 1 << 24
                    },
                    "{backend:?}"
                ),
                other => panic!(
                    "{backend:?}: expected a refusal, got {:?}",
                    other.map(|p| p.choice)
                ),
            }
        }
    }

    /// POOT-1146: a greedy pick whose rows exceed the target's Y workgroup limit is refused with the typed
    /// `DispatchGridAxisLimit` naming the axis, the count and the limit, before any dispatch; a row count
    /// at the limit plans. A narrow limit on X (the combine stage's rows, the chunk stage's groups) is
    /// refused the same way.
    ///
    /// Mutation: delete the axis loop in `greedy`; the over-limit rows plan and the refusal assertions fail.
    #[test]
    fn greedy_refuses_a_grid_past_the_target_limit() {
        let backend = Backend::SpirvVulkan;
        let plan_with = |grid: [u32; 3], rows: usize, vocab: usize| {
            let graph = greedy_graph(rows, vocab);
            let eqn = graph
                .eqns()
                .iter()
                .find(|e| matches!(e.op, OpKind::SampleToken { .. }))
                .unwrap();
            let mut caps = default_caps_for(backend);
            caps.max_grid = grid;
            crate::planner::plan_eqn_compiled(
                &graph,
                eqn,
                &Target::Gpu { backend, caps },
                &HashMap::new(),
                &crate::test_support::roomy_body_limits(),
            )
        };
        let refused = |result: Result<Planned, crate::PlanError>| match result {
            Err(crate::PlanError::Refused(refusal)) => refusal.missing,
            other => panic!("expected a refusal, got {:?}", other.map(|p| p.choice)),
        };
        plan_with([65_535, 4, 65_535], 4, 5000).expect("rows at the Y limit plan");
        assert!(matches!(
            refused(plan_with([65_535, 4, 65_535], 5, 5000)),
            Capability::DispatchGridAxisLimit {
                axis: 1,
                workgroups: 5,
                limit: 4
            }
        ));
        plan_with([5, 65_535, 65_535], 5, 5000).expect("groups and rows at the X limit plan");
        assert!(matches!(
            refused(plan_with([4, 65_535, 65_535], 1, 5000)),
            Capability::DispatchGridAxisLimit {
                axis: 0,
                workgroups: 5,
                limit: 4
            }
        ));
    }
}
