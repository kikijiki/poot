//! CPU-only pre-check (no GPU, no checkpoint): classify every eqn of the qwen2 family's batched paged
//! decode graph via `plan_eqn` at `n_slots = 2`. Every eqn planning
//! (the planner has no host plan: an unplannable eqn is a typed refusal, Card 626) is what makes the
//! executor contract's capture-on-first-step default (Card 548: `Engine::step` records once, replays every step after) safe for this graph
//! on ROCm (see `card209_batched_decode_plan_classify.rs` for qwen3next).
//!
//! Builds the graph with `Model::trace` of the qwen2 family (a paged decode step over a checkpoint of zeros) and applies the pre-`compile` pass chain locally, which classifies the fused equations
//! `compile` plans.

#![cfg(feature = "rocm")]

use poot_executor_parity::dense::{Dense, Family, paged_step, plain};
use poot_graph_plan::{Plan, widen_mismatched_matmul_dtypes};
use poot_llm::driver::block_table::BLOCK_SIZE;
use poot_models::model::{LogitRows, Phase};
use poot_target::AmdArch;
use poot_target::Backend;
use poot_tensor::DType;
use poot_test_util::graph_fixtures::plan_eqn;

use poot_graph_plan::passes_without_target as optimize;
use std::collections::BTreeMap;

/// qwen2.5-0.5b dims (a BF16 checkpoint of zeros: only the traced graph is read).
fn qwen25_05b() -> Dense {
    Dense::new(Family::Qwen2)
        .vocab(151_936)
        .dims(896, 4864, 24)
        .heads(14, 2)
        .head_dim(64)
        .max_positions(32_768)
}

fn plan_tag(p: &Plan) -> &'static str {
    match p {
        Plan::Compute { .. } => "Compute",
        Plan::ComputeMeta { .. } => "ComputeMeta",
        Plan::ComputeChunks(_) => "ComputeChunks",
        Plan::Alias(_) => "Alias",
        Plan::View { .. } => "View",
        Plan::Collective { .. } => "Collective",
    }
}

#[test]
fn card234_rocm_batched_decode_plan_classify_qwen2_n_slots_2() {
    let n_slots = 2usize;
    // cap/pool_slots approximate `batch_engine_loop_rocm_runner`'s computation in
    // `crates/poot-serve/src/batch.rs`; exact values change tensor sizes, not which ops plan.
    let cap = 32;
    let num_blocks = (n_slots * cap).div_ceil(BLOCK_SIZE);
    let pool_slots = num_blocks * BLOCK_SIZE + 1;

    let m = qwen25_05b().zeroed_model(DType::BF16);
    let g = plain(
        m.model
            .trace(
                Phase::Decode,
                paged_step(n_slots, 1, cap, pool_slots, LogitRows::Last),
            )
            .unwrap(),
    );
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    // The pre-`compile` passes, so the equations classified are the fused ones `compile` plans: the
    // BF16 checkpoint's mixed-dtype matmuls are widened as `compile` widens them.
    let g = widen_mismatched_matmul_dtypes(
        &optimize(&g),
        backend,
        &poot_test_util::device_caps::default_caps_for(backend),
    );
    let mut tally: BTreeMap<&'static str, usize> = BTreeMap::new();
    for eqn in &g.eqns {
        let plan = plan_eqn(
            &g,
            eqn,
            backend,
            &poot_test_util::device_caps::default_caps_for(backend),
        )
        .unwrap_or_else(|e| panic!("plan_eqn failed on eqn op={}: {e}", eqn.op.name()));
        *tally.entry(plan_tag(&plan)).or_insert(0) += 1;
    }

    eprintln!(
        "card234_rocm_batched_decode_plan_classify: n_slots={n_slots} total_eqns={} tally={tally:?}",
        g.eqns.len()
    );
}
