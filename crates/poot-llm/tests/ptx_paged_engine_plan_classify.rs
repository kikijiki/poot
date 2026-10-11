//! Classifies every eqn of the qwen2 family's batched shared-pool decode graph
//! via `plan_eqn` on `Backend::Nvptx` at `n_slots = 2`: every eqn must plan (the planner has no host
//! plan; an unplannable eqn is a typed refusal, Card 626). CPU-only, no GPU or checkpoint. The graph is what the family's batched shared-pool decode trace produces for a batched paged-KV decode entry (Card 549: staged through `compile_staged` and
//! stepped on `Engine<PtxDevice>`).
//!
//! PTX twin of `rocm_batched_decode_plan_classify.rs` (`Backend::AmdGcn`): checks the graph is
//! plannable before using a rented GPU. It traces
//! the qwen2 family's paged decode step directly (no real checkpoint needed)
//! and applies the pre-`compile` pass chain locally (below), so the equations classified are the fused ones `compile` plans.

use poot_executor_parity::dense::{Dense, Family, paged_step};
use poot_graph_plan::{Plan, widen_mismatched_matmul_dtypes};
use poot_llm::driver::block_table::BLOCK_SIZE;
use poot_models::model::{LogitRows, Phase};
use poot_target::Backend;
use poot_tensor::DType;

use poot_graph_plan::passes_without_target as optimize;
use poot_test_util::graph_fixtures::plan_eqn;
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
        Plan::Dispatches(dispatches) => match &dispatches[..] {
            [dispatch] if dispatch.launch().meta.is_empty() => "Dispatch",
            [_] => "DispatchWithMeta",
            _ => "SplitDispatches",
        },
        Plan::Alias(_) => "Alias",
        Plan::View { .. } => "View",
    }
}

#[test]
fn ptx_paged_engine_plan_classify_qwen2_n_slots_2() {
    let n_slots = 2usize;
    // cap/pool_slots mirror a representative paged pool size
    // (`num_blocks = (n_slots*cap).div_ceil(BLOCK_SIZE); pool_slots = num_blocks*BLOCK_SIZE + 1`, the
    // same shape `poot-llm/src/driver/block_table.rs::BLOCK_SIZE`-backed batched decode uses); exact values
    // only change tensor sizes, not which ops plan.
    let cap = 32;
    let num_blocks = (n_slots * cap).div_ceil(BLOCK_SIZE);
    let pool_slots = num_blocks * BLOCK_SIZE + 1;

    let m = qwen25_05b().zeroed_model(DType::BF16);
    let g = m
        .model
        .trace(
            Phase::Decode,
            paged_step(n_slots, 1, cap, pool_slots, LogitRows::Last),
        )
        .unwrap();
    let backend = Backend::Nvptx;
    // The pre-`compile` passes, so the equations classified are the fused ones `compile` plans: the
    // BF16 checkpoint's mixed-dtype matmuls are widened as `compile` widens them.
    let g = widen_mismatched_matmul_dtypes(&optimize(&g).unwrap(), backend).unwrap();
    let mut tally: BTreeMap<&'static str, usize> = BTreeMap::new();
    for eqn in g.eqns() {
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
        "ptx_paged_engine_plan_classify: n_slots={n_slots} total_eqns={} tally={tally:?}",
        g.eqns().len()
    );
}
