//! Classifies every eqn of the qwen2 family's batched shared-pool decode graph
//! via `compile` on `Backend::Nvptx` at `n_slots = 2`: every eqn must plan (the planner has no host
//! plan; an unplannable eqn is a typed refusal, Card 626). CPU-only, no GPU or checkpoint. The graph is what the family's batched shared-pool decode trace produces for a batched paged-KV decode entry (Card 549: staged through `compile_staged` and
//! stepped on `Engine<PtxDevice>`).
//!
//! PTX twin of `rocm_batched_decode_plan_classify.rs` (`Backend::AmdGcn`): checks the graph is
//! plannable before using a rented GPU. It traces
//! the qwen2 family's paged decode step directly (no real checkpoint needed)
//! and compiles it, so the equations classified are the fused ones `compile` plans.

use poot_graph_plan::{CompileLimits, CompileOptions, Plan, Target, compile};
use poot_llm::driver::block_table::BLOCK_SIZE;
use poot_models::model::{LogitRows, StepSpec};
use poot_models::test_support::dense::{Dense, Family, paged_step};
use poot_target::Backend;
use poot_tensor::DType;

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
        Plan::Graph => "Graph",
        Plan::SideOutput { .. } => "SideOutput",
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
            &StepSpec::step(paged_step(n_slots, 1, cap, pool_slots, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let backend = Backend::Nvptx;
    // Through `compile`, the one entry every executor runs: the equations classified are the fused ones it
    // plans, after the weight folds a BF16 checkpoint needs (no pass widens a mixed-dtype matmul).
    let target = Target::Gpu {
        backend,
        caps: poot_test_util::device_caps::default_caps_for(backend),
    };
    let program = compile(&g, &target, &CompileOptions::new(CompileLimits::STANDARD))
        .unwrap_or_else(|e| {
            panic!("the qwen2 batched decode graph must compile on {backend:?}: {e}")
        });
    let mut tally: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut total = 0usize;
    for (_, plan) in program.planned() {
        *tally.entry(plan_tag(plan)).or_insert(0) += 1;
        total += 1;
    }

    eprintln!(
        "ptx_paged_engine_plan_classify: n_slots={n_slots} total_eqns={} tally={tally:?}",
        total
    );
}
