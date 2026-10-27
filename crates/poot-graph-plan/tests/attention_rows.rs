//! A long prefill attention through the production planner (POOT-1313): on a device with a display-watchdog
//! budget it plans as dispatches over disjoint runs of rows, each within the budget; on a device without one
//! it stays a single launch. No device: `poot-wgpu-device`'s `gemma_global_attention_prefill_at_8192_tokens_runs_split_and_matches_a_host_reference` runs the same split on hardware.
//!
//! Every row drives `compile` over a traced prefill graph; none builds a `DispatchPlan` by hand.

use poot_graph_ir::{Builder, Graph, OpKind, Slot, TensorType};
use poot_graph_plan::{CompileOptions, DispatchPlan, KernelChoice, Plan, Program, Target, compile};
use poot_kernelgen::{AttentionRows, KernelRequest};
use poot_target::{AmdArch, Backend};
use poot_test_util::device_caps::default_caps_for;

const OPTIONS: CompileOptions = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);

/// One prefill attention over `hq` heads of width `d` and `len` positions, with an additive `[len, len]` mask.
/// Q, K and V are activations, so the constant size gate binds only the mask.
fn prefill(hq: usize, len: usize, d: usize) -> Graph {
    let b = Builder::for_tests();
    let shape = TensorType::f32(vec![1, hq, len, d]);
    let q = b.slot_named(Slot::Activation, "q", shape.clone()).unwrap();
    let k = b.slot_named(Slot::Activation, "k", shape.clone()).unwrap();
    let v = b.slot_named(Slot::Activation, "v", shape).unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, len, len]))
        .unwrap();
    let out = poot_graph_ir::ops::attention_masked(&b, q, k, v, 1, 0.125, mask).unwrap();
    b.finish(out).unwrap()
}

fn compiled(graph: &Graph, backend: Backend) -> Program {
    let target = Target::Gpu {
        backend,
        caps: default_caps_for(backend),
    };
    compile(graph, &target, &OPTIONS).expect("the prefill compiles")
}

/// The plan and the first request of the one attention equation of `program`.
fn the_attention(program: &Program) -> (&DispatchPlan, &KernelRequest) {
    let (eqn, plan) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::Attention(_)))
        .expect("one attention");
    let Plan::Dispatches(dispatches) = plan else {
        panic!("an attention launches: {plan:?}");
    };
    let KernelChoice::Generated(request) = program.kernel_choice(eqn) else {
        panic!(
            "generated attention expected: {:?}",
            program.kernel_choice(eqn)
        );
    };
    (dispatches, request)
}

/// 4 heads of 4096 positions at `dk + dv` 512 is 16384 rows of 2.1e6 each: 3.4e10 against the Strix Halo
/// budget of 1e10, so the wgpu plan is several runs of rows, each run within the budget, together covering
/// every row once. The device without a watchdog budget launches the attention once.
///
/// Mutation: in `attention_rows_per_dispatch`, return `None`; the wgpu plan is one launch.
#[test]
fn a_prefill_over_the_watchdog_budget_plans_runs_of_rows() {
    let (hq, len, d) = (4, 4096, 256);
    let graph = prefill(hq, len, d);
    let rows = hq * len;
    let row_work = (len * 2 * d) as u64;

    let wgpu = compiled(&graph, Backend::SpirvVulkan);
    let budget = default_caps_for(Backend::SpirvVulkan)
        .watchdog_budget
        .expect("the wgpu reference device has a watchdog budget")
        .attention_work;
    let (plan, request) = the_attention(&wgpu);
    let launches = plan.len();
    let KernelRequest::AttentionRows(AttentionRows { first, count, spec }) = request else {
        panic!("a run of rows expected: {request:?}");
    };
    assert_eq!(*first, 0, "the choice names the first run");
    // A run counts row groups: one row each under the streaming schedule, a block of rows under the tiled one.
    let group_rows = spec.rows_per_workgroup();
    let run_rows = *count * group_rows;
    assert!(
        run_rows as u64 * row_work <= budget,
        "a run of {run_rows} rows carries {} against {budget}",
        run_rows as u64 * row_work
    );
    assert_eq!(
        launches,
        spec.row_groups().div_ceil(*count),
        "runs of {count} groups of {group_rows} rows cover {rows} rows once"
    );
    assert!(spec.row_groups() * group_rows >= rows);
    assert!(launches > 1);
    // The flush weight follows the keys a run walks, not its output elements: every full run weighs one
    // submit, so the executor never queues two budgets' worth of attention in one submit.
    for (index, dispatch) in plan.iter().enumerate() {
        let full = index + 1 < launches;
        let work = dispatch.work();
        assert!(
            work <= poot_target::SUBMIT_WORK,
            "run {index} weighs {work}"
        );
        assert!(
            !full || work * 10 >= poot_target::SUBMIT_WORK * 9,
            "full run {index} weighs {work} of {}",
            poot_target::SUBMIT_WORK
        );
    }

    for backend in [Backend::AmdGcn(AmdArch::gfx1151()), Backend::Nvptx] {
        let program = compiled(&graph, backend);
        let (plan, request) = the_attention(&program);
        let launches = plan.len();
        assert_eq!(launches, 1, "{backend:?} has no watchdog budget");
        assert!(
            matches!(request, KernelRequest::Attention(_)),
            "{backend:?}"
        );
    }
}

/// A prefill under the budget is one launch of the whole attention on the watchdog device too: the sliding
/// window of a local layer and a short prompt both stay unsplit.
#[test]
fn a_prefill_within_the_watchdog_budget_stays_one_launch() {
    let program = compiled(&prefill(4, 512, 256), Backend::SpirvVulkan);
    let (plan, request) = the_attention(&program);
    let launches = plan.len();
    assert_eq!(launches, 1);
    assert!(
        matches!(request, KernelRequest::Attention(_)),
        "{request:?}"
    );
}
