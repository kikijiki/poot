//! `ptx-graph-check`: pre-compile every Compute kernel a set of representative graphs need to the
//! on-disk PTX cache, with no CUDA context (R-546-13 as amended by R-592a-2: this traces models, via
//! `poot-models`, so it lives here - in `poot-llm/src/bin/` - rather than in a device crate, which
//! cannot see upward to `poot-models`).
//!
//! Card 549 deleted every device-dispatching check mode this binary used to have (they drove
//! `PtxGraphExecutor`, which no longer exists): what remains is the one GPU-free `--precompile` mode,
//! run locally (nix devshell, with `llc`, no GPU present) to populate the on-disk PTX cache before a
//! pod run, which has `llc` but no CUDA runtime bundled. `--precompile` goes over the shared
//! `poot_graph_plan::compile` (never a second, bespoke per-equation plan walk - R472-001) plus one
//! `poot_codegen::KernelCache`, exactly as `Engine::kernel` compiles a plan-sourced kernel; it proves a
//! plan exists and lowers for Nvptx, not that the lowering is correct (hardware verification is the
//! pod's job).

use std::num::NonZeroUsize;

use poot_graph_ir::{Graph, NoValidations};
use poot_models::model::{KvLayout, LogitRows, Model, Phase, StepShape};
use poot_models::registry::Registry;
use poot_ptx_gpu::precompile_graph;
use poot_target::DeviceCaps;

mod moe;

/// The qwen2 family's fixture checkpoint through the shipped registry: a tiny model whose graphs
/// carry the family's real structure (the precompile reads the graphs, never the weights).
fn qwen2_fixture() -> Box<dyn Model> {
    let registry = Registry::builtin().expect("the shipped registry");
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "qwen2")
        .expect("qwen2 is a shipped family");
    let fixture = (entry.fixture)();
    registry
        .build(&fixture.raw(), &fixture.store)
        .expect("the qwen2 fixture builds")
}

/// `shape`d step of `model`, as the ordinary graph `precompile_graph` takes. A dense family declares no
/// validation output, so none is dropped.
fn trace(model: &dyn Model, phase: Phase, shape: StepShape) -> Graph {
    let g = model.trace(phase, shape).expect("the fixture traces");
    assert!(g.validation_outputs().is_empty());
    let Graph {
        values,
        inputs,
        consts,
        slots,
        eqns,
        output,
        state,
        validations: _,
    } = g;
    Graph {
        values,
        inputs,
        consts,
        slots,
        eqns,
        output,
        validations: NoValidations,
        state,
    }
}

fn step(rows: usize, tokens: usize, capacity: usize, kv: KvLayout) -> StepShape {
    StepShape {
        rows: NonZeroUsize::new(rows).unwrap(),
        tokens: NonZeroUsize::new(tokens).unwrap(),
        capacity: NonZeroUsize::new(capacity).unwrap(),
        kv,
        logits: LogitRows::Last,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args != ["--precompile"] {
        eprintln!("usage: ptx-graph-check --precompile");
        eprintln!(
            "(every other check mode was deleted with PtxGraphExecutor - card 549; this binary is \
             GPU-free precompile only now)"
        );
        std::process::exit(2);
    }
    run_precompile();
}

fn run_precompile() {
    // No live CUDA context by design (this tool runs on an `llc`-only host with no GPU): NVPTX has no
    // confirmed display watchdog or tiled-GEMM miscompile ceiling to measure (card 522), so
    // `ptx_default()` is the real, documented fact for every NVPTX device, not a guessed fixture.
    let caps = DeviceCaps::ptx_default();
    let model = qwen2_fixture();
    let hidden = 16; // the MoE probe's hidden width
    let (tokens, cap) = (5usize, 5usize);
    let mut total = 0;

    // the decode graph (constant shape, one capture) and its paged variant - `compile` applies its own
    // fusion (flash attention included, Card 557), so no caller-side `fuse(cse(..))` or flash tracer
    // variant is needed here any more (R472-001).
    let gm = trace(
        &*model,
        Phase::Decode,
        step(1, 1, cap, KvLayout::Contiguous),
    );
    total += precompile_graph(&gm, &caps).expect("precompile masked");
    let paged = KvLayout::Paged {
        pool_slots: NonZeroUsize::new(cap).unwrap(),
    };
    let gpg = trace(&*model, Phase::Decode, step(1, 1, cap, paged));
    total += precompile_graph(&gpg, &caps).expect("precompile paged masked");

    // the MoE block lowers to PTX: the gate is the stable `top_k_gate` primitive composition, so the
    // comparison primitive, pairwise rank with index ties, exact selected normalization and expert
    // swiglu/matmul mix all need NVPTX plans.
    let (n_exp, k_top, moe_inter) = (8usize, 2usize, 16usize);
    let bm = poot_graph_ir::builder::Builder::new();
    let mx = bm.constant(
        "moe.x",
        poot_graph_ir::types::TensorType::f32(vec![1, cap, hidden]),
    );
    let rw = bm.constant(
        "moe.rw",
        poot_graph_ir::types::TensorType::f32(vec![hidden, n_exp]),
    );
    let win = bm.constant(
        "moe.win",
        poot_graph_ir::types::TensorType::f32(vec![n_exp, hidden, 2 * moe_inter]),
    );
    let wout = bm.constant(
        "moe.wout",
        poot_graph_ir::types::TensorType::f32(vec![n_exp, moe_inter, hidden]),
    );
    let mo = poot_graph_ir::ops::moe(&bm, mx, rw, win, wout, n_exp, k_top, moe_inter);
    let gmoe = bm.finish(mo);
    total += precompile_graph(&gmoe, &caps).expect("precompile moe (Ge gate)");
    // the dense + gather-free sparse MoE decode graphs at the fixed check dims.
    total += precompile_graph(&moe::build_moe(false), &caps).expect("precompile moe dense");
    total += precompile_graph(&moe::build_moe(true), &caps).expect("precompile moe sparse");
    // every tied/non-finite/edge-k routing graph the stable MoE tie-break tests dispatch.
    for case in moe::stable_moe_probe_cases() {
        let bt = poot_graph_ir::builder::Builder::new();
        let scores = bt.constant(
            &case.score_const_name(),
            poot_graph_ir::types::TensorType::f32(vec![1, 5]),
        );
        let rank = poot_graph_ir::ops::stable_descending_rank(&bt, scores);
        let ids = bt.arg_top_k(rank, case.k);
        let mask = poot_graph_ir::ops::top_k_keep_mask(&bt, rank, case.k);
        let gate = poot_graph_ir::ops::top_k_gate(&bt, scores, case.k);
        let ties = bt.concat(1, &[ids, mask, gate]);
        total +=
            precompile_graph(&bt.finish(ties), &caps).expect("precompile stable MoE routing case");
    }

    // the M>1 prefill graph: shared-weight projection tiled GEMMs + batched-weight per-head attention
    // tiled GEMMs run the imported ComputeMeta kernels on NVPTX (no chunking).
    let gp = trace(
        &*model,
        Phase::Prefill,
        step(1, tokens, cap, KvLayout::Contiguous),
    );
    total += precompile_graph(&gp, &caps).expect("precompile prefill");
    // the B>1 batched decode graph: the projection GEMVs run the imported batched gemv ComputeMeta
    // (dims = [B]) on NVPTX.
    let gb = trace(
        &*model,
        Phase::Decode,
        step(3, 1, cap, KvLayout::Contiguous),
    );
    total += precompile_graph(&gb, &caps).expect("precompile batched decode");

    println!("precompiled {total} kernels to the PTX cache");
}
