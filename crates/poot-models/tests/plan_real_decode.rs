//! The planner on real traced decode graphs: the exact dispatch pins of the Qwen2.5-0.5B and Qwen3-0.6B
//! configs, strided-view promotion parity across backends, and name independence of planning. They trace
//! models through `Model::trace`, so they sit above poot-graph-plan (POOT-1130); its own unit tests plan
//! hand-built graphs.

use poot_graph_ir::op::OpKind;
use poot_graph_ir::test_support::graph_fixtures::derive;
use poot_graph_ir::{Graph, Operand, Storage, ValueId};
use poot_graph_plan::passes_without_target;
use poot_graph_plan::{
    CompileLimits, CompileOptions, Plan, Program, Target, compile, compute_views, plan_eqn_views,
};
use poot_target::{AmdArch, Backend};
use poot_tensor::DType as GDt;
use poot_test_util::device_caps::default_caps_for;

/// `g` compiled for `backend` through the full pipeline.
fn compiled(g: &Graph, backend: Backend) -> Program {
    let target = Target::Gpu {
        backend,
        caps: default_caps_for(backend),
    };
    let options = CompileOptions::new(CompileLimits::STANDARD);
    compile(g, &target, &options).expect("the graph compiles")
}

/// Dispatch count with the planner's strided-view promotion applied: a promoted eqn counts as 0
/// dispatches, like `Reshape`.
fn dispatch_count_planned(g: &Graph, backend: Backend) -> usize {
    let views = compute_views(g, backend);
    g.eqns()
        .iter()
        .filter(|e| !matches!(e.op, OpKind::Reshape { .. }) && !views.contains_key(&e.out))
        .count()
}

/// The graph `Model::trace` gives for one step of a real checkpoint's dimensions (zero weights: only
/// the structure is read), as an ordinary graph.
fn dense_graph(
    dense: poot_models::test_support::dense::Dense,
    tokens: usize,
    cap: usize,
    logits: poot_models::model::LogitRows,
) -> Graph {
    use poot_models::test_support::dense::step;
    dense
        .zeroed_model(GDt::BF16)
        .model
        .trace(
            &poot_models::model::StepSpec::step(step(1, tokens, cap, logits)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .expect("the family traces this step")
}

/// Qwen2.5-0.5B's dimensions (biased projections, no Q/K norm).
fn qwen2_0_5b() -> poot_models::test_support::dense::Dense {
    use poot_models::test_support::dense::{Dense, Family};
    Dense::new(Family::Qwen2)
        .vocab(151936)
        .dims(896, 4864, 24)
        .heads(14, 2)
        .head_dim(64)
        .max_positions(32768)
}

/// Qwen3-0.6B's dimensions (no bias, per-head Q/K norm, explicit head dimension).
fn qwen3_0_6b() -> poot_models::test_support::dense::Dense {
    use poot_models::test_support::dense::{Dense, Family};
    Dense::new(Family::Qwen3)
        .vocab(151936)
        .dims(1024, 3072, 28)
        .heads(16, 8)
        .head_dim(128)
        .max_positions(40960)
}

/// One decode token of a real config over `cap` positions.
fn real_decode_graph(dense: poot_models::test_support::dense::Dense, cap: usize) -> Graph {
    dense_graph(dense, 1, cap, poot_models::model::LogitRows::Last)
}

/// Dispatches per token of the 24-layer Qwen2.5-0.5B real-config decode graph (the legacy tracer planned 415).
/// POOT-725 made the `Cast(Slot::Pos -> F32)` of the position a kernel (368); POOT-1108 folds it into the
/// mask region that reads it, so the launch is gone again. POOT-1150: +3, the hidden pad column of the
/// padded KV stride (one iota-built column and its concat, shared by every layer's mask; capacity 64 camps).
/// POOT-1209: +1, the `[rows]` view of the position every layer's live attention bound reads.
/// POOT-1229: the pad column is the legalizer's, on a target with DRAM channel data only (SPIR-V here, the Strix
/// Halo RADV device), so a target without channel data (PTX) runs the 368 of the exact capacity.
/// POOT-1238: -1 on both, the mask region's launch: the attention states its causal mask by the position, so no
/// mask is built, and -3 more with channel data, the pad column of that mask is not needed either.
const QWEN2_0_5B_DISPATCHES: Dispatches = Dispatches {
    with_channels: 367,
    without: 367,
};

/// Dispatches per token of the 28-layer Qwen3-0.6B real-config decode graph (the legacy tracer planned 539).
/// POOT-725: 484; POOT-1108: 483, the position cast folded as for Qwen2.5 above; POOT-1150: 486, the hidden
/// pad column as above. POOT-1209: 487, the `[rows]` view of the position as above. POOT-1229: 484 on a target
/// without channel data, as above. POOT-1238: 483 on both, the mask launch and the pad column gone as above.
const QWEN3_0_6B_DISPATCHES: Dispatches = Dispatches {
    with_channels: 483,
    without: 483,
};

/// The dispatch pin of a decode graph by whether the target carries DRAM channel data (the legalizer pads the
/// camping KV caches there and nowhere else).
struct Dispatches {
    with_channels: usize,
    without: usize,
}

impl Dispatches {
    fn on(&self, backend: Backend) -> usize {
        match backend {
            Backend::SpirvVulkan => self.with_channels,
            Backend::Nvptx | Backend::AmdGcn(_) => self.without,
        }
    }
}

/// Dispatches per token of the same Qwen2.5-0.5B config traced over a paged KV pool: the contiguous decode's
/// minus the one the in-place write does not need. Before attention read the pools in place this was 96 more
/// (one gather and one transpose per layer and per K and V).
const QWEN2_0_5B_PAGED_DISPATCHES: usize = 366;

/// Dispatches `program` runs per step: every planned equation but the free ones (an alias or a strided view).
fn dispatches(program: &Program) -> usize {
    program
        .planned()
        .filter(|(_, plan)| !matches!(plan, Plan::Alias(_) | Plan::View { .. }))
        .count()
}

#[test]
fn dispatch_count_planned_regression_on_real_qwen2_decode_configs() {
    // Pins the dispatches per token of the real Qwen2.5-0.5B / Qwen3-0.6B decode graphs (`Model::trace`,
    // one token) through `compile`, the production pipeline, on both planner families (SPIR-V and PTX).
    // The pins are exact: a bound above the real count would still pass with a fusion pass deleted, so any
    // change in either direction fails here and the pin is updated deliberately with the change that moved
    // it. Weights-free: the count depends only on graph structure.
    let cap = 64usize;
    for (name, dense, pin) in [
        ("qwen2_0_5b", qwen2_0_5b(), QWEN2_0_5B_DISPATCHES),
        ("qwen3_0_6b", qwen3_0_6b(), QWEN3_0_6B_DISPATCHES),
    ] {
        let g = real_decode_graph(dense, cap);
        for backend in [Backend::SpirvVulkan, Backend::Nvptx] {
            let program = compiled(&g, backend);
            assert_eq!(
                dispatches(&program),
                pin.on(backend),
                "{name} {backend:?} dispatches per token moved off its pin"
            );
            // Every norm scale and bias is read narrow by the region that uses it: no cast of a weight runs.
            let weight_casts = program
                .graph()
                .eqns()
                .iter()
                .filter(|eqn| {
                    matches!(eqn.op, OpKind::Cast { .. })
                        && matches!(
                            eqn.inputs.as_slice(),
                            [Operand::Value(src)] if program.graph().meta(*src).storage == Storage::Const
                        )
                })
                .count();
            assert_eq!(
                weight_casts, 0,
                "{name} {backend:?}: a weight cast still dispatches"
            );
        }
    }
}

/// POOT-1149: a paged decode of the real Qwen2.5-0.5B config reads each layer's K and V pools in place. Every
/// layer's attention is one `Paged` attention over the scatter-updated pools and the read map, so the plan holds
/// no gather and no transpose of a materialized `[rows, hkv, capacity, d]` view: the 96 per-step gather and
/// transpose dispatches of the gathered view are gone and the step dispatches no more than the contiguous one.
///
/// Mutation: drop the `peel_paged_kv` call in `match_attention`; the attentions stay `Single` over a gathered
/// view, the gather and transpose dispatches return, and the count rows go red.
#[test]
fn a_paged_decode_of_the_real_qwen2_config_dispatches_no_gather_or_transpose_of_the_pools() {
    use poot_graph_ir::KeyLayout;
    use poot_models::model::{LogitRows, StepSpec};
    use poot_models::test_support::dense::paged_step;

    let (cap, layers) = (64usize, 24usize);
    let paged = qwen2_0_5b()
        .zeroed_model(GDt::BF16)
        .model
        .trace(
            &StepSpec::step(paged_step(1, 1, cap, 4 * cap, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .expect("the family traces a paged decode step");
    for backend in [Backend::SpirvVulkan, Backend::Nvptx] {
        let program = compiled(&paged, backend);
        let graph = program.graph();
        let attentions: Vec<_> = graph
            .eqns()
            .iter()
            .filter_map(|eqn| match &eqn.op {
                OpKind::Attention(attrs) => Some(attrs.key),
                _ => None,
            })
            .collect();
        assert_eq!(
            attentions,
            vec![KeyLayout::Paged; layers],
            "{backend:?}: every layer's attention reads its pools in place"
        );
        let written: std::collections::HashSet<ValueId> = graph
            .eqns()
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::ScatterUpdate))
            .map(|eqn| eqn.out)
            .collect();
        assert_eq!(
            written.len(),
            2 * layers,
            "{backend:?}: one pool write per K and V"
        );
        let reshaped_pool = graph.eqns().iter().find(|eqn| {
            matches!(eqn.op, OpKind::Gather { .. } | OpKind::Transpose { .. })
                && eqn.inputs.iter().any(
                    |operand| matches!(operand, Operand::Value(value) if written.contains(value)),
                )
        });
        assert!(
            reshaped_pool.is_none(),
            "{backend:?}: the plan still gathers or transposes a written pool: {reshaped_pool:?}"
        );
        assert_eq!(
            dispatches(&program),
            QWEN2_0_5B_PAGED_DISPATCHES,
            "{backend:?}: a paged decode's dispatches moved off its pin"
        );
    }
}

/// The threads every launch of an equation donated in place launches and the watchdog weight the planner gives
/// them, in equation order: the state writes of a step (the contiguous K/V appends of a decode).
fn in_place_write_launches(program: &Program) -> Vec<(u128, u64)> {
    let donated = &program.state_commit().inplace;
    program
        .graph()
        .eqns()
        .iter()
        .zip(program.plans())
        .filter(|(eqn, _)| donated.contains_key(&eqn.out))
        .map(|(_, plan)| {
            let threads = plan
                .dispatches()
                .iter()
                .map(|dispatch| {
                    dispatch
                        .launch()
                        .grid
                        .iter()
                        .map(|&extent| u128::from(extent))
                        .product::<u128>()
                })
                .sum();
            let work = plan
                .dispatches()
                .iter()
                .map(|dispatch| dispatch.work())
                .sum();
            (threads, work)
        })
        .collect()
}

/// POOT-1389: a decode step's state writes cost what they write, not what the state holds. Every K/V append of
/// the real Qwen3-0.6B decode graph is donated to its cache and launches the same threads at a 256- and an
/// 8320-position cache (the capacity A/B POOT-1388 timed at 10.2 vs 28.9 ms per token on ROCm), on every
/// planner family, and no more threads than the token's own K/V row. The watchdog weight follows: an append
/// weighs the threads it launches, not the cache it lands in.
///
/// Mutation: plan a donated `DynamicUpdateSlice` with the copy form (in `plan_graph`'s update phase, add
/// `&& eqn.out != eqn.out` to the `state_commit.inplace` test); each write launches one thread per cache element
/// and the row goes red with "SpirvVulkan: a KV append's launch depends on the cache capacity" (263168 threads
/// per append at capacity 256, 8520704 at 8320). Dropping `in_place_work` in `plan_eqn_views_impl` (the weight
/// falls back to the output element count) fails the weight row the same way.
#[test]
fn a_decode_steps_kv_appends_launch_independently_of_the_cache_capacity() {
    let (layers, kv_row) = (28usize, 8 * 128);
    for backend in [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ] {
        let [small, large] = [256usize, 8320].map(|cap| {
            in_place_write_launches(&compiled(&real_decode_graph(qwen3_0_6b(), cap), backend))
        });
        assert_eq!(
            small.len(),
            2 * layers,
            "{backend:?}: one donated K and V append per layer"
        );
        assert_eq!(
            small, large,
            "{backend:?}: a KV append's launch depends on the cache capacity"
        );
        for &(threads, work) in &small {
            assert!(
                threads < 2 * kv_row as u128,
                "{backend:?}: an append launches {threads} threads for a {kv_row}-element row"
            );
            assert_eq!(
                u128::from(work),
                threads,
                "{backend:?}: an append weighs what it launches"
            );
        }
    }
}

/// A copy of `g` with every renamable named value renamed by `rename(value_index, old_name)`. Ids, shapes,
/// storage and equations are untouched, so only the strings differ. A mapped weight and a keyed slot have no
/// name to vary: their spelling is their identity.
fn rename_values(g: &Graph, rename: impl Fn(usize, &str) -> String) -> Graph {
    derive(g, |draft| {
        for (index, value) in draft.values_mut().iter_mut().enumerate() {
            if let Some(name) = value.name().map(std::borrow::Cow::into_owned) {
                let _typed_identity_keeps_its_spelling = value.rename(rename(index, &name));
            }
        }
    })
    .unwrap()
}

/// Everything the per-equation planner decides for `g` on `backend`, as comparable text: the strided
/// views, then the plan (or typed refusal) of every equation. `Plan` carries kernel bodies and cache
/// keys, so a different body, key, grid metadata or refusal shows up as a different string.
fn planned_graph_summary(g: &Graph, backend: Backend) -> Vec<String> {
    let views = compute_views(g, backend);
    let mut view_ids: Vec<ValueId> = views.keys().copied().collect();
    view_ids.sort();
    let mut summary: Vec<String> = view_ids
        .into_iter()
        .map(|id| format!("view {id}: {:?}", views[&id]))
        .collect();
    for eqn in g.eqns() {
        let plan = plan_eqn_views(
            g,
            eqn,
            backend,
            &views,
            &default_caps_for(backend),
            &poot_graph_plan::test_support::roomy_body_limits(),
        );
        summary.push(format!("{}: {plan:?}", eqn.out));
    }
    summary
}

/// SC-002 (card 520a): planning is a function of the graph's structure, never of a name. The same
/// dense decode and prefill graphs (raw and through `optimize`) are planned on every planned backend
/// under their tracer names and under names of other model families and role words; every equation
/// must plan identically. Mutation observed red: the per-equation planner refuses (or plans
/// differently) when an operand name contains "qwen".
#[test]
fn planning_a_graph_does_not_depend_on_its_value_names() {
    use poot_models::model::LogitRows;
    use poot_models::test_support::dense::{Dense, Family};

    let mut graphs = Vec::new();
    for family in [Family::Qwen2, Family::Qwen3] {
        let dense = Dense::new(family)
            .vocab(32)
            .dims(16, 32, 2)
            .heads(4, 2)
            .head_dim(4)
            .max_positions(16);
        let decode = dense_graph(dense.clone(), 1, 16, LogitRows::Last);
        let prefill = dense_graph(dense, 8, 16, LogitRows::Last);
        graphs.push(passes_without_target(&decode).unwrap());
        graphs.push(decode);
        graphs.push(passes_without_target(&prefill).unwrap());
        graphs.push(prefill);
    }

    type Naming = (&'static str, fn(usize, &str) -> String);
    let namings: [Naming; 4] = [
        ("qwen", |index, _| format!("qwen2.layers.{index}.weight")),
        ("deepseek", |index, _| {
            format!("deepseek.blocks.{index}.expert_{index}")
        }),
        ("role words", |index, _| {
            format!("layer_idx.{index}.tensor_name.expert_role")
        }),
        ("opaque", |index, _| format!("v{index}")),
    ];
    let backends = [
        Backend::SpirvVulkan,
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
    ];

    for (graph_index, graph) in graphs.iter().enumerate() {
        assert!(
            graph.values().iter().any(|value| value.name().is_some()),
            "graph {graph_index} has no named value, so renaming would prove nothing"
        );
        for backend in backends {
            let original = planned_graph_summary(graph, backend);
            assert!(
                original.iter().filter(|line| line.contains("Ok(")).count()
                    > graph.eqns().len() / 2,
                "graph {graph_index} on {backend:?} must mostly plan, or the comparison is vacuous"
            );
            for (family, rename) in namings {
                let renamed = planned_graph_summary(&rename_values(graph, rename), backend);
                assert_eq!(
                    renamed.len(),
                    original.len(),
                    "graph {graph_index} on {backend:?} under {family} names: entry count"
                );
                // Bodies are large, so name the first differing equation and clip both sides.
                if let Some((was, now)) = original.iter().zip(&renamed).find(|(a, b)| a != b) {
                    let clip = |text: &str| text.chars().take(160).collect::<String>();
                    panic!(
                        "graph {graph_index} on {backend:?} planned differently under {family} \
                         names:\n  tracer names: {}\n  {family} names: {}",
                        clip(was),
                        clip(now)
                    );
                }
            }
        }
    }
}

#[test]
fn amdgcn_view_promotion_matches_spirv_on_real_qwen2_decode_configs() {
    // Card 137: `compute_views`'s gate is `SpirvVulkan | AmdGcn(_)` (spec 132 phase 1). Pins that ROCm
    // agrees exactly with wgpu on the real Qwen2.5-0.5B and Qwen3-0.6B decode graphs (same fixtures as
    // `dispatch_count_planned_regression_on_real_qwen2_decode_configs` above): identical promoted
    // value-id set and identical composed Layout (strides+offset) per promoted value. Also asserts the
    // planned dispatch counts agree between backends and are strictly below the plan-independent
    // `transform::dispatch_count`, so ROCm gets the same free-view win as wgpu.
    //
    // Uses the raw (pre-`transform::optimize`) traced graph: optimize's fusion passes absorb every
    // Transpose/Slice/Broadcast before `compute_views` sees them (the fully optimized graph
    // promotes 0 views on both configs), so only the raw graph, which keeps the movement ops as distinct
    // eqns, exercises the promotion path.
    use poot_graph_ir::analysis::dispatch_count;
    let cap = 64usize;
    let amd = Backend::AmdGcn(AmdArch::gfx1151());

    for (name, cfg) in [("qwen2_0_5b", qwen2_0_5b()), ("qwen3_0_6b", qwen3_0_6b())] {
        let g = real_decode_graph(cfg, cap);

        let views_spirv = compute_views(&g, Backend::SpirvVulkan);
        let views_amd = compute_views(&g, amd);

        let mut spirv_ids: Vec<ValueId> = views_spirv.keys().copied().collect();
        let mut amd_ids: Vec<ValueId> = views_amd.keys().copied().collect();
        spirv_ids.sort();
        amd_ids.sort();
        assert_eq!(
            spirv_ids, amd_ids,
            "{name}: AmdGcn must promote the IDENTICAL value-id set as SpirvVulkan"
        );
        assert!(
            !spirv_ids.is_empty(),
            "{name}: this real decode graph is known to have promotable movement ops on the raw \
             (pre-fusion) trace - an empty set would mean the test fixture stopped exercising the \
             promotion path"
        );
        for id in &spirv_ids {
            let l_spirv = &views_spirv[id];
            let l_amd = &views_amd[id];
            assert_eq!(
                l_spirv.strides, l_amd.strides,
                "{name}: value {id} strides diverge between SpirvVulkan and AmdGcn"
            );
            assert_eq!(
                l_spirv.offset, l_amd.offset,
                "{name}: value {id} offset diverges between SpirvVulkan and AmdGcn"
            );
        }

        let n_spirv = dispatch_count_planned(&g, Backend::SpirvVulkan);
        let n_amd = dispatch_count_planned(&g, amd);
        assert_eq!(
            n_amd, n_spirv,
            "{name}: dispatch_count_planned must agree between AmdGcn and SpirvVulkan"
        );
        let n_unpromoted = dispatch_count(&g);
        assert!(
            n_amd < n_unpromoted,
            "{name}: AmdGcn's planned dispatch count ({n_amd}) must be strictly below the \
             unpromoted dispatch_count ({n_unpromoted}) - else the ROCm gate relaxation bought \
             nothing on this graph"
        );
    }
}

/// Card 549 SC-003 (R-546-9): the same fixture plans the same number of views on Nvptx as on
/// AmdGcn, now that `poot_codegen::capability(Nvptx).strided_views` is `true` - every
/// `is_strided_capable` kernel body PTX dispatches is the same backend-neutral `poot-kernelgen`
/// body wgpu/ROCm already read strided operands through, so there was never a codegen reason for
/// PTX alone to pay the extra materialize copy. GPU-free (planning only). Mirrors
/// `amdgcn_view_promotion_matches_spirv_on_real_qwen2_decode_configs` above, against Nvptx
/// instead of SpirvVulkan.
///
/// Mutation: restore `strided_views: false` for `Target::Nvptx` in
/// `poot-codegen/src/lib.rs::capability`; `compute_views(&g, Nvptx)` goes back to returning an
/// empty map (its `if !codegen_capability(backend).strided_views { return views; }` early-out),
/// so `nvptx_ids` is empty while `amd_ids` is not, and this row goes red.
#[test]
fn nvptx_view_promotion_matches_amdgcn_on_real_qwen2_decode_configs() {
    let cap = 64usize;
    let amd = Backend::AmdGcn(AmdArch::gfx1151());

    for (name, cfg) in [("qwen2_0_5b", qwen2_0_5b()), ("qwen3_0_6b", qwen3_0_6b())] {
        let g = real_decode_graph(cfg, cap);

        let views_nvptx = compute_views(&g, Backend::Nvptx);
        let views_amd = compute_views(&g, amd);

        let mut nvptx_ids: Vec<ValueId> = views_nvptx.keys().copied().collect();
        let mut amd_ids: Vec<ValueId> = views_amd.keys().copied().collect();
        nvptx_ids.sort();
        amd_ids.sort();
        assert_eq!(
            nvptx_ids, amd_ids,
            "{name}: Nvptx must promote the IDENTICAL value-id set as AmdGcn"
        );
        assert!(
            !nvptx_ids.is_empty(),
            "{name}: this real decode graph is known to have promotable movement ops on the raw \
             (pre-fusion) trace - an empty set would mean Nvptx fell back to the pre-549 \
             materialize-everything behavior"
        );
        for id in &nvptx_ids {
            let l_nvptx = &views_nvptx[id];
            let l_amd = &views_amd[id];
            assert_eq!(
                l_nvptx.strides, l_amd.strides,
                "{name}: value {id} strides diverge between Nvptx and AmdGcn"
            );
            assert_eq!(
                l_nvptx.offset, l_amd.offset,
                "{name}: value {id} offset diverges between Nvptx and AmdGcn"
            );
        }

        let n_nvptx = dispatch_count_planned(&g, Backend::Nvptx);
        let n_amd = dispatch_count_planned(&g, amd);
        assert_eq!(
            n_nvptx, n_amd,
            "{name}: dispatch_count_planned must agree between Nvptx and AmdGcn"
        );
    }
}

/// Card 549 SC-004 (F1): the strided-view fixture plans the same copy
/// (dispatch) count on Nvptx as on AmdGcn - a GPU-free plan-level restatement of SC-004's acceptance
/// text ("the strided-view fixture runs on PTX without the extra copy ... the counter reads one more
/// copy per view") as a planning-time assertion instead of a pod-measured native copy count: every
/// view the planner promotes (`Plan::View`, no dispatch) is one fewer
/// materialize-copy dispatch than an unpromoted plan would need, so an equal promoted-dispatch count
/// between backends is exactly "PTX pays no extra copy for the views it shares with ROCm". Separate
/// from `nvptx_view_promotion_matches_amdgcn_on_real_qwen2_decode_configs` above (which checks the
/// promoted value-id SET and each one's Layout): this row isolates the one property SC-004 itself
/// names, with its own mutation record.
///
/// Mutation: restore `strided_views: false` for `Target::Nvptx` in
/// `poot-codegen/src/lib.rs::capability`. `compute_views(&g, Nvptx)` goes back to its early-out empty
/// map while `compute_views(&g, amd)` still promotes every view, so every one of those views costs
/// Nvptx one extra materialize-copy dispatch it does not pay today:
/// `dispatch_count_planned(&g, Nvptx)` reads higher than `dispatch_count_planned(&g, amd)` by exactly
/// the view count, and this row goes red. Reverted: GREEN.
#[test]
fn sc004_nvptx_strided_view_fixture_pays_no_extra_copy_vs_amdgcn() {
    let cap = 64usize;
    let amd = Backend::AmdGcn(AmdArch::gfx1151());

    for (name, cfg) in [("qwen2_0_5b", qwen2_0_5b()), ("qwen3_0_6b", qwen3_0_6b())] {
        let g = real_decode_graph(cfg, cap);

        let views_nvptx = compute_views(&g, Backend::Nvptx);
        assert!(
            !views_nvptx.is_empty(),
            "{name}: this fixture is known to have promotable views on Nvptx - an empty set means \
             the mutation (or a real regression) already dropped Nvptx's promotion, which is exactly \
             the failure this row exists to catch"
        );

        let n_nvptx = dispatch_count_planned(&g, Backend::Nvptx);
        let n_amd = dispatch_count_planned(&g, amd);
        assert_eq!(
            n_nvptx,
            n_amd,
            "{name}: Nvptx's planned copy (dispatch) count must equal AmdGcn's - a gap here is \
             exactly {} extra materialize-copy dispatch(es) PTX would pay per decode that ROCm does \
             not",
            views_nvptx.len()
        );
    }
}
