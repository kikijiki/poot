//! The compile pipeline on routed-expert model traces: every MoE family's prefill and decode runs the full
//! pass list on the ROCm, PTX and wgpu presets, each program computes what the traced graph computes
//! (ADR-0101 tier 2), and the routers plan a tiled schedule. They trace models through the registry, so they
//! sit above poot-graph-plan (POOT-1130); its own `compile` unit tests build their graphs by hand.

use poot_graph_ir::OpKind;
use poot_graph_plan::{
    CompileLimits, CompileOptions, KernelChoice, Program, Target, compile,
    multi_device::topology::DeviceId, routed_expert_sites, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::TargetSet, staged::compile_staged,
};
use poot_kernelgen::{ContractionSpec, KernelRequest};
use poot_models::model::StepSpec;
use poot_target::{AmdArch, Backend, DeviceCaps};
use poot_test_util::device_caps::default_caps_for;

fn wgpu() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: default_caps_for(Backend::SpirvVulkan),
    }
}

const REPLAY: CompileOptions = CompileOptions::new(CompileLimits::STANDARD);

/// The first step of the granitemoe registry fixture (two routed layers, four experts) at
/// `tokens` tokens: the production family's own trace, not a hand-built graph.
fn granitemoe_step(tokens: usize) -> poot_graph_ir::Graph {
    granitemoe_step_and_weights(tokens).0
}

/// [`granitemoe_step`] with the model's weight map and the fixture store it binds.
fn granitemoe_step_and_weights(
    tokens: usize,
) -> (
    poot_graph_ir::Graph,
    poot_weights::WeightMap,
    poot_quant::weights::WeightStore,
) {
    registered_step_and_weights("granite", Some("granitemoe"), tokens)
}

/// The first step of a registered family's fixture (its `variant` fixture when named) at
/// `tokens` tokens, with the model's weight map and the store it binds.
fn registered_step_and_weights(
    family: &str,
    variant: Option<&str>,
    tokens: usize,
) -> (
    poot_graph_ir::Graph,
    poot_weights::WeightMap,
    poot_quant::weights::WeightStore,
) {
    use poot_models::model::StepShape;
    use std::num::NonZeroUsize;

    let registry = poot_models::registry::Registry::builtin().expect("the builtin registry");
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == family)
        .unwrap_or_else(|| panic!("{family} is registered"));
    let fixture = match variant {
        None => poot_models::test_support::family_fixtures::base(entry.family),
        Some(variant) => poot_models::test_support::family_fixtures::variant(entry.family, variant),
    };
    let model = (entry.build)(&fixture.raw(), &fixture.store).expect("the fixture builds");
    let n = |v| NonZeroUsize::new(v).unwrap();
    let graph = model
        .model
        .trace(
            &StepSpec::step(StepShape::new(n(1), n(tokens), n(8))),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .expect("the fixture traces");
    (graph, model.model.weights().clone(), fixture.store)
}

/// The CPU oracle's output for `g` over the granitemoe fixture's weights and a fixed `tokens` prompt at
/// positions `0..` with zeroed caches.
fn oracle_logits(
    g: &poot_graph_ir::Graph,
    map: &poot_weights::WeightMap,
    store: &poot_quant::weights::WeightStore,
    tokens: &[i32],
) -> Vec<f32> {
    use poot_graph_ir::{Slot, SlotKey};
    let n = tokens.len();
    let slots = [
        (
            SlotKey::new(Slot::Token, None),
            poot_tensor::HostTensor::i32(vec![1, n], tokens.to_vec()),
        ),
        (
            SlotKey::new(Slot::Pos, None),
            poot_tensor::HostTensor::i32(vec![1, n], (0..n as i32).collect()),
        ),
    ];
    let zeros: Vec<poot_tensor::HostTensor> = g
        .state()
        .map(|(si, _)| {
            poot_tensor::HostTensor::zeros_of(g.aval(si).dtype, g.aval(si).shape.clone())
        })
        .collect();
    let (out, _) =
        poot_eval::test_support::weight_map_oracle::eval_mapped(g, store, map, &slots, &zeros);
    out.as_f32().expect("f32 logits").to_vec()
}

/// Whether `program` ran the pass named `pass` (it changed the graph).
fn ran(program: &Program, pass: &str) -> bool {
    program.passes().any(|ran| ran == pass)
}

const PIPELINE_PASSES: [&str; 3] = ["flash_attention", "fuse_contraction_epilogues", "fuse"];

/// SC-002 (ruling XC2-05, production presets, no synthetic `Target`): the granitemoe prefill, whose two
/// routers contract more than one row, runs the whole pipeline on the ROCm, PTX and wgpu presets exactly as
/// the decode step does, and each program computes what the traced graph computes (ADR-0101 tier 2: every
/// element within 1e-4 relative, NaN fails).
///
/// Mutation (observed red, restored; see the card record): skip the three passes in `compile` whenever
/// `routed_expert_sites(g)` is non-empty; the prefill program loses them on every preset and this row goes
/// red, as would the decode step's, which is a site too.
#[test]
fn routed_expert_prefill_runs_the_full_pipeline_on_every_preset() {
    let presets = [
        (
            "ROCm",
            Target::Gpu {
                backend: Backend::AmdGcn(AmdArch::gfx1151()),
                caps: DeviceCaps::rocm_default(),
            },
        ),
        (
            "PTX",
            Target::Gpu {
                backend: Backend::Nvptx,
                caps: DeviceCaps::ptx_default(),
            },
        ),
        ("wgpu", wgpu()),
    ];
    let (prefill, map, store) = granitemoe_step_and_weights(4);
    let decode = granitemoe_step(1);
    assert!(
        !routed_expert_sites(&prefill).is_empty(),
        "prefill has a site"
    );
    assert!(
        !routed_expert_sites(&decode).is_empty(),
        "a decode step's router is a site as well: the pool pages it"
    );

    let prompt = [3, 17, 40, 8];
    let want = oracle_logits(&prefill, &map, &store, &prompt);
    for (label, target) in &presets {
        let program = compile(&prefill, target, &REPLAY).expect("prefill compiles");
        for pass in PIPELINE_PASSES {
            assert!(ran(&program, pass), "{label} prefill skipped {pass}");
        }
        let got = oracle_logits(program.graph(), &map, &store, &prompt);
        assert_eq!(got.len(), want.len(), "{label}: logits length");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                "{label}: logit {i} is {g}, the traced graph gives {w}"
            );
        }
        let program = compile(&decode, target, &REPLAY).expect("decode compiles");
        for pass in PIPELINE_PASSES {
            assert!(ran(&program, pass), "{label} decode skipped {pass}");
        }
    }
}

/// POOT-742 SC-003 (ruling XC2-05): every MoE family's prefill fixture, compiled through `compile_staged`
/// for the ROCm preset, runs the full pass list, and `routed_expert_sites` (which POOT-746's pool transform
/// reads in production) finds the family's routed experts: each family routes them through `components::moe`,
/// never a dense broadcast over all `E` experts. (`tile_matmuls` is gone: the planner makes that choice, so
/// the contraction and pointwise passes are `fuse_contraction_epilogues` and `fuse`. `flash_attention`
/// is not asserted: deepseek4's compressed-sparse and gpt_oss's sink attention have no flash form.)
///
/// Mutation (observed red, restored; see the card record): restore deepseek3's broadcast form (the routed
/// experts as a dense `MatMul` over all `E`); `routed_expert_sites` finds no site in `deepseek3` and this row
/// goes red naming that family.
#[test]
fn every_moe_family_prefill_has_routed_expert_sites_and_runs_every_pass_on_rocm() {
    // (registered family, fixture variant): every family whose FFN is routed experts.
    const MOE_FAMILIES: [(&str, Option<&str>); 11] = [
        ("llama", Some("mixtral")),
        ("qwen3moe", None),
        ("olmoe", None),
        ("deepseek2", None),
        ("deepseek3", None),
        ("deepseek32", None),
        ("deepseek4", None),
        ("glm53_flash", None),
        ("minimax_m2", None),
        ("gpt_oss", None),
        ("gemma4", Some("moe")),
    ];
    let rocm = Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps: DeviceCaps::rocm_default(),
    };
    let targets = TargetSet::single(DeviceId(0), rocm);
    let partition = Partition {
        experts: ExpertPlacement::AllResident,
        devices: DevicePlacement::Single(DeviceId(0)),
    };
    for (family, variant) in MOE_FAMILIES {
        let prefill = registered_step_and_weights(family, variant, 4).0;
        assert!(
            !routed_expert_sites(&prefill).is_empty(),
            "{family}: the prefill finds no routed-expert site"
        );
        let staged = compile_staged(&prefill, &targets, &partition, &REPLAY)
            .unwrap_or_else(|e| panic!("{family}: prefill compiles: {e:?}"));
        let stages: Vec<_> = staged.stages().collect();
        assert_eq!(stages.len(), 1, "{family}: one stage");
        for pass in ["fuse_contraction_epilogues", "fuse"] {
            assert!(
                ran(stages[0].2, pass),
                "{family}: ROCm prefill skipped {pass}"
            );
        }
    }
}

/// SC-001 (POOT-575, ruling XC2-05): the Qwen3-Next prefill, whose four routers contract eight rows, compiles
/// for the ROCm, PTX and wgpu presets with every pass (`flash_attention`, both fuses) and plans each
/// router contraction on a generated tiled schedule, never the one-thread-per-output serial kernel; each
/// compiled program computes what the traced graph computes (ADR-0101 tier 2). The decode step, whose
/// router has one row, runs the same passes.
///
/// Mutation (observed red, restored; see the card record): skip the three passes in `compile` whenever
/// `routed_expert_sites(g)` is non-empty; the pass check fails with "ROCm prefill skipped
/// flash_attention". The router contraction's own schedule is guarded by the serial-kernel
/// assertion: planning it on the one-thread-per-output kernel instead of a generated tiled schedule
/// fails the `routers` check.
#[test]
fn qwen3next_prefill_tiles_its_routers_and_runs_every_pass_on_every_preset() {
    const TOKENS: usize = 8;
    const EXPERTS: usize = 4;
    let presets = [
        (
            "ROCm",
            Target::Gpu {
                backend: Backend::AmdGcn(AmdArch::gfx1151()),
                caps: DeviceCaps::rocm_default(),
            },
        ),
        (
            "PTX",
            Target::Gpu {
                backend: Backend::Nvptx,
                caps: DeviceCaps::ptx_default(),
            },
        ),
        ("wgpu", wgpu()),
    ];
    let (prefill, map, store) = registered_step_and_weights("qwen3next", None, TOKENS);
    assert!(
        !routed_expert_sites(&prefill).is_empty(),
        "the prefill's routers contract several rows"
    );
    let decode = registered_step_and_weights("qwen3next", None, 1).0;
    let prompt: Vec<i32> = (0..TOKENS as i32).map(|t| (3 * t + 1) % 48).collect();
    let want = oracle_logits(&prefill, &map, &store, &prompt);
    for (label, target) in &presets {
        let program = compile(&prefill, target, &REPLAY).expect("prefill compiles");
        for pass in PIPELINE_PASSES {
            assert!(ran(&program, pass), "{label} prefill skipped {pass}");
        }
        let routers = router_choices_are_serial(&program, TOKENS, EXPERTS);
        assert_eq!(routers.len(), 4, "{label}: one router per layer");
        assert!(
            routers.iter().all(|&serial| !serial),
            "{label}: the routers take the tiled schedule: {routers:?}"
        );
        let got = oracle_logits(program.graph(), &map, &store, &prompt);
        assert_eq!(got.len(), want.len(), "{label}: logits length");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                "{label}: logit {i} is {g}, the traced graph gives {w}"
            );
        }
        let program = compile(&decode, target, &REPLAY).expect("decode compiles");
        for pass in PIPELINE_PASSES {
            assert!(ran(&program, pass), "{label} decode skipped {pass}");
        }
    }
}

/// Whether each eqn of `program` that produces the router's `[tokens, experts]` logits plans the
/// one-thread-per-output serial kernel (`true`) or something else, the generated tiled GEMM among them.
fn router_choices_are_serial(program: &Program, tokens: usize, experts: usize) -> Vec<bool> {
    let g = program.graph();
    g.eqns()
        .iter()
        .filter(|eqn| {
            matches!(
                eqn.op,
                OpKind::MatMul { .. }
                    | OpKind::DenseContraction { .. }
                    | OpKind::ContractionEpilogue { .. }
            ) && g.aval(eqn.out).shape == [tokens, experts]
        })
        .map(|eqn| {
            matches!(
                program.kernel_choice(eqn),
                KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Serial { .. }))
            )
        })
        .collect()
}

/// Card 557 SC-005 and the card-575 router row: the routed prefill's router contraction takes the generated
/// tiled schedule on the ROCm and PTX presets as on wgpu, never the one-thread-per-output serial kernel.
/// Mutation: route a router `MatMul` with `L > 1` to the serial kernel in `planner/matmul.rs`; the
/// router plans serial and this row goes red.
#[test]
fn routed_prefill_router_takes_the_tiled_schedule_on_every_preset() {
    let g = granitemoe_step(4);
    for (label, target) in [
        (
            "ROCm",
            Target::Gpu {
                backend: Backend::AmdGcn(AmdArch::gfx1151()),
                caps: DeviceCaps::rocm_default(),
            },
        ),
        (
            "PTX",
            Target::Gpu {
                backend: Backend::Nvptx,
                caps: DeviceCaps::ptx_default(),
            },
        ),
    ] {
        let program = compile(&g, &target, &REPLAY).expect("the routed prefill compiles");
        let choices = router_choices_are_serial(&program, 4, 4);
        assert_eq!(
            choices.len(),
            2,
            "{label}: one router in each of the two routed layers"
        );
        assert!(
            choices.iter().all(|&serial| !serial),
            "{label}: the router takes the tiled schedule: {choices:?}"
        );
    }
}
