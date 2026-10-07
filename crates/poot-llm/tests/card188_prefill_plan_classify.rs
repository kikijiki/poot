//! CPU-only plan classification of the Gemma4 fixture (no GPU, no checkpoint): every attention
//! layer plans as the fused flash kernel on every planner target, in a prefill and in a decode
//! graph (POOT-574 SC-001 and SC-003: the PTX row's plan half). Local layers have 16-wide heads and
//! global layers 32-wide heads, so a matcher or planner that admits only the narrow heads leaves the
//! global layers decomposed and the count short of the layer count.
//!
//! Mutation: cap the flash matcher's admitted head dim below Gemma4's global head dim. In
//! `attention_match`'s `flash_attention_capped`, the `fits` closure returns early for a decode
//! match and checks `max_prefill_groups` after that, so put the cap first: `if
//! g.aval(m.q).shape[3] > 16 { return false; }` as its first statement. The global layers
//! (32-wide heads) fall back, the fused count drops to the local layers' and each row fails.

use poot_graph_ir::op::OpKind;
use poot_graph_plan::{CompileOptions, Plan, Submission, Target, compile};
use poot_models::model::{KvLayout, LogitRows, Model, Phase, StepShape};
use poot_models::registry::Registry;
use poot_target::{AmdArch, Backend};
use std::num::NonZeroUsize;

const LAYERS: usize = 4;

/// The dense fixture, or the one with routed experts beside every layer (`"moe"`).
fn gemma4(variant: Option<&str>) -> Box<dyn Model> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "gemma4")
        .expect("the builtin registry holds gemma4");
    let fixture = match variant {
        None => (entry.fixture)(),
        Some(name) => {
            let (_, make) = entry
                .variants
                .iter()
                .find(|(n, _)| *n == name)
                .expect("the gemma4 variant");
            make()
        }
    };
    registry.build(&fixture.raw(), &fixture.store).unwrap()
}

fn step(tokens: usize, capacity: usize) -> StepShape {
    StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::new(tokens).unwrap(),
        capacity: NonZeroUsize::new(capacity).unwrap(),
        kv: KvLayout::Contiguous,
        logits: LogitRows::Last,
    }
}

/// The dispatches of every fused attention equation of `phase`'s graph compiled for `backend`: one
/// entry per attention equation, each the dispatch count of its plan.
fn attention_dispatches(
    model: &dyn Model,
    backend: Backend,
    phase: Phase,
    shape: StepShape,
) -> Vec<usize> {
    let graph = model.trace(phase, shape).unwrap();
    let target = Target {
        backend,
        caps: poot_test_util::device_caps::default_caps_for(backend),
    };
    let options = CompileOptions {
        execution: Submission::Replay,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    let program = compile(&graph, &target, &options).unwrap();
    program
        .planned()
        .filter(|(eqn, _)| matches!(eqn.op, OpKind::Attention(_)))
        .map(|(_, plan)| match plan {
            Plan::Dispatches(dispatches) => dispatches.len(),
            other => panic!("a fused attention planned as {other:?}"),
        })
        .collect()
}

/// The routed fixture too where the target has no routed-expert hang guard: a target that has one
/// (ROCm and PTX, cards 186/192) compiles a routed graph without the flash rewrite by design.
fn assert_every_layer_is_flash(backend: Backend, label: &str, routed_too: bool) {
    let variants: &[Option<&str>] = if routed_too {
        &[None, Some("moe")]
    } else {
        &[None]
    };
    for &variant in variants {
        assert_every_layer_of_is_flash(&*gemma4(variant), backend, &format!("{label} {variant:?}"));
    }
}

fn assert_every_layer_of_is_flash(model: &dyn Model, backend: Backend, label: &str) {
    for (phase, shape) in [(Phase::Prefill, step(4, 8)), (Phase::Decode, step(1, 8))] {
        let dispatches = attention_dispatches(model, backend, phase, shape);
        eprintln!("gemma4 {label} {phase:?}: attention dispatches per layer {dispatches:?}");
        assert_eq!(
            dispatches, [1; LAYERS],
            "{label} {phase:?}: every attention layer is one fused flash dispatch"
        );
    }
}

#[test]
fn gemma4_attention_layers_plan_as_flash_on_wgpu_and_vulkan() {
    assert_every_layer_is_flash(Backend::SpirvVulkan, "spirv", true);
}

#[test]
fn gemma4_attention_layers_plan_as_flash_on_amd_gcn() {
    assert_every_layer_is_flash(Backend::AmdGcn(AmdArch::gfx1151()), "amd_gcn", false);
}

#[test]
fn gemma4_attention_layers_plan_as_flash_on_nvptx() {
    assert_every_layer_is_flash(Backend::Nvptx, "nvptx", false);
}
