//! POOT-666 SC-001, trace side (moved to POOT-733, then to poot-models by POOT-1130): the compile budgets on a
//! real model trace. It traces a model, so it sits above poot-graph-plan, whose own `compile_budgets.rs` rows
//! run on hand-built graphs.

use std::num::NonZeroUsize;

use poot_graph_plan::{CompileError, CompileLimits, CompileOptions, Target, compile};
use poot_target::Backend;
use poot_test_util::device_caps::default_caps_for;

fn wgpu() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: default_caps_for(Backend::SpirvVulkan),
    }
}

fn options(limits: CompileLimits) -> CompileOptions {
    CompileOptions::new(limits)
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

/// POOT-666 SC-001, trace side (moved to POOT-733): a real model traced under explicit caller limits
/// that exactly fit it reaches `compile`, which then runs under its own limits. The two budgets are
/// independent: the same traced graph is refused by `compile` once the compile's intermediate-equation
/// cap is one below its size, and a trace that its caller's limits refuse never reaches compile at all.
#[test]
fn a_real_model_trace_under_explicit_limits_reaches_compile_and_each_budget_refuses_alone() {
    use poot_graph_ir::TraceLimits;
    use poot_models::model::{StepShape, StepSpec};
    use poot_models::registry::Registry;

    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family == poot_models::qwen2::model::FAMILY)
        .copied()
        .expect("qwen2 is shipped");
    let fixture = (entry.fixture)();
    let model = registry.build(&fixture.raw(), &fixture.store).unwrap();
    let shape = |tokens: usize| StepShape::new(NonZeroUsize::MIN, nz(tokens), nz(16));
    let roomy = TraceLimits::for_tests();
    let traced = model.model.trace(&StepSpec::step(shape(1)), roomy).unwrap();
    let exact = TraceLimits {
        max_tokens: nz(1),
        max_values: nz(traced.values().len()),
        max_eqns: nz(traced.eqns().len()),
        ..roomy
    };
    let graph = model.model.trace(&StepSpec::step(shape(1)), exact).unwrap();
    compile(&graph, &wgpu(), &options(CompileLimits::STANDARD))
        .expect("an exactly traced step compiles");

    let tight = CompileLimits {
        max_intermediate_eqns: nz(graph.eqns().len() - 1),
        ..CompileLimits::STANDARD
    };
    assert!(
        matches!(
            compile(&graph, &wgpu(), &options(tight)),
            Err(CompileError::Expansion(_))
        ),
        "the compile's own cap refuses the graph the trace's cap admitted"
    );

    // A step above the caller's token limit is refused by the trace before its body: no graph exists
    // for `compile` to see.
    let four = TraceLimits {
        max_tokens: nz(4),
        ..roomy
    };
    model.model.trace(&StepSpec::step(shape(4)), four).unwrap();
    assert!(model.model.trace(&StepSpec::step(shape(6)), four).is_err());
}
