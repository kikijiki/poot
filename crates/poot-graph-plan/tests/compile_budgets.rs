//! Card 666: compilation has explicit size budgets. Every row drives a production entry (`compile`, the
//! per-equation planner, an individual pass through the test-support surface) with limits small enough
//! that the contract's two sides differ, and reads the typed refusal. The limits here are test limits,
//! not production defaults ([`CompileLimits::STANDARD`] is the production policy).
//!
//! What moved: the trace-side rows (a `Model` traced under `TraceLimits`) belong to the card that makes
//! tracing fallible; this file covers the compiler half only.

use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};

use poot_graph_ir::op::SampleRule;
use poot_graph_ir::{BinOp, Builder, Graph, OpKind, RedOp, TensorType, UnOp};
use poot_graph_plan::{
    Capability, CompileError, CompileLimits, CompileOptions, ExactI32StorageAnalysis,
    ExpansionError, FusionPolicy, GraphResource, KernelChoice, PlanError, Submission, Target,
    compile, decompose_large_vocab_greedy, plan_eqn_choice_analyzed,
};
use poot_kernelgen::{BodyLimits, BodyResource, BodyStage, KernelGenError, KernelRequest};
use poot_target::Backend;
use poot_test_util::device_caps::default_caps_for;

const DECOMPOSE: &str = "decompose_large_vocab_greedy";

fn wgpu() -> Target {
    Target {
        backend: Backend::SpirvVulkan,
        caps: default_caps_for(Backend::SpirvVulkan),
    }
}

fn options(limits: CompileLimits) -> CompileOptions {
    CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits,
    }
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

/// The production limits with the graph caps replaced.
fn graph_limits(values: usize, eqns: usize) -> CompileLimits {
    CompileLimits {
        max_intermediate_values: nz(values),
        max_intermediate_eqns: nz(eqns),
        ..CompileLimits::STANDARD
    }
}

/// The production limits with the kernel-body caps replaced.
fn body_limits(instructions: usize, locals: usize) -> CompileLimits {
    CompileLimits {
        max_body_instructions: nz(instructions),
        max_body_locals: nz(locals),
        ..CompileLimits::STANDARD
    }
}

fn expansion(result: Result<impl std::fmt::Debug, CompileError>, what: &str) -> ExpansionError {
    match result {
        Err(CompileError::Expansion(error)) => *error,
        other => panic!("{what}: expected a typed expansion refusal, got {other:?}"),
    }
}

/// `argmax` over one `[vocab]` row: above the decomposition threshold it expands into a chunked
/// two-stage reduction of about thirty equations, which is the transform fixture the limits are read
/// against.
fn greedy_over(vocab: usize) -> Graph {
    let b = Builder::new();
    let logits = b.constant("logits", TensorType::f32(vec![vocab]));
    let out = b.sample_token(SampleRule::Greedy, logits, None, None, None);
    b.finish(out)
}

/// SC-001 (transform expansion): with `max_intermediate_eqns = 8` the decomposition is refused when it
/// would append its ninth equation: the pass reports 8 held, 9 attempted, and `compile` publishes no
/// program. The cap is the same one at the exact fit and one under it: the full expansion of this
/// fixture is accepted at its own length and refused at one less, at the equation it cannot append.
///
/// Mutation: check after the push instead of before (`GraphBudget::push_eqn` pushes, then compares);
/// the refusal then reports 9 held, and the held assertions fail. Mutation: drop the reservation
/// (push unchecked); the pass finishes and the post-pass admission reports the finished graph instead
/// of the ninth equation.
#[test]
fn a_transform_that_would_append_a_ninth_equation_is_refused_before_appending_it() {
    let g = greedy_over(3000);

    let refusal = expansion(
        compile(&g, &wgpu(), &options(graph_limits(1 << 20, 8))),
        "compile",
    );
    assert_eq!(
        refusal,
        ExpansionError {
            stage: DECOMPOSE,
            resource: GraphResource::Eqns,
            held: 8,
            attempted: 9,
            limit: 8,
        },
        "the ninth equation is never appended"
    );

    let whole = decompose_large_vocab_greedy(&g, &CompileLimits::STANDARD).unwrap();
    let n = whole.eqns.len();
    assert!(n > 9, "the fixture expands past the card's limit: {n}");
    let fits = decompose_large_vocab_greedy(&g, &graph_limits(1 << 20, n)).unwrap();
    assert_eq!(fits.eqns.len(), n, "a cap equal to the expansion admits it");
    assert_eq!(
        decompose_large_vocab_greedy(&g, &graph_limits(1 << 20, n - 1)).unwrap_err(),
        ExpansionError {
            stage: DECOMPOSE,
            resource: GraphResource::Eqns,
            held: n - 1,
            attempted: n,
            limit: n - 1,
        }
    );
    compile(&g, &wgpu(), &options(CompileLimits::STANDARD))
        .expect("the production limits compile it");
}

/// SC-001 (values): the value table is capped independently of the equation list. Mutation: swap the two
/// caps in `GraphBudget::limit`; the resource and held assertions name the wrong resource.
#[test]
fn the_value_table_is_capped_independently_of_the_equation_list() {
    let g = greedy_over(3000);
    let refusal = expansion(
        compile(&g, &wgpu(), &options(graph_limits(6, 1 << 20))),
        "compile",
    );
    assert_eq!(refusal.resource, GraphResource::Values);
    assert_eq!(refusal.stage, DECOMPOSE);
    assert_eq!(
        (refusal.held, refusal.attempted, refusal.limit),
        (6, 7, 6),
        "the seventh value is never appended"
    );
}

/// The traced graph is itself within the caps: a graph already over them is refused at the door, before a
/// pass clones it. Mutation: drop the `compile input` admission; the first pass then runs on the
/// oversized graph and the stage name changes.
#[test]
fn a_traced_graph_over_the_caps_is_refused_before_any_pass_runs() {
    let g = greedy_over(16);
    assert_eq!(g.eqns.len(), 1);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4]));
    let y = b.binary(BinOp::Add, x, x);
    let z = b.binary(BinOp::Mul, y, y);
    let three = b.finish(z);
    assert_eq!(three.eqns.len(), 2);
    let refusal = expansion(
        compile(&three, &wgpu(), &options(graph_limits(1 << 20, 1))),
        "compile",
    );
    assert_eq!(
        refusal,
        ExpansionError {
            stage: "compile input",
            resource: GraphResource::Eqns,
            held: 2,
            attempted: 2,
            limit: 1,
        }
    );
}

/// A pass the compiler cannot make count its own appends is sized once, from its input, before it runs:
/// `lower_nonlast_reduces` turns one `keepdim` non-last-axis reduce into a transpose, a reduce and a
/// reshape, so a one-equation graph needs room for three. At two the pass is refused with the graph
/// untouched (one held, three attempted); at three it runs. Mutation: skip the `reserve` call; the pass
/// runs and the post-pass admission refuses the finished graph with three held, not one.
#[test]
fn a_pass_sized_ahead_of_time_is_refused_before_it_runs() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, 8]));
    let out = b.reduce(RedOp::Sum, x, 0, true);
    let g = b.finish(out);
    assert_eq!(g.eqns.len(), 1);

    let refusal = expansion(
        compile(&g, &wgpu(), &options(graph_limits(1 << 20, 2))),
        "compile",
    );
    assert_eq!(
        refusal,
        ExpansionError {
            stage: "lower_nonlast_reduces",
            resource: GraphResource::Eqns,
            held: 1,
            attempted: 3,
            limit: 2,
        }
    );
    compile(&g, &wgpu(), &options(graph_limits(1 << 20, 3))).expect("room for the lowered reduce");
}

/// The dtype widening is sized the same way: a matmul can gain a cast per operand, so a one-equation
/// graph needs room for three whether or not both operands end up widened. Here only the BF16 weight is
/// cast, which is why the refusal at two is the conservative bound and not the exact growth. Mutation:
/// skip the `reserve` call; the pass runs, the graph (two equations) fits the cap, and compile succeeds
/// instead of refusing.
#[test]
fn the_dtype_widening_is_sized_before_it_runs() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![3, 5]));
    let w = b.constant("w", TensorType::f32(vec![5, 4]));
    let out = b.matmul(x, w);
    let mut g = b.finish(out);
    g.values[w.id].aval.dtype = poot_tensor::DType::BF16;
    assert_eq!(g.eqns.len(), 1);

    let refusal = expansion(
        compile(&g, &wgpu(), &options(graph_limits(1 << 20, 2))),
        "compile",
    );
    assert_eq!(
        refusal,
        ExpansionError {
            stage: "widen_mismatched_matmul_dtypes",
            resource: GraphResource::Eqns,
            held: 1,
            attempted: 3,
            limit: 2,
        }
    );
    compile(&g, &wgpu(), &options(graph_limits(1 << 20, 3))).expect("room for both casts");
}

/// `exp(x * s) + b` repeated: a float pointwise chain `fuse` turns into one fused region of `depth * 2`
/// steps.
fn pointwise_chain(depth: usize) -> Graph {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, 8]));
    let s = b.constant("s", TensorType::f32(vec![4, 8]));
    let bias = b.constant("bias", TensorType::f32(vec![4, 8]));
    let mut v = x;
    for _ in 0..depth {
        v = b.binary(
            BinOp::Add,
            b.unary(UnOp::Exp, b.binary(BinOp::Mul, v, s)),
            bias,
        );
    }
    b.finish(v)
}

fn fused_eqns(program: &poot_graph_plan::Program) -> usize {
    program
        .planned()
        .filter(|(eqn, _)| matches!(eqn.op, OpKind::Fused(_)))
        .count()
}

/// SC-002 (production generator): a real fused region is planned through the production planner under
/// body limits. The region's request is refused from its checked size before the generator builds
/// anything (`BodyStage::Sizing`, the size the request would need as `attempted`), at one under the
/// bound and not at the bound. Through `compile`, the same limits make the rewrite unplannable, so the
/// target keeps the decomposition and no fused region reaches the program, while the production limits
/// fuse it.
///
/// Mutation: build every request with a generous `BodyLimits` instead of the site's (bypass the budget
/// propagation); the planner row then plans the region and `compile` keeps the fusion, so both fail.
/// Mutation: remove the `recipe_size` reservation in `poot_kernelgen::generate`; the refusal arrives as
/// `Finished` with the built body's size and the stage assertion fails.
#[test]
fn a_real_fused_region_is_refused_before_it_is_generated_under_small_body_limits() {
    let g = pointwise_chain(6);
    let fused = compile(&g, &wgpu(), &options(CompileLimits::STANDARD)).expect("compiles");
    assert_eq!(
        fused_eqns(&fused),
        1,
        "the chain fuses under production limits"
    );
    let (eqn, _) = fused
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::Fused(_)))
        .unwrap();
    let KernelChoice::Generated(KernelRequest::Pointwise(spec)) = fused.kernel_choice(eqn) else {
        panic!("a fused float region generates a pointwise request");
    };
    assert!(spec.kernel.steps.len() >= 12, "a real region, not one step");

    let graph = fused.graph();
    let analysis = ExactI32StorageAnalysis::new(graph);
    let target = wgpu();
    let plan_with = |limits: &BodyLimits| {
        plan_eqn_choice_analyzed(
            &analysis,
            graph,
            eqn,
            target.backend,
            1,
            &HashMap::new(),
            &target.caps,
            limits,
        )
    };
    let limits = |instructions, locals| BodyLimits {
        max_instructions: nz(instructions),
        max_locals: nz(locals),
    };
    let sizing_refusal = |result: Result<_, PlanError>| match result {
        Err(PlanError::Refused(refusal)) => match refusal.missing {
            Capability::KernelGen(error) => error,
            other => panic!("expected a kernel-generator refusal, got {other:?}"),
        },
        Err(other) => panic!("expected a typed refusal, got {other:?}"),
        Ok(_) => panic!("the region fit limits it must not fit"),
    };

    let KernelGenError::BodyLimit {
        resource: BodyResource::Instructions,
        stage: BodyStage::Sizing,
        attempted: bound,
        ..
    } = sizing_refusal(plan_with(&limits(8, 4)))
    else {
        panic!("the 8-instruction cap refuses the region from its size");
    };
    plan_with(&limits(bound, 1 << 20)).expect("a cap equal to the checked size admits the region");
    assert_eq!(
        sizing_refusal(plan_with(&limits(bound - 1, 1 << 20))),
        KernelGenError::BodyLimit {
            generator: "pointwise",
            resource: BodyResource::Instructions,
            stage: BodyStage::Sizing,
            attempted: bound,
            limit: bound - 1,
        },
        "one instruction under the checked size is refused before generation"
    );
    assert!(matches!(
        sizing_refusal(plan_with(&limits(1 << 20, 4))),
        KernelGenError::BodyLimit {
            resource: BodyResource::Locals,
            stage: BodyStage::Sizing,
            limit: 4,
            ..
        }
    ));

    // Through `compile`: a cap the unfused kernels fit but the fused region's size does not.
    let held_back = compile(&g, &wgpu(), &options(body_limits(bound - 1, 1 << 18)))
        .expect("every unfused kernel fits; only the fused region is over the cap");
    assert_eq!(
        fused_eqns(&held_back),
        0,
        "a region over the body limits stays decomposed"
    );
}

/// A kernel no decomposition can shrink is refused by `compile` as a typed planner refusal carrying the
/// body-limit error. Mutation: map the generator's error to an unrelated capability; the match fails.
#[test]
fn a_kernel_over_the_body_limits_with_no_smaller_form_is_a_typed_planner_refusal() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, 8]));
    let out = b.unary(UnOp::Exp, x);
    let g = b.finish(out);
    let refusal = match compile(&g, &wgpu(), &options(body_limits(2, 2))) {
        Err(CompileError::Plan(error)) => match *error {
            PlanError::Refused(refusal) => *refusal,
            other => panic!("expected a refusal, got {other:?}"),
        },
        other => panic!("expected a planner refusal, got {other:?}"),
    };
    assert!(
        matches!(
            refusal.missing,
            Capability::KernelGen(KernelGenError::BodyLimit { limit: 2, .. })
        ),
        "{refusal:?}"
    );
}

/// The artifact cap travels with the program an executor loads.
#[test]
fn a_program_carries_the_artifact_limit_it_was_compiled_under() {
    let limits = CompileLimits {
        max_artifact_bytes: NonZeroU64::new(64).unwrap(),
        ..CompileLimits::STANDARD
    };
    let program = compile(&greedy_over(16), &wgpu(), &options(limits)).expect("compiles");
    assert_eq!(program.limits().max_artifact_bytes.get(), 64);
}
