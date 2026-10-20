//! The plan digests: a committed record of what `compile` plans for a named fixture corpus, small enough
//! to review, plus the invariants that say a plan is right.
//!
//! For each graph in [`corpus`] and each reference target the test compiles through [`compile`], the one
//! graph-to-program entry point every executor runs (ADR-0100 decision 1), and describes every equation
//! (graph, equation index, op, plan kind, kernel choice, a digest of its request, kernel launches, launch
//! grid, detail). A graph `compile` refuses gets one description naming the typed [`CompileError`]. The
//! description is digested per graph: `tests/plan_summary/digests.tsv` holds one line per target and
//! graph with the plan verdict, the equation and launch counts and the digest. A digest says a plan moved,
//! never that it is right, so it is the tripwire and not the evidence: the evidence is the invariant rows
//! below (every launch fits the device, every one-element-per-thread kernel covers its output) and the
//! literal launch counts. A planner regression is the planner's to catch (`poot-graph-plan`'s own tests);
//! this file only notices that a model's plan changed.
//!
//! A card that changes plans by design regenerates the file with `just test-plan-summary-regen` and lists
//! every changed graph in its landing note. To see which equations of a changed graph moved, run the test
//! with `POOT_PLAN_SUMMARY_DUMP=<dir>` before and after: it writes the full per-equation description of
//! every target to `<dir>/<target>.tsv` (uncommitted), and the two directories diff.
//!
//! The corpus needs no checkpoint: every graph is a registered family's fixture traced at tiny fixed
//! dims, plus the dense parity checkpoints (`support/corpus.rs`), and the invariants also run on the
//! real-dimension models of `support/real_dims.rs`, built over zero weights. A family registered with a
//! fixture joins the corpus with no edit here.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use poot_graph_ir::{Eqn, Operand};
use poot_graph_plan::{
    CompileError, Dispatch, KernelChoice, Plan, PlanError, Program, Target, compile,
};
use poot_models::registry::Registry;

#[path = "support/corpus.rs"]
mod corpus;
#[path = "support/invariants.rs"]
mod invariants;
#[path = "support/real_dims.rs"]
mod real_dims;
#[path = "support/targets.rs"]
mod targets;

use corpus::corpus;
use poot_target::Backend;
use targets::{RefTarget, preset, reference_targets};

/// The environment variable that turns the comparison into a rewrite of the expected file. The
/// `test-plan-summary-regen` recipe sets it.
const REGEN_ENV: &str = "POOT_PLAN_SUMMARY_REGEN";
/// The environment variable naming a directory to write the full per-equation descriptions to.
const DUMP_ENV: &str = "POOT_PLAN_SUMMARY_DUMP";

fn registry() -> Registry {
    Registry::builtin().expect("the builtin registry")
}

// ---- the summary -----------------------------------------------------------------------------

const DUMP_HEADER: &str = "graph\teqn\top\tplan\tchoice\trequest\tlaunches\tgrid\tdetail\n";

/// Tabs and newlines in a free-text field would break the line format.
fn cell(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

/// A 64-bit FNV-1a digest of a text, so a short cell still changes whenever the text does.
fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The `request` cell of a kernel choice: a digest of its whole `Debug` encoding (every parameter that
/// selects a different kernel, shapes and layouts included), so any change to a request changes the
/// line even though the `choice` column only names its kind. The plan's own `key` is not written: it
/// folds in the codegen epoch, so it changes with the toolchain and every `poot-codegen` edit, which
/// would make this committed file churn for reasons that have nothing to do with planning.
fn request_cell(choice: &KernelChoice) -> String {
    format!("{:016x}", fnv1a(&format!("{choice:?}")))
}

/// The variant name of a debug-printed value: `Unsupported("x")` is `Unsupported`.
fn variant_name(debug: &str) -> &str {
    debug
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
        .unwrap_or(debug)
}

/// One summary line body (everything after `graph`, `eqn` and `op`) for a plan result.
struct PlanCells {
    plan: &'static str,
    /// [`KernelChoice::label`], or `-` for a refused graph.
    choice: String,
    request: String,
    launches: usize,
    grid: String,
    detail: String,
}

impl PlanCells {
    fn no_launch(plan: &'static str, choice: &KernelChoice, detail: String) -> Self {
        Self {
            plan,
            choice: choice.label(),
            request: request_cell(choice),
            launches: 0,
            grid: "-".to_string(),
            detail,
        }
    }
}

/// What an attention equation's plan records beyond its grid: the schedule, its lane count, the workgroup
/// memory the body states, the key layout and the key bound (`live(v<id>)` names the length operand of
/// `eqn`). `-` for any other choice.
fn attention_detail(eqn: &Eqn, choice: &KernelChoice, dispatch: &Dispatch) -> String {
    let KernelChoice::Generated(poot_kernelgen::KernelRequest::Attention(spec)) = choice else {
        return "-".to_string();
    };
    let lanes = spec.schedule.lanes();
    let used = poot_kernelgen::BodyUse::measure(dispatch.body())
        .expect("an attention body states measurable workgroup memory");
    let key = match spec.key {
        poot_kernelgen::KeyLayout::Single => "single".to_string(),
        poot_kernelgen::KeyLayout::Split { shared_dim } => format!("split({shared_dim})"),
        poot_kernelgen::KeyLayout::Paged => "paged".to_string(),
    };
    let bound = match (spec.bound, eqn.inputs.last()) {
        (poot_kernelgen::KeyBound::Static, _) => "capacity".to_string(),
        (poot_kernelgen::KeyBound::Live, Some(Operand::Value(len))) => format!("live({len})"),
        (poot_kernelgen::KeyBound::Live, _) => {
            unreachable!("a live bound reads its length operand")
        }
    };
    format!(
        "schedule=streaming lanes={lanes} lds={} key={key} bound={bound}",
        used.lds_bytes
    )
}

/// `plan`'s summary cells, for one equation of a successfully compiled [`poot_graph_plan::Program`],
/// with the kernel choice the program recorded for it.
fn describe_plan(eqn: &Eqn, plan: &Plan, choice: &KernelChoice) -> PlanCells {
    let grid_str = |[x, y, z]: [u32; 3]| format!("{x},{y},{z}");
    let cells = |plan, grid, detail| PlanCells {
        plan,
        choice: choice.label(),
        request: request_cell(choice),
        launches: 1,
        grid,
        detail,
    };
    match plan {
        Plan::Dispatches(dispatches) => match &dispatches[..] {
            [dispatch] if dispatch.launch().meta.is_empty() => cells(
                "compute",
                grid_str(dispatch.launch().grid),
                attention_detail(eqn, choice, dispatch),
            ),
            [dispatch] => cells(
                "compute_meta",
                grid_str(dispatch.launch().grid),
                format!("meta={:?}", dispatch.launch().meta),
            ),
            dispatches => PlanCells {
                plan: "chunks",
                choice: choice.label(),
                request: request_cell(choice),
                launches: dispatches.len(),
                grid: format!(
                    "groups={:?}",
                    dispatches
                        .iter()
                        .map(|d| d.launch().grid[0] / d.body().workgroup_size[0])
                        .collect::<Vec<_>>()
                ),
                detail: format!(
                    "meta={:?}",
                    dispatches
                        .iter()
                        .map(|d| d.launch().meta.as_slice())
                        .collect::<Vec<_>>()
                ),
            },
        },
        Plan::Alias(src) => PlanCells::no_launch("alias", choice, format!("src={src}")),
        Plan::View {
            src,
            strides,
            offset,
        } => PlanCells::no_launch(
            "view",
            choice,
            format!("src={src} strides={strides:?} offset={offset}"),
        ),
        Plan::Graph => PlanCells::no_launch("graph", choice, String::new()),
    }
}

/// The `eqn`, `op` and cells of the one summary line a whole-graph [`CompileError`] produces: `compile`
/// refuses (or otherwise fails) the graph as a unit, not one equation at a time, so a refused corpus
/// graph gets exactly one line, naming the refused equation and op when the error is a typed
/// [`PlanError::Refused`] (the only refusal kind the corpus is expected to hit) and `-` for any other
/// [`CompileError`] variant (numerics, resident-E4M3, legalize, or a non-refusal [`PlanError`]).
fn describe_compile_error(error: &CompileError) -> (String, String, PlanCells) {
    let (eqn, op, key) = match error {
        CompileError::Plan(plan_error) => match plan_error.as_ref() {
            PlanError::Refused(refusal) => (
                refusal.eqn.to_string(),
                refusal.op.name(),
                variant_name(&format!("{:?}", refusal.missing)).to_string(),
            ),
            other => (
                "-".to_string(),
                "-".to_string(),
                variant_name(&format!("{other:?}")).to_string(),
            ),
        },
        other => (
            "-".to_string(),
            "-".to_string(),
            variant_name(&format!("{other:?}")).to_string(),
        ),
    };
    (
        eqn,
        op,
        PlanCells {
            plan: "refused",
            choice: "-".to_string(),
            request: key,
            launches: 0,
            grid: "-".to_string(),
            detail: error.to_string(),
        },
    )
}

/// One summary line.
fn write_line(out: &mut String, name: &str, eqn: &str, op: &str, c: &PlanCells) {
    writeln!(
        out,
        "{name}\t{eqn}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        cell(op),
        c.plan,
        cell(&c.choice),
        cell(&c.request),
        c.launches,
        c.grid,
        cell(&c.detail),
    )
    .expect("write to a String");
}

/// One graph's description, as `compile` planned (or refused) it.
struct GraphPlan {
    /// The description lines, one per planned equation or one for a refused graph.
    text: String,
    /// `plans`, or `refused` when `compile` returned an error.
    verdict: &'static str,
    /// Planned equations, or 0 for a refused graph.
    eqns: usize,
    /// Kernel launches over all equations.
    launches: usize,
}

/// `g` compiled for `target` and described: one line per planned equation, or one line for the whole
/// graph when `compile` refuses it.
fn describe_graph(name: &str, g: &poot_graph_ir::Graph, target: &Target) -> GraphPlan {
    let mut text = String::new();
    match compile(g, target, &targets::options()) {
        Ok(program) => {
            let mut launches = 0;
            let mut eqns = 0;
            for (i, (eqn, plan)) in program.planned().enumerate() {
                let c = describe_plan(eqn, plan, program.kernel_choice(eqn));
                write_line(&mut text, name, &i.to_string(), &eqn.op.name(), &c);
                launches += c.launches;
                eqns += 1;
            }
            GraphPlan {
                text,
                verdict: "plans",
                eqns,
                launches,
            }
        }
        Err(error) => {
            let (eqn, op, c) = describe_compile_error(&error);
            write_line(&mut text, name, &eqn, &op, &c);
            GraphPlan {
                text,
                verdict: "refused",
                eqns: 0,
                launches: 0,
            }
        }
    }
}

const DIGEST_HEADER: &str = "target\tgraph\tplan\teqns\tlaunches\tdigest\n";

/// Every corpus graph on `reference`, as `(graph, description)`, in corpus order.
fn plans(registry: &Registry, reference: &RefTarget) -> Vec<(String, GraphPlan)> {
    corpus(registry)
        .into_iter()
        .map(|row| {
            let plan = describe_graph(&row.name, &row.graph(), &reference.target);
            (row.name, plan)
        })
        .collect()
}

/// The digest file's lines for every reference target: the verdict, the counts and the digest of the
/// graph's full description.
fn render_digests(registry: &Registry) -> String {
    let mut out = String::from(DIGEST_HEADER);
    for reference in reference_targets() {
        for (graph, plan) in plans(registry, &reference) {
            writeln!(
                out,
                "{}\t{graph}\t{}\t{}\t{}\t{:016x}",
                reference.id,
                plan.verdict,
                plan.eqns,
                plan.launches,
                fnv1a(&plan.text)
            )
            .expect("write to a String");
        }
    }
    out
}

/// The full per-equation description of `reference`, for [`DUMP_ENV`].
fn render_dump(registry: &Registry, reference: &RefTarget) -> String {
    let mut out = String::from(DUMP_HEADER);
    for (_, plan) in plans(registry, reference) {
        out.push_str(&plan.text);
    }
    out
}

fn digests_path() -> PathBuf {
    // `std::env::var`, not `env!`: the latter bakes the path into this test binary's compiled object code
    // at build time, and a shared compile cache (kache) that reuses that object across worktrees by
    // source-content hash would then serve whichever worktree's path happened to compile it first.
    // `std::env::var` reads the environment cargo sets fresh for every test-binary invocation.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set by cargo for test binaries");
    PathBuf::from(manifest_dir)
        .join("tests")
        .join("plan_summary")
        .join("digests.tsv")
}

// ---- the comparison --------------------------------------------------------------------------

/// A digest line with its tabs as ` | `, for a failure message.
fn spaced(line: &str) -> String {
    line.replace('\t', " | ")
}

/// The graphs whose line differs between two digest files, one `old -> new` pair per graph, keyed by
/// (target, graph). Empty when the files are equal.
fn diff(expected: &str, actual: &str) -> String {
    fn by_key(text: &str) -> BTreeMap<(&str, &str), &str> {
        text.lines()
            .skip(1)
            .map(|line| {
                let mut cols = line.splitn(3, '\t');
                let key = (cols.next().unwrap_or(""), cols.next().unwrap_or(""));
                (key, line)
            })
            .collect()
    }
    let (old, new) = (by_key(expected), by_key(actual));
    let mut changes = Vec::new();
    for (key, old_line) in &old {
        match new.get(key) {
            Some(new_line) if new_line == old_line => {}
            Some(new_line) => changes.push(format!(
                "changed {} {}\n  - {}\n  + {}",
                key.0,
                key.1,
                spaced(old_line),
                spaced(new_line)
            )),
            None => changes.push(format!(
                "removed {} {}\n  - {}",
                key.0,
                key.1,
                spaced(old_line)
            )),
        }
    }
    for (key, new_line) in &new {
        if !old.contains_key(key) {
            changes.push(format!(
                "added {} {}\n  + {}",
                key.0,
                key.1,
                spaced(new_line)
            ));
        }
    }
    if changes.is_empty() && expected != actual {
        return "every graph line is unchanged but the lines are in a different order".to_string();
    }
    changes.join("\n")
}

/// The checked-in digests equal what `compile` plans now. Mutation: change one op's lowering (for
/// example the row-kernel workgroup size in the planner): the graphs that contain it change digest and
/// the failure names them, target by target.
#[test]
fn plan_digests_match_the_checked_in_file() {
    let registry = registry();
    if let Some(dir) = std::env::var_os(DUMP_ENV) {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
        for reference in reference_targets() {
            let path = dir.join(format!("{}.tsv", reference.id));
            std::fs::write(&path, render_dump(&registry, &reference))
                .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        }
    }
    let actual = render_digests(&registry);
    let path = digests_path();
    if std::env::var_os(REGEN_ENV).is_some() {
        std::fs::create_dir_all(path.parent().expect("the digest file has a parent"))
            .expect("create tests/plan_summary");
        std::fs::write(&path, &actual).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e}\nregenerate with `just test-plan-summary-regen`",
            path.display()
        )
    });
    let changes = diff(&expected, &actual);
    assert!(
        changes.is_empty(),
        "the plan digests differ from {}\n{changes}\n\
         If the plan change is intended, run `just test-plan-summary-regen` and list every changed \
         graph in the card's landing note.",
        path.display()
    );
}

/// The plan kinds `poot_executor::Engine::load` accepts: every kind a single-device `compile` may emit.
/// An exhaustive match (no wildcard), so a new `Plan` variant must be classified here before this
/// file compiles.
fn is_loadable(plan: &Plan) -> bool {
    match plan {
        Plan::Dispatches(_) | Plan::Alias(_) | Plan::View { .. } | Plan::Graph => true,
    }
}

/// Card 626 SC-003: `compile` of every corpus graph, on every target, never yields a plan kind an
/// executor cannot load (the deleted `Plan::Host` was one). Mutation: re-add a `Plan::Host` variant,
/// classify it `false` above, and make the planner emit it for a corpus op; this row goes red naming
/// the graph, target and equation.
#[test]
fn compile_emits_only_loadable_plan_kinds_over_the_whole_corpus() {
    let mut planned = 0usize;
    let registry = registry();
    let options = targets::options();
    for RefTarget { column, target, .. } in reference_targets() {
        for row in corpus(&registry) {
            let g = row.graph();
            let Ok(program) = compile(&g, &target, &options) else {
                continue;
            };
            for (i, (eqn, plan)) in program.planned().enumerate() {
                assert!(
                    is_loadable(plan),
                    "{} on {column}: eqn {i} ({}) compiled to the unloadable kind {:?}",
                    row.name,
                    eqn.op.name(),
                    plan.kind()
                );
                planned += 1;
            }
        }
    }
    assert!(
        planned > 1000,
        "the corpus planned only {planned} equations"
    );
}

/// Rendering the digests twice, from freshly traced graphs, gives the same text: nothing in the
/// description depends on hash-map order, an address or a timestamp. Without this a checked-in file
/// could pass once and fail on the next run.
#[test]
fn plan_summary_rendering_is_deterministic() {
    let registry = registry();
    let (first, second) = (render_digests(&registry), render_digests(&registry));
    assert!(
        first == second,
        "two renders of the same corpus differ\n{}",
        diff(&first, &second)
    );
}

/// Card 557 SC-003: a tracer states primitives and `ops::` compositions only. Every corpus graph, as
/// traced and before `compile`, holds no compiler-fusion or schedule-choice composite (`OpClass::
/// Composite`: the flash ops, `Rope`, `ContractionEpilogue` and the fused regions; the packed and dense claims
/// are `Primitive` until Card 629 gives them a decomposition). The authorized bounded semantic source
/// carriers are the recurrence `Scan` (it states a recurrence and its bounded expansion, not a schedule) and
/// the indexer's `BlockTopK` and `SelectedAttention` (they state a selection and the attention over it, with a dense oracle definition, not a schedule). Then `compile`'s passes do form composites, so
/// the check is not vacuous. Mutation: emit `ContractionEpilogue` from `ops::linear`, or mark one of those ops
/// `OpClass::Primitive` and trace it; this row names the graph and equation.
#[test]
fn every_corpus_trace_is_primitive_before_compile() {
    use poot_graph_ir::OpClass;

    let mut composites_after_compile = 0usize;
    let target = preset("wgpu", "spirv_vulkan", Backend::SpirvVulkan).target;
    let options = targets::options();
    for row in corpus(&registry()) {
        let g = row.graph();
        for (i, eqn) in g.eqns().iter().enumerate() {
            if matches!(
                eqn.op,
                poot_graph_ir::OpKind::Scan(_)
                    | poot_graph_ir::OpKind::BlockTopK(_)
                    | poot_graph_ir::OpKind::SelectedAttention(_)
            ) {
                continue;
            }
            assert_eq!(
                eqn.op.class(),
                OpClass::Primitive,
                "{}: traced eqn {i} is the composite {}",
                row.name,
                eqn.op.name()
            );
        }
        if let Ok(program) = compile(&g, &target, &options) {
            composites_after_compile += program
                .graph()
                .eqns()
                .iter()
                .filter(|eqn| eqn.op.class() == OpClass::Composite)
                .count();
        }
    }
    assert!(
        composites_after_compile > 0,
        "compile formed no composite over the corpus, so the traced check proves nothing"
    );
}

// ---- the invariants --------------------------------------------------------------------------

/// SC-003: every plan `compile` makes for the fixture corpus satisfies the launch invariants on its
/// target's caps: each launch is a whole number of workgroups the device can run within its workgroup
/// memory, and each one-element-per-thread kernel launches at least its output's element count.
/// Mutation: have the planner launch one workgroup fewer for a row kernel, or drop the cover check in
/// `support/invariants.rs`: the failure names the equation and the launch.
#[test]
fn every_corpus_plan_satisfies_the_launch_invariants() {
    let registry = registry();
    let mut checked = 0usize;
    for RefTarget { column, target, .. } in reference_targets() {
        for row in corpus(&registry) {
            let Ok(program) = compile(&row.graph(), &target, &targets::options()) else {
                continue;
            };
            let found = invariants::violations(&program);
            assert!(
                found.is_empty(),
                "{} on {column} breaks a launch invariant:\n{}",
                row.name,
                found.join("\n")
            );
            checked += program.planned().count();
        }
    }
    assert!(
        checked > 1000,
        "the corpus checked only {checked} equations"
    );
}

/// The same invariants at real dimensions, on the targets' preset caps: a launch grid, a workgroup
/// memory budget or a limit that only a real width reaches. A graph the planner refuses at these
/// dimensions is a matrix cell (`feature_matrix`), not an invariant violation.
#[test]
fn every_real_dimension_plan_satisfies_the_launch_invariants() {
    let root = std::path::Path::new(
        &std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    )
    .join("../..");
    let mut planned = 0usize;
    for model in real_dims::real_models(&root) {
        for &shape in model.shapes() {
            let graph = model.trace(shape);
            for RefTarget { column, target, .. } in reference_targets() {
                let Ok(program) = compile(&graph, &target, &targets::options()) else {
                    continue;
                };
                let found = invariants::violations(&program);
                assert!(
                    found.is_empty(),
                    "{} {} on {column} breaks a launch invariant:\n{}",
                    model.label,
                    shape.name,
                    found.join("\n")
                );
                planned += 1;
            }
        }
    }
    assert!(
        planned > 20,
        "only {planned} real-dimension programs planned"
    );
}

/// The first equation of `program` whose only launch writes one element per thread and whose grid has
/// no slack: one workgroup fewer cannot cover its output.
fn tight_launch(program: &Program) -> usize {
    program
        .planned()
        .position(|(eqn, plan)| {
            let [dispatch] = plan.dispatches() else {
                return false;
            };
            let workgroup = u64::from(dispatch.body().workgroup_size[0]);
            let threads = dispatch
                .launch()
                .grid
                .iter()
                .map(|&x| u64::from(x))
                .product::<u64>();
            let elements = program
                .graph()
                .aval(eqn.out)
                .shape
                .iter()
                .map(|&d| d as u64)
                .product::<u64>();
            invariants::writes_one_element_per_thread(dispatch)
                && dispatch.launch().grid[1] == 1
                && dispatch.launch().grid[2] == 1
                && threads > workgroup
                && threads - workgroup < elements
        })
        .expect("the corpus holds a tight one-element-per-thread launch")
}

/// SC-003 control: a plan whose launch grid is one workgroup short of covering its output fails the
/// invariants, naming the equation. Without it the rows above would pass whatever the checker did.
/// Mutation: drop the grid-covers-output check in `support/invariants.rs`; this row goes red on
/// `a grid one workgroup short went undetected`.
#[test]
fn a_launch_grid_one_workgroup_short_is_caught() {
    let wgpu = preset("wgpu", "spirv_vulkan", Backend::SpirvVulkan);
    let graph = corpus(&registry())
        .into_iter()
        .find(|row| row.name == "parity.qwen2.decode")
        .expect("the parity corpus has a qwen2 decode graph")
        .graph();
    let mut program = compile(&graph, &wgpu.target, &targets::options()).expect("qwen2 plans");
    assert_eq!(
        invariants::violations(&program),
        Vec::<String>::new(),
        "the unmodified plan is sound"
    );

    let index = tight_launch(&program);
    program
        .shrink_first_launch_by_one_workgroup(index)
        .expect("a shorter grid is still a valid dispatch plan");
    let found = invariants::violations(&program);
    assert!(
        found.iter().any(|v| v.contains(&format!("eqn {index} "))
            && v.contains("threads launched for an output of")),
        "a grid one workgroup short went undetected: {found:?}"
    );
}

/// Launch counts, written out. A digest changes whenever anything about a plan does; these numbers are
/// the ones a reviewer can hold the planner to. One decode step of the two-layer dense qwen2 launches 38
/// kernels (a token gather, two layers of norm, Q/K/V projections, two rotaries, two cache updates,
/// attention, output projection, norm and the gate/up/down feed-forward, then the final norm and the
/// head), and qwen3 adds one query-norm and one key-norm launch per layer: 38 + 2 * 2 = 42. The counts
/// are the same on every target because no target here lacks a fusion another has. The same graph takes
/// far fewer launches than it has equations. Mutation: disable a fusion pass (for example
/// `fuse_contraction_epilogues`): the launch count rises and the row names the graph and the target.
#[test]
fn literal_launch_counts() {
    let registry = registry();
    let corpus = corpus(&registry);
    for (name, launches) in [
        ("parity.qwen2.decode", 38usize),
        ("parity.qwen3.decode", 42),
    ] {
        let graph = corpus
            .iter()
            .find(|row| row.name == name)
            .unwrap_or_else(|| panic!("the parity corpus has {name}"))
            .graph();
        for reference in reference_targets() {
            let program = compile(&graph, &reference.target, &targets::options()).expect("plans");
            let counted: usize = program
                .planned()
                .map(|(_, plan)| plan.dispatches().len())
                .sum();
            assert_eq!(counted, launches, "{name} launches on {}", reference.column);
            assert!(
                counted < program.graph().eqns().len(),
                "{name} on {}: {counted} launches for {} equations: nothing fused",
                reference.column,
                program.graph().eqns().len()
            );
        }
    }
}
