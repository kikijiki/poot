//! Card 599: a committed plan-summary dump over a named fixture corpus.
//!
//! Cards that change plans on purpose (or by accident) gate on "every fixture graph plans with unchanged
//! dispatch counts". This test is that gate. For each graph in [`corpus`] and each target it compiles
//! through [`compile`], the one graph-to-program entry point every executor runs (ADR-0100 decision 1):
//! no hand-rolled sub-pipeline here duplicates what `compile` already does (card 523b review: that
//! duplication is the exact "per-executor pre-plan pipelines drift" this epic removes). A
//! graph `compile` accepts gets one summary line per equation (graph, equation index, op, plan kind,
//! kernel choice, a digest of its request, kernel launches, launch grid, detail); a graph `compile` refuses gets exactly one summary
//! line for the whole graph, carrying the typed [`CompileError`] as `compile` reports it - never a
//! partial per-equation dump of an intermediate pipeline stage `compile` itself would still rewrite.
//! The lines are compared with the checked-in `tests/plan_summary/<target>.tsv`.
//!
//! A raw hash of the plans would change whenever a plan type changes; a line per equation names what
//! moved. A card that changes plans by design regenerates the files with `just test-plan-summary-regen`
//! and lists every changed line in its landing note.
//!
//! The corpus needs no checkpoint: every graph is a registered family's fixture traced at tiny fixed
//! dims, plus the dense parity checkpoints (`support/corpus.rs`). A family registered with a fixture
//! joins it with no edit here.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use poot_graph_ir::{Eqn, Graph, Operand};
use poot_graph_plan::{
    CompileError, CompileOptions, Dispatch, KernelChoice, Plan, PlanError, Target, compile,
};
use poot_models::registry::Registry;

#[path = "support/corpus.rs"]
mod corpus;
#[path = "support/targets.rs"]
mod targets;

use corpus::corpus;
use poot_target::Backend;
use targets::{RefTarget, preset, reference_targets};

/// The environment variable that turns the comparison into a rewrite of the expected files. The
/// `test-plan-summary-regen` recipe sets it.
const REGEN_ENV: &str = "POOT_PLAN_SUMMARY_REGEN";

fn registry() -> Registry {
    Registry::builtin().expect("the builtin registry")
}

// ---- the summary -----------------------------------------------------------------------------

const HEADER: &str = "graph\teqn\top\tplan\tchoice\trequest\tlaunches\tgrid\tdetail\n";

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

/// The whole summary for one target: every corpus graph, in corpus order, compiled through the real
/// [`compile`] entry point (card 523b: a hand-rolled `optimize`/`widen_mismatched_matmul_dtypes`/
/// `plan_eqn_views` sub-pipeline here was the same "second entry point" anti-pattern R469-021
/// found in the wgpu/ROCm/PTX executors before card 534a unified them onto `compile`).
fn render(registry: &Registry, target: &Target) -> String {
    let mut out = String::from(HEADER);
    let options = targets::options();
    for row in corpus(registry) {
        summarize(&mut out, &row.name, &row.graph(), target, &options);
    }
    out
}

/// One graph's lines: one per planned equation, or one for the whole graph when `compile` refuses it.
fn summarize(out: &mut String, name: &str, g: &Graph, target: &Target, options: &CompileOptions) {
    match compile(g, target, options) {
        Ok(program) => {
            for (i, (eqn, plan)) in program.planned().enumerate() {
                let c = describe_plan(eqn, plan, program.kernel_choice(eqn));
                write_line(out, name, &i.to_string(), &eqn.op.name(), &c);
            }
        }
        Err(error) => {
            let (eqn, op, c) = describe_compile_error(&error);
            write_line(out, name, &eqn, &op, &c);
        }
    }
}

fn expected_path(target: &str) -> PathBuf {
    // `std::env::var`, not `env!`: the latter bakes the path into this test binary's compiled object code
    // at build time, and a shared compile cache (kache) that reuses that object across worktrees by
    // source-content hash would then serve whichever worktree's path happened to compile it first
    // (card 530's build.rs fix; card 543 review). `std::env::var` reads the environment cargo sets fresh
    // for every test-binary invocation, so it is correct regardless of which worktree compiled the binary.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set by cargo for test binaries");
    PathBuf::from(manifest_dir)
        .join("tests")
        .join("plan_summary")
        .join(format!("{target}.tsv"))
}

// ---- the comparison --------------------------------------------------------------------------

/// The `graph` and `eqn` columns of a summary line: the identity a changed line is paired by.
fn line_key(line: &str) -> (&str, &str) {
    let mut cols = line.splitn(3, '\t');
    (cols.next().unwrap_or(""), cols.next().unwrap_or(""))
}

/// Lines shown in a failure before the rest are counted.
const DIFF_LIMIT: usize = 40;

/// The columns of two lines for the same equation that differ, one `column: old -> new` line each.
fn changed_columns(old: &str, new: &str) -> String {
    let names = HEADER.trim_end().split('\t');
    let mut old_cols = old.split('\t');
    let mut new_cols = new.split('\t');
    let mut lines = Vec::new();
    for name in names {
        let (o, n) = (old_cols.next().unwrap_or(""), new_cols.next().unwrap_or(""));
        if o != n {
            lines.push(format!("    {name}: {o} -> {n}"));
        }
    }
    lines.join("\n")
}

/// A readable diff of two summaries: each changed equation once, with the columns that moved (old to
/// new), paired by (graph, equation). Empty when they are equal.
fn diff(expected: &str, actual: &str) -> String {
    fn by_key(text: &str) -> BTreeMap<(&str, usize), &str> {
        text.lines()
            .skip(1)
            .map(|l| {
                let (g, e) = line_key(l);
                ((g, e.parse::<usize>().unwrap_or(usize::MAX)), l)
            })
            .collect()
    }
    let (old, new) = (by_key(expected), by_key(actual));
    let mut changes: Vec<(&str, String)> = Vec::new();
    for (key, old_line) in &old {
        match new.get(key) {
            Some(new_line) if new_line == old_line => {}
            Some(new_line) => changes.push((
                key.0,
                format!(
                    "changed {}#{} ({})\n{}",
                    key.0,
                    key.1,
                    old_line.split('\t').nth(2).unwrap_or("?"),
                    changed_columns(old_line, new_line)
                ),
            )),
            None => changes.push((
                key.0,
                format!("removed {}#{}\n  - {old_line}", key.0, key.1),
            )),
        }
    }
    for (key, new_line) in &new {
        if !old.contains_key(key) {
            changes.push((key.0, format!("added {}#{}\n  + {new_line}", key.0, key.1)));
        }
    }
    if changes.is_empty() {
        // Pairing by (graph, equation) cannot see order: the same lines in another order are a difference.
        return if expected == actual {
            String::new()
        } else {
            "every equation line is unchanged but the lines are in a different order".to_string()
        };
    }
    let mut per_graph: BTreeMap<&str, usize> = BTreeMap::new();
    for (graph, _) in &changes {
        *per_graph.entry(graph).or_default() += 1;
    }
    let total = changes.len();
    let mut out = format!(
        "{total} changed equations: {}\n",
        per_graph
            .iter()
            .map(|(graph, n)| format!("{graph} ({n})"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let shown: Vec<String> = changes
        .into_iter()
        .take(DIFF_LIMIT)
        .map(|(_, text)| text)
        .collect();
    out.push_str(&shown.join("\n"));
    if total > DIFF_LIMIT {
        write!(out, "\n... and {} more", total - DIFF_LIMIT).expect("write to a String");
    }
    out
}

/// Compare (or, under [`REGEN_ENV`], rewrite) one target's checked-in summary.
fn check_target(reference: &RefTarget) {
    let actual = render(&registry(), &reference.target);
    let target = reference.file;
    let path = expected_path(target);
    if std::env::var_os(REGEN_ENV).is_some() {
        std::fs::create_dir_all(path.parent().expect("expected file has a parent"))
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
        "{target}: the plan summary differs from {}\n{changes}\n\
         If the plan change is intended, run `just test-plan-summary-regen` and list every changed \
         line in the card's landing note.",
        path.display()
    );
}

#[test]
fn spirv_vulkan_plan_summary_matches_the_checked_in_file() {
    check_target(&reference_targets()[0]);
}

#[test]
fn amd_gcn_plan_summary_matches_the_checked_in_file() {
    check_target(&reference_targets()[1]);
}

#[test]
fn nvptx_plan_summary_matches_the_checked_in_file() {
    check_target(&reference_targets()[2]);
}

/// The plan kinds `poot_executor::Engine::load` accepts: every kind a single-device `compile` may emit.
/// An exhaustive match (no wildcard), so a new `Plan` variant must be classified here before this
/// file compiles.
fn is_loadable(plan: &Plan) -> bool {
    match plan {
        Plan::Dispatches(_) | Plan::Alias(_) | Plan::View { .. } => true,
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

/// Rendering the same corpus twice, from freshly traced graphs, gives the same text: nothing in the
/// summary depends on hash-map order, an address or a timestamp. Without this a checked-in file could
/// pass once and fail on the next run.
#[test]
fn plan_summary_rendering_is_deterministic() {
    let registry = registry();
    for reference in reference_targets() {
        let first = render(&registry, &reference.target);
        let second = render(&registry, &reference.target);
        assert!(
            first == second,
            "{}: two renders of the same corpus differ\n{}",
            reference.column,
            diff(&first, &second)
        );
    }
}

/// Card 557 SC-003: a tracer states primitives and `ops::` compositions only. Every corpus graph, as
/// traced and before `compile`, holds no compiler-fusion or schedule-choice composite (`OpClass::
/// Composite`: the flash ops, `Rope`, `ContractionEpilogue` and the fused regions; the packed and dense claims
/// are `Primitive` until Card 629 gives them a decomposition). The one authorized bounded semantic source
/// carrier is the recurrence `Scan` (it states a recurrence and its bounded expansion, not a schedule). Then `compile`'s passes do form composites, so
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
            if matches!(eqn.op, poot_graph_ir::OpKind::Scan(_)) {
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
