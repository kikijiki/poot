//! The real-dimension launch census: a committed record of every contraction and attention schedule `compile`
//! plans for the reference models at their real dimensions, on each reference target.
//!
//! The plan digests (`plan_summary.rs`) are taken over fixtures a few dozen wide, where the Gemv cols rule, the fill
//! factor and the register budget mostly do not bind: a digest there can stay equal while a real plan moves. This
//! census is the evidence a launch-geometry change is judged by. `tests/launch_census/real_dims.tsv` has one row per
//! distinct request: the model, step shape, target, op and the request's `kind` (family, form and, for a dense Gemv,
//! its schedule and shape), how many equations of the program plan exactly that request, and the request's
//! canonical text, which states every field of its schedule (width, cols, unroll, tiling lanes, output tile, micro
//! tile, depth, key block, grid). A change to a figure the planner reads therefore changes a row.
//!
//! A card that changes plans by design regenerates the file with `just test-launch-census-regen`, reads `git diff`
//! of it and lists every changed row in its landing note. A card that must not change plans leaves the diff empty
//! on the targets it names.
//!
//! The census answers questions of the form "which request does this contraction reach": `rg dense_gemv` over the
//! file lists every equation the dense decode Gemv serves.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use poot_graph_plan::KernelChoice;
use poot_kernelgen::KernelRequest;

#[path = "support/corpus.rs"]
mod corpus;
#[path = "support/real_dims.rs"]
mod real_dims;
#[path = "support/targets.rs"]
mod targets;

use targets::{RefTarget, reference_targets};

/// The environment variable that turns the comparison into a rewrite of the committed file. The
/// `test-launch-census-regen` recipe sets it.
const REGEN_ENV: &str = "POOT_LAUNCH_CENSUS_REGEN";

const HEADER: &str = "model\tshape\ttarget\top\tkind\tequations\trequest\n";

/// The request of `choice` when it is a contraction or an attention schedule: the equations this census records.
fn scheduled(choice: &KernelChoice) -> Option<&KernelRequest> {
    let (KernelChoice::Generated(request) | KernelChoice::Serialized { request, .. }) = choice
    else {
        return None;
    };
    match request {
        KernelRequest::Contraction(_)
        | KernelRequest::Attention(_)
        | KernelRequest::AttentionRows(_)
        | KernelRequest::AttentionCombine(_) => Some(request),
        _ => None,
    }
}

/// One census row's identity; equal rows are counted, not repeated (a 56-layer model plans the same request
/// once per layer).
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Row {
    model: String,
    shape: &'static str,
    target: &'static str,
    op: String,
    kind: String,
    request: String,
}

/// The census text over every real-dimension model, step shape and reference target. A graph `compile` refuses
/// at these dimensions has no row (the supported-models tables record the refusal).
fn render_census() -> String {
    let root = poot_test_util::repo_root();
    let options = targets::options();
    let mut rows: BTreeMap<Row, usize> = BTreeMap::new();
    for model in real_dims::real_models(&root) {
        for &shape in model.shapes() {
            let graph = model.trace(shape);
            for RefTarget { id, target, .. } in reference_targets() {
                let Ok(program) = targets::compile_production(&graph, &target, &options) else {
                    continue;
                };
                for (_, _, stage) in program.stages() {
                    for (eqn, _) in stage.planned() {
                        let choice = stage.kernel_choice(eqn);
                        let Some(request) = scheduled(choice) else {
                            continue;
                        };
                        let row = Row {
                            model: model.label.clone(),
                            shape: shape.name,
                            target: id,
                            op: eqn.op.name().to_string(),
                            kind: request.display_label(),
                            request: choice.canonical_text(),
                        };
                        *rows.entry(row).or_default() += 1;
                    }
                }
            }
        }
    }
    let mut out = String::from(HEADER);
    for (row, equations) in rows {
        // Every free-text cell is a Debug or label string: none holds a tab or a newline.
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{equations}\t{}",
            row.model, row.shape, row.target, row.op, row.kind, row.request
        )
        .expect("write to a String");
    }
    out
}

fn census_path() -> PathBuf {
    poot_test_util::manifest_dir()
        .join("tests")
        .join("launch_census")
        .join("real_dims.tsv")
}

/// The first rows that differ between two census texts, one `- old` / `+ new` pair each, capped so a sweeping
/// change stays readable (the full diff is `git diff` after the regen recipe).
fn diff(expected: &str, actual: &str) -> String {
    const SHOWN: usize = 12;
    let (old, new): (Vec<&str>, Vec<&str>) = (expected.lines().collect(), actual.lines().collect());
    let removed = old.iter().filter(|line| !new.contains(line));
    let added = new.iter().filter(|line| !old.contains(line));
    let mut out = String::new();
    let (mut removed_count, mut added_count) = (0usize, 0usize);
    for line in removed {
        removed_count += 1;
        if removed_count <= SHOWN {
            writeln!(out, "- {}", line.replace('\t', " | ")).expect("write to a String");
        }
    }
    for line in added {
        added_count += 1;
        if added_count <= SHOWN {
            writeln!(out, "+ {}", line.replace('\t', " | ")).expect("write to a String");
        }
    }
    format!("{removed_count} rows removed, {added_count} rows added\n{out}")
}

/// The committed census equals what `compile` plans now. Mutation: change a figure the planner reads for a
/// real-dimension contraction (for example `GEMV_WIDTH` in `poot-graph-plan`'s predicates, or the Gemv `cols` rule):
/// the rows that carry it change and the failure names them. Without this row the census is a stale report.
#[test]
fn the_launch_census_matches_the_checked_in_file() {
    let actual = render_census();
    let path = census_path();
    if std::env::var_os(REGEN_ENV).is_some() {
        std::fs::create_dir_all(path.parent().expect("the census file has a parent"))
            .expect("create tests/launch_census");
        std::fs::write(&path, &actual).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e}\nregenerate with `just test-launch-census-regen`",
            path.display()
        )
    });
    assert!(
        expected == actual,
        "the launch census differs from {}\n{}\n\
         If the plan change is intended, run `just test-launch-census-regen` and list every changed row in \
         the card's landing note.",
        path.display(),
        diff(&expected, &actual)
    );
}
