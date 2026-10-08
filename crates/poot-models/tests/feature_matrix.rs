//! The generated feature matrix (ADR-0104 decision 4: support is decided by planning).
//!
//! A model runs on a backend when compiling its graph succeeds, so the support table is not written by
//! hand: this test compiles the fixture graphs of every registered family, and a reference model over
//! every weight format, for every reference target, and renders what `compile` said. A cell is
//! `Plans`, or the typed refusal (the op and the capability the target lacks). The measured table joins
//! the benchmark tool's recorded rows (`benchmarks/results/*/results.jsonl`) to the family the runner
//! resolved, so a refusal the tool recorded appears with its reason.
//!
//! The page `website/docs/reference/feature-matrix.md` keeps its prose; this test owns only the text
//! between [`BEGIN`] and [`END`]. The test fails when the page differs from what the generator renders;
//! `just feature-matrix-regen` rewrites it. The output is sorted, carries no timestamp and escapes `|`.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use poot_graph_ir::{Graph, ValidationOutputs};
use poot_graph_plan::{CompileError, Target, compile};
use poot_models::model::{LogitRows, Phase};
use poot_models::registry::{ConfigSource, FamilyEntry, Registry};
use poot_quant::format::{GroupMap, ScaleEncoding, WeightFormat};
use poot_quant::weights::{WeightEntry, WeightStore};

#[path = "support/corpus.rs"]
mod corpus;
#[path = "support/targets.rs"]
mod targets;

use corpus::{CAP, Fixtured, fixtured};
use targets::{RefTarget, reference_targets};

const BEGIN: &str = "<!-- generated:feature-matrix:begin -->";
const END: &str = "<!-- generated:feature-matrix:end -->";
const REGEN_ENV: &str = "POOT_FEATURE_MATRIX_REGEN";

/// What compiling one family's (or one weight format's) fixture graphs for one target said.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Cell {
    /// Every fixture graph plans.
    Plans,
    /// The planner refused an equation: the shape being compiled, the op and the capability the target
    /// lacks.
    Refused {
        shape: &'static str,
        op: String,
        missing: String,
    },
    /// `compile` failed for a reason that is not a capability refusal (numerics, legalize, limits).
    Rejected { shape: &'static str, reason: String },
}

/// One recorded benchmark row, joined to the matrix by its resolved family and backend.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Measured {
    family: String,
    backend: String,
    status: String,
    reason: Option<String>,
    /// The result directory the row came from.
    run: String,
}

/// Compile every `(shape, graph)` for `target`; the first graph `compile` does not accept is the cell.
fn cell(graphs: &[(&'static str, Graph<ValidationOutputs>)], target: &Target) -> Cell {
    let options = targets::options();
    for (shape, graph) in graphs {
        let Err(error) = compile(graph, target, &options) else {
            continue;
        };
        return match &error {
            CompileError::Plan(plan) => match plan.as_ref() {
                poot_graph_plan::PlanError::Refused(refusal) => Cell::Refused {
                    shape,
                    op: refusal.op.name().to_string(),
                    missing: refusal.missing.to_string(),
                },
                other => Cell::Rejected {
                    shape,
                    reason: other.to_string(),
                },
            },
            other => Cell::Rejected {
                shape,
                reason: other.to_string(),
            },
        };
    }
    Cell::Plans
}

// ---- table (a): families x targets ------------------------------------------------------------

/// The fixture's graphs, one per shape, traced once and compiled for every target.
fn family_graphs(fixtured: &Fixtured) -> Vec<(&'static str, Graph<ValidationOutputs>)> {
    fixtured
        .shapes()
        .iter()
        .map(|&shape| (shape.name, fixtured.trace(shape)))
        .collect()
}

// ---- table (b): weight formats x targets ------------------------------------------------------

/// The packed formats the weight-format table renders: one representative per packed variant of
/// [`WeightFormat`], with the parameters that change the lowering (a GPTQ group map, a scale encoding).
/// The dense dtypes are not packed payloads; the table's dense row is the reference checkpoint itself.
fn weight_formats() -> Vec<WeightFormat> {
    let group = std::num::NonZeroUsize::new(128).expect("nonzero");
    let act_order = std::num::NonZeroUsize::new(2).expect("nonzero");
    vec![
        WeightFormat::Q4_0,
        WeightFormat::Q4_1,
        WeightFormat::Q5_0,
        WeightFormat::Q5_1,
        WeightFormat::Q8_0,
        WeightFormat::Q2_K,
        WeightFormat::Q3_K,
        WeightFormat::Q4_K,
        WeightFormat::Q5_K,
        WeightFormat::Q6_K,
        WeightFormat::Iq4_Nl,
        WeightFormat::Iq4_Xs,
        WeightFormat::Mxfp4,
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::Bf16,
        },
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::F32,
        },
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16,
        },
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        WeightFormat::E2m1Row32,
        WeightFormat::Gptq {
            groups: GroupMap::Contiguous { size: group },
        },
        WeightFormat::Gptq {
            groups: GroupMap::Indexed { groups: act_order },
        },
        WeightFormat::Awq { group_size: group },
    ]
}

/// How the table names a stored scale encoding.
fn scale_name(scale: ScaleEncoding) -> &'static str {
    match scale {
        ScaleEncoding::F32 => "F32",
        ScaleEncoding::Bf16 => "BF16",
        ScaleEncoding::E8m0 => "E8M0",
    }
}

/// How the table names a format. An exhaustive match: a new [`WeightFormat`] variant does not compile
/// here until it has a name, which is the reminder to add it to [`weight_formats`].
fn format_name(format: WeightFormat) -> String {
    match format {
        WeightFormat::F32 | WeightFormat::F16 | WeightFormat::Bf16 => "dense".into(),
        WeightFormat::Q4_0 => "GGUF Q4_0".into(),
        WeightFormat::Q4_1 => "GGUF Q4_1".into(),
        WeightFormat::Q5_0 => "GGUF Q5_0".into(),
        WeightFormat::Q5_1 => "GGUF Q5_1".into(),
        WeightFormat::Q8_0 => "GGUF Q8_0".into(),
        WeightFormat::Q2_K => "GGUF Q2_K".into(),
        WeightFormat::Q3_K => "GGUF Q3_K".into(),
        WeightFormat::Q4_K => "GGUF Q4_K".into(),
        WeightFormat::Q5_K => "GGUF Q5_K".into(),
        WeightFormat::Q6_K => "GGUF Q6_K".into(),
        WeightFormat::Iq4_Nl => "GGUF IQ4_NL".into(),
        WeightFormat::Iq4_Xs => "GGUF IQ4_XS".into(),
        WeightFormat::Mxfp4 => "GGUF MXFP4".into(),
        WeightFormat::E4m3PerChannel { scale } => {
            format!("FP8 E4M3, per-channel {} scale", scale_name(scale))
        }
        WeightFormat::E4m3Block128 { scale } => {
            format!("FP8 E4M3, 128x128 block {} scale", scale_name(scale))
        }
        WeightFormat::E2m1Row32 => "E2M1 (FP4), per-32 E8M0 scale".into(),
        WeightFormat::Gptq {
            groups: GroupMap::Contiguous { .. },
        } => "GPTQ int4".into(),
        WeightFormat::Gptq {
            groups: GroupMap::Indexed { .. },
        } => "GPTQ int4, act-order".into(),
        WeightFormat::Awq { .. } => "AWQ int4".into(),
    }
}

/// The reference dense model the weight formats are bound into: a one-layer Qwen2 checkpoint whose
/// projection widths (256, and 128 for the key and value heads) every format's block admits.
fn reference_dense() -> poot_executor_parity::dense::Dense {
    use poot_executor_parity::dense::{Dense, Family};
    Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(256, 256, 1)
        .heads(4, 2)
        .head_dim(64)
        .max_positions(32)
        .with("rms_norm_eps", 1e-5)
}

/// The reference dense checkpoint with every projection stored as a `format` payload from
/// `poot_test_util::packed::random_payload` (`None`: left as the dense BF16 the checkpoint holds), traced
/// for decode and prefill.
fn format_graphs(format: Option<WeightFormat>) -> Vec<(&'static str, Graph<ValidationOutputs>)> {
    use poot_executor_parity::weight_map::Projections;
    let dense = reference_dense();
    let base = dense.store(Projections::Bf16, poot_tensor::DType::F32, None);
    let mut store = WeightStore::builder();
    for (seed, (key, entry)) in base.iter().enumerate() {
        let entry = match format {
            Some(format) if key.as_str().ends_with("_proj.weight") => {
                let shape = entry.shape();
                WeightEntry::Packed(Arc::new(poot_test_util::packed::random_payload(
                    format,
                    [shape[0], shape[1]],
                    seed as u64 + 1,
                )))
            }
            _ => entry.clone(),
        };
        store
            .insert(key.clone(), entry)
            .expect("the base store's keys are unique");
    }
    let store = store.build();
    let config = dense.config();
    let raw = poot_models::registry::RawConfig::HfJson {
        config: &config,
        generation: None,
    };
    let model = Registry::builtin()
        .expect("the builtin registry")
        .build(&raw, &store)
        .unwrap_or_else(|e| panic!("{format:?}: {e}"));
    let trace = |phase, tokens, capacity| {
        model
            .trace(
                phase,
                poot_executor_parity::dense::step(1, tokens, capacity, LogitRows::Last),
            )
            .unwrap_or_else(|e| panic!("{format:?}: {e}"))
    };
    vec![
        ("decode", trace(Phase::Decode, 1, CAP)),
        ("prefill", trace(Phase::Prefill, 4, 4)),
    ]
}

// ---- table (c): the measured rows -------------------------------------------------------------

/// The latest poot row per (family, backend) of every `*/results.jsonl` under `results`, with the
/// refusal reason as the tool recorded it. Run directories sort by name, which starts with the date, so
/// a later directory replaces an earlier row. A row with no resolved `family` (a row recorded before the
/// runner reported it, or a refusal that came before the registry resolved one) has no family to join.
fn latest_measured(results: &Path) -> Vec<Measured> {
    let mut runs: Vec<PathBuf> = match std::fs::read_dir(results) {
        Ok(dir) => dir
            .map(|entry| entry.expect("a results entry").path())
            .filter(|path| path.join("results.jsonl").is_file())
            .collect(),
        Err(_) => Vec::new(),
    };
    runs.sort();
    let mut latest: BTreeMap<(String, String), Measured> = BTreeMap::new();
    for run in runs {
        let name = run
            .file_name()
            .expect("a run directory has a name")
            .to_string_lossy()
            .into_owned();
        let text = std::fs::read_to_string(run.join("results.jsonl"))
            .unwrap_or_else(|e| panic!("read {}: {e}", run.display()));
        for row in measured_rows(&text, &name) {
            latest.insert((row.family.clone(), row.backend.clone()), row);
        }
    }
    latest.into_values().collect()
}

/// The joinable rows of one `results.jsonl`.
fn measured_rows(jsonl: &str, run: &str) -> Vec<Measured> {
    jsonl
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let row: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("{run}: a results row is not JSON: {e}"));
            let text = |key: &str| row.get(key).and_then(|v| v.as_str()).map(str::to_owned);
            if text("framework")? != "poot" {
                return None;
            }
            let status = text("status")?;
            if status != "ok" && status != "unsupported" {
                return None;
            }
            Some(Measured {
                family: text("family")?,
                backend: text("backend")?,
                status,
                reason: text("reason"),
                run: run.to_owned(),
            })
        })
        .collect()
}

// ---- rendering --------------------------------------------------------------------------------

/// Text safe in a table cell: one line, no `|`, ASCII.
fn plain(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '|' => '/',
            '\n' | '\r' | '\t' => ' ',
            c if c.is_ascii() => c,
            _ => '?',
        })
        .collect()
}

/// Inline code in a table cell: backticks cannot nest, so they become quotes.
fn code(text: &str) -> String {
    format!("`{}`", plain(text).replace('`', "'"))
}

fn badge(kind: &str, label: &str) -> String {
    format!("<span class=\"matrix-status matrix-status--{kind}\">{label}</span>")
}

fn render_cell(cell: &Cell) -> String {
    match cell {
        Cell::Plans => badge("yes", "Plans"),
        Cell::Refused { shape, op, missing } => format!(
            "{} {shape}: {} {}",
            badge("no", "Refused"),
            code(op),
            code(missing)
        ),
        Cell::Rejected { shape, reason } => {
            format!("{} {shape}: {}", badge("no", "Rejected"), code(reason))
        }
    }
}

fn header(first: &str, targets: &[RefTarget]) -> String {
    let columns: Vec<&str> = targets.iter().map(|t| t.column).collect();
    format!(
        "| {first} | {} |\n| --- |{}\n",
        columns.join(" | "),
        " --- |".repeat(columns.len())
    )
}

/// The generated region: three tables and the sentences that say how to read them.
fn render(registry: &Registry, targets: &[RefTarget], measured: &[Measured]) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "This region is generated by `just feature-matrix-regen` from what `compile` says about each fixture \
         graph, never edited by hand. See [Reading the matrix](#reading-the-matrix)."
    )
    .unwrap();

    writeln!(out, "\n### Model families\n").unwrap();
    writeln!(
        out,
        "Every registered family's fixture graphs (decode and prefill for a decoder, one forward for an \
         encoder) compiled for each target. A family joins this table by being registered with a fixture.\n"
    )
    .unwrap();
    out.push_str(&header("Family", targets));
    let mut families = fixtured(registry);
    families.sort_by(|a, b| a.label.cmp(&b.label));
    for family in &families {
        let graphs = family_graphs(family);
        let cells: Vec<String> = targets
            .iter()
            .map(|t| render_cell(&cell(&graphs, &t.target)))
            .collect();
        writeln!(out, "| {} | {} |", code(&family.label), cells.join(" | ")).unwrap();
    }

    writeln!(out, "\n### Weight formats\n").unwrap();
    writeln!(
        out,
        "A one-layer dense decoder whose projections are stored in each format (a random admitted payload, \
         bound as the stored bytes), compiled for each target.\n"
    )
    .unwrap();
    out.push_str(&header("Format", targets));
    let mut formats: Vec<(String, Option<WeightFormat>)> = weight_formats()
        .into_iter()
        .map(|format| (format_name(format), Some(format)))
        .collect();
    formats.push(("BF16 (dense)".to_string(), None));
    formats.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, format) in formats {
        let graphs = format_graphs(format);
        let cells: Vec<String> = targets
            .iter()
            .map(|t| render_cell(&cell(&graphs, &t.target)))
            .collect();
        writeln!(out, "| {} | {} |", plain(&name), cells.join(" | ")).unwrap();
    }

    writeln!(out, "\n### Measured\n").unwrap();
    if measured.is_empty() {
        writeln!(
            out,
            "No stored benchmark row names a resolved family yet. A row appears here once a benchmark run \
             records one."
        )
        .unwrap();
    } else {
        writeln!(
            out,
            "The latest benchmark row per family and backend, from the stored results. A refusal carries the \
             reason the tool reported.\n"
        )
        .unwrap();
        writeln!(
            out,
            "| Family | Backend | Result | Run |\n| --- | --- | --- | --- |"
        )
        .unwrap();
        let mut rows: Vec<&Measured> = measured.iter().collect();
        rows.sort_by(|a, b| (&a.family, &a.backend).cmp(&(&b.family, &b.backend)));
        for row in rows {
            let result = match (row.status.as_str(), &row.reason) {
                ("unsupported", Some(reason)) => {
                    format!("{} {}", badge("no", "Refused"), code(reason))
                }
                ("unsupported", None) => badge("no", "Refused"),
                _ => badge("yes", "Measured"),
            };
            writeln!(
                out,
                "| {} | {} | {} | {} |",
                code(&row.family),
                plain(&row.backend),
                result,
                code(&row.run)
            )
            .unwrap();
        }
    }
    assert_no_card_numbers(&out);
    out
}

/// The site never names an internal card number; the generator checks its own output the way
/// `just site-check` checks the page.
fn assert_no_card_numbers(text: &str) {
    for (i, line) in text.lines().enumerate() {
        let lower = line.to_ascii_lowercase();
        for word in ["card ", "cards ", "poot-"] {
            for (at, _) in lower.match_indices(word) {
                let rest = &lower[at + word.len()..];
                let digits = rest.chars().take_while(char::is_ascii_digit).count();
                assert!(
                    digits < 3,
                    "the generated matrix names a card number on line {i}: {line}"
                );
            }
        }
    }
}

// ---- the page ---------------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    // Read at run time, not baked in with `env!`: a shared compile cache can serve one worktree's
    // object code to another, and a baked-in path would then name the wrong checkout.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set by cargo for test binaries");
    Path::new(&manifest_dir).join("../..")
}

fn page_path() -> PathBuf {
    workspace_root().join("website/docs/reference/feature-matrix.md")
}

/// `page` with the text between the markers replaced by `generated`.
fn splice(page: &str, generated: &str) -> String {
    let begin = page
        .find(BEGIN)
        .unwrap_or_else(|| panic!("the page has no {BEGIN} marker"));
    let end = page
        .find(END)
        .unwrap_or_else(|| panic!("the page has no {END} marker"));
    assert!(begin < end, "the markers are out of order");
    format!(
        "{}{BEGIN}\n\n{}\n{END}{}",
        &page[..begin],
        generated.trim_end(),
        &page[end + END.len()..]
    )
}

fn registry() -> Registry {
    Registry::builtin().expect("the builtin registry")
}

fn measured_results() -> Vec<Measured> {
    latest_measured(&workspace_root().join("benchmarks/results"))
}

/// The checked-in page equals what the generator renders now. Mutation: render from the committed page
/// instead of compiling (or cache the previous matrix): a planner change leaves the page unchanged and
/// SC-002's refusal test goes red.
#[test]
fn the_page_is_what_the_generator_renders() {
    let rendered = render(&registry(), &reference_targets(), &measured_results());
    let path = page_path();
    let page =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let expected = splice(&page, &rendered);
    if std::env::var_os(REGEN_ENV).is_some() {
        std::fs::write(&path, expected).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        return;
    }
    assert!(
        page == expected,
        "{} differs from the generated matrix; run `just feature-matrix-regen`",
        path.display()
    );
}

/// SC-001: rendering twice gives the same bytes. Mutation: iterate a `HashMap` in `render`, or add a
/// timestamp: the second render differs.
#[test]
fn rendering_twice_gives_identical_bytes() {
    let registry = registry();
    let targets = reference_targets();
    let measured = measured_results();
    let first = render(&registry, &targets, &measured);
    let second = render(&registry, &targets, &measured);
    assert!(first == second, "two renders of the same inputs differ");
}

/// A target whose workgroup memory cannot hold one attention lane.
fn tiny_lds_target() -> RefTarget {
    let mut target = targets::preset("tiny-lds", "-", poot_target::Backend::SpirvVulkan);
    target.target.caps.lds_bytes = 16;
    target
}

/// SC-002: a target whose `DeviceCaps` lacks what an op needs shows the typed refusal in the row, and the
/// same family plans on the unmodified target. Mutation: cache the previous matrix instead of compiling:
/// the tiny-LDS column reads `Plans` and this goes red.
#[test]
fn a_target_lacking_a_capability_shows_the_typed_refusal() {
    let registry = registry();
    let wgpu = &reference_targets()[0];
    let tiny = tiny_lds_target();
    let families = fixtured(&registry);
    let qwen2 = families
        .iter()
        .find(|f| f.label == "qwen2")
        .expect("the qwen2 family is registered");

    let graphs = family_graphs(qwen2);
    assert_eq!(cell(&graphs, &wgpu.target), Cell::Plans);
    let refused = cell(&graphs, &tiny.target);
    let Cell::Refused { shape, op, missing } = &refused else {
        panic!("a 16-byte workgroup memory cell must be a refusal, got {refused:?}");
    };
    assert_eq!(
        (*shape, op.as_str()),
        ("decode", "matmul"),
        "the first equation the planner cannot lower: {missing}"
    );

    let rendered = render(&registry, &[tiny], &[]);
    assert!(
        rendered.contains("Refused") && rendered.contains(missing.as_str()),
        "the rendered row shows the typed refusal ({op}: {missing}):\n{rendered}"
    );
}

fn refusal_row(family: &str, backend: &str, reason: &str) -> String {
    serde_json::json!({
        "framework": "poot",
        "model": "m",
        "precision": "bf16",
        "scenario": "decode-128",
        "status": "unsupported",
        "reason": reason,
        "build_sha": "0123456789abcdef0123456789abcdef01234567",
        "family": family,
        "backend": backend,
    })
    .to_string()
}

/// SC-003: a refusal row the benchmark tool recorded for a family and backend appears in the matrix with
/// its reason, verbatim. Mutation: drop the join (`measured_rows` returns nothing): the reason is missing.
#[test]
fn a_recorded_bench_refusal_appears_with_its_reason() {
    let reason = "no kernel lowers this op for these operand and output dtypes (v41)";
    let rows = measured_rows(&refusal_row("qwen2", "ptx", reason), "2026-10-01-a100-abc");
    assert_eq!(rows.len(), 1, "the runner's refusal row joins");

    let rendered = render(&registry(), &reference_targets()[..1], &rows);
    assert!(
        rendered.contains(reason) && rendered.contains("2026-10-01-a100-abc"),
        "the measured table carries the recorded reason:\n{rendered}"
    );
}

/// The newest run directory wins per (family, backend), and a row with no resolved family has nothing to
/// join.
#[test]
fn the_latest_run_per_family_and_backend_is_the_measured_row() {
    let root = std::env::temp_dir().join(format!("poot-matrix-results-{}", std::process::id()));
    for (run, reason) in [
        ("2026-09-01-old", "old reason"),
        ("2026-10-01-new", "new reason"),
    ] {
        let dir = root.join(run);
        std::fs::create_dir_all(&dir).unwrap();
        let mut text = refusal_row("qwen2", "ptx", reason);
        text.push('\n');
        text.push_str(&refusal_row("qwen2", "rocm", "rocm only"));
        text.push('\n');
        text.push_str(
            &serde_json::json!({"framework": "poot", "status": "unsupported", "reason": "x",
                "family": null, "backend": "ptx"})
            .to_string(),
        );
        std::fs::write(dir.join("results.jsonl"), text).unwrap();
    }
    let rows = latest_measured(&root);
    std::fs::remove_dir_all(&root).unwrap();

    let ptx = rows.iter().find(|r| r.backend == "ptx").expect("a ptx row");
    assert_eq!(
        (ptx.run.as_str(), ptx.reason.as_deref()),
        ("2026-10-01-new", Some("new reason"))
    );
    assert_eq!(
        rows.len(),
        2,
        "one row per family and backend, none for a null family: {rows:?}"
    );
}

/// S6: registering a family with a fixture adds its matrix row with no edit to the generator. Mutation:
/// hard-code the family list in `corpus.rs`: the clone has no row.
#[test]
fn a_registered_family_adds_a_row_with_no_generator_edit() {
    let mut registry = registry();
    let qwen2 = registry
        .entries()
        .iter()
        .find(|e| e.family.as_str() == "qwen2")
        .copied()
        .expect("qwen2 is registered");
    let clone = FamilyEntry {
        family: poot_models::model::FamilyKey::new("poot_fixture_clone"),
        keys: &[(ConfigSource::HfModelType, "poot_fixture_clone")],
        ..qwen2
    };
    registry
        .register(clone)
        .expect("a new family key registers");

    let rendered = render(&registry, &reference_targets()[..1], &[]);
    assert!(
        rendered.contains("`poot_fixture_clone`"),
        "the new family has a row:\n{rendered}"
    );
}

/// The generator names no card number in what it writes (site-check checks the page; this checks the
/// source of the text). Mutation: add a literal card number to a cell: the assertion in `render` fires.
#[test]
fn the_rendered_region_names_no_card_number() {
    let rendered = render(&registry(), &reference_targets()[..1], &[]);
    assert_no_card_numbers(&rendered);
    let result = std::panic::catch_unwind(|| assert_no_card_numbers("| a | see Card 577 |"));
    assert!(result.is_err(), "a card number is caught");
}
