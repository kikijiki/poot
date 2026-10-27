//! The generated supported-models tables (ADR-0104 decision 4: support is decided by planning).
//!
//! A model runs on a backend when compiling its graph succeeds, so the support table is not written by
//! hand: this test compiles the fixture graphs of every registered family, the real-dimension models
//! (`support/real_dims.rs`: the shapes and context a real checkpoint traces, over zero weights), and a
//! reference model over every weight format, for every reference target, and renders what `compile`
//! said. A cell is `Plans`, or the typed refusal (the op or constant and the capability the target
//! lacks). A third table pins one refusal per backend by compiling a real-dimension model against a
//! device whose limit it exceeds. What the planner cannot say, whether a path runs on hardware, comes from
//! the required-case lists of the device lanes (`scripts/required-device-cases/`), rendered as
//! `Required` with the number of cases, so a feature the PTX lane requires is not reported as absent.
//! A cell holds a short status; the planner's message is written once in a numbered note under the table
//! and again as the badge's tooltip.
//!
//! The page `website/docs/develop/supported-models.md` keeps its prose; this test owns only the text
//! between [`BEGIN`] and [`END`]. The test fails when the page differs from what the generator renders;
//! `just supported-models-regen` rewrites it. The output is sorted, carries no timestamp and escapes `|`.
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use poot_graph_ir::{BinOp, Builder, Graph, TensorType, UnOp};
use poot_graph_plan::{CompileError, CompileOptions, LegalizeError, NpuTarget, Target, compile};
use poot_models::model::{LogitRows, StepSpec};
use poot_models::registry::Registry;
use poot_quant::format::{GroupMap, ScaleEncoding, WeightFormat};
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_target::aie::RegionLimits;
use poot_target::{AieCaps, AieTargetId, Queried};

#[path = "support/corpus.rs"]
mod corpus;
#[path = "support/real_dims.rs"]
mod real_dims;
#[path = "support/targets.rs"]
mod targets;

use corpus::{CAP, Fixtured, fixtured};
use poot_graph_plan::test_support::gpu_caps_mut;
use targets::{RefTarget, reference_targets};

const BEGIN: &str = "<!-- generated:supported-models:begin -->";
const END: &str = "<!-- generated:supported-models:end -->";
const REGEN_ENV: &str = "POOT_SUPPORTED_MODELS_REGEN";

/// What compiling one family's (or one weight format's) fixture graphs for one target said.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Cell {
    /// Every fixture graph plans.
    Plans,
    /// The planner refused: the shape being compiled, what it refused (an op, or a constant) and the
    /// capability or limit the target lacks.
    Refused {
        shape: &'static str,
        subject: String,
        missing: String,
    },
    /// `compile` failed for a reason that is not a capability refusal (numerics, legalize, limits).
    Rejected { shape: &'static str, reason: String },
}

/// Compile every `(shape, graph)` for `target`; the first graph `compile` does not accept is the cell.
/// A typed refusal is a [`Cell::Refused`]: the planner's own [`poot_graph_plan::Refusal`], or the
/// legalizer's refusal of a constant over the device's single-buffer limit. Any other failure is a
/// [`Cell::Rejected`].
fn cell(graphs: &[(&'static str, Graph)], target: &Target) -> Cell {
    let options = CompileOptions {
        npu: Some(poot_npu::npu_lowering()),
        ..targets::options()
    };
    for (shape, graph) in graphs {
        let Err(error) = compile(graph, target, &options) else {
            continue;
        };
        let shape = *shape;
        return match &error {
            CompileError::Plan(plan) => match plan.as_ref() {
                poot_graph_plan::PlanError::Refused(refusal) => Cell::Refused {
                    shape,
                    subject: refusal.op.name().to_string(),
                    missing: refusal.missing.to_string(),
                },
                other => Cell::Rejected {
                    shape,
                    reason: other.to_string(),
                },
            },
            CompileError::Legalize(legalize) => match legalize.as_ref() {
                LegalizeError::OversizedConstant { name, bytes, limit } => Cell::Refused {
                    shape,
                    subject: format!("constant {name}"),
                    missing: format!(
                        "a buffer of {bytes} bytes (the device's single-buffer limit is {limit})"
                    ),
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
fn family_graphs(fixtured: &Fixtured) -> Vec<(&'static str, Graph)> {
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
        WeightFormat::Iq2_Xs,
        WeightFormat::Iq3_Xxs,
        WeightFormat::Iq3_S,
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
        WeightFormat::Iq2_Xs => "GGUF IQ2_XS".into(),
        WeightFormat::Iq3_Xxs => "GGUF IQ3_XXS".into(),
        WeightFormat::Iq3_S => "GGUF IQ3_S".into(),
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
fn reference_dense() -> poot_models::test_support::dense::Dense {
    use poot_models::test_support::dense::{Dense, Family};
    Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(256, 256, 1)
        .heads(4, 2)
        .head_dim(64)
        .max_positions(32)
        .with("rms_norm_eps", 1e-5)
}

/// The reference dense checkpoint with every projection stored as a `format` payload from
/// `poot_quant::test_support::packed::random_payload` (`None`: left as the dense BF16 the checkpoint holds), traced
/// for decode and prefill.
fn format_graphs(format: Option<WeightFormat>) -> Vec<(&'static str, Graph)> {
    use poot_models::test_support::weight_map::Projections;
    let dense = reference_dense();
    let base = dense.store(Projections::Bf16, poot_tensor::DType::F32, None);
    let mut store = WeightStore::builder();
    for (seed, (key, entry)) in base.iter().enumerate() {
        let entry = match format {
            Some(format) if key.as_str().ends_with("_proj.weight") => {
                let shape = entry.shape();
                WeightEntry::Packed(Arc::new(poot_quant::test_support::packed::random_payload(
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
        preprocessor: None,
    };
    let model = Registry::builtin()
        .expect("the builtin registry")
        .build(&raw, &store)
        .unwrap_or_else(|e| panic!("{format:?}: {e}"));
    let trace = |tokens, capacity| {
        model
            .model
            .trace(
                &StepSpec::step(poot_models::test_support::dense::step(
                    1,
                    tokens,
                    capacity,
                    LogitRows::Last,
                )),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap_or_else(|e| panic!("{format:?}: {e}"))
    };
    vec![("decode", trace(1, CAP)), ("prefill", trace(4, 4))]
}

// ---- table (b2): the pinned refusals ----------------------------------------------------------

/// A refusal pinned on every backend: a model against a device limit the model exceeds, so the matrix
/// always shows what a typed refusal looks like on that backend, rendered from the planner's own words.
struct Pin {
    /// The row heading.
    label: &'static str,
    /// The real-dimension model (a [`Fixtured::label`] of `real_dims::real_models`) it compiles.
    model: &'static str,
    /// The limit lowered on the backend's preset caps. Applied to every backend alike.
    limit: fn(&mut poot_target::DeviceCaps),
}

/// The single-buffer limit of the wgpu preset (RADV's safe figure): the largest constant a device of
/// that class holds. Mixtral-8x22B's stacked expert gate, 1.5 GiB, is over it. ROCm and PTX report the
/// device's memory as the limit and so plan the model; the pin gives them the same limit.
fn wgpu_buffer_limit() -> u64 {
    poot_test_util::device_caps::default_caps_for(poot_target::Backend::SpirvVulkan)
        .max_buffer_bytes
}

/// A workgroup memory too small for one attention lane or one tile.
const TINY_LDS_BYTES: u32 = 16;

fn pins() -> [Pin; 1] {
    [Pin {
        label: "Mixtral-8x22B, wgpu single-buffer limit",
        model: "Mixtral-8x22B",
        limit: |caps| caps.max_buffer_bytes = wgpu_buffer_limit(),
    }]
}

/// `pin`'s cell on `reference`: the model's graphs compiled for the backend's preset caps with the
/// pin's limit lowered.
fn pin_cell(pin: &Pin, real: &[Fixtured], reference: &RefTarget) -> Cell {
    let model = real
        .iter()
        .find(|m| m.label == pin.model)
        .unwrap_or_else(|| {
            panic!(
                "the pin names {}, which is not a real-dimension model",
                pin.model
            )
        });
    let mut target = reference.target;
    (pin.limit)(gpu_caps_mut(&mut target));
    cell(&family_graphs(model), &target)
}

// ---- table (d): what the device lanes verify ---------------------------------------------------

/// A feature of the library or the server, and the required device-lane cases that exercise it.
struct Feature {
    /// The row heading.
    name: &'static str,
    /// Whether a required case (its binary id and its test name) exercises the feature.
    exercised_by: fn(binary: &str, test: &str) -> bool,
}

fn features() -> [Feature; 11] {
    [
        Feature {
            name: "Served models and server startup (`poot-serve`)",
            exercised_by: |binary, _| binary.starts_with("poot-serve"),
        },
        Feature {
            name: "Paged KV cache",
            exercised_by: |_, test| test.contains("paged"),
        },
        Feature {
            name: "Batched decode",
            exercised_by: |_, test| test.contains("batch"),
        },
        Feature {
            name: "Prefix caching (reuse across requests)",
            exercised_by: |_, test| test.contains("prefix"),
        },
        Feature {
            name: "Speculative decoding: prompt lookup and draft model",
            exercised_by: |binary, test| {
                binary.contains("speculate") || test.contains("speculation")
            },
        },
        Feature {
            name: "Speculative decoding: the model's own MTP layer",
            exercised_by: |_, test| test.contains("mtp"),
        },
        Feature {
            name: "Sampled decoding",
            exercised_by: |_, test| test.contains("sampled"),
        },
        Feature {
            name: "LoRA adapters",
            exercised_by: |_, test| test.contains("lora"),
        },
        Feature {
            name: "Guided decoding (regex, JSON schema, grammar)",
            exercised_by: |_, test| test.contains("guided"),
        },
        Feature {
            name: "Embeddings and prompt embeds",
            exercised_by: |_, test| test.contains("embed"),
        },
        Feature {
            name: "Planning against the attached device's own limits",
            exercised_by: |_, test| test.contains("attached_device"),
        },
    ]
}

/// The cases each device lane requires: `(binary id, test name)` lines of
/// `scripts/required-device-cases/<lane>.txt`, in file order.
struct LaneCases {
    /// `(column heading, cases)` per lane, in the matrix's column order.
    lanes: Vec<(&'static str, Vec<(String, String)>)>,
}

impl LaneCases {
    /// Read the lists of the wgpu, raw Vulkan, ROCm and PTX lanes.
    fn read(root: &Path) -> Self {
        let lanes = [
            ("wgpu", "wgpu"),
            ("Vulkan", "vulkan"),
            ("ROCm", "rocm"),
            ("PTX", "ptx"),
        ]
        .into_iter()
        .map(|(column, lane)| {
            let path = root.join(format!("scripts/required-device-cases/{lane}.txt"));
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            (column, Self::parse(&text))
        })
        .collect();
        Self { lanes }
    }

    /// The `binary<TAB>test` lines of a list; `#` lines are comments.
    fn parse(text: &str) -> Vec<(String, String)> {
        text.lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| {
                let (binary, test) = line
                    .split_once('\t')
                    .unwrap_or_else(|| panic!("a required case is `binary<TAB>test`: {line:?}"));
                (binary.to_string(), test.to_string())
            })
            .collect()
    }

    /// How many cases of `column`'s lane exercise `feature`.
    fn count(&self, column: &str, feature: &Feature) -> usize {
        self.lanes
            .iter()
            .filter(|(lane, _)| *lane == column)
            .flat_map(|(_, cases)| cases)
            .filter(|(binary, test)| (feature.exercised_by)(binary, test))
            .count()
    }
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

/// The long messages of one table, each once. A cell carries a badge, the shape and a link to the note;
/// the note under the table carries the planner's message, and the badge repeats it as a tooltip.
struct Notes {
    /// Makes the note anchors unique across the page's tables.
    table: &'static str,
    notes: Vec<(String, String)>,
}

impl Notes {
    fn new(table: &'static str) -> Self {
        Self {
            table,
            notes: Vec::new(),
        }
    }

    /// The 1-based number of the note for `subject` and `detail`, adding it on first use.
    fn number(&mut self, subject: &str, detail: &str) -> usize {
        let note = (plain(subject), plain(detail));
        let at = self
            .notes
            .iter()
            .position(|n| *n == note)
            .unwrap_or_else(|| {
                self.notes.push(note);
                self.notes.len() - 1
            });
        at + 1
    }

    /// A refused cell: a focusable badge whose tooltip is the message, the shape, and the note's link.
    fn refused(&mut self, kind: &str, shape: &str, subject: &str, detail: &str) -> String {
        let n = self.number(subject, detail);
        let tip = html(&format!("{subject} {detail}"));
        format!(
            "<span class=\"matrix-status matrix-status--no\" tabindex=\"0\" title=\"{tip}\">{kind}</span> \
             {shape}<sup><a href=\"#note-{}-{n}\">{n}</a></sup>",
            self.table
        )
    }

    /// A numbered list under the table, empty when no cell refused.
    fn render(&self) -> String {
        if self.notes.is_empty() {
            return String::new();
        }
        let items: Vec<String> = self
            .notes
            .iter()
            .enumerate()
            .map(|(i, (subject, detail))| {
                format!(
                    "<li id=\"note-{}-{}\"><code>{}</code> <code>{}</code></li>\n",
                    self.table,
                    i + 1,
                    html(subject),
                    html(detail)
                )
            })
            .collect();
        format!("\n<ol>\n{}</ol>\n", items.join(""))
    }
}

/// Text safe inside HTML in the page (MDX): markup and braces escaped.
fn html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('{', "&#123;")
        .replace('}', "&#125;")
}

fn render_cell(cell: &Cell, notes: &mut Notes) -> String {
    match cell {
        Cell::Plans => badge("yes", "Plans"),
        Cell::Refused {
            shape,
            subject,
            missing,
        } => notes.refused("Refused", shape, subject, missing),
        Cell::Rejected { shape, reason } => {
            notes.refused("Rejected", shape, "compile failed:", reason)
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

/// The generated region: the tables and the sentences that say how to read them.
fn render(
    registry: &Registry,
    targets: &[RefTarget],
    real: &[Fixtured],
    lanes: &LaneCases,
) -> String {
    let sections = [
        families_table(registry, targets),
        real_table(real, targets),
        refusals_table(real, targets),
        formats_table(targets),
        lanes_table(lanes, targets),
        npu_table(),
    ];
    let out = format!(
        "This region is generated by `just supported-models-regen`, never edited by hand. It comes from what \
         `compile` says about each graph and from the required cases of the device lanes. See \
         [Reading the tables](#reading-the-tables).\n\n{}",
        sections.join("\n")
    );
    assert_no_card_numbers(&out);
    out
}

/// Every registered family's fixture graphs on each target, at the fixture's small dimensions.
fn families_table(registry: &Registry, targets: &[RefTarget]) -> String {
    let mut out = String::new();
    writeln!(out, "### Model families\n").unwrap();
    writeln!(
        out,
        "Every registered family's fixture graphs, compiled for each target at the fixture's small \
         dimensions. A decoder has a decode and a prefill graph; an encoder has one forward. A family joins \
         this table by having a row in the registry's `SHIPPED` slice and a checkpoint in the fixture \
         table.\n"
    )
    .unwrap();
    out.push_str(&header("Family", targets));
    let mut families = fixtured(registry);
    families.sort_by(|a, b| a.label.cmp(&b.label));
    let mut notes = Notes::new("families");
    for family in &families {
        writeln!(
            out,
            "{}",
            graph_row(&code(&family.label), family, targets, &mut notes)
        )
        .unwrap();
    }
    out.push_str(&notes.render());
    out
}

/// The real-dimension models on each target's preset device limits.
fn real_table(real: &[Fixtured], targets: &[RefTarget]) -> String {
    let mut out = String::new();
    writeln!(out, "### Real dimensions\n").unwrap();
    writeln!(
        out,
        "The graphs of reference models at their real dimensions, from the model's `config.json` or published \
         sizes. They cover decode and prefill over the whole context, and prefill into a full-context cache. \
         Each is compiled for each target's preset device limits. The weights are zeros: the planner reads \
         shapes, types and limits, not values.\n"
    )
    .unwrap();
    out.push_str(&header("Model", targets));
    let mut notes = Notes::new("real");
    for model in real {
        writeln!(
            out,
            "{}",
            graph_row(&code(&model.label), model, targets, &mut notes)
        )
        .unwrap();
    }
    out.push_str(&notes.render());
    out
}

/// One pinned refusal per row, rendered for every backend.
fn refusals_table(real: &[Fixtured], targets: &[RefTarget]) -> String {
    let mut out = String::new();
    writeln!(out, "### Refusals\n").unwrap();
    writeln!(
        out,
        "One refusal on every backend, from a real model against a device limit it exceeds. The device is the \
         preset with one limit lowered, as the row says. The note under the table is the planner's own \
         message.\n"
    )
    .unwrap();
    out.push_str(&header("Model and limit", targets));
    let mut notes = Notes::new("refusals");
    for pin in pins() {
        let cells: Vec<String> = targets
            .iter()
            .map(|t| render_cell(&pin_cell(&pin, real, t), &mut notes))
            .collect();
        writeln!(out, "| {} | {} |", plain(pin.label), cells.join(" | ")).unwrap();
    }
    out.push_str(&notes.render());
    out
}

/// The reference dense model with every weight format on each target.
fn formats_table(targets: &[RefTarget]) -> String {
    let mut out = String::new();
    writeln!(out, "### Weight formats\n").unwrap();
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
    let mut notes = Notes::new("formats");
    for (name, format) in formats {
        let graphs = format_graphs(format);
        let cells: Vec<String> = targets
            .iter()
            .map(|t| render_cell(&cell(&graphs, &t.target), &mut notes))
            .collect();
        writeln!(out, "| {} | {} |", plain(&name), cells.join(" | ")).unwrap();
    }
    out.push_str(&notes.render());
    out
}

/// The XDNA2 part with the limits measured on the Strix Halo NPU (see the NPU page): 32 compute tiles, 16 host
/// streams each way, 2 channels per tile and 64512 tile-local bytes.
fn xdna2_target() -> Target {
    Target::Npu(NpuTarget::Aie {
        target: AieTargetId::Xdna2,
        caps: AieCaps {
            target: AieTargetId::Xdna2,
            tile_local_bytes: Queried::Known(64512),
            host_bindings: Queried::Known(16),
            instruction_bytes: Queried::Known(1 << 20),
            memory_budget_bytes: Queried::Known(1 << 30),
            region: RegionLimits {
                compute_tiles: Queried::Known(32),
                tile_input_channels: Queried::Known(2),
                tile_output_channels: Queried::Known(2),
                host_input_streams: Queried::Known(16),
                host_output_streams: Queried::Known(16),
            },
        },
    })
}

#[derive(Clone, Copy)]
enum PointwiseOp {
    Unary(UnOp),
    Binary(BinOp),
}

/// One F32 pointwise equation over one or two `[2, 4]` inputs.
fn pointwise_graph(op: PointwiseOp) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2, 4])).unwrap();
    let out = match op {
        PointwiseOp::Unary(op) => b.unary(op, x).unwrap(),
        PointwiseOp::Binary(op) => {
            let y = b.constant("y", TensorType::f32(vec![2, 4])).unwrap();
            b.binary(op, x, y).unwrap()
        }
    };
    b.finish(out).unwrap()
}

/// The F32 pointwise ops the matrix probes on the XDNA2, by the name the NPU page uses. Written by hand: a
/// probed op that the NPU page does not state the same way fails
/// `the_xdna2_table_agrees_with_the_npu_page`.
fn npu_ops() -> [(&'static str, PointwiseOp); 13] {
    [
        ("add", PointwiseOp::Binary(BinOp::Add)),
        ("sub", PointwiseOp::Binary(BinOp::Sub)),
        ("mul", PointwiseOp::Binary(BinOp::Mul)),
        ("div", PointwiseOp::Binary(BinOp::Div)),
        ("max", PointwiseOp::Binary(BinOp::Max)),
        ("ge", PointwiseOp::Binary(BinOp::Ge)),
        ("neg", PointwiseOp::Unary(UnOp::Neg)),
        ("recip", PointwiseOp::Unary(UnOp::Recip)),
        ("exp", PointwiseOp::Unary(UnOp::Exp)),
        ("log", PointwiseOp::Unary(UnOp::Log)),
        ("sqrt", PointwiseOp::Unary(UnOp::Sqrt)),
        ("tanh", PointwiseOp::Unary(UnOp::Tanh)),
        ("erf", PointwiseOp::Unary(UnOp::Erf)),
    ]
}

/// The XDNA2 ops the planner admits, one single-equation graph each. The numerical contract of each
/// admitted op is on the NPU page; this table says only whether the planner admits it.
fn npu_table() -> String {
    let ops = npu_ops();
    let target = xdna2_target();
    let mut out = String::new();
    writeln!(out, "### XDNA2 NPU\n").unwrap();
    writeln!(
        out,
        "The AMD XDNA2 NPU is not a column above because no model compiles for it. This table compiles one \
         F32 pointwise equation per row for the XDNA2 target with the limits measured on the hardware. \
         [NPU offload](../architecture/npu.mdx) states what is verified on the hardware and each op's \
         numerical contract.\n"
    )
    .unwrap();
    writeln!(out, "| Op | XDNA2 |\n| --- | --- |").unwrap();
    let mut notes = Notes::new("npu");
    for (name, op) in ops {
        let graphs = [("one F32 equation", pointwise_graph(op))];
        writeln!(
            out,
            "| {} | {} |",
            code(name),
            render_cell(&cell(&graphs, &target), &mut notes)
        )
        .unwrap();
    }
    out.push_str(&notes.render());
    out
}

/// What each device lane requires to run, per feature.
fn lanes_table(lanes: &LaneCases, targets: &[RefTarget]) -> String {
    let mut out = String::new();
    writeln!(out, "### Device lane cases\n").unwrap();
    writeln!(
        out,
        "Library and server features the planner cannot decide. They are listed from the cases each device lane \
         requires to run and pass (`scripts/required-device-cases/`).\n\n\
         - **Required** carries the number of cases that exercise the feature on that backend. It is a \
         requirement of the lane; the lane's run on the device turns it into a result.\n\
         - **No case** says the lane requires none. That is not a refusal.\n"
    )
    .unwrap();
    out.push_str(&header("Feature", targets));
    for feature in features() {
        let cells: Vec<String> = targets
            .iter()
            .map(|t| match lanes.count(t.column, &feature) {
                0 => badge("unverified", "No case"),
                1 => format!("{} 1 case", badge("lane", "Required")),
                n => format!("{} {n} cases", badge("lane", "Required")),
            })
            .collect();
        writeln!(out, "| {} | {} |", plain(feature.name), cells.join(" | ")).unwrap();
    }
    out
}

/// One table row: `heading`, then `model`'s cell on each target.
fn graph_row(heading: &str, model: &Fixtured, targets: &[RefTarget], notes: &mut Notes) -> String {
    let graphs = family_graphs(model);
    let cells: Vec<String> = targets
        .iter()
        .map(|t| render_cell(&cell(&graphs, &t.target), notes))
        .collect();
    format!("| {heading} | {} |", cells.join(" | "))
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
                    "the generated tables name a card number on line {i}: {line}"
                );
            }
        }
    }
}

// ---- the page ---------------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    poot_test_util::repo_root()
}

fn page_path() -> PathBuf {
    workspace_root().join("website/docs/develop/supported-models.md")
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

fn real_models() -> Vec<Fixtured> {
    real_dims::real_models(&workspace_root())
}

fn lane_cases() -> LaneCases {
    LaneCases::read(&workspace_root())
}

/// The whole generated region, from the checked-in inputs.
fn render_region(registry: &Registry) -> String {
    render(
        registry,
        &reference_targets(),
        &real_models(),
        &lane_cases(),
    )
}

/// The checked-in page equals what the generator renders now. Mutation: render from the committed page
/// instead of compiling (or cache the previous matrix): a planner change leaves the page unchanged and
/// this row goes red.
#[test]
fn the_page_is_what_the_generator_renders() {
    let rendered = render_region(&registry());
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
        "{} differs from the generated tables; run `just supported-models-regen`",
        path.display()
    );
}

/// POOT-1231 SC-001: the matrix has a raw Vulkan column, and it is compiled for the reference caps of the raw
/// Vulkan device (`poot_test_util::device_caps::vulkan_rdna3_igpu`, the figures `poot-vulkan-runtime` measures), not for
/// wgpu's. Both lower to SPIR-V, so only the caps tell the columns apart. Mutation: build the `Vulkan`
/// reference target from `preset("Vulkan", "vulkan", Backend::SpirvVulkan)` (wgpu's caps): the caps equal
/// wgpu's and this goes red.
#[test]
fn the_raw_vulkan_column_is_rendered_from_its_own_reference_caps() {
    let targets = reference_targets();
    let caps_of = |column: &str| match targets
        .iter()
        .find(|t| t.column == column)
        .unwrap_or_else(|| panic!("the matrix has no {column} column"))
        .target
    {
        Target::Gpu { caps, .. } => caps,
        ref other => panic!("the {column} column is a GPU target, got {other:?}"),
    };
    let (vulkan, wgpu) = (caps_of("Vulkan"), caps_of("wgpu"));
    assert_eq!(vulkan, poot_test_util::device_caps::vulkan_rdna3_igpu());
    assert_ne!(
        vulkan, wgpu,
        "the raw Vulkan column is compiled for wgpu's device caps"
    );
    let rendered = families_table(&registry(), &targets);
    let header = rendered
        .lines()
        .find(|line| line.starts_with("| Family"))
        .expect("a header row");
    assert!(
        header.contains("| Vulkan |"),
        "the rendered table has no Vulkan column: {header}"
    );
}

/// The ops the XDNA2 table shows as planned are the ops the NPU page lists as verified, and no others.
/// The page's verified list is the "Verified: pointwise F32 ops" section (before the reductions section). Mutation: add an op to `npu_ops` that plans but that the page does not
/// name, or delete an op's bullet from the page: the sets differ and this row goes red.
#[test]
fn the_xdna2_table_agrees_with_the_npu_page() {
    let path = workspace_root().join("website/docs/architecture/npu.mdx");
    let page =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let from = page
        .find("### Verified: pointwise F32 ops")
        .expect("the NPU page has a Verified: pointwise F32 ops section");
    let to = page[from..]
        .find("### Verified: reductions, norms and gather")
        .expect("the pointwise section ends where the reductions section starts")
        + from;
    let verified = &page[from..to];
    let target = xdna2_target();
    for (name, op) in npu_ops() {
        let plans = cell(&[("one F32 equation", pointwise_graph(op))], &target) == Cell::Plans;
        let stated = verified.contains(&format!("`{name}`"));
        assert_eq!(
            plans,
            stated,
            "`{name}`: the planner {} it for the XDNA2 but the NPU page {} it as verified",
            if plans { "admits" } else { "refuses" },
            if stated { "lists" } else { "does not list" }
        );
    }
}

/// Rendering twice gives the same bytes. Mutation: iterate a `HashMap` in a table, or add a timestamp:
/// the second render differs. The real-dimension models are built once and shared, so the row costs the
/// tables and not a second pass of the 27B-parameter graphs.
#[test]
fn rendering_twice_gives_identical_bytes() {
    let registry = registry();
    let targets = reference_targets();
    let real = &real_models()[..2];
    let lanes = lane_cases();
    let render = || {
        [
            families_table(&registry, &targets),
            real_table(real, &targets),
            formats_table(&targets),
            lanes_table(&lanes, &targets),
        ]
    };
    assert!(
        render() == render(),
        "two renders of the same inputs differ"
    );
}

/// A target whose workgroup memory cannot hold one attention lane.
fn tiny_lds_target() -> RefTarget {
    let mut target = targets::preset("tiny-lds", "-", poot_target::Backend::SpirvVulkan);
    gpu_caps_mut(&mut target.target).lds_bytes = TINY_LDS_BYTES;
    target
}

/// A target whose `DeviceCaps` lacks what an op needs shows the typed refusal in the row, and the same
/// family plans on the unmodified target. Mutation: cache the previous matrix instead of compiling: the
/// tiny-LDS column reads `Plans` and this goes red.
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
    let Cell::Refused {
        shape,
        subject,
        missing,
    } = &refused
    else {
        panic!("a 16-byte workgroup memory cell must be a refusal, got {refused:?}");
    };
    assert_eq!(
        (*shape, subject.as_str()),
        ("decode", "matmul"),
        "the first equation the planner cannot lower: {missing}"
    );

    let mut notes = Notes::new("test");
    let rendered = graph_row("`qwen2`", qwen2, &[tiny], &mut notes);
    let notes = notes.render();
    assert!(
        rendered.contains(">Refused</span> decode<sup><a href=\"#note-test-1\">1</a></sup>"),
        "the cell is a short status and the link to the note:\n{rendered}"
    );
    assert!(
        notes.contains(missing.as_str()),
        "the note under the table shows the typed refusal ({subject}: {missing}):\n{notes}"
    );
}

/// The one model of `real` labelled `label`.
fn real_model<'a>(real: &'a [Fixtured], label: &str) -> &'a Fixtured {
    real.iter()
        .find(|m| m.label == label)
        .unwrap_or_else(|| panic!("{label} is not a real-dimension model"))
}

/// POOT-1219 SC-001: the matrix holds a `Refused` cell on every backend, rendered from a real refusal.
/// wgpu refuses Mixtral-8x22B on its own preset limit (a stacked expert table over the single-buffer
/// limit), and each backend refuses the same model under that limit. Mutation: raise the wgpu preset's `max_buffer_bytes` in `poot-target` so the table
/// plans, or raise a pin's lowered limit: the cell reads `Plans` and the assertion names the backend and
/// the pin.
#[test]
fn every_backend_has_a_pinned_refusal_rendered_from_the_planner() {
    let real = real_models();
    let targets = reference_targets();

    let mixtral = family_graphs(real_model(&real, "Mixtral-8x22B"));
    let on_preset = cell(&mixtral, &targets[0].target);
    let Cell::Refused {
        subject, missing, ..
    } = &on_preset
    else {
        panic!("Mixtral-8x22B on the wgpu preset caps must be refused, got {on_preset:?}");
    };
    assert!(
        subject.starts_with("constant ") && subject.contains("moe.stack"),
        "the refused constant is a stacked expert table: {subject}"
    );
    assert!(
        missing.contains(&wgpu_buffer_limit().to_string()),
        "the refusal names the preset's single-buffer limit: {missing}"
    );

    for pin in pins() {
        for reference in &targets {
            let pinned = pin_cell(&pin, &real, reference);
            assert!(
                matches!(pinned, Cell::Refused { .. }),
                "{} on {}: a pinned refusal must be a typed refusal, got {pinned:?}",
                pin.label,
                reference.column
            );
        }
    }

    let rendered = refusals_table(&real, &targets);
    assert_eq!(
        rendered.matches("Refused").count(),
        pins().len() * targets.len(),
        "every pinned cell renders as a refusal:\n{rendered}"
    );
}

/// POOT-1219 SC-002 (amended: the real Qwen3.8-27B cell plans on the wgpu preset; the real-dimension
/// refusal that satisfies the row is Mixtral-8x22B's, asserted in the SC-001 row above): a real-dimension cell is what the planner says about the real graph, not about the
/// fixture-sized one. Qwen3.8-27B's real graph holds a constant (the 2.5 GB embedding table) over the
/// wgpu preset's single-buffer limit, which no fixture-sized graph does; the planner plans it by
/// splitting the table, and the cell is that verdict. Mutation: build the model from its fixture
/// (`qwen35`'s tiny config) instead of the real `config.json`: no constant is over the limit and the
/// first assertion goes red.
#[test]
fn a_real_dimension_cell_is_the_planners_verdict_at_the_real_width() {
    let real = real_models();
    let wgpu = &reference_targets()[0];
    let limit = wgpu_buffer_limit();

    let qwen3_8_27b = real_model(&real, "Qwen3.8-27B");
    let graphs = family_graphs(qwen3_8_27b);
    for (shape, graph) in &graphs {
        let largest = graph
            .consts()
            .iter()
            .map(|&id| const_bytes(graph, id))
            .max()
            .expect("the model has constants");
        assert!(
            largest > limit,
            "{shape}: the largest constant is {largest} bytes, not over the preset's {limit}: this is not the \
             real width"
        );
    }
    let options = CompileOptions {
        npu: Some(poot_npu::npu_lowering()),
        ..targets::options()
    };
    for (shape, graph) in &graphs {
        let program = compile(graph, &wgpu.target, &options)
            .unwrap_or_else(|e| panic!("{shape}: the planner refuses Qwen3.8-27B: {e}"));
        let largest = program
            .graph()
            .consts()
            .iter()
            .map(|&id| const_bytes(program.graph(), id))
            .max()
            .expect("the program has constants");
        assert!(
            largest <= limit,
            "{shape}: the plan holds a {largest}-byte constant over the {limit}-byte limit"
        );
    }
    let rendered = graph_row(
        "`Qwen3.8-27B`",
        qwen3_8_27b,
        std::slice::from_ref(wgpu),
        &mut Notes::new("test"),
    );
    assert_eq!(
        rendered,
        "| `Qwen3.8-27B` | <span class=\"matrix-status matrix-status--yes\">Plans</span> |",
        "the cell is the planner's verdict"
    );
}

/// The bytes constant `id` of `graph` holds.
fn const_bytes(graph: &Graph, id: poot_graph_ir::ValueId) -> u64 {
    let aval = graph.aval(id);
    aval.shape.iter().map(|&d| d as u64).product::<u64>() * aval.dtype.byte_size() as u64
}

/// A lane's required cases count per feature, and `No case` is not a refusal. Mutation: count every
/// case for every feature: the rows below read the same number.
#[test]
fn a_lane_count_is_the_cases_that_exercise_the_feature() {
    let text = "# comment\n\
                poot-llm\tdriver::device_tests::prefix_reuse_hits_ptx\n\
                poot-llm\tdriver::device_tests::guided_rows_ptx\n\
                poot-llm\tdriver::device_tests::guided_requests_ptx\n\
                \n";
    let lanes = LaneCases {
        lanes: vec![
            ("PTX", LaneCases::parse(text)),
            ("wgpu", Vec::new()),
            ("Vulkan", Vec::new()),
        ],
    };
    let count = |column, name| {
        let feature = features()
            .into_iter()
            .find(|f| f.name.starts_with(name))
            .expect("a listed feature");
        lanes.count(column, &feature)
    };
    assert_eq!(count("PTX", "Guided decoding"), 2);
    assert_eq!(count("PTX", "Prefix caching"), 1);
    assert_eq!(count("PTX", "LoRA"), 0);
    assert_eq!(count("wgpu", "Guided decoding"), 0);
    let rendered = lanes_table(&lanes, &reference_targets());
    assert!(
        rendered.contains("Guided decoding (regex, JSON schema, grammar) | <span class=\"matrix-status matrix-status--unverified\">No case</span> | <span class=\"matrix-status matrix-status--unverified\">No case</span> | <span class=\"matrix-status matrix-status--unverified\">No case</span> | <span class=\"matrix-status matrix-status--lane\">Required</span> 2 cases"),
        "{rendered}"
    );
}

/// The PTX lane's real list requires cases for the features the hand-written table used to mark `No` for
/// PTX (prefix caching, speculative decoding, LoRA, guided decoding, embeddings): they render `Required`.
/// Mutation: read the wgpu list for the PTX column: the counts differ from the PTX list's and the page
/// check goes red; empty the PTX list: these rows read `No case`.
#[test]
fn the_ptx_lane_requires_what_the_old_table_marked_absent() {
    let lanes = lane_cases();
    for name in [
        "Prefix caching",
        "Speculative decoding: prompt lookup",
        "Speculative decoding: the model's own MTP",
        "LoRA adapters",
        "Guided decoding",
        "Embeddings",
    ] {
        let feature = features()
            .into_iter()
            .find(|f| f.name.starts_with(name))
            .expect("a listed feature");
        assert!(
            lanes.count("PTX", &feature) > 0,
            "the PTX lane requires no case for {name}"
        );
    }
}

/// S6: every shipped family has a matrix row, with no edit to the generator: it walks the registry
/// and the fixture table. Mutation: hard-code the family list in `corpus.rs`: a family it does not
/// list has no row.
#[test]
fn every_shipped_family_has_a_row_with_no_generator_edit() {
    let registry = registry();
    let rendered = families_table(&registry, &reference_targets()[..1]);
    for entry in registry.entries() {
        assert!(
            rendered.contains(&format!("`{}`", entry.family.as_str())),
            "{} has no row:\n{rendered}",
            entry.family
        );
    }
}

/// The generator names no card number in what it writes (site-check checks the page; this checks the
/// source of the text). Mutation: add a literal card number to a cell: the assertion in `render` fires.
#[test]
fn the_rendered_region_names_no_card_number() {
    let rendered = families_table(&registry(), &reference_targets()[..1]);
    assert_no_card_numbers(&rendered);
    let result = std::panic::catch_unwind(|| assert_no_card_numbers("| a | see Card 577 |"));
    assert!(result.is_err(), "a card number is caught");
}

/// POOT-1293: every device lane requires a case that plans against the caps its attached device
/// reported, so the preset caps the matrix compiles against are not the only caps a lane's refusal is
/// checked on. Mutation: delete a lane's `attached_device_limit_*` line from
/// `scripts/required-device-cases/<lane>.txt`: that column reads `No case` and this test names it.
#[test]
fn every_device_lane_requires_planning_against_its_attached_device() {
    let lanes = lane_cases();
    let feature = features()
        .into_iter()
        .find(|f| f.name.starts_with("Planning against the attached device"))
        .expect("a listed feature");
    for column in ["wgpu", "Vulkan", "ROCm", "PTX"] {
        assert!(
            lanes.count(column, &feature) > 0,
            "the {column} lane requires no case that plans against its attached device's caps"
        );
    }
}
