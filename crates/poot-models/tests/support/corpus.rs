//! The fixture corpus the plan-summary gate and the feature matrix compile.
//!
//! The registered part is not a list: it is `Registry::builtin().entries()` crossed with each entry's
//! fixture (and its option fixtures), the towers of the model container each fixture builds
//! (`<family>/<tower>`), and the programs each model traces. A family registered with a fixture joins
//! the corpus, and so the matrix, with no edit here.
//!
//! The `parity` rows are the dense checkpoints `poot_models::test_support::dense` builds, at configurations the
//! registered fixtures do not cover (a bias, a mixed-width bf16 upload, a batched decode).
#![allow(
    dead_code,
    reason = "shared by two test binaries (plan_summary, feature_matrix); each uses part of it"
)]
use std::sync::Arc;

use poot_graph_ir::Graph;
use poot_models::io::ProgramId;
use poot_models::model::{
    LogitRows, Model, ModelOutput, ShapeReason, StepShape, StepSpec, TraceError,
};
use poot_models::registry::{FamilyEntry, Registry};
use poot_models::test_support::dense::{Dense, Family, step};
use poot_models::test_support::weight_map::Projections;

/// Layers in every dense fixture: two is the least that has a first and a later layer (dense then
/// sparse, local then global) while keeping the expected files small.
const LAYERS: usize = 2;
/// Decode KV capacity, and the prefill prompt length and cache capacity.
pub(crate) const CAP: usize = 8;
pub(crate) const PROMPT: usize = 4;
const BATCH: usize = 2;

/// One graph of the corpus: the name its summary lines carry, and a builder (graphs are rebuilt per run
/// so a planner pass cannot leak state from one target to the next).
pub(crate) struct Row {
    pub name: String,
    build: Box<dyn Fn() -> Graph>,
}

impl Row {
    fn new(name: impl Into<String>, build: impl Fn() -> Graph + 'static) -> Self {
        Self {
            name: name.into(),
            build: Box::new(build),
        }
    }

    pub(crate) fn graph(&self) -> Graph {
        (self.build)()
    }
}

/// A step shape a registered fixture is traced at: one sequence of `tokens` new tokens over `capacity`
/// positions, last-token logits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub name: &'static str,
    pub program: ProgramId,
    pub tokens: usize,
    pub capacity: usize,
}

const DECODE: Shape = Shape {
    name: "decode",
    program: ProgramId::STEP,
    tokens: 1,
    capacity: CAP,
};
const PREFILL: Shape = Shape {
    name: "prefill",
    program: ProgramId::STEP,
    tokens: PROMPT,
    capacity: PROMPT,
};
const PREFILL_KV: Shape = Shape {
    name: "prefill_kv",
    program: ProgramId::STEP,
    tokens: PROMPT,
    capacity: CAP,
};
const ENCODE: Shape = Shape {
    name: "encode",
    program: ProgramId::ENCODE,
    tokens: PROMPT,
    capacity: PROMPT,
};

/// A registered family's fixture, built into its model.
pub(crate) struct Fixtured {
    /// The family key, or `family/variant` for one of the family's option fixtures.
    pub label: String,
    /// The config JSON the model reads: the fixture's own for the served model, its `vision_config` for a
    /// tower.
    pub config: serde_json::Value,
    model: Arc<dyn Model>,
    shapes: Vec<Shape>,
}

impl Fixtured {
    /// The corpus rows of a fixture's model container: its served model, then each tower under
    /// `<label>/<tower>`.
    fn build(
        label: String,
        entry: &FamilyEntry,
        fixture: poot_models::registry::Fixture,
    ) -> Vec<Self> {
        let parts = (entry.build)(&fixture.raw(), &fixture.store)
            .unwrap_or_else(|e| panic!("{label}: the fixture does not build: {e}"));
        let towers = parts.towers.into_iter().map(|tower| {
            Self::of(
                format!("{label}/{}", tower.id),
                fixture.config["vision_config"].clone(),
                tower.model,
            )
        });
        std::iter::once(Self::of(label.clone(), fixture.config.clone(), parts.model))
            .chain(towers)
            .collect()
    }

    fn of(label: String, config: serde_json::Value, model: Box<dyn Model>) -> Self {
        let model: Arc<dyn Model> = Arc::from(model);
        let schema = model.schema();
        let program = schema
            .step()
            .or(schema.encode())
            .unwrap_or_else(|| panic!("{label}: the schema has no step or encode program"));
        let shapes = match program.output() {
            // A family that returns logits is a decoder: decode, prefill, prefill into a larger cache.
            ModelOutput::Logits { .. } => vec![DECODE, PREFILL, PREFILL_KV],
            // An encoder is one forward over the whole input. A family that fixes the token count
            // (a vision tower: one patch per token) states it in its shape refusal.
            ModelOutput::Hidden { .. } | ModelOutput::Scores { .. } => {
                let tokens = match model.trace(
                    &StepSpec::encode(encode_step(ENCODE.tokens)),
                    poot_graph_ir::TraceLimits::for_tests(),
                ) {
                    Err(TraceError::ShapeUnsupported {
                        reason: ShapeReason::CapacityAboveMax { max },
                        ..
                    }) => max,
                    _ => ENCODE.tokens,
                };
                vec![Shape {
                    tokens,
                    capacity: tokens,
                    ..ENCODE
                }]
            }
        };
        Self::at(label, config, model, shapes)
    }

    /// A fixture of an already built `model`, traced at `shapes`.
    pub(crate) fn at(
        label: String,
        config: serde_json::Value,
        model: Arc<dyn Model>,
        shapes: Vec<Shape>,
    ) -> Self {
        Self {
            label,
            config,
            model,
            shapes,
        }
    }

    /// The built model.
    pub(crate) fn model(&self) -> &dyn Model {
        &*self.model
    }

    /// The shapes this fixture is compiled at.
    pub(crate) fn shapes(&self) -> &[Shape] {
        &self.shapes
    }

    /// The graph of `shape`. A fixture that cannot trace a shape its own list holds is a broken fixture.
    pub(crate) fn trace(&self, shape: Shape) -> Graph {
        let traced = self
            .model
            .trace(
                &StepSpec::new(
                    shape.program,
                    step(1, shape.tokens, shape.capacity, LogitRows::Last),
                ),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap_or_else(|e| panic!("{} {}: {e}", self.label, shape.name));
        // The driver path: the model's packed weights (a dense model has none) are bound before compile.
        poot_graph_plan::bind_packed_weights(
            &traced,
            &poot_graph_plan::WeightFormats::from_weight_map(self.model.weights()),
        )
        .unwrap_or_else(|e| panic!("{} {}: {e}", self.label, shape.name))
    }
}

fn encode_step(tokens: usize) -> StepShape {
    step(1, tokens, tokens, LogitRows::Last)
}

/// Every fixture of every family `registry` holds, in registration order.
pub(crate) fn fixtured(registry: &Registry) -> Vec<Fixtured> {
    let mut out = Vec::new();
    for entry in registry.entries() {
        out.extend(Fixtured::build(
            entry.family.as_str().to_string(),
            entry,
            poot_models::test_support::family_fixtures::base(entry.family),
        ));
        for (variant, make) in poot_models::test_support::family_fixtures::of(entry.family).variants
        {
            out.extend(Fixtured::build(
                format!("{}/{variant}", entry.family.as_str()),
                entry,
                make(),
            ));
        }
    }
    out
}

/// One row per registered fixture and shape, named `<label>.<shape>`.
fn registered_rows(registry: &Registry) -> Vec<Row> {
    let mut rows = Vec::new();
    for fixtured in fixtured(registry) {
        let fixtured = Arc::new(fixtured);
        for shape in fixtured.shapes().to_vec() {
            let name = format!("{}.{}", fixtured.label, shape.name);
            let fixtured = Arc::clone(&fixtured);
            rows.push(Row::new(name, move || fixtured.trace(shape)));
        }
    }
    rows
}

// ---- the dense parity rows ---------------------------------------------------------------------

/// The tiny dense checkpoint of `family`: the fixture dimensions through the family's own config (the
/// head dimension is explicit, as the family fixtures give it).
fn dense(family: Family) -> Dense {
    Dense::new(family)
        .vocab(24)
        .dims(16, 16, LAYERS)
        .heads(4, 2)
        .head_dim(6)
        .max_positions(32)
        .with("rms_norm_eps", 1e-5)
}

/// Gemma 3 with a query scalar of one (attention scale 1), a window of three on the local layers,
/// every second layer global.
fn gemma() -> Dense {
    dense(Family::Gemma3)
        .with("query_pre_attn_scalar", 1.0)
        .with("sliding_window", 3)
        .with("sliding_window_pattern", 2)
}

/// A parity shape: its name, the program, and `rows sequences of `tokens` new tokens over `cap` positions.
type ParityShape = (&'static str, ProgramId, usize, usize, usize);

/// A parity checkpoint: its name, its builder and the shapes it is traced at.
type ParityCheckpoint<'a> = (&'a str, fn() -> Dense, &'a [ParityShape]);

const PARITY_DECODE: ParityShape = ("decode", ProgramId::STEP, 1, 1, CAP);
const PARITY_PREFILL: ParityShape = ("prefill", ProgramId::STEP, 1, PROMPT, PROMPT);
const PARITY_PREFILL_KV: ParityShape = ("prefill_kv", ProgramId::STEP, 1, PROMPT, CAP);
const PARITY_DECODE_BATCHED: ParityShape = ("decode_batched", ProgramId::STEP, BATCH, 1, CAP);

/// The dense parity rows, named `parity.<checkpoint>.<shape>`: each checkpoint at its shapes, last-token
/// logits, then the qwen2 checkpoint with projections and head stored BF16 and the embedding F32 (the
/// mixed-width upload path).
fn parity_rows() -> Vec<Row> {
    let checkpoints: [ParityCheckpoint; 4] = [
        (
            "qwen2",
            || dense(Family::Qwen2),
            &[
                PARITY_DECODE,
                PARITY_PREFILL,
                PARITY_PREFILL_KV,
                PARITY_DECODE_BATCHED,
            ],
        ),
        (
            "qwen3",
            || dense(Family::Qwen3),
            &[PARITY_DECODE, PARITY_PREFILL],
        ),
        (
            "gemma",
            gemma,
            &[PARITY_DECODE, PARITY_PREFILL, PARITY_PREFILL_KV],
        ),
        (
            "olmo2",
            || dense(Family::Olmo2),
            &[PARITY_DECODE, PARITY_PREFILL, PARITY_PREFILL_KV],
        ),
    ];
    let mut rows = Vec::new();
    for (checkpoint, make, shapes) in checkpoints {
        for &(shape, program, batch, tokens, cap) in shapes {
            rows.push(Row::new(
                format!("parity.{checkpoint}.{shape}"),
                move || {
                    make()
                        .f32_model()
                        .model
                        .trace(
                            &StepSpec::new(program, step(batch, tokens, cap, LogitRows::Last)),
                            poot_graph_ir::TraceLimits::for_tests(),
                        )
                        .unwrap()
                },
            ));
        }
    }
    for (shape, program, tokens, cap) in [
        ("decode", ProgramId::STEP, 1, CAP),
        ("prefill", ProgramId::STEP, PROMPT, PROMPT),
    ] {
        rows.push(Row::new(format!("parity.qwen2_bf16.{shape}"), move || {
            dense(Family::Qwen2)
                .build(Projections::Bf16, poot_tensor::DType::F32, None)
                .model
                .trace(
                    &StepSpec::new(program, step(1, tokens, cap, LogitRows::Last)),
                    poot_graph_ir::TraceLimits::for_tests(),
                )
                .unwrap()
        }));
    }
    rows
}

/// The whole corpus: the registered fixtures first, then the dense parity rows.
pub(crate) fn corpus(registry: &Registry) -> Vec<Row> {
    let mut rows = registered_rows(registry);
    rows.extend(parity_rows());
    rows
}
