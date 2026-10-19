//! POOT-733 SC-005 and SC-007 through families registered from outside the production crates.
//!
//! A scripted tracer (SC-005) declares exactly the values, equations and literals its config names, so
//! each of `max_tokens`, `max_values`, `max_eqns` and `max_literal_bytes` is isolated in its own
//! fixture: the exact boundary traces, the first request past it is refused by the typed budget before
//! the excess is built, and a step above `max_tokens` never reaches the tracer body. A second family
//! (SC-007) calls a shared component (`gated_ffn`, `Embedding`) under a restrictive equation limit.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::num::{NonZeroU64, NonZeroUsize};

use poot_graph_ir::{
    AllocationSite, Attempted, BinOp, BudgetResource, Builder, Graph, TensorType,
    TraceBudgetExceeded, TraceLimits, UnOp,
};
use poot_models::components::embed::Embedding;
use poot_models::components::ffn::{GatedFfnWeights, gated_ffn};
use poot_models::components::standard::Step;
use poot_models::io::{ModelSchema, ProgramId, ProgramSpec};
use poot_models::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, StepShape, StepSpec,
    TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, ModelParts, RawConfig, Registry};
use poot_quant::weights::{
    DenseWeight, FfnRole, WeightEntry, WeightId, WeightMap, WeightRole, WeightStore, WeightView,
};
use poot_tensor::DType;
use poot_text::ChatFormat;

thread_local! {
    /// How many times a scripted tracer body ran on this thread.
    static BODY_CALLS: Cell<usize> = const { Cell::new(0) };
}

const SCRIPTED: FamilyKey = FamilyKey::new("scripted");
const SHARED: FamilyKey = FamilyKey::new("shared-components");

fn count(raw: &RawConfig<'_>, family: FamilyKey, field: &'static str) -> Result<usize, ModelError> {
    let RawConfig::HfJson { config, .. } = raw else {
        return Err(ModelError::Config {
            family,
            field,
            reason: ConfigReason::Unsupported,
        });
    };
    config
        .get(field)
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .ok_or(ModelError::Config {
            family,
            field,
            reason: ConfigReason::Missing,
        })
}

/// A schema of one `step` program that carries no state: neither family keeps a cache.
fn schema_of(family: FamilyKey, vocab: usize) -> ModelSchema {
    ModelSchema::builder()
        .step(ProgramSpec::new(
            ProgramId::STEP,
            ModelOutput::Logits { vocab },
        ))
        .build_for(family)
        .expect("a stateless step program is a schema")
}

fn config_of(family: FamilyKey, vocab: usize) -> ModelConfig {
    ModelConfig {
        family,
        vocab,
        max_positions: 16,
        eos: BTreeSet::from([0]),
        bos: None,
        prompt_bos: None,
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
    }
}

// ---- the scripted family --------------------------------------------------------------------

/// Declares `constants` constants, then `unaries` negations of the first, then `literals` scalar
/// additions: so a trace holds `constants + unaries + literals` values and `unaries + literals`
/// equations and `4 * literals` literal bytes.
#[derive(Debug)]
struct Scripted {
    config: ModelConfig,
    schema: ModelSchema,
    weights: WeightMap,
    constants: usize,
    unaries: usize,
    literals: usize,
    /// The length of the name of the first constant: the controlled metadata allocation.
    name_len: usize,
}

impl Model for Scripted {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, _spec: &StepSpec, b: Builder) -> Result<Graph, TraceError> {
        BODY_CALLS.with(|calls| calls.set(calls.get() + 1));
        let ty = TensorType::f32(vec![4]);
        let first = b.constant(&"w".repeat(self.name_len), ty.clone())?;
        for index in 1..self.constants {
            b.constant(&format!("c{index}"), ty.clone())?;
        }
        let mut x = first;
        for _ in 0..self.unaries {
            x = b.unary(UnOp::Neg, x)?;
        }
        for _ in 0..self.literals {
            x = b.binary_scalar(BinOp::Add, x, poot_graph_ir::Scalar::F32(1.0))?;
        }
        Ok(b.finish(x)?)
    }
}

fn build_scripted(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let weights = WeightMap::builder(store).build();
    Ok(ModelParts::new(Box::new(Scripted {
        config: config_of(SCRIPTED, 4),
        schema: schema_of(SCRIPTED, 4),
        weights,
        constants: count(raw, SCRIPTED, "constants")?,
        unaries: count(raw, SCRIPTED, "unaries")?,
        literals: count(raw, SCRIPTED, "literals")?,
        name_len: count(raw, SCRIPTED, "name_len")?,
    })))
}

fn scripted_fixture() -> Fixture {
    Fixture {
        config: serde_json::json!({
            "model_type": "scripted", "constants": 1, "unaries": 0, "literals": 0, "name_len": 1
        }),
        generation: None,
        preprocessor: None,
        store: WeightStore::default(),
    }
}

const SCRIPTED_ENTRY: FamilyEntry = FamilyEntry {
    family: SCRIPTED,
    keys: &[(ConfigSource::HfModelType, "scripted")],
    build: build_scripted,
    fixture: scripted_fixture,
    variants: &[],
};

fn scripted(constants: usize, unaries: usize, literals: usize, name_len: usize) -> Box<dyn Model> {
    let mut registry = Registry::empty();
    registry.register(SCRIPTED_ENTRY).unwrap();
    let fixture = Fixture {
        config: serde_json::json!({
            "model_type": "scripted",
            "constants": constants,
            "unaries": unaries,
            "literals": literals,
            "name_len": name_len,
        }),
        generation: None,
        preprocessor: None,
        store: WeightStore::default(),
    };
    registry
        .build(&fixture.raw(), &fixture.store)
        .unwrap()
        .model
}

fn shape(tokens: usize) -> StepShape {
    StepShape::new(
        NonZeroUsize::MIN,
        NonZeroUsize::new(tokens).unwrap(),
        NonZeroUsize::new(8).unwrap(),
    )
}

/// The SC-005 limits with one field set; every other resource is out of the way.
fn only(tokens: usize, values: usize, eqns: usize, literal: u64, owned: u64) -> TraceLimits {
    TraceLimits {
        max_tokens: NonZeroUsize::new(tokens).unwrap(),
        max_values: NonZeroUsize::new(values).unwrap(),
        max_eqns: NonZeroUsize::new(eqns).unwrap(),
        max_literal_bytes: NonZeroU64::new(literal).unwrap(),
        max_owned_bytes: NonZeroU64::new(owned).unwrap(),
    }
}

fn budget(error: TraceError) -> TraceBudgetExceeded {
    match error {
        TraceError::Budget(budget) => budget,
        other => panic!("expected a typed budget refusal, got {other:?}"),
    }
}

const ROOMY: u64 = 1 << 30;

#[test]
fn the_fifth_token_is_refused_before_the_tracer_body_runs() {
    let model = scripted(1, 0, 0, 1);
    let limits = only(4, 1 << 10, 1 << 10, ROOMY, ROOMY);
    BODY_CALLS.with(|calls| calls.set(0));
    model.trace(&StepSpec::step(shape(4)), limits).unwrap();
    assert_eq!(BODY_CALLS.with(Cell::get), 1, "the exact boundary traces");

    BODY_CALLS.with(|calls| calls.set(0));
    let refused = budget(model.trace(&StepSpec::step(shape(5)), limits).unwrap_err());
    assert_eq!(
        (refused.resource, refused.limit, refused.requested),
        (BudgetResource::Tokens, 4, 5)
    );
    assert_eq!(refused.attempted, Attempted::Total(5));
    assert_eq!(refused.site, AllocationSite::TokenRequest);
    assert_eq!(
        BODY_CALLS.with(Cell::get),
        0,
        "the tracer body ran for a step the limit refuses"
    );
}

#[test]
fn the_ninth_value_is_refused_where_the_eighth_is_admitted() {
    let limits = only(4, 8, 1 << 10, ROOMY, ROOMY);
    scripted(8, 0, 0, 1)
        .trace(&StepSpec::step(shape(1)), limits)
        .unwrap();
    let refused = budget(
        scripted(9, 0, 0, 1)
            .trace(&StepSpec::step(shape(1)), limits)
            .unwrap_err(),
    );
    assert_eq!(
        (
            refused.resource,
            refused.limit,
            refused.used,
            refused.requested
        ),
        (BudgetResource::Values, 8, 8, 1)
    );
    assert_eq!(refused.attempted, Attempted::Total(9));
}

#[test]
fn the_seventh_equation_is_refused_where_the_sixth_is_admitted() {
    let limits = only(4, 1 << 10, 6, ROOMY, ROOMY);
    scripted(1, 6, 0, 1)
        .trace(&StepSpec::step(shape(1)), limits)
        .unwrap();
    let refused = budget(
        scripted(1, 7, 0, 1)
            .trace(&StepSpec::step(shape(1)), limits)
            .unwrap_err(),
    );
    assert_eq!(
        (
            refused.resource,
            refused.limit,
            refused.used,
            refused.requested
        ),
        (BudgetResource::Equations, 6, 6, 1)
    );
    assert_eq!(refused.attempted, Attempted::Total(7));
}

#[test]
fn the_thirty_sixth_literal_byte_is_refused_where_the_thirty_second_is_admitted() {
    let limits = only(4, 1 << 10, 1 << 10, 32, ROOMY);
    scripted(1, 0, 8, 1)
        .trace(&StepSpec::step(shape(1)), limits)
        .unwrap();
    let refused = budget(
        scripted(1, 0, 9, 1)
            .trace(&StepSpec::step(shape(1)), limits)
            .unwrap_err(),
    );
    assert_eq!(
        (
            refused.resource,
            refused.limit,
            refused.used,
            refused.requested
        ),
        (BudgetResource::LiteralBytes, 32, 32, 4)
    );
    assert_eq!(refused.attempted, Attempted::Total(36));
}

#[test]
fn an_owned_ceiling_below_a_controlled_name_refuses_before_the_name_is_copied() {
    // The first constant's name is the controlled metadata allocation. Find the smallest ceiling that
    // admits a one-byte name, then show a 4 KiB name needs 4095 more bytes and is refused at the input
    // record's site, before the name exists.
    let admits = |ceiling: u64, name_len: usize| {
        scripted(1, 0, 0, name_len)
            .trace(&StepSpec::step(shape(1)), only(4, 64, 64, 64, ceiling))
            .map(drop)
    };
    let smallest = (1..1 << 16)
        .find(|&ceiling| admits(ceiling, 1).is_ok())
        .expect("some ceiling admits a one-byte name");
    admits(smallest, 1).unwrap();
    let refused = budget(admits(smallest, 4096).unwrap_err());
    assert_eq!(refused.resource, BudgetResource::OwnedBytes);
    assert_eq!(refused.site, AllocationSite::InputRecord);
    assert_eq!(refused.limit, smallest);
    // A 4096-byte name asks for exactly 4095 more bytes than the one-byte name that fits.
    let Attempted::Total(attempted) = refused.attempted else {
        panic!("a request this size does not overflow");
    };
    assert_eq!(attempted - refused.used, refused.requested);
    assert!(refused.requested > 4095, "{refused:?}");
}

// ---- a family that calls shared components ---------------------------------------------------

const WIDTH: usize = 4;
const INTER: usize = 6;
const VOCAB: usize = 8;

#[derive(Debug)]
struct SharedLm {
    config: ModelConfig,
    schema: ModelSchema,
    weights: WeightMap,
    embed: Embedding,
    ffn: GatedFfnWeights,
}

impl Model for SharedLm {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, spec: &StepSpec, b: Builder) -> Result<Graph, TraceError> {
        let step = Step::new(&b, spec.shape)?;
        let x = self.embed.embed(&b, step.tokens())?;
        let y = gated_ffn(&b, x, &self.ffn)?;
        Ok(step.finish(b, y)?)
    }
}

fn dense(shape: &[usize]) -> WeightEntry {
    let elements: usize = shape.iter().product();
    let bytes = poot_quant::StoredBytes::from(vec![0u8; elements * 4]);
    WeightEntry::Dense(DenseWeight::try_new(DType::F32, shape.to_vec(), bytes).unwrap())
}

fn build_shared(_raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let wrap = |source| ModelError::Weight {
        family: SHARED,
        source,
    };
    let mut map = WeightMap::builder(store);
    map.map(
        WeightId::model(WeightRole::Embed),
        WeightView::Stored("embed".into()),
    )
    .map_err(wrap)?;
    for (role, name) in [
        (FfnRole::Gate, "gate"),
        (FfnRole::Up, "up"),
        (FfnRole::Down, "down"),
    ] {
        map.map(
            WeightId::layer(0, WeightRole::Ffn(role)),
            WeightView::Stored(name.into()),
        )
        .map_err(wrap)?;
    }
    let weights = map.build();
    let embed = Embedding::new(&weights, VOCAB, WIDTH).map_err(|e| e.for_family(SHARED))?;
    let ffn = GatedFfnWeights::new(&weights, 0, WIDTH, INTER).map_err(|e| e.for_family(SHARED))?;
    Ok(ModelParts::new(Box::new(SharedLm {
        config: config_of(SHARED, VOCAB),
        schema: schema_of(SHARED, VOCAB),
        weights,
        embed,
        ffn,
    })))
}

fn shared_fixture() -> Fixture {
    let mut store = WeightStore::builder();
    store.insert("embed", dense(&[VOCAB, WIDTH])).unwrap();
    store.insert("gate", dense(&[INTER, WIDTH])).unwrap();
    store.insert("up", dense(&[INTER, WIDTH])).unwrap();
    store.insert("down", dense(&[WIDTH, INTER])).unwrap();
    Fixture {
        config: serde_json::json!({ "model_type": "shared-components" }),
        generation: None,
        preprocessor: None,
        store: store.build(),
    }
}

const SHARED_ENTRY: FamilyEntry = FamilyEntry {
    family: SHARED,
    keys: &[(ConfigSource::HfModelType, "shared-components")],
    build: build_shared,
    fixture: shared_fixture,
    variants: &[],
};

fn shared_model() -> Box<dyn Model> {
    let mut registry = Registry::empty();
    registry.register(SHARED_ENTRY).unwrap();
    let fixture = shared_fixture();
    registry
        .build(&fixture.raw(), &fixture.store)
        .unwrap()
        .model
}

#[test]
fn a_shared_component_charges_the_callers_budget_and_the_third_equation_is_refused() {
    let model = shared_model();
    let roomy = only(8, 1 << 10, 1 << 10, ROOMY, ROOMY);
    let adequate = model.trace(&StepSpec::step(shape(1)), roomy).unwrap();
    // The embedding gather and the gated FFN are the equations the limit counts.
    assert!(
        adequate.eqns().len() > 3,
        "the fixture is deeper than the limit"
    );

    let tight = only(8, 1 << 10, 2, ROOMY, ROOMY);
    let refused = budget(model.trace(&StepSpec::step(shape(1)), tight).unwrap_err());
    assert_eq!(
        (
            refused.resource,
            refused.limit,
            refused.used,
            refused.requested
        ),
        (BudgetResource::Equations, 2, 2, 1)
    );
    assert_eq!(refused.attempted, Attempted::Total(3));

    // The same limits that admit the trace leave its graph exactly as an unrestricted trace builds it.
    let finite = only(
        8,
        adequate.values().len(),
        adequate.eqns().len(),
        ROOMY,
        ROOMY,
    );
    let again = model.trace(&StepSpec::step(shape(1)), finite).unwrap();
    assert_eq!(
        format!("{again:?}"),
        format!("{adequate:?}"),
        "an exactly sufficient budget changes nothing"
    );
    let one_short = only(
        8,
        adequate.values().len(),
        adequate.eqns().len() - 1,
        ROOMY,
        ROOMY,
    );
    let refused = budget(
        model
            .trace(&StepSpec::step(shape(1)), one_short)
            .unwrap_err(),
    );
    assert_eq!(refused.resource, BudgetResource::Equations);
    assert_eq!(
        refused.attempted,
        Attempted::Total(adequate.eqns().len() as u64)
    );
}
