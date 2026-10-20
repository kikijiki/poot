//! A second out-of-tree family, "tagged", that declares a request input no driver code has ever heard
//! of (POOT-1224 SC-001): one I32 tag per prompt position that selects a row of a tag table added to
//! the token's embedding before the head. The driver binds it through the schema's request-kind
//! binder like any other input: this crate has no access to driver internals, and no file under
//! `crates/poot-llm/src/driver` names the input.
//!
//! The family is deliberately small: embedding, tag table and head, no attention, so it carries no
//! state and every row of its logits is a function of the token and the tag bound at that position.
//! The tag's presence is `Required`, so a request without it is refused at admission, before it holds
//! a slot or a block; beyond the prompt the tag continues at a fixed value (`Fill::Advance` with step
//! 0), so the generated tokens are tagged too.

use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_graph_ir::{Builder, Graph, IndexRange, Slot, SlotKey, TensorType};
use poot_models::components::dims::{Field, Fields};
use poot_models::components::embed::Embedding;
use poot_models::components::linear::{Weight, WeightError, linear};
use poot_models::components::standard::Step;
use poot_models::io::{
    Fill, InputId, Layout, ModelSchema, PadValue, Per, Presence, ProgramId, ProgramSpec,
    RequestForm, RequestInput,
};
use poot_models::model::{
    FamilyKey, Model, ModelConfig, ModelError, ModelOutput, StepSpec, TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, ModelParts, RawConfig};
use poot_quant::weights::{
    DenseWeight, WeightEntry, WeightId, WeightKey, WeightMap, WeightRole, WeightStore, WeightView,
};
use poot_tensor::DType;
use poot_text::{ChatFormat, HermesToolCodec};

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("poot_fixture_tagged");

/// The HF `model_type` that names the family.
const MODEL_TYPE: &str = "poot_fixture_tagged";

/// The registry entry: what `Registry::builtin_plus` takes.
pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, MODEL_TYPE)],
    build,
};

/// The new request input: one tag per prompt position.
pub(crate) const TAG: InputId = InputId::new("tag");
/// The `Slot::Activation` tag the family declares [`TAG`] under, `[rows, tokens]` I32.
const TAG_SLOT: &str = "tag";

/// Tags the table has rows for.
pub(crate) const TAGS: usize = 4;
/// The tag the generated positions carry: the value `TAG`'s values continue with past the prompt.
pub(crate) const GENERATED_TAG: i32 = 3;
const WIDTH: usize = 16;
pub(crate) const VOCAB: usize = 48;
const EOS: u32 = 47;
const MAX_POSITIONS: usize = 64;

#[derive(Debug)]
struct Tagged {
    weights: WeightMap,
    embed: Embedding,
    tags: Weight,
    head: Weight,
    config: ModelConfig,
    schema: ModelSchema,
}

/// The schema entry of [`TAG`]: dense, one I32 per position, padded with tag 0.
fn tag_input() -> RequestInput {
    RequestInput::new(
        TAG,
        SlotKey::new(Slot::Activation, Some(TAG_SLOT)),
        RequestForm::new(
            DType::I32,
            Per::Token {
                width: 1,
                layout: Layout::TokenMajor,
                presence: None,
            },
            PadValue::I32(0),
        ),
    )
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let fields = Fields::new(raw, FAMILY);
    let vocab = fields.vocab()?;
    let width = fields.count(Field::hf("hidden_size"), None)?;
    let tags = fields.count(Field::hf("num_tags"), None)?;
    let weight_error = |source| ModelError::Weight {
        family: FAMILY,
        source,
    };
    let mut map = WeightMap::builder(store);
    for (role, name) in [
        (WeightRole::Embed, "tagged.embed.weight"),
        (WeightRole::TypeEmbed, "tagged.tags.weight"),
        (WeightRole::Head, "tagged.head.weight"),
    ] {
        map.map(
            WeightId::model(role),
            WeightView::Stored(WeightKey::new(name)),
        )
        .map_err(weight_error)?;
    }
    let weights = map.build();
    let weight = |role, shape: &[usize]| {
        Weight::new(&weights, WeightId::model(role), shape)
            .map_err(|e: WeightError| e.for_family(FAMILY))
    };
    let schema = ModelSchema::builder()
        .input(tag_input())
        .step(
            ProgramSpec::new(ProgramId::STEP, ModelOutput::Logits { vocab })
                .reading(TAG, Presence::Required),
        )
        .build_for(FAMILY)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab,
        max_positions: fields.count(Field::hf("max_position_embeddings"), None)?,
        eos: fields.eos(None)?,
        bos: fields.bos()?,
        prompt_bos: None,
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
        tool_codec: Some(Arc::new(HermesToolCodec)),
    };
    Ok(ModelParts::new(Box::new(Tagged {
        embed: Embedding::new(&weights, vocab, width).map_err(|e| e.for_family(FAMILY))?,
        tags: weight(WeightRole::TypeEmbed, &[tags, width])?,
        head: weight(WeightRole::Head, &[vocab, width])?,
        weights,
        config,
        schema,
    })))
}

impl Model for Tagged {
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
        let shape = spec.shape;
        let step = Step::new(&b, shape)?;
        let (rows, tokens) = (shape.rows.get(), shape.tokens.get());
        // The tag table is indexed by the bound tags: declaring their range makes an out-of-range tag a
        // refused bind, not an unchecked gather.
        let tags = b.slot_named_ranged(
            Slot::Activation,
            TAG_SLOT,
            TensorType::new(vec![rows, tokens], DType::I32),
            IndexRange::below(TAGS).expect("the table has rows"),
        )?;
        let tag_rows = b.gather(self.tags.declare(&b)?, 0, tags)?;
        let tag_rows = if b.aval(tag_rows)?.dtype == DType::F32 {
            tag_rows
        } else {
            b.cast(tag_rows, DType::F32)?
        };
        let mut x = b.binary(
            poot_graph_ir::BinOp::Add,
            self.embed.embed(&b, step.tokens())?,
            tag_rows,
        )?;
        if shape.logits == poot_models::model::LogitRows::Last && tokens > 1 {
            x = b.slice(x, 1, tokens - 1, tokens)?;
        }
        let logits = linear(&b, x, &self.head, None)?;
        Ok(step.finish(b, logits)?)
    }
}

/// The fixture's config: a 16-wide table model over the 48-word vocabulary and four tags.
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": MODEL_TYPE,
        "vocab_size": VOCAB,
        "hidden_size": WIDTH,
        "num_tags": TAGS,
        "max_position_embeddings": MAX_POSITIONS,
        "eos_token_id": EOS,
    })
}

/// The synthetic checkpoint `ENTRY` ships for tests: BF16 tensors of deterministic pseudo-random
/// values (one xorshift stream over every tensor in order).
pub(crate) fn fixture() -> Fixture {
    let mut store = WeightStore::builder();
    let mut seed = 0x51ed_270bu32;
    for (name, shape, spread) in [
        ("tagged.embed.weight", vec![VOCAB, WIDTH], 2.0f32),
        ("tagged.tags.weight", vec![TAGS, WIDTH], 2.0),
        ("tagged.head.weight", vec![VOCAB, WIDTH], 2.0),
    ] {
        let n: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(2 * n);
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let u = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            bytes.extend_from_slice(&(((spread * u).to_bits() >> 16) as u16).to_le_bytes());
        }
        let dense = DenseWeight::try_new(DType::BF16, shape, bytes.into())
            .expect("the fixture bytes match their shape");
        store
            .insert(name, WeightEntry::Dense(dense))
            .expect("the fixture names are unique");
    }
    Fixture {
        config: fixture_config(),
        generation: None,
        preprocessor: None,
        store: store.build(),
    }
}

/// The value of [`TAG`] a request over a `prompt_len`-token prompt supplies: `tags` at the prompt's
/// positions, and [`GENERATED_TAG`] at every position past it.
pub(crate) fn tag_values(tags: &[i32]) -> poot_models::io::TokenValues {
    poot_models::io::TokenValues::dense(poot_tensor::HostTensor::i32(
        vec![tags.len(), 1],
        tags.to_vec(),
    ))
    .expect("a column of tags is [positions, 1]")
    .with_beyond(Fill::Advance {
        from: vec![GENERATED_TAG],
        step: 0,
    })
}
