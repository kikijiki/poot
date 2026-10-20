//! POOT-1224 SC-001: the out-of-tree "tagged" family ([`tagged`]) serves a row with a per-request input
//! kind (an extra I32 tensor) that no driver code knows, and no file under `crates/poot-llm/src/driver`
//! was edited for it: this crate reaches the product through its public surface only.
//!
//! - Host half (model-free): the family registers, a driver over the fake device prepares and steps it
//!   (the executor refuses a step whose bound slots differ from the entry's schema, so a missing or
//!   mis-shaped tag fails here), and admission refuses a request without the tag or with a tag column
//!   of the wrong length before the request holds a slot.
//! - Device half: `Driver::generate_with_inputs` on a real device equals the CPU oracle's tokens for
//!   the same tags, and the tags decide the tokens (two tag assignments decode differently).

use std::cell::Cell;
use std::num::NonZeroUsize;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use poot_eval::test_support::weight_map_oracle::eval_mapped;
use poot_executor::Engine;
use poot_graph_ir::{Slot, SlotKey};
use poot_llm::driver::error::{DriverError, InvalidRequest};
use poot_llm::driver::{
    AdapterRef, BackendChoice, Driver, DriverOptions, GenerateRequest, Head, Layout, LoadOptions,
    ModelHandle, SeqRequest, Serving, ServingShapes, Warm, open_executor,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::io::{InputError, InputValue, RequestInputs};
use poot_models::model::{Model, StepShape, StepSpec};
use poot_models::registry::{Fixture, Registry};
use poot_quant::weights::WeightStore;
use poot_runtime_common::DeviceBackend;
use poot_tensor::HostTensor;

use crate::tagged::{self, GENERATED_TAG, TAG, TAGS, VOCAB};
use crate::{CAPACITY, Counting, FakeDevice, driver_options, greedy, nz, write_checkpoint};

/// The prompt, as token ids.
const PROMPT: [u32; 6] = [3, 17, 8, 29, 5, 21];
/// The tag of each prompt position.
const PROMPT_TAGS: [i32; 6] = [0, 1, 2, 1, 0, 2];
/// A second assignment of the same prompt's tags.
const OTHER_TAGS: [i32; 6] = [3, 3, 0, 2, 1, 1];
const NEW_TOKENS: usize = 8;

/// The greedy tokens of [`PROMPT`] under [`PROMPT_TAGS`] on the CPU oracle, pinned: the fixture's
/// weights are fixed bytes, so a different token here is a different model.
const PINNED: [u32; NEW_TOKENS] = [24, 37, 3, 45, 10, 20, 45, 10];
/// The greedy tokens of [`PROMPT`] under [`OTHER_TAGS`], pinned: different from [`PINNED`] because the
/// tag is an input the logits depend on.
const PINNED_OTHER: [u32; NEW_TOKENS] = [20, 45, 10, 20, 45, 10, 20, 45];

fn registry() -> Registry {
    Registry::builtin_plus(&[tagged::ENTRY]).unwrap()
}

fn load(dir: &Path) -> ModelHandle {
    ModelHandle::load(dir, &registry(), &LoadOptions::default()).unwrap()
}

fn tagged_model(fixture: &Fixture) -> Box<dyn Model> {
    registry()
        .build(&fixture.raw(), &fixture.store)
        .unwrap()
        .model
}

/// The request inputs of [`PROMPT`] with `tags` at its positions.
fn tag_inputs(tags: &[i32]) -> RequestInputs {
    RequestInputs::none().with(TAG, InputValue::Tokens(tagged::tag_values(tags)))
}

/// One oracle step: `tokens` at positions `start..`, each tagged `tag_of(position)`. The family
/// carries no state, so every step is stateless. Returns the last token's logits.
fn oracle_step(
    model: &dyn Model,
    store: &WeightStore,
    tokens: &[u32],
    start: usize,
    tag_of: &dyn Fn(usize) -> i32,
) -> Vec<f32> {
    let n = tokens.len();
    let shape = StepShape::new(NonZeroUsize::MIN, nz(n), nz(CAPACITY));
    let graph = model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let slots = [
        (
            SlotKey::new(Slot::Token, None),
            HostTensor::i32(vec![1, n], tokens.iter().map(|&t| t as i32).collect()),
        ),
        (
            SlotKey::new(Slot::Pos, None),
            HostTensor::i32(vec![1, n], (start..start + n).map(|p| p as i32).collect()),
        ),
        (
            SlotKey::new(Slot::Activation, Some("tag")),
            HostTensor::i32(vec![1, n], (start..start + n).map(tag_of).collect()),
        ),
    ];
    let (logits, _) = eval_mapped(&graph, store, model.weights(), &slots, &[]);
    logits.as_f32().expect("f32 logits").to_vec()
}

/// `NEW_TOKENS` greedy tokens after [`PROMPT`] with `tags` over the prompt and [`GENERATED_TAG`] past it.
fn oracle_tokens(model: &dyn Model, store: &WeightStore, tags: &[i32]) -> Vec<u32> {
    let tag_of = |position: usize| tags.get(position).copied().unwrap_or(GENERATED_TAG);
    let mut out = vec![greedy(&oracle_step(model, store, &PROMPT, 0, &tag_of)).0];
    for i in 1..NEW_TOKENS {
        let position = PROMPT.len() + i - 1;
        let last = out[i - 1];
        out.push(greedy(&oracle_step(model, store, &[last], position, &tag_of)).0);
    }
    out
}

fn request() -> GenerateRequest {
    GenerateRequest {
        prompt: PROMPT.to_vec(),
        max_new: NEW_TOKENS,
        sampler: Sampler::greedy(),
        stops: Vec::new(),
        ignore_eos: true,
    }
}

/// A driver over the tagged family on the fake device.
fn fake_driver(
    options: DriverOptions,
) -> (
    Result<Driver, DriverError>,
    poot_test_util::UniqueTempPath,
    FakeDevice,
) {
    let device = FakeDevice::default();
    let dir = poot_test_util::unique_temp_path("tagged_driver_checkpoint");
    write_checkpoint(&dir, &tagged::fixture());
    let executor = Counting {
        inner: Engine::new(device.clone()),
        loads: Rc::new(Cell::new(0)),
    };
    let driver = Driver::new(Arc::new(load(&dir)), Box::new(executor), options);
    (driver, dir, device)
}

/// SC-001 (host half): the family declares an input kind the driver has never heard of; the driver
/// prepares the entry, binds the tag at every step of a generation (the executor refuses a step whose
/// bound slots differ from the entry's schema) and generates. Mutation: make `bind::resolve` refuse a
/// request kind it has no arm for (or bind it as padding only); the prepare is refused as an unbound
/// slot, or the step's bound tag is the wrong shape, and this row is red.
#[test]
fn a_family_with_a_new_input_kind_is_prepared_and_served_with_no_driver_edit() {
    let (driver, _dir, device) = fake_driver(driver_options(4, 64));
    let mut driver = driver.unwrap().serve(Layout::Contiguous);
    let mut sink = |_: u32, _: &str| GenerationControl::Continue(());
    let generation = driver
        .generate_with_inputs(request(), tag_inputs(&PROMPT_TAGS), &mut sink)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(generation.tokens.len(), NEW_TOKENS);
    assert!(driver.stats().steps as usize >= NEW_TOKENS);
    // The values the driver bound: the prompt's tags in the two prefill pieces it planned (4 + 2 tokens)
    // and the continuation tag at the decode steps. Mutation: shift every bound tag by one in
    // `bind::cell`; the pieces below are other bytes and this row is red.
    let uploaded = |tags: &[i32]| {
        let want: Vec<u8> = tags.iter().flat_map(|t| t.to_le_bytes()).collect();
        device.writes.borrow().contains(&want)
    };
    assert!(
        uploaded(&PROMPT_TAGS[..4]),
        "the first piece's tags were not bound"
    );
    assert!(
        uploaded(&PROMPT_TAGS[4..]),
        "the second piece's tags were not bound"
    );
    assert!(
        uploaded(&[GENERATED_TAG]),
        "the decode step's tag was not bound"
    );
}

/// SC-001 (admission): a request that omits the tag is refused with the typed `Missing` before it
/// holds a slot, and the driver then serves a request that carries it. Mutation: skip the required
/// check in `ModelSchema::check_request`; the first request is admitted and this row is red.
#[test]
fn a_request_without_the_required_tag_is_refused_before_it_holds_a_slot() {
    let (driver, _dir, _) = fake_driver(driver_options(4, 64));
    let driver = driver.unwrap();
    let mut driver = driver
        .prepare(&ServingShapes {
            layout: Layout::Contiguous,
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY],
            warm: Warm::Prompts(&[PROMPT.len()]),
            windows: &[],
            adapters: &[],
            chain: None,
        })
        .unwrap();
    let missing = driver
        .open(SeqRequest::new(&PROMPT, NEW_TOKENS, AdapterRef::Base))
        .unwrap_err();
    assert!(
        matches!(
            missing,
            DriverError::InvalidRequest(InvalidRequest::Input(InputError::Missing { id })) if id == TAG
        ),
        "{missing:?}"
    );
    // The contiguous layout has one slot: had the refusal held it, this open would find it busy.
    let opened = driver
        .open(
            SeqRequest::new(&PROMPT, NEW_TOKENS, AdapterRef::Base)
                .with_inputs(tag_inputs(&PROMPT_TAGS)),
        )
        .unwrap();
    assert!(matches!(opened, poot_llm::driver::Admission::Opened { .. }));
}

/// SC-001 (admission): a tag column that does not cover the prompt is refused as a `Support` defect,
/// and a column of the wrong type as a `Form` defect, before the request holds anything. Mutation:
/// skip the support check in `check_value`; the short column is admitted and this row is red.
#[test]
fn a_tag_column_that_does_not_fit_the_prompt_is_refused_with_a_typed_defect() {
    let (driver, _dir, _) = fake_driver(driver_options(4, 64));
    let driver = driver.unwrap();
    let mut driver = driver
        .prepare(&ServingShapes {
            layout: Layout::Contiguous,
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY],
            warm: Warm::Prompts(&[PROMPT.len()]),
            windows: &[],
            adapters: &[],
            chain: None,
        })
        .unwrap();
    let open = |driver: &mut Driver<Serving>, inputs| {
        driver
            .open(SeqRequest::new(&PROMPT, NEW_TOKENS, AdapterRef::Base).with_inputs(inputs))
            .unwrap_err()
    };
    let short = open(&mut driver, tag_inputs(&PROMPT_TAGS[..4]));
    assert!(
        matches!(
            short,
            DriverError::InvalidRequest(InvalidRequest::Input(InputError::Support { id, .. })) if id == TAG
        ),
        "{short:?}"
    );
    let floats = poot_models::io::TokenValues::dense(HostTensor::f32(
        vec![PROMPT.len(), 1],
        vec![0.0; PROMPT.len()],
    ))
    .unwrap();
    let wrong = open(
        &mut driver,
        RequestInputs::none().with(TAG, InputValue::Tokens(floats)),
    );
    assert!(
        matches!(
            wrong,
            DriverError::InvalidRequest(InvalidRequest::Input(InputError::Form { id, .. })) if id == TAG
        ),
        "{wrong:?}"
    );
}

/// The oracle's tokens depend on the tags, and are the pinned lists: without this the device row
/// below could pass with a tag that never reaches the logits.
#[test]
fn the_oracle_decodes_the_pinned_tokens_and_the_tags_decide_them() {
    let fixture = tagged::fixture();
    let model = tagged_model(&fixture);
    let tokens = oracle_tokens(&*model, &fixture.store, &PROMPT_TAGS);
    let other = oracle_tokens(&*model, &fixture.store, &OTHER_TAGS);
    assert_eq!(
        tokens, PINNED,
        "the oracle's greedy tokens under the first tags"
    );
    assert_eq!(
        other, PINNED_OTHER,
        "the oracle's greedy tokens under the second tags"
    );
    assert_ne!(tokens, other, "the tags must decide the tokens");
    assert!(
        tokens.iter().chain(&other).all(|&t| (t as usize) < VOCAB),
        "tokens are vocabulary ids"
    );
    assert_eq!(TAGS, 4);
}

/// `Driver::generate_with_inputs` on `choice`'s device over the tagged checkpoint: the tokens equal
/// the oracle's under both tag assignments. The device skips cleanly when absent (and fails when the
/// lane requires it).
fn assert_generate_matches_the_oracle(choice: BackendChoice, backend: DeviceBackend) {
    let Some(executor) =
        poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice))
    else {
        return;
    };
    let fixture = tagged::fixture();
    let dir = poot_test_util::unique_temp_path("tagged_device_checkpoint");
    write_checkpoint(&dir, &fixture);
    let model = tagged_model(&fixture);
    let mut driver = Driver::new(Arc::new(load(&dir)), executor, driver_options(8, 64))
        .unwrap()
        .serve(Layout::Contiguous);
    for (tags, pinned) in [(&PROMPT_TAGS, PINNED), (&OTHER_TAGS, PINNED_OTHER)] {
        assert_eq!(oracle_tokens(&*model, &fixture.store, tags), pinned);
        let generation = driver
            .generate_with_inputs(request(), tag_inputs(tags), &mut |_: u32, _: &str| {
                GenerationControl::Continue(())
            })
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            generation.tokens, pinned,
            "the device's greedy tokens differ from the oracle's under tags {tags:?}"
        );
    }
}

/// SC-001 (device half) on wgpu. Mutation: bind the tag column as padding only in the request binder
/// (`bind::token_values`); the device's tokens diverge from the oracle's.
#[test]
fn tagged_generate_equals_the_oracle_tokens_wgpu() {
    assert_generate_matches_the_oracle(BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

/// SC-001 (device half) on ROCm (the `rocm` feature; without it the device does not open and the row
/// skips).
#[test]
fn tagged_generate_equals_the_oracle_tokens_rocm() {
    assert_generate_matches_the_oracle(BackendChoice::Rocm, DeviceBackend::Rocm);
}

/// SC-001 (device half) on PTX, in the pod batch.
#[test]
fn tagged_generate_equals_the_oracle_tokens_ptx() {
    assert_generate_matches_the_oracle(BackendChoice::Ptx, DeviceBackend::Ptx);
}
