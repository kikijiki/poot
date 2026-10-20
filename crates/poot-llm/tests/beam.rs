//! Beam search as a driver capability (Card 752, dserve.md ruling V4/S7), on the add-a-model fixture
//! family and on a recurrent fake family registered by this test: families defined outside every
//! production crate beam-search through `Driver::beam` exactly as a built-in one does, because no list
//! of families decides it.
//!
//! Every row compares the device with an independent host reference ([`reference`]): its own beam search
//! over the CPU oracle's logits of each beam's whole token list, one fresh prefill per beam per step, in
//! f64. The cases are chosen so that the reference's own rankings survive logit noise ([`NOISE`]), so a
//! device rounding error cannot reorder them, and so that each mutation named on a row changes the
//! reference's answer (`Case::assert_discriminates`).

#[path = "add_a_model/family.rs"]
mod family;

use std::collections::BTreeSet;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use poot_eval::test_support::weight_map_oracle::eval_mapped;
use poot_executor::{Device, Dispatch, Engine};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, SlotKey, StateRole, TensorType, UnOp};
use poot_graph_plan::{CompileLimits, CompileOptions, Target};
use poot_llm::driver::{
    BackendChoice, Beam, BeamError, BeamRequest, DraftPolicy, Driver, DriverOptions, FinishReason,
    GenerateRequest, Head, Layout, LoadOptions, ModelHandle, PoolShape, PreparedSetLimits,
    ServingShapes, State, Warm, open_executor, program_retention,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::components::standard::Step;
use poot_models::io::{ModelSchema, ProgramId, ProgramSpec, StateSpec};
use poot_models::model::{
    FamilyKey, LogitRows, Model, ModelConfig, ModelError, ModelOutput, StepShape, StepSpec,
    TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, ModelParts, RawConfig, Registry};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::{BufferRole, DeviceBackend};
use poot_target::{Backend, DeviceCaps};
use poot_tensor::{DType, HostTensor};
use poot_test_util::per_backend;
use poot_text::{ChatFormat, HermesToolCodec};
use poot_weights::WeightMap;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

use family::{ENTRY, EOS, VOCAB};

/// The KV positions of every driver in this file (the families' `max_position_embeddings`).
const CAPACITY: usize = 64;

/// How far a device log-probability may differ from the oracle's: measured differences are about 2e-6, so
/// a wrong state or ranking is far outside it and a rounding difference far inside.
const LOGPROB_TOLERANCE: f64 = 1e-4;

/// How far a case's logits may move without changing what the reference returns: far more than a
/// device's rounding differs from the oracle, so no ranking in a chosen case is a near-tie.
const NOISE: f32 = 5e-3;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Checkpoints.
// ---------------------------------------------------------------------------------------------

/// A word-level tokenizer over the families' 48-word vocabulary.
fn tokenizer() -> Tokenizer {
    let mut words: Vec<String> = ('a'..='z').map(String::from).collect();
    words.extend(('0'..='9').map(String::from));
    words.extend((0..VOCAB - words.len()).map(|i| format!("p{i}")));
    let vocab = words
        .into_iter()
        .enumerate()
        .map(|(i, w)| (w, i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("a".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
}

fn write_checkpoint(dir: &Path, fixture: &Fixture) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), fixture.config.to_string()).unwrap();
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    poot_quant::test_support::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
}

/// The fixture with its output head scaled by `factor`: the same family, token for token ranked as the
/// fixture ranks them, but with logit gaps wide enough that a beam's candidates are far from ties.
fn sharpened(fixture: &Fixture, factor: f32) -> Fixture {
    let mut store = WeightStore::builder();
    for (key, entry) in fixture.store.iter() {
        let WeightEntry::Dense(dense) = entry else {
            panic!("the fixture is dense")
        };
        let mut bytes = dense.bytes().as_slice().to_vec();
        if key.as_str() == "embed_out.weight" {
            for pair in bytes.as_chunks_mut::<2>().0 {
                let value = f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16) * factor;
                *pair = ((value.to_bits() >> 16) as u16).to_le_bytes();
            }
        }
        let weight = DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), bytes.into())
            .expect("the bytes keep their shape");
        store
            .insert(key.as_str(), WeightEntry::Dense(weight))
            .expect("the names are unique");
    }
    Fixture {
        config: fixture.config.clone(),
        generation: None,
        preprocessor: None,
        store: store.build(),
    }
}

/// The built-in families plus the fixture family and the recurrent fake, registered from outside the
/// production crates.
fn registry() -> Registry {
    Registry::builtin_plus(&[ENTRY, RECURRENT_ENTRY]).unwrap()
}

/// `fixture` written as a checkpoint and loaded.
fn load(fixture: &Fixture, name: &str) -> (Arc<ModelHandle>, poot_test_util::UniqueTempPath) {
    let dir = poot_test_util::unique_temp_path(name);
    write_checkpoint(&dir, fixture);
    let handle = ModelHandle::load(&dir, &registry(), &LoadOptions::default()).unwrap();
    (Arc::new(handle), dir)
}

// ---------------------------------------------------------------------------------------------
// A recurrent fake family: one `StateRole::Recurrent` state, decayed and folded with every token, that
// the logits read. A beam whose state is a different beam's gets different logits.
// ---------------------------------------------------------------------------------------------

const RECURRENT: FamilyKey = FamilyKey::new("poot_fixture_beam_recurrent");

const RECURRENT_ENTRY: FamilyEntry = FamilyEntry {
    family: RECURRENT,
    keys: &[(ConfigSource::HfModelType, "poot_fixture_beam_recurrent")],
    build: build_recurrent,
};

#[derive(Debug)]
struct Recurrentish {
    config: ModelConfig,
    schema: ModelSchema,
    weights: WeightMap,
}

fn build_recurrent(_: &RawConfig<'_>, _: &WeightStore) -> Result<ModelParts, ModelError> {
    Ok(ModelParts::new(Box::new(Recurrentish {
        config: ModelConfig {
            family: RECURRENT,
            vocab: VOCAB,
            max_positions: CAPACITY,
            eos: BTreeSet::from([EOS]),
            bos: None,
            prompt_bos: None,
            prefill_granule: NonZeroUsize::MIN,
            chat: ChatFormat::ChatML,
            tool_codec: Some(Arc::new(HermesToolCodec)),
        },
        schema: ModelSchema::builder()
            .step(
                ProgramSpec::new(ProgramId::STEP, ModelOutput::Logits { vocab: VOCAB })
                    .carrying(StateSpec::recurrent("toy.state")),
            )
            .build_for(RECURRENT)?,
        weights: WeightMap::default(),
    })))
}

fn recurrent_fixture() -> Fixture {
    let mut store = WeightStore::builder();
    let dummy = DenseWeight::try_new(DType::BF16, vec![1], vec![0u8, 0].into()).unwrap();
    store.insert("dummy", WeightEntry::Dense(dummy)).unwrap();
    Fixture {
        config: serde_json::json!({
            "model_type": "poot_fixture_beam_recurrent",
            "vocab_size": VOCAB,
            "max_position_embeddings": CAPACITY,
            "eos_token_id": EOS,
        }),
        generation: None,
        preprocessor: None,
        store: store.build(),
    }
}

impl Model for Recurrentish {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    /// Per token `t`: `state += (t + 1) / 48 * [1, 2, 3, 4] / 4`; per position the logits are
    /// `-0.5 * (v - 47 * m)^2` with `m = (1 + tanh(1.2 * state . [-1, 1, -1, 1] - 1.8)) / 2`, a peak
    /// at the vocabulary position the state selects: it climbs toward the end-of-sequence token as
    /// tokens accumulate.
    fn trace_into(&self, spec: &StepSpec, b: poot_graph_ir::Builder) -> Result<Graph, TraceError> {
        let shape = spec.shape;
        let step = Step::new(&b, shape)?;
        let (n, rows) = (shape.tokens.get(), shape.rows.get());
        let tokens = b.cast(
            b.guard_index_bounds(step.tokens(), VOCAB)
                .expect("the step's ids are a rank-2 I32 tensor")
                .guarded,
            DType::F32,
        )?;
        let state_dims = vec![rows, 4];
        let state = b.state_input(
            "toy.state",
            TensorType::f32(state_dims.clone()),
            StateRole::Recurrent,
        )?;
        let coefficient = b.binary_scalar(
            BinOp::Mul,
            b.binary_scalar(BinOp::Add, b.iota(4)?, Builder::f32(1.0))?,
            Builder::f32(0.25),
        )?;
        let coefficient = b.broadcast(coefficient, state_dims.clone())?;
        let readout = {
            let one = b.binary_scalar(BinOp::Add, b.iota(1)?, Builder::f32(1.0))?;
            let minus = b.binary_scalar(BinOp::Sub, b.iota(1)?, Builder::f32(1.0))?;
            b.broadcast(b.concat(0, &[minus, one, minus, one])?, state_dims.clone())?
        };
        let vocab_axis = b.broadcast(b.iota(VOCAB)?, vec![rows, VOCAB])?;
        let mut state_now = state;
        let mut logits = Vec::new();
        for j in 0..n {
            let token = b.broadcast(b.slice(tokens, 1, j, j + 1)?, state_dims.clone())?;
            let drive = b.binary(
                BinOp::Mul,
                b.binary_scalar(
                    BinOp::Mul,
                    b.binary_scalar(BinOp::Add, token, Builder::f32(1.0))?,
                    Builder::f32(1.0 / 48.0),
                )?,
                coefficient,
            )?;
            state_now = b.binary(BinOp::Add, state_now, drive)?;
            if shape.logits == LogitRows::Last && j + 1 < n {
                continue;
            }
            let read = b.reduce(
                poot_graph_ir::op::RedOp::Sum,
                b.binary(BinOp::Mul, state_now, readout)?,
                1,
                true,
            )?;
            let m = b.binary_scalar(
                BinOp::Mul,
                b.binary_scalar(
                    BinOp::Add,
                    b.unary(
                        UnOp::Tanh,
                        b.binary_scalar(
                            BinOp::Add,
                            b.binary_scalar(BinOp::Mul, read, Builder::f32(1.2))?,
                            Builder::f32(-1.8),
                        )?,
                    )?,
                    Builder::f32(1.0),
                )?,
                Builder::f32(0.5),
            )?;
            let peak = b.broadcast(
                b.binary_scalar(BinOp::Mul, m, Builder::f32(47.0))?,
                vec![rows, VOCAB],
            )?;
            let distance = b.binary(BinOp::Sub, vocab_axis, peak)?;
            let row = b.binary_scalar(
                BinOp::Mul,
                b.binary(BinOp::Mul, distance, distance)?,
                Builder::f32(-0.5),
            )?;
            logits.push(b.reshape(row, vec![rows, 1, VOCAB])?);
        }
        step.carry(state, state_now);
        let out = if logits.len() == 1 {
            logits[0]
        } else {
            b.concat(1, &logits)?
        };
        Ok(step.finish(b, out)?)
    }
}

// ---------------------------------------------------------------------------------------------
// The CPU oracle: a family's own traced graph, evaluated state-free over a whole token list.
// ---------------------------------------------------------------------------------------------

struct Oracle {
    model: Box<dyn Model>,
    store: WeightStore,
    /// Added to every logit as a deterministic pseudo-random value in `[-noise, noise]` that depends on
    /// the token list and the vocabulary position: how a case shows it is not a near-tie.
    noise: f32,
}

impl Oracle {
    fn new(fixture: &Fixture, noise: f32) -> Self {
        let model = registry().build(&fixture.raw(), &fixture.store).unwrap();
        Self {
            model: model.model,
            store: fixture.store.clone(),
            noise,
        }
    }

    /// The logits after the last of `tokens`, from one fresh prefill of the whole list at position 0.
    fn last_logits(&self, tokens: &[u32]) -> Vec<f32> {
        let n = tokens.len();
        let shape = StepShape::new(NonZeroUsize::MIN, nz(n), nz(CAPACITY));
        let graph = self
            .model
            .trace(
                &StepSpec::step(shape),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let state: Vec<HostTensor> = graph
            .state()
            .map(|(input, _)| HostTensor::zeros(graph.aval(input).shape.clone()))
            .collect();
        let slots = [
            (
                SlotKey::new(Slot::Token, None),
                HostTensor::i32(vec![1, n], tokens.iter().map(|&t| t as i32).collect()),
            ),
            (
                SlotKey::new(Slot::Pos, None),
                HostTensor::i32(vec![1, n], (0..n as i32).collect()),
            ),
        ];
        let (logits, _) = eval_mapped(&graph, &self.store, self.model.weights(), &slots, &state);
        let mut logits = logits.as_f32().expect("f32 logits").to_vec();
        if self.noise != 0.0 {
            let mut seed = tokens.iter().fold(0x9e37_79b9u32, |h, &t| {
                (h ^ t).wrapping_mul(0x85eb_ca6b).rotate_left(13)
            });
            for logit in &mut logits {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *logit += self.noise * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0);
            }
        }
        logits
    }
}

// ---------------------------------------------------------------------------------------------
// The host reference: an independent beam search over the oracle, in f64.
// ---------------------------------------------------------------------------------------------

/// How the reference's length penalty counts tokens.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Length {
    /// The whole list, prompt included (vLLM's rule).
    WithPrompt,
    /// The generated tokens only.
    WithoutPrompt,
}

/// The reference's knobs: the rule, and the two deviations each mutation of a row makes.
#[derive(Clone, Copy, Debug)]
struct Rules {
    width: usize,
    penalty: f64,
    length: Length,
    /// Continuations taken from each live beam.
    breadth: usize,
    /// Whether a beam that ends on EOS keeps being expanded.
    expand_eos: bool,
}

impl Rules {
    fn new(width: usize, penalty: f64) -> Self {
        Self {
            width,
            penalty,
            length: Length::WithPrompt,
            breadth: 2 * width,
            expand_eos: false,
        }
    }
}

#[derive(Clone, Debug)]
struct Hyp {
    tokens: Vec<u32>,
    logprobs: Vec<f64>,
}

impl Hyp {
    fn cum(&self) -> f64 {
        self.logprobs.iter().sum()
    }

    fn score(&self, prompt: usize, rules: &Rules) -> f64 {
        let mut len = match rules.length {
            Length::WithPrompt => prompt + self.tokens.len(),
            Length::WithoutPrompt => self.tokens.len(),
        };
        if self.tokens.last() == Some(&EOS) {
            len -= 1;
        }
        self.cum() / (len.max(1) as f64).powf(rules.penalty)
    }
}

struct Reference {
    /// The result, best first: the beam, whether it ended on EOS.
    beams: Vec<(Hyp, bool)>,
    /// Per step, the parent row of each surviving beam.
    parents: Vec<Vec<usize>>,
}

/// `log_softmax` of `logits` in f64.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&x| (x as f64 - max).exp()).sum();
    let lse = max + sum.ln();
    logits.iter().map(|&x| x as f64 - lse).collect()
}

fn reference(oracle: &Oracle, prompt: &[u32], max_new: usize, rules: Rules) -> Reference {
    let order = |a: &Hyp, b: &Hyp| {
        b.score(prompt.len(), &rules)
            .total_cmp(&a.score(prompt.len(), &rules))
            .then_with(|| a.tokens.last().cmp(&b.tokens.last()))
            .then_with(|| a.tokens.cmp(&b.tokens))
    };
    let mut live = vec![Hyp {
        tokens: Vec::new(),
        logprobs: Vec::new(),
    }];
    let mut completed: Vec<Hyp> = Vec::new();
    let mut parents = Vec::new();
    for _ in 0..max_new {
        let mut children: Vec<(usize, Hyp)> = Vec::new();
        for (row, beam) in live.iter().enumerate() {
            let mut context = prompt.to_vec();
            context.extend_from_slice(&beam.tokens);
            let lp = log_softmax(&oracle.last_logits(&context));
            let mut ranked: Vec<(u32, f64)> = lp
                .iter()
                .enumerate()
                .map(|(token, &lp)| (token as u32, lp))
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for &(token, lp) in &ranked[..rules.breadth] {
                let mut child = beam.clone();
                child.tokens.push(token);
                child.logprobs.push(lp);
                if token == EOS && !rules.expand_eos {
                    completed.push(child);
                } else {
                    children.push((row, child));
                }
            }
        }
        children.sort_by(|a, b| order(&a.1, &b.1));
        children.truncate(rules.width);
        if children.is_empty() {
            live.clear();
            break;
        }
        parents.push(children.iter().map(|&(parent, _)| parent).collect());
        live = children.into_iter().map(|(_, child)| child).collect();
    }
    let mut all: Vec<(Hyp, bool)> = completed
        .into_iter()
        .map(|beam| (beam, true))
        .chain(live.into_iter().map(|beam| (beam, false)))
        .collect();
    all.sort_by(|a, b| order(&a.0, &b.0));
    all.truncate(rules.width);
    Reference {
        beams: all,
        parents,
    }
}

/// Whether two references return different beams (tokens, in order).
fn differs(a: &Reference, b: &Reference) -> bool {
    let tokens = |r: &Reference| -> Vec<Vec<u32>> {
        r.beams
            .iter()
            .map(|(beam, _)| beam.tokens.clone())
            .collect()
    };
    tokens(a) != tokens(b)
}

// ---------------------------------------------------------------------------------------------
// Driver and comparison helpers.
// ---------------------------------------------------------------------------------------------

fn options() -> DriverOptions {
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(CAPACITY),
        prefill_chunk: nz(8),
        trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(64)),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
        hidden_window: None,
    }
}

fn request(prompt: &[u32], max_new: usize, width: usize, length_penalty: f32) -> BeamRequest {
    BeamRequest {
        prompt: prompt.to_vec(),
        max_new,
        width: nz(width),
        length_penalty,
        ignore_eos: false,
    }
}

/// Assert `beams` are the reference's: the same tokens in the same order, the same log-probabilities,
/// sums and scores to `tolerance`, and the same end. A NaN fails every comparison.
fn assert_matches(beams: &[Beam], prompt: &[u32], reference: &Reference, rules: Rules) {
    let tokens: Vec<_> = beams.iter().map(|b| b.tokens.clone()).collect();
    let wanted: Vec<_> = reference
        .beams
        .iter()
        .map(|(b, _)| b.tokens.clone())
        .collect();
    assert_eq!(tokens, wanted, "the beams differ from the reference");
    for (beam, (wanted, ended)) in beams.iter().zip(&reference.beams) {
        let tolerance = LOGPROB_TOLERANCE * (1.0 + beam.tokens.len() as f64);
        for (k, (&got, &want)) in beam.logprobs.iter().zip(&wanted.logprobs).enumerate() {
            assert!(
                (f64::from(got) - want).abs() <= LOGPROB_TOLERANCE,
                "{:?} token {k}: logprob {got} against {want}",
                beam.tokens
            );
        }
        assert!(
            (f64::from(beam.cum_logprob) - wanted.cum()).abs() <= tolerance,
            "{:?}: cum {} against {}",
            beam.tokens,
            beam.cum_logprob,
            wanted.cum()
        );
        let score = wanted.score(prompt.len(), &rules);
        assert!(
            (f64::from(beam.score) - score).abs() <= tolerance,
            "{:?}: score {} against {score}",
            beam.tokens,
            beam.score
        );
        let finish = if *ended {
            FinishReason::Eos
        } else {
            FinishReason::MaxTokens
        };
        assert_eq!(beam.finish, finish, "{:?}", beam.tokens);
    }
}

fn device(choice: BackendChoice) -> Option<Box<dyn poot_executor::Executor>> {
    let backend = match choice {
        BackendChoice::Wgpu => DeviceBackend::Wgpu,
        BackendChoice::Rocm => DeviceBackend::Rocm,
        BackendChoice::Ptx => DeviceBackend::Ptx,
        other => panic!("the beam rows have no device row on {other:?}"),
    };
    poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice))
}

fn nothing(_: u32, _: &str) -> GenerationControl {
    GenerationControl::Continue(())
}

// ---------------------------------------------------------------------------------------------
// The chosen cases.
// ---------------------------------------------------------------------------------------------

/// How much the dense family's output head is sharpened: logit gaps wide enough for a ranking to survive
/// [`NOISE`].
const SHARPEN: f32 = 30.0;

/// A case: a family, a prompt, a width and a length penalty, and the host reference of what it returns.
struct Case {
    fixture: Fixture,
    prompt: &'static [u32],
    max_new: usize,
    width: usize,
    penalty: f64,
}

impl Case {
    fn rules(&self) -> Rules {
        Rules::new(self.width, self.penalty)
    }

    fn oracle(&self) -> Oracle {
        Oracle::new(&self.fixture, 0.0)
    }

    fn reference(&self) -> Reference {
        reference(&self.oracle(), self.prompt, self.max_new, self.rules())
    }

    fn request(&self) -> BeamRequest {
        request(self.prompt, self.max_new, self.width, self.penalty as f32)
    }

    /// The case's ranking is not a near-tie: the reference returns the same beams, ending the same way,
    /// with every logit moved by up to [`NOISE`] either way.
    fn assert_stable(&self) {
        let honest = self.reference();
        for noise in [NOISE, -NOISE] {
            let moved = reference(
                &Oracle::new(&self.fixture, noise),
                self.prompt,
                self.max_new,
                self.rules(),
            );
            assert!(
                !differs(&honest, &moved) && ends_alike(&honest, &moved),
                "the case is a near-tie: logits moved by {noise} change the beams"
            );
        }
    }

    /// Each deviation `deviations` names returns different beams on this case.
    fn assert_discriminates(&self, deviations: &[Rules]) {
        let honest = self.reference();
        for deviation in deviations {
            let deviated = reference(&self.oracle(), self.prompt, self.max_new, *deviation);
            assert!(
                differs(&honest, &deviated),
                "{deviation:?} returns the same beams"
            );
        }
    }

    /// Some beam changes parent after the prompt's first step: the reorder moves state between rows.
    fn assert_reorders(&self) {
        let reference = self.reference();
        assert!(
            reference.parents.iter().skip(1).any(|parents| parents
                .iter()
                .enumerate()
                .any(|(row, &parent)| row != parent)),
            "no beam changes parent: {:?}",
            reference.parents
        );
    }
}

fn ends_alike(a: &Reference, b: &Reference) -> bool {
    a.beams.iter().zip(&b.beams).all(|(x, y)| x.1 == y.1)
}

/// The dense case: a beam ends on EOS at the second step, a longer one stays live, the ranking changes
/// between steps and a parent change moves rows. Breadth `width` loses a beam; expanding past EOS adds
/// continuations of the ended one.
fn dense_case() -> Case {
    Case {
        fixture: sharpened(&family::fixture(), SHARPEN),
        prompt: &[14, 11, 17, 35],
        max_new: 8,
        width: 2,
        penalty: 1.0,
    }
}

/// The recurrent case: three beams, several parent changes (the best beam's parent is the second one at
/// the second step) and every beam ends on EOS.
fn recurrent_case() -> Case {
    Case {
        fixture: recurrent_fixture(),
        prompt: &[4, 27, 42, 38, 28],
        max_new: 8,
        width: 3,
        penalty: 1.0,
    }
}

/// The length case: an ended beam of 6 generated tokens against a live one of 8. Counted with the prompt
/// (vLLM's rule) the ended one is ranked second; counted without it, first.
fn length_case() -> Case {
    Case {
        fixture: sharpened(&family::fixture(), SHARPEN),
        prompt: &[23, 8, 0, 5, 15, 2],
        max_new: 8,
        width: 2,
        penalty: 1.0,
    }
}

/// Prompts whose beam search at width 1 and penalty 0 is the greedy walk: one ends on EOS after three
/// tokens, one runs to `max_new`.
const GREEDY_PROMPTS: [&[u32]; 2] = [&[14, 44, 25, 42, 28, 44], &[1, 34, 11]];

// ---------------------------------------------------------------------------------------------
// Host rows: the cases are not near-ties and discriminate the mutations of the device rows.
// ---------------------------------------------------------------------------------------------

/// The dense case is stable under logit noise, breadth `width` and expanding past EOS both change what
/// it returns, and its beams change parents (so the first reorder is not the only one).
#[test]
fn the_dense_case_discriminates_breadth_and_eos_and_reorders() {
    let case = dense_case();
    case.assert_stable();
    let rules = case.rules();
    case.assert_discriminates(&[
        Rules {
            breadth: rules.width,
            ..rules
        },
        Rules {
            expand_eos: true,
            ..rules
        },
    ]);
    case.assert_reorders();
}

/// The recurrent case is stable under logit noise and its beams change parents.
#[test]
fn the_recurrent_case_is_stable_and_reorders() {
    let case = recurrent_case();
    case.assert_stable();
    case.assert_reorders();
}

/// The length case is stable under logit noise, and counting the generated tokens only (the
/// mutation of the length row) returns the beams in the other order.
#[test]
fn the_length_case_ranks_differently_without_the_prompt() {
    let case = length_case();
    case.assert_stable();
    case.assert_discriminates(&[Rules {
        length: Length::WithoutPrompt,
        ..case.rules()
    }]);
}

/// The greedy prompts' reference at width 1 and penalty 0 is the oracle's greedy walk.
#[test]
fn width_one_penalty_zero_is_the_greedy_walk_on_the_oracle() {
    let fixture = sharpened(&family::fixture(), SHARPEN);
    let oracle = Oracle::new(&fixture, 0.0);
    for prompt in GREEDY_PROMPTS {
        let beams = reference(&oracle, prompt, 12, Rules::new(1, 0.0)).beams;
        assert_eq!(beams.len(), 1);
        assert_eq!(
            beams[0].0.tokens,
            greedy_walk(&oracle, prompt, 12),
            "{prompt:?}"
        );
    }
}

/// The greedy tokens after `prompt` on the oracle, up to and including an end-of-sequence token.
fn greedy_walk(oracle: &Oracle, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let mut tokens = prompt.to_vec();
    for _ in 0..max_new {
        let logits = oracle.last_logits(&tokens);
        let best = logits
            .iter()
            .enumerate()
            .fold(0, |best, (i, &v)| if v > logits[best] { i } else { best });
        tokens.push(best as u32);
        if best as u32 == EOS {
            break;
        }
    }
    tokens.split_off(prompt.len())
}

// ---------------------------------------------------------------------------------------------
// Device rows.
// ---------------------------------------------------------------------------------------------

fn handle_of(case: &Case, name: &str) -> (Arc<ModelHandle>, poot_test_util::UniqueTempPath) {
    load(&case.fixture, name)
}

fn driver_of(handle: &Arc<ModelHandle>, executor: Box<dyn poot_executor::Executor>) -> Driver {
    Driver::new(Arc::clone(handle), executor, options()).unwrap()
}

fn beam_or_panic<S: State>(driver: &mut Driver<S>, request: BeamRequest) -> Vec<Beam> {
    driver.beam(request).unwrap_or_else(|e| panic!("{e}"))
}

/// SC-001: beam search on the fixture family returns the beams, in order, with the scores, of the host
/// reference that expands every beam from the oracle's logits of its own token list. The case has an
/// ended beam, a live one, a ranking that changes between steps and a parent change.
/// Mutations: expand each beam by `width` continuations instead of `2 * width` (`beam`: `breadth`); keep
/// expanding a beam after EOS (`beam`: push the ended child to `children`).
fn assert_beam_matches_the_reference(choice: BackendChoice) {
    let Some(executor) = device(choice) else {
        return;
    };
    let case = dense_case();
    let (handle, _dir) = handle_of(&case, "beam_dense");
    let mut driver = driver_of(&handle, executor);
    let beams = beam_or_panic(&mut driver, case.request());
    assert_matches(&beams, case.prompt, &case.reference(), case.rules());
}

per_backend!(
    BackendChoice,
    assert_beam_matches_the_reference,
    beam_matches_the_host_reference_wgpu,
    beam_matches_the_host_reference_rocm,
    beam_matches_the_host_reference_ptx
);

/// SC-002 (narrowed: the chosen token's log-probability at every step, not the whole logits row, which
/// `Beam` does not expose): every returned beam's log-probability at every generated token equals the log-softmax of a
/// fresh prefill of the beam's own token list on the oracle, so the state a beam decodes from after a
/// reorder is the state of its own tokens. Mutation: skip the state reorder (`beam_reorder` returns
/// before the gather); each row keeps its old state and the log-probabilities diverge.
fn assert_reordered_beams_equal_a_fresh_prefill(choice: BackendChoice, case: Case, name: &str) {
    let Some(executor) = device(choice) else {
        return;
    };
    let (handle, _dir) = handle_of(&case, name);
    let mut driver = driver_of(&handle, executor);
    let beams = beam_or_panic(&mut driver, case.request());
    let oracle = case.oracle();
    for beam in &beams {
        for (k, &token) in beam.tokens.iter().enumerate() {
            let mut context = case.prompt.to_vec();
            context.extend_from_slice(&beam.tokens[..k]);
            let fresh = log_softmax(&oracle.last_logits(&context))[token as usize];
            assert!(
                (f64::from(beam.logprobs[k]) - fresh).abs() <= LOGPROB_TOLERANCE,
                "{:?} token {k}: logprob {} against a fresh prefill's {fresh}",
                beam.tokens,
                beam.logprobs[k]
            );
        }
    }
    assert_matches(&beams, case.prompt, &case.reference(), case.rules());
}

per_backend!(
    BackendChoice,
    |choice| assert_reordered_beams_equal_a_fresh_prefill(
        choice,
        dense_case(),
        "beam_dense_reorder"
    ),
    reordered_dense_beams_equal_a_fresh_prefill_wgpu,
    reordered_dense_beams_equal_a_fresh_prefill_rocm,
    reordered_dense_beams_equal_a_fresh_prefill_ptx
);

per_backend!(
    BackendChoice,
    |choice| assert_reordered_beams_equal_a_fresh_prefill(
        choice,
        recurrent_case(),
        "beam_recurrent_reorder"
    ),
    reordered_recurrent_beams_equal_a_fresh_prefill_wgpu,
    reordered_recurrent_beams_equal_a_fresh_prefill_rocm,
    reordered_recurrent_beams_equal_a_fresh_prefill_ptx
);

/// SC-003, width 1: beam search at width 1 and length penalty 0 returns the tokens `Driver::generate`
/// returns for the same greedy request (the end-of-sequence token, which `generate` leaves out, ends the
/// beam).
fn assert_width_one_beam_is_greedy(choice: BackendChoice) {
    let Some(executor) = device(choice) else {
        return;
    };
    let case = dense_case();
    let (handle, _dir) = handle_of(&case, "beam_greedy");
    let mut driver = driver_of(&handle, executor).serve(Layout::Contiguous);
    for prompt in GREEDY_PROMPTS {
        let generated = driver
            .generate(
                GenerateRequest {
                    prompt: prompt.to_vec(),
                    max_new: 12,
                    sampler: Sampler::greedy(),
                    stops: Vec::new(),
                    ignore_eos: false,
                },
                &mut nothing,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        let beams = beam_or_panic(&mut driver, request(prompt, 12, 1, 0.0));
        assert_eq!(beams.len(), 1, "{prompt:?}");
        let mut tokens = beams[0].tokens.clone();
        if beams[0].finish == FinishReason::Eos {
            assert_eq!(tokens.pop(), Some(EOS));
        }
        assert_eq!(tokens, generated.tokens, "{prompt:?}");
        assert_eq!(beams[0].finish, generated.finish, "{prompt:?}");
    }
}

per_backend!(
    BackendChoice,
    assert_width_one_beam_is_greedy,
    width_one_beam_is_the_greedy_loop_wgpu,
    width_one_beam_is_the_greedy_loop_rocm,
    width_one_beam_is_the_greedy_loop_ptx
);

/// SC-003, penalty 1: the beams are ranked by score over the prompt-included length. Mutation: count
/// the generated tokens only (`Ranking::score`: drop `self.prompt_len`); the two beams swap.
fn assert_length_penalty_counts_the_prompt(choice: BackendChoice) {
    let Some(executor) = device(choice) else {
        return;
    };
    let case = length_case();
    let (handle, _dir) = handle_of(&case, "beam_length");
    let mut driver = driver_of(&handle, executor);
    let beams = beam_or_panic(&mut driver, case.request());
    assert_matches(&beams, case.prompt, &case.reference(), case.rules());
}

per_backend!(
    BackendChoice,
    assert_length_penalty_counts_the_prompt,
    penalty_one_ranks_by_the_prompt_included_length_wgpu,
    penalty_one_ranks_by_the_prompt_included_length_rocm,
    penalty_one_ranks_by_the_prompt_included_length_ptx
);

/// SC-004: on a driver that prepared the paged pool, `beam` runs on contiguous entries of its own and
/// returns the beams a contiguous driver returns. Mutation: run the row gather over the paged pool
/// (`beam_key`: take the serving layout's `kv`); the paged pool's leading axis is blocks, not rows,
/// and the beams diverge.
fn assert_paged_beam_runs_on_contiguous_entries(choice: BackendChoice) {
    let Some(first) = device(choice) else {
        return;
    };
    let Some(second) = device(choice) else {
        return;
    };
    let case = dense_case();
    let (handle, _dir) = handle_of(&case, "beam_paged");
    let mut contiguous = driver_of(&handle, first);
    let wanted = beam_or_panic(&mut contiguous, case.request());
    drop(contiguous);

    let paged = driver_of(&handle, second);
    let mut paged = paged
        .prepare(&ServingShapes {
            layout: Layout::Paged(PoolShape {
                blocks: nz(16),
                max_seqs: nz(2),
            }),
            rows: &[nz(2)],
            heads: &[Head::GREEDY],
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[],
            chain: None,
        })
        .unwrap_or_else(|e| panic!("{e}"));
    let caps = paged.caps().expect("prepared").clone();
    let beams = beam_or_panic(&mut paged, case.request());
    assert_eq!(beams, wanted, "the paged driver's beams differ");
    assert_matches(&beams, case.prompt, &case.reference(), case.rules());
    assert_eq!(
        paged.caps(),
        Some(&caps),
        "beam entries are not serving entries: the caps are unchanged"
    );
}

per_backend!(
    BackendChoice,
    assert_paged_beam_runs_on_contiguous_entries,
    paged_beam_runs_on_contiguous_entries_wgpu,
    paged_beam_runs_on_contiguous_entries_rocm,
    paged_beam_runs_on_contiguous_entries_ptx
);

// ---------------------------------------------------------------------------------------------
// Refusals and the last step, on a fake device: the real driver, engine and planner; only the device is
// fake. Its reads return `fill` in every byte.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
#[error("fake device error")]
struct FakeError;

struct FakeDevice {
    fill: u8,
}

impl Device for FakeDevice {
    type Buffer = ();
    type Kernel = ();
    type Recording = ();
    type Error = FakeError;

    fn copy_mechanism(&mut self) -> poot_executor::CopyMechanism<'_, Self::Error> {
        poot_executor::CopyMechanism::Native
    }
    fn backend_name(&self) -> &'static str {
        "fake"
    }

    fn target(&self) -> Target {
        Target::Gpu {
            backend: Backend::SpirvVulkan,
            caps: DeviceCaps::wgpu_rdna3_igpu(),
        }
    }
    fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
        Vec::new()
    }
    fn allocate(
        &mut self,
        _role: BufferRole,
        _storage: poot_target::BufferStorage,
        _elems: usize,
    ) -> Result<(), FakeError> {
        Ok(())
    }
    /// This fake keeps no storage, so it states no bound: the engine's length check passes every
    /// dispatch it records.
    fn buffer_elems(&self, _buffer: &()) -> usize {
        usize::MAX
    }

    fn load_kernel(
        &mut self,
        _key: &str,
        _kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<(), FakeError> {
        Ok(())
    }
    fn begin(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
    fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), FakeError> {
        Ok(())
    }
    fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), FakeError> {
        Ok(())
    }
    fn finish(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
    fn replay(&mut self, _recording: &()) -> Result<(), FakeError> {
        Ok(())
    }
    fn write_with(
        &mut self,
        _dst: &(),
        _offset: usize,
        _len: usize,
        _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), FakeError> {
        Ok(())
    }
    fn read(&mut self, _src: &(), out: &mut [u8]) -> Result<(), FakeError> {
        out.fill(self.fill);
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
    fn device_time(&self) -> poot_executor::DeviceTime {
        poot_executor::DeviceTime::Unknown
    }
    fn abort(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
}

/// A driver over the dense fixture on the fake device. `fill` 0 reads all-zero logits (uniform), `0xff`
/// reads NaN.
fn fake_driver(fill: u8) -> (Driver, poot_test_util::UniqueTempPath) {
    let (handle, dir) = load(&family::fixture(), "beam_fake");
    let driver = Driver::new(
        handle,
        Box::new(Engine::new(FakeDevice { fill })),
        options(),
    )
    .unwrap();
    (driver, dir)
}

fn beam_refusal<S: State>(driver: &mut Driver<S>, request: BeamRequest) -> BeamError {
    driver.beam(request).expect_err("the request is refused")
}

/// A sequence open on the driver refuses `beam` with `Busy { open: 1 }` before anything compiles, and the
/// sequence is unharmed: it still prefills and releases. Mutation: drop the open-sequence check in
/// `beam`; the call goes on to prepare entries and the first assertion is red.
#[test]
fn beam_on_a_driver_with_an_open_sequence_is_busy_and_leaves_it_unharmed() {
    use poot_llm::driver::{AdapterRef, Admission, Release, RowWork, SeqRequest, StepRow};
    // The prefill below picks through the greedy suffix, whose (token, non-finite index) pair of all-ones
    // words is the token -1 with no fault.
    let (driver, _dir) = fake_driver(0xff);
    let mut driver = driver
        .prepare(&ServingShapes {
            layout: Layout::Contiguous,
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY, Head::Logits],
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[],
            chain: None,
        })
        .unwrap();
    let prompt = [3u32, 17, 8];
    let Admission::Opened { seq, .. } = driver
        .open(SeqRequest::new(&prompt, 4, AdapterRef::Base))
        .unwrap()
    else {
        panic!("a fresh driver admits one sequence")
    };
    let (entries, compiles) = (driver.prepared_entries(), driver.stats().compiles);
    match beam_refusal(&mut driver, request(&[1, 2], 4, 2, 1.0)) {
        BeamError::Busy { open: 1 } => {}
        other => panic!("not Busy {{ open: 1 }}: {other:?}"),
    }
    assert_eq!(driver.prepared_entries(), entries, "nothing prepared");
    assert_eq!(driver.stats().compiles, compiles, "nothing compiled");
    let mut sampler = Sampler::greedy();
    let mut rows = [StepRow {
        seq,
        work: RowWork::Prefill { upto: prompt.len() },
        sampler: &mut sampler,
    }];
    while driver.step(&mut rows).unwrap().remove(0).tokens.is_empty() {}
    driver.release(seq, Release::Finished).unwrap();
}

/// A length penalty that is not finite is refused typed, before anything compiles. Mutation: drop the
/// check in `beam`; NaN reaches the score and the call returns beams.
#[test]
fn a_non_finite_length_penalty_is_refused() {
    let (mut driver, _dir) = fake_driver(0);
    for penalty in [f32::NAN, f32::INFINITY] {
        match beam_refusal(&mut driver, request(&[1, 2], 4, 2, penalty)) {
            BeamError::LengthPenalty(got) => assert_eq!(got.to_bits(), penalty.to_bits()),
            other => panic!("not LengthPenalty: {other:?}"),
        }
    }
    assert_eq!(driver.stats().compiles, 0);
}

/// Logits that are NaN rank nothing: the first step names its beam and step. Mutation: let
/// `top_logprobs` accept NaN logits; the search ranks NaN scores and returns beams.
#[test]
fn nan_logits_are_refused_naming_the_beam_and_step() {
    let (mut driver, _dir) = fake_driver(0xff);
    match beam_refusal(&mut driver, request(&[1, 2, 3], 4, 2, 1.0)) {
        BeamError::NonFiniteLogits { beam: 0, step: 0 } => {}
        other => panic!("not NonFiniteLogits at beam 0 step 0: {other:?}"),
    }
}

/// The last expansion runs no reorder: nothing decodes after it. At width 2 over uniform logits the first
/// expansion's parents are `[0, 0]`, a permutation that is not the identity, so a reorder would be one more
/// executor step than the same request at width 1 (whose parents are the identity). Mutation: reorder
/// before the `step + 1 == max_new` break in `beam`; the width-2 run takes one step more.
#[test]
fn the_last_expansion_runs_no_reorder() {
    let steps = |width| {
        let (mut driver, _dir) = fake_driver(0);
        let beams = driver
            .beam(request(&[1, 2, 3], 1, width, 1.0))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(beams.len(), width);
        driver.stats().steps
    };
    assert_eq!(steps(2), steps(1));
}
