//! `DraftPolicy::Mtp { k }` through the driver (POOT-1111).
//!
//! Two halves. The host half runs the production driver over a fake family and a device that does
//! nothing: the typed refusals, prefix reuse, resume and the validity of a stored draft are the
//! driver's bookkeeping, which needs no values. The device half runs the GLM-5.3-Flash fixture, the
//! one family with an MTP layer, on each backend: the round's accepted tokens, the drafts the policy
//! proposes against an oracle chain evaluated from scratch on the CPU over the traced graphs, and a
//! step of rows at different widths.

use std::collections::{BTreeMap, BTreeSet};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::{Arc, Mutex};

use poot_executor::{Device, Dispatch, Engine};
use poot_graph_ir::op::SampleRule;
use poot_graph_ir::ops::sampling::SampleSpec;
use poot_graph_ir::{BinOp, Graph, Slot, SlotKey};
use poot_graph_plan::{CompileLimits, CompileOptions, Target};
use poot_models::components::embed::{EMBEDS, embeds_input};
use poot_models::components::standard::Step;
use poot_models::io::{ModelSchema, Presence, ProgramId, ProgramSpec, RequestInputs, StateSpec};
use poot_models::model::{
    FamilyKey, HiddenCarry, LogitRows, Model, ModelConfig, ModelError, ModelOutput, StepShape,
    StepSpec, TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, ModelParts, RawConfig, Registry};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightMap, WeightStore};
use poot_runtime_common::BufferRole;
use poot_target::{Backend, DeviceCaps};
use poot_tensor::{DType, HostTensor};
use poot_test_util::per_backend;
use poot_text::{ChatFormat, HermesToolCodec};

use crate::core::sampler::Sampler;
use crate::driver::block_table::BLOCK_SIZE;
use crate::driver::device_tests::{NO_EOS, fixture_text, nz, open, options};
use crate::driver::error::{DriverError, InvalidOptions, Unsupported};
use crate::driver::speculate::{NotSpeculated, ProposalObserver};
use crate::driver::tests::tiny_text;
use crate::driver::{
    AdapterRef, Admission, BackendChoice, DraftPolicy, Driver, DriverOptions, Feature,
    GenerationId, Head, Incompatible, Layout, LoraIdx, ModelHandle, PassTiming, PoolShape,
    PrefixOutcome, PrefixSkip, PreparedSetLimits, Release, RowResult, RowSpeculation, RowWork,
    SeqId, SeqRequest, Serving, ServingShapes, StepRow, Warm,
};

// ---------------------------------------------------------------------------------------------
// The host half: a fake family with an MTP program and a device that does nothing.
// ---------------------------------------------------------------------------------------------

const TOY: FamilyKey = FamilyKey::new("poot_fixture_mtp_toy");
const TOY_VOCAB: usize = 12;
const TOY_WIDTH: usize = 4;
const TOY_CAPACITY: usize = 64;

/// What the fake family's MTP program is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// An MTP program over the carry, KV-only.
    Plain,
    /// An MTP program that carries a `Recurrent` value of its own.
    RecurrentMtp,
    /// A family with no MTP program: its schema declares no `draft` program.
    NoMtp,
    /// A target whose own programs fold a `Recurrent` value, beside a KV-only MTP program.
    RecurrentTarget,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::RecurrentMtp => "recurrent_mtp",
            Self::NoMtp => "no_mtp",
            Self::RecurrentTarget => "recurrent_target",
        }
    }

    fn parse(name: &str) -> Self {
        match name {
            "plain" => Self::Plain,
            "recurrent_mtp" => Self::RecurrentMtp,
            "no_mtp" => Self::NoMtp,
            "recurrent_target" => Self::RecurrentTarget,
            other => panic!("no toy kind {other}"),
        }
    }
}

#[derive(Debug)]
struct Toy {
    config: ModelConfig,
    schema: ModelSchema,
    weights: WeightMap,
    kind: Kind,
}

impl Model for Toy {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, spec: &StepSpec, b: poot_graph_ir::Builder) -> Result<Graph, TraceError> {
        let shape = spec.shape;
        let draft = spec.program == ProgramId::DRAFT;
        let mut step = Step::new(&b, shape)?;
        let embeds = spec.carries(EMBEDS);
        if embeds {
            step = step.with_embeds(&b, TOY_WIDTH)?;
        }
        let (n, rows) = (shape.tokens.get(), shape.rows.get());
        let token = b.cast(
            b.guard_index_bounds(step.tokens(), TOY_VOCAB)
                .expect("the step's ids are a rank-2 I32 tensor")
                .guarded,
            DType::F32,
        )?;
        let seed = b.binary(BinOp::Add, token, b.cast(step.pos(), DType::F32)?)?;
        let mut hidden =
            b.broadcast(b.reshape(seed, vec![rows, n, 1])?, vec![rows, n, TOY_WIDTH])?;
        if let Some(rows_in) = step.embeds() {
            hidden = rows_in.scatter(&b, hidden)?;
        }
        let state = if draft {
            let carry = step
                .hidden_carry(&b, TOY_WIDTH)?
                .expect("check_draft_step requires the carry");
            let read = step.carried_hidden(&b, &carry)?;
            if self.kind == Kind::RecurrentMtp {
                let own = step.recurrent_state(&b, "toy.mtp", DType::F32, &[TOY_WIDTH])?;
                let last = b.slice(seed, 1, n - 1, n)?;
                let folded = b.binary(
                    BinOp::Add,
                    own.current(),
                    b.broadcast(last, vec![rows, TOY_WIDTH])?,
                )?;
                step.carry_recurrent(&b, own, folded)?;
            }
            let mixed = b.binary(BinOp::Add, read, hidden)?;
            step.carry_last_hidden(&b, &carry, mixed)?;
            mixed
        } else {
            if self.kind == Kind::RecurrentTarget {
                let own = step.recurrent_state(&b, "toy.target", DType::F32, &[TOY_WIDTH])?;
                let last = b.slice(seed, 1, n - 1, n)?;
                let folded = b.binary(
                    BinOp::Add,
                    own.current(),
                    b.broadcast(last, vec![rows, TOY_WIDTH])?,
                )?;
                step.carry_recurrent(&b, own, folded)?;
            }
            step.carry_hidden(&b, hidden)?;
            hidden
        };
        let out = if shape.logits == LogitRows::All { n } else { 1 };
        let column = b.slice(state, 2, 0, 1)?;
        let logits = b.broadcast(b.slice(column, 1, n - out, n)?, vec![rows, out, TOY_VOCAB])?;
        Ok(step.finish(b, logits)?)
    }
}

fn build_toy(raw: &RawConfig<'_>, _: &WeightStore) -> Result<ModelParts, ModelError> {
    let RawConfig::HfJson { config, .. } = raw else {
        panic!("the fixture is an HF checkpoint")
    };
    let kind = Kind::parse(config["kind"].as_str().expect("a toy kind"));
    // The graphs carry the hand-off and the recurrent values the kind folds, and no KV cache:
    // `trace` checks the schema declares exactly the states the traced graph carries. The `step`
    // program reads the embedding rows as an optional carriable input.
    let mut step = ProgramSpec::new(ProgramId::STEP, ModelOutput::Logits { vocab: TOY_VOCAB })
        .carrying(StateSpec::hand_off())
        .reading(EMBEDS, Presence::Optional);
    if kind == Kind::RecurrentTarget {
        step = step.carrying(StateSpec::recurrent("toy.target"));
    }
    let mut schema = ModelSchema::builder()
        .input(embeds_input(TOY_WIDTH))
        .step(step);
    if kind != Kind::NoMtp {
        let mut draft =
            ProgramSpec::new(ProgramId::DRAFT, ModelOutput::Logits { vocab: TOY_VOCAB })
                .carrying(StateSpec::hand_off());
        if kind == Kind::RecurrentMtp {
            draft = draft.carrying(StateSpec::recurrent("toy.mtp"));
        }
        schema = schema.draft(draft);
    }
    Ok(ModelParts::new(Box::new(Toy {
        schema: schema.build_for(TOY)?,
        config: ModelConfig {
            family: TOY,
            vocab: TOY_VOCAB,
            max_positions: TOY_CAPACITY,
            eos: BTreeSet::from([TOY_VOCAB as u32 - 1]),
            bos: None,
            prompt_bos: None,
            prefill_granule: nz(1),
            chat: ChatFormat::ChatML,
            tool_codec: Some(Arc::new(HermesToolCodec)),
        },
        weights: WeightMap::default(),
        kind,
    })))
}

fn toy_handle(kind: Kind) -> Arc<ModelHandle> {
    let registry = Registry::of(&[FamilyEntry {
        family: TOY,
        keys: &[(ConfigSource::HfModelType, "poot_fixture_mtp_toy")],
        build: build_toy,
    }])
    .unwrap();
    let config = serde_json::json!({
        "model_type": "poot_fixture_mtp_toy",
        "kind": kind.name(),
    });
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    Arc::new(
        ModelHandle::from_checkpoint(&raw, WeightStore::default(), &registry, |_| Ok(tiny_text()))
            .unwrap(),
    )
}

#[derive(Debug, thiserror::Error)]
#[error("fake device error")]
struct FakeError;

/// A device that does nothing: the driver, engine and planner are real, so `prepare` compiles and a
/// step runs its bookkeeping. Every pick it reads back is token 0.
struct FakeDevice;

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
        // The sampling suffix's output is `(token, first non-finite index)` per pick: token 0 and
        // no fault, for every row.
        for pair in out.as_chunks_mut::<8>().0 {
            pair[..4].copy_from_slice(&0i32.to_le_bytes());
            pair[4..].copy_from_slice(&(-1i32).to_le_bytes());
        }
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

fn mtp(k: usize) -> DraftPolicy {
    DraftPolicy::Mtp { k: nz(k) }
}

/// The options of a toy driver under `draft`, its carry window as the policy derives it.
fn toy_options(draft: DraftPolicy) -> DriverOptions {
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(TOY_CAPACITY),
        prefill_chunk: nz(BLOCK_SIZE),
        trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(BLOCK_SIZE)),
        prepared: PreparedSetLimits {
            max_entries: nz(256),
            max_retained_bytes: NonZeroU64::new(1 << 40).unwrap(),
        },
        charge: crate::driver::program_retention,
        hidden_window: draft.carry_window(),
        draft,
    }
}

fn toy_driver(kind: Kind, draft: DraftPolicy) -> Result<Driver, DriverError> {
    Driver::new(
        toy_handle(kind),
        Box::new(Engine::new(FakeDevice)),
        toy_options(draft),
    )
}

fn toy_shapes(layout: Layout) -> ServingShapes<'static> {
    ServingShapes {
        layout,
        rows: &[NonZeroUsize::MIN],
        heads: &[Head::GREEDY],
        warm: Warm::Admitted,
        windows: &[],
        adapters: &[],
        chain: None,
    }
}

fn paged() -> Layout {
    Layout::Paged(PoolShape {
        blocks: nz(8),
        max_seqs: nz(2),
    })
}

/// Open `prompt`, prefill it in the driver's own pieces and commit the first token.
fn open_prefilled(driver: &mut Driver<Serving>, prompt: &[u32], max_new: usize) -> (SeqId, usize) {
    open_prefilled_with(driver, SeqRequest::new(prompt, max_new, AdapterRef::Base))
}

/// [`open_prefilled`] for a request that may carry inputs.
fn open_prefilled_with(driver: &mut Driver<Serving>, request: SeqRequest<'_>) -> (SeqId, usize) {
    let prompt = request.prompt;
    let Admission::Opened { seq, reused } = driver.open(request).unwrap() else {
        panic!("the driver admits the sequence")
    };
    let mut sampler = Sampler::greedy();
    loop {
        let mut rows = [StepRow {
            seq,
            work: RowWork::Prefill { upto: prompt.len() },
            sampler: &mut sampler,
        }];
        let result = driver.step(&mut rows).unwrap().remove(0);
        if !result.tokens.is_empty() {
            break;
        }
    }
    driver.commit(seq, 1, &mut sampler).unwrap();
    (seq, reused)
}

/// Every proposal the driver's policy made, in order, as the observer was told them.
type ProposalLog = Arc<Mutex<Vec<(SeqId, Vec<u32>)>>>;

struct RecordProposals(ProposalLog);

impl ProposalObserver for RecordProposals {
    fn proposed(&mut self, seq: SeqId, drafts: &[u32]) {
        self.0.lock().unwrap().push((seq, drafts.to_vec()));
    }
}

/// Install a proposal recorder on `driver`; the oracle rows compare what was proposed, which no stat or
/// event carries.
fn log_proposals(driver: &mut Driver<Serving>) -> ProposalLog {
    let log = ProposalLog::default();
    driver.speculation.observer = Some(Box::new(RecordProposals(Arc::clone(&log))));
    log
}

fn decode(driver: &mut Driver<Serving>, seq: SeqId, sampler: &mut Sampler) -> RowResult {
    let mut rows = [StepRow {
        seq,
        work: RowWork::Decode { ahead: &[] },
        sampler,
    }];
    driver.step(&mut rows).unwrap().remove(0)
}

/// An MTP policy reads the carry its verify window wrote: options that name none, or one narrower
/// than the window, are refused typed at construction, and the policy derives the right window.
/// Mutation: drop the `MtpCarryWindow` return in `Driver::check_options`; both constructions succeed
/// and the row is red.
#[test]
fn an_mtp_policy_without_a_carry_window_is_refused_typed() {
    for (window, k) in [(None, 2), (Some(2), 2)] {
        let draft = mtp(k);
        let options = DriverOptions {
            hidden_window: window.map(nz),
            ..toy_options(draft)
        };
        let Err(error) = Driver::new(
            toy_handle(Kind::Plain),
            Box::new(Engine::new(FakeDevice)),
            options,
        ) else {
            panic!("a missing or narrow carry window is refused");
        };
        assert!(
            matches!(
                error,
                DriverError::InvalidOptions(InvalidOptions::MtpCarryWindow { k: refused, window: found })
                    if refused == k && found == window
            ),
            "{error:?}"
        );
    }
    assert_eq!(mtp(3).carry_window(), Some(nz(4)));
    assert_eq!(DraftPolicy::Off.carry_window(), None);
    assert!(toy_driver(Kind::Plain, mtp(2)).is_ok());
}

/// `Mtp` with the draft program carrying recurrent state, refused by `prepare` of a family whose MTP
/// program folds a recurrent value.
pub(in crate::driver) fn witness_mtp_recurrent_draft_state() -> DriverError {
    toy_driver(Kind::RecurrentMtp, mtp(2))
        .unwrap()
        .prepare(&toy_shapes(Layout::Contiguous))
        .expect_err("a recurrent MTP program is refused")
        .error
}

/// A family whose MTP program carries recurrent state is refused typed at `prepare`: the snapshot
/// schema is the target's decode program, so the state would never be rolled back. A KV-only MTP
/// program prepares. Mutation: drop the `RecurrentDraftState` compose in `Driver::prepared`; the family
/// prepares and the row is red.
#[test]
fn an_mtp_program_with_recurrent_state_is_refused_typed() {
    let error = witness_mtp_recurrent_draft_state();
    assert!(
        matches!(
            error,
            DriverError::Unsupported(Unsupported::Incompatible(Incompatible {
                first: Feature::Mtp,
                second: Feature::RecurrentDraftState,
                ..
            }))
        ),
        "{error:?}"
    );
    toy_driver(Kind::Plain, mtp(2))
        .unwrap()
        .prepare(&toy_shapes(Layout::Contiguous))
        .expect("a KV-only MTP program prepares");
}

/// A family whose schema declares no `draft` program is refused typed before anything compiles.
/// Mutation: none of the driver's; the refusal is `Program::spec_in` (the schema has no draft
/// program), reported through `Unsupported::Trace`.
#[test]
fn a_family_without_an_mtp_program_is_refused_typed() {
    let driver = toy_driver(Kind::NoMtp, mtp(2)).unwrap();
    let error = driver
        .prepare(&toy_shapes(Layout::Contiguous))
        .expect_err("a family with no MTP program is refused")
        .error;
    assert!(
        matches!(
            error,
            DriverError::Unsupported(Unsupported::Trace(TraceError::ProgramUnsupported {
                program: ProgramId::DRAFT,
                ..
            }))
        ),
        "{error:?}"
    );
}

/// `Mtp` with beam search, refused by `beam`.
pub(in crate::driver) fn witness_mtp_beam() -> crate::driver::beam::BeamError {
    use crate::driver::beam::BeamRequest;
    let mut driver = toy_driver(Kind::Plain, mtp(2)).unwrap();
    driver
        .beam(BeamRequest {
            prompt: vec![1, 2, 3],
            max_new: 2,
            width: nz(2),
            length_penalty: 0.0,
            ignore_eos: true,
        })
        .expect_err("beam search is refused under an MTP policy")
}

/// `Mtp` with an adapter, refused by `open` of a request that names one: the layer drafts for the base
/// weights, so the refusal comes before the registry is asked whether the generation exists.
pub(in crate::driver) fn witness_mtp_adapters() -> DriverError {
    let mut driver = toy_driver(Kind::Plain, mtp(2))
        .unwrap()
        .prepare(&toy_shapes(Layout::Contiguous))
        .unwrap();
    let adapter = AdapterRef::Adapter {
        generation: GenerationId::unissued(1),
        index: LoraIdx::new(1).expect("an index of one"),
    };
    driver
        .open(SeqRequest::new(&[1, 2, 3], 4, adapter))
        .expect_err("an adapter request is refused under an MTP policy")
}

/// Beam search runs the target's programs without the carry: refused typed under an MTP policy.
/// Mutation: drop the `Beam` compose in `Driver::prepared`; the search goes on into the beam's own
/// preparation, which this family has no state for, and the row is red.
#[test]
fn beam_search_is_refused_typed_under_an_mtp_policy() {
    use crate::driver::beam::BeamError;
    let error = witness_mtp_beam();
    assert!(
        matches!(
            error,
            BeamError::Driver(DriverError::Unsupported(Unsupported::Incompatible(
                Incompatible {
                    first: Feature::Mtp,
                    second: Feature::Beam,
                    ..
                }
            )))
        ),
        "{error:?}"
    );
}

/// A prompt with embedding rows runs under an MTP policy but is never speculated: the layer would
/// embed the placeholder tokens the rows replace, so a draft program that does not read the sequence's
/// request inputs serves no sequence that carries one. The row decodes one token, reports
/// `NotSpeculated::Inputs`, counts `skipped_inputs` and asks the policy for no proposal; the control,
/// the same prompt without rows, speculates. Mutation: drop the `drafts_for` arm of the decode row's
/// speculation choice in `Driver::step`; the sequence speculates, its proposal is observed and the
/// first assertion is red.
#[test]
fn a_prompt_with_embedding_rows_is_not_speculated_under_an_mtp_policy() {
    let embeds =
        poot_models::components::embed::embeds_inputs(TOY_WIDTH, vec![1], vec![0.0; TOY_WIDTH])
            .unwrap();
    let run = |inputs: RequestInputs| {
        let shapes = toy_shapes(Layout::Contiguous);
        let mut driver = toy_driver(Kind::Plain, mtp(2))
            .unwrap()
            .prepare(&shapes)
            .unwrap();
        driver.prepare_carried(&shapes, &[EMBEDS]).unwrap();
        let proposals = log_proposals(&mut driver);
        let request = SeqRequest::new(&[1, 2, 3, 4], 16, AdapterRef::Base).with_inputs(inputs);
        let (seq, _) = open_prefilled_with(&mut driver, request);
        let mut sampler = Sampler::greedy();
        let result = decode(&mut driver, seq, &mut sampler);
        let made = proposals.lock().unwrap().len();
        (result, made, driver.stats().speculation)
    };
    let (result, made, stats) = run(embeds);
    assert_eq!(
        result.speculation,
        RowSpeculation::NotSpeculated(NotSpeculated::Inputs)
    );
    assert_eq!(
        result.tokens.len(),
        1,
        "an unspeculated row decodes one token"
    );
    assert_eq!((made, stats.skipped_inputs, stats.rounds), (0, 1, 0));

    let (control, made, stats) = run(RequestInputs::none());
    assert!(
        matches!(control.speculation, RowSpeculation::Verified { .. }),
        "control: without rows the same prompt speculates: {:?}",
        control.speculation
    );
    assert_eq!(
        control.tokens[0], result.tokens[0],
        "both rows decode the same first token"
    );
    assert!(made > 0 && stats.rounds == 1 && stats.skipped_inputs == 0);
}

/// Prefix reuse under an MTP policy is refused, typed: the layer's cache at a block's last position
/// depends on the token after the block, which a later prompt sharing the block need not share. A
/// finished sequence registers nothing and the next identical prompt reuses nothing; under `Off` the
/// same run registers its two full blocks and the next prompt reuses them (the control the row
/// differs from). Mutation: drop the `mtp` arm of `release`; the sequence registers its blocks and
/// the second prompt reuses 32 tokens.
#[test]
fn prefix_reuse_is_refused_under_an_mtp_policy() {
    let prompt: Vec<u32> = (0..2 * BLOCK_SIZE as u32 + 1).map(|i| 1 + i % 9).collect();
    let run = |draft: DraftPolicy| {
        let mut driver = toy_driver(Kind::Plain, draft)
            .unwrap()
            .prepare(&toy_shapes(paged()))
            .unwrap();
        let (seq, _) = open_prefilled(&mut driver, &prompt, 4);
        let outcome = driver.release(seq, Release::Finished).unwrap();
        let Admission::Opened { reused, .. } = driver
            .open(SeqRequest::new(&prompt, 4, AdapterRef::Base))
            .unwrap()
        else {
            panic!("the pool admits the sequence")
        };
        (outcome, reused)
    };
    assert_eq!(
        run(DraftPolicy::Off),
        (
            PrefixOutcome::Registered {
                tokens: 2 * BLOCK_SIZE
            },
            2 * BLOCK_SIZE
        ),
        "control: without the policy the prompt's blocks are cached and reused"
    );
    assert_eq!(
        run(mtp(2)),
        (PrefixOutcome::Skipped(PrefixSkip::MtpState), 0),
        "under Mtp nothing is registered and nothing reused"
    );
}

/// A resumed conversation reuses one position less under an MTP policy: the layer's cache at the last
/// shared position was computed from the token that followed it in the old context, so that position
/// runs again from the target's hidden state. A target with recurrent state has folded that position
/// and cannot step back: it restarts from zero. Mutation: drop the `saturating_sub` in `resume`; the
/// reused count equals the control's (and a recurrent target would replay a position it folded).
#[test]
fn a_resumed_sequence_reuses_one_position_less_under_an_mtp_policy() {
    let reused = |kind: Kind, draft: DraftPolicy| {
        let mut driver = toy_driver(kind, draft)
            .unwrap()
            .prepare(&toy_shapes(Layout::Contiguous))
            .unwrap();
        let (seq, _) = open_prefilled(&mut driver, &[1, 2, 3, 4, 5, 6], 8);
        driver.resume(seq, &[1, 2, 3, 4, 5, 6, 7, 8], 8).unwrap()
    };
    assert_eq!(
        reused(Kind::Plain, DraftPolicy::Off),
        6,
        "control: the whole shared prefix"
    );
    assert_eq!(
        reused(Kind::Plain, mtp(2)),
        5,
        "one position less under Mtp"
    );
    assert_eq!(
        reused(Kind::RecurrentTarget, DraftPolicy::Off),
        6,
        "control: a recurrent state that folded the whole prefix reuses it"
    );
    assert_eq!(
        reused(Kind::RecurrentTarget, mtp(2)),
        0,
        "a recurrent state cannot step back one position: it restarts"
    );
}

/// The draft a step leaves is valid only while the sequence is exactly as long as the step's whole
/// result made it: a caller that commits fewer tokens than it was handed gets no proposal next round
/// (the layer's draft followed a token the sequence dropped), and a caller that commits the whole
/// result gets one. Mutation: drop the length check in `propose_mtp`; the short commit proposes the
/// stale draft and the first assertion is red.
#[test]
fn a_short_commit_leaves_no_stale_draft() {
    let mut driver = toy_driver(Kind::Plain, mtp(2))
        .unwrap()
        .prepare(&toy_shapes(Layout::Contiguous))
        .unwrap();
    let (seq, _) = open_prefilled(&mut driver, &[1, 2, 3, 4], 16);
    let mut sampler = Sampler::greedy();
    // The fake device picks token 0 for every position and the layer's draft is token 0 too, so
    // every draft verifies and a round returns more than one token.
    let full = decode(&mut driver, seq, &mut sampler);
    assert!(
        matches!(full.speculation, RowSpeculation::Verified { .. }),
        "the first round after the prompt has the catch-up's draft: {:?}",
        full.speculation
    );
    driver.commit(seq, full.tokens.len(), &mut sampler).unwrap();
    let again = decode(&mut driver, seq, &mut sampler);
    assert!(
        matches!(again.speculation, RowSpeculation::Verified { .. }),
        "a whole commit keeps the draft: {:?}",
        again.speculation
    );
    assert!(
        again.tokens.len() > 1,
        "the round kept drafts, so a one-token commit is a short one: {:?}",
        again.tokens
    );
    driver.commit(seq, 1, &mut sampler).unwrap();
    let next = decode(&mut driver, seq, &mut sampler);
    assert_eq!(
        next.speculation,
        RowSpeculation::NotSpeculated(NotSpeculated::NoProposal),
        "a short commit leaves no draft to propose"
    );
}

// ---------------------------------------------------------------------------------------------
// The device half: the GLM-5.3-Flash fixture, the one family with an MTP layer.
// ---------------------------------------------------------------------------------------------

/// Positions of every device driver here: the fixture's prompt and the tokens a row generates.
const CAPACITY: usize = 32;
const PROMPT: [u32; 7] = [3, 17, 40, 8, 25, 11, 2];
/// A second prompt, shorter and different, for the rows with two sequences.
const OTHER_PROMPT: [u32; 4] = [9, 30, 5, 21];
/// Drafts a round proposes.
const K: usize = 3;
/// Tokens each device row generates.
const NEW: usize = 18;

/// What the fixture's head is: its random weights, in which the layer's draft rarely matches the
/// target's pick (so rounds reject), or zeroed, in which every logit is exactly 0 and both the target
/// and the layer pick token 0 (so every round keeps all its drafts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixtureHead {
    Random,
    Zeroed,
}

fn glm_handle(head: FixtureHead) -> Arc<ModelHandle> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "glm53_flash")
        .expect("glm53_flash is a shipped family");
    let fixture = poot_models::test_support::family_fixtures::base(entry.family);
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    let mut store = WeightStore::builder();
    for (key, entry) in fixture.store.iter() {
        let entry = match (head, key.as_str(), entry) {
            (FixtureHead::Zeroed, "lm_head.weight", WeightEntry::Dense(dense)) => {
                let zeros = vec![0u8; dense.bytes().len()];
                WeightEntry::Dense(
                    DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), zeros.into())
                        .unwrap(),
                )
            }
            _ => entry.clone(),
        };
        store.insert(key.clone(), entry).unwrap();
    }
    Arc::new(
        ModelHandle::from_checkpoint(&raw, store.build(), &registry, |_| Ok(fixture_text()))
            .unwrap(),
    )
}

fn glm_driver(
    choice: BackendChoice,
    handle: &Arc<ModelHandle>,
    draft: DraftPolicy,
) -> Option<Driver> {
    let executor = open(choice)?;
    let options = DriverOptions {
        hidden_window: draft.carry_window(),
        draft,
        ..options(8, CAPACITY)
    };
    Some(Driver::new(Arc::clone(handle), executor, options).unwrap())
}

/// The k drafts an MTP chain proposes after `tokens`, from scratch on the CPU over the family's own
/// traced graphs: the target prefills every token with the carry, the layer catches up over all but
/// the last position with the token that follows each, and `k - 1` one-token steps each feed the
/// draft before it at the position after and read the state that step left in the carry. Nothing
/// is shared with the driver's cache, rollback or catch-up: one fresh state per call.
fn oracle_drafts(handle: &ModelHandle, tokens: &[u32], k: usize) -> Vec<u32> {
    let model = handle.model();
    let n = tokens.len();
    let carry = HiddenCarry {
        positions: nz(n.max(k + 1)),
    };
    let formats = poot_graph_plan::WeightFormats::from_weight_map(model.weights());
    let mut state: BTreeMap<String, HostTensor> = BTreeMap::new();
    let mut run = |program: ProgramId, fed: &[u32], start: usize| -> Vec<f32> {
        let shape = StepShape::new(nz(1), nz(fed.len()), nz(CAPACITY)).with_hidden(Some(carry));
        let graph = model
            .trace(
                &StepSpec::new(program, shape),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let graph = poot_graph_plan::bind_packed_weights(&graph, &formats).unwrap();
        let slots = [
            (
                SlotKey::new(Slot::Token, None),
                HostTensor::i32(vec![1, fed.len()], fed.iter().map(|&t| t as i32).collect()),
            ),
            (
                SlotKey::new(Slot::Pos, None),
                HostTensor::i32(
                    vec![1, fed.len()],
                    (start as i32..).take(fed.len()).collect(),
                ),
            ),
        ];
        let names: Vec<String> = graph
            .state()
            .map(|(input, _)| graph.meta(input).name.clone().expect("state is named"))
            .collect();
        let held: Vec<HostTensor> = graph
            .state()
            .zip(&names)
            .map(|((input, _), name)| {
                state.get(name).cloned().unwrap_or_else(|| {
                    let aval = graph.aval(input);
                    match aval.dtype {
                        DType::I32 => HostTensor::i32(aval.shape.clone(), vec![0; aval.numel()]),
                        _ => HostTensor::zeros(aval.shape.clone()),
                    }
                })
            })
            .collect();
        let (logits, next) = poot_eval::test_support::weight_map_oracle::eval_mapped(
            &graph,
            handle.store(),
            model.weights(),
            &slots,
            &held,
        );
        state.extend(names.into_iter().zip(next));
        logits.as_f32().unwrap().to_vec()
    };
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    run(ProgramId::STEP, tokens, 0);
    let mut drafts = vec![argmax(&run(ProgramId::DRAFT, &tokens[1..], 0))];
    for j in 1..k {
        let last = *drafts.last().unwrap();
        drafts.push(argmax(&run(ProgramId::DRAFT, &[last], n - 2 + j)));
    }
    drafts
}

/// What two sequences served together returned: each one's tokens, and every non-empty proposal paired
/// with the tokens its sequence had committed when it was made.
type Pair = (Vec<Vec<u32>>, Vec<(Vec<u32>, Vec<u32>)>);

/// What a served request returned: its tokens, what the policy did each round, and the drafts every
/// speculative round was proposed against the tokens committed before it.
struct Served {
    tokens: Vec<u32>,
    rounds: Vec<RowSpeculation>,
    proposals: Vec<(Vec<u32>, Vec<u32>)>,
}

fn prepare(
    driver: Driver,
    prompts: &[usize],
    layout: Layout,
    rows: &[NonZeroUsize],
) -> Driver<Serving> {
    driver
        .prepare(&ServingShapes {
            layout,
            rows,
            heads: &[
                Head::GREEDY,
                Head::Sample(SampleSpec::draw(SampleRule::GumbelTopKTopP)),
                Head::Logits,
            ],
            warm: Warm::Prompts(prompts),
            windows: &[],
            adapters: &[],
            chain: None,
        })
        .unwrap_or_else(|e| panic!("{e}"))
}

/// `driver` prepared for one `prompt` on the contiguous layout, the shapes [`serve`] runs.
fn prepare_one(driver: Driver, prompt: &[u32]) -> Driver<Serving> {
    prepare(
        driver,
        &[prompt.len()],
        Layout::Contiguous,
        &[NonZeroUsize::MIN],
    )
}

/// `NEW` tokens after `prompt` through `open`/`step`/`commit`, each round committed whole (a
/// recurrent row's state holds its folded kept prefix). `proposals` pairs each round's proposed
/// drafts with the committed tokens it followed.
fn serve(driver: &mut Driver<Serving>, prompt: &[u32], sampler: &mut Sampler) -> Served {
    let log = log_proposals(driver);
    let compiled = driver.stats().compiles;
    let (seq, _) = {
        let Admission::Opened { seq, reused } = driver
            .open(SeqRequest::new(prompt, NEW, AdapterRef::Base))
            .unwrap()
        else {
            panic!("a fresh driver admits one sequence")
        };
        (seq, reused)
    };
    let mut first = loop {
        let mut rows = [StepRow {
            seq,
            work: RowWork::Prefill { upto: prompt.len() },
            sampler: &mut *sampler,
        }];
        let result = driver.step(&mut rows).unwrap().remove(0);
        if !result.tokens.is_empty() {
            break result;
        }
    };
    let mut committed = prompt.to_vec();
    let mut tokens = Vec::new();
    let mut rounds = Vec::new();
    let mut proposals = Vec::new();
    loop {
        let kept = first.tokens.len().min(NEW - tokens.len());
        driver.commit(seq, kept, sampler).unwrap();
        tokens.extend_from_slice(&first.tokens[..kept]);
        committed.extend_from_slice(&first.tokens[..kept]);
        rounds.push(first.speculation);
        if tokens.len() == NEW {
            break;
        }
        first = decode(driver, seq, sampler);
        for (_, drafts) in log.lock().unwrap().drain(..) {
            proposals.push((committed.clone(), drafts));
        }
    }
    driver.release(seq, Release::Finished).unwrap();
    assert_eq!(
        driver.stats().compiles,
        compiled,
        "serving the prepared set compiles nothing, whatever the rounds propose"
    );
    Served {
        tokens,
        rounds,
        proposals,
    }
}

fn is_partial(round: &RowSpeculation) -> bool {
    matches!(round, RowSpeculation::Verified { proposed, accepted } if accepted < proposed)
}

fn is_full(round: &RowSpeculation) -> bool {
    matches!(round, RowSpeculation::Verified { proposed, accepted } if *proposed > 0 && accepted == proposed)
}

/// SC-001 (POOT-1111): a GLM fixture under `DraftPolicy::Mtp { k: 3 }` proposes drafts that verify
/// through the verify-window round and produce the tokens `DraftPolicy::Off` produces for the same
/// prompt, greedy. With the fixture's random weights the layer's drafts mostly miss, so rounds reject
/// and the restore-and-replay path runs on the recurrent target (`restored > 0`); with the head zeroed
/// both pick token 0 and every round keeps all its drafts (`mtp_all_accepted_equals_off`). Mutation:
/// accept every draft without verification (`verify_row` returns the drafts and the last pick); a
/// rejected draft enters the output and the tokens leave `Off`'s.
fn mtp_greedy_equals_off(choice: BackendChoice) {
    let handle = glm_handle(FixtureHead::Random);
    let Some(off) = glm_driver(choice, &handle, DraftPolicy::Off) else {
        return;
    };
    let mut off = prepare_one(off, &PROMPT);
    let reference = serve(&mut off, &PROMPT, &mut Sampler::greedy());
    let driver = glm_driver(choice, &handle, mtp(K))
        .expect("the device opened a moment ago")
        .with_phase_timing();
    let mut driver = prepare_one(driver, &PROMPT);
    let served = serve(&mut driver, &PROMPT, &mut Sampler::greedy());
    assert_eq!(
        served.tokens, reference.tokens,
        "Mtp differs from Off on {choice:?}; rounds {:?}",
        served.rounds
    );
    assert!(
        served.rounds.iter().any(is_partial),
        "a round rejected a draft on {choice:?}: {:?}",
        served.rounds
    );
    let stats = driver.stats().speculation;
    assert!(
        stats.restored > 0 && stats.replayed >= stats.restored,
        "{stats:?}"
    );
    eprintln!(
        "MTP k={K} on {choice:?}: rounds {} proposed {} accepted {}",
        stats.rounds, stats.proposed, stats.accepted
    );
    let positions = driver.stats().positions;
    assert_eq!(
        positions.proposed.iter().sum::<u64>(),
        stats.proposed,
        "{positions:?}"
    );
    assert_eq!(
        positions.accepted.iter().sum::<u64>(),
        stats.accepted,
        "{positions:?}"
    );
    assert!(
        positions.accepted.windows(2).all(|pair| pair[0] >= pair[1])
            && positions
                .accepted
                .iter()
                .zip(&positions.proposed)
                .all(|(a, p)| a <= p),
        "a draft is kept only after the ones before it: {positions:?}"
    );
    let passes = driver.stats().passes;
    assert_eq!(
        passes.verifies.steps, stats.rounds,
        "one verify window a round, none for the prefill or a draft-less decode: {passes:?}"
    );
    assert!(
        passes.drafts.steps > 0 && !passes.drafts.wall.is_zero() && !passes.verifies.wall.is_zero(),
        "the draft passes and the verify windows were timed: {passes:?}"
    );
}

/// SC-001 (POOT-1111), the other half: with the head zeroed the target and the layer both pick token
/// 0, so every round keeps all its drafts and returns `k + 1` tokens, the live recurrent state is kept
/// (nothing restores or replays), and the tokens are `Off`'s. Mutation: reject the last draft of every
/// verified row (`accept_greedy` stops one short); `restored` is no longer zero and the full-round
/// assertion is red.
fn mtp_all_accepted_equals_off(choice: BackendChoice) {
    let handle = glm_handle(FixtureHead::Zeroed);
    let Some(off) = glm_driver(choice, &handle, DraftPolicy::Off) else {
        return;
    };
    let mut off = prepare_one(off, &PROMPT);
    let reference = serve(&mut off, &PROMPT, &mut Sampler::greedy());
    let driver = glm_driver(choice, &handle, mtp(K)).expect("the device opened a moment ago");
    let mut driver = prepare_one(driver, &PROMPT);
    let served = serve(&mut driver, &PROMPT, &mut Sampler::greedy());
    assert_eq!(
        served.tokens, reference.tokens,
        "Mtp differs from Off on {choice:?}"
    );
    assert!(
        served.rounds.iter().filter(|round| is_full(round)).count() >= 3,
        "rounds kept every draft on {choice:?}: {:?}",
        served.rounds
    );
    assert!(
        !served.rounds.iter().any(is_partial),
        "no round rejected a draft: {:?}",
        served.rounds
    );
    let stats = driver.stats().speculation;
    assert_eq!((stats.restored, stats.replayed), (0, 0), "{stats:?}");
}

/// SC-002 and SC-003 (POOT-1111): at every round the drafts the policy proposes equal the oracle
/// chain evaluated from scratch on the CPU over the tokens committed so far, including the rounds
/// after a partial accept (the layer's cache is back at the committed prefix, and the chain starts
/// from the state the catch-up left). Mutations: skip the catch-up over the accepted positions (feed
/// only the last position); skip the rewrite of the query position the chain wrote speculatively
/// (catch up from `start + 1`); run the catch-up before `roll_back` (the replay chunk rewrites the
/// carry's column 0 the chain recycles). Each moves the second round's drafts and the row is red.
fn mtp_proposals_equal_the_oracle_chain(choice: BackendChoice) {
    let handle = glm_handle(FixtureHead::Random);
    let Some(driver) = glm_driver(choice, &handle, mtp(K)) else {
        return;
    };
    let mut driver = prepare_one(driver, &PROMPT);
    let served = serve(&mut driver, &PROMPT, &mut Sampler::greedy());
    assert!(
        served.rounds.iter().any(is_partial),
        "a round rejected a draft, so a later proposal follows a rollback: {:?}",
        served.rounds
    );
    // A round with no room left for a window proposes nothing, and is the one kind that may be empty.
    let proposed: Vec<_> = served
        .proposals
        .iter()
        .filter(|(_, drafts)| !drafts.is_empty())
        .collect();
    assert!(proposed.len() >= 10, "{:?}", served.proposals);
    for (round, (committed, drafts)) in proposed.into_iter().enumerate() {
        let want = oracle_drafts(&handle, committed, drafts.len());
        assert_eq!(
            drafts,
            &want,
            "round {round} on {choice:?}: the drafts after {} committed tokens {committed:?}",
            committed.len()
        );
    }
}

/// Two sequences of different lengths step together on the paged layout: the first step prefills a
/// seven-token and a four-token prompt side by side, then both decode in one step. Each width runs as
/// its own group, so the carry's columns are the positions the layer catches up over; each sequence's
/// tokens equal what `Off` returns for it and every proposal of every round equals the from-scratch
/// oracle chain over that sequence's own committed tokens. Mutation: run the catch-up of every
/// width at once at the widest one (`mtp_catch_up` selects the rows with `feed.len() <= width`): the
/// narrower prompt is caught up left-padded, reads the carry's columns shifted by the padding, and
/// its first proposal leaves the oracle's.
fn mtp_two_rows_equal_off(choice: BackendChoice) {
    let handle = glm_handle(FixtureHead::Random);
    let layout = Layout::Paged(PoolShape {
        blocks: nz(8),
        max_seqs: nz(2),
    });
    let rows = [nz(1), nz(2)];
    let prepare_pair =
        |driver: Driver| prepare(driver, &[PROMPT.len(), OTHER_PROMPT.len()], layout, &rows);
    let serve_pair = |driver: &mut Driver<Serving>| -> Pair {
        let log = log_proposals(driver);
        let mut seqs = Vec::new();
        for prompt in [&PROMPT[..], &OTHER_PROMPT[..]] {
            let Admission::Opened { seq, .. } = driver
                .open(SeqRequest::new(prompt, NEW, AdapterRef::Base))
                .unwrap()
            else {
                panic!("the pool admits both sequences")
            };
            seqs.push((seq, prompt.len()));
        }
        let mut samplers = [Sampler::greedy(), Sampler::greedy()];
        let mut out: Vec<Vec<u32>> = vec![Vec::new(), Vec::new()];
        let mut prefilled = [false, false];
        let prompts = [&PROMPT[..], &OTHER_PROMPT[..]];
        // Each proposal paired with the tokens its sequence had committed when it was made.
        let mut proposals = Vec::new();
        while out.iter().any(|tokens| tokens.len() < NEW) {
            let held: Vec<Vec<u32>> = (0..2)
                .map(|i| prompts[i].iter().chain(&out[i]).copied().collect())
                .collect();
            let mut live = Vec::new();
            let mut work = Vec::new();
            for (i, sampler) in samplers.iter_mut().enumerate() {
                if out[i].len() >= NEW {
                    continue;
                }
                let (seq, len) = seqs[i];
                let step_work = if prefilled[i] {
                    RowWork::Decode { ahead: &[] }
                } else {
                    RowWork::Prefill { upto: len }
                };
                live.push(i);
                work.push(StepRow {
                    seq,
                    work: step_work,
                    sampler,
                });
            }
            let results = driver.step(&mut work).unwrap();
            drop(work);
            for (seq, drafts) in log.lock().unwrap().drain(..) {
                let i = seqs
                    .iter()
                    .position(|(s, _)| *s == seq)
                    .expect("a stepped row");
                if !drafts.is_empty() {
                    proposals.push((held[i].clone(), drafts));
                }
            }
            for (&i, result) in live.iter().zip(results) {
                if result.tokens.is_empty() {
                    continue;
                }
                prefilled[i] = true;
                let kept = result.tokens.len().min(NEW - out[i].len());
                driver.commit(seqs[i].0, kept, &mut samplers[i]).unwrap();
                out[i].extend_from_slice(&result.tokens[..kept]);
            }
        }
        for (seq, _) in seqs {
            driver.release(seq, Release::Finished).unwrap();
        }
        (out, proposals)
    };
    let Some(off) = glm_driver(choice, &handle, DraftPolicy::Off) else {
        return;
    };
    let mut off = prepare_pair(off);
    let reference = serve_pair(&mut off);
    assert!(reference.1.is_empty(), "Off proposes nothing");
    let driver = glm_driver(choice, &handle, mtp(K)).expect("the device opened a moment ago");
    let mut driver = prepare_pair(driver);
    let (served, proposals) = serve_pair(&mut driver);
    assert_eq!(
        served, reference.0,
        "Mtp rows differ from Off on {choice:?}"
    );
    assert!(driver.stats().speculation.rounds > 0);
    assert!(proposals.len() >= 10, "{}", proposals.len());
    for (round, (committed, drafts)) in proposals.iter().enumerate() {
        assert_eq!(
            drafts,
            &oracle_drafts(&handle, committed, drafts.len()),
            "proposal {round} on {choice:?} after {} committed tokens {committed:?}",
            committed.len()
        );
    }
}

/// A sampled row speculates under Mtp with the point-mass draft and finishes its tokens: the policy
/// verifies it through speculative sampling (`rounds > 0`) rather than declining it. The
/// distribution the output follows is `spec_sampling_tests`'s chi-squared row over the production
/// proposal. Mutation: decline sampled rows for MTP in `speculation_sampler_ok`; no round verifies.
fn mtp_serves_a_sampled_row(choice: BackendChoice) {
    let handle = glm_handle(FixtureHead::Random);
    let Some(driver) = glm_driver(choice, &handle, mtp(K)) else {
        return;
    };
    let mut driver = prepare_one(driver, &PROMPT);
    let served = serve(&mut driver, &PROMPT, &mut Sampler::new(1.0, 0, 1.0, 0x5EED));
    assert_eq!(served.tokens.len(), NEW);
    assert!(
        served
            .tokens
            .iter()
            .all(|&t| (t as usize) < crate::driver::device_tests::VOCAB)
    );
    let stats = driver.stats().speculation;
    assert!(stats.rounds > 0, "{stats:?}");
}

/// The draft passes the driver times are the decode window's: the catch-up over the prompt pieces
/// runs inside the prefill steps, before the first token, so a tally that took them would put work in
/// a bench's per-round breakdown that its first-token-to-last span never held. After the prefill
/// nothing is tallied; the decode step that follows tallies its chain and catch-up passes and one
/// verify window. Mutation: count every catch-up as a decode pass (`run_mtp` called with
/// `decode = true` from `mtp_catch_up`); the prefill assertion is red.
fn mtp_prefill_catch_up_is_not_a_decode_pass(choice: BackendChoice) {
    let handle = glm_handle(FixtureHead::Zeroed);
    let Some(driver) = glm_driver(choice, &handle, mtp(K)) else {
        return;
    };
    let mut driver = prepare_one(driver.with_phase_timing(), &PROMPT);
    let sampler = &mut Sampler::greedy();
    let Admission::Opened { seq, .. } = driver
        .open(SeqRequest::new(&PROMPT, NEW, AdapterRef::Base))
        .unwrap()
    else {
        panic!("a fresh driver admits one sequence")
    };
    let first = loop {
        let mut rows = [StepRow {
            seq,
            work: RowWork::Prefill { upto: PROMPT.len() },
            sampler: &mut *sampler,
        }];
        let result = driver.step(&mut rows).unwrap().remove(0);
        if !result.tokens.is_empty() {
            break result;
        }
    };
    let prefilled = driver.stats().passes;
    assert_eq!(
        prefilled,
        PassTiming::default(),
        "the prefill's catch-up passes are not decode passes on {choice:?}"
    );
    driver.commit(seq, first.tokens.len(), sampler).unwrap();
    decode(&mut driver, seq, sampler);
    let decoded = driver.stats().passes;
    assert!(
        decoded.drafts.steps > 0 && decoded.verifies.steps == 1,
        "a decode round tallies its draft passes and one verify window on {choice:?}: {decoded:?}"
    );
}

per_backend!(
    BackendChoice,
    mtp_prefill_catch_up_is_not_a_decode_pass,
    mtp_prefill_catch_up_is_not_a_decode_pass_wgpu,
    mtp_prefill_catch_up_is_not_a_decode_pass_rocm,
    mtp_prefill_catch_up_is_not_a_decode_pass_ptx
);
per_backend!(
    BackendChoice,
    mtp_greedy_equals_off,
    mtp_greedy_equals_off_wgpu,
    mtp_greedy_equals_off_rocm,
    mtp_greedy_equals_off_ptx
);
per_backend!(
    BackendChoice,
    mtp_all_accepted_equals_off,
    mtp_all_accepted_equals_off_wgpu,
    mtp_all_accepted_equals_off_rocm,
    mtp_all_accepted_equals_off_ptx
);
per_backend!(
    BackendChoice,
    mtp_proposals_equal_the_oracle_chain,
    mtp_proposals_equal_the_oracle_chain_wgpu,
    mtp_proposals_equal_the_oracle_chain_rocm,
    mtp_proposals_equal_the_oracle_chain_ptx
);
per_backend!(
    BackendChoice,
    mtp_two_rows_equal_off,
    mtp_two_rows_equal_off_wgpu,
    mtp_two_rows_equal_off_rocm,
    mtp_two_rows_equal_off_ptx
);
per_backend!(
    BackendChoice,
    mtp_serves_a_sampled_row,
    mtp_serves_a_sampled_row_wgpu,
    mtp_serves_a_sampled_row_rocm,
    mtp_serves_a_sampled_row_ptx
);
