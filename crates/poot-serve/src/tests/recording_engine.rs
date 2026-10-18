//! The recording fake behind the scheduling rows: a [`ServeEngine`] that records every call, picks the
//! tokens a test scripts and models the driver's prefix reuse, so the one loop is exercised without a
//! model or a device.

use std::collections::BTreeSet;

use poot_llm::driver::{Encoded, PrefixOutcome, PrefixSkip, PromptEmbeds, Release, RowSpeculation};

use super::*;
use crate::types::{
    AdapterInfo, AdminError, AdminReply, AdminRequest, ChoiceEvents, ChoiceReply, EncodeRow,
    PromptRows,
};

/// The block the fake's prefix cache registers and reuses in: whole multiples of it.
const BLOCK: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FakeSeq(pub(crate) u64);

/// One call the loop made on the engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    Open {
        prompt_len: usize,
        max_new: usize,
        reused: usize,
        /// The sequence opened with embedding rows.
        embeds: bool,
    },
    Step(Vec<(u64, Work)>),
    Commit {
        seq: u64,
        kept: usize,
    },
    Release {
        seq: u64,
        mode: Release,
    },
    /// One `Encode` step over this many rows.
    Encode {
        rows: usize,
    },
    Admin(AdminRequest),
}

pub(crate) type Log = Arc<Mutex<Vec<Event>>>;

/// What a test scripts into a step: its 1-based index and rows; an error fails the step.
type StepHook = Box<dyn FnMut(usize, &[(u64, Work)]) -> Result<(), EngineError> + Send>;

struct Open {
    prompt: Vec<u32>,
    done: usize,
    committed: usize,
    pending: usize,
    /// Opened with embedding rows: no prefix is matched and none registered.
    embeds: bool,
}

/// What a test scripts into an `Encode` step: the rows it gets and the answer.
type EncodeHook = Box<dyn FnMut(&[EncodeRow]) -> Result<Encoded, EngineError> + Send>;

/// The fake's default `Encode` answer: a two-value hidden state per token, `[id, 1]`, so the pooled
/// vectors of different texts differ.
fn default_encoded(rows: &[EncodeRow]) -> Encoded {
    Encoded {
        width: 2,
        rows: rows
            .iter()
            .map(|row| row.ids.iter().flat_map(|&id| [id as f32, 1.0]).collect())
            .collect(),
    }
}

/// What a test does with each row's sampler inside a step: the fake's stand-in for the driver's pick.
type SamplerProbe = Box<dyn FnMut(u64, &mut Sampler) + Send>;

/// A refusal of the request itself, as the fake's hooks return it.
pub(crate) fn refused(reason: &str) -> EngineError {
    EngineError::Refused {
        class: poot_llm::driver::error::FailureClass::Request,
        reason: reason.to_string(),
    }
}

/// A scripted engine. `picks` gives the tokens a row picks given how many tokens it has committed;
/// `on_step` runs inside every step and may panic, fail the step or flip a cancel flag.
pub(crate) struct RecordingEngine {
    pub(crate) rows: usize,
    pub(crate) max_seq: usize,
    pub(crate) widest: usize,
    /// Sequence slots, which can be fewer than `rows`.
    pub(crate) slots: usize,
    pub(crate) eos: BTreeSet<u32>,
    /// A recurrent-state model: `release` registers no prefix.
    pub(crate) recurrent: bool,
    pub(crate) log: Log,
    pub(crate) picks: Box<dyn FnMut(u64, usize) -> Vec<u32> + Send>,
    pub(crate) on_step: StepHook,
    /// Set by a faulting step; a real driver's `release` frees the sequence's state and so would hide it.
    pub(crate) poisoned: bool,
    /// What every decode row reports of the draft policy.
    pub(crate) speculation: Option<RowSpeculation>,
    /// Every `release` fails with this.
    pub(crate) release_error: Option<String>,
    /// Answers each `Encode` step.
    pub(crate) on_encode: EncodeHook,
    /// Sees each row's sampler once per step, as the driver's pick would.
    pub(crate) probe: SamplerProbe,
    /// The adapters `Admin` has loaded, in index order.
    pub(crate) adapters: Vec<String>,
    /// Every `Admin` call fails with this.
    pub(crate) admin_error: Option<AdminError>,
    /// The executor timing the engine reports; `None` is an engine with no executor.
    pub(crate) timing: Option<poot_profile::TimingSnapshot>,
    open: HashMap<u64, Open>,
    registered: Vec<Vec<u32>>,
    next: u64,
    steps: usize,
}

impl RecordingEngine {
    pub(crate) fn new(rows: usize) -> Self {
        Self {
            rows,
            max_seq: 64,
            widest: 8,
            slots: rows,
            eos: BTreeSet::from([99]),
            recurrent: false,
            log: Log::default(),
            picks: Box::new(|_, _| vec![7]),
            on_step: Box::new(|_, _| Ok(())),
            poisoned: false,
            speculation: None,
            release_error: None,
            on_encode: Box::new(|rows| Ok(default_encoded(rows))),
            probe: Box::new(|_, _| {}),
            adapters: Vec::new(),
            admin_error: None,
            timing: None,
            open: HashMap::new(),
            registered: Vec::new(),
            next: 0,
            steps: 0,
        }
    }

    pub(crate) fn events(&self) -> Vec<Event> {
        self.log.lock().unwrap().clone()
    }

    pub(crate) fn picking(
        mut self,
        picks: impl FnMut(u64, usize) -> Vec<u32> + Send + 'static,
    ) -> Self {
        self.picks = Box::new(picks);
        self
    }

    pub(crate) fn probing(mut self, probe: impl FnMut(u64, &mut Sampler) + Send + 'static) -> Self {
        self.probe = Box::new(probe);
        self
    }

    pub(crate) fn on_step(
        mut self,
        hook: impl FnMut(usize, &[(u64, Work)]) -> Result<(), EngineError> + Send + 'static,
    ) -> Self {
        self.on_step = Box::new(hook);
        self
    }

    fn record(&self, event: Event) {
        self.log.lock().unwrap().push(event);
    }
}

/// How many sequences are open at once, by replaying the log: the most any prefix of it reaches.
pub(crate) fn peak_open(events: &[Event]) -> usize {
    let (mut open, mut peak) = (0usize, 0usize);
    for event in events {
        match event {
            Event::Open { .. } => open += 1,
            Event::Release { .. } => open -= 1,
            _ => {}
        }
        peak = peak.max(open);
    }
    peak
}

impl ServeEngine for RecordingEngine {
    type Seq = FakeSeq;

    fn rows(&self) -> usize {
        self.rows
    }

    fn max_seq(&self) -> usize {
        self.max_seq
    }

    fn widest_piece(&self) -> usize {
        self.widest
    }

    fn eos(&self) -> &BTreeSet<u32> {
        &self.eos
    }

    /// One `<id>` per token.
    fn decode(&self, tokens: &[u32]) -> anyhow::Result<String> {
        Ok(tokens.iter().map(|token| format!("<{token}>")).collect())
    }

    /// `<id>` for the newest token, so a stop string names tokens.
    fn piece(&self, tokens: &[u32], _gen_start: usize) -> anyhow::Result<String> {
        Ok(format!("<{}>", tokens.last().expect("a token")))
    }

    fn open(&mut self, request: OpenRequest<'_>) -> Result<Opening<FakeSeq>, EngineError> {
        if self.open.len() >= self.slots {
            return Ok(Opening::NoRoom { needed_blocks: 0 });
        }
        let prompt = request.prompt.to_vec();
        let embeds = request.rows.is_some();
        let common = self
            .registered
            .iter()
            .map(|cached| {
                cached
                    .iter()
                    .zip(&prompt)
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .max()
            .unwrap_or(0);
        // A prompt with embedding rows has no tokens-only identity to match a cached prefix on.
        let reused = if embeds {
            0
        } else {
            common.min(prompt.len() - 1) / BLOCK * BLOCK
        };
        self.next += 1;
        self.record(Event::Open {
            prompt_len: prompt.len(),
            max_new: request.max_new,
            reused,
            embeds,
        });
        self.open.insert(
            self.next,
            Open {
                prompt,
                done: reused,
                committed: 0,
                pending: 0,
                embeds,
            },
        );
        Ok(Opening::Opened {
            seq: FakeSeq(self.next),
            reused,
        })
    }

    fn step(
        &mut self,
        rows: &mut [Row<'_, FakeSeq>],
    ) -> Result<Vec<RowResult<FakeSeq>>, EngineError> {
        let call: Vec<(u64, Work)> = rows.iter().map(|row| (row.seq.0, row.work)).collect();
        self.record(Event::Step(call.clone()));
        self.steps += 1;
        if let Err(error) = (self.on_step)(self.steps, &call) {
            if matches!(error, EngineError::Fault(_)) {
                self.poisoned = true;
            }
            return Err(error);
        }
        let mut results = Vec::new();
        for row in rows.iter_mut() {
            (self.probe)(row.seq.0, row.sampler);
            let state = self.open.get_mut(&row.seq.0).expect("an open sequence");
            let (consumed, picks) = match row.work {
                Work::Prefill { upto } => {
                    let consumed = (upto - state.done).min(self.widest);
                    state.done += consumed;
                    (consumed, state.done == state.prompt.len())
                }
                Work::Decode => (0, true),
            };
            let tokens = if picks {
                (self.picks)(row.seq.0, state.committed)
            } else {
                Vec::new()
            };
            state.pending = tokens.len();
            results.push(RowResult {
                seq: row.seq,
                consumed,
                tokens,
                logprobs: Vec::new(),
                speculation: match (row.work, self.speculation) {
                    (Work::Decode, Some(speculation)) => speculation,
                    _ => RowSpeculation::NotApplicable,
                },
            });
        }
        Ok(results)
    }

    fn commit(
        &mut self,
        seq: FakeSeq,
        kept: usize,
        _sampler: &mut Sampler,
    ) -> Result<(), EngineError> {
        self.record(Event::Commit { seq: seq.0, kept });
        if let Some(state) = self.open.get_mut(&seq.0) {
            state.committed += kept;
            state.pending = 0;
        }
        Ok(())
    }

    fn release(&mut self, seq: FakeSeq, mode: Release) -> Result<PrefixOutcome, EngineError> {
        self.record(Event::Release { seq: seq.0, mode });
        if let Some(reason) = &self.release_error {
            return Err(EngineError::Fault(reason.clone()));
        }
        // A release frees the sequence's state, which clears a poisoned device in this fake.
        self.poisoned = false;
        let state = self.open.remove(&seq.0).expect("an open sequence");
        Ok(if state.embeds {
            PrefixOutcome::Skipped(PrefixSkip::NoBlockTable)
        } else if self.recurrent {
            PrefixOutcome::Skipped(PrefixSkip::RecurrentState)
        } else if mode == Release::Preempted {
            PrefixOutcome::Skipped(PrefixSkip::Preempted)
        } else {
            let filled = state.done.min(state.prompt.len());
            self.registered.push(state.prompt[..filled].to_vec());
            PrefixOutcome::Registered {
                tokens: filled / BLOCK * BLOCK,
            }
        })
    }

    fn encode(&mut self, rows: &[EncodeRow]) -> Result<Encoded, EngineError> {
        assert!(
            self.open.is_empty(),
            "the loop runs an encode step with no sequence open"
        );
        self.record(Event::Encode { rows: rows.len() });
        (self.on_encode)(rows)
    }

    fn admin(&mut self, request: AdminRequest) -> Result<AdminReply, AdminError> {
        assert!(
            self.open.is_empty(),
            "the loop runs an administration job with no sequence open"
        );
        self.record(Event::Admin(request.clone()));
        if let Some(error) = &self.admin_error {
            return Err(error.clone());
        }
        let info = |position: usize, name: &String| AdapterInfo {
            name: name.clone(),
            index: position as u32 + 1,
        };
        match request {
            AdminRequest::List => Ok(AdminReply::Adapters(
                self.adapters
                    .iter()
                    .enumerate()
                    .map(|(position, name)| info(position, name))
                    .collect(),
            )),
            AdminRequest::Load { name, .. } => {
                self.adapters.push(name.clone());
                Ok(AdminReply::Loaded(info(self.adapters.len() - 1, &name)))
            }
            AdminRequest::Unload { name } => {
                let position = self
                    .adapters
                    .iter()
                    .position(|loaded| *loaded == name)
                    .ok_or_else(|| AdminError::NotFound(name.clone()))?;
                self.adapters.remove(position);
                Ok(AdminReply::Unloaded(info(position, &name)))
            }
        }
    }

    fn kv_blocks(&self) -> Option<(usize, usize)> {
        Some((8 - self.open.len().min(8), 8))
    }

    fn resident_bytes(&self) -> u64 {
        1024
    }

    fn compiles(&self) -> u64 {
        0
    }

    fn timing(&self) -> Option<poot_profile::TimingSnapshot> {
        self.timing.clone()
    }
}

/// A job for `prompt`, and what the test keeps of it: the reply receiver and the cancel flag.
pub(crate) struct Submitted {
    pub(crate) replies: ReplyStream,
    pub(crate) cancel: Arc<AtomicBool>,
}

/// The events of a job's one choice, untagged: what a test that reads a single job's replies sees.
pub(crate) struct ReplyStream(ChoiceEvents);

impl ReplyStream {
    pub(crate) fn recv(&self) -> Result<GenEvent, mpsc::RecvError> {
        self.0.recv().map(|(_, event)| event)
    }

    pub(crate) fn try_recv(&self) -> Result<GenEvent, mpsc::TryRecvError> {
        self.0.try_recv().map(|(_, event)| event)
    }

    pub(crate) fn recv_timeout(&self, wait: Duration) -> Result<GenEvent, mpsc::RecvTimeoutError> {
        self.0.recv_timeout(wait).map(|(_, event)| event)
    }

    pub(crate) fn try_iter(&self) -> impl Iterator<Item = GenEvent> + '_ {
        self.0.try_iter().map(|(_, event)| event)
    }
}

pub(crate) struct JobSpec {
    pub(crate) prompt: String,
    pub(crate) max_new: usize,
    pub(crate) stop: Vec<String>,
    pub(crate) stream: bool,
    /// Embedding rows standing in for the first prompt token: a multimodal prompt.
    pub(crate) embeds: bool,
    /// The job has waited past its maximum queue wait before the engine sees it.
    pub(crate) overdue: bool,
}

/// One token per character, its code point: the prompts the tests write as strings.
pub(crate) fn tokens_of(prompt: &str) -> Vec<u32> {
    prompt.chars().map(u32::from).collect()
}

impl JobSpec {
    pub(crate) fn new(prompt: &str, max_new: usize) -> Self {
        Self {
            prompt: prompt.to_string(),
            max_new,
            stop: Vec::new(),
            stream: false,
            embeds: false,
            overdue: false,
        }
    }

    /// A multimodal prompt: one embedding row stands in for the first token.
    pub(crate) fn with_embeds(mut self) -> Self {
        self.embeds = true;
        self
    }

    pub(crate) fn stop(mut self, stop: &str) -> Self {
        self.stop.push(stop.to_string());
        self
    }

    pub(crate) fn streaming(mut self) -> Self {
        self.stream = true;
        self
    }

    pub(crate) fn overdue(mut self) -> Self {
        self.overdue = true;
        self
    }

    pub(crate) fn build(self) -> (Job, Submitted) {
        let (mut replies, events) = ChoiceReply::channel(1);
        let reply = replies.remove(0);
        let cancel = Arc::new(AtomicBool::new(false));
        let tokens = tokens_of(&self.prompt);
        let prompt = if self.embeds {
            PromptInput::Multimodal {
                tokens,
                rows: PromptRows {
                    embeds: PromptEmbeds::new(1, vec![0], vec![1.0]).expect("one row of one value"),
                    mrope: None,
                },
            }
        } else {
            PromptInput::Tokens(tokens)
        };
        let job = Job {
            prompt,
            max_new: self.max_new,
            sampler: Sampler::greedy(),
            stop: self.stop,
            stream: self.stream,
            reply,
            submitted: Instant::now(),
            start_by: if self.overdue {
                Instant::now()
            } else {
                Instant::now() + Duration::from_secs(3600)
            },
            lora_adapter: AdapterRef::Base,
            cancel: Arc::clone(&cancel),
        };
        (
            job,
            Submitted {
                replies: ReplyStream(events),
                cancel,
            },
        )
    }
}

/// Queue `specs`, close the queue and run the loop on this thread to its end.
pub(crate) fn serve(
    engine: &mut RecordingEngine,
    specs: Vec<JobSpec>,
) -> (Exit, Arc<Metrics>, Vec<Submitted>) {
    let (tx, rx) = mpsc::channel();
    let submitted = specs
        .into_iter()
        .map(|spec| {
            let (job, submitted) = spec.build();
            tx.send(EngineMsg::Generate(Box::new(job)))
                .expect("queue the job");
            submitted
        })
        .collect();
    drop(tx);
    let metrics = Arc::new(test_metrics("recording-engine"));
    let exit = run_engine(engine, &rx, &metrics);
    (exit, metrics, submitted)
}

/// The terminal event of a request whose loop has ended, with the streamed pieces before it.
pub(crate) fn outcome(submitted: &Submitted) -> (Vec<String>, GenEvent) {
    let mut pieces = Vec::new();
    for event in submitted.replies.try_iter() {
        match event {
            GenEvent::Token(piece, _) => pieces.push(piece),
            terminal => return (pieces, terminal),
        }
    }
    panic!("the request ended without a terminal event");
}
