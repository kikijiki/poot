//! The recording fake behind the scheduling rows: a [`ServeEngine`] that records every call, picks the
//! tokens a test scripts and models the driver's prefix reuse, so the one loop is exercised without a
//! model or a device.

use std::collections::BTreeSet;

use poot_llm::driver::{PrefixOutcome, PrefixSkip, Release};

use super::*;

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
}

pub(crate) type Log = Arc<Mutex<Vec<Event>>>;

/// What a test scripts into a step: its 1-based index and rows; an error fails the step.
type StepHook = Box<dyn FnMut(usize, &[(u64, Work)]) -> Result<(), EngineError> + Send>;

struct Open {
    prompt: Vec<u32>,
    done: usize,
    committed: usize,
    pending: usize,
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
    /// Every `release` fails with this.
    pub(crate) release_error: Option<String>,
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
            release_error: None,
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

    /// One token per character, its code point.
    fn encode(&self, prompt: &str) -> anyhow::Result<Vec<u32>> {
        Ok(prompt.chars().map(u32::from).collect())
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
        let reused = common.min(prompt.len() - 1) / BLOCK * BLOCK;
        self.next += 1;
        self.record(Event::Open {
            prompt_len: prompt.len(),
            max_new: request.max_new,
            reused,
        });
        self.open.insert(
            self.next,
            Open {
                prompt,
                done: reused,
                committed: 0,
                pending: 0,
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
        for row in rows.iter() {
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
        Ok(if self.recurrent {
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

    fn kv_blocks(&self) -> Option<(usize, usize)> {
        Some((8 - self.open.len().min(8), 8))
    }

    fn resident_bytes(&self) -> u64 {
        1024
    }

    fn compiles(&self) -> u64 {
        0
    }
}

/// A job for `prompt`, and what the test keeps of it: the reply receiver and the cancel flag.
pub(crate) struct Submitted {
    pub(crate) replies: Receiver<GenEvent>,
    pub(crate) cancel: Arc<AtomicBool>,
}

pub(crate) struct JobSpec {
    pub(crate) prompt: String,
    pub(crate) max_new: usize,
    pub(crate) stop: Vec<String>,
    pub(crate) stream: bool,
}

impl JobSpec {
    pub(crate) fn new(prompt: &str, max_new: usize) -> Self {
        Self {
            prompt: prompt.to_string(),
            max_new,
            stop: Vec::new(),
            stream: false,
        }
    }

    pub(crate) fn stop(mut self, stop: &str) -> Self {
        self.stop.push(stop.to_string());
        self
    }

    pub(crate) fn streaming(mut self) -> Self {
        self.stream = true;
        self
    }

    pub(crate) fn build(self) -> (Job, Submitted) {
        let (reply, replies) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let job = Job {
            prompt: self.prompt,
            max_new: self.max_new,
            sampler: Sampler::greedy(),
            stop: self.stop,
            stream: self.stream,
            reply,
            submitted: Instant::now(),
            lora_adapter: AdapterRef::Base,
            cancel: Arc::clone(&cancel),
        };
        (job, Submitted { replies, cancel })
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
            tx.send(job).expect("queue the job");
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
