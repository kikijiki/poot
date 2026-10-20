//! The one serving loop: every iteration drains the job queue, releases cancelled requests, opens as
//! many waiting requests as the engine has rows and room for, steps the rows that have work and commits
//! what the step picked. It runs over [`ServeEngine`], the driver's serving surface, so the same loop
//! serves every backend and family the driver prepares, and a recording fake stands in for the driver in
//! the scheduling tests.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::num::{NonZeroU64, NonZeroUsize};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError};
use std::time::{Duration, Instant};

use poot_llm::driver::error::{DriverError, FailureClass};
use poot_llm::driver::{
    AdapterRef, AdapterSet, AdapterSource, AdapterSpec, Admission, Driver, EncodeRequest, Encoded,
    GenerationId, Head, Layout, LoraIdx, ModelHandle, PoolShape, PrefixOutcome, Release,
    RowSpeculation, RowWork, SeqId, SeqRequest, Serving, ServingShapes, StepRow, Warm,
};
use poot_llm::{Sampler, TokenLogprob};
use poot_load::lora::LoraAdapter;
use poot_models::io::RequestInputs;
use poot_profile::TimingSnapshot;

use crate::batch::commit::{
    SlotFate, commit_tokens, record_prefix_admission, record_prefix_outcome,
};
use crate::batch::schedule::schedule_prefill_admissions;
use crate::batch::slot::{Slot, reject_job};
use crate::engine_plan::{Death, EngineHealth};
use crate::metrics::Metrics;
use crate::types::{
    AccountedGenerationFailure, AdapterInfo, AdminError, AdminJob, AdminReply, AdminRequest,
    Completion, EncodeJob, EncodeReply, EncodeRow, EngineMsg, EngineView, FinishReason, GenEvent,
    Job, deliver_terminal,
};

/// How long an idle loop waits for a job before it looks again.
const IDLE_POLL: Duration = Duration::from_millis(10);
/// The most encode and administration jobs one loop iteration runs before the step. Exclusive work runs
/// between steps, so a generation request waits at most this many jobs, plus the step in flight, however
/// many are queued; the rest run on the iterations after, each followed by a step.
pub(crate) const EXCLUSIVE_JOBS_PER_ITERATION: usize = 1;
/// The most often the executor's timing summary is published while the loop is stepping.
const TIMING_PUBLISH_INTERVAL: Duration = Duration::from_millis(250);

/// Why an engine call failed, as the loop must treat it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum EngineError {
    /// Refused before anything was applied. The engine is as it was, so only the requests the call names
    /// fail, each answered by the refusal's class.
    Refused { class: FailureClass, reason: String },
    /// The call may have left a step half applied, or the device unusable: the engine is not served from
    /// again.
    Fault(String),
}

impl From<DriverError> for EngineError {
    fn from(error: DriverError) -> Self {
        let reason = error.to_string();
        match error.class() {
            FailureClass::Fault => EngineError::Fault(reason),
            class => EngineError::Refused { class, reason },
        }
    }
}

/// What a sequence is asked to do in one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Work {
    /// Feed prompt tokens up to this position.
    Prefill { upto: usize },
    /// Feed the last committed token and pick the next.
    Decode,
}

/// One row of a step: a sequence, its work and the sampler its pick goes through.
pub(crate) struct Row<'a, S> {
    pub(crate) seq: S,
    pub(crate) work: Work,
    pub(crate) sampler: &'a mut Sampler,
}

/// What a row produced: picks the caller has not committed yet.
#[derive(Debug)]
pub(crate) struct RowResult<S> {
    pub(crate) seq: S,
    /// Prompt tokens a prefill row consumed; zero for a decode row.
    pub(crate) consumed: usize,
    /// The picks: empty for a prefill row that has not reached the end of its prompt, else one or more.
    pub(crate) tokens: Vec<u32>,
    /// One record per token when the row's sampler records logprobs.
    pub(crate) logprobs: Vec<TokenLogprob>,
    /// What the driver's draft policy did for this row: the drafts it verified and kept, or why the row
    /// decoded one token. The loop counts it as reported.
    pub(crate) speculation: RowSpeculation,
}

/// The outcome of opening a sequence.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Opening<S> {
    /// `reused` leading prompt tokens are already resident.
    Opened { seq: S, reused: usize },
    /// The pool cannot hold the request now (`needed_blocks` more blocks), or every sequence slot is taken
    /// (`needed_blocks` is 0).
    NoRoom { needed_blocks: usize },
}

/// A request to open a sequence.
pub(crate) struct OpenRequest<'a> {
    pub(crate) prompt: &'a [u32],
    /// The most tokens that will be committed after the prompt: their KV blocks are reserved at open.
    pub(crate) max_new: usize,
    pub(crate) adapter: AdapterRef,
    /// The values a front end produced for the inputs the model declares (embedding rows standing in
    /// for prompt tokens, sectioned positions): the sequence matches and registers no prefix.
    pub(crate) inputs: &'a RequestInputs,
}

/// The driver's serving surface as the loop uses it, plus the text services a request needs.
pub(crate) trait ServeEngine {
    /// A sequence the engine has opened.
    type Seq: Copy + Eq + std::fmt::Debug;

    /// The most rows one step runs: the prepared capacity.
    fn rows(&self) -> usize;
    /// The positions a sequence may hold, prompt and generated tokens together.
    fn max_seq(&self) -> usize;
    /// The widest prefill piece one step feeds a row.
    fn widest_piece(&self) -> usize;
    /// The model's end-of-sequence tokens.
    fn eos(&self) -> &BTreeSet<u32>;
    /// The text of `tokens`.
    fn decode(&self, tokens: &[u32]) -> anyhow::Result<String>;
    /// The text the last token of `tokens` adds, `gen_start` being the first generated position.
    fn piece(&self, tokens: &[u32], gen_start: usize) -> anyhow::Result<String>;
    fn open(&mut self, request: OpenRequest<'_>) -> Result<Opening<Self::Seq>, EngineError>;
    fn step(
        &mut self,
        rows: &mut [Row<'_, Self::Seq>],
    ) -> Result<Vec<RowResult<Self::Seq>>, EngineError>;
    /// Keep the first `kept` tokens of `seq`'s last result; the sampler sees those and no others.
    fn commit(
        &mut self,
        seq: Self::Seq,
        kept: usize,
        sampler: &mut Sampler,
    ) -> Result<(), EngineError>;
    fn release(&mut self, seq: Self::Seq, mode: Release) -> Result<PrefixOutcome, EngineError>;
    /// One `Encode` step over token rows. The loop calls it with no sequence open.
    fn encode(&mut self, rows: &[EncodeRow]) -> Result<Encoded, EngineError>;
    /// One LoRA administration operation. The loop calls it between steps, sequences open or not.
    fn admin(&mut self, request: AdminRequest) -> Result<AdminReply, AdminError>;
    /// `(free, total)` blocks of the paged KV pool.
    fn kv_blocks(&self) -> Option<(usize, usize)>;
    /// Device bytes the engine holds: its weights, KV pool and workspace.
    fn resident_bytes(&self) -> u64;
    /// Programs compiled so far: constant across steps once the prepared set is warm.
    fn compiles(&self) -> u64;
    /// The executor's timing snapshot, for an engine that runs on an executor. `None` publishes nothing.
    fn timing(&self) -> Option<TimingSnapshot> {
        None
    }

    /// Used over total blocks of the KV pool, 0 without one.
    fn kv_used_ratio(&self) -> f64 {
        match self.kv_blocks() {
            Some((free, total)) if total > 0 => (total - free.min(total)) as f64 / total as f64,
            _ => 0.0,
        }
    }
}

/// The adapters the engine has loaded and the generation that serves them together.
#[derive(Default)]
struct AdapterBook {
    /// Loaded adapters in index order: a request's index is its position here plus one.
    loaded: Vec<(String, LoraAdapter)>,
    /// The generation holding every loaded adapter, `None` when none is loaded.
    generation: Option<GenerationId>,
    /// Generations a newer one replaced that running requests still hold: dropped once they let go.
    retired: Vec<GenerationId>,
    /// The last source id issued: a reloaded set is never the one an old entry was built for.
    issued: u64,
}

impl AdapterBook {
    fn refs(&self) -> BTreeMap<String, AdapterRef> {
        let Some(generation) = self.generation else {
            return BTreeMap::new();
        };
        self.loaded
            .iter()
            .enumerate()
            .map(|(position, (name, _))| {
                let index = LoraIdx::new(position as u32 + 1).expect("one-based");
                (name.clone(), AdapterRef::Adapter { generation, index })
            })
            .collect()
    }

    fn infos(&self) -> Vec<AdapterInfo> {
        self.loaded
            .iter()
            .enumerate()
            .map(|(position, (name, _))| AdapterInfo {
                name: name.clone(),
                index: position as u32 + 1,
            })
            .collect()
    }
}

/// A prepared [`Driver`] and the loaded model it serves.
pub(crate) struct DriverEngine {
    driver: Driver<Serving>,
    handle: Arc<ModelHandle>,
    view: EngineView,
    adapters: AdapterBook,
}

impl DriverEngine {
    pub(crate) fn new(driver: Driver<Serving>, handle: Arc<ModelHandle>, view: EngineView) -> Self {
        Self {
            driver,
            handle,
            view,
            adapters: AdapterBook::default(),
        }
    }

    fn caps(&self) -> &poot_llm::driver::DriverCaps {
        self.driver
            .caps()
            .expect("an engine is built over a prepared driver")
    }

    /// Publish what the driver serves now for the connection threads to read.
    fn publish(&self) {
        self.view.publish(self.caps().clone(), self.adapters.refs());
    }

    /// The driver's admitted shapes again, naming `generation`: the layout, row counts and heads the
    /// caps report, so entries for a new adapter set exist beside the ones already prepared.
    fn prepare_adapters(&mut self, generation: GenerationId) -> Result<(), DriverError> {
        let caps = self.caps().clone();
        let layout = match (caps.kv, NonZeroUsize::new(caps.kv_blocks)) {
            (poot_models::model::KvLayout::Paged { .. }, Some(blocks)) => {
                Layout::Paged(PoolShape {
                    blocks,
                    max_seqs: caps.rows,
                })
            }
            _ => Layout::Contiguous,
        };
        let mut heads: Vec<Head> = caps
            .heads
            .suffix
            .iter()
            .map(|&rule| Head::Sample(rule))
            .collect();
        if caps.heads.with_logits {
            heads.push(Head::Logits);
        }
        self.driver.extend(&ServingShapes {
            layout,
            rows: &caps.row_counts,
            heads: &heads,
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[generation],
            chain: None,
        })?;
        Ok(())
    }

    /// Drop the retired generations no running request holds any more.
    fn reap_retired(&mut self) {
        let retired = std::mem::take(&mut self.adapters.retired);
        for generation in retired {
            if self.driver.unload_adapter_set(generation).is_err() {
                self.adapters.retired.push(generation);
            }
        }
    }

    /// Replace the loaded set with `loaded`: capture it as one new generation, prepare its entries and
    /// publish it, then drop the generation it replaces (or retire it while requests hold it).
    fn install(&mut self, loaded: Vec<(String, LoraAdapter)>) -> Result<(), AdminError> {
        let refuse = |error: DriverError| AdminError::Refused {
            class: error.class(),
            reason: error.to_string(),
        };
        let previous = self.adapters.generation;
        let new = if loaded.is_empty() {
            None
        } else {
            self.adapters.issued += 1;
            let set = AdapterSet {
                adapters: loaded
                    .iter()
                    .map(|(name, adapter)| AdapterSpec {
                        name: name.clone(),
                        shape: vec![adapter.config.r],
                    })
                    .collect(),
                source: AdapterSource(
                    NonZeroU64::new(self.adapters.issued).expect("the counter starts at one"),
                ),
            };
            let stacks: Vec<LoraAdapter> = loaded.iter().map(|(_, a)| a.clone()).collect();
            let generation = self
                .driver
                .capture_adapter_set(set, &stacks)
                .map_err(refuse)?;
            if let Err(error) = self.prepare_adapters(generation) {
                // Nothing serves the generation: drop it so a refused load leaves the engine as it was.
                let _ = self.driver.unload_adapter_set(generation);
                return Err(refuse(error));
            }
            Some(generation)
        };
        if let Some(old) = previous
            && let Err(error) = self.driver.unload_adapter_set(old)
        {
            if new.is_none() {
                // Nothing replaces it and a running request holds it: the unload did not happen.
                return Err(AdminError::Busy(error.to_string()));
            }
            self.adapters.retired.push(old);
        }
        self.adapters.loaded = loaded;
        self.adapters.generation = new;
        self.publish();
        Ok(())
    }
}

impl ServeEngine for DriverEngine {
    type Seq = SeqId;

    fn rows(&self) -> usize {
        self.caps().rows.get()
    }

    fn max_seq(&self) -> usize {
        self.caps().max_seq.get()
    }

    fn widest_piece(&self) -> usize {
        self.caps()
            .widths
            .iter()
            .map(|w| w.get())
            .max()
            .unwrap_or(1)
    }

    fn eos(&self) -> &BTreeSet<u32> {
        &self.handle.config().eos
    }

    fn decode(&self, tokens: &[u32]) -> anyhow::Result<String> {
        Ok(self.handle.text().decode(tokens)?)
    }

    fn piece(&self, tokens: &[u32], gen_start: usize) -> anyhow::Result<String> {
        Ok(self.handle.text().stream_piece(tokens, gen_start)?)
    }

    fn open(&mut self, request: OpenRequest<'_>) -> Result<Opening<SeqId>, EngineError> {
        let OpenRequest {
            prompt,
            max_new,
            adapter,
            inputs,
        } = request;
        let admission = self
            .driver
            .open(SeqRequest::new(prompt, max_new, adapter).with_inputs(inputs.clone()))?;
        Ok(match admission {
            Admission::Opened { seq, reused } => Opening::Opened { seq, reused },
            Admission::NoRoom { needed_blocks } => Opening::NoRoom { needed_blocks },
        })
    }

    fn step(&mut self, rows: &mut [Row<'_, SeqId>]) -> Result<Vec<RowResult<SeqId>>, EngineError> {
        let mut driver_rows: Vec<StepRow<'_>> = rows
            .iter_mut()
            .map(|row| StepRow {
                seq: row.seq,
                work: match row.work {
                    Work::Prefill { upto } => RowWork::Prefill { upto },
                    Work::Decode => RowWork::Decode { ahead: &[] },
                },
                sampler: &mut *row.sampler,
            })
            .collect();
        Ok(self
            .driver
            .step(&mut driver_rows)?
            .into_iter()
            .map(|result| RowResult {
                seq: result.seq,
                consumed: result.consumed,
                tokens: result.tokens,
                logprobs: result.logprobs,
                speculation: result.speculation,
            })
            .collect())
    }

    fn commit(
        &mut self,
        seq: SeqId,
        kept: usize,
        sampler: &mut Sampler,
    ) -> Result<(), EngineError> {
        Ok(self.driver.commit(seq, kept, sampler)?)
    }

    fn release(&mut self, seq: SeqId, mode: Release) -> Result<PrefixOutcome, EngineError> {
        Ok(self.driver.release(seq, mode)?)
    }

    fn encode(&mut self, rows: &[EncodeRow]) -> Result<Encoded, EngineError> {
        let inputs: Vec<RequestInputs> = rows
            .iter()
            .map(|row| poot_llm::encoders::segment_inputs(&row.types))
            .collect();
        let rows: Vec<poot_llm::driver::EncodeRow<'_>> = rows
            .iter()
            .zip(&inputs)
            .map(|(row, inputs)| poot_llm::driver::EncodeRow {
                tokens: Some(&row.ids),
                inputs,
            })
            .collect();
        Ok(self.driver.encode(&EncodeRequest { rows: &rows })?)
    }

    fn admin(&mut self, request: AdminRequest) -> Result<AdminReply, AdminError> {
        self.reap_retired();
        match request {
            AdminRequest::List => Ok(AdminReply::Adapters(self.adapters.infos())),
            AdminRequest::Load { .. } if self.caps().row_counts.is_empty() => {
                Err(AdminError::Refused {
                    class: FailureClass::Request,
                    reason: "the served model is an encoder: it generates nothing to adapt".into(),
                })
            }
            AdminRequest::Load { name, path } => {
                if self
                    .adapters
                    .loaded
                    .iter()
                    .any(|(loaded, _)| *loaded == name)
                {
                    return Err(AdminError::Refused {
                        class: FailureClass::Request,
                        reason: format!("lora adapter {name:?} is already loaded"),
                    });
                }
                let adapter = LoraAdapter::load(&path).map_err(|error| AdminError::Refused {
                    class: FailureClass::Request,
                    reason: format!("load the adapter at {path:?}: {error}"),
                })?;
                let mut loaded = self.adapters.loaded.clone();
                loaded.push((name.clone(), adapter));
                self.install(loaded)?;
                let index = self.adapters.loaded.len() as u32;
                Ok(AdminReply::Loaded(AdapterInfo { name, index }))
            }
            AdminRequest::Unload { name } => {
                let Some(position) = self
                    .adapters
                    .loaded
                    .iter()
                    .position(|(loaded, _)| *loaded == name)
                else {
                    return Err(AdminError::NotFound(name));
                };
                let mut loaded = self.adapters.loaded.clone();
                loaded.remove(position);
                self.install(loaded)?;
                Ok(AdminReply::Unloaded(AdapterInfo {
                    name,
                    index: position as u32 + 1,
                }))
            }
        }
    }

    fn kv_blocks(&self) -> Option<(usize, usize)> {
        let free = self.driver.free_blocks()?;
        Some((free, self.caps().kv_blocks))
    }

    fn resident_bytes(&self) -> u64 {
        self.driver
            .live_role_bytes()
            .iter()
            .map(|&(_, bytes)| bytes)
            .sum()
    }

    fn compiles(&self) -> u64 {
        self.driver.stats().compiles
    }

    fn timing(&self) -> Option<TimingSnapshot> {
        Some(self.driver.executor_stats().timing)
    }
}

/// Why [`run_engine`] returned.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Exit {
    /// Every sender is gone and every request finished.
    Shutdown,
    /// The engine's state cannot be trusted any more. Every request was failed; nothing was released.
    Fault(String),
}

/// The loop's state: the requests that hold an engine sequence and those waiting for one, and the
/// encode and administration jobs that need the engine to themselves.
struct Scheduler<'a, E: ServeEngine> {
    engine: &'a mut E,
    metrics: &'a Metrics,
    rx: &'a Receiver<EngineMsg>,
    slots: Vec<Slot<E::Seq>>,
    waiting: VecDeque<Job>,
    encodes: VecDeque<EncodeJob>,
    admins: VecDeque<AdminJob>,
    closed: bool,
    /// The last exclusive job run was an encode: the next one prefers an administration job.
    last_was_encode: bool,
    /// Steps ran since the executor's timing was last published.
    timing_dirty: bool,
    /// When the executor's timing was last published.
    timing_published: Option<Instant>,
}

/// Serve `rx` over `engine` until every sender is gone and the work is done, or the engine faults.
pub(crate) fn run_engine<E: ServeEngine>(
    engine: &mut E,
    rx: &Receiver<EngineMsg>,
    metrics: &Metrics,
) -> Exit {
    let mut scheduler = Scheduler {
        engine,
        metrics,
        rx,
        slots: Vec::new(),
        waiting: VecDeque::new(),
        encodes: VecDeque::new(),
        admins: VecDeque::new(),
        closed: false,
        last_was_encode: false,
        timing_dirty: false,
        timing_published: None,
    };
    match scheduler.run() {
        Ok(()) => Exit::Shutdown,
        Err(reason) => {
            scheduler.fail_everything(&reason);
            Exit::Fault(reason)
        }
    }
}

impl<E: ServeEngine> Scheduler<'_, E> {
    fn run(&mut self) -> Result<(), String> {
        loop {
            self.intake();
            if self.closed && self.idle() {
                return Ok(());
            }
            self.release_cancelled()?;
            self.run_exclusive()?;
            // An encode job needs the engine empty (its cache would alias an open sequence's), so
            // nothing new opens while one waits. An administration job does not: it ran between steps.
            if self.encodes.is_empty() {
                self.admit()?;
            }
            if self.slots.is_empty() {
                // Idle: whatever the last steps measured is published before the loop blocks.
                self.publish_timing(true);
                continue;
            }
            self.step()?;
        }
    }

    /// No request is running, waiting or queued.
    fn idle(&self) -> bool {
        self.slots.is_empty()
            && self.waiting.is_empty()
            && self.encodes.is_empty()
            && self.admins.is_empty()
    }

    /// Move queued messages into their queues without blocking; block only when there is nothing else
    /// to do.
    fn intake(&mut self) {
        if self.idle() {
            match self.rx.recv_timeout(IDLE_POLL) {
                Ok(msg) => self.accept(msg),
                Err(RecvTimeoutError::Timeout) => return,
                Err(RecvTimeoutError::Disconnected) => {
                    self.closed = true;
                    return;
                }
            }
        }
        loop {
            match self.rx.try_recv() {
                Ok(msg) => self.accept(msg),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.closed = true;
                    return;
                }
            }
        }
    }

    fn accept(&mut self, msg: EngineMsg) {
        match msg {
            EngineMsg::Generate(job) => self.waiting.push_back(*job),
            EngineMsg::Encode(job) => self.encodes.push_back(job),
            EngineMsg::Admin(job) => self.admins.push_back(job),
        }
    }

    fn count_cancelled(&self) {
        self.metrics
            .requests_cancelled
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Release every request whose client stopped waiting, and drop waiting ones before they open.
    fn release_cancelled(&mut self) -> Result<(), String> {
        let before = self.waiting.len();
        self.waiting
            .retain(|job| !job.cancel.load(Ordering::SeqCst));
        for _ in self.waiting.len()..before {
            self.count_cancelled();
        }
        self.fail_overdue();
        let before = self.encodes.len();
        self.encodes
            .retain(|job| !job.cancel.load(Ordering::SeqCst));
        for _ in self.encodes.len()..before {
            self.count_cancelled();
        }
        while let Some(index) = self.slots.iter().position(Slot::cancelled) {
            let slot = self.slots.remove(index);
            self.finish(slot, SlotFate::Cancelled)?;
        }
        Ok(())
    }

    /// Fail the waiting requests the engine has not opened by their `start_by`, as unavailable: a request
    /// does not queue without bound behind a long-running one.
    fn fail_overdue(&mut self) {
        let now = Instant::now();
        let (overdue, waiting): (VecDeque<Job>, VecDeque<Job>) = std::mem::take(&mut self.waiting)
            .into_iter()
            .partition(|job| now >= job.start_by);
        self.waiting = waiting;
        for job in overdue {
            let waited = job.submitted.elapsed().as_secs();
            reject_job(
                job,
                self.metrics,
                FailureClass::Unsupported,
                format!("the request waited {waited} s for the engine and was not started"),
            );
        }
    }

    /// Run at most [`EXCLUSIVE_JOBS_PER_ITERATION`] queued jobs that need the engine to themselves, between
    /// steps. An encode step shares the cache of an open sequence, so it runs once none is open; an
    /// administration job runs whatever is open: the driver keeps a generation a running request holds
    /// until the request lets go.
    fn run_exclusive(&mut self) -> Result<(), String> {
        for _ in 0..EXCLUSIVE_JOBS_PER_ITERATION {
            let encode_ready = self.slots.is_empty() && !self.encodes.is_empty();
            let admin_ready = !self.admins.is_empty();
            // The two classes alternate when both wait, so a steady stream of one never starves the other.
            if encode_ready && !(admin_ready && self.last_was_encode) {
                let job = self.encodes.pop_front().expect("checked non-empty");
                self.last_was_encode = true;
                self.encode(job)?;
            } else if admin_ready {
                let job = self.admins.pop_front().expect("checked non-empty");
                self.last_was_encode = false;
                let _ = job.reply.send(self.engine.admin(job.request));
            } else {
                break;
            }
        }
        Ok(())
    }

    fn encode(&mut self, job: EncodeJob) -> Result<(), String> {
        let EncodeJob { rows, reply, .. } = job;
        match self.engine.encode(&rows) {
            Ok(encoded) => {
                deliver_terminal::<EncodeReply>(self.metrics, &reply, Ok(encoded));
                Ok(())
            }
            Err(EngineError::Refused { class, reason }) => {
                let failure = AccountedGenerationFailure::new(self.metrics, class, reason);
                let _ = reply.send(Err(failure));
                Ok(())
            }
            Err(EngineError::Fault(reason)) => {
                let message = format!("the engine faulted: {reason}");
                let failure =
                    AccountedGenerationFailure::new(self.metrics, FailureClass::Fault, message);
                let _ = reply.send(Err(failure));
                Err(format!("encode: {reason}"))
            }
        }
    }

    /// Open waiting requests, oldest first, while a row and KV room remain.
    ///
    /// Known limit: the queue is strictly first-in-first-out. A request at the front that finds no KV room
    /// waits for a running one to finish and blocks the smaller requests behind it, which could have opened.
    fn admit(&mut self) -> Result<(), String> {
        let max_seq = self.engine.max_seq();
        while self.slots.len() < self.engine.rows() {
            let Some(front) = self.waiting.front() else {
                break;
            };
            let prompt_len = front.prompt.tokens().len();
            if prompt_len == 0 || prompt_len >= max_seq {
                let job = self.waiting.pop_front().expect("peeked above");
                reject_job(
                    job,
                    self.metrics,
                    FailureClass::Request,
                    format!(
                        "a prompt of {prompt_len} tokens does not fit the engine's {max_seq} positions"
                    ),
                );
                continue;
            }
            // The KV cap shortens the generation, as `max_tokens` past the window always has.
            let max_new = front.max_new.min(max_seq - prompt_len);
            if max_new == 0 {
                let job = self.waiting.pop_front().expect("peeked above");
                let completion = Completion {
                    prompt_tokens: prompt_len,
                    tokens: Vec::new(),
                    text: String::new(),
                    logprobs: Vec::new(),
                    finish: FinishReason::Length,
                };
                GenEvent::deliver_done(self.metrics, &job.reply, completion);
                continue;
            }
            let request = OpenRequest {
                prompt: front.prompt.tokens(),
                max_new,
                adapter: front.lora_adapter,
                inputs: &front.prompt.inputs,
            };
            match self.engine.open(request) {
                Ok(Opening::Opened { seq, reused }) => {
                    let job = self.waiting.pop_front().expect("peeked above");
                    self.metrics
                        .queue_wait
                        .observe(job.submitted.elapsed().as_secs_f64() * 1000.0);
                    record_prefix_admission(&self.metrics.cache, prompt_len, reused);
                    self.slots.push(Slot::new(seq, job, reused, max_new));
                }
                Ok(Opening::NoRoom { needed_blocks }) => {
                    if !self.slots.is_empty() {
                        // A running request frees its blocks when it ends: wait.
                        break;
                    }
                    let job = self.waiting.pop_front().expect("peeked above");
                    reject_job(
                        job,
                        self.metrics,
                        FailureClass::Unsupported,
                        format!(
                            "the KV pool cannot hold the request: {needed_blocks} more blocks needed"
                        ),
                    );
                }
                Err(EngineError::Refused { class, reason }) => {
                    let job = self.waiting.pop_front().expect("peeked above");
                    reject_job(job, self.metrics, class, reason);
                }
                Err(EngineError::Fault(reason)) => {
                    return Err(format!("open a sequence: {reason}"));
                }
            }
        }
        Ok(())
    }

    /// One step over the rows that have work, then the commit of what it picked.
    fn step(&mut self) -> Result<(), String> {
        let widest = self.engine.widest_piece();
        let decoding = self.slots.iter().filter(|s| !s.prefilling()).count();
        let pending: Vec<(usize, usize)> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.prefilling())
            .map(|(index, slot)| (index, (slot.prompt_len - slot.prefilled).min(widest)))
            .collect();
        let prefill = schedule_prefill_admissions(&pending, decoding, widest + self.engine.rows());
        let order: Vec<usize> = (0..self.slots.len())
            .filter(|index| !self.slots[*index].prefilling() || prefill.contains(index))
            .collect();
        let results = {
            let mut rows: Vec<Row<'_, E::Seq>> = self
                .slots
                .iter_mut()
                .enumerate()
                .filter(|(index, _)| order.contains(index))
                .map(|(_, slot)| Row {
                    seq: slot.seq,
                    work: if slot.prefilling() {
                        Work::Prefill {
                            upto: slot.prompt_len,
                        }
                    } else {
                        Work::Decode
                    },
                    sampler: &mut slot.sampler,
                })
                .collect();
            self.engine.step(&mut rows)
        };
        let results = match results {
            Ok(results) => results,
            Err(EngineError::Refused { class, reason }) => {
                return self.fail_rows(&order, class, &reason);
            }
            Err(EngineError::Fault(reason)) => return Err(format!("step: {reason}")),
        };
        let matches_rows = results.len() == order.len()
            && results
                .iter()
                .zip(&order)
                .all(|(result, &index)| result.seq == self.slots[index].seq);
        if !matches_rows {
            return Err("a step returned results that do not match its rows".to_string());
        }
        self.metrics
            .batch_size
            .store(self.slots.len() as u64, Ordering::Relaxed);
        let mut fates = Vec::new();
        for (result, &index) in results.iter().zip(&order) {
            let slot = &mut self.slots[index];
            slot.prefilled += result.consumed;
            self.metrics.speculative.record(result.speculation);
            self.metrics
                .prompt_tokens
                .fetch_add(result.consumed as u64, Ordering::Relaxed);
            let fate = if result.tokens.is_empty() {
                SlotFate::Continue
            } else {
                match commit_tokens(self.engine, self.metrics, slot, result) {
                    Ok(fate) => fate,
                    Err(EngineError::Refused { class, reason }) => {
                        fates.push((index, Err((class, reason))));
                        continue;
                    }
                    Err(EngineError::Fault(reason)) => return Err(format!("commit: {reason}")),
                }
            };
            if fate != SlotFate::Continue {
                fates.push((index, Ok(fate)));
            }
        }
        // Remove from the back so earlier indices stay valid.
        for (index, fate) in fates.into_iter().rev() {
            let slot = self.slots.remove(index);
            match fate {
                Ok(fate) => self.finish(slot, fate)?,
                Err((class, reason)) => self.fail_slot(slot, class, &reason)?,
            }
        }
        self.refresh_gauges();
        Ok(())
    }

    /// Fail the requests of a refused step; the engine is as it was, so it keeps serving the rest.
    fn fail_rows(
        &mut self,
        order: &[usize],
        class: FailureClass,
        reason: &str,
    ) -> Result<(), String> {
        for &index in order.iter().rev() {
            let slot = self.slots.remove(index);
            self.fail_slot(slot, class, reason)?;
        }
        Ok(())
    }

    /// End `slot` the way `fate` decided. A finished or cancelled request registers its filled prompt
    /// prefix; the terminal event goes out after the engine took the sequence back.
    fn finish(&mut self, slot: Slot<E::Seq>, fate: SlotFate) -> Result<(), String> {
        let outcome = match self.engine.release(slot.seq, Release::Finished) {
            Ok(outcome) => outcome,
            Err(error) => {
                // The client hears the fault, not a closed channel, before the loop stops.
                let reason = format!("release: {error:?}");
                let message = format!("the engine faulted: {reason}");
                let _ =
                    slot.reply
                        .send(GenEvent::failed(self.metrics, FailureClass::Fault, message));
                return Err(reason);
            }
        };
        record_prefix_outcome(&self.metrics.cache, outcome);
        tracing::debug!(generated = slot.generated, ?fate, ?outcome, "request done");
        match fate {
            SlotFate::Finished { reason } => {
                let (reply, completion) = slot.into_completion(reason);
                GenEvent::deliver_done(self.metrics, &reply, completion);
            }
            SlotFate::Cancelled => self.count_cancelled(),
            SlotFate::Continue => unreachable!("a continuing request is not finished"),
        }
        Ok(())
    }

    /// End `slot` with a failure. Its sequence did not run to an end, so it registers no prefix.
    fn fail_slot(
        &mut self,
        slot: Slot<E::Seq>,
        class: FailureClass,
        reason: &str,
    ) -> Result<(), String> {
        let released = self.engine.release(slot.seq, Release::Preempted);
        // The client hears why its request failed even when the release failed too.
        let _ = slot
            .reply
            .send(GenEvent::failed(self.metrics, class, reason));
        released
            .map(drop)
            .map_err(|error| format!("release: {error:?}"))
    }

    fn refresh_gauges(&mut self) {
        let cache = &self.metrics.cache;
        if let Some((free, total)) = self.engine.kv_blocks() {
            cache
                .kv_used_blocks
                .store((total - free.min(total)) as u64, Ordering::Relaxed);
        }
        cache
            .gpu_vram_used_bytes
            .store(self.engine.resident_bytes(), Ordering::Relaxed);
        cache
            .engine_compiles
            .store(self.engine.compiles(), Ordering::Relaxed);
        self.metrics
            .batch_size
            .store(self.slots.len() as u64, Ordering::Relaxed);
        self.timing_dirty = true;
        self.publish_timing(false);
    }

    /// Publish the executor's timing summary when steps ran since the last publish: at most once per
    /// [`TIMING_PUBLISH_INTERVAL`] while stepping, and always (`force`) once the loop goes idle. The
    /// summary is fixed-size, so a metrics reader that never comes retains nothing.
    fn publish_timing(&mut self, force: bool) {
        if !self.timing_dirty {
            return;
        }
        let due = self
            .timing_published
            .is_none_or(|at| at.elapsed() >= TIMING_PUBLISH_INTERVAL);
        if !(force || due) {
            return;
        }
        if let Some(snapshot) = self.engine.timing() {
            self.metrics.timing.publish(&snapshot);
        }
        self.timing_dirty = false;
        self.timing_published = Some(Instant::now());
    }

    /// The engine faulted: every request fails with the reason and nothing is released, because the
    /// device may still own the buffers a release would hand back.
    fn fail_everything(&mut self, reason: &str) {
        let message = format!("the engine faulted: {reason}");
        let fault = FailureClass::Fault;
        for slot in self.slots.drain(..) {
            let _ = slot
                .reply
                .send(GenEvent::failed(self.metrics, fault, &message));
        }
        let mut queued: Vec<EngineMsg> = Vec::new();
        while let Ok(msg) = self.rx.try_recv() {
            queued.push(msg);
        }
        for job in self.waiting.drain(..) {
            reject_job(job, self.metrics, fault, &message);
        }
        for job in self.encodes.drain(..) {
            let failure = AccountedGenerationFailure::new(self.metrics, fault, &message);
            let _ = job.reply.send(Err(failure));
        }
        for job in self.admins.drain(..) {
            let _ = job.reply.send(Err(AdminError::Refused {
                class: fault,
                reason: message.clone(),
            }));
        }
        for msg in queued {
            match msg {
                EngineMsg::Generate(job) => reject_job(*job, self.metrics, fault, &message),
                EngineMsg::Encode(job) => {
                    let failure = AccountedGenerationFailure::new(self.metrics, fault, &message);
                    let _ = job.reply.send(Err(failure));
                }
                EngineMsg::Admin(job) => {
                    let _ = job.reply.send(Err(AdminError::Refused {
                        class: fault,
                        reason: message.clone(),
                    }));
                }
            }
        }
    }
}

/// Publishes the engine's death when the engine thread ends for any reason other than a clean shutdown.
/// The thread's last act: a guard that never ran leaves `/health` claiming a dead engine serves.
struct DeathGuard {
    health: EngineHealth,
    death: Option<Death>,
}

impl Drop for DeathGuard {
    fn drop(&mut self) {
        if let Some(death) = self.death.take() {
            self.health.die(death);
        }
    }
}

pub(crate) fn panic_reason(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|reason| (*reason).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "the engine thread panicked".to_string())
}

/// The engine thread's body: run the loop and publish how it ended. The engine is returned, not
/// dropped, so the caller decides when its device buffers go: after a fault they stay owned until the
/// process shuts down.
pub(crate) fn run_engine_thread<E: ServeEngine>(
    mut engine: E,
    rx: Receiver<EngineMsg>,
    metrics: &Metrics,
    health: &EngineHealth,
) -> E {
    let mut guard = DeathGuard {
        health: health.clone(),
        death: None,
    };
    match catch_unwind(AssertUnwindSafe(|| run_engine(&mut engine, &rx, metrics))) {
        Ok(Exit::Shutdown) => {}
        Ok(Exit::Fault(reason)) => {
            tracing::error!(reason, "engine fault");
            guard.death = Some(Death::Fault(reason));
        }
        Err(payload) => {
            let reason = panic_reason(payload.as_ref());
            tracing::error!(reason, "engine thread panicked");
            guard.death = Some(Death::Panicked(reason));
        }
    }
    drop(guard);
    engine
}
