//! The one serving loop: every iteration drains the job queue, releases cancelled requests, opens as
//! many waiting requests as the engine has rows and room for, steps the rows that have work and commits
//! what the step picked. It runs over [`ServeEngine`], the driver's serving surface, so the same loop
//! serves every backend and family the driver prepares, and a recording fake stands in for the driver in
//! the scheduling tests.

use std::any::Any;
use std::collections::{BTreeSet, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError};
use std::time::Duration;

use poot_llm::driver::error::DriverError;
use poot_llm::driver::step::PickPolicy;
use poot_llm::driver::{
    AdapterRef, Admission, Driver, ModelHandle, PrefixOutcome, Release, RowWork, SeqId, SeqRequest,
    StepRow,
};
use poot_llm::{Sampler, TokenLogprob};

use crate::batch::commit::{
    SlotFate, commit_tokens, record_prefix_admission, record_prefix_outcome,
};
use crate::batch::schedule::schedule_prefill_admissions;
use crate::batch::slot::{Slot, Waiting, reject_job};
use crate::engine_plan::{Death, EngineHealth};
use crate::metrics::Metrics;
use crate::types::{GenEvent, Job};

/// How long an idle loop waits for a job before it looks again.
const IDLE_POLL: Duration = Duration::from_millis(10);

/// Why an engine call failed, as the loop must treat it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum EngineError {
    /// Refused before anything was applied. The engine is as it was, so only the requests the call names fail.
    Refused(String),
    /// The call may have left a step half applied, or the device unusable: the engine is not served from
    /// again.
    Fault(String),
}

impl From<DriverError> for EngineError {
    fn from(error: DriverError) -> Self {
        // A refusal runs before anything is applied. So does an unusable pick (no finite logit among a row's
        // logits): the driver advances its sequences only after every row picked, so the step left no trace.
        let refused = matches!(
            error,
            DriverError::InvalidRequest(_)
                | DriverError::InvalidOptions(_)
                | DriverError::Unsupported(_)
                | DriverError::PreparedSet(_)
                | DriverError::Sampler(_)
        );
        if refused {
            EngineError::Refused(error.to_string())
        } else {
            EngineError::Fault(error.to_string())
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
    fn encode(&self, prompt: &str) -> anyhow::Result<Vec<u32>>;
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
    /// `(free, total)` blocks of the paged KV pool.
    fn kv_blocks(&self) -> Option<(usize, usize)>;
    /// Device bytes the engine holds: its weights, KV pool and workspace.
    fn resident_bytes(&self) -> u64;
    /// Programs compiled so far: constant across steps once the prepared set is warm.
    fn compiles(&self) -> u64;

    /// Used over total blocks of the KV pool, 0 without one.
    fn kv_used_ratio(&self) -> f64 {
        match self.kv_blocks() {
            Some((free, total)) if total > 0 => (total - free.min(total)) as f64 / total as f64,
            _ => 0.0,
        }
    }
}

/// A prepared [`Driver`] and the loaded model it serves.
pub(crate) struct DriverEngine {
    driver: Driver,
    handle: Arc<ModelHandle>,
}

impl DriverEngine {
    pub(crate) fn new(driver: Driver, handle: Arc<ModelHandle>) -> Self {
        Self { driver, handle }
    }

    fn caps(&self) -> &poot_llm::driver::DriverCaps {
        self.driver
            .caps()
            .expect("an engine is built over a prepared driver")
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

    fn encode(&self, prompt: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self.handle.text().encode(prompt)?)
    }

    fn piece(&self, tokens: &[u32], gen_start: usize) -> anyhow::Result<String> {
        Ok(self.handle.text().stream_piece(tokens, gen_start)?)
    }

    fn open(&mut self, request: OpenRequest<'_>) -> Result<Opening<SeqId>, EngineError> {
        let OpenRequest {
            prompt,
            max_new,
            adapter,
        } = request;
        Ok(
            match self.driver.open(SeqRequest {
                prompt,
                max_new,
                adapter,
            })? {
                Admission::Opened { seq, reused } => Opening::Opened { seq, reused },
                Admission::NoRoom { needed_blocks } => Opening::NoRoom { needed_blocks },
            },
        )
    }

    fn step(&mut self, rows: &mut [Row<'_, SeqId>]) -> Result<Vec<RowResult<SeqId>>, EngineError> {
        let mut driver_rows: Vec<StepRow<'_>> = rows
            .iter_mut()
            .map(|row| StepRow {
                seq: row.seq,
                work: match row.work {
                    Work::Prefill { upto } => RowWork::Prefill { upto },
                    Work::Decode => RowWork::Decode {
                        ahead: &[],
                        pick: PickPolicy::Greedy,
                    },
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
}

/// Why [`run_engine`] returned.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Exit {
    /// Every sender is gone and every request finished.
    Shutdown,
    /// The engine's state cannot be trusted any more. Every request was failed; nothing was released.
    Fault(String),
}

/// The loop's state: the requests that hold an engine sequence and those waiting for one.
struct Scheduler<'a, E: ServeEngine> {
    engine: &'a mut E,
    metrics: &'a Metrics,
    rx: &'a Receiver<Job>,
    slots: Vec<Slot<E::Seq>>,
    waiting: VecDeque<Waiting>,
    closed: bool,
}

/// Serve `rx` over `engine` until every sender is gone and the work is done, or the engine faults.
pub(crate) fn run_engine<E: ServeEngine>(
    engine: &mut E,
    rx: &Receiver<Job>,
    metrics: &Metrics,
) -> Exit {
    let mut scheduler = Scheduler {
        engine,
        metrics,
        rx,
        slots: Vec::new(),
        waiting: VecDeque::new(),
        closed: false,
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
            if self.closed && self.slots.is_empty() && self.waiting.is_empty() {
                return Ok(());
            }
            self.release_cancelled()?;
            self.admit()?;
            if self.slots.is_empty() {
                continue;
            }
            self.step()?;
        }
    }

    /// Move queued jobs into `waiting` without blocking; block only when there is nothing else to do.
    fn intake(&mut self) {
        if self.slots.is_empty() && self.waiting.is_empty() {
            match self.rx.recv_timeout(IDLE_POLL) {
                Ok(job) => self.accept(job),
                Err(RecvTimeoutError::Timeout) => return,
                Err(RecvTimeoutError::Disconnected) => {
                    self.closed = true;
                    return;
                }
            }
        }
        loop {
            match self.rx.try_recv() {
                Ok(job) => self.accept(job),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.closed = true;
                    return;
                }
            }
        }
    }

    fn accept(&mut self, job: Job) {
        match self.engine.encode(&job.prompt) {
            Ok(prompt) => self.waiting.push_back(Waiting { job, prompt }),
            Err(error) => reject_job(job, self.metrics, format!("encode the prompt: {error:#}")),
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
            .retain(|waiting| !waiting.job.cancel.load(Ordering::SeqCst));
        for _ in self.waiting.len()..before {
            self.count_cancelled();
        }
        while let Some(index) = self.slots.iter().position(Slot::cancelled) {
            let slot = self.slots.remove(index);
            self.finish(slot, SlotFate::Cancelled)?;
        }
        Ok(())
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
            let prompt_len = front.prompt.len();
            if prompt_len == 0 || prompt_len >= max_seq {
                let waiting = self.waiting.pop_front().expect("peeked above");
                reject_job(
                    waiting.job,
                    self.metrics,
                    format!(
                        "a prompt of {prompt_len} tokens does not fit the engine's {max_seq} positions"
                    ),
                );
                continue;
            }
            // The KV cap shortens the generation, as `max_tokens` past the window always has.
            let max_new = front.job.max_new.min(max_seq - prompt_len);
            if max_new == 0 {
                let waiting = self.waiting.pop_front().expect("peeked above");
                GenEvent::deliver_done(
                    self.metrics,
                    &waiting.job.reply,
                    waiting.prompt,
                    Vec::new(),
                    false,
                );
                continue;
            }
            let request = OpenRequest {
                prompt: &front.prompt,
                max_new,
                adapter: front.job.lora_adapter,
            };
            match self.engine.open(request) {
                Ok(Opening::Opened { seq, reused }) => {
                    let waiting = self.waiting.pop_front().expect("peeked above");
                    self.metrics
                        .queue_wait
                        .observe(waiting.job.submitted.elapsed().as_secs_f64() * 1000.0);
                    record_prefix_admission(&self.metrics.cache, prompt_len, reused);
                    self.slots.push(Slot::new(seq, waiting, reused, max_new));
                }
                Ok(Opening::NoRoom { needed_blocks }) => {
                    if !self.slots.is_empty() {
                        // A running request frees its blocks when it ends: wait.
                        break;
                    }
                    let waiting = self.waiting.pop_front().expect("peeked above");
                    reject_job(
                        waiting.job,
                        self.metrics,
                        format!(
                            "the KV pool cannot hold the request: {needed_blocks} more blocks needed"
                        ),
                    );
                }
                Err(EngineError::Refused(reason)) => {
                    let waiting = self.waiting.pop_front().expect("peeked above");
                    reject_job(waiting.job, self.metrics, reason);
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
            Err(EngineError::Refused(reason)) => return self.fail_rows(&order, &reason),
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
            self.metrics
                .prompt_tokens
                .fetch_add(result.consumed as u64, Ordering::Relaxed);
            let fate = if result.tokens.is_empty() {
                SlotFate::Continue
            } else {
                match commit_tokens(self.engine, self.metrics, slot, result) {
                    Ok(fate) => fate,
                    Err(EngineError::Refused(reason)) => {
                        fates.push((index, Err(reason)));
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
                Err(reason) => self.fail_slot(slot, &reason)?,
            }
        }
        self.refresh_gauges();
        Ok(())
    }

    /// Fail the requests of a refused step; the engine is as it was, so it keeps serving the rest.
    fn fail_rows(&mut self, order: &[usize], reason: &str) -> Result<(), String> {
        for &index in order.iter().rev() {
            let slot = self.slots.remove(index);
            self.fail_slot(slot, reason)?;
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
                let _ = slot.reply.send(GenEvent::failed(self.metrics, message));
                return Err(reason);
            }
        };
        record_prefix_outcome(&self.metrics.cache, outcome);
        tracing::debug!(generated = slot.generated, ?fate, ?outcome, "request done");
        match fate {
            SlotFate::Finished { natural } => {
                GenEvent::deliver_done(
                    self.metrics,
                    &slot.reply,
                    slot.tokens,
                    slot.logprobs,
                    natural,
                );
            }
            SlotFate::Cancelled => self.count_cancelled(),
            SlotFate::Continue => unreachable!("a continuing request is not finished"),
        }
        Ok(())
    }

    /// End `slot` with a failure. Its sequence did not run to an end, so it registers no prefix.
    fn fail_slot(&mut self, slot: Slot<E::Seq>, reason: &str) -> Result<(), String> {
        let released = self.engine.release(slot.seq, Release::Preempted);
        // The client hears why its request failed even when the release failed too.
        let _ = slot.reply.send(GenEvent::failed(self.metrics, reason));
        released
            .map(drop)
            .map_err(|error| format!("release: {error:?}"))
    }

    fn refresh_gauges(&self) {
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
    }

    /// The engine faulted: every request fails with the reason and nothing is released, because the
    /// device may still own the buffers a release would hand back.
    fn fail_everything(&mut self, reason: &str) {
        let message = format!("the engine faulted: {reason}");
        for slot in self.slots.drain(..) {
            let _ = slot.reply.send(GenEvent::failed(self.metrics, &message));
        }
        for waiting in self.waiting.drain(..) {
            reject_job(waiting.job, self.metrics, &message);
        }
        while let Ok(job) = self.rx.try_recv() {
            reject_job(job, self.metrics, &message);
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

fn panic_reason(payload: &(dyn Any + Send)) -> String {
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
    rx: Receiver<Job>,
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
