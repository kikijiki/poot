//! The driver's serving surface: `open`, `step`, `commit` and `release` over a paged KV pool or the one
//! private contiguous cache of a single sequence (Card 735, dserve.md section 3.1).
//!
//! A sequence is a token list: its prompt and the generated tokens the caller has committed. A step
//! feeds each row's next tokens at their absolute positions and writes their K/V through the row's
//! block table; the row's pick is the token that follows the last fed position. What a step returns is
//! not yet part of the sequence: [`Driver::commit`] advances the sequence by the tokens the caller
//! keeps (S1), so a stop, end-of-sequence or `max_new` inside a multi-token result needs no rollback
//! for KV-only state: the surplus positions hold stale K/V that no later step reads (the mask hides
//! every key past a query's own position) and a later write overwrites.
//!
//! Every step runs on one prepared entry whose row count is a prepared capacity: rows beyond the live
//! ones are masked, and a row with fewer tokens than the step's widest is padded on the left (a
//! padding token's write-map entry is -1, so it writes nothing and the row's last position stays the
//! entry's last), so a changing composition never compiles. A step in which every picking row is
//! suffix-expressible and shares one rule runs that suffix entry; any other mix runs the logits entry
//! and every row picks on the host through its own `Sampler` (S3).

use std::collections::HashMap;
use std::num::NonZeroU64;

use poot_graph_ir::op::SampleRule;
use poot_models::model::{KvLayout, LogitRows, Phase, StepShape};

use super::block_table::{BLOCK_SIZE, PagedKvCache, PoolExhausted};
use super::caps::{Layout, PoolShape, StateResidency};
use super::chunks::Piece;
use super::error::{DriverError, InvalidRequest, PreparedSetRefusal};
use super::lora::AdapterRef;
use super::prefix_cache::{PrefixIdentity, PrefixOutcome, PrefixSkip};
use super::suffix::{self, Head, SuffixRows};
use super::{Driver, EntryKey, StepData, host_sample};
use crate::core::sampler::{Sampler, TokenLogprob};

/// A driver-issued sequence id, never reused while the driver lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeqId(NonZeroU64);

impl SeqId {
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

/// A sequence to admit.
#[derive(Clone, Copy, Debug)]
pub struct SeqRequest<'a> {
    pub prompt: &'a [u32],
    /// The most tokens the caller will commit after the prompt; the blocks for them are reserved at
    /// admission.
    pub max_new: usize,
    pub adapter: AdapterRef,
}

/// The outcome of [`Driver::open`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// `reused` leading prompt tokens are already resident: prefill resumes at that position.
    Opened { seq: SeqId, reused: usize },
    /// The pool cannot hold the request (`needed_blocks` more blocks), or every sequence slot is taken
    /// (`needed_blocks` is 0): the caller preempts a sequence or waits.
    NoRoom { needed_blocks: usize },
}

/// What one row does this step.
#[derive(Clone, Copy, Debug)]
pub enum RowWork<'a> {
    /// Feed prompt tokens up to position `upto` (exclusive). The driver feeds the largest prepared
    /// piece of the remainder and reports how many tokens it consumed; the row picks only when this
    /// step reaches the end of the prompt.
    Prefill { upto: usize },
    /// Feed the sequence's last committed token followed by `ahead` (verify-window drafts), and pick
    /// at every fed position: `ahead.len() + 1` tokens, the model's choice after each prefix, by
    /// `pick`. With nothing ahead it is a plain decode step and the row's own sampler picks.
    Decode { ahead: &'a [u32], pick: PickPolicy },
}

/// How a verify window picks at each of its positions. Only the greedy argmax is accepted today; a
/// sampled window (speculative sampling's residual draws) is a further variant, so callers match
/// with a wildcard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PickPolicy {
    Greedy,
}

/// One row of a step: a sequence, its work and the sampler its pick goes through.
pub struct StepRow<'a> {
    pub seq: SeqId,
    pub work: RowWork<'a>,
    pub sampler: &'a mut Sampler,
}

/// What a row produced.
#[derive(Debug)]
pub struct RowResult {
    pub seq: SeqId,
    /// Prompt tokens a `Prefill` row consumed; zero for a `Decode` row.
    pub consumed: usize,
    /// The picks, uncommitted: empty for a prefill row that has not reached the end of its prompt,
    /// one for a decode row, one per fed position for a verify window.
    pub tokens: Vec<u32>,
    /// One record per token when the row's sampler records logprobs.
    pub logprobs: Vec<TokenLogprob>,
}

/// Why a sequence is released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Release {
    Finished,
    Preempted,
}

/// One open sequence.
#[derive(Debug)]
pub(super) struct Seq {
    /// The sequence's slot in the paged cache: its block table.
    slot: usize,
    /// The prompt, then the committed generated tokens.
    tokens: Vec<u32>,
    prompt_len: usize,
    /// Prompt positions whose K/V is resident: reused at open, then advanced by prefill.
    done: usize,
    /// Positions the block table covers: the prompt and `max_new`.
    reserved: usize,
    adapter: AdapterRef,
    identity: PrefixIdentity,
    seeded: bool,
    /// The last step's picks, until `commit` takes them.
    pending: Vec<u32>,
}

/// Where the open sequences keep their K/V.
#[derive(Debug)]
enum KvStore {
    /// The one private cache of a single sequence: no block table, no prefix reuse.
    Contiguous,
    /// The shared paged pool.
    Paged {
        pool: PoolShape,
        cache: PagedKvCache,
    },
}

/// The KV store, the open sequences and their slots.
#[derive(Debug)]
pub(super) struct Serving {
    kv: KvStore,
    pub(super) seqs: HashMap<SeqId, Seq>,
    free_slots: Vec<usize>,
    next_seq: u64,
}

impl Serving {
    pub(super) fn new(layout: Layout) -> Self {
        let (kv, max_seqs) = match layout {
            Layout::Contiguous => (KvStore::Contiguous, 1),
            Layout::Paged(pool) => (
                KvStore::Paged {
                    pool,
                    cache: PagedKvCache::new(pool.max_seqs.get(), pool.blocks.get()),
                },
                pool.max_seqs.get(),
            ),
        };
        Self {
            kv,
            seqs: HashMap::new(),
            free_slots: (0..max_seqs).rev().collect(),
            next_seq: 0,
        }
    }

    /// The layout every entry this driver runs has.
    pub(super) fn kv(&self) -> KvLayout {
        match &self.kv {
            KvStore::Contiguous => KvLayout::Contiguous,
            KvStore::Paged { pool, .. } => pool.kv(),
        }
    }

    /// Blocks free in the paged pool; `None` under the contiguous layout.
    pub(super) fn free_blocks(&self) -> Option<usize> {
        match &self.kv {
            KvStore::Contiguous => None,
            KvStore::Paged { cache, .. } => Some(cache.free_blocks()),
        }
    }

    /// Drop the cached prefix blocks of an unloaded adapter generation.
    pub(super) fn evict_identities(&mut self, identities: &[PrefixIdentity]) {
        if let KvStore::Paged { cache, .. } = &mut self.kv {
            cache.evict_identities(identities);
        }
    }
}

/// One row of a step, planned: what it feeds and where.
struct RowPlan {
    feed: Vec<u32>,
    start: usize,
    /// Picks the row produces: none, one, or one per fed position of a verify window.
    picks: usize,
    /// A prefill piece (as opposed to a decode token).
    prefill: bool,
}

/// The phase a serving entry traces: a single-token step is the decode step whether its rows decode or
/// prefill their last prompt token (the graphs are the same), so a deployment that serves one-token
/// steps prepares one entry per row count, not two.
pub(super) fn serving_phase(tokens: usize, logits: LogitRows) -> Phase {
    if tokens == 1 && logits == LogitRows::Last {
        Phase::Decode
    } else {
        Phase::Prefill
    }
}

impl Driver {
    /// Admit `request`: reuse the cached prefix blocks its identity matches, reserve blocks for the
    /// rest of the prompt and `max_new` tokens. A pool that cannot hold it, after dropping cached
    /// prefixes nothing references, is [`Admission::NoRoom`].
    pub fn open(&mut self, request: SeqRequest<'_>) -> Result<Admission, DriverError> {
        let capacity = self.options.capacity.get();
        let SeqRequest {
            prompt,
            max_new,
            adapter,
        } = request;
        if prompt.is_empty() {
            return Err(InvalidRequest::EmptyPrompt.into());
        }
        if prompt.len() + max_new > capacity {
            return Err(InvalidRequest::Capacity {
                prompt: prompt.len(),
                max_new,
                max: capacity,
            }
            .into());
        }
        let residency = self.residency;
        let serving = self.serving.as_mut().ok_or(InvalidRequest::NotPrepared)?;
        let Some(&slot) = serving.free_slots.last() else {
            return Ok(Admission::NoRoom { needed_blocks: 0 });
        };
        let identity = self.lora.retain(adapter)?;
        let reused = match &mut serving.kv {
            KvStore::Contiguous => 0,
            KvStore::Paged { cache, .. } => {
                let mut admitted = cache.admit_prefix_for(slot, prompt, max_new, identity);
                if admitted.is_err() {
                    cache.evict_prefix();
                    admitted = cache.admit_prefix_for(slot, prompt, max_new, identity);
                }
                match admitted {
                    Ok(reused) => reused,
                    Err(PoolExhausted) => {
                        self.lora.release(adapter);
                        let needed = (prompt.len() + max_new).div_ceil(BLOCK_SIZE);
                        return Ok(Admission::NoRoom {
                            needed_blocks: needed.saturating_sub(cache.free_blocks()).max(1),
                        });
                    }
                }
            }
        };
        let private = matches!(serving.kv, KvStore::Contiguous);
        serving.free_slots.pop();
        serving.next_seq += 1;
        let seq = SeqId(NonZeroU64::new(serving.next_seq).expect("the counter starts at one"));
        serving.seqs.insert(
            seq,
            Seq {
                slot,
                tokens: prompt.to_vec(),
                prompt_len: prompt.len(),
                done: reused,
                reserved: prompt.len() + max_new,
                adapter,
                identity,
                seeded: false,
                pending: Vec::new(),
            },
        );
        if private || residency == StateResidency::Recurrent {
            // The private cache holds one sequence and recurrent state is one value (prepare enforces
            // a single sequence): a new sequence starts from zero state.
            self.executor
                .reset_state(self.exe, poot_executor::StateScope::All)?;
        }
        Ok(Admission::Opened { seq, reused })
    }

    /// Run one step over `rows`. Each row's sampler picks its token: the suffix entry when every
    /// picking row shares a suffix-expressible rule, otherwise the logits entry and the host path.
    /// The picks are not part of any sequence until [`Driver::commit`].
    pub fn step(&mut self, rows: &mut [StepRow<'_>]) -> Result<Vec<RowResult>, DriverError> {
        let serving = self.serving.as_ref().ok_or(InvalidRequest::NotPrepared)?;
        if rows.is_empty() {
            return Err(InvalidRequest::EmptyStep.into());
        }
        let max_rows = self.caps.as_ref().map_or(0, |caps| caps.rows.get());
        if rows.len() > max_rows {
            return Err(InvalidRequest::TooManyRows {
                rows: rows.len(),
                max: max_rows,
            }
            .into());
        }
        let plans = self.plan_rows(rows)?;
        for row in rows.iter_mut() {
            let seq = &serving.seqs[&row.seq];
            if !seq.seeded {
                row.sampler.seed_context(&seq.tokens[..seq.prompt_len]);
            }
        }
        let adapter = rows
            .iter()
            .map(|row| serving.seqs[&row.seq].adapter.generation())
            .try_fold(None, |held: Option<_>, generation| {
                match (held, generation) {
                    (held, None) => Ok(held),
                    (None, generation) => Ok(generation),
                    (Some(held), Some(generation)) if held == generation => Ok(Some(held)),
                    _ => Err(InvalidRequest::MixedAdapters),
                }
            })?;
        let window = plans.iter().any(|plan| plan.picks > 1);
        let tokens = plans.iter().map(|plan| plan.feed.len()).max().unwrap_or(1);
        let head = self.step_head(rows, &plans, window)?;
        let key = self.step_entry(rows.len(), tokens, window, head, adapter)?;
        let entry = self.entries[&key].clone();
        let StepShape {
            rows: entry_rows,
            tokens: width,
            capacity,
            kv,
            ..
        } = key.shape;
        let (entry_rows, width, capacity) = (entry_rows.get(), width.get(), capacity.get());
        let mut data = StepData {
            tokens: vec![0; entry_rows * width],
            pos: vec![0; entry_rows * width],
            lora: vec![0.0; entry_rows],
            ..StepData::default()
        };
        if let KvLayout::Paged { pool_slots } = kv {
            data.read = vec![0; entry_rows * capacity];
            data.write = vec![-1; pool_slots.get()];
        }
        let serving = self.serving.as_ref().expect("checked above");
        for (i, (row, plan)) in rows.iter().zip(&plans).enumerate() {
            let seq = &serving.seqs[&row.seq];
            let n = plan.feed.len();
            let map = match &serving.kv {
                KvStore::Paged { cache, .. } => {
                    let map = cache.slot_mapping(seq.slot, capacity);
                    data.read[i * capacity..(i + 1) * capacity]
                        .iter_mut()
                        .zip(&map)
                        .for_each(|(dst, &slot)| *dst = slot as i32);
                    Some(map)
                }
                KvStore::Contiguous => None,
            };
            data.lora[i] = seq.adapter.row_value();
            for k in 0..width {
                // Left padding: the row's real tokens are the last `n` of the step's width.
                let real = k.checked_sub(width - n);
                let at = i * width + k;
                match real {
                    Some(j) => {
                        let position = plan.start + j;
                        data.tokens[at] = plan.feed[j] as i32;
                        data.pos[at] = position as i32;
                        if let Some(map) = &map
                            && position < seq.reserved
                        {
                            data.write[map[position] as usize] = at as i32;
                        }
                    }
                    None => data.pos[at] = plan.start as i32,
                }
            }
        }
        // Sampler inputs of a non-greedy suffix: one row each, placeholders for rows that do not pick
        // and for the masked rows.
        if let Head::Sample(rule) = key.head
            && rule != SampleRule::Greedy
        {
            let mut seed = Vec::new();
            let mut params = Vec::new();
            let mut top_k = Vec::new();
            for (row, plan) in rows.iter_mut().zip(&plans) {
                let suffix_row = if plan.picks > 0 {
                    SuffixRows::push(row.sampler, rule)
                } else {
                    SuffixRows::placeholder(row.sampler, rule)
                };
                seed.push(suffix_row.seed);
                params.extend(suffix_row.params);
                top_k.extend(suffix_row.top_k);
            }
            // Masked rows repeat the first row's placeholder.
            let per_params = params.len() / rows.len();
            let masked = entry_rows - rows.len();
            seed.extend(std::iter::repeat_n(0, masked));
            for _ in 0..masked {
                params.extend_from_within(..per_params);
                if !top_k.is_empty() {
                    top_k.push(top_k[0]);
                }
            }
            (data.seed, data.params, data.top_k) = (seed, params, top_k);
        }

        let mut outputs = self.execute(&entry, &data)?;
        let positions = match key.shape.logits {
            LogitRows::Last => 1,
            LogitRows::All => width,
        };
        let mut picked: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        let mut host_picks = false;
        match key.head {
            // Nothing reads the output of a step no row picks in: no readback.
            _ if plans.iter().all(|plan| plan.picks == 0) => {
                picked.resize_with(rows.len(), Vec::new);
            }
            Head::Sample(_) => {
                let bytes = outputs.read()?;
                let ints: &[i32] = bytemuck::cast_slice(&bytes);
                for (i, plan) in plans.iter().enumerate() {
                    let mut tokens = Vec::with_capacity(plan.picks);
                    for j in 0..plan.picks {
                        // The row's last `picks` positions under All, its one position under Last.
                        let position = positions - plan.picks + j;
                        let at = 2 * (i * positions + position);
                        tokens.push(suffix::read_tokens(bytemuck::cast_slice(
                            &ints[at..at + 2],
                        ))?);
                    }
                    picked.push(tokens);
                }
            }
            Head::Logits => {
                let logits = host_sample::read_logits(&mut outputs)?;
                let vocab = logits.len() / entry_rows;
                host_picks = true;
                for (i, (row, plan)) in rows.iter_mut().zip(&plans).enumerate() {
                    let tokens = if plan.picks == 0 {
                        Vec::new()
                    } else {
                        vec![host_sample::pick(
                            row.sampler,
                            &logits[i * vocab..(i + 1) * vocab],
                        )?]
                    };
                    picked.push(tokens);
                }
            }
        }
        drop(outputs);
        self.stats.host_picks += u64::from(host_picks);

        let serving = self.serving.as_mut().expect("checked above");
        let mut results = Vec::with_capacity(rows.len());
        for ((row, plan), tokens) in rows.iter_mut().zip(&plans).zip(picked) {
            let seq = serving.seqs.get_mut(&row.seq).expect("planned above");
            let consumed = if plan.prefill { plan.feed.len() } else { 0 };
            seq.done += consumed;
            seq.seeded = true;
            seq.pending.clone_from(&tokens);
            results.push(RowResult {
                seq: row.seq,
                consumed,
                logprobs: if tokens.is_empty() {
                    Vec::new()
                } else {
                    row.sampler.take_logprobs()
                },
                tokens,
            });
        }
        Ok(results)
    }

    /// Advance `seq` by the first `kept` tokens of its last step's result. The rest of
    /// the result is discarded; KV written for it is stale and never read. The kept tokens, and only
    /// those, enter `sampler`'s penalty history and guided-decoding state, so a dropped or truncated
    /// result leaves the sampler exactly as the sequence is: `sampler` is the one the row stepped with.
    pub fn commit(
        &mut self,
        seq: SeqId,
        kept: usize,
        sampler: &mut Sampler,
    ) -> Result<(), DriverError> {
        let serving = self.serving.as_mut().ok_or(InvalidRequest::NotPrepared)?;
        let state = serving
            .seqs
            .get_mut(&seq)
            .ok_or(InvalidRequest::UnknownSeq(seq))?;
        if kept > state.pending.len() {
            return Err(InvalidRequest::CommitBeyondResult {
                kept,
                produced: state.pending.len(),
            }
            .into());
        }
        let pending = std::mem::take(&mut state.pending);
        for &token in &pending[..kept] {
            sampler.observe(token);
        }
        state.tokens.extend_from_slice(&pending[..kept]);
        Ok(())
    }

    /// Close `seq` and free its blocks. A finished sequence registers its filled prompt blocks in the
    /// prefix cache under its adapter identity, unless the model's state is recurrent: a cached block
    /// holds no recurrent state, so a later request reusing it would run without it.
    pub fn release(&mut self, seq: SeqId, mode: Release) -> Result<PrefixOutcome, DriverError> {
        let residency = self.residency;
        let serving = self.serving.as_mut().ok_or(InvalidRequest::NotPrepared)?;
        let state = serving
            .seqs
            .remove(&seq)
            .ok_or(InvalidRequest::UnknownSeq(seq))?;
        let outcome = match &mut serving.kv {
            KvStore::Contiguous => PrefixOutcome::Skipped(PrefixSkip::NoBlockTable),
            KvStore::Paged { cache, .. } => {
                if residency == StateResidency::Recurrent {
                    cache.reset(state.slot);
                    PrefixOutcome::Skipped(PrefixSkip::RecurrentState)
                } else if mode == Release::Preempted {
                    cache.reset(state.slot);
                    PrefixOutcome::Skipped(PrefixSkip::Preempted)
                } else {
                    let filled = state.done.min(state.prompt_len);
                    cache.register_and_reset_for(
                        state.slot,
                        &state.tokens[..filled],
                        state.identity,
                    );
                    PrefixOutcome::Registered {
                        tokens: filled / BLOCK_SIZE * BLOCK_SIZE,
                    }
                }
            }
        };
        serving.free_slots.push(state.slot);
        self.lora.release(state.adapter);
        Ok(outcome)
    }

    /// Plan what each row feeds and picks, refusing a row the sequence's state cannot take.
    fn plan_rows(&self, rows: &[StepRow<'_>]) -> Result<Vec<RowPlan>, DriverError> {
        let serving = self.serving.as_ref().expect("checked by the caller");
        let capacity = self.options.capacity.get();
        let granule = self.handle.config().prefill_granule.get();
        let mut seen = Vec::with_capacity(rows.len());
        let mut plans = Vec::with_capacity(rows.len());
        for row in rows {
            if seen.contains(&row.seq) {
                return Err(InvalidRequest::DuplicateRow(row.seq).into());
            }
            seen.push(row.seq);
            let seq = serving
                .seqs
                .get(&row.seq)
                .ok_or(InvalidRequest::UnknownSeq(row.seq))?;
            if !seq.pending.is_empty() {
                return Err(InvalidRequest::UncommittedResult(row.seq).into());
            }
            plans.push(match row.work {
                RowWork::Prefill { upto } => {
                    if upto <= seq.done || upto > seq.prompt_len {
                        return Err(InvalidRequest::PrefillRange {
                            seq: row.seq,
                            done: seq.done,
                            upto,
                            prompt: seq.prompt_len,
                        }
                        .into());
                    }
                    // A piece starts at and is sized in multiples of the granule; until the start is
                    // aligned (a reused prefix need not be) the row advances one token at a time.
                    let n = if seq.done.is_multiple_of(granule) {
                        match self.chunks.plan(upto - seq.done).first() {
                            Some(Piece::Prefill(n)) => n.get(),
                            _ => 1,
                        }
                    } else {
                        1
                    };
                    RowPlan {
                        feed: seq.tokens[seq.done..seq.done + n].to_vec(),
                        start: seq.done,
                        picks: usize::from(seq.done + n == seq.prompt_len),
                        prefill: true,
                    }
                }
                RowWork::Decode { ahead, pick } => {
                    let PickPolicy::Greedy = pick;
                    if seq.done < seq.prompt_len || seq.tokens.len() == seq.prompt_len {
                        return Err(InvalidRequest::DecodeBeforePrefill(row.seq).into());
                    }
                    let start = seq.tokens.len() - 1;
                    if start >= seq.reserved {
                        return Err(InvalidRequest::BeyondReservation(row.seq).into());
                    }
                    if start + 1 + ahead.len() > capacity {
                        return Err(InvalidRequest::Capacity {
                            prompt: seq.prompt_len,
                            max_new: start + 1 + ahead.len() - seq.prompt_len,
                            max: capacity,
                        }
                        .into());
                    }
                    if !ahead.is_empty() && suffix::rule_of(row.sampler) != Some(SampleRule::Greedy)
                    {
                        return Err(InvalidRequest::WindowNeedsGreedy(row.seq).into());
                    }
                    let mut feed = vec![seq.tokens[start]];
                    feed.extend_from_slice(ahead);
                    RowPlan {
                        picks: feed.len(),
                        feed,
                        start,
                        prefill: false,
                    }
                }
            });
        }
        Ok(plans)
    }

    /// The head a step runs: the suffix of the one rule every picking row shares, else the logits
    /// head. A step with a verify window runs the greedy suffix entry, so every picking row must be
    /// greedy-suffix-expressible: one that is not would silently get an unmasked argmax, and the
    /// whole step is refused instead.
    fn step_head(
        &self,
        rows: &[StepRow<'_>],
        plans: &[RowPlan],
        window: bool,
    ) -> Result<Option<Head>, InvalidRequest> {
        if window {
            let offender = rows
                .iter()
                .zip(plans)
                .find(|(row, plan)| {
                    plan.picks > 0 && suffix::rule_of(row.sampler) != Some(SampleRule::Greedy)
                })
                .map(|(row, _)| row.seq);
            return match offender {
                Some(seq) => Err(InvalidRequest::WindowStepNeedsGreedy(seq)),
                None => Ok(Some(Head::GREEDY)),
            };
        }
        let mut rules = rows
            .iter()
            .zip(plans)
            .filter(|(_, plan)| plan.picks > 0)
            .map(|(row, _)| suffix::rule_of(row.sampler));
        // No row picks: the output is unread, so any prepared head of the shape serves.
        Ok(rules.next().map(|first| match first {
            Some(rule) if rules.all(|other| other == Some(rule)) => Head::Sample(rule),
            _ => Head::Logits,
        }))
    }

    /// The prepared entry a step runs: the smallest admitted row count holding `live` rows, at the
    /// step's width, with `head` (`None`: any head, the greedy suffix first, then the logits head). A
    /// shape `prepare` never admitted is a typed refusal; a step never compiles.
    fn step_entry(
        &self,
        live: usize,
        width: usize,
        window: bool,
        head: Option<Head>,
        adapters: Option<super::lora::GenerationId>,
    ) -> Result<EntryKey, DriverError> {
        let serving = self.serving.as_ref().expect("checked by the caller");
        let logits = if window {
            LogitRows::All
        } else {
            LogitRows::Last
        };
        let phase = serving_phase(width, logits);
        let kv = serving.kv();
        let mut candidates: Vec<EntryKey> = self
            .entries
            .keys()
            .filter(|key| {
                key.phase == phase
                    && head.is_none_or(|head| key.head == head)
                    && key.adapters == adapters
                    && key.shape.tokens.get() == width
                    && key.shape.logits == logits
                    && key.shape.kv == kv
                    && key.shape.rows.get() >= live
            })
            .copied()
            .collect();
        candidates.sort_by_key(|key| {
            let rank = match key.head {
                Head::Sample(SampleRule::Greedy) => 0,
                Head::Logits => 1,
                Head::Sample(_) => 2,
            };
            (key.shape.rows, rank)
        });
        candidates.first().copied().ok_or_else(|| {
            PreparedSetRefusal::NotAdmitted {
                phase,
                shape: StepShape {
                    rows: std::num::NonZeroUsize::new(live).expect("a step has a row"),
                    tokens: std::num::NonZeroUsize::new(width).expect("a step has a token"),
                    capacity: self.options.capacity,
                    kv,
                    logits,
                },
                head: head.unwrap_or(Head::GREEDY),
                adapters,
            }
            .into()
        })
    }
}
