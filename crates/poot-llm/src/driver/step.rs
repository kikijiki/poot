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
//!
//! Under a draft policy ([`super::speculate`]) a decode row with nothing `ahead` proposes drafts and
//! is verified in a window step of the policy's width; the row's result is the accepted drafts and
//! one more token, and `commit(kept)` keeps a prefix of it like any other result.

use std::collections::HashMap;
use std::num::NonZeroU64;

use poot_graph_ir::ForwardSlotMap;
use poot_graph_ir::op::SampleRule;
use poot_models::model::{KvLayout, LogitRows, Phase, StepShape};
use poot_models::mrope::MropePositionIds;

use super::block_table::{BLOCK_SIZE, PagedKvCache, PoolExhausted};
use super::caps::{Layout, PoolShape, StateResidency};
use super::chunks::Piece;
use super::embeds::PromptEmbeds;
use super::error::{DriverError, EmbedsError, InvalidRequest, MropeError, PreparedSetRefusal};
use super::lora::{AdapterRef, GenerationId};
use super::mrope::rotary_position;
use super::prefix_cache::{PrefixIdentity, PrefixOutcome, PrefixSkip};
use super::speculate::{
    self, DraftPolicy, NotSpeculated, PassTiming, Proposal, RowSpeculation, SpecRow, StepTime,
};
use super::suffix::{self, Head, SuffixRows};
use super::{Driver, EntryKey, ModelRole, SlotRole, StepData, host_sample, window_wall};
use crate::core::sampler::{Sampler, TokenLogprob};

#[cfg(test)]
mod mtp_tests;

/// A driver-issued sequence id, never reused while the driver lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeqId(NonZeroU64);

impl SeqId {
    /// The id as an integer, for logs and wire formats.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

/// A sequence to admit.
#[derive(Clone, Copy, Debug)]
pub struct SeqRequest<'a> {
    /// The prompt's token ids.
    pub prompt: &'a [u32],
    /// The most tokens the caller will commit after the prompt; the blocks for them are reserved at
    /// admission.
    pub max_new: usize,
    /// The weights the sequence runs under: the base model or one adapter.
    pub adapter: AdapterRef,
}

/// The outcome of [`Driver::open`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// `reused` leading prompt tokens are already resident: prefill resumes at that position.
    Opened {
        /// The admitted sequence.
        seq: SeqId,
        /// Leading prompt tokens already resident through the prefix cache.
        reused: usize,
    },
    /// The pool cannot hold the request (`needed_blocks` more blocks), or every sequence slot is taken
    /// (`needed_blocks` is 0): the caller preempts a sequence or waits.
    NoRoom {
        /// Blocks short; 0 when every sequence slot is taken.
        needed_blocks: usize,
    },
}

/// What one row does this step.
#[derive(Clone, Copy, Debug)]
pub enum RowWork<'a> {
    /// Feed prompt tokens up to position `upto` (exclusive). The driver feeds the largest prepared
    /// piece of the remainder and reports how many tokens it consumed; the row picks only when this
    /// step reaches the end of the prompt.
    Prefill {
        /// Prompt position to feed up to, exclusive.
        upto: usize,
    },
    /// Feed the sequence's last committed token followed by `ahead` (verify-window drafts), and pick
    /// at every fed position: `ahead.len() + 1` tokens, the model's choice after each prefix, by
    /// `pick`. With nothing ahead it is a plain decode step and the row's own sampler picks; under a
    /// draft policy it is a speculative step, and the row returns the accepted drafts and one more
    /// (`1..=k+1` tokens).
    Decode {
        /// Verify-window draft tokens to feed after the last committed token; empty for a plain
        /// decode.
        ahead: &'a [u32],
        /// How each position of the window picks.
        pick: PickPolicy,
    },
}

/// How a verify window picks at each of its positions. Only the greedy argmax is accepted today; a
/// sampled window (speculative sampling's residual draws) is a further variant, so callers match
/// with a wildcard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PickPolicy {
    /// The argmax at every position.
    Greedy,
}

/// One row of a step: a sequence, its work and the sampler its pick goes through.
pub struct StepRow<'a> {
    /// The sequence the row steps.
    pub seq: SeqId,
    /// What the row does this step.
    pub work: RowWork<'a>,
    /// The sampler the row's pick goes through.
    pub sampler: &'a mut Sampler,
}

/// What a row produced.
#[derive(Debug)]
pub struct RowResult {
    /// The sequence the result belongs to.
    pub seq: SeqId,
    /// Prompt tokens a `Prefill` row consumed; zero for a `Decode` row.
    pub consumed: usize,
    /// The picks, uncommitted: empty for a prefill row that has not reached the end of its prompt,
    /// one for a decode row, one per fed position for a verify window, and for a speculative row
    /// the accepted drafts and the token after them.
    pub tokens: Vec<u32>,
    /// One record per token when the row's sampler records logprobs.
    pub logprobs: Vec<TokenLogprob>,
    /// What the draft policy did for this row: verified drafts, or why the row decoded one token.
    pub speculation: RowSpeculation,
}

/// Why a sequence is released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Release {
    /// The sequence is done and its state is dropped.
    Finished,
    /// The sequence will be recomputed. One opened with `open_with_embeds` must be reopened with it.
    Preempted,
}

/// One open sequence.
#[derive(Debug)]
pub(super) struct Seq {
    /// The sequence's slot in the paged cache: its block table.
    pub(super) slot: usize,
    /// The prompt, then the committed generated tokens.
    pub(super) tokens: Vec<u32>,
    prompt_len: usize,
    /// Prompt positions whose K/V is resident: reused at open, then advanced by prefill.
    done: usize,
    /// Positions the block table covers: the prompt and `max_new`.
    reserved: usize,
    pub(super) adapter: AdapterRef,
    identity: PrefixIdentity,
    seeded: bool,
    /// The last step's picks, until `commit` takes them.
    pub(super) pending: Vec<u32>,
    /// The tokens of `pending` a recurrent state already holds: a verified row's result, whose kept
    /// prefix the round folded (or replayed) into it. Zero for a row whose state folds nothing ahead.
    folded: usize,
    /// A commit kept fewer tokens than `folded`, so the recurrent state holds tokens the sequence
    /// dropped. Only a release, or a resume that resets the row, ends it.
    state_ahead: bool,
    /// Rows the prompt supplies in place of its tokens' embeddings, bound on every prefill step
    /// that covers one.
    embeds: Option<PromptEmbeds>,
    /// The sectioned rotary positions of the prompt, bound on every step; absent, every token
    /// rotates at its text position.
    mrope: Option<MropePositionIds>,
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
    pub(super) free_slots: Vec<usize>,
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
#[derive(Debug)]
pub(super) struct RowPlan {
    pub(super) feed: Vec<u32>,
    pub(super) start: usize,
    /// Picks the row produces: none, one, or one per fed position of a verify window.
    pub(super) picks: usize,
    /// A prefill piece (as opposed to a decode token).
    pub(super) prefill: bool,
    /// The piece covers a position the sequence's embedding rows replace: the step runs the
    /// `PrefillEmbeds` entry.
    pub(super) embeds: bool,
    /// Set for a decode row under a draft policy: what the policy proposed for it.
    pub(super) spec: Option<SpecRow>,
    /// Set for a decode row under a draft policy that cannot speculate on it: the sampler is one
    /// the window does not verify.
    pub(super) not_speculated: Option<NotSpeculated>,
}

impl RowPlan {
    /// What the policy did for the row, given the `kept` tokens its step returned.
    pub(super) fn speculation(&self, kept: usize) -> RowSpeculation {
        if let Some(reason) = self.not_speculated {
            return RowSpeculation::NotSpeculated(reason);
        }
        let Some(spec) = &self.spec else {
            return RowSpeculation::NotApplicable;
        };
        match spec.proposal.drafts.len() {
            0 if spec.room == 0 => RowSpeculation::NotSpeculated(NotSpeculated::NoRoom),
            0 => RowSpeculation::NotSpeculated(NotSpeculated::NoProposal),
            proposed => RowSpeculation::Verified {
                proposed,
                accepted: kept - 1,
            },
        }
    }

    /// The drafts the row proposed, none for a row that does not speculate.
    pub(super) fn drafts(&self) -> &[u32] {
        self.spec
            .as_ref()
            .map_or(&[], |spec| spec.proposal.drafts.as_slice())
    }

    /// Whether the row's tokens come from speculative sampling over the window's logits: drafts to
    /// verify and a sampler that is not greedy.
    pub(super) fn verifies_sampled(&self, sampler: &Sampler) -> bool {
        !self.drafts().is_empty() && !sampler.is_greedy()
    }
}

/// A live row of an executed entry: its sequence, and the sampler a suffix input or host pick reads.
pub(super) struct Lane<'a> {
    pub(super) seq: SeqId,
    pub(super) sampler: &'a mut Sampler,
}

/// What an executed entry returned for its live rows.
#[derive(Debug)]
pub(super) enum Readback {
    /// The sampling suffix's tokens: per live row, one per pick.
    Tokens(Vec<Vec<u32>>),
    /// The logits head: `rows` holds `[entry rows, positions, vocab]` row-major.
    Logits {
        rows: Vec<f32>,
        vocab: usize,
        positions: usize,
    },
}

impl Readback {
    /// One readback in row order from the readbacks of `groups` of rows (`groups[g]` lists the rows
    /// readback `g` covers, in the order it holds them).
    fn merge(readbacks: Vec<Readback>, groups: &[Vec<usize>]) -> Readback {
        let mut readbacks = readbacks;
        if readbacks.len() == 1 {
            return readbacks.remove(0);
        }
        let rows = groups.iter().map(Vec::len).sum::<usize>();
        match readbacks.first() {
            Some(Readback::Tokens(_)) => {
                let mut tokens = vec![Vec::new(); rows];
                for (readback, group) in readbacks.into_iter().zip(groups) {
                    let Readback::Tokens(group_tokens) = readback else {
                        unreachable!("every group of a step runs the same head")
                    };
                    for (&row, picked) in group.iter().zip(group_tokens) {
                        tokens[row] = picked;
                    }
                }
                Readback::Tokens(tokens)
            }
            Some(Readback::Logits { vocab, .. }) => {
                let vocab = *vocab;
                // A row's picks are its last positions, so a group that read fewer positions than
                // the widest is right-aligned in the merged stride; the positions in front hold
                // nothing a pick reads.
                let positions = readbacks
                    .iter()
                    .map(|readback| match readback {
                        Readback::Logits { positions, .. } => *positions,
                        Readback::Tokens(_) => unreachable!("every group of a step runs one head"),
                    })
                    .max()
                    .expect("a step runs at least one group");
                let stride = positions * vocab;
                let mut merged = vec![0.0; rows * stride];
                for (readback, group) in readbacks.into_iter().zip(groups) {
                    let Readback::Logits {
                        rows: group_rows,
                        vocab: group_vocab,
                        positions: group_positions,
                    } = readback
                    else {
                        unreachable!("every group of a step runs the same head")
                    };
                    assert_eq!(
                        group_vocab, vocab,
                        "the groups of a step share a vocabulary"
                    );
                    let group_stride = group_positions * vocab;
                    for (k, &row) in group.iter().enumerate() {
                        let end = (row + 1) * stride;
                        merged[end - group_stride..end]
                            .copy_from_slice(&group_rows[k * group_stride..(k + 1) * group_stride]);
                    }
                }
                Readback::Logits {
                    rows: merged,
                    vocab,
                    positions,
                }
            }
            None => unreachable!("a step runs at least one group"),
        }
    }
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
        self.open_seq(request, None, None)
    }

    fn open_seq(
        &mut self,
        request: SeqRequest<'_>,
        embeds: Option<PromptEmbeds>,
        mrope: Option<MropePositionIds>,
    ) -> Result<Admission, DriverError> {
        let SeqRequest {
            prompt,
            max_new,
            adapter,
        } = request;
        let reserved = self.check_admission(prompt, max_new)?;
        let residency = self.residency;
        let serving = self.serving.as_mut().ok_or(InvalidRequest::NotPrepared)?;
        let Some(&slot) = serving.free_slots.last() else {
            return Ok(Admission::NoRoom { needed_blocks: 0 });
        };
        let private = matches!(serving.kv, KvStore::Contiguous);
        let identity = self.lora.retain(adapter)?;
        let reused = match &mut serving.kv {
            KvStore::Contiguous => 0,
            KvStore::Paged { cache, .. } => {
                // Embedding rows change what the same tokens mean: such a prompt reuses no cached prefix.
                let admit = |cache: &mut PagedKvCache| {
                    if embeds.is_some() {
                        cache
                            .admit_unshared(slot, prompt.len(), max_new)
                            .map(|()| 0)
                    } else {
                        cache.admit_prefix_for(slot, prompt, max_new, identity)
                    }
                };
                let mut admitted = admit(cache);
                if admitted.is_err() {
                    cache.evict_prefix();
                    admitted = admit(cache);
                }
                match admitted {
                    Ok(reused) => reused,
                    Err(PoolExhausted) => {
                        self.lora.release(adapter);
                        let needed = reserved.div_ceil(BLOCK_SIZE);
                        return Ok(Admission::NoRoom {
                            needed_blocks: needed.saturating_sub(cache.free_blocks()).max(1),
                        });
                    }
                }
            }
        };
        if private || residency == StateResidency::Recurrent {
            // The private cache holds one sequence's row and a recurrent state pool one row per
            // slot: a new sequence starts from zero in its own slot's row and leaves every other
            // sequence's state alone. It runs once admission and the lease are settled and before
            // the slot and sequence are taken; a failure gives the lease and the blocks back.
            if let Err(error) = self.reset_row_state(slot) {
                self.lora.release(adapter);
                if let Some(KvStore::Paged { cache, .. }) =
                    self.serving.as_mut().map(|serving| &mut serving.kv)
                {
                    cache.reset(slot);
                }
                return Err(error);
            }
        }
        let serving = self.serving.as_mut().expect("checked above");
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
                reserved,
                adapter,
                identity,
                seeded: false,
                pending: Vec::new(),
                folded: 0,
                state_ahead: false,
                embeds,
                mrope,
            },
        );
        self.speculation.open(seq, reused);
        Ok(Admission::Opened { seq, reused })
    }

    /// Zero slot `slot`'s row of every executable's state.
    fn reset_row_state(&mut self, slot: usize) -> Result<(), DriverError> {
        let scope = poot_executor::StateScope::Row(slot);
        self.executor.reset_state(self.exe, scope.clone())?;
        if let Some(draft) = &self.speculation.draft {
            self.executor.reset_state(draft.exe, scope.clone())?;
        }
        for &exe in self.adapter_exes.values() {
            self.executor.reset_state(exe, scope.clone())?;
        }
        Ok(())
    }

    /// Continue the open, fully committed `seq` with `prompt`, the whole next context: the tokens it
    /// already holds, then the new ones. The longest leading run of `prompt` that the sequence's K/V
    /// already covers is kept and the rest is prefilled, so a conversation's next turn pays for its
    /// new tokens only. Returns the positions kept, which leave at least the prompt's last token to
    /// prefill (its pick is the turn's first token). Room for `max_new` more tokens is reserved.
    ///
    /// A recurrent state cannot be rewound to a shorter context: unless it all lies inside `prompt`
    /// the row restarts from zero and the whole prompt is prefilled. Only the contiguous layout is
    /// resumed in place; a paged pool shares turns through its prefix cache ([`Driver::open`]).
    pub fn resume(
        &mut self,
        seq: SeqId,
        prompt: &[u32],
        max_new: usize,
    ) -> Result<usize, DriverError> {
        let reserved = self.check_admission(prompt, max_new)?;
        let residency = self.residency;
        let serving = self.serving.as_mut().ok_or(InvalidRequest::NotPrepared)?;
        if !matches!(serving.kv, KvStore::Contiguous) {
            return Err(InvalidRequest::ResumePaged.into());
        }
        let state = serving
            .seqs
            .get_mut(&seq)
            .ok_or(InvalidRequest::UnknownSeq(seq))?;
        if !state.pending.is_empty() {
            return Err(InvalidRequest::UncommittedResult(seq).into());
        }
        if state.embeds.is_some() || state.mrope.is_some() {
            return Err(InvalidRequest::ResumeEmbedded(seq).into());
        }
        // A decode step writes the K/V of the token it feeds, so the last committed token has none.
        let resident = state.done.max(state.tokens.len() - 1);
        let shared = state.tokens[..resident]
            .iter()
            .zip(prompt)
            .take_while(|(held, new)| held == new)
            .count();
        let mut reused = shared.min(prompt.len() - 1);
        // The MTP layer's cache at the last shared position was computed from the token after it,
        // which the new prompt may not share: that position runs again, so the layer rewrites it
        // from the target's hidden state (the private cache is this sequence's own to rewrite). A
        // recurrent state has folded that position already and cannot step back, so it restarts
        // from zero below.
        if matches!(self.speculation.policy, DraftPolicy::Mtp { .. }) {
            reused = reused.saturating_sub(1);
        }
        if residency == StateResidency::Recurrent && (reused != resident || state.state_ahead) {
            reused = 0;
        }
        let slot = state.slot;
        if reused == 0 {
            // The row restarts from zero: reset it before the sequence is rewritten, so a failed
            // reset leaves the sequence as it was.
            self.reset_row_state(slot)?;
        }
        let state = self
            .serving
            .as_mut()
            .and_then(|serving| serving.seqs.get_mut(&seq))
            .expect("looked up above");
        state.tokens = prompt.to_vec();
        state.prompt_len = prompt.len();
        state.done = reused;
        state.reserved = reserved;
        state.seeded = false;
        if reused == 0 {
            state.state_ahead = false;
        }
        self.speculation.rewind(seq, reused);
        Ok(reused)
    }

    /// [`Driver::open`] for a prompt that supplies embedding rows in place of some tokens'
    /// embeddings (`embeds`). Its prefill steps run the `PrefillEmbeds` entries
    /// ([`Driver::prepare_embeds`]); decode steps are ordinary. Under the paged layout the prompt
    /// reuses no cached prefix and registers none: the prefix cache keys on tokens alone, which these
    /// rows change the meaning of. An adapter sequence is refused (only the base weights have
    /// `PrefillEmbeds` entries), and so is any sequence under a draft-model policy (the draft prefills
    /// over placeholder tokens); prompt lookup speculates on the tokens and is fine. A preempted
    /// embeds sequence is recomputed: reopen it with this method, not [`Driver::open`], or its
    /// prefill would run over the placeholders.
    pub fn open_with_embeds(
        &mut self,
        request: SeqRequest<'_>,
        embeds: PromptEmbeds,
    ) -> Result<Admission, DriverError> {
        self.open_embedded(request, embeds, None)
    }

    /// [`Driver::open_with_embeds`] for a model that rotates at sectioned positions
    /// (`Slot::MropePosition`, Qwen2.5-VL's text tower): the prompt's image tokens do not sit at their
    /// index, so `positions` (the processor's, one entry per prompt token) says where every token of
    /// the prompt rotates, and generated tokens continue from where it ends. A preempted sequence is
    /// reopened with the same positions. A model without sectioned positions refuses the sequence here.
    pub fn open_with_mrope(
        &mut self,
        request: SeqRequest<'_>,
        embeds: PromptEmbeds,
        positions: MropePositionIds,
    ) -> Result<Admission, DriverError> {
        if positions.len() != request.prompt.len() {
            return Err(InvalidRequest::Mrope(MropeError::Length {
                positions: positions.len(),
                prompt: request.prompt.len(),
            })
            .into());
        }
        self.open_embedded(request, embeds, Some(positions))
    }

    fn open_embedded(
        &mut self,
        request: SeqRequest<'_>,
        embeds: PromptEmbeds,
        mrope: Option<MropePositionIds>,
    ) -> Result<Admission, DriverError> {
        if self.serving.is_none() {
            return Err(InvalidRequest::NotPrepared.into());
        }
        if request.adapter != AdapterRef::Base {
            return Err(InvalidRequest::Embeds(EmbedsError::Adapter).into());
        }
        if matches!(self.speculation.policy, DraftPolicy::DraftModel { .. }) {
            return Err(InvalidRequest::Embeds(EmbedsError::DraftModel).into());
        }
        if matches!(self.speculation.policy, DraftPolicy::Mtp { .. }) {
            return Err(InvalidRequest::Embeds(EmbedsError::Mtp).into());
        }
        embeds
            .check_prompt(request.prompt.len())
            .map_err(InvalidRequest::Embeds)?;
        // The prepared entries say whether the model takes sectioned positions: refuse before the
        // sequence holds a slot or blocks.
        if mrope.is_some()
            && !self.entries.iter().any(|(key, entry)| {
                key.role == ModelRole::Target
                    && entry
                        .slots
                        .iter()
                        .any(|spec| spec.role == SlotRole::MropePosition)
            })
        {
            return Err(InvalidRequest::Mrope(MropeError::NotSectioned).into());
        }
        self.open_seq(request, Some(embeds), mrope)
    }

    /// Run one step over `rows`. Each row's sampler picks its token: the suffix entry when every
    /// picking row shares a suffix-expressible rule, otherwise the logits entry and the host path.
    /// The picks are not part of any sequence until [`Driver::commit`].
    pub fn step(&mut self, rows: &mut [StepRow<'_>]) -> Result<Vec<RowResult>, DriverError> {
        if self.serving.is_none() {
            return Err(InvalidRequest::NotPrepared.into());
        }
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
        let mut plans = self.plan_rows(rows)?;
        self.propose(rows, &mut plans)?;
        let serving = self.serving.as_ref().expect("checked above");
        for row in rows.iter_mut() {
            let seq = &serving.seqs[&row.seq];
            if !seq.seeded {
                row.sampler.seed_context(&seq.tokens[..seq.prompt_len]);
            }
        }
        // A generation's entries run on their own executable, whose KV state holds only the sequences
        // opened under that generation: one step cannot mix generations, nor base rows with adapter rows.
        let mut generations = rows
            .iter()
            .map(|row| serving.seqs[&row.seq].adapter.generation());
        let adapter = generations.next().expect("a step has a row");
        if generations.any(|generation| generation != adapter) {
            return Err(InvalidRequest::MixedAdapters.into());
        }
        let window = plans.iter().any(|plan| plan.picks > 1);
        let head = self.step_head(rows, &plans, window)?;
        let lanes: Vec<Lane<'_>> = rows
            .iter_mut()
            .map(|row| Lane {
                seq: row.seq,
                sampler: &mut *row.sampler,
            })
            .collect();
        let readback = self.run_rows(lanes, &mut plans, (head, adapter))?;

        let mut picked: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        let mut host_picks = false;
        match &readback {
            Readback::Tokens(tokens) => picked.clone_from(tokens),
            Readback::Logits {
                rows: logits,
                vocab,
                positions,
            } => {
                host_picks = true;
                for (i, (row, plan)) in rows.iter_mut().zip(&plans).enumerate() {
                    // A sampled row with drafts draws through speculative sampling instead, below.
                    let tokens = if plan.verifies_sampled(row.sampler) {
                        Vec::new()
                    } else {
                        // A row's picks are its last `picks` positions, the one position under `Last`.
                        let first = (i * positions + positions - plan.picks) * vocab;
                        (0..plan.picks)
                            .map(|j| {
                                let at = first + j * vocab;
                                host_sample::pick(row.sampler, &logits[at..at + vocab])
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    };
                    picked.push(tokens);
                }
            }
        }
        self.stats.host_picks += u64::from(host_picks);
        // A speculative row keeps the drafts that verify and the pick after them.
        for (i, (row, plan)) in rows.iter_mut().zip(&plans).enumerate() {
            if let Some(tokens) =
                speculate::verify_row(row.sampler, plan, i, &picked[i], &readback)?
            {
                picked[i] = tokens;
            }
        }
        self.mirror_prefill(rows, &plans)?;

        let recurrent = self.residency == StateResidency::Recurrent;
        let serving = self.serving.as_mut().expect("checked above");
        let mut results = Vec::with_capacity(rows.len());
        let mut settled = Vec::new();
        for ((row, plan), tokens) in rows.iter_mut().zip(&plans).zip(picked) {
            let seq = serving.seqs.get_mut(&row.seq).expect("planned above");
            let consumed = if plan.prefill { plan.feed.len() } else { 0 };
            seq.done += consumed;
            seq.seeded = true;
            seq.pending.clone_from(&tokens);
            seq.folded = if recurrent && !plan.drafts().is_empty() {
                tokens.len()
            } else {
                0
            };
            if !plan.drafts().is_empty() {
                settled.push((row.seq, seq.tokens.len(), plan.drafts().len(), tokens.len()));
            }
            let speculation = plan.speculation(tokens.len());
            self.stats.speculation.record(speculation);
            if let RowSpeculation::Verified { proposed, accepted } = speculation {
                self.stats.positions.record(proposed, accepted);
            }
            results.push(RowResult {
                seq: row.seq,
                consumed,
                speculation,
                logprobs: if tokens.is_empty() {
                    Vec::new()
                } else {
                    row.sampler.take_logprobs()
                },
                tokens,
            });
        }
        for &(seq, committed, proposed, kept) in &settled {
            self.settle(seq, committed, proposed, kept);
        }
        self.roll_back(&settled)?;
        self.mtp_catch_up(rows, &plans)?;
        Ok(results)
    }

    /// Run `lanes` once per feed width and merge what the runs return into one readback in lane order.
    /// A row feeding fewer tokens than the step's widest is left-padded, which is harmless to attention
    /// (the padding is masked) and fatal to recurrent state (the padding would fold into it): a
    /// recurrent model's rows are grouped by the tokens they feed and each group runs its own entry,
    /// reading every position only when one of its rows verifies a window. Every other model runs all
    /// its rows as one.
    fn run_rows(
        &mut self,
        lanes: Vec<Lane<'_>>,
        plans: &mut Vec<RowPlan>,
        (head, adapter): (Option<Head>, Option<GenerationId>),
    ) -> Result<Readback, DriverError> {
        let grouped = self.groups_rows_by_width();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (i, plan) in plans.iter().enumerate() {
            let same_width =
                |group: &Vec<usize>| !grouped || plans[group[0]].feed.len() == plan.feed.len();
            match groups.iter_mut().find(|group| same_width(group)) {
                Some(group) => group.push(i),
                None => groups.push(vec![i]),
            }
        }
        let mut parts: Vec<Option<(Lane<'_>, RowPlan)>> =
            lanes.into_iter().zip(plans.drain(..)).map(Some).collect();
        let mut readbacks = Vec::with_capacity(groups.len());
        let mut failure = None;
        for group in &groups {
            let (mut group_lanes, group_plans): (Vec<_>, Vec<_>) = group
                .iter()
                .map(|&i| parts[i].take().expect("each row is in one group"))
                .unzip();
            let tokens = group_plans
                .iter()
                .map(|plan| plan.feed.len())
                .max()
                .unwrap_or(1);
            let embeds = group_plans.iter().any(|plan| plan.embeds);
            let window = group_plans.iter().any(|plan| plan.picks > 1);
            let verifies = group_plans.iter().any(|plan| !plan.drafts().is_empty());
            let ran = self
                .step_entry(
                    ModelRole::Target,
                    group.len(),
                    tokens,
                    (window, embeds),
                    head,
                    adapter,
                )
                .and_then(|key| {
                    self.timed(
                        verifies,
                        |passes| &mut passes.verifies,
                        |driver| driver.run_plans(&mut group_lanes, &group_plans, key),
                    )
                });
            for (&i, part) in group.iter().zip(group_lanes.into_iter().zip(group_plans)) {
                parts[i] = Some(part);
            }
            match ran {
                Ok(readback) => readbacks.push(readback),
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        plans.extend(
            parts
                .into_iter()
                .map(|part| part.expect("restored above").1),
        );
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(Readback::merge(readbacks, &groups))
    }

    /// Run one prepared entry over `plans`: bind each live row's feed, positions and cache mapping,
    /// execute, and read back what the rows pick. `lanes` are the live rows in plan order; rows beyond
    /// them are masked. The caller owns the sequences' bookkeeping.
    pub(super) fn run_plans(
        &mut self,
        lanes: &mut [Lane<'_>],
        plans: &[RowPlan],
        key: EntryKey,
    ) -> Result<Readback, DriverError> {
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
            lora: vec![0; entry_rows],
            ..StepData::default()
        };
        // The forward write maps, one entry per flat token (KV) and per step row (recurrent state): -1
        // writes nothing, so padding and masked rows leave the pool alone.
        let mut kv_write = Vec::new();
        if let KvLayout::Paged { .. } = kv {
            data.read = vec![0; entry_rows * capacity];
            kv_write = vec![ForwardSlotMap::PADDING; entry_rows * width];
        }
        // A recurrent state pool: masked rows read pool row 0 and write nothing.
        let pools_state = entry
            .slots
            .iter()
            .any(|spec| spec.role == SlotRole::StateWrite);
        let mut state_write = Vec::new();
        if pools_state {
            data.state_read = vec![0; entry_rows];
            state_write = vec![ForwardSlotMap::PADDING; entry_rows];
        }
        let hidden = if key.phase == Phase::PrefillEmbeds {
            let spec = entry
                .slots
                .iter()
                .find(|spec| spec.role == SlotRole::Embeds)
                .expect("a PrefillEmbeds program declares its embedding rows");
            data.embeds = vec![0.0; spec.shape.iter().product()];
            data.embeds_at = vec![0; entry_rows * width];
            spec.shape[2]
        } else {
            0
        };
        let serving = self.serving.as_ref().expect("a step is served");
        let supplies = lanes
            .iter()
            .any(|lane| serving.seqs[&lane.seq].mrope.is_some());
        // Axis-major `[3, entry_rows * width]`; a row without positions of its own, and the padding,
        // rotate at their text positions like `Slot::Pos`.
        let mut mrope = supplies.then(|| vec![0; 3 * entry_rows * width]);
        for (i, (lane, plan)) in lanes.iter().zip(plans).enumerate() {
            let seq = &serving.seqs[&lane.seq];
            let n = plan.feed.len();
            if let Some(embeds) = seq.embeds.as_ref().filter(|_| plan.embeds) {
                if embeds.width() != hidden {
                    return Err(InvalidRequest::Embeds(EmbedsError::Width {
                        got: embeds.width(),
                        want: hidden,
                    })
                    .into());
                }
                for j in 0..n {
                    // Left padding: the piece's tokens are the last `n` columns of the step.
                    if let Some(row) = embeds.row_at(plan.start + j) {
                        let column = i * width + (width - n) + j;
                        data.embeds[column * hidden..(column + 1) * hidden].copy_from_slice(row);
                        data.embeds_at[column] = 1;
                    }
                }
            }
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
            if pools_state {
                data.state_read[i] = seq.slot as i32;
                state_write[i] = seq.slot as i32;
            }
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
                        if let Some(mrope) = &mut mrope {
                            let axes = seq
                                .mrope
                                .as_ref()
                                .map_or([position as i32; 3], |ids| rotary_position(ids, position));
                            for (axis, value) in axes.into_iter().enumerate() {
                                mrope[axis * entry_rows * width + at] = value;
                            }
                        }
                        if let Some(map) = &map
                            && position < seq.reserved
                        {
                            kv_write[at] = map[position] as i32;
                        }
                    }
                    None => {
                        data.pos[at] = plan.start as i32;
                        if let Some(mrope) = &mut mrope {
                            let axes = seq.mrope.as_ref().map_or([plan.start as i32; 3], |ids| {
                                rotary_position(ids, plan.start)
                            });
                            for (axis, value) in axes.into_iter().enumerate() {
                                mrope[axis * entry_rows * width + at] = value;
                            }
                        }
                    }
                }
            }
        }
        data.mrope = mrope;
        // Each pool slot has one owner: a block table that handed two tokens one slot is refused here.
        if let KvLayout::Paged { pool_slots, seqs } = kv {
            data.write = ForwardSlotMap::new(kv_write, pool_slots.get())?;
            if pools_state {
                data.state_write = ForwardSlotMap::new(state_write, seqs.get())?;
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
            for (lane, plan) in lanes.iter_mut().zip(plans) {
                let suffix_row = if plan.picks > 0 {
                    SuffixRows::push(lane.sampler, rule)
                } else {
                    SuffixRows::placeholder(lane.sampler, rule)
                };
                seed.push(suffix_row.seed);
                params.extend(suffix_row.params);
                top_k.extend(suffix_row.top_k);
            }
            // Masked rows repeat the first row's placeholder.
            let per_params = params.len() / lanes.len();
            let masked = entry_rows - lanes.len();
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
        // Nothing reads the output of a step no row picks in: no readback.
        if plans.iter().all(|plan| plan.picks == 0) {
            return Ok(Readback::Tokens(vec![Vec::new(); lanes.len()]));
        }
        match key.head {
            Head::Sample(_) => {
                let bytes = outputs.read()?;
                let ints: &[i32] = bytemuck::cast_slice(&bytes);
                let mut picked = Vec::with_capacity(lanes.len());
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
                Ok(Readback::Tokens(picked))
            }
            Head::Logits => {
                let rows = host_sample::read_logits(&mut outputs)?;
                let vocab = rows.len() / (entry_rows * positions);
                Ok(Readback::Logits {
                    rows,
                    vocab,
                    positions,
                })
            }
        }
    }

    /// Run `pass`, and on a phase-timed driver add the engine time it took to the tally `account`
    /// names when `counts` holds. A pass that is not counted still runs untimed.
    pub(super) fn timed<T>(
        &mut self,
        counts: bool,
        account: fn(&mut PassTiming) -> &mut StepTime,
        pass: impl FnOnce(&mut Self) -> Result<T, DriverError>,
    ) -> Result<T, DriverError> {
        let started = (self.time_phases && counts).then(|| self.executor.stats().timing.mark());
        let result = pass(self)?;
        if let Some(started) = started {
            let wall = window_wall(&self.executor.stats().timing, &started);
            let tally = account(&mut self.stats.passes);
            tally.steps += 1;
            tally.wall += wall;
        }
        Ok(result)
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
        // A recurrent state cannot step back from the tokens a verified row folded: keeping fewer
        // leaves it ahead of the sequence.
        state.state_ahead |= kept < std::mem::take(&mut state.folded);
        for &token in &pending[..kept] {
            sampler.observe(token);
        }
        state.tokens.extend_from_slice(&pending[..kept]);
        Ok(())
    }

    /// Close `seq` and free its blocks. A finished sequence registers its filled prompt blocks in the
    /// prefix cache under its adapter identity, unless the model's state is recurrent (a cached block
    /// holds no recurrent state, so a later request reusing it would run without it) or the driver
    /// speculates with an MTP layer (its cache at a block's last position depends on the token after
    /// the block, and a shared block cannot be rewritten).
    pub fn release(&mut self, seq: SeqId, mode: Release) -> Result<PrefixOutcome, DriverError> {
        let residency = self.residency;
        let mtp = matches!(self.speculation.policy, DraftPolicy::Mtp { .. });
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
                } else if mtp {
                    cache.reset(state.slot);
                    PrefixOutcome::Skipped(PrefixSkip::MtpState)
                } else if state.embeds.is_some() {
                    cache.reset(state.slot);
                    PrefixOutcome::Skipped(PrefixSkip::PromptEmbeds)
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
        self.speculation.close(seq);
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
            if seq.state_ahead {
                return Err(InvalidRequest::RecurrentStateAhead(row.seq).into());
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
                        embeds: seq
                            .embeds
                            .as_ref()
                            .is_some_and(|embeds| embeds.covers(seq.done, seq.done + n)),
                        spec: None,
                        not_speculated: None,
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
                    // Under a draft policy a decode row with nothing ahead speculates, when its
                    // sampler is one the window verifies; any other row decodes one token.
                    let (spec, not_speculated) = match self.speculation.policy.caps() {
                        Some(_) if !ahead.is_empty() => (None, None),
                        Some(_) if !speculate::speculation_sampler_ok(row.sampler) => {
                            (None, Some(NotSpeculated::Sampler))
                        }
                        Some(caps) => {
                            // The window writes `window` consecutive positions: a sequence too
                            // near its capacity to hold them decodes one token.
                            let room = if start + caps.window.get() <= capacity {
                                seq.reserved.saturating_sub(seq.tokens.len() + 1)
                            } else {
                                0
                            };
                            let spec = SpecRow {
                                room: room.min(caps.k.get()),
                                proposal: Proposal::default(),
                            };
                            (Some(spec), None)
                        }
                        None => (None, None),
                    };
                    RowPlan {
                        picks: feed.len(),
                        feed,
                        start,
                        prefill: false,
                        embeds: false,
                        spec,
                        not_speculated,
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
                None => Ok(Some(Head::GREEDY)),
                // Under a draft policy a sampled row beside a window picks on the host, from the
                // window entry's logits.
                Some(_) if self.speculation.policy.caps().is_some() => Ok(Some(Head::Logits)),
                Some(seq) => Err(InvalidRequest::WindowStepNeedsGreedy(seq)),
            };
        }
        let mut rules = rows
            .iter()
            .zip(plans)
            .filter(|(_, plan)| plan.picks > 0)
            .map(|(row, _)| suffix::rule_of(row.sampler));
        // No row picks: the output is unread, so any prepared head of the shape serves. A rule the
        // prepared set has no suffix entry for picks on the host from the logits head, as `HeadSet` says.
        let prepared = |rule| {
            self.caps
                .as_ref()
                .is_some_and(|caps| caps.heads.suffix.contains(&rule))
        };
        Ok(rules.next().map(|first| match first {
            Some(rule) if prepared(rule) && rules.all(|other| other == Some(rule)) => {
                Head::Sample(rule)
            }
            _ => Head::Logits,
        }))
    }

    /// The prepared entry a step runs: the smallest admitted row count holding `live` rows, at the
    /// step's width, with `head` (`None`: any head, the greedy suffix first, then the logits head). A
    /// shape `prepare` never admitted is a typed refusal; a step never compiles.
    pub(super) fn step_entry(
        &self,
        role: ModelRole,
        live: usize,
        width: usize,
        (window, embeds): (bool, bool),
        head: Option<Head>,
        adapters: Option<super::lora::GenerationId>,
    ) -> Result<EntryKey, DriverError> {
        let serving = self.serving.as_ref().expect("checked by the caller");
        let logits = if window {
            LogitRows::All
        } else {
            LogitRows::Last
        };
        let phase = if role == ModelRole::Mtp {
            Phase::Mtp
        } else if embeds {
            Phase::PrefillEmbeds
        } else {
            serving_phase(width, logits)
        };
        let kv = serving.kv();
        let mut candidates: Vec<EntryKey> = self
            .entries
            .keys()
            .filter(|key| {
                key.role == role
                    && key.phase == phase
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
                    hidden: self.hidden_of(role),
                },
                head: head.unwrap_or(Head::GREEDY),
                adapters,
            }
            .into()
        })
    }
}
