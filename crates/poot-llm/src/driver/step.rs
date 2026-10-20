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
//! entry's last), so a changing composition never compiles. A step runs the narrowest prepared sampling
//! head that serves every picking row (a draw of the widest rule among them, penalties, bias and a guided
//! mask when any row has them, logprobs when any row records them); the other rows bind the neutral value
//! of what they do not use. A request no prepared head serves is a typed refusal. Several decode steps can
//! be chained on the device with one host wait ([`RowWork::Chain`]).
//!
//! Under a draft policy ([`super::speculate`]) a decode row with nothing `ahead` proposes drafts and
//! is verified in a window step of the policy's width; the row's result is the accepted drafts and
//! one more token, and `commit(kept)` keeps a prefix of it like any other result.

use std::num::{NonZeroU64, NonZeroUsize};

use poot_graph_ir::ForwardSlotMap;
use poot_graph_ir::op::SampleRule;
use poot_graph_ir::ops::sampling::{Adjust, HISTORY_STATE, SampleSpec};
use poot_models::io::{InputError, InputSet, InputValue, ProgramId, RequestInputs, StepInput};
use poot_models::model::{KvLayout, LogitRows, StepShape, TraceError};

use super::bind::{Binder, PassInput, RequestFrame, RowFeed, Side, StepColumns};
use super::block_table::{BLOCK_SIZE, PagedKvCache, PoolExhausted};
use super::caps::StateResidency;
use super::chunks::Piece;
use super::error::{DriverError, InvalidRequest, PreparedSetRefusal, Unsupported};
use super::lora::{AdapterRef, GenerationId};
use super::prefix_cache::{PrefixOutcome, PrefixSkip};
use super::sequences::{KvStore, Seq, Serving};
use super::speculate::{
    self, DraftPolicy, NotSpeculated, PassTiming, Proposal, RowSpeculation, SpecRow, StepTime,
};
use super::suffix::{self, DrawRow, Head, NEUTRAL_PENALTY};
use super::{
    Bound, Driver, EntryKey, Feature, FeatureSet, ModelRole, Program, host_sample, window_wall,
};
use crate::core::sampler::{Sampler, TokenLogprob};
use poot_executor::{ChainFeed, ChainStep, ExecError, LateInputs, NoLate, NoSync, StepInputs};
use poot_graph_ir::SlotKey;
use poot_tensor::{DType, HostTensor, HostView};

#[cfg(test)]
pub(super) mod mtp_tests;

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
#[derive(Clone, Debug)]
pub struct SeqRequest<'a> {
    /// The prompt's token ids.
    pub prompt: &'a [u32],
    /// The most tokens the caller will commit after the prompt; the blocks for them are reserved at
    /// admission.
    pub max_new: usize,
    /// The weights the sequence runs under: the base model or one adapter.
    pub adapter: AdapterRef,
    /// The values the request supplies for inputs its model's `step` program reads (a multimodal
    /// prompt's embedding rows, sectioned positions); empty for a plain token prompt. Checked against
    /// the model's schema at admission, owned by the sequence, rebound on every step that carries them
    /// and again if the sequence is recomputed.
    pub inputs: RequestInputs,
}

impl<'a> SeqRequest<'a> {
    /// The features of a request: what it asks of the driver beyond a token prompt, computed once at
    /// admission and composed with the driver's own.
    fn features_of(adapter: AdapterRef, inputs: &RequestInputs) -> FeatureSet {
        FeatureSet::NONE
            .with_if(adapter.generation().is_some(), Feature::Adapters)
            .with_if(!inputs.is_empty(), Feature::RequestInputs)
    }

    /// A plain token prompt: no request inputs.
    pub fn new(prompt: &'a [u32], max_new: usize, adapter: AdapterRef) -> Self {
        Self {
            prompt,
            max_new,
            adapter,
            inputs: RequestInputs::none(),
        }
    }

    /// This request supplying `inputs` beside its prompt.
    #[must_use]
    pub fn with_inputs(mut self, inputs: RequestInputs) -> Self {
        self.inputs = inputs;
        self
    }
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
    /// at every fed position: `ahead.len() + 1` tokens, the model's greedy choice after each prefix.
    /// With nothing ahead it is a plain decode step and the row's own sampler picks; under a draft
    /// policy it is a speculative step, and the row returns the accepted drafts and one more
    /// (`1..=k+1` tokens).
    Decode {
        /// Verify-window draft tokens to feed after the last committed token; empty for a plain
        /// decode.
        ahead: &'a [u32],
    },
    /// Decode `steps` tokens one after another on the device with one host wait: each step's pick is
    /// fed to the next on the device, so the host sees none of them until the last has run. The row
    /// returns `steps` tokens, uncommitted like any result; a stop inside them is a `commit` that keeps
    /// fewer. Every picking row of the step chains the same number of steps, with a plain or adjusted
    /// head (no guided mask and no logprobs, whose per-step host work cannot ride a device chain), and
    /// no row of the step is a prefill or a window.
    Chain {
        /// The tokens to decode, at least two.
        steps: NonZeroUsize,
    },
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
    /// The sequence will be recomputed: it registers no prefix. Its request inputs travel with the
    /// request, so the caller reopens it through [`Driver::open`] like any other.
    Preempted,
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

/// The guided-decoding mask of a step's picking lanes, as the late input of an [`Adjust::Guided`] head:
/// `0` for a permitted token and [`MASKED_LOGIT`](poot_graph_ir::ops::sampling::MASKED_LOGIT) for an
/// excluded one, the neutral zero row for every lane that does not pick. The executor asks for it once
/// the device is running the forward pass, so the host work of walking each constraint overlaps it.
struct GuidedMask<'l, 'a> {
    lanes: &'l [Lane<'a>],
    plans: &'l [RowPlan],
    rows: usize,
    vocab: usize,
    keys: [SlotKey; 1],
}

impl<'l, 'a> GuidedMask<'l, 'a> {
    /// The late input of a step run on `head`, `None` when the head has no mask.
    fn of(
        head: Head,
        lanes: &'l [Lane<'a>],
        plans: &'l [RowPlan],
        rows: usize,
        vocab: usize,
    ) -> Option<Self> {
        matches!(head, Head::Sample(spec) if spec.adjust == Adjust::Guided).then(|| Self {
            lanes,
            plans,
            rows,
            vocab,
            keys: [PassInput::SamplerMask.slot_key()],
        })
    }
}

impl LateInputs for GuidedMask<'_, '_> {
    fn keys(&self) -> &[SlotKey] {
        &self.keys
    }

    fn build(&mut self) -> Result<Vec<HostTensor>, ExecError> {
        let mut mask = Vec::with_capacity(self.rows * self.vocab);
        for (lane, plan) in self.lanes.iter().zip(self.plans) {
            match plan.picks > 0 {
                true => match lane.sampler.guided_mask(self.vocab) {
                    Some(row) => mask.extend(row),
                    None => mask.resize(mask.len() + self.vocab, 0.0),
                },
                false => mask.resize(mask.len() + self.vocab, 0.0),
            }
        }
        mask.resize(self.rows * self.vocab, 0.0);
        Ok(vec![HostTensor::f32(vec![self.rows, self.vocab], mask)])
    }
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

impl Driver<Serving> {
    /// Admit `request`: check its inputs against the model's schema, reuse the cached prefix blocks its
    /// identity matches, reserve blocks for the rest of the prompt and `max_new` tokens. A pool that
    /// cannot hold it, after dropping cached prefixes nothing references, is [`Admission::NoRoom`].
    ///
    /// A sequence that carries request inputs reuses no cached prefix and registers none: the prefix
    /// cache keys on tokens alone, which these values change the meaning of.
    pub fn open(&mut self, request: SeqRequest<'_>) -> Result<Admission, DriverError> {
        let SeqRequest {
            prompt,
            max_new,
            adapter,
            inputs,
        } = request;
        let reserved = self.check_admission(prompt, max_new)?;
        let features = SeqRequest::features_of(adapter, &inputs);
        self.compose(features)?;
        self.check_inputs(&inputs, prompt.len(), adapter)?;
        let residency = self.ledger.residency;
        let serving = &mut self.state;
        let Some(&slot) = serving.free_slots.last() else {
            return Ok(Admission::NoRoom { needed_blocks: 0 });
        };
        let private = matches!(serving.kv, KvStore::Contiguous);
        let identity = self.lora.retain(adapter)?;
        let reused = match &mut serving.kv {
            KvStore::Contiguous => 0,
            KvStore::Paged { cache, .. } => {
                // Request inputs change what the same tokens mean: such a prompt reuses no cached prefix.
                let admit = |cache: &mut PagedKvCache| {
                    if !inputs.is_empty() {
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
                if let KvStore::Paged { cache, .. } = &mut self.state.kv {
                    cache.reset(slot);
                }
                return Err(error);
            }
        }
        let serving = &mut self.state;
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
                inputs,
                history_synced: false,
            },
        );
        self.speculation.open(seq, reused);
        Ok(Admission::Opened { seq, reused })
    }

    /// Check a request's input values before it holds anything: they satisfy the model's schema for the
    /// `step` program, and the prepared set binds every carriable input the request carries under the
    /// request's weights ([`Driver::prepare_carried`]).
    fn check_inputs(
        &self,
        inputs: &RequestInputs,
        prompt: usize,
        adapter: AdapterRef,
    ) -> Result<(), DriverError> {
        let schema = self.handle.model().schema();
        let program = schema
            .step()
            .ok_or(Unsupported::Trace(TraceError::ProgramUnsupported {
                family: self.handle.config().family,
                program: ProgramId::STEP,
            }))?;
        schema
            .check_request(program, inputs, prompt)
            .map_err(InvalidRequest::Input)?;
        let generation = adapter.generation();
        for id in self.carriable_in(inputs) {
            let index = schema
                .index_of(id)
                .expect("a carriable input is one the schema declares");
            let entry = self
                .ledger
                .entries
                .iter()
                .find(|(key, _)| {
                    key.role == ModelRole::Target
                        && key.program == Program::Step
                        && key.adapters == generation
                        && key.carried.contains(index)
                })
                .map(|(_, entry)| entry)
                .ok_or(InvalidRequest::InputNotPrepared { input: id })?;
            // A per-row value is padded into the extent its program was traced at: a longer one has
            // nowhere to go, and is refused here rather than at the bind.
            if let Some(InputValue::Row(tensor)) = inputs.get(id) {
                let slot = entry.slots.iter().find_map(|slot| match &slot.binder {
                    Binder::Request(request) if request.is_value_of(id) => Some(&slot.shape),
                    _ => None,
                });
                let held = slot.map(|shape| shape[1..].iter().product::<usize>());
                if held.is_some_and(|held| tensor.numel() > held) {
                    return Err(DriverError::from(InvalidRequest::Input(
                        InputError::Support {
                            id,
                            what: "a per-row value is longer than the extent the program was traced at",
                        },
                    )));
                }
            }
        }
        Ok(())
    }

    /// Zero slot `slot`'s row of every executable's state.
    fn reset_row_state(&mut self, slot: usize) -> Result<(), DriverError> {
        let scope = poot_executor::StateScope::Row(slot);
        self.runtime
            .executor
            .reset_state(self.runtime.exe, scope.clone())?;
        if let Some(draft) = &self.speculation.draft {
            self.runtime
                .executor
                .reset_state(draft.exe, scope.clone())?;
        }
        for &exe in self.runtime.adapter_exes.values() {
            self.runtime.executor.reset_state(exe, scope.clone())?;
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
        let carries_inputs = self
            .state
            .seqs
            .get(&seq)
            .is_some_and(|state| !state.inputs.is_empty());
        self.compose(
            FeatureSet::NONE
                .with(Feature::Resume)
                .with_if(carries_inputs, Feature::RequestInputs),
        )?;
        let residency = self.ledger.residency;
        let serving = &mut self.state;
        let state = serving
            .seqs
            .get_mut(&seq)
            .ok_or(InvalidRequest::UnknownSeq(seq))?;
        if !state.pending.is_empty() {
            return Err(InvalidRequest::UncommittedResult(seq).into());
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
        let state = self.state.seqs.get_mut(&seq).expect("looked up above");
        state.tokens = prompt.to_vec();
        state.prompt_len = prompt.len();
        state.done = reused;
        state.reserved = reserved;
        state.seeded = false;
        state.history_synced = false;
        if reused == 0 {
            state.state_ahead = false;
        }
        self.speculation.rewind(seq, reused);
        Ok(reused)
    }

    /// Run one step over `rows`. Each row's sampler picks its token: the suffix entry when every
    /// picking row shares a suffix-expressible rule, otherwise the logits entry and the host path.
    /// The picks are not part of any sequence until [`Driver::commit`].
    pub fn step(&mut self, rows: &mut [StepRow<'_>]) -> Result<Vec<RowResult>, DriverError> {
        if rows.is_empty() {
            return Err(InvalidRequest::EmptyStep.into());
        }
        let max_rows = self.ledger.caps.as_ref().map_or(0, |caps| caps.rows.get());
        if rows.len() > max_rows {
            return Err(InvalidRequest::TooManyRows {
                rows: rows.len(),
                max: max_rows,
            }
            .into());
        }
        let mut plans = self.plan_rows(rows)?;
        self.propose(rows, &mut plans)?;
        let serving = &self.state;
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
        let chain = Self::chain_depth(rows)?;
        let lanes: Vec<Lane<'_>> = rows
            .iter_mut()
            .map(|row| Lane {
                seq: row.seq,
                sampler: &mut *row.sampler,
            })
            .collect();
        let readback = match chain {
            Some(depth) => self.run_chain_rows(lanes, &plans, (head, adapter), depth),
            None => self.run_rows(lanes, &mut plans, (head, adapter)),
        };
        let readback = match readback {
            Ok(readback) => readback,
            Err(error) => {
                // A step that failed part-way may have updated a history row the sampler never saw.
                for row in rows.iter() {
                    if let Some(seq) = self.state.seqs.get_mut(&row.seq) {
                        seq.history_synced = false;
                    }
                }
                return Err(error);
            }
        };

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

        let recurrent = self.ledger.residency == StateResidency::Recurrent;
        let serving = &mut self.state;
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
            let carried = self.carried_by(&group_lanes, &group_plans);
            let window = group_plans.iter().any(|plan| plan.picks > 1);
            let verifies = group_plans.iter().any(|plan| !plan.drafts().is_empty());
            let ran = self
                .step_entry(
                    ModelRole::Target,
                    group.len(),
                    tokens,
                    (window, carried),
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

    /// The number of steps every row of the step chains, `None` when none does.
    fn chain_depth(rows: &[StepRow<'_>]) -> Result<Option<NonZeroUsize>, InvalidRequest> {
        let mut depths = rows.iter().map(|row| match row.work {
            RowWork::Chain { steps } => Some(steps),
            RowWork::Prefill { .. } | RowWork::Decode { .. } => None,
        });
        let Some(first) = depths.next() else {
            return Ok(None);
        };
        if depths.all(|depth| depth == first) {
            Ok(first)
        } else {
            Err(InvalidRequest::MixedChain)
        }
    }

    /// Run the rows of a chained step: one entry per step of the chain, the first the ordinary decode
    /// entry of the head, the rest its chain copies, run back to back on the device with one wait. The
    /// head must be one whose pick the device carries into the next step.
    fn run_chain_rows(
        &mut self,
        mut lanes: Vec<Lane<'_>>,
        plans: &[RowPlan],
        (head, adapter): (Option<Head>, Option<GenerationId>),
        depth: NonZeroUsize,
    ) -> Result<Readback, DriverError> {
        let chainable = matches!(head, Some(Head::Sample(spec))
            if spec.logprobs.is_none() && spec.adjust != Adjust::Guided);
        if !chainable {
            return Err(InvalidRequest::ChainHead { head }.into());
        }
        let carried = self.carried_by(&lanes, plans);
        let key = self.step_entry(
            ModelRole::Target,
            lanes.len(),
            1,
            (false, carried),
            head,
            adapter,
        )?;
        self.run_chain(&mut lanes, plans, key, depth.get())
    }

    /// Run `depth` chained steps of the head `key` names and read back every step's picks. Each step has
    /// its own entry and its own bound inputs, all written before the first runs; the device carries
    /// each step's pick into the next step's token slot ([`Executor::step_chain`]).
    fn run_chain(
        &mut self,
        lanes: &mut [Lane<'_>],
        plans: &[RowPlan],
        key: EntryKey,
        depth: usize,
    ) -> Result<Readback, DriverError> {
        let Head::Sample(spec) = key.head else {
            unreachable!("a chain runs a sampling head")
        };
        let keys: Vec<EntryKey> = (0..depth)
            .map(|step| EntryKey {
                chain: u8::try_from(step).expect("a chain is shorter than 256 steps"),
                ..key
            })
            .collect();
        let mut entries = Vec::with_capacity(depth);
        for key in &keys {
            entries.push(self.ledger.entries.get(key).cloned().ok_or(
                PreparedSetRefusal::NotAdmitted {
                    program: key.program,
                    shape: key.shape,
                    head: key.head,
                    adapters: key.adapters,
                },
            )?);
        }
        let mut staged = Vec::with_capacity(depth);
        for (step, entry) in entries.iter().enumerate() {
            staged.push(self.step_columns(entry, lanes, plans, keys[step], step)?);
        }
        let width = key.shape.tokens.get();
        let feeds: Vec<Vec<RowFeed<'_>>> = staged
            .iter()
            .map(|(_, owned)| {
                owned
                    .iter()
                    .zip(plans)
                    .map(|(inputs, plan)| RowFeed {
                        inputs,
                        start: plan.start,
                        fed: plan.feed.len(),
                    })
                    .collect()
            })
            .collect();
        let frames: Vec<RequestFrame<'_>> = feeds
            .iter()
            .map(|feeds| RequestFrame {
                rows: key.shape.rows.get(),
                tokens: width,
                side: Side::Left,
                feeds,
            })
            .collect();
        let bound: Vec<_> = entries
            .iter()
            .zip(&staged)
            .zip(&frames)
            .enumerate()
            .map(|(step, ((entry, (columns, _)), frame))| {
                Self::bound_inputs(entry, columns, frame, step > 0)
            })
            .collect();
        let inputs: Vec<StepInputs<'_>> = bound
            .iter()
            .map(|bound| {
                let mut inputs = StepInputs::new();
                for (key, value) in bound {
                    match value {
                        Bound::Tensor(tensor) => {
                            inputs.push((*key).clone(), tensor.shape(), tensor.view());
                        }
                        Bound::WriteMap(map) => inputs.push_write_map((*key).clone(), map),
                    }
                }
                inputs
            })
            .collect();
        for (entry, (columns, _)) in entries.iter().zip(&staged) {
            self.account_step(entry, columns);
        }
        let picked = entries[0].picked.ok_or(InvalidRequest::ChainHead {
            head: Some(key.head),
        })?;
        let feed = ChainFeed {
            picked,
            slot: StepInput::Tokens.slot_key(),
        };
        let steps: Vec<ChainStep<'_, '_>> = entries
            .iter()
            .zip(&inputs)
            .map(|(entry, inputs)| ChainStep {
                entry: entry.id,
                inputs,
            })
            .collect();
        self.stats.chains += 1;
        let mut outputs =
            self.runtime
                .executor
                .step_chain(entries[0].exe, &steps, &feed, &mut NoSync)?;
        let mut picked = vec![Vec::with_capacity(depth); lanes.len()];
        for step in 0..depth {
            let bytes = outputs.read(step)?;
            for (i, tokens) in picked.iter_mut().enumerate() {
                tokens.push(suffix::read_pick(spec, &bytes, i)?.token);
            }
        }
        drop(outputs);
        self.settle_history(key.head, lanes, plans);
        Ok(Readback::Tokens(picked))
    }

    /// The columns `entry` binds for `plans`: each live row's feed, positions and cache mapping, the
    /// sampling suffix's inputs, and the request values the lanes' sequences carry. `offset` is the step's
    /// index within a chain: every row's token sits that many positions further on, and the sampler
    /// draws its seeds for that step.
    fn step_columns(
        &mut self,
        entry: &super::Prepared,
        lanes: &mut [Lane<'_>],
        plans: &[RowPlan],
        key: EntryKey,
        offset: usize,
    ) -> Result<(StepColumns, Vec<RequestInputs>), DriverError> {
        let StepShape {
            rows: entry_rows,
            tokens: width,
            capacity,
            kv,
            ..
        } = key.shape;
        let (entry_rows, width, capacity) = (entry_rows.get(), width.get(), capacity.get());
        let mut columns = StepColumns {
            tokens: vec![0; entry_rows * width],
            pos: vec![0; entry_rows * width],
            real: vec![0; entry_rows * width],
            lora: vec![0; entry_rows],
            ..StepColumns::default()
        };
        // The forward write maps, one entry per flat token (KV) and per step row (recurrent state): -1
        // writes nothing, so padding and masked rows leave the pool alone.
        let mut kv_write = Vec::new();
        if let KvLayout::Paged { .. } = kv {
            columns.kv_read = vec![0; entry_rows * capacity];
            kv_write = vec![ForwardSlotMap::PADDING; entry_rows * width];
        }
        // A recurrent state pool: masked rows read pool row 0 and write nothing.
        let pools_state = entry
            .slots
            .iter()
            .any(|spec| matches!(spec.binder, Binder::Step(StepInput::StateWriteRows)));
        let mut state_write = Vec::new();
        if pools_state {
            columns.state_read = vec![0; entry_rows];
            state_write = vec![ForwardSlotMap::PADDING; entry_rows];
        }
        let serving = &self.state;
        for (i, (lane, plan)) in lanes.iter().zip(plans).enumerate() {
            let seq = &serving.seqs[&lane.seq];
            let n = plan.feed.len();
            let map = match &serving.kv {
                KvStore::Paged { cache, .. } => {
                    let map = cache.slot_mapping(seq.slot, capacity);
                    columns.kv_read[i * capacity..(i + 1) * capacity]
                        .iter_mut()
                        .zip(&map)
                        .for_each(|(dst, &slot)| *dst = slot as i32);
                    Some(map)
                }
                KvStore::Contiguous => None,
            };
            if pools_state {
                columns.state_read[i] = seq.slot as i32;
                state_write[i] = seq.slot as i32;
            }
            columns.lora[i] = seq.adapter.row_value();
            for k in 0..width {
                // Left padding: the row's real tokens are the last `n` of the step's width.
                let real = k.checked_sub(width - n);
                let at = i * width + k;
                match real {
                    Some(j) => {
                        let position = plan.start + j + offset;
                        columns.tokens[at] = plan.feed[j] as i32;
                        columns.pos[at] = position as i32;
                        columns.real[at] = 1;
                        if let Some(map) = &map
                            && position < seq.reserved
                        {
                            kv_write[at] = map[position] as i32;
                        }
                    }
                    None => columns.pos[at] = (plan.start + offset) as i32,
                }
            }
        }
        // The request values the lanes' sequences carry, held apart from the pool so the step can
        // borrow the driver mutably while it binds them.
        let owned: Vec<RequestInputs> = lanes
            .iter()
            .map(|lane| serving.seqs[&lane.seq].inputs.clone())
            .collect();
        // Each pool slot has one owner: a block table that handed two tokens one slot is refused here.
        if let KvLayout::Paged { pool_slots, seqs } = kv {
            columns.kv_write = ForwardSlotMap::new(kv_write, pool_slots.get())?;
            if pools_state {
                columns.state_write = ForwardSlotMap::new(state_write, seqs.get())?;
            }
        }
        // The sampling suffix's inputs: the draw's per-row seed and knobs, and an adjusted head's
        // penalties, guided mask and history rows.
        if let Head::Sample(spec) = key.head {
            self.bind_sampler(entry, key.shape, spec, lanes, plans, &mut columns)?;
        }
        Ok((columns, owned))
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
        let entry = self.ledger.entries[&key].clone();
        let StepShape {
            rows: entry_rows,
            tokens: width,
            ..
        } = key.shape;
        let (entry_rows, width) = (entry_rows.get(), width.get());
        let (columns, owned) = self.step_columns(&entry, lanes, plans, key, 0)?;
        let feeds: Vec<RowFeed<'_>> = owned
            .iter()
            .zip(plans)
            .map(|(inputs, plan)| RowFeed {
                inputs,
                start: plan.start,
                fed: plan.feed.len(),
            })
            .collect();
        let frame = RequestFrame {
            rows: entry_rows,
            tokens: width,
            side: Side::Left,
            feeds: &feeds,
        };
        // A guided head's mask is built while the device runs the forward pass, not before it.
        let mut guided = GuidedMask::of(
            key.head,
            lanes,
            plans,
            entry_rows,
            self.handle.config().vocab,
        );
        let mut outputs = match &mut guided {
            Some(mask) => self.execute(&entry, &columns, &frame, mask)?,
            None => self.execute(&entry, &columns, &frame, &mut NoLate)?,
        };
        let positions = match key.shape.logits {
            LogitRows::Last => 1,
            LogitRows::All => width,
        };
        // Nothing reads the output of a step no row picks in: no readback.
        if plans.iter().all(|plan| plan.picks == 0) {
            return Ok(Readback::Tokens(vec![Vec::new(); lanes.len()]));
        }
        match key.head {
            Head::Sample(spec) => {
                let bytes = outputs.read()?;
                let mut picked = Vec::with_capacity(lanes.len());
                for (i, (lane, plan)) in lanes.iter_mut().zip(plans).enumerate() {
                    let mut tokens = Vec::with_capacity(plan.picks);
                    for j in 0..plan.picks {
                        // The row's last `picks` positions under All, its one position under Last.
                        let position = positions - plan.picks + j;
                        let pick = suffix::read_pick(spec, &bytes, i * positions + position)?;
                        tokens.push(pick.token);
                        if let (Some(mut record), Some(n)) =
                            (pick.logprob, lane.sampler.logprob_count())
                        {
                            record.top.truncate(n);
                            lane.sampler.push_logprob(record);
                        }
                    }
                    picked.push(tokens);
                }
                drop(outputs);
                self.settle_history(key.head, lanes, plans);
                Ok(Readback::Tokens(picked))
            }
            Head::Logits => {
                let rows = host_sample::read_logits(&mut outputs)?;
                drop(outputs);
                self.settle_history(key.head, lanes, plans);
                let vocab = rows.len() / (entry_rows * positions);
                Ok(Readback::Logits {
                    rows,
                    vocab,
                    positions,
                })
            }
        }
    }

    /// Bind the sampling suffix's per-step inputs of `spec` into `columns`. The draw: one row each of seed,
    /// knobs and top-k, placeholders for lanes that do not pick and for the masked rows. An adjusted
    /// head: each picking lane's penalties and guided mask, the history pool row it reads and the one
    /// its drawn token is added to, the neutral value for every other row; a picking lane whose history
    /// row does not hold its sampler's counts is seeded first.
    fn bind_sampler(
        &mut self,
        entry: &super::Prepared,
        shape: StepShape,
        spec: SampleSpec,
        lanes: &mut [Lane<'_>],
        plans: &[RowPlan],
        columns: &mut StepColumns,
    ) -> Result<(), DriverError> {
        let (entry_rows, live) = (shape.rows.get(), lanes.len());
        let masked = entry_rows - live;
        if spec.rule != SampleRule::Greedy {
            let mut seed = Vec::new();
            let mut params = Vec::new();
            let mut top_k = Vec::new();
            for (lane, plan) in lanes.iter_mut().zip(plans) {
                let row = if plan.picks > 0 {
                    DrawRow::pick(lane.sampler, spec.rule)
                } else {
                    DrawRow::placeholder(lane.sampler, spec.rule)
                };
                seed.push(row.seed);
                params.extend(row.params);
                top_k.extend(row.top_k);
            }
            // Masked rows repeat the first row's placeholder.
            let per_params = params.len() / live;
            seed.extend(std::iter::repeat_n(0, masked));
            for _ in 0..masked {
                params.extend_from_within(..per_params);
                if !top_k.is_empty() {
                    top_k.push(top_k[0]);
                }
            }
            (columns.seed, columns.params, columns.top_k) = (seed, params, top_k);
        }
        if spec.adjust == Adjust::None {
            return Ok(());
        }
        let vocab = self.handle.config().vocab;
        let pool_rows = match shape.kv {
            KvLayout::Paged { seqs, .. } => seqs.get(),
            KvLayout::Contiguous => entry_rows,
        };
        let mut write = vec![ForwardSlotMap::PADDING; entry_rows];
        columns.history_read = vec![0; entry_rows];
        columns.penalty = Vec::with_capacity(entry_rows * 3);
        for (i, (lane, plan)) in lanes.iter().zip(plans).enumerate() {
            let slot = self.state.seqs[&lane.seq].slot;
            if plan.picks > 0 {
                columns.history_read[i] = slot as i32;
                write[i] = slot as i32;
                columns.penalty.extend(lane.sampler.penalty_row());
            } else {
                columns.penalty.extend(NEUTRAL_PENALTY);
            }
        }
        for _ in live..entry_rows {
            columns.penalty.extend(NEUTRAL_PENALTY);
        }
        columns.history_write = ForwardSlotMap::new(write, pool_rows)?;
        // A picking lane reads its history row: it must hold the sampler's counts and bias.
        for (lane, plan) in lanes.iter().zip(plans) {
            let seq = self.state.seqs.get_mut(&lane.seq).expect("planned above");
            if plan.picks == 0 || seq.history_synced {
                continue;
            }
            let row = lane.sampler.history_row(vocab);
            let view = HostView::new(DType::F32, row.len(), bytemuck::cast_slice(&row))
                .expect("an f32 row is whole words");
            self.runtime
                .executor
                .write_state_row(entry.exe, HISTORY_STATE, seq.slot, view)?;
            seq.history_synced = true;
        }
        Ok(())
    }

    /// Record what a step did to the lanes' history rows: an adjusted head added each picking lane's
    /// drawn token to its row, which the host commits next, so the row stays the sampler's; any other
    /// head leaves a picking lane's row behind the sampler.
    fn settle_history(&mut self, head: Head, lanes: &[Lane<'_>], plans: &[RowPlan]) {
        let updated = matches!(head, Head::Sample(spec) if spec.adjust != Adjust::None);
        for (lane, plan) in lanes.iter().zip(plans) {
            if plan.picks > 0 && !updated {
                let seq = self.state.seqs.get_mut(&lane.seq).expect("planned above");
                seq.history_synced = false;
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
        let started = (self.settings.time_phases && counts)
            .then(|| self.runtime.executor.stats().timing.mark());
        let result = pass(self)?;
        if let Some(started) = started {
            let wall = window_wall(&self.runtime.executor.stats().timing, &started);
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
        let serving = &mut self.state;
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
        // The adjusted head added the token it drew to the sampler history on the device: a draw the
        // caller drops is in the row but not in the sampler.
        state.history_synced &= kept == pending.len();
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
        let residency = self.ledger.residency;
        let mtp = matches!(self.speculation.policy, DraftPolicy::Mtp { .. });
        let serving = &mut self.state;
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
                } else if !state.inputs.is_empty() {
                    cache.reset(state.slot);
                    PrefixOutcome::Skipped(PrefixSkip::RequestInputs)
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
        let serving = &self.state;
        let capacity = self.settings.options.capacity.get();
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
                        match self.settings.chunks.plan(upto - seq.done).first() {
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
                        spec: None,
                        not_speculated: None,
                    }
                }
                RowWork::Decode { .. } | RowWork::Chain { .. } => {
                    let (ahead, steps): (&[u32], usize) = match row.work {
                        RowWork::Decode { ahead } => (ahead, 1),
                        RowWork::Chain { steps } => (&[], steps.get()),
                        RowWork::Prefill { .. } => unreachable!("matched above"),
                    };
                    if seq.done < seq.prompt_len || seq.tokens.len() == seq.prompt_len {
                        return Err(InvalidRequest::DecodeBeforePrefill(row.seq).into());
                    }
                    let start = seq.tokens.len() - 1;
                    // A chained step writes the K/V of every token it decodes: the whole chain must
                    // lie inside the blocks the sequence reserved.
                    if start + steps > seq.reserved {
                        return Err(InvalidRequest::BeyondReservation(row.seq).into());
                    }
                    if start + steps.max(1 + ahead.len()) > capacity {
                        return Err(InvalidRequest::Capacity {
                            prompt: seq.prompt_len,
                            max_new: start + steps.max(1 + ahead.len()) - seq.prompt_len,
                            max: capacity,
                        }
                        .into());
                    }
                    if !ahead.is_empty() && !row.sampler.is_greedy() {
                        return Err(InvalidRequest::WindowNeedsGreedy(row.seq).into());
                    }
                    let mut feed = vec![seq.tokens[start]];
                    feed.extend_from_slice(ahead);
                    // Under a draft policy a decode row with nothing ahead speculates, when its
                    // sampler is one the window verifies; any other row decodes one token.
                    let (spec, not_speculated) = match self.speculation.policy.caps() {
                        Some(_) if !ahead.is_empty() || steps > 1 => (None, None),
                        Some(_) if !self.drafts_for(&seq.inputs) => {
                            (None, Some(NotSpeculated::Inputs))
                        }
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
                        spec,
                        not_speculated,
                    }
                }
            });
        }
        Ok(plans)
    }

    /// The head a step runs: the narrowest prepared sampling head that serves every picking row. A step
    /// with a verify window runs the greedy suffix entry, so every picking row must be a plain greedy
    /// row: one that is not would silently get an unmasked argmax, and the whole step is refused instead
    /// (under a draft policy a sampled row beside a window picks from the window entry's logits, the
    /// head of a caller that asked for the full logits). A head no prepared set covers is a typed
    /// refusal: there is no host pick to fall back to.
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
                .find(|(row, plan)| plan.picks > 0 && !row.sampler.is_greedy())
                .map(|(row, _)| row.seq);
            return match offender {
                None => Ok(Some(Head::GREEDY)),
                Some(_) if self.speculation.policy.caps().is_some() => Ok(Some(Head::Logits)),
                Some(seq) => Err(InvalidRequest::WindowStepNeedsGreedy(seq)),
            };
        }
        let mut needs = rows
            .iter()
            .zip(plans)
            .filter(|(_, plan)| plan.picks > 0)
            .map(|(row, _)| suffix::need_of(row.sampler));
        // No row picks: the output is unread, so any prepared head of the shape serves.
        let Some(first) = needs.next() else {
            return Ok(None);
        };
        let need = needs.fold(first, suffix::join);
        let prepared = self
            .ledger
            .caps
            .as_ref()
            .map_or(&[][..], |caps| caps.heads.suffix.as_slice());
        suffix::narrowest(prepared, need)
            .map(|spec| Some(Head::Sample(spec)))
            .ok_or(InvalidRequest::HeadNotPrepared { need })
    }

    /// The prepared entry a step runs: the smallest admitted row count holding `live` rows, at the
    /// step's width, with `head` (`None`: any head, the greedy suffix first, then the logits head). A
    /// shape `prepare` never admitted is a typed refusal; a step never compiles.
    pub(super) fn step_entry(
        &self,
        role: ModelRole,
        live: usize,
        width: usize,
        (window, carried): (bool, InputSet),
        head: Option<Head>,
        adapters: Option<super::lora::GenerationId>,
    ) -> Result<EntryKey, DriverError> {
        let serving = &self.state;
        let logits = if window {
            LogitRows::All
        } else {
            LogitRows::Last
        };
        let program = if role == ModelRole::Mtp {
            Program::Draft
        } else {
            Program::Step
        };
        let kv = serving.kv();
        let mut candidates: Vec<EntryKey> = self
            .ledger
            .entries
            .keys()
            .filter(|key| {
                key.role == role
                    && key.program == program
                    && key.carried == carried
                    && head.is_none_or(|head| key.head == head)
                    && key.adapters == adapters
                    && key.chain == 0
                    && key.shape.tokens.get() == width
                    && key.shape.logits == logits
                    && key.shape.kv == kv
                    && key.shape.rows.get() >= live
            })
            .copied()
            .collect();
        candidates.sort_by_key(|key| {
            let rank = match key.head {
                Head::Sample(spec) if spec == SampleSpec::draw(SampleRule::Greedy) => 0,
                Head::Logits => 1,
                Head::Sample(_) => 2,
            };
            (key.shape.rows, rank)
        });
        candidates.first().copied().ok_or_else(|| {
            PreparedSetRefusal::NotAdmitted {
                program,
                shape: StepShape::new(
                    std::num::NonZeroUsize::new(live).expect("a step has a row"),
                    std::num::NonZeroUsize::new(width).expect("a step has a token"),
                    self.settings.options.capacity,
                )
                .with_kv(kv)
                .with_logits(logits)
                .with_hidden(self.hidden_of(role)),
                head: head.unwrap_or(Head::GREEDY),
                adapters,
            }
            .into()
        })
    }
}
