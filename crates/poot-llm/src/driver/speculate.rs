//! Speculative decoding as a driver policy (Card 751, dserve.md ruling V4/S7).
//!
//! Speculation is not a loop and not a family list. [`DriverOptions::draft`](super::DriverOptions)
//! names a [`DraftPolicy`]; under one other than `Off`, a `Decode` row of [`Driver::step`] with nothing
//! `ahead` is a speculative row: the driver proposes up to `k` draft tokens, verifies them in one
//! window step, and the row returns `1..=k+1` tokens, the accepted drafts and one more. The window is
//! the prefill program at a nonzero start with every position's logits (`StepShape { tokens: k + 1,
//! logits: All, .. }`), so every family that traces chunked prefill verifies; nothing lists families.
//! `prepare` compiles one window entry per row count beside decode. A row with fewer than `k`
//! proposals repeats its last draft as filler to the window's width, and verification reads only the
//! picks of the real drafts (masked by length), so a changing proposal count never recompiles. A row
//! with no proposal, or too near its capacity to hold the window, is an ordinary decode step.
//!
//! A [`DraftPolicy::DraftModel`] loads a second executable on the same executor. Its entries are the
//! same shapes as the target's and its state is addressed through the same block tables, so the
//! sequences mirror each other through the one `open`/`commit`/`release` surface. Speculation is
//! refused typed for a model whose state is recurrent: a rejected draft token has already been folded
//! into that state and `commit(kept)` cannot undo it.
//!
//! Acceptance is greedy (a draft survives while it equals the target's argmax) or, for a sampled row,
//! speculative sampling over host distributions ([`spec_sample_round`]): the target's `p` from
//! [`Sampler::probs`] over the window's logits, the draft's `q` from its own forked stream (the
//! one-hot row of the proposed token for prompt lookup, so the accept test is `u < p(d)` with no second
//! code path).

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_executor::ExecutableId;
use poot_graph_plan::WeightFormats;

use super::error::{DraftMismatch, DriverError, SpeculationRefusal, Unsupported};
use super::handle::ModelHandle;
use super::step::{Lane, Readback, RowPlan, SeqId, StepRow};
use super::suffix::{self, Head};
use super::{Driver, ModelRole, StateResidency};
use crate::core::sampler::{Sampler, SamplerFault};
use crate::text::tokenize::TextCodec;

/// How a driver proposes tokens for a decode row to verify. No `Default`: every driver states it.
#[derive(Clone, Debug)]
pub enum DraftPolicy {
    /// No speculation: a decode row is one token.
    Off,
    /// Propose the tokens that followed the most recent earlier occurrence of the sequence's last
    /// `ngram` tokens ([`propose_lookup`]).
    PromptLookup {
        ngram: NonZeroUsize,
        k: NonZeroUsize,
    },
    /// Propose with a second model, `k` tokens a round. Its vocabulary must agree with the target's
    /// and its state must be positional.
    DraftModel {
        handle: Arc<ModelHandle>,
        k: NonZeroUsize,
    },
}

impl DraftPolicy {
    /// What the policy publishes: `None` for `Off`.
    pub(super) fn caps(&self) -> Option<SpeculationCaps> {
        let k = match self {
            DraftPolicy::Off => return None,
            DraftPolicy::PromptLookup { k, .. } | DraftPolicy::DraftModel { k, .. } => *k,
        };
        Some(SpeculationCaps {
            k,
            window: k.checked_add(1).expect("k + 1 fits in a usize"),
        })
    }
}

/// The speculation a driver serves: at most `k` drafts a round, verified in a window of `window`
/// positions (the query token plus the drafts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculationCaps {
    pub k: NonZeroUsize,
    pub window: NonZeroUsize,
}

/// What the draft policy did for one row of a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSpeculation {
    /// The policy proposed nothing to judge here: it is `Off`, the row is a prefill piece, or the
    /// caller supplied its own `ahead` window.
    NotApplicable,
    /// `proposed` drafts were verified and `accepted` of them kept; the row's result is the accepted
    /// drafts and one more token.
    Verified { proposed: usize, accepted: usize },
    /// The row decoded one token under a draft policy, for this reason.
    NotSpeculated(NotSpeculated),
}

/// Why a decode row under a draft policy decoded one token. Its output is correct either way; this
/// names the missed speedup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotSpeculated {
    /// The row's sampler has a penalty, a logit bias, a constraint or logprob recording, which the
    /// window's acceptance does not reproduce.
    Sampler,
    /// The sequence is too near its capacity or its reserved length to hold a window.
    NoRoom,
    /// The policy found nothing to propose (prompt lookup: no earlier match).
    NoProposal,
}

/// Rounds the driver's policy ran, by outcome: the engine's own acceptance accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeculationStats {
    /// Rows verified.
    pub rounds: u64,
    /// Drafts those rows proposed.
    pub proposed: u64,
    /// Drafts those rows kept.
    pub accepted: u64,
    /// Decode rows that decoded one token because their sampler is not verifiable.
    pub skipped_sampler: u64,
    /// Decode rows that decoded one token for want of room.
    pub skipped_no_room: u64,
    /// Decode rows that decoded one token because nothing was proposed.
    pub skipped_no_proposal: u64,
}

impl SpeculationStats {
    pub(super) fn record(&mut self, row: RowSpeculation) {
        match row {
            RowSpeculation::NotApplicable => {}
            RowSpeculation::Verified { proposed, accepted } => {
                self.rounds += 1;
                self.proposed += proposed as u64;
                self.accepted += accepted as u64;
            }
            RowSpeculation::NotSpeculated(NotSpeculated::Sampler) => self.skipped_sampler += 1,
            RowSpeculation::NotSpeculated(NotSpeculated::NoRoom) => self.skipped_no_room += 1,
            RowSpeculation::NotSpeculated(NotSpeculated::NoProposal) => {
                self.skipped_no_proposal += 1;
            }
        }
    }
}

/// The salt that derives the draft's proposal stream from a row's sampler: one request seed gives two
/// independent streams, the draft's proposals and the target's accept and residual draws.
pub const DRAFT_STREAM_SALT: u64 = 0x0D2A_F751_5EED_D2AF;

/// The draft model's own weights, executable and packed-weight formats.
pub(super) struct DraftScope {
    pub(super) handle: Arc<ModelHandle>,
    pub(super) exe: ExecutableId,
    pub(super) formats: WeightFormats,
}

/// What the driver keeps of one sequence's draft: how many leading positions hold the draft's K/V
/// for the sequence's committed tokens, and the stream a sampled row's proposals draw from.
struct DraftSeq {
    valid: usize,
    sampler: Option<Sampler>,
}

/// The driver's speculation state: the policy, the loaded draft and the per-sequence draft progress.
pub(super) struct Speculation {
    pub(super) policy: DraftPolicy,
    pub(super) draft: Option<DraftScope>,
    seqs: HashMap<SeqId, DraftSeq>,
}

impl Speculation {
    pub(super) fn new(policy: DraftPolicy) -> Self {
        Self {
            policy,
            draft: None,
            seqs: HashMap::new(),
        }
    }

    /// A sequence opened with `reused` leading positions already resident in the pool: the draft's K/V
    /// for them was written by the run that registered them.
    pub(super) fn open(&mut self, seq: SeqId, reused: usize) {
        if matches!(self.policy, DraftPolicy::DraftModel { .. }) {
            self.seqs.insert(
                seq,
                DraftSeq {
                    valid: reused,
                    sampler: None,
                },
            );
        }
    }

    /// `seq` continues with a new prompt that shares its first `kept` positions with the old context:
    /// the draft's K/V holds nothing valid past them.
    pub(super) fn rewind(&mut self, seq: SeqId, kept: usize) {
        if let Some(state) = self.seqs.get_mut(&seq) {
            state.valid = state.valid.min(kept);
            state.sampler = None;
        }
    }

    pub(super) fn close(&mut self, seq: SeqId) {
        self.seqs.remove(&seq);
    }
}

/// What a speculative row proposed: the drafts and, for a sampled row, the draft's distribution at each.
#[derive(Debug, Default)]
pub(super) struct Proposal {
    pub(super) drafts: Vec<u32>,
    /// `q[i]` is the distribution `drafts[i]` was drawn from; empty for a greedy row.
    pub(super) q: Vec<Vec<f32>>,
}

/// A speculative row of a step: the most drafts its sequence has room for, then what was proposed.
#[derive(Debug)]
pub(super) struct SpecRow {
    pub(super) room: usize,
    pub(super) proposal: Proposal,
}

impl Driver {
    /// Refuse a draft the target cannot verify, before anything compiles or uploads.
    pub(super) fn check_draft(&self) -> Result<(), DriverError> {
        let DraftPolicy::DraftModel { handle: draft, .. } = &self.speculation.policy else {
            return Ok(());
        };
        let (target, drafted) = (self.handle.config(), draft.config());
        let mismatch = if drafted.vocab > target.vocab {
            Some(DraftMismatch::Vocab {
                draft: drafted.vocab,
                target: target.vocab,
            })
        } else if drafted.max_positions < self.options.capacity.get() {
            Some(DraftMismatch::Positions {
                draft: drafted.max_positions,
                capacity: self.options.capacity.get(),
            })
        } else if !same_token_ids(draft.text(), self.handle.text(), target.vocab) {
            Some(DraftMismatch::Tokens)
        } else {
            None
        };
        match mismatch {
            Some(mismatch) => {
                Err(Unsupported::Speculation(SpeculationRefusal::DraftMismatch(mismatch)).into())
            }
            None => Ok(()),
        }
    }

    /// Refuse speculation on a model whose state is recurrent, read from the state roles of what
    /// compiled.
    pub(super) fn check_speculation_state(&self) -> Result<(), DriverError> {
        if self.speculation.policy.caps().is_some() && self.residency == StateResidency::Recurrent {
            return Err(Unsupported::Speculation(SpeculationRefusal::RecurrentState).into());
        }
        Ok(())
    }

    /// Load the draft's weights as a second executable on this executor, once.
    pub(super) fn load_draft(&mut self) -> Result<(), DriverError> {
        let DraftPolicy::DraftModel { handle, .. } = &self.speculation.policy else {
            return Ok(());
        };
        if self.speculation.draft.is_some() {
            return Ok(());
        }
        let handle = Arc::clone(handle);
        let formats = WeightFormats::from_weight_map(handle.model().weights());
        let weights = Arc::new(handle.model().weights().clone());
        let exe = self
            .executor
            .load_weights(handle.shared_store().clone(), weights)?;
        self.speculation.draft = Some(DraftScope {
            handle,
            exe,
            formats,
        });
        Ok(())
    }

    /// Fill each speculative row's proposal. A row with nothing to propose stays a plain decode row.
    pub(super) fn propose(
        &mut self,
        rows: &mut [StepRow<'_>],
        plans: &mut [RowPlan],
    ) -> Result<(), DriverError> {
        for (row, plan) in rows.iter().zip(plans.iter_mut()) {
            let Some(room) = plan.spec.as_ref().map(|spec| spec.room) else {
                continue;
            };
            let sampled = !row.sampler.is_greedy();
            let proposal = match &self.speculation.policy {
                DraftPolicy::Off => Proposal::default(),
                DraftPolicy::PromptLookup { ngram, .. } => {
                    let tokens = self.committed(row.seq);
                    let drafts = propose_lookup(&tokens, ngram.get(), room);
                    let vocab = self.handle.config().vocab;
                    let q = if sampled {
                        drafts.iter().map(|&d| one_hot(d, vocab)).collect()
                    } else {
                        Vec::new()
                    };
                    Proposal { drafts, q }
                }
                DraftPolicy::DraftModel { .. } => {
                    self.propose_with_draft(row.seq, row.sampler, room)?
                }
            };
            if let Some(&last) = proposal.drafts.last() {
                // Every window is `window` wide, so a changing proposal count never changes the
                // entry. A row with fewer drafts repeats its last one as filler: causal attention
                // keeps a position's pick independent of what follows it, and verification reads
                // only the picks of the real drafts (masked by length).
                plan.feed.extend_from_slice(&proposal.drafts);
                let window = self
                    .speculation
                    .policy
                    .caps()
                    .expect("a proposal needs a policy")
                    .window;
                plan.feed.resize(window.get(), last);
                plan.picks = plan.feed.len();
            }
            plan.spec = Some(SpecRow { room, proposal });
        }
        Ok(())
    }

    /// The committed tokens of `seq`: its prompt and the generated tokens the caller has kept.
    fn committed(&self, seq: SeqId) -> Vec<u32> {
        self.serving.as_ref().expect("a step is served").seqs[&seq]
            .tokens
            .clone()
    }

    /// Up to `room` draft tokens from the draft model. It first catches up to the committed tokens it
    /// has not seen, one decode step each; the step after the last known token yields a draft, which
    /// is fed in turn for the next. A sampled row draws its drafts from the stream forked from the
    /// row's sampler on its first round, and keeps the draft's distribution at each.
    fn propose_with_draft(
        &mut self,
        seq: SeqId,
        row_sampler: &Sampler,
        room: usize,
    ) -> Result<Proposal, DriverError> {
        if room == 0 {
            return Ok(Proposal::default());
        }
        let tokens = self.committed(seq);
        let committed = tokens.len();
        let vocab = self.handle.config().vocab;
        let state = self
            .speculation
            .seqs
            .get_mut(&seq)
            .expect("a sequence opened under a draft model is mirrored");
        // A caller that kept fewer tokens than it was handed has not moved the draft past them.
        let mut position = state.valid.min(committed - 1);
        let mut draws = (!row_sampler.is_greedy()).then(|| {
            state
                .sampler
                .take()
                .unwrap_or_else(|| row_sampler.fork(DRAFT_STREAM_SALT))
        });
        let mut proposal = Proposal::default();
        let outcome = loop {
            let known = |at: usize| match at.checked_sub(committed) {
                None => tokens[at],
                Some(draft) => proposal.drafts[draft],
            };
            let pick = position + 1 == committed + proposal.drafts.len();
            let kind = match (pick, &draws) {
                (false, _) => None,
                (true, Some(_)) => Some(DraftPick::Distribution),
                (true, None) => Some(DraftPick::Argmax),
            };
            let out = match self.draft_step(seq, known(position), position, kind) {
                Ok(out) => out,
                Err(error) => break Err(error),
            };
            position += 1;
            let draft = match (out, &mut draws) {
                (None, _) => continue,
                (Some(DraftOut::Token(token)), _) => token,
                (Some(DraftOut::Logits(logits)), Some(draws)) => {
                    let drawn = draws.probs(&logits).and_then(|mut q| {
                        q.resize(vocab, 0.0);
                        draws.sample_from_probs(&q).map(|token| (q, token))
                    });
                    match drawn {
                        Ok((q, token)) => {
                            proposal.q.push(q);
                            token
                        }
                        Err(fault) => break Err(fault.into()),
                    }
                }
                (Some(DraftOut::Logits(_)), None) => unreachable!("logits are asked for by draws"),
            };
            proposal.drafts.push(draft);
            if proposal.drafts.len() == room {
                break Ok(());
            }
        };
        let state = self
            .speculation
            .seqs
            .get_mut(&seq)
            .expect("looked up above");
        state.sampler = draws;
        outcome.map(|()| proposal)
    }

    /// One decode step of the draft over the single row `seq`: `token` at `position`. With a `pick`,
    /// the draft's choice after it: its argmax token, or its logits row.
    fn draft_step(
        &mut self,
        seq: SeqId,
        token: u32,
        position: usize,
        pick: Option<DraftPick>,
    ) -> Result<Option<DraftOut>, DriverError> {
        let head = match pick {
            Some(DraftPick::Distribution) => Head::Logits,
            _ => Head::GREEDY,
        };
        let plan = RowPlan {
            feed: vec![token],
            start: position,
            picks: usize::from(pick.is_some()),
            prefill: false,
            embeds: false,
            spec: None,
            not_speculated: None,
        };
        let key = self.step_entry(ModelRole::Draft, 1, 1, (false, false), Some(head), None)?;
        let mut idle = Sampler::greedy();
        let mut lanes = [Lane {
            seq,
            sampler: &mut idle,
        }];
        let readback = self.run_plans(&mut lanes, std::slice::from_ref(&plan), key)?;
        Ok(match (pick, readback) {
            (None, _) => None,
            (Some(_), Readback::Tokens(tokens)) => Some(DraftOut::Token(tokens[0][0])),
            (Some(_), Readback::Logits { rows, vocab, .. }) => {
                Some(DraftOut::Logits(rows[..vocab].to_vec()))
            }
        })
    }

    /// Run the draft over the prompt pieces the target just prefilled, so both hold the same
    /// positions. Rows that did not prefill are untouched.
    pub(super) fn mirror_prefill(
        &mut self,
        rows: &[StepRow<'_>],
        plans: &[RowPlan],
    ) -> Result<(), DriverError> {
        if self.speculation.draft.is_none() {
            return Ok(());
        }
        let prefilled: Vec<usize> = (0..plans.len()).filter(|&i| plans[i].prefill).collect();
        let Some(width) = prefilled.iter().map(|&i| plans[i].feed.len()).max() else {
            return Ok(());
        };
        let key = self.step_entry(
            ModelRole::Draft,
            prefilled.len(),
            width,
            (false, false),
            Some(Head::GREEDY),
            None,
        )?;
        let mut idle: Vec<Sampler> = prefilled.iter().map(|_| Sampler::greedy()).collect();
        let mut lanes: Vec<Lane<'_>> = prefilled
            .iter()
            .zip(&mut idle)
            .map(|(&i, sampler)| Lane {
                seq: rows[i].seq,
                sampler,
            })
            .collect();
        let silent: Vec<RowPlan> = prefilled
            .iter()
            .map(|&i| RowPlan {
                feed: plans[i].feed.clone(),
                start: plans[i].start,
                picks: 0,
                prefill: true,
                embeds: false,
                spec: None,
                not_speculated: None,
            })
            .collect();
        self.run_plans(&mut lanes, &silent, key)?;
        for &i in &prefilled {
            if let Some(state) = self.speculation.seqs.get_mut(&rows[i].seq) {
                state.valid = plans[i].start + plans[i].feed.len();
            }
        }
        Ok(())
    }

    /// A speculative row settled: `kept` tokens came back for `proposed` drafts, so the draft's K/V
    /// is valid for the committed tokens and the accepted drafts it fed, `committed` positions plus
    /// the accepted ones below the last draft (which it never fed).
    pub(super) fn settle(&mut self, seq: SeqId, committed: usize, proposed: usize, kept: usize) {
        if let Some(state) = self.speculation.seqs.get_mut(&seq) {
            let accepted = kept - 1;
            state.valid = committed + accepted.min(proposed - 1);
        }
    }
}

/// Whether the draft model's tokenizer gives every token it has, within the target's ids, the id the
/// target gives it. Ids beyond the target's vocabulary are the vocabulary check's.
fn same_token_ids(draft: &TextCodec, target: &TextCodec, target_vocab: usize) -> bool {
    let target = target.tokenizer.get_vocab(true);
    draft
        .tokenizer
        .get_vocab(true)
        .iter()
        .filter(|&(_, &id)| (id as usize) < target_vocab)
        .all(|(token, id)| target.get(token) == Some(id))
}

/// What a draft step is asked to return.
#[derive(Clone, Copy, Debug)]
enum DraftPick {
    Argmax,
    Distribution,
}

/// What a draft step returned.
enum DraftOut {
    Token(u32),
    /// The draft's logits row.
    Logits(Vec<f32>),
}

/// What a speculative row commits from its window, or `None` for a row with nothing to verify.
/// `picked` is the window's picks by the greedy suffix or the host argmax; a sampled row ignores it
/// and draws through [`spec_sample_round`] over the logits at its last `plan.picks` positions of row
/// `index`.
pub(super) fn verify_row(
    sampler: &mut Sampler,
    plan: &RowPlan,
    index: usize,
    picked: &[u32],
    readback: &Readback,
) -> Result<Option<Vec<u32>>, DriverError> {
    let Some(spec) = &plan.spec else {
        return Ok(None);
    };
    let drafts = &spec.proposal.drafts;
    if drafts.is_empty() {
        return Ok(None);
    }
    // Masked by length: only the picks of the real drafts and the one after them count.
    let real = drafts.len() + 1;
    if sampler.is_greedy() {
        return Ok(Some(accept_greedy(drafts, &picked[..real])));
    }
    let Readback::Logits {
        rows,
        vocab,
        positions,
    } = readback
    else {
        unreachable!("a sampled window reads the logits head");
    };
    let first = (index * positions + positions - plan.picks) * vocab;
    let logits: Vec<&[f32]> = (0..real)
        .map(|j| &rows[first + j * vocab..first + (j + 1) * vocab])
        .collect();
    Ok(Some(accept_sampled(
        drafts,
        &spec.proposal.q,
        &logits,
        sampler,
    )?))
}

/// The one-hot distribution of `token` over `vocab` tokens: prompt lookup's `q`.
fn one_hot(token: u32, vocab: usize) -> Vec<f32> {
    let mut row = vec![0.0; vocab];
    row[token as usize] = 1.0;
    row
}

/// Greedy acceptance over a window `[query, d0..d(m-1)]`: `picks[i]` is the target's argmax after
/// the window's first `i + 1` tokens. A draft survives while it equals the pick before it; the result
/// is the surviving drafts and the pick after them, so it is `1..=m+1` tokens.
fn accept_greedy(drafts: &[u32], picks: &[u32]) -> Vec<u32> {
    debug_assert_eq!(
        picks.len(),
        drafts.len() + 1,
        "one pick per window position"
    );
    let accepted = drafts
        .iter()
        .zip(picks)
        .take_while(|(draft, pick)| draft == pick)
        .count();
    picks[..=accepted].to_vec()
}

/// Sampled acceptance: [`spec_sample_round`] over the target's `p` at each window position.
fn accept_sampled(
    drafts: &[u32],
    q: &[Vec<f32>],
    logits: &[&[f32]],
    sampler: &mut Sampler,
) -> Result<Vec<u32>, SamplerFault> {
    let p = logits
        .iter()
        .map(|row| sampler.probs(row))
        .collect::<Result<Vec<_>, _>>()?;
    let (accepted, next) = spec_sample_round(drafts, q, &p, sampler)?;
    let mut tokens = drafts[..accepted].to_vec();
    tokens.push(next);
    Ok(tokens)
}

/// Speculative sampling's accept/reject test (Leviathan et al. 2023 Algorithm 1; Chen et al. 2023):
/// whether to accept a draft token proposed with probability `q_tok` under the draft, given the target's
/// probability `p_tok` and a uniform draw `u` in `[0, 1)`. Accepts with probability `min(1, p_tok/q_tok)`.
/// `q_tok <= 0.0` (a float-underflow edge case; a real draw has `q_tok > 0`) is treated as an infinite
/// ratio, i.e. always accept.
fn spec_accept(p_tok: f32, q_tok: f32, u: f32) -> bool {
    q_tok <= 0.0 || u < (p_tok / q_tok).min(1.0)
}

/// The residual distribution resampled from after a rejected draft token: `max(0, p(x) - q(x))`
/// normalized to sum to 1. Resampling from this rather than `p` is what makes the whole procedure
/// sample exactly from `p`: the accept step already spends `min(p(x), q(x))` of `p`'s mass. Falls back to
/// `p` if the residual mass is numerically <= 0 (only when `p == q` everywhere, so the accept test never
/// rejects in exact arithmetic; a float-rounding safety net).
fn residual_probs(p: &[f32], q: &[f32]) -> Vec<f32> {
    let mut r: Vec<f32> = p
        .iter()
        .zip(q)
        .map(|(&pi, &qi)| (pi - qi).max(0.0))
        .collect();
    let sum: f32 = r.iter().sum();
    if sum <= 0.0 {
        return p.to_vec();
    }
    for x in &mut r {
        *x /= sum;
    }
    r
}

/// One round's accept/reject/residual bookkeeping for sampled speculative decoding, a pure host
/// function. Inputs: the draft's proposals `drafts`, its per-position distributions `q` (`q[i]` at the
/// moment it proposed `drafts[i]`), and the target's `p` (`p[i]` is for the token after `drafts[..i]`, so
/// `p.len() == drafts.len() + 1`). Walks the drafts left to right, accepting via [`spec_accept`] until the
/// first rejection (resampled from [`residual_probs`]) or the drafts run out (a fresh draw from `p`'s last
/// row). Returns `(accepted_count, next_token)`, or a [`SamplerFault`] when a distribution row holds a NaN
/// or infinity. All draws come from `sampler`, deterministically.
pub fn spec_sample_round(
    drafts: &[u32],
    q: &[Vec<f32>],
    p: &[Vec<f32>],
    sampler: &mut Sampler,
) -> Result<(usize, u32), SamplerFault> {
    debug_assert_eq!(q.len(), drafts.len(), "one proposal distribution per draft");
    debug_assert_eq!(
        p.len(),
        drafts.len() + 1,
        "one target row per draft plus the bonus row"
    );
    for i in 0..drafts.len() {
        let tok = drafts[i] as usize;
        let p_tok = p[i].get(tok).copied().unwrap_or(0.0);
        let q_tok = q[i].get(tok).copied().unwrap_or(0.0);
        let u = sampler.uniform();
        if spec_accept(p_tok, q_tok, u) {
            continue;
        }
        let residual = residual_probs(&p[i], &q[i]);
        return Ok((i, sampler.sample_from_probs(&residual)?));
    }
    Ok((drafts.len(), sampler.sample_from_probs(&p[drafts.len()])?))
}

/// Prompt-lookup draft proposer (spec 029): finds the most recent earlier occurrence of the last `ngram`
/// tokens of `tokens` and returns up to `max_k` tokens that followed it, to be verified in one forward.
/// Returns empty if there is no earlier match. Pure function over the token sequence.
pub fn propose_lookup(tokens: &[u32], ngram: usize, max_k: usize) -> Vec<u32> {
    let n = tokens.len();
    if ngram == 0 || max_k == 0 || n < ngram + 1 {
        return Vec::new();
    }
    let pattern = &tokens[n - ngram..];
    // earlier pattern starts, most-recent first; a match at `start` has a following token (start+ngram < n).
    for start in (0..n - ngram).rev() {
        if &tokens[start..start + ngram] == pattern {
            let follow = start + ngram;
            let take = max_k.min(n - follow);
            return tokens[follow..follow + take].to_vec();
        }
    }
    Vec::new()
}

/// Whether `sampler`'s row can speculate: its draws are the plain greedy or temperature ones
/// [`Sampler::probs`] reproduces. A penalty, bias, constraint or logprob record is not.
pub(super) fn speculation_sampler_ok(sampler: &Sampler) -> bool {
    suffix::rule_of(sampler).is_some()
}

#[cfg(test)]
mod prompt_lookup_tests {
    use super::propose_lookup;

    #[test]
    fn proposes_recent_continuation() {
        // last 3 = [10,11,12]; the only earlier match is at start 0; the tokens after it are [99,10,...].
        let toks = [10u32, 11, 12, 99, 10, 11, 12];
        assert_eq!(propose_lookup(&toks, 3, 2), vec![99, 10]);
    }

    #[test]
    fn most_recent_match_wins() {
        // pattern [7,8] occurs at start 0 and start 3; the MOST RECENT (start 3) wins -> follows with [5,9].
        let toks = [7u32, 8, 0, 7, 8, 5, 9, 7, 8];
        assert_eq!(propose_lookup(&toks, 2, 2), vec![5, 9]);
    }

    #[test]
    fn empty_on_no_match_or_degenerate() {
        assert!(propose_lookup(&[1, 2, 3, 4], 2, 3).is_empty()); // last [3,4] never recurs
        assert!(propose_lookup(&[1, 2], 3, 2).is_empty()); // sequence shorter than ngram+1
        assert!(propose_lookup(&[1, 2, 3], 2, 0).is_empty()); // max_k = 0
    }

    #[test]
    fn truncates_to_available_follow_tokens() {
        // pattern [1,2] recurs at start 0 (follow=2); the proposer drafts the next min(max_k, n-follow)=3
        // tokens to the END of the sequence - [3,1,2] - even with max_k=5 it never reads past the end.
        let toks = [1u32, 2, 3, 1, 2];
        assert_eq!(propose_lookup(&toks, 2, 5), vec![3, 1, 2]);
    }

    #[test]
    fn unigram_pattern_matches_most_recent() {
        // ngram=1: the last token (5) recurs; the MOST RECENT earlier 5 is at start 3, followed by [6,7].
        let toks = [5u32, 9, 0, 5, 6, 7, 5];
        assert_eq!(propose_lookup(&toks, 1, 2), vec![6, 7]);
    }
}

/// Model/GPU-free tests of the speculative-sampling algorithm ([`spec_accept`], [`residual_probs`],
/// [`spec_sample_round`]) and of greedy acceptance. The bar is distributional: an empirical chi-squared
/// test checks that accept/residual reproduces the target's distribution `p`. Fixed seeds, so
/// deterministic.
#[cfg(test)]
mod spec_sampling_tests {
    use super::*;

    /// Pearson's chi-squared goodness-of-fit statistic for `observed` counts (summing to `n`) against
    /// `expected_probs` (summing to ~1).
    fn chi_squared(observed: &[u64], expected_probs: &[f32], n: u64) -> f64 {
        observed
            .iter()
            .zip(expected_probs)
            .map(|(&o, &p)| {
                let e = f64::from(p) * n as f64;
                if e <= 0.0 {
                    0.0
                } else {
                    (o as f64 - e).powi(2) / e
                }
            })
            .sum()
    }

    #[test]
    fn greedy_acceptance_keeps_the_agreeing_prefix_and_the_pick_after_it() {
        // drafts [5, 6, 7]; the target's picks after each window prefix are [5, 6, 9, 4].
        assert_eq!(accept_greedy(&[5, 6, 7], &[5, 6, 9, 4]), [5, 6, 9]);
        // the first draft already disagrees: only the target's own pick survives.
        assert_eq!(accept_greedy(&[5, 6, 7], &[8, 6, 7, 4]), [8]);
        // every draft agrees: the bonus pick follows them.
        assert_eq!(accept_greedy(&[5, 6], &[5, 6, 3]), [5, 6, 3]);
    }

    #[test]
    fn spec_accept_matches_min_one_p_over_q() {
        // p >= q: ratio clamped to 1 -> always accept (u is drawn in [0,1), always < 1).
        assert!(spec_accept(0.6, 0.3, 0.0));
        assert!(spec_accept(0.6, 0.3, 0.999));
        assert!(spec_accept(0.5, 0.5, 0.999999));
        // p < q: ratio is p/q < 1 -> accept iff u is below that ratio, reject at/above it.
        assert!(spec_accept(0.2, 0.8, 0.1)); // 0.1 < 0.25
        assert!(!spec_accept(0.2, 0.8, 0.25)); // 0.25 is not < 0.25
        assert!(!spec_accept(0.2, 0.8, 0.9));
        // p == 0: never accept (ratio 0) unless q is also <= 0 (the infinite-ratio edge case).
        assert!(!spec_accept(0.0, 0.5, 0.0));
        // q <= 0: treated as an infinite ratio -> always accept regardless of p or u.
        assert!(spec_accept(0.0, 0.0, 0.999));
        assert!(spec_accept(0.9, 0.0, 0.999));
    }

    #[test]
    fn residual_probs_is_normalized_positive_part_of_p_minus_q() {
        let p = [0.5f32, 0.3, 0.2];
        let q = [0.1f32, 0.1, 0.8];
        // raw p-q = [0.4, 0.2, -0.6] -> clamp negatives to 0 -> [0.4, 0.2, 0.0] -> normalize (sum 0.6).
        let r = residual_probs(&p, &q);
        let expect = [0.4f32 / 0.6, 0.2 / 0.6, 0.0];
        for (a, b) in r.iter().zip(expect) {
            assert!((a - b).abs() < 1e-6, "{r:?} vs {expect:?}");
        }
        let sum: f32 = r.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "residual must sum to 1: {r:?}");

        // p == q everywhere: no residual mass -> falls back to p itself (the safety net; in exact
        // arithmetic this branch of the algorithm is unreachable since spec_accept never rejects then).
        let same = [0.25f32, 0.25, 0.25, 0.25];
        assert_eq!(residual_probs(&same, &same), same.to_vec());
    }

    #[test]
    fn residual_correction_matches_target_distribution_chi_squared() {
        // For any two distributions p (target) and q (draft) over the same finite support, "propose x~q, accept
        // with prob min(1,p(x)/q(x)), else resample from max(0,p-q) normalized" samples exactly x~p
        // (Leviathan et al. 2023, Theorem 1; Chen et al. 2023). Pin it with an empirical chi-squared test over
        // several (p,q) pairs.
        let cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
            // draft and target nearly disagree (opposite-shaped distributions) - the hard, high-rejection
            // case that most directly exercises the residual-resample path.
            (vec![0.7, 0.1, 0.1, 0.1], vec![0.1, 0.1, 0.1, 0.7]),
            // same support, different shape.
            (vec![0.4, 0.3, 0.2, 0.1], vec![0.1, 0.2, 0.3, 0.4]),
            // draft nearly ignores the target's dominant token.
            (vec![0.9, 0.05, 0.03, 0.02], vec![0.02, 0.03, 0.05, 0.9]),
            // identical distributions - should reduce to "always accept" (never touches the residual path).
            (vec![0.25, 0.25, 0.25, 0.25], vec![0.25, 0.25, 0.25, 0.25]),
            // a wider, vocab-shaped case (8 categories).
            (
                vec![0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03],
                vec![0.05, 0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10],
            ),
        ];
        for (p, q) in cases {
            let n = 200_000u64;
            let mut draft_sampler = Sampler::new(1.0, 0, 1.0, 0x0DAF7);
            let mut target_sampler = Sampler::new(1.0, 0, 1.0, 0x7A5CE7);
            let mut counts = vec![0u64; p.len()];
            for _ in 0..n {
                // One round of the production function with one draft: the output token is the draft
                // when it is accepted, else the residual draw. The bonus row after it is unused.
                let x = draft_sampler.sample_from_probs(&q).unwrap();
                let (accepted, next) = spec_sample_round(
                    &[x],
                    std::slice::from_ref(&q),
                    &[p.clone(), p.clone()],
                    &mut target_sampler,
                )
                .unwrap();
                let out = if accepted == 1 { x } else { next };
                counts[out as usize] += 1;
            }
            let chi2 = chi_squared(&counts, &p, n);
            // Threshold: the alpha=0.001 critical chi-squared value for 7 degrees of freedom is ~24.3. A correct
            // implementation lands near dof (single digits at n=200k); a broken residual/accept formula (e.g.
            // resampling from `p` on rejection) is off by orders of magnitude.
            assert!(
                chi2 < 60.0,
                "chi-squared {chi2:.2} too high for p={p:?} q={q:?} (counts={counts:?}) - the accept/\
                 residual procedure's output does not match the target distribution"
            );
        }
    }

    #[test]
    fn spec_sample_round_stops_at_first_rejection_and_ignores_later_rows() {
        // 3 drafts; position 0 is forced accept (q==p one-hot on the drafted token), position 1 is forced
        // reject (target gives the drafted token zero probability) with all residual mass on token 2, so the
        // result is deterministic for any u. Rows 2 and 3 are garbage and must never be consulted: the loop
        // stops at the first rejection.
        let drafts = [0u32, 1, 2];
        let q = vec![
            vec![1.0, 0.0, 0.0], // step 0: drafted token 0, one-hot
            vec![0.0, 1.0, 0.0], // step 1: drafted token 1, one-hot
            vec![0.0, 0.0, 0.0], // step 2: never read (loop already returned)
        ];
        let p = vec![
            vec![1.0, 0.0, 0.0], // row 0: agrees with the draft -> forced accept
            vec![0.0, 0.0, 1.0], // row 1: puts ALL mass on token 2, none on the drafted token 1 -> reject
            vec![9.0, 9.0, 9.0], // row 2: never read
            vec![9.0, 9.0, 9.0], // row 3: never read
        ];
        let mut sampler = Sampler::new(1.0, 0, 1.0, 42);
        let (accepted, bonus) = spec_sample_round(&drafts, &q, &p, &mut sampler).unwrap();
        assert_eq!(
            accepted, 1,
            "only the forced-accept draft at position 0 survives"
        );
        assert_eq!(
            bonus, 2,
            "residual at the rejected position is one-hot on token 2"
        );
    }

    #[test]
    fn spec_sample_round_all_accepted_draws_bonus_from_final_target_row() {
        // both drafts forced-accept (q==p one-hot on each drafted token); the bonus must come from p's last
        // row (index drafts.len()), one-hot on token 2, so deterministic regardless of the rng draw.
        let drafts = [0u32, 1];
        let q = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let p = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0], // the bonus row
        ];
        let mut sampler = Sampler::new(1.0, 0, 1.0, 7);
        let (accepted, bonus) = spec_sample_round(&drafts, &q, &p, &mut sampler).unwrap();
        assert_eq!(accepted, 2, "both forced-accept drafts must be taken");
        assert_eq!(bonus, 2, "bonus drawn from the final target row");
    }
}
