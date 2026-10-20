//! Beam search as a driver capability (Card 752, dserve.md ruling V4/S7).
//!
//! The semantics are vLLM's (`vllm/entrypoints/generate/beam_search`):
//!
//! - a hypothesis scores `cum_logprob / seq_len ** length_penalty`, `seq_len` its whole token count
//!   (prompt included) minus one when its last token is an end-of-sequence token ([`Ranking::score`]);
//! - each live beam is expanded by its `2 * width` best continuations over the log-softmax of its
//!   logits. Only one candidate of a beam can end the sequence, so a breadth of `width` leaves
//!   `width - 1` live beams after the prompt's first step and the search never recovers;
//! - a candidate that ends the sequence goes to the completed pool and is never expanded again; the
//!   rest are pooled across all live beams, ranked and truncated to `width`;
//! - the result is the completed pool plus the live beams, ranked and truncated to `width`. Width 1 with
//!   a length penalty of 0 ranks by the raw log-probability, which is greedy decoding.
//!
//! # Rows
//!
//! The beams are the rows of a batched step. The driver prefills the prompt in row 0 of an entry at
//! `rows = width`, then decodes every live beam in one step at the same row count. Beam entries are
//! contiguous-layout entries of their own ([`ModelRole::Beam`]), so a paged driver's pool is never a
//! beam's state: state is keyed by name and aval inside the executable, and a leading axis of `width`
//! is a different aval from the pool's.
//!
//! After each expansion the survivors' state moves to their rows with one compiled row gather
//! ([`reorder`](super::reorder)) over every carried state value, KV and recurrent alike. The gather
//! runs only when the parent permutation is not the identity, and the first one, after the prompt's
//! prefill, fills rows `1..width` from row 0. State is copied, not shared by block: block sharing
//! needs copy-on-write of the partial tail block and does not cover recurrent state.
//!
//! The top-`2 * width` log-probabilities are taken on the host from the logits head's output
//! ([`host_sample`]): the sampling suffix cannot express a top-k log-probability head.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_executor::NoLate;
use poot_models::io::{InputSet, ProgramId, RequestInputs};
use poot_models::model::StepShape;

use super::bind::{self, RequestFrame, RowFeed, Side, StepColumns};
use super::chunks::Piece;
use super::error::{InvalidRequest, PreparedSetRefusal, Unsupported};
use super::{
    Driver, DriverError, EntryIo, EntryKey, FinishReason, Head, ModelRole, Program, State,
    host_sample, reorder,
};

/// One beam search to run. `width` beams are kept; the search generates at most `max_new` tokens.
#[derive(Clone, Debug)]
pub struct BeamRequest {
    /// The prompt's token ids; must not be empty.
    pub prompt: Vec<u32>,
    /// The most new tokens to generate.
    pub max_new: usize,
    /// Beams kept alive at every step.
    pub width: NonZeroUsize,
    /// The exponent `seq_len` is raised to in the score: 1 is plain length normalization, 0 ranks by
    /// the raw cumulative log-probability.
    pub length_penalty: f32,
    /// Treat the end-of-sequence token as an ordinary token: no hypothesis completes on it.
    pub ignore_eos: bool,
}

/// One returned hypothesis.
#[derive(Clone, Debug, PartialEq)]
pub struct Beam {
    /// The generated tokens, including the end-of-sequence token that completed the beam.
    pub tokens: Vec<u32>,
    /// The log-softmax value of each generated token at its step.
    pub logprobs: Vec<f32>,
    /// The sum of `logprobs`.
    pub cum_logprob: f32,
    /// The ranking score: `cum_logprob / seq_len ** length_penalty`.
    pub score: f32,
    /// `Eos` for a completed beam, `MaxTokens` for one still live when the search ended.
    pub finish: FinishReason,
}

/// Why a beam search was refused or failed.
#[derive(Debug, thiserror::Error)]
pub enum BeamError {
    /// Beam search resets and overwrites contiguous state, which an open sequence may be using.
    #[error("beam search needs a driver with no open sequence; {open} are open")]
    Busy {
        /// Sequences currently open on the driver.
        open: usize,
    },
    /// The length penalty is NaN or infinite.
    #[error("the length penalty {0} is not finite")]
    LengthPenalty(f32),
    /// A beam's logits held a NaN or no finite value, so no ranking of its continuations exists.
    #[error("the logits of beam {beam} at generated token {step} are not finite")]
    NonFiniteLogits {
        /// Index of the beam whose logits were not finite.
        beam: usize,
        /// Generated-token index at which it happened.
        step: usize,
    },
    /// Prefill, compile or device failure of the driver under the search.
    #[error(transparent)]
    Driver(#[from] DriverError),
}

impl From<InvalidRequest> for BeamError {
    fn from(error: InvalidRequest) -> Self {
        Self::Driver(error.into())
    }
}

impl From<poot_executor::ExecError> for BeamError {
    fn from(error: poot_executor::ExecError) -> Self {
        Self::Driver(error.into())
    }
}

/// A hypothesis during the search: its generated tokens and their log-probabilities.
#[derive(Clone, Debug)]
struct Hypothesis {
    tokens: Vec<u32>,
    logprobs: Vec<f32>,
    cum_logprob: f32,
}

impl Hypothesis {
    fn child(&self, token: u32, logprob: f32) -> Self {
        let mut tokens = Vec::with_capacity(self.tokens.len() + 1);
        tokens.extend_from_slice(&self.tokens);
        tokens.push(token);
        let mut logprobs = Vec::with_capacity(self.logprobs.len() + 1);
        logprobs.extend_from_slice(&self.logprobs);
        logprobs.push(logprob);
        Self {
            tokens,
            logprobs,
            cum_logprob: self.cum_logprob + logprob,
        }
    }
}

/// How hypotheses rank: the score and the total order it induces.
struct Ranking<'a> {
    prompt_len: usize,
    length_penalty: f32,
    eos: &'a BTreeSet<u32>,
}

impl Ranking<'_> {
    /// `cum_logprob / seq_len ** length_penalty` over the whole token count, prompt included, less
    /// the end-of-sequence token that completed the hypothesis.
    fn score(&self, hypothesis: &Hypothesis) -> f32 {
        let mut seq_len = self.prompt_len + hypothesis.tokens.len();
        if hypothesis
            .tokens
            .last()
            .is_some_and(|token| self.eos.contains(token))
        {
            seq_len -= 1;
        }
        hypothesis.cum_logprob / (seq_len.max(1) as f32).powf(self.length_penalty)
    }

    /// Score descending, then the last token ascending, then the token lists ascending: a total order,
    /// so the returned set is reproducible across platforms. `total_cmp` keeps a NaN from making the
    /// sort inconsistent.
    fn order(&self, a: &Hypothesis, b: &Hypothesis) -> Ordering {
        self.score(b)
            .total_cmp(&self.score(a))
            .then_with(|| a.tokens.last().cmp(&b.tokens.last()))
            .then_with(|| a.tokens.cmp(&b.tokens))
    }
}

/// The `k` highest `(token, logprob)` pairs of `log_softmax(logits)`, by logprob descending then token
/// ascending, or `None` when the logits hold a NaN or no finite value. The softmax is over the whole
/// vocabulary: a truncated and renormalized distribution has log-probabilities that are not comparable
/// across beams.
fn top_logprobs(logits: &[f32], k: usize) -> Option<Vec<(u32, f32)>> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if logits.iter().any(|x| x.is_nan()) || !max.is_finite() {
        return None;
    }
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let lse = max + sum.ln();
    let by_rank = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(token, &x)| (token as u32, x - lse))
        .collect();
    let k = k.min(ranked.len());
    ranked.select_nth_unstable_by(k - 1, by_rank);
    ranked.truncate(k);
    ranked.sort_by(by_rank);
    Some(ranked)
}

impl<S: State> Driver<S> {
    /// Beam search over `request.prompt` ([module docs](self)). The driver must have no open
    /// sequence. Every shape it runs is prepared here, within the prepared-set limits: the prompt's
    /// prefill pieces and the decode step at `rows = width`, and the row gather.
    pub fn beam(&mut self, request: BeamRequest) -> Result<Vec<Beam>, BeamError> {
        let BeamRequest {
            prompt,
            max_new,
            width,
            length_penalty,
            ignore_eos,
        } = request;
        if !length_penalty.is_finite() {
            return Err(BeamError::LengthPenalty(length_penalty));
        }
        if max_new == 0 {
            return Ok(Vec::new());
        }
        self.check_admission(&prompt, max_new)?;
        let open = self.state.open_sequences();
        if open > 0 {
            return Err(BeamError::Busy { open });
        }
        self.prepare_beam(width, prompt.len())?;
        if self.ledger.residency == super::StateResidency::Recurrent {
            // A recurrent state folds every token in: the prompt must start from zero.
            self.reset_exe(self.runtime.exe)?;
        }

        let handle = Arc::clone(&self.handle);
        let no_eos = BTreeSet::new();
        let ranking = Ranking {
            prompt_len: prompt.len(),
            length_penalty,
            eos: if ignore_eos {
                &no_eos
            } else {
                &handle.config().eos
            },
        };
        let breadth = 2 * width.get();
        let mut live = vec![Hypothesis {
            tokens: Vec::new(),
            logprobs: Vec::new(),
            cum_logprob: 0.0,
        }];
        let mut completed: Vec<Hypothesis> = Vec::new();
        let mut logits = self.beam_prefill(width, &prompt)?;
        for step in 0..max_new {
            let vocab = logits.len() / width;
            let mut children: Vec<(usize, Hypothesis)> = Vec::new();
            for (row, beam) in live.iter().enumerate() {
                let candidates = top_logprobs(&logits[row * vocab..(row + 1) * vocab], breadth)
                    .ok_or(BeamError::NonFiniteLogits { beam: row, step })?;
                for (token, logprob) in candidates {
                    let child = beam.child(token, logprob);
                    if ranking.eos.contains(&token) {
                        completed.push(child);
                    } else {
                        children.push((row, child));
                    }
                }
            }
            children.sort_by(|(_, a), (_, b)| ranking.order(a, b));
            children.truncate(width.get());
            if children.is_empty() {
                live.clear();
                break;
            }
            let parents: Vec<usize> = children.iter().map(|&(parent, _)| parent).collect();
            live = children.into_iter().map(|(_, child)| child).collect();
            if step + 1 == max_new {
                break;
            }
            // Rows `i >= live.len()` decode masked tokens and write KV or state nothing reads; the
            // reorder overwrites every row, so the garbage never reaches a live beam.
            self.beam_reorder(width, &parents)?;
            logits = self.beam_decode(width, &prompt, &live)?;
        }

        let mut ranked: Vec<(Hypothesis, FinishReason)> = completed
            .into_iter()
            .map(|beam| (beam, FinishReason::Eos))
            .chain(live.into_iter().map(|beam| (beam, FinishReason::MaxTokens)))
            .collect();
        ranked.sort_by(|(a, _), (b, _)| ranking.order(a, b));
        ranked.truncate(width.get());
        Ok(ranked
            .into_iter()
            .map(|(hypothesis, finish)| Beam {
                score: ranking.score(&hypothesis),
                tokens: hypothesis.tokens,
                logprobs: hypothesis.logprobs,
                cum_logprob: hypothesis.cum_logprob,
                finish,
            })
            .collect())
    }

    /// The beam entry of `tokens` new tokens per row at `rows` rows: contiguous layout, the logits head.
    fn beam_key(&self, rows: NonZeroUsize, tokens: usize) -> EntryKey {
        EntryKey {
            chain: 0,
            role: ModelRole::Beam,
            program: Program::Step,
            shape: StepShape::new(
                rows,
                NonZeroUsize::new(tokens).expect("a step carries a token"),
                self.settings.options.capacity,
            ),
            head: Head::Logits,
            adapters: None,
            carried: InputSet::EMPTY,
        }
    }

    /// Prepare the prefill pieces of a `prompt_len`-token prompt, the decode step and the row gather,
    /// each at `rows` rows.
    fn prepare_beam(&mut self, rows: NonZeroUsize, prompt_len: usize) -> Result<(), DriverError> {
        let mut pieces = self.settings.chunks.plan(prompt_len);
        pieces.push(Piece::Decode);
        for piece in pieces {
            let tokens = match piece {
                Piece::Prefill(n) => n.get(),
                Piece::Decode => 1,
            };
            self.prepared(self.beam_key(rows, tokens))?;
        }
        self.prepare_reorder(rows)
    }

    /// The row gather for `rows` rows, compiled from the compiled decode entry's state avals and added to the
    /// same executable, so it moves the state the decode entry carries.
    fn prepare_reorder(&mut self, rows: NonZeroUsize) -> Result<(), DriverError> {
        if self.ledger.reorders.contains_key(&rows) {
            return Ok(());
        }
        let held = self.prepared_entries();
        let limit = self.settings.options.prepared.max_entries.get();
        if held >= limit {
            return Err(PreparedSetRefusal::Entries { held, limit }.into());
        }
        let decode = self.beam_key(rows, 1);
        // The state the decode entry carries is the compiled program's, at the stored extent the planner chose.
        let carried = self.prepared(decode)?.carried;
        let graph = reorder::reorder_graph(&carried, self.settings.options.trace)
            .map_err(|error| Unsupported::RowGraph(Box::new(error)))?;
        let slots = bind::resolve_passes(&graph, ProgramId::STEP)?;
        let output = graph.aval(graph.result()).shape.clone();
        let program = self.compile(
            &graph,
            &self.runtime.formats,
            &self.settings.options.decode.clone(),
        )?;
        let reservation = self.admit(&program)?;
        let entry = self.add_entry(
            self.runtime.exe,
            &program,
            reservation,
            EntryIo {
                slots,
                output,
                picked: None,
            },
            (decode.program, decode.shape),
        )?;
        self.ledger.reorders.insert(rows, entry);
        Ok(())
    }

    /// Prefill `prompt` in row 0 of the `rows`-row entries, the other rows masked. Returns the
    /// `[rows, vocab]` logits of the prompt's last position; only row 0 holds the prompt's.
    fn beam_prefill(&mut self, rows: NonZeroUsize, prompt: &[u32]) -> Result<Vec<f32>, BeamError> {
        let pieces = self.settings.chunks.plan(prompt.len());
        let mut start = 0;
        let mut logits = Vec::new();
        for (i, piece) in pieces.iter().enumerate() {
            let n = match piece {
                Piece::Prefill(n) => n.get(),
                Piece::Decode => 1,
            };
            let entry = self.ledger.entries[&self.beam_key(rows, n)].clone();
            let mut columns = StepColumns {
                tokens: vec![0; rows.get() * n],
                pos: vec![0; rows.get() * n],
                real: vec![0; rows.get() * n],
                lora: vec![0; rows.get()],
                ..StepColumns::default()
            };
            for j in 0..n {
                columns.tokens[j] = prompt[start + j] as i32;
                columns.pos[j] = (start + j) as i32;
                columns.real[j] = 1;
            }
            let none = RequestInputs::none();
            let feeds = [RowFeed {
                inputs: &none,
                start,
                fed: n,
            }];
            let frame = RequestFrame {
                rows: rows.get(),
                tokens: n,
                side: Side::Left,
                feeds: &feeds,
            };
            let mut outputs = self.execute(&entry, &columns, &frame, &mut NoLate)?;
            if i + 1 == pieces.len() {
                logits = host_sample::read_logits(&mut outputs)?;
            }
            start += n;
        }
        Ok(logits)
    }

    /// One decode step of the live beams, beam `i` in row `i`: each feeds its last token at its
    /// position. Returns the `[rows, vocab]` logits; the rows beyond the live beams are masked.
    fn beam_decode(
        &mut self,
        rows: NonZeroUsize,
        prompt: &[u32],
        live: &[Hypothesis],
    ) -> Result<Vec<f32>, BeamError> {
        let entry = self.ledger.entries[&self.beam_key(rows, 1)].clone();
        let mut columns = StepColumns {
            tokens: vec![0; rows.get()],
            pos: vec![0; rows.get()],
            real: vec![0; rows.get()],
            lora: vec![0; rows.get()],
            ..StepColumns::default()
        };
        for (row, beam) in live.iter().enumerate() {
            let last = *beam
                .tokens
                .last()
                .expect("a live beam has generated a token");
            columns.tokens[row] = last as i32;
            columns.pos[row] = (prompt.len() + beam.tokens.len() - 1) as i32;
            columns.real[row] = 1;
        }
        let none = RequestInputs::none();
        let feeds: Vec<RowFeed<'_>> = (0..live.len())
            .map(|row| RowFeed {
                inputs: &none,
                start: usize::try_from(columns.pos[row]).expect("a position is not negative"),
                fed: 1,
            })
            .collect();
        let frame = RequestFrame {
            rows: rows.get(),
            tokens: 1,
            side: Side::Left,
            feeds: &feeds,
        };
        let mut outputs = self.execute(&entry, &columns, &frame, &mut NoLate)?;
        Ok(host_sample::read_logits(&mut outputs)?)
    }

    /// Move row `parents[i]`'s state to row `i`, for every state value. Rows beyond `parents` keep
    /// their own, so a permutation that changes nothing runs nothing.
    fn beam_reorder(&mut self, rows: NonZeroUsize, parents: &[usize]) -> Result<(), BeamError> {
        let mut index: Vec<i32> = (0..rows.get() as i32).collect();
        for (row, &parent) in parents.iter().enumerate() {
            index[row] = parent as i32;
        }
        if index
            .iter()
            .enumerate()
            .all(|(row, &parent)| row as i32 == parent)
        {
            return Ok(());
        }
        let entry = self.ledger.reorders[&rows].clone();
        let columns = StepColumns {
            beam_rows: index,
            ..StepColumns::default()
        };
        self.execute(&entry, &columns, &RequestFrame::empty(), &mut NoLate)?;
        Ok(())
    }
}
