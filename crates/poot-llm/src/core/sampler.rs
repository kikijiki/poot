//! Sampler: temperature/top-k/top-p/min-p/penalties sampling and logprobs.

use std::collections::HashMap;

use poot_graph_ir::op::{SampleRule, gumbel_noise, random_uniform_at};
use poot_graph_ir::ops::sampling::{Adjust, MASKED_LOGIT};
use poot_text::Constraint;

/// The index of the largest logit (the first on a tie). A [`SamplerFault`] when `v` holds a NaN or `+inf`, or no
/// finite element: an unchecked maximum turns an all-NaN row into token 0.
pub(crate) fn argmax(v: &[f32]) -> Result<usize, SamplerFault> {
    check_pickable(v)?;
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    Ok(best)
}

/// The logits the sampler was given cannot yield a token: a numerical fault upstream, reported instead of
/// served as a completion (ADR-0101 decision 4). NaN and `+inf` are faults wherever they sit in the row.
/// `-inf` is a token masked out (logit bias, a guided-decoding constraint), so it is legal as long as some
/// element is finite.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SamplerFault {
    /// A NaN or `+inf` logit at `index`. The host path (`check_pickable` sees the raw row) always
    /// carries the offending value; a device suffix's readback (card 551b, R-551a-2) carries only
    /// the index, never the value - `None` there.
    #[error("non-finite logit{} at token {index}", value.map(|v| format!(" {v}")).unwrap_or_default())]
    NonFiniteLogit {
        /// Position of the offending logit in the row (the token id).
        index: usize,
        /// The offending value; `None` when only the index survived the device readback.
        value: Option<f32>,
    },
    /// The row is empty or every logit is `-inf`, so no token can be chosen.
    #[error("no finite logit among {len} logits")]
    NoFiniteLogit {
        /// The row's length (0 for an empty row).
        len: usize,
    },
}

/// Checks that `logits` can yield a token: no NaN, no `+inf`, and at least one finite element.
pub(crate) fn check_pickable(logits: &[f32]) -> Result<(), SamplerFault> {
    let mut any_finite = false;
    for (index, &value) in logits.iter().enumerate() {
        if value.is_nan() || value == f32::INFINITY {
            return Err(SamplerFault::NonFiniteLogit {
                index,
                value: Some(value),
            });
        }
        any_finite |= value.is_finite();
    }
    if any_finite {
        Ok(())
    } else {
        Err(SamplerFault::NoFiniteLogit { len: logits.len() })
    }
}

/// Next-token sampling: greedy (temperature 0), or temperature + top-k + top-p (nucleus) sampling with a
/// small deterministic RNG (no `rand` dep). Used by the runner's `*_sampled` generate methods + the server.
pub struct Sampler {
    /// 0 = greedy (argmax); otherwise logits are divided by this before softmax.
    pub temperature: f32,
    /// keep only the top-k logits (0 = no limit).
    pub top_k: usize,
    /// nucleus: keep the smallest set whose probability mass >= top_p (1.0 = no limit).
    pub top_p: f32,
    /// min-p: keep only tokens whose probability is >= min_p * (top token's probability). 0 = no limit.
    pub min_p: f32,
    /// repetition penalty (HF/CTRL convention): a seen token's logit is divided by this if positive,
    /// multiplied if negative. 1.0 = off.
    pub repetition_penalty: f32,
    /// presence penalty (OpenAI): subtracted once from any token seen at least once. 0 = off.
    pub presence_penalty: f32,
    /// frequency penalty (OpenAI): subtracted `count` times from a seen token. 0 = off.
    pub frequency_penalty: f32,
    /// additive per-token logit bias (OpenAI `logit_bias`): a large negative value forbids a token, a
    /// large positive forces it.
    pub logit_bias: HashMap<u32, f32>,
    /// observed token counts (prompt + generated so far); the penalties read this history.
    counts: HashMap<u32, u32>,
    /// when `Some(n)`, record each chosen token's log-probability plus the `n` highest-probability
    /// alternatives (the OpenAI `logprobs` / `top_logprobs` response fields). None = off.
    top_logprobs: Option<usize>,
    /// the per-step logprob records accumulated when `top_logprobs` is set, in generation order.
    recorded: Vec<TokenLogprob>,
    /// guided decoding: when set, each step is masked to the tokens the constraint allows. The constraint
    /// is stateful (it advances as tokens are observed).
    constraint: Option<Constraint>,
    rng: u64,
}

/// One generated token's logprob record: the chosen token, its log-probability, and the top alternatives
/// (token id + log-probability), sorted most-probable first. Log-probabilities are the natural-log
/// softmax over the bias/penalty-adjusted logits at the effective temperature (1.0 when greedy).
#[derive(Clone, Debug)]
pub struct TokenLogprob {
    /// The sampled token id.
    pub token: u32,
    /// Natural-log probability of `token` under the adjusted distribution.
    pub logprob: f32,
    /// The highest-probability alternatives as `(token id, logprob)`, most probable first; at most the requested `top_logprobs`.
    pub top: Vec<(u32, f32)>,
}

impl Sampler {
    /// A greedy sampler: temperature 0 (argmax), no truncation, no penalties, no bias, no logprob recording, fixed RNG state.
    pub fn greedy() -> Self {
        Sampler {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: HashMap::new(),
            counts: HashMap::new(),
            top_logprobs: None,
            recorded: Vec::new(),
            constraint: None,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// True when the sampler is a plain argmax: no logit adjustment, no logprob recording and no guided
    /// decoding. Only such a row can ride a verify window, whose positions pick without a history of
    /// their own.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
            && !self.adjusts_logits()
            && self.top_logprobs.is_none()
            && self.constraint.is_none()
    }

    /// The draw this row asks of the device sampling head: the argmax at temperature 0, otherwise the
    /// rule its truncation knobs select ([`SampleRule::for_truncation`], the one classification the host
    /// reference and the device share).
    pub fn draw_rule(&self) -> SampleRule {
        if self.temperature <= 0.0 {
            SampleRule::Greedy
        } else {
            SampleRule::for_truncation(self.top_k, self.top_p)
        }
    }

    /// What the row needs the head to do to the logits: nothing, apply its penalties and bias, or also
    /// mask to a guided-decoding constraint.
    pub fn adjustment(&self) -> Adjust {
        if self.constraint.is_some() {
            Adjust::Guided
        } else if self.adjusts_logits() {
            Adjust::History
        } else {
            Adjust::None
        }
    }

    /// The alternatives per token the row records, `None` when it records no logprobs.
    pub fn logprob_count(&self) -> Option<usize> {
        self.top_logprobs
    }

    /// The `(repetition, presence, frequency)` penalties; `(1, 0, 0)` when none is set.
    pub(crate) fn penalty_row(&self) -> [f32; 3] {
        [
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
        ]
    }

    /// The row's device-resident history over a `vocab`-token vocabulary: `[counts, bias]`, each `vocab`
    /// long. The counts are the prompt and the tokens observed so far; the bias is the logit bias. A token
    /// id past the vocabulary is dropped, as the host adjustment does.
    pub(crate) fn history_row(&self, vocab: usize) -> Vec<f32> {
        let mut row = vec![0.0f32; 2 * vocab];
        for (&token, &count) in &self.counts {
            if let Some(slot) = row
                .get_mut(token as usize)
                .filter(|_| (token as usize) < vocab)
            {
                *slot = count as f32;
            }
        }
        for (&token, &bias) in &self.logit_bias {
            if (token as usize) < vocab {
                row[vocab + token as usize] = bias;
            }
        }
        row
    }

    /// The additive guided-decoding mask of the next step over `vocab` tokens (`0` for a permitted token,
    /// [`MASKED_LOGIT`] for an excluded one), `None` for a row without a constraint. It is the host mask
    /// of [`Self::pick`] with the finite stand-in the device draw takes for `-inf`.
    pub(crate) fn guided_mask(&self, vocab: usize) -> Option<Vec<f32>> {
        let keep = self.constraint_keep(vocab)?;
        Some(
            keep.into_iter()
                .map(|keep| if keep { 0.0 } else { MASKED_LOGIT })
                .collect(),
        )
    }

    /// Record the logprobs the device head computed for the token it drew, as [`Self::pick`] records
    /// them on the host.
    pub(crate) fn push_logprob(&mut self, record: TokenLogprob) {
        self.recorded.push(record);
    }

    /// Whether any penalty or bias would change the raw logits (and thus the argmax).
    fn adjusts_logits(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || !self.logit_bias.is_empty()
    }

    /// A sampler at `temperature` with `top_k` and `top_p` truncation (`temperature` clamped to >= 0, `top_p` to 0..=1) and an RNG state derived from `seed`; every other knob starts as in [`Self::greedy`].
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Sampler {
            temperature: temperature.max(0.0),
            top_k,
            top_p: top_p.clamp(0.0, 1.0),
            rng: Self::seed_to_rng_state(seed),
            ..Sampler::greedy()
        }
    }

    /// Maps a caller-provided `seed` to a nonzero xorshift64 state. The old `seed | 1` collided any two
    /// seeds differing only in bit 0 (e.g. `(2k, 2k+1)`), so the server's per-choice seeding
    /// `base_seed.wrapping_add(i)` (n>1 sampling) gave byte-identical completions for adjacent choices when
    /// `base_seed` was even (the default 0). The seed is mixed with splitmix64 first, with a fixed nonzero
    /// constant if the mix itself is 0 (xorshift64 is stuck at zero forever). The same seed always maps to
    /// the same state.
    fn seed_to_rng_state(seed: u64) -> u64 {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z }
    }

    /// Set the OpenAI-style penalties (builder style). `repetition` is the HF/CTRL multiplicative penalty
    /// (1.0 = off); `presence`/`frequency` are the additive OpenAI penalties (0.0 = off).
    pub fn with_penalties(mut self, repetition: f32, presence: f32, frequency: f32) -> Self {
        self.repetition_penalty = if repetition > 0.0 { repetition } else { 1.0 };
        self.presence_penalty = presence;
        self.frequency_penalty = frequency;
        self
    }

    /// Set the min-p floor (builder style); clamped to [0, 1].
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p.clamp(0.0, 1.0);
        self
    }

    /// Set the additive per-token logit bias map (builder style).
    pub fn with_logit_bias(mut self, bias: HashMap<u32, f32>) -> Self {
        self.logit_bias = bias;
        self
    }

    /// Enable logprob recording (builder style): record each chosen token's log-probability plus the `n`
    /// highest-probability alternatives. `n = 0` records just the chosen token's logprob.
    pub fn with_logprobs(mut self, n: usize) -> Self {
        self.top_logprobs = Some(n);
        self
    }

    /// Take the accumulated per-step logprob records (clears the internal buffer). Empty when logprob
    /// recording was not enabled.
    pub fn take_logprobs(&mut self) -> Vec<TokenLogprob> {
        std::mem::take(&mut self.recorded)
    }

    /// Enable guided decoding (builder style): mask each step to the tokens the constraint allows.
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraint = Some(constraint);
        self
    }

    /// Reset the RNG seed (builder style). Used to give each choice of an `n>1` request a distinct seed so
    /// sampled choices differ (greedy choices are identical regardless, which is acceptable).
    pub fn reseed(mut self, seed: u64) -> Self {
        self.rng = Self::seed_to_rng_state(seed);
        self
    }

    /// Records a generated token: bumps the penalty history and advances the guided-decoding constraint.
    /// Called by the decode loops for each emitted token (not prompt context; see `seed_context`).
    pub fn observe(&mut self, token: u32) {
        *self.counts.entry(token).or_insert(0) += 1;
        if let Some(c) = self.constraint.as_mut() {
            c.advance(token);
        }
    }

    /// Seed the penalty history with the prompt context before decoding. The prompt does NOT advance the
    /// guided-decoding constraint (which tracks only generated tokens), so this bumps counts directly.
    pub fn seed_context(&mut self, tokens: &[u32]) {
        for &t in tokens {
            *self.counts.entry(t).or_insert(0) += 1;
        }
    }

    /// One step of the sampler's xorshift64 stream, advancing `self.rng` and returning the raw
    /// 64-bit state. Shared by [`Self::uniform`] (speculative decoding's own draw, card 094) and
    /// [`Self::next_device_seed`] (card 551b's suffix/host-draw seed): one PRNG stream per
    /// `Sampler`, two different consumers.
    fn next_u64(&mut self) -> u64 {
        let mut s = self.rng;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.rng = s;
        s
    }

    /// A uniform f32 in [0, 1) from an xorshift64 step. `pub(crate)` for speculative sampling's accept-test
    /// draw (`driver::speculate::spec_sample_round`).
    pub(crate) fn uniform(&mut self) -> f32 {
        let s = self.next_u64();
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// The next decode step's `RandomUniform` seed (card 551b, R-551b-2): both the host pick
    /// ([`Self::pick_from`]) and a device suffix row ([`crate::driver::suffix::SuffixRows::push`]) consume
    /// one per decode step, so a seeded `Sampler` on either path draws the identical per-step noise. A
    /// distinct draw from [`Self::uniform`]'s own stream position (callers never mix the two per step).
    pub(crate) fn next_device_seed(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Apply the logit_bias and the repetition / presence / frequency penalties to a mutable logits copy,
    /// in place. Only called on the slow path (`adjusts_logits()`); the order is bias then penalties.
    fn adjust(&self, logits: &mut [f32]) {
        for (&tok, &bias) in &self.logit_bias {
            if let Some(l) = logits.get_mut(tok as usize) {
                *l += bias;
            }
        }
        if self.repetition_penalty == 1.0
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
        {
            return;
        }
        for (&tok, &count) in &self.counts {
            let Some(l) = logits.get_mut(tok as usize) else {
                continue;
            };
            if self.repetition_penalty != 1.0 {
                *l = if *l > 0.0 {
                    *l / self.repetition_penalty
                } else {
                    *l * self.repetition_penalty
                };
            }
            *l -= self.presence_penalty;
            *l -= self.frequency_penalty * count as f32;
        }
    }

    /// Choose the next token id from `logits`. Applies logit_bias + penalties (slow path only), then the
    /// greedy argmax or the temperature + min-p / top-k / top-p draw. Records the logprob of the chosen
    /// token + top alternatives when logprob recording is enabled. A [`SamplerFault`] when the logits (raw,
    /// or after bias/penalties/constraint) hold a NaN or `+inf`, or no finite element.
    pub fn pick(&mut self, logits: &[f32]) -> Result<usize, SamplerFault> {
        // Fast path: no penalties/bias, no logprob recording, no constraint means no per-step allocation
        // (keeps greedy/temp decode cheap).
        if !self.adjusts_logits() && self.constraint.is_none() {
            let chosen = self.pick_from(logits)?;
            if self.top_logprobs.is_some() {
                self.record_logprobs(logits, chosen);
            }
            return Ok(chosen);
        }
        // The bias and the constraint mask overwrite logits, which would hide a fault in the raw row.
        check_pickable(logits)?;
        let mut adj = logits.to_vec();
        if self.adjusts_logits() {
            self.adjust(&mut adj);
        }
        if self.constraint.is_some() {
            self.apply_constraint_mask(&mut adj);
        }
        let chosen = self.pick_from(&adj)?;
        if self.top_logprobs.is_some() {
            self.record_logprobs(&adj, chosen);
        }
        Ok(chosen)
    }

    /// Whether each of the `vocab` tokens is permitted by the guided-decoding constraint at the current
    /// generated prefix, `None` without a constraint. EOS is allowed once a choice is complete; if nothing
    /// continues (the constraint is exhausted), EOS is forced so the masked logits are never all excluded
    /// (FR-003). The one definition the host mask and the device mask share.
    fn constraint_keep(&self, vocab: usize) -> Option<Vec<bool>> {
        let constraint = self.constraint.as_ref()?;
        let (allowed, eos_ok) = constraint.allowed_next();
        let eos_ok = eos_ok || allowed.is_empty();
        let mut keep = vec![false; vocab];
        for &id in &allowed {
            if let Some(slot) = keep.get_mut(id as usize) {
                *slot = true;
            }
        }
        if eos_ok && let Some(slot) = keep.get_mut(constraint.eos() as usize) {
            *slot = true;
        }
        Some(keep)
    }

    /// Mask `logits` to the tokens the guided-decoding constraint permits ([`Self::constraint_keep`]):
    /// excluded ids become -inf.
    fn apply_constraint_mask(&self, logits: &mut [f32]) {
        let keep = self
            .constraint_keep(logits.len())
            .expect("a mask is applied only under a constraint");
        for (l, keep) in logits.iter_mut().zip(keep) {
            if !keep {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// Push one logprob record: the natural-log softmax of `logits` (the bias/penalty-adjusted logits) at
    /// the effective temperature (1.0 when greedy), for the chosen token plus the top-n alternatives.
    fn record_logprobs(&mut self, logits: &[f32], chosen: usize) {
        let n = self.top_logprobs.unwrap_or(0);
        let t = if self.temperature > 0.0 {
            self.temperature
        } else {
            1.0
        };
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // log-sum-exp over the scaled logits, so logprob(i) = (logit_i - max)/t - lse.
        let sumexp: f32 = logits.iter().map(|&l| ((l - max) / t).exp()).sum();
        let lse = sumexp.ln();
        let logprob_of = |i: usize| (logits[i] - max) / t - lse;
        // top-(n+1) tokens by logit (monotone in logprob): a bounded insertion keeps it O(vocab * n).
        let mut top: Vec<(usize, f32)> = Vec::with_capacity(n + 2);
        for (i, &l) in logits.iter().enumerate() {
            if top.len() < n + 1 || l > top.last().unwrap().1 {
                let at = top.partition_point(|&(_, tl)| tl >= l);
                top.insert(at, (i, l));
                top.truncate(n + 1);
            }
        }
        let alts: Vec<(u32, f32)> = top
            .iter()
            .take(n)
            .map(|&(i, _)| (i as u32, logprob_of(i)))
            .collect();
        self.recorded.push(TokenLogprob {
            token: chosen as u32,
            logprob: logprob_of(chosen),
            top: alts,
        });
    }

    /// The min-p / top-k / top-p truncated, unnormalized candidate list (`(token id, exp((logit-max)/T))`,
    /// the top token's weight exactly 1) for a temperature > 0 draw, in `(logit descending, index
    /// ascending)` order: the contract's tie rule ([`SampleRule`]). The exact reference
    /// distribution (ADR-0114 amendment): weights and the nucleus mass are summed in f64, the top-p
    /// nucleus is the smallest prefix whose mass reaches `top_p` of the survivors' total, and no
    /// quantization or search threshold stands between the logits and the set. Which truncations apply is
    /// [`SampleRule::for_truncation`], the classification the device suffix reads. Shared by
    /// [`Self::pick_from`] (one inverse-CDF draw) and [`Self::probs`] (the full distribution for
    /// speculative sampling's accept/reject math), so the two cannot drift. Greedy (temperature <= 0) is
    /// the caller's job (a one-hot distribution); `-inf` logits (masked tokens) carry no weight.
    fn truncated_candidates(&self, logits: &[f32]) -> Vec<(usize, f64)> {
        let rule = SampleRule::for_truncation(self.top_k, self.top_p);
        let max = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let temperature = f64::from(self.temperature);
        let mut order: Vec<usize> = (0..logits.len()).collect();
        // `check_pickable` admits no NaN, so the comparison is total; the sort is stable, so ties keep
        // index order.
        order.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cand: Vec<(usize, f64)> = order
            .into_iter()
            .map(|i| (i, ((f64::from(logits[i]) - max) / temperature).exp()))
            .filter(|&(_, w)| w > 0.0)
            .collect();
        // min-p: weights are relative to the top token's (1), so the floor is `min_p` itself.
        if self.min_p > 0.0 {
            let floor = f64::from(self.min_p);
            cand.truncate(cand.partition_point(|c| c.1 >= floor).max(1));
        }
        // `GumbelTopKTopP` also declares the top-k operand when `top_k` is 0 (disabled), so the rule alone
        // does not say a cut applies.
        if rule.has_top_k() && self.top_k > 0 {
            cand.truncate(self.top_k);
        }
        if rule.has_top_p() {
            if self.top_p <= 0.0 {
                cand.truncate(1);
            } else {
                let target = f64::from(self.top_p) * cand.iter().map(|c| c.1).sum::<f64>();
                let mut cum = 0.0f64;
                let cut = cand
                    .iter()
                    .position(|c| {
                        cum += c.1;
                        cum >= target
                    })
                    .map_or(cand.len(), |i| i + 1);
                cand.truncate(cut);
            }
        }
        cand
    }

    /// The temperature + truncation draw over (already-adjusted) logits: the Gumbel-max trick over
    /// [`Self::truncated_candidates`], the exact distribution, as a plain function of the logits, the knobs
    /// and this step's [`Self::next_device_seed`] (no graph, no evaluator: ADR-0102 decision 5). The
    /// noise is `RandomUniform`'s own value at the token's index and the score is the rule's
    /// `logits[i] * inv_temp + noise[i]` (two roundings), ties to the lowest index (`poot-graph-ir`'s
    /// [`SampleRule`] semantics), so given the same kept set a device suffix row picks the same token
    /// from the same seed. The kept sets differ only at a top-p boundary the device's integer-mass
    /// bisection places within the ADR-0114 tolerance.
    fn pick_from(&mut self, logits: &[f32]) -> Result<usize, SamplerFault> {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        check_pickable(logits)?;
        let cand = self.truncated_candidates(logits);
        let seed = self.next_device_seed();
        let inv_temp = 1.0 / self.temperature;
        let mut best: Option<(usize, f32)> = None;
        for &(i, _) in &cand {
            let noise = gumbel_noise(random_uniform_at(seed, i as u32));
            let score = logits[i] * inv_temp + noise;
            if best.is_none_or(|(b, v)| score > v || (score == v && i < b)) {
                best = Some((i, score));
            }
        }
        Ok(best.expect("the top token always survives").0)
    }

    /// The full-vocab probability distribution [`Self::pick_from`] draws from, given already-adjusted logits
    /// (bias/penalties are the caller's job, as in [`Self::pick`]). Building block for speculative sampling
    /// (card 094), which needs each drafted token's probability under this sampler. Temperature <= 0 is a
    /// one-hot distribution on the argmax. Entries outside the min-p/top-k/top-p candidate set are exactly
    /// 0; kept entries are normalized to sum to 1. Pure (`&self`, no rng draw); the same truncation as
    /// [`Self::pick_from`], materialized as a vector. A [`SamplerFault`] on the logits [`Self::pick`] rejects.
    /// Speculative sampling's target `p` over a window's logits (`driver::speculate`).
    pub fn probs(&self, logits: &[f32]) -> Result<Vec<f32>, SamplerFault> {
        let mut v = vec![0.0f32; logits.len()];
        if self.temperature <= 0.0 {
            v[argmax(logits)?] = 1.0;
            return Ok(v);
        }
        check_pickable(logits)?;
        let cand = self.truncated_candidates(logits);
        let total: f64 = cand.iter().map(|c| c.1).sum();
        for &(i, w) in &cand {
            v[i] = (w / total) as f32;
        }
        Ok(v)
    }

    /// Draws a token id from an arbitrary full-vocab probability vector (normalized to sum to 1, e.g. from
    /// [`Self::probs`], or the residual `max(0, p-q)` speculative sampling resamples from after a rejected
    /// draft): the inverse-CDF draw `pick_from` does internally. Falls back to `probs`'s argmax if the mass
    /// is numerically <= 0 (a float-rounding safety net). A [`SamplerFault`] when `probs` holds a NaN or
    /// infinity.
    pub fn sample_from_probs(&mut self, probs: &[f32]) -> Result<u32, SamplerFault> {
        check_pickable(probs)?;
        let total: f32 = probs.iter().sum();
        if total <= 0.0 {
            return Ok(argmax(probs)? as u32);
        }
        let r = self.uniform() * total;
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum {
                return Ok(i as u32);
            }
        }
        Ok((probs.len().saturating_sub(1)) as u32)
    }

    /// Derives an independent but deterministic `Sampler` sharing this one's temperature/top_k/top_p/min_p
    /// (no penalties/bias/logprobs/constraint), reseeded by mixing this sampler's current rng state with
    /// `salt`. Sampled speculative decoding (card 094) needs two independent draw streams per request (the
    /// draft's proposals, the target's accept-test/residual draws), reproducible from one request seed. Call
    /// once before the round loop, not per round, or the streams would resync every round.
    pub fn fork(&self, salt: u64) -> Sampler {
        Sampler::new(self.temperature, self.top_k, self.top_p, self.rng ^ salt)
            .with_min_p(self.min_p)
    }
}

#[cfg(test)]
mod device_head_tests;

#[cfg(test)]
mod sampler_tests {
    use std::collections::HashMap;

    use poot_graph_ir::op::SampleRule;
    use poot_graph_ir::ops::sampling::Adjust;

    use super::argmax;
    use super::{Constraint, Sampler, SamplerFault};

    /// `result` is the NaN fault at `index`.
    fn assert_nan_at<T: std::fmt::Debug>(result: Result<T, SamplerFault>, index: usize) {
        match result {
            Err(SamplerFault::NonFiniteLogit { index: got, value }) => {
                assert_eq!(got, index);
                assert!(
                    value.is_some_and(|v| v.is_nan()),
                    "expected a NaN fault, got {value:?}"
                );
            }
            other => panic!("expected a NaN sampler fault at token {index}, got {other:?}"),
        }
    }

    /// SC-001: an all-NaN row is a typed fault from the greedy pick, the sampled pick, `probs` and `argmax`.
    /// The unchecked maximum returned token 0 for every one of these.
    #[test]
    fn all_nan_logits_are_a_sampler_fault_on_every_entry_point() {
        let logits = [f32::NAN; 6];
        assert_nan_at(Sampler::greedy().pick(&logits), 0);
        assert_nan_at(Sampler::new(0.8, 0, 1.0, 7).pick(&logits), 0);
        assert_nan_at(Sampler::greedy().probs(&logits), 0);
        assert_nan_at(Sampler::new(0.8, 0, 1.0, 7).probs(&logits), 0);
        assert_nan_at(argmax(&logits), 0);
        assert_nan_at(Sampler::greedy().sample_from_probs(&logits), 0);
    }

    /// A NaN beside finite logits is still a fault: the unchecked maximum skipped it and served the finite
    /// argmax, hiding the numerical fault.
    #[test]
    fn a_nan_beside_finite_logits_is_still_a_sampler_fault() {
        let logits = [0.1, 3.0, f32::NAN, 2.9];
        assert_nan_at(Sampler::greedy().pick(&logits), 2);
        assert_nan_at(Sampler::new(0.8, 0, 1.0, 7).pick(&logits), 2);
        assert_nan_at(Sampler::new(0.8, 0, 1.0, 7).probs(&logits), 2);
        assert_nan_at(argmax(&logits), 2);
    }

    #[test]
    fn positive_infinity_is_a_fault_and_negative_infinity_is_a_masked_token() {
        let inf = [0.1, f32::INFINITY, 0.5];
        for result in [
            Sampler::greedy().pick(&inf),
            Sampler::new(0.8, 0, 1.0, 7).pick(&inf),
        ] {
            assert_eq!(
                result,
                Err(SamplerFault::NonFiniteLogit {
                    index: 1,
                    value: Some(f32::INFINITY)
                })
            );
        }
        // -inf is a token masked out, legal beside a finite logit and never drawn.
        let masked = [f32::NEG_INFINITY, 0.5, f32::NEG_INFINITY, 0.2];
        assert_eq!(Sampler::greedy().pick(&masked), Ok(1));
        let p = Sampler::new(1.0, 0, 1.0, 7).probs(&masked).unwrap();
        assert_eq!(p[0], 0.0);
        assert_eq!(p[2], 0.0);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        for seed in 0..64 {
            let pick = Sampler::new(1.0, 0, 1.0, seed).pick(&masked).unwrap();
            assert!(pick == 1 || pick == 3, "drew masked token {pick}");
        }
    }

    #[test]
    fn a_row_without_a_finite_logit_has_no_token() {
        for logits in [&[][..], &[f32::NEG_INFINITY; 4][..]] {
            let want = Err(SamplerFault::NoFiniteLogit { len: logits.len() });
            assert_eq!(Sampler::greedy().pick(logits), want);
            assert_eq!(Sampler::new(0.8, 0, 1.0, 7).pick(logits), want);
            assert_eq!(argmax(logits), want);
        }
    }

    /// The bias, penalties and guided-decoding mask overwrite logits before the draw; a NaN there must not be
    /// hidden by the overwrite (a masked NaN became -inf and the pick succeeded).
    #[test]
    fn a_nan_under_a_constraint_mask_or_penalty_is_still_a_sampler_fault() {
        let logits = [0.1, 3.0, 0.5, f32::NAN];
        let mut constrained =
            Sampler::greedy().with_constraint(Constraint::choices(vec![vec![1u32]], 2));
        assert_nan_at(constrained.pick(&logits), 3);
        let mut penalized = Sampler::greedy().with_penalties(1.0, 0.5, 0.0);
        assert_nan_at(penalized.pick(&logits), 3);
    }

    #[test]
    fn greedy_picks_argmax() {
        let logits = [0.1, 3.0, 0.5, 2.9];
        let mut s = Sampler::greedy();
        assert_eq!(s.pick(&logits).unwrap(), 1);
    }

    #[test]
    fn top_k_one_is_argmax_regardless_of_temperature() {
        let logits = [0.1, 3.0, 0.5, 2.9];
        let mut s = Sampler::new(1.0, 1, 1.0, 42);
        // top_k=1 keeps only the max logit, so the draw is forced to it.
        for _ in 0..16 {
            assert_eq!(s.pick(&logits).unwrap(), 1);
        }
    }

    #[test]
    fn same_seed_is_deterministic() {
        let logits = [1.0, 1.0, 1.0, 1.0, 1.0];
        let mut a = Sampler::new(1.0, 0, 1.0, 7);
        let mut b = Sampler::new(1.0, 0, 1.0, 7);
        let pa: Vec<usize> = (0..20).map(|_| a.pick(&logits).unwrap()).collect();
        let pb: Vec<usize> = (0..20).map(|_| b.pick(&logits).unwrap()).collect();
        assert_eq!(pa, pb);
    }

    #[test]
    fn reseed_per_choice_streams_differ_like_the_server_n_gt_1_path() {
        // The old seed->rng-state mapping `seed | 1` collided any two seeds differing only in bit 0 (e.g. (0,1),
        // (2,3)). The server builds each `n>1` choice's seed as `base_seed.wrapping_add(i)` (`reseed()` call
        // sites in crates/poot-serve/src/main.rs), so with the default `base_seed=0` choices 0 and 1 drew the
        // same token stream. This reproduces that pattern for two even bases (0 and 4) and asserts the
        // per-choice draws differ. Reverting `Sampler::seed_to_rng_state` to `seed | 1` makes it fail.
        let logits = [0.3, 1.7, -0.4, 2.1, 0.9, -1.2, 0.5, 1.1];
        let draw_choice = |base: u64, i: u64| -> Vec<usize> {
            let mut s = Sampler::new(0.9, 0, 1.0, 0).reseed(base.wrapping_add(i));
            (0..8).map(|_| s.pick(&logits).unwrap()).collect()
        };
        for &base in &[0u64, 4u64] {
            let streams: Vec<Vec<usize>> = (0..3).map(|i| draw_choice(base, i)).collect();
            for i in 0..streams.len() {
                for j in (i + 1)..streams.len() {
                    assert_ne!(
                        streams[i],
                        streams[j],
                        "base_seed={base}: choices {i} and {j} (seeds {}, {}) drew identical streams \
                         {:?} - n>1 sampled choices must not collide",
                        base.wrapping_add(i as u64),
                        base.wrapping_add(j as u64),
                        streams[i]
                    );
                }
            }
        }

        // same-seed determinism must still hold (unchanged behavior for repeated requests).
        let mut r1 = Sampler::new(0.9, 0, 1.0, 0).reseed(123);
        let mut r2 = Sampler::new(0.9, 0, 1.0, 0).reseed(123);
        let d1: Vec<usize> = (0..8).map(|_| r1.pick(&logits).unwrap()).collect();
        let d2: Vec<usize> = (0..8).map(|_| r2.pick(&logits).unwrap()).collect();
        assert_eq!(d1, d2, "same seed must still be deterministic");
    }

    #[test]
    fn top_p_restricts_to_nucleus() {
        // one dominant logit: top_p=0.5 should keep only it, so the draw is forced to it.
        let logits = [10.0, 0.0, 0.0, 0.0];
        let mut s = Sampler::new(1.0, 0, 0.5, 99);
        for _ in 0..16 {
            assert_eq!(s.pick(&logits).unwrap(), 0);
        }
    }

    #[test]
    fn repetition_penalty_suppresses_seen_token() {
        // token 1 is the clear argmax; once it is in history a large repetition penalty pulls its logit
        // down (positive -> divided by 100) so greedy picks the next-best token 3 instead (SC-001/FR-002).
        let logits = [0.1, 3.0, 0.5, 2.9];
        let mut s = Sampler::greedy().with_penalties(100.0, 0.0, 0.0);
        assert_eq!(s.pick(&logits).unwrap(), 1); // no history yet -> argmax
        s.observe(1);
        assert_eq!(s.pick(&logits).unwrap(), 3); // token 1 penalized -> next-best
    }

    #[test]
    fn presence_and_frequency_penalize_history() {
        // token 0 and 1 nearly tie; token 0 was seen 3 times. A frequency penalty scales with the count,
        // dragging token 0 below token 1 (FR-003).
        let logits = [2.0, 1.9, 0.0];
        let mut s = Sampler::greedy().with_penalties(1.0, 0.0, 0.5);
        for _ in 0..3 {
            s.observe(0);
        }
        // token 0: 2.0 - 0.5*3 = 0.5; token 1: 1.9 -> picked.
        assert_eq!(s.pick(&logits).unwrap(), 1);
    }

    #[test]
    fn min_p_trims_the_tail() {
        // dominant token 0; min_p=0.5 drops everything below half its probability, forcing the draw to 0.
        let logits = [10.0, 0.0, 0.0, 0.0];
        let mut s = Sampler::new(1.0, 0, 1.0, 5).with_min_p(0.5);
        for _ in 0..16 {
            assert_eq!(s.pick(&logits).unwrap(), 0);
        }
    }

    /// R486-006 (RU-5): the top-p cut is inclusive (`cum >= top_p * total`). Two tokens at the same logit
    /// have weight exactly 1 each, so the running mass after the first is exactly half of the total in
    /// f64: `top_p = 0.5` keeps the first (lower index) and stops. The `>=` -> `>` mutation takes the
    /// second too and the literal candidate set changes.
    #[test]
    fn top_p_boundary_is_inclusive() {
        let s = Sampler::new(1.0, 0, 0.5, 11);
        assert_eq!(
            s.truncated_candidates(&[0.0, 0.0]),
            vec![(0, 1.0)],
            "mass exactly equal to top_p ends the nucleus (cum >= top_p), ties by lower index first"
        );
    }

    /// R486-006 (RU-6): the min-p cut drops only weights strictly below `min_p * top`; a token equal to
    /// the floor survives. With `min_p = 1.0` the floor is exactly the top weight `1.0`, and the second
    /// token (the same logit) sits on it. The `>=` -> `>` mutation drops it and the literal candidate set
    /// changes.
    #[test]
    fn min_p_boundary_keeps_tokens_equal_to_the_floor() {
        let s = Sampler::new(1.0, 0, 1.0, 11).with_min_p(1.0);
        assert_eq!(
            s.truncated_candidates(&[0.0, 0.0, -1.0]),
            vec![(0, 1.0), (1, 1.0)],
            "a token equal to the min-p floor survives (weight < floor drops)"
        );
    }

    /// What a request asks of the device head is read off the sampler: the draw rule from the
    /// temperature and the truncation knobs, the adjustment from any penalty, bias or constraint, the
    /// alternatives from the logprob count. Every row, whatever it asks, has a head; none is a host pick.
    #[test]
    fn the_sampler_names_what_it_asks_of_the_head() {
        let ask = |s: &Sampler| (s.draw_rule(), s.adjustment(), s.logprob_count());
        assert_eq!(
            ask(&Sampler::greedy()),
            (SampleRule::Greedy, Adjust::None, None)
        );
        assert_eq!(
            ask(&Sampler::new(0.0, 5, 0.9, 1)),
            (SampleRule::Greedy, Adjust::None, None),
            "temperature 0 is the argmax whatever the truncation knobs say"
        );
        assert_eq!(
            ask(&Sampler::new(0.8, 0, 1.0, 1)),
            (SampleRule::Gumbel, Adjust::None, None)
        );
        assert_eq!(
            ask(&Sampler::new(0.8, 0, 1.0, 1).with_min_p(0.5)),
            (SampleRule::Gumbel, Adjust::None, None),
            "min-p rides in every Gumbel rule's params"
        );
        assert_eq!(
            ask(&Sampler::new(0.8, 5, 1.0, 1)),
            (SampleRule::GumbelTopK, Adjust::None, None)
        );
        assert_eq!(
            ask(&Sampler::new(0.8, 0, 0.9, 1)),
            (SampleRule::GumbelTopKTopP, Adjust::None, None)
        );
        assert_eq!(
            ask(&Sampler::new(0.8, 5, 0.9, 1)),
            (SampleRule::GumbelTopKTopP, Adjust::None, None)
        );
        assert_eq!(
            ask(&Sampler::new(0.8, 0, 1.0, 1).with_penalties(1.2, 0.0, 0.0)),
            (SampleRule::Gumbel, Adjust::History, None)
        );
        assert_eq!(
            ask(&Sampler::new(0.8, 0, 1.0, 1).with_logprobs(3)),
            (SampleRule::Gumbel, Adjust::None, Some(3))
        );
        assert_eq!(
            ask(&Sampler::new(0.8, 0, 1.0, 1)
                .with_logit_bias(std::collections::HashMap::from([(0u32, -1.0)]))),
            (SampleRule::Gumbel, Adjust::History, None),
            "a bias reshapes the argmax: the adjusted head applies it"
        );
        assert_eq!(
            ask(&Sampler::greedy().with_constraint(Constraint::choices(vec![vec![1u32]], 2))),
            (SampleRule::Greedy, Adjust::Guided, None),
            "a guided row masks the logits on the device"
        );
    }

    /// The device history row is the host sampler's own: the counts it observed (the prompt and the
    /// committed tokens) and its logit bias, over the vocabulary, dropping an id past it.
    #[test]
    fn the_history_row_is_the_counts_and_the_bias() {
        let mut s = Sampler::greedy().with_logit_bias(HashMap::from([(1u32, 0.5f32), (9, 2.0)]));
        s.seed_context(&[3, 3, 7]);
        s.observe(3);
        s.observe(9);
        let row = s.history_row(8);
        assert_eq!(row.len(), 16);
        assert_eq!(&row[..8], &[0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(&row[8..], &[0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// `top_p = 0.0` must mean "the minimal nucleus" (top-1), not "disabled": `pick_from`'s `top_p < 1.0`
    /// gate with `cum >= top_p` firing on the first candidate already does this. `pick`/`pick_from` picks
    /// the argmax at top_p=0.0 over many random logit rows and seeds.
    #[test]
    fn top_p_zero_collapses_to_top1_on_the_host_path() {
        fn xs(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        let mut seed = 0x0700_c0ff_ee00_u64;
        for _ in 0..100u64 {
            let vocab = 2 + (xs(&mut seed) % 200) as usize;
            let logits: Vec<f32> = (0..vocab)
                .map(|_| {
                    let u = (xs(&mut seed) >> 40) as f32 / (1u64 << 24) as f32;
                    u * 20.0 - 10.0
                })
                .collect();
            let (argmax, _) = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let temp = 0.1 + (xs(&mut seed) % 1000) as f32 / 1000.0 * 1.9;
            let rng_seed = xs(&mut seed);
            let mut s = Sampler::new(temp, 0, 0.0, rng_seed);
            let pick = s.pick(&logits).unwrap();
            assert_eq!(
                pick, argmax,
                "top_p=0.0 must collapse to the argmax (vocab={vocab} temp={temp})"
            );
        }
    }

    #[test]
    fn logit_bias_forbids_and_forces() {
        let logits = [0.1, 3.0, 0.5, 2.9];
        // a large negative bias on the argmax (token 1) forces the next-best (token 3).
        let mut forbid =
            Sampler::greedy().with_logit_bias(std::collections::HashMap::from([(1u32, -1e9)]));
        assert_eq!(forbid.pick(&logits).unwrap(), 3);
        // a large positive bias forces an otherwise-unlikely token (token 0).
        let mut force =
            Sampler::greedy().with_logit_bias(std::collections::HashMap::from([(0u32, 1e9)]));
        assert_eq!(force.pick(&logits).unwrap(), 0);
    }

    #[test]
    fn bias_makes_sampler_non_greedy() {
        // an active bias must force the host-side logits readback path (FR-006).
        assert!(Sampler::greedy().is_greedy());
        assert!(
            !Sampler::greedy()
                .with_logit_bias(std::collections::HashMap::from([(0u32, -1.0)]))
                .is_greedy()
        );
        assert!(!Sampler::greedy().with_penalties(1.1, 0.0, 0.0).is_greedy());
        // A guided-decoding constraint masks the logits, so the on-device argmax (which never sees the mask)
        // would pick a disallowed token: a constrained row MUST force the host readback path.
        assert!(
            !Sampler::greedy()
                .with_constraint(Constraint::choices(vec![vec![1u32]], 2))
                .is_greedy()
        );
    }

    #[test]
    fn adjust_applies_bias_and_penalties_with_hf_openai_conventions() {
        // `Sampler::adjust` shapes real serving output; this pins each convention so a refactor cannot
        // silently break one.

        // logit_bias is additive; an out-of-range token id is ignored (not a panic).
        let s = Sampler::greedy().with_logit_bias(std::collections::HashMap::from([
            (1u32, 2.0),
            (999u32, 5.0),
        ]));
        let mut l = vec![0.0f32, 1.0, -1.0];
        s.adjust(&mut l);
        assert_eq!(
            l,
            vec![0.0, 3.0, -1.0],
            "bias adds to id 1; id 999 out of range is skipped"
        );

        // repetition_penalty: HF/CTRL convention - a seen token's POSITIVE logit is divided, a NEGATIVE one
        // multiplied (pushes both toward 0/negative). Applies to every seen token; unseen tokens untouched.
        let mut s = Sampler::greedy().with_penalties(2.0, 0.0, 0.0);
        s.observe(0); // a token with a positive logit
        s.observe(2); // a token with a negative logit
        let mut l = vec![4.0f32, 1.0, -4.0];
        s.adjust(&mut l);
        assert_eq!(l[0], 2.0, "positive logit divided by penalty");
        assert_eq!(l[1], 1.0, "unseen token unchanged");
        assert_eq!(l[2], -8.0, "negative logit multiplied by penalty");

        // presence_penalty subtracts ONCE per distinct seen token; frequency_penalty subtracts `count` times.
        let mut s = Sampler::greedy().with_penalties(1.0, 0.5, 0.1);
        s.observe(0);
        s.observe(0);
        s.observe(0); // token 0 seen 3x
        let mut l = vec![10.0f32, 10.0];
        s.adjust(&mut l);
        assert!(
            (l[0] - 9.2).abs() < 1e-6,
            "-presence(0.5) - frequency(0.1)*3 = -0.8"
        );
        assert_eq!(l[1], 10.0, "unseen token unchanged");

        // Prompt tokens seeded via `seed_context` are penalized by presence/frequency (llama.cpp-style
        // penalty over prompt+generated). This diverges from vLLM/OpenAI (output-only) intentionally; pinned
        // so it cannot change silently.
        let mut s = Sampler::greedy().with_penalties(1.0, 1.0, 0.0);
        s.seed_context(&[0]); // token 0 appears ONLY in the prompt
        let mut l = vec![5.0f32, 5.0];
        s.adjust(&mut l);
        assert!(
            (l[0] - 4.0).abs() < 1e-6,
            "prompt-only token is presence-penalized (llama.cpp semantics)"
        );
        assert_eq!(l[1], 5.0);
    }

    #[test]
    fn logprobs_record_chosen_and_top_alternatives() {
        // greedy with logprobs(2): the chosen token is the argmax and its logprob equals the top entry's;
        // exactly 2 alternatives are recorded, sorted most-probable first.
        let logits = [0.1, 3.0, 0.5, 2.9];
        let mut s = Sampler::greedy().with_logprobs(2);
        assert!(
            !s.is_greedy(),
            "logprobs must force the readback path (FR-006)"
        );
        let chosen = s.pick(&logits).unwrap();
        assert_eq!(chosen, 1); // argmax
        let rec = s.take_logprobs();
        assert_eq!(rec.len(), 1);
        assert_eq!(rec[0].token, 1);
        assert_eq!(rec[0].top.len(), 2);
        // most-probable first; the top entry is the chosen token with the same logprob.
        assert_eq!(rec[0].top[0].0, 1);
        assert!((rec[0].top[0].1 - rec[0].logprob).abs() < 1e-6);
        assert!(rec[0].top[1].1 <= rec[0].top[0].1);
        assert!(rec[0].logprob <= 0.0);
        // take_logprobs cleared the buffer.
        assert!(s.take_logprobs().is_empty());
    }

    #[test]
    fn logprobs_are_normalized() {
        // four equal logits -> a uniform distribution -> each logprob is ln(1/4).
        let logits = [1.0, 1.0, 1.0, 1.0];
        let mut s = Sampler::greedy().with_logprobs(4);
        s.pick(&logits).unwrap();
        let rec = s.take_logprobs();
        let expected = (0.25f32).ln();
        assert!((rec[0].logprob - expected).abs() < 1e-5);
        let sum: f32 = rec[0].top.iter().map(|&(_, lp)| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "top-4 probabilities sum to 1");
    }

    #[test]
    fn constraint_allowed_next_prefix_and_completion() {
        use super::Constraint;
        // two choices sharing a first token: [1,2,3] and [1,4]. The constraint is stateful (advance()).
        let mut c = Constraint::choices(vec![vec![1, 2, 3], vec![1, 4]], 0);
        let (a, eos) = c.allowed_next();
        assert_eq!(a, std::collections::HashSet::from([1]));
        assert!(!eos);
        c.advance(1);
        let (a, _) = c.allowed_next();
        assert_eq!(a, std::collections::HashSet::from([2, 4])); // branch
        c.advance(2);
        c.advance(3);
        let (a, eos) = c.allowed_next();
        assert!(a.is_empty() && eos, "complete choice -> EOS allowed");
        // a prefix that is also a full choice: [1] complete AND [1,2] continues.
        let mut c2 = Constraint::choices(vec![vec![1], vec![1, 2]], 0);
        c2.advance(1);
        let (a, eos) = c2.allowed_next();
        assert_eq!(a, std::collections::HashSet::from([2]));
        assert!(eos, "[1] is a complete choice so EOS is also allowed");
    }

    #[test]
    fn constraint_regex_allowed_next_walks_dfa_and_matches_at_eoi() {
        use super::Constraint;
        use regex_automata::dfa::{Automaton, StartKind, dense};
        use regex_automata::{Anchored, Input};
        use std::collections::HashSet;
        use std::sync::Arc;

        // Exercises the regex branch of `Constraint::allowed_next`/`advance` directly: poot's own byte-by-byte
        // DFA walk that masks tokens during guided regex/JSON decoding.
        // `poot-text`'s `json_schema_to_regex_accepts_and_rejects` only validates the generated regex string via a different
        // engine (meta::Regex). Builds the anchored whole-match DFA as `TextCodec::build_regex_constraint` /
        // `guided_dfa` do: `(?:pattern)$`, `StartKind::Anchored`.
        let dfa = Arc::new(
            dense::Builder::new()
                .configure(dense::Config::new().start_kind(StartKind::Anchored))
                .build("(?:ab)$")
                .unwrap(),
        );
        let start = dfa
            .start_state_forward(&Input::new("").anchored(Anchored::Yes))
            .unwrap();
        // token id -> decoded bytes. 0 = eos (empty). 1="a", 2="b", 3="c", 4="ab" (one multi-byte token).
        let table = Arc::new(vec![
            b"".to_vec(),
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"ab".to_vec(),
        ]);
        let mut c = Constraint::regex(dfa, start, table, 0);

        // At the start, only tokens whose bytes keep the whole-match alive are allowed: "a" (a live prefix)
        // and the multi-byte "ab" (the whole match); "b"/"c" dead-end on the first byte. EOS not yet allowed
        // (the empty string is not a full match of `ab$`).
        let (a, eos) = c.allowed_next();
        assert_eq!(a, HashSet::from([1, 4]));
        assert!(!eos, "empty output is not a full match of ab$");

        // After "a": only "b" continues; "a"/"c" dead-end, and "ab" dead-ends on its first byte from here.
        c.advance(1);
        let (a, eos) = c.allowed_next();
        assert_eq!(a, HashSet::from([2]));
        assert!(!eos, "\"a\" alone is not a full match");

        // After "ab": nothing can extend a complete match, but the pattern now matches at end-of-input, so
        // EOS is allowed and the allowed set is empty (the `is_match_state(next_eoi_state)` path).
        c.advance(2);
        let (a, eos) = c.allowed_next();
        assert!(a.is_empty(), "no token extends a complete ab$ match");
        assert!(eos, "\"ab\" is a full match -> EOS allowed");
    }

    #[test]
    fn constraint_masks_sampling_to_choice_tokens() {
        use super::Constraint;
        // choice = tokens [1, 3], eos id 0. Logits favor token 2 (argmax), but the constraint forces 1 then 3.
        let c = Constraint::choices(vec![vec![1, 3]], 0);
        let mut s = Sampler::greedy().with_constraint(c);
        assert!(
            !s.is_greedy(),
            "a constraint forces the readback path (FR-004)"
        );
        let logits = [5.0, 0.0, 9.0, 1.0]; // token 2 is the unconstrained argmax
        let t0 = s.pick(&logits).unwrap();
        assert_eq!(t0, 1, "masked to the only allowed first token");
        s.observe(t0 as u32);
        let t1 = s.pick(&logits).unwrap();
        assert_eq!(t1, 3, "masked to the only allowed continuation");
        s.observe(t1 as u32);
        // the choice is complete -> EOS (0) is forced even though token 2 has the highest raw logit.
        assert_eq!(s.pick(&logits).unwrap(), 0);
    }

    #[test]
    fn constraint_regex_masks_to_pattern() {
        use super::Constraint;
        use regex_automata::dfa::{Automaton, StartKind, dense};
        use regex_automata::{Anchored, Input};
        use std::sync::Arc;
        // anchored byte-DFA for [0-9]+; a tiny byte-table: id0="1", id1="a", id2="23", id3=eos(empty bytes).
        let dfa = dense::Builder::new()
            .configure(dense::Config::new().start_kind(StartKind::Anchored))
            .build("[0-9]+$")
            .unwrap();
        let start = dfa
            .start_state_forward(&Input::new("").anchored(Anchored::Yes))
            .unwrap();
        let bytes = Arc::new(vec![b"1".to_vec(), b"a".to_vec(), b"23".to_vec(), vec![]]);
        let mut c = Constraint::regex(Arc::new(dfa), start, bytes, 9);

        let (a, eos_ok) = c.allowed_next();
        assert!(
            a.contains(&0) && a.contains(&2) && !a.contains(&1),
            "digits allowed, letter not"
        );
        assert!(!eos_ok, "empty output does not match [0-9]+");
        c.advance(0); // emit "1"
        let (a, eos_ok) = c.allowed_next();
        assert!(eos_ok, "'1' matches [0-9]+ so EOS is allowed");
        assert!(
            a.contains(&0) && a.contains(&2) && !a.contains(&1),
            "more digits allowed, letter still not"
        );
    }

    #[test]
    fn neutral_params_match_baseline() {
        // SC-004: with all new params neutral the draw sequence is identical to the plain temperature path.
        let logits = [1.0, 2.0, 0.5, 1.5, 0.2];
        let mut base = Sampler::new(0.8, 3, 0.9, 1234);
        let mut rich = Sampler::new(0.8, 3, 0.9, 1234)
            .with_penalties(1.0, 0.0, 0.0)
            .with_min_p(0.0);
        let a: Vec<usize> = (0..32).map(|_| base.pick(&logits).unwrap()).collect();
        let b: Vec<usize> = (0..32).map(|_| rich.pick(&logits).unwrap()).collect();
        assert_eq!(a, b);
    }

    /// Card 148 sampling fuzz (host reference side). The on-device Gumbel-max kernels are validated
    /// GPU-vs-CPU-oracle elsewhere; the host `Sampler::pick` uses a structurally different inverse-CDF draw
    /// that only coincides with the kernels' Gumbel-max when the candidate set collapses to one token. So
    /// the CPU-only property here is that the host reference honors its own truncation across random logits
    /// and params: whatever `pick` returns lies inside the min-p/top-k/top-p nucleus. Bounds are
    /// boundary-tolerant (strict rank, log-space floor with tolerance), so a failure is a real truncation
    /// bug, not a tie.
    ///
    /// Random space (fixed seed, 300 cases): vocab in 1..300 (incl. vocab=1 and non-multiple-of-64),
    /// unique logits, temperature in {0 (greedy)} u (0.1, 2.0], top_k in {0=off, 1, .., >=vocab}, top_p in
    /// (0, 1], min_p in [0, 0.99), random seed. Checks: greedy/top_k=1/tiny-top_p collapse to argmax; picked
    /// rank < top_k; picked logit >= max + T*ln(min_p); determinism for a seed.
    #[test]
    fn sampler_pick_respects_truncation_bounds_fuzz() {
        fn xs(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        let mut seed = 0x0014_85a3_c0ff_ee11_u64;
        let mut cases = 0usize;
        let (mut greedy_c, mut topk1_c, mut minp_c, mut topp_collapse_c) = (0, 0, 0, 0);

        for _ in 0..300u64 {
            let vocab = 1 + (xs(&mut seed) % 299) as usize;
            // UNIQUE logits in ~[-5,5): random base + a strictly-increasing tiebreak so argmax and the
            // sort order are unambiguous (no float-boundary flakiness in the bound checks below).
            let logits: Vec<f32> = (0..vocab)
                .map(|i| {
                    let u = (xs(&mut seed) >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
                    (u * 10.0 - 5.0) + i as f32 * 1e-3
                })
                .collect();
            let greedy = xs(&mut seed).is_multiple_of(5);
            let temp = if greedy {
                0.0
            } else {
                0.1 + (xs(&mut seed) % 1000) as f32 / 1000.0 * 1.9
            };
            // top_k: 0 (off), 1, or a random cap that may exceed vocab.
            let top_k = match xs(&mut seed) % 4 {
                0 => 0usize,
                1 => 1usize,
                2 => 1 + (xs(&mut seed) % vocab as u64) as usize,
                _ => vocab + (xs(&mut seed) % 8) as usize, // >= vocab (effectively off)
            };
            let top_p = ((xs(&mut seed) % 1000) as f32 / 1000.0).max(1e-3); // (0, ~1]
            let min_p = if xs(&mut seed).is_multiple_of(3) {
                (xs(&mut seed) % 99) as f32 / 100.0 // [0, 0.98]
            } else {
                0.0
            };
            let rng_seed = xs(&mut seed);

            let build = || Sampler::new(temp, top_k, top_p, rng_seed).with_min_p(min_p);
            let mut s = build();
            let pick = s.pick(&logits).unwrap();
            assert!(pick < vocab, "pick {pick} out of range {vocab}");

            let (argmax, &max) = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            if greedy {
                assert_eq!(pick, argmax, "greedy must return argmax");
                greedy_c += 1;
            } else {
                // rank = number of tokens with a STRICTLY greater logit; top-k keeps only the top_k.
                let rank = logits.iter().filter(|&&x| x > logits[pick]).count();
                if top_k > 0 {
                    assert!(
                        rank < top_k,
                        "top_k={top_k} but pick rank {rank} (vocab={vocab})"
                    );
                    if top_k == 1 {
                        assert_eq!(pick, argmax, "top_k=1 must return argmax");
                        topk1_c += 1;
                    }
                }
                // min-p: a surviving token has exp((l-max)/T) >= min_p, i.e. l >= max + T*ln(min_p).
                if min_p > 0.0 {
                    let floor = max + temp * min_p.ln();
                    assert!(
                        logits[pick] >= floor - 1e-2 * max.abs().max(1.0),
                        "min_p={min_p} floor={floor} but pick logit {} (vocab={vocab})",
                        logits[pick]
                    );
                    minp_c += 1;
                }
                // top-p collapse: a singleton exact nucleus must be the argmax pick.
                if build().truncated_candidates(&logits).len() == 1 {
                    assert_eq!(
                        pick, argmax,
                        "a singleton top-p/min-p/top-k nucleus must be the pick"
                    );
                    topp_collapse_c += 1;
                }
            }

            // determinism: a freshly built identical sampler reproduces the same draw sequence.
            let mut s1 = build();
            let mut s2 = build();
            let d1: Vec<usize> = (0..6).map(|_| s1.pick(&logits).unwrap()).collect();
            let d2: Vec<usize> = (0..6).map(|_| s2.pick(&logits).unwrap()).collect();
            assert_eq!(d1, d2, "same seed must be deterministic");

            cases += 1;
        }
        eprintln!(
            "sampler truncation-bounds fuzz: {cases} cases ({greedy_c} greedy, {topk1_c} top_k=1, \
             {minp_c} min_p, {topp_collapse_c} singleton nucleus); all picks inside the nucleus"
        );
        assert!(
            greedy_c > 0 && topk1_c > 0 && minp_c > 0,
            "fuzz missed collapse/min_p coverage (greedy={greedy_c}, topk1={topk1_c}, minp={minp_c})"
        );
    }

    #[test]
    fn probs_is_one_hot_argmax_when_greedy() {
        let logits = [0.1f32, 3.0, 0.5, 2.9];
        let s = Sampler::greedy();
        let p = s.probs(&logits).unwrap();
        assert_eq!(p, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn probs_matches_the_temperature_softmax_before_any_truncation() {
        // plain temperature (no min-p/top-k/top-p), so `probs` is exactly the softmax(logits/T).
        let logits = [1.0f32, 2.0, 0.0];
        let s = Sampler::new(0.5, 0, 1.0, 1);
        let p = s.probs(&logits).unwrap();
        let expected: Vec<f32> = {
            let scaled: Vec<f32> = logits.iter().map(|&l| l / 0.5).collect();
            let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scaled.iter().map(|&l| (l - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|&e| e / sum).collect()
        };
        for (a, b) in p.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-6, "{p:?} vs {expected:?}");
        }
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn probs_zeroes_out_truncated_candidates() {
        // top_k=1 keeps only the argmax; `probs` must be one-hot on it, same as `pick_from`'s forced draw
        // in the `top_k_one_is_argmax_regardless_of_temperature` test above.
        let logits = [0.1f32, 3.0, 0.5, 2.9];
        let s = Sampler::new(1.0, 1, 1.0, 42);
        assert_eq!(s.probs(&logits).unwrap(), vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn probs_matches_pick_from_empirically() {
        // `probs` must be the SAME distribution `pick_from` draws from, including a real top-p
        // nucleus with NO other truncation active (SC-002, card 677). Check by drawing many samples
        // with `pick` and comparing the empirical histogram to `probs`'s output via chi-squared
        // (deterministic seed, no flakiness).
        let logits = [2.0f32, 1.0, 0.5, 0.5, -1.0, 3.0, 0.2, 1.5];
        let s_probs = Sampler::new(0.8, 0, 0.95, 1);
        let expected = s_probs.probs(&logits).unwrap();
        assert!(
            expected.iter().filter(|&&p| p > 0.0).count() < logits.len(),
            "top_p=0.95 with no min-p/top-k must still truncate some candidates to zero mass: \
             {expected:?}"
        );
        let mut s = Sampler::new(0.8, 0, 0.95, 1);
        let n = 200_000u64;
        let mut counts = vec![0u64; logits.len()];
        for _ in 0..n {
            counts[s.pick(&logits).unwrap()] += 1;
        }
        let chi2: f64 = counts
            .iter()
            .zip(&expected)
            .map(|(&o, &p)| {
                let e = f64::from(p) * n as f64;
                if e <= 0.0 {
                    0.0
                } else {
                    (o as f64 - e).powi(2) / e
                }
            })
            .sum();
        assert!(
            chi2 < 60.0,
            "chi-squared {chi2:.2} too high: probs()={expected:?} counts={counts:?} - probs() must \
             describe the same distribution pick_from() draws from"
        );
    }

    #[test]
    fn sample_from_probs_draws_proportionally() {
        let probs = [0.1f32, 0.6, 0.3];
        let mut s = Sampler::greedy(); // temperature is irrelevant to sample_from_probs
        let n = 100_000u64;
        let mut counts = [0u64; 3];
        for _ in 0..n {
            counts[s.sample_from_probs(&probs).unwrap() as usize] += 1;
        }
        let chi2: f64 = counts
            .iter()
            .zip(&probs)
            .map(|(&o, &p)| {
                let e = f64::from(p) * n as f64;
                (o as f64 - e).powi(2) / e
            })
            .sum();
        assert!(
            chi2 < 30.0,
            "chi-squared {chi2:.2} too high: counts={counts:?}"
        );
    }

    #[test]
    fn sample_from_probs_falls_back_to_argmax_on_zero_mass() {
        let zeros = [0.0f32, 0.0, 0.0];
        let mut s = Sampler::greedy();
        // argmax of an all-zero (tied) vector is the FIRST index - a defined, deterministic fallback.
        assert_eq!(s.sample_from_probs(&zeros).unwrap(), 0);
    }

    #[test]
    fn fork_derives_an_independent_deterministic_stream() {
        let base = Sampler::new(0.8, 5, 0.9, 123).with_min_p(0.1);
        let mut a = base.fork(0xAAAA);
        let mut b = base.fork(0xAAAA);
        let mut c = base.fork(0xBBBB);
        let logits = [0.3f32, 1.7, -0.4, 2.1, 0.9];
        let draw =
            |s: &mut Sampler| -> Vec<usize> { (0..8).map(|_| s.pick(&logits).unwrap()).collect() };
        let da = draw(&mut a);
        let db = draw(&mut b);
        let dc = draw(&mut c);
        assert_eq!(
            da, db,
            "same salt from the same parent must be deterministic"
        );
        assert_ne!(da, dc, "a different salt must give an independent stream");
        // the fork must carry the parent's temperature/top_k/top_p/min_p through (not reset to greedy).
        assert_eq!(a.draw_rule(), SampleRule::GumbelTopKTopP);
        assert_eq!(a.top_k, 5);
        assert_eq!(a.top_p, 0.9);
        assert_eq!(a.min_p, 0.1);
    }

    /// The exact top-p reference (POOT-1195): the smallest prefix of the tokens sorted by probability, in
    /// descending order, whose f64 mass reaches `top_p` of the total. Test-local; no production code
    /// imports it. Returns the nucleus as token ids.
    fn exact_nucleus(logits: &[(u32, f32)], temperature: f32, top_p: f32) -> Vec<u32> {
        let mut sorted = logits.to_vec();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max = f64::from(sorted[0].1);
        let mass: Vec<f64> = sorted
            .iter()
            .map(|&(_, l)| ((f64::from(l) - max) / f64::from(temperature)).exp())
            .collect();
        let target = f64::from(top_p) * mass.iter().sum::<f64>();
        let mut cum = 0.0f64;
        let mut size = sorted.len();
        for (i, m) in mass.iter().enumerate() {
            cum += m;
            if cum >= target {
                size = i + 1;
                break;
            }
        }
        sorted[..size].iter().map(|&(id, _)| id).collect()
    }

    fn zipf_logits(vocab: usize, s: f64) -> Vec<f32> {
        (0..vocab)
            .map(|i| (-s * ((i + 1) as f64).ln()) as f32)
            .collect()
    }

    /// The top-p probe distributions of POOT-1195: the two Zipf rows and a head with a flat tail (strictly
    /// descending so the boundary is one token, not a tie group). Each is `(name, logits, top_p, pinned
    /// nucleus size)`.
    fn nucleus_rows() -> Vec<(&'static str, Vec<f32>, f32, Option<usize>)> {
        let mut head_tail = vec![0.5f64.ln() as f32];
        head_tail.extend((0..500).map(|j| (0.001f64 * (1.0 - 1.0e-4 * j as f64)).ln() as f32));
        vec![
            ("zipf s=1.1", zipf_logits(32_000, 1.1), 0.95, Some(12_413)),
            ("zipf s=0.8", zipf_logits(32_000, 0.8), 0.95, Some(25_496)),
            ("head + flat tail", head_tail, 0.9, None),
        ]
    }

    /// SC-001: the host sampler's nucleus (the support of `probs`) is the exact reference's. The old
    /// quantized mass kept 20 of the 12,413 tokens on the first row.
    #[test]
    fn host_nucleus_equals_the_exact_reference() {
        for (name, logits, top_p, pinned) in nucleus_rows() {
            let tagged: Vec<(u32, f32)> = (0..).zip(logits.iter().copied()).collect();
            let reference = exact_nucleus(&tagged, 1.0, top_p);
            if let Some(pinned) = pinned {
                assert_eq!(
                    reference.len(),
                    pinned,
                    "{name}: the card pins the reference"
                );
            }
            let probs = Sampler::new(1.0, 0, top_p, 1).probs(&logits).unwrap();
            let kept: Vec<u32> = (0..)
                .zip(&probs)
                .filter(|(_, p)| **p > 0.0)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                kept.len(),
                reference.len(),
                "{name}: host nucleus size vs exact f64"
            );
            assert_eq!(kept, reference, "{name}: host nucleus members");
        }
    }

    /// SC-002 (draws): with penalties and a logit bias active (the rows the device cannot sample), the
    /// host draw follows the exact distribution. The expected distribution is computed here, apart from
    /// the sampler: the adjustments by hand, then the exact f64 softmax and nucleus. 100k draws, chi-square
    /// against the `p = 0.001` critical value of the nucleus's degrees of freedom.
    ///
    /// Mutation: round the candidate weights to `2^-4` in `truncated_candidates`; the statistic exceeds
    /// the critical value.
    #[test]
    fn host_draws_with_penalties_and_bias_follow_the_exact_distribution() {
        // chi-square critical values at p = 0.001, indexed by degrees of freedom.
        const CRITICAL: [f64; 9] = [0.0, 10.83, 13.82, 16.27, 18.47, 20.52, 22.46, 24.32, 26.12];
        let logits = [2.0f32, 1.6, 1.5, 1.2, 0.9, 0.4, 0.0, -0.5, -1.0];
        let (temperature, top_p) = (0.9f32, 0.92f32);
        let mut s = Sampler::new(temperature, 0, top_p, 20_261_018)
            .with_penalties(1.3, 0.2, 0.1)
            .with_logit_bias(HashMap::from([(4u32, 0.8f32), (8, -3.0)]));
        for (token, times) in [(0u32, 3), (2, 1), (5, 2)] {
            for _ in 0..times {
                s.observe(token);
            }
        }
        // The adjustments by hand: bias first, then each seen token's repetition / presence / frequency.
        let mut adjusted = logits;
        adjusted[4] += 0.8;
        adjusted[8] -= 3.0;
        for (token, count) in [(0usize, 3.0f32), (2, 1.0), (5, 2.0)] {
            let l = adjusted[token];
            adjusted[token] = (if l > 0.0 { l / 1.3 } else { l * 1.3 }) - 0.2 - 0.1 * count;
        }
        let tagged: Vec<(u32, f32)> = (0..).zip(adjusted.iter().copied()).collect();
        let nucleus = exact_nucleus(&tagged, temperature, top_p);
        let max = adjusted.iter().copied().fold(f32::MIN, f32::max);
        let weight = |i: u32| {
            ((f64::from(adjusted[i as usize]) - f64::from(max)) / f64::from(temperature)).exp()
        };
        let total: f64 = nucleus.iter().map(|&i| weight(i)).sum();
        assert!(
            (3..CRITICAL.len() + 1).contains(&nucleus.len()),
            "the fixture needs a multi-token nucleus, got {}",
            nucleus.len()
        );

        let n = 100_000u64;
        let mut counts = vec![0u64; logits.len()];
        for _ in 0..n {
            counts[s.pick(&logits).unwrap()] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            assert!(
                c == 0 || nucleus.contains(&(i as u32)),
                "token {i} drawn outside the exact nucleus {nucleus:?}"
            );
        }
        let chi2: f64 = nucleus
            .iter()
            .map(|&i| {
                let e = weight(i) / total * n as f64;
                (counts[i as usize] as f64 - e).powi(2) / e
            })
            .sum();
        let critical = CRITICAL[nucleus.len() - 1];
        assert!(
            chi2 < critical,
            "chi-squared {chi2:.2} over the {}-token nucleus exceeds the p=0.001 value {critical}: counts {counts:?}",
            nucleus.len()
        );
    }

    /// The support of the device schedule's draw: `SampleToken(GumbelTopKTopP)` evaluated by the oracle (the
    /// bit-exact reference of the device kernels, ADR-0114 tier 1) over `rows` identical copies of `logits`
    /// with distinct seeds, as a test-local graph. Returns the tokens drawn, ascending.
    fn device_support(logits: &[f32], temperature: f32, top_p: f32, rows: usize) -> Vec<usize> {
        use poot_eval::{EvalOptions, Value};
        use poot_graph_ir::ops::sampling::{SampleSpec, sample_head};
        use poot_graph_ir::{Builder, Slot, TensorType};
        use poot_tensor::{DType, HostTensor};

        let vocab = logits.len();
        let b = Builder::for_tests();
        let logits_in = b
            .slot(
                Slot::Activation,
                TensorType::new(vec![rows, vocab], DType::F32),
            )
            .unwrap();
        let head = sample_head(
            &b,
            logits_in,
            SampleSpec::draw(SampleRule::GumbelTopKTopP),
            None,
        )
        .unwrap();
        let g = b.finish(head.out).unwrap();
        let row_params = [1.0 / temperature, f32::NEG_INFINITY, 1.0, top_p];
        let mut inputs = HashMap::new();
        inputs.insert(
            logits_in.id(),
            Value::from(HostTensor::f32(
                vec![rows, vocab],
                logits.iter().copied().cycle().take(rows * vocab).collect(),
            )),
        );
        inputs.insert(
            head.seed.unwrap().id(),
            Value::from(HostTensor::i32(vec![rows], (0..rows as i32).collect())),
        );
        inputs.insert(
            head.params.unwrap().id(),
            Value::from(HostTensor::f32(
                vec![rows, 4],
                row_params.iter().copied().cycle().take(rows * 4).collect(),
            )),
        );
        inputs.insert(
            head.top_k.unwrap().id(),
            Value::from(HostTensor::i32(vec![rows], vec![0; rows])),
        );
        let out = poot_eval::eval(&g, &inputs, EvalOptions::default())
            .unwrap()
            .output
            .into_host()
            .unwrap();
        let pairs = out.as_i32().unwrap();
        let mut tokens: Vec<usize> = pairs.chunks(2).map(|p| p[0] as usize).collect();
        tokens.sort_unstable();
        tokens.dedup();
        tokens
    }

    /// SC-002 (mass): the nucleus the device schedule selects is held to the exact reference by the
    /// ADR-0114 tolerance `epsilon = vocab * 2^-25` of probability mass (every token's integer mass rounds
    /// by at most half a unit of `2^-24` of the top token). The device support is a prefix of the exact
    /// order; its mass reaches `top_p - epsilon`, and dropping its last token leaves less than `top_p +
    /// epsilon`. The host nucleus (the support of `probs`) meets the same bounds exactly.
    ///
    /// The device half evaluates the oracle (`poot_eval`, the bit-exact reference of the device kernels), so
    /// it goes red for a change to the oracle's top-p, never for one to the host sampler.
    ///
    /// Mutations: in `truncated_candidates` cut the nucleus one token early (`i` instead of `i + 1`); the
    /// host row's lower bound fails. In poot-eval's `top_p_threshold` halve the product (`top_p * total /
    /// 2.0`); the device row's lower bound fails.
    #[test]
    fn device_nucleus_mass_is_within_the_adr_0114_tolerance_of_the_exact_nucleus() {
        let weights = [0.3f64, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.03, 0.025, 0.015];
        let logits: Vec<f32> = weights.iter().map(|w| w.ln() as f32).collect();
        let total: f64 = weights.iter().sum();
        let epsilon = logits.len() as f64 * 2f64.powi(-25);
        for top_p in [0.3f32, 0.5, 0.8, 0.95] {
            let mass = |tokens: &[usize]| tokens.iter().map(|&i| weights[i] / total).sum::<f64>();
            let check = |who: &str, support: Vec<usize>| {
                let size = support.len();
                assert_eq!(
                    support,
                    (0..size).collect::<Vec<_>>(),
                    "{who} top_p={top_p}: the nucleus is a prefix of the descending order"
                );
                assert!(
                    mass(&support) >= f64::from(top_p) - epsilon,
                    "{who} top_p={top_p}: nucleus mass {} below {top_p} - {epsilon}",
                    mass(&support)
                );
                assert!(
                    mass(&support[..size - 1]) < f64::from(top_p) + epsilon,
                    "{who} top_p={top_p}: the nucleus without its last token already holds {}",
                    mass(&support[..size - 1])
                );
            };
            let probs = Sampler::new(1.0, 0, top_p, 1).probs(&logits).unwrap();
            check(
                "host",
                (0..)
                    .zip(&probs)
                    .filter(|(_, p)| **p > 0.0)
                    .map(|(i, _)| i)
                    .collect(),
            );
            check("device", device_support(&logits, 1.0, top_p, 4096));
        }
    }

    /// 50 allowed ids out of 200, with distinct logits, and the constraint that allows exactly them.
    fn guided_row() -> (Vec<f32>, Vec<u32>, Constraint) {
        let allowed: Vec<u32> = (0..50).map(|i| 3 + 4 * i).collect();
        // A distinct, scrambled logit per token so the allowed set has a unique argmax and a short nucleus.
        let logits: Vec<f32> = (0..200u32)
            .map(|i| ((i * 37) % 200) as f32 * 0.07)
            .collect();
        let constraint = Constraint::choices(allowed.iter().map(|&id| vec![id]).collect(), 199);
        (logits, allowed, constraint)
    }

    /// SC-004: under a constraint, `top_k = 1` returns the argmax of the allowed set for every seed. The
    /// `f32::MIN` mask sentinel used to be the bisection floor, which left top-k a no-op.
    #[test]
    fn guided_top_k_one_returns_the_allowed_argmax() {
        let (logits, allowed, _) = guided_row();
        let best = *allowed
            .iter()
            .max_by(|&&a, &&b| logits[a as usize].total_cmp(&logits[b as usize]))
            .unwrap();
        for seed in 1..=64u64 {
            let (_, _, constraint) = guided_row();
            let mut s = Sampler::new(0.8, 1, 1.0, seed).with_constraint(constraint);
            assert_eq!(s.pick(&logits).unwrap(), best as usize, "seed {seed}");
        }
    }

    /// SC-005: under the same constraint, `top_p = 0.5` draws only from the nucleus the exact reference
    /// computes over the allowed set, and still draws more than one distinct token.
    #[test]
    fn guided_top_p_draws_only_from_the_allowed_nucleus() {
        let (logits, allowed, _) = guided_row();
        let allowed_logits: Vec<(u32, f32)> = allowed
            .iter()
            .map(|&id| (id, logits[id as usize]))
            .collect();
        let nucleus = exact_nucleus(&allowed_logits, 0.8, 0.5);
        assert!(
            (2..allowed.len()).contains(&nucleus.len()),
            "the fixture needs a proper multi-token nucleus, got {}",
            nucleus.len()
        );
        let mut seen = std::collections::BTreeSet::new();
        for seed in 1..=300u64 {
            let (_, _, constraint) = guided_row();
            let mut s = Sampler::new(0.8, 0, 0.5, seed).with_constraint(constraint);
            let token = s.pick(&logits).unwrap() as u32;
            assert!(
                nucleus.contains(&token),
                "seed {seed}: {token} is outside the nucleus {nucleus:?}"
            );
            seen.insert(token);
        }
        assert!(
            seen.len() > 1,
            "a nucleus of {} tokens must be sampled, not collapsed: {seen:?}",
            nucleus.len()
        );
    }
}

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;

    impl Sampler {
        /// How many times `token` is in the penalty history (prompt context and observed tokens).
        pub(crate) fn seen(&self, token: u32) -> u32 {
            self.counts.get(&token).copied().unwrap_or(0)
        }
    }
}
