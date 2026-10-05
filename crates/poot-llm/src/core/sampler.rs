//! Sampler: temperature/top-k/top-p/min-p/penalties sampling and logprobs.

use std::collections::HashMap;

use crate::core::generate::argmax;
use crate::text::guided::Constraint;

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
    NonFiniteLogit { index: usize, value: Option<f32> },
    /// The row is empty or every logit is `-inf`, so no token can be chosen.
    #[error("no finite logit among {len} logits")]
    NoFiniteLogit { len: usize },
    /// The device-matching host draw's own graph evaluation failed (card 551b):
    /// `pick_from`'s `suffix::eval_sample_token` call is built and bound by `pick_from`
    /// itself, so a failure here (e.g. an allocation/budget limit) is not data-dependent the way
    /// the other two faults are, but it is still a real `poot_eval::eval` error the production host
    /// pick must surface through this `Result` rather than panic on.
    #[error("sampling evaluation failed: {message}")]
    Eval { message: String },
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
    pub token: u32,
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

impl Sampler {
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

    /// True when the sampler picks the argmax with no logit adjustment, so GPU decode can choose the token
    /// on-device and skip the logits readback. Any active penalty/bias reshapes the argmax, logprob
    /// recording needs the full logits, and a guided-decoding constraint masks them; each forces the
    /// host-side readback path (FR-006 / FR-004).
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
            && !self.adjusts_logits()
            && self.top_logprobs.is_none()
            && self.constraint.is_none()
    }

    /// Whether this row can sample on the GPU via an on-device top-p-Gumbel-max kernel instead of the host
    /// `pick()` path (card 148): min-p folds into a logit floor, top-k into a bisected count threshold, and
    /// top-p into a bisected integer-mass threshold (each token's mass quantized to fixed-point u32 and
    /// summed via `atomic_add`, so the sum is order-independent and GPU==CPU bit-exact). Same "no
    /// penalties/bias/logprobs/guided" bar as [`Self::is_greedy`].
    pub fn is_simple_temperature(&self) -> bool {
        self.temperature > 0.0
            && !self.adjusts_logits()
            && self.top_logprobs.is_none()
            && self.constraint.is_none()
    }

    /// Whether any penalty or bias would change the raw logits (and thus the argmax).
    fn adjusts_logits(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || !self.logit_bias.is_empty()
    }

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

    /// How many times `token` is in the penalty history (prompt context and observed tokens).
    #[cfg(test)]
    pub(crate) fn seen(&self, token: u32) -> u32 {
        self.counts.get(&token).copied().unwrap_or(0)
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
    /// draw (`speculative::spec_sample_round`).
    ///
    /// Its one caller is the held `spec_sample_round` (POOT-751), so it is dead with that root; the
    /// hold sits there.
    pub(crate) fn uniform(&mut self) -> f32 {
        let s = self.next_u64();
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// The next decode step's device-style `RandomUniform` seed (card 551b, R-551b-2): both the host
    /// pick ([`Self::pick_from`]'s Gumbel-max draw, via `suffix::eval_sample_token`) and a
    /// device suffix row ([`crate::driver::suffix::SuffixRows::push`]) consume one of these
    /// per decode step, so a seeded `Sampler` used on either path draws the identical per-step
    /// noise. A distinct draw from [`Self::uniform`]'s own stream position (callers never mix the
    /// two per step).
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

    /// Mask `logits` to the tokens the guided-decoding constraint permits at the current generated prefix:
    /// disallowed ids become -inf. EOS is allowed once a choice is complete; if nothing continues (the
    /// constraint is exhausted), EOS is forced so the masked logits are never all -inf (FR-003).
    fn apply_constraint_mask(&self, logits: &mut [f32]) {
        let c = self.constraint.as_ref().unwrap();
        let (allowed, eos_ok) = c.allowed_next();
        let eos_ok = eos_ok || allowed.is_empty();
        let eos = c.eos();
        for (i, l) in logits.iter_mut().enumerate() {
            let id = i as u32;
            let keep = allowed.contains(&id) || (eos_ok && id == eos);
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

    /// The min-p / top-k / top-p truncated, unnormalized candidate list (`(token id, exp((logit-max)/T))`)
    /// for a temperature > 0 draw. Shared by [`Self::pick_from`] (one inverse-CDF draw) and [`Self::probs`]
    /// (the full distribution for speculative sampling's accept/reject math), so the two cannot drift on
    /// what "the sampler's distribution" means. Greedy (temperature <= 0) is the caller's job (a one-hot
    /// distribution).
    fn truncated_candidates(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut cand: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, ((l - max) / self.temperature).exp()))
            .collect();
        // descending by probability for min-p / top-k / top-p.
        cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // min-p: drop the tail below min_p * (top prob). cand is unnormalized exp, sorted descending, so
        // the ratio to cand[0] is the probability ratio - no normalization needed.
        if let Some(&(_, top)) = cand.first().filter(|_| self.min_p > 0.0) {
            let floor = self.min_p * top;
            let cut = cand.iter().position(|c| c.1 < floor).unwrap_or(cand.len());
            cand.truncate(cut.max(1));
        }
        if self.top_k > 0 && self.top_k < cand.len() {
            cand.truncate(self.top_k);
        }
        if self.top_p <= 0.0 {
            cand.truncate(1);
        } else if self.top_p < 1.0 {
            // ADR-0114 (tier 2): mirrors `poot-eval`'s `sample_one_row` quantized top-p bisection
            // (integer mass in units of `round(exp((l-max)/T) * 16)`), not an exact real-valued
            // cumulative sum, so `probs()` (built from this list) describes the nucleus
            // `pick_from`'s device-equivalent `eval_sample_token` draws from (R-551b-2's "one
            // sampling semantics" extended to top-p, card 551b). `cand` is already the
            // min-p/top-k floor's surviving set (tier 1, unaffected), sorted descending by
            // `exp((l-max)/T)`, so a quantized prefix sum over it is exactly the oracle's
            // `mass_at(thresh)` restricted to this set, for the same reason the oracle's own
            // bisection only ever searches thresholds within it (card 677: that bisection now
            // starts its search at a finite floor - `min_logit` at worst - whether or not min-p or
            // top-k ran, so this cut is unconditional).
            let z_trunc: f32 = cand.iter().map(|c| (c.1 * 16.0).round()).sum();
            let tp_thresh = (self.top_p * z_trunc).round();
            let mut cum = 0.0f32;
            let mut cut = cand.len();
            for (i, c) in cand.iter().enumerate() {
                cum += (c.1 * 16.0).round();
                if cum >= tp_thresh {
                    cut = i + 1;
                    break;
                }
            }
            // The oracle cuts by VALUE (`v >= t_p`), so two candidates at the identical logit (and
            // therefore the identical unnormalized `exp`) are always kept or dropped together; a
            // sorted-list rank cut can split such a tie if it lands between them (`sort_unstable_by`
            // makes no promise about their relative order). Extend the cut past any trailing ties
            // with the just-included candidate so this list never disagrees with the oracle.
            while cut < cand.len() && cand[cut].1 == cand[cut - 1].1 {
                cut += 1;
            }
            cand.truncate(cut.max(1));
        }
        cand
    }

    /// The temperature + truncation draw over (already-adjusted) logits. R-551b-2: this is no
    /// longer its own inverse-CDF draw over the xorshift stream - it evaluates the identical
    /// `SampleToken`/`RandomUniform` semantics a device suffix row would (`suffix::
    /// eval_sample_token`, through `poot-eval`'s own walk), so the host pick and a device suffix
    /// agree bit-for-bit given the same seed and the same (already-adjusted) logits.
    fn pick_from(&mut self, logits: &[f32]) -> Result<usize, SamplerFault> {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        check_pickable(logits)?;
        // `OpKind::SampleToken`'s non-finite column flags ANY non-finite value, including a masked
        // `-inf` (card 551a SC-004 pins this for the device suffix, whose production callers never
        // see a masked row - `rule_of` routes those to the host). The host path's contract is
        // different and pre-existing (`check_pickable` already passed, so `-inf` here is a
        // legitimately masked token, never a fault): substitute it with a large-but-finite sentinel
        // so it reads as a real, vanishingly-low-probability candidate instead of tripping the
        // oracle's non-finite flag. `check_pickable` guarantees at least one strictly finite logit,
        // so this sentinel is never the row's max.
        let patched: Option<Vec<f32>> = logits.contains(&f32::NEG_INFINITY).then(|| {
            logits
                .iter()
                .map(|&v| if v == f32::NEG_INFINITY { f32::MIN } else { v })
                .collect()
        });
        let logits = patched.as_deref().unwrap_or(logits);
        let rule = crate::driver::suffix::temperature_rule(self.top_k, self.top_p);
        let seed = self.next_device_seed();
        let (token, non_finite) = crate::driver::suffix::eval_sample_token(
            rule,
            logits,
            seed,
            self.temperature,
            self.min_p,
            self.top_k,
            self.top_p,
        )
        .map_err(|e| SamplerFault::Eval {
            message: e.to_string(),
        })?;
        debug_assert!(
            non_finite < 0,
            "check_pickable already rejected a non-finite logit in this row, and -inf was patched out above"
        );
        Ok(token as usize)
    }

    /// The full-vocab probability distribution [`Self::pick_from`] draws from, given already-adjusted logits
    /// (bias/penalties are the caller's job, as in [`Self::pick`]). Building block for speculative sampling
    /// (card 094), which needs each drafted token's probability under this sampler. Temperature <= 0 is a
    /// one-hot distribution on the argmax. Entries outside the min-p/top-k/top-p candidate set are exactly
    /// 0; kept entries are normalized to sum to 1. Pure (`&self`, no rng draw); the same truncation as
    /// [`Self::pick_from`], materialized as a vector. A [`SamplerFault`] on the logits [`Self::pick`] rejects.
    ///
    /// No caller outside this crate's own tests (`core::speculative`'s sampled speculative decoding,
    /// card 094): held for POOT-751 (formerly 586a) (sampled two-model speculation over host p/q, the draft rng stream).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "held for POOT-751 (formerly 586a)")
    )]
    pub(crate) fn probs(&self, logits: &[f32]) -> Result<Vec<f32>, SamplerFault> {
        let mut v = vec![0.0f32; logits.len()];
        if self.temperature <= 0.0 {
            v[argmax(logits)?] = 1.0;
            return Ok(v);
        }
        check_pickable(logits)?;
        let cand = self.truncated_candidates(logits);
        let total: f32 = cand.iter().map(|c| c.1).sum();
        for &(i, p) in &cand {
            v[i] = p / total;
        }
        Ok(v)
    }

    /// Draws a token id from an arbitrary full-vocab probability vector (normalized to sum to 1, e.g. from
    /// [`Self::probs`], or the residual `max(0, p-q)` speculative sampling resamples from after a rejected
    /// draft): the inverse-CDF draw `pick_from` does internally. Falls back to `probs`'s argmax if the mass
    /// is numerically <= 0 (a float-rounding safety net). A [`SamplerFault`] when `probs` holds a NaN or
    /// infinity.
    pub(crate) fn sample_from_probs(&mut self, probs: &[f32]) -> Result<u32, SamplerFault> {
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
    ///
    /// No caller outside this crate's own tests (`core::speculative`'s sampled speculative decoding,
    /// card 094): held for POOT-751 (formerly 586a) (sampled two-model speculation over host p/q, the draft rng stream).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "held for POOT-751 (formerly 586a)")
    )]
    pub(crate) fn fork(&self, salt: u64) -> Sampler {
        Sampler::new(self.temperature, self.top_k, self.top_p, self.rng ^ salt)
            .with_min_p(self.min_p)
    }
}

#[cfg(test)]
mod sampler_tests {
    use super::{Constraint, Sampler, SamplerFault};
    use crate::core::generate::argmax;
    use crate::text::guided::{int_bound_regex, json_schema_to_regex};

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

    /// R486-006 (RU-5): the top-p cut is inclusive (`cum >= top_p`). The candidate weights are exactly
    /// `1.0, 0.75, 0.25` (the max logit is `exp(0) == 1`, the others are `exp(ln(0.75))`/`exp(ln(0.25))`,
    /// which round-trip through f32), so the running (quantized, ADR-0114) mass after the top
    /// candidate is exactly half of the total. `top_p = 0.5` must therefore keep the top candidate
    /// and stop; the `>=` -> `>` mutation takes one more and the literal candidate set changes.
    #[test]
    fn top_p_boundary_is_inclusive() {
        let logits = [0.0f32, 0.75f32.ln(), 0.25f32.ln()];
        let s = Sampler::new(1.0, 0, 0.5, 11);
        assert_eq!(
            s.truncated_candidates(&logits),
            vec![(0, 1.0)],
            "mass exactly equal to top_p ends the nucleus (cum >= top_p)"
        );
    }

    /// R486-006 (RU-6): the min-p cut drops only weights strictly below `min_p * top`; a token equal to
    /// the floor survives. With `min_p = 0.75` and unnormalized top `1.0`, the floor is exactly `0.75`
    /// and the token at `exp(ln(0.75))` sits on it. The `<` -> `<=` mutation drops it and the literal
    /// candidate set changes.
    #[test]
    fn min_p_boundary_keeps_tokens_equal_to_the_floor() {
        let logits = [0.0f32, 0.75f32.ln(), 0.25f32.ln()];
        let s = Sampler::new(1.0, 0, 1.0, 11).with_min_p(0.75);
        assert_eq!(
            s.truncated_candidates(&logits),
            vec![(0, 1.0), (1, 0.75)],
            "a token equal to the min-p floor survives (weight < floor drops)"
        );
    }

    #[test]
    fn is_simple_temperature_gates_like_the_serving_kernel() {
        // Gate for the on-device top-p-Gumbel-max kernel (card 148). A plain temperature draw (no
        // penalties/bias/logprobs/guided) qualifies, as do min-p, top-k, and top-p (each folds into the
        // kernel's logit floor / integer-mass nucleus threshold). Greedy (temperature<=0) and every host-only
        // knob must not.
        assert!(Sampler::new(0.8, 0, 1.0, 1).is_simple_temperature());
        assert!(
            Sampler::new(0.8, 0, 1.0, 1)
                .with_min_p(0.5)
                .is_simple_temperature()
        );
        assert!(
            Sampler::new(0.8, 5, 1.0, 1).is_simple_temperature(),
            "top_k alone (card 148 phase 2a) is now on-device, not a host fallback"
        );
        assert!(
            Sampler::new(0.8, 0, 0.9, 1).is_simple_temperature(),
            "top_p alone (card 148 phase 2b) is now on-device, not a host fallback"
        );
        assert!(
            Sampler::new(0.8, 5, 0.9, 1).is_simple_temperature(),
            "top_k + top_p combined (card 148 phase 2b) is now on-device"
        );
        assert!(!Sampler::greedy().is_simple_temperature());
        assert!(!Sampler::new(0.0, 0, 1.0, 1).is_simple_temperature());
        assert!(
            !Sampler::new(0.8, 0, 1.0, 1)
                .with_penalties(1.2, 0.0, 0.0)
                .is_simple_temperature()
        );
        assert!(
            !Sampler::new(0.8, 0, 1.0, 1)
                .with_logprobs(0)
                .is_simple_temperature()
        );
        // A logit_bias reshapes the argmax and the gumbel kernel never sees it, so a biased row must take the
        // host path (the same divergence class as the fixed top_p=0 GPU-fastpath bug). Covered for is_greedy by
        // `bias_makes_sampler_non_greedy`; pinned here too.
        assert!(
            !Sampler::new(0.8, 0, 1.0, 1)
                .with_logit_bias(std::collections::HashMap::from([(0u32, -1.0)]))
                .is_simple_temperature()
        );
        // A guided-decoding constraint masks the logits every step and the gumbel kernel cannot enforce it, so
        // a constrained row must take the host path.
        assert!(
            !Sampler::new(0.8, 0, 1.0, 1)
                .with_constraint(Constraint::choices(vec![vec![1u32]], 2))
                .is_simple_temperature()
        );
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
        // `json_schema_to_regex_accepts_and_rejects` only validates the generated regex string via a different
        // engine (meta::Regex). Builds the anchored whole-match DFA as `Runner::build_regex_constraint` /
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
    fn json_schema_to_regex_accepts_and_rejects() {
        // full-match helper via regex-automata's meta engine (\A..\z = whole string).
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "active": {"type": "boolean"},
                "kind": {"enum": ["a", "b"]}
            }
        });
        let rx = json_schema_to_regex(&schema).unwrap();
        // keys are emitted in sorted order (serde_json sorts object keys): active, age, kind, name, tags.
        // conforming (compact and with whitespace) -> match.
        assert!(full_match(
            &rx,
            r#"{"active":true,"age":3,"kind":"a","name":"bob","tags":["x","y"]}"#
        ));
        assert!(full_match(
            &rx,
            "{ \"active\": false, \"age\": 3, \"kind\": \"b\", \"name\": \"bob\", \"tags\": [] }"
        ));
        // non-conforming -> no match.
        assert!(
            !full_match(&rx, r#"{"active":true,"age":3}"#),
            "missing properties"
        );
        assert!(
            !full_match(
                &rx,
                r#"{"active":true,"age":3,"kind":"a","name":3,"tags":[]}"#
            ),
            "name must be a string"
        );
        assert!(
            !full_match(
                &rx,
                r#"{"active":true,"age":3,"kind":"c","name":"b","tags":[]}"#
            ),
            "kind must be one of the enum"
        );
        // an unsupported construct errors rather than mis-constraining.
        assert!(json_schema_to_regex(&serde_json::json!({"type": "null"})).is_err());
    }

    #[test]
    fn json_schema_to_regex_optional_properties_unions_const_and_ref() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }

        // Optional properties: `required` selects which keys must appear (the rest are any in-order subset).
        // Exercises object_regex's alternation-by-first-present-property and comma placement, which the
        // all-required test never hits and which emit malformed JSON (stray/double commas, a skippable
        // required field) if wrong. Props sorted: a, b, c; only b is required.
        let rx = json_schema_to_regex(&serde_json::json!({
            "type":"object",
            "properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}},
            "required":["b"]
        }))
        .unwrap();
        for s in [
            r#"{"b":1}"#,
            r#"{"b":1,"c":2}"#,
            r#"{"a":0,"b":1}"#,
            r#"{"a":0,"b":1,"c":2}"#,
        ] {
            assert!(full_match(&rx, s), "required-subset should accept {s}");
        }
        for s in [
            r#"{}"#,            // b (required) missing
            r#"{"a":0}"#,       // b missing
            r#"{"a":0,"c":2}"#, // b missing
            r#"{"c":2,"b":1}"#, // wrong (non-sorted) order
            r#"{"b":1,}"#,      // trailing comma
        ] {
            assert!(!full_match(&rx, s), "required-subset should reject {s}");
        }

        // ALL-OPTIONAL object (`required: []`) accepts the empty object and any in-order subset, and never a
        // leading/trailing comma.
        let rx = json_schema_to_regex(&serde_json::json!({
            "type":"object",
            "properties":{"a":{"type":"integer"},"b":{"type":"integer"}},
            "required":[]
        }))
        .unwrap();
        for s in [r#"{}"#, r#"{"a":1}"#, r#"{"b":2}"#, r#"{"a":1,"b":2}"#] {
            assert!(full_match(&rx, s), "all-optional should accept {s}");
        }
        for s in [r#"{"b":2,"a":1}"#, r#"{"a":1,}"#, r#"{,"a":1}"#] {
            assert!(!full_match(&rx, s), "all-optional should reject {s}");
        }

        // oneOf/anyOf: a union of subschemas -> an alternation (oneOf's exactly-one relaxed to at-least-one).
        let rx = json_schema_to_regex(&serde_json::json!({
            "oneOf":[{"type":"integer"},{"type":"boolean"}]
        }))
        .unwrap();
        assert!(full_match(&rx, "42") && full_match(&rx, "true"));
        assert!(!full_match(&rx, "\"x\"") && !full_match(&rx, "1.5"));

        // const: a single fixed scalar value (its escaped canonical JSON serialization).
        let rx = json_schema_to_regex(&serde_json::json!({"const":"hello"})).unwrap();
        assert!(full_match(&rx, "\"hello\"") && !full_match(&rx, "\"world\""));
        let rx = json_schema_to_regex(&serde_json::json!({"const":42})).unwrap();
        assert!(full_match(&rx, "42") && !full_match(&rx, "43"));

        // $ref: a local JSON-Pointer ref is resolved + inlined (here into a bounded integer, so the bound is
        // enforced through the ref). External and recursive refs error rather than loop/mis-constrain.
        let rx = json_schema_to_regex(&serde_json::json!({
            "$defs":{"Id":{"type":"integer","minimum":1}},
            "type":"object",
            "properties":{"id":{"$ref":"#/$defs/Id"}},
            "required":["id"]
        }))
        .unwrap();
        assert!(
            full_match(&rx, r#"{"id":5}"#),
            "ref resolves to the bounded integer"
        );
        assert!(
            !full_match(&rx, r#"{"id":0}"#),
            "ref's minimum:1 is enforced"
        );
        assert!(
            json_schema_to_regex(&serde_json::json!({"$ref":"http://x/y"})).is_err(),
            "external $ref errors"
        );
        assert!(
            json_schema_to_regex(&serde_json::json!({
                "$defs":{"Node":{"type":"object","properties":{"next":{"$ref":"#/$defs/Node"}},"required":["next"]}},
                "$ref":"#/$defs/Node"
            }))
            .is_err(),
            "recursive $ref errors (not expressible as a finite regex)"
        );
    }

    #[test]
    fn json_schema_string_format_constrains_the_content() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // (format, a conforming value, a non-conforming value) - each tested as the bare quoted string.
        let cases = [
            ("uuid", "550e8400-e29b-41d4-a716-446655440000", "not-a-uuid"),
            ("date", "2026-06-24", "2026-13-40"),
            ("date-time", "2026-06-24T14:30:00Z", "2026-06-24 14:30"),
            ("time", "14:30:00", "25:61:00"),
            ("email", "bob@example.com", "bob@@no"),
            ("ipv4", "192.168.0.1", "999.1.1.1"),
        ];
        for (fmt, good, bad) in cases {
            let rx =
                json_schema_to_regex(&serde_json::json!({"type":"string","format":fmt})).unwrap();
            assert!(
                full_match(&rx, &format!("\"{good}\"")),
                "{fmt}: should accept {good}"
            );
            assert!(
                !full_match(&rx, &format!("\"{bad}\"")),
                "{fmt}: should reject {bad}"
            );
        }
        // pattern governs over format when both are present.
        let rx = json_schema_to_regex(
            &serde_json::json!({"type":"string","format":"uuid","pattern":"^x+$"}),
        )
        .unwrap();
        assert!(full_match(&rx, "\"xxx\""), "pattern governs over format");
        // an UNKNOWN format is ignored (advisory) - the string stays generically constrained, not an error.
        let rx = json_schema_to_regex(&serde_json::json!({"type":"string","format":"hostname"}))
            .unwrap();
        assert!(
            full_match(&rx, "\"anything goes\""),
            "unknown format ignored"
        );
    }

    #[test]
    fn int_range_regex_matches_exactly_the_range() {
        let int_range_regex = |lo: i64, hi: i64| int_bound_regex(Some(lo), Some(hi));
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // EXHAUSTIVELY verify the regex accepts exactly the integers in [lo,hi] (the only safe check for a
        // range-to-regex). Small ranges so the brute force over a +-8 margin stays fast.
        let small = [
            (0i64, 9),
            (0, 0),
            (5, 5),
            (1, 12),
            (8, 123),
            (0, 255),
            (100, 150),
            (-5, 5),
            (-128, 127),
            (-99, -10),
            (-1, 1),
            (3, 3),
        ];
        for (lo, hi) in small {
            let rx = int_range_regex(lo, hi).unwrap();
            for v in (lo - 8)..=(hi + 8) {
                let want = v >= lo && v <= hi;
                assert_eq!(
                    full_match(&rx, &v.to_string()),
                    want,
                    "range [{lo},{hi}] on {v}: expected {want} (regex {rx})"
                );
            }
        }
        // a large range: boundary off-by-ones + interior/exterior samples.
        let rx = int_range_regex(0, 1_000_000).unwrap();
        for (v, want) in [
            (-1i64, false),
            (0, true),
            (1, true),
            (500_000, true),
            (999_999, true),
            (1_000_000, true),
            (1_000_001, false),
        ] {
            assert_eq!(full_match(&rx, &v.to_string()), want, "1e6 range on {v}");
        }
        // canonical-form rejects: no leading zeros, digits only, non-empty.
        let rx = int_range_regex(0, 100).unwrap();
        assert!(!full_match(&rx, "007") && !full_match(&rx, "1a") && !full_match(&rx, ""));
        assert!(int_range_regex(5, 2).is_err(), "min > max errors");
    }

    #[test]
    fn int_bound_regex_handles_one_sided_and_open() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // One-sided bounds are infinite but regular. Verify exactly over a window wide enough to span the
        // bound and several digit-count boundaries (1, 2, 3 digits) on both sides.
        let lows = [
            Some(0i64),
            Some(1),
            Some(5),
            Some(10),
            Some(100),
            Some(-7),
            Some(-100),
        ];
        let highs = [
            Some(0i64),
            Some(9),
            Some(12),
            Some(100),
            Some(-1),
            Some(-50),
        ];
        for lo in lows {
            let rx = int_bound_regex(lo, None).unwrap();
            for v in -250i64..=250 {
                let want = v >= lo.unwrap();
                assert_eq!(
                    full_match(&rx, &v.to_string()),
                    want,
                    ">= {lo:?} on {v} (regex {rx})"
                );
            }
        }
        for hi in highs {
            let rx = int_bound_regex(None, hi).unwrap();
            for v in -250i64..=250 {
                let want = v <= hi.unwrap();
                assert_eq!(
                    full_match(&rx, &v.to_string()),
                    want,
                    "<= {hi:?} on {v} (regex {rx})"
                );
            }
        }
        // No bounds: any canonical integer (positive, negative, zero), no leading zeros.
        let rx = int_bound_regex(None, None).unwrap();
        for s in ["0", "-1", "42", "-1000", "999999999999"] {
            assert!(full_match(&rx, s), "unbounded should accept {s}");
        }
        assert!(!full_match(&rx, "007") && !full_match(&rx, "1.5") && !full_match(&rx, ""));
    }

    #[test]
    fn json_schema_to_regex_integer_bounds() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // inclusive minimum/maximum.
        let rx =
            json_schema_to_regex(&serde_json::json!({"type":"integer","minimum":1,"maximum":5}))
                .unwrap();
        for v in 1..=5 {
            assert!(full_match(&rx, &v.to_string()));
        }
        assert!(!full_match(&rx, "0") && !full_match(&rx, "6"));
        // exclusive forms (draft-07 numeric): >0 and <5 -> [1,4].
        let rx = json_schema_to_regex(
            &serde_json::json!({"type":"integer","exclusiveMinimum":0,"exclusiveMaximum":5}),
        )
        .unwrap();
        assert!(full_match(&rx, "1") && full_match(&rx, "4"));
        assert!(!full_match(&rx, "0") && !full_match(&rx, "5"));
        // one-sided `minimum` -> integers >= 3 (an open upper range, now enforced exactly).
        let rx = json_schema_to_regex(&serde_json::json!({"type":"integer","minimum":3})).unwrap();
        assert!(
            !full_match(&rx, "1") && !full_match(&rx, "2"),
            "below the minimum"
        );
        assert!(
            full_match(&rx, "3") && full_match(&rx, "1000"),
            "at/above the minimum"
        );
        // one-sided `maximum` -> integers <= 10 (all negatives included, nothing above 10).
        let rx = json_schema_to_regex(&serde_json::json!({"type":"integer","maximum":10})).unwrap();
        assert!(
            full_match(&rx, "10") && full_match(&rx, "-99"),
            "at/below the maximum"
        );
        assert!(!full_match(&rx, "11"), "above the maximum");
    }

    #[test]
    fn json_schema_number_bounds_error_not_silently_ignored() {
        // `number` with NO bounds is fine (an unbounded JSON number).
        assert!(json_schema_to_regex(&serde_json::json!({"type":"number"})).is_ok());
        // But range bounds on a `number` must ERROR (the documented contract - float-range regex is not
        // modeled, so we fail loud instead of silently emitting an unbounded number that ignores the bounds).
        for k in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
            let err = json_schema_to_regex(&serde_json::json!({"type":"number", k: 1.5}))
                .expect_err(&format!(
                    "number + {k} must error, not silently under-constrain"
                ));
            assert!(
                format!("{err:#}").contains("number"),
                "error should name the number-bounds limitation, got: {err:#}"
            );
        }
        // `integer` bounds are still supported (a different, expressible path) - sanity that we didn't break it.
        assert!(
            json_schema_to_regex(&serde_json::json!({"type":"integer","minimum":1,"maximum":5}))
                .is_ok()
        );
    }

    #[test]
    fn json_schema_to_regex_string_pattern() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // a string field constrained to a 4-digit code; the pattern is a FULL match (anchored by the quotes).
        let rx = json_schema_to_regex(&serde_json::json!({"type":"string","pattern":"[0-9]{4}"}))
            .unwrap();
        assert!(full_match(&rx, r#""1234""#));
        assert!(!full_match(&rx, r#""123""#), "too short");
        assert!(!full_match(&rx, r#""12345""#), "too long (full match)");
        assert!(!full_match(&rx, r#""abcd""#), "wrong char class");
        assert!(!full_match(&rx, "1234"), "must be a quoted JSON string");

        // surrounding ^...$ anchors are stripped (the quotes already anchor).
        let rx2 =
            json_schema_to_regex(&serde_json::json!({"type":"string","pattern":"^[A-Z]{2}$"}))
                .unwrap();
        assert!(full_match(&rx2, r#""US""#) && !full_match(&rx2, r#""USA""#));

        // a nested pattern field inside an object property.
        let obj = serde_json::json!({
            "type":"object",
            "properties":{"date":{"type":"string","pattern":"[0-9]{4}-[0-9]{2}-[0-9]{2}"}},
            "required":["date"]
        });
        let rxo = json_schema_to_regex(&obj).unwrap();
        assert!(full_match(&rxo, r#"{"date":"2026-06-23"}"#));
        assert!(!full_match(&rxo, r#"{"date":"2026/06/23"}"#));
    }

    #[test]
    fn json_schema_to_regex_size_bounds() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // array minItems/maxItems: 1..2 integers.
        let arr = serde_json::json!({
            "type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2
        });
        let ra = json_schema_to_regex(&arr).unwrap();
        assert!(full_match(&ra, "[5]") && full_match(&ra, "[5,6]"));
        assert!(full_match(&ra, "[5, 6]"), "whitespace allowed");
        assert!(!full_match(&ra, "[]"), "below minItems");
        assert!(!full_match(&ra, "[5,6,7]"), "above maxItems");

        // maxItems:0 -> empty array only.
        let empty = serde_json::json!({"type":"array","items":{"type":"integer"},"maxItems":0});
        let re0 = json_schema_to_regex(&empty).unwrap();
        assert!(full_match(&re0, "[]") && !full_match(&re0, "[1]"));

        // string minLength/maxLength: exactly 2..4 non-escape chars.
        let s = serde_json::json!({"type":"string","minLength":2,"maxLength":4});
        let rs = json_schema_to_regex(&s).unwrap();
        assert!(full_match(&rs, r#""ab""#) && full_match(&rs, r#""abcd""#));
        assert!(!full_match(&rs, r#""a""#) && !full_match(&rs, r#""abcde""#));

        // unbounded string is unchanged (escapes still allowed).
        let su = json_schema_to_regex(&serde_json::json!({"type":"string"})).unwrap();
        assert!(
            full_match(&su, r#""a\"b""#),
            "escapes allowed when unbounded"
        );

        // invalid bounds error rather than producing a bad regex.
        assert!(
            json_schema_to_regex(&serde_json::json!({"type":"string","minLength":5,"maxLength":2}))
                .is_err()
        );
    }

    #[test]
    fn json_schema_to_regex_const_enum_oneof() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // non-string enum (integers).
        let ints = serde_json::json!({ "enum": [1, 2, 30] });
        let rx = json_schema_to_regex(&ints).unwrap();
        assert!(full_match(&rx, "2") && full_match(&rx, "30"));
        assert!(!full_match(&rx, "3") && !full_match(&rx, "\"2\""));

        // const (a fixed scalar; a `.` in the number must be regex-escaped, not a wildcard).
        let c = serde_json::json!({ "const": 4.2 });
        let rxc = json_schema_to_regex(&c).unwrap();
        assert!(full_match(&rxc, "4.2"));
        assert!(
            !full_match(&rxc, "412"),
            "the dot is a literal, not a wildcard"
        );

        // oneOf: a property that is a string OR null (a common "nullable" union).
        let union = serde_json::json!({
            "type": "object",
            "properties": { "v": { "oneOf": [ {"type": "string"}, {"const": null} ] } },
            "required": ["v"]
        });
        let rxu = json_schema_to_regex(&union).unwrap();
        assert!(full_match(&rxu, r#"{"v":"hi"}"#));
        assert!(full_match(&rxu, r#"{"v":null}"#));
        assert!(!full_match(&rxu, r#"{"v":3}"#), "neither string nor null");

        // composite const: a fixed object is matched structurally with SORTED keys and optional whitespace.
        let cobj = serde_json::json!({ "const": {"b": 2, "a": 1} });
        let rxo = json_schema_to_regex(&cobj).unwrap();
        assert!(
            full_match(&rxo, r#"{"a":1,"b":2}"#),
            "sorted-key canonical form"
        );
        assert!(
            full_match(&rxo, "{ \"a\" : 1 , \"b\" : 2 }"),
            "insignificant whitespace ok"
        );
        assert!(!full_match(&rxo, r#"{"a":1,"b":3}"#), "wrong value");
        assert!(!full_match(&rxo, r#"{"a":1}"#), "missing key");

        // composite enum: an array literal, and an enum over composites.
        let carr = serde_json::json!({ "const": [1, "x", true] });
        let rxa = json_schema_to_regex(&carr).unwrap();
        assert!(full_match(&rxa, r#"[1,"x",true]"#));
        assert!(full_match(&rxa, "[ 1 , \"x\" , true ]"));
        assert!(!full_match(&rxa, r#"[1,"x",false]"#));
        assert!(!full_match(&rxa, r#"[1,"x"]"#), "wrong length");

        let cenum = serde_json::json!({ "enum": [ {"a": 1}, [2, 3] ] });
        let rxe = json_schema_to_regex(&cenum).unwrap();
        assert!(full_match(&rxe, r#"{"a":1}"#) && full_match(&rxe, r#"[2,3]"#));
        assert!(!full_match(&rxe, r#"{"a":2}"#) && !full_match(&rxe, r#"[2,4]"#));
    }

    #[test]
    fn json_schema_to_regex_honors_required_optional_properties() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // sorted keys: age, name. required = [name] -> `name` mandatory, `age` optional (and age sorts FIRST,
        // so it exercises the first-present alternation: age may lead, or name may be the first present).
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "age": {"type": "integer"}, "name": {"type": "string"} },
            "required": ["name"]
        });
        let rx = json_schema_to_regex(&schema).unwrap();
        assert!(full_match(&rx, r#"{"name":"bob"}"#), "optional age omitted");
        assert!(
            full_match(&rx, r#"{"age":3,"name":"bob"}"#),
            "optional age present (in sorted order)"
        );
        assert!(
            !full_match(&rx, r#"{"age":3}"#),
            "required name must appear"
        );
        assert!(
            !full_match(&rx, r#"{}"#),
            "required name must appear (empty)"
        );
        assert!(
            !full_match(&rx, r#"{"name":"bob","age":3}"#),
            "properties must be in sorted-key order"
        );

        // required:[] -> a fully optional object: every subset (incl. {}) conforms, in sorted order.
        let opt = serde_json::json!({
            "type": "object",
            "properties": { "a": {"type": "boolean"}, "b": {"type": "integer"} },
            "required": []
        });
        let rxo = json_schema_to_regex(&opt).unwrap();
        for s in [
            r#"{}"#,
            r#"{"a":true}"#,
            r#"{"b":7}"#,
            r#"{"a":false,"b":7}"#,
        ] {
            assert!(full_match(&rxo, s), "all-optional should accept {s}");
        }
        assert!(!full_match(&rxo, r#"{"b":7,"a":true}"#), "wrong order");

        // no `required` field -> backward-compatible all-required behavior (a missing property is rejected).
        let allreq = serde_json::json!({
            "type": "object",
            "properties": { "a": {"type": "boolean"}, "b": {"type": "integer"} }
        });
        let rxr = json_schema_to_regex(&allreq).unwrap();
        assert!(full_match(&rxr, r#"{"a":true,"b":7}"#));
        assert!(
            !full_match(&rxr, r#"{"a":true}"#),
            "all required by default"
        );
    }

    #[test]
    fn json_schema_to_regex_resolves_local_refs() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // The Pydantic/OpenAI-SDK shape: properties point at `$defs` via `$ref`. A `$ref` is inlined as the
        // referenced subschema, so the constraint is identical to writing the subschema inline.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "home": { "$ref": "#/$defs/Address" },
                "work": { "$ref": "#/$defs/Address" }
            },
            "required": ["home", "work"],
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": { "city": {"type": "string"}, "zip": {"type": "integer"} },
                    "required": ["city", "zip"]
                }
            }
        });
        let rx = json_schema_to_regex(&schema).unwrap();
        // sorted keys throughout: home<work, city<zip. The same $def is inlined twice (a diamond, not a cycle).
        assert!(full_match(
            &rx,
            r#"{"home":{"city":"AMS","zip":1011},"work":{"city":"BER","zip":10115}}"#
        ));
        assert!(
            !full_match(
                &rx,
                r#"{"home":{"city":"AMS"},"work":{"city":"BER","zip":10115}}"#
            ),
            "home.zip is required by the referenced subschema"
        );

        // `definitions` (draft-7 spelling) and a ref at the document root also resolve.
        let root_ref = serde_json::json!({
            "$ref": "#/definitions/N",
            "definitions": { "N": {"type": "integer", "minimum": 1, "maximum": 9} }
        });
        let rxn = json_schema_to_regex(&root_ref).unwrap();
        assert!(full_match(&rxn, "7"));
        assert!(!full_match(&rxn, "0"), "below the referenced range");

        // A recursive ref (a node referring to itself) is unbounded - rejected, not looped forever.
        let recursive = serde_json::json!({
            "$ref": "#/$defs/Tree",
            "$defs": {
                "Tree": {
                    "type": "object",
                    "properties": { "child": { "$ref": "#/$defs/Tree" } },
                    "required": ["child"]
                }
            }
        });
        let err = json_schema_to_regex(&recursive).unwrap_err().to_string();
        assert!(err.contains("recursive"), "got: {err}");

        // An unresolvable ref and an external ref both error rather than silently dropping the constraint.
        assert!(
            json_schema_to_regex(&serde_json::json!({"$ref": "#/$defs/Missing"})).is_err(),
            "dangling ref"
        );
        assert!(
            json_schema_to_regex(&serde_json::json!({"$ref": "https://example.com/s.json"}))
                .is_err(),
            "external ref"
        );
    }

    // Property fuzzer: a random schema from the supported subset, paired with a conforming-by-construction
    // instance, must be accepted by the compiled regex in both compact and pretty-printed form (the
    // latter also stresses optional-whitespace handling). Targets over-constraining (a conforming value
    // wrongly rejected), the dominant compiler-bug class.
    #[test]
    fn json_schema_to_regex_accepts_generated_conforming_json() {
        use serde_json::{Value, json};

        // deterministic xorshift64 PRNG (seeded per iteration; no external rng dep, replayable).
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                x
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next() % n.max(1)
            }
            fn coin(&mut self) -> bool {
                self.below(2) == 0
            }
        }

        // a random schema node; `depth` bounds nesting (scalars only at depth 0, keeping it regular + small).
        fn gen_schema(rng: &mut Rng, depth: u32) -> Value {
            let kinds = if depth == 0 { 4 } else { 7 };
            match rng.below(kinds) {
                0 => {
                    let mut m = serde_json::Map::new();
                    m.insert("type".into(), json!("integer"));
                    let a = rng.below(40) as i64 - 20;
                    let b = a + rng.below(40) as i64;
                    match rng.below(4) {
                        1 => {
                            m.insert("minimum".into(), json!(a));
                        }
                        2 => {
                            m.insert("maximum".into(), json!(b));
                        }
                        3 => {
                            m.insert("minimum".into(), json!(a));
                            m.insert("maximum".into(), json!(b));
                        }
                        _ => {}
                    }
                    Value::Object(m)
                }
                1 => json!({"type": "boolean"}),
                2 => {
                    let mut m = serde_json::Map::new();
                    m.insert("type".into(), json!("string"));
                    let min = rng.below(4);
                    let max = min + rng.below(4);
                    match rng.below(3) {
                        1 => {
                            m.insert("minLength".into(), json!(min));
                        }
                        2 => {
                            m.insert("minLength".into(), json!(min));
                            m.insert("maxLength".into(), json!(max));
                        }
                        _ => {}
                    }
                    Value::Object(m)
                }
                3 => {
                    let pool = [
                        json!("x"),
                        json!("y"),
                        json!("z"),
                        json!(1),
                        json!(2),
                        json!(true),
                    ];
                    let n = (2 + rng.below(3) as usize).min(pool.len());
                    let mut idxs: Vec<usize> = (0..pool.len()).collect();
                    let mut vars = Vec::new();
                    for _ in 0..n {
                        let j = rng.below(idxs.len() as u64) as usize;
                        vars.push(pool[idxs.remove(j)].clone());
                    }
                    json!({ "enum": vars })
                }
                4 => {
                    let item = gen_schema(rng, depth.saturating_sub(1));
                    let mut m = serde_json::Map::new();
                    m.insert("type".into(), json!("array"));
                    m.insert("items".into(), item);
                    let min = rng.below(3);
                    let max = min + rng.below(3);
                    match rng.below(3) {
                        1 => {
                            m.insert("minItems".into(), json!(min));
                        }
                        2 => {
                            m.insert("minItems".into(), json!(min));
                            m.insert("maxItems".into(), json!(max));
                        }
                        _ => {}
                    }
                    Value::Object(m)
                }
                5 => {
                    let names = ["a", "b", "c"];
                    let nprops = 1 + rng.below(3) as usize;
                    let mut props = serde_json::Map::new();
                    let mut req = Vec::new();
                    for &name in &names[..nprops] {
                        props.insert(name.into(), gen_schema(rng, depth.saturating_sub(1)));
                        if rng.coin() {
                            req.push(json!(name));
                        }
                    }
                    json!({ "type": "object", "properties": props, "required": req })
                }
                _ => {
                    let n = 2 + rng.below(2) as usize;
                    let subs: Vec<Value> = (0..n)
                        .map(|_| gen_schema(rng, depth.saturating_sub(1)))
                        .collect();
                    json!({ "oneOf": subs })
                }
            }
        }

        fn rand_string(rng: &mut Rng, len: u64) -> String {
            // no quote/backslash: fits both the length-bounded `[^"\\]{m,n}` and unbounded string patterns.
            const CH: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
            (0..len)
                .map(|_| CH[rng.below(CH.len() as u64) as usize] as char)
                .collect()
        }

        // a value that conforms to `schema` by construction.
        fn gen_instance(rng: &mut Rng, schema: &Value) -> Value {
            if let Some(Value::Array(vars)) = schema.get("enum") {
                return vars[rng.below(vars.len() as u64) as usize].clone();
            }
            if let Some(Value::Array(subs)) = schema.get("oneOf") {
                let pick = rng.below(subs.len() as u64) as usize;
                return gen_instance(rng, &subs[pick]);
            }
            match schema.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                "boolean" => json!(rng.coin()),
                "integer" => {
                    let lo = schema.get("minimum").and_then(|v| v.as_i64());
                    let hi = schema.get("maximum").and_then(|v| v.as_i64());
                    let v = match (lo, hi) {
                        (Some(lo), Some(hi)) => lo + rng.below((hi - lo + 1) as u64) as i64,
                        (Some(lo), None) => lo + rng.below(50) as i64,
                        (None, Some(hi)) => hi - rng.below(50) as i64,
                        (None, None) => rng.below(100) as i64 - 50,
                    };
                    json!(v)
                }
                "string" => {
                    let min = schema
                        .get("minLength")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let max = schema
                        .get("maxLength")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(min + 3);
                    let len = min + rng.below(max - min + 1);
                    json!(rand_string(rng, len))
                }
                "array" => {
                    let items = schema.get("items").unwrap();
                    let min = schema.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0);
                    let max = schema
                        .get("maxItems")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(min + 2);
                    let k = min + rng.below(max - min + 1);
                    Value::Array((0..k).map(|_| gen_instance(rng, items)).collect())
                }
                "object" => {
                    let props = schema
                        .get("properties")
                        .and_then(|p| p.as_object())
                        .unwrap();
                    let req: std::collections::HashSet<&str> = schema
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                        .unwrap_or_default();
                    let mut obj = serde_json::Map::new();
                    for (name, psch) in props {
                        if req.contains(name.as_str()) || rng.coin() {
                            obj.insert(name.clone(), gen_instance(rng, psch));
                        }
                    }
                    Value::Object(obj)
                }
                _ => json!(null),
            }
        }

        let mut failures = 0usize;
        for i in 0..400u64 {
            let seed =
                0x9E37_79B9_7F4A_7C15u64 ^ i.wrapping_mul(0xD1B5_4A32_D192_ED03).wrapping_add(1);
            let mut rng = Rng(seed | 1);
            let schema = gen_schema(&mut rng, 3);
            // every generated construct is in the supported subset, so compilation must succeed.
            let pattern = json_schema_to_regex(&schema)
                .unwrap_or_else(|e| panic!("schema did not compile: {e}\nschema: {schema}"));
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z"))
                .unwrap_or_else(|e| panic!("emitted an invalid regex: {e}\npattern: {pattern}"));
            for _ in 0..4 {
                let inst = gen_instance(&mut rng, &schema);
                let compact = serde_json::to_string(&inst).unwrap();
                if !re.is_match(&compact) {
                    eprintln!(
                        "REJECTED conforming (compact): {compact}\n  schema:  {schema}\n  pattern: {pattern}"
                    );
                    failures += 1;
                }
                let pretty = serde_json::to_string_pretty(&inst).unwrap();
                if !re.is_match(&pretty) {
                    eprintln!(
                        "REJECTED conforming (pretty): {pretty:?}\n  schema:  {schema}\n  pattern: {pattern}"
                    );
                    failures += 1;
                }
            }
        }
        assert_eq!(
            failures, 0,
            "{failures} conforming instances were wrongly rejected"
        );
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
                // top-p collapse (card 551b): the exact real-number nucleus for a tiny
                // `top_p` can differ from the production quantized bisection's by one low-weight
                // candidate (ADR-0114: top-p's membership boundary is tier 2), so this asserts
                // against `truncated_candidates`'s own quantized nucleus (now the same semantics
                // `pick_from` draws through) instead of real-valued `p_top` math - a genuine
                // cross-check between this Rust mirror and `pick_from`'s independent
                // `eval_sample_token` call, not a tautology.
                if build().truncated_candidates(&logits).len() == 1 {
                    assert_eq!(
                        pick, argmax,
                        "a singleton quantized top-p/min-p/top-k nucleus must be the pick"
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
        // `probs` must be the SAME distribution `pick_from`'s Gumbel-max draw samples from,
        // including a real top-p nucleus with NO other truncation active (SC-002, card 677):
        // `truncated_candidates`'s top-p cut mirrors the device's quantized tier-2 bisection
        // (ADR-0114), the same semantics `pick_from` draws through via `eval_sample_token`, so
        // `probs()` and `pick_from()` agree on which candidates even have nonzero mass - including
        // when top-p alone does the truncating (min_p=0, top_k=0 here), which needs the oracle's
        // pre-top-p floor to start from a finite bound rather than `-inf` (card 677, found by card
        // 551b's review). Check by drawing many samples with `pick` and comparing the empirical
        // histogram to `probs`'s output via chi-squared (deterministic seed, no flakiness).
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
        assert!(a.is_simple_temperature());
        assert_eq!(a.top_k, 5);
        assert_eq!(a.top_p, 0.9);
        assert_eq!(a.min_p, 0.1);
    }
}
