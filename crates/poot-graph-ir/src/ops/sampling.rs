//! Sampling primitives (card 551a, R472-007, deval.md section 8): `hash32` (the integer mix
//! [`OpKind::RandomUniform`] and the oracle share), the Gumbel-noise composition, and [`sample_head`],
//! the suffix a driver appends after a graph's raw logits.
//!
//! Everything a request can ask of its sampler runs in that suffix (POOT-1268): per-sequence token counts
//! and logit bias kept on the device, the repetition, presence and frequency penalties, a guided-decoding
//! mask, the draw, and the top-n logprobs of the adjusted distribution. All of it is composed from the
//! primitives below and the one [`OpKind::SampleToken`] draw, so it lowers on every backend without a
//! kernel of its own. The exact host sampler stays the reference the suffix is held to.

use std::num::NonZeroUsize;

use super::{
    BinOp, Builder, BuilderError, DType, RedOp, Requirement, Scalar, TensorType, Traced, UnOp,
};
use crate::graph::{Slot, StateRole};
use crate::op::SampleRule;
use crate::range::IndexRange;

/// The integer hash `RandomUniform` and its CPU oracle both compute (murmur3's `fmix32` finalizer,
/// mixed with the position via an odd golden-ratio multiplier): deterministic, counter-based, no RNG
/// state. A pure `fn` (not a graph composition - `OpKind::RandomUniform` is one primitive, not built
/// from smaller ops), so this is the one place the formula is written; the oracle's counterpart in
/// `poot-eval` calls it by value.
pub const fn hash32(seed: u32, idx: u32) -> u32 {
    let mut x = seed ^ idx.wrapping_mul(0x9E3779B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EBCA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2AE35);
    x ^= x >> 16;
    x
}

/// Standard Gumbel(0,1) noise from a uniform `u` in `(0, 1)` (deval.md section 8.1): `-log(-log(u))`,
/// an ordinary composition of [`UnOp::Log`] and [`UnOp::Neg`] that `fuse` folds into the rest of the
/// sampler suffix's elementwise kernel. Device-versus-oracle agreement on this transform is tier 2
/// (ADR-0101): exact only when both sides are handed the identical `u`.
pub fn gumbel(b: &Builder, u: Traced) -> Result<Traced, BuilderError> {
    let neg_log_u = b.unary(UnOp::Neg, b.unary(UnOp::Log, u)?)?;
    b.unary(UnOp::Neg, b.unary(UnOp::Log, neg_log_u)?)
}

/// How far a head adjusts the logits before the draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Adjust {
    /// The raw logits.
    None,
    /// The repetition, presence and frequency penalties and the logit bias, read from the per-sequence
    /// history the head keeps on the device ([`HISTORY_STATE`]).
    History,
    /// [`Adjust::History`] and a guided-decoding mask, a `[rows, vocab]` input the host builds each step.
    Guided,
}

/// What a sampler suffix computes besides the draw. One compiled head per distinct spec: a request whose
/// knobs a head covers runs it, a row that needs less binds the neutral value of each input. Ordered by the
/// draw, then the adjustment, then the alternatives: the cheapest head first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleSpec {
    /// The draw. A step mixing rows of different rules runs the widest one; a row's own knobs turn the
    /// truncations it does not use off.
    pub rule: SampleRule,
    /// What the head applies to the logits before the draw; any adjustment reads and updates the
    /// device-resident sequence history ([`HISTORY_STATE`]).
    pub adjust: Adjust,
    /// Top-n logprobs of the adjusted distribution beside the token (`Some(0)`: the drawn token's
    /// logprob only). The result is then the F32 record [`LogprobRecord`] describes.
    pub logprobs: Option<usize>,
}

impl SampleSpec {
    /// The plain draw of `rule`: no adjustment, no logprobs.
    pub const fn draw(rule: SampleRule) -> Self {
        Self {
            rule,
            adjust: Adjust::None,
            logprobs: None,
        }
    }

    /// The same head with the penalties and the logit bias applied.
    #[must_use]
    pub const fn adjusted(self) -> Self {
        Self {
            adjust: Adjust::History,
            ..self
        }
    }

    /// The same head with the penalties, the logit bias and a guided-decoding mask applied.
    #[must_use]
    pub const fn guided(self) -> Self {
        Self {
            adjust: Adjust::Guided,
            ..self
        }
    }

    /// The same head with top-`n` logprobs.
    #[must_use]
    pub const fn with_logprobs(self, n: usize) -> Self {
        Self {
            logprobs: Some(n),
            ..self
        }
    }
}

/// The name of the carried per-sequence history state an adjusted head declares: `[pool_rows, 2, vocab]`
/// F32, channel 0 the token counts (prompt and generated) the penalties read, channel 1 the additive logit
/// bias. Every entry that declares it shares one buffer, because the executor shares carried state by
/// name and aval.
pub const HISTORY_STATE: &str = "sampler.history";

/// Tag of the `[rows, 3]` F32 `(repetition, presence, frequency)` penalty slot. The neutral row is
/// `(1, 0, 0)`.
pub const PENALTY_TAG: &str = "penalty";
/// Tag of the `[rows, vocab]` F32 additive guided-decoding mask slot of an [`Adjust::Guided`] head: `0` for
/// an allowed token, [`MASKED_LOGIT`] for an excluded one. The neutral row is all zeros.
pub const MASK_TAG: &str = "mask";
/// Tag of the `[rows]` I32 slot naming the history pool row each step row reads.
pub const HISTORY_READ_TAG: &str = "history_read";
/// Tag of the `[rows]` forward write map naming the history pool row each step row writes, `-1` for a
/// row whose draw is not kept (a prefill chunk, a masked row).
pub const HISTORY_WRITE_TAG: &str = "history_write";

/// The finite stand-in a caller writes for a token it excluded from the row (a guided-decoding mask).
/// `SampleToken` flags a non-finite logit, so a masked token cannot be `-inf`; it is the lowest finite
/// f32 instead, and the bisection floor ignores it so top-k and top-p search the real logits only.
pub const MASKED_LOGIT: f32 = f32::MIN;

/// The layout of the F32 record a head with logprobs publishes, per row:
/// `[token, non_finite_index, chosen_logprob, (alternative_token, alternative_logprob) x n]`.
/// Token ids and indices ride as exact F32 integers (a vocabulary is below 2^24, which the greedy pick
/// already requires). The alternatives are the `n` highest adjusted logits, most probable first, the
/// lower index first on a tie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogprobRecord {
    /// The alternatives per row.
    pub n: usize,
}

impl LogprobRecord {
    /// Columns before the alternatives: token, non-finite index, the drawn token's logprob.
    pub const HEADER: usize = 3;

    /// F32 words per row.
    pub const fn width(self) -> usize {
        Self::HEADER + 2 * self.n
    }
}

/// A finished sampler suffix (card 551a, R-551a-3/R-551a-4): the result, plus the `Slot::Sampler` inputs
/// this call declared (`None` for [`SampleRule::Greedy`], which reads only `logits`). A caller binds
/// `seed`/`params`/`top_k` by their [`crate::graph::SlotKey`] name (`"sampler.seed"`, `"sampler.params"`,
/// `"sampler.top_k"`); the driver wires that through its binder instead of exposing these fields.
pub struct SampleHead {
    /// The result: `[.., 2]` I32 `(token, non_finite)`, or the F32 [`LogprobRecord`] of a head with
    /// logprobs.
    pub out: Traced,
    /// The `sampler.seed` input; `None` for greedy sampling.
    pub seed: Option<Traced>,
    /// The `sampler.params` input; `None` for greedy sampling.
    pub params: Option<Traced>,
    /// The `sampler.top_k` input; only the two top-k rules declare it.
    pub top_k: Option<Traced>,
    /// The history state pair `(input, updated)` an adjusted head carries.
    pub history: Option<(Traced, Traced)>,
}

/// Append a sampling suffix after `logits` (`[rows, V]` F32 when the spec adjusts or records logprobs,
/// `[.., V]` otherwise): the graph suffix L7 asks for, so every backend samples the same way instead of a
/// per-backend fused entry point (R472-007). `spec.rule` selects the draw's operand set (see
/// [`SampleRule`]). `Greedy` needs nothing else. Every other rule declares one `Slot::Sampler` input per
/// tag it needs (R-551a-4: `seed` always, `params` always, `top_k` for the two `TopK` rules), builds the
/// `RandomUniform` noise over `logits`' shape from `seed`, and transforms it with [`gumbel`];
/// `sample_token` then maximizes `logits[i]*inv_temp + noise_scale*noise[i]` inside the kept set
/// (deval.md section 8.1).
///
/// An adjusted head (`spec.adjust`) first turns the raw logits into the adjusted ones the host sampler
/// draws from, in the host's order: the bias, the repetition, presence and frequency penalties of the
/// tokens the history has seen, then the guided-decoding mask when the head has one. A NaN or infinity in the raw row survives
/// every step (the mask is added, never selected), so the draw's non-finite flag still names it. The
/// history pool `[pool_rows, 2, V]` supplies the counts and the bias, and the head writes the counts back
/// with the drawn token added to the rows its write map names. `pool_rows` is required exactly when the
/// spec adjusts.
///
/// One call declares at most one occurrence of each tag, so a graph with more than one sampler suffix
/// (none does today) would need distinct tags - out of scope here.
pub fn sample_head(
    b: &Builder,
    logits: Traced,
    spec: SampleSpec,
    pool_rows: Option<NonZeroUsize>,
) -> Result<SampleHead, BuilderError> {
    let SampleSpec {
        rule,
        adjust,
        logprobs,
    } = spec;
    let mut shape = b.aval(logits)?.shape;
    // The adjustment and the logprobs are per step row: they take one position per row, `[rows, V]` or
    // the `[rows, 1, V]` a last-position decode step publishes.
    let logits = if adjust != Adjust::None || logprobs.is_some() {
        match *shape.as_slice() {
            [rows, 1, vocab] => {
                shape = vec![rows, vocab];
                b.reshape(logits, shape.clone())?
            }
            [_, _] => logits,
            _ => {
                return Err(BuilderError::Requirement {
                    operation: "sample_head",
                    requirement: Requirement::Rank {
                        expected: 2,
                        found: shape,
                    },
                });
            }
        }
    } else {
        logits
    };
    let rank = shape.len();
    let leading = shape[..rank - 1].to_vec();
    let vocab = super::last_axis("sample_head", &shape)?;

    let (z, history) = if adjust != Adjust::None {
        let pool_rows = pool_rows.ok_or(BuilderError::Requirement {
            operation: "sample_head",
            requirement: Requirement::AtLeastOne { what: "pool rows" },
        })?;
        let adjusted = adjust_logits(b, logits, pool_rows.get(), adjust == Adjust::Guided)?;
        (adjusted.logits, Some(adjusted))
    } else {
        (logits, None)
    };

    let mut seed = None;
    let mut params = None;
    let mut top_k = None;
    let token_pair = if matches!(rule, SampleRule::Greedy) {
        b.sample_token(rule, z, None, None, None)?
    } else {
        let seed_in = b.slot_named(
            Slot::Sampler,
            "seed",
            TensorType::new(leading.clone(), DType::I32),
        )?;
        let mut params_shape = leading.clone();
        params_shape.push(rule.params_cols());
        let params_in = b.slot_named(
            Slot::Sampler,
            "params",
            TensorType::new(params_shape, DType::F32),
        )?;
        let top_k_in = if rule.has_top_k() {
            Some(b.slot_named(
                Slot::Sampler,
                "top_k",
                TensorType::new(leading.clone(), DType::I32),
            )?)
        } else {
            None
        };
        let u = b.random_uniform(seed_in, vocab)?; // [leading.., vocab] == logits' shape
        let noise = gumbel(b, u)?;
        seed = Some(seed_in);
        params = Some(params_in);
        top_k = top_k_in;
        b.sample_token(rule, z, Some(noise), Some(params_in), top_k_in)?
    };

    let history = match history {
        Some(adjusted) => Some(adjusted.carry(b, token_pair)?),
        None => None,
    };
    let out = match logprobs {
        None => token_pair,
        Some(n) => logprob_record(b, z, token_pair, params, n)?,
    };
    Ok(SampleHead {
        out,
        seed,
        params,
        top_k,
        history,
    })
}

/// The inputs and intermediates of an adjusted head between the adjustment and the history update.
struct Adjusted {
    logits: Traced,
    pool: Traced,
    counts: Traced,
    bias: Traced,
    write: Traced,
}

/// The adjusted logits of an adjusted head, in the host sampler's order (`Sampler::adjust`, then the
/// guided mask): `x = raw + bias`; for a token with a positive count, `x` divided by the repetition
/// penalty when positive and multiplied when not, then minus the presence penalty, then minus the
/// frequency penalty times the count; finally plus the mask. A row with neutral inputs (`(1, 0, 0)`, a zero
/// bias row, a zero mask) is its raw logits exactly.
fn adjust_logits(
    b: &Builder,
    raw: Traced,
    pool_rows: usize,
    guided: bool,
) -> Result<Adjusted, BuilderError> {
    let shape = b.aval(raw)?.shape;
    let (rows, vocab) = (shape[0], shape[1]);
    let pool = b.state_input(
        HISTORY_STATE,
        TensorType::new(vec![pool_rows, 2, vocab], DType::F32),
        StateRole::Sampler,
    )?;
    let read = b.slot_named_ranged(
        Slot::Sampler,
        HISTORY_READ_TAG,
        TensorType::new(vec![rows], DType::I32),
        IndexRange::below(pool_rows).expect("a pool has rows"),
    )?;
    let write = b.write_map(HISTORY_WRITE_TAG, rows, pool_rows)?;
    let penalty = b.slot_named(
        Slot::Sampler,
        PENALTY_TAG,
        TensorType::new(vec![rows, 3], DType::F32),
    )?;
    let mask = if guided {
        Some(b.slot_named(
            Slot::Sampler,
            MASK_TAG,
            TensorType::new(vec![rows, vocab], DType::F32),
        )?)
    } else {
        None
    };

    let history = b.gather(pool, 0, read)?; // [rows, 2, vocab]
    let channel = |index: usize| -> Result<Traced, BuilderError> {
        b.reshape(b.slice(history, 1, index, index + 1)?, vec![rows, vocab])
    };
    let counts = channel(0)?;
    let bias = channel(1)?;
    let column = |index: usize| -> Result<Traced, BuilderError> {
        b.broadcast(b.slice(penalty, 1, index, index + 1)?, vec![rows, vocab])
    };
    let (repetition, presence, frequency) = (column(0)?, column(1)?, column(2)?);

    let biased = b.binary(BinOp::Add, raw, bias)?;
    let seen = b.binary_scalar(BinOp::Ge, counts, Scalar::F32(1.0))?; // 1.0 / 0.0
    let positive = b.binary_scalar(BinOp::Ge, biased, Scalar::F32(0.0))?;
    let penalized = blend(
        b,
        positive,
        b.binary(BinOp::Div, biased, repetition)?,
        b.binary(BinOp::Mul, biased, repetition)?,
    )?;
    let repeated = blend(b, seen, penalized, biased)?;
    let presented = b.binary(BinOp::Sub, repeated, b.binary(BinOp::Mul, presence, seen)?)?;
    let counted = b.binary(
        BinOp::Sub,
        presented,
        b.binary(BinOp::Mul, frequency, counts)?,
    )?;
    let logits = match mask {
        Some(mask) => b.binary(BinOp::Add, counted, mask)?,
        None => counted,
    };
    Ok(Adjusted {
        logits,
        pool,
        counts,
        bias,
        write,
    })
}

impl Adjusted {
    /// Add the drawn token (`token_pair[.., 0]`) to the counts of the rows the write map names and carry
    /// the pool: the bias channel passes through, every other pool row is untouched.
    fn carry(self, b: &Builder, token_pair: Traced) -> Result<(Traced, Traced), BuilderError> {
        let shape = b.aval(self.counts)?.shape;
        let (rows, vocab) = (shape[0], shape[1]);
        let token = b.cast(
            b.reshape(b.slice(token_pair, 1, 0, 1)?, vec![rows, 1])?,
            DType::F32,
        )?;
        let hit = one_hot(b, token, vocab)?;
        let counts = b.binary(BinOp::Add, self.counts, hit)?;
        let as_channel = |value: Traced| b.reshape(value, vec![rows, 1, vocab]);
        let next = b.concat(1, &[as_channel(counts)?, as_channel(self.bias)?])?;
        let updated = b.scatter_update(self.pool, next, self.write)?;
        Ok((self.pool, updated))
    }
}

/// `flag * if_true + (1 - flag) * if_false` for an exact `1.0` / `0.0` `flag` (a [`BinOp::Ge`] result): the
/// F32 select. An operand of the branch `flag` rejects must be finite or the product is NaN, which only
/// ever happens in a row that already carries a non-finite raw logit, and that row's fault is reported
/// from the raw value. (A float-to-integer cast has no lowering, so `Select`'s integer condition is not
/// reachable from a float comparison.)
fn blend(
    b: &Builder,
    flag: Traced,
    if_true: Traced,
    if_false: Traced,
) -> Result<Traced, BuilderError> {
    let complement = b.binary_scalar(
        BinOp::Add,
        b.binary_scalar(BinOp::Mul, flag, Scalar::F32(-1.0))?,
        Scalar::F32(1.0),
    )?;
    b.binary(
        BinOp::Add,
        b.binary(BinOp::Mul, flag, if_true)?,
        b.binary(BinOp::Mul, complement, if_false)?,
    )
}

/// `[rows, vocab]` F32 with `1.0` at each row's `token` (`[rows, 1]` F32, an exact integer) and `0.0`
/// elsewhere.
fn one_hot(b: &Builder, token: Traced, vocab: usize) -> Result<Traced, BuilderError> {
    let rows = b.aval(token)?.shape[0];
    let index = b.broadcast(
        b.reshape(b.iota(vocab)?, vec![1, vocab])?,
        vec![rows, vocab],
    )?;
    let token = b.broadcast(token, vec![rows, vocab])?;
    b.binary(
        BinOp::Mul,
        b.binary(BinOp::Ge, index, token)?,
        b.binary(BinOp::Ge, token, index)?,
    )
}

/// A `[rows, vocab]` F32 tensor of `value`.
fn splat(b: &Builder, value: f32, rows: usize, vocab: usize) -> Result<Traced, BuilderError> {
    let zero = b.binary_scalar(BinOp::Mul, b.iota(vocab)?, Scalar::F32(0.0))?;
    let filled = b.binary_scalar(BinOp::Add, zero, Scalar::F32(value))?;
    b.broadcast(b.reshape(filled, vec![1, vocab])?, vec![rows, vocab])
}

/// The F32 [`LogprobRecord`] of `n` alternatives over the adjusted logits `z` (`[rows, V]`), after the
/// draw `token_pair`. The logprob of token `i` is the natural-log softmax of `z` at the draw's
/// temperature: `(z[i] - max) * inv_temp - ln(sum exp((z - max) * inv_temp))`, `inv_temp` the row's
/// `params` column 0 (1 for a greedy suffix, whose host reference reports temperature 1). The
/// alternatives are `n` greedy picks of the not yet taken logits, so the order is the greedy rule's:
/// largest first, the lower index on a tie.
fn logprob_record(
    b: &Builder,
    z: Traced,
    token_pair: Traced,
    params: Option<Traced>,
    n: usize,
) -> Result<Traced, BuilderError> {
    let shape = b.aval(z)?.shape;
    let (rows, vocab) = (shape[0], shape[1]);
    let max = b.reduce(RedOp::Max, z, 1, true)?;
    let centered = b.binary(BinOp::Sub, z, b.broadcast(max, vec![rows, vocab])?)?;
    let scaled = match params {
        Some(params) => b.binary(
            BinOp::Mul,
            centered,
            b.broadcast(b.slice(params, 1, 0, 1)?, vec![rows, vocab])?,
        )?,
        None => centered,
    };
    let lse = b.unary(
        UnOp::Log,
        b.reduce(RedOp::Sum, b.unary(UnOp::Exp, scaled)?, 1, true)?,
    )?;
    // A masked token's logprob is below the finite range: clamp it so the one-hot sum below is finite.
    let logprob = b.binary_scalar(
        BinOp::Max,
        b.binary(BinOp::Sub, scaled, b.broadcast(lse, vec![rows, vocab])?)?,
        Scalar::F32(f32::MIN),
    )?;
    let at = |token: Traced| -> Result<Traced, BuilderError> {
        // `token` is `[rows, 1]` F32.
        b.reduce(
            RedOp::Sum,
            b.binary(BinOp::Mul, logprob, one_hot(b, token, vocab)?)?,
            1,
            true,
        )
    };
    let column =
        |value: Traced| -> Result<Traced, BuilderError> { b.reshape(value, vec![rows, 1]) };
    let drawn = b.cast(b.slice(token_pair, 1, 0, 1)?, DType::F32)?;
    let flag = b.cast(b.slice(token_pair, 1, 1, 2)?, DType::F32)?;

    let mut parts = vec![drawn, flag, at(drawn)?];
    let masked = splat(b, MASKED_LOGIT, rows, vocab)?;
    let mut remaining = z;
    for _ in 0..n {
        let top = b.sample_token(SampleRule::Greedy, remaining, None, None, None)?;
        let top = b.cast(b.slice(top, 1, 0, 1)?, DType::F32)?;
        parts.push(top);
        parts.push(at(top)?);
        remaining = blend(b, one_hot(b, top, vocab)?, masked, remaining)?;
    }
    let parts = parts
        .into_iter()
        .map(column)
        .collect::<Result<Vec<_>, _>>()?;
    b.concat(1, &parts)
}
