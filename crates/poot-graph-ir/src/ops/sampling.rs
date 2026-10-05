//! Sampling primitives (card 551a, R472-007, deval.md section 8): `hash32` (the integer mix
//! [`OpKind::RandomUniform`] and the oracle share), the Gumbel-noise composition, and [`sample_head`],
//! the suffix a driver (card 551b) appends after a graph's raw logits.

use super::*;
use crate::graph::Slot;
use crate::op::SampleRule;

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
pub fn gumbel(b: &Builder, u: Traced) -> Traced {
    let neg_log_u = b.unary(UnOp::Neg, b.unary(UnOp::Log, u));
    b.unary(UnOp::Neg, b.unary(UnOp::Log, neg_log_u))
}

/// A finished sampler suffix (card 551a, R-551a-3/R-551a-4): the `[.., 2]` I32 `(token, non_finite)`
/// output, plus the `Slot::Sampler` inputs this call declared (`None` for [`SampleRule::Greedy`], which
/// reads only `logits`). A caller binds `seed`/`params`/`top_k` by their [`crate::graph::SlotKey`] name
/// (`"sampler.seed"`, `"sampler.params"`, `"sampler.top_k"`); card 551b's driver wires that through
/// 550's binder instead of exposing these fields.
pub struct SampleHead {
    pub out: Traced,
    pub seed: Option<Traced>,
    pub params: Option<Traced>,
    pub top_k: Option<Traced>,
}

/// Append a sampling suffix after `logits` (`[.., V]` F32): the graph suffix L7 asks for, so every
/// backend samples the same way instead of a per-backend fused entry point (R472-007). `rule` selects
/// the operand set (see [`SampleRule`]). `Greedy` needs nothing else. Every other rule declares one
/// `Slot::Sampler` input per tag it needs (R-551a-4: `seed` always, `params` always, `top_k` for the two
/// `TopK` rules), builds the `RandomUniform` noise over `logits`' shape from `seed`, and transforms it
/// with [`gumbel`]; `sample_token` then maximizes `logits[i]*inv_temp + noise_scale*noise[i]` inside the
/// kept set (deval.md section 8.1). One call declares at most one occurrence of each tag, so a graph
/// with more than one sampler suffix (none does today) would need distinct tags - out of scope here.
pub fn sample_head(b: &Builder, logits: Traced, rule: SampleRule) -> SampleHead {
    if matches!(rule, SampleRule::Greedy) {
        let out = b.sample_token(rule, logits, None, None, None);
        return SampleHead {
            out,
            seed: None,
            params: None,
            top_k: None,
        };
    }
    let shape = b.aval(logits).shape;
    let rank = shape.len();
    let leading = shape[..rank - 1].to_vec();
    let vocab = *shape
        .last()
        .expect("sample_head: logits needs a vocab axis");

    let seed = b.slot_named(
        Slot::Sampler,
        "seed",
        TensorType::new(leading.clone(), DType::I32),
    );
    let params_cols = if matches!(rule, SampleRule::GumbelTopKTopP) {
        4
    } else {
        3
    };
    let mut params_shape = leading.clone();
    params_shape.push(params_cols);
    let params = b.slot_named(
        Slot::Sampler,
        "params",
        TensorType::new(params_shape, DType::F32),
    );
    let top_k = if matches!(rule, SampleRule::GumbelTopK | SampleRule::GumbelTopKTopP) {
        Some(b.slot_named(
            Slot::Sampler,
            "top_k",
            TensorType::new(leading.clone(), DType::I32),
        ))
    } else {
        None
    };

    let u = b.random_uniform(seed, vocab); // [leading.., vocab] == logits' shape
    let noise = gumbel(b, u);
    let out = b.sample_token(rule, logits, Some(noise), Some(params), top_k);
    SampleHead {
        out,
        seed: Some(seed),
        params: Some(params),
        top_k,
    }
}
