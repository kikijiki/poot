//! Speculative decoding's pure pieces: the prompt-lookup proposer and the speculative-sampling
//! accept/reject arithmetic. The Runner entry points that drove them are deleted with the dense Runner
//! (POOT-737); POOT-751 moves these functions into `driver/speculate.rs` and deletes this file. The holds
//! sit on the two roots, [`spec_sample_round`] and [`propose_lookup`]: what only a held root calls is
//! dead with it, so the compiler reports the root alone.

use crate::core::sampler::{Sampler, SamplerFault};

/// Speculative sampling's accept/reject test (Leviathan et al. 2023 Algorithm 1; Chen et al. 2023):
/// whether to accept a draft token proposed with probability `q_tok` under the draft, given the target's
/// probability `p_tok` and a uniform draw `u` in `[0, 1)`. Accepts with probability `min(1, p_tok/q_tok)`.
/// `q_tok <= 0.0` (a float-underflow edge case; a real draw has `q_tok > 0`) is treated as an infinite
/// ratio, i.e. always accept.
pub(crate) fn spec_accept(p_tok: f32, q_tok: f32, u: f32) -> bool {
    q_tok <= 0.0 || u < (p_tok / q_tok).min(1.0)
}

/// The residual distribution resampled from after a rejected draft token: `max(0, p(x) - q(x))`
/// normalized to sum to 1. Resampling from this rather than `p` is what makes the whole procedure
/// sample exactly from `p`: the accept step already spends `min(p(x), q(x))` of `p`'s mass. Falls back to
/// `p` if the residual mass is numerically <= 0 (only when `p == q` everywhere, so the accept test never
/// rejects in exact arithmetic; a float-rounding safety net).
pub(crate) fn residual_probs(p: &[f32], q: &[f32]) -> Vec<f32> {
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

/// One round's accept/reject/residual bookkeeping for sampled speculative decoding, a pure GPU/model-free
/// function. Inputs: the draft's
/// proposals `drafts`, its per-position distributions `q` (`q[i]` at the moment it proposed
/// `drafts[i]`), and the target's `p` (`p[i]` is for the token after `drafts[..i]`, so
/// `p.len() == drafts.len() + 1`). Walks the drafts left to right, accepting via [`spec_accept`] until the
/// first rejection (resampled from [`residual_probs`]) or the drafts run out (a fresh draw from `p`'s last
/// row). Returns `(accepted_count, next_token)`, or a [`SamplerFault`] when a distribution row holds a NaN
/// or infinity. All draws come from `sampler`, deterministically, one call per round.
#[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-751"))]
pub(crate) fn spec_sample_round(
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
#[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-751"))]
pub(crate) fn propose_lookup(tokens: &[u32], ngram: usize, max_k: usize) -> Vec<u32> {
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
/// [`spec_sample_round`]). The bar is distributional: an empirical chi-squared test checks that
/// accept/residual reproduces the target's distribution `p`. Fixed seeds, so deterministic.
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
                let x = draft_sampler.sample_from_probs(&q).unwrap() as usize;
                let u = target_sampler.uniform();
                let out = if spec_accept(p[x], q[x], u) {
                    x
                } else {
                    let r = residual_probs(&p, &q);
                    target_sampler.sample_from_probs(&r).unwrap() as usize
                };
                counts[out] += 1;
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
