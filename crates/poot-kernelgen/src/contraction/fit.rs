//! The least-squares fit of the cost model's ratios to measured device times (the POOT-1403 method, `poot-bench
//! calibrate-launch`): the modelled time ([`CostTerms::modelled_time`]) is linear in the LDS-read and staged-value
//! ratios once the unit of time is left free, so a measured time per tiling regresses on the terms directly.

use super::CostTerms;

/// The ratios fitted to a set of measured tilings, and how well the model follows the measurements under them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostFit {
    /// The fitted [`CostRatios::lds_read`], in FMA issues (the unit).
    pub lds_read: f64,
    /// The fitted [`CostRatios::staged_value`].
    pub staged_value: f64,
    /// The time one FMA issue of the model takes on this device, in the unit of the samples.
    pub seconds_per_fma: f64,
    /// The fraction of the samples' variance (about their mean) the fitted model explains: 1 is a perfect fit.
    pub r_squared: f64,
}

/// Fit `time ~ a * f1 + b * f2 + c * f3` to `samples` (a tiling's terms and its measured time), where `f1` is the
/// FMA and rounding issues, `f2` the LDS reads and `f3` the staged values, each over the occupancy. The ratios are
/// `b / a` and `c / a`. `None` when the samples cannot determine them: fewer than three, terms that are collinear
/// (every sample the same tile geometry), or a non-positive FMA time.
pub fn fit_cost_ratios(samples: &[(CostTerms, f64)]) -> Option<CostFit> {
    if samples.len() < 3 {
        return None;
    }
    let features = |terms: &CostTerms| {
        [
            terms.occupancy * (terms.fma + terms.staged_per_lane * terms.rounding),
            terms.occupancy * terms.lds_reads,
            terms.occupancy * terms.staged_per_lane,
        ]
    };
    // The normal equations `(X^T X) w = X^T y`, solved by Gaussian elimination with partial pivoting.
    let mut matrix = [[0.0f64; 4]; 3];
    for (terms, time) in samples {
        let f = features(terms);
        for row in 0..3 {
            for col in 0..3 {
                matrix[row][col] += f[row] * f[col];
            }
            matrix[row][3] += f[row] * time;
        }
    }
    let scale = matrix
        .iter()
        .flat_map(|row| row[..3].iter())
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
    for pivot in 0..3 {
        let best = (pivot..3)
            .max_by(|&a, &b| matrix[a][pivot].abs().total_cmp(&matrix[b][pivot].abs()))?;
        if matrix[best][pivot].abs() <= scale * 1e-12 {
            return None;
        }
        matrix.swap(pivot, best);
        for row in pivot + 1..3 {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for col in pivot..4 {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
        }
    }
    let mut weights = [0.0f64; 3];
    for row in (0..3).rev() {
        let known: f64 = (row + 1..3)
            .map(|col| matrix[row][col] * weights[col])
            .sum();
        weights[row] = (matrix[row][3] - known) / matrix[row][row];
    }
    let [fma, lds, staged] = weights;
    if fma <= 0.0 {
        return None;
    }
    let mean = samples.iter().map(|(_, time)| time).sum::<f64>() / samples.len() as f64;
    let (mut residual, mut total) = (0.0, 0.0);
    for (terms, time) in samples {
        let f = features(terms);
        let modelled = fma * f[0] + lds * f[1] + staged * f[2];
        residual += (time - modelled).powi(2);
        total += (time - mean).powi(2);
    }
    Some(CostFit {
        lds_read: lds / fma,
        staged_value: staged / fma,
        seconds_per_fma: fma,
        r_squared: if total > 0.0 {
            1.0 - residual / total
        } else {
            1.0
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_target::calibration::CostRatios;

    /// A tiling's terms for tile `(rows, cols)` of `lanes` lanes over `tiles` tiles on a `device_lanes` device:
    /// the scalar-tile arithmetic of `RegisterTiling::cost_terms`.
    fn terms(rows: f64, cols: f64, lanes: f64, tiles: f64, device_lanes: f64) -> CostTerms {
        CostTerms {
            fma: rows * cols,
            lds_reads: rows + cols,
            staged_per_lane: (rows * 8.0 + cols * 8.0) / lanes,
            direct_per_lane: 0.0,
            rounding: 0.0,
            occupancy: (tiles * lanes / device_lanes).max(1.0),
        }
    }

    fn corpus() -> Vec<CostTerms> {
        let mut out = Vec::new();
        for (rows, cols) in [(2.0, 4.0), (4.0, 4.0), (4.0, 8.0), (8.0, 4.0), (8.0, 8.0)] {
            for (lanes, tiles) in [(64.0, 40.0), (64.0, 4000.0), (256.0, 900.0), (256.0, 90.0)] {
                out.push(terms(rows, cols, lanes, tiles, 10_240.0));
            }
        }
        out
    }

    /// Times generated from known ratios are recovered, with a perfect fit; the model's own unit is arbitrary
    /// (the time is scaled by 3.5e-9). Mutation: swap the LDS and staged columns in `features`; the recovered
    /// ratios swap and the assertion names the LDS read ratio.
    #[test]
    fn the_fit_recovers_the_ratios_the_times_were_made_with() {
        let truth = CostRatios {
            lds_read: 1.7,
            staged_value: 5.2,
            direct_value: 4.0,
        };
        let samples: Vec<_> = corpus()
            .into_iter()
            .map(|t| (t, t.modelled_time(truth) * 3.5e-9))
            .collect();
        let fit = fit_cost_ratios(&samples).expect("a determined fit");
        assert!((fit.lds_read - 1.7).abs() < 1e-9, "{fit:?}");
        assert!((fit.staged_value - 5.2).abs() < 1e-9, "{fit:?}");
        assert!((fit.seconds_per_fma - 3.5e-9).abs() < 1e-18, "{fit:?}");
        assert!(fit.r_squared > 1.0 - 1e-12, "{fit:?}");
    }

    /// Samples that cannot determine three ratios give no fit rather than a guess: too few, or one tile geometry
    /// whose terms are proportional. Mutation: drop the pivot check; the collinear row returns a fit of noise.
    #[test]
    fn an_underdetermined_corpus_gives_no_fit() {
        let one = terms(8.0, 8.0, 64.0, 4000.0, 10_240.0);
        let same = vec![(one, 1.0), (one, 1.0), (one, 1.0), (one, 1.0)];
        assert_eq!(fit_cost_ratios(&same), None, "collinear");
        let few: Vec<_> = corpus().into_iter().take(2).map(|t| (t, 1.0)).collect();
        assert_eq!(fit_cost_ratios(&few), None, "two samples");
    }
}
