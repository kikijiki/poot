//! The roofline of a decode run: bytes the plan reads per token, milliseconds per token, and the
//! fraction of a calibrated memory-bandwidth ceiling they amount to.
//!
//! Decode is memory-bound, so `bytes_per_token / ms_per_token` against the device's streaming ceiling
//! says in one number whether a run is at the floor. Both inputs are engine quantities: the bytes come
//! from the compiled plan (`poot_graph_plan::ReadTraffic`), the time from the executor's own step clock,
//! and the ceiling is a pinned value in the checked-in table `data/ceilings.tsv` with the receipt of the
//! measurement that produced it. A caller never supplies a bandwidth of its own.

use std::sync::OnceLock;

/// One decimal gigabyte, the unit ceilings and fits are stated in.
pub const GB: f64 = 1e9;

/// A device's calibrated memory-bandwidth ceiling: a row of `data/ceilings.tsv`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ceiling {
    /// The executor target label the row applies to ([`Ceiling::pinned`]'s key).
    pub target: &'static str,
    /// Streaming read bandwidth in decimal GB/s.
    pub gb_per_s: f64,
    /// The hardware the number was measured on.
    pub device: &'static str,
    /// The checked-in receipt of the measurement, a path relative to the repository root.
    pub receipt: &'static str,
}

const TABLE: &str = include_str!("../data/ceilings.tsv");

/// One tab-separated row of [`TABLE`]: `target gb_per_s device receipt`.
fn parse_row(line: &'static str) -> Ceiling {
    let mut cols = line.split('\t');
    let mut col = |name: &str| {
        cols.next()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| panic!("ceilings.tsv row {line:?} has no {name}"))
    };
    let target = col("target");
    let gb_per_s = col("gb_per_s")
        .parse::<f64>()
        .ok()
        .filter(|gb| gb.is_finite() && *gb > 0.0)
        .unwrap_or_else(|| panic!("ceilings.tsv row {line:?} has a bad gb_per_s"));
    Ceiling {
        target,
        gb_per_s,
        device: col("device"),
        receipt: col("receipt"),
    }
}

impl Ceiling {
    /// Every row of the checked-in table, parsed once. A malformed row is a bug in the checked-in file
    /// (this crate's own tests read the whole table), so it panics rather than reading as an absent
    /// device.
    pub fn table() -> &'static [Ceiling] {
        static ROWS: OnceLock<Vec<Ceiling>> = OnceLock::new();
        ROWS.get_or_init(|| {
            TABLE
                .lines()
                .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
                .map(parse_row)
                .collect()
        })
    }

    /// The pinned ceiling of executor target `target`, or `None` for a target nobody calibrated: a
    /// run on it reports no fraction rather than one against a borrowed number.
    pub fn pinned(target: &str) -> Option<Ceiling> {
        Self::table()
            .iter()
            .copied()
            .find(|row| row.target == target)
    }
}

/// One decode run against its ceiling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roofline {
    pub bytes_per_token: f64,
    pub ms_per_token: f64,
    /// `bytes_per_token / ms_per_token` in decimal GB/s.
    pub achieved_gb_per_s: f64,
    /// `achieved_gb_per_s / ceiling.gb_per_s`.
    pub fraction_of_ceiling: f64,
    pub ceiling: Ceiling,
}

impl Roofline {
    /// The roofline of a run that moved `bytes_per_token` bytes in `ms_per_token`, or `None` when
    /// either is not a positive finite number (no tokens, no measured time).
    pub fn of(bytes_per_token: f64, ms_per_token: f64, ceiling: Ceiling) -> Option<Self> {
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !positive(bytes_per_token) || !positive(ms_per_token) {
            return None;
        }
        let achieved_gb_per_s = bytes_per_token / GB / (ms_per_token / 1e3);
        Some(Self {
            bytes_per_token,
            ms_per_token,
            achieved_gb_per_s,
            fraction_of_ceiling: achieved_gb_per_s / ceiling.gb_per_s,
            ceiling,
        })
    }

    /// A one-line rendering for a CLI's status output.
    pub fn render(&self) -> String {
        format!(
            "roofline: {:.3} ms/token, {:.1} MB/token read, {:.1} GB/s = {:.3} of the {:.1} GB/s ceiling ({}; {})",
            self.ms_per_token,
            self.bytes_per_token / 1e6,
            self.achieved_gb_per_s,
            self.fraction_of_ceiling,
            self.ceiling.gb_per_s,
            self.ceiling.device,
            self.ceiling.receipt,
        )
    }
}

/// The least-squares line `ms_per_token = a * gb_per_token + b` over a set of runs, with its R^2.
/// `a` is the marginal milliseconds per gigabyte (the reciprocal of the effective streaming rate), `b`
/// the per-token cost that does not scale with bytes (dispatch, sampling, the fixed term).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    pub ms_per_gb: f64,
    pub fixed_ms: f64,
    pub r2: f64,
}

impl Fit {
    /// Fit `(gb_per_token, ms_per_token)` points. `None` for fewer than two points, a degenerate
    /// (all-equal) byte axis, or a non-finite point.
    pub fn over(points: &[(f64, f64)]) -> Option<Self> {
        if points.len() < 2 || points.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
            return None;
        }
        let n = points.len() as f64;
        let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
        let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
        let sxx: f64 = points.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
        let sxy: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
        let syy: f64 = points.iter().map(|p| (p.1 - mean_y).powi(2)).sum();
        if sxx == 0.0 {
            return None;
        }
        let ms_per_gb = sxy / sxx;
        let fixed_ms = mean_y - ms_per_gb * mean_x;
        let r2 = if syy == 0.0 {
            1.0
        } else {
            sxy * sxy / (sxx * syy)
        };
        Some(Self {
            ms_per_gb,
            fixed_ms,
            r2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every checked-in row parses, names a receipt, and a target is calibrated once. Mutation: add a
    /// second row for one target, or blank a receipt column.
    #[test]
    fn the_checked_in_table_parses_with_one_receipted_row_per_target() {
        let table = Ceiling::table();
        assert!(!table.is_empty(), "no calibrated ceiling is pinned");
        for (index, row) in table.iter().enumerate() {
            assert!(row.receipt.ends_with(".json"), "{row:?} names no receipt");
            assert!(
                table[index + 1..]
                    .iter()
                    .all(|other| other.target != row.target),
                "target {} is calibrated twice",
                row.target
            );
        }
    }

    /// An uncalibrated target has no ceiling, so a run on it reports no fraction. Mutation: fall back
    /// to the first row.
    #[test]
    fn an_uncalibrated_target_has_no_ceiling() {
        assert_eq!(Ceiling::pinned("no-such-target"), None);
    }

    /// 100 MB read in 1 ms is 100 GB/s: 0.5 of a 200 GB/s ceiling. Mutation: divide by `GB / 1e3`.
    #[test]
    fn the_fraction_is_achieved_bandwidth_over_the_ceiling() {
        let ceiling = Ceiling {
            target: "t",
            gb_per_s: 200.0,
            device: "d",
            receipt: "r.json",
        };
        let roofline = Roofline::of(100e6, 1.0, ceiling).unwrap();
        assert!((roofline.achieved_gb_per_s - 100.0).abs() < 1e-9);
        assert!((roofline.fraction_of_ceiling - 0.5).abs() < 1e-9);
        assert!(Roofline::of(0.0, 1.0, ceiling).is_none());
        assert!(Roofline::of(1.0, 0.0, ceiling).is_none());
    }

    /// Points on `y = 3x + 2` recover a=3, b=2, R^2=1; a perturbed point lowers R^2. Mutation: use the
    /// residual sum without dividing by `syy`.
    #[test]
    fn the_fit_recovers_the_line_and_penalizes_scatter() {
        let exact = Fit::over(&[(1.0, 5.0), (2.0, 8.0), (4.0, 14.0)]).unwrap();
        assert!((exact.ms_per_gb - 3.0).abs() < 1e-9);
        assert!((exact.fixed_ms - 2.0).abs() < 1e-9);
        assert!((exact.r2 - 1.0).abs() < 1e-9);
        let noisy = Fit::over(&[(1.0, 5.0), (2.0, 9.5), (4.0, 14.0)]).unwrap();
        assert!(noisy.r2 < 0.99, "scatter must lower R^2, got {}", noisy.r2);
        assert!(Fit::over(&[(1.0, 5.0)]).is_none());
        assert!(Fit::over(&[(1.0, 5.0), (1.0, 6.0)]).is_none());
    }
}
