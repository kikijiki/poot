"""POOT-1547: the grouped-fragment round and the grouping-scan step in `crates/poot-target/src/calibration.rs` are the
fit's result over the committed corpus (`benchmarks/calibration/rocm-gfx1151.tsv`), so a hand edit of either figure, or
rows without a refit, goes red.

POOT-1565: the same figures, scored on a packed format the corpus does not hold (Q4_K, `rocm-gfx1151-heldout-q4k.tsv`,
never fitted), choose every cell within the recorded bound of its fastest measured schedule."""

import importlib.util
import unittest
from pathlib import Path

CALIBRATION = Path(__file__).resolve().parents[2] / "calibration"


def _load():
    spec = importlib.util.spec_from_file_location("fit_fragment_cost", CALIBRATION / "fit_fragment_cost.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = _load()


class FragmentCostCalibration(unittest.TestCase):
    def setUp(self):
        self.rows = fit.load(CALIBRATION / "rocm-gfx1151.tsv")

    def test_the_corpus_covers_both_packed_cases_and_the_dense_cells(self):
        cells = {row["cell"] for row in self.rows}
        self.assertEqual(len(self.rows), 611)
        self.assertEqual(
            {name.split(":")[0] for name in cells}, {"packed-iq2xs-up", "packed-iq4nl-down", "dense"}
        )
        self.assertEqual(len(cells), 12)

    def test_the_committed_figures_are_the_fits_result(self):
        result = fit.fit(self.rows)
        have = fit.constants()
        self.assertEqual(have[0], round(result["fragment_round"], -1))
        self.assertEqual(have[1], round(result["group_scan"], 1))

    def test_the_figures_serve_the_corpus_better_than_the_scalar_model(self):
        result = fit.fit(self.rows)
        _, mean_without = fit.shipped(self.rows)
        self.assertLess(result["mean"], mean_without)


class HeldOutQ4K(unittest.TestCase):
    """Worst regret 1.260 and mean 1.1802 were measured on the quiet box of POOT-1565; the scalar model of before
    POOT-1547 scores 1.218 and 1.1247 on the same cells (the fragment-round and grouping terms win the fitted formats
    and cost Q4_K 5 percent of mean regret: the open finding of the card)."""

    WORST_REGRET = 1.27
    MEAN_REGRET = 1.19

    def setUp(self):
        self.corpus = fit.load(CALIBRATION / "rocm-gfx1151.tsv")
        self.rows = fit.load(CALIBRATION / "rocm-gfx1151-heldout-q4k.tsv")

    def test_the_held_out_corpus_is_cells_the_fit_never_saw(self):
        cells = {row["cell"] for row in self.rows}
        self.assertEqual(len(self.rows), 264)
        self.assertEqual({name.split(":")[0] for name in cells}, {"packed-q4k-up", "packed-q4k-down"})
        self.assertEqual(len(cells), 6)
        self.assertFalse(cells & {row["cell"] for row in self.corpus})

    def test_the_shipped_figures_serve_a_format_they_were_not_fitted_on(self):
        worst, mean, _ = fit.heldout(self.corpus, self.rows)["shipped"]
        self.assertLessEqual(worst, self.WORST_REGRET)
        self.assertLessEqual(mean, self.MEAN_REGRET)


class Sm89GemvFigures(unittest.TestCase):
    """POOT-1602: the RTX 4090's direct value and Gemv dot cost (`SM_89_COST_RATIOS`) are the fit's result over both
    corpora of that part, and they choose the two packed cells at 128 tokens within 1.10 of their fastest schedule (the
    IQ2_XS up projection's Gemv, the IQ4_NL down projection's narrowed tile), which no direct value does without the
    dot term. Mutation: set `gemv_dot` to 0 in the second row's call; the IQ4_NL cell
    chooses the Gemv (`'gemv' != 'fragment'`, regret 1.457)."""

    def setUp(self):
        self.fragment = fit.load(CALIBRATION / "ptx-sm89.tsv")
        self.indexed = fit.load_indexed(CALIBRATION / "ptx-sm89-indexed.tsv")

    def test_the_committed_figures_are_the_fits_result(self):
        self.assertEqual(len(self.fragment), 551)
        self.assertEqual(len(self.indexed), 88)
        result = fit.fit_sm89(self.fragment, self.indexed)
        self.assertEqual(fit.sm89_constants(), (round(result["direct_value"], 1), float(round(result["gemv_dot"]))))

    def test_both_packed_cells_at_128_tokens_choose_within_the_band(self):
        direct_value, gemv_dot = fit.sm89_constants()
        with fit.ratios(fit.SM89_PRIOR_LDS_READ, fit.SM89_PRIOR_STAGED_VALUE, direct_value, gemv_dot):
            cells = fit.regrets_of(self.fragment)
        self.assertEqual(cells["packed-iq2xs-up:128"][1]["kind"], "gemv")
        self.assertLessEqual(cells["packed-iq2xs-up:128"][0], 1.10)
        self.assertEqual(cells["packed-iq4nl-down:128"][1]["kind"], "fragment")
        self.assertLessEqual(cells["packed-iq4nl-down:128"][0], 1.10)
        mean = sum(v[0] for v in cells.values()) / len(cells)
        self.assertLessEqual(mean, 1.099)

    def test_no_direct_value_serves_both_cells_without_the_dot_term(self):
        for direct_value in fit.DIRECT_GRID:
            with fit.ratios(fit.SM89_PRIOR_LDS_READ, fit.SM89_PRIOR_STAGED_VALUE, direct_value, 0.0):
                cells = fit.regrets_of(self.fragment)
            self.assertGreater(
                max(cells["packed-iq2xs-up:128"][0], cells["packed-iq4nl-down:128"][0]), 1.10, direct_value
            )


if __name__ == "__main__":
    unittest.main()
