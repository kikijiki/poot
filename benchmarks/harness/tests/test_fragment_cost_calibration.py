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


if __name__ == "__main__":
    unittest.main()
