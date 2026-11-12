"""POOT-1547: the grouped-fragment round and the grouping-scan step in `crates/poot-target/src/calibration.rs` are the
fit's result over the committed corpus (`benchmarks/calibration/rocm-gfx1151.tsv`), so a hand edit of either figure, or
rows without a refit, goes red."""

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


if __name__ == "__main__":
    unittest.main()
