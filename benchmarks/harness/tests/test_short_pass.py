import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import short_pass  # noqa: E402


def point(n, p50, lo, hi):
    return {"isl": 128, "iters": n, "ttft_ms": {"p50": p50, "min": lo, "max": hi}}


class Coverage(unittest.TestCase):
    def test_a_long_pass_is_not_a_short_pass(self):
        self.assertIsNone(short_pass.coverage(point(3, 18050, 18000, 18100)))

    def test_three_flat_iterations_cover_a_short_pass(self):
        covered, window, spread = short_pass.coverage(point(3, 622, 620, 624))
        self.assertTrue(covered)
        self.assertLess(window, short_pass.TIMED_WINDOW_S)

    def test_three_iterations_with_a_spike_do_not(self):
        covered, _, spread = short_pass.coverage(point(3, 370, 365, 690))
        self.assertFalse(covered)
        self.assertGreater(spread, short_pass.FLAT_SPREAD)

    def test_one_iteration_is_not_flat_evidence(self):
        self.assertFalse(short_pass.coverage(point(1, 400, 400, 400))[0])

    def test_a_timed_window_covers_a_spiky_pass(self):
        covered, window, _ = short_pass.coverage(point(58, 365, 360, 900))
        self.assertTrue(covered)
        self.assertGreaterEqual(window, short_pass.TIMED_WINDOW_S)

    def test_uncovered_lists_only_ok_rows_with_uncovered_points(self):
        runs = [("r1", {}, [{"status": "ok", "framework": "x", "curve": [point(3, 370, 365, 690)]},
                           {"status": "error"}, {"status": "ok", "curve": [point(3, 101, 100, 102)]}])]
        found = short_pass.uncovered(runs)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][0], "r1")


if __name__ == "__main__":
    unittest.main()
