"""The decode-curve progress line's ETA: unknown after one point, and close to the real remaining time afterwards.

The cost of a context length is a fixed part (model load, the decode window) plus a part that grows with the context,
so an ETA that scales the last point linearly with the ISL is wildly wrong (it said 25.8 min for a 6.5 min curve)."""

import io
import re
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402

# The curve of a 0.6B model on a pod: cumulative minutes at which each ISL line appeared (measured, 2026-10-22).
ISLS = [128, 512, 2048, 8192]
CUMULATIVE_MIN = [0.3, 0.6, 1.2, 6.5]


def run_curve():
    clock = {"now": 1000.0}
    lines = []
    with mock.patch.object(bench.time, "time", lambda: clock["now"]):
        cb = bench._curve_progress("poot", ISLS)
        clock["now"] = 1000.0
        for isl, cum in zip(ISLS, CUMULATIVE_MIN):
            clock["now"] = 1000.0 + cum * 60
            buf = io.StringIO()
            with redirect_stderr(buf):
                cb(f"isl={isl} done")
            lines.append(buf.getvalue().strip())
    return lines


def eta_minutes(line):
    m = re.search(r"eta ~([0-9.]+)m", line)
    return float(m.group(1)) if m else None


class CurveProgressTest(unittest.TestCase):
    def test_fewer_than_three_points_cannot_estimate(self):
        lines = run_curve()
        self.assertIsNone(eta_minutes(lines[0]), lines[0])
        self.assertIsNone(eta_minutes(lines[1]), lines[1])

    def test_the_eta_after_three_points_is_close_to_the_real_remaining_time(self):
        lines = run_curve()
        real_remaining = CUMULATIVE_MIN[-1] - CUMULATIVE_MIN[2]   # 5.3 min
        eta = eta_minutes(lines[2])
        self.assertIsNotNone(eta, lines[2])
        self.assertLess(abs(eta - real_remaining) / real_remaining, 0.4, f"eta {eta} vs real {real_remaining}")

    def test_the_last_point_has_no_eta(self):
        self.assertIsNone(eta_minutes(run_curve()[3]))

    def test_a_longer_context_never_costs_less_than_the_last_point(self):
        pts = [(128, 18.0), (512, 18.0)]
        self.assertGreaterEqual(bench._predict_increment(pts, 2048), 18.0)


if __name__ == "__main__":
    unittest.main()
