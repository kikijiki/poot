import importlib.util
import statistics
import sys
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402


def llamacpp_runner():
    spec = importlib.util.spec_from_file_location("llamacpp_run", bench.BENCH_ROOT / "runners" / "llamacpp" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rate(ms, tokens=128):
    """The tokens/s of a repetition that took `ms` to process `tokens`."""
    return tokens / ms * 1000.0


class CurveIterations(unittest.TestCase):
    def test_the_warm_up_repetition_is_discarded(self):
        # Flash-Next on ROCm: the first repetition of a fresh process is about 10% slower than the rest.
        pp = {"samples_ts": [rate(406), rate(364), rate(369)], "avg_ts": 0}
        tg = {"samples_ts": [30.0, 31.0, 31.0], "avg_ts": 0}
        iters = llamacpp_runner().curve_iterations(128, 128, pp, tg, warmup=1)
        self.assertEqual(len(iters), 2)
        self.assertAlmostEqual(iters[0]["ttft_ms"], 364, places=3)
        self.assertAlmostEqual(iters[0]["itl_ms"][0], 1000 / 31.0, places=6)
        self.assertEqual(len(iters[0]["itl_ms"]), 127)

    def test_one_slow_repetition_does_not_move_the_median_of_the_iterations(self):
        # llama-bench's avg_ts is a mean: 400, 363 and a 690 ms spike average 484 ms. The runner reports the
        # repetitions, so the contract's median is the typical one.
        pp = {"samples_ts": [rate(x) for x in (404, 363, 366, 690, 365)], "avg_ts": 0}
        tg = {"samples_ts": [31.0] * 5, "avg_ts": 0}
        iters = llamacpp_runner().curve_iterations(128, 128, pp, tg, warmup=1)
        self.assertAlmostEqual(statistics.median(i["ttft_ms"] for i in iters), 365.5, places=3)

    def test_a_short_prefill_pass_is_repeated_for_a_timed_window_and_a_long_one_is_not(self):
        run = llamacpp_runner()
        self.assertEqual(run.prefill_repetitions(0.4, warmup=1, iters=3), 1 + 58)  # 20 s x 1.15 / 0.4 s
        self.assertEqual(run.prefill_repetitions(0.001, warmup=1, iters=3), 1 + run.MAX_REPETITIONS)
        self.assertEqual(run.prefill_repetitions(2.0, warmup=1, iters=3), 4)  # long passes keep the suite's count
        self.assertEqual(run.prefill_repetitions(18.0, warmup=1, iters=3), 4)

    def test_every_timed_prefill_repetition_is_an_iteration_whatever_the_decode_count(self):
        pp = {"samples_ts": [rate(406)] + [rate(365)] * 50 + [rate(900)] * 8}  # one episode of 8 slow passes
        tg = {"samples_ts": [30.0, 31.0, 31.0, 32.0]}
        iters = llamacpp_runner().curve_iterations(128, 128, pp, tg, warmup=1)
        self.assertEqual(len(iters), 58)
        self.assertAlmostEqual(statistics.median(i["ttft_ms"] for i in iters), 365, places=3)
        self.assertAlmostEqual(iters[0]["itl_ms"][0], 1000 / 31.0, places=6)  # median of the timed decode rates

    def test_a_run_with_no_repetition_after_the_warm_up_is_refused(self):
        with self.assertRaises(ValueError):
            llamacpp_runner().timed_samples({"samples_ts": [30.0]}, warmup=1)

    def test_the_runner_command_passes_the_suite_warm_up(self):
        runners = bench.load_toml(bench.BENCH_ROOT / "runners.toml")
        self.assertIn("{warmup}", runners["llamacpp"]["cmd"])


if __name__ == "__main__":
    unittest.main()
