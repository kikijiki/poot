"""A stored run's memory is split into the load and inference by cutting its 1 Hz series at the end of the load plus the warmup."""

import sys
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import memory_phases as phases  # noqa: E402

GIB = 2**30


def series(rss, vram=None):
    """One sample a second from t=1: `[t, gpu%, cpu%, vram, rss, power]` with the memory given in GiB."""
    vram = vram or [0] * len(rss)
    return [[t, 0, 0, int(v * GIB), int(r * GIB), 0.0] for t, (r, v) in enumerate(zip(rss, vram), 1)]


def row(rss, vram=None, open_ms=None, **fields):
    """A stored row; `open_ms` is the engine's recorded open time (`preparation.open_ms`) when given."""
    preparation = {"preparation": {"open_ms": open_ms}} if open_ms else {}
    return {"util_series": series(rss, vram), **preparation, **fields}


# The Flash-Next shape: the load reads 25 GiB through the resident set in the first seconds, inference holds 5.7.
LOAD_THEN_INFERENCE = [4.7, 26.0, 26.0, 5.1, 5.4, 5.7, 5.7, 5.7, 5.7, 1.1]


class MemoryPhasesTest(unittest.TestCase):
    def test_the_recorded_open_time_cuts_the_series_into_load_and_inference(self):
        got = phases.memory_phases(row(LOAD_THEN_INFERENCE, open_ms=3000), 0)
        self.assertEqual(got["load"]["rss_bytes"], int(26.0 * GIB))
        self.assertEqual(got["inference"]["rss_bytes"], int(5.7 * GIB))

    def test_a_later_open_time_moves_the_cut_and_the_inference_peak_with_it(self):
        # open_ms 1000 puts the 26 GiB samples (t=2, 3) into inference: the cut decides the number.
        got = phases.memory_phases(row(LOAD_THEN_INFERENCE, open_ms=1000), 0)
        self.assertEqual(got["inference"]["rss_bytes"], int(26.0 * GIB))

    def test_without_an_open_time_the_last_load_ramp_of_the_series_is_the_boundary(self):
        got = phases.memory_phases(row(LOAD_THEN_INFERENCE), 0)
        self.assertEqual(got["load"]["rss_bytes"], int(26.0 * GIB))
        self.assertEqual(got["inference"]["rss_bytes"], int(5.7 * GIB))

    def test_a_server_restarted_per_curve_point_is_bounded_by_its_last_load(self):
        # Three loads of 40 GiB, each followed by inference at 3 GiB; the last inference grows to 9.
        rss = [40, 3, 3, 3, 40, 3, 3, 3, 40, 3, 6, 9, 9]
        got = phases.memory_phases(row(rss), 0)
        self.assertEqual(got["inference"]["rss_bytes"], 9 * GIB)
        self.assertEqual(got["load"]["rss_bytes"], 40 * GIB)

    def test_the_load_peak_of_a_multi_load_series_is_the_last_loads_only(self):
        rss = [40, 3, 3, 3, 40, 3, 3, 3, 40, 3, 6, 9, 9]
        vram = [10, 20, 20, 20, 12, 20, 20, 20, 11, 30, 30, 30, 30]
        got = phases.memory_phases(row(rss, vram), 0)
        self.assertEqual(got["load"]["vram_bytes"], 11 * GIB)

    def test_a_series_with_no_load_ramp_and_no_open_time_has_no_boundary(self):
        self.assertIsNone(phases.memory_phases(row([2.0, 2.1, 2.1, 2.2, 2.2, 2.2]), 0))

    def test_a_fall_below_the_floor_is_not_a_load(self):
        small = [0.4, 0.1, 0.1, 0.1, 0.1, 0.1]  # 0.3 GiB: over a quarter of the peak, under the absolute floor
        self.assertIsNone(phases.memory_phases(row(small), 0))

    def test_the_process_exiting_at_the_last_sample_is_not_a_finished_load(self):
        # Inference holds 19 GiB and the exit drops it to 0.3: only the fall after the 40 GiB load bounds the window.
        rss = [40, 3, 3, 3, 3, 19, 19, 19, 19, 0.3]
        got = phases.memory_phases(row(rss), 0)
        self.assertEqual(got["inference"]["rss_bytes"], 19 * GIB)

    def test_a_cut_that_leaves_too_few_samples_has_no_boundary(self):
        self.assertIsNone(phases.memory_phases(row([1, 30, 5, 5], open_ms=2000), 0))

    def test_the_warmup_after_the_load_belongs_to_neither_the_load_nor_inference(self):
        # The warmup request touches 12 GiB at t=4 for one second; inference after it holds 5.
        rss = [4.7, 26.0, 26.0, 12.0, 5.4, 5.0, 5.0, 5.0, 5.0]
        curve = [{"osl": 100, "ttft_ms": {"p50": 500.0}, "tpot_ms": {"p50": 5.0}}]
        warmed = phases.memory_phases(row(rss, open_ms=3000, curve=curve), 1)
        self.assertEqual(warmed["inference"]["rss_bytes"], int(5.4 * GIB))
        unwarmed = phases.memory_phases(row(rss, open_ms=3000, curve=curve), 0)
        self.assertEqual(unwarmed["inference"]["rss_bytes"], 12 * GIB)

    def test_the_warmup_lasts_the_requests_of_the_first_curve_point(self):
        curve = [{"osl": 128, "ttft_ms": {"p50": 800.0}, "tpot_ms": {"p50": 25.0}}, {"osl": 128, "ttft_ms": {"p50": 9000.0}, "tpot_ms": {"p50": 40.0}}]
        self.assertAlmostEqual(phases.warmup_seconds({"curve": curve}, 2), 2 * (0.8 + 128 * 0.025))
        self.assertEqual(phases.warmup_seconds({}, 2), 0.0)

    def test_gpu_memory_is_the_runs_own_not_the_devices_baseline(self):
        got = phases.memory_phases(row(LOAD_THEN_INFERENCE, [2, 50, 50, 90, 94, 94, 94, 94, 94, 30], open_ms=3000, vram_baseline_bytes=2 * GIB), 0)
        self.assertEqual(got["inference"]["vram_bytes"], 92 * GIB)
        self.assertEqual(got["load"]["vram_bytes"], 48 * GIB)

    def test_a_row_with_no_series_has_no_boundary(self):
        self.assertIsNone(phases.memory_phases({}, 0))
        self.assertIsNone(phases.memory_phases({"util_series": series([1.0])}, 0))


class StoredResultsTest(unittest.TestCase):
    """The split on the committed series the card names, not on a fixture."""

    def stored(self, run_dir_glob, framework):
        import json

        from bench import BENCH_ROOT

        for path in sorted((BENCH_ROOT / "results").glob(f"{run_dir_glob}/results.jsonl")):
            for line in path.read_text().splitlines():
                found = json.loads(line)
                if found["framework"] == framework:
                    return found
        self.fail(f"no stored {framework} run under {run_dir_glob}")

    def test_poot_ptx_flash_next_holds_6_gib_of_host_memory_at_inference_after_a_26_gib_load(self):
        got = phases.memory_phases(self.stored("2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-fla*-r1", "poot"), 1)
        self.assertLess(got["inference"]["rss_bytes"], 6.5 * GIB)
        self.assertGreater(got["load"]["rss_bytes"], 25 * GIB)
        self.assertGreater(got["inference"]["vram_bytes"], 90 * GIB)

    def test_llamacpp_flash_next_is_bounded_by_its_last_load(self):
        got = phases.memory_phases(self.stored("2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-fla*-r1", "llamacpp"), 1)
        self.assertLess(got["inference"]["rss_bytes"], 10 * GIB)
        self.assertGreater(got["load"]["rss_bytes"], 40 * GIB)


if __name__ == "__main__":
    unittest.main()
