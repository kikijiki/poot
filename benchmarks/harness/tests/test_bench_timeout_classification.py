from pathlib import Path
import sys
import unittest
from unittest import mock


HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import observer  # noqa: E402


def observation(returncode, timed_out, stderr, stdout=""):
    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "timed_out": timed_out,
        "peak_vram_bytes": None,
        "vram_baseline_bytes": None,
        "peak_rss_bytes": None,
        "gpu_util_mean_pct": None,
        "gpu_util_peak_pct": None,
        "cpu_util_mean_pct": None,
        "cpu_util_peak_pct": None,
        "peak_power_w": None,
        "energy_j": None,
        "energy_wh": None,
        "util_series": None,
        "sampler_interval_ms": 50.0,
        "samples": 1,
    }


class BenchTimeoutClassificationTests(unittest.TestCase):
    def run_failure(self, result):
        with mock.patch.object(observer, "run_observed", return_value=result):
            return bench.run_cell(
                model={"id": "fixture"},
                scen={"id": "single"},
                fw="poot",
                fw_cfg={"support": True},
                runners={"poot": {"bin": sys.executable, "cmd": ["{bin}"]}},
                models_dir=".",
                suite={},
                interval_ms=50,
            )

    def test_timeout_and_ordinary_failure_have_distinct_reasons(self):
        timeout = self.run_failure(observation(-9, True, "owned group timed out"))
        failure = self.run_failure(observation(7, False, "ordinary runner failure"))

        self.assertEqual(timeout["status"], "error")
        self.assertTrue(timeout["reason"].startswith("timeout (rc=-9):"))
        self.assertEqual(failure["status"], "error")
        self.assertTrue(failure["reason"].startswith("rc=7:"))
        self.assertNotIn("timeout", failure["reason"])

    def test_timeout_is_error_with_zero_status_and_valid_timing_json(self):
        result = self.run_failure(
            observation(
                0,
                True,
                "timeout after runner emitted output",
                stdout='{"framework":"fixture","ttft_ms":1.0,"decode_tok_s":2.0}\n',
            )
        )

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["reason"].startswith("timeout (rc=0):"))


if __name__ == "__main__":
    unittest.main()
