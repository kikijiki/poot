"""`bench compare-committed` skips a committed cell whose every row is an error (nothing to judge); that a required
cell was never measured is the grid check's job."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

from test_benchmark_compare import BASELINE, repeats, write_run  # noqa: E402

MANIFEST = """\
[performance]
min_trials = 5

[[scenario]]
id = "curve"
isl = [128]

[[model]]
id = "m"
[model.frameworks.poot]
precision = "bf16"

[[model]]
id = "broken"
[model.frameworks.poot]
precision = "bf16"
"""

GRID_HEAD = """\
repeats = 5

[[machine]]
id = "box"
hardware = "fixture-gpu"

[[group]]
machine = "box"
scenario = "curve"
models = ["m"]
engines = [{ framework = "poot", backend = "ptx" }]
"""

BROKEN_GROUP = """
[[group]]
machine = "box"
scenario = "curve"
models = ["broken"]
engines = [{ framework = "poot", backend = "ptx" }]
"""


def broken_group(status=None):
    if status is None:
        return BROKEN_GROUP
    reason = '\nreason = "cannot load"' if status == "not-run" else ""
    return BROKEN_GROUP.replace('models = ["broken"]', f'status = "{status}"{reason}\nmodels = ["broken"]')


def run_gate(grid_tail):
    """`compare-committed` over a verified cell `m` and an error-only cell `broken`, with the given grid groups."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        results = root / "results"
        results.mkdir()
        good = repeats(BASELINE, scenario="curve")
        bad = repeats([0.0] * 5, scenario="curve", model="broken", status="error", reason="cannot load")
        for i, row in enumerate(good, 1):
            write_run(results, f"good-r{i}", [row])
        for i, row in enumerate(bad, 1):
            write_run(results, f"bad-r{i}", [row])
        (root / "manifest.toml").write_text(MANIFEST)
        (root / "grid.toml").write_text(GRID_HEAD + grid_tail)
        return subprocess.run(
            [sys.executable, str(HARNESS_DIR / "bench.py"), "compare-committed", "--results-dir", str(results)],
            capture_output=True, text=True, check=False,
        )


class ErrorOnlyCells(unittest.TestCase):
    def test_an_error_only_cell_is_skipped_whatever_the_grid_says(self):
        for tail in (broken_group("pending"), broken_group("not-run"), broken_group(), ""):
            with self.subTest(grid=tail.strip()[:40]):
                done = run_gate(tail)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertNotIn("broken", done.stdout)

    def test_a_verified_cell_is_still_judged(self):
        done = run_gate("")
        self.assertIn("within noise", done.stdout)


if __name__ == "__main__":
    unittest.main()
