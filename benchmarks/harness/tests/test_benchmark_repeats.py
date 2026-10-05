import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import observer  # noqa: E402
from runner_fixtures import run_fixture_runner, valid_single  # noqa: E402

MANIFEST = """
[suite]
default_gen_tokens = 4
warmup_iters = 1
timed_iters = 1

[[scenario]]
id = "single"
gen_tokens = 4

[[model]]
id = "fixture"
match_precision = "bf16"
frameworks.poot = {}
"""


def write_stub_workspace(directory):
    """A manifest with one poot cell and a runner that prints a valid result.

    The runner's throughput cycles through five values, so repeats differ the way real ones do.
    """
    directory = Path(directory)
    counter = directory / "runs"
    payload = json.dumps(valid_single() | {"decode_tok_s": "@TOK_S@"}).replace('"@TOK_S@"', "%s")
    binary = directory / "stub-runner"
    binary.write_text(
        f"#!/bin/sh\nn=$(cat '{counter}' 2>/dev/null || echo 0); n=$((n + 1)); echo $n > '{counter}'\n"
        f"printf '{payload}\\n' \"$((500 + n % 5))\"\n",
        encoding="ascii",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    manifest = directory / "manifest.toml"
    manifest.write_text(MANIFEST, encoding="ascii")
    runners = directory / "runners.toml"
    runners.write_text(f'[poot]\nbin = "{binary}"\ncmd = ["{{bin}}"]\n', encoding="ascii")
    return manifest, runners


def bench_run(directory, results, *extra):
    """`bench run` over the stub workspace, in process (the observer is the fixture runner, not a sampler).

    Returns the exit status `bench` ends with.
    """
    manifest, runners = write_stub_workspace(directory)
    argv = ["bench", "--manifest", str(manifest), "--runners", str(runners), "run",
            "--framework", "poot", "--models-dir", str(directory), "--results-dir", str(results), *extra]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(observer, "run_observed", side_effect=run_fixture_runner),
        mock.patch("sys.stdout"),
        mock.patch("sys.stderr"),
    ):
        try:
            bench.main()
        except SystemExit as done:
            return done.code
    raise AssertionError("bench.main returned instead of exiting")


def compare_cli(baseline_runs, candidate_runs):
    return subprocess.run(
        [sys.executable, str(HARNESS_DIR / "bench.py"), "compare",
         "--baseline", *baseline_runs, "--candidate", *candidate_runs],
        capture_output=True,
        text=True,
        check=False,
    )


class RepeatsTests(unittest.TestCase):
    def test_repeats_write_one_valid_run_directory_each_and_compare_counts_them_as_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            self.assertEqual(bench_run(tmp, results, "--run-id", "base", "--repeats", "3"), 0)

            runs = sorted(path.name for path in results.iterdir())
            self.assertEqual(runs, ["base-r1", "base-r2", "base-r3"])
            for run in runs:
                self.assertTrue((results / run / "env.json").is_file(), run)
                self.assertTrue((results / run / "report.md").is_file(), run)
            rows = [row for run in runs for row in bench._load_rows(results / run)]
            (cell,) = bench.compare_cells(rows, rows)

        self.assertEqual((cell["n_a"], cell["n_b"]), (3, 3))
        self.assertEqual(cell["key"][:3], ("fixture", "single", "poot"))

    def test_five_repeats_of_each_side_give_compare_a_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            for side in ("baseline", "candidate"):
                self.assertEqual(bench_run(tmp, results, "--run-id", side, "--repeats", str(bench.NOISE_REPEATS)), 0)
            runs = range(1, bench.NOISE_REPEATS + 1)
            baseline = [str(results / f"baseline-r{n}") for n in runs]
            candidate = [str(results / f"candidate-r{n}") for n in runs]
            done = compare_cli(baseline, candidate)

        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(bench.WITHIN_NOISE, done.stdout)

    def test_one_repeat_is_the_run_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            self.assertEqual(bench_run(tmp, results, "--run-id", "solo"), 0)

            self.assertEqual([path.name for path in results.iterdir()], ["solo"])

    def test_fewer_than_one_repeat_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            self.assertIn("--repeats must be at least 1", str(bench_run(tmp, results, "--repeats", "0")))
            self.assertFalse(results.exists())


if __name__ == "__main__":
    unittest.main()
