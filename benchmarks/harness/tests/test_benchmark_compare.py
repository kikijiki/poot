import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402

DEVICE = "fixture-gpu"


def result_row(tok_s, *, backend="ptx", device=DEVICE, binary="bin-1", status="ok", model="m", **extra):
    row = {
        "framework": "poot",
        "model": model,
        "precision": "bf16",
        "scenario": "single",
        "status": status,
        "backend": backend,
        "device": device,
        "binary_sha256": binary,
        "build_sha": binary.ljust(40, "0"),
    }
    if status == "ok":
        row["decode_tok_s"] = tok_s
        row["peak_vram_bytes"] = 2**30
    else:
        row["reason"] = "runner crashed"
    return {**row, **extra}


def repeats(samples, **kwargs):
    """One row per repeated run of a cell."""
    return [result_row(tok_s, **kwargs) for tok_s in samples]


# Five repeats with median 100 and range 4: the band is 96..104 and a change needs to clear 2 * 4 = 8.
BASELINE = [98.0, 99.0, 100.0, 101.0, 102.0]


def verdicts(rows_a, rows_b):
    return {cell["key"]: cell["verdict"] for cell in bench.compare_cells(rows_a, rows_b)}


def write_run(directory, name, rows):
    run = Path(directory) / name
    run.mkdir()
    (run / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(run)


def compare_cli(baseline_runs, candidate_runs):
    return subprocess.run(
        [
            sys.executable,
            str(HARNESS_DIR / "bench.py"),
            "compare",
            "--baseline",
            *baseline_runs,
            "--candidate",
            *candidate_runs,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


class CompareCellKeyTests(unittest.TestCase):
    def test_backends_of_the_same_model_are_separate_cells(self):
        ptx = [100.0, 101.0, 99.0, 100.5, 99.5]
        rocm = [300.0, 301.0, 299.0, 300.5, 299.5]
        baseline = repeats(ptx, backend="ptx") + repeats(rocm, backend="rocm")
        # Only the rocm cell regresses; a merged cell could not say which backend it was.
        candidate = repeats(ptx, backend="ptx") + repeats([x * 0.8 for x in rocm], backend="rocm")

        cells = bench.compare_cells(baseline, candidate)

        self.assertEqual(len(cells), 2)
        by_backend = {cell["key"][3]: cell for cell in cells}
        self.assertEqual(sorted(by_backend), ["ptx", "rocm"])
        self.assertEqual(by_backend["ptx"]["verdict"], bench.WITHIN_NOISE)
        self.assertEqual(by_backend["ptx"]["median_a"], 100.0)
        self.assertEqual(by_backend["rocm"]["verdict"], bench.REGRESSION)
        self.assertEqual(by_backend["rocm"]["median_a"], 300.0)
        self.assertEqual((by_backend["ptx"]["n_a"], by_backend["rocm"]["n_a"]), (5, 5))

    def test_devices_of_the_same_model_and_backend_are_separate_cells(self):
        baseline = repeats(BASELINE, device="gpu-x") + repeats([x * 2 for x in BASELINE], device="gpu-y")
        candidate = repeats(BASELINE, device="gpu-x") + repeats([x * 2 for x in BASELINE], device="gpu-y")

        cells = bench.compare_cells(baseline, candidate)

        self.assertEqual([cell["key"][4] for cell in cells], ["gpu-x", "gpu-y"])
        self.assertEqual([cell["median_a"] for cell in cells], [100.0, 200.0])

    def test_a_backend_or_device_the_row_does_not_record_is_its_own_cell(self):
        legacy = repeats(BASELINE, backend=None, device=None)
        legacy = [{k: v for k, v in row.items() if k not in ("backend", "device")} for row in legacy]

        cells = bench.compare_cells(legacy + repeats(BASELINE), legacy + repeats(BASELINE))

        self.assertEqual({cell["key"][3:] for cell in cells}, {(None, None), ("ptx", DEVICE)})

    def test_cell_present_on_one_side_only_is_reported_not_dropped(self):
        got = verdicts(
            repeats(BASELINE, model="only-a") + repeats(BASELINE),
            repeats(BASELINE) + repeats(BASELINE, model="only-b"),
        )

        self.assertEqual(
            got,
            {
                ("only-a", "single", "poot", "ptx", DEVICE): bench.NOT_RUN,
                ("m", "single", "poot", "ptx", DEVICE): bench.WITHIN_NOISE,
                ("only-b", "single", "poot", "ptx", DEVICE): bench.NEW,
            },
        )


class CompareNoiseBandTests(unittest.TestCase):
    key = ("m", "single", "poot", "ptx", DEVICE)

    def verdict(self, candidate_samples, **kwargs):
        return verdicts(repeats(BASELINE), repeats(candidate_samples, **kwargs))[self.key]

    def test_twenty_percent_slower_is_a_regression(self):
        self.assertEqual(self.verdict([80.0, 79.0, 80.5, 81.0, 80.0]), bench.REGRESSION)

    def test_a_change_inside_the_observed_range_is_noise(self):
        # Median 97 is 3 below the baseline median, inside its 4-wide range.
        self.assertEqual(self.verdict([96.0, 97.0, 97.5, 96.5, 97.0]), bench.WITHIN_NOISE)
        self.assertEqual(self.verdict([103.0, 104.0, 103.5, 102.5, 103.0]), bench.WITHIN_NOISE)

    def test_the_band_edge_is_median_minus_twice_the_range(self):
        # Baseline median 100, range 4: the edge is 92. Exactly on it is noise, one step below regresses.
        self.assertEqual(self.verdict([92.0] * 5), bench.WITHIN_NOISE)
        self.assertEqual(self.verdict([91.5] * 5), bench.REGRESSION)
        self.assertEqual(self.verdict([108.0] * 5), bench.WITHIN_NOISE)
        self.assertEqual(self.verdict([108.5] * 5), bench.IMPROVEMENT)

    def test_a_noisier_baseline_widens_the_band(self):
        noisy = [80.0, 90.0, 100.0, 110.0, 120.0]
        candidate = repeats([75.0] * 5)

        quiet_verdict = verdicts(repeats(BASELINE), candidate)[self.key]
        noisy_verdict = verdicts(repeats(noisy), candidate)[self.key]

        self.assertEqual(quiet_verdict, bench.REGRESSION)
        self.assertEqual(noisy_verdict, bench.WITHIN_NOISE)

    def test_identical_baseline_runs_borrow_the_candidates_observed_noise(self):
        steady = repeats([100.0] * 5)

        # Baseline range 0: the candidate's own range (4) sizes the band, so 97 is noise and 80 is not.
        self.assertEqual(verdicts(steady, repeats([95.0, 97.0, 97.0, 97.0, 99.0]))[self.key],
                         bench.WITHIN_NOISE)
        self.assertEqual(verdicts(steady, repeats([78.0, 80.0, 80.0, 80.0, 82.0]))[self.key],
                         bench.REGRESSION)

    def test_identical_runs_on_both_sides_have_no_observed_noise_and_are_unverified(self):
        steady = repeats([100.0] * 5)

        for candidate in ([99.99] * 5, [80.0] * 5, [100.0] * 5):
            with self.subTest(candidate=candidate[0]):
                self.assertEqual(verdicts(steady, repeats(candidate))[self.key], bench.UNVERIFIED)

    def test_the_median_ignores_one_outlier_run(self):
        self.assertEqual(self.verdict([100.0, 100.0, 100.0, 100.0, 300.0]), bench.WITHIN_NOISE)

    def test_fewer_than_five_repeats_on_either_side_is_unverified(self):
        for baseline_runs, candidate_runs in ((5, 4), (4, 5), (1, 1)):
            with self.subTest(baseline=baseline_runs, candidate=candidate_runs):
                got = verdicts(repeats(BASELINE[:baseline_runs]), repeats(BASELINE[:candidate_runs]))
                self.assertEqual(got[self.key], bench.UNVERIFIED)

    def test_runs_of_different_binaries_do_not_make_one_band(self):
        mixed = repeats(BASELINE[:3], binary="bin-1") + repeats(BASELINE[3:], binary="bin-2")

        self.assertEqual(verdicts(mixed, repeats(BASELINE))[self.key], bench.UNVERIFIED)
        self.assertEqual(verdicts(repeats(BASELINE), mixed)[self.key], bench.UNVERIFIED)

    def test_a_candidate_that_fails_where_the_baseline_ran_is_a_failure(self):
        candidate = repeats(BASELINE[:4]) + repeats([0], status="error")
        failed = repeats([0], status="error") * 5

        self.assertEqual(verdicts(repeats(BASELINE), failed)[self.key], bench.CANDIDATE_FAILED)
        # A partly failed candidate is judged on its successful runs, which are too few for a band.
        self.assertEqual(verdicts(repeats(BASELINE), candidate)[self.key], bench.UNVERIFIED)

    def test_a_non_finite_sample_is_invalid_never_within_noise(self):
        for bad in (float("nan"), float("inf"), None):
            with self.subTest(bad=bad):
                candidate = repeats(BASELINE[:4]) + [result_row(bad)]
                self.assertEqual(verdicts(repeats(BASELINE), candidate)[self.key], bench.INVALID)
                self.assertEqual(verdicts(candidate, repeats(BASELINE))[self.key], bench.INVALID)


class CompareExitCodeTests(unittest.TestCase):
    def exit_code(self, rows_a, rows_b):
        return bench.compare_exit_code(bench.compare_cells(rows_a, rows_b))

    def test_regression_exits_1_noise_exits_0(self):
        self.assertEqual(self.exit_code(repeats(BASELINE), repeats([80.0] * 5)), 1)
        self.assertEqual(self.exit_code(repeats(BASELINE), repeats([97.0] * 5)), 0)
        self.assertEqual(self.exit_code(repeats(BASELINE), repeats([130.0] * 5)), 0)

    def test_unverifiable_compare_exits_2(self):
        self.assertEqual(self.exit_code(repeats(BASELINE[:2]), repeats(BASELINE[:2])), 2)
        # Nothing in common: nothing was compared, which is not a pass.
        self.assertEqual(self.exit_code(repeats(BASELINE, model="x"), repeats(BASELINE, model="y")), 2)
        self.assertEqual(self.exit_code([], []), 2)

    def test_a_regression_outranks_an_unverified_cell(self):
        baseline = repeats(BASELINE) + repeats(BASELINE[:1], model="thin")
        candidate = repeats([80.0] * 5) + repeats(BASELINE[:1], model="thin")

        self.assertEqual(self.exit_code(baseline, candidate), 1)

    def test_cli_exits_non_zero_on_a_twenty_percent_regression_and_zero_inside_the_range(self):
        cases = (
            ("twenty percent slower", [80.0, 79.0, 80.5, 81.0, 80.0], 1, "regression"),
            ("inside the observed range", [96.0, 97.0, 97.5, 96.5, 97.0], 0, "within noise"),
            ("only one run", [80.0], 2, "unverified"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            baseline = [write_run(tmp, f"base-{i}", [row]) for i, row in enumerate(repeats(BASELINE))]
            for name, samples, code, verdict in cases:
                with self.subTest(name):
                    candidate = [
                        write_run(tmp, f"{name}-{i}".replace(" ", "-"), [row])
                        for i, row in enumerate(repeats(samples))
                    ]

                    done = compare_cli(baseline, candidate)

                    self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                    self.assertIn(verdict, done.stdout)

    def test_cli_table_shows_one_row_per_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = repeats(BASELINE, backend="ptx") + repeats(BASELINE, backend="rocm")
            runs = [write_run(tmp, f"run-{i}", rows[i::5]) for i in range(5)]

            done = compare_cli(runs, runs)

        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        cells = [line for line in done.stdout.splitlines() if line.startswith("| m | single | poot |")]
        self.assertEqual(len(cells), 2, done.stdout)
        self.assertEqual({line.split("|")[4].strip() for line in cells}, {"ptx", "rocm"})


if __name__ == "__main__":
    unittest.main()
