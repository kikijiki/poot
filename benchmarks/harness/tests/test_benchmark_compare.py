import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import performance_receipt as receipts  # noqa: E402
from runner_fixtures import POOT_IDENTITY  # noqa: E402

DEVICE = "fixture-gpu"
HOST_CPU = "fixture-cpu"
DRIVER = "fixture-driver"


def result_row(tok_s, *, backend="ptx", device=DEVICE, host_cpu=HOST_CPU, gpu_driver=DRIVER, binary="bin-1", status="ok", model="m", **extra):
    row = {
        "framework": "poot",
        "model": model,
        "precision": "bf16",
        "scenario": "single",
        "status": status,
        "backend": backend,
        "device": device,
        "host_cpu": host_cpu,
        "gpu_driver": gpu_driver,
        "binary_sha256": binary,
        **POOT_IDENTITY,
        "build_sha": binary.ljust(40, "0"),
        "profiled": False,
        "prompt_tokens": 3,
        "gen_tokens": 4,
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


# Five repeats with median 100 and interquartile range 2: the noise is 2.33 * 2 / 1.349 = 3.45 and a change
# needs to clear twice that, 6.9.
BASELINE = [98.0, 99.0, 100.0, 101.0, 102.0]


def verdicts(rows_a, rows_b):
    return {cell["key"]: cell["verdict"] for cell in bench.compare_cells(rows_a, rows_b, receipts.POLICY)}


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

        cells = bench.compare_cells(baseline, candidate, receipts.POLICY)

        self.assertEqual(len(cells), 2)
        by_backend = {cell["key"][3]: cell for cell in cells}
        self.assertEqual(sorted(by_backend), ["ptx", "rocm"])
        self.assertEqual(by_backend["ptx"]["verdict"], receipts.WITHIN_NOISE)
        self.assertEqual(by_backend["ptx"]["median_a"], 100.0)
        self.assertEqual(by_backend["rocm"]["verdict"], receipts.REGRESSION)
        self.assertEqual(by_backend["rocm"]["median_a"], 300.0)
        self.assertEqual((by_backend["ptx"]["n_a"], by_backend["rocm"]["n_a"]), (5, 5))

    def test_devices_of_the_same_model_and_backend_are_separate_cells(self):
        baseline = repeats(BASELINE, device="gpu-x") + repeats([x * 2 for x in BASELINE], device="gpu-y")
        candidate = repeats(BASELINE, device="gpu-x") + repeats([x * 2 for x in BASELINE], device="gpu-y")

        cells = bench.compare_cells(baseline, candidate, receipts.POLICY)

        self.assertEqual([cell["key"][4] for cell in cells], ["gpu-x", "gpu-y"])
        self.assertEqual([cell["median_a"] for cell in cells], [100.0, 200.0])

    def test_a_backend_or_device_the_row_does_not_record_is_its_own_cell(self):
        legacy = repeats(BASELINE, backend=None, device=None)
        legacy = [{k: v for k, v in row.items() if k not in ("backend", "device", "host_cpu", "gpu_driver")} for row in legacy]

        cells = bench.compare_cells(legacy + repeats(BASELINE), legacy + repeats(BASELINE), receipts.POLICY)

        self.assertEqual({cell["key"][3:] for cell in cells}, {(None, None, None, None), ("ptx", DEVICE, HOST_CPU, DRIVER)})

    def test_hosts_of_the_same_gpu_are_separate_cells(self):
        # The same GPU on two host CPUs: a host-bound metric differs between them, so one band over both would
        # be two-mode. Each host keeps its own cell and its own verdict.
        baseline = repeats(BASELINE, host_cpu="cpu-a") + repeats([x * 2 for x in BASELINE], host_cpu="cpu-b")

        cells = bench.compare_cells(baseline, baseline, receipts.POLICY)

        self.assertEqual([cell["key"][5] for cell in cells], ["cpu-a", "cpu-b"])
        self.assertEqual([(cell["n_a"], cell["median_a"]) for cell in cells], [(5, 100.0), (5, 200.0)])

    def test_drivers_of_the_same_gpu_are_separate_cells(self):
        rows = repeats(BASELINE, gpu_driver="535") + repeats(BASELINE, gpu_driver="550")

        cells = bench.compare_cells(rows, rows, receipts.POLICY)

        self.assertEqual([cell["key"][6] for cell in cells], ["535", "550"])

    def test_the_same_hardware_still_pools_its_repeats(self):
        rows = repeats(BASELINE[:3]) + repeats(BASELINE[3:])

        (cell,) = bench.compare_cells(rows, rows, receipts.POLICY)

        self.assertEqual(cell["n_a"], 5)

    def test_a_host_the_run_did_not_record_is_never_pooled_with_a_known_one(self):
        unknown = repeats(BASELINE, host_cpu=None)

        cells = bench.compare_cells(unknown + repeats(BASELINE), unknown + repeats(BASELINE), receipts.POLICY)

        self.assertEqual({cell["key"][5] for cell in cells}, {None, HOST_CPU})
        self.assertEqual([cell["n_a"] for cell in cells], [5, 5])

    def test_a_run_directory_with_no_recorded_cpu_is_not_pooled_with_a_known_cpu(self):
        # The production path: the host comes from each run directory's env.json, and a directory whose env.json
        # lacks the cpu (or has no env.json) is an unknown host. Its repeats stay out of the known host's band.
        def bare(samples):
            return [{k: v for k, v in row.items() if k not in ("host_cpu", "gpu_driver")} for row in repeats(samples)]

        with tempfile.TemporaryDirectory() as tmp:
            known = write_run(tmp, "known", bare(BASELINE))
            (Path(known) / "env.json").write_text(json.dumps({"cpu": HOST_CPU, "gpu_driver": DRIVER}))
            no_cpu = write_run(tmp, "no-cpu", bare(BASELINE))
            (Path(no_cpu) / "env.json").write_text(json.dumps({"gpu_driver": DRIVER}))
            rows = [row for run in (known, no_cpu) for row in bench._load_rows(run)]

        cells = bench.compare_cells(rows, rows, receipts.POLICY)

        self.assertEqual([(cell["key"][5], cell["n_a"]) for cell in cells], [(None, 5), (HOST_CPU, 5)])

    def test_cell_present_on_one_side_only_is_reported_not_dropped(self):
        got = verdicts(
            repeats(BASELINE, model="only-a") + repeats(BASELINE),
            repeats(BASELINE) + repeats(BASELINE, model="only-b"),
        )

        self.assertEqual(
            got,
            {
                ("only-a", "single", "poot", "ptx", DEVICE, HOST_CPU, DRIVER): receipts.NOT_RUN,
                ("m", "single", "poot", "ptx", DEVICE, HOST_CPU, DRIVER): receipts.WITHIN_NOISE,
                ("only-b", "single", "poot", "ptx", DEVICE, HOST_CPU, DRIVER): receipts.NEW,
            },
        )


class CompareNoiseBandTests(unittest.TestCase):
    key = ("m", "single", "poot", "ptx", DEVICE, HOST_CPU, DRIVER)

    def verdict(self, candidate_samples, **kwargs):
        return verdicts(repeats(BASELINE), repeats(candidate_samples, **kwargs))[self.key]

    def test_twenty_percent_slower_is_a_regression(self):
        self.assertEqual(self.verdict([80.0, 79.0, 80.5, 81.0, 80.0]), receipts.REGRESSION)

    def test_a_change_inside_the_observed_range_is_noise(self):
        # Median 97 is 3 below the baseline median, inside its 4-wide range.
        self.assertEqual(self.verdict([96.0, 97.0, 97.5, 96.5, 97.0]), receipts.WITHIN_NOISE)
        self.assertEqual(self.verdict([103.0, 104.0, 103.5, 102.5, 103.0]), receipts.WITHIN_NOISE)

    def test_the_band_edge_is_median_minus_twice_the_robust_noise(self):
        # Baseline median 100, noise 3.45: the edge is 93.1. Inside it is noise, past it regresses.
        self.assertEqual(self.verdict([93.2] * 5), receipts.WITHIN_NOISE)
        self.assertEqual(self.verdict([93.0] * 5), receipts.REGRESSION)
        self.assertEqual(self.verdict([106.8] * 5), receipts.WITHIN_NOISE)
        self.assertEqual(self.verdict([107.0] * 5), receipts.IMPROVEMENT)

    def test_a_noisier_baseline_widens_the_band_until_it_cannot_exclude_a_regression(self):
        noisy = [80.0, 90.0, 100.0, 110.0, 120.0]
        candidate = repeats([75.0] * 5)

        quiet_verdict = verdicts(repeats(BASELINE), candidate)[self.key]
        noisy_verdict = verdicts(repeats(noisy), candidate)[self.key]

        self.assertEqual(quiet_verdict, receipts.REGRESSION)
        # A band this wide swallows a 25% loss: that is unverified, never "within noise".
        self.assertEqual(noisy_verdict, receipts.UNVERIFIED)

    def test_identical_baseline_runs_borrow_the_candidates_observed_noise(self):
        steady = repeats([100.0] * 5)

        # Baseline range 0: the candidate's own noise (3.45) sizes the band, so 97 is noise and 80 is not.
        self.assertEqual(verdicts(steady, repeats([94.0, 96.0, 97.0, 98.0, 100.0]))[self.key],
                         receipts.WITHIN_NOISE)
        self.assertEqual(verdicts(steady, repeats([76.0, 78.0, 80.0, 82.0, 84.0]))[self.key],
                         receipts.REGRESSION)

    def test_identical_runs_on_both_sides_have_no_observed_noise_and_are_unverified(self):
        steady = repeats([100.0] * 5)

        for candidate in ([99.99] * 5, [80.0] * 5, [100.0] * 5):
            with self.subTest(candidate=candidate[0]):
                self.assertEqual(verdicts(steady, repeats(candidate))[self.key], receipts.UNVERIFIED)

    def test_one_wild_run_does_not_set_the_band(self):
        # The median ignores the 300 and so does the robust noise: the baseline's own 3.45 sizes the band.
        self.assertEqual(self.verdict([100.0, 100.0, 100.0, 100.0, 300.0]), receipts.WITHIN_NOISE)

    def test_fewer_than_five_repeats_on_either_side_is_unverified(self):
        for baseline_runs, candidate_runs in ((5, 4), (4, 5), (1, 1)):
            with self.subTest(baseline=baseline_runs, candidate=candidate_runs):
                got = verdicts(repeats(BASELINE[:baseline_runs]), repeats(BASELINE[:candidate_runs]))
                self.assertEqual(got[self.key], receipts.UNVERIFIED)

    def test_runs_of_different_binaries_do_not_make_one_band(self):
        mixed = repeats(BASELINE[:3], binary="bin-1") + repeats(BASELINE[3:], binary="bin-2")

        self.assertEqual(verdicts(mixed, repeats(BASELINE))[self.key], receipts.UNVERIFIED)
        self.assertEqual(verdicts(repeats(BASELINE), mixed)[self.key], receipts.UNVERIFIED)

    def test_a_candidate_that_fails_where_the_baseline_ran_is_a_failure(self):
        candidate = repeats(BASELINE[:4]) + repeats([0], status="error")
        failed = repeats([0], status="error") * 5

        self.assertEqual(verdicts(repeats(BASELINE), failed)[self.key], receipts.CANDIDATE_FAILED)
        # A failed repeat is never dropped: the four that succeeded cannot stand in for the one that did not.
        self.assertEqual(verdicts(repeats(BASELINE), candidate)[self.key], receipts.REPEAT_FAILED)
        sufficient = repeats(BASELINE) + repeats([0], status="error")
        self.assertEqual(verdicts(repeats(BASELINE), sufficient)[self.key], receipts.REPEAT_FAILED)

    def test_a_non_finite_sample_is_invalid_never_within_noise(self):
        for bad in (float("nan"), float("inf"), None):
            with self.subTest(bad=bad):
                candidate = repeats(BASELINE[:4]) + [result_row(bad)]
                self.assertEqual(verdicts(repeats(BASELINE), candidate)[self.key], receipts.INVALID)
                self.assertEqual(verdicts(candidate, repeats(BASELINE))[self.key], receipts.INVALID)


def curve_row(isl_values, *, tpot_scale=None, ttft_scale=None, tok_s=100.0, jitter=1.0, **extra):
    """One decode-curve row: a point per context length, TPOT/TTFT scaled per context by the given maps."""
    tpot_scale, ttft_scale = tpot_scale or {}, ttft_scale or {}
    points = [
        {
            "isl": isl,
            "prompt_tokens": isl,
            "tpot_ms": {"p50": 10.0 * jitter * tpot_scale.get(isl, 1.0)},
            "ttft_ms": {"p50": 0.5 * isl * jitter * ttft_scale.get(isl, 1.0)},
        }
        for isl in isl_values
    ]
    row = result_row(tok_s, curve=points, osl=128, **extra)
    # A curve row states its prompt lengths per point; it has no top-level prompt_tokens.
    del row["prompt_tokens"]
    return row


def curve_repeats(isls, **kwargs):
    # Every repeat differs a little so the runs have a band; the headline decode_tok_s stays in 98..102.
    return [curve_row(isls, tok_s=tok_s, jitter=tok_s / 100.0, **kwargs) for tok_s in BASELINE]


class CompareCurveCoverageTests(unittest.TestCase):
    key = ("m", "single", "poot", "ptx", DEVICE, HOST_CPU, DRIVER)
    ISLS = (128, 2048, 8192)

    def cell(self, candidate, baseline=None):
        (got,) = bench.compare_cells(baseline or curve_repeats(self.ISLS), candidate, receipts.POLICY)
        return got

    def test_matching_curves_are_within_noise_on_every_point(self):
        got = self.cell(curve_repeats(self.ISLS))
        self.assertEqual(got["verdict"], receipts.WITHIN_NOISE)
        self.assertEqual(len(got["entries"]), 1 + 2 * len(self.ISLS))

    def test_a_doubled_long_context_tpot_regresses_with_an_unchanged_headline(self):
        got = self.cell(curve_repeats(self.ISLS, tpot_scale={8192: 2.0}))
        self.assertEqual(got["verdict"], receipts.REGRESSION)
        self.assertIn("isl=8192 tpot_ms", got["note"])
        headline = [e for e in got["entries"] if e["cell"] == "headline"][0]
        self.assertEqual(headline["verdict"], receipts.WITHIN_NOISE)

    def test_a_doubled_long_prefill_ttft_regresses_with_an_unchanged_headline(self):
        got = self.cell(curve_repeats(self.ISLS, ttft_scale={8192: 2.0}))
        self.assertEqual(got["verdict"], receipts.REGRESSION)
        self.assertIn("isl=8192 ttft_ms", got["note"])

    def test_curve_rows_are_compared_by_their_prompt_lengths_not_a_missing_top_level_one(self):
        """Mutation: read only a top-level `prompt_tokens` for identity and every curve cell is incomparable."""
        for framework in ("poot", "llamacpp"):
            with self.subTest(framework=framework):
                extra = {"framework": framework, "framework_version": "b1"}
                got = self.cell(curve_repeats(self.ISLS, **extra), baseline=curve_repeats(self.ISLS, **extra))
                self.assertEqual(got["verdict"], receipts.WITHIN_NOISE, got["note"])

    def test_a_curve_and_a_single_prompt_state_different_workloads(self):
        candidate = [{**row, "prompt_tokens": 3} for row in curve_repeats(self.ISLS)]
        for row in candidate:
            del row["curve"]
        (got,) = bench.compare_cells(curve_repeats(self.ISLS), candidate, receipts.POLICY)
        self.assertEqual(got["verdict"], receipts.INCOMPARABLE)
        self.assertIn("prompt_tokens", got["note"])

    def test_a_context_length_the_candidate_did_not_run_is_not_run_and_exits_2(self):
        candidate = curve_repeats((128, 2048))
        got = self.cell(candidate)
        self.assertEqual(got["verdict"], receipts.NOT_RUN)
        self.assertIn("isl=8192", got["note"])
        self.assertEqual(bench.compare_exit_code([got]), 2)

    def test_a_failed_repeat_among_sufficient_curve_repeats_prevents_a_pass(self):
        got = self.cell(curve_repeats(self.ISLS) + repeats([0], status="error"))
        self.assertEqual(got["verdict"], receipts.REPEAT_FAILED)
        self.assertEqual(bench.compare_exit_code([got]), 1)

    def test_curve_rows_state_their_prompt_length_per_point(self):
        def with_prompts(row, prompt_for):
            return {**row, "curve": [{**p, "prompt_tokens": prompt_for(p["isl"])} for p in row["curve"]]}

        baseline = [with_prompts(r, lambda isl: isl) for r in curve_repeats(self.ISLS)]
        (same,) = bench.compare_cells(baseline, baseline, receipts.POLICY)
        self.assertNotEqual(same["verdict"], receipts.INCOMPARABLE)
        shifted = [with_prompts(r, lambda isl: isl + 1) for r in curve_repeats(self.ISLS)]
        (got,) = bench.compare_cells(baseline, shifted, receipts.POLICY)
        self.assertEqual(got["verdict"], receipts.INCOMPARABLE)

    def test_rows_that_state_different_workloads_are_incomparable(self):
        for field, value in (("precision", "fp16"), ("prompt_tokens", 99), ("gen_tokens", 7)):
            with self.subTest(field=field):
                candidate = [{**row, field: value} for row in repeats(BASELINE)]
                (got,) = bench.compare_cells(repeats(BASELINE), candidate, receipts.POLICY)
                self.assertEqual(got["verdict"], receipts.INCOMPARABLE)
                self.assertEqual(bench.compare_exit_code([got]), 1)


class CompareExitCodeTests(unittest.TestCase):
    def exit_code(self, rows_a, rows_b):
        return bench.compare_exit_code(bench.compare_cells(rows_a, rows_b, receipts.POLICY))

    def test_a_cell_the_candidate_did_not_run_is_not_a_complete_pass(self):
        got = self.exit_code(repeats(BASELINE) + repeats(BASELINE, model="x"), repeats(BASELINE))
        self.assertEqual(got, 2)

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
