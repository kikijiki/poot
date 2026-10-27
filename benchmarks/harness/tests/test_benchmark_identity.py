"""A result is attributable: the committed rows state their identity, the comparator refuses a pair that differs in
any of it, the runner's precision is the store's, and `just bench-compare` runs the comparator over the committed set."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import performance_receipt as receipts  # noqa: E402
from result_contract import (  # noqa: E402
    BUILD_IDENTITY_FIELDS,
    MEASUREMENT_IDENTITY_FIELDS,
    committed_row_errors,
)
from runner_fixtures import POOT_IDENTITY, run_fixture_runner, valid_single  # noqa: E402
from test_benchmark_compare import BASELINE, repeats, result_row  # noqa: E402
from test_benchmark_manifest import refusing_runner, run_poot_cell  # noqa: E402

RESULTS = bench.BENCH_ROOT / "results"
REPO_ROOT = bench.BENCH_ROOT.parent

# The identity a comparison must agree on, written out here so that dropping a field from
# `ROW_IDENTITY_FIELDS` is a test failure and not a silently narrower comparison.
COMPARED_IDENTITY = (
    "precision",
    "prompt_tokens",
    "build_profile",
    "build_features",
    "rustc_version",
    "adapter",
    "driver_version",
    "checkpoint_digest",
    "compute_dtype",
    "prefill_chunk",
    "profiled",
)
# The local backends a dense cell is committed for (the PTX backend needs an NVIDIA device).
LOCAL_BACKENDS = ("wgpu", "rocm")


def committed_poot_rows():
    return [
        (run.name, row)
        for run in sorted(RESULTS.iterdir())
        if (run / "results.jsonl").is_file()
        for row in bench._load_rows(run)
        if row["framework"] == "poot"
    ]


class CommittedRowTests(unittest.TestCase):
    def test_every_committed_row_states_its_identity(self):
        rows = [
            (run, row)
            for run, row in committed_poot_rows()
            if row["status"] in ("ok", "unsupported")
        ]

        self.assertTrue(rows, "no committed poot row: the identity contract has nothing recorded under it")
        for run, row in rows:
            with self.subTest(run=run, scenario=row["scenario"]):
                self.assertEqual(committed_row_errors(row), [])

    def test_a_dense_cell_is_committed_for_each_local_backend_with_enough_repeats(self):
        sets = {}
        for run, row in committed_poot_rows():
            if row["status"] == "ok":
                sets.setdefault((row["backend"], row["model"], row["scenario"]), []).append(run)

        for backend in LOCAL_BACKENDS:
            with self.subTest(backend=backend):
                cells = {key: runs for key, runs in sets.items() if key[0] == backend}
                self.assertTrue(cells, f"no ok poot row on {backend}")
                self.assertTrue(
                    any(len(runs) >= bench.NOISE_REPEATS for runs in cells.values()),
                    f"{backend}: no cell has {bench.NOISE_REPEATS} committed repeats: {cells}",
                )

    def test_a_row_lacking_a_field_of_its_identity_is_rejected(self):
        (_, row) = next((run, row) for run, row in committed_poot_rows() if row["status"] == "ok")

        self.assertEqual(committed_row_errors(row), [])
        for field in (*BUILD_IDENTITY_FIELDS, *MEASUREMENT_IDENTITY_FIELDS, "backend"):
            with self.subTest(field=field):
                lacking = {key: value for key, value in row.items() if key != field}
                self.assertEqual(committed_row_errors(lacking), [f"missing required identity field {field!r}"])

    def test_a_row_of_a_dirty_debug_or_profiled_build_is_not_a_committed_result(self):
        (_, row) = next((run, row) for run, row in committed_poot_rows() if row["status"] == "ok")

        for change, expected in (
            ({"build_sha": row["build_sha"].removesuffix("-dirty") + "-dirty"}, "clean commit"),
            ({"build_profile": "debug"}, "only a release build"),
            ({"profiled": True}, "diagnostic"),
            ({"checkpoint_digest": "0" * 64}, "blake3"),
        ):
            with self.subTest(change=change):
                errors = committed_row_errors({**row, **change})
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_an_external_ok_row_names_the_version_of_its_tool(self):
        row = {"framework": "vllm", "status": "ok", "framework_version": "0.22.1"}

        self.assertEqual(committed_row_errors(row), [])
        self.assertEqual(committed_row_errors({"framework": "vllm", "status": "ok"}), [
            "an ok row of an external tool names its framework_version"
        ])


class EnvironmentTests(unittest.TestCase):
    def test_the_pod_image_digest_is_part_of_the_environment(self):
        digest = "sha256:" + "ab" * 32
        with mock.patch.dict(os.environ, {"POOT_BENCH_IMAGE_DIGEST": digest}):
            self.assertEqual(bench.capture_env("manifest.toml")["image_digest"], digest)
        with mock.patch.dict(os.environ):
            os.environ.pop("POOT_BENCH_IMAGE_DIGEST", None)
            self.assertIsNone(bench.capture_env("manifest.toml")["image_digest"])


class DiagnosticRunTests(unittest.TestCase):
    def test_a_committed_diagnostic_run_is_profiled_and_never_a_committed_set(self):
        diagnostics = RESULTS / bench.DIAGNOSTICS_DIR
        runs = [run for run in sorted(diagnostics.iterdir()) if (run / "results.jsonl").is_file()] if diagnostics.is_dir() else []

        self.assertTrue(runs, "no committed diagnostic run: the harness has never requested the device-time data")
        for run in runs:
            for row in bench._load_rows(run):
                with self.subTest(run=run.name, scenario=row["scenario"]):
                    self.assertEqual(row["status"], "ok")
                    self.assertIs(row["profiled"], True)
                    self.assertIn(row["device_time"]["kind"], ("measured", "unknown"))
                    self.assertTrue(committed_row_errors(row), "a profiled row must not pass as a committed result")
        self.assertNotIn(bench.DIAGNOSTICS_DIR, {name for name in bench.committed_sets(RESULTS)})
        self.assertFalse(any(run.parent == diagnostics for dirs in bench.committed_sets(RESULTS).values() for run in dirs))

    def test_a_profiled_row_compares_only_with_a_profiled_one(self):
        profiled = [{**row, "profiled": True} for row in repeats(BASELINE)]

        (cell,) = bench.compare_cells(repeats(BASELINE), profiled, receipts.POLICY)

        self.assertEqual(cell["verdict"], receipts.INCOMPARABLE)


class IdentityComparisonTests(unittest.TestCase):
    def verdict(self, baseline, candidate):
        (cell,) = bench.compare_cells(baseline, candidate, receipts.POLICY)
        return cell["verdict"]

    def test_the_compared_identity_is_the_documented_one(self):
        self.assertTrue(set(COMPARED_IDENTITY) <= set(receipts.ROW_IDENTITY_FIELDS))

    def test_rows_that_differ_in_any_identity_field_are_incomparable(self):
        changed = {
            "precision": "f16",
            "prompt_tokens": 99,
            "build_profile": "debug",
            "build_features": "fast-math",
            "rustc_version": "rustc 1.0.0",
            "adapter": "Other GPU",
            "driver_version": "other driver 2.0",
            "checkpoint_digest": "blake3:" + "00" * 32,
            "compute_dtype": "bf16",
            "prefill_chunk": 2,
            "profiled": True,
        }
        self.assertEqual(set(changed), set(COMPARED_IDENTITY))
        for field, value in changed.items():
            with self.subTest(field=field):
                candidate = [{**row, field: value} for row in repeats(BASELINE)]
                self.assertEqual(self.verdict(repeats(BASELINE), candidate), receipts.INCOMPARABLE)

    def test_a_chunked_prefill_is_another_workload_than_the_whole_prompt(self):
        """The fixture prompt is 3 tokens. A row that states no chunk (recorded before the field) and one whose
        chunk holds the prompt prefilled it whole, so they compare; a chunk below the prompt does not.
        Mutation: drop `prefill_chunk` from `ROW_IDENTITY_FIELDS`; the chunk-2 candidate compares as noise."""
        whole = repeats(BASELINE, prefill_chunk=None)
        self.assertEqual(self.verdict(whole, repeats(BASELINE, prefill_chunk=3)), "within noise")
        self.assertEqual(self.verdict(whole, repeats(BASELINE, prefill_chunk=2)), receipts.INCOMPARABLE)
        self.assertEqual(
            self.verdict(repeats(BASELINE, prefill_chunk=1), repeats(BASELINE, prefill_chunk=2)),
            receipts.INCOMPARABLE,
        )

    def test_the_build_sha_is_what_a_comparison_varies_not_identity(self):
        candidate = [
            {**row, "build_sha": "f" * 40, "binary_sha256": "bin-2"} for row in repeats([80.0, 79.0, 80.5, 81.0, 80.0])
        ]

        self.assertEqual(self.verdict(repeats(BASELINE), candidate), receipts.REGRESSION)

    def test_repeats_of_one_side_that_loaded_different_checkpoints_have_no_band(self):
        mixed = repeats(BASELINE)
        mixed[0] = {**mixed[0], "checkpoint_digest": "blake3:" + "00" * 32}

        (cell,) = bench.compare_cells(mixed, repeats(BASELINE), receipts.POLICY)

        self.assertEqual(cell["verdict"], receipts.UNVERIFIED)
        self.assertIn("checkpoint_digest", cell["note"])

    def test_a_baseline_tool_whose_version_moved_is_not_the_cell_it_was_recorded_as(self):
        def external(version):
            return [
                {
                    "framework": "vllm",
                    "framework_version": version,
                    "model": "m",
                    "scenario": "single",
                    "status": "ok",
                    "precision": "bf16",
                    "prompt_tokens": 3,
                    "gen_tokens": 4,
                    "decode_tok_s": tok_s,
                }
                for tok_s in BASELINE
            ]

        self.assertEqual(self.verdict(external("0.22.1"), external("0.23.0")), receipts.INCOMPARABLE)


class PrecisionIsTheStoresTests(unittest.TestCase):
    def test_a_cell_whose_manifest_precision_the_store_does_not_hold_is_an_error_row(self):
        # The manifest says bf16; the runner reports what its store holds (here Q8_0 weights).
        with tempfile.TemporaryDirectory() as tmp:
            row = run_poot_cell(refusing_runner(tmp, valid_single() | {"backend": "wgpu", "precision": "q8_0"}))

        self.assertEqual(row["status"], "error")
        self.assertIn("precision mismatch: requested 'bf16', reported 'q8_0'", row["reason"])

    def test_the_manifest_precision_the_store_holds_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = run_poot_cell(refusing_runner(tmp, valid_single() | {"backend": "wgpu"}))

        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["precision"], "bf16")
        self.assertEqual({field: row[field] for field in POOT_IDENTITY}, POOT_IDENTITY)


class BenchCompareRecipeTests(unittest.TestCase):
    """`just bench-compare`: the one place the comparator runs over the committed results."""

    @classmethod
    def setUpClass(cls):
        cls.just = shutil.which("just")
        if cls.just is None:
            raise AssertionError("`just` is required to test the bench-compare recipe (run inside `nix develop`)")
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)

    def fixture_results(self):
        """A controlled results directory: two cells, each five repeats of `BASELINE` in its own run set (the
        older one recorded a day before), so the recipe contract does not depend on the noise of measured data."""
        results = Path(tempfile.mkdtemp(dir=self.work.name)) / "results"
        for day, model in (("2026-10-20", "older"), ("2026-10-21", "newer")):
            for i, tok_s in enumerate(BASELINE, 1):
                run = results / f"{day}-fixture-gpu-{model}-r{i}"
                run.mkdir(parents=True)
                (run / "results.jsonl").write_text(json.dumps(result_row(tok_s, model=model)) + "\n")
                (run / "env.json").write_text(json.dumps({"captured_at_utc": f"{day}T00:00:00+00:00"}))
        return results

    def recipe(self, results, *candidate):
        return subprocess.run(
            [self.just, "bench-compare", str(results), *map(str, candidate)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def planted(self, results, scale):
        """Candidate run directories beside (not inside) the committed results: the baseline's sets again (the
        newest set of each cell), the newest one with every `decode_tok_s` scaled by `scale`."""
        baseline_dirs, _ = bench.last_committed_set(results)
        sets = {}
        for source in baseline_dirs:
            sets.setdefault(bench._REPEAT_SUFFIX.sub("", source.name), []).append(source)
        newest = max(sets)
        candidate_root = Path(tempfile.mkdtemp(dir=self.work.name))
        dirs = []
        for name, sources in sets.items():
            for source in sources:
                target = candidate_root / source.name
                target.mkdir()
                factor = scale if name == newest else 1.0
                rows = [
                    {**row, "decode_tok_s": row["decode_tok_s"] * factor} if row.get("decode_tok_s") else row
                    for row in bench._load_rows(source)
                ]
                (target / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                dirs.append(target)
        return dirs

    def test_a_verifiable_set_is_a_verifiable_baseline(self):
        done = self.recipe(self.fixture_results())

        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(receipts.WITHIN_NOISE, done.stdout)

    def test_the_committed_results_are_a_verifiable_baseline(self):
        # The data gate `just test` runs through `test-bench-harness`, on the real measurements: a cell whose
        # robust band stays over its limit (a two-mode cell) fails it until it is re-recorded, by design.
        done = self.recipe(RESULTS)

        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(receipts.WITHIN_NOISE, done.stdout)

    def test_a_twenty_percent_regression_in_the_candidate_exits_non_zero(self):
        results = self.fixture_results()

        done = self.recipe(results, *self.planted(results, 0.8))

        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn(receipts.REGRESSION, done.stdout)

    def test_the_same_candidate_unchanged_passes(self):
        results = self.fixture_results()

        done = self.recipe(results, *self.planted(results, 1.0))

        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_the_recipe_runs_the_comparator(self):
        listing = subprocess.run(
            [self.just, "--dry-run", "bench-compare"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )

        self.assertIn("benchmarks/harness/bench.py compare-committed", listing.stderr)

    def test_the_benchmark_gate_runs_the_recipe_and_the_harness_tests(self):
        listing = subprocess.run(
            [self.just, "--dry-run", "test-bench-harness"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )

        for command in ("compare-committed", "unittest discover -s benchmarks/harness/tests", "performance_bank.py"):
            self.assertIn(command, listing.stderr)


if __name__ == "__main__":
    unittest.main()
