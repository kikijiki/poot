import hashlib
import json
import os
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
from runner_fixtures import BUILD_SHA, run_fixture_runner, valid_single  # noqa: E402


def checkout_head():
    """The commit of the checkout the harness runs in: what the harness used to record as the binary's sha."""
    return subprocess.run(
        ["git", "-C", str(bench.BENCH_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def install_build(directory, name, build_sha, backend="ptx"):
    """A fake runner binary built at `build_sha`: an executable that prints that build's result."""
    payload = valid_single() | {"build_sha": build_sha, "backend": backend}
    path = Path(directory) / name
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{json.dumps(payload)}'\n", encoding="ascii")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def run_poot_cell(binary, *, backend="ptx", device=None):
    with (
        mock.patch.object(observer, "run_observed", side_effect=run_fixture_runner),
        mock.patch.dict(os.environ, {"POOT_BENCH_BACKEND": backend}),
    ):
        return bench.run_cell(
            model={"id": "fixture", "match_precision": "bf16"},
            scen={"id": "single", "gen_tokens": 4},
            fw="poot",
            fw_cfg={"support": True, "precision": "bf16"},
            runners={"poot": {"bin": str(binary), "cmd": ["{bin}"]}},
            models_dir=".",
            suite={},
            interval_ms=50,
            device=device,
        )


class BinaryProvenanceTests(unittest.TestCase):
    def test_row_records_the_sha_the_binary_was_built_from_not_the_checkout_head(self):
        head = checkout_head()
        with tempfile.TemporaryDirectory() as tmp:
            builds = {}
            for name, build_sha in (("build-a", "a" * 40), ("build-b", "b" * 40)):
                self.assertNotEqual(build_sha, head)
                binary = install_build(tmp, name, build_sha)
                builds[build_sha] = (binary, run_poot_cell(binary))

        for build_sha, (binary, row) in builds.items():
            with self.subTest(build_sha=build_sha):
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["build_sha"], build_sha)
                self.assertNotEqual(row["build_sha"], head)
        (_, row_a), (_, row_b) = builds.values()
        self.assertNotEqual(row_a["binary_sha256"], row_b["binary_sha256"])

    def test_row_records_the_digest_of_the_executable_that_ran(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = install_build(tmp, "build-a", "a" * 40)
            expected = hashlib.sha256(binary.read_bytes()).hexdigest()
            row = run_poot_cell(binary)
            # The same build path holding different bytes is a different binary.
            install_build(tmp, "build-a", "c" * 40)
            rebuilt = run_poot_cell(binary)

        self.assertEqual(row["binary_sha256"], expected)
        self.assertNotEqual(rebuilt["binary_sha256"], expected)
        self.assertEqual(rebuilt["build_sha"], "c" * 40)

    def test_row_records_the_backend_and_device_the_cell_ran_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = install_build(tmp, "build-a", "a" * 40, backend="rocm")
            row = run_poot_cell(binary, backend="rocm", device="AMD Radeon 8060S Graphics")

        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["backend"], "rocm")
        self.assertEqual(row["device"], "AMD Radeon 8060S Graphics")

    def test_failed_cell_still_records_which_binary_was_asked_to_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = install_build(tmp, "build-a", "a" * 40)
            expected = hashlib.sha256(binary.read_bytes()).hexdigest()
            failed = dict(
                run_fixture_runner([str(binary)]), returncode=3, stderr="runner crashed"
            )
            with (
                mock.patch.object(observer, "run_observed", return_value=failed),
                mock.patch.dict(os.environ, {"POOT_BENCH_BACKEND": "wgpu"}),
            ):
                row = bench.run_cell(
                    model={"id": "fixture"},
                    scen={"id": "single", "gen_tokens": 4},
                    fw="poot",
                    fw_cfg={"support": True},
                    runners={"poot": {"bin": str(binary), "cmd": ["{bin}"]}},
                    models_dir=".",
                    suite={},
                    interval_ms=50,
                    device="fixture-gpu",
                )

        self.assertEqual(row["status"], "error")
        self.assertEqual(row["binary_sha256"], expected)
        self.assertEqual(row["backend"], "wgpu")
        self.assertEqual(row["device"], "fixture-gpu")
        self.assertNotIn("build_sha", row)

    def test_env_capture_does_not_claim_a_commit_for_the_binary(self):
        env = bench.capture_env("manifest.toml")

        self.assertNotIn("poot_git_sha", env)
        self.assertNotIn("poot_git_describe", env)


class BuildIdentityContractTests(unittest.TestCase):
    single_scenario = {"id": "single", "gen_tokens": 4}

    def validate(self, payload, backend="ptx"):
        return bench.validate_runner_result(
            payload,
            framework="poot",
            precision="bf16",
            scenario=self.single_scenario,
            gen_tokens=4,
            backend=backend,
        )

    def test_poot_result_without_build_sha_is_rejected(self):
        payload = valid_single()
        del payload["build_sha"]

        with self.assertRaisesRegex(
            bench.ResultValidationError, "missing required identity field 'build_sha'"
        ):
            self.validate(payload)

    def test_poot_result_without_backend_is_rejected(self):
        payload = valid_single()
        del payload["backend"]

        with self.assertRaisesRegex(
            bench.ResultValidationError, "missing required identity field 'backend'"
        ):
            self.validate(payload)

    def test_poot_result_from_another_backend_than_requested_is_rejected(self):
        with self.assertRaisesRegex(
            bench.ResultValidationError,
            "backend mismatch: requested 'wgpu', reported 'ptx'",
        ):
            self.validate(valid_single(), backend="wgpu")

    def test_build_sha_must_be_a_full_git_sha_with_an_optional_dirty_mark(self):
        for build_sha in ("unknown", "", "0123abcd", BUILD_SHA.upper(), BUILD_SHA + "-wip", 7):
            with self.subTest(build_sha=build_sha):
                with self.assertRaisesRegex(bench.ResultValidationError, "build_sha must be"):
                    self.validate(valid_single() | {"build_sha": build_sha})
        for build_sha in (BUILD_SHA, BUILD_SHA + "-dirty"):
            with self.subTest(build_sha=build_sha):
                self.assertEqual(self.validate(valid_single() | {"build_sha": build_sha})["build_sha"], build_sha)

    def test_backend_must_be_a_string(self):
        with self.assertRaisesRegex(bench.ResultValidationError, "backend must be a string"):
            self.validate(valid_single() | {"backend": 3}, backend=None)

    def test_other_frameworks_keep_the_backend_they_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = {}
            for backend in ("vulkan", "rocm"):
                payload = valid_single() | {"framework": "llamacpp", "backend": backend}
                del payload["build_sha"]
                script = Path(tmp) / f"llama-{backend}"
                script.write_text(f"#!/bin/sh\nprintf '%s\\n' '{json.dumps(payload)}'\n", encoding="ascii")
                script.chmod(script.stat().st_mode | stat.S_IXUSR)
                with mock.patch.object(observer, "run_observed", side_effect=run_fixture_runner):
                    rows[backend] = bench.run_cell(
                        model={"id": "fixture", "match_precision": "bf16"},
                        scen={"id": "single", "gen_tokens": 4},
                        fw="llamacpp",
                        fw_cfg={"support": True, "precision": "bf16"},
                        runners={"llamacpp": {"bin": str(script), "cmd": ["{bin}"]}},
                        models_dir=".",
                        suite={},
                        interval_ms=50,
                        device="fixture-gpu",
                    )

        self.assertEqual({b: r["backend"] for b, r in rows.items()}, {"vulkan": "vulkan", "rocm": "rocm"})
        # Same engine, same device, different backend: two cells, not one.
        keys = {bench.cell_key(r) for r in rows.values()}
        self.assertEqual(len(keys), 2)

    def test_other_frameworks_need_no_build_identity(self):
        payload = valid_single() | {"framework": "candle"}
        del payload["build_sha"], payload["backend"]

        result = bench.validate_runner_result(
            payload,
            framework="candle",
            precision="bf16",
            scenario=self.single_scenario,
            gen_tokens=4,
        )
        self.assertEqual(result["framework"], "candle")


if __name__ == "__main__":
    unittest.main()
