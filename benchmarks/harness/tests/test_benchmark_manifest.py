"""The manifest carries model facts only for poot; a cell poot refuses is recorded with poot's own text."""

import json
import os
import stat
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

REFUSAL = "exact model stub_model has no accepted Wgpu Decode route"


def manifest():
    return bench.load_toml(bench.BENCH_ROOT / "manifest.toml")


class ManifestTests(unittest.TestCase):
    def test_no_poot_entry_carries_a_hand_written_support_list(self):
        for model in manifest()["model"]:
            entry = model["frameworks"].get("poot")
            with self.subTest(model=model["id"]):
                self.assertIsNotNone(
                    entry,
                    "an undeclared framework is recorded unsupported: poot must be declared for every model",
                )
                self.assertNotIn(
                    "support",
                    entry,
                    "whether poot runs a model is the runner's answer (a refusal), never a manifest flag",
                )
                self.assertNotIn("reason", entry)

    def test_every_poot_cell_is_run(self):
        poot_cells = [
            (model["id"], scenario["id"], fw_cfg)
            for model, scenario, fw, fw_cfg in bench.cells(manifest())
            if fw == "poot"
        ]

        self.assertTrue(poot_cells)
        for model_id, scenario_id, fw_cfg in poot_cells:
            with self.subTest(model=model_id, scenario=scenario_id):
                self.assertTrue(bench.is_supported(fw_cfg))

    def test_external_tools_keep_their_checked_gaps(self):
        gaps = [
            (model["id"], fw)
            for model, _scenario, fw, fw_cfg in bench.cells(manifest())
            if fw != "poot" and not bench.is_supported(fw_cfg)
        ]

        self.assertIn(("phi-4", "candle"), gaps)
        for model in manifest()["model"]:
            for fw, fw_cfg in model["frameworks"].items():
                if fw != "poot":
                    # An absent key means "run" (that is poot's rule); an external tool says so explicitly.
                    self.assertIn("support", fw_cfg, f"{model['id']}/{fw}: declare support explicitly")
                if fw_cfg.get("support") is False:
                    self.assertTrue(fw_cfg.get("reason"), f"{model['id']}/{fw}: a known gap needs its reason")


def refusing_runner(directory, payload):
    """A runner that reports `payload` as its result line and exits 0, as the poot runner does on a refusal."""
    path = Path(directory) / "refuser"
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{json.dumps(payload)}'\n", encoding="ascii")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def refusal_payload(**overrides):
    payload = {
        "framework": "poot",
        "build_sha": BUILD_SHA,
        "backend": "wgpu",
        "mode": "single",
        "precision": "bf16",
        "status": "unsupported",
        "reason": REFUSAL,
    }
    return payload | overrides


def run_poot_cell(binary, *, fw_cfg=None):
    with (
        mock.patch.object(observer, "run_observed", side_effect=run_fixture_runner),
        mock.patch.dict(os.environ, {"POOT_BENCH_BACKEND": "wgpu"}),
    ):
        return bench.run_cell(
            model={"id": "fixture", "match_precision": "bf16"},
            scen={"id": "single", "gen_tokens": 4},
            fw="poot",
            fw_cfg=fw_cfg or {"precision": "bf16"},
            runners={"poot": {"bin": str(binary), "cmd": ["{bin}"]}},
            models_dir=".",
            suite={},
            interval_ms=50,
            device="fixture-gpu",
        )


class RefusalTests(unittest.TestCase):
    def test_a_cell_the_runner_refuses_is_recorded_with_the_runners_own_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = run_poot_cell(refusing_runner(tmp, refusal_payload()))

        self.assertEqual(row["status"], "unsupported")
        self.assertEqual(row["reason"], REFUSAL)
        self.assertEqual(row["build_sha"], BUILD_SHA)
        self.assertEqual(row["backend"], "wgpu")

    def test_the_manifest_has_no_say_when_the_runner_answers(self):
        # A poot entry says nothing about support, so a runner that measures is recorded as ok.
        with tempfile.TemporaryDirectory() as tmp:
            measured = valid_single() | {"backend": "wgpu"}
            row = run_poot_cell(refusing_runner(tmp, measured))

        self.assertEqual(row["status"], "ok")

    def test_a_refusal_without_a_reason_is_an_error_not_a_silent_unsupported(self):
        for payload in (
            refusal_payload(reason=""),
            {k: v for k, v in refusal_payload().items() if k != "reason"},
        ):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                row = run_poot_cell(refusing_runner(tmp, payload))

            self.assertEqual(row["status"], "error")
            self.assertIn("reason", row["reason"])

    def test_a_refusal_names_the_build_and_backend_it_came_from(self):
        for payload in (
            refusal_payload(backend="rocm"),
            {k: v for k, v in refusal_payload().items() if k != "build_sha"},
        ):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                row = run_poot_cell(refusing_runner(tmp, payload))

            self.assertEqual(row["status"], "error")
            self.assertIn("invalid runner result", row["reason"])


if __name__ == "__main__":
    unittest.main()
