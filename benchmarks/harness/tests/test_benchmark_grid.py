"""The standing grid is complete: every required cell is verified in the Performance page data, pending, or has a reason."""

import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import grid  # noqa: E402

MANIFEST = {
    "performance": {"min_trials": 5},
    "scenario": [{"id": "decode-128", "gen_tokens": 128}, {"id": "curve", "isl": [128, 8192]}],
    "model": [
        {"id": "m", "frameworks": {"poot": {"precision": "bf16"}, "llamacpp": {"support": True}, "vllm": {"support": False}}},
    ],
}


def point(isl, backend="wgpu", framework="poot", **overrides):
    return {"model": "m", "framework": framework, "backend": backend, "isl": isl, "latest": True, "verified": True,
            "repeats": 5, **overrides}


def performance(*points):
    return {"hardware": [{"id": "hw", "points": list(points)}]}


def declared(*groups, repeats=5):
    return {
        "repeats": repeats,
        "machine": [{"id": "box", "hardware": "hw"}],
        "group": [{"machine": "box", "scenario": "curve", "models": ["m"], **g} for g in groups],
    }


WGPU = {"engines": [{"framework": "poot", "backend": "wgpu"}]}
BOTH = [point(128), point(8192)]


class JudgeTests(unittest.TestCase):
    def test_a_cell_with_every_length_verified_is_satisfied(self):
        report = grid.judge(declared(WGPU), MANIFEST, performance(*BOTH))
        self.assertEqual((report.failures, report.satisfied), ([], 1))

    def test_a_required_cell_missing_a_length_fails_naming_the_length(self):
        report = grid.judge(declared(WGPU), MANIFEST, performance(point(128)))
        self.assertEqual(len(report.failures), 1)
        self.assertIn("[8192]", report.failures[0])

    def test_unverified_superseded_or_thin_points_do_not_count(self):
        for bad in ({"verified": False}, {"latest": False}, {"repeats": 4}):
            with self.subTest(bad=bad):
                report = grid.judge(declared(WGPU), MANIFEST, performance(point(128), point(8192, **bad)))
                self.assertEqual(len(report.failures), 1)

    def test_another_backend_framework_or_machine_does_not_satisfy_the_cell(self):
        for other in (point(8192, backend="rocm"), point(8192, framework="llamacpp")):
            with self.subTest(other=other):
                report = grid.judge(declared(WGPU), MANIFEST, performance(point(128), other))
                self.assertEqual(len(report.failures), 1)
        elsewhere = {"hardware": [{"id": "other-hw", "points": BOTH}]}
        self.assertEqual(len(grid.judge(declared(WGPU), MANIFEST, elsewhere).failures), 1)

    def test_a_pending_cell_may_lack_data_and_is_listed(self):
        report = grid.judge(declared({**WGPU, "status": "pending"}), MANIFEST, performance())
        self.assertEqual((report.failures, len(report.pending)), ([], 1))

    def test_a_pending_cell_with_its_data_is_reported_measured(self):
        report = grid.judge(declared({**WGPU, "status": "pending"}), MANIFEST, performance(*BOTH))
        self.assertEqual((report.pending, len(report.measured_pending)), ([], 1))

    def test_a_not_run_cell_is_never_judged_against_data(self):
        report = grid.judge(declared({**WGPU, "status": "not-run", "reason": "too slow"}), MANIFEST, performance())
        self.assertEqual((report.failures, len(report.not_run)), ([], 1))


class DeclarationTests(unittest.TestCase):
    def assertRejected(self, grid_value, text):
        with self.assertRaises(ValueError) as caught:
            grid.load_cells(grid_value, MANIFEST)
        self.assertIn(text, str(caught.exception))

    def test_a_not_run_cell_needs_its_reason(self):
        self.assertRejected(declared({**WGPU, "status": "not-run"}), "needs its reason")

    def test_only_a_not_run_cell_carries_a_reason(self):
        self.assertRejected(declared({**WGPU, "status": "pending", "reason": "why"}), "only a not-run cell")
        self.assertRejected(declared({**WGPU, "reason": "why"}), "only a not-run cell")

    def test_an_unknown_status_model_scenario_or_machine_is_rejected(self):
        self.assertRejected(declared({**WGPU, "status": "later"}), "status 'later'")
        self.assertRejected(declared({**WGPU, "models": ["nope"]}), "not in the manifest")
        self.assertRejected(declared({**WGPU, "scenario": "decode-128"}), "not a decode-curve scenario")
        self.assertRejected(declared({**WGPU, "machine": "nowhere"}), "not declared")

    def test_a_framework_the_manifest_says_does_not_run_the_model_is_rejected(self):
        self.assertRejected(declared({"engines": [{"framework": "vllm", "backend": "cuda"}]}), "does not run")

    def test_a_cell_declared_twice_is_rejected(self):
        self.assertRejected(declared(WGPU, WGPU), "twice")

    def test_fewer_repeats_than_the_verdict_minimum_are_rejected(self):
        self.assertRejected(declared(WGPU, repeats=4), "at least 5")


class ReleaseGateTests(unittest.TestCase):
    """`check --strict` through the real entry point, on a grid that declares one pending cell."""

    def check(self, strict):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "grid.toml").write_text(
                'repeats = 5\n[[machine]]\nid = "box"\nhardware = "hw"\n'
                '[[group]]\nmachine = "box"\nscenario = "curve"\nmodels = ["m"]\nstatus = "pending"\n'
                'engines = [{ framework = "poot", backend = "wgpu" }]\n'
            )
            (root / "manifest.toml").write_text(
                '[performance]\nmin_trials = 5\n[[scenario]]\nid = "curve"\nisl = [128, 8192]\n'
                '[[model]]\nid = "m"\n[model.frameworks.poot]\nprecision = "bf16"\n'
            )
            (root / "performance.json").write_text(json.dumps(performance()))
            out = io.StringIO()
            code = grid.check(strict, root / "grid.toml", root / "manifest.toml", root / "performance.json", out)
            return code, out.getvalue()

    def test_the_release_gate_fails_while_a_cell_is_pending(self):
        code, text = self.check(strict=True)
        self.assertEqual(code, 1, text)
        self.assertIn("pending (release gate)", text)

    def test_the_standing_check_allows_a_pending_cell(self):
        code, text = self.check(strict=False)
        self.assertEqual(code, 0, text)


class CommittedGridTests(unittest.TestCase):
    """The committed grid against the committed manifest and Performance page data."""

    def test_the_committed_grid_is_complete(self):
        out = io.StringIO()
        self.assertEqual(grid.check(strict=False, out=out), 0, out.getvalue())

    def test_deleting_a_measured_cell_from_the_data_fails_the_check(self):
        data = json.loads(grid.PERFORMANCE.read_text())
        mutated = copy.deepcopy(data)
        for entry in mutated["hardware"]:
            entry["points"] = [
                p for p in entry["points"]
                if (p["model"], p["framework"], p["backend"]) != ("qwen3.8-flash-next-q2", "poot", "ptx")
            ]
        report = grid.judge(grid.load_toml(grid.GRID), grid.load_toml(grid.MANIFEST), mutated)
        self.assertEqual(len(report.failures), 1)
        self.assertIn("qwen3.8-flash-next-q2 / poot ptx / decode-curve-large", report.failures[0])

    def test_the_nvidia_machine_declares_the_secure_rtx_pro_6000_cap(self):
        flags = grid.machine_arguments("nvidia-rtx-pro-6000")
        self.assertEqual(flags[flags.index("--cloud") + 1], "secure")
        self.assertEqual(flags[flags.index("--max-price-per-hour") + 1], "2.7")
        self.assertIn("RTX PRO 6000 Blackwell Server Edition", flags[flags.index("--gpu-types") + 1])


if __name__ == "__main__":
    unittest.main()
