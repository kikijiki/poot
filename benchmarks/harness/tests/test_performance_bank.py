"""The checked-in performance bank: identities, claims and the receipt files that back them."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import performance_bank as bank  # noqa: E402
import performance_receipt as pr  # noqa: E402


def verified_pairs(root=bank.BANK_ROOT):
    manifest = bank.load_json(Path(root) / "receipts.json")
    return [(e["case"], e["backend"]) for e in manifest["entries"] if e["status"] == "verified"]


class BankRoot(unittest.TestCase):
    """A scratch copy of the bank that a test may corrupt."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "performance-bank"
        shutil.copytree(bank.BANK_ROOT, self.root)

    def cases(self):
        return bank.load_json(self.root / "cases.json")

    def write_cases(self, data):
        (self.root / "cases.json").write_text(json.dumps(data, indent=2))

    def manifest(self):
        return bank.load_json(self.root / "receipts.json")

    def write_manifest(self, data):
        (self.root / "receipts.json").write_text(json.dumps(data, indent=2))

    def errors(self):
        return bank.validate_bank(self.root)

    def entry(self, case, backend):
        return next(e for e in self.manifest()["entries"] if (e["case"], e["backend"]) == (case, backend))

    def edit_entry(self, case, backend, **changes):
        data = self.manifest()
        next(e for e in data["entries"] if (e["case"], e["backend"]) == (case, backend)).update(changes)
        self.write_manifest(data)

    def edit_receipt(self, case, backend, edit, *, resign=True):
        """Change a receipt file; `resign` keeps the claim's receipt_sha256 honest so only the content is wrong."""
        entry = self.entry(case, backend)
        path = self.root / entry["receipt"]
        receipt = json.loads(path.read_text())
        edit(receipt)
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        if resign:
            self.edit_entry(case, backend, receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class CheckedInBankTests(unittest.TestCase):
    def test_the_checked_in_bank_and_its_claims_validate(self):
        self.assertEqual(bank.validate_bank(), [])

    def test_every_claimed_backend_of_every_case_has_a_claim(self):
        data = bank.load_json(bank.BANK_ROOT / "cases.json")
        claims = {(e["case"], e["backend"]) for e in bank.load_json(bank.BANK_ROOT / "receipts.json")["entries"]}
        for case in data["cases"]:
            for backend in case["backends"]:
                self.assertIn((case["id"], backend), claims)

    def test_the_bank_spans_the_required_regimes(self):
        regimes = {case["regime"] for case in bank.load_json(bank.BANK_ROOT / "cases.json")["cases"]}
        for needed in (
            "dense bandwidth decode",
            "packed bandwidth decode",
            "batched decode",
            "compute-heavy short prefill",
            "compute-heavy long prefill",
            "low-head low-occupancy attention",
            "long recurrence",
            "sorted expert routing",
            "shuffled expert routing",
        ):
            self.assertIn(needed, regimes)

    def test_planned_cases_are_never_coverage(self):
        data = bank.load_json(bank.BANK_ROOT / "cases.json")
        planned = {c["id"] for c in data["cases"] if c["status"] == bank.PLANNED}
        self.assertTrue(planned, "recurrence and the unproduced variants stay explicit missing coverage")
        for case, _ in verified_pairs():
            self.assertNotIn(case, planned)

    def test_a_cold_weights_case_rotates_more_than_the_cache_it_names(self):
        data = bank.load_json(bank.BANK_ROOT / "cases.json")
        for case in data["cases"]:
            for backend in case["backends"]:
                if (case["id"], backend) not in verified_pairs():
                    continue
                with self.subTest(case=case["id"], backend=backend):
                    receipt = bank.load_json(bank.BANK_ROOT / "receipts" / backend / f"{case['id']}.json")
                    self.assertEqual(bank.observed_errors(data, case, receipt["observed"]), [])

    def test_a_verified_receipt_compares_with_a_later_candidate_of_the_same_identity(self):
        pairs = verified_pairs()
        self.assertTrue(pairs, "this card records hardware baselines for the runnable cases")
        case, backend = pairs[0]
        receipt = bank.load_json(bank.BANK_ROOT / "receipts" / backend / f"{case}.json")
        spec = {"cells": {"kernel": [bank.KERNEL_METRIC]}}
        samples = [r["metrics"][bank.KERNEL_METRIC] for r in receipt["cells"]["kernel"]["repeats"]]
        shift = 20 * sorted(samples)[len(samples) // 2]
        slower = {**receipt, "cells": {"kernel": {"repeats": [
            {"status": "ok", "metrics": {bank.KERNEL_METRIC: sample + shift}} for sample in samples
        ]}}}
        report = pr.compare_receipts(receipt, slower, spec, pr.POLICY)
        # Same identity: the pair compares. A slowdown of twenty medians clears even a noisy shared-GPU baseline's band.
        self.assertEqual(report["identity_mismatches"], [])
        self.assertEqual(report["entries"][0]["verdict"], pr.REGRESSION)


class DefinitionIdentityTests(BankRoot):
    def test_editing_a_shape_without_restamping_is_refused(self):
        data = self.cases()
        data["cases"][0]["shape"]["k"] += 64
        self.write_cases(data)
        errors = self.errors()
        self.assertTrue(any("fixture_id" in e for e in errors), errors)
        self.assertTrue(any("identity_digest" in e for e in errors), errors)

    def test_editing_the_cache_recipe_without_restamping_is_refused(self):
        data = self.cases()
        data["cache_conditions"]["weights-rotated-beyond-cache"]["min_rotation_bytes"] //= 8
        self.write_cases(data)
        errors = self.errors()
        self.assertTrue(any("cache_id" in e for e in errors), errors)

    def test_editing_the_token_fixture_without_restamping_is_refused(self):
        data = self.cases()
        data["token_fixtures"]["splitmix64-seed0-unit"]["seed"] = 1
        self.write_cases(data)
        self.assertTrue(any("fixture_id" in e for e in self.errors()))

    def test_editing_a_schedule_without_restamping_is_refused(self):
        data = self.cases()
        data["cases"][0]["schedules"]["candidate"] = "gemv-w128"
        self.write_cases(data)
        self.assertTrue(any("schedule_id" in e for e in self.errors()))

    def test_restamping_an_intended_edit_changes_every_dependent_identity_and_stales_receipts(self):
        data = self.cases()
        data["cache_conditions"]["weights-rotated-beyond-cache"]["min_rotation_bytes"] *= 2
        self.write_cases(data)
        bank.stamp_bank(self.root)
        errors = self.errors()
        # The bank is internally sound again, but every receipt measured under the old recipe is refused.
        self.assertTrue(errors)
        self.assertFalse(any(e.startswith("dense-decode-gemv:") and "identity_digest is" in e for e in errors))
        self.assertTrue(any("identity_digest is not the case's" in e or "identity differs" in e for e in errors), errors)

    def test_a_case_naming_an_undefined_fixture_or_condition_is_refused(self):
        data = self.cases()
        data["cases"][0]["token_fixture"] = "nope"
        data["cases"][1]["cache"] = "nope"
        self.write_cases(data)
        errors = self.errors()
        self.assertTrue(any("token_fixture 'nope'" in e for e in errors), errors)
        self.assertTrue(any("cache 'nope'" in e for e in errors), errors)

    def test_a_planned_case_states_why(self):
        data = self.cases()
        planned = next(c for c in data["cases"] if c["status"] == bank.PLANNED)
        del planned["planned_reason"]
        self.write_cases(data)
        self.assertTrue(any("states why" in e for e in self.errors()))


class ClaimTests(BankRoot):
    def first_verified(self):
        pairs = verified_pairs(self.root)
        self.assertTrue(pairs)
        return pairs[0]

    def test_a_required_backend_with_no_claim_is_refused(self):
        data = self.manifest()
        data["entries"] = [e for e in data["entries"] if (e["case"], e["backend"]) != ("dense-decode-gemv", "ptx")]
        self.write_manifest(data)
        self.assertIn(
            "('dense-decode-gemv', 'ptx'): required but neither verified nor declared missing or refused",
            self.errors(),
        )

    def test_a_missing_backend_marked_verified_without_a_receipt_is_refused(self):
        # A claim that names a receipt file the bank does not hold (here: a deleted one) is not coverage.
        (self.root / self.entry("dense-decode-gemv", "ptx")["receipt"]).unlink()
        errors = self.errors()
        self.assertTrue(any("('dense-decode-gemv', 'ptx')" in e and "is not a file" in e for e in errors), errors)

    def test_a_planned_case_cannot_be_verified_even_with_a_receipt(self):
        case, backend = self.first_verified()
        source = self.root / self.entry(case, backend)["receipt"]
        planned = next(c for c in self.cases()["cases"] if c["status"] == bank.PLANNED)
        target = self.root / "receipts" / "wgpu" / f"{planned['id']}.json"
        shutil.copy(source, target)
        self.edit_entry(
            planned["id"],
            "wgpu",
            status="verified",
            receipt=f"receipts/wgpu/{planned['id']}.json",
            receipt_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            command=bank.bank_command(self.cases(), planned, "wgpu"),
        )
        self.assertTrue(any("a planned case cannot be verified" in e for e in self.errors()))

    def test_a_missing_claim_needs_its_reason_and_names_no_receipt(self):
        data = self.manifest()
        entry = next(e for e in data["entries"] if e["status"] == "missing")
        del entry["reason"]
        entry["receipt"] = "receipts/wgpu/x.json"
        self.write_manifest(data)
        errors = self.errors()
        self.assertTrue(any("needs the reason it is not covered" in e for e in errors), errors)
        self.assertTrue(any("cannot name a receipt" in e for e in errors), errors)

    def test_a_claim_for_a_backend_the_case_does_not_claim_is_refused(self):
        data = self.manifest()
        data["entries"].append({"case": "dense-decode-gemv", "backend": "vulkan", "status": "missing", "reason": "x"})
        self.write_manifest(data)
        self.assertTrue(any("does not claim this backend" in e for e in self.errors()))

    def test_a_claimed_twice_pair_is_refused(self):
        data = self.manifest()
        data["entries"].append(dict(data["entries"][0]))
        self.write_manifest(data)
        self.assertTrue(any("claimed twice" in e for e in self.errors()))


class SpreadTests(BankRoot):
    def test_a_claimed_spread_that_the_repeats_do_not_give_is_refused(self):
        self.edit_entry("dense-decode-gemv", "wgpu", spread=0.001, exceeds_max_uncertainty=False)
        errors = self.errors()
        self.assertTrue(any("spread is 0.001" in e for e in errors), errors)
        self.assertTrue(any("exceeds_max_uncertainty is False" in e for e in errors), errors)

    def test_a_noisy_baseline_is_flagged_and_a_quiet_one_is_not(self):
        base = bank.load_json(self.root / self.entry("dense-decode-gemv", "wgpu")["receipt"])
        def with_samples(values):
            return {**base, "cells": {"kernel": {"repeats": [
                {"status": "ok", "metrics": {bank.KERNEL_METRIC: v}} for v in values]}}}
        self.assertTrue(bank.spread_of(with_samples([100, 130, 100, 100, 100]))["exceeds_max_uncertainty"])
        self.assertFalse(bank.spread_of(with_samples([100, 101, 100, 100, 100]))["exceeds_max_uncertainty"])


class ReceiptTests(BankRoot):
    def pair(self):
        # A cold-weights (rotation) case, so the rotation fixtures apply.
        pair = ("dense-decode-gemv", "wgpu")
        self.assertIn(pair, verified_pairs(self.root))
        return pair

    def assert_refused(self, fragment, edit, *, resign=True):
        case, backend = self.pair()
        self.edit_receipt(case, backend, edit, resign=resign)
        errors = self.errors()
        self.assertTrue(any(fragment in e for e in errors), (fragment, errors))

    def test_a_receipt_edited_after_it_was_claimed_is_refused(self):
        self.assert_refused("receipt_sha256 does not match", lambda r: r.update(plan_id="forged"), resign=False)

    def test_an_edited_fixture_condition_with_an_unchanged_identity_is_refused(self):
        # The receipt keeps the case's identity digest but reports a measurement under another condition.
        def lighter_rotation(receipt):
            receipt["observed"]["weight_copies"] = 1

        self.assert_refused("rotate", lighter_rotation)

    def test_an_edited_observed_shape_is_refused(self):
        self.assert_refused("observed shape", lambda r: r["observed"]["shape"].update(k=1))

    def test_an_identity_that_drifted_from_the_case_is_refused(self):
        def drift(receipt):
            receipt["identity"]["precision"] = "bf16"

        self.assert_refused("identity differs from the case definition in: precision", drift)

    def test_an_identity_digest_that_is_not_the_cases_is_refused(self):
        self.assert_refused("identity_digest is not the case's", lambda r: r.update(identity_digest="0" * 64))

    def test_a_schedule_identity_that_is_not_the_cases_is_refused(self):
        self.assert_refused("schedule_id is not the case's", lambda r: r.update(schedule_id="x"))

    def test_internal_parity_is_not_an_independent_reference(self):
        self.assert_refused(
            "only an 'independent-golden' backs a speed receipt",
            lambda r: r["reference"].update(provenance="internal-parity"),
        )

    def test_a_measurement_that_disagrees_with_its_reference_is_refused(self):
        self.assert_refused("does not agree with its reference", lambda r: r["reference"].update(agrees=False))

    def test_a_receipt_for_another_backend_is_refused(self):
        self.assert_refused("receipt is for", lambda r: r.update(backend="ptx"))

    def test_a_non_baseline_role_is_refused(self):
        self.assert_refused("role must be 'baseline'", lambda r: r.update(role="candidate"))

    def test_too_few_or_failed_repeats_are_refused(self):
        self.assert_refused("repeats recorded", lambda r: r["cells"]["kernel"].update(repeats=r["cells"]["kernel"]["repeats"][:2]))

        def fail_one(receipt):
            receipt["cells"]["kernel"]["repeats"][0]["status"] = "failed"

        self.assert_refused("repeat 0 is not a finite successful", fail_one)

    def test_a_forged_binary_or_revision_is_refused(self):
        self.assert_refused("binary_sha256 must be", lambda r: r.update(binary_sha256="abc"))
        self.assert_refused("parent_revision must be", lambda r: r.update(parent_revision="HEAD"))

    def test_a_command_that_is_not_the_checked_in_one_is_refused(self):
        self.assert_refused("command is not the checked-in bank command", lambda r: r.update(command=["true"]))
        case, backend = self.pair()
        self.edit_entry(case, backend, command=["true"])
        self.assertTrue(any("claimed command is not the checked-in bank command" in e for e in self.errors()))


def fake_runner(case, backend, *, observed=None):
    """What a backend example writes: raw measurement and what it executed, with no identity of its own."""

    def run(command, *, cwd, env, check):
        out = Path(cwd) / command[command.index("--out") + 1]
        out.parent.mkdir(parents=True, exist_ok=True)
        binary = Path(cwd) / "fake-binary"
        binary.write_bytes(b"fake binary")
        data = bank.load_json(Path(cwd) / "benchmarks/performance-bank/cases.json")
        case_def = next(c for c in data["cases"] if c["id"] == case)
        out.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "case": case,
                    "backend": backend,
                    "role": "baseline",
                    "hardware": {"backend": backend, "device": env["POOT_BANK_DEVICE"], "driver": env["POOT_BANK_DRIVER"]},
                    "binary": str(binary),
                    "observed": observed or {
                        "shape": case_def["shape"],
                        "storage": case_def["storage"],
                        "layout": case_def["layout"],
                        "weight_copies": 16,
                        "weight_bytes_per_copy": 16 * 2**20,
                    },
                    "plan": ["fixture kernel"],
                    "plan_id": "plan",
                    "reference": {"provenance": "independent-golden", "agrees": True},
                    "cells": {"kernel": {"repeats": [
                        {"slot": i, "status": "ok", "metrics": {bank.KERNEL_METRIC: 100.0 + i}} for i in range(9)
                    ]}},
                }
            )
        )

    return run


class RecordTests(unittest.TestCase):
    """`record` on a scratch git repository, with a fake runner in place of cargo."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        self.root = self.repo / "benchmarks" / "performance-bank"
        self.root.mkdir(parents=True)
        shutil.copy(bank.BANK_ROOT / "cases.json", self.root / "cases.json")
        cases = bank.load_json(self.root / "cases.json")["cases"]
        (self.root / "receipts.json").write_text(json.dumps({"schema": 1, "entries": [
            {"case": c["id"], "backend": b, "status": "missing", "reason": "not measured"}
            for c in cases for b in c["backends"]
        ]}))
        for step in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "bank"]):
            subprocess.run(["git", "-C", str(self.repo), *step], check=True, capture_output=True)

    def record(self, case="dense-decode-gemv", backend="wgpu", **kwargs):
        return bank.record(
            case, backend, root=self.root, repo=self.repo, device=("fixture-gpu", "fixture-driver"),
            run=kwargs.pop("run", fake_runner(case, backend)), **kwargs,
        )

    def test_a_recorded_receipt_is_a_verified_claim_that_validates(self):
        path = self.record()

        self.assertEqual(bank.validate_bank(self.root), [])
        receipt = bank.load_json(path)
        self.assertEqual(receipt["identity"]["device"], "fixture-gpu")
        self.assertEqual(len(receipt["parent_revision"]), 40)
        self.assertEqual(receipt["binary_sha256"], hashlib.sha256(b"fake binary").hexdigest())
        self.assertIn(("dense-decode-gemv", "wgpu"), verified_pairs(self.root))
        self.assertNotIn("hardware", receipt)
        entry = next(e for e in bank.load_json(self.root / "receipts.json")["entries"]
                     if (e["case"], e["backend"]) == ("dense-decode-gemv", "wgpu"))
        self.assertEqual(entry["spread"], round(8 / 104, 4))

    def test_a_runner_that_executed_another_workload_is_refused_and_registers_nothing(self):
        wrong = {"shape": {"m": 1, "k": 1024, "n": 2048}, "storage": "f32", "layout": "kn",
                 "weight_copies": 16, "weight_bytes_per_copy": 2**23}
        before = (self.root / "receipts.json").read_text()
        with self.assertRaises(SystemExit) as refused:
            self.record(run=fake_runner("dense-decode-gemv", "wgpu", observed=wrong))
        self.assertIn("observed shape", str(refused.exception))
        self.assertEqual((self.root / "receipts.json").read_text(), before)

    def test_uncommitted_source_changes_refuse_the_receipt(self):
        (self.repo / "source.rs").write_text("fn main() {}\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "source.rs"], check=True)
        with self.assertRaises(SystemExit) as refused:
            self.record()
        self.assertIn("uncommitted source changes", str(refused.exception))

    def test_a_planned_case_cannot_be_recorded(self):
        with self.assertRaises(SystemExit):
            self.record(case="long-recurrence-scan")

    def test_a_host_that_names_no_device_is_refused(self):
        with self.assertRaises(SystemExit) as refused:
            bank.record(
                "dense-decode-gemv", "wgpu", root=self.root, repo=self.repo, device=(None, None),
                run=fake_runner("dense-decode-gemv", "wgpu"),
            )
        self.assertIn("reports no device", str(refused.exception))


if __name__ == "__main__":
    unittest.main()
