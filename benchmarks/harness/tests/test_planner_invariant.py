"""POOT-1520: the planner invariant on the device. A narrow case never measures slower than its exact twin."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import performance_bank as bank  # noqa: E402
import performance_receipt as pr  # noqa: E402
import planner_invariant as inv  # noqa: E402

EXACT_US = [1000.0, 1004.0, 996.0, 1002.0, 998.0, 1001.0, 999.0, 1003.0, 997.0]


def case(case_id, precision=None):
    numerics = {"accumulate": "f32", "reference": "f64 loop"}
    if precision:
        numerics.update(operand_precision=precision, operand_rounding="rounded")
    return {
        "id": case_id,
        "producer": "dense_contraction",
        "shape": {"m": 1, "k": 8, "n": 8},
        "storage": "bf16",
        "layout": "nk",
        "regime": case_id,
        "numerics": numerics,
        "backends": ["rocm"],
        "status": "runnable",
    }


def receipt(case_id, samples, *, binary="a" * 64, device="gpu", device_cell=True, plan_id=None):
    cells = {"kernel": {"repeats": []}}
    if device_cell:
        cells["device"] = {"repeats": [{"status": "ok", "metrics": {bank.DEVICE_METRIC: s}} for s in samples]}
    return {
        "case": case_id,
        "plan_id": plan_id or case_id,
        "backend": "rocm",
        "binary_sha256": binary,
        "parent_revision": "b" * 40,
        "identity": {"device": device, "driver": "d"},
        "cells": cells,
    }


class PlannerInvariant(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.definitions = {"cases": [case("x"), case("x-narrow", "narrow")]}

    def verdicts(self, exact, narrow):
        entries = []
        for name, rec in (("x", exact), ("x-narrow", narrow)):
            path = self.root / f"{name}.json"
            path.write_text(json.dumps(rec))
            entries.append({"case": name, "backend": "rocm", "status": "verified", "receipt": path.name})
        return inv.pair_verdicts(self.definitions, {"entries": entries}, self.root)

    def test_a_twin_is_the_case_that_differs_only_in_operand_precision(self):
        self.assertEqual([(e["id"], n["id"]) for e, n in inv.case_pairs(self.definitions)], [("x", "x-narrow")])
        self.assertEqual(inv.pair_errors(self.definitions), [])
        orphan = copy.deepcopy(self.definitions)
        orphan["cases"][1]["shape"] = {"m": 2, "k": 8, "n": 8}
        self.assertEqual(len(inv.pair_errors(orphan)), 1)

    def test_a_narrow_schedule_that_loses_beyond_the_band_is_a_regression(self):
        slower = [s * 1.3 for s in EXACT_US]
        [row] = self.verdicts(receipt("x", EXACT_US), receipt("x-narrow", slower))
        self.assertEqual(row["verdict"], pr.REGRESSION, row)

    def test_a_narrow_schedule_that_wins_or_ties_is_not(self):
        for factor, expected in ((0.5, pr.IMPROVEMENT), (1.001, pr.WITHIN_NOISE)):
            [row] = self.verdicts(receipt("x", EXACT_US), receipt("x-narrow", [s * factor for s in EXACT_US]))
            self.assertEqual(row["verdict"], expected, row)

    def test_twins_that_realized_one_plan_are_not_judged_by_their_drift(self):
        slower = [s * 1.3 for s in EXACT_US]
        [row] = self.verdicts(receipt("x", EXACT_US, plan_id="p"), receipt("x-narrow", slower, plan_id="p"))
        self.assertEqual(row["verdict"], inv.SAME_PLAN, row)
        self.assertEqual(inv.invariant_errors_of([row]), [])

    def test_twins_from_two_sessions_are_unmeasured_not_judged(self):
        slower = [s * 1.3 for s in EXACT_US]
        for narrow in (receipt("x-narrow", slower, binary="c" * 64), receipt("x-narrow", slower, device="other")):
            [row] = self.verdicts(receipt("x", EXACT_US), narrow)
            self.assertEqual(row["verdict"], "unmeasured", row)

    def test_a_side_without_a_device_cell_is_unmeasured(self):
        [row] = self.verdicts(receipt("x", EXACT_US, device_cell=False), receipt("x-narrow", EXACT_US))
        self.assertEqual(row["verdict"], "unmeasured", row)

    def loss_row(self):
        slower = [s * 1.3 for s in EXACT_US]
        [row] = self.verdicts(receipt("x", EXACT_US), receipt("x-narrow", slower))
        return row

    def declared(self, **overrides):
        entry = {
            "case": "x-narrow",
            "backend": "rocm",
            "median_narrow_us": 1300.0,
            "median_exact_us": 1000.0,
            "reason": "r",
            "follow_up": "POOT-1",
        }
        entry.update(overrides)
        return {(entry["case"], entry["backend"]): entry}

    def test_an_undeclared_loss_fails(self):
        self.assertEqual(len(inv.invariant_errors_of([self.loss_row()], {})), 1)

    def test_a_declared_loss_is_reported_with_its_follow_up_and_does_not_fail(self):
        row, declared = self.loss_row(), self.declared()
        self.assertEqual(inv.invariant_errors_of([row], declared), [])
        self.assertEqual(inv.declared_label(row, declared), "DECLARED LOSS (POOT-1)")
        self.assertEqual(inv.declared_label(row, {}), pr.REGRESSION)

    def test_a_declared_pair_that_no_longer_loses_fails(self):
        for factor in (1.001, 0.5):
            [row] = self.verdicts(receipt("x", EXACT_US), receipt("x-narrow", [s * factor for s in EXACT_US]))
            [error] = inv.invariant_errors_of([row], self.declared())
            self.assertIn("no longer loses", error)
        self.assertEqual(len(inv.invariant_errors_of([], self.declared())), 1)

    def test_a_declaration_covers_only_its_own_backend(self):
        self.assertEqual(len(inv.invariant_errors_of([self.loss_row()], self.declared(backend="ptx"))), 2)

    def test_the_declared_loss_list_is_checked_for_completeness(self):
        path = self.root / "losses.toml"
        good = (
            '[[loss]]\ncase = "c"\nbackend = "rocm"\nmedian_narrow_us = 2.0\nmedian_exact_us = 1.0\n'
            'reason = "r"\nfollow_up = "POOT-9"\n'
        )
        path.write_text(good)
        self.assertEqual(list(inv.load_losses(path)), [("c", "rocm")])
        self.assertEqual(inv.load_losses(self.root / "absent.toml"), {})
        for bad in (
            good.replace('follow_up = "POOT-9"\n', ""),
            good.replace("POOT-9", "later"),
            good.replace("2.0", "-2.0"),
            good + good,
        ):
            path.write_text(bad)
            with self.assertRaises(ValueError):
                inv.load_losses(path)

    def test_the_checked_in_declarations_name_a_follow_up_card(self):
        declared = inv.load_losses(inv.committed_losses_path())
        self.assertIn(("expert-prefill-packed-iq2xs-up-narrow", "rocm"), declared)

    def test_the_checked_in_bank_pairs_every_narrow_case_and_has_no_loss(self):
        self.assertEqual(inv.invariant_errors(), [])


if __name__ == "__main__":
    unittest.main()
