"""The performance comparator and receipt validator: identity, completeness, uncertainty, windows, preparation.

Every fixture is synthetic and deterministic; none reads a clock.
"""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import performance_receipt as pr  # noqa: E402

IDENTITY = {
    "checkpoint_digest": "sha256:aaaa",
    "token_fixture": "seeded-random/seed=0/vocab=151936",
    "precision": "bf16",
    "layout": "row-major",
    "dimensions": {"hidden": 896, "layers": 24},
    "prompt_tokens": 128,
    "live_context": 128,
    "max_context": 8192,
    "requested_output": 128,
    "actual_output": 128,
    "sampler": "greedy",
    "template": "none",
    "cache_state": {"state": pr.COLD_POPULATED, "weights": "resident"},
    "prepared_set": ["prefill@128", "decode@1"],
    "timing_policy": {"warmup": 2, "measured": 5, "profiled": False},
    "backend": "rocm",
    "device": "fixture-gpu",
    "driver": "fixture-driver-1",
}
# The identity fields a record must state, written out here and not read back from the module under test:
# a field dropped from the module must fail its named fixture, not silently drop out of the loop below.
EXPECTED_IDENTITY_FIELDS = (
    "checkpoint_digest", "token_fixture", "precision", "layout", "dimensions", "prompt_tokens", "live_context",
    "max_context", "requested_output", "actual_output", "sampler", "template", "cache_state", "prepared_set",
    "timing_policy", "backend", "device", "driver",
)
SPEC = {"cells": {"isl=128": ["tpot_ms", "ttft_ms"], "isl=8192": ["tpot_ms", "ttft_ms"]}}


def repeats(samples, **extra):
    """One repeat per sample, all of one cell metric; `extra` is merged into every repeat."""
    return [
        {"slot": slot, "status": "ok", "metrics": dict(sample), **extra}
        for slot, sample in enumerate(samples)
    ]


def cell(tpot, ttft, **extra):
    return {
        "repeats": [
            {"slot": slot, "status": "ok", "metrics": {"tpot_ms": a, "ttft_ms": b}, **extra}
            for slot, (a, b) in enumerate(zip(tpot, ttft, strict=True))
        ]
    }


TPOT = [9.8, 9.9, 10.0, 10.1, 10.2]
TTFT = [98.0, 99.0, 100.0, 101.0, 102.0]


def receipt(**overrides):
    cells = {"isl=128": cell(TPOT, TTFT), "isl=8192": cell([t * 4 for t in TPOT], [t * 40 for t in TTFT])}
    cells.update(overrides.pop("cells", {}))
    return {"identity": overrides.pop("identity", copy.deepcopy(IDENTITY)), "cells": cells}


def verdicts(report):
    return {(e["cell"], e["metric"]): e["verdict"] for e in report["entries"]}


class IdentityTests(unittest.TestCase):
    def test_the_module_requires_exactly_the_declared_identity_fields(self):
        self.assertEqual(set(pr.IDENTITY_FIELDS), set(EXPECTED_IDENTITY_FIELDS))
        self.assertEqual(set(IDENTITY), set(EXPECTED_IDENTITY_FIELDS))

    def test_every_identity_field_independently_makes_records_incomparable(self):
        base = receipt()
        for field in EXPECTED_IDENTITY_FIELDS:
            with self.subTest(field=field):
                changed = copy.deepcopy(IDENTITY)
                changed[field] = {"different": True}
                report = pr.compare_receipts(base, receipt(identity=changed), SPEC, pr.POLICY)
                self.assertEqual(report["verdict"], pr.INCOMPARABLE)
                self.assertEqual([f for f, _ in report["identity_mismatches"]], [field])
                self.assertFalse(report["passed"])
                self.assertEqual(report["entries"], [], "an incomparable pair must not produce a speed comparison")

    def test_a_field_one_side_does_not_state_is_a_mismatch_not_a_wildcard(self):
        for field in EXPECTED_IDENTITY_FIELDS:
            with self.subTest(field=field):
                partial = {k: v for k, v in IDENTITY.items() if k != field}
                for baseline, candidate in ((receipt(identity=partial), receipt()), (receipt(), receipt(identity=partial))):
                    report = pr.compare_receipts(baseline, candidate, SPEC, pr.POLICY)
                    self.assertEqual(report["verdict"], pr.INCOMPARABLE)
                    self.assertEqual([f for f, _ in report["identity_mismatches"]], [field])

    def test_identical_identities_compare(self):
        report = pr.compare_receipts(receipt(), receipt(), SPEC, pr.POLICY)
        self.assertEqual((report["status"], report["passed"]), (pr.COMPLETE, True), report)

    def test_the_digest_covers_exactly_the_identity_fields(self):
        digest = pr.identity_digest(IDENTITY)
        self.assertEqual(digest, pr.identity_digest({**IDENTITY, "build_sha": "extra provenance"}))
        for field in EXPECTED_IDENTITY_FIELDS:
            with self.subTest(field=field):
                self.assertNotEqual(digest, pr.identity_digest({**IDENTITY, field: "changed"}))

    def test_a_perturbed_diagnostic_repeat_cannot_supply_a_speed_metric(self):
        profiled = cell(TPOT, TTFT)
        profiled["repeats"][2]["perturbed"] = True
        for side in ("candidate", "baseline"):
            with self.subTest(side=side):
                pair = (receipt(), receipt(cells={"isl=128": profiled}))
                report = pr.compare_receipts(*(pair if side == "candidate" else pair[::-1]), SPEC, pr.POLICY)
                self.assertEqual(verdicts(report)[("isl=128", "tpot_ms")], pr.INVALID)
                self.assertIn("perturbed diagnostic", [e for e in report["entries"] if e["cell"] == "isl=128"][0]["note"])
                self.assertFalse(report["passed"])


class CompleteComparisonTests(unittest.TestCase):
    def test_a_long_context_token_regression_fails_with_an_unchanged_headline(self):
        slow_long = cell([t * 8 for t in TPOT], [t * 40 for t in TTFT])  # TPOT doubled at isl=8192
        report = pr.compare_receipts(receipt(), receipt(cells={"isl=8192": slow_long}), SPEC, pr.POLICY)

        got = verdicts(report)
        self.assertEqual(got[("isl=128", "tpot_ms")], pr.WITHIN_NOISE)
        self.assertEqual(got[("isl=128", "ttft_ms")], pr.WITHIN_NOISE)
        self.assertEqual(got[("isl=8192", "tpot_ms")], pr.REGRESSION)
        self.assertFalse(report["passed"])

    def test_a_long_prefill_ttft_regression_fails_with_an_unchanged_headline(self):
        slow_prefill = cell([t * 4 for t in TPOT], [t * 80 for t in TTFT])
        report = pr.compare_receipts(receipt(), receipt(cells={"isl=8192": slow_prefill}), SPEC, pr.POLICY)

        self.assertEqual(verdicts(report)[("isl=8192", "ttft_ms")], pr.REGRESSION)
        self.assertEqual(verdicts(report)[("isl=128", "ttft_ms")], pr.WITHIN_NOISE)
        self.assertFalse(report["passed"])

    def test_latency_is_worse_higher_and_throughput_worse_lower(self):
        spec = {"cells": {"c": ["tpot_ms", "decode_tok_s"]}}

        def one(tpot, tok_s):
            return {"identity": IDENTITY, "cells": {"c": {"repeats": [
                {"status": "ok", "metrics": {"tpot_ms": tpot + d, "decode_tok_s": tok_s + d * 10}} for d in (-1, -0.5, 0, 0.5, 1)
            ]}}}

        slower = pr.compare_receipts(one(10, 100), one(20, 50), spec, pr.POLICY)
        self.assertEqual(verdicts(slower), {("c", "tpot_ms"): pr.REGRESSION, ("c", "decode_tok_s"): pr.REGRESSION})
        faster = pr.compare_receipts(one(10, 100), one(5, 200), spec, pr.POLICY)
        self.assertEqual(verdicts(faster), {("c", "tpot_ms"): pr.IMPROVEMENT, ("c", "decode_tok_s"): pr.IMPROVEMENT})

    def test_a_missing_required_candidate_cell_is_incomplete_not_passed(self):
        candidate = receipt()
        del candidate["cells"]["isl=8192"]
        report = pr.compare_receipts(receipt(), candidate, SPEC, pr.POLICY)

        self.assertEqual(verdicts(report)[("isl=8192", "tpot_ms")], pr.NOT_RUN)
        self.assertEqual((report["status"], report["passed"]), (pr.INCOMPLETE, False))
        # The partial report stays inspectable: the cells that were measured still carry verdicts.
        self.assertEqual(verdicts(report)[("isl=128", "tpot_ms")], pr.WITHIN_NOISE)

    def test_a_refused_required_cell_is_incomplete_with_the_refusal(self):
        refused = {"outcome": "refused", "reason": "no accepted Rocm Decode route"}
        report = pr.compare_receipts(receipt(), receipt(cells={"isl=8192": refused}), SPEC, pr.POLICY)

        self.assertEqual(verdicts(report)[("isl=8192", "tpot_ms")], pr.REFUSED)
        self.assertIn("no accepted Rocm Decode route", report["entries"][2]["note"])
        self.assertEqual((report["status"], report["passed"]), (pr.INCOMPLETE, False))

    def test_a_failed_repeat_among_sufficient_successes_prevents_a_pass(self):
        flaky = cell(TPOT + [10.0], TTFT + [100.0])
        flaky["repeats"][5]["status"] = "failed"
        report = pr.compare_receipts(receipt(), receipt(cells={"isl=128": flaky}), SPEC, pr.POLICY)

        self.assertEqual(verdicts(report)[("isl=128", "tpot_ms")], pr.REPEAT_FAILED)
        self.assertFalse(report["passed"])

    def test_an_inconclusive_repeat_prevents_a_pass_like_a_failed_one(self):
        flaky = cell(TPOT + [10.0], TTFT + [100.0])
        flaky["repeats"][5]["status"] = "inconclusive"
        report = pr.compare_receipts(receipt(), receipt(cells={"isl=128": flaky}), SPEC, pr.POLICY)

        self.assertEqual(verdicts(report)[("isl=128", "tpot_ms")], pr.REPEAT_FAILED)
        self.assertFalse(report["passed"])

    def test_coverage_matrix_separates_measured_missing_refused_and_inconclusive(self):
        candidate = receipt(cells={"isl=8192": {"outcome": "refused", "reason": "x"}})
        flaky = cell(TPOT, TTFT)
        flaky["repeats"][0]["status"] = "failed"
        candidate["cells"]["isl=128"] = flaky
        spec = {"cells": {**SPEC["cells"], "isl=32768": ["tpot_ms"]}}

        matrix = pr.coverage_matrix(spec, candidate)

        self.assertEqual(matrix["missing"], [("isl=32768", "tpot_ms")])
        self.assertEqual(matrix["refused"], [("isl=8192", "tpot_ms"), ("isl=8192", "ttft_ms")])
        self.assertEqual(matrix["inconclusive"], [("isl=128", "tpot_ms"), ("isl=128", "ttft_ms")])
        self.assertEqual(matrix["measured"], [])


class UncertaintyTests(unittest.TestCase):
    A = [98.0, 99.0, 100.0, 101.0, 102.0]
    METRIC = "decode_tok_s"

    def verdict(self, b, **kwargs):
        return pr.metric_verdict(self.A, b, pr.POLICY, self.METRIC, **kwargs)

    def test_a_narrow_band_with_no_change_is_within_noise(self):
        self.assertEqual(self.verdict([99.0, 100.0, 100.5, 100.0, 99.5])["verdict"], pr.WITHIN_NOISE)

    def test_a_band_too_wide_to_exclude_a_material_regression_is_unverified(self):
        # Baseline spread 40 on a median of 100: the band swallows any practical loss.
        wide = [80.0, 90.0, 100.0, 110.0, 120.0]
        got = pr.metric_verdict(wide, [75.0] * 5, pr.POLICY, self.METRIC)
        self.assertEqual(got["verdict"], pr.UNVERIFIED)
        self.assertIn("exceeds the tolerated uncertainty", got["note"])

    def test_one_outlier_widens_the_band_instead_of_being_averaged_away(self):
        got = self.verdict([100.0, 100.0, 100.0, 100.0, 300.0])
        self.assertEqual(got["verdict"], pr.UNVERIFIED)

    def test_a_wide_band_that_still_clears_a_big_loss_is_a_regression(self):
        wide = [80.0, 90.0, 100.0, 110.0, 120.0]
        self.assertEqual(pr.metric_verdict(wide, [10.0] * 5, pr.POLICY, self.METRIC)["verdict"], pr.REGRESSION)

    def test_an_exhausted_repeat_budget_is_unverified_and_says_so(self):
        wide = [80.0, 90.0, 100.0, 110.0, 120.0] * 3
        got = pr.metric_verdict(wide, list(wide), pr.POLICY, self.METRIC)
        self.assertEqual(got["verdict"], pr.UNVERIFIED)
        self.assertTrue(got["budget_exhausted"])
        self.assertIn("budget is exhausted", got["note"])
        # With budget left the same width asks for more repeats instead.
        early = pr.metric_verdict(wide[:5], wide[:5], pr.POLICY, self.METRIC)
        self.assertFalse(early["budget_exhausted"])
        self.assertIn("more repeats may narrow it", early["note"])

    def test_too_few_repeats_is_unverified(self):
        self.assertEqual(pr.metric_verdict(self.A[:4], self.A, pr.POLICY, self.METRIC)["verdict"], pr.UNVERIFIED)

    def test_failed_repeats_are_never_dropped(self):
        got = pr.metric_verdict(self.A, self.A, pr.POLICY, self.METRIC, failed_b=1)
        self.assertEqual(got["verdict"], pr.REPEAT_FAILED)

    def test_insufficient_evidence_is_never_reported_as_within_noise(self):
        steady = [100.0] * 5
        for b in (steady, [99.99] * 5, [50.0] * 5):
            with self.subTest(b=b[0]):
                self.assertEqual(pr.metric_verdict(steady, b, pr.POLICY, self.METRIC)["verdict"], pr.UNVERIFIED)

    def test_non_finite_samples_are_invalid(self):
        for bad in (float("nan"), float("inf"), None):
            with self.subTest(bad=bad):
                self.assertEqual(pr.metric_verdict(self.A, self.A[:4] + [bad], pr.POLICY, self.METRIC)["verdict"], pr.INVALID)

    def test_a_practical_loss_that_the_band_cannot_rule_out_is_unverified(self):
        # A policy that tolerates this band's width but not the band plus the observed loss.
        metrics = {**pr.POLICY["metrics"], self.METRIC: {**pr.POLICY["metrics"][self.METRIC],
                                                          "max_uncertainty": 0.50, "practical_threshold": 0.10}}
        policy = {**pr.POLICY, "metrics": metrics}
        a = [97.0, 98.0, 100.0, 102.0, 103.0]
        b = [91.0, 92.0, 94.0, 96.0, 97.0]
        got = pr.metric_verdict(a, b, policy, self.METRIC)
        self.assertEqual(got["verdict"], pr.UNVERIFIED)
        self.assertIn("cannot be excluded", got["note"])

    def test_the_policy_comes_from_the_manifest_and_a_partial_one_is_refused(self):
        self.assertEqual(pr.POLICY, pr.load_policy(pr.MANIFEST))
        for metric in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms", "decode_tok_s", "prefill_tok_s", "step_us"):
            self.assertIn(metric, pr.POLICY["metrics"])
        self.assertEqual(pr.POLICY["metrics"]["tpot_ms"]["direction"], pr.LOWER_IS_BETTER)
        self.assertEqual(pr.POLICY["metrics"]["decode_tok_s"]["direction"], pr.HIGHER_IS_BETTER)
        with tempfile.TemporaryDirectory() as tmp:
            for name, text, fragment in (
                ("no-trials", '[performance]\nmax_trials = 3\n', "min_trials"),
                ("inverted", '[performance]\nmin_trials = 5\nmax_trials = 3\n', "max_trials is below min_trials"),
                ("bad-direction", '[performance]\nmin_trials = 1\nmax_trials = 3\nheadline_metric = "m"\n'
                 'curve_point_metrics = []\n[performance.metrics.m]\ndirection = "up"\n'
                 'practical_threshold = 0.2\nmax_uncertainty = 0.1\n', "direction must be"),
                ("no-tolerance", '[performance]\nmin_trials = 1\nmax_trials = 3\nheadline_metric = "m"\n'
                 'curve_point_metrics = []\n[performance.metrics.m]\ndirection = "lower"\n', "max_uncertainty"),
                ("unknown-headline", '[performance]\nmin_trials = 1\nmax_trials = 3\nheadline_metric = "x"\n'
                 'curve_point_metrics = []\n', "headline_metric names 'x'"),
            ):
                with self.subTest(name):
                    path = Path(tmp) / f"{name}.toml"
                    path.write_text(text)
                    with self.assertRaises(ValueError) as refused:
                        pr.load_policy(path)
                    self.assertIn(fragment, str(refused.exception))


class RequestWindowTests(unittest.TestCase):
    def request(self, stamps, complete, steps=None):
        """The boundaries a runner derives from a fake clock: token stamps and the completion time."""
        tokens = len(stamps)
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        return {
            "tokens": tokens,
            "first_token_ms": stamps[0] if stamps else None,
            "last_token_ms": stamps[-1] if stamps else None,
            "ttft_ms": stamps[0] if stamps else 0.0,
            "itl_ms": gaps,
            "finish_ms": complete - (stamps[-1] if stamps else 0.0),
            "e2e_ms": complete,
            "steps": tokens + 1 if steps is None else steps,
        }

    def test_a_consistent_window_has_no_errors(self):
        self.assertEqual(pr.request_window_errors(self.request([40.0, 50.0, 65.0], 72.0), 3), [])

    def test_delayed_cleanup_moves_finish_and_e2e_but_not_token_tpot(self):
        quick = self.request([40.0, 50.0, 65.0], 72.0)
        slow = self.request([40.0, 50.0, 65.0], 172.0)
        self.assertEqual(pr.request_window_errors(slow, 3), [])
        self.assertEqual(slow["finish_ms"] - quick["finish_ms"], 100.0)
        self.assertEqual(slow["e2e_ms"] - quick["e2e_ms"], 100.0)
        self.assertEqual(pr.request_tpot_ms(slow), pr.request_tpot_ms(quick))

    def test_a_delayed_token_changes_that_gap(self):
        base = self.request([40.0, 50.0, 65.0], 72.0)
        late = self.request([40.0, 50.0, 85.0], 92.0)
        self.assertEqual((base["itl_ms"], late["itl_ms"]), ([10.0, 15.0], [10.0, 35.0]))

    def test_e2e_is_ttft_plus_itls_plus_finish_from_unrounded_values(self):
        request = self.request([40.123456789, 50.987654321, 65.5], 72.25)
        self.assertEqual(pr.request_window_errors(request, 3), [])

    def test_completion_reported_as_last_token_delivery_is_caught(self):
        request = self.request([40.0, 50.0, 65.0], 172.0)
        request["last_token_ms"] = 172.0
        request["finish_ms"] = 0.0
        errors = pr.request_window_errors(request, 3)
        self.assertTrue(any("completion is not last-token delivery" in e for e in errors), errors)

    def test_an_omitted_first_gap_is_caught(self):
        request = self.request([40.0, 50.0, 65.0], 72.0)
        request["itl_ms"] = request["itl_ms"][1:]
        errors = pr.request_window_errors(request, 3)
        self.assertTrue(any("first gap may not be omitted" in e for e in errors), errors)

    def test_an_inconsistent_e2e_is_caught(self):
        request = self.request([40.0, 50.0, 65.0], 72.0)
        request["e2e_ms"] = 80.0
        errors = pr.request_window_errors(request, 3)
        self.assertTrue(any("must equal TTFT + sum(ITL) + finish" in e for e in errors), errors)

    def test_decode_hidden_before_the_request_is_caught_by_the_step_counter(self):
        # Three tokens need at least one prefill step plus two decode steps; two steps means one token's
        # computation ran in warmup, outside the timed window.
        request = self.request([40.0, 50.0, 65.0], 72.0, steps=2)
        errors = pr.request_window_errors(request, 3)
        self.assertTrue(any("hidden outside the request window" in e for e in errors), errors)

    def test_zero_token_request_is_all_finish(self):
        request = self.request([], 25.0)
        self.assertEqual(pr.request_window_errors(request, 0), [])
        self.assertEqual((request["ttft_ms"], request["finish_ms"], request["e2e_ms"]), (0.0, 25.0, 25.0))
        self.assertIsNone(pr.request_tpot_ms(request))
        request["finish_ms"] = 10.0
        self.assertTrue(pr.request_window_errors(request, 0))

    def test_one_token_request_has_ttft_and_finish_but_no_gap(self):
        request = self.request([10.0], 25.0)
        self.assertEqual(pr.request_window_errors(request, 1), [])
        self.assertEqual((request["ttft_ms"], request["finish_ms"], request["itl_ms"]), (10.0, 15.0, []))
        self.assertIsNone(pr.request_tpot_ms(request))

    def test_an_early_stop_is_defined_and_cannot_supply_exact_work_timing(self):
        request = self.request([40.0, 50.0], 55.0)
        errors = pr.request_window_errors(request, 3)
        self.assertEqual(errors, ["early stop: 2 of 3 requested tokens delivered"])
        self.assertEqual(pr.request_window_errors(request, 2), [])


def preparation(state=pr.COLD_POPULATED, cached=3, **layers):
    """A preparation receipt whose layers default to a coherent receipt for the state."""
    prepare = {
        pr.COLD_EMPTY: {"artifact_builds": 4, "artifact_hits": 0},
        pr.COLD_POPULATED: {"artifact_builds": 0, "artifact_hits": 4},
        pr.REUSED_DRIVER: {},
    }[state]
    cold = state != pr.REUSED_DRIVER
    base = {
        "plan": 2 if cold else 0,
        "entries": 2 if cold else 0,
        "generated_bodies": 4 if cold else 0,
        "artifact_builds": 0,
        "artifact_hits": 0,
        "runtime_pipelines": 4 if cold else 0,
        "recordings": 2 if cold else 0,
        "graph_instantiations": 2 if cold else 0,
        **prepare,
    }
    shaped = {}
    for layer, value in base.items():
        shaped[layer] = {"prepare": value, "repeat": 0}
    for layer, override in layers.items():
        shaped[layer] = {**shaped[layer], **override}
    return {"state": state, "artifact_entries_at_start": {pr.COLD_EMPTY: 0, pr.COLD_POPULATED: cached, pr.REUSED_DRIVER: cached}[state], "layers": shaped}


class PreparationTests(unittest.TestCase):
    def test_each_of_the_three_states_has_a_coherent_receipt(self):
        for state in pr.PREPARATION_STATES:
            with self.subTest(state=state):
                report = pr.preparation_report(preparation(state))
                self.assertEqual(report, {"errors": [], "unknown": [], "complete": True})

    def test_the_states_are_labelled_distinctly(self):
        self.assertEqual(len({p["state"] for p in map(preparation, pr.PREPARATION_STATES)}), 3)
        self.assertEqual(pr.preparation_report({"state": "warm-ish", "layers": {}})["complete"], False)

    def test_repeating_the_prepared_set_adds_no_work_at_any_layer(self):
        for layer in pr.PREPARATION_LAYERS:
            for state in pr.PREPARATION_STATES:
                with self.subTest(layer=layer, state=state):
                    report = pr.preparation_report(preparation(state, **{layer: {"repeat": 1}}))
                    self.assertTrue(any(f"added 1 {layer} work" in e for e in report["errors"]), report)

    def test_a_runtime_pipeline_built_in_the_window_is_not_hidden_by_a_quiet_plan_layer(self):
        report = pr.preparation_report(preparation(runtime_pipelines={"repeat": 3}))
        self.assertEqual(report["errors"], ["repeating the prepared set added 3 runtime_pipelines work: the layer is not warm"])

    def test_a_cold_claim_that_reused_artifacts_is_refused(self):
        silently_warm = preparation(pr.COLD_EMPTY)
        silently_warm["artifact_entries_at_start"] = 7
        report = pr.preparation_report(silently_warm)
        self.assertEqual(report["errors"], ["a cold-artifact claim started with 7 cached artifacts"])
        served = preparation(pr.COLD_EMPTY, artifact_hits={"prepare": 2})
        self.assertTrue(any("served 2 artifacts from cache" in e for e in pr.preparation_report(served)["errors"]))
        built_none = preparation(pr.COLD_EMPTY, artifact_builds={"prepare": 0})
        self.assertTrue(any("built no artifact" in e for e in pr.preparation_report(built_none)["errors"]))

    def test_a_populated_claim_must_serve_from_the_cache(self):
        rebuilt = preparation(pr.COLD_POPULATED, artifact_builds={"prepare": 1})
        self.assertTrue(any("still built 1 artifacts" in e for e in pr.preparation_report(rebuilt)["errors"]))
        empty = preparation(pr.COLD_POPULATED, cached=0)
        self.assertTrue(any("started with no cached artifacts" in e for e in pr.preparation_report(empty)["errors"]))

    def test_a_reused_driver_prepared_nothing(self):
        report = pr.preparation_report(preparation(pr.REUSED_DRIVER, plan={"prepare": 1}))
        self.assertEqual(report["errors"], ["a reused prepared driver did 1 plan work to prepare"])

    def test_a_layer_no_producer_counts_is_unknown_and_the_receipt_incomplete(self):
        receipt_ = preparation()
        receipt_["layers"]["artifact_hits"] = {"prepare": None, "repeat": None}
        receipt_["layers"]["graph_instantiations"] = {"prepare": None, "repeat": None}
        report = pr.preparation_report(receipt_)
        self.assertFalse(report["complete"])
        self.assertEqual(
            report["unknown"],
            ["artifact_hits.prepare", "artifact_hits.repeat", "graph_instantiations.prepare", "graph_instantiations.repeat"],
        )
        self.assertEqual(report["errors"], [], "Unknown is not an error and not a pass")

    def test_a_layer_the_receipt_leaves_out_is_malformed_not_unknown(self):
        receipt_ = preparation()
        del receipt_["layers"]["graph_instantiations"]
        report = pr.preparation_report(receipt_)
        self.assertFalse(report["complete"])
        self.assertEqual(report["errors"], ["preparation layer 'graph_instantiations' must be an object; got None"])


if __name__ == "__main__":
    unittest.main()
