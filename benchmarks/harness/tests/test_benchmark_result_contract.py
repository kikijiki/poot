import copy
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import observer  # noqa: E402
from runner_fixtures import (  # noqa: E402
    timed_request,
    OPTIONAL_OBSERVER_FIELDS,
    ROW_LEVEL_FRAMEWORKS,
    row_level_curve,
    row_level_single,
    curve_point,
    observation,
    run_fixture_runner,
    valid_curve,
    valid_single,
)
from schema_validator import schema_errors  # noqa: E402


class BenchmarkResultContractTests(unittest.TestCase):
    single_scenario = {"id": "single", "gen_tokens": 4}
    curve_scenario = {
        "id": "decode-curve",
        "kind": "decode-curve",
        "isl": [4, 8],
        "osl": 3,
        "synthetic": True,
    }
    schema = json.loads(
        (HARNESS_DIR.parent / "schema" / "result.schema.json").read_text()
    )

    def assert_schema_valid(self, row):
        self.assertEqual(schema_errors(row, self.schema), [])
        json.dumps(row, allow_nan=False)

    def run_payload(
        self, payload, *, scenario=None, result=None, stdout=None, support=True, framework="poot"
    ):
        scenario = scenario or self.single_scenario
        observed = (
            mock.patch.object(observer, "run_observed", return_value=result)
            if result is not None
            else mock.patch.object(
                observer, "run_observed", side_effect=run_fixture_runner
            )
        )
        with observed as run, mock.patch.dict(os.environ, {"POOT_BENCH_BACKEND": "ptx"}):
            row = bench.run_cell(
                model={"id": "fixture", "match_precision": "bf16"},
                scen=scenario,
                fw=framework,
                fw_cfg={
                    "support": support,
                    "precision": "bf16",
                    "reason": "fixture unsupported",
                    "caveat": "manifest caveat",
                },
                runners={
                    framework: {
                        "bin": sys.executable,
                        "cmd": [
                            "{bin}",
                            "-c",
                            (
                                "import sys; print(sys.argv[1]); "
                                "print('fixture stderr', file=sys.stderr)"
                            ),
                            stdout if stdout is not None else json.dumps(payload),
                        ],
                    }
                },
                models_dir=".",
                suite={},
                interval_ms=50,
            )
        return row, run

    def test_valid_single_result_is_accepted_with_stable_caveats(self):
        row, _ = self.run_payload(valid_single())

        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["framework"], "poot")
        self.assertEqual(row["precision"], "bf16")
        self.assertEqual(row["decode_tok_s"], 500.0)
        self.assertEqual(row["caveats"], ["manifest caveat", "runner caveat"])
        self.assert_schema_valid(row)

    def test_each_non_claiming_framework_is_accepted_on_row_level_timing(self):
        for framework in ROW_LEVEL_FRAMEWORKS:
            for name, payload, scenario in (
                ("single", row_level_single(framework), self.single_scenario),
                ("curve", row_level_curve(framework), self.curve_scenario),
            ):
                with self.subTest(framework=framework, shape=name):
                    row, _ = self.run_payload(payload, scenario=scenario, framework=framework)
                    self.assertEqual(row["status"], "ok", row.get("reason"))
                    self.assertEqual(row["timing_claim"], "row-level")
                    self.assert_schema_valid(row)

    def test_poot_claims_the_request_window_and_is_refused_without_it(self):
        row, _ = self.run_payload(valid_single())
        self.assertEqual((row["status"], row["timing_claim"]), ("ok", "request-window"))
        curve, _ = self.run_payload(valid_curve(), scenario=self.curve_scenario)
        self.assertEqual(curve["timing_claim"], "request-window")
        # The same row-level payloads a non-claiming framework passes are refused for poot.
        for payload, scenario, fragment in (
            (row_level_single("poot") | {"build_sha": valid_single()["build_sha"], "family": "qwen2",
                                          "backend": "ptx"}, self.single_scenario, "runs must be a non-empty array"),
            (row_level_curve("poot") | {"build_sha": valid_single()["build_sha"], "family": "qwen2",
                                         "backend": "ptx"}, self.curve_scenario, "missing required field"),
        ):
            with self.subTest(fragment=fragment):
                refused, _ = self.run_payload(payload, scenario=scenario)
                self.assertEqual(refused["status"], "error")
                self.assertIn(fragment, refused["reason"])

    def test_device_time_is_typed_and_only_a_device_timing_runner_may_report_it(self):
        measured = {"kind": "measured", "sum_of_dispatch_ms": 1.5, "device_span_ms": None}
        for name, value in (("null", None), ("unknown", {"kind": "unknown"}), ("measured", measured)):
            with self.subTest(accepted=name):
                single, _ = self.run_payload(valid_single() | {"device_time": value})
                self.assertEqual(single["status"], "ok", single.get("reason"))
                self.assert_schema_valid(single)
                payload = valid_curve()
                payload["curve"][0]["device_time"] = value
                curve, _ = self.run_payload(payload, scenario=self.curve_scenario)
                self.assertEqual(curve["status"], "ok", curve.get("reason"))
                self.assertEqual(curve["curve"][0]["device_time"], value)
                self.assert_schema_valid(curve)
        for name, value, fragment in (
            ("measured with no measure", {"kind": "measured", "sum_of_dispatch_ms": None, "device_span_ms": None},
             "neither measure"),
            ("measured with no keys", {"kind": "measured"}, "neither measure"),
            ("negative", {"kind": "measured", "sum_of_dispatch_ms": -1.0, "device_span_ms": 1.0}, "nonnegative"),
            ("unknown with a number", {"kind": "unknown", "sum_of_dispatch_ms": 0.0}, "unknown carries"),
            ("bad kind", {"kind": "zero"}, "kind must be"),
            ("not an object", 0.0, "null or an object"),
        ):
            with self.subTest(refused=name):
                single, _ = self.run_payload(valid_single() | {"device_time": value})
                self.assertEqual(single["status"], "error")
                self.assertIn(fragment, single["reason"])
                payload = valid_curve()
                payload["curve"][1]["device_time"] = value
                curve, _ = self.run_payload(payload, scenario=self.curve_scenario)
                self.assertEqual(curve["status"], "error")
                self.assertIn(f"curve[1].device_time", curve["reason"])
        # A runner with no device timestamps has no device time to report.
        payload = row_level_single("vllm") | {"device_time": {"kind": "unknown"}}
        refused, _ = self.run_payload(payload, framework="vllm")
        self.assertEqual(refused["status"], "error")
        self.assertIn("does not claim device timing", refused["reason"])

    def test_the_schema_declares_the_device_time_shape(self):
        row, _ = self.run_payload(valid_single() | {"device_time": {"kind": "unknown"}})
        self.assert_schema_valid(row)
        for bad in (
            {"kind": "measured"},
            {"kind": "zero"},
            {"kind": "measured", "sum_of_dispatch_ms": -1.0, "device_span_ms": 1.0},
        ):
            with self.subTest(bad=bad):
                self.assertNotEqual(schema_errors(row | {"device_time": bad}, self.schema), [])

    def test_a_row_level_curve_still_checks_its_own_gaps(self):
        payload = row_level_curve("vllm")
        payload["curve"][0]["iters"][0]["itl_ms"] = [2.0]
        row, _ = self.run_payload(payload, scenario=self.curve_scenario, framework="vllm")
        self.assertEqual(row["status"], "error")
        self.assertIn("must contain 2 samples", row["reason"])
        self.assertIn("vllm generated 2, expected 3 tokens", row["reason"])

    def test_invalid_single_results_are_error_rows(self):
        cases = (
            (
                "error key",
                lambda row: row.update(error="runner exploded"),
                "runner reported error",
            ),
            (
                "error status",
                lambda row: row.update(status="error", reason="reported failure"),
                "runner reported status",
            ),
            (
                "wrong framework",
                lambda row: row.update(framework="candle"),
                "framework mismatch",
            ),
            (
                "wrong precision",
                lambda row: row.update(precision="f16"),
                "precision mismatch",
            ),
            (
                "missing timing",
                lambda row: row.pop("ttft_ms"),
                "missing required field 'ttft_ms'",
            ),
            (
                "nan timing",
                lambda row: row.update(ttft_ms=float("nan")),
                "finite nonnegative",
            ),
            (
                "infinite timing",
                lambda row: row.update(decode_tok_s=float("inf")),
                "finite nonnegative",
            ),
            (
                "zero iterations",
                lambda row: row.update(iterations=0),
                "iterations must be a positive",
            ),
            (
                "boolean count",
                lambda row: row.update(iterations=True),
                "iterations must be a positive",
            ),
            (
                "wrong token count",
                lambda row: row.update(gen_tokens=3),
                "gen_tokens mismatch",
            ),
            (
                "negative timing",
                lambda row: row.update(tpot_ms=-1.0),
                "finite nonnegative",
            ),
            (
                "reversed timing",
                lambda row: row.update(e2e_ms=0.5),
                "greater than or equal",
            ),
            (
                "request window inconsistent",
                lambda row: row["runs"][0].update(e2e_ms=0.5),
                "runs[0].e2e_ms",
            ),
            (
                "no per-request runs",
                lambda row: row.pop("runs"),
                "runs must be a non-empty array",
            ),
        )
        for name, mutate, expected in cases:
            with self.subTest(name=name):
                payload = valid_single()
                mutate(payload)
                row, _ = self.run_payload(payload)
                self.assertEqual(row["status"], "error")
                self.assertIn("invalid runner result", row["reason"])
                self.assertIn(expected, row["reason"])
                self.assertNotIn("decode_tok_s", row)
                self.assertEqual(row["peak_vram_bytes"], 101)
                self.assertEqual(row["caveats"], ["manifest caveat", "runner caveat"])
                self.assertIn("fixture stderr", row["reason"])

    def test_final_error_objects_are_authoritative_and_keep_diagnostics(self):
        valid = json.dumps(valid_single())
        cases = (
            (
                "error only",
                '{"error":"runner exploded"}',
                "runner error: runner exploded",
            ),
            (
                "status only",
                '{"status":"error","reason":"final runner failure"}',
                "runner reason: final runner failure",
            ),
            (
                "valid then error",
                valid + '\n{"status":"error","reason":"final runner failure"}',
                "runner reason: final runner failure",
            ),
        )
        for name, stdout, expected in cases:
            with self.subTest(name=name):
                row, _ = self.run_payload(
                    {}, stdout=stdout + " shutdown complete\nlater shutdown"
                )
                self.assertEqual(row["status"], "error")
                self.assertIn("invalid runner result", row["reason"])
                self.assertIn(expected, row["reason"])
                self.assertNotIn("decode_tok_s", row)

    def test_final_top_level_non_objects_reach_contract_and_are_rejected(self):
        valid = json.dumps(valid_single())
        cases = (
            ("array containing valid object", json.dumps([valid_single()]), "list"),
            ("number", "42", "int"),
            ("string", json.dumps("runner result"), "str"),
            ("boolean", "true", "bool"),
            ("null", "null", "NoneType"),
            ("valid then final null", valid + "\nnull", "NoneType"),
            (
                "valid then final array on same line",
                valid + json.dumps([valid_single()]),
                "list",
            ),
        )
        for name, stdout, reported_type in cases:
            with self.subTest(name=name):
                row, _ = self.run_payload({}, stdout=stdout)
                self.assertEqual(row["status"], "error")
                self.assertIn(
                    f"runner result must be an object; got {reported_type}",
                    row["reason"],
                )
                self.assertNotIn("no result JSON", row["reason"])
                self.assertNotIn("decode_tok_s", row)

    def test_noisy_json_stream_preserves_top_level_boundaries_and_order(self):
        nested = valid_single()
        nested["ignored_metadata"] = {
            "nested": {"message": "braces stay in string: { not JSON }"}
        }
        valid = json.dumps(nested)
        error = '{"status":"error","reason":"later same-line error"}'
        accepted = (
            (
                "nested objects and braces in strings",
                valid + " shutdown complete pid=42",
            ),
            (
                "malformed prefix and shutdown suffix",
                "runner {{malformed prefix " + valid + " shutdown complete pid=42",
            ),
            (
                "later valid same-line document",
                error + valid + " trailing non-JSON shutdown pid=42",
            ),
        )
        for name, stdout in accepted:
            with self.subTest(name=name):
                row, _ = self.run_payload({}, stdout=stdout)
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["decode_tok_s"], 500.0)

        row, _ = self.run_payload(
            {}, stdout=valid + error + " trailing non-JSON shutdown pid=42"
        )
        self.assertEqual(row["status"], "error")
        self.assertIn("runner reason: later same-line error", row["reason"])

    def test_invalid_curves_are_error_rows(self):
        def mutate_partial(row):
            row["curve"] = row["curve"][:1]

        def mutate_duplicate(row):
            row["curve"] = [row["curve"][0], copy.deepcopy(row["curve"][0])]

        def mutate_extra(row):
            row["curve"].append(curve_point(16))

        def mutate_overflowing_throughput(row):
            for iteration in row["curve"][0]["iters"]:
                iteration.update(timed_request([0.0, 1e-323, 2e-323], 2e-323))

        cases = (
            ("empty", lambda row: row.update(curve=[]), "non-empty array"),
            ("partial", mutate_partial, "coverage mismatch"),
            ("duplicate", mutate_duplicate, "coverage mismatch"),
            ("extra", mutate_extra, "coverage mismatch"),
            (
                "wrong prompt count",
                lambda row: row["curve"][0].update(prompt_tokens=3),
                "prompt_tokens must equal",
            ),
            (
                "empty iterations",
                lambda row: row["curve"][0].update(iters=[]),
                "non-empty array",
            ),
            (
                "short ITL samples",
                lambda row: row["curve"][0]["iters"][0].update(itl_ms=[2.0]),
                "first gap may not be omitted",
            ),
            (
                "nan sample",
                lambda row: row["curve"][0]["iters"][0].update(
                    itl_ms=[float("nan"), 2.0]
                ),
                "finite nonnegative",
            ),
            (
                "missing iteration timing",
                lambda row: row["curve"][0]["iters"][0].pop("e2e_ms"),
                "missing required field",
            ),
            (
                "reversed iteration timing",
                lambda row: row["curve"][0]["iters"][0].update(e2e_ms=1.0),
                "must equal TTFT + sum(ITL) + finish",
            ),
            (
                "zero TPOT",
                lambda row: row["curve"][0]["iters"][0].update(
                    timed_request([4.0, 4.0, 4.0], 10.0)
                ),
                "TPOT must be a finite positive",
            ),
            (
                "early stop",
                lambda row: row["curve"][0]["iters"][0].update(
                    timed_request([4.0, 6.0], 8.0)
                ),
                "early stop: 2 of 3 requested tokens",
            ),
            (
                "token computed outside the window",
                lambda row: row["curve"][0]["iters"][0].update(steps=2),
                "hidden outside the request window",
            ),
            (
                "completion as last token",
                lambda row: row["curve"][0]["iters"][0].update(last_token_ms=10.0, finish_ms=0.0),
                "completion is not last-token delivery",
            ),
            (
                "overflowing derived throughput",
                mutate_overflowing_throughput,
                "decode_tok_s must be a finite positive",
            ),
            ("wrong OSL", lambda row: row.update(osl=4), "osl mismatch"),
            ("wrong mode", lambda row: row.update(mode="single"), "mode mismatch"),
        )
        for name, mutate, expected in cases:
            with self.subTest(name=name):
                payload = valid_curve()
                mutate(payload)
                row, _ = self.run_payload(payload, scenario=self.curve_scenario)
                self.assertEqual(row["status"], "error")
                self.assertIn(expected, row["reason"])
                self.assertNotIn("curve", row)

    def test_complete_curve_is_accepted_and_aggregated(self):
        row, _ = self.run_payload(valid_curve(), scenario=self.curve_scenario)

        self.assertEqual(row["status"], "ok")
        self.assertEqual([point["isl"] for point in row["curve"]], [4, 8])
        self.assertEqual([point["iters"] for point in row["curve"]], [2, 2])
        self.assertEqual(row["isl"], 4)
        self.assertGreater(row["decode_tok_s"], 0)
        self.assertTrue(all(point["tpot_ms"]["p50"] > 0 for point in row["curve"]))
        self.assert_schema_valid(row)

    def test_a_summary_over_few_iterations_is_the_median_and_range_with_no_tail_percentile(self):
        def curve_of(iterations):
            payload = valid_curve()
            for point in payload["curve"]:
                point["iters"] = [
                    timed_request([point["isl"] + k * 0.1, point["isl"] + 2.0 + k * 0.1, point["isl"] + 4.0 + k * 0.1],
                                  point["isl"] + 6.0 + k * 0.1)
                    for k in range(iterations)
                ]
            return payload

        few, _ = self.run_payload(curve_of(5), scenario=self.curve_scenario)
        many, _ = self.run_payload(curve_of(20), scenario=self.curve_scenario)

        for field in ("ttft_ms", "tpot_ms"):
            self.assertEqual(sorted(few["curve"][0][field]), ["max", "min", "p50"], field)
            self.assertEqual(sorted(many["curve"][0][field]), ["max", "min", "p50", "p90", "p99"], field)
        self.assertLess(few["curve"][0]["ttft_ms"]["min"], few["curve"][0]["ttft_ms"]["max"])
        # The ITL stream is token-weighted: five iterations of two gaps are ten samples, still under the threshold.
        self.assertEqual(sorted(few["curve"][0]["itl_ms"]), ["max", "min", "p50"])
        self.assert_schema_valid(few)
        self.assert_schema_valid(many)

    def test_an_external_tool_result_names_the_version_it_measured(self):
        payload = row_level_single("vllm")
        del payload["framework_version"]

        row, _ = self.run_payload(payload, framework="vllm")

        self.assertEqual(row["status"], "error")
        self.assertIn("framework_version", row["reason"])

    def test_accepted_shapes_are_schema_valid_with_observer_metrics_unavailable(self):
        cases = (
            ("single", valid_single(), self.single_scenario),
            ("curve", valid_curve(), self.curve_scenario),
        )
        for name, payload, scenario in cases:
            with self.subTest(shape=name):
                result = observation(payload, observer_metrics=False)
                row, _ = self.run_payload(payload, scenario=scenario, result=result)
                self.assertEqual(row["status"], "ok")
                for field in OPTIONAL_OBSERVER_FIELDS:
                    self.assertNotIn(field, row)
                self.assertEqual(row["sampler_interval_ms"], 50.0)
                self.assert_schema_valid(row)

    def test_representative_populated_observer_metrics_are_schema_valid(self):
        cases = (
            ("single", valid_single(), self.single_scenario),
            ("curve", valid_curve(), self.curve_scenario),
        )
        for name, payload, scenario in cases:
            with self.subTest(shape=name):
                row, _ = self.run_payload(payload, scenario=scenario)
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["peak_vram_bytes"], 101)
                self.assertEqual(row["peak_rss_bytes"], 202)
                self.assert_schema_valid(row)

    def test_process_error_preserves_runner_and_observer_diagnostics(self):
        payload = valid_single()
        payload.update(
            error="subprocess detail", caveats=["runner caveat", "manifest caveat"]
        )
        row, _ = self.run_payload(
            payload,
            result=observation(payload, returncode=7, stderr="process stderr"),
        )

        self.assertEqual(row["status"], "error")
        self.assertTrue(row["reason"].startswith("rc=7"))
        self.assertIn("subprocess detail", row["reason"])
        self.assertIn("process stderr", row["reason"])
        self.assertEqual(row["peak_rss_bytes"], 202)
        self.assertEqual(row["caveats"], ["manifest caveat", "runner caveat"])
        self.assert_schema_valid(row)

    def test_error_row_omits_unavailable_observer_metrics(self):
        payload = valid_single()
        result = observation(
            payload,
            returncode=7,
            stderr="process stderr",
            observer_metrics=False,
        )
        row, _ = self.run_payload(payload, result=result)

        self.assertEqual(row["status"], "error")
        for field in OPTIONAL_OBSERVER_FIELDS:
            self.assertNotIn(field, row)
        self.assertEqual(row["sampler_interval_ms"], 50.0)
        self.assert_schema_valid(row)

    def test_manifest_unsupported_row_bypasses_runner_and_keeps_diagnostics(self):
        row, run = self.run_payload(valid_single(), support=False, framework="llamacpp")

        run.assert_not_called()
        self.assertEqual(row["framework"], "llamacpp")
        self.assertEqual(row["status"], "unsupported")
        self.assertEqual(row["reason"], "fixture unsupported")
        self.assertEqual(row["caveats"], ["manifest caveat"])
        self.assert_schema_valid(row)

    def test_contract_rejects_non_object_payload(self):
        with self.assertRaisesRegex(bench.ResultValidationError, "must be an object"):
            bench.validate_runner_result(
                [],
                framework="poot",
                precision="bf16",
                scenario=self.single_scenario,
                gen_tokens=4,
            )


if __name__ == "__main__":
    unittest.main()
