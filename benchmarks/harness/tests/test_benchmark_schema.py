import json
import sys
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

from schema_validator import schema_errors  # noqa: E402

SCHEMA = json.loads((HARNESS_DIR.parent / "schema" / "result.schema.json").read_text())
FIXTURE_ROWS = HARNESS_DIR / "fixtures" / "run-a" / "results.jsonl"

BUILD_SHA = "0123456789abcdef0123456789abcdef01234567"
BUILD_IDENTITY = ("build_sha", "build_profile", "build_features", "rustc_version")
MEASUREMENT_IDENTITY = ("adapter", "driver_version", "checkpoint_digest", "compute_dtype")


def poot_row(**overrides):
    row = {
        "framework": "poot",
        "model": "m",
        "precision": "bf16",
        "scenario": "decode-128",
        "status": "ok",
        "build_sha": BUILD_SHA,
        "build_profile": "release",
        "build_features": "",
        "rustc_version": "rustc 1.101.0-nightly (fixture)",
        "adapter": "AMD Radeon 8060S Graphics",
        "driver_version": "radv Mesa 26.1.3",
        "checkpoint_digest": "blake3:" + "cd" * 32,
        "compute_dtype": "f32",
        "binary_sha256": "ab" * 32,
        "family": "qwen2",
        "backend": "rocm",
        "device": "AMD Radeon 8060S Graphics",
    }
    row.update(overrides)
    return row


def without(row, field):
    return {key: value for key, value in row.items() if key != field}


class ResultSchemaTests(unittest.TestCase):
    def errors(self, row):
        return schema_errors(row, SCHEMA)

    def test_the_committed_fixture_rows_are_valid(self):
        rows = [json.loads(line) for line in FIXTURE_ROWS.read_text().splitlines() if line.strip()]

        self.assertGreater(len(rows), 1)
        for row in rows:
            with self.subTest(framework=row["framework"], status=row["status"]):
                self.assertEqual(self.errors(row), [])

    def test_an_ok_poot_row_states_its_build_and_what_it_ran_on(self):
        self.assertEqual(self.errors(poot_row()), [])

        for field in (*BUILD_IDENTITY, *MEASUREMENT_IDENTITY, "binary_sha256", "backend", "family"):
            with self.subTest(field=field):
                self.assertEqual(self.errors(without(poot_row(), field)), [f"$ is missing {field!r}"])

    def test_only_a_release_build_is_a_poot_result(self):
        self.assertEqual(self.errors(poot_row(build_profile="debug")), ["$.build_profile is not in the schema enum"])

    def test_a_poot_refusal_names_the_build_backend_and_family_that_refused(self):
        refusal = poot_row(status="unsupported", reason="no such architecture")

        self.assertEqual(self.errors(refusal), [])
        for field in (*BUILD_IDENTITY, "backend", "family"):
            with self.subTest(field=field):
                self.assertEqual(self.errors(without(refusal, field)), [f"$ is missing {field!r}"])

    def test_the_family_is_null_only_on_a_refusal_that_came_before_one_resolved(self):
        before_resolution = poot_row(status="unsupported", reason="no family is registered", family=None)

        self.assertEqual(self.errors(before_resolution), [])
        self.assertEqual(self.errors(poot_row(family=None)), ["$.family is not a schema string"])

    def test_a_binary_that_names_no_file_has_no_digest(self):
        self.assertEqual(self.errors(poot_row(binary_sha256=None)), [])

    def test_an_unsupported_row_without_a_reason_is_rejected(self):
        row = {"framework": "vllm", "model": "m", "precision": "bf16", "scenario": "s", "status": "unsupported"}

        self.assertEqual(self.errors(row), ["$ is missing 'reason'"])
        self.assertEqual(self.errors({**row, "reason": "no such kernel"}), [])

    def test_an_error_row_carries_its_reason_and_needs_no_build_sha(self):
        row = poot_row(status="error", reason="rc=3: runner crashed")

        self.assertEqual(self.errors(without(without(row, "build_sha"), "backend")), [])
        self.assertEqual(self.errors(without(row, "reason")), ["$ is missing 'reason'"])

    def test_another_engine_has_no_build_but_names_its_version(self):
        row = {"framework": "llamacpp", "model": "m", "precision": "f16", "scenario": "s", "status": "ok"}

        self.assertEqual(self.errors(row), ["$ is missing 'framework_version'"])
        self.assertEqual(self.errors({**row, "framework_version": "4c65955"}), [])

    def test_the_build_sha_is_a_full_sha_and_the_binary_a_sha256(self):
        for field, bad in (("build_sha", "unknown"), ("build_sha", BUILD_SHA[:12]), ("binary_sha256", "ab" * 8)):
            with self.subTest(field=field, value=bad):
                self.assertEqual(self.errors(poot_row(**{field: bad})), [f"$.{field} does not match the schema pattern"])
        self.assertEqual(self.errors(poot_row(build_sha=BUILD_SHA + "-dirty")), [])

    def test_the_checkpoint_digest_names_its_algorithm(self):
        self.assertEqual(
            self.errors(poot_row(checkpoint_digest="cd" * 32)), ["$.checkpoint_digest does not match the schema pattern"]
        )

    def test_a_curve_summary_is_the_median_and_range_with_tail_percentiles_optional(self):
        point = {"p50": 1.0, "min": 0.5, "max": 2.0}
        curve = {"isl": 128, "osl": 8, "prompt_tokens": 128, "iters": 5, "ttft_ms": point, "tpot_ms": point,
                 "itl_ms": point, "decode_tok_s": 10.0}

        self.assertEqual(self.errors(poot_row(curve=[curve])), [])
        self.assertEqual(self.errors(poot_row(curve=[{**curve, "ttft_ms": {"p50": 1.0}}])), [
            "$.curve[0].ttft_ms is missing 'min'", "$.curve[0].ttft_ms is missing 'max'"])

    def test_the_device_is_a_name_or_null(self):
        self.assertEqual(self.errors(poot_row(device=None)), [])
        self.assertEqual(self.errors(poot_row(device=3)), ["$.device is not a schema string or null"])


if __name__ == "__main__":
    unittest.main()
