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


def poot_row(**overrides):
    row = {
        "framework": "poot",
        "model": "m",
        "precision": "bf16",
        "scenario": "decode-128",
        "status": "ok",
        "build_sha": BUILD_SHA,
        "binary_sha256": "ab" * 32,
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

    def test_an_ok_poot_row_without_the_build_sha_is_rejected(self):
        self.assertEqual(self.errors(poot_row()), [])

        self.assertEqual(self.errors(without(poot_row(), "build_sha")), ["$ is missing 'build_sha'"])

    def test_an_ok_poot_row_names_its_binary_and_backend(self):
        for field in ("binary_sha256", "backend"):
            with self.subTest(field=field):
                self.assertEqual(self.errors(without(poot_row(), field)), [f"$ is missing {field!r}"])

    def test_a_poot_refusal_names_the_build_and_backend_that_refused(self):
        refusal = poot_row(status="unsupported", reason="no such architecture")

        self.assertEqual(self.errors(refusal), [])
        for field in ("build_sha", "backend"):
            with self.subTest(field=field):
                self.assertEqual(self.errors(without(refusal, field)), [f"$ is missing {field!r}"])

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

    def test_another_engine_reports_no_build_sha(self):
        row = {"framework": "llamacpp", "model": "m", "precision": "f16", "scenario": "s", "status": "ok"}

        self.assertEqual(self.errors(row), [])

    def test_the_build_sha_is_a_full_sha_and_the_binary_a_sha256(self):
        for field, bad in (("build_sha", "unknown"), ("build_sha", BUILD_SHA[:12]), ("binary_sha256", "ab" * 8)):
            with self.subTest(field=field, value=bad):
                self.assertEqual(self.errors(poot_row(**{field: bad})), [f"$.{field} does not match the schema pattern"])
        self.assertEqual(self.errors(poot_row(build_sha=BUILD_SHA + "-dirty")), [])

    def test_the_device_is_a_name_or_null(self):
        self.assertEqual(self.errors(poot_row(device=None)), [])
        self.assertEqual(self.errors(poot_row(device=3)), ["$.device is not a schema string or null"])


if __name__ == "__main__":
    unittest.main()
