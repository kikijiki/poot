"""Where the result sets are read from: `POOT_BENCH_RESULTS`, else the archive beside the repository, and a missing baseline fails
with the lookups it tried instead of reading nothing."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import results_root  # noqa: E402


class CommittedRootTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.parent = Path(self.tmp.name)
        self.repo = self.parent / "poot"
        self.repo.mkdir()
        env = {key: value for key, value in os.environ.items() if key != results_root.ENV}
        patches = [mock.patch.dict(os.environ, env, clear=True), mock.patch.object(results_root, "REPO_ROOT", self.repo)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_an_archive_checkout_beside_the_repository_is_found(self):
        archive = self.parent / "notes" / "projects" / "poot" / "benchmark-results"
        archive.mkdir(parents=True)

        self.assertEqual(results_root.committed_root(), archive)

    def test_the_setting_overrides_the_archive(self):
        (self.parent / "notes" / "projects" / "poot" / "benchmark-results").mkdir(parents=True)
        chosen = self.parent / "elsewhere"
        chosen.mkdir()
        os.environ[results_root.ENV] = str(chosen)

        self.assertEqual(results_root.committed_root(), chosen)

    def test_a_missing_archive_fails_with_the_lookup_it_tried(self):
        with self.assertRaises(results_root.ResultsRootMissing) as raised:
            results_root.committed_root()

        message = str(raised.exception)
        self.assertIn(results_root.ENV, message)
        self.assertIn(f"{self.parent}/{results_root.ARCHIVE_GLOB}", message)

    def test_a_setting_naming_no_directory_fails_with_the_path(self):
        os.environ[results_root.ENV] = str(self.parent / "absent")

        with self.assertRaisesRegex(results_root.ResultsRootMissing, "absent"):
            results_root.committed_root()

    def test_a_recording_run_writes_where_the_setting_says_else_into_the_checkout(self):
        self.assertEqual(results_root.recording_root(), results_root.BENCH_ROOT / "results")
        os.environ[results_root.ENV] = str(self.parent / "out")

        self.assertEqual(results_root.recording_root(), self.parent / "out")


if __name__ == "__main__":
    unittest.main()
