"""The half of provenance the harness cannot see: which sha build.rs embeds in the runner binary.

build.rs is compiled with rustc once and run as cargo would run it (CARGO_MANIFEST_DIR set) against
temporary git checkouts and against a copy with no .git, so the rules are tested on the real file.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

BUILD_RS = Path(__file__).resolve().parents[2] / "runners" / "poot" / "build.rs"
MANIFEST_SUBDIR = Path("benchmarks") / "runners" / "poot"
STALE = "5" * 40


def run(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, check=True, **kwargs)


class BuildShaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.script = Path(cls.work.name) / "build-script"
        try:
            run(["rustc", "--edition", "2021", str(BUILD_RS), "-o", str(cls.script)])
        except FileNotFoundError:
            raise AssertionError("rustc is required to test build.rs (run inside `nix develop`)")

    def make_checkout(self):
        """A committed git checkout with the directories build.rs watches, and its HEAD."""
        root = Path(tempfile.mkdtemp(dir=self.work.name))
        (root / MANIFEST_SUBDIR).mkdir(parents=True)
        (root / "crates").mkdir()
        (root / "crates" / "lib.rs").write_text("// v1\n")
        (root / "Cargo.toml").write_text("[workspace]\n")
        git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
        run([*git, "init", "-q"])
        run([*git, "add", "."])
        run([*git, "commit", "-q", "-m", "first"])
        return root, run([*git, "rev-parse", "HEAD"]).stdout.strip()

    def build(self, root, exported=None):
        env = {k: v for k, v in os.environ.items() if k != "POOT_BUILD_SHA"}
        env["CARGO_MANIFEST_DIR"] = str(root / MANIFEST_SUBDIR)
        if exported is not None:
            env["POOT_BUILD_SHA"] = exported
        done = subprocess.run(
            [str(self.script)], capture_output=True, text=True, check=False, env=env
        )
        prefix = "cargo:rustc-env=POOT_BUILD_SHA="
        embedded = [line[len(prefix):] for line in done.stdout.splitlines() if line.startswith(prefix)]
        return done, (embedded[0] if embedded else None)

    def export(self, root):
        """The same sources with no .git, as a source export would be."""
        copy = Path(tempfile.mkdtemp(dir=self.work.name))
        shutil.copytree(root, copy, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
        return copy

    def test_checkout_embeds_its_head(self):
        root, head = self.make_checkout()

        done, embedded = self.build(root)

        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(embedded, head)

    def test_tracked_edit_to_a_build_input_marks_the_sha_dirty(self):
        root, head = self.make_checkout()
        (root / "crates" / "lib.rs").write_text("// v2\n")

        _, embedded = self.build(root)

        self.assertEqual(embedded, head + "-dirty")

    def test_git_wins_over_a_stale_variable_in_a_checkout(self):
        root, head = self.make_checkout()
        self.assertNotEqual(STALE, head)

        done, embedded = self.build(root, exported=STALE)

        # Never a binary that claims the stale sha: the build refuses.
        self.assertNotEqual(done.returncode, 0)
        self.assertIsNone(embedded)
        self.assertIn("disagrees with the checkout", done.stderr)

    def test_variable_agreeing_with_the_checkout_is_accepted(self):
        root, head = self.make_checkout()

        done, embedded = self.build(root, exported=head)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(embedded, head)
        # A clean checkout contradicts a variable that claims a dirty build.
        done, _ = self.build(root, exported=head + "-dirty")
        self.assertNotEqual(done.returncode, 0)

        # In a dirty checkout the variable may name HEAD or HEAD-dirty; the binary says dirty either way.
        (root / "crates" / "lib.rs").write_text("// v2\n")
        for exported in (head, head + "-dirty"):
            with self.subTest(exported=exported):
                done, embedded = self.build(root, exported=exported)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual(embedded, head + "-dirty")

    def test_export_with_no_git_embeds_the_variable(self):
        root, head = self.make_checkout()
        export = self.export(root)

        done, embedded = self.build(export, exported=head)

        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(embedded, head)

    def test_export_with_no_variable_fails(self):
        root, _ = self.make_checkout()

        done, embedded = self.build(self.export(root))

        self.assertNotEqual(done.returncode, 0)
        self.assertIsNone(embedded)
        self.assertIn("POOT_BUILD_SHA is not set", done.stderr)

    def test_malformed_variable_fails_in_a_checkout_and_in_an_export(self):
        root, head = self.make_checkout()
        export = self.export(root)
        malformed = ("stale-not-a-sha", head[:12], head.upper(), head + "-wip", head + " x")

        for where, source in (("checkout", root), ("export", export)):
            for value in malformed:
                with self.subTest(where=where, value=value):
                    done, embedded = self.build(source, exported=value)
                    self.assertNotEqual(done.returncode, 0)
                    self.assertIsNone(embedded)
                    self.assertIn("40 lowercase hex", done.stderr)


if __name__ == "__main__":
    unittest.main()
