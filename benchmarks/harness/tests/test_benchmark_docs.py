"""What the recipes, the README and the devshell banner name exists, and the index matches the results."""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import known_unverified  # noqa: E402

BENCH_ROOT = bench.BENCH_ROOT
REPO_ROOT = BENCH_ROOT.parent
# The recipes, the devshell banner, every markdown doc under benchmarks/ except the generated result reports, and
# the orchestrator README, which drives the same image recipes.
_SKIPPED_DIRS = {"results", "target", ".venv", "node_modules"}
DOCS = [
    BENCH_ROOT / "justfile",
    BENCH_ROOT / "flake.nix",
    REPO_ROOT / "crates" / "poot-orchestrator" / "README.md",
    *sorted(
        doc
        for doc in BENCH_ROOT.glob("**/*.md")
        if not _SKIPPED_DIRS & set(doc.relative_to(BENCH_ROOT).parts)
    ),
]

# A path under one of the benchmark directories, or a file name with a known extension, that is not part of
# a longer path, a placeholder or a URL. Pod-side absolute paths (/root/run-poot.sh) are not repo files.
_DIR_PATH = re.compile(
    r"(?<![\w/.<{-])((?:docker|harness|setup|runners|prompts|schema|results|\.github)/[\w][\w./-]*[\w])(?![\w<{*>-])"
)
_FILE_NAME = re.compile(r"(?<![\w/.<{@-])([\w][\w-]*\.(?:yml|yaml|sh|py|toml|nix))(?![\w<{*>-])")
_URL = re.compile(r"https?://\S+")
# A recipe is named as `just <recipe>` in backticks, or after a colon ("build: just image-build"). Prose such as
# "it just means" is neither.
_JUST_RECIPE = re.compile(r"(?:`|: )just ([a-z][\w-]*)")


def references(text):
    """The file paths `text` names, minus placeholders and URLs."""
    text = _URL.sub("", text)
    found = set(_DIR_PATH.findall(text)) | set(_FILE_NAME.findall(text))
    return sorted(
        ref for ref in found if not any(mark in ref for mark in "*<>{}") and not ref.endswith(("/", "."))
    )


# Scripts of other projects that the docs name, which the repository does not hold.
_EXTERNAL = {"convert_hf_to_gguf.py"}


def exists(ref, doc=None):
    """A reference resolves against the benchmarks directory, its harness and setup scripts, the doc's own
    directory (a runner's NOTES.md names its neighbours), or the repository root."""
    roots = [BENCH_ROOT, BENCH_ROOT / "harness", BENCH_ROOT / "setup", REPO_ROOT]
    if doc is not None:
        roots.append(doc.parent)
    return ref in _EXTERNAL or any((root / ref).exists() for root in roots)


def recipes(justfile_text):
    """A recipe line is a name, its parameters (which may have defaults) and a colon; `name := value` is a variable."""
    return set(re.findall(r"^([a-z][\w-]*)(?:[ \t]+[^:\n]*)?:(?!=)", justfile_text, flags=re.MULTILINE))


class NamedFilesExistTests(unittest.TestCase):
    def test_every_file_a_recipe_the_readme_or_the_banner_names_exists(self):
        for doc in DOCS:
            for ref in references(doc.read_text()):
                with self.subTest(doc=doc.name, ref=ref):
                    self.assertTrue(exists(ref, doc), f"{doc.relative_to(REPO_ROOT)} names {ref}, which does not exist")

    def test_every_recipe_the_docs_name_exists(self):
        # The image and sweep recipes live in the benchmarks justfile; the bench recipes the README documents
        # (`bench-record`, `bench-compare`, `test-bench-harness`) live in the repository's.
        known = recipes((BENCH_ROOT / "justfile").read_text()) | recipes((REPO_ROOT / "justfile").read_text())
        self.assertIn("harness-test", known)
        self.assertIn("bench-compare", known)
        for doc in DOCS:
            for name in sorted(set(_JUST_RECIPE.findall(doc.read_text()))):
                with self.subTest(doc=doc.name, recipe=name):
                    self.assertIn(name, known, f"{doc.relative_to(REPO_ROOT)} names `just {name}`, which is not a recipe")

    def test_the_reference_scan_finds_a_stale_name(self):
        # The scan must be able to fail: it sees the names the docs used to get wrong.
        stale = "e.g. `just sweep-image --dockerfile docker/Dockerfile.slim`; gh workflow run bench-image.yml"

        self.assertEqual(references(stale), ["bench-image.yml", "docker/Dockerfile.slim"])
        self.assertFalse(exists("docker/Dockerfile.slim"))
        self.assertFalse(exists("bench-image.yml"))
        self.assertTrue(exists("docker/Dockerfile"))
        self.assertNotIn("image-ci", recipes((BENCH_ROOT / "justfile").read_text()))


class IndexTests(unittest.TestCase):
    def test_the_committed_index_is_the_index_the_results_generate(self):
        results = BENCH_ROOT / "results"
        regenerated = bench.render_index(bench.load_runs(results), known_unverified.load(known_unverified.beside(results)))

        self.assertEqual(
            (results / "INDEX.md").read_text(),
            regenerated,
            "results/INDEX.md is stale: run `python3 harness/bench.py index`",
        )

    def test_the_index_lists_every_run_with_an_ok_row(self):
        results = BENCH_ROOT / "results"
        index = (results / "INDEX.md").read_text()
        for run_id, _env, rows in bench.load_runs(results):
            if any(row.get("status") == "ok" for row in rows):
                with self.subTest(run=run_id):
                    self.assertIn(f"| {run_id} |", index)


class PartialSetTests(unittest.TestCase):
    """A `<set>.partial` directory is the orchestrator's mirror of a sweep in flight, never a result."""

    ROW = {"model": "m", "scenario": "decode-curve", "framework": "poot", "backend": "ptx", "status": "ok"}

    def results_with_a_partial(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for name in ("2026-10-22-gpu-m-000000-r1", "2026-10-22-gpu-m-000000-r2.partial"):
            d = Path(tmp.name) / name
            d.mkdir()
            (d / "results.jsonl").write_text(json.dumps(self.ROW) + "\n")
            (d / "env.json").write_text("{}")
        return Path(tmp.name)

    def test_the_index_does_not_list_a_partial(self):
        """Mutation: load_runs iterates every directory and the index lists the in-progress mirror."""
        results = self.results_with_a_partial()
        self.assertEqual([run for run, _env, _rows in bench.load_runs(results)], ["2026-10-22-gpu-m-000000-r1"])
        self.assertNotIn(".partial", bench.render_index(bench.load_runs(results)))

    def test_a_partial_is_not_part_of_a_committed_set(self):
        """Mutation: committed_sets iterates every directory and the half-finished repeat joins the baseline."""
        results = self.results_with_a_partial()
        (dirs,) = bench.committed_sets(results).values()
        self.assertEqual([d.name for d in dirs], ["2026-10-22-gpu-m-000000-r1"])

    def test_the_index_ignores_the_failed_results_beside_results(self):
        """Sets with no ok row are gathered into `failed-results/`, a sibling of `results/` the harness never reads.

        Mutation: gather such a set into `results/` and the index test (every run needs an ok row) is red."""
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root))
        failed = root / "failed-results" / "2026-10-22-gpu-m-000000-r1"
        failed.mkdir(parents=True)
        (failed / "results.jsonl").write_text(json.dumps({**self.ROW, "status": "error"}) + "\n")
        (root / "results").mkdir()
        self.assertEqual(bench.load_runs(root / "results"), [])
        self.assertEqual(bench.committed_sets(root / "results"), {})
        ignore = (BENCH_ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("failed-results/", ignore)

    def test_git_ignores_a_partial(self):
        ignore = (BENCH_ROOT / ".gitignore").read_text().splitlines()
        self.assertIn(f"results/*{bench.PARTIAL_SUFFIX}/", ignore)


if __name__ == "__main__":
    unittest.main()
