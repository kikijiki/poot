#!/usr/bin/env python3
"""Fail when a tracked file names a `docs/<path>.md` file that does not exist.

The repository's own docs/ directory holds a few files; project notes live in Backstage, so a comment that
points at `docs/updates/...` or `docs/<name>.md` is a dangling reference. Only repo-root `docs/` paths are
checked: `website/docs/...` and other prefixed paths are not (the lookbehind rejects any path character before
`docs/`). website/ and benchmarks/results/ are skipped, as in `just docs-check`.

usage: scripts/check-docs-paths.py [--self-test]
"""
import re
import subprocess
import sys
from pathlib import Path

REFERENCE = re.compile(r"(?<![\w./-])docs/[\w./*-]*?\.md\b")
SKIPPED = ("website/", "benchmarks/results/")


def dangling(text: str, exists) -> list[str]:
    return [m.group(0) for m in REFERENCE.finditer(text) if not exists(m.group(0))]


def self_test() -> None:
    # The examples are assembled from parts so this file does not itself hold a dangling reference.
    d = "docs" + "/"
    present = {d + "dscan.md"}.__contains__
    assert dangling(f"see {d}dscan.md", present) == []
    assert dangling(f"see `{d}gone.md`.", present) == [d + "gone.md"]
    assert dangling(f"see {d}updates/0615-x.md", present) == [d + "updates/0615-x.md"]
    assert dangling(f"see {d}tasks/*/125-x.md", present) == [d + "tasks/*/125-x.md"]
    assert dangling(f"website/{d}gone.md and projects/poot/{d}gone.md", present) == []


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    failures = 0
    for name in filter(None, tracked):
        path = root / name
        if name.startswith(SKIPPED) or not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for ref in dangling(line, lambda ref: (root / ref).is_file()):
                print(f"{name}:{number}: {ref} does not exist")
                failures += 1
    if failures:
        print(f"{failures} reference(s) to docs/ files that do not exist")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
