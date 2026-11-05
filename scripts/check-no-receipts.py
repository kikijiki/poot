#!/usr/bin/env python3
"""Fail when a tracked path sits under the top-level `receipts/` directory.

Per-card receipts, evidence and PR text are private material: they are kept outside
this public repository. `/receipts/` is gitignored; this check
reads the index, so it also catches a file added with `git add -f`. The published bank data under
`benchmarks/performance-bank/receipts*` is a different path and is not matched.

usage: scripts/check-no-receipts.py [--self-test]
"""
import subprocess
import sys

ROOT_DIR = "receipts"


def offending(paths: list[str]) -> list[str]:
    return [p for p in paths if p == ROOT_DIR or p.startswith(ROOT_DIR + "/")]


def self_test() -> None:
    assert offending(["README.md", "crates/a/src/lib.rs"]) == []
    assert offending(["receipts/1480-x/outcome.md", "justfile"]) == ["receipts/1480-x/outcome.md"]
    assert offending(["benchmarks/performance-bank/receipts/wgpu/x.json"]) == []
    assert offending(["receipts.json", "receiptsx/a"]) == []


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    found = offending([p for p in tracked if p])
    for path in found:
        print(f"{path}: receipts are private material and do not belong in this repository")
    if found:
        print(f"{len(found)} tracked path(s) under receipts/")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
