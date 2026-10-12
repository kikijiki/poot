#!/usr/bin/env python3
"""Fail when a test runs in no lane (card 1058).

`just test-lane-coverage` passes the union of the lane selections defined in the justfile (the device lanes'
nextest filtersets and the isolated package) as arguments. This gate asks nextest for the whole workspace's
active tests with that union as the filter, so nextest itself decides which tests a selection matches, and
classes every test that matches nothing:

  model-free, real-dims   run by the model-free lane (`just test`); not this gate's concern
  checkpoint              loads a model, so it runs in `just test-optional`; never a hardware test
  device                  dispatches on a device: it MUST match a lane selection
  not in the inventory    a test of a package the model-free lane excludes (poot-wgpu-device, poot-rocm-device, poot-npu,
                          poot-vulkan-device, poot-bench-runner), or one that exists only under a feature
                          (poot-llm/rocm): it MUST match a lane selection

The classes are scripts/test-model-free-inventory.tsv's. An ignored test is not an active one and is never
run by a lane; it is out of scope here. The lane selections and the inventory are independent of each other, so
a gate that read both from one source would pass trivially: `--self-test` proves the gate red against fixtures.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "scripts" / "test-model-free-inventory.tsv"

COVERED_ELSEWHERE = {"model-free", "real-dims", "checkpoint"}


def load_inventory(path: Path) -> dict[tuple[str, str], str]:
    """(binary id, test name) -> class, for every row of the committed inventory."""
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    return {(row[1], row[3]): row[2] for row in rows[1:]}


def uncovered(listing: dict, inventory: dict[tuple[str, str], str]) -> list[str]:
    """`package binary test (why)` for each active test that must be selected by a lane and is not."""
    problems = []
    for binary, suite in sorted(listing["rust-suites"].items()):
        for name, case in sorted(suite["testcases"].items()):
            if case["ignored"] or case["filter-match"]["status"] == "matches":
                continue
            cls = inventory.get((binary, name))
            if cls in COVERED_ELSEWHERE:
                continue
            why = f"class {cls}" if cls else "not in the inventory"
            problems.append(f"{suite['package-name']}  {binary}  {name}  ({why})")
    return problems


def nextest_listing(selections: list[str]) -> dict:
    union = " | ".join(f"({selection})" for selection in selections)
    command = [
        "cargo", "nextest", "list", "--workspace", "--lib", "--bins", "--tests",
        "--message-format", "json", "-E", union,
    ]  # fmt: skip
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"{' '.join(command[:4])} failed ({result.returncode})")
    return json.loads(result.stdout)


def check(selections: list[str]) -> int:
    problems = uncovered(nextest_listing(selections), load_inventory(INVENTORY))
    if problems:
        print("tests that no lane selects (a device test must match a selection in the justfile; select the")
        print("crate by package(...); a test that loads a model is class `checkpoint` in the inventory):")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("lane coverage: every device test matches a lane selection")
    return 0


def fixture(cases: list[tuple[str, str, str, bool, bool]]) -> dict:
    """A nextest listing: (package, binary, test, ignored, matched) per case."""
    suites: dict[str, dict] = {}
    for package, binary, name, ignored, matched in cases:
        suite = suites.setdefault(binary, {"package-name": package, "testcases": {}})
        suite["testcases"][name] = {
            "ignored": ignored,
            "filter-match": {"status": "matches" if matched else "mismatch"},
        }
    return {"rust-suites": suites}


def self_test() -> int:
    inventory = {
        ("pkg", "host"): "model-free",
        ("pkg", "slow"): "real-dims",
        ("pkg", "model"): "checkpoint",
        ("pkg", "dispatch"): "device",
    }
    base = [
        ("pkg", "pkg", "host", False, False),
        ("pkg", "pkg", "slow", False, False),
        ("pkg", "pkg", "model", False, False),
        ("pkg", "pkg", "dispatch", False, True),
        ("pkg-gpu", "pkg-gpu", "gpu_test", False, True),
    ]
    clean = uncovered(fixture(base), inventory)
    if clean:
        print(f"self-test: a covered workspace was reported: {clean}", file=sys.stderr)
        return 1

    mutations = {
        "a device test no selection matches": (
            [c for c in base if c[2] != "dispatch"] + [("pkg", "pkg", "dispatch", False, False)],
            "dispatch",
        ),
        "a test of a package outside the inventory that no selection matches": (
            base + [("pkg-gpu", "pkg-gpu", "added_later", False, False)],
            "added_later",
        ),
        "a feature-only test that no selection matches": (
            base + [("pkg", "pkg::rocm_only", "rocm_only_test", False, False)],
            "rocm_only_test",
        ),
    }
    for label, (cases, needle) in mutations.items():
        found = uncovered(fixture(cases), inventory)
        if not any(needle in line for line in found):
            print(f"self-test: {label} was not reported (got {found})", file=sys.stderr)
            return 1
        print(f"mutation rejected: {label}: {found[0]}")

    ignored = uncovered(fixture(base + [("pkg-gpu", "pkg-gpu", "hangs", True, False)]), inventory)
    if ignored:
        print(f"self-test: an ignored test was reported: {ignored}", file=sys.stderr)
        return 1
    print("lane coverage self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    if not argv:
        print("usage: check-lane-coverage.py --self-test | <selection>...", file=sys.stderr)
        return 2
    return check(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
