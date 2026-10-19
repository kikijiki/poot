#!/usr/bin/env python3
"""Fail when a test runs in no lane (card 1058).

`just test-lane-coverage` passes the union of the device lane selections defined in the justfile as
arguments. This gate asks nextest for the whole workspace's active tests with that union as the filter, so
nextest itself decides which tests a selection matches, and
classes every test that matches nothing:

  model-free, real-dims   run by the model-free lane (`just test`); not this gate's concern
  checkpoint              loads a model: it MUST be a row of a checkpoint lane (scripts/required-checkpoint-cases.txt, or
                          scripts/required-device-cases/checkpoint-*.txt for a device row), whose recipe sets
                          POOT_REQUIRE_MODELS=1 so a missing model fails
  device                  dispatches on a device: it MUST match a lane selection
  not in the inventory    a test of a package the model-free lane excludes (poot-wgpu-device, poot-rocm-device, poot-npu,
                          poot-vulkan-device), or one that exists only under a feature
                          (poot-llm/rocm): it MUST match a lane selection

An ignored test is run by no selection, so it must be a row of a lane's required list instead (the checkpoint
list, or a scripts/required-device-cases/<lane>.txt list), or be deleted: an ignored test in no list runs nowhere.

The classes are scripts/test-model-free-inventory.tsv's. The required-checkpoint list is hand-maintained
(`lane<TAB>binary id<TAB>test name`, sorted) and never generated from a run, and so are the device lanes'
`checkpoint-*` lists: a row that names no test of the workspace is stale and fails, so a renamed or deleted
test cannot leave a lane selecting nothing. `--lane-filter <lane>` prints the nextest filterset of one lane's
rows, the exact names the recipe runs (an ignored row is run by exact name only). The lane selections, the
inventory and the lists are independent of each other, so a gate that read them from one source would pass
trivially: `--self-test` proves the gate red against fixtures.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "scripts" / "test-model-free-inventory.tsv"
REQUIRED_CHECKPOINTS = ROOT / "scripts" / "required-checkpoint-cases.txt"
REQUIRED_DEVICE_CASES = ROOT / "scripts" / "required-device-cases"

COVERED_ELSEWHERE = {"model-free", "real-dims"}

# The lanes of the required-checkpoint list: each is a `just` recipe that runs its rows by exact name (a
# `checkpoint*` lane with POOT_REQUIRE_MODELS=1). The device lanes' rows (`checkpoint-<backend>`) are in
# scripts/required-device-cases/, where `check-required-cases.py` also reads them.
CHECKPOINT_LANES = {
    "checkpoint",
    "checkpoint-large",
    "npu-toolchain",
    "real-dims-long",
    "regen-kernel-assets",
}


def load_inventory(path: Path) -> dict[tuple[str, str], str]:
    """(binary id, test name) -> class, for every row of the committed inventory."""
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    return {(row[1], row[3]): row[2] for row in rows[1:]}


def load_checkpoint_cases(path: Path) -> tuple[dict[tuple[str, str], str], list[str]]:
    """(binary id, test name) -> lane for every row of the required-checkpoint list, plus its schema problems."""
    if not path.is_file():
        return {}, [f"required-checkpoint list missing: {path.relative_to(ROOT)}"]
    cases: dict[tuple[str, str], str] = {}
    problems = []
    keys = []
    for number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3 or not all(parts):
            problems.append(f"required-checkpoint list line {number}: expected lane<TAB>binary id<TAB>test name")
            continue
        lane, binary, name = parts
        if lane not in CHECKPOINT_LANES:
            problems.append(f"required-checkpoint list line {number}: unknown lane {lane} ({sorted(CHECKPOINT_LANES)})")
        if (binary, name) in cases:
            problems.append(f"required-checkpoint list line {number}: duplicate row {binary} {name}")
        cases[(binary, name)] = lane
        keys.append((lane, binary, name))
    if keys != sorted(keys):
        problems.append("required-checkpoint list is not sorted by lane, binary id, test name (LC_ALL=C order)")
    if not cases and not problems:
        problems.append("required-checkpoint list names no test")
    return cases, problems


def device_lane_path(lane: str) -> Path:
    return REQUIRED_DEVICE_CASES / f"{lane}.txt"


def load_device_cases(directory: Path) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """(binary id, test name) -> lane for every device-lane list row, and the `checkpoint-*` subset of it.

    The device lanes' lists are `binary id<TAB>test name` (`check-required-cases.py` owns their schema checks)."""
    every: dict[tuple[str, str], str] = {}
    for path in sorted(directory.glob("*.txt")):
        for line in path.read_text(encoding="ascii").splitlines():
            parts = line.split("\t")
            if line.startswith("#") or len(parts) < 2:
                continue
            every[(parts[0], parts[1])] = path.stem
    checkpoint = {key: lane for key, lane in every.items() if lane.startswith("checkpoint-")}
    return every, checkpoint


def stale_checkpoint_cases(listing: dict, cases: dict[tuple[str, str], str]) -> list[str]:
    """`lane binary test` for each list row that names no test of the workspace, ignored or not."""
    existing = {
        (binary, name) for binary, suite in listing["rust-suites"].items() for name in suite["testcases"]
    }
    return [
        f"{lane}  {binary}  {name}  (names no test)"
        for (binary, name), lane in sorted(cases.items())
        if (binary, name) not in existing
    ]


def lane_filter(cases: dict[tuple[str, str], str], lane: str) -> str:
    """The nextest filterset that selects exactly the rows of `lane` by identity."""
    rows = sorted(key for key, row_lane in cases.items() if row_lane == lane)
    if not rows:
        raise SystemExit(f"no required row for lane {lane}")
    return " | ".join(f"(binary_id(={binary}) & test(={name}))" for binary, name in rows)


def uncovered(
    listing: dict,
    inventory: dict[tuple[str, str], str],
    lane_cases: dict[tuple[str, str], str],
    device_cases: dict[tuple[str, str], str],
) -> list[str]:
    """`package binary test (why)` for each test that must be selected by a lane and is not.

    An active test must match a selection, be model-free or real-dims, or (a checkpoint) be a lane row. An
    ignored test is run by no selection, so it must be a row of a lane list: `lane_cases` (the checkpoint
    list) or `device_cases` (the device lanes' lists)."""
    problems = []
    for binary, suite in sorted(listing["rust-suites"].items()):
        for name, case in sorted(suite["testcases"].items()):
            key = (binary, name)
            where = f"{suite['package-name']}  {binary}  {name}"
            if case["ignored"]:
                if key not in lane_cases and key not in device_cases:
                    problems.append(f"{where}  (ignored, in no lane's required list: list it or delete it)")
                continue
            if case["filter-match"]["status"] == "matches":
                continue
            cls = inventory.get(key)
            if cls in COVERED_ELSEWHERE:
                continue
            if cls == "checkpoint" and (key in lane_cases or key in device_cases):
                continue
            why = f"class {cls}" if cls else "not in the inventory"
            problems.append(f"{where}  ({why})")
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
    listing = nextest_listing(selections)
    cases, problems = load_checkpoint_cases(REQUIRED_CHECKPOINTS)
    every_device, checkpoint_device = load_device_cases(REQUIRED_DEVICE_CASES)
    problems += stale_checkpoint_cases(listing, {**cases, **checkpoint_device})
    unselected = uncovered(listing, load_inventory(INVENTORY), cases, every_device)
    if problems:
        print("required-checkpoint lists (scripts/required-checkpoint-cases.txt, required-device-cases/checkpoint-*.txt):")
        for problem in problems:
            print(f"  {problem}")
    if unselected:
        print("tests that no lane selects (a device test must match a selection in the justfile; select the")
        print("crate by package(...); a test that loads a model is class `checkpoint` in the inventory and a row")
        print("of a checkpoint lane's list; an ignored test is a row of a lane's list or is deleted):")
        for problem in unselected:
            print(f"  {problem}")
    if problems or unselected:
        return 1
    print("lane coverage: every device test matches a lane selection, every checkpoint and ignored test is a required row")
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
        ("pkg", "second_model"): "checkpoint",
    }
    cases = {("pkg", "model"): "checkpoint"}
    base = [
        ("pkg", "pkg", "host", False, False),
        ("pkg", "pkg", "slow", False, False),
        ("pkg", "pkg", "model", False, False),
        ("pkg", "pkg", "dispatch", False, True),
        ("pkg-gpu", "pkg-gpu", "gpu_test", False, True),
    ]
    clean = uncovered(fixture(base), inventory, cases, {})
    if clean:
        print(f"self-test: a covered workspace was reported: {clean}", file=sys.stderr)
        return 1
    stale = stale_checkpoint_cases(fixture(base), cases)
    if stale:
        print(f"self-test: a list naming an existing test was reported stale: {stale}", file=sys.stderr)
        return 1

    mutations = {
        "a device test no selection matches": (
            [c for c in base if c[2] != "dispatch"] + [("pkg", "pkg", "dispatch", False, False)],
            cases,
            "dispatch",
        ),
        "a test of a package outside the inventory that no selection matches": (
            base + [("pkg-gpu", "pkg-gpu", "added_later", False, False)],
            cases,
            "added_later",
        ),
        "a feature-only test that no selection matches": (
            base + [("pkg", "pkg::rocm_only", "rocm_only_test", False, False)],
            cases,
            "rocm_only_test",
        ),
        "a checkpoint test that is no row of the required-checkpoint list": (base, {}, "model"),
        "a second checkpoint test the list does not name": (
            base + [("pkg", "pkg", "second_model", False, False)],
            cases,
            "second_model",
        ),
    }
    for label, (listed, listed_cases, needle) in mutations.items():
        found = uncovered(fixture(listed), inventory, listed_cases, {})
        if not any(needle in line for line in found):
            print(f"self-test: {label} was not reported (got {found})", file=sys.stderr)
            return 1
        print(f"mutation rejected: {label}: {found[0]}")

    stale_cases = {**cases, ("pkg", "renamed_away"): "checkpoint"}
    found = stale_checkpoint_cases(fixture(base), stale_cases)
    if not any("renamed_away" in line for line in found):
        print(f"self-test: a list row naming no test was not reported (got {found})", file=sys.stderr)
        return 1
    print(f"mutation rejected: a list row naming no test: {found[0]}")

    hung = fixture(base + [("pkg-gpu", "pkg-gpu", "hangs", True, False)])
    found = uncovered(hung, inventory, cases, {})
    if not any("hangs" in line and "ignored" in line for line in found):
        print(f"self-test: an ignored test in no lane list was not reported (got {found})", file=sys.stderr)
        return 1
    print(f"mutation rejected: an ignored test in no lane list: {found[0]}")
    for label, lane_cases, device_cases in (
        ("the checkpoint list", {**cases, ("pkg-gpu", "hangs"): "checkpoint"}, {}),
        ("a device lane's list", cases, {("pkg-gpu", "hangs"): "wgpu"}),
    ):
        found = uncovered(hung, inventory, lane_cases, device_cases)
        if found:
            print(f"self-test: an ignored test listed in {label} was reported: {found}", file=sys.stderr)
            return 1
    print("lane coverage self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    if len(argv) == 2 and argv[0] == "--lane-filter":
        cases, problems = load_checkpoint_cases(REQUIRED_CHECKPOINTS)
        if problems:
            raise SystemExit("\n".join(problems))
        _, checkpoint_device = load_device_cases(REQUIRED_DEVICE_CASES)
        print(lane_filter({**cases, **checkpoint_device}, argv[1]))
        return 0
    if not argv:
        print("usage: check-lane-coverage.py --self-test | --lane-filter <lane> | <selection>...", file=sys.stderr)
        return 2
    return check(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
