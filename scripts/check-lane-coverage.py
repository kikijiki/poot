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
                          (poot-npu/vpu): it MUST match a lane selection

An ignored test is run by no selection, so it must be a row of a lane's required list instead (the checkpoint
list, or a scripts/required-device-cases/<lane>.txt list), or be deleted: an ignored test in no list runs nowhere.

The classes are scripts/test-model-free-inventory.tsv's. The required-checkpoint list is hand-maintained
(`lane<TAB>binary id<TAB>test name`, sorted) and never generated from a run, and so are the device lanes'
device lanes' lists, required and unlisted alike (a moved test is a row of the lane that now runs it): a row that names no test of the workspace is stale and fails, so a renamed or deleted
test cannot leave a lane selecting nothing. `--lane-filter <lane>` prints the nextest filterset of one lane's
rows, the exact names the recipe runs (an ignored row is run by exact name only). The lane selections, the
inventory and the lists are independent of each other, so a gate that read them from one source would pass
trivially: `--self-test` proves the gate red against fixtures.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "scripts" / "test-model-free-inventory.tsv"
REQUIRED_CHECKPOINTS = ROOT / "scripts" / "required-checkpoint-cases.txt"
REQUIRED_DEVICE_CASES = ROOT / "scripts" / "required-device-cases"
NEXTEST_CONFIG = ROOT / ".config" / "nextest.toml"

# The features the device lanes build, named here so the listing sees the tests that exist only under them
# (R2-L13-012). The lane recipes in the justfile pass the same features. `poot-npu/xdna` needs `nix develop .#npu`
# (XRT), so its listing runs in that shell: `npu_listing`.
FEATURES = "poot-npu/vpu"
NPU_FEATURES = "poot-npu/xdna,poot-npu/vpu"
# A host with no NPU shell sets this to skip the xdna listing, as `just npu-build` does.
ALLOW_NO_NPU_SHELL = "POOT_ALLOW_NO_NPU_SHELL"

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


def stale_list_cases(listing: dict, cases: dict[tuple[str, str], str]) -> list[str]:
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


def run_listing(command: list[str]) -> dict:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"{' '.join(command[:6])} failed ({result.returncode})")
    # `nix develop` prints the devshell banner on stdout before the command's own output; the listing is the
    # one line that is a JSON object.
    document = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if len(document) != 1:
        raise SystemExit(f"{' '.join(command[:6])}: expected one JSON document on stdout, found {len(document)}")
    return json.loads(document[0])


def nextest_listing(filterset: str) -> dict:
    """Every test of the workspace built with the lane features, `filter-match` evaluated against `filterset`.

    The poot-npu suites come from the `xdna` build in `nix develop .#npu`, which lists a superset of the default
    shell's (the `vpu` and `xdna` rows)."""
    listing = run_listing([
        "cargo", "nextest", "list", "--workspace", "--lib", "--bins", "--tests",
        "--features", FEATURES, "--message-format", "json", "-E", filterset,
    ])  # fmt: skip
    npu_shell = subprocess.run(["nix", "develop", ".#npu", "-c", "true"], cwd=ROOT, capture_output=True)
    if npu_shell.returncode != 0:
        if os.environ.get(ALLOW_NO_NPU_SHELL) != "1":
            raise SystemExit(
                f"nix develop .#npu cannot be entered: the xdna rows cannot be listed (set {ALLOW_NO_NPU_SHELL}=1 "
                "on a host that has no NPU shell)"
            )
        print(f"{ALLOW_NO_NPU_SHELL}=1: the xdna rows are not listed", file=sys.stderr)
        return listing
    npu = run_listing([
        "nix", "develop", ".#npu", "-c", "cargo", "nextest", "list", "-p", "poot-npu", "--lib", "--bins", "--tests",
        "--features", NPU_FEATURES, "--message-format", "json", "-E", filterset,
    ])  # fmt: skip
    listing["rust-suites"] = {
        binary: suite
        for binary, suite in listing["rust-suites"].items()
        if suite["package-name"] != "poot-npu"
    } | npu["rust-suites"]
    return listing


def gpu_group_filter(path: Path) -> str:
    """The filterset of the nextest override that puts tests into the `gpu` test group."""
    overrides = tomllib.loads(path.read_text(encoding="ascii"))["profile"]["default"]["overrides"]
    filters = [entry["filter"].strip() for entry in overrides if entry.get("test-group") == "gpu"]
    if len(filters) != 1:
        raise SystemExit(f"{path.relative_to(ROOT)}: expected one override into the gpu group, found {len(filters)}")
    return filters[0]


def ungrouped(
    selected: dict, grouped: dict, inventory: dict[tuple[str, str], str]
) -> list[str]:
    """`package binary test` for each active test a lane selects that the `gpu` test group does not hold.

    A selected test that is model-free or real-dims is host-only (a `package(...)` selection also picks up its
    package's host tests) and needs no group; every other selected test can reach the device, and two of them
    on the one GPU at once hang it (R2-L13-008)."""
    in_group = {
        (binary, name)
        for binary, suite in grouped["rust-suites"].items()
        for name, case in suite["testcases"].items()
        if case["filter-match"]["status"] == "matches"
    }
    problems = []
    for binary, suite in sorted(selected["rust-suites"].items()):
        for name, case in sorted(suite["testcases"].items()):
            key = (binary, name)
            if case["filter-match"]["status"] != "matches" or case["ignored"]:
                continue
            if inventory.get(key) in COVERED_ELSEWHERE or key in in_group:
                continue
            problems.append(f"{suite['package-name']}  {binary}  {name}")
    return problems


def check(selections: list[str]) -> int:
    union = " | ".join(f"({selection})" for selection in selections)
    listing = nextest_listing(union)
    cases, problems = load_checkpoint_cases(REQUIRED_CHECKPOINTS)
    every_device, _ = load_device_cases(REQUIRED_DEVICE_CASES)
    problems += stale_list_cases(listing, {**cases, **every_device})
    inventory = load_inventory(INVENTORY)
    unselected = uncovered(listing, inventory, cases, every_device)
    escaped = ungrouped(listing, nextest_listing(gpu_group_filter(NEXTEST_CONFIG)), inventory)
    if problems:
        print("required-case lists (scripts/required-checkpoint-cases.txt, scripts/required-device-cases/*.txt):")
        for problem in problems:
            print(f"  {problem}")
    if unselected:
        print("tests that no lane selects (a device test must match a selection in the justfile; select the")
        print("crate by package(...); a test that loads a model is class `checkpoint` in the inventory and a row")
        print("of a checkpoint lane's list; an ignored test is a row of a lane's list or is deleted):")
        for problem in unselected:
            print(f"  {problem}")
    if escaped:
        print("tests a lane selects that the `gpu` test group of .config/nextest.toml does not hold (add the crate, binary")
        print("or test name to that override's filter; two device tests at once can hang the GPU):")
        for problem in escaped:
            print(f"  {problem}")
    if problems or unselected or escaped:
        return 1
    print("lane coverage: every device test matches a lane selection and the gpu test group, every checkpoint and ignored test is a required row")
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
    stale = stale_list_cases(fixture(base), cases)
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
    found = stale_list_cases(fixture(base), stale_cases)
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

    # A selected test the gpu group lacks (R2-L13-008): `escapee` dispatches but no group filter holds it.
    selected = fixture([
        ("pkg-gpu", "pkg-gpu", "grouped", False, True),
        ("pkg-gpu", "pkg-gpu", "escapee", False, True),
        ("pkg", "pkg", "host", False, True),
        ("pkg", "pkg", "unselected", False, False),
    ])  # fmt: skip
    group = fixture([
        ("pkg-gpu", "pkg-gpu", "grouped", False, True),
        ("pkg-gpu", "pkg-gpu", "escapee", False, False),
        ("pkg", "pkg", "host", False, False),
    ])  # fmt: skip
    found = ungrouped(selected, group, inventory)
    if found != ["pkg-gpu  pkg-gpu  escapee"]:
        print(f"self-test: a selected test outside the gpu group was not the only report (got {found})", file=sys.stderr)
        return 1
    print(f"mutation rejected: a selected test outside the gpu group: {found[0]}")
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
    usage = "usage: check-lane-coverage.py --self-test | --lane-filter <lane> | <selection>..."
    if argv in (["--help"], ["-h"]):
        print(usage)
        return 0
    if not argv or any(arg.startswith("-") for arg in argv):
        print(usage, file=sys.stderr)
        return 2
    return check(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
