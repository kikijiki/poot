"""The standing benchmark grid and its completeness check (stdlib only).

`benchmarks/grid.toml` declares every cell the Performance page is expected to show. This module expands the file into
cells and judges each against the committed Performance page data (`website/src/data/performance.json`):

* a cell without a status is required: every context length of its scenario has a latest, verified point of at
  least the grid's `repeats` repeats on the machine's hardware, else the check fails;
* a `pending` cell is allowed to lack data and is listed; `--strict` (the release gate) fails on it;
* a `not-run` cell needs a `reason` and is never judged against data.

A pending cell that already has its data is reported as measured, and its marker is stale but not an error: the lane
that measured it need not edit the grid in the same commit.

    python3 benchmarks/harness/grid.py check [--strict]
    python3 benchmarks/harness/grid.py machine <id>     # the machine's orchestrator arguments
"""

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
BENCH_ROOT = HARNESS_DIR.parent
REPO_ROOT = BENCH_ROOT.parent
GRID = BENCH_ROOT / "grid.toml"
MANIFEST = BENCH_ROOT / "manifest.toml"
PERFORMANCE = REPO_ROOT / "website" / "src" / "data" / "performance.json"

STATUS_REQUIRED, STATUS_PENDING, STATUS_NOT_RUN = "required", "pending", "not-run"
STATUSES = (STATUS_PENDING, STATUS_NOT_RUN)
MACHINE_FIELDS = {"id", "hardware", "label", "cloud", "gpu_types", "max_price_per_hour"}
GROUP_FIELDS = {"machine", "scenario", "models", "engines", "status", "reason"}


@dataclass(frozen=True)
class Cell:
    machine: str
    model: str
    framework: str
    backend: str
    scenario: str
    status: str
    reason: str

    def label(self):
        return f"{self.machine} / {self.model} / {self.framework} {self.backend} / {self.scenario}"


def load_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_cells(grid, manifest):
    """Every cell the grid declares, or ValueError naming the first malformed declaration."""
    unknown = set(grid) - {"repeats", "machine", "group"}
    if unknown:
        raise ValueError(f"grid.toml: unknown top-level keys {sorted(unknown)}")
    minimum = manifest["performance"]["min_trials"]
    if not isinstance(grid.get("repeats"), int) or grid["repeats"] < minimum:
        raise ValueError(f"grid.toml: repeats must be an integer of at least {minimum} (manifest min_trials)")

    machines = {}
    for machine in grid.get("machine", []):
        if set(machine) - MACHINE_FIELDS or not {"id", "hardware"} <= set(machine):
            raise ValueError(f"grid.toml: machine {machine.get('id')!r} must hold id and hardware and only {sorted(MACHINE_FIELDS)}")
        if machine["id"] in machines:
            raise ValueError(f"grid.toml: machine {machine['id']!r} declared twice")
        machines[machine["id"]] = machine

    models = {model["id"]: model for model in manifest["model"]}
    scenarios = {scenario["id"]: scenario for scenario in manifest["scenario"]}
    cells, seen = [], set()
    for group in grid.get("group", []):
        where = f"grid.toml group {group.get('machine')}/{group.get('scenario')}/{group.get('models')}"
        if set(group) - GROUP_FIELDS or not {"machine", "scenario", "models", "engines"} <= set(group):
            raise ValueError(f"{where}: needs machine, scenario, models, engines and only {sorted(GROUP_FIELDS)}")
        status = group.get("status", STATUS_REQUIRED)
        if status != STATUS_REQUIRED and status not in STATUSES:
            raise ValueError(f"{where}: status {status!r} is neither absent, {STATUS_PENDING!r} nor {STATUS_NOT_RUN!r}")
        reason = group.get("reason", "")
        if status == STATUS_NOT_RUN and not reason:
            raise ValueError(f"{where}: a not-run cell needs its reason")
        if status != STATUS_NOT_RUN and reason:
            raise ValueError(f"{where}: only a not-run cell carries a reason")
        if group["machine"] not in machines:
            raise ValueError(f"{where}: machine {group['machine']!r} is not declared")
        scenario = scenarios.get(group["scenario"])
        if scenario is None or not scenario.get("isl"):
            raise ValueError(f"{where}: scenario {group['scenario']!r} is not a decode-curve scenario of the manifest")
        for model in group["models"]:
            if model not in models:
                raise ValueError(f"{where}: model {model!r} is not in the manifest")
            for engine in group["engines"]:
                framework = engine["framework"]
                declared = models[model]["frameworks"].get(framework, {})
                if not declared.get("support", True):
                    raise ValueError(f"{where}: the manifest says {framework} does not run {model}: {declared.get('reason')}")
                cell = Cell(group["machine"], model, framework, engine["backend"], group["scenario"], status, reason)
                key = cell.label()
                if key in seen:
                    raise ValueError(f"grid.toml declares the cell {key} twice")
                seen.add(key)
                cells.append(cell)
    return cells


def verified_isls(performance, hardware, cell, repeats):
    """The context lengths at which the page data holds a latest, verified point of the cell with enough repeats."""
    for entry in performance["hardware"]:
        if entry["id"] == hardware:
            return {
                point["isl"]
                for point in entry["points"]
                if point["latest"]
                and point["verified"]
                and point["repeats"] >= repeats
                and (point["model"], point["framework"], point["backend"]) == (cell.model, cell.framework, cell.backend)
            }
    return set()


@dataclass
class Report:
    failures: list
    pending: list
    measured_pending: list
    not_run: list
    satisfied: int


def judge(grid, manifest, performance):
    cells = load_cells(grid, manifest)
    machines = {machine["id"]: machine for machine in grid["machine"]}
    scenarios = {scenario["id"]: scenario for scenario in manifest["scenario"]}
    report = Report([], [], [], [], 0)
    for cell in cells:
        if cell.status == STATUS_NOT_RUN:
            report.not_run.append(cell)
            continue
        wanted = set(scenarios[cell.scenario]["isl"])
        have = verified_isls(performance, machines[cell.machine]["hardware"], cell, grid["repeats"])
        missing = sorted(wanted - have)
        if cell.status == STATUS_PENDING:
            (report.pending if missing else report.measured_pending).append(cell)
        elif missing:
            report.failures.append(
                f"{cell.label()}: no verified point with {grid['repeats']} repeats at context length(s) {missing}; "
                f"measure it, mark it pending, or mark it not-run with a reason"
            )
        else:
            report.satisfied += 1
    return report


def check(strict, grid_path=GRID, manifest_path=MANIFEST, performance_path=PERFORMANCE, out=sys.stdout):
    """Print the grid's state; the exit code is 1 when a required cell lacks data (or, under `strict`, is pending)."""
    report = judge(load_toml(grid_path), load_toml(manifest_path), json.loads(Path(performance_path).read_text()))
    failures = list(report.failures)
    if strict:
        failures += [f"{cell.label()}: pending (release gate)" for cell in report.pending]
    print(
        f"grid: {report.satisfied} cells verified, {len(report.pending)} pending, "
        f"{len(report.not_run)} not run, {len(report.measured_pending)} pending but already measured",
        file=out,
    )
    for cell in report.pending:
        print(f"  pending: {cell.label()}", file=out)
    for cell in report.measured_pending:
        print(f"  measured, drop the pending status: {cell.label()}", file=out)
    for failure in report.failures:
        print(f"  FAIL {failure}", file=out)
    if strict:
        for cell in report.pending:
            print(f"  FAIL {cell.label()}: pending (release gate)", file=out)
    return 1 if failures else 0


def machine_arguments(machine_id, grid_path=GRID):
    """The orchestrator flags the machine declares (GPU types, cloud, price cap); only those its entry states."""
    for machine in load_toml(grid_path)["machine"]:
        if machine["id"] == machine_id:
            flags = []
            if "gpu_types" in machine:
                flags += ["--gpu-types", ",".join(machine["gpu_types"])]
            if "cloud" in machine:
                flags += ["--cloud", machine["cloud"]]
            if "max_price_per_hour" in machine:
                flags += ["--max-price-per-hour", str(machine["max_price_per_hour"])]
            return flags
    raise ValueError(f"grid.toml declares no machine {machine_id!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check", help="every required cell is verified, pending or has a reason")
    check_parser.add_argument("--strict", action="store_true", help="fail on pending cells too (release gate)")
    machine_parser = sub.add_parser("machine", help="print a machine's orchestrator flags")
    machine_parser.add_argument("id")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return check(args.strict)
        print(" ".join(f"'{flag}'" if " " in flag else flag for flag in machine_arguments(args.id)))
        return 0
    except ValueError as err:
        print(f"grid: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
