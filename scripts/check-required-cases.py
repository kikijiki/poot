#!/usr/bin/env python3
"""Required-case gate for the `just test-device-*` lanes (card 670) and the checkpoint-style lanes (POOT-1467).

Each lane has a hand-maintained list, scripts/required-device-cases/<lane>.txt - one line per
required case, `binary_id<TAB>test_name` (the same identity JUnit reports as `classname` and
`name`), committed as text and never generated from the lane's selection. After the lane's
nextest run, this gate compares that list with the run's actual results
(`target/nextest/device-<lane>/results.xml`, written by the lane profile in .config/nextest.toml)
and fails when a required case did not run, failed, or was skipped. The two lists are
independently sourced, so a comparison that read its required cases from the run's own results
passes trivially; `--self-test` proves both of the card's mutations red against fixture results.

The four non-device lanes that run required-checkpoint rows by exact name (`checkpoint`, `checkpoint-large`,
`real-dims-long`, `npu-toolchain`) are gated the same way: their required rows are the lane's rows of
scripts/required-checkpoint-cases.txt (`lane<TAB>binary_id<TAB>test_name`), they run under the nextest profile
`lane-<lane>`, and they have no unlisted or parked rows, because the lane's selection is its required list: a
row that ran and is not required fails the gate as unclassified.

Every lane classifies every row it ran: a passing row that is in neither the required list nor the lane's
`<lane>.unlisted.txt` (`binary_id<TAB>test_name<TAB>reason`, a deliberate non-requirement with its reason; the
file exists for every lane, empty when the lane's selection is its required list) fails the gate, so a row added
to the lane's selection cannot go unclassified (cards 1125, 1218). A row that printed a skip report fails the
gate whether it is required, unlisted or neither: a skip is never a pass for any row.

A lane may also have `<lane>.parked.txt` (same schema): a measured row that does not pass yet, `#[ignore]`d with the
same reason, which must name the card that makes it pass (POOT-1454). The lane does not run it, and the lane
coverage gate (scripts/check-lane-coverage.py) counts it as listed, so the test stays in the inventory and
visible. A parked row that runs and passes fails the gate until it is moved back to the required list.

The results file is bound to its run. A lane recipe calls `begin <lane> <results>` before the run, which deletes
the results file and stamps a start marker beside it, and `check <lane> <results>` after it, which rejects a
results file that is missing, older than the marker, or written by another nextest profile than
`device-<lane>`. There is no default results path, so a stale file from an earlier lane or an earlier run
cannot satisfy the gate.

A skip is never a pass. nextest reports a runtime skip as a passing testcase - the test printed
its skip report and returned - so the gate reads the captured output (`store-success-output = true`
in the lane profile) and classifies a testcase whose output carries a skip report as skipped:

    POOT-SKIP: wgpu device unavailable: no adapter          (DeviceBackend::skip_unavailable)
    POOT-SKIP poot-kernelgen: case (no GPU agent)           (a runtime's own report)

A skip report is a line carrying the reserved token `POOT-SKIP`, at the start of the line or after
whitespace and followed by `:` or whitespace. The token is the one skip channel of the workspace: every
test that skips prints it, and nothing else is read as a skip. Free text that merely mentions a skip
(`add: SKIP (not staged)`, `skip: ...`, `skipping ...`, `foo_skips_bar`) is ordinary output, so a host
test of a tool that talks about skipped inputs is not mistaken for a skipped row (cards 1125, 1218, 1325).
A `<skipped>` testcase (a test the runner itself did not run) is a skip too.
"""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_DIR = ROOT / "scripts" / "required-device-cases"
REQUIRED_CHECKPOINTS = ROOT / "scripts" / "required-checkpoint-cases.txt"

# The non-device lanes whose required rows are in REQUIRED_CHECKPOINTS, and run under the profile `lane-<lane>`.
CHECKPOINT_STYLE_LANES = ("checkpoint", "checkpoint-large", "npu-toolchain", "real-dims-long")

SKIP_REPORT = re.compile(r"(?:^|\s)POOT-SKIP(?::|\s|$)")


class Problem(Exception):
    """A results-file problem: reported as the gate's own failure message."""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def profile_of(lane: str) -> str:
    """The nextest profile a lane runs under, and so the report name its results file carries."""
    return f"lane-{lane}" if lane in CHECKPOINT_STYLE_LANES else f"device-{lane}"


def parse_required_list(path: Path, lane: str | None = None) -> tuple[list[tuple[str, str]], list[str]]:
    """The required rows of a hand-maintained list, plus every schema problem in it.

    With a `lane`, the list is the checkpoint list (`lane<TAB>binary_id<TAB>test_name`) and the rows are the lane's."""
    if not path.is_file():
        return [], [f"required-case list missing: {rel(path)}"]
    rows: list[tuple[str, str]] = []
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    text = path.read_text(encoding="ascii")
    columns = 2 if lane is None else 3
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != columns or not all(parts):
            expected = "binary_id<TAB>test_name" if lane is None else "lane<TAB>binary_id<TAB>test_name"
            problems.append(f"required-case list malformed at line {lineno}: expected {expected}: {line}")
            continue
        if lane is not None:
            if parts[0] != lane:
                continue
            parts = parts[1:]
        row = (parts[0], parts[1])
        if row in seen:
            problems.append(f"required-case list has a duplicate row: {row[0]} {row[1]}")
        seen.add(row)
        rows.append(row)
    if not rows and not problems:
        problems.append(f"required-case list is empty: {rel(path)} names no test" + (f" for lane {lane}" if lane else ""))
    data_lines = [f"{binary}\t{test}" for binary, test in rows]
    if data_lines != sorted(data_lines):
        problems.append(f"required-case list is not in canonical sorted order: {rel(path)}")
    return rows, problems


def parse_unlisted_list(path: Path | None) -> tuple[dict[tuple[str, str], str], list[str]]:
    """The deliberately unlisted rows of a lane (row -> reason), plus every schema problem in it.

    Every device lane has the file; a lane with nothing to leave unlisted has one with only comments. A lane
    whose selection is its required list (`CHECKPOINT_STYLE_LANES`) has no file (`None`) and leaves nothing unlisted."""
    if path is None:
        return {}, []
    if not path.is_file():
        return {}, [f"unlisted-case list missing: {rel(path)}"]
    rows: dict[tuple[str, str], str] = {}
    problems: list[str] = []
    data_lines: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            problems.append(
                f"unlisted-case list malformed at line {lineno}: expected binary_id<TAB>test_name<TAB>reason: {line}"
            )
            continue
        row = (parts[0], parts[1])
        if row in rows:
            problems.append(f"unlisted-case list has a duplicate row: {row[0]} {row[1]}")
        rows[row] = parts[2]
        data_lines.append(f"{parts[0]}\t{parts[1]}")
    if data_lines != sorted(data_lines):
        problems.append(f"unlisted-case list is not in canonical sorted order: {rel(path)}")
    return rows, problems


PARKING_CARD = re.compile(r"\bPOOT-\d+\b")


def parse_parked_list(path: Path | None) -> tuple[dict[tuple[str, str], str], list[str]]:
    """The parked rows of a lane (row -> reason), plus every schema problem in it.

    A parked row is a measured, not-yet-passing row: the test is `#[ignore]`d with the same reason, its lane does
    not run it, and the reason names the card that makes it pass. The file is optional; a lane without one parks
    nothing."""
    if path is None or not path.is_file():
        return {}, []
    rows: dict[tuple[str, str], str] = {}
    problems: list[str] = []
    data_lines: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            problems.append(
                f"parked-case list malformed at line {lineno}: expected binary_id<TAB>test_name<TAB>reason: {line}"
            )
            continue
        row = (parts[0], parts[1])
        if row in rows:
            problems.append(f"parked-case list has a duplicate row: {row[0]} {row[1]}")
        if not PARKING_CARD.search(parts[2]):
            problems.append(f"parked row names no card that makes it pass (POOT-NNNN): {row[0]} {row[1]}")
        rows[row] = parts[2]
        data_lines.append(f"{parts[0]}\t{parts[1]}")
    if data_lines != sorted(data_lines):
        problems.append(f"parked-case list is not in canonical sorted order: {rel(path)}")
    return rows, problems


def skip_report(text: str | None) -> str | None:
    """The first skip-report line in captured output, if it carries one."""
    if not text:
        return None
    for line in text.splitlines():
        if SKIP_REPORT.search(line):
            return line.strip()
    return None


def element_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    return element.text


def marker_path(results: Path) -> Path:
    """The start marker `begin` stamps beside a results file."""
    return results.with_name(results.name + ".started")


def begin(lane: str, results: Path) -> int:
    """Delete the lane's results file and stamp the start of its run."""
    results.parent.mkdir(parents=True, exist_ok=True)
    results.unlink(missing_ok=True)
    marker = marker_path(results)
    marker.write_text(f"{lane}\n", encoding="ascii")
    return 0


def load_results(
    path: Path, lane: str | None = None
) -> dict[tuple[str, str], tuple[str, str]]:
    """(binary id, test name) -> (passed|failed|skipped, detail) from the lane's JUnit report.

    With a lane, the file must belong to this run of it: newer than the start marker `begin` stamped and
    written by the lane's own nextest profile. Testcases appear in execution order; a later entry for the same
    identity wins (the lanes configure no retries, so each identity appears once).
    """
    if not path.is_file():
        raise Problem(f"no lane results at {rel(path)}")
    if lane is not None:
        marker = marker_path(path)
        if not marker.is_file():
            raise Problem(f"lane run has no start marker {rel(marker)}: the recipe must call `begin` before the run")
        if path.stat().st_mtime_ns < marker.stat().st_mtime_ns:
            raise Problem(f"lane results are stale: {rel(path)} is older than the start of this run")
    try:
        tree = ET.parse(path)
    except ET.ParseError as error:
        raise Problem(f"lane results are not valid XML: {rel(path)}: {error}") from error
    if lane is not None:
        profile = tree.getroot().get("name")
        if profile != profile_of(lane):
            raise Problem(
                f"lane results are foreign: {rel(path)} was written by profile {profile!r}, not {profile_of(lane)}"
            )
    results: dict[tuple[str, str], tuple[str, str]] = {}
    for case in tree.getroot().iter("testcase"):
        identity = (case.get("classname") or "", case.get("name") or "")
        failure = case.find("failure")
        if failure is not None:
            detail = next(
                (line for line in (failure.text or "").splitlines() if line.strip()),
                failure.get("type") or "failure",
            )
            results[identity] = ("failed", detail.strip())
            continue
        if case.find("skipped") is not None:
            results[identity] = ("skipped", "reported skipped by the runner")
            continue
        report = skip_report(element_text(case.find("system-err"))) or skip_report(
            element_text(case.find("system-out"))
        )
        if report is not None:
            results[identity] = ("skipped", report)
            continue
        results[identity] = ("passed", "")
    return results


def check(
    list_path: Path,
    results_path: Path,
    label: str,
    unlisted_path: Path | None,
    bound_lane: str | None = None,
    parked_path: Path | None = None,
    list_lane: str | None = None,
) -> int:
    """Fail unless every required case in list_path ran and passed in results_path.

    Every other row of the results must be named in unlisted_path, and a row named there must have run, so the
    classification cannot rot either. A skipped row fails wherever it is listed. bound_lane binds the results
    file to its run (see `load_results`). A parked row (parked_path) is in neither list and is not run; if it
    runs and passes, it fails the gate until it is moved back to the required list, so a parking cannot outlive
    its fix."""
    rows, problems = parse_required_list(list_path, list_lane)
    unlisted, unlisted_problems = parse_unlisted_list(unlisted_path)
    parked, parked_problems = parse_parked_list(parked_path)
    problems += unlisted_problems + parked_problems
    both = sorted(set(rows) & set(unlisted))
    problems += [f"row is both required and unlisted: {b} {t}" for b, t in both]
    problems += [f"row is both required and parked: {b} {t}" for b, t in sorted(set(rows) & set(parked))]
    problems += [f"row is both unlisted and parked: {b} {t}" for b, t in sorted(set(unlisted) & set(parked))]
    results: dict[tuple[str, str], tuple[str, str]] | None = None
    if not problems:
        try:
            results = load_results(results_path, bound_lane)
        except Problem as error:
            problems.append(str(error))
    if results is not None:
        for binary, test in rows:
            verdict = results.get((binary, test))
            if verdict is None:
                problems.append(f"required case did not run: {binary} {test}")
                continue
            status, detail = verdict
            if status == "failed":
                problems.append(f"required case failed: {binary} {test}: {detail}")
            elif status == "skipped":
                problems.append(f"required case skipped: {binary} {test}: {detail}")
        required = set(rows)
        for identity, (status, detail) in sorted(results.items()):
            if identity in required:
                continue
            if status == "skipped":
                problems.append(f"unlisted case skipped: {identity[0]} {identity[1]}: {detail}")
            elif identity in parked and status == "passed":
                problems.append(
                    f"parked case passed (move it back to {rel(list_path)} and drop its #[ignore]): "
                    f"{identity[0]} {identity[1]}"
                )
            elif identity not in unlisted and identity not in parked and status == "passed":
                where = f"list it in {rel(list_path)}"
                if unlisted_path is not None:
                    where += f" or give it a reason in {rel(unlisted_path)}"
                problems.append(f"unclassified passing row ({where}): {identity[0]} {identity[1]}")
        for binary, test in sorted(unlisted):
            if (binary, test) not in results:
                problems.append(f"unlisted case did not run: {binary} {test}")
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"required-case gate: {len(problems)} problem(s) for {label}", file=sys.stderr)
        return 1
    parked_note = f", {len(parked)} parked" if parked else ""
    print(
        f"required cases ({label}): {len(rows)} ran and passed, {len(unlisted)} deliberately unlisted{parked_note}"
    )
    return 0


FIXTURE_LIST_AB = "poot-wgpu-device::fixture\talpha_case\npoot-wgpu-device::fixture\tbeta_case\n"

FIXTURE_GREEN = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="device-fixture" tests="2" failures="0" errors="0">
  <testsuite name="poot-wgpu-device::fixture" tests="2" disabled="0" errors="0" failures="0">
    <testcase name="alpha_case" classname="poot-wgpu-device::fixture" time="0.001"/>
    <testcase name="beta_case" classname="poot-wgpu-device::fixture" time="0.001"/>
  </testsuite>
</testsuites>
"""

FIXTURE_ALPHA_ONLY = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="device-fixture" tests="1" failures="0" errors="0">
  <testsuite name="poot-wgpu-device::fixture" tests="1" disabled="0" errors="0" failures="0">
    <testcase name="alpha_case" classname="poot-wgpu-device::fixture" time="0.001"/>
  </testsuite>
</testsuites>
"""


def fixture_with_alpha(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<testsuites name="device-fixture" tests="2" failures="0" errors="0">\n'
        '  <testsuite name="poot-wgpu-device::fixture" tests="2" disabled="0" errors="0" failures="0">\n'
        f'    <testcase name="alpha_case" classname="poot-wgpu-device::fixture" time="0.001">\n'
        f"{body}\n"
        "    </testcase>\n"
        '    <testcase name="beta_case" classname="poot-wgpu-device::fixture" time="0.001"/>\n'
        "  </testsuite>\n"
        "</testsuites>\n"
    )


def run_fixture(
    list_text: str,
    results_xml: str | None,
    unlisted_text: str | None = "",
    run: "BoundRun | None" = None,
    parked_text: str | None = None,
) -> tuple[int, str]:
    """Run the gate against fixture files; returns its exit status and captured stderr.

    unlisted_text None leaves the lane without an unlisted list. run binds the results file to a lane run: its
    start marker and the age of the results relative to it. parked_text None leaves the lane without a parked
    list."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        list_path = directory / "required.txt"
        list_path.write_text(list_text, encoding="ascii")
        results_path = directory / "results.xml"
        if run is not None:
            begin(run.lane, results_path)
        if results_xml is not None:
            results_path.write_text(results_xml, encoding="ascii")
        if run is not None and results_xml is not None:
            marker_ns = marker_path(results_path).stat().st_mtime_ns
            os.utime(results_path, ns=(marker_ns + run.results_offset_ns,) * 2)
        unlisted_path = directory / "unlisted.txt"
        if unlisted_text is not None:
            unlisted_path.write_text(unlisted_text, encoding="ascii")
        parked_path = directory / "parked.txt"
        if parked_text is not None:
            parked_path.write_text(parked_text, encoding="ascii")
        captured = io.StringIO()
        with redirect_stderr(captured), redirect_stdout(io.StringIO()):
            status = check(
                list_path, results_path, "fixture", unlisted_path, run.lane if run else None, parked_path
            )
    return status, captured.getvalue()


def run_checkpoint_fixture(
    list_text: str, results_xml: str, lane: str, run: "BoundRun | None" = None
) -> tuple[int, str]:
    """Run the gate for a checkpoint-style lane (rows `lane<TAB>binary_id<TAB>test_name`, no unlisted list)."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        list_path = directory / "required-checkpoint-cases.txt"
        list_path.write_text(list_text, encoding="ascii")
        results_path = directory / "results.xml"
        if run is not None:
            begin(run.lane, results_path)
        results_path.write_text(results_xml, encoding="ascii")
        if run is not None:
            marker_ns = marker_path(results_path).stat().st_mtime_ns
            os.utime(results_path, ns=(marker_ns + run.results_offset_ns,) * 2)
        captured = io.StringIO()
        with redirect_stderr(captured), redirect_stdout(io.StringIO()):
            status = check(list_path, results_path, lane, None, run.lane if run else None, None, lane)
    return status, captured.getvalue()


class BoundRun:
    """A fixture lane run: the lane the gate is told it ran, and how far the results file's mtime sits from the
    start marker (negative: written before the run began)."""

    def __init__(self, lane: str, results_offset_ns: int) -> None:
        self.lane = lane
        self.results_offset_ns = results_offset_ns


def expect_rejection(
    label: str,
    needle: str,
    list_text: str,
    results_xml: str | None,
    unlisted_text: str | None = "",
    run: BoundRun | None = None,
    parked_text: str | None = None,
) -> bool:
    """The gate must fail the fixture and name the mismatch; a pass here means a mutation lives."""
    status, output = run_fixture(list_text, results_xml, unlisted_text, run, parked_text)
    if status == 0:
        print(f"required-case mutation unexpectedly passed: {label}", file=sys.stderr)
        return False
    if needle not in output:
        print(f"required-case check did not name its mismatch: {label} (wanted {needle})", file=sys.stderr)
        print(output, file=sys.stderr)
        return False
    print(f"required-case check rejected: {label} ({needle})")
    return True


def self_test() -> int:
    status, output = run_fixture(FIXTURE_LIST_AB, FIXTURE_GREEN)
    if status != 0:
        print("required-case check rejected a green fixture", file=sys.stderr)
        print(output, file=sys.stderr)
        return 1
    print("required-case check accepted: every required case ran and passed")

    # Free text that mentions a skip is ordinary output, not a skip report (card 1325): the ptx-check host tests
    # print `add: SKIP (not staged)` for a fixture they deliberately leave out, and a lane must pass them.
    for ordinary in (
        "  add: SKIP (not staged)",
        "SKIP: wgpu device unavailable: no adapter",
        "skip: no wgpu adapter",
        "SKIP case (no GPU agent)",
        "skipping the dispatch",
        "foo_skips_bar",
    ):
        status, output = run_fixture(
            FIXTURE_LIST_AB, fixture_with_alpha(f"      <system-out>{ordinary}\n</system-out>")
        )
        if status != 0:
            print(f"required-case check read ordinary output as a skip: {ordinary!r}", file=sys.stderr)
            print(output, file=sys.stderr)
            return 1
    print("required-case check accepted: free text mentioning a skip is ordinary output")

    cases = [
        (
            "required case absent from the results",
            "required case did not run: poot-wgpu-device::fixture beta_case",
            FIXTURE_LIST_AB,
            FIXTURE_ALPHA_ONLY,
        ),
        (
            "runtime skip reported on stderr",
            "required case skipped: poot-wgpu-device::fixture alpha_case: POOT-SKIP: wgpu device unavailable: no adapter",
            FIXTURE_LIST_AB,
            fixture_with_alpha("      <system-err>POOT-SKIP: wgpu device unavailable: no adapter\n</system-err>"),
        ),
        (
            "runtime skip reported after a crate prefix",
            "required case skipped: poot-wgpu-device::fixture alpha_case: POOT-SKIP poot-wgpu-device: no adapter",
            FIXTURE_LIST_AB,
            fixture_with_alpha("      <system-err>  POOT-SKIP poot-wgpu-device: no adapter\n</system-err>"),
        ),
        (
            "runtime skip reported on stdout",
            "required case skipped: poot-wgpu-device::fixture alpha_case: POOT-SKIP case (no GPU agent)",
            FIXTURE_LIST_AB,
            fixture_with_alpha("      <system-out>POOT-SKIP case (no GPU agent)\n</system-out>"),
        ),
        (
            "testcase the runner skipped",
            "required case skipped: poot-wgpu-device::fixture alpha_case: reported skipped by the runner",
            FIXTURE_LIST_AB,
            fixture_with_alpha("      <skipped/>"),
        ),
        (
            "required case failed",
            "required case failed: poot-wgpu-device::fixture alpha_case: assertion left == right failed",
            FIXTURE_LIST_AB,
            fixture_with_alpha(
                '      <failure type="test failure">assertion left == right failed\n</failure>'
            ),
        ),
        (
            "lane wrote no results",
            "no lane results at",
            FIXTURE_LIST_AB,
            None,
        ),
        (
            "empty required list",
            "required-case list is empty",
            "# only a comment\n",
            FIXTURE_GREEN,
        ),
        (
            "duplicate required row",
            "required-case list has a duplicate row: poot-wgpu-device::fixture alpha_case",
            "poot-wgpu-device::fixture\talpha_case\npoot-wgpu-device::fixture\talpha_case\n",
            FIXTURE_GREEN,
        ),
        (
            "unsorted required rows",
            "required-case list is not in canonical sorted order",
            "poot-wgpu-device::fixture\tbeta_case\npoot-wgpu-device::fixture\talpha_case\n",
            FIXTURE_GREEN,
        ),
        (
            "malformed required row",
            "required-case list malformed at line 1",
            "poot-wgpu-device::fixture alpha_case\n",
            FIXTURE_GREEN,
        ),
    ]
    for label, needle, list_text, results_xml in cases:
        if not expect_rejection(label, needle, list_text, results_xml):
            return 1

    # Exhaustive classification (card 1125): the same two-row results, with only alpha required.
    only_alpha = "poot-wgpu-device::fixture\talpha_case\n"
    beta_unlisted = "poot-wgpu-device::fixture\tbeta_case\thost-only fixture row\n"
    status, output = run_fixture(only_alpha, FIXTURE_GREEN, beta_unlisted)
    if status != 0:
        print("required-case check rejected a fully classified fixture", file=sys.stderr)
        print(output, file=sys.stderr)
        return 1
    print("required-case check accepted: every passing row required or unlisted with a reason")
    classification = [
        (
            "passing row in neither list",
            "unclassified passing row (list it in",
            only_alpha,
            FIXTURE_GREEN,
            "",
        ),
        (
            "unlisted row that did not run",
            "unlisted case did not run: poot-wgpu-device::fixture beta_case",
            only_alpha,
            FIXTURE_ALPHA_ONLY,
            beta_unlisted,
        ),
        (
            "row both required and unlisted",
            "row is both required and unlisted: poot-wgpu-device::fixture alpha_case",
            FIXTURE_LIST_AB,
            FIXTURE_GREEN,
            "poot-wgpu-device::fixture\talpha_case\treason\n",
        ),
        (
            "unlisted row without a reason",
            "unlisted-case list malformed at line 1",
            only_alpha,
            FIXTURE_GREEN,
            "poot-wgpu-device::fixture\tbeta_case\n",
        ),
        (
            "unlisted row with an empty reason",
            "unlisted-case list malformed at line 1",
            only_alpha,
            FIXTURE_GREEN,
            "poot-wgpu-device::fixture\tbeta_case\t \n",
        ),
    ]
    for label, needle, list_text, results_xml, unlisted_text in classification:
        if not expect_rejection(label, needle, list_text, results_xml, unlisted_text):
            return 1
    # Unlisted rows are no escape from a skip, in any lane (card 1218, SC-002): a wgpu row that printed POOT-SKIP
    # fails the gate though the lane's unlisted list names it.
    skipped_beta = FIXTURE_GREEN.replace(
        '<testcase name="beta_case" classname="poot-wgpu-device::fixture" time="0.001"/>',
        '<testcase name="beta_case" classname="poot-wgpu-device::fixture" time="0.001">'
        "<system-err>POOT-SKIP: wgpu device unavailable: no adapter\n</system-err></testcase>",
    )
    if not expect_rejection(
        "unlisted passing row that printed SKIP",
        "unlisted case skipped: poot-wgpu-device::fixture beta_case: POOT-SKIP: wgpu device unavailable: no adapter",
        only_alpha,
        skipped_beta,
        beta_unlisted,
    ):
        return 1
    if not expect_rejection(
        "lane without an unlisted list",
        "unlisted-case list missing",
        only_alpha,
        FIXTURE_GREEN,
        None,
    ):
        return 1

    # A parked row (POOT-1454): ignored, not run, named with the card that makes it pass.
    parked_beta = "poot-wgpu-device::fixture\tbeta_case\tPOOT-9999: misses its ceiling until the guard lands\n"
    status, output = run_fixture(only_alpha, FIXTURE_ALPHA_ONLY, "", None, parked_beta)
    if status != 0:
        print("required-case check rejected a parked row that did not run", file=sys.stderr)
        print(output, file=sys.stderr)
        return 1
    print("required-case check accepted: a parked row that did not run")
    parking = [
        # Mutation: drop the parked-and-passed check in `check`; the row reads as parked forever after its fix.
        ("a parked row that passed", "parked case passed", only_alpha, FIXTURE_GREEN, parked_beta),
        ("a row both required and parked", "row is both required and parked", FIXTURE_LIST_AB, FIXTURE_GREEN, parked_beta),
        (
            "a parked row that names no card",
            "names no card",
            only_alpha,
            FIXTURE_ALPHA_ONLY,
            "poot-wgpu-device::fixture\tbeta_case\tfails for now\n",
        ),
    ]
    for label, needle, list_text, results_xml, parked_text in parking:
        if not expect_rejection(label, needle, list_text, results_xml, "", None, parked_text):
            return 1

    # The results file is bound to its run (card 1218, SC-001).
    status, output = run_fixture(FIXTURE_LIST_AB, FIXTURE_GREEN, "", BoundRun("fixture", 1_000_000))
    if status != 0:
        print("required-case check rejected a results file written after the lane began", file=sys.stderr)
        print(output, file=sys.stderr)
        return 1
    print("required-case check accepted: results newer than the lane start")
    binding = [
        (
            "results file older than the lane start",
            "lane results are stale",
            BoundRun("fixture", -1_000_000),
        ),
        (
            "results written by another lane's profile",
            "lane results are foreign",
            BoundRun("other", 1_000_000),
        ),
    ]
    for label, needle, run in binding:
        if not expect_rejection(label, needle, FIXTURE_LIST_AB, FIXTURE_GREEN, "", run):
            return 1
    if not checkpoint_lane_self_test():
        return 1
    print("required-case gate self-test: pass")
    return 0


def checkpoint_lane_self_test() -> bool:
    """The checkpoint-style lanes (POOT-1467): only the lane's own rows of the shared list are required, and a
    skip, an absent row, a failed row or an unrequired row that ran fails the gate."""
    alpha = "poot-wgpu-device::fixture\talpha_case"
    both_lanes = f"checkpoint\t{alpha}\ncheckpoint-large\tpoot-wgpu-device::fixture\tbeta_case\n"
    status, output = run_checkpoint_fixture(both_lanes, FIXTURE_ALPHA_ONLY, "checkpoint")
    if status != 0:
        print("checkpoint-lane check rejected a lane that ran exactly its rows", file=sys.stderr)
        print(output, file=sys.stderr)
        return False
    print("required-case check accepted: a checkpoint lane that ran exactly its own rows")
    rejections = [
        (
            "checkpoint lane row skipped",
            "required case skipped: poot-wgpu-device::fixture alpha_case: POOT-SKIP: llama-perplexity is not on PATH",
            fixture_with_alpha("      <system-err>POOT-SKIP: llama-perplexity is not on PATH\n</system-err>"),
            both_lanes,
            None,
        ),
        (
            "checkpoint lane row absent",
            "required case did not run: poot-wgpu-device::fixture beta_case",
            FIXTURE_ALPHA_ONLY,
            f"checkpoint\t{alpha}\ncheckpoint\tpoot-wgpu-device::fixture\tbeta_case\n",
            None,
        ),
        (
            "checkpoint lane ran a row the lane does not require",
            "unclassified passing row (list it in",
            FIXTURE_GREEN,
            f"checkpoint\t{alpha}\n",
            None,
        ),
        (
            "checkpoint lane results written by another profile",
            "lane results are foreign",
            FIXTURE_ALPHA_ONLY,
            f"checkpoint\t{alpha}\n",
            BoundRun("checkpoint", 1_000_000),
        ),
        (
            "lane with no row in the list",
            "required-case list is empty",
            FIXTURE_ALPHA_ONLY,
            "checkpoint-large\tpoot-wgpu-device::fixture\tbeta_case\n",
            None,
        ),
    ]
    for label, needle, results_xml, list_text, run in rejections:
        status, output = run_checkpoint_fixture(list_text, results_xml, "checkpoint", run)
        if status == 0:
            print(f"required-case mutation unexpectedly passed: {label}", file=sys.stderr)
            return False
        if needle not in output:
            print(f"required-case check did not name its mismatch: {label} (wanted {needle})", file=sys.stderr)
            print(output, file=sys.stderr)
            return False
        print(f"required-case check rejected: {label} ({needle})")
    return True


def main(argv: list[str]) -> int:
    usage = f"usage: {Path(sys.argv[0]).name} --self-test | begin <lane> <results.xml> | check <lane> <results.xml>"
    if argv == ["--self-test"]:
        return self_test()
    if len(argv) == 3 and argv[0] in ("begin", "check") and not any(a.startswith("-") for a in argv[1:]):
        command, lane, results = argv
        results_path = Path(results)
        if command == "begin":
            return begin(lane, results_path)
        if lane in CHECKPOINT_STYLE_LANES:
            return check(REQUIRED_CHECKPOINTS, results_path, lane, None, lane, None, lane)
        return check(
            REQUIRED_DIR / f"{lane}.txt",
            results_path,
            lane,
            REQUIRED_DIR / f"{lane}.unlisted.txt",
            lane,
            REQUIRED_DIR / f"{lane}.parked.txt",
        )
    print(usage, file=sys.stderr)
    return 0 if argv in (["--help"], ["-h"]) else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
