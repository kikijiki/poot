#!/usr/bin/env python3
"""Required-case gate for the `just test-device-*` lanes (card 670).

Each lane has a hand-maintained list, scripts/required-device-cases/<lane>.txt - one line per
required case, `binary_id<TAB>test_name` (the same identity JUnit reports as `classname` and
`name`), committed as text and never generated from the lane's selection. After the lane's
nextest run, this gate compares that list with the run's actual results
(`target/nextest/device-<lane>/results.xml`, written by the lane profile in .config/nextest.toml)
and fails when a required case did not run, failed, or was skipped. The two lists are
independently sourced, so a comparison that read its required cases from the run's own results
passes trivially; `--self-test` proves both of the card's mutations red against fixture results.

A skip is never a pass. nextest reports a runtime skip as a passing testcase - the test printed
its skip report and returned - so the gate reads the captured output (`store-success-output = true`
in the lane profile) and classifies a testcase whose output carries a skip report as skipped:

    SKIP: wgpu device unavailable: no adapter      (DeviceBackend::skip_unavailable)
    skip: no wgpu adapter                          (a guard that opens without the requirement)
    poot-kernelgen: SKIP case (no GPU agent)       (a runtime's own report)

A skip report is a line where the word `skip` (any case) is followed by `:` or whitespace and is
not part of a longer word; `skipping ...` and `foo_skips_bar` are ordinary output. A `<skipped>`
testcase (a test the runner itself did not run) is a skip too.
"""

from __future__ import annotations

import io
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_DIR = ROOT / "scripts" / "required-device-cases"

SKIP_REPORT = re.compile(r"(?:^|[^A-Za-z0-9_])skip(?::|[ \t])", re.IGNORECASE)


class Problem(Exception):
    """A results-file problem: reported as the gate's own failure message."""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_required_list(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """The required rows of a hand-maintained list, plus every schema problem in it."""
    if not path.is_file():
        return [], [f"required-case list missing: {rel(path)}"]
    rows: list[tuple[str, str]] = []
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    text = path.read_text(encoding="ascii")
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            problems.append(
                f"required-case list malformed at line {lineno}: expected binary_id<TAB>test_name: {line}"
            )
            continue
        row = (parts[0], parts[1])
        if row in seen:
            problems.append(f"required-case list has a duplicate row: {row[0]} {row[1]}")
        seen.add(row)
        rows.append(row)
    if not rows and not problems:
        problems.append(f"required-case list is empty: {rel(path)} names no test")
    data_lines = [f"{binary}\t{test}" for binary, test in rows]
    if data_lines != sorted(data_lines):
        problems.append(f"required-case list is not in canonical sorted order: {rel(path)}")
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


def load_results(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(binary id, test name) -> (passed|failed|skipped, detail) from the lane's JUnit report.

    Testcases appear in execution order; a later entry for the same identity wins (the lanes
    configure no retries, so each identity appears once).
    """
    if not path.is_file():
        raise Problem(f"no lane results at {rel(path)}")
    try:
        tree = ET.parse(path)
    except ET.ParseError as error:
        raise Problem(f"lane results are not valid XML: {rel(path)}: {error}") from error
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


def check(list_path: Path, results_path: Path, label: str) -> int:
    """Fail unless every required case in list_path ran and passed in results_path."""
    rows, problems = parse_required_list(list_path)
    results: dict[tuple[str, str], tuple[str, str]] | None = None
    if not problems:
        try:
            results = load_results(results_path)
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
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"required-case gate: {len(problems)} problem(s) for {label}", file=sys.stderr)
        return 1
    print(f"required cases ({label}): {len(rows)} ran and passed")
    return 0


FIXTURE_LIST_AB = "poot-gpu::fixture\talpha_case\npoot-gpu::fixture\tbeta_case\n"

FIXTURE_GREEN = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="fixture" tests="2" failures="0" errors="0">
  <testsuite name="poot-gpu::fixture" tests="2" disabled="0" errors="0" failures="0">
    <testcase name="alpha_case" classname="poot-gpu::fixture" time="0.001"/>
    <testcase name="beta_case" classname="poot-gpu::fixture" time="0.001"/>
  </testsuite>
</testsuites>
"""

FIXTURE_ALPHA_ONLY = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="fixture" tests="1" failures="0" errors="0">
  <testsuite name="poot-gpu::fixture" tests="1" disabled="0" errors="0" failures="0">
    <testcase name="alpha_case" classname="poot-gpu::fixture" time="0.001"/>
  </testsuite>
</testsuites>
"""


def fixture_with_alpha(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<testsuites name="fixture" tests="2" failures="0" errors="0">\n'
        '  <testsuite name="poot-gpu::fixture" tests="2" disabled="0" errors="0" failures="0">\n'
        f'    <testcase name="alpha_case" classname="poot-gpu::fixture" time="0.001">\n'
        f"{body}\n"
        "    </testcase>\n"
        '    <testcase name="beta_case" classname="poot-gpu::fixture" time="0.001"/>\n'
        "  </testsuite>\n"
        "</testsuites>\n"
    )


def run_fixture(list_text: str, results_xml: str | None) -> tuple[int, str]:
    """Run the gate against fixture files; returns its exit status and captured stderr."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        list_path = directory / "required.txt"
        list_path.write_text(list_text, encoding="ascii")
        results_path = directory / "results.xml"
        if results_xml is not None:
            results_path.write_text(results_xml, encoding="ascii")
        captured = io.StringIO()
        with redirect_stderr(captured), redirect_stdout(io.StringIO()):
            status = check(list_path, results_path, "fixture")
    return status, captured.getvalue()


def expect_rejection(label: str, needle: str, list_text: str, results_xml: str | None) -> bool:
    """The gate must fail the fixture and name the mismatch; a pass here means a mutation lives."""
    status, output = run_fixture(list_text, results_xml)
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

    cases = [
        (
            "required case absent from the results",
            "required case did not run: poot-gpu::fixture beta_case",
            FIXTURE_LIST_AB,
            FIXTURE_ALPHA_ONLY,
        ),
        (
            "runtime skip reported on stderr",
            "required case skipped: poot-gpu::fixture alpha_case: SKIP: wgpu device unavailable: no adapter",
            FIXTURE_LIST_AB,
            fixture_with_alpha("      <system-err>SKIP: wgpu device unavailable: no adapter\n</system-err>"),
        ),
        (
            "runtime skip reported in the lowercase form",
            "required case skipped: poot-gpu::fixture alpha_case: skip: no wgpu adapter",
            FIXTURE_LIST_AB,
            fixture_with_alpha("      <system-err>skip: no wgpu adapter\n</system-err>"),
        ),
        (
            "runtime skip reported on stdout",
            "required case skipped: poot-gpu::fixture alpha_case: SKIP case (no GPU agent)",
            FIXTURE_LIST_AB,
            fixture_with_alpha("      <system-out>SKIP case (no GPU agent)\n</system-out>"),
        ),
        (
            "testcase the runner skipped",
            "required case skipped: poot-gpu::fixture alpha_case: reported skipped by the runner",
            FIXTURE_LIST_AB,
            fixture_with_alpha("      <skipped/>"),
        ),
        (
            "required case failed",
            "required case failed: poot-gpu::fixture alpha_case: assertion left == right failed",
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
            "required-case list has a duplicate row: poot-gpu::fixture alpha_case",
            "poot-gpu::fixture\talpha_case\npoot-gpu::fixture\talpha_case\n",
            FIXTURE_GREEN,
        ),
        (
            "unsorted required rows",
            "required-case list is not in canonical sorted order",
            "poot-gpu::fixture\tbeta_case\npoot-gpu::fixture\talpha_case\n",
            FIXTURE_GREEN,
        ),
        (
            "malformed required row",
            "required-case list malformed at line 1",
            "poot-gpu::fixture alpha_case\n",
            FIXTURE_GREEN,
        ),
    ]
    for label, needle, list_text, results_xml in cases:
        if not expect_rejection(label, needle, list_text, results_xml):
            return 1
    print("required-case gate self-test: pass")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    if len(argv) != 1 or argv[0].startswith("-"):
        print(f"usage: {Path(sys.argv[0]).name} <lane>|--self-test", file=sys.stderr)
        return 2
    lane = argv[0]
    list_path = REQUIRED_DIR / f"{lane}.txt"
    results_path = ROOT / "target" / "nextest" / f"device-{lane}" / "results.xml"
    return check(list_path, results_path, lane)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))