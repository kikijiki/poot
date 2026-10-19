#!/usr/bin/env python3
"""Mutation-sample gate (POOT-1221, R2-C5-009).

scripts/mutation-sample.tsv is the committed report of the cargo-mutants sample: how many mutants it
generated, how many the suite caught, and the surviving ones, one per line. The survivor count is a
ratchet:

  check-mutation-sample.py                  the committed report is consistent (runs in `just test`);
  check-mutation-sample.py OUT_DIR...       the shards' results (cargo-mutants `--output` directories)
                                            leave no more survivors than the report commits.

A survivor is a mutant the tests did not catch (`missed`) or a run that timed out and so proved nothing.
When a run has fewer survivors than committed, the gate says so: lower the committed count with the
report so the improvement cannot be lost.

Report format (tab separated): `# ` comments, `key<TAB>value` for sampled, mutants, caught, unviable,
survivors, then `survivor<TAB><mutant description>` lines, one per survivor.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "scripts" / "mutation-sample.tsv"
KEYS = ("mutants", "caught", "unviable", "survivors")


def read_report(path: Path) -> tuple[dict[str, int], list[str]]:
    values: dict[str, int] = {}
    survivors: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, _, rest = line.partition("\t")
        if key == "survivor":
            survivors.append(rest)
        elif key in KEYS:
            values[key] = int(rest)
        elif key != "sampled":
            raise SystemExit(f"{path}: unknown row {line!r}")
    missing = [k for k in KEYS if k not in values]
    if missing:
        raise SystemExit(f"{path}: missing {missing}")
    return values, survivors


def consistency(values: dict[str, int], survivors: list[str]) -> list[str]:
    problems = []
    if len(survivors) != values["survivors"]:
        problems.append(
            f"the report counts {values['survivors']} survivors but lists {len(survivors)}"
        )
    if values["caught"] + values["unviable"] + values["survivors"] != values["mutants"]:
        problems.append(
            "caught + unviable + survivors "
            f"({values['caught']} + {values['unviable']} + {values['survivors']}) "
            f"is not mutants ({values['mutants']})"
        )
    return problems


def lines(path: Path) -> list[str]:
    return [l for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def measured(shards: list[Path]) -> tuple[dict[str, int], list[str]]:
    caught = unviable = 0
    survivors: list[str] = []
    for shard in shards:
        out = shard / "mutants.out"
        if not out.is_dir():
            raise SystemExit(f"{shard}: no mutants.out: the sample did not run")
        caught += len(lines(out / "caught.txt"))
        unviable += len(lines(out / "unviable.txt"))
        survivors += lines(out / "missed.txt") + lines(out / "timeout.txt")
    total = caught + unviable + len(survivors)
    return {
        "mutants": total,
        "caught": caught,
        "unviable": unviable,
        "survivors": len(survivors),
    }, survivors


def gate(report: Path, shards: list[Path]) -> int:
    committed, listed = read_report(report)
    problems = consistency(committed, listed)
    if problems:
        print("mutation-sample: the committed report is inconsistent:")
        for p in problems:
            print(f"  {p}")
        return 1
    if not shards:
        print(
            f"mutation-sample: report consistent: {committed['survivors']} survivors "
            f"of {committed['mutants']} mutants"
        )
        return 0
    run, survivors = measured(shards)
    print(
        f"mutation-sample: measured {run['survivors']} survivors of {run['mutants']} mutants "
        f"(committed bound {committed['survivors']})"
    )
    if run["survivors"] > committed["survivors"]:
        new = sorted(set(survivors) - set(listed))
        print("mutation-sample: survivors rose above the committed count; new survivors:")
        for s in new:
            print(f"  {s}")
        return 1
    if run["survivors"] < committed["survivors"]:
        print("mutation-sample: fewer survivors than committed: lower the report to the measured count")
    return 0


def self_test() -> int:
    failures = 0

    def check(name: str, ok: bool) -> None:
        nonlocal failures
        print(f"  {'ok' if ok else 'FAIL'}: {name}")
        failures += not ok

    def report_text(survivors: int, listed: int) -> str:
        return (
            f"mutants\t10\ncaught\t{9 - survivors}\nunviable\t1\nsurvivors\t{survivors}\n"
            + "".join(f"survivor\tm{i}\n" for i in range(listed))
        )

    def shard(root: Path, name: str, missed: int, caught: int) -> Path:
        out = root / name / "mutants.out"
        out.mkdir(parents=True)
        (out / "missed.txt").write_text("".join(f"m{i}\n" for i in range(missed)))
        (out / "caught.txt").write_text("".join(f"c{i}\n" for i in range(caught)))
        (out / "unviable.txt").write_text("u\n")
        return root / name

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rep = root / "r.tsv"
        rep.write_text(report_text(2, 2))
        check("a consistent report passes", gate(rep, []) == 0)
        rep.write_text(report_text(2, 1))
        check("a report whose count disagrees with its list fails", gate(rep, []) == 1)
        rep.write_text(report_text(2, 2))
        check("a run at the committed count passes", gate(rep, [shard(root, "a", 2, 7)]) == 0)
        check(
            "a run with more survivors than committed fails",
            gate(rep, [shard(root, "b", 3, 6)]) == 1,
        )
        rep.write_text(report_text(1, 1))
        check(
            "a committed bound below the measured count fails",
            gate(rep, [shard(root, "c", 2, 7)]) == 1,
        )
        check(
            "survivors add up across shards",
            gate(rep, [shard(root, "d", 1, 4), shard(root, "e", 1, 3)]) == 1,
        )
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    if argv[1:] == ["--self-test"]:
        print("mutation-sample self-test")
        return self_test()
    return gate(REPORT, [Path(a) for a in argv[1:]])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
