#!/usr/bin/env python3
"""Self-test of `just gate`, the local merge gate (card 1218, SC-005).

`just gate` is `fmt-check`, `clippy` and `test`. This proves the two properties a reviewer cannot see by reading
the recipe once:

  1. the gate runs all three steps, in order, each failing the recipe: `just --dry-run gate` prints the format
     check, the clippy run with `-D warnings` and the model-free lane, in that order;
  2. the format step is not vacuous: the real `fmt-check` recipe, run in a fixture crate with a rustfmt diff, fails,
     and in the same crate formatted, passes.

The clippy step is proven by its command line (1) and by the workspace being clippy clean under it. A fixture cannot
run it: the recipe names workspace features the fixture does not have.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUSTFILE = ROOT / "justfile"

UNFORMATTED = "pub   fn   answer ( )  -> u32 { 42 }\n"
FORMATTED = "pub fn answer() -> u32 {\n    42\n}\n"
MANIFEST = '[package]\nname = "fixture"\nversion = "0.0.0"\nedition = "2024"\n\n[lib]\npath = "src/lib.rs"\n'


def just(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["just", "--justfile", str(JUSTFILE), "--working-directory", str(cwd), *args],
        capture_output=True,
        text=True,
    )


def gate_steps() -> list[str]:
    """The commands `just gate` would run, in order."""
    result = just("--dry-run", "gate", cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"just --dry-run gate failed:\n{result.stderr}")
    return result.stderr.splitlines() + result.stdout.splitlines()


def check_steps() -> int:
    lines = gate_steps()
    wanted = ["cargo fmt --all --check", "-- -D warnings", "scripts/test-model-free.sh run"]
    at = -1
    for needle in wanted:
        found = next((i for i, line in enumerate(lines) if i > at and needle in line), None)
        if found is None:
            print(f"gate: `just gate` does not run `{needle}` after step {at}", file=sys.stderr)
            return 1
        at = found
    print("gate: `just gate` runs fmt-check, clippy -D warnings and the model-free lane, in order")
    return 0


def check_fmt_fixture() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        crate = Path(tmp)
        (crate / "src").mkdir()
        (crate / "Cargo.toml").write_text(MANIFEST, encoding="ascii")
        lib = crate / "src" / "lib.rs"
        lib.write_text(UNFORMATTED, encoding="ascii")
        failing = just("fmt-check", cwd=crate)
        if failing.returncode == 0:
            print("gate: fmt-check passed a tree with a rustfmt diff", file=sys.stderr)
            return 1
        print("gate: fmt-check rejected a tree with a rustfmt diff")
        lib.write_text(FORMATTED, encoding="ascii")
        passing = just("fmt-check", cwd=crate)
        if passing.returncode != 0:
            print(f"gate: fmt-check rejected a formatted tree:\n{passing.stderr}", file=sys.stderr)
            return 1
        print("gate: fmt-check accepted the same tree formatted")
    return 0


def main(argv: list[str]) -> int:
    if argv != ["--self-test"]:
        print("usage: check-gate.py --self-test", file=sys.stderr)
        return 0 if argv in (["--help"], ["-h"]) else 2
    return check_steps() or check_fmt_fixture()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
