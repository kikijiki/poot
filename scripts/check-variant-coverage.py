#!/usr/bin/env python3
"""Variant-coverage gate (POOT-1221, R2-C5-002).

An error or refusal variant that no test names is a refusal nobody has watched fire. The committed list
scripts/variant-coverage.tsv maps each variant of the registered enums to the test that covers it, and
this gate fails when the list and the tree disagree:

  * a registered enum has a variant with no row (a new variant must name its test, or be an explicit gap);
  * a row names a variant the enum no longer has;
  * a `covered` row's test does not exist (deleted or renamed), is not a #[test], or no longer names
    the variant in its body;
  * a `gap` row's variant is now named by some test (promote the row to covered).

An optional path-prefix list scopes the tests that may cover (or reveal a gap for) an enum whose name
another crate also uses.

"Covered" means only that the named #[test] fn exists and its body names the variant (as a word, comments
excluded). It does not prove the test makes that variant happen: a test that merely mentions the name, or
matches it in an assertion that cannot fail, passes this gate. Each covering test carries its own
mutation-checked refusal; this gate keeps the list from rotting.

The list is text, committed by hand: the rows are not generated from the run, so a deleted test cannot
take its own requirement with it. `--suggest` prints candidate rows for the registered enums to start
or extend the list; it never writes the file.

Format (tab separated; `#` starts a comment):

    @enum   <Enum>  <file defining it>  [<path prefix>,...]
    <Enum>  <Variant>   covered <file>  <test fn>
    <Enum>  <Variant>   gap     -   -

`--self-test` runs the gate against fixture trees and requires each failure above to be reported.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "scripts" / "variant-coverage.tsv"

TEST_FN = re.compile(r"#\[test\][^\n]*\n(?:[ \t]*#\[[^\n]*\]\n)*[ \t]*(?:pub )?fn (\w+)")


def blank(src: str, keep_strings: bool = False) -> str:
    """`src` with comments, string and char literals replaced by spaces, so braces and words inside
    them neither unbalance a body nor count as naming a variant. Offsets and newlines are kept.
    With `keep_strings` only the comments go: a test may name a variant through its Debug text."""
    out = list(src)
    i, n = 0, len(src)

    def wipe(a: int, b: int, literal: bool = False) -> None:
        if literal and keep_strings:
            return
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            wipe(i, j)
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            wipe(i, j)
            i = j
        elif c == "r" and re.match(r'r#*"', src[i:]):
            hashes = len(re.match(r"r(#*)", src[i:]).group(1))
            end = src.find('"' + "#" * hashes, i + 2 + hashes)
            j = n if end < 0 else end + 1 + hashes
            wipe(i, j, literal=True)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            wipe(i, j + 1, literal=True)
            i = j + 1
        elif c == "'":
            m = re.match(r"'(?:\\.[^']*|[^\\'])'", src[i:])
            if m:
                wipe(i, i + m.end(), literal=True)
                i += m.end()
            else:
                i += 1  # a lifetime
        else:
            i += 1
    return "".join(out)


def enum_variants(path: Path, name: str) -> list[str]:
    src = blank(path.read_text())
    m = re.search(r"\benum\s+%s\b[^{]*\{" % re.escape(name), src)
    if not m:
        raise SystemExit(f"variant-coverage: no `enum {name}` in {path}")
    depth, i, start = 1, m.end(), m.end()
    body = []
    while depth:
        c = src[i]
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
        elif depth == 1:
            body.append(c)
        i += 1
    # Attributes were nested brackets, so only their `#` is left at this depth.
    names = re.findall(r"(?:^|,)[\s#]*([A-Z]\w*)", "".join(body))
    return names


def test_bodies(path: Path) -> dict[str, str]:
    """Every `#[test]` fn of `path` -> its body, comments removed and string literals kept."""
    raw = path.read_text()
    code = blank(raw)
    text = blank(raw, keep_strings=True)
    out: dict[str, str] = {}
    for m in TEST_FN.finditer(code):
        brace = code.find("{", m.end())
        if brace < 0:
            continue
        depth, j = 0, brace
        while j < len(code):
            if code[j] == "{":
                depth += 1
            elif code[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out[m.group(1)] = text[brace : j + 1]
    return out


def names(body: str, variant: str) -> bool:
    return re.search(r"\b%s\b" % re.escape(variant), body) is not None


def tracked_rust(root: Path) -> list[Path]:
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "*.rs"], capture_output=True, text=True, check=True
        ).stdout.split()
        return [root / f for f in listed if (root / f).exists()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sorted(root.rglob("*.rs"))


def read_list(path: Path) -> tuple[list[tuple[str, str, tuple[str, ...]]], list[tuple[str, str, str, str, str]]]:
    enums, rows = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0] == "@enum" and len(cols) in (3, 4):
            scope = tuple(cols[3].split(",")) if len(cols) == 4 else ("",)
            enums.append((cols[1], cols[2], scope))
        elif len(cols) == 5 and cols[2] in ("covered", "gap"):
            rows.append((cols[0], cols[1], cols[2], cols[3], cols[4]))
        else:
            raise SystemExit(f"{path}:{number}: malformed row {line!r}")
    return enums, rows


def check(root: Path, list_path: Path) -> list[str]:
    problems: list[str] = []
    enums, rows = read_list(list_path)
    declared = {name: variants_of(root, name, file, problems) for name, file, _ in enums}
    scopes = {name: scope for name, _, scope in enums}
    listed = {(e, v) for e, v, *_ in rows}
    for name, variants in declared.items():
        for v in variants:
            if (name, v) not in listed:
                problems.append(f"{name}::{v} has no row: name its covering test, or list it as a gap")
    cache: dict[Path, dict[str, str]] = {}
    for enum, variant, status, file, test in rows:
        if enum not in declared:
            problems.append(f"{enum}::{variant}: {enum} is not a registered @enum")
            continue
        if variant not in declared[enum]:
            problems.append(f"{enum}::{variant}: {enum} has no such variant any more")
            continue
        if status != "covered":
            continue
        path = root / file
        if not path.exists():
            problems.append(f"{enum}::{variant}: covering file {file} does not exist")
            continue
        bodies = cache.setdefault(path, test_bodies(path))
        if test not in bodies:
            problems.append(f"{enum}::{variant}: no #[test] fn {test} in {file}")
        elif not names(bodies[test], variant):
            problems.append(f"{enum}::{variant}: {file}::{test} no longer names {variant}")
    gaps = {(e, v) for e, v, s, *_ in rows if s == "gap" and e in declared and v in declared[e]}
    if gaps:
        found = {}
        for path in tracked_rust(root):
            text = path.read_text()
            rel = str(path.relative_to(root))
            wanted = [
                (e, v) for e, v in gaps if e in text and rel.startswith(scopes[e])
            ]
            if not wanted:
                continue
            for test, body in test_bodies(path).items():
                for e, v in wanted:
                    if (e, v) not in found and re.search(
                        r"\b%s::%s\b" % (re.escape(e), re.escape(v)), body
                    ):
                        found[(e, v)] = f"{path.relative_to(root)}::{test}"
        for (e, v), where in sorted(found.items()):
            problems.append(f"{e}::{v} is listed as a gap but {where} names it: promote the row to covered")
    return problems


def variants_of(root: Path, name: str, file: str, problems: list[str]) -> list[str]:
    path = root / file
    if not path.exists():
        problems.append(f"@enum {name}: {file} does not exist")
        return []
    return enum_variants(path, name)


def suggest(root: Path, list_path: Path) -> None:
    enums, _ = read_list(list_path)
    files = tracked_rust(root)
    bodies = {p: test_bodies(p) for p in files}
    for name, file, scope in enums:
        print("\t".join(["@enum", name, file] + ([",".join(scope)] if scope != ("",) else [])))
        for v in enum_variants(root / file, name):
            hit = None
            for p, tests in bodies.items():
                if not str(p.relative_to(root)).startswith(scope):
                    continue
                for test, body in tests.items():
                    if re.search(r"\b%s::%s\b" % (re.escape(name), re.escape(v)), body):
                        hit = (p.relative_to(root), test)
                        break
                if hit:
                    break
            if hit:
                print(f"{name}\t{v}\tcovered\t{hit[0]}\t{hit[1]}")
            else:
                print(f"{name}\t{v}\tgap\t-\t-")


# ---------------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------------

FIXTURE_ENUM = """
pub enum Fault {
    /// doc { with a brace
    Alpha,
    Beta { at: usize },
    Gamma(u32),
}
"""

FIXTURE_TESTS = """
#[test]
fn alpha_row() {
    let _ = "}"; // a brace in a literal
    assert!(matches!(Fault::Alpha, Fault::Alpha));
}

#[test]
fn beta_row() {
    let _ = Fault::Beta { at: 1 };
}
"""


def self_test() -> int:
    failures = 0

    def expect(name: str, list_text: str, enum_text: str, tests_text: str, needle: str | None) -> None:
        nonlocal failures
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "enum.rs").write_text(enum_text)
            (root / "tests.rs").write_text(tests_text)
            lst = root / "list.tsv"
            lst.write_text(list_text)
            problems = check(root, lst)
        ok = (not problems) if needle is None else any(needle in p for p in problems)
        print(f"  {'ok' if ok else 'FAIL'}: {name}")
        if not ok:
            failures += 1
            print(f"    problems: {problems}")

    head = "@enum\tFault\tenum.rs\n"
    full = (
        head
        + "Fault\tAlpha\tcovered\ttests.rs\talpha_row\n"
        + "Fault\tBeta\tcovered\ttests.rs\tbeta_row\n"
        + "Fault\tGamma\tgap\t-\t-\n"
    )
    expect("a consistent list passes", full, FIXTURE_ENUM, FIXTURE_TESTS, None)
    expect(
        "a deleted covering test is reported",
        full,
        FIXTURE_ENUM,
        FIXTURE_TESTS.replace("fn beta_row", "fn renamed_row"),
        "no #[test] fn beta_row",
    )
    expect(
        "a test that stopped naming its variant is reported",
        full,
        FIXTURE_ENUM,
        FIXTURE_TESTS.replace("Fault::Beta { at: 1 }", "0"),
        "no longer names Beta",
    )
    expect(
        "a variant added to the enum without a row is reported",
        full,
        FIXTURE_ENUM.replace("Gamma(u32),", "Gamma(u32),\n    Delta,"),
        FIXTURE_TESTS,
        "Fault::Delta has no row",
    )
    expect(
        "a row for a removed variant is reported",
        full,
        FIXTURE_ENUM.replace("    Gamma(u32),\n", ""),
        FIXTURE_TESTS,
        "no such variant",
    )
    expect(
        "a gap that a test now names is reported",
        full,
        FIXTURE_ENUM,
        FIXTURE_TESTS + "\n#[test]\nfn gamma_row() {\n    let _ = Fault::Gamma(3);\n}\n",
        "promote the row",
    )
    expect(
        "a name only in a comment does not cover",
        full,
        FIXTURE_ENUM,
        FIXTURE_TESTS.replace("Fault::Beta { at: 1 }", "0 /* Beta */"),
        "no longer names Beta",
    )
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    if argv[1:] == ["--self-test"]:
        print("variant-coverage self-test")
        return self_test()
    if argv[1:] == ["--suggest"]:
        suggest(ROOT, LIST)
        return 0
    if len(argv) != 1:
        print(__doc__)
        return 2
    problems = check(ROOT, LIST)
    enums, rows = read_list(LIST)
    gaps = sum(1 for r in rows if r[2] == "gap")
    if problems:
        print("variant-coverage: the list and the tree disagree:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(
        f"variant-coverage: {len(rows) - gaps} covered, {gaps} listed gaps, "
        f"{len(enums)} enums"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
