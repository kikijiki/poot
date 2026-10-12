#!/usr/bin/env python3
"""Find `pub` items that nothing uses.

rustc's `dead_code` lint skips every item reachable from a library crate root, so an unused `pub fn` passes
`cargo check` and `cargo clippy` with no warning. This check closes that gap for every workspace crate, the
benchmark runner included.

Cargo builds a library, each bin, each integration test, each example and each bench as its own crate. An item
declared with a bare `pub` (fn, struct, enum, union, trait, type, const, static, or an fn or const in an inherent
impl) is dead when its name is referenced nowhere except:

- its own definition and the impl blocks that define it,
- `pub use` re-exports (a re-export is not a use),
- code inside another dead item (a dead item keeps nothing alive; the scan runs to a fixpoint).

A reference from test code is never a use (ruling DP1), in any crate. Test code is a `#[cfg(test)]` item or the
out-of-line modules it declares, an item gated by `feature = "test-support"` (POOT-ADR-0113: the sanctioned gate
for test-only API that needs private state), and every `tests/`, `examples/` and `benches/` target, `#[ignore]`d
tests included. A crate that every workspace dependent takes only as a dev-dependency (`poot-test-util`, the
parity fixtures) is test infrastructure: its own source is test code for the crates it references, and a test
that uses one of its items keeps that item alive.

Every other reference counts: a crate's non-test code, its `src/bin`, a build script, or another production
crate. The scan is name based: a name shared with another item reads as used, so it can miss a dead item but
never invents one. Only whole identifier tokens count; comments and string contents are not references. Items
marked `#[kernel]`, `#[test]`, `#[no_mangle]` and the like are reachable by other means and are skipped.

Two more checks fail the scan:

- a bare `#[cfg(test)]` item (or `cfg(all(test, ...))`) outside a test-module tree: `#[cfg(test)]` code lives in
  a `#[cfg(test)] mod`, whose file and submodules are the tree (raudit2 R9). `cfg(any(test, ...))` and
  `feature = "test-support"` gates are not bare.
- a bare `#[cfg(test)]` below item level in a production file: on an enum variant, a struct field, a match arm
  or a statement (`let`, expression) of a production fn. Such code compiles in the tested build and not in the
  shipped one, so production would differ from what the tests exercise. Make the code identical in both and
  inject the test behaviour through a trait or a test-support type; scaffolding that is genuinely test-only
  moves into a `#[cfg(test)] mod` tree. The gate is the bare `cfg(test)` only: an ADR-0113 `test-support`
  gate is the sanctioned one and is not reported here either.
- a `test-support` feature reachable from a normal or build edge of a production crate (`cargo tree`), which
  would put test-only API into a release build (POOT-ADR-0113).

Limits, all toward missing a dead item: an item whose name is common (`new`, `run`, `len`, ...) reads as used
because some other item shares the name; enum variants, struct fields, trait methods, `pub(crate)` items and
`pub use` glob targets are not examined; a name that appears only in a macro-generated identifier is not seen.
`#[cfg(any(test, feature = "x"))]` and `#[cfg(not(test))]` code counts as production code.

There is no allow-list and no exemption: every dead `pub` item the scan finds fails the check.

Usage: dead_pub.py [--root DIR]   check the workspace; exit 1 on a finding
       dead_pub.py --self-test    run the built-in fixtures
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ITEM_KEYWORDS = {"fn", "struct", "enum", "union", "trait", "type", "const", "static"}
DECL_KEYWORDS = ITEM_KEYWORDS | {"mod", "let", "impl"}
# Sentinel `SourceFile.crate` for a crate's `tests/kernels/*.rs` (never a real Cargo package name): real,
# deployed usage from outside cargo's normal target graph, so it is exempt from the dependency check below.
KERNEL_SOURCE_CRATE = "<kernel-source>"
# Attributes that make an item reachable by something other than a Rust path.
EXPORT_ATTRS = {
    "kernel",
    "test",
    "bench",
    "no_mangle",
    "export_name",
    "proc_macro",
    "proc_macro_derive",
    "proc_macro_attribute",
    "ctor",
}
IDENT_START = re.compile(r"[A-Za-z_]")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CHAR_LITERAL = re.compile(r"'(?:\\.[^']*|[^\\'])'")
RAW_STRING_OPEN = re.compile(r'b?r(#*)"')


@dataclass
class Tok:
    kind: str  # 'id', 'p' (punctuation), 's' (string content)
    text: str
    line: int


def tokenize(src: str) -> list[Tok]:
    """Rust source to identifier, punctuation and string tokens. Comments, numbers and lifetimes are dropped."""
    toks: list[Tok] = []
    i, n, line = 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
        elif c.isspace():
            i += 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            depth = 1
            i += 2
            while i < n and depth:
                if src.startswith("/*", i):
                    depth += 1
                    i += 2
                elif src.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    line += src[i] == "\n"
                    i += 1
        elif (m := RAW_STRING_OPEN.match(src, i)) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            close = '"' + m.group(1)
            j = src.find(close, m.end())
            j = n if j < 0 else j
            toks.append(Tok("s", src[m.end() : j], line))
            line += src.count("\n", i, j)
            i = min(n, j + len(close))
        elif c == '"' or (c == "b" and src.startswith('b"', i)):
            i += 1 if c == '"' else 2
            start = i
            while i < n and src[i] != '"':
                if src[i] == "\\":
                    i += 1
                i += 1
            toks.append(Tok("s", src[start:i], line))
            line += src.count("\n", start, i)
            i += 1
        elif c == "'":
            m = CHAR_LITERAL.match(src, i)
            if m:
                i = m.end()
            else:  # a lifetime: drop the quote, the name is skipped as an identifier below
                i += 1
                m = IDENT.match(src, i)
                i = m.end() if m else i
        elif c == "b" and src.startswith("b'", i) and (m := CHAR_LITERAL.match(src, i + 1)):
            i = m.end()
        elif IDENT_START.match(c):
            m = IDENT.match(src, i)
            text = m.group(0)
            i = m.end()
            if text == "r" and src.startswith("#", i) and i + 1 < n and IDENT_START.match(src[i + 1]):
                m = IDENT.match(src, i + 1)  # raw identifier r#name
                text, i = m.group(0), m.end()
            toks.append(Tok("id", text, line))
        elif c.isdigit():
            while i < n and (src[i].isalnum() or src[i] == "_"):
                i += 1
            if i + 1 < n and src[i] == "." and src[i + 1].isdigit():
                i += 1
                while i < n and (src[i].isalnum() or src[i] == "_"):
                    i += 1
        else:
            two = src[i : i + 2]
            if two in ("->", "=>"):
                toks.append(Tok("p", two, line))
                i += 2
            else:
                toks.append(Tok("p", c, line))
                i += 1
    return toks


def match_brackets(toks: list[Tok]) -> dict[int, int]:
    closers = {"{": "}", "(": ")", "[": "]"}
    out: dict[int, int] = {}
    stack: list[int] = []
    for i, t in enumerate(toks):
        if t.kind != "p":
            continue
        if t.text in closers:
            stack.append(i)
        elif t.text in closers.values() and stack:
            open_i = stack.pop()
            out[open_i] = i
    return out


@dataclass
class Item:
    crate: str
    path: str
    line: int
    kind: str
    name: str
    start: int  # token span in its file
    end: int
    exported_type: bool  # struct/enum/union/trait/type: impl blocks that define it are part of the item

    def key(self) -> str:
        return f"{self.path} {self.kind} {self.name}"


@dataclass
class ImplBlock:
    start: int
    end: int
    self_name: str
    trait_name: str


@dataclass
class ModDecl:
    name: str
    in_test: bool
    dir: Path
    path_attr: str | None


@dataclass
class SourceFile:
    crate: str
    path: Path
    rel: str
    target: Path  # the crate root this file is compiled into; cargo builds tests, examples and bins as their own crates
    toks: list[Tok]
    exports: bool = True  # false in tests, examples and benches: their `pub` items are not an API
    is_test: bool = False
    test_ranges: list[tuple[int, int]] = field(default_factory=list)
    reexport_ranges: list[tuple[int, int]] = field(default_factory=list)
    impls: list[ImplBlock] = field(default_factory=list)
    mods: list[ModDecl] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    build_script: bool = False  # a build script is compiled and run by cargo: its references are uses
    bare_cfg_test: list[tuple[int, str]] = field(default_factory=list)  # (line, "kind name") outside a test tree
    # (line, shape) of a test-only `cfg` attribute below item level in production code: an enum variant, a struct
    # field, a match arm, a statement. Filled by `scan_member_cfg_test` once `test_ranges` is known.
    member_cfg_test: list[tuple[int, str]] = field(default_factory=list)


def attr_text(toks: list[Tok], open_i: int, close_i: int) -> list[str]:
    return [t.text for t in toks[open_i + 1 : close_i] if t.kind == "id"]


def cfg_test_only(toks: list[Tok], open_i: int, close_i: int) -> bool:
    """Whether the `#[cfg(...)]` attribute in toks[open_i + 1 : close_i] compiles only in test code: `test` or
    `feature = "test-support"` (ADR-0113), through `all` (one operand suffices) and `any` (every operand
    must). `not(...)` and any other predicate compile outside tests."""
    i = open_i + 1
    if close_i - i < 3 or toks[i].text != "cfg" or toks[i + 1].text != "(":
        return False

    def predicate(i: int) -> tuple[bool, int]:
        t = toks[i]
        if t.text in ("all", "any", "not") and toks[i + 1].text == "(":
            operands: list[bool] = []
            i += 2
            while toks[i].text != ")":
                value, i = predicate(i)
                operands.append(value)
                if toks[i].text == ",":
                    i += 1
            i += 1
            if t.text == "all":
                return any(operands), i
            return (t.text == "any" and bool(operands) and all(operands)), i
        if t.text == "test":
            return True, i + 1
        if toks[i + 1].text == "=" and toks[i + 2].kind == "s":
            return (t.text == "feature" and toks[i + 2].text == "test-support"), i + 3
        return False, i + 1

    return predicate(i + 2)[0]


def mentions_test_support(toks: list[Tok], open_i: int, close_i: int) -> bool:
    return any(t.kind == "s" and t.text == "test-support" for t in toks[open_i + 1 : close_i])


def is_export_attr(words: list[str]) -> bool:
    if not words:
        return False
    if words[0] in EXPORT_ATTRS:
        return True
    if words[0] == "unsafe":  # #[unsafe(no_mangle)]
        return any(w in EXPORT_ATTRS for w in words[1:])
    return words[-1] in EXPORT_ATTRS  # #[tokio::test]


def impl_names(toks: list[Tok], start: int, brace: int) -> tuple[str, str]:
    """(self type name, trait name) of an impl header spanning toks[start:brace]."""
    i = start + 1
    if toks[i].text == "<":  # skip the impl generics
        depth = 0
        while i < brace:
            depth += toks[i].text == "<"
            depth -= toks[i].text == ">"
            i += 1
            if depth == 0:
                break
    segments: list[list[str]] = [[]]
    depth = 0
    for j in range(i, brace):
        t = toks[j]
        if t.text == "where" and depth == 0:
            break
        if t.text == "<":
            depth += 1
        elif t.text == ">":
            depth -= 1
        elif t.text == "for" and depth == 0:
            segments.append([])
        elif depth == 0 and t.kind == "id" and t.text not in ("dyn", "mut", "const"):
            segments[-1].append(t.text)
    last = lambda seg: seg[-1] if seg else ""  # noqa: E731
    if len(segments) > 1:
        return last(segments[1]), last(segments[0])
    return last(segments[0]), ""


def parse_zone(
    f: SourceFile, match: dict[int, int], i: int, end: int, zone: str, in_test: bool, mdir: Path
) -> None:
    """Walk the items of one file, inline module or inherent impl body in toks[i:end]."""
    toks = f.toks
    while i < end:
        attrs: list[list[str]] = []
        gates: list[str] = []  # per attribute: "bare" (cfg(test)), "support" (test-support gate) or ""
        first = i
        while i < end and toks[i].text == "#":
            j = i + 1
            inner = j < end and toks[j].text == "!"
            j += inner
            if j < end and toks[j].text == "[":
                words = attr_text(toks, j, match[j])
                gate = ""
                if cfg_test_only(toks, j, match[j]):
                    gate = "support" if mentions_test_support(toks, j, match[j]) else "bare"
                if inner and zone == "file" and gate:
                    f.is_test = True
                attrs.append(words)
                gates.append(gate)
                i = match[j] + 1
            else:
                break
        if i >= end:
            break
        head = i
        j = i
        while j < end and toks[j].text not in ("{", ";"):
            if toks[j].text in ("(", "["):
                j = match.get(j, j)
            j += 1
        terminator = j
        # visibility
        k = head
        bare_pub = False
        is_pub = False
        if k < end and toks[k].text == "pub":
            is_pub = True
            k += 1
            if k < end and toks[k].text == "(":
                k = match[k] + 1
            else:
                bare_pub = True
        while k < end and toks[k].text in ("async", "unsafe", "default", "extern", "safe") or (
            k + 1 < end and toks[k].text == "const" and toks[k + 1].text in ("fn", "unsafe", "async", "extern")
        ):
            k += 1
            if k < end and toks[k].kind == "s":
                k += 1
        keyword = toks[k].text if k < end and toks[k].kind == "id" else ""
        name = toks[k + 1].text if keyword and k + 1 < end and toks[k + 1].kind == "id" else ""

        # the end of the item
        if keyword in ("const", "static", "type", "use") or terminator >= end:
            e = head
            while e < end and toks[e].text != ";":
                if toks[e].text in ("{", "(", "["):
                    e = match.get(e, e)
                e += 1
            item_end = min(e, end - 1)
        elif toks[terminator].text == "{":
            item_end = match.get(terminator, end - 1)
        else:
            item_end = terminator

        test = in_test or any(gates) or any(a and a[0] == "test" for a in attrs)
        if test:
            f.test_ranges.append((first, item_end))
        if "bare" in gates and not in_test and keyword != "mod":
            f.bare_cfg_test.append((toks[head].line, f"{keyword or 'item'} {name}".strip()))

        if keyword == "use" and is_pub:
            f.reexport_ranges.append((head, item_end))
        elif keyword == "mod" and name:
            if terminator >= end or toks[terminator].text == ";":
                f.mods.append(ModDecl(name, test, mdir, mod_path_attr(toks, first, head)))
            else:
                parse_zone(f, match, terminator + 1, item_end, "mod", test, mdir / name)
        elif keyword == "impl":
            brace = terminator if terminator < end and toks[terminator].text == "{" else None
            if brace is not None:
                self_name, trait_name = impl_names(toks, k, brace)
                f.impls.append(ImplBlock(head, item_end, self_name, trait_name))
                if not trait_name:
                    parse_zone(f, match, brace + 1, item_end, "impl", test, mdir)
        elif keyword in ITEM_KEYWORDS and name and name not in ("_", "main") and bare_pub and not test:
            if zone in ("file", "mod", "impl") and not any(is_export_attr(a) for a in attrs):
                kind = keyword
                f.items.append(
                    Item(
                        f.crate,
                        f.rel,
                        toks[k + 1].line,
                        "method" if zone == "impl" and keyword == "fn" else kind,
                        name,
                        head,
                        item_end,
                        keyword in ("struct", "enum", "union", "trait", "type"),
                    )
                )
        i = item_end + 1


ITEM_START = ITEM_KEYWORDS | {"use", "mod", "impl", "macro_rules", "extern"}


def enclosing_openers(toks: list[Tok], match: dict[int, int]) -> dict[int, int]:
    """Token index -> index of the innermost bracket that encloses it (absent at file level)."""
    parent: dict[int, int] = {}
    stack: list[int] = []
    closers = set(match.values())
    for i, t in enumerate(toks):
        if stack:
            parent[i] = stack[-1]
        if i in closers and stack:
            stack.pop()
            parent[i] = stack[-1] if stack else -1
        elif i in match:
            stack.append(i)
    return {i: p for i, p in parent.items() if p >= 0}


def header_words(toks: list[Tok], opener: int) -> set[str]:
    """The identifiers between the previous statement or block boundary and the bracket at `opener`."""
    words: set[str] = set()
    j = opener - 1
    while j >= 0 and toks[j].text not in (";", "{", "}"):
        if toks[j].kind == "id":
            words.add(toks[j].text)
        j -= 1
    return words


def member_shape(toks: list[Tok], match: dict[int, int], parent: dict[int, int], idx: int) -> str:
    """What the attribute at toks[idx] decorates, below item level: `match arm`, `enum variant`, `struct field`,
    `statement` (a `let`, an expression or an item written in a block), or `parameter or argument`."""
    j = idx
    while j < len(toks) and toks[j].text == "#" and toks[j + 1].text == "[":
        j = match[j + 1] + 1
    k = j  # the element after its attributes: an arm ends at `=>` before any `,` or `;`
    while k < len(toks) and toks[k].text not in (",", ";", "=>", "}", ")", "]"):
        k = match[k] + 1 if toks[k].text in ("{", "(", "[") and k in match else k + 1
    if k < len(toks) and toks[k].text == "=>":
        return "match arm"
    opener = parent.get(idx)
    if opener is not None:
        words = header_words(toks, opener)
        if toks[opener].text == "{" and "enum" in words:
            return "enum variant"
        if "struct" in words or "union" in words:
            return "struct field"
        if toks[opener].text == "(":
            return "parameter or argument"
    return "statement"


def scan_member_cfg_test(f: SourceFile, match: dict[int, int]) -> None:
    """Every bare test-only `#[cfg(...)]` (no `test-support` gate: ADR-0113 sanctions that) that parse_zone did not
    see as an item attribute: an enum variant, a struct field, a match arm or a statement of a production file. Such code compiles in the tested build and not in the
    shipped one, so production would differ from what the tests exercise. Item attributes sit inside a recorded
    test range, so only what lies outside every range is a member."""
    toks = f.toks
    parent = enclosing_openers(toks, match)
    i = 0
    while i + 1 < len(toks):
        if toks[i].text == "#" and toks[i + 1].text == "[" and not in_ranges(f.test_ranges, i):
            if cfg_test_only(toks, i + 1, match[i + 1]) and not mentions_test_support(toks, i + 1, match[i + 1]):
                f.member_cfg_test.append((toks[i].line, member_shape(toks, match, parent, i)))
            i = match[i + 1]
        i += 1


def mod_path_attr(toks: list[Tok], first: int, head: int) -> str | None:
    """The string of a `#[path = "..."]` attribute among toks[first:head], if any."""
    for i in range(first, head - 2):
        if toks[i].text == "path" and toks[i + 1].text == "=" and toks[i + 2].kind == "s":
            return toks[i + 2].text
    return None


def kernel_source_files(crate_dir: Path) -> list[Path]:
    """`.rs` files a crate's own tests feed to `pootc` as standalone kernel source (`pootc --crate-type
    lib <path>`, its own subprocess invocation, compiled outside cargo's normal target graph -- never a
    `mod`, so `load_workspace`'s target/mod walk never sees them): every file under the crate's own
    `tests/kernels/` directory (pootc's own compiler fixtures) and `kernels/` directory (card 559: the
    shipped kernel sources, one family subdirectory each, regenerated into committed assets), if it has
    either. Not hardcoded to one crate: any crate that grows such a directory is picked up the same way
    (today only `pootc` has them)."""
    dirs = (crate_dir / "tests" / "kernels", crate_dir / "kernels")
    return sorted(
        p for d in dirs if d.is_dir() for p in d.rglob("*.rs") if p.is_file()
    )


def target_roots(crate_dir: Path, manifest: dict) -> list[tuple[Path, bool]]:
    """Crate roots cargo compiles, each with whether its `pub` items are an API: autodiscovered targets plus
    explicit `path` overrides. Tests, examples and benches consume an API and export none."""
    roots: dict[Path, bool] = {}

    def add(rel: str, api: bool) -> None:
        p = crate_dir / rel
        if p.is_file():
            roots.setdefault(p, api)

    def add_dir(rel: str, api: bool) -> None:
        d = crate_dir / rel
        if not d.is_dir():
            return
        for child in sorted(d.iterdir()):
            if child.suffix == ".rs" and child.is_file():
                roots.setdefault(child, api)
            elif (child / "main.rs").is_file():
                roots.setdefault(child / "main.rs", api)

    add("src/lib.rs", True)
    add("src/main.rs", True)
    add("build.rs", False)
    add_dir("src/bin", True)
    for rel in ("tests", "examples", "benches"):
        add_dir(rel, False)
    if "path" in manifest.get("lib", {}):
        add(manifest["lib"]["path"], True)
    for table, api in (("bin", True), ("test", False), ("example", False), ("bench", False)):
        for target in manifest.get(table, []):
            if "path" in target:
                add(target["path"], api)
    return list(roots.items())


def transitive_closure(direct: dict[str, set[str]]) -> dict[str, set[str]]:
    """Every crate's direct dependencies, closed over transitively."""
    closure = {k: set(v) for k, v in direct.items()}
    changed = True
    while changed:
        changed = False
        for name, deps in closure.items():
            grown = deps | {d for dep in deps for d in closure.get(dep, ())}
            if grown != deps:
                closure[name] = grown
                changed = True
    return closure


@dataclass
class Graph:
    """The workspace's crate graph, read from `cargo metadata` rather than a hand parse of each `Cargo.toml`: a
    manifest table key is the dependency's local alias, not necessarily its real crate name (`foo = { package =
    "real-name", ... }`), and a `[target.'cfg(...)'.dependencies]` table sits outside the three top-level tables
    a manual TOML walk checks. `dependencies[].name` is always the real name; `--no-deps` still reports each
    workspace member's own declared dependencies, target-gated ones included, without resolving or fetching the
    external graph."""

    direct: dict[str, set[str]]  # every direct dependency name (normal, dev and build)
    # (dependent, dependency) -> the kinds of the edge that count as production: "normal" or "build". A dev
    # edge, and an optional dependency only a `test-support` feature enables, are test edges and are absent.
    production_edges: set[tuple[str, str]]

    def dev_only(self) -> set[str]:
        """Test infrastructure: crates with workspace dependents that no production crate takes as a normal or
        build dependency. A dev-only crate's own normal edges do not make its dependencies production (the
        greatest fixpoint: `poot-executor-parity` is dev-only and so is the `poot-test-util` it depends on)."""
        dependents = {d for deps in self.direct.values() for d in deps if d in self.direct}
        dev = set(dependents)
        changed = True
        while changed:
            changed = False
            for dependent, dependency in self.production_edges:
                if dependency in dev and dependent not in dev:
                    dev.discard(dependency)
                    changed = True
        return dev


def cargo_metadata_graph(root: Path) -> Graph:
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--manifest-path", str(root / "Cargo.toml")],
        capture_output=True,
        text=True,
        check=True,
    )
    packages = json.loads(out.stdout)["packages"]
    names = {pkg["name"] for pkg in packages}
    direct: dict[str, set[str]] = {}
    production: set[tuple[str, str]] = set()
    for pkg in packages:
        name = pkg["name"]
        features: dict[str, list[str]] = pkg["features"]
        direct[name] = {dep["name"] for dep in pkg["dependencies"]}
        for dep in pkg["dependencies"]:
            if dep["kind"] == "dev" or dep["name"] not in names or dep["name"] == name:
                continue
            alias = dep.get("rename") or dep["name"]
            if dep["optional"] and optional_only_for_test_support(features, alias):
                continue
            production.add((name, dep["name"]))
    return Graph(direct, production)


def optional_only_for_test_support(features: dict[str, list[str]], alias: str) -> bool:
    """Whether every feature that enables the optional dependency `alias` is the `test-support` feature."""
    enablers = [
        feature
        for feature, values in features.items()
        if any(v in (alias, f"dep:{alias}") or v.startswith(f"{alias}/") or v.startswith(f"{alias}?/") for v in values)
    ]
    return bool(enablers) and all(feature == "test-support" for feature in enablers)


def load_workspace(root: Path) -> tuple[list[SourceFile], Graph]:
    """Every source file cargo compiles in the workspace (each target root and the modules it declares),
    and each workspace crate's dependency closure (direct + transitive, from `cargo metadata`): a same-named
    item in a crate that never depends on the declaring crate cannot be what keeps it alive (F7)."""
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    files: dict[Path, SourceFile] = {}
    graph = cargo_metadata_graph(root)
    for member in workspace["workspace"]["members"]:
        crate_dir = root / member
        manifest = tomllib.loads((crate_dir / "Cargo.toml").read_text())
        crate = manifest["package"]["name"]
        # a crate root's modules sit beside it
        pending = [(r.resolve(), r.parent, r.resolve(), api) for r, api in target_roots(crate_dir, manifest)]
        while pending:
            path, mdir, target, api = pending.pop()
            if path in files:
                continue
            rel = str(path.relative_to(root.resolve()))
            f = SourceFile(
                crate,
                path,
                rel,
                target,
                tokenize(path.read_text(encoding="utf-8")),
                exports=api,
                build_script=target.name == "build.rs",
            )
            brackets = match_brackets(f.toks)
            parse_zone(f, brackets, 0, len(f.toks), "file", False, mdir)
            scan_member_cfg_test(f, brackets)
            files[path] = f
            for decl in f.mods:
                for child, child_dir in resolve_mod(f, decl):
                    pending.append((child.resolve(), child_dir, target, api))
        # A consumer, not a target: tokenized only for its occurrences (own_spans/find_dead never draw
        # candidates from it -- exports=False and it is never parsed with parse_zone, so it has no items).
        # `crate` is a sentinel no real Cargo package name can equal (never "<...>"), not the owning
        # crate's own name: a kernel is real, deployed usage, never exempted as "this crate's own test
        # code" even when its own crate is the one whose items it references.
        for path in kernel_source_files(crate_dir):
            if path in files:
                continue
            rel = str(path.relative_to(root.resolve()))
            files[path] = SourceFile(
                KERNEL_SOURCE_CRATE, path, rel, path, tokenize(path.read_text(encoding="utf-8")), exports=False
            )
    ordered = sorted(files.values(), key=lambda f: f.rel)
    propagate_test_files(ordered)
    return ordered, graph


def resolve_mod(f: SourceFile, decl: ModDecl) -> list[tuple[Path, Path]]:
    """The file an out-of-line `mod` declaration names, with the directory its own submodules live in."""
    if decl.path_attr:
        target = f.path.parent / decl.path_attr
        return [(target, target.parent)] if target.is_file() else []
    flat = decl.dir / f"{decl.name}.rs"
    nested = decl.dir / decl.name / "mod.rs"
    if flat.is_file():
        return [(flat, decl.dir / decl.name)]
    if nested.is_file():
        return [(nested, decl.dir / decl.name)]
    return []


def propagate_test_files(files: list[SourceFile]) -> None:
    """Mark the files that `#[cfg(test)] mod x;` declarations pull in, and every module below them."""
    by_path = {f.path: f for f in files}
    work = [(f, m) for f in files for m in f.mods if m.in_test or f.is_test]
    seen: set[Path] = set()
    while work:
        f, decl = work.pop()
        for child_path, _ in resolve_mod(f, decl):
            child = by_path.get(child_path.resolve())
            if child is None or child.path in seen:
                continue
            seen.add(child.path)
            child.is_test = True
            work.extend((child, m) for m in child.mods)


def in_ranges(ranges: list[tuple[int, int]], idx: int) -> bool:
    return any(a <= idx <= b for a, b in ranges)


def find_dead(files: list[SourceFile], graph: Graph) -> list[Item]:
    depends = transitive_closure(graph.direct)
    dev_only = graph.dev_only()
    occurrences: dict[str, list[tuple[int, int]]] = {}
    for fi, f in enumerate(files):
        for ti, t in enumerate(f.toks):
            if t.kind == "id":
                occurrences.setdefault(t.text, []).append((fi, ti))

    candidates = [(fi, it) for fi, f in enumerate(files) if f.exports and not f.is_test for it in f.items]
    dead_spans: dict[int, list[tuple[int, int]]] = {}
    dead: dict[tuple[int, int], Item] = {}

    def own_spans(fi: int, it: Item) -> list[tuple[int, int]]:
        spans = [(it.start, it.end)]
        if it.exported_type:
            spans += [
                (b.start, b.end) for b in files[fi].impls if it.name in (b.self_name, b.trait_name)
            ]
        return spans

    def referenced(fi: int, it: Item) -> bool:
        own = own_spans(fi, it)
        for ofi, ti in occurrences[it.name]:
            f = files[ofi]
            if ofi == fi and in_ranges(own, ti):
                continue
            if ti > 0 and f.toks[ti - 1].kind == "id" and f.toks[ti - 1].text in DECL_KEYWORDS:
                continue  # the declaration of another item with the same name
            if in_ranges(f.reexport_ranges, ti) or in_ranges(dead_spans.get(ofi, ()), ti):
                continue
            in_test_code = (
                f.is_test
                or in_ranges(f.test_ranges, ti)
                or (not f.exports and not f.build_script and f.crate != KERNEL_SOURCE_CRATE)
            )
            if in_test_code and (it.crate not in dev_only or f.crate == it.crate):
                continue  # a test reference is never a use: only a test-infrastructure crate's items live on it
            if f.crate in dev_only and it.crate not in dev_only:
                continue  # test infrastructure referencing production API: a test reference
            if (
                f.crate != it.crate
                and f.crate != KERNEL_SOURCE_CRATE
                and it.crate not in depends.get(f.crate, ())
            ):
                continue  # a same-named item in a crate that does not depend on the declaring crate (F7)
            return True
        return False

    changed = True
    while changed:
        changed = False
        for fi, it in candidates:
            key = (fi, it.start)
            if key in dead or referenced(fi, it):
                continue
            dead[key] = it
            dead_spans.setdefault(fi, []).extend(own_spans(fi, it))
            changed = True
    return sorted(dead.values(), key=lambda it: (it.path, it.line))


TEST_SUPPORT_NODE = re.compile(r'([A-Za-z0-9_-]+) feature "test-support"')


def test_support_leaks(root: Path, graph: Graph) -> list[str]:
    """Production crates whose normal or build dependency graph enables a `test-support` feature (ADR-0113: the
    feature is enabled only from `[dev-dependencies]` edges and from test infrastructure). One `cargo tree` per
    production crate: a workspace-wide tree unifies the features its roots enable."""
    leaks: list[str] = []
    for crate in sorted(set(graph.direct) - graph.dev_only()):
        out = subprocess.run(
            [
                "cargo", "tree", "--manifest-path", str(root / "Cargo.toml"), "--offline",
                "-p", crate, "-e", "features", "--edges", "normal,build",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        owners = sorted({m.group(1) for m in TEST_SUPPORT_NODE.finditer(out.stdout)})
        leaks += [
            f"{crate}: the `test-support` feature of `{owner}` is enabled by a normal or build dependency edge"
            for owner in owners
        ]
    return leaks


def bare_cfg_test_violations(files: list[SourceFile]) -> list[str]:
    return [
        f"{f.rel}:{line}: bare #[cfg(test)] {what} outside a test-module tree ({f.crate}): move it into a "
        "`#[cfg(test)] mod`, or gate it `cfg(any(test, feature = \"test-support\"))` (ADR-0113) if another crate's tests need it"
        for f in files
        if f.exports and not f.is_test
        for line, what in f.bare_cfg_test
    ]


def member_cfg_test_violations(files: list[SourceFile]) -> list[str]:
    return [
        f"{f.rel}:{line}: test-only cfg ({shape}) in production code ({f.crate}): the shipped build and the "
        "tested build differ. Make the code identical in both and inject the test behaviour through a trait or a "
        "test-support type; genuine test scaffolding moves into a `#[cfg(test)] mod` tree"
        for f in files
        if f.exports and not f.is_test
        for line, shape in f.member_cfg_test
    ]


def check(root: Path) -> int:
    files, graph = load_workspace(root)
    dead = find_dead(files, graph)
    bare = bare_cfg_test_violations(files) + member_cfg_test_violations(files)
    leaks = test_support_leaks(root, graph)
    for it in dead:
        print(
            f"{it.path}:{it.line}: dead pub {it.kind} `{it.name}` ({it.crate}): used by no production code "
            "of any crate; a test reference is not a use"
        )
    for line in bare + leaks:
        print(line)
    if dead:
        print(
            f"dead-pub: {len(dead)} dead pub item(s). Each must be deleted, made non-pub, or moved into a "
            "test module tree (`poot-test-util` for a cross-crate fixture); an item another crate's tests "
            "need that is not public API stays "
            '`#[cfg(any(test, feature = "test-support"))]` (ADR-0113). There is no hold: an '
            "`expect(dead_code)` attribute does not exempt a dead pub item."
        )
    if bare:
        print(f"dead-pub: {len(bare)} bare #[cfg(test)] item(s) or member(s) outside a test-module tree.")
    if leaks:
        print(f"dead-pub: {len(leaks)} `test-support` feature leak(s) into a normal or build edge.")
    if dead or bare or leaks:
        return 1
    print("dead-pub: no dead pub items, no bare cfg(test) outside a test tree, no test-support leak")
    return 0


# ---- self-test --------------------------------------------------------------------------------------------------


def write_fixture(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


A_LIB = """
pub mod inner;
pub use inner::ReexportedOnly;
pub fn used_by_b() {}
pub fn used_inside() {}
pub fn only_own_test() {}
pub fn used_by_b_integration_test() {} // another crate's test: not a use (DP1)
pub fn only_cfg_test() {}
pub fn only_unit_test_mod() {}
pub fn used_under_any_test_or_feature() {}
pub fn used_under_all_test_and_feature() {}
pub fn unused_probe() {}
#[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-1"))]
pub fn held_but_dead() {} // a hold attribute exempts nothing: there is no allow-list
pub fn used_only_by_kernel_source() {}
pub fn used_only_by_build_script() {}
pub fn used_only_by_test_infrastructure() {} // `t` is a dev-only crate: its source is test code
pub fn cross_crate_shadow() {} // F7: dead, but `c` (which does not depend on `a`) has a same-named fn
pub const DEAD_CONST: u32 = 1; // used only by this crate's own integration test: still dead
pub const USED_CONST: u32 = 2;
pub struct DeadType;
impl DeadType {
    pub fn new() -> Self { DeadType }
    pub fn recurse(&self) { self.recurse() }
}
pub struct LiveType;
impl LiveType {
    pub fn live_method(&self) {}
    pub fn dead_method(&self) {}
}
impl Default for LiveType { fn default() -> Self { LiveType } }
pub fn chain_root_dead() { chain_leaf(); }
pub fn chain_leaf() {}
pub(crate) fn crate_only() {}
#[test]
pub fn exported_by_attr() {}
fn caller() { used_inside(); let _ = b'x'; let _ = '\\''; }
pub fn text_only() { let _ = "unused_probe in a string"; } // unused_probe in a comment
#[cfg(test)]
mod tests;
#[cfg(test)]
mod inline_tests {
    fn t() { super::only_cfg_test(); }
    #[cfg(test)]
    fn inside_the_tree() {} // a test-module tree may nest cfg(test)
}
#[cfg(any(test, feature = "x"))]
mod any_tests {
    fn t() { super::used_under_any_test_or_feature(); }
}
#[cfg(all(test, feature = "x"))]
mod all_tests {
    fn t() { super::used_under_all_test_and_feature(); }
}
#[cfg(test)]
fn bare_test_helper() {} // R9: a bare cfg(test) fn in production code
#[cfg(all(test, feature = "x"))]
pub(crate) struct BareAllTest; // R9
#[cfg(any(test, feature = "test-support"))]
pub fn sanctioned_support() {} // ADR-0113: not bare, and not a production `pub` item
mod prod {
    #[cfg(test)]
    pub(super) fn nested_bare() {} // R9, in an inline production module
    pub struct Holder;
    impl Holder {
        #[cfg(test)]
        pub(crate) fn bare_method(&self) {} // R9, in an inherent impl
    }
}
enum Shaped {
    Plain,
    #[cfg(test)]
    OnlyTested, // POOT-1135: a variant that exists only in the tested build
    #[cfg(not(test))]
    OnlyShipped, // compiles outside tests: production code
    #[cfg(any(test, feature = "x"))]
    Either,
}
struct Fielded {
    plain: u32,
    #[cfg(test)]
    probe: u32, // POOT-1135: a field that exists only in the tested build
    #[cfg(any(test, feature = "test-support"))]
    sanctioned: u32, // ADR-0113: the sanctioned gate, not a bare cfg(test)
}
fn shaped_fn(s: Shaped) -> u32 {
    #[cfg(test)]
    let tested = 1; // POOT-1135: a `let` that exists only in the tested build
    #[cfg(test)]
    bump(); // POOT-1135: an expression statement
    #[cfg(not(test))]
    let shipped = 1;
    match s {
        Shaped::Plain => 0,
        #[cfg(test)]
        Shaped::OnlyTested => 1, // POOT-1135: an arm that exists only in the tested build
        #[cfg(test)]
        Shaped::Struct { .. } => 3, // an arm whose pattern holds braces
        _ => 2,
    }
}
fn takes(#[cfg(test)] extra: u32, plain: u32) {} // POOT-1135: a parameter that exists only in the tested build
macro_rules! scaffold {
    () => {
        #[cfg(test)]
        let from_macro = 1; // POOT-1135: a macro body expands into production code
    };
}
#[cfg(test)]
mod member_tests {
    enum InTree {
        #[cfg(test)]
        A, // inside a test-module tree: not production code
    }
}
"""


def self_test() -> int:
    manifest = '[package]\nname = "{}"\nversion = "0.0.0"\n'
    features = '[features]\ntest-support = []\n'
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_fixture(
            root,
            {
                "Cargo.toml": '[workspace]\nmembers = ["crates/a", "crates/b", "crates/c", "crates/t"]\n',
                "crates/a/Cargo.toml": manifest.format("a") + features,
                # `b` takes `a` as a normal dependency, and `a`'s test-support feature from a dev edge only.
                "crates/b/Cargo.toml": manifest.format("b")
                + '[dependencies]\na = { path = "../a" }\n'
                + '[dev-dependencies]\na = { path = "../a", features = ["test-support"] }\nt = { path = "../t" }\n',
                "crates/c/Cargo.toml": manifest.format("c"),
                # `t` is test infrastructure: only `b` takes it, as a dev-dependency. Its own normal edge
                # may enable `a`'s test-support feature.
                "crates/t/Cargo.toml": manifest.format("t")
                + '[dependencies]\na = { path = "../a", features = ["test-support"] }\n',
                "crates/a/src/lib.rs": A_LIB,
                "crates/a/build.rs": "fn main() { used_only_by_build_script(); }\n",
                "crates/a/src/inner.rs": "pub struct ReexportedOnly;\n",
                "crates/a/src/tests.rs": "fn t() { super::only_unit_test_mod(); }\n#[cfg(test)]\nfn helper() {}\n",
                "crates/a/tests/it.rs": "fn t() { a::only_own_test(); a::DEAD_CONST; }\n",
                "crates/a/tests/kernels/probe.rs": "fn t() { a::used_only_by_kernel_source(); }\n",
                "crates/b/src/main.rs": "fn main() { a::used_by_b(); a::live_method(); let _ = a::USED_CONST; a::LiveType.live_method(); }\n",
                "crates/b/tests/it.rs": "fn t() { a::used_by_b_integration_test(); t::fixture_used_by_b_test(); }\n",
                # `c` depends on neither `a` nor `b`: its own same-named fn must not keep `a::cross_crate_shadow` alive.
                "crates/c/src/lib.rs": "fn cross_crate_shadow() {}\nfn caller() { cross_crate_shadow(); }\n",
                "crates/t/src/lib.rs": "pub fn fixture_used_by_b_test() { a::used_only_by_test_infrastructure(); }\npub fn fixture_nobody_uses() {}\n",
            },
        )
        files, graph = load_workspace(root)
        dead = {it.key() for it in find_dead(files, graph)}
        expected = {
            "crates/a/src/lib.rs fn only_own_test",
            "crates/a/src/lib.rs fn cross_crate_shadow",
            "crates/a/src/lib.rs const DEAD_CONST",
            "crates/a/src/lib.rs fn only_cfg_test",
            "crates/a/src/lib.rs fn only_unit_test_mod",
            "crates/a/src/lib.rs fn used_under_all_test_and_feature",
            "crates/a/src/lib.rs fn used_by_b_integration_test",
            "crates/a/src/lib.rs fn used_only_by_test_infrastructure",
            "crates/a/src/lib.rs fn unused_probe",
            "crates/a/src/lib.rs fn held_but_dead",
            "crates/a/src/lib.rs struct DeadType",
            "crates/a/src/lib.rs method new",
            "crates/a/src/lib.rs method recurse",
            "crates/a/src/lib.rs method dead_method",
            "crates/a/src/lib.rs fn chain_root_dead",
            "crates/a/src/lib.rs fn chain_leaf",
            "crates/a/src/lib.rs fn text_only",
            "crates/a/src/inner.rs struct ReexportedOnly",
            "crates/a/src/lib.rs struct Holder",
            "crates/t/src/lib.rs fn fixture_nobody_uses",
        }
        # `new` on a type nothing uses is dead even though `new` is a common name
        if dead != expected:
            print("self-test FAILED: dead items")
            print("  unexpected:", sorted(dead - expected))
            print("  missing:   ", sorted(expected - dead))
            return 1
        bare = sorted(bare_cfg_test_violations(files))
        wanted = [
            ("crates/a/src/lib.rs", "fn bare_test_helper"),
            ("crates/a/src/lib.rs", "struct BareAllTest"),
            ("crates/a/src/lib.rs", "fn nested_bare"),
            ("crates/a/src/lib.rs", "fn bare_method"),
        ]
        got = sorted((m.group(1), m.group(2)) for m in (re.match(r"([^:]+):\d+: bare #\[cfg\(test\)\] (.+?) outside", v) for v in bare))
        if got != sorted(wanted):
            print("self-test FAILED: bare cfg(test) items")
            print("  unexpected:", sorted(set(got) - set(wanted)))
            print("  missing:   ", sorted(set(wanted) - set(got)))
            return 1
        # POOT-1135: a bare cfg(test) below item level is reported by shape; each shape has its own row, so
        # dropping one shape's detection reds the self-test (the others still pass).
        shapes = sorted(
            m.group(1)
            for v in member_cfg_test_violations(files)
            if (m := re.search(r"test-only cfg \((.+?)\) in production code", v))
        )
        wanted_shapes = sorted(
            [
                "enum variant",
                "struct field",
                "statement",
                "statement",
                "statement",  # the one in the macro_rules body
                "match arm",
                "match arm",
                "parameter or argument",
            ]
        )
        if shapes != wanted_shapes:
            print("self-test FAILED: member-level cfg(test) shapes")
            print("  got:   ", shapes)
            print("  wanted:", wanted_shapes)
            return 1
        leaks = test_support_leaks(root, graph)
        if leaks:
            print("self-test FAILED: a dev edge and a dev-only crate's normal edge are not leaks:", leaks)
            return 1
        # the same feature on a normal edge of a production crate is a leak
        write_fixture(
            root,
            {
                "crates/b/Cargo.toml": manifest.format("b")
                + '[dependencies]\na = { path = "../a", features = ["test-support"] }\n'
                + '[dev-dependencies]\nt = { path = "../t" }\n'
            },
        )
        files, graph = load_workspace(root)
        leaks = test_support_leaks(root, graph)
        if len(leaks) != 1 or not leaks[0].startswith("b: the `test-support` feature of `a`"):
            print("self-test FAILED: a test-support feature on a normal edge must be reported once:", leaks)
            return 1
        write_fixture(
            root,
            {
                "crates/b/Cargo.toml": manifest.format("b")
                + '[dependencies]\na = { path = "../a" }\n'
                + '[dev-dependencies]\na = { path = "../a", features = ["test-support"] }\nt = { path = "../t" }\n'
            },
        )
        write_fixture(root, {"crates/a/src/lib.rs": A_LIB + "pub fn probe_added() {}\n"})
        files, graph = load_workspace(root)
        dead2 = {it.key() for it in find_dead(files, graph)}
        if "crates/a/src/lib.rs fn probe_added" not in dead2:
            print("self-test FAILED: an added unused pub fn was not reported")
            return 1
        keys = sorted({it.key() for it in find_dead(files, graph)})
        # An allow.txt beside the fixture names every dead item. The check has no allow-list, so it ignores
        # the file and still fails, naming the items.
        (root / "allow.txt").write_text(
            "# a stale list the check must ignore\n" + "\n".join(f"{k}  # reason" for k in keys) + "\n"
        )
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = check(root)
        text = out.getvalue()
        if (
            rc != 1
            or "dead pub fn `probe_added`" not in text
            or "bare #[cfg(test)] fn bare_test_helper" not in text
            or "test-only cfg (enum variant) in production code" not in text
        ):
            print(f"self-test FAILED: a present allow.txt must be ignored, rc={rc}")
            return 1
    print("self-test ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--root", type=Path, default=here.parent.parent)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    return check(args.root.resolve())


if __name__ == "__main__":
    sys.exit(main())
