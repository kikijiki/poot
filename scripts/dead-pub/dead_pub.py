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
- test code of the crate that declares it -- `#[cfg(test)]` items and the out-of-line modules they declare, and
  that crate's own `tests/`, `examples/` and `benches/` targets -- which needs no `pub` and cannot by itself keep
  an item alive,
- code inside another dead item (a dead item keeps nothing alive; the scan runs to a fixpoint).

Every other reference counts: the crate's own `src/bin`, another workspace crate, or another crate's integration
test, example or bench needs the item to be `pub`, and the crate's own non-test code needs it to exist.
`poot-test-util` is used that way.

The scan is name based: a name shared with another item reads as used, so it can miss a dead item but never
invents one. Only whole identifier tokens count; comments and string contents are not references. Items marked
`#[kernel]`, `#[test]`, `#[no_mangle]` and the like are reachable by other means and are skipped.

Limits, all toward missing a dead item: an item whose name is common (`new`, `run`, `len`, ...) reads as used
because some other item shares the name; enum variants, struct fields, trait methods, `pub(crate)` items and
`pub use` glob targets are not examined; a name that appears only in a macro-generated identifier is not seen.
`#[cfg(any(test, ...))]` and `#[cfg(not(test))]` code counts as production code, so only `cfg(test)` and
`cfg(all(test, ...))` hide a reference.

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


def attr_text(toks: list[Tok], open_i: int, close_i: int) -> list[str]:
    return [t.text for t in toks[open_i + 1 : close_i] if t.kind == "id"]


def is_cfg_test(words: list[str]) -> bool:
    """`cfg(test)` and `cfg(all(test, ...))` only: `any(test, ...)` and `not(test)` also compile outside tests.

    `words` are the identifiers of the attribute, so `cfg(all(test, feature = "x"))` is [cfg, all, test, feature].
    """
    if len(words) < 2 or words[0] != "cfg":
        return False
    if words[1] == "test":
        return True
    return words[1] == "all" and "test" in words and "any" not in words and "not" not in words


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
        first = i
        while i < end and toks[i].text == "#":
            j = i + 1
            inner = j < end and toks[j].text == "!"
            j += inner
            if j < end and toks[j].text == "[":
                words = attr_text(toks, j, match[j])
                if inner and zone == "file" and is_cfg_test(words):
                    f.is_test = True
                attrs.append(words)
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

        test = in_test or any(is_cfg_test(a) or (a and a[0] == "test") for a in attrs)
        if test:
            f.test_ranges.append((first, item_end))

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


def cargo_metadata_direct_deps(root: Path) -> dict[str, set[str]]:
    """Each workspace crate's direct dependency names (normal, dev and build), read from `cargo metadata`
    rather than a hand parse of each `Cargo.toml`: a manifest table key is the dependency's local alias, not
    necessarily its real crate name (`foo = { package = "real-name", ... }`), and a `[target.'cfg(...)'.
    dependencies]` table sits outside the three top-level tables a manual TOML walk checks. `cargo metadata`
    resolves both for free (`dependencies[].name` is always the real name; `--no-deps` still reports each
    workspace member's own declared dependencies, target-gated ones included, without resolving or fetching
    the external graph)."""
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--manifest-path", str(root / "Cargo.toml")],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout)
    return {
        pkg["name"]: {dep["name"] for dep in pkg["dependencies"]}
        for pkg in data["packages"]
    }


def load_workspace(root: Path) -> tuple[list[SourceFile], dict[str, set[str]]]:
    """Every source file cargo compiles in the workspace (each target root and the modules it declares),
    and each workspace crate's dependency closure (direct + transitive, from `cargo metadata`): a same-named
    item in a crate that never depends on the declaring crate cannot be what keeps it alive (F7)."""
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    files: dict[Path, SourceFile] = {}
    direct_deps = cargo_metadata_direct_deps(root)
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
            f = SourceFile(crate, path, rel, target, tokenize(path.read_text(encoding="utf-8")), exports=api)
            parse_zone(f, match_brackets(f.toks), 0, len(f.toks), "file", False, mdir)
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
    return ordered, transitive_closure(direct_deps)


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


def find_dead(files: list[SourceFile], depends: dict[str, set[str]]) -> list[Item]:
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
            if f.crate == files[fi].crate and (not f.exports or f.is_test or in_ranges(f.test_ranges, ti)):
                continue  # this crate's own test code: unit tests, or its own tests/examples/benches targets
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


def check(root: Path) -> int:
    files, depends = load_workspace(root)
    dead = find_dead(files, depends)
    for it in dead:
        print(f"{it.path}:{it.line}: dead pub {it.kind} `{it.name}` ({it.crate}): used by no other crate target and by no non-test code of its own crate")
    if dead:
        print(
            f"dead-pub: {len(dead)} dead pub item(s). "
            "Delete the dead item, or if planned work will use it, mark it held "
            '(`#[expect(dead_code, reason = "held for <tag>")]`, naming the planned work).'
        )
        return 1
    print("dead-pub: no dead pub items")
    return 0


# ---- self-test --------------------------------------------------------------------------------------------------


def write_fixture(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def self_test() -> int:
    manifest = '[package]\nname = "{}"\nversion = "0.0.0"\n'
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_fixture(
            root,
            {
                "Cargo.toml": '[workspace]\nmembers = ["crates/a", "crates/b", "crates/c"]\n',
                "crates/a/Cargo.toml": manifest.format("a"),
                "crates/b/Cargo.toml": manifest.format("b") + '[dependencies]\na = { path = "../a" }\n',
                "crates/c/Cargo.toml": manifest.format("c"),
                "crates/a/src/lib.rs": """
pub mod inner;
pub use inner::ReexportedOnly;
pub fn used_by_b() {}
pub fn used_inside() {}
pub fn only_own_test() {}
pub fn used_by_b_integration_test() {}
pub fn only_cfg_test() {}
pub fn only_unit_test_mod() {}
pub fn used_under_any_test_or_feature() {}
pub fn used_under_all_test_and_feature() {}
pub fn unused_probe() {}
pub fn used_only_by_kernel_source() {}
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
}
#[cfg(any(test, feature = "x"))]
mod any_tests {
    fn t() { super::used_under_any_test_or_feature(); }
}
#[cfg(all(test, feature = "x"))]
mod all_tests {
    fn t() { super::used_under_all_test_and_feature(); }
}
""",
                "crates/a/src/inner.rs": "pub struct ReexportedOnly;\n",
                "crates/a/src/tests.rs": "fn t() { super::only_unit_test_mod(); }\n",
                "crates/a/tests/it.rs": "fn t() { a::only_own_test(); a::DEAD_CONST; }\n",
                "crates/a/tests/kernels/probe.rs": "fn t() { a::used_only_by_kernel_source(); }\n",
                "crates/b/src/main.rs": "fn main() { a::used_by_b(); a::live_method(); let _ = a::USED_CONST; a::LiveType.live_method(); }\n",
                "crates/b/tests/it.rs": "fn t() { a::used_by_b_integration_test(); }\n",
                # `c` depends on neither `a` nor `b`: its own same-named fn must not keep `a::cross_crate_shadow` alive.
                "crates/c/src/lib.rs": "fn cross_crate_shadow() {}\nfn caller() { cross_crate_shadow(); }\n",
            },
        )
        files, depends = load_workspace(root)
        dead = {it.key() for it in find_dead(files, depends)}
        expected = {
            "crates/a/src/lib.rs fn only_own_test",
            "crates/a/src/lib.rs fn cross_crate_shadow",
            "crates/a/src/lib.rs const DEAD_CONST",
            "crates/a/src/lib.rs fn only_cfg_test",
            "crates/a/src/lib.rs fn only_unit_test_mod",
            "crates/a/src/lib.rs fn used_under_all_test_and_feature",
            "crates/a/src/lib.rs fn unused_probe",
            "crates/a/src/lib.rs struct DeadType",
            "crates/a/src/lib.rs method new",
            "crates/a/src/lib.rs method recurse",
            "crates/a/src/lib.rs method dead_method",
            "crates/a/src/lib.rs fn chain_root_dead",
            "crates/a/src/lib.rs fn chain_leaf",
            "crates/a/src/lib.rs fn text_only",
            "crates/a/src/inner.rs struct ReexportedOnly",
        }
        # `new` on a type nothing uses is dead even though `new` is a common name
        ok = dead == expected
        if not ok:
            print("self-test FAILED")
            print("  unexpected:", sorted(dead - expected))
            print("  missing:   ", sorted(expected - dead))
            return 1
        write_fixture(root, {"crates/a/src/lib.rs": (root / "crates/a/src/lib.rs").read_text() + "pub fn probe_added() {}\n"})
        files, depends = load_workspace(root)
        dead2 = {it.key() for it in find_dead(files, depends)}
        if "crates/a/src/lib.rs fn probe_added" not in dead2:
            print("self-test FAILED: an added unused pub fn was not reported")
            return 1
        files, depends = load_workspace(root)
        keys = sorted({it.key() for it in find_dead(files, depends)})
        # An allow.txt beside the fixture names every dead item. The check has no allow-list, so it ignores
        # the file and still fails, naming the items.
        (root / "allow.txt").write_text(
            "# a stale list the check must ignore\n" + "\n".join(f"{k}  # reason" for k in keys) + "\n"
        )
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = check(root)
        text = out.getvalue()
        if rc != 1 or "dead pub fn `probe_added`" not in text:
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
