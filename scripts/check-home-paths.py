#!/usr/bin/env python3
"""Keep checkout-specific home paths out of tracked files, and strip the ones already published.

A receipt written on a developer machine records absolute paths (the manifest it read, the binary it ran). They name the
user and the checkout, carry no meaning elsewhere, and do not belong in a public tree. `--check` fails when a tracked
file contains a home-directory path; `--strip` rewrites JSON files, replacing only the path prefix, and proves the
rewrite changed nothing else: the parsed documents must have the same shape, keys, key order and numbers, and each
changed string must be the old string with its prefix substituted.

usage: scripts/check-home-paths.py [--self-test | --check | --strip]
"""
import json
import re
import subprocess
import sys

# Built from pieces so this file does not contain the thing it forbids.
HOME_ROOT = "/ho" + "me/"
MAC_ROOT = "/Us" + "ers/"
FOUND = re.compile(rf"({re.escape(HOME_ROOT)}|{re.escape(MAC_ROOT)})[A-Za-z0-9_.-]+/")

# Prefix substitutions, most specific first. A worktree path becomes repo-relative, a cargo target dir becomes `target/`.
RULES = [
    (re.compile(rf"(?:{re.escape(HOME_ROOT)}|{re.escape(MAC_ROOT)})[A-Za-z0-9_.-]+/dev/[A-Za-z0-9_.-]+/"), ""),
    (re.compile(rf"(?:{re.escape(HOME_ROOT)}|{re.escape(MAC_ROOT)})[A-Za-z0-9_.-]+/\.cache/[A-Za-z0-9_.-]+/"), "target/"),
]


def strip_text(text: str) -> str:
    for pattern, replacement in RULES:
        text = pattern.sub(replacement, text)
    return text


def tracked() -> list[str]:
    return [p for p in subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0") if p]


def read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return None


def equivalent(before, after, changed: list) -> str | None:
    """None when `after` is `before` with only home-path prefixes substituted in string values, else why not."""
    if isinstance(before, dict):
        if not isinstance(after, dict) or list(before) != list(after):
            return "object keys or key order differ"
        for key in before:
            if why := equivalent(before[key], after[key], changed):
                return why
        return None
    if isinstance(before, list):
        if not isinstance(after, list) or len(before) != len(after):
            return "array length differs"
        for b, a in zip(before, after):
            if why := equivalent(b, a, changed):
                return why
        return None
    if isinstance(before, str):
        if not isinstance(after, str):
            return "a string became another type"
        if before != after:
            if not FOUND.search(before) or strip_text(before) != after:
                return f"string changed beyond its path prefix: {before!r} -> {after!r}"
            changed.append((before, after))
        return None
    if type(before) is not type(after) or before != after:
        return f"scalar differs: {before!r} -> {after!r}"
    return None


def strip_json(text: str) -> tuple[str, list]:
    """The stripped text and the (old, new) string pairs, or raises ValueError when the rewrite is not path-only."""
    stripped = strip_text(text)
    changed: list = []
    if why := equivalent(json.loads(text), json.loads(stripped), changed):
        raise ValueError(why)
    return stripped, changed


def offending(files: dict[str, str]) -> list[str]:
    return [path for path, text in files.items() if FOUND.search(text)]


def self_test() -> None:
    home = HOME_ROOT + "someone"
    doc = '{"manifest": "%s/dev/poot-c1/benchmarks/manifest.toml", "n": 1.50, "k": ["%s/.cache/t-2/release/x"]}' % (home, home)
    stripped, changed = strip_json(doc)
    assert stripped == '{"manifest": "benchmarks/manifest.toml", "n": 1.50, "k": ["target/release/x"]}', stripped
    assert len(changed) == 2
    assert offending({"a": doc, "b": stripped}) == ["a"]
    assert offending({"c": "see " + HOME_ROOT + "user/ in prose"}) == ["c"]
    assert offending({"d": "/home is a directory, /homestead too"}) == []

    # The proof refuses a rewrite that touches anything but the prefix.
    for bad in ('{"manifest": "x", "n": 2}', '{"n": 1, "manifest": "x"}', '{"manifest": "other"}', '{"manifest": ["x"]}'):
        assert equivalent(json.loads(doc), json.loads(bad), []) is not None, bad


def main() -> int:
    args = sys.argv[1:]
    if args == ["--self-test"]:
        self_test()
        return 0
    if args == ["--strip"]:
        files = {p: t for p in tracked() if p.endswith(".json") and (t := read_text(p)) is not None and FOUND.search(t)}
        for path, text in files.items():
            stripped, changed = strip_json(text)
            with open(path, "w", encoding="utf-8") as f:
                f.write(stripped)
            print(f"{path}: {len(changed)} string(s) re-prefixed, nothing else changed")
        print(f"{len(files)} file(s) rewritten")
        return 0
    if args == ["--check"]:
        files = {p: t for p in tracked() if (t := read_text(p)) is not None}
        found = offending(files)
        for path in found:
            print(f"{path}: contains a home-directory path; record a repo-relative path (scripts/check-home-paths.py --strip fixes JSON)")
        if found:
            print(f"{len(found)} tracked file(s) contain a home-directory path")
        return 1 if found else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
