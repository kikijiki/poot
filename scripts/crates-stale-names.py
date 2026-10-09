#!/usr/bin/env python3
"""Fail when crate sources or manifests name an API the code no longer has.

The executor-contract work deleted `Runner`, `GpuExecutor`, `run_resident_kv`, `run_prefill`,
`trace_batched_shared_pool_decode` and the `POOT_SPEC_*` variables, but crate descriptions, doc comments and test
messages kept naming them as if they were current. A deleted name may stay only on a comment line that says it is
deleted or gone (a historical note); in code, strings or manifests it is always a leftover.

A comment line must say so itself ("deleted", "gone", ...) - one such word elsewhere in the block does not excuse it.

Run with no arguments to scan the tracked crate sources; `--self-test` proves the rule on synthetic input.
"""
import re
import subprocess
import sys

DELETED = re.compile(
    r"\b(Runner|GpuExecutor|run_resident_kv\w*|run_prefill|trace_batched_shared_pool_decode|POOT_SPEC_\w+)\b"
)
# A comment line (or markdown line) that says in the same line that the name is gone is history, not a claim that
# the name exists.
HISTORY = re.compile(
    r"delet|\bgone\b|no longer|removed|pre-contract|pre-549|never existed|does not exist|There is no",
    re.IGNORECASE,
)
COMMENT = re.compile(r"^\s*(//|#)")


def scan(path, lines):
    """Yield (line number, text) for each deleted name that is not on a historical comment line."""
    for i, line in enumerate(lines):
        match = DELETED.search(line)
        if not match:
            continue
        # Position of a trailing `//` comment, when the line is not a comment line itself.
        trailing = re.search(r"(?<!:)//", line)
        in_comment = (
            path.endswith(".md")
            or bool(COMMENT.match(line))
            or (trailing is not None and trailing.start() < match.start())
        )
        if not in_comment:
            yield i + 1, line.strip()
            continue
        if not HISTORY.search(line):
            yield i + 1, line.strip()


def self_test():
    stale = ["//! Runs `run_resident_kv` on the device.", "let r = Runner::new();", 'description = "twin of GpuExecutor"']
    history = ["// `GpuExecutor::run` was deleted;", "// the deleted `run_prefill` zero-seeded state."]
    assert list(scan("x.rs", ["// the old note.", "// `GpuExecutor::run` hit the cache."])), "history is per line"
    assert list(scan("x.md", ["`Runner` owns the weights."])), "markdown is scanned"
    assert list(scan("x.rs", stale[:1])), "a current-tense doc comment must fail"
    assert list(scan("x.rs", stale[1:2])), "code must fail"
    assert list(scan("x.toml", stale[2:])), "a manifest line must fail"
    assert not list(scan("x.rs", history)), "a historical comment line must pass"
    assert list(scan("x.rs", ["let a = 1; // Runner", "// plain"])), "a trailing comment without history fails"
    print("crates-stale-names: self-test ok")


def main():
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    files = subprocess.run(
        ["git", "ls-files", "-z", "--", "crates/*.rs", "crates/*.toml", "crates/*.md"], capture_output=True, check=True
    ).stdout.decode().split("\0")
    bad = 0
    for path in filter(None, files):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        for number, text in scan(path, lines):
            print(f"{path}:{number}: deleted name still cited as current: {text}")
            bad += 1
    if bad:
        print(f"crates-stale-names: {bad} stale citation(s)")
        sys.exit(1)
    print("crates-stale-names: ok")


if __name__ == "__main__":
    main()
