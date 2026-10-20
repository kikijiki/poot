#!/usr/bin/env python3
"""Fail when a workspace library does not build on its own (card 1296).

`cargo check --workspace` unifies features across every member, so a crate that only compiles because a sibling
turned one of its features on (an `Arc` import gated behind `test-support` while production code used it) passes
it. `cargo check -p <crate> --lib` unifies only that crate's own dependency closure: the build a downstream
`cargo test -p <other>` sees. This runs it once per workspace library.
"""

from __future__ import annotations

import json
import subprocess
import sys


def libraries() -> list[str]:
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    members = set(meta["workspace_members"])
    return sorted(
        pkg["name"]
        for pkg in meta["packages"]
        if pkg["id"] in members and any({"lib", "rlib", "proc-macro"} & set(t["kind"]) for t in pkg["targets"])
    )


def main() -> int:
    failed = []
    for name in libraries():
        result = subprocess.run(["cargo", "check", "-q", "-p", name, "--lib"], capture_output=True, text=True)
        if result.returncode != 0:
            failed.append(name)
            print(f"crates-standalone: {name} does not build alone:\n{result.stderr}", file=sys.stderr)
    if failed:
        print(f"crates-standalone: FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("crates-standalone: every workspace library builds alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
