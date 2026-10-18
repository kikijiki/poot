#!/usr/bin/env python3
"""Check the Backstage architecture guide and overview against this repository.

Four rules (POOT-1213):
  1. Layer table <-> workspace: every workspace member appears in the section 2 layer table, and every crate
     named there is a workspace member.
  2. No stale claims: `architecture.md` contains none of the words that carried the old, false narrative of the
     driver, the per-family runner and the generation 503 (see STALE_WORDS).
  3. Freshness: every `## N.` section of `architecture.md`, and `overview.md`, carries a
     `Checked against: <sha>` line, and that sha is at most MAX_BEHIND commits behind HEAD.
  4. Names resolve: every backticked snake_case name in the guide (`split_stages`, `plan_eqn`) occurs as a word in
     `crates/`, so a function that was renamed or never existed fails the check. Sections 4 and 7 describe deleted
     code by name on purpose and are exempt.

Usage: check-architecture-doc.py [--backstage DIR] [--max-behind N]
       check-architecture-doc.py --self-test
"""

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BACKSTAGE = REPO.parent / "backstage"
MAX_BEHIND = 200
STALE_WORDS = ("Runner", "503", "does not exist")
CHECKED = re.compile(r"^Checked against: `([0-9a-f]{7,40})` on (\d{4}-\d{2}-\d{2})\.?$", re.M)
SNAKE = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?:\(\))?`")
CRATE = re.compile(r"`((?:poot-[a-z0-9-]+)|pootc)`")


def workspace_crates():
    """Package names of every workspace member."""
    members = tomllib.loads((REPO / "Cargo.toml").read_text())["workspace"]["members"]
    names = set()
    for member in members:
        manifest = tomllib.loads((REPO / member / "Cargo.toml").read_text())
        names.add(manifest["package"]["name"])
    return names


def sections(text):
    """Split a guide into (heading, body) pairs on its numbered `## N.` headings."""
    parts = re.split(r"^(## \d+\. .*)$", text, flags=re.M)
    return list(zip(parts[1::2], parts[2::2]))


def layer_table(text):
    for heading, body in sections(text):
        if heading.startswith("## 2."):
            return body
    return None


def check_layers(text, crates):
    body = layer_table(text)
    if body is None:
        return ["architecture.md has no section 2"]
    named = set(CRATE.findall("\n".join(l for l in body.splitlines() if l.startswith("|"))))
    problems = [f"workspace crate missing from the layer table: {c}" for c in sorted(crates - named)]
    problems += [f"layer table names a crate that is not a workspace member: {c}" for c in sorted(named - crates)]
    return problems


def check_stale(text):
    return [
        f"architecture.md line {n}: contains {word!r}; describe the code as it is, not what it is not"
        for n, line in enumerate(text.splitlines(), 1)
        for word in STALE_WORDS
        if word in line
    ]


HISTORY_SECTIONS = ("## 4.", "## 7.")


def check_names(text):
    problems = []
    live = "\n".join(body for heading, body in sections(text) if not heading.startswith(HISTORY_SECTIONS))
    for name in sorted(set(SNAKE.findall(live))):
        found = subprocess.run(["git", "grep", "-qw", "--", name, "crates"], cwd=REPO)
        if found.returncode != 0:
            problems.append(f"architecture.md names `{name}`, which occurs nowhere in crates/")
    return problems


def behind(sha):
    """Commits between sha and HEAD, or None when sha is not an ancestor of HEAD."""
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=REPO, capture_output=True)
    if ancestor.returncode != 0:
        return None
    count = subprocess.run(
        ["git", "rev-list", "--count", f"{sha}..HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return int(count.stdout)


def check_fresh(label, body, max_behind):
    found = CHECKED.findall(body)
    if not found:
        return [f"{label}: no `Checked against: `<sha>` on <date>.` line"]
    problems = []
    for sha, _ in found:
        n = behind(sha)
        if n is None:
            problems.append(f"{label}: checked-against sha {sha} is not an ancestor of HEAD")
        elif n > max_behind:
            problems.append(f"{label}: checked against {sha}, {n} commits behind HEAD (limit {max_behind}); re-read the code")
    return problems


def check(architecture, overview, max_behind):
    crates = workspace_crates()
    problems = check_layers(architecture, crates) + check_stale(architecture) + check_names(architecture)
    for heading, body in sections(architecture):
        problems += check_fresh(heading, body, max_behind)
    problems += check_fresh("overview.md", overview, max_behind)
    return problems


def self_test(architecture, overview):
    """Each rule must turn red on its own mutation of the real documents, and the real ones must be green."""
    assert not check(architecture, overview, MAX_BEHIND), "documents are not green before mutation"
    fake = architecture.replace("`poot-graph-ir`", "`poot-graph-ir`, `poot-fake-crate`", 1)
    assert any("poot-fake-crate" in p for p in check(fake, overview, MAX_BEHIND)), "fake crate not caught"
    dropped = architecture.replace("`poot-eval`", "", 1)
    assert any("poot-eval" in p for p in check(dropped, overview, MAX_BEHIND)), "omitted crate not caught"
    assert check(architecture + "\nThe Runner dispatches by family.\n", overview, MAX_BEHIND), "stale word not caught"
    assert any("poot_nonexistent_fn" in p for p in check(architecture + "\n`poot_nonexistent_fn`\n", overview, MAX_BEHIND)), (
        "unresolved name not caught"
    )
    # HEAD is 0 commits behind itself, so a limit of -1 makes every real sha too old (works on a shallow clone).
    assert any("commits behind" in p and p.startswith("## 1.") for p in check(architecture, overview, -1)), "old sha not caught"
    assert any(p.startswith("overview.md") and "commits behind" in p for p in check(architecture, overview, -1)), (
        "old overview sha not caught"
    )
    assert any("no `Checked" in p for p in check(CHECKED.sub("", architecture), overview, MAX_BEHIND)), "missing line not caught"
    assert any(p.startswith("overview.md: no") for p in check(architecture, CHECKED.sub("", overview), MAX_BEHIND)), (
        "overview without a line not caught"
    )
    print("check-architecture-doc self-test: ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backstage", type=Path, default=DEFAULT_BACKSTAGE)
    ap.add_argument("--max-behind", type=int, default=MAX_BEHIND)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    poot = args.backstage / "projects" / "poot"
    architecture = (poot / "docs" / "architecture.md").read_text()
    overview = (poot / "overview.md").read_text()
    if args.self_test:
        self_test(architecture, overview)
        return 0
    problems = check(architecture, overview, args.max_behind)
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
