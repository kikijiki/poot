#!/usr/bin/env python3
"""Render the authored diagram sources (website/diagrams/*.svg) to the PNGs the README and the website use.

    just diagrams [--out DIR] [--only a,b] [--publish-dir DIR] [--check]
    scripts/diagrams.py --self-test

The SVGs are the source of truth; the PNGs and index.json are generated and committed. Rendering uses the
devshell's pinned resvg and the one font directory it exports (POOT_DIAGRAM_FONT_DIR), at a fixed width, so the
output is byte-reproducible. The script writes only to --out and --publish-dir; it never runs git or the network.

--check renders to a temp dir and fails when a committed PNG or index.json differs from its source, when an image
in --out has no source, or when the README or the website reference a different set of images than index.json
(or retype an alt text that index.json holds).
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "website" / "diagrams"
DEFAULT_OUT = ROOT / "website" / "static" / "img" / "architecture"
RENDER_WIDTH = 2400
FONT_FAMILY = "DejaVu Sans Mono"
MANIFEST = "index.json"
# Where the committed images are referenced from (README.md plus every page and component of the site).
REFERENCE_ROOTS = ["README.md", "website/docs", "website/src"]
IMAGE_REF = re.compile(r"img/architecture/([A-Za-z0-9_.-]+\.png)")
MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]*img/architecture/([A-Za-z0-9_.-]+\.png))\)")


class DiagramError(Exception):
    """A failure that is the user's to fix: a usage error, a missing tool, or a bad source."""


def sources():
    found = sorted(SOURCE_DIR.glob("*.svg"))
    if not found:
        raise DiagramError(f"no diagram sources under {SOURCE_DIR.relative_to(ROOT)}")
    return found


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def alt_text(svg: Path) -> str:
    desc = ET.parse(svg).getroot().find("{http://www.w3.org/2000/svg}desc")
    if desc is None or not (desc.text or "").strip():
        raise DiagramError(f"{svg.relative_to(ROOT)} has no <desc>: it holds the image's alt text")
    return " ".join(desc.text.split())


def png_size(data: bytes):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise DiagramError("resvg did not produce a PNG")
    return struct.unpack(">II", data[16:24])


def rasterise(svg: Path, dest: Path):
    fonts = os.environ.get("POOT_DIAGRAM_FONT_DIR")
    if not fonts or shutil.which("resvg") is None:
        raise DiagramError("resvg and POOT_DIAGRAM_FONT_DIR come from the devshell: run inside `nix develop`")
    cmd = [
        "resvg", "--skip-system-fonts", "--use-fonts-dir", fonts,
        "--font-family", FONT_FAMILY, "--monospace-family", FONT_FAMILY,
        "--width", str(RENDER_WIDTH), str(svg), str(dest),
    ]  # fmt: skip
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise DiagramError(f"resvg failed on {svg.name}: {done.stderr.strip()}")


def render(names, out: Path):
    """Render the named sources into `out`; return their manifest entries."""
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for svg in sources():
        if svg.stem not in names:
            continue
        png = out / f"{svg.stem}.png"
        rasterise(svg, png)
        data = png.read_bytes()
        width, height = png_size(data)
        entries.append(
            {
                "name": svg.stem,
                "file": png.name,
                "width": width,
                "height": height,
                "sha256": sha256(data),
                "source": svg.relative_to(ROOT).as_posix(),
                "sourceSha256": sha256(svg.read_bytes()),
                "alt": alt_text(svg),
            }
        )
    return entries


def manifest_text(entries) -> str:
    return json.dumps({"images": entries}, indent=2) + "\n"


def references():
    """Every `img/architecture/<file>.png` named by the README or the site, with the file that names it."""
    found = {}
    for root in REFERENCE_ROOTS:
        base = ROOT / root
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        for path in files:
            if path.suffix not in {".md", ".mdx", ".ts", ".tsx"}:
                continue
            for match in IMAGE_REF.finditer(path.read_text(encoding="utf-8")):
                found.setdefault(match.group(1), set()).add(path.relative_to(ROOT).as_posix())
    return found


def check_references(entries):
    problems = []
    by_file = {e["file"]: e for e in entries}
    referenced = references()
    for file in sorted(set(referenced) - set(by_file)):
        problems.append(f"{file} is referenced by {sorted(referenced[file])} but is not in {MANIFEST}")
    for file in sorted(set(by_file) - set(referenced)):
        problems.append(f"{file} is in {MANIFEST} but neither the README nor the website references it")
    for root in REFERENCE_ROOTS:
        base = ROOT / root
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.suffix in {".md", ".mdx"})
        for path in files:
            for match in MARKDOWN_IMAGE.finditer(path.read_text(encoding="utf-8")):
                alt, file = match.group(1), match.group(3)
                if file in by_file and alt != by_file[file]["alt"]:
                    problems.append(
                        f"{path.relative_to(ROOT)}: alt text of {file} is {alt!r}, "
                        f"{MANIFEST} holds {by_file[file]['alt']!r}"
                    )
    return problems


def check(out: Path):
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        entries = render({s.stem for s in sources()}, fresh)
        (fresh / MANIFEST).write_text(manifest_text(entries), encoding="utf-8")
        expected = {p.name for p in fresh.iterdir()}
        for name in sorted(expected):
            committed = out / name
            if not committed.is_file():
                problems.append(f"{committed.relative_to(ROOT)} is missing: run `just diagrams`")
            elif committed.read_bytes() != (fresh / name).read_bytes():
                problems.append(f"{committed.relative_to(ROOT)} differs from its source: run `just diagrams`")
    for extra in sorted(p.name for p in out.glob("*.png")) if out.is_dir() else []:
        if extra not in expected:
            problems.append(f"{(out / extra).relative_to(ROOT)} has no source in {SOURCE_DIR.relative_to(ROOT)}")
    problems += check_references(entries)
    if problems:
        print("diagrams: out of date or unreferenced:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"diagrams: {len(entries)} images match their sources and are referenced exactly by {MANIFEST}")
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="diagrams", description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for the PNGs and index.json")
    parser.add_argument("--only", help="comma-separated diagram basenames; index.json is written only for a full run")
    parser.add_argument("--publish-dir", type=Path, help="also copy the PNGs and index.json here")
    parser.add_argument("--check", action="store_true", help="fail when a committed PNG differs from its source")
    args = parser.parse_args(argv)
    known = [s.stem for s in sources()]
    args.names = set(known)
    if args.only is not None:
        args.names = {n for n in args.only.split(",") if n}
        unknown = sorted(args.names - set(known))
        if unknown or not args.names:
            parser.error(f"unknown diagram {', '.join(unknown) or '(none given)'}; known: {', '.join(known)}")
    if args.check and (args.only or args.publish_dir):
        parser.error("--check compares every diagram and writes nothing: drop --only and --publish-dir")
    return args


def main(argv):
    if argv == ["--self-test"]:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DiagramsTest)
        return 0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1
    args = parse_args(argv)
    out = args.out.resolve()
    if args.check:
        return check(out)
    entries = render(args.names, out)
    full = args.names == {s.stem for s in sources()}
    if full:
        (out / MANIFEST).write_text(manifest_text(entries), encoding="utf-8")
    if args.publish_dir:
        publish = args.publish_dir.resolve()
        publish.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            shutil.copyfile(out / entry["file"], publish / entry["file"])
        if full:
            shutil.copyfile(out / MANIFEST, publish / MANIFEST)
    for entry in entries:
        print(f"{entry['file']}  {entry['width']}x{entry['height']}  {entry['sha256'][:12]}")
    if not full:
        print(f"{MANIFEST} not written: it describes every diagram, so only a full run updates it")
    return 0


class DiagramsTest(unittest.TestCase):
    """Model-free self-test (`just diagrams-check` runs it): the CLI paths and the reference check's failure paths."""

    def run_main(self, *argv):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = main(list(argv))
            except SystemExit as exit_:
                code = exit_.code
        return code, out.getvalue(), err.getvalue()

    def tmp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_only_renders_just_the_named_diagram_and_writes_no_manifest(self):
        out = self.tmp() / "out"
        code, _, _ = self.run_main("--out", str(out), "--only", "capture-replay")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["capture-replay.png"])

    def test_publish_dir_gets_every_png_and_the_manifest(self):
        tmp = self.tmp()
        code, _, _ = self.run_main("--out", str(tmp / "out"), "--publish-dir", str(tmp / "pub"))
        self.assertEqual(code, 0)
        names = sorted(p.name for p in (tmp / "pub").iterdir())
        self.assertEqual(names, sorted([f"{s.stem}.png" for s in sources()] + [MANIFEST]))
        self.assertEqual((tmp / "pub" / MANIFEST).read_bytes(), (tmp / "out" / MANIFEST).read_bytes())

    def test_unknown_only_name_is_a_usage_error(self):
        out = self.tmp() / "out"
        code, _, err = self.run_main("--out", str(out), "--only", "capture-replay,nope")
        self.assertEqual(code, 2)
        self.assertIn("unknown diagram nope", err)
        self.assertFalse(out.exists(), "a usage error must render nothing")

    def test_check_refuses_only_and_publish_dir(self):
        self.assertEqual(self.run_main("--check", "--only", "capture-replay")[0], 2)

    def fixture(self, readme_alt, readme_image="rust-kernels.png"):
        """A repo of one diagram whose committed PNG is current, with a README referencing `readme_image`."""
        root = self.tmp()
        (root / "website" / "diagrams").mkdir(parents=True)
        shutil.copy(SOURCE_DIR / "rust-kernels.svg", root / "website" / "diagrams")
        (root / "website" / "docs").mkdir()
        (root / "website" / "src").mkdir()
        out = root / "website" / "static" / "img" / "architecture"
        patches = [
            mock.patch.object(sys.modules[__name__], "ROOT", root),
            mock.patch.object(sys.modules[__name__], "SOURCE_DIR", root / "website" / "diagrams"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        entries = render({"rust-kernels"}, out)
        (out / MANIFEST).write_text(manifest_text(entries), encoding="utf-8")
        (root / "README.md").write_text(f"![{readme_alt}](website/static/img/architecture/{readme_image})\n")
        return out, entries

    def test_check_passes_on_a_consistent_tree(self):
        out, entries = self.fixture("One Rust kernel compiled to SPIR-V, PTX and AMDGCN")
        self.assertEqual(check_references(entries), [])
        self.assertEqual(self.run_main("--out", str(out), "--check")[0], 0)

    def test_check_names_a_retyped_alt_text(self):
        _, entries = self.fixture("A retyped alt")
        problems = check_references(entries)
        self.assertEqual(len(problems), 1)
        self.assertIn("alt text of rust-kernels.png", problems[0])

    def test_check_names_a_missing_and_an_unreferenced_image(self):
        _, entries = self.fixture("One Rust kernel compiled to SPIR-V, PTX and AMDGCN", "gone.png")
        problems = check_references(entries)
        self.assertTrue(any("gone.png is referenced" in p and "not in index.json" in p for p in problems), problems)
        self.assertTrue(any("rust-kernels.png is in index.json but" in p for p in problems), problems)

    def test_check_fails_when_a_source_changes_without_regenerating(self):
        out, _ = self.fixture("One Rust kernel compiled to SPIR-V, PTX and AMDGCN")
        svg = SOURCE_DIR / "rust-kernels.svg"
        svg.write_text(svg.read_text().replace("pootc", "pootd"), encoding="utf-8")
        code, _, err = self.run_main("--out", str(out), "--check")
        self.assertEqual(code, 1)
        self.assertIn("rust-kernels.png differs from its source", err)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except DiagramError as err:
        print(f"diagrams: {err}", file=sys.stderr)
        sys.exit(1)
