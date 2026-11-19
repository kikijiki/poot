#!/usr/bin/env python3
"""Derive the website's headline numbers from the committed benchmark data, and fail when a stated number drifts.

The claims are the "limitations" lines of `website/src/data/status.json`. The data is
`website/src/data/performance.json` (what `perf_page.py` derives from the committed result sets). Nothing here
edits either file.

THE PAIRING RULE (the one place it is written down; `benchmarks/README.md` repeats it):

* A *point* counts when it is `latest`, verified, and a single-stream point (no `concurrency`). A superseded set and
  a serve-throughput point are never a side of a pair.
* A *cell* is one hardware entry, one backend pair, one model and one input length: `(hardware, pair, model, isl)`.
  The backend pairs are the page's `BACKEND_PAIRS` (website/src/components/performance/format.ts): poot Vulkan with
  llama.cpp Vulkan, poot ROCm with llama.cpp ROCm, poot PTX with llama.cpp CUDA. A framework is never paired with a
  different backend. poot wgpu, candle and transformers have no twin and are in no cell.
* Precision is NOT part of the cell. poot and llama.cpp run a model in the precision each supports (bf16 against f16,
  q4_k against q4_k_m, and on qwen3.8-27b-q4 poot q5_k against llama.cpp q4_k_xl), so the pairing is by model, as the
  page's legend is. The cell is reported with both precisions so a cross-precision cell is visible.
* A cell needs a point on both sides. When one side has several `latest` points (two sets on one cell) the side's
  value is the median of their medians. Every number is the point's `metrics.<name>.median`.
* Decode ratio = llama.cpp decode tok/s / poot decode tok/s (above 1: poot is slower).
  First-token ratio = poot ttft ms / llama.cpp ttft ms (above 1: poot is slower).
  Memory ratio = poot peak VRAM / llama.cpp peak VRAM, over the cells that record both, in the `inference` scope.

A claim is a regular expression over one limitation line plus the derived values it must equal. A stated number is
compared to the derived one rounded half-up to the decimals the prose writes. A claim whose text is absent, whose
line matches twice, or a limitation line with a number that no claim covers, fails: prose and derivation cannot
drift apart silently. A claim that does not reproduce is declared in `benchmarks/known-claim-gaps.toml` with the
number the data give; it passes only while the data still give that number, and a declared claim that reproduces
again fails as stale (as `known-losses.toml` does for the planner invariant).

Usage: `headline_claims.py` (check, exit 1 on a failure), `headline_claims.py --list` (every derivation).
"""

import json
import re
import statistics
import sys
import tomllib
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE = REPO_ROOT / "website" / "src" / "data" / "performance.json"
STATUS = REPO_ROOT / "website" / "src" / "data" / "status.json"
GAPS = REPO_ROOT / "benchmarks" / "known-claim-gaps.toml"

# (framework, backend) -> pair key: the page's BACKEND_PAIRS.
PAIRS = {
    ("poot", "vulkan"): "vulkan",
    ("llamacpp", "vulkan"): "vulkan",
    ("poot", "rocm"): "rocm",
    ("llamacpp", "rocm"): "rocm",
    ("poot", "ptx"): "cuda",
    ("llamacpp", "cuda"): "cuda",
}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


@dataclass(frozen=True)
class Cell:
    hardware: str
    pair: str
    model: str
    isl: int
    poot_precision: str
    llama_precision: str
    decode_ratio: float
    ttft_ratio: float
    vram_ratio: float | None
    memory_scope: str


def _side_value(points, metric):
    values = [p["metrics"][metric]["median"] for p in points if metric in p["metrics"]]
    return statistics.median(values) if values else None


def cells(performance):
    """Every paired cell of the committed data under the pairing rule in the module docstring."""
    sides = {}
    for hardware in performance["hardware"]:
        for p in hardware["points"]:
            pair = PAIRS.get((p["framework"], p["backend"]))
            if pair is None or p.get("concurrency") or not p["latest"] or not p["verified"]:
                continue
            key = (hardware["id"], pair, p["model"], p["isl"])
            sides.setdefault(key, {"poot": [], "llamacpp": []})[p["framework"]].append(p)
    out = []
    for (hardware, pair, model, isl), side in sorted(sides.items()):
        poot, llama = side["poot"], side["llamacpp"]
        if not poot or not llama:
            continue
        pv, lv = _side_value(poot, "peak_vram_bytes"), _side_value(llama, "peak_vram_bytes")
        scopes = {p["memory_scope"] for p in poot + llama}
        out.append(
            Cell(
                hardware=hardware,
                pair=pair,
                model=model,
                isl=isl,
                poot_precision="/".join(sorted({str(p["precision"]) for p in poot})),
                llama_precision="/".join(sorted({str(p["precision"]) for p in llama})),
                decode_ratio=_side_value(llama, "decode_tok_s") / _side_value(poot, "decode_tok_s"),
                ttft_ratio=_side_value(poot, "ttft_ms") / _side_value(llama, "ttft_ms"),
                vram_ratio=(pv / lv) if pv and lv else None,
                memory_scope="/".join(sorted(scopes)),
            )
        )
    return out


def round_to(value, decimals):
    """Round half up to `decimals` places, through the decimal digits of the float's shortest repr."""
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def _decimals(text):
    return len(text.split(".")[1]) if "." in text else 0


@dataclass(frozen=True)
class Check:
    """One stated number and what the data give for it."""

    claim: str
    stated: str
    derived: str
    reproduces: bool
    derivation: str


def _number(claim, text, derived, derivation):
    d = _decimals(text)
    shown = f"{round_to(derived, d)}"
    return Check(claim, text, shown, Decimal(text) == round_to(derived, d), derivation)


def _count(claim, text, derived, derivation):
    return Check(claim, text, str(derived), int(text) == derived, derivation)


def _word_count(claim, text, derived, derivation):
    return Check(claim, text, str(derived), NUMBER_WORDS[text] == derived, derivation)


def _worst(rows, key):
    return max(rows, key=key)


def _name(c):
    return f"{c.hardware} {c.pair} {c.model} isl {c.isl} (poot {c.poot_precision}, llama.cpp {c.llama_precision})"


def claim_decode(m, cs, performance):
    low, high = min(cs, key=lambda c: c.decode_ratio), max(cs, key=lambda c: c.decode_ratio)
    return [
        Check("decode-every-pair-behind", "every pair", "every pair" if low.decode_ratio > 1 else f"{_name(low)} is ahead", low.decode_ratio > 1, f"min decode ratio {low.decode_ratio:.4f} > 1"),
        _number("decode-low", m[1], low.decode_ratio, f"min decode ratio over {len(cs)} cells: {_name(low)}"),
        _number("decode-high", m[2], high.decode_ratio, f"max decode ratio over {len(cs)} cells: {_name(high)}"),
    ]


def claim_first_token(m, cs, performance):
    behind = [c for c in cs if c.ttft_ratio > 1]
    worst = _worst(cs, lambda c: c.ttft_ratio)
    return [
        _count("first-token-behind-cells", m[1], len(behind), f"cells with first-token ratio above 1, of {len(cs)}"),
        _count("first-token-cells", m[2], len(cs), "paired cells"),
        _number("first-token-max", m[3], worst.ttft_ratio, f"max first-token ratio: {_name(worst)}"),
    ]


SMALL_QWEN3 = {"qwen3-0.6b", "qwen3-4b"}
AHEAD_ON = {("nvidia-rtx-pro-6000-blackwell", "cuda"), ("amd-radeon-8060s-graphics", "rocm")}


def claim_ahead(m, cs, performance):
    ahead = [c for c in cs if c.ttft_ratio < 1]
    where = {(c.hardware, c.pair) for c in ahead}
    models = {c.model for c in ahead}
    ok = where == AHEAD_ON and models <= SMALL_QWEN3
    derived = f"ahead on {sorted(where)} models {sorted(models)}"
    return [Check("first-token-ahead", "small qwen3 rows on RTX PRO 6000 PTX and Strix Halo ROCm only", derived, ok, f"{len(ahead)} cells with first-token ratio below 1: " + "; ".join(_name(c) for c in ahead))]


def claim_memory(m, cs, performance):
    rows = [c for c in cs if c.vram_ratio is not None and c.memory_scope == "inference"]
    worst = _worst(rows, lambda c: c.vram_ratio)
    return [_number("memory-max", m[1], worst.vram_ratio, f"max peak-VRAM ratio over {len(rows)} cells with both recorded: {_name(worst)}")]


def claim_grid(m, cs, performance):
    models = {p["model"] for h in performance["hardware"] for p in h["points"] if PAIRS.get((p["framework"], p["backend"])) and not p.get("concurrency")}
    return [
        _word_count("grid-machines", m[1], len(performance["hardware"]), "hardware entries"),
        _word_count("grid-models", m[2], len(models), f"distinct models of poot and llama.cpp points: {sorted(models)}"),
    ]


# (limitation line pattern, derivation). The pattern's groups are the stated numbers.
CLAIMS = [
    (re.compile(r"^Behind llama\.cpp on the same backend on decode in every pair, from (\d+\.\d+)x to (\d+\.\d+)x slower\.$"), claim_decode),
    (re.compile(r"^Behind on first token in (\d+) of (\d+) cells, up to (\d+\.\d+)x slower\.$"), claim_first_token),
    (re.compile(r"^Ahead of llama\.cpp on first token on the small qwen3 rows \(RTX PRO 6000 PTX, Strix Halo ROCm\); never against a different backend\.$"), claim_ahead),
    (re.compile(r"^Holds up to (\d+\.\d+)x the GPU memory of llama\.cpp at inference\.$"), claim_memory),
    (re.compile(r"^The benchmark grid is (\w+) machines and (\w+) models, with llama\.cpp as the reference\.$"), claim_grid),
]


def load_gaps(path=GAPS):
    with open(path, "rb") as f:
        return {g["claim"]: g for g in tomllib.load(f).get("gap", [])}


def run(performance, limitations, gaps):
    """Return `(checks, failures)`. A failure is a string; `checks` holds every derivation for `--list`."""
    cs = cells(performance)
    checks, failures, used = [], [], set()
    for pattern, derive in CLAIMS:
        lines = [(line, pattern.match(line)) for line in limitations]
        lines = [(line, m) for line, m in lines if m]
        if len(lines) != 1:
            failures.append(f"{len(lines)} limitation lines match the claim {pattern.pattern!r}; exactly one must (the prose changed without its derivation)")
            continue
        used.add(lines[0][0])
        for c in derive(lines[0][1], cs, performance):
            checks.append(c)
            gap = gaps.get(c.claim)
            if c.reproduces and gap:
                failures.append(f"{c.claim}: declared a gap in known-claim-gaps.toml but the data reproduce the stated {c.stated}; delete the entry")
            elif not c.reproduces and not gap:
                failures.append(f"{c.claim}: the site states {c.stated}, the data give {c.derived} ({c.derivation})")
            elif not c.reproduces and (gap["stated"] != c.stated or gap["derived"] != c.derived):
                failures.append(f"{c.claim}: the declared gap says stated {gap['stated']} data {gap['derived']}, now stated {c.stated} data {c.derived}; review the entry")
    for line in limitations:
        if line not in used and re.search(r"\d|\b(" + "|".join(NUMBER_WORDS) + r")\b", line):
            failures.append(f"a limitation states a number no claim derives: {line!r}")
    known = {c.claim for c in checks}
    failures += [f"known-claim-gaps.toml declares {name!r}, which no claim produces" for name in gaps if name not in known]
    return checks, failures


def main(argv):
    performance = json.loads(PERFORMANCE.read_text())
    limitations = json.loads(STATUS.read_text())["limitations"]
    checks, failures = run(performance, limitations, load_gaps())
    if "--list" in argv:
        for c in checks:
            status = "reproduces" if c.reproduces else "DOES NOT REPRODUCE"
            print(f"{c.claim}: stated {c.stated}, data {c.derived}: {status}\n    {c.derivation}")
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    if not failures:
        print(f"headline claims: {len(checks)} numbers checked, {sum(1 for c in checks if not c.reproduces)} declared gaps")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
