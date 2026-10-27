#!/usr/bin/env python3
"""Derives website/src/data/performance.json, the data of the Performance page, from the committed evidence.

Two sources, nothing typed by hand:

* every committed run set under `benchmarks/results` (`<run-id>-r1` ... `-rN`: `results.jsonl` and `env.json`),
  one cell per (hardware, model, scenario, framework, backend, precision): the median, minimum and maximum over
  the repeats of each metric, with the identity the rows state;
* the performance bank's verified kernel receipts (`benchmarks/performance-bank/receipts.json`).

A run set is charted only when it passes the contract the harness itself applies: every row is attributable
(`result_contract.committed_row_errors`), enough repeats ran, one binary produced them, every identity field
agrees across repeats, and the relative noise band is inside the manifest's tolerated uncertainty. A set that
fails is listed under `excluded` with the reason and is not charted.

    python3 benchmarks/harness/perf_page.py          # rewrite the JSON
    python3 benchmarks/harness/perf_page.py --check  # exit 1 when the committed JSON differs
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import known_unverified  # noqa: E402
import memory_phases as phases  # noqa: E402
import performance_bank as bank  # noqa: E402
import performance_receipt as receipts  # noqa: E402
from result_contract import committed_row_errors  # noqa: E402

REPO_ROOT = HARNESS_DIR.parents[1]
RESULTS_DIR = bench.BENCH_ROOT / "results"
OUTPUT = REPO_ROOT / "website" / "src" / "data" / "performance.json"

SCHEMA = 5

# Memory is charted at inference: the peak after the model is loaded and the warmup request has run (memory_phases.py).
# The load peak is stated beside it. A cell none of whose runs has a usable load boundary keeps the lifetime peak and says
# it includes the load.
WARMUP_ITERS = phases.warmup_iters(receipts.MANIFEST)
MEMORY_SCOPE_INFERENCE = "inference"
MEMORY_SCOPE_LIFETIME = "includes_load"

# What a scenario measures. The page charts decode throughput from `decode` scenarios, prefill throughput from
# `prefill` scenarios and decode against context length from `curve` scenarios.
SCENARIO_ROLES = {"decode-128": "decode", "prefill-1k": "prefill", "decode-curve": "curve",
                  "decode-curve-large": "curve"}
# A request whose prompt is at least this long has a prefill rate worth stating; a 5-token prompt does not.
PREFILL_MIN_PROMPT_TOKENS = 128

BACKEND_LABELS = {"ptx": "PTX", "rocm": "ROCm", "vulkan": "Vulkan", "wgpu": "wgpu", "cuda": "CUDA", "cpu": "CPU"}
# The accelerator backend of a baseline row recorded before the runners wrote a `backend` field, by (framework,
# vendor of the device). The NVIDIA baselines run from the benchmark image (benchmarks/docker/Dockerfile): llama.cpp
# built with GGML_CUDA, the vLLM and transformers wheels and candle with their CUDA backends. The AMD llama.cpp rows
# ran the llama-bench of the dev box's nixpkgs llama-cpp 0.6.0 (commit d812350, framework_version in the rows), which
# loads the Vulkan backend only. A recorded row is never edited: a row without a `backend` and without an entry here
# is an error, so a new baseline cannot be charted with an unstated backend.
UNRECORDED_BASELINE_BACKENDS = {
    ("llamacpp", "NVIDIA"): "cuda",
    ("vllm", "NVIDIA"): "cuda",
    ("transformers", "NVIDIA"): "cuda",
    ("candle", "NVIDIA"): "cuda",
    ("llamacpp", "AMD"): "vulkan",
}
FRAMEWORK_LABELS = {
    "poot": "poot",
    "llamacpp": "llama.cpp",
    "vllm": "vLLM",
    "candle": "candle",
    "transformers": "transformers",
}
# Row fields shown as the identity of a cell (a tooltip and a collapsed table on the page).
IDENTITY_FIELDS = (
    "build_sha", "adapter", "driver_version", "compute_dtype", "checkpoint_digest", "rustc_version", "framework_version",
)
ROW_METRICS = ("decode_tok_s", "ttft_ms", "tpot_ms", "e2e_ms", "peak_rss_bytes", "peak_vram_bytes", "energy_j")


def _round(value):
    return round(value, 3)


def _stat(values):
    """Median and observed range of `values`, or None when there are none."""
    values = [v for v in values if receipts.is_finite(v)]
    if not values:
        return None
    return {"median": _round(statistics.median(values)), "min": _round(min(values)), "max": _round(max(values)), "n": len(values)}


def slug(text):
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def row_backend(row):
    """The backend a row ran on: the one it records, else the documented mapping for an older baseline row."""
    recorded = row.get("backend")
    if recorded is not None:
        if recorded in ("", "unknown"):
            raise ValueError(f"{row['framework']} on {row.get('device')!r} records the backend {recorded!r}")
        return recorded
    vendor = (row.get("device") or "").split(" ")[0]
    try:
        return UNRECORDED_BASELINE_BACKENDS[(row["framework"], vendor)]
    except KeyError:
        raise ValueError(f"{row['framework']} on {row.get('device')!r} records no backend and UNRECORDED_BASELINE_BACKENDS has no entry") from None


def series_label(framework, backend):
    """The framework and the backend it ran on, e.g. `poot (PTX)` or `llama.cpp (Vulkan)`."""
    return f"{FRAMEWORK_LABELS.get(framework, framework)} ({BACKEND_LABELS.get(backend, backend)})"


def prefill_tok_s(row):
    """Prompt tokens per second of the first-token window, for a request long enough to say."""
    prompt, ttft = row.get("prompt_tokens"), row.get("ttft_ms")
    if receipts.is_finite(prompt) and receipts.is_finite(ttft) and ttft > 0 and prompt >= PREFILL_MIN_PROMPT_TOKENS:
        return prompt / ttft * 1000.0
    return None


def row_value(row, metric):
    if metric == "prefill_tok_s":
        return prefill_tok_s(row)
    if metric == "peak_vram_bytes":
        # What the run itself holds: the device's use before the run (desktop, other sessions) is not the run's.
        peak, base = row.get("peak_vram_bytes"), row.get("vram_baseline_bytes")
        return max(0, peak - base) if peak is not None and base is not None else peak
    return row.get(metric)


def set_exclusion(rows, policy):
    """Why a set's rows for one cell are not charted, or None when they are."""
    return contract_exclusion(rows, policy) or noise_exclusion(rows, policy)


def contract_exclusion(rows, policy):
    """Why a cell's rows break the contract every charted or listed cell keeps, or None when they hold it."""
    for row in rows:
        if row.get("status") != "ok":
            return f"a repeat has status {row.get('status')!r}"
        errors = committed_row_errors(row)
        if errors:
            return errors[0]
    if len(rows) < policy["min_trials"]:
        return f"{len(rows)} repeats, the policy needs {policy['min_trials']}"
    if len({row.get("binary_sha256") for row in rows}) > 1:
        return "the repeats mix binaries"
    identities = [receipts.row_identity(row) for row in rows]
    for field in receipts.row_identity_fields(rows[0]["framework"]):
        if len({repr(identity[field]) for identity in identities}) > 1:
            return f"the repeats differ in {field}"
    metric = policy["headline_metric"]
    if not all(receipts.is_finite(row.get(metric)) for row in rows):
        return f"non-finite {metric}"
    return None


def relative_band(rows, policy):
    """The noise band of the headline metric over the repeats as a fraction of its median, or None for a zero median."""
    values = [row.get(policy["headline_metric"]) for row in rows]
    median = statistics.median(values)
    return 2 * receipts.noise(values, policy["min_trials"]) / median if median else None


def noise_exclusion(rows, policy):
    """Why the headline metric of a cell's rows is too noisy to be a verified cell, or None when it is narrow enough."""
    band = relative_band(rows, policy)
    tolerated = policy["metrics"][policy["headline_metric"]]["max_uncertainty"]
    if band is None:
        return "zero median"
    return f"noise band {band:.1%} of the median exceeds {tolerated:.0%}" if band > tolerated else None


def cell_record(set_name, date, rows, policy, known=None):
    """The record of one cell. Its measured medians are always published; a cell listed as known-unverified carries
    `unverified` (the band and the reason) and the page draws its numbers muted, never as a verified result."""
    first = rows[0]
    metrics = {m: _stat([row_value(row, m) for row in rows]) for m in (*ROW_METRICS, "prefill_tok_s")}
    split = [phases.memory_phases(row, WARMUP_ITERS) for row in rows]
    inferred = all(split)
    if inferred:
        # One repeat without a boundary would mix two meanings in one median, so the whole cell keeps the lifetime peak.
        for key, part, field in (("peak_rss_bytes", "inference", "rss_bytes"), ("peak_vram_bytes", "inference", "vram_bytes"),
                                 ("load_rss_bytes", "load", "rss_bytes"), ("load_vram_bytes", "load", "vram_bytes")):
            metrics[key] = _stat([s[part][field] for s in split])
    backend = row_backend(first)
    record = {
        "set": set_name,
        "date": date,
        "model": first["model"],
        "scenario": first["scenario"],
        "role": SCENARIO_ROLES.get(first["scenario"], "other"),
        "framework": first["framework"],
        "backend": backend,
        "series": series_label(first["framework"], backend),
        "precision": first.get("precision"),
        "prompt_tokens": first.get("prompt_tokens"),
        "gen_tokens": first.get("gen_tokens") or first.get("osl"),
        "repeats": len(rows),
        "verified": known is None,
        "memory_scope": MEMORY_SCOPE_INFERENCE if inferred else MEMORY_SCOPE_LIFETIME,
        "band": known["band"] if known else _round(relative_band(rows, policy) or 0.0),
        "metrics": {m: s for m, s in metrics.items() if s},
        "identity": {f: first[f] for f in IDENTITY_FIELDS if first.get(f)},
    }
    if known:
        record["unverified"] = {"pair": known["pair"], "failed_repeats": known.get("failed_repeats", 0), "reason": known["reason"]}
    curve = curve_record(rows)
    if curve:
        record["curve"] = curve
        longest = [point for point in curve if point.get("prefill_tok_s")]
        if longest and "prefill_tok_s" not in record["metrics"]:
            # A decode curve states its prefill rate at each context length; the cell's own is the longest one.
            record["metrics"]["prefill_tok_s"] = longest[-1]["prefill_tok_s"]
            record["prefill_isl"] = longest[-1]["isl"]
    return record


def curve_record(rows):
    """Per context length, the statistics over the repeats of each repeat's median."""
    isls = sorted({point["isl"] for row in rows for point in row.get("curve") or []})
    points = []
    for isl in isls:
        at = [point for row in rows for point in row.get("curve") or [] if point["isl"] == isl]
        point = {"isl": isl}
        for metric in ("decode_tok_s",):
            point[metric] = _stat([p.get(metric) for p in at])
        for metric in ("ttft_ms", "tpot_ms"):
            point[metric] = _stat([p[metric]["p50"] for p in at])
        if at and at[0].get("prompt_tokens"):
            point["prefill_tok_s"] = _stat([prefill_tok_s({"prompt_tokens": p["prompt_tokens"], "ttft_ms": p["ttft_ms"]["p50"]}) for p in at])
        points.append({k: v for k, v in point.items() if v})
    return points


def host_entry(hosts, device, host_cpu, driver):
    """The entry of one host: a device on one host CPU with one GPU driver. Two hosts, or two drivers, of the same
    GPU are two entries, and a host or driver the run did not record is an entry of its own (unknown), never merged
    into a known one."""
    key = (device, host_cpu, driver)
    if key not in hosts:
        hosts[key] = {"cpu": host_cpu, "driver": driver, "cells": [], "excluded": [], "host_kernels": set()}
    return hosts[key]


def collect_runs(results_dir, policy, hosts, known_path):
    """Every committed run set as cells (or exclusions) of the host that ran it."""
    sets = bench.committed_sets(results_dir)
    listed = known_unverified.load(known_path)
    for set_name, dirs in sets.items():
        rows = [row for run_dir in dirs for row in bench._load_rows(run_dir)]
        env_path = dirs[0] / "env.json"
        env = json.loads(env_path.read_text()) if env_path.exists() else {}
        date = (env.get("captured_at_utc") or "")[:10]
        cells = {}
        for row in rows:
            cells.setdefault(bench.cell_key(row), []).append(row)
        measured = {cell["key"]: cell for cell in bench.compare_cells(rows, rows, policy)}
        for key, cell_rows in cells.items():
            device = key[4] or "unknown"
            entry = host_entry(hosts, device, key[5], key[6])
            entry["host_kernels"].update(row["host_kernel"] for row in cell_rows if row.get("host_kernel"))
            known = listed.get(key)
            if known and known_unverified.judge(measured[key], known)["verdict"] == receipts.KNOWN_UNVERIFIED:
                cell_rows = [row for row in cell_rows if row.get("status") == "ok"]
                reason = contract_exclusion(cell_rows, policy)
            else:
                known, reason = None, set_exclusion(cell_rows, policy)
            if reason:
                entry["excluded"].append({
                    "set": set_name, "model": key[0], "series": series_label(key[2], row_backend(cell_rows[0])),
                    "reason": reason,
                })
            else:
                entry["cells"].append(cell_record(set_name, date, cell_rows, policy, known))
    for entry in hosts.values():
        newest = {}
        for cell in sorted(entry["cells"], key=lambda c: (c["date"], c["set"])):
            newest[(cell["model"], cell["scenario"], cell["series"])] = cell["set"]
        for cell in entry["cells"]:
            cell["latest"] = newest[(cell["model"], cell["scenario"], cell["series"])] == cell["set"]
        entry["cells"].sort(key=lambda c: (c["model"], c["scenario"], c["series"], c["date"], c["set"]))


def collect_kernels(bank_root):
    """Every verified bank receipt as a `(device, kernel row)` pair, for the device it was measured on."""
    claims = bank.load_json(bank_root / "receipts.json")
    cases = {case["id"]: case for case in bank.load_json(bank_root / "cases.json")["cases"]}
    kernels = []
    for claim in claims["entries"]:
        if claim["status"] != "verified":
            continue
        receipt = bank.load_json(bank_root / claim["receipt"])
        if not receipt.get("reference", {}).get("agrees"):
            continue
        identity = receipt["identity"]
        samples = [r["metrics"][bank.KERNEL_METRIC] for r in receipt["cells"]["kernel"]["repeats"] if r["status"] == "ok"]
        case = cases[claim["case"]]
        kernels.append((identity["device"], {
            "case": claim["case"],
            "regime": case["regime"],
            "backend": claim["backend"],
            "series": series_label("poot", claim["backend"]),
            "storage": identity["precision"],
            "dimensions": identity["dimensions"],
            "live_context": identity["live_context"],
            "driver": identity["driver"],
            "step_us": _stat(samples),
            "receipt": f"benchmarks/performance-bank/{claim['receipt']}",
        }))
    return kernels


# The page has one workload family: decode and prefill rates over INPUT TOKENS. A decode-curve cell is a sweep over the
# context lengths of its curve; a single-point cell (decode-128, prefill-1k) is one point of the same axis at the prompt
# length its rows record. The scenario names stay internal to this module.
CURVE_METRICS = ("decode_tok_s", "ttft_ms", "tpot_ms", "prefill_tok_s")
CELL_LEVEL_METRICS = ("peak_vram_bytes", "peak_rss_bytes", "load_vram_bytes", "load_rss_bytes", "energy_j")
# Decode throughput, time per output token and end-to-end time depend on how many tokens are generated, so a point carries them
# only when its run generated DECODE_GEN_TOKENS, the length of every decode series (a context sweep and the short run). The
# 1000-token prefill run generates 32 tokens: it feeds prefill, first-token time and memory, never a decode series.
DECODE_GEN_TOKENS = 128
GENERATION_LENGTH_METRICS = ("decode_tok_s", "tpot_ms", "e2e_ms")
POINT_FIELDS = ("set", "date", "model", "framework", "backend", "series", "precision", "repeats", "verified", "memory_scope", "band", "latest", "identity")


def cell_points(cell):
    """The points of one cell on the input-token axis. A sweep's memory and energy are cell-level (one peak over every
    context length, one integral over the whole cell), so they sit on its longest context length, flagged `over_sweep`."""
    base = {field: cell[field] for field in POINT_FIELDS}
    comparable = cell["gen_tokens"] == DECODE_GEN_TOKENS

    def kept(metrics):
        return {m: s for m, s in metrics.items() if comparable or m not in GENERATION_LENGTH_METRICS}

    if "unverified" in cell:
        base["unverified"] = cell["unverified"]
    if cell.get("curve"):
        longest = cell["curve"][-1]["isl"]
        points = []
        for point in cell["curve"]:
            metrics = kept({m: point[m] for m in CURVE_METRICS if m in point})
            record = {**base, "isl": point["isl"], "metrics": metrics}
            if point["isl"] == longest:
                metrics.update({m: cell["metrics"][m] for m in CELL_LEVEL_METRICS if m in cell["metrics"]})
                record["over_sweep"] = True
            points.append(record)
        return points
    return [{**base, "isl": cell["prompt_tokens"], "metrics": kept(cell["metrics"])}]


def host_label(host_cpu, driver, hosts_of_cpu):
    """A host named by its CPU, with the driver added where one CPU appears with several drivers."""
    label = host_cpu or "unknown host"
    return f"{label}, driver {driver or 'unknown'}" if hosts_of_cpu > 1 else label


# Editions of one chip are one hardware choice. Max-Q is not stripped: it is a different power class of the chip, so it
# stays its own entry (no committed run is on one yet).
EDITION_SUFFIXES = (" Workstation Edition", " Server Edition")


def device_class(device):
    """The hardware choice a device is listed under: its name without the edition (a workstation and a server board of
    one chip are the same choice), so the page offers one entry per GPU model."""
    for suffix in EDITION_SUFFIXES:
        if device.endswith(suffix):
            return device[: -len(suffix)]
    return device


def combine_hosts(points, labels):
    """One point from the points of several hosts at one (model, series, input length, scenario): the median of the
    per-host medians over the verified points (every point when none is verified, and then the point is unverified),
    with the observed range of every repeat behind it and the spread of the host medians. The page aggregates here, the
    harness still judges each cell on its own host. `labels` name the host of each point; the point states the hosts it is taken over."""
    verified = [p for p in points if p["verified"]]
    used = verified or points
    used_labels = sorted(label for p, label in zip(points, labels) if p["verified"] or not verified)
    metrics, spread = {}, {}
    for metric in sorted({m for p in used for m in p["metrics"]}):
        stats = [p["metrics"][metric] for p in used if metric in p["metrics"]]
        medians = [s["median"] for s in stats]
        metrics[metric] = {
            "median": _round(statistics.median(medians)),
            "min": min(s["min"] for s in stats),
            "max": max(s["max"] for s in stats),
            "n": sum(s["n"] for s in stats),
        }
        spread[metric] = {"min": min(medians), "max": max(medians)}
    first = used[0]
    point = {
        **first,
        "set": ", ".join(sorted({p["set"] for p in used})),
        "date": max(p["date"] for p in used),
        "repeats": sum(p["repeats"] for p in used),
        "band": _round(statistics.median(p["band"] for p in used)),
        "identity": {f: v for f, v in first["identity"].items() if all(p["identity"].get(f) == v for p in used)},
        "metrics": metrics,
        "spread": spread,
        "memory_scope": MEMORY_SCOPE_LIFETIME if any(p["memory_scope"] == MEMORY_SCOPE_LIFETIME for p in used) else MEMORY_SCOPE_INFERENCE,
        "hosts": len(used),
        "host_labels": used_labels,
        "cells": {"verified": len(verified), "unverified": len(points) - len(verified)},
    }
    if any(p.get("over_sweep") for p in used):
        point["over_sweep"] = True
    if verified:
        point.pop("unverified", None)
    return point


def aggregate_hosts(host_points):
    """The points of a hardware choice from the points of its hosts, as `(host, scenario, point)`. The newest sets of
    the hosts are combined into one point per (model, series, input length, scenario), so a host-only difference never
    splits a series. A set older than the newest verified one of its point, on any host, is superseded: it stays a row of its own
    (the table lists it, no chart draws it) and feeds no median."""
    # The date of the newest verified set of a point, or of the newest set when none is verified: an unverified re-measurement
    # does not supersede the last verified one (the page still prefers the verified values, see `combine_hosts`).
    newest, newest_verified = {}, {}
    for host, scenario, point in host_points:
        key = (point["model"], point["series"], point["isl"], scenario)
        if point["latest"]:
            newest[key] = max(newest.get(key, ""), point["date"])
            if point["verified"]:
                newest_verified[key] = max(newest_verified.get(key, ""), point["date"])
    groups, older = {}, []
    for host, scenario, point in host_points:
        key = (point["model"], point["series"], point["isl"], scenario)
        if not point["latest"] or point["date"] < newest_verified.get(key, newest[key]):
            # A set another host has since re-measured is superseded: it is a row of its own and feeds no headline value.
            older.append(combine_hosts([{**point, "latest": False}], [host]))
            continue
        group = groups.setdefault(key, {})
        if host in group:
            raise ValueError(f"host {host!r} has two newest points for {point['model']} {point['series']} at {point['isl']} tokens")
        group[host] = point
    return [combine_hosts(list(group.values()), list(group)) for group in groups.values()] + older


def hardware_classes(hosts, kernels):
    """One entry per GPU model (`device_class`): its hosts, its aggregated points, the exclusions of every host and
    its kernel receipts. A receipt records its device but not its host, so receipts belong to the device. A device
    with no end-to-end point is not a page entry."""
    classes = {}
    for (device, host_cpu, driver), entry in sorted(hosts.items(), key=lambda kv: tuple(x or "" for x in kv[0])):
        name = device_class(device)
        cls = classes.setdefault(name, {"id": slug(name), "label": name, "hosts": [], "points": [], "excluded": [], "kernels": [], "cells": []})
        cls["cells"].append((host_cpu, driver, entry))
    for device, kernel in kernels:
        if device_class(device) in classes:
            classes[device_class(device)]["kernels"].append(kernel)
    for cls in classes.values():
        host_points, cells = [], cls.pop("cells")
        for host_cpu, driver, entry in cells:
            same_cpu = sum(1 for cpu, _, _ in cells if cpu == host_cpu)
            label = host_label(host_cpu, driver, same_cpu)
            host = {"label": label, **{k: v for k, v in {"cpu": host_cpu, "driver": driver}.items() if v}}
            if entry["host_kernels"]:
                host["kernel"] = ", ".join(sorted(entry["host_kernels"]))
            cls["hosts"].append(host)
            host_points += [(label, cell["scenario"], point) for cell in entry["cells"] for point in cell_points(cell)]
            cls["excluded"] += [{**item, "host": label} for item in entry["excluded"]]
        cls["points"] = aggregate_hosts(host_points)
        cls["points"].sort(key=lambda p: (p["model"], p["series"], p["isl"], p["date"], p["set"]))
        cls["excluded"].sort(key=lambda e: (e["host"], e["set"], e["model"], e["series"]))
        cls["kernels"].sort(key=lambda k: (k["case"], k["backend"]))
    return sorted((c for c in classes.values() if c["points"]), key=lambda c: c["label"])


def generate(results_dir=RESULTS_DIR, bank_root=bank.BANK_ROOT, known_path=None):
    policy = receipts.load_policy()
    hosts = {}
    collect_runs(Path(results_dir), policy, hosts, known_path or known_unverified.beside(results_dir))
    return {
        "schema": SCHEMA,
        "policy": {"min_repeats": policy["min_trials"], "max_uncertainty": policy["metrics"][policy["headline_metric"]]["max_uncertainty"]},
        "hardware": hardware_classes(hosts, collect_kernels(Path(bank_root))),
    }


def render(data):
    return json.dumps(data, indent=1, sort_keys=True) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when the committed JSON differs from the derived one")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    text = render(generate())
    if args.check:
        if not args.output.exists() or args.output.read_text() != text:
            sys.exit(f"{args.output} differs from what perf_page.py derives; run `just perf-page-regen`")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"wrote {args.output} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
