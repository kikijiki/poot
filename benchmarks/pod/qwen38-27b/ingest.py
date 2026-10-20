"""POOT-1184: turn a pod-sweep results directory into a committed benchmarks/results/<run-id>/ snapshot.

The poot curve goes through the harness's own `validate_runner_result` (the same contract `bench run` applies),
so the row is exactly what `bench.py run` would have written; env.json carries the pod facts; the sweep's own
summary.tsv, phases.tsv and commands.tsv are copied next to it as the full record (serve / vLLM / llama.cpp
rows have no results.jsonl shape and live there). Index regeneration is `bench index`, run after this.

Usage: python -I ingest.py <sweep-results-dir> <snapshot-dir> --run-id ID --model MODEL --captured-at ISO8601
           [--note TEXT] [--caveat TEXT]...

Each --caveat is recorded verbatim on the row (run-specific facts - why a baseline is missing, a known defect - come
from the caller, not from this script). logs/oracle_receipt.* and the small serve_*.err logs are copied into the snapshot when present.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2] / "harness"
sys.path.insert(0, str(HARNESS))
from result_contract import validate_runner_result  # noqa: E402


def read_env(path):
    env = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.startswith(" "):
            key, _, value = line.partition("=")
            env[key] = value
    return env


def driver_cuda(sweep, env):
    """image_cuda is empty when the image has no nvcc (older images): fall back to the driver's CUDA version from nvidia-smi."""
    if env.get("driver_cuda"):
        return env["driver_cuda"].replace("CUDA Version: ", "")
    for line in (sweep / "nvidia-smi-q.txt").read_text().splitlines():
        if line.startswith("CUDA Version"):
            return line.partition(":")[2].strip() + " (driver)"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("snapshot")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--captured-at", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--caveat", action="append", default=[])
    args = ap.parse_args()
    sweep, snap = Path(args.sweep), Path(args.snapshot)

    timing = json.loads((sweep / "raw" / "poot_curve.json").read_text())
    env = read_env(sweep / "env.txt")
    scenario = {"id": "decode-curve", "kind": "decode-curve",
                "isl": [p["isl"] for p in timing["curve"]], "osl": timing["osl"]}
    prepared = validate_runner_result(timing, framework="poot", precision=timing["precision"], scenario=scenario,
                                      gen_tokens=timing["osl"], backend=timing["backend"])
    phases = {}
    for line in (sweep / "phases.tsv").read_text().splitlines()[1:]:
        name, rc, wall, vram, _note = (line.split("\t") + [""] * 5)[:5]
        phases[name] = (rc, wall, vram)
    peak_mib = float(phases.get("poot_curve", ("", "", ""))[2] or 0)

    row = {"framework": "poot", "model": args.model, "precision": timing["precision"], "scenario": "decode-curve",
           "backend": timing["backend"], "status": "ok", "timing_claim": "request-window"}
    for key in ("framework_version", "build_sha", "family", "preparation"):
        if key in prepared:
            row[key] = prepared[key]
    pts = prepared["curve"]
    row.update(osl=prepared["osl"], curve=pts, isl=pts[0]["isl"], decode_tok_s=pts[0]["decode_tok_s"],
               ttft_ms=pts[0]["ttft_ms"]["p50"], tpot_ms=pts[0]["tpot_ms"]["p50"])
    if peak_mib:
        row["peak_vram_bytes"] = int(peak_mib * 1024 * 1024)
    row["caveats"] = prepared.get("caveats", []) + args.caveat + [
        "Context is a repeated phrase cycled to exactly ISL tokens, not the suite's synthetic random tokens.",
        "peak_vram_bytes is an external nvidia-smi sample over the whole poot_curve phase, not the observer's.",
    ]

    snap.mkdir(parents=True, exist_ok=False)
    (snap / "results.jsonl").write_text(json.dumps(row) + "\n")
    (snap / "env.json").write_text(json.dumps({
        "captured_at_utc": args.captured_at,
        "gpu_name": env.get("gpu_name"), "gpu_driver": env.get("driver"),
        "gpu_memory_total": f"{env.get('gpu_mem_mib')} MiB", "cuda_version": env.get("image_cuda") or driver_cuda(sweep, env),
        "host_kernel": env.get("kernel"), "cpu": f"{env.get('host_cpus')} vCPU, {env.get('host_mem_gib')} GiB host RAM",
        "harness_version": "pod-sweep (benchmarks/pod/qwen38-27b), not bench.py run",
        "poot_git_sha": timing.get("build_sha"), "backend": "ptx (nvptx, cudarc driver)",
        "checkpoint": "Qwen/Qwen3.8-27B-FP8 (FP8 weights as stored; --precision bf16 labels the compute dtype)",
        "metrics_source": "poot engine timing (poot-bench-runner decode-curve, batch 1)",
        "fp8_note": env.get("fp8_note"), "note": args.note,
    }, indent=2) + "\n")
    for name in ("summary.tsv", "phases.tsv", "commands.tsv", "env.txt", "model-revisions.txt"):
        shutil.copy(sweep / name, snap / name)
    (snap / "logs").mkdir()
    # The logs that explain a missing row: the oracle receipt and a server that failed to start.
    for pattern in ("oracle_receipt.*", "serve_*.err"):
        for log in sorted((sweep / "logs").glob(pattern)):
            if pattern == "oracle_receipt.*" or log.stat().st_size < 1_000_000:
                shutil.copy(log, snap / "logs" / log.name)


if __name__ == "__main__":
    main()
