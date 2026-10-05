#!/usr/bin/env python3
"""poot benchmark suite orchestrator.

Subcommands:
  matrix   Print the (model x framework x scenario) cells the manifest enumerates (no run).
  run      Run the matrix (or a filtered subset), wrap each runner with the external observer,
           write env.json + results.jsonl + report.md under a timestamped run directory.
  report   Render a committed run directory to a markdown comparison table (pure stdlib, no GPU).
  compare  Diff a baseline against a candidate (each one or more run directories, ideally NOISE_REPEATS
           repeats of the same matrix), cell by cell. A cell is (model, scenario, framework, backend,
           device). Exits 1 on a regression beyond the baseline's noise band, 2 when a cell has too few
           repeats to say, 0 otherwise. Speed is recorded, never gated: the band guards a regression only.

Design (spec 112): runners produce timing; the observer (observer.py) produces memory uniformly across
engines; the manifest declares matched precision per cell and the known gaps of external tools, while poot
refuses a model it cannot run with its own text; unsupported and failed cells are recorded explicitly, never
dropped (FR-005, FR-007).

The report/compare paths are pure stdlib, so they run without a GPU or framework.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

try:
    from .result_contract import (
        BUILD_IDENTITY_FRAMEWORKS,
        CURVE_KIND,
        ResultValidationError,
        runner_refusal,
        validate_runner_result,
    )
except ImportError:  # direct `python harness/bench.py` execution
    from result_contract import (
        BUILD_IDENTITY_FRAMEWORKS,
        CURVE_KIND,
        ResultValidationError,
        runner_refusal,
        validate_runner_result,
    )

FRAMEWORKS = ["poot", "candle", "transformers", "vllm", "llamacpp"]


# --------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------
def load_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def is_supported(fw_cfg):
    """Whether the manifest lets a cell run. Only a hand-checked gap in an external tool says no.

    poot entries carry no `support` key: whether poot runs a model is decided by the runner, which refuses
    with its own text (`runner_refusal`).
    """
    return fw_cfg.get("support", True)


def cells(manifest):
    """Yield every (model, scenario, framework, fw_cfg) cell the manifest declares."""
    for model in manifest["model"]:
        for scen in manifest["scenario"]:
            for fw in FRAMEWORKS:
                fw_cfg = model.get("frameworks", {}).get(fw)
                if fw_cfg is None:
                    fw_cfg = {"support": False, "reason": f"not declared for {fw} in manifest"}
                yield model, scen, fw, fw_cfg


# --------------------------------------------------------------------------------------------------
# Environment capture (FR-004)
# --------------------------------------------------------------------------------------------------
def sh(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _rocm_gpu_name():
    """Marketing name of the first GPU agent `rocminfo` lists (AMD devices have no nvidia-smi), or None."""
    listing = sh(["rocminfo"])
    marketing_name = None
    for line in (listing or "").splitlines():
        field, _, value = line.partition(":")
        if field.strip() == "Marketing Name":
            marketing_name = value.strip()
        elif field.strip() == "Device Type" and value.strip() == "GPU":
            return marketing_name
    return None


def capture_env(manifest_path):
    smi = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
              "--format=csv,noheader"])
    gpu_name = gpu_driver = gpu_total = None
    if smi:
        parts = [p.strip() for p in smi.splitlines()[0].split(",")]
        if len(parts) == 3:
            gpu_name, gpu_driver, gpu_total = parts
    # The poot commit is not captured here: the checkout the harness runs in says nothing about the binary
    # under test. Each poot result row carries the sha the binary reports it was built from.
    if gpu_name is None:
        gpu_name = _rocm_gpu_name()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": gpu_name,
        "gpu_driver": gpu_driver,
        "gpu_memory_total": gpu_total,
        "cuda_version": sh(["bash", "-lc", "nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1"]),
        "host_kernel": sh(["uname", "-r"]),
        "cpu": sh(["bash", "-lc", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]),
        "manifest": str(manifest_path),
        "harness_version": "0.1.0",
    }


# --------------------------------------------------------------------------------------------------
# Running one cell
# --------------------------------------------------------------------------------------------------
def resolve_binary(runner):
    bin_env = runner.get("bin_env")
    return (os.environ.get(bin_env) if bin_env else None) or runner.get("bin")


def binary_sha256(binary):
    """SHA-256 of the executable `binary` resolves to (PATH lookup included), or None if it does not."""
    path = shutil.which(binary)
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_cmd(runner, subst):
    binary = resolve_binary(runner)
    out = []
    for tok in runner["cmd"]:
        t = tok.replace("{bin}", binary)
        for k, v in subst.items():
            t = t.replace("{" + k + "}", str(v))
        out.append(t)
    return out


_ISL_DONE_RE = re.compile(r"isl=(\d+)\s+done")


def _curve_progress(fw, isls):
    """Callback for the observer's live stream: runners print `isl=<N> done ...` per context length; this
    reports progress (k/total, elapsed) and a rough ETA. The ETA assumes per-ISL cost is linear in ISL
    (prefill dominates slow cells), using the latest ISL's incremental time as the rate."""
    start = time.time()
    done = []  # (isl, cumulative_elapsed_s)
    total = len(isls)

    def cb(line):
        m = _ISL_DONE_RE.search(line)
        if not m:
            return
        isl = int(m.group(1))
        if any(d[0] == isl for d in done):   # dedupe: a runner may echo the line on both streams
            return
        el = time.time() - start
        done.append((isl, el))
        remaining = [s for s in isls if s not in {d[0] for d in done}]
        eta = ""
        if remaining:
            prev_cum = done[-2][1] if len(done) >= 2 else 0.0
            inc = max(el - prev_cum, 1e-6)
            rate = inc / max(isl, 1)            # seconds per prompt token at the last ISL
            est = sum(rate * s for s in remaining)
            eta = f", eta ~{est / 60:.1f}m"
        print(f"      [progress] {fw}: isl {isl} done ({len(done)}/{total}), elapsed {el / 60:.1f}m{eta}",
              file=sys.stderr, flush=True)

    return cb


def run_cell(model, scen, fw, fw_cfg, runners, models_dir, suite, interval_ms, overrides=None,
             device=None):
    """Run one cell and return its result row.

    `device` names the accelerator the cell runs on (env capture); it and the backend are part of the row's
    identity, so a compare never merges cells that ran on different ones.
    """
    base = {
        "framework": fw,
        "model": model["id"],
        "precision": fw_cfg.get("precision", model.get("match_precision", "")),
        "scenario": scen["id"],
    }
    if not is_supported(fw_cfg):
        return {**base, "status": "unsupported", "reason": fw_cfg.get("reason", "unsupported"),
                "caveats": _caveats(model, fw_cfg)}

    from observer import NO_JSON_RESULT, parse_last_json_line, run_observed

    model_dir = str(Path(models_dir) / model["local_dir"]) if model.get("local_dir") else model["id"]
    # decode-curve uses synthetic tokens; fall back to short.txt so {prompt_file} still substitutes
    # (ignored under --synthetic, but candle/llamacpp need a real path).
    prompt_file = str(BENCH_ROOT / "prompts" / scen.get("prompt_file", "short.txt"))
    image = ""
    if model.get("image_file"):
        image = str(BENCH_ROOT / "prompts" / model["image_file"])
    overrides = overrides or {}
    subst = {
        "model_dir": model_dir,
        "prompt_file": prompt_file,
        "gen_tokens": overrides.get("gen_tokens") or scen.get("gen_tokens", suite.get("default_gen_tokens", 128)),
        "warmup": overrides.get("warmup") if overrides.get("warmup") is not None else suite.get("warmup_iters", 2),
        "iters": overrides.get("iters") or suite.get("timed_iters", 5),
        "precision": base["precision"],
        "image": image,
        "bench_root": str(BENCH_ROOT),
        # poot backend: ptx by default; a dev-box run sets POOT_BENCH_BACKEND=rocm|wgpu. Only the poot cmd
        # references {backend}.
        "backend": os.environ.get("POOT_BENCH_BACKEND", "ptx"),
    }
    cmd = resolve_cmd(runners[fw], subst)
    # Attribution: the digest of the executable that is about to run, read from the file itself.
    base["binary_sha256"] = binary_sha256(resolve_binary(runners[fw]))
    base["device"] = device
    requested_backend = subst["backend"] if fw in BUILD_IDENTITY_FRAMEWORKS else None
    if requested_backend is not None:
        base["backend"] = requested_backend
    curve = is_curve(scen)
    on_line = None
    if curve:
        isls = list(scen.get("isl", []))
        cmd += ["--mode", "decode-curve",
                "--isl-list", ",".join(str(x) for x in isls),
                "--osl", str(scen.get("osl", 128))]
        if scen.get("synthetic", True):
            cmd.append("--synthetic")
        on_line = _curve_progress(fw, isls)
    obs = run_observed(cmd, interval_s=interval_ms / 1000.0, on_line=on_line)
    timing = parse_last_json_line(obs["stdout"])
    caveats = _merge_caveats(_caveats(model, fw_cfg), _runner_caveats(timing))

    if obs["timed_out"] or obs["returncode"] != 0:
        failure = f"timeout (rc={obs['returncode']})" if obs["timed_out"] else f"rc={obs['returncode']}"
        return _error_row(base, obs, _diagnostic_reason(failure + ":", timing, obs["stderr"]), caveats)

    if timing is NO_JSON_RESULT:
        return _error_row(
            base,
            obs,
            _diagnostic_reason("invalid runner result: no result JSON object", None, obs["stderr"]),
            caveats,
        )

    try:
        refusal = runner_refusal(timing, framework=fw, backend=requested_backend)
    except ResultValidationError as error:
        return _error_row(
            base,
            obs,
            _diagnostic_reason(f"invalid runner result: {error}", timing, obs["stderr"]),
            caveats,
        )
    if refusal is not None:
        row = {**base, "status": "unsupported", "reason": refusal, "caveats": caveats}
        if "build_sha" in timing:
            row["build_sha"] = timing["build_sha"]
        return row

    try:
        timing = validate_runner_result(
            timing,
            framework=fw,
            precision=base["precision"],
            scenario=scen,
            gen_tokens=subst["gen_tokens"],
            backend=requested_backend,
        )
    except ResultValidationError as error:
        return _error_row(
            base,
            obs,
            _diagnostic_reason(f"invalid runner result: {error}", timing, obs["stderr"]),
            caveats,
        )

    row = {**base, "status": "ok"}
    # Timing fields come from the runner; copy only the schema-known keys for this scenario.
    timing_fields = ["framework_version", "build_sha", "backend", "self_reported_vram_bytes"]
    if not curve:
        timing_fields += [
            "prompt_tokens", "gen_tokens", "ttft_ms", "tpot_ms", "e2e_ms",
            "decode_tok_s", "iterations", "ttft_ms_stdev", "decode_tok_s_stdev",
        ]
    for k in timing_fields:
        if k in timing:
            row[k] = timing[k]
    # For curves, the contract has already aggregated raw samples into schema-safe points.
    if curve:
        osl = timing["osl"]
        pts = timing["curve"]
        row["osl"] = osl
        row["curve"] = pts
        # Top-level summary is the smallest-ISL point, so report/index show one headline number.
        if pts:
            p0 = pts[0]
            row["isl"] = p0["isl"]
            row["decode_tok_s"] = p0["decode_tok_s"]
            row["ttft_ms"] = (p0.get("ttft_ms") or {}).get("p50")
            row["tpot_ms"] = (p0.get("tpot_ms") or {}).get("p50")
    # Memory and utilization fields come from the observer.
    _copy_observer_fields(row, obs)
    row["caveats"] = caveats
    return row


_OBSERVER_RESULT_FIELDS = (
    "peak_vram_bytes",
    "vram_baseline_bytes",
    "peak_rss_bytes",
    "gpu_util_mean_pct",
    "gpu_util_peak_pct",
    "cpu_util_mean_pct",
    "cpu_util_peak_pct",
    "peak_power_w",
    "energy_j",
    "energy_wh",
    "util_series",
    "sampler_interval_ms",
)


def _error_row(base, observation, reason, caveats):
    row = {**base, "status": "error", "reason": reason}
    _copy_observer_fields(row, observation)
    if caveats:
        row["caveats"] = caveats
    return row


def _copy_observer_fields(row, observation):
    """Copy only available optional observer values into a stored result row."""
    for field in _OBSERVER_RESULT_FIELDS:
        if observation.get(field) is not None:
            row[field] = observation[field]


def _diagnostic_reason(prefix, timing, stderr):
    details = []
    if isinstance(timing, dict):
        for field in ("error", "reason"):
            if field in timing:
                value = timing[field]
                rendered = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
                details.append(f"runner {field}: {rendered}")
    tail = (stderr or "").strip().splitlines()[-8:]
    if tail:
        details.append("stderr: " + " | ".join(tail))
    separator = " " if prefix.endswith(":") else " | "
    reason = prefix + (separator + " | ".join(details) if details else "")
    # Strip progress bars and other non-ASCII runner output before writing snapshots and reports.
    return reason.encode("ascii", "ignore").decode()[:1200]


def _runner_caveats(timing):
    if not isinstance(timing, dict) or not isinstance(timing.get("caveats"), list):
        return []
    return [caveat for caveat in timing["caveats"] if isinstance(caveat, str)]


def _merge_caveats(*groups):
    return list(dict.fromkeys(caveat for group in groups for caveat in group))


def _caveats(model, fw_cfg):
    cv = []
    if fw_cfg.get("caveat"):
        cv.append(fw_cfg["caveat"])
    fw_prec = fw_cfg.get("precision")
    target = model.get("match_precision")
    if fw_prec and target and fw_prec != target:
        cv.append(f"precision {fw_prec} != matched target {target}")
    return cv


def is_curve(scen):
    return scen.get("kind") == CURVE_KIND


# --------------------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------------------
def cmd_matrix(args):
    manifest = load_toml(args.manifest)
    n_ok = n_unsup = 0
    print(f"{'model':22} {'scenario':12} {'framework':14} {'precision':12} support")
    for model, scen, fw, fw_cfg in cells(manifest):
        sup = is_supported(fw_cfg)
        n_ok += sup
        n_unsup += not sup
        prec = fw_cfg.get("precision", "-") if sup else "-"
        flag = "yes" if sup else f"NO ({fw_cfg.get('reason','')[:40]})"
        if args.framework and fw != args.framework:
            continue
        if args.model and model["id"] != args.model:
            continue
        print(f"{model['id']:22} {scen['id']:12} {fw:14} {prec:12} {flag}")
    print(f"\n{n_ok} runnable cells, {n_unsup} unsupported.")


def cmd_run(args):
    if args.repeats < 1:
        sys.exit("bench run: --repeats must be at least 1")
    manifest = load_toml(args.manifest)
    runners = load_toml(args.runners)
    suite = manifest.get("suite", {})
    # Checkpoint root: --models-dir flag > POOT_MODELS_DIR env > manifest `models_dir`; no other default.
    models_dir = args.models_dir or os.environ.get("POOT_MODELS_DIR") or suite.get("models_dir")
    if not models_dir:
        sys.exit("bench run: no models directory: pass --models-dir or set POOT_MODELS_DIR")

    sys.path.insert(0, str(HERE))  # so run_cell can import observer
    results_root = Path(args.results_dir or (BENCH_ROOT / "results"))
    run_id = args.run_id
    for repeat in range(1, args.repeats + 1):
        env = capture_env(args.manifest)
        run_id = run_id or _default_run_id(env, args.model)
        # One repeat is the run itself. Several are one run directory each (`<run-id>-r1` ...): each is a valid
        # input to `compare`, which counts one run per repeat of a cell.
        repeat_id = run_id if args.repeats == 1 else f"{run_id}-r{repeat}"
        print(f"[run] repeat {repeat}/{args.repeats}: {repeat_id}", file=sys.stderr, flush=True)
        _run_matrix(args, manifest, runners, suite, models_dir, env, results_root / repeat_id)


def _run_matrix(args, manifest, runners, suite, models_dir, env, run_dir):
    """Run the selected cells once and write the run directory: env.json, results.jsonl and report.md."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    results_path = run_dir / "results.jsonl"

    rows = []
    with open(results_path, "w") as out:
        for model, scen, fw, fw_cfg in cells(manifest):
            if args.framework and fw != args.framework:
                continue
            if args.model and model["id"] != args.model:
                continue
            if args.scenario and scen["id"] != args.scenario:
                continue
            if args.skip_unsupported and not is_supported(fw_cfg):
                continue
            label = f"{model['id']} / {scen['id']} / {fw}"
            print(f"[run] {label} ...", file=sys.stderr, flush=True)
            row = run_cell(model, scen, fw, fw_cfg, runners, models_dir, suite,
                           interval_ms=args.sample_interval_ms, device=env["gpu_name"],
                           overrides={"gen_tokens": args.gen_tokens, "warmup": args.warmup,
                                      "iters": args.iters})
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows.append(row)
            print(f"      -> {row['status']}"
                  + (f"  {row.get('decode_tok_s','?')} tok/s" if row["status"] == "ok" else
                     f"  ({row.get('reason','')[:60]})"), file=sys.stderr)

    report = render_report(env, rows)
    (run_dir / "report.md").write_text(report)
    print(f"\nrun complete: {run_dir}\n  results.jsonl ({len(rows)} rows), env.json, report.md",
          file=sys.stderr)
    print(report)


def _default_run_id(env, model=None):
    captured = env.get("captured_at_utc") or ""
    date = captured[:10] or "undated"
    # The time of day keeps repeated runs of one matrix (a compare needs several) in distinct directories.
    clock = captured[11:19].replace(":", "") or "000000"
    gpu = re.sub(r"[^a-z0-9]+", "-", (env.get("gpu_name") or "nogpu").lower()).replace("nvidia-", "").strip("-")
    # Include the model for single-model sweeps (the orchestrator runs one `bench run` per model) so runs
    # do not overwrite one date-gpu-clock dir.
    mid = (model or "").replace("/", "-").replace(".", "-")
    return f"{date}-{gpu}-{mid}-{clock}" if mid else f"{date}-{gpu}-{clock}"


def cmd_report(args):
    run_dir = Path(args.run_dir)
    env = json.loads((run_dir / "env.json").read_text()) if (run_dir / "env.json").exists() else {}
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    print(render_report(env, rows))


def load_runs(results_dir):
    """Every snapshot under `results_dir` as (run id, env, rows), in run-id order."""
    runs = []
    for d in sorted(Path(results_dir).iterdir()):
        rj = d / "results.jsonl"
        if not rj.is_dir() and rj.exists():
            env = json.loads((d / "env.json").read_text()) if (d / "env.json").exists() else {}
            rows = [json.loads(l) for l in rj.read_text().splitlines() if l.strip()]
            runs.append((d.name, env, rows))
    return runs


def _run_date(env):
    """The date a run was captured, as a date; None when the snapshot recorded none."""
    captured = (env.get("captured_at_utc") or "")[:10]
    try:
        return datetime.strptime(captured, "%Y-%m-%d").date()
    except ValueError:
        return None


def render_index(runs):
    """The progress index for `runs` (see `load_runs`), a pure function of the results.

    Staleness is measured against the newest run in the results, never the clock, so the index only changes
    when a result does.
    """
    dates = [d for d in (_run_date(env) for _, env, _ in runs) if d]
    newest = max(dates) if dates else None
    out = ["# poot benchmark index", "",
           "Every committed run, grouped by model and scenario. The point is to watch poot's column move as we "
           "optimize. Regenerate with `bench index`. Full tables: `bench report results/<run>`; to test a "
           "change, `bench compare --baseline <runs> --candidate <runs>` (5 repeats per side). `behind` is "
           "how many days a run is older than the newest run here.", ""]
    # Group by (model, scenario); one table, rows = runs.
    groups = {}
    for run_id, env, rows in runs:
        for r in rows:
            if r.get("status") != "ok":
                continue
            groups.setdefault((r["model"], r["scenario"]), {}).setdefault(run_id, {})[r["framework"]] = r
            groups[(r["model"], r["scenario"])].setdefault("_env", {})[run_id] = env

    for (model, scenario), per_run in sorted(groups.items()):
        env_by_run = per_run.pop("_env", {})
        out += [f"## {model} - {scenario}", "",
                "| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | "
                "poot VRAM GiB | fastest baseline | poot vs fastest |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for run_id in sorted(per_run):
            cells = per_run[run_id]
            env = env_by_run.get(run_id, {})
            poot = cells.get("poot")
            baselines = {k: v for k, v in cells.items() if k != "poot" and v.get("decode_tok_s")}
            best = max(baselines.items(), key=lambda kv: kv[1]["decode_tok_s"]) if baselines else None
            ptoks = poot.get("decode_tok_s") if poot else None
            ratio = (f"{ptoks / best[1]['decode_tok_s']:.3f}x"
                     if poot and best and best[1]["decode_tok_s"] else "-")
            date = _run_date(env)
            behind = str((newest - date).days) if date and newest else "-"
            # Snapshots from before rows carried `build_sha` recorded the commit in env.json.
            commit = (poot.get("build_sha") or env.get("poot_git_sha")) if poot else None
            out.append(
                f"| {run_id} | {date or ''} | {behind} | {(poot.get('backend') or '-') if poot else '-'} | "
                f"{_short_sha(commit)} | "
                f"{(env.get('gpu_name') or '').replace('NVIDIA ', '')} | "
                f"{_num(ptoks) if poot else '-'} | {_num(poot.get('ttft_ms')) if poot else '-'} | "
                f"{_gib(poot.get('peak_vram_bytes')) if poot else '-'} | "
                f"{(best[0] + ' ' + _num(best[1]['decode_tok_s']) + ' tok/s') if best else '-'} | {ratio} |")
        out.append("")
    return "\n".join(out)


def cmd_index(args):
    """Scan every results/<run>/ snapshot into results/INDEX.md."""
    results_dir = Path(args.results_dir or (BENCH_ROOT / "results"))
    runs = load_runs(results_dir)
    text = render_index(runs)
    (results_dir / "INDEX.md").write_text(text)
    print(f"wrote {results_dir / 'INDEX.md'} ({len(runs)} run(s))")
    print(text)


def cmd_compare(args):
    rows_a = [row for run_dir in args.baseline for row in _load_rows(run_dir)]
    rows_b = [row for run_dir in args.candidate for row in _load_rows(run_dir)]
    cells = compare_cells(rows_a, rows_b)
    print(render_compare(args.baseline, args.candidate, cells))
    return compare_exit_code(cells)


def _load_rows(run_dir):
    p = Path(run_dir) / "results.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------------------------------
# Compare: cells, the noise band and the exit code (pure stdlib)
# --------------------------------------------------------------------------------------------------
# Repeats of one cell (one runner invocation each, across as many run directories) that make a band.
NOISE_REPEATS = 5

REGRESSION = "regression"
IMPROVEMENT = "improvement"
WITHIN_NOISE = "within noise"
UNVERIFIED = "unverified"
INVALID = "invalid"
CANDIDATE_FAILED = "candidate failed"
NO_BASELINE = "no baseline"
NEW = "new"
NOT_RUN = "not run"

_FAILING_VERDICTS = frozenset({REGRESSION, INVALID, CANDIDATE_FAILED})
_COMPARED_VERDICTS = frozenset({REGRESSION, IMPROVEMENT, WITHIN_NOISE})


def cell_key(row):
    """What makes two rows the same cell: the same model and scenario on the same engine, backend and device.

    A backend or device the row does not record is part of the key as None, never a wildcard.
    """
    return (row["model"], row["scenario"], row["framework"], row.get("backend"), row.get("device"))


def _is_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare_cells(rows_a, rows_b):
    """Compare a baseline against a candidate, one verdict per cell.

    The noise model: the baseline's repeated runs of a cell give a median and a range (max - min); the band
    is median +/- range. A candidate median more than one further range outside the band is a change
    (median_b < median_a - 2 * range_a is a regression, above median_a + 2 * range_a an improvement); anything
    closer is within noise. A baseline whose runs are all identical has range 0; the candidate's range is used
    in its place, and when that is 0 too the cell is unverified. Only the headline decode tok/s is compared. A cell with fewer than NOISE_REPEATS
    runs on either side, or whose runs mix binaries, has no honest band and is unverified.
    """
    cells_a, cells_b = _rows_by_cell(rows_a), _rows_by_cell(rows_b)

    def order(key):
        return tuple("" if part is None else part for part in key)

    return [
        _compare_cell(key, cells_a.get(key), cells_b.get(key))
        for key in sorted(set(cells_a) | set(cells_b), key=order)
    ]


def _rows_by_cell(rows):
    cells = {}
    for row in rows:
        cells.setdefault(cell_key(row), []).append(row)
    return cells


def _distinct(rows, field):
    return sorted({row[field] for row in rows if row.get(field)})


def _compare_cell(key, rows_a, rows_b):
    cell = {"key": key, "verdict": None, "note": "", "n_a": 0, "n_b": 0,
            "median_a": None, "range_a": None, "median_b": None, "range_b": None,
            "vram_a": None, "vram_b": None, "builds_a": [], "builds_b": []}
    ok_a = [r for r in rows_a or [] if r.get("status") == "ok"]
    ok_b = [r for r in rows_b or [] if r.get("status") == "ok"]
    cell.update(n_a=len(ok_a), n_b=len(ok_b),
                builds_a=_distinct(ok_a, "build_sha"), builds_b=_distinct(ok_b, "build_sha"))

    def settle(verdict, note=""):
        cell.update(verdict=verdict, note=note)
        return cell

    if rows_b is None:
        return settle(NOT_RUN, "cell is not in the candidate")
    if rows_a is None:
        return settle(NEW, "cell is not in the baseline")
    bad = [side for side, ok in (("baseline", ok_a), ("candidate", ok_b))
           if not all(_is_finite(r.get("decode_tok_s")) for r in ok)]
    if bad:
        return settle(INVALID, f"non-finite decode tok/s in the {' and '.join(bad)}")
    if not ok_a:
        return settle(NO_BASELINE, "the baseline has no successful run of this cell")
    if not ok_b:
        reasons = [r.get("reason", "") for r in rows_b if r.get("status") != "ok"]
        return settle(CANDIDATE_FAILED, (reasons[0] if reasons else "no successful run")[:80])

    samples_a = [r["decode_tok_s"] for r in ok_a]
    samples_b = [r["decode_tok_s"] for r in ok_b]
    cell.update(median_a=statistics.median(samples_a), range_a=max(samples_a) - min(samples_a),
                median_b=statistics.median(samples_b), range_b=max(samples_b) - min(samples_b),
                vram_a=_median_of(ok_a, "peak_vram_bytes"), vram_b=_median_of(ok_b, "peak_vram_bytes"))

    if len(ok_a) < NOISE_REPEATS or len(ok_b) < NOISE_REPEATS:
        return settle(UNVERIFIED, f"needs {NOISE_REPEATS} runs per side, have {len(ok_a)} and {len(ok_b)}")
    for side, ok in (("baseline", ok_a), ("candidate", ok_b)):
        if len({r.get("binary_sha256") for r in ok}) > 1:
            return settle(UNVERIFIED, f"the {side} runs mix binaries")

    # Identical baseline runs show no noise to size a band from (a real measurement is never that steady, so
    # the values are quantized or the runner is not measuring); the candidate's spread is then the only
    # observed noise. With neither, any difference would be judged against a zero-width band: unverified.
    noise = cell["range_a"] or cell["range_b"]
    if not noise:
        return settle(UNVERIFIED, "the runs of both sides are identical, so no noise was observed")
    margin = 2 * noise
    if cell["median_b"] < cell["median_a"] - margin:
        return settle(REGRESSION)
    if cell["median_b"] > cell["median_a"] + margin:
        return settle(IMPROVEMENT)
    return settle(WITHIN_NOISE)


def _median_of(rows, field):
    values = [r[field] for r in rows if _is_finite(r.get(field))]
    return statistics.median(values) if values else None


def compare_exit_code(cells):
    """1 when a cell regressed, failed or is invalid; 2 when nothing failed but a cell (or the whole compare)
    could not be verified; 0 when at least one cell was compared and none regressed."""
    if any(cell["verdict"] in _FAILING_VERDICTS for cell in cells):
        return 1
    if any(cell["verdict"] == UNVERIFIED for cell in cells):
        return 2
    if not any(cell["verdict"] in _COMPARED_VERDICTS for cell in cells):
        return 2
    return 0


# --------------------------------------------------------------------------------------------------
# Rendering (pure stdlib)
# --------------------------------------------------------------------------------------------------
def _gib(b):
    return f"{b / 2**30:.2f}" if isinstance(b, (int, float)) and b else "-"


def _num(x, fmt="{:.1f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else "-"


def render_curve(grp):
    """The decode-degradation tables: decode tok/s and TPOT-p50 vs context length (ISL), per engine."""
    curved = [r for r in grp if r.get("status") == "ok" and r.get("curve")]
    if not curved:
        return []
    isls = sorted({p["isl"] for r in curved for p in r["curve"]})

    def cell(r, isl, field, sub="p50"):
        for p in r["curve"]:
            if p["isl"] == isl:
                v = p.get(field)
                return _num(v.get(sub) if isinstance(v, dict) else v)
        return "-"

    head = "| framework | " + " | ".join(f"ISL {i}" for i in isls) + " |"
    sep = "|---|" + "---|" * len(isls)
    lines = ["", "**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**",
             "", head, sep]
    for r in sorted(curved, key=lambda x: x["framework"]):
        lines.append("| " + _framework_label(r) + " | "
                     + " | ".join(cell(r, i, "decode_tok_s") for i in isls) + " |")
    lines += ["", "**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**",
              "", head, sep]
    for r in sorted(curved, key=lambda x: x["framework"]):
        lines.append("| " + _framework_label(r) + " | "
                     + " | ".join(cell(r, i, "tpot_ms", "p50") for i in isls) + " |")
    lines.append("")
    return lines


def render_report(env, rows):
    lines = ["# poot benchmark report", ""]
    if env:
        lines += [
            f"- GPU: {env.get('gpu_name','?')} (driver {env.get('gpu_driver','?')}, "
            f"{env.get('gpu_memory_total','?')})",
            f"- captured: {env.get('captured_at_utc','?')}",
            "",
        ]
    builds = sorted({r["build_sha"] for r in rows if r.get("build_sha")})
    if builds:
        lines += [f"- poot build: {', '.join(_short_sha(b) for b in builds)}", ""]
    # Group by (model, scenario); one table per group, frameworks as rows.
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["scenario"]), []).append(r)

    for (model, scenario), grp in sorted(groups.items()):
        lines += [f"## {model} - {scenario}", "",
                  "| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | "
                  "peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(grp, key=lambda x: x["framework"]):
            if r["status"] == "ok":
                notes = "; ".join(r.get("caveats", [])) or ""
                lines.append(
                    f"| {_framework_label(r)} | {r['precision']} | ok | {_num(r.get('ttft_ms'))} | "
                    f"{_num(r.get('tpot_ms'))} | {_num(r.get('decode_tok_s'))} | "
                    f"{_gib(r.get('peak_vram_bytes'))} | {_gib(r.get('peak_rss_bytes'))} | "
                    f"{_num(r.get('peak_power_w'))} | {_num(r.get('energy_wh'), '{:.3f}')} | {notes} |")
            else:
                lines.append(
                    f"| {_framework_label(r)} | {r.get('precision','-')} | {r['status']} | - | - | - | "
                    f"- | - | - | - | {r.get('reason','')[:80]} |")
        lines.append("")
        lines += render_curve(grp)
    # Footer.
    lines += ["---", "",
              "_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under "
              "the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler "
              "(NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. "
              "Timing comes from each runner. Cells with a precision or "
              "format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool "
              "reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._"]
    return "\n".join(lines)


def _short_sha(sha):
    """First 8 characters of a build sha, keeping a `-dirty` mark; `-` when there is none."""
    if not sha:
        return "-"
    return sha[:8] + ("-dirty" if sha.endswith("-dirty") else "")


def _framework_label(row):
    return f"{row['framework']} ({row['backend']})" if row.get("backend") else row["framework"]


def _band(median, spread):
    return "-" if median is None else f"{median:.1f} +/- {spread:.1f}"


def render_compare(baseline_dirs, candidate_dirs, cells):
    lines = ["# benchmark diff", "",
             *(f"- baseline: `{d}`" for d in baseline_dirs),
             *(f"- candidate: `{d}`" for d in candidate_dirs), "",
             f"A cell is (model, scenario, framework, backend, device) and needs {NOISE_REPEATS} runs per "
             "side. tok/s is the median +/- the range (max - min) over those runs. A candidate median more "
             "than one range beyond the baseline's band (median +/- range) is a regression or an "
             "improvement. Speed is recorded, not gated.", "",
             "| model | scenario | framework | backend | device | runs A/B | tok/s A | tok/s B | delta % | "
             "verdict | build A | build B | VRAM A GiB | VRAM B GiB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in cells:
        model, scenario, framework, backend, device = cell["key"]
        delta = (
            f"{100.0 * (cell['median_b'] - cell['median_a']) / cell['median_a']:+.1f}"
            if cell["median_a"] and cell["median_b"] is not None else "-"
        )
        verdict = cell["verdict"] + (f" ({cell['note']})" if cell["note"] else "")
        lines.append(
            f"| {model} | {scenario} | {framework} | {backend or '-'} | {device or '-'} | "
            f"{cell['n_a']}/{cell['n_b']} | {_band(cell['median_a'], cell['range_a'])} | "
            f"{_band(cell['median_b'], cell['range_b'])} | {delta} | {verdict} | "
            f"{', '.join(_short_sha(b) for b in cell['builds_a']) or '-'} | "
            f"{', '.join(_short_sha(b) for b in cell['builds_b']) or '-'} | "
            f"{_gib(cell['vram_a'])} | {_gib(cell['vram_b'])} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="poot cross-framework inference benchmark suite")
    p.add_argument("--manifest", default=str(BENCH_ROOT / "manifest.toml"))
    p.add_argument("--runners", default=str(BENCH_ROOT / "runners.toml"))
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("matrix", help="enumerate the model x framework x scenario cells")
    m.add_argument("--model")
    m.add_argument("--framework", choices=FRAMEWORKS)
    m.set_defaults(func=cmd_matrix)

    r = sub.add_parser("run", help="run the matrix and write a result snapshot")
    r.add_argument("--model")
    r.add_argument("--framework", choices=FRAMEWORKS)
    r.add_argument("--scenario")
    r.add_argument("--models-dir")
    r.add_argument("--results-dir")
    r.add_argument("--run-id")
    r.add_argument("--repeats", type=int, default=1, metavar="N",
                   help="run the whole matrix N times, one run directory per repeat (`<run-id>-r1` ...); "
                        f"`compare` needs {NOISE_REPEATS} repeats of a side")
    r.add_argument("--skip-unsupported", action="store_true",
                   help="omit unsupported cells instead of recording them")
    r.add_argument("--sample-interval-ms", type=float, default=50.0)
    r.add_argument("--gen-tokens", type=int, help="override the scenario gen-tokens (quick smoke runs)")
    r.add_argument("--warmup", type=int, help="override the suite warmup iters")
    r.add_argument("--iters", type=int, help="override the suite timed iters")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="render a run directory to a table")
    rp.add_argument("run_dir")
    rp.set_defaults(func=cmd_report)

    c = sub.add_parser("compare", help="diff a baseline against a candidate (exit 1: regression, 2: unverified)")
    c.add_argument("--baseline", nargs="+", required=True, metavar="RUN_DIR",
                   help=f"run directories of the baseline: repeats of one matrix, at least {NOISE_REPEATS}")
    c.add_argument("--candidate", nargs="+", required=True, metavar="RUN_DIR",
                   help=f"run directories of the candidate: repeats of one matrix, at least {NOISE_REPEATS}")
    c.set_defaults(func=cmd_compare)

    ix = sub.add_parser("index", help="regenerate results/INDEX.md from all snapshots (progress view)")
    ix.add_argument("--results-dir")
    ix.set_defaults(func=cmd_index)

    args = p.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
