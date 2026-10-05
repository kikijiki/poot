"""Runner-result fixtures shared by the harness tests: valid poot payloads and a fake observed run."""

import json
import subprocess

BUILD_SHA = "0123456789abcdef0123456789abcdef01234567"

OPTIONAL_OBSERVER_FIELDS = (
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
)


def observation(
    payload,
    *,
    returncode=0,
    timed_out=False,
    stderr="fixture stderr",
    observer_metrics=True,
):
    metrics: dict[str, object] = {
        "peak_vram_bytes": 101,
        "vram_baseline_bytes": 11,
        "peak_rss_bytes": 202,
        "gpu_util_mean_pct": 30.0,
        "gpu_util_peak_pct": 40.0,
        "cpu_util_mean_pct": 50.0,
        "cpu_util_peak_pct": 60.0,
        "peak_power_w": 70.0,
        "energy_j": 80.0,
        "energy_wh": 0.02,
        "util_series": [[0.0, 30.0, 50.0, 101, 202, 70.0]],
    }
    if not observer_metrics:
        metrics = {field: None for field in OPTIONAL_OBSERVER_FIELDS}
    return {
        "stdout": json.dumps(payload) + "\n",
        "stderr": stderr,
        "returncode": returncode,
        "timed_out": timed_out,
        **metrics,
        "sampler_interval_ms": 50.0,
        "samples": 1,
    }


def run_fixture_runner(cmd, **_kwargs):
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    result = observation({})
    result.update(
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
    return result


def valid_single():
    return {
        "framework": "poot",
        "framework_version": "fixture-1",
        "build_sha": BUILD_SHA,
        "backend": "ptx",
        "precision": "bf16",
        "prompt_tokens": 3,
        "gen_tokens": 4,
        "iterations": 2,
        "ttft_ms": 1.0,
        "tpot_ms": 2.0,
        "e2e_ms": 7.0,
        "decode_tok_s": 500.0,
        "ttft_ms_stdev": 0.1,
        "decode_tok_s_stdev": 0.2,
        "self_reported_vram_bytes": 99,
        "caveats": ["runner caveat"],
    }


def curve_point(isl):
    return {
        "isl": isl,
        "prompt_tokens": isl,
        "iters": [
            {"ttft_ms": float(isl), "e2e_ms": float(isl + 6), "itl_ms": [2.0, 2.0]},
            {"ttft_ms": float(isl + 1), "e2e_ms": float(isl + 9), "itl_ms": [3.0, 3.0]},
        ],
    }


def valid_curve():
    return {
        "framework": "poot",
        "framework_version": "fixture-1",
        "build_sha": BUILD_SHA,
        "backend": "ptx",
        "precision": "bf16",
        "mode": "decode-curve",
        "osl": 3,
        "curve": [curve_point(4), curve_point(8)],
        "caveats": ["runner caveat"],
    }

