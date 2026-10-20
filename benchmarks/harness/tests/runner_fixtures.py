"""Runner-result fixtures shared by the harness tests: valid poot payloads and a fake observed run."""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from performance_receipt import PREPARATION_LAYERS  # noqa: E402

BUILD_SHA = "0123456789abcdef0123456789abcdef01234567"
CHECKPOINT_DIGEST = "blake3:" + "ab" * 32

# What a repository-built runner states beyond its timing: the build (sha, profile, features, compiler), the
# adapter and driver the executor reported, the digest of the loaded weights and the compute dtype.
POOT_IDENTITY = {
    "build_sha": BUILD_SHA,
    "build_profile": "release",
    "build_features": "",
    "rustc_version": "rustc 1.101.0-nightly (fixture)",
    "adapter": "Fixture GPU",
    "driver_version": "fixture-driver 1.0",
    "checkpoint_digest": CHECKPOINT_DIGEST,
    "compute_dtype": "f32",
}

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


def timed_request(stamps, complete, *, steps=None):
    """A request's raw boundaries from token delivery times and the completion time (all ms from request start)."""
    return {
        "tokens": len(stamps),
        "first_token_ms": stamps[0] if stamps else None,
        "last_token_ms": stamps[-1] if stamps else None,
        "ttft_ms": stamps[0] if stamps else 0.0,
        "itl_ms": [b - a for a, b in zip(stamps, stamps[1:])],
        "finish_ms": complete - (stamps[-1] if stamps else 0.0),
        "e2e_ms": complete,
        "steps": len(stamps) + 1 if steps is None else steps,
    }


def preparation_block(state=None):
    """What a runner reports for preparation: the plan layers it counts, the rest Unknown (null)."""
    counted = {"plan": 2, "entries": 2, "recordings": 2}
    layers = {
        layer: {"prepare": counted.get(layer), "repeat": 0 if layer in counted else None}
        for layer in PREPARATION_LAYERS
    }
    return {"state": state, "artifact_entries_at_start": None, "open_ms": 12.5, "layers": layers}


def valid_single():
    return {
        "framework": "poot",
        "framework_version": "fixture-1",
        **POOT_IDENTITY,
        "family": "qwen2",
        "profiled": False,
        "backend": "ptx",
        "precision": "bf16",
        "prompt_tokens": 3,
        "gen_tokens": 4,
        "iterations": 2,
        "runs": [timed_request([1.0, 3.0, 5.0, 7.0], 7.0), timed_request([1.0, 3.0, 5.0, 7.0], 7.0)],
        "ttft_ms": 1.0,
        "tpot_ms": 2.0,
        "e2e_ms": 7.0,
        "decode_tok_s": 500.0,
        "ttft_ms_stdev": 0.1,
        "decode_tok_s_stdev": 0.2,
        "self_reported_vram_bytes": 99,
        "preparation": preparation_block(),
        "caveats": ["runner caveat"],
    }


def curve_point(isl):
    return {
        "isl": isl,
        "prompt_tokens": isl,
        "iters": [
            timed_request([float(isl), isl + 2.0, isl + 4.0], float(isl + 6)),
            timed_request([isl + 1.0, isl + 4.0, isl + 7.0], float(isl + 9)),
        ],
    }


def valid_curve():
    return {
        "framework": "poot",
        "framework_version": "fixture-1",
        **POOT_IDENTITY,
        "family": "qwen2",
        "profiled": False,
        "backend": "ptx",
        "precision": "bf16",
        "mode": "decode-curve",
        "osl": 3,
        "curve": [curve_point(4), curve_point(8)],
        "preparation": preparation_block(),
        "caveats": ["runner caveat"],
    }



ROW_LEVEL_FRAMEWORKS = ("transformers", "vllm", "llamacpp", "candle")


def row_level_single(framework):
    """What a runner that claims row-level timing only reports: medians, no per-request boundaries."""
    return {
        "framework": framework,
        "framework_version": "fixture-1",
        "precision": "bf16",
        "backend": "cuda",
        "prompt_tokens": 3,
        "gen_tokens": 4,
        "iterations": 2,
        "ttft_ms": 1.0,
        "tpot_ms": 2.0,
        "e2e_ms": 7.0,
        "decode_tok_s": 500.0,
    }


def row_level_curve(framework):
    """The same, for decode-curve: per iteration only ttft, e2e and the osl - 1 gaps."""
    def point(isl):
        return {
            "isl": isl,
            "prompt_tokens": isl,
            "iters": [
                {"ttft_ms": float(isl), "e2e_ms": float(isl + 6), "itl_ms": [2.0, 2.0]},
                {"ttft_ms": isl + 1.0, "e2e_ms": isl + 9.0, "itl_ms": [3.0, 3.0]},
            ],
        }

    return {
        "framework": framework,
        "framework_version": "fixture-1",
        "precision": "bf16",
        "backend": "cuda",
        "mode": "decode-curve",
        "osl": 3,
        "curve": [point(4), point(8)],
    }
