"""Splits a stored run's memory into what loading the model took and what inference holds.

The observer's `peak_rss_bytes` and `peak_vram_bytes` are maxima over the whole process lifetime, so they include the
load of the checkpoint (file-backed pages, staging buffers). Every result also stores the observer's 1 Hz `util_series`
(`[t_s, gpu_pct, cpu_pct, vram_bytes, rss_bytes, power_w]`). This module cuts that series at the end of the load plus the
warmup and reports the peak of each part, so every stored result gets the split without a re-run.

Inference is what follows the load and the warmup request of the run: the allocator, KV cache and kernels are at steady
state and the timed iterations run. The load ends at

* the engine's own `preparation.open_ms` when the row records it (one process, one load), else
* the last finished load the series shows: the peak sample before the last fall of host RSS by a quarter of the series
  peak (the load's high resident set is released once the weights are on the device; a fall at the very last sample is
  the process exiting and is not one). llama.cpp restarts its server for every curve point, so its series holds several
  loads and only the last one bounds the inference window.

A series with no such fall and no recorded open time, or with too few samples after the cut, has no usable boundary: the run
keeps its lifetime peak and is labelled as including the load.

The series is sampled at 1 Hz, the lifetime peak at the sampler interval, so a phase peak can sit slightly below the
lifetime peak it belongs to.
"""

import tomllib
from pathlib import Path

# A finished load shows as host RSS falling by at least this fraction of the series' peak between two consecutive samples (the
# mapped checkpoint is released once its tensors are on the device, or the staging buffers are freed)...
LOAD_DROP_FRACTION = 0.25
# ...and by at least this much, so noise on a series of a few hundred MiB is not a load.
LOAD_DROP_MIN_BYTES = 512 * 2**20
# Samples after the cut a usable inference window needs.
MIN_INFERENCE_SAMPLES = 3

T, VRAM, RSS = 0, 3, 4


def warmup_iters(manifest):
    """The untimed warmup generations every runner makes before a timed series, as the manifest's suite declares."""
    with open(manifest, "rb") as f:
        return tomllib.load(f)["suite"]["warmup_iters"]


def warmup_seconds(row, iters):
    """The duration of the warmup generations: `iters` requests of the run's first curve point (first-token time plus
    one decode step per generated token), or zero for a row with no curve."""
    curve = row.get("curve") or []
    if not curve or not iters:
        return 0.0
    first = curve[0]
    return iters * (first["ttft_ms"]["p50"] + first["osl"] * first["tpot_ms"]["p50"]) / 1000.0


def load_end_seconds(row, series):
    """When the load ended on the series' clock, or None when neither the row nor the series says."""
    open_ms = (row.get("preparation") or {}).get("open_ms")
    if open_ms:
        return open_ms / 1000.0
    resident = [sample[RSS] for sample in series if sample[RSS] is not None]
    if not resident:
        return None
    peak = max(resident)
    drop = max(LOAD_DROP_FRACTION * peak, LOAD_DROP_MIN_BYTES)
    # The last sample is the process exiting, not a load that finished.
    ends = [prev[T] for prev, cur in zip(series, series[1:-1]) if None not in (prev[RSS], cur[RSS]) and prev[RSS] - cur[RSS] >= drop]
    return ends[-1] if ends else None


def last_load(series, end, recorded):
    """The samples of the last load, ending at `end`. With a recorded open time the engine loaded once, from the start of
    the series. Otherwise the load is the ramp into the peak sample at `end`: the samples just before it that hold at
    least half its resident set, so the inference of an earlier llama.cpp invocation is not counted as load."""
    upto = [s for s in series if s[T] <= end]
    if recorded:
        return upto
    ramp = [upto[-1]]
    for sample in reversed(upto[:-1]):
        if sample[RSS] is None or sample[RSS] * 2 < upto[-1][RSS]:
            break
        ramp.append(sample)
    return ramp


def memory_phases(row, warmup_iters_count):
    """`{"load": {...}, "inference": {...}}` with `rss_bytes` and `vram_bytes` peaks of each part, or None when the run
    has no usable boundary. The device use before the run (`vram_baseline_bytes`) is not the run's and is subtracted
    from the GPU figures, as the page does for the lifetime peak."""
    series = row.get("util_series")
    if not series or len(series) < 2:
        return None
    load_end = load_end_seconds(row, series)
    if load_end is None:
        return None
    start = load_end + warmup_seconds(row, warmup_iters_count)
    inference = [s for s in series if s[T] > start]
    load = last_load(series, load_end, recorded=bool((row.get("preparation") or {}).get("open_ms")))
    if len(inference) < MIN_INFERENCE_SAMPLES or not load:
        return None
    base = row.get("vram_baseline_bytes") or 0

    def peak(samples, column):
        values = [s[column] for s in samples if s[column] is not None]
        return max(values) if values else None

    def peaks(samples):
        vram = peak(samples, VRAM)
        return {"rss_bytes": peak(samples, RSS), "vram_bytes": None if vram is None else max(0, vram - base)}

    return {"load": peaks(load), "inference": peaks(inference)}
