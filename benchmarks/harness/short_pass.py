"""Which short prefill passes of the committed results carry enough timed evidence, for every framework (POOT-1529).

A first-token pass under `SHORT_PASS_MS` is a host-launch measurement of a fraction of a second, and an engine can
stall for several seconds right after it loads (llama.cpp's ROCm build does: its GPU kernels that read host-mapped
tensors run hundreds of times slower while the kernel compacts memory). A handful of iterations can all fall in one
such episode. The contract therefore asks every framework the same of a short pass: its timed iterations span at
least `TIMED_WINDOW_S`, or the iterations are flat (spread within `FLAT_SPREAD` of the median over at least
`MIN_FLAT_ITERATIONS` of them), which shows the pass has no episode for a longer window to average out. A point that
is neither is `uncovered`: its median is not evidence yet.

The llama.cpp runner meets the first branch by construction (runners/llamacpp/run.py repeats a short pass for the
window). poot, candle and transformers run three timed iterations, so they meet the second.
"""

SHORT_PASS_MS = 2000.0
TIMED_WINDOW_S = 20.0
FLAT_SPREAD = 0.10
MIN_FLAT_ITERATIONS = 3


def coverage(point):
    """`(covered, window_s, spread)` of one committed curve point (`iters` is the timed iteration count, `ttft_ms` the
    summary of their first-token times); None for a pass that is not short."""
    ttft, n = point.get("ttft_ms"), point.get("iters")
    if not isinstance(ttft, dict) or not isinstance(n, int) or not ttft.get("p50"):
        return None
    median = ttft["p50"]
    if median >= SHORT_PASS_MS:
        return None
    window_s = n * median / 1000.0
    spread = (ttft["max"] - ttft["min"]) / median
    flat = n >= MIN_FLAT_ITERATIONS and spread <= FLAT_SPREAD
    return (window_s >= TIMED_WINDOW_S or flat), window_s, spread


def uncovered(runs):
    """Every uncovered short pass in `runs` (see bench.load_runs): (run id, row, point, window_s, spread)."""
    out = []
    for run_id, _, rows in runs:
        for row in rows:
            if row.get("status") != "ok":
                continue
            for point in row.get("curve") or []:
                got = coverage(point)
                if got is not None and not got[0]:
                    out.append((run_id, row, point, got[1], got[2]))
    return out
