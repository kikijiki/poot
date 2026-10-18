"""POOT-1184: flatten a pod-sweep results directory into one TSV (stdout).

Reads only what the sweep wrote: raw/poot_curve*.json (poot-bench-runner result lines), raw/<engine>_isl*_c*.json
(`vllm bench serve --save-result`, run against poot-serve, vLLM and llama-server alike) and
raw/llamacpp_bench_{pp,tg}.json (`llama-bench -o json`). A missing or unparsable file is a row with a `note`, never
a silent omission. Usage: python -I summarize.py <results-dir>
"""

import glob
import json
import os
import statistics
import sys

COLUMNS = ["engine", "kind", "isl", "osl", "conc", "ttft_ms", "prefill_tok_s", "tpot_ms", "decode_tok_s",
           "out_tok_s_total", "req_s", "completed", "mean_out_tokens", "note"]


def row(**kw):
    return "\t".join("" if kw.get(c) is None else str(kw[c]) for c in COLUMNS)


def median(xs):
    return round(statistics.median(xs), 3) if xs else None


def poot_rows(path, kind):
    try:
        with open(path) as f:
            line = json.load(f)
    except (OSError, ValueError) as e:
        yield row(engine="poot", kind=kind, note=f"unreadable {os.path.basename(path)}: {e}")
        return
    if line.get("status") not in (None, "ok"):
        yield row(engine="poot", kind=kind, note=f"status={line.get('status')} {line.get('reason', '')}"[:200])
        return
    osl = line.get("osl")
    for note in line.get("caveats", []):
        yield row(engine="poot", kind=kind, note=note)
    fit = line.get("roofline_fit")
    if fit and fit.get("kind") == "unknown":
        yield row(engine="poot", kind=kind + "_fit", note=f"unknown: {fit.get('reason')}")
    for point in line.get("curve", []):
        runs = point["iters"]
        ttft = median([r["ttft_ms"] for r in runs if r.get("ttft_ms")])
        tpot = median([r["tpot_ms"] for r in runs if r.get("tpot_ms")])
        yield row(engine="poot", kind=kind, isl=point["isl"], osl=osl, conc=1, ttft_ms=ttft,
                  prefill_tok_s=round(point["isl"] / ttft * 1000, 1) if ttft else None, tpot_ms=tpot,
                  decode_tok_s=round(1000 / tpot, 2) if tpot else None, completed=len(runs),
                  mean_out_tokens=median([r["tokens"] for r in runs]))
        roofline = point.get("roofline")
        if roofline:
            # An uncalibrated target reports {kind: unknown, reason}: say so instead of printing four Nones.
            keys = ("kind", "reason") if roofline.get("kind") == "unknown" else \
                ("ms_per_token", "bytes_per_token", "achieved_gb_per_s", "fraction_of_ceiling")
            yield row(engine="poot", kind=kind + "_ceiling", isl=point["isl"], conc=1,
                      note="; ".join(f"{k}={roofline.get(k)}" for k in keys))
        device = point.get("device_time")
        if device:
            yield row(engine="poot", kind=kind + "_device_time", isl=point["isl"], conc=1, note=json.dumps(device))


def client_rows(path):
    name = os.path.basename(path)[:-5]
    engine, isl, conc = name.split("_")
    isl, conc = int(isl[3:]), int(conc[1:])
    try:
        with open(path) as f:
            r = json.load(f)
    except (OSError, ValueError) as e:
        yield row(engine=engine, kind="serve", isl=isl, conc=conc, note=f"unreadable: {e}")
        return
    completed = r.get("completed")
    out_tokens = r.get("total_output_tokens")
    mean_out = round(out_tokens / completed, 1) if completed else None
    note = ""
    if mean_out is not None and r.get("random_output_len") and mean_out < 0.95 * r["random_output_len"]:
        note = f"early stops: mean {mean_out} of {r['random_output_len']} output tokens (no ignore_eos honoured)"
    if not completed:
        note = f"all {r.get('failed', '?')} requests failed (see the per-request errors in raw/{name}.json when saved with --save-detailed)"
    elif r.get("failed"):
        note = (note + "; " if note else "") + f"{r['failed']} of {completed + r['failed']} requests failed"
    tpot = r.get("median_tpot_ms")
    yield row(engine=engine, kind="serve", isl=isl, osl=r.get("random_output_len"), conc=conc,
              ttft_ms=r.get("median_ttft_ms"), tpot_ms=tpot,
              decode_tok_s=round(1000 / tpot, 2) if tpot else None,
              out_tok_s_total=r.get("output_throughput"), req_s=r.get("request_throughput"),
              completed=completed, mean_out_tokens=mean_out, note=note)


def llamacpp_rows(path, kind):
    try:
        with open(path) as f:
            rows = json.load(f)
    except (OSError, ValueError) as e:
        yield row(engine="llamacpp", kind=f"bench_{kind}", note=f"unreadable {os.path.basename(path)}: {e}")
        return
    for r in rows:
        if kind == "pp":
            yield row(engine="llamacpp", kind="bench_pp", isl=r["n_prompt"], conc=1, prefill_tok_s=round(r["avg_ts"], 1),
                      ttft_ms=round(r["n_prompt"] / r["avg_ts"] * 1000, 1), note=f"stddev_ts={r.get('stddev_ts')}")
        else:
            yield row(engine="llamacpp", kind="bench_tg", isl=r.get("n_depth"), osl=r["n_gen"], conc=1,
                      decode_tok_s=round(r["avg_ts"], 2), tpot_ms=round(1000 / r["avg_ts"], 3),
                      note=f"stddev_ts={r.get('stddev_ts')}")


def main(out):
    print("\t".join(COLUMNS))
    raw = os.path.join(out, "raw")
    for kind in ("poot_curve", "poot_curve_roofline", "poot_curve_profile"):
        path = os.path.join(raw, kind + ".json")
        if os.path.exists(path):
            print(*poot_rows(path, kind), sep="\n")
        else:
            print(row(engine="poot", kind=kind, note="no result file (phase did not run)"))
    for path in sorted(glob.glob(os.path.join(raw, "*_isl*_c*.json"))):
        print(*client_rows(path), sep="\n")
    for kind in ("pp", "tg"):
        path = os.path.join(raw, f"llamacpp_bench_{kind}.json")
        if os.path.exists(path):
            print(*llamacpp_rows(path, kind), sep="\n")


if __name__ == "__main__":
    main(sys.argv[1])
