"""Pretty-print a decode-curve results.jsonl: per-engine curve (ttft/tpot/itl p50, decode tok/s) + util.

Usage: python show_curve.py <results.jsonl>   (dev helper; not part of the harness API)
"""
import json
import sys


def fmt(x, w=9):
    if x is None:
        return "-".rjust(w)
    if isinstance(x, float):
        return f"{x:.1f}".rjust(w)
    return str(x).rjust(w)


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for r in rows:
        fw = r.get("framework")
        st = r.get("status")
        if st != "ok":
            print(f"\n== {fw}: {st} ({str(r.get('reason'))[:120]})")
            continue
        print(f"\n== {fw}  headline={fmt(r.get('decode_tok_s'),0)} tok/s  osl={r.get('osl')}")
        gm, gp = r.get("gpu_util_mean_pct"), r.get("gpu_util_peak_pct")
        cm, cp = r.get("cpu_util_mean_pct"), r.get("cpu_util_peak_pct")
        vram = r.get("peak_vram_bytes")
        vgb = f"{vram/1e9:.2f}GB" if vram else "-"
        print(f"   gpu_util mean/peak {fmt(gm,0)}/{fmt(gp,0)}%   cpu_util mean/peak {fmt(cm,0)}/{fmt(cp,0)}%   "
              f"vram {vgb}   series_pts {len(r.get('util_series') or [])}")
        print(f"   {'isl':>6} {'ttft_p50':>10} {'tpot_p50':>10} {'itl_p50':>9} {'decode_tok_s':>13}")
        for p in r.get("curve", []):
            t = (p.get("ttft_ms") or {}).get("p50")
            tp = (p.get("tpot_ms") or {}).get("p50")
            it = (p.get("itl_ms") or {}).get("p50")
            print(f"   {fmt(p.get('isl'),6)} {fmt(t,10)} {fmt(tp,10)} {fmt(it,9)} {fmt(p.get('decode_tok_s'),13)}")


if __name__ == "__main__":
    main(sys.argv[1])
