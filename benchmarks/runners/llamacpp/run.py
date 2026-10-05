#!/usr/bin/env python3
"""llama.cpp baseline runner (suite contract, spec 112). See NOTES.md for the build + quant caveats.

Uses `llama-bench` (LLAMA_BENCH_BIN, default `llama-bench`), which is non-interactive with structured JSON
output, not llama-cli (which enters interactive conversation mode for instruct models and hangs).
llama-bench measures pure model-eval throughput on a synthetic prompt: a `pp` row (prefill) and a `tg` row
(decode). Decode tok/s (tg) is prompt-independent; TTFT is derived from the pp rate as a proxy (caveat). GGUF
only: --gguf is a dir (the single .gguf is found) or a .gguf path. Prints one JSON line; VRAM is measured
externally by the harness observer.
"""
import argparse
import json
import os
import re
import subprocess
import sys

BENCH = os.environ.get("LLAMA_BENCH_BIN", "llama-bench")
# The HF->GGUF converter ships in the image next to the binaries; quantize sits in the same build/bin.
CONVERT = os.environ.get("LLAMACPP_CONVERT_PY",
                         os.path.join(os.path.dirname(os.path.dirname(BENCH)) if BENCH else "",
                                      "..", "convert_hf_to_gguf.py"))
QUANTIZE = os.environ.get("LLAMA_QUANTIZE_BIN",
                          os.path.join(os.path.dirname(BENCH), "llama-quantize") if BENCH else "llama-quantize")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# Map the manifest precision to a converter outtype and an optional llama-quantize k-quant type. The
# harness hands every framework the same safetensors checkpoint dir; llama.cpp is GGUF-only, so convert on
# demand and cache the result in the dir. f16/bf16 convert directly; a k-quant (qN_k_m) converts to f16 then
# quantizes; gpt-oss mxfp4 has native GGUF support, so the converter picks (auto).
def _gguf_plan(precision):
    p = (precision or "f16").lower()
    if p in ("f16", "fp16"):
        return "f16", None
    if p == "bf16":
        return "bf16", None
    if p == "mxfp4":
        return "auto", None          # gpt-oss: converter emits native mxfp4 tensors
    if p.startswith("q") or "k_m" in p or "k_s" in p:
        return "f16", p.upper()      # e.g. q4_k_m -> convert f16, then quantize Q4_K_M
    return "f16", None


def _convert_hf_to_gguf(hf_dir, precision):
    """Produce (and cache) a GGUF for this checkpoint dir, returning its path. Raises on failure."""
    outtype, kquant = _gguf_plan(precision)
    f16_path = os.path.join(hf_dir, f"converted-{outtype}.gguf")
    if not os.path.exists(f16_path):
        convert = os.path.abspath(CONVERT)
        if not os.path.exists(convert):
            raise FileNotFoundError(f"HF->GGUF converter not found at {convert} (set LLAMACPP_CONVERT_PY)")
        cmd = [sys.executable, convert, hf_dir, "--outfile", f16_path, "--outtype", outtype]
        log("llamacpp convert:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError(f"convert_hf_to_gguf rc={r.returncode}: " +
                               " ".join((r.stderr or "").strip().splitlines()[-6:])[:600])
    if not kquant:
        return f16_path
    q_path = os.path.join(hf_dir, f"converted-{kquant.lower()}.gguf")
    if not os.path.exists(q_path):
        cmd = [QUANTIZE, f16_path, q_path, kquant]
        log("llamacpp quantize:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError(f"llama-quantize rc={r.returncode}: " +
                               " ".join((r.stderr or "").strip().splitlines()[-6:])[:600])
    return q_path


def resolve_gguf(gguf, precision="f16"):
    """Resolve --gguf to a single .gguf file. If the dir holds no GGUF (the usual case: it's the HF
    safetensors checkpoint the harness hands every framework), convert on demand and cache in the dir."""
    if not os.path.isdir(gguf):
        return gguf
    ggufs = [f for f in os.listdir(gguf) if f.endswith(".gguf")]
    if len(ggufs) == 1:
        return os.path.join(gguf, ggufs[0])
    if len(ggufs) == 0:
        return _convert_hf_to_gguf(gguf, precision)
    # More than one GGUF and no way to pick: prefer one matching the requested precision, else error.
    _, kquant = _gguf_plan(precision)
    want = (kquant.lower() if kquant else _gguf_plan(precision)[0])
    match = [f for f in ggufs if want in f.lower()]
    if len(match) == 1:
        return os.path.join(gguf, match[0])
    raise RuntimeError(f"multiple .gguf in {gguf} and none uniquely match precision {precision}: {ggufs}")


def run_decode_curve(args):
    """decode-curve (spec 115) via two llama-bench tests per ISL: `-p ISL` for prefill (ttft) and
    `-d ISL -n OSL` for generation throughput at KV depth ISL (decode tok/s at that context length).
    `-d/--n-depth` pre-fills the cache to ISL, then times pure generation; `-pg`/`-gp` measure decode from
    depth 0 and would read flat across ISLs. llama-bench exposes only aggregate rates (avg over -r reps), so
    ITL/TPOT is the per-ISL aggregate, not a per-token distribution (recorded as a caveat). The prompt is
    synthetic (tokenizer-neutral)."""
    try:
        gguf = resolve_gguf(args.gguf, args.precision)
    except Exception as e:
        print(json.dumps({"framework": "llamacpp", "precision": args.precision, "mode": "decode-curve",
                          "error": f"GGUF resolve/convert failed: {e}"}))
        sys.exit(1)
    isls = [int(x) for x in args.isl_list.split(",") if x.strip()]
    osl = args.osl
    reps = str(max(args.iters, 1))
    caveats = ["llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text)",
               "ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution"]
    if args.precision.startswith("q") or "k_m" in args.precision:
        caveats.append(f"{args.precision} is a llama.cpp k-quant, not GPTQ/AWQ - bit-width parity only")
    curve, version = [], "unknown"

    def bench(extra):
        cmd = [BENCH, "-m", gguf, "-ngl", "99", "-r", reps, "-o", "json"] + extra
        log("llamacpp:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            log(f"llamacpp rc={r.returncode}: {(r.stderr or '')[-300:]}")
            return None
        return json.loads(r.stdout)

    for isl in isls:
        pp_rows = bench(["-p", str(isl), "-n", "0"])              # prefill ISL tokens at depth 0 -> ttft
        tg_rows = bench(["-p", "0", "-n", str(osl), "-d", str(isl)])  # generate OSL at KV-depth ISL -> decode
        if not pp_rows or not tg_rows:
            continue
        pp = next((x for x in pp_rows if x.get("n_prompt", 0) > 0 and x.get("n_gen", 0) == 0), None)
        gp = next((x for x in tg_rows if x.get("n_gen", 0) > 0), None)
        version = (gp or pp or {}).get("build_commit", version)
        pp_ts = pp["avg_ts"] if pp else None
        tg_ts = gp["avg_ts"] if gp else None
        if not tg_ts:
            continue
        ttft = (isl / pp_ts * 1000.0) if pp_ts else None
        tpot = 1000.0 / tg_ts
        e2e = (ttft or 0.0) + tpot * osl
        curve.append({"isl": isl, "prompt_tokens": isl,
                      "iters": [{"ttft_ms": ttft, "e2e_ms": e2e, "itl_ms": [tpot] * max(osl - 1, 0)}]})
    print(json.dumps({"framework": "llamacpp", "framework_version": version, "precision": args.precision,
                      "mode": "decode-curve", "osl": osl, "caveats": caveats, "curve": curve}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, help="dir containing one .gguf, or a .gguf path")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--precision", default="f16")
    p.add_argument("--mode", default="single")  # single | decode-curve (spec 115)
    p.add_argument("--isl-list", default="128,512,2048,8192")
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    if args.mode == "decode-curve":
        run_decode_curve(args)
        return

    try:
        gguf = resolve_gguf(args.gguf, args.precision)
    except Exception as e:
        print(json.dumps({"framework": "llamacpp", "precision": args.precision,
                          "error": f"GGUF resolve/convert failed: {e}"}))
        sys.exit(1)

    # Approximate prompt length in tokens (llama-bench uses a synthetic prompt of -p tokens; we size it to
    # the scenario prompt's word count as a rough proxy so the pp/TTFT number tracks the real prompt).
    words = len(open(args.prompt_file).read().split())
    p_tokens = max(4, int(words * 1.3))
    n = args.gen_tokens
    caveats = ["llama-bench uses a synthetic prompt (throughput, not the literal prompt text)",
               "TTFT derived from pp (prefill) rate - a proxy, not a measured first-token time"]
    if args.precision.startswith("q") or "k_m" in args.precision:
        caveats.append(f"{args.precision} is a llama.cpp k-quant, not GPTQ/AWQ - bit-width parity only")

    cmd = [BENCH, "-m", gguf, "-ngl", "99", "-p", str(p_tokens), "-n", str(n),
           "-r", str(max(args.iters, 1)), "-o", "json"]
    log("llamacpp:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        print(json.dumps({"framework": "llamacpp", "precision": args.precision,
                          "error": f"llama-bench rc={r.returncode}: " +
                          " ".join((r.stderr or "").strip().splitlines()[-4:])[:400], "caveats": caveats}))
        sys.exit(1)

    rows = json.loads(r.stdout)
    pp = next((x for x in rows if x.get("n_prompt", 0) > 0 and x.get("n_gen", 0) == 0), None)
    tg = next((x for x in rows if x.get("n_gen", 0) > 0 and x.get("n_prompt", 0) == 0), None)
    version = (pp or tg or {}).get("build_commit", "unknown")

    pp_ts = pp["avg_ts"] if pp else None      # prefill tok/s
    tg_ts = tg["avg_ts"] if tg else None      # decode tok/s
    tg_sd = tg.get("stddev_ts", 0.0) if tg else 0.0
    ttft_ms = (p_tokens / pp_ts * 1000.0) if pp_ts else None
    tpot_ms = (1000.0 / tg_ts) if tg_ts else None
    e2e_ms = (ttft_ms or 0.0) + (tpot_ms or 0.0) * n

    out = {
        "framework": "llamacpp",
        "framework_version": version,
        "precision": args.precision,
        "prompt_tokens": p_tokens,
        "gen_tokens": n,
        "iterations": max(args.iters, 1),
        "ttft_ms": round(ttft_ms, 3) if ttft_ms else None,
        "tpot_ms": round(tpot_ms, 3) if tpot_ms else None,
        "e2e_ms": round(e2e_ms, 3),
        "decode_tok_s": round(tg_ts, 3) if tg_ts else None,
        "decode_tok_s_stdev": round(tg_sd, 3),
        "caveats": caveats,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
