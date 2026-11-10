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
import math
import statistics
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


# The memory llama.cpp itself allocates is on its log, one line per buffer, from the same process that is measured:
#   load_tensors:      Vulkan0 model buffer size =  1137.00 MiB
#   llama_kv_cache:    Vulkan0 KV buffer size =    28.00 MiB            (llama_memory_recurrent: ... RS buffer size)
#   sched_reserve:    Vulkan0 compute buffer size =    77.25 MiB        (llama_context: ... in older builds)
# llama-bench prints them only with -v. llama-cli's "memory breakdown" table has the same three columns but is for llama-cli's
# own context and batch, whose compute buffer differs from llama-bench's, so it is not the table of the measured process.
_BUFFER_LINE = re.compile(
    r"^(?P<source>load_tensors|llama_kv_cache\w*|llama_memory_recurrent\w*|sched_reserve|llama_context):\s+"
    r"(?P<buffer>\S+)\s+(?P<kind>model|KV|RS|compute) buffer size =\s+(?P<mib>\d+(?:\.\d+)?) MiB\s*$", re.M)
_CLASS_OF_KIND = {"model": "weights_bytes", "KV": "kv_bytes", "RS": "kv_bytes", "compute": "workspace_bytes"}


def is_host_buffer(name):
    """A llama.cpp buffer name on the host (`CPU`, `CPU_Mapped`, `Vulkan_Host`, `CUDA_Host`), not on a device."""
    return name.startswith("CPU") or name.endswith("_Host")


def parse_memory_classes(log):
    """The device memory llama.cpp allocated, by class, from a `llama-bench -v` log: the model's weights, the KV cache and
    recurrent state, and the compute buffers, summed over the device buffers (host buffers are not the GPU's). A buffer line
    appears twice only where llama.cpp itself repeats it, so the first sight of (kind, buffer) per source is kept. None when
    any of the three classes has no device line: a build whose log differs must not produce a partial breakdown."""
    seen, classes = set(), {}
    for found in _BUFFER_LINE.finditer(log):
        key = (found["source"], found["kind"], found["buffer"])
        if key in seen or is_host_buffer(found["buffer"]):
            continue
        seen.add(key)
        name = _CLASS_OF_KIND[found["kind"]]
        classes[name] = classes.get(name, 0) + round(float(found["mib"]) * 2**20)
    if set(classes) != {"weights_bytes", "kv_bytes", "workspace_bytes"}:
        return None
    return classes


def probe_memory_classes(gguf, test_args):
    """`(classes, caveat)`: llama-bench's own device memory for the test `test_args` (the largest context of the run, whose
    KV cache is the largest), read from a separate verbose run. It runs before the measured invocations, so the observer's
    last load is still the measured one, and verbose logging never touches a timed run."""
    cmd = [BENCH, "-m", gguf, "-ngl", "99", "-r", "1", "-o", "json", "-v"] + test_args
    log("llamacpp memory probe:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        return None, f"llama.cpp memory breakdown unavailable: the probe run exited rc={r.returncode}"
    classes = parse_memory_classes(r.stderr)
    if classes is None:
        return None, "llama.cpp memory breakdown unavailable: this build's log has no model, KV and compute buffer lines"
    return classes, None


def cpu_fallback(rows):
    """The reason llama-bench ran on the CPU although `-ngl 99` asked for the GPU, or None. With no usable GPU
    the tool still exits 0 and prints CPU numbers; a GPU cell must not report them. Each row names the backends
    the build loaded and used (`backends`, e.g. "CUDA"); one that lists only CPU/BLAS ran on the host."""
    for row in rows:
        backends = {b.strip().upper() for b in str(row.get("backends", "")).split(",") if b.strip()}
        if not backends - {"CPU", "BLAS"}:
            return f"llama-bench ran on backends {row.get('backends')!r}, not a GPU (-ngl 99 was requested)"
    return None


def reported_backend(rows):
    """The accelerator backend llama-bench loaded and used (lower case: "cuda", "vulkan", "rocm"), from the
    `backends` field of its rows. A run that used only the host has no such backend: cpu_fallback refuses it first."""
    for row in rows:
        for backend in str(row.get("backends", "")).split(","):
            if backend.strip().upper() not in ("", "CPU", "BLAS"):
                return backend.strip().lower()
    raise ValueError("llama-bench reported no accelerator backend")


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


_SHARD = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")


def _first_shard(ggufs):
    """The first shard of `ggufs` when they are exactly one complete split set (llama.cpp's `-0000N-of-0000M.gguf`
    files), else None. llama.cpp loads the whole set from its first shard."""
    shards = [_SHARD.match(f) for f in ggufs]
    if not ggufs or not all(shards):
        return None
    stems = {m["stem"] for m in shards}
    counts = {int(m["count"]) for m in shards}
    if len(stems) != 1 or len(counts) != 1 or sorted(int(m["index"]) for m in shards) != list(range(1, counts.pop() + 1)):
        return None
    return next(m.string for m in shards if int(m["index"]) == 1)


def resolve_gguf(gguf, precision="f16"):
    """Resolve --gguf to the .gguf file llama.cpp loads: a file as given (the first shard of a split set names the
    set), a directory holding one GGUF or one complete shard set. If the dir holds no GGUF (the usual case: it's the HF
    safetensors checkpoint the harness hands every framework), convert on demand and cache in the dir."""
    if not os.path.isdir(gguf):
        return gguf
    ggufs = [f for f in os.listdir(gguf) if f.endswith(".gguf")]
    if len(ggufs) == 1:
        return os.path.join(gguf, ggufs[0])
    first = _first_shard(ggufs)
    if first is not None:
        return os.path.join(gguf, first)
    if len(ggufs) == 0:
        return _convert_hf_to_gguf(gguf, precision)
    # More than one GGUF and no way to pick: prefer one matching the requested precision, else error.
    _, kquant = _gguf_plan(precision)
    want = (kquant.lower() if kquant else _gguf_plan(precision)[0])
    match = [f for f in ggufs if want in f.lower()]
    if len(match) == 1:
        return os.path.join(gguf, match[0])
    raise RuntimeError(f"multiple .gguf in {gguf} and none uniquely match precision {precision}: {ggufs}")


def timed_samples(row, warmup):
    """The per-repetition rates (tokens/s) of one llama-bench row after its `warmup` first repetitions, which are
    discarded exactly as every other framework discards its warm-up iterations. llama-bench's own `avg_ts` is the
    mean of all repetitions: one slow repetition moves it, and the first one is always slower (lazy device
    initialisation), where the contract takes the median of the iterations the runner reports."""
    samples = row.get("samples_ts") or []
    if len(samples) <= warmup:
        raise ValueError(f"llama-bench reported {len(samples)} repetitions, the run needs more than its {warmup} warm-up")
    return samples[warmup:]


# A short prefill pass on the ROCm build stalls in episodes of several seconds after the model loads: the GPU kernels that
# read the host-mapped tensors (the embedding gather) run up to 500x slower while the kernel compacts memory
# (kcompactd, page migration), and consecutive passes stay slow until it settles. Three repetitions can all fall inside
# one episode, so a pass shorter than SHORT_PASS_S is repeated for TIMED_WINDOW_S of timed passes, and the median of
# those is the iteration the contract sees. It is a property of the engine's first seconds, the same for every backend.
SHORT_PASS_S = 2.0
TIMED_WINDOW_S = 20.0
# The pilot is a process's first pass, slower than the passes after it; sizing from it alone leaves the window short.
PILOT_MARGIN = 1.15
MAX_REPETITIONS = 100


def prefill_repetitions(pass_s, warmup, iters):
    """llama-bench repetitions (warm-up included) for a prefill pass that took `pass_s` seconds in a pilot."""
    if pass_s >= SHORT_PASS_S:
        return warmup + iters
    return warmup + max(iters, min(MAX_REPETITIONS, math.ceil(TIMED_WINDOW_S * PILOT_MARGIN / pass_s)))


def curve_iterations(isl, osl, pp_row, tg_row, warmup):
    """The timed iterations of one context length: one per timed prefill repetition after the warm-up, each with the
    median decode rate of the context length's timed decode repetitions (a short prefill pass is repeated many times
    and a decode pass few, so the two do not pair one to one)."""
    pp, tg = timed_samples(pp_row, warmup), timed_samples(tg_row, warmup)
    tpot = 1000.0 / statistics.median(tg)
    out = []
    for pp_ts in pp:
        ttft = isl / pp_ts * 1000.0
        out.append({"ttft_ms": ttft, "e2e_ms": ttft + tpot * osl, "itl_ms": [tpot] * max(osl - 1, 0)})
    return out


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
    reps = str(max(args.iters, 1) + args.warmup)
    caveats = ["llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text)",
               "ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution"]
    if args.precision.startswith("q") or "k_m" in args.precision:
        caveats.append(f"{args.precision} is a llama.cpp k-quant, not GPTQ/AWQ - bit-width parity only")
    curve, version, backend = [], "unknown", None
    memory_classes, memory_caveat = probe_memory_classes(gguf, ["-p", "0", "-n", str(osl), "-d", str(max(isls))])
    if memory_caveat:
        caveats.append(memory_caveat)

    def bench(extra, repetitions=None):
        cmd = [BENCH, "-m", gguf, "-ngl", "99", "-r", str(repetitions or reps), "-o", "json"] + extra
        log("llamacpp:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            log(f"llamacpp rc={r.returncode}: {(r.stderr or '')[-300:]}")
            return None
        rows = json.loads(r.stdout)
        fallback = cpu_fallback(rows)
        if fallback:
            print(json.dumps({"framework": "llamacpp", "precision": args.precision, "mode": "decode-curve",
                              "error": fallback}))
            sys.exit(1)
        return rows

    for isl in isls:
        pilot = bench(["-p", str(isl), "-n", "0"], repetitions=1)    # how long one prefill pass takes
        pilot_row = next((x for x in pilot or [] if x.get("n_prompt", 0) > 0 and x.get("n_gen", 0) == 0), None)
        pass_s = isl / pilot_row["avg_ts"] if pilot_row and pilot_row.get("avg_ts") else SHORT_PASS_S
        pp_reps = prefill_repetitions(pass_s, args.warmup, max(args.iters, 1))
        pp_rows = bench(["-p", str(isl), "-n", "0"], repetitions=pp_reps)  # prefill ISL tokens at depth 0 -> ttft
        tg_rows = bench(["-p", "0", "-n", str(osl), "-d", str(isl)])  # generate OSL at KV-depth ISL -> decode
        if not pp_rows or not tg_rows:
            continue
        pp = next((x for x in pp_rows if x.get("n_prompt", 0) > 0 and x.get("n_gen", 0) == 0), None)
        gp = next((x for x in tg_rows if x.get("n_gen", 0) > 0), None)
        version = (gp or pp or {}).get("build_commit", version)
        if not pp or not gp:
            continue
        backend = reported_backend(tg_rows)
        curve.append({"isl": isl, "prompt_tokens": isl, "iters": curve_iterations(isl, osl, pp, gp, args.warmup)})
    print(json.dumps({"framework": "llamacpp", **({"backend": backend} if backend else {}), "framework_version": version,
                      "precision": args.precision, "mode": "decode-curve", "osl": osl, "caveats": caveats, "curve": curve,
                      **({"memory_classes": memory_classes} if memory_classes else {})}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, help="dir containing one .gguf, or a .gguf path")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1, help="leading repetitions of each llama-bench test to discard")
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

    memory_classes, memory_caveat = probe_memory_classes(gguf, ["-p", str(p_tokens), "-n", str(n)])
    if memory_caveat:
        caveats.append(memory_caveat)

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
    fallback = cpu_fallback(rows)
    if fallback:
        print(json.dumps({"framework": "llamacpp", "precision": args.precision, "error": fallback, "caveats": caveats}))
        sys.exit(1)
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
        "backend": reported_backend(rows),
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
        **({"memory_classes": memory_classes} if memory_classes else {}),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
