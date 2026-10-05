#!/usr/bin/env python3
"""vLLM offline baseline runner (suite contract, spec 112). See NOTES.md for pins + the VRAM caveat.

vLLM V1 offline `RequestOutput.metrics` is None, so TTFT is estimated from a prefill-only (max_tokens=1)
request and subtracted from the full run. Generation forces exactly --gen-tokens (min_tokens == max_tokens,
ignore_eos). VRAM: vLLM pre-allocates a KV pool sized by gpu_memory_utilization, which bounds its entire
VRAM budget (weights + KV cache). --gpu-mem-util defaults to 0.85; it must exceed the model's weight
footprint (xl-tier ~14B+ bf16 included) or vLLM fails "No available memory for the cache blocks". --max-model-len is pinned low by default to cap the KV pool. The harness external sampler
still sees the pool, so the comparable number is the weights footprint plus this capped working set (caveat
recorded in the output). Prints one JSON line (last stdout line).
"""
import argparse
import json
import os
import statistics
import sys
import time

# vLLM 0.22.1's V1 EngineCore profiling run picks FlashInfer for top-k/top-p sampling when importable, and
# FlashInfer JIT-compiles its CUDA kernel with nvcc on first use. The bench image's runtime base has no
# nvcc (docker/Dockerfile), so the JIT dies with "nvcc: not found" during
# EngineCore._initialize_kv_caches -> profile_run, which the multiprocess engine reports only as
# "Engine core initialization failed. See root cause above." Force the PyTorch sampler; this runner
# samples greedily (temperature=0.0). setdefault lets an operator override it for a comparison.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import vllm
from vllm import LLM, SamplingParams


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def run_decode_curve(args):
    """decode-curve (spec 115): per-token ITL via AsyncLLMEngine streaming (offline LLM.generate has no
    per-token timing). Prefix caching disabled; synthetic fixed-N token prompts."""
    import asyncio
    import random
    import uuid
    import torch
    import vllm
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    try:
        from vllm.inputs import TokensPrompt
    except Exception:
        TokensPrompt = None

    isls = [int(x) for x in args.isl_list.split(",") if x.strip()]
    osl = args.osl
    dtype = {"bf16": "bfloat16", "fp16": "float16", "f16": "float16",
             "f32": "float32"}.get(args.precision, "auto")
    quant = {"gptq-int4": "gptq", "awq-int4": "awq", "mxfp4": "mxfp4"}.get(args.precision)
    caveats = ["prefix caching disabled; per-token ITL from AsyncLLM streaming",
               f"vLLM KV pool at gpu_memory_utilization={args.gpu_mem_util} inflates external VRAM"]
    cfg = json.load(open(f"{args.model_dir}/config.json"))
    vocab = int(cfg.get("vocab_size", 32000))
    rng = random.Random(0)

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model_dir, dtype=dtype, quantization=quant, tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util, max_model_len=max(isls) + osl + 16,
        enable_prefix_caching=False))

    async def gen_once(ids):
        sp = SamplingParams(min_tokens=osl, max_tokens=osl, temperature=0.0, ignore_eos=True)
        prompt = TokensPrompt(prompt_token_ids=ids) if TokensPrompt else {"prompt_token_ids": ids}
        t0 = time.perf_counter()
        ttft, last, prev, itls = None, None, 0, []
        async for out in engine.generate(prompt, sp, request_id=str(uuid.uuid4())):
            now = time.perf_counter()
            n = len(out.outputs[0].token_ids)
            if n > prev:
                if ttft is None:
                    ttft, last = (now - t0) * 1000.0, now
                else:
                    new = n - prev
                    itls.extend([(now - last) * 1000.0 / new] * new)
                    last = now
                prev = n
        return ttft, (time.perf_counter() - t0) * 1000.0, itls

    async def run():
        curve = []
        for isl in isls:
            ids = [rng.randrange(0, vocab) for _ in range(isl)]
            for _ in range(args.warmup):
                await gen_once(ids)
            iters = []
            for _ in range(args.iters):
                ttft, e2e, gaps = await gen_once(ids)
                iters.append({"ttft_ms": ttft, "e2e_ms": e2e, "itl_ms": gaps})
            log(f"vllm decode-curve: isl={isl} done ({args.iters} iters)")
            curve.append({"isl": isl, "prompt_tokens": isl, "iters": iters})
        return curve

    curve = asyncio.run(run())
    print(json.dumps({
        "framework": "vllm", "framework_version": vllm.__version__, "precision": args.precision,
        "mode": "decode-curve", "osl": osl,
        "self_reported_vram_bytes": int(torch.cuda.memory_allocated()),
        "caveats": caveats, "curve": curve}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--image", default="")
    # gpu_memory_utilization bounds vLLM's entire VRAM budget (weights + KV cache). The old 0.30 default
    # fails "No available memory for the cache blocks" once weights exceed it (xl-tier ~14B+ bf16 on a
    # 48GB-class GPU: ~28 GiB > 0.30 * ~44 GiB usable). 0.85 covers small and xl
    # models; override per run. max-model-len stays low by default so the KV pool does not balloon.
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--mode", default="single")  # single | decode-curve (spec 115)
    p.add_argument("--isl-list", default="128,512,2048,8192")
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    if args.mode == "decode-curve":
        run_decode_curve(args)
        return

    caveats = [f"vLLM pre-allocates a KV pool at gpu_memory_utilization={args.gpu_mem_util}; "
               f"external peak VRAM reflects the reservation, not demand"]
    dtype = {"bf16": "bfloat16", "fp16": "float16", "f16": "float16",
             "f32": "float32"}.get(args.precision, "auto")
    quant = None
    if args.precision in ("gptq-int4",):
        quant = "gptq"
    elif args.precision in ("awq-int4",):
        quant = "awq"
    elif args.precision == "mxfp4":
        quant = "mxfp4"
        caveats.append("gpt-oss MXFP4 native kernel needs Hopper/Blackwell; verify on the pod GPU")

    prompt = open(args.prompt_file).read().rstrip()
    import torch
    llm = LLM(model=args.model_dir, dtype=dtype, quantization=quant, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len)
    weights_vram = int(torch.cuda.memory_allocated())

    n = args.gen_tokens
    sp = SamplingParams(min_tokens=n, max_tokens=n, temperature=0.0, ignore_eos=True)
    sp_prefill = SamplingParams(max_tokens=1, temperature=0.0)

    def full():
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)[0]
        e2e = (time.perf_counter() - t0) * 1000.0
        return e2e, len(out.prompt_token_ids), len(out.outputs[0].token_ids)

    def ttft_est():
        t0 = time.perf_counter()
        llm.generate([prompt], sp_prefill, use_tqdm=False)
        return (time.perf_counter() - t0) * 1000.0

    for w in range(args.warmup):
        full()
        log(f"vllm: warmup {w + 1}/{args.warmup}")

    ttfts, tpots, e2es, decs = [], [], [], []
    prompt_tokens = gen = n
    for it in range(args.iters):
        ttft = ttft_est()
        e2e, prompt_tokens, gen = full()
        tpot = (e2e - ttft) / max(gen - 1, 1)
        decode = gen / (e2e / 1000.0)
        ttfts.append(ttft); tpots.append(tpot); e2es.append(e2e); decs.append(decode)
        log(f"vllm: iter {it + 1}/{args.iters}  TTFT~{ttft:.0f}ms  {decode:.1f} tok/s")

    caveats.append("TTFT is estimated (prefill-only request); vLLM V1 offline has no per-request metrics")
    med = statistics.median
    sd = lambda xs: statistics.pstdev(xs) if len(xs) > 1 else 0.0
    out = {
        "framework": "vllm",
        "framework_version": vllm.__version__,
        "precision": args.precision,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen,
        "iterations": args.iters,
        "ttft_ms": round(med(ttfts), 3),
        "tpot_ms": round(med(tpots), 3),
        "e2e_ms": round(med(e2es), 3),
        "decode_tok_s": round(med(decs), 3),
        "ttft_ms_stdev": round(sd(ttfts), 3),
        "decode_tok_s_stdev": round(sd(decs), 3),
        "self_reported_vram_bytes": weights_vram,
        "caveats": caveats,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
