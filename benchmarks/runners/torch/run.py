#!/usr/bin/env python3
"""transformers baseline runner (suite contract, spec 112). See NOTES.md for pins + gotchas.

CUDA timing: synchronize before/after, warm up then reset peak stats, TTFT from the first
TextIteratorStreamer chunk. Forces exactly --gen-tokens (min_new_tokens == max_new_tokens, no EOS early
stop) so token counts match across frameworks. Prints one JSON line (last stdout line); memory is measured
externally by the harness observer (torch's own VRAM is a cross-check only).
"""
import argparse
import json
import statistics
import sys
import threading
import time

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

_TF_MAJOR = int(transformers.__version__.split(".")[0])
DTYPE_KW = "dtype" if _TF_MAJOR >= 5 else "torch_dtype"  # v5 renamed torch_dtype -> dtype
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "f16": torch.float16,
         "fp32": torch.float32, "f32": torch.float32}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def run_decode_curve(model, tok, device, args, caveats):
    """decode-curve (spec 115): synthetic ISL prefill, fixed OSL decode, per-token timing. Emits raw
    per-iteration samples for the harness to aggregate. Decodes manually (past_key_values) to time each
    token, syncing around every step."""
    vocab = int(getattr(model.config, "vocab_size", 32000))
    isls = [int(x) for x in args.isl_list.split(",") if x.strip()]
    osl = args.osl
    gen = torch.Generator(device="cpu").manual_seed(0)  # deterministic synthetic ids (tokenizer-neutral)

    def decode_once(input_ids):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize(device)
            ttft = (time.perf_counter() - t0) * 1000.0
            itls = []
            last = time.perf_counter()
            for _ in range(max(osl - 1, 0)):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                itls.append((now - last) * 1000.0)
                last = now
            e2e = (time.perf_counter() - t0) * 1000.0
        return ttft, e2e, itls

    curve = []
    for isl in isls:
        ids = torch.randint(0, vocab, (1, isl), generator=gen).to(device)
        for _ in range(args.warmup):
            decode_once(ids)
        torch.cuda.reset_peak_memory_stats(device)
        iters = []
        for _ in range(args.iters):
            ttft, e2e, itls = decode_once(ids)
            iters.append({"ttft_ms": ttft, "e2e_ms": e2e, "itl_ms": itls})
        log(f"torch decode-curve: isl={isl} done ({args.iters} iters)")
        curve.append({"isl": isl, "prompt_tokens": isl, "iters": iters})

    print(json.dumps({
        "framework": "transformers",
        "framework_version": transformers.__version__,
        "precision": args.precision,
        "mode": "decode-curve",
        "osl": osl,
        "self_reported_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        "caveats": caveats,
        "curve": curve,
    }))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--precision", default="bf16")  # bf16|fp16|f32|gptq-int4|awq-int4 (quant auto-detected)
    p.add_argument("--image", default="")
    p.add_argument("--mode", default="single")  # single | decode-curve (spec 115)
    p.add_argument("--isl-list", default="128,512,2048,8192")
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    assert torch.cuda.is_available(), "no CUDA device"
    device = torch.device("cuda:0")
    caveats = []
    # For pre-quantized GPTQ/AWQ the quant is read from config.json; --precision controls the surround dtype.
    dtype = DTYPE.get(args.precision, "auto")
    if args.precision in ("gptq-int4", "awq-int4"):
        dtype = torch.float16
        caveats.append(f"{args.precision}: quant from checkpoint, surround dtype fp16")

    prompt = open(args.prompt_file).read().rstrip()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, **{DTYPE_KW: dtype}, device_map={"": 0}, attn_implementation="sdpa").eval()

    if args.mode == "decode-curve":
        run_decode_curve(model, tok, device, args, caveats)
        return

    # return_dict=True works across transformers versions (v5 apply_chat_template returns a dict).
    if tok.chat_template:
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
    else:
        enc = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in enc.items()}
    prompt_tokens = int(inputs["input_ids"].shape[1])
    n = args.gen_tokens
    gen_kw = dict(min_new_tokens=n, max_new_tokens=n, do_sample=False,  # exactly n tokens, greedy
                  pad_token_id=tok.pad_token_id, **inputs)

    def one():
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        th = threading.Thread(target=lambda: model.generate(streamer=streamer, **gen_kw))
        with torch.inference_mode():
            th.start()
            t_first = None
            for _ in streamer:
                if t_first is None:
                    t_first = time.perf_counter()
            th.join()
        torch.cuda.synchronize(device)
        t_end = time.perf_counter()
        ttft = (t_first - t0) * 1000.0
        e2e = (t_end - t0) * 1000.0
        tpot = (t_end - t_first) / max(n - 1, 1) * 1000.0
        decode = n / (t_end - t0)
        return ttft, tpot, e2e, decode

    for w in range(args.warmup):
        one()
        log(f"torch: warmup {w + 1}/{args.warmup}")
    torch.cuda.reset_peak_memory_stats(device)

    ttfts, tpots, e2es, decs = [], [], [], []
    for it in range(args.iters):
        ttft, tpot, e2e, decode = one()
        ttfts.append(ttft); tpots.append(tpot); e2es.append(e2e); decs.append(decode)
        log(f"torch: iter {it + 1}/{args.iters}  TTFT {ttft:.0f}ms  {decode:.1f} tok/s")

    med = statistics.median
    sd = lambda xs: statistics.pstdev(xs) if len(xs) > 1 else 0.0
    out = {
        "framework": "transformers",
        "framework_version": transformers.__version__,
        "precision": args.precision,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": n,
        "iterations": args.iters,
        "ttft_ms": round(med(ttfts), 3),
        "tpot_ms": round(med(tpots), 3),
        "e2e_ms": round(med(e2es), 3),
        "decode_tok_s": round(med(decs), 3),
        "ttft_ms_stdev": round(sd(ttfts), 3),
        "decode_tok_s_stdev": round(sd(decs), 3),
        "self_reported_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        "caveats": caveats,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
