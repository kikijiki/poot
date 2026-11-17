#!/usr/bin/env python3
"""The serve-throughput runner (POOT-1334): start an engine's server, drive it with the load client, print one JSON line.

One invocation is one cell: one engine at one concurrency of the `serve-throughput` scenario. It starts the engine's
own server (poot-serve, `vllm serve`'s OpenAI entry point, llama-server) on a free local port with as many slots as
the concurrency, waits until it is healthy, reads what the server says about itself, runs `client.run_level` and stops
the server. The result is the load client's raw requests plus the identity the server states; the harness validates
and aggregates it (harness/result_contract.py). See NOTES.md.
"""
import argparse
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import client  # noqa: E402

HOST = "127.0.0.1"
# Room for the prompt, the output and the server's own margin (a BOS, a chat template, the position of the last token).
MARGIN_TOKENS = 16
HEALTH_TIMEOUT_S = 1800
REQUEST_TIMEOUT_S = 600


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def mark(event):
    """Tell the observer the model is loaded (`load_end`) or the warmup is done (`warmup_end`): harness/memory_phases.py."""
    print(f"POOT_BENCH_MARK {event} {time.time():.6f}", file=sys.stderr, flush=True)


def free_port():
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def max_seq(args):
    return args.prompt_tokens + args.output_tokens + MARGIN_TOKENS


VLLM_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "f16": "float16", "f32": "float32"}
VLLM_QUANT = {"gptq-int4": "gptq", "awq-int4": "awq", "mxfp4": "mxfp4"}


def server_command(args, port, gguf=None):
    """The argv of the engine's server for this cell: `concurrency` slots, positions for one request each, and no
    prefix reuse where the server has the switch (the load client already sends distinct prompts)."""
    if args.framework == "poot":
        return [args.server_bin, args.model_dir, f"{HOST}:{port}", "--backend", args.backend,
                "--rows", str(args.concurrency), "--max-seq", str(max_seq(args))]
    if args.framework == "vllm":
        cmd = [args.server_bin, "-m", "vllm.entrypoints.openai.api_server", "--model", args.model_dir,
               "--host", HOST, "--port", str(port), "--dtype", VLLM_DTYPES.get(args.precision, "auto"),
               "--max-num-seqs", str(args.concurrency), "--max-model-len", str(max_seq(args)),
               "--gpu-memory-utilization", str(args.gpu_mem_util), "--no-enable-prefix-caching"]
        if args.precision in VLLM_QUANT:
            cmd += ["--quantization", VLLM_QUANT[args.precision]]
        return cmd
    if args.framework == "llamacpp":
        # llama-server splits one context across its slots: `-c` holds every slot's positions.
        return [args.server_bin, "-m", gguf, "--host", HOST, "--port", str(port), "-ngl", "99",
                "--parallel", str(args.concurrency), "-c", str(args.concurrency * max_seq(args))]
    raise ValueError(f"no server for {args.framework!r}")


def request_extras(framework):
    """Fields of a request body only one engine's server reads: llama-server's own prompt cache switch."""
    return {"cache_prompt": False} if framework == "llamacpp" else {}


def get_json(port, path, timeout=10):
    with urllib.request.urlopen(f"http://{HOST}:{port}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def wait_healthy(port, proc, timeout_s):
    """Block until `/health` answers 200, or fail naming why: the server exited, or it never became healthy."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise client.ClientError(f"the server exited with code {proc.returncode} before it was healthy")
        try:
            with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1.0)
    raise client.ClientError(f"the server was not healthy within {timeout_s} s")


_LLAMA_BACKENDS = (("ROCm devices", "rocm"), ("CUDA devices", "cuda"), ("ggml_vulkan: Found", "vulkan"),
                   ("ggml_metal_init", "metal"))


def llama_backend(log_text):
    """The accelerator llama-server loaded, from its own start-up log: the first backend line it printed."""
    found = [(log_text.find(needle), name) for needle, name in _LLAMA_BACKENDS if needle in log_text]
    if not found:
        raise client.ClientError("llama-server's log names no accelerator backend (it may have run on the CPU)")
    return min(found)[1]


def llama_version(props):
    """The build llama-server reports on `/props`, or a refusal: an external tool names the version it measured."""
    info = props.get("build_info") if isinstance(props, dict) else None
    if not (isinstance(info, str) and info.strip()):
        raise client.ClientError(f"llama-server's /props names no build_info: {props!r}")
    return info.strip()


def poot_identity(port):
    """What poot-serve says it runs on (`GET /v1/identity`): the build and device identity of the result, as stated."""
    identity = get_json(port, "/v1/identity")
    missing = [k for k in ("build_sha", "build_profile", "build_features", "rustc_version", "family", "precision",
                           "adapter", "driver_version", "checkpoint_digest", "compute_dtype") if k not in identity]
    if missing:
        raise client.ClientError(f"/v1/identity lacks {', '.join(missing)}: {identity!r}")
    return identity


def llamacpp_gguf(args):
    """The GGUF llama.cpp serves: the same resolution (and conversion) as the single-stream llama.cpp runner."""
    spec = importlib.util.spec_from_file_location("llamacpp_run", os.path.join(HERE, "..", "llamacpp", "run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_gguf(args.model_dir, args.precision)


def describe(args, port, stderr_path):
    """What the server states about itself, as the fields of the result."""
    if args.framework == "poot":
        # The server was started on the backend the cell asked for and fails startup rather than open another.
        return {**poot_identity(port), "backend": args.backend}, []
    if args.framework == "vllm":
        version = get_json(port, "/version").get("version")
        if not version:
            raise client.ClientError("vLLM's /version names no version")
        return {"framework_version": version, "backend": "cuda"}, [
            f"vLLM KV pool at gpu_memory_utilization={args.gpu_mem_util} inflates external VRAM; prefix caching disabled"]
    with open(stderr_path, errors="replace") as f:
        backend = llama_backend(f.read())
    return {"framework_version": llama_version(get_json(port, "/props")), "backend": backend}, [
        "llama-server: one context split over the slots; prompt cache off"]


def stop(proc):
    """Stop the server and everything it started (vLLM spawns an engine process): a SIGTERM to its process group."""
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", required=True, choices=("poot", "vllm", "llamacpp"))
    p.add_argument("--server-bin", required=True, help="poot-serve, a python with vllm, or llama-server")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--precision", required=True)
    p.add_argument("--backend", default="ptx", help="poot: the backend its server opens")
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--prompt-tokens", type=int, required=True)
    p.add_argument("--output-tokens", type=int, required=True)
    p.add_argument("--warmup-requests", type=int, required=True)
    p.add_argument("--requests", type=int, required=True)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = p.parse_args()

    port = free_port()
    gguf = llamacpp_gguf(args) if args.framework == "llamacpp" else None
    cmd = server_command(args, port, gguf)
    log("serve runner:", " ".join(cmd))
    with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False) as server_log:
        proc = subprocess.Popen(cmd, stdout=server_log, stderr=server_log, start_new_session=True)
    try:
        wait_healthy(port, proc, HEALTH_TIMEOUT_S)
        mark("load_end")
        stated, caveats = describe(args, port, server_log.name)
        slots = args.concurrency
        if args.framework == "poot":
            slots = get_json(port, "/metrics")["slot_capacity"]
        model = get_json(port, "/v1/models")["data"][0]["id"]
        level = client.run_level(
            HOST, port, model, concurrency=args.concurrency, prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens, warmup_requests=args.warmup_requests, requests=args.requests,
            extra=request_extras(args.framework), timeout_s=REQUEST_TIMEOUT_S,
            on_warmed=lambda: mark("warmup_end"))
    except client.ClientError as error:
        print(json.dumps({"framework": args.framework, "mode": "serve", "error": str(error)}))
        sys.exit(1)
    finally:
        stop(proc)
        os.unlink(server_log.name)
    out = {"framework": args.framework, "precision": stated.pop("precision", args.precision), "mode": "serve",
           "concurrency": args.concurrency, "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
           "server_slots": slots, "caveats": caveats, **stated, **level}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
