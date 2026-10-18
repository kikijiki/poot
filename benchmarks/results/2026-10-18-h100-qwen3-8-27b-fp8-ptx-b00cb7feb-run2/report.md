# poot benchmark report

- GPU: NVIDIA H100 80GB HBM3 (driver 580.126.09, 81559 MiB)
- captured: 2026-10-18T12:10:46+00:00

- poot build: b00cb7fe

## qwen3.8-27b-fp8 - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot (ptx) | bf16 | ok | 3905.5 | 122.2 | 8.2 | 39.48 | - | - | - | HEADLINE (POOT-1265): on the same H100, poot decodes 8.18 tok/s and prefills ~35 tok/s; llama.cpp (Q4_K_XL GGUF, so not the same weights/precision) llama-bench tg is 79.6 tok/s and pp 2500-2800 tok/s at ISL 512-8192: about a 10x decode and ~75x prefill gap. See summary.tsv for the llama.cpp rows.; vLLM baseline FAILED to start: DeepGEMM's JIT needs nvcc and the bench image has none (logs/serve_vllm.err line 173: std::filesystem::exists(nvcc_path)). There is no vLLM row. Follow-up: POOT-1269 (image with nvcc).; poot-serve rows: 1 of 3 requests failed at ISL 128 and at ISL 512, conc 1 only (OSL 32); the conc 4 row hit its 300 s cap (rc 124). No higher-concurrency poot serve rows.; llama.cpp ISL 2048 conc 4 timed out (rc 124, no row); several llama.cpp rows report failed requests (512 c4: 4 of 64, 512 c16: 4 of 64, 512 c32: 1 of 64). Kept as measured.; Roofline ceiling unknown: no pinned H100 ceiling in crates/poot-profile/data/ceilings.tsv, so no fraction-of-ceiling.; DEFECT RECEIPT: prefill executes as sequential per-token steps in this build (TTFT linear in ISL, ~28.6 ms/token, ~35 tok/s) and decode is 122 ms/token flat across ISL; the H100 weights-read floor is ~8.4 ms/token. Not poot's attainable performance.; Context is a repeated phrase cycled to exactly ISL tokens, not the suite's synthetic random tokens.; peak_vram_bytes is an external nvidia-smi sample over the whole poot_curve phase, not the observer's. |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| poot (ptx) | 8.2 | 8.2 | 8.2 | 8.2 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| poot (ptx) | 122.2 | 122.3 | 122.2 | 122.2 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
