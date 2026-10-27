# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-24T07:51:13.393233+00:00

- poot build: d4653ef3

## qwen3-0.6b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 33.8 | 3.1 | 322.9 | 7.98 | 1.40 | 585.0 | 1.736 |  |
| llamacpp (cuda) | f16 | ok | 15.9 | 1.4 | 697.5 | 3.66 | 1.71 | 506.6 | 0.590 | GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 54.6 | 6.3 | 159.8 | 5.09 | 1.86 | 394.5 | 5.084 |  |
| transformers | bf16 | error | - | - | - | - | - | - | - | rc=1: stderr:            ^^^^^^^^^^^^ |   File "/opt/tf-venv/lib/python3.11/site |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 322.9 | 240.7 | 101.9 |
| llamacpp (cuda) | 697.5 | 626.9 | 487.8 |
| poot (ptx) | 159.8 | 124.6 | 124.4 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 3.1 | 4.2 | 9.8 |
| llamacpp (cuda) | 1.4 | 1.6 | 2.0 |
| poot (ptx) | 6.3 | 8.0 | 8.0 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._