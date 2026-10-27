# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-24T07:54:29.589359+00:00

- poot build: d4653ef3

## qwen3-4b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 110.3 | 8.3 | 120.4 | 22.01 | 7.84 | 601.9 | 5.938 |  |
| llamacpp (cuda) | f16 | ok | 51.6 | 5.9 | 169.2 | 10.30 | 7.80 | 603.1 | 2.601 | GGUF f16 vs bf16; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 151.3 | 12.7 | 78.9 | 12.94 | 7.98 | 529.9 | 16.203 |  |
| transformers | bf16 | error | - | - | - | - | - | - | - | rc=1: stderr:            ^^^^^^^^^^^^ |   File "/opt/tf-venv/lib/python3.11/site |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 120.4 | 84.6 | 40.6 |
| llamacpp (cuda) | 169.2 | 162.3 | 149.4 |
| poot (ptx) | 78.9 | 68.0 | 51.7 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 8.3 | 11.8 | 24.6 |
| llamacpp (cuda) | 5.9 | 6.2 | 6.7 |
| poot (ptx) | 12.7 | 14.7 | 19.3 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._