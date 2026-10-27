# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-24T07:58:13.512754+00:00

- poot build: d4653ef3

## qwen3-4b - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| candle | bf16 | ok | 110.4 | 8.3 | 120.6 | 21.79 | 7.81 | 603.5 | 5.898 |  |
| llamacpp (cuda) | f16 | ok | 52.1 | 5.9 | 169.1 | 10.30 | 7.80 | 602.1 | 2.502 | GGUF f16 vs bf16; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot (ptx) | bf16 | ok | 152.1 | 12.7 | 78.7 | 13.49 | 7.96 | 540.6 | 16.547 |  |
| transformers | bf16 | error | - | - | - | - | - | - | - | rc=1: stderr:            ^^^^^^^^^^^^ |   File "/opt/tf-venv/lib/python3.11/site |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 120.6 | 84.6 | 40.5 |
| llamacpp (cuda) | 169.1 | 162.3 | 149.4 |
| poot (ptx) | 78.7 | 67.8 | 51.7 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| candle | 8.3 | 11.8 | 24.7 |
| llamacpp (cuda) | 5.9 | 6.2 | 6.7 |
| poot (ptx) | 12.7 | 14.7 | 19.3 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._