# poot benchmark report

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.91.07, 97887 MiB)
- captured: 2026-10-26T15:16:58.958245+00:00

- poot build: 9b55b888

## qwen3.8-flash-next-q2 - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp (cuda) | q2_k_xl | ok | 126.4 | 10.3 | 97.2 | 48.47 | 46.56 | 349.4 | 3.979 | the same quant files as poot; whether the image's llama.cpp tag knows qwen4exp is the cell's result; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; q2_k_xl is a llama.cpp k-quant, not GPTQ/AWQ - bit-width parity only |
| poot (ptx) | iq4_nl | ok | 867.4 | 19.7 | 50.9 | 94.65 | 32.07 | 376.0 | 9.844 | UD-Q2_K_XL stores IQ4_NL, IQ2_XS, Q5_K, Q8_0 and smaller shares of other types; the row's precision is the dominant one; precision iq4_nl != matched target q2_k_xl |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| llamacpp (cuda) | 97.2 | 94.6 | 88.0 |
| poot (ptx) | 50.9 | 51.3 | 51.9 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 2048 | ISL 8192 |
|---|---|---|---|
| llamacpp (cuda) | 10.3 | 10.6 | 11.4 |
| poot (ptx) | 19.7 | 19.5 | 19.3 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._