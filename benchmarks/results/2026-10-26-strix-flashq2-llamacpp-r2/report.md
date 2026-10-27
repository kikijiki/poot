# poot benchmark report

- GPU: AMD Radeon 8060S Graphics (driver None, None)
- captured: 2026-10-26T15:00:33.605241+00:00

## qwen3.8-flash-next-q2 - decode-curve-large

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp (vulkan) | q2_k_xl | ok | 341.7 | 30.5 | 32.8 | 49.03 | 27.42 | 120.9 | 4.981 | the same quant files as poot; whether the image's llama.cpp tag knows qwen4exp is the cell's result; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; q2_k_xl is a llama.cpp k-quant, not GPTQ/AWQ - bit-width parity only |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 8192 |
|---|---|---|
| llamacpp (vulkan) | 32.8 | 31.6 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 8192 |
|---|---|---|
| llamacpp (vulkan) | 30.5 | 31.6 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, or amdgpu hwmon power on AMD, integrated over wall-clock time); omitted when the device exposes none. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._