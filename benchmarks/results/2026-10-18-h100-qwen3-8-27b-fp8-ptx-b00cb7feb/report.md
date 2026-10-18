# poot benchmark report

- GPU: NVIDIA H100 80GB HBM3 (driver 580.126.09, 81559 MiB)
- captured: 2026-10-18T06:13:37+00:00

- poot build: b00cb7fe

## qwen3.8-27b-fp8 - decode-curve

| framework | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | peak power W | energy Wh | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| poot (ptx) | bf16 | ok | 3885.7 | 122.3 | 8.2 | 39.48 | - | - | - | poot only: no vLLM / llama.cpp baseline; the budget (14400 s) was spent on poot-serve rows that timed out.; Oracle row failed (rc=96, logs/oracle_receipt.err): oracle-lane.tar.zst had a './' entry and nextest requires every path under 'target/'. No oracle fixture from this run; fixed in stage.sh afterwards.; DEFECT RECEIPT: prefill executes as sequential per-token steps in this build (TTFT linear in ISL: ISL 128 = 3.9 s, ISL 8192 = 234 s, ~28.6 ms/token, ~35 tok/s) and decode is 122 ms/token (8.2 tok/s, flat across ISL). The weights-read floor on an H100 is ~8.4 ms/token (~28 GB at 3.35 TB/s), so decode is ~14x off it. These numbers are a defect receipt, not poot's attainable performance.; Context is a repeated phrase cycled to exactly ISL tokens, not the suite's synthetic random tokens.; peak_vram_bytes is an external nvidia-smi sample over the whole poot_curve phase, not the observer's. |


**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| poot (ptx) | 8.2 | 8.2 | 8.2 | 8.2 |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
|---|---|---|---|---|
| poot (ptx) | 122.3 | 122.5 | 122.4 | 122.2 |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Peak power W and energy Wh come from the same sampler (NVML board power draw, integrated over wall-clock time); NVIDIA-only, omitted elsewhere. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
