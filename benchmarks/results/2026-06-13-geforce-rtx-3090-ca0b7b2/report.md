poot-bench devshell (image build/push tooling)
podman: podman version 5.8.2
skopeo: skopeo version 1.22.2
recipes: just --list (build: just image-build | CI build+push: just image-ci)
image: ghcr.io/${GHCR_OWNER:-kikijiki}/poot-bench

# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: bench/decode-curve@ca0b7b2
- captured: 2026-06-13T17:07:58.854022+00:00

## qwen3-0.6b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                                                            |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| candle       | bf16      | ok     | 8.7     | 6.7     | 149.2        | 8.57          | 1.77         |                                                                                                                                                                                                                                                                                                  |
| llamacpp     | f16       | ok     | 7.0     | 2.6     | 387.5        | 3.06          | 1.68         | ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; GGUF f16 vs bf16 - closest full-precision GGUF; precision f16 != matched target bf16; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text) |
| poot         | bf16      | ok     | 4224.6  | 37.8    | 26.4         | 2.69          | 3.24         |                                                                                                                                                                                                                                                                                                  |
| transformers | bf16      | ok     | 25.7    | 24.7    | 40.6         | 5.83          | 2.31         |                                                                                                                                                                                                                                                                                                  |
| vllm         | bf16      | ok     | 10.2    | 2.5     | 403.6        | 8.12          | 3.63         | prefix caching disabled; per-token ITL from AsyncLLM streaming; vLLM KV pool at gpu_memory_utilization=0.3 inflates external VRAM                                                                                                                                                                |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 149.2   | 156.3   | 117.2    | 43.9     |
| llamacpp     | 387.5   | 377.2   | 347.0    | 269.8    |
| poot         | 26.4    | 19.5    | 10.5     | -        |
| transformers | 40.6    | 38.1    | 37.1     | 38.2     |
| vllm         | 403.6   | 399.9   | 359.2    | 265.6    |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 6.7     | 6.4     | 8.5      | 22.8     |
| llamacpp     | 2.6     | 2.7     | 2.9      | 3.7      |
| poot         | 37.8    | 51.2    | 95.0     | -        |
| transformers | 24.7    | 26.3    | 27.0     | 26.2     |
| vllm         | 2.5     | 2.5     | 2.8      | 3.8      |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
