poot-bench devshell (image build/push tooling)
podman: podman version 5.8.2
skopeo: skopeo version 1.22.2
recipes: just --list (build: just image-build | CI build+push: just image-ci)
image: ghcr.io/${GHCR_OWNER:-kikijiki}/poot-bench

# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.126.20, 24576 MiB)
- poot commit: bench/decode-curve@4b8e0b3
- captured: 2026-06-13T14:39:55.887966+00:00

## qwen2.5-0.5b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                                                                                     |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| candle       | bf16      | ok     | 10.0    | 7.3     | 137.0        | 9.54          | 1.36         |                                                                                                                                                                                                                                                                                                                           |
| llamacpp     | f16       | ok     | 5.6     | 2.2     | 463.8        | 2.06          | 1.20         | llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text); precision f16 != matched target bf16; GGUF f16 vs bf16 - closest full-precision GGUF; bf16 GGUF also possible; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution |
| poot         | bf16      | ok     | 3170.5  | 25.9    | 38.7         | 2.20          | 2.16         |                                                                                                                                                                                                                                                                                                                           |
| transformers | bf16      | ok     | 17.3    | 16.1    | 62.0         | 4.81          | 1.74         |                                                                                                                                                                                                                                                                                                                           |
| vllm         | bf16      | ok     | 7.6     | 2.1     | 484.2        | 7.97          | 3.33         | vLLM KV pool at gpu_memory_utilization=0.3 inflates external VRAM; prefix caching disabled; per-token ITL from AsyncLLM streaming                                                                                                                                                                                         |

**Decode degradation - steady decode tok/s vs context length (ISL), higher is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 137.0   | 139.8   | 136.3    | 66.2     |
| llamacpp     | 463.8   | 460.7   | 458.3    | 427.8    |
| poot         | 38.7    | 35.7    | 24.6     | 11.8     |
| transformers | 62.0    | 59.6    | 58.3     | 60.0     |
| vllm         | 484.2   | 484.6   | 482.9    | 477.3    |

**TPOT p50 ms (per-token decode cost, prefill excluded) vs ISL, lower is better:**

| framework    | ISL 128 | ISL 512 | ISL 2048 | ISL 8192 |
| ------------ | ------- | ------- | -------- | -------- |
| candle       | 7.3     | 7.2     | 7.3      | 15.1     |
| llamacpp     | 2.2     | 2.2     | 2.2      | 2.3      |
| poot         | 25.9    | 28.0    | 40.7     | 84.7     |
| transformers | 16.1    | 16.8    | 17.2     | 16.7     |
| vllm         | 2.1     | 2.1     | 2.1      | 2.1      |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
