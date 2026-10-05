# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: None
- captured: 2026-06-14T16:52:01.558469+00:00

## qwen2.5-3b - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                                                                                                                                                                                                              |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| candle       | bf16      | error  | -       | -       | -            | -             | -            | rc=1: Error: No such file or directory (os error 2)                                                                                                                                                                                                                |
| llamacpp     | f16       | ok     | -       | -       | -            | 0.44          | 0.02         | precision f16 != matched target bf16; GGUF f16 vs bf16; ITL/TPOT is the llama-bench aggregate tg rate at depth (avg over -r reps), NOT a per-token distribution; llama-bench generates at synthetic KV-depth ISL (tokenizer-neutral; throughput, not literal text) |
| poot         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: Error: load /root/models/qwen2.5-3b-instruct (bf16): io: No such file or d                                                                                                                                                                                   |
| transformers | bf16      | error  | -       | -       | -            | -             | -            | rc=1: resolved_config_file = cached_file(                                                                                                                                                                                                                          | ^^^^^ |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: File "/opt/benchmarks/runners/vllm/run.py", line 187, in <module>                                                                                                                                                                                            |       |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
