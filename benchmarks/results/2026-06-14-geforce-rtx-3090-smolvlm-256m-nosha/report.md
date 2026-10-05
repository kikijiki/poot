# poot benchmark report

- GPU: NVIDIA GeForce RTX 3090 (driver 580.159.03, 24576 MiB)
- poot commit: None
- captured: 2026-06-14T17:10:48.160309+00:00

## smolvlm-256m - decode-curve

| framework    | precision | status | TTFT ms | TPOT ms | decode tok/s | peak VRAM GiB | peak RSS GiB | notes                                                                            |
| ------------ | --------- | ------ | ------- | ------- | ------------ | ------------- | ------------ | -------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| llamacpp     | f16       | error  | -       | -       | -            | -             | -            | rc=1:                                                                            |
| poot         | f32       | error  | -       | -       | -            | -             | -            | rc=1: Error: load /root/models/smolvlm-256m (f32): config: unsupported architect |
| transformers | bf16      | error  | -       | -       | -            | -             | -            | rc=1: main()                                                                     | File "/opt/benchmarks/runners/torch/run.py", line 114, in m |
| vllm         | bf16      | error  | -       | -       | -            | -             | -            | rc=1: File "/opt/vllm-venv/lib/python3.11/site-packages/vllm/engine/arg_utils.   |

---

_Memory numbers come from one external sampler (total device VRAM, process-tree RSS) under the one-engine-per-GPU assumption. Timing comes from each runner. Cells with a precision or format mismatch vs the matched target carry a note; vLLM VRAM reflects its KV-pool reservation unless run at a low gpu_memory_utilization. See benchmarks/README.md._
