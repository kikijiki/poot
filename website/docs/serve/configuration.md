---
id: configuration
title: Configuration reference
sidebar_position: 13
---

# Configuration reference

The command-line options of `poot-serve`. Every setting is a flag: no environment variable selects a mode or a limit, and a value that does not parse stops startup. Batch size and context length are fitted to the device by the engine plan. See [Serving](./index.md) for what the server does.

## CLI

| Flag / argument | Default | Notes |
| --- | --- | --- |
| `MODEL` (positional) | developer-local path | Safetensors directory or `.gguf` file. Always pass it. |
| `ADDR` (positional) | `127.0.0.1:8080` | Bind address. |
| `--backend` | `auto` | `wgpu`, `rocm`, `ptx`, `vulkan` or `auto`. `auto` opens PTX if a CUDA device opens, else ROCm if an HSA GPU agent opens, else wgpu, and never falls back after a refusal. A backend whose device is absent is refused at startup. `vulkan` fits the workload to the free memory `VK_EXT_memory_budget` reports and is refused on a driver without it. See [Picking a backend](./choosing-a-backend.md). |
| `--placement` | `single` | `single`, `pipeline:ORD,ORD,...` (one layer stage per GPU ordinal) or `tensor:ORD,ORD,...` (one tensor-parallel rank per ordinal). A group needs two or more distinct ordinals and the `ptx` backend. See [Tensor parallelism](./tensor-parallelism.md). |
| `--draft-lookup NGRAM:K` | off | Speculate by prompt lookup: match the last `NGRAM` tokens against earlier text and propose up to `K`. |
| `--draft-model PATH:K` | off | Speculate with a draft model loaded from `PATH`, `K` tokens a round. Split at the last colon, so `PATH` may contain colons. |
| `--draft-mtp K` | off | Speculate with the served model's own multi-token-prediction layer, `K` tokens a round. |
| `--chat-template JINJA` | off | Render chats through this jinja template instead of the model's own. |
| `--chat-template-file FILE` | off | Read the jinja chat template from `FILE`. |
| `--rows N` | `4` (decoder), `32` (encoder) | Concurrent sequences of a generative model, or texts per request of an embedding or reranking model. At least 1. |
| `--max-seq N` | `4096` | Positions per sequence, or tokens per text of an embedding or reranking model, capped at the model's own limit. At least 1. |
| `--expert-pool ROWS` | off | Page every routed expert table through a pool of `ROWS` resident rows per layer. |
| `--kv-dtype DTYPE` | model's preference, else `f16` | What the KV cache stores: `f16`, `bf16` or `f32` (the exact reference, twice the bytes). Attention computes in f32 either way. See [KV cache storage dtype](../architecture/attention.mdx#storage-dtype). |

Refusals at startup:

- Any `--` option not listed here.
- More than one of the three draft flags.
- A draft flag for a model whose recurrent state advances in granules of more than one token, or for a draft model with recurrent state.
- `--draft-lookup` or `--draft-model` values that are not positive integers, a draft model that does not load, or a draft the target cannot verify (larger vocabulary, different token ids, fewer positions).
- `--draft-mtp` for a model with no MTP layer or one with recurrent state, with adapters, with beam search, or over a device group. See [Speculative decoding](./speculative-decoding.md#what-is-refused).
- `--placement` groups on a backend other than `ptx`, and a GPU that cannot hold its share (the error names it). `--device-timing` is refused over a group.
- `--chat-template` with a value that has no jinja syntax (poot has no built-in templates), a template that does not compile, or a vision-language checkpoint. A template that fails to render a conversation answers that request with an error, never another format.
- `--expert-pool` on a dense model, or `ROWS` below the experts one token selects or not below the model's expert count.
- `--kv-dtype` with a value other than `f16`, `bf16` or `f32`, and `bf16` on `wgpu` or `vulkan` (SPIR-V has no bf16 type).

`--rows` and `--max-seq` are halved by the memory fit (`--max-seq` to a floor of 128) until the workload fits. `/health` reports the values it settled on.

The listener opens once the model is prepared on the device, so a client gets connection refused while the server starts. Encode shapes are fitted and prepared before then, so a request never compiles or allocates on the device. A model that does not fit stops startup with the capacity error.

`/health` is then `200` with `"status": "serving"` and the `backend`, `rows` and `max_seq` the plan settled on.

Logging uses `RUST_LOG` (default `info`). `SIGINT`/`SIGTERM` stop accepting connections and drain in-flight requests.

```bash
cargo run -p poot-serve --release -- ~/models/bge-small-en 0.0.0.0:8080
```

## HTTP and administration

| Flag | Default | Purpose |
| --- | --- | --- |
| `--rate-limit-rpm N` | `0` (disabled) | Fixed 60 s window per peer IP on every POST. GETs are not limited. |
| `--lora-admin-key-file F` | unset | File holding the bearer token (one line) for `/v1/lora_adapters`, on loopback too; a file keeps the key out of `ps`. See [LoRA adapters](./lora-adapters.md). |
| `--max-connections N` | `4096` (`0` = unlimited) | Cap on simultaneous connection-handler threads. |
| `--max-pending-requests N` | `256` | Model requests admitted at once. One more gets `429` with `Retry-After`. At least `1`. |
| `--max-receiving-requests N` | `64` | Admitted requests reading their body at once. One more gets `429` with `Retry-After`. At least `1`. |
| `--max-queue-wait-secs N` | `60` | Seconds an admitted generation request may wait for the engine; then `503`. At least `1`. |
| `--max-body-mib N` | `4` | Largest request body, in MiB, of every route but a chat on a server that takes images. Larger gets `413`. |
| `--max-image-body-mib N` | `64` | Largest request body, in MiB, of a chat on a server that takes images. |
| `--request-timeout-secs N` | `30` | Seconds from accept for the whole request to arrive; slower gets `408`. At least `1`. |
| `--write-timeout-secs N` | `30` | Seconds one response write may take. Positive integer. |
| `--drain-timeout-secs N` | `30` (`1` to `86400`) | Shutdown drain time before in-flight requests are cancelled. |

Request headers are capped at 64 KiB. A POST must carry `Content-Type: application/json`, a `Content-Length`, and a `Host` that names the listener (`localhost`, its loopback address, or its bound address; a wildcard listener checks only the port). A request with `Transfer-Encoding`, a malformed or repeated `Content-Length`, or a body that is not UTF-8 gets a 4xx before it reaches the engine. The `/v1/lora_adapters` routes get the same `Host` check. poot does not validate API keys for inference.

## Device timing

By default the server keeps cumulative counters only. `--device-timing` asks the device for per-dispatch timestamps and keeps bounded detail. It changes how work is submitted, so a run with it is diagnostic and not throughput evidence. Raw Vulkan cannot time its dispatches and reports the device time as unknown.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--device-timing` | off | Request device timestamps and retain bounded per-dispatch detail. |
| `--timing-max-steps N` | `4096` | Most steps whose detail is retained (`1` to `16384`). |
| `--timing-max-dispatch-records N` | `65536` | Most per-dispatch records across those steps (`1` to `262144`). |
| `--timing-max-in-flight-queries N` | `64` | Most timestamp queries in flight at once (`1` to `4096`). |
| `--timing-sample-every N` | `1` | Collect detail for one of every `N` steps. |

The bound flags need `--device-timing`; zero and values over a bound stop startup. `/metrics` reports `device_timing`: the step count, the host wall, the device time (`null` when unmeasured, never `0`), whether the detail is `complete`, `sampled`, `truncated` or `unavailable`, and whether collecting it perturbed submission. `/metrics/prometheus` carries the same figures; its device-time series is absent when the device time is unknown.
