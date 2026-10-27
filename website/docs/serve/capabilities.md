---
id: capabilities
title: Serving capabilities by backend
sidebar_position: 14
---

# Serving capabilities by backend

`poot-serve` serves generation (`/v1/completions`, `/v1/chat/completions`) on every [supported backend](./choosing-a-backend.md#supported-backends). Embeddings, reranking and LoRA administration run on the same engine thread. The scheduling loop is backend-neutral; what differs is which paths a backend's device lane requires.

## Which paths each backend's lane runs

The generated table [Device lane cases](../develop/supported-models.md#device-lane-cases) lists, per backend, how many cases the device lane requires for served models, paged KV, batched decode, prefix caching, speculative decoding, sampled decoding, LoRA, guided decoding and embeddings. **No case** means the lane requires none; it is not a refusal.

Three capabilities are not lane rows:

- **Tensor parallelism:** PTX serves it; wgpu, ROCm and raw Vulkan do not. See [Tensor parallelism](./tensor-parallelism.md).
- **KV preemption and swap:** none. A request's blocks are reserved when it opens, and a request that finds no room waits ([Continuous batching](./continuous-batching.md)).
- **On-device sampling:** every backend picks the next token on-device, so no request reads full logits back unless asked (see [Serving design](../architecture/serving-design.mdx#the-scheduling-loop)).

## Expert pooling

Expert pooling keeps a fixed number of resident expert rows per MoE layer; the experts a step selects are made resident before they are read, least recently used rows making room. Enable it with `--expert-pool ROWS` ([Configuration](./configuration.md)). Without it every expert's weights are resident.

- **Applies to** a model with routed experts whose stacked tables the planner can page.
- **Pool size:** below the expert count and at least the experts one token selects.
- **Prefill** longer than the pool serves is split into chunks by the driver.
- **Decode** whose rows times experts per token exceed the pool is refused at prepare with a typed capacity error, never split.

Pooled decode is verified token for token against the resident run; the wgpu, ROCm and PTX lanes require those cases.
