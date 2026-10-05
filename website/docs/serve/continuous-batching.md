---
id: continuous-batching
title: Continuous batching and concurrency
sidebar_position: 9
---

# Continuous batching and concurrency

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

`poot-serve` does not batch requests yet: `/v1/completions` and `/v1/chat/completions` return `503` naming the missing engine. The `POOT_SLOTS`, `POOT_CAP`, `POOT_KV_QUANT`, `POOT_PREFILL_CHUNK_SIZE` and `POOT_MAX_BATCHED_TOKENS` settings are not read, and a command-line option the server does not know fails startup.

What stays in the server today:

- Request parsing and bounds: `max_tokens` is bounded once when the request is parsed, `n`,
  `logprobs` and `stop` are clamped, and a request that is invalid on its face is a `400` before
  anything else happens.
- The scheduling pieces the future loop builds on (slot admission, KV-fit budgeting, preemption,
  token commit, terminal delivery) are kept with their unit tests. Nothing calls them in the running
  server.

There is no runnable example of the batched decode loop for now; `poot-llm`'s own test suite exercises those code paths.
