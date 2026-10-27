---
id: continuous-batching
title: Continuous batching and concurrency
sidebar_position: 9
---

# Continuous batching and concurrency

`poot-serve` runs every generation request through one scheduling loop on its engine thread. The loop drives the driver's `step` on the backend the engine plan chose (`wgpu`, `rocm`, `ptx` or `vulkan`); there is no second loop per backend. [Speculative decoding](./speculative-decoding.md) is a draft policy of the driver, not a branch of the loop.

Each iteration of the loop:

1. takes the jobs the connection threads queued and tokenizes them;
2. releases every request whose client stopped waiting, streaming or not;
3. opens waiting requests, oldest first, while a row and KV room remain;
4. steps the rows that have work: decoding rows, and prefill rows within a per-step token budget so long prompts cannot stall decoding ones;
5. commits the tokens each row picked, up to an end-of-sequence token, a stop string or `max_tokens`, and streams each kept token.

## Admission

The driver reserves a request's KV blocks for its whole `max_tokens` when it opens the sequence. A running request is never starved and nothing is preempted; a request that finds no room waits for a running one to finish.

Batch composition changes between steps without compiling anything: every step runs on a prepared entry sized for the engine's row capacity. `/metrics` reports `engine_compiles`, which stays constant once the first request has been served.

## Cancellation and prefix reuse

- **Cancellation:** a client that disconnects ends its request within one step, streaming or not. The engine releases the sequence and registers the prompt prefix it had filled.
- **Prefix reuse:** a finished request registers its prompt's whole blocks in the prefix cache, and a later request with the same prefix reuses them. A model with recurrent state registers nothing. `/metrics` reports `prefix_registered_tokens`, `prefix_skipped_releases` and `prefix_cache_hit_rate`.

## When the engine fails

- A step the driver refuses before running fails only the requests of that step; serving goes on.
- A step that may have been half applied (a device fault, a poisoned device) fails every request, and the engine is dead: `GET /health` answers `503` with `"status": "dead"`, the cause (`panicked` or `fault`) and the reason.
- During shutdown `/health` answers `503` with `"status": "draining"` while in-flight requests finish.

Row count and positions are not options; the engine plan fits them to the device ([Configuration](./configuration.md)).
