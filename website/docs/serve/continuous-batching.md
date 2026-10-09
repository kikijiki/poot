---
id: continuous-batching
title: Continuous batching and concurrency
sidebar_position: 9
---

# Continuous batching and concurrency

`poot-serve` runs every generation request through one scheduling loop on its engine thread. The loop
drives the driver's `step` on whichever backend the engine plan chose (`wgpu`, `rocm`, `ptx` or `vulkan`) and for
every family the driver serves; there is no second loop per backend, and speculative decoding
([Speculative decoding](./speculative-decoding.md)) is a draft policy of the driver, not a branch of
the loop.

Each iteration of the loop:

1. takes the jobs the connection threads queued and tokenizes them;
2. releases every request whose client stopped waiting, streaming or not;
3. opens waiting requests, oldest first, while a row and KV room remain (the driver reserves a request's
   blocks for its whole `max_tokens` when it opens the sequence, so a running request is never starved
   and nothing is preempted; a request that finds no room waits for a running one to finish);
4. steps the rows that have work: decoding rows, and prefill rows within a per-step token budget so a
   burst of long prompts cannot stall the ones that are decoding;
5. commits the tokens each row picked (one, or a verified round's accepted drafts and one more), keeping tokens only up to an end-of-sequence token, a stop
   string, or `max_tokens`, and streams each kept token.

Batch composition changes between steps (a request is admitted, another retires) without compiling
anything: every step runs on a prepared entry sized for the engine's row capacity, and `/metrics`
reports the driver's compile count (`engine_compiles`), which stays constant once the first request
has been served.

Cancellation: a client that disconnects ends its request within one step, for non-streaming requests
too. The server notices the hang-up while it waits for the result, and the engine releases the
sequence, registering the prompt prefix it had filled.

Prefix reuse: a finished request registers its prompt's whole blocks in the prefix cache, and a later
request with the same prefix reuses them. A model with recurrent state (a Qwen3-Next style hybrid)
registers nothing, because a cached block holds none of that state. `/metrics` reports
`prefix_registered_tokens`, `prefix_skipped_releases` and `prefix_cache_hit_rate`.

When the engine fails: a step the driver refuses before running fails only the requests of that step and
serving goes on. A step that may have been half applied (a device fault, a poisoned device) fails every
request, releases nothing, and the engine is dead: `GET /health` answers `503` with `"status": "dead"`,
the cause (`panicked` or `fault`) and the reason. During shutdown `/health` answers `503` with
`"status": "draining"` while in-flight requests finish.

The `POOT_SLOTS`, `POOT_CAP` and `POOT_MAX_BATCHED_TOKENS` settings are not read: the engine plan fits
the row count and positions to the device. A command-line option the server does not know fails
startup.
