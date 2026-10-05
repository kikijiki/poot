---
id: api
title: OpenAI-compatible API
sidebar_position: 5
---

# OpenAI-compatible API

`poot-serve` is an OpenAI-compatible HTTP server. Per-connection threads accept and parse requests, so
a slow request never blocks new ones.

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

Until then, `POST /v1/completions` and `POST /v1/chat/completions` return `503` naming the missing engine, and `GET /health` reports `"engine_loaded": false`.

```bash
cargo run -p poot-serve --release -- /path/to/model
```

```bash
curl -i http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
# HTTP/1.1 503 Service Unavailable
# {"error":{"message":"no generation engine is loaded: this server parses and bounds requests but cannot generate until the serving loop lands","type":"server_error","code":503}}
```

`/health` reports liveness and whether an engine is loaded; `/metrics` exposes counters as JSON and
`/metrics/prometheus` in the Prometheus text-exposition format. With no engine the request, token and
device gauges stay at zero.

## Endpoints

| Endpoint | Notes |
| -------- | ----- |
| `POST /v1/completions` | Parsed and bounded, then `503`. Fields: `prompt`, `max_tokens`, `stream`, `stop`, `n`, `logprobs`, sampling, guided fields, `lora_adapter`. |
| `POST /v1/chat/completions` | Parsed and bounded, then `503`. Fields: `messages`, `max_tokens` or `max_completion_tokens`, `logprobs`/`top_logprobs`, `tools`/`tool_choice`, `response_format`, plus the completions fields. |
| `POST /v1/embeddings`, `POST /v1/rerank` | Served by an encoder or cross-encoder; a causal decoder answers `400`. See [Embeddings and reranking](./embeddings-and-reranking.md). |
| `GET /v1/models`, `GET /v1/models/{id}` | One served model; an unknown id is 404. |
| `GET`/`POST`/`DELETE /v1/lora_adapters` | Not available yet (poot is being refactored; planned to return): the list is empty and a hot-load is refused; see [LoRA adapters](./lora-adapters.md). |
| `GET /health`, `GET /metrics`, `GET /metrics/prometheus` | Liveness and metrics. |

A request that is invalid on its face is a `400` before the engine lookup, so it gets the same answer with
or without an engine: malformed JSON, a `max_tokens` that is not an integer, an unknown `lora_adapter`,
an invalid guided-decoding constraint, or image input on a text-only model. `max_tokens` above the cap
(32768) is clamped, not refused, and `n`, `logprobs` and `stop` are clamped the same way. Errors are
OpenAI-shaped: `{"error": {"message", "type", "code"}}`. Over-limit requests get `429` with
`Retry-After` when `POOT_RATE_LIMIT_RPM` is set.

## Checkpoints

The model can be a safetensors directory or a single **GGUF** file. A causal decoder of a mixture-of-experts
or hybrid family loads as a `Runner`; its generation routes return `503`. A dense decoder (Qwen2/3, Llama,
Mistral, Gemma, Granite, OLMo 2, Phi-3, SmolLM3, BLOOM, MPT) is refused at startup until serving runs on the
driver. A BERT encoder or cross-encoder serves embeddings and reranking. Gemma 4 and Qwen3-Next checkpoints and vision-language checkpoints are not available yet (poot is being refactored; planned to return). See the [feature matrix](../reference/feature-matrix.md) for what the
libraries support.

Related: [Configuration reference](./configuration.md), [LoRA adapters](./lora-adapters.md),
[Using quantized checkpoints](./quantized-checkpoints.md).
