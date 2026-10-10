---
id: api
title: OpenAI-compatible API
sidebar_position: 5
---

# OpenAI-compatible API

`poot-serve` is an OpenAI-compatible HTTP server. Per-connection threads accept and parse requests, so
a slow request never blocks new ones.

`POST /v1/completions` and `POST /v1/chat/completions` run on the engine thread's one scheduling loop
([Continuous batching](./continuous-batching.md)). The embeddings, rerank and LoRA administration routes send
their work to the same engine thread, which runs it between requests, so every route that runs the model uses
the one queue and the one set of request metrics.

```bash
cargo run -p poot-serve --release -- /path/to/model
```

```bash
curl -i http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
# HTTP/1.1 200 OK
# {"id":"cmpl-poot","object":"text_completion","choices":[{"index":0,"text":" Paris. ...","finish_reason":"length"}],"usage":{...}}
```

`/health` is `200` with `"status": "serving"` once the engine plan is prepared (the listener opens only then), with the backend, decode rows and positions it settled on. It is `503` with `"status": "draining"` while a shutdown lets in-flight requests finish, and `503` with `"status": "dead"`, a `cause` (`panicked` or `fault`) and a `reason` once the engine thread has ended. `engine_loaded` is true while the engine thread runs. `/metrics` exposes counters as JSON and
`/metrics/prometheus` in the Prometheus text-exposition format.

## Endpoints

| Endpoint                                                 | Notes                                                                                                                                                                                                          |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /v1/completions`                                   | Parsed, bounded and run by the scheduling loop. Fields: `prompt`, `max_tokens`, `stream`, `stop`, `n`, `logprobs`, sampling, guided fields, `lora_adapter`.                                                    |
| `POST /v1/chat/completions`                              | Parsed, bounded and run by the scheduling loop. Fields: `messages`, `max_tokens` or `max_completion_tokens`, `logprobs`/`top_logprobs`, `tools`/`tool_choice`, `response_format`, plus the completions fields. |
| `POST /v1/embeddings`, `POST /v1/rerank`                 | Run as `Encode` steps on the engine thread; hidden states are pooled on the host. See [Embeddings and reranking](./embeddings-and-reranking.md).                                                               |
| `GET /v1/models`, `GET /v1/models/{id}`                  | One served model; an unknown id is 404.                                                                                                                                                                        |
| `GET`/`POST`/`DELETE /v1/lora_adapters`                  | List, hot-load and unload adapters on the engine thread, under the trust policy in [LoRA adapters](./lora-adapters.md).                                                                                        |
| `GET /health`, `GET /metrics`, `GET /metrics/prometheus` | Liveness and metrics.                                                                                                                                                                                          |

A request that is invalid on its face is a `400` before the engine is asked: malformed JSON, a `max_tokens` that is not an integer, an unknown `lora_adapter`,
an invalid guided-decoding constraint, image input the served model cannot take, `n` above the engine's rows (each choice is one concurrent job), or `logprobs` on a
stream (a stream has nowhere to put them). `max_tokens` above the cap
(32768) is clamped, not refused, and `n`, `logprobs` and `stop` are clamped the same way. Errors are
OpenAI-shaped: `{"error": {"message", "type", "code"}}`. Over-limit requests get `429` with
`Retry-After` when `--rate-limit-rpm` is set.

An error the engine reports is answered by its class: a request the engine refuses (an empty or too long
prompt, an adapter it does not have) is a `400`, a request this prepared engine cannot serve (a shape or
retention limit) is a `503`, and a fault of the device or the driver is a `500`. On a stream the `200` is
already sent, so the class is the `code` of the closing error event, which carries no finish reason, usage
receipt or `[DONE]`.

## Checkpoints

The model can be a safetensors directory or a single **GGUF** file. A causal decoder (Qwen2/3, Llama, Mistral,
Mixtral, Gemma, Granite, Granite MoE, OLMo 2, OLMoE, Phi-3, SmolLM3, BLOOM, MPT, Qwen3-MoE, gpt-oss, Gemma 4,
DeepSeek-V2/V3/V3.2, Nemotron-H, Qwen3-Next, Qwen3.8-Flash-Next) is served if compiling its graph for the chosen backend succeeds; a refusal names the equation and the missing capability. A BERT encoder or cross-encoder is refused at startup, because an encoder prepares its entries on first use and its memory cannot be planned; a decoder answers embeddings and rerank through its embedding pass. A SmolVLM (idefics3) or Qwen2.5-VL checkpoint also takes images; see [Multimodal](./multimodal.md). See the [feature matrix](../reference/feature-matrix.md) for what the
libraries support.

Related: [Configuration reference](./configuration.md), [LoRA adapters](./lora-adapters.md),
[Using quantized checkpoints](./quantized-checkpoints.md).
