---
id: api
title: OpenAI-compatible API
sidebar_position: 5
---

# OpenAI-compatible API

`poot-serve` is an OpenAI-compatible HTTP server. A slow request never blocks new ones. Completions and chat run on the engine thread's one scheduling loop ([Continuous batching](./continuous-batching.md)); embeddings, rerank and LoRA administration use the same queue and metrics.

```bash
cargo run -p poot-serve --release -- /path/to/model
```

```bash
curl -i http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
# HTTP/1.1 200 OK
# {"id":"cmpl-poot","object":"text_completion",
#  "choices":[{"index":0,"text":" Paris. ...","finish_reason":"length"}],
#  "usage":{...}}
```

`/health` answers:

- `200` with `"status": "serving"` once the engine plan is prepared, with the backend, decode rows and positions it settled on;
- `503` with `"status": "draining"` while a shutdown lets in-flight requests finish;
- `503` with `"status": "dead"`, a `cause` (`panicked` or `fault`) and a `reason` once the engine thread has ended.

`engine_loaded` is true while the engine thread runs. `/metrics` is JSON; `/metrics/prometheus` is Prometheus text.

## Endpoints

| Endpoint | Notes |
| --- | --- |
| `POST /v1/completions` | `prompt`, `max_tokens`, `stream`, `stop`, `n`, `logprobs`, sampling, guided fields, `lora_adapter`. |
| `POST /v1/chat/completions` | `messages`, `max_tokens` or `max_completion_tokens`, `logprobs`/`top_logprobs`, `tools`/`tool_choice`, `response_format`, plus the completions fields. |
| `POST /v1/embeddings`, `POST /v1/rerank` | See [Embeddings and reranking](./embeddings-and-reranking.md). |
| `GET /v1/models`, `GET /v1/models/{id}` | One served model; an unknown id is 404. |
| `GET`/`POST`/`DELETE /v1/lora_adapters` | See [LoRA adapters](./lora-adapters.md). |
| `GET /health`, `GET /metrics`, `GET /metrics/prometheus` | Liveness and metrics. |

A request that is invalid on its face is a `400` before the engine is asked. Errors are OpenAI-shaped: `{"error": {"message", "type", "code"}}`. Over-limit requests get `429` with `Retry-After` when `--rate-limit-rpm` is set.

### Refused and clamped

A request invalid on its face is a `400` before the engine is asked:

- malformed JSON, a `max_tokens` that is not an integer, an unknown `lora_adapter`;
- an invalid guided-decoding constraint, or image input the served model cannot take;
- `n` above the engine's rows, or `logprobs` on a stream;
- more than 10,000 `logit_bias` entries;
- a `stop` string over 1,024 bytes, or `stop` strings totalling over 4,096 bytes.

`max_tokens` above 32768, `n`, `logprobs` and more than 16 `stop` strings are clamped, not refused.

### Sampling

Sampling runs on the device: temperature, top-k, top-p, min-p, the repetition, presence and frequency penalties, `logit_bias`, a guided-decoding constraint and the top-n `logprobs`. Only token ids (and the logprob values asked for) come back. A request without `seed` draws one from the operating system; a request with a `seed` is reproducible.

### Control tokens

A control token spelled in a message (contents, system prompts, tool definitions, image-chat text) is plain text: it cannot open a turn. Only the chat template's own text carries control tokens.

- A chat template that fails to render the conversation (it raises, spends its instruction budget, or prints more than its output budget) is a `400` naming the template error.
- Not covered: a template that builds a control token's text out of a message by string surgery.
- Not covered yet: image chats, whose prompt is tokenized with control tokens parsed.

### Engine errors

| Class | Status |
| --- | --- |
| The engine refuses the request (empty or too long prompt, unknown adapter) | `400` |
| This prepared engine cannot serve it (a shape or retention limit) | `503` |
| A fault of the device or the driver | `500` |

On a stream the `200` is already sent, so the class is the `code` of the closing error event, which carries no finish reason, usage receipt or `[DONE]`.

## Checkpoints

The model is a safetensors directory or a single **GGUF** file.

- A causal decoder of a registered family is served if compiling its graph for the chosen backend succeeds. A refusal names the equation and the missing capability.
- An embedding encoder or cross-encoder serves `/v1/embeddings` and `/v1/rerank`. So does a decoder whose family has an embedding pass.
- A checkpoint whose family has a vision front end also takes images; see [Multimodal](./multimodal.md).

[Supported models and formats](../develop/supported-models.md) lists the registered families.
