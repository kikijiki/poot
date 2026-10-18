# poot-serve

An OpenAI-compatible HTTP server.

It loads any registered model as a `ModelHandle`, with request parsing and bounds, metrics and graceful
shutdown, and serves completions, chat, embeddings, rerank and LoRA administration from one engine thread over
the driver: generation is one scheduling loop over `step`, and the other routes are `Encode` steps and adapter
operations the loop runs between requests. See [serving](../../website/docs/serve/index.md) for what each
endpoint does.

## Quick start

```sh
cargo run -p poot-serve --release -- /path/to/model 127.0.0.1:8080
```

## CLI flags

- Positional: `[MODEL_DIR] [ADDR]`. `ADDR` defaults to `127.0.0.1:8080`; a request's `Host` must name it (a wildcard address checks only the port). Without `MODEL_DIR` the model is
  `qwen2.5-0.5b` under the `POOT_MODELS_DIR` directory; with neither, startup fails naming the variable.
- `--backend wgpu|rocm|ptx|vulkan|auto` (default `auto`) - the device. `auto` opens PTX if a CUDA device opens,
  else ROCm if an HSA GPU agent opens, else wgpu, once; a model the chosen backend refuses fails startup and no
  other backend is tried.

- `--rate-limit-rpm N` - per-client requests-per-minute limit (default 0 = unlimited)
- `--max-connections N` - connection-handler thread cap (default 4096, 0 = unlimited)
- `--max-pending-requests N` - model requests admitted at once, running or waiting for the engine; one more is
  answered 429 (default 256, at least 1)
- `--max-receiving-requests N` - admitted requests that may be reading their body at once; one more is answered 429
  (default 64, at least 1)
- `--max-queue-wait-secs N` - seconds an admitted generation request may wait for the engine to open it, else 503
  (default 60, at least 1)
- `--max-body-mib N` - request body cap in MiB for every route but chat on an image server (default 4)
- `--max-image-body-mib N` - request body cap in MiB for chat on a server that takes images (default 64)
- `--request-timeout-secs N` - seconds from accept for the whole request head and body to arrive, else 408
  (default 30, at least 1)
- `--write-timeout-secs N` - positive whole-second limit on one response write or stream frame, however the reader
  paces it (default 30)
- `--drain-timeout-secs N` - graceful shutdown drain timeout before cancellation (default 30, 1 to 86400)
- `--lora-admin-key-file FILE` - file holding the bearer token for `/v1/lora_adapters`
- `--draft-lookup NGRAM:K`, `--draft-model PATH:K` - speculative decoding
- `--chat-template JINJA`, `--chat-template-file FILE` - chat template override
- `--device-timing` and `--timing-max-steps`, `--timing-max-dispatch-records`,
  `--timing-max-in-flight-queries`, `--timing-sample-every` - bounded device timing (off by default)

Every flag is parsed strictly: a value that does not parse, a zero where zero is meaningless, or a value past
its bound stops startup; nothing falls back to a default. Any other `--` option is refused at startup.
`GET /health` is `503` (`starting`) until the engine plan is prepared and `200` (`serving`, with `backend`,
`rows`, `max_seq`) after; `503` `draining` during shutdown and `503` `dead` (with `cause` and `reason`) once the
engine thread has ended.

## Environment

- `POOT_MODELS_DIR` - directory that holds the models, used when no `MODEL_DIR` is given. It locates a
  path and selects no mode.
- `RUST_LOG` - log verbosity (default `info`)

## Endpoints

- `POST /v1/completions`, `POST /v1/chat/completions` - parsed and bounded (`max_tokens` is clamped once at
  parse time)
- `GET /v1/models` - list models
- `GET /v1/models/{id}` - retrieve a model
- `POST /v1/embeddings` - sentence embeddings (decoder pooling, BERT encoder)
- `POST /v1/rerank` - document reranking (decoder pooling, BERT encoder, cross-encoder)
- `GET`/`POST`/`DELETE /v1/lora_adapters` - LoRA administration
- `GET /health` - health check and engine state
- `GET /metrics` - JSON metrics
- `GET /metrics/prometheus` - Prometheus exposition format

`src/batch/` holds the scheduling loop (`engine.rs`), the per-step prefill budget, token commit and terminal
delivery. A request's KV blocks are reserved when it opens, so nothing is preempted: a request that finds no
room waits.

## Graceful shutdown

SIGINT/SIGTERM triggers a graceful shutdown: the server stops accepting new connections and
lets in-flight requests finish before exiting.

## Models

Pass a `.gguf` file path or a safetensors directory. A BERT encoder or cross-encoder loads as such; anything
else loads as a causal decoder through the driver.
