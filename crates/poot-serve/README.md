# poot-serve

An OpenAI-compatible HTTP server.

It serves embeddings and reranking for BERT-class encoders, with request parsing and bounds, metrics, LoRA
administration and graceful shutdown. The completion endpoints validate and bound requests; see
[serving](../../website/docs/serve/index.md) for what each endpoint does.

## Quick start

```sh
cargo run -p poot-serve --release -- /path/to/model 127.0.0.1:8080
```

## CLI flags

- Positional: `[MODEL_DIR] [ADDR]`. `ADDR` defaults to `127.0.0.1:8080`. Without `MODEL_DIR` the model is
  `qwen2.5-0.5b` under the `POOT_MODELS_DIR` directory; with neither, startup fails naming the variable.
- `--lora-adapter NAME=DIR` - repeatable; registers a PEFT adapter at startup
- `--lora-pool-capacity N` - LoRA pool slots to provision for later hot-loads

Any other `--` option fails startup with an `unknown option` error.

## Environment variables

- `POOT_RATE_LIMIT_RPM` - per-key requests-per-minute limit (default 0 = unlimited)
- `POOT_MAX_CONNECTIONS` - connection-handler thread cap (default 4096, 0 = unlimited)
- `POOT_READ_TIMEOUT_SECS` - idle read timeout for the request head and body (default 30, 0 disables)
- `POOT_WRITE_TIMEOUT_SECS` - positive whole-second idle timeout for HTTP and SSE response writes (default
  30 when unset; zero, negative, malformed, or overflowing values fail before model loading)
- `POOT_DRAIN_TIMEOUT_SECS` - positive graceful shutdown drain timeout before cancellation (default 30,
  maximum 86400)
- `POOT_LORA_ADMIN_KEY` - bearer token for `/v1/lora_adapters`
- `POOT_MODELS_DIR` - directory that holds the models, used when no `MODEL_DIR` is given
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

`src/batch/` holds the scheduling pieces: slot admission, KV-fit budgeting, recompute preemption, token commit and
terminal delivery.

## Graceful shutdown

SIGINT/SIGTERM triggers a graceful shutdown: the server stops accepting new connections and
lets in-flight requests finish before exiting.

## Models

Pass a `.gguf` file path or a safetensors directory. A BERT encoder or cross-encoder loads as such; anything
else loads as a causal decoder `Runner`.
