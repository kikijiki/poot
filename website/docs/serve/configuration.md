---
id: configuration
title: Configuration reference
sidebar_position: 13
---

# Configuration reference

One place for `poot-serve` command-line options and environment variables. Generation serving is not available yet (poot is being refactored; planned to return), so the batch, backend, speculation, tensor-parallel and pooled-MoE settings do not exist for now; see [Serving](./index.md) for what the server does today.

## CLI

| Flag / argument | Default | Notes |
| --------------- | ------- | ----- |
| `MODEL` (positional) | developer-local `qwen2.5-0.5b` path | Safetensors directory or `.gguf` file. Always pass it. |
| `ADDR` (positional) | `127.0.0.1:8080` | Bind address. |
| `--lora-adapter NAME=DIR` | none | Repeatable. Registers a PEFT adapter at startup; refused for now (not available yet), see [LoRA adapters](./lora-adapters.md). |
| `--lora-pool-capacity N` | no headroom | LoRA pool slots for later hot-loads; refused for now with the registration, see [LoRA adapters](./lora-adapters.md). |

Any other `--` option fails startup with an `unknown option` error. `--backend`, `--tensor-parallel` and
`--profile` are not available yet.

Logging uses `RUST_LOG` (default `info`). `SIGINT`/`SIGTERM` stop accepting connections and drain in-flight requests.

Example:

```bash
cargo run -p poot-serve --release -- ~/models/bge-small-en 0.0.0.0:8080
```

## HTTP and administration

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `POOT_RATE_LIMIT_RPM` | `0` (disabled) | Per client IP fixed 60s window on every POST (`/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`). Keyed on peer IP. `/health`, `/metrics` and other GETs are not limited. |
| `POOT_LORA_ADMIN_KEY` | unset | Bearer token required for the `/v1/lora_adapters` routes when set (including on loopback). Those routes register nothing for now. Inference stays unauthenticated. See [LoRA adapters](./lora-adapters.md). |
| `POOT_MAX_CONNECTIONS` | `4096` (`0` = unlimited) | Cap on simultaneous connection-handler threads. |
| `POOT_READ_TIMEOUT_SECS` | `30` (`0` disables) | Idle read timeout for request head and body. |
| `POOT_WRITE_TIMEOUT_SECS` | `30` | Idle write timeout for HTTP and SSE writes. Must be a positive integer or startup fails. |
| `POOT_DRAIN_TIMEOUT_SECS` | `30` (max `86400`) | Graceful-shutdown drain time before in-flight requests are cancelled. |

Request bodies are capped at 64 MiB and request headers at 64 KiB. poot does not validate API keys for
inference unless you add policy in front of the server.

## Related pages

- [OpenAI-compatible API](./api.md) - endpoints and metrics
- [Feature matrix](../reference/feature-matrix.md) - what each backend supports
