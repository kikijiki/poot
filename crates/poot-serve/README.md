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

The list is generated from the flags the server declares (`just serve-readme-regen`); the positional `MODEL` falls back to `qwen2.5-0.5b` under `POOT_MODELS_DIR`, and with neither startup fails naming the variable.

<!-- generated:flags:begin -->

- `[MODEL]` - Model directory or GGUF file; default: `qwen2.5-0.5b` under `POOT_MODELS_DIR`
- `[ADDR]` - Listen address. A request's `Host` must name it: a loopback address accepts `localhost`, `127.0.0.1` and `[::1]`, a specific address only itself, and a wildcard address (`0.0.0.0`, `[::]`) checks only the port, since it cannot know which names reach the machine (default `127.0.0.1:8080`)
- `--backend BACKEND` - The device backend: wgpu, rocm, ptx, vulkan, or auto (PTX if a CUDA device opens, else ROCm if an HSA GPU agent opens, else wgpu). A model the chosen backend refuses fails startup; no other backend is tried (default `auto`)
- `--placement PLACEMENT` - Where the model runs: `single` (the backend's default device), `pipeline:ORD,ORD,...` (one layer stage per GPU ordinal, in order) or `tensor:ORD,ORD,...` (one tensor-parallel rank per ordinal). A group needs at least two distinct ordinals and the ptx backend; a backend that cannot group fails startup (default `single`)
- `--rows N` - Rows the engine is fitted for: concurrent sequences of a generative model (default 4), texts per request of an embedding or rerank model (default 32; a generative model's embeddings take the same rows). The memory fit may reduce it. At least 1
- `--max-seq N` - Positions per sequence of a generative model, or tokens per text of an embedding or rerank model, before the model's own limit caps it. The memory fit may reduce it. At least 1 (default `4096`)
- `--draft-lookup NGRAM:K` - Speculate by prompt lookup: propose the K tokens that followed the latest earlier occurrence of the sequence's last NGRAM tokens, written `NGRAM:K` (split at the first colon). Exclusive with `--draft-model`
- `--draft-model PATH:K` - Speculate with a second model that proposes K tokens a round, written `PATH:K` (split at the last colon, so the path may contain colons). Its vocabulary must agree with the served model's
- `--draft-mtp K` - Speculate with the served model's own multi-token-prediction layer, K tokens a round. There is no default K (1 is the usual start; 2 to 3 are worth measuring); a model with no such layer, or one whose layer carries recurrent state, is refused at startup. Exclusive with `--draft-lookup` and `--draft-model`, and with adapters, beam search and a device group
- `--expert-pool ROWS` - `--expert-pool ROWS` pages every routed expert table through a pool of ROWS resident rows per layer, instead of holding every expert on the device. At least the experts one token selects and below the model's expert count; a model with no routed experts is refused at startup
- `--kv-dtype DTYPE` - The dtype the KV cache stores: f16, bf16 or f32 (the exact reference, twice the bytes). Without it the model's own preference applies, else f16. Attention computes in f32 whichever is stored
- `--matmul-precision POLICY` - How contraction and attention operands are read: `exact` (as the graph declares them), `narrow` (rounded to the target's matrix-unit dtype where it has one, accumulated in f32) or `split` (hi + lo parts of that dtype, near-f32 accuracy). Without it the engine default (`narrow`) applies. A target with no matrix unit runs `exact` whichever is asked
- `--check-ranges` - Debug aid: find where a narrowed run leaves f16's range. Every operand a `narrow` or `split` op rounds to f16 gets a witness, and a step with an element past f16's range fails with a validation fault naming the first op, its layer and the operand's role, instead of a non-finite logit. Costs a row reduction per watched value and a validation readback per step; off by default, when programs carry no witness
- `--chat-template JINJA` - Render chats through this jinja template instead of the model's own. A value with no jinja syntax names a built-in template, which poot lacks, and is refused
- `--chat-template-file FILE` - Render chats through the jinja template in this file instead of the model's own
- `--rate-limit-rpm N` - Requests per minute allowed per client address on the model routes; 0 disables the limit (default `0`)
- `--max-connections N` - Most simultaneous connections; 0 means unlimited (default `4096`)
- `--max-pending-requests N` - Most model requests admitted at once, running or waiting for the engine; one more is answered 429 without its body being read. At least 1 (default `256`)
- `--max-receiving-requests N` - Most admitted requests that may be reading their body at once; a request that would be one more is answered 429 before its body is read. Smaller than `--max-pending-requests` by default, so clients that send slowly cannot hold every slot while no work reaches the engine. At least 1 (default `64`)
- `--max-queue-wait-secs SECS` - Seconds an admitted generation request may wait for the engine to open it; a request still queued then is answered 503. At least 1 (default `60`)
- `--max-body-mib MIB` - Largest request body, in MiB, of every route but a chat on a server that takes images. A larger `Content-Length` is answered 413 before the body is read (default `4`)
- `--max-image-body-mib MIB` - Largest request body, in MiB, of a chat on a server that takes images (a vision-language model) (default `64`)
- `--request-timeout-secs SECS` - Seconds from accept in which the whole request, head and body, must arrive; a client that is still sending then is answered 408 and closed. At least 1 (default `30`)
- `--write-timeout-secs SECS` - Seconds one response write (a whole response or one stream frame) may take before the connection closes, however slowly the reader drains it; at least 1 (default `30`)
- `--drain-timeout-secs SECS` - Seconds in-flight requests get to finish after a shutdown signal; 1 to 86400 (default `30`)
- `--lora-admin-key-file FILE` - File holding the bearer key every `/v1/lora_adapters` request must present. The key is one RFC 6750 token on its own line; a file keeps it out of `ps`. Without it, administration is open on a loopback listener and disabled on any other
- `--device-timing` - Request device-side timing from the executor and retain bounded per-dispatch detail. Diagnostic: it perturbs submission grouping. Off by default, when the server keeps cumulative counters only
- `--timing-max-steps N` - With `--device-timing`: most timing steps retained (default 4096)
- `--timing-max-dispatch-records N` - With `--device-timing`: most per-dispatch records retained across those steps (default 65536)
- `--timing-max-in-flight-queries N` - With `--device-timing`: most timestamp queries in flight at once (default 64)
- `--timing-sample-every N` - With `--device-timing`: collect detail for one of every N steps (default 1, every step)

<!-- generated:flags:end -->

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
- `POST /v1/embeddings` - sentence embeddings (an encoder checkpoint, or a decoder whose family has an embedding pass)
- `POST /v1/rerank` - document reranking (an encoder or cross-encoder checkpoint, or a decoder with an embedding pass)
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

Pass a `.gguf` file path or a safetensors directory. An encoder or cross-encoder checkpoint loads as such; anything
else loads as a causal decoder through the driver.
