---
id: serving-your-first-model
title: Serving your first model
sidebar_position: 2
---

# Serving your first model

This walks through starting `poot-serve` on a checkpoint and calling it. It assumes poot is built inside `nix develop`; see [Build](../develop/build.md) if not.

The server starts on any model whose family the registry knows. An embedding or reranking checkpoint starts the same way and serves only the embeddings and rerank routes; see [Embeddings and reranking](./embeddings-and-reranking.md).

## 1. Get a checkpoint

poot loads a single **GGUF** file, or a **safetensors** model directory (`config.json`, `tokenizer.json`, one or more `*.safetensors` files).

```bash
hf download <org>/<small-instruct-model> --local-dir ~/models/<model>
```

## 2. Start the server

```bash
cargo run -p poot-serve --release -- ~/models/<model>
```

- The address defaults to `127.0.0.1:8080`.
- The model path falls back to a developer-local default, so always pass one.
- `--release` keeps loading usable: debug builds are 10-50x slower.
- `--backend` picks the device; see [Picking a backend](./choosing-a-backend.md).

## 3. Check it, then call it

```bash
curl http://localhost:8080/health
# {"backend":"spirv-vulkan","engine_loaded":true,"max_seq":4096,
#  "rows":4,"status":"serving"}

curl http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 8}'
# 200: {"id":"cmpl-poot","object":"text_completion",
#  "choices":[{"index":0,"text":" Paris. ..."}],...}
```

`GET /v1/models` lists what is being served; `GET /metrics` and `GET /metrics/prometheus` report counters
(see [OpenAI-compatible API](./api.md)).

## Which models load

- Every family the registry knows loads; [Supported models and formats](../develop/supported-models.md) lists them. An encoder checkpoint serves embeddings and rerank only.
- A family the registry does not know is refused at startup with the registry's error naming its `model_type`.
- A family with a vision front end is served with image input ([Multimodal](./multimodal.md)).

## Next steps

- [Embeddings and reranking](./embeddings-and-reranking.md): the other routes.
- [Configuration reference](./configuration.md): every option.
