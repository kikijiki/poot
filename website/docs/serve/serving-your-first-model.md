---
id: serving-your-first-model
title: Serving your first model
sidebar_position: 2
---

# Serving your first model

This walks through starting `poot-serve` on a checkpoint and calling it. It assumes you have already built
poot inside `nix develop`. See [Develop: Build and test](../develop/build-and-test.md) if not.

The server starts on any decoder whose family the registry knows and serves completions, chat, embeddings, rerank and LoRA administration from one engine thread; see [Embeddings and reranking](./embeddings-and-reranking.md). This walkthrough uses a small decoder; an embedding or reranking checkpoint (an embedding encoder or cross-encoder) starts the same way and serves only the embeddings and rerank routes.

## 1. Get a checkpoint

poot loads two checkpoint shapes: a single **GGUF** file, or a **safetensors** model directory (the usual
Hugging Face layout: `config.json`, `tokenizer.json`, one or more `*.safetensors` files).

```bash
hf download <org>/<small-instruct-model> --local-dir ~/models/<model>
```

## 2. Start the server

```bash
cargo run -p poot-serve --release -- ~/models/<model>
```

If you omit the address, poot-serve binds `127.0.0.1:8080`. If you omit the model path, it falls back to a
developer-local default path, so always pass one. `--release` keeps model loading usable: debug
builds are 10-50x slower. `--backend` picks the device; see [Picking a backend](./choosing-a-backend.md).

## 3. Check it, then call it

```bash
curl http://localhost:8080/health
# {"backend":"spirv-vulkan","engine_loaded":true,"max_seq":4096,"rows":4,"status":"serving"}

curl http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 8}'
# 200: {"id":"cmpl-poot","object":"text_completion","choices":[{"index":0,"text":" Paris. ..."}],...}
```

`GET /v1/models` lists what is being served; `GET /metrics` and `GET /metrics/prometheus` report counters
(see [OpenAI-compatible API](./api.md)).

## Which models load

- Every family the registry knows loads as a `ModelHandle`: decoders, embedding encoders and cross-encoders (the
  [feature matrix](../reference/feature-matrix.md) lists the families).
  A decoder serves generation from the scheduling loop, and its embeddings, reranking and LoRA routes run on the
  same engine thread. An encoder checkpoint serves embeddings and rerank only.
- A checkpoint whose family the registry does not know is refused at startup with the registry's error naming its
  `model_type`. A checkpoint whose family has a vision front end is served with image input ([Multimodal](./multimodal.md)).

## Next steps

- [Embeddings and reranking](./embeddings-and-reranking.md) for what the server serves today.
- [Configuration reference](./configuration.md) for every option.
