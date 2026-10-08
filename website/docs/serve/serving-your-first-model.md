---
id: serving-your-first-model
title: Serving your first model
sidebar_position: 2
---

# Serving your first model

This walks through starting `poot-serve` on a checkpoint and calling it. It assumes you have already built
poot inside `nix develop`. See [Develop: Build and test](../develop/build-and-test.md) if not.

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

Today the server starts on any model the registry knows, answers health, metrics and model listing, and returns `503` on every route that runs the model; see [Embeddings and reranking](./embeddings-and-reranking.md). This walkthrough uses an embedding model because it is the quickest checkpoint to fetch.

## 1. Get a checkpoint

poot loads two checkpoint shapes: a single **GGUF** file, or a **safetensors** model directory (the usual
Hugging Face layout: `config.json`, `tokenizer.json`, one or more `*.safetensors` files).

```bash
hf download BAAI/bge-small-en-v1.5 --local-dir ~/models/bge-small-en
```

## 2. Start the server

```bash
cargo run -p poot-serve --release -- ~/models/bge-small-en
```

If you omit the address, poot-serve binds `127.0.0.1:8080`. If you omit the model path, it falls back to a
developer-local `qwen2.5-0.5b` path, so always pass one. `--release` keeps model loading usable: debug
builds are 10-50x slower. The server has no `--backend` option; see [Picking a backend](./choosing-a-backend.md).

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

- Every family the registry knows loads as a `ModelHandle`: the decoders (Qwen2/3, Llama, Mistral, Mixtral,
  Gemma, Granite, Granite MoE, OLMo 2, OLMoE, Phi-3, SmolLM3, BLOOM, MPT, Qwen3-MoE, gpt-oss, Gemma 4,
  DeepSeek-V2/V3/V3.2, Nemotron-H, Qwen3-Next, Qwen3.8-Flash-Next), BERT-class encoders (all-MiniLM, E5, BGE) and cross-encoders (ms-marco-MiniLM).
  A decoder serves generation from the scheduling loop; the embeddings, reranking and LoRA routes return `503`
  until they are restored.
- A checkpoint whose family the registry does not know, SmolVLM / idefics3 among them, is refused at
  startup with the registry's error naming its `model_type`. SmolVLM runs through the library (`poot-caption`).

## Next steps

- [Embeddings and reranking](./embeddings-and-reranking.md) for what the server serves today.
- [Configuration reference](./configuration.md) for the options and environment variables.
