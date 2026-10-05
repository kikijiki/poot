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

Today the server answers health, embeddings and reranking, and refuses a dense decoder (Qwen, Llama, ...) at startup; see [Embeddings and reranking](./embeddings-and-reranking.md). This walkthrough uses an embedding model for that reason.

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
# {"engine_loaded":false,"status":"ok"}

curl http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["a cute kitten", "a fast car"]}'
# {"object":"list","data":[{"object":"embedding","index":0,"embedding":[...]}, ...], ...}
```

`GET /v1/models` lists what is being served; `GET /metrics` and `GET /metrics/prometheus` report counters
(see [OpenAI-compatible API](./api.md)).

## Which models load

- A dense decoder (Qwen2/3, Llama, Mistral, Gemma, Granite, OLMo 2, Phi-3, SmolLM3, BLOOM, MPT) is refused at
  startup: not available yet (poot is being refactored; planned to return).
- A mixture-of-experts or hybrid decoder (Qwen3-MoE, GraniteMoE, Mixtral, OLMoE, gpt-oss, DeepSeek) loads; its
  generation routes return `503`.
- A BERT-class encoder (all-MiniLM, E5, BGE, ...) serves `/v1/embeddings`.
- A BERT-class cross-encoder (ms-marco-MiniLM, ...) serves `/v1/rerank`.
- Gemma 4 and Qwen3-Next checkpoints and SmolVLM / idefics3 checkpoints are not available yet (poot is being refactored; planned to return).

## Next steps

- [Embeddings and reranking](./embeddings-and-reranking.md) for what the server serves today.
- [Configuration reference](./configuration.md) for the options and environment variables.
