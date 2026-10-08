---
id: embeddings-and-reranking
title: Embeddings and reranking
sidebar_position: 7
---

# Embeddings and reranking

:::caution[Being restored]
poot is being refactored. The server's embeddings and reranking routes run the model on the driver, which the
server has no engine for yet, so `/v1/embeddings` and `/v1/rerank` return `503` naming the missing engine. They
are planned to return. The library path works today.
:::

The server picks what to load from the model directory's `config.json` at startup, so you do not set a mode; you
just point it at a different model:

- a **BERT embedding encoder** (`BertModel`: all-MiniLM, E5, BGE) is a sentence embedder, pooling mean or `[CLS]`
  per the model's `1_Pooling` config;
- a **cross-encoder** (`BertForSequenceClassification`: ms-marco-MiniLM) is a reranker that scores a query and a
  document together in one pass.

Endpoints a given model cannot serve return `400` (an encoder has no LM head, so it rejects generation; a
cross-encoder has no embedding output; a causal decoder serves neither embeddings nor reranking). A route the
model could serve returns `503` until the engine is back.

## In a program

Both run on the driver's `Encode` step, on wgpu, ROCm or PTX. `SentenceEncoder` returns L2-normalized vectors,
so a dot product is the cosine similarity; `CrossEncoder` returns the classifier's raw logit (higher is more
relevant, not bounded to `[0, 1]`). See [Embedding poot](../develop/embedding.md) for the API.
