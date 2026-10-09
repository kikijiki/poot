---
id: embeddings-and-reranking
title: Embeddings and reranking
sidebar_position: 7
---

# Embeddings and reranking

`/v1/embeddings` and `/v1/rerank` run the model as `Encode` steps on the engine thread, the same thread that
runs generation, so they share its queue and its request metrics. An `Encode` step shares the cache of an open
sequence, so the engine runs it with no request in flight: running requests finish first, and nothing new opens
until the step has run.

## What the server serves

The served model is the checkpoint the server was started on:

- a **causal decoder** answers `/v1/embeddings` with its embedding pass: the final hidden state of every
  token, pooled on the host (mean by default, `[CLS]`-style first-token pooling with `"pooling": "cls"`) and
  scaled to unit length, so a dot product of two vectors is their cosine. `/v1/rerank` orders the documents by
  the cosine of their vectors to the query's.
- a **BERT embedding encoder** or a **cross-encoder** (`BertForSequenceClassification`) answers the same
  routes through the same code, a cross-encoder scoring each query and document pair together and returning the
  raw logit as `relevance_score` (higher is more relevant, not bounded to `[0, 1]`). The server does not start
  on an encoder checkpoint yet: an encoder prepares its entries on first use, so its device memory cannot be
  planned, and startup refuses it with a typed error. They run in a program, below.

A request the engine cannot serve is answered by its class: see [OpenAI-compatible API](./api.md).

## Request fields

| Route | Fields |
| ----- | ------ |
| `POST /v1/embeddings` | `input` (one string or a list), `pooling` (`mean` or `cls`; default from the checkpoint's `1_Pooling` config, else `mean`), `encoding_format` (`float` or `base64`, the little-endian f32 bytes). |
| `POST /v1/rerank` | `query`, `documents`, `top_n`. The response is `{"results": [{"index", "relevance_score"}]}`, most relevant first. |

An empty text, an empty `input`, an unknown `pooling` or `encoding_format`, and malformed JSON are `400`.
Every request counts in the request metrics: its latency, and the tokens it encoded as prompt tokens.

## In a program

Both run on the driver's `Encode` step, on wgpu, ROCm or PTX. `SentenceEncoder` returns L2-normalized vectors,
so a dot product is the cosine similarity; `CrossEncoder` returns the classifier's raw logit. See
[Embedding poot](../develop/embedding.md) for the API.
