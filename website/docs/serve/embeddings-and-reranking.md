---
id: embeddings-and-reranking
title: Embeddings and reranking
sidebar_position: 7
---

# Embeddings and reranking

`/v1/embeddings` and `/v1/rerank` run the model as `encode` steps on the engine thread, so they share its queue
and its request metrics. The encode shapes are fitted to the device memory and prepared at startup, before
`/health` reports `serving`: a request compiles nothing and allocates nothing on the device.

## What the server serves

The served model is the checkpoint the server was started on:

- a **causal decoder** whose family has an embedding pass answers `/v1/embeddings` with the final hidden state
  of every token, pooled on the host (mean by default, `[CLS]`-style first-token pooling with
  `"pooling": "cls"`) and scaled to unit length, so a dot product of two vectors is their cosine.
  `/v1/rerank` orders the documents by the cosine of their vectors to the query's. A family with no embedding
  pass serves generation only and answers these routes with `400`.
- a **BERT embedding encoder** answers the same two routes. It serves nothing else: a chat or completion request
  is `400`.
- a **cross-encoder** (`BertForSequenceClassification`) answers `/v1/rerank` by scoring each query and
  document pair together and returning the raw logit as `relevance_score` (higher is more relevant, not
  bounded to `[0, 1]`).

## Shapes and limits

`--rows N` is the most texts one request may carry (documents plus the query for a bi-encoder rerank, pairs for
a cross-encoder), and `--max-seq N` the most tokens one text may have, special tokens included, capped at the
model's own positions. Both are fitted to the device memory at startup, together with the decode shapes for a
decoder, and `/health` reports the values it settled on. The default is 32 rows for an encoder checkpoint,
which holds activations only, and 4 for a decoder, whose rows are also its concurrent sequences. A request with
more texts or a longer text is answered `400` naming the limit and the flag. See
[Configuration](./configuration.md).

Texts are padded to the next power of two up to `--max-seq`, and the padding is masked, so a text's vector does
not depend on the texts beside it.

Encode and administration jobs run between generation steps, one per loop iteration, and take turns when both
wait, so neither starves the other. An encode step needs no sequence open, so on a decoder the engine lets the
running requests finish and opens none until it has run.

A request the engine cannot serve is answered by its class: see [OpenAI-compatible API](./api.md).

## Request fields

| Route | Fields |
| ----- | ------ |
| `POST /v1/embeddings` | `input` (one string or a list), `pooling` (`mean` or `cls`; default from the checkpoint's `1_Pooling` config, else `mean`), `encoding_format` (`float` or `base64`, the little-endian f32 bytes). |
| `POST /v1/rerank` | `query`, `documents`, `top_n`. The response is `{"results": [{"index", "relevance_score"}]}`, most relevant first. |

An empty text, an empty `input`, an unknown `pooling` or `encoding_format`, and malformed JSON are `400`.
Every request counts in the request metrics: its latency, and the tokens it encoded as prompt tokens.

## In a program

Both run on the driver's `encode` step, on wgpu, ROCm or PTX. `SentenceEncoder` returns L2-normalized vectors,
so a dot product is the cosine similarity; `CrossEncoder` returns the classifier's raw logit. See
[Embedding poot](../develop/embedding.md) for the API.
