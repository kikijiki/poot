---
id: embeddings-and-reranking
title: Embeddings and reranking
sidebar_position: 7
---

# Embeddings and reranking

`/v1/embeddings` and `/v1/rerank` run the model as `encode` steps on the engine thread, so they share its queue and request metrics. The encode shapes are fitted and prepared at startup, before `/health` reports `serving`: a request compiles and allocates nothing on the device.

## What the server serves

The served model is the checkpoint the server was started on:

| Model                                | `/v1/embeddings`                                                                      | `/v1/rerank`                                                           |
| ------------------------------------ | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Causal decoder with an embedding pass | Final hidden state of every token, pooled on the host and scaled to unit length       | Documents ordered by cosine to the query                               |
| Embedding encoder                    | The same                                                                              | The same                                                               |
| Cross-encoder                        | Not served                                                                            | Each query and document scored together; the raw logit is `relevance_score` (higher is more relevant, not bounded to `[0, 1]`) |

- A unit-length vector makes a dot product the cosine. Pooling is mean by default, or first-token with `"pooling": "cls"`.
- A decoder family with no embedding pass serves generation only and answers these routes with `400`.
- An embedding encoder or cross-encoder serves nothing else: a chat or completion request is `400`.

## Shapes and limits

Two flags bound a request; see [Configuration](./configuration.md):

- `--rows N`: the most texts one request may carry (documents plus the query for a bi-encoder rerank, pairs for a cross-encoder). The default is 32 for an encoder and 4 for a decoder, whose rows are also its concurrent sequences.
- `--max-seq N`: the most tokens one text may have, special tokens included, capped at the model's own positions.

`/health` reports the values the device fit settled on. A request with more texts or a longer text is a `400` naming the limit and the flag. Texts are padded to the next power of two up to `--max-seq` and the padding is masked, so a text's vector does not depend on the texts beside it.

Encode and administration jobs take turns between generation steps. On a decoder, an encode step lets running requests finish and opens none until it has run.

## Request fields

| Route                 | Fields                                                                                                                                                          |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /v1/embeddings` | `input` (a string or a list), `pooling` (`mean` or `cls`; default from the checkpoint's `1_Pooling` config, else `mean`), `encoding_format` (`float` or `base64`, little-endian f32 bytes) |
| `POST /v1/rerank`     | `query`, `documents`, `top_n`. Response: `{"results": [{"index", "relevance_score"}]}`, most relevant first.                                                  |

An empty text or `input`, an unknown `pooling` or `encoding_format`, and malformed JSON are `400`. Engine errors are answered by class; see [OpenAI-compatible API](./api.md). Every request counts in the request metrics: its latency, and its encoded tokens as prompt tokens.

## In a program

`SentenceEncoder` returns L2-normalized vectors; `CrossEncoder` returns the classifier's raw logit. See [Embedding poot](../develop/embedding.md). Device tests cover wgpu, ROCm and PTX, not raw Vulkan.
