---
id: embeddings-and-reranking
title: Embeddings and reranking
sidebar_position: 7
---

# Embeddings and reranking

The same server also serves embeddings and reranking. It picks what to run from the model
directory's `config.json` at startup, so you do not set a mode; you just point it at a different
model:

- a **BERT embedding encoder** (`BertModel`: all-MiniLM, E5, BGE) serves higher-quality sentence
  embeddings, pooling mean or `[CLS]` per the model's `1_Pooling` config;
- a **cross-encoder** (`BertForSequenceClassification`: ms-marco-MiniLM) serves reranking by scoring
  a query and document together in one pass.

Embedding and reranking requests run on the CPU evaluator on the connection thread.
Endpoints a given model cannot serve return `400` (an encoder has no LM head, so it rejects
generation; a cross-encoder has no embedding output; a causal decoder serves neither embeddings nor
reranking).

## `/v1/embeddings` (OpenAI-shaped)

Point the server at an embedding model and call it like the OpenAI endpoint. `input` takes a string
or an array of strings:

```bash
cargo run -p poot-serve --release -- /path/to/bge-small-en

curl http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["a cute kitten", "a fast car"]}'
```

```json
{
  "object": "list",
  "data": [{ "object": "embedding", "index": 0, "embedding": [0.02, 0.08, ...] }],
  "model": "bge-small-en",
  "usage": { "prompt_tokens": 5, "total_tokens": 5 }
}
```

Vectors are L2-normalized, so a dot product is the cosine similarity. `encoding_format` is honored:
the default `"float"` returns a JSON number array, `"base64"` returns the packed little-endian f32
bytes (what the OpenAI Python client requests by default). A poot extension, `pooling`, overrides the pooling
per request: `"mean"` or `"cls"`.

## `/v1/rerank` (Cohere/Jina-shaped)

Order documents by relevance to a query. With a cross-encoder model this scores each pair jointly;
with an embedding encoder it ranks by bi-encoder cosine over the embeddings. `top_n` truncates the
result:

```bash
cargo run -p poot-serve --release -- /path/to/ms-marco-minilm-l6

curl http://localhost:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query": "How many people live in Berlin?",
       "documents": ["Berlin is known for its museums.",
                     "Berlin has 3.5 million registered inhabitants."]}'
```

```json
{
  "results": [
    { "index": 1, "relevance_score": 8.85 },
    { "index": 0, "relevance_score": -4.32 }
  ]
}
```

`relevance_score` is the cross-encoder's raw logit (higher = more relevant; not bounded to `[0, 1]`).
