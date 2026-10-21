---
id: models
title: Supported models
sidebar_position: 7
---

# Supported models

poot traces model forwards into a primitive graph and runs them on its own kernels. Which model families are
supported, and on which backends, is listed per family in the [feature matrix](../reference/feature-matrix.md);
this page describes what a checkpoint must look like and what the library does with each kind of model.

## Checkpoint formats

- **Safetensors**: a Hugging Face directory (`config.json`, `tokenizer.json`, the weight files). Dense, GPTQ, AWQ
  and FP8 layouts load; quantized projections stay packed as stored, and dense BF16/F16 weights stay BF16/F16.
- **GGUF**: one file, or a split set named by any of its `-0000N-of-0000M.gguf` shards, with the config and
  tokenizer embedded. K-quant and IQ tensor types stay packed. A tensor of a type poot has no format for is refused
  by tensor name and type.

The checkpoint's config decides the family: `ModelHandle::load` resolves it through the model registry, and a
family the registry does not know is a typed refusal. `poot-serve` serves the same families through the same
driver. A value in a config that a family cannot trace (an unsupported scaling mode, per-layer input embeddings)
is refused with an unsupported-config error that names it. Layer, expert and head counts come from the config, so
a checkpoint with other counts loads the same way.

## Kinds of model

### Dense decoders

A standard causal stack: normalization, position embeddings, a gated or plain MLP and multi-head or
grouped-query attention, with optional sliding-window layers. They load with `ModelHandle::load` and generate
with `Driver`; see the matrix for backend coverage.

### Mixture-of-experts

A router plus sparse expert MLPs. Every expert's weights stay resident unless expert pooling is asked for
(`Driver::with_expert_pool`, `--expert-pool` on the server). Experts are read as the checkpoint stores them
(per-expert tensors or a stored stack, dense or packed); nothing is fused at load. A NaN or infinite router logit
stops the step with a validation error naming the layer.

### Latent attention and hybrid models

Latent-attention models keep their compressed latent and rope key as the KV cache. A hybrid model mixes
attention layers with state-space or linear-attention layers and carries each recurrent layer's convolution
window and state next to the attention caches, so a prompt prefilled in chunks continues exactly.

### Models with a block indexer or compressed attention

Some families attend only to cached blocks or compression windows that a small indexer selects. Their prefill
step covers whole blocks or windows: the driver's prefill granule is the least common multiple of the layers'
rates, and the tokens after the last whole window run as decode steps. An indexer that keeps a per-row cache of
block keys generates through `Driver`, but `poot-serve` refuses to start on it under the paged KV pool (a typed
refusal) until paged block-key caching lands. A token outside the vocabulary stops a step with a validation error.

### Multi-stream residuals

Some families carry several parallel residual streams per token and mix them with learned weights around each
block. They load and generate like any other family. A family whose recurrence is traced token by token makes a
prefill step cost graph size in proportion to its tokens: use a modest prefill chunk.

### Embeddings and reranking

Encoder models for sentence embeddings and cross-encoder reranking load with `ModelHandle::load` and run on the
driver's `encode` step; `poot_llm::encoders` is the library path. `poot-serve` starts on an encoder checkpoint and
serves `/v1/embeddings` and `/v1/rerank` from it, and from a decoder's embedding pass.

### Vision-language

A vision-language family is one registry entry whose build function returns a model container: the text decoder,
one or more `encode`-only vision towers and the family's front end. The front end preprocesses the images (and
video frames where the family takes them), runs the tower on each item and renders the conversation through the
checkpoint's own chat template. It returns the prompt's tokens and the request inputs the text model declares, such
as the embedding rows at the image placeholders (see [Multimodal](../serve/multimodal.md)). `poot-llm` keeps only
what is not family-specific: decoding an image's bytes, decoding an animated PNG's frames, and `FrontEndSession`,
which runs a loaded model's front end on one driver per tower. The library path is `ModelHandle::load`,
`ModelHandle::front_end` and `FrontEndSession`, and the `poot caption` command. Check the matrix before relying on
a modality on a backend.

## Adding a model

A new family touches only its own code. The driver, planner, executors and serve code do not change. The
repository checks this with a test (`crates/poot-llm/tests/add_a_model/`) that defines small decoder families
inside the test, outside the model crate, and serves them through the public API: `family.rs` is a decoder
with biased LayerNorm, partial RoPE and one fused QKV tensor read as three row ranges, `parallel.rs` the same
with the parallel residual and a per-head interleaved fused QKV, and `tagged.rs` adds a request input the driver
has never heard of.

A family is a file under `crates/poot-models/src/architectures/` and one row of the `SHIPPED` slice in
`registry.rs`; nothing else names it. A decoder of the standard shape implements `StandardFamily`
(`poot_models::components::standard_family`), which is three things plus a checkpoint to load:

1. **Keys.** The `model_type` values (and GGUF architecture names) that select the family.
2. **A config reader.** `configure` reads the checkpoint's `config.json` once, checks every value and returns the
   typed `StandardParams`: the dimensions, the attention and the feed-forward of each layer (gated, GeGLU,
   plain or routed), how the layer's norms sit around its blocks (pre, post, sandwich or the parallel
   residual), the chat format, the tool-call codec (none for a family without a tool format) and the end-of-sequence
   ids. `StandardFamily` is the public interface. Inside the crate, a dense decoder with the usual HF fields
   uses the crate-private `DenseFamily` shorthand instead: its keys, its defaults and one attention option,
   such as a Q/K norm.
3. **A weight-name table.** `weights` maps each role of the model (embedding, a layer's query rows, a norm
   bias, the head) to a tensor of the checkpoint with `WeightMap::builder`. A fused tensor such as
   `query_key_value` is read whole when its rows interleave per head, or through `WeightView::RowRange`
   views when `q | k | v` are contiguous, so a packed checkpoint stays packed.

The stack, the layers, the `Model` implementation and its input schema are the shared `StandardModel`; the
family writes none of them. A family whose layers are not of this shape (a latent attention, a state-space
mixer, an encoder) implements `Model` itself: `Model::trace_into` is plain Rust over the shared components
(`standard_stack` for the embedding, the layers, the final norm and the head; `standard_layer` for the
residual pair), and `Model::schema` returns a `ModelSchema`: the programs the model offers, the states each
carries, and the request inputs it reads. A family that needs a value no shipped family does (a tag per
token, a per-row tensor) declares it there and traces it as a slot; the driver binds it from the request's
`RequestInputs` with no driver change.

A registry is the shipped families: there is no registration from outside the model crate. A test that serves
a family of its own builds its registry with `Registry::builtin_plus`, which the `test-support` feature
enables, and passes it to `ModelHandle::load`.

The checkpoint directory that `ModelHandle::load` reads holds `config.json`, the safetensors weights and a
`tokenizer.json`: the loader builds the text services from the tokenizer, so a checkpoint without one is
refused. The worked examples write a small word-level tokenizer for their fixtures.

There is no support list to edit. A family runs on a backend when its graph compiles for that backend, and a
refusal is a typed error that names the equation and the missing capability.

### Adding a vision-language model

A vision-language family adds two parts to its container, both built by the same registry entry and both its own
code: the vision towers (each an `encode`-only `Model`, added with `ModelParts::with_tower`) and a front end
(`ModelParts::with_front_end`). The front end implements `FrontEnd::prepare`: it receives the conversation's
turns, the decoded images (or video clips) in the order of the turns, a `Towers` handle that runs a tower on one
row of request inputs and returns its embedding rows, and the model's text codec. It returns the prompt's token
ids and the `RequestInputs` the text model declares. The image resize, patching, chat assembly and position
tables all live in the family's module, beside its tracer; `poot-llm` and `poot-serve` name no family. The test
`crates/poot-llm/tests/add_a_model/vlm.rs` defines such a family outside the model crate and serves
an image request through `FrontEndSession` with no edit to `poot-llm`.

## Next steps

- Library generation: [Run a model (library)](./run-a-model.md)
- HTTP serving: [Serving your first model](../serve/serving-your-first-model.md)
- Backend choice: [Picking a backend](../serve/choosing-a-backend.md)
- Full capability tables: [Feature matrix](../reference/feature-matrix.md)
