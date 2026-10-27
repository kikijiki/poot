---
id: models
title: Models
sidebar_position: 7
---

# Models

poot traces model forwards into a primitive graph and runs them on its own kernels. This page covers what a checkpoint must look like, what the library does with each kind of model, and how to add a family. Which families plan on which backends is in [Supported models and formats](./supported-models.md).

## Checkpoint formats

- **Safetensors**: a Hugging Face directory (`config.json`, `tokenizer.json`, the weight files). Dense, GPTQ, AWQ and FP8 layouts load. Quantized projections stay packed as stored, and dense BF16/F16 weights stay BF16/F16.
- **GGUF**: one file, or a split set named by any of its `-0000N-of-0000M.gguf` shards, with the config and tokenizer embedded. K-quant and IQ tensor types stay packed. A tensor of a type poot has no format for is refused by tensor name and type.

The checkpoint's config decides the family: `ModelHandle::load` resolves it through the model registry. A family the registry does not know is a typed refusal. A config value a family cannot trace (an unsupported scaling mode, per-layer input embeddings) is refused with an unsupported-config error that names it. Layer, expert and head counts come from the config.

## Kinds of model

All kinds load with `ModelHandle::load` and generate with `Driver`.

- **Dense decoders.** A standard causal stack with multi-head or grouped-query attention and optional sliding-window layers.
- **Mixture-of-experts.** A router plus sparse expert MLPs. Experts are read as the checkpoint stores them, dense or packed, and stay resident unless expert pooling is asked for (`Driver::with_expert_pool`, `--expert-pool` on the server). A NaN or infinite router logit stops the step with a validation error naming the layer.
- **Latent attention and hybrid.** Latent-attention models keep their compressed latent and rope key as the KV cache. A hybrid model also carries each recurrent layer's convolution window and state, so a prompt prefilled in chunks continues exactly.
- **Block indexer or compressed attention.** Prefill covers whole blocks or windows: the driver's prefill granule is the least common multiple of the layers' rates, and the tokens after the last whole window run as decode steps. `poot-serve` refuses to start on an indexer that keeps a per-row cache of block keys under the paged KV pool (a typed refusal). A token outside the vocabulary stops a step with a validation error.
- **Multi-stream residuals.** Several parallel residual streams per token. A family whose recurrence is traced token by token makes a prefill step cost graph size in proportion to its tokens: use a modest prefill chunk.
- **Embeddings and reranking.** Encoder models run on the driver's `encode` step; `poot_llm::encoders` is the library path. `poot-serve` serves `/v1/embeddings` and `/v1/rerank` from an encoder checkpoint or a decoder's embedding pass.
- **Vision-language.** See below.

### Vision-language

A vision-language family is one registry entry whose build function returns a container: the text decoder, one or more `encode`-only vision towers and the family's front end. The front end preprocesses images (and video frames where the family takes them), runs the tower on each item, renders the conversation through the checkpoint's chat template, and returns the prompt's tokens and the request inputs the text model declares (see [Multimodal](../serve/multimodal.md)).

The library path is `ModelHandle::load`, `ModelHandle::front_end` and `FrontEndSession`, which runs a loaded front end on one driver per tower, plus the `poot caption` command.

## Adding a model

A new family touches only its own code: the driver, planner, executors and serve code do not change. The tests in `crates/poot-llm/tests/add_a_model/` define families outside the model crate and serve them through the public API:

| Test          | Family                                                                  |
| ------------- | ----------------------------------------------------------------------- |
| `family.rs`   | Biased LayerNorm, partial RoPE, one fused QKV tensor read as three row ranges |
| `parallel.rs` | The same with the parallel residual and a per-head interleaved fused QKV |
| `tagged.rs`   | Adds a request input the driver has never heard of                      |
| `vlm.rs`      | A vision-language family, served through `FrontEndSession`              |

A family is a file under `crates/poot-models/src/architectures/` and one row of the `SHIPPED` slice in `registry.rs`; nothing else names it. There is no registration from outside the model crate and no support list to edit: a family runs on a backend when its graph compiles for it, and a refusal is a typed error naming the equation and the missing capability.

### A standard decoder

A decoder of the standard shape implements `StandardFamily` (`poot_models::components::standard_family`):

1. **Keys.** The `model_type` values (and GGUF architecture names) that select the family.
2. **A config reader.** `configure` reads `config.json` once, checks every value and returns the typed `StandardParams`: the dimensions, each layer's attention and feed-forward (gated, GeGLU, plain or routed), how its norms sit around the blocks (pre, post, sandwich or parallel residual), the chat format, the tool-call codec and the end-of-sequence ids.
3. **A weight-name table.** `weights` maps each role (embedding, a layer's query rows, a norm bias, the head) to a checkpoint tensor with `WeightMap::builder`. A fused tensor such as `query_key_value` is read whole when its rows interleave per head, or through `WeightView::RowRange` views when `q | k | v` are contiguous, so a packed checkpoint stays packed.

The stack, layers, `Model` implementation and input schema are the shared `StandardModel`. A dense decoder with the usual HF fields can use the crate-private `DenseFamily` shorthand instead.

### A family of another shape

A latent attention, state-space mixer or encoder implements `Model` itself:

- `Model::trace_into` is plain Rust over the shared components: `standard_stack` (embedding, layers, final norm, head) and `standard_layer` (the residual pair).
- `Model::schema` returns a `ModelSchema`: the programs the model offers, the states each carries and the request inputs it reads.

A value no shipped family needs (a tag per token, a per-row tensor) is declared in the schema and traced as a slot. The driver binds it from the request's `RequestInputs`.

A test that serves a family of its own builds its registry with `Registry::builtin_plus` (feature `test-support`) and passes it to `ModelHandle::load`. The checkpoint needs a `tokenizer.json`: a checkpoint without one is refused.

### A vision-language family

Add two parts to the container, both built by the same registry entry:

- the vision towers, each an `encode`-only `Model`, with `ModelParts::with_tower`;
- a front end, with `ModelParts::with_front_end`.

The front end implements `FrontEnd::prepare`. It receives the conversation's turns, the decoded images (or video clips) in turn order, a `Towers` handle that runs a tower on one row of request inputs and returns its embedding rows, and the model's text codec. It returns the prompt's token ids and the `RequestInputs` the text model declares. Resize, patching, chat assembly and position tables live in the family's module; `poot-llm` and `poot-serve` name no family.

## Next steps

- Library generation: [Run a model (library)](./run-a-model.md)
- HTTP serving: [Serving your first model](../serve/serving-your-first-model.md)
- Backend choice: [Picking a backend](../serve/choosing-a-backend.md)
- Planner coverage by family and backend: [Supported models and formats](./supported-models.md)
