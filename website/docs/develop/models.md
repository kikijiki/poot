---
id: models
title: Supported models
sidebar_position: 7
---

# Supported models

poot traces model forwards into a primitive graph and runs them on its own kernels. This page lists
families by architecture shape. It does not duplicate the full backend tables; use the [feature
matrix](../reference/feature-matrix.md) for what compiles on wgpu, ROCm and PTX.

Checkpoint formats: **safetensors** (including GPTQ, AWQ, and FP8 layouts) and **GGUF** (one file, or a split set
named by any of its `-0000N-of-0000M.gguf` shards, with embedded config and tokenizer). Both work from the library (`poot-llm`). A dense decoder loads as a
`driver::ModelHandle` and generates through the driver, and so do the mixture-of-experts families Qwen3-MoE,
Granite MoE, Mixtral, OLMoE, gpt-oss, MiniMax-M2 and Gemma 4, the MLA families (DeepSeek-V2/V3/V3.2) and the hybrids Nemotron-H, Qwen3-Next, Qwen3.8-Flash-Next and Qwen3.5 (the Qwen3.8-27B-FP8 text tower). The
checkpoint's config decides which: the families that load through the driver are the ones the model registry knows.
`poot-serve` serves the families the registry knows through the same driver.

## Dense decoders

Qwen2/3, Llama, Mistral, Gemma 2/3, Granite, OLMo 2, Phi-3/4, SmolLM3, BLOOM, MPT.

Standard causal LM stack: RMSNorm (or equivalent), RoPE, SwiGLU or GELU MLP, multi-head attention. GGUF and
safetensors are both in routine use, including GPTQ, AWQ and FP8 safetensors, whose quantized projections stay
packed as stored. These load with `ModelHandle::load` and generate with `Driver` on wgpu, ROCm, or PTX; see
the matrix for backend coverage. Gemma 2's sliding window covers the even layers, as in the HF reference.

## Mixture-of-experts (MoE)

Qwen3-MoE, Granite MoE, Mixtral, OLMoE, gpt-oss, MiniMax-M2.

Router plus sparse expert MLPs, with every expert's weights resident unless expert pooling is asked for (`Driver::with_expert_pool`, `--expert-pool` on the server). These load with `ModelHandle::load` and generate with `Driver`, from safetensors and from GGUF. Mixtral is the Llama family with experts: a Mixtral GGUF names the `llama` architecture, and it carries experts when its metadata has `llama.expert_count`. Granite MoE is the Granite family with experts. Experts are read as the checkpoint stores them, per-expert tensors or a stored stack, dense or packed; nothing is fused at load. A NaN or infinite router logit stops the step with a validation error naming the layer. MiniMax-M2 routes with a sigmoid score and a correction bias that only decides which experts run, and it is an HF safetensors family. Its layer, expert and head counts come from the checkpoint's config, so a fine-tune with other counts loads the same way. Check the matrix for your architecture and backend.

## Gemma 4

Dense and mixture-of-experts text models, from an HF `gemma4` checkpoint or a GGUF `gemma4` file. The HF
checkpoint is the multimodal wrapper: the text tower loads from `text_config`, and the vision and audio tower
tensors are left unread. Attention has two kinds of layer. A local layer is sliding-window attention. A global
layer is full attention over wider heads, rotates only a fraction of each head's dimensions, and when the
checkpoint says so reads its values from its key projection. Scores are not scaled, values are normalized per
head with no learned scale, each layer ends in a learned scalar, and the logits are softcapped. In a mixture-of-experts
checkpoint every layer sums a dense feed-forward and a routed block (softmax router, top-k gates renormalized
and scaled per expert, GeGLU experts). These load with `ModelHandle::load` and generate with `Driver`. Per-layer
input embeddings and cross-layer KV sharing are refused with an unsupported-config error. The GGUF tensor names and metadata keys are verified against the dense 31B file only; a mixture-of-experts GGUF's names are not yet verified against a real file.

## MLA and hybrid attention

DeepSeek-V2/V3 (multi-head latent attention), DeepSeek-V3.2 (adds a lightning indexer that selects the keys
each query attends to) and Nemotron-H (Mamba2 mixers, attention and plain MLP layers in a configured order).

These load with `ModelHandle::load` and generate with `Driver`, from safetensors and from GGUF. The latent
attention keeps its compressed latent and rope key as the cache; a GGUF stores the key and value up-projections
split per head, a safetensors checkpoint stores them fused, and both load. DeepSeek-V2 routes with a softmax
and a group limit, DeepSeek-V3 with a biased sigmoid; the routed scaling factor is applied to the selected
weights. Nemotron-H carries each Mamba2 layer's convolution window and state, so a prompt prefilled in chunks
continues exactly. Qwen3-Next (hybrid) carries each gated delta rule (GDN) layer's convolution window and recurrent state next to the attention layers' caches, so a chunked prefill continues exactly; its gated attention, gated delta rule and sigmoid-gated shared expert are components other models can reuse. Qwen3.5 (HF `qwen3_5`) is the same hybrid with a dense gated feed-forward; its BF16 and block-scaled FP8 checkpoints both load, the FP8 projections staying packed as stored. Backend wiring differs by model; the matrix is authoritative.

## Qwen3.8-Flash-Next

An HF `qwen4_exp` checkpoint, read from its `text_config`, or a GGUF `qwen4exp` (the unsloth quants), read from its
metadata. Most layers are gated delta rule (GDN) linear attention; the
rest are gated attention that reads only the blocks of cached positions a small indexer selects (Qwen Sparse
Attention). The residual is several parallel streams gated by Hyper-Connections, the layers the config names
start with a per-layer n-gram embedding, and every layer ends in routed experts with a gated shared expert.
It loads with `ModelHandle::load` and generates with `Driver`. Its layer, expert, head and stream counts come
from the checkpoint's config, so a fine-tune with other counts loads the same way. A prompt is prefilled in whole
blocks of `indexer_compress_ratio` tokens and its tail runs as decode steps. Routed experts are read as the
checkpoint stores them: dense stacks, packed E4M3 tensors per expert, or a GGUF's stacked K-quant, IQ4, IQ2_XS,
IQ3_XXS and IQ3_S tensors kept packed, one owner per expert (UD-Q2_K_XL stores the expert gate and up stacks as
IQ2_XS and IQ3_XXS). A GGUF stores the n-gram hash constants as metadata and the table as one tensor, and poot reads
both as stored. A tensor of a type poot has no format for (for example `Q8_K` or an IQ1 type) is refused by tensor name and
type. The vision tower is not loaded. A token outside the vocabulary stops the step with a validation error naming the
n-gram embedding's layer. The official
checkpoint stores the n-gram table as 128 E4M3 row-range shards under one scale; each shard stays a resident weight
and a lookup reads the shard that owns the row, with exact integer ids. The shard layout is checked on small fixtures on
the CPU oracle and the wgpu, ROCm and PTX executors. The E4M3 shard lookup is gathered bit-exactly against the
reference decode on all three executors on small fixtures; no real checkpoint has run. The indexer keeps a per-row
cache of block keys, so the model generates with `Driver` but `poot-serve` cannot serve it under the paged KV pool
until paged block-key caching lands; startup stops with a typed refusal.

## DeepSeek-V4

DeepSeek-V4, from an HF `deepseek_v4` safetensors checkpoint (the Flash-0731 layout).

The residual stream is several parallel streams per token (`hc_mult`), mixed around the attention block and
the feed-forward block of every layer by learned doubly stochastic mixing weights, and collapsed to one stream
before the closing norm. Each layer's attention is chosen by the config's `compress_ratios`: a sliding window
over a cache of keys that are also the values, plus (for a compressed layer) entries that pool every `rate`
positions into one. Heavily compressed layers attend every complete entry. Sparse layers pool overlapping
windows and keep the best `index_topk` entries per token, chosen by a small indexer. Queries are low rank,
attention has a learned sink per head, the last `qk_rope_head_dim` dimensions of each head rotate (and rotate
back on the output), and the output projection is grouped. Every layer routes through a mixture of experts
scored by `sqrt(softplus(logit))`; the first `num_hash_layers` layers take their experts from a table indexed
by the token id instead. The routed experts (4-bit E2M1) and the projections (E4M3) are read packed, as stored.

This family loads with `ModelHandle::load` and generates with `Driver`. Its prefill step covers whole
compression windows: the driver's prefill granule is the least common multiple of the layers' rates, and the
tokens of a prompt after the last whole window run as decode steps. Layer, expert and head counts, the
compression rates and the schedule come from the checkpoint's config, so a fine-tune with other values loads
the same way. YaRN rope scaling is refused with an unsupported-config error. No real public checkpoint has
run; the matrix says what is verified.

## GLM-5.3-Flash

GLM-5.3-Flash, from an HF `glm5_next` safetensors checkpoint (the multimodal wrapper: the text tower loads
from `text_config`, and the vision tensors are not read). The checkpoint's multi-token-prediction layer loads and
traces as its own program over the target's hidden state; no serving path drafts with it yet.

The residual stream is several parallel streams per token (`hc_mult`), mixed around every block by learned
doubly stochastic weights as in DeepSeek-V4, and collapsed by their mean before the closing norm. The config's
`layer_types` choose each layer's mixer. A linear-attention layer is Kimi Delta Attention: a causal conv over the
query, key and value channels, then a gated delta rule over a matrix state per head with a decay per key
channel. A sparse-attention layer is latent attention without rotary embedding whose queries read only the
positions a pooled indexer keeps: the indexer pools every `index_kpool` cached keys into one, scores the
complete pools, keeps the best `index_topk / index_kpool` of them and always keeps the positions after the last
complete pool. The first layers' feed-forward is dense, the rest a shared expert plus routed experts chosen by a
sigmoid router with a selection-only bias; both use a clamped SwiGLU.

This family loads with `ModelHandle::load` and generates with `Driver`. It has no prefill granule. The Kimi
Delta Attention recurrence is traced token by token, so a prefill step costs graph size in proportion to its
tokens: use a modest prefill chunk. Layer, expert and head counts and the mixer schedule come from the
checkpoint's config, so a fine-tune with other values loads the same way. Indexers other than the full pooled
one, and rotary attention layers, are refused with an unsupported-config error. No real public checkpoint has
run; the matrix says what is verified.

## Embeddings and reranking

BERT-class encoder models for sentence embeddings and cross-encoder reranking.

Safetensors is the usual format for HF-style encoder checkpoints. The `bert` family (an embedding encoder, or
with the classifier head a cross-encoder) loads with `ModelHandle::load` and runs on the driver's `encode` step on
wgpu, ROCm and PTX; `poot_llm::encoders` is the library path. `poot-serve` starts on an encoder
checkpoint and serves `/v1/embeddings` and `/v1/rerank` from it, and from a decoder's embedding pass.

## Vision-language

SmolVLM, idefics3, and Qwen2.5-VL.

Multimodal chat: each of these is one registry entry keyed by the checkpoint's own `model_type` (`idefics3`,
`qwen2_5_vl`) whose build function returns a model container. The container holds the text decoder (the `llama`
family read from the checkpoint's `text_config` for SmolVLM; the `qwen2` body with sectioned RoPE for Qwen2.5-VL),
the vision tower (an `encode`-only model) and the family's front end. The front end preprocesses the images
(SmolVLM splits them into frames; Qwen2.5-VL resizes and patches them, and takes a video as sampled frames),
runs the tower on each item and renders the conversation through the checkpoint's own chat template. It returns
the prompt's tokens and the request inputs the text model declares: the embedding rows at the image
placeholders and, for Qwen2.5-VL, the prompt's sectioned positions (see [Multimodal](../serve/multimodal.md)).
`poot-llm` keeps only what is not family-specific: decoding an image's bytes, decoding an animated PNG's frames,
and `FrontEndSession`, which runs a loaded model's front end on one driver per tower. The library path is
`ModelHandle::load`, `ModelHandle::front_end` and `FrontEndSession`, and the `poot-caption` binary. ROCm and
PTX GPU verification is thinner than wgpu for some modalities; see the matrix before relying on it.

## Adding a model

A new family touches only its own code. The driver, planner, executors and serve code do not change. The
repository checks this with a test (`crates/poot-llm/tests/add_a_model/`) that defines small decoder families
inside the test, outside the model crate, and serves them through the public API: `family.rs` is a gpt-neox
style decoder (biased LayerNorm, partial RoPE, one `query_key_value` tensor read as three row ranges),
`parallel.rs` the same style with the parallel residual and a per-head interleaved fused QKV, and `tagged.rs`
adds a request input the driver has never heard of.

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
