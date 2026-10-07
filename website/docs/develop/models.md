---
id: models
title: Supported models
sidebar_position: 7
---

# Supported models

poot traces model forwards into a primitive graph and runs them on its own kernels. This page lists
families by architecture shape. It does not duplicate the full backend tables; use the [feature
matrix](../reference/feature-matrix.md) for verified wgpu, ROCm, PTX, and NPU coverage.

Checkpoint formats: **safetensors** (including GPTQ, AWQ, and FP8 layouts) and **GGUF** (single file with
embedded config and tokenizer). Both work from the library (`poot-llm`). A dense decoder loads as a
`driver::ModelHandle` and generates through the driver, and so do the mixture-of-experts families Qwen3-MoE,
Granite MoE, Mixtral, OLMoE, gpt-oss, MiniMax-M2 and Gemma 4, the MLA families (DeepSeek-V2/V3/V3.2) and the hybrid Nemotron-H. The
checkpoint's config decides which: the families that load through the driver are the ones the model registry knows.
Generation serving in `poot-serve` is not available yet (poot is being refactored; planned to return). Qwen3-Next is also not available yet (poot is being refactored; planned to return). Both loaders refuse it with an unsupported-model error naming the architecture.

## Dense decoders

Qwen2/3, Llama, Mistral, Gemma 2/3, Granite, OLMo 2, Phi-3/4, SmolLM3, BLOOM, MPT.

Standard causal LM stack: RMSNorm (or equivalent), RoPE, SwiGLU or GELU MLP, multi-head attention. GGUF and
safetensors are both in routine use, including GPTQ, AWQ and FP8 safetensors, whose quantized projections stay
packed as stored. These load with `ModelHandle::load` and generate with `Driver` on wgpu, ROCm, or PTX; see
the matrix for backend coverage. Gemma 2's sliding window covers the even layers, as in the HF reference.

## Mixture-of-experts (MoE)

Qwen3-MoE, Granite MoE, Mixtral, OLMoE, gpt-oss, MiniMax-M2.

Router plus sparse expert MLPs, with every expert's weights resident (expert pooling is not available yet (poot is being refactored; planned to return)). These load with `ModelHandle::load` and generate with `Driver`, from safetensors and from GGUF. Mixtral is the Llama family with experts: a Mixtral GGUF names the `llama` architecture, and it carries experts when its metadata has `llama.expert_count`. Granite MoE is the Granite family with experts. Experts are read as the checkpoint stores them, per-expert tensors or a stored stack, dense or packed; nothing is fused at load. A NaN or infinite router logit stops the step with a validation error naming the layer. MiniMax-M2 routes with a sigmoid score and a correction bias that only decides which experts run, and it is an HF safetensors family. Its layer, expert and head counts come from the checkpoint's config, so a fine-tune with other counts loads the same way. Check the matrix for your architecture and backend.

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
continues exactly. Qwen3-Next (hybrid linear-attention plus MoE) is not available yet (poot is being refactored; planned to return). Backend wiring differs by model; the matrix is authoritative.

## Embeddings and reranking

BERT-class encoder models for `/v1/embeddings` and cross-encoder reranking.

Safetensors is the usual format for HF-style encoder checkpoints. In `poot-serve`, encoders and cross-encoders
run on the CPU evaluator, not a GPU backend.

## Vision-language

SmolVLM, idefics3.

Multimodal chat: vision tower plus text decoder, library path only (the `poot-caption` binary). Vision-language serving is not available yet (poot is being refactored; planned to return). ROCm and
PTX GPU verification is thinner than wgpu for some modalities; see the matrix before relying on it.

## Families that do not load yet

Loading refuses these families with an unsupported-model error that names the architecture string
(`deepseek_v4`, `glm5_next`, `qwen3_5`, `qwen4_exp`):

- **DeepSeek-V4-Flash-0731**
- **GLM-5.3-Flash**
- **Qwen3.8-27B-FP8** (`qwen3_5` text tower)
- **Qwen3.8 QSA** (`qwen4_exp`)

These families are being restored (poot is being refactored; planned to return). Until then no backend runs them, and the cells for them in the
[feature matrix](../reference/feature-matrix.md) are **No (being restored)**.

## Adding a model

A new family touches only its own code. The runner, planner, executors and serve code do not change. The
repository checks this with a test (`crates/poot-llm/tests/add_a_model/`) that defines a small decoder family
inside the test, outside every production crate, and serves it through the public API. Its `family.rs`
is the worked example. A family is four things, plus a checkpoint to load:

1. **A config type.** Read the checkpoint's `config.json` once, check every value, and keep the result in a
   typed struct, so tracing cannot fail on a bad config.
2. **A weight-name table.** Map each role of the model (embedding, a layer's query rows, a norm bias, the head)
   to a tensor of the checkpoint with `WeightMap::builder`. A fused tensor such as `query_key_value` is read
   through three `WeightView::RowRange` views, so a packed checkpoint stays packed.
3. **A tracer.** Implement `Model::trace` as plain Rust over the shared components: `standard_stack` for the
   embedding, the layers, the final norm and the head, and `standard_layer` for the residual pair, with the
   family's own attention and MLP in the closures (the MLP closure receives a `BlockInput`: its `block` is the
   normalized input, and `residual` is the stream it joins).
4. **A registry entry.** A `FamilyEntry` names the `model_type` (or GGUF architecture) that selects the
   family, a build function and a small fixture checkpoint. The build function takes the raw config and the
   weights and returns the `Model`; the model's `ModelConfig` carries what the driver needs to know without
   naming the family: the vocabulary size, the maximum positions, the end-of-sequence ids, the prefill
   granule and the fallback chat format. Register the entry with `Registry::register`, then pass the registry
   to `ModelHandle::load`.

The checkpoint directory that `ModelHandle::load` reads holds `config.json`, the safetensors weights and a
`tokenizer.json`: the loader builds the text services from the tokenizer, so a checkpoint without one is
refused. The worked example writes a small word-level tokenizer for its fixture.

There is no support list to edit. A family runs on a backend when its graph compiles for that backend, and a
refusal is a typed error that names the equation and the missing capability.

## Next steps

- Library generation: [Run a model (library)](./run-a-model.md)
- HTTP serving: [Serving your first model](../serve/serving-your-first-model.md)
- Backend choice: [Picking a backend](../serve/choosing-a-backend.md)
- Full capability tables: [Feature matrix](../reference/feature-matrix.md)
