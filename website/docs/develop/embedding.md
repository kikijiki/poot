---
id: embedding
title: Embedding poot (LLM)
sidebar_position: 5
---

# Embedding poot (LLM)

This page is the **LLM** API (`poot-llm`): load a checkpoint and generate. For
generic tensor graphs, fusion, GPU executors, and `#[kernel]` dispatch without a model, see
[Compute engine](./compute-engine.md).

HTTP serving is a separate binary (`poot-serve`); this page is in-process generation.

## Depend on `poot-llm`

Inside this workspace, other crates already depend on it. From an external crate, point at the path (or a
git dependency on this repo):

```toml
[dependencies]
poot-llm = { path = "/path/to/poot/crates/poot-llm" }
# Optional AMD path:
# poot-llm = { path = "...", features = ["rocm"] }
```

Build and run your crate under `nix develop` so nightly Rust, `llc` (SPIR-V/NVPTX), and Vulkan match what the GPU
crates expect. Release mode matters for real models:

```bash
cargo run --release
```

## Load a checkpoint

`ModelHandle::load` reads a checkpoint's config, resolves its family through the model registry, and loads
its weights exactly as stored. A directory is a Hugging Face checkpoint (`config.json`, `tokenizer.json`,
`*.safetensors`); a file is a GGUF (config, weights and tokenizer inside).

```rust
use poot_llm::driver::{LoadOptions, ModelHandle};
use poot_models::registry::Registry;

let registry = Registry::builtin()?;
let handle = ModelHandle::load("/path/to/qwen2.5-0.5b".as_ref(), &registry, &LoadOptions::default())?;
let handle = ModelHandle::load("/path/to/model.gguf".as_ref(), &registry, &LoadOptions::default())?;

// GPTQ, AWQ and FP8 projections stay packed as stored; dense BF16/F16 weights stay BF16/F16 words,
// never widened to f32 copies.
let handle = ModelHandle::load("/path/to/gptq-model".as_ref(), &registry, &LoadOptions::default())?;
```

A family the registry does not know is a typed refusal, `DriverError::Unsupported(Unsupported::Registry(..))`.
The dense, mixture-of-experts (Qwen3-MoE, Granite MoE, Mixtral, OLMoE, gpt-oss), MLA (DeepSeek) and hybrid
(Nemotron-H) families are registered, as are the BERT encoders and the SmolVLM vision tower (below).
`LoadOptions::integrity` names a manifest of file lengths and SHA-256 digests the load checks before it reads a tensor.

Useful accessors after load: `handle.config()` (family, vocabulary, positions, end-of-sequence ids),
`handle.text()` (tokenizer and chat template), `handle.store()` (the weights as stored).

## Generate

`Driver::new` runs one model on one device; `Driver::placed` runs it over a `Coordinator` of several (the
`ptx` backend's `open_group`) as a layer pipeline or a tensor-parallel group, each device fitted against its own
memory by `capacity::Planner::new(handle, targets, placement)`. Open the executor with `open_executor` (`Wgpu`, `Rocm` with the
`rocm` feature, `Ptx`, `Vulkan` for raw Vulkan, or `Auto`: PTX if a CUDA device opens, else ROCm, else
wgpu; `Auto` never picks `Vulkan`), then generate:

```rust
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;
use poot_graph_plan::{CompileLimits, CompileOptions, Submission};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest, PreparedSetLimits, open_executor,
};
use poot_llm::{GenerationControl, Sampler};

let handle = Arc::new(handle);
let compile = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
let options = DriverOptions {
    prefill: compile.clone(),
    decode: compile,
    capacity: NonZeroUsize::new(512).unwrap(),      // KV positions: prompt plus new tokens
    prefill_chunk: NonZeroUsize::new(128).unwrap(), // new tokens per prefill step
    max_trace_tokens: NonZeroUsize::new(128).unwrap(),
    prepared: PreparedSetLimits {
        max_entries: NonZeroUsize::new(64).unwrap(),
        max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
    },
    charge: poot_llm::driver::program_retention,
    draft: DraftPolicy::Off, // or PromptLookup / DraftModel: see Speculative decoding
    hidden_window: None,    // Some(window): every program also writes a hidden-state carry
};
let mut driver = Driver::new(Arc::clone(&handle), open_executor(BackendChoice::Auto)?, options)?;
let generation = driver.generate(
    GenerateRequest {
        prompt: handle.text().encode("The capital of France is")?,
        max_new: 32,
        sampler: Sampler::greedy(), // or Sampler::new(temperature, top_k, top_p, seed)
        stops: Vec::new(),
        ignore_eos: false,
    },
    &mut |_id: u32, piece: &str| {
        print!("{piece}");
        GenerationControl::Continue(()) // Break(()) stops after this token
    },
)?;
```

The prompt prefills in chunks, then decode runs one token per step on a recorded entry that every later
step replays. A step shape the family cannot trace, or an operation the device cannot run, is a typed
`DriverError::Unsupported` before any work starts.

## Chat prompts

Instruct models ship a jinja `chat_template`. Render an OpenAI-style messages array with the handle's text
services:

```rust
let rendered = handle.text().render_chat_value(&serde_json::json!([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Say hello in one sentence."}
]), None);
// rendered.prompt is the text to encode; rendered.stops holds the turn-end markers.
```

The second argument accepts tool definitions for templates that support them.

## Tokenize without generating

```rust
let ids = handle.text().encode("hello world")?;
let text = handle.text().decode(&ids)?;
```

## Sentence embeddings and reranking

Embeddings and reranking come from BERT-class models, which run on the driver's `Encode` step on the device
you open. `SentenceEncoder` and `CrossEncoder` load a checkpoint and a backend, tokenize, run the step and
do the host side (pooling, normalization, ranking):

```rust
use std::path::Path;
use poot_llm::driver::{BackendChoice, open_executor};
use poot_llm::encoders::{CrossEncoder, SentenceEncoder};

let mut enc = SentenceEncoder::load(Path::new("/path/to/all-minilm"), open_executor(BackendChoice::Auto)?)?;
let vectors = enc.embed(&["a small domestic cat", "a young kitten"])?; // L2-normalized
let ranked = enc.rerank("query", &["doc a", "doc b"])?;

let mut ce = CrossEncoder::load(Path::new("/path/to/ms-marco-minilm"), open_executor(BackendChoice::Auto)?)?;
let ranked = ce.rerank("query", &["doc a", "doc b"])?;
```

A decoder's embedding pass is the same step: `Driver::encode` over token rows returns the final hidden state of
every position, and the pooling is yours.

## What not to do in-process

- Multi-tenant HTTP, embeddings/rerank endpoints, metrics: use
  [Serving](../serve/index.md).
- Writing GPU kernels: [Authoring kernels](./authoring-kernels.md).
- Backend tradeoffs and capability tables: [Picking a backend](../serve/choosing-a-backend.md) and the
  [feature matrix](../reference/feature-matrix.md).
