---
id: embedding
title: Embedding poot (LLM)
sidebar_position: 5
---

# Embedding poot (LLM)

This page is the in-process **LLM** API (`poot-llm`): load a checkpoint and generate. For tensor graphs, fusion, GPU executors and `#[kernel]` dispatch without a model, see [Compute engine](./compute-engine.md). HTTP serving is the separate `poot-serve` binary.

## Depend on `poot-llm`

From an external crate, point at the path (or a git dependency on this repo):

```toml
[dependencies]
poot-llm = { path = "/path/to/poot/crates/poot-llm" }
```

Build and run under `nix develop` so nightly Rust, `llc` (SPIR-V/NVPTX) and Vulkan match what the GPU crates expect. Use release mode for real models:

```bash
cargo run --release
```

## Load a checkpoint

`ModelHandle::load` reads a checkpoint's config, resolves its family through the model registry and loads its weights as stored. A directory is a Hugging Face checkpoint (`config.json`, `tokenizer.json`, `*.safetensors`); a file is a GGUF.

```rust
use poot_llm::driver::{LoadOptions, ModelHandle};
use poot_models::registry::Registry;

let registry = Registry::builtin()?;
let opts = LoadOptions::default();

// a safetensors directory
let handle = ModelHandle::load("/path/to/checkpoint".as_ref(), &registry, &opts)?;
// a GGUF file
let handle = ModelHandle::load("/path/to/model.gguf".as_ref(), &registry, &opts)?;
// a GPTQ, AWQ or FP8 checkpoint
let handle = ModelHandle::load("/path/to/quantized".as_ref(), &registry, &opts)?;
```

GPTQ, AWQ and FP8 projections stay packed as stored. Dense BF16/F16 weights stay BF16/F16 words, never widened to f32 copies.

A family the registry does not know is a typed refusal, `DriverError::Unsupported(Unsupported::Registry(..))`; [Supported models and formats](./supported-models.md#model-families) lists the registered ones.

`LoadOptions::integrity` names a manifest of file lengths and SHA-256 digests. Every file the load reads must be listed: weight files are checked before any tensor is read, and config, tokenizer and chat-template files are hashed once and parsed from those same bytes. An unlisted file is refused.

Accessors after load: `handle.config()` (family, vocabulary, positions, end-of-sequence ids), `handle.text()` (tokenizer and chat template), `handle.store()` (the weights as stored).

## Generate

`Driver::new` runs one model on one device. `Driver::placed` runs it over a `Coordinator` of several (the `ptx` backend's `open_group`) as a layer pipeline or tensor-parallel group, each device fitted against its own memory by `capacity::Planner::new(handle, targets, placement)`.

`open_executor` takes `Wgpu`, `Rocm`, `Ptx`, `Vulkan` (raw Vulkan) or `Auto`: PTX if a CUDA device opens, else ROCm, else wgpu. `Auto` never picks `Vulkan`. Then generate:

```rust
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;
use poot_graph_plan::{CompileLimits, CompileOptions};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest, Layout,
    PreparedSetLimits, open_executor,
};
use poot_llm::{GenerationControl, Sampler};

let handle = Arc::new(handle);
let compile = CompileOptions::new(CompileLimits::STANDARD);
let options = DriverOptions {
    prefill: compile.clone(),
    decode: compile,
    // KV positions: prompt plus new tokens
    capacity: NonZeroUsize::new(512).unwrap(),
    // new tokens per prefill step
    prefill_chunk: NonZeroUsize::new(128).unwrap(),
    // The finite limits every traced step runs under; `max_tokens` is the
    // most new tokens a step may carry. `serving_trace_limits` is the
    // measured profile of this crate's own workloads (see its docs); state
    // your own `TraceLimits` for another workload.
    trace: poot_llm::driver::serving_trace_limits(NonZeroUsize::new(128).unwrap()),
    prepared: PreparedSetLimits {
        max_entries: NonZeroUsize::new(64).unwrap(),
        max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
    },
    charge: poot_llm::driver::program_retention,
    // or PromptLookup / DraftModel / Mtp { k }: see Speculative decoding
    draft: DraftPolicy::Off,
    // Some(window): every program also writes a hidden-state carry;
    // required under Mtp, where draft.carry_window() supplies it
    hidden_window: None,
    // KV cache storage dtype: None takes the model's preference, else f16;
    // Some(KvDtype::F32) is the exact reference
    kv_dtype: None,
};
let executor = open_executor(BackendChoice::Auto)?;
// or `prepare(&shapes)`: the Serving driver is the one that generates
let mut driver = Driver::new(Arc::clone(&handle), executor, options)?
    .serve(Layout::Contiguous);
let generation = driver.generate(
    GenerateRequest {
        prompt: handle.text().encode("The capital of France is")?,
        max_new: 32,
        // or Sampler::new(temperature, top_k, top_p, seed)
        sampler: Sampler::greedy(),
        stops: Vec::new(),
        ignore_eos: false,
    },
    &mut |_id: u32, piece: &str| {
        print!("{piece}");
        // Break(()) stops after this token
        GenerationControl::Continue(())
    },
)?;
```

The prompt prefills in chunks, then decode runs one token per step on a recorded entry that every later step replays. A step shape the family cannot trace, or an operation the device cannot run, is a typed `DriverError::Unsupported` before any work starts.

`prefill_chunk` is the chunk you ask for. `capacity::fit_prefill_chunk` finds the largest chunk the device holds: it plans each candidate (the asked chunk, then halves rounded down to a granule multiple) and takes the first whose footprint fits the device's free memory less a support reserve. A device that cannot report its memory keeps the asked chunk.

When no candidate fits, the error is `AdmitError::NoChunkFits { chunk, refusal }`: the smallest chunk tried and the `CapacityRefusal` that refused it (`OverBudget` names the largest allocations and the shortfall). A caller that runs `generate_turn` on a fitted driver prepares `ServingShapes::turn`, so the fitted footprint is the one the first turn allocates.

## Chat prompts

Instruct models ship a jinja `chat_template`. Render an OpenAI-style messages array with the handle's text
services:

```rust
let rendered = handle.text().render_chat_value(&serde_json::json!([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Say hello in one sentence."}
]), None);
```

- `handle.text().encode_chat(&rendered)` is the prompt as token ids; `rendered.stops` holds the turn-end markers.
- The second argument accepts tool definitions for templates that support them.
- `encode_chat` tokenizes the control tokens a message spells as plain text, so a message that spells a control token cannot open a turn.
- A template that fails to render is an error, not a fallback format.

## Tokenize without generating

```rust
let ids = handle.text().encode("hello world")?;
let text = handle.text().decode(&ids)?;
```

## Sentence embeddings and reranking

`SentenceEncoder` and `CrossEncoder` load an encoder checkpoint and a backend, tokenize, run the `encode` step and do the host side (pooling, normalization, ranking). The load takes the most texts one call encodes. Entries for that many rows and every length up to the model's positions are prepared at load, and a larger call is a typed refusal:

```rust
use std::num::NonZeroUsize;
use std::path::Path;
use poot_llm::driver::{BackendChoice, open_executor};
use poot_llm::encoders::{CrossEncoder, SentenceEncoder};

let rows = NonZeroUsize::new(8).unwrap();
let path = Path::new("/path/to/embedding-checkpoint");
let mut enc = SentenceEncoder::load(path, open_executor(BackendChoice::Auto)?, rows)?;
// L2-normalized
let vectors = enc.embed(&["a small domestic cat", "a young kitten"])?;
let ranked = enc.rerank("query", &["doc a", "doc b"])?;

let path = Path::new("/path/to/reranker-checkpoint");
let mut ce = CrossEncoder::load(path, open_executor(BackendChoice::Auto)?, rows)?;
let ranked = ce.rerank("query", &["doc a", "doc b"])?;
```

A decoder's embedding pass is the same step: call `Driver::prepare_encode` with the rows and longest row you need, then `Driver::encode` returns the final hidden state of every position; pooling is yours. A token-row shape that was not prepared is a typed refusal.

## See also

[Serving](../serve/index.md) for HTTP endpoints and metrics; [Authoring kernels](./authoring-kernels.md) for GPU kernels; [Supported models and formats](./supported-models.md) for coverage.
