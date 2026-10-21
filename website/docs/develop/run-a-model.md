---
id: run-a-model
title: Run a model (library)
sidebar_position: 6
---

# Run a model (library)

`poot-llm` loads checkpoints and runs generation from Rust. This is the library path; for the
OpenAI-compatible HTTP API, see [Serving](../serve/index.md).

## Checkpoint formats

poot loads **safetensors** (dense plus GPTQ, AWQ, and FP8 quant layouts) and **GGUF** (single-file
checkpoints with embedded config and tokenizer). Point either at a local path on disk.

## Example: GGUF generation

```bash
cargo run -p poot-llm --release --example gguf_generate -- /path/to/model.gguf
```

Optional prompt as a second argument. The example loads the GGUF through the model registry and generates
greedily through the `Driver` on the first device that opens (PTX, then ROCm, then wgpu; `auto` never picks raw Vulkan), so it is also the
smallest end-to-end GPU decode example. Other examples under `crates/poot-llm/examples/` cover graph
tracing and fusion (`trace_and_fuse`, `flash_attention_fusion`), CPU-executor building blocks
(`build_and_eval`), sampling (`sampling`) and speculative policies (`speculation_bench`). Encoder embeddings
and reranking have a command line, `poot embed`
(`cargo run --release -p poot-llm --features cli --bin poot -- embed similarity /path/to/embedding-checkpoint "a cat" "a kitten"`),
which opens a device and runs the encoder on it.

## Command line: poot generate

`poot generate` loads a checkpoint and generates text from a prompt on the device `--backend` names. It takes the
flags of llama.cpp's `llama-cli` where poot has the capability:

```bash
cargo run -p poot-llm --release --features cli --bin poot -- generate -m /path/to/checkpoint -p "The capital of France is" -n 16
```

- `-m` the checkpoint (a safetensors directory or a `.gguf` file); `-p` the prompt or `-f` a prompt file;
  `--system-prompt` a system turn (no `-sys` short form, unlike llama-cli).
- A model that ships a chat template answers the prompt as a user turn of it and prints the reply; `--no-chat-template`,
  or a base model with no template, continues the text as written.
- `--chat-template JINJA` renders through that jinja template instead of the model's own, and `--chat-template-file FILE`
  reads it from a file; a base model then answers turns too. Unlike llama-cli there are no built-in template names: a
  value with no jinja syntax (`chatml`), an unreadable file or a template that does not compile is refused before the
  checkpoint is read, and a template that fails to render a conversation is an error, never a fallback to another format.
  Neither flag combines with `--no-chat-template`, and the two do not combine with each other.
- `--conversation` reads one user turn per line of stdin (`-p`/`-f` is the first turn, answered before stdin is read) and
  keeps the conversation's K/V between turns, so a turn prefills only its new tokens; `--single-turn` ends after one reply.
  It needs a model with a chat template (no `-cnv`/`-st` short forms, unlike llama-cli).
- `-n` is the number of new tokens; the default, -1, runs to the end-of-sequence token, a `-r` string or the end of the
  context window. `-c` sets the window (default: the model's own). A prompt plus `-n` that does not fit the window is refused.
- `--temp` defaults to 0, greedy, poot's exact-token convention (`llama-cli` samples at 0.80). `--top-k`, `--top-p`,
  `--min-p`, `--repeat-penalty` and `-s` apply when `--temp` is above 0 (the repeat penalty also at 0).
- `-r` stops at a string (repeatable) and `--ignore-eos` generates through the end-of-sequence token.
- `--show-timings` (on by default) prints the engine-measured prefill and decode tokens per second to stderr;
  `-v` logs what the run does.

`poot` is one binary with three commands, `generate`, `caption` and `embed`, and every command takes the same
`--backend` flag.

## Backends

`poot-serve` and the `poot` commands take `--backend wgpu|rocm|ptx|vulkan|auto`; the benchmark runner
takes `--backend ptx|rocm|wgpu` (raw Vulkan is not benchmarked). Every backend is in every build: a backend whose device is absent is refused when it
opens. In library code you select the backend by wiring the matching executor crate (`poot-wgpu-device`,
`poot-rocm-device`, `poot-ptx-device`, or `poot-vulkan-device`). The backends are described under
[Develop: Backends](./index.md#backends); see [Picking a backend](../serve/choosing-a-backend.md) for tradeoffs.

:::note
The AMD XDNA2 and Intel NPUs are library devices, not serving backends yet; they are under active development, with prefill as the goal. See
[Develop: Backends](./index.md#backends).
:::

## What runs where

Model families and per-backend verification live in [Supported models](./models.md) and the
[feature matrix](../reference/feature-matrix.md). Use the matrix before assuming a family works on your
target backend. `poot-serve` serves generation, embeddings and reranking on these backends; see [Serving](../serve/index.md).
