---
id: run-a-model
title: Run a model (library)
sidebar_position: 6
---

# Run a model (library)

`poot-llm` loads checkpoints and runs generation from Rust. For the OpenAI-compatible HTTP API, see [Serving](../serve/index.md).

## Checkpoint formats

poot loads **safetensors** (dense plus GPTQ, AWQ and FP8 layouts) and **GGUF** (single-file, config and tokenizer embedded) from a local path. See [Models](./models.md).

## Example: GGUF generation

```bash
cargo run -p poot-llm --release --example gguf_generate -- /path/to/model.gguf
```

Optional prompt as a second argument. The example loads the GGUF through the model registry and generates greedily through the `Driver` on the first device that opens (PTX, then ROCm, then wgpu; `auto` never picks raw Vulkan). It is also the smallest end-to-end GPU decode example.

Other examples under `crates/poot-llm/examples/`:

| Example                                          | Covers                           |
| ------------------------------------------------ | -------------------------------- |
| `trace_and_fuse`, `flash_attention_fusion`       | Graph tracing and fusion         |
| `build_and_eval`                                 | CPU-executor building blocks     |
| `sampling`                                       | Sampling                         |
| `speculation_bench`                              | Speculative policies             |

Encoder embeddings and reranking have a command line, `poot embed`, which opens a device and runs the encoder on it:

```bash
cargo run --release -p poot-llm --features cli --bin poot -- \
  embed similarity /path/to/embedding-checkpoint "a cat" "a kitten"
```

## Command line: poot generate

`poot generate` generates text from a prompt on the device `--backend` names. It takes the flags of llama.cpp's `llama-cli` where poot has the capability:

```bash
cargo run -p poot-llm --release --features cli --bin poot -- \
  generate -m /path/to/checkpoint -p "The capital of France is" -n 16
```

| Flag                                      | Meaning                                                                                                |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `-m`                                      | The checkpoint (a safetensors directory or a `.gguf` file).                                            |
| `-p`, `-f`                                | The prompt, or a prompt file.                                                                          |
| `--system-prompt`                         | A system turn.                                                                                         |
| `--no-chat-template`                      | Continue the text as written. A model without a chat template does this anyway.                        |
| `--chat-template JINJA`, `--chat-template-file FILE` | Render through this jinja template instead of the model's own; a base model then answers turns too. |
| `--conversation`                          | One user turn per line of stdin (`-p`/`-f` is the first turn). K/V is kept between turns. Needs a chat template. |
| `--single-turn`                           | End after one reply.                                                                                   |
| `-n`                                      | New tokens. Default -1: run to the end-of-sequence token, a `-r` string or the end of the window.      |
| `-c`                                      | The context window (default: the model's own). A prompt plus `-n` that does not fit is refused.        |
| `--temp`                                  | Default 0: greedy (`llama-cli` samples at 0.80).                                                       |
| `--top-k`, `--top-p`, `--min-p`, `-s`     | Apply when `--temp` is above 0.                                                                        |
| `--repeat-penalty`                        | Applies at any temperature.                                                                            |
| `-r`                                      | Stop at a string (repeatable).                                                                         |
| `--ignore-eos`                            | Generate through the end-of-sequence token.                                                            |
| `--show-timings`                          | On by default: prefill and decode tokens per second on stderr.                                         |
| `-v`                                      | Log what the run does.                                                                                 |

A model that ships a chat template answers the prompt as a user turn of it and prints the reply.

- Unlike llama-cli there are no `-sys`, `-cnv` or `-st` short forms and no built-in template names.
- A template value with no jinja syntax (`chatml`), an unreadable file or a template that does not compile is refused before the checkpoint is read. A template that fails to render is an error, never a fallback format.
- `--chat-template` and `--chat-template-file` do not combine with each other or with `--no-chat-template`.

`poot` is one binary with three commands, `generate`, `caption` and `embed`, each taking `--backend`.

## Backends

`poot-serve` and the `poot` commands take `--backend wgpu|rocm|ptx|vulkan|auto`; the benchmark runner takes `--backend ptx|rocm|wgpu` (raw Vulkan is not benchmarked). Every backend is in every build, and a backend whose device is absent is refused when it opens. Library code selects a backend by wiring its executor crate (`poot-wgpu-device`, `poot-rocm-device`, `poot-ptx-device` or `poot-vulkan-device`). See [Develop: Backends](./index.md#backends) and [Picking a backend](../serve/choosing-a-backend.md).

:::note
The AMD XDNA2 and Intel NPUs are library devices, not serving backends yet; they are under active development, with prefill as the goal.
:::

## What runs where

Check [Supported models and formats](./supported-models.md) for planner coverage per family and backend before assuming a family works on your target.
