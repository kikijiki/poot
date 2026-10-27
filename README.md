# poot

> _I-I need to, I need to COMPOOOOT_

poot is an LLM inference engine written from scratch in Rust. A model is an ordinary Rust function that is
traced once into a backend-neutral graph of primitive tensor ops. A shared planner and code generator
compile that graph into fused kernels, and the GPU backends (wgpu, ROCm, PTX, raw Vulkan) replay a captured
decode step once per token. There is no cuBLAS and no cuDNN: every kernel is generated or written in Rust.

**[Documentation](https://kikijiki.github.io/poot/)**

## How it works

![From model code to GPU: graph IR, graph passes, kernel planning, one shared LLVM emitter, four backends](website/static/img/architecture/compilation-pipeline.png)

1. **Model code.** A model family is a config, a weight-name map and a tracer function over shared
   components (`poot-models`). Tracing records equations through `poot_graph_ir::Builder`:

   ```rust
   let h  = ops::rmsnorm(b, x, attn_norm, eps);
   let q  = ops::linear(b, h, q_proj, q_bias);
   let q  = ops::rope(b, q, cos_table, sin_table, pos);
   let o  = ops::attention_masked(b, q, k_cache, v_cache, n_rep, scale, mask);
   let ff = ops::swiglu(b, gate, up);
   ```

2. **Graph IR.** Each helper appends primitive equations (elementwise, reduce, broadcast, matmul, gather,
   view); attention, norms, RoPE and MoE routing are compositions, not opcodes (`poot-graph-ir`). See
   [the graph IR](https://kikijiki.github.io/poot/docs/architecture/graph-ir) and [tracing](https://kikijiki.github.io/poot/docs/architecture/tracing).

   ![A model is a Rust function: the builder records every call into a flat graph IR](website/static/img/architecture/model-tracing.png)

3. **CPU oracle.** `poot-eval` evaluates the same graph tensor by tensor. It is the reference every
   backend is checked against.

4. **Optimization and planning.** `poot-graph-plan` fuses pointwise, reduction and epilogue work, rewrites
   attention to flash attention, binds packed quantized weights, and chooses a kernel schedule for each
   op. See [fusion and planning](https://kikijiki.github.io/poot/docs/architecture/fusion).

5. **Kernels.** `#[kernel]` Rust functions are imported by the `pootc` rustc driver, and `poot-kernelgen`
   generates bodies for fused ops. Both become kernel IR (`poot-kernel-ir`), which `poot-codegen` lowers to
   SPIR-V, PTX or AMDGCN through LLVM. See [kernels from Rust](https://kikijiki.github.io/poot/docs/architecture/kernel-import).

   ![One Rust kernel compiled to SPIR-V, PTX and AMDGCN](website/static/img/architecture/rust-kernels.png)

6. **Executors.** `poot-executor` runs a planned graph on a device: `poot-wgpu-device` (wgpu),
   `poot-rocm-device` (AMD, raw HSA), `poot-ptx-device` (NVIDIA) and `poot-vulkan-device` (raw Vulkan). A decode step is captured once against
   stable buffers and replayed per token: a CUDA graph launch on PTX, resubmitted AQL packets on ROCm, a
   cached command-buffer re-encode on wgpu, a native command-buffer record and replay on raw Vulkan.
   `poot-vulkan-device` implements the executor contract over `poot-vulkan-runtime`, which drives Vulkan directly. See
   [execution](https://kikijiki.github.io/poot/docs/architecture/execution) and [backends](https://kikijiki.github.io/poot/docs/architecture/backends).

   ![Capture the decode step once and replay it every token, one host call per token](website/static/img/architecture/capture-replay.png)

7. **Driver.** `poot-llm` runs generation over an executor: `open` a sequence, `step` a batch of rows,
   `commit` tokens, `release` the sequence. It manages the paged KV cache and the prefix cache.
   `poot-serve` exposes an OpenAI-compatible HTTP server. See
   [serving design](https://kikijiki.github.io/poot/docs/architecture/serving-design).

## Try it

```
nix develop        # pinned Rust nightly, LLVM 22, Vulkan stack
just test          # model-free tests: no GPU, no checkpoint
cargo run -p poot-llm --release --features cli --bin poot -- generate -m /path/to/checkpoint -p "The capital of France is" -n 16
```

`poot generate` takes the flags of llama.cpp's `llama-cli` where poot has the capability.

- **Model and prompt.**
  - `-m` the model: a safetensors directory or a `.gguf` file. Without it the tool reads a default model directory under
    `POOT_MODELS_DIR`, and tests that load checkpoints read them from the same directory.
  - `-p` or `-f` the prompt.
  - `--system-prompt` (no `-sys` short form, unlike llama-cli).
- **Length.** `-n` (default -1: run to the model's end-of-sequence token, a `-r` string or the end of the context
  window `-c`).
- **Sampling.** `--temp` (default 0, greedy), `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `-s`, `-r`,
  `--ignore-eos`.
- **Output.** `--show-timings` (engine-measured prefill and decode tokens/s, on by default) and `-v`.
- **Chat templates.**
  - A model that ships a chat template answers the prompt as a user turn of it; `--no-chat-template` (or a base
    model) continues the text as written.
  - `--chat-template JINJA` or `--chat-template-file FILE` renders through that jinja template instead of the
    model's own (a base model then answers turns too). A bare built-in template name such as `chatml` is refused,
    and neither flag combines with `--no-chat-template`.
- **Conversation.** `--conversation` reads one user turn per line of stdin and keeps the conversation's K/V between
  turns (`-p`/`-f` is the first turn; `--single-turn` ends after one; no `-cnv`/`-st` short forms).
- **Device.** `--backend wgpu|rocm|ptx|vulkan` (default `auto`). wgpu and vulkan need a Vulkan device, PTX an
  NVIDIA GPU, and ROCm an AMD GPU.

### Serving

`poot-serve` exposes an OpenAI-compatible HTTP server over the same engine:

```
cargo run -p poot-serve --release -- /path/to/checkpoint 127.0.0.1:8080
```

See [serving your first model](https://kikijiki.github.io/poot/docs/serve/serving-your-first-model) and
[configuration](https://kikijiki.github.io/poot/docs/serve/configuration) for the endpoints and flags.

## Repository layout

- `crates/` - the workspace, one crate per stage above, plus `poot-load`, `poot-quant` and `poot-weights` (checkpoint
  loading, quantized formats, and the weight roles and map) and `poot-wgpu-runtime*` (device runtimes).
- `benchmarks/` - the cross-framework benchmark suite and its recorded results.
- `crates/poot-orchestrator` - runs the benchmark sweeps on rented RunPod GPUs.
- `website/` - the documentation site (Docusaurus).
- `scripts/` - test-gate and tooling scripts.
