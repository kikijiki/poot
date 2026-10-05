---
id: compute-engine
title: Compute engine
sidebar_position: 3
---

# Compute engine

poot is a tensor compute stack first. `poot-llm` sits on top of it. If you want matmuls, norms, custom
graphs, or your own `#[kernel]`s without loading an LLM, use the crates below.

There are two entry points:

1. **Graph IR**: build a primitive tensor graph, optionally fuse it, run it on the CPU oracle or on a
   GPU through the executor contract.
2. **Imported kernels**: write ordinary Rust `#[kernel]` functions and dispatch them through
   `poot-runtime` (see [Authoring kernels](./authoring-kernels.md)).

LLM generation uses the same engine: trace a model into a graph, then run that graph with capture/replay.

## Crates

| Crate                            | Role                                                             |
| -------------------------------- | ---------------------------------------------------------------- |
| `poot-graph-ir`                  | `Builder`, `ops::*`, `Graph`                                     |
| `poot-eval`                      | CPU oracle: `eval(&graph, &inputs, EvalOptions) -> Evaluation`   |
| `poot-executor`                  | Executor contract: `Engine<D>` over a per-backend `Device` trait |
| `poot-gpu`                       | wgpu/Vulkan: `WgpuDevice`, the contract's wgpu `Device`          |
| `poot-ptx-gpu` / `poot-rocm-gpu` | NVIDIA / AMD: `PtxDevice` / `RocmDevice` on the same contract    |
| `poot-kernel-attr` + `pootc`     | `#[kernel]` import: `pootc` lowers marked fns to GPU code        |
| `poot-runtime`                   | Host `Context` + typed launchers for imported kernels            |

`poot-graph-plan::compile` owns optimization, legalization and planning. It returns a `Program` for a
typed target. The executor contract consumes that program: `compile_staged` wraps `compile` for one
device, `Executor::add_entry` loads the staged program, the first `step` records it, and every later
`step` replays it.

## Build a graph

High-level ops (`rmsnorm`, `linear`, attention helpers, ...) are compositions of primitives. You never need
an LLM checkpoint to use them:

```rust
use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{Builder, TensorType, ops};
use poot_tensor::HostTensor;

let n = 8usize;
let b = Builder::new();
let x = b.constant("x", TensorType::f32(vec![1, 1, n]));
let w = b.constant("w", TensorType::f32(vec![n]));
let out = ops::rmsnorm(&b, x, w, 1e-6);
let g = b.finish(out);

let mut inputs = HashMap::new();
inputs.insert(x.id, Value::from(HostTensor::f32(vec![1, 1, n], vec![1.0; n])));
inputs.insert(w.id, Value::from(HostTensor::f32(vec![n], vec![1.0; n])));
let y = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))?;
```

`Builder::constant` registers a named const/input binder. At eval time you supply a `HashMap<ValueId,
Value>` keyed by those ids (`Value::from(HostTensor::f32(shape, data))` for an F32 tensor). A bound tensor must have exactly the
dtype the graph declares for that input: the evaluator never narrows or widens at bind, so a BF16 input
is bound from BF16 words (`HostTensor::bf16`). Shapes are static on the graph. `EvalOptions` carries the work budget; `EvalBudget::UNBOUNDED` disables it. The result is
an `Evaluation`; its `.output` is the primary output `Value`.

Try it:

```bash
cargo run -p poot-llm --release --example build_and_eval
```

## Compile for a target

Use the compiler entry instead of assembling a production pass sequence:

```rust
use poot_executor::Device;
use poot_graph_plan::{compile, CompileLimits, CompileOptions, FusionPolicy, Submission, Target};
use poot_gpu::device::WgpuDevice;

let device = WgpuDevice::new()?;
let target: Target = device.target(); // backend plus measured DeviceCaps
let program = compile(&g, &target, &CompileOptions {
    execution: Submission::Replay,
    fusion: FusionPolicy::Full,
    limits: CompileLimits::STANDARD,
})?;
```

`CompileLimits` bounds what one compile may construct: the value and equation counts of every graph a
pass produces, the instruction and local counts of one generated kernel, and the byte size of one
compiled kernel artifact. Every field is finite and nonzero, and there is no unlimited setting;
`CompileLimits::STANDARD` is the production policy. A compile that would pass a limit returns a typed
error before it builds the excess: `CompileError::Expansion` for a graph, a planner refusal carrying
`KernelGenError::BodyLimit` for a kernel body, and a load error for an oversize artifact. The artifact
limit bounds the artifact's bytes only, not the memory or time the external compiler uses.

The program exposes its graph, equation plans, storage/slot decisions and numerics. The executor
contract consumes the staged form: `compile_staged` wraps `compile` for one device and hands the
`StagedProgram` to `Executor::add_entry`. For a CPU-only illustration of individual fusion passes:

```bash
cargo run -p poot-llm --release --example trace_and_fuse
```

## Run on GPU (wgpu)

The wgpu backend runs the same contract as PTX and ROCm: build a `WgpuDevice`, wrap it in an
`Engine`, load the weights, add an entry per graph, then step it.

```rust
use std::sync::Arc;

use poot_executor::{Device, Engine, Executor, HostView, NoSync, StepInputs, WeightSource};
use poot_gpu::device::WgpuDevice;
use poot_graph_plan::{
    compile_staged, CompileLimits, CompileOptions, DeviceId, DevicePlacement, ExpertPlacement,
    FusionPolicy, Partition, Submission, TargetSet,
};
use poot_quant::weights::WeightStore;

// use Builder / ops::* to trace the graph, as above; `store` is a WeightStore
// naming every Storage::Const the graph declares.

let device = WgpuDevice::new()?; // Err if no Vulkan adapter
let target = device.target();

// Stage the graph for one device: compile runs the passes, then planning.
let staged = compile_staged(
    &g.clone().with_validations(Vec::new()),
    &TargetSet::single(DeviceId(0), target),
    &Partition {
        experts: ExpertPlacement::AllResident,
        devices: DevicePlacement::Single(DeviceId(0)),
    },
    &CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: CompileLimits::STANDARD,
    },
)?;

let mut exec = Engine::new(device);
// One load, reused by every entry. This hand-traced graph names each const by its store key, so it
// binds by name; a `Model`'s graph passes `WeightSource::Map(map)` with the model's weight map.
let exe = exec.load_weights(Arc::new(store), WeightSource::ConstNames)?;
let entry = exec.add_entry(exe, &staged)?; // binds consts, loads plans

// Bind this step's slot inputs (consts come from the store through the weight source), then run:
// let mut inputs = StepInputs::new();
// inputs.push(slot_key, &shape, HostView::new(...)?);
let out = exec.step(exe, entry, &StepInputs::new(), &mut NoSync)?.read()?;
```

The first `step` on an entry records the dispatch sequence; every later `step` replays it. Drop the
entry with `Executor::remove_entry` when the graph shape changes, and `Executor::unload` to free the
executable. `Engine<PtxDevice>` and `Engine<RocmDevice>` take the same `Graph` and the same calls;
that is the engine `poot-llm` uses under `generate_kv_*`.

## Direct `#[kernel]` dispatch

When you do not want a graph at all, author a kernel and launch it with a `poot_runtime::Context`:

```rust
use poot_runtime::{Context, KernelBuffer};

let ctx = Context::new()?;
let a = [1.0f32, 2.0, 3.0];
let b = [10.0f32, 20.0, 30.0];
let mut bufs = [
    KernelBuffer::read_only_f32(&a),
    KernelBuffer::read_only_f32(&b),
    KernelBuffer::write_f32(a.len()),
];
// `kernel` is a CompiledKernel for this body and target, produced by poot-codegen.
ctx.dispatch("add", &kernel, [64, 1, 1], [a.len() as u32, 1, 1], &mut bufs)?;
assert_eq!(bufs[2].as_f32(), &[11.0, 22.0, 33.0]);
```

Dispatch requires a `CompiledKernel` with its argument schema, not raw SPIR-V bytes. Prefer the typed
launcher emitted by `#[kernel]`; a low-level caller must pair the exact compiled body, target and
artifact when creating a handle through `poot-codegen`.

Full workflow (attribute, `pootc`, accepted Rust subset): [Authoring kernels](./authoring-kernels.md).

## How this relates to `poot-llm`

| Layer   | What you call                                                                 |
| ------- | ----------------------------------------------------------------------------- |
| Compute | `Builder` / `ops` / `eval` / `Engine<D>` / `#[kernel]`                        |
| Models  | `poot-models` tracers produce a `Graph` from a config                         |
| LLM API | `ModelHandle::load` then `Driver::generate` (loads weights, traces, runs GPU) |
| HTTP    | `poot-serve`                                                                  |

For in-process LLM use, see [Embedding poot](./embedding.md). For IR design depth, see
[Architecture](../architecture/index.mdx) (Graph IR, Fusion, Kernel import, Execution).
