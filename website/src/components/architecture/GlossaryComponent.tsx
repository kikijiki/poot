import React, { useState, useMemo } from "react";
import styles from "./architecture.module.css";

type Entry = {
  term: string;
  def: string;
  see?: string; // link text -> href
};

const ENTRIES: Entry[] = [
  {
    term: "AQL (Architected Queue Language)",
    def: "The packet format AMD HSA runtimes use to submit work to a GPU queue. poot-rocm-runtime records AQL kernel-dispatch packets and replays them for capture/replay decode.",
    see: "backends",
  },
  {
    term: "ash",
    def: "A Rust crate of low-level Vulkan API bindings (unsafe, close to the C API). poot-vulkan-runtime uses ash to record and resubmit command buffers without going through wgpu.",
    see: "backends",
  },
  {
    term: "TensorType (aval)",
    def: "A shape + dtype pair with no data. The graph IR describes every value with a TensorType, stored in ValueMeta.aval, without holding any concrete data. Shape inference (OpKind::infer) operates on TensorTypes.",
    see: "graph-ir",
  },
  {
    term: "Backend",
    def: "A specific GPU target that poot can lower kernels to. First-class backends are wgpu (SPIR-V/Vulkan), ROCm (AMD HSA), and PTX (NVIDIA); raw Vulkan is a fourth, lower-priority one. A backend consists of a codegen target, a runtime, and an executor.",
    see: "backends",
  },
  {
    term: "Block table",
    def: "A host-side per-sequence table that maps logical KV block indices to physical block indices in the paged KV pool. The host flattens it into the Slot::SlotMap graph input (logical position to physical slot), which the paged decode graph uses to gather K/V.",
    see: "attention",
  },
  {
    term: "Bucket",
    def: "A power-of-two range used to discretize a dynamic dimension (sequence length or batch size). Graphs are compiled or captured once per bucket. Amortizes compilation cost over all sequences in that length range.",
    see: "execution",
  },
  {
    term: "Causal mask",
    def: "An additive attention mask (shape [cap]) that is 0 for valid key positions t <= pos and a large negative value for later ones, added to the scores before softmax. It also hides KV cache positions not yet written, enabling a fixed-capacity KV allocation. For the dense families (qwen2 and its relatives, bloom, granite, mpt, smollm3, gemma2) it is a graph computation over Slot::Pos; other families still take it as a host-refilled Slot::Mask input.",
    see: "attention",
  },
  {
    term: "ComputeMeta",
    def: "A Plan variant (Plan::ComputeMeta) that binds a small read-only u32 metadata buffer to an imported kernel: shape and parameter information such as head counts, capacity, and the scale as f32 bits. Allows one kernel binary to serve multiple configurations.",
    see: "fusion",
  },
  {
    term: "Const",
    def: "A weight or table value in the graph IR (Storage::Const). Consts are bound by name from the checkpoint, cached on the device, and never reallocated between tokens. Weights, embedding tables, and normalization weights are all Consts.",
    see: "graph-ir",
  },
  {
    term: "Continuous batching",
    def: "A serving strategy where in-flight requests are combined into one batch per token rather than at request boundaries. When a request finishes, its batch slot is freed immediately. New requests claim free slots. Not served: not available yet (poot is being refactored; planned to return).",
    see: "serving-design",
  },
  {
    term: "Cooperative kernel",
    def: "A kernel launched so that all workgroups are co-resident (cuLaunchCooperativeKernel on PTX, an HSA cooperative queue on ROCm). A persistent-grid kernel relies on this to coordinate across workgroups with event counters.",
    see: "megakernel",
  },
  {
    term: "cudarc",
    def: "The Rust crate poot uses to interface with the NVIDIA CUDA driver API (https://github.com/coreylowman/cudarc). Handles JIT compilation of PTX strings, kernel launches, device memory, and CUDA graph capture/replay.",
    see: "backends",
  },
  {
    term: "CUDA graph",
    def: "A NVIDIA API that records a sequence of GPU dispatches and their parameters, then replays the entire sequence with a single cuGraphLaunch call. Eliminates per-dispatch driver overhead. poot uses CUDA graphs on the PTX backend as the primary mechanism for reducing decode latency.",
    see: "execution",
  },
  {
    term: "Packed dequantization",
    def: "Decoding stored weights according to a PackedWeight descriptor from poot-quant. PackedDequant is the primitive; compiler-only PackedContraction and PackedRowGather decode values while consuming them. Current packed MoE paths still materialize expert tables before indexed matmul.",
    see: "graph-ir",
  },
  {
    term: "Device (storage class)",
    def: "The storage class for ordinary intermediate values (Storage::Device). The planner derives storage requirements; the executor allocates or aliases GPU buffers and retains them for cached execution. Activations use this class.",
    see: "graph-ir",
  },
  {
    term: "CPU oracle",
    def: "The graph reference evaluator in poot-eval: one walk, eval(&graph, &inputs, EvalOptions). Device float results use stated tolerances; packed decode and integer semantics have exact checks.",
    see: "execution",
  },
  {
    term: "Eqn (equation)",
    def: "One node in the graph IR: an OpKind (with its parameters), a list of input Operands (a value or an inline literal), and one output ValueId. The graph is an ordered list of Eqns in topological order.",
    see: "graph-ir",
  },
  {
    term: "Flash attention",
    def: "A family of attention algorithms that avoid materializing the full [L, L] attention score matrix by computing the softmax online. Peak memory is O(L x head_dim) instead of O(L^2). In poot, a compiler pass rewrites the primitive attention subgraph to the fused kernel automatically.",
    see: "attention",
  },
  {
    term: "Fusion",
    def: "A compiler pass that groups primitive eqns into regions that become single kernel dispatches. Reduces the number of dispatches and the number of DRAM round-trips.",
    see: "fusion",
  },
  {
    term: "GBNF",
    def: "GGML/llama.cpp Backus-Naur Form: a grammar syntax used to constrain model output. The HTTP schema retains guided_grammar, but generation serving is not available yet (poot is being refactored; planned to return).",
    see: "serving-design",
  },
  {
    term: "GGUF",
    def: "A file format for storing quantized model weights, used by llama.cpp. poot supports loading GGUF files via poot-load. Supported quantization types include Q8_0, Q4_0, and the common K-quants.",
  },
  {
    term: "Graph IR",
    def: "poot's core intermediate representation. A flat, ordered list of primitive tensor operation equations in SSA (static single assignment) form. Every other subsystem produces or consumes it.",
    see: "graph-ir",
  },
  {
    term: "GQA (grouped-query attention)",
    def: "An attention variant where K and V have fewer heads than Q. Multiple Q heads share one K/V head. poot handles GQA via Broadcast ops over the head axis - no special opcode.",
    see: "attention",
  },
  {
    term: "HSA",
    def: "Heterogeneous System Architecture: AMD's low-level runtime model. poot-rocm-runtime talks HSA directly (hsa-runtime64) rather than HIP/rocBLAS.",
    see: "backends",
  },
  {
    term: "iGPU",
    def: "Integrated GPU - built into the same chip as the CPU, sharing system memory (unified memory architecture). Accessed via the wgpu/Vulkan backend.",
  },
  {
    term: "KIR / kernel IR",
    def: "The kernel intermediate representation in poot-kernel-ir. A tiny CFG-based IR for one kernel function body, between Stable MIR and LLVM IR. Decouples the importer from codegen backends.",
    see: "kernel-import",
  },
  {
    term: "KV cache",
    def: "Device buffers that store keys (K) and values (V) for all previous tokens in a sequence. The attention kernel reads from the KV cache. Without it, K and V would need to be recomputed for every token.",
    see: "attention",
  },
  {
    term: "State (storage class)",
    def: "The storage class for values carried across decode steps, such as the KV cache (Storage::State). State buffers are pre-allocated to capacity, and each state_in binder is paired with a state_out value that the executor writes back into the same buffer.",
    see: "graph-ir",
  },
  {
    term: "LDS (local data store)",
    def: 'GPU hardware shared memory within a workgroup (called "shared memory" in CUDA). In imported poot kernels it is accessed with wg_write and wg_read on a literal array id. Required for flash attention accumulators.',
    see: "kernel-import",
  },
  {
    term: "LLVM IR",
    def: "The intermediate representation used by the LLVM compiler. poot-codegen emits LLVM IR from kernel IR, then lowers it with llc (SPIR-V, PTX) or ROCm clang (AMDGCN).",
    see: "kernel-import",
  },
  {
    term: "LoRA",
    def: "Low-Rank Adaptation (Hu et al., 2021): fine-tune small rank adapters instead of full weights. poot-serve can register, hot-load and unload PEFT-format LoRA adapters; applying one to generation returns when the new scheduling loop lands.",
    see: "serving-design",
  },
  {
    term: "Megakernel",
    def: "A persistent device launch combining work from several kernels. poot has no megakernel today. A future bounded single-step implementation needs compiler lowering and a co-residency proof; an entire token loop needs additional contracts. The required grid-wide synchronization is unavailable on wgpu.",
    see: "megakernel",
  },
  {
    term: "Megatron",
    def: "Megatron-LM style tensor parallelism: shard attention/MLP weights across GPUs with collectives. Tensor parallelism is not available yet (poot is being refactored; planned to return).",
    see: "backends",
  },
  {
    term: "MHA (multi-head attention)",
    def: "Attention with the same number of heads in Q, K, and V. The standard Transformer attention mechanism.",
    see: "attention",
  },
  {
    term: "MLA (multi-head latent attention)",
    def: "DeepSeek-style attention that compresses K/V into a latent cache. Can use different head widths for Q/K vs V; poot keeps unequal-width cases on the primitive decomposition rather than flash fusion.",
    see: "attention",
  },
  {
    term: "MoE (mixture of experts)",
    def: "A model architecture where each token is routed to a subset of expert FFN layers. poot traces routing as ArgTopK plus IndexedMatMul over stacked expert weights. Packed expert paths currently decode and stack those weights into device tables first.",
    see: "tracing",
  },
  {
    term: "Builder / Traced",
    def: "poot_graph_ir::Builder records equations into a Graph. A Traced value is a type-only handle (a ValueId plus a TensorType) returned by each builder call. Model tracers combine them with the composite helpers in poot_graph_ir::ops.",
    see: "tracing",
  },
  {
    term: "MQA (multi-query attention)",
    def: "Extreme GQA where K and V have exactly one head. Multiple Q heads all attend to the same K/V. Same mechanism as GQA - Broadcast over the head axis.",
    see: "attention",
  },
  {
    term: "Online softmax",
    def: "A one-pass algorithm to compute softmax without materializing the full score vector. Maintains a running maximum and denominator while streaming over inputs. Flash attention uses this to avoid the O(L^2) score matrix.",
    see: "attention",
  },
  {
    term: "OpenVINO",
    def: "Intel's model compiler/runtime stack. poot had an experimental Intel NPU prefill offload path that emitted OpenVINO IR from the graph; it was removed.",
    see: "npu",
  },
  {
    term: "OpKind",
    def: "The enum of all operations in the graph IR: primitives (Unary, Binary, Select, Reduce, Broadcast, Cast, Reshape, Transpose, Slice, Concat, Gather, Scatter, MatMul, ...) and compiler-introduced composites (Fused, FusedRow, MatMulBias, FlashAttention*, Rope, packed and dense contractions). No op records a kernel choice; the planner records it with each plan.",
    see: "graph-ir",
  },
  {
    term: "Paged KV cache",
    def: "A KV cache implementation that divides the K/V space into fixed-size blocks from a shared pool. Memory scales with actual sequence length. Enables block sharing across requests with common prefixes.",
    see: "attention",
  },
  {
    term: "Slot",
    def: "A per-step graph input (Storage::Slot) such as Token, Pos, Mask, or SlotMap. Stored in a device buffer at a stable address. The captured graph references these addresses; per-token updates write the new value in place.",
    see: "execution",
  },
  {
    term: "Plan",
    def: "The per-eqn output of planning (poot-graph-plan plan_eqn_analyzed): a compute body (optionally with a metadata buffer), chunked compute, a buffer alias or strided view, or a collective. The executor maps the Plans of a graph to a sequence of device dispatches.",
    see: "fusion",
  },
  {
    term: "pootc",
    def: "A rustc driver that compiles a crate normally, then hooks after_analysis to import every #[kernel] fn's Stable MIR into poot's kernel IR. Output: .kir.json files for each kernel.",
    see: "kernel-import",
  },
  {
    term: "Prefix cache",
    def: "A serving optimization where KV blocks from a previous request are reused for a new request that starts with the same prefix tokens. Reduces prefill cost for common system prompts. Not served: not available yet (poot is being refactored; planned to return).",
    see: "attention",
  },
  {
    term: "PTX (Parallel Thread eXecution)",
    def: "NVIDIA's virtual instruction set architecture. poot lowers kernel IR to PTX via LLVM's nvptx64 backend. The PTX text is JIT-compiled by the CUDA driver to the actual GPU ISA at load time.",
    see: "backends",
  },
  {
    term: "RADV",
    def: "Mesa's open-source Vulkan driver for AMD GPUs. poot's wgpu path is verified against RADV; several SPIR-V subset rules exist because RADV crashed on otherwise-valid patterns.",
    see: "backends",
  },
  {
    term: "Rust kernel subset",
    def: "The subset of Rust accepted by pootc for #[kernel] functions. Covers loops, slice indexing, arithmetic, float intrinsics, LDS arrays, barriers, and atomics. Violations produce a named diagnostic.",
    see: "kernel-import",
  },
  {
    term: "Safetensors",
    def: "A file format for storing model weights, used by Hugging Face. poot-load reads tensor byte ranges into typed weight owners, preserving their stored dtype or packed format.",
  },
  {
    term: "Scatter",
    def: "The inverse of an axis-0 Gather: writes rows of a source into a target through a permutation index. Used in MoE routing. KV cache writes use DynamicUpdateSlice (dense) or ScatterUpdate (paged).",
    see: "graph-ir",
  },
  {
    term: "Speculative decoding",
    def: "A latency optimization where a cheap draft model generates N token proposals, and the full target model verifies all N in one forward pass. Accepted tokens advance the sequence by up to N+1 positions. Not served: not available yet (poot is being refactored; planned to return).",
    see: "serving-design",
  },
  {
    term: "SPIR-V",
    def: "A portable binary intermediate language for GPU kernels, used by Vulkan. poot lowers kernel IR to SPIR-V via LLVM's spirv64 backend and loads the binary via wgpu.",
    see: "backends",
  },
  {
    term: "SSA (static single assignment)",
    def: "A property of the graph IR: each computed ValueId is defined by one Eqn; graph inputs are binders. Makes transforms simple - no ambiguity about which definition a use refers to.",
    see: "graph-ir",
  },
  {
    term: "Stable MIR (rustc_public)",
    def: "A public API to rustc's Mid-level Intermediate Representation (https://github.com/rust-lang/rustc_public). pootc uses it to read #[kernel] function bodies without depending on rustc's unstable internal APIs.",
    see: "kernel-import",
  },
  {
    term: "Structurization",
    def: "A compiler pass that converts the unstructured goto-based CFG of the kernel IR into structured control flow (proper loops, if/else). Required for SPIR-V emit. Applied once per kernel.",
    see: "backends",
  },
  {
    term: "Token",
    def: "A unit of text (word, subword, or character) from the model's vocabulary. In poot, the current token id is a Slot input updated each decode step.",
  },
  {
    term: "Tracing",
    def: "Running a model function once with type-only Traced handles so each builder call records primitive eqns into the graph IR. Traced values carry no data, so model code cannot branch on activation values.",
    see: "tracing",
  },
  {
    term: "ValueId",
    def: "A dense integer index into the graph's value table. Every tensor in the graph is identified by a ValueId. Every Eqn output and every graph input gets a unique ValueId.",
    see: "graph-ir",
  },
  {
    term: "ValueMeta",
    def: "The metadata for a ValueId: its TensorType (aval), storage class (Device, Const, Slot, State, or Computed), and optional name.",
    see: "graph-ir",
  },
  {
    term: "wgpu",
    def: "A Rust crate providing a safe, portable GPU API over Vulkan, Metal, DX12, and WebGPU. poot uses wgpu as the runtime for the SPIR-V/Vulkan backend.",
    see: "backends",
  },
  {
    term: "Workgroup",
    def: 'The unit of GPU parallelism in which threads share LDS and can synchronize with a barrier. Called a "thread block" in CUDA, "workgroup" in Vulkan/SPIR-V. poot kernels are written in terms of workgroups.',
    see: "kernel-import",
  },
  {
    term: "XDNA2",
    def: "AMD's second-generation AI Engine tile array, present in Ryzen AI Max processors. poot retains Peano codegen support (an LLVM fork with aie2p target support), but its NPU runtime and executor were removed.",
    see: "npu",
  },
].sort((a, b) => a.term.localeCompare(b.term));

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

export default function GlossaryComponent(): React.ReactElement {
  const [query, setQuery] = useState("");
  const q = query.toLowerCase();

  const filtered = q
    ? ENTRIES.filter(
        (e) =>
          e.term.toLowerCase().includes(q) || e.def.toLowerCase().includes(q),
      )
    : ENTRIES;

  const byLetter = useMemo(() => {
    const map: Record<string, Entry[]> = {};
    for (const e of filtered) {
      const l = e.term[0].toUpperCase();
      if (!map[l]) map[l] = [];
      map[l].push(e);
    }
    return map;
  }, [filtered]);

  const usedLetters = new Set(Object.keys(byLetter));

  return (
    <div className={styles.glossaryWrap}>
      <input
        className={styles.glossarySearch}
        placeholder="Search terms..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {!q && (
        <div className={styles.glossaryJumps}>
          {ALPHABET.map((l) => (
            <button
              key={l}
              className={`${styles.glossaryJump} ${usedLetters.has(l) ? "" : styles.glossaryJumpDisabled}`}
              onClick={() =>
                usedLetters.has(l) &&
                document
                  .getElementById(`gloss-${l}`)
                  ?.scrollIntoView({ behavior: "smooth" })
              }
            >
              {l}
            </button>
          ))}
        </div>
      )}
      {Object.keys(byLetter)
        .sort()
        .map((letter) => (
          <div
            key={letter}
            className={styles.glossaryGroup}
            id={`gloss-${letter}`}
          >
            <div className={styles.glossaryGroupLetter}>{letter}</div>
            {byLetter[letter].map((entry) => (
              <div key={entry.term} className={styles.glossaryEntry}>
                <span className={styles.glossaryTerm}>{entry.term}</span>
                <span className={styles.glossaryDef}>{entry.def}</span>
                {entry.see && (
                  <a className={styles.glossaryLink} href={`./${entry.see}`}>
                    see {entry.see.replace("-", " ")}
                  </a>
                )}
              </div>
            ))}
          </div>
        ))}
      {filtered.length === 0 && (
        <p style={{ color: "var(--ifm-color-emphasis-500)" }}>
          No terms match.
        </p>
      )}
    </div>
  );
}
