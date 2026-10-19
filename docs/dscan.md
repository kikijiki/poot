# The recurrence scan contract (GDN delta rule and Mamba2 SSD)

Status: accepted (phase A gate), implemented. Owner card: POOT-634. Prototype and receipts: Backstage
`research/634-scan-contract-prototype/`.

## Problem

Qwen3-Next, Qwen3.8 (Gated DeltaNet) and Nemotron-H (Mamba2 SSD) prefill is a recurrence over the sequence axis
with a carried state. Traced as primitives it is one unrolled chunk body per chunk (about 60 equations a chunk
for GDN; 42006 equations at `L = 4096`, `C = 8`), and a kernel for it would have to pattern-match a graph whose
size grows with `L / C`. The recurrence also has TWO results (`y` and the carried state `s_out`), while `Eqn`,
`OpKind::infer` and `Decomposition` carry exactly one.

## The contract

**One equation, one typed packed result.** `OpKind::Scan(ScanAttrs { family, chunk })` has exactly one result:
the packed tensor `[1, H, L + S, W]`. `poot_graph_ir::ScanLayout` names its two row ranges:

| range                      | rows       | value                               |
| -------------------------- | ---------- | ----------------------------------- |
| `ScanLayout::y_rows()`     | `0..L`     | the sequence output `y [1,H,L,W]`   |
| `ScanLayout::state_rows()` | `L..L + S` | the carried state `s_out [1,H,S,W]` |

`S` is `D` for the delta rule (`W = D`) and `N` for SSD (`W = P`). The layout is derived from the operand types
by `ScanAttrs::layout`, the one function the typing rule, the builder, the decomposition and the planner all
call, so no consumer states a row offset.

`Builder::scan(attrs, operands) -> ScanResult { packed, y, s_out }` emits the one equation and returns the two
projections as `Slice`s of its result at the layout's rows. Result identity is therefore by value: `y` and
`s_out` are two distinct values with two buffers (a projection is a copy), `packed` is their producer, and
liveness is the ordinary single-result liveness. The state is published by pairing `s_out` with its `s_in` in
`Graph::state`; `state_commit` and `buffer_plan` need no change.

No generic tuple or multi-result IR was introduced; a broader result representation would be a separately scoped
card.

**Families and operands** (`ScanFamily::operands`; all F32, `[1, H, ..]`, already at the head count):

- `DeltaRule`: `q k v [L,D]`, `g beta [L,1]` (`g` log-domain), `s_in [D,D]`. `q`/`k` arrive repeated to the value
  head count (the component owns `Tiled` versus `Grouped` head order). The `1/sqrt(D)` query scale is inside.
- `Ssd`: `x [L,P]`, `b c [L,N]`, `delta [L,1]`, `a d [1,1]` (per head), `h_in [N,P]`.

**One semantic definition.** `decompose(Scan)` is the shared primitive chain: `ops::scan_delta_rule_packed` is
`ops::gdn_prefill_chunked` (the pre-existing chain, its triangular tables now computed in the graph) followed by
a concat; `ops::scan_ssd_packed` runs `ops::mamba2_ssd_prefill` per chunk of `chunk` positions, each seeded with
the previous chunk's final state, then the concat (a sequence of at most one chunk is exactly the pre-existing
call). The CPU oracle evaluates the composite through that decomposition, one chunk window at a time (below);
there is no second recurrence evaluator.

**Chunk-by-chunk evaluation.** The traced graph holds one equation however long the sequence is, and the CPU
oracle holds one chunk body at a time: it slices each window of `chunk` positions (`ScanFamily::sequence_operands`),
evaluates that window's decomposition seeded with the previous window's `s_out`, and concatenates the `y` windows
with the final state. The oracle has no sequence-length ceiling and its working set does not grow with `L`.

**Numerics.** The composite is `NumericsProperty::Reassociating`: a device schedule is compared with the
decomposition at tier 2 (ADR-0101), every element of `y` and `s_out`. The GDN decomposition equals the
pre-existing chain bit for bit; SSD chunking matches the token-by-token decode loop and the `h_in`-seeded
whole-sequence prefill within tier 2 (bitwise when the sequence fits one chunk).

## The schedule

`poot-graph-plan/src/planner/scan.rs` plans every scan as launches of one generated streaming kernel
(`poot-kernelgen/src/scan.rs`), recorded as `Generated(Scan)`:

- One workgroup per `(head, column block)`; a lane owns one state column in workgroup memory. Both recurrences
  update a column from that column alone, so lanes share nothing and the body has no barrier.
- The sequence loop is a runtime `while`: the body is the same size for every length (107 instructions for the
  delta rule and 93 for SSD at every `L`; the phase-A prototype unrolled per step was 472, 3440 and 217136
  instructions at `L = 8, 64, 4096`).
- Lanes per workgroup: the widest power of two (at most 64) whose `rows * lanes` floats fit the device's
  workgroup memory. Not even one column fitting is `Capability::ScanLds`, a typed refusal before dispatch.
- One launch covers the whole sequence. The dispatch count is a plan property: a scan is one dispatch.

## Limits and what is not here

- **A chunk-parallel (matmul-form) kernel was built, measured and removed: it is not faster.** POOT-1052 built the
  UT/WY form of both recurrences (per chunk of 16 tokens, pairwise `k . k` / `q . k` tables built cooperatively in
  workgroup memory, in-chunk forward substitution per state column, one pass writing the new column). It matched the
  recurrence and, on wgpu and ROCm, the oracle and the streaming kernel at `L` in 64, 1024, 4096. Timed against
  the streaming kernel at `L = 1024` and `4096` (Strix Halo iGPU; 32 heads `D = 128` delta rule, 64 heads
  `P = 64, N = 128` SSD; schedules alternated in one process) it was 0-10% slower on the delta rule and equal on SSD
  on both wgpu and ROCm: the scan is bound by how few threads (heads times column blocks) fill the device and by
  the `q`/`k` row loads, and the chunk form issues more of them. A kernel no planner rule selects is dead code, so
  it was deleted. The patch, the measurement harness and the receipts are in Backstage
  `research/1052-chunk-parallel-scan-prototype/`; revisit only if a device (PTX has no number) shows a win.
  Streaming time is linear in `L`: 3.9 ms at `L = 64`, 60 ms at `L = 1024` (wgpu, 32 heads, `D = 128`; the
  unrolled primitive chain is 7.2 and 113 ms with 107 and 1577 dispatches).
- There is no per-dispatch work bound: no device runtime can calibrate one, so the scan is never split into token
  ranges and a long sequence is one launch whose serial work grows with it.
- KDA's per-channel gate is not covered: GLM-5.3 has no prefill and keeps its primitive decode.
- The three Gumbel sampling kernels run one 64-lane workgroup per row, and each lane strides the row's
  vocabulary once per pass (2, 32 and 63 passes by rule, counting each bisection step). A row's reduction
  cannot cross launches, so a sampling dispatch is never split, and no work bound refuses it. The greedy pick is
  not one of these: it runs as two dispatches (`ArgmaxChunk` over 1024-logit chunks, then `ArgmaxCombine` per
  row) whose per-lane loops are bounded by `vocab / 1024`.
