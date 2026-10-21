import React, { useState } from "react";
import styles from "./architecture.module.css";

type Op = {
  name: string;
  category: "elementwise" | "shape" | "data" | "reduce" | "compute";
  inputs: string;
  outputShape: string;
  notes: string;
};

const OPS: Op[] = [
  // Elementwise
  {
    name: "Unary(op)",
    category: "elementwise",
    inputs: "x: [...]",
    outputShape: "[...] (same)",
    notes:
      "Neg, Exp, Log, Sqrt, Recip, Tanh, Erf; Not, Clz on I32",
  },
  {
    name: "Binary(op)",
    category: "elementwise",
    inputs: "a: [...], b: [...]",
    outputShape: "broadcast([a], [b])",
    notes:
      "Add, Sub, Mul, Div, Max, Ge (indicator); GeU, And, Or, Xor, Shl, Shr on I32. Numpy broadcast rules",
  },
  {
    name: "Select",
    category: "elementwise",
    inputs: "cond, if_true, if_false",
    outputShape: "[...] (same)",
    notes:
      "cond is I32, nonzero picks if_true; branches share one dtype. A native hardware select that moves the chosen element bit for bit, so a NaN or Inf in the unselected branch never reaches the output. I32, F32, F16 and BF16 branches; ROCm and PTX lower all four, while SPIR-V (wgpu, raw Vulkan) refuses F16 and BF16 with a typed DtypeLowering",
  },
  {
    name: "Cast(to)",
    category: "elementwise",
    inputs: "x: [...]",
    outputShape: "[...] (same)",
    notes: "Storage dtype change; f32 -> bf16 rounds to bf16 precision",
  },
  // Shape
  {
    name: "Broadcast",
    category: "shape",
    inputs: "x: [...], shape",
    outputShape: "shape",
    notes: "Expand dims",
  },
  {
    name: "Reshape",
    category: "shape",
    inputs: "x: [...], shape",
    outputShape: "shape",
    notes: "Same element count; lowers to a buffer alias (no dispatch)",
  },
  {
    name: "Transpose",
    category: "shape",
    inputs: "x: [...], perm",
    outputShape: "permuted [...]",
    notes: "Reorder axes; may lower to a strided view",
  },
  {
    name: "Slice",
    category: "shape",
    inputs: "x: [...], axis, start, end",
    outputShape: "sliced [...]",
    notes: "Contiguous subrange on one axis",
  },
  {
    name: "Concat",
    category: "shape",
    inputs: "xs: [[...]], axis",
    outputShape: "concatenated [...]",
    notes: "Fusion boundary",
  },
  // Data
  {
    name: "Gather",
    category: "data",
    inputs: "src: [...], idx, axis",
    outputShape: "indexed [...]",
    notes: "Embedding lookup, RoPE table row, paged KV read",
  },
  {
    name: "ScatterUpdate",
    category: "data",
    inputs: "base: [P, ...], src: [N, ...], slots: [N]",
    outputShape: "same as base",
    notes: "Paged KV write: row t of src goes to pool slot slots[t] (-1 writes nothing), in place on the donated pool; cost follows N, not P",
  },
  {
    name: "DynamicUpdateSlice",
    category: "data",
    inputs: "operand, update, index",
    outputShape: "same as operand",
    notes: "Dense KV-cache slot write at a scalar index",
  },
  {
    name: "ArgTopK(k)",
    category: "data",
    inputs: "rank: [..., E]",
    outputShape: "[..., k]",
    notes: "Expert ids from a stable rank tensor (MoE routing)",
  },
  // Reduce
  {
    name: "Reduce(op, axis, keepdim)",
    category: "reduce",
    inputs: "x: [...]",
    outputShape: "[...] (axis reduced)",
    notes: "Sum or Max over one axis; keepdim preserves rank",
  },
  // Compute
  {
    name: "MatMul",
    category: "compute",
    inputs: "a: [..., M, K], b: [..., K, N]",
    outputShape: "[..., M, N]",
    notes:
      "Batched; B=1 (GEMV) and B>1 (GEMM) are one op; specialization at plan time",
  },
  {
    name: "IndexedMatMul",
    category: "compute",
    inputs: "x: [M, K], W: [E, K, N], idx: [M]",
    outputShape: "[M, N]",
    notes: "Each row contracts against its own expert; no weight copy",
  },
  {
    name: "PackedDequant",
    category: "compute",
    inputs: "packed sources by descriptor role",
    outputShape: "[out, K]",
    notes: "Packed-weight decode primitive; format and shape from poot-quant",
  },
  {
    name: "PackedContraction / PackedRowGather",
    category: "compute",
    inputs: "activation or row ids, packed sources",
    outputShape: "contraction or selected rows",
    notes: "Compiler-only; decode packed values while consuming them",
  },
  {
    name: "DenseContraction / DenseRowGather",
    category: "compute",
    inputs: "bf16, f32 or f16 weight, or bf16 table",
    outputShape: "f32",
    notes:
      "Compiler-only; contract a weight in checkpoint [N, K] order (bf16, f32, or f16 stored as two-byte words packed in u32) and gather rows of a bf16 table, with no transpose or widened copy",
  },
  {
    name: "GroupedDenseContraction",
    category: "compute",
    inputs: "activation [G, M, K] f32, bf16 weight [G*N, K]",
    outputShape: "[G, M, N] f32",
    notes:
      "Compiler-only; contract a checkpoint [G*N, K] weight read as G blocks of N rows against a [G, M, K] activation, so a grouped projection keeps its bf16 weight resident with no transpose or widened copy",
  },
  // Compiler-introduced
  {
    name: "Rope(rot)",
    category: "compute",
    inputs: "x, cos, sin",
    outputShape: "same as x",
    notes: "Introduced by rope_fusion from the rotate-half chain",
  },
  {
    name: "Attention",
    category: "compute",
    inputs: "q, k, v, mask",
    outputShape: "same as q",
    notes:
      "Introduced by the attention rewrite in compile, for decode and prefill; online softmax, no score matrix. The planner generates its kernel",
  },
  {
    name: "Fused / FusedRow",
    category: "compute",
    inputs: "region inputs",
    outputShape: "region root",
    notes: "Pointwise and row-wise (reduction-rooted) fused regions",
  },
  {
    name: "ContractionEpilogue",
    category: "compute",
    inputs: "head operands (a, b), then the epilogue's extra inputs (bias, residual, ...)",
    outputShape: "as the head contraction",
    notes:
      "Introduced by the epilogue fusion from a dense contraction and the single-use pointwise chain that consumes it. The head's result keeps its dtype (a BF16 head rounds to BF16 before the first step reads it) and every step rounds through it",
  },
  {
    name: "AllReduce / AllGather",
    category: "compute",
    inputs: "x: [...]",
    outputShape: "same shape",
    notes:
      "Tensor-parallel collectives. AllGather takes the number of ranks and multiplies the gather axis by it. A collective is a cut between stages: compiling one inside a stage graph is refused, and the reference evaluator refuses an AllGather over more than one rank",
  },
];

const CAT_LABEL: Record<Op["category"], string> = {
  elementwise: "Elementwise",
  shape: "Shape",
  data: "Data",
  reduce: "Reduce",
  compute: "Compute",
};

const CAT_CLASS: Record<Op["category"], string> = {
  elementwise: styles.catElemwise,
  shape: styles.catShape,
  data: styles.catData,
  reduce: styles.catReduce,
  compute: styles.catCompute,
};

export default function OpTable(): React.ReactElement {
  const [query, setQuery] = useState("");
  const q = query.toLowerCase();
  const filtered = q
    ? OPS.filter(
        (op) =>
          op.name.toLowerCase().includes(q) ||
          op.notes.toLowerCase().includes(q) ||
          op.category.includes(q),
      )
    : OPS;

  return (
    <div className={styles.opTableWrap}>
      <input
        className={styles.opSearch}
        placeholder="Search ops..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <table className={styles.opTable}>
        <thead>
          <tr>
            <th>Op</th>
            <th>Category</th>
            <th>Inputs</th>
            <th>Output shape</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((op) => (
            <tr key={op.name}>
              <td>
                <code>{op.name}</code>
              </td>
              <td>
                <span
                  className={`${styles.opCategory} ${CAT_CLASS[op.category]}`}
                >
                  {CAT_LABEL[op.category]}
                </span>
              </td>
              <td>
                <code style={{ fontSize: "0.75rem" }}>{op.inputs}</code>
              </td>
              <td>
                <code style={{ fontSize: "0.75rem" }}>{op.outputShape}</code>
              </td>
              <td>{op.notes}</td>
            </tr>
          ))}
          {filtered.length === 0 && (
            <tr>
              <td
                colSpan={5}
                style={{
                  textAlign: "center",
                  color: "var(--ifm-color-emphasis-500)",
                  padding: "1rem",
                }}
              >
                No ops match.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
