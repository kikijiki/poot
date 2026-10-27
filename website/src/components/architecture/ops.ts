// The graph IR op reference; docs/architecture/graph-ir.mdx holds each category's heading and description and renders its ops with OpList.
// An op lists its inputs and its output as named fields: a name, a tensor shape or dtype where it has one, and what the value is in plain words
// (taken from the op's doc in poot-graph-ir's OpKind).

export type Field = {
  name: string;
  shape?: string;
  meaning: string;
};

export type Op = {
  name: string;
  inputs: Field[];
  output: Field;
  notes: string;
};

export type Category = {
  id: string;
  ops: Op[];
};

export const CATEGORIES: Category[] = [
  {
    "id": "elementwise",
    "ops": [
      {
        "name": "Unary(op)",
        "inputs": [
          {
            "name": "x",
            "shape": "[...]",
            "meaning": "the operand, transformed element by element"
          }
        ],
        "output": {
          "name": "out",
          "shape": "[...]",
          "meaning": "same shape and dtype as x"
        },
        "notes": "Neg, Exp, Log, Sqrt, Recip, Tanh, Erf; Not, Clz on I32"
      },
      {
        "name": "Binary(op)",
        "inputs": [
          {
            "name": "a",
            "shape": "[...]",
            "meaning": "left operand"
          },
          {
            "name": "b",
            "shape": "[...]",
            "meaning": "right operand; broadcasts against a"
          }
        ],
        "output": {
          "name": "out",
          "shape": "broadcast([a], [b])",
          "meaning": "the elementwise result, at the broadcast of a and b"
        },
        "notes": "Add, Sub, Mul, Div, Max, Ge (indicator); GeU, And, Or, Xor, Shl, Shr on I32. Numpy broadcast rules"
      },
      {
        "name": "Select",
        "inputs": [
          {
            "name": "cond",
            "shape": "I32 [...]",
            "meaning": "condition; nonzero picks if_true"
          },
          {
            "name": "if_true",
            "meaning": "value taken where cond is nonzero"
          },
          {
            "name": "if_false",
            "meaning": "value taken where cond is zero"
          }
        ],
        "output": {
          "name": "out",
          "shape": "[...] (same)",
          "meaning": "same shape and dtype as the branches"
        },
        "notes": "cond is I32, nonzero picks if_true; branches share one dtype. A native hardware select that moves the chosen element bit for bit, so a NaN or Inf in the unselected branch never reaches the output. I32, F32, F16 and BF16 branches; ROCm and PTX lower all four, while SPIR-V (wgpu, raw Vulkan) refuses F16 and BF16 with a typed DtypeLowering"
      },
      {
        "name": "Cast(to)",
        "inputs": [
          {
            "name": "x",
            "shape": "[...]",
            "meaning": "the tensor to convert"
          }
        ],
        "output": {
          "name": "out",
          "shape": "[...]",
          "meaning": "same shape as x, stored in dtype to"
        },
        "notes": "Storage dtype change; f32 -> bf16 rounds to bf16 precision"
      }
    ]
  },
  {
    "id": "shape",
    "ops": [
      {
        "name": "Broadcast",
        "inputs": [
          {
            "name": "x",
            "shape": "[...]",
            "meaning": "the tensor to expand"
          },
          {
            "name": "shape",
            "meaning": "target shape, attribute: trailing axes align, extent-1 axes expand"
          }
        ],
        "output": {
          "name": "out",
          "shape": "shape",
          "meaning": "x expanded to the target shape"
        },
        "notes": "Expand dims"
      },
      {
        "name": "Reshape",
        "inputs": [
          {
            "name": "x",
            "shape": "[...]",
            "meaning": "the tensor to reinterpret"
          },
          {
            "name": "shape",
            "meaning": "target shape, attribute: same element count as x"
          }
        ],
        "output": {
          "name": "out",
          "shape": "shape",
          "meaning": "the same elements under the new shape"
        },
        "notes": "Same element count; lowers to a buffer alias (no dispatch)"
      },
      {
        "name": "Transpose",
        "inputs": [
          {
            "name": "x",
            "shape": "[...]",
            "meaning": "the tensor to reorder"
          },
          {
            "name": "perm",
            "meaning": "axis permutation, attribute: output axis i is input axis perm[i]"
          }
        ],
        "output": {
          "name": "out",
          "shape": "permuted [...]",
          "meaning": "x with its axes permuted"
        },
        "notes": "Reorder axes; may lower to a strided view"
      },
      {
        "name": "Slice",
        "inputs": [
          {
            "name": "x",
            "shape": "[...]",
            "meaning": "the tensor to cut"
          },
          {
            "name": "axis",
            "meaning": "axis to cut along, attribute"
          },
          {
            "name": "start",
            "meaning": "first index kept, attribute"
          },
          {
            "name": "end",
            "meaning": "one past the last index kept, attribute"
          }
        ],
        "output": {
          "name": "out",
          "shape": "sliced [...]",
          "meaning": "x restricted to start..end along axis"
        },
        "notes": "Contiguous subrange on one axis"
      },
      {
        "name": "Concat",
        "inputs": [
          {
            "name": "xs",
            "shape": "[[...]]",
            "meaning": "the operands; they agree on dtype and every axis but axis"
          },
          {
            "name": "axis",
            "meaning": "axis to join along, attribute"
          }
        ],
        "output": {
          "name": "out",
          "shape": "concatenated [...]",
          "meaning": "the operands joined end to end along axis"
        },
        "notes": "Fusion boundary"
      }
    ]
  },
  {
    "id": "data",
    "ops": [
      {
        "name": "Gather",
        "inputs": [
          {
            "name": "src",
            "shape": "[...]",
            "meaning": "the data to take rows from"
          },
          {
            "name": "idx",
            "shape": "I32",
            "meaning": "index of the row to take; scalar index only"
          },
          {
            "name": "axis",
            "meaning": "axis the row is taken along, attribute"
          }
        ],
        "output": {
          "name": "out",
          "shape": "indexed [...]",
          "meaning": "the selected row, with axis dropped"
        },
        "notes": "Embedding lookup, RoPE table row, paged KV read"
      },
      {
        "name": "ScatterUpdate",
        "inputs": [
          {
            "name": "base",
            "shape": "[P, ...]",
            "meaning": "the pool being written to"
          },
          {
            "name": "src",
            "shape": "[N, ...]",
            "meaning": "rows to write"
          },
          {
            "name": "slots",
            "shape": "[N]",
            "meaning": "pool slot each src row goes to, or -1 to write nothing"
          }
        ],
        "output": {
          "name": "out",
          "shape": "[P, ...]",
          "meaning": "base with the named rows replaced; the rest pass through"
        },
        "notes": "Paged KV write: row t of src goes to pool slot slots[t] (-1 writes nothing), in place on the donated pool; cost follows N, not P"
      },
      {
        "name": "DynamicUpdateSlice",
        "inputs": [
          {
            "name": "operand",
            "meaning": "the tensor to update (the KV cache)"
          },
          {
            "name": "update",
            "meaning": "the block to write"
          },
          {
            "name": "index",
            "meaning": "scalar start position along axis"
          }
        ],
        "output": {
          "name": "out",
          "shape": "same as operand",
          "meaning": "operand with update written at index"
        },
        "notes": "Dense KV-cache slot write at a scalar index"
      },
      {
        "name": "ArgTopK(k)",
        "inputs": [
          {
            "name": "rank",
            "shape": "[..., E]",
            "meaning": "stable descending rank of each score; each row permutes 0..E"
          }
        ],
        "output": {
          "name": "out",
          "shape": "I32 [..., k]",
          "meaning": "per row, the index whose rank is r, for r in 0..k: the top-k expert ids"
        },
        "notes": "Expert ids from a stable rank tensor (MoE routing)"
      },
      {
        "name": "ExpertGroup(experts)",
        "inputs": [
          {
            "name": "ids",
            "shape": "[M]",
            "meaning": "expert id of each row"
          }
        ],
        "output": {
          "name": "out",
          "shape": "I32",
          "meaning": "row 0 order: row at each sorted position; row 1 dest: sorted position of each row"
        },
        "notes": "Rows sorted by expert (stable) and the inverse permutation; an id outside the table sorts last (MoE prefill)"
      }
    ]
  },
  {
    "id": "reduce",
    "ops": [
      {
        "name": "Reduce(op, axis, keepdim)",
        "inputs": [
          {
            "name": "x",
            "shape": "[...]",
            "meaning": "the tensor to reduce"
          }
        ],
        "output": {
          "name": "out",
          "shape": "[...] (axis reduced)",
          "meaning": "x with axis combined by op (extent 1 if keepdim)"
        },
        "notes": "Sum or Max over one axis; keepdim preserves rank"
      }
    ]
  },
  {
    "id": "compute",
    "ops": [
      {
        "name": "MatMul",
        "inputs": [
          {
            "name": "a",
            "shape": "[..., M, K]",
            "meaning": "left matrix; batch dims broadcast"
          },
          {
            "name": "b",
            "shape": "[..., K, N]",
            "meaning": "right matrix"
          }
        ],
        "output": {
          "name": "out",
          "shape": "[..., M, N]",
          "meaning": "batched product, in a's dtype or the F32 tensor-core accumulator"
        },
        "notes": "Batched; B=1 (GEMV) and B>1 (GEMM) are one op; specialization at plan time"
      },
      {
        "name": "IndexedMatMul",
        "inputs": [
          {
            "name": "x",
            "shape": "[M, K]",
            "meaning": "activations, one row per token"
          },
          {
            "name": "W",
            "shape": "[E, K, N]",
            "meaning": "stacked expert weights"
          },
          {
            "name": "idx",
            "shape": "[M]",
            "meaning": "expert id for each row of x"
          }
        ],
        "output": {
          "name": "out",
          "shape": "[M, N]",
          "meaning": "each row contracted against its own expert W[idx[m]]"
        },
        "notes": "Each row contracts against its own expert; no weight copy"
      },
      {
        "name": "PackedDequant",
        "inputs": [
          {
            "name": "sources",
            "meaning": "opaque I8 byte carriers of the packed weight, one per source role"
          }
        ],
        "output": {
          "name": "out",
          "shape": "F32",
          "meaning": "the weight decoded to checkpoint order"
        },
        "notes": "Packed-weight decode primitive; format and shape from poot-quant"
      },
      {
        "name": "PackedContraction / PackedRowGather",
        "inputs": [
          {
            "name": "operand",
            "meaning": "activation to contract, or the row ids to gather"
          },
          {
            "name": "sources",
            "meaning": "opaque I8 carriers of the packed weight"
          }
        ],
        "output": {
          "name": "out",
          "meaning": "activation times the weight, or the gathered rows decoded to F32"
        },
        "notes": "Compiler-only; decode packed values while consuming them"
      },
      {
        "name": "DenseContraction / DenseRowGather",
        "inputs": [
          {
            "name": "weight",
            "meaning": "bf16, f32 or f16 weight in checkpoint [N, K] order, or a bf16 table to gather from"
          }
        ],
        "output": {
          "name": "out",
          "shape": "F32",
          "meaning": "the contraction, or the gathered rows widened to F32"
        },
        "notes": "Compiler-only; contract a weight in checkpoint [N, K] order (bf16, f32, or f16 stored as two-byte words packed in u32) and gather rows of a bf16 table, with no transpose or widened copy"
      },
      {
        "name": "GroupedDenseContraction",
        "inputs": [
          {
            "name": "activation",
            "shape": "[G, M, K] f32",
            "meaning": "one activation block per group"
          },
          {
            "name": "weight",
            "shape": "bf16 [G*N, K]",
            "meaning": "G stacked [N, K] weight blocks in checkpoint order"
          }
        ],
        "output": {
          "name": "out",
          "shape": "[G, M, N] f32",
          "meaning": "each group contracted against its own weight block"
        },
        "notes": "Compiler-only; contract a checkpoint [G*N, K] weight read as G blocks of N rows against a [G, M, K] activation, so a grouped projection keeps its bf16 weight resident with no transpose or widened copy"
      },
      {
        "name": "Rope(rot)",
        "inputs": [
          {
            "name": "x",
            "meaning": "per-head tensor; the first rot elements of the last axis rotate"
          },
          {
            "name": "cos",
            "meaning": "position-selected cosine table, width rot"
          },
          {
            "name": "sin",
            "meaning": "position-selected sine table, width rot"
          }
        ],
        "output": {
          "name": "out",
          "shape": "same as x",
          "meaning": "x with the leading rot elements rotated, the rest unchanged"
        },
        "notes": "Introduced by rope_fusion from the rotate-half chain"
      },
      {
        "name": "Attention",
        "inputs": [
          {
            "name": "q",
            "shape": "[B, Hq, Lq, D]",
            "meaning": "queries"
          },
          {
            "name": "k",
            "shape": "[B, Hkv, Lk, D]",
            "meaning": "keys; Hq/Hkv query heads share each"
          },
          {
            "name": "v",
            "shape": "[B, Hkv, Lk, D]",
            "meaning": "values"
          },
          {
            "name": "mask",
            "shape": "[B, Hm, Lq, Lk]",
            "meaning": "additive mask; Hm is 1 or Hq"
          }
        ],
        "output": {
          "name": "out",
          "shape": "[B, Hq, Lq, D]",
          "meaning": "softmax(q k scaled + mask) times v"
        },
        "notes": "Introduced by the attention rewrite in compile, for decode and prefill; online softmax, no score matrix. The planner generates its kernel"
      },
      {
        "name": "Fused / FusedRow",
        "inputs": [
          {
            "name": "inputs",
            "meaning": "the region's external inputs, in order"
          }
        ],
        "output": {
          "name": "out",
          "meaning": "the value of the region's root"
        },
        "notes": "Pointwise and row-wise (reduction-rooted) fused regions"
      },
      {
        "name": "ContractionEpilogue",
        "inputs": [
          {
            "name": "a, b",
            "meaning": "the head contraction's two operands"
          },
          {
            "name": "extras",
            "meaning": "bias, residual or per-row scale, each broadcast to the output"
          }
        ],
        "output": {
          "name": "out",
          "shape": "head's shape",
          "meaning": "head result passed through the pointwise chain"
        },
        "notes": "Introduced by the epilogue fusion from a dense contraction and the single-use pointwise chain that consumes it. The head's result keeps its dtype (a BF16 head rounds to BF16 before the first step reads it) and every step rounds through it"
      },
      {
        "name": "AllReduce / AllGather",
        "inputs": [
          {
            "name": "x",
            "shape": "[...]",
            "meaning": "this rank's tensor"
          }
        ],
        "output": {
          "name": "out",
          "shape": "same shape",
          "meaning": "AllReduce: partials combined across ranks; AllGather: slices concatenated, axis grows by the world size"
        },
        "notes": "Tensor-parallel collectives. AllGather takes the number of ranks and multiplies the gather axis by it. A collective is a cut between stages: compiling one inside a stage graph is refused, and the reference evaluator refuses an AllGather over more than one rank"
      }
    ]
  }
];
