//! The primitive op set and its host-side shape/dtype inference (the `abstract_eval` analog).
//!
//! Attention, RMSNorm, RoPE, and SwiGLU are not here: they are compositions of these primitives (see
//! [`crate::ops`]). Each variant carries its by-value params (the jaxpr eqn `params`); tensor operands are passed
//! separately to `infer`.

use crate::error::ShapeError;
use crate::types::{DType, DTypeClass, DTypeClassExt, Scalar, TensorType, broadcast_shapes};
pub use poot_quant::format::{ScaleEncoding, WeightFormat};
pub use poot_quant::{OperandRole, PackedWeight, SourceRole};

/// Unary primitives. The transcendentals are `Exp`, `Log`, `Sqrt`, `Recip`, `Tanh` and `Erf`; every
/// activation (`silu`, `gelu`, `gelu_erf`, `sigmoid`, `softplus`) is an [`crate::ops`] composition over them,
/// so none has a variant here:
///
/// ```compile_fail,E0599
/// let _ = poot_graph_ir::UnOp::Silu;
/// ```
///
/// ```compile_fail,E0599
/// let _ = poot_graph_ir::UnOp::Gelu;
/// ```
///
/// ```compile_fail,E0599
/// let _ = poot_graph_ir::UnOp::GeluErf;
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum UnOp {
    Neg,
    Exp,
    /// Natural log (a primitive IR op). Needed by softplus (`log(1+exp(x))`, Mamba's Delta) and log-space cumulative
    /// products. NVPTX uses the hardware `lg2.approx` scaled by `ln(2)`, like Exp.
    Log,
    Sqrt,
    Recip,
    /// Round to nearest integer (ties away from zero), `llvm.round.f32` (a primitive IR op).
    Round,
    /// Hyperbolic tangent (a primitive IR op, `libm::tanhf` on the oracle). Devices expand it in the stable
    /// `exp(x - |x|)` form, so `tanh(+-inf)` and `tanh(+-100)` are exactly `+-1`. Logit softcap
    /// ([`crate::ops::softcap`]) and the transcendental half of the activation set build on it.
    ///
    /// Accuracy: the device form `sign(x) * (1 - t) / (1 + t)` cancels in `1 - t` for `|x| << 1`, so its
    /// relative error grows as `|x|` shrinks (a few percent at `|x| = 1e-6`, where the absolute error is
    /// still about `3e-8`). Large `|x|` is exact. Consumers see an absolute error bound near zero, not a
    /// relative one.
    Tanh,
    /// Error function (a primitive IR op, `libm::erff` on the oracle). Devices expand it with an abs-folded
    /// Abramowitz-Stegun 7.1.26 polynomial (about 1.5e-7 max abs error). Exact GELU
    /// ([`crate::ops::gelu_erf`]) is a composition over it.
    Erf,
    /// Bitwise complement of I32 storage bits. Float tensors reject this form; it has no legacy f32 meaning.
    Not,
    /// Leading-zero count of the I32 value interpreted as u32, returning `0..=32`. Float tensors reject this
    /// form.
    Clz,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    /// Elementwise `a >= b -> 1.0` else `0.0` (the comparison/indicator primitive). Output dtype follows the
    /// operands like the arithmetic ops. All other comparisons derive from it: `gt(a,b) = 1 - ge(b,a)`,
    /// `lt(a,b) = gt(b,a)`, `eq(a,b) = ge(a,b) * ge(b,a)`. Enables masking / ReLU (`x * ge(x,0)`) and the on-device
    /// MoE top-k gate (card 061).
    Ge,
    /// Elementwise unsigned comparison of I32 storage bit patterns: `(a as u32) >= (b as u32)`, returning I32 1 or
    /// 0. The integer counterpart needed by multi-limb counters. Invalid for float dtypes; signed I32 comparison
    /// is [`BinOp::Ge`].
    GeU,
    /// Elementwise unsigned remainder of I32 storage bit patterns: `(a as u32) % (b as u32)`, returning the
    /// I32 bit pattern of the u32 result. Invalid for float dtypes. Division by a zero divisor has no
    /// defined value: the CPU oracle rejects it with a typed error, and a graph that can produce a zero
    /// divisor is the caller's bug (hash moduli are fixed nonzero primes).
    RemU,
    /// Bitwise AND of I32 storage bits. Float tensors reject this form.
    And,
    /// Bitwise OR of I32 storage bits. Float tensors reject this form.
    Or,
    /// Bitwise XOR of I32 storage bits. Float tensors reject this form.
    Xor,
    /// Shift left of I32 storage bits. The shift amount is taken modulo 32. Float tensors reject this form.
    Shl,
    /// Logical (zero-fill) shift right of I32 storage bits. The shift amount is taken modulo 32. Float
    /// tensors reject this form.
    Shr,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum RedOp {
    Sum,
    Max,
}

/// Which sampler [`OpKind::SampleToken`] runs, each a wider superset of the last (card 551a, R472-007,
/// deval.md section 8). Every rule maximizes `logits[i]*inv_temp + noise_scale*noise[i]` (two roundings,
/// no fused multiply-add) over the kept index set, ties to the lowest index, after a shared row-max /
/// non-finite pass over the raw logits (R-551a-2). `noise_scale` is 0 for `Greedy` (no operand reads
/// `noise`/`params` at all) and 1 for the Gumbel-family rules (where it rides in `params`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SampleRule {
    /// `token = min { i : logits[i] == max(logits) }`. Operands: `logits` only.
    Greedy,
    /// Temperature + min-p via the Gumbel-max trick. Operands: `logits`, `noise`, `params = [inv_temp,
    /// floor_offset, noise_scale]`. The kept set is `logits[i] >= max(logits) + floor_offset`.
    Gumbel,
    /// [`SampleRule::Gumbel`] plus a top-k value-threshold floor (an exact-integer bisection, tier 1).
    /// Operands: `logits`, `noise`, `params` (as `Gumbel`), `top_k`.
    GumbelTopK,
    /// [`SampleRule::GumbelTopK`] plus a top-p (nucleus) floor (an `exp`-dependent integer-mass
    /// bisection, tier 2 on the noise/membership boundary). Operands: `logits`, `noise`, `params =
    /// [inv_temp, floor_offset, noise_scale, top_p]` (one extra column versus the other rules), `top_k`.
    GumbelTopKTopP,
}

/// A scalar pointwise op inside a fused region (the subset G5a fuses). A composition of these over the region's
/// inputs lowers to one kernel that keeps every intermediate in registers.
#[derive(Clone, Copy, PartialEq, Debug, Hash)]
pub enum FusedOp {
    Unary(UnOp),
    Binary(BinOp),
    /// Pointwise `Select(cond, if_true, if_false)` with three operands.
    Select,
}

/// An operand of a [`FusedStep`]: a local value id, or an inline literal captured from the source graph. Locals
/// `0..n_inputs` are the region's external inputs (matching the fused eqn's `inputs`, all `Operand::Value`, in
/// order); locals `n_inputs + j` are the result of step `j`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FusedOperand {
    Local(usize),
    Lit(Scalar),
    /// Last-axis lane of an external packed I32 input. The input tensor's last dimension is the pack
    /// width; this operand has a trailing unit axis.
    PackLane {
        input: usize,
        lane: usize,
    },
}

/// One straight-line step of a fused region: `local[n_inputs + j] = op(inputs...)`.
#[derive(Clone, PartialEq, Debug)]
pub struct FusedStep {
    pub op: FusedOp,
    pub inputs: Vec<FusedOperand>,
}

/// A fused elementwise region (G5): a straight-line SSA of pointwise steps over `n_inputs`
/// external inputs, produced by `poot-graph-plan`'s `fuse` pass and lowered to one synthesized kernel
/// (`poot_kernelgen::fused`) instead of one kernel per primitive. `output` is the local id of the region's
/// result (a step result, or degenerately an input). Every step's output shares the region's output shape;
/// inputs may broadcast on read.
#[derive(Clone, PartialEq, Debug)]
pub struct FusedRegion {
    pub n_inputs: usize,
    pub steps: Vec<FusedStep>,
    pub output: usize,
    /// When non-empty, the equation output is the last-axis concatenation of these locals, each of
    /// which has a trailing unit axis. Empty means the single [`Self::output`] local.
    pub pack: Vec<usize>,
}

impl FusedRegion {
    /// The output type, threading the input types through the steps. `ins` are the `n_inputs` external input
    /// types, in order. Mirrors how the un-fused chain would infer.
    pub fn infer(&self, ins: &[TensorType]) -> Result<TensorType, ShapeError> {
        if ins.len() != self.n_inputs {
            return Err(ShapeError::Arity {
                expected: self.n_inputs,
                got: ins.len(),
            });
        }
        let mut locals: Vec<TensorType> = ins.to_vec();
        for step in &self.steps {
            let mut in_tys = Vec::with_capacity(step.inputs.len());
            for o in &step.inputs {
                in_tys.push(match o {
                    FusedOperand::Local(id) => locals[*id].clone(),
                    FusedOperand::Lit(s) => s.ty(),
                    FusedOperand::PackLane { input, lane } => pack_lane_ty(&ins[*input], *lane)?,
                });
            }
            let ty = match step.op {
                FusedOp::Unary(u) => OpKind::Unary(u).infer(&in_tys)?,
                FusedOp::Binary(b) => OpKind::Binary(b).infer(&in_tys)?,
                FusedOp::Select => OpKind::Select.infer(&in_tys)?,
            };
            locals.push(ty);
        }
        if self.pack.is_empty() {
            return Ok(locals[self.output].clone());
        }
        let first = locals[self.pack[0]].clone();
        if first.shape.last() != Some(&1) {
            return Err(ShapeError::Concat {
                dim: first.rank().saturating_sub(1),
                a: first.shape.last().copied().unwrap_or(0),
                b: 1,
            });
        }
        for &id in &self.pack {
            if locals[id].dtype != first.dtype || locals[id].shape != first.shape {
                return Err(ShapeError::Concat {
                    dim: first.rank().saturating_sub(1),
                    a: locals[id].numel(),
                    b: first.numel(),
                });
            }
        }
        let mut shape = first.shape;
        let last = shape.len() - 1;
        shape[last] = self.pack.len();
        Ok(TensorType::new(shape, first.dtype))
    }
}

fn pack_lane_ty(input: &TensorType, lane: usize) -> Result<TensorType, ShapeError> {
    let last = input.rank().saturating_sub(1);
    let width = input.shape.last().copied().unwrap_or(0);
    if input.shape.is_empty() || width <= lane {
        return Err(ShapeError::Concat {
            dim: last,
            a: width,
            b: lane + 1,
        });
    }
    let mut shape = input.shape.clone();
    shape[last] = 1;
    Ok(TensorType::new(shape, input.dtype))
}

/// One step of a row-wise fused region ([`RowRegion`]): a pointwise op (a Map over the row, or a Scalar
/// op on per-row reduction results - the synthesizer tells them apart by shape) or a last-axis reduction.
#[derive(Clone, PartialEq, Debug)]
pub enum RowStep {
    Pointwise {
        op: FusedOp,
        inputs: Vec<FusedOperand>,
    },
    /// A `keepdim` reduction over the region's last (row) axis: a row `[.,N]` value -> a scalar `[.,1]`.
    Reduce { op: RedOp, input: FusedOperand },
}

/// A row-wise (reduction-rooted) fused region (G5b): all values are rows `[.,N]` over the reduction axis or
/// per-row scalars `[.,1]`, mixing last-axis reductions with the pointwise ops feeding and consuming them.
/// Lowered to one kernel that processes one row at a time, computing each reduction into a register and
/// recomputing the pointwise chain per pass (one-pass RMSNorm, fused softmax). The pointwise [`FusedRegion`]
/// is the no-reduction case.
#[derive(Clone, PartialEq, Debug)]
pub struct RowRegion {
    pub n_inputs: usize,
    /// the reduction axis (the last axis of the row shape); reductions keep it as size 1.
    pub axis: usize,
    pub steps: Vec<RowStep>,
    pub output: usize,
}

impl RowRegion {
    /// The output type, threading input types through the steps (the un-fused chain's inference).
    pub fn infer(&self, ins: &[TensorType]) -> Result<TensorType, ShapeError> {
        if ins.len() != self.n_inputs {
            return Err(ShapeError::Arity {
                expected: self.n_inputs,
                got: ins.len(),
            });
        }
        let mut locals: Vec<TensorType> = ins.to_vec();
        let ty_of = |o: &FusedOperand, locals: &[TensorType]| match o {
            FusedOperand::Local(id) => locals[*id].clone(),
            FusedOperand::Lit(s) => s.ty(),
            FusedOperand::PackLane { .. } => {
                panic!("PackLane cannot appear in an f32 FusedRow region")
            }
        };
        for step in &self.steps {
            let ty = match step {
                RowStep::Pointwise { op, inputs } => {
                    let in_tys: Vec<TensorType> =
                        inputs.iter().map(|o| ty_of(o, &locals)).collect();
                    match op {
                        FusedOp::Unary(u) => OpKind::Unary(*u).infer(&in_tys)?,
                        FusedOp::Binary(b) => OpKind::Binary(*b).infer(&in_tys)?,
                        FusedOp::Select => OpKind::Select.infer(&in_tys)?,
                    }
                }
                RowStep::Reduce { op, input } => OpKind::Reduce {
                    op: *op,
                    axis: self.axis,
                    keepdim: true,
                }
                .infer(&[ty_of(input, &locals)])?,
            };
            locals.push(ty);
        }
        Ok(locals[self.output].clone())
    }
}

/// The weight dtypes [`OpKind::DenseContraction`] admits, as a table. Card 380 shipped a checkpoint BF16 weight,
/// the Qwen3.8-27B LM head; Card 645 adds F32, so a family written against checkpoint `[out, in]` orientation
/// reads its weight without a device `transpose`; Card 1007 adds F16, whose weight the planner stores packed two
/// elements per `u32` word for the same generated bodies to decode. A further dtype is a row here plus a kernel that
/// reads it, not another `OpKind` variant or hand-written dispatch branch.
///
/// `infer` is the only gate, so an unadmitted dtype never reaches a planner or kernel.
///
/// A tracer cannot build one. [`OpKind::DenseContraction`] is compiler-only, so `Builder` exposes no method for
/// it, enforced by the type system:
///
/// ```compile_fail,E0599
/// use poot_graph_ir::builder::Builder;
///
/// let b = Builder::new();
/// let _ = b.dense_contraction();
/// ```
pub const DENSE_CONTRACTION_WEIGHT_DTYPES: &[DType] = &[DType::BF16, DType::F32, DType::F16];

/// The table source dtypes [`OpKind::DenseRowGather`] admits, as a table. Card 381: a checkpoint BF16 embedding
/// table (Qwen3.8-27B token embedding). Card 405: a checkpoint E4M3FN embedding table (Qwen4Exp PLE, card 363),
/// decoded with `poot_quant::scalar::e4m3fn_to_f32`. A further narrow float is a row here plus a kernel that
/// decodes it, not another `OpKind` variant or hand-written dispatch branch.
///
/// `infer` is the only gate, so an unadmitted dtype never reaches a planner or kernel.
///
/// A tracer cannot build one. [`OpKind::DenseRowGather`] is compiler-only, so `Builder` exposes no method for
/// it, enforced by the type system:
///
/// ```compile_fail,E0599
/// use poot_graph_ir::builder::Builder;
///
/// let b = Builder::new();
/// let _ = b.dense_row_gather();
/// ```
pub const DENSE_ROW_GATHER_SOURCE_DTYPES: &[DType] = &[DType::BF16, DType::E4M3FN];

/// Whether an [`OpKind`] is defined directly or by its decomposition into primitives (see
/// [`OpKind::class`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpClass {
    /// The oracle and every lowering define it directly.
    Primitive,
    /// Defined by [`crate::decompose::decompose`]: the oracle evaluates that decomposition. Today a
    /// `compile` pass forms every composite from the primitive chain its decomposition restates, and
    /// `Builder` has no method that emits one.
    Composite,
}

/// Orthogonal tensor algebra. The whole transformer composes from these (graph-architecture.md 3.2).
#[derive(Clone, PartialEq, Debug)]
pub enum OpKind {
    Unary(UnOp),
    Binary(BinOp),
    /// Pointwise I32 select: `out = if_false + cond * (if_true - if_false)` with wrapping arithmetic.
    /// Inputs are `cond`, `if_true`, `if_false`. Float tensors reject this form.
    Select,
    Reduce {
        op: RedOp,
        axis: usize,
        keepdim: bool,
    },
    Broadcast {
        shape: Vec<usize>,
    },
    /// Cast a tensor to another storage dtype (same shape). f32 -> bf16 rounds to bf16 precision (bf16 is the top
    /// 16 bits of an f32); a widening/same-dtype cast is the identity. Feeds bf16 operands into the tensor-core
    /// matmul path (card 045).
    Cast {
        to: DType,
    },
    Reshape {
        shape: Vec<usize>,
    },
    Transpose {
        perm: Vec<usize>,
    },
    Slice {
        axis: usize,
        start: usize,
        end: usize,
    },
    Concat {
        axis: usize,
    },
    /// A nullary range `[0, 1, .., len)`, shape `[len]`, in F32: the explicit iota primitive that
    /// replaces the reserved named per-expert range graph constants (R466-019). Ranges are computed from
    /// this equation, not bound by a checkpoint binder. The compiler folds every `Iota` into a
    /// [`crate::Storage::Computed`] graph constant ([`crate::transform::fold_iota`]) before planning, so
    /// the planner never lowers an iota kernel; an unfolded `Iota` at the planner is a typed refusal.
    /// The dtype stays F32 until Card 558b switches indices and ranges to I32 end to end.
    Iota {
        len: usize,
    },
    /// Gather rows of `data` (input 0) along `axis` by an index (input 1). Scalar-index case only for now
    /// (embedding lookup, RoPE cos/sin row): drops `axis` from the result.
    Gather {
        axis: usize,
    },
    /// Scatter rows of `src` (input 0) along `axis` by a `[N]` permutation index (input 1), the inverse of an
    /// axis-0 [`OpKind::Gather`]: `out[index[j], ..] = src[j, ..]`, output the same shape as `src`. `index` must be a
    /// permutation of `0..N` (N = `src.shape[axis]`); used for argtop-k expert indices in sparse MoE dispatch
    /// (card 034: scatter `iota` by the per-expert rank to invert it). Axis 0 only.
    Scatter {
        axis: usize,
    },
    /// Scatter-update rows into a base (card 059, the multi-token paged KV write): `out[p, ..] =
    /// inv[p] >= 0 ? src[inv[p], ..] : base[p, ..]`. Inputs are `base [POOL, ..rest]`, `src [N, ..rest]`, and the
    /// `[POOL]` inverse map `inv` (input 2, f32: the src row that writes slot `p`, or `-1` to keep `base[p]`).
    /// Output the same shape as `base`. Unlike [`OpKind::Scatter`] (a permutation, every row written), this writes the `N`
    /// mapped rows and passes the rest through: the scattered analog of [`OpKind::DynamicUpdateSlice`] a paged prefill's N
    /// tokens need (kernel `scatter_update_dt`).
    ScatterUpdate,
    /// Batched matmul A[..,M,K] x B[..,K,N] -> [..,M,N], batch dims broadcast (numpy semantics).
    MatMul,
    /// Decode one validated rank-2 packed payload into checkpoint-order `[out,K]` f32.
    /// Its `descriptor.sources()` I8 operands are opaque const byte carriers, not numeric tensors,
    /// one per named [`poot_quant::SourceRole`] in that order (no positional slots): a GGUF block
    /// format has one (`Blocks`), a registered E4M3/E2M1 cell has two (`Codes`, `Scale`), and GPTQ or
    /// AWQ have three or four (`Codes`, `Zero`, `Scale`, and `GroupIndex` for GPTQ act-order).
    PackedDequant {
        descriptor: PackedWeight,
    },
    /// Compiler-only replacement for activation matmul against a packed weight. Builders do not expose it; Card 356
    /// lowers it to an imported kernel.
    ///
    /// `blocks` partitions the weight's `out` axis into that many contiguous, equal-width blocks and restricts each
    /// activation block to its own weight block (Card 385): `out[b, i, j] = sum over k of
    /// activation[b, i, k] * decode(weight[b * (out / blocks) + j, k])`, `b` ranging over `blocks`. `blocks == 1` is
    /// the ordinary dense contraction (the activation may be any rank ending in `K`); `blocks > 1` requires a
    /// rank-3 `[blocks, M, K]` activation and produces `[blocks, M, out / blocks]`. One field, not a second
    /// variant, because both are the same index map; see [`crate::ops::packed_linear`] (`blocks == 1`) and
    /// [`crate::ops::packed_block_diagonal_linear`] (`blocks > 1`).
    PackedContraction {
        descriptor: PackedWeight,
        blocks: usize,
    },
    /// Compiler-only replacement for an axis-0 `Gather` of a packed weight's rows (card 642/545a,
    /// quantized GGUF token embeddings): `out[.., :] = decode(ids[..], :)`. Inputs are the
    /// weight's `descriptor.sources()` I8 carriers, in that order, then the row ids (F32 or I32, any
    /// shape); the output is `ids.shape ++ [K]` F32. Builders do not expose it: the claim pass rewrites
    /// `Gather { axis: 0 }(PackedDequant(..), ids)` into it, so no `[out, K]` table is ever materialized.
    PackedRowGather {
        descriptor: PackedWeight,
    },
    /// Compiler-only replacement for an activation matmul against a dense weight (BF16, F32 or F16) held in checkpoint
    /// `[N, K]` order: `out[.., m, n] = sum over k of activation[.., m, k] * decode(weight[n, k])`. Inputs are the
    /// activation `[.., M, K]` (F32) and the weight `[N, K]` (`weight` dtype); the output is `[.., M, N]` F32.
    ///
    /// Card 380. Builders do not expose it; like [`OpKind::PackedContraction`], it exists so a recognizer can fold
    /// `MatMul(a, Transpose([1, 0], w))` into one equation that reads the weight in checkpoint order
    /// (`transform::fold_dense_contractions`). Without the fold, lowering the Qwen3.8-27B LM head needs a
    /// transposed copy of a `[vocab, hidden]` matrix or widening it to F32, and the epic forbids both.
    ///
    /// `weight` is an admitted-dtype table: [`OpKind::infer`] rejects anything outside
    /// [`DENSE_CONTRACTION_WEIGHT_DTYPES`] (BF16, F32 and F16).
    DenseContraction {
        weight: DType,
    },
    /// Gather rows from a dense narrow-float table along axis 0 and widen them to F32 in one equation:
    /// `out[s.., r] = decode(table[index[s..], r])`. Inputs are the table `[V, R]` (`source` dtype) and the index
    /// (`I32`, any shape `S`, including a scalar); the output is `S ++ [R]` F32.
    ///
    /// Card 381, the sibling of [`OpKind::DenseContraction`]. Builders do not expose it: it exists so a recognizer
    /// can fold `Cast(F32, Gather { axis: 0 })` into one equation (`transform::fold_dense_bf16_row_gathers`).
    /// Without the fold, lowering the Qwen3.8-27B token embedding on wgpu means widening a `[vocab, hidden]` table
    /// to F32 and casting it back to BF16, and the epic forbids both copies.
    ///
    /// The gather and the widening are one equation on purpose: splitting them would leave a BF16 value between
    /// them, which wgpu can only represent as packed `u32` lanes, so every kernel producing one would write two
    /// elements per lane.
    ///
    /// `source` is an admitted-dtype table: [`OpKind::infer`] rejects anything outside
    /// [`DENSE_ROW_GATHER_SOURCE_DTYPES`].
    DenseRowGather {
        source: DType,
    },
    /// MatMul with a fused bias epilogue: `(A@B)[..,m,n] + bias[n]`, `bias` (input 2) an `[N]` vector. Same
    /// `[..,M,N]` output as `MatMul`; folds `linear`'s post-matmul bias add into the matmul kernel (one dispatch,
    /// not two). One definition (the eager eval is matmul-then-add; the kernel is `matmul_batched_bias_dt`).
    /// Compiler-only (Card 557): `ops::linear` traces `matmul` then a broadcast `add`, and `compile`'s one
    /// contraction-epilogue fuse rule forms this op.
    MatMulBias,
    /// Batched argtop-k index extraction (spec 136, grouped-MoE prefill routing): `rank [..,E] -> [..,k]` expert
    /// ids. `out[..,r]` is the index `i` along the last axis with `rank[..,i] == r`, for `r` in `0..k`. One shared
    /// `[..,E]` stable rank tensor feeds this directly, with no per-row host loop. Output dtype is F32 (required):
    /// `IndexedMatMul` reads `idx.data[m] as usize` from the f32 lane, and an `I32` tensor's codes live in the
    /// separate `.ints` lane, which would read back as zeros. Precondition: each last-axis `rank` slice is a
    /// permutation of `0..E`; [`crate::ops::stable_descending_rank`] guarantees that even for tied scores. `k` is a
    /// trace-time constant (static shape, capture-safe). This op inverts rank and has no score-order or tie-break
    /// semantic of its own.
    ArgTopK {
        k: usize,
    },
    /// Indexed matmul (card 088, sparse-MoE gather-free GEMM): `out[m,n] = sum_k x[m,k] * W[idx[m],k,n]`. Inputs are
    /// the activation `x [M,K]`, a stacked weight `W [E,K,N]`, and per-row expert ids `idx [M]` (f32). Each row `m`
    /// contracts against its own expert `W[idx[m]]` with no prior copy of the selected experts' weights (that copy
    /// made `moe_sparse` slower than dense). Output `[M,N]`, dtype = x's. One definition (the eager eval gathers
    /// per-row; the kernel is `indexed_matmul_dt`, reading the selected expert's rows in-kernel).
    IndexedMatMul,
    /// Write `update` (input 1) into a copy of `operand` (input 0) starting at the scalar index `index` (input 2)
    /// along `axis`, leaving the rest unchanged (XLA DynamicUpdateSlice, single axis). The KV-cache slot write:
    /// `operand[.., index..index+update.shape[axis], ..] = update`. Result type == `operand`'s type. The index is
    /// typically an inline literal (the decode graph is pos-specialized).
    DynamicUpdateSlice {
        axis: usize,
    },
    /// Pack int8 codes (f32 in `[-127, 127]`) into i32 words, 4 codes per word along the last axis (spec 048): output
    /// dtype `I32`, last dim -> `ceil(L/4)`. wgpu has no 8-bit buffer, so quantized KV is stored as i32 words (the
    /// GPTQ-weight pattern), a 4x reduction. Little-endian bytes; a partial tail word zero-pads. Paired with
    /// [`OpKind::UnpackI8`].
    PackI8,
    /// Unpack int8 codes from i32 words (inverse of [`OpKind::PackI8`]): output dtype `F32`, last dim -> `len` (the original
    /// code count, which the word count alone does not determine). Each byte is read as a signed i8 widened to f32.
    UnpackI8 {
        len: usize,
    },
    /// A fused elementwise region (G5): a composition of pointwise steps lowered to one kernel. The eqn's
    /// `inputs` are the region's external inputs in order; see [`FusedRegion`].
    Fused(FusedRegion),
    /// A row-wise (reduction-rooted) fused region (G5b): last-axis reductions + their pointwise feed and
    /// epilogue, lowered to one row-at-a-time kernel; see [`RowRegion`].
    FusedRow(RowRegion),
    /// Fused decode attention (spec 014, flash attention), inputs `[q, k_cache, v_cache, mask]` with `q[B,Hq,1,D]`,
    /// `k`/`v[B,Hq/n_rep,cap,D]`, `mask[B,Hm,1,cap]` (`Hm` is 1 or Hq), all dense: `infer` rejects any other
    /// operand shape. Output `[B,Hq,1,D]` (== q's type). The op does GQA (`n_rep = Hq/Hkv`),
    /// the implicit transpose, and the online softmax internally; its definition is the `ops::attention_masked`
    /// decomposition (eval reproduces it). The `flash_attention_capped` pass introduces it on every backend
    /// for the shapes in the LDS cap; the planner chooses its kernel (Card 557: the generated region decode
    /// within the LDS cap, the generated private-array decode above it).
    FlashAttentionDecode {
        n_rep: usize,
        scale: f32,
    },
    /// Fused prefill attention (flash prefill, card 038), inputs `[q, k, v, mask]` with `q[1,Hq,L,D]`,
    /// `k`/`v[1,Hq/n_rep,L,D]`, `mask[1,Hm,L,L]` (causal; `Hm` is 1 or Hq), all dense: `infer` rejects any other
    /// operand shape. Output `[1,Hq,L,D]` (== q's type). The op does GQA
    /// (`n_rep = Hq/Hkv`), the implicit transpose, and an online (running max/sum) softmax over the KV internally,
    /// never materializing the `[1,Hq,L,L]` scores (peak attention memory `O(L*D)`, not `O(L^2)`). Its definition
    /// is the `ops::attention_prefill_softcap` decomposition. The planner chooses its kernel (Card 557): the
    /// generated region prefill for a single sequence within `FLASH_PREFILL_LDS_CAP` and no softcap, else the
    /// imported `flash_prefill` kernel (one workgroup per (head, query row)). `softcap` (card 198): `Some(c)`
    /// applies Gemma2/Grok attention-logit softcapping (`c * tanh(scores / c)`, [`crate::ops::softcap`]) to the
    /// scaled `Q K^T` before the mask-add, matching `attention_prefill_softcap`; `None` applies no softcap.
    FlashAttentionPrefill {
        n_rep: usize,
        scale: f32,
        softcap: Option<f32>,
    },
    /// Fused RoPE (rotary position embedding), inputs `[x, cos, sin]`. `x[.., D]` is the per-head tensor;
    /// `cos`/`sin` are the position-selected tables of width `rot` (`rot <= D`; `rot == D` for full rotary,
    /// `rot < D` for phi3/qwen3next partial rotary where `[rot, D)` passes through unrotated). cos/sin broadcast
    /// into `x`'s leading dims (as in the decomposition's `Mul(xr, cos)`) without growing them; `infer` rejects a
    /// table that would broadcast `x` up. Output type == `x`'s type. The
    /// definition is the `ops::rope_partial` half-split-rotate decomposition
    /// (`out = xr*cos + rotate_half(xr)*sin`, `rotate_half(x) = concat(-x2, x1)` with `x1 = x[..half]`,
    /// `x2 = x[half..rot]`, `half = rot/2`); eval reproduces it exactly. Introduced by `transform::rope_fusion`
    /// (structural match of that chain), lowered to the synthesized `kernelgen::rope` kernel: one dispatch replacing
    /// the slice/slice/neg/concat/mul/mul/add chain (~6-8 dispatches per rope call). GPT-NeoX / HF convention
    /// (non-interleaved); GGUF weights are deinterleaved to half-split at load.
    Rope {
        rot: usize,
    },
    /// A cross-replica collective reduce (card 049a, tensor-parallel). Each rank holds a same-shaped partial tensor;
    /// the collective combines them with `op` and distributes the result back to every rank. The output type is
    /// identical to the input type.
    ///
    /// `axis` is the shard axis (which logical dimension was split across ranks at the row-parallel matmul's weight
    /// sharding). It is metadata for the multi-device lowering (ring-allreduce, card 049b) and shard-shape checking;
    /// it does not index an axis that gets reduced here.
    ///
    /// Single-rank (world_size=1) eval: the identity (AllReduce(x) with R=1 replicas gives x). The test harness that
    /// simulates multi-rank sums the per-rank partial results outside the graph, so the node's within-graph eval
    /// stays identity.
    AllReduce {
        op: RedOp,
        /// The shard (contraction) axis this collective collapses across replicas. Metadata only for this slice; the
        /// ring-allreduce lowering (card 049b) uses it for buffer sizing.
        axis: usize,
    },
    /// Cross-replica all-gather (card 049a, column-parallel matmul). Each rank holds a slice of the output tensor
    /// along `axis`; the collective concatenates the slices across ranks, growing `axis` by world_size.
    ///
    /// The output type is shape-preserving in the single-rank (world_size=1) graph (the slice is the full tensor). In
    /// a multi-rank lowering (card 049b) the output's `axis` dimension grows by world_size. The node marks where the
    /// gather goes so the multi-rank lowering can find it.
    ///
    /// Single-rank eval: the identity. The test harness simulates multi-rank by concatenating per-rank evals outside
    /// the graph, so the in-graph eval stays identity.
    AllGather {
        /// The output axis along which ranks are concatenated. In column-parallel matmul this is the last axis (the
        /// N / output-feature axis); metadata for card 049b.
        axis: usize,
    },
    /// Exact uniform noise from an integer seed (card 551a, R472-007): one integer hash plus a
    /// power-of-two multiply, bit-exact on every backend (tier 1, ADR-0101). Input `seed [..]` I32 (u32
    /// bits); output `[.., cols]` F32. For row-major flat index `r` (the seed's own flat position) and
    /// column `i`, `x = fmix32(seed[r] XOR wrapping_mul(r * cols + i, 0x9E3779B9))`, `u = (2 * (x >> 9) +
    /// 1) * 2^-24`: an exact I32->F32 conversion below `2^24` and a power-of-two multiply, so `u` is
    /// always in `[2^-24, 1 - 2^-24]` and a double `log` of it never diverges. See
    /// [`crate::ops::sampling::hash32`] and [`crate::ops::sampling::random_uniform`].
    RandomUniform {
        cols: usize,
    },
    /// Pick one token per row (card 551a, R472-007, R-551a-2, deval.md section 8): fixes the three
    /// lane-order/NaN/sentinel bugs of the per-backend sampler bodies Card 447 consolidated, by making
    /// sampling a graph suffix with one semantic definition (ADR-0101, L7). `rule` selects the operand
    /// list (see [`SampleRule`]); every rule shares one row-max / non-finite pass over the raw `logits`
    /// (before any top-k/top-p truncation) and one `(value desc, index asc)` tie rule. Output `[.., 2]`
    /// I32 = `(token, non_finite_index)`: `non_finite_index` is the lowest index `i` with
    /// `!logits[i].is_finite()`, or `-1` when every logit in the row is finite; `token` is forced to `0`
    /// when `non_finite_index >= 0`, so the output is fully determined even though the driver (card
    /// 551b) never commits a faulted row's token. F32 logits only (every other dtype is a typed
    /// `ShapeError`); see [`crate::ops::sampling::sample_head`].
    SampleToken {
        rule: SampleRule,
    },
}

/// Fold one fused-region operand into a hasher. A literal hashes its dtype tag and bit pattern, so `1.0`,
/// `-0.0` versus `0.0` and distinct NaN payloads all stay distinct. `#[cfg(test)]` alongside
/// [`OpKind::hash_params`], its only caller.
#[cfg(test)]
fn hash_fused_operand<H: std::hash::Hasher>(operand: &FusedOperand, h: &mut H) {
    use std::hash::Hash;
    match operand {
        FusedOperand::Local(id) => {
            0u8.hash(h);
            id.hash(h);
        }
        FusedOperand::Lit(Scalar::F32(v)) => {
            1u8.hash(h);
            v.to_bits().hash(h);
        }
        FusedOperand::Lit(Scalar::I32(v)) => {
            2u8.hash(h);
            v.hash(h);
        }
        FusedOperand::PackLane { input, lane } => {
            3u8.hash(h);
            input.hash(h);
            lane.hash(h);
        }
    }
}

#[cfg(test)]
fn hash_fused_operands<H: std::hash::Hasher>(operands: &[FusedOperand], h: &mut H) {
    use std::hash::Hash;
    operands.len().hash(h);
    for operand in operands {
        hash_fused_operand(operand, h);
    }
}

/// Card 529 (R466-011): reject a `DTypeClass::StorageOnly` or `DTypeClass::Packed` operand of an
/// arithmetic op (`Unary`, `Binary`, `Reduce`). Only `Cast` and the movement ops, which never read a
/// value's bit pattern, admit those dtypes; `DTypeClass::Index` (I32) stays admitted here alongside
/// `DTypeClass::Arithmetic`, since these ops still need exact-integer arithmetic (position math, hash
/// bit ops).
fn reject_non_arithmetic(op: &'static str, dtype: DType) -> Result<(), ShapeError> {
    match dtype.class() {
        DTypeClass::StorageOnly | DTypeClass::Packed => Err(ShapeError::DtypeOp { op, dtype }),
        DTypeClass::Arithmetic | DTypeClass::Index => Ok(()),
    }
}

/// Card 529 (R466-010): reject an index-role operand (`Gather`'s and `Scatter`'s index,
/// `ScatterUpdate`'s `inv`, `IndexedMatMul`'s and friends' `idx`, `DynamicUpdateSlice`'s runtime index)
/// outside [`DType::is_index_operand`].
fn check_index_dtype(op: &'static str, dtype: DType) -> Result<(), ShapeError> {
    if dtype.is_index_operand() {
        Ok(())
    } else {
        Err(ShapeError::DtypeOp { op, dtype })
    }
}

/// Card 529 (R466-012): the one `Binary`-operand dtype-agreement rule. `infer`'s own `Binary` arm calls
/// this for the pair once it has ruled out the literal-coercion carve-out (a differing, scalar-shaped
/// right operand: `Builder::binary_scalar`'s baked immediate, which `infer` cannot otherwise tell apart
/// from a same-shaped `Value`). The four call sites that additionally check this ahead of `infer`, for
/// an earlier and more specific error, call it too: `Builder::binary`, `BuilderAppendPlan::equation_named`,
/// `Builder::preflight_append`, and `Graph::validate`; unlike `infer`, each of those sees the real
/// `Operand`s and only runs this rule when BOTH sides are `Operand::Value`, so the literal exception
/// never applies there in the first place.
pub(crate) fn check_binary_value_dtypes(a: &TensorType, b: &TensorType) -> Result<(), ShapeError> {
    if a.dtype == b.dtype {
        Ok(())
    } else {
        Err(ShapeError::Dtype {
            a: a.clone(),
            b: b.clone(),
        })
    }
}

/// Card 623 (R466-011, SC-001): the closed set of `MatMul` operand-dtype pairings `infer` accepts. Not
/// strict equality: two passes deliberately build a genuinely mismatched F32-activation x BF16/F16-weight
/// `MatMul` on purpose, so this is a stated exception, not a gap. `poot_graph_plan`'s `fold_dense_contractions`
/// folds exactly that pair (behind a `Transpose`) into `DenseContraction` (its own `weight` dtype table is
/// `DENSE_CONTRACTION_WEIGHT_DTYPES`); the planner's own
/// `poot_graph_plan::widen_mismatched_matmul_dtypes` (card 273) either widens that exact pair with a Cast
/// or, when `matmul_tensor_core_eligible`/`matmul_bf16_decode_gemv_eligible` says so, runs the mismatched
/// pair directly on a WMMA/coopmat/decode-GEMV kernel built for it. Every other pairing (F16 x BF16, I32 x
/// F32, a BF16 activation x F32 weight, two different storage/packed dtypes, ...) is rejected.
fn matmul_operand_dtypes_allowed(a: DType, b: DType) -> bool {
    (a.class() == DTypeClass::Arithmetic && a == b)
        || (a == DType::F32 && matches!(b, DType::BF16 | DType::F16))
}

impl OpKind {
    /// Fold this op's own parameters (not just its discriminant) into a hasher, allocation-free (no `Debug` or
    /// `format!`). `OpKind` cannot derive `Hash` because `FlashAttentionDecode`/`Prefill` etc. carry an `f32 scale`,
    /// which is not `Eq`/`Hash` under IEEE NaN semantics; every f32 field hashes its bit pattern instead. Hashing
    /// only the discriminant cannot distinguish two eqns of the same op kind with different params (a `Slice` at
    /// a different offset, a `DynamicUpdateSlice` on a different axis, a different `DType`).
    ///
    /// `Fused`/`FusedRow` hash their full region description: every step's op and operands (local ids, literal
    /// bit patterns, pack lanes), the output local, the pack list and the row axis. A fused region is the whole
    /// program of one synthesized kernel, so two regions that differ anywhere (`x + 1` versus `x * 2`) must hash
    /// differently.
    ///
    /// `pub(crate)`/test-only (card 546b): the production consumer was `poot-gpu`'s `GraphIdentity`
    /// (guarding `CseMemo`/`DecodeCache` reuse), gone with the rest of the pre-contract executor.
    #[cfg(test)]
    pub(crate) fn hash_params<H: std::hash::Hasher>(&self, h: &mut H) {
        use std::hash::Hash;
        std::mem::discriminant(self).hash(h);
        match self {
            OpKind::Unary(op) => op.hash(h),
            OpKind::Binary(op) => op.hash(h),
            OpKind::Reduce { op, axis, keepdim } => {
                op.hash(h);
                axis.hash(h);
                keepdim.hash(h);
            }
            OpKind::Broadcast { shape } => shape.hash(h),
            OpKind::Cast { to } => to.hash(h),
            OpKind::Reshape { shape } => shape.hash(h),
            OpKind::Transpose { perm } => perm.hash(h),
            OpKind::Slice { axis, start, end } => {
                axis.hash(h);
                start.hash(h);
                end.hash(h);
            }
            OpKind::Concat { axis } => axis.hash(h),
            OpKind::Iota { len } => len.hash(h),
            OpKind::Gather { axis } => axis.hash(h),
            OpKind::Scatter { axis } => axis.hash(h),
            OpKind::Select
            | OpKind::ScatterUpdate
            | OpKind::MatMul
            | OpKind::MatMulBias
            | OpKind::IndexedMatMul
            | OpKind::PackI8 => {} // no params beyond the discriminant already hashed
            OpKind::PackedDequant { descriptor } => descriptor.hash(h),
            OpKind::PackedContraction { descriptor, blocks } => {
                descriptor.hash(h);
                blocks.hash(h);
            }
            OpKind::PackedRowGather { descriptor } => descriptor.hash(h),
            OpKind::DenseContraction { weight } => weight.hash(h),
            OpKind::DenseRowGather { source } => source.hash(h),
            OpKind::ArgTopK { k } => k.hash(h),
            OpKind::DynamicUpdateSlice { axis } => axis.hash(h),
            OpKind::UnpackI8 { len } => len.hash(h),
            OpKind::Fused(region) => {
                region.n_inputs.hash(h);
                region.steps.len().hash(h);
                for step in &region.steps {
                    step.op.hash(h);
                    hash_fused_operands(&step.inputs, h);
                }
                region.output.hash(h);
                region.pack.hash(h);
            }
            OpKind::FusedRow(region) => {
                region.n_inputs.hash(h);
                region.axis.hash(h);
                region.steps.len().hash(h);
                for step in &region.steps {
                    match step {
                        RowStep::Pointwise { op, inputs } => {
                            0u8.hash(h);
                            op.hash(h);
                            hash_fused_operands(inputs, h);
                        }
                        RowStep::Reduce { op, input } => {
                            1u8.hash(h);
                            op.hash(h);
                            hash_fused_operand(input, h);
                        }
                    }
                }
                region.output.hash(h);
            }
            OpKind::FlashAttentionDecode { n_rep, scale } => {
                n_rep.hash(h);
                scale.to_bits().hash(h);
            }
            OpKind::FlashAttentionPrefill {
                n_rep,
                scale,
                softcap,
            } => {
                n_rep.hash(h);
                scale.to_bits().hash(h);
                softcap.map(f32::to_bits).hash(h);
            }
            OpKind::Rope { rot } => rot.hash(h),
            OpKind::AllReduce { op, axis } => {
                op.hash(h);
                axis.hash(h);
            }
            OpKind::AllGather { axis } => axis.hash(h),
            OpKind::RandomUniform { cols } => cols.hash(h),
            OpKind::SampleToken { rule } => rule.hash(h),
        }
    }

    /// Infer the output type from the operand types. Pure, host-only, no data. The single source of
    /// truth for every eqn's result shape/dtype.
    pub fn infer(&self, ins: &[TensorType]) -> Result<TensorType, ShapeError> {
        match self {
            OpKind::Unary(op) => {
                arity(ins, 1)?;
                reject_non_arithmetic("Unary", ins[0].dtype)?;
                if ins[0].dtype == DType::I32 {
                    if !matches!(op, UnOp::Not | UnOp::Clz) {
                        return Err(ShapeError::DtypeOp {
                            op: "Unary",
                            dtype: DType::I32,
                        });
                    }
                    return Ok(ins[0].clone());
                }
                if matches!(op, UnOp::Not | UnOp::Clz) {
                    return Err(ShapeError::DtypeOp {
                        op: match op {
                            UnOp::Not => "Unary(Not)",
                            UnOp::Clz => "Unary(Clz)",
                            _ => "Unary",
                        },
                        dtype: ins[0].dtype,
                    });
                }
                Ok(ins[0].clone())
            }
            OpKind::Binary(op) => {
                arity(ins, 2)?;
                reject_non_arithmetic("Binary", ins[0].dtype)?;
                reject_non_arithmetic("Binary", ins[1].dtype)?;
                if ins[0].dtype == DType::I32 {
                    check_binary_value_dtypes(&ins[0], &ins[1])?;
                    if *op == BinOp::Div {
                        return Err(ShapeError::DtypeOp {
                            op: "Binary(Div)",
                            dtype: DType::I32,
                        });
                    }
                } else {
                    // A differing scalar rhs is the established literal-coercion form (the builder bakes it into
                    // the kernel, e.g. `Builder::binary_scalar`). A non-scalar rhs can only be a real tensor and
                    // must share the lhs dtype (R466-012, `check_binary_value_dtypes`); `Graph::validate` and the
                    // builder call sites close the remaining scalar-Value ambiguity because they can inspect
                    // `Operand` rather than `TensorType` alone.
                    if !ins[1].shape.is_empty() {
                        check_binary_value_dtypes(&ins[0], &ins[1])?;
                    }
                    if matches!(
                        op,
                        BinOp::GeU
                            | BinOp::RemU
                            | BinOp::And
                            | BinOp::Or
                            | BinOp::Xor
                            | BinOp::Shl
                            | BinOp::Shr
                    ) {
                        return Err(ShapeError::DtypeOp {
                            op: match op {
                                BinOp::GeU => "Binary(GeU)",
                                BinOp::RemU => "Binary(RemU)",
                                BinOp::And => "Binary(And)",
                                BinOp::Or => "Binary(Or)",
                                BinOp::Xor => "Binary(Xor)",
                                BinOp::Shl => "Binary(Shl)",
                                BinOp::Shr => "Binary(Shr)",
                                _ => "Binary",
                            },
                            dtype: ins[0].dtype,
                        });
                    }
                }
                let shape = broadcast_shapes(&ins[0].shape, &ins[1].shape).ok_or_else(|| {
                    ShapeError::Broadcast {
                        a: ins[0].shape.clone(),
                        b: ins[1].shape.clone(),
                    }
                })?;
                // dtype follows the first operand (the tensor); scalar literals adopt it.
                Ok(TensorType::new(shape, ins[0].dtype))
            }
            OpKind::Select => {
                arity(ins, 3)?;
                if ins.iter().any(|ty| ty.dtype != DType::I32) {
                    return Err(ShapeError::DtypeOp {
                        op: "Select",
                        dtype: ins
                            .iter()
                            .find(|ty| ty.dtype != DType::I32)
                            .map(|ty| ty.dtype)
                            .unwrap_or(DType::I32),
                    });
                }
                let ab = broadcast_shapes(&ins[1].shape, &ins[2].shape).ok_or_else(|| {
                    ShapeError::Broadcast {
                        a: ins[1].shape.clone(),
                        b: ins[2].shape.clone(),
                    }
                })?;
                let shape =
                    broadcast_shapes(&ins[0].shape, &ab).ok_or_else(|| ShapeError::Broadcast {
                        a: ins[0].shape.clone(),
                        b: ab,
                    })?;
                Ok(TensorType::new(shape, DType::I32))
            }
            OpKind::Reduce { axis, keepdim, .. } => {
                arity(ins, 1)?;
                // Reduce is strictly arithmetic (unlike Unary/Binary, it has no exact-integer form), so
                // DTypeClass::Index (I32) is rejected here too, not just StorageOnly/Packed (R466-011).
                if ins[0].dtype.class() != DTypeClass::Arithmetic {
                    return Err(ShapeError::DtypeOp {
                        op: "Reduce",
                        dtype: ins[0].dtype,
                    });
                }
                let mut shape = ins[0].shape.clone();
                check_axis(*axis, shape.len())?;
                if *keepdim {
                    shape[*axis] = 1;
                } else {
                    shape.remove(*axis);
                }
                Ok(TensorType::new(shape, ins[0].dtype))
            }
            OpKind::Broadcast { shape } => {
                arity(ins, 1)?;
                // `broadcast_shapes` is symmetric, but this op is directional: the source may only expand to the requested
                // target. Some compatible joint shape would also accept rank or non-unit extent shrinking when that joint
                // shape is larger than `shape`.
                let joint = broadcast_shapes(&ins[0].shape, shape).ok_or_else(|| {
                    ShapeError::Broadcast {
                        a: ins[0].shape.clone(),
                        b: shape.clone(),
                    }
                })?;
                if joint != *shape {
                    return Err(ShapeError::Broadcast {
                        a: ins[0].shape.clone(),
                        b: shape.clone(),
                    });
                }
                Ok(TensorType::new(shape.clone(), ins[0].dtype))
            }
            OpKind::Cast { to } => {
                arity(ins, 1)?;
                let from = ins[0].dtype;
                // The nine pairs the evaluator never defines (card 555): moved here from
                // the walk's own declared `REFUSED_CASTS` table, so an unadmitted pair fails at trace
                // time instead of reaching `eval`.
                const REFUSED: &[(DType, DType)] = &[
                    (DType::BF16, DType::I32),
                    (DType::F16, DType::I32),
                    (DType::BF16, DType::I8),
                    (DType::F16, DType::I8),
                    (DType::I32, DType::E4M3FN),
                    (DType::I8, DType::E4M3FN),
                    (DType::E4M3FN, DType::I32),
                    (DType::E4M3FN, DType::I8),
                    (DType::F32, DType::I8),
                ];
                if REFUSED.contains(&(from, *to)) {
                    return Err(ShapeError::CastUnsupported { from, to: *to });
                }
                // same shape, new dtype.
                Ok(TensorType::new(ins[0].shape.clone(), *to))
            }
            OpKind::Reshape { shape } => {
                arity(ins, 1)?;
                let checked_numel = |shape: &[usize]| {
                    shape
                        .iter()
                        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
                        .ok_or_else(|| ShapeError::ReshapeElementCountOverflow {
                            shape: shape.to_vec(),
                        })
                };
                let from_numel = checked_numel(&ins[0].shape)?;
                let to_numel = checked_numel(shape)?;
                if to_numel != from_numel {
                    return Err(ShapeError::Reshape {
                        from: ins[0].shape.clone(),
                        to: shape.clone(),
                        from_numel,
                        to_numel,
                    });
                }
                Ok(TensorType::new(shape.clone(), ins[0].dtype))
            }
            OpKind::Transpose { perm } => {
                arity(ins, 1)?;
                let rank = ins[0].rank();
                // `perm` must be a permutation of `0..rank`: correct length, every entry in range, no duplicates. infer is a
                // validator: an out-of-range perm must return ShapeError::Perm, never panic on `ins[0].shape[p]`, and a
                // duplicate like `[0, 0]` must not silently drop an axis.
                let is_permutation = perm.len() == rank && {
                    let mut seen = vec![false; rank];
                    perm.iter()
                        .all(|&p| p < rank && !std::mem::replace(&mut seen[p], true))
                };
                if !is_permutation {
                    return Err(ShapeError::Perm {
                        perm: perm.clone(),
                        rank,
                    });
                }
                let shape = perm.iter().map(|&p| ins[0].shape[p]).collect::<Vec<_>>();
                Ok(TensorType::new(shape, ins[0].dtype))
            }
            OpKind::Slice { axis, start, end } => {
                arity(ins, 1)?;
                check_axis(*axis, ins[0].rank())?;
                let len = ins[0].shape[*axis];
                if start > end || *end > len {
                    return Err(ShapeError::Slice {
                        start: *start,
                        end: *end,
                        len,
                    });
                }
                let mut shape = ins[0].shape.clone();
                shape[*axis] = end - start;
                Ok(TensorType::new(shape, ins[0].dtype))
            }
            OpKind::Concat { axis } => {
                if ins.is_empty() {
                    return Err(ShapeError::Arity {
                        expected: 1,
                        got: 0,
                    });
                }
                let rank = ins[0].rank();
                check_axis(*axis, rank)?;
                let mut shape = ins[0].shape.clone();
                let mut total = 0;
                for t in ins {
                    // Every operand must share ins[0]'s rank: otherwise the zip dim-check below ignores the extra dims of a
                    // longer operand, and `t.shape[*axis]` panics when a shorter operand has rank <= axis. Report the rank
                    // mismatch (a/b carry the ranks).
                    if t.rank() != rank {
                        return Err(ShapeError::Concat {
                            dim: *axis,
                            a: t.rank(),
                            b: rank,
                        });
                    }
                    for (d, (&a, &b)) in t.shape.iter().zip(shape.iter()).enumerate() {
                        if d != *axis && a != b {
                            return Err(ShapeError::Concat { dim: d, a, b });
                        }
                    }
                    total += t.shape[*axis];
                }
                shape[*axis] = total;
                Ok(TensorType::new(shape, ins[0].dtype))
            }
            OpKind::Iota { len } => {
                // Nullary: a compile-time-computed range, shape [len]. `fold_iota` replaces every
                // Iota with a computed graph constant before planning.
                arity(ins, 0)?;
                Ok(TensorType::new(vec![*len], DType::F32))
            }
            OpKind::Gather { axis } => {
                arity(ins, 2)?;
                let (data, index) = (&ins[0], &ins[1]);
                check_index_dtype("Gather", index.dtype)?;
                check_axis(*axis, data.rank())?;
                // out = data.shape[..axis] ++ index.shape ++ data.shape[axis+1..]. A scalar index (shape []) drops the axis
                // (embedding/RoPE-row lookup); a vector index [L] replaces the axis with L (e.g. a whole prompt's embeddings).
                let mut shape = data.shape[..*axis].to_vec();
                shape.extend_from_slice(&index.shape);
                shape.extend_from_slice(&data.shape[*axis + 1..]);
                Ok(TensorType::new(shape, data.dtype))
            }
            OpKind::Scatter { axis } => {
                // out[index[j]] = src[j]: a same-shape axis-permutation; out type == src type.
                arity(ins, 2)?;
                check_index_dtype("Scatter", ins[1].dtype)?;
                check_axis(*axis, ins[0].rank())?;
                Ok(ins[0].clone())
            }
            OpKind::ArgTopK { k } => {
                // rank [..,E] -> [..,k]: same leading dims, last axis replaced by k. Output dtype is always F32 (the idx
                // convention IndexedMatMul reads via its f32 lane).
                arity(ins, 1)?;
                if ins[0].rank() == 0 {
                    return Err(ShapeError::Arity {
                        expected: 1,
                        got: 0,
                    });
                }
                let mut shape = ins[0].shape.clone();
                let extent = *shape.last().unwrap();
                if *k > extent {
                    return Err(ShapeError::TopK { k: *k, extent });
                }
                *shape.last_mut().unwrap() = *k;
                Ok(TensorType::new(shape, DType::F32))
            }
            OpKind::PackI8 => {
                arity(ins, 1)?;
                if ins[0].rank() == 0 {
                    return Err(ShapeError::Arity {
                        expected: 1,
                        got: 0,
                    });
                }
                let mut shape = ins[0].shape.clone();
                let last = shape.last_mut().unwrap();
                *last = last.div_ceil(4);
                Ok(TensorType::new(shape, DType::I32))
            }
            OpKind::UnpackI8 { len } => {
                arity(ins, 1)?;
                if ins[0].rank() == 0 {
                    return Err(ShapeError::Arity {
                        expected: 1,
                        got: 0,
                    });
                }
                let mut shape = ins[0].shape.clone();
                *shape.last_mut().unwrap() = *len;
                Ok(TensorType::new(shape, DType::F32))
            }
            OpKind::ScatterUpdate => {
                // base[POOL,..rest], src[N,..rest], inv[POOL] -> base's type. The N rows mapped by inv are
                // written from src, the rest passed through from base.
                arity(ins, 3)?;
                let (base, src, inv) = (&ins[0], &ins[1], &ins[2]);
                check_index_dtype("ScatterUpdate", inv.dtype)?;
                if base.rank() < 1 || src.rank() != base.rank() || inv.rank() != 1 {
                    return Err(ShapeError::MatMulRank {
                        a: base.shape.clone(),
                        b: src.shape.clone(),
                    });
                }
                if base.shape[1..] != src.shape[1..] {
                    return Err(ShapeError::MatMulContract {
                        a: base.shape.clone(),
                        b: src.shape.clone(),
                    });
                }
                if inv.shape[0] != base.shape[0] {
                    return Err(ShapeError::Broadcast {
                        a: base.shape.clone(),
                        b: inv.shape.clone(),
                    });
                }
                Ok(base.clone())
            }
            OpKind::MatMul => {
                arity(ins, 2)?;
                let (a, b) = (&ins[0], &ins[1]);
                reject_non_arithmetic("MatMul", a.dtype)?;
                reject_non_arithmetic("MatMul", b.dtype)?;
                if !matmul_operand_dtypes_allowed(a.dtype, b.dtype) {
                    return Err(ShapeError::MatMulOperandDtype {
                        a: a.dtype,
                        b: b.dtype,
                    });
                }
                if a.rank() < 2 || b.rank() < 2 {
                    return Err(ShapeError::MatMulRank {
                        a: a.shape.clone(),
                        b: b.shape.clone(),
                    });
                }
                let (ra, rb) = (a.rank(), b.rank());
                let (m, k) = (a.shape[ra - 2], a.shape[ra - 1]);
                let (k2, n) = (b.shape[rb - 2], b.shape[rb - 1]);
                if k != k2 {
                    return Err(ShapeError::MatMulContract {
                        a: a.shape.clone(),
                        b: b.shape.clone(),
                    });
                }
                let batch =
                    broadcast_shapes(&a.shape[..ra - 2], &b.shape[..rb - 2]).ok_or_else(|| {
                        ShapeError::Broadcast {
                            a: a.shape[..ra - 2].to_vec(),
                            b: b.shape[..rb - 2].to_vec(),
                        }
                    })?;
                let mut shape = batch;
                shape.push(m);
                shape.push(n);
                Ok(TensorType::new(shape, a.dtype))
            }
            OpKind::PackedDequant { descriptor } => {
                let roles = descriptor.sources();
                arity(ins, roles.len())?;
                for (input, role) in ins.iter().zip(roles.iter().copied()) {
                    let expected = Box::new(TensorType::new(
                        descriptor.source_shape(role).to_vec(),
                        DType::I8,
                    ));
                    if input.dtype != DType::I8 {
                        return Err(ShapeError::PackedDequantOperand {
                            role,
                            field: "dtype",
                            expected,
                            actual: Box::new(input.clone()),
                        });
                    }
                    if input.shape != expected.shape {
                        return Err(ShapeError::PackedDequantOperand {
                            role,
                            field: "shape",
                            expected,
                            actual: Box::new(input.clone()),
                        });
                    }
                }
                Ok(TensorType::f32(descriptor.shape().to_vec()))
            }
            OpKind::PackedContraction { descriptor, blocks } => {
                arity(ins, 1 + descriptor.sources().len())?;
                OpKind::PackedDequant {
                    descriptor: *descriptor,
                }
                .infer(&ins[1..])?;
                let activation = &ins[0];
                if activation.dtype != DType::F32 {
                    return Err(ShapeError::PackedContractionDtype {
                        operand: "activation",
                        actual: activation.dtype,
                    });
                }
                let [out, k] = descriptor.shape();
                if *blocks == 0 || out % blocks != 0 {
                    return Err(ShapeError::PackedContractionBlocks {
                        out,
                        blocks: *blocks,
                    });
                }
                let block_out = out / blocks;
                if *blocks == 1 {
                    if activation.rank() < 2 || activation.shape.last().copied() != Some(k) {
                        return Err(ShapeError::PackedContractionActivation {
                            expected_k: k,
                            actual: activation.clone(),
                        });
                    }
                    let mut shape = activation.shape.clone();
                    *shape.last_mut().expect("rank checked") = out;
                    Ok(TensorType::f32(shape))
                } else {
                    let shape = activation.shape.as_slice();
                    if shape.len() != 3 || shape[0] != *blocks || shape[2] != k {
                        return Err(ShapeError::PackedContractionActivation {
                            expected_k: k,
                            actual: activation.clone(),
                        });
                    }
                    Ok(TensorType::f32(vec![*blocks, shape[1], block_out]))
                }
            }
            OpKind::PackedRowGather { descriptor } => {
                let roles = descriptor.sources().len();
                arity(ins, roles + 1)?;
                OpKind::PackedDequant {
                    descriptor: *descriptor,
                }
                .infer(&ins[..roles])?;
                let ids = &ins[roles];
                if !matches!(ids.dtype, DType::F32 | DType::I32) {
                    return Err(ShapeError::PackedContractionDtype {
                        operand: "row ids",
                        actual: ids.dtype,
                    });
                }
                let mut shape = ids.shape.clone();
                shape.push(descriptor.shape()[1]);
                Ok(TensorType::f32(shape))
            }
            OpKind::DenseContraction { weight } => {
                // activation [..,M,K] F32 x weight [N,K] (`weight` dtype) -> [..,M,N] F32. The weight is in checkpoint order,
                // so its contraction axis is the last one, not the second to last.
                arity(ins, 2)?;
                if !DENSE_CONTRACTION_WEIGHT_DTYPES.contains(weight) {
                    return Err(ShapeError::DenseContractionWeightDtype { actual: *weight });
                }
                let (activation, source) = (&ins[0], &ins[1]);
                if activation.dtype != DType::F32 {
                    return Err(ShapeError::DenseContractionDtype {
                        operand: "activation",
                        actual: activation.dtype,
                    });
                }
                if source.dtype != *weight {
                    return Err(ShapeError::DenseContractionDtype {
                        operand: "weight",
                        actual: source.dtype,
                    });
                }
                if source.rank() != 2 {
                    return Err(ShapeError::DenseContractionWeightRank {
                        actual: source.clone(),
                    });
                }
                let k = source.shape[1];
                if activation.rank() < 2 || activation.shape.last().copied() != Some(k) {
                    return Err(ShapeError::DenseContractionActivation {
                        expected_k: k,
                        actual: activation.clone(),
                    });
                }
                let mut shape = activation.shape.clone();
                *shape.last_mut().expect("rank checked") = source.shape[0];
                Ok(TensorType::f32(shape))
            }
            OpKind::DenseRowGather { source: dtype } => {
                // table [V, R] (`source` dtype) x index [S..] I32 -> [S.., R] F32. Same row selection as `Gather { axis: 0 }`
                // with the widening decode folded in, so the output is F32 whatever the table's storage is.
                arity(ins, 2)?;
                if !DENSE_ROW_GATHER_SOURCE_DTYPES.contains(dtype) {
                    return Err(ShapeError::DenseRowGatherSourceDtype { actual: *dtype });
                }
                let (table, index) = (&ins[0], &ins[1]);
                if table.dtype != *dtype {
                    return Err(ShapeError::DenseRowGatherDtype {
                        operand: "table",
                        actual: table.dtype,
                    });
                }
                if index.dtype != DType::I32 {
                    return Err(ShapeError::DenseRowGatherDtype {
                        operand: "index",
                        actual: index.dtype,
                    });
                }
                if table.rank() != 2 {
                    return Err(ShapeError::DenseRowGatherTableRank {
                        actual: table.clone(),
                    });
                }
                let mut shape = index.shape.clone();
                shape.push(table.shape[1]);
                Ok(TensorType::f32(shape))
            }
            OpKind::MatMulBias => {
                // A[..,M,K], B[..,K,N], bias[N] -> [..,M,N] (same as MatMul; bias does not change the shape).
                // The bias is exactly what its decomposition's `Add` reads: `[N]` of the product's dtype.
                arity(ins, 3)?;
                let product = OpKind::MatMul.infer(&ins[0..2])?;
                let bias = &ins[2];
                check_binary_value_dtypes(&product, bias)?;
                if bias.shape.as_slice() != &product.shape[product.rank() - 1..] {
                    return Err(ShapeError::Broadcast {
                        a: product.shape.clone(),
                        b: bias.shape.clone(),
                    });
                }
                Ok(product)
            }
            OpKind::IndexedMatMul => {
                // x[M,K], W[E,K,N], idx[M] -> [M,N]. Each row m contracts x[m] against expert W[idx[m]].
                arity(ins, 3)?;
                let (x, w, idx) = (&ins[0], &ins[1], &ins[2]);
                check_index_dtype("IndexedMatMul", idx.dtype)?;
                if x.rank() != 2 || w.rank() != 3 || idx.rank() != 1 {
                    return Err(ShapeError::MatMulRank {
                        a: x.shape.clone(),
                        b: w.shape.clone(),
                    });
                }
                let (m, k) = (x.shape[0], x.shape[1]);
                if w.shape[1] != k {
                    return Err(ShapeError::MatMulContract {
                        a: x.shape.clone(),
                        b: w.shape.clone(),
                    });
                }
                if idx.shape[0] != m {
                    return Err(ShapeError::Broadcast {
                        a: x.shape.clone(),
                        b: idx.shape.clone(),
                    });
                }
                Ok(TensorType::new(vec![m, w.shape[2]], x.dtype))
            }
            OpKind::DynamicUpdateSlice { axis } => {
                arity(ins, 3)?;
                let (operand, update, index) = (&ins[0], &ins[1], &ins[2]);
                check_index_dtype("DynamicUpdateSlice", index.dtype)?;
                check_axis(*axis, operand.rank())?;
                if update.rank() != operand.rank() {
                    return Err(ShapeError::UpdateSlice {
                        operand: operand.shape.clone(),
                        update: update.shape.clone(),
                        axis: *axis,
                    });
                }
                // every non-axis dim must match; the axis extent must fit (bound checked against the
                // index at eval time, where the concrete index is known).
                for (d, (&o, &u)) in operand.shape.iter().zip(&update.shape).enumerate() {
                    if d != *axis && o != u {
                        return Err(ShapeError::UpdateSlice {
                            operand: operand.shape.clone(),
                            update: update.shape.clone(),
                            axis: *axis,
                        });
                    }
                }
                if update.shape[*axis] > operand.shape[*axis] || !index.shape.is_empty() {
                    return Err(ShapeError::UpdateSlice {
                        operand: operand.shape.clone(),
                        update: update.shape.clone(),
                        axis: *axis,
                    });
                }
                Ok(operand.clone())
            }
            OpKind::Fused(region) => region.infer(ins),
            OpKind::FusedRow(region) => reject_i32_region("FusedRow", region.infer(ins)?),
            OpKind::FlashAttentionDecode { n_rep, .. } => {
                infer_flash_attention(ins, *n_rep, FlashForm::Decode)
            }
            OpKind::FlashAttentionPrefill { n_rep, .. } => {
                infer_flash_attention(ins, *n_rep, FlashForm::Prefill)
            }
            OpKind::Rope { rot } => infer_rope(ins, *rot),
            OpKind::AllReduce { .. } => {
                // Shape-preserving: each rank holds a same-shaped partial; the collective combines across ranks and returns
                // the full result in the same shape and dtype.
                arity(ins, 1)?;
                Ok(ins[0].clone())
            }
            OpKind::AllGather { .. } => {
                // Shape-preserving in the single-rank graph: the shard is the full tensor. A multi-rank lowering (card 049b)
                // would grow the axis dim by world_size; that lowering is deferred. The in-graph aval stays identity so the
                // rest of the graph sees the full output shape in simulation.
                arity(ins, 1)?;
                Ok(ins[0].clone())
            }
            OpKind::RandomUniform { cols } => {
                arity(ins, 1)?;
                let seed = &ins[0];
                if seed.dtype != DType::I32 {
                    return Err(ShapeError::RandomUniformSeedDtype { actual: seed.dtype });
                }
                let mut shape = seed.shape.clone();
                shape.push(*cols);
                Ok(TensorType::new(shape, DType::F32))
            }
            OpKind::SampleToken { rule } => infer_sample_token(ins, *rule),
        }
    }

    /// Whether this op is defined directly (a primitive) or by its decomposition into primitives (a
    /// composite), ADR-0101 tier 1. ADR-0007, ADR-0039: kernel choice is a plan property, so no op
    /// records one.
    ///
    /// `class() == Composite` exactly when [`crate::decompose::decompose`] returns `Some` for the op's
    /// operand types, whichever producer built the equation (Card 556's consistency row): the oracle
    /// evaluates a composite from that decomposition, so the composite and the chain it replaces are
    /// one definition, bit for bit.
    ///
    /// The IR primitives are the elementwise, select, reduce, broadcast, cast, movement, gather and
    /// scatter, update-slice, iota and matmul ops. The composites, each formed by a `compile` pass from
    /// the primitive chain its decomposition restates:
    ///
    /// | Op | Formed by |
    /// | --- | --- |
    /// | `FlashAttentionDecode`, `FlashAttentionPrefill` | `flash_attention_capped` from the attention chain |
    /// | `Rope` | `rope_fusion` from the half-split rotation chain |
    /// | `MatMulBias` | the contraction-epilogue fuse rule from `matmul` then a broadcast `[N]` add |
    /// | `Fused`, `FusedRow` | `fuse` from pointwise chains and their last-axis reductions |
    ///
    /// The ops that are neither an IR primitive above nor a composite, and why each stays primitive:
    ///
    /// | Op | Reason |
    /// | --- | --- |
    /// | `PackedDequant` | the one quantized primitive (ADR-0103 decision 2); its definition is `poot-quant`'s decoder |
    /// | `PackedContraction`, `PackedRowGather` | the packed claims; the oracle defines them directly until Card 629 gives the contraction a decomposition |
    /// | `DenseContraction`, `DenseRowGather` | the narrow-float folds; the oracle defines them directly until Card 629 gives them a decomposition |
    /// | `ArgTopK` | inverts a rank permutation without the `[.., E]` scatter a decomposition would need |
    /// | `PackI8`, `UnpackI8` | bit packing of KV codes; no arithmetic decomposition |
    /// | `AllReduce`, `AllGather` | cross-rank collectives; the identity at world size 1 |
    /// | `IndexedMatMul` | the row-indexed MoE contraction; revisited when contraction lowering lands (Card 727) |
    /// | `RandomUniform`, `SampleToken` | the sampling primitives (Card 719): counter-based noise and the token rule |
    ///
    /// The match is exhaustive: a new variant states its class here.
    pub fn class(&self) -> OpClass {
        match self {
            Self::Unary(_)
            | Self::Binary(_)
            | Self::Select
            | Self::Reduce { .. }
            | Self::Broadcast { .. }
            | Self::Cast { .. }
            | Self::Reshape { .. }
            | Self::Transpose { .. }
            | Self::Slice { .. }
            | Self::Concat { .. }
            | Self::Iota { .. }
            | Self::Gather { .. }
            | Self::Scatter { .. }
            | Self::ScatterUpdate
            | Self::MatMul
            | Self::DynamicUpdateSlice { .. }
            | Self::PackedDequant { .. }
            | Self::ArgTopK { .. }
            | Self::PackI8
            | Self::UnpackI8 { .. }
            | Self::AllReduce { .. }
            | Self::AllGather { .. }
            | Self::IndexedMatMul
            | Self::RandomUniform { .. }
            | Self::SampleToken { .. }
            | Self::PackedContraction { .. }
            | Self::PackedRowGather { .. }
            | Self::DenseContraction { .. }
            | Self::DenseRowGather { .. } => OpClass::Primitive,
            Self::FlashAttentionDecode { .. }
            | Self::FlashAttentionPrefill { .. }
            | Self::Rope { .. }
            | Self::MatMulBias
            | Self::Fused(_)
            | Self::FusedRow(_) => OpClass::Composite,
        }
    }

    /// The semantic report key for device-time/telemetry buckets (Card 552, R-552-2): the variant
    /// name alone, with no parameters - `&'static str` so it can never be a formatted, allocated
    /// per-call string like [`Self::name`]. Two equations whose `OpKind` differs land in two buckets
    /// even if the planner happens to give them the same kernel key; one `OpKind` planned to two
    /// different kernel keys (a shape-free plan widened at a large extent) still lands in one
    /// bucket, because this is a property of the op, not the plan (`DispatchTiming`'s optional plan
    /// display label is the separate, non-aggregating column for that).
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Unary(_) => "unary",
            Self::Binary(_) => "binary",
            Self::Select => "select",
            Self::Reduce { .. } => "reduce",
            Self::Broadcast { .. } => "broadcast",
            Self::Cast { .. } => "cast",
            Self::Reshape { .. } => "reshape",
            Self::Transpose { .. } => "transpose",
            Self::Slice { .. } => "slice",
            Self::Concat { .. } => "concat",
            Self::Iota { .. } => "iota",
            Self::Gather { .. } => "gather",
            Self::Scatter { .. } => "scatter",
            Self::ScatterUpdate => "scatter_update",
            Self::MatMul => "matmul",
            Self::PackedDequant { .. } => "packed_dequant",
            Self::PackedContraction { .. } => "packed_contraction",
            Self::PackedRowGather { .. } => "packed_row_gather",
            Self::DenseContraction { .. } => "dense_contraction",
            Self::DenseRowGather { .. } => "dense_row_gather",
            Self::MatMulBias => "matmul_bias",
            Self::ArgTopK { .. } => "arg_top_k",
            Self::IndexedMatMul => "indexed_matmul",
            Self::DynamicUpdateSlice { .. } => "dynamic_update_slice",
            Self::PackI8 => "pack_i8",
            Self::UnpackI8 { .. } => "unpack_i8",
            Self::Fused(_) => "fused",
            Self::FusedRow(_) => "fused_row",
            Self::FlashAttentionDecode { .. } => "flash_attention_decode",
            Self::FlashAttentionPrefill { .. } => "flash_attention_prefill",
            Self::Rope { .. } => "rope",
            Self::AllReduce { .. } => "all_reduce",
            Self::AllGather { .. } => "all_gather",
            Self::RandomUniform { .. } => "random_uniform",
            Self::SampleToken { .. } => "sample_token",
        }
    }

    /// The short name shown in a graph dump.
    pub fn name(&self) -> String {
        match self {
            OpKind::Unary(u) => format!("{u:?}").to_lowercase(),
            OpKind::Binary(b) => format!("{b:?}").to_lowercase(),
            OpKind::Select => "select".to_string(),
            OpKind::Reduce { op, axis, keepdim } => {
                format!(
                    "reduce.{} ax={axis} keep={keepdim}",
                    format!("{op:?}").to_lowercase()
                )
            }
            OpKind::Broadcast { shape } => format!("broadcast {shape:?}"),
            OpKind::Cast { to } => format!("cast {to:?}").to_lowercase(),
            OpKind::Reshape { shape } => format!("reshape {shape:?}"),
            OpKind::Transpose { perm } => format!("transpose {perm:?}"),
            OpKind::Slice { axis, start, end } => format!("slice ax={axis} {start}..{end}"),
            OpKind::Concat { axis } => format!("concat ax={axis}"),
            OpKind::Iota { len } => format!("iota len={len}"),
            OpKind::Gather { axis } => format!("gather ax={axis}"),
            OpKind::Scatter { axis } => format!("scatter ax={axis}"),
            OpKind::ArgTopK { k } => format!("arg_top_k k={k}"),
            OpKind::ScatterUpdate => "scatter_update".to_string(),
            OpKind::PackI8 => "pack_i8".to_string(),
            OpKind::UnpackI8 { len } => format!("unpack_i8 len={len}"),
            OpKind::MatMul => "matmul".to_string(),
            OpKind::PackedDequant { descriptor } => {
                format!("packed_dequant {:?}", descriptor.format()).to_lowercase()
            }
            OpKind::PackedContraction { descriptor, blocks } => {
                let base = format!("packed_contraction {:?}", descriptor.format()).to_lowercase();
                if *blocks == 1 {
                    base
                } else {
                    format!("{base} blocks={blocks}")
                }
            }
            OpKind::PackedRowGather { descriptor } => {
                format!("packed_row_gather {:?}", descriptor.format()).to_lowercase()
            }
            OpKind::DenseContraction { weight } => {
                format!("dense_contraction {weight:?}").to_lowercase()
            }
            OpKind::DenseRowGather { source } => {
                format!("dense_row_gather {source:?}").to_lowercase()
            }
            OpKind::MatMulBias => "matmul_bias".to_string(),
            OpKind::IndexedMatMul => "indexed_matmul".to_string(),
            OpKind::DynamicUpdateSlice { axis } => format!("dyn_update_slice ax={axis}"),
            OpKind::Fused(r) => format!("fused n_in={} steps={}", r.n_inputs, r.steps.len()),
            OpKind::FusedRow(r) => format!("fused_row n_in={} steps={}", r.n_inputs, r.steps.len()),
            OpKind::FlashAttentionDecode { n_rep, scale } => {
                format!("flash_attn_decode n_rep={n_rep} scale={scale}")
            }
            OpKind::FlashAttentionPrefill {
                n_rep,
                scale,
                softcap,
            } => {
                format!("flash_attn_prefill n_rep={n_rep} scale={scale} softcap={softcap:?}")
            }
            OpKind::Rope { rot } => format!("rope rot={rot}"),
            OpKind::AllReduce { op, axis } => {
                format!("all_reduce.{} ax={axis}", format!("{op:?}").to_lowercase())
            }
            OpKind::AllGather { axis } => format!("all_gather ax={axis}"),
            OpKind::RandomUniform { cols } => format!("random_uniform cols={cols}"),
            OpKind::SampleToken { rule } => format!("sample_token rule={rule:?}").to_lowercase(),
        }
    }
}

fn reject_i32_region(op: &'static str, output: TensorType) -> Result<TensorType, ShapeError> {
    if output.dtype == DType::I32 {
        Err(ShapeError::DtypeOp {
            op,
            dtype: DType::I32,
        })
    } else {
        Ok(output)
    }
}

/// Which fused flash-attention form [`infer_flash_attention`] types.
#[derive(Clone, Copy)]
enum FlashForm {
    /// One query row per (batch, head) against a `cap`-long KV cache.
    Decode,
    /// Batch 1, `L` query rows against the same `L` keys (no carried cache).
    Prefill,
}

/// The typing rule shared by the fused flash-attention ops: inputs `[q, k, v, mask]`, output q's type.
/// Every consumer (the oracle, the imported kernels, the synthesized regions) indexes the operands
/// densely, with no broadcast strides: `q[B,Hq,M,D]` with `n_rep | Hq`, `k`/`v[B,Hq/n_rep,T,D]` and
/// `mask[B,Hm,M,T]` with `Hm` either 1 (one row/plane shared by every head) or `Hq` (one per head, for
/// ALiBi). Decode has `M == 1`; prefill has `B == 1` and `T == M`. A K/V or mask that the decomposition
/// would merely broadcast (a size-1 head or batch axis) is not this op.
fn infer_flash_attention(
    ins: &[TensorType],
    n_rep: usize,
    form: FlashForm,
) -> Result<TensorType, ShapeError> {
    arity(ins, 4)?;
    let (q, k, v, mask) = (&ins[0], &ins[1], &ins[2], &ins[3]);
    // Float operands only: the decomposition's softmax (`Exp`, `Div`) does not type over I32.
    if q.dtype.class() != DTypeClass::Arithmetic {
        return Err(ShapeError::DtypeOp {
            op: "FlashAttention",
            dtype: q.dtype,
        });
    }
    for (operand, t) in [("k", k), ("v", v), ("mask", mask)] {
        if t.dtype != q.dtype {
            return Err(ShapeError::FlashAttentionDtype {
                operand,
                expected: q.dtype,
                actual: t.dtype,
            });
        }
    }
    let &[bsz, hq, m, d] = q.shape.as_slice() else {
        return Err(flash_query_error(form, n_rep, q));
    };
    let form_holds = match form {
        FlashForm::Decode => m == 1,
        FlashForm::Prefill => bsz == 1,
    };
    if !form_holds || n_rep == 0 || hq % n_rep != 0 {
        return Err(flash_query_error(form, n_rep, q));
    }
    let kv_len = match (form, k.shape.get(2)) {
        (FlashForm::Decode, Some(&cap)) => cap,
        _ => m,
    };
    let kv = [bsz, hq / n_rep, kv_len, d];
    for (operand, t) in [("k", k), ("v", v)] {
        if t.shape != kv {
            return Err(ShapeError::FlashAttentionOperand {
                operand,
                expected: kv.to_vec(),
                actual: t.shape.clone(),
            });
        }
    }
    let mask_heads = if mask.shape.get(1) == Some(&hq) {
        hq
    } else {
        1
    };
    let dense_mask = [bsz, mask_heads, m, kv_len];
    if mask.shape != dense_mask {
        return Err(ShapeError::FlashAttentionOperand {
            operand: "mask",
            expected: dense_mask.to_vec(),
            actual: mask.shape.clone(),
        });
    }
    Ok(q.clone())
}

fn flash_query_error(form: FlashForm, n_rep: usize, q: &TensorType) -> ShapeError {
    ShapeError::FlashAttentionQuery {
        form: match form {
            FlashForm::Decode => "decode: M == 1",
            FlashForm::Prefill => "prefill: B == 1",
        },
        n_rep,
        actual: q.shape.clone(),
    }
}

/// The typing rule of `Rope { rot }`: inputs `[x, cos, sin]`, output x's type. `rot` is even and within
/// `1..=D`; cos/sin broadcast into the rotated slice `x[.., ..rot]` without growing it (the rotation is
/// shape-preserving, so a table that would broadcast x up to more heads or rows is not this op).
fn infer_rope(ins: &[TensorType], rot: usize) -> Result<TensorType, ShapeError> {
    arity(ins, 3)?;
    let x = &ins[0];
    // Float operands only: the decomposition's `Neg` and products do not type over I32.
    if x.dtype.class() != DTypeClass::Arithmetic {
        return Err(ShapeError::DtypeOp {
            op: "Rope",
            dtype: x.dtype,
        });
    }
    for (operand, t) in [("cos", &ins[1]), ("sin", &ins[2])] {
        if t.dtype != x.dtype {
            return Err(ShapeError::RopeDtype {
                operand,
                expected: x.dtype,
                actual: t.dtype,
            });
        }
    }
    let d = x.shape.last().copied().unwrap_or(0);
    if rot == 0 || !rot.is_multiple_of(2) || rot > d {
        return Err(ShapeError::RopeWidth {
            rot,
            x: x.shape.clone(),
        });
    }
    let mut rotated = x.shape.clone();
    *rotated.last_mut().expect("rot <= D implies rank >= 1") = rot;
    for (operand, t) in [("cos", &ins[1]), ("sin", &ins[2])] {
        if broadcast_shapes(&rotated, &t.shape).as_ref() != Some(&rotated) {
            return Err(ShapeError::RopeTable {
                operand,
                rotated,
                actual: t.shape.clone(),
            });
        }
    }
    Ok(x.clone())
}

/// [`OpKind::SampleToken`]'s shape/dtype admission (SC-008): the operand list and arity are fixed by
/// `rule` (R472-007, deval.md section 8.1): `logits` only for [`SampleRule::Greedy`]; `logits, noise,
/// params` for [`SampleRule::Gumbel`]; `logits, noise, params, top_k` for the two `TopK` rules.
/// `logits` is F32 (every other dtype is refused: the oracle and every lowering are F32-only); `noise`
/// matches `logits`' shape exactly; `params`' leading dims match `logits`' leading (all but the last)
/// axis with a trailing `3` column (`inv_temp, floor_offset, noise_scale`), widened to `4` for
/// [`SampleRule::GumbelTopKTopP`] (`+ top_p`); `top_k` is I32, shaped like `logits`' leading dims (one
/// scalar per row). Output `[.., 2]` I32 (`token, non_finite_index`), leading dims from `logits`.
fn infer_sample_token(ins: &[TensorType], rule: SampleRule) -> Result<TensorType, ShapeError> {
    let expected_arity = match rule {
        SampleRule::Greedy => 1,
        SampleRule::Gumbel => 3,
        SampleRule::GumbelTopK | SampleRule::GumbelTopKTopP => 4,
    };
    arity(ins, expected_arity)?;
    let logits = &ins[0];
    if logits.dtype != DType::F32 {
        return Err(ShapeError::SampleTokenDtype {
            operand: "logits",
            expected: DType::F32,
            actual: logits.dtype,
        });
    }
    if logits.rank() == 0 {
        return Err(ShapeError::Arity {
            expected: 1,
            got: 0,
        });
    }
    let leading = &logits.shape[..logits.rank() - 1];
    if !matches!(rule, SampleRule::Greedy) {
        let noise = &ins[1];
        if noise.dtype != DType::F32 {
            return Err(ShapeError::SampleTokenDtype {
                operand: "noise",
                expected: DType::F32,
                actual: noise.dtype,
            });
        }
        if noise.shape != logits.shape {
            return Err(ShapeError::SampleTokenShape {
                operand: "noise",
                expected: logits.shape.clone(),
                actual: noise.shape.clone(),
            });
        }
        let params = &ins[2];
        if params.dtype != DType::F32 {
            return Err(ShapeError::SampleTokenDtype {
                operand: "params",
                expected: DType::F32,
                actual: params.dtype,
            });
        }
        let params_cols = if matches!(rule, SampleRule::GumbelTopKTopP) {
            4
        } else {
            3
        };
        let mut expected_params_shape = leading.to_vec();
        expected_params_shape.push(params_cols);
        if params.shape != expected_params_shape {
            return Err(ShapeError::SampleTokenShape {
                operand: "params",
                expected: expected_params_shape,
                actual: params.shape.clone(),
            });
        }
    }
    if matches!(rule, SampleRule::GumbelTopK | SampleRule::GumbelTopKTopP) {
        let top_k = &ins[3];
        if top_k.dtype != DType::I32 {
            return Err(ShapeError::SampleTokenDtype {
                operand: "top_k",
                expected: DType::I32,
                actual: top_k.dtype,
            });
        }
        if top_k.shape != leading {
            return Err(ShapeError::SampleTokenShape {
                operand: "top_k",
                expected: leading.to_vec(),
                actual: top_k.shape.clone(),
            });
        }
    }
    let mut out_shape = leading.to_vec();
    out_shape.push(2);
    Ok(TensorType::new(out_shape, DType::I32))
}

fn arity(ins: &[TensorType], expected: usize) -> Result<(), ShapeError> {
    if ins.len() != expected {
        return Err(ShapeError::Arity {
            expected,
            got: ins.len(),
        });
    }
    Ok(())
}

fn check_axis(axis: usize, rank: usize) -> Result<(), ShapeError> {
    if axis >= rank {
        return Err(ShapeError::Axis { axis, rank });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DType;
    use poot_quant::format::ScaleEncoding;
    use poot_quant::{OperandRole, PackedWeight, SourceRole};

    fn f(shape: &[usize]) -> TensorType {
        TensorType::f32(shape.to_vec())
    }

    /// Card 552 (R-552-2): `kind_name` ignores parameters (two `Reduce`s with different axes share
    /// one key) and distinguishes different variants (a `Reduce` and a `MatMul` never share a key),
    /// independent of any planner kernel key.
    #[test]
    fn kind_name_ignores_params_and_distinguishes_variants() {
        let reduce_ax0 = OpKind::Reduce {
            op: RedOp::Sum,
            axis: 0,
            keepdim: false,
        };
        let reduce_ax1 = OpKind::Reduce {
            op: RedOp::Max,
            axis: 1,
            keepdim: true,
        };
        assert_eq!(reduce_ax0.kind_name(), reduce_ax1.kind_name());
        assert_eq!(reduce_ax0.kind_name(), "reduce");
        assert_ne!(reduce_ax0.kind_name(), OpKind::MatMul.kind_name());
        assert_eq!(OpKind::MatMul.kind_name(), "matmul");
        // Every call is the same `&'static str` pointer: no per-call formatting/allocation.
        assert_eq!(
            OpKind::MatMul.kind_name().as_ptr(),
            OpKind::MatMul.kind_name().as_ptr()
        );
    }

    /// Card 529 (R466-010, SC-002): `infer` rejects an index-role operand outside
    /// [`DType::is_index_operand`] (F32, I32). The baseline (before Card 529) returned `Ok` for `Gather`
    /// and `IndexedMatMul` with a BF16 index; both are typed errors now.
    #[test]
    fn gather_and_indexed_matmul_reject_a_bf16_index() {
        let err = OpKind::Gather { axis: 0 }
            .infer(&[f(&[4, 5]), TensorType::new(vec![2], DType::BF16)])
            .expect_err("Gather with a BF16 index must be rejected");
        assert!(matches!(
            err,
            ShapeError::DtypeOp {
                op: "Gather",
                dtype: DType::BF16
            }
        ));

        let err = OpKind::IndexedMatMul
            .infer(&[
                f(&[2, 3]),
                f(&[4, 3, 5]),
                TensorType::new(vec![2], DType::BF16),
            ])
            .expect_err("IndexedMatMul with a BF16 idx must be rejected");
        assert!(matches!(
            err,
            ShapeError::DtypeOp {
                op: "IndexedMatMul",
                dtype: DType::BF16
            }
        ));
    }

    #[test]
    fn matmul_batch_broadcast() {
        let out = OpKind::MatMul
            .infer(&[f(&[1, 14, 1, 64]), f(&[1, 14, 64, 10])])
            .unwrap();
        assert_eq!(out.shape, vec![1, 14, 1, 10]);
        // linear: [1,1,896] x [896,896] -> [1,1,896]
        let out = OpKind::MatMul
            .infer(&[f(&[1, 1, 896]), f(&[896, 896])])
            .unwrap();
        assert_eq!(out.shape, vec![1, 1, 896]);
    }

    /// Card 380 FR-001. The weight is in checkpoint `[N, K]` order, so the contraction axis is its last one:
    /// `[.., M, K] x [N, K] -> [.., M, N]`. A plain `MatMul` would need `[K, N]`, the transpose this operation
    /// exists to avoid materializing.
    #[test]
    fn dense_contraction_infers_over_a_checkpoint_order_weight() {
        for dtype in [DType::BF16, DType::F32, DType::F16] {
            let op = OpKind::DenseContraction { weight: dtype };
            let weight = TensorType::new(vec![7, 5], dtype);
            assert_eq!(
                op.infer(&[f(&[1, 3, 5]), weight.clone()]),
                Ok(f(&[1, 3, 7])),
                "{dtype:?}"
            );
            // The same operand pair read in contraction order would not type-check as a MatMul: these are
            // different equations over the same buffer.
            assert!(OpKind::MatMul.infer(&[f(&[1, 3, 5]), weight]).is_err());
        }
    }

    /// Card 380 FR-002. The admitted weight dtype is a TABLE, and `infer` is the only gate on it.
    #[test]
    fn dense_contraction_rejects_an_unadmitted_weight_dtype() {
        for dtype in [DType::I32, DType::I8, DType::E4M3FN] {
            assert!(
                !DENSE_CONTRACTION_WEIGHT_DTYPES.contains(&dtype),
                "{dtype:?} is in the table; this test needs an unadmitted dtype"
            );
            let op = OpKind::DenseContraction { weight: dtype };
            assert!(matches!(
                op.infer(&[f(&[1, 3, 5]), TensorType::new(vec![7, 5], dtype)]),
                Err(ShapeError::DenseContractionWeightDtype { actual }) if actual == dtype
            ));
        }
    }

    /// Card 381 FR-001. The row gather selects rows the way `Gather { axis: 0 }` does and widens them, so
    /// its output is F32 whatever the table stores: `[V, R] x [S..] -> [S.., R]`.
    #[test]
    fn dense_row_gather_infers_widened_rows() {
        let op = OpKind::DenseRowGather {
            source: DType::BF16,
        };
        let table = TensorType::new(vec![7, 5], DType::BF16);
        let index = TensorType::new(vec![3], DType::I32);
        assert_eq!(op.infer(&[table.clone(), index]), Ok(f(&[3, 5])));
        // A scalar index is the decode step: one row, no leading axis.
        let scalar = TensorType::scalar(DType::I32);
        assert_eq!(op.infer(&[table, scalar]), Ok(f(&[5])));
    }

    /// Card 381 FR-002. The admitted source dtype is a TABLE, and `infer` is the only gate on it.
    #[test]
    fn dense_row_gather_rejects_an_unadmitted_source_dtype() {
        for dtype in [DType::F32, DType::F16, DType::I32, DType::I8] {
            assert!(
                !DENSE_ROW_GATHER_SOURCE_DTYPES.contains(&dtype),
                "{dtype:?} is in the table; this test needs an unadmitted dtype"
            );
            let op = OpKind::DenseRowGather { source: dtype };
            assert!(matches!(
                op.infer(&[
                    TensorType::new(vec![7, 5], dtype),
                    TensorType::new(vec![3], DType::I32)
                ]),
                Err(ShapeError::DenseRowGatherSourceDtype { actual }) if actual == dtype
            ));
        }
    }

    /// Card 381 FR-002. The index dtype is pinned to I32 rather than inherited from the f32-token
    /// convention, because the wgpu typed walk stores an I32 value as `Ty::I32` and the kernel reads it
    /// there.
    #[test]
    fn dense_row_gather_rejects_a_non_i32_index() {
        let op = OpKind::DenseRowGather {
            source: DType::BF16,
        };
        let table = TensorType::new(vec![7, 5], DType::BF16);
        for dtype in [DType::F32, DType::BF16, DType::I8] {
            assert!(matches!(
                op.infer(&[table.clone(), TensorType::new(vec![3], dtype)]),
                Err(ShapeError::DenseRowGatherDtype { operand: "index", actual }) if actual == dtype
            ));
        }
    }

    /// Card 381 FR-002. The table is `[V, R]`; any other rank has no row width to gather.
    #[test]
    fn dense_row_gather_rejects_a_rank_1_table() {
        let op = OpKind::DenseRowGather {
            source: DType::BF16,
        };
        let index = TensorType::new(vec![3], DType::I32);
        assert!(matches!(
            op.infer(&[TensorType::new(vec![7], DType::BF16), index.clone()]),
            Err(ShapeError::DenseRowGatherTableRank { .. })
        ));
        assert!(matches!(
            op.infer(&[TensorType::new(vec![7, 5, 2], DType::BF16), index]),
            Err(ShapeError::DenseRowGatherTableRank { .. })
        ));
        // A mismatched declared source is rejected by name too, so the operation cannot lie about the storage its
        // kernel reads.
        assert!(matches!(
            OpKind::DenseRowGather {
                source: DType::BF16
            }
            .infer(&[
                TensorType::f32(vec![7, 5]),
                TensorType::new(vec![3], DType::I32)
            ]),
            Err(ShapeError::DenseRowGatherDtype {
                operand: "table",
                actual: DType::F32
            })
        ));
    }

    /// Card 405 FR-001. `DType::E4M3FN` is a second admitted `DenseRowGather` source through the same table-driven
    /// gate BF16 uses: no new `OpKind` variant or hand-written branch.
    #[test]
    fn dense_row_gather_e4m3_source_is_admitted() {
        assert!(DENSE_ROW_GATHER_SOURCE_DTYPES.contains(&DType::E4M3FN));
        let op = OpKind::DenseRowGather {
            source: DType::E4M3FN,
        };
        let table = TensorType::new(vec![2, 5], DType::E4M3FN);
        let index = TensorType::new(vec![1], DType::I32);
        assert_eq!(op.infer(&[table, index]), Ok(f(&[1, 5])));
    }

    /// Card 405 FR-004/SC-004. `DENSE_ROW_GATHER_SOURCE_DTYPES` is a `pub const` slice and cannot be un-compiled at
    /// test time, so this test cannot make `DType::E4M3FN` unadmitted. It checks that the admission gate at
    /// `infer`'s `DenseRowGather` arm is one table-driven line for every dtype, not a per-dtype branch: it asserts
    /// the exact variant and field for `DType::F32` (still unadmitted). The same line would raise
    /// `DenseRowGatherSourceDtype { actual: DType::E4M3FN }` if the E4M3FN row were removed.
    #[test]
    fn dense_row_gather_missing_e4m3_row_rejects_by_name() {
        let unadmitted = DType::F32;
        assert!(!DENSE_ROW_GATHER_SOURCE_DTYPES.contains(&unadmitted));
        let op = OpKind::DenseRowGather { source: unadmitted };
        assert!(matches!(
            op.infer(&[
                TensorType::new(vec![2, 5], unadmitted),
                TensorType::new(vec![1], DType::I32),
            ]),
            Err(ShapeError::DenseRowGatherSourceDtype { actual }) if actual == unadmitted
        ));
    }

    /// Card 380 FR-001. Every other operand condition is rejected by name, so a malformed contraction can
    /// never reach a planner or a kernel.
    #[test]
    fn dense_contraction_rejects_malformed_operands() {
        let op = OpKind::DenseContraction {
            weight: DType::BF16,
        };
        let weight = TensorType::new(vec![7, 5], DType::BF16);
        // A narrowed activation: the card 376 definition keeps it F32 and never rounds it.
        assert!(matches!(
            op.infer(&[TensorType::new(vec![1, 3, 5], DType::BF16), weight.clone()]),
            Err(ShapeError::DenseContractionDtype {
                operand: "activation",
                ..
            })
        ));
        // A weight whose declared dtype disagrees with the operation's own table row.
        assert!(matches!(
            op.infer(&[f(&[1, 3, 5]), f(&[7, 5])]),
            Err(ShapeError::DenseContractionDtype {
                operand: "weight",
                ..
            })
        ));
        // A batched weight: the kernel indexes one `[N, K]` matrix.
        assert!(matches!(
            op.infer(&[f(&[1, 3, 5]), TensorType::new(vec![2, 7, 5], DType::BF16)]),
            Err(ShapeError::DenseContractionWeightRank { .. })
        ));
        // A contraction-axis mismatch.
        assert!(matches!(
            op.infer(&[f(&[1, 3, 4]), weight.clone()]),
            Err(ShapeError::DenseContractionActivation { expected_k: 5, .. })
        ));
        // A rank-1 activation has no M axis.
        assert!(matches!(
            op.infer(&[f(&[5]), weight.clone()]),
            Err(ShapeError::DenseContractionActivation { .. })
        ));
        assert!(op.infer(&[f(&[1, 3, 5])]).is_err());
    }

    #[test]
    fn packed_dequant_inference_uses_two_source_packed_formats() {
        let formats = [
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::Bf16,
            },
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::F32,
            },
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::E8m0,
            },
            WeightFormat::E2m1Row32,
        ];
        for (index, format) in formats.into_iter().enumerate() {
            let logical = if index % 2 == 0 { [2, 256] } else { [3, 129] };
            let descriptor = PackedWeight::try_new(format, logical).unwrap();
            let inputs = [
                TensorType::new(
                    descriptor
                        .source_shape(SourceRole::Planar(OperandRole::Codes))
                        .to_vec(),
                    DType::I8,
                ),
                TensorType::new(
                    descriptor
                        .source_shape(SourceRole::Planar(OperandRole::Scale))
                        .to_vec(),
                    DType::I8,
                ),
            ];
            assert_eq!(
                OpKind::PackedDequant { descriptor }.infer(&inputs),
                Ok(TensorType::f32(logical.to_vec()))
            );
            let mut wrong = inputs.clone();
            wrong[1].shape[1] += 1;
            assert!(matches!(
                OpKind::PackedDequant { descriptor }.infer(&wrong),
                Err(ShapeError::PackedDequantOperand {
                    role: SourceRole::Planar(OperandRole::Scale),
                    field: "shape",
                    ..
                })
            ));

            let mut malformed = Vec::new();
            let mut weight_dtype = inputs.clone();
            weight_dtype[0].dtype = DType::F32;
            malformed.push((
                SourceRole::Planar(OperandRole::Codes),
                "dtype",
                weight_dtype,
            ));
            let mut weight_shape = inputs.clone();
            weight_shape[0].shape[1] += 1;
            malformed.push((
                SourceRole::Planar(OperandRole::Codes),
                "shape",
                weight_shape,
            ));
            let mut scale_dtype = inputs.clone();
            scale_dtype[1].dtype = DType::F32;
            malformed.push((SourceRole::Planar(OperandRole::Scale), "dtype", scale_dtype));
            for (expected_role, expected_field, malformed_inputs) in malformed {
                assert!(matches!(
                    OpKind::PackedDequant { descriptor }.infer(&malformed_inputs),
                    Err(ShapeError::PackedDequantOperand { role, field, .. })
                        if role == expected_role && field == expected_field
                ));
            }
            assert_eq!(
                OpKind::PackedDequant { descriptor }.infer(&inputs[..1]),
                Err(ShapeError::Arity {
                    expected: 2,
                    got: 1,
                })
            );
        }
    }

    /// SC-002: `PackedDequant::infer` has no positional slots - operand `i` is checked against
    /// `descriptor.sources()[i]`'s role, so a block format (one `Blocks` source), a registered
    /// E4M3/E2M1 cell (`Codes`, `Scale`), and GPTQ/AWQ (`Codes`, `Zero`, `Scale`, and act-order's
    /// `GroupIndex`) all infer through the same code with the arity their own descriptor names - not
    /// a hardcoded two. Mutation: drop the per-role loop and go back to reading `ins[0]`/`ins[1]` as
    /// `Codes`/`Scale` unconditionally; a one-source block format's single operand would then be
    /// checked against `Codes`'s shape instead of `Blocks`'s and this row goes red (a block format's
    /// `Blocks` source shape differs from any planar `Codes` shape it could be confused with).
    #[test]
    fn packed_dequant_inference_has_no_positional_slots() {
        use poot_quant::format::{GroupMap, WeightFormat};

        fn nonzero(value: usize) -> std::num::NonZeroUsize {
            std::num::NonZeroUsize::new(value).unwrap()
        }

        for (format, shape) in [
            (WeightFormat::Q4_0, [3, 64]),
            (WeightFormat::Q6_K, [2, 256]),
            (
                WeightFormat::Gptq {
                    groups: GroupMap::Contiguous { size: nonzero(8) },
                },
                [8, 16],
            ),
            (
                WeightFormat::Gptq {
                    groups: GroupMap::Indexed { groups: nonzero(2) },
                },
                [8, 16],
            ),
            (
                WeightFormat::Awq {
                    group_size: nonzero(8),
                },
                [8, 16],
            ),
        ] {
            let descriptor = PackedWeight::try_new(format, shape).unwrap();
            let roles = descriptor.sources();
            let inputs: Vec<TensorType> = roles
                .iter()
                .map(|&role| TensorType::new(descriptor.source_shape(role).to_vec(), DType::I8))
                .collect();
            assert_eq!(
                OpKind::PackedDequant { descriptor }.infer(&inputs),
                Ok(TensorType::f32(shape.to_vec())),
                "{format:?}"
            );
            assert_eq!(
                OpKind::PackedDequant { descriptor }.infer(&inputs[..inputs.len() - 1]),
                Err(ShapeError::Arity {
                    expected: roles.len(),
                    got: roles.len() - 1,
                }),
                "{format:?}"
            );
            // Mutating the dtype of the LAST operand must name that operand's own role, not the
            // first role a hardcoded two-slot reading would report.
            let mut wrong_dtype = inputs.clone();
            let last = wrong_dtype.len() - 1;
            wrong_dtype[last].dtype = DType::F32;
            assert_eq!(
                OpKind::PackedDequant { descriptor }.infer(&wrong_dtype),
                Err(ShapeError::PackedDequantOperand {
                    role: roles[last],
                    field: "dtype",
                    expected: Box::new(inputs[last].clone()),
                    actual: Box::new(wrong_dtype[last].clone()),
                }),
                "{format:?}"
            );
        }
    }

    #[test]
    fn packed_contraction_requires_f32_activation_and_output() {
        let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [3, 35]).unwrap();
        let carriers = [
            TensorType::new(
                descriptor
                    .source_shape(SourceRole::Planar(OperandRole::Codes))
                    .to_vec(),
                DType::I8,
            ),
            TensorType::new(
                descriptor
                    .source_shape(SourceRole::Planar(OperandRole::Scale))
                    .to_vec(),
                DType::I8,
            ),
        ];
        let operation = OpKind::PackedContraction {
            descriptor,
            blocks: 1,
        };
        let output = operation
            .infer(&[
                TensorType::f32(vec![2, 35]),
                carriers[0].clone(),
                carriers[1].clone(),
            ])
            .unwrap();
        assert_eq!(output, TensorType::f32(vec![2, 3]));
        for dtype in [DType::BF16, DType::F16, DType::I32, DType::I8] {
            assert_eq!(
                operation.infer(&[
                    TensorType::new(vec![2, 35], dtype),
                    carriers[0].clone(),
                    carriers[1].clone(),
                ]),
                Err(ShapeError::PackedContractionDtype {
                    operand: "activation",
                    actual: dtype,
                })
            );
        }
    }

    #[test]
    fn packed_contraction_blocks_require_rank_three_activation() {
        let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [8, 35]).unwrap();
        let carriers = [
            TensorType::new(
                descriptor
                    .source_shape(SourceRole::Planar(OperandRole::Codes))
                    .to_vec(),
                DType::I8,
            ),
            TensorType::new(
                descriptor
                    .source_shape(SourceRole::Planar(OperandRole::Scale))
                    .to_vec(),
                DType::I8,
            ),
        ];
        let blocked = OpKind::PackedContraction {
            descriptor,
            blocks: 4,
        };
        // out=8, blocks=4 -> block_out=2. A well-formed [blocks, m, k] activation infers
        // [blocks, m, block_out].
        let output = blocked
            .infer(&[
                TensorType::f32(vec![4, 5, 35]),
                carriers[0].clone(),
                carriers[1].clone(),
            ])
            .unwrap();
        assert_eq!(output, TensorType::f32(vec![4, 5, 2]));

        // Rank 2 (the dense shape) is rejected once blocks > 1: the leading block axis is required.
        assert!(matches!(
            blocked.infer(&[
                TensorType::f32(vec![5, 35]),
                carriers[0].clone(),
                carriers[1].clone(),
            ]),
            Err(ShapeError::PackedContractionActivation { expected_k: 35, .. })
        ));
        // A leading extent that disagrees with `blocks` is rejected too.
        assert!(matches!(
            blocked.infer(&[
                TensorType::f32(vec![3, 5, 35]),
                carriers[0].clone(),
                carriers[1].clone(),
            ]),
            Err(ShapeError::PackedContractionActivation { expected_k: 35, .. })
        ));

        let indivisible = OpKind::PackedContraction {
            descriptor,
            blocks: 3,
        };
        assert_eq!(
            indivisible.infer(&[
                TensorType::f32(vec![3, 5, 35]),
                carriers[0].clone(),
                carriers[1].clone(),
            ]),
            Err(ShapeError::PackedContractionBlocks { out: 8, blocks: 3 })
        );
        let zero_blocks = OpKind::PackedContraction {
            descriptor,
            blocks: 0,
        };
        assert_eq!(
            zero_blocks.infer(&[
                TensorType::f32(vec![5, 35]),
                carriers[0].clone(),
                carriers[1].clone(),
            ]),
            Err(ShapeError::PackedContractionBlocks { out: 8, blocks: 0 })
        );
    }

    #[test]
    fn e4m3fn_cast_preserves_logical_shape() {
        let input = TensorType::new(vec![2, 5], DType::F32);
        let stored = OpKind::Cast { to: DType::E4M3FN }
            .infer(std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(stored, TensorType::new(vec![2, 5], DType::E4M3FN));
        let widened = OpKind::Cast { to: DType::F32 }
            .infer(std::slice::from_ref(&stored))
            .unwrap();
        assert_eq!(widened, input);
    }

    #[test]
    fn reshape_rejects_input_or_output_element_count_overflow_without_panicking() {
        let overflowing = vec![usize::MAX, 2];
        assert!(matches!(
            OpKind::Reshape {
                shape: vec![1]
            }
            .infer(&[f(&overflowing)]),
            Err(ShapeError::ReshapeElementCountOverflow { shape }) if shape == overflowing
        ));
        assert!(matches!(
            OpKind::Reshape {
                shape: overflowing.clone()
            }
            .infer(&[f(&[1])]),
            Err(ShapeError::ReshapeElementCountOverflow { shape }) if shape == overflowing
        ));
    }

    #[test]
    fn matmul_contract_mismatch() {
        let e = OpKind::MatMul
            .infer(&[f(&[1, 1, 896]), f(&[897, 896])])
            .unwrap_err();
        assert!(matches!(e, ShapeError::MatMulContract { .. }));
    }

    /// Card 623 (R466-011 follow-up, SC-002): `MatMul` rejects a storage-only (`E4M3FN`) or packed
    /// (`I8`) operand the same way `Unary`/`Binary`/`Reduce` already do (Card 529). The baseline (before
    /// this card) accepted either dtype from `MatMul`, per the 529 review's probe.
    #[test]
    fn matmul_rejects_a_storage_only_or_packed_operand() {
        for bad_dtype in [DType::E4M3FN, DType::I8] {
            let bad = TensorType::new(vec![2, 3], bad_dtype);
            assert!(matches!(
                OpKind::MatMul.infer(&[bad.clone(), f(&[3, 4])]),
                Err(ShapeError::DtypeOp { op: "MatMul", dtype }) if dtype == bad_dtype
            ));
            assert!(matches!(
                OpKind::MatMul.infer(&[f(&[2, 3]), bad]),
                Err(ShapeError::DtypeOp { op: "MatMul", dtype }) if dtype == bad_dtype
            ));
        }
    }

    /// Card 623 (R466-011, SC-001): `MatMul` accepts same-arithmetic-dtype operands and the one stated
    /// mixed exception (F32 activation x BF16/F16 weight, per `matmul_operand_dtypes_allowed`'s doc
    /// comment), and rejects every other pairing with a typed error naming both dtypes. The baseline
    /// (before this card) followed operand A's dtype with no comparison at all, so a mismatched pair like
    /// F32 x BF16 (in either direction) or F16 x BF16 was silently accepted.
    #[test]
    fn matmul_accepts_only_the_closed_operand_dtype_set() {
        // Allowed: same arithmetic dtype.
        for dt in [DType::F32, DType::BF16, DType::F16] {
            assert!(
                OpKind::MatMul
                    .infer(&[
                        TensorType::new(vec![2, 3], dt),
                        TensorType::new(vec![3, 4], dt)
                    ])
                    .is_ok(),
                "{dt} x {dt} must be allowed"
            );
        }
        // Allowed: the stated F32-activation x BF16/F16-weight exception (fold_dense_contractions,
        // to_f16, poot_graph_plan::widen_mismatched_matmul_dtypes).
        for weight_dt in [DType::BF16, DType::F16] {
            assert!(
                OpKind::MatMul
                    .infer(&[f(&[2, 3]), TensorType::new(vec![3, 4], weight_dt)])
                    .is_ok(),
                "F32 x {weight_dt} must be allowed"
            );
        }
        // Rejected: the exception does not run in reverse, and unrelated arithmetic pairs are not
        // silently widened either.
        let rejected = [
            (DType::BF16, DType::F32), // reverse of the stated exception
            (DType::F16, DType::BF16),
            (DType::I32, DType::F32),
        ];
        for (a_dt, b_dt) in rejected {
            assert!(
                matches!(
                    OpKind::MatMul.infer(&[
                        TensorType::new(vec![2, 3], a_dt),
                        TensorType::new(vec![3, 4], b_dt)
                    ]),
                    Err(ShapeError::MatMulOperandDtype { a, b }) if a == a_dt && b == b_dt
                ),
                "{a_dt} x {b_dt} must be rejected"
            );
        }
    }

    #[test]
    fn flash_attention_decode_rejects_a_kv_or_mask_dtype_mismatch() {
        let op = OpKind::FlashAttentionDecode {
            n_rep: 1,
            scale: 1.0,
        };
        let (q, k, v, mask) = (
            f(&[1, 1, 1, 4]),
            f(&[1, 1, 3, 4]),
            f(&[1, 1, 3, 4]),
            f(&[1, 1, 1, 3]),
        );
        assert!(
            op.infer(&[q.clone(), k.clone(), v.clone(), mask.clone()])
                .is_ok()
        );

        let bf16_k = TensorType::new(vec![1, 1, 3, 4], DType::BF16);
        assert!(matches!(
            op.infer(&[q.clone(), bf16_k, v.clone(), mask.clone()]),
            Err(ShapeError::FlashAttentionDtype {
                operand: "k",
                expected: DType::F32,
                actual: DType::BF16
            })
        ));

        let bf16_v = TensorType::new(vec![1, 1, 3, 4], DType::BF16);
        assert!(matches!(
            op.infer(&[q.clone(), k.clone(), bf16_v, mask.clone()]),
            Err(ShapeError::FlashAttentionDtype {
                operand: "v",
                expected: DType::F32,
                actual: DType::BF16
            })
        ));

        let bf16_mask = TensorType::new(vec![1, 1, 1, 3], DType::BF16);
        assert!(matches!(
            op.infer(&[q, k, v, bf16_mask]),
            Err(ShapeError::FlashAttentionDtype {
                operand: "mask",
                expected: DType::F32,
                actual: DType::BF16
            })
        ));
    }

    /// Card 623 (R466-011 follow-up, SC-004): `Rope` rejects a dtype mismatch between `x` and
    /// `cos`/`sin`, naming the mismatched operand. The baseline (before this card) accepted any dtype
    /// pairing since `infer_rope` never checked operand dtypes at all.
    #[test]
    fn rope_rejects_a_cos_or_sin_dtype_mismatch() {
        let op = OpKind::Rope { rot: 4 };
        let (x, cos, sin) = (f(&[1, 1, 1, 4]), f(&[4]), f(&[4]));
        assert!(op.infer(&[x.clone(), cos.clone(), sin.clone()]).is_ok());

        let bf16_cos = TensorType::new(vec![4], DType::BF16);
        assert!(matches!(
            op.infer(&[x.clone(), bf16_cos, sin.clone()]),
            Err(ShapeError::RopeDtype {
                operand: "cos",
                expected: DType::F32,
                actual: DType::BF16
            })
        ));

        let bf16_sin = TensorType::new(vec![4], DType::BF16);
        assert!(matches!(
            op.infer(&[x, cos, bf16_sin]),
            Err(ShapeError::RopeDtype {
                operand: "sin",
                expected: DType::F32,
                actual: DType::BF16
            })
        ));
    }

    #[test]
    fn reduce_keepdim() {
        let out = OpKind::Reduce {
            op: RedOp::Sum,
            axis: 2,
            keepdim: true,
        }
        .infer(&[f(&[1, 1, 896])])
        .unwrap();
        assert_eq!(out.shape, vec![1, 1, 1]);
    }

    #[test]
    fn exact_i32_binary_inference_is_typed_and_broadcasts_normally() {
        let i = |shape: &[usize]| TensorType::new(shape.to_vec(), DType::I32);
        for op in [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::Max,
            BinOp::Ge,
            BinOp::GeU,
            BinOp::RemU,
            BinOp::And,
            BinOp::Or,
            BinOp::Xor,
            BinOp::Shl,
            BinOp::Shr,
        ] {
            let out = OpKind::Binary(op).infer(&[i(&[2, 1]), i(&[3])]).unwrap();
            assert_eq!(out, i(&[2, 3]), "{op:?}");
        }

        assert!(matches!(
            OpKind::Binary(BinOp::Div).infer(&[i(&[2]), i(&[2])]),
            Err(ShapeError::DtypeOp {
                op: "Binary(Div)",
                dtype: DType::I32
            })
        ));
        assert!(matches!(
            OpKind::Binary(BinOp::GeU).infer(&[f(&[2]), f(&[2])]),
            Err(ShapeError::DtypeOp {
                op: "Binary(GeU)",
                dtype: DType::F32
            })
        ));
        assert!(matches!(
            OpKind::Binary(BinOp::RemU).infer(&[f(&[2]), f(&[2])]),
            Err(ShapeError::DtypeOp {
                op: "Binary(RemU)",
                dtype: DType::F32
            })
        ));
        assert!(matches!(
            OpKind::Binary(BinOp::And).infer(&[f(&[2]), f(&[2])]),
            Err(ShapeError::DtypeOp {
                op: "Binary(And)",
                dtype: DType::F32
            })
        ));
        assert!(matches!(
            OpKind::Unary(UnOp::Not).infer(&[f(&[2])]),
            Err(ShapeError::DtypeOp {
                op: "Unary(Not)",
                dtype: DType::F32
            })
        ));
        assert_eq!(OpKind::Unary(UnOp::Not).infer(&[i(&[2])]).unwrap(), i(&[2]));
        assert_eq!(
            OpKind::Unary(UnOp::Clz).infer(&[i(&[2, 3])]).unwrap(),
            i(&[2, 3])
        );
        assert_eq!(
            OpKind::Select
                .infer(&[i(&[2, 1]), i(&[3]), i(&[])])
                .unwrap(),
            i(&[2, 3])
        );
        assert!(matches!(
            OpKind::Select.infer(&[f(&[2]), f(&[2]), f(&[2])]),
            Err(ShapeError::DtypeOp {
                op: "Select",
                dtype: DType::F32
            })
        ));
        assert!(matches!(
            OpKind::Binary(BinOp::Add).infer(&[i(&[2]), TensorType::scalar(DType::F32)]),
            Err(ShapeError::Dtype { .. })
        ));
        assert!(matches!(
            OpKind::Binary(BinOp::Add).infer(&[f(&[2]), i(&[2])]),
            Err(ShapeError::Dtype { .. })
        ));
        assert!(matches!(
            OpKind::Unary(UnOp::Neg).infer(&[i(&[2])]),
            Err(ShapeError::DtypeOp {
                op: "Unary",
                dtype: DType::I32
            })
        ));
        assert!(matches!(
            OpKind::Reduce {
                op: RedOp::Sum,
                axis: 0,
                keepdim: false
            }
            .infer(&[i(&[2])]),
            Err(ShapeError::DtypeOp {
                op: "Reduce",
                dtype: DType::I32
            })
        ));

        // The established float-tensor/I32-literal coercion remains valid.
        assert_eq!(
            OpKind::Binary(BinOp::Add)
                .infer(&[f(&[2]), TensorType::scalar(DType::I32)])
                .unwrap(),
            f(&[2])
        );
    }

    #[test]
    fn broadcast_rejects_rank_and_non_unit_extent_shrinking() {
        for (source, target) in [(vec![2, 3], vec![3]), (vec![2, 1], vec![1, 3])] {
            let err = OpKind::Broadcast {
                shape: target.clone(),
            }
            .infer(&[f(&source)])
            .unwrap_err();
            assert_eq!(
                err,
                ShapeError::Broadcast {
                    a: source,
                    b: target,
                }
            );
        }
    }

    #[test]
    fn broadcast_accepts_directional_expansion_edge_cases() {
        for (source, target) in [
            (vec![], vec![]),
            (vec![], vec![2, 3]),
            (vec![3], vec![2, 3]),
            (vec![2, 1], vec![2, 3]),
            (vec![1, 3, 1], vec![2, 3, 4]),
            (vec![2, 3], vec![2, 3]),
        ] {
            let out = OpKind::Broadcast {
                shape: target.clone(),
            }
            .infer(&[TensorType::new(source, DType::BF16)])
            .unwrap();
            assert_eq!(out, TensorType::new(target, DType::BF16));
        }
    }

    #[test]
    fn e4m3fn_broadcast_keeps_the_generic_directional_shape_rejections() {
        for (source, target) in [(vec![2, 3], vec![3]), (vec![2, 1], vec![1, 3])] {
            let err = OpKind::Broadcast {
                shape: target.clone(),
            }
            .infer(&[TensorType::new(source.clone(), DType::E4M3FN)])
            .unwrap_err();
            assert_eq!(
                err,
                ShapeError::Broadcast {
                    a: source,
                    b: target,
                }
            );
        }
    }

    #[test]
    fn gather_scalar_drops_axis() {
        let out = OpKind::Gather { axis: 0 }
            .infer(&[f(&[151936, 896]), TensorType::scalar(DType::I32)])
            .unwrap();
        assert_eq!(out.shape, vec![896]);
    }

    #[test]
    fn concat_sums_axis() {
        let out = OpKind::Concat { axis: 2 }
            .infer(&[f(&[1, 2, 9, 64]), f(&[1, 2, 1, 64])])
            .unwrap();
        assert_eq!(out.shape, vec![1, 2, 10, 64]);
    }

    #[test]
    fn concat_rejects_rank_mismatch_without_panicking() {
        // A later operand with rank <= axis would panic on `t.shape[axis]`; infer must return a ShapeError instead.
        // axis=2 is valid for the rank-3 first operand but out of range for the rank-2 second.
        let e = OpKind::Concat { axis: 2 }.infer(&[f(&[1, 2, 3]), f(&[1, 2])]);
        assert!(matches!(e, Err(ShapeError::Concat { .. })), "got {e:?}");
        // A rank mismatch that does not put axis out of range must also be rejected.
        let e2 = OpKind::Concat { axis: 0 }.infer(&[f(&[2, 3]), f(&[2, 3, 4])]);
        assert!(matches!(e2, Err(ShapeError::Concat { .. })), "got {e2:?}");
    }

    #[test]
    fn transpose_permutes_and_rejects_bad_perm() {
        // a valid permutation reorders the shape.
        let out = OpKind::Transpose { perm: vec![1, 0] }
            .infer(&[f(&[3, 4])])
            .unwrap();
        assert_eq!(out.shape, vec![4, 3]);
        let out3 = OpKind::Transpose {
            perm: vec![2, 0, 1],
        }
        .infer(&[f(&[3, 4, 5])])
        .unwrap();
        assert_eq!(out3.shape, vec![5, 3, 4]);
        // right length but an OUT-OF-RANGE index: must be ShapeError::Perm, not an OOB panic.
        let e = OpKind::Transpose { perm: vec![0, 2] }.infer(&[f(&[3, 4])]);
        assert!(matches!(e, Err(ShapeError::Perm { .. })), "got {e:?}");
        // right length, in range, but a DUPLICATE index: not a permutation (would drop an axis) -> Perm.
        let e2 = OpKind::Transpose { perm: vec![0, 0] }.infer(&[f(&[3, 4])]);
        assert!(matches!(e2, Err(ShapeError::Perm { .. })), "got {e2:?}");
        // wrong length is still rejected.
        let e3 = OpKind::Transpose { perm: vec![0] }.infer(&[f(&[3, 4])]);
        assert!(matches!(e3, Err(ShapeError::Perm { .. })), "got {e3:?}");
    }

    #[test]
    fn dyn_update_slice_keeps_operand_type() {
        // write a [2,1,4] slot into a [2,16,4] cache at axis 1: result is the cache's type.
        let out = OpKind::DynamicUpdateSlice { axis: 1 }
            .infer(&[
                f(&[2, 16, 4]),
                f(&[2, 1, 4]),
                TensorType::scalar(DType::I32),
            ])
            .unwrap();
        assert_eq!(out.shape, vec![2, 16, 4]);
    }

    #[test]
    fn dyn_update_slice_rejects_oversize_update() {
        // update extent on the axis exceeds the operand: error.
        let e = OpKind::DynamicUpdateSlice { axis: 1 }
            .infer(&[
                f(&[2, 16, 4]),
                f(&[2, 17, 4]),
                TensorType::scalar(DType::I32),
            ])
            .unwrap_err();
        assert!(matches!(e, ShapeError::UpdateSlice { .. }));
        // non-axis dim mismatch: error.
        let e = OpKind::DynamicUpdateSlice { axis: 1 }
            .infer(&[
                f(&[2, 16, 4]),
                f(&[3, 1, 4]),
                TensorType::scalar(DType::I32),
            ])
            .unwrap_err();
        assert!(matches!(e, ShapeError::UpdateSlice { .. }));
    }

    fn params_hash(op: &OpKind) -> u64 {
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        op.hash_params(&mut h);
        h.finish()
    }

    /// One `local[1] = op(local[0], lit)` region over a single input.
    fn fused_literal_step(op: BinOp, lit: Scalar) -> OpKind {
        OpKind::Fused(FusedRegion {
            n_inputs: 1,
            steps: vec![FusedStep {
                op: FusedOp::Binary(op),
                inputs: vec![FusedOperand::Local(0), FusedOperand::Lit(lit)],
            }],
            output: 1,
            pack: Vec::new(),
        })
    }

    /// A row region `local[1] = reduce(local[0])`, `local[2] = local[0] op local[1]`.
    fn row_region(reduce: RedOp, combine: BinOp, output: usize) -> OpKind {
        OpKind::FusedRow(RowRegion {
            n_inputs: 1,
            axis: 1,
            steps: vec![
                RowStep::Reduce {
                    op: reduce,
                    input: FusedOperand::Local(0),
                },
                RowStep::Pointwise {
                    op: FusedOp::Binary(combine),
                    inputs: vec![FusedOperand::Local(0), FusedOperand::Local(1)],
                },
            ],
            output,
        })
    }

    /// R469-005-style key correctness for the nullary `Iota`: `hash_params` fingerprints the op's content, so two
    /// iotas of different lengths must not hash alike while equal lengths do.
    #[test]
    fn iota_hash_params_distinguish_lengths() {
        use std::hash::Hasher;

        fn hash(op: &OpKind) -> u64 {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            op.hash_params(&mut hasher);
            hasher.finish()
        }

        assert_eq!(
            hash(&OpKind::Iota { len: 4 }),
            hash(&OpKind::Iota { len: 4 })
        );
        assert_ne!(
            hash(&OpKind::Iota { len: 4 }),
            hash(&OpKind::Iota { len: 5 })
        );
    }

    /// Card 528a SC-003. `hash_params` is the content fingerprint that guards the wgpu CSE memo and decode
    /// cache, so two fused regions that differ anywhere must not hash alike. The first pair is the R466-020
    /// collision: `x + 1` and `x * 2` share input count, output local and step count, which is all the baseline
    /// hashed.
    #[test]
    fn distinct_fused_regions_hash_differently() {
        let one = Scalar::F32(1.0);
        let pairs: Vec<(&str, OpKind, OpKind)> = vec![
            (
                "x+1 versus x*2 (the R466-020 pair)",
                fused_literal_step(BinOp::Add, one),
                fused_literal_step(BinOp::Mul, Scalar::F32(2.0)),
            ),
            (
                "same op, different literal",
                fused_literal_step(BinOp::Add, one),
                fused_literal_step(BinOp::Add, Scalar::F32(2.0)),
            ),
            (
                "same op, literals differ only in sign of zero",
                fused_literal_step(BinOp::Add, Scalar::F32(0.0)),
                fused_literal_step(BinOp::Add, Scalar::F32(-0.0)),
            ),
            (
                "F32 literal versus I32 literal with the same bits",
                fused_literal_step(BinOp::Add, Scalar::F32(f32::from_bits(1))),
                fused_literal_step(BinOp::Add, Scalar::I32(1)),
            ),
            (
                "pack list differs",
                OpKind::Fused(FusedRegion {
                    n_inputs: 1,
                    steps: vec![FusedStep {
                        op: FusedOp::Unary(UnOp::Neg),
                        inputs: vec![FusedOperand::Local(0)],
                    }],
                    output: 1,
                    pack: vec![1],
                }),
                OpKind::Fused(FusedRegion {
                    n_inputs: 1,
                    steps: vec![FusedStep {
                        op: FusedOp::Unary(UnOp::Neg),
                        inputs: vec![FusedOperand::Local(0)],
                    }],
                    output: 1,
                    pack: Vec::new(),
                }),
            ),
            (
                "row region, sum versus max reduction",
                row_region(RedOp::Sum, BinOp::Sub, 2),
                row_region(RedOp::Max, BinOp::Sub, 2),
            ),
            (
                "row region, different pointwise op after the reduction",
                row_region(RedOp::Sum, BinOp::Sub, 2),
                row_region(RedOp::Sum, BinOp::Div, 2),
            ),
            (
                "row region, different output local",
                row_region(RedOp::Sum, BinOp::Sub, 2),
                row_region(RedOp::Sum, BinOp::Sub, 1),
            ),
        ];
        for (what, a, b) in pairs {
            // `PartialEq` compares f32 by value (`0.0 == -0.0`), so the precondition is `Debug` text.
            assert_ne!(
                format!("{a:?}"),
                format!("{b:?}"),
                "{what}: the pair must be distinct regions"
            );
            assert_ne!(
                params_hash(&a),
                params_hash(&b),
                "{what}: distinct fused regions must not share a fingerprint"
            );
        }
    }

    #[test]
    fn equal_fused_regions_hash_equally() {
        let region = fused_literal_step(BinOp::Add, Scalar::F32(1.0));
        assert_eq!(params_hash(&region), params_hash(&region.clone()));
        let row = row_region(RedOp::Sum, BinOp::Sub, 2);
        assert_eq!(params_hash(&row), params_hash(&row.clone()));
    }

    /// SC-008 (card 551a): `RandomUniform` rejects a non-I32 seed with a typed `ShapeError`.
    #[test]
    fn random_uniform_rejects_a_non_i32_seed() {
        let err = OpKind::RandomUniform { cols: 4 }
            .infer(&[f(&[2])])
            .expect_err("a F32 seed must be rejected");
        assert!(matches!(
            err,
            ShapeError::RandomUniformSeedDtype { actual: DType::F32 }
        ));
    }

    /// SC-008: `SampleToken` rejects non-F32 logits, a mismatched-shape noise operand, and a
    /// non-I32 `top_k`, each with a typed `ShapeError`. Mutation (recorded, not left in the tree):
    /// drop the noise shape check (replace `if noise.shape != logits.shape` with `if false`) - the
    /// mismatched graph type-checks and this row goes red.
    #[test]
    fn sample_token_rejects_non_f32_logits_mismatched_noise_shape_and_non_i32_top_k() {
        let err = OpKind::SampleToken {
            rule: SampleRule::Greedy,
        }
        .infer(&[TensorType::new(vec![4], DType::I32)])
        .expect_err("I32 logits must be rejected");
        assert!(matches!(
            err,
            ShapeError::SampleTokenDtype {
                operand: "logits",
                expected: DType::F32,
                actual: DType::I32,
            }
        ));

        let err = OpKind::SampleToken {
            rule: SampleRule::Gumbel,
        }
        .infer(&[f(&[4]), f(&[5]), f(&[3])])
        .expect_err("a noise operand of another shape must be rejected");
        assert!(matches!(
            err,
            ShapeError::SampleTokenShape {
                operand: "noise",
                ..
            }
        ));

        let err = OpKind::SampleToken {
            rule: SampleRule::GumbelTopK,
        }
        .infer(&[
            f(&[4]),
            f(&[4]),
            f(&[3]),
            TensorType::new(vec![], DType::F32),
        ])
        .expect_err("a non-I32 top_k must be rejected");
        assert!(matches!(
            err,
            ShapeError::SampleTokenDtype {
                operand: "top_k",
                expected: DType::I32,
                actual: DType::F32,
            }
        ));
    }

    /// Card 551a: `SampleToken`'s output is `[.., 2]` I32, leading dims from `logits`, for every rule.
    #[test]
    fn sample_token_output_is_leading_dims_plus_two_columns_i32() {
        let out = OpKind::SampleToken {
            rule: SampleRule::Greedy,
        }
        .infer(&[f(&[3, 7])])
        .unwrap();
        assert_eq!(out, TensorType::new(vec![3, 2], DType::I32));

        let out = OpKind::SampleToken {
            rule: SampleRule::GumbelTopKTopP,
        }
        .infer(&[
            f(&[3, 7]),
            f(&[3, 7]),
            f(&[3, 4]),
            TensorType::new(vec![3], DType::I32),
        ])
        .unwrap();
        assert_eq!(out, TensorType::new(vec![3, 2], DType::I32));
    }

    /// Card 551a: `RandomUniform`'s output is the seed's shape with one trailing `cols` axis.
    #[test]
    fn random_uniform_output_is_seed_shape_plus_cols() {
        let out = OpKind::RandomUniform { cols: 5 }
            .infer(&[TensorType::new(vec![3], DType::I32)])
            .unwrap();
        assert_eq!(out, TensorType::new(vec![3, 5], DType::F32));
    }
}
