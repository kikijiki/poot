//! The primitive op set and its host-side shape/dtype inference (the `abstract_eval` analog).
//!
//! Attention, RMSNorm, RoPE, and SwiGLU are not here: they are compositions of these primitives (see
//! [`crate::ops`]). Each variant carries its by-value params (the jaxpr eqn `params`); tensor operands are passed
//! separately to `infer`.

use poot_quant::PackedWeight;
use std::num::NonZeroU32;

use crate::error::{FusedOperandReason, ScanReason, ShapeError};
use crate::types::{DType, DTypeClass, DTypeClassExt, Scalar, TensorType, broadcast_shapes};

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
    /// Negation, `-x`.
    Neg,
    /// Natural exponential, `e^x`.
    Exp,
    /// Natural log (a primitive IR op). Needed by softplus (`log(1+exp(x))`, Mamba's Delta) and log-space cumulative
    /// products. NVPTX uses the hardware `lg2.approx` scaled by `ln(2)`, like Exp.
    Log,
    /// Square root.
    Sqrt,
    /// Reciprocal, `1 / x`.
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

/// A binary elementwise primitive; operands broadcast numpy-style and share one dtype.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum BinOp {
    /// Addition.
    Add,
    /// Subtraction, `a - b`.
    Sub,
    /// Multiplication.
    Mul,
    /// Division, `a / b`.
    Div,
    /// Elementwise maximum. On floats it is max-number: a NaN operand is ignored and the other operand is the
    /// result, NaN only when both are NaN. This is `llvm.maxnum`, which the LLVM-based lowerings emit, and
    /// [`crate::ops::canonical_router_scores`] relies on it to turn NaN scores into numbers. Integer `Max` is the
    /// plain signed maximum.
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

/// The combiner of a last-axis style reduction ([`OpKind::Reduce`], [`OpKind::AllReduce`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum RedOp {
    /// Sum of the reduced elements.
    Sum,
    /// Maximum of the reduced elements, with the float max-number rule of [`BinOp::Max`]: NaN elements are
    /// ignored, so a row of only NaN reduces to the identity, negative infinity.
    Max,
}

/// Which sampler [`OpKind::SampleToken`] runs, each a wider superset of the last (card 551a, R472-007,
/// deval.md section 8). Every rule maximizes `logits[i]*inv_temp + noise_scale*noise[i]` (two roundings,
/// no fused multiply-add) over the kept index set, ties to the lowest index, after a shared row-max /
/// non-finite pass over the raw logits (R-551a-2). `noise_scale` is 0 for `Greedy` (no operand reads
/// `noise`/`params` at all) and 1 for the Gumbel-family rules (where it rides in `params`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
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

impl SampleRule {
    /// The rule a temperature draw's `top_k`/`top_p` knobs select (`top_k == 0` and `top_p >= 1.0` mean
    /// "disabled"; min-p rides in every Gumbel-family rule's `params`). The one classification the
    /// device suffix, the host sampler and the oracle's operand lists all read, so a request cannot be
    /// a different rule on two paths.
    pub const fn for_truncation(top_k: usize, top_p: f32) -> Self {
        if top_p < 1.0 {
            Self::GumbelTopKTopP
        } else if top_k > 0 {
            Self::GumbelTopK
        } else {
            Self::Gumbel
        }
    }

    /// Whether the rule truncates to the `top_k` largest logits (it declares the `top_k` operand).
    pub const fn has_top_k(self) -> bool {
        matches!(self, Self::GumbelTopK | Self::GumbelTopKTopP)
    }

    /// Whether the rule truncates to the top-p nucleus (it carries the fourth `params` column).
    pub const fn has_top_p(self) -> bool {
        matches!(self, Self::GumbelTopKTopP)
    }

    /// The number of `params` columns the rule reads: `[inv_temp, floor_offset, noise_scale]`, plus
    /// `top_p` for [`SampleRule::GumbelTopKTopP`]. `Greedy` reads no `params` operand.
    pub const fn params_cols(self) -> usize {
        match self {
            Self::Greedy => 0,
            Self::Gumbel | Self::GumbelTopK => 3,
            Self::GumbelTopKTopP => 4,
        }
    }
}

/// The `RandomUniform` value at column `idx` under `seed`: `u = (2 * (hash32(seed, idx) >> 9) + 1) *
/// 2^-24`, an exact I32->F32 conversion and a power-of-two multiply, always in `[2^-24, 1 - 2^-24]`. The
/// one definition of the per-element noise coordinate, read by the host sampler (the device kernels and
/// the oracle compute the same value; tier 1, ADR-0101). The coordinate is the column alone, so a seed
/// draws the same noise whichever row of a batch it rides in.
pub fn random_uniform_at(seed: u32, idx: u32) -> f32 {
    let mantissa = (crate::ops::sampling::hash32(seed, idx) >> 9) as i32;
    (2.0f32 * (mantissa as f32) + 1.0f32) * 5.9604645e-8f32
}

/// Standard Gumbel(0,1) noise from a uniform `u` in `(0, 1)`: `-ln(-ln(u))`, the scalar form of
/// [`crate::ops::sampling::gumbel`]'s graph composition (tier 2 against a device `log`, ADR-0114).
pub fn gumbel_noise(u: f32) -> f32 {
    -(-u.ln()).ln()
}

/// A scalar pointwise op inside a fused region (the subset G5a fuses). A composition of these over the region's
/// inputs lowers to one kernel that keeps every intermediate in registers.
#[derive(Clone, Copy, PartialEq, Debug, Hash)]
pub enum FusedOp {
    /// A [`UnOp`] step.
    Unary(UnOp),
    /// A [`BinOp`] step.
    Binary(BinOp),
    /// Pointwise `Select(cond, if_true, if_false)` with three operands: the bit pattern of `if_true` where the
    /// I32 condition word is nonzero, else of `if_false`. A select, never arithmetic: a NaN, Inf or `-0.0`
    /// in the unselected branch cannot reach the result and the selected one passes through bit for bit.
    Select,
    /// [`OpKind::Cast`] of one operand to `to`. A lowering reads a narrow (BF16, F16) region input and widens it
    /// in registers, so a norm scale or a bias that a region reads as F32 is never a separate dispatch and
    /// never an f32 copy of the stored weight.
    Cast {
        /// The dtype the operand is converted to.
        to: DType,
    },
}

impl FusedOp {
    /// The primitive op this step computes: the one definition [`FusedRegion::infer`], [`RowRegion::infer`]
    /// and [`crate::decompose`] share.
    pub fn kind(self) -> OpKind {
        match self {
            Self::Unary(op) => OpKind::Unary(op),
            Self::Binary(op) => OpKind::Binary(op),
            Self::Select => OpKind::Select,
            Self::Cast { to } => OpKind::Cast { to },
        }
    }
}

/// An operand of a [`FusedStep`]: a local value id, or an inline literal captured from the source graph. Locals
/// `0..n_inputs` are the region's external inputs (matching the fused eqn's `inputs`, all `Operand::Value`, in
/// order); locals `n_inputs + j` are the result of step `j`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FusedOperand {
    /// A local value: an external input (`0..n_inputs`) or an earlier step's result.
    Local(usize),
    /// An inline scalar literal captured from the source graph.
    Lit(Scalar),
    /// Last-axis lane of an external packed I32 input. The input tensor's last dimension is the pack
    /// width; this operand has a trailing unit axis.
    PackLane {
        /// The external packed input read.
        input: usize,
        /// The lane (index along the input's last axis) taken.
        lane: usize,
    },
}

/// One straight-line step of a fused region: `local[n_inputs + j] = op(inputs...)`.
#[derive(Clone, PartialEq, Debug)]
pub struct FusedStep {
    /// The pointwise operation this step computes.
    pub op: FusedOp,
    /// The step's operands, in the operation's operand order.
    pub inputs: Vec<FusedOperand>,
}

/// A fused elementwise region (G5): a straight-line SSA of pointwise steps over `n_inputs`
/// external inputs, produced by `poot-graph-plan`'s `fuse` pass and lowered to one synthesized kernel
/// (`poot_kernelgen::fused`) instead of one kernel per primitive. `output` is the local id of the region's
/// result (a step result, or degenerately an input). Every step's output shares the region's output shape;
/// inputs may broadcast on read.
#[derive(Clone, PartialEq, Debug)]
pub struct FusedRegion {
    /// The number of external inputs, which are locals `0..n_inputs`.
    pub n_inputs: usize,
    /// The steps in execution order; step `j` defines local `n_inputs + j`.
    pub steps: Vec<FusedStep>,
    /// The local that is the region's result.
    pub output: usize,
    /// When non-empty, the equation output is the last-axis concatenation of these locals, each of
    /// which has a trailing unit axis. Empty means the single [`Self::output`] local.
    pub pack: Vec<usize>,
}

impl FusedRegion {
    /// Whether external input `input` is read: by a step, or as the region's output.
    pub fn reads_input(&self, input: usize) -> bool {
        self.output == input
            || self.steps.iter().any(|step| {
                step.inputs
                    .iter()
                    .any(|operand| matches!(operand, FusedOperand::Local(local) if *local == input))
            })
    }

    /// Whether a [`FusedOp::Cast`] step reads external input `input`: the input a lowering widens as it reads it.
    pub fn widens_input(&self, input: usize) -> bool {
        self.steps.iter().any(|step| {
            matches!(step.op, FusedOp::Cast { .. })
                && matches!(step.inputs.as_slice(), [FusedOperand::Local(local)] if *local == input)
        })
    }

    /// The type of every local, threading the input types through the steps: the `n_inputs` external inputs
    /// first (`ins`, in order), then each step's result. Mirrors how the un-fused chain would infer.
    #[deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
    pub fn local_types(&self, ins: &[TensorType]) -> Result<Vec<TensorType>, ShapeError> {
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
                    FusedOperand::Local(id) => local_ty(&locals, *id)?.clone(),
                    FusedOperand::Lit(s) => s.ty(),
                    FusedOperand::PackLane { input, lane } => {
                        let source = ins.get(*input).ok_or(ShapeError::FusedOperand(
                            FusedOperandReason::PackLaneInput {
                                input: *input,
                                inputs: ins.len(),
                            },
                        ))?;
                        pack_lane_ty(source, *input, *lane)?
                    }
                });
            }
            let ty = step.op.kind().infer(&in_tys)?;
            locals.push(ty);
        }
        Ok(locals)
    }

    /// The output type: [`Self::local_types`] read at the output local, or at the packed lanes. `ins` are the
    /// `n_inputs` external input types, in order.
    #[deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
    pub fn infer(&self, ins: &[TensorType]) -> Result<TensorType, ShapeError> {
        let locals = self.local_types(ins)?;
        let Some(&first_local) = self.pack.first() else {
            return Ok(local_ty(&locals, self.output)?.clone());
        };
        let first = local_ty(&locals, first_local)?;
        let Some((&1, leading)) = first.shape.split_last() else {
            let mut unit = first.shape.clone();
            match unit.last_mut() {
                Some(last) => *last = 1,
                None => unit.push(1),
            }
            return Err(ShapeError::FusedPack {
                local: first_local,
                expected: Box::new(TensorType::new(unit, first.dtype)),
                actual: Box::new(first.clone()),
            });
        };
        for &id in &self.pack {
            let member = local_ty(&locals, id)?;
            if member != first {
                return Err(ShapeError::FusedPack {
                    local: id,
                    expected: Box::new(first.clone()),
                    actual: Box::new(member.clone()),
                });
            }
        }
        let mut shape = leading.to_vec();
        shape.push(self.pack.len());
        Ok(TensorType::new(shape, first.dtype))
    }
}

/// The type of fused-region local `id`, or the typed fault for a local the region does not define.
fn local_ty(locals: &[TensorType], id: usize) -> Result<&TensorType, ShapeError> {
    locals.get(id).ok_or(ShapeError::FusedOperand(
        FusedOperandReason::LocalNotDefined {
            local: id,
            defined: locals.len(),
        },
    ))
}

/// The type a [`FusedOperand::PackLane`] of external input `input` (type `source`) reads: the source with
/// its last axis reduced to the one lane.
fn pack_lane_ty(source: &TensorType, input: usize, lane: usize) -> Result<TensorType, ShapeError> {
    let mut shape = source.shape.clone();
    match shape.last_mut() {
        Some(width) if lane < *width => *width = 1,
        other => {
            return Err(ShapeError::FusedOperand(
                FusedOperandReason::PackLaneWidth {
                    input,
                    lane,
                    width: other.map_or(0, |width| *width),
                },
            ));
        }
    }
    Ok(TensorType::new(shape, source.dtype))
}

/// One step of a row-wise fused region ([`RowRegion`]): a pointwise op (a Map over the row, or a Scalar
/// op on per-row reduction results - the synthesizer tells them apart by shape) or a last-axis reduction.
#[derive(Clone, PartialEq, Debug)]
pub enum RowStep {
    /// A pointwise step.
    Pointwise {
        /// The pointwise operation.
        op: FusedOp,
        /// The operands, in the operation's operand order.
        inputs: Vec<FusedOperand>,
    },
    /// A `keepdim` reduction over the region's last (row) axis: a row `[.,N]` value -> a scalar `[.,1]`.
    Reduce {
        /// The reduction combiner.
        op: RedOp,
        /// The row value reduced.
        input: FusedOperand,
    },
}

/// A row-wise (reduction-rooted) fused region (G5b): all values are rows `[.,N]` over the reduction axis or
/// per-row scalars `[.,1]`, mixing last-axis reductions with the pointwise ops feeding and consuming them.
/// Lowered to one kernel that processes one row at a time, computing each reduction into a register and
/// recomputing the pointwise chain per pass (one-pass RMSNorm, fused softmax). The pointwise [`FusedRegion`]
/// is the no-reduction case.
#[derive(Clone, PartialEq, Debug)]
pub struct RowRegion {
    /// The number of external inputs, which are locals `0..n_inputs`.
    pub n_inputs: usize,
    /// the reduction axis (the last axis of the row shape); reductions keep it as size 1.
    pub axis: usize,
    /// The steps in execution order; step `j` defines local `n_inputs + j`.
    pub steps: Vec<RowStep>,
    /// The local that is the region's result.
    pub output: usize,
}

impl RowRegion {
    /// Whether a [`FusedOp::Cast`] step reads external input `input`: the input a lowering widens as it reads it.
    pub fn widens_input(&self, input: usize) -> bool {
        self.steps.iter().any(|step| {
            matches!(
                step,
                RowStep::Pointwise { op: FusedOp::Cast { .. }, inputs }
                    if matches!(inputs.as_slice(), [FusedOperand::Local(local)] if *local == input)
            )
        })
    }

    /// The output type, threading input types through the steps (the un-fused chain's inference).
    #[deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
    pub fn infer(&self, ins: &[TensorType]) -> Result<TensorType, ShapeError> {
        if ins.len() != self.n_inputs {
            return Err(ShapeError::Arity {
                expected: self.n_inputs,
                got: ins.len(),
            });
        }
        let mut locals: Vec<TensorType> = ins.to_vec();
        let ty_of = |o: &FusedOperand, locals: &[TensorType]| match o {
            FusedOperand::Local(id) => local_ty(locals, *id).cloned(),
            FusedOperand::Lit(s) => Ok(s.ty()),
            FusedOperand::PackLane { .. } => Err(ShapeError::FusedOperand(
                FusedOperandReason::PackLaneInRowRegion,
            )),
        };
        for step in &self.steps {
            let ty = match step {
                RowStep::Pointwise { op, inputs } => {
                    let in_tys = inputs.iter().map(|o| ty_of(o, &locals)).collect::<Result<
                        Vec<TensorType>,
                        ShapeError,
                    >>(
                    )?;
                    op.kind().infer(&in_tys)?
                }
                RowStep::Reduce { op, input } => OpKind::Reduce {
                    op: *op,
                    axis: self.axis,
                    keepdim: true,
                }
                .infer(&[ty_of(input, &locals)?])?,
            };
            locals.push(ty);
        }
        Ok(local_ty(&locals, self.output)?.clone())
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

/// How a [`OpKind::PackedContraction`] pairs activation rows with weight rows.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ContractionRows {
    /// The weight's `out` axis is split into this many contiguous, equal-width blocks, each contracted only
    /// against its own activation block (Card 385). `Blocks(1)` is the ordinary dense contraction: the activation
    /// may be any rank ending in `K`, and operands are `[activation, carriers..]`.
    Blocks(usize),
    /// A stacked table of `experts` packed weights of one descriptor, output row `m` reading expert `ids[m]`:
    /// `out[m, n] = sum over k of x[m, k] * decode(expert ids[m], n, k)`. Operands are `[x [M, K], ids [M],
    /// carriers..]`, each carrier the expert-stacked, word-padded source `[experts, slab_bytes]`
    /// ([`crate::packed_source::packed_stack_source_type`]). An id outside `0..experts`, or not an integer, makes
    /// that row NaN (R-632-4), as for [`OpKind::IndexedMatMul`].
    Indexed {
        /// The number of experts stacked in the table; row ids index `0..experts`.
        experts: usize,
    },
}

impl ContractionRows {
    /// The operands before the carriers: the activation, and the row ids for an indexed table.
    pub const fn leading_operands(self) -> usize {
        match self {
            Self::Blocks(_) => 1,
            Self::Indexed { .. } => 2,
        }
    }

    /// The block count of a block-diagonal contraction, `None` for an indexed one.
    pub const fn blocks(self) -> Option<usize> {
        match self {
            Self::Blocks(blocks) => Some(blocks),
            Self::Indexed { .. } => None,
        }
    }
}

/// The contraction an [`OpKind::ContractionEpilogue`] runs before its pointwise region: the heads the planner
/// lowers with an epilogue. Each names the primitive op whose result local 0 of the region reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ContractionHead {
    /// [`OpKind::MatMul`] over `A[.., M, K]` and `B[.., K, N]`.
    MatMul,
    /// [`OpKind::DenseContraction`] over an activation and a `[N, K]` weight of dtype `weight`.
    Dense {
        /// The weight's storage dtype, one of [`DENSE_CONTRACTION_WEIGHT_DTYPES`].
        weight: DType,
    },
    /// [`OpKind::PackedContraction`] over an f32 activation and the carriers of the packed weight
    /// `descriptor`, its rows selected as `rows` says. Its result is f32, so the head applies no
    /// narrowing before the region reads it.
    Packed {
        /// The packed weight the activation is contracted against.
        descriptor: PackedWeight,
        /// How activation rows meet weight rows.
        rows: ContractionRows,
    },
    /// [`OpKind::IndexedMatMul`] over an activation `x [M, K]`, a stacked weight `W [E, K, N]` and the per-row
    /// expert ids `idx [M]`: a routed expert matmul followed by its gate weighting.
    Indexed,
}

impl ContractionHead {
    /// The number of operands the head reads: its activation and its weight, which for a packed weight is one
    /// carrier per source role of its descriptor, and for an indexed head the weight and the row ids.
    pub fn operands(self) -> usize {
        match self {
            Self::MatMul | Self::Dense { .. } => 2,
            Self::Indexed => 3,
            Self::Packed { descriptor, rows } => {
                rows.leading_operands() + descriptor.sources().len()
            }
        }
    }

    /// The primitive op that computes the head's result.
    pub fn op(self) -> OpKind {
        match self {
            Self::MatMul => OpKind::MatMul,
            Self::Dense { weight } => OpKind::DenseContraction { weight },
            Self::Indexed => OpKind::IndexedMatMul,
            Self::Packed { descriptor, rows } => OpKind::PackedContraction { descriptor, rows },
        }
    }
}

/// The parameters of [`OpKind::Attention`]: the one attention family, whatever its phase.
///
/// `mask`, `key` and `bound` name the operand conventions the kernels read. A variant added to one is a new case
/// every consumer matches, never a flag on the others. The operand list is `[q, k, v, mask]`, then the shared key
/// part when `key` is [`KeyLayout::Split`] or the slot map `read` when it is [`KeyLayout::Paged`], then `len`
/// when `bound` is [`KvBound::Live`], then the per-head `sinks [Hq]` when `sink` is [`SinkOperand::PerHead`].
#[derive(Clone, PartialEq, Debug)]
pub struct AttentionAttrs {
    /// Query heads per key/value head (`Hq / Hkv`).
    pub n_rep: usize,
    /// Multiplier on the `Q K^T` scores.
    pub scale: f32,
    /// `Some(c)` applies `c * tanh(scores / c)` (Gemma2/Grok logit softcapping, [`crate::ops::softcap`]) to the
    /// scaled scores before the mask is added; `None` applies none.
    pub softcap: Option<f32>,
    /// How the mask operand enters the scores.
    pub mask: MaskOperand,
    /// How the key operand is laid out.
    pub key: KeyLayout,
    /// Which keys the kernel visits.
    pub bound: KvBound,
    /// Whether the softmax carries a learned per-head sink logit.
    pub sink: SinkOperand,
}

impl AttentionAttrs {
    /// Attention over a dense key tensor, an additive mask operand and every key position (the form the
    /// matcher emits).
    pub fn new(n_rep: usize, scale: f32, softcap: Option<f32>) -> Self {
        Self {
            n_rep,
            scale,
            softcap,
            mask: MaskOperand::Additive,
            key: KeyLayout::Single,
            bound: KvBound::Capacity,
            sink: SinkOperand::Absent,
        }
    }
}

/// Whether [`AttentionAttrs`]'s softmax includes a sink: one position-independent logit per query head that
/// takes part in the row maximum and the normalizer but is no key, so it adds nothing to the weighted value
/// sum (gpt-oss, DeepSeek-V4). Equivalent to a sink column concatenated to the scores with a zero value row.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SinkOperand {
    /// No sink: the softmax runs over the keys alone.
    Absent,
    /// A `[Hq]` operand, the last of the operand list, holds the sink logit of each query head, in the
    /// query's dtype.
    PerHead,
}

/// How [`AttentionAttrs`]'s mask operand enters the scores.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MaskOperand {
    /// A `[B,Hm,Lq,Lk]` tensor added to the scaled (and softcapped) scores.
    Additive,
    /// The causal mask of absolute positions, stated rather than materialized: the operand is the I32
    /// `[B,Lq]` position of each query row, and key `j` is visible to query `(b, i)` when `j <= pos[b,i]`
    /// and, under a window, `pos[b,i] - j < window`. A hidden key adds nothing to the softmax, so a kernel
    /// visits only the keys between the window's start and the position instead of reading a mask element
    /// per key. The decomposition is `ops::causal_mask_from_pos` over a key axis of `Lk`. A row whose
    /// position is negative (a padded query) sees no key there and key 0 in a kernel; its output is dropped
    /// by its caller either way.
    Causal {
        /// The most recent keys a query sees (`pos - j < window`); `None` is every earlier key. At least 1.
        window: Option<usize>,
    },
}

/// How [`AttentionAttrs`]'s key operand is laid out.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum KeyLayout {
    /// One dense `[B,Hkv,Lk,D]` tensor.
    Single,
    /// The key is two operands concatenated on the head axis: `k_head [B,Hkv,Lk,D - shared_dim]` and
    /// `k_shared [B,1,Lk,shared_dim]`, the latter broadcast over every key head (MLA's `k_nope` and the
    /// rotary `k_pe`). The shared operand follows `mask` in the operand list.
    Split {
        /// The width `s` of the shared operand's last axis; the head part is `D - s` wide.
        shared_dim: usize,
    },
    /// The key and the value are read in place from the paged KV pools they live in: `k [S,Hkv,Dk]` and
    /// `v [S,Hkv,Dv]` are the pools of `S` slots, and the I32 `read [B,Lk]` operand, which follows `mask`
    /// in the operand list, names the pool slot holding logical position `t` of batch row `b`. It is the
    /// decomposition `transpose(reshape(gather(pool, 0, read)))` of each pool, without the materialized
    /// `[B,Hkv,Lk,D]` copies. Every `read` word must be a slot of the pool; the matcher fuses only a map
    /// whose declared range proves it.
    Paged,
}

/// Which keys [`AttentionAttrs`]'s kernel visits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum KvBound {
    /// All `Lk` positions: the mask removes the ones that must not count.
    Capacity,
    /// The first `len[b]` positions of batch row `b`, `1 <= len[b] <= Lk`, read from an I32 `len [B]` operand
    /// that follows the key operands. Inactive keys and values are selected away before either matmul and
    /// inactive scores to `-inf` before softmax, so whatever they hold (NaN included) never reaches the output
    /// ([`crate::ops::attention_live`] is the one definition).
    Live,
}

/// Orthogonal tensor algebra. The whole transformer composes from these (graph-architecture.md 3.2).
#[derive(Clone, PartialEq, Debug)]
pub enum OpKind {
    /// Elementwise [`UnOp`] over one operand.
    Unary(UnOp),
    /// Elementwise [`BinOp`] over two operands with numpy broadcasting.
    Binary(BinOp),
    /// Pointwise select, `out = cond != 0 ? if_true : if_false`, as a bit-pattern select and never arithmetic:
    /// a NaN, Inf or `-0.0` in the unselected branch cannot reach `out`, and the selected one passes through bit
    /// for bit. Inputs are `cond` (I32), `if_true`, `if_false`; the branches share one I32, F32, F16 or BF16
    /// dtype, which is `out`'s. Every executor and the oracle share this one definition.
    Select,
    /// Reduce one axis with a [`RedOp`] combiner.
    Reduce {
        /// The reduction combiner.
        op: RedOp,
        /// The axis reduced.
        axis: usize,
        /// Whether the reduced axis stays as extent 1.
        keepdim: bool,
    },
    /// Broadcast the operand to `shape`: trailing axes align and extent-1 axes expand, never shrinking.
    Broadcast {
        /// The target shape.
        shape: Vec<usize>,
    },
    /// Cast a tensor to another storage dtype (same shape). f32 -> bf16 rounds to bf16 precision (bf16 is the top
    /// 16 bits of an f32); a widening/same-dtype cast is the identity. Feeds bf16 operands into the tensor-core
    /// matmul path (card 045).
    Cast {
        /// The dtype the operand is converted to.
        to: DType,
    },
    /// Reinterpret the operand's elements under a new shape of the same element count.
    Reshape {
        /// The target shape.
        shape: Vec<usize>,
    },
    /// Permute the operand's axes.
    Transpose {
        /// Output axis `i` is input axis `perm[i]`.
        perm: Vec<usize>,
    },
    /// Take `start..end` along one axis.
    Slice {
        /// The axis sliced.
        axis: usize,
        /// The first index kept.
        start: usize,
        /// One past the last index kept.
        end: usize,
    },
    /// Concatenate the operands along one axis; the operands agree on dtype and every other axis.
    Concat {
        /// The concatenation axis.
        axis: usize,
    },
    /// A nullary range `[0, 1, .., len)`, shape `[len]`, in `dtype` (F32 or I32): the explicit iota primitive that
    /// replaces the reserved named per-expert range graph constants (R466-019). Ranges are computed from
    /// this equation, not bound by a checkpoint binder. The compiler folds every `Iota` into a
    /// [`crate::Storage::Computed`] graph constant ([`crate::transform::fold_iota`]) before planning, so
    /// the planner never lowers an iota kernel; an unfolded `Iota` at the planner is a typed refusal.
    /// An I32 range is exact for every `len` an `i32` holds; an F32 one only below 2^24 (any other `dtype`,
    /// or an I32 `len` above `i32::MAX`, is a shape error).
    Iota {
        /// The number of elements.
        len: usize,
        /// The element dtype.
        dtype: DType,
    },
    /// Gather rows of `data` (input 0) along `axis` by an index (input 1). Scalar-index case only for now
    /// (embedding lookup, RoPE cos/sin row): drops `axis` from the result.
    Gather {
        /// The axis the rows are taken along.
        axis: usize,
    },
    /// Scatter rows of `src` (input 0) along `axis` by a `[N]` permutation index (input 1), the inverse of an
    /// axis-0 [`OpKind::Gather`]: `out[index[j], ..] = src[j, ..]`, output the same shape as `src`. `index` must be a
    /// permutation of `0..N` (N = `src.shape[axis]`); used for argtop-k expert indices in sparse MoE dispatch
    /// (card 034: scatter `iota` by the per-expert rank to invert it). Axis 0 only.
    Scatter {
        /// The axis scattered along; axis 0 only.
        axis: usize,
    },
    /// Scatter-update rows into a base (the multi-token paged KV write): `out = base`, then
    /// `out[slots[t], ..] = src[t, ..]` for each source row `t` whose `slots[t] >= 0`. Inputs are
    /// `base [POOL, ..rest]`, `src [N, ..rest]`, and the forward map `slots [N]` (input 2, I32: the pool
    /// slot row `t` is written to, or `-1` for a row that writes nothing). Output the same shape as `base`.
    /// The cost is `O(N * rest)`, never `O(POOL * rest)`: one thread per source element.
    ///
    /// The kernel writes only the named rows, so the planner requires the output to be donated to `base`'s
    /// own buffer (a state pair whose old value nothing else reads), and refuses the graph otherwise. Two
    /// rows naming one slot would race; the map's single binder, [`crate::ForwardSlotMap`], refuses a
    /// repeated slot, and the oracle refuses one too. Unlike [`OpKind::Scatter`] (a permutation, every row
    /// written), this passes the unnamed rows through: the scattered analog of
    /// [`OpKind::DynamicUpdateSlice`].
    ScatterUpdate,
    /// Batched matmul A[..,M,K] x B[..,K,N] -> [..,M,N], batch dims broadcast (numpy semantics).
    MatMul,
    /// Decode one validated rank-2 packed payload into checkpoint-order `[out,K]` f32.
    /// Its `descriptor.sources()` I8 operands are opaque const byte carriers, not numeric tensors,
    /// one per named [`poot_quant::SourceRole`] in that order (no positional slots): a GGUF block
    /// format has one (`Blocks`), a registered E4M3/E2M1 cell has two (`Codes`, `Scale`), and GPTQ or
    /// AWQ have three or four (`Codes`, `Zero`, `Scale`, and `GroupIndex` for GPTQ act-order).
    PackedDequant {
        /// The packed weight whose payload is decoded.
        descriptor: PackedWeight,
    },
    /// Compiler-only replacement for activation matmul against a packed weight. Builders do not expose it; Card 356
    /// lowers it to an imported kernel.
    ///
    /// `rows` says how activation rows meet weight rows. [`ContractionRows::Blocks`] partitions the weight's `out`
    /// axis into that many contiguous, equal-width blocks and restricts each activation block to its own weight
    /// block (Card 385): `out[b, i, j] = sum over k of activation[b, i, k] * decode(weight[b * (out / blocks) + j,
    /// k])`, `b` ranging over `blocks`. `Blocks(1)` is the ordinary dense contraction (the activation may be any
    /// rank ending in `K`); `Blocks(n > 1)` requires a rank-3 `[n, M, K]` activation and produces `[n, M, out /
    /// n]`. See `packed_graphs::packed_linear` and `packed_graphs::packed_block_diagonal_linear`.
    /// [`ContractionRows::Indexed`] contracts each row of `x [M, K]` against its own expert of a stacked table
    /// (the routed MoE projection, `packed_graphs::packed_indexed_linear`); one field, not a second variant,
    /// because both are the same index map over the same descriptor-driven decode.
    PackedContraction {
        /// The packed weight the activation is contracted against.
        descriptor: PackedWeight,
        /// How activation rows meet weight rows.
        rows: ContractionRows,
    },
    /// Compiler-only replacement for an axis-0 `Gather` of a packed weight's rows (card 642/545a,
    /// quantized GGUF token embeddings): `out[.., :] = decode(ids[..], :)`. Inputs are the
    /// weight's `descriptor.sources()` I8 carriers, in that order, then the row ids (I32, any
    /// shape); the output is `ids.shape ++ [K]` F32. Builders do not expose it: the claim pass rewrites
    /// `Gather { axis: 0 }(PackedDequant(..), ids)` into it, so no `[out, K]` table is ever materialized.
    PackedRowGather {
        /// The packed weight whose rows are gathered.
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
        /// The weight's storage dtype, one of [`DENSE_CONTRACTION_WEIGHT_DTYPES`].
        weight: DType,
    },
    /// Compiler-only grouped sibling of [`OpKind::DenseContraction`]: `groups` independent contractions, each over
    /// its own block of rows of one checkpoint-order weight. Inputs are the activation `[G, M, K]` (F32,
    /// `G == groups`) and the weight `[G * N, K]` (`weight` dtype), whose rows `g * N .. (g + 1) * N` are group
    /// `g`'s `[N, K]` block; the output is `[G, M, N]` F32:
    /// `out[g, m, n] = sum over k of activation[g, m, k] * decode(weight[g * N + n, k])`.
    ///
    /// BF16 is the one admitted `weight` dtype ([`OpKind::infer`] rejects the rest): it is the only dtype with
    /// grouped device rows, and any other dtype keeps the traced chain. `fold_dense_contractions` forms it from `MatMul(a, Transpose([0, 2, 1], Reshape([G, N, K], w)))`, the
    /// spelling of a grouped projection (DeepSeek V4's `wo_a`), so the weight stays one resident const in its
    /// checkpoint dtype instead of a widened F32 copy.
    GroupedDenseContraction {
        /// The weight's storage dtype; BF16 is the only admitted one.
        weight: DType,
        /// The number of independent groups `G`.
        groups: usize,
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
        /// The table's storage dtype, one of [`DENSE_ROW_GATHER_SOURCE_DTYPES`].
        source: DType,
    },
    /// A contraction followed by a pointwise chain: the head's result, then `region`'s steps with local 0 bound to
    /// the head's output. Inputs are the head's two operands, then the region's other external inputs (the region's
    /// locals `1..n_inputs`, in order): a bias `[N]`, a residual `[.., M, N]` or a per-row scale, each broadcasting to
    /// the head's output shape on read. The output has the head's shape; its dtype is the region's.
    ///
    /// Compiler-only: `fuse` forms it from a head and the single-use pointwise chain that consumes it, so there is
    /// one epilogue mechanism for every contraction head ([`ContractionHead`]) and every pointwise op the fused
    /// regions admit. A composite: its definition is the head op, then the region's steps ([`crate::decompose`]).
    /// The head's output keeps its declared dtype before the first step reads it, so a narrow head (BF16, F16)
    /// rounds once at the head, exactly as the unfused program stores it; a lowering keeps that rounding inside the
    /// kernel.
    ContractionEpilogue {
        /// The contraction that produces local 0.
        head: ContractionHead,
        /// The pointwise chain applied to the head's result.
        region: FusedRegion,
    },
    /// Batched argtop-k index extraction (spec 136, grouped-MoE prefill routing): `rank [..,E] -> [..,k]` expert
    /// ids. `out[..,r]` is the index `i` along the last axis with `rank[..,i] == r`, for `r` in `0..k`. One shared
    /// `[..,E]` stable rank tensor feeds this directly, with no per-row host loop. The ids are `I32`, the one
    /// index dtype (`IndexedMatMul`, `Gather` and the other index consumers read them as is). Precondition: each last-axis `rank` slice is a
    /// permutation of `0..E`; [`crate::ops::stable_descending_rank`] guarantees that even for tied scores. `k` is a
    /// trace-time constant (static shape, capture-safe). This op inverts rank and has no score-order or tie-break
    /// semantic of its own.
    ArgTopK {
        /// The number of indices extracted per row.
        k: usize,
    },
    /// Indexed matmul (card 088, sparse-MoE gather-free GEMM): `out[m,n] = sum_k x[m,k] * W[idx[m],k,n]`. Inputs are
    /// the activation `x [M,K]`, a stacked weight `W [E,K,N]`, and per-row expert ids `idx [M]` (I32). Each row `m`
    /// contracts against its own expert `W[idx[m]]` with no prior copy of the selected experts' weights (that copy
    /// made `moe_sparse` slower than dense). Output `[M,N]`, dtype = x's. One definition (the eager eval gathers
    /// per-row; the kernel reads the selected expert's rows in-kernel through `ContractionSpec::Indexed`). An id
    /// outside `0..E` is a typed oracle error and a NaN output row on every device.
    IndexedMatMul,
    /// Write `update` (input 1) into a copy of `operand` (input 0) starting at the scalar index `index` (input 2)
    /// along `axis`, leaving the rest unchanged (XLA DynamicUpdateSlice, single axis). The KV-cache slot write:
    /// `operand[.., index..index+update.shape[axis], ..] = update`. Result type == `operand`'s type. The index is
    /// typically an inline literal (the decode graph is pos-specialized).
    DynamicUpdateSlice {
        /// The axis the update is written along.
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
        /// The number of int8 codes to unpack, which the word count alone does not determine.
        len: usize,
    },
    /// A fused elementwise region (G5): a composition of pointwise steps lowered to one kernel. The eqn's
    /// `inputs` are the region's external inputs in order; see [`FusedRegion`].
    Fused(FusedRegion),
    /// A row-wise (reduction-rooted) fused region (G5b): last-axis reductions + their pointwise feed and
    /// epilogue, lowered to one row-at-a-time kernel; see [`RowRegion`].
    FusedRow(RowRegion),
    /// Fused attention, inputs `[q, k, v, mask]` with `q[B,Hq,Lq,D]`, `k`/`v[B,Hq/n_rep,Lk,D]` and
    /// `mask[B,Hm,Lq,Lk]` (`Hm` is 1 or Hq), all dense: `infer` rejects any other operand shape. Output
    /// `[B,Hq,Lq,D]` (== q's type). `Lq` and `Lk` are independent: decode is `Lq == 1` against a capacity-long cache, a
    /// first prefill `Lq == Lk`, a prefill into a cache `1 < Lq < Lk`; all one op. The op does GQA (`n_rep = Hq/Hkv`), the implicit transpose and an online (running max/sum)
    /// softmax over the keys internally, never materializing the `[B,Hq,Lq,Lk]` scores. Its definition is the
    /// `ops::attention_masked_softcap` decomposition. The planner chooses its kernel and schedule.
    Attention(AttentionAttrs),
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
        /// The rotary width: how many leading elements of the last axis are rotated, even and at most `D`.
        rot: usize,
    },
    /// A bounded recurrence over the sequence axis (POOT-634): the gated delta rule (GDN) or the Mamba2
    /// SSD, with ONE typed packed result `[1, H, L + S, W]`. Rows `..L` are the sequence output `y`, rows
    /// `L..L + S` the carried state `s_out` ([`ScanLayout`] names the ranges; [`crate::Builder::scan`]
    /// returns the two projections as a [`crate::ScanResult`]). The operands are [`ScanFamily::operands`]
    /// long, in the order the family documents. Its definition is the family's chunked primitive chain
    /// (`ops::scan_delta_rule_packed` / `ops::scan_ssd_packed`), one chunk body per chunk; the CPU oracle evaluates one chunk window at a time with the state carried; a planner
    /// schedule runs the sequence in a loop of bounded dispatches instead. One source-admissible
    /// recurrence carrier, not a fusion or schedule choice.
    Scan(ScanAttrs),
    /// The per-query block selection of a sparse-attention indexer (POOT-1143): the best `k` of the blocks
    /// of `ratio` consecutive positions that a query's position has completed, ranked by score. Operands
    /// are the block scores `[R, 1, T, NB]` (F32) and the queries' absolute positions `[R, T]` (I32). The
    /// one typed result is I32 `[R, 1, T, K + NB]` ([`BlockTopKAttrs::layout`] names the halves): the
    /// first `K` entries are the selected block ids in ascending order, `-1` after the last when fewer than
    /// `K` blocks are eligible; the last `NB` are the `0`/`1` keep indicator of each block.
    ///
    /// Block `b` is eligible for a query at position `p` when every position of it is at or before `p`:
    /// `b * ratio + ratio - 1 <= p`. Eligible blocks rank by canonical score
    /// ([`crate::ops::canonical_router_scores`]) descending, ties to the lower block; ineligible blocks are
    /// never selected. Its definition is the primitive chain of [`crate::ops::block_top_k_packed`]: a stable
    /// rank over all `NB` blocks, quadratic in `NB`, so the CPU oracle refuses past a block-count bound
    /// and a device runs a linear selection kernel instead. The selected set is exact on every executor.
    ///
    /// A tracer never emits it: it states the composition above and `compile`'s selection pass forms the op.
    BlockTopK(BlockTopKAttrs),
    /// Attention over the blocks a [`OpKind::BlockTopK`] selected (POOT-1143): each query reads only the keys of
    /// its `K` selected blocks and of its own unfinished block, not all `Lk`. Operands `[q, k, v, ids, pos]`:
    /// `q [B,Hq,Lq,Dk]`, `k [B,Hkv,Lk,Dk]`, `v [B,Hkv,Lk,Dv]` (the cache, stored positions past `Lk` never
    /// read), `ids [B,1,Lq,K]` I32 (the selected block ids, ascending, `-1` after the last: the ids half of a
    /// `BlockTopK` result; the contract is that `ids` come from one: eligible, complete blocks, each once, none
    /// the query's own unfinished block, which a device kernel would otherwise read twice) and `pos [B,Lq]` I32 (the queries' absolute positions). Output `[B,Hq,Lq,Dv]`.
    ///
    /// Query `(b, i)` at position `p` reads key `j` when `j <= p` and either block `j / ratio` is among its
    /// `ids`, or `j >= ((p + 1) / ratio) * ratio` (the unfinished block, always visible, unscored); the
    /// scores are `q . k * scale`, softcapped if asked, and softmax runs over the visible keys only. Its
    /// definition is the dense [`OpKind::Attention`] under the additive mask those conditions make
    /// ([`crate::ops::selected_attention`]); a device kernel visits `(K + 1) * ratio` positions per query
    /// instead of `Lk`, so its sum is reassociated (ADR-0101 tier 2) while the keys it reads are exact.
    ///
    /// A tracer never emits it: it states [`crate::ops::selected_attention`] and `compile`'s selection pass forms
    /// the op.
    SelectedAttention(SelectedAttentionAttrs),
    /// A cross-replica collective reduce (card 049a, tensor-parallel). Each rank holds a same-shaped partial tensor;
    /// the collective combines them with `op` and distributes the result back to every rank. The output type is
    /// identical to the input type.
    ///
    /// `axis` is the shard axis (which logical dimension was split across ranks at the row-parallel matmul's weight
    /// sharding). It is metadata for the multi-device lowering (ring-allreduce, card 049b) and shard-shape checking;
    /// it does not index an axis that gets reduced here.
    ///
    /// The oracle evaluates one rank: the identity (`AllReduce(x)` over one replica is `x`). A collective is a cut
    /// point between stage programs, never an equation a stage program plans (`compile` refuses it).
    AllReduce {
        /// The combiner applied across replicas.
        op: RedOp,
        /// The shard (contraction) axis this collective collapses across replicas. Metadata only for this slice; the
        /// ring-allreduce lowering (card 049b) uses it for buffer sizing.
        axis: usize,
    },
    /// Cross-replica all-gather (column-parallel matmul). Each rank holds a slice of the output tensor along
    /// `axis`; the collective concatenates the slices of `world` ranks, so the output's `axis` dimension is
    /// `world` times the input's.
    ///
    /// A collective is a cut point between stage programs, never an equation a stage program plans: `compile`
    /// refuses one inside a stage graph, and the oracle refuses `world > 1` (there is no identity to fall
    /// back on once the axis has grown).
    AllGather {
        /// The output axis along which ranks are concatenated. In column-parallel matmul this is the last axis
        /// (the N / output-feature axis).
        axis: usize,
        /// The number of ranks whose slices are concatenated. `1` is the identity.
        world: NonZeroU32,
    },
    /// Exact uniform noise from an integer seed (card 551a, R472-007): one integer hash plus a
    /// power-of-two multiply, bit-exact on every backend (tier 1, ADR-0101). Input `seed [..]` I32 (u32
    /// bits); output `[.., cols]` F32. For row-major flat index `r` (the seed's own flat position) and
    /// column `i`, `x = fmix32(seed[r] XOR wrapping_mul(i, 0x9E3779B9))` (the column alone, so a seed
    /// draws the same noise whichever row of a batch it rides in), `u = (2 * (x >> 9) +
    /// 1) * 2^-24`: an exact I32->F32 conversion below `2^24` and a power-of-two multiply, so `u` is
    /// always in `[2^-24, 1 - 2^-24]` and a double `log` of it never diverges. See
    /// [`crate::ops::sampling::hash32`] and [`crate::ops::sampling::random_uniform`].
    RandomUniform {
        /// The number of noise columns per seed: the last extent of the output.
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
    ///
    /// A primitive with a closed rule set, not a composite: its top-k and top-p floors are value-threshold
    /// bisections over the row (a data-dependent search no elementwise or last-axis reduction composition
    /// states), and the contract's tiers are stated per rule (ADR-0114). The rule set is the
    /// [`SampleRule`] enum, exhaustively matched at every consumer, so a new rule is a compile error
    /// at each one rather than an open extension point. The exact host sampler is the reference for the
    /// distribution (`poot-llm`'s `Sampler`), never this op's evaluator.
    SampleToken {
        /// Which sampler the operand list and semantics follow.
        rule: SampleRule,
    },
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

/// Card 529 (R466-010), POOT-725: reject an index-role operand (`Gather`'s and `Scatter`'s index,
/// `ScatterUpdate`'s `slots`, `IndexedMatMul`'s and friends' `idx`, `DynamicUpdateSlice`'s runtime index)
/// that is not I32. I32 is the only index dtype.
fn check_index_dtype(op: &'static str, dtype: DType) -> Result<(), ShapeError> {
    if dtype == DType::I32 {
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
/// or, when `matmul_tensor_core_eligible` says so, runs the mismatched pair directly on a WMMA/coopmat
/// kernel built for it (a BF16 `[K, N]` decode projection, `matmul_bf16_column_gemv`, stays mixed and is refused). Every other pairing (F16 x BF16, I32 x
/// F32, a BF16 activation x F32 weight, two different storage/packed dtypes, ...) is rejected.
fn matmul_operand_dtypes_allowed(a: DType, b: DType) -> bool {
    (a.class() == DTypeClass::Arithmetic && a == b)
        || (a == DType::F32 && matches!(b, DType::BF16 | DType::F16))
}

impl OpKind {
    /// Infer the output type from the operand types. Pure, host-only, no data. The single source of
    /// truth for every eqn's result shape/dtype.
    #[deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
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
                // The condition is an I32 word (nonzero picks `if_true`); the branches share one
                // arithmetic or I32 dtype, and the result is that dtype.
                if ins[0].dtype != DType::I32 {
                    return Err(ShapeError::DtypeOp {
                        op: "Select",
                        dtype: ins[0].dtype,
                    });
                }
                for branch in &ins[1..] {
                    if !matches!(
                        branch.dtype.class(),
                        DTypeClass::Arithmetic | DTypeClass::Index
                    ) {
                        return Err(ShapeError::DtypeOp {
                            op: "Select",
                            dtype: branch.dtype,
                        });
                    }
                }
                if ins[1].dtype != ins[2].dtype {
                    return Err(ShapeError::Dtype {
                        a: ins[1].clone(),
                        b: ins[2].clone(),
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
                Ok(TensorType::new(shape, ins[1].dtype))
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
                    // longer operand, and `t.shape[*axis]` panics when a shorter operand has rank <= axis.
                    if t.rank() != rank {
                        return Err(ShapeError::OperandRank {
                            op: "Concat",
                            operand: "input",
                            expected: rank,
                            actual: t.rank(),
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
            OpKind::Iota { len, dtype } => {
                // Nullary: a compile-time-computed range, shape [len]. `fold_iota` replaces every
                // Iota with a computed graph constant before planning.
                arity(ins, 0)?;
                let exact = match dtype {
                    DType::F32 => *len <= 1 << f32::MANTISSA_DIGITS,
                    DType::I32 => i32::try_from(*len).is_ok(),
                    _ => false,
                };
                if !exact {
                    return Err(ShapeError::IotaRange {
                        len: *len,
                        dtype: *dtype,
                    });
                }
                Ok(TensorType::new(vec![*len], *dtype))
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
                // rank [..,E] -> [..,k]: same leading dims, last axis replaced by k. The ids are I32, the index dtype
                // `IndexedMatMul` and the other index consumers read.
                arity(ins, 1)?;
                let mut shape = ins[0].shape.clone();
                let extent = last_axis_mut(&mut shape, "ArgTopK", "input")?;
                if *k > *extent {
                    return Err(ShapeError::TopK {
                        k: *k,
                        extent: *extent,
                    });
                }
                *extent = *k;
                Ok(TensorType::new(shape, DType::I32))
            }
            OpKind::PackI8 => {
                arity(ins, 1)?;
                let mut shape = ins[0].shape.clone();
                let last = last_axis_mut(&mut shape, "PackI8", "input")?;
                *last = last.div_ceil(4);
                Ok(TensorType::new(shape, DType::I32))
            }
            OpKind::UnpackI8 { len } => {
                arity(ins, 1)?;
                let mut shape = ins[0].shape.clone();
                *last_axis_mut(&mut shape, "UnpackI8", "codes")? = *len;
                Ok(TensorType::new(shape, DType::F32))
            }
            OpKind::ScatterUpdate => {
                // base[POOL,..rest], src[N,..rest], slots[N] -> base's type. Source row t is written to
                // pool slot slots[t]; every other pool row passes through from base.
                arity(ins, 3)?;
                let (base, src, slots) = (&ins[0], &ins[1], &ins[2]);
                check_index_dtype("ScatterUpdate", slots.dtype)?;
                let Some((_, rest)) = base.shape.split_first() else {
                    return Err(ShapeError::RankTooLow {
                        op: "ScatterUpdate",
                        operand: "base",
                        min: 1,
                        actual: 0,
                    });
                };
                if src.rank() != base.rank() {
                    return Err(ShapeError::OperandRank {
                        op: "ScatterUpdate",
                        operand: "src",
                        expected: base.rank(),
                        actual: src.rank(),
                    });
                }
                let &[slots_len] = slots.shape.as_slice() else {
                    return Err(ShapeError::OperandRank {
                        op: "ScatterUpdate",
                        operand: "slots",
                        expected: 1,
                        actual: slots.rank(),
                    });
                };
                for (dim, (&expected, &actual)) in
                    rest.iter().zip(src.shape.iter().skip(1)).enumerate()
                {
                    if expected != actual {
                        return Err(ShapeError::OperandDim {
                            op: "ScatterUpdate",
                            operand: "src",
                            dim: dim + 1,
                            expected,
                            actual,
                        });
                    }
                }
                if slots_len != src.shape[0] {
                    return Err(ShapeError::OperandDim {
                        op: "ScatterUpdate",
                        operand: "slots",
                        dim: 0,
                        expected: src.shape[0],
                        actual: slots_len,
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
            OpKind::PackedContraction { descriptor, rows } => {
                let lead = rows.leading_operands();
                arity(ins, lead + descriptor.sources().len())?;
                let carriers = &ins[lead..];
                match rows {
                    ContractionRows::Blocks(_) => {
                        OpKind::PackedDequant {
                            descriptor: *descriptor,
                        }
                        .infer(carriers)?;
                    }
                    ContractionRows::Indexed { experts } => {
                        check_packed_stack(*descriptor, *experts, carriers)?;
                    }
                }
                let activation = &ins[0];
                if activation.dtype != DType::F32 {
                    return Err(ShapeError::PackedContractionDtype {
                        operand: "activation",
                        actual: activation.dtype,
                    });
                }
                let [out, k] = descriptor.shape();
                match *rows {
                    ContractionRows::Blocks(blocks) => {
                        if blocks == 0 || out % blocks != 0 {
                            return Err(ShapeError::PackedContractionBlocks { out, blocks });
                        }
                        let block_out = out / blocks;
                        if blocks == 1 {
                            let mut shape = activation.shape.clone();
                            match shape.last_mut() {
                                Some(last) if activation.rank() >= 2 && *last == k => *last = out,
                                _ => {
                                    return Err(ShapeError::PackedContractionActivation {
                                        expected_k: k,
                                        actual: activation.clone(),
                                    });
                                }
                            }
                            Ok(TensorType::f32(shape))
                        } else {
                            let shape = activation.shape.as_slice();
                            if shape.len() != 3 || shape[0] != blocks || shape[2] != k {
                                return Err(ShapeError::PackedContractionActivation {
                                    expected_k: k,
                                    actual: activation.clone(),
                                });
                            }
                            Ok(TensorType::f32(vec![blocks, shape[1], block_out]))
                        }
                    }
                    ContractionRows::Indexed { .. } => {
                        let ids = &ins[1];
                        check_index_dtype("PackedContraction", ids.dtype)?;
                        let &[rows, activation_k] = activation.shape.as_slice() else {
                            return Err(ShapeError::PackedContractionActivation {
                                expected_k: k,
                                actual: activation.clone(),
                            });
                        };
                        if activation_k != k {
                            return Err(ShapeError::PackedContractionActivation {
                                expected_k: k,
                                actual: activation.clone(),
                            });
                        }
                        let &[ids_len] = ids.shape.as_slice() else {
                            return Err(ShapeError::OperandRank {
                                op: "PackedContraction",
                                operand: "ids",
                                expected: 1,
                                actual: ids.rank(),
                            });
                        };
                        if ids_len != rows {
                            return Err(ShapeError::OperandDim {
                                op: "PackedContraction",
                                operand: "ids",
                                dim: 0,
                                expected: rows,
                                actual: ids_len,
                            });
                        }
                        Ok(TensorType::f32(vec![rows, out]))
                    }
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
                check_index_dtype("PackedRowGather", ids.dtype)?;
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
                let &[n, k] = source.shape.as_slice() else {
                    return Err(ShapeError::DenseContractionWeightRank {
                        actual: source.clone(),
                    });
                };
                let mut shape = activation.shape.clone();
                match shape.last_mut() {
                    Some(last) if activation.rank() >= 2 && *last == k => *last = n,
                    _ => {
                        return Err(ShapeError::DenseContractionActivation {
                            expected_k: k,
                            actual: activation.clone(),
                        });
                    }
                }
                Ok(TensorType::f32(shape))
            }
            OpKind::GroupedDenseContraction { weight, groups } => {
                // activation [G, M, K] F32 x weight [G * N, K] (BF16 only) -> [G, M, N] F32.
                arity(ins, 2)?;
                if *weight != DType::BF16 {
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
                let (rows, k) = (source.shape[0], source.shape[1]);
                if *groups == 0 || rows % groups != 0 {
                    return Err(ShapeError::GroupedDenseContractionGroups {
                        groups: *groups,
                        rows,
                    });
                }
                if activation.rank() != 3
                    || activation.shape[0] != *groups
                    || activation.shape[2] != k
                {
                    return Err(ShapeError::DenseContractionActivation {
                        expected_k: k,
                        actual: activation.clone(),
                    });
                }
                Ok(TensorType::f32(vec![
                    *groups,
                    activation.shape[1],
                    rows / groups,
                ]))
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
            OpKind::ContractionEpilogue { head, region } => {
                // The head's operands, then the region's inputs after local 0 (the head's result). The region
                // runs over the head's output shape, so the epilogue never reshapes or reduces it.
                let head_operands = head.operands();
                arity(ins, head_operands + region.n_inputs.saturating_sub(1))?;
                if !region.pack.is_empty() || !region.reads_input(0) {
                    return Err(ShapeError::ContractionEpilogueRegion {
                        packed: !region.pack.is_empty(),
                    });
                }
                let head_ty = head.op().infer(&ins[..head_operands])?;
                let mut region_ins = vec![head_ty.clone()];
                region_ins.extend_from_slice(&ins[head_operands..]);
                let out = region.infer(&region_ins)?;
                if out.shape != head_ty.shape {
                    return Err(ShapeError::EpilogueShape {
                        head: head_ty.shape,
                        region: out.shape,
                    });
                }
                Ok(out)
            }
            OpKind::IndexedMatMul => {
                // x[M,K], W[E,K,N], idx[M] -> [M,N]. Each row m contracts x[m] against expert W[idx[m]].
                arity(ins, 3)?;
                let (x, w, idx) = (&ins[0], &ins[1], &ins[2]);
                check_index_dtype("IndexedMatMul", idx.dtype)?;
                let rank_error = |operand, expected, actual| ShapeError::OperandRank {
                    op: "IndexedMatMul",
                    operand,
                    expected,
                    actual,
                };
                let &[m, k] = x.shape.as_slice() else {
                    return Err(rank_error("x", 2, x.rank()));
                };
                let &[_, w_k, n] = w.shape.as_slice() else {
                    return Err(rank_error("w", 3, w.rank()));
                };
                let &[idx_len] = idx.shape.as_slice() else {
                    return Err(rank_error("idx", 1, idx.rank()));
                };
                if w_k != k {
                    return Err(ShapeError::OperandDim {
                        op: "IndexedMatMul",
                        operand: "w",
                        dim: 1,
                        expected: k,
                        actual: w_k,
                    });
                }
                if idx_len != m {
                    return Err(ShapeError::OperandDim {
                        op: "IndexedMatMul",
                        operand: "idx",
                        dim: 0,
                        expected: m,
                        actual: idx_len,
                    });
                }
                Ok(TensorType::new(vec![m, n], x.dtype))
            }
            OpKind::DynamicUpdateSlice { axis } => {
                arity(ins, 3)?;
                let (operand, update, index) = (&ins[0], &ins[1], &ins[2]);
                check_index_dtype("DynamicUpdateSlice", index.dtype)?;
                check_axis(*axis, operand.rank())?;
                if update.rank() != operand.rank() {
                    return Err(ShapeError::OperandRank {
                        op: "DynamicUpdateSlice",
                        operand: "update",
                        expected: operand.rank(),
                        actual: update.rank(),
                    });
                }
                if !index.shape.is_empty() {
                    return Err(ShapeError::OperandRank {
                        op: "DynamicUpdateSlice",
                        operand: "index",
                        expected: 0,
                        actual: index.rank(),
                    });
                }
                // every non-axis dim must match; the axis extent must fit (bound checked against the
                // index at eval time, where the concrete index is known).
                for (dim, (&expected, &actual)) in
                    operand.shape.iter().zip(&update.shape).enumerate()
                {
                    if dim != *axis && expected != actual {
                        return Err(ShapeError::OperandDim {
                            op: "DynamicUpdateSlice",
                            operand: "update",
                            dim,
                            expected,
                            actual,
                        });
                    }
                }
                if update.shape.get(*axis) > operand.shape.get(*axis) {
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
            OpKind::Attention(attrs) => infer_attention(ins, attrs),
            OpKind::Rope { rot } => infer_rope(ins, *rot),
            OpKind::Scan(attrs) => infer_scan(ins, attrs),
            OpKind::BlockTopK(attrs) => infer_block_top_k(ins, attrs),
            OpKind::SelectedAttention(attrs) => infer_selected_attention(ins, attrs),
            OpKind::AllReduce { .. } => {
                // Shape-preserving: each rank holds a same-shaped partial; the collective combines across ranks and returns
                // the full result in the same shape and dtype.
                arity(ins, 1)?;
                Ok(ins[0].clone())
            }
            OpKind::AllGather { axis, world } => {
                arity(ins, 1)?;
                infer_all_gather(&ins[0], *axis, *world)
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
    /// | `Attention` | `flash_attention_capped` from the attention chain |
    /// | `Rope` | `rope_fusion` from the half-split rotation chain |
    /// | `BlockTopK` | `form_selection` from [`crate::ops::block_top_k_packed`], the rank chain a sparse indexer's tracer states |
    /// | `SelectedAttention` | `form_selection` from an additive `Attention` whose mask is [`crate::ops::selected_block_mask`] |
    /// | `ContractionEpilogue` | `fuse` from a contraction and the single-use pointwise chain that consumes it |
    /// | `Fused`, `FusedRow` | `fuse` from pointwise chains and their last-axis reductions |
    ///
    /// The ops that are neither an IR primitive above nor a composite, and why each stays primitive:
    ///
    /// | Op | Reason |
    /// | --- | --- |
    /// | `PackedDequant` | the one quantized primitive (ADR-0103 decision 2); its definition is `poot-quant`'s decoder |
    /// | `PackedRowGather` | the packed gather claim; the oracle defines it directly and decodes only the rows it gathers |
    /// | `ArgTopK` | inverts a rank permutation without the `[.., E]` scatter a decomposition would need |
    /// | `PackI8`, `UnpackI8` | bit packing of KV codes; no arithmetic decomposition |
    /// | `AllReduce`, `AllGather` | cross-rank collectives; `AllReduce` keeps the shape, `AllGather` multiplies `axis` by `world` |
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
            | Self::PackedRowGather { .. }
            | Self::PackedContraction {
                rows: ContractionRows::Indexed { .. },
                ..
            } => OpClass::Primitive,
            Self::PackedContraction {
                rows: ContractionRows::Blocks(_),
                ..
            }
            | Self::DenseContraction { .. }
            | Self::GroupedDenseContraction { .. }
            | Self::DenseRowGather { .. }
            | Self::Attention(_)
            | Self::Rope { .. }
            | Self::Scan(_)
            | Self::BlockTopK(_)
            | Self::SelectedAttention(_)
            | Self::ContractionEpilogue { .. }
            | Self::Fused(_)
            | Self::FusedRow(_) => OpClass::Composite,
        }
    }

    /// The weight dtype when this equation contracts an activation against a `[N, K]` dense weight: a
    /// [`OpKind::DenseContraction`], a [`OpKind::GroupedDenseContraction`], or a
    /// [`OpKind::ContractionEpilogue`] over a dense head. The weight is input 1 in all three.
    pub fn dense_weight(&self) -> Option<DType> {
        match self {
            Self::DenseContraction { weight }
            | Self::GroupedDenseContraction { weight, .. }
            | Self::ContractionEpilogue {
                head: ContractionHead::Dense { weight },
                ..
            } => Some(*weight),
            _ => None,
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
            Self::GroupedDenseContraction { .. } => "grouped_dense_contraction",
            Self::DenseRowGather { .. } => "dense_row_gather",
            Self::ContractionEpilogue { .. } => "contraction_epilogue",
            Self::ArgTopK { .. } => "arg_top_k",
            Self::IndexedMatMul => "indexed_matmul",
            Self::DynamicUpdateSlice { .. } => "dynamic_update_slice",
            Self::PackI8 => "pack_i8",
            Self::UnpackI8 { .. } => "unpack_i8",
            Self::Fused(_) => "fused",
            Self::FusedRow(_) => "fused_row",
            Self::Attention(_) => "attention",
            Self::Rope { .. } => "rope",
            Self::Scan(_) => "scan",
            Self::BlockTopK(_) => "block_top_k",
            Self::SelectedAttention(_) => "selected_attention",
            Self::AllReduce { .. } => "all_reduce",
            Self::AllGather { .. } => "all_gather",
            Self::RandomUniform { .. } => "random_uniform",
            Self::SampleToken { .. } => "sample_token",
        }
    }

    /// The bytes of heap storage this operation's attributes own beyond its inline size: the shape
    /// and permutation lists, and the step lists of a fused region. `None` when the sum overflows
    /// `u64`. The tracer's owned-byte budget charges this with the record that holds the operation.
    pub(crate) fn owned_heap_bytes(&self) -> Option<u64> {
        const WORD: u64 = std::mem::size_of::<usize>() as u64;
        fn list(len: usize, element: usize) -> Option<u64> {
            (len as u64).checked_mul(element as u64)
        }
        fn fused(region: &FusedRegion) -> Option<u64> {
            let mut total = list(region.pack.len(), std::mem::size_of::<usize>())?;
            total =
                total.checked_add(list(region.steps.len(), std::mem::size_of::<FusedStep>())?)?;
            for step in &region.steps {
                total = total.checked_add(list(
                    step.inputs.len(),
                    std::mem::size_of::<FusedOperand>(),
                )?)?;
            }
            Some(total)
        }
        match self {
            OpKind::Broadcast { shape } | OpKind::Reshape { shape } => {
                (shape.len() as u64).checked_mul(WORD)
            }
            OpKind::Transpose { perm } => (perm.len() as u64).checked_mul(WORD),
            OpKind::Fused(region) => fused(region),
            OpKind::ContractionEpilogue { region, .. } => fused(region),
            OpKind::FusedRow(region) => {
                let mut total = list(region.steps.len(), std::mem::size_of::<RowStep>())?;
                for step in &region.steps {
                    if let RowStep::Pointwise { inputs, .. } = step {
                        total = total.checked_add(list(
                            inputs.len(),
                            std::mem::size_of::<FusedOperand>(),
                        )?)?;
                    }
                }
                Some(total)
            }
            OpKind::Unary(_)
            | OpKind::Binary(_)
            | OpKind::Select
            | OpKind::Reduce { .. }
            | OpKind::Cast { .. }
            | OpKind::Slice { .. }
            | OpKind::Concat { .. }
            | OpKind::Iota { .. }
            | OpKind::Gather { .. }
            | OpKind::Scatter { .. }
            | OpKind::ScatterUpdate
            | OpKind::MatMul
            | OpKind::PackedDequant { .. }
            | OpKind::PackedContraction { .. }
            | OpKind::PackedRowGather { .. }
            | OpKind::DenseContraction { .. }
            | OpKind::GroupedDenseContraction { .. }
            | OpKind::DenseRowGather { .. }
            | OpKind::ArgTopK { .. }
            | OpKind::IndexedMatMul
            | OpKind::DynamicUpdateSlice { .. }
            | OpKind::PackI8
            | OpKind::UnpackI8 { .. }
            | OpKind::Attention(_)
            | OpKind::Rope { .. }
            | OpKind::Scan(_)
            | OpKind::BlockTopK(_)
            | OpKind::SelectedAttention(_)
            | OpKind::AllReduce { .. }
            | OpKind::AllGather { .. }
            | OpKind::RandomUniform { .. }
            | OpKind::SampleToken { .. } => Some(0),
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
            OpKind::Iota { len, dtype } => format!("iota len={len} {dtype}"),
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
            OpKind::PackedContraction { descriptor, rows } => {
                let base = format!("packed_contraction {:?}", descriptor.format()).to_lowercase();
                match rows {
                    ContractionRows::Blocks(1) => base,
                    ContractionRows::Blocks(blocks) => format!("{base} blocks={blocks}"),
                    ContractionRows::Indexed { experts } => format!("{base} experts={experts}"),
                }
            }
            OpKind::PackedRowGather { descriptor } => {
                format!("packed_row_gather {:?}", descriptor.format()).to_lowercase()
            }
            OpKind::DenseContraction { weight } => {
                format!("dense_contraction {weight:?}").to_lowercase()
            }
            OpKind::GroupedDenseContraction { weight, groups } => {
                format!("grouped_dense_contraction {weight:?} groups={groups}").to_lowercase()
            }
            OpKind::DenseRowGather { source } => {
                format!("dense_row_gather {source:?}").to_lowercase()
            }
            OpKind::ContractionEpilogue { head, region } => format!(
                "contraction_epilogue {head:?} n_in={} steps={}",
                region.n_inputs,
                region.steps.len()
            ),
            OpKind::IndexedMatMul => "indexed_matmul".to_string(),
            OpKind::DynamicUpdateSlice { axis } => format!("dyn_update_slice ax={axis}"),
            OpKind::Fused(r) => format!("fused n_in={} steps={}", r.n_inputs, r.steps.len()),
            OpKind::FusedRow(r) => format!("fused_row n_in={} steps={}", r.n_inputs, r.steps.len()),
            OpKind::Attention(a) => format!(
                "attention n_rep={} scale={} softcap={:?} mask={:?} key={:?} bound={:?} sink={:?}",
                a.n_rep, a.scale, a.softcap, a.mask, a.key, a.bound, a.sink
            ),
            OpKind::Rope { rot } => format!("rope rot={rot}"),
            OpKind::Scan(attrs) => format!("scan {} chunk={}", attrs.family.name(), attrs.chunk),
            OpKind::BlockTopK(attrs) => format!("block_top_k ratio={} k={}", attrs.ratio, attrs.k),
            OpKind::SelectedAttention(a) => format!(
                "selected_attention n_rep={} scale={} softcap={:?} ratio={} k={}",
                a.n_rep, a.scale, a.softcap, a.ratio, a.k
            ),
            OpKind::AllReduce { op, axis } => {
                format!("all_reduce.{} ax={axis}", format!("{op:?}").to_lowercase())
            }
            OpKind::AllGather { axis, world } => format!("all_gather ax={axis} world={world}"),
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

/// The typing rule of `Attention`. Operands are `[q, k, v, mask]`, then the shared key part for a
/// [`KeyLayout::Split`] or the I32 `read [B,Lk]` slot map of the `[S,Hkv,D]` pools `k` and `v` for a
/// [`KeyLayout::Paged`], then the I32 `len [B]` for a [`KvBound::Live`]. Every consumer (the oracle, the
/// generated kernels) indexes the operands densely, with no broadcast strides: `q[B,Hq,Lq,Dk]` with `n_rep | Hq`,
/// `k[B,Hq/n_rep,Lk,Dk]` (or `Dk - shared_dim` wide next to `k_shared[B,1,Lk,shared_dim]`), `v[B,Hq/n_rep,Lk,Dv]`
/// and `mask[B,Hm,Lq,Lk]` with `Hm` either 1 (one plane shared by every head) or `Hq` (one per head, for ALiBi).
/// The output is `[B,Hq,Lq,Dv]` in q's dtype. A K/V or mask that the decomposition would merely broadcast (a
/// size-1 head or batch axis) is not this op. `Lq` and `Lk` are independent: decode (`Lq == 1`), a first prefill
/// (`Lq == Lk`) and a prefill into a cache (`1 < Lq < Lk`) all type here, at every float dtype and either
/// bound: a live bound's decomposition selects inactive keys with a bit-moving `Select`, which every float width
/// has.
fn infer_attention(ins: &[TensorType], attrs: &AttentionAttrs) -> Result<TensorType, ShapeError> {
    let split = matches!(attrs.key, KeyLayout::Split { .. });
    let paged = matches!(attrs.key, KeyLayout::Paged);
    let live = matches!(attrs.bound, KvBound::Live);
    let sink = matches!(attrs.sink, SinkOperand::PerHead);
    arity(
        ins,
        4 + usize::from(split || paged) + usize::from(live) + usize::from(sink),
    )?;
    let (q, k, v, mask) = (&ins[0], &ins[1], &ins[2], &ins[3]);
    let shared = split.then(|| &ins[4]);
    let read = paged.then(|| &ins[4]);
    let len = live.then(|| &ins[ins.len() - 1 - usize::from(sink)]);
    let sinks = sink.then(|| &ins[ins.len() - 1]);
    // Float operands only: the decomposition's softmax (`Exp`, `Div`) does not type over I32.
    if q.dtype.class() != DTypeClass::Arithmetic {
        return Err(ShapeError::DtypeOp {
            op: "Attention",
            dtype: q.dtype,
        });
    }
    let additive = matches!(attrs.mask, MaskOperand::Additive);
    for (operand, t) in [("k", k), ("v", v)]
        .into_iter()
        .chain(additive.then_some(("mask", mask)))
        .chain(shared.map(|t| ("k_shared", t)))
        .chain(sinks.map(|t| ("sinks", t)))
    {
        if t.dtype != q.dtype {
            return Err(ShapeError::AttentionDtype {
                operand,
                expected: q.dtype,
                actual: t.dtype,
            });
        }
    }
    if let Some(read) = read
        && read.dtype != DType::I32
    {
        return Err(ShapeError::AttentionDtype {
            operand: "read",
            expected: DType::I32,
            actual: read.dtype,
        });
    }
    let n_rep = attrs.n_rep;
    let &[bsz, hq, lq, dk] = q.shape.as_slice() else {
        return Err(ShapeError::AttentionQuery {
            n_rep,
            actual: q.shape.clone(),
        });
    };
    if n_rep == 0 || hq % n_rep != 0 {
        return Err(ShapeError::AttentionQuery {
            n_rep,
            actual: q.shape.clone(),
        });
    }
    let hkv = hq / n_rep;
    // The key and value extents `Lk` and `Dv`, with the key and value operands checked against them: a dense
    // `[B,Hkv,Lk,..]` pair, or one `[S,Hkv,..]` pool each read through the `[B,Lk]` slot map.
    let (lk, dv) = match read {
        Some(read) => {
            let &[rows, lk] = read.shape.as_slice() else {
                return Err(ShapeError::AttentionOperand {
                    operand: "read",
                    expected: vec![bsz, 1],
                    actual: read.shape.clone(),
                });
            };
            if rows != bsz {
                return Err(ShapeError::AttentionOperand {
                    operand: "read",
                    expected: vec![bsz, lk],
                    actual: read.shape.clone(),
                });
            }
            let Some(&slots) = k.shape.first() else {
                return Err(ShapeError::AttentionOperand {
                    operand: "k",
                    expected: vec![1, hkv, dk],
                    actual: k.shape.clone(),
                });
            };
            let k_expected = [slots, hkv, dk];
            if k.shape != k_expected {
                return Err(ShapeError::AttentionOperand {
                    operand: "k",
                    expected: k_expected.to_vec(),
                    actual: k.shape.clone(),
                });
            }
            let Some(&dv) = v.shape.get(2) else {
                return Err(ShapeError::AttentionOperand {
                    operand: "v",
                    expected: vec![slots, hkv, dk],
                    actual: v.shape.clone(),
                });
            };
            let v_expected = [slots, hkv, dv];
            if dv == 0 || v.shape != v_expected {
                return Err(ShapeError::AttentionOperand {
                    operand: "v",
                    expected: v_expected.to_vec(),
                    actual: v.shape.clone(),
                });
            }
            (lk, dv)
        }
        None => {
            let Some(&lk) = k.shape.get(2) else {
                return Err(ShapeError::AttentionOperand {
                    operand: "k",
                    expected: vec![bsz, hkv, lq, dk],
                    actual: k.shape.clone(),
                });
            };
            let key_width = match attrs.key {
                KeyLayout::Single | KeyLayout::Paged => dk,
                KeyLayout::Split { shared_dim } => {
                    if shared_dim == 0 || shared_dim >= dk {
                        return Err(ShapeError::AttentionOperand {
                            operand: "k_shared",
                            expected: vec![bsz, 1, lk, shared_dim],
                            actual: vec![bsz, 1, lk, dk],
                        });
                    }
                    dk - shared_dim
                }
            };
            let k_expected = [bsz, hkv, lk, key_width];
            if k.shape != k_expected {
                return Err(ShapeError::AttentionOperand {
                    operand: "k",
                    expected: k_expected.to_vec(),
                    actual: k.shape.clone(),
                });
            }
            if let (KeyLayout::Split { shared_dim }, Some(shared)) = (attrs.key, shared) {
                let expected = [bsz, 1, lk, shared_dim];
                if shared.shape != expected {
                    return Err(ShapeError::AttentionOperand {
                        operand: "k_shared",
                        expected: expected.to_vec(),
                        actual: shared.shape.clone(),
                    });
                }
            }
            let Some(&dv) = v.shape.get(3) else {
                return Err(ShapeError::AttentionOperand {
                    operand: "v",
                    expected: vec![bsz, hkv, lk, dk],
                    actual: v.shape.clone(),
                });
            };
            let v_expected = [bsz, hkv, lk, dv];
            if dv == 0 || v.shape != v_expected {
                return Err(ShapeError::AttentionOperand {
                    operand: "v",
                    expected: v_expected.to_vec(),
                    actual: v.shape.clone(),
                });
            }
            (lk, dv)
        }
    };
    match attrs.mask {
        MaskOperand::Additive => {
            let mask_heads = if mask.shape.get(1) == Some(&hq) {
                hq
            } else {
                1
            };
            let dense_mask = [bsz, mask_heads, lq, lk];
            if mask.shape != dense_mask {
                return Err(ShapeError::AttentionOperand {
                    operand: "mask",
                    expected: dense_mask.to_vec(),
                    actual: mask.shape.clone(),
                });
            }
        }
        MaskOperand::Causal { window } => {
            if mask.dtype != DType::I32 {
                return Err(ShapeError::AttentionDtype {
                    operand: "positions",
                    expected: DType::I32,
                    actual: mask.dtype,
                });
            }
            if mask.shape != [bsz, lq] || window == Some(0) {
                return Err(ShapeError::AttentionOperand {
                    operand: "positions",
                    expected: vec![bsz, lq],
                    actual: mask.shape.clone(),
                });
            }
        }
    }
    if let Some(len) = len
        && (len.dtype != DType::I32 || len.shape != [bsz])
    {
        return Err(ShapeError::AttentionOperand {
            operand: "len",
            expected: vec![bsz],
            actual: len.shape.clone(),
        });
    }
    if let Some(sinks) = sinks
        && sinks.shape != [hq]
    {
        return Err(ShapeError::AttentionOperand {
            operand: "sinks",
            expected: vec![hq],
            actual: sinks.shape.clone(),
        });
    }
    Ok(TensorType::new(vec![bsz, hq, lq, dv], q.dtype))
}

/// The recurrence a [`OpKind::Scan`] computes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ScanFamily {
    /// The gated delta rule. Operands `[q, k, v, g, beta, s_in]`: `q`, `k`, `v` `[1,H,L,D]` (already at
    /// the value-head count, `q` unscaled), `g` and `beta` `[1,H,L,1]` (`g` log-domain), `s_in`
    /// `[1,H,D,D]`. Result `[1,H,L+D,D]`.
    DeltaRule,
    /// The Mamba2 SSD. Operands `[x, b, c, delta, a, d, h_in]`: `x` `[1,H,L,P]`, `b` and `c` `[1,H,L,N]`
    /// (already at the head count), `delta` `[1,H,L,1]`, `a` and `d` `[1,H,1,1]`, `h_in` `[1,H,N,P]`.
    /// Result `[1,H,L+N,P]`.
    Ssd,
}

impl ScanFamily {
    /// How many operands the family takes.
    pub const fn operands(self) -> usize {
        match self {
            Self::DeltaRule => 6,
            Self::Ssd => 7,
        }
    }

    /// The operands with a sequence axis (axis 2, length `L`), a prefix of the operand list: a chunk window of
    /// the scan slices exactly these. The remaining operands are per-head parameters, then the carried state
    /// (always the last operand).
    #[cfg(any(test, feature = "test-support"))]
    pub const fn sequence_operands(self) -> std::ops::Range<usize> {
        match self {
            Self::DeltaRule => 0..5,
            Self::Ssd => 0..4,
        }
    }

    /// A stable lowercase name, for plan labels and summaries.
    pub const fn name(self) -> &'static str {
        match self {
            Self::DeltaRule => "delta_rule",
            Self::Ssd => "ssd",
        }
    }
}

/// The attributes of [`OpKind::Scan`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ScanAttrs {
    /// Which recurrence the scan computes.
    pub family: ScanFamily,
    /// The chunk length (a power of two) of the reference decomposition. A device schedule is free to
    /// differ; the semantics do not depend on it beyond rounding.
    pub chunk: usize,
}

/// The row layout of a scan's packed result `[1, heads, len + state_rows, width]`: the sequence output
/// `y [1,heads,len,width]` first, then the carried state `s_out [1,heads,state_rows,width]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ScanLayout {
    /// The head count `H`.
    pub heads: usize,
    /// The sequence length `L`.
    pub len: usize,
    /// The rows of the carried state per head.
    pub state_rows: usize,
    /// The head width, the last axis.
    pub width: usize,
}

impl ScanAttrs {
    /// The layout the operand types `ins` imply, or why they are not a scan of this family.
    pub fn layout(&self, ins: &[TensorType]) -> Result<ScanLayout, ShapeError> {
        let bad = ShapeError::Scan;
        arity(ins, self.family.operands())?;
        if ins.iter().any(|t| t.dtype != DType::F32) {
            return Err(bad(ScanReason::NotF32));
        }
        let first = &ins[0];
        if first.shape.len() != 4 || first.shape[0] != 1 {
            return Err(bad(ScanReason::LeadOperand {
                shape: first.shape.clone(),
            }));
        }
        let (h, l, w) = (first.shape[1], first.shape[2], first.shape[3]);
        if h == 0 || l == 0 || w == 0 || !self.chunk.is_power_of_two() {
            return Err(bad(ScanReason::Extents {
                shape: first.shape.clone(),
                chunk: self.chunk,
            }));
        }
        let (state_rows, expected): (usize, Vec<(&str, Vec<usize>)>) = match self.family {
            ScanFamily::DeltaRule => (
                w,
                vec![
                    ("k", vec![1, h, l, w]),
                    ("v", vec![1, h, l, w]),
                    ("g", vec![1, h, l, 1]),
                    ("beta", vec![1, h, l, 1]),
                    ("s_in", vec![1, h, w, w]),
                ],
            ),
            ScanFamily::Ssd => {
                let n = ins[1].shape.get(3).copied().unwrap_or(0);
                (
                    n,
                    vec![
                        ("b", vec![1, h, l, n]),
                        ("c", vec![1, h, l, n]),
                        ("delta", vec![1, h, l, 1]),
                        ("a", vec![1, h, 1, 1]),
                        ("d", vec![1, h, 1, 1]),
                        ("h_in", vec![1, h, n, w]),
                    ],
                )
            }
        };
        if state_rows == 0 {
            return Err(bad(ScanReason::NoStateRows));
        }
        for (i, (name, want)) in expected.into_iter().enumerate() {
            if ins[i + 1].shape != want {
                return Err(bad(ScanReason::Operand {
                    name,
                    want,
                    got: ins[i + 1].shape.clone(),
                }));
            }
        }
        Ok(ScanLayout {
            heads: h,
            len: l,
            state_rows,
            width: w,
        })
    }
}

impl ScanLayout {
    /// The packed result's shape.
    pub fn packed_shape(&self) -> Vec<usize> {
        vec![1, self.heads, self.len + self.state_rows, self.width]
    }

    /// The packed rows holding the sequence output `y`.
    pub fn y_rows(&self) -> std::ops::Range<usize> {
        0..self.len
    }

    /// The packed rows holding the carried state `s_out`.
    pub fn state_rows(&self) -> std::ops::Range<usize> {
        self.len..self.len + self.state_rows
    }
}

/// `AllGather`: the output is the input with `axis` multiplied by `world`; the dtype is unchanged.
fn infer_all_gather(
    input: &TensorType,
    axis: usize,
    world: NonZeroU32,
) -> Result<TensorType, ShapeError> {
    let mut shape = input.shape.clone();
    let rank = shape.len();
    let len = shape.get_mut(axis).ok_or(ShapeError::Axis { axis, rank })?;
    let gathered = len
        .checked_mul(world.get() as usize)
        .ok_or(ShapeError::AllGatherOverflow {
            axis,
            len: *len,
            world,
        })?;
    *len = gathered;
    Ok(TensorType::new(shape, input.dtype))
}

/// The attributes of [`OpKind::SelectedAttention`].
#[derive(Clone, PartialEq, Debug)]
pub struct SelectedAttentionAttrs {
    /// Query heads per key/value head (`Hq / Hkv`).
    pub n_rep: usize,
    /// Multiplier on the `Q K^T` scores.
    pub scale: f32,
    /// `Some(c)` applies `c * tanh(scores / c)` to the scaled scores; `None` applies none.
    pub softcap: Option<f32>,
    /// The positions each selected block covers.
    pub ratio: usize,
    /// The blocks each query reads: the width of the `ids` operand.
    pub k: usize,
}

/// The extents a [`OpKind::SelectedAttention`]'s operands fix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SelectedAttentionLayout {
    /// Batch rows `B`.
    pub batch: usize,
    /// Query heads `Hq`.
    pub hq: usize,
    /// Query rows per batch row `Lq`.
    pub lq: usize,
    /// Stored key positions `Lk`.
    pub lk: usize,
    /// The query and key head width.
    pub dk: usize,
    /// The value and output head width.
    pub dv: usize,
}

impl SelectedAttentionAttrs {
    /// The layout `ins` (`[q, k, v, ids, pos]`) imply, or the typing error.
    pub fn layout(&self, ins: &[TensorType]) -> Result<SelectedAttentionLayout, ShapeError> {
        arity(ins, 5)?;
        let (q, k, v, ids, pos) = (&ins[0], &ins[1], &ins[2], &ins[3], &ins[4]);
        if self.ratio == 0 || self.k == 0 {
            return Err(ShapeError::SelectedAttention {
                ratio: self.ratio,
                k: self.k,
            });
        }
        if q.dtype.class() != DTypeClass::Arithmetic {
            return Err(ShapeError::DtypeOp {
                op: "SelectedAttention",
                dtype: q.dtype,
            });
        }
        for (operand, t) in [("k", k), ("v", v)] {
            if t.dtype != q.dtype {
                return Err(ShapeError::AttentionDtype {
                    operand,
                    expected: q.dtype,
                    actual: t.dtype,
                });
            }
        }
        for (operand, t) in [("ids", ids), ("pos", pos)] {
            if t.dtype != DType::I32 {
                return Err(ShapeError::AttentionDtype {
                    operand,
                    expected: DType::I32,
                    actual: t.dtype,
                });
            }
        }
        let n_rep = self.n_rep;
        let &[batch, hq, lq, dk] = q.shape.as_slice() else {
            return Err(ShapeError::AttentionQuery {
                n_rep,
                actual: q.shape.clone(),
            });
        };
        if n_rep == 0 || hq % n_rep != 0 {
            return Err(ShapeError::AttentionQuery {
                n_rep,
                actual: q.shape.clone(),
            });
        }
        let hkv = hq / n_rep;
        let Some(&lk) = k.shape.get(2) else {
            return Err(ShapeError::AttentionOperand {
                operand: "k",
                expected: vec![batch, hkv, lq, dk],
                actual: k.shape.clone(),
            });
        };
        let k_expected = [batch, hkv, lk, dk];
        if k.shape != k_expected {
            return Err(ShapeError::AttentionOperand {
                operand: "k",
                expected: k_expected.to_vec(),
                actual: k.shape.clone(),
            });
        }
        let Some(&dv) = v.shape.get(3) else {
            return Err(ShapeError::AttentionOperand {
                operand: "v",
                expected: vec![batch, hkv, lk, dk],
                actual: v.shape.clone(),
            });
        };
        let v_expected = [batch, hkv, lk, dv];
        if dv == 0 || lk == 0 || v.shape != v_expected {
            return Err(ShapeError::AttentionOperand {
                operand: "v",
                expected: v_expected.to_vec(),
                actual: v.shape.clone(),
            });
        }
        let ids_expected = [batch, 1, lq, self.k];
        if ids.shape != ids_expected {
            return Err(ShapeError::AttentionOperand {
                operand: "ids",
                expected: ids_expected.to_vec(),
                actual: ids.shape.clone(),
            });
        }
        if pos.shape != [batch, lq] {
            return Err(ShapeError::AttentionOperand {
                operand: "pos",
                expected: vec![batch, lq],
                actual: pos.shape.clone(),
            });
        }
        Ok(SelectedAttentionLayout {
            batch,
            hq,
            lq,
            lk,
            dk,
            dv,
        })
    }
}

/// The typing rule of `SelectedAttention`: the dense attention's output `[B, Hq, Lq, Dv]`.
fn infer_selected_attention(
    ins: &[TensorType],
    attrs: &SelectedAttentionAttrs,
) -> Result<TensorType, ShapeError> {
    let layout = attrs.layout(ins)?;
    Ok(TensorType::new(
        vec![layout.batch, layout.hq, layout.lq, layout.dv],
        ins[0].dtype,
    ))
}

/// The attributes of [`OpKind::BlockTopK`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct BlockTopKAttrs {
    /// The positions each block covers.
    pub ratio: usize,
    /// The blocks kept per query.
    pub k: usize,
}

/// The two halves of a [`OpKind::BlockTopK`] result `[R, 1, T, K + NB]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct BlockTopKLayout {
    /// Rows `R`.
    pub rows: usize,
    /// Queries per row `T`.
    pub tokens: usize,
    /// The blocks `NB` the scores rank.
    pub blocks: usize,
    /// The blocks kept `K`.
    pub k: usize,
}

impl BlockTopKLayout {
    /// The packed result's shape `[R, 1, T, K + NB]`.
    pub fn packed_shape(&self) -> Vec<usize> {
        vec![self.rows, 1, self.tokens, self.k + self.blocks]
    }

    /// The selected block ids' columns, `0..K`.
    pub fn ids(&self) -> std::ops::Range<usize> {
        0..self.k
    }

    /// The keep indicator's columns, `K..K + NB`.
    pub fn keep(&self) -> std::ops::Range<usize> {
        self.k..self.k + self.blocks
    }
}

impl BlockTopKAttrs {
    /// The layout `ins` (`[scores, pos]`) imply, or the typing error.
    pub fn layout(&self, ins: &[TensorType]) -> Result<BlockTopKLayout, ShapeError> {
        arity(ins, 2)?;
        let (scores, pos) = (&ins[0], &ins[1]);
        if scores.dtype != DType::F32 {
            return Err(ShapeError::DtypeOp {
                op: "BlockTopK",
                dtype: scores.dtype,
            });
        }
        if pos.dtype != DType::I32 {
            return Err(ShapeError::DtypeOp {
                op: "BlockTopK",
                dtype: pos.dtype,
            });
        }
        let &[rows, one, tokens, blocks] = scores.shape.as_slice() else {
            return Err(ShapeError::OperandRank {
                op: "BlockTopK",
                operand: "scores",
                expected: 4,
                actual: scores.shape.len(),
            });
        };
        if one != 1 {
            return Err(ShapeError::OperandDim {
                op: "BlockTopK",
                operand: "scores",
                dim: 1,
                expected: 1,
                actual: one,
            });
        }
        if pos.shape != [rows, tokens] {
            return Err(match pos.shape.as_slice() {
                &[r, t] => ShapeError::OperandDim {
                    op: "BlockTopK",
                    operand: "pos",
                    dim: usize::from(r == rows),
                    expected: if r == rows { tokens } else { rows },
                    actual: if r == rows { t } else { r },
                },
                other => ShapeError::OperandRank {
                    op: "BlockTopK",
                    operand: "pos",
                    expected: 2,
                    actual: other.len(),
                },
            });
        }
        if self.ratio == 0 || self.k == 0 || self.k > blocks {
            return Err(ShapeError::BlockTopK {
                ratio: self.ratio,
                k: self.k,
                blocks,
            });
        }
        Ok(BlockTopKLayout {
            rows,
            tokens,
            blocks,
            k: self.k,
        })
    }
}

/// The typing rule of `BlockTopK`: the packed I32 result of the layout the operands imply.
fn infer_block_top_k(ins: &[TensorType], attrs: &BlockTopKAttrs) -> Result<TensorType, ShapeError> {
    Ok(TensorType::new(
        attrs.layout(ins)?.packed_shape(),
        DType::I32,
    ))
}

/// The typing rule of `Scan`: the packed result of the layout the operands imply.
fn infer_scan(ins: &[TensorType], attrs: &ScanAttrs) -> Result<TensorType, ShapeError> {
    Ok(TensorType::new(
        attrs.layout(ins)?.packed_shape(),
        DType::F32,
    ))
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
    match rotated.last_mut() {
        Some(last) => *last = rot,
        None => {
            return Err(ShapeError::RopeWidth {
                rot,
                x: x.shape.clone(),
            });
        }
    }
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
    let Some((_, leading)) = logits.shape.split_last() else {
        return Err(ShapeError::RankTooLow {
            op: "SampleToken",
            operand: "logits",
            min: 1,
            actual: 0,
        });
    };
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

/// Check the carriers of an indexed packed contraction: one I8 `[experts, slab_bytes]` source per role of
/// `descriptor`, in role order ([`crate::packed_source::packed_stack_source_type`]).
fn check_packed_stack(
    descriptor: PackedWeight,
    experts: usize,
    carriers: &[TensorType],
) -> Result<(), ShapeError> {
    if experts == 0 {
        return Err(ShapeError::PackedStackExperts);
    }
    for (carrier, role) in carriers.iter().zip(descriptor.sources()) {
        let expected = crate::packed_source::packed_stack_source_type(descriptor, experts, role);
        if *carrier != expected {
            return Err(ShapeError::PackedStackOperand {
                role,
                expected: Box::new(expected),
                actual: Box::new(carrier.clone()),
            });
        }
    }
    Ok(())
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

/// The last-axis extent of `shape`, or [`ShapeError::RankTooLow`] for a scalar.
fn last_axis_mut<'a>(
    shape: &'a mut [usize],
    op: &'static str,
    operand: &'static str,
) -> Result<&'a mut usize, ShapeError> {
    shape.last_mut().ok_or(ShapeError::RankTooLow {
        op,
        operand,
        min: 1,
        actual: 0,
    })
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
    use poot_quant::format::WeightFormat;
    use poot_quant::{OperandRole, PackedWeight, SourceRole};

    fn f(shape: &[usize]) -> TensorType {
        TensorType::f32(shape.to_vec())
    }

    /// POOT-747 SC-002 (R466-015): `AllGather` over `world` ranks multiplies the gather axis by `world`.
    /// The baseline returned the input type for every `world`.
    #[test]
    fn all_gather_grows_the_gather_axis_by_world() {
        let world = |n| NonZeroU32::new(n).unwrap();
        let gather = |axis, n| OpKind::AllGather {
            axis,
            world: world(n),
        };
        assert_eq!(gather(1, 2).infer(&[f(&[3, 5])]), Ok(f(&[3, 10])));
        assert_eq!(gather(0, 4).infer(&[f(&[3, 5])]), Ok(f(&[12, 5])));
        assert_eq!(gather(1, 1).infer(&[f(&[3, 5])]), Ok(f(&[3, 5])));
        assert_eq!(
            gather(2, 2).infer(&[f(&[3, 5])]),
            Err(ShapeError::Axis { axis: 2, rank: 2 })
        );
        assert_eq!(
            gather(0, 2).infer(&[f(&[usize::MAX, 1])]),
            Err(ShapeError::AllGatherOverflow {
                axis: 0,
                len: usize::MAX,
                world: world(2)
            })
        );
        // The world is part of the op's identity: two worlds never share a hash or a label.
        assert_ne!(gather(1, 2).name(), gather(1, 4).name());
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

    /// POOT-725: I32 is the only index dtype. Every index-role operand refuses an F32 index with the typed
    /// `DtypeOp` error naming the op, and `ArgTopK` produces I32 ids.
    ///
    /// Mutation: admit `F32` in `check_index_dtype`; every row below goes red.
    #[test]
    fn every_index_role_operand_refuses_an_f32_index() {
        let refused = |op: &'static str, result: Result<TensorType, ShapeError>| {
            assert!(
                matches!(
                    result,
                    Err(ShapeError::DtypeOp { op: got, dtype: DType::F32 }) if got == op
                ),
                "{op}: {result:?}"
            );
        };
        refused(
            "Gather",
            OpKind::Gather { axis: 0 }.infer(&[f(&[4, 5]), f(&[2])]),
        );
        refused(
            "Scatter",
            OpKind::Scatter { axis: 0 }.infer(&[f(&[4, 5]), f(&[4])]),
        );
        refused(
            "ScatterUpdate",
            OpKind::ScatterUpdate.infer(&[f(&[4, 5]), f(&[2, 5]), f(&[2])]),
        );
        refused(
            "IndexedMatMul",
            OpKind::IndexedMatMul.infer(&[f(&[2, 3]), f(&[4, 3, 5]), f(&[2])]),
        );
        refused(
            "DynamicUpdateSlice",
            OpKind::DynamicUpdateSlice { axis: 0 }.infer(&[f(&[4, 5]), f(&[2, 5]), f(&[])]),
        );
        assert_eq!(
            OpKind::ArgTopK { k: 2 }.infer(&[f(&[3, 8])]),
            Ok(TensorType::new(vec![3, 2], DType::I32))
        );
    }

    /// Card 529 (R466-010, SC-002): `infer` rejects an index-role operand outside the index dtype. The baseline
    /// (before Card 529) returned `Ok` for `Gather` and `IndexedMatMul` with a BF16 index; both are typed
    /// errors now.
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

    /// POOT-1048. The grouped contraction types `[G, M, K] x [G * N, K] -> [G, M, N]`, and every malformed
    /// operand is rejected by name, so a grouped chain that does not block the weight the way the activation
    /// groups can never reach a planner or a kernel.
    #[test]
    fn grouped_dense_contraction_types_blocks_and_rejects_malformed_operands() {
        let op = OpKind::GroupedDenseContraction {
            weight: DType::BF16,
            groups: 3,
        };
        let weight = TensorType::new(vec![12, 5], DType::BF16);
        assert_eq!(
            op.infer(&[f(&[3, 2, 5]), weight.clone()]),
            Ok(f(&[3, 2, 4]))
        );
        // The activation groups differently from the op.
        assert!(matches!(
            op.infer(&[f(&[2, 2, 5]), weight.clone()]),
            Err(ShapeError::DenseContractionActivation { expected_k: 5, .. })
        ));
        // Not a `[groups, M, K]` activation.
        assert!(matches!(
            op.infer(&[f(&[3, 5]), weight.clone()]),
            Err(ShapeError::DenseContractionActivation { .. })
        ));
        // Weight rows that do not split into the groups.
        assert!(matches!(
            op.infer(&[f(&[3, 2, 5]), TensorType::new(vec![10, 5], DType::BF16)]),
            Err(ShapeError::GroupedDenseContractionGroups {
                groups: 3,
                rows: 10
            })
        ));
        // A weight already blocked: the op blocks it itself.
        assert!(matches!(
            op.infer(&[f(&[3, 2, 5]), TensorType::new(vec![3, 4, 5], DType::BF16)]),
            Err(ShapeError::DenseContractionWeightRank { .. })
        ));
        // A weight dtype that is not the op's own table row.
        assert!(matches!(
            op.infer(&[f(&[3, 2, 5]), f(&[12, 5])]),
            Err(ShapeError::DenseContractionDtype {
                operand: "weight",
                ..
            })
        ));
        // Only BF16 has grouped device rows.
        for dtype in [DType::F32, DType::F16] {
            assert!(matches!(
                OpKind::GroupedDenseContraction { weight: dtype, groups: 3 }
                    .infer(&[f(&[3, 2, 5]), TensorType::new(vec![12, 5], dtype)]),
                Err(ShapeError::DenseContractionWeightDtype { actual }) if actual == dtype
            ));
        }
        // No groups at all.
        assert!(matches!(
            OpKind::GroupedDenseContraction {
                weight: DType::BF16,
                groups: 0
            }
            .infer(&[f(&[0, 2, 5]), weight]),
            Err(ShapeError::GroupedDenseContractionGroups { groups: 0, .. })
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
            rows: ContractionRows::Blocks(1),
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
            rows: ContractionRows::Blocks(4),
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
            rows: ContractionRows::Blocks(3),
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
            rows: ContractionRows::Blocks(0),
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

    /// An indexed packed contraction (`rows: Indexed`) types `x [M, K]`, ids `[M]` and one expert-stacked,
    /// word-padded carrier per source role to `[M, out]`; a carrier of the per-expert (unstacked) type, one
    /// with the wrong expert count, or ids of the wrong length is refused by name.
    #[test]
    fn indexed_packed_contraction_types_stacked_carriers() {
        // Q8_0, out=3, K=32: 102 bytes per expert, padded to 104.
        let descriptor = PackedWeight::try_new(WeightFormat::Q8_0, [3, 32]).unwrap();
        let experts = 3;
        let slab = descriptor.slab_bytes(SourceRole::Blocks);
        assert_eq!(slab, 104);
        let stacked = TensorType::new(vec![experts, slab], DType::I8);
        let operation = OpKind::PackedContraction {
            descriptor,
            rows: ContractionRows::Indexed { experts },
        };
        let x = TensorType::f32(vec![5, 32]);
        let ids = TensorType::new(vec![5], DType::I32);
        assert_eq!(
            operation.infer(&[x.clone(), ids.clone(), stacked.clone()]),
            Ok(TensorType::f32(vec![5, 3]))
        );
        let per_expert = TensorType::new(
            descriptor.source_shape(SourceRole::Blocks).to_vec(),
            DType::I8,
        );
        for (label, carrier) in [
            ("per-expert shape", per_expert),
            ("two experts", TensorType::new(vec![2, slab], DType::I8)),
            ("unpadded", TensorType::new(vec![experts, 102], DType::I8)),
        ] {
            assert!(
                matches!(
                    operation.infer(&[x.clone(), ids.clone(), carrier]),
                    Err(ShapeError::PackedStackOperand { .. })
                ),
                "{label}"
            );
        }
        assert_eq!(
            operation.infer(&[
                x.clone(),
                TensorType::new(vec![4], DType::I32),
                stacked.clone()
            ]),
            Err(ShapeError::OperandDim {
                op: "PackedContraction",
                operand: "ids",
                dim: 0,
                expected: 5,
                actual: 4,
            })
        );
        assert_eq!(
            OpKind::PackedContraction {
                descriptor,
                rows: ContractionRows::Indexed { experts: 0 },
            }
            .infer(&[x, ids, stacked]),
            Err(ShapeError::PackedStackExperts)
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
    fn attention_rejects_a_kv_or_mask_dtype_mismatch() {
        let op = OpKind::Attention(AttentionAttrs::new(1, 1.0, None));
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
            Err(ShapeError::AttentionDtype {
                operand: "k",
                expected: DType::F32,
                actual: DType::BF16
            })
        ));

        let bf16_v = TensorType::new(vec![1, 1, 3, 4], DType::BF16);
        assert!(matches!(
            op.infer(&[q.clone(), k.clone(), bf16_v, mask.clone()]),
            Err(ShapeError::AttentionDtype {
                operand: "v",
                expected: DType::F32,
                actual: DType::BF16
            })
        ));

        let bf16_mask = TensorType::new(vec![1, 1, 1, 3], DType::BF16);
        assert!(matches!(
            op.infer(&[q, k, v, bf16_mask]),
            Err(ShapeError::AttentionDtype {
                operand: "mask",
                expected: DType::F32,
                actual: DType::BF16
            })
        ));
    }

    /// A sink types over the per-head `[Hq]` logits that follow every other operand: the output is the
    /// sinkless one's, and a missing operand, a logit vector of another width and one of another dtype are
    /// typed refusals naming it.
    #[test]
    fn a_sink_attention_types_over_per_head_logits_and_refuses_each_disagreement() {
        let op = OpKind::Attention(AttentionAttrs {
            sink: SinkOperand::PerHead,
            ..AttentionAttrs::new(2, 1.0, None)
        });
        let ok = [
            f(&[2, 4, 1, 8]),
            f(&[2, 2, 6, 8]),
            f(&[2, 2, 6, 8]),
            f(&[2, 1, 1, 6]),
            f(&[4]),
        ];
        assert_eq!(op.infer(&ok).unwrap(), f(&[2, 4, 1, 8]));
        assert!(matches!(op.infer(&ok[..4]), Err(ShapeError::Arity { .. })));
        let mut wide = ok.clone();
        wide[4] = f(&[2]);
        assert!(matches!(
            op.infer(&wide),
            Err(ShapeError::AttentionOperand {
                operand: "sinks",
                ..
            })
        ));
        let mut bf16 = ok.clone();
        bf16[4] = TensorType::new(vec![4], DType::BF16);
        assert!(matches!(
            op.infer(&bf16),
            Err(ShapeError::AttentionDtype {
                operand: "sinks",
                ..
            })
        ));
        // With a live bound the logits follow the length.
        let live = OpKind::Attention(AttentionAttrs {
            sink: SinkOperand::PerHead,
            bound: KvBound::Live,
            ..AttentionAttrs::new(2, 1.0, None)
        });
        let with_len = [
            ok[0].clone(),
            ok[1].clone(),
            ok[2].clone(),
            ok[3].clone(),
            TensorType::new(vec![2], DType::I32),
            ok[4].clone(),
        ];
        assert_eq!(live.infer(&with_len).unwrap(), f(&[2, 4, 1, 8]));
    }

    /// A causal mask types over the I32 `[B,Lq]` positions in the mask's place: the output is the additive
    /// one's, and positions of another dtype, another shape or a zero window are typed refusals naming them.
    #[test]
    fn a_causal_attention_types_over_query_positions_and_refuses_each_disagreement() {
        let causal = |window| {
            OpKind::Attention(AttentionAttrs {
                mask: MaskOperand::Causal { window },
                ..AttentionAttrs::new(2, 1.0, None)
            })
        };
        let positions = |shape: &[usize], dtype| TensorType::new(shape.to_vec(), dtype);
        let operands = |positions: TensorType| {
            [
                f(&[2, 4, 3, 8]),
                f(&[2, 2, 6, 8]),
                f(&[2, 2, 6, 8]),
                positions,
            ]
        };
        let ok = operands(positions(&[2, 3], DType::I32));
        assert_eq!(causal(None).infer(&ok).unwrap(), f(&[2, 4, 3, 8]));
        assert_eq!(causal(Some(4)).infer(&ok).unwrap(), f(&[2, 4, 3, 8]));
        assert!(matches!(
            causal(None).infer(&operands(f(&[2, 3]))),
            Err(ShapeError::AttentionDtype {
                operand: "positions",
                ..
            })
        ));
        for wrong in [&[1usize, 3][..], &[2, 1], &[2, 1, 3, 6]] {
            assert!(
                matches!(
                    causal(None).infer(&operands(positions(wrong, DType::I32))),
                    Err(ShapeError::AttentionOperand {
                        operand: "positions",
                        ..
                    })
                ),
                "{wrong:?}"
            );
        }
        assert!(matches!(
            causal(Some(0)).infer(&ok),
            Err(ShapeError::AttentionOperand {
                operand: "positions",
                ..
            })
        ));
    }

    /// A paged key types over `[S,Hkv,D]` pools and the `[B,Lk]` I32 slot map: the output is the dense one's,
    /// and every operand that disagrees with the pool or the map is a typed refusal naming it.
    #[test]
    fn a_paged_attention_types_over_pools_and_a_slot_map_and_refuses_each_disagreement() {
        let op = OpKind::Attention(AttentionAttrs {
            key: KeyLayout::Paged,
            ..AttentionAttrs::new(2, 1.0, None)
        });
        let i32_map = |shape: &[usize]| TensorType::new(shape.to_vec(), DType::I32);
        let q = f(&[2, 4, 1, 8]);
        let mask = f(&[2, 1, 1, 6]);
        let ok = [
            q.clone(),
            f(&[9, 2, 8]),
            f(&[9, 2, 5]),
            mask.clone(),
            i32_map(&[2, 6]),
        ];
        assert_eq!(op.infer(&ok).unwrap(), f(&[2, 4, 1, 5]));

        let refused = |operand: usize, replacement: TensorType| {
            let mut ins = ok.clone();
            ins[operand] = replacement;
            op.infer(&ins)
        };
        // The key pool's head width must be the query's, its head count `Hq / n_rep`.
        assert!(matches!(
            refused(1, f(&[9, 2, 7])),
            Err(ShapeError::AttentionOperand { operand: "k", .. })
        ));
        assert!(matches!(
            refused(1, f(&[9, 3, 8])),
            Err(ShapeError::AttentionOperand { operand: "k", .. })
        ));
        // The value pool has the key pool's slots and head count.
        assert!(matches!(
            refused(2, f(&[8, 2, 5])),
            Err(ShapeError::AttentionOperand { operand: "v", .. })
        ));
        // The slot map is I32 `[B, Lk]`, `Lk` the mask's key width.
        assert!(matches!(
            refused(4, f(&[2, 6])),
            Err(ShapeError::AttentionDtype {
                operand: "read",
                expected: DType::I32,
                actual: DType::F32
            })
        ));
        assert!(matches!(
            refused(4, i32_map(&[3, 6])),
            Err(ShapeError::AttentionOperand {
                operand: "read",
                ..
            })
        ));
        assert!(matches!(
            refused(3, f(&[2, 1, 1, 7])),
            Err(ShapeError::AttentionOperand {
                operand: "mask",
                ..
            })
        ));
        // A paged key without its slot map is an arity error, not a dense read of the pool.
        assert!(matches!(op.infer(&ok[..4]), Err(ShapeError::Arity { .. })));
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
        // The condition is an I32 word; the branches share one dtype and the result carries it.
        assert_eq!(
            OpKind::Select
                .infer(&[i(&[2, 1]), f(&[3]), f(&[])])
                .unwrap(),
            f(&[2, 3])
        );
        assert!(matches!(
            OpKind::Select.infer(&[f(&[2]), f(&[2]), f(&[2])]),
            Err(ShapeError::DtypeOp {
                op: "Select",
                dtype: DType::F32
            })
        ));
        assert!(matches!(
            OpKind::Select.infer(&[i(&[2]), f(&[2]), i(&[2])]),
            Err(ShapeError::Dtype { .. })
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
        assert!(
            matches!(
                e,
                Err(ShapeError::OperandRank {
                    expected: 3,
                    actual: 2,
                    ..
                })
            ),
            "got {e:?}"
        );
        // A rank mismatch that does not put axis out of range must also be rejected.
        let e2 = OpKind::Concat { axis: 0 }.infer(&[f(&[2, 3]), f(&[2, 3, 4])]);
        assert!(
            matches!(
                e2,
                Err(ShapeError::OperandRank {
                    expected: 2,
                    actual: 3,
                    ..
                })
            ),
            "got {e2:?}"
        );
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
        assert!(matches!(e, ShapeError::OperandDim { dim: 0, .. }));
    }

    fn i32_ty(shape: &[usize]) -> TensorType {
        TensorType::new(shape.to_vec(), DType::I32)
    }

    /// `n_inputs` external inputs, one step `op(operands)`, the step result as output.
    fn one_step_fused(n_inputs: usize, op: FusedOp, operands: Vec<FusedOperand>) -> FusedRegion {
        FusedRegion {
            n_inputs,
            steps: vec![FusedStep {
                op,
                inputs: operands,
            }],
            output: n_inputs,
            pack: Vec::new(),
        }
    }

    /// SC-002 (R466-025): a `PackLane` operand in a row region is a typed error. The baseline panicked
    /// (`panic!("PackLane cannot appear in an f32 FusedRow region")`), which a malformed region from a
    /// buggy pass reached through `infer`.
    #[test]
    fn row_region_pack_lane_is_a_typed_error_not_a_panic() {
        let region = RowRegion {
            n_inputs: 1,
            axis: 1,
            steps: vec![RowStep::Pointwise {
                op: FusedOp::Unary(UnOp::Neg),
                inputs: vec![FusedOperand::PackLane { input: 0, lane: 0 }],
            }],
            output: 1,
        };
        assert_eq!(
            region.infer(&[i32_ty(&[2, 4])]),
            Err(ShapeError::FusedOperand(
                FusedOperandReason::PackLaneInRowRegion
            ))
        );
        let reduce = RowRegion {
            n_inputs: 1,
            axis: 1,
            steps: vec![RowStep::Reduce {
                op: RedOp::Sum,
                input: FusedOperand::PackLane { input: 0, lane: 0 },
            }],
            output: 1,
        };
        assert_eq!(
            reduce.infer(&[i32_ty(&[2, 4])]),
            Err(ShapeError::FusedOperand(
                FusedOperandReason::PackLaneInRowRegion
            ))
        );
    }

    /// SC-003 (R466-024): each shape-failure class returns its own `ShapeError` variant, asserted
    /// exactly. The rows are the classes the baseline folded into `Arity`, `Concat`, `MatMulRank`,
    /// `MatMulContract`, `Broadcast`, `UpdateSlice` and `PackedContractionActivation`.
    #[test]
    fn each_shape_failure_class_has_its_own_variant() {
        let scalar = TensorType::scalar(DType::F32);
        let i32_scalar = TensorType::scalar(DType::I32);
        let local_neg = |local| {
            OpKind::Fused(one_step_fused(
                1,
                FusedOp::Unary(UnOp::Neg),
                vec![FusedOperand::Local(local)],
            ))
        };
        let pack_neg = |input, lane| {
            OpKind::Fused(one_step_fused(
                1,
                FusedOp::Unary(UnOp::Neg),
                vec![FusedOperand::PackLane { input, lane }],
            ))
        };
        let pack_of = |locals: Vec<usize>| {
            OpKind::Fused(FusedRegion {
                n_inputs: 2,
                steps: Vec::new(),
                output: 0,
                pack: locals,
            })
        };
        let rank_too_low = |op, operand| ShapeError::RankTooLow {
            op,
            operand,
            min: 1,
            actual: 0,
        };
        let operand_rank = |op, operand, expected, actual| ShapeError::OperandRank {
            op,
            operand,
            expected,
            actual,
        };
        let operand_dim = |op, operand, dim, expected, actual| ShapeError::OperandDim {
            op,
            operand,
            dim,
            expected,
            actual,
        };
        let epilogue = OpKind::ContractionEpilogue {
            head: ContractionHead::MatMul,
            region: one_step_fused(
                2,
                FusedOp::Binary(BinOp::Add),
                vec![FusedOperand::Local(0), FusedOperand::Local(1)],
            ),
        };
        let cases: Vec<(&str, OpKind, Vec<TensorType>, ShapeError)> = vec![
            (
                "arg_top_k of a scalar",
                OpKind::ArgTopK { k: 1 },
                vec![scalar.clone()],
                rank_too_low("ArgTopK", "input"),
            ),
            (
                "pack_i8 of a scalar",
                OpKind::PackI8,
                vec![scalar.clone()],
                rank_too_low("PackI8", "input"),
            ),
            (
                "unpack_i8 of a scalar",
                OpKind::UnpackI8 { len: 4 },
                vec![i32_scalar.clone()],
                rank_too_low("UnpackI8", "codes"),
            ),
            (
                "scatter_update of a scalar base",
                OpKind::ScatterUpdate,
                vec![scalar.clone(), f(&[1]), i32_ty(&[1])],
                rank_too_low("ScatterUpdate", "base"),
            ),
            (
                "sample_token of scalar logits",
                OpKind::SampleToken {
                    rule: SampleRule::Greedy,
                },
                vec![scalar.clone()],
                rank_too_low("SampleToken", "logits"),
            ),
            (
                "scatter_update src rank",
                OpKind::ScatterUpdate,
                vec![f(&[4, 2]), f(&[4]), i32_ty(&[4])],
                operand_rank("ScatterUpdate", "src", 2, 1),
            ),
            (
                "scatter_update slots rank",
                OpKind::ScatterUpdate,
                vec![f(&[4, 2]), f(&[3, 2]), i32_ty(&[3, 1])],
                operand_rank("ScatterUpdate", "slots", 1, 2),
            ),
            (
                "concat operand rank",
                OpKind::Concat { axis: 0 },
                vec![f(&[2, 3]), f(&[2, 3, 4])],
                operand_rank("Concat", "input", 2, 3),
            ),
            (
                "indexed_matmul x rank",
                OpKind::IndexedMatMul,
                vec![f(&[2, 3, 4]), f(&[5, 4, 6]), i32_ty(&[2])],
                operand_rank("IndexedMatMul", "x", 2, 3),
            ),
            (
                "indexed_matmul w rank",
                OpKind::IndexedMatMul,
                vec![f(&[2, 4]), f(&[4, 6]), i32_ty(&[2])],
                operand_rank("IndexedMatMul", "w", 3, 2),
            ),
            (
                "indexed_matmul idx rank",
                OpKind::IndexedMatMul,
                vec![f(&[2, 4]), f(&[5, 4, 6]), i32_ty(&[2, 1])],
                operand_rank("IndexedMatMul", "idx", 1, 2),
            ),
            (
                "dynamic_update_slice update rank",
                OpKind::DynamicUpdateSlice { axis: 0 },
                vec![f(&[2, 4]), f(&[1]), i32_scalar.clone()],
                operand_rank("DynamicUpdateSlice", "update", 2, 1),
            ),
            (
                "dynamic_update_slice index rank",
                OpKind::DynamicUpdateSlice { axis: 0 },
                vec![f(&[2, 4]), f(&[1, 4]), i32_ty(&[1])],
                operand_rank("DynamicUpdateSlice", "index", 0, 1),
            ),
            (
                "scatter_update src trailing dim",
                OpKind::ScatterUpdate,
                vec![f(&[4, 2]), f(&[3, 5]), i32_ty(&[3])],
                operand_dim("ScatterUpdate", "src", 1, 2, 5),
            ),
            (
                "scatter_update slots extent",
                OpKind::ScatterUpdate,
                vec![f(&[4, 2]), f(&[3, 2]), i32_ty(&[4])],
                operand_dim("ScatterUpdate", "slots", 0, 3, 4),
            ),
            (
                "indexed_matmul contraction extent",
                OpKind::IndexedMatMul,
                vec![f(&[2, 4]), f(&[5, 3, 6]), i32_ty(&[2])],
                operand_dim("IndexedMatMul", "w", 1, 4, 3),
            ),
            (
                "indexed_matmul idx extent",
                OpKind::IndexedMatMul,
                vec![f(&[2, 4]), f(&[5, 4, 6]), i32_ty(&[3])],
                operand_dim("IndexedMatMul", "idx", 0, 2, 3),
            ),
            (
                "dynamic_update_slice non-axis extent",
                OpKind::DynamicUpdateSlice { axis: 1 },
                vec![f(&[2, 16, 4]), f(&[3, 1, 4]), i32_scalar.clone()],
                operand_dim("DynamicUpdateSlice", "update", 0, 2, 3),
            ),
            (
                "dynamic_update_slice axis extent",
                OpKind::DynamicUpdateSlice { axis: 1 },
                vec![f(&[2, 16, 4]), f(&[2, 17, 4]), i32_scalar.clone()],
                ShapeError::UpdateSlice {
                    operand: vec![2, 16, 4],
                    update: vec![2, 17, 4],
                    axis: 1,
                },
            ),
            (
                "concat non-axis extent",
                OpKind::Concat { axis: 0 },
                vec![f(&[2, 3]), f(&[2, 4])],
                ShapeError::Concat { dim: 1, a: 4, b: 3 },
            ),
            (
                "fused local not defined",
                local_neg(5),
                vec![f(&[2])],
                ShapeError::FusedOperand(FusedOperandReason::LocalNotDefined {
                    local: 5,
                    defined: 1,
                }),
            ),
            (
                "fused pack lane input",
                pack_neg(3, 0),
                vec![i32_ty(&[2, 4])],
                ShapeError::FusedOperand(FusedOperandReason::PackLaneInput {
                    input: 3,
                    inputs: 1,
                }),
            ),
            (
                "fused pack lane width",
                pack_neg(0, 4),
                vec![i32_ty(&[2, 4])],
                ShapeError::FusedOperand(FusedOperandReason::PackLaneWidth {
                    input: 0,
                    lane: 4,
                    width: 4,
                }),
            ),
            (
                "fused pack member without a unit last axis",
                pack_of(vec![0]),
                vec![f(&[2, 3]), f(&[2, 3])],
                ShapeError::FusedPack {
                    local: 0,
                    expected: Box::new(f(&[2, 1])),
                    actual: Box::new(f(&[2, 3])),
                },
            ),
            (
                "fused pack members of different types",
                pack_of(vec![0, 1]),
                vec![f(&[2, 1]), f(&[3, 1])],
                ShapeError::FusedPack {
                    local: 1,
                    expected: Box::new(f(&[2, 1])),
                    actual: Box::new(f(&[3, 1])),
                },
            ),
            (
                "contraction epilogue region result shape",
                epilogue,
                vec![f(&[2, 3]), f(&[3, 4]), f(&[5, 2, 4])],
                ShapeError::EpilogueShape {
                    head: vec![2, 4],
                    region: vec![5, 2, 4],
                },
            ),
        ];
        for (class, op, ins, expected) in cases {
            assert_eq!(op.infer(&ins), Err(expected), "{class}");
        }
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
    /// iotas of different lengths or dtypes must not hash alike while equal ones do.
    #[test]
    fn iota_hash_params_distinguish_lengths_and_dtypes() {
        use std::hash::Hasher;

        fn hash(op: &OpKind) -> u64 {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            op.hash_params(&mut hasher);
            hasher.finish()
        }

        let iota = |len, dtype| OpKind::Iota { len, dtype };
        assert_eq!(hash(&iota(4, DType::F32)), hash(&iota(4, DType::F32)));
        assert_ne!(hash(&iota(4, DType::F32)), hash(&iota(5, DType::F32)));
        assert_ne!(hash(&iota(4, DType::F32)), hash(&iota(4, DType::I32)));
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

    /// Fold one fused-region operand into a hasher. A literal hashes its dtype tag and bit pattern, so `1.0`,
    /// `-0.0` versus `0.0` and distinct NaN payloads all stay distinct. `#[cfg(test)]` alongside
    /// [`OpKind::hash_params`], its only caller.
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

    fn hash_fused_operands<H: std::hash::Hasher>(operands: &[FusedOperand], h: &mut H) {
        use std::hash::Hash;
        operands.len().hash(h);
        for operand in operands {
            hash_fused_operand(operand, h);
        }
    }

    impl OpKind {
        /// Fold this op's own parameters (not just its discriminant) into a hasher, allocation-free (no `Debug` or
        /// `format!`). `OpKind` cannot derive `Hash` because `Attention` etc. carry an `f32 scale`,
        /// which is not `Eq`/`Hash` under IEEE NaN semantics; every f32 field hashes its bit pattern instead. Hashing
        /// only the discriminant cannot distinguish two eqns of the same op kind with different params (a `Slice` at
        /// a different offset, a `DynamicUpdateSlice` on a different axis, a different `DType`).
        ///
        /// `Fused`/`FusedRow` hash their full region description: every step's op and operands (local ids, literal
        /// bit patterns, pack lanes), the output local, the pack list and the row axis. A fused region is the whole
        /// program of one synthesized kernel, so two regions that differ anywhere (`x + 1` versus `x * 2`) must hash
        /// differently.
        ///
        /// `pub(crate)`/test-only (card 546b): the production consumer was `poot-wgpu-device`'s `GraphIdentity`
        /// (guarding `CseMemo`/`DecodeCache` reuse), gone with the rest of the pre-contract executor.
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
                OpKind::Iota { len, dtype } => {
                    len.hash(h);
                    dtype.hash(h);
                }
                OpKind::Gather { axis } => axis.hash(h),
                OpKind::Scatter { axis } => axis.hash(h),
                OpKind::Select
                | OpKind::ScatterUpdate
                | OpKind::MatMul
                | OpKind::IndexedMatMul
                | OpKind::PackI8 => {} // no params beyond the discriminant already hashed
                OpKind::ContractionEpilogue { head, region } => {
                    head.hash(h);
                    region.n_inputs.hash(h);
                    region.steps.len().hash(h);
                    for step in &region.steps {
                        step.op.hash(h);
                        hash_fused_operands(&step.inputs, h);
                    }
                    region.output.hash(h);
                }
                OpKind::PackedDequant { descriptor } => descriptor.hash(h),
                OpKind::PackedContraction { descriptor, rows } => {
                    descriptor.hash(h);
                    rows.hash(h);
                }
                OpKind::PackedRowGather { descriptor } => descriptor.hash(h),
                OpKind::DenseContraction { weight } => weight.hash(h),
                OpKind::GroupedDenseContraction { weight, groups } => {
                    weight.hash(h);
                    groups.hash(h);
                }
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
                OpKind::Attention(attrs) => {
                    attrs.n_rep.hash(h);
                    attrs.scale.to_bits().hash(h);
                    attrs.softcap.map(f32::to_bits).hash(h);
                    attrs.mask.hash(h);
                    attrs.key.hash(h);
                    attrs.bound.hash(h);
                    attrs.sink.hash(h);
                }
                OpKind::Rope { rot } => rot.hash(h),
                OpKind::Scan(attrs) => attrs.hash(h),
                OpKind::BlockTopK(attrs) => attrs.hash(h),
                OpKind::SelectedAttention(attrs) => {
                    attrs.n_rep.hash(h);
                    attrs.scale.to_bits().hash(h);
                    attrs.softcap.map(f32::to_bits).hash(h);
                    attrs.ratio.hash(h);
                    attrs.k.hash(h);
                }
                OpKind::AllReduce { op, axis } => {
                    op.hash(h);
                    axis.hash(h);
                }
                OpKind::AllGather { axis, world } => {
                    axis.hash(h);
                    world.hash(h);
                }
                OpKind::RandomUniform { cols } => cols.hash(h),
                OpKind::SampleToken { rule } => rule.hash(h),
            }
        }
    }
}
