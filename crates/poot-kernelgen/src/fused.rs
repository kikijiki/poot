use std::collections::HashMap;

use poot_target::BufferStorage;

use crate::KernelGenError;
use crate::helpers::{
    Alloc, BodySize, Layout, WeightLane, copy, elem, emit_lds_tree_blocks_at, guard, ld,
    lds_tree_block_count, local, row_major_strides, slice_dtype, slice_f32, view_eff_strides,
};

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, IntScalarOp, Local, MathOp, Operand,
    Place, Rvalue, Statement, Terminator, Ty, UnOp, WorkgroupLocalDecl,
};

/// A scalar op inside a fused elementwise kernel (G5), already lowered to kernel-ir ops by the caller
/// (`poot_graph_plan` translates the graph-ir region). Unary covers `neg`; Math covers the float
/// intrinsics (exp/sqrt); Binary covers the arithmetic ops.
#[derive(Clone, Copy, Debug)]
pub enum FusedScalarOp {
    Unary(UnOp),
    Math(MathOp),
    Binary(BinOp),
    /// Reciprocal `1 / x`. Kernel-ir `UnOp` has no Recip variant, so it is expanded inline as a division
    /// (numerator 1.0). Lets Gated-DeltaNet normalization (`x * recip(sum)`) fuse into one kernel.
    Recip,
    /// Hyperbolic tangent. Kernel-ir has no tanh, so it is expanded inline in the overflow-free
    /// `sign(x) * (1 - t) / (1 + t)`, `t = exp(-2|x|)` form.
    Tanh,
    /// Error function. Kernel-ir has no erf, so it is expanded inline with an abs-folded
    /// Abramowitz-Stegun 7.1.26 polynomial (about 1.5e-7 max abs error).
    Erf,
    /// Pointwise select `(cond, if_true, if_false)`: the bit pattern of `if_true` where the I32 condition
    /// word is nonzero, else of `if_false` ([`emit_select`]). The condition is an I32 leaf; the
    /// branches are F32 or a two-byte float in a float region and I32 in an exact-I32 region.
    Select,
    /// Leading-zero count of an I32 word interpreted as u32. I32 fused regions only.
    Clz,
    /// Unsigned `>=` over I32 storage bits, writing I32 0 or 1. I32 fused regions only.
    GeU,
    /// Unsigned remainder of I32 storage bits (`(a as u32) % (b as u32)`), writing the I32 bit pattern
    /// of the u32 result. I32 fused regions only.
    RemU,
    /// `Cast` of a narrow leaf to f32. The leaf's lane ([`RowKernel::leaf_storage`], an epilogue's
    /// `leaf_storage`) already decodes its elements to f32 as it reads them, so the step is the register
    /// copy that names the widened value; no f32 copy of the stored leaf exists.
    Widen,
    /// `Cast` of an I32 value (a leaf, or an I32 step) to f32, rounding to nearest like the standalone cast
    /// kernel. The one step that takes an I32 word into a float region: the words are read in place and
    /// converted in a register, so neither a producer nor a consumer of the cast needs a kernel for it.
    I32ToF32,
}

/// A kernel-ir comparison op: yields i1, so when it appears in an f32 pipeline its result is cast to f32
/// (1.0 / 0.0). The graph IR currently produces only `Ge`; the others are handled for completeness.
pub(crate) fn is_cmp(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
    )
}

fn f32_lit(v: f32) -> Operand {
    Operand::Const(Constant::F32(v))
}

/// Straight-line emitter of f32 temporaries for the multi-instruction scalar expansions.
struct Emitter<'a> {
    al: &'a mut Alloc,
    stmts: &'a mut Vec<Statement>,
}

impl Emitter<'_> {
    /// `tmp = rv` into a fresh f32 local; returns a copy-operand of it.
    fn rv(&mut self, rv: Rvalue) -> Operand {
        self.typed(Ty::F32, rv)
    }

    /// `tmp = rv` into a fresh local of `ty`; returns a copy-operand of it.
    fn typed(&mut self, ty: Ty, rv: Rvalue) -> Operand {
        let d = self.al.add(ty, false);
        self.stmts.push(Statement::Assign(Place::local(d), rv));
        copy(Place::local(d))
    }

    fn bin(&mut self, a: Operand, op: BinOp, b: Operand) -> Operand {
        self.rv(Rvalue::BinaryOp(op, a, b))
    }

    fn math(&mut self, op: MathOp, x: Operand) -> Operand {
        self.rv(Rvalue::MathUnary(op, x))
    }

    /// `+1.0` where `x >= 0`, `-1.0` where `x < 0`; NaN maps to `-1.0`, harmless because the magnitude
    /// it multiplies is already NaN.
    fn sign(&mut self, x: Operand) -> Operand {
        let ge = self.al.add(Ty::Bool, false);
        self.stmts.push(Statement::Assign(
            Place::local(ge),
            Rvalue::BinaryOp(BinOp::Ge, x, f32_lit(0.0)),
        ));
        let one_or_zero = self.rv(Rvalue::Cast {
            to: Ty::F32,
            operand: copy(Place::local(ge)),
        });
        let twice = self.bin(one_or_zero, BinOp::Mul, f32_lit(2.0));
        self.bin(twice, BinOp::Sub, f32_lit(1.0))
    }
}

/// Emit `dst = op(operands...)` into `stmts`, allocating any temporaries from `al`. Most ops are a single
/// instruction; `Tanh` and `Erf` expand to inline sequences. Shared by both fused synthesizers.
pub(crate) fn emit_scalar_op(
    dst: Local,
    op: FusedScalarOp,
    operands: &[Operand],
    al: &mut Alloc,
    stmts: &mut Vec<Statement>,
) {
    let rv = match op {
        FusedScalarOp::Unary(u) => Rvalue::UnaryOp(u, operands[0].clone()),
        FusedScalarOp::Math(m) => Rvalue::MathUnary(m, operands[0].clone()),
        FusedScalarOp::Binary(b) if is_cmp(b) => {
            // a comparison yields i1; compute the bool then cast to f32 (1.0 / 0.0) to stay in the f32 chain.
            let bcmp = al.add(Ty::Bool, false);
            stmts.push(Statement::Assign(
                Place::local(bcmp),
                Rvalue::BinaryOp(b, operands[0].clone(), operands[1].clone()),
            ));
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(Place::local(bcmp)),
            }
        }
        FusedScalarOp::Binary(b) => Rvalue::BinaryOp(b, operands[0].clone(), operands[1].clone()),
        FusedScalarOp::Widen => Rvalue::Use(operands[0].clone()),
        FusedScalarOp::I32ToF32 => Rvalue::Cast {
            to: Ty::F32,
            operand: operands[0].clone(),
        },
        FusedScalarOp::Recip => Rvalue::BinaryOp(
            BinOp::Div,
            Operand::Const(Constant::F32(1.0)),
            operands[0].clone(),
        ),
        FusedScalarOp::Tanh => {
            // tanh(x) = sign(x) * (1 - t) / (1 + t), t = exp(-2|x|). The exponent is never positive, so
            // `exp` cannot overflow: tanh(+-100) and tanh(+-inf) are exactly +-1 (the textbook
            // `(e^x - e^-x) / (e^x + e^-x)` is inf/inf = NaN there).
            let x = operands[0].clone();
            let mut em = Emitter { al, stmts };
            let ax = em.math(MathOp::Abs, x.clone());
            let m2a = em.bin(ax, BinOp::Mul, f32_lit(-2.0));
            let t = em.math(MathOp::Exp, m2a);
            let num = em.bin(f32_lit(1.0), BinOp::Sub, t.clone());
            let den = em.bin(f32_lit(1.0), BinOp::Add, t);
            let mag = em.bin(num, BinOp::Div, den);
            let sign = em.sign(x);
            Rvalue::BinaryOp(BinOp::Mul, mag, sign)
        }
        FusedScalarOp::Erf => {
            // erf(x) = sign(x) * E(a), a = |x|, with Abramowitz-Stegun 7.1.26 (no branch):
            //   t = 1/(1 + p*a),  E(a) = 1 - poly(t)*exp(-a^2),
            //   poly(t) = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))  (Horner).
            const P: f32 = 0.327_591_1;
            const A1: f32 = 0.254_829_6;
            const A2: f32 = -0.284_496_72;
            const A3: f32 = 1.421_413_8;
            const A4: f32 = -1.453_152_1;
            const A5: f32 = 1.061_405_4;
            let x = operands[0].clone();
            let mut em = Emitter { al, stmts };
            let ax = em.math(MathOp::Abs, x.clone());
            let pa = em.bin(ax, BinOp::Mul, f32_lit(P));
            let den = em.bin(pa, BinOp::Add, f32_lit(1.0));
            let t = em.bin(f32_lit(1.0), BinOp::Div, den);
            let mut h = em.rv(Rvalue::Use(f32_lit(A5)));
            for &ak in &[A4, A3, A2, A1] {
                let th = em.bin(t.clone(), BinOp::Mul, h);
                h = em.bin(f32_lit(ak), BinOp::Add, th);
            }
            let poly = em.bin(t, BinOp::Mul, h);
            let x2 = em.bin(x.clone(), BinOp::Mul, x.clone());
            let na2 = em.bin(x2, BinOp::Mul, f32_lit(-1.0));
            let e = em.math(MathOp::Exp, na2);
            let pe = em.bin(poly, BinOp::Mul, e);
            let mag = em.bin(f32_lit(1.0), BinOp::Sub, pe);
            let sign = em.sign(x);
            Rvalue::BinaryOp(BinOp::Mul, mag, sign)
        }
        FusedScalarOp::Select => emit_select(
            al,
            stmts,
            operands[0].clone(),
            operands[1].clone(),
            operands[2].clone(),
        ),
        FusedScalarOp::Clz | FusedScalarOp::GeU | FusedScalarOp::RemU => {
            panic!("I32 fused op in the f32 synthesizer")
        }
    };
    stmts.push(Statement::Assign(Place::local(dst), rv));
}

/// An operand of a fused step: a leaf input (read with broadcast strides), a prior step's register, or an
/// inline literal.
#[derive(Clone, Copy, Debug)]
pub enum FusedInput {
    Leaf(usize),
    Step(usize),
    Lit(f32),
    LitI32(i32),
    /// Last-axis lane of a packed I32 leaf. The leaf's last dimension is the pack width.
    LeafLane {
        leaf: usize,
        lane: usize,
    },
}

/// One step of a fused region: `step[s] = op(inputs...)`, computed in a register.
#[derive(Clone, Debug)]
pub struct FusedStep {
    pub op: FusedScalarOp,
    pub inputs: Vec<FusedInput>,
}

/// The element type a fused kernel reads one leaf as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafTy {
    F32,
    I32,
    /// A two-byte float leaf, read as stored. Only a `Select` step may read it (it moves the element, it
    /// never computes on it), and a float region's output may be such a `Select`.
    BF16,
    F16,
}

impl LeafTy {
    fn ty(self) -> Ty {
        match self {
            Self::F32 => Ty::F32,
            Self::I32 => Ty::I32,
            Self::BF16 => Ty::BF16,
            Self::F16 => Ty::F16,
        }
    }
}

/// The recipe for a fused elementwise kernel: one input slice per entry of `leaf_tys`, a straight-line list of scalar
/// steps, and which value is written out. The kernelgen-native mirror of `poot_graph_ir::FusedRegion`
/// (kept separate so kernelgen does not depend on the graph IR).
#[derive(Clone, Debug)]
pub struct FusedKernel {
    /// The element type of each leaf, one per leaf. A float region reads F32 leaves, plus I32 leaves that
    /// only a `Select` step may read (its condition); an exact-I32 region reads I32 leaves only.
    pub leaf_tys: Vec<LeafTy>,
    pub steps: Vec<FusedStep>,
    pub output: FusedInput,
}

/// A last-axis reduction op inside a row-wise fused region.
#[derive(Clone, Copy, Debug)]
pub enum RowReduce {
    Sum,
    Max,
}

/// One step of a row-wise fused region: a pointwise op (a Map over the row when any operand is a row
/// value, a per-row Scalar op otherwise) or a last-axis reduction (a row value -> a per-row scalar).
#[derive(Clone, Debug)]
pub enum RowOp {
    Pointwise {
        op: FusedScalarOp,
        inputs: Vec<FusedInput>,
    },
    Reduce {
        op: RowReduce,
        input: FusedInput,
    },
}

/// The recipe for a row-wise (reduction-rooted) fused kernel: `n_leaves` input slices, `n_cols` = the
/// reduction extent N, a straight-line list of steps, and which value is written out (a row value). The
/// kernelgen-native mirror of `poot_graph_ir::RowRegion`. Lowered by [`fused_row`] to one kernel that
/// processes a row per thread, computing each reduction into a register and recomputing the pointwise
/// chain per pass.
#[derive(Clone, Debug)]
pub struct RowKernel {
    pub n_leaves: usize,
    /// How each leaf's elements sit in its buffer, one per leaf: f32 for a leaf read as stored, packed
    /// BF16 or F16 for a narrow leaf a [`FusedScalarOp::Widen`] step widens in registers.
    pub leaf_storage: Vec<BufferStorage>,
    pub n_cols: usize,
    pub steps: Vec<RowOp>,
    pub output: usize,
}

impl FusedKernel {
    /// The number of input leaves.
    pub fn n_leaves(&self) -> usize {
        self.leaf_tys.len()
    }
}

impl FusedScalarOp {
    /// An upper bound of what one step of this op adds to a body, over the f32 and the I32 emitters: its
    /// statements and the fresh temporaries it allocates, not counting the step's own result register.
    fn size(self) -> BodySize {
        let (instructions, locals) = match self {
            Self::Unary(_)
            | Self::Math(_)
            | Self::Recip
            | Self::Clz
            | Self::Widen
            | Self::I32ToF32 => (1, 0),
            Self::Binary(BinOp::Shl) => (2, 1),
            Self::Binary(BinOp::Shr) => (5, 4),
            Self::Binary(op) if is_cmp(op) => (2, 1),
            Self::Binary(_) => (1, 0),
            Self::Tanh => (11, 10),
            Self::Erf => (24, 23),
            Self::Select => (2, 1),
            Self::GeU | Self::RemU => (4, 3),
        };
        BodySize {
            instructions,
            locals,
        }
    }

    /// [`Self::size`] plus the step's result register.
    fn step_size(self) -> BodySize {
        self.size().plus(BodySize {
            instructions: 0,
            locals: 1,
        })
    }
}

/// What reading one leaf at its broadcast offset adds, over `rank` output axes: the offset init, five
/// statements per strided axis and the load; the offset and value registers, and four index temporaries
/// per axis where the emitter allocates them per read.
fn leaf_read_size(rank: usize) -> BodySize {
    BodySize {
        instructions: rank.saturating_mul(5).saturating_add(2),
        locals: rank.saturating_mul(4).saturating_add(2),
    }
}

/// What decoding one element of a packed 16-bit leaf adds over a plain load: the word index, the half select,
/// the shift and mask, and the widening (the binary16 decode is the larger of the two lanes).
const PACKED_LANE_READ: BodySize = BodySize {
    instructions: 32,
    locals: 32,
};

/// A checked upper bound on the body [`fused_views`], [`fused_i32_views_grid_pack`] build for `k` over
/// an output of `rank` axes and a `pack_lanes`-wide packed store. Computed from the recipe alone, so a
/// caller can refuse a region before any statement is built; `fused_size_bounds_the_generated_bodies`
/// holds it at or above what the generators emit.
pub(crate) fn pointwise_size(k: &FusedKernel, rank: usize, pack_lanes: usize) -> BodySize {
    const FIXED: BodySize = BodySize {
        instructions: 32,
        // The thread index, length, guard and index scratch, the packed store offset and the 2-D grid
        // fold registers, with the output parameter.
        locals: 24,
    };
    let lane_refs = k
        .steps
        .iter()
        .flat_map(|step| &step.inputs)
        .chain(std::iter::once(&k.output))
        .filter(|input| matches!(input, FusedInput::LeafLane { .. }))
        .count()
        .saturating_add(pack_lanes);
    let read = leaf_read_size(rank);
    let params = BodySize {
        instructions: 0,
        locals: k.n_leaves(),
    };
    let steps = k.steps.iter().fold(BodySize::default(), |sum, step| {
        sum.plus(step.op.step_size())
    });
    FIXED
        .plus(params)
        .plus(read.times(k.n_leaves().saturating_add(lane_refs)))
        .plus(steps)
        .plus(BodySize {
            instructions: pack_lanes.saturating_mul(3),
            locals: 0,
        })
}

/// A checked upper bound on the body [`fused_row_parallel_views`] builds for `k` over an output of
/// `rank` axes at workgroup width `w`: the straight-line scalar steps once, and per reduction (plus the
/// output pass) one recomputation of the whole pointwise chain, a strided loop and an LDS tree.
pub(crate) fn row_size(k: &RowKernel, rank: usize, w: usize) -> BodySize {
    const FIXED: BodySize = BodySize {
        instructions: 40,
        locals: 24,
    };
    let reductions = k
        .steps
        .iter()
        .filter(|step| matches!(step, RowOp::Reduce { .. }))
        .count();
    let chain = k
        .steps
        .iter()
        .fold(BodySize::default(), |sum, step| match step {
            RowOp::Pointwise { op, .. } => sum.plus(op.step_size()),
            RowOp::Reduce { .. } => sum,
        });
    let packed_leaves = k
        .leaf_storage
        .iter()
        .filter(|&&storage| storage != BufferStorage::f32())
        .count();
    let evaluation = leaf_read_size(rank)
        .times(k.n_leaves)
        .plus(PACKED_LANE_READ.times(packed_leaves))
        .plus(chain)
        .plus(BodySize {
            instructions: 8,
            locals: 1,
        });
    let reduction = BodySize {
        instructions: lds_tree_block_count(w)
            .max(1)
            .saturating_mul(6)
            .saturating_add(8) as usize,
        locals: 5,
    };
    FIXED
        .plus(BodySize {
            instructions: 0,
            locals: k.n_leaves.saturating_add(k.steps.len()),
        })
        .plus(chain)
        .plus(evaluation.times(reductions.saturating_add(1)))
        .plus(reduction.times(reductions))
}

/// Synthesize one elementwise kernel for a fused region: a 1-D grid covers the output; each thread reads
/// every leaf (with its broadcast strides), evaluates the straight-line recipe in registers and writes one
/// output element. One dispatch, no intermediate buffers. Generalizes `binary_broadcast` to N leaves and a
/// DAG of scalar steps. `leaf_shapes[j]` is leaf `j`'s shape (for its broadcast-effective strides into `out_shape`).
pub fn fused(
    name: &str,
    out_shape: &[usize],
    leaf_shapes: &[&[usize]],
    k: &FusedKernel,
) -> Result<Body, KernelGenError> {
    let layouts: Vec<Layout> = leaf_shapes.iter().map(|s| Layout::contiguous(s)).collect();
    fused_views(name, out_shape, leaf_shapes, &layouts, k, None)
}

/// The element types of a float region: each step's result type and the output's.
struct FloatRegionTys {
    steps: Vec<Ty>,
    output: Ty,
}

/// Type a region the float synthesizer lowers and refuse what no body can state. A step computes in F32,
/// except that a step over I32 words is an I32 step: exact integer arithmetic (a comparison writes I32
/// 0 or 1) that only a `Select` or an `I32ToF32` step turns into a float. So an I32 producer chain and the
/// float chain that consumes it are one region, with the cast between them. An I32 leaf feeds an I32 step,
/// a `Select` condition or a cast; mixing I32 and F32 operands in one arithmetic step has no meaning and is
/// refused. A two-byte float leaf or step is moved, never computed on, so only a `Select`'s branches (and
/// the region output) may be one; a `Select` over words of one type is itself that type, its two branches
/// agree. The region output is a float: an I32 result is the exact-I32 synthesizer's.
fn check_float_region_tys(
    generator: &'static str,
    k: &FusedKernel,
) -> Result<FloatRegionTys, KernelGenError> {
    let leaf_ty = |leaf: usize| k.leaf_tys.get(leaf).copied().map(LeafTy::ty);
    let mut steps: Vec<Ty> = Vec::with_capacity(k.steps.len());
    let input_ty = |input: &FusedInput, steps: &[Ty]| match input {
        FusedInput::Leaf(leaf) => leaf_ty(*leaf).unwrap_or(Ty::F32),
        FusedInput::Step(step) => steps.get(*step).cloned().unwrap_or(Ty::F32),
        FusedInput::LitI32(_) => Ty::I32,
        FusedInput::Lit(_) | FusedInput::LeafLane { .. } => Ty::F32,
    };
    let consumer_error = |input: &FusedInput, ty: &Ty, consumer: String| match (input, ty) {
        (FusedInput::Leaf(leaf), Ty::I32) => KernelGenError::IndexLeafOperand {
            generator,
            leaf: *leaf,
            consumer,
        },
        _ => KernelGenError::NarrowOperand {
            generator,
            consumer,
        },
    };
    for (index, step) in k.steps.iter().enumerate() {
        let operand_tys: Vec<Ty> = step
            .inputs
            .iter()
            .map(|input| input_ty(input, &steps))
            .collect();
        let refuse = |position: usize| {
            consumer_error(
                &step.inputs[position],
                &operand_tys[position],
                format!("{:?}", step.op),
            )
        };
        let result = match step.op {
            FusedScalarOp::I32ToF32 => {
                if operand_tys.first() != Some(&Ty::I32) {
                    return Err(KernelGenError::IndexCastOperand {
                        generator,
                        step: index,
                    });
                }
                Ty::F32
            }
            FusedScalarOp::Select => {
                if operand_tys.first() != Some(&Ty::I32) {
                    return Err(refuse(0));
                }
                let branches = operand_tys.get(1..).unwrap_or_default();
                let first = branches.first().cloned();
                if branches.iter().any(|ty| Some(ty) != first.as_ref()) {
                    return Err(KernelGenError::NarrowOperand {
                        generator,
                        consumer: "a Select whose branches differ in type".to_string(),
                    });
                }
                first.unwrap_or(Ty::F32)
            }
            FusedScalarOp::GeU
            | FusedScalarOp::RemU
            | FusedScalarOp::Clz
            | FusedScalarOp::Unary(UnOp::Not)
            | FusedScalarOp::Binary(_)
                if !operand_tys.is_empty() && operand_tys.iter().all(|ty| ty == &Ty::I32) =>
            {
                Ty::I32
            }
            _ => {
                // A float step: an I32 or a two-byte operand has no float meaning here.
                if let Some(position) = operand_tys.iter().position(|ty| ty != &Ty::F32) {
                    return Err(refuse(position));
                }
                Ty::F32
            }
        };
        steps.push(result);
    }
    let output = input_ty(&k.output, &steps);
    if output == Ty::I32 {
        return Err(consumer_error(
            &k.output,
            &output,
            "the region output".to_string(),
        ));
    }
    Ok(FloatRegionTys { steps, output })
}

/// [`fused`] reading each leaf through its own physical [`Layout`]. Byte-identical to `fused` when every layout
/// is `Layout::contiguous` and `x_groups` is `None`. `Some(xg)` folds the 1-D element index onto a 2-D grid like
/// [`fused_i32_views_grid`], past the target's X grid cap: it must match the launch's `x_groups`
/// (`fold_groups`), and the caller must bake `workgroup_size[0] = ELEMENTWISE_2D_WORKGROUP_SIZE`.
pub fn fused_views(
    name: &str,
    out_shape: &[usize],
    leaf_shapes: &[&[usize]],
    leaf_layouts: &[Layout],
    k: &FusedKernel,
    x_groups: Option<usize>,
) -> Result<Body, KernelGenError> {
    if leaf_shapes.len() != k.n_leaves() {
        return Err(KernelGenError::CountMismatch {
            generator: "fused_views",
            what: "leaf shape count".to_string(),
            expected: k.n_leaves(),
            actual: leaf_shapes.len(),
        });
    }
    if leaf_layouts.len() != k.n_leaves() {
        return Err(KernelGenError::CountMismatch {
            generator: "fused_views",
            what: "leaf layout count".to_string(),
            expected: k.n_leaves(),
            actual: leaf_layouts.len(),
        });
    }
    let region_tys = check_float_region_tys("fused_views", k)?;
    let r = out_shape.len();
    let out_strides = row_major_strides(out_shape);
    let n = k.n_leaves();

    // params: leaves are locals 1..=n (read-only slices), the output is local n+1 (mut slice).
    let mut locals = vec![ld(Ty::Unit, false)];
    for leaf in &k.leaf_tys {
        locals.push(ld(slice_dtype(leaf.ty(), false), false));
    }
    locals.push(ld(slice_dtype(region_tys.output.clone(), true), true));
    let mut al = Alloc::new(locals);
    let out_local = local((n + 1) as u32);

    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    // index-math scratch, reused across leaves.
    let div = al.add(Ty::Usize, false);
    let md = al.add(Ty::Usize, false);
    let term = al.add(Ty::Usize, false);
    let off_new = al.add(Ty::Usize, false);
    // per-leaf running offset + the read value register.
    let leaf_off: Vec<Local> = (0..n).map(|_| al.add(Ty::Usize, true)).collect();
    let leaf_val: Vec<Local> = k
        .leaf_tys
        .iter()
        .map(|leaf| al.add(leaf.ty(), false))
        .collect();
    // per-step result register.
    let step_reg: Vec<Local> = region_tys
        .steps
        .iter()
        .map(|ty| al.add(ty.clone(), false))
        .collect();
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));

    let mut body: Vec<Statement> = Vec::new();
    // read each leaf at its view-mapped offset (base 0, row-major eff strides, for a plain buffer).
    for (li, lshape) in leaf_shapes.iter().enumerate() {
        let (eff, base) = view_eff_strides(out_shape, lshape, &leaf_layouts[li]);
        let off = leaf_off[li];
        body.push(Statement::Assign(Place::local(off), Rvalue::Use(cu(base))));
        for (d, &e) in eff.iter().enumerate().take(r) {
            if e == 0 {
                continue; // broadcast dim
            }
            body.push(Statement::Assign(
                Place::local(div),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(out_strides[d])),
            ));
            body.push(Statement::Assign(
                Place::local(md),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(div)), cu(out_shape[d])),
            ));
            body.push(Statement::Assign(
                Place::local(term),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(md)), cu(e)),
            ));
            body.push(Statement::Assign(
                Place::local(off_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(off)),
                    copy(Place::local(term)),
                ),
            ));
            body.push(Statement::Assign(
                Place::local(off),
                Rvalue::Use(copy(Place::local(off_new))),
            ));
        }
        body.push(Statement::Assign(
            Place::local(leaf_val[li]),
            Rvalue::Use(copy(elem(local((li + 1) as u32), off))),
        ));
    }

    // resolve a recipe operand to a register copy or an inline constant.
    let resolve = |inp: &FusedInput| -> Operand {
        match inp {
            FusedInput::Leaf(j) => copy(Place::local(leaf_val[*j])),
            FusedInput::Step(s) => copy(Place::local(step_reg[*s])),
            FusedInput::Lit(c) => Operand::Const(Constant::F32(*c)),
            FusedInput::LitI32(c) => Operand::Const(Constant::I32(*c)),
            FusedInput::LeafLane { .. } => {
                panic!("a float fused kernel cannot carry an I32 packed lane")
            }
        }
    };
    // evaluate the steps in order (each references only earlier leaves/steps); a step over I32 words is
    // emitted by the exact-integer emitter.
    for (s, step) in k.steps.iter().enumerate() {
        let operands: Vec<Operand> = step.inputs.iter().map(&resolve).collect();
        if region_tys.steps[s] == Ty::I32 {
            emit_scalar_op_i32(step_reg[s], step.op, &operands, &mut al, &mut body);
        } else {
            emit_scalar_op(step_reg[s], step.op, &operands, &mut al, &mut body);
        }
    }
    // out[i] = the region output value.
    body.push(Statement::Assign(
        elem(out_local, i),
        Rvalue::Use(resolve(&k.output)),
    ));

    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(out_local))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: body,
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Ok(pointwise_body(
        name,
        (n + 1) as u32,
        al,
        i,
        bb1,
        bb2,
        bb3,
        x_groups,
    ))
}

/// Typed-I32 fused elementwise kernel. Locals, buffers, and literals stay I32; comparison steps write
/// I32 0/1. This is the exact-word counterpart of [`fused_views`], with the same optional 2-D grid fold
/// as [`crate::binary_broadcast_dt_views_grid`] (`x_groups: None` keeps the original flat
/// `ThreadIndexCall(X)` body - the convenience `fused_i32_views` used to expose before it moved to
/// `poot_test_util::kernel_fixtures`, card 671, its only caller; `Some(xg)` must match
/// the launch's `x_groups` (`kg::fold_groups`), and the caller must bake `workgroup_size[0] =
/// ELEMENTWISE_2D_WORKGROUP_SIZE`).
pub fn fused_i32_views_grid(
    name: &str,
    out_shape: &[usize],
    leaf_shapes: &[&[usize]],
    leaf_layouts: &[Layout],
    k: &FusedKernel,
    x_groups: Option<usize>,
) -> Result<Body, KernelGenError> {
    fused_i32_views_grid_pack(name, out_shape, leaf_shapes, leaf_layouts, k, x_groups, &[])
}

/// [`fused_i32_views_grid`] with a last-axis packed store. `pack` locals are written as consecutive
/// lanes of `out_shape`'s last axis. The launch covers the prefix numel, not the packed numel.
pub fn fused_i32_views_grid_pack(
    name: &str,
    out_shape: &[usize],
    leaf_shapes: &[&[usize]],
    leaf_layouts: &[Layout],
    k: &FusedKernel,
    x_groups: Option<usize>,
    pack: &[FusedInput],
) -> Result<Body, KernelGenError> {
    if leaf_shapes.len() != k.n_leaves() {
        return Err(KernelGenError::CountMismatch {
            generator: "fused_i32_views_grid_pack",
            what: "leaf shape count".to_string(),
            expected: k.n_leaves(),
            actual: leaf_shapes.len(),
        });
    }
    if leaf_layouts.len() != k.n_leaves() {
        return Err(KernelGenError::CountMismatch {
            generator: "fused_i32_views_grid_pack",
            what: "leaf layout count".to_string(),
            expected: k.n_leaves(),
            actual: leaf_layouts.len(),
        });
    }
    let float_leaves = k
        .leaf_tys
        .iter()
        .filter(|&&leaf| leaf != LeafTy::I32)
        .count();
    if k.leaf_tys.len() != k.n_leaves() || float_leaves != 0 {
        return Err(KernelGenError::NotEqualTo {
            generator: "fused_i32_views_grid_pack",
            dim: "F32 leaves of an exact-I32 region".to_string(),
            value: float_leaves,
            expected: 0,
        });
    }
    let lanes = pack.len();
    let index_shape: Vec<usize> = if lanes == 0 {
        out_shape.to_vec()
    } else {
        let out_last = out_shape.last().copied().unwrap_or(0);
        if out_last != lanes {
            return Err(KernelGenError::CountMismatch {
                generator: "fused_i32_views_grid_pack",
                what: "packed fused output last axis vs pack width".to_string(),
                expected: lanes,
                actual: out_last,
            });
        }
        let mut shape = out_shape.to_vec();
        if let Some(last) = shape.last_mut() {
            *last = 1;
        }
        if shape.is_empty() {
            shape.push(1);
        }
        shape
    };
    let r = index_shape.len();
    let out_strides = row_major_strides(&index_shape);
    let n = k.n_leaves();

    let mut locals = vec![ld(Ty::Unit, false)];
    for _ in 0..n {
        locals.push(ld(slice_dtype(Ty::I32, false), false));
    }
    locals.push(ld(slice_dtype(Ty::I32, true), true));
    let mut al = Alloc::new(locals);
    let out_local = local((n + 1) as u32);

    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let div = al.add(Ty::Usize, false);
    let md = al.add(Ty::Usize, false);
    let term = al.add(Ty::Usize, false);
    let off_new = al.add(Ty::Usize, false);
    let leaf_off: Vec<Local> = (0..n).map(|_| al.add(Ty::Usize, true)).collect();
    let leaf_val: Vec<Local> = (0..n).map(|_| al.add(Ty::I32, false)).collect();
    let step_reg: Vec<Local> = (0..k.steps.len()).map(|_| al.add(Ty::I32, false)).collect();
    let store_off = if lanes == 0 {
        None
    } else {
        Some(al.add(Ty::Usize, false))
    };
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));

    let mut used_leaf = vec![false; n];
    let mut used_lanes: Vec<(usize, usize)> = Vec::new();
    let mut mark = |inp: &FusedInput| match inp {
        FusedInput::Leaf(j) => used_leaf[*j] = true,
        FusedInput::LeafLane { leaf, lane } => used_lanes.push((*leaf, *lane)),
        _ => {}
    };
    for step in &k.steps {
        for inp in &step.inputs {
            mark(inp);
        }
    }
    mark(&k.output);
    for packed in pack {
        mark(packed);
    }
    used_lanes.sort_unstable();
    used_lanes.dedup();
    let mut lane_val: HashMap<(usize, usize), Local> = HashMap::new();
    for &(leaf, lane) in &used_lanes {
        lane_val.insert((leaf, lane), al.add(Ty::I32, false));
    }

    let mut body: Vec<Statement> = Vec::new();
    for (li, lshape) in leaf_shapes.iter().enumerate() {
        if !used_leaf[li] {
            continue;
        }
        let (eff, base) = view_eff_strides(&index_shape, lshape, &leaf_layouts[li]);
        let off = leaf_off[li];
        body.push(Statement::Assign(Place::local(off), Rvalue::Use(cu(base))));
        for (d, &e) in eff.iter().enumerate().take(r) {
            if e == 0 {
                continue;
            }
            body.push(Statement::Assign(
                Place::local(div),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(out_strides[d])),
            ));
            body.push(Statement::Assign(
                Place::local(md),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(div)), cu(index_shape[d])),
            ));
            body.push(Statement::Assign(
                Place::local(term),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(md)), cu(e)),
            ));
            body.push(Statement::Assign(
                Place::local(off_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(off)),
                    copy(Place::local(term)),
                ),
            ));
            body.push(Statement::Assign(
                Place::local(off),
                Rvalue::Use(copy(Place::local(off_new))),
            ));
        }
        body.push(Statement::Assign(
            Place::local(leaf_val[li]),
            Rvalue::Use(copy(elem(local((li + 1) as u32), off))),
        ));
    }
    for &(leaf, lane) in &used_lanes {
        let lshape = leaf_shapes[leaf];
        let layout = &leaf_layouts[leaf];
        let (eff, base) = view_eff_strides(&index_shape, lshape, layout);
        let off = leaf_off[leaf];
        body.push(Statement::Assign(Place::local(off), Rvalue::Use(cu(base))));
        for (d, &e) in eff.iter().enumerate().take(r) {
            if e == 0 {
                continue;
            }
            body.push(Statement::Assign(
                Place::local(div),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(out_strides[d])),
            ));
            body.push(Statement::Assign(
                Place::local(md),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(div)), cu(index_shape[d])),
            ));
            body.push(Statement::Assign(
                Place::local(term),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(md)), cu(e)),
            ));
            body.push(Statement::Assign(
                Place::local(off_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(off)),
                    copy(Place::local(term)),
                ),
            ));
            body.push(Statement::Assign(
                Place::local(off),
                Rvalue::Use(copy(Place::local(off_new))),
            ));
        }
        let last_stride = layout.strides.last().copied().unwrap_or(1);
        if lane != 0 && last_stride != 0 {
            body.push(Statement::Assign(
                Place::local(off),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(off)), cu(lane * last_stride)),
            ));
        }
        body.push(Statement::Assign(
            Place::local(lane_val[&(leaf, lane)]),
            Rvalue::Use(copy(elem(local((leaf + 1) as u32), off))),
        ));
    }

    let resolve = |inp: &FusedInput| -> Operand {
        match inp {
            FusedInput::Leaf(j) => copy(Place::local(leaf_val[*j])),
            FusedInput::LeafLane { leaf, lane } => copy(Place::local(lane_val[&(*leaf, *lane)])),
            FusedInput::Step(s) => copy(Place::local(step_reg[*s])),
            FusedInput::LitI32(c) => Operand::Const(Constant::I32(*c)),
            FusedInput::Lit(_) => {
                panic!("I32 fused kernel cannot carry an f32 literal")
            }
        }
    };
    for (s, step) in k.steps.iter().enumerate() {
        let operands: Vec<Operand> = step.inputs.iter().map(&resolve).collect();
        emit_scalar_op_i32(step_reg[s], step.op, &operands, &mut al, &mut body);
    }
    if lanes == 0 {
        body.push(Statement::Assign(
            elem(out_local, i),
            Rvalue::Use(resolve(&k.output)),
        ));
    } else {
        let store_off = store_off.expect("packed fused store allocates an offset local");
        for (lane, packed) in pack.iter().enumerate() {
            body.push(Statement::Assign(
                Place::local(store_off),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(i)), cu(lanes)),
            ));
            body.push(Statement::Assign(
                Place::local(store_off),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(store_off)), cu(lane)),
            ));
            body.push(Statement::Assign(
                elem(out_local, store_off),
                Rvalue::Use(resolve(packed)),
            ));
        }
    }

    let mut bb1_stmts = vec![Statement::Assign(
        Place::local(len),
        Rvalue::Len(Place::local(out_local)),
    )];
    if lanes != 0 {
        bb1_stmts.push(Statement::Assign(
            Place::local(len),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(len)), cu(lanes)),
        ));
    }
    bb1_stmts.push(Statement::Assign(
        Place::local(cmp),
        Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
    ));
    let bb1 = BasicBlock {
        statements: bb1_stmts,
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: body,
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Ok(pointwise_body(
        name,
        (n + 1) as u32,
        al,
        i,
        bb1,
        bb2,
        bb3,
        x_groups,
    ))
}

/// The body of a one-thread-per-element pointwise kernel from its blocks: `bb1` is the bounds check
/// (entered with the flat element index `i`), `bb2` the work, `bb3` the early exit. `x_groups: None` reads
/// the flat thread index; `Some(xg)` reads the 2-D folded grid (`i = (GroupY * xg + GroupX) * wg + LocalX`),
/// which the launch's `x_groups` (`fold_groups`) and workgroup width must match.
#[allow(clippy::too_many_arguments)]
fn pointwise_body(
    name: &str,
    param_count: u32,
    mut al: Alloc,
    i: Local,
    bb1: BasicBlock,
    bb2: BasicBlock,
    bb3: BasicBlock,
    x_groups: Option<usize>,
) -> Body {
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    match x_groups {
        None => {
            let bb0 = BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(i),
                    dim: IndexAxis::X,
                    target: BlockId { index: 1 },
                },
            };
            Body::new(name, param_count, al.locals, vec![bb0, bb1, bb2, bb3])
        }
        Some(xg) => {
            let wg = crate::elementwise::ELEMENTWISE_2D_WORKGROUP_SIZE;
            let gx = al.add(Ty::Usize, false);
            let gy = al.add(Ty::Usize, false);
            let group_id = al.add(Ty::Usize, false);
            let lane = al.add(Ty::Usize, false);
            let gy_idx: u32 = 4;
            let lane_idx: u32 = 5;
            let combine_idx: u32 = 6;
            let bb0 = BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(gx),
                    dim: IndexAxis::GroupX,
                    target: BlockId { index: gy_idx },
                },
            };
            let bb_gy = BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(gy),
                    dim: IndexAxis::GroupY,
                    target: BlockId { index: lane_idx },
                },
            };
            let bb_lane = BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(lane),
                    dim: IndexAxis::LocalX,
                    target: BlockId { index: combine_idx },
                },
            };
            let bb_combine = BasicBlock {
                statements: vec![
                    Statement::Assign(
                        Place::local(group_id),
                        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(gy)), cu(xg)),
                    ),
                    Statement::Assign(
                        Place::local(group_id),
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            copy(Place::local(group_id)),
                            copy(Place::local(gx)),
                        ),
                    ),
                    Statement::Assign(
                        Place::local(i),
                        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(group_id)), cu(wg)),
                    ),
                    Statement::Assign(
                        Place::local(i),
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            copy(Place::local(i)),
                            copy(Place::local(lane)),
                        ),
                    ),
                ],
                terminator: Terminator::Goto {
                    target: BlockId { index: 1 },
                },
            };
            Body::new(
                name,
                param_count,
                al.locals,
                vec![bb0, bb1, bb2, bb3, bb_gy, bb_lane, bb_combine],
            )
        }
    }
}

fn emit_scalar_op_i32(
    dst: Local,
    op: FusedScalarOp,
    operands: &[Operand],
    al: &mut Alloc,
    stmts: &mut Vec<Statement>,
) {
    let rv = match op {
        FusedScalarOp::Unary(UnOp::Not) => Rvalue::UnaryOp(UnOp::Not, operands[0].clone()),
        FusedScalarOp::Clz => {
            Rvalue::IntScalarUnary(IntScalarOp::LeadingZeros, operands[0].clone())
        }
        FusedScalarOp::GeU => emit_i32_cmp(
            al,
            stmts,
            BinOp::Ge,
            operands[0].clone(),
            operands[1].clone(),
            true,
        ),
        FusedScalarOp::RemU => emit_i32_rem_u(operands[0].clone(), operands[1].clone(), al, stmts),
        FusedScalarOp::Binary(b) if is_cmp(b) => emit_i32_cmp(
            al,
            stmts,
            b,
            operands[0].clone(),
            operands[1].clone(),
            false,
        ),
        FusedScalarOp::Binary(b @ (BinOp::Shl | BinOp::Shr)) => {
            emit_i32_shift(b, operands[0].clone(), operands[1].clone(), al, stmts)
        }
        FusedScalarOp::Binary(b) => Rvalue::BinaryOp(b, operands[0].clone(), operands[1].clone()),
        FusedScalarOp::Select => emit_select(
            al,
            stmts,
            operands[0].clone(),
            operands[1].clone(),
            operands[2].clone(),
        ),
        other => panic!("float fused op in the I32 synthesizer: {other:?}"),
    };
    stmts.push(Statement::Assign(Place::local(dst), rv));
}

/// The one kernel definition of `Select`: a hardware select of the branch whose condition word is nonzero, so
/// the chosen branch's bits reach the result and the other is never read as a number. `branch` is the
/// branches' type (`F32`, `I32`, `BF16` or `F16`): no arithmetic ever touches a branch and a NaN, Inf or
/// `-0.0` passes through (or stays out) bit for bit at every width. Returns the rvalue that writes the result.
fn emit_select(
    al: &mut Alloc,
    stmts: &mut Vec<Statement>,
    condition: Operand,
    if_true: Operand,
    if_false: Operand,
) -> Rvalue {
    let mut em = Emitter { al, stmts };
    let selected = em.typed(
        Ty::Bool,
        Rvalue::BinaryOp(BinOp::Ne, condition, Operand::Const(Constant::I32(0))),
    );
    Rvalue::Select {
        condition: selected,
        if_true,
        if_false,
    }
}

fn emit_i32_cmp(
    al: &mut Alloc,
    stmts: &mut Vec<Statement>,
    op: BinOp,
    mut left: Operand,
    mut right: Operand,
    unsigned: bool,
) -> Rvalue {
    if unsigned {
        let left_u = al.add(Ty::U32, false);
        stmts.push(Statement::Assign(
            Place::local(left_u),
            Rvalue::Bitcast {
                to: Ty::U32,
                operand: left,
            },
        ));
        let right_u = al.add(Ty::U32, false);
        stmts.push(Statement::Assign(
            Place::local(right_u),
            Rvalue::Bitcast {
                to: Ty::U32,
                operand: right,
            },
        ));
        left = copy(Place::local(left_u));
        right = copy(Place::local(right_u));
    }
    let flag = al.add(Ty::Bool, false);
    stmts.push(Statement::Assign(
        Place::local(flag),
        Rvalue::BinaryOp(op, left, right),
    ));
    Rvalue::Cast {
        to: Ty::I32,
        operand: copy(Place::local(flag)),
    }
}

/// Unsigned remainder over I32 storage bits: bitcast both operands to U32, `Rem`, then bitcast the
/// u32 result back to the I32 storage pattern. Codegen picks `urem` (or the NVPTX `a-(a/b)*b`
/// expansion) from the U32 operand type.
fn emit_i32_rem_u(
    value: Operand,
    divisor: Operand,
    al: &mut Alloc,
    stmts: &mut Vec<Statement>,
) -> Rvalue {
    let value_u = al.add(Ty::U32, false);
    stmts.push(Statement::Assign(
        Place::local(value_u),
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: value,
        },
    ));
    let divisor_u = al.add(Ty::U32, false);
    stmts.push(Statement::Assign(
        Place::local(divisor_u),
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: divisor,
        },
    ));
    let rem = al.add(Ty::U32, false);
    stmts.push(Statement::Assign(
        Place::local(rem),
        Rvalue::BinaryOp(
            BinOp::Rem,
            copy(Place::local(value_u)),
            copy(Place::local(divisor_u)),
        ),
    ));
    Rvalue::Bitcast {
        to: Ty::I32,
        operand: copy(Place::local(rem)),
    }
}

/// Mask the I32 shift amount with 31. `Shr` bitcasts through U32 so codegen emits `lshr` rather than
/// `ashr`; graph I32 `Shr` is logical on storage bits.
pub(crate) fn emit_i32_shift(
    op: BinOp,
    value: Operand,
    amount: Operand,
    al: &mut Alloc,
    stmts: &mut Vec<Statement>,
) -> Rvalue {
    debug_assert!(matches!(op, BinOp::Shl | BinOp::Shr));
    let masked = al.add(Ty::I32, false);
    stmts.push(Statement::Assign(
        Place::local(masked),
        Rvalue::BinaryOp(BinOp::BitAnd, amount, Operand::Const(Constant::I32(31))),
    ));
    let amount = copy(Place::local(masked));
    if op == BinOp::Shl {
        return Rvalue::BinaryOp(BinOp::Shl, value, amount);
    }
    let value_u = al.add(Ty::U32, false);
    stmts.push(Statement::Assign(
        Place::local(value_u),
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: value,
        },
    ));
    let amount_u = al.add(Ty::U32, false);
    stmts.push(Statement::Assign(
        Place::local(amount_u),
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: amount,
        },
    ));
    let shifted = al.add(Ty::U32, false);
    stmts.push(Statement::Assign(
        Place::local(shifted),
        Rvalue::BinaryOp(
            BinOp::Shr,
            copy(Place::local(value_u)),
            copy(Place::local(amount_u)),
        ),
    ));
    Rvalue::Bitcast {
        to: Ty::I32,
        operand: copy(Place::local(shifted)),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RowKind {
    Row,
    Scalar,
}

/// Compute a leaf's element offset from the linear output index `i_local`, into `off`: `off` starts at the
/// leaf's layout `base`, then for each output dim with a nonzero effective stride,
/// `off += (i / out_strides[d] % out_shape[d]) * eff[d]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_leaf_offset(
    leaf_eff: &[usize],
    base: usize,
    out_strides: &[usize],
    out_shape: &[usize],
    i_local: Local,
    off: Local,
    al: &mut Alloc,
    stmts: &mut Vec<Statement>,
) {
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    stmts.push(Statement::Assign(Place::local(off), Rvalue::Use(cu(base))));
    for (d, &e) in leaf_eff.iter().enumerate() {
        if e == 0 {
            continue;
        }
        let div = al.add(Ty::Usize, false);
        let md = al.add(Ty::Usize, false);
        let term = al.add(Ty::Usize, false);
        let off_new = al.add(Ty::Usize, false);
        stmts.push(Statement::Assign(
            Place::local(div),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i_local)), cu(out_strides[d])),
        ));
        stmts.push(Statement::Assign(
            Place::local(md),
            Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(div)), cu(out_shape[d])),
        ));
        stmts.push(Statement::Assign(
            Place::local(term),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(md)), cu(e)),
        ));
        stmts.push(Statement::Assign(
            Place::local(off_new),
            Rvalue::BinaryOp(
                BinOp::Add,
                copy(Place::local(off)),
                copy(Place::local(term)),
            ),
        ));
        stmts.push(Statement::Assign(
            Place::local(off),
            Rvalue::Use(copy(Place::local(off_new))),
        ));
    }
}

/// The static context the row-value emitter reads (everything except the mutable Alloc/stmts/memo).
struct RowCtx<'a> {
    k: &'a RowKernel,
    kind: &'a [RowKind],
    leaf_eff: &'a [Vec<usize>],
    /// per-leaf layout base offset (0 for a plain buffer).
    leaf_base: &'a [usize],
    /// per-leaf lane: how a leaf's buffer reads back as f32.
    lanes: &'a [WeightLane],
    out_strides: &'a [usize],
    out_shape: &'a [usize],
    scalar_reg: &'a [Option<Local>],
}

/// Emit the per-element value of region-local `local` at output index `i_local`, returning the register
/// holding it. Row values (leaves + Map steps) are computed inline; Scalar values (reductions + scalar
/// steps) are read from their already-computed registers. Memoized per pass so shared subexpressions
/// compute once.
fn emit_row_value(
    ctx: &RowCtx,
    local: usize,
    i_local: Local,
    al: &mut Alloc,
    stmts: &mut Vec<Statement>,
    memo: &mut std::collections::HashMap<usize, Local>,
) -> Local {
    if let Some(&l) = memo.get(&local) {
        return l;
    }
    let n = ctx.k.n_leaves;
    let res = if local < n {
        // a leaf: read it at the broadcast-mapped offset.
        let off = al.add(Ty::Usize, true);
        emit_leaf_offset(
            &ctx.leaf_eff[local],
            ctx.leaf_base[local],
            ctx.out_strides,
            ctx.out_shape,
            i_local,
            off,
            al,
            stmts,
        );
        let v = al.add(Ty::F32, false);
        stmts.extend(ctx.lanes[local].read(al, local_for_leaf(local), off, v));
        v
    } else if ctx.kind[local] == RowKind::Scalar {
        ctx.scalar_reg[local].expect("scalar value computed before use")
    } else {
        // a Map step: recurse into its operands, apply the op.
        let RowOp::Pointwise { op, inputs } = &ctx.k.steps[local - n] else {
            unreachable!("a Row-kind step must be Pointwise");
        };
        let operands: Vec<Operand> = inputs
            .iter()
            .map(|inp| match inp {
                FusedInput::Leaf(j) => copy(Place::local(emit_row_value(
                    ctx, *j, i_local, al, stmts, memo,
                ))),
                FusedInput::Step(s) => copy(Place::local(emit_row_value(
                    ctx,
                    n + *s,
                    i_local,
                    al,
                    stmts,
                    memo,
                ))),
                FusedInput::Lit(c) => Operand::Const(Constant::F32(*c)),
                FusedInput::LitI32(_) | FusedInput::LeafLane { .. } => {
                    panic!("f32 fused kernel cannot carry an I32 packed lane or literal")
                }
            })
            .collect();
        let out = al.add(Ty::F32, false);
        emit_scalar_op(out, *op, &operands, al, stmts);
        out
    };
    memo.insert(local, res);
    res
}

/// The lane each leaf of `k` is read through: its [`RowKernel::leaf_storage`] entry, which must exist for every
/// leaf and name a storage a generated body reads.
fn row_lanes(generator: &'static str, k: &RowKernel) -> Result<Vec<WeightLane>, KernelGenError> {
    if k.leaf_storage.len() != k.n_leaves {
        return Err(KernelGenError::CountMismatch {
            generator,
            what: "leaf storage count".to_string(),
            expected: k.n_leaves,
            actual: k.leaf_storage.len(),
        });
    }
    k.leaf_storage
        .iter()
        .map(|&storage| WeightLane::of(generator, storage))
        .collect()
}

/// leaf index j -> its parameter local (`_1..`).
fn local_for_leaf(j: usize) -> Local {
    local((j + 1) as u32)
}

/// Synthesize one row-wise kernel for a reduction-rooted region: one thread per row computes each last-axis
/// reduction into a register (recomputing the pointwise chain that feeds it), threads the per-row scalar chain,
/// then writes the output row (the one-pass RMSNorm / fused softmax shape). Serial over the row (like
/// `reduce_last`), so it lowers on both backends with no private array. The launch grid is the output numel;
/// threads with `row >= num_rows` no-op via the baked guard.
///
/// Only this crate's own tests call the serial form directly today; the production path is the parallel
/// twin, [`fused_row_parallel_views`].
#[cfg(test)]
fn fused_row(
    name: &str,
    out_shape: &[usize],
    leaf_shapes: &[&[usize]],
    k: &RowKernel,
) -> Result<Body, KernelGenError> {
    let layouts: Vec<Layout> = leaf_shapes.iter().map(|s| Layout::contiguous(s)).collect();
    fused_row_views(name, out_shape, leaf_shapes, &layouts, k)
}

/// [`fused_row`] reading each leaf through its own physical [`Layout`]. Byte-identical to `fused_row` when every
/// layout is `Layout::contiguous`.
#[cfg(test)]
fn fused_row_views(
    name: &str,
    out_shape: &[usize],
    leaf_shapes: &[&[usize]],
    leaf_layouts: &[Layout],
    k: &RowKernel,
) -> Result<Body, KernelGenError> {
    if leaf_shapes.len() != k.n_leaves {
        return Err(KernelGenError::CountMismatch {
            generator: "fused_row_views",
            what: "leaf shape count".to_string(),
            expected: k.n_leaves,
            actual: leaf_shapes.len(),
        });
    }
    if leaf_layouts.len() != k.n_leaves {
        return Err(KernelGenError::CountMismatch {
            generator: "fused_row_views",
            what: "leaf layout count".to_string(),
            expected: k.n_leaves,
            actual: leaf_layouts.len(),
        });
    }
    let lanes = row_lanes("fused_row_views", k)?;
    let n = k.n_leaves;
    let n_cols = k.n_cols;
    let out_numel: usize = out_shape.iter().product::<usize>().max(1);
    let num_rows = out_numel / n_cols.max(1);
    let out_strides = row_major_strides(out_shape);
    let (leaf_eff, leaf_base): (Vec<Vec<usize>>, Vec<usize>) = leaf_shapes
        .iter()
        .zip(leaf_layouts.iter())
        .map(|(s, layout)| view_eff_strides(out_shape, s, layout))
        .unzip();

    // classify each region-local as Row (per-element) or Scalar (per-row).
    let total = n + k.steps.len();
    let mut kind = vec![RowKind::Row; total];
    for (s, step) in k.steps.iter().enumerate() {
        kind[n + s] = match step {
            RowOp::Reduce { .. } => RowKind::Scalar,
            RowOp::Pointwise { inputs, .. } => {
                let any_row = inputs.iter().any(|i| match i {
                    FusedInput::Leaf(_) => true,
                    FusedInput::Step(st) => kind[n + *st] == RowKind::Row,
                    FusedInput::Lit(_) | FusedInput::LitI32(_) | FusedInput::LeafLane { .. } => {
                        false
                    }
                });
                if any_row {
                    RowKind::Row
                } else {
                    RowKind::Scalar
                }
            }
        };
    }

    // params: leaves are locals 1..=n, the output is local n+1.
    let mut locals = vec![ld(Ty::Unit, false)];
    for lane in &lanes {
        locals.push(lane.param());
    }
    locals.push(ld(slice_f32(true), true));
    let mut al = Alloc::new(locals);
    let out_local = local((n + 1) as u32);
    let row = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let j = al.add(Ty::Usize, true);
    let i_local = al.add(Ty::Usize, false);
    let base = al.add(Ty::Usize, false);
    let j_new = al.add(Ty::Usize, false);
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));

    // scalar registers: an acc local per Reduce, a result local per Scalar Pointwise. Pre-allocate.
    let mut scalar_reg: Vec<Option<Local>> = vec![None; total];
    for (s, step) in k.steps.iter().enumerate() {
        let id = n + s;
        if kind[id] == RowKind::Scalar {
            scalar_reg[id] = Some(al.add(Ty::F32, true));
        } else {
            let _ = step;
        }
    }

    // helper to read a scalar operand (already-computed register or a literal).
    let scalar_operand = |inp: &FusedInput, scalar_reg: &[Option<Local>]| -> Operand {
        match inp {
            FusedInput::Step(s) => copy(Place::local(scalar_reg[n + *s].expect("scalar ready"))),
            FusedInput::Lit(c) => Operand::Const(Constant::F32(*c)),
            FusedInput::LitI32(_) | FusedInput::LeafLane { .. } => {
                panic!("f32 fused kernel cannot carry an I32 packed lane or literal")
            }
            FusedInput::Leaf(_) => unreachable!("a scalar step cannot take a row leaf operand"),
        }
    };

    // build the passes. `blocks` starts at id 2 (bb0/bb1 are the prologue); `pending` holds straight-line
    // scalar assignments waiting to be placed at the head of the next pass.
    let mut blocks: Vec<BasicBlock> = Vec::new(); // body blocks, ids 2..
    let mut pending: Vec<Statement> = Vec::new();
    let bid = |body_idx: usize| (2 + body_idx) as u32; // body block index -> absolute block id

    // emit i = row*n_cols + j at the top of a col-loop body, into i_local.
    let emit_i = |stmts: &mut Vec<Statement>| {
        stmts.push(Statement::Assign(
            Place::local(base),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(n_cols)),
        ));
        stmts.push(Statement::Assign(
            Place::local(i_local),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(base)), copy(Place::local(j))),
        ));
    };

    let ctx_kind = kind.clone();
    let ctx_leaf_eff = leaf_eff.clone();
    let ctx_leaf_base = leaf_base.clone();

    for (s, step) in k.steps.iter().enumerate() {
        match step {
            RowOp::Pointwise { op, inputs } if kind[n + s] == RowKind::Scalar => {
                // a per-row scalar op: straight-line, queued into `pending`.
                let operands: Vec<Operand> = inputs
                    .iter()
                    .map(|i| scalar_operand(i, &scalar_reg))
                    .collect();
                let dst = scalar_reg[n + s].unwrap();
                emit_scalar_op(dst, *op, &operands, &mut al, &mut pending);
            }
            RowOp::Pointwise { .. } => { /* a Map: recomputed on demand, nothing to emit here */ }
            RowOp::Reduce { op, input } => {
                let (init, bop) = match op {
                    RowReduce::Sum => (0.0f32, BinOp::Add),
                    RowReduce::Max => (f32::NEG_INFINITY, BinOp::Max),
                };
                let acc = scalar_reg[n + s].unwrap();
                let pre = blocks.len();
                let hdr = pre + 1;
                let bodyb = pre + 2;
                let next = pre + 3;
                // pre: pending + acc=init + j=0 -> hdr
                let mut pre_stmts = std::mem::take(&mut pending);
                pre_stmts.push(Statement::Assign(
                    Place::local(acc),
                    Rvalue::Use(Operand::Const(Constant::F32(init))),
                ));
                pre_stmts.push(Statement::Assign(Place::local(j), Rvalue::Use(cu(0))));
                blocks.push(BasicBlock {
                    statements: pre_stmts,
                    terminator: Terminator::Goto {
                        target: BlockId { index: bid(hdr) },
                    },
                });
                // hdr: cmp = j < n_cols; if false -> next, else body
                blocks.push(BasicBlock {
                    statements: vec![Statement::Assign(
                        Place::local(cmp),
                        Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(j)), cu(n_cols)),
                    )],
                    terminator: guard(cmp, bid(next), bid(bodyb)),
                });
                // body: i = row*N+j; v = row_value(input); acc = acc bop v; j += 1 -> hdr
                let ctx = RowCtx {
                    k,
                    kind: &ctx_kind,
                    leaf_eff: &ctx_leaf_eff,
                    leaf_base: &ctx_leaf_base,
                    lanes: &lanes,
                    out_strides: &out_strides,
                    out_shape,
                    scalar_reg: &scalar_reg,
                };
                let mut body_stmts: Vec<Statement> = Vec::new();
                emit_i(&mut body_stmts);
                let mut memo = std::collections::HashMap::new();
                let in_local = match input {
                    FusedInput::Leaf(jx) => {
                        emit_row_value(&ctx, *jx, i_local, &mut al, &mut body_stmts, &mut memo)
                    }
                    FusedInput::Step(st) => {
                        emit_row_value(&ctx, n + *st, i_local, &mut al, &mut body_stmts, &mut memo)
                    }
                    FusedInput::Lit(_) | FusedInput::LitI32(_) | FusedInput::LeafLane { .. } => {
                        unreachable!("reduce over a literal")
                    }
                };
                let acc_new = al.add(Ty::F32, false);
                body_stmts.push(Statement::Assign(
                    Place::local(acc_new),
                    Rvalue::BinaryOp(bop, copy(Place::local(acc)), copy(Place::local(in_local))),
                ));
                body_stmts.push(Statement::Assign(
                    Place::local(acc),
                    Rvalue::Use(copy(Place::local(acc_new))),
                ));
                body_stmts.push(Statement::Assign(
                    Place::local(j_new),
                    Rvalue::BinaryOp(BinOp::Add, copy(Place::local(j)), cu(1)),
                ));
                body_stmts.push(Statement::Assign(
                    Place::local(j),
                    Rvalue::Use(copy(Place::local(j_new))),
                ));
                blocks.push(BasicBlock {
                    statements: body_stmts,
                    terminator: Terminator::Goto {
                        target: BlockId { index: bid(hdr) },
                    },
                });
            }
        }
    }

    // the output pass: pending (any trailing scalar steps) + j=0 -> hdr; hdr loops; body stores out[i].
    let pre = blocks.len();
    let hdr = pre + 1;
    let bodyb = pre + 2;
    let output_hdr_block = hdr;
    let mut pre_stmts = std::mem::take(&mut pending);
    pre_stmts.push(Statement::Assign(Place::local(j), Rvalue::Use(cu(0))));
    blocks.push(BasicBlock {
        statements: pre_stmts,
        terminator: Terminator::Goto {
            target: BlockId { index: bid(hdr) },
        },
    });
    // hdr: placeholder `next` target patched to the exit block after we know its id.
    blocks.push(BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(cmp),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(j)), cu(n_cols)),
        )],
        terminator: guard(cmp, 0, bid(bodyb)), // zero_target patched below
    });
    // body: i = row*N+j; v = row_value(output); out[i] = v; j += 1 -> hdr
    let ctx = RowCtx {
        k,
        kind: &ctx_kind,
        leaf_eff: &ctx_leaf_eff,
        leaf_base: &ctx_leaf_base,
        lanes: &lanes,
        out_strides: &out_strides,
        out_shape,
        scalar_reg: &scalar_reg,
    };
    let mut body_stmts: Vec<Statement> = Vec::new();
    emit_i(&mut body_stmts);
    let mut memo = std::collections::HashMap::new();
    let out_val = emit_row_value(&ctx, k.output, i_local, &mut al, &mut body_stmts, &mut memo);
    body_stmts.push(Statement::Assign(
        elem(out_local, i_local),
        Rvalue::Use(copy(Place::local(out_val))),
    ));
    body_stmts.push(Statement::Assign(
        Place::local(j_new),
        Rvalue::BinaryOp(BinOp::Add, copy(Place::local(j)), cu(1)),
    ));
    body_stmts.push(Statement::Assign(
        Place::local(j),
        Rvalue::Use(copy(Place::local(j_new))),
    ));
    blocks.push(BasicBlock {
        statements: body_stmts,
        terminator: Terminator::Goto {
            target: BlockId { index: bid(hdr) },
        },
    });

    // the exit block (after all body blocks) + the prologue (bb0 row=tid, bb1 num_rows guard).
    let exit_id = bid(blocks.len());
    // patch the output loop header's exit target.
    if let Terminator::SwitchInt { targets, .. } = &mut blocks[output_hdr_block].terminator {
        targets.branches[0].1 = BlockId { index: exit_id };
    }
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(row),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(cmp),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(row)), cu(num_rows)),
        )],
        terminator: guard(cmp, exit_id, bid(0)), // row >= num_rows -> exit, else first pass
    };
    let mut all = vec![bb0, bb1];
    all.append(&mut blocks);
    all.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    });
    Ok(Body::new(name, (n + 1) as u32, al.locals, all))
}

/// The parallel twin of [`fused_row`]: one workgroup of `w` lanes per output row. Each `Reduce` step is an LDS
/// reduction: lanes stride the row into a per-lane partial, the `w` partials are combined by
/// [`emit_lds_tree_blocks_at`]'s log2-tree (`bop`: `Sum` or `Max`), and every lane reads the result from `LDS[0]`
/// into the reduction's scalar register. Pointwise `Scalar` steps run identically on every lane; the
/// per-element epilogue has lanes stride the row, writing disjoint columns. Launch `num_rows * w` threads with
/// workgroup `[w,1,1]` (row = `GroupY * x_groups + GroupX`, LocalX = lane): the `num_rows` workgroups are laid
/// over a 2-D `[x_groups, y_groups]` grid so a row count above the target's X grid cap still launches (a 1-D
/// launch has `GroupY = 0`, so `x_groups` is then moot). Same `RowKernel` recipe as the serial kernel; `Max`
/// reductions stay bit-identical, `Sum` reductions reassociate (lane partials reorder the adds) so they match
/// the serial kernel to a tight tolerance, not bit-for-bit.
///
/// Reads each leaf through its own physical [`Layout`]; byte-identical to a hypothetical contiguous-only
/// form when every layout is `Layout::contiguous`.
#[allow(clippy::too_many_arguments)]
pub fn fused_row_parallel_views(
    name: &str,
    out_shape: &[usize],
    leaf_shapes: &[&[usize]],
    leaf_layouts: &[Layout],
    k: &RowKernel,
    w: usize,
    x_groups: usize,
) -> Result<Body, KernelGenError> {
    if leaf_shapes.len() != k.n_leaves {
        return Err(KernelGenError::CountMismatch {
            generator: "fused_row_parallel_views",
            what: "leaf shape count".to_string(),
            expected: k.n_leaves,
            actual: leaf_shapes.len(),
        });
    }
    if leaf_layouts.len() != k.n_leaves {
        return Err(KernelGenError::CountMismatch {
            generator: "fused_row_parallel_views",
            what: "leaf layout count".to_string(),
            expected: k.n_leaves,
            actual: leaf_layouts.len(),
        });
    }
    if w < 1 {
        return Err(KernelGenError::BelowMinimum {
            generator: "fused_row_parallel_views",
            what: "workgroup width".to_string(),
            value: w,
            min: 1,
        });
    }
    let lanes = row_lanes("fused_row_parallel_views", k)?;
    let n = k.n_leaves;
    let n_cols = k.n_cols;
    let out_numel: usize = out_shape.iter().product::<usize>().max(1);
    let num_rows = out_numel / n_cols.max(1);
    let out_strides = row_major_strides(out_shape);
    let (leaf_eff, leaf_base): (Vec<Vec<usize>>, Vec<usize>) = leaf_shapes
        .iter()
        .zip(leaf_layouts.iter())
        .map(|(s, layout)| view_eff_strides(out_shape, s, layout))
        .unzip();

    // classify each region-local as Row (per-element) or Scalar (per-row) - identical to fused_row.
    let total = n + k.steps.len();
    let mut kind = vec![RowKind::Row; total];
    for (s, step) in k.steps.iter().enumerate() {
        kind[n + s] = match step {
            RowOp::Reduce { .. } => RowKind::Scalar,
            RowOp::Pointwise { inputs, .. } => {
                let any_row = inputs.iter().any(|i| match i {
                    FusedInput::Leaf(_) => true,
                    FusedInput::Step(st) => kind[n + *st] == RowKind::Row,
                    FusedInput::Lit(_) | FusedInput::LitI32(_) | FusedInput::LeafLane { .. } => {
                        false
                    }
                });
                if any_row {
                    RowKind::Row
                } else {
                    RowKind::Scalar
                }
            }
        };
    }

    // params: leaves are locals 1..=n, the output is local n+1.
    let mut locals = vec![ld(Ty::Unit, false)];
    for lane in &lanes {
        locals.push(lane.param());
    }
    locals.push(ld(slice_f32(true), true));
    let mut al = Alloc::new(locals);
    let out_local = local((n + 1) as u32);
    let row = al.add(Ty::Usize, false);
    let gx = al.add(Ty::Usize, false);
    let gy = al.add(Ty::Usize, false);
    let lane = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let jcol = al.add(Ty::Usize, true);
    let jcol_new = al.add(Ty::Usize, false);
    let i_local = al.add(Ty::Usize, false);
    let base = al.add(Ty::Usize, false);
    let partial = al.add(Ty::F32, true);
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));

    // a scalar register per reduction / per Scalar pointwise step (every lane holds the broadcast value).
    let mut scalar_reg: Vec<Option<Local>> = vec![None; total];
    for s in 0..k.steps.len() {
        let id = n + s;
        if kind[id] == RowKind::Scalar {
            scalar_reg[id] = Some(al.add(Ty::F32, true));
        }
    }

    let scalar_operand = |inp: &FusedInput, scalar_reg: &[Option<Local>]| -> Operand {
        match inp {
            FusedInput::Step(s) => copy(Place::local(scalar_reg[n + *s].expect("scalar ready"))),
            FusedInput::Lit(c) => Operand::Const(Constant::F32(*c)),
            FusedInput::LitI32(_) | FusedInput::LeafLane { .. } => {
                panic!("f32 fused kernel cannot carry an I32 packed lane or literal")
            }
            FusedInput::Leaf(_) => unreachable!("a scalar step cannot take a row leaf operand"),
        }
    };

    // body blocks start at absolute id 4 (bb0 gx=GroupX, bb1 gy=GroupY, bb2 lane=LocalX, bb3 row + num_rows guard).
    let mut blocks: Vec<BasicBlock> = Vec::new();
    let mut pending: Vec<Statement> = Vec::new();
    let bid = |body_idx: usize| (4 + body_idx) as u32;

    // emit base = row*n_cols; i = base + jcol at the top of a strided col-loop body.
    let emit_i = |stmts: &mut Vec<Statement>| {
        stmts.push(Statement::Assign(
            Place::local(base),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(n_cols)),
        ));
        stmts.push(Statement::Assign(
            Place::local(i_local),
            Rvalue::BinaryOp(
                BinOp::Add,
                copy(Place::local(base)),
                copy(Place::local(jcol)),
            ),
        ));
    };

    let ctx_kind = kind.clone();
    let ctx_leaf_eff = leaf_eff.clone();
    let ctx_leaf_base = leaf_base.clone();

    for (s, step) in k.steps.iter().enumerate() {
        match step {
            RowOp::Pointwise { op, inputs } if kind[n + s] == RowKind::Scalar => {
                // a per-row scalar op: straight-line on every lane (reads broadcast scalars only).
                let operands: Vec<Operand> = inputs
                    .iter()
                    .map(|i| scalar_operand(i, &scalar_reg))
                    .collect();
                let dst = scalar_reg[n + s].unwrap();
                emit_scalar_op(dst, *op, &operands, &mut al, &mut pending);
            }
            RowOp::Pointwise { .. } => { /* a Map: recomputed on demand */ }
            RowOp::Reduce { op, input } => {
                let (init, bop) = match op {
                    RowReduce::Sum => (0.0f32, BinOp::Add),
                    RowReduce::Max => (f32::NEG_INFINITY, BinOp::Max),
                };
                let acc_reg = scalar_reg[n + s].unwrap();
                let pre = blocks.len();
                // pre+0 reduce-pre: pending + partial=init + jcol=lane -> hdr
                let mut pre_stmts = std::mem::take(&mut pending);
                pre_stmts.push(Statement::Assign(
                    Place::local(partial),
                    Rvalue::Use(Operand::Const(Constant::F32(init))),
                ));
                pre_stmts.push(Statement::Assign(
                    Place::local(jcol),
                    Rvalue::Use(copy(Place::local(lane))),
                ));
                blocks.push(BasicBlock {
                    statements: pre_stmts,
                    terminator: Terminator::Goto {
                        target: BlockId {
                            index: bid(pre + 1),
                        },
                    },
                });
                // pre+1 reduce-hdr: cmp = jcol < n_cols; false -> write-lds(pre+3), else body(pre+2)
                blocks.push(BasicBlock {
                    statements: vec![Statement::Assign(
                        Place::local(cmp),
                        Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(jcol)), cu(n_cols)),
                    )],
                    terminator: guard(cmp, bid(pre + 3), bid(pre + 2)),
                });
                // pre+2 reduce-body: i = row*N+jcol; partial op= row_value(input); jcol += w -> hdr
                let rctx = RowCtx {
                    k,
                    kind: &ctx_kind,
                    leaf_eff: &ctx_leaf_eff,
                    leaf_base: &ctx_leaf_base,
                    lanes: &lanes,
                    out_strides: &out_strides,
                    out_shape,
                    scalar_reg: &scalar_reg,
                };
                let mut body_stmts: Vec<Statement> = Vec::new();
                emit_i(&mut body_stmts);
                let mut memo = std::collections::HashMap::new();
                let in_local = match input {
                    FusedInput::Leaf(jx) => {
                        emit_row_value(&rctx, *jx, i_local, &mut al, &mut body_stmts, &mut memo)
                    }
                    FusedInput::Step(st) => {
                        emit_row_value(&rctx, n + *st, i_local, &mut al, &mut body_stmts, &mut memo)
                    }
                    FusedInput::Lit(_) | FusedInput::LitI32(_) | FusedInput::LeafLane { .. } => {
                        unreachable!("reduce over a literal")
                    }
                };
                let part_new = al.add(Ty::F32, false);
                body_stmts.push(Statement::Assign(
                    Place::local(part_new),
                    Rvalue::BinaryOp(
                        bop,
                        copy(Place::local(partial)),
                        copy(Place::local(in_local)),
                    ),
                ));
                body_stmts.push(Statement::Assign(
                    Place::local(partial),
                    Rvalue::Use(copy(Place::local(part_new))),
                ));
                body_stmts.push(Statement::Assign(
                    Place::local(jcol_new),
                    Rvalue::BinaryOp(BinOp::Add, copy(Place::local(jcol)), cu(w)),
                ));
                body_stmts.push(Statement::Assign(
                    Place::local(jcol),
                    Rvalue::Use(copy(Place::local(jcol_new))),
                ));
                blocks.push(BasicBlock {
                    statements: body_stmts,
                    terminator: Terminator::Goto {
                        target: BlockId {
                            index: bid(pre + 1),
                        },
                    },
                });
                // pre+3 write-lds: LDS[lane] = partial; barrier -> the tree's first round.
                let round_start = bid(pre + 4);
                blocks.push(BasicBlock {
                    statements: vec![Statement::WorkgroupLocalWrite {
                        idx: copy(Place::local(lane)),
                        value: copy(Place::local(partial)),
                        array: 0,
                    }],
                    terminator: Terminator::Barrier {
                        target: BlockId { index: round_start },
                    },
                });
                // round_start..final_branch: the log2-tree combine, `bop`-parameterized: Sum or Max (RowReduce::Max) are
                // both valid reassociations for the tree (see emit_lds_tree_blocks_at's doc).
                let final_branch = round_start + lds_tree_block_count(w);
                emit_lds_tree_blocks_at(
                    &mut al,
                    &mut blocks,
                    lane,
                    w,
                    0,
                    bop,
                    round_start,
                    final_branch,
                );
                // final_branch (read-bcast, all lanes): acc_reg = LDS[0]; barrier -> next pass. No lane-0 store guard is
                // needed: the tree already leaves LDS[0] holding the full combine for every lane.
                blocks.push(BasicBlock {
                    statements: vec![Statement::Assign(
                        Place::local(acc_reg),
                        Rvalue::WorkgroupLocalRead {
                            idx: cu(0),
                            array: 0,
                        },
                    )],
                    terminator: Terminator::Barrier {
                        target: BlockId {
                            index: final_branch + 1,
                        },
                    },
                });
            }
        }
    }

    // the output pass: pending (trailing scalar steps) + jcol=lane -> hdr; hdr loops; body writes out[i].
    let pre = blocks.len();
    let output_hdr_block = pre + 1;
    let mut pre_stmts = std::mem::take(&mut pending);
    pre_stmts.push(Statement::Assign(
        Place::local(jcol),
        Rvalue::Use(copy(Place::local(lane))),
    ));
    blocks.push(BasicBlock {
        statements: pre_stmts,
        terminator: Terminator::Goto {
            target: BlockId {
                index: bid(pre + 1),
            },
        },
    });
    // hdr: cmp = jcol < n_cols; false -> exit (patched), else body
    blocks.push(BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(cmp),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(jcol)), cu(n_cols)),
        )],
        terminator: guard(cmp, 0, bid(pre + 2)),
    });
    // body: i = row*N+jcol; out[i] = row_value(output); jcol += w -> hdr
    let rctx = RowCtx {
        k,
        kind: &ctx_kind,
        leaf_eff: &ctx_leaf_eff,
        leaf_base: &ctx_leaf_base,
        lanes: &lanes,
        out_strides: &out_strides,
        out_shape,
        scalar_reg: &scalar_reg,
    };
    let mut body_stmts: Vec<Statement> = Vec::new();
    emit_i(&mut body_stmts);
    let mut memo = std::collections::HashMap::new();
    let out_val = emit_row_value(
        &rctx,
        k.output,
        i_local,
        &mut al,
        &mut body_stmts,
        &mut memo,
    );
    body_stmts.push(Statement::Assign(
        elem(out_local, i_local),
        Rvalue::Use(copy(Place::local(out_val))),
    ));
    body_stmts.push(Statement::Assign(
        Place::local(jcol_new),
        Rvalue::BinaryOp(BinOp::Add, copy(Place::local(jcol)), cu(w)),
    ));
    body_stmts.push(Statement::Assign(
        Place::local(jcol),
        Rvalue::Use(copy(Place::local(jcol_new))),
    ));
    blocks.push(BasicBlock {
        statements: body_stmts,
        terminator: Terminator::Goto {
            target: BlockId {
                index: bid(pre + 1),
            },
        },
    });

    // exit block + prologue (bb0 gx=GroupX, bb1 gy=GroupY, bb2 lane=LocalX, bb3 row=gy*x_groups+gx and the
    // num_rows guard).
    let exit_id = bid(blocks.len());
    if let Terminator::SwitchInt { targets, .. } = &mut blocks[output_hdr_block].terminator {
        targets.branches[0].1 = BlockId { index: exit_id };
    }
    let thread_index = |destination: Local, dim: IndexAxis, target: u32| BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(destination),
            dim,
            target: BlockId { index: target },
        },
    };
    let bb0 = thread_index(gx, IndexAxis::GroupX, 1);
    let bb1 = thread_index(gy, IndexAxis::GroupY, 2);
    let bb2 = thread_index(lane, IndexAxis::LocalX, 3);
    let bb3 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(row),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(gy)), cu(x_groups)),
            ),
            Statement::Assign(
                Place::local(row),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(row)), copy(Place::local(gx))),
            ),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(row)), cu(num_rows)),
            ),
        ],
        terminator: guard(cmp, exit_id, bid(0)),
    };
    let mut all = vec![bb0, bb1, bb2, bb3];
    all.append(&mut blocks);
    all.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    });
    let mut body = Body::new(name, (n + 1) as u32, al.locals, all);
    body.workgroup_size = [w as u32, 1, 1];
    body.workgroup_locals = vec![WorkgroupLocalDecl {
        elem_ty: Ty::F32,
        len: w as u32,
    }];
    Ok(body)
}

#[cfg(test)]
mod tests {
    use poot_runtime::KernelBuffer;

    use super::*;
    use crate::test_support::{ctx, spv, stage_nvptx};

    /// Build the one-pass RMSNorm RowKernel (leaves x, w) for `n_cols` columns. Shared by the serial and
    /// parallel equivalence test.
    fn rmsnorm_kernel(n_cols: usize, eps: f32) -> RowKernel {
        RowKernel {
            n_leaves: 2, // x, w
            leaf_storage: vec![BufferStorage::f32(); 2],
            n_cols,
            steps: vec![
                RowOp::Pointwise {
                    op: FusedScalarOp::Binary(BinOp::Mul),
                    inputs: vec![FusedInput::Leaf(0), FusedInput::Leaf(0)],
                },
                RowOp::Reduce {
                    op: RowReduce::Sum,
                    input: FusedInput::Step(0),
                },
                RowOp::Pointwise {
                    op: FusedScalarOp::Binary(BinOp::Mul),
                    inputs: vec![FusedInput::Step(1), FusedInput::Lit(1.0 / n_cols as f32)],
                },
                RowOp::Pointwise {
                    op: FusedScalarOp::Binary(BinOp::Add),
                    inputs: vec![FusedInput::Step(2), FusedInput::Lit(eps)],
                },
                RowOp::Pointwise {
                    op: FusedScalarOp::Math(MathOp::Sqrt),
                    inputs: vec![FusedInput::Step(3)],
                },
                RowOp::Pointwise {
                    op: FusedScalarOp::Binary(BinOp::Div),
                    inputs: vec![FusedInput::Leaf(0), FusedInput::Step(4)],
                },
                RowOp::Pointwise {
                    op: FusedScalarOp::Binary(BinOp::Mul),
                    inputs: vec![FusedInput::Step(5), FusedInput::Leaf(1)],
                },
            ],
            output: 2 + 6,
        }
    }

    #[test]
    fn fused_row_rmsnorm_one_pass() {
        // One-pass RMSNorm as a row-wise fused kernel. Region (leaves x[1,1,N], w[N]):
        //   sq = x*x ; ss = sum(sq) ; ms = ss*(1/N) ; mse = ms+eps ; den = sqrt(mse) ;
        //   xn = x/den ; out = xn*w.
        // One reduction + a scalar chain + a recomputed epilogue, one kernel.
        let Some(ctx) = ctx() else { return };
        let n = 4usize;
        let eps = 1e-6f32;
        let k = rmsnorm_kernel(n, eps);
        let body = fused_row("rmsnorm", &[1, 1, n], &[&[1, 1, n], &[n]], &k)
            .expect("fused_row precondition");
        stage_nvptx(&body);
        let s = spv(&body, "rmsnorm");
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [2.0f32, 0.5, 1.0, 1.5];
        let mut bufs = [
            KernelBuffer::read_only_f32(&x),
            KernelBuffer::read_only_f32(&w),
            KernelBuffer::write_f32(n),
        ];
        ctx.dispatch("test", &s, [64, 1, 1], [n as u32, 1, 1], &mut bufs)
            .unwrap();
        // reference
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let den = (ss / n as f32 + eps).sqrt();
        let want: Vec<f32> = x.iter().zip(&w).map(|(a, b)| (a / den) * b).collect();
        let got = bufs[2].as_f32();
        for (g, e) in got.iter().zip(&want) {
            assert!((g - e).abs() <= 1e-5, "rmsnorm: got {g} want {e}");
        }
    }

    fn softmax_kernel(n_cols: usize) -> RowKernel {
        RowKernel {
            n_leaves: 1,
            leaf_storage: vec![BufferStorage::f32()],
            n_cols,
            steps: vec![
                RowOp::Reduce {
                    op: RowReduce::Max,
                    input: FusedInput::Leaf(0),
                },
                RowOp::Pointwise {
                    op: FusedScalarOp::Binary(BinOp::Sub),
                    inputs: vec![FusedInput::Leaf(0), FusedInput::Step(0)],
                },
                RowOp::Pointwise {
                    op: FusedScalarOp::Math(MathOp::Exp),
                    inputs: vec![FusedInput::Step(1)],
                },
                RowOp::Reduce {
                    op: RowReduce::Sum,
                    input: FusedInput::Step(2),
                },
                RowOp::Pointwise {
                    op: FusedScalarOp::Binary(BinOp::Div),
                    inputs: vec![FusedInput::Step(2), FusedInput::Step(3)],
                },
            ],
            output: 1 + 4,
        }
    }

    #[test]
    fn fused_row_softmax_two_reductions() {
        // Fused softmax = two chained reductions in one row kernel. Region (leaf scores[1,1,1,S]):
        //   m = max(scores) ; shifted = scores - m ; e = exp(shifted) ; s = sum(e) ; p = e / s.
        // The sum pass and the output pass both recompute e = exp(scores-m) from the already-reduced m.
        let Some(ctx) = ctx() else { return };
        let sz = 4usize;
        let k = softmax_kernel(sz);
        let body = fused_row("softmax", &[1, 1, 1, sz], &[&[1, 1, 1, sz]], &k)
            .expect("fused_row precondition");
        stage_nvptx(&body);
        let s = spv(&body, "softmax");
        let scores = [1.0f32, 2.0, 3.0, 4.0];
        let mut bufs = [
            KernelBuffer::read_only_f32(&scores),
            KernelBuffer::write_f32(sz),
        ];
        ctx.dispatch("test", &s, [64, 1, 1], [sz as u32, 1, 1], &mut bufs)
            .unwrap();
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = scores.iter().map(|v| (v - m).exp()).collect();
        let den: f32 = e.iter().sum();
        let want: Vec<f32> = e.iter().map(|v| v / den).collect();
        let got = bufs[1].as_f32();
        let sum: f32 = got.iter().sum();
        assert!(
            (sum - 1.0).abs() <= 1e-5,
            "softmax must sum to 1, got {sum}"
        );
        for (g, e) in got.iter().zip(&want) {
            assert!((g - e).abs() <= 1e-5, "softmax: got {g} want {e}");
        }
    }

    #[test]
    fn fused_row_rmsnorm_multi_row() {
        // num_rows > 1: 3 independent rows of length N. Validates row indexing + the num_rows guard (the
        // launch grid is 3*N threads; only the first 3 do work).
        let Some(ctx) = ctx() else { return };
        let (rows, n) = (3usize, 4usize);
        let eps = 1e-6f32;
        let k = rmsnorm_kernel(n, eps);
        let body = fused_row("rmsnorm3", &[rows, n], &[&[rows, n], &[n]], &k)
            .expect("fused_row precondition");
        let s = spv(&body, "rmsnorm3");
        let x: Vec<f32> = (0..rows * n).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let w = [1.0f32, 2.0, 0.5, 1.5];
        let mut bufs = [
            KernelBuffer::read_only_f32(&x),
            KernelBuffer::read_only_f32(&w),
            KernelBuffer::write_f32(rows * n),
        ];
        ctx.dispatch("test", &s, [64, 1, 1], [(rows * n) as u32, 1, 1], &mut bufs)
            .unwrap();
        let got = bufs[2].as_f32();
        for r in 0..rows {
            let row = &x[r * n..r * n + n];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let den = (ss / n as f32 + eps).sqrt();
            for c in 0..n {
                let want = (row[c] / den) * w[c];
                assert!(
                    (got[r * n + c] - want).abs() <= 1e-5,
                    "row {r} col {c}: got {} want {want}",
                    got[r * n + c]
                );
            }
        }
    }

    #[test]
    fn fused_row_parallel_matches_serial_rmsnorm() {
        // The workgroup-per-row parallel RMSNorm kernel matches the serial fused_row within a tight
        // tolerance (the sum reassociates over lane partials). Multi-row (GroupX) and n_cols > w (each lane
        // strides several columns).
        let Some(ctx) = ctx() else { return };
        let num_rows = 3usize;
        let n_cols = 8usize;
        let w = 4usize; // lanes < cols: each lane handles 2 columns
        let eps = 1e-6f32;
        let k = rmsnorm_kernel(n_cols, eps);
        let out_shape = [num_rows, n_cols];
        let leaves: [&[usize]; 2] = [&[num_rows, n_cols], &[n_cols]];
        let layouts = [
            Layout::contiguous(&[num_rows, n_cols]),
            Layout::contiguous(&[n_cols]),
        ];

        // deterministic inputs.
        let x: Vec<f32> = (0..num_rows * n_cols)
            .map(|i| (i as f32 * 0.37).sin() + 1.5)
            .collect();
        let wt: Vec<f32> = (0..n_cols).map(|j| 0.5 + j as f32 * 0.1).collect();

        let serial = fused_row("rms_ser", &out_shape, &leaves, &k).expect("fused_row precondition");
        let parallel = fused_row_parallel_views("rms_par", &out_shape, &leaves, &layouts, &k, w, 1)
            .expect("fused_row_parallel_views precondition");
        stage_nvptx(&parallel);
        let s_ser = spv(&serial, "rms_ser");
        let s_par = spv(&parallel, "rms_par");

        let run = |s: &poot_runtime::CompiledKernel, wg: [u32; 3], threads: [u32; 3]| -> Vec<f32> {
            let mut bufs = [
                KernelBuffer::read_only_f32(&x),
                KernelBuffer::read_only_f32(&wt),
                KernelBuffer::write_f32(num_rows * n_cols),
            ];
            ctx.dispatch("test", s, wg, threads, &mut bufs).unwrap();
            bufs[2].as_f32().to_vec()
        };
        let got_ser = run(&s_ser, [64, 1, 1], [num_rows as u32, 1, 1]);
        let got_par = run(&s_par, [w as u32, 1, 1], [(num_rows * w) as u32, 1, 1]);

        // host reference.
        let mut want = vec![0.0f32; num_rows * n_cols];
        for r in 0..num_rows {
            let ss: f32 = (0..n_cols).map(|j| x[r * n_cols + j].powi(2)).sum();
            let den = (ss / n_cols as f32 + eps).sqrt();
            for j in 0..n_cols {
                want[r * n_cols + j] = x[r * n_cols + j] / den * wt[j];
            }
        }
        for i in 0..num_rows * n_cols {
            let rel = (got_par[i] - want[i]).abs() / want[i].abs().max(1e-6);
            assert!(
                rel <= 1e-5,
                "parallel vs host at {i}: {} vs {}",
                got_par[i],
                want[i]
            );
            let rel2 = (got_par[i] - got_ser[i]).abs() / got_ser[i].abs().max(1e-6);
            assert!(
                rel2 <= 1e-5,
                "parallel vs serial at {i}: {} vs {}",
                got_par[i],
                got_ser[i]
            );
        }
    }

    #[test]
    fn fused_row_parallel_matches_serial_softmax() {
        // Fused softmax (max + sum, two reductions) parallel vs serial. Max has no reassociation; the final
        // probabilities match the serial kernel within tolerance (the sum reassociates).
        let Some(ctx) = ctx() else { return };
        let num_rows = 2usize;
        let sz = 10usize;
        let w = 4usize;
        let k = softmax_kernel(sz);
        let out_shape = [num_rows, sz];
        let leaves: [&[usize]; 1] = [&[num_rows, sz]];
        let layouts = [Layout::contiguous(&[num_rows, sz])];
        let scores: Vec<f32> = (0..num_rows * sz)
            .map(|i| (i as f32 * 0.7).cos() * 3.0)
            .collect();

        let serial = fused_row("sm_ser", &out_shape, &leaves, &k).expect("fused_row precondition");
        let parallel = fused_row_parallel_views("sm_par", &out_shape, &leaves, &layouts, &k, w, 1)
            .expect("fused_row_parallel_views precondition");
        let s_ser = spv(&serial, "sm_ser");
        let s_par = spv(&parallel, "sm_par");
        let run = |s: &poot_runtime::CompiledKernel, wg: [u32; 3], threads: [u32; 3]| -> Vec<f32> {
            let mut bufs = [
                KernelBuffer::read_only_f32(&scores),
                KernelBuffer::write_f32(num_rows * sz),
            ];
            ctx.dispatch("test", s, wg, threads, &mut bufs).unwrap();
            bufs[1].as_f32().to_vec()
        };
        let got_ser = run(&s_ser, [64, 1, 1], [num_rows as u32, 1, 1]);
        let got_par = run(&s_par, [w as u32, 1, 1], [(num_rows * w) as u32, 1, 1]);

        // each row's probabilities sum to 1 and match the serial kernel.
        for r in 0..num_rows {
            let sum: f32 = (0..sz).map(|j| got_par[r * sz + j]).sum();
            assert!((sum - 1.0).abs() <= 1e-4, "row {r} softmax sums to {sum}");
            for j in 0..sz {
                let i = r * sz + j;
                let rel = (got_par[i] - got_ser[i]).abs() / got_ser[i].abs().max(1e-6);
                assert!(
                    rel <= 1e-5,
                    "softmax parallel vs serial at {i}: {} vs {}",
                    got_par[i],
                    got_ser[i]
                );
            }
        }
    }
    /// A step reading both leaves, then `steps` chained steps of `op`, each reading `arity` operands (the
    /// first from the step before, the rest from the leaves). The leading step keeps both leaves read
    /// whatever `steps` is, so a body with none differs from one with eight by the ops alone.
    fn chain(op: FusedScalarOp, arity: usize, steps: usize, leaf: LeafTy) -> FusedKernel {
        let mut all = vec![FusedStep {
            op: FusedScalarOp::Binary(BinOp::Add),
            inputs: vec![FusedInput::Leaf(0), FusedInput::Leaf(1)],
        }];
        all.extend((0..steps).map(|s| {
            FusedStep {
                op,
                inputs: (0..arity)
                    .map(|i| {
                        if i == 0 {
                            FusedInput::Step(s)
                        } else {
                            FusedInput::Leaf(i % 2)
                        }
                    })
                    .collect(),
            }
        }));
        FusedKernel {
            leaf_tys: vec![leaf; 2],
            output: FusedInput::Step(steps),
            steps: all,
        }
    }

    /// The size bounds are what a caller refuses on before any generator runs, so they must never sit
    /// below the body a generator really builds. Every op's emitter is measured, over the f32 and the
    /// I32 synthesizers, at ranks that change the leaf-read cost: a region of eight steps against the same
    /// region with none, so the fixed and per-leaf terms cancel and each op's own cost is what is compared.
    /// Mutation: lower one op's `size()` (the Erf locals, say); the row for that op fails with the bound's
    /// growth under the built body's.
    #[test]
    fn pointwise_size_bounds_every_ops_generated_body() {
        let float_ops = [
            (FusedScalarOp::Unary(UnOp::Neg), 1),
            (FusedScalarOp::Math(MathOp::Exp), 1),
            (FusedScalarOp::Recip, 1),
            (FusedScalarOp::Binary(BinOp::Add), 2),
            (FusedScalarOp::Binary(BinOp::Ge), 2),
            (FusedScalarOp::Tanh, 1),
            (FusedScalarOp::Erf, 1),
        ];
        let i32_ops = [
            (FusedScalarOp::Unary(UnOp::Not), 1),
            (FusedScalarOp::Clz, 1),
            (FusedScalarOp::GeU, 2),
            (FusedScalarOp::RemU, 2),
            (FusedScalarOp::Select, 3),
            (FusedScalarOp::Binary(BinOp::Shl), 2),
            (FusedScalarOp::Binary(BinOp::Shr), 2),
            (FusedScalarOp::Binary(BinOp::Lt), 2),
            (FusedScalarOp::Binary(BinOp::Add), 2),
        ];
        let grew = |built: (BodySize, BodySize), bound: (BodySize, BodySize)| {
            let over = |built: (usize, usize), bound: (usize, usize)| {
                built.1 - built.0 <= bound.1 - bound.0
            };
            over(
                (built.0.instructions, built.1.instructions),
                (bound.0.instructions, bound.1.instructions),
            ) && over(
                (built.0.locals, built.1.locals),
                (bound.0.locals, bound.1.locals),
            )
        };
        for rank in [1usize, 2, 4] {
            let out_shape = vec![2usize; rank];
            let leaves: [&[usize]; 2] = [&out_shape, &out_shape];
            let layouts = [
                Layout::contiguous(&out_shape),
                Layout::contiguous(&out_shape),
            ];
            for (op, arity) in float_ops {
                let (empty, full) = (
                    chain(op, arity, 0, LeafTy::F32),
                    chain(op, arity, 8, LeafTy::F32),
                );
                let build = |k: &FusedKernel| {
                    BodySize::of(&fused_views("k", &out_shape, &leaves, &layouts, k, None).unwrap())
                };
                let (built, bound) = (
                    (build(&empty), build(&full)),
                    (
                        pointwise_size(&empty, rank, 0),
                        pointwise_size(&full, rank, 0),
                    ),
                );
                assert!(
                    grew(built, bound),
                    "f32 {op:?} rank {rank}: bound grows {bound:?}, the built body {built:?}"
                );
            }
            // A float `Select` reads its I32 condition leaf beside two float branch leaves.
            {
                let select = |steps: usize| FusedKernel {
                    leaf_tys: vec![LeafTy::I32, LeafTy::F32, LeafTy::F32],
                    steps: (0..steps)
                        .map(|s| FusedStep {
                            op: FusedScalarOp::Select,
                            inputs: vec![
                                FusedInput::Leaf(0),
                                if s == 0 {
                                    FusedInput::Leaf(1)
                                } else {
                                    FusedInput::Step(s - 1)
                                },
                                FusedInput::Leaf(2),
                            ],
                        })
                        .collect(),
                    output: if steps == 0 {
                        FusedInput::Leaf(1)
                    } else {
                        FusedInput::Step(steps - 1)
                    },
                };
                let shapes: [&[usize]; 3] = [&out_shape, &out_shape, &out_shape];
                let select_layouts = [
                    Layout::contiguous(&out_shape),
                    Layout::contiguous(&out_shape),
                    Layout::contiguous(&out_shape),
                ];
                for x_groups in [None, Some(4)] {
                    let (empty, full) = (select(0), select(8));
                    let build = |k: &FusedKernel| {
                        BodySize::of(
                            &fused_views("k", &out_shape, &shapes, &select_layouts, k, x_groups)
                                .unwrap(),
                        )
                    };
                    let (built, bound) = (
                        (build(&empty), build(&full)),
                        (
                            pointwise_size(&empty, rank, 0),
                            pointwise_size(&full, rank, 0),
                        ),
                    );
                    assert!(
                        grew(built, bound),
                        "f32 Select rank {rank} grid {x_groups:?}: bound grows {bound:?}, the built body {built:?}"
                    );
                }
            }
            for (op, arity) in i32_ops {
                for x_groups in [None, Some(4)] {
                    let (empty, full) = (
                        chain(op, arity, 0, LeafTy::I32),
                        chain(op, arity, 8, LeafTy::I32),
                    );
                    let build = |k: &FusedKernel| {
                        BodySize::of(
                            &fused_i32_views_grid("k", &out_shape, &leaves, &layouts, k, x_groups)
                                .unwrap(),
                        )
                    };
                    let (built, bound) = (
                        (build(&empty), build(&full)),
                        (
                            pointwise_size(&empty, rank, 0),
                            pointwise_size(&full, rank, 0),
                        ),
                    );
                    assert!(
                        grew(built, bound),
                        "i32 {op:?} rank {rank} grid {x_groups:?}: bound grows {bound:?}, the built body {built:?}"
                    );
                }
            }
        }
    }

    /// The packed store and the lane reads are the I32 form's own growth; the bound has to cover them.
    #[test]
    fn pointwise_size_bounds_a_packed_lane_region() {
        let out_shape = [3usize, 2];
        let leaf_shape = [3usize, 2];
        let k = FusedKernel {
            leaf_tys: vec![LeafTy::I32; 1],
            steps: vec![FusedStep {
                op: FusedScalarOp::Binary(BinOp::Add),
                inputs: vec![
                    FusedInput::LeafLane { leaf: 0, lane: 0 },
                    FusedInput::LeafLane { leaf: 0, lane: 1 },
                ],
            }],
            output: FusedInput::Step(0),
        };
        let pack = [
            FusedInput::Step(0),
            FusedInput::LeafLane { leaf: 0, lane: 1 },
        ];
        let body = fused_i32_views_grid_pack(
            "k",
            &out_shape,
            &[&leaf_shape],
            &[Layout::contiguous(&leaf_shape)],
            &k,
            None,
            &pack,
        )
        .unwrap();
        let (built, bound) = (BodySize::of(&body), pointwise_size(&k, 2, pack.len()));
        assert!(
            built.instructions <= bound.instructions && built.locals <= bound.locals,
            "packed bound {bound:?} is under the built body {built:?}"
        );
    }

    /// The row bound counts one pointwise-chain recomputation per reduction plus the output pass; the
    /// two production-shaped kernels and a widths sweep must stay under it. Mutation: drop the
    /// `reductions + 1` evaluation factor to `1`; the softmax rows (two reductions) go red.
    #[test]
    fn row_size_bounds_the_generated_row_bodies() {
        for (name, k) in [
            ("rmsnorm", rmsnorm_kernel(8, 1e-6)),
            ("softmax", softmax_kernel(8)),
        ] {
            let out_shape = [2usize, 8];
            let leaf_shapes: Vec<Vec<usize>> = (0..k.n_leaves)
                .map(|leaf| if leaf == 0 { vec![2, 8] } else { vec![8] })
                .collect();
            let leaves: Vec<&[usize]> = leaf_shapes.iter().map(Vec::as_slice).collect();
            let layouts: Vec<Layout> = leaves.iter().map(|s| Layout::contiguous(s)).collect();
            for w in [1usize, 2, 7, 64, 128, 256] {
                let body =
                    fused_row_parallel_views("k", &out_shape, &leaves, &layouts, &k, w, 1).unwrap();
                let (built, bound) = (BodySize::of(&body), row_size(&k, 2, w));
                assert!(
                    built.instructions <= bound.instructions && built.locals <= bound.locals,
                    "{name} w={w}: bound {bound:?} is under the built body {built:?}"
                );
            }
        }
    }
}
