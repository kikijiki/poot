use std::num::NonZeroUsize;

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, Local, LocalDecl, Operand, Place, ProjectionElem,
    Rvalue, Statement, SwitchTargets, Terminator, Ty,
};
use poot_quant::format::FloatFormat;
use poot_target::BufferStorage;

use crate::KernelGenError;
use crate::emit::{self, Emit};
use crate::error::{BodyResource, BodyStage};

/// The `(half, thresh)` schedule for an unrolled ceil-halving LDS tree reduction over `w` live partials:
/// round `i` combines `LDS[lane] += LDS[lane+half]` for `lane < thresh`, leaving `half` live partials for the
/// next round; `thresh = live - half`, so every lane's value is folded in exactly once even when `w` is not a
/// power of two (a plain sequential-addressing tree assumes a power-of-2 block size and drops an element
/// otherwise; with `half = ceil(live/2)` the unpaired index for odd `live` rides forward in `LDS[thresh]`).
/// Terminates when `live == 1` (`LDS[0]` holds the full sum). For a power-of-two `w` (`GEMV_WIDTH = 128`) this
/// is the standard log2 tree: strides 64,32,16,8,4,2,1.
fn lds_tree_rounds(w: usize) -> Vec<(usize, usize)> {
    let mut rounds = Vec::new();
    let mut live = w;
    while live > 1 {
        let half = live.div_ceil(2);
        let thresh = live - half;
        rounds.push((half, thresh));
        live = half;
    }
    rounds
}

/// Number of basic blocks [`emit_lds_tree_blocks`] pushes for `w` lanes (3 per round: guard, combine,
/// barrier) - callers use this to compute the index of the block that follows the tree (`round_start +
/// lds_tree_block_count(w)`).
pub(crate) fn lds_tree_block_count(w: usize) -> u32 {
    3 * lds_tree_rounds(w).len() as u32
}

/// Emits the unrolled log2-tree LDS combine shared by `gemv_lds`, `attn_scores_v_gemv_lds`, and the
/// `dequant_gemv_lds_*`/`indexed_dequant_gemv_lds_q6k` family. Precondition: the caller has already emitted
/// `WorkgroupLocalWrite(LDS[lane], partial)` + a `Barrier` targeting `round_start`. Each round is an
/// if/then-merge diamond (guard: `lane < thresh` -> combine, else straight to the barrier) followed by a
/// barrier; every lane reaches every round's barrier, so it is convergent by construction. The rounds are
/// unrolled (one set of blocks per round, the count is a Rust-side constant), so there is no back-edge and the
/// barrier-in-a-loop hazard does not apply. After the last round's barrier, control reaches `final_branch` with
/// `LDS[0]` holding the full sum; the caller emits the lane==0 store guard from there.
pub(crate) fn emit_lds_tree_blocks(
    al: &mut Alloc,
    blocks: &mut Vec<BasicBlock>,
    lane: Local,
    w: usize,
    round_start: u32,
    final_branch: u32,
) {
    emit_lds_tree_blocks_at(
        al,
        blocks,
        lane,
        w,
        0,
        BinOp::Add,
        round_start,
        final_branch,
    );
}

/// `array`/`bop`-parameterized sibling of [`emit_lds_tree_blocks`]: same log2-tree shape, but (a) combines
/// the LDS array `array` (a `WorkgroupLocalDecl`) instead of always array 0, for a kernel with more than one
/// LDS array (a reduction scratch array beside a data array, say), and (b) combines with `bop` instead of
/// always `BinOp::Add`, for a different commutative monoid (e.g. [`crate::fused::fused_row_parallel_views`]'s
/// `RowReduce::Max`, which needs `BinOp::Max`). `emit_lds_tree_blocks` is the `array == 0, bop == Add` special
/// case.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_lds_tree_blocks_at(
    al: &mut Alloc,
    blocks: &mut Vec<BasicBlock>,
    lane: Local,
    w: usize,
    array: u8,
    bop: BinOp,
    round_start: u32,
    final_branch: u32,
) {
    let rounds = lds_tree_rounds(w);
    let t_cmp = al.add(Ty::Bool, false);
    let t_idx2 = al.add(Ty::Usize, false);
    let t_cur = al.add(Ty::F32, false);
    let t_other = al.add(Ty::F32, false);
    let t_sum = al.add(Ty::F32, false);
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let goto = |bb: u32| Terminator::Goto {
        target: BlockId { index: bb },
    };
    for (i, (half, thresh)) in rounds.iter().enumerate() {
        let guard_idx = round_start + 3 * i as u32;
        let combine_idx = guard_idx + 1;
        let merge_idx = guard_idx + 2;
        let next_target = if i + 1 < rounds.len() {
            round_start + 3 * (i as u32 + 1)
        } else {
            final_branch
        };
        // guard: t_cmp = lane < thresh; false (lane >= thresh, this lane sits idle) -> merge (the barrier)
        // directly; true -> combine.
        blocks.push(BasicBlock {
            statements: vec![Statement::Assign(
                Place::local(t_cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(lane)), cu(*thresh)),
            )],
            terminator: guard(t_cmp, merge_idx, combine_idx),
        });
        // combine: LDS[lane] += LDS[lane+half] -> merge.
        blocks.push(BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(t_idx2),
                    Rvalue::BinaryOp(BinOp::Add, copy(Place::local(lane)), cu(*half)),
                ),
                Statement::Assign(
                    Place::local(t_cur),
                    Rvalue::WorkgroupLocalRead {
                        idx: copy(Place::local(lane)),
                        array,
                    },
                ),
                Statement::Assign(
                    Place::local(t_other),
                    Rvalue::WorkgroupLocalRead {
                        idx: copy(Place::local(t_idx2)),
                        array,
                    },
                ),
                Statement::Assign(
                    Place::local(t_sum),
                    Rvalue::BinaryOp(bop, copy(Place::local(t_cur)), copy(Place::local(t_other))),
                ),
                Statement::WorkgroupLocalWrite {
                    idx: copy(Place::local(lane)),
                    value: copy(Place::local(t_sum)),
                    array,
                },
            ],
            terminator: goto(merge_idx),
        });
        // merge (ALL lanes, whether they combined or not): barrier -> next round's guard, or final_branch.
        blocks.push(BasicBlock {
            statements: vec![],
            terminator: Terminator::Barrier {
                target: BlockId { index: next_target },
            },
        });
    }
}

pub(crate) fn slice_f32(mutable: bool) -> Ty {
    Ty::Ref {
        mutable,
        pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
    }
}
/// Per-thread private array element `arr[idx]` (projection `[Index]`, no Deref - distinguishing a private
/// `Ty::Array` local from a slice param `[Deref, Index]`).
pub(crate) fn arr_elem(arr: Local, idx: Local) -> Place {
    Place {
        local: arr,
        projection: vec![ProjectionElem::Index(idx)],
    }
}

pub(crate) fn ld(ty: Ty, mutable: bool) -> LocalDecl {
    LocalDecl { ty, mutable }
}
pub(crate) fn local(i: u32) -> Local {
    Local { index: i }
}
pub(crate) fn elem(slice: Local, idx: Local) -> Place {
    Place {
        local: slice,
        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(idx)],
    }
}
pub(crate) fn copy(p: Place) -> Operand {
    Operand::Copy(p)
}

/// `if cmp == 0 -> zero_target, else -> otherwise` (the 2-way bool guard shape).
pub(crate) fn guard(cmp: Local, zero_target: u32, otherwise: u32) -> Terminator {
    Terminator::SwitchInt {
        discr: copy(Place::local(cmp)),
        targets: SwitchTargets {
            branches: vec![(0, BlockId { index: zero_target })],
            otherwise: BlockId { index: otherwise },
        },
    }
}

/// The most a generated body may hold: statements plus block terminators, and locals (parameters
/// included). Caller policy, finite by construction; a request over either is a typed
/// [`KernelGenError::BodyLimit`], never a truncated or oversized body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodyLimits {
    pub max_instructions: NonZeroUsize,
    pub max_locals: NonZeroUsize,
}

/// A body's size in the units [`BodyLimits`] caps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BodySize {
    pub instructions: usize,
    pub locals: usize,
}

impl BodySize {
    /// The size of a finished body.
    pub(crate) fn of(body: &Body) -> Self {
        Self {
            instructions: body
                .blocks
                .iter()
                .map(|block| block.statements.len() + 1)
                .sum(),
            locals: body.locals.len(),
        }
    }

    /// Checked sum; an overflowing total saturates, which no finite limit admits.
    pub(crate) fn plus(self, other: Self) -> Self {
        Self {
            instructions: self.instructions.saturating_add(other.instructions),
            locals: self.locals.saturating_add(other.locals),
        }
    }

    /// `self` repeated `times` times, saturating.
    pub(crate) fn times(self, times: usize) -> Self {
        Self {
            instructions: self.instructions.saturating_mul(times),
            locals: self.locals.saturating_mul(times),
        }
    }
}

/// Reserves a body's growth against [`BodyLimits`] before the growth happens. A reservation either
/// fits both caps and is recorded, or is refused with nothing recorded, so a refusal leaves the budget
/// exactly where it was.
#[derive(Debug)]
pub struct BodyBudget {
    limits: BodyLimits,
    reserved: BodySize,
}

impl BodyBudget {
    pub fn new(limits: BodyLimits) -> Self {
        Self {
            limits,
            reserved: BodySize::default(),
        }
    }

    /// What has been reserved so far.
    pub fn reserved(&self) -> BodySize {
        self.reserved
    }

    /// Reserve `size` more at `stage`, or refuse naming the first cap it would pass.
    pub fn reserve(
        &mut self,
        generator: &'static str,
        stage: BodyStage,
        size: BodySize,
    ) -> Result<(), KernelGenError> {
        let total = self.reserved.plus(size);
        let over = |resource, attempted: usize, limit: NonZeroUsize| {
            (attempted > limit.get()).then_some(KernelGenError::BodyLimit {
                generator,
                resource,
                stage,
                attempted,
                limit: limit.get(),
            })
        };
        if let Some(error) = over(
            BodyResource::Instructions,
            total.instructions,
            self.limits.max_instructions,
        )
        .or_else(|| over(BodyResource::Locals, total.locals, self.limits.max_locals))
        {
            return Err(error);
        }
        self.reserved = total;
        Ok(())
    }
}

/// A small local allocator for generators that mint many temporaries.
pub struct Alloc {
    pub locals: Vec<LocalDecl>,
}
impl Alloc {
    pub fn new(initial: Vec<LocalDecl>) -> Self {
        Alloc { locals: initial }
    }
    pub fn add(&mut self, ty: Ty, mutable: bool) -> Local {
        let i = self.locals.len() as u32;
        self.locals.push(ld(ty, mutable));
        local(i)
    }
}

/// batch-effective strides aligned to `out_batch_rank` out batch dims, using the operand's FULL
/// row-major strides; 0 where the operand's batch dim is 1 or missing (broadcast). `op_shape` is the
/// full `[..batch, X, Y]` operand shape.
pub(crate) fn batch_eff(
    out_batch_rank: usize,
    op_shape: &[usize],
    op_strides: &[usize],
) -> Vec<usize> {
    let op_batch_rank = op_shape.len() - 2;
    let offset = out_batch_rank - op_batch_rank;
    let mut eff = vec![0usize; out_batch_rank];
    for (d, slot) in eff.iter_mut().enumerate() {
        if d < offset {
            continue;
        }
        let bd = d - offset;
        *slot = if op_shape[bd] == 1 { 0 } else { op_strides[bd] };
    }
    eff
}

pub(crate) fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// Broadcast-effective strides of `operand` against `out` (right-aligned): 0 where the operand is
/// broadcast (dim is 1 or missing), else the operand's own row-major stride for that dim.
pub(crate) fn broadcast_eff_strides(out: &[usize], operand: &[usize]) -> Vec<usize> {
    let r = out.len();
    let own = row_major_strides(operand);
    let offset = r - operand.len();
    let mut eff = vec![0usize; r];
    for (d, slot) in eff.iter_mut().enumerate() {
        if d < offset {
            continue; // operand is broadcast over this leading dim
        }
        let od = d - offset;
        *slot = if operand[od] == 1 { 0 } else { own[od] };
    }
    eff
}

/// A value's physical GPU binding as (strides, offset): either a fresh row-major buffer or a strided view of a
/// source buffer. A contiguous value (the default) has row-major strides and offset 0 (`Layout::contiguous`).
///
/// Lives here, not in `poot-graph-plan` (which owns the planner-facing `Plan::View`), because the dependency
/// runs `poot-graph-plan -> poot-kernelgen`: the kernel bodies that read a view (`binary_broadcast_dt`, `fused`,
/// `fused_row`, `fused_row_parallel`) need the type. `poot-graph-plan` re-exports it as `poot_graph_plan::Layout`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl Layout {
    /// The default row-major layout of `shape`, offset 0 - "not a view" (today's only lowering).
    pub fn contiguous(shape: &[usize]) -> Layout {
        Layout {
            strides: row_major_strides(shape),
            offset: 0,
        }
    }
}

/// Effective per-output-dim strides (+ base offset) for reading an operand of declared shape `operand_shape`,
/// physical layout `layout`, against `out_shape`. Generalizes [`broadcast_eff_strides`] to operands that may be
/// strided views (a transpose/slice/broadcast read directly by a strided-capable consumer).
///
/// Equals `(broadcast_eff_strides(out_shape, operand_shape), 0)` when `layout` is
/// `Layout::contiguous(operand_shape)`: same right-alignment for a lower-rank operand, and the same zero-stride
/// override wherever `operand_shape[d] == 1`. A broadcast dim must never be read past its single element
/// whatever `layout.strides[d]` holds; the override depends on the logical shape, so it is re-applied here.
pub fn view_eff_strides(
    out_shape: &[usize],
    operand_shape: &[usize],
    layout: &Layout,
) -> (Vec<usize>, usize) {
    debug_assert_eq!(
        operand_shape.len(),
        layout.strides.len(),
        "a layout's stride count must match its value's declared rank"
    );
    let r = out_shape.len();
    let offset = r - operand_shape.len();
    let mut eff = vec![0usize; r];
    for (d, slot) in eff.iter_mut().enumerate() {
        if d < offset {
            continue; // operand is broadcast over this leading dim
        }
        let od = d - offset;
        *slot = if operand_shape[od] == 1 {
            0
        } else {
            layout.strides[od]
        };
    }
    (eff, layout.offset)
}

/// A bf16-typed `Ref<Slice<BF16>>`, the bf16 analogue of [`slice_f32`].
pub(crate) fn slice_bf16(mutable: bool) -> Ty {
    Ty::Ref {
        mutable,
        pointee: Box::new(Ty::Slice(Box::new(Ty::BF16))),
    }
}

/// A `Ref<Slice<elem>>` for an arbitrary storage element type (spec 024: f32 or bf16). Subsumes
/// [`slice_f32`]/[`slice_bf16`].
pub(crate) fn slice_dtype(elem: Ty, mutable: bool) -> Ty {
    Ty::Ref {
        mutable,
        pointee: Box::new(Ty::Slice(Box::new(elem))),
    }
}

// Direct unit tests for `broadcast_eff_strides` and `view_eff_strides`, the stride math every fused and
// binary-broadcast kernel body reads an operand through. `broadcast_eff_strides` and `row_major_strides` are
// private, so the tests live in this file rather than under `tests/`.
/// How a contraction reads its second operand, the weight. It is part of a request's key, so a body that reads
/// the weight one way can never be served for the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightLayout {
    /// `[K, N]`, the contraction axis first: `b[k * N + n]`. What `MatMul` reads.
    Kn,
    /// `[N, K]`, the checkpoint orientation: `b[n * K + k]`. What `DenseContraction` reads, so a `transpose`
    /// of the weight never runs.
    Nk,
}

/// How a dense contraction's weight elements sit in their buffer: the generated body's side of the weight's
/// planned `poot_target::BufferStorage` (Card 1007). A request names the storage, and [`WeightLane::of`] is the
/// one place that storage becomes a parameter type and a read, so a body can never read a weight in a storage the
/// plan did not choose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WeightLane {
    /// One f32 per element ([`BufferStorage::f32`]).
    F32,
    /// Two IEEE binary16 elements per `u32` word ([`BufferStorage::f16_packed`]): element `i` in word `i / 2` at
    /// bit `(i % 2) * 16`, decoded in-register by the packed emitter's binary16 decode. The weight keeps the
    /// checkpoint's two bytes per element; no f32 copy of it exists.
    F16Packed,
    /// Two BF16 elements per `u32` word ([`BufferStorage::bf16_packed`], the Card 380 lane layout): element `i`
    /// in word `i / 2` at bit `(i % 2) * 16`, widened in-register by placing its bits in the high half of an f32
    /// (exact).
    Bf16Packed,
}

impl WeightLane {
    /// The lane that reads a weight held in `storage`, or [`KernelGenError::Unsupported`] naming `generator`
    /// when no generated body reads that storage.
    pub(crate) fn of(
        generator: &'static str,
        storage: BufferStorage,
    ) -> Result<Self, KernelGenError> {
        if storage == BufferStorage::f32() {
            Ok(Self::F32)
        } else if storage == BufferStorage::f16_packed() {
            Ok(Self::F16Packed)
        } else if storage == BufferStorage::bf16_packed() {
            Ok(Self::Bf16Packed)
        } else {
            Err(KernelGenError::Unsupported {
                request: generator,
                reason: "a dense weight is read as f32, f16-packed or bf16-packed storage only",
            })
        }
    }

    /// Logical elements per buffer word: the length of the shortest run [`Self::emit_run`] reads with whole
    /// word loads.
    pub(crate) fn per_word(self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16Packed | Self::Bf16Packed => 2,
        }
    }

    /// The weight parameter's declaration.
    pub(crate) fn param(self) -> LocalDecl {
        match self {
            Self::F32 => ld(slice_f32(false), false),
            Self::F16Packed | Self::Bf16Packed => ld(slice_dtype(Ty::U32, false), false),
        }
    }

    /// Statements that assign weight element `idx` of `weight`, as f32, to `dst`.
    pub(crate) fn read(
        self,
        al: &mut Alloc,
        weight: Local,
        idx: Local,
        dst: Local,
    ) -> Vec<Statement> {
        let mut e = Emit::new(al);
        let value = self.emit_elem(&mut e, weight, idx);
        e.assign(Place::local(dst), Rvalue::Use(emit::copy(value)));
        e.take()
    }

    /// Weight element `idx` of `weight`, as an f32 local. A packed lane loads the one word that holds the element:
    /// a half never straddles a word, so (unlike `emit_read_bits`, which also loads the following word against a
    /// guard word this buffer does not have) the read stays inside the weight's own `numel.div_ceil(2)` words.
    pub(crate) fn emit_elem(self, e: &mut Emit, weight: Local, idx: Local) -> Local {
        if self == Self::F32 {
            return e.let_(Ty::F32, Rvalue::Use(copy(elem(weight, idx))));
        }
        let word_idx = emit::bin(e, Ty::Usize, BinOp::Div, emit::copy(idx), emit::cu(2));
        let half = emit::bin(e, Ty::Usize, BinOp::Rem, emit::copy(idx), emit::cu(2));
        let shift_usize = emit::bin(e, Ty::Usize, BinOp::Mul, emit::copy(half), emit::cu(16));
        let shift = e.let_(
            Ty::U32,
            Rvalue::Cast {
                to: Ty::U32,
                operand: emit::copy(shift_usize),
            },
        );
        let word = e.let_(Ty::U32, Rvalue::Use(copy(elem(weight, word_idx))));
        let shifted = emit::bin(e, Ty::U32, BinOp::Shr, emit::copy(word), emit::copy(shift));
        let bits = emit::bin(
            e,
            Ty::U32,
            BinOp::BitAnd,
            emit::copy(shifted),
            emit::c32(0xffff),
        );
        self.decode_low_half(e, bits)
    }

    /// The `count` weight elements `first .. first + count` of `weight`, as f32 locals in order: the
    /// vector-run read. When `word_aligned` (the caller proves `first` and `count` are multiples of
    /// [`Self::per_word`]) a packed lane loads each word once and splits both of its halves, so a run of
    /// `count` elements costs `count / 2` loads and no per-element index arithmetic; otherwise every element is
    /// read on its own through [`Self::emit_elem`].
    pub(crate) fn emit_run(
        self,
        e: &mut Emit,
        weight: Local,
        first: Local,
        count: usize,
        word_aligned: bool,
    ) -> Vec<Local> {
        let per_word = self.per_word();
        if per_word == 1 || !word_aligned {
            return (0..count)
                .map(|j| {
                    let idx = emit::bin(e, Ty::Usize, BinOp::Add, emit::copy(first), emit::cu(j));
                    self.emit_elem(e, weight, idx)
                })
                .collect();
        }
        debug_assert!(count.is_multiple_of(per_word), "an aligned run of {count}");
        let first_word = emit::bin(
            e,
            Ty::Usize,
            BinOp::Div,
            emit::copy(first),
            emit::cu(per_word),
        );
        (0..count / per_word)
            .flat_map(|w| {
                let idx = emit::bin(
                    e,
                    Ty::Usize,
                    BinOp::Add,
                    emit::copy(first_word),
                    emit::cu(w),
                );
                let word = e.let_(Ty::U32, Rvalue::Use(copy(elem(weight, idx))));
                let low = emit::bin(
                    e,
                    Ty::U32,
                    BinOp::BitAnd,
                    emit::copy(word),
                    emit::c32(0xffff),
                );
                let high = emit::bin(e, Ty::U32, BinOp::Shr, emit::copy(word), emit::c32(16));
                [self.decode_low_half(e, low), self.decode_low_half(e, high)]
            })
            .collect()
    }

    /// The f32 value of the 16-bit element in the low half of `bits` (the high half zero).
    fn decode_low_half(self, e: &mut Emit, bits: Local) -> Local {
        match self {
            Self::F32 => unreachable!("an f32 lane holds no 16-bit elements"),
            Self::F16Packed => {
                crate::packed::float_decode::emit_float_decode(e, FloatFormat::F16, bits)
            }
            Self::Bf16Packed => {
                let widened = emit::bin(e, Ty::U32, BinOp::Shl, emit::copy(bits), emit::c32(16));
                e.let_(
                    Ty::F32,
                    Rvalue::Bitcast {
                        to: Ty::F32,
                        operand: emit::copy(widened),
                    },
                )
            }
        }
    }
}

#[cfg(test)]
mod stride_tests {
    use super::*;

    #[test]
    fn row_major_strides_contiguous() {
        assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
        assert_eq!(row_major_strides(&[5]), vec![1]);
        assert_eq!(row_major_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        // a size-1 leading dim still gets its row-major stride (the dim's own extent doesn't zero it -
        // only broadcast-against-a-larger-out does that, which is `broadcast_eff_strides`'s job).
        assert_eq!(row_major_strides(&[1, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn broadcast_eff_strides_same_rank_no_broadcast_is_contiguous() {
        // no broadcasting at all: effective strides equal the operand's own row-major strides.
        let out = [2usize, 3, 4];
        assert_eq!(
            broadcast_eff_strides(&out, &out),
            row_major_strides(&out),
            "same-shape operand: effective strides == own row-major strides"
        );
    }

    #[test]
    fn broadcast_eff_strides_single_axis_zeroed() {
        // a single broadcast dim (size 1 vs out's larger extent) gets stride 0; the other axes keep their
        // row-major stride.
        let out = [2usize, 3, 4];
        // operand [1,3,4]: axis 0 broadcasts (size 1 -> 2), 1 and 2 do not.
        assert_eq!(broadcast_eff_strides(&out, &[1, 3, 4]), vec![0, 4, 1]);
        // operand [2,1,4]: axis 1 broadcasts.
        assert_eq!(broadcast_eff_strides(&out, &[2, 1, 4]), vec![4, 0, 1]);
        // operand [2,3,1]: axis 2 broadcasts.
        assert_eq!(broadcast_eff_strides(&out, &[2, 3, 1]), vec![3, 1, 0]);
    }

    #[test]
    fn broadcast_eff_strides_multi_axis_zeroed() {
        // MULTIPLE broadcast axes at once (the fusion audit gap 7 case, [1,N,1] -> [M,N,K]): both the
        // leading and trailing size-1 dims must be zeroed simultaneously, not just one.
        let out = [5usize, 3, 7];
        assert_eq!(
            broadcast_eff_strides(&out, &[1, 3, 1]),
            vec![0, 1, 0],
            "both axis 0 and axis 2 broadcast: both strides must be 0"
        );
        // all three axes broadcast (a true scalar-like operand read against every out dim).
        assert_eq!(broadcast_eff_strides(&out, &[1, 1, 1]), vec![0, 0, 0]);
    }

    #[test]
    fn broadcast_eff_strides_rank_promotion() {
        // lower-rank operand: missing leading dims are implicitly size-1 and right-aligned, so they are
        // omitted from the operand's own strides but always zero in the effective strides.
        let out = [2usize, 3, 4];
        // operand [4] (rank 1): right-aligns to out's last axis; the two missing leading dims are 0.
        assert_eq!(broadcast_eff_strides(&out, &[4]), vec![0, 0, 1]);
        // operand [3,4] (rank 2): right-aligns to out's last two axes.
        assert_eq!(broadcast_eff_strides(&out, &[3, 4]), vec![0, 4, 1]);
    }

    #[test]
    fn view_eff_strides_contiguous_matches_broadcast_eff_strides() {
        // Layout::contiguous(operand_shape) must reproduce broadcast_eff_strides exactly (offset 0) - the
        // doc comment's byte-identical claim, checked directly rather than only transitively through a
        // strided-view fuzzer.
        let cases: &[(&[usize], &[usize])] = &[
            (&[2, 3, 4], &[2, 3, 4]),
            (&[2, 3, 4], &[1, 3, 4]),
            (&[2, 3, 4], &[2, 1, 4]),
            (&[5, 3, 7], &[1, 3, 1]),
            (&[2, 3, 4], &[4]),
            (&[2, 3, 4], &[3, 4]),
        ];
        for (out, operand) in cases {
            let layout = Layout::contiguous(operand);
            let (eff, offset) = view_eff_strides(out, operand, &layout);
            assert_eq!(
                offset, 0,
                "out={out:?} operand={operand:?}: contiguous offset is 0"
            );
            assert_eq!(
                eff,
                broadcast_eff_strides(out, operand),
                "out={out:?} operand={operand:?}: view_eff_strides(contiguous) must match broadcast_eff_strides"
            );
        }
    }

    #[test]
    fn view_eff_strides_transposed_layout_is_not_row_major() {
        // a transposed physical layout (not row-major): operand shape [2,3], but the underlying
        // buffer is column-major (strides [1,2] instead of [3,1]) plus a nonzero base offset. The broadcast
        // override (zero wherever the LOGICAL shape has a size-1 dim) still applies on top of the
        // transposed strides.
        let operand_shape = [2usize, 3];
        let layout = Layout {
            strides: vec![1, 2], // column-major: element (r,c) at r*1 + c*2
            offset: 7,
        };
        let (eff, offset) = view_eff_strides(&operand_shape, &operand_shape, &layout);
        assert_eq!(
            eff,
            vec![1, 2],
            "no broadcasting: strides pass through unchanged"
        );
        assert_eq!(offset, 7, "the layout's base offset is threaded through");

        // now broadcast that transposed operand's ROW to a larger out (out=[5,3], operand=[1,3]): axis 0
        // must be zeroed even though the physical layout's stride there (1) is nonzero - the broadcast
        // override depends on the LOGICAL shape, not the physical stride.
        let layout_row = Layout {
            strides: vec![1, 2],
            offset: 7,
        };
        let (eff2, offset2) = view_eff_strides(&[5, 3], &[1, 3], &layout_row);
        assert_eq!(
            eff2,
            vec![0, 2],
            "broadcast axis 0 zeroed even though the transposed layout's stride there is nonzero"
        );
        assert_eq!(offset2, 7);
    }
}
