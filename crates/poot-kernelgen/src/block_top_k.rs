//! The block top-k generator (POOT-1143): per query, the best `k` of the blocks its position has completed,
//! by an exact radix select over the order-preserving key of each block's score.
//!
//! One workgroup of [`LANES`] lanes per query; the query's `blocks` scores stay in global memory and every
//! pass re-reads them (the row is a few hundred KiB at 128K positions, resident in cache). The steps:
//!
//! 1. The eligible blocks are `0..E`, `E = min(blocks, (pos + 1) / ratio)`, and `want = min(k, E)` of them
//!    are kept. A score's key is its canonical value (`canonical_router_scores` of the graph IR: NaN and
//!    `-inf` are `f32::MIN`, `+inf` is `f32::MAX`; `-0.0` is `+0.0`, as the comparisons treat it) as an
//!    order-preserving `u32`.
//! 2. Seven radix passes of five bits (the top two, then six of five), or eight of four when a lane may own
//!    more than 255 blocks, find the key `T` of the `want`-th largest score: each pass counts the next digit
//!    of the keys that match the digits found so far, every lane into a row of its own in workgroup memory
//!    (no atomics: LLVM's SPIR-V backend miscompiles a workgroup atomic's `syncscope` in a body this
//!    size), totals the rows, and names the digit where the running count from the top reaches the
//!    remaining need. A lane's counts are packed, four 8-bit or two 16-bit to a word, which is what lets a
//!    pass read five or four bits where unpacked 32-bit counts fit three in 16 KiB. A lane's count of one
//!    digit never exceeds the blocks it owns, so the width is chosen from the block count and cannot
//!    overflow.
//! 3. Every block with a key above `T` is kept, and so are the first `need` blocks equal to `T`, lowest
//!    block first. Each lane owns a contiguous chunk of the blocks, counts what it keeps, and two block-wide
//!    prefix sums give each lane its first output slot and its share of the ties, so the ids come out in
//!    ascending order with no sort.
//!
//! Parameters: the scores `[queries, blocks]` F32, the positions `[queries]` I32, then the I32 result
//! `[queries, k + blocks]`: the ids (`-1` after the last kept) then the `0`/`1` keep indicator of each block
//! (see `poot_graph_ir::BlockTopKLayout`).

use poot_kernel_ir::{
    BinOp, Body, Constant, IndexAxis, Operand, Place, Rvalue, Ty, WorkgroupLocalDecl,
};

use crate::KernelGenError;
use crate::attention::{count_to, local_op, ubin, uc};
use crate::data_movement::Cfg;
use crate::emit::{c32, cu};
use crate::helpers::{copy, elem, ld, slice_dtype, slice_f32};

/// Lanes of the workgroup.
pub const LANES: usize = 256;

const GENERATOR: &str = "block_top_k_body";

/// What the planner asks the block top-k generator for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockTopKSpec {
    /// Queries: `rows * tokens`, one workgroup each.
    pub queries: usize,
    /// Blocks scored per query.
    pub blocks: usize,
    /// Positions each block covers.
    pub ratio: usize,
    /// Blocks kept per query.
    pub k: usize,
}

impl BlockTopKSpec {
    /// Static workgroup memory in bytes: every lane's digit counts, the group partials, the two scan buffers
    /// and the broadcast cells.
    pub fn lds_bytes(&self) -> u64 {
        ((COUNTS_LEN + 3 * LANES + CELLS) * 4) as u64
    }
}

/// How a lane's digit counts are packed, chosen from the block count: a lane owns at most
/// `ceil(blocks / LANES)` blocks, so no count of one digit exceeds that, and the narrowest width that holds it
/// gives the most bins per pass in the same workgroup memory.
#[derive(Clone, Copy)]
struct CountWidth {
    /// Bits of one packed count: 8 or 16.
    count_bits: u32,
    /// Bits of the key one radix pass reads; `1 << digit_bits` bins fill [`WORDS_PER_ROW`] words.
    digit_bits: u32,
}

/// Words of a lane's row of packed counts: 32 eight-bit counts or 16 sixteen-bit ones. Eight words per lane
/// keep the rows at 8 KiB, which with the partials, scan buffers and cells fits the 16 KiB a Vulkan device
/// guarantees.
const WORDS_PER_ROW: usize = 8;

impl CountWidth {
    const EIGHT: Self = Self {
        count_bits: 8,
        digit_bits: 5,
    };
    const SIXTEEN: Self = Self {
        count_bits: 16,
        digit_bits: 4,
    };

    /// The width that holds a lane's count of any digit over `blocks` scores, if there is one.
    fn for_blocks(blocks: usize) -> Option<Self> {
        match blocks.div_ceil(LANES) {
            owned if owned <= u8::MAX as usize => Some(Self::EIGHT),
            owned if owned <= u16::MAX as usize => Some(Self::SIXTEEN),
            _ => None,
        }
    }

    fn bins(self) -> usize {
        1 << self.digit_bits
    }

    /// Packed counts to a word.
    fn per_word(self) -> usize {
        (32 / self.count_bits) as usize
    }

    /// Radix passes to read the 32-bit key.
    fn passes(self) -> u32 {
        32u32.div_ceil(self.digit_bits)
    }

    /// `(shift, bits)` of radix pass `pass`: the digit is `(key >> shift) & ((1 << bits) - 1)`. The first pass
    /// takes the bits left over when the digit width does not divide 32.
    fn digit_of_pass(self, pass: u32) -> (u32, u32) {
        let first = match 32 % self.digit_bits {
            0 => self.digit_bits,
            leftover => leftover,
        };
        if pass == 0 {
            (32 - first, first)
        } else {
            (32 - first - self.digit_bits * pass, self.digit_bits)
        }
    }
}

/// Workgroup-local arrays, all `u32`: each lane's count of every digit, the group partials of a digit, the
/// two scan buffers and the cells that broadcast a pass's result.
const COUNTS: u8 = 0;
const PARTIAL: u8 = 1;
const SCAN_A: u8 = 2;
const SCAN_B: u8 = 3;
const CELL: u8 = 4;
const COUNTS_LEN: usize = LANES * WORDS_PER_ROW;
const CELLS: usize = 2;
/// The cell the winning digit and the remaining need are broadcast through.
const CELL_DIGIT: usize = 0;
const CELL_NEED: usize = 1;

fn validate(spec: &BlockTopKSpec) -> Result<(), KernelGenError> {
    for (what, value) in [
        ("queries", spec.queries),
        ("blocks", spec.blocks),
        ("ratio", spec.ratio),
        ("k", spec.k),
    ] {
        if value == 0 {
            return Err(KernelGenError::BelowMinimum {
                generator: GENERATOR,
                what: what.to_string(),
                value,
                min: 1,
            });
        }
    }
    if spec.k > spec.blocks {
        return Err(KernelGenError::ExceedsBound {
            generator: GENERATOR,
            what: "k".to_string(),
            value: spec.k,
            bound: spec.blocks,
        });
    }
    if CountWidth::for_blocks(spec.blocks).is_none() {
        return Err(KernelGenError::ExceedsBound {
            generator: GENERATOR,
            what: "blocks".to_string(),
            value: spec.blocks,
            bound: u16::MAX as usize * LANES,
        });
    }
    Ok(())
}

fn u32_array(len: usize) -> WorkgroupLocalDecl {
    WorkgroupLocalDecl {
        elem_ty: Ty::U32,
        len: len as u32,
    }
}

fn i32c(value: i32) -> Operand {
    Operand::Const(Constant::I32(value))
}

fn lds_read_u32(c: &mut Cfg, array: u8, idx: poot_kernel_ir::Local) -> poot_kernel_ir::Local {
    c.emit(|e| e.read_lds(Ty::U32, array, local_op(idx)))
}

fn lds_write_u32(c: &mut Cfg, array: u8, idx: poot_kernel_ir::Local, value: Operand) {
    c.emit(|e| e.write_lds(array, local_op(idx), value));
}

fn ubin32(c: &mut Cfg, op: BinOp, a: Operand, b: Operand) -> poot_kernel_ir::Local {
    c.bin(Ty::U32, op, a, b)
}

/// `min(a, b)` of two usize locals as a compare and a select: the emitter's integer `Min` intrinsic is the
/// 32-bit one, so a 64-bit index takes the select.
fn min_usize(
    c: &mut Cfg,
    a: poot_kernel_ir::Local,
    b: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    let below = c.bin(Ty::Bool, BinOp::Lt, local_op(a), local_op(b));
    select(c, Ty::Usize, below, local_op(a), local_op(b))
}

pub(crate) fn select(
    c: &mut Cfg,
    ty: Ty,
    condition: poot_kernel_ir::Local,
    if_true: Operand,
    if_false: Operand,
) -> poot_kernel_ir::Local {
    c.let_(
        ty,
        Rvalue::Select {
            condition: local_op(condition),
            if_true,
            if_false,
        },
    )
}

/// The order-preserving key of `scores[at]`: its canonical value's bits, flipped so that unsigned order is
/// float order.
fn key_of(
    c: &mut Cfg,
    scores: poot_kernel_ir::Local,
    at: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    const MIN_BITS: u32 = 0xFF7F_FFFF;
    const MAX_BITS: u32 = 0x7F7F_FFFF;
    const INF_BITS: u32 = 0x7F80_0000;
    let value = c.let_(Ty::F32, Rvalue::Use(copy(elem(scores, at))));
    let bits = c.let_(
        Ty::U32,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: local_op(value),
        },
    );
    let magnitude = ubin32(c, BinOp::BitAnd, local_op(bits), c32(0x7FFF_FFFF));
    let sign = ubin32(c, BinOp::Shr, local_op(bits), c32(31));
    let negative = c.bin(Ty::Bool, BinOp::Ne, local_op(sign), c32(0));
    let nan = c.bin(Ty::Bool, BinOp::Gt, local_op(magnitude), c32(INF_BITS));
    let infinite = c.bin(Ty::Bool, BinOp::Eq, local_op(magnitude), c32(INF_BITS));
    let zero = c.bin(Ty::Bool, BinOp::Eq, local_op(magnitude), c32(0));
    let inf_bits = select(c, Ty::U32, negative, c32(MIN_BITS), c32(MAX_BITS));
    let bits = select(c, Ty::U32, nan, c32(MIN_BITS), local_op(bits));
    let bits = select(c, Ty::U32, infinite, local_op(inf_bits), local_op(bits));
    let bits = select(c, Ty::U32, zero, c32(0), local_op(bits));
    // Negative floats sort descending by magnitude: flip every bit; positive ones flip the sign bit.
    let sign = ubin32(c, BinOp::Shr, local_op(bits), c32(31));
    let negative = c.bin(Ty::Bool, BinOp::Ne, local_op(sign), c32(0));
    let flip = select(c, Ty::U32, negative, c32(u32::MAX), c32(0x8000_0000));
    ubin32(c, BinOp::BitXor, local_op(bits), local_op(flip))
}

/// The inclusive prefix sum across the workgroup of every lane's `value`, by `log2(LANES)` doubling steps
/// over two buffers. Every lane reaches every barrier. The result is read back from the lane's own slot, so
/// the next use of the buffers may rewrite it without a barrier.
fn block_scan(
    c: &mut Cfg,
    lane: poot_kernel_ir::Local,
    value: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    lds_write_u32(c, SCAN_A, lane, local_op(value));
    c.barrier();
    let (mut from, mut to) = (SCAN_A, SCAN_B);
    let mut distance = 1;
    while distance < LANES {
        let own = lds_read_u32(c, from, lane);
        // `lane - distance` where it exists; elsewhere an unrelated slot whose value is masked off.
        let wrapped = uc(c, BinOp::Add, lane, LANES - distance);
        let behind = uc(c, BinOp::BitAnd, wrapped, LANES - 1);
        let other = lds_read_u32(c, from, behind);
        let has_left = c.bin(Ty::Bool, BinOp::Ge, local_op(lane), cu(distance));
        let left = select(c, Ty::U32, has_left, local_op(other), c32(0));
        let sum = ubin32(c, BinOp::Add, local_op(own), local_op(left));
        lds_write_u32(c, to, lane, local_op(sum));
        c.barrier();
        std::mem::swap(&mut from, &mut to);
        distance *= 2;
    }
    lds_read_u32(c, from, lane)
}

/// Build the block top-k kernel `spec` describes, launched as `spec.queries` workgroups of [`LANES`] lanes
/// over a 2-D grid `x_groups` workgroups across: `gi = GroupY * x_groups + GroupX` is the query, and a
/// workgroup past the last returns.
pub(crate) fn block_top_k_body(
    name: &str,
    spec: &BlockTopKSpec,
    x_groups: usize,
) -> Result<Body, KernelGenError> {
    validate(spec)?;
    let BlockTopKSpec {
        queries,
        blocks,
        ratio,
        k,
    } = *spec;
    let params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        ld(slice_dtype(Ty::I32, false), false),
        ld(slice_dtype(Ty::I32, true), true),
    ];
    let (scores, pos, out) = (
        poot_kernel_ir::Local { index: 1 },
        poot_kernel_ir::Local { index: 2 },
        poot_kernel_ir::Local { index: 3 },
    );
    let mut c = Cfg::new(params);
    let group_x = c.index(IndexAxis::GroupX);
    let group_y = c.index(IndexAxis::GroupY);
    let lane = c.index(IndexAxis::LocalX);
    let row_offset = uc(&mut c, BinOp::Mul, group_y, x_groups);
    let query = ubin(&mut c, BinOp::Add, row_offset, group_x);
    let in_range = c.bin(Ty::Bool, BinOp::Lt, local_op(query), cu(queries));
    c.if_then(in_range, |c| {
        let score_base = uc(c, BinOp::Mul, query, blocks);
        let out_base = uc(c, BinOp::Mul, query, k + blocks);
        let score_at =
            |c: &mut Cfg, block: poot_kernel_ir::Local| ubin(c, BinOp::Add, score_base, block);

        // The eligible blocks and how many are kept.
        let position = c.let_(Ty::I32, Rvalue::Use(copy(elem(pos, query))));
        let next = c.bin(Ty::I32, BinOp::Add, local_op(position), i32c(1));
        let completed = c.bin(Ty::I32, BinOp::Max, local_op(next), i32c(0));
        let completed = c.let_(
            Ty::Usize,
            Rvalue::Cast {
                to: Ty::Usize,
                operand: local_op(completed),
            },
        );
        let blocks_done = uc(c, BinOp::Div, completed, ratio);
        let all_blocks = c.usize_const(blocks);
        let eligible = min_usize(c, blocks_done, all_blocks);
        let kept_blocks = c.usize_const(k);
        let want = min_usize(c, eligible, kept_blocks);
        let want32 = c.let_(
            Ty::U32,
            Rvalue::Cast {
                to: Ty::U32,
                operand: local_op(want),
            },
        );

        // The radix select: after the last pass, `prefix` is the key of the `want`-th largest score and
        // `need` the number of blocks equal to it that are kept.
        let prefix = c.var(Ty::U32, Rvalue::Use(c32(0)));
        let need = c.var(Ty::U32, Rvalue::Use(local_op(want32)));
        let width = CountWidth::for_blocks(blocks).expect("validate bounds the block count");
        let bins = width.bins();
        let per_word = width.per_word();
        let per_word_log = per_word.trailing_zeros();
        let count_mask = (1u32 << width.count_bits) - 1;
        let row = uc(c, BinOp::Mul, lane, WORDS_PER_ROW);
        let digit_in_group = uc(c, BinOp::BitAnd, lane, bins - 1);
        let group_row = {
            let group = uc(c, BinOp::Div, lane, bins);
            uc(c, BinOp::Mul, group, bins)
        };
        // Where digit `d` is counted: word `d / per_word` of a row, at bit `(d % per_word) * count_bits`.
        let word_and_shift = |c: &mut Cfg, digit: poot_kernel_ir::Local| {
            let word = ubin32(c, BinOp::Shr, local_op(digit), c32(per_word_log));
            let word = c.let_(
                Ty::Usize,
                Rvalue::Cast {
                    to: Ty::Usize,
                    operand: local_op(word),
                },
            );
            let within = ubin32(c, BinOp::BitAnd, local_op(digit), c32(per_word as u32 - 1));
            let shift = ubin32(c, BinOp::Mul, local_op(within), c32(width.count_bits));
            (word, shift)
        };
        let reduced_digit = c.let_(
            Ty::U32,
            Rvalue::Cast {
                to: Ty::U32,
                operand: local_op(digit_in_group),
            },
        );
        let (reduced_word, reduced_shift) = word_and_shift(c, reduced_digit);
        for pass in 0..width.passes() {
            let (shift, bits) = width.digit_of_pass(pass);
            for word in 0..WORDS_PER_ROW {
                let at = uc(c, BinOp::Add, row, word);
                lds_write_u32(c, COUNTS, at, c32(0));
            }
            // This lane's elements, counted into its own row: no other lane touches it, so no atomic.
            count_to(c, local_op(lane), local_op(eligible), LANES, |c, block| {
                let at = score_at(c, block);
                let key = key_of(c, scores, at);
                let count = |c: &mut Cfg| {
                    let moved = ubin32(c, BinOp::Shr, local_op(key), c32(shift));
                    let digit = ubin32(c, BinOp::BitAnd, local_op(moved), c32((1 << bits) - 1));
                    let (word, bit) = word_and_shift(c, digit);
                    let slot = ubin(c, BinOp::Add, row, word);
                    let seen = lds_read_u32(c, COUNTS, slot);
                    // The width holds any count a lane can reach, so the add never carries into a neighbour.
                    let unit = ubin32(c, BinOp::Shl, c32(1), local_op(bit));
                    let more = ubin32(c, BinOp::Add, local_op(seen), local_op(unit));
                    lds_write_u32(c, COUNTS, slot, local_op(more));
                };
                if pass == 0 {
                    count(c);
                } else {
                    let high = ubin32(c, BinOp::Shr, local_op(key), c32(shift + bits));
                    let matches = c.bin(Ty::Bool, BinOp::Eq, local_op(high), local_op(prefix));
                    c.if_then(matches, count);
                }
            });
            c.barrier();
            // Lane `l` totals digit `l % bins` over the `bins` lanes of group `l / bins`.
            let partial = c.var(Ty::U32, Rvalue::Use(c32(0)));
            for member in 0..bins {
                let lane_of_row = uc(c, BinOp::Add, group_row, member);
                let lane_row = uc(c, BinOp::Mul, lane_of_row, WORDS_PER_ROW);
                let at = ubin(c, BinOp::Add, lane_row, reduced_word);
                let packed = lds_read_u32(c, COUNTS, at);
                let moved = ubin32(c, BinOp::Shr, local_op(packed), local_op(reduced_shift));
                let part = ubin32(c, BinOp::BitAnd, local_op(moved), c32(count_mask));
                let sum = ubin32(c, BinOp::Add, local_op(partial), local_op(part));
                c.assign(Place::local(partial), Rvalue::Use(local_op(sum)));
            }
            lds_write_u32(c, PARTIAL, lane, local_op(partial));
            c.barrier();
            // Lane 0 walks the digits from the top, adding each digit's total (the sum of its column of
            // group partials) until the running count reaches the need: that digit holds the `need`-th
            // largest key, and what is left of the need is counted within it.
            let first = c.bin(Ty::Bool, BinOp::Eq, local_op(lane), cu(0));
            c.if_then(first, |c| {
                let above = c.var(Ty::U32, Rvalue::Use(c32(0)));
                let found = c.var(Ty::Bool, Rvalue::Use(Operand::Const(Constant::Bool(false))));
                let digit = c.var(Ty::U32, Rvalue::Use(c32(0)));
                let rest = c.var(Ty::U32, Rvalue::Use(local_op(need)));
                for bin in (0..1usize << bits).rev() {
                    let total = c.var(Ty::U32, Rvalue::Use(c32(0)));
                    for group in 0..LANES / bins {
                        let at = c.usize_const(group * bins + bin);
                        let part = lds_read_u32(c, PARTIAL, at);
                        let sum = ubin32(c, BinOp::Add, local_op(total), local_op(part));
                        c.assign(Place::local(total), Rvalue::Use(local_op(sum)));
                    }
                    let through = ubin32(c, BinOp::Add, local_op(above), local_op(total));
                    let reaches = c.bin(Ty::Bool, BinOp::Ge, local_op(through), local_op(need));
                    let unfound = c.let_(
                        Ty::Bool,
                        Rvalue::UnaryOp(poot_kernel_ir::UnOp::Not, local_op(found)),
                    );
                    let here = c.both(reaches, unfound);
                    let left = ubin32(c, BinOp::Sub, local_op(need), local_op(above));
                    let this_digit = c32(bin as u32);
                    let next_digit = select(c, Ty::U32, here, this_digit, local_op(digit));
                    let next_rest = select(c, Ty::U32, here, local_op(left), local_op(rest));
                    c.assign(Place::local(digit), Rvalue::Use(local_op(next_digit)));
                    c.assign(Place::local(rest), Rvalue::Use(local_op(next_rest)));
                    let next_found = c.let_(
                        Ty::Bool,
                        Rvalue::Select {
                            condition: local_op(here),
                            if_true: Operand::Const(Constant::Bool(true)),
                            if_false: local_op(found),
                        },
                    );
                    c.assign(Place::local(found), Rvalue::Use(local_op(next_found)));
                    c.assign(Place::local(above), Rvalue::Use(local_op(through)));
                }
                let digit_cell = c.usize_const(CELL_DIGIT);
                lds_write_u32(c, CELL, digit_cell, local_op(digit));
                let need_cell = c.usize_const(CELL_NEED);
                lds_write_u32(c, CELL, need_cell, local_op(rest));
            });
            c.barrier();
            let digit = {
                let at = c.usize_const(CELL_DIGIT);
                lds_read_u32(c, CELL, at)
            };
            let rest = {
                let at = c.usize_const(CELL_NEED);
                lds_read_u32(c, CELL, at)
            };
            let widened = ubin32(c, BinOp::Shl, local_op(prefix), c32(bits));
            let extended = ubin32(c, BinOp::BitOr, local_op(widened), local_op(digit));
            c.assign(Place::local(prefix), Rvalue::Use(local_op(extended)));
            c.assign(Place::local(need), Rvalue::Use(local_op(rest)));
            // The next pass rewrites its rows and the cells only after its own barriers.
            c.barrier();
        }

        // Selection: each lane keeps, in its contiguous chunk, the blocks above `prefix` and its share of
        // the ties (the first `need` equal ones in block order, assigned by an exclusive scan of the tie counts).
        let span = {
            let padded = uc(c, BinOp::Add, eligible, LANES - 1);
            uc(c, BinOp::Div, padded, LANES)
        };
        let chunk_start = ubin(c, BinOp::Mul, lane, span);
        let chunk_end = {
            let end = ubin(c, BinOp::Add, chunk_start, span);
            min_usize(c, end, eligible)
        };
        let above_count = c.var(Ty::U32, Rvalue::Use(c32(0)));
        let tied_count = c.var(Ty::U32, Rvalue::Use(c32(0)));
        count_to(
            c,
            local_op(chunk_start),
            local_op(chunk_end),
            1,
            |c, block| {
                let at = score_at(c, block);
                let key = key_of(c, scores, at);
                let above = c.bin(Ty::Bool, BinOp::Gt, local_op(key), local_op(prefix));
                let tied = c.bin(Ty::Bool, BinOp::Eq, local_op(key), local_op(prefix));
                for (flag, total) in [(above, above_count), (tied, tied_count)] {
                    let step = c.let_(
                        Ty::U32,
                        Rvalue::Cast {
                            to: Ty::U32,
                            operand: local_op(flag),
                        },
                    );
                    let next = ubin32(c, BinOp::Add, local_op(total), local_op(step));
                    c.assign(Place::local(total), Rvalue::Use(local_op(next)));
                }
            },
        );
        let tied_through = block_scan(c, lane, tied_count);
        let tied_before = ubin32(c, BinOp::Sub, local_op(tied_through), local_op(tied_count));
        // The ties left for this lane: the need minus the ties earlier lanes hold, at most its own.
        let bigger = ubin32(c, BinOp::Max, local_op(need), local_op(tied_before));
        let left = ubin32(c, BinOp::Sub, local_op(bigger), local_op(tied_before));
        let quota = ubin32(c, BinOp::Min, local_op(left), local_op(tied_count));
        let keeps = ubin32(c, BinOp::Add, local_op(above_count), local_op(quota));
        let kept_through = block_scan(c, lane, keeps);
        let first_slot = ubin32(c, BinOp::Sub, local_op(kept_through), local_op(keeps));
        let slot = c.let_(
            Ty::Usize,
            Rvalue::Cast {
                to: Ty::Usize,
                operand: local_op(first_slot),
            },
        );
        let slot = c.var(Ty::Usize, Rvalue::Use(local_op(slot)));
        let taken_ties = c.var(Ty::U32, Rvalue::Use(c32(0)));
        count_to(
            c,
            local_op(chunk_start),
            local_op(chunk_end),
            1,
            |c, block| {
                let at = score_at(c, block);
                let key = key_of(c, scores, at);
                let above = c.bin(Ty::Bool, BinOp::Gt, local_op(key), local_op(prefix));
                let tied = c.bin(Ty::Bool, BinOp::Eq, local_op(key), local_op(prefix));
                let room = c.bin(Ty::Bool, BinOp::Lt, local_op(taken_ties), local_op(quota));
                let tie_taken = c.both(tied, room);
                let tie_step = c.let_(
                    Ty::U32,
                    Rvalue::Cast {
                        to: Ty::U32,
                        operand: local_op(tie_taken),
                    },
                );
                let next = ubin32(c, BinOp::Add, local_op(taken_ties), local_op(tie_step));
                c.assign(Place::local(taken_ties), Rvalue::Use(local_op(next)));
                let above_step = c.let_(
                    Ty::U32,
                    Rvalue::Cast {
                        to: Ty::U32,
                        operand: local_op(above),
                    },
                );
                let kept32 = ubin32(c, BinOp::BitOr, local_op(above_step), local_op(tie_step));
                let kept = c.bin(Ty::Bool, BinOp::Ne, local_op(kept32), c32(0));
                let flag = c.let_(
                    Ty::I32,
                    Rvalue::Cast {
                        to: Ty::I32,
                        operand: local_op(kept32),
                    },
                );
                let flag_at = {
                    let base = uc(c, BinOp::Add, block, k);
                    ubin(c, BinOp::Add, out_base, base)
                };
                c.assign(elem(out, flag_at), Rvalue::Use(local_op(flag)));
                c.if_then(kept, |c| {
                    let id_at = ubin(c, BinOp::Add, out_base, slot);
                    let id = c.let_(
                        Ty::I32,
                        Rvalue::Cast {
                            to: Ty::I32,
                            operand: local_op(block),
                        },
                    );
                    c.assign(elem(out, id_at), Rvalue::Use(local_op(id)));
                    bump(c, slot);
                });
            },
        );
        // Blocks no query position has completed are not kept; the ids past the last kept block are `-1`.
        let zero_from = ubin(c, BinOp::Add, eligible, lane);
        count_to(c, local_op(zero_from), cu(blocks), LANES, |c, block| {
            let at = {
                let base = uc(c, BinOp::Add, block, k);
                ubin(c, BinOp::Add, out_base, base)
            };
            c.assign(elem(out, at), Rvalue::Use(i32c(0)));
        });
        let pad_from = ubin(c, BinOp::Add, want, lane);
        count_to(c, local_op(pad_from), cu(k), LANES, |c, id| {
            let at = ubin(c, BinOp::Add, out_base, id);
            c.assign(elem(out, at), Rvalue::Use(i32c(-1)));
        });
    });
    let mut body = c.finish(name);
    body.workgroup_size = [LANES as u32, 1, 1];
    body.workgroup_locals = vec![
        u32_array(COUNTS_LEN),
        u32_array(LANES),
        u32_array(LANES),
        u32_array(LANES),
        u32_array(CELLS),
    ];
    Ok(body)
}

/// `var <- var + 1` for a usize.
fn bump(c: &mut Cfg, var: poot_kernel_ir::Local) {
    let next = uc(c, BinOp::Add, var, 1);
    c.assign(Place::local(var), Rvalue::Use(local_op(next)));
}
