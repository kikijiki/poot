//! The stable rank generator (POOT-1543): per row, the position of every element under "eligible first,
//! canonical score descending, lower index first", by a bitonic sort of the row's distinct
//! `(class, key, index)` tuples in workgroup memory.
//!
//! One workgroup per row of [`StableRankSpec::lanes`] lanes. The row is padded to the next power of two `N`
//! with entries that sort after every real one. The steps:
//!
//! 1. Each lane loads its elements. An entry is two words: the order-preserving key of the canonical score
//!    ([`crate::order_key::order_key`]) and a tag holding the class in bit 31 and the element's index below it.
//!    The class is `1` for a real element (`eligible >= 1` when masked, else `0`) and `0` for padding; a pad's
//!    key is `0`, below the key of any canonical score, and its index is its position, at or past `E`.
//! 2. A bitonic network of `log2(N) * (log2(N) + 1) / 2` compare-exchange stages with a workgroup barrier
//!    after each orders the entries by `class` descending, `key` descending, `index` ascending. That order is
//!    total: the indices are distinct, so no two entries compare equal and the sorted order is the unique one,
//!    whatever the stage order, lane count, wave size or backend. This is also why the network need not be
//!    stable. It is the definition's order: `#{j : s_j > s_i} + #{j < i : s_j == s_i}` is the position of `i`
//!    under it, and the masked definition adds the class as the first key.
//! 3. The entry at sorted position `p` with index `i < E` writes `rank[row, i] = p`.
//!
//! No atomics. Parameters: the scores `[rows, E]` F32, the eligibility `[rows, E]` F32 when masked, then the
//! F32 result `[rows, E]`.

use poot_kernel_ir::{BinOp, Body, Constant, IndexAxis, Operand, Rvalue, Ty, WorkgroupLocalDecl};

use crate::KernelGenError;
use crate::attention::{count_to, local_op, ubin, uc};
use crate::block_top_k::{select, ubin32};
use crate::data_movement::Cfg;
use crate::emit::{c32, cu};
use crate::helpers::{copy, elem, ld, slice_f32};
use crate::order_key::order_key;

/// The most lanes a workgroup of this kernel has.
pub const STABLE_RANK_MAX_LANES: usize = 256;

const GENERATOR: &str = "stable_rank_body";

/// A rank is an exact F32 integer below `2^24`.
const MAX_EXPERTS: usize = 1 << 24;

/// What the planner asks the stable rank generator for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StableRankSpec {
    /// Rows: `numel / experts`, one workgroup each.
    pub rows: usize,
    /// Elements ranked per row.
    pub experts: usize,
    /// Whether an eligibility operand ranks the eligible elements first.
    pub masked: bool,
}

impl StableRankSpec {
    /// Entries sorted per row: the row padded to a power of two, at least two.
    pub fn padded(&self) -> usize {
        self.experts.next_power_of_two().max(2)
    }

    /// Lanes of the workgroup: half the entries, at most [`STABLE_RANK_MAX_LANES`]. Every stage compares
    /// `padded / 2` pairs.
    pub fn lanes(&self) -> usize {
        (self.padded() / 2).min(STABLE_RANK_MAX_LANES)
    }

    /// Static workgroup memory in bytes: a key word and a tag word per entry.
    pub fn lds_bytes(&self) -> u64 {
        8 * self.padded() as u64
    }
}

/// Workgroup-local arrays, both `u32` of the padded length.
const KEYS: u8 = 0;
const TAGS: u8 = 1;

/// The class bit of a tag; the index is below it.
const CLASS_BIT: u32 = 0x8000_0000;
const INDEX_MASK: u32 = 0x7FFF_FFFF;

fn validate(spec: &StableRankSpec) -> Result<(), KernelGenError> {
    for (what, value) in [("rows", spec.rows), ("experts", spec.experts)] {
        if value == 0 {
            return Err(KernelGenError::BelowMinimum {
                generator: GENERATOR,
                what: what.to_string(),
                value,
                min: 1,
            });
        }
    }
    if spec.experts > MAX_EXPERTS {
        return Err(KernelGenError::ExceedsBound {
            generator: GENERATOR,
            what: "experts".to_string(),
            value: spec.experts,
            bound: MAX_EXPERTS,
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

fn bool_const(value: bool) -> Operand {
    Operand::Const(Constant::Bool(value))
}

fn read(c: &mut Cfg, array: u8, at: poot_kernel_ir::Local) -> poot_kernel_ir::Local {
    c.emit(|e| e.read_lds(Ty::U32, array, local_op(at)))
}

fn write(c: &mut Cfg, array: u8, at: poot_kernel_ir::Local, value: Operand) {
    c.emit(|e| e.write_lds(array, local_op(at), value));
}

/// Whether entry `a` sorts before entry `b`: class descending, key descending, index ascending.
fn sorts_before(
    c: &mut Cfg,
    (key_a, tag_a): (poot_kernel_ir::Local, poot_kernel_ir::Local),
    (key_b, tag_b): (poot_kernel_ir::Local, poot_kernel_ir::Local),
) -> poot_kernel_ir::Local {
    let class_a = ubin32(c, BinOp::Shr, local_op(tag_a), c32(31));
    let class_b = ubin32(c, BinOp::Shr, local_op(tag_b), c32(31));
    let index_a = ubin32(c, BinOp::BitAnd, local_op(tag_a), c32(INDEX_MASK));
    let index_b = ubin32(c, BinOp::BitAnd, local_op(tag_b), c32(INDEX_MASK));
    let class_above = c.bin(Ty::Bool, BinOp::Gt, local_op(class_a), local_op(class_b));
    let class_same = c.bin(Ty::Bool, BinOp::Eq, local_op(class_a), local_op(class_b));
    let key_above = c.bin(Ty::Bool, BinOp::Gt, local_op(key_a), local_op(key_b));
    let key_same = c.bin(Ty::Bool, BinOp::Eq, local_op(key_a), local_op(key_b));
    let index_below = c.bin(Ty::Bool, BinOp::Lt, local_op(index_a), local_op(index_b));
    let on_tie = select(
        c,
        Ty::Bool,
        key_same,
        local_op(index_below),
        bool_const(false),
    );
    let in_class = select(c, Ty::Bool, key_above, bool_const(true), local_op(on_tie));
    let in_order = select(
        c,
        Ty::Bool,
        class_same,
        local_op(in_class),
        bool_const(false),
    );
    select(
        c,
        Ty::Bool,
        class_above,
        bool_const(true),
        local_op(in_order),
    )
}

/// Build the stable rank kernel `spec` describes, launched as `spec.rows` workgroups of [`StableRankSpec::lanes`]
/// lanes over a 2-D grid `x_groups` workgroups across: `gi = GroupY * x_groups + GroupX` is the row, and a
/// workgroup past the last returns.
pub(crate) fn stable_rank_body(
    name: &str,
    spec: &StableRankSpec,
    x_groups: usize,
) -> Result<Body, KernelGenError> {
    validate(spec)?;
    let StableRankSpec {
        rows,
        experts,
        masked,
    } = *spec;
    let padded = spec.padded();
    let lanes = spec.lanes();
    let mut params = vec![ld(Ty::Unit, false), ld(slice_f32(false), false)];
    if masked {
        params.push(ld(slice_f32(false), false));
    }
    params.push(ld(slice_f32(true), true));
    let scores = poot_kernel_ir::Local { index: 1 };
    let eligible = masked.then_some(poot_kernel_ir::Local { index: 2 });
    let out = poot_kernel_ir::Local {
        index: params.len() as u32 - 1,
    };
    let mut c = Cfg::new(params);
    let group_x = c.index(IndexAxis::GroupX);
    let group_y = c.index(IndexAxis::GroupY);
    let lane = c.index(IndexAxis::LocalX);
    let row_offset = uc(&mut c, BinOp::Mul, group_y, x_groups);
    let row = ubin(&mut c, BinOp::Add, row_offset, group_x);
    let in_range = c.bin(Ty::Bool, BinOp::Lt, local_op(row), cu(rows));
    c.if_then(in_range, |c| {
        let base = uc(c, BinOp::Mul, row, experts);

        // Load: this lane's entries, real ones from the row and padding past `experts`.
        count_to(c, local_op(lane), cu(padded), lanes, |c, entry| {
            let real = c.bin(Ty::Bool, BinOp::Lt, local_op(entry), cu(experts));
            let zero = c.usize_const(0);
            let index = select(c, Ty::Usize, real, local_op(entry), local_op(zero));
            let at = ubin(c, BinOp::Add, base, index);
            let value = c.let_(Ty::F32, Rvalue::Use(copy(elem(scores, at))));
            let key = order_key(c, value);
            let key = select(c, Ty::U32, real, local_op(key), c32(0));
            let class = match eligible {
                Some(eligible) => {
                    let level = c.let_(Ty::F32, Rvalue::Use(copy(elem(eligible, at))));
                    let is_eligible = c.bin(
                        Ty::Bool,
                        BinOp::Ge,
                        local_op(level),
                        Operand::Const(Constant::F32(1.0)),
                    );
                    select(c, Ty::U32, is_eligible, c32(CLASS_BIT), c32(0))
                }
                None => c.let_(Ty::U32, Rvalue::Use(c32(CLASS_BIT))),
            };
            let class = select(c, Ty::U32, real, local_op(class), c32(0));
            let index = c.let_(
                Ty::U32,
                Rvalue::Cast {
                    to: Ty::U32,
                    operand: local_op(entry),
                },
            );
            let tag = ubin32(c, BinOp::BitOr, local_op(class), local_op(index));
            write(c, KEYS, entry, local_op(key));
            write(c, TAGS, entry, local_op(tag));
        });
        c.barrier();

        // The bitonic network: `size` doubles, `half` (the distance compared) halves inside each size.
        let mut size = 2;
        while size <= padded {
            let mut half = size / 2;
            while half >= 1 {
                count_to(c, local_op(lane), cu(padded / 2), lanes, |c, pair| {
                    // The pair's lower entry has bit `half` clear: spread the pair index around that bit.
                    let high = uc(c, BinOp::BitAnd, pair, (padded / 2 - 1) & !(half - 1));
                    let low = uc(c, BinOp::BitAnd, pair, half - 1);
                    let high = uc(c, BinOp::Mul, high, 2);
                    let lower = ubin(c, BinOp::BitOr, high, low);
                    let upper = uc(c, BinOp::Add, lower, half);
                    let a = (read(c, KEYS, lower), read(c, TAGS, lower));
                    let b = (read(c, KEYS, upper), read(c, TAGS, upper));
                    // Blocks of `size` entries alternate direction; the last block, `size == padded`, ascends.
                    let direction = uc(c, BinOp::BitAnd, lower, size);
                    let descending = c.bin(Ty::Bool, BinOp::Ne, local_op(direction), cu(0));
                    let b_first = sorts_before(c, b, a);
                    let a_first = sorts_before(c, a, b);
                    let swap = select(
                        c,
                        Ty::Bool,
                        descending,
                        local_op(a_first),
                        local_op(b_first),
                    );
                    c.if_then(swap, |c| {
                        write(c, KEYS, lower, local_op(b.0));
                        write(c, TAGS, lower, local_op(b.1));
                        write(c, KEYS, upper, local_op(a.0));
                        write(c, TAGS, upper, local_op(a.1));
                    });
                });
                c.barrier();
                half /= 2;
            }
            size *= 2;
        }

        // Each real entry writes the position it sorted to.
        count_to(c, local_op(lane), cu(padded), lanes, |c, position| {
            let tag = read(c, TAGS, position);
            let index = ubin32(c, BinOp::BitAnd, local_op(tag), c32(INDEX_MASK));
            let real = c.bin(Ty::Bool, BinOp::Lt, local_op(index), c32(experts as u32));
            c.if_then(real, |c| {
                let index = c.let_(
                    Ty::Usize,
                    Rvalue::Cast {
                        to: Ty::Usize,
                        operand: local_op(index),
                    },
                );
                let at = ubin(c, BinOp::Add, base, index);
                let position = c.let_(
                    Ty::U32,
                    Rvalue::Cast {
                        to: Ty::U32,
                        operand: local_op(position),
                    },
                );
                let rank = c.let_(
                    Ty::F32,
                    Rvalue::Cast {
                        to: Ty::F32,
                        operand: local_op(position),
                    },
                );
                c.assign(elem(out, at), Rvalue::Use(local_op(rank)));
            });
        });
    });
    let mut body = c.finish(name);
    body.workgroup_size = [lanes as u32, 1, 1];
    body.workgroup_locals = vec![u32_array(padded), u32_array(padded)];
    Ok(body)
}
