//! A small statement-accumulating builder shared by the generated contraction and packed bodies, and
//! the one bit-level primitive every packed field read goes through: [`emit_read_bits`].

use poot_kernel_ir::{BinOp, Constant, Local, Operand, Place, Rvalue, Statement, Ty};

use crate::helpers::{Alloc, elem};

/// Accumulates the statements of one straight-line block while minting fresh locals: a field read or
/// an operand load has no branches inside it, so every `DecodePlan`/`emit_*` function appends to one
/// `Emit` and the caller (`crate::contraction`, `crate::packed`) folds the result into a
/// [`poot_kernel_ir::BasicBlock`].
pub(crate) struct Emit<'a> {
    al: &'a mut Alloc,
    stmts: Vec<Statement>,
}

impl<'a> Emit<'a> {
    pub(crate) fn new(al: &'a mut Alloc) -> Self {
        Self {
            al,
            stmts: Vec::new(),
        }
    }

    /// Consume `self`, handing back every statement emitted so far.
    pub(crate) fn take(self) -> Vec<Statement> {
        self.stmts
    }

    /// Mint a fresh local of `ty`, assign it `rvalue`, and return it.
    pub(crate) fn let_(&mut self, ty: Ty, rvalue: Rvalue) -> Local {
        let local = self.al.add(ty, false);
        self.stmts
            .push(Statement::Assign(Place::local(local), rvalue));
        local
    }

    pub(crate) fn assign(&mut self, place: Place, rvalue: Rvalue) {
        self.stmts.push(Statement::Assign(place, rvalue));
    }

    /// `lds[array][idx] = value` (a workgroup-local write): the cooperative-load seam
    /// `crate::packed`'s `Schedule::Tiled` writes its decoded weight and activation tile elements
    /// through.
    pub(crate) fn write_lds(&mut self, array: u8, idx: Operand, value: Operand) {
        self.stmts
            .push(Statement::WorkgroupLocalWrite { idx, value, array });
    }

    /// `lds[array][idx]`, minted as a fresh `ty`-typed local: the dot-product loop's read side of
    /// [`Self::write_lds`].
    pub(crate) fn read_lds(&mut self, ty: Ty, array: u8, idx: Operand) -> Local {
        self.let_(ty, Rvalue::WorkgroupLocalRead { idx, array })
    }
}

pub(crate) fn cu(value: usize) -> Operand {
    Operand::Const(Constant::Usize(value as u64))
}

pub(crate) fn c32(value: u32) -> Operand {
    Operand::Const(Constant::U32(value))
}

pub(crate) fn copy(local: Local) -> Operand {
    Operand::Copy(Place::local(local))
}

pub(crate) fn bin(e: &mut Emit, ty: Ty, op: BinOp, a: Operand, b: Operand) -> Local {
    e.let_(ty, Rvalue::BinaryOp(op, a, b))
}

/// `metadata[index]` (a `&[u32]` shape parameter) cast to `Ty::Usize`: how a shape-generic body reads
/// a shape fact the planner passes per call instead of baking it in.
pub(crate) fn emit_read_meta(e: &mut Emit, metadata: Local, index: usize) -> Local {
    let idx = e.let_(Ty::Usize, Rvalue::Use(cu(index)));
    let word = e.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(metadata, idx))));
    e.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(word),
        },
    )
}

/// Branchless `if cond { if_true } else { if_false }` over raw `u32` bit patterns: `poot-kernel-ir`
/// has no expression-level select, so every format's two-way subnormal/normal split in
/// `emit_float_decode` goes through this one XOR-mask trick (`mask` is `0` or `0xFFFF_FFFF`,
/// never treated as a numeric magnitude the way a `0`/`1` multiply-select would).
pub(crate) fn select_u32(e: &mut Emit, cond: Local, if_true: Local, if_false: Local) -> Local {
    let cond_u32 = e.let_(
        Ty::U32,
        Rvalue::Cast {
            to: Ty::U32,
            operand: copy(cond),
        },
    );
    let mask = bin(e, Ty::U32, BinOp::Sub, c32(0), copy(cond_u32));
    let differs = bin(e, Ty::U32, BinOp::BitXor, copy(if_true), copy(if_false));
    let masked = bin(e, Ty::U32, BinOp::BitAnd, copy(mask), copy(differs));
    bin(e, Ty::U32, BinOp::BitXor, copy(if_false), copy(masked))
}

/// `words[bit..bit+width]` as an unsigned integer (`width` between 1 and 32 inclusive): two `u32`
/// word loads, a double shift and a mask, exact whether or not the field crosses a word boundary.
/// `words` is a `&[u32]` slice (one zero guard word past the last real word, D9), `bit` a runtime
/// bit offset from the start of `words` and `width` a compile-time constant (the field's own
/// [`poot_quant::format::Piece::width`]).
///
/// No branch and no shift by 32 (undefined for a 32-bit operand): the high word's shift amount is
/// reduced mod 32 first, and its contribution is masked to zero, not skipped, whenever the field
/// does not reach it (`bit_in_word <= 32 - width`), via a zero/one integer selector rather than a
/// runtime branch (this decode has no control flow other than the caller's one dispatch guard).
pub(crate) fn emit_read_bits(e: &mut Emit, words: Local, bit: Local, width: u32) -> Local {
    assert!((1..=32).contains(&width), "emit_read_bits: width {width}");
    let word_idx = bin(e, Ty::Usize, BinOp::Div, copy(bit), cu(32));
    let bit_in_word = bin(e, Ty::Usize, BinOp::Rem, copy(bit), cu(32));
    let word_idx2 = bin(e, Ty::Usize, BinOp::Add, copy(word_idx), cu(1));

    let lo = e.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(words, word_idx))));
    let hi = e.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(words, word_idx2))));

    let shift = e.let_(
        Ty::U32,
        Rvalue::Cast {
            to: Ty::U32,
            operand: copy(bit_in_word),
        },
    );
    let lo_shifted = bin(e, Ty::U32, BinOp::Shr, copy(lo), copy(shift));

    // inv_shift = (32 - bit_in_word) % 32: always a valid (0..=31) shift amount, including when
    // bit_in_word == 0 (where the raw 32 - 0 = 32 would be an undefined shift).
    let inv_raw = bin(e, Ty::Usize, BinOp::Sub, cu(32), copy(bit_in_word));
    let inv_shift = bin(e, Ty::Usize, BinOp::Rem, copy(inv_raw), cu(32));
    let inv_shift_u32 = e.let_(
        Ty::U32,
        Rvalue::Cast {
            to: Ty::U32,
            operand: copy(inv_shift),
        },
    );
    let hi_shifted = bin(e, Ty::U32, BinOp::Shl, copy(hi), copy(inv_shift_u32));

    // keep_hi = bit_in_word > (32 - width): the field actually reaches the high word. `threshold`
    // is a generation-time constant (width is fixed per call), so this is the only runtime
    // comparison, not a per-width branch table.
    let threshold = 32usize - width as usize;
    let keep_hi = bin(e, Ty::Bool, BinOp::Gt, copy(bit_in_word), cu(threshold));
    let keep_hi_u32 = e.let_(
        Ty::U32,
        Rvalue::Cast {
            to: Ty::U32,
            operand: copy(keep_hi),
        },
    );
    let hi_contrib = bin(e, Ty::U32, BinOp::Mul, copy(hi_shifted), copy(keep_hi_u32));

    let combined = bin(e, Ty::U32, BinOp::BitOr, copy(lo_shifted), copy(hi_contrib));
    let mask: u32 = if width == 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    };
    bin(e, Ty::U32, BinOp::BitAnd, copy(combined), c32(mask))
}

#[cfg(test)]
mod tests {
    use poot_kernel_ir::interp::{Buffer, run};
    use poot_kernel_ir::{BasicBlock, Body, Terminator};
    use poot_test_util::kernel_fixtures::workgroups_covering;

    use super::*;
    use crate::helpers::{ld, local, slice_dtype, slice_f32};

    /// Build a one-thread body that reads `width` bits at a fixed `bit` offset out of `words` and
    /// stores the raw bits, bitcast to `f32` (the interpreter's `Buffer` has no `u32` readback), into
    /// a one-element `out`. The test recovers the exact bits via `f32::to_bits`.
    fn read_bits_body(bit: usize, width: u32) -> Body {
        let mut al = Alloc::new(vec![
            ld(Ty::Unit, false),
            ld(slice_dtype(Ty::U32, false), false),
            ld(slice_f32(true), true),
        ]);
        let (words, out) = (local(1), local(2));
        let mut emit = Emit::new(&mut al);
        let bit_local = emit.let_(Ty::Usize, Rvalue::Use(cu(bit)));
        let result = emit_read_bits(&mut emit, words, bit_local, width);
        let bits_f32 = emit.let_(
            Ty::F32,
            Rvalue::Bitcast {
                to: Ty::F32,
                operand: copy(result),
            },
        );
        let zero = emit.let_(Ty::Usize, Rvalue::Use(cu(0)));
        emit.assign(elem(out, zero), Rvalue::Use(copy(bits_f32)));
        let block = BasicBlock {
            statements: emit.take(),
            terminator: Terminator::Return,
        };
        Body::new("read_bits_probe", 2, al.locals, vec![block])
    }

    fn read_at(words: &[u32], bit: usize, width: u32) -> u32 {
        let body = read_bits_body(bit, width);
        let mut buffers = [Buffer::from_u32s(words), Buffer::from_f32s(&[0.0])];
        run(&body, workgroups_covering(&body, 1), &mut buffers).unwrap();
        buffers[1].to_f32s().unwrap()[0].to_bits()
    }

    #[test]
    fn reads_within_one_word() {
        let words = [0b1010_1100_u32, 0, 0];
        assert_eq!(read_at(&words, 0, 4), 0b1100);
        assert_eq!(read_at(&words, 4, 4), 0b1010);
        assert_eq!(read_at(&words, 0, 8), 0b1010_1100);
    }

    #[test]
    fn reads_spanning_a_word_boundary() {
        // low word's top 4 bits are 0b1101, high word's low 4 bits are 0b0110: a 8-bit field
        // starting at bit 28 must read 0b0110_1101.
        let words = [0b1101u32 << 28, 0b0110u32, 0];
        assert_eq!(read_at(&words, 28, 8), 0b0110_1101);
    }

    #[test]
    fn reads_a_full_word_at_a_word_boundary() {
        let words = [0xdead_beefu32, 0x1234_5678, 0];
        assert_eq!(read_at(&words, 0, 32), 0xdead_beef);
        assert_eq!(read_at(&words, 32, 32), 0x1234_5678);
    }

    #[test]
    fn reads_the_last_bit_of_a_word_without_touching_the_guard() {
        // width 1 at bit 31: entirely inside the low word (no overflow), so the guard word (all
        // zero here) must contribute nothing.
        let words = [0x8000_0000u32, 0];
        assert_eq!(read_at(&words, 31, 1), 1);
    }
}
