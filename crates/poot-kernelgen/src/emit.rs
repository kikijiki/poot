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

    /// The local allocator and the statements so far, for an emitter written against them directly.
    pub(crate) fn parts(&mut self) -> (&mut Alloc, &mut Vec<Statement>) {
        (&mut *self.al, &mut self.stmts)
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

    /// Mint a fresh local of `ty` that no statement assigns yet: the destination of a statement that fills it
    /// itself (a matrix-fragment load or element read), or a register a later statement rewrites (`mutable`).
    pub(crate) fn fresh(&mut self, ty: Ty, mutable: bool) -> Local {
        self.al.add(ty, mutable)
    }

    /// Append `statement` as it is.
    pub(crate) fn push(&mut self, statement: Statement) {
        self.stmts.push(statement);
    }

    /// `lds[array][idx] = value` (a workgroup-local write): the cooperative-load seam
    /// the register-tiled body (`crate::contraction`'s `tiled_body`) writes its decoded weight and activation tile elements
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

/// `words[bit..bit+width]` for a field the emitter proved never straddles a word (`bit % 32 + width <= 32`, which
/// `bit` aligned to at least `width` bits guarantees): one word load, a shift and a mask. No second load, no
/// comparison, no multiply-select, unlike [`emit_read_bits`].
pub(crate) fn emit_read_narrow(e: &mut Emit, words: Local, bit: Local, width: u32) -> Local {
    assert!((1..=32).contains(&width), "emit_read_narrow: width {width}");
    let word_idx = bin(e, Ty::Usize, BinOp::Div, copy(bit), cu(32));
    let bit_in_word = bin(e, Ty::Usize, BinOp::Rem, copy(bit), cu(32));
    let word = e.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(words, word_idx))));
    let shift = e.let_(
        Ty::U32,
        Rvalue::Cast {
            to: Ty::U32,
            operand: copy(bit_in_word),
        },
    );
    let shifted = bin(e, Ty::U32, BinOp::Shr, copy(word), copy(shift));
    if width == 32 {
        return shifted;
    }
    bin(
        e,
        Ty::U32,
        BinOp::BitAnd,
        copy(shifted),
        c32((1u32 << width) - 1),
    )
}

/// The `span_bits` bits of `words` starting at bit `bit`, as `span_bits.div_ceil(32)` consecutive 32-bit windows:
/// window `c` holds bits `bit + 32 * c ..` as a `u32`, and bits at or past `bit + span_bits` are unspecified. This is
/// the read a run of neighbouring fields shares.
///
/// The emitter proved two facts about `bit` ([`crate::packed::PieceAlignment`]): the word it starts in,
/// `bit / 32`, is a multiple of `word_align_bits / 32` words (a power of two, at most
/// [`poot_kernel_ir::BUFFER_PARAM_ALIGN`] bytes), and `bit % 32 <= max_residue`. The read is exactly the words
/// `[bit / 32, (bit % 32 + span_bits).div_ceil(32))` the span can touch, which is at most
/// `(max_residue + span_bits).div_ceil(32)` words: in `u32x4` loads where the word index is 4-aligned and four words
/// remain, `u32x2` loads where it is 2-aligned and two remain, scalar loads for the rest. When `max_residue` is
/// zero the windows are the words. Otherwise one funnel shift per window, `(lo >> sh) | (hi << (32 - sh))` with the
/// left shift split in two (`(hi << 1) << (31 - sh)`) so `sh == 0` never shifts by 32; a window whose high word the
/// span cannot reach is the low word shifted alone. No word past the span's last bit is read and nothing is clamped.
pub(crate) fn emit_read_run(
    e: &mut Emit,
    words: Local,
    bit: Local,
    span_bits: usize,
    word_align_bits: u32,
    max_residue: u32,
) -> Vec<Local> {
    assert!(span_bits >= 1, "emit_read_run: empty span");
    assert!(
        word_align_bits.is_power_of_two()
            && (32..=poot_kernel_ir::BUFFER_PARAM_ALIGN as u32 * 8).contains(&word_align_bits)
            && max_residue < 32,
        "emit_read_run: word alignment {word_align_bits} bits, residue {max_residue}"
    );
    let chunks = span_bits.div_ceil(32);
    let touched = (max_residue as usize + span_bits).div_ceil(32);
    let first = bin(e, Ty::Usize, BinOp::Div, copy(bit), cu(32));
    let loaded = emit_aligned_words(e, words, first, touched, word_align_bits);
    if max_residue == 0 {
        return loaded;
    }
    let bit_in_word = bin(e, Ty::Usize, BinOp::Rem, copy(bit), cu(32));
    let shift = e.let_(
        Ty::U32,
        Rvalue::Cast {
            to: Ty::U32,
            operand: copy(bit_in_word),
        },
    );
    let complement = bin(e, Ty::U32, BinOp::Sub, c32(31), copy(shift));
    (0..chunks)
        .map(|c| {
            let low = bin(e, Ty::U32, BinOp::Shr, copy(loaded[c]), copy(shift));
            let Some(&next) = loaded.get(c + 1) else {
                return low;
            };
            let once = bin(e, Ty::U32, BinOp::Shl, copy(next), c32(1));
            let high = bin(e, Ty::U32, BinOp::Shl, copy(once), copy(complement));
            bin(e, Ty::U32, BinOp::BitOr, copy(low), copy(high))
        })
        .collect()
}

/// `chunks` words of `words` from word `first`, a multiple of `align_bits / 32` words: the widest vector load the
/// alignment proves for each group the remaining words fill.
fn emit_aligned_words(
    e: &mut Emit,
    words: Local,
    first: Local,
    chunks: usize,
    align_bits: u32,
) -> Vec<Local> {
    let mut windows = Vec::with_capacity(chunks);
    while windows.len() < chunks {
        let at = windows.len();
        let left = chunks - at;
        let lanes = [4usize, 2, 1]
            .into_iter()
            .find(|&lanes| left >= lanes && align_bits as usize >= lanes * 32)
            .expect("one word always fits");
        let index = if at == 0 {
            first
        } else {
            bin(e, Ty::Usize, BinOp::Add, copy(first), cu(at))
        };
        if lanes == 1 {
            windows.push(e.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(words, index)))));
            continue;
        }
        let group = e.let_(
            Ty::Vec {
                elem: Box::new(Ty::U32),
                lanes: lanes as u32,
            },
            Rvalue::VectorLoad {
                place: elem(words, index),
                align_bytes: lanes as u32 * 4,
            },
        );
        for lane in 0..lanes as u32 {
            windows.push(e.let_(
                Ty::U32,
                Rvalue::VectorExtract {
                    vector: copy(group),
                    lane,
                },
            ));
        }
    }
    windows
}

#[cfg(test)]
mod tests {
    use poot_kernel_ir::interp::{Buffer, run};
    use poot_kernel_ir::test_support::workgroups_covering;
    use poot_kernel_ir::{BasicBlock, Body, Terminator};

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

    /// A one-thread body that stores the windows of [`emit_read_run`] over `span` bits at bit `bit` of `words`, whose
    /// start the emitter proves word-aligned to `word_align` bits with at most `residue` bits in its word, bitcast to
    /// `f32` (the interpreter has no `u32` readback), into `out[0..windows]`.
    fn windows_body(bit: usize, span: usize, word_align: u32, residue: u32) -> Body {
        let mut al = Alloc::new(vec![
            ld(Ty::Unit, false),
            ld(slice_dtype(Ty::U32, false), false),
            ld(slice_f32(true), true),
        ]);
        let (words, out) = (local(1), local(2));
        let mut emit = Emit::new(&mut al);
        let bit_local = emit.let_(Ty::Usize, Rvalue::Use(cu(bit)));
        let windows = emit_read_run(&mut emit, words, bit_local, span, word_align, residue);
        for (c, window) in windows.into_iter().enumerate() {
            let as_f32 = emit.let_(
                Ty::F32,
                Rvalue::Bitcast {
                    to: Ty::F32,
                    operand: copy(window),
                },
            );
            let at = emit.let_(Ty::Usize, Rvalue::Use(cu(c)));
            emit.assign(elem(out, at), Rvalue::Use(copy(as_f32)));
        }
        let block = BasicBlock {
            statements: emit.take(),
            terminator: Terminator::Return,
        };
        Body::new("read_windows_probe", 2, al.locals, vec![block])
    }

    fn windows_at(
        words: &[u32],
        bit: usize,
        span: usize,
        word_align: u32,
        residue: u32,
    ) -> Result<Vec<u32>, poot_kernel_ir::interp::InterpError> {
        let body = windows_body(bit, span, word_align, residue);
        let mut buffers = [
            Buffer::from_u32s(words),
            Buffer::from_f32s(&vec![0.0; span.div_ceil(32)]),
        ];
        run(&body, workgroups_covering(&body, 1), &mut buffers)?;
        Ok(buffers[1]
            .to_f32s()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    /// Bits `at .. at + 32` of `words` read one bit at a time, zero past the last word.
    fn scalar_window(words: &[u32], at: usize) -> u32 {
        (0..32).fold(0u32, |acc, k| {
            let bit = at + k;
            let set = words
                .get(bit / 32)
                .is_some_and(|word| word >> (bit % 32) & 1 == 1);
            acc | u32::from(set) << k
        })
    }

    /// The widest alignment in bits, from 32 to 128, that the word `bit` starts in is a multiple of.
    fn word_alignment_of(bit: usize) -> u32 {
        1 << ((bit - bit % 32) | 128).trailing_zeros().min(7)
    }

    /// Every window over every span of 1 to 160 bits at every start bit of the data words (the last word is the
    /// zero guard), read at the word alignment the start's word has and the residue it has: the widest vector
    /// load, plain words, or the funnel (`bit % 32 == 0` with a residue bound, where the shift is 0 and `hi << 32`
    /// would be undefined unless split; the last data bit). Only the bits of the span are compared: the rest of
    /// the last window is unspecified.
    /// Mutation: `31 - shift` -> `30 - shift` in `emit_read_run`; every start bit but a word's first goes red.
    #[test]
    fn windows_equal_the_scalar_read_at_every_start_bit() {
        let words = [
            0xdead_beef_u32,
            0x1234_5678,
            0x8000_0001,
            0xffff_0000,
            0x0f0f_a5a5,
            0x7777_1111,
            0xcafe_f00d,
            0x0123_4567,
            0x89ab_cdef,
            0x1357_9bdf,
            0,
        ];
        for span in (1..=160).step_by(7).chain([32, 64, 96, 128, 160]) {
            for bit in 0..(words.len() - 1) * 32 {
                let residue = (bit % 32) as u32;
                // The words the span touches, all inside the slice: the data and its zero guard.
                if bit / 32 + (bit % 32 + span).div_ceil(32) > words.len() {
                    continue;
                }
                let align = word_alignment_of(bit);
                let got = windows_at(&words, bit, span, align, residue).unwrap();
                assert_eq!(got.len(), span.div_ceil(32));
                for (c, window) in got.into_iter().enumerate() {
                    let used = (span - 32 * c).min(32);
                    let mask = if used == 32 {
                        u32::MAX
                    } else {
                        (1 << used) - 1
                    };
                    assert_eq!(
                        window & mask,
                        scalar_window(&words, bit + 32 * c) & mask,
                        "bit {bit} (word align {align}), span {span}, window {c}"
                    );
                }
            }
        }
    }

    /// An aligned run reads exactly the words of its span: no word past it, whatever the vector split, so the last
    /// word of the data needs no guard. Mutation: read one word more than `touched` in `emit_read_run`; the
    /// buffer ends and the interpreter faults.
    #[test]
    fn an_aligned_run_reads_only_its_words() {
        let words = [
            0xdead_beef_u32,
            0x1234_5678,
            0x8000_0001,
            0xffff_0000,
            0x0f0f_a5a5,
        ];
        for count in 1..=5 {
            let got = windows_at(&words[..count], 0, count * 32, 128, 0).unwrap();
            assert_eq!(
                got,
                words[..count],
                "{count} word(s) of a {count}-word buffer"
            );
        }
    }

    /// A run declared more aligned than its start is a wrong proof: the interpreter refuses the vector load.
    #[test]
    fn a_run_declared_more_aligned_than_its_start_is_refused() {
        let words = [1u32, 2, 3, 4, 5, 6, 7, 8, 0];
        let refused = windows_at(&words, 32, 128, 128, 0).unwrap_err();
        assert!(
            matches!(
                refused,
                poot_kernel_ir::interp::InterpError::Fault {
                    fault: poot_kernel_ir::interp::Fault::MisalignedVectorLoad { .. },
                    ..
                }
            ),
            "{refused:?}"
        );
    }
}
