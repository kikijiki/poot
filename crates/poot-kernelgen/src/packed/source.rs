//! A packed source buffer as a decode reads it: its words and, for an expert-stacked table, the bit
//! offset of the expert slab the read starts in; and the alignment a descriptor proves for the bits a piece reads.

use poot_kernel_ir::{BUFFER_PARAM_ALIGN, BinOp, Local, Ty};
use poot_quant::format::{BlockLayout, Piece};

use crate::emit::{Emit, bin, copy, emit_read_bits, emit_read_narrow, emit_read_run};

/// The widest alignment a read can be proven to have, in bits: a buffer parameter starts on
/// [`BUFFER_PARAM_ALIGN`].
const MAX_ALIGN_BITS: u32 = BUFFER_PARAM_ALIGN as u32 * 8;

/// What the descriptor proves about the bit positions a piece reads at the starts of its runs
/// (`PieceAlignment::of`). It is a fact of the format and the run, independent of `K` and of the shape: a block
/// starts at a multiple of its own size in bits, a row is whole blocks, the buffer starts on
/// [`BUFFER_PARAM_ALIGN`], and the piece's offset in the block is `layout.bit(i)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PieceAlignment {
    /// The largest power of two, in bits, every run start is a multiple of.
    start_bits: u32,
    /// The largest power of two, in bits and at least 32, every run start's word (its start rounded down to a
    /// multiple of 32) is a multiple of: what a vector load of the run's words needs.
    word_bits: u32,
    /// The most bits a run start sits past the start of its word.
    max_residue: u32,
}

/// The largest power of two dividing `value`, at most the buffer's alignment; every alignment divides zero.
fn lowest_set_bit(value: u32) -> u32 {
    if value == 0 {
        MAX_ALIGN_BITS
    } else {
        (1 << value.trailing_zeros()).min(MAX_ALIGN_BITS)
    }
}

impl PieceAlignment {
    /// The alignment of the run of `count` elements of `piece` that starts `delta` bits past the element
    /// `first`'s bit, for every `first` a multiple of `count` below the block's values (a run of one element is
    /// every element). A block starts at a multiple of `block_bits`, so its residue in a word is a multiple of
    /// `gcd(block_bits, 32)`; every such residue is tried.
    pub(crate) fn of(layout: BlockLayout, piece: &Piece, count: usize, delta: usize) -> Self {
        let block_bits = u32::try_from(layout.bytes * 8).expect("a block is under 2^29 bytes");
        let block_align = lowest_set_bit(block_bits);
        let residue_step = lowest_set_bit(block_bits).min(32) as usize;
        let mut aligned = Self {
            start_bits: block_align,
            word_bits: block_align.max(32),
            max_residue: 0,
        };
        for first in (0..layout.values).step_by(count) {
            let offset = piece.layout.bit(first as u32) + delta as u32;
            aligned.start_bits = aligned.start_bits.min(lowest_set_bit(offset));
            // Below a word the block's own start moves the word; only a word-aligned block fixes it.
            aligned.word_bits = if block_align >= 32 {
                aligned.word_bits.min(lowest_set_bit(offset - offset % 32))
            } else {
                32
            };
            let residues = (0..32).step_by(residue_step);
            let worst = residues.map(|block| (block + offset) % 32).max();
            aligned.max_residue = aligned.max_residue.max(worst.unwrap_or(0));
        }
        aligned
    }

    /// The alignment of a single field read, in bits.
    pub(crate) fn start_bits(self) -> u32 {
        self.start_bits
    }
}

/// One source buffer of `u32` words. `slab_bit` is the bit offset of the slab (one expert's copy of this
/// source in a stacked table, [`poot_quant::PackedWeight::slab_words`] words per expert) that every read
/// is relative to; `None` for a source that is one weight's own bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Source {
    pub(crate) words: Local,
    pub(crate) slab_bit: Option<Local>,
}

impl Source {
    /// A weight's own source: reads are relative to the buffer's first word.
    pub(crate) fn plain(words: Local) -> Self {
        Self {
            words,
            slab_bit: None,
        }
    }

    /// `bit` relative to the buffer's first word.
    fn absolute(self, e: &mut Emit, bit: Local) -> Local {
        match self.slab_bit {
            Some(slab) => bin(e, Ty::Usize, BinOp::Add, copy(slab), copy(bit)),
            None => bit,
        }
    }

    /// What a piece's alignment proves for a read of this source: a stacked table's slab starts on a word, not on
    /// anything wider, so the slab caps the alignments of the starts and of their words at 32 bits.
    fn proven(self, piece: PieceAlignment) -> PieceAlignment {
        match self.slab_bit {
            Some(_) => PieceAlignment {
                start_bits: piece.start_bits.min(32),
                word_bits: 32,
                ..piece
            },
            None => piece,
        }
    }

    /// `width` bits starting at bit `bit` of this source's slab ([`emit_read_bits`]), at any alignment.
    pub(crate) fn read_bits(self, e: &mut Emit, bit: Local, width: u32) -> Local {
        let bit = self.absolute(e, bit);
        emit_read_bits(e, self.words, bit, width)
    }

    /// `width` bits starting at bit `bit` of this source's slab, whose alignment the descriptor proves. A field no
    /// wider than that alignment never straddles a word, so it is one load ([`emit_read_narrow`]); a wider one is
    /// read at any alignment ([`Self::read_bits`]).
    pub(crate) fn read_field(
        self,
        e: &mut Emit,
        bit: Local,
        width: u32,
        align: PieceAlignment,
    ) -> Local {
        if width > self.proven(align).start_bits().min(32) {
            return self.read_bits(e, bit, width);
        }
        let bit = self.absolute(e, bit);
        emit_read_narrow(e, self.words, bit, width)
    }

    /// The `span_bits` bits starting at bit `bit` of this source's slab, as `span_bits.div_ceil(32)` consecutive
    /// 32-bit windows ([`emit_read_run`]): `bit` is aligned as `align` proves, so the read is exactly the words the
    /// span can touch, in vector loads where the alignment allows, and a funnel shift only where a run start can
    /// sit inside a word.
    pub(crate) fn read_run(
        self,
        e: &mut Emit,
        bit: Local,
        span_bits: usize,
        align: PieceAlignment,
    ) -> Vec<Local> {
        let bit = self.absolute(e, bit);
        let align = self.proven(align);
        emit_read_run(
            e,
            self.words,
            bit,
            span_bits,
            align.word_bits,
            align.max_residue,
        )
    }
}
