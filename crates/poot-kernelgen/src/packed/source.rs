//! A packed source buffer as a decode reads it: its words and, for an expert-stacked table, the bit
//! offset of the expert slab the read starts in.

use poot_kernel_ir::{BinOp, Local, Ty};

use crate::emit::{Emit, bin, copy, emit_read_bits};

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

    /// `width` bits starting at bit `bit` of this source's slab ([`emit_read_bits`]).
    pub(crate) fn read_bits(self, e: &mut Emit, bit: Local, width: u32) -> Local {
        let bit = match self.slab_bit {
            Some(slab) => bin(e, Ty::Usize, BinOp::Add, copy(slab), copy(bit)),
            None => bit,
        };
        emit_read_bits(e, self.words, bit, width)
    }
}
