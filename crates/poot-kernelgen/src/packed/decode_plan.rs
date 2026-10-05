//! `DecodePlan`: a [`FormatDescriptor`] evaluated for emission (dquant.md 3.2-3.3). 542a covers block
//! storage without sub-block factors (`SubScale`/`SubMin` absent): one `Scale` and, for `Q4_1`/`Q5_1`,
//! one `Min` value per whole block. 542b adds the K-quant super-blocks, whose `Scale`/`Min` are one
//! value per whole block and whose `SubScale`/`SubMin` are one value per sub-block (`format::Field`'s
//! digits index the sub-block through the element index `i`, never a separately hoisted index):
//! `scale = Scale * SubScale`, `min = Min * SubMin`, either factor absent when the format has no such
//! role (dquant.md's value formula). `DecodePlan` holds no layout of its own - every constant it uses
//! comes straight from the descriptor's `Field`/`BitLayout`/`Piece` data.

use poot_kernel_ir::{BinOp, Local, Rvalue, Ty};
use poot_quant::format::{
    BitLayout, BlockLayout, Field, FieldEncoding, FormatDescriptor, MinSign, OperandRole, Storage,
    ValueMap, WeightFormat,
};

use super::float_decode::emit_float_decode;
use crate::KernelGenError;
use crate::emit::{Emit, bin, c32, copy, cu, select_u32};

/// The `Scale`/`Min` factors of one block, hoisted once and reused by every element the block holds
/// (`emit_value` combines them with the per-element code through the value formula).
pub(crate) struct Factors {
    pub(crate) scale: Option<Local>,
    pub(crate) min: Option<Local>,
}

pub(crate) struct DecodePlan {
    layout: BlockLayout,
    codes: Field,
    scale: Option<Field>,
    sub_scale: Option<Field>,
    min: Option<Field>,
    sub_min: Option<Field>,
    value: ValueMap,
    min_sign: Option<MinSign>,
}

fn reject(format: WeightFormat, reason: &'static str) -> KernelGenError {
    KernelGenError::UnsupportedDescriptor { format, reason }
}

impl DecodePlan {
    pub(crate) fn of(descriptor: &FormatDescriptor) -> Result<Self, KernelGenError> {
        let format = descriptor.format;
        let Storage::Blocks(layout) = descriptor.storage else {
            return Err(reject(format, "planar storage (card 542c's front end)"));
        };
        let Some(codes) = layout.field(OperandRole::Codes) else {
            return Err(reject(format, "no Codes field"));
        };
        for field in [
            Some(codes),
            layout.field(OperandRole::Scale),
            layout.field(OperandRole::SubScale),
            layout.field(OperandRole::Min),
            layout.field(OperandRole::SubMin),
        ]
        .into_iter()
        .flatten()
        {
            if field.field.pieces.len() > 2 {
                return Err(reject(format, "a field of more than two pieces"));
            }
        }
        Ok(Self {
            layout,
            codes: codes.field,
            scale: layout.field(OperandRole::Scale).map(|f| f.field),
            sub_scale: layout.field(OperandRole::SubScale).map(|f| f.field),
            min: layout.field(OperandRole::Min).map(|f| f.field),
            sub_min: layout.field(OperandRole::SubMin).map(|f| f.field),
            value: descriptor.value,
            min_sign: descriptor.min_sign,
        })
    }

    /// Values per block (the block-32 formats' `32`).
    pub(crate) fn values(&self) -> usize {
        self.layout.values
    }

    /// The length of the aligned element runs whose `Scale`/`SubScale`/`Min`/`SubMin` bits are all
    /// the same, so one [`Factors`] serves every element of a run of `n` elements starting at a
    /// multiple of `n` whenever `n` divides this (card 653's Gemv runs). Per factor piece it is the
    /// product of the least-significant digits that do not move the bit address (a piece whose
    /// digits all have stride 0, or that has none, covers the whole block); the minimum over every
    /// factor piece, or the whole block when the format has no factor.
    pub(crate) fn factor_run(&self) -> usize {
        [self.scale, self.sub_scale, self.min, self.sub_min]
            .into_iter()
            .flatten()
            .flat_map(|field| field.pieces.iter())
            .map(|piece| {
                let digits = piece.layout.digits;
                if digits.iter().all(|digit| digit.stride == 0) {
                    self.layout.values
                } else {
                    digits
                        .iter()
                        .rev()
                        .take_while(|digit| digit.stride == 0)
                        .map(|digit| digit.extent as usize)
                        .product()
                }
            })
            .min()
            .unwrap_or(self.layout.values)
    }

    fn block_base_bit(&self, e: &mut Emit, block: Local) -> Local {
        let block_bits = self.layout.bytes * 8;
        bin(e, Ty::Usize, BinOp::Mul, copy(block), cu(block_bits))
    }

    /// `layout.bit(i)` (see `format::BitLayout`), emitted for a runtime `i`: the affine mixed-radix
    /// digit decomposition, least-significant digit first (the array is declared most-significant
    /// first, so this walks it in reverse - the same order `BitLayout::bit`'s own loop takes).
    fn emit_bit_layout(&self, e: &mut Emit, base: Local, layout: &BitLayout, i: Local) -> Local {
        let mut bit = bin(
            e,
            Ty::Usize,
            BinOp::Add,
            copy(base),
            cu(layout.base as usize),
        );
        let mut remaining = i;
        for digit in layout.digits.iter().rev() {
            let extent = digit.extent as usize;
            let stride = digit.stride as usize;
            let this_digit = bin(e, Ty::Usize, BinOp::Rem, copy(remaining), cu(extent));
            let contribution = bin(e, Ty::Usize, BinOp::Mul, copy(this_digit), cu(stride));
            bit = bin(e, Ty::Usize, BinOp::Add, copy(bit), copy(contribution));
            remaining = bin(e, Ty::Usize, BinOp::Div, copy(remaining), cu(extent));
        }
        bit
    }

    /// `field`'s raw bits at in-block index `i` (a broadcast field, no digits, ignores `i`): every
    /// piece read via [`crate::emit::emit_read_bits`] and concatenated low bits first.
    fn emit_field_bits(
        &self,
        e: &mut Emit,
        words: Local,
        block_bit: Local,
        field: &Field,
        i: Local,
    ) -> Local {
        assert!(
            !field.pieces.is_empty() && field.pieces.len() <= 2,
            "format tests / DecodePlan::of: a field of one or two pieces"
        );
        let low_piece = &field.pieces[0];
        let low_bit = self.emit_bit_layout(e, block_bit, &low_piece.layout, i);
        let low = crate::emit::emit_read_bits(e, words, low_bit, low_piece.width);
        let Some(high_piece) = field.pieces.get(1) else {
            return low;
        };
        let high_bit = self.emit_bit_layout(e, block_bit, &high_piece.layout, i);
        let high = crate::emit::emit_read_bits(e, words, high_bit, high_piece.width);
        // Low bits first (`format::Field` docs): the high piece shifts up by the low piece's width.
        let shifted = bin(e, Ty::U32, BinOp::Shl, copy(high), c32(low_piece.width));
        bin(e, Ty::U32, BinOp::BitOr, copy(low), copy(shifted))
    }

    /// `field`'s value at in-block index `i`, through its own [`FieldEncoding`] (never the `Codes`
    /// field's [`ValueMap`], which [`Self::emit_value`] applies separately): used for `Scale`/`Min`,
    /// which are always plain numbers (`Unsigned`, `Signed` or `Float`).
    fn emit_field_number(
        &self,
        e: &mut Emit,
        words: Local,
        block_bit: Local,
        field: &Field,
        i: Local,
    ) -> Local {
        let bits = self.emit_field_bits(e, words, block_bit, field, i);
        emit_encoded_number(e, field.encoding, field.width(), bits)
    }

    /// The `Scale * SubScale` and `Min * SubMin` factors covering in-block index `i` (dquant.md's
    /// value formula): 542a's block-32 formats have no `SubScale`/`SubMin`, so `i` reads `Scale`/`Min`
    /// at whatever index the caller holds (their digits are all stride 0, one factor per whole
    /// block); 542b's K-quants have `SubScale`/`SubMin` digits with a nonzero stride over `i`'s
    /// sub-block digits, so the real `i` (never a hoisted 0) must reach them to select the right
    /// sub-block's pair (`format::Field` docs: "a per-sub-block operand is indexed by the element
    /// index").
    pub(crate) fn emit_factors(
        &self,
        e: &mut Emit,
        words: Local,
        block: Local,
        i: Local,
    ) -> Factors {
        let block_bit = self.block_base_bit(e, block);
        let scale_whole = self
            .scale
            .as_ref()
            .map(|field| self.emit_field_number(e, words, block_bit, field, i));
        let scale_sub = self
            .sub_scale
            .as_ref()
            .map(|field| self.emit_field_number(e, words, block_bit, field, i));
        let scale = combine_product(e, scale_whole, scale_sub);
        let min_whole = self
            .min
            .as_ref()
            .map(|field| self.emit_field_number(e, words, block_bit, field, i));
        let min_sub = self
            .sub_min
            .as_ref()
            .map(|field| self.emit_field_number(e, words, block_bit, field, i));
        let min = combine_product(e, min_whole, min_sub);
        Factors { scale, min }
    }

    /// One logical value at in-block index `i` of block `block`, combined with `factors` through the
    /// value formula (`(Scale * SubScale) * value (+|-) (Min * SubMin)`, dquant.md 3.3; `factors`
    /// already folds `SubScale`/`SubMin` in where 542b's K-quants have them). The combining add/sub is
    /// `Rvalue::BinaryOpNoContract` (card 628): `Min` exists for `Q4_1`/`Q5_1` and every K-quant with a
    /// `dmin` (Q2_K/Q4_K/Q5_K), exactly the formats an FMA-contracting backend could otherwise fuse
    /// into a different rounding than the CPU oracle's separate multiply and add/sub (dquant.md 7, R1).
    pub(crate) fn emit_value(
        &self,
        e: &mut Emit,
        words: Local,
        block: Local,
        i: Local,
        factors: &Factors,
    ) -> Local {
        let block_bit = self.block_base_bit(e, block);
        let coded_bits = self.emit_field_bits(e, words, block_bit, &self.codes, i);
        let value = match self.value {
            ValueMap::Integer => {
                emit_encoded_number(e, self.codes.encoding, self.codes.width(), coded_bits)
            }
            ValueMap::Codebook(table) => emit_codebook(e, coded_bits, table),
            ValueMap::Float => {
                let FieldEncoding::Float(format) = self.codes.encoding else {
                    unreachable!("format tests: a Float value map has a Float-encoded Codes field")
                };
                emit_float_decode(e, format, coded_bits)
            }
        };
        let scaled = match factors.scale {
            Some(scale) => e.let_(
                Ty::F32,
                Rvalue::BinaryOp(BinOp::Mul, copy(scale), copy(value)),
            ),
            None => value,
        };
        match (factors.min, self.min_sign) {
            (Some(min), Some(MinSign::Add)) => e.let_(
                Ty::F32,
                Rvalue::BinaryOpNoContract(BinOp::Add, copy(scaled), copy(min)),
            ),
            (Some(min), Some(MinSign::Subtract)) => e.let_(
                Ty::F32,
                Rvalue::BinaryOpNoContract(BinOp::Sub, copy(scaled), copy(min)),
            ),
            _ => scaled,
        }
    }
}

/// `lhs * rhs` when both factors are present, else whichever is present, else `None` (the KIR
/// counterpart of `poot_quant::decode`'s scalar `product`, 542b's `Scale * SubScale`/`Min * SubMin`).
fn combine_product(e: &mut Emit, lhs: Option<Local>, rhs: Option<Local>) -> Option<Local> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(bin(e, Ty::F32, BinOp::Mul, copy(lhs), copy(rhs))),
        (Some(lhs), None) => Some(lhs),
        (None, Some(rhs)) => Some(rhs),
        (None, None) => None,
    }
}

/// A plain number (never the `Codes` field's [`ValueMap`]) from its raw bits: `Unsigned`/`Signed`
/// through an `i32`/f32 cast, `Float` through [`emit_float_decode`].
fn emit_encoded_number(e: &mut Emit, encoding: FieldEncoding, width: u32, bits: Local) -> Local {
    match encoding {
        FieldEncoding::Unsigned { offset } => {
            let signed = e.let_(
                Ty::I32,
                Rvalue::Bitcast {
                    to: Ty::I32,
                    operand: copy(bits),
                },
            );
            let offset_applied = bin(
                e,
                Ty::I32,
                BinOp::Add,
                copy(signed),
                poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::I32(offset)),
            );
            e.let_(
                Ty::F32,
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(offset_applied),
                },
            )
        }
        FieldEncoding::Signed => {
            let unused =
                poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::I32((32 - width) as i32));
            let signed = e.let_(
                Ty::I32,
                Rvalue::Bitcast {
                    to: Ty::I32,
                    operand: copy(bits),
                },
            );
            let shifted_up = e.let_(
                Ty::I32,
                Rvalue::BinaryOp(BinOp::Shl, copy(signed), unused.clone()),
            );
            let extended = e.let_(
                Ty::I32,
                Rvalue::BinaryOp(BinOp::Shr, copy(shifted_up), unused),
            );
            e.let_(
                Ty::F32,
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(extended),
                },
            )
        }
        FieldEncoding::Float(format) => emit_float_decode(e, format, bits),
    }
}

/// `table[bits & 0xf]` (a 16-entry codebook, `IQ4_NL`/`IQ4_XS`): a depth-4 binary select tree over
/// `bits`' low 4 bits, exact (the table constants are stored bit for bit, `select_u32`'s XOR-mask
/// choosing between them), no LDS table and no data-dependent branch.
fn emit_codebook(e: &mut Emit, bits: Local, table: &'static [f32; 16]) -> Local {
    let bit_n = |e: &mut Emit, n: u32| -> Local {
        let masked = bin(e, Ty::U32, BinOp::BitAnd, copy(bits), c32(1 << n));
        bin(e, Ty::Bool, BinOp::Ne, copy(masked), c32(0))
    };
    let mut level: Vec<Local> = table
        .iter()
        .map(|&value| e.let_(Ty::U32, Rvalue::Use(c32(value.to_bits()))))
        .collect();
    for n in 0..4 {
        let selector = bit_n(e, n);
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            next.push(select_u32(e, selector, pair[1], pair[0]));
        }
        level = next;
    }
    let result_bits = level[0];
    e.let_(
        Ty::F32,
        Rvalue::Bitcast {
            to: Ty::F32,
            operand: copy(result_bits),
        },
    )
}
