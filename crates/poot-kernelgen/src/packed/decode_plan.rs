//! `DecodePlan`: a [`FormatDescriptor`] evaluated for emission (dquant.md 3.2-3.3). 542a covers block
//! storage without sub-block factors (`SubScale`/`SubMin` absent): one `Scale` and, for `Q4_1`/`Q5_1`,
//! one `Min` value per whole block. 542b adds the K-quant super-blocks, whose `Scale`/`Min` are one
//! value per whole block and whose `SubScale`/`SubMin` are one value per sub-block (`format::Field`'s
//! digits index the sub-block through the element index `i`, never a separately hoisted index):
//! `scale = Scale * SubScale`, `min = Min * SubMin`, either factor absent when the format has no such
//! role (dquant.md's value formula). `DecodePlan` holds no layout of its own - every constant it uses
//! comes straight from the descriptor's `Field`/`BitLayout`/`Piece` data.

use poot_kernel_ir::{BinOp, ConstantTableDecl, Local, Rvalue, Ty};
use poot_quant::format::{
    BitLayout, BlockLayout, Field, FieldEncoding, FormatDescriptor, MinSign, OperandRole, Piece,
    Storage, WeightFormat,
};

use super::family::{Codes, Family, Position};
use super::float_decode::emit_float_decode;
use super::kernel::emit_block_address;
use super::source::{PieceAlignment, Source};
use crate::KernelGenError;
use crate::emit::{Emit, bin, c32, copy, cu};

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
    family: Family,
    /// The fields the family reads besides `Codes` ([`Family::operands`]), in that order.
    extras: Vec<Field>,
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
        let family = Family::of(descriptor.value);
        let extras = family
            .operands()
            .iter()
            .map(|&role| {
                layout
                    .field(role)
                    .map(|field| field.field)
                    .ok_or(reject(format, "a value family's operand is missing"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layout,
            codes: codes.field,
            scale: layout.field(OperandRole::Scale).map(|f| f.field),
            sub_scale: layout.field(OperandRole::SubScale).map(|f| f.field),
            min: layout.field(OperandRole::Min).map(|f| f.field),
            sub_min: layout.field(OperandRole::SubMin).map(|f| f.field),
            family,
            extras,
            min_sign: descriptor.min_sign,
        })
    }

    /// The constant tables the decode reads, for the kernel body to declare
    /// ([`poot_kernel_ir::Body::constant_tables`]): the value family's, if it has any.
    pub(crate) fn constant_tables(&self) -> Vec<ConstantTableDecl> {
        self.family.tables()
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
        src: Source,
        block_bit: Local,
        field: &Field,
        i: Local,
    ) -> Local {
        self.emit_pieces(e, block_bit, field, i, |e, bit, piece| {
            src.read_bits(e, bit, piece.width)
        })
    }

    /// `field`'s raw bits at in-block index `i`, each piece read by `read` at the bit offset the piece's layout
    /// gives (`read(e, bit, piece)`) and concatenated low bits first.
    fn emit_pieces(
        &self,
        e: &mut Emit,
        block_bit: Local,
        field: &Field,
        i: Local,
        mut read: impl FnMut(&mut Emit, Local, &Piece) -> Local,
    ) -> Local {
        assert!(
            !field.pieces.is_empty(),
            "format tests: a field has at least one piece"
        );
        let mut bits: Option<Local> = None;
        let mut below = 0;
        for piece in field.pieces {
            let bit = self.emit_bit_layout(e, block_bit, &piece.layout, i);
            let piece_bits = read(e, bit, piece);
            bits = Some(match bits {
                None => piece_bits,
                // Low bits first (`format::Field` docs): a piece shifts up by the widths below it.
                Some(low) => {
                    let shifted = bin(e, Ty::U32, BinOp::Shl, copy(piece_bits), c32(below));
                    bin(e, Ty::U32, BinOp::BitOr, copy(low), copy(shifted))
                }
            });
            below += piece.width;
        }
        bits.expect("a field has at least one piece")
    }

    /// `field`'s value at in-block index `i`, through its own [`FieldEncoding`] (never the `Codes`
    /// field's value family, which [`Self::emit_value`] applies separately): used for `Scale`/`Min`,
    /// which are always plain numbers (`Unsigned`, `Signed` or `Float`).
    fn emit_field_number(
        &self,
        e: &mut Emit,
        src: Source,
        block_bit: Local,
        field: &Field,
        i: Local,
    ) -> Local {
        // A factor field sits at a position the descriptor aligns, so each piece is one load of its own word.
        let bits = self.emit_pieces(e, block_bit, field, i, |e, bit, piece| {
            let align = PieceAlignment::of(self.layout, piece, 1, 0);
            src.read_field(e, bit, piece.width, align)
        });
        emit_encoded_number(e, field.encoding, field.width(), bits)
    }

    /// The `count` logical values `(row, k0 + j)` for `j in 0..count`, `k0` a multiple of `count` and every index
    /// below `k_total`. A run of `count` elements starting at a multiple of `count` stays inside one block when
    /// `count` divides the block, so the block is addressed once; when `count` also divides the factor run, the
    /// run's factors are decoded once and shared by all its elements, and the run's code and sign bits are read
    /// through shared 32-bit windows ([`Self::emit_field_run`]) rather than two word loads per element. Otherwise
    /// each element addresses its own block and decodes its own factors.
    pub(crate) fn emit_run(
        &self,
        e: &mut Emit,
        src: Source,
        row: Local,
        k0: Local,
        count: usize,
        k_total: Local,
    ) -> Vec<Local> {
        let values = self.values();
        let run_block = values
            .is_multiple_of(count)
            .then(|| emit_block_address(e, values, row, k0, k_total));
        let run_factors = run_block
            .filter(|_| self.factor_run().is_multiple_of(count))
            .map(|(block, first)| self.emit_factors(e, src, block, first));
        if let (Some((block, first)), Some(factors)) = (run_block, &run_factors) {
            let block_bit = self.block_base_bit(e, block);
            let codes = self.emit_field_run(e, src, block_bit, &self.codes, first, count);
            let extras: Vec<Vec<Local>> = self
                .extras
                .iter()
                .map(|field| self.emit_field_run(e, src, block_bit, field, first, count))
                .collect();
            return (0..count)
                .map(|j| {
                    let index = bin(e, Ty::Usize, BinOp::Add, copy(first), cu(j));
                    let extra_bits: Vec<Local> = extras.iter().map(|run| run[j]).collect();
                    self.emit_value_of_bits(
                        e,
                        factors,
                        (codes[j], &extra_bits),
                        Position {
                            index,
                            known: Some((j, count)),
                        },
                    )
                })
                .collect();
        }
        (0..count)
            .map(|j| {
                let (block, i_in_block) = match run_block {
                    Some((block, first)) => {
                        (block, bin(e, Ty::Usize, BinOp::Add, copy(first), cu(j)))
                    }
                    None => {
                        let k_index = bin(e, Ty::Usize, BinOp::Add, copy(k0), cu(j));
                        emit_block_address(e, values, row, k_index, k_total)
                    }
                };
                let own_factors;
                let factors = match &run_factors {
                    Some(factors) => factors,
                    None => {
                        own_factors = self.emit_factors(e, src, block, i_in_block);
                        &own_factors
                    }
                };
                self.emit_value(e, src, block, i_in_block, factors)
            })
            .collect()
    }

    /// `field`'s raw bits for the `count` in-block indices `first + j`, `first` a multiple of `count` and
    /// `count` dividing the block. When every piece's bit layout moves by a fixed amount per `j`
    /// ([`run_deltas`]), the run's bits are read once through shared 32-bit windows
    /// ([`Self::emit_piece_run`]); a field whose layout does not factor that way for this `count` is read element
    /// by element, as [`Self::emit_field_bits`] does.
    fn emit_field_run(
        &self,
        e: &mut Emit,
        src: Source,
        block_bit: Local,
        field: &Field,
        first: Local,
        count: usize,
    ) -> Vec<Local> {
        assert!(
            !field.pieces.is_empty(),
            "format tests: a field has at least one piece"
        );
        let pieces: Option<Vec<Vec<Local>>> = field
            .pieces
            .iter()
            .map(|piece| {
                let deltas = run_deltas(&piece.layout, count)?;
                let first_bit = self.emit_bit_layout(e, block_bit, &piece.layout, first);
                Some(emit_piece_run(
                    e,
                    src,
                    first_bit,
                    (self.layout, piece, count),
                    &deltas,
                ))
            })
            .collect();
        match pieces {
            Some(pieces) => (0..count)
                .map(|j| {
                    let mut bits = pieces[0][j];
                    let mut below = field.pieces[0].width;
                    for (run, piece) in pieces.iter().zip(field.pieces).skip(1) {
                        // Low bits first: a piece shifts up by the widths below it.
                        let shifted = bin(e, Ty::U32, BinOp::Shl, copy(run[j]), c32(below));
                        bits = bin(e, Ty::U32, BinOp::BitOr, copy(bits), copy(shifted));
                        below += piece.width;
                    }
                    bits
                })
                .collect(),
            None => (0..count)
                .map(|j| {
                    let i = bin(e, Ty::Usize, BinOp::Add, copy(first), cu(j));
                    self.emit_field_bits(e, src, block_bit, field, i)
                })
                .collect(),
        }
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
        src: Source,
        block: Local,
        i: Local,
    ) -> Factors {
        let block_bit = self.block_base_bit(e, block);
        let scale_whole = self
            .scale
            .as_ref()
            .map(|field| self.emit_field_number(e, src, block_bit, field, i));
        let scale_sub = self
            .sub_scale
            .as_ref()
            .map(|field| self.emit_field_number(e, src, block_bit, field, i));
        let scale = combine_product(e, scale_whole, scale_sub);
        let min_whole = self
            .min
            .as_ref()
            .map(|field| self.emit_field_number(e, src, block_bit, field, i));
        let min_sub = self
            .sub_min
            .as_ref()
            .map(|field| self.emit_field_number(e, src, block_bit, field, i));
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
        src: Source,
        block: Local,
        i: Local,
        factors: &Factors,
    ) -> Local {
        let block_bit = self.block_base_bit(e, block);
        let coded_bits = self.emit_field_bits(e, src, block_bit, &self.codes, i);
        let extra_bits: Vec<Local> = self
            .extras
            .iter()
            .map(|field| self.emit_field_bits(e, src, block_bit, field, i))
            .collect();
        self.emit_value_of_bits(
            e,
            factors,
            (coded_bits, &extra_bits),
            Position {
                index: i,
                known: None,
            },
        )
    }

    /// [`Self::emit_value`] over bits already read: the `Codes` field's `bits.0` and the family's other
    /// fields' `bits.1` ([`Family::operands`]), at the element `at`.
    fn emit_value_of_bits(
        &self,
        e: &mut Emit,
        factors: &Factors,
        bits: (Local, &[Local]),
        at: Position,
    ) -> Local {
        let (coded_bits, extra_bits) = bits;
        let codes = Codes {
            bits: coded_bits,
            field: self.codes,
        };
        let value = self.family.emit_value(e, codes, extra_bits, at);
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

/// The bit offset of element `j` of an aligned run of `count` elements from the offset of the run's first
/// element, for each `j`: `Some` when `layout`'s digits (least significant first) split the run cleanly, either
/// a digit whose extent `count` divides, where `j` moves that digit alone, or a digit whose extent divides what
/// is left of `count`, which `j` fills before the next digit moves. The run starts at a multiple of `count`, so
/// no carry leaves the digits consumed. `None` when `count` does not factor over the digits (a three-element
/// run over a power-of-two layout).
fn run_deltas(layout: &BitLayout, count: usize) -> Option<Vec<usize>> {
    let mut deltas = vec![0usize; count];
    let mut remaining = count;
    let mut below = 1usize;
    for digit in layout.digits.iter().rev() {
        if remaining == 1 {
            break;
        }
        let extent = digit.extent as usize;
        let stride = digit.stride as usize;
        if extent.is_multiple_of(remaining) {
            for (j, delta) in deltas.iter_mut().enumerate() {
                *delta += (j / below) * stride;
            }
            remaining = 1;
        } else if remaining.is_multiple_of(extent) {
            for (j, delta) in deltas.iter_mut().enumerate() {
                *delta += (j / below % extent) * stride;
            }
            below *= extent;
            remaining /= extent;
        } else {
            return None;
        }
    }
    (remaining == 1).then_some(deltas)
}

/// The most bits a stretch of one run read spans from its first value's offset: a 128-bit span is four words.
const STRETCH_BITS: usize = 128;

/// The `width`-bit values at bit `first_bit + deltas[j]` for each `j`, each stretch of nearby offsets read once
/// through shared 32-bit windows ([`Source::read_run`]) and cut into its values with constant shifts. `run` is the
/// block layout, the piece and the run length whose starts `first_bit` ranges over ([`PieceAlignment::of`]). A
/// stretch ends where the next offset is a whole word past the last value's end, or more than [`STRETCH_BITS`] past
/// the stretch's start, so a run whose elements sit far apart reads each part separately and never a word nothing
/// uses; elements at the same offset (a broadcast field) share one extraction.
fn emit_piece_run(
    e: &mut Emit,
    src: Source,
    first_bit: Local,
    run: (BlockLayout, &Piece, usize),
    deltas: &[usize],
) -> Vec<Local> {
    let (layout, piece, count) = run;
    let width = piece.width as usize;
    let mut distinct: Vec<usize> = deltas.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let mut values: Vec<(usize, Local)> = Vec::with_capacity(distinct.len());
    let mut at = 0;
    while at < distinct.len() {
        let start = distinct[at];
        let end = (at + 1..distinct.len())
            .find(|&next| {
                distinct[next] - start > STRETCH_BITS
                    || distinct[next] - distinct[next - 1] >= 32 + width
            })
            .unwrap_or(distinct.len());
        let stretch = &distinct[at..end];
        let span = stretch[stretch.len() - 1] - start + width;
        let bit = if start == 0 {
            first_bit
        } else {
            bin(e, Ty::Usize, BinOp::Add, copy(first_bit), cu(start))
        };
        let align = PieceAlignment::of(layout, piece, count, start);
        let windows = src.read_run(e, bit, span, align);
        for &delta in stretch {
            let (window, shift) = ((delta - start) / 32, ((delta - start) % 32) as u32);
            let low = bin(e, Ty::U32, BinOp::Shr, copy(windows[window]), c32(shift));
            let joined = if shift as usize + width > 32 {
                let high = bin(
                    e,
                    Ty::U32,
                    BinOp::Shl,
                    copy(windows[window + 1]),
                    c32(32 - shift),
                );
                bin(e, Ty::U32, BinOp::BitOr, copy(low), copy(high))
            } else {
                low
            };
            let value = if width == 32 {
                joined
            } else {
                bin(
                    e,
                    Ty::U32,
                    BinOp::BitAnd,
                    copy(joined),
                    c32((1u32 << width) - 1),
                )
            };
            values.push((delta, value));
        }
        at = end;
    }
    deltas
        .iter()
        .map(|delta| {
            values
                .iter()
                .find(|(offset, _)| offset == delta)
                .expect("every offset was read")
                .1
        })
        .collect()
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

/// A plain number (never the `Codes` field's value family) from its raw bits: `Unsigned`/`Signed`
/// through an `i32`/f32 cast, `Float` through [`emit_float_decode`].
pub(super) fn emit_encoded_number(
    e: &mut Emit,
    encoding: FieldEncoding,
    width: u32,
    bits: Local,
) -> Local {
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
        FieldEncoding::HalfStep { exponent } => {
            // `(bits + 1/2) * 2^exponent`, both steps exact (`scalar::half_step_to_f32`).
            let signed = e.let_(
                Ty::I32,
                Rvalue::Bitcast {
                    to: Ty::I32,
                    operand: copy(bits),
                },
            );
            let integer = e.let_(
                Ty::F32,
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(signed),
                },
            );
            let half = bin(
                e,
                Ty::F32,
                BinOp::Add,
                copy(integer),
                poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::F32(0.5)),
            );
            bin(
                e,
                Ty::F32,
                BinOp::Mul,
                copy(half),
                poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::F32(f32::from_bits(
                    ((127 + exponent) as u32) << 23,
                ))),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use poot_kernel_ir::interp::{Buffer, run};
    use poot_kernel_ir::test_support::workgroups_covering;
    use poot_kernel_ir::{BasicBlock, Body, Terminator};
    use poot_quant::format::{Digit, Piece};

    use super::*;
    use crate::helpers::{Alloc, elem, ld, local, slice_dtype, slice_f32};

    const fn digit(extent: u32, stride: u32) -> Digit {
        Digit { extent, stride }
    }

    /// Six elements, 5 bits apart: a count of 3 or 6 factors over the digit, a count of 4 does not.
    const SIX: BitLayout = BitLayout {
        base: 3,
        digits: &[digit(6, 5)],
    };
    /// A lattice-style broadcast: the eight lanes of a group share one offset (stride 0), groups 9 bits apart.
    const GROUPS: BitLayout = BitLayout {
        base: 1,
        digits: &[digit(4, 9), digit(8, 0)],
    };
    /// Two digits the run crosses: four elements 6 bits apart, then 24 bits per group of four.
    const NESTED: BitLayout = BitLayout {
        base: 0,
        digits: &[digit(4, 24), digit(4, 6)],
    };

    /// Sixteen elements 3 bits apart: with 6-bit values one of them (bit 27) reaches exactly one bit into the
    /// next window.
    const DENSE: BitLayout = BitLayout {
        base: 5,
        digits: &[digit(16, 3)],
    };

    #[test]
    fn a_run_moves_the_bit_offset_by_constants_over_the_digits() {
        assert_eq!(run_deltas(&SIX, 3), Some(vec![0, 5, 10]));
        assert_eq!(run_deltas(&SIX, 6), Some(vec![0, 5, 10, 15, 20, 25]));
        assert_eq!(run_deltas(&SIX, 1), Some(vec![0]));
        assert_eq!(run_deltas(&GROUPS, 8), Some(vec![0; 8]));
        assert_eq!(
            run_deltas(&NESTED, 8),
            Some(vec![0, 6, 12, 18, 24, 30, 36, 42])
        );
        assert_eq!(run_deltas(&NESTED, 2), Some(vec![0, 6]));
    }

    /// A count that neither divides nor is divided by the digit it meets does not factor: `None`, so the caller
    /// reads element by element. A run longer than the layout has no digit left for its tail either.
    #[test]
    fn a_count_that_does_not_factor_over_the_digits_has_no_constant_offsets() {
        assert_eq!(run_deltas(&SIX, 4), None);
        assert_eq!(run_deltas(&SIX, 12), None);
        assert_eq!(run_deltas(&NESTED, 32), None);
    }

    fn field(pieces: &'static [Piece]) -> Field {
        Field {
            pieces,
            encoding: FieldEncoding::Unsigned { offset: 0 },
        }
    }

    /// `width` bits at bit `at` of `words`, one bit at a time, zero past the last word.
    fn scalar_bits(words: &[u32], at: usize, width: u32) -> u32 {
        (0..width as usize).fold(0u32, |acc, k| {
            let bit = at + k;
            let set = words
                .get(bit / 32)
                .is_some_and(|word| word >> (bit % 32) & 1 == 1);
            acc | u32::from(set) << k
        })
    }

    /// The field's raw bits at the `count` indices `first + j` of a block starting at bit `block_bit`, through
    /// `emit_field_run` in the KIR interpreter.
    fn run_bits(
        words: &[u32],
        field: &Field,
        block_bit: usize,
        first: usize,
        count: usize,
    ) -> Vec<u32> {
        let plan = DecodePlan::of(&poot_quant::format::WeightFormat::Q8_0.descriptor()).unwrap();
        let mut al = Alloc::new(vec![
            ld(Ty::Unit, false),
            ld(slice_dtype(Ty::U32, false), false),
            ld(slice_f32(true), true),
        ]);
        let (source, out) = (local(1), local(2));
        let mut e = Emit::new(&mut al);
        let block_bit = e.let_(Ty::Usize, Rvalue::Use(cu(block_bit)));
        let first = e.let_(Ty::Usize, Rvalue::Use(cu(first)));
        let bits = plan.emit_field_run(
            &mut e,
            Source::plain(source),
            block_bit,
            field,
            first,
            count,
        );
        for (j, value) in bits.into_iter().enumerate() {
            let as_f32 = e.let_(
                Ty::F32,
                Rvalue::Bitcast {
                    to: Ty::F32,
                    operand: copy(value),
                },
            );
            let at = e.let_(Ty::Usize, Rvalue::Use(cu(j)));
            e.assign(elem(out, at), Rvalue::Use(copy(as_f32)));
        }
        let block = BasicBlock {
            statements: e.take(),
            terminator: Terminator::Return,
        };
        let body = Body::new("field_run_probe", 2, al.locals, vec![block]);
        let mut buffers = [
            Buffer::from_u32s(words),
            Buffer::from_f32s(&vec![0.0; count]),
        ];
        run(&body, workgroups_covering(&body, 1), &mut buffers).unwrap();
        buffers[1]
            .to_f32s()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn words() -> Vec<u32> {
        let mut state = 0x9e37_79b9_u32;
        let mut words: Vec<u32> = (0..48)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state
            })
            .collect();
        words.push(0); // the zero guard word
        words
    }

    /// `emit_field_run` equals a scalar read of every element: through the shared windows for a count that
    /// factors (blocks starting at bit 0, a word boundary, and 272, which is not one), through the element-by-
    /// element fallback for one that does not, for a broadcast field (one extraction shared by the group), for
    /// runs crossing two digits, and for a two-piece field whose pieces are joined low bits first.
    /// Mutation: `31 - shift` -> `30 - shift` in `emit_read_run`; the windowed rows go red.
    #[test]
    fn a_run_of_field_bits_equals_the_scalar_read_of_each_element() {
        static SIX_FIELD: [Piece; 1] = [Piece {
            layout: SIX,
            width: 5,
        }];
        static GROUP_FIELD: [Piece; 1] = [Piece {
            layout: GROUPS,
            width: 9,
        }];
        static NESTED_FIELD: [Piece; 1] = [Piece {
            layout: NESTED,
            width: 6,
        }];
        static DENSE_FIELD: [Piece; 1] = [Piece {
            layout: DENSE,
            width: 6,
        }];
        static TWO_PIECES: [Piece; 2] = [
            Piece {
                layout: SIX,
                width: 4,
            },
            Piece {
                layout: NESTED,
                width: 2,
            },
        ];
        let words = words();
        let cases: [(&Field, usize, usize); 10] = [
            (&field(&SIX_FIELD), 3, 3),
            (&field(&SIX_FIELD), 3, 0),
            (&field(&SIX_FIELD), 6, 0),
            (&field(&SIX_FIELD), 4, 0),
            (&field(&GROUP_FIELD), 8, 8),
            (&field(&NESTED_FIELD), 8, 8),
            (&field(&NESTED_FIELD), 2, 6),
            (&field(&TWO_PIECES), 3, 3),
            (&field(&TWO_PIECES), 1, 5),
            (&field(&DENSE_FIELD), 16, 0),
        ];
        for (field, count, first) in cases {
            for block_bit in [0, 64, 272, 31 * 32] {
                let got = run_bits(&words, field, block_bit, first, count);
                let want: Vec<u32> = (0..count)
                    .map(|j| {
                        let i = (first + j) as u32;
                        let mut value = 0;
                        let mut shift = 0;
                        for piece in field.pieces {
                            let at = block_bit + piece.layout.bit(i) as usize;
                            value |= scalar_bits(&words, at, piece.width) << shift;
                            shift += piece.width;
                        }
                        value
                    })
                    .collect();
                assert_eq!(
                    got, want,
                    "{count} elements from {first} of {field:?} in a block at bit {block_bit}"
                );
            }
        }
    }

    /// POOT-1539 SC-4: a field of four pieces decodes in the emitted KIR as direct arithmetic on the block's
    /// bytes does, bit for bit: element by element (`emit_value`, the four scale nibbles through
    /// `emit_field_bits`) and as runs of 4, 8 and 32 (`emit_run`, the four code bit planes through
    /// `emit_field_run`'s shared windows). Mutation: skip the last piece in `emit_field_bits` or in
    /// `emit_field_run`; the matching rows go red.
    #[test]
    fn a_field_of_four_pieces_decodes_in_the_emitted_kernel() {
        use poot_quant::test_support::synthetic::{
            FOUR_PIECE, four_piece_blocks, four_piece_reference,
        };

        let blocks = 6;
        let bytes = four_piece_blocks(blocks, 0xfeed_f00d);
        let want: Vec<u32> = bytes
            .as_chunks::<24>()
            .0
            .iter()
            .flat_map(|block| four_piece_reference(block))
            .map(f32::to_bits)
            .collect();
        let plan = DecodePlan::of(&FOUR_PIECE).unwrap();
        let k_total = blocks * 32;
        for count in [1, 4, 8, 32] {
            let mut al = Alloc::new(vec![
                ld(Ty::Unit, false),
                ld(slice_dtype(Ty::U32, false), false),
                ld(slice_f32(true), true),
            ]);
            let (words, out) = (local(1), local(2));
            let mut e = Emit::new(&mut al);
            let source = Source::plain(words);
            let row = e.let_(Ty::Usize, Rvalue::Use(cu(0)));
            let total = e.let_(Ty::Usize, Rvalue::Use(cu(k_total)));
            for first in (0..k_total).step_by(count) {
                let values = if count == 1 {
                    let block = e.let_(Ty::Usize, Rvalue::Use(cu(first / 32)));
                    let element = e.let_(Ty::Usize, Rvalue::Use(cu(first % 32)));
                    let factors = plan.emit_factors(&mut e, source, block, element);
                    vec![plan.emit_value(&mut e, source, block, element, &factors)]
                } else {
                    let k0 = e.let_(Ty::Usize, Rvalue::Use(cu(first)));
                    plan.emit_run(&mut e, source, row, k0, count, total)
                };
                for (j, value) in values.into_iter().enumerate() {
                    let at = e.let_(Ty::Usize, Rvalue::Use(cu(first + j)));
                    e.assign(elem(out, at), Rvalue::Use(copy(value)));
                }
            }
            let block = BasicBlock {
                statements: e.take(),
                terminator: Terminator::Return,
            };
            let body = Body::new("four_pieces", 2, al.locals, vec![block]);
            let mut buffers = [
                Buffer::from_u32s(&super::super::fixtures::pack_words(&bytes)),
                Buffer::from_f32s(&vec![0.0; k_total]),
            ];
            run(&body, workgroups_covering(&body, 1), &mut buffers).unwrap();
            let got: Vec<u32> = buffers[1]
                .to_f32s()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(got, want, "runs of {count}");
        }
    }
}
