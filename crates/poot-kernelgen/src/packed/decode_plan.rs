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
    Segment, Storage, WeightFormat,
};

use super::family::{Codes, Family, Position};
use super::float_decode::emit_float_decode;
use super::kernel::emit_block_address;
use super::source::{PieceAlignment, Source};
use crate::KernelGenError;
use crate::emit::{Emit, bin, c32, copy, cu};
use poot_target::RoundingConversions;

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
    /// The fields read per element besides `Codes`: the family's ([`Family::operands`]), in that order, then
    /// `Shift` when the format has one.
    extras: Vec<Field>,
    /// The `Shift` field, which is the last of [`Self::extras`] when the format has one.
    shift: Option<Field>,
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
        let family = Family::of(descriptor.value, &codes.field).ok_or(reject(
            format,
            "a float value map over a Codes field that is not float-encoded",
        ))?;
        let mut extras = family
            .operands()
            .iter()
            .map(|&role| {
                layout
                    .field(role)
                    .map(|field| field.field)
                    .ok_or(reject(format, "a value family's operand is missing"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shift = layout.field(OperandRole::Shift).map(|f| f.field);
        extras.extend(shift);
        Ok(Self {
            layout,
            codes: codes.field,
            scale: layout.field(OperandRole::Scale).map(|f| f.field),
            sub_scale: layout.field(OperandRole::SubScale).map(|f| f.field),
            min: layout.field(OperandRole::Min).map(|f| f.field),
            sub_min: layout.field(OperandRole::SubMin).map(|f| f.field),
            family,
            extras,
            shift,
            min_sign: descriptor.min_sign,
        })
    }

    /// This plan decoding its values through what the device keeps ([`Family::decoding_on`]): a value
    /// that meets a scale takes the cheaper decode, which leaves the scale a power of two to absorb
    /// ([`Self::emit_factors`] does, once per factor run).
    pub(crate) fn on(mut self, conversions: RoundingConversions) -> Self {
        if self.scale.is_some() || self.sub_scale.is_some() {
            self.family = self.family.decoding_on(conversions);
        }
        self
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
    /// digits all have stride 0, or that has none, covers the whole block), kept inside one segment
    /// ([`BitLayout::constant_run`]); the minimum over every factor piece, or the whole block when the format has
    /// no factor.
    pub(crate) fn factor_run(&self) -> usize {
        [self.scale, self.sub_scale, self.min, self.sub_min]
            .into_iter()
            .flatten()
            .flat_map(|field| field.pieces.iter())
            .map(|piece| piece.layout.constant_run() as usize)
            .min()
            .unwrap_or(self.layout.values)
    }

    fn block_base_bit(&self, e: &mut Emit, block: Local) -> Local {
        let block_bits = self.layout.bytes * 8;
        bin(e, Ty::Usize, BinOp::Mul, copy(block), cu(block_bits))
    }

    /// `layout.bit(i)` (see `format::BitLayout`), emitted for a runtime `i`: each segment's affine mixed-radix
    /// digit decomposition, least-significant digit first (the array is declared most-significant
    /// first, so this walks it in reverse - the same order `BitLayout::bit`'s own loop takes), and, for a layout
    /// of several segments, the one `i` falls in ([`emit_piecewise`]).
    fn emit_bit_layout(&self, e: &mut Emit, base: Local, layout: &BitLayout, i: Local) -> Local {
        emit_piecewise(e, layout.segments, i, |e, segment, relative| {
            let mut bit = bin(
                e,
                Ty::Usize,
                BinOp::Add,
                copy(base),
                cu(segment.base as usize),
            );
            let mut remaining = relative;
            for digit in segment.digits.iter().rev() {
                let extent = digit.extent as usize;
                let stride = digit.stride as usize;
                let this_digit = bin(e, Ty::Usize, BinOp::Rem, copy(remaining), cu(extent));
                let contribution = bin(e, Ty::Usize, BinOp::Mul, copy(this_digit), cu(stride));
                bit = bin(e, Ty::Usize, BinOp::Add, copy(bit), copy(contribution));
                remaining = bin(e, Ty::Usize, BinOp::Div, copy(remaining), cu(extent));
            }
            bit
        })
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
            bits = Some(append_piece(e, bits, piece_bits, below));
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
    /// below `k_total`: [`Self::emit_run_load`] then [`Self::emit_run_decode`] of what it loaded.
    pub(crate) fn emit_run(
        &self,
        e: &mut Emit,
        src: Source,
        row: Local,
        k0: Local,
        count: usize,
        k_total: Local,
    ) -> Vec<Local> {
        let loaded = self.emit_run_load(e, src, row, k0, count, k_total);
        self.emit_run_decode(e, src, (row, k0), count, k_total, &loaded)
    }

    /// Whether a run of `count` elements starting at a multiple of `count` shares its block address, its factors and
    /// its code windows: `count` divides the block (the run stays inside one block, addressed once) and the factor
    /// run (one [`Factors`] serves every element).
    fn shares_run(&self, count: usize) -> bool {
        self.values().is_multiple_of(count) && self.factor_run().is_multiple_of(count)
    }

    /// The loads of the run [`Self::emit_run`] decodes, apart from its decode. A run that [`Self::shares_run`]
    /// addresses its block once, loads and decodes the run's factors once, and reads the code and sign bits through
    /// shared 32-bit windows ([`Self::read_field_run`]); the locals it returns are the block and first in-block
    /// index, the factors that exist (scale, then min) and the windows of the codes field then of each extra field,
    /// the state [`Self::emit_run_decode`] cuts into values. Any other run addresses each element's block and
    /// factors on its own, where the element is decoded, and loads nothing here.
    pub(crate) fn emit_run_load(
        &self,
        e: &mut Emit,
        src: Source,
        row: Local,
        k0: Local,
        count: usize,
        k_total: Local,
    ) -> Vec<Local> {
        if !self.shares_run(count) {
            return Vec::new();
        }
        let (block, first) = emit_block_address(e, self.values(), row, k0, k_total);
        let factors = self.emit_factors(e, src, block, first);
        let block_bit = self.block_base_bit(e, block);
        let mut loaded = vec![block, first];
        loaded.extend(factors.scale);
        loaded.extend(factors.min);
        for field in std::iter::once(&self.codes).chain(&self.extras) {
            loaded.extend(self.read_field_run(e, src, block_bit, field, first, count));
        }
        loaded
    }

    /// The `count` values of the run [`Self::emit_run_load`] loaded as `loaded` (empty for a run that is not
    /// shared), each combined with the run's factors through the value formula.
    pub(crate) fn emit_run_decode(
        &self,
        e: &mut Emit,
        src: Source,
        (row, k0): (Local, Local),
        count: usize,
        k_total: Local,
        loaded: &[Local],
    ) -> Vec<Local> {
        let values = self.values();
        if !self.shares_run(count) {
            let run_block = values
                .is_multiple_of(count)
                .then(|| emit_block_address(e, values, row, k0, k_total));
            return (0..count)
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
                    let factors = self.emit_factors(e, src, block, i_in_block);
                    self.emit_value(e, src, block, i_in_block, &factors)
                })
                .collect();
        }
        let (block, first) = (loaded[0], loaded[1]);
        let mut rest = loaded[2..].iter().copied();
        let has_scale = self.scale.is_some() || self.sub_scale.is_some();
        let has_min = self.min.is_some() || self.sub_min.is_some();
        let factors = Factors {
            scale: has_scale.then(|| rest.next().expect("the loaded run has its scale")),
            min: has_min.then(|| rest.next().expect("the loaded run has its min")),
        };
        let windows: Vec<Local> = rest.collect();
        let mut windows = windows.as_slice();
        // Only a field read element by element addresses its block's bits.
        let fields = || std::iter::once(&self.codes).chain(&self.extras);
        let block_bit = fields()
            .any(|field| field_run_deltas(field, count).is_none())
            .then(|| self.block_base_bit(e, block));
        let mut take_field = |e: &mut Emit, field: &Field| {
            let used = field_run_windows(field, count);
            let (mine, after) = windows.split_at(used);
            windows = after;
            self.extract_field_run(e, src, block_bit, field, first, count, mine)
        };
        let codes = take_field(e, &self.codes);
        let extras: Vec<Vec<Local>> = self
            .extras
            .iter()
            .map(|field| take_field(e, field))
            .collect();
        let lanes = self.family.emit_run_lanes(e, first, count);
        (0..count)
            .map(|j| {
                let index = bin(e, Ty::Usize, BinOp::Add, copy(first), cu(j));
                let extra_bits: Vec<Local> = extras.iter().map(|run| run[j]).collect();
                self.emit_value_of_bits(
                    e,
                    &factors,
                    (codes[j], &extra_bits),
                    Position {
                        index,
                        known: Some((j, count)),
                        lane: lanes.as_ref().map(|lanes| lanes[j]),
                    },
                )
            })
            .collect()
    }

    /// The windows `field`'s raw bits for the `count` in-block indices `first + j` are cut from, `first` a multiple
    /// of `count` and `count` dividing the block: when every piece's bit layout moves by a fixed amount per `j`
    /// ([`run_deltas`]), each piece's run is read once through shared 32-bit windows ([`read_piece_run`]), the
    /// pieces' windows in order ([`field_run_windows`] of them). Empty for a field whose layout does not factor
    /// that way for this `count`, which [`Self::extract_field_run`] reads element by element.
    fn read_field_run(
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
        let Some(pieces) = field_run_deltas(field, count) else {
            return Vec::new();
        };
        field
            .pieces
            .iter()
            .zip(&pieces)
            .flat_map(|(piece, deltas)| {
                let first_bit = self.emit_bit_layout(e, block_bit, &piece.layout, first);
                read_piece_run(e, src, first_bit, (self.layout, piece, count), deltas)
            })
            .collect()
    }

    /// `field`'s raw bits for the `count` in-block indices `first + j` of the block at bit `block_bit`, cut from the
    /// `windows` [`Self::read_field_run`] loaded (none, for a field read element by element as
    /// [`Self::emit_field_bits`] does, which needs `block_bit`; `None` when no field of the run does).
    #[expect(clippy::too_many_arguments)] // the source and the run's address, with the windows read for it
    fn extract_field_run(
        &self,
        e: &mut Emit,
        src: Source,
        block_bit: Option<Local>,
        field: &Field,
        first: Local,
        count: usize,
        windows: &[Local],
    ) -> Vec<Local> {
        let Some(pieces) = field_run_deltas(field, count) else {
            let block_bit = block_bit.expect("a field read element by element has its block's bit");
            if let Some(offsets) = piecewise_field_offsets(field, count) {
                return self.read_piecewise_run(e, src, block_bit, (field, first, count), &offsets);
            }
            return (0..count)
                .map(|j| {
                    let i = bin(e, Ty::Usize, BinOp::Add, copy(first), cu(j));
                    self.emit_field_bits(e, src, block_bit, field, i)
                })
                .collect();
        };
        let mut windows = windows;
        let pieces: Vec<Vec<Local>> = field
            .pieces
            .iter()
            .zip(&pieces)
            .map(|(piece, deltas)| {
                let (mine, after) = windows.split_at(piece_run_windows(deltas, piece.width));
                windows = after;
                extract_piece_run(e, piece.width as usize, deltas, mine)
            })
            .collect();
        (0..count)
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
            .collect()
    }

    /// `field`'s raw bits for the `count` indices `first + j` of a run in a block at bit `block_bit`, when its layouts
    /// are piecewise and each segment moves the run by constant offsets, though not the same ones in every
    /// segment (`offsets`: [`piecewise_field_offsets`]): each piece's first bit is addressed once, the segment the run
    /// is in is selected once ([`emit_segment_offsets`]), and every element reads at that bit plus its offset, where
    /// [`Self::emit_field_bits`] would address each element through every segment of every layout.
    fn read_piecewise_run(
        &self,
        e: &mut Emit,
        src: Source,
        block_bit: Local,
        run: (&Field, Local, usize),
        offsets: &[Vec<Vec<usize>>],
    ) -> Vec<Local> {
        let (field, first, count) = run;
        let addressed: Vec<(Local, Vec<RunOffset>)> = field
            .pieces
            .iter()
            .zip(offsets)
            .map(|(piece, offsets)| {
                let first_bit = self.emit_bit_layout(e, block_bit, &piece.layout, first);
                let run_offsets = emit_segment_offsets(e, piece.layout.segments, first, offsets);
                (first_bit, run_offsets)
            })
            .collect();
        (0..count)
            .map(|j| {
                let mut bits: Option<Local> = None;
                let mut below = 0;
                for (piece, (first_bit, run_offsets)) in field.pieces.iter().zip(&addressed) {
                    let bit = run_offsets[j].after(e, *first_bit);
                    let piece_bits = src.read_bits(e, bit, piece.width);
                    bits = Some(append_piece(e, bits, piece_bits, below));
                    below += piece.width;
                }
                bits.expect("a field has at least one piece")
            })
            .collect()
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
        let scale = combine_product(e, scale_whole, scale_sub)
            .map(|scale| self.family.fold_scale(e, scale));
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
                lane: None,
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
        let (family_bits, shift_bits) =
            extra_bits.split_at(extra_bits.len() - usize::from(self.shift.is_some()));
        let value = self.family.emit_value(e, codes, family_bits, at);
        // `value + Shift` before the scale multiplies it (`dl * (grid + delta)`): the sum is exact, and the
        // product that follows takes it as a sum, so no backend can fuse the two into one rounding.
        let value = match (shift_bits, self.shift) {
            ([bits], Some(field)) => {
                let shift = emit_encoded_number(e, field.encoding, field.width(), *bits);
                e.let_(
                    Ty::F32,
                    Rvalue::BinaryOp(BinOp::Add, copy(value), copy(shift)),
                )
            }
            _ => value,
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

/// The bit offset of element `j` of an aligned run of `count` elements from the offset of the run's first
/// element, for each `j`: `Some` when every segment of `layout` gives the same offsets
/// ([`segment_run_offsets`]). `None` when `count` does not factor over the digits (a three-element run over a
/// power-of-two layout), a run could straddle two segments, or the segments move a run differently, which
/// [`DecodePlan::extract_field_run`] then reads element by element.
fn run_deltas(layout: &BitLayout, count: usize) -> Option<Vec<usize>> {
    let by_segment = segment_run_offsets(layout.segments, count)?;
    let (first, rest) = by_segment.split_first()?;
    rest.iter()
        .all(|deltas| deltas == first)
        .then(|| first.clone())
}

/// The offset of element `j` of an aligned run of `count` elements from the run's first element, for each `j`, in
/// each segment of a piecewise map: `Some` when every segment starts on a multiple of `count` (no run straddles two)
/// and its digits split the run cleanly ([`segment_run_deltas`]). The segments need not agree: an aligned run lies
/// in one of them, which [`emit_segment_offsets`] picks once per run.
pub(super) fn segment_run_offsets(segments: &[Segment], count: usize) -> Option<Vec<Vec<usize>>> {
    segments
        .iter()
        .map(|segment| {
            (segment.first as usize)
                .is_multiple_of(count)
                .then(|| segment_run_deltas(segment, count))
                .flatten()
        })
        .collect()
}

/// [`run_deltas`] of one segment: a digit whose extent `count` divides, where `j` moves that digit alone, or a
/// digit whose extent divides what is left of `count`, which `j` fills before the next digit moves. The run
/// starts at a multiple of `count`, so no carry leaves the digits consumed.
fn segment_run_deltas(segment: &Segment, count: usize) -> Option<Vec<usize>> {
    let mut deltas = vec![0usize; count];
    let mut remaining = count;
    let mut below = 1usize;
    for digit in segment.digits.iter().rev() {
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

/// A piece's bit offsets from a run's first value, as [`run_deltas`] gives them for each piece of `field`, or `None`
/// when any piece does not factor over a run of `count`.
fn field_run_deltas(field: &Field, count: usize) -> Option<Vec<Vec<usize>>> {
    field
        .pieces
        .iter()
        .map(|piece| run_deltas(&piece.layout, count))
        .collect()
}

/// The offsets of each piece of `field` over a run of `count`, per segment ([`segment_run_offsets`]), for a field
/// with a piecewise piece: `Some` when every piece's segments each move the run by constants. The pieces of a field
/// whose layouts are each one segment never take this path.
fn piecewise_field_offsets(field: &Field, count: usize) -> Option<Vec<Vec<Vec<usize>>>> {
    if field
        .pieces
        .iter()
        .all(|piece| piece.layout.segments.len() == 1)
    {
        return None;
    }
    field
        .pieces
        .iter()
        .map(|piece| segment_run_offsets(piece.layout.segments, count))
        .collect()
}

/// The 32-bit windows [`read_piece_run`] reads for every piece of `field` over a run of `count`, none when the field
/// does not factor ([`field_run_deltas`]).
fn field_run_windows(field: &Field, count: usize) -> usize {
    field_run_deltas(field, count).map_or(0, |pieces| {
        field
            .pieces
            .iter()
            .zip(&pieces)
            .map(|(piece, deltas)| piece_run_windows(deltas, piece.width))
            .sum()
    })
}

/// One stretch of a piece's run offsets read together (see [`stretches`]): its first offset, the offsets in it and
/// the bits it spans.
struct Stretch<'a> {
    start: usize,
    offsets: &'a [usize],
    span: usize,
}

/// The `width`-bit offsets `deltas[j]` split into stretches that are read once each, through shared 32-bit windows.
/// A stretch ends where the next offset is a whole word past the last value's end, or more than [`STRETCH_BITS`] past
/// the stretch's start, so a run whose elements sit far apart reads each part separately and never a word nothing
/// uses. `distinct` is the offsets sorted and deduplicated: elements at the same offset (a broadcast field) share
/// one extraction.
fn stretches(distinct: &[usize], width: usize) -> Vec<Stretch<'_>> {
    let mut stretches = Vec::new();
    let mut at = 0;
    while at < distinct.len() {
        let start = distinct[at];
        let end = (at + 1..distinct.len())
            .find(|&next| {
                distinct[next] - start > STRETCH_BITS
                    || distinct[next] - distinct[next - 1] >= 32 + width
            })
            .unwrap_or(distinct.len());
        let offsets = &distinct[at..end];
        stretches.push(Stretch {
            start,
            offsets,
            span: offsets[offsets.len() - 1] - start + width,
        });
        at = end;
    }
    stretches
}

/// `deltas` sorted and deduplicated.
fn distinct_offsets(deltas: &[usize]) -> Vec<usize> {
    let mut distinct = deltas.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    distinct
}

/// The windows [`read_piece_run`] reads for a piece of `width` bits at offsets `deltas`.
fn piece_run_windows(deltas: &[usize], width: u32) -> usize {
    stretches(&distinct_offsets(deltas), width as usize)
        .iter()
        .map(|stretch| stretch.span.div_ceil(32))
        .sum()
}

/// The 32-bit windows covering the `width`-bit values at bit `first_bit + deltas[j]` for each `j`, each stretch of
/// nearby offsets read once ([`Source::read_run`]), the stretches' windows in order. `run` is the block layout, the
/// piece and the run length whose starts `first_bit` ranges over ([`PieceAlignment::of`]).
fn read_piece_run(
    e: &mut Emit,
    src: Source,
    first_bit: Local,
    run: (BlockLayout, &Piece, usize),
    deltas: &[usize],
) -> Vec<Local> {
    let (layout, piece, count) = run;
    let distinct = distinct_offsets(deltas);
    stretches(&distinct, piece.width as usize)
        .iter()
        .flat_map(|stretch| {
            let bit = if stretch.start == 0 {
                first_bit
            } else {
                bin(e, Ty::Usize, BinOp::Add, copy(first_bit), cu(stretch.start))
            };
            let align = PieceAlignment::of(layout, piece, count, stretch.start);
            src.read_run(e, bit, stretch.span, align)
        })
        .collect()
}

/// The `width`-bit values at the offsets `deltas[j]`, cut with constant shifts out of the `windows` that
/// [`read_piece_run`] read for them.
fn extract_piece_run(
    e: &mut Emit,
    width: usize,
    deltas: &[usize],
    windows: &[Local],
) -> Vec<Local> {
    let distinct = distinct_offsets(deltas);
    let mut windows = windows;
    let mut values: Vec<(usize, Local)> = Vec::with_capacity(distinct.len());
    for stretch in stretches(&distinct, width) {
        let (mine, after) = windows.split_at(stretch.span.div_ceil(32));
        windows = after;
        for &delta in stretch.offsets {
            let (window, shift) = (
                (delta - stretch.start) / 32,
                ((delta - stretch.start) % 32) as u32,
            );
            let low = bin(e, Ty::U32, BinOp::Shr, copy(mine[window]), c32(shift));
            let joined = if shift as usize + width > 32 {
                let high = bin(
                    e,
                    Ty::U32,
                    BinOp::Shl,
                    copy(mine[window + 1]),
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

/// One element's offset from the first element of an aligned run, in a piecewise map: the same in every segment,
/// or the constant of the segment the run falls in.
#[derive(Clone, Copy)]
pub(super) enum RunOffset {
    /// The offset every segment gives.
    Fixed(usize),
    /// The offset of the segment the run's first element is in, selected at run time.
    Chosen(Local),
}

impl RunOffset {
    /// `base` plus the offset, for a `base` at the run's first element.
    pub(super) fn after(self, e: &mut Emit, base: Local) -> Local {
        match self {
            Self::Fixed(0) => base,
            Self::Fixed(offset) => bin(e, Ty::Usize, BinOp::Add, copy(base), cu(offset)),
            Self::Chosen(offset) => bin(e, Ty::Usize, BinOp::Add, copy(base), copy(offset)),
        }
    }
}

/// The offsets of an aligned run's elements from its first element `first`, in a piecewise map whose segments move
/// a run by `offsets[segment][j]` ([`segment_run_offsets`]): the offset itself where every segment agrees, else the
/// segment's constant selected by comparing `first` with where each later segment starts ([`emit_piecewise`]'s
/// selection, made once for the run's elements and not for each of them).
pub(super) fn emit_segment_offsets(
    e: &mut Emit,
    segments: &[Segment],
    first: Local,
    offsets: &[Vec<usize>],
) -> Vec<RunOffset> {
    let count = offsets[0].len();
    let agrees = |j: usize| offsets.iter().all(|segment| segment[j] == offsets[0][j]);
    let before_segment: Vec<Local> = if (0..count).all(agrees) {
        Vec::new()
    } else {
        segments[1..]
            .iter()
            .map(|later| {
                bin(
                    e,
                    Ty::Bool,
                    BinOp::Lt,
                    copy(first),
                    cu(later.first as usize),
                )
            })
            .collect()
    };
    (0..count)
        .map(|j| {
            if agrees(j) {
                return RunOffset::Fixed(offsets[0][j]);
            }
            let last = offsets.len() - 1;
            let mut chosen = cu(offsets[last][j]);
            let mut local = None;
            for segment in (0..last).rev() {
                let picked = e.let_(
                    Ty::Usize,
                    Rvalue::Select {
                        condition: copy(before_segment[segment]),
                        if_true: cu(offsets[segment][j]),
                        if_false: chosen,
                    },
                );
                chosen = copy(picked);
                local = Some(picked);
            }
            RunOffset::Chosen(local.expect("segments that disagree are at least two"))
        })
        .collect()
}

/// A piecewise map's value at in-block index `i` (`Ty::Usize`): `value(e, segment, relative)` emits one
/// segment's value for the index relative to the segment's first element, and the segment `i` falls in is
/// selected by comparing `i` with where each later segment starts. A map of one segment, which starts at element
/// 0, is `value` over `i` itself: no subtraction, no comparison.
pub(super) fn emit_piecewise(
    e: &mut Emit,
    segments: &[Segment],
    i: Local,
    mut value: impl FnMut(&mut Emit, &Segment, Local) -> Local,
) -> Local {
    let mut chosen: Option<Local> = None;
    // An element belongs to the last segment that starts at or before it, so build the selection from the end.
    for (index, segment) in segments.iter().enumerate().rev() {
        let relative = if segment.first == 0 {
            i
        } else {
            // Wraps for an element before the segment, which the selection never takes.
            bin(
                e,
                Ty::Usize,
                BinOp::Sub,
                copy(i),
                cu(segment.first as usize),
            )
        };
        let here = value(e, segment, relative);
        chosen = Some(match chosen {
            None => here,
            Some(later) => {
                let next_first = segments[index + 1].first as usize;
                let before_next = bin(e, Ty::Bool, BinOp::Lt, copy(i), cu(next_first));
                e.let_(
                    Ty::Usize,
                    Rvalue::Select {
                        condition: copy(before_next),
                        if_true: copy(here),
                        if_false: copy(later),
                    },
                )
            }
        });
    }
    chosen.expect("format tests: a map has at least one segment")
}

/// The bits of a field read so far (`low`, `below` bits wide) with the next piece's bits above them: low bits first
/// (`format::Field` docs), a piece shifts up by the widths below it.
fn append_piece(e: &mut Emit, low: Option<Local>, piece_bits: Local, below: u32) -> Local {
    match low {
        None => piece_bits,
        Some(low) => {
            let shifted = bin(e, Ty::U32, BinOp::Shl, copy(piece_bits), c32(below));
            bin(e, Ty::U32, BinOp::BitOr, copy(low), copy(shifted))
        }
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
        FieldEncoding::SignedPowerOfTwo { exponent } => {
            // `(-1)^bits * 2^exponent` as the float with that sign bit and exponent
            // (`scalar::signed_power_of_two_to_f32`).
            let sign = bin(e, Ty::U32, BinOp::Shl, copy(bits), c32(31));
            let pattern = bin(
                e,
                Ty::U32,
                BinOp::BitOr,
                copy(sign),
                c32(((127 + exponent) as u32) << 23),
            );
            e.let_(
                Ty::F32,
                Rvalue::Bitcast {
                    to: Ty::F32,
                    operand: copy(pattern),
                },
            )
        }
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
    use poot_quant::format::{Digit, Piece, Segment};

    use super::*;
    use crate::helpers::{Alloc, elem, ld, local, slice_dtype, slice_f32};

    const fn digit(extent: u32, stride: u32) -> Digit {
        Digit { extent, stride }
    }

    /// Six elements, 5 bits apart: a count of 3 or 6 factors over the digit, a count of 4 does not.
    const SIX: BitLayout = BitLayout {
        segments: &[Segment {
            first: 0,
            base: 3,
            digits: &[digit(6, 5)],
        }],
    };
    /// A lattice-style broadcast: the eight lanes of a group share one offset (stride 0), groups 9 bits apart.
    const GROUPS: BitLayout = BitLayout {
        segments: &[Segment {
            first: 0,
            base: 1,
            digits: &[digit(4, 9), digit(8, 0)],
        }],
    };
    /// Two digits the run crosses: four elements 6 bits apart, then 24 bits per group of four.
    const NESTED: BitLayout = BitLayout {
        segments: &[Segment {
            first: 0,
            base: 0,
            digits: &[digit(4, 24), digit(4, 6)],
        }],
    };

    /// Sixteen elements 3 bits apart: with 6-bit values one of them (bit 27) reaches exactly one bit into the
    /// next window.
    const DENSE: BitLayout = BitLayout {
        segments: &[Segment {
            first: 0,
            base: 5,
            digits: &[digit(16, 3)],
        }],
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

    /// `TQ1_0`'s codes: 160 bytes-per-trit-lane elements 8 bits apart in 32 bytes, 80 in 16, 16 in 4, over three
    /// segments with their own digits.
    const THREE_STRETCHES: BitLayout = BitLayout {
        segments: &[
            Segment {
                first: 0,
                base: 0,
                digits: &[digit(5, 0), digit(32, 8)],
            },
            Segment {
                first: 160,
                base: 256,
                digits: &[digit(5, 0), digit(16, 8)],
            },
            Segment {
                first: 240,
                base: 384,
                digits: &[digit(4, 0), digit(4, 8)],
            },
        ],
    };

    /// A piecewise layout has shared run offsets only where every segment moves a run alike and no run straddles
    /// two: runs of 4 are 8 bits apart in all three stretches, runs of 8 are not (the last stretch has four
    /// bytes to a row), and a run of 32 would start at 240 mid-run.
    #[test]
    fn a_piecewise_layout_shares_run_offsets_only_where_every_segment_agrees() {
        assert_eq!(
            run_deltas(&THREE_STRETCHES, 4),
            Some(vec![0, 8, 16, 24]),
            "every stretch steps a byte per element"
        );
        assert_eq!(
            run_deltas(&THREE_STRETCHES, 8),
            None,
            "the last stretch wraps after four"
        );
        assert_eq!(
            run_deltas(&THREE_STRETCHES, 32),
            None,
            "a run starting at 240 straddles"
        );
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
    /// `read_field_run`/`extract_field_run` in the KIR interpreter.
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
        let source = Source::plain(source);
        let windows = plan.read_field_run(&mut e, source, block_bit, field, first, count);
        let bits = plan.extract_field_run(
            &mut e,
            source,
            Some(block_bit),
            field,
            first,
            count,
            &windows,
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

    /// `read_field_run`/`extract_field_run` equals a scalar read of every element: through the shared windows for a count that
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
    /// the shared windows). Mutation: skip the last piece in `emit_field_bits` or in
    /// `read_field_run`/`extract_field_run`; the matching rows go red.
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

    /// POOT-1541 SC-1 (kernel side): `IQ1_S` and `IQ1_M`, whose value is `scale * (grid + shift)` with the shift a
    /// one-bit `+/-1/8` field read per 32 or per eight values, decode in the emitted KIR as
    /// `poot_quant::plan::decode_blocks` does, bit for bit: element by element (`emit_value`) and as runs of 8, 16
    /// (a run the factors and windows are shared over) and 32 (one that is not: `IQ1_M`'s scale changes every 16).
    /// Mutation: emit `value - shift`, or drop the shift field from `DecodePlan::of`'s extras; both formats go
    /// red naming the run length.
    #[test]
    fn the_shift_formats_decode_in_the_emitted_kernel_as_the_plan_decodes() {
        use poot_quant::test_support::packed::every_field_blocks;

        let blocks = 4;
        for format in [WeightFormat::Iq1_S, WeightFormat::Iq1_M] {
            let descriptor = format.descriptor();
            let Storage::Blocks(layout) = descriptor.storage else {
                unreachable!("a GGUF block format")
            };
            let bytes = &every_field_blocks(format)[..blocks * layout.bytes];
            let mut want = vec![0.0f32; blocks * layout.values];
            poot_quant::plan::decode_blocks(format, bytes, &mut want).unwrap();
            let want: Vec<u32> = want.into_iter().map(f32::to_bits).collect();
            let plan = DecodePlan::of(&descriptor).unwrap();
            let k_total = blocks * layout.values;
            for count in [1, 8, 16, 32] {
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
                        let block = e.let_(Ty::Usize, Rvalue::Use(cu(first / layout.values)));
                        let element = e.let_(Ty::Usize, Rvalue::Use(cu(first % layout.values)));
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
                let mut body = Body::new("shifted", 2, al.locals, vec![block]);
                body.constant_tables = plan.constant_tables();
                let mut buffers = [
                    Buffer::from_u32s(&super::super::fixtures::pack_words(bytes)),
                    Buffer::from_f32s(&vec![0.0; k_total]),
                ];
                run(&body, workgroups_covering(&body, 1), &mut buffers).unwrap();
                let got: Vec<u32> = buffers[1]
                    .to_f32s()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect();
                assert_eq!(got, want, "{format:?} runs of {count}");
            }
        }
    }
}
