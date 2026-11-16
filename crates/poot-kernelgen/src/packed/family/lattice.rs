//! The lattice family ([`poot_quant::families::lattice`]): `sign * cell[code * lanes + lane(i)]`.
//!
//! The cells are one constant table ([`VALUE_TABLE`]) of `u32` words, `32 / width` cells to a word. A
//! one-lane lattice indexes its cells by the code. A multi-lane lattice reads the word of the code's
//! point and then the lane's cell inside it; the lane is the descriptor's [`LaneMap`] of `i`, an
//! emission-time constant when the run the element belongs to fixes it. The cell's bits become a number through the table's encoding;
//! a sign rule flips the sign bit of the float by the lane's bit of the group's sign pattern.

use poot_kernel_ir::{BinOp, ConstantTableDecl, Local, Rvalue, Ty};
use poot_quant::families::lattice::{LaneMap, Lattice, SignRule};
use poot_quant::format::{FieldEncoding, OperandRole};

use super::Position;
use crate::emit::{Emit, bin, c32, copy, cu};
use crate::packed::decode_plan::{
    emit_encoded_number, emit_piecewise, emit_segment_offsets, segment_run_offsets,
};
use crate::packed::float_decode::emit_float_decode;

/// The one constant table a lattice kernel declares: the lattice's cell words, read as table 0.
const VALUE_TABLE: u8 = 0;

pub(super) fn tables(lattice: &Lattice) -> Vec<ConstantTableDecl> {
    vec![ConstantTableDecl {
        words: lattice.cells.words.to_vec(),
    }]
}

pub(super) fn operands(lattice: &Lattice) -> &'static [OperandRole] {
    match lattice.signs {
        Some(_) => &[OperandRole::Signs],
        None => &[],
    }
}

pub(super) fn emit_value(
    e: &mut Emit,
    lattice: &Lattice,
    code: Local,
    extras: &[Local],
    at: Position,
) -> Local {
    let cell_bits = if lattice.lanes == 1 {
        emit_single_lane_cell(e, lattice, code)
    } else {
        emit_lane_cell(e, lattice, code, at)
    };
    let magnitude = emit_cell_number(e, lattice, cell_bits);
    let Some(rule) = lattice.signs else {
        return magnitude;
    };
    let [sign_bits] = extras else {
        unreachable!("DecodePlan::of: a lattice with signs has a Signs field")
    };
    emit_signed(e, rule, magnitude, *sign_bits, at)
}

/// The bits of cell `code` of a one-lane lattice.
fn emit_single_lane_cell(e: &mut Emit, lattice: &Lattice, code: Local) -> Local {
    let width = lattice.cells.width;
    let index = e.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(code),
        },
    );
    if width == 32 {
        return e.let_(
            Ty::U32,
            Rvalue::ConstantTableRead {
                idx: copy(index),
                table: VALUE_TABLE,
            },
        );
    }
    let per_word = (32 / width) as usize;
    let word_index = bin(e, Ty::Usize, BinOp::Div, copy(index), cu(per_word));
    let cell = bin(e, Ty::Usize, BinOp::Rem, copy(index), cu(per_word));
    let shift = shift_of(e, copy(cell), width);
    read_cell(e, word_index, shift, width)
}

/// The bits of cell `lane(i)` of point `code` of a multi-lane lattice.
fn emit_lane_cell(e: &mut Emit, lattice: &Lattice, code: Local, at: Position) -> Local {
    let width = lattice.cells.width;
    let per_word = (32 / width) as usize;
    let words_per_point = lattice.lanes * width as usize / 32;
    let (lane_word, shift) = match at.known.and_then(|known| lattice.lane.fixed_by(known)) {
        Some(lane) => (cu(lane / per_word), c32((lane % per_word) as u32 * width)),
        None => {
            let lane = at
                .lane
                .unwrap_or_else(|| emit_lane(e, lattice.lane, at.index));
            let lane_word = bin(e, Ty::Usize, BinOp::Div, copy(lane), cu(per_word));
            let lane_cell = bin(e, Ty::Usize, BinOp::Rem, copy(lane), cu(per_word));
            (copy(lane_word), shift_of(e, copy(lane_cell), width))
        }
    };
    let code_index = e.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(code),
        },
    );
    let point_word = bin(
        e,
        Ty::Usize,
        BinOp::Mul,
        copy(code_index),
        cu(words_per_point),
    );
    let word_index = bin(e, Ty::Usize, BinOp::Add, copy(point_word), lane_word);
    read_cell(e, word_index, shift, width)
}

/// The lanes of the aligned run of `count` elements from `first`, for a lattice whose lane map is piecewise: the lane
/// of the first element once, and the offsets of the others from it, which the segment the run is in decides
/// ([`emit_segment_offsets`]). `None` for a map of one segment, whose lane is a constant of the position wherever
/// the run fixes it and is emitted for each element otherwise, and for a map whose segments do not move a run by
/// constants.
pub(super) fn emit_run_lanes(
    e: &mut Emit,
    lattice: &Lattice,
    first: Local,
    count: usize,
) -> Option<Vec<Local>> {
    let segments = lattice.lane.segments;
    if lattice.lanes == 1 || segments.len() == 1 {
        return None;
    }
    let offsets = segment_run_offsets(segments, count)?;
    let first_lane = emit_lane(e, lattice.lane, first);
    Some(
        emit_segment_offsets(e, segments, first, &offsets)
            .into_iter()
            .map(|offset| offset.after(e, first_lane))
            .collect(),
    )
}

/// The lane of element `index` at run time: each segment of the map is the sum of its digits' contributions
/// (a digit that does not move the lane contributes nothing and is not emitted; a stride of one is the digit
/// itself), and the segment `index` falls in is selected ([`emit_piecewise`]). A lane that cycles is `index %
/// lanes`.
fn emit_lane(e: &mut Emit, lane_map: LaneMap, index: Local) -> Local {
    emit_piecewise(e, lane_map.segments, index, |e, segment, relative| {
        let mut lane: Option<Local> = None;
        let mut remaining = relative;
        // Digits above the most significant one that moves the lane add nothing, so nothing is emitted for them.
        let first_moving = segment.digits.iter().position(|digit| digit.stride != 0);
        for (position, digit) in segment.digits.iter().enumerate().rev() {
            if first_moving.is_none_or(|first| position < first) {
                break;
            }
            let extent = digit.extent as usize;
            if digit.stride != 0 {
                let this_digit = bin(e, Ty::Usize, BinOp::Rem, copy(remaining), cu(extent));
                let contribution = match digit.stride {
                    1 => this_digit,
                    stride => bin(
                        e,
                        Ty::Usize,
                        BinOp::Mul,
                        copy(this_digit),
                        cu(stride as usize),
                    ),
                };
                lane = Some(match lane {
                    None => contribution,
                    Some(sum) => bin(e, Ty::Usize, BinOp::Add, copy(sum), copy(contribution)),
                });
            }
            if Some(position) != first_moving {
                remaining = bin(e, Ty::Usize, BinOp::Div, copy(remaining), cu(extent));
            }
        }
        match (lane, segment.base) {
            (Some(lane), 0) => lane,
            (Some(lane), base) => bin(e, Ty::Usize, BinOp::Add, copy(lane), cu(base as usize)),
            (None, base) => e.let_(Ty::Usize, Rvalue::Use(cu(base as usize))),
        }
    })
}

/// `cell * width` as a `u32` shift amount, for a runtime cell index within its word.
fn shift_of(e: &mut Emit, cell: poot_kernel_ir::Operand, width: u32) -> poot_kernel_ir::Operand {
    let cell = e.let_(
        Ty::U32,
        Rvalue::Cast {
            to: Ty::U32,
            operand: cell,
        },
    );
    copy(bin(e, Ty::U32, BinOp::Mul, copy(cell), c32(width)))
}

/// The `width` bits at `shift` of table word `word_index`.
fn read_cell(e: &mut Emit, word_index: Local, shift: poot_kernel_ir::Operand, width: u32) -> Local {
    let word = e.let_(
        Ty::U32,
        Rvalue::ConstantTableRead {
            idx: copy(word_index),
            table: VALUE_TABLE,
        },
    );
    let shifted = bin(e, Ty::U32, BinOp::Shr, copy(word), shift);
    bin(
        e,
        Ty::U32,
        BinOp::BitAnd,
        copy(shifted),
        c32((1u32 << width) - 1),
    )
}

/// `i % modulus` when the run the element belongs to fixes it at emission (`known`: `i % m == r` with
/// `modulus` dividing `m`).
fn constant_mod(at: Position, modulus: usize) -> Option<usize> {
    at.known
        .filter(|&(_, m)| m.is_multiple_of(modulus))
        .map(|(r, _)| r % modulus)
}

/// A cell's bits as the number its table's encoding gives: an unsigned integer widened to f32 (exact:
/// the cells are small), or a float of the table's format.
fn emit_cell_number(e: &mut Emit, lattice: &Lattice, bits: Local) -> Local {
    match lattice.cells.encoding {
        FieldEncoding::Unsigned { offset: 0 } => e.let_(
            Ty::F32,
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(bits),
            },
        ),
        encoding @ FieldEncoding::Unsigned { .. } => {
            emit_encoded_number(e, encoding, lattice.cells.width, bits)
        }
        FieldEncoding::Float(format) => emit_float_decode(e, format, bits),
        FieldEncoding::Signed
        | FieldEncoding::HalfStep { .. }
        | FieldEncoding::SignedPowerOfTwo { .. } => {
            unreachable!("format tests: a cell is an unsigned integer or a float")
        }
    }
}

/// `magnitude` with its sign flipped by the element's bit of the group's sign pattern: eight stored
/// bits, or seven plus their parity as bit 7 (`ksigns_iq2xs`). Exact: a sign-bit XOR.
fn emit_signed(
    e: &mut Emit,
    rule: SignRule,
    magnitude: Local,
    sign_bits: Local,
    at: Position,
) -> Local {
    let pattern = match rule {
        SignRule::Direct => sign_bits,
        SignRule::ParityOfSeven => {
            // The parity of seven bits, folded: `x ^ x >> 4`, then `^ >> 2`, then `^ >> 1`, bit 0.
            let mut folded = sign_bits;
            for shift in [4, 2, 1] {
                let high = bin(e, Ty::U32, BinOp::Shr, copy(folded), c32(shift));
                folded = bin(e, Ty::U32, BinOp::BitXor, copy(folded), copy(high));
            }
            let parity = bin(e, Ty::U32, BinOp::BitAnd, copy(folded), c32(1));
            let parity_bit = bin(e, Ty::U32, BinOp::Shl, copy(parity), c32(7));
            bin(e, Ty::U32, BinOp::BitOr, copy(sign_bits), copy(parity_bit))
        }
    };
    let group_shift = match constant_mod(at, 8) {
        Some(group_lane) => c32(group_lane as u32),
        None => {
            let group_lane = bin(e, Ty::Usize, BinOp::Rem, copy(at.index), cu(8));
            copy(e.let_(
                Ty::U32,
                Rvalue::Cast {
                    to: Ty::U32,
                    operand: copy(group_lane),
                },
            ))
        }
    };
    let selected = bin(e, Ty::U32, BinOp::Shr, copy(pattern), group_shift);
    let negative = bin(e, Ty::U32, BinOp::BitAnd, copy(selected), c32(1));
    let flip = bin(e, Ty::U32, BinOp::Shl, copy(negative), c32(31));
    let magnitude_raw = e.let_(
        Ty::U32,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: copy(magnitude),
        },
    );
    let signed_raw = bin(e, Ty::U32, BinOp::BitXor, copy(magnitude_raw), copy(flip));
    e.let_(
        Ty::F32,
        Rvalue::Bitcast {
            to: Ty::F32,
            operand: copy(signed_raw),
        },
    )
}
