//! The value families' emitters: one module per [`poot_quant::families`] family, selected by the
//! descriptor's [`ValueMap`], never by a format name.
//!
//! A family says which constant tables its kernel declares ([`Family::tables`]), which extra
//! per-element fields of the block it reads besides `Codes` ([`Family::operands`]: a lattice's
//! `Signs`), and how one element's value follows from the bits read ([`Family::emit_value`]).
//! `DecodePlan` reads the fields and combines the value with the block's factors. A closed set of
//! families is an enum, not a trait object.

mod float;
mod integer;
mod lattice;

use poot_kernel_ir::{ConstantTableDecl, Local};
use poot_quant::families::lattice::Lattice;
use poot_quant::format::{Field, FieldEncoding, OperandRole, ValueMap};
use poot_target::RoundingConversions;

use super::float_decode::FloatDecode;
use crate::emit::Emit;

/// The value family a descriptor selects.
#[derive(Clone, Copy)]
pub(super) enum Family {
    Integer,
    Float(FloatDecode),
    Lattice(&'static Lattice),
}

/// The `Codes` operand of one element: its raw bits and the field they came from.
#[derive(Clone, Copy)]
pub(super) struct Codes {
    pub(super) bits: Local,
    pub(super) field: Field,
}

/// Where an element sits in its block. `known` is `Some((r, m))` when `index % m == r` is known at
/// emission (element `j` of a run of `count`: `(j, count)`), which lets a family fold the lane
/// arithmetic that depends on the position to constants. `lane` is the element's lane when the run's
/// lanes were computed once for all its elements ([`Family::emit_run_lanes`]).
#[derive(Clone, Copy)]
pub(super) struct Position {
    pub(super) index: Local,
    pub(super) known: Option<(usize, usize)>,
    pub(super) lane: Option<Local>,
}

impl Family {
    /// The family `value` selects over the `Codes` field `codes`; `None` for a float family whose codes are not
    /// float-encoded.
    pub(super) fn of(value: ValueMap, codes: &Field) -> Option<Self> {
        Some(match value {
            ValueMap::Integer => Self::Integer,
            ValueMap::Float => {
                let FieldEncoding::Float(format) = codes.encoding else {
                    return None;
                };
                Self::Float(FloatDecode::exact(format))
            }
            ValueMap::Lattice(lattice) => Self::Lattice(lattice),
        })
    }

    /// The family decoding through what the device keeps ([`FloatDecode::shifted_on`]), for a decode whose
    /// value meets a scale ([`Self::fold_scale`] is then owed on it).
    pub(super) fn decoding_on(self, conversions: RoundingConversions) -> Self {
        match self {
            Self::Float(decode) => Self::Float(decode.shifted_on(conversions)),
            other => other,
        }
    }

    /// `scale` with what the family's value owes it: the power of two of a shifted float decode, nothing for
    /// any other.
    pub(super) fn fold_scale(&self, e: &mut Emit, scale: Local) -> Local {
        match self {
            Self::Float(decode) => decode.fold_scale(e, scale),
            Self::Integer | Self::Lattice(_) => scale,
        }
    }

    /// The constant tables the family's decode reads, for the kernel body to declare
    /// ([`poot_kernel_ir::Body::constant_tables`]): a lattice's cells as the `u32` words a device
    /// reads them in; nothing for a family whose value is computed from the code.
    pub(super) fn tables(&self) -> Vec<ConstantTableDecl> {
        match self {
            Self::Integer | Self::Float(_) => Vec::new(),
            Self::Lattice(lattice) => lattice::tables(lattice),
        }
    }

    /// The lane of each element of the aligned run of `count` elements from in-block index `first`, when the family
    /// reads one that is not a constant of the position and the run's lanes follow from its first element's by
    /// offsets (a lattice whose lane map is piecewise); `None` when [`Self::emit_value`] finds each element's lane
    /// itself.
    pub(super) fn emit_run_lanes(
        &self,
        e: &mut Emit,
        first: Local,
        count: usize,
    ) -> Option<Vec<Local>> {
        match self {
            Self::Integer | Self::Float(_) => None,
            Self::Lattice(lattice) => lattice::emit_run_lanes(e, lattice, first, count),
        }
    }

    /// The block fields, besides `Codes`, that [`Self::emit_value`] reads, in the order it takes
    /// their bits.
    pub(super) fn operands(&self) -> &'static [OperandRole] {
        match self {
            Self::Integer | Self::Float(_) => &[],
            Self::Lattice(lattice) => lattice::operands(lattice),
        }
    }

    /// The value of one element, before the block's factors: `codes` read through the family, with
    /// `extras` the bits of [`Self::operands`] in order.
    pub(super) fn emit_value(
        &self,
        e: &mut Emit,
        codes: Codes,
        extras: &[Local],
        at: Position,
    ) -> Local {
        match self {
            Self::Integer => integer::emit_value(e, codes),
            Self::Float(decode) => float::emit_value(e, *decode, codes),
            Self::Lattice(lattice) => lattice::emit_value(e, lattice, codes.bits, extras, at),
        }
    }
}
