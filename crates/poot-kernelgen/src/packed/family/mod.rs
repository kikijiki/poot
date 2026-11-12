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
use poot_quant::format::{Field, OperandRole, ValueMap};

use crate::emit::Emit;

/// The value family a descriptor selects.
#[derive(Clone, Copy)]
pub(super) enum Family {
    Integer,
    Float,
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
/// arithmetic that depends on the position to constants.
#[derive(Clone, Copy)]
pub(super) struct Position {
    pub(super) index: Local,
    pub(super) known: Option<(usize, usize)>,
}

impl Family {
    pub(super) fn of(value: ValueMap) -> Self {
        match value {
            ValueMap::Integer => Self::Integer,
            ValueMap::Float => Self::Float,
            ValueMap::Lattice(lattice) => Self::Lattice(lattice),
        }
    }

    /// The constant tables the family's decode reads, for the kernel body to declare
    /// ([`poot_kernel_ir::Body::constant_tables`]): a lattice's cells as the `u32` words a device
    /// reads them in; nothing for a family whose value is computed from the code.
    pub(super) fn tables(&self) -> Vec<ConstantTableDecl> {
        match self {
            Self::Integer | Self::Float => Vec::new(),
            Self::Lattice(lattice) => lattice::tables(lattice),
        }
    }

    /// The block fields, besides `Codes`, that [`Self::emit_value`] reads, in the order it takes
    /// their bits.
    pub(super) fn operands(&self) -> &'static [OperandRole] {
        match self {
            Self::Integer | Self::Float => &[],
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
            Self::Float => float::emit_value(e, codes),
            Self::Lattice(lattice) => lattice::emit_value(e, lattice, codes.bits, extras, at),
        }
    }
}
