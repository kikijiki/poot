//! Op-level numerics classification and the tier-2 class a compiled program publishes (ADR-0101
//! decision 2). A pass's contribution is derived from the graphs it rewrote ([`NumericsDelta`]).
//!
//! These classes describe device scheduling and comparison tolerances. They are not proofs of
//! semantic equivalence: every graph rewrite must separately preserve the CPU computation, including
//! intermediate rounding and every observable output and state value.
//!
//! A graph rewrite is described by the strongest of these over the equations it produced:
//!
//! - [`NumericsProperty::Exact`] preserves every result and carried value bit for bit, so the oracle's
//!   tier-1 equality still holds.
//! - [`NumericsProperty::Unfused`] is the hardware baseline: ordinary float arithmetic a device runs
//!   without a pass rewriting it (an untiled contraction, a reduction, a float elementwise chain). The
//!   CPU oracle is not bit-identical to a real device for it (ADR-0101 Context: the unoptimized wgpu
//!   route differs in every logit, max 2.8e-6), so it is a tier-2 class, never `BitExact`.
//! - [`NumericsProperty::Reassociating`] reorders float arithmetic (fusion, tiling, flash, rope).
//! - [`NumericsProperty::Narrowing`] narrows storage (a dtype retype or a quantized decode), losing
//!   precision the wider computation kept.
//!
//! The properties are ordered by severity, so the strongest one a graph exhibits is a fold. A compiled
//! program publishes [`NumericsProperty::tier2_class`] of that fold, and a device result is compared with
//! the oracle within [`Tier2Class::tolerance`]. [`NumericsDelta`] reads the ops a rewrite produced off the two
//! graphs, so a pass that reassociates cannot publish a tolerance that is too tight.

use std::collections::HashMap;

use super::{Graph, OpKind, ValueId};
use crate::graph::Eqn;
use crate::types::{DType, DTypeClass, DTypeClassExt};

/// Device scheduling behavior (ADR-0101 decision 2), ordered by severity. This describes comparison
/// tolerances; semantic rewrite legality requires a separate proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum NumericsProperty {
    /// The pass adds no device rounding or reassociation. A whole program reaches `BitExact` only
    /// when its baseline and every pass are exact; removing a float identity does not erase the
    /// surrounding program's float baseline.
    Exact,
    /// The unfused hardware baseline: a real device's float contraction, reduction or elementwise
    /// arithmetic, which the CPU oracle does not reproduce bit for bit.
    Unfused,
    /// Float arithmetic is reassociated (fusion, tiling, flash, rope fusion).
    Reassociating,
    /// Storage is narrowed (a dtype retype or a packed/quantized decode).
    Narrowing,
}

impl NumericsProperty {
    /// The stronger (looser) of two declarations.
    pub fn strongest(self, other: Self) -> Self {
        self.max(other)
    }

    /// The comparison class a program whose strongest property is `self` publishes.
    pub fn tier2_class(self) -> Tier2Class {
        match self {
            NumericsProperty::Exact => Tier2Class::BitExact,
            NumericsProperty::Unfused => Tier2Class::Unfused,
            NumericsProperty::Reassociating => Tier2Class::Reassociating,
            NumericsProperty::Narrowing => Tier2Class::NarrowStorage,
        }
    }
}

/// The tier-2 comparison class of a compiled program (ADR-0101 decision 2), derived from the strongest
/// [`NumericsProperty`] its graph exhibits and ordered like it. A device is compared with the oracle
/// within [`Tier2Class::tolerance`]; [`Tier2Class::BitExact`] is the tier-1 bit-for-bit comparison,
/// reachable only for a graph whose float arithmetic a device never executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier2Class {
    /// The graph runs no float arithmetic a device executes: compare every element bit for bit.
    BitExact,
    /// The graph runs ordinary unfused float arithmetic: compare within the hardware baseline
    /// tolerance.
    Unfused,
    /// A pass reassociated float arithmetic: compare within the reassociating tolerance.
    Reassociating,
    /// A pass narrowed storage: compare within the wider narrow-storage tolerance.
    NarrowStorage,
}

impl Tier2Class {
    /// The one published table from a class to its comparison tolerance, relative to
    /// `max(|oracle element|, 1)`: a device element agrees when `|device - oracle| <= tolerance *
    /// max(|oracle element|, 1)`. Each class is looser than the one before it, because a pass of that
    /// class does more to the arithmetic. Every parity comparison reads this table; no row picks its own.
    pub const fn tolerance(self) -> f32 {
        match self {
            Tier2Class::BitExact => 0.0,
            Tier2Class::Unfused => 1e-4,
            Tier2Class::Reassociating => 1e-3,
            Tier2Class::NarrowStorage => 5e-3,
        }
    }
}

/// The property a single op implies, independent of any pass declaration. `out_dtype` decides whether
/// an elementwise op is float arithmetic (a real-device tier-2 class) or an exact-integer op (tier 1).
fn implied_op_numerics(op: &OpKind, out_dtype: DType) -> NumericsProperty {
    match op {
        OpKind::Fused(_)
        | OpKind::FusedRow(_)
        | OpKind::Attention(_)
        | OpKind::Rope { .. }
        | OpKind::Scan(_)
        | OpKind::SelectedAttention(_) => NumericsProperty::Reassociating,
        OpKind::PackedDequant { .. }
        | OpKind::PackedContraction { .. }
        | OpKind::PackedRowGather { .. }
        | OpKind::DenseContraction { .. }
        | OpKind::GroupedDenseContraction { .. }
        | OpKind::DenseRowGather { .. }
        | OpKind::IndexedMatMul
        | OpKind::PackI8
        | OpKind::UnpackI8 { .. } => NumericsProperty::Narrowing,
        OpKind::Cast { to }
            if matches!(to.class(), DTypeClass::StorageOnly | DTypeClass::Packed) =>
        {
            NumericsProperty::Narrowing
        }
        OpKind::Cast { to } if *to == DType::F16 || *to == DType::BF16 => {
            NumericsProperty::Narrowing
        }
        // The unfused hardware baseline. A contraction or reduction sums in a device-defined order, and
        // a float elementwise chain is another device-executed arithmetic path, so none of them is
        // bit-exact against the CPU oracle.
        OpKind::MatMul | OpKind::ContractionEpilogue { .. } | OpKind::Reduce { .. } => {
            NumericsProperty::Unfused
        }
        OpKind::Binary(_) | OpKind::Unary(_)
            if matches!(out_dtype.class(), DTypeClass::Arithmetic) =>
        {
            NumericsProperty::Unfused
        }
        // `Select` is a select, not arithmetic: it moves a branch's bits, so it stays exact at any dtype.
        // Movement, I32/quantized elementwise and every other op preserve their operand bits: the
        // exact-integer and exact-I32 paths are the only ones that stay bit-exact on a device.
        _ => NumericsProperty::Exact,
    }
}

/// The strongest property the equations of `g` exhibit, read off their ops and output dtypes. This is
/// the ground truth a program publishes and a declaration is checked against, never a restatement of a
/// declaration.
pub fn implied_numerics(g: &Graph) -> NumericsProperty {
    g.eqns()
        .iter()
        .map(|eqn| implied_op_numerics(&eqn.op, g.aval(eqn.out).dtype))
        .fold(NumericsProperty::Exact, NumericsProperty::strongest)
}

/// An arithmetic rewrite withheld because the graph supplies no exact numerical proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericalRewriteReason {
    /// Moving a scale across a contraction can change overflow, underflow and rounding.
    ScalarAcrossContraction,
    /// Multiplication by a rounded reciprocal is not division.
    ReciprocalRounding,
}

/// A retained primitive candidate and the numerical condition preventing its rewrite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericalRewriteDecline {
    /// The value the declined rewrite would have produced.
    pub value: ValueId,
    /// The numerical condition that makes the rewrite unsound.
    pub reason: NumericalRewriteReason,
}

/// Why a pass's declaration is too weak for the graph it produced.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NumericsError {
    /// The pass declared a property weaker than its output exhibits.
    #[error(
        "pass {pass:?} declares {declared:?} but produced ops that imply {implied:?}, so the program's \
         tier-2 tolerance would be too tight"
    )]
    UnderDeclared {
        /// The pass whose declaration is too weak.
        pass: &'static str,
        /// The property the pass declared.
        declared: NumericsProperty,
        /// The strongest property the produced ops exhibit.
        implied: NumericsProperty,
    },
}

/// What a rewrite did to a graph's device arithmetic, read off the two graphs and nothing else: the strongest
/// property among the equations it produced. An equation is produced when its output is new or its op or output
/// type changed. A pass declares nothing: the program's class absorbs what it produced, so a pass that
/// introduces a reassociating op cannot publish a tolerance that is too tight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericsDelta {
    /// The strongest property among the equations the rewrite added or changed.
    pub produced: NumericsProperty,
}

impl NumericsDelta {
    /// The delta of rewriting `before` into `after`.
    pub fn between(before: &Graph, after: &Graph) -> Self {
        let before_at: HashMap<ValueId, &Eqn> = before.eqns().iter().map(|e| (e.out, e)).collect();
        let class = |g: &Graph, eqn: &Eqn| implied_op_numerics(&eqn.op, g.aval(eqn.out).dtype);
        let same = |old: &Eqn, new: &Eqn| {
            old.op == new.op && before.aval(old.out).dtype == after.aval(new.out).dtype
        };
        let produced = after
            .eqns()
            .iter()
            .filter(|new| before_at.get(&new.out).is_none_or(|old| !same(old, new)))
            .map(|eqn| class(after, eqn))
            .fold(NumericsProperty::Exact, NumericsProperty::strongest);
        Self { produced }
    }
}
