//! Per-pass numerics declarations and the tier-2 class a compiled program publishes (ADR-0101
//! decision 2).
//!
//! These classes describe device scheduling and comparison tolerances. They are not proofs of
//! semantic equivalence: every graph rewrite must separately preserve the CPU computation, including
//! intermediate rounding and every observable output and state value.
//!
//! A pass that rewrites floating-point work declares its device behavior:
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
//! program publishes [`NumericsProperty::tier2_class`] of that fold. [`verify_pass_numerics`] checks a
//! declaration against the ops the pass produced: a pass that reassociates while declaring `Exact` is
//! refused instead of publishing a tolerance that is too tight.

use super::{Graph, OpKind, ValueId};
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
    #[cfg(any(test, feature = "test-support"))]
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
/// [`NumericsProperty`] its graph exhibits. A device is compared with the oracle within a tolerance
/// sized by this class; [`Tier2Class::BitExact`] is the tier-1 bit-for-bit comparison, reachable only
/// for a graph whose float arithmetic a device never executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(any(test, feature = "test-support"))]
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

/// The property a single op implies, independent of any pass declaration. `out_dtype` decides whether
/// an elementwise op is float arithmetic (a real-device tier-2 class) or an exact-integer op (tier 1).
fn implied_op_numerics(op: &OpKind, out_dtype: DType) -> NumericsProperty {
    match op {
        OpKind::Fused(_)
        | OpKind::FusedRow(_)
        | OpKind::Attention(_)
        | OpKind::Rope { .. }
        | OpKind::Scan(_) => NumericsProperty::Reassociating,
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

/// One pass `compile` (`poot-graph-plan`) runs, its numerics declaration, and the ops it produces (the
/// set [`verify_pass_numerics`] reads the graph for).
#[derive(Clone, Copy, Debug)]
pub struct PassDeclaration {
    /// The pass's name as `compile` reports it.
    pub pass: &'static str,
    /// The numerics property the pass declares for the ops it produces.
    pub property: NumericsProperty,
    /// Whether an op is one the pass produces; [`verify_pass_numerics`] checks the declaration against these ops in the output graph.
    pub produces: fn(&OpKind) -> bool,
}

fn produces_nothing(_: &OpKind) -> bool {
    false
}

fn produces_rope(op: &OpKind) -> bool {
    matches!(op, OpKind::Rope { .. })
}

fn produces_flash(op: &OpKind) -> bool {
    matches!(op, OpKind::Attention(_))
}

fn produces_fused(op: &OpKind) -> bool {
    matches!(op, OpKind::Fused(_) | OpKind::FusedRow(_))
}

fn produces_contraction_epilogue(op: &OpKind) -> bool {
    matches!(op, OpKind::ContractionEpilogue { .. })
}

fn produces_dense_contraction(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::DenseContraction { .. } | OpKind::GroupedDenseContraction { .. }
    )
}

fn produces_packed_contraction(op: &OpKind) -> bool {
    matches!(op, OpKind::PackedContraction { .. })
}

fn produces_packed_row_gather(op: &OpKind) -> bool {
    matches!(op, OpKind::PackedRowGather { .. })
}

fn produces_dense_row_gather(op: &OpKind) -> bool {
    matches!(op, OpKind::DenseRowGather { .. })
}

/// `legalize`'s (`poot-graph-plan`) rewrite shapes: a hosted-embed `Gather` replaced by a device
/// `MatMul`-free lookup, or an oversized matmul weight split and rejoined by `Concat`.
fn produces_legalize_ops(op: &OpKind) -> bool {
    matches!(op, OpKind::MatMul | OpKind::Concat { .. })
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

/// Every pass `compile` (`poot-graph-plan`) runs, with its numerics declaration, in `compile`'s
/// pipeline order. Card 626: every pass function named here now lives in `poot-graph-plan` (a
/// `pub(crate)` module there, reachable only through `compile`), so this table cannot reference a pass
/// module's own constant or predicate without a dependency cycle (this crate is below
/// `poot-graph-plan`); every `property` is inlined and every `produces` predicate reads only `OpKind`,
/// duplicated locally rather than imported. A pass added to `compile` without a row here is a bug the
/// pipeline's fail-closed check cannot see.
pub const PASS_DECLARATIONS: &[PassDeclaration] = &[
    PassDeclaration {
        pass: "canonicalize",
        property: NumericsProperty::Exact,
        produces: produces_nothing,
    },
    PassDeclaration {
        pass: "cse",
        property: NumericsProperty::Exact,
        produces: produces_nothing,
    },
    PassDeclaration {
        pass: "fold_iota",
        property: NumericsProperty::Exact,
        produces: produces_nothing,
    },
    PassDeclaration {
        pass: "rope_fusion",
        property: NumericsProperty::Reassociating,
        produces: produces_rope,
    },
    PassDeclaration {
        pass: "flash_attention_capped",
        property: NumericsProperty::Reassociating,
        produces: produces_flash,
    },
    PassDeclaration {
        pass: "dce",
        property: NumericsProperty::Exact,
        produces: produces_nothing,
    },
    // Card 642: `compile` claims `ops::packed_linear`'s chains as `PackedContraction`
    // right after `dce`, then refuses every `PackedDequant` left unclaimed (Card 355's escape gate), so
    // no later pass sees a packed decode outside a contraction. The claim replaces a `PackedDequant`
    // (already `Narrowing`) and its `MatMul` with one `PackedContraction` (`Narrowing`).
    PassDeclaration {
        pass: "recognize_packed_contractions",
        property: NumericsProperty::Narrowing,
        produces: produces_packed_contraction,
    },
    // Card 545a: the row-gather claim (a quantized embedding lookup) after the
    // contraction claim, before the gate.
    PassDeclaration {
        pass: "recognize_packed_row_gathers",
        property: NumericsProperty::Narrowing,
        produces: produces_packed_row_gather,
    },
    PassDeclaration {
        pass: "reject_packed_dequant_escapes",
        property: NumericsProperty::Exact,
        produces: produces_nothing,
    },
    // Card 534a: `compile`'s pipeline runs these four between `dce` and `fuse`, so the folds and the
    // reduce legalization see the traced `MatMul`/`Transpose`, `Cast`/`Gather` and non-last-axis
    // `Reduce` chains before `fuse` could absorb them into a region no target plans.
    PassDeclaration {
        pass: "fold_dense_contractions",
        property: NumericsProperty::Narrowing,
        produces: produces_dense_contraction,
    },
    PassDeclaration {
        pass: "fold_dense_bf16_row_gathers",
        property: NumericsProperty::Narrowing,
        produces: produces_dense_row_gather,
    },
    PassDeclaration {
        pass: "lower_nonlast_reduces",
        property: NumericsProperty::Exact,
        produces: produces_nothing,
    },
    PassDeclaration {
        pass: "widen_mismatched_matmul_dtypes",
        property: NumericsProperty::Narrowing,
        produces: produces_nothing,
    },
    // The contraction epilogue rule folds a contraction and the pointwise chain that consumes it into one
    // equation. The lowering rounds the head through its declared dtype and every step through its own, so
    // the fused result is the unfused program's, bit for bit, on the same schedule: an unfused contraction.
    PassDeclaration {
        pass: "fuse_contraction_epilogues",
        property: NumericsProperty::Unfused,
        produces: produces_contraction_epilogue,
    },
    PassDeclaration {
        pass: "fuse",
        property: NumericsProperty::Reassociating,
        produces: produces_fused,
    },
    // `fuse_widening_casts` rewrites a region's own steps: the `Cast` of a narrow weight becomes the region's
    // first step. Widening a narrow float is exact and no op kind is produced that was not already counted, so
    // the class of the result is read off its ops, as for every pass.
    PassDeclaration {
        pass: "fuse_widening_casts",
        property: NumericsProperty::Exact,
        produces: produces_nothing,
    },
    // `fuse_index_casts` makes the `Cast` of an I32 word a step of the pointwise region that reads or writes
    // it, wrapping a lone pointwise equation in a one-step region where it has none. The conversion is the
    // cast kernel's own, so the class is the fused regions it produces.
    PassDeclaration {
        pass: "fuse_index_casts",
        property: NumericsProperty::Reassociating,
        produces: produces_fused,
    },
    // `legalize` needs `Target::caps` (card 522); `compile` (poot-graph-plan) runs it as its own pass,
    // between `dce` and `fuse` (card 523a).
    PassDeclaration {
        pass: "legalize",
        property: NumericsProperty::Unfused,
        produces: produces_legalize_ops,
    },
];

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

/// Check a declaration against the strongest property among the ops the pass produces in `out`. Fails
/// closed when the declaration is weaker than what it produced (a pass that reassociates while claiming
/// `Exact`).
pub fn verify_pass_numerics(
    declaration: &PassDeclaration,
    out: &Graph,
) -> Result<(), NumericsError> {
    let implied = out
        .eqns()
        .iter()
        .filter(|eqn| (declaration.produces)(&eqn.op))
        .map(|eqn| implied_op_numerics(&eqn.op, out.aval(eqn.out).dtype))
        .fold(NumericsProperty::Exact, NumericsProperty::strongest);
    if declaration.property >= implied {
        return Ok(());
    }
    Err(NumericsError::UnderDeclared {
        pass: declaration.pass,
        declared: declaration.property,
        implied,
    })
}
