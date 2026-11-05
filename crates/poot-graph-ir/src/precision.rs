//! Operand precision: the declared narrowing of an op that multiplies two float operands (POOT-1430,
//! `receipts/1429-narrow-operands/design.md` section 2).
//!
//! Narrowing is part of an op's semantics. The op states it, the oracle evaluates exactly it, and every schedule
//! (fragment, scalar micro-tile, GEMV) implements it, so numerics never depend on which schedule the planner picked.
//! The `select_operand_precision` planner pass is the one writer of a non-[`OperandPrecision::Exact`] value.

use std::fmt;

use poot_tensor::DType;

pub use poot_target::FragmentDtype;

/// The lowercase name a program dump and a result line print.
const fn fragment_name(dtype: FragmentDtype) -> &'static str {
    match dtype {
        FragmentDtype::Bf16 => "bf16",
        FragmentDtype::F16 => "f16",
    }
}

/// How the two float operands of a contraction or attention product are read.
///
/// Products always accumulate in f32, in `k` order; the accumulator is never narrower.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum OperandPrecision {
    /// Operands enter the products as the graph declares them (f32 activations, decoded weights): today's
    /// semantics.
    #[default]
    Exact,
    /// Every operand not already of the dtype is rounded to it (round to nearest even, non-saturating: an f32
    /// beyond the dtype's range becomes infinity) before the products.
    Narrow(FragmentDtype),
    /// Every operand not already of the dtype enters as `hi + lo`, two values of the dtype (`hi = rne(x)`,
    /// `lo = rne(x - hi)`); an operand already of the dtype enters once. The product is the sum over every
    /// pair of terms, one from each operand.
    Split(FragmentDtype),
}

impl OperandPrecision {
    /// The narrow dtype the operands are read as, `None` for [`OperandPrecision::Exact`].
    pub const fn fragment_dtype(self) -> Option<FragmentDtype> {
        match self {
            Self::Exact => None,
            Self::Narrow(dtype) | Self::Split(dtype) => Some(dtype),
        }
    }

    /// Whether the operands enter unrounded.
    pub const fn is_exact(self) -> bool {
        matches!(self, Self::Exact)
    }

    /// The dtype a contraction's activation may be stored in under this precision besides f32 (POOT-1448): the
    /// fragment dtype of a narrowed read, whose products read the activation in that dtype anyway, so a value
    /// stored in it enters unrounded and the product is the same. `None` for an exact or split read, which needs
    /// the f32 value (`x` itself, or `x - hi`).
    pub const fn stored_activation(self) -> Option<DType> {
        match self {
            Self::Narrow(dtype) => Some(fragment_storage_dtype(dtype)),
            Self::Exact | Self::Split(_) => None,
        }
    }
}

/// The tensor dtype that holds values of a fragment dtype.
const fn fragment_storage_dtype(dtype: FragmentDtype) -> DType {
    match dtype {
        FragmentDtype::Bf16 => DType::BF16,
        FragmentDtype::F16 => DType::F16,
    }
}

impl fmt::Display for OperandPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact => f.write_str("exact"),
            Self::Narrow(dtype) => write!(f, "narrow({})", fragment_name(*dtype)),
            Self::Split(dtype) => write!(f, "split({})", fragment_name(*dtype)),
        }
    }
}

/// The driver-wide policy the `select_operand_precision` planner pass reads (design 2.2): a permission to narrow
/// where the target's matrix units admit it, never an obligation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MatmulPrecision {
    /// No op narrows.
    Exact,
    /// Contractions and attention split where a matrix unit admits them.
    Split,
    /// Contractions and attention narrow where a matrix unit admits them.
    #[default]
    Narrow,
}

impl MatmulPrecision {
    /// The engine default: the tier-3 gates of the narrow policy pass on every backend (POOT-1461).
    pub const DEFAULT: Self = Self::Narrow;

    /// The lowercase name a flag, a config key and a result line use.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Split => "split",
            Self::Narrow => "narrow",
        }
    }

    /// Parse a flag or config value; `None` for an unknown name.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "exact" => Some(Self::Exact),
            "split" => Some(Self::Split),
            "narrow" => Some(Self::Narrow),
            _ => None,
        }
    }

    /// The precision this policy gives an op whose operands a matrix unit admits as `dtype`.
    pub const fn operand_precision(self, dtype: FragmentDtype) -> OperandPrecision {
        match self {
            Self::Exact => OperandPrecision::Exact,
            Self::Split => OperandPrecision::Split(dtype),
            Self::Narrow => OperandPrecision::Narrow(dtype),
        }
    }
}
