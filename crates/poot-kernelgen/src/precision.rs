//! Operand precision in generated bodies (POOT-1430): the one place a kernel turns a value it read into the
//! terms its products multiply.
//!
//! Every schedule that multiplies two float operands (the register-tiled and GEMV contraction nests, the serial
//! matmul, the decode GEMVs, the attention bodies) reads each operand through [`operand_terms`], so the declared
//! rounding is the same whichever schedule the planner picked. The definition is `poot-eval`'s: round to nearest
//! even, non-saturating, `lo = rne(x - hi)` for a finite `hi` and `0` for an infinite or NaN one.

use poot_graph_ir::{FragmentDtype, OperandPrecision};
use poot_kernel_ir::{BinOp, Constant, Local, Operand, Place, Rvalue, Statement, Ty};

use crate::emit::{Emit, bin, c32, copy};
use crate::helpers::{Alloc, WeightLane};

/// How a body reads its two operands: the declared precision, and the fragment dtype each operand is already
/// stored in, if it is (an operand stored in the fragment dtype is exact in it and enters once). The activation of
/// a generated contraction is f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProductRead {
    pub(crate) precision: OperandPrecision,
    pub(crate) activation_stored: Option<FragmentDtype>,
    pub(crate) weight_stored: Option<FragmentDtype>,
}

/// What a register-tiled contraction stages for each operand: one tile, or two (`hi` and `lo`) for a split operand,
/// and the fragment dtype the operands narrow to, if they do (a matrix unit multiplies them in it). What the
/// tiling's LDS budget is sized for, and whether a fragment tiling is offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StagedTerms {
    /// Tiles of the activation.
    pub activation: usize,
    /// Tiles of the weight.
    pub weight: usize,
    /// The fragment dtype of a narrowed or split precision, `None` for an exact one.
    pub fragment: Option<FragmentDtype>,
}

impl StagedTerms {
    /// One tile per operand, exact operands.
    pub const ONE: Self = Self {
        activation: 1,
        weight: 1,
        fragment: None,
    };

    /// These terms for a contraction that keeps the scalar micro-tile (an expert walk, whose rows select different
    /// weight tiles): the same tiles, no fragment dtype.
    pub const fn scalar_only(self) -> Self {
        Self {
            fragment: None,
            ..self
        }
    }

    /// The tiles `precision` stages for a weight held as `weight_stored` (a weight stored in the fragment dtype is
    /// exact in it and stages once; an activation is f32 and splits).
    pub fn of(
        precision: OperandPrecision,
        activation_stored: Option<FragmentDtype>,
        weight_stored: Option<FragmentDtype>,
    ) -> Self {
        if precision.is_exact() {
            return Self::ONE;
        }
        let split = |stored: Option<FragmentDtype>| match precision {
            OperandPrecision::Split(dtype) if stored != Some(dtype) => 2,
            _ => 1,
        };
        Self {
            activation: split(activation_stored),
            weight: split(weight_stored),
            fragment: precision.fragment_dtype(),
        }
    }
}

/// The fragment dtype a weight held in `storage` is stored in: the dtype of a packed 16-bit float weight, `None`
/// for any other (f32 words, or a packed-quant format decoded to f32).
pub fn stored_fragment_dtype(storage: poot_target::BufferStorage) -> Option<FragmentDtype> {
    if storage == poot_target::BufferStorage::bf16_packed() {
        Some(FragmentDtype::Bf16)
    } else if storage == poot_target::BufferStorage::f16_packed() {
        Some(FragmentDtype::F16)
    } else {
        None
    }
}

impl ProductRead {
    /// The tiles this read stages.
    pub(crate) fn staged_terms(self) -> StagedTerms {
        StagedTerms::of(self.precision, self.activation_stored, self.weight_stored)
    }

    /// A read of a weight decoded to f32 from a packed format or held as f32 words: never stored in the
    /// fragment dtype, so a narrowed or split precision rounds it.
    pub(crate) const fn decoded(precision: OperandPrecision) -> Self {
        Self {
            precision,
            activation_stored: None,
            weight_stored: None,
        }
    }

    /// A read of two operands of one storage dtype `dt` (the serial matmul): an operand stored in the fragment
    /// dtype is exact in it.
    pub(crate) fn of_dtype(precision: OperandPrecision, dt: &Ty) -> Self {
        let stored = match dt {
            Ty::BF16 => Some(FragmentDtype::Bf16),
            Ty::F16 => Some(FragmentDtype::F16),
            _ => None,
        };
        Self {
            precision,
            activation_stored: stored,
            weight_stored: stored,
        }
    }

    /// A read of a weight held in the dense `lane` storage.
    pub(crate) fn dense(precision: OperandPrecision, lane: WeightLane) -> Self {
        Self {
            precision,
            activation_stored: None,
            weight_stored: match lane {
                WeightLane::F32 => None,
                WeightLane::F16Packed => Some(FragmentDtype::F16),
                WeightLane::Bf16Packed => Some(FragmentDtype::Bf16),
            },
        }
    }

    /// The terms of an activation value.
    pub(crate) fn activation_terms(self, e: &mut Emit, x: Local) -> Vec<Local> {
        operand_terms(e, self.precision, self.activation_stored, x)
    }

    /// The terms of a decoded weight value.
    pub(crate) fn weight_terms(self, e: &mut Emit, x: Local) -> Vec<Local> {
        operand_terms(e, self.precision, self.weight_stored, x)
    }
}

/// `x` rounded to `dtype` (nearest even, ties to even, non-saturating) and widened back to f32, bit for bit the
/// oracle's `round_to_fragment`.
///
/// The rounding is integer arithmetic on the f32 bits, not a `Cast` to a 16-bit float and back. A narrow-and-widen
/// pair of casts is a no-op the driver compiler is free to keep as f32 (RADV does: `f32(f16(65519.0))` comes back
/// `65519.0` on gfx1151), and there is no `Ty::BF16` on SPIR-V at all, so the cast pair would round on some targets
/// and not on others. Integer bit operations round the same value to the same bits on every target, including the
/// kernel-IR interpreter.
pub(crate) fn round_to_fragment(e: &mut Emit, dtype: FragmentDtype, x: Local) -> Local {
    let bits = e.let_(
        Ty::U32,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: copy(x),
        },
    );
    let sign = bin(e, Ty::U32, BinOp::BitAnd, copy(bits), c32(0x8000_0000));
    let magnitude = bin(e, Ty::U32, BinOp::BitAnd, copy(bits), c32(0x7fff_ffff));
    let rounded = match dtype {
        FragmentDtype::Bf16 => round_bf16_magnitude(e, magnitude),
        FragmentDtype::F16 => round_f16_magnitude(e, magnitude),
    };
    let signed = bin(e, Ty::U32, BinOp::BitOr, copy(sign), copy(rounded));
    e.let_(
        Ty::F32,
        Rvalue::Bitcast {
            to: Ty::F32,
            operand: copy(signed),
        },
    )
}

/// `if condition { yes } else { no }` over `u32` values.
fn pick(e: &mut Emit, condition: Local, yes: Local, no: Local) -> Local {
    e.let_(
        Ty::U32,
        Rvalue::Select {
            condition: copy(condition),
            if_true: copy(yes),
            if_false: copy(no),
        },
    )
}

fn constant(e: &mut Emit, value: u32) -> Local {
    e.let_(Ty::U32, Rvalue::Use(c32(value)))
}

fn compare(e: &mut Emit, op: BinOp, a: Local, b: u32) -> Local {
    bin(e, Ty::Bool, op, copy(a), c32(b))
}

/// The magnitude bits of an f32 rounded to the 8 significant bits of a bf16: add the rounding bias (`0x7fff` plus
/// the kept lsb, so a tie goes to even), drop the low 16 bits. A carry into the exponent is the correct next
/// binade, and past the largest finite bf16 it lands on infinity. A NaN keeps its bits.
fn round_bf16_magnitude(e: &mut Emit, magnitude: Local) -> Local {
    let lsb = bin(e, Ty::U32, BinOp::Shr, copy(magnitude), c32(16));
    let lsb = bin(e, Ty::U32, BinOp::BitAnd, copy(lsb), c32(1));
    let bias = bin(e, Ty::U32, BinOp::Add, copy(lsb), c32(0x7fff));
    let sum = bin(e, Ty::U32, BinOp::Add, copy(magnitude), copy(bias));
    let rounded = bin(e, Ty::U32, BinOp::BitAnd, copy(sum), c32(0xffff_0000));
    let is_nan = compare(e, BinOp::Gt, magnitude, 0x7f80_0000);
    pick(e, is_nan, magnitude, rounded)
}

/// The magnitude bits of an f32 rounded to a binary16 (11 significant bits in the normal range, a fixed spacing of
/// `2^-24` below it), then widened: ties to even, an overflow to infinity, a NaN or infinity unchanged.
///
/// - `|x| >= 2^-14` (normal): the bias add of [`round_bf16_magnitude`] at bit 13; a result above 65504 is infinity.
/// - `2^-25 < |x| < 2^-14` (subnormal): the 24-bit significand `m` shifted right by `126 - exponent` to a count of
///   `2^-24` steps, rounded to nearest even from the bits shifted out, and scaled back (`n * 2^-24` is exact).
/// - `|x| <= 2^-25`: zero (`2^-25` is the tie between 0 and the smallest subnormal, and zero is even).
fn round_f16_magnitude(e: &mut Emit, magnitude: Local) -> Local {
    // Normal range.
    let lsb = bin(e, Ty::U32, BinOp::Shr, copy(magnitude), c32(13));
    let lsb = bin(e, Ty::U32, BinOp::BitAnd, copy(lsb), c32(1));
    let bias = bin(e, Ty::U32, BinOp::Add, copy(lsb), c32(0x0fff));
    let sum = bin(e, Ty::U32, BinOp::Add, copy(magnitude), copy(bias));
    let kept = bin(e, Ty::U32, BinOp::BitAnd, copy(sum), c32(0xffff_e000));
    let overflows = compare(e, BinOp::Gt, kept, 0x477f_e000);
    let infinity = constant(e, 0x7f80_0000);
    let normal = pick(e, overflows, infinity, kept);

    // Subnormal range, over a shift clamped into range where the value is not subnormal.
    let in_subnormal = {
        let above = compare(e, BinOp::Gt, magnitude, 0x3300_0000);
        let below = compare(e, BinOp::Lt, magnitude, 0x3880_0000);
        bin(e, Ty::Bool, BinOp::BitAnd, copy(above), copy(below))
    };
    let exponent = bin(e, Ty::U32, BinOp::Shr, copy(magnitude), c32(23));
    let shift = bin(e, Ty::U32, BinOp::Sub, c32(126), copy(exponent));
    let fallback = constant(e, 24);
    let shift = pick(e, in_subnormal, shift, fallback);
    let fraction = bin(e, Ty::U32, BinOp::BitAnd, copy(magnitude), c32(0x007f_ffff));
    let significand = bin(e, Ty::U32, BinOp::BitOr, copy(fraction), c32(0x0080_0000));
    let steps = bin(e, Ty::U32, BinOp::Shr, copy(significand), copy(shift));
    let one = constant(e, 1);
    let unit = bin(e, Ty::U32, BinOp::Shl, copy(one), copy(shift));
    let mask = bin(e, Ty::U32, BinOp::Sub, copy(unit), c32(1));
    let dropped = bin(e, Ty::U32, BinOp::BitAnd, copy(significand), copy(mask));
    let shift_less_one = bin(e, Ty::U32, BinOp::Sub, copy(shift), c32(1));
    let half = bin(e, Ty::U32, BinOp::Shl, copy(one), copy(shift_less_one));
    let past_half = bin(e, Ty::Bool, BinOp::Gt, copy(dropped), copy(half));
    let at_half = bin(e, Ty::Bool, BinOp::Eq, copy(dropped), copy(half));
    let odd = bin(e, Ty::U32, BinOp::BitAnd, copy(steps), c32(1));
    let odd = bin(e, Ty::Bool, BinOp::Eq, copy(odd), c32(1));
    let tie_up = bin(e, Ty::Bool, BinOp::BitAnd, copy(at_half), copy(odd));
    let up = bin(e, Ty::Bool, BinOp::BitOr, copy(past_half), copy(tie_up));
    let up = e.let_(
        Ty::U32,
        Rvalue::Cast {
            to: Ty::U32,
            operand: copy(up),
        },
    );
    let count = bin(e, Ty::U32, BinOp::Add, copy(steps), copy(up));
    let count = e.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(count),
        },
    );
    let scaled = bin(
        e,
        Ty::F32,
        BinOp::Mul,
        copy(count),
        Operand::Const(Constant::F32(2f32.powi(-24))),
    );
    let subnormal = e.let_(
        Ty::U32,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: copy(scaled),
        },
    );

    // Compose, innermost first: zero at or below the tie, then the subnormal and normal ranges, then a NaN or
    // infinity unchanged.
    let zero = constant(e, 0);
    let small = pick(e, in_subnormal, subnormal, zero);
    let finite = {
        let is_normal = compare(e, BinOp::Ge, magnitude, 0x3880_0000);
        pick(e, is_normal, normal, small)
    };
    let is_special = compare(e, BinOp::Ge, magnitude, 0x7f80_0000);
    pick(e, is_special, magnitude, finite)
}

/// The terms the f32 value `x` enters the products as under `precision`: `x` itself when the precision is exact or
/// the operand is stored in the fragment dtype (`stored`), its rounded value for a narrowed operand, and `hi`, `lo`
/// for a split one.
pub(crate) fn operand_terms(
    e: &mut Emit,
    precision: OperandPrecision,
    stored: Option<FragmentDtype>,
    x: Local,
) -> Vec<Local> {
    let Some(dtype) = precision.fragment_dtype() else {
        return vec![x];
    };
    if stored == Some(dtype) {
        return vec![x];
    }
    let hi = round_to_fragment(e, dtype, x);
    let OperandPrecision::Split(_) = precision else {
        return vec![hi];
    };
    // `lo` is zero for a non-finite `hi`: `hi - hi` is `0` exactly when `hi` is finite.
    let residual = bin(e, Ty::F32, BinOp::Sub, copy(x), copy(hi));
    let lo_raw = round_to_fragment(e, dtype, residual);
    let spread = bin(e, Ty::F32, BinOp::Sub, copy(hi), copy(hi));
    let finite = bin(
        e,
        Ty::Bool,
        BinOp::Eq,
        copy(spread),
        Operand::Const(Constant::F32(0.0)),
    );
    let lo = e.let_(
        Ty::F32,
        Rvalue::Select {
            condition: copy(finite),
            if_true: copy(lo_raw),
            if_false: Operand::Const(Constant::F32(0.0)),
        },
    );
    vec![hi, lo]
}

/// The one narrowed term of an operand a staging step writes as a single tile. A split operand has two terms and
/// cannot be staged this way; a schedule that stages one tile per operand refuses `Split` before it calls this.
pub(crate) fn narrowed(
    e: &mut Emit,
    precision: OperandPrecision,
    stored: Option<FragmentDtype>,
    x: Local,
) -> Local {
    debug_assert!(
        !matches!(precision, OperandPrecision::Split(_)),
        "a single staged tile has no lo part"
    );
    operand_terms(e, precision, stored, x)[0]
}

/// `dst = acc + sum of the products of every pair of terms of a and b`, as statements: the `k` step of a hand-built
/// accumulate loop whose operands are the f32 locals `a` and `b` (already widened from their storage). `Exact`
/// is the single product `a * b`; the kernels keep their original statements for it, so their keys do not change.
pub(crate) fn accumulate_statements(
    al: &mut Alloc,
    read: ProductRead,
    (a, b): (Local, Local),
    acc: Local,
    dst: Local,
) -> Vec<Statement> {
    let mut e = Emit::new(al);
    let (a_terms, b_terms) = (
        read.activation_terms(&mut e, a),
        read.weight_terms(&mut e, b),
    );
    let mut sum = acc;
    for (x, y) in pair_products(&a_terms, &b_terms) {
        let product = bin(&mut e, Ty::F32, BinOp::Mul, copy(x), copy(y));
        sum = bin(&mut e, Ty::F32, BinOp::Add, copy(sum), copy(product));
    }
    e.assign(Place::local(dst), Rvalue::Use(copy(sum)));
    e.take()
}

/// The pairs a schedule that multiplies operand terms `a` by `b` adds the product of: every pair, `a` terms
/// outermost, the order the oracle sums them in.
pub(crate) fn pair_products<'t>(
    a: &'t [Local],
    b: &'t [Local],
) -> impl Iterator<Item = (Local, Local)> + 't {
    a.iter().flat_map(move |&x| b.iter().map(move |&y| (x, y)))
}
