//! Operand precision in generated bodies (POOT-1430): the one place a kernel turns a value it read into the
//! terms its products multiply.
//!
//! Only a matrix-unit schedule multiplies narrowed or split operands (the planner refuses every other
//! schedule a narrowed precision, POOT-1476): the fragment-tiled contraction nests and the fragment attention
//! read each operand through [`operand_terms`]. A Gemv, a scalar micro-tile, the serial matmul and the
//! streaming attention multiply exact operands and refuse any other precision ([`require_exact`]). The
//! definition is `poot-eval`'s: round to nearest even, non-saturating, `lo = rne(x - hi)` for a finite `hi` and
//! `0` for an infinite or NaN one.

use poot_graph_ir::{FragmentDtype, OperandPrecision};
use poot_kernel_ir::{BinOp, Constant, Local, Operand, Rvalue, Ty};
use poot_target::RoundingConversions;

use crate::KernelGenError;
use crate::emit::{Emit, bin, c32, copy};
use crate::helpers::{Alloc, WeightLane};

/// How a body reads its two operands: the declared precision, the fragment dtype each operand is already stored
/// in, if it is (an operand stored in the fragment dtype is exact in it and enters once), and how the device rounds
/// to a fragment dtype ([`round_to_fragment`]). The activation of a generated contraction is f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProductRead {
    pub(crate) precision: OperandPrecision,
    pub(crate) activation_stored: Option<FragmentDtype>,
    pub(crate) weight_stored: Option<FragmentDtype>,
    pub(crate) conversions: RoundingConversions,
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

    /// The Kernel IR operations a staged f32 value takes to become the operand a tile stores: the rounding to the
    /// fragment dtype ([`round_to_fragment`] under `conversions`, counted from the statements it emits), once more
    /// for each extra part of a split operand, `0` for exact operands. What a tiling's cost model adds to every
    /// staged value, so a tile that stages many values per multiply (a small one) pays for the narrowing it repeats.
    pub(crate) fn rounding_ops(self, conversions: RoundingConversions) -> usize {
        let Some(dtype) = self.fragment else {
            return 0;
        };
        rounding_statements(conversions, dtype) * (self.activation + self.weight) / 2
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

/// The bits one value of a dense weight held in `storage` occupies in global memory
/// ([`crate::TiledShape::weight_bits`]): 16 for a packed half, 32 for an f32 word.
pub fn dense_value_bits(storage: poot_target::BufferStorage) -> u32 {
    match stored_fragment_dtype(storage) {
        Some(_) => 16,
        None => 32,
    }
}

/// The bits one activation value occupies in global memory ([`crate::TiledShape::activation_bits`]): 32 for an f32
/// activation, 16 for one the planner stored in a fragment dtype (POOT-1448).
pub const fn activation_bits(stored: Option<FragmentDtype>) -> u32 {
    match stored {
        Some(_) => 16,
        None => 32,
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

/// The typed refusal of a schedule that multiplies its operands as read (a Gemv, a register-tiled scalar micro-tile)
/// when `precision` is narrowed or split: only a matrix unit multiplies narrowed operands (the planner's
/// `NoMatrixUnitSchedule` refusal), so no scalar body rounds them.
pub(crate) fn require_exact(
    precision: OperandPrecision,
    request: &'static str,
) -> Result<(), KernelGenError> {
    if precision.is_exact() {
        Ok(())
    } else {
        Err(KernelGenError::Unsupported {
            request,
            reason: "a narrowed or split precision is a matrix-unit schedule's: a scalar schedule multiplies exact operands",
        })
    }
}

impl ProductRead {
    /// The tiles this read stages.
    pub(crate) fn staged_terms(self) -> StagedTerms {
        StagedTerms::of(self.precision, self.activation_stored, self.weight_stored)
    }

    /// A read of a weight decoded to f32 from a packed format or held as f32 words: never stored in the
    /// fragment dtype, so a narrowed or split precision rounds it.
    pub(crate) const fn decoded(
        precision: OperandPrecision,
        conversions: RoundingConversions,
    ) -> Self {
        Self {
            precision,
            activation_stored: None,
            weight_stored: None,
            conversions,
        }
    }

    /// A read of a weight held in the dense `lane` storage.
    pub(crate) fn dense(
        precision: OperandPrecision,
        lane: WeightLane,
        conversions: RoundingConversions,
    ) -> Self {
        Self {
            precision,
            activation_stored: None,
            weight_stored: match lane {
                WeightLane::F32 => None,
                WeightLane::F16Packed => Some(FragmentDtype::F16),
                WeightLane::Bf16Packed => Some(FragmentDtype::Bf16),
            },
            conversions,
        }
    }

    /// The terms of an activation value.
    pub(crate) fn activation_terms(self, e: &mut Emit, x: Local) -> Vec<Local> {
        operand_terms(
            e,
            self.precision,
            self.activation_stored,
            self.conversions,
            x,
        )
    }

    /// The terms of a decoded weight value.
    pub(crate) fn weight_terms(self, e: &mut Emit, x: Local) -> Vec<Local> {
        operand_terms(e, self.precision, self.weight_stored, self.conversions, x)
    }
}

/// The statements [`round_to_fragment`] emits for one value of `dtype` under `conversions`, each into a fresh
/// local: counted from the emitter itself, so a size or cost bound cannot drift from it. The integer form
/// (`RoundingConversions::NONE`) is the larger, so it bounds every form.
pub(crate) fn rounding_statements(conversions: RoundingConversions, dtype: FragmentDtype) -> usize {
    let mut al = Alloc::new(Vec::new());
    let mut e = Emit::new(&mut al);
    let x = e.fresh(Ty::F32, false);
    round_to_fragment(&mut e, conversions, dtype, x);
    e.take().len()
}

/// The Kernel IR element type that holds values of `dtype`.
pub(crate) const fn fragment_ty(dtype: FragmentDtype) -> Ty {
    match dtype {
        FragmentDtype::Bf16 => Ty::BF16,
        FragmentDtype::F16 => Ty::F16,
    }
}

/// The element type of an output stored as `store` says (POOT-1448): f32, or the fragment dtype of a body whose
/// output is a [`crate::FusedScalarOp::Narrow`] step.
pub(crate) const fn stored_ty(store: Option<FragmentDtype>) -> Ty {
    match store {
        Some(dtype) => fragment_ty(dtype),
        None => Ty::F32,
    }
}

/// The element a body stores for its f32 output `x` under `store`: `x` itself, or `x` narrowed to the fragment
/// dtype. A narrowed output is a [`crate::FusedScalarOp::Narrow`] step, already rounded by [`round_to_fragment`]
/// and so exact in the dtype: the narrowing drops zero bits ([`Rvalue::Narrow`]; the interpreter faults on one that
/// would round).
pub(crate) fn store_value(e: &mut Emit, store: Option<FragmentDtype>, x: Local) -> Local {
    match store {
        Some(dtype) => e.let_(
            fragment_ty(dtype),
            Rvalue::Narrow {
                to: fragment_ty(dtype),
                operand: copy(x),
            },
        ),
        None => x,
    }
}

/// `x` rounded to `dtype` (nearest even, ties to even, non-saturating) and widened back to f32, bit for bit the
/// oracle's `round_to_fragment` for every value but a NaN, which becomes a quiet NaN of `dtype` (the oracle keeps
/// part of its payload; a product is NaN either way). Every result is a value of `dtype`, so a staging step narrows
/// it exactly ([`poot_kernel_ir::Rvalue::Narrow`]) and the operand is rounded once.
///
/// Two forms, the same bits (POOT-1458):
///
/// - Where the device keeps the conversion (`conversions.keeps(dtype)`: f16 on AMDGPU and NVPTX), a `Cast` to
///   `dtype` and a `Cast` back: one hardware round-to-nearest-even conversion and one widening, which `llc` never
///   folds into the identity. A staging step's `Narrow` of the widened value folds into the conversion.
/// - Anywhere else, integer arithmetic on the f32 bits. A narrow-and-widen pair of casts is a no-op a driver
///   compiler may keep as f32 (RADV does: `f32(f16(65519.0))` comes back `65519.0` on gfx1151), and there is no
///   `Ty::BF16` on SPIR-V at all. Integer bit operations round the same value to the same bits on every target,
///   including the kernel-IR interpreter. A NaN becomes [`QUIET_NAN`] with its sign.
pub(crate) fn round_to_fragment(
    e: &mut Emit,
    conversions: RoundingConversions,
    dtype: FragmentDtype,
    x: Local,
) -> Local {
    if conversions.keeps(dtype) {
        let narrow = e.let_(
            fragment_ty(dtype),
            Rvalue::Cast {
                to: fragment_ty(dtype),
                operand: copy(x),
            },
        );
        return e.let_(
            Ty::F32,
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(narrow),
            },
        );
    }
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

/// The magnitude bits a rounded NaN takes: the quiet NaN whose payload is all in the top bits, a value of bf16 and
/// of f16 alike.
const QUIET_NAN: u32 = 0x7fc0_0000;

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
/// binade, and past the largest finite bf16 it lands on infinity. A NaN becomes [`QUIET_NAN`].
fn round_bf16_magnitude(e: &mut Emit, magnitude: Local) -> Local {
    let lsb = bin(e, Ty::U32, BinOp::Shr, copy(magnitude), c32(16));
    let lsb = bin(e, Ty::U32, BinOp::BitAnd, copy(lsb), c32(1));
    let bias = bin(e, Ty::U32, BinOp::Add, copy(lsb), c32(0x7fff));
    let sum = bin(e, Ty::U32, BinOp::Add, copy(magnitude), copy(bias));
    let rounded = bin(e, Ty::U32, BinOp::BitAnd, copy(sum), c32(0xffff_0000));
    let is_nan = compare(e, BinOp::Gt, magnitude, 0x7f80_0000);
    let nan = constant(e, QUIET_NAN);
    pick(e, is_nan, nan, rounded)
}

/// The magnitude bits of an f32 rounded to a binary16 (11 significant bits in the normal range, a fixed spacing of
/// `2^-24` below it), then widened: ties to even, an overflow to infinity (an infinity stays one), a NaN
/// [`QUIET_NAN`].
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

    // Compose, innermost first: zero at or below the tie, then the subnormal and normal ranges (an infinity is
    // normal: its bias add carries nothing out and it stays above the overflow bound), then a NaN.
    let zero = constant(e, 0);
    let small = pick(e, in_subnormal, subnormal, zero);
    let finite = {
        let is_normal = compare(e, BinOp::Ge, magnitude, 0x3880_0000);
        pick(e, is_normal, normal, small)
    };
    let is_nan = compare(e, BinOp::Gt, magnitude, 0x7f80_0000);
    let nan = constant(e, QUIET_NAN);
    pick(e, is_nan, nan, finite)
}

/// The terms the f32 value `x` enters the products as under `precision`: `x` itself when the precision is exact or
/// the operand is stored in the fragment dtype (`stored`), its rounded value for a narrowed operand, and `hi`, `lo`
/// for a split one, each rounded as `conversions` says.
pub(crate) fn operand_terms(
    e: &mut Emit,
    precision: OperandPrecision,
    stored: Option<FragmentDtype>,
    conversions: RoundingConversions,
    x: Local,
) -> Vec<Local> {
    let Some(dtype) = precision.fragment_dtype() else {
        return vec![x];
    };
    if stored == Some(dtype) {
        return vec![x];
    }
    let hi = round_to_fragment(e, conversions, dtype, x);
    let OperandPrecision::Split(_) = precision else {
        return vec![hi];
    };
    // `lo` is zero for a non-finite `hi`: `hi - hi` is `0` exactly when `hi` is finite.
    let residual = bin(e, Ty::F32, BinOp::Sub, copy(x), copy(hi));
    let lo_raw = round_to_fragment(e, conversions, dtype, residual);
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
    conversions: RoundingConversions,
    x: Local,
) -> Local {
    debug_assert!(
        !matches!(precision, OperandPrecision::Split(_)),
        "a single staged tile has no lo part"
    );
    operand_terms(e, precision, stored, conversions, x)[0]
}

#[cfg(test)]
mod tests {
    use poot_kernel_ir::interp::{Buffer, run};
    use poot_kernel_ir::{BasicBlock, BlockId, Body, IndexAxis, Place, Terminator};
    use poot_target::FragmentLayout;

    use super::*;
    use crate::helpers::{elem, ld, local, slice_dtype, slice_f32};
    use crate::microtile::{FragmentTile, MicroTile};

    /// A fragment tile's staging step of one narrowed value per lane, as the tiled schedules emit it under
    /// `conversions`: `out[i] = stage(narrowed(input[i]))`.
    fn staging_body(dtype: FragmentDtype, conversions: RoundingConversions, lanes: u32) -> Body {
        let micro = MicroTile::Fragment(FragmentTile {
            dtype,
            layout: FragmentLayout::Opaque,
            rows: 1,
            cols: 1,
        });
        let mut al = Alloc::new(vec![
            ld(Ty::Unit, false),
            ld(slice_f32(false), false),
            ld(slice_dtype(micro.staged_ty(), true), true),
        ]);
        let (input, out) = (local(1), local(2));
        let lane = al.add(Ty::Usize, false);
        let mut e = Emit::new(&mut al);
        let x = e.let_(Ty::F32, Rvalue::Use(Operand::Copy(elem(input, lane))));
        let rounded = narrowed(
            &mut e,
            OperandPrecision::Narrow(dtype),
            None,
            conversions,
            x,
        );
        let staged = micro.stage_value(&mut e, rounded);
        e.assign(elem(out, lane), Rvalue::Use(copy(staged)));
        let blocks = vec![
            BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(lane),
                    dim: IndexAxis::X,
                    target: BlockId { index: 1 },
                },
            },
            BasicBlock {
                statements: e.take(),
                terminator: Terminator::Return,
            },
        ];
        let mut body = Body::new("staging_probe", 2, al.locals, blocks);
        body.workgroup_size = [lanes, 1, 1];
        body
    }

    /// The staging step rounds once and narrows exactly, in both forms (the integer form, and the kept f16
    /// conversion of AMDGPU and NVPTX): every staged value is the oracle's rounding, a NaN of any payload (one in
    /// the bits the narrow type drops included) stages as a NaN, and no value makes the exact narrowing fault.
    /// Mutation: make `round_bf16_magnitude` keep a NaN's bits (`pick(e, is_nan, magnitude, rounded)`); the NaN
    /// whose payload is in the low bits faults the narrowing (`InexactNarrow`) and this row goes red. For the
    /// conversion form, the interpreter's `Cast` to f16 keeping a NaN's whole payload faults it the same way.
    #[test]
    fn a_staged_operand_is_rounded_once_and_narrowed_exactly() {
        let inputs = [
            1.0 + 2.0f32.powi(-9),
            -65519.0,
            1.0e9,
            2.0f32.powi(-25),
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0xffc0_1234),
            f32::INFINITY,
        ];
        let forms = [RoundingConversions::NONE, RoundingConversions::F16];
        for (dtype, conversions) in [FragmentDtype::Bf16, FragmentDtype::F16]
            .into_iter()
            .flat_map(|d| forms.map(|c| (d, c)))
        {
            let body = staging_body(dtype, conversions, inputs.len() as u32);
            let staged_ty = match dtype {
                FragmentDtype::Bf16 => Ty::BF16,
                FragmentDtype::F16 => Ty::F16,
            };
            let mut buffers = [
                Buffer::from_f32s(&inputs),
                Buffer::from_halves(staged_ty, &vec![0.0; inputs.len()]),
            ];
            run(&body, [1, 1, 1], &mut buffers)
                .unwrap_or_else(|e| panic!("{dtype:?} {conversions:?}: {e}"));
            for (&x, staged) in inputs.iter().zip(buffers[1].to_halves().unwrap()) {
                let want = poot_eval::round_to_fragment(dtype, x);
                if want.is_nan() {
                    assert!(
                        staged.is_nan(),
                        "{dtype:?} {conversions:?} {x:e}: staged {staged:e}"
                    );
                } else {
                    assert_eq!(
                        staged.to_bits(),
                        want.to_bits(),
                        "{dtype:?} {conversions:?} {x:e}"
                    );
                }
            }
        }
    }

    /// The f32 bit patterns one sweep dispatch covers, and the lanes of one of its workgroups.
    pub(super) const SWEEP_CHUNK: u32 = 1 << 24;
    pub(super) const SWEEP_LANES: u32 = 256;

    /// What a sweep compares in its two forms (the integer arithmetic and the kept f16 conversion).
    #[derive(Clone, Copy, Debug)]
    pub(super) enum Sweep {
        /// The rounding of the f32 whose bits are the pattern to f16 ([`round_to_fragment`]); every pattern.
        Rounding,
        /// The decode of the f16 whose bits are the pattern to f32 (`WeightLane::decode_low_half`, POOT-1458); the
        /// finite f16 patterns, the packed-f16 lane's domain.
        Decode,
    }

    /// The exhaustive sweep body (POOT-1458, design section 3.4): lane `i` takes the bit pattern `meta[0] + i`,
    /// computes `sweep` both ways (the integer form and the kept conversion), and where the two differ (a NaN
    /// matches any NaN) counts one in `out[0]` and leaves the pattern in `out[1]`.
    pub(super) fn sweep_body(sweep: Sweep) -> Body {
        let mut al = Alloc::new(vec![
            ld(Ty::Unit, false),
            ld(slice_dtype(Ty::U32, false), false),
            ld(slice_dtype(Ty::U32, true), true),
        ]);
        let (meta, out) = (local(1), local(2));
        let lane = al.add(Ty::Usize, false);
        let mut e = Emit::new(&mut al);
        let zero = e.let_(Ty::Usize, Rvalue::Use(Operand::Const(Constant::Usize(0))));
        let base = e.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(meta, zero))));
        let offset = e.let_(
            Ty::U32,
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(lane),
            },
        );
        let bits = bin(&mut e, Ty::U32, BinOp::Add, copy(base), copy(offset));
        let x = e.let_(
            Ty::F32,
            Rvalue::Bitcast {
                to: Ty::F32,
                operand: copy(bits),
            },
        );
        let form = |e: &mut Emit, conversions| match sweep {
            Sweep::Rounding => round_to_fragment(e, conversions, FragmentDtype::F16, x),
            Sweep::Decode => WeightLane::F16Packed.decode_low_half(e, conversions, bits),
        };
        let integer = form(&mut e, RoundingConversions::NONE);
        let converted = form(&mut e, RoundingConversions::F16);
        let as_bits = |e: &mut Emit, v: Local| {
            e.let_(
                Ty::U32,
                Rvalue::Bitcast {
                    to: Ty::U32,
                    operand: copy(v),
                },
            )
        };
        let (integer_bits, converted_bits) = (as_bits(&mut e, integer), as_bits(&mut e, converted));
        let differ = bin(
            &mut e,
            Ty::Bool,
            BinOp::Ne,
            copy(integer_bits),
            copy(converted_bits),
        );
        // A NaN by its bits: two NaNs of different payload differ bitwise but agree as "a NaN".
        let nan = |e: &mut Emit, bits: Local| {
            let magnitude = bin(e, Ty::U32, BinOp::BitAnd, copy(bits), c32(0x7fff_ffff));
            compare(e, BinOp::Gt, magnitude, 0x7f80_0000)
        };
        let (integer_nan, converted_nan) = (nan(&mut e, integer_bits), nan(&mut e, converted_bits));
        let both_nan = bin(
            &mut e,
            Ty::Bool,
            BinOp::BitAnd,
            copy(integer_nan),
            copy(converted_nan),
        );
        let as_u32 = |e: &mut Emit, b: Local| {
            e.let_(
                Ty::U32,
                Rvalue::Cast {
                    to: Ty::U32,
                    operand: copy(b),
                },
            )
        };
        let (differ, both_nan) = (as_u32(&mut e, differ), as_u32(&mut e, both_nan));
        let mismatch = bin(&mut e, Ty::Bool, BinOp::Gt, copy(differ), copy(both_nan));
        let mismatch = match sweep {
            Sweep::Rounding => mismatch,
            Sweep::Decode => {
                // A finite f16: a half (the high bits zero) whose exponent is not all ones.
                let half = compare(&mut e, BinOp::Lt, bits, 0x1_0000);
                let exponent = bin(&mut e, Ty::U32, BinOp::BitAnd, copy(bits), c32(0x7c00));
                let finite = compare(&mut e, BinOp::Ne, exponent, 0x7c00);
                let domain = bin(&mut e, Ty::Bool, BinOp::BitAnd, copy(half), copy(finite));
                bin(
                    &mut e,
                    Ty::Bool,
                    BinOp::BitAnd,
                    copy(domain),
                    copy(mismatch),
                )
            }
        };
        let compare = e.take();
        let mut e = Emit::new(&mut al);
        let one = e.let_(Ty::Usize, Rvalue::Use(Operand::Const(Constant::Usize(1))));
        e.let_(
            Ty::U32,
            Rvalue::GlobalAtomic {
                place: elem(out, zero),
                value: Operand::Const(Constant::U32(1)),
                op: poot_kernel_ir::AtomicOp::Add,
            },
        );
        e.let_(
            Ty::U32,
            Rvalue::GlobalAtomic {
                place: elem(out, one),
                value: copy(bits),
                op: poot_kernel_ir::AtomicOp::Exchange,
            },
        );
        let record = e.take();
        let blocks = vec![
            BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(lane),
                    dim: IndexAxis::X,
                    target: BlockId { index: 1 },
                },
            },
            BasicBlock {
                statements: compare,
                terminator: crate::helpers::guard(mismatch, 3, 2),
            },
            BasicBlock {
                statements: record,
                terminator: Terminator::Goto {
                    target: BlockId { index: 3 },
                },
            },
            BasicBlock {
                statements: vec![],
                terminator: Terminator::Return,
            },
        ];
        let mut body = Body::new(format!("f16_{sweep:?}_sweep"), 2, al.locals, blocks);
        body.workgroup_size = [SWEEP_LANES, 1, 1];
        body
    }

    /// The rounding sweep on the interpreter, over a few patterns of each kind (normal, subnormal, ties, overflow,
    /// infinities, NaNs of every payload), and the decode sweep over every f16: no mismatch. What the device rows
    /// hold over every pattern. Mutation: make `round_f16_magnitude`'s subnormal tie round up instead of to even
    /// (`up = past_half | at_half`); the subnormal ties mismatch and this row goes red.
    #[test]
    fn the_sweep_bodies_find_no_mismatch_on_the_interpreter() {
        let decode = sweep_body(Sweep::Decode);
        let mut buffers = [Buffer::from_u32s(&[0]), Buffer::from_u32s(&[0, 0])];
        run(&decode, [(1 << 16) / SWEEP_LANES, 1, 1], &mut buffers).unwrap();
        let out = buffers[1].to_u32s().unwrap();
        assert_eq!(
            out[0], 0,
            "decode: {} f16 patterns differ, e.g. {:#06x}",
            out[0], out[1]
        );
        let body = sweep_body(Sweep::Rounding);
        for base in [
            0x0000_0000u32,
            0x3300_0000 - 128,
            0x3380_0000 - 128,
            0x3880_0000 - 128,
            0x3f80_1000 - 128,
            0x477f_f000 - 128,
            0x7f80_0000 - 128,
            0xffc0_0000 - 128,
        ] {
            let mut buffers = [Buffer::from_u32s(&[base]), Buffer::from_u32s(&[0, 0])];
            run(&body, [1, 1, 1], &mut buffers).unwrap();
            let out = buffers[1].to_u32s().unwrap();
            assert_eq!(
                out[0], 0,
                "base {base:#010x}: {} mismatches, e.g. {:#010x}",
                out[0], out[1]
            );
        }
    }
}

/// POOT-1458 (design section 3.4): the kept f16 conversion rounds every one of the 2^32 f32 bit patterns to the
/// bits the integer form does, on a real ROCm device (gfx1151): `v_cvt_f16_f32` is round to nearest even with f16
/// subnormals kept, and `llc` keeps the narrow-and-widen pair. Skips (reported) with no HSA library or GPU agent;
/// run serial under `/tmp/poot-gpu.lock`.
#[cfg(test)]
mod rocm_sweep {
    use poot_rocm_runtime::{RocmContext, RocmError};
    use poot_runtime_common::BufferRole;
    use poot_target::ElementKind;

    use super::tests::{SWEEP_CHUNK, SWEEP_LANES, Sweep, sweep_body};

    /// Mutation: make the kept conversion truncate (in `round_to_fragment`, `Cast` the f32 whose bits are
    /// `x & 0xffff_e000` instead of `x`); 260031486 patterns mismatch and this row goes red.
    #[test]
    fn the_f16_conversion_rounds_every_f32_as_the_integer_form_on_rocm() {
        if let Some((count, example)) = sweep_on_rocm(Sweep::Rounding, 1 << 32) {
            assert_eq!(
                count, 0,
                "{count} of 2^32 f32 patterns round differently, e.g. {example:#010x}"
            );
        }
    }

    /// The hardware decode of a packed f16 (`FromLowBits` and a widening `Cast`) gives every finite f16 the f32 the
    /// integer decode gives it, subnormals and signed zeros included, on a real ROCm device. Mutation: decode the
    /// half as a bf16 (`FromLowBits { to: BF16 }`) in `decode_low_half`; nearly every pattern mismatches and this
    /// row goes red.
    #[test]
    fn the_f16_conversion_decodes_every_finite_f16_as_the_integer_form_on_rocm() {
        if let Some((count, example)) = sweep_on_rocm(Sweep::Decode, 1 << 16) {
            assert_eq!(
                count, 0,
                "{count} finite f16 patterns decode differently, e.g. {example:#06x}"
            );
        }
    }

    /// `sweep` over the first `patterns` bit patterns on ROCm: `(mismatches, an example)`, or `None` (a reported
    /// skip) with no HSA library or GPU agent.
    fn sweep_on_rocm(sweep: Sweep, patterns: u64) -> Option<(u32, u32)> {
        let ctx = match RocmContext::new() {
            Ok(ctx) => ctx,
            Err(RocmError::LibraryNotFound(_) | RocmError::NoGpuAgent) => {
                poot_test_util::skip_report(
                    "poot-kernelgen",
                    &format_args!("the {sweep:?} sweep on ROCm: no ROCm device"),
                );
                return None;
            }
            Err(error) => panic!("{error}"),
        };
        let body = sweep_body(sweep);
        let arch = poot_target::AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront())
            .expect("a classified device arch");
        let dir = std::env::temp_dir().join("poot-kernelgen-f16-sweep");
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, &body.name, poot_codegen::Target::AmdGcn(arch));
        let artifact =
            poot_codegen::compile_kernel(&body, poot_target::Backend::AmdGcn(arch), &out)
                .expect("compile");
        let compiled = poot_codegen::kernel_handle(&body, &artifact).unwrap();
        let module = ctx.load_hsaco(&compiled).expect("load_hsaco");
        let kernel = ctx
            .lookup_kernel(&module, compiled.entry_point())
            .expect("lookup_kernel");
        let words =
            |data: &[u32]| -> Vec<f32> { data.iter().map(|&w| f32::from_bits(w)).collect() };
        let result = ctx
            .upload_f32(&words(&[0, 0]), BufferRole::Output)
            .expect("upload");
        let chunk_len = u64::from(SWEEP_CHUNK).min(patterns);
        for chunk in 0..patterns / chunk_len {
            let base = (chunk * chunk_len) as u32;
            let meta = ctx
                .upload_f32(&words(&[base]), BufferRole::Input)
                .expect("upload");
            let kernarg = ctx
                .bind_kernarg(&kernel, &[(&meta, 1), (&result, 2)])
                .expect("bind kernarg");
            ctx.dispatch(
                &kernel,
                &kernarg,
                &[ElementKind::I32, ElementKind::I32],
                (chunk_len as u32, 1, 1),
                (SWEEP_LANES, 1, 1),
            )
            .expect("dispatch");
        }
        let mut got = [0.0f32; 2];
        ctx.download_f32(&result, &mut got).expect("download");
        Some((got[0].to_bits(), got[1].to_bits()))
    }
}

/// POOT-1459 (design section 3.4): the PTX twin of [`rocm_sweep`]. NVPTX lowers the kept conversion to
/// `cvt.rn.f16.f32` (round to nearest even, f16 subnormals kept) and `cvt.f32.f16`; these rows run the same sweep
/// bodies over every pattern on a real NVIDIA device. Skips (reported) with no NVIDIA device; the PTX lane
/// (`just test-device-ptx`) requires them.
#[cfg(test)]
mod ptx_sweep {
    use poot_ptx_runtime::PtxContext;

    use super::tests::{SWEEP_CHUNK, SWEEP_LANES, Sweep, sweep_body};

    /// Mutation: make the kept conversion truncate (in `round_to_fragment`, `Cast` the f32 whose bits are
    /// `x & 0xffff_e000` instead of `x`); the truncated patterns mismatch and this row goes red.
    #[test]
    fn the_f16_conversion_rounds_every_f32_as_the_integer_form_on_ptx() {
        if let Some((count, example)) = sweep_on_ptx(Sweep::Rounding, 1 << 32) {
            assert_eq!(
                count, 0,
                "{count} of 2^32 f32 patterns round differently, e.g. {example:#010x}"
            );
        }
    }

    /// The hardware decode of a packed f16 (`FromLowBits` and a widening `Cast`) gives every finite f16 the f32 the
    /// integer decode gives it, subnormals and signed zeros included, on a real NVIDIA device. Mutation: decode the
    /// half as a bf16 (`FromLowBits { to: BF16 }`) in `decode_low_half`; nearly every pattern mismatches and this
    /// row goes red.
    #[test]
    fn the_f16_conversion_decodes_every_finite_f16_as_the_integer_form_on_ptx() {
        if let Some((count, example)) = sweep_on_ptx(Sweep::Decode, 1 << 16) {
            assert_eq!(
                count, 0,
                "{count} finite f16 patterns decode differently, e.g. {example:#06x}"
            );
        }
    }

    /// `sweep` over the first `patterns` bit patterns on PTX: `(mismatches, an example)`, or `None` (a reported
    /// skip) with no NVIDIA device.
    fn sweep_on_ptx(sweep: Sweep, patterns: u64) -> Option<(u32, u32)> {
        let ctx = match PtxContext::new() {
            Ok(ctx) => ctx,
            Err(error) => {
                poot_test_util::skip_report(
                    "poot-kernelgen",
                    &format_args!("the {sweep:?} sweep on PTX: no PTX device ({error})"),
                );
                return None;
            }
        };
        let body = sweep_body(sweep);
        let dir = std::env::temp_dir().join("poot-kernelgen-f16-sweep-ptx");
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, &body.name, poot_codegen::Target::Nvptx);
        let artifact = poot_codegen::compile_kernel(&body, poot_target::Backend::Nvptx, &out)
            .expect("compile");
        let kernel = poot_codegen::kernel_handle(&body, &artifact).unwrap();
        let result = ctx.upload_i32(&[0, 0]).expect("upload");
        let chunk_len = u64::from(SWEEP_CHUNK).min(patterns);
        for chunk in 0..patterns / chunk_len {
            let base = (chunk * chunk_len) as u32;
            let meta = ctx.upload_i32(&[base as i32]).expect("upload");
            ctx.dispatch_dev(
                "f16_sweep",
                &kernel,
                [SWEEP_LANES, 1, 1],
                [chunk_len as u32, 1, 1],
                &[&meta],
                &[1],
                &result,
                2,
            )
            .expect("dispatch");
        }
        let got = ctx.download_i32(&result).expect("download");
        Some((got[0] as u32, got[1] as u32))
    }
}
