//! `emit_float_decode`: [`poot_quant::scalar::float_to_f32`] as integer ops and one bitcast, for
//! every [`FloatFormat`]. This is the one place a narrow float's bit layout is stated in KIR: every
//! packed kernel body (and, from Card 559 on, region lowering's E4M3 operand decode) calls it rather
//! than restating a format's exponent bias and mantissa width.
//!
//! Every decoder here implements only the *admitted* domain: `PackedPayload::try_new`'s content
//! validation (`poot-quant`) refuses every non-finite `Float`-encoded value before a payload exists,
//! so no kernel ever sees an E8M0 `0xff`, an E4M3FN NaN byte, or an F16/BF16 infinity or NaN. Each
//! decoder therefore has exactly the branches its finite domain needs (design dquant.md 4; rejected:
//! a wasted branch, or worse a wrong reconstruction, for a byte that can never arrive - the bug this
//! design replaces, `dequant_mxfp4.rs`'s ggml half-scale E8M0 form).

use poot_kernel_ir::{BinOp, Local, Rvalue, Ty};
use poot_quant::format::FloatFormat;

use crate::emit::{Emit, bin, c32, copy, select_u32};

pub(crate) fn emit_float_decode(e: &mut Emit, format: FloatFormat, bits: Local) -> Local {
    match format {
        FloatFormat::F32 => bitcast_f32(e, bits),
        FloatFormat::F16 => emit_f16(e, bits),
        FloatFormat::Bf16 => emit_bf16(e, bits),
        FloatFormat::E4m3Fn => emit_e4m3fn(e, bits),
        FloatFormat::E2m1 => emit_e2m1(e, bits),
        FloatFormat::E8m0 => emit_e8m0(e, bits),
    }
}

fn bitcast_f32(e: &mut Emit, bits: Local) -> Local {
    e.let_(
        Ty::F32,
        Rvalue::Bitcast {
            to: Ty::F32,
            operand: copy(bits),
        },
    )
}

/// `crate::packed` never reads more than 32 bits of a code in one field, but the mantissa->f32
/// widen below is always exact (mantissa fits well under 2^24) regardless of the source format.
fn uitofp(e: &mut Emit, value: Local) -> Local {
    e.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(value),
        },
    )
}

/// bfloat16 to f32: bf16 is the high half of an f32 (`scalar::bf16_to_f32`), no branch.
fn emit_bf16(e: &mut Emit, bits: Local) -> Local {
    let widened = bin(e, Ty::U32, BinOp::Shl, copy(bits), c32(16));
    bitcast_f32(e, widened)
}

/// IEEE-754 binary16 to f32 (`scalar::f16_to_f32`), the zero/subnormal and normal cases (the
/// admitted domain excludes exponent `0x1f`, infinities and NaN).
fn emit_f16(e: &mut Emit, bits: Local) -> Local {
    let sign = {
        let masked = bin(e, Ty::U32, BinOp::BitAnd, copy(bits), c32(0x8000));
        bin(e, Ty::U32, BinOp::Shl, copy(masked), c32(16))
    };
    let exponent = {
        let shifted = bin(e, Ty::U32, BinOp::Shr, copy(bits), c32(10));
        bin(e, Ty::U32, BinOp::BitAnd, copy(shifted), c32(0x1f))
    };
    let mantissa = bin(e, Ty::U32, BinOp::BitAnd, copy(bits), c32(0x03ff));
    let is_subnormal = bin(e, Ty::Bool, BinOp::Eq, copy(exponent), c32(0));

    // subnormal: mantissa * 2^-24, sign applied to the resulting bits.
    let mantissa_f32 = uitofp(e, mantissa);
    let magnitude = e.let_(
        Ty::F32,
        Rvalue::BinaryOp(
            BinOp::Mul,
            copy(mantissa_f32),
            poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::F32(1.0 / 16_777_216.0)),
        ),
    );
    let magnitude_bits = e.let_(
        Ty::U32,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: copy(magnitude),
        },
    );
    let subnormal_bits = bin(e, Ty::U32, BinOp::BitOr, copy(magnitude_bits), copy(sign));

    // normal: sign | (exponent+112)<<23 | mantissa<<13.
    let biased = bin(e, Ty::U32, BinOp::Add, copy(exponent), c32(112));
    let exponent_bits = bin(e, Ty::U32, BinOp::Shl, copy(biased), c32(23));
    let mantissa_bits = bin(e, Ty::U32, BinOp::Shl, copy(mantissa), c32(13));
    let normal_bits = {
        let combined = bin(e, Ty::U32, BinOp::BitOr, copy(sign), copy(exponent_bits));
        bin(
            e,
            Ty::U32,
            BinOp::BitOr,
            copy(combined),
            copy(mantissa_bits),
        )
    };

    let result_bits = select_u32(e, is_subnormal, subnormal_bits, normal_bits);
    bitcast_f32(e, result_bits)
}

/// OCP E4M3FN to f32 (`scalar::e4m3fn_to_f32`), the zero/subnormal and normal cases (the admitted
/// domain excludes `0x7f`/`0xff`, the two NaN bytes).
fn emit_e4m3fn(e: &mut Emit, bits: Local) -> Local {
    let sign = {
        let masked = bin(e, Ty::U32, BinOp::BitAnd, copy(bits), c32(0x80));
        bin(e, Ty::U32, BinOp::Shl, copy(masked), c32(24))
    };
    let exponent = {
        let shifted = bin(e, Ty::U32, BinOp::Shr, copy(bits), c32(3));
        bin(e, Ty::U32, BinOp::BitAnd, copy(shifted), c32(0x0f))
    };
    let mantissa = bin(e, Ty::U32, BinOp::BitAnd, copy(bits), c32(0x07));
    let is_subnormal = bin(e, Ty::Bool, BinOp::Eq, copy(exponent), c32(0));

    // subnormal: mantissa * 2^-9, sign applied to the resulting bits.
    let mantissa_f32 = uitofp(e, mantissa);
    let magnitude = e.let_(
        Ty::F32,
        Rvalue::BinaryOp(
            BinOp::Mul,
            copy(mantissa_f32),
            poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::F32(1.0 / 512.0)),
        ),
    );
    let magnitude_bits = e.let_(
        Ty::U32,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: copy(magnitude),
        },
    );
    let subnormal_bits = bin(e, Ty::U32, BinOp::BitOr, copy(magnitude_bits), copy(sign));

    // normal: sign | (exponent+120)<<23 | mantissa<<20.
    let biased = bin(e, Ty::U32, BinOp::Add, copy(exponent), c32(120));
    let exponent_bits = bin(e, Ty::U32, BinOp::Shl, copy(biased), c32(23));
    let mantissa_bits = bin(e, Ty::U32, BinOp::Shl, copy(mantissa), c32(20));
    let normal_bits = {
        let combined = bin(e, Ty::U32, BinOp::BitOr, copy(sign), copy(exponent_bits));
        bin(
            e,
            Ty::U32,
            BinOp::BitOr,
            copy(combined),
            copy(mantissa_bits),
        )
    };

    let result_bits = select_u32(e, is_subnormal, subnormal_bits, normal_bits);
    bitcast_f32(e, result_bits)
}

/// OCP E2M1 (FP4) to f32 (`scalar::e2m1_to_f32`): every code is admitted (E2M1 has no NaN or
/// infinity encoding), so both cases are the complete decoder.
fn emit_e2m1(e: &mut Emit, bits: Local) -> Local {
    let sign = {
        let masked = bin(e, Ty::U32, BinOp::BitAnd, copy(bits), c32(0x08));
        bin(e, Ty::U32, BinOp::Shl, copy(masked), c32(28))
    };
    let exponent = {
        let shifted = bin(e, Ty::U32, BinOp::Shr, copy(bits), c32(1));
        bin(e, Ty::U32, BinOp::BitAnd, copy(shifted), c32(0x03))
    };
    let mantissa = bin(e, Ty::U32, BinOp::BitAnd, copy(bits), c32(0x01));
    let is_subnormal = bin(e, Ty::Bool, BinOp::Eq, copy(exponent), c32(0));

    // subnormal: mantissa * 0.5 (mantissa is 0 or 1: this is either 0.0 or 0.5).
    let mantissa_f32 = uitofp(e, mantissa);
    let magnitude = e.let_(
        Ty::F32,
        Rvalue::BinaryOp(
            BinOp::Mul,
            copy(mantissa_f32),
            poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::F32(0.5)),
        ),
    );
    let subnormal_bits = e.let_(
        Ty::U32,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: copy(magnitude),
        },
    );

    // normal: (exponent+126)<<23 | mantissa<<22 (unsigned magnitude; sign is ORed in below).
    let biased = bin(e, Ty::U32, BinOp::Add, copy(exponent), c32(126));
    let exponent_bits = bin(e, Ty::U32, BinOp::Shl, copy(biased), c32(23));
    let mantissa_bits = bin(e, Ty::U32, BinOp::Shl, copy(mantissa), c32(22));
    let normal_bits = bin(
        e,
        Ty::U32,
        BinOp::BitOr,
        copy(exponent_bits),
        copy(mantissa_bits),
    );

    let magnitude_bits = select_u32(e, is_subnormal, subnormal_bits, normal_bits);
    let result_bits = bin(e, Ty::U32, BinOp::BitOr, copy(magnitude_bits), copy(sign));
    bitcast_f32(e, result_bits)
}

/// OCP E8M0 to f32 (`scalar::e8m0_to_f32`): `2^(byte-127)`, `0x00` the one subnormal case. No `0xff`
/// (NaN) branch: the admitted domain is `0..=254` (dquant.md D2/D4; the NaN detector is
/// `Materialize`, not this decoder).
fn emit_e8m0(e: &mut Emit, bits: Local) -> Local {
    let is_zero = bin(e, Ty::Bool, BinOp::Eq, copy(bits), c32(0));
    let normal_bits = bin(e, Ty::U32, BinOp::Shl, copy(bits), c32(23));
    let subnormal_bits = e.let_(Ty::U32, Rvalue::Use(c32(0x0040_0000)));
    let result_bits = select_u32(e, is_zero, subnormal_bits, normal_bits);
    bitcast_f32(e, result_bits)
}
