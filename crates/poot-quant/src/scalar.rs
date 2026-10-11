//! The one bit-exact decoder for each narrow float format a weight code or scale is stored in.
//!
//! Each function is a total conversion of the stored bits to the f32 with the same value: NaN
//! encodings decode to NaN (callers that must refuse NaN check [`f32::is_nan`]), and signed zeros
//! keep their sign.

use crate::format::FloatFormat;

/// IEEE-754 binary16 to f32 (exact; subnormals, infinities and NaN payloads carry through).
pub const fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;
    match exponent {
        0 => {
            // Zero or subnormal: mantissa * 2^-24 is exact in f32.
            let magnitude = mantissa as f32 * (1.0 / 16_777_216.0);
            f32::from_bits(magnitude.to_bits() | sign)
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
    }
}

/// bfloat16 to f32: bf16 is the high half of an f32.
pub const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 -> IEEE-754 binary16 bits, round to nearest with ties to even (ADR-0101 decision 4). Subnormal
/// results are kept, overflow rounds to infinity, and a NaN stays a NaN (sign and top payload bits carry
/// through; the quiet bit is forced so a payload confined to the dropped low bits cannot collapse to
/// infinity). The one host f16 encoder: `poot-eval`'s Cast oracle and the resident uploads of `poot-gpu` and
/// `poot-ptx-gpu` narrow with it, as does [`crate::encode`] for every f16 block scale.
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00 | (mant >> 13) as u16
        };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00; // overflow
    }
    if half_exp <= 0 {
        // Subnormal in f16 (or zero): shift the significand, with its implicit bit, into the 10-bit field.
        if half_exp < -10 {
            return sign; // below half the smallest subnormal
        }
        let full = mant | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = full >> shift;
        let round_bit = 1u32 << (shift - 1);
        // Round up above the tie, or on the tie when the kept mantissa is odd.
        if full & round_bit != 0 && full & (3 * round_bit - 1) != 0 {
            half_mant += 1;
        }
        return sign | half_mant as u16;
    }
    let base = sign | ((half_exp as u16) << 10) | (mant >> 13) as u16;
    // A carry out of the mantissa lands in the exponent, which is the correct next binade (or infinity).
    if mant & 0x1000 != 0 && mant & 0x2fff != 0 {
        base + 1
    } else {
        base
    }
}

/// Round an f32 to bf16 bits (round-to-nearest, ties-to-even) and return the 16-bit pattern.
/// NaN keeps its sign: a quiet NaN stays quiet with its high payload bits, and a signaling NaN is
/// quieted (bf16 has no signaling NaN). Infinities, the sign, and the exponent range carry through;
/// values above bf16's max finite round to infinity.
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        // Keep it a NaN: force the quiet bit (bf16's top mantissa bit) and preserve sign + payload.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

/// OCP E4M3FN to f32: bias 7, no infinities; `0x7f` and `0xff` both decode to the one canonical
/// positive quiet NaN (`0x7fc0_0000`).
pub const fn e4m3fn_to_f32(byte: u8) -> f32 {
    let sign = ((byte & 0x80) as u32) << 24;
    let exponent = ((byte >> 3) & 0x0f) as u32;
    let mantissa = (byte & 0x07) as u32;
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::from_bits(0x7fc0_0000);
    }
    if exponent == 0 {
        // Zero or subnormal: mantissa * 2^-9.
        let magnitude = mantissa as f32 * (1.0 / 512.0);
        return f32::from_bits(magnitude.to_bits() | sign);
    }
    f32::from_bits(sign | ((exponent + 120) << 23) | (mantissa << 20))
}

/// OCP E2M1 (FP4) to f32 from the low four bits of `code`: bias 1, one mantissa bit, no
/// infinities or NaN. Code `0b1000` is `-0.0` (see the MXFP4 note in [`crate::format`]).
pub const fn e2m1_to_f32(code: u8) -> f32 {
    let sign = ((code & 0x08) as u32) << 28;
    let exponent = ((code >> 1) & 0x03) as u32;
    let mantissa = (code & 0x01) as u32;
    let magnitude = if exponent == 0 {
        // Zero or subnormal: mantissa * 0.5.
        mantissa as f32 * 0.5
    } else {
        f32::from_bits(((exponent + 126) << 23) | (mantissa << 22))
    };
    f32::from_bits(magnitude.to_bits() | sign)
}

/// OCP E8M0 to f32: `2^(byte - 127)`; `0x00` is the f32 subnormal `2^-127`; `0xff` is NaN.
pub const fn e8m0_to_f32(byte: u8) -> f32 {
    match byte {
        0x00 => f32::from_bits(0x0040_0000),
        0xff => f32::from_bits(0x7fc0_0000),
        _ => f32::from_bits((byte as u32) << 23),
    }
}

/// Decode `bits` (the low [`FloatFormat::bits`] bits) of a stored float.
pub const fn float_to_f32(format: FloatFormat, bits: u32) -> f32 {
    match format {
        FloatFormat::F32 => f32::from_bits(bits),
        FloatFormat::F16 => f16_to_f32(bits as u16),
        FloatFormat::Bf16 => bf16_to_f32(bits as u16),
        FloatFormat::E4m3Fn => e4m3fn_to_f32(bits as u8),
        FloatFormat::E2m1 => e2m1_to_f32(bits as u8),
        FloatFormat::E8m0 => e8m0_to_f32(bits as u8),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One f16 ulp in `[1, 2)`.
    const ULP_1: f32 = 1.0 / 1024.0;

    /// ADR-0101 decision 4 (R482-003): the encoder rounds to nearest, ties to even. The rows that round up
    /// (0.75 ulp above 1, the odd-mantissa tie) fail under truncation; the exact and round-down rows are
    /// exactness and no-overshoot cases that truncation also passes.
    #[test]
    fn f16_encode_rounds_to_nearest_even() {
        let cases: [(f32, u16, &str); 8] = [
            (
                1.0 + ULP_1,
                0x3c01,
                "exactness case: 1 + 1 ulp is representable (truncation passes too)",
            ),
            (
                1.0 + 0.75 * ULP_1,
                0x3c01,
                "0.75 ulp above 1 rounds up (SC-001 rounding case)",
            ),
            (1.0 + 0.25 * ULP_1, 0x3c00, "0.25 ulp above 1 rounds down"),
            (
                1.0 + 0.5 * ULP_1,
                0x3c00,
                "tie below an even mantissa stays",
            ),
            (
                1.0 + 1.5 * ULP_1,
                0x3c02,
                "tie above an odd mantissa rounds up",
            ),
            (65519.0, 0x7bff, "below the overflow midpoint stays finite"),
            (65520.0, 0x7c00, "the overflow midpoint ties to even = inf"),
            (-(1.0 + 0.75 * ULP_1), 0xbc01, "the sign carries through"),
        ];
        for (x, want, why) in cases {
            let got = f32_to_f16(x);
            assert_eq!(
                got, want,
                "f32_to_f16({x}) = {got:#06x}, want {want:#06x}: {why}"
            );
        }
    }

    /// R482-003: values below f16's normal range keep their subnormal encoding instead of flushing to zero.
    #[test]
    fn f16_encode_keeps_subnormals() {
        let min_sub = 2f32.powi(-24);
        let cases: [(f32, u16, &str); 6] = [
            (min_sub, 0x0001, "the smallest subnormal"),
            (
                1.0e-5,
                0x00a8,
                "1e-5 is 167.77 subnormal ulps, rounds to 168",
            ),
            (2f32.powi(-15), 0x0200, "a mid-range subnormal is exact"),
            (2f32.powi(-14) - min_sub, 0x03ff, "the largest subnormal"),
            (
                0.5 * min_sub,
                0x0000,
                "half the smallest subnormal ties to even = 0",
            ),
            (
                0.75 * min_sub,
                0x0001,
                "above the tie rounds up to the smallest subnormal",
            ),
        ];
        for (x, want, why) in cases {
            let got = f32_to_f16(x);
            assert_eq!(
                got, want,
                "f32_to_f16({x:e}) = {got:#06x}, want {want:#06x}: {why}"
            );
        }
        assert_eq!(
            f32_to_f16(-1.0e-5),
            0x80a8,
            "a negative subnormal keeps its sign"
        );
        assert_eq!(
            f32_to_f16(-0.0),
            0x8000,
            "negative zero stays negative zero"
        );
    }

    /// R482-003: NaN is never encoded as infinity, whatever its payload or sign; infinities stay infinite.
    #[test]
    fn f16_encode_keeps_nan_and_inf() {
        let nans = [
            f32::NAN,
            f32::from_bits(0x7f80_0001), // signaling, payload only in the low bits
            f32::from_bits(0xffc0_0000), // negative quiet
            f32::from_bits(0x7fff_ffff),
        ];
        for x in nans {
            let got = f32_to_f16(x);
            assert!(
                f16_to_f32(got).is_nan(),
                "f32_to_f16({:#010x}) = {got:#06x} decodes to {}, want NaN",
                x.to_bits(),
                f16_to_f32(got)
            );
            assert_eq!(
                got >> 15,
                (x.to_bits() >> 31) as u16,
                "NaN sign carries through"
            );
        }
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
    }

    /// Every f16 bit pattern survives a decode-encode round trip (NaN patterns stay NaN).
    #[test]
    fn f16_decode_encode_round_trips_every_bit_pattern() {
        for bits in 0..=u16::MAX {
            let decoded = f16_to_f32(bits);
            let back = f32_to_f16(decoded);
            if decoded.is_nan() {
                assert!(
                    f16_to_f32(back).is_nan(),
                    "{bits:#06x}: NaN re-encoded as {back:#06x}"
                );
            } else {
                assert_eq!(
                    back, bits,
                    "{bits:#06x} decodes to {decoded} and re-encodes as {back:#06x}"
                );
            }
        }
    }

    #[test]
    fn f16_decodes_normals_subnormals_and_specials_exactly() {
        let cases: [(u16, u32); 10] = [
            (0x0000, 0x0000_0000),
            (0x8000, 0x8000_0000),
            (0x3c00, 0x3f80_0000), // 1.0
            (0xc000, 0xc000_0000), // -2.0
            (0x3555, 0x3eaa_a000), // 0.333251953125
            (0x7bff, 0x477f_e000), // 65504, the max normal
            (0x0001, 0x3380_0000), // 2^-24, the min subnormal
            (0x03ff, 0x387f_c000), // the max subnormal
            (0x7c00, 0x7f80_0000), // +inf
            (0xfc00, 0xff80_0000), // -inf
        ];
        for (bits, expected) in cases {
            assert_eq!(f16_to_f32(bits).to_bits(), expected, "f16 {bits:#06x}");
        }
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    /// Golden table for [`bf16_to_f32`] (moved from `poot-runtime-common`): the exact widen, expectations as hex literals.
    #[test]
    fn bf16_to_f32_golden_table() {
        const CASES: &[(u16, u32)] = &[
            (0x0000, 0x0000_0000),
            (0x8000, 0x8000_0000),
            (0x3F80, 0x3F80_0000),
            (0xBF80, 0xBF80_0000),
            (0x3F81, 0x3F81_0000),
            (0x7F80, 0x7F80_0000),
            (0xFF80, 0xFF80_0000),
            (0x007F, 0x007F_0000),
            (0x0080, 0x0080_0000),
            (0x7FC0, 0x7FC0_0000),
            (0xFFC0, 0xFFC0_0000),
        ];
        for &(bits, want) in CASES {
            let got = bf16_to_f32(bits);
            assert_eq!(got.to_bits(), want, "bf16_to_f32({bits:#06x})");
        }
        // Both NaN rows are NaNs and both inf rows are infinities.
        assert!(bf16_to_f32(0x7FC0).is_nan());
        assert!(bf16_to_f32(0xFFC0).is_nan());
        assert_eq!(bf16_to_f32(0x7F80), f32::INFINITY);
        assert_eq!(bf16_to_f32(0xFF80), f32::NEG_INFINITY);
    }

    #[test]
    fn e4m3fn_and_e8m0_edges_are_the_ocp_values() {
        for (byte, expected) in [
            (0x00, 0x0000_0000),
            (0x80, 0x8000_0000),
            (0x01, 0x3b00_0000), // 2^-9, the min subnormal
            (0x38, 0x3f80_0000), // 1.0
            (0xb8, 0xbf80_0000), // -1.0
            (0x7e, 0x43e0_0000), // 448, the max finite
        ] {
            assert_eq!(
                e4m3fn_to_f32(byte).to_bits(),
                expected,
                "E4M3FN {byte:#04x}"
            );
        }
        assert_eq!(e4m3fn_to_f32(0x7f).to_bits(), 0x7fc0_0000);
        assert_eq!(e4m3fn_to_f32(0xff).to_bits(), 0x7fc0_0000);
        for (byte, expected) in [
            (0x00, 0x0040_0000), // 2^-127, an f32 subnormal
            (0x01, 0x0080_0000),
            (0x7f, 0x3f80_0000),
            (0x80, 0x4000_0000),
            (0xfe, 0x7f00_0000),
        ] {
            assert_eq!(e8m0_to_f32(byte).to_bits(), expected, "E8M0 {byte:#04x}");
        }
        assert!(e8m0_to_f32(0xff).is_nan());
    }

    #[test]
    fn e2m1_codes_are_the_ocp_values_with_negative_zero() {
        let expected: [u32; 16] = [
            0.0f32.to_bits(),
            0.5f32.to_bits(),
            1.0f32.to_bits(),
            1.5f32.to_bits(),
            2.0f32.to_bits(),
            3.0f32.to_bits(),
            4.0f32.to_bits(),
            6.0f32.to_bits(),
            0x8000_0000, // -0.0
            (-0.5f32).to_bits(),
            (-1.0f32).to_bits(),
            (-1.5f32).to_bits(),
            (-2.0f32).to_bits(),
            (-3.0f32).to_bits(),
            (-4.0f32).to_bits(),
            (-6.0f32).to_bits(),
        ];
        for (code, expected) in expected.into_iter().enumerate() {
            assert_eq!(
                e2m1_to_f32(code as u8).to_bits(),
                expected,
                "E2M1 code {code}"
            );
        }
    }

    /// Golden table for [`f32_to_bf16`]. Every expectation is a hex literal, never computed by the
    /// function under test. Covers exact ties (round to even both directions), values just above and
    /// below the midpoint, max-finite overflow to infinity, +/-inf, qNaN and sNaN staying NaN,
    /// +/-0, and subnormals (including a tie at the subnormal/normal boundary).
    #[test]
    fn f32_to_bf16_golden_table() {
        const CASES: &[(u32, u16)] = &[
            // Exact ties: low16 == 0x8000, round to even (even result both down and up).
            (0x3F80_8000, 0x3F80),
            (0x3F81_8000, 0x3F82),
            (0xBF80_8000, 0xBF80),
            (0xBF81_8000, 0xBF82),
            // Just below / just above the same midpoints.
            (0x3F80_7FFF, 0x3F80),
            (0x3F80_8001, 0x3F81),
            (0x3F81_7FFF, 0x3F81),
            (0x3F81_8001, 0x3F82),
            (0xBF80_7FFF, 0xBF80),
            (0xBF80_8001, 0xBF81),
            // Above the midpoint but far from it (the value the old ROCm body got wrong).
            (0x3F80_A000, 0x3F81),
            (0x3F80_9FFF, 0x3F81),
            // Exactly representable: identity on the high 16 bits.
            (0x3F80_0000, 0x3F80),
            (0xC000_0000, 0xC000),
            (0x3F81_0000, 0x3F81),
            // Max finite overflows to infinity; a tie at the top boundary goes to the even side.
            (0x7F7F_FFFF, 0x7F80),
            (0xFF7F_FFFF, 0xFF80),
            (0x7F7F_8000, 0x7F80),
            (0x7F7E_8000, 0x7F7E),
            // +/-infinity.
            (0x7F80_0000, 0x7F80),
            (0xFF80_0000, 0xFF80),
            // Quiet NaN stays quiet NaN, sign and high payload bits preserved.
            (0x7FC0_0000, 0x7FC0),
            (0x7FD2_ABCD, 0x7FD2),
            (0xFFC0_0000, 0xFFC0),
            // Signaling NaN is quieted (bf16 has no sNaN), sign preserved.
            (0x7F80_0001, 0x7FC0),
            (0xFF80_0001, 0xFFC0),
            (0x7FBF_FFFF, 0x7FFF),
            // +/-0.
            (0x0000_0000, 0x0000),
            (0x8000_0000, 0x8000),
            // Subnormals: flush to zero, tie to even, just off the tie, and the sub/normal boundary.
            (0x0000_0001, 0x0000),
            (0x8000_0001, 0x8000),
            (0x0000_8000, 0x0000),
            (0x0000_8001, 0x0001),
            (0x0001_8000, 0x0002),
            (0x007F_FFFF, 0x0080),
            (0x007F_8000, 0x0080),
        ];
        for &(bits, want) in CASES {
            let got = f32_to_bf16(f32::from_bits(bits));
            assert_eq!(got, want, "f32_to_bf16({bits:#010x}) = {got:#06x}");
        }
    }
}
