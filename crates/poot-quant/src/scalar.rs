//! The bit-exact decoders for the block-scale and sub-byte float formats a weight code or scale is stored
//! in, and the f16/bf16 encoders. The f16, bf16 and e4m3fn decoders are [`poot_tensor::float`]'s (the leaf
//! `poot-tensor` and this crate both sit on).
//!
//! Each function is a total conversion of the stored bits to the f32 with the same value: NaN
//! encodings decode to NaN (callers that must refuse NaN check [`f32::is_nan`]), and signed zeros
//! keep their sign.

use crate::format::FloatFormat;
use poot_tensor::float::{bf16_to_f32, e4m3fn_to_f32, f16_to_f32};

/// f32 -> IEEE-754 binary16 bits, round to nearest with ties to even (ADR-0101 decision 4). Subnormal
/// results are kept, overflow rounds to infinity, and a NaN stays a NaN (sign and top payload bits carry
/// through; the quiet bit is forced so a payload confined to the dropped low bits cannot collapse to
/// infinity). The one host f16 encoder: `poot-eval`'s Cast oracle and the resident uploads of `poot-wgpu-device` and
/// `poot-ptx-device` narrow with it, as does [`crate::encode`] for every f16 block scale.
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

#[cfg(any(test, feature = "test-support"))]
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

/// `(bits + 1/2) * 2^exponent`, the value of a [`crate::format::FieldEncoding::HalfStep`] field.
/// `bits` is a few bits wide and `exponent` a normal f32 exponent, so both the half step and the
/// scaling are exact.
pub const fn half_step_to_f32(bits: u32, exponent: i32) -> f32 {
    assert!(
        -126 <= exponent && exponent <= 127,
        "a normal f32 power of two"
    );
    (bits as f32 + 0.5) * f32::from_bits(((127 + exponent) as u32) << 23)
}

/// `(-1)^bits * 2^exponent`, the value of a one-bit [`crate::format::FieldEncoding::SignedPowerOfTwo`] field.
pub const fn signed_power_of_two_to_f32(bits: u32, exponent: i32) -> f32 {
    assert!(bits <= 1, "a one-bit field");
    assert!(
        -126 <= exponent && exponent <= 127,
        "a normal f32 power of two"
    );
    f32::from_bits((bits << 31) | (((127 + exponent) as u32) << 23))
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

    /// A one-bit shift field is `(-1)^bit * 2^exponent`, exact: ggml's `IQ1S_DELTA` is `+/-0.125`, and the sign
    /// bit of the result is the stored bit. Mutation: return `+2^exponent` for bit 1; the second assertion fails
    /// with `left: 0.125, right: -0.125`.
    #[test]
    fn a_signed_power_of_two_is_the_sign_of_its_bit() {
        assert_eq!(signed_power_of_two_to_f32(0, -3), 0.125);
        assert_eq!(signed_power_of_two_to_f32(1, -3), -0.125);
        assert_eq!(
            signed_power_of_two_to_f32(1, 2).to_bits(),
            (-4.0f32).to_bits()
        );
        assert_eq!(signed_power_of_two_to_f32(0, 0), 1.0);
    }

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
    fn e8m0_edges_are_the_ocp_values() {
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
