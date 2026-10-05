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
}
