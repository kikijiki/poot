//! The one bit-exact decoder for each IEEE/OCP narrow float format `poot-tensor` stores or widens
//! (`f16`, `bf16`, `e4m3fn`). `poot-quant`'s own `scalar` module holds the encoders and the block-scale
//! formats on top of these.
//!
//! Each function is a total conversion of the stored bits to the f32 with the same value: NaN
//! encodings decode to NaN (callers that must refuse NaN check [`f32::is_nan`]), and signed zeros
//! keep their sign.

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
    fn e4m3fn_edges_are_the_ocp_values() {
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
    }
}
