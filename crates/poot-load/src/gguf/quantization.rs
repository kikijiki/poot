//! GGML quantization encoders, packed representations, and f32 decoders.

#[cfg(test)]
use crate::LoadError;
#[cfg(test)]
use poot_quant::scalar::{e2m1_to_f32, e8m0_to_f32};

/// Encode f32 values into ggml Q8_0 blocks: per 32 elements, `d = max|x|/127` as f16, then 32
/// `round(x/d)` int8. `vals.len()` must be a multiple of 32. Inverse of [`dequant_q8_0`]; a test
/// fixture helper (pair with `write_gguf`).
#[cfg(test)]
pub(crate) fn encode_q8_0(vals: &[f32]) -> Vec<u8> {
    assert!(
        vals.len().is_multiple_of(32),
        "Q8_0 needs a multiple of 32 values"
    );
    let mut out = Vec::with_capacity((vals.len() / 32) * 34);
    for blk in vals.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let d = amax / 127.0;
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for &x in blk {
            let q = if d != 0.0 {
                (x / d).round().clamp(-127.0, 127.0) as i8
            } else {
                0
            };
            out.push(q as u8);
        }
    }
    out
}

/// Encode f32 values into ggml Q4_0 blocks: per 32 elements, an f16 scale `d = max|x|/8` then 16
/// bytes packing two nibbles `clamp(round(x/d) + 8, 0, 15)` in ggml's low-half/high-half
/// interleave (byte `j` low nibble = element `j`, high nibble = element `j+16`). Inverse of
/// [`dequant_q4_0`]; a test fixture helper. `d` is always positive (simple encoder); the
/// pack/dequant round trip is exact regardless since both read the same bytes.
#[cfg(test)]
pub(crate) fn encode_q4_0(vals: &[f32]) -> Vec<u8> {
    assert!(
        vals.len().is_multiple_of(32),
        "Q4_0 needs a multiple of 32 values"
    );
    let nib = |x: f32, d: f32| -> u8 {
        if d != 0.0 {
            ((x / d).round() + 8.0).clamp(0.0, 15.0) as u8
        } else {
            8
        }
    };
    let mut out = Vec::with_capacity((vals.len() / 32) * 18);
    for blk in vals.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let d = amax / 8.0;
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for half in 0..16 {
            let lo = nib(blk[half], d);
            let hi = nib(blk[half + 16], d);
            out.push(lo | (hi << 4));
        }
    }
    out
}

/// Encode f32 values into ggml Q5_0 blocks (card 179): per 32 elements, an f16 scale
/// `d = max|x|/16`, a `u32 qh` (bit `j` = 5th bit of element `j`), then 16 bytes packing two 4-bit
/// halves with the same interleave as [`encode_q4_0`]. Inverse of [`dequant_q5_0`]; a test
/// fixture helper. `d` is always positive; the round trip is exact regardless.
#[cfg(test)]
pub(crate) fn encode_q5_0(vals: &[f32]) -> Vec<u8> {
    assert!(
        vals.len().is_multiple_of(32),
        "Q5_0 needs a multiple of 32 values"
    );
    let q5 = |x: f32, d: f32| -> u8 {
        if d != 0.0 {
            ((x / d).round() + 16.0).clamp(0.0, 31.0) as u8
        } else {
            16
        }
    };
    let mut out = Vec::with_capacity((vals.len() / 32) * 22);
    for blk in vals.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let d = amax / 16.0;
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        let mut qh: u32 = 0;
        let mut nibs = [0u8; 32];
        for (j, &v) in blk.iter().enumerate() {
            let q = q5(v, d);
            nibs[j] = q & 0x0F;
            qh |= (((q >> 4) & 1) as u32) << j;
        }
        out.extend_from_slice(&qh.to_le_bytes());
        for half in 0..16 {
            out.push(nibs[half] | (nibs[half + 16] << 4));
        }
    }
    out
}

/// f32 -> IEEE-754 binary16 bits, round to nearest with ties to even (ADR-0101 decision 4). Subnormal
/// results are kept, overflow rounds to infinity, and a NaN stays a NaN (sign and top payload bits carry
/// through; the quiet bit is forced so a payload confined to the dropped low bits cannot collapse to
/// infinity). The one host f16 encoder: `poot-eval`'s Cast oracle and the resident uploads of `poot-gpu` and
/// `poot-ptx-gpu` narrow with it, as do the Q5_0/Q8_0 scale encoders and test fixtures.
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

/// Test-only now: production GGUF reads bound-check through `poot_quant`'s block decoder, not this
/// helper. Kept for the test-oracle `dequant` (in `gguf.rs`) and its own truncation-guard test below.
#[cfg(test)]
pub(crate) fn bytes_ok(raw: &[u8], need: usize) -> Result<(), LoadError> {
    if raw.len() < need {
        return Err(LoadError::GgufTensorTruncated {
            need,
            have: raw.len(),
        });
    }
    Ok(())
}

/// Encode f32 values into ggml MXFP4 blocks (a test fixture helper). Mirrors ggml's
/// `quantize_row_mxfp4_ref` (`ggml/src/ggml-quants.c`): per 32 elements the shared exponent is
/// `e = amax > 0 ? floor(log2(amax)) - 2 + 127 : 0`, then each element takes the code whose decoded
/// value `e2m1(code) * 2^(e - 127)` is nearest (ggml's `best_index_mxfp4`), in the `Q4_0` nibble
/// order (byte `j` = element `j` low, element `j + 16` high).
#[cfg(test)]
pub(crate) fn encode_mxfp4(vals: &[f32]) -> Vec<u8> {
    assert!(
        vals.len().is_multiple_of(32),
        "MXFP4 needs a multiple of 32 values"
    );
    let mut out = Vec::with_capacity((vals.len() / 32) * 17);
    for blk in vals.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let e: u8 = if amax > 0.0 {
            (amax.log2().floor() - 2.0 + 127.0).clamp(0.0, 254.0) as u8
        } else {
            0
        };
        let scale = e8m0_to_f32(e);
        let code_of = |x: f32| -> u8 {
            (0..16u8)
                .min_by(|&a, &b| {
                    let error = |code| (x - e2m1_to_f32(code) * scale).abs();
                    error(a).total_cmp(&error(b))
                })
                .expect("sixteen codes")
        };
        out.push(e);
        for j in 0..16 {
            out.push(code_of(blk[j]) | (code_of(blk[j + 16]) << 4));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{bytes_ok, f32_to_f16};
    use crate::LoadError;
    use poot_quant::scalar::f16_to_f32;

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

    /// Mutant M5 (mutants-m4.md): `bytes_ok -> Ok(())` lets a tensor one byte short of its blocks
    /// reach a `raw[..need]` slice and panic. `bytes_ok` must refuse it with the typed
    /// `GgufTensorTruncated` error (card 545a: no longer a `SafeTensors(String)`), and accept a
    /// buffer of exactly (or more than) the needed length.
    #[test]
    fn truncated_tensor_data_is_a_typed_error() {
        let need = 68;
        assert!(bytes_ok(&vec![0u8; need], need).is_ok());
        assert!(bytes_ok(&vec![0u8; need + 1], need).is_ok());
        let result = bytes_ok(&vec![0u8; need - 1], need);
        assert!(
            matches!(
                result,
                Err(LoadError::GgufTensorTruncated { need: n, have }) if n == need && have == need - 1
            ),
            "{result:?}"
        );
    }
}
