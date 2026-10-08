//! Packed payloads of the V4 checkpoint formats (E2M1 experts, E4M3 projections) with a decoder
//! written from the two format definitions, not from the production decode: comparing a model or a
//! projection against it compares two independent implementations.

use std::sync::Arc;

use poot_quant::format::{ScaleEncoding, WeightFormat};
use poot_quant::{OperandRole, PackedPayload, PackedWeight, SourceRole};

/// The routed experts: adjacent-K E2M1 codes with one E8M0 scale per row per 32 values of K.
pub(super) const FP4: WeightFormat = WeightFormat::E2m1Row32;
/// Every other packed projection: E4M3 codes with one E8M0 scale per 128 x 128 block.
pub(super) const E4M3: WeightFormat = WeightFormat::E4m3Block128 {
    scale: ScaleEncoding::E8m0,
};

/// The eight E2M1 magnitudes; bit 3 of a code is the sign.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The finite E4M3 codes a fixture draws from: zero and `+-0.5, 0.75, 1.0, 1.5`.
const E4M3_CODES: [u8; 9] = [0x00, 0x30, 0x34, 0x38, 0x3c, 0xb0, 0xb4, 0xb8, 0xbc];

fn mix(seed: u64, index: usize) -> u64 {
    let mut z = seed.wrapping_add((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn e2m1(code: u8) -> f32 {
    let magnitude = E2M1[usize::from(code & 0x07)];
    if code & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let (exponent, mantissa) = (i32::from((byte >> 3) & 0x0f), f32::from(byte & 0x07));
    if exponent == 0 {
        sign * mantissa * 2.0f32.powi(-9)
    } else {
        sign * (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
    }
}

/// A packed `[out, k]` weight of `format` with codes drawn from `seed` and the E8M0 scale byte of
/// each scale block `(row block, column block)` given by `scale`.
pub(super) fn payload(
    format: WeightFormat,
    shape: [usize; 2],
    seed: u64,
    scale: impl Fn(usize, usize) -> u8,
) -> PackedPayload {
    let weight = PackedWeight::try_new(format, shape).expect("a valid packed shape");
    let codes = SourceRole::Planar(OperandRole::Codes);
    let scales = SourceRole::Planar(OperandRole::Scale);
    let code_bytes: Vec<u8> = (0..weight.source_bytes(codes))
        .map(|i| match format {
            WeightFormat::E2m1Row32 => mix(seed, i) as u8,
            _ => E4M3_CODES[(mix(seed, i) % E4M3_CODES.len() as u64) as usize],
        })
        .collect();
    let [_, scale_columns] = weight.source_shape(scales);
    let scale_bytes: Vec<u8> = (0..weight.source_bytes(scales))
        .map(|i| scale(i / scale_columns, i % scale_columns))
        .collect();
    PackedPayload::try_new(
        weight,
        [
            (codes, Arc::<[u8]>::from(code_bytes)),
            (scales, Arc::<[u8]>::from(scale_bytes)),
        ],
    )
    .expect("a valid packed payload")
}

/// The logical `[out, k]` values of `payload`, row-major, decoded from the format definitions:
/// E2M1 stores two codes per byte (even K in the low nibble) with one E8M0 byte per row per 32 K;
/// E4M3 stores one code per byte with one E8M0 byte per 128 x 128 block; a value is the code's
/// number times `2^(scale - 127)`.
pub(super) fn literal(payload: &PackedPayload) -> Vec<f32> {
    let weight = payload.weight();
    let [out, k] = weight.shape();
    let codes = SourceRole::Planar(OperandRole::Codes);
    let scales = SourceRole::Planar(OperandRole::Scale);
    let [_, code_columns] = weight.source_shape(codes);
    let [_, scale_columns] = weight.source_shape(scales);
    let (code_bytes, scale_bytes) = (payload.bytes(codes), payload.bytes(scales));
    let fp4 = weight.format() == FP4;
    let mut values = Vec::with_capacity(out * k);
    for row in 0..out {
        for column in 0..k {
            let (value, scale) = if fp4 {
                let byte = code_bytes[row * code_columns + column / 2];
                let code = if column % 2 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                (e2m1(code), scale_bytes[row * scale_columns + column / 32])
            } else {
                (
                    e4m3(code_bytes[row * code_columns + column]),
                    scale_bytes[(row / 128) * scale_columns + column / 128],
                )
            };
            values.push(value * 2.0f32.powi(i32::from(scale) - 127));
        }
    }
    values
}
