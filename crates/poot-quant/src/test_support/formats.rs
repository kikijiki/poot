//! The GGUF block formats as one list, for tests that cover every format (POOT-1514).
//!
//! The list is the iteration side of two mappings the compiler already checks:
//! [`WeightFormat::descriptor`] and the const-plan dispatch behind [`crate::plan::decode_blocks`]
//! both match every variant, so a new block format does not compile without a descriptor and a
//! plan. The loader anchors the list: `poot-load` pins that the ggml types it reads are exactly
//! these formats, so a format the loader accepts is never missing here. A test that covers "every
//! block format" iterates this list (or a filter of it over the descriptor) rather than writing
//! its own.

use std::num::NonZeroUsize;

use crate::format::{GroupMap, ScaleEncoding, Storage, WeightFormat};

/// Every format a GGUF file stores and poot reads, in ggml type order (`ggml.h` `ggml_type`).
pub const GGUF_FORMATS: [WeightFormat; 25] = [
    WeightFormat::F32,
    WeightFormat::F16,
    WeightFormat::Q4_0,
    WeightFormat::Q4_1,
    WeightFormat::Q5_0,
    WeightFormat::Q5_1,
    WeightFormat::Q8_0,
    WeightFormat::Q2_K,
    WeightFormat::Q3_K,
    WeightFormat::Q4_K,
    WeightFormat::Q5_K,
    WeightFormat::Q6_K,
    WeightFormat::Iq2_Xxs,
    WeightFormat::Iq2_Xs,
    WeightFormat::Iq3_Xxs,
    WeightFormat::Iq1_S,
    WeightFormat::Iq4_Nl,
    WeightFormat::Iq3_S,
    WeightFormat::Iq2_S,
    WeightFormat::Iq4_Xs,
    WeightFormat::Iq1_M,
    WeightFormat::Bf16,
    WeightFormat::Tq1_0,
    WeightFormat::Tq2_0,
    WeightFormat::Mxfp4,
];

/// The safetensors formats poot reads, one entry per variant parameter that changes the descriptor: every
/// scale encoding of the two E4M3 layouts, the two GPTQ group maps, AWQ and E2M1.
pub const SAFETENSORS_FORMATS: [WeightFormat; 10] = [
    WeightFormat::E4m3PerChannel {
        scale: ScaleEncoding::F32,
    },
    WeightFormat::E4m3PerChannel {
        scale: ScaleEncoding::Bf16,
    },
    WeightFormat::E4m3PerChannel {
        scale: ScaleEncoding::E8m0,
    },
    WeightFormat::E4m3Block128 {
        scale: ScaleEncoding::F32,
    },
    WeightFormat::E4m3Block128 {
        scale: ScaleEncoding::Bf16,
    },
    WeightFormat::E4m3Block128 {
        scale: ScaleEncoding::E8m0,
    },
    WeightFormat::E2m1Row32,
    WeightFormat::Gptq {
        groups: GroupMap::Contiguous {
            size: NonZeroUsize::new(128).unwrap(),
        },
    },
    WeightFormat::Gptq {
        groups: GroupMap::Indexed {
            groups: NonZeroUsize::new(32).unwrap(),
        },
    },
    WeightFormat::Awq {
        group_size: NonZeroUsize::new(128).unwrap(),
    },
];

/// Every format that is a packed weight rather than a dense dtype: the GGUF block formats of more than one value
/// per block and every safetensors format.
pub fn packed_formats() -> Vec<WeightFormat> {
    GGUF_FORMATS
        .into_iter()
        .chain(SAFETENSORS_FORMATS)
        .filter(|format| !matches!(format.descriptor().storage, Storage::Blocks(layout) if layout.values == 1))
        .collect()
}

/// The packed GGUF block formats: every GGUF format of more than one value per block.
pub fn packed_block_formats() -> Vec<WeightFormat> {
    GGUF_FORMATS
        .into_iter()
        .filter(|format| matches!(format.descriptor().storage, Storage::Blocks(layout) if layout.values > 1))
        .collect()
}

/// The planar formats of exactly two source tensors, codes and scale (the E4M3 layouts and E2M1).
pub fn two_source_formats() -> Vec<WeightFormat> {
    SAFETENSORS_FORMATS
        .into_iter()
        .filter(|format| matches!(format.descriptor().storage, Storage::Planar(layout) if layout.operands.len() == 2))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// The position of `format`'s variant. The match has no wildcard: a new variant does not compile until it
    /// is given a position here, and the list below must then name it.
    fn variant_position(format: WeightFormat) -> usize {
        match format {
            WeightFormat::F32 => 0,
            WeightFormat::F16 => 1,
            WeightFormat::Bf16 => 2,
            WeightFormat::Q4_0 => 3,
            WeightFormat::Q4_1 => 4,
            WeightFormat::Q5_0 => 5,
            WeightFormat::Q5_1 => 6,
            WeightFormat::Q8_0 => 7,
            WeightFormat::Q2_K => 8,
            WeightFormat::Q3_K => 9,
            WeightFormat::Q4_K => 10,
            WeightFormat::Q5_K => 11,
            WeightFormat::Q6_K => 12,
            WeightFormat::Iq4_Nl => 13,
            WeightFormat::Iq4_Xs => 14,
            WeightFormat::Iq1_S => 27,
            WeightFormat::Iq1_M => 28,
            WeightFormat::Iq2_Xxs => 15,
            WeightFormat::Iq2_Xs => 16,
            WeightFormat::Iq3_Xxs => 17,
            WeightFormat::Iq3_S => 18,
            WeightFormat::Iq2_S => 19,
            WeightFormat::Tq1_0 => 29,
            WeightFormat::Tq2_0 => 20,
            WeightFormat::Mxfp4 => 21,
            WeightFormat::E4m3PerChannel { .. } => 22,
            WeightFormat::E4m3Block128 { .. } => 23,
            WeightFormat::E2m1Row32 => 24,
            WeightFormat::Gptq { .. } => 25,
            WeightFormat::Awq { .. } => 26,
        }
    }

    /// The two lists together name every [`WeightFormat`] variant at least once and no format twice. Mutation:
    /// delete one entry of either list; this goes red naming the variant position.
    #[test]
    fn the_format_lists_name_every_variant_once() {
        let all: Vec<WeightFormat> = GGUF_FORMATS
            .into_iter()
            .chain(SAFETENSORS_FORMATS)
            .collect();
        let positions: BTreeSet<usize> =
            all.iter().map(|&format| variant_position(format)).collect();
        assert_eq!(
            positions,
            (0..30).collect::<BTreeSet<usize>>(),
            "a variant is missing from GGUF_FORMATS and SAFETENSORS_FORMATS"
        );
        for (index, format) in all.iter().enumerate() {
            assert!(!all[..index].contains(format), "{format:?} is listed twice");
        }
    }
}
