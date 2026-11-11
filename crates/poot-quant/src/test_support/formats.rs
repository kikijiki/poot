//! The GGUF block formats as one list, for tests that cover every format (POOT-1514).
//!
//! The list is the iteration side of two mappings the compiler already checks:
//! [`WeightFormat::descriptor`] and the const-plan dispatch behind [`crate::plan::decode_blocks`]
//! both match every variant, so a new block format does not compile without a descriptor and a
//! plan. The loader anchors the list: `poot-load` pins that the ggml types it reads are exactly
//! these formats, so a format the loader accepts is never missing here. A test that covers "every
//! block format" iterates this list (or a filter of it over the descriptor) rather than writing
//! its own.

use crate::format::WeightFormat;

/// Every format a GGUF file stores and poot reads, in ggml type order (`ggml.h` `ggml_type`).
pub const GGUF_FORMATS: [WeightFormat; 19] = [
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
    WeightFormat::Iq2_Xs,
    WeightFormat::Iq3_Xxs,
    WeightFormat::Iq4_Nl,
    WeightFormat::Iq3_S,
    WeightFormat::Iq4_Xs,
    WeightFormat::Bf16,
    WeightFormat::Mxfp4,
];
