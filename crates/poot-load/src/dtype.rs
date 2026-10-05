use super::*;
use poot_tensor::DType;

use poot_quant::scalar;

pub(crate) fn miss(name: &str, field: &str) -> LoadError {
    LoadError::SafeTensors(format!("{name}: missing {field}"))
}

pub(crate) fn checked_header_end(header_len: u64, file_len: usize) -> Result<usize, LoadError> {
    let header_len = usize::try_from(header_len)
        .map_err(|_| LoadError::SafeTensors("header length out of range".into()))?;
    8usize
        .checked_add(header_len)
        .filter(|&end| end <= file_len)
        .ok_or_else(|| LoadError::SafeTensors("header length out of range".into()))
}

/// Parse a safetensors header `dtype` string into `poot_quant`'s [`DType`] (card 540a moved
/// the enum there so a stored tensor's dtype is one type both crates use). A string that names no
/// dtype this crate recognizes is a typed error, never a silent `None` (SC-001).
pub(crate) fn parse_dtype(dtype: &str) -> Result<DType, LoadError> {
    DType::parse(dtype).map_err(|_| LoadError::UnknownCheckpointDtype {
        name: dtype.to_string(),
    })
}

/// The stored byte width of one element for a safetensors header `dtype` string, or `None` when
/// the string names no dtype this crate recognizes ([`parse_dtype`]). The one width table
/// `SafeTensors::load_ranges_opts`'s byte-count check and `packed_safetensors::validation::parse_header_entries`
/// both read (R484-006): mutating [`DType::byte_size`] moves both call sites'
/// expectations together.
pub(crate) fn element_width_bytes(dtype: &str) -> Option<usize> {
    DType::parse(dtype).ok().map(DType::byte_size)
}

pub(crate) fn decode(dtype: &str, raw: &[u8]) -> Result<Vec<f32>, LoadError> {
    match parse_dtype(dtype)? {
        DType::F32 => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
            .collect()),
        DType::BF16 => Ok(raw
            .chunks_exact(2)
            .map(|c| scalar::bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect()),
        DType::F16 => Ok(raw
            .chunks_exact(2)
            .map(|c| scalar::f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect()),
        // FP8 E4M3 codes, unscaled: a quantized linear's codes and scale are packed together by
        // `safetensors::pack_quantized_linears` and decoded by `poot-quant`, never here.
        DType::E4M3FN => Ok(raw.iter().map(|&b| scalar::e4m3fn_to_f32(b)).collect()),
        // The block FP8 E8M0 scale dtype (one byte per block), as a plain tensor.
        DType::E8M0 => Ok(raw.iter().map(|&b| scalar::e8m0_to_f32(b)).collect()),
        DType::Bool
        | DType::I8
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::I32
        | DType::U32
        | DType::I64
        | DType::U64
        | DType::F64 => Err(LoadError::UndecodableCheckpointDtype { name: dtype.into() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SC-001: every dtype `DType` names has one element width. `element_width_bytes` (the
    /// `&str` entry point `SafeTensors`'s reader and `packed_safetensors::validation::parse_header_entries`
    /// both call) must agree with it for every recognized string.
    #[test]
    fn every_recognized_dtype_string_has_one_element_width() {
        let cases: &[(&str, usize)] = &[
            ("BOOL", 1),
            ("I8", 1),
            ("U8", 1),
            ("F8_E4M3", 1),
            ("F8_E8M0", 1),
            ("I16", 2),
            ("U16", 2),
            ("F16", 2),
            ("BF16", 2),
            ("I32", 4),
            ("U32", 4),
            ("F32", 4),
            ("I64", 8),
            ("U64", 8),
            ("F64", 8),
        ];
        for &(dtype, width) in cases {
            assert_eq!(
                parse_dtype(dtype).unwrap().byte_size(),
                width,
                "DType::byte_size({dtype})"
            );
            assert_eq!(
                element_width_bytes(dtype),
                Some(width),
                "element_width_bytes({dtype})"
            );
        }
    }

    /// SC-001: an unknown dtype string is a typed error, not a silent `None` or a panic.
    #[test]
    fn parse_of_an_unrecognized_dtype_string_is_a_typed_error() {
        assert!(matches!(
            parse_dtype("COMPLEX64"),
            Err(LoadError::UnknownCheckpointDtype { name }) if name == "COMPLEX64"
        ));
        assert_eq!(element_width_bytes("COMPLEX64"), None);
    }

    /// `decode` supports exactly the five float dtypes poot-load decodes to an f32 oracle; every
    /// other recognized dtype (kept raw or skipped by `SafeTensors::load_ranges_opts`) is a typed
    /// error rather than silently decoding garbage.
    #[test]
    fn decode_rejects_a_recognized_but_non_decodable_dtype() {
        assert!(matches!(
            decode("I32", &[0u8; 4]),
            Err(LoadError::UndecodableCheckpointDtype { name }) if name == "I32"
        ));
    }
}
