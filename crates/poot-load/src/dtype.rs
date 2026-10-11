use super::*;
use poot_tensor::DType;

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
}
