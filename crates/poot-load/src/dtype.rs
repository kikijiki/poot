//! Safetensors header helpers shared by the single-file and sharded readers: dtype-string parsing, header
//! length bounds and the missing-field error.

use super::LoadError;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// SC-001: every dtype `DType` names has one element width, read through `parse_dtype` for every
    /// recognized string.
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
        }
    }

    /// SC-001: an unknown dtype string is a typed error, not a silent `None` or a panic.
    #[test]
    fn parse_of_an_unrecognized_dtype_string_is_a_typed_error() {
        assert!(matches!(
            parse_dtype("COMPLEX64"),
            Err(LoadError::UnknownCheckpointDtype { name }) if name == "COMPLEX64"
        ));
    }
}
