//! A safetensors fixture writer: the HF file layout `poot_load::safetensors::load_weight_store`
//! reads, from a [`WeightStore`] of dense entries.

use std::fmt::Write as _;
use std::path::Path;

use poot_quant::weights::{WeightEntry, WeightStore};
use poot_tensor::DType;

/// The safetensors spelling of a stored dtype.
fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "F32",
        DType::BF16 => "BF16",
        DType::F16 => "F16",
        DType::I32 => "I32",
        DType::I64 => "I64",
        other => panic!("the fixture writer has no safetensors dtype for {other:?}"),
    }
}

/// Write `store`'s tensors, every one dense and stored as it is, to the safetensors file `path`.
/// Panics on a packed entry or a tensor name that is not plain ASCII: both are fixture bugs.
pub fn write(path: &Path, store: &WeightStore) {
    let mut header = String::from("{");
    let mut data = Vec::new();
    for (key, entry) in store.iter() {
        let WeightEntry::Dense(dense) = entry else {
            panic!("the fixture store is dense: {key} is packed")
        };
        let name = key.as_str();
        assert!(
            name.chars()
                .all(|c| c.is_ascii_graphic() && c != '"' && c != '\\'),
            "tensor name {name:?} needs JSON escaping"
        );
        let start = data.len();
        data.extend_from_slice(dense.bytes().as_slice());
        if header.len() > 1 {
            header.push(',');
        }
        write!(
            header,
            "\"{name}\":{{\"dtype\":\"{}\",\"shape\":{:?},\"data_offsets\":[{start},{}]}}",
            dtype_name(dense.dtype()),
            dense.shape(),
            data.len()
        )
        .expect("writing to a String");
    }
    header.push('}');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(path, file).expect("write the safetensors fixture");
}
