//! POOT-1289: each `LoadError` refusal the variant-coverage list (scripts/variant-coverage.tsv) recorded
//! as a gap, made to fire through its production entry. Model-free, in-memory or temp-dir fixtures.
//!

use poot_load::LoadError;
use poot_load::config::QuantConfig;
use poot_load::gguf::{GgufIndex, TensorNameMap, read_gguf, write_gguf};
use poot_load::lora::LoraAdapter;
use poot_load::plan::default_workers;
use poot_load::safetensors::map_weight_files;
use poot_quant::StoredBytes;
use poot_quant::weights::WeightKey;

const GGML_F32: u32 = 0;

/// A safetensors file: the 8-byte little-endian header length, `header`, then `data`.
fn safetensors(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header.as_bytes());
    out.extend(data);
    out
}

/// The error of a load that must be refused; the `Ok` payloads here are not `Debug`.
fn refusal<T>(result: Result<T, LoadError>, what: &str) -> LoadError {
    match result {
        Ok(_) => panic!("{what}: expected a refusal, the load succeeded"),
        Err(error) => error,
    }
}

/// A `model.safetensors` that is a directory, not a regular file: the mapping refuses it, and the
/// refusal reaches the caller as `LoadError::Map`, not as an io error or a panic.
///
/// Mutation: in `safetensors::map_file` convert the mapping's error to `LoadError::SafeTensors`; red:
/// `expected LoadError::Map, got: safetensors: cannot map a file that is not a regular file (mode 40755)`.
#[test]
fn a_shard_that_is_not_a_regular_file_is_a_map_refusal() {
    let dir = poot_test_util::unique_temp_path("rc-map");
    std::fs::create_dir_all(dir.join("model.safetensors")).unwrap();
    let error = refusal(
        map_weight_files(&*dir, |name| {
            std::fs::read(dir.join(name)).map_err(LoadError::from)
        }),
        "a directory named model.safetensors",
    );
    assert!(
        matches!(error, LoadError::Map(_)),
        "expected LoadError::Map, got: {error}"
    );
}

/// A LoRA adapter whose `lora_A` is stored as I64: the adapter loader refuses it by dtype instead of
/// reading integers as floats.
///
/// Mutation: in `lora::factor` report the dtype as `LoadError::SafeTensors`; red:
/// `expected LoraFactorDtype naming I64, got: safetensors: I64`.
#[test]
fn a_lora_tensor_of_a_non_float_dtype_is_refused() {
    let dir = poot_test_util::unique_temp_path("rc-lora");
    std::fs::create_dir_all(&*dir).unwrap();
    std::fs::write(
        dir.join("adapter_config.json"),
        r#"{"r":1,"lora_alpha":1.0,"target_modules":["q_proj"]}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("adapter_model.safetensors"),
        safetensors(
            r#"{"base_model.model.q_proj.lora_A.weight":{"dtype":"I64","shape":[1,1],"data_offsets":[0,8]},"base_model.model.q_proj.lora_B.weight":{"dtype":"F32","shape":[1,1],"data_offsets":[8,12]}}"#,
            &[0u8; 12],
        ),
    )
    .unwrap();
    let error = refusal(LoraAdapter::load(&*dir), "an I64 lora_A");
    assert!(
        matches!(
            &error,
            LoadError::LoraFactorDtype {
                dtype: poot_tensor::DType::I64,
                ..
            }
        ),
        "expected LoraFactorDtype naming I64, got: {error}"
    );
}

/// A `quantization_config` the scheme resolver has no decoder for: a GPTQ `checkpoint_format` other
/// than `gptq` stores zeros one unit off, so it is refused by the config key and the value it carried.
///
/// Mutation: make the `checkpoint_format` filter in `QuantConfig::scheme` never match; red:
/// `gptq_v2: expected a refusal, the load succeeded`.
#[test]
fn a_gptq_checkpoint_format_other_than_gptq_is_an_unsupported_quant_config() {
    let config: QuantConfig = serde_json::from_str(
        r#"{"quant_method":"gptq","bits":4,"sym":true,"desc_act":false,"checkpoint_format":"gptq_v2"}"#,
    )
    .unwrap();
    let error = refusal(config.scheme(), "gptq_v2");
    assert!(
        matches!(
            &error,
            LoadError::UnsupportedQuantConfig { field: "checkpoint_format", value } if value == "gptq_v2"
        ),
        "expected UnsupportedQuantConfig on checkpoint_format, got: {error}"
    );
}

/// A name map that sends every tensor to one weight-store key: two GGUF tensors resolve to the same
/// key, and the store refuses to hold both rather than keeping the last.
///
/// Mutation: let `WeightStoreBuilder::insert` accept an occupied key; red:
/// `two tensors under one key: expected a refusal, the load succeeded`.
#[test]
fn two_gguf_tensors_mapped_to_one_weight_key_are_a_duplicate_weight_key() {
    struct OneKey;
    impl TensorNameMap for OneKey {
        fn weight_key(&self, _gguf_name: &str) -> Option<WeightKey> {
            Some(WeightKey::from("shared"))
        }
    }
    let bytes = write_gguf(
        &[],
        &[
            ("a.weight", vec![1], GGML_F32, vec![0u8; 4]),
            ("b.weight", vec![1], GGML_F32, vec![0u8; 4]),
        ],
    );
    let index = GgufIndex::parse(&StoredBytes::from(bytes)).unwrap();
    let error = refusal(
        read_gguf(&index, &OneKey, default_workers()),
        "two tensors under one key",
    );
    assert!(
        matches!(error, LoadError::GgufDuplicateWeightKey(_)),
        "expected GgufDuplicateWeightKey, got: {error}"
    );
}
