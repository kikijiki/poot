use super::super::{SAFETENSORS_SUPPORTED_ARCHS, validate_safetensors_arch};
use crate::{Runner, RunnerError};

// Card-190-class regression: the safetensors `Runner::load` path must reject any model_type it cannot
// trace instead of falling through to another family's tracer. Pure-host: no config, SafeTensors, or
// file.
#[test]
fn validate_safetensors_arch_rejects_unsupported() {
    for arch in SAFETENSORS_SUPPORTED_ARCHS {
        assert!(
            validate_safetensors_arch(arch).is_ok(),
            "expected supported arch {arch:?} to be allowed"
        );
    }
    // "mixtral" is a supported arch (see SAFETENSORS_SUPPORTED_ARCHS above and the loop above), so it is
    // not repeated here: a supported arch in both loops would let the first loop's pass swallow the second
    // loop's failure.
    for arch in [
        "qwen2",
        "llama",
        "gemma4",
        "qwen3_next",
        "qwen2_vl",
        "qwen2_5_vl",
        "gpt-oss",
        "totally_unknown",
    ] {
        assert!(
            validate_safetensors_arch(arch).is_err(),
            "expected unsupported arch {arch:?} to be rejected"
        );
    }
    let qwen25_vl_err = validate_safetensors_arch("qwen2_5_vl")
        .unwrap_err()
        .to_string();
    assert!(
        qwen25_vl_err.contains("load_qwen2_5_vl_text_tower_config_from_files"),
        "qwen2_5_vl error should name the closed config handoff: {qwen25_vl_err}"
    );
}

/// Card 540b SC-003: `Runner::weight_bytes` must count each weight's stored payload at its stored
/// dtype width (R475-011): a bf16 tensor's resident upload is its 2 bytes per element, an f32 tensor's
/// 4. Reading an f32-width count for every dense weight (the pre-R475-011 bug) would overcount the
/// bf16 weight 2x.
#[test]
fn weight_bytes_counts_each_weights_stored_dtype_width() {
    let mut runner = crate::core::decode_arch::fixtures::runner_for(
        crate::core::decode_arch::DecodeArch::Mixtral,
    );
    runner.weights.clear();
    runner.weights.insert(
        "bf16".to_string(),
        poot_tensor::HostTensor::bf16(vec![4, 8], vec![0u16; 32]).into(),
    );
    runner.weights.insert(
        "f32".to_string(),
        poot_tensor::HostTensor::f32(vec![4, 8], vec![0.0f32; 32]).into(),
    );
    assert_eq!(
        runner.weight_bytes(),
        64 + 128,
        "the bf16 weight counts 2 bytes per element, the f32 weight 4"
    );
}

/// Card 1008: `Runner::weight_value` binds a stored weight as stored. A stored dtype that differs from the
/// dtype the traced graph declares for its const is a typed error naming the weight and both dtypes, in every
/// direction (a stored BF16 weight bound to an F32 const included: no widening at bind).
///
/// Mutation: restore the BF16/F16 -> F32 widening arm in `weight_value`; the first case turns red.
#[test]
fn weight_value_binds_as_stored_and_refuses_every_dtype_mismatch() {
    use poot_graph_ir::TensorType;
    use poot_tensor::{DType, HostTensor};

    let mut runner = crate::core::decode_arch::fixtures::runner_for(
        crate::core::decode_arch::DecodeArch::Mixtral,
    );
    runner.weights.clear();
    let words: Vec<u16> = vec![0x3f80, 0x8000, 0xc02e, 0x0080, 0x7f80, 0x4049];
    runner.weights.insert(
        "bf16".to_string(),
        HostTensor::bf16(vec![2, 3], words.clone()).into(),
    );
    runner.weights.insert(
        "f32".to_string(),
        HostTensor::f32(vec![2, 3], vec![1.0; 6]).into(),
    );
    runner.weights.insert(
        "f16".to_string(),
        HostTensor::f16(vec![2, 3], vec![0x3c00; 6]).into(),
    );
    let declared = |dtype| TensorType::new(vec![2, 3], dtype);

    let as_declared = runner
        .weight_value("bf16", &declared(DType::BF16))
        .expect("a matching dtype binds as stored");
    assert_eq!(
        as_declared.into_host().unwrap().as_half().unwrap(),
        words.as_slice()
    );

    for (name, stored, declared_dtype) in [
        ("bf16", DType::BF16, DType::F32),
        ("f16", DType::F16, DType::F32),
        ("f32", DType::F32, DType::BF16),
        ("f16", DType::F16, DType::BF16),
    ] {
        let error = runner
            .weight_value(name, &declared(declared_dtype))
            .expect_err("a stored/declared dtype mismatch is refused");
        assert!(
            matches!(
                super::stored_dtype_tests::stored_dtype_error(&error),
                crate::core::runner::weights::StoredDtypeError::Mismatch {
                    name: n, stored: s, declared: d
                } if n == name && *s == stored && *d == declared_dtype
            ),
            "{error}"
        );
    }
}

/// Write `config.json` into a fresh directory and return it. The loader must refuse before it looks for a
/// tokenizer or any weight file, so the directory holds nothing else.
fn config_only_dir(name: &str, config: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("poot_arch_refusal_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    std::fs::write(dir.join("config.json"), config).expect("write config.json");
    dir
}

fn load_refusal(dir: &std::path::Path) -> RunnerError {
    let loaded = Runner::load(dir);
    let _ = std::fs::remove_dir_all(dir);
    match loaded {
        Ok(_) => panic!("{} must not load", dir.display()),
        Err(error) => error,
    }
}

// Card 573 SC-001: a DeepSeek-V4 checkpoint (flat config, `model_type: "deepseek_v4"`) is refused at load
// with the typed unknown-architecture error naming the architecture string. The exact route that used to
// admit it is deleted, so there is no fallback: not a panic, not a CPU route.
#[test]
fn deepseek_v4_checkpoint_is_refused_with_the_typed_unknown_architecture_error() {
    let dir = config_only_dir(
        "deepseek_v4",
        r#"{"model_type":"deepseek_v4","architectures":["DeepseekV4ForCausalLM"],
            "vocab_size":129280,"hidden_size":4096,"intermediate_size":2048,
            "num_hidden_layers":43,"num_attention_heads":64,"num_key_value_heads":1,
            "max_position_embeddings":1048576,"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-6}"#,
    );
    let error = load_refusal(&dir);
    let RunnerError::UnsupportedModel { model_type, .. } = &error else {
        panic!("expected RunnerError::UnsupportedModel, got {error:?}");
    };
    assert_eq!(model_type, "deepseek_v4");
    assert!(
        error
            .to_string()
            .starts_with("deepseek_v4 is not supported"),
        "the message names the architecture string: {error}"
    );
}

// The other exact families are refused the same way, including the nested VLM-style configs that the flat
// `Qwen2HfConfig` parse cannot read: the architecture check runs on the raw `model_type` first.
#[test]
fn every_deleted_exact_family_is_refused_with_its_architecture_string() {
    for (model_type, config) in [
        // Card 595 SC-002: gemma4 and qwen3_next lost their private runtimes; the loader names them like any
        // other unknown architecture until cards 574 and 575 land them on the driver.
        ("gemma4", r#"{"model_type":"gemma4"}"#),
        ("qwen3_next", r#"{"model_type":"qwen3_next"}"#),
        ("minimax_m2", r#"{"model_type":"minimax_m2"}"#),
        (
            "glm5_next",
            r#"{"model_type":"glm5_next","text_config":{"model_type":"glm5_next_text"}}"#,
        ),
        (
            "qwen3_5",
            r#"{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text"}}"#,
        ),
        (
            "qwen4_exp",
            r#"{"model_type":"qwen4_exp","text_config":{"model_type":"qwen4_exp_text"}}"#,
        ),
    ] {
        let dir = config_only_dir(model_type, config);
        let error = load_refusal(&dir);
        let RunnerError::UnsupportedModel {
            model_type: refused,
            ..
        } = &error
        else {
            panic!("{model_type}: expected RunnerError::UnsupportedModel, got {error:?}");
        };
        assert_eq!(refused, model_type);
    }
}

// Card 190 class: a `config.json` with no string `model_type` must be refused before any weight read. The
// flat config below is otherwise a complete qwen2 config, so accepting it would build the qwen2 tracer for
// an unnamed architecture.
#[test]
fn config_without_a_string_model_type_is_refused_before_any_weight_read() {
    let fields = r#""vocab_size":32,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,
        "num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":16"#;
    for (name, model_type) in [
        ("absent", String::new()),
        ("number", r#""model_type":3,"#.to_string()),
        ("null", r#""model_type":null,"#.to_string()),
    ] {
        let dir = config_only_dir(
            &format!("no_model_type_{name}"),
            &format!("{{{model_type}{fields}}}"),
        );
        let error = load_refusal(&dir);
        let RunnerError::UnsupportedModel { reason, .. } = &error else {
            panic!("{name}: expected RunnerError::UnsupportedModel, got {error:?}");
        };
        assert_eq!(*reason, "config.json has no string model_type", "{name}");
    }
}

// Card 595 SC-002 (GGUF route): a GGUF whose `general.architecture` is gemma4 or qwen35moe (Qwen3-Next) is
// refused with the typed unknown-architecture error naming the architecture string, before any tensor is
// read. The file holds metadata only, so a loader that kept a route for either family would have to read
// tensors (and fail differently) rather than name the architecture.
#[test]
fn gemma4_and_qwen3next_ggufs_are_refused_with_the_typed_unknown_architecture_error() {
    use poot_load::gguf::{GgufValue, write_gguf};
    for arch in ["gemma4", "qwen35moe"] {
        let bytes = write_gguf(
            &[("general.architecture", GgufValue::Str(arch.to_string()))],
            &[],
        );
        let path = poot_test_util::unique_temp_path(format!("{arch}.gguf"));
        std::fs::write(&path, bytes).expect("write metadata-only gguf");
        let error = match Runner::load_gguf(&path) {
            Ok(_) => panic!("{arch} gguf must not load"),
            Err(error) => error,
        };
        let RunnerError::UnsupportedModel { model_type, .. } = &error else {
            panic!("{arch}: expected RunnerError::UnsupportedModel, got {error:?}");
        };
        assert_eq!(model_type, arch);
    }
}

/// POOT-737: a registered family never loads on the Runner. A checkpoint whose config names one is
/// refused before any tensor is read, as the typed `RunnerError::RegisteredFamily` naming the family:
/// it runs through `driver::ModelHandle` and the driver. Mutation: drop `refuse_registered` from
/// `load_impl`; the load falls through to the Runner's own list (an `UnsupportedModel`) and the row
/// goes red.
#[test]
fn a_registered_familys_checkpoint_is_refused_by_the_runner() {
    for model_type in ["qwen2", "llama", "granite", "bloom"] {
        let dir = poot_test_util::unique_temp_path(format!("poot_card737_registered_{model_type}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({ "model_type": model_type }).to_string(),
        )
        .unwrap();
        match Runner::load(&dir) {
            Err(RunnerError::RegisteredFamily { family }) => {
                assert_eq!(family.as_str(), model_type);
            }
            Err(other) => panic!("{model_type}: expected RegisteredFamily, got {other}"),
            Ok(_) => panic!("{model_type}: a registered family loaded on the Runner"),
        }
    }
}
