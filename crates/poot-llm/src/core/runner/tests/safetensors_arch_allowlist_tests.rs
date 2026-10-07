use super::super::unsupported_safetensors_arch;
use crate::{Runner, RunnerError};

// Card-190-class regression: the safetensors `Runner::load` path must refuse any model_type it cannot
// trace instead of falling through to another family's tracer. Every decoder family is a registered
// `Model` now, so no model_type loads on the Runner. Pure-host: no config, SafeTensors, or file.
#[test]
fn unsupported_safetensors_arch_refuses_every_model_type() {
    for arch in [
        "qwen2",
        "llama",
        "gemma4",
        "qwen3_next",
        "qwen2_vl",
        "qwen2_5_vl",
        "gpt-oss",
        "deepseek_v3",
        "nemotron_h",
        "totally_unknown",
    ] {
        let error = unsupported_safetensors_arch(arch).to_string();
        assert!(
            !error.is_empty(),
            "expected {arch:?} to be refused with a typed error"
        );
    }
    let qwen25_vl_err = unsupported_safetensors_arch("qwen2_5_vl").to_string();
    assert!(
        qwen25_vl_err.contains("load_qwen2_5_vl_text_tower_config_from_files"),
        "qwen2_5_vl error should name the closed config handoff: {qwen25_vl_err}"
    );
    assert!(matches!(
        unsupported_safetensors_arch("llama"),
        RunnerError::UnsupportedModel { .. }
    ));
}

/// Card 540b SC-003: `Runner::weight_bytes` must count each weight's stored payload at its stored
/// dtype width (R475-011): a bf16 tensor's resident upload is its 2 bytes per element, an f32 tensor's
/// 4. Reading an f32-width count for every dense weight (the pre-R475-011 bug) would overcount the
/// bf16 weight 2x.
#[test]
fn weight_bytes_counts_each_weights_stored_dtype_width() {
    let mut runner = crate::core::runner::fixtures::bare_runner();
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
        // Card 595 SC-002: qwen3_next lost its private runtime; the loader names it like any other unknown
        // architecture until card 575 lands it on the driver.
        ("qwen3_next", r#"{"model_type":"qwen3_next"}"#),
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

// Card 595 SC-002 (GGUF route): a GGUF whose `general.architecture` is qwen35moe (Qwen3-Next) is refused
// with the typed unknown-architecture error naming the architecture string, before any tensor is read; a
// gemma4 GGUF is a registered family (POOT-574) and is refused as one. The file holds metadata only, so a
// loader that kept a route for either family would have to read tensors (and fail differently) rather
// than name the architecture.
#[test]
fn qwen3next_and_gemma4_ggufs_are_refused_by_the_runner_naming_the_architecture() {
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
        match (arch, &error) {
            ("gemma4", RunnerError::RegisteredFamily { family }) => {
                assert_eq!(family.as_str(), arch);
            }
            ("qwen35moe", RunnerError::UnsupportedModel { model_type, .. }) => {
                assert_eq!(model_type, arch);
            }
            _ => panic!("{arch}: unexpected refusal {error:?}"),
        }
    }
}

/// POOT-737: a registered family never loads on the Runner. A checkpoint whose config names one is
/// refused before any tensor is read, as the typed `RunnerError::RegisteredFamily` naming the family:
/// it runs through `driver::ModelHandle` and the driver. Mutation: drop `refuse_registered` from
/// `load_impl`; the load falls through to the Runner's own list (an `UnsupportedModel`) and the row
/// goes red.
#[test]
fn a_registered_familys_checkpoint_is_refused_by_the_runner() {
    for model_type in ["qwen2", "llama", "granite", "bloom", "minimax_m2", "gemma4"] {
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
