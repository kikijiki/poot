use super::*;
use poot_tensor::DType;

use poot_quant::scalar::{e4m3fn_to_f32, e8m0_to_f32, f16_to_f32};
use poot_quant::weights::{WeightEntry, WeightStore};
use safetensors::{load_weight_store_bytes, pack_quantized_linears};

fn cfg(model_type: &str, attention_bias: Option<bool>) -> Qwen2HfConfig {
    Qwen2HfConfig {
        model_type: model_type.to_string(),
        vocab_size: 1,
        hidden_size: 8,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        rms_norm_eps: 1e-6,
        rope_theta: Some(1e4),
        rope_parameters: None,
        max_position_embeddings: 16,
        tie_word_embeddings: false,
        head_dim: None,
        partial_rotary_factor: None,
        original_max_position_embeddings: None,
        attention_bias,
        eos_token_id: Some(0),
        bos_token_id: None,
        rope_scaling: None,
        query_pre_attn_scalar: None,
        rope_local_base_freq: None,
        sliding_window_pattern: None,
        sliding_window: None,
        use_sliding_window: None,
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        num_local_experts: None,
        num_experts_per_tok: None,
        num_experts: None,
        moe_intermediate_size: None,
        norm_topk_prob: None,
        decoder_sparse_step: None,
        mlp_only_layers: None,
        embedding_multiplier: None,
        attention_multiplier: None,
        residual_multiplier: None,
        logits_scaling: None,
        quantization_config: None,
        swiglu_limit: None,
        layer_types: None,
        kv_lora_rank: None,
        q_lora_rank: None,
        qk_nope_head_dim: None,
        qk_rope_head_dim: None,
        v_head_dim: None,
        n_routed_experts: None,
        n_shared_experts: None,
        first_k_dense_replace: None,
        routed_scaling_factor: None,
        n_group: None,
        topk_group: None,
        index_n_heads: None,
        index_head_dim: None,
        index_topk: None,
        hc_mult: None,
        hc_sinkhorn_iters: None,
        hc_eps: None,
        o_groups: None,
        o_lora_rank: None,
        compress_rope_theta: None,
        compress_ratios: None,
    }
}

/// Build a minimal safetensors byte blob covering the resident dtype branches: 8-byte LE header
/// length, JSON header, then contiguous little-endian data.
fn synthetic_safetensors() -> Vec<u8> {
    // Payloads (little-endian) laid out contiguously in the data section.
    let mut data = Vec::new();
    let f32_off = data.len();
    for v in [1.0f32, -2.0, 3.5, 0.0] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let bf16_off = data.len();
    for bits in [0x3f80u16, 0xbf00, 0x4049] {
        // bf16 for 1.0, -0.5, ~3.14
        data.extend_from_slice(&bits.to_le_bytes());
    }
    let f16_off = data.len();
    for bits in [0x3c00u16, 0xc000] {
        // f16 for 1.0, -2.0
        data.extend_from_slice(&bits.to_le_bytes());
    }
    let e4m3_off = data.len();
    data.extend_from_slice(&[0x38, 0xb8, 0x40]);
    let i32_off = data.len();
    for w in [7i32, -3] {
        data.extend_from_slice(&w.to_le_bytes());
    }
    let i64_off = data.len();
    data.extend_from_slice(&42i64.to_le_bytes());
    let end = data.len();
    let header = serde_json::json!({
        "w_f32":  {"dtype": "F32",  "shape": [2, 2], "data_offsets": [f32_off, bf16_off]},
        "w_bf16": {"dtype": "BF16", "shape": [3],    "data_offsets": [bf16_off, f16_off]},
        "w_f16":  {"dtype": "F16",  "shape": [2],    "data_offsets": [f16_off, e4m3_off]},
        "w_e4m3": {"dtype": "F8_E4M3", "shape": [3], "data_offsets": [e4m3_off, i32_off]},
        "idx":    {"dtype": "I32",  "shape": [2],    "data_offsets": [i32_off, i64_off]},
        "position_ids": {"dtype": "I64", "shape": [1], "data_offsets": [i64_off, end]},
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut blob = Vec::new();
    blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&header_bytes);
    blob.extend_from_slice(&data);
    blob
}

/// The file entry point (`load`) and the in-memory one (`load_bytes`) must parse a blob
/// identically (they share one parser).
fn dense_of<'a>(store: &'a WeightStore, name: &str) -> &'a poot_quant::weights::DenseWeight {
    match store.get(name).unwrap() {
        WeightEntry::Dense(dense) => dense,
        WeightEntry::Packed(_) => panic!("{name}: expected a dense entry"),
    }
}

/// Build a [`WeightStore`] from `(name, dtype, shape, bytes)` rows, for tests that used to build a
/// `SafeTensors { tensors, ints, .. }` literal by hand.
fn store_of(entries: Vec<(&str, DType, Vec<usize>, Vec<u8>)>) -> WeightStore {
    let mut builder = WeightStore::builder();
    for (name, dtype, shape, bytes) in entries {
        let dense = poot_quant::weights::DenseWeight::try_new(dtype, shape, bytes.into()).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    builder.build()
}

fn u32_words_le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn f32_words_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// f16 bits of a normal `value` with at most 10 mantissa bits (exact, no rounding).
fn f16_bits_exact(value: f32) -> u16 {
    let bits = value.to_bits();
    assert_eq!(bits & 0x1fff, 0, "{value} is not exact in f16");
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exponent), "{value} is not a normal f16");
    (((bits >> 16) & 0x8000) | ((exponent as u32) << 10) | ((bits >> 13) & 0x3ff)) as u16
}

fn f16_words_le(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| f16_bits_exact(v).to_le_bytes())
        .collect()
}

/// Row `row` of `payload`, decoded through the one per-row entry.
fn decoded_row(payload: &poot_quant::PackedPayload, row: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; payload.weight().shape()[1]];
    payload.decode_row(row, &mut out).unwrap();
    out
}

/// The one packed payload `pack_quantized_linears` built for linear `prefix`.
fn packed_linear(store: &WeightStore, prefix: &str) -> std::sync::Arc<poot_quant::PackedPayload> {
    match store.get(&format!("{prefix}.weight")) {
        Some(WeightEntry::Packed(payload)) => payload.clone(),
        other => panic!("{prefix}.weight is not packed: {other:?}"),
    }
}

#[test]
fn load_and_load_bytes_agree_on_the_same_archive() {
    let blob = synthetic_safetensors();
    let from_bytes = load_weight_store_bytes(&blob).expect("load_weight_store_bytes");

    let dir = poot_test_util::unique_temp_path("poot-load-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("st-{}.safetensors", std::process::id()));
    std::fs::write(&path, &blob).unwrap();
    let from_file = safetensors::load_weight_store_file(&path).expect("load_weight_store_file");

    for name in ["w_f32", "w_bf16", "w_f16", "w_e4m3", "idx"] {
        let a = dense_of(&from_file, name);
        let b = dense_of(&from_bytes, name);
        assert_eq!(a.shape(), b.shape(), "{name} shape");
        assert_eq!(a.dtype(), b.dtype(), "{name} dtype");
        assert_eq!(
            a.bytes().as_slice(),
            b.bytes().as_slice(),
            "{name} stored bytes"
        );
    }
    // Bytes are kept exactly as stored (no widening): a BF16 tensor's stored bytes are 2/elem, an
    // F32 tensor's are 4/elem, never decoded here.
    assert_eq!(dense_of(&from_file, "w_f32").dtype(), DType::F32);
    assert_eq!(dense_of(&from_file, "w_f32").bytes().len(), 4 * 4);
    assert_eq!(dense_of(&from_file, "w_bf16").dtype(), DType::BF16);
    assert_eq!(dense_of(&from_file, "w_bf16").bytes().len(), 3 * 2);
    assert_eq!(dense_of(&from_file, "w_f16").dtype(), DType::F16);
    assert_eq!(dense_of(&from_file, "w_f16").bytes().len(), 2 * 2);
    assert_eq!(dense_of(&from_file, "w_e4m3").dtype(), DType::E4M3FN);
    assert_eq!(dense_of(&from_file, "w_e4m3").bytes().len(), 3);
    // I64 index/position buffers are skipped by both entry points (never a weight).
    assert!(!from_file.contains("position_ids") && !from_bytes.contains("position_ids"));
    // I32 tensors land as ordinary dense entries.
    assert_eq!(dense_of(&from_file, "idx").dtype(), DType::I32);
    assert_eq!(
        dense_of(&from_file, "idx").bytes().as_slice(),
        &7i32
            .to_le_bytes()
            .iter()
            .chain((-3i32).to_le_bytes().iter())
            .copied()
            .collect::<Vec<u8>>()[..]
    );
}

#[test]
fn safetensors_rejects_malformed_header_instead_of_panicking() {
    // A .safetensors is an untrusted file; a crafted header must yield a LoadError, never a
    // panic on a slice, index, or unwrap.
    let blob = |header: serde_json::Value, data: &[u8]| -> Vec<u8> {
        let hb = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        out.extend_from_slice(&hb);
        out.extend_from_slice(data);
        out
    };
    let reject = |label: &str, archive: &[u8]| {
        assert!(
            load_weight_store_bytes(archive).is_err(),
            "{label}: byte load must error"
        );
        let dir = poot_test_util::unique_temp_path("poot-load-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "malformed-{label}-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, archive).unwrap();
        let result = safetensors::load_weight_store_file(&path);
        assert!(result.is_err(), "{label}: file load must error");
    };

    // (D) data_offsets end past the data section (`data[start..end]` OOB panic).
    let past_eof = blob(
        serde_json::json!({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 1_000_000]}}),
        &[0u8; 8],
    );
    reject("past-eof", &past_eof);

    // (D) reversed range, start > end (slice panic).
    let reversed = blob(
        serde_json::json!({"w": {"dtype": "F32", "shape": [1], "data_offsets": [8, 4]}}),
        &[0u8; 16],
    );
    reject("reversed", &reversed);

    // (B) data_offsets with the wrong arity (`offs[1]` index panic).
    let bad_arity = blob(
        serde_json::json!({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0]}}),
        &[0u8; 4],
    );
    reject("bad-arity", &bad_arity);

    // (C) non-integer offset (`.as_u64().unwrap()` panic).
    let non_int = blob(
        serde_json::json!({"w": {"dtype": "F32", "shape": [1], "data_offsets": ["x", "y"]}}),
        &[0u8; 4],
    );
    reject("non-integer", &non_int);

    let malformed_shape = blob(
        serde_json::json!({"w": {"dtype": "F32", "shape": ["four"], "data_offsets": [0, 4]}}),
        &[0u8; 4],
    );
    reject("malformed-shape", &malformed_shape);

    for (dtype, bytes_per_element) in [
        ("F32", 4usize),
        ("BF16", 2),
        ("F16", 2),
        ("F8_E4M3", 1),
        ("F8_E8M0", 1),
        ("I32", 4),
        ("U32", 4),
        ("I64", 8),
    ] {
        let data = vec![0u8; bytes_per_element + 1];
        let trailing_partial = blob(
            serde_json::json!({
                "w": {
                    "dtype": dtype,
                    "shape": [1],
                    "data_offsets": [0, data.len()],
                }
            }),
            &data,
        );
        reject(&format!("trailing-partial-{dtype}"), &trailing_partial);
    }

    // (A) header length near u64::MAX: `8 + header_len` overflowed, wrapped small, and
    // panicked on `bytes[8..header_end]` (start > end).
    let mut overflow = Vec::new();
    overflow.extend_from_slice(&u64::MAX.to_le_bytes());
    overflow.extend_from_slice(b"{}padding-bytes");
    reject("header-overflow", &overflow);

    // A valid archive still loads.
    assert!(
        load_weight_store_bytes(&synthetic_safetensors()).is_ok(),
        "a valid archive must still load"
    );
}

fn proc_status_kib(field: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    status
        .lines()
        .find_map(|line| {
            let rest = line.strip_prefix(field)?.strip_prefix(':')?;
            rest.split_whitespace().next()?.parse().ok()
        })
        .unwrap_or_else(|| panic!("missing {field} in /proc/self/status"))
}

fn write_bounded_bf16_archive(path: &Path, tensor_count: usize, tensor_bytes: usize) {
    use std::io::Write as _;

    assert_eq!(tensor_bytes % 2, 0);
    let mut header = serde_json::Map::new();
    for index in 0..tensor_count {
        let start = index * tensor_bytes;
        let end = start + tensor_bytes;
        header.insert(
            format!("tensor_{index:03}"),
            serde_json::json!({
                "dtype": "BF16",
                "shape": [tensor_bytes / 2],
                "data_offsets": [start, end],
            }),
        );
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    let mut payload = vec![0u8; tensor_bytes];
    for pair in payload.chunks_exact_mut(2) {
        pair.copy_from_slice(&0x3f80u16.to_le_bytes());
    }
    for _ in 0..tensor_count {
        file.write_all(&payload).unwrap();
    }
    file.sync_all().unwrap();
}

#[test]
fn file_load_peak_rss_is_bounded() {
    const CHILD: &str = "POOT_CARD319_RSS_CHILD";
    const ARCHIVE: &str = "POOT_CARD319_RSS_ARCHIVE";
    const TENSOR_COUNT: usize = 128;
    const TENSOR_BYTES: usize = 1024 * 1024;
    const PAYLOAD_BYTES: usize = TENSOR_COUNT * TENSOR_BYTES;

    if std::env::var_os(CHILD).is_some() {
        let path = std::env::var_os(ARCHIVE).expect("RSS archive path");
        let baseline_rss = proc_status_kib("VmRSS") * 1024;
        let baseline_peak = proc_status_kib("VmHWM") * 1024;
        let loaded = safetensors::load_weight_store_file(&path).expect("bounded file load");
        std::hint::black_box(&loaded);
        let peak_rss = proc_status_kib("VmHWM") * 1024;
        let incremental_peak = peak_rss.saturating_sub(baseline_rss);
        // Card 540b: the store keeps every tensor's bytes exactly as stored (bf16, 2 bytes/elem),
        // never widened to f32 - so this bound is much tighter than the old reader's ~3.25x (which
        // kept a decoded f32 copy alongside the native bytes for every tensor).
        let limit = (PAYLOAD_BYTES * 3 / 2 + 16 * 1024 * 1024) as u64;
        println!(
            "card319 rss: payload_bytes={PAYLOAD_BYTES} baseline_rss_bytes={baseline_rss} \
                 baseline_peak_bytes={baseline_peak} peak_rss_bytes={peak_rss} \
                 incremental_peak_bytes={incremental_peak} limit_bytes={limit}"
        );
        assert_eq!(loaded.len(), TENSOR_COUNT);
        let tensor_000 = dense_of(&loaded, "tensor_000");
        assert_eq!(tensor_000.dtype(), DType::BF16);
        assert_eq!(
            tensor_000.bytes().as_slice()[0..2],
            0x3f80u16.to_le_bytes(),
            "fixture must exercise BF16 stored bytes (1.0)"
        );
        assert!(
            incremental_peak <= limit,
            "incremental peak RSS {incremental_peak} exceeds {limit} bytes"
        );
        return;
    }

    let dir = poot_test_util::unique_temp_path("poot-load-card319");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("rss-{}.safetensors", std::process::id()));
    write_bounded_bf16_archive(&path, TENSOR_COUNT, TENSOR_BYTES);
    assert_eq!(
        std::fs::metadata(&path).unwrap().len() as usize - 8 - {
            let mut file = File::open(&path).unwrap();
            let mut len = [0u8; 8];
            file.read_exact(&mut len).unwrap();
            u64::from_le_bytes(len) as usize
        },
        PAYLOAD_BYTES,
        "fixture source payload"
    );

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("tests::file_load_peak_rss_is_bounded")
        .arg("--nocapture")
        .env(CHILD, "1")
        .env(ARCHIVE, &path)
        .output()
        .expect("spawn fresh RSS process");
    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    assert!(
        output.status.success(),
        "fresh RSS process exited {}",
        output.status
    );
}

#[test]
fn phi3_longrope_config_parses() {
    // Phi-3/Phi-4 config shape: rope_scaling uses the LongRoPE schema (`type: "longrope"`
    // with `long_factor`/`short_factor`, no `factor`) and the config carries
    // `partial_rotary_factor`. Trimmed from phi-4-mini/config.json. phi3 keeps
    // `original_max_position_embeddings` at the top level (llama3 nests it in rope_scaling);
    // reading the wrong place gave 0, an infinite LongRoPE attention factor, and a poisoned
    // forward.
    let json = r#"{
            "model_type": "phi3", "vocab_size": 200064, "hidden_size": 3072,
            "intermediate_size": 8192, "num_hidden_layers": 32, "num_attention_heads": 24,
            "num_key_value_heads": 8, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "max_position_embeddings": 131072, "tie_word_embeddings": true,
            "partial_rotary_factor": 0.75,
            "original_max_position_embeddings": 4096,
            "rope_scaling": {
                "type": "longrope",
                "long_factor": [1.0, 1.1, 1.2],
                "short_factor": [1.0, 1.0, 1.0]
            }
        }"#;
    let c: Qwen2HfConfig = serde_json::from_str(json).expect("phi3 config must parse");
    // head_dim 3072/24 = 128; 0.75 * 128 = 96 (already even).
    assert_eq!(c.head_dim(), 128);
    assert_eq!(c.rotary_dim(), 96);
    assert_eq!(c.original_max_position_embeddings, Some(4096)); // top-level, not in rope_scaling
    let rs = c.rope_scaling.expect("rope_scaling present");
    assert_eq!(rs.rope_type, "longrope"); // came in under the `type` alias
    assert_eq!(rs.long_factor.unwrap().len(), 3);
    // A non-phi arch with no partial_rotary_factor keeps full rotary.
    assert_eq!(
        cfg("qwen2", None).rotary_dim(),
        cfg("qwen2", None).head_dim()
    );
}

#[test]
fn qkv_bias_per_arch_default() {
    // qwen2 and Qwen-VL text towers: no field means biased; mistral/llama: no field means unbiased.
    assert!(cfg("qwen2", None).qkv_bias());
    assert!(cfg("qwen2_vl", None).qkv_bias());
    assert!(cfg("qwen2_5_vl", None).qkv_bias());
    assert!(!cfg("mistral", None).qkv_bias());
    assert!(!cfg("llama", None).qkv_bias());
    // An explicit value wins over the per-arch default.
    assert!(!cfg("qwen2", Some(false)).qkv_bias());
    assert!(cfg("llama", Some(true)).qkv_bias());
}

#[test]
fn effective_sliding_window_honors_use_sliding_window_false() {
    // Qwen/Qwen2.5-0.5B-Instruct config.json shape: sliding_window=32768,
    // use_sliding_window=false, so the window is dormant.
    let mut c = cfg("qwen2", None);
    c.sliding_window = Some(32768);
    c.use_sliding_window = Some(false);
    assert_eq!(c.effective_sliding_window(), None);

    // use_sliding_window=true (or omitted) passes the raw value through unchanged.
    c.use_sliding_window = Some(true);
    assert_eq!(c.effective_sliding_window(), Some(32768));
    c.use_sliding_window = None;
    assert_eq!(c.effective_sliding_window(), Some(32768));

    // No sliding_window key at all -> None regardless of the flag (nothing to suppress).
    c.sliding_window = None;
    c.use_sliding_window = Some(false);
    assert_eq!(c.effective_sliding_window(), None);

    // Archs with no such flag (Mistral/Gemma, which never set use_sliding_window) pass
    // sliding_window through.
    let mistral = cfg("mistral", None);
    let mut mistral_swa = mistral.clone();
    mistral_swa.sliding_window = Some(4096);
    assert_eq!(mistral_swa.effective_sliding_window(), Some(4096));
}

#[test]
fn qwen3_moe_detected_and_reuses_qwen3_attention_flags() {
    let c = cfg("qwen3_moe", Some(false));
    assert!(c.is_qwen3_moe());
    assert!(!c.is_granite_moe());
    // qwen3-moe reuses qwen3's attention flags: no qkv bias, per-head QK-norm.
    assert!(!c.qkv_bias());
    assert!(c.qk_norm());
    // qwen3 dense also gets qk_norm; a plain qwen2/mistral does not.
    assert!(cfg("qwen3", None).qk_norm());
    assert!(!cfg("qwen2", None).qk_norm());
}

#[test]
fn mixtral_detected_and_defaults_to_no_bias_no_qk_norm() {
    let c = cfg("mixtral", None);
    assert!(c.is_mixtral());
    assert!(!c.is_granite_moe());
    assert!(!c.is_qwen3_moe());
    // Mixtral has no qkv bias and no QK-norm (unlike qwen3-moe): plain Mistral-shaped attention.
    assert!(!c.qkv_bias());
    assert!(!c.qk_norm());
}

#[test]
fn deepseek2_detected_and_mla_fields_parse() {
    let mut c = cfg("deepseek_v2", None);
    assert!(c.is_deepseek2());
    assert!(!c.is_mixtral());
    assert!(!c.is_olmoe());
    assert!(!c.is_gpt_oss());
    // q_lora_rank absent (DeepSeek-V2-Lite: Q-side compression disabled).
    assert_eq!(c.q_lora_rank, None);
    c.q_lora_rank = Some(1536);
    c.kv_lora_rank = Some(512);
    c.qk_nope_head_dim = Some(128);
    c.qk_rope_head_dim = Some(64);
    c.v_head_dim = Some(128);
    c.n_routed_experts = Some(160);
    c.n_shared_experts = Some(2);
    c.first_k_dense_replace = Some(1);
    c.routed_scaling_factor = Some(1.0);
    assert_eq!(c.q_lora_rank, Some(1536));
    assert_eq!(c.kv_lora_rank, Some(512));
}

#[test]
fn deepseek3_detected_and_group_routing_fields_parse() {
    // `deepseek-ai/DeepSeek-V3` `config.json`: `model_type: "deepseek_v3"` (distinct from V2's
    // "deepseek_v2", unlike the GGUF path's shared "deepseek2"), plus flat top-level
    // `n_group`/`topk_group` group-limited routing fields (not nested under a MoE sub-block).
    let json = r#"{
            "model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"],
            "vocab_size": 129280, "hidden_size": 7168, "intermediate_size": 18432,
            "num_hidden_layers": 61, "num_attention_heads": 128, "num_key_value_heads": 128,
            "rms_norm_eps": 1e-6, "rope_theta": 10000, "max_position_embeddings": 163840,
            "tie_word_embeddings": false,
            "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
            "v_head_dim": 128, "n_routed_experts": 256, "n_shared_experts": 1,
            "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "first_k_dense_replace": 3,
            "routed_scaling_factor": 2.5, "n_group": 8, "topk_group": 4,
            "topk_method": "noaux_tc", "scoring_func": "sigmoid", "norm_topk_prob": true,
            "bos_token_id": 0, "eos_token_id": 1
        }"#;
    let c: Qwen2HfConfig =
        serde_json::from_str(json).expect("real DeepSeek-V3 config.json must parse");
    assert!(c.is_deepseek3());
    assert!(!c.is_deepseek2());
    assert_eq!(c.n_group, Some(8));
    assert_eq!(c.topk_group, Some(4));
    assert_eq!(c.first_k_dense_replace, Some(3));
    assert_eq!(c.n_shared_experts, Some(1));
    assert_eq!(c.n_routed_experts, Some(256));
    assert_eq!(c.routed_scaling_factor, Some(2.5));
    assert!(!c.tie_word_embeddings);
}

#[test]
fn deepseek2_yarn_rope_scaling_parses_mscale_and_mscale_all_dim() {
    // `deepseek-ai/DeepSeek-V2-Lite`/`-Chat` rope_scaling block: `type: "yarn"`,
    // `mscale`/`mscale_all_dim` both `0.707`. `RopeScaling` once lacked these fields; unknown
    // keys parse, but the values were silently dropped, which would give a wrong
    // attention_factor/softmax scale.
    let json = r#"{
            "model_type": "deepseek_v2", "vocab_size": 102400, "hidden_size": 2048,
            "intermediate_size": 10944, "num_hidden_layers": 27, "num_attention_heads": 16,
            "num_key_value_heads": 16, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "max_position_embeddings": 163840,
            "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
            "n_routed_experts": 64, "n_shared_experts": 2, "num_experts_per_tok": 6,
            "moe_intermediate_size": 1408, "first_k_dense_replace": 1, "routed_scaling_factor": 1.0,
            "n_group": 1, "topk_group": 1, "topk_method": "greedy",
            "rope_scaling": {
                "type": "yarn",
                "factor": 40,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 0.707,
                "mscale_all_dim": 0.707,
                "original_max_position_embeddings": 4096
            }
        }"#;
    let c: Qwen2HfConfig = serde_json::from_str(json).expect("deepseek-v2-lite config must parse");
    assert!(c.is_deepseek2());
    let rs = c.rope_scaling.expect("rope_scaling present");
    assert_eq!(rs.rope_type, "yarn");
    assert_eq!(rs.factor, 40.0);
    assert_eq!(rs.mscale, Some(0.707));
    assert_eq!(rs.mscale_all_dim, Some(0.707));
    assert_eq!(rs.attention_factor, None); // no explicit override on the real checkpoint
}

#[test]
fn olmoe_detected_and_defaults_to_no_bias_no_generic_qk_norm() {
    let c = cfg("olmoe", None);
    assert!(c.is_olmoe());
    assert!(!c.is_mixtral());
    assert!(!c.is_qwen3_moe());
    assert!(!c.is_granite_moe());
    // OlmoE has no qkv bias (attention_bias: false) and does not set the per-head qk_norm()
    // flag: its QK-norm is full-dimension, handled in `poot_models::olmoe`'s tracer.
    assert!(!c.qkv_bias());
    assert!(!c.qk_norm());
}

#[test]
fn gpt_oss_detected_with_bias_and_no_generic_qk_norm() {
    let mut c = cfg("gpt_oss", Some(true));
    c.swiglu_limit = Some(7.0);
    c.layer_types = Some(vec![
        "sliding_attention".to_string(),
        "full_attention".to_string(),
    ]);
    assert!(c.is_gpt_oss());
    assert!(!c.is_mixtral());
    assert!(!c.is_olmoe());
    assert!(!c.is_qwen3_moe());
    assert!(!c.is_granite_moe());
    // gpt-oss has qkv bias (attention_bias: true) but no QK-norm at all (per
    // GptOssAttention.forward; see poot_models::gpt_oss).
    assert!(c.qkv_bias());
    assert!(!c.qk_norm());
    assert_eq!(c.swiglu_limit, Some(7.0));
    assert_eq!(
        c.layer_types,
        Some(vec![
            "sliding_attention".to_string(),
            "full_attention".to_string()
        ])
    );
}

#[test]
fn effective_rope_theta_prefers_nested_rope_parameters_over_flat_key() {
    let mut c = cfg("olmoe", None);
    // Flat key only (all checkpoints before card 135d).
    c.rope_theta = Some(10_000.0);
    assert_eq!(c.effective_rope_theta().unwrap(), 10_000.0);

    // hf-tiny-v2/tiny-random-OlmoeForCausalLM: only the nested key; the flat key is absent.
    c.rope_theta = None;
    c.rope_parameters = Some(RopeParameters {
        rope_theta: Some(20_000.0),
    });
    assert_eq!(c.effective_rope_theta().unwrap(), 20_000.0);

    // With both keys, the nested key wins.
    c.rope_theta = Some(999.0);
    assert_eq!(c.effective_rope_theta().unwrap(), 20_000.0);

    // rope_parameters present without rope_theta falls back to the flat key.
    c.rope_parameters = Some(RopeParameters { rope_theta: None });
    assert_eq!(c.effective_rope_theta().unwrap(), 999.0);
}

#[test]
fn rope_theta_config_json_without_a_top_level_key_parses_via_default() {
    // hf-tiny-v2/tiny-random-OlmoeForCausalLM shape: no top-level "rope_theta", only a nested
    // "rope_parameters": {"rope_theta": ..., "rope_type": "default"}. rope_theta was once a
    // required serde field, which made this config error on load.
    let json = br#"{
            "model_type": "olmoe",
            "vocab_size": 8,
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 16,
            "rope_parameters": {"rope_theta": 12345.0, "rope_type": "default"}
        }"#;
    let c: Qwen2HfConfig =
        serde_json::from_slice(json).expect("parse config without flat rope_theta");
    assert_eq!(c.rope_theta, None);
    assert_eq!(c.effective_rope_theta().unwrap(), 12345.0);
}

/// A config.json body with neither `rope_theta` nor `eos_token_id`, for `model_type`.
fn config_json_without_rope_theta_or_eos(model_type: &str) -> Qwen2HfConfig {
    let json = format!(
        r#"{{
            "model_type": "{model_type}",
            "vocab_size": 8,
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 16
        }}"#
    );
    serde_json::from_str(&json).expect("parse config without rope_theta or eos_token_id")
}

/// SC-002: a missing `rope_theta` resolves to the family's reference value, never 0 (which made NaN RoPE
/// tables); a family with no reference value is a typed error (next test).
#[test]
fn missing_rope_theta_resolves_to_the_family_reference() {
    // (model_type, transformers config-class default rope_theta), literals read from transformers 4.57.6
    // and 5.17.0
    let with_reference = [
        ("llama", 10_000.0),
        ("mistral", 10_000.0),
        ("qwen2", 10_000.0),
        ("deepseek_v2", 10_000.0),
        ("deepseek_v3", 10_000.0),
        ("deepseek_v32", 10_000.0),
        ("mixtral", 1_000_000.0),
        ("gemma3_text", 1_000_000.0),
        ("smollm3", 2_000_000.0),
        ("gpt_oss", 150_000.0),
    ];
    for (model_type, expected) in with_reference {
        let c = config_json_without_rope_theta_or_eos(model_type);
        assert_eq!(c.rope_theta, None, "{model_type}: key is absent");
        assert_eq!(c.effective_rope_theta().unwrap(), expected, "{model_type}");
    }
}

#[test]
fn missing_rope_theta_without_a_family_reference_is_a_typed_error() {
    for model_type in ["", "bloom", "no_such_family"] {
        let c = config_json_without_rope_theta_or_eos(model_type);
        match c.effective_rope_theta() {
            Err(LoadError::MissingRopeTheta { model_type: got }) => assert_eq!(got, model_type),
            other => panic!("{model_type:?}: expected MissingRopeTheta, got {other:?}"),
        }
    }
}

#[test]
fn explicit_rope_theta_that_is_not_positive_and_finite_is_a_typed_error() {
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        let mut c = cfg("llama", None);
        c.rope_theta = Some(bad);
        match c.effective_rope_theta() {
            Err(LoadError::InvalidRopeTheta { value, .. }) => {
                assert_eq!(value.to_bits(), bad.to_bits())
            }
            other => panic!("rope_theta {bad}: expected InvalidRopeTheta, got {other:?}"),
        }
    }
}

/// SC-003: a missing `eos_token_id` never resolves to another family's id: a family yields its own
/// reference eos, and a family whose reference declares none is a typed error (next test).
#[test]
fn missing_eos_token_id_resolves_to_the_family_reference() {
    // (model_type, transformers config-class default eos_token_id), as literals read from transformers
    // 4.57.6 and 5.17.0 (`AutoConfig.for_model(model_type).eos_token_id`), not copied from the loader's
    // table. Every id differs from the old Qwen default 151643 that every family used to get.
    let with_reference = [
        ("llama", 2),
        ("mistral", 2),
        ("gemma3_text", 1),
        ("phi3", 32000),
        ("olmo2", 50279),
        ("deepseek_v2", 2),
        ("deepseek_v3", 1),
        ("deepseek_v32", 1),
        ("smollm3", 128_001),
    ];
    for (model_type, expected) in with_reference {
        let c = config_json_without_rope_theta_or_eos(model_type);
        assert_eq!(c.eos_token_id, None, "{model_type}: key is absent");
        assert_eq!(
            c.effective_eos_token_id().unwrap(),
            expected,
            "{model_type}"
        );
    }
    // An explicit value always wins over the family reference.
    let mut c = config_json_without_rope_theta_or_eos("llama");
    c.eos_token_id = Some(7);
    assert_eq!(c.effective_eos_token_id().unwrap(), 7);
}

#[test]
fn missing_eos_token_id_without_a_family_reference_is_a_typed_error() {
    for model_type in ["", "qwen2", "qwen3", "gpt_oss", "no_such_family"] {
        let c = config_json_without_rope_theta_or_eos(model_type);
        match c.effective_eos_token_id() {
            Err(LoadError::MissingEosTokenId { model_type: got }) => assert_eq!(got, model_type),
            other => panic!("{model_type:?}: expected MissingEosTokenId, got {other:?}"),
        }
    }
}

#[test]
fn rms_norm_eps_config_json_without_the_key_parses_via_default() {
    // allenai/OLMoE-1B-7B-0924-Instruct config.json (card 135d, spec 262): the top-level
    // "rms_norm_eps" key is absent (OlmoE's tiny-random fixture sets it). It was once a
    // required serde field, giving a "missing field `rms_norm_eps`" error on load.
    let json = br#"{
            "model_type": "olmoe",
            "vocab_size": 8,
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 16,
            "rope_theta": 10000.0
        }"#;
    let c: Qwen2HfConfig = serde_json::from_slice(json).expect("parse config without rms_norm_eps");
    assert_eq!(c.rms_norm_eps, 1e-5);
}

#[test]
fn qwen3_moe_norm_topk_prob_rejects_false_accepts_true_or_absent() {
    let mut c = cfg("qwen3_moe", None);
    // Absent is accepted (no released checkpoint omits it; poot has no non-renormalized path).
    assert!(c.qwen3_moe_norm_topk_prob().is_ok());
    c.norm_topk_prob = Some(true);
    assert!(c.qwen3_moe_norm_topk_prob().is_ok());
    c.norm_topk_prob = Some(false);
    assert!(
        c.qwen3_moe_norm_topk_prob().is_err(),
        "norm_topk_prob=false must be rejected - poot's moe op has no non-renormalized form"
    );
}

#[test]
fn qwen3_moe_layer_is_sparse_matches_hf_decoder_sparse_step_semantics() {
    // HF defaults (decoder_sparse_step=1, mlp_only_layers=[]): every layer is MoE
    // (Qwen3-30B-A3B).
    let dense_every_layer = cfg("qwen3_moe", None);
    for li in 0..4 {
        assert!(dense_every_layer.qwen3_moe_layer_is_sparse(li));
    }

    // yujiepan/qwen3-moe-tiny-random's shape: decoder_sparse_step=2 -> layer_idx 0 dense (1%2!=0),
    // layer_idx 1 MoE (2%2==0), alternating: (li+1) % 2 == 0 iff li is odd.
    let mut alternating = cfg("qwen3_moe", None);
    alternating.decoder_sparse_step = Some(2);
    assert!(!alternating.qwen3_moe_layer_is_sparse(0));
    assert!(alternating.qwen3_moe_layer_is_sparse(1));
    assert!(!alternating.qwen3_moe_layer_is_sparse(2));
    assert!(alternating.qwen3_moe_layer_is_sparse(3));

    // mlp_only_layers forces a layer dense even when the step pattern would select it for MoE.
    let mut forced_dense = cfg("qwen3_moe", None);
    forced_dense.mlp_only_layers = Some(vec![1]);
    assert!(forced_dense.qwen3_moe_layer_is_sparse(0));
    assert!(
        !forced_dense.qwen3_moe_layer_is_sparse(1),
        "mlp_only_layers must force this layer dense even though decoder_sparse_step=1 selects every layer"
    );
}

#[test]
fn bf16_roundtrip() {
    // 1.0 in bf16 is 0x3F80; high 16 bits of f32 1.0 (0x3F800000).
    let raw = 0x3F80u16.to_le_bytes();
    let v = decode("BF16", &raw).unwrap();
    assert_eq!(v, vec![1.0]);
}

#[test]
fn f16_one() {
    // 1.0 in f16 is 0x3C00.
    assert_eq!(f16_to_f32(0x3C00), 1.0);
}

#[test]
fn gptq_linear_packs_its_checkpoint_bytes_and_decodes_to_the_reference() {
    // Tiny GPTQ linear: in=8, out=8, one group. The packed payload (card 545a: the checkpoint's own
    // qweight/qzeros/scales bytes, no dequant copy) must decode to (qw - (qz+1)) * scale (AutoGPTQ
    // convention); a trivial g_idx packs as the contiguous group map.
    let (in_dim, out) = (8usize, 8usize);
    let v = |i: usize, o: usize| ((i + o) % 16) as u32; // input nibble at (i,o)
    let z = |o: usize| (o % 7) as u32; // zero nibble per out col
    let scale = |o: usize| 0.125 * (o as f32 + 1.0);

    // qweight [in/8, out]: out col o packs input rows 0..8 low-to-high.
    let mut qweight = vec![0u32; out];
    for (o, qw) in qweight.iter_mut().enumerate() {
        for i in 0..in_dim {
            *qw |= v(i, o) << (4 * i);
        }
    }
    // qzeros [1, out/8]: packs 8 out zero-points into one word.
    let mut qz_word = 0u32;
    for o in 0..out {
        qz_word |= z(o) << (4 * o);
    }
    let scales: Vec<f32> = (0..out).map(scale).collect();

    let store = store_of(vec![
        (
            "l.qweight",
            DType::I32,
            vec![1, out],
            u32_words_le(&qweight),
        ),
        (
            "l.qzeros",
            DType::I32,
            vec![1, out / 8],
            u32_words_le(&[qz_word]),
        ),
        (
            "l.g_idx",
            DType::I32,
            vec![in_dim],
            u32_words_le(&vec![0u32; in_dim]),
        ),
        ("l.scales", DType::F16, vec![1, out], f16_words_le(&scales)),
    ]);

    let packed = pack_quantized_linears(
        &store,
        QuantScheme {
            kind: QuantKind::Gptq,
            group_size: 8,
        },
    )
    .unwrap();
    assert!(!packed.contains("l.qweight") && !packed.contains("l.g_idx"));
    let payload = packed_linear(&packed, "l");
    assert_eq!(payload.weight().shape(), [out, in_dim]);
    assert!(matches!(
        payload.weight().format(),
        poot_quant::format::WeightFormat::Gptq {
            groups: poot_quant::format::GroupMap::Contiguous { .. }
        }
    ));
    for i in 0..in_dim {
        for o in 0..out {
            let want = (v(i, o) as f32 - (z(o) as f32 + 1.0)) * scale(o);
            let got = decoded_row(&payload, o)[i];
            assert_eq!(got, want, "({i},{o})");
        }
    }
}

#[test]
fn fp8_e4m3_decode() {
    // OCP E4M3 reference values.
    assert_eq!(e4m3fn_to_f32(0x00), 0.0); // +0
    assert_eq!(e4m3fn_to_f32(0x38), 1.0); // exp=7,mant=0 -> 2^0 * 1.0
    assert_eq!(e4m3fn_to_f32(0xB8), -1.0); // sign + 1.0
    assert_eq!(e4m3fn_to_f32(0x7E), 448.0); // exp=15,mant=6 -> 2^8 * 1.75 (E4M3 max)
    assert_eq!(e4m3fn_to_f32(0x01), 2.0f32.powi(-9)); // smallest subnormal
    assert!(e4m3fn_to_f32(0x7F).is_nan()); // exp=15,mant=7 -> NaN
}

#[test]
fn fp8_e8m0_decode() {
    // OCP MX E8M0 reference values: value = 2^(byte-127), no sign, no mantissa, 0xFF is NaN.
    assert_eq!(e8m0_to_f32(127), 1.0); // 2^0
    assert_eq!(e8m0_to_f32(128), 2.0); // 2^1
    assert_eq!(e8m0_to_f32(126), 0.5); // 2^-1
    assert_eq!(e8m0_to_f32(0), 2.0f32.powi(-127)); // smallest representable value, NOT zero
    assert!(e8m0_to_f32(0xFF).is_nan());
}

#[test]
fn awq_linear_packs_its_checkpoint_bytes_and_decodes_to_the_reference() {
    // Tiny AWQ linear: in=8, out=8, one group. AWQ packs along the out axis with AutoAWQ's
    // [0, 4, 1, 5, 2, 6, 3, 7] interleave and is asymmetric (no +1): w == (qw - qz) * scale.
    const AWQ_ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    let (in_dim, out) = (8usize, 8usize);
    let v = |i: usize, o: usize| ((i * 3 + o) % 16) as u32;
    let z = |o: usize| (o % 5) as u32;
    let scale = |o: usize| 0.0625 * (o as f32 + 1.0);

    // qweight [in, out/8]: each input row packs 8 out nibbles at AWQ_ORDER positions.
    let mut qweight = vec![0u32; in_dim];
    for (i, qw) in qweight.iter_mut().enumerate() {
        for (o, &ord) in AWQ_ORDER.iter().enumerate().take(out) {
            *qw |= v(i, o) << (4 * ord);
        }
    }
    let mut qz_word = 0u32;
    for (o, &ord) in AWQ_ORDER.iter().enumerate().take(out) {
        qz_word |= z(o) << (4 * ord);
    }
    let scales: Vec<f32> = (0..out).map(scale).collect();

    let store = store_of(vec![
        (
            "l.qweight",
            DType::I32,
            vec![in_dim, out / 8],
            u32_words_le(&qweight),
        ),
        (
            "l.qzeros",
            DType::I32,
            vec![1, out / 8],
            u32_words_le(&[qz_word]),
        ),
        ("l.scales", DType::F16, vec![1, out], f16_words_le(&scales)),
    ]);

    let packed = pack_quantized_linears(
        &store,
        QuantScheme {
            kind: QuantKind::Awq,
            group_size: 8,
        },
    )
    .unwrap();
    let payload = packed_linear(&packed, "l");
    assert_eq!(payload.weight().shape(), [out, in_dim]);
    for i in 0..in_dim {
        for o in 0..out {
            let want = (v(i, o) as f32 - z(o) as f32) * scale(o);
            let got = decoded_row(&payload, o)[i];
            assert_eq!(got, want, "({i},{o})");
        }
    }
}

/// Acceptance: a DeepSeek-style block-FP8 linear (`weight_scale_inv`, one E8M0 scale per
/// 128x128 block) packs from its checkpoint bytes and decodes to `e4m3(q) * 2^(e - 127)`, and an
/// E8M0 `0xff` scale (NaN) is refused with a typed error at load, never widened into NaN weights.
/// Mutation: dropping the finiteness check in poot-quant's payload content validation turns the
/// refusal row red.
/// Card 654 (fail closed): an `F8_E4M3` linear `.weight` with no recognized scale
/// sibling - none at all, or only DeepSeek-V4's `.scale` - is refused naming the tensor, never passed
/// through as a dense tensor of unscaled codes. Mutation: removing the refusal leaves `l.weight` a
/// dense `F8E4m3` entry in the returned store, and both cases go red.
#[test]
fn fp8_weight_without_a_recognized_scale_sibling_is_refused() {
    let fp8 = QuantScheme {
        kind: QuantKind::Fp8,
        group_size: 0,
    };
    let weight = ("l.weight", DType::E4M3FN, vec![2, 4], vec![0x38; 8]);
    for store in [
        store_of(vec![weight.clone()]),
        store_of(vec![
            weight.clone(),
            ("l.scale", DType::E8M0, vec![1, 1], vec![128u8]),
        ]),
    ] {
        match pack_quantized_linears(&store, fp8) {
            Err(LoadError::QuantizedLinearLayout { tensor, .. }) => assert_eq!(tensor, "l.weight"),
            Err(other) => panic!("expected a typed layout refusal, got {other:?}"),
            Ok(packed) => panic!(
                "unscaled F8_E4M3 codes passed through: l.weight is {:?}",
                packed.get("l.weight").map(|entry| match entry {
                    WeightEntry::Dense(dense) => format!("dense {:?}", dense.dtype()),
                    WeightEntry::Packed(_) => "packed".to_string(),
                })
            ),
        }
    }
}

#[test]
fn block_fp8_linear_packs_and_refuses_a_nan_e8m0_scale() {
    let codes: Vec<u8> = vec![0x38, 0x40, 0xb8, 0x01, 0x30, 0x44, 0x00, 0xc0];
    let fp8 = QuantScheme {
        kind: QuantKind::Fp8,
        group_size: 0,
    };
    let store = |scale: u8| {
        store_of(vec![
            ("l.weight", DType::E4M3FN, vec![2, 4], codes.clone()),
            ("l.weight_scale_inv", DType::E8M0, vec![1, 1], vec![scale]),
        ])
    };
    let packed = pack_quantized_linears(&store(128), fp8).unwrap();
    let payload = packed_linear(&packed, "l");
    assert!(matches!(
        payload.weight().format(),
        poot_quant::format::WeightFormat::E4m3Block128 {
            scale: poot_quant::format::ScaleEncoding::E8m0
        }
    ));
    for (index, &code) in codes.iter().enumerate() {
        let want = e4m3fn_to_f32(code) * e8m0_to_f32(128);
        assert_eq!(decoded_row(&payload, index / 4)[index % 4], want, "{index}");
    }
    assert!(matches!(
        pack_quantized_linears(&store(0xff), fp8),
        Err(LoadError::QuantizedLinearPayload {
            ref linear,
            source: poot_quant::PackedWeightError::NonFiniteField { .. },
        }) if linear == "l.weight"
    ));
}

/// Review minors 9/10: an untrusted quantized-linear layout is a typed refusal, never a panic or a
/// misclassification: a zero GPTQ or AWQ `group_size` (a checkpoint-wide config value, refused by
/// field name - card 545b), a rank-0 `scales`, a `g_idx` shorter
/// than K (even with a contiguous-looking prefix) naming the tensor, and an FP8 linear naming both
/// scale spellings.
#[test]
fn malformed_quantized_linears_are_refused_by_tensor_name() {
    let gptq = |group_size: usize, scales_shape: Vec<usize>, g_idx_len: usize| {
        let store = store_of(vec![
            ("l.qweight", DType::I32, vec![1, 8], u32_words_le(&[0; 8])),
            ("l.qzeros", DType::I32, vec![1, 1], u32_words_le(&[0])),
            (
                "l.scales",
                DType::F16,
                scales_shape.clone(),
                vec![0; 2 * scales_shape.iter().product::<usize>()],
            ),
            (
                "l.g_idx",
                DType::I32,
                vec![g_idx_len],
                u32_words_le(&vec![0u32; g_idx_len]),
            ),
        ]);
        pack_quantized_linears(
            &store,
            QuantScheme {
                kind: QuantKind::Gptq,
                group_size,
            },
        )
    };
    let tensor_of = |result: Result<WeightStore, LoadError>| match result {
        Err(LoadError::QuantizedLinearLayout { tensor, .. }) => tensor,
        other => panic!("expected a typed layout refusal, got {other:?}"),
    };
    assert!(gptq(8, vec![1, 8], 8).is_ok());
    assert!(matches!(
        gptq(0, vec![1, 8], 8),
        Err(LoadError::InvalidQuantConfig {
            field: "group_size",
            value: 0
        })
    ));
    assert_eq!(tensor_of(gptq(8, vec![8], 8)), "l.scales");
    assert_eq!(tensor_of(gptq(8, vec![1, 8], 4)), "l.g_idx");

    // Card 545b: the shared `group_size` refusal is a separate `expect` site
    // per kind (`ValidatedScheme::Awq`), so the GPTQ zero row above does not cover it.
    let awq = |group_size: usize| {
        let store = store_of(vec![
            ("l.qweight", DType::I32, vec![8, 1], u32_words_le(&[0; 8])),
            ("l.qzeros", DType::I32, vec![1, 1], u32_words_le(&[0])),
            ("l.scales", DType::F16, vec![1, 8], vec![0; 2 * 8]),
        ]);
        pack_quantized_linears(
            &store,
            QuantScheme {
                kind: QuantKind::Awq,
                group_size,
            },
        )
    };
    assert!(matches!(
        awq(0),
        Err(LoadError::InvalidQuantConfig {
            field: "group_size",
            value: 0
        })
    ));

    let fp8 = store_of(vec![
        ("l.weight", DType::E4M3FN, vec![2, 4], vec![0x38; 8]),
        (
            "l.weight_scale",
            DType::F32,
            vec![2, 1],
            f32_words_le(&[1.0, 1.0]),
        ),
        (
            "l.weight_scale_inv",
            DType::F32,
            vec![1, 1],
            f32_words_le(&[1.0]),
        ),
    ]);
    assert!(matches!(
        pack_quantized_linears(
            &fp8,
            QuantScheme {
                kind: QuantKind::Fp8,
                group_size: 0,
            }
        ),
        Err(LoadError::Fp8AmbiguousScale { ref first, ref second, .. })
            if first == "l.weight_scale" && second == "l.weight_scale_inv"
    ));
}
