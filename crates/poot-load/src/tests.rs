use super::*;
use poot_tensor::DType;
use std::io::Read;

use poot_quant::scalar::{e4m3fn_to_f32, e8m0_to_f32, f16_to_f32};
use poot_quant::weights::{WeightEntry, WeightStore};
use safetensors::{load_weight_store_bytes, pack_quantized_linears};

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
    // I64 index tables land as ordinary dense entries through both entry points.
    for store in [&from_file, &from_bytes] {
        assert_eq!(dense_of(store, "position_ids").dtype(), DType::I64);
        assert_eq!(dense_of(store, "position_ids").bytes().len(), 8);
    }
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
    for pair in payload.as_chunks_mut::<2>().0.iter_mut() {
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

/// POOT-1059: the Qwen4Exp n-gram PLE table is stored as numbered E4M3 shards
/// (`<table>.shard_N.weight`) under ONE `<table>.weight_scale`; they carry no per-shard scale, yet
/// load as dense E4M3 entries (the model applies the shared scale) instead of tripping the
/// fail-closed rule. Mutation: removing the `is_ple_table_shard` exemption turns this red with the
/// shard named in the layout refusal.
#[test]
fn ple_table_shards_under_a_shared_scale_load_as_dense_e4m3() {
    let fp8 = QuantScheme {
        kind: QuantKind::Fp8,
        group_size: 0,
    };
    let table = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
    let shard = |n: u32| format!("{table}.shard_{n}.weight");
    let store = store_of(vec![
        (&shard(0), DType::E4M3FN, vec![2, 4], vec![0x38; 8]),
        (&shard(1), DType::E4M3FN, vec![2, 4], vec![0x38; 8]),
        (
            &format!("{table}.weight_scale"),
            DType::F32,
            vec![],
            1.0f32.to_le_bytes().to_vec(),
        ),
    ]);
    let packed = pack_quantized_linears(&store, fp8)
        .unwrap_or_else(|error| panic!("PLE shards refused: {error:?}"));
    for n in [0, 1] {
        assert!(matches!(
            packed.get(&shard(n)),
            Some(WeightEntry::Dense(dense)) if dense.dtype() == DType::E4M3FN
        ));
    }
}

/// POOT-1059: the exemption is exactly the named PLE shard shape. Each near miss stays refused
/// naming its tensor: a shard with no shared table scale, an ordinary `*.shard_N.weight` that is not
/// an `ngram_embedding` table, and a non-numeric ordinal. Mutation: widening the predicate (dropping the
/// `ngram_embedding`, scale or digit check) turns the matching row red.
#[test]
fn e4m3_weights_that_are_not_ple_table_shards_stay_refused() {
    let fp8 = QuantScheme {
        kind: QuantKind::Fp8,
        group_size: 0,
    };
    let scale = |table: &str| {
        (
            format!("{table}.weight_scale"),
            DType::F32,
            vec![],
            1.0f32.to_le_bytes().to_vec(),
        )
    };
    let weight = |name: &str| (name.to_string(), DType::E4M3FN, vec![2, 4], vec![0x38; 8]);
    let cases = [
        (
            "no shared scale",
            vec![weight("l.ple.ngram_embedding.shard_0.weight")],
            "l.ple.ngram_embedding.shard_0.weight",
        ),
        (
            "not an ngram_embedding table",
            vec![weight("l.table.shard_0.weight"), scale("l.table")],
            "l.table.shard_0.weight",
        ),
        (
            "non-numeric ordinal",
            vec![
                weight("l.ngram_embedding.shard_x.weight"),
                scale("l.ngram_embedding"),
            ],
            "l.ngram_embedding.shard_x.weight",
        ),
    ];
    for (what, tensors, refused) in cases {
        let store = store_of(
            tensors
                .iter()
                .map(|(name, dtype, shape, bytes)| {
                    (name.as_str(), *dtype, shape.clone(), bytes.clone())
                })
                .collect(),
        );
        match pack_quantized_linears(&store, fp8) {
            Err(LoadError::QuantizedLinearLayout { tensor, .. }) => {
                assert_eq!(tensor, refused, "{what}")
            }
            other => panic!("{what}: expected a typed layout refusal, got {other:?}"),
        }
    }
}

/// An HF `quant_method: "fp8"` block-scaled checkpoint (the `Qwen/Qwen3.8-27B-FP8` block) is the FP8
/// scheme, and a block other than the 128x128 one `E4m3Block128` describes is refused by name. Mutation:
/// drop the `"fp8"` arm of `QuantConfig::scheme`; the supported block is refused as an unknown
/// `quant_method` and the row goes red.
#[test]
fn an_hf_block_fp8_quantization_config_is_the_fp8_scheme() {
    let config = |block: serde_json::Value| {
        serde_json::json!({
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": block,
            }
        })
    };
    let scheme = crate::config::quant_scheme(&config(serde_json::json!([128, 128])))
        .unwrap()
        .expect("a quantization_config names a scheme");
    assert_eq!(scheme.kind, QuantKind::Fp8);
    match crate::config::quant_scheme(&config(serde_json::json!([64, 64]))) {
        Err(LoadError::SafeTensors(message)) => assert!(
            message.contains("weight_block_size"),
            "the refusal names the field: {message}"
        ),
        other => panic!("expected a typed refusal of a 64x64 block, got {other:?}"),
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
