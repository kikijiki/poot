//! POOT-1099 (ADR-0103 amendment, design `dmap.md` section 10): checkpoint bytes are views of one read-only
//! mapping per file. Every row drives a production entry (`StoredBytes::map`, `read_gguf`,
//! `load_weight_store`) over a fixture written to a real temp file. Model-free.
//!
//! Mutations (applied by hand, shown red, restored; recorded in the POOT-1099 report):
//! MB-1 `slice` starts one byte late; MB-2 `slice` copies into a new owned region; MB-3 an extra `Arc`
//! clone kept in a static by `StoredBytes::map`; MB-4 drop the `is_file` check in `MappedFile::new`;
//! MB-5 `check_unchanged` compares `dev`/`ino` only; MB-6 `check_unchanged` stats the path instead of
//! the descriptor; MB-7 skip validation for a mapped source.

use std::fs::File;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;

use poot_load::LoadError;
use poot_load::gguf::{GgufIndex, IdentityNames, TensorNameMap, read_gguf, write_gguf};
use poot_load::plan::default_workers;
use poot_quant::format::{Storage, WeightFormat};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightKey, WeightStore};
use poot_quant::{MapError, PackedWeightError, SourceRole, StoredBytes};
use poot_tensor::DType;

const GGML_F32: u32 = 0;
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;

/// `out` rows of `k` values of `format`, every block zero (every field finite).
fn zero_blocks(format: WeightFormat, out: usize, k: usize) -> Vec<u8> {
    let Storage::Blocks(layout) = format.descriptor().storage else {
        unreachable!("a GGUF format is a block format")
    };
    vec![0u8; out * (k / layout.values) * layout.bytes]
}

/// A dense F32 norm, a Q8_0 and a Q4_K matrix, and a rank-3 Q4_K expert tensor of three experts, each
/// distinguishable by its fill byte (< 16 keeps every f16 word in a block subnormal, hence finite).
fn fixture_gguf() -> Vec<u8> {
    let mut q8 = zero_blocks(WeightFormat::Q8_0, 2, 64);
    q8.iter_mut()
        .enumerate()
        .for_each(|(i, b)| *b = (i % 7) as u8);
    let mut q4k = zero_blocks(WeightFormat::Q4_K, 2, 256);
    q4k.iter_mut()
        .enumerate()
        .for_each(|(i, b)| *b = (i % 5) as u8);
    let mut experts = zero_blocks(WeightFormat::Q4_K, 3 * 2, 256);
    let per_expert = experts.len() / 3;
    for (expert, chunk) in experts.chunks_mut(per_expert).enumerate() {
        chunk.fill(expert as u8 + 1);
    }
    let norm: Vec<u8> = [1.0f32, -2.5, 0.0, 3.25, 8.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    write_gguf(
        &[],
        &[
            ("norm.weight", vec![5], GGML_F32, norm),
            ("blk.0.attn_q.weight", vec![64, 2], GGML_Q8_0, q8),
            ("blk.0.attn_k.weight", vec![256, 2], GGML_Q4_K, q4k),
            ("blk.0.ffn_exps.weight", vec![256, 2, 3], GGML_Q4_K, experts),
        ],
    )
}

fn map_path(path: &Path) -> StoredBytes {
    StoredBytes::map(File::open(path).expect("open fixture")).expect("map fixture")
}

/// Every stored byte run of `store`, by key: a dense tensor's bytes, or a packed owner's blocks.
fn entry_bytes<'s>(store: &'s WeightStore, key: &str) -> &'s [u8] {
    match store.get(key).unwrap_or_else(|| panic!("missing {key}")) {
        WeightEntry::Dense(dense) => dense.bytes().as_slice(),
        WeightEntry::Packed(payload) => payload.bytes(SourceRole::Blocks),
    }
}

const GGUF_KEYS: [&str; 6] = [
    "norm.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.ffn_exps.weight.0",
    "blk.0.ffn_exps.weight.1",
    "blk.0.ffn_exps.weight.2",
];

/// `(absolute offset in the file, length)` of every entry of [`GGUF_KEYS`], read from the index. The
/// fixture writer puts tensor data last with no trailing pad, so the data start is `file_len - data_len`.
fn gguf_extents(
    index: &GgufIndex,
    bytes: &[u8],
    store: &WeightStore,
) -> Vec<(&'static str, u64, usize)> {
    let end_of_data = index
        .tensors
        .iter()
        .map(|(name, info)| info.offset + entry_len(store, name) as u64)
        .max()
        .unwrap();
    let data_start = bytes.len() as u64 - end_of_data;
    let mut extents = Vec::new();
    for key in GGUF_KEYS {
        let (tensor, expert) = match key.strip_suffix(|c: char| c.is_ascii_digit()) {
            Some(rest) if rest.ends_with('.') => (
                &rest[..rest.len() - 1],
                key[rest.len()..].parse::<u64>().unwrap(),
            ),
            _ => (key, 0),
        };
        let len = entry_bytes(store, key).len();
        let offset = index.tensors[tensor].offset + expert * len as u64;
        extents.push((key, data_start + offset, len));
    }
    extents
}

fn entry_len(store: &WeightStore, tensor: &str) -> usize {
    match store.get(tensor) {
        Some(_) => entry_bytes(store, tensor).len(),
        None => (0..3)
            .map(|e| entry_bytes(store, &format!("{tensor}.{e}")).len())
            .sum(),
    }
}

/// MB-1 (gguf): each entry's bytes equal a `read_exact_at` of the same file range.
#[test]
fn mb1_mapped_gguf_equals_a_positional_read() {
    let bytes = fixture_gguf();
    let path = poot_test_util::unique_temp_path("mb1.gguf");
    std::fs::write(&*path, &bytes).unwrap();
    let file = map_path(&path);
    let index = GgufIndex::parse(&file).unwrap();
    let store = read_gguf(&index, &IdentityNames, default_workers()).unwrap();
    let reader = File::open(&*path).unwrap();
    for (key, offset, len) in gguf_extents(&index, &bytes, &store) {
        let mut expected = vec![0u8; len];
        reader.read_exact_at(&mut expected, offset).unwrap();
        assert_eq!(entry_bytes(&store, key), expected.as_slice(), "{key}");
    }
}

/// MB-1 (safetensors): a shard written to a temp file loads with each tensor equal to a positional read.
#[test]
fn mb1_mapped_safetensors_equals_a_positional_read() {
    let mut fixture = WeightStore::builder();
    for (name, dtype, shape, fill) in [
        ("a.weight", DType::F32, vec![3, 2], 1u8),
        ("b.weight", DType::BF16, vec![5], 2),
        ("c.bias", DType::F16, vec![7], 3),
    ] {
        let len = shape.iter().product::<usize>() * dtype.byte_size();
        let dense = DenseWeight::try_new(dtype, shape, StoredBytes::from(vec![fill; len])).unwrap();
        fixture.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    let dir = poot_test_util::unique_temp_path("mb1_shard");
    std::fs::create_dir_all(&*dir).unwrap();
    let shard = dir.join("model.safetensors");
    poot_quant::test_support::safetensors::write(&shard, &fixture.build());

    let store = poot_load::safetensors::load_weight_store(&*dir).unwrap();
    let raw = std::fs::read(&shard).unwrap();
    let header_end = 8 + u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&raw[8..header_end]).unwrap();
    let reader = File::open(&shard).unwrap();
    for name in ["a.weight", "b.weight", "c.bias"] {
        let offsets = header[name]["data_offsets"].as_array().unwrap();
        let (start, end) = (offsets[0].as_u64().unwrap(), offsets[1].as_u64().unwrap());
        let mut expected = vec![0u8; (end - start) as usize];
        reader
            .read_exact_at(&mut expected, header_end as u64 + start)
            .unwrap();
        assert_eq!(entry_bytes(&store, name), expected.as_slice(), "{name}");
    }
}

/// MB-2: every entry of one mapped file points into one region at its file offset.
#[test]
fn mb2_entries_are_views_of_one_region() {
    let bytes = fixture_gguf();
    let path = poot_test_util::unique_temp_path("mb2.gguf");
    std::fs::write(&*path, &bytes).unwrap();
    let file = map_path(&path);
    let index = GgufIndex::parse(&file).unwrap();
    let store = read_gguf(&index, &IdentityNames, default_workers()).unwrap();
    let extents = gguf_extents(&index, &bytes, &store);
    let (first_key, first_offset, _) = extents[0];
    let first_ptr = entry_bytes(&store, first_key).as_ptr() as isize;
    for &(key, offset, _) in &extents[1..] {
        let delta = entry_bytes(&store, key).as_ptr() as isize - first_ptr;
        assert_eq!(delta, offset as isize - first_offset as isize, "{key}");
    }
    // The mapped file itself is the same region as a view cut from it.
    let view = file.slice(0..4).unwrap();
    assert!(view.same_region(&file));
    assert_eq!(view.as_slice().as_ptr(), file.as_slice().as_ptr());
}

/// The path appears in `/proc/self/maps` while any view of its mapping lives.
fn is_mapped(path: &Path) -> bool {
    std::fs::read_to_string("/proc/self/maps")
        .unwrap()
        .contains(path.to_str().unwrap())
}

/// MB-3: the mapping outlives the store while a clone of an entry lives (the index holds one too, until
/// it drops), and is gone after the last drop.
#[test]
fn mb3_the_last_owner_unmaps() {
    let path = poot_test_util::unique_temp_path("mb3.gguf");
    std::fs::write(&*path, fixture_gguf()).unwrap();
    assert!(!is_mapped(&path));
    let file = map_path(&path);
    assert!(is_mapped(&path));
    let index = GgufIndex::parse(&file).unwrap();
    let store = read_gguf(&index, &IdentityNames, default_workers()).unwrap();
    let Some(WeightEntry::Dense(norm)) = store.get("norm.weight") else {
        panic!("norm.weight is dense")
    };
    let held = norm.bytes().clone();
    drop(file);
    drop(index);
    drop(store);
    assert!(
        is_mapped(&path),
        "a clone held after the store dropped keeps the mapping"
    );
    assert_eq!(held.len(), 20);
    drop(held);
    assert!(!is_mapped(&path), "the last drop unmaps the file");
}

/// MB-4: unmappable sources are typed; an empty file is an empty owned region; a fixture from memory
/// loads through the same `read_gguf` with equal entries.
#[test]
fn mb4_unmappable_sources_are_typed() {
    let dir = poot_test_util::unique_temp_path("mb4");
    std::fs::create_dir_all(&*dir).unwrap();

    let directory = File::open(&*dir).unwrap();
    assert!(matches!(
        StoredBytes::map(directory),
        Err(MapError::NotRegularFile { .. })
    ));

    let fifo = dir.join("pipe");
    let made = std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap();
    assert!(made.success());
    // O_NONBLOCK: opening a FIFO for reading otherwise waits for a writer.
    let fifo = File::options()
        .read(true)
        .custom_flags(0o4000)
        .open(&fifo)
        .unwrap();
    assert!(matches!(
        StoredBytes::map(fifo),
        Err(MapError::NotRegularFile { .. })
    ));

    let empty = dir.join("empty");
    std::fs::write(&empty, []).unwrap();
    let mapped = map_path(&empty);
    assert!(mapped.is_empty());
    assert!(mapped.check_unchanged().is_ok());
    assert!(!is_mapped(&empty), "an empty file is not mapped");

    let bytes = fixture_gguf();
    let path = dir.join("fixture.gguf");
    std::fs::write(&path, &bytes).unwrap();
    let from_file = map_path(&path);
    let from_memory = StoredBytes::from(bytes);
    let load = |source: &StoredBytes| {
        let index = GgufIndex::parse(source).unwrap();
        read_gguf(&index, &IdentityNames, default_workers()).unwrap()
    };
    assert_eq!(load(&from_file), load(&from_memory));
}

/// A name map that keeps no tensor: `read_gguf` then touches no tensor page, so a truncated file is
/// reported by the end-of-load check without a `SIGBUS`.
struct KeepNothing;

impl TensorNameMap for KeepNothing {
    fn weight_key(&self, _gguf_name: &str) -> Option<WeightKey> {
        None
    }
}

/// MB-5: a file truncated after mapping is refused as `SourceChanged` at every boundary, and the test
/// process survives (no page of the truncated file is touched before a check).
#[test]
fn mb5_a_truncated_file_is_refused_typed() {
    let path = poot_test_util::unique_temp_path("mb5.gguf");
    std::fs::write(&*path, fixture_gguf()).unwrap();
    let file = map_path(&path);
    let index = GgufIndex::parse(&file).unwrap();
    let store = read_gguf(&index, &IdentityNames, default_workers()).unwrap();
    assert!(store.check_sources().is_ok());

    File::options()
        .write(true)
        .open(&*path)
        .unwrap()
        .set_len(0)
        .unwrap();

    // Boundary 1, end of load: the reader's own check, reached with no tensor read.
    assert!(matches!(
        read_gguf(&index, &KeepNothing, default_workers()),
        Err(LoadError::SourceChanged(_))
    ));
    // Boundary 2's check, on the store loaded before the truncation.
    let changed = store.check_sources().unwrap_err();
    assert_eq!(changed.was.len, file.len() as u64);
    assert_eq!(changed.now.map(|now| now.len), Some(0));
}

/// MB-6: replacing the file by rename leaves the mapped inode alone: entries keep the old bytes and the
/// identity check (which stats the descriptor, not the path) passes.
#[test]
fn mb6_rename_replacement_is_safe() {
    let path = poot_test_util::unique_temp_path("mb6.gguf");
    let original = fixture_gguf();
    std::fs::write(&*path, &original).unwrap();
    let file = map_path(&path);
    let index = GgufIndex::parse(&file).unwrap();
    let store = read_gguf(&index, &IdentityNames, default_workers()).unwrap();
    let before: Vec<Vec<u8>> = GGUF_KEYS
        .iter()
        .map(|k| entry_bytes(&store, k).to_vec())
        .collect();

    let replacement = path.with_extension("new");
    std::fs::write(&replacement, vec![0xAAu8; original.len() + 512]).unwrap();
    std::fs::rename(&replacement, &*path).unwrap();

    for (key, old) in GGUF_KEYS.iter().zip(&before) {
        assert_eq!(entry_bytes(&store, key), old.as_slice(), "{key}");
    }
    assert!(file.check_unchanged().is_ok());
    assert!(store.check_sources().is_ok());
}

/// MB-7: validation still runs over mapped bytes: a NaN `d` scale in the last Q4_K block of a mapped
/// file is refused, naming the role and element.
#[test]
fn mb7_validation_runs_over_mapped_bytes() {
    let mut q4k = zero_blocks(WeightFormat::Q4_K, 2, 256);
    let block = q4k.len() - 144;
    q4k[block..block + 2].copy_from_slice(&0x7e00u16.to_le_bytes()); // f16 NaN `d`
    let bytes = write_gguf(
        &[],
        &[("blk.0.attn_k.weight", vec![256, 2], GGML_Q4_K, q4k)],
    );
    let path = poot_test_util::unique_temp_path("mb7.gguf");
    std::fs::write(&*path, bytes).unwrap();
    let file = map_path(&path);
    let index = GgufIndex::parse(&file).unwrap();
    let error = read_gguf(&index, &IdentityNames, default_workers()).unwrap_err();
    match error {
        LoadError::GgufPackedWeight(PackedWeightError::NonFiniteField {
            role, element, ..
        }) => {
            assert_eq!(role, SourceRole::Blocks);
            assert_eq!(element, 256, "the second block's `d`");
        }
        other => panic!("expected NonFiniteField, got {other:?}"),
    }
}

/// MB-5 for the plain safetensors loader: a shard truncated after mapping is refused at the end of load.
/// The fixture is smaller than a page, so truncating it by one byte leaves the mapped page backed (no
/// `SIGBUS`) and the loader reads the whole shard before the check that refuses it.
/// Mutation: ignore the `check_unchanged` results in `load_mapped_weight_store`; the load returns `Ok`.
#[test]
fn mb5_a_truncated_safetensors_shard_is_refused_at_the_end_of_load() {
    let mut fixture = WeightStore::builder();
    let dense =
        DenseWeight::try_new(DType::F32, vec![4], StoredBytes::from(vec![1u8; 16])).unwrap();
    fixture
        .insert("a.weight", WeightEntry::Dense(dense))
        .unwrap();
    let dir = poot_test_util::unique_temp_path("mb5_shard");
    std::fs::create_dir_all(&*dir).unwrap();
    let shard = dir.join("model.safetensors");
    poot_quant::test_support::safetensors::write(&shard, &fixture.build());
    let files = poot_load::safetensors::map_weight_files(&*dir, |name| {
        std::fs::read(dir.join(name)).map_err(poot_load::LoadError::from)
    })
    .unwrap();
    assert!(files[0].1.len() < 4096);

    File::options()
        .write(true)
        .open(&shard)
        .unwrap()
        .set_len(files[0].1.len() as u64 - 1)
        .unwrap();
    assert!(matches!(
        poot_load::safetensors::load_mapped_weight_store(&files, default_workers()),
        Err(LoadError::SourceChanged(_))
    ));
}
