//! A split GGUF checkpoint (llama.cpp's `-0000N-of-0000M.gguf` sets) is one index over its shards:
//! every row drives the production entries (`GgufIndex::parse_shards`, `read_gguf`, `shard_paths`)
//! over shards written to memory. Model-free.
//!
//! Mutations (applied by hand, shown red, restored; recorded in the POOT-1189 report):
//! SH-1 `gguf_plan` slices every tensor out of shard 0; SH-2 `parse_shards` skips the `split.count`
//! check; SH-3 skips the `split.no` check; SH-4 lets a later shard replace a tensor of an earlier
//! one; SH-5 `gguf_plan` skips the `split.tensors.count` check; SH-6 `split_name` accepts a shard
//! number above the count.

use std::path::Path;

use poot_load::gguf::{GgufIndex, GgufValue, IdentityNames, read_gguf, shard_paths, write_gguf};
use poot_load::plan::default_workers;
use poot_load::{GgufSplitError, LoadError};
use poot_quant::StoredBytes;
use poot_quant::weights::{WeightEntry, WeightStore};

const GGML_F32: u32 = 0;

/// One shard of a set of `count` shards that together hold `total` tensors: `tensors` are F32
/// vectors of four values, each tensor's values distinct from every other's.
fn shard(no: u32, count: u32, total: u32, tensors: &[(&str, f32)]) -> StoredBytes {
    let mut kvs = vec![
        ("split.no", GgufValue::U32(no)),
        ("split.count", GgufValue::U32(count)),
        ("split.tensors.count", GgufValue::U32(total)),
    ];
    if no == 0 {
        kvs.push(("general.architecture", GgufValue::Str("test".into())));
    }
    let tensors: Vec<_> = tensors
        .iter()
        .map(|&(name, base)| {
            let values: Vec<u8> = (0..4)
                .flat_map(|i| (base + i as f32).to_le_bytes())
                .collect();
            (name, vec![4], GGML_F32, values)
        })
        .collect();
    StoredBytes::from(write_gguf(&kvs, &tensors))
}

fn three() -> [StoredBytes; 3] {
    [
        shard(0, 3, 3, &[("a", 1.0)]),
        shard(1, 3, 3, &[("b", 10.0)]),
        shard(2, 3, 3, &[("c", 100.0)]),
    ]
}

fn values(store: &WeightStore, key: &str) -> Vec<f32> {
    match store.get(key).unwrap_or_else(|| panic!("missing {key}")) {
        WeightEntry::Dense(dense) => dense
            .bytes()
            .as_slice()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        other => panic!("{key}: {other:?}"),
    }
}

/// SH-1: a tensor of the second and third shard is read from its own shard's bytes, and the
/// metadata is the first shard's.
#[test]
fn sh1_each_tensor_reads_its_own_shards_bytes() {
    let files = three();
    let index = GgufIndex::parse_shards(&files).unwrap();
    assert_eq!(index.architecture(), Some("test"));
    let shards: Vec<_> = ["a", "b", "c"].map(|n| index.tensors[n].shard).into();
    assert_eq!(shards, [0, 1, 2]);
    let store = read_gguf(&index, &IdentityNames, default_workers()).unwrap();
    assert_eq!(values(&store, "a"), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(values(&store, "b"), [10.0, 11.0, 12.0, 13.0]);
    assert_eq!(values(&store, "c"), [100.0, 101.0, 102.0, 103.0]);
}

fn split_error(result: Result<GgufIndex, LoadError>) -> GgufSplitError {
    match result {
        Err(LoadError::GgufSplit(error)) => error,
        Err(other) => panic!("expected a split error, got {other}"),
        Ok(_) => panic!("expected a split error, got an index"),
    }
}

/// SH-2: fewer shards than the first declares are refused with the declared and given counts.
#[test]
fn sh2_a_missing_shard_is_refused() {
    let [a, b, _] = three();
    assert_eq!(
        split_error(GgufIndex::parse_shards(&[a, b])),
        GgufSplitError::Count {
            declared: Some(3),
            given: 2
        }
    );
}

/// SH-3: shards given out of order are refused with the position and the `split.no` found there.
#[test]
fn sh3_shards_out_of_order_are_refused() {
    let [a, b, c] = three();
    assert_eq!(
        split_error(GgufIndex::parse_shards(&[a, c, b])),
        GgufSplitError::Order {
            position: 1,
            declared: Some(2)
        }
    );
}

/// SH-4: a tensor listed by two shards is refused, naming both.
#[test]
fn sh4_a_tensor_in_two_shards_is_refused() {
    let files = [
        shard(0, 2, 2, &[("a", 1.0)]),
        shard(1, 2, 2, &[("a", 10.0)]),
    ];
    assert_eq!(
        split_error(GgufIndex::parse_shards(&files)),
        GgufSplitError::DuplicateTensor {
            tensor: "a".into(),
            shards: (0, 1)
        }
    );
}

/// SH-5: the first shard alone parses (its metadata is readable) but is not a loadable checkpoint:
/// its directory lists one tensor of the three the set declares.
#[test]
fn sh5_a_lone_shard_is_not_loadable() {
    let [a, ..] = three();
    let index = GgufIndex::parse(&a).unwrap();
    let error = read_gguf(&index, &IdentityNames, default_workers()).unwrap_err();
    assert!(
        matches!(
            error,
            LoadError::GgufSplit(GgufSplitError::Incomplete {
                declared: 3,
                found: 1
            })
        ),
        "{error}"
    );
}

/// SH-6: naming any shard names the set, in shard order; a file that is not a split name, or whose
/// number is outside `1..=count`, names itself.
#[test]
fn sh6_a_shard_name_names_the_set() {
    let set = |name: &str| -> Vec<String> {
        shard_paths(Path::new("/m").join(name).as_path())
            .iter()
            .map(|p| p.to_str().unwrap().to_string())
            .collect()
    };
    let expected = [
        "/m/Q-00001-of-00003.gguf",
        "/m/Q-00002-of-00003.gguf",
        "/m/Q-00003-of-00003.gguf",
    ];
    assert_eq!(set("Q-00001-of-00003.gguf"), expected);
    assert_eq!(set("Q-00003-of-00003.gguf"), expected);
    for alone in [
        "model.gguf",
        "Q-00004-of-00003.gguf",
        "Q-00000-of-00003.gguf",
        "Q-1-of-3.gguf",
        "Q-00001-of-00003.bin",
    ] {
        assert_eq!(set(alone), [format!("/m/{alone}")], "{alone}");
    }
}
