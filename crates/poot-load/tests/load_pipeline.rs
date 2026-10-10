//! POOT-1100 (design `dmap.md` section 11, rows LP-1 and LP-2): the load pipeline's determinism. Every
//! row drives the production entry (`gguf_plan` + `build_store`, via `read_gguf`) over an in-memory
//! fixture. Model-free; no wall-clock assertion anywhere.
//!
//! Mutations (applied by hand, shown red, restored; recorded in the POOT-1100 report):
//! LP-1 insert results into the builder in completion order; LP-2 return the first error that completes.

use std::num::NonZeroUsize;

use poot_load::LoadError;
use poot_load::gguf::{GgufIndex, IdentityNames, read_gguf, write_gguf};
use poot_quant::format::{Storage, WeightFormat};
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_quant::{PackedWeightError, SourceRole, StoredBytes};

const GGML_F32: u32 = 0;
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;

fn workers(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

/// `out` rows of `k` values of `format`, every byte `fill` (< 16 keeps every f16 word subnormal, so
/// every field is finite).
fn blocks(format: WeightFormat, out: usize, k: usize, fill: u8) -> Vec<u8> {
    let Storage::Blocks(layout) = format.descriptor().storage else {
        unreachable!("a GGUF format is a block format")
    };
    vec![fill; out * (k / layout.values) * layout.bytes]
}

fn load(bytes: Vec<u8>, workers: NonZeroUsize) -> Result<WeightStore, LoadError> {
    let file = StoredBytes::from(bytes);
    let index = GgufIndex::parse(&file).unwrap();
    read_gguf(&index, &file, &IdentityNames, workers)
}

fn stored(store: &WeightStore, key: &str) -> Vec<u8> {
    match store.get(key).unwrap_or_else(|| panic!("missing {key}")) {
        WeightEntry::Dense(dense) => dense.bytes().as_slice().to_vec(),
        WeightEntry::Packed(payload) => payload.bytes(SourceRole::Blocks).to_vec(),
    }
}

/// The store's keys, ordered by the generation each was inserted under: the insertion order.
fn insertion_order(store: &WeightStore) -> Vec<String> {
    let mut keys: Vec<_> = store
        .iter()
        .map(|(key, _)| (store.generation(&key.to_string()).unwrap(), key.to_string()))
        .collect();
    keys.sort();
    keys.into_iter().map(|(_, key)| key).collect()
}

/// Tensors written in an order that is neither the name order nor the size order: dense, Q8_0, Q4_K and
/// a rank-3 expert tensor.
fn fixture() -> Vec<u8> {
    let norm: Vec<u8> = [1.0f32, -2.5, 0.0, 3.25]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    write_gguf(
        &[],
        &[
            ("m.norm", vec![4], GGML_F32, norm),
            (
                "z.q8",
                vec![64, 4],
                GGML_Q8_0,
                blocks(WeightFormat::Q8_0, 4, 64, 3),
            ),
            (
                "b.experts",
                vec![256, 2, 3],
                GGML_Q4_K,
                blocks(WeightFormat::Q4_K, 6, 256, 5),
            ),
            (
                "a.q4k",
                vec![256, 8],
                GGML_Q4_K,
                blocks(WeightFormat::Q4_K, 8, 256, 7),
            ),
        ],
    )
}

/// LP-1: the store does not depend on the worker count: the same entries, bytes and insertion order
/// (generations) with 1 and with 4 workers, and the insertion order is the plan's (tensor name, then
/// expert), not the order in which workers finished.
#[test]
fn lp1_store_is_independent_of_worker_count() {
    let bytes = fixture();
    let one = load(bytes.clone(), workers(1)).unwrap();
    let four = load(bytes, workers(4)).unwrap();
    let expected = [
        "a.q4k",
        "b.experts.0",
        "b.experts.1",
        "b.experts.2",
        "m.norm",
        "z.q8",
    ];
    assert_eq!(insertion_order(&one), expected);
    assert_eq!(insertion_order(&four), expected);
    for key in expected {
        assert_eq!(stored(&one, key), stored(&four, key), "{key}");
    }
}

/// LP-1b: the insertion order is the plan's whichever tensor finishes first. `z.tiny` is first in the file, so
/// a worker takes it first and it finishes while `a.big` (first in the plan, last in the file, 36 MB to scan)
/// is still being validated; with two workers the finish order is therefore `z.tiny`, `a.big` every time, the
/// reverse of the plan. The generations must still follow the plan.
/// Mutation: record each job's index as it completes and insert the results in that order (a true
/// completion-order build); `z.tiny` is inserted before `a.big` and this row goes red.
#[test]
fn lp1_insertion_order_is_the_plans_whichever_tensor_finishes_first() {
    let rows = 16384;
    let big = blocks(WeightFormat::Q4_K, rows, 4096, 7);
    let tiny = blocks(WeightFormat::Q4_K, 1, 256, 7);
    let bytes = write_gguf(
        &[],
        &[
            ("z.tiny", vec![256, 1], GGML_Q4_K, tiny),
            ("a.big", vec![4096, rows as u64], GGML_Q4_K, big),
        ],
    );
    let store = load(bytes, workers(2)).unwrap();
    assert_eq!(insertion_order(&store), ["a.big", "z.tiny"]);
}

/// LP-2: a failure reports the lowest failing job (plan order), whichever finished first. `a.big` (first
/// in the plan, last in the file) and `z.tiny` (last in the plan, first in the file, so a worker takes
/// it first) both hold a non-finite scale; the error names `a.big`'s block.
#[test]
fn lp2_the_lowest_failing_tensor_is_reported() {
    let rows = 64;
    let mut big = blocks(WeightFormat::Q4_K, rows, 256, 0);
    let last_block = big.len() - 144;
    big[last_block..last_block + 2].copy_from_slice(&0x7e00u16.to_le_bytes()); // f16 NaN `d`
    let mut tiny = blocks(WeightFormat::Q4_K, 1, 256, 0);
    tiny[0..2].copy_from_slice(&0x7e00u16.to_le_bytes());
    let bytes = write_gguf(
        &[],
        &[
            ("z.tiny", vec![256, 1], GGML_Q4_K, tiny),
            ("a.big", vec![256, rows as u64], GGML_Q4_K, big),
        ],
    );
    for n in [1, 4] {
        match load(bytes.clone(), workers(n)).unwrap_err() {
            LoadError::GgufPackedWeight(PackedWeightError::NonFiniteField {
                role,
                element,
                ..
            }) => {
                assert_eq!(role, SourceRole::Blocks);
                assert_eq!(
                    element,
                    (rows - 1) * 256,
                    "a.big's last block, with {n} workers"
                );
            }
            other => panic!("expected NonFiniteField, got {other:?}"),
        }
    }
}
