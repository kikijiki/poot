//! POOT-1100 / POOT-1416 (design `dmap.md` section 11, rows LP-1 and LP-2): the load pipeline's determinism and
//! its single content pass. Every row drives the production entry (`gguf_plan` + `build_store`, via
//! `read_gguf`) over an in-memory fixture. Model-free; no wall-clock assertion anywhere.
//!
//! Mutations (applied by hand, shown red, restored): LP-1 insert the jobs in reverse plan order; LP-2 make
//! `PackedPayload::try_new_deferred` run the content check at construction.

use poot_load::LoadError;
use poot_load::gguf::{GgufIndex, IdentityNames, read_gguf, write_gguf};
use poot_quant::format::{Storage, WeightFormat};
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_quant::{PackedWeightError, PayloadReadError, SourceRole, StoredBytes};

const GGML_F32: u32 = 0;
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;

/// `out` rows of `k` values of `format`, every byte `fill` (< 16 keeps every f16 word subnormal, so
/// every field is finite).
fn blocks(format: WeightFormat, out: usize, k: usize, fill: u8) -> Vec<u8> {
    let Storage::Blocks(layout) = format.descriptor().storage else {
        unreachable!("a GGUF format is a block format")
    };
    vec![fill; out * (k / layout.values) * layout.bytes]
}

fn load(bytes: Vec<u8>) -> Result<WeightStore, LoadError> {
    let file = StoredBytes::from(bytes);
    let index = GgufIndex::parse(&file).unwrap();
    read_gguf(&index, &IdentityNames)
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

/// LP-1: the insertion order is the plan's (tensor name, then expert), not the file's: the generations follow
/// the plan. Mutation: insert the jobs in reverse plan order; the order below reverses and this row goes red.
#[test]
fn lp1_insertion_order_is_the_plans() {
    let store = load(fixture()).unwrap();
    let expected = [
        "a.q4k",
        "b.experts.0",
        "b.experts.1",
        "b.experts.2",
        "m.norm",
        "z.q8",
    ];
    assert_eq!(insertion_order(&store), expected);
    for key in expected {
        assert!(!stored(&store, key).is_empty(), "{key}");
    }
}

/// LP-2 (ADR-0103 amendment of 2026-10-27, single-pass validation): the load builds the entry of a tensor
/// holding a non-finite scale without reading its content, and the first read of the bytes refuses it,
/// naming the first failing block of the range it read. The control is the same payload with the scale
/// finite, which reads clean. Mutation: make `try_new_deferred` run `validate_content`; the load itself
/// fails and `load(..)` below panics.
#[test]
fn lp2_content_is_checked_by_the_read_not_the_load() {
    let rows = 64;
    let mut big = blocks(WeightFormat::Q4_K, rows, 256, 0);
    let last_block = big.len() - 144;
    big[last_block..last_block + 2].copy_from_slice(&0x7e00u16.to_le_bytes()); // f16 NaN `d`
    let bytes = write_gguf(
        &[],
        &[("a.big", vec![256, rows as u64], GGML_Q4_K, big.clone())],
    );
    let store = load(bytes).expect("the load does not read content");
    let Some(WeightEntry::Packed(payload)) = store.get("a.big") else {
        panic!("a.big is packed");
    };
    let mut out = vec![0u8; big.len()];
    match payload.read_verified(SourceRole::Blocks, 0..big.len(), &mut out) {
        Err(PayloadReadError::Content(PackedWeightError::NonFiniteField {
            role, element, ..
        })) => {
            assert_eq!(role, SourceRole::Blocks);
            assert_eq!(element, (rows - 1) * 256, "the last block");
        }
        other => panic!("expected NonFiniteField, got {other:?}"),
    }
    // Control: a read that stops short of the bad block is clean.
    let clean = big.len() - 144;
    payload
        .read_verified(SourceRole::Blocks, 0..clean, &mut out[..clean])
        .expect("the blocks before the bad one are finite");
}

/// LP-3: the structure of a packed tensor is still refused by the load itself, as the GGUF container's own
/// packed-weight error: a Q8_0 tensor whose `K` (33) is not a whole number of 32-value blocks has no block
/// layout, so no read could ever deliver it. The control is the same tensor at `K = 64`.
///
/// Mutation: drop the `k.is_multiple_of(layout.values)` refusal in `PackedWeight::try_new`; the load admits
/// the tensor (or refuses it as another variant) and this row goes red.
#[test]
fn lp3_a_packed_tensor_without_a_block_layout_is_refused_by_the_load() {
    let tensor = |k: u64| {
        let bytes = blocks(WeightFormat::Q8_0, 2, 64, 3);
        write_gguf(&[], &[("w", vec![k, 2], GGML_Q8_0, bytes)])
    };
    load(tensor(64)).expect("control: K = 64 is two whole blocks");
    match load(tensor(33)) {
        Err(LoadError::GgufPackedWeight(_)) => {}
        other => panic!("expected GgufPackedWeight, got {:?}", other.map(|_| ())),
    }
}
