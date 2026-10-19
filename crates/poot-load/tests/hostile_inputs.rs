//! POOT-1221 (R2-C5-001): the hostile-input layer of the loader. A checkpoint is untrusted bytes, so
//! the GGUF and safetensors readers meet three kinds of rows here, all through the production entries
//! (`GgufIndex::parse`, `read_gguf`, `load_mapped_weight_store`) over in-memory files. Model-free.
//!
//! 1. Corpora: a valid file is truncated at every byte, and corrupted at every header byte and at
//!    seeded random positions. No input may panic, and a file shorter than the valid one never
//!    loads.
//! 2. Named defects: one header per defect class (a count, length or offset at its extreme, a rank
//!    or element count over the cap, a range one byte past the file), each refused with the typed
//!    error that names it.
//! 3. A valid file still loads, so a refusal row is not green because everything is refused.
//!
//! Mutations (applied by hand, shown red, restored; recorded in the POOT-1221 report):
//! HI-1 `Cursor::take` stops checking `end > self.b.len()`; HI-2 `parse_header` drops the
//! `MAX_TENSOR_DIMS` check; HI-3 `tensor_range` compares `end >= file.len()`; HI-4
//! `checked_header_end` accepts a header ending past the file.

use std::panic::{AssertUnwindSafe, catch_unwind};

use poot_load::gguf::{GgufIndex, GgufValue, IdentityNames, read_gguf, write_gguf};
use poot_load::plan::default_workers;
use poot_load::safetensors::load_mapped_weight_store;
use poot_load::{GgufSplitError, LoadError};
use poot_quant::StoredBytes;

const GGML_F32: u32 = 0;
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;

/// A deterministic xorshift stream (the repository's tests carry no rng dependency).
fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed
        .wrapping_mul(0x100000001B3)
        .wrapping_add(0x9E3779B97F4A7C15)
        | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

// ---------------------------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------------------------

/// A valid checkpoint: a dense F32 vector, a Q8_0 matrix, a rank-3 Q4_K expert tensor, and
/// metadata with a string and a short array, so every parser branch has a byte to corrupt.
fn valid_gguf() -> Vec<u8> {
    write_gguf(
        &[
            ("general.architecture", GgufValue::Str("test".into())),
            ("test.layers", GgufValue::U32(2)),
            (
                "test.kinds",
                GgufValue::Array(vec![GgufValue::U32(1), GgufValue::U32(0)]),
            ),
        ],
        &[
            (
                "norm.weight",
                vec![4],
                GGML_F32,
                vec![0u8; 4 * std::mem::size_of::<f32>()],
            ),
            ("blk.0.attn_q.weight", vec![32, 2], GGML_Q8_0, vec![0u8; 68]),
            (
                "blk.0.ffn_up.weight",
                vec![256, 1, 2],
                GGML_Q4_K,
                vec![0u8; 2 * 144],
            ),
        ],
    )
}

/// Bytes of tensor data in [`valid_gguf`]: 16 (padded to 32), 68 (padded to 96) and 288.
const VALID_GGUF_DATA_LEN: usize = 96 + 288;

/// `parse` then `read_gguf` over `bytes`, the loader's two production steps.
fn load_gguf(bytes: Vec<u8>) -> Result<(), LoadError> {
    let index = GgufIndex::parse(&StoredBytes::from(bytes))?;
    read_gguf(&index, &IdentityNames, default_workers()).map(drop)
}

/// A GGUF header written field by field, so a row can put any value in any field. `tensors` and
/// `kvs` are the declared counts (not the number written).
struct RawGguf {
    version: u32,
    tensors: u64,
    kvs: u64,
    body: Vec<u8>,
}

impl RawGguf {
    fn new(tensors: u64, kvs: u64) -> Self {
        Self {
            version: 3,
            tensors,
            kvs,
            body: Vec::new(),
        }
    }

    fn string(mut self, s: &str) -> Self {
        self.body.extend((s.len() as u64).to_le_bytes());
        self.body.extend(s.as_bytes());
        self
    }

    fn raw(mut self, bytes: &[u8]) -> Self {
        self.body.extend(bytes);
        self
    }

    fn u32(self, v: u32) -> Self {
        self.raw(&v.to_le_bytes())
    }

    fn u64(self, v: u64) -> Self {
        self.raw(&v.to_le_bytes())
    }

    /// One tensor directory entry.
    fn tensor(self, name: &str, dims: &[u64], ggml_type: u32, offset: u64) -> Self {
        let mut me = self.string(name).u32(dims.len() as u32);
        for &d in dims {
            me = me.u64(d);
        }
        me.u32(ggml_type).u64(offset)
    }

    /// The file: header, body, zero padding to the 32-byte data start, then `data`.
    fn finish(self, data: &[u8]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(self.version.to_le_bytes());
        out.extend(self.tensors.to_le_bytes());
        out.extend(self.kvs.to_le_bytes());
        out.extend(self.body);
        out.resize(out.len().next_multiple_of(32), 0);
        out.extend(data);
        out
    }
}

fn refusal(result: Result<(), LoadError>, what: &str) -> LoadError {
    result.expect_err(what)
}

fn assert_gguf_message(result: Result<(), LoadError>, needle: &str, what: &str) {
    let error = refusal(result, what);
    assert!(
        matches!(&error, LoadError::SafeTensors(message) if message.contains(needle)),
        "{what}: expected a gguf refusal naming {needle:?}, got: {error}"
    );
}

#[test]
fn the_valid_gguf_loads() {
    load_gguf(valid_gguf()).expect("the fixture every corpus row corrupts must itself load");
}

/// Every proper prefix of a valid checkpoint is refused: the header is cut short, or a tensor's
/// range ends past the file. Never a panic, never a store.
#[test]
fn every_truncation_of_a_valid_gguf_is_refused() {
    let full = valid_gguf();
    for len in 0..full.len() {
        let prefix = full[..len].to_vec();
        let outcome = catch_unwind(AssertUnwindSafe(|| load_gguf(prefix)));
        match outcome {
            Err(_) => panic!("a GGUF cut at byte {len} of {} panicked", full.len()),
            Ok(Ok(())) => panic!("a GGUF cut at byte {len} of {} loaded", full.len()),
            Ok(Err(_)) => {}
        }
    }
}

/// Every byte of the header region, replaced by each of a handful of extreme values, parses or is
/// refused; it never panics or exhausts memory (a count or length field set to 0xFF.. is the case
/// that allocates before it checks).
#[test]
fn no_single_byte_corruption_of_a_gguf_header_panics() {
    let full = valid_gguf();
    let header_len = full.len() - VALID_GGUF_DATA_LEN;
    for position in 0..header_len {
        for value in [0x00u8, 0x01, 0x7F, 0x80, 0xFF] {
            let mut bytes = full.clone();
            bytes[position] = value;
            let outcome = catch_unwind(AssertUnwindSafe(|| load_gguf(bytes)));
            assert!(
                outcome.is_ok(),
                "header byte {position} set to {value:#04x} panicked the loader"
            );
        }
    }
}

/// Seeded random multi-byte damage anywhere in the file: still no panic.
#[test]
fn seeded_random_damage_to_a_gguf_never_panics() {
    let full = valid_gguf();
    let mut next = stream(0xC0FF_EE11);
    for round in 0..4000 {
        let mut bytes = full.clone();
        for _ in 0..1 + next() % 6 {
            let position = (next() % bytes.len() as u64) as usize;
            bytes[position] = next() as u8;
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| load_gguf(bytes)));
        assert!(outcome.is_ok(), "damage round {round} panicked the loader");
    }
}

#[test]
fn a_file_that_is_not_gguf_is_refused_by_its_magic() {
    let mut bytes = valid_gguf();
    bytes[0] = b'X';
    assert_gguf_message(load_gguf(bytes), "bad magic", "wrong magic");
}

#[test]
fn an_unknown_gguf_version_is_refused() {
    for version in [0u32, 1, 4, u32::MAX] {
        let mut raw = RawGguf::new(0, 0);
        raw.version = version;
        assert_gguf_message(
            load_gguf(raw.finish(&[])),
            &format!("unsupported GGUF version {version}"),
            "version",
        );
    }
}

/// A declared count the file cannot hold ends at the first missing byte; it must not size a
/// collection from the count.
#[test]
fn counts_of_u64_max_end_at_the_first_missing_byte() {
    assert_gguf_message(
        load_gguf(RawGguf::new(u64::MAX, 0).finish(&[])),
        "unexpected end of file",
        "tensor count u64::MAX",
    );
    assert_gguf_message(
        load_gguf(RawGguf::new(0, u64::MAX).finish(&[])),
        "unexpected end of file",
        "metadata count u64::MAX",
    );
}

#[test]
fn a_string_length_past_the_file_is_refused() {
    let bytes = RawGguf::new(0, 1).u64(u64::MAX).finish(&[]);
    assert_gguf_message(load_gguf(bytes), "", "key length u64::MAX");
    let bytes = RawGguf::new(1, 0).u64(1 << 40).finish(&[]);
    assert_gguf_message(load_gguf(bytes), "", "tensor name length 2^40");
}

#[test]
fn a_non_utf8_key_is_refused() {
    let bytes = RawGguf::new(0, 1)
        .u64(2)
        .raw(&[0xC3, 0x28])
        .u32(4)
        .u32(0)
        .finish(&[]);
    assert_gguf_message(load_gguf(bytes), "non-utf8 string", "key bytes C3 28");
}

#[test]
fn an_unknown_metadata_value_type_is_refused() {
    let bytes = RawGguf::new(0, 1).string("k").u32(99).u32(0).finish(&[]);
    assert_gguf_message(load_gguf(bytes), "unknown GGUF value type 99", "type 99");
}

#[test]
fn a_nested_metadata_array_is_refused() {
    let bytes = RawGguf::new(0, 1)
        .string("k")
        .u32(9)
        .u32(9)
        .u64(0)
        .finish(&[]);
    assert_gguf_message(load_gguf(bytes), "nested GGUF arrays", "array of arrays");
}

#[test]
fn a_tensor_rank_over_the_cap_is_refused_before_it_sizes_a_vector() {
    // MAX_TENSOR_DIMS is 8; the declared rank is u32::MAX and no dims follow.
    let bytes = RawGguf::new(1, 0).string("t").u32(u32::MAX).finish(&[]);
    assert_gguf_message(
        load_gguf(bytes),
        "tensor rank exceeds maximum",
        "rank u32::MAX",
    );
    let bytes = RawGguf::new(1, 0)
        .tensor("t", &[1; 9], GGML_F32, 0)
        .finish(&[]);
    assert_gguf_message(load_gguf(bytes), "tensor rank exceeds maximum", "rank 9");
}

#[test]
fn a_tensor_element_count_that_overflows_or_exceeds_the_cap_is_refused() {
    let bytes = RawGguf::new(1, 0)
        .tensor("t", &[u64::MAX, 2], GGML_F32, 0)
        .finish(&[]);
    assert_gguf_message(load_gguf(bytes), "overflows usize", "u64::MAX x 2");
    let bytes = RawGguf::new(1, 0)
        .tensor("t", &[(1 << 40) + 1], GGML_F32, 0)
        .finish(&[]);
    assert_gguf_message(load_gguf(bytes), "exceeds maximum", "2^40 + 1 elements");
}

#[test]
fn a_ggml_type_poot_cannot_read_is_refused_by_tensor_name() {
    for ggml_type in [4u32, 9, 24, 39 + 1, 9999, u32::MAX] {
        let bytes = RawGguf::new(1, 0)
            .tensor("odd.weight", &[32, 2], ggml_type, 0)
            .finish(&[0u8; 256]);
        let error = refusal(load_gguf(bytes), "unsupported type");
        assert!(
            matches!(
                &error,
                LoadError::UnsupportedGgufType { tensor, ggml_type: t }
                    if tensor == "odd.weight" && *t == ggml_type
            ),
            "ggml type {ggml_type}: {error}"
        );
    }
}

/// The range guard, both sides of the line: a tensor whose last byte is the file's last byte loads,
/// one byte further is `GgufTensorTruncated` naming the bytes needed and held.
#[test]
fn a_tensor_range_one_byte_past_the_file_is_refused() {
    let at_end = RawGguf::new(1, 0)
        .tensor("t", &[4], GGML_F32, 0)
        .finish(&[0u8; 16]);
    load_gguf(at_end.clone()).expect("a tensor ending at the last byte of the file loads");

    let mut one_short = at_end;
    one_short.pop();
    let error = refusal(load_gguf(one_short.clone()), "one byte short");
    assert!(
        matches!(
            error,
            LoadError::GgufTensorTruncated { need, have } if need == have + 1 && have == one_short.len()
        ),
        "one byte past the file: {error}"
    );
}

#[test]
fn a_tensor_offset_that_wraps_the_address_space_is_refused() {
    for offset in [u64::MAX, u64::MAX - 15, 1 << 63] {
        let bytes = RawGguf::new(1, 0)
            .tensor("t", &[4], GGML_F32, offset)
            .finish(&[0u8; 16]);
        let error = refusal(load_gguf(bytes), "offset near u64::MAX");
        assert!(
            matches!(
                &error,
                LoadError::GgufTensorTruncated { .. } | LoadError::SafeTensors(_)
            ),
            "offset {offset:#x}: {error}"
        );
    }
}

#[test]
fn a_rank_three_expert_count_the_file_cannot_hold_is_refused() {
    // ne = [256, 1, 2^30]: a billion experts of 144 bytes each, over a file of a few hundred bytes.
    let bytes = RawGguf::new(1, 0)
        .tensor("experts.weight", &[256, 1, 1 << 30], GGML_Q4_K, 0)
        .finish(&[0u8; 288]);
    let error = refusal(load_gguf(bytes), "a billion experts");
    assert!(
        matches!(error, LoadError::GgufTensorTruncated { .. }),
        "{error}"
    );
}

/// A quantized tensor has an owner shape only at rank 2 (a weight) or 3 (stacked experts); any other
/// rank is refused by name with the rank it declared.
#[test]
fn a_quantized_tensor_of_another_rank_is_refused() {
    for dims in [vec![32u64], vec![32, 1, 1, 1]] {
        let rank = dims.len();
        let bytes = RawGguf::new(1, 0)
            .tensor("q.weight", &dims, GGML_Q8_0, 0)
            .finish(&[0u8; 68]);
        let error = refusal(load_gguf(bytes), "a quantized tensor of the wrong rank");
        assert!(
            matches!(
                &error,
                LoadError::UnsupportedGgufTensorRank { tensor, rank: r }
                    if tensor == "q.weight" && *r == rank
            ),
            "rank {rank}: {error}"
        );
    }
}

/// No file at all is not a checkpoint: the split reader names the empty set.
#[test]
fn an_empty_shard_list_is_refused() {
    let error = GgufIndex::parse_shards(&[])
        .err()
        .expect("an empty list of shards");
    assert!(
        matches!(error, LoadError::GgufSplit(GgufSplitError::NoShards)),
        "{error}"
    );
}

/// A GGUF with one metadata array `key` of `n` elements of type `elem_ty` made of `body`, then a
/// u32 key `after`, so a parser that mis-steps over the array misreads what follows. No padding to the
/// data start: the file ends where `after` does unless `pad` asks for it.
fn array_then_marker(key: &str, elem_ty: u32, n: u64, body: &[u8], pad: bool) -> Vec<u8> {
    let raw = RawGguf::new(0, 2)
        .string(key)
        .u32(9)
        .u32(elem_ty)
        .u64(n)
        .raw(body)
        .string("after")
        .u32(4)
        .u32(7);
    if pad {
        raw.finish(&[])
    } else {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(raw.version.to_le_bytes());
        bytes.extend(raw.tensors.to_le_bytes());
        bytes.extend(raw.kvs.to_le_bytes());
        bytes.extend(raw.body);
        bytes
    }
}

/// An array no reader consumes is stepped over without being stored, and the parser lands exactly on
/// the next key whatever the element width: a stride off by one element misreads the marker.
#[test]
fn a_skipped_array_leaves_the_cursor_on_the_next_key() {
    const LONG: usize = 4097; // one past the longest array a layer table may store
    let strings: Vec<u8> = (0..LONG).flat_map(|_| 0u64.to_le_bytes()).collect();
    for (elem_ty, width) in [(0u32, 1usize), (2, 2), (4, 4), (6, 4), (10, 8), (12, 8)] {
        let bytes = array_then_marker(
            "some.table",
            elem_ty,
            LONG as u64,
            &vec![0u8; LONG * width],
            true,
        );
        let index = GgufIndex::parse(&StoredBytes::from(bytes))
            .unwrap_or_else(|e| panic!("element type {elem_ty}: {e}"));
        assert!(
            index.get("some.table").is_none(),
            "type {elem_ty} was stored"
        );
        assert_eq!(
            index.get("after"),
            Some(&GgufValue::U32(7)),
            "type {elem_ty}"
        );
    }
    let bytes = array_then_marker("some.table", 8, LONG as u64, &strings, true);
    let index = GgufIndex::parse(&StoredBytes::from(bytes)).expect("4097 empty strings");
    assert_eq!(index.get("after"), Some(&GgufValue::U32(7)));
}

/// A stored per-layer array keeps its elements at every width, and the boundary of "stored" is
/// exactly 4096 elements.
#[test]
fn a_layer_array_is_stored_up_to_exactly_4096_elements() {
    for (elem_ty, width) in [(2u32, 2usize), (4, 4), (10, 8)] {
        let bytes = array_then_marker("some.layer_table", elem_ty, 3, &vec![0u8; 3 * width], true);
        let index = GgufIndex::parse(&StoredBytes::from(bytes)).expect("a short array");
        assert_eq!(
            index
                .get("some.layer_table")
                .and_then(GgufValue::as_array)
                .map(<[_]>::len),
            Some(3),
            "type {elem_ty}"
        );
    }
    for (n, stored) in [(4096usize, true), (4097, false)] {
        let bytes = array_then_marker("some.layer_table", 0, n as u64, &vec![0u8; n], true);
        let index = GgufIndex::parse(&StoredBytes::from(bytes)).unwrap();
        assert_eq!(
            index.get("some.layer_table").is_some(),
            stored,
            "{n} elements"
        );
    }
}

/// A header that ends in one array of `n` declared u8 elements of which `present` bytes follow.
fn last_array(n: u64, present: usize) -> Vec<u8> {
    let raw = RawGguf::new(0, 1)
        .string("some.layer_table")
        .u32(9)
        .u32(0)
        .u64(n)
        .raw(&vec![1u8; present]);
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(raw.version.to_le_bytes());
    bytes.extend(raw.tensors.to_le_bytes());
    bytes.extend(raw.kvs.to_le_bytes());
    bytes.extend(raw.body);
    bytes
}

/// The array length guard, both sides of the line: elements that are exactly the bytes left in the
/// file are not "longer than the file" (the refusal that follows is the missing data section), and one
/// element more than the bytes left is.
#[test]
fn an_array_is_refused_exactly_when_it_outruns_the_file() {
    let error = GgufIndex::parse(&StoredBytes::from(last_array(3, 3)))
        .err()
        .expect("no data section follows");
    assert!(
        error.to_string().contains("data start past end of file"),
        "an array that fits: {error}"
    );
    let error = GgufIndex::parse(&StoredBytes::from(last_array(4, 3)))
        .err()
        .expect("one element more than the file holds");
    assert!(
        error
            .to_string()
            .contains("array length exceeds the bytes remaining"),
        "an array one element too long: {error}"
    );
}

/// A tokenizer table is stored up to exactly the cap (4 Mi elements) and refused one past it.
#[test]
fn a_tokenizer_table_of_exactly_the_cap_is_stored() {
    const CAP: usize = 1 << 22;
    let raw = RawGguf::new(0, 1)
        .string("tokenizer.ggml.scores")
        .u32(9)
        .u32(6)
        .u64(CAP as u64)
        .raw(&vec![0u8; 4 * CAP]);
    let index = GgufIndex::parse(&StoredBytes::from(raw.finish(&[]))).expect("the cap itself");
    assert_eq!(
        index
            .get("tokenizer.ggml.scores")
            .and_then(GgufValue::as_array)
            .map(<[_]>::len),
        Some(CAP)
    );
}

/// The rank cap is a cap, not an off-by-one: eight dimensions load, nine are refused.
#[test]
fn a_tensor_of_the_maximum_rank_loads() {
    let bytes = RawGguf::new(1, 0)
        .tensor("t", &[1; 8], GGML_F32, 0)
        .finish(&[0u8; 4]);
    load_gguf(bytes).expect("rank 8 is the largest accepted");
}

/// The element cap is inclusive at the header: exactly 2^40 elements parse, and the refusal is then the
/// file being far too short for them, not the cap.
#[test]
fn a_tensor_of_exactly_the_element_cap_is_refused_for_its_bytes_not_its_count() {
    let bytes = RawGguf::new(1, 0)
        .tensor("t", &[1 << 40], GGML_F32, 0)
        .finish(&[0u8; 4]);
    let error = refusal(load_gguf(bytes), "2^40 elements in a tiny file");
    assert!(
        matches!(error, LoadError::GgufTensorTruncated { .. }),
        "{error}"
    );
}

/// Two files that declare no split keys are not a split set: the first must declare the count.
#[test]
fn two_files_without_split_keys_are_refused_with_the_missing_count() {
    let plain = || StoredBytes::from(RawGguf::new(0, 0).finish(&[]));
    let error = GgufIndex::parse_shards(&[plain(), plain()])
        .err()
        .expect("two plain files");
    assert!(
        matches!(
            error,
            LoadError::GgufSplit(GgufSplitError::Count {
                declared: None,
                given: 2
            })
        ),
        "{error}"
    );
}

/// The data start is inclusive: a checkpoint with no tensors whose header ends exactly at the end of
/// the file (the data section is empty) is a valid file, one byte shorter is not.
#[test]
fn a_data_start_exactly_at_the_end_of_the_file_loads() {
    let exact = RawGguf::new(0, 0).finish(&[]);
    assert_eq!(
        exact.len() % 32,
        0,
        "the header is padded to its data start"
    );
    load_gguf(exact).expect("an empty data section is a data section");
}

/// The smallest safetensors file is the 8-byte length prefix: with a zero-length header it is refused
/// as JSON that is not there, not as a file too small to hold a prefix.
#[test]
fn an_eight_byte_safetensors_file_has_a_prefix_and_no_header() {
    let error = refusal(
        load_safetensors(vec![0u64.to_le_bytes().to_vec()]),
        "a zero-length header",
    );
    assert!(matches!(error, LoadError::Json(_)), "{error}");
}

#[test]
fn a_data_start_past_the_file_is_refused() {
    // A header that ends in the last byte, then pads to 32: the data section starts past the end.
    let mut bytes = RawGguf::new(0, 0).finish(&[]);
    bytes.truncate(bytes.len() - 1);
    assert_gguf_message(
        load_gguf(bytes),
        "data start past end of file",
        "no padding",
    );
}

// ---------------------------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------------------------

/// A safetensors file: the 8-byte little-endian header length, `header`, then `data`.
fn safetensors(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header.as_bytes());
    out.extend(data);
    out
}

fn valid_safetensors() -> Vec<u8> {
    safetensors(
        r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"BF16","shape":[2,2],"data_offsets":[8,16]}}"#,
        &[0u8; 16],
    )
}

fn load_safetensors(files: Vec<Vec<u8>>) -> Result<(), LoadError> {
    let files: Vec<(String, StoredBytes)> = files
        .into_iter()
        .enumerate()
        .map(|(i, bytes)| (format!("shard-{i}.safetensors"), StoredBytes::from(bytes)))
        .collect();
    load_mapped_weight_store(&files, default_workers()).map(drop)
}

fn one(header: &str, data: &[u8]) -> Result<(), LoadError> {
    load_safetensors(vec![safetensors(header, data)])
}

fn assert_safetensors_message(result: Result<(), LoadError>, needle: &str, what: &str) {
    let error = refusal(result, what);
    assert!(
        matches!(&error, LoadError::SafeTensors(message) if message.contains(needle)),
        "{what}: expected a safetensors refusal naming {needle:?}, got: {error}"
    );
}

#[test]
fn the_valid_safetensors_loads() {
    load_safetensors(vec![valid_safetensors()]).expect("the fixture every corpus row corrupts");
}

#[test]
fn every_truncation_of_a_valid_safetensors_is_refused() {
    let full = valid_safetensors();
    for len in 0..full.len() {
        let prefix = full[..len].to_vec();
        let outcome = catch_unwind(AssertUnwindSafe(|| load_safetensors(vec![prefix])));
        match outcome {
            Err(_) => panic!("a safetensors file cut at byte {len} panicked"),
            Ok(Ok(())) => panic!(
                "a safetensors file cut at byte {len} of {} loaded",
                full.len()
            ),
            Ok(Err(_)) => {}
        }
    }
}

#[test]
fn no_single_byte_corruption_of_a_safetensors_header_panics() {
    let full = valid_safetensors();
    let header_end = 8 + u64::from_le_bytes(full[..8].try_into().unwrap()) as usize;
    for position in 0..header_end {
        for value in [0x00u8, 0x01, 0x22, 0x7B, 0x7F, 0x80, 0xFF] {
            let mut bytes = full.clone();
            bytes[position] = value;
            let outcome = catch_unwind(AssertUnwindSafe(|| load_safetensors(vec![bytes])));
            assert!(
                outcome.is_ok(),
                "safetensors byte {position} set to {value:#04x} panicked the loader"
            );
        }
    }
}

#[test]
fn seeded_random_damage_to_a_safetensors_file_never_panics() {
    let full = valid_safetensors();
    let mut next = stream(0x5AFE_7E45);
    for round in 0..4000 {
        let mut bytes = full.clone();
        for _ in 0..1 + next() % 6 {
            let position = (next() % bytes.len() as u64) as usize;
            bytes[position] = next() as u8;
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| load_safetensors(vec![bytes])));
        assert!(outcome.is_ok(), "damage round {round} panicked the loader");
    }
}

#[test]
fn a_header_length_past_the_file_is_refused() {
    for length in [u64::MAX, 1 << 63, 1 << 40] {
        let mut bytes = valid_safetensors();
        bytes[..8].copy_from_slice(&length.to_le_bytes());
        assert_safetensors_message(
            load_safetensors(vec![bytes]),
            "header length out of range",
            "header length past the file",
        );
    }
    // One byte past: the declared header is one byte longer than the file holds.
    let mut bytes = safetensors("{}", &[]);
    bytes[..8].copy_from_slice(&3u64.to_le_bytes());
    assert_safetensors_message(
        load_safetensors(vec![bytes]),
        "header length out of range",
        "header one byte past the file",
    );
}

#[test]
fn a_file_too_small_for_the_length_prefix_is_refused() {
    assert_safetensors_message(
        load_safetensors(vec![vec![0u8; 7]]),
        "file too small",
        "7 bytes",
    );
}

#[test]
fn a_header_that_is_not_json_is_a_json_error() {
    let error = refusal(one("not json", &[]), "not json");
    assert!(matches!(error, LoadError::Json(_)), "{error}");
}

#[test]
fn a_tensor_entry_missing_a_field_is_refused_by_name() {
    for (header, field) in [
        (r#"{"t":{"shape":[1],"data_offsets":[0,4]}}"#, "dtype"),
        (r#"{"t":{"dtype":"F32","data_offsets":[0,4]}}"#, "shape"),
        (r#"{"t":{"dtype":"F32","shape":[1]}}"#, "data_offsets"),
    ] {
        assert_safetensors_message(one(header, &[0u8; 4]), field, field);
    }
}

#[test]
fn an_unknown_dtype_is_refused() {
    let error = refusal(
        one(
            r#"{"t":{"dtype":"F7","shape":[1],"data_offsets":[0,4]}}"#,
            &[0u8; 4],
        ),
        "dtype F7",
    );
    assert!(
        matches!(&error, LoadError::UnknownCheckpointDtype { name } if name == "F7"),
        "{error}"
    );
}

#[test]
fn malformed_shapes_and_offsets_are_refused() {
    let data = [0u8; 8];
    let cases = [
        (
            r#"{"t":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
            "shape[0] is not a non-negative integer",
        ),
        (
            r#"{"t":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}}"#,
            "shape[0] is not a non-negative integer",
        ),
        (
            r#"{"t":{"dtype":"F32","shape":[18446744073709551616],"data_offsets":[0,4]}}"#,
            "shape[0] is not a non-negative integer",
        ),
        (
            r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#,
            "data_offsets must be [start, end]",
        ),
        (
            r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4,8]}}"#,
            "data_offsets must be [start, end]",
        ),
        (
            r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":["0",4]}}"#,
            "data_offsets[0] not a u64",
        ),
        (
            r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,-4]}}"#,
            "data_offsets[1] not a u64",
        ),
        (
            r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
            "out of range",
        ),
        (
            r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,18446744073709551615]}}"#,
            "out of range",
        ),
        (
            r#"{"t":{"dtype":"F32","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,4]}}"#,
            "shape element count overflows usize",
        ),
        (
            r#"{"t":{"dtype":"F64","shape":[2305843009213693952],"data_offsets":[0,0]}}"#,
            "overflows usize",
        ),
    ];
    for (header, needle) in cases {
        assert_safetensors_message(one(header, &data), needle, header);
    }
}

/// The range guard, both sides: a tensor ending at the last data byte loads, one byte further is
/// `out of range` naming the span and the data length.
#[test]
fn a_tensor_range_one_byte_past_the_data_is_refused() {
    one(
        r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        &[0u8; 8],
    )
    .expect("a tensor ending at the last byte loads");
    assert_safetensors_message(
        one(
            r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &[0u8; 7],
        ),
        "data_offsets [0, 8] out of range for 7 data bytes",
        "one byte past",
    );
}

#[test]
fn a_span_that_disagrees_with_the_shape_is_refused() {
    for (offsets, shape) in [("[0,7]", "[2]"), ("[0,9]", "[2]"), ("[0,8]", "[3]")] {
        let header =
            format!(r#"{{"t":{{"dtype":"F32","shape":{shape},"data_offsets":{offsets}}}}}"#);
        assert_safetensors_message(one(&header, &[0u8; 12]), "expected", &header);
    }
}

#[test]
fn a_tensor_name_repeated_in_one_header_is_refused() {
    let header = r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"t":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    let error = refusal(one(header, &[0u8; 8]), "twice in one header");
    assert!(
        matches!(&error, LoadError::DuplicateTensorName { name, scope }
            if name == "t" && scope == "within one header"),
        "{error}"
    );
}

#[test]
fn two_shards_declaring_one_name_are_refused() {
    let shard = |header: &str| safetensors(header, &[0u8; 4]);
    let a = shard(r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#);
    let b = shard(r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#);
    let error = refusal(load_safetensors(vec![a, b]), "w in both shards");
    assert!(
        matches!(&error, LoadError::DuplicateTensorName { name, scope }
            if name == "w" && scope == "across model shards"),
        "{error}"
    );
}
